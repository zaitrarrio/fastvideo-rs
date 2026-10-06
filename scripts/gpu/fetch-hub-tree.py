#!/usr/bin/env python3
"""Add one Hub weight tree to a network volume, add-only (fetch-hub-tree.sh).

Runs on a CPU pod with the volume at /workspace. Downloads REPO@REVISION
(the manifest globs plus model_index.json) with `local_dir` (plain files)
into weights/.<DEST>.partial-<stamp> (for a nested DEST such as
auxiliary/upscalers/seedvr2: <parent>/.<name>.partial-<stamp>, the parent
created if missing), checks every file against the Hub
listing at that revision (size; LFS files by SHA-256, the others by their
git blob SHA-1), writes .complete and sha256.txt, and only then renames the
temp folder to weights/<DEST>. Refuses to start if weights/<DEST> exists.
With EXPECT_SHA256 (the sha256.txt of another volume's copy, base64), every
file must also match that list (the second volume of a sync).
FETCH_MIN_FREE_GB (default 50): refuse to start unless the volume's free
space minus the tree's bytes stays at or above this many GB; the free space
is logged before and after. Runpod network volumes report the whole cluster
to statvfs (hundreds of PB free), so with FETCH_VOLUME_GB (the volume's size,
passed by fetch-hub-tree.sh) free = size - `du -sb` of the mount instead.

FETCH_ADD_INTO=1 instead adds the glob files that weights/<DEST> lacks to that
existing tree, add-only: they are downloaded into <parent>/.<name>.add-partial-<stamp>,
checked against the Hub the same way, fsync'd, and only then each one is
renamed into weights/<DEST>/ (refusing any name that already exists there).
Nothing already in the tree is read, changed or removed; .complete is left as it is.

Progress goes to /srv/log.txt; /srv/DONE holds 0 (ok) or 1; /srv/sha256.txt
lists "<sha256>  <relative path>" for every file of the tree.
"""
import base64
import fnmatch
import hashlib
import os
import shutil
import time
import traceback
from pathlib import Path

REPO = os.environ["FETCH_REPO"]
REV = os.environ["FETCH_REVISION"]
DEST = os.environ["FETCH_DEST"]
GLOBS = os.environ["FETCH_GLOBS"].split() + ["model_index.json"]
WEIGHTS = Path(os.environ.get("FETCH_WEIGHTS", "/workspace/weights"))
SRV = Path(os.environ.get("FETCH_SRV", "/srv"))
LOG = SRV / "log.txt"


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with LOG.open("a") as f:
        f.write(s + "\n")


def free_bytes(path):
    cap = os.environ.get("FETCH_VOLUME_GB")
    if cap:
        import subprocess
        mount = os.environ.get("FETCH_MOUNT", "/workspace")
        out = subprocess.run(["du", "-sb", mount], capture_output=True, text=True).stdout
        return int(float(cap) * 1e9) - int(out.split()[0])
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def file_hash(path, algo, prefix=b""):
    h = hashlib.new(algo)
    h.update(prefix)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    final = WEIGHTS / DEST
    add_into = os.environ.get("FETCH_ADD_INTO") == "1"
    if add_into and not final.is_dir():
        raise SystemExit(f"FETCH_ADD_INTO: {final} is not an existing tree")
    if final.exists() and not add_into:
        raise SystemExit(f"{final} exists: add-only fetch refuses to touch it")
    final.parent.mkdir(parents=True, exist_ok=True)
    # This script's own unfinished temp folders for DEST (a failed earlier
    # run); never anything else.
    for old in final.parent.glob(f".{final.name}.{'add-' if add_into else ''}partial-*"):
        shutil.rmtree(old, ignore_errors=True)
        log(f"removed own partial {old}")
    # Gated repos: the volume's token when the pod env has none.
    tok = Path("/workspace/hf/token")
    if not os.environ.get("HF_TOKEN") and tok.is_file():
        os.environ["HF_TOKEN"] = tok.read_text().strip()
        log("HF token: /workspace/hf/token")
    from huggingface_hub import HfApi, snapshot_download

    tmp = final.parent / f".{final.name}.{'add-' if add_into else ''}partial-{time.strftime('%Y%m%d%H%M%S')}"
    info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
    want = [s for s in info.siblings if any(fnmatch.fnmatch(s.rfilename, g) for g in GLOBS)]
    if add_into:
        have = [s.rfilename for s in want if (final / s.rfilename).exists()]
        want = [s for s in want if not (final / s.rfilename).exists()]
        log(f"FETCH_ADD_INTO: {final} already has {have}; adding {[s.rfilename for s in want]}")
        if not want:
            return
    total = sum(s.size or 0 for s in want)
    log(f"{REPO}@{REV}: {len(want)} files, {total} bytes -> {tmp}")
    free = free_bytes(WEIGHTS)
    min_free = float(os.environ.get("FETCH_MIN_FREE_GB", "50")) * 1e9
    log(f"volume free {free} bytes ({free / 1e9:.1f} GB); after this tree {(free - total) / 1e9:.1f} GB")
    if free - total < min_free:
        raise SystemExit(f"not enough space: {free} free - {total} < {min_free:.0f} floor; nothing written")
    t0 = time.time()
    pats = [s.rfilename for s in want] if add_into else GLOBS
    snapshot_download(REPO, revision=REV, local_dir=str(tmp), allow_patterns=pats, max_workers=int(os.environ.get("FETCH_WORKERS", "4")))
    log(f"downloaded in {time.time() - t0:.0f} s; verifying")
    shutil.rmtree(tmp / ".cache", ignore_errors=True)
    expect = {}
    if os.environ.get("EXPECT_SHA256"):
        for line in base64.b64decode(os.environ["EXPECT_SHA256"]).decode().splitlines():
            h, p = line.split("  ", 1)
            expect[p] = h
    lines, bad = [], []
    on_disk = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
    if on_disk != sorted(s.rfilename for s in want):
        bad.append(f"file list differs: disk {on_disk} vs hub {sorted(s.rfilename for s in want)}")
    for s in want:
        p = tmp / s.rfilename
        if not p.is_file() or p.stat().st_size != s.size:
            bad.append(f"{s.rfilename}: size {p.stat().st_size if p.is_file() else None} != {s.size}")
            continue
        sha = file_hash(p, "sha256")
        lfs_sha = getattr(s.lfs, "sha256", None) or (s.lfs.get("sha256") if isinstance(s.lfs, dict) else None)
        if s.lfs is not None:
            ok = sha == lfs_sha
        else:
            ok = file_hash(p, "sha1", f"blob {s.size}\0".encode()) == s.blob_id
        if not ok:
            bad.append(f"{s.rfilename}: hash differs from the Hub")
        if expect and expect.get(s.rfilename) != sha:
            bad.append(f"{s.rfilename}: sha256 differs from the other volume's copy")
        lines.append(f"{sha}  {s.rfilename}")
        log(f"ok {s.rfilename} {s.size} {sha[:16]}{' lfs' if s.lfs else ''}")
    if expect and set(expect) != {s.rfilename for s in want}:
        bad.append("file list differs from the other volume's copy")
    (SRV / "sha256.txt").write_text("\n".join(lines) + "\n")
    if bad:
        for b in bad:
            log("BAD", b)
        raise SystemExit(f"{len(bad)} problems; {tmp} left in place, not renamed")
    if not add_into:
        (tmp / ".complete").write_text(f"{time.time() - t0:.0f}\n")
    for fd in [os.open(tmp / l.split("  ", 1)[1], os.O_RDONLY) for l in lines]:
        os.fsync(fd)
        os.close(fd)
    if add_into:
        for s in want:
            dst = final / s.rfilename
            if dst.exists():
                raise SystemExit(f"{dst} appeared during the fetch; {tmp} left in place")
        for s in want:
            dst = final / s.rfilename
            dst.parent.mkdir(parents=True, exist_ok=True)
            (tmp / s.rfilename).rename(dst)
            log(f"added {dst}")
        shutil.rmtree(tmp)  # now empty of files: only this run's own folder
        log(f"added {len(want)} files, {total} bytes in {time.time() - t0:.0f} s; volume free {free_bytes(WEIGHTS)} bytes")
        return
    if final.exists():
        raise SystemExit(f"{final} appeared during the fetch; {tmp} left in place")
    tmp.rename(final)
    log(f"renamed to {final}; {total} bytes in {time.time() - t0:.0f} s; volume free {free_bytes(WEIGHTS)} bytes")


if __name__ == "__main__":
    SRV.mkdir(parents=True, exist_ok=True)
    try:
        main()
        (SRV / "DONE").write_text("0")
    except BaseException:
        log(traceback.format_exc())
        (SRV / "DONE").write_text("1")
