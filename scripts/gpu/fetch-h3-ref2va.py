#!/usr/bin/env python3
"""Add-only fetch of the H3 Ref2VA tree onto a mounted weight volume.

Runs on a CPU pod (fetch-h3-ref2va.sh). Writes nothing outside
``/workspace/weights/.h3-ref2va.partial-<stamp>`` until every file is verified;
then renames that folder to ``/workspace/weights/h3-ref2va``. It refuses to
start when ``h3-ref2va`` already exists, and never deletes anything on the
volume (the temp folder is left in place on failure, for inspection).

Tree (plain files, ``local_dir`` layout):

    transformer_ref/*                  MiniMaxAI/MiniMax-H3 @ REV_H3 (66.28 GB)
    processor/*, model_index.json, LICENSE, README.md   same revision (small)
    Minimax-h3-Turbo/minimax_h3_ref2v_turbo_*_bf16.safetensors
                                       lightx2v/Minimax-h3-Turbo @ REV_TURBO

Verification: every LFS file's SHA-256 against the Hub's ``lfs.sha256`` at the
pinned revision, every small file's size against the Hub's size; the list is
written to ``sha256.txt`` in the tree. Progress goes to /srv/log.txt; /srv/DONE
holds the exit status.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REV_H3 = "42ed227ee7df40d41602854ae760620d6eb651fe"
REV_TURBO = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
ROOT = Path(os.environ.get("FETCH_WEIGHTS", "/workspace/weights"))
DEST = ROOT / "h3-ref2va"
SRV = Path("/srv")
TURBO_FILES = [
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(SRV / "log.txt", "a") as f:
        f.write(line + "\n")


def plan() -> list[tuple[str, str, str, dict]]:
    """(repo, revision, repo path, local path, sibling info)."""
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    info = api.model_info("MiniMaxAI/MiniMax-H3", revision=REV_H3, files_metadata=True)
    for s in info.siblings:
        r = s.rfilename
        if r.startswith(("transformer_ref/", "processor/")) or r in ("model_index.json", "LICENSE", "README.md"):
            out.append(("MiniMaxAI/MiniMax-H3", REV_H3, r, r, s))
    info = api.model_info("lightx2v/Minimax-h3-Turbo", revision=REV_TURBO, files_metadata=True)
    for s in info.siblings:
        if s.rfilename in TURBO_FILES or s.rfilename == "README.md":
            local = "Minimax-h3-Turbo/" + s.rfilename
            out.append(("lightx2v/Minimax-h3-Turbo", REV_TURBO, s.rfilename, local, s))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import hf_hub_download

    if DEST.exists():
        log(f"REFUSE: {DEST} exists (add-only)")
        return 3
    files = plan()
    total = sum(s.size or 0 for *_, s in files)
    log(f"plan: {len(files)} files, {total} bytes ({total / 1e9:.2f} GB) -> {DEST}")
    for repo, rev, path, local, s in files:
        log(f"  {repo}@{rev[:7]} {path} -> {local} {s.size}")
    free = shutil.disk_usage(ROOT).free
    log(f"volume free {free / 1e9:.1f} GB")
    if free < total * 1.1:
        log("REFUSE: not enough free space")
        return 4
    tmp = ROOT / f".h3-ref2va.partial-{time.strftime('%m%d%H%M%S', time.gmtime())}"
    tmp.mkdir(parents=False, exist_ok=False)
    stage = tmp / ".stage"
    t0 = time.time()
    rows = []
    for repo, rev, path, local, s in files:
        t = time.time()
        # Download into a per-repo staging dir, then move to the tree path.
        got = Path(hf_hub_download(repo, path, revision=rev, local_dir=stage / repo.replace("/", "__")))
        dst = tmp / local
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, dst)
        size = dst.stat().st_size
        if s.size is not None and size != s.size:
            log(f"FAIL size {local}: {size} != hub {s.size}")
            return 5
        want = s.lfs.sha256 if s.lfs else None
        got_sha = sha256(dst)
        if want and got_sha != want:
            log(f"FAIL sha256 {local}: {got_sha} != hub {want}")
            return 6
        rows.append((local, size, got_sha, "lfs-ok" if want else "size-ok"))
        log(f"ok {local} {size} {got_sha[:16]} {'lfs' if want else 'small'} {time.time() - t:.0f}s")
    shutil.rmtree(stage)  # only this fetch's own staging dir inside the temp folder
    # Re-read every file from the volume after an fsync, as the sync did.
    os.sync()
    for local, size, sha, _ in rows:
        again = sha256(tmp / local)
        if again != sha:
            log(f"FAIL re-read {local}: {again} != {sha}")
            return 7
    (tmp / "sha256.txt").write_text("".join(f"{sha}  {size}  {local}  {how}\n" for local, size, sha, how in rows))
    (tmp / "SOURCE.json").write_text(json.dumps({
        "MiniMaxAI/MiniMax-H3": REV_H3, "lightx2v/Minimax-h3-Turbo": REV_TURBO,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bytes": total,
    }, indent=1))
    (tmp / ".complete").write_text(f"{time.time() - t0:.0f}\n")
    if DEST.exists():
        log(f"REFUSE rename: {DEST} appeared meanwhile; tree left at {tmp}")
        return 8
    os.rename(tmp, DEST)
    log(f"DONE {DEST}: {len(rows)} files, {total} bytes, {time.time() - t0:.0f}s")
    (SRV / "sha256.txt").write_text((DEST / "sha256.txt").read_text())
    # Context for the report: the h3-base snapshot revision on this volume.
    snaps = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "h3-base").glob("**/snapshots/*"))
    log(f"h3-base snapshots: {snaps}")
    return 0


if __name__ == "__main__":
    SRV.mkdir(exist_ok=True)
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        log(traceback.format_exc())
        rc = 1
    (SRV / "DONE").write_text(str(rc))
    sys.exit(rc)
