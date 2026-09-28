#!/usr/bin/env python3
"""Pod side of fetch-ltx-iclora.sh: one LTX-2.5 IC-LoRA onto a network volume, add-only.

ROLE=hub   download the pinned Hub revision (HF token from HF_TOKEN or
           /workspace/hf/token), check every file's SHA-256 against the Hub's
           LFS oid (README: recorded), then serve the finished folder
           read-only under a random path for the other volume.
ROLE=copy  pull the same files from SRC_URL (the proxy root of a ROLE=hub pod),
           check SHA-256 against its sha256.json (and the pinned LFS oid).

Both write into <dest>.partial-<stamp>, fsync, re-read every file from the
volume and compare its SHA-256, and only then rename the folder to <dest>.
Nothing else on the volume is touched; an existing <dest> stops the run.
Status in /srv/log.txt, /srv/sha256.json and /srv/DONE (json).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import secrets
import sys
import time
import urllib.error
import urllib.request

REPO = os.environ["IC_REPO"]
REV = os.environ["IC_REV"]
DEST = pathlib.Path(os.environ["IC_DEST"])
# name -> (sha256 or "", size)
FILES = json.loads(os.environ["IC_FILES"])
ROLE = os.environ.get("ROLE", "hub")
SRV = pathlib.Path("/srv")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(SRV / "log.txt", "a") as f:
        f.write(line + "\n")


def done(ok: bool, **kv) -> None:
    (SRV / "DONE").write_text(json.dumps({"ok": ok, **kv}, indent=1))
    log(f"DONE ok={ok} {kv}")


def sha_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def fetch(url: str, out: pathlib.Path, headers: dict) -> str:
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers=headers)
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
        f.flush()
        os.fsync(f.fileno())
    log(f"fetched {out.name}: {n} bytes")
    return h.hexdigest()


def probe() -> int:
    """ROLE=probe: remove this script's stale partials; report Hub access per file."""
    drop_partials()
    tok = os.environ.get("HF_TOKEN", "")
    tp = pathlib.Path("/workspace/hf/token")
    if not tok and tp.exists():
        tok = tp.read_text().strip()
    status = {"token": bool(tok), "dest_exists": DEST.exists()}
    for name in FILES:
        req = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REV}/{name}", method="HEAD",
                                     headers={"Authorization": f"Bearer {tok}"} if tok else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status[name] = r.status
        except urllib.error.HTTPError as e:
            status[name] = e.code
    done(True, **status)
    return 0


def main() -> int:
    if ROLE == "probe":
        return probe()
    if DEST.exists():
        done(False, error=f"{DEST} exists (add-only)")
        return 1
    drop_partials()  # an earlier failed run of this script
    tmp = DEST.parent / f".{DEST.name}.partial-{time.strftime('%Y%m%d%H%M%S')}"
    tmp.mkdir(parents=True)
    headers = {}
    if ROLE == "hub":
        tok = os.environ.get("HF_TOKEN", "")
        tp = pathlib.Path("/workspace/hf/token")
        if not tok and tp.exists():
            tok = tp.read_text().strip()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        base = f"https://huggingface.co/{REPO}/resolve/{REV}"
        expect = {k: v[0] for k, v in FILES.items()}
    else:
        root = os.environ["SRC_URL"].rstrip("/")
        with urllib.request.urlopen(root + "/sha256.json", timeout=60) as r:
            src = json.load(r)
        with urllib.request.urlopen(root + "/PUB", timeout=60) as r:
            base = root + "/" + r.read().decode().strip()
        expect = {}
        for k, (pinned, _) in FILES.items():
            if pinned and src.get(k) != pinned:
                done(False, error=f"source sha of {k} {src.get(k)} != pinned {pinned}")
                return 1
            expect[k] = src[k]
    t0 = time.time()
    got = {}
    for name, (_, size) in FILES.items():
        got[name] = fetch(f"{base}/{name}", tmp / name, headers)
        if (tmp / name).stat().st_size != size:
            done(False, error=f"{name}: {(tmp / name).stat().st_size} bytes, want {size}")
            return 1
        if expect.get(name) and got[name] != expect[name]:
            done(False, error=f"{name}: sha {got[name]} != {expect[name]}")
            return 1
    # Re-read from the volume (not the page cache's word for it: a fresh open).
    os.sync()
    for name in FILES:
        again = sha_file(tmp / name)
        if again != got[name]:
            done(False, error=f"{name}: re-read sha {again} != {got[name]}")
            return 1
    if DEST.exists():
        done(False, error=f"{DEST} appeared during the fetch")
        return 1
    tmp.rename(DEST)
    (SRV / "sha256.json").write_text(json.dumps(got, indent=1))
    info = {"dest": str(DEST), "seconds": round(time.time() - t0, 1), "sha256": got,
            "bytes": {k: (DEST / k).stat().st_size for k in FILES}}
    if ROLE == "hub":
        pub = "pub-" + secrets.token_hex(16)
        (SRV / pub).symlink_to(DEST)
        (SRV / "PUB").write_text(pub)
        info["pub"] = pub
    done(True, **info)
    return 0


def drop_partials() -> None:
    """Remove this script's own unfinished `<dest>.partial-*` folders (never <dest>)."""
    import shutil

    for p in DEST.parent.glob(f".{DEST.name}.partial-*"):
        shutil.rmtree(p, ignore_errors=True)
        log(f"removed own partial {p}")


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:  # noqa: BLE001
        done(False, error=f"{type(e).__name__}: {e}")
        rc = 1
    if rc:
        drop_partials()
    sys.exit(rc)
