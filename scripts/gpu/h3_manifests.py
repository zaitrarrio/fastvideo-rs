#!/usr/bin/env python3
"""Key manifests of the FastH3 checkpoint, without downloading a weight.

A safetensors file starts with a little-endian u64 header length and a JSON
header; two HTTP range requests per shard fetch it. The output is one compact
`{key: [dtype, shape]}` JSON per component under
crates/fastvideo-cudarc/src/h3/manifests/, with the repo and the resolved
revision recorded, which `h3/manifest_tests.rs` checks every loader against.

Only the standard library is used. Text-encoder manifests keep the language
model keys of the shards the port reads (1..11) and drop the vision tower.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import urllib.request

REPO = "FastVideo/FastVideo-FastH3-8-Step-V2"


def get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url)
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def header(repo: str, rev: str, path: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    (n,) = struct.unpack("<Q", get(url, (0, 7)))
    return json.loads(get(url, (8, 8 + n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", default="crates/fastvideo-cudarc/src/h3/manifests")
    args = ap.parse_args()

    info = json.loads(get(f"https://huggingface.co/api/models/{args.repo}"))
    rev = info["sha"]
    files = sorted(s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(".safetensors"))

    def shard_no(path: str) -> int:
        name = path.rsplit("/", 1)[-1]
        return int(name.split("-")[1]) if "-of-" in name else 1

    components = {
        "transformer": (lambda p: p.startswith("transformer/"), lambda k: True),
        "vae": (lambda p: p.startswith("vae/"), lambda k: True),
        "audio_vae": (lambda p: p.startswith("audio_vae/"), lambda k: True),
        # The port maps shards 1..11 only and never reads the vision tower.
        "text_encoder": (
            lambda p: p.startswith("text_encoder/") and shard_no(p) <= 11,
            lambda k: k.startswith("model.language_model."),
        ),
    }
    os.makedirs(args.out, exist_ok=True)
    for name, (want_file, want_key) in components.items():
        chosen = [f for f in files if want_file(f)]
        tensors = {}
        for f in chosen:
            for key, meta in header(args.repo, rev, f).items():
                if key != "__metadata__" and want_key(key):
                    assert key not in tensors, f"{key} appears in two shards"
                    tensors[key] = [meta["dtype"], meta["shape"]]
        doc = {"repo": args.repo, "revision": rev, "files": chosen, "tensors": dict(sorted(tensors.items()))}
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        print(f"{path}: {len(tensors)} tensors from {len(chosen)} file(s) at {rev[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
