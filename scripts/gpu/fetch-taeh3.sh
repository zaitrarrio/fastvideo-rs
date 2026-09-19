#!/usr/bin/env bash
# Download and verify taeh3.safetensors into $1. Run by remote.sh fetch-taeh3.
set -euo pipefail
dest="$1"
url="https://github.com/madebyollin/taehv/raw/main/safetensors/taeh3.safetensors"
curl -fsSL --retry 5 --retry-delay 3 -o "$dest/taeh3.safetensors.part" "$url"
python3 - "$dest/taeh3.safetensors.part" <<'CHECK'
import json, os, struct, sys
p = sys.argv[1]
with open(p, "rb") as fh:
    (hlen,) = struct.unpack("<Q", fh.read(8))
    header = json.loads(fh.read(hlen))
end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
assert os.path.getsize(p) == 8 + hlen + end, "truncated taeh3 weights"
print("taeh3 weights verified:", len(header), "tensors")
CHECK
mv "$dest/taeh3.safetensors.part" "$dest/taeh3.safetensors"
touch "$dest/.complete"
