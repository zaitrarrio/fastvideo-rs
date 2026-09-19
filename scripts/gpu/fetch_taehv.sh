#!/usr/bin/env bash
# Download and verify taew2_1.safetensors into $1. Run by remote.sh fetch-taehv.
set -euo pipefail
dest="$1"
url="https://github.com/madebyollin/taehv/raw/main/safetensors/taew2_1.safetensors"
curl -fsSL --retry 5 --retry-delay 3 -o "$dest/taew2_1.safetensors.part" "$url"
python3 - "$dest/taew2_1.safetensors.part" <<'CHECK'
import json, os, struct, sys
p = sys.argv[1]
with open(p, "rb") as fh:
    (hlen,) = struct.unpack("<Q", fh.read(8))
    header = json.loads(fh.read(hlen))
end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
assert os.path.getsize(p) == 8 + hlen + end, "truncated taehv weights"
print("taehv weights verified:", len(header), "tensors")
CHECK
mv "$dest/taew2_1.safetensors.part" "$dest/taew2_1.safetensors"
touch "$dest/.complete"
