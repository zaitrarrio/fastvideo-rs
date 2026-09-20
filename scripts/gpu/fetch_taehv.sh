#!/usr/bin/env bash
# Download and verify taew2_1.safetensors into $1. Run by remote.sh fetch-taehv.
set -euo pipefail
dest="$1"
url="https://github.com/madebyollin/taehv/raw/main/safetensors/taew2_1.safetensors"
curl -fsSL --retry 5 --retry-delay 3 -o "$dest/taew2_1.safetensors.part" "$url"
here="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$here/verify-safetensors.sh" ]]; then
  bash "$here/verify-safetensors.sh" "$dest/taew2_1.safetensors.part"
else
  size=$(wc -c <"$dest/taew2_1.safetensors.part" | tr -d ' ')
  (( size > 1000000 )) || { echo "taehv too small: $size" >&2; exit 1; }
fi
mv "$dest/taew2_1.safetensors.part" "$dest/taew2_1.safetensors"
touch "$dest/.complete"
