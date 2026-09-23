#!/usr/bin/env bash
# Download and verify taeh3.safetensors into $1. Run by remote.sh fetch-taeh3.
set -euo pipefail
dest="$1"
url="https://github.com/madebyollin/taehv/raw/main/safetensors/taeh3.safetensors"
curl -fsSL --retry 5 --retry-delay 3 -o "$dest/taeh3.safetensors.part" "$url"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer repo-relative verify script; fall back to same-dir when uploaded alone.
if [[ -x "$ROOT/scripts/gpu/verify-safetensors.sh" ]]; then
  bash "$ROOT/scripts/gpu/verify-safetensors.sh" "$dest/taeh3.safetensors.part"
elif [[ -x "$(dirname "$0")/verify-safetensors.sh" ]]; then
  bash "$(dirname "$0")/verify-safetensors.sh" "$dest/taeh3.safetensors.part"
else
  # Header-length sanity only.
  size=$(wc -c <"$dest/taeh3.safetensors.part" | tr -d ' ')
  (( size > 1000000 )) || { echo "taeh3 too small: $size" >&2; exit 1; }
fi
mv "$dest/taeh3.safetensors.part" "$dest/taeh3.safetensors"
touch "$dest/.complete"
