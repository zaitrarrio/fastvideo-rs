#!/usr/bin/env bash
# Download and verify taew2_1.safetensors into $1 (default
# ~/.cache/fastvideo/taehv, where the Wan pipeline's distilled presets look for
# it; FASTVIDEO_TAE_DIR=<dir> points them anywhere else). Run by remote.sh
# fetch-taehv. Pinned to the madebyollin/taehv commit fetch-tae.sh pins.
set -euo pipefail
dest="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/fastvideo/taehv}"
mkdir -p "$dest"
commit=e589fddc076e77f5ba8cd6baabe4ba3260b261cd
sha=04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f
if [[ -f "$dest/taew2_1.safetensors" ]] && [[ "$(sha256sum "$dest/taew2_1.safetensors" | awk '{print $1}')" == "$sha" ]]; then
  touch "$dest/.complete"
  exit 0
fi
url="https://raw.githubusercontent.com/madebyollin/taehv/$commit/safetensors/taew2_1.safetensors"
curl -fsSL --retry 5 --retry-delay 3 -o "$dest/taew2_1.safetensors.part" "$url"
got="$(sha256sum "$dest/taew2_1.safetensors.part" | awk '{print $1}')"
if [[ "$got" != "$sha" ]]; then
  echo "taew2_1 sha256 $got, expected $sha" >&2
  rm -f "$dest/taew2_1.safetensors.part"
  exit 1
fi
mv "$dest/taew2_1.safetensors.part" "$dest/taew2_1.safetensors"
touch "$dest/.complete"
