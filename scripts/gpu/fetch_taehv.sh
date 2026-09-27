#!/usr/bin/env bash
# Download and verify taew2_1.safetensors (Wan 2.1, 16 channels) and
# taew2_2.safetensors (Wan 2.2 TI2V-5B, 48 channels) into $1 (default
# ~/.cache/fastvideo/taehv, where the Wan pipeline looks for them;
# FASTVIDEO_TAE_DIR=<dir> points it anywhere else). Run by remote.sh
# fetch-taehv. Pinned to the madebyollin/taehv commit fetch-tae.sh pins.
# The weight volume's copy comes first: $FV_AUX_DIR/tae (default
# <weights>/auxiliary/tae) is copied when its SHA-256 matches; the pinned URL
# is the fallback.
set -euo pipefail
dest="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/fastvideo/taehv}"
vol="${FV_AUX_DIR:-${FV_WEIGHTS:-${FV_WORK:-/workspace}/weights}/auxiliary}/tae"
mkdir -p "$dest"
commit=e589fddc076e77f5ba8cd6baabe4ba3260b261cd

# fetch <name> <sha256>
fetch() {
  local name="$1" sha="$2" got url
  if [[ -f "$dest/$name.safetensors" ]] && [[ "$(sha256sum "$dest/$name.safetensors" | awk '{print $1}')" == "$sha" ]]; then
    return 0
  fi
  if [[ -f "$vol/$name.safetensors" ]] && cp -f "$vol/$name.safetensors" "$dest/$name.safetensors.part" \
    && [[ "$(sha256sum "$dest/$name.safetensors.part" | awk '{print $1}')" == "$sha" ]]; then
    mv "$dest/$name.safetensors.part" "$dest/$name.safetensors"
    echo "$name: volume copy $vol/$name.safetensors"
    return 0
  fi
  rm -f "$dest/$name.safetensors.part"
  url="https://raw.githubusercontent.com/madebyollin/taehv/$commit/safetensors/$name.safetensors"
  curl -fsSL --retry 5 --retry-delay 3 -o "$dest/$name.safetensors.part" "$url"
  got="$(sha256sum "$dest/$name.safetensors.part" | awk '{print $1}')"
  if [[ "$got" != "$sha" ]]; then
    echo "$name sha256 $got, expected $sha" >&2
    rm -f "$dest/$name.safetensors.part"
    return 1
  fi
  mv "$dest/$name.safetensors.part" "$dest/$name.safetensors"
}

fetch taew2_1 04766eac0221b5390b985ae3fdcca652cbb4b1e8b82b28ea7ff89dfad1b1a93f
# 22 848 048 bytes.
fetch taew2_2 b84609b2a133d48434bd9636bfcb44bf05168dc436e2d3cecf26256faa1f5325
touch "$dest/.complete"
