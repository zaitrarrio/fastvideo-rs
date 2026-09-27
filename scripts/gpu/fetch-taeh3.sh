#!/usr/bin/env bash
# Download and verify taeh3.safetensors into $1. Run by remote.sh fetch-taeh3.
# Pinned to the madebyollin/taehv commit and SHA-256 fetch-tae.sh pins (the
# file on main was the same bytes on 2026-09-27). The weight volume's copy
# comes first: $FV_AUX_DIR/tae/taeh3.safetensors (default
# <weights>/auxiliary/tae, the auxiliary/ rows of weights-manifest.tsv) is
# copied when its SHA-256 matches; the pinned URL is the fallback.
set -euo pipefail
dest="$1"
commit=e589fddc076e77f5ba8cd6baabe4ba3260b261cd
sha=4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13
url="https://raw.githubusercontent.com/madebyollin/taehv/$commit/safetensors/taeh3.safetensors"
vol="${FV_AUX_DIR:-${FV_WEIGHTS:-${FV_WORK:-/workspace}/weights}/auxiliary}/tae/taeh3.safetensors"
mkdir -p "$dest"
part="$dest/taeh3.safetensors.part"
got=""
if [[ -f "$vol" ]] && cp -f "$vol" "$part"; then
  got="$(sha256sum "$part" | awk '{print $1}')"
  [[ "$got" == "$sha" ]] && echo "taeh3: volume copy $vol"
fi
if [[ "$got" != "$sha" ]]; then
  curl -fsSL --retry 5 --retry-delay 3 -o "$part" "$url"
  got="$(sha256sum "$part" | awk '{print $1}')"
fi
if [[ "$got" != "$sha" ]]; then
  echo "taeh3 sha256 $got, expected $sha" >&2
  rm -f "$part"
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer repo-relative verify script; fall back to same-dir when uploaded alone.
if [[ -x "$ROOT/scripts/gpu/verify-safetensors.sh" ]]; then
  bash "$ROOT/scripts/gpu/verify-safetensors.sh" "$part"
elif [[ -x "$(dirname "$0")/verify-safetensors.sh" ]]; then
  bash "$(dirname "$0")/verify-safetensors.sh" "$part"
fi
mv "$part" "$dest/taeh3.safetensors"
touch "$dest/.complete"
