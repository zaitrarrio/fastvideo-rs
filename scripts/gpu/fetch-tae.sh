#!/usr/bin/env bash
# Download the madebyollin/taehv tiny-autoencoder weights the matrix cells use
# into $1 (default $FV_SCRATCH/tae, i.e. the pod's container disk — the weight
# volume is read-only for us). Idempotent: a file whose SHA-256 already
# matches is kept. Pinned to the commits sol-engine pins, as safetensors (our
# runtime image has no Python to convert the .pth):
#
#   taeh3.safetensors           e589fdd  (sol-engine models/minimax_h3/
#       super_acceleration/stage1/taeh3_decoder_telemetry_overlay.py:55-62 pins
#       taeh3.pth af92965c...; the safetensors at the same commit holds the
#       same 128 tensors bit for bit)
#   taeltx2_3_wide.safetensors  32ac014  (sol-engine models/ltx2.5-refiner/
#       GB200/prepare_taehv_weight.sh pins taeltx2_3_wide.pth 007788e6...;
#       the safetensors at the same commit holds the same 146 tensors bit for
#       bit)
#
#   fetch-tae.sh [dest]    -> dest/taeh3.safetensors, dest/taeltx2_3_wide.safetensors
set -euo pipefail
dest="${1:-${FV_SCRATCH:-/fvscratch}/tae}"
raw="https://raw.githubusercontent.com/madebyollin/taehv"
mkdir -p "$dest"

fetch() {
  local name="$1" commit="$2" sha="$3" size="$4" out part got
  out="$dest/$name"
  if [[ -f "$out" ]] && [[ "$(sha256sum "$out" | awk '{print $1}')" == "$sha" ]]; then
    echo "tae ok (cached): $out"
    return 0
  fi
  part="$out.part.$$"
  rm -f "$part"
  curl -fsSL --proto '=https' --retry 5 --retry-delay 3 --retry-all-errors \
    -o "$part" "$raw/$commit/safetensors/$name"
  got="$(sha256sum "$part" | awk '{print $1}')"
  if [[ "$got" != "$sha" ]]; then
    echo "tae FAIL: $name sha256 $got, expected $sha" >&2
    rm -f "$part"
    return 1
  fi
  [[ "$(wc -c <"$part" | tr -d ' ')" == "$size" ]] || { echo "tae FAIL: $name size" >&2; rm -f "$part"; return 1; }
  mv -f "$part" "$out"
  echo "tae ok: $out ($size bytes)"
}

rc=0
fetch taeh3.safetensors e589fddc076e77f5ba8cd6baabe4ba3260b261cd \
  4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13 22709752 || rc=1
fetch taeltx2_3_wide.safetensors 32ac0146b11007cda5a57b60a3b35653361fb8a4 \
  0a69291425015e8eb4309028e7de1d17d2a11f58ba88e0120453bc36f352e082 60359856 || rc=1
exit $rc
