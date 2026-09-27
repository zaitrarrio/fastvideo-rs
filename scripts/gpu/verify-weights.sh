#!/usr/bin/env bash
# Verify that a weight tree is complete for a model cell before any GPU time is
# spent on it. Runs on the box (no Python needed).
#
#   verify-weights.sh <cell>...        cells: fasth3-8step h3-base fasth3-4step-vsa
#                                      fasth3-4step-dense sol-h3 sol-h3-spark
#                                      ltx25-two-stage ltx23 fastwan21-1.3b
#                                      wan22-ti2v-5b wan21-t2v-14b sfwan21-1.3b
#                                      hy15-480-t2v hy15-480-i2v hy15-720-t2v
#                                      hy15-720-i2v aux
#   verify-weights.sh --list           print the cells and what each needs
#
# For every weight root a cell needs, this checks:
#   1. each required component directory / file exists (dir, or file for LoRAs);
#   2. every `*.safetensors.index.json` lists shards that all exist on disk;
#   3. every safetensors file is its full length (header + data, via
#      verify-safetensors.sh), following HF-cache symlinks.
# The `aux` cell instead checks every auxiliary/ row of weights-manifest.tsv (TAE and
# LPIPS files): present, the listed size, the listed SHA-256.
# Exit status is non-zero on the first cell with a gap; the report names it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${FV_WEIGHTS:-${FV_WORK:-/workspace}/weights}"
UPSCALER_REL="upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

# cell -> space-separated "root:component" requirements. A component ending in
# .safetensors is a single file; anything else is a directory.
needs() {
  case "$1" in
    fasth3-8step)
      echo "h3-8step:transformer h3-8step:vae h3-8step:audio_vae h3-base:tokenizer h3-base:text_encoder" ;;
    h3-base)
      echo "h3-base:transformer h3-base:vae h3-base:audio_vae h3-base:tokenizer h3-base:text_encoder" ;;
    # FastH3 Preview v1 = the base transformer + one Preview LoRA (FastVideo's
    # run_fasth3_lora_preview_*_datafree.sh); the 8-step export is refused.
    fasth3-4step-vsa)
      echo "$(needs h3-base) FastH3-4-step-Preview-v1-LoRA:vsa-datafree/adapter_model.safetensors" ;;
    fasth3-4step-dense)
      echo "$(needs h3-base) FastH3-4-step-Preview-v1-LoRA:dense-datafree/adapter_model.safetensors" ;;
    sol-h3)
      echo "h3-base:transformer h3-base:vae h3-base:audio_vae h3-base:tokenizer h3-base:text_encoder FastH3-4-step-Preview-v1-LoRA:dense-datafree/adapter_model.safetensors" ;;
    sol-h3-spark)
      echo "h3-base:transformer h3-base:vae h3-base:audio_vae h3-base:tokenizer h3-base:text_encoder FastH3-4-step-Preview-v1-LoRA:vsa-datafree/adapter_model.safetensors :$UPSCALER_REL h3-to-ltx:model.safetensors $(needs ltx25-two-stage)" ;;
    ltx25-two-stage)
      echo "ltx25:transformer ltx25:connectors ltx25:vae ltx25:audio_vae ltx25:vocoder ltx25:latent_upsampler ltx25:text_encoder ltx25:tokenizer" ;;
    # FastVideo/FastWan2.1-T2V-1.3B-Diffusers (the wan family).
    fastwan21-1.3b)
      echo "fastwan21-1.3b:transformer fastwan21-1.3b:vae fastwan21-1.3b:text_encoder fastwan21-1.3b:tokenizer" ;;
    # Wan Diffusers trees (weights-manifest.tsv rows of the same name).
    wan22-ti2v-5b | wan21-t2v-14b | sfwan21-1.3b)
      echo "$1:transformer $1:vae $1:text_encoder $1:tokenizer $1:scheduler" ;;
    # HunyuanVideo 1.5 Diffusers trees (weights-manifest.tsv rows of the same name).
    hy15-480-t2v | hy15-480-i2v | hy15-720-t2v | hy15-720-i2v)
      echo "$1:transformer $1:vae $1:text_encoder $1:text_encoder_2 $1:tokenizer $1:tokenizer_2 $1:scheduler" ;;
    ltx23)
      echo "ltx23:transformer ltx23:vae ltx23:audio_vae ltx23:vocoder ltx23:text_encoder ltx23:text_encoder/gemma ltx23:tokenizer ltx23:text_embedding_projection ltx23:spatial_upscaler" ;;
    aux) echo "aux" ;;
    *) return 1 ;;
  esac
}

CELLS=(fasth3-8step h3-base fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark ltx25-two-stage ltx23 fastwan21-1.3b
  wan22-ti2v-5b wan21-t2v-14b sfwan21-1.3b hy15-480-t2v hy15-480-i2v hy15-720-t2v hy15-720-i2v aux)

if [[ "${1:-}" == "--list" ]]; then
  for c in "${CELLS[@]}"; do printf '%-20s %s\n' "$c" "$(needs "$c")"; done
  exit 0
fi
(( $# >= 1 )) || { echo "usage: $0 <cell>... | --list" >&2; exit 2; }

fail=0
declare -A checked=()

check_index() {
  # Every shard named in an index's weight_map must be present.
  local idx="$1" dir shard
  dir="$(dirname "$idx")"
  while IFS= read -r shard; do
    [[ -n "$shard" ]] || continue
    if [[ ! -e "$dir/$shard" ]]; then
      echo "  MISSING shard $dir/$shard (listed in $(basename "$idx"))" >&2
      return 1
    fi
  done < <(grep -oE '"[^"]+\.safetensors"' "$idx" | tr -d '"' | sort -u)
}

check_path() {
  local p="$1"
  [[ -n "${checked[$p]:-}" ]] && return "${checked[$p]}"
  local rc=0
  if [[ "$p" == *.safetensors ]]; then
    bash "$HERE/verify-safetensors.sh" "$p" >/dev/null || rc=1
  elif [[ ! -d "$p" ]]; then
    echo "  MISSING dir $p" >&2
    rc=1
  else
    local idx
    while IFS= read -r -d '' idx; do
      check_index "$idx" || rc=1
    done < <(find -L "$p" -name '*.safetensors.index.json' -print0)
    if find -L "$p" -name '*.safetensors' -print -quit | grep -q .; then
      bash "$HERE/verify-safetensors.sh" --dir "$p" >/dev/null || rc=1
    elif ! find -L "$p" -type f -print -quit | grep -q .; then
      echo "  EMPTY dir $p" >&2
      rc=1
    fi
  fi
  checked[$p]=$rc
  return $rc
}

# aux: every auxiliary/ row of weights-manifest.tsv, by size and SHA-256.
check_aux() {
  local rc=0 rel url meta sha size p got n=0
  while IFS=$'\t' read -r rel url meta; do
    sha="${meta#sha256:}"; sha="${sha%% *}"
    size="${meta##*size:}"
    p="$W/$rel"
    n=$((n + 1))
    if [[ ! -f "$p" ]]; then
      echo "  MISSING $p" >&2; rc=1; continue
    fi
    if [[ "$(wc -c <"$p" | tr -d ' ')" != "$size" ]]; then
      echo "  SIZE $p: $(wc -c <"$p" | tr -d ' '), expected $size" >&2; rc=1; continue
    fi
    got="$(sha256sum "$p" | awk '{print $1}')"
    if [[ "$got" != "$sha" ]]; then
      echo "  SHA256 $p: $got, expected $sha" >&2; rc=1
    fi
  done < <(grep -E '^auxiliary/' "$HERE/weights-manifest.tsv")
  (( n > 0 )) || { echo "  no auxiliary/ rows in weights-manifest.tsv" >&2; rc=1; }
  return $rc
}

for cell in "$@"; do
  if [[ "$cell" == aux ]]; then
    if check_aux; then echo "weights ok: aux"; else echo "weights INCOMPLETE: aux" >&2; fail=1; fi
    continue
  fi
  reqs="$(needs "$cell")" || { echo "unknown cell $cell (see --list)" >&2; exit 2; }
  cell_rc=0
  for r in $reqs; do
    root="${r%%:*}"
    comp="${r#*:}"
    p="$W/${root:+$root/}$comp"
    check_path "$p" || cell_rc=1
  done
  if (( cell_rc == 0 )); then
    echo "weights ok: $cell"
  else
    echo "weights INCOMPLETE: $cell" >&2
    fail=1
  fi
done
exit $fail
