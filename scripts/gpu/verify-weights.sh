#!/usr/bin/env bash
# Verify that a weight tree is complete for a model cell before any GPU time is
# spent on it. Runs on the box (no Python needed), against whatever volume is
# mounted. Volumes: EU only since 2026-10-06 (jg48s6o1w0; every cell passed
# there that day); the US volume s2k01690bi was deleted (docs/ops/runpod-volumes.md).
#
#   verify-weights.sh <cell>...        cells: fasth3-8step h3-base h3-ref2va h3-ref2va-turbo fasth3-4step-vsa
#                                      fasth3-4step-dense sol-h3 sol-h3-spark
#                                      ltx25-two-stage ltx23 fastwan21-1.3b
#                                      wan22-ti2v-5b fastwan22-ti2v-5b wan21-t2v-14b
#                                      sfwan21-1.3b sana-video-2b-480p
#                                      mmaudio-44k-v2
#                                      hy15-480-t2v hy15-480-i2v hy15-720-t2v
#                                      hy15-720-i2v aux text-fp8 dit-prequant upscalers
#                                      ltx25-ic-lora-ingredients ltx25-ref2v
#                                      ltx25-dev ltx25-a2v-guided ltx2
#                                      longlive-1.3b longlive2-5b longlive2-5b-nvfp4
#                                      longlive-plug
#                                      wan21-t2v-1.3b wan22-t2v-a14b ltx23-hq
#                                      lingbot-moe cosmos3-super
#                                      sha:<dest> (dest in weights-sha256.tsv)
#   verify-weights.sh --list           print the cells and what each needs
#
# For every weight root a cell needs, this checks:
#   1. each required component directory / file exists (dir, or file for LoRAs);
#   2. every `*.safetensors.index.json` lists shards that all exist on disk;
#   3. every safetensors file is its full length (header + data, via
#      verify-safetensors.sh), following HF-cache symlinks.
# The `aux` cell instead checks every auxiliary/ url: row of weights-manifest.tsv (TAE and
# LPIPS files): present, the listed size, the listed SHA-256.
# The `upscalers` cell checks the video super-resolution trees under
# auxiliary/upscalers/ (SeedVR2 3B, FlashVSR v1.1): a .complete marker and
# every pinned file with its size and SHA-256 (about 14 GB read).
# The `text-fp8` cell checks the optional pre-quantized FP8 text encoders
# (`fv-gpucheck quantize-text-encoder`, E13) in h3-base/ and ltx25/: the
# manifest's SHA-256 and the model's size and full length, plus the model's
# SHA-256 when FV_VERIFY_FP8_SHA=1 (about 40 GB read). A missing tree only
# means the loader quantizes at load; the cell reports it as INCOMPLETE.
# The `dit-prequant` cell checks the optional pre-quantized resident H3 DiT
# trees (`fv-gpucheck quantize-dit`, fast boot A) in h3-base/: the manifest's
# SHA-256 and the model's size and full length, plus the model's SHA-256 when
# FV_VERIFY_DIT_SHA=1 (about 26 GB read). A missing tree only means the
# loader quantizes at load; the cell reports it as INCOMPLETE.
# A `sha:<dest>` cell checks every file weights-sha256.tsv lists for <dest>
# (hashes recorded at fetch time): present, the listed size, the listed
# SHA-256 or md5 (alternatives a|b accepted). It reads the whole of each file (h3-ref2va: 69 GB).
# The longlive* cells (docs/serve/research-longlive.md) are composite: their
# directories / safetensors as above, plus the sha:<dest> lists of their trees
# (the .pt checkpoints are only checked that way; about 8.5 GB read for
# longlive-1.3b, 10 GB for longlive2-5b, 5.9 GB for -nvfp4, 13 GB for -plug).
# Exit status is non-zero on the first cell with a gap; the report names it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${FV_WEIGHTS:-${FV_WORK:-/workspace}/weights}"
UPSCALER_REL="upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
# The Spark latent upscaler at the revision pinned in weights-manifest.tsv:
# size and the Hub's LFS SHA-256 (0.69 GB read), checked with its length.
UPSCALER_SIZE=690592992
UPSCALER_SHA256=4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6

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
    # H3 Ref2VA (docs/ports/h3-ref2v.md): transformer_ref + the lightx2v
    # turbo LoRAs in h3-ref2va (fetch-h3-ref2va.sh), the rest from h3-base.
    h3-ref2va)
      echo "h3-ref2va:transformer_ref h3-base:vae h3-base:audio_vae h3-base:tokenizer h3-base:text_encoder" ;;
    h3-ref2va-turbo)
      echo "$(needs h3-ref2va) h3-ref2va:Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors" ;;
    ltx25-two-stage)
      echo "ltx25:transformer ltx25:connectors ltx25:vae ltx25:audio_vae ltx25:vocoder ltx25:latent_upsampler ltx25:text_encoder ltx25:tokenizer" ;;
    # FastVideo/FastWan2.1-T2V-1.3B-Diffusers (the wan family).
    fastwan21-1.3b)
      echo "fastwan21-1.3b:transformer fastwan21-1.3b:vae fastwan21-1.3b:text_encoder fastwan21-1.3b:tokenizer" ;;
    # Wan Diffusers trees (weights-manifest.tsv rows of the same name).
    wan22-ti2v-5b | fastwan22-ti2v-5b | wan21-t2v-14b | sfwan21-1.3b | wan21-t2v-1.3b)
      echo "$1:transformer $1:vae $1:text_encoder $1:tokenizer $1:scheduler" ;;
    # MMAudio large-44k-v2 V2A (scripts/gpu/fetch-mmaudio.py): the converted
    # safetensors the port reads, plus the upstream .pth tree and the CLIP
    # tokenizer. The .pth files are md5-checked at fetch time.
    mmaudio-44k-v2)
      local m=mmaudio-44k-v2/safetensors
      echo ":$m/mmaudio_large_44k_v2.safetensors :$m/vae_44k.safetensors :$m/synchformer.safetensors :$m/bigvgan_v2_44k.safetensors :$m/clip_dfn5b_h14_384.safetensors mmaudio-44k-v2:weights mmaudio-44k-v2:ext_weights mmaudio-44k-v2:bigvgan_v2_44khz_128band_512x mmaudio-44k-v2:DFN5B-CLIP-ViT-H-14-384" ;;
    # SANA-Video 2B 480p Diffusers tree (weights-manifest.tsv sana-video-2b-480p;
    # docs/ports/sana-video.md). vae/ is the Wan 2.1 VAE, byte-identical.
    sana-video-2b-480p)
      echo "$1:transformer $1:vae $1:text_encoder $1:tokenizer $1:scheduler" ;;
    # HunyuanVideo 1.5 Diffusers trees (weights-manifest.tsv rows of the same name).
    hy15-480-t2v | hy15-480-i2v | hy15-720-t2v | hy15-720-i2v)
      echo "$1:transformer $1:vae $1:text_encoder $1:text_encoder_2 $1:tokenizer $1:tokenizer_2 $1:scheduler" ;;
    ltx23)
      echo "ltx23:transformer ltx23:vae ltx23:audio_vae ltx23:vocoder ltx23:text_encoder ltx23:text_encoder/gemma ltx23:tokenizer ltx23:text_embedding_projection ltx23:spatial_upscaler" ;;
    # LTX-2.5 reference mode (docs/ports/ltx-ref2v.md; fetch-ltx-iclora.sh).
    ltx25-ic-lora-ingredients)
      echo ":ltx25-ic-lora-ingredients/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors" ;;
    # LTX-2.5 reference-to-video: the two-stage base plus the Ingredients IC-LoRA.
    ltx25-ref2v) echo "$(needs ltx25-two-stage) $(needs ltx25-ic-lora-ingredients)" ;;
    # The LTX-2.5 dev transformer (weights-manifest.tsv ltx25-dev; fetch-hub-tree.sh).
    ltx25-dev) echo "ltx25-dev:transformer_full" ;;
    # Guided audio-to-video (A2VidPipelineTwoStage): the two-stage bundle, the
    # dev transformer and the distilled LoRA beside the bundle.
    ltx25-a2v-guided)
      echo "$(needs ltx25-two-stage) $(needs ltx25-dev) :ltx25/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" ;;
    # LTX-2 19B distilled (weights-manifest.tsv ltx2): the single-file DiT the
    # ltx2 matrix cells pass as --dit, and the Diffusers parts beside it.
    ltx2)
      echo "ltx2:ltx-2-19b-distilled.safetensors ltx2:text_encoder ltx2:tokenizer ltx2:vae ltx2:audio_vae ltx2:vocoder" ;;
    # LongLive-1.3B (NON-COMMERCIAL weights): the Hub .pt tree, its converted
    # safetensors (convert-longlive.py) and SF-Wan's text encoder, tokenizer,
    # VAE and scheduler (wan stream --longlive). Plus sha:longlive-1.3b(-safetensors).
    longlive-1.3b)
      echo "longlive-1.3b:models longlive-1.3b:prompts :longlive-1.3b-safetensors/longlive_base.safetensors :longlive-1.3b-safetensors/lora.safetensors sfwan21-1.3b:vae sfwan21-1.3b:text_encoder sfwan21-1.3b:tokenizer sfwan21-1.3b:scheduler" ;;
    # LongLive-2.0-5B (merged BF16 .pt) on the Wan2.2-TI2V-5B tree. Plus sha:longlive2-5b.
    longlive2-5b) echo ":longlive2-5b $(needs wan22-ti2v-5b)" ;;
    # The two NVFP4 (FourOverSix) checkpoints. Plus their sha: lists.
    longlive2-5b-nvfp4) echo ":longlive2-5b-nvfp4-s4 :longlive2-5b-nvfp4-s2" ;;
    # The six LongLive-Plug LoRA trees. Plus their sha: lists.
    longlive-plug)
      echo "longlive-plug:minimax-h3-few-step longlive-plug:minimax-h3-cfg longlive-plug:wan21-t2v-14b-few-step longlive-plug:wan21-t2v-14b-cfg longlive-plug:wan22-ti2v-5b-few-step longlive-plug:wan22-ti2v-5b-cfg" ;;
    # LingBot-Video MoE 30B-A3B: base + 1080p refiner DiTs, Qwen3-VL, Wan 2.1 VAE
    # (weights-manifest.tsv lingbot-video-moe-30b-a3b; docs/ports/lingbot.md).
    lingbot-moe)
      local l=lingbot-video-moe-30b-a3b
      echo "$l:transformer $l:refiner $l:text_encoder $l:processor $l:vae $l:scheduler" ;;
    # Cosmos3-Super 64B T2V: the MoT transformer, Wan 2.2 VAE, Qwen2 tokenizer
    # (weights-manifest.tsv cosmos3-super; docs/ports/cosmos3.md).
    cosmos3-super)
      echo "cosmos3-super:transformer cosmos3-super:vae cosmos3-super:text_tokenizer cosmos3-super:scheduler" ;;
    # Wan2.2 T2V-A14B: the high-noise (transformer) and low-noise
    # (transformer_2) experts (docs/ports/wan.md "Wan 2.2 T2V-A14B").
    wan22-t2v-a14b)
      echo "$1:transformer $1:transformer_2 $1:vae $1:text_encoder $1:tokenizer $1:scheduler" ;;
    # LTX-2.3 official HQ (sol-engine models/ltx23.toml): the ltx23 tree for
    # Gemma, the VAEs and the vocoder, plus Lightricks/LTX-2.3's dev DiT, the
    # distilled LoRA 384 v1.1 and the x2 v1.1 upscaler in ltx23-dev/.
    ltx23-hq)
      echo "$(needs ltx23) :ltx23-dev/ltx-2.3-22b-dev.safetensors :ltx23-dev/ltx-2.3-22b-distilled-lora-384-1.1.safetensors :ltx23-dev/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" ;;
    aux) echo "aux" ;;
    text-fp8) echo "text-fp8" ;;
    dit-prequant) echo "dit-prequant" ;;
    upscalers) echo "upscalers" ;;
    *) return 1 ;;
  esac
}

CELLS=(fasth3-8step h3-base fasth3-4step-vsa fasth3-4step-dense sol-h3 sol-h3-spark h3-ref2va h3-ref2va-turbo ltx25-two-stage ltx23 fastwan21-1.3b
  wan22-ti2v-5b fastwan22-ti2v-5b wan21-t2v-14b sfwan21-1.3b mmaudio-44k-v2 hy15-480-t2v hy15-480-i2v hy15-720-t2v hy15-720-i2v aux text-fp8 dit-prequant upscalers
  ltx25-ic-lora-ingredients ltx25-ref2v ltx25-dev ltx25-a2v-guided ltx2 longlive-1.3b longlive2-5b longlive2-5b-nvfp4 longlive-plug
  sana-video-2b-480p wan21-t2v-1.3b wan22-t2v-a14b ltx23-hq lingbot-moe cosmos3-super)

# The weights-sha256.tsv dests a composite cell also checks (sha:<dest>).
sha_dests() {
  case "$1" in
    longlive-1.3b) echo "longlive-1.3b longlive-1.3b-safetensors" ;;
    longlive2-5b) echo "longlive2-5b" ;;
    longlive2-5b-nvfp4) echo "longlive2-5b-nvfp4-s4 longlive2-5b-nvfp4-s2" ;;
    longlive-plug)
      echo "longlive-plug/minimax-h3-few-step longlive-plug/minimax-h3-cfg longlive-plug/wan21-t2v-14b-few-step longlive-plug/wan21-t2v-14b-cfg longlive-plug/wan22-ti2v-5b-few-step longlive-plug/wan22-ti2v-5b-cfg" ;;
  esac
}
SHA_LIST="$HERE/weights-sha256.tsv"

if [[ "${1:-}" == "--list" ]]; then
  for c in "${CELLS[@]}"; do printf '%-20s %s\n' "$c" "$(needs "$c")"; done
  if [[ -f "$SHA_LIST" ]]; then
    grep -vE '^[[:space:]]*(#|$)' "$SHA_LIST" | cut -f1 | sort | uniq -c \
      | awk '{printf "%-20s %s files by recorded hash (weights-sha256.tsv)\n", "sha:" $2, $1}'
  fi
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
    if [[ "$p" == "$W/$UPSCALER_REL" && -f "$p" ]]; then
      local got
      got="$(wc -c <"$p" | tr -d ' ')"
      if [[ "$got" != "$UPSCALER_SIZE" ]]; then
        echo "  SIZE $p: $got, expected $UPSCALER_SIZE" >&2; rc=1
      elif [[ "$(sha256sum "$p" | awk '{print $1}')" != "$UPSCALER_SHA256" ]]; then
        echo "  SHA256 $p differs from $UPSCALER_SHA256" >&2; rc=1
      fi
    fi
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
  done < <(grep -E $'^auxiliary/[^\t]+\turl:' "$HERE/weights-manifest.tsv")
  (( n > 0 )) || { echo "  no auxiliary/ rows in weights-manifest.tsv" >&2; rc=1; }
  return $rc
}

# text-fp8: the E13 trees as written on the US volume and copied to EU
# (docs/gaps/2026-09-27-volume-sync.md). rel<TAB>bytes<TAB>sha256.
FP8_TREES="h3-base/text_encoder_fp8/manifest.json	2965	e5504561d8fcf188c33d9ad9bb282e80bd45693dc087f9ad980034c9ee190a42
h3-base/text_encoder_fp8/model.safetensors	25950724552	c13fab5c3d7225f9582fdc57a6aa6ba83ed8830114fa78340307cfd0b9072c2d
ltx25/text_encoder_fp8/manifest.json	1408	bb5a30499c4e4e647b27a608b11384aa6aaba712534b1d737fcca164adba06db
ltx25/text_encoder_fp8/model.safetensors	12923848536	0615832bb0a0e0b4eabd31a27eb1e674cb7b7dc59534d03889b50fa49eb4ae2f"
check_text_fp8() {
  local rc=0 rel size sha p got
  while IFS=$'\t' read -r rel size sha; do
    p="$W/$rel"
    if [[ ! -f "$p" ]]; then echo "  MISSING $p" >&2; rc=1; continue; fi
    got="$(wc -c <"$p" | tr -d ' ')"
    if [[ "$got" != "$size" ]]; then echo "  SIZE $p: $got, expected $size" >&2; rc=1; continue; fi
    if [[ "$p" == *.safetensors ]]; then
      bash "$HERE/verify-safetensors.sh" "$p" >/dev/null || rc=1
      [[ "${FV_VERIFY_FP8_SHA:-0}" == 1 ]] || continue
    fi
    got="$(sha256sum "$p" | awk '{print $1}')"
    if [[ "$got" != "$sha" ]]; then echo "  SHA256 $p: $got, expected $sha" >&2; rc=1; fi
  done <<<"$FP8_TREES"
  return $rc
}

# dit-prequant: the fast-boot trees as written on EU by `fv-gpucheck
# quantize-dit` (docs/gaps/2026-10-07-fast-boot.md). rel<TAB>bytes<TAB>sha256.
DIT_PREQUANT_TREES="h3-base/transformer_prequant_4step-vsa_mxfp8/manifest.json	3809	e80ec8c23a1e18581eef48c2c0e7d95f181072e0af4eae8c611f0c710945ceee
h3-base/transformer_prequant_4step-vsa_mxfp8/model.safetensors	27288660164	d3dcc7b3ff3b05783cbab37e95cc1c3e8113b750400bc27db14a1a37ae6ddc18
h3-base/transformer_prequant_sol-h3_mxfp8/manifest.json	3809	6dc6db7d53469e8b2b230b36b957e8f7b3aef76a9a0e0cef04c571f90c2aeeef
h3-base/transformer_prequant_sol-h3_mxfp8/model.safetensors	23435142712	cf58b8d2f264509a61df819026b9f63cf1955f6476a5519b8300717cfc7849f3"
check_dit_prequant() {
  local rc=0 rel size sha p got
  while IFS=$'\t' read -r rel size sha; do
    p="$W/$rel"
    if [[ ! -f "$p" ]]; then echo "  MISSING $p" >&2; rc=1; continue; fi
    got="$(wc -c <"$p" | tr -d ' ')"
    if [[ "$got" != "$size" ]]; then echo "  SIZE $p: $got, expected $size" >&2; rc=1; continue; fi
    if [[ "$p" == *.safetensors ]]; then
      bash "$HERE/verify-safetensors.sh" "$p" >/dev/null || rc=1
      [[ "${FV_VERIFY_DIT_SHA:-0}" == 1 ]] || continue
    fi
    got="$(sha256sum "$p" | awk '{print $1}')"
    if [[ "$got" != "$sha" ]]; then echo "  SHA256 $p: $got, expected $sha" >&2; rc=1; fi
  done <<<"$DIT_PREQUANT_TREES"
  return $rc
}

# upscalers: the auxiliary/upscalers/ trees (weights-manifest.tsv Hub rows at
# the revisions pinned there). rel<TAB>bytes<TAB>sha256: the Hub's LFS SHA-256;
# the two small JSON files hashed at the pinned revision.
UPSCALER_TREES="auxiliary/upscalers/seedvr2/seedvr2_ema_3b_fp16.safetensors	6783018808	2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304
auxiliary/upscalers/seedvr2/ema_vae_fp16.safetensors	501324814	20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1
auxiliary/upscalers/flashvsr-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors	5676070392	bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be
auxiliary/upscalers/flashvsr-v1.1/LQ_proj_in.ckpt	575694948	d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2
auxiliary/upscalers/flashvsr-v1.1/TCDecoder.ckpt	189018333	e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161
auxiliary/upscalers/flashvsr-v1.1/Wan2.1_VAE.pth	507609880	38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
auxiliary/upscalers/flashvsr-v1.1/config.json	30	2c51a6dc7dfcb5aaef299c6c2e1d25b870666760713f54ca6287c022ff189bd1
auxiliary/upscalers/flashvsr-v1.1/model_index.json	73	85be89792ac836177eba736bce70ea44b034cf18bf5d527e40e8a4a5299e73b6"
check_upscalers() {
  local rc=0 rel size sha p got d
  for d in seedvr2 flashvsr-v1.1; do
    [[ -f "$W/auxiliary/upscalers/$d/.complete" ]] || { echo "  NO MARKER $W/auxiliary/upscalers/$d/.complete" >&2; rc=1; }
  done
  while IFS=$'\t' read -r rel size sha; do
    p="$W/$rel"
    if [[ ! -f "$p" ]]; then echo "  MISSING $p" >&2; rc=1; continue; fi
    got="$(wc -c <"$p" | tr -d ' ')"
    if [[ "$got" != "$size" ]]; then echo "  SIZE $p: $got, expected $size" >&2; rc=1; continue; fi
    got="$(sha256sum "$p" | awk '{print $1}')"
    if [[ "$got" != "$sha" ]]; then echo "  SHA256 $p: $got, expected $sha" >&2; rc=1; fi
  done <<<"$UPSCALER_TREES"
  return $rc
}

# sha:<dest>: every weights-sha256.tsv row of <dest>, by size and hash.
check_sha_list() {
  local want="$1" rc=0 dest rel size hash algo hex p got n=0
  [[ -f "$SHA_LIST" ]] || { echo "  no $SHA_LIST" >&2; return 1; }
  while IFS=$'\t' read -r dest rel size hash; do
    [[ "$dest" == "$want" ]] || continue
    n=$((n + 1))
    algo="${hash%%:*}"; hex="${hash#*:}"
    p="$W/$dest/$rel"
    if [[ ! -f "$p" ]]; then echo "  MISSING $p" >&2; rc=1; continue; fi
    got="$(wc -c <"$p" | tr -d ' ')"
    if [[ "$size" != "-" && "$got" != "$size" ]]; then echo "  SIZE $p: $got, expected $size" >&2; rc=1; continue; fi
    case "$algo" in
      sha256) got="$(sha256sum "$p" | awk '{print $1}')" ;;
      md5) got="$(md5sum "$p" | awk '{print $1}')" ;;
      *) echo "  unknown hash $algo for $p" >&2; rc=1; continue ;;
    esac
    # hex may list per-volume alternatives as a|b (a derived file whose bytes
    # differ by volume only in safetensors metadata order; weights-sha256.tsv says which).
    if [[ "|$hex|" != *"|$got|"* ]]; then echo "  ${algo^^} $p: $got, expected $hex" >&2; rc=1; fi
  done < <(grep -vE '^[[:space:]]*(#|$)' "$SHA_LIST")
  (( n > 0 )) || { echo "  no weights-sha256.tsv rows for $want" >&2; rc=1; }
  return $rc
}

for cell in "$@"; do
  if [[ "$cell" == sha:* ]]; then
    if check_sha_list "${cell#sha:}"; then echo "weights ok: $cell"; else echo "weights INCOMPLETE: $cell" >&2; fail=1; fi
    continue
  fi
  if [[ "$cell" == upscalers ]]; then
    if check_upscalers; then echo "weights ok: upscalers"; else echo "weights INCOMPLETE: upscalers" >&2; fail=1; fi
    continue
  fi
  if [[ "$cell" == dit-prequant ]]; then
    if check_dit_prequant; then echo "weights ok: dit-prequant"; else echo "weights INCOMPLETE: dit-prequant" >&2; fail=1; fi
    continue
  fi
  if [[ "$cell" == text-fp8 ]]; then
    if check_text_fp8; then echo "weights ok: text-fp8"; else echo "weights INCOMPLETE: text-fp8" >&2; fail=1; fi
    continue
  fi
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
  for d in $(sha_dests "$cell"); do
    check_sha_list "$d" || cell_rc=1
  done
  if (( cell_rc == 0 )); then
    echo "weights ok: $cell"
  else
    echo "weights INCOMPLETE: $cell" >&2
    fail=1
  fi
done
exit $fail
