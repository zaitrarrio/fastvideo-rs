#!/usr/bin/env bash
# Oracle cells (sourced by pod.sh): the Python references run once each with
# oracle_dump.py's hooks (FV_ORACLE_DUMP_DIR), writing fastvideo-rs's dump
# format. Each cell leaves <run>/oracle-<target>/oracle-dump.tar (the dump
# directory; the Rust pod of the `oracle` matrix family downloads it, injects
# its noise/text and diffs its own dump against it) and oracle_meta.json.
#
#   UP_STEPS="... oracle"                    every target below
#   UP_STEPS="... oracle:fasth3-8step,ltx25-512p"   a subset
#
# Targets: wan22-ti2v (Wan 2.2 TI2V-5B modules through Diffusers, oracle_wan22.py),
# fastwan22-ti2v (the same modules with the FastWan2.2 TI2V-5B FullAttn weights,
# DiT at the DMD timestep 757),
# fasth3-8step (FastVideo FastH3 8-step V2, 768x1344x124),
# fasth3-4step-vsa (MiniMax-H3 + Preview v1 vsa-datafree LoRA), the dense
# controls fasth3-8step-vsa0 (the 8-step checkpoint at VSA sparsity 0) and
# fasth3-4step-dense (dense-datafree LoRA, FLASH_ATTN), and the
# sol-engine LTX-2.5 distilled two-stage: ltx25-512p / ltx25-4k with Sol
# stage 2, ltx25-512p-dense / ltx25-4k-dense with dense stage 2, ltx25-i2v /
# ltx25-kf (512p dense, first-frame / first+last image conditioning),
# ltx25-ref2v (ltx_pipelines ICLoraPipeline with the Ingredients IC-LoRA on a
# reference sheet, 1536x896x121: the LoRA's 768x448 stage-1 bucket);
# ltx25-a2v / ltx25-a2v-i2v (audio-to-video, ltx25_a2v.py); ltx25-retake /
# ltx25-retake-v / ltx25-retake-a / ltx25-extend (retake of both streams, of
# the video, of the audio, and an extension, on one distilled stage at the
# source size: ltx25_edit.py); sfwan13
# (FastVideo SF-Wan 1.3B causal DMD, 480x832x81, bench_fastwan.py);
# h3-ref2va-4step (FastVideo MiniMaxH3Ref2VAModularPipeline on transformer_ref,
# one image reference, dense, 4 forwards on the uniform grid; ours: `base-4step`).
# FastVideo runs its strict eager route (--profile strict
# --no-inference-torch-compile): no report-only fusions, no compiled blocks
# for the hooks to break. FASTVIDEO_DUMP_OPS (default 0,1,24,47) picks the
# blocks whose inside is dumped at the first step.

ORACLE_OPS="${FASTVIDEO_DUMP_OPS:-0,1,24,47}"

# oracle_cell <name> <cmd...>: run_cell with the dump hooks on, then pack the dump.
oracle_cell() {
  local name="oracle-$1"; shift
  local c="$OUT/$name"
  if [[ -n "${UP_ORACLE:-}" && " $UP_ORACLE " != *" ${name#oracle-} "* ]]; then return 0; fi
  rm -rf "$c/dump" "$c/oracle-dump.tar"
  UP_CELLS="" run_cell "$name" env PYTHONPATH="$HERE/oracle_site${PYTHONPATH:+:$PYTHONPATH}" \
    FV_ORACLE_DUMP_DIR="$c/dump" FASTVIDEO_DUMP_OPS="$ORACLE_OPS" "$@"
  if [[ -d "$c/dump" ]]; then
    cp "$c/dump/oracle_meta.json" "$c/" 2>/dev/null
    # The reference's own video, for the frame metrics (LPIPS / PSNR).
    local mp4
    mp4="$(find "$c" -name '*.mp4' -not -path '*/warmup/*' -not -path '*/dump/*' 2>/dev/null | head -1)"
    [[ -n "$mp4" ]] && cp "$mp4" "$c/dump/ref.mp4"
    ls -la "$c/dump" >"$c/dump.ls" 2>&1
    du -sh "$c/dump" | tee -a "$LIVE" >&2
    # Written whole, then renamed: the Rust pod polls for this name.
    tar -cf "$c/oracle-dump.tar.part" -C "$c" dump && mv "$c/oracle-dump.tar.part" "$c/oracle-dump.tar" \
      && rm -rf "$c/dump"
  else
    log "oracle $name: no dump written"
  fi
  echo "done" >"$c/ORACLE_DONE"
}

# oracle_fv <name> <recipe> [--dense] -- <example args>: bench_fastvideo.py
# through oracle_fastvideo.py (--dense: FLASH_ATTN, no VSA, no gate).
oracle_fv() {
  local name="$1" recipe="$2" dense=(); shift 2
  [[ "${1:-}" == --dense ]] && { dense=(--dense); shift; }
  oracle_cell "$name" env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/oracle_fastvideo.py" \
    "${dense[@]}" --fastvideo-src "$SRC/FastVideo" --recipe "$recipe" --out "$OUT/oracle-$name" --repeats 1 -- \
    --prompt "$PROMPT_OURS" --seed "$SEED_OURS" --num-gpus 1 --vsa-kernel triton --no-fa4 \
    --no-warmup --profile strict --no-inference-torch-compile --no-compile-vae "$@"
}

# sol-engine LTX-2.5 two-stage; arm sol|dense (bench_ltx25.py), geometry per workload,
# then any extra official args (image conditioning: --image PATH FRAME_IDX STRENGTH).
oracle_ltx() {
  local name="$1" arm="$2" wl="$3" geo L="$UW/LTX-2.5" a2v=() prompt="$PROMPT_OURS"
  shift 3
  # --a2v AUDIO [PROMPT] first: audio-to-video (ltx25_a2v.py) on that audio.
  if [[ "${1:-}" == --a2v ]]; then
    a2v=(--a2v-audio "$2"); shift 2
    [[ -n "${1:-}" && "${1:-}" != --* ]] && { prompt="$1"; shift; }
  fi
  # --edit SPEC [PROMPT] first: retake / extend (ltx25_edit.py); pass the
  # generated clip's --num-frames after it when it differs from the source's.
  if [[ "${1:-}" == --edit ]]; then
    a2v=(--edit "$2"); shift 2
    [[ -n "${1:-}" && "${1:-}" != --* ]] && { prompt="$1"; shift; }
  fi
  case "$wl" in
    512p) geo=(--width 768 --height 512 --num-frames 121) ;;
    4k) geo=(--width 3840 --height 2176 --num-frames 121) ;;
  esac
  geo+=("$@")
  oracle_cell "$name" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "$UP/sol-ltx25/LTX-2/.venv/bin/python" "$HERE/bench_ltx25.py" --sol-engine "$SRC/sol-engine" \
    --arm "$arm" "${a2v[@]}" --result "$OUT/oracle-$name/result.json" -- \
    --pipeline bf16 --metrics "$OUT/oracle-$name/benchmark.json" -- \
    --transformer-path "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$L/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae-path "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --offload cpu "${geo[@]}" --frame-rate 24 --seed "$SEED_OURS" --prompt "$prompt" \
    --output-path "$OUT/oracle-$name/out.mp4"
}

# LTX-2.5 reference-to-video (docs/ports/ltx-ref2v.md): ltx_pipelines'
# ICLoraPipeline (`python -m ltx_pipelines.ic_lora`) at Lightricks/LTX-2 fd4ded7
# with the Ingredients IC-LoRA fused into stage 1, on the reference sheet
# looped into a lossless static clip of the output's length (the model card's
# input; PNG frames in QuickTime, so the decoded frames are the sheet's pixels
# exactly). The prompt and sheet are shared with runpod-matrix.sh (`oracle`
# target ltx25-ref2v).
LTX_REF_PROMPT="${FV_LTX_REF_PROMPT:-Reference sheet: Top Row Left (Setting): a rocky coastline at golden hour, dark boulders in the surf and green hills behind a sandy beach. Top Row Right (Setting): a closer view of the same boulders with waves breaking around them. Bottom Row Left (Prop): a red and white striped beach umbrella, shown twice. Bottom Row Right (Character): a cartoon orange crab with big claws and eyes on stalks, shown twice. Generated video: A bright 3D animated shot on the rocky beach at golden hour. The cheerful orange cartoon crab scuttles sideways across the wet sand in front of the dark boulders, waving its big claws, next to the red and white striped beach umbrella planted in the sand, while waves roll in and break into white foam behind it.}"
LTX_REF_SHEET="$HERE/../fixtures/ltx-ref-sheet-768x448.png"
LTX_IC_LORA="${FV_LTX_IC_LORA:-$W/ltx25-ic-lora-ingredients/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors}"

oracle_ltx_ref() {
  local name="$1" L="$UW/LTX-2.5" c="$OUT/oracle-$1" py="$UP/sol-ltx25/LTX-2/.venv/bin/python"
  if [[ -n "${UP_ORACLE:-}" && " $UP_ORACLE " != *" $name "* ]]; then return 0; fi
  [[ -f "$LTX_REF_SHEET" ]] || { log "oracle $name: no reference sheet $LTX_REF_SHEET (clone failed?)"; return 1; }
  [[ -f "$LTX_IC_LORA" ]] || { log "oracle $name: no IC-LoRA at $LTX_IC_LORA"; return 1; }
  mkdir -p "$c"
  "$py" - "$LTX_REF_SHEET" "$c/reference.mov" 121 24 <<'PY' || { log "oracle $name: static clip failed"; return 1; }
import sys
from fractions import Fraction

import av
import numpy as np

src, dst, frames, fps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with av.open(src) as img:
    rgb = next(img.decode(video=0)).to_rgb().to_ndarray()
with av.open(dst, "w") as out:
    st = out.add_stream("png", rate=Fraction(fps))
    st.width, st.height, st.pix_fmt = rgb.shape[1], rgb.shape[0], "rgb24"
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    for _ in range(frames):
        for p in st.encode(frame):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
with av.open(dst) as inp:
    got = [f.to_rgb().to_ndarray() for f in inp.decode(video=0)]
assert len(got) == frames and all(np.array_equal(g, rgb) for g in got), "static clip is not lossless"
print(f"static clip {dst}: {frames} frames {rgb.shape[1]}x{rgb.shape[0]} lossless")
PY
  oracle_cell "$name" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "$py" -m ltx_pipelines.ic_lora \
    --transformer-path "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$L/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae-path "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --lora "$LTX_IC_LORA" 1.0 --video-conditioning "$c/reference.mov" 1.0 \
    --offload cpu --width 1536 --height 896 --num-frames 121 --frame-rate 24 --seed "$SEED_OURS" \
    --prompt "$LTX_REF_PROMPT" --output-path "$c/out.mp4"
  rm -f "$c/reference.mov"
}

# Guided audio-to-video (docs/oracle.md "LTX-2.5 guided audio-to-video"):
# ltx_pipelines' A2VidPipelineTwoStage (`python -m ltx_pipelines.a2vid_two_stage`)
# at Lightricks/LTX-2 fd4ded7 as published: the dev transformer with the
# multimodal guider at stage 1 (the CLI defaults of a 2.5 checkpoint: 30 steps,
# CFG 3, STG 1 on block 28, modality 3, rescale 0.7, the default negative
# prompt), the distilled LoRA at 1.0 at stage 2. Weights: weights:ltx25-dev.
# oracle_ltx_a2v <name> <audio> <prompt> [extra args, e.g. --image PATH 0 1.0]
oracle_ltx_a2v() {
  local name="$1" audio="$2" prompt="$3" L="$UW/LTX-2.5" py="$UP/sol-ltx25/LTX-2/.venv/bin/python"
  shift 3
  if [[ -n "${UP_ORACLE:-}" && " $UP_ORACLE " != *" $name "* ]]; then return 0; fi
  local dit="$L/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
  local lora="$L/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
  [[ -f "$dit" && -f "$lora" ]] || { log "oracle $name: no dev DiT / distilled LoRA under $L (weights:ltx25-dev)"; return 1; }
  oracle_cell "$name" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "$py" -m ltx_pipelines.a2vid_two_stage \
    --transformer-path "$dit" \
    --text-encoder-path "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$L/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae-path "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --distilled-lora "$lora" 1.0 --audio-path "$audio" \
    --offload cpu --width 768 --height 512 --num-frames 121 --frame-rate 24 --seed "$SEED_OURS" \
    --prompt "$prompt" --output-path "$OUT/oracle-$name/out.mp4" "$@"
}

# Audio-to-video prompts (targets ltx25-a2v, ltx25-a2v-i2v, ltx25-a2v-guided;
# shared with runpod-matrix.sh's `oracle` family).
LTX_A2V_PROMPT="${FV_LTX_A2V_PROMPT:-A close-up of a woman with short dark hair talking directly to the camera in a bright living room, natural light, she speaks clearly and calmly, her lips moving with every word.}"
LTX_A2V_I2V_PROMPT="${FV_LTX_A2V_I2V_PROMPT:-A calm beach at golden hour, gentle waves rolling in, while a narrator speaks.}"

# Retake / extend (targets ltx25-retake*, ltx25-extend; shared with
# runpod-matrix.sh): the beach push-in fixture with the speech clip as its
# soundtrack (scripts/gpu/fixtures/beach-push-768x512-24fps.mp4, 121 frames).
LTX_EDIT_SOURCE_NAME=beach-push-768x512-24fps.mp4
LTX_RETAKE_PROMPT="${FV_LTX_RETAKE_PROMPT:-A huge wave crashes over the dark rocks at golden hour, white spray bursting high into the air, a narrator speaks calmly.}"
LTX_EXTEND_PROMPT="${FV_LTX_EXTEND_PROMPT:-The camera keeps pushing in slowly over the rocky beach at golden hour, waves rolling onto the sand, a narrator speaks calmly.}"

# Wan 2.2 TI2V-5B modules (Diffusers, oracle_wan22.py): VAE encode/decode
# of a fixed 704x1280 clip and one DiT forward per timestep layout (t2v, and
# i2v's frame-0-at-timestep-0), inputs dumped for `fv-gpucheck wan oracle`.
# Weights: fv-weights-h3-ltx-hy (EU; fv-weights-b200-us was deleted 2026-10).
oracle_wan22() {
  oracle_cell wan22-ti2v env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/oracle_wan22.py" \
    --model "$W/wan22-ti2v-5b" --image "$HERE/../fixtures/ti2v-beach-832x480.jpg"
}

# FastWan2.2 TI2V-5B FullAttn (FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers):
# the TI2V-5B network DMD-distilled, so the same Diffusers modules; the DiT
# forwards at the middle DMD timestep (1000/757/522). Weights: both volumes.
oracle_fastwan22() {
  oracle_cell fastwan22-ti2v env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/oracle_wan22.py" \
    --model "$W/fastwan22-ti2v-5b" --image "$HERE/../fixtures/ti2v-beach-832x480.jpg" --timestep 757
}

# FastVideo SF-Wan 1.3B (WanCausalDMDPipeline at its defaults), one request,
# no warm-up: the hooks dump the first denoise (oracle_dump.py _patch_sf_*).
oracle_sfwan() {
  oracle_cell sfwan13 env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/bench_fastwan.py" \
    --model "$W/sfwan21-1.3b" --hf-name SFWan2.1-T2V-1.3B-Diffusers --out "$OUT/oracle-sfwan13" \
    --repeats 1 --no-warmup --prompt "$PROMPT_OURS" --seed "$SEED_OURS" --attention FLASH_ATTN
}

# The Ref2VA reference prompt and image, shared with runpod-matrix.sh's
# `oracle` family (target h3-ref2va-*).
REF2VA_PROMPT="${FV_REF2VA_PROMPT:-The camera glides slowly forward along the shoreline of the beach in <Picture 1>, turquoise waves rolling in and breaking into white foam, bright sunny day, the sound of the surf and a light wind.}"
REF2VA_IMAGE="$HERE/../fixtures/ti2v-beach-832x480.jpg"

# MiniMax-H3 Ref2VA (docs/ports/h3-ref2v.md): the diffusers view of h3-base
# (weights_h3_diffusers) plus transformer_ref linked from the h3-ref2va tree
# (same LFS objects at every revision since bfc8ed0). Base engine config
# (text encoder and VAEs offloaded), FLASH_ATTN, `--steps <forwards+1>`.
oracle_ref2va() {
  local name="$1" forwards="$2" root="$UW/MiniMax-H3"
  # The baked image's fallback scripts (/opt/fvrs, used when the clone at the
  # run's sha fails) carry no fixtures: stop before loading 130 GB of weights.
  [[ -f "$REF2VA_IMAGE" ]] || { log "oracle $name: no reference image $REF2VA_IMAGE (clone failed?)"; return 1; }
  [[ -f "$root/model_index.json" && -d "$root/transformer" ]] || weights_h3_diffusers || return 1
  [[ -e "$root/transformer_ref" ]] || ln -s "$W/h3-ref2va/transformer_ref" "$root/transformer_ref"
  oracle_cell "$name" env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/oracle_fastvideo.py" \
    --fastvideo-src "$SRC/FastVideo" --recipe ref2va --reference "$REF2VA_IMAGE" \
    --out "$OUT/oracle-$name" --repeats 1 -- \
    --model-path "$root" --prompt "$REF2VA_PROMPT" --seed "$SEED_OURS" --num-gpus 1 \
    --vsa-kernel triton --no-fa4 --no-warmup --profile strict --no-inference-torch-compile --no-compile-vae \
    --steps "$((forwards + 1))" --height 768 --width 1344 --num-frames 124
}

run_oracle() {
  oracle_ref2va h3-ref2va-4step 4
  oracle_sfwan
  local f8="$UW/FastVideo-FastH3-8-Step-V2" lora="$W/FastH3-4-step-Preview-v1-LoRA"
  oracle_wan22
  oracle_fastwan22
  local g768=(--height 768 --width 1344 --num-frames 124)
  oracle_fv fasth3-8step 8step --model-path "$f8" "${g768[@]}"
  oracle_fv fasth3-4step-vsa lora --model-path "$UW/MiniMax-H3" \
    --lora-path "$lora/vsa-datafree/adapter_model.safetensors" "${g768[@]}"
  # Controls without VSA's top-k tile selection: the 8-step checkpoint at
  # sparsity 0 (every tile, gated compression branch kept; FastVideo will not
  # load its gates without VSA) and the dense-datafree LoRA (no gate at all).
  oracle_fv fasth3-8step-vsa0 8step --model-path "$f8" "${g768[@]}" --vsa-sparsity 0.0
  oracle_fv fasth3-4step-dense lora --model-path "$UW/MiniMax-H3" \
    --lora-path "$lora/dense-datafree/adapter_model.safetensors" "${g768[@]}"
  oracle_ltx ltx25-512p sol 512p
  oracle_ltx ltx25-512p-dense dense 512p
  oracle_ltx ltx25-4k sol 4k
  oracle_ltx ltx25-4k-dense dense 4k
  # Image conditioning (E5 / E9, docs/ports/ltx25.md "Image conditioning"):
  # first-frame I2V, and first + last frame (a keyframe at pixel frame 120).
  local fx="$HERE/../fixtures"
  oracle_ltx ltx25-i2v dense 512p --image "$fx/ti2v-beach-832x480.jpg" 0 1.0
  oracle_ltx ltx25-kf dense 512p --image "$fx/ti2v-beach-832x480.jpg" 0 1.0 \
    --image "$fx/ti2v-beach-zoom-832x480.jpg" 120 1.0
  oracle_ltx_ref ltx25-ref2v
  # Audio-to-video (docs/oracle.md "LTX-2.5 audio-to-video"): the distilled
  # two-stage with a2vid_two_stage.py's frozen driving audio (ltx25_a2v.py) on
  # a speech clip, prompt only (a talking head), then with the beach image as
  # the first frame. The prompt is shared with runpod-matrix.sh.
  oracle_ltx ltx25-a2v dense 512p --a2v "$fx/speech-flite-44k.flac" "$LTX_A2V_PROMPT"
  oracle_ltx ltx25-a2v-i2v dense 512p --a2v "$fx/speech-flite-44k.flac" "$LTX_A2V_I2V_PROMPT" \
    --image "$fx/ti2v-beach-832x480.jpg" 0 1.0
  # Retake / extend (docs/oracle.md "LTX-2.5 retake and extend"): one
  # distilled stage at the source size (ltx25_edit.py). Retake [1.5, 3.5) s of
  # both streams, of the video only (the audio frozen) and of the audio only
  # (the video frozen); extend 48 frames (2 s) after the source.
  local src="$fx/$LTX_EDIT_SOURCE_NAME"
  oracle_ltx ltx25-retake dense 512p --edit "retake:$src:1.5:3.5:av" "$LTX_RETAKE_PROMPT"
  oracle_ltx ltx25-retake-v dense 512p --edit "retake:$src:1.5:3.5:v" "$LTX_RETAKE_PROMPT"
  oracle_ltx ltx25-retake-a dense 512p --edit "retake:$src:1.5:3.5:a" "$LTX_RETAKE_PROMPT"
  oracle_ltx ltx25-extend dense 512p --edit "extend:$src:48:end" "$LTX_EXTEND_PROMPT" --num-frames 169
  # Guided audio-to-video on the dev transformer (A2VidPipelineTwoStage as
  # published), the talking head of ltx25-a2v.
  oracle_ltx_a2v ltx25-a2v-guided "$fx/speech-flite-44k.flac" "$LTX_A2V_PROMPT"
}
