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
# Targets: fasth3-8step (FastVideo FastH3 8-step V2, 768x1344x124),
# fasth3-4step-vsa (MiniMax-H3 + Preview v1 vsa-datafree LoRA), the dense
# controls fasth3-8step-dense (the 8-step checkpoint on FLASH_ATTN, no gate)
# and fasth3-4step-dense (dense-datafree LoRA), and the
# sol-engine LTX-2.5 distilled two-stage: ltx25-512p / ltx25-4k with Sol
# stage 2, ltx25-512p-dense / ltx25-4k-dense with dense stage 2.
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

# sol-engine LTX-2.5 two-stage; arm sol|dense (bench_ltx25.py), geometry per workload.
oracle_ltx() {
  local name="$1" arm="$2" wl="$3" geo L="$UW/LTX-2.5"
  case "$wl" in
    512p) geo=(--width 768 --height 512 --num-frames 121) ;;
    4k) geo=(--width 3840 --height 2176 --num-frames 121) ;;
  esac
  oracle_cell "$name" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "$UP/sol-ltx25/LTX-2/.venv/bin/python" "$HERE/bench_ltx25.py" --sol-engine "$SRC/sol-engine" \
    --arm "$arm" --result "$OUT/oracle-$name/result.json" -- \
    --pipeline bf16 --metrics "$OUT/oracle-$name/benchmark.json" -- \
    --transformer-path "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$L/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae-path "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --offload cpu "${geo[@]}" --frame-rate 24 --seed "$SEED_OURS" --prompt "$PROMPT_OURS" \
    --output-path "$OUT/oracle-$name/out.mp4"
}

run_oracle() {
  local f8="$UW/FastVideo-FastH3-8-Step-V2" lora="$W/FastH3-4-step-Preview-v1-LoRA"
  local g768=(--height 768 --width 1344 --num-frames 124)
  oracle_fv fasth3-8step 8step --model-path "$f8" "${g768[@]}"
  oracle_fv fasth3-4step-vsa lora --model-path "$UW/MiniMax-H3" \
    --lora-path "$lora/vsa-datafree/adapter_model.safetensors" "${g768[@]}"
  # Dense controls: the same denoisers without VSA's top-k tile selection.
  oracle_fv fasth3-8step-dense 8step --dense --model-path "$f8" "${g768[@]}"
  oracle_fv fasth3-4step-dense lora --model-path "$UW/MiniMax-H3" \
    --lora-path "$lora/dense-datafree/adapter_model.safetensors" "${g768[@]}"
  oracle_ltx ltx25-512p sol 512p
  oracle_ltx ltx25-512p-dense dense 512p
  oracle_ltx ltx25-4k sol 4k
  oracle_ltx ltx25-4k-dense dense 4k
}
