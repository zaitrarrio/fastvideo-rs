#!/usr/bin/env bash
# Pod-side driver for the upstream (Python) reference benchmarks.
# Runs from the Runpod start command (runpod-http.sh upstream): the repo is
# cloned at the requested sha, this script installs/reuses the venvs on the
# volume, prepares weight views, and runs the requested cells. Every cell
# writes <run>/<cell>/{result.json,cell.json,stdout.log,stderr.log,*.mp4}.
#
#   UP_STEPS  space-separated, run in order, e.g.
#             "setup:fastvideo setup:sol-h3-rtx5090 setup:sol-h3-4step setup:sol-ltx25
#              weights:h3-fl2va weights:h3-diffusers weights:fasth3-8step info:ltx25 cells"
#   UP_STEPS_BG  steps run in the background alongside UP_STEPS
#   UP_CELLS  cells for the "cells" step (see cell_* below); default: all
#   UP_CELL_TIMEOUT_S  per-cell wall cap (default 5400)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: pod.sh <run-dir>}"
mkdir -p "$OUT"
export LIVE="$OUT/live.log"
UP="${UP:-/workspace/upstream}"
W="${W:-/workspace/weights}"
UW="$UP/weights"
# shellcheck source=scripts/gpu/upstream/setup.sh
. "$HERE/setup.sh"
PROMPT_OURS="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
SEED_OURS="${FV_SEED:-1024}"
H3_REV=bfc8ed0353f5a9733be73e6b2c98ec0948195b86        # sol-engine H3 configs' H3_MODEL_REVISION
F8_REV=3da2ddfe1954d9cda4c05b643dc0f26007a655c5        # FastVideo/FastVideo-FastH3-8-Step-V2
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_DISABLE_TELEMETRY=1

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LIVE" >&2; }
pyn() { uv run -q --no-project --python 3.12 --with numpy python "$@"; }

step_status() { printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%FT%TZ)" >>"$OUT/steps.tsv"; }

run_step() {
  local s="$1" t0 rc
  t0=$(date +%s)
  log "▶ step $s"
  case "$s" in
    setup:fastvideo) install_fastvideo ;;
    setup:sol-h3-rtx5090) install_sol_h3_rtx5090 ;;
    setup:sol-h3-4step) install_sol_h3_4step ;;
    setup:sol-ltx25) install_sol_ltx25 ;;
    weights:h3-fl2va) weights_h3_fl2va ;;
    weights:h3-diffusers) weights_h3_diffusers ;;
    weights:fasth3-8step) weights_fasth3_8step ;;
    weights:ltx25) weights_ltx25 ;;
    info:ltx25) info_ltx25 ;;
    info:box) info_box ;;
    cells) run_cells ;;
    *) log "unknown step $s"; false ;;
  esac >>"$OUT/steps.log" 2>&1
  rc=$?
  log "$( ((rc == 0)) && echo ok || echo FAIL) step $s  $(( $(date +%s) - t0 ))s"
  step_status "$s" "$rc"
  (( rc == 0 )) || tail -25 "$OUT/steps.log" | tee -a "$LIVE" >&2
  return 0
}

info_box() {
  {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "cpus=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
    df -h /workspace | tail -1
    du -sh "$UP"/* 2>/dev/null
    ls -la "$HF_HOME" 2>/dev/null | sed 's/token.*/token (present)/'
    for v in fastvideo sol-h3-rtx5090 sol-h3-4step; do
      [[ -x "$UP/$v/bin/python" ]] && echo "$v: $(cat "$UP/$v/.stamp" 2>/dev/null)"
    done
    [[ -f "$UP/sol-ltx25/.stamp" ]] && echo "sol-ltx25: $(cat "$UP/sol-ltx25/.stamp")"
  } >"$OUT/box.txt" 2>&1
}

# ---------------------------------------------------------------- weights
weights_h3_fl2va() {
  local root="$UW/MiniMax-H3" files=()
  for i in $(seq -w 1 13); do files+=("FL2VA/transformer/model-000$i-of-00013.safetensors"); done
  pyn "$HERE/reconstruct.py" --plan h3_fl2va_dit --src "$W/h3-base/transformer" \
    --repo MiniMaxAI/MiniMax-H3 --revision "$H3_REV" --files "${files[@]}" \
    --out-root "$root" --jobs 4 --report "$OUT/reconstruct-h3-dit.json" || return 1
  pyn "$HERE/reconstruct.py" --plan h3_fl2va_video_vae --src "$W/h3-base/vae" \
    --repo MiniMaxAI/MiniMax-H3 --revision "$H3_REV" --files FL2VA/video_vae/source/model.safetensors \
    --out-root "$root" --jobs 1 --report "$OUT/reconstruct-h3-vae.json" || return 1
  # Everything else under FL2VA: text encoder/tokenizer are the same LFS objects
  # as the root layout (symlinked); audio VAE (0.6 GB) and small files download.
  pyn "$HERE/overlay.py" --repo MiniMaxAI/MiniMax-H3 --rev "$H3_REV" --out "$root" --include FL2VA/ \
    --map "FL2VA/text_encoder=$W/h3-base/text_encoder" --map "FL2VA/tokenizer=$W/h3-base/tokenizer" \
    --max-download-mb 700 >"$OUT/overlay-h3-fl2va.json"
}

weights_h3_diffusers() {
  local inc=()
  for p in model_index.json modular_model_index.json scheduler/ audio_scheduler/ processor/ tokenizer/ \
    text_encoder/ transformer/ vae/ audio_vae/; do inc+=(--include "$p"); done
  pyn "$HERE/overlay.py" --repo MiniMaxAI/MiniMax-H3 --rev "$H3_REV" --out "$UW/MiniMax-H3" \
    --local "$W/h3-base" "${inc[@]}" --max-download-mb 50 >"$OUT/overlay-h3-diffusers.json"
}

weights_fasth3_8step() {
  pyn "$HERE/overlay.py" --repo FastVideo/FastVideo-FastH3-8-Step-V2 --rev "$F8_REV" \
    --out "$UW/FastVideo-FastH3-8-Step-V2" --local "$W/h3-8step" --max-download-mb 50 \
    >"$OUT/overlay-fasth3-8step.json"
}

# Lightricks/LTX-2.5 single-file packs the sol-engine RTX5090 driver loads,
# rebuilt byte-exactly from the Diffusers copy (needs an HF token with access to
# the gated repo, for headers and the few remote tensors only).
LTX_REV=5e6e71018ee1756ed329b697a7b4aedc934dfce9
weights_ltx25() {
  local L="$W/ltx25" o="$UW/LTX-2.5" rc=0
  pyn "$HERE/reconstruct.py" --plan ltx25_upsampler --src "$L/latent_upsampler" --repo Lightricks/LTX-2.5 \
    --revision "$LTX_REV" --files latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --out-root "$o" --report "$OUT/reconstruct-ltx-up.json" || rc=1
  pyn "$HERE/reconstruct.py" --plan ltx25_video_vae --src "$L/vae" --repo Lightricks/LTX-2.5 \
    --revision "$LTX_REV" --files vae/ltx-2.5-video-vae-conv-bf16.safetensors \
    --out-root "$o" --report "$OUT/reconstruct-ltx-vae.json" || rc=1
  pyn "$HERE/reconstruct.py" --plan ltx25_audio_vae --src "A=$L/audio_vae" --src "V=$L/vocoder" \
    --repo Lightricks/LTX-2.5 --revision "$LTX_REV" --files vae/ltx-2.5-audio-vae-bf16.safetensors \
    --out-root "$o" --report "$OUT/reconstruct-ltx-avae.json" || rc=1
  pyn "$HERE/reconstruct.py" --plan ltx25_text_encoder --src "G=$L/text_encoder" --src "C=$L/connectors" \
    --repo Lightricks/LTX-2.5 --revision "$LTX_REV" --files text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --out-root "$o" --report "$OUT/reconstruct-ltx-te.json" || rc=1
  pyn "$HERE/reconstruct.py" --plan ltx25_transformer --src "T=$L/transformer" --src "C=$L/connectors" \
    --repo Lightricks/LTX-2.5 --revision "$LTX_REV" --files diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --out-root "$o" --report "$OUT/reconstruct-ltx-dit.json" || rc=1
  return $rc
}

# Headers only (keys, dtypes, shapes, metadata) — to plan the LTX-2.5
# single-file reconstruction; no weights are downloaded here.
info_ltx25() {
  mkdir -p "$OUT/ltx25-headers"
  pyn - "$OUT/ltx25-headers" "$W/ltx25" "$HERE" <<'EOF'
import json, sys, struct, pathlib
sys.path.insert(0, sys.argv[3])
import reconstruct as R
out = pathlib.Path(sys.argv[1]); local = pathlib.Path(sys.argv[2])
print("token:", "yes" if R.hf_token() else "no")
files = ["diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
         "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
         "vae/ltx-2.5-video-vae-conv-bf16.safetensors", "vae/ltx-2.5-audio-vae-bf16.safetensors",
         "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"]
for f in files:
    try:
        raw, h = R.remote_header("Lightricks/LTX-2.5", "main", f)
        oid, size = R.lfs_oid("Lightricks/LTX-2.5", "main", f)
        h["__lfs__"] = {"oid": oid, "size": size, "hlen": len(raw)}
        (out / (f.replace("/", "__") + ".json")).write_text(json.dumps(h))
        print("ok", f, len(h))
    except Exception as e:
        print("FAIL", f, e)
for p in sorted(local.rglob("*.safetensors")):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    (out / ("local__" + str(p.relative_to(local)).replace("/", "__") + ".json")).write_text(json.dumps(h))
for p in sorted(local.rglob("*.json")):
    if p.stat().st_size < 2_000_000:
        (out / ("localcfg__" + str(p.relative_to(local)).replace("/", "__"))).write_text(p.read_text())
EOF
}

# ------------------------------------------------------------------ cells
smi_start() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >"$1/smi.log" 2>/dev/null & echo $!; }
smi_peak() { sort -n "$1/smi.log" 2>/dev/null | tail -1; }

gpu_clean() {
  local used i
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    [[ -n "$used" && "$used" -lt 1024 ]] && return 0
    (( i == 10 )) && nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 2
  done
  return 1
}

# run_cell <name> <cmd...>: timeout, stdout/stderr, nvidia-smi peak, cell.json.
run_cell() {
  local name="$1"; shift
  local c="$OUT/$name" t0 rc smi
  if [[ -n "${UP_CELLS:-}" && " $UP_CELLS " != *" $name "* ]]; then return 0; fi
  mkdir -p "$c"
  gpu_clean || log "WARN GPU not clean before $name"
  log "▶ cell $name"
  t0=$(date +%s)
  smi="$(smi_start "$c")"
  (cd "$c" && timeout --signal=TERM --kill-after=30 "${UP_CELL_TIMEOUT_S:-5400}" "$@") >"$c/stdout.log" 2>"$c/stderr.log"
  rc=$?
  kill "$smi" 2>/dev/null
  local secs=$(( $(date +%s) - t0 )) peak
  peak="$(smi_peak "$c")"
  printf '{"cell":"%s","exit":%s,"seconds":%s,"peak_smi_mib":%s,"started":%s}\n' \
    "$name" "$rc" "$secs" "${peak:-null}" "$t0" >"$c/cell.json"
  if (( rc == 0 )); then log "ok $name ${secs}s peak ${peak}MiB"; else
    log "FAIL $name ${secs}s exit=$rc"; tail -15 "$c/stderr.log" | tee -a "$LIVE" >&2; fi
}

# sol-engine MiniMax-H3 RTX5090 profile: env from config/minimax_h3/rtx5090_<arm>.toml,
# then the reference's own run_minimax_h3_gpu.sh (which writes benchmark.json).
cell_sol_h3_rtx() {
  local arm="$1" prompt="$2" name="sol-h3r5090-$1-$2" envs
  envs="$(python3 - "$SRC/sol-engine/config/minimax_h3/rtx5090_${arm}.toml" <<'EOF'
import sys, tomllib, shlex
env = tomllib.load(open(sys.argv[1], "rb"))["env"]
print(" ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()))
EOF
)"
  local extra=(H3_MODEL_PATH="$UW/MiniMax-H3" H3_MODEL_SUBFOLDER=FL2VA HF_HUB_OFFLINE=1
    PYTHON_BIN="$UP/sol-h3-rtx5090/bin/python" H3_ROOT="$UP/cache/h3-rtx5090")
  if [[ "$prompt" == ours ]]; then
    extra+=(H3_PROMPT="$PROMPT_OURS" H3_SEED="$SEED_OURS")
  fi
  mkdir -p "$UP/cache/h3-rtx5090"
  # shellcheck disable=SC2086
  eval "run_cell $name env $envs \"\${extra[@]}\" OUT_DIR=\"$OUT/$name\" \
    bash \"$SRC/sol-engine/models/minimax_h3/RTX5090/run_minimax_h3_gpu.sh\""
}

cell_sol_h3_4step() {
  run_cell sol-h3-4step-ours "$UP/sol-h3-4step/bin/python" "$HERE/bench_sol_h3_4step.py" \
    --sol-h3 "$SRC/sol-engine/models/minimax_h3/Sol-H3" --model "$UW/MiniMax-H3" \
    --adapter "$W/FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors" \
    --prompt "$PROMPT_OURS" --seed "$SEED_OURS" --repeats 3 --te-offload --out "$OUT/sol-h3-4step-ours"
}

cell_ltx25() {
  local wl="$1" arm="$2" prompt="$3" name="sol-ltx25-$1-$2-$3" geo seed p
  case "$wl" in
    4k5s) geo=(--width 3840 --height 2176 --num-frames 121) ;;
    1080p20s) geo=(--width 1920 --height 1088 --num-frames 481) ;;
  esac
  local L="$UW/LTX-2.5"
  if [[ "$prompt" == ours ]]; then p="$PROMPT_OURS"; seed="$SEED_OURS"; else
    p="$(cat "$SRC/sol-engine/models/ltx25/prompts/p01_multishot.txt")"; seed=42; fi
  run_cell "$name" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    "$UP/sol-ltx25/LTX-2/.venv/bin/python" "$HERE/bench_ltx25.py" --sol-engine "$SRC/sol-engine" \
    --arm "$arm" --result "$OUT/$name/result.json" -- \
    --pipeline bf16 --metrics "$OUT/$name/benchmark.json" -- \
    --transformer-path "$L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" \
    --text-encoder-path "$L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --video-vae-path "$L/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae-path "$L/vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --spatial-upsampler-path "$L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --offload cpu "${geo[@]}" --frame-rate 24 --seed "$seed" --prompt "$p" --output-path "$OUT/$name/out.mp4"
}

# FastVideo upstream. One GPU => --num-gpus 1; sm_120 => Triton VSA kernel.
cell_fv() {
  local name="$1" recipe="$2"; shift 2
  local fa4=(--no-fa4)
  [[ "${UP_FV_FA4:-0}" == 1 ]] && fa4=(--fa4)
  run_cell "$name" env PYTHONUNBUFFERED=1 "$UP/fastvideo/bin/python" "$HERE/bench_fastvideo.py" \
    --fastvideo-src "$SRC/FastVideo" --recipe "$recipe" --out "$OUT/$name" --repeats "${UP_FV_REPEATS:-3}" -- \
    --prompt "$PROMPT_OURS" --seed "$SEED_OURS" --num-gpus 1 --vsa-kernel triton "${fa4[@]}" "$@"
}

run_cells() {
  local g768=(--height 768 --width 1344 --num-frames 124) g480=(--height 480 --width 832 --num-frames 124)
  local f8="$UW/FastVideo-FastH3-8-Step-V2" lora="$W/FastH3-4-step-Preview-v1-LoRA"
  # FastVideo (short cells first)
  cell_fv fv-fasth3-8step-768p 8step --model-path "$f8" "${g768[@]}"
  cell_fv fv-fasth3-8step-480p 8step --model-path "$f8" "${g480[@]}"
  cell_fv fv-fasth3-4step-vsa-768p lora --model-path "$UW/MiniMax-H3" --lora-path "$lora/vsa-datafree/adapter_model.safetensors" "${g768[@]}"
  cell_fv fv-fasth3-4step-vsa-480p lora --model-path "$UW/MiniMax-H3" --lora-path "$lora/vsa-datafree/adapter_model.safetensors" "${g480[@]}"
  cell_fv fv-fasth3-4step-dense-768p lora --model-path "$UW/MiniMax-H3" --lora-path "$lora/dense-datafree/adapter_model.safetensors" "${g768[@]}"
  cell_fv fv-h3-base-768p base --model-path "$UW/MiniMax-H3" --steps 50 --profile strict \
    --no-inference-torch-compile "${g768[@]}"
  # sol-engine
  cell_sol_h3_4step
  local arm
  for arm in fullopt sol dense; do cell_sol_h3_rtx "$arm" ours; done
  for arm in fullopt sol dense; do cell_sol_h3_rtx "$arm" demo; done
  for wl in 4k5s 1080p20s; do for arm in sol dense; do cell_ltx25 "$wl" "$arm" ours; done; done
}

log "pod.sh start steps: ${UP_STEPS:-}"
info_box
ensure_base >>"$OUT/steps.log" 2>&1 || log "WARN ensure_base failed"
# UP_STEPS_BG run concurrently with UP_STEPS (e.g. weight conversion while
# venvs install); both are awaited before the final info.
bg=""
if [[ -n "${UP_STEPS_BG:-}" ]]; then
  ( for s in $UP_STEPS_BG; do run_step "$s"; done ) &
  bg=$!
fi
for s in ${UP_STEPS:-info:box}; do run_step "$s"; done
[[ -n "$bg" ]] && wait "$bg"
info_box
log "pod.sh done"
