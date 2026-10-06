#!/usr/bin/env bash
# Runs ON a Runpod pod (runpod-http.sh with FV_POD_SCRIPT=sol-bench-pod.sh):
# our stack against NVlabs sol-engine's published numbers
# (docs/perf/sol-bench.md). Prebuilt image binary only: no compile, no
# downloads; weights are read from the EU volume mounted at /workspace.
#
#   FV_SOL_SET   a1 | a2 | b1 | b2 ...   which cell list (see the case below)
#   FV_SOL_BUDGET_S   seconds after container start by which every cell must
#                     have ended (default 3300 = 55 min); a cell whose
#                     estimate does not fit is recorded as skipped
#   FV_SOL_CAP_S      pod self-delete after this many seconds of container
#                     uptime (default 3840 = 64 min), independent of the
#                     driver's own 65-min backstop
#
# Every cell writes <cell>/summary.json (exit, seconds) and the stage's
# benchmark.json; clips land as <cell>/frames/output.mp4.
set -uo pipefail
FAMILY="${1:-sol-bench}"
SET="${FV_SOL_SET:?FV_SOL_SET}"
WORK="${FV_WORK:-/workspace}"
W="$WORK/weights"
SCRATCH="${FV_SCRATCH:-/fvscratch}"
RUNS="$SCRATCH/runs/$FAMILY/${FV_RUN_TAG:?FV_RUN_TAG}"
LOG="$RUNS/live.log"
BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
VERIFY=/opt/fastvideo-rs/scripts/gpu/verify-weights.sh
PROMPT="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
SEED="${FV_SEED:-1024}"
BUDGET_S="${FV_SOL_BUDGET_S:-3300}"
CAP_S="${FV_SOL_CAP_S:-3840}"
mkdir -p "$RUNS" "$SCRATCH/fv-libs"

# Container start: /proc/1's timestamp (falls back to this script's start).
T0="$(stat -c %Y /proc/1 2>/dev/null || date +%s)"
uptime_s() { echo $(( $(date +%s) - T0 )); }
log() { printf '[%s] [%s +%ss] %s\n' "$(date -u +%H:%M:%S)" "$SET" "$(uptime_s)" "$*" | tee -a "$LOG"; }

# Pod-side backstop: the pod deletes itself (pod-scoped key injected by Runpod).
if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
  ( sleep $(( CAP_S - $(uptime_s) )); curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null 2>&1 ) >/dev/null 2>&1 &
fi

# CUDA libraries under the unversioned names cudarc loads (as runpod-matrix.sh).
for pins in /opt/fastvideo-rs/scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins; do
  [[ -f $pins ]] && { . "$pins"; break; }
done
for spec in \
  "cudnn:/lib/x86_64-linux-gnu/${CUDA_CUDNN_SONAME:-}" \
  "cublas:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLAS_SONAME:-}" \
  "cublasLt:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLASLT_SONAME:-}" \
  "nvrtc:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_NVRTC_SONAME:-}"; do
  name="${spec%%:*}" src="${spec#*:}"
  [[ -f "$src" ]] && ln -sf "$(readlink -f "$src")" "$SCRATCH/fv-libs/lib${name}.so"
done
export LD_LIBRARY_PATH="$SCRATCH/fv-libs:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/fastvideo-rs/target/release:$PATH"
[[ -e $SCRATCH/fv-libs/libcudnn.so ]] || log "WARN: libcudnn.so not linked"
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "build_id=$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null)"
  echo "cpus=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
} >"$RUNS/sysinfo.txt" 2>&1

gpu_clean() {
  local used i
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    [[ -n "$used" && "$used" -lt 1024 ]] && return 0
    sleep 2
  done
  return 1
}

# cell <name> <weight cell> <estimate s> <cmd...>
cell() {
  local name="$1" wcell="$2" est="$3" dir="$RUNS/$1" rc t0 secs left
  shift 3
  mkdir -p "$dir"
  left=$(( BUDGET_S - $(uptime_s) ))
  if (( est > left )); then
    log "SKIP $name: estimate ${est}s > ${left}s left in budget"
    printf '{"cell":"%s","exit":null,"skipped":"budget","estimate_s":%s,"left_s":%s}\n' "$name" "$est" "$left" >"$dir/summary.json"
    return 0
  fi
  if ! FV_WEIGHTS="$W" bash "$VERIFY" "$wcell" >"$dir/weights.log" 2>&1; then
    log "SKIP $name: weights incomplete ($wcell)"
    printf '{"cell":"%s","exit":null,"skipped":"weights incomplete"}\n' "$name" >"$dir/summary.json"
    return 0
  fi
  gpu_clean || log "WARN: GPU not clean before $name"
  log "▶ $name (est ${est}s, hard stop ${left}s): $*"
  t0=$(date +%s)
  (cd "$dir" && timeout --signal=TERM --kill-after=15 "$left" "$@") >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  log "$([[ $rc == 0 ]] && echo ok || echo FAIL) $name ${secs}s exit=$rc"
  [[ $rc == 0 ]] || tail -15 "$dir/stderr.log" | tee -a "$LOG"
  printf '{"cell":"%s","exit":%s,"seconds":%s,"estimate_s":%s,"ended":"%s"}\n' \
    "$name" "$rc" "$secs" "$est" "$(date -u +%FT%TZ)" >"$dir/summary.json"
  # Keep the mp4 and reports; frame PNGs are not fetched (FV_FETCH_SKIP).
  return 0
}

# ---- cell definitions -------------------------------------------------------
# MiniMax-H3 1344x768, 124 frames, 50 steps: sol-engine config/minimax_h3/
# rtx5090_{dense,fullopt}.toml (our profiles/h3/rtx5090_*.toml are equal to them).
h3_common=(--prompt "$PROMPT" --seconds 5 --seed "$SEED" --text-encoder auto
  --text-cache "$SCRATCH/h3-text-cache" --text-weights "$W/h3-base")
h3() { # <name> <est> <warm 0|1> <env...> -- (the sol-h3-rtx recipe)
  local name="$1" est="$2" warm="$3"; shift 3
  local wf=(); [[ $warm == 1 ]] && wf=(--warm)
  cell "$name" h3-base "$est" env "$@" \
    "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx \
      --adaln-cache "$RUNS/h3-768p-adaln.cache" --clip-dir "$RUNS/$name/frames" "${h3_common[@]}" "${wf[@]}" \
      ${H3_EXTRA[@]+"${H3_EXTRA[@]}"}
}
H3_EXTRA=()
# LTX-2.5 distilled two-stage at sol-engine's RTX 5090 workloads
# (models/ltx25/RTX5090: 4k5s, 1080p20s), Sol stage 2, BF16 or NVFP4 video FFN.
ltx() { # <name> <est> <workload> <profile>
  cell "$1" ltx25-two-stage "$2" \
    "$BIN" --techniques "$4" --mode fast ltx2 gen --model-version 2.5 \
      --weights "$W/ltx25" --dit "$W/ltx25" --workload "$3" \
      --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
      --clip "$RUNS/$1/frames"
}
# Wan2.2 TI2V-5B: sol-engine models/wan22_ti2v_5b (704x1280, 121 f, 50 steps,
# CFG 5, shift 5); opt = config/wan22_ti2v_5b/wan5b_kernel_easycache_pisa.toml
# (EasyCache 0.036 / retain 7 / cooldown 1 = our "5b" profile, PISA 5B route).
wan_neg_cn="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
wan5b() { # <name> <est> <env...>
  local name="$1" est="$2"; shift 2
  cell "$name" wan22-ti2v-5b "$est" env "$@" \
    "$BIN" --mode fast wan gen --weights "$W/wan22-ti2v-5b" --preset wan_2_2_ti2v_5b --unipc --steps 50 \
      --guidance 5.0 --flow-shift 5.0 --fps 24 --negative "$wan_neg_cn" --seed "$SEED" --warm \
      --height 704 --width 1280 --num-frames 121 --prompt "$PROMPT" --clip-dir "$RUNS/$name/frames"
}
# Wan2.1 T2V-14B: sol-engine models/wan21_t2v_14b.toml (1280x720, 81 f, 50
# steps, CFG 5, shift 5); fullstack = config/wan21_t2v_14b/fullstack.toml
# (EasyCache 0.036 + Sol-Attn tau 1.0, 10 dense steps, layer 0 dense). One
# request after load (no --warm: a warm-up request would double a ~30-min cell);
# load is reported separately in benchmark.json either way.
wan14() { # <name> <est> <steps> <env...>
  local name="$1" est="$2" steps="$3"; shift 3
  cell "$name" wan21-t2v-14b "$est" env "$@" \
    "$BIN" --mode fast wan gen --weights "$W/wan21-t2v-14b" --preset wan_t2v_14b --unipc --steps "$steps" \
      --guidance 5.0 --flow-shift 5.0 --height 720 --width 1280 --num-frames 81 \
      --prompt "$PROMPT" --seed "$SEED" --clip-dir "$RUNS/$name/frames"
}

# ---- phase B (models merged 2026-10-06: #29 SANA, #30 LingBot/Cosmos3, #31
# Wan 1.3B / A14B / LTX-2.3). Arguments mirror runpod-matrix.sh's solbench,
# sana-video, sol-lingbot and sol-cosmos3 families on main (0f7edc8).
SOL_IN=/opt/fastvideo-rs/scripts/gpu/sol
sol_wan_prompts=/opt/fastvideo-rs/scripts/gpu/prompts-sol-wan-t2v5.json
sol_p0="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
# Wan reference arms: Diffusers' UniPC sigmas, bf16 GEMMs, the full Wan VAE
# (pins FASTVIDEO_WAN_QUANT=off: sm_100 would default to MXFP8).
wan_ref=(FASTVIDEO_WAN_UNIPC_SIGMAS=diffusers FASTVIDEO_WAN_QUANT=off FASTVIDEO_WAN_VAE=full)
# SANA-Video 2B (sol-engine sana_video 2B line): 832x480, 81 f, 50 steps,
# cfg 6, warm; arms baseline | full (EasyCache 0.1 + QKV merge + bf16 linear attn).
sana() { # <arm> <est>
  cell "sana-$1" sana-video-2b-480p "$2" \
    "$BIN" --mode fast sana-video gen --weights "$W/sana-video-2b-480p" \
      --prompt "a corgi running on the beach" --arm "$1" --seed 42 --warm --clip "$RUNS/sana-$1/frames"
}
# Wan2.1-T2V-1.3B (models/wan21_t2v_1_3b.toml): 832x480, 81 f @ 16 fps, 50
# UniPC steps, CFG 6, shift 3, 5 prompts after one warm generation.
wan13() { # <name> <est> <env...>
  local name="$1" est="$2"; shift 2
  cell "$name" wan21-t2v-1.3b "$est" env "${wan_ref[@]}" "$@" \
    "$BIN" --mode fast wan gen --weights "$W/wan21-t2v-1.3b" --preset wan_t2v_1_3b --unipc --steps 50 \
      --guidance 6.0 --flow-shift 3.0 --fps 16 --height 480 --width 832 --num-frames 81 --seed 1024 \
      --negative "$wan_neg_cn" --warm --prompts "$sol_wan_prompts" --clip-dir "$RUNS/$name/frames"
}
# Wan2.2-T2V-A14B (models/wan22_t2v_a14b.toml): 1280x720, 81 f, 40 steps,
# shift 12, CFG 4 / 3, boundary 0.875; one prompt, one request after load;
# experts swapped at the boundary on a 96 GB card (FASTVIDEO_WAN_MOE=auto).
a14b() { # <name> <est> <env...>
  local name="$1" est="$2"; shift 2
  cell "$name" wan22-t2v-a14b "$est" env "${wan_ref[@]}" FASTVIDEO_WAN_MOE=auto "$@" \
    "$BIN" --mode fast wan gen --weights "$W/wan22-t2v-a14b" --preset wan_2_2_t2v_a14b --unipc --steps 40 \
      --guidance 4.0 --guidance-2 3.0 --flow-shift 12.0 --fps 16 --height 720 --width 1280 --num-frames 81 \
      --seed 1024 --negative "$wan_neg_cn" --prompt "$sol_p0" --clip-dir "$RUNS/$name/frames"
}
# LTX-2.3 HQ (models/ltx23.toml): dev DiT + distilled LoRA 0.25 / 0.5,
# 1920x1088, 241 f, 15-step res2s stage 1 + 3-sigma stage 2, seed 42, warm.
ltx23_prompt="A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail"
ltx23() { # <name> <est> <stage-2 flag> <env...>
  local name="$1" est="$2" s2="$3"; shift 3
  cell "$name" ltx23-hq "$est" env \
    FASTVIDEO_LTX2_UPSAMPLER="$W/ltx23-dev/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$@" \
    "$BIN" --mode fast ltx2 gen --model-version 2.3 --hq --weights "$W/ltx23" \
      --dit "$W/ltx23-dev/ltx-2.3-22b-dev.safetensors" --prompt "$ltx23_prompt" --seed 42 \
      --text streamed --dit-offload resident --warm "$s2" --clip "$RUNS/$name/frames"
}
# LingBot-Video MoE (models/lingbot_video.toml): base 832x480x121 / 40 steps +
# 1080p refiner / 8 steps, CFG 3, seed 42; one prompt (val3 #0), base and
# refiner swapped on a 96 GB card.
lingbot() { # <arm> <est>
  cell "lingbot-$1" lingbot-moe "$2" \
    "$BIN" --mode fast sol lingbot-gen --weights "$W/lingbot-video-moe-30b-a3b" \
      --prompts "$SOL_IN/lingbot-t2v-val3.txt" --num-prompts 1 --arm "$1" --residency swap --seed 42 \
      --clip "$RUNS/lingbot-$1/clips"
}
# Cosmos3-Super (models/cosmos3.toml): 1280x720x189, 35 steps, CFG 6, seed 42,
# one 1-step warm-up request (WARMUP=true), then the timed request.
cosmos3() { # <name> <arm> <est> <env...>
  local name="$1" arm="$2" est="$3"; shift 3
  cell "$name" cosmos3-super "$est" env "$@" \
    "$BIN" --mode fast sol cosmos3-gen --weights "$W/cosmos3-super" --prompt "$(cat "$SOL_IN/cosmos3-default.txt")" \
      --negative-prompt "$(cat "$SOL_IN/cosmos3-negative.txt")" --seed 42 --warm --arm "$arm" \
      --clip "$RUNS/$name/clip"
}

log "set $SET budget ${BUDGET_S}s cap ${CAP_S}s $(head -1 "$RUNS/sysinfo.txt")"
case "$SET" in
  a1) # Phase A, instance 1 (RTX 5090 cells; PRO 6000 when no 5090 is free)
    h3 h3-768p-fullopt 720 1 FASTVIDEO_H3_SOL_CACHE=teacache
    ltx ltx25-4k5s-sol-bf16 480 4k5s ltx2/ltx25_distill_sol
    ltx ltx25-1080p20s-sol-bf16 420 1080p20s ltx2/ltx25_distill_sol
    ltx ltx25-4k5s-sol-nvfp4 480 4k5s ltx2/ltx25_distill_sol_nvfp4
    ltx ltx25-1080p20s-sol-nvfp4 420 1080p20s ltx2/ltx25_distill_sol_nvfp4
    # Dense last (owner priority); one request after load, load excluded.
    h3 h3-768p-dense 690 0 FASTVIDEO_H3_SOL_ATTN=off
    ;;
  a3) # Phase A, instance 1 follow-up: H3 with BF16 linears as sol-engine's
    # rtx5090_{dense,fullopt}.toml (our H3 default on sm_100+ is MXFP8 since
    # c054278), and the prompt encoded in every request (no text cache), so E2E
    # includes text encoding as theirs does.
    H3_EXTRA=(--no-text-cache)
    h3 h3-768p-fullopt-bf16 480 1 FASTVIDEO_H3_QUANT=off FASTVIDEO_H3_SOL_CACHE=teacache
    h3 h3-768p-dense-bf16 760 0 FASTVIDEO_H3_QUANT=off FASTVIDEO_H3_SOL_ATTN=off
    ;;
  a2) # Phase A, instance 2 (RTX PRO 6000)
    wan5b wan5b-base 480 -u FASTVIDEO_WAN_SOL_CACHE
    wan5b wan5b-easycache 330 FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=5b
    wan14 wan14-720p-fullstack 900 50 FASTVIDEO_WAN_SOL_ATTN=1 FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack
    # Base at 15 of the 50 steps: every base step costs the same (CFG, no
    # cache), so the 50-step denoise is 50/15 of this one (docs/perf/sol-bench.md).
    wan14 wan14-720p-base-s15 900 15 -u FASTVIDEO_WAN_SOL_CACHE
    wan5b wan5b-opt 330 FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=5b FASTVIDEO_WAN_PISA=1
    ;;
  b1) # Phase B: SANA-Video 2B + Wan2.1 1.3B
    sana baseline 900
    sana full 720
    wan13 wan13-sol-base 720
    wan13 wan13-sol-fullstack 480 FASTVIDEO_WAN_SOL_ATTN=fullstack FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack
    ;;
  b2) # Phase B: Wan2.2 T2V-A14B (config/wan22_t2v_a14b/singlegpu_opt.toml: EasyCache 0.30 + PISA 0.10)
    a14b a14b-sol-base 1800
    a14b a14b-sol-fullopt 900 FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_PISA=1
    ;;
  b3) # Phase B: LTX-2.3 HQ, then LingBot fullopt (one prompt)
    ltx23 ltx23-hq-base 660 --dense-stage2
    ltx23 ltx23-hq-fullopt 600 --pisa-stage2 FASTVIDEO_LTX2_STAGE1_CACHE=1 FASTVIDEO_LTX2_MIDPOINT_PRUNE=1 FASTVIDEO_NVFP4=1
    cell lingbot-router lingbot-moe 60 "$BIN" --mode fast sol lingbot-router
    lingbot fullopt 1200
    ;;
  b4) # Phase B: LingBot baseline (one prompt)
    lingbot baseline 2400
    ;;
  b5) # Phase B: Cosmos3-Super baseline
    cosmos3 cosmos3-baseline baseline 2400
    ;;
  b6) # Phase B: Cosmos3-Super TeaCache 1.15/10/3 (BF16), then + W8A8 FP8 (theirs: NVFP4 middle steps)
    cosmos3 cosmos3-teacache teacache 1800
    cosmos3 cosmos3-teacache-fp8 teacache 1200 FASTVIDEO_FP8=1
    ;;
  *) log "unknown set $SET"; exit 2 ;;
esac
log "set $SET done"
