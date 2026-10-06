#!/bin/bash
# Raw generation-time benchmark on one pod (docs/perf/raw-inference.md).
# Weights are read from /workspace/weights (network volume, never written);
# everything this writes goes under /root/work on the container disk.
# Usage: bench.sh [job ...]   (default: every job)
set -u
B=/opt/fastvideo-rs/target/release/fv-gpucheck
W=/workspace/weights
TAE=$W/auxiliary/tae
WK=/root/work
RES=$WK/res
mkdir -p "$RES" "$WK/cache"
export FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache RUST_LOG=info
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$CC" in
  12.*) WQ=mxfp8; LTXMAX=ltx2/ltx25_distill_sol_nvfp4 ;;
  10.*) WQ=mxfp8; LTXMAX=ltx2/ltx25_distill_sol_nvfp4 ;;
  *)    WQ=w8a8;  LTXMAX=ltx2/ltx25_distill_sol_fp8 ;;
esac
PROMPT="A red fox trotting through fresh snow in a birch forest at golden hour, cinematic lighting, shallow depth of field"
P=$WK/prompts.json
python3 - "$P" "$PROMPT" <<'EOF'
import json, sys
json.dump({"name": "raw5", "prompts": [{"name": f"r{i}", "prompt": sys.argv[2], "seed": 7} for i in range(1, 6)]}, open(sys.argv[1], "w"))
EOF

run() {  # run <job> <env...> -- <args...>
  local job=$1; shift
  local J=$RES/$job envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  rm -rf "$J"; mkdir -p "$J"
  echo "=== $(date -u +%T) $job start" >> "$WK/progress.log"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$J/smi.csv" 2>/dev/null &
  local smi=$!
  local t0=$(date +%s)
  ( cd "$J" && env "${envs[@]}" timeout 1200 "$B" --out "$J/out" "$@" ) > "$J/log.txt" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null
  echo "=== $(date -u +%T) $job rc=$rc wall=$(( $(date +%s) - t0 ))s" >> "$WK/progress.log"
  # Clips are not kept (no quality testing): drop PNG/WAV to save disk.
  find "$J" \( -name '*.png' -o -name '*.wav' -o -name '*.mp4' -o -name '*.jpg' \) -delete 2>/dev/null
}

h3common=(h3 gen --weights $W/h3-base --prompt "$PROMPT" --prompts $P --height 480 --width 832 --seconds 5
  --warm --no-mp4 --text-encoder resident-fp8 --dit-offload resident --text-cache $WK/h3-text)
ltxcommon=(ltx2 gen --model-version 2.5 --weights $W/ltx25 --dit $W/ltx25 --two-stage --prompt "$PROMPT" --prompts $P
  --height 768 --width 1280 --num-frames 121 --frame-rate 24 --warm --no-mp4 --text resident --dit-offload resident
  --offload none --text-cache $WK/ltx-text)
w5common=(wan gen --weights $W/fastwan22-ti2v-5b --preset fast_wan_2_2_ti2v_5b --steps 3 --flow-shift 5.0 --fps 24
  --height 480 --width 832 --num-frames 121 --prompt "$PROMPT" --prompts $P --warm --no-mp4 --text-cache $WK/wan5-text)
w13common=(wan gen --weights $W/fastwan21-1.3b --height 480 --width 832 --num-frames 81 --fps 16
  --prompt "$PROMPT" --prompts $P --warm --no-mp4 --text-cache $WK/wan13-text)
sfruns=(--run warm,seconds=10,rope=rebased,sink=3,sheet=0)
for i in 1 2 3 4 5; do sfruns+=(--run r$i,seconds=10,rope=rebased,sink=3,sheet=0); done

job() {
  case "$1" in
    h3t-def) run h3t-def -- --mode fast --techniques h3/fasth3_4step_vsa "${h3common[@]}" --h3-recipe 4step-vsa \
      --adaln-cache $WK/adaln-4step-vsa.cache --clip-dir $RES/h3t-def/clips ;;
    h3t-max) run h3t-max -- --mode fast --techniques h3/fasth3_4step_vsa "${h3common[@]}" --h3-recipe 4step-vsa \
      --adaln-cache $WK/adaln-4step-vsa.cache --taeh3-weights $TAE/taeh3.safetensors \
      --arm tae=- --arm tae-fp8attn=h3/fasth3_4step_vsa_fp8attn --clip-dir "$RES/h3t-max/{arm}/clips" ;;
    h3m-def) run h3m-def -- --mode fast --techniques h3/sol_h3_4step_engine_ladder "${h3common[@]}" --h3-recipe sol-h3 \
      --adaln-cache $WK/adaln-sol-h3.cache --clip-dir $RES/h3m-def/clips ;;
    h3m-max) run h3m-max -- --mode fast --techniques h3/sol_h3_4step_engine_ladder "${h3common[@]}" --h3-recipe sol-h3 \
      --adaln-cache $WK/adaln-sol-h3.cache --taeh3-weights $TAE/taeh3.safetensors --clip-dir $RES/h3m-max/clips ;;
    ltx-def) run ltx-def -- --mode fast --techniques ltx2/ltx25_distill_sol "${ltxcommon[@]}" --clip $RES/ltx-def/clips ;;
    ltx-max) run ltx-max -- --mode fast --techniques $LTXMAX "${ltxcommon[@]}" \
      --ltx-tae-weights $TAE/taeltx2_3_wide.safetensors --clip $RES/ltx-max/clips ;;
    w5-def) run w5-def FASTVIDEO_WAN_VAE=full -- --mode fast "${w5common[@]}" --clip-dir $RES/w5-def/clips ;;
    w5-max) run w5-max FASTVIDEO_WAN_VAE=taehv FASTVIDEO_TAE_DIR=$TAE FASTVIDEO_WAN_QUANT=$WQ -- --mode fast \
      "${w5common[@]}" --clip-dir $RES/w5-max/clips ;;
    w13-def) run w13-def FASTVIDEO_WAN_VAE=full -- --mode fast --vsa "${w13common[@]}" --clip-dir $RES/w13-def/clips ;;
    w13-max) run w13-max FASTVIDEO_WAN_VAE=taehv FASTVIDEO_TAE_DIR=$TAE FASTVIDEO_WAN_QUANT=$WQ -- --mode fast --vsa \
      "${w13common[@]}" --clip-dir $RES/w13-max/clips ;;
    sf-def) run sf-def FASTVIDEO_TAE_DIR=$TAE -- --mode fast wan stream --weights $W/sfwan21-1.3b --prompt "$PROMPT" \
      --seed 7 "${sfruns[@]}" ;;
    sf-max) run sf-max FASTVIDEO_TAE_DIR=$TAE FASTVIDEO_WAN_QUANT=$WQ -- --mode fast wan stream --weights $W/sfwan21-1.3b \
      --prompt "$PROMPT" --seed 7 "${sfruns[@]}" ;;
    *) echo "unknown job $1" >> "$WK/progress.log" ;;
  esac
}

JOBS=("$@")
[ ${#JOBS[@]} -gt 0 ] || JOBS=(h3t-def h3t-max h3m-def h3m-max ltx-def ltx-max w5-def w5-max w13-def w13-max sf-def sf-max)
for j in "${JOBS[@]}"; do job "$j"; done
echo "=== $(date -u +%T) ALL DONE" >> "$WK/progress.log"
