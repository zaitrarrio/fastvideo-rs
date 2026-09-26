#!/usr/bin/env bash
# Runs ON a Runpod GPU box. Family: h3 | ltx | hunyuan | wan | b200 | ... | trace
# One process per family. Continues to the next cell on failure.
# Spark joint LTX refine stays off (FASTVIDEO_LTX2_WEIGHTS unset) except in
# the rtx6000 parity family, which runs the full Spark bridge.
#
# Evaluation layer (precision, fastvideo and eval families):
#   every H3 / LTX-2 gen cell writes <cell>/benchmark.json (timings per stage
#   and step, peak memory, TeaCache / Sol-Attn / VSA / FFN-chunk / quantized
#   linear counters; see crates/fastvideo-gpucheck/src/benchmark.rs).
#   FV_PROMPTS=5 (or a prompt-set JSON): each gen cell runs the five
#     sol-engine prompts of scripts/gpu/prompts-eval.json in one warm process,
#     clips under <cell>/frames/<name>/, per-prompt numbers and their medians
#     in <cell>/benchmark.json; compare_cells compares prompt by prompt.
#     Unset: one prompt ($FV_PROMPT), as before.
#   FV_LPIPS=1: fetch-lpips.sh, then compare-clips adds LPIPS(alex).
#   gate_cells: fv-gpucheck gate with scripts/gpu/gate-policy.toml
#     (FV_GATE_POLICY overrides) into $RUNS/gate/.
set -euo pipefail
FAMILY="${1:?usage: runpod-matrix.sh h3|ltx|hunyuan|wan|b200|rtx6000|rtx5090|fastvideo|precision|precision-debug|trace|oracle|ltxvae|eval}"
WORK="${FV_WORK:-/workspace}"
BIN="${FV_GPUCHECK:-/opt/fastvideo-rs/target/release/fv-gpucheck}"
W="$WORK/weights"
# Everything this script writes (runs, caches, library links) goes under
# SCRATCH: the pod's container disk when FV_SCRATCH is set. The weight volume
# is only read (some hosts have silently dropped data writes to it).
SCRATCH="${FV_SCRATCH:-$WORK}"
RUNS="$SCRATCH/runs/${FAMILY}${FV_RUN_TAG:+/$FV_RUN_TAG}"
LOG="$RUNS/live.log"
PROMPT="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
SEED="${FV_SEED:-1024}"
export PATH="/opt/fastvideo-rs/target/release:/usr/local/bin:/usr/local/cuda/bin:${PATH:-}"
unset FASTVIDEO_LTX2_WEIGHTS
mkdir -p "$RUNS" "$SCRATCH/gpucheck-out/logs" "$SCRATCH/fv-libs"
# shellcheck source=scripts/gpu/cuda-13.pins
. "$(dirname "${BASH_SOURCE[0]}")/cuda-13.pins"
# cudarc looks for libcudnn.so (unversioned). The image ships the pinned SONAME.
if [[ ! -e $SCRATCH/fv-libs/libcudnn.so ]]; then
  bash /opt/fastvideo-rs/scripts/gpu/remote.sh env >/tmp/fv-env.json 2>/tmp/fv-env.err || true
fi
if [[ ! -e $SCRATCH/fv-libs/libcudnn.so ]]; then
  for spec in \
    "cudnn:/lib/x86_64-linux-gnu/${CUDA_CUDNN_SONAME}" \
    "cublas:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLAS_SONAME}" \
    "cublasLt:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLASLT_SONAME}" \
    "nvrtc:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_NVRTC_SONAME}"; do
    name="${spec%%:*}"
    src="${spec#*:}"
    [[ -e "$src" ]] && ln -sf "$(readlink -f "$src")" "$SCRATCH/fv-libs/lib${name}.so"
  done
fi
# CUPTI (FASTVIDEO_GPU_TRACE) is linked on its own: the block above is skipped
# once libcudnn.so exists, and an older image may not ship it at all (the
# tracer then reports "libcupti not found" and the run continues untraced).
if [[ ! -e $SCRATCH/fv-libs/libcupti.so ]]; then
  cupti_src="$(ldconfig -p 2>/dev/null | awk -v n="${CUDA_CUPTI_SONAME:-libcupti.so.13}" '$1 == n { print $NF; exit }' || true)"
  [[ -z "$cupti_src" ]] && cupti_src="/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUPTI_SONAME:-libcupti.so.13}"
  [[ -e "$cupti_src" ]] && ln -sf "$(readlink -f "$cupti_src")" "$SCRATCH/fv-libs/libcupti.so"
fi
export LD_LIBRARY_PATH="$SCRATCH/fv-libs:/lib/x86_64-linux-gnu:/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
[[ -e $SCRATCH/fv-libs/libcudnn.so ]] || { echo "FATAL: libcudnn.so not linked" | tee -a "$LOG"; exit 2; }

log() {
  local line
  line="$(printf '[%s] [%s] %s' "$(date -u +%H:%M:%S)" "$FAMILY" "$*")"
  # The log lives on the network volume; a transient write error there must
  # not end the whole matrix under set -e (it once did, mid-suite).
  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$LOG" 2>/dev/null || true
}

# ---- evaluation layer (see the header) ----
if [[ "$FAMILY" == eval ]]; then
  : "${FV_PROMPTS:=5}" "${FV_LPIPS:=1}"
fi
PROMPT_ARGS=()
case "${FV_PROMPTS:-}" in
  "" | 0 | 1) PROMPTS_FILE="" ;;
  5 | all) PROMPTS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prompts-eval.json" ;;
  *) PROMPTS_FILE="$FV_PROMPTS" ;;
esac
if [[ -n "$PROMPTS_FILE" ]]; then
  PROMPT_ARGS=(--prompts "$PROMPTS_FILE")
fi
LPIPS_ARGS=()
LPIPS_DIR="${FV_LPIPS_DIR:-$SCRATCH/lpips}"
GATE_POLICY="${FV_GATE_POLICY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gate-policy.toml}"

write_json() {
  local dest="$1"
  shift
  printf '%s\n' "$*" >"$dest" 2>/dev/null || true
}

smi_snap() {
  local dest="$1"
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$dest" 2>/dev/null || true
}

# Every cell starts on an empty GPU: wait for the previous process's memory
# to drain, kill any leftover compute process, and refuse to start otherwise.
gpu_clean() {
  local used i pid
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    [[ -n "$used" && "$used" -lt 1024 ]] && return 0
    if (( i == 10 )); then
      for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        log "killing leftover GPU process $pid (${used} MiB in use)"
        kill -9 "$pid" 2>/dev/null || true
      done
    fi
    sleep 2
  done
  log "GPU not clean before next cell: ${used:-?} MiB in use"
  return 1
}

run_cell() {
  local name="$1"
  shift
  if ! gpu_clean; then
    mkdir -p "$RUNS/$name"
    write_json "$RUNS/$name/summary.json" "$(printf '{"cell":"%s","family":"%s","exit":null,"skipped":"gpu not clean"}' "$name" "$FAMILY")"
    return 0
  fi
  local cell="$RUNS/$name"
  mkdir -p "$cell"
  local t0 rc secs
  t0=$(date +%s)
  local cap="${FV_GEN_TIMEOUT_S:-300}"
  log "▶ $name  $*  (wall cap ${cap}s; 0=none)"
  smi_snap "$cell/nvidia-smi-start.txt"
  set +e
  # Run inside the cell dir so fv-gpucheck's relative report dir
  # (gpucheck-out/<stage>.json) lands next to the logs.
  if [[ "$cap" == "0" ]]; then
    (cd "$cell" && "$@") >"$cell/stdout.log" 2>"$cell/stderr.log"
  else
    (cd "$cell" && timeout --signal=TERM --kill-after=15 "$cap" "$@") >"$cell/stdout.log" 2>"$cell/stderr.log"
  fi
  rc=$?
  set -e
  secs=$(( $(date +%s) - t0 ))
  smi_snap "$cell/nvidia-smi-end.txt"
  # last useful lines (skip percent spam)
  if [[ $rc -ne 0 ]]; then
    log "FAIL $name  ${secs}s  exit=$rc"
    tail -20 "$cell/stderr.log" 2>/dev/null | grep -Ev 'MiB/|%\]|block [0-9]+/' | tail -12 | tee -a "$LOG" || true
  else
    local peak=""
    peak="$(grep -E -i 'peak.*(mib|vram)|peak_mib' "$cell/stdout.log" "$cell/stderr.log" 2>/dev/null | tail -1 || true)"
    log "ok $name  ${secs}s  ${peak}"
  fi
  write_json "$cell/summary.json" "$(printf '{"cell":"%s","family":"%s","exit":%s,"seconds":%s,"ended":"%s"}' \
    "$name" "$FAMILY" "$rc" "$secs" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
  return 0
}

# Cells gated on a complete weight tree (verify-weights.sh) and FV_CELLS.
VERIFY="$(dirname "${BASH_SOURCE[0]}")/verify-weights.sh"
gated_cell() {
  local name="$1" wcell="$2"
  shift 2
  if [[ -n "${FV_CELLS:-}" && " $FV_CELLS " != *" $name "* ]]; then
    log "skip $name (not in FV_CELLS)"
    return 0
  fi
  if ! FV_WEIGHTS="$W" bash "$VERIFY" "$wcell" >"$RUNS/$name.weights.log" 2>&1; then
    mkdir -p "$RUNS/$name"
    log "SKIP $name: weights incomplete"
    tee -a "$LOG" <"$RUNS/$name.weights.log"
    write_json "$RUNS/$name/summary.json" "$(printf '{"cell":"%s","family":"%s","exit":null,"skipped":"weights incomplete"}' "$name" "$FAMILY")"
    return 0
  fi
  run_cell "$name" "$@"
}

# Tiny-autoencoder weights (madebyollin/taehv) live on the container disk:
# runpod-http's start command fetches them, and this fetches again only when a
# file is missing. They are not part of the weight tree, so verify-weights.sh
# never gates on them; a TAE cell whose file is absent is recorded as skipped.
TAE="${FV_TAE_DIR:-$SCRATCH/tae}"
TAEH3="$TAE/taeh3.safetensors"
TAELTX="$TAE/taeltx2_3_wide.safetensors"
tae_fetched=""
tae_gated_cell() {
  local name="$1" file="$2"
  shift 2
  if [[ -n "${FV_CELLS:-}" && " $FV_CELLS " != *" $name "* ]]; then
    log "skip $name (not in FV_CELLS)"
    return 0
  fi
  if [[ ! -f "$file" && -z "$tae_fetched" ]]; then
    tae_fetched=1
    bash "$(dirname "${BASH_SOURCE[0]}")/fetch-tae.sh" "$TAE" >>"$RUNS/tae-fetch.log" 2>&1 || true
  fi
  if [[ ! -f "$file" ]]; then
    mkdir -p "$RUNS/$name"
    log "SKIP $name: $file missing (see tae-fetch.log)"
    write_json "$RUNS/$name/summary.json" "$(printf '{"cell":"%s","family":"%s","exit":null,"skipped":"tae weights missing"}' "$name" "$FAMILY")"
    return 0
  fi
  gated_cell "$name" "$@"
}

# Paired-clip quality gate (`fv-gpucheck compare-clips`): the candidate
# cell's frames against the baseline cell's, sol-engine collect_run.py
# metrics (+ LPIPS with FV_LPIPS=1, on the GPU). Extra args (e.g.
# --off-identity for a switch that must not change a byte) pass through.
# Report: $RUNS/compare/compare-clips-<baseline>--<candidate>.json, or with a
# prompt set one per prompt: ...--<candidate>-<prompt>.json.
compare_one() {
  local tag="$1" bf="$2" cf="$3" dir="$RUNS/compare"
  shift 3
  mkdir -p "$dir"
  if ! compgen -G "$bf/*.png" >/dev/null || ! compgen -G "$cf/*.png" >/dev/null; then
    log "skip compare $tag (frames missing)"
    write_json "$dir/compare-clips-$tag.json" "$(printf '{"stage":"compare-clips-%s","status":"skipped","reason":"frames missing"}' "$tag")"
    return 0
  fi
  local rc
  set +e
  "$BIN" --out "$dir" --tag "$tag" compare-clips --baseline "$bf" --candidate "$cf" "${LPIPS_ARGS[@]}" "$@" \
    >"$dir/$tag.stdout.log" 2>"$dir/$tag.stderr.log"
  rc=$?
  set -e
  log "compare $tag exit=$rc $(grep -o '"psnr_mean":[^,]*' "$dir/$tag.stderr.log" 2>/dev/null | tail -1) $(grep -oE 'lpips \{"backend[^}]*"max":[^,]*,"mean":[^,]*' "$dir/$tag.stderr.log" 2>/dev/null | grep -oE '"(max|mean)":[^,]*' | tr '\n' ' ')"
  return 0
}

compare_cells() {
  local base="$1" cand="$2"
  shift 2
  local bf="$RUNS/$base/frames" cf="$RUNS/$cand/frames" p found=""
  # Single prompt: frames directly under frames/. Prompt set: one directory
  # per prompt (cold/ and warmup/ are the untimed passes).
  if compgen -G "$bf/*.png" >/dev/null || [[ ! -d "$bf" ]]; then
    compare_one "$base--$cand" "$bf" "$cf" "$@"
    return 0
  fi
  for p in "$bf"/*/; do
    p="$(basename "$p")"
    [[ "$p" == cold || "$p" == warmup ]] && continue
    compgen -G "$bf/$p/*.png" >/dev/null || continue
    found=1
    compare_one "$base--$cand-$p" "$bf/$p" "$cf/$p" "$@"
  done
  [[ -n "$found" ]] || compare_one "$base--$cand" "$bf" "$cf" "$@"
  return 0
}

# Promotion gate (`fv-gpucheck gate`, $GATE_POLICY): the candidate cell's
# benchmark.json and compare reports against the baseline's. $3 = kind
# (lossy | exact); $4 = an OFF-arm cell whose compare with the baseline (made
# with --off-identity) must be byte-identical. Report: $RUNS/gate/gate-<b>--<c>.json.
gate_cells() {
  local base="$1" cand="$2" kind="${3:-lossy}" off="${4:-}" tag="$1--$2" dir="$RUNS/gate" f
  local cmp=() offs=()
  mkdir -p "$dir"
  for f in "$RUNS/compare/compare-clips-$tag.json" "$RUNS/compare/compare-clips-$tag"-*.json; do
    [[ -f "$f" ]] && cmp+=(--compare "$f")
  done
  if [[ -n "$off" ]]; then
    for f in "$RUNS/compare/compare-clips-$base--$off.json" "$RUNS/compare/compare-clips-$base--$off"-*.json; do
      [[ -f "$f" ]] && offs+=(--off-compare "$f")
    done
  fi
  local rc
  set +e
  "$BIN" --out "$dir" --tag "$tag" gate --baseline "$RUNS/$base" --candidate "$RUNS/$cand" \
    --policy "$GATE_POLICY" --kind "$kind" "${cmp[@]}" "${offs[@]}" \
    >"$dir/$tag.stdout.log" 2>"$dir/$tag.stderr.log"
  rc=$?
  set -e
  log "gate $tag exit=$rc $(grep -o 'gate verdict: .*' "$dir/$tag.stderr.log" 2>/dev/null | tail -1)"
  return 0
}

if [[ -n "$PROMPTS_FILE" ]]; then
  log "prompt set: $PROMPTS_FILE"
fi
if [[ "${FV_LPIPS:-0}" == 1 ]]; then
  if bash "$(dirname "${BASH_SOURCE[0]}")/fetch-lpips.sh" "$LPIPS_DIR" >>"$RUNS/lpips-fetch.log" 2>&1; then
    LPIPS_ARGS=(--lpips "$LPIPS_DIR")
    log "lpips weights: $LPIPS_DIR"
  else
    log "WARN: LPIPS weights unavailable (lpips-fetch.log); compare-clips runs without LPIPS"
  fi
fi

# oracle_fetch <target> <dir>: wait (FV_ORACLE_WAIT_S, default 3 h) until an
# upstream pod under one of FV_ORACLE_URL's space-separated base URLs has
# finished oracle-<target> (scripts/gpu/upstream/oracle.sh), then download its
# oracle-dump.tar and unpack it to <dir>/dump. Fails when the reference wrote
# no dump or never finished.
oracle_fetch() {
  local target="$1" dest="$2" t0 base="" u attempt
  [[ -n "${FV_ORACLE_URL:-}" ]] || { log "oracle: FV_ORACLE_URL unset"; return 1; }
  t0=$(date +%s)
  while [[ -z "$base" ]]; do
    for u in $FV_ORACLE_URL; do
      if curl -sS --max-time 30 --fail -o /dev/null "$u/oracle-$target/ORACLE_DONE" 2>/dev/null; then
        base="$u/oracle-$target"
        break
      fi
    done
    [[ -n "$base" ]] && break
    if (( $(date +%s) - t0 >= ${FV_ORACLE_WAIT_S:-10800} )); then
      log "oracle: $target not finished upstream after ${FV_ORACLE_WAIT_S:-10800}s"
      return 1
    fi
    sleep 30
  done
  log "oracle: fetching $base/oracle-dump.tar"
  rm -rf "$dest"
  mkdir -p "$dest"
  for attempt in 1 2 3; do
    if curl -sS --fail --retry 3 --max-time 3600 -o "$dest/dump.tar" "$base/oracle-dump.tar" \
      && tar -xf "$dest/dump.tar" -C "$dest" && [[ -d "$dest/dump" ]]; then
      rm -f "$dest/dump.tar"
      log "oracle: $target reference $(du -sh "$dest/dump" | cut -f1)"
      return 0
    fi
    log "oracle: fetch of $target failed (attempt $attempt)"
    sleep 10
  done
  return 1
}

log "matrix start image=$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null || echo unknown)"
if [[ ! -x "$BIN" ]]; then
  log "FATAL: fv-gpucheck missing at $BIN"
  exit 2
fi
command -v nvidia-smi >/dev/null || { log "FATAL: nvidia-smi missing"; exit 2; }
nvidia-smi -L | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"

case "$FAMILY" in
  h3)
    run_cell fasth3-8step-warm \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-8step" \
        --prompt "$PROMPT" \
        --seconds 5 \
        --seed "$SEED" \
        --text-encoder streamed \
        --text-cache "$SCRATCH/h3-text-cache" \
        --text-weights "$W/h3-base" \
        --adaln-cache "$RUNS/h3-adaln.cache" \
        --clip-dir "$RUNS/fasth3-8step-warm/frames"
    run_cell sol-h3-4step \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-base" \
        --prompt "$PROMPT" \
        --seconds 5 \
        --seed "$SEED" \
        --text-encoder streamed \
        --text-cache "$SCRATCH/h3-text-cache" \
        --text-weights "$W/h3-base" \
        --h3-recipe sol-h3 \
        --adaln-cache "$RUNS/sol-h3-adaln.cache" \
        --clip-dir "$RUNS/sol-h3-4step/frames"
    spark_up=""
    spark_ad=""
    for p in \
      "$W/upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
      "$W/upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-spark-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-spark/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-base/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/minimax_h3_latent_upscaler_3d_bf16.safetensors"; do
      [[ -f "$p" ]] && spark_up="$p" && break
    done
    for p in \
      "$W/h3-to-ltx" \
      "$W/h3_ltx_adapter" \
      "$W/H3-to-LTX-Latent-Adapter" \
      "$W/h3-base/h3-to-ltx" \
      "$W/h3-base/h3_ltx_adapter"; do
      if [[ -f "$p/model.safetensors" && -f "$p/config.json" ]]; then
        spark_ad="$p"
        break
      fi
    done
    if [[ -n "$spark_up" && -n "$spark_ad" ]]; then
      log "spark files present ($spark_up + $spark_ad) — sol-h3-spark, no joint LTX refine"
      run_cell sol-h3-spark \
        env -u FASTVIDEO_LTX2_WEIGHTS \
        FASTVIDEO_H3_QUANT=w8a8 FASTVIDEO_BF16_ACT=1 \
        FASTVIDEO_H3_UPSCALER="$spark_up" \
        FASTVIDEO_H3_LTX_ADAPTER="$spark_ad" \
        "$BIN" --mode fast h3 gen \
          --weights "$W/h3-base" \
          --prompt "$PROMPT" \
          --seconds 5 \
          --seed "$SEED" \
          --text-encoder streamed \
          --text-cache "$SCRATCH/h3-text-cache" \
          --text-weights "$W/h3-base" \
          --h3-recipe sol-h3-spark \
          --adaln-cache "$RUNS/sol-h3-spark-adaln.cache" \
          --clip-dir "$RUNS/sol-h3-spark/frames"
    else
      log "skip sol-h3-spark: upscaler=${spark_up:-missing} adapter=${spark_ad:-missing}"
    fi
    ;;
  ltx)
    run_cell ltx25-two-stage \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.5 \
        --weights "$W/ltx25" \
        --dit "$W/ltx25" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --two-stage \
        --text streamed \
        --warm \
        --clip "$RUNS/ltx25-two-stage/frames"
    run_cell ltx23-two-stage \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.3 \
        --weights "$W/ltx23" \
        --dit "$W/ltx23" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --two-stage \
        --text streamed \
        --clip "$RUNS/ltx23-two-stage/frames"
    run_cell ltx20-distilled-8step \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.0 \
        --weights "$W/ltx2" \
        --dit "$W/ltx2/ltx-2-19b-distilled.safetensors" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --clip "$RUNS/ltx20-distilled-8step/frames"
    ;;
  hunyuan)
    if ! "$BIN" hunyuan --help >/dev/null 2>&1; then
      log "skip hunyuan: fv-gpucheck in this image has no hunyuan gen tier"
      write_json "$RUNS/skipped.json" '{"status":"skipped","reason":"fv-gpucheck hunyuan gen not in published image"}'
      exit 0
    fi
    run_cell hy15-480-t2v \
      "$BIN" --mode fast hunyuan gen \
        --preset hy15_480p_t2v \
        --weights "$W/hy15-480-t2v" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --clip "$RUNS/hy15-480-t2v/frames"
    ;;
  wan)
    wan_w=""
    for p in "$W/fastwan21-1.3b" "$W/wan21-1.3b" "$W/Wan2.1-T2V-1.3B-Diffusers"; do
      [[ -d "$p/transformer" || -d "$p" ]] && wan_w="$p" && break
    done
    if [[ -z "$wan_w" ]]; then
      log "skip wan: no 1.3B tree under $W (fastwan21-1.3b / wan21-1.3b)"
      write_json "$RUNS/skipped.json" '{"status":"skipped","reason":"wan 1.3B weights not on this volume"}'
      exit 0
    fi
    # UMT5 first (own process), then 17-frame / 3-step DMD. `--embeds` is a file.
    mkdir -p "$RUNS/wan13-embeds"
    neg='Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'
    printf '{"negative":%s,"prompts":[{"name":"wan13-dmd-3step","prompt":%s}]}\n' \
      "$(printf '%s' "$neg" | sed 's/\\/\\\\/g;s/"/\\"/g;s/.*/"&"/')" \
      "$(printf '%s' "$PROMPT" | sed 's/\\/\\\\/g;s/"/\\"/g;s/.*/"&"/')" \
      >"$RUNS/prompt.json"
    run_cell wan13-embed \
      "$BIN" --mode fast embed \
        --weights "$wan_w" \
        --prompts "$RUNS/prompt.json" \
        --embeds "$RUNS/wan13-embeds"
    run_cell wan13-dmd-3step \
      "$BIN" --mode fast clip \
        --weights "$wan_w" \
        --embeds "$RUNS/wan13-embeds/wan13-dmd-3step.safetensors" \
        --dmd \
        --frames 17 \
        --steps 3 \
        --seed "$SEED" \
        --name wan13-dmd-3step
    ;;
  b200)
    # Warm B200 parity: H3 / FastH3 / LTX only. Official VAE stays the
    # default elsewhere; this family opts into TAEH3. Oxide GEMM stays off.
    unset FASTVIDEO_NVFP4_OXIDE_GEMM
    taeh3=""
    for p in "$W/taeh3/taeh3.safetensors" "$W/taeh3" "$WORK/taeh3/taeh3.safetensors"; do
      if [[ -f "$p" || ( -d "$p" && -f "$p/taeh3.safetensors" ) ]]; then
        taeh3="$p"
        break
      fi
    done
    if [[ -z "$taeh3" ]]; then
      log "WARN: TAEH3 missing — H3 cells will use official VAE"
    else
      log "taeh3=$taeh3"
    fi
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    if [[ -n "$taeh3" ]]; then
      h3_common+=(--taeh3-weights "$taeh3")
    fi
    # FastH3 Preview recipes: base transformer + the Preview LoRA found at
    # $W/FastH3-4-step-Preview-v1-LoRA/<vsa|dense>-datafree (refused on h3-8step).
    run_cell fasth3-4step-vsa \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-base" \
        --h3-recipe 4step-vsa \
        --adaln-cache "$RUNS/fasth3-4step-vsa-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-vsa/frames" \
        "${h3_common[@]}"
    run_cell fasth3-8step \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-8step" \
        --h3-recipe 8step \
        --adaln-cache "$RUNS/fasth3-8step-adaln.cache" \
        --clip-dir "$RUNS/fasth3-8step/frames" \
        "${h3_common[@]}"
    run_cell fasth3-4step-dense \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-base" \
        --h3-recipe 4step-dense \
        --adaln-cache "$RUNS/fasth3-4step-dense-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-dense/frames" \
        "${h3_common[@]}"
    run_cell sol-h3 \
      "$BIN" --mode fast h3 gen \
        --weights "$W/h3-base" \
        --h3-recipe sol-h3 \
        --adaln-cache "$RUNS/sol-h3-adaln.cache" \
        --clip-dir "$RUNS/sol-h3/frames" \
        "${h3_common[@]}"
    spark_up=""
    spark_ad=""
    for p in \
      "$W/upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
      "$W/upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-spark-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-spark/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/h3-base/minimax_h3_latent_upscaler_3d_bf16.safetensors" \
      "$W/minimax_h3_latent_upscaler_3d_bf16.safetensors"; do
      [[ -f "$p" ]] && spark_up="$p" && break
    done
    for p in \
      "$W/h3-to-ltx" \
      "$W/h3_ltx_adapter" \
      "$W/H3-to-LTX-Latent-Adapter" \
      "$W/h3-base/h3-to-ltx" \
      "$W/h3-base/h3_ltx_adapter"; do
      if [[ -f "$p/model.safetensors" && -f "$p/config.json" ]]; then
        spark_ad="$p"
        break
      fi
    done
    if [[ -n "$spark_up" && -n "$spark_ad" ]]; then
      log "spark files present ($spark_up + $spark_ad) — sol-h3-spark, no joint LTX refine"
      run_cell sol-h3-spark \
        env -u FASTVIDEO_LTX2_WEIGHTS \
        FASTVIDEO_H3_QUANT=w8a8 FASTVIDEO_BF16_ACT=1 \
        FASTVIDEO_H3_UPSCALER="$spark_up" \
        FASTVIDEO_H3_LTX_ADAPTER="$spark_ad" \
        "$BIN" --mode fast h3 gen \
          --weights "$W/h3-base" \
          --h3-recipe sol-h3-spark \
          --adaln-cache "$RUNS/sol-h3-spark-adaln.cache" \
          --clip-dir "$RUNS/sol-h3-spark/frames" \
          "${h3_common[@]}"
    else
      log "skip sol-h3-spark: upscaler=${spark_up:-missing} adapter=${spark_ad:-missing}"
    fi
    run_cell ltx25-two-stage \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.5 \
        --weights "$W/ltx25" \
        --dit "$W/ltx25" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --two-stage \
        --text streamed \
        --warm \
        --clip "$RUNS/ltx25-two-stage/frames"
    run_cell ltx23-two-stage \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.3 \
        --weights "$W/ltx23" \
        --dit "$W/ltx23" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --two-stage \
        --text streamed \
        --warm \
        --clip "$RUNS/ltx23-two-stage/frames"
    run_cell ltx20-distilled-8step \
      "$BIN" --mode fast ltx2 gen \
        --model-version 2.0 \
        --weights "$W/ltx2" \
        --dit "$W/ltx2/ltx-2-19b-distilled.safetensors" \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --warm \
        --clip "$RUNS/ltx20-distilled-8step/frames"
    ;;
  rtx6000)
    # RTX PRO 6000 parity baseline: H3, FastH3 and LTX-2.5 on the sol-engine
    # single-GPU contracts. Every cell first proves its weight tree complete
    # (verify-weights.sh); an incomplete tree is recorded, not run.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    gated_cell fasth3-8step fasth3-8step \
      "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
        --adaln-cache "$RUNS/fasth3-8step-adaln.cache" \
        --clip-dir "$RUNS/fasth3-8step/frames" "${h3_common[@]}"
    gated_cell fasth3-4step-vsa fasth3-4step-vsa \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
        --adaln-cache "$RUNS/fasth3-4step-vsa-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-vsa/frames" "${h3_common[@]}"
    # Sol-H3 4-step on one GPU is dense upstream (engine.py refuses Sol at world_size 1).
    gated_cell sol-h3 sol-h3 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3 \
        --adaln-cache "$RUNS/sol-h3-adaln.cache" \
        --clip-dir "$RUNS/sol-h3/frames" "${h3_common[@]}"
    # The upstream single-GPU Sol-Attn + TeaCache route (RTX4090/5090 profile).
    gated_cell sol-h3-rtx h3-base \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx \
        --adaln-cache "$RUNS/sol-h3-rtx-adaln.cache" \
        --clip-dir "$RUNS/sol-h3-rtx/frames" "${h3_common[@]}"
    # The RTX 5090 `fullopt` arm: the same Sol route plus TeaCache 0.10 / 5 / 1.
    gated_cell sol-h3-rtx-teacache h3-base \
      env FASTVIDEO_H3_SOL_CACHE=teacache \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx \
        --adaln-cache "$RUNS/sol-h3-rtx-adaln.cache" \
        --clip-dir "$RUNS/sol-h3-rtx-teacache/frames" "${h3_common[@]}"
    # Upstream Spark stage 1 is W8A8 FP8 (stage1.py `W8A8_FP8_after_BF16_LoRA_merge`).
    gated_cell sol-h3-spark sol-h3-spark \
      env FASTVIDEO_LTX2_WEIGHTS="$W/ltx25" \
        FASTVIDEO_H3_QUANT=w8a8 FASTVIDEO_BF16_ACT=1 \
        FASTVIDEO_H3_UPSCALER="$W/upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors" \
        FASTVIDEO_H3_LTX_ADAPTER="$W/h3-to-ltx" \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-spark \
        --adaln-cache "$RUNS/sol-h3-spark-adaln.cache" \
        --clip-dir "$RUNS/sol-h3-spark/frames" "${h3_common[@]}"
    gated_cell ltx25-two-stage ltx25-two-stage \
      "$BIN" --mode fast ltx2 gen --model-version 2.5 \
        --weights "$W/ltx25" --dit "$W/ltx25" \
        --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
        --clip "$RUNS/ltx25-two-stage/frames"
    ;;
  rtx5090)
    # The sol-engine RTX 5090 single-GPU suite (config/minimax_h3/rtx5090_*.toml,
    # models/ltx25/RTX5090) plus 480p. On a 32 GB card the DiT streams from
    # pinned host memory (FASTVIDEO_DIT_OFFLOAD=auto picks it); the same cells
    # run resident on larger cards for a like-for-like comparison.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    for res in 768p 480p; do
      geo=()
      [[ "$res" == 480p ]] && geo=(--height 480 --width 832)
      # dense: rtx5090_dense.toml; sol: rtx5090_sol.toml; fullopt: + TeaCache.
      gated_cell "h3-$res-dense" h3-base \
        env FASTVIDEO_H3_SOL_ATTN=off \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx "${geo[@]}" \
          --adaln-cache "$RUNS/h3-$res-adaln.cache" \
          --clip-dir "$RUNS/h3-$res-dense/frames" "${h3_common[@]}"
      gated_cell "h3-$res-sol" h3-base \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx "${geo[@]}" \
          --adaln-cache "$RUNS/h3-$res-adaln.cache" \
          --clip-dir "$RUNS/h3-$res-sol/frames" "${h3_common[@]}"
      gated_cell "h3-$res-fullopt" h3-base \
        env FASTVIDEO_H3_SOL_CACHE=teacache \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx "${geo[@]}" \
          --adaln-cache "$RUNS/h3-$res-adaln.cache" \
          --clip-dir "$RUNS/h3-$res-fullopt/frames" "${h3_common[@]}"
      # fullopt with the TAEH3 video decoder (sol-engine super_acceleration
      # stage 1 decodes with TAEH3 instead of the official video VAE).
      tae_gated_cell "h3-$res-fullopt-taeh3" "$TAEH3" h3-base \
        env FASTVIDEO_H3_SOL_CACHE=teacache \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx "${geo[@]}" \
          --taeh3-weights "$TAEH3" \
          --adaln-cache "$RUNS/h3-$res-adaln.cache" \
          --clip-dir "$RUNS/h3-$res-fullopt-taeh3/frames" "${h3_common[@]}"
      # Each technique against the step before it, and fullopt against dense.
      compare_cells "h3-$res-dense" "h3-$res-sol"
      compare_cells "h3-$res-sol" "h3-$res-fullopt"
      compare_cells "h3-$res-dense" "h3-$res-fullopt"
      compare_cells "h3-$res-fullopt" "h3-$res-fullopt-taeh3"
    done
    # LTX-2.5 distilled two-stage: the reference's 4k5s and 1080p20s workloads,
    # dense vs Sol stage 2, plus 480p-class (768x512, two-stage needs /64).
    for wl in 4k5s 1080p20s 512p; do
      geo=(--workload "$wl")
      [[ "$wl" == 512p ]] && geo=(--height 512 --width 768 --num-frames 121)
      for arm in sol dense; do
        flag=()
        [[ "$arm" == dense ]] && flag=(--dense-stage2)
        gated_cell "ltx25-$wl-$arm" ltx25-two-stage \
          "$BIN" --mode fast ltx2 gen --model-version 2.5 \
            --weights "$W/ltx25" --dit "$W/ltx25" "${geo[@]}" "${flag[@]}" \
            --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
            --clip "$RUNS/ltx25-$wl-$arm/frames"
      done
      # Sol stage 2, decoded by the wide LTX tiny autoencoder (sol-engine
      # models/ltx2.5-refiner: taeltx2_3_wide replaces the conv VAE decode).
      tae_gated_cell "ltx25-$wl-sol-taehv" "$TAELTX" ltx25-two-stage \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" "${geo[@]}" \
          --ltx-tae-weights "$TAELTX" \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-$wl-sol-taehv/frames"
      compare_cells "ltx25-$wl-dense" "ltx25-$wl-sol"
      compare_cells "ltx25-$wl-sol" "ltx25-$wl-sol-taehv"
    done
    ;;
  fastvideo)
    # Our optimized stack with no sol-engine technique (no Sol-Attn, no
    # TeaCache): base H3 on the dense flash kernel, the FastH3 distilled VSA
    # recipes, and LTX-2.5 distilled two-stage with a dense stage 2.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
      "${PROMPT_ARGS[@]}"
    )
    for res in 768p 480p; do
      geo=()
      [[ "$res" == 480p ]] && geo=(--height 480 --width 832)
      gated_cell "h3-base-$res" h3-base \
        env FASTVIDEO_H3_SOL_ATTN=off \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx "${geo[@]}" \
          --adaln-cache "$RUNS/h3-base-$res-adaln.cache" \
          --clip-dir "$RUNS/h3-base-$res/frames" "${h3_common[@]}"
      gated_cell "fasth3-8step-$res" fasth3-8step \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step "${geo[@]}" \
          --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-$res/frames" "${h3_common[@]}"
      gated_cell "fasth3-4step-vsa-$res" fasth3-4step-vsa \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa "${geo[@]}" \
          --adaln-cache "$RUNS/fasth3-4step-vsa-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-4step-vsa-$res/frames" "${h3_common[@]}"
      # The same FastH3 recipes decoded by TAEH3 instead of the official VAE.
      tae_gated_cell "fasth3-8step-$res-taeh3" "$TAEH3" fasth3-8step \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step "${geo[@]}" \
          --taeh3-weights "$TAEH3" \
          --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-$res-taeh3/frames" "${h3_common[@]}"
      tae_gated_cell "fasth3-4step-vsa-$res-taeh3" "$TAEH3" fasth3-4step-vsa \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa "${geo[@]}" \
          --taeh3-weights "$TAEH3" \
          --adaln-cache "$RUNS/fasth3-4step-vsa-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-4step-vsa-$res-taeh3/frames" "${h3_common[@]}"
      compare_cells "fasth3-8step-$res" "fasth3-8step-$res-taeh3"
      compare_cells "fasth3-4step-vsa-$res" "fasth3-4step-vsa-$res-taeh3"
      gate_cells "fasth3-8step-$res" "fasth3-8step-$res-taeh3" lossy
      gate_cells "fasth3-4step-vsa-$res" "fasth3-4step-vsa-$res-taeh3" lossy
    done
    gated_cell fasth3-4step-dense-768p fasth3-4step-dense \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-dense \
        --adaln-cache "$RUNS/fasth3-4step-dense-768p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-dense-768p/frames" "${h3_common[@]}"
    for wl in 4k5s 1080p20s 512p; do
      geo=(--workload "$wl")
      [[ "$wl" == 512p ]] && geo=(--height 512 --width 768 --num-frames 121)
      gated_cell "ltx25-$wl" ltx25-two-stage \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" "${geo[@]}" --dense-stage2 \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-$wl/frames" "${PROMPT_ARGS[@]}"
      tae_gated_cell "ltx25-$wl-taehv" "$TAELTX" ltx25-two-stage \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" "${geo[@]}" --dense-stage2 \
          --ltx-tae-weights "$TAELTX" \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-$wl-taehv/frames" "${PROMPT_ARGS[@]}"
      compare_cells "ltx25-$wl" "ltx25-$wl-taehv"
      gate_cells "ltx25-$wl" "ltx25-$wl-taehv" lossy
    done
    ;;
  precision)
    # Re-measure the precision switches on the current build against the
    # baselines of the fastvideo / rtx5090 families (same prompt, seed, warm).
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    # The reference recipes replace the retired per-tensor switches
    # (FASTVIDEO_H3_FFN_FP8 / FASTVIDEO_FP8 on H3): w8a8 = Spark stage 1,
    # mxfp8 = Sol-H3 blocks 2..=46; both run bf16 activations.
    for v in f32act bf16act w8a8 mxfp8; do
      envs=()
      case "$v" in
        f32act) envs=(FASTVIDEO_BF16_ACT=0) ;;
        bf16act) envs=(FASTVIDEO_BF16_ACT=1) ;;
        w8a8) envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=w8a8) ;;
        mxfp8) envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=mxfp8) ;;
      esac
      gated_cell "fasth3-8step-768p-$v" fasth3-8step \
        env "${envs[@]}" \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
          --adaln-cache "$RUNS/fasth3-8step-768p-$v-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-768p-$v/frames" "${h3_common[@]}" "${PROMPT_ARGS[@]}"
    done
    for v in bf16act w8a8 mxfp8; do
      compare_cells fasth3-8step-768p-f32act "fasth3-8step-768p-$v"
      gate_cells fasth3-8step-768p-f32act "fasth3-8step-768p-$v" lossy
    done
    # Both FP8 recipes run bf16 activations: against bf16act alone, the
    # pairs isolate what the weight/activation quantization changes.
    for v in w8a8 mxfp8; do
      compare_cells fasth3-8step-768p-bf16act "fasth3-8step-768p-$v"
    done
    for v in f32act bf16act fp8; do
      envs=()
      case "$v" in
        f32act) envs=(FASTVIDEO_BF16_ACT=0) ;;
        bf16act) envs=(FASTVIDEO_BF16_ACT=1) ;;
        fp8) envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_FP8=1) ;;
      esac
      gated_cell "ltx25-4k5s-sol-$v" ltx25-two-stage \
        env "${envs[@]}" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 4k5s \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-4k5s-sol-$v/frames" "${PROMPT_ARGS[@]}"
    done
    for v in bf16act fp8; do
      compare_cells ltx25-4k5s-sol-f32act "ltx25-4k5s-sol-$v"
      gate_cells ltx25-4k5s-sol-f32act "ltx25-4k5s-sol-$v" lossy
    done
    ;;
  precision-debug)
    # Two precision-run findings, measured cheaply.
    # (1) LTX-2.5 FASTVIDEO_FP8 at 512p, with and without bf16 activations,
    #     each step CUPTI-traced (FASTVIDEO_GPU_TRACE_STEP=1: the second
    #     stage-1 step): host enqueue vs GPU busy, memcpy directions, top
    #     kernels; `ltx2 step transfers` lines count PCIe traffic per step.
    for v in f32act fp8 bf16act bf16act-fp8; do
      envs=(FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_STEP=1)
      case "$v" in
        f32act) envs+=(FASTVIDEO_BF16_ACT=0) ;;
        fp8) envs+=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_FP8=1) ;;
        bf16act) envs+=(FASTVIDEO_BF16_ACT=1) ;;
        bf16act-fp8) envs+=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_FP8=1) ;;
      esac
      gated_cell "ltx25-512p-$v" ltx25-two-stage \
        env "${envs[@]}" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --height 512 --width 768 --num-frames 121 \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed \
          --clip "$RUNS/ltx25-512p-$v/frames"
    done
    for cell in ltx25-512p-f32act ltx25-512p-fp8 ltx25-512p-bf16act ltx25-512p-bf16act-fp8; do
      grep -hE 'ancestral step|step transfers|fp8 linear|FASTVIDEO_FP8' "$RUNS/$cell/stderr.log" 2>/dev/null \
        | head -24 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    # (2) H3 8-step 768p: f32 vs bf16 activations (and W8A8 as a control
    #     perturbation) from the same seed, every step's latents and velocity
    #     and the first step's block outputs dumped (FASTVIDEO_DUMP_DIR), then
    #     compared tensor by tensor. The dumps are deleted afterwards.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
    )
    for v in f32act bf16act w8a8; do
      envs=(FASTVIDEO_DUMP_DIR="$RUNS/h3dump-$v")
      case "$v" in
        f32act) envs+=(FASTVIDEO_BF16_ACT=0) ;;
        bf16act) envs+=(FASTVIDEO_BF16_ACT=1) ;;
        w8a8) envs+=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=w8a8) ;;
      esac
      gated_cell "fasth3-8step-768p-$v" fasth3-8step \
        env "${envs[@]}" \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
          --adaln-cache "$RUNS/fasth3-8step-768p-$v-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-768p-$v/frames" "${h3_common[@]}"
    done
    for pair in f32act:bf16act bf16act:w8a8 f32act:w8a8; do
      a="${pair%%:*}" b="${pair##*:}"
      if [[ -d "$RUNS/h3dump-$a" && -d "$RUNS/h3dump-$b" ]]; then
        run_cell "dumps-$a--$b" "$BIN" compare-dumps \
          --baseline "$RUNS/h3dump-$a" --candidate "$RUNS/h3dump-$b"
        grep -h 'compare-dumps' "$RUNS/dumps-$a--$b/stderr.log" | sed "s/^/[$a--$b] /" | tee -a "$LOG" || true
      fi
    done
    compare_cells fasth3-8step-768p-f32act fasth3-8step-768p-bf16act
    compare_cells fasth3-8step-768p-bf16act fasth3-8step-768p-w8a8
    rm -rf "$RUNS"/h3dump-*
    ;;
  trace)
    # FASTVIDEO_GPU_TRACE=1: a CUPTI activity trace of one warm denoise step
    # (the timed pass's second step; the `[INFO] h3/gpu_trace` line in
    # stderr.log and `gpu_trace` in gpucheck-out/*.json). Does the GPU go
    # idle inside a step (CUDA-graph replay worth building), and which
    # kernels hold the time? Command lines are the fastvideo family's.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    gated_cell fasth3-4step-vsa-768p fasth3-4step-vsa \
      env FASTVIDEO_GPU_TRACE=1 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
        --adaln-cache "$RUNS/fasth3-4step-vsa-768p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-vsa-768p/frames" "${h3_common[@]}"
    gated_cell fasth3-8step-768p fasth3-8step \
      env FASTVIDEO_GPU_TRACE=1 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
        --adaln-cache "$RUNS/fasth3-8step-768p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-8step-768p/frames" "${h3_common[@]}"
    gated_cell fasth3-4step-dense-768p fasth3-4step-dense \
      env FASTVIDEO_GPU_TRACE=1 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-dense \
        --adaln-cache "$RUNS/fasth3-4step-dense-768p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-dense-768p/frames" "${h3_common[@]}"
    # LTX-2.5 Sol two-stage: steps count across both stages (8 stage-1, then
    # 3 stage-2), so step 9 is the second full-resolution stage-2 step.
    gated_cell ltx25-4k5s-sol ltx25-two-stage \
      env FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_STEP=9 \
      "$BIN" --mode fast ltx2 gen --model-version 2.5 \
        --weights "$W/ltx25" --dit "$W/ltx25" --workload 4k5s \
        --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
        --clip "$RUNS/ltx25-4k5s-sol/frames"
    for cell in fasth3-4step-vsa-768p fasth3-8step-768p fasth3-4step-dense-768p ltx25-4k5s-sol; do
      grep -h '/gpu_trace ' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -1 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    ;;
  eval)
    # The evaluation layer end to end on one card: precision A/B pairs with
    # benchmark.json, the prompt set (FV_PROMPTS, default 5 here), LPIPS
    # (FV_LPIPS, default on) and the promotion gate.
    #   H3 FastH3 8-step 480p: FASTVIDEO_BF16_ACT=0 (baseline) vs the bf16
    #     default, plus a second f32 arm that must reproduce the baseline
    #     byte for byte (the OFF arm of the switch).
    #   LTX-2.5 two-stage 512p: FASTVIDEO_BF16_ACT=0 vs the bf16 default.
    # FV_LPIPS_SELFTEST=1 (default) first checks the LPIPS port against the
    # pinned official numbers (fv-gpucheck lpips, CPU and device).
    if [[ "${FV_LPIPS_SELFTEST:-1}" == 1 && ${#LPIPS_ARGS[@]} -gt 0 ]]; then
      run_cell lpips-selftest "$BIN" --out "$RUNS/lpips-selftest/gpucheck-out" lpips --weights "$LPIPS_DIR"
      grep -hE '^\[(PASS|FAIL)\] lpips/' "$RUNS/lpips-selftest/stderr.log" 2>/dev/null | cut -c1-220 | tee -a "$LOG" || true
    fi
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
      --height 480 --width 832
      "${PROMPT_ARGS[@]}"
    )
    for v in f32act bf16act f32act-off; do
      envs=(FASTVIDEO_BF16_ACT=1)
      [[ "$v" == f32act* ]] && envs=(FASTVIDEO_BF16_ACT=0)
      gated_cell "fasth3-8step-480p-$v" fasth3-8step \
        env "${envs[@]}" \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
          --adaln-cache "$RUNS/fasth3-8step-480p-$v-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-480p-$v/frames" "${h3_common[@]}"
    done
    compare_cells fasth3-8step-480p-f32act fasth3-8step-480p-bf16act
    compare_cells fasth3-8step-480p-f32act fasth3-8step-480p-f32act-off --off-identity
    gate_cells fasth3-8step-480p-f32act fasth3-8step-480p-bf16act lossy fasth3-8step-480p-f32act-off
    # The OFF arm judged as its own candidate: byte-identical, and no faster.
    gate_cells fasth3-8step-480p-f32act fasth3-8step-480p-f32act-off exact fasth3-8step-480p-f32act-off
    for v in f32act bf16act; do
      envs=(FASTVIDEO_BF16_ACT=1)
      [[ "$v" == f32act ]] && envs=(FASTVIDEO_BF16_ACT=0)
      gated_cell "ltx25-512p-$v" ltx25-two-stage \
        env "${envs[@]}" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --height 512 --width 768 --num-frames 121 \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-512p-$v/frames" "${PROMPT_ARGS[@]}"
    done
    compare_cells ltx25-512p-f32act ltx25-512p-bf16act
    gate_cells ltx25-512p-f32act ltx25-512p-bf16act lossy
    for f in "$RUNS"/gate/gate-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -o '"verdict": "[a-z]*"' "$f" | head -1)"
    done
    ;;
  ltxvae)
    # LTX-2.5 conv VAE decode alone (`ltx2 vae-bench`), at the sol-engine
    # 4k5s and 1080p20s geometries on synthetic latents, tiled as gen tiles:
    # the channels-last bf16 decoder against the f32 streaming one, each
    # warm, CUPTI-traced (FASTVIDEO_GPU_TRACE_DECODE, set FV_VAE_TRACE=0 to
    # time untraced) and with frames for compare-clips. The fast cell also
    # checks its first tile against the streaming decoder
    # (FASTVIDEO_LTX_VAE_CHECK). FV_VAE_GEN=1 adds one cold (no --warm) gen
    # per workload that saves its latents, then decodes those with the
    # streaming decoder, for a real-content PSNR.
    trace="${FV_VAE_TRACE:-1}"
    for wl in ${FV_VAE_WORKLOADS:-4k5s 1080p20s}; do
      for dec in fast streaming; do
        gated_cell "ltxvae-$wl-$dec" ltx25-two-stage \
          env FASTVIDEO_GPU_TRACE_DECODE="$trace" FASTVIDEO_LTX_VAE_CHECK=1 \
          "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload "$wl" \
            --decoder "$dec" --warm --clip "$RUNS/ltxvae-$wl-$dec/frames"
      done
      compare_cells "ltxvae-$wl-streaming" "ltxvae-$wl-fast"
      if [[ "${FV_VAE_GEN:-0}" == 1 ]]; then
        gated_cell "ltx25-$wl-gen" ltx25-two-stage \
          env FASTVIDEO_LTX2_SAVE_LATENTS="$RUNS/latents-$wl" FASTVIDEO_GPU_TRACE_DECODE="$trace" \
          "$BIN" --mode fast ltx2 gen --model-version 2.5 \
            --weights "$W/ltx25" --dit "$W/ltx25" --workload "$wl" --dense-stage2 \
            --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed \
            --clip "$RUNS/ltx25-$wl-gen/frames"
        gated_cell "ltx25-$wl-gen-streaming" ltx25-two-stage \
          "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload "$wl" \
            --latents "$RUNS/latents-$wl" --decoder streaming --warm \
            --clip "$RUNS/ltx25-$wl-gen-streaming/frames"
        compare_cells "ltx25-$wl-gen-streaming" "ltx25-$wl-gen"
      fi
    done
    for cell in $(ls "$RUNS" 2>/dev/null | grep -E '^(ltxvae|ltx25-.*-gen)'); do
      grep -h 'vae-bench\|vae_check\|video decode' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -3 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    ;;
  oracle)
    # GPU oracle diff (docs/oracle.md): the Python references' dumps
    # (scripts/gpu/upstream/oracle.sh on an upstream pod, served under
    # FV_ORACLE_URL) are downloaded, their noise and text conditioning
    # injected into our run (FASTVIDEO_INJECT_DIR), our run dumped the same
    # way (FASTVIDEO_DUMP_DIR), and the two compared tensor by tensor
    # (compare-dumps; report under oracle-<target>-diff/gpucheck-out).
    # FV_ORACLE_TARGETS picks targets; the default is every H3 and LTX one.
    ops="${FASTVIDEO_DUMP_OPS:-0,1,24,47}"
    h3_oracle=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder streamed
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
    )
    ltx_oracle=(--prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed)
    for target in ${FV_ORACLE_TARGETS:-fasth3-8step fasth3-4step-vsa ltx25-512p ltx25-512p-dense ltx25-4k ltx25-4k-dense}; do
      ref="$SCRATCH/oracle-ref/$target"
      if ! oracle_fetch "$target" "$ref"; then
        mkdir -p "$RUNS/oracle-$target"
        write_json "$RUNS/oracle-$target/summary.json" "$(printf '{"cell":"oracle-%s","family":"%s","exit":null,"skipped":"reference dump unavailable"}' "$target" "$FAMILY")"
        continue
      fi
      ours="$RUNS/oracle-$target-dump"
      rm -rf "$ours"
      envs=(FASTVIDEO_INJECT_DIR="$ref/dump" FASTVIDEO_DUMP_DIR="$ours" FASTVIDEO_DUMP_OPS="$ops")
      case "$target" in
        fasth3-8step)
          gated_cell "oracle-$target" fasth3-8step env "${envs[@]}" \
            "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
              --adaln-cache "$RUNS/oracle-$target-adaln.cache" \
              --clip-dir "$RUNS/oracle-$target/frames" "${h3_oracle[@]}" ;;
        fasth3-4step-vsa)
          gated_cell "oracle-$target" fasth3-4step-vsa env "${envs[@]}" \
            "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
              --adaln-cache "$RUNS/oracle-$target-adaln.cache" \
              --clip-dir "$RUNS/oracle-$target/frames" "${h3_oracle[@]}" ;;
        ltx25-*)
          geo=(--height 512 --width 768 --num-frames 121)
          [[ "$target" == ltx25-4k* ]] && geo=(--workload 4k5s)
          arm=()
          [[ "$target" == *-dense ]] && arm=(--dense-stage2)
          gated_cell "oracle-$target" ltx25-two-stage env "${envs[@]}" \
            "$BIN" --mode fast ltx2 gen --model-version 2.5 \
              --weights "$W/ltx25" --dit "$W/ltx25" "${geo[@]}" "${arm[@]}" "${ltx_oracle[@]}" \
              --clip "$RUNS/oracle-$target/frames" ;;
        *) log "unknown oracle target $target"; continue ;;
      esac
      if [[ -d "$ours" ]]; then
        run_cell "oracle-$target-diff" "$BIN" compare-dumps --baseline "$ref/dump" --candidate "$ours"
        grep -h 'compare-dumps' "$RUNS/oracle-$target-diff/stderr.log" | sed "s/^/[oracle-$target] /" | tee -a "$LOG" || true
      fi
      # The dumps are hundreds of MB each; the report keeps the numbers.
      rm -rf "$ours" "$ref"
    done
    ;;
  *)
    log "FATAL: unknown family $FAMILY"
    exit 2
    ;;
esac

log "matrix done"
write_json "$RUNS/done.json" "$(printf '{"family":"%s","ended":"%s"}' "$FAMILY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
