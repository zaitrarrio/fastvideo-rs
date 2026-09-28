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
FAMILY="${1:?usage: runpod-matrix.sh serve-engine|hd|headline|mmaudio|speechtest|h3|ltx|hunyuan|wan|b200|rtx6000|rtx5090|fastvideo|precision|precision-debug|trace|fuse|oracle|ltxvae|ltxfps|writer|eval|ltxoffload|techniques|h3arms|h3attn}"
WORK="${FV_WORK:-/workspace}"
BIN="${FV_GPUCHECK:-/opt/fastvideo-rs/target/release/fv-gpucheck}"
W="$WORK/weights"
# Everything this script writes (runs, caches, library links) goes under
# SCRATCH: the pod's container disk when FV_SCRATCH is set. The weight volume
# is only read (some hosts have silently dropped data writes to it).
SCRATCH="${FV_SCRATCH:-$WORK}"
# FV_RUNS_DIR lets one family run cells of others into its own run dir.
RUNS="${FV_RUNS_DIR:-$SCRATCH/runs/${FAMILY}${FV_RUN_TAG:+/$FV_RUN_TAG}}"
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
# Small pinned non-Hub weights (TAE, LPIPS) live on the weight volume under
# $W/auxiliary (weights-manifest.tsv auxiliary/ rows, verify-weights.sh aux).
# They are read in place when complete; otherwise the fetch scripts fill the
# container disk (volume copy per file first, then the pinned URLs).
AUX="${FV_AUX_DIR:-$W/auxiliary}"
export FV_AUX_DIR="$AUX"
if [[ -z "${FV_LPIPS_DIR:-}" && -f "$AUX/lpips/.complete" ]]; then
  LPIPS_DIR="$AUX/lpips"
else
  LPIPS_DIR="${FV_LPIPS_DIR:-$SCRATCH/lpips}"
fi
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
  # Families that call run_cell directly (b200, ...) honour FV_CELLS too.
  if [[ -n "${FV_CELLS:-}" && " $FV_CELLS " != *" $name "* ]]; then
    log "skip $name (not in FV_CELLS)"
    return 0
  fi
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

# Tiny-autoencoder weights (madebyollin/taehv): the volume copy
# ($W/auxiliary/tae, complete and hash-checked by verify-weights.sh aux) is
# read in place. Without it they live on the container disk: runpod-http's
# start command fetches them, and this fetches again only when a file is
# missing (fetch-tae.sh copies from the volume first, then downloads). TAE
# cells do not gate on verify-weights.sh; a TAE cell whose file is absent is
# recorded as skipped.
if [[ -z "${FV_TAE_DIR:-}" && -f "$AUX/tae/.complete" ]]; then
  TAE="$AUX/tae"
else
  TAE="${FV_TAE_DIR:-$SCRATCH/tae}"
fi
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
  # A prompt set's reports are <tag>-<prompt>.json, one per prompt directory
  # of the candidate's frames. Globbing <tag>-* instead would also take the
  # reports of a longer-named cell (<cand>-taeh3 against the same baseline).
  compare_reports() {
    local t="$1" c="$2" p
    [[ -f "$RUNS/compare/compare-clips-$t.json" ]] && echo "$RUNS/compare/compare-clips-$t.json"
    for p in "$RUNS/$c/frames"/*/; do
      p="$(basename "$p")"
      [[ "$p" == cold || "$p" == warmup || "$p" == "*" ]] && continue
      [[ -f "$RUNS/compare/compare-clips-$t-$p.json" ]] && echo "$RUNS/compare/compare-clips-$t-$p.json"
    done
    return 0
  }
  for f in $(compare_reports "$tag" "$cand"); do
    cmp+=(--compare "$f")
  done
  if [[ -n "$off" ]]; then
    for f in $(compare_reports "$base--$off" "$off"); do
      offs+=(--off-compare "$f")
    done
  fi
  # A pair whose cells did not run (FV_CELLS subset, skipped weights) has
  # nothing to gate; a gate there only logs a failure about missing files.
  if [[ ! -f "$RUNS/$base/benchmark.json" || ! -f "$RUNS/$cand/benchmark.json" ]]; then
    log "skip gate $tag (benchmark.json missing)"
    return 0
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

# oracle_run <cell> <dump dir> [env...]: our run of the current oracle target
# ($wcell, $cmd, $envs set by the `oracle` family) on the reference's inputs.
oracle_run() {
  local cell="$1" dump="$2"
  shift 2
  local out=(--clip-dir "$RUNS/$cell/frames" --adaln-cache "$RUNS/oracle-$target-adaln.cache")
  [[ "$target" == ltx25-* ]] && out=(--clip "$RUNS/$cell/frames")
  [[ "$target" == sfwan* ]] && out=(--clip-dir "$RUNS/$cell/frames")
  rm -rf "$dump"
  gated_cell "$cell" "$wcell" env "${envs[@]}" FASTVIDEO_DUMP_DIR="$dump" "$@" "${cmd[@]}" "${out[@]}"
}

# oracle_diff <cell> <baseline dump> <candidate dump>: compare-dumps, logged.
oracle_diff() {
  [[ -d "$2" && -d "$3" ]] || return 0
  run_cell "$1" "$BIN" compare-dumps --baseline "$2" --candidate "$3"
  grep -h 'compare-dumps' "$RUNS/$1/stderr.log" | sed "s/^/[$1] /" | tee -a "$LOG" || true
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
  mmaudio)
    # The MMAudio V2A sidecar on the video-only Wan family (docs/ports/mmaudio.md):
    #   wan13-audio / sfwan13-audio  FastWan 1.3B DMD and SF-Wan 1.3B, 480x832x81,
    #                                 warm, with `--audio mmaudio` (video, audio,
    #                                 audio RTF, e2e, peak VRAM in benchmark.json)
    #   up-sidecar                    upstream MMAudio (Python, strobe's sidecar
    #                                 settings) on wan13-audio's clip: timing + the
    #                                 oracle dump
    #   ours-v2a                      our port on the same clip, timed
    #   mm-pixels / mm-features / mm-e2e  our oracle runs against the dump
    #                                 (reference pixels; reference features, latent
    #                                 and mel; reference noise only), compare-dumps
    #   wave-*                        waveform similarity (mmaudio_wave_compare.py)
    GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MMW="$W/mmaudio-44k-v2"
    MM_PROMPT="${FV_MMAUDIO_PROMPT:-A red fox running through tall grass at dusk. A narrator says <S>The fox cuts through the meadow at dusk.<E>
Audio: male narration, grass rustle, wind, distant strings, cricket night}"
    export FASTVIDEO_MMAUDIO_WEIGHTS="$MMW"
    TAEW="$TAE/taew2_1.safetensors"
    if [[ ! -f "$TAEW" ]]; then
      bash "$GPU_DIR/fetch_taehv.sh" "$TAE" >>"$RUNS/tae-fetch.log" 2>&1 || log "WARN: taew2_1 fetch failed"
    fi
    export FASTVIDEO_TAE_DIR="$TAE"
    FV_WEIGHTS="$W" bash "$VERIFY" mmaudio-44k-v2 >"$RUNS/mmaudio.weights.log" 2>&1 || log "WARN: mmaudio weights incomplete"
    gated_cell wan13-audio fastwan21-1.3b \
      "$BIN" --mode fast --vsa wan gen --weights "$W/fastwan21-1.3b" --prompt "$MM_PROMPT" --seed "$SEED" \
        --warm --audio mmaudio --clip-dir "$RUNS/wan13-audio/frames"
    gated_cell sfwan13-audio sfwan21-1.3b \
      "$BIN" --mode fast wan gen --weights "$W/sfwan21-1.3b" --preset sf_wan_t2v_1_3b --steps 4 --flow-shift 5 \
        --num-frames 81 --prompt "$MM_PROMPT" --seed "$SEED" --warm --audio mmaudio \
        --clip-dir "$RUNS/sfwan13-audio/frames"
    CLIP="$RUNS/wan13-audio/frames/output.mp4"
    [[ -f "$CLIP" ]] || CLIP="$(find "$RUNS/wan13-audio" -name output.mp4 | grep -v cold | head -1)"
    log "V2A clip: $CLIP"
    # Upstream Python (MMAudio 974010a) in a venv on the container disk.
    UPV="$SCRATCH/mm-venv"
    run_cell up-env bash -c "
      set -e
      (command -v python3 && python3 -m venv --help) >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-venv python3-pip git >/dev/null; }
      command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
      python3 -m venv $UPV
      $UPV/bin/pip install -q --upgrade pip
      $UPV/bin/pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
      $UPV/bin/pip install -q av open_clip_torch einops timm omegaconf librosa soundfile torchdiffeq colorlog requests tqdm safetensors numpy
      rm -rf $SCRATCH/MMAudio && git clone -q https://github.com/hkchengrex/MMAudio.git $SCRATCH/MMAudio
      git -C $SCRATCH/MMAudio checkout -q 974010a026c731054592d8f777218bd9d85a6c24
      $UPV/bin/python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
    "
    REF="$SCRATCH/mm-ref"
    run_cell up-sidecar env PYTHONPATH="$SCRATCH/MMAudio" "$UPV/bin/python" "$GPU_DIR/mmaudio_oracle.py" \
      --weights "$MMW" --video "$CLIP" --prompt "$MM_PROMPT" --seed 1000 --runs 3 \
      --out "$RUNS/up-sidecar" --dump "$REF"
    run_cell ours-v2a "$BIN" --mode fast mmaudio v2a --weights "$MMW" --video "$CLIP" --prompt "$MM_PROMPT" --seed 1000 \
      --runs 3 --out "$RUNS/ours-v2a"
    for arm in "pixels:pixels" "features:features,x1,mel" "e2e:"; do
      name="${arm%%:*}"; inj="${arm#*:}"
      rm -rf "$SCRATCH/mm-ours-$name"
      run_cell "mm-$name" env FASTVIDEO_DUMP_DIR="$SCRATCH/mm-ours-$name" FASTVIDEO_INJECT_DIR="$REF" \
        FASTVIDEO_MMAUDIO_INJECT="$inj" "$BIN" --mode fast mmaudio v2a --weights "$MMW" --video "$CLIP" \
        --prompt "$MM_PROMPT" --seed 1000 --runs 1 --out "$RUNS/mm-$name"
      oracle_diff "diff-$name" "$REF" "$SCRATCH/mm-ours-$name"
    done
    for pair in "e2e-vs-up:$REF/mm_wave.f32:$RUNS/mm-e2e/mmaudio.f32" \
                "e2e-vs-upf32:$REF/mm_wave_f32.f32:$RUNS/mm-e2e/mmaudio.f32" \
                "up-bf16-vs-f32:$REF/mm_wave_f32.f32:$REF/mm_wave.f32" \
                "features-vs-upf32:$REF/mm_wave_f32.f32:$RUNS/mm-features/mmaudio.f32" \
                "ours-seed-vs-up:$REF/mm_wave.f32:$RUNS/ours-v2a/mmaudio.f32"; do
      IFS=: read -r label a b <<<"$pair"
      run_cell "wave-$label" "$UPV/bin/python" "$GPU_DIR/mmaudio_wave_compare.py" --ref "$a" --cand "$b" \
        --label "$label" --out "$RUNS/wave-$label/wave.json"
    done
    ;;
  speechtest)
    # Can the audio models say a given line? (docs/ports/mmaudio.md "Speech")
    # Seeds 1000..1002 everywhere; every wav goes into $RUNS/manifest.tsv
    # (id, wav, intended line, note) and one Whisper pass transcribes them all.
    #   t2a-fox / t2a-station / t2a-fox-tagged  MMAudio text-to-audio, 8 s: a plain
    #                               quoted line, and strobe's <S>..<E> + "Audio:" style
    #   wan-talk                    FastWan 1.3B, a person speaking to camera (no audio)
    #   v2a-{notext,text}-<seed>    MMAudio video-to-audio on that clip, without and
    #                               with the spoken line as the text condition
    #   h3-speech                   FastH3 4-step VSA 480p, 5 s, joint audio, the
    #                               same lines in H3's <d>[English] ..</d> markup
    #   whisper                     openai-whisper (FV_WHISPER_MODEL, default large-v3)
    GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MMW="$W/mmaudio-44k-v2"
    export FASTVIDEO_MMAUDIO_WEIGHTS="$MMW"
    SEEDS="1000 1001 1002"
    FOX="The quick brown fox jumps over the lazy dog."
    STATION="Welcome to the station, the next train leaves at nine."
    MAN="A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions, soft window light."
    WOMAN="A woman station announcer speaking into a microphone on a train platform, medium close-up, daytime."
    MANIFEST="$RUNS/manifest.tsv"
    : >"$MANIFEST"
    add() { local n="${4//$'\n'/ | }"; printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "${n//$'\t'/ }" >>"$MANIFEST"; }
    t2a() {
      local name="$1" line="$2" prompt="$3" seed
      gated_cell "$name" mmaudio-44k-v2 "$BIN" --mode fast mmaudio t2a --weights "$MMW" --prompt "$prompt" \
        --seeds "${SEEDS// /,}" --duration 8 --out "$RUNS/$name"
      for seed in $SEEDS; do add "$name-$seed" "$RUNS/$name/seed-$seed.wav" "$line" "mmaudio t2a: $prompt"; done
    }
    t2a t2a-fox "$FOX" "A man says clearly: \"$FOX\""
    t2a t2a-station "$STATION" "A woman announces: \"$STATION\""
    t2a t2a-fox-tagged "$FOX" "A man talks to the camera. He says <S>$FOX<E>
Audio: male speech, clear voice, quiet room"
    TALK="A man in his thirties speaking to the camera in a bright living room, medium close-up, his lips moving as he talks, natural expressions and hand gestures, soft window light."
    gated_cell wan-talk fastwan21-1.3b \
      "$BIN" --mode fast --vsa wan gen --weights "$W/fastwan21-1.3b" --prompt "$TALK" --seed 1024 \
        --clip-dir "$RUNS/wan-talk/frames"
    CLIP="$(find "$RUNS/wan-talk" -name output.mp4 | grep -v cold | head -1)"
    log "V2A clip: ${CLIP:-missing}"
    for seed in $SEEDS; do
      for arm in notext text; do
        prompt=""
        [[ "$arm" == text ]] && prompt="A man says clearly: \"$FOX\""
        gated_cell "v2a-$arm-$seed" mmaudio-44k-v2 "$BIN" --mode fast mmaudio v2a --weights "$MMW" --video "$CLIP" \
          --prompt "$prompt" --seed "$seed" --runs 1 --out "$RUNS/v2a-$arm-$seed"
        add "v2a-$arm-$seed" "$RUNS/v2a-$arm-$seed/mmaudio.wav" "$FOX" "mmaudio v2a on wan-talk, prompt: ${prompt:-none}"
      done
    done
    H3P="$RUNS/h3-prompts.json"
    # (No jq in the runtime image; the lines hold no JSON-special characters.)
    {
      printf '{"name": "speechtest", "prompts": [\n'
      sep=""
      for seed in $SEEDS; do
        printf '%s{"name": "fox-%s", "seed": %s, "prompt": "%s He says clearly: <d>[English] %s</d>"},\n' "$sep" "$seed" "$seed" "$MAN" "$FOX"
        printf '{"name": "station-%s", "seed": %s, "prompt": "%s She announces: <d>[English] %s</d>"}' "$seed" "$seed" "$WOMAN" "$STATION"
        sep=$',\n'
      done
      printf '\n]}\n'
    } >"$H3P"
    gated_cell h3-speech fasth3-4step-vsa \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832 \
        --prompt "$PROMPT" --seconds 5 --seed "$SEED" --text-encoder auto --text-cache "$SCRATCH/h3-text-cache" \
        --text-weights "$W/h3-base" --warm --prompts "$H3P" \
        --adaln-cache "$RUNS/h3-speech-adaln.cache" --clip-dir "$RUNS/h3-speech/frames"
    for seed in $SEEDS; do
      add "h3-fox-$seed" "$RUNS/h3-speech/frames/fox-$seed/audio.wav" "$FOX" "h3 fasth3-4step-vsa 480p"
      add "h3-station-$seed" "$RUNS/h3-speech/frames/station-$seed/audio.wav" "$STATION" "h3 fasth3-4step-vsa 480p"
    done
    WV="$SCRATCH/whisper-venv"
    FV_GEN_TIMEOUT_S=2400 run_cell whisper bash -c "
      set -e
      (command -v python3 && python3 -m venv --help) >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null; }
      command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
      python3 -m venv $WV
      $WV/bin/pip install -q --upgrade pip
      $WV/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
      $WV/bin/pip install -q openai-whisper
      $WV/bin/python $GPU_DIR/speech_transcribe.py --manifest $MANIFEST --model ${FV_WHISPER_MODEL:-large-v3} \
        --out $RUNS/whisper/speech.json
    "
    ;;
  sfwan)
    # SF-Wan 1.3B on one pod: the oracle diff against FastVideo's dump (when
    # FV_ORACLE_URL serves one), then the wan family's SF-Wan cells and the
    # causal kernel checks.
    if [[ -n "${FV_ORACLE_URL:-}" ]]; then
      FV_RUNS_DIR="$RUNS" FV_ORACLE_TARGETS="${FV_ORACLE_TARGETS:-sfwan13}" bash "${BASH_SOURCE[0]}" oracle || true
    fi
    FV_RUNS_DIR="$RUNS" FV_CELLS="${FV_SFWAN_CELLS:-kernels-wan sfwan13-81f-flash sfwan13-81f-composed sfwan13-81f-wholeclip sfwan13-81f-fullvae}" \
      bash "${BASH_SOURCE[0]}" wan || true
    ;;
  sfstream)
    # Open-ended causal SF-Wan (wan::stream, serve E6): parity with the
    # bounded 81-frame path, long runs (drift, fps, TTFF), prompt switches,
    # and a 10-minute memory-growth run. TAEHV decodes every block.
    TAE_W="$TAE/taew2_1.safetensors"
    [[ -f "$TAE_W" ]] || bash "$(dirname "${BASH_SOURCE[0]}")/fetch_taehv.sh" "$TAE" >>"$RUNS/tae-fetch.log" 2>&1 \
      || log "WARN: taew2_1 fetch failed (tae-fetch.log)"
    export FASTVIDEO_TAE_DIR="$TAE"
    SF_PROMPT="${FV_SF_PROMPT:-A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed}"
    SF_SWITCH="${FV_SF_SWITCH:-A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed}"
    sf_stream() {
      local name="$1"; shift
      gated_cell "$name" sfwan21-1.3b "$BIN" --keep-going --mode fast wan stream --weights "$W/sfwan21-1.3b" \
        --prompt "$SF_PROMPT" --switch-prompt "$SF_SWITCH" --seed "$SEED" "$@"
    }
    # shellcheck disable=SC2086
    sf_stream sfstream-main --parity ${FV_SFSTREAM_RUNS:---run reb-sink3-120s,seconds=120,rope=rebased,sink=3 \
      --run switch-keep-30s,seconds=30,switch_at=15,switch=keep --run switch-reset-20s,seconds=20,switch_at=10,switch=reset}
    sf_stream sfstream-10min --run reb-sink3-600s,seconds=600,rope=rebased,sink=3 --window-s 30
    ;;
  headline)
    # The headline configurations on one pod (a new GPU type, one run):
    # cells borrowed from other families, all written into this run dir.
    sub() { local fam="$1"; shift
      FV_RUNS_DIR="$RUNS" FV_CELLS="$*" bash "${BASH_SOURCE[0]}" "$fam" || true; }
    sub fastvideo ${FV_HEADLINE_FASTVIDEO:-fasth3-4step-vsa-480p fasth3-8step-480p fasth3-4step-vsa-768p fasth3-8step-768p}
    sub rtx6000 ${FV_HEADLINE_SOLH3:-sol-h3}
    sub rtx5090 ${FV_HEADLINE_ROUTES:-h3-768p-fullopt ltx25-4k5s-sol}
    sub precision ${FV_HEADLINE_PRECISION:-ltx25-4k5s-sol-bf16act-fp8}
    ;;
  identity)
    # E12 identity bisect: the same generation under several environments in
    # one pod, frames hashed (sorted PNG sha256s, hashed again: the serverless
    # worker's frames_sha256), FASTVIDEO_DIGEST=1 stage digests in stderr.log.
    # FV_ID_CASES: space-separated "<cell>|<model>|K=V,K=V" (model ltx25 or
    # fasth3; "-" for no env). FV_ID_EVICT=1 drops the weights from the page
    # cache before each cell (fv-gpucheck evict-cache).
    id_hash() {
      local dir="$1" out="$2" h f
      h="$(find "$dir" -name '*.png' 2>/dev/null | sort | xargs -r sha256sum | awk '{print $1}' | sha256sum | awk '{print $1}')"
      f="$(find "$dir" -name '*.png' 2>/dev/null | sort | head -1 | xargs -r sha256sum | awk '{print $1}')"
      printf '{"frames":%s,"frames_sha256":"%s","first_frame_sha256":"%s"}\n' \
        "$(find "$dir" -name '*.png' 2>/dev/null | wc -l)" "$h" "$f" >"$out"
      log "frames $(basename "$(dirname "$out")") $h"
    }
    for spec in ${FV_ID_CASES:?FV_ID_CASES}; do
      IFS='|' read -r name model envs <<<"$spec"
      envs="${envs//,/ }"
      [[ "$envs" == "-" ]] && envs=""
      if [[ "${FV_ID_EVICT:-0}" == 1 ]]; then
        "$BIN" --out "$RUNS/evict" evict-cache "$W/h3-base" "$W/ltx25" >/dev/null 2>&1 || true
      fi
      case "$model" in
        ltx25)
          # shellcheck disable=SC2086
          gated_cell "$name" ltx25-two-stage env FASTVIDEO_DIGEST=1 $envs \
            "$BIN" --mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25" \
              --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --no-text-cache \
              --clip "$RUNS/$name/frames" ;;
        fasth3)
          # shellcheck disable=SC2086
          gated_cell "$name" fasth3-4step-vsa env FASTVIDEO_DIGEST=1 $envs \
            "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
              --adaln-cache "$RUNS/$name/adaln.cache" --clip-dir "$RUNS/$name/frames" \
              --prompt "$PROMPT" --seconds 5 --seed "$SEED" --text-encoder auto --no-text-cache \
              --text-weights "$W/h3-base" ;;
        *) log "identity: unknown model $model"; continue ;;
      esac
      id_hash "$RUNS/$name/frames" "$RUNS/$name/frames.json"
      grep -E '^\[fastvideo\] (digest|load/|ltx2 load|h3 load|llm prefetch)' "$RUNS/$name/stderr.log" \
        | grep -v 'digest w:\|digest lin:' >"$RUNS/$name/digests.txt" 2>/dev/null || true
      rm -rf "$RUNS/$name/frames"
    done
    ;;
  cold)
    # Timings measured the way the references measure them, where the warm
    # cells do not. LTX-2.5: sol-engine times one request end to end in a fresh
    # process (--offload cpu, weights loading lazily inside the stages), so
    # these cells run once, without --warm, and always encode the prompt;
    # compare load_s + total_s with its e2e. They run first, with the page
    # cache dropped where the container allows it, so load reads the volume.
    # H3: FastVideo times warm requests but encodes the prompt each time, so
    # the fresh-prompt cells keep --warm and pass --no-text-cache.
    drop_page_cache() {
      sync
      if echo 3 >/proc/sys/vm/drop_caches 2>/dev/null; then
        log "cold: page cache dropped before $1"
      else
        log "cold: page cache NOT dropped before $1 (no permission); load may read cached pages"
      fi
    }
    for wl in 4k5s 1080p20s; do
      for mode in cpu resident; do
        name="ltx25-$wl-sol-cold-$mode"
        [[ -n "${FV_CELLS:-}" && " $FV_CELLS " != *" $name "* ]] && continue
        envs=(env -u FASTVIDEO_LTX_OFFLOAD)
        [[ "$mode" == cpu ]] && envs=(env FASTVIDEO_LTX_OFFLOAD=cpu)
        drop_page_cache "$name"
        gated_cell "$name" ltx25-two-stage \
          "${envs[@]}" "$BIN" --mode fast ltx2 gen --model-version 2.5 \
            --weights "$W/ltx25" --dit "$W/ltx25" --workload "$wl" \
            --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --no-text-cache \
            --clip "$RUNS/$name/frames"
        # The 4K / 20 s PNGs are GBs; the reports keep the numbers.
        rm -rf "$RUNS/$name/frames"/*.png
      done
    done
    h3_fresh=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --no-text-cache
      --text-weights "$W/h3-base"
      --warm
    )
    for res in 768p 480p; do
      geo=()
      [[ "$res" == 480p ]] && geo=(--height 480 --width 832)
      gated_cell "fasth3-8step-$res-fresh" fasth3-8step \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step "${geo[@]}" \
          --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-8step-$res-fresh/frames" "${h3_fresh[@]}"
      gated_cell "fasth3-4step-vsa-$res-fresh" fasth3-4step-vsa \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa "${geo[@]}" \
          --adaln-cache "$RUNS/fasth3-4step-vsa-$res-adaln.cache" \
          --clip-dir "$RUNS/fasth3-4step-vsa-$res-fresh/frames" "${h3_fresh[@]}"
    done
    gated_cell sol-h3-fresh sol-h3 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3 \
        --adaln-cache "$RUNS/sol-h3-adaln.cache" \
        --clip-dir "$RUNS/sol-h3-fresh/frames" "${h3_fresh[@]}"
    gated_cell h3-768p-fullopt-fresh h3-base \
      env FASTVIDEO_H3_SOL_CACHE=teacache \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe sol-h3-rtx \
        --adaln-cache "$RUNS/h3-768p-adaln.cache" \
        --clip-dir "$RUNS/h3-768p-fullopt-fresh/frames" "${h3_fresh[@]}"
    ;;
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
    # FastWan2.1 1.3B DMD, 3 steps (1000/757/522), 480x832, 81 frames: the
    # published FastVideo recipe (VSA sparsity 0.8, guidance 1). Every cell
    # is one warm process (--warm: one untimed generation first) with UMT5,
    # DiT and decoder resident; with FV_PROMPTS=5 each cell runs the five
    # prompts of prompts-eval.json and benchmark.json holds their medians.
    # benchmark.json frames_sha256 is the byte-identity check between cells.
    wan_common=(
      --weights "$W/fastwan21-1.3b"
      --prompt "$PROMPT"
      --seed "$SEED"
      --warm
      "${PROMPT_ARGS[@]}"
    )
    # The distilled presets decode through TAEHV when taew2_1 is found
    # (FASTVIDEO_TAE_DIR): fetch it onto the container disk once.
    TAEW="$TAE/taew2_1.safetensors"
    if [[ ! -f "$TAEW" || ! -f "$TAE/taew2_2.safetensors" ]]; then
      bash "$(dirname "${BASH_SOURCE[0]}")/fetch_taehv.sh" "$TAE" >>"$RUNS/tae-fetch.log" 2>&1 \
        || log "WARN: taew2_1 fetch failed (tae-fetch.log); distilled cells decode with the Wan VAE"
    fi
    export FASTVIDEO_TAE_DIR="$TAE"
    # Baseline: VSA (the checkpoint's to_gate_compress), the default decoder.
    gated_cell wan13-dmd fastwan21-1.3b \
      "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" --clip-dir "$RUNS/wan13-dmd/frames"
    # Dense attention, same checkpoint (the gates unused).
    gated_cell wan13-dmd-dense fastwan21-1.3b \
      "$BIN" --mode fast wan gen "${wan_common[@]}" --clip-dir "$RUNS/wan13-dmd-dense/frames"
    compare_cells wan13-dmd wan13-dmd-dense
    # Kernel arms, same recipe as wan13-dmd (VSA). wan13-f32act is the
    # numerics before bf16 activations (FASTVIDEO_BF16_ACT=0: f32 residual
    # stream, unfused f32 chain); every arm is compared and gated against it.
    #   wan13-bf16act  bf16 activations, residual + norm as two kernels
    #   wan13-fuse     the same math fused (must equal wan13-bf16act byte for byte)
    #   wan13-mxfp8 / wan13-w8a8  FASTVIDEO_WAN_QUANT on top of wan13-fuse
    # kernels-wan: fv-gpucheck kernels --groups wan_fusion,wan_causal_attn.
    if [[ -z "${FV_CELLS:-}" || " $FV_CELLS " == *" kernels-wan "* ]]; then
      run_cell kernels-wan "$BIN" --out "$RUNS/kernels-wan/gpucheck-out" kernels --groups wan_fusion,wan_causal_attn
      grep -hE '^\[(PASS|FAIL)\]' "$RUNS/kernels-wan/stderr.log" 2>/dev/null | cut -c1-200 | tee -a "$LOG" || true
    fi
    wan_arm() {
      local name="$1"
      shift
      gated_cell "$name" fastwan21-1.3b env "$@" \
        "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" --clip-dir "$RUNS/$name/frames"
    }
    wan_arm wan13-f32act FASTVIDEO_BF16_ACT=0 FASTVIDEO_WAN_QUANT=off
    wan_arm wan13-bf16act FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_FUSE=0 FASTVIDEO_WAN_QUANT=off
    wan_arm wan13-fuse FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_FUSE=1 FASTVIDEO_WAN_QUANT=off
    wan_arm wan13-mxfp8 FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_FUSE=1 FASTVIDEO_WAN_QUANT=mxfp8
    wan_arm wan13-w8a8 FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_FUSE=1 FASTVIDEO_WAN_QUANT=w8a8
    for arm in wan13-bf16act wan13-fuse wan13-mxfp8 wan13-w8a8; do
      compare_cells wan13-f32act "$arm"
      gate_cells wan13-f32act "$arm" lossy
    done
    # The fusion is exact: its OFF identity is the fused/unfused pair itself.
    compare_cells wan13-bf16act wan13-fuse --off-identity
    gate_cells wan13-bf16act wan13-fuse exact wan13-fuse
    for arm in wan13-mxfp8 wan13-w8a8; do
      compare_cells wan13-fuse "$arm"
      gate_cells wan13-fuse "$arm" lossy
    done
    # SF-Wan 1.3B as FastVideo generates it: 3-frame blocks through the KV
    # cache, 4 DMD steps (1000/750/500/250 warped by the Self-Forcing
    # scheduler, shift 5), guidance 1, 480x832, 81 frames. The block
    # attention on the flash kernel (default) against its composed
    # reference (FASTVIDEO_WAN_CAUSAL_FLASH=0: f32 scores and softmax), and
    # the whole-clip masked path (FASTVIDEO_WAN_CAUSAL_AR=0).
    sf_common=(
      --weights "$W/sfwan21-1.3b"
      --preset sf_wan_t2v_1_3b
      --steps 4
      --flow-shift 5
      --prompt "$PROMPT"
      --seed "$SEED"
      --warm
      "${PROMPT_ARGS[@]}"
    )
    sf_arm() {
      local name="$1" frames="$2"
      shift 2
      gated_cell "$name" sfwan21-1.3b env "$@" \
        "$BIN" --mode fast wan gen "${sf_common[@]}" --num-frames "$frames" --clip-dir "$RUNS/$name/frames"
    }
    sf_arm sfwan13-81f-flash 81 FASTVIDEO_WAN_CAUSAL_FLASH=1
    sf_arm sfwan13-81f-composed 81 FASTVIDEO_WAN_CAUSAL_FLASH=0
    sf_arm sfwan13-81f-wholeclip 81 FASTVIDEO_WAN_CAUSAL_AR=0
    compare_cells sfwan13-81f-composed sfwan13-81f-flash
    gate_cells sfwan13-81f-composed sfwan13-81f-flash lossy
    compare_cells sfwan13-81f-flash sfwan13-81f-wholeclip
    # Base Wan2.2 TI2V-5B (704x1280x121) and Wan2.1 T2V-14B (480x832x81),
    # UniPC with guidance 5 over 12 steps (not the 50-step official recipe:
    # an A/B of the kernel arms, same schedule on every arm). f32act is the
    # pre-bf16 numerics; bf16act the default; mxfp8 FASTVIDEO_WAN_QUANT.
    base_arm() {
      local name="$1" wcell="$2" preset="$3" h="$4" w="$5" f="$6"
      shift 6
      gated_cell "$name" "$wcell" env "$@" \
        "$BIN" --mode fast wan gen --weights "$W/$wcell" --preset "$preset" \
          --unipc --steps 12 --guidance 5 --flow-shift 5 \
          --height "$h" --width "$w" --num-frames "$f" \
          --prompt "$PROMPT" --seed "$SEED" --warm "${PROMPT_ARGS[@]}" \
          --clip-dir "$RUNS/$name/frames"
    }
    for m in "wan5b wan22-ti2v-5b wan_2_2_ti2v_5b 704 1280 121" "wan14b wan21-t2v-14b wan_t2v_14b 480 832 81"; do
      read -r tag wcell preset h w f <<<"$m"
      base_arm "$tag-f32act" "$wcell" "$preset" "$h" "$w" "$f" FASTVIDEO_BF16_ACT=0 FASTVIDEO_WAN_QUANT=off
      base_arm "$tag-bf16act" "$wcell" "$preset" "$h" "$w" "$f" FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_QUANT=off
      base_arm "$tag-mxfp8" "$wcell" "$preset" "$h" "$w" "$f" FASTVIDEO_BF16_ACT=1 FASTVIDEO_WAN_QUANT=mxfp8
      for arm in "$tag-bf16act" "$tag-mxfp8"; do
        compare_cells "$tag-f32act" "$arm"
        gate_cells "$tag-f32act" "$arm" lossy
      done
    done
    # The opt-out to the full Wan VAE (2 latent frames per pass by default),
    # and the old one-frame-per-pass decode.
    gated_cell wan13-dmd-fullvae fastwan21-1.3b \
      "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" --full-vae \
        --clip-dir "$RUNS/wan13-dmd-fullvae/frames"
    gated_cell wan13-dmd-fullvae-chunk1 fastwan21-1.3b \
      "$BIN" --mode fast --vsa --vae-chunk 1 wan gen "${wan_common[@]}" --full-vae \
        --clip-dir "$RUNS/wan13-dmd-fullvae-chunk1/frames"
    # TAEHV (the default) against the full VAE: LPIPS / PSNR / sharpness.
    compare_cells wan13-dmd-fullvae wan13-dmd
    gate_cells wan13-dmd-fullvae wan13-dmd lossy
    compare_cells wan13-dmd-fullvae-chunk1 wan13-dmd-fullvae
    # Exact switches off: text K/V, text and time embeddings recomputed every
    # forward, every prompt encoded. Must be byte-identical to the baseline
    # (compare --off-identity; benchmark.json frames_sha256).
    gated_cell wan13-dmd-nocache fastwan21-1.3b \
      env FASTVIDEO_WAN_COND_CACHE=0 \
      "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" --no-text-cache \
        --clip-dir "$RUNS/wan13-dmd-nocache/frames"
    compare_cells wan13-dmd wan13-dmd-nocache --off-identity
    gate_cells wan13-dmd wan13-dmd-nocache exact
    # Lossy arms, never default; LPIPS against the baseline.
    # Sol-Attn on 1.3B (tau 1.0, layer 0 dense, Morton3D; replaces VSA).
    gated_cell wan13-dmd-sol fastwan21-1.3b \
      env FASTVIDEO_WAN_SOL_ATTN=1 \
      "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" \
        --clip-dir "$RUNS/wan13-dmd-sol/frames"
    # TeaCache4Wan2.1 1.3B (poly-rescaled rel-L1 on the latents, thresh 0.08).
    gated_cell wan13-dmd-teacache fastwan21-1.3b \
      env FASTVIDEO_TEACACHE=1 \
      "$BIN" --mode fast --vsa wan gen "${wan_common[@]}" \
        --clip-dir "$RUNS/wan13-dmd-teacache/frames"
    for arm in sol teacache; do
      compare_cells wan13-dmd "wan13-dmd-$arm"
      gate_cells wan13-dmd "wan13-dmd-$arm" lossy
    done

    # ---- Wan2.1 T2V-14B, Wan2.2 TI2V-5B, SF-Wan 1.3B (weights on the US
    # volume, fv-weights-b200-us). One prompt ($PROMPT), the upstream
    # FastVideo sampling defaults of each checkpoint. The 50-step cells run
    # one generation (no --warm): first-request overhead is small next to
    # 100 forwards, and each cell would otherwise cost twice the time.
    one=(--prompt "$PROMPT" --seed "$SEED")
    w14=(--weights "$W/wan21-t2v-14b" --preset wan_t2v_14b --unipc --guidance 5.0
      --flow-shift 3.0 "${one[@]}")
    gated_cell wan14 wan21-t2v-14b \
      "$BIN" --mode fast wan gen "${w14[@]}" --steps 50 --clip-dir "$RUNS/wan14/frames"
    # Identity of the exact caches on 14B (CFG batch, 40 blocks): 4 steps each way.
    gated_cell wan14-4step wan21-t2v-14b \
      "$BIN" --mode fast wan gen "${w14[@]}" --steps 4 --clip-dir "$RUNS/wan14-4step/frames"
    gated_cell wan14-4step-nocache wan21-t2v-14b \
      env FASTVIDEO_WAN_COND_CACHE=0 \
      "$BIN" --mode fast wan gen "${w14[@]}" --steps 4 --no-text-cache \
        --clip-dir "$RUNS/wan14-4step-nocache/frames"
    compare_cells wan14-4step wan14-4step-nocache --off-identity
    w14+=(--steps 50)
    # sol-engine config/wan21_t2v_14b/fullstack.toml: EasyCache 0.036 + Sol-Attn
    # (tau 1.0, 10 dense forwards, layer 0 dense, Morton3D); and each alone,
    # plus the Sol TeaCache preset.
    gated_cell wan14-easycache wan21-t2v-14b \
      env FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack \
      "$BIN" --mode fast wan gen "${w14[@]}" --clip-dir "$RUNS/wan14-easycache/frames"
    gated_cell wan14-teacache wan21-t2v-14b \
      env FASTVIDEO_WAN_SOL_CACHE=teacache \
      "$BIN" --mode fast wan gen "${w14[@]}" --clip-dir "$RUNS/wan14-teacache/frames"
    gated_cell wan14-sol wan21-t2v-14b \
      env FASTVIDEO_WAN_SOL_ATTN=1 \
      "$BIN" --mode fast wan gen "${w14[@]}" --clip-dir "$RUNS/wan14-sol/frames"
    gated_cell wan14-fullstack wan21-t2v-14b \
      env FASTVIDEO_WAN_SOL_ATTN=1 FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack \
      "$BIN" --mode fast wan gen "${w14[@]}" --clip-dir "$RUNS/wan14-fullstack/frames"
    for arm in easycache teacache sol fullstack; do
      compare_cells wan14 "wan14-$arm"
      gate_cells wan14 "wan14-$arm" lossy
    done
    # Wan2.2 TI2V-5B module parity against Diffusers (docs/oracle.md "Wan 2.2 TI2V-5B"),
    # first so the upstream pod serving the dump can go as soon as it is fetched:
    # the reference dump of the upstream pod's oracle:wan22-ti2v step
    # (FV_ORACLE_URL), then `wan oracle` (VAE encode/decode, the t2v and
    # i2v DiT forwards) in the production mode and the VAE again in exact
    # f32 math, each diffed with compare-dumps.
    if [[ -z "${FV_CELLS:-}" || " $FV_CELLS " == *" wan5b-oracle "* ]]; then
      target=wan22-ti2v
      ref="$SCRATCH/oracle-ref/$target"
      if oracle_fetch "$target" "$ref"; then
        gated_cell wan5b-oracle wan22-ti2v-5b \
          "$BIN" --mode fast --keep-going wan oracle --weights "$W/wan22-ti2v-5b" --reference "$ref/dump" \
            --dump-out "$RUNS/wan5b-oracle-dump" --taehv "$TAE/taew2_2.safetensors"
        oracle_diff wan5b-oracle-diff "$ref/dump" "$RUNS/wan5b-oracle-dump"
        gated_cell wan5b-oracle-exact wan22-ti2v-5b \
          "$BIN" --mode exact --keep-going wan oracle --weights "$W/wan22-ti2v-5b" --reference "$ref/dump" \
            --dump-out "$RUNS/wan5b-oracle-exact-dump" --skip-dit
        oracle_diff wan5b-oracle-exact-diff "$ref/dump" "$RUNS/wan5b-oracle-exact-dump"
        rm -rf "$ref" "$RUNS/wan5b-oracle-dump" "$RUNS/wan5b-oracle-exact-dump"
      else
        mkdir -p "$RUNS/wan5b-oracle"
        write_json "$RUNS/wan5b-oracle/summary.json" '{"cell":"wan5b-oracle","exit":null,"skipped":"reference dump unavailable"}'
      fi
    fi
    # ---- Wan2.2 TI2V-5B: the checkpoint's recommended recipe (FastVideo
    # WAN_2_2_TI2V_5B preset / Diffusers model card): 704x1280, 121 frames at
    # 24 fps, 50 UniPC steps, CFG 5, flow shift 5 (scheduler_config.json),
    # FastVideo's Chinese negative prompt. Warm, the five prompts of
    # prompts-eval.json (FV_PROMPTS=5), the full Wan 2.2 VAE.
    wan_neg_cn="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    fixtures="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures"
    ti2v=(--weights "$W/wan22-ti2v-5b" --preset wan_2_2_ti2v_5b --unipc --steps 50 --guidance 5.0
      --flow-shift 5.0 --fps 24 --negative "$wan_neg_cn" --seed "$SEED" --warm)
    gated_cell wan5b wan22-ti2v-5b \
      "$BIN" --mode fast wan gen "${ti2v[@]}" --height 704 --width 1280 --num-frames 121 \
        --prompt "$PROMPT" "${PROMPT_ARGS[@]}" --clip-dir "$RUNS/wan5b/frames"
    # TAEHV (taew2_2, opt-in for the base checkpoint) on the same recipe.
    gated_cell wan5b-taehv wan22-ti2v-5b \
      env FASTVIDEO_WAN_VAE=taehv \
      "$BIN" --mode fast wan gen "${ti2v[@]}" --height 704 --width 1280 --num-frames 121 \
        --prompt "$PROMPT" "${PROMPT_ARGS[@]}" --clip-dir "$RUNS/wan5b-taehv/frames"
    compare_cells wan5b wan5b-taehv
    # Image-to-video: the 832x480 fixture pinned to latent frame 0 (its
    # tokens at timestep 0). 480x832 because FastVideo resizes a TI2V image to
    # the 480x832 area and generates at that size; one prompt.
    ti2v_prompt="Aerial drone shot of a tropical beach: turquoise sea waves roll in and break into white foam on the sand, the camera glides slowly forward along the shoreline, bright sunny day."
    gated_cell wan5b-i2v wan22-ti2v-5b \
      "$BIN" --mode fast wan gen "${ti2v[@]}" --height 480 --width 832 --num-frames 121 \
        --image "$fixtures/ti2v-beach-832x480.jpg" --prompt "$ti2v_prompt" \
        --clip-dir "$RUNS/wan5b-i2v/frames"
    gated_cell wan5b-easycache wan22-ti2v-5b \
      env FASTVIDEO_WAN_SOL_CACHE=easycache \
      "$BIN" --mode fast wan gen "${ti2v[@]}" --height 704 --width 1280 --num-frames 121 \
        --prompt "$PROMPT" "${PROMPT_ARGS[@]}" --clip-dir "$RUNS/wan5b-easycache/frames"
    compare_cells wan5b wan5b-easycache
    # SF-Wan 81 frames: TAEHV is the distilled default (sfwan13-81f-flash);
    # the full Wan VAE opt-out on the same recipe, for the decoder A/B.
    sf_arm sfwan13-81f-fullvae 81 FASTVIDEO_WAN_VAE=full
    compare_cells sfwan13-81f-fullvae sfwan13-81f-flash
    ;;
  b200)
    # Warm B200 parity: H3 / FastH3 / LTX only. Official VAE stays the
    # default elsewhere; this family opts into TAEH3. Oxide GEMM stays off.
    unset FASTVIDEO_NVFP4_OXIDE_GEMM
    taeh3=""
    for p in "$AUX/tae/taeh3.safetensors" "$W/taeh3/taeh3.safetensors" "$W/taeh3" "$WORK/taeh3/taeh3.safetensors" "$TAE/taeh3.safetensors"; do
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
    if [[ "${FV_PRECISION_ARM:-all}" == ltx-nvfp4 ]]; then
      # NVFP4 video FFN (sol-engine nvfp4_ffn.py; profile
      # ltx2/ltx25_distill_sol_nvfp4) against the bf16 default
      # (ltx2/ltx25_distill_sol): Sol stage 2, same prompt / seed, warm
      # process, every run encoding its prompt (no text cache, so both arms
      # feed the DiT the same contexts). FV_NVFP4_WORKLOADS picks workloads.
      for wl in ${FV_NVFP4_WORKLOADS:-4k5s 1080p20s}; do
        for v in bf16 nvfp4; do
          prof=ltx2/ltx25_distill_sol
          [[ "$v" == nvfp4 ]] && prof=ltx2/ltx25_distill_sol_nvfp4
          gated_cell "ltx25-$wl-sol-$v" ltx25-two-stage \
            "$BIN" --techniques "$prof" --mode fast ltx2 gen --model-version 2.5 \
              --weights "$W/ltx25" --dit "$W/ltx25" --workload "$wl" \
              --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
              --no-text-cache --clip "$RUNS/ltx25-$wl-sol-$v/frames" "${PROMPT_ARGS[@]}"
        done
        compare_cells "ltx25-$wl-sol-bf16" "ltx25-$wl-sol-nvfp4"
        gate_cells "ltx25-$wl-sol-bf16" "ltx25-$wl-sol-nvfp4" lossy
        rm -rf "$RUNS/ltx25-$wl-sol-bf16/frames" "$RUNS/ltx25-$wl-sol-nvfp4/frames"
      done
      log "matrix done"
      write_json "$RUNS/done.json" "$(printf '{"family":"%s","ended":"%s"}' "$FAMILY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
      exit 0
    fi
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
        f32act) envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off) ;;
        bf16act) envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off) ;;
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
    for v in f32act bf16act fp8 bf16act-fp8; do
      envs=()
      case "$v" in
        f32act) envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off) ;;
        bf16act) envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off) ;;
        fp8) envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_FP8=1 FASTVIDEO_H3_QUANT=off) ;;
        bf16act-fp8) envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_FP8=1 FASTVIDEO_H3_QUANT=off) ;;
      esac
      gated_cell "ltx25-4k5s-sol-$v" ltx25-two-stage \
        env "${envs[@]}" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 4k5s \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-4k5s-sol-$v/frames" "${PROMPT_ARGS[@]}"
    done
    for v in bf16act fp8 bf16act-fp8; do
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
        f32act) envs+=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off) ;;
        fp8) envs+=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_FP8=1 FASTVIDEO_H3_QUANT=off) ;;
        bf16act) envs+=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off) ;;
        bf16act-fp8) envs+=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_FP8=1 FASTVIDEO_H3_QUANT=off) ;;
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
        f32act) envs+=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off) ;;
        bf16act) envs+=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off) ;;
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
    # The long-sequence Sol workload (481 frames), same traced step.
    gated_cell ltx25-1080p20s-sol ltx25-two-stage \
      env FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_STEP=9 \
      "$BIN" --mode fast ltx2 gen --model-version 2.5 \
        --weights "$W/ltx25" --dit "$W/ltx25" --workload 1080p20s \
        --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
        --clip "$RUNS/ltx25-1080p20s-sol/frames"
    for cell in fasth3-4step-vsa-768p fasth3-8step-768p fasth3-4step-dense-768p ltx25-4k5s-sol ltx25-1080p20s-sol; do
      grep -h '/gpu_trace ' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -1 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    ;;
  attn3)
    # sm_120 dense attention A/B (FVID-2026-09-27-attention-sm120): flash_mma_fwd2
    # vs cuDNN's unified SDPA node, FastH3 4-step dense 768p denoise and LTX-2.5
    # 1080p 20 s dense stage 2, same prompt/seed, then the frame diff.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    for arm in ${FV_ATTN3_ARMS:-v2 cudnn}; do
      gated_cell "fasth3-4step-dense-768p-$arm" fasth3-4step-dense \
        env FASTVIDEO_FLASH_KERNEL="$arm" \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-dense \
          --adaln-cache "$RUNS/fasth3-4step-dense-768p-adaln.cache" \
          --clip-dir "$RUNS/fasth3-4step-dense-768p-$arm/frames" "${h3_common[@]}"
    done
    for arm in ${FV_ATTN3_ARMS:-v2 cudnn}; do
      gated_cell "ltx25-1080p20s-dense-$arm" ltx25-two-stage \
        env FASTVIDEO_FLASH_KERNEL="$arm" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 1080p20s --dense-stage2 \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-1080p20s-dense-$arm/frames"
    done
    # VSA on bf16 activations read in place (FASTVIDEO_VSA_BF16) vs the f32
    # widening path: the frames must match bit for bit.
    for vb in ${FV_ATTN3_VSA:-0 1}; do
      gated_cell "fasth3-4step-vsa-768p-b16$vb" fasth3-4step-vsa \
        env FASTVIDEO_VSA_BF16="$vb" \
        "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
          --adaln-cache "$RUNS/fasth3-4step-vsa-768p-adaln.cache" \
          --clip-dir "$RUNS/fasth3-4step-vsa-768p-b16$vb/frames" "${h3_common[@]}"
    done
    compare_cells fasth3-4step-dense-768p-v2 fasth3-4step-dense-768p-cudnn
    compare_cells ltx25-1080p20s-dense-v2 ltx25-1080p20s-dense-cudnn
    compare_cells fasth3-4step-vsa-768p-b160 fasth3-4step-vsa-768p-b161
    ;;
  fuse)
    # Phase 3c DiT block fusions (FASTVIDEO_H3_FUSE, FASTVIDEO_LTX_FUSE).
    # 1. kernels-fuse: `fv-gpucheck kernels --groups dit_fusion,bf16_act`,
    #    every fused kernel bit for bit against its unfused op chain, plus
    #    fused / unfused timings at production shapes.
    # 2. trace-*: a CUPTI-traced warm step of each workload with the fusions
    #    on (the trace family's command lines; compare with a trace run of
    #    the previous build).
    # 3. <workload>-off / -on: untraced warm generations with the fusions off
    #    (the ops the previous build runs) and on. The fusions keep every
    #    rounding point, so compare-clips must find the clips byte-identical
    #    (--off-identity) and the gate runs as an `exact` switch.
    run_cell kernels-fuse "$BIN" --out "$RUNS/kernels-fuse/gpucheck-out" kernels --groups dit_fusion,bf16_act
    grep -hE '^\[(PASS|FAIL)\]' "$RUNS/kernels-fuse/stderr.log" 2>/dev/null | cut -c1-200 | tee -a "$LOG" || true
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    off=(FASTVIDEO_H3_FUSE=0 FASTVIDEO_LTX_FUSE=0 FASTVIDEO_SPLIT_ROWS=0)
    fuse_h3() {
      local name="$1" wcell="$2" weights="$3" recipe="$4"
      shift 4
      gated_cell "$name" "$wcell" \
        env "$@" \
        "$BIN" --mode fast h3 gen --weights "$W/$weights" --h3-recipe "$recipe" \
          --adaln-cache "$RUNS/$name-adaln.cache" \
          --clip-dir "$RUNS/$name/frames" "${h3_common[@]}"
    }
    fuse_ltx() {
      local name="$1"
      shift
      gated_cell "$name" ltx25-two-stage \
        env "$@" \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 4k5s \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/$name/frames"
    }
    fuse_h3 trace-fasth3-8step-768p fasth3-8step h3-8step 8step FASTVIDEO_GPU_TRACE=1
    fuse_h3 trace-fasth3-4step-vsa-768p fasth3-4step-vsa h3-base 4step-vsa FASTVIDEO_GPU_TRACE=1
    fuse_ltx trace-ltx25-4k5s-sol FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_STEP=9
    for cell in trace-fasth3-8step-768p trace-fasth3-4step-vsa-768p trace-ltx25-4k5s-sol; do
      grep -h '/gpu_trace ' "$RUNS/$cell/stderr.log" 2>/dev/null | tail -1 | cut -c1-400 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    fuse_h3 fasth3-8step-768p-off fasth3-8step h3-8step 8step "${off[@]}"
    fuse_h3 fasth3-8step-768p-on fasth3-8step h3-8step 8step FASTVIDEO_H3_FUSE=1
    fuse_h3 fasth3-4step-vsa-768p-off fasth3-4step-vsa h3-base 4step-vsa "${off[@]}"
    fuse_h3 fasth3-4step-vsa-768p-on fasth3-4step-vsa h3-base 4step-vsa FASTVIDEO_H3_FUSE=1
    fuse_ltx ltx25-4k5s-sol-off "${off[@]}"
    fuse_ltx ltx25-4k5s-sol-on FASTVIDEO_LTX_FUSE=1
    for wl in fasth3-8step-768p fasth3-4step-vsa-768p ltx25-4k5s-sol; do
      compare_cells "$wl-off" "$wl-on" --off-identity
      # `exact`: the OFF-identity report is the on/off one itself (the
      # fused path must equal the unfused one byte for byte).
      gate_cells "$wl-off" "$wl-on" exact "$wl-on"
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
      envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off)
      [[ "$v" == f32act* ]] && envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off)
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
      envs=(FASTVIDEO_BF16_ACT=1 FASTVIDEO_H3_QUANT=off)
      [[ "$v" == f32act ]] && envs=(FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off)
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
  techniques)
    # The technique composition layer (docs/techniques.md) against the build
    # before it. Each cell runs three times on this card:
    #   <cell>-base: the baseline binary (FV_BASELINE_SHA's runtime image,
    #                scripts/gpu/fetch-baseline.sh) with the legacy flags;
    #   <cell>-env:  this build, the same command line;
    #   <cell>-prof: this build, the matching --techniques profile.
    # compare-clips --off-identity: base vs env and base vs prof must be
    # byte-identical; benchmark.json carries the timings of each arm.
    # FV_TECH_CELLS narrows the cells (default: all four).
    base_sha="${FV_BASELINE_SHA:?techniques family needs FV_BASELINE_SHA (the pre-refactor commit)}"
    BASE="$SCRATCH/baseline-$base_sha/fv-gpucheck"
    if ! bash "$(dirname "${BASH_SOURCE[0]}")/fetch-baseline.sh" "$base_sha" "$(dirname "$BASE")" >>"$RUNS/baseline-fetch.log" 2>&1; then
      log "FATAL: baseline binary sha-$base_sha unavailable (baseline-fetch.log)"
      tee -a "$LOG" <"$RUNS/baseline-fetch.log"
      exit 2
    fi
    log "baseline $(tail -1 "$RUNS/baseline-fetch.log")"
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder auto
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
    )
    tech_cells="${FV_TECH_CELLS:-fasth3-8step-768p fasth3-4step-vsa-768p h3-480p-fullopt ltx25-512p-sol}"
    want() { [[ " $tech_cells " == *" $1 "* ]]; }
    # tech_h3 <cell> <weight cell> <weights dir> <profile> <env for base/env arms> -- <recipe args>
    tech_h3() {
      local cell="$1" wcell="$2" weights="$3" profile="$4" envs="$5" prof_envs="$6"
      shift 6
      local arm bin extra
      for arm in base env prof; do
        bin="$BIN"; extra=("$@")
        [[ "$arm" == base ]] && bin="$BASE"
        # shellcheck disable=SC2206
        local e=($envs)
        if [[ "$arm" == prof ]]; then
          # shellcheck disable=SC2206
          e=($prof_envs)
          extra=(--techniques "$profile")
          # Keep the geometry arguments (everything after the recipe pair).
          local a skip=0
          for a in "$@"; do
            if (( skip )); then skip=0; continue; fi
            [[ "$a" == --h3-recipe ]] && { skip=1; continue; }
            extra+=("$a")
          done
        fi
        gated_cell "$cell-$arm" "$wcell" \
          env ${e[@]+"${e[@]}"} \
          "$bin" --mode fast h3 gen --weights "$W/$weights" "${extra[@]}" \
            --adaln-cache "$RUNS/$cell-adaln.cache" \
            --clip-dir "$RUNS/$cell-$arm/frames" "${h3_common[@]}"
      done
      compare_cells "$cell-base" "$cell-env" --off-identity
      compare_cells "$cell-base" "$cell-prof" --off-identity
    }
    if want fasth3-8step-768p; then
      tech_h3 fasth3-8step-768p fasth3-8step h3-8step h3/fasth3_8step "" "" --h3-recipe 8step
    fi
    if want fasth3-4step-vsa-768p; then
      tech_h3 fasth3-4step-vsa-768p fasth3-4step-vsa h3-base h3/fasth3_4step_vsa "" "" --h3-recipe 4step-vsa
    fi
    if want h3-480p-fullopt; then
      # The rtx5090 family's fullopt cell (MXFP8 by default on this card).
      # The profile mirrors sol-engine's BF16 config, so its arm keeps the
      # matrix cell's precision with the env override FASTVIDEO_H3_QUANT=mxfp8
      # (env flags override a profile).
      tech_h3 h3-480p-fullopt h3-base h3-base h3/rtx5090_fullopt \
        "FASTVIDEO_H3_SOL_CACHE=teacache" "FASTVIDEO_H3_QUANT=mxfp8" \
        --h3-recipe sol-h3-rtx --height 480 --width 832
    fi
    if want ltx25-512p-sol; then
      # The rtx5090 family's 512p Sol cell; -prof runs it from the
      # ltx2/ltx25_distill_sol profile (Sol stage 2 from the profile).
      # Every arm encodes its prompt (--no-text-cache): a conditioning-cache
      # hit feeds the DiT the stored f32 connector output instead of the
      # freshly encoded device tensor, which is not the same clip, so a
      # shared cache would compare hit against miss. -cache is this build
      # with the cache on (its warmup writes, its timed pass hits), compared
      # with -env to show exactly that.
      for arm in base env prof cache; do
        bin="$BIN"; extra=(); tc=(--no-text-cache)
        [[ "$arm" == base ]] && bin="$BASE"
        [[ "$arm" == prof ]] && extra=(--techniques ltx2/ltx25_distill_sol)
        [[ "$arm" == cache ]] && tc=(--text-cache "$SCRATCH/ltx2-text-cache-$arm")
        gated_cell "ltx25-512p-sol-$arm" ltx25-two-stage \
          "$bin" ${extra[@]+"${extra[@]}"} --mode fast ltx2 gen --model-version 2.5 \
            --weights "$W/ltx25" --dit "$W/ltx25" --height 512 --width 768 --num-frames 121 \
            --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm "${tc[@]}" \
            --clip "$RUNS/ltx25-512p-sol-$arm/frames"
      done
      compare_cells ltx25-512p-sol-base ltx25-512p-sol-env --off-identity
      compare_cells ltx25-512p-sol-base ltx25-512p-sol-prof --off-identity
      compare_cells ltx25-512p-sol-env ltx25-512p-sol-cache
    fi
    for f in "$RUNS"/compare/compare-clips-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
    for c in "$RUNS"/*-base "$RUNS"/*-env "$RUNS"/*-prof; do
      [[ -d "$c" ]] || continue
      log "$(basename "$c") $(grep -ho '"denoise_s":[0-9.]*\|"total_s":[0-9.]*' "$c/stderr.log" 2>/dev/null | tail -2 | tr '\n' ' ')"
    done
    ;;
  h3arms)
    # FastH3 8-step technique arms on the technique layer (docs/techniques.md),
    # each gated against the plain FastH3 8-step cell of the same run:
    #   -sol     profiles/h3/fasth3_8step_sol: Sol-Attn on every step and block
    #   -tea     profiles/h3/fasth3_8step_teacache: TeaCache free to reuse the
    #            middle steps 3-4 of 8 (threshold 1.0, retain 3, cooldown 3)
    #   -soltea  both
    # at 768p and 480p; at 480p also each arm decoded by TAEH3 (compared with
    # its own twin, isolating the decoder, and with the plain baseline), and
    # FastH3 4-step VSA 480p against its TAEH3 twin. Meant for FV_PROMPTS=5
    # and FV_LPIPS=1 (the gate reads the per-prompt reports). The text encoder
    # streams: the prompts are encoded once, into the shared cache.
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder streamed
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
      "${PROMPT_ARGS[@]}"
    )
    arm_profile() {
      case "$1" in
        sol) echo h3/fasth3_8step_sol ;;
        tea) echo h3/fasth3_8step_teacache ;;
        soltea) echo h3/fasth3_8step_sol_teacache ;;
      esac
    }
    for res in 768p 480p; do
      geo=()
      [[ "$res" == 480p ]] && geo=(--height 480 --width 832)
      base="fasth3-8step-$res"
      gated_cell "$base" fasth3-8step \
        "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step "${geo[@]}" \
          --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
          --clip-dir "$RUNS/$base/frames" "${h3_common[@]}"
      for arm in sol tea soltea; do
        gated_cell "$base-$arm" fasth3-8step \
          "$BIN" --techniques "$(arm_profile "$arm")" --mode fast h3 gen --weights "$W/h3-8step" "${geo[@]}" \
            --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
            --clip-dir "$RUNS/$base-$arm/frames" "${h3_common[@]}"
        grep -h "h3 teacache step" "$RUNS/$base-$arm/stderr.log" 2>/dev/null | grep -c REUSE \
          | sed "s/^/[$base-$arm] teacache reused steps (all passes): /" | tee -a "$LOG" || true
      done
      if [[ "$res" == 480p ]]; then
        for arm in sol tea soltea; do
          tae_gated_cell "$base-$arm-taeh3" "$TAEH3" fasth3-8step \
            "$BIN" --techniques "$(arm_profile "$arm")" --mode fast h3 gen --weights "$W/h3-8step" "${geo[@]}" \
              --taeh3-weights "$TAEH3" \
              --adaln-cache "$RUNS/fasth3-8step-$res-adaln.cache" \
              --clip-dir "$RUNS/$base-$arm-taeh3/frames" "${h3_common[@]}"
        done
      fi
      for arm in sol tea soltea; do
        compare_cells "$base" "$base-$arm"
        gate_cells "$base" "$base-$arm" lossy
      done
      if [[ "$res" == 480p ]]; then
        for arm in sol tea soltea; do
          compare_cells "$base-$arm" "$base-$arm-taeh3"
          compare_cells "$base" "$base-$arm-taeh3"
          gate_cells "$base" "$base-$arm-taeh3" lossy
        done
      fi
    done
    gated_cell fasth3-4step-vsa-480p fasth3-4step-vsa \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832 \
        --adaln-cache "$RUNS/fasth3-4step-vsa-480p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-vsa-480p/frames" "${h3_common[@]}"
    tae_gated_cell fasth3-4step-vsa-480p-taeh3 "$TAEH3" fasth3-4step-vsa \
      "$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832 \
        --taeh3-weights "$TAEH3" \
        --adaln-cache "$RUNS/fasth3-4step-vsa-480p-adaln.cache" \
        --clip-dir "$RUNS/fasth3-4step-vsa-480p-taeh3/frames" "${h3_common[@]}"
    compare_cells fasth3-4step-vsa-480p fasth3-4step-vsa-480p-taeh3
    gate_cells fasth3-4step-vsa-480p fasth3-4step-vsa-480p-taeh3 lossy
    for f in "$RUNS"/gate/gate-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -o '"verdict": "[a-z]*"' "$f" | head -1)"
    done
    ;;
  h3attn)
    # H3 attention arms (docs/techniques.md "H3 attention arms"): each arm is
    # a profiles/h3/*.toml, gated (lossy) against the recipe's plain profile
    # from the same process. All arms of one recipe run in ONE process
    # (`h3 gen --arm NAME=PROFILE`): the weights load once, the first arm is
    # the baseline, and every arm runs the whole prompt set warm.
    #   attn-fp8-kernels   fv-gpucheck kernels --groups attn_fp8 (FP8 vs bf16
    #                      parity + timing); on a FAIL the *fp8attn arms are dropped
    #   fasth3-8step-768p  VSA sparsity sweep / schedules, milder Sol routes, FP8
    #   fasth3-4step-vsa-768p  VSA sparsity sweep / schedule, FP8
    #   sol-h3-768p        Sol-H3 4-step engine route tuning, FP8 dense
    #   h3-480p-fullopt    sol-h3-rtx fullopt route tuning, FP8 dense (480p:
    #                      49 forwards at 768p cost ~10 min per arm)
    # FV_CELLS picks cells; FV_H3ATTN_ARMS_<CELL> (dashes as underscores)
    # (comma-separated) overrides a cell's arm list; FV_H3ATTN_RES=480p runs
    # every cell at 480p (the smoke run).
    export FV_GEN_TIMEOUT_S="${FV_ARM_TIMEOUT_S:-7200}"
    h3_common=(
      --prompt "$PROMPT"
      --seconds 5
      --seed "$SEED"
      --text-encoder streamed
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
      --warm
      "${PROMPT_ARGS[@]}"
    )
    fp8_ok=1
    if [[ -z "${FV_CELLS:-}" || " $FV_CELLS " == *" attn-fp8-kernels "* ]]; then
      run_cell attn-fp8-kernels \
        "$BIN" --out "$RUNS/attn-fp8-kernels" --keep-going kernels --groups attn_fp8
      if grep -q '\[FAIL\]' "$RUNS/attn-fp8-kernels/stderr.log" 2>/dev/null \
        || ! grep -q 'attn_fp8_dense' "$RUNS/attn-fp8-kernels/stderr.log" 2>/dev/null; then
        fp8_ok=0
        log "attn_fp8 parity did not pass: the fp8attn arms are dropped"
      fi
      grep -hE 'attn_fp8_(dense|vsa)' "$RUNS/attn-fp8-kernels/stderr.log" 2>/dev/null | cut -c1-400 | tee -a "$LOG" || true
    fi
    # arms_cell <cell> <weight cell> <weights dir> <base profile> <geometry> <arm>...
    arms_cell() {
      local cell="$1" wcell="$2" weights="$3" base="$4" geo="$5"
      shift 5
      local var="FV_H3ATTN_ARMS_${cell//-/_}" arms=() args=() a
      # shellcheck disable=SC2206
      if [[ -n "${!var:-}" ]]; then arms=(${!var//,/ }); else arms=("$@"); fi
      [[ "${FV_H3ATTN_RES:-}" == 480p ]] && geo="--height 480 --width 832"
      for a in "${arms[@]}"; do
        [[ "$a" == *fp8attn* && "$fp8_ok" != 1 ]] && continue
        args+=(--arm "$a=h3/$a")
      done
      # shellcheck disable=SC2086
      gated_cell "$cell" "$wcell" \
        env ${ARM_ENV:-} "$BIN" --techniques "h3/$base" --mode fast h3 gen --weights "$W/$weights" \
          $geo \
          --adaln-cache "$RUNS/$cell-adaln.cache" \
          --arm "base=h3/$base" "${args[@]}" \
          --clip-dir "$RUNS/$cell-{arm}/frames" "${h3_common[@]}"
      for a in "${arms[@]}"; do
        [[ -d "$RUNS/$cell-$a" ]] || continue
        compare_cells "$cell-base" "$cell-$a"
        gate_cells "$cell-base" "$cell-$a" lossy
      done
      for a in base "${arms[@]}"; do
        [[ -f "$RUNS/$cell-$a/benchmark.json" ]] || continue
        log "$cell-$a $(grep -oE '"(denoise_s|total_s)": *[0-9.]+' "$RUNS/$cell-$a/benchmark.json" | head -2 | tr '\n' ' ')"
      done
    }
    arms_cell fasth3-8step-768p fasth3-8step h3-8step fasth3_8step "" \
      fasth3_8step_vsa085 fasth3_8step_vsa09 fasth3_8step_vsa0925 fasth3_8step_vsa095 \
      fasth3_8step_vsa09_edges fasth3_8step_vsa095_edges \
      fasth3_8step_sol_vsa_d2 fasth3_8step_sol_vsa_d2_t05 fasth3_8step_sol_d2 \
      fasth3_8step_fp8attn
    arms_cell fasth3-4step-vsa-768p fasth3-4step-vsa h3-base fasth3_4step_vsa "" \
      fasth3_4step_vsa0925 fasth3_4step_vsa095 fasth3_4step_vsa095_edges fasth3_4step_vsa_fp8attn
    arms_cell sol-h3-768p sol-h3 h3-base sol_h3_4step "" \
      sol_h3_4step_engine sol_h3_4step_engine_t075 sol_h3_4step_engine_ladder sol_h3_4step_fp8attn
    # The rtx5090 profiles are sol-engine's BF16 configs; the matrix cell runs
    # this runtime's MXFP8 default (the env flag overrides every arm alike).
    ARM_ENV=FASTVIDEO_H3_QUANT=mxfp8 arms_cell h3-480p-fullopt h3-base h3-base rtx5090_fullopt "--height 480 --width 832" \
      rtx5090_fullopt_d6 rtx5090_fullopt_t125 rtx5090_fullopt_fp8attn
    for f in "$RUNS"/gate/gate-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -o '"verdict": "[a-z]*"' "$f" | head -1)"
    done
    ;;
  ltxfps)
    # Serve E4: LTX-2.5 distilled two-stage at 1080p (the 1920x1088 engine
    # canvas) at the frame rates the APIs accept beyond 24. Each cell checks
    # the frame count and the mp4's rate and audio track with ffprobe
    # (gen.mp4_frames_and_rate, benchmark.json "mp4_probe"). The 25 fps cell
    # and the silent one (--skip-audio-decode) also run serve E2's frame sink
    # beside the PNG writer and check its frames byte for byte against the
    # PNGs of the same run (gen.sink_frames_identical). fasth3-8step-sink does
    # the same for H3 (FV_SINK_CHECK=1).
    frames="${FV_LTXFPS_FRAMES:-121}"
    for fps in ${FV_LTXFPS:-25 48 50}; do
      extra=()
      [[ "$fps" == 25 ]] && extra=(--sink-check)
      gated_cell "ltx25-1080p-${fps}fps" ltx25-two-stage \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 1080p20s \
          --num-frames "$frames" --frame-rate "$fps" \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed \
          --clip "$RUNS/ltx25-1080p-${fps}fps/frames" "${extra[@]}"
    done
    gated_cell ltx25-1080p-50fps-silent ltx25-two-stage \
      "$BIN" --mode fast ltx2 gen --model-version 2.5 \
        --weights "$W/ltx25" --dit "$W/ltx25" --workload 1080p20s \
        --num-frames "$frames" --frame-rate 50 --skip-audio-decode --sink-check \
        --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed \
        --clip "$RUNS/ltx25-1080p-50fps-silent/frames"
    gated_cell fasth3-8step-sink fasth3-8step \
      env FV_SINK_CHECK=1 \
      "$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step \
        --prompt "$PROMPT" --seconds 5 --seed "$SEED" --text-encoder auto \
        --text-cache "$SCRATCH/h3-text-cache" --text-weights "$W/h3-base" \
        --clip-dir "$RUNS/fasth3-8step-sink/frames"
    for cell in $(ls "$RUNS" 2>/dev/null | grep -E '^(ltx25-1080p|fasth3-8step-sink)'); do
      grep -h 'PASS\|FAIL' "$RUNS/$cell/stderr.log" 2>/dev/null | grep -E 'sink|mp4_frames|no_wav|gen\.frames' \
        | cut -c1-300 | sed "s/^/[$cell] /" | tee -a "$LOG" || true
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
      # Untraced timings (CUPTI adds per-launch cost), and the fast decoder
      # with every cuDNN algorithm timed per shape during the warm-up.
      if [[ "${FV_VAE_UNTRACED:-1}" == 1 ]]; then
        for dec in fast streaming; do
          gated_cell "ltxvae-$wl-$dec-untraced" ltx25-two-stage \
            "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload "$wl" \
              --decoder "$dec" --warm
        done
        # With gen's writer (PNG frames + ffmpeg mp4) behind the drain, as in
        # a generation: decode_video_s then includes writer backpressure.
        gated_cell "ltxvae-$wl-fast-mp4" ltx25-two-stage \
          "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload "$wl" \
            --decoder fast --warm --mp4 --clip "$RUNS/ltxvae-$wl-fast-mp4/frames"
        gated_cell "ltxvae-$wl-fast-tune" ltx25-two-stage \
          env FASTVIDEO_LTX_VAE_CONV_ALGO=tune \
          "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload "$wl" \
            --decoder fast --warm
      fi
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
      # The references run bf16 without FP8: pin our H3 default (MXFP8) off.
      envs=(FASTVIDEO_INJECT_DIR="$ref/dump" FASTVIDEO_DUMP_OPS="$ops" FASTVIDEO_H3_QUANT=off)
      case "$target" in
        fasth3-8step | fasth3-8step-vsa0)
          wcell=fasth3-8step
          cmd=("$BIN" --mode fast h3 gen --weights "$W/h3-8step" --h3-recipe 8step "${h3_oracle[@]}")
          [[ "$target" == *-vsa0 ]] && envs+=(FASTVIDEO_VSA_SPARSITY=0) ;;
        fasth3-4step-dense | fasth3-4step-vsa)
          wcell="$target"
          cmd=("$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe "${target#fasth3-}" "${h3_oracle[@]}") ;;
        h3-ref2va-*)
          # MiniMax-H3 Ref2VA (docs/ports/h3-ref2v.md) against FastVideo's
          # Ref2VA pipeline: transformer_ref from h3-ref2va, one image
          # reference, dense, `base-<N>step` = the reference's --steps N+1.
          # Prompt and image as scripts/gpu/upstream/oracle.sh oracle_ref2va.
          wcell=h3-ref2va
          ref2va_prompt="${FV_REF2VA_PROMPT:-The camera glides slowly forward along the shoreline of the beach in <Picture 1>, turquoise waves rolling in and breaking into white foam, bright sunny day, the sound of the surf and a light wind.}"
          cmd=("$BIN" --mode fast h3 gen --weights "$W/h3-base" --ref-root "$W/h3-ref2va"
            --h3-recipe "base-${target#h3-ref2va-}" --dense
            --ref "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures/ti2v-beach-832x480.jpg"
            --prompt "$ref2va_prompt" --seconds 5 --seed "$SEED" --text-encoder streamed
            --text-cache "$SCRATCH/h3-text-cache" --text-weights "$W/h3-base") ;;
        ltx25-*)
          wcell=ltx25-two-stage
          geo=(--height 512 --width 768 --num-frames 121)
          [[ "$target" == ltx25-4k* ]] && geo=(--workload 4k5s)
          arm=()
          [[ "$target" == *-dense ]] && arm=(--dense-stage2)
          cmd=("$BIN" --mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25"
            "${geo[@]}" "${arm[@]}" "${ltx_oracle[@]}") ;;
        sfwan13)
          # FastVideo SF-Wan 1.3B at its defaults; the full Wan VAE (the
          # reference decodes with it) for the frame metrics.
          wcell=sfwan21-1.3b
          envs+=(FASTVIDEO_WAN_VAE=full)
          cmd=("$BIN" --mode fast wan gen --weights "$W/sfwan21-1.3b" --preset sf_wan_t2v_1_3b
            --steps 4 --flow-shift 5 --prompt "$PROMPT" --seed "$SEED") ;;
        *) log "unknown oracle target $target"; continue ;;
      esac
      oracle_run "oracle-$target" "$ours"
      oracle_diff "oracle-$target-diff" "$ref/dump" "$ours"
      # The bf16 noise floor (FV_ORACLE_F32, default on for H3): ours with f32
      # activations against the reference, and our bf16 run against our f32
      # run -- how far bf16 rounding alone moves the same pipeline.
      if [[ "${FV_ORACLE_F32:-auto}" == 1 || ( "${FV_ORACLE_F32:-auto}" == auto && ( "$target" == fasth3-* || "$target" == sfwan* ) ) ]]; then
        oracle_run "oracle-$target-f32" "$ours-f32" FASTVIDEO_BF16_ACT=0 FASTVIDEO_H3_QUANT=off
        oracle_diff "oracle-$target-f32-diff" "$ref/dump" "$ours-f32"
        oracle_diff "oracle-$target-bf16-vs-f32" "$ours-f32" "$ours"
      fi
      # FV_ORACLE_OWN_TEXT=1 (LTX): ours again on our own text contexts
      # (FASTVIDEO_INJECT_TEXT=0, noise still injected) -- the end-to-end
      # effect of the text path -- against the reference and against the
      # text-injected run.
      if [[ "${FV_ORACLE_OWN_TEXT:-0}" == 1 && "$target" == ltx25-* ]]; then
        oracle_run "oracle-$target-owntext" "$ours-owntext" FASTVIDEO_INJECT_TEXT=0
        oracle_diff "oracle-$target-owntext-diff" "$ref/dump" "$ours-owntext"
        oracle_diff "oracle-$target-owntext-vs-injected" "$ours" "$ours-owntext"
      fi
      if [[ "$target" == sfwan* ]]; then
        # Frames: the reference's mp4 (its full VAE, bf16 decode, then its
        # encoder) against our PNG frames; our own mp4 against our PNGs is
        # the codec floor of that comparison.
        for d in "$ref/frames" "$RUNS/oracle-$target/mp4frames"; do mkdir -p "$d"; done
        if [[ -f "$ref/dump/ref.mp4" ]]; then
          ffmpeg -loglevel error -y -i "$ref/dump/ref.mp4" -start_number 0 "$ref/frames/frame-%03d.png" || true
        fi
        m="$(ls "$RUNS/oracle-$target"/frames/*.mp4 2>/dev/null | head -1)"
        [[ -n "$m" ]] && ffmpeg -loglevel error -y -i "$m" -start_number 0 "$RUNS/oracle-$target/mp4frames/frame-%03d.png"
        compare_one "oracle-$target-frames-vs-ref" "$ref/frames" "$RUNS/oracle-$target/frames"
        compare_one "oracle-$target-codec-floor" "$RUNS/oracle-$target/frames" "$RUNS/oracle-$target/mp4frames"
        # Before the fix: the whole clip at once under the per-frame mask
        # (the path this port ran before), same injected inputs.
        oracle_run "oracle-$target-legacy" "$ours-legacy" FASTVIDEO_WAN_CAUSAL_AR=0 FASTVIDEO_WAN_CAUSAL_FPB=1
        oracle_diff "oracle-$target-legacy-diff" "$ref/dump" "$ours-legacy"
        compare_one "oracle-$target-legacy-frames-vs-ref" "$ref/frames" "$RUNS/oracle-$target-legacy/frames"
        # The composed reference path of the KV-window attention.
        oracle_run "oracle-$target-composed" "$ours-composed" FASTVIDEO_WAN_CAUSAL_FLASH=0
        oracle_diff "oracle-$target-composed-diff" "$ref/dump" "$ours-composed"
        oracle_diff "oracle-$target-flash-vs-composed" "$ours-composed" "$ours"
        rm -rf "$ours-legacy" "$ours-composed"
      fi
      # The dumps are hundreds of MB each; the report keeps the numbers.
      rm -rf "$ours" "$ours-f32" "$ours-owntext" "$ref"
    done
    ;;
  writer)
    # The frame writer behind a timed decode (PNG frames + ffmpeg mp4), cheaply:
    # the box (CPUs, memory, cgroup limits, dirty-page settings) and the
    # container disk's write throughput; the writer alone on synthetic 4K
    # frames paced as a 13 s decode, per PNG mode (`writer-bench`); the real
    # 4K conv-VAE decode into the writer (`vae-bench --mp4`); LTX-2.5 512p
    # warm gens; and one LTX-2.5 4K Sol warm gen (FV_WRITER_4K=0 skips it).
    # FASTVIDEO_PNG=inline is the writer before deferral. A 1 Hz sampler logs
    # load, CPU jiffies, dirty/writeback memory, ffmpeg processes and live
    # fv-video-writer threads to $RUNS/sampler.log throughout.
    {
      echo "== nproc"; nproc
      echo "== lscpu"; lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)' || head -30 /proc/cpuinfo
      echo "== cgroup cpu.max"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a
      echo "== meminfo"; grep -E 'MemTotal|MemAvailable|Dirty|Writeback:' /proc/meminfo
      echo "== cgroup memory.max"; cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a
      echo "== vm dirty"; for f in dirty_ratio dirty_background_ratio dirty_bytes dirty_background_bytes dirty_expire_centisecs; do echo "$f=$(cat /proc/sys/vm/$f 2>/dev/null)"; done
      echo "== df"; df -h "$SCRATCH" /workspace 2>/dev/null
      echo "== mount"; grep -E " (/|$SCRATCH|/workspace) " /proc/mounts || true
      echo "== ffmpeg"; ffmpeg -hide_banner -version 2>/dev/null | head -1
    } >"$RUNS/sysinfo.txt" 2>&1
    sed 's/^/[sysinfo] /' "$RUNS/sysinfo.txt" | tee -a "$LOG" >/dev/null
    (
      while :; do
        ff=0; wt=0
        for c in /proc/[0-9]*/comm; do n=""; read -r n <"$c" 2>/dev/null; [[ "$n" == ffmpeg ]] && ff=$((ff + 1)); done
        for c in /proc/[0-9]*/task/*/comm; do n=""; read -r n <"$c" 2>/dev/null; [[ "$n" == fv-video-writer ]] && wt=$((wt + 1)); done
        printf '%s load %s cpu %s dirty_kb %s writeback_kb %s ffmpeg %s writer_threads %s\n' \
          "$(date -u +%H:%M:%S)" "$(cut -d' ' -f1-3 /proc/loadavg)" "$(head -1 /proc/stat | cut -d' ' -f3-9)" \
          "$(awk '/^Dirty:/{print $2}' /proc/meminfo)" "$(awk '/^Writeback:/{print $2}' /proc/meminfo)" "$ff" "$wt"
        sleep 1
      done
    ) >"$RUNS/sampler.log" 2>&1 &
    SAMPLER_PID=$!
    # Container disk: 3 GiB synced (about two 4K clips of PNG frames), then
    # the same through the page cache alone.
    for mode in fdatasync cache; do
      flag="conv=fdatasync"; [[ "$mode" == cache ]] && flag=""
      dd if=/dev/zero of="$SCRATCH/dd-test.bin" bs=16M count=192 $flag 2>&1 | tail -1 | sed "s/^/[dd $mode] /" | tee -a "$LOG"
      rm -f "$SCRATCH/dd-test.bin"
    done
    run_cell writer-bench-4k-mp4 "$BIN" writer-bench --mp4 --png inline,deferred,off --produce-s 13 --dir "$SCRATCH/writer-bench"
    run_cell writer-bench-4k-nomp4 "$BIN" writer-bench --png inline,deferred --produce-s 13 --dir "$SCRATCH/writer-bench"
    for cell in writer-bench-4k-mp4 writer-bench-4k-nomp4; do
      grep -h 'writer-bench \|video writer:' "$RUNS/$cell/stderr.log" 2>/dev/null | sed "s/^/[$cell] /" | tee -a "$LOG" || true
    done
    for png in inline deferred; do
      gated_cell "ltxvae-4k5s-fast-mp4-$png" ltx25-two-stage \
        env FASTVIDEO_PNG="$png" FASTVIDEO_WRITER_TRACE=1 \
        "$BIN" --mode fast ltx2 vae-bench --weights "$W/ltx25" --workload 4k5s \
          --decoder fast --warm --mp4 --clip "$RUNS/ltxvae-4k5s-fast-mp4-$png/frames"
      grep -h 'vae-bench\|video writer:' "$RUNS/ltxvae-4k5s-fast-mp4-$png/stderr.log" 2>/dev/null | tail -3 | sed "s/^/[vae-$png] /" | tee -a "$LOG" || true
    done
    for png in inline deferred; do
      gated_cell "ltx25-512p-$png" ltx25-two-stage \
        env FASTVIDEO_PNG="$png" FASTVIDEO_WRITER_TRACE=1 \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --height 512 --width 768 --num-frames 121 \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-512p-$png/frames"
      grep -h 'video decode\|video writer:' "$RUNS/ltx25-512p-$png/stderr.log" 2>/dev/null | sed "s/^/[512p-$png] /" | tee -a "$LOG" || true
    done
    compare_cells ltx25-512p-inline ltx25-512p-deferred
    if [[ "${FV_WRITER_4K:-1}" == 1 ]]; then
      gated_cell ltx25-4k5s-sol-bf16act ltx25-two-stage \
        env FASTVIDEO_BF16_ACT=1 FASTVIDEO_WRITER_TRACE=1 \
        "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" --workload 4k5s \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --text streamed --warm \
          --clip "$RUNS/ltx25-4k5s-sol-bf16act/frames"
      grep -h 'video decode\|video writer:\|warmup' "$RUNS/ltx25-4k5s-sol-bf16act/stderr.log" 2>/dev/null | sed "s/^/[4k] /" | tee -a "$LOG" || true
    fi
    kill "$SAMPLER_PID" 2>/dev/null || true
    # The frames are GBs; the reports keep the numbers.
    rm -rf "$SCRATCH/writer-bench" "$RUNS"/ltxvae-4k5s-fast-mp4-*/frames
    ;;
  ltxoffload)
    # FASTVIDEO_LTX_OFFLOAD=cpu (sol-engine's BF16 RTX 5090 `--offload cpu`
    # placement) against the default run, LTX-2.5 distilled two-stage, Sol
    # stage 2, bf16 activations (the defaults). Every cell encodes its prompt
    # (--no-text-cache): run a73173e-09270009 showed a cache hit does not
    # reproduce a fresh encode bit for bit (every frame differs from step 0),
    # which would hide what placement does. 512p first: the default
    # (resident), the mode explicitly off, and cpu; frame/wav hashes, the
    # byte-identity compares and the `exact` gate. Then, only when the 512p cpu
    # frames match the resident ones byte for byte, cpu at 4k5s (warm) and
    # 1080p20s (one cold request, as the reference times it): the reference's
    # 30.36 / 29.07 GiB cells. FV_OFFLOAD_BIG=0 skips those;
    # FV_OFFLOAD_4K_RESIDENT=1 adds a resident 4k5s cell.
    ltx_off_gen() {
      local name="$1" mode="$2"
      shift 2
      local envs=(env -u FASTVIDEO_LTX_OFFLOAD)
      [[ "$mode" != default ]] && envs=(env FASTVIDEO_LTX_OFFLOAD="$mode")
      gated_cell "$name" ltx25-two-stage \
        "${envs[@]}" "$BIN" --mode fast ltx2 gen --model-version 2.5 \
          --weights "$W/ltx25" --dit "$W/ltx25" "$@" \
          --prompt "$PROMPT" --seed "$SEED" --two-stage --no-text-cache \
          --clip "$RUNS/$name/frames"
      grep -h 'ltx2 offload\|ltx2 dit offload\|ltx2 memory\|offload stage\|ltx2 text:' "$RUNS/$name/stderr.log" 2>/dev/null \
        | sed "s/^/[$name] /" | tee -a "$LOG" >/dev/null || true
    }
    frames_hash() {
      local d="$RUNS/$1/frames"
      compgen -G "$d/*.png" >/dev/null || { echo missing; return 0; }
      local png wav
      png="$(cd "$d" && sha256sum -- *.png | sha256sum | cut -c1-16)"
      wav="$(cd "$d" && sha256sum -- *.wav 2>/dev/null | sha256sum | cut -c1-16)"
      echo "png:$png wav:$wav"
    }
    small=(--height 512 --width 768 --num-frames 121 --warm)
    ltx_off_gen ltx25-512p-resident default "${small[@]}"
    ltx_off_gen ltx25-512p-off none "${small[@]}"
    ltx_off_gen ltx25-512p-cpu cpu "${small[@]}"
    for c in ltx25-512p-resident ltx25-512p-off ltx25-512p-cpu; do
      log "frames+wav sha256 $c $(frames_hash "$c")"
    done
    compare_cells ltx25-512p-resident ltx25-512p-off --off-identity
    compare_cells ltx25-512p-resident ltx25-512p-cpu --off-identity
    gate_cells ltx25-512p-resident ltx25-512p-cpu exact ltx25-512p-off
    same=""
    h="$(frames_hash ltx25-512p-cpu)"
    r="$(frames_hash ltx25-512p-resident)"
    [[ "$h" != missing && "${h%% *}" == "${r%% *}" ]] && same=1
    if [[ "${FV_OFFLOAD_BIG:-1}" == 1 && ( -n "$same" || "${FV_OFFLOAD_FORCE:-0}" == 1 ) ]]; then
      ltx_off_gen ltx25-4k5s-cpu cpu --workload 4k5s --warm
      if [[ "${FV_OFFLOAD_4K_RESIDENT:-0}" == 1 ]]; then
        ltx_off_gen ltx25-4k5s-resident default --workload 4k5s --warm
        compare_cells ltx25-4k5s-resident ltx25-4k5s-cpu --off-identity
      fi
      ltx_off_gen ltx25-1080p20s-cpu cpu --workload 1080p20s
      # The 4K / 20 s PNGs are GBs; the reports and hashes keep the result.
      for c in ltx25-4k5s-cpu ltx25-4k5s-resident ltx25-1080p20s-cpu; do
        [[ -d "$RUNS/$c/frames" ]] && log "frames+wav sha256 $c $(frames_hash "$c")"
      done
      rm -rf "$RUNS"/ltx25-4k5s-*/frames "$RUNS"/ltx25-1080p20s-*/frames
    else
      log "skip the 4k5s / 1080p20s cpu cells (512p frames identical: ${same:-no}; FV_OFFLOAD_BIG=${FV_OFFLOAD_BIG:-1})"
    fi
    ;;
  hd)
    # H3 at 1080p-class canvases (docs/serve/h3-1080p-and-upscaler.md):
    # h3-turbo (FastH3 4-step VSA) and h3-max (Sol-H3 tau ladder) at the
    # trained 768p canvas and at 1920x1088 / 1088x1920 (`h3 gen
    # --oversize-canvas`, test path only: 2.02x the trained pixel cap), the
    # prompts of scripts/gpu/prompts-hd.json in one process per cell. Then
    # each 768p clip is Lanczos-upscaled (ffmpeg) to the 1080p canvas and
    # compared with the native 1080p clip of the same prompt and seed
    # (compare-clips: sharpness, jitter, patch-boundary ratios; LPIPS with
    # FV_LPIPS=1). Keyframes 0/40/80/120 of every clip are kept; the rest of
    # the PNGs are deleted after the compares. FV_HD_POST_URL: a script fetched
    # and run before that with $RUNS (scripts/gpu/hd-upscaler.sh, the upscaler
    # benchmark).
    : "${FV_PROMPTS:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prompts-hd.json}"
    hd_common=(
      --seconds 5
      --prompt "$PROMPT"
      --seed "$SEED"
      --prompts "$FV_PROMPTS"
      --text-encoder streamed
      --text-cache "$SCRATCH/h3-text-cache"
      --text-weights "$W/h3-base"
    )
    hd_cell() {
      local name="$1" wcell="$2" profile="$3" recipe="$4" h="$5" w="$6" over=()
      (( h * w > 768 * 1344 )) && over=(--oversize-canvas)
      gated_cell "$name" "$wcell" \
        "$BIN" --mode fast --techniques "$profile" h3 gen --weights "$W/h3-base" --h3-recipe "$recipe" \
          --height "$h" --width "$w" ${over[@]+"${over[@]}"} \
          --adaln-cache "$RUNS/$recipe-adaln.cache" \
          --clip-dir "$RUNS/$name/frames" "${hd_common[@]}"
      log "$name $(grep -oE '"(denoise_s|total_s|inference_time_s|peak_memory_mb|peak_allocated_gib)": *[0-9.]+' "$RUNS/$name/benchmark.json" 2>/dev/null | head -8 | tr '\n' ' ')"
    }
    T=(fasth3-4step-vsa h3/fasth3_4step_vsa 4step-vsa)
    M=(sol-h3 h3/sol_h3_4step_engine_ladder sol-h3)
    hd_cell turbo-768p "${T[@]}" 768 1344
    hd_cell turbo-1080p "${T[@]}" 1088 1920
    hd_cell turbo-768p-v "${T[@]}" 1344 768
    hd_cell turbo-1080p-v "${T[@]}" 1920 1088
    hd_cell max-768p "${M[@]}" 768 1344
    hd_cell max-1080p "${M[@]}" 1088 1920
    # 768p -> 1080p Lanczos baselines, then native 1080p against them.
    for pair in turbo-768p:turbo-1080p:1920:1088 turbo-768p-v:turbo-1080p-v:1088:1920 max-768p:max-1080p:1920:1088; do
      IFS=: read -r lo hi uw uh <<<"$pair"
      for d in "$RUNS/$lo/frames"/*/; do
        p="$(basename "$d")"
        [[ "$p" == cold || "$p" == warmup ]] && continue
        compgen -G "$d/frame-*.png" >/dev/null || continue
        up="$RUNS/$lo-lanczos/frames/$p"
        mkdir -p "$up"
        ffmpeg -nostdin -loglevel error -y -start_number 0 -i "$d/frame-%03d.png" \
          -vf "scale=$uw:$uh:flags=lanczos" -start_number 0 "$up/frame-%03d.png" >>"$RUNS/lanczos.log" 2>&1 \
          || log "lanczos $lo/$p failed"
      done
      compare_cells "$lo-lanczos" "$hi"
    done
    if [[ -n "${FV_HD_POST_URL:-}" ]]; then
      log "post script $FV_HD_POST_URL"
      if curl -fsSL "$FV_HD_POST_URL" -o "$SCRATCH/hd-post.sh"; then
        FV_GEN_TIMEOUT_S="${FV_HD_POST_TIMEOUT_S:-3600}" run_cell upscaler env FV_BIN="$BIN" FV_LPIPS_ARGS="${LPIPS_ARGS[*]+${LPIPS_ARGS[*]}}" bash "$SCRATCH/hd-post.sh" "$RUNS"
      else
        log "post script fetch failed"
      fi
    fi
    for c in "$RUNS"/*/frames; do
      for d in "$c"/*/; do
        [[ -d "$d" ]] || continue
        k="${d%/frames/*}/keyframes/$(basename "$d")"
        mkdir -p "$k"
        for n in 000 040 080 120; do
          [[ -f "$d/frame-$n.png" ]] && cp "$d/frame-$n.png" "$k/"
        done
        rm -f "$d"/frame-*.png
      done
    done
    ;;
  serve-engine)
    # Serve engine CUDA backend (WP-11): for each family, the CLI generation
    # (`h3|ltx2|wan gen`) and the same recipe / canvas / seed through
    # EngineService + CudaBackend (`fv-gpucheck engine`): MP4 out, frames
    # byte-compared with the CLI clip, then a second job cancelled mid-run.
    # FV_ENGINE_CELLS picks families (h3 ltx wan).
    want() { [[ " ${FV_ENGINE_CELLS:-h3 ltx wan} " == *" $1 "* ]]; }
    if want h3; then
      h3geo=(--height 480 --width 832 --num-frames 124 --prompt "$PROMPT" --seed "$SEED" --adaln-cache "$RUNS/h3-adaln.cache")
      gated_cell cli-h3-turbo fasth3-4step-vsa \
        "$BIN" --mode fast --techniques h3/fasth3_4step_vsa h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
          "${h3geo[@]}" --text-encoder streamed --text-weights "$W/h3-base" --no-text-cache --no-mp4 \
          --clip-dir "$RUNS/cli-h3-turbo/frames"
      gated_cell engine-h3-turbo fasth3-4step-vsa \
        "$BIN" --keep-going --mode fast --techniques h3/fasth3_4step_vsa engine --model h3-turbo --weights-root "$W" --tae-dir "$TAE" \
          "${h3geo[@]}" --text-encoder streamed --reference "$RUNS/cli-h3-turbo/frames" --cancel-after-step 2 \
          --clip-out "$RUNS/engine-h3-turbo/out"
    fi
    if want ltx; then
      ltxgeo=(--height 704 --width 1280 --num-frames 121 --prompt "$PROMPT" --seed "$SEED")
      gated_cell cli-ltx-turbo ltx25-two-stage \
        "$BIN" --mode fast --techniques ltx2/ltx25_distill_sol ltx2 gen --model-version 2.5 --weights "$W/ltx25" \
          --dit "$W/ltx25" "${ltxgeo[@]}" --two-stage --text streamed --no-text-cache --no-mp4 \
          --clip "$RUNS/cli-ltx-turbo/frames"
      gated_cell engine-ltx-turbo ltx25-two-stage \
        "$BIN" --keep-going --mode fast --techniques ltx2/ltx25_distill_sol engine --model ltx-turbo --weights-root "$W" \
          --tae-dir "$TAE" "${ltxgeo[@]}" --ltx-text streamed --reference "$RUNS/cli-ltx-turbo/frames" \
          --cancel-after-step 3 --clip-out "$RUNS/engine-ltx-turbo/out"
    fi
    if want wan; then
      wangeo=(--height 480 --width 832 --num-frames 81 --prompt "$PROMPT" --seed "$SEED")
      gated_cell cli-wan-turbo fastwan21-1.3b env FASTVIDEO_WAN_VAE=full \
        "$BIN" --mode fast --vsa wan gen --weights "$W/fastwan21-1.3b" "${wangeo[@]}" --no-text-cache --no-mp4 \
          --clip-dir "$RUNS/cli-wan-turbo/frames"
      gated_cell engine-wan-turbo fastwan21-1.3b \
        "$BIN" --keep-going --mode fast engine --model wan-turbo --weights-root "$W" --tae-dir "$TAE" "${wangeo[@]}" \
          --reference "$RUNS/cli-wan-turbo/frames" --cancel-after-step 1 --clip-out "$RUNS/engine-wan-turbo/out"
    fi
    # Keep the reports and MP4s; the PNG frames were compared on the box.
    rm -rf "$RUNS"/cli-*/frames "$RUNS"/engine-*/out/*/frames
    ;;
  *)
    log "FATAL: unknown family $FAMILY"
    exit 2
    ;;
esac

log "matrix done"
write_json "$RUNS/done.json" "$(printf '{"family":"%s","ended":"%s"}' "$FAMILY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
