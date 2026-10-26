#!/usr/bin/env bash
# Runs ON a Runpod GPU box. Family: h3 | ltx | hunyuan | wan | b200 | ... | b4c2
# One process per family. Continues to the next cell on failure.
# Spark joint LTX refine stays off (FASTVIDEO_LTX2_WEIGHTS unset) unless a
# cell sets it.
#
# Retired to fv-lab (docs/adr/0019-fv-lab.md §10): the families of
# fv-lab-retired.sh (rtx6000, techniques, b6, rtx5090, fastvideo, h3arms,
# trace, ltxfps, ltxvae, precision; b4a, with its verdict step, §11.6)
# passed their GPU gate and their branches are gone; they are tiers of lab/ (`fv-lab run <tier>` on the box,
# `runpod-http.sh run` / `fv-lab pod up` from a host). This script refuses
# them (exit 2), and `headline` runs its cells of them through fv-lab. Two
# modes of them are fv-lab tiers too (§12, FV_LAB_RESTORED):
# FV_PRECISION_ARM=ltx-nvfp4 is `precision-nvfp4`, FV_VAE_GEN=1 `ltxvae-gen`;
# the refusal names the tier.
#
# Evaluation layer (eval family and the gen cells):
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
#
# FV_DRY_RUN=1: no GPU and no weights needed. Each cell's command line is
# logged instead of run (weights not verified, no baseline fetched), so a
# family's cells can be read off-pod: FV_DRY_RUN=1 FV_WORK=/tmp/fv-dry
# FV_BASELINE_SHA=<sha> bash scripts/gpu/runpod-matrix.sh b4b (or b4c, b4c2). A
# dry run does not fetch the baseline binary, and with
# FV_DRY_RUN_GATES=<file> it appends one line per gated cell ("arm", its
# verify-weights.sh cell, -) and per promotion gate ("gate", kind, OFF arm
# or -), for fv-lab's goldens (scripts/gpu/tests/fv-lab-goldens.sh) when a
# family moves to lab/.
set -euo pipefail
FAMILY="${1:?usage: runpod-matrix.sh solbench|sol-lingbot|sol-cosmos3|determinism|det-short|serve-engine|hd|headline|mmaudio|speechtest|h3|ltx|hunyuan|sana-video|wan|b200|precision-debug|fuse|oracle|writer|eval|ltxoffload|b4b|b4c|b4c2|h3attn (retired to fv-lab: fv-lab-retired.sh)}"
DRY_RUN="${FV_DRY_RUN:-0}"
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
# shellcheck source=scripts/gpu/b4-verdict.sh
. "$(dirname "${BASH_SOURCE[0]}")/b4-verdict.sh"
# shellcheck source=scripts/gpu/fv-lab-retired.sh
. "$(dirname "${BASH_SOURCE[0]}")/fv-lab-retired.sh"
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
[[ -e $SCRATCH/fv-libs/libcudnn.so || "$DRY_RUN" == 1 ]] || { echo "FATAL: libcudnn.so not linked" | tee -a "$LOG"; exit 2; }

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

# frames_digest <cell dir> <name>: <cell>/frames.sha256.json, every *.png
# under <cell>/frames (by path, LC_ALL=C order) with its SHA-256 and the
# count. The fetches take it (a JSON in the cell directory), so two runs of
# one commit compare byte for byte without the frames; fv-lab run writes the
# same file (docs/adr/0019-fv-lab.md §8, crates/fastvideo-lab run.rs).
frames_digest() {
  local cell="$1" name="$2" n=0 sep="" p
  [[ -d "$cell/frames" ]] || return 0
  {
    printf '{"arm":"%s","frames":[' "$name"
    while IFS= read -r p; do
      printf '%s\n{"path":"%s","sha256":"%s"}' "$sep" "$p" "$(sha256sum "$cell/frames/$p" 2>/dev/null | cut -d' ' -f1)"
      sep=","
      n=$((n + 1))
    done < <(cd "$cell/frames" && find . -type f -name '*.png' -printf '%P\n' | LC_ALL=C sort)
    printf '\n],"count":%s}\n' "$n"
  } >"$cell/frames.sha256.json" 2>/dev/null || true
  return 0
}

run_cell() {
  local name="$1"
  shift
  # Families that call run_cell directly (b200, ...) honour FV_CELLS too.
  if [[ -n "${FV_CELLS:-}" && " $FV_CELLS " != *" $name "* ]]; then
    log "skip $name (not in FV_CELLS)"
    return 0
  fi
  if [[ "$DRY_RUN" == 1 ]]; then
    log "dry-run $name: $(printf '%q ' "$@")"
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
  frames_digest "$cell" "$name"
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
  # FV_DRY_RUN_GATES=<file> (dry run only): one "arm\t<name>\t<weight
  # cell>\t-" line per gated cell, for the goldens of fv-lab
  # (scripts/gpu/tests/fv-lab-goldens.sh); gate_cells adds its "gate" lines.
  # The last column was the TAE file of a tae_gated_cell (none left here).
  if [[ "$DRY_RUN" == 1 && -n "${FV_DRY_RUN_GATES:-}" ]]; then
    printf 'arm\t%s\t%s\t-\n' "$name" "$wcell" >>"$FV_DRY_RUN_GATES"
  fi
  if [[ "$DRY_RUN" != 1 ]] && ! FV_WEIGHTS="$W" bash "$VERIFY" "$wcell" >"$RUNS/$name.weights.log" 2>&1; then
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
# start command fetches them (taeh3, taeltx2_3_wide; the Wan families fetch
# their own with fetch_taehv.sh). The TAE-gated arms (tae_gated_cell) left
# with the families retired to fv-lab (fv-lab-retired.sh; a cell's `tae`,
# docs/adr/0019-fv-lab.md §9.2).
if [[ -z "${FV_TAE_DIR:-}" && -f "$AUX/tae/.complete" ]]; then
  TAE="$AUX/tae"
else
  TAE="${FV_TAE_DIR:-$SCRATCH/tae}"
fi

# Paired-clip quality gate (`fv-gpucheck compare-clips`): the candidate
# cell's frames against the baseline cell's, sol-engine collect_run.py
# metrics (+ LPIPS with FV_LPIPS=1, on the GPU). Extra args (e.g.
# --off-identity for a switch that must not change a byte) pass through.
# Report: $RUNS/compare/compare-clips-<baseline>--<candidate>.json, or with a
# prompt set one per prompt: ...--<candidate>-<prompt>.json.
compare_one() {
  local tag="$1" bf="$2" cf="$3" dir="$RUNS/compare"
  shift 3
  if [[ "$DRY_RUN" == 1 ]]; then
    log "dry-run compare $tag $*"
    return 0
  fi
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
  # FV_DRY_RUN_GATES (dry run only): "gate\t<baseline>\t<candidate>\t<kind>\t<off or ->".
  if [[ "$DRY_RUN" == 1 && -n "${FV_DRY_RUN_GATES:-}" ]]; then
    printf 'gate\t%s\t%s\t%s\t%s\n' "$base" "$cand" "$kind" "${off:--}" >>"$FV_DRY_RUN_GATES"
  fi
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
  # A dry run fetches nothing: it names the reference and goes on as if it
  # had it, so the target's commands are logged (fv-lab's goldens).
  if [[ "$DRY_RUN" == 1 ]]; then
    log "dry-run reference $target: $dest/dump"
    return 0
  fi
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
  # (A dry run has no dumps; it logs the command.)
  [[ "$DRY_RUN" == 1 || ( -d "$2" && -d "$3" ) ]] || return 0
  run_cell "$1" "$BIN" compare-dumps --baseline "$2" --candidate "$3"
  [[ "$DRY_RUN" == 1 ]] && return 0
  grep -h 'compare-dumps' "$RUNS/$1/stderr.log" | sed "s/^/[$1] /" | tee -a "$LOG" || true
}

log "matrix start image=$(cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null || echo unknown)"
if [[ "$DRY_RUN" != 1 ]]; then
  if [[ ! -x "$BIN" ]]; then
    log "FATAL: fv-gpucheck missing at $BIN"
    exit 2
  fi
  command -v nvidia-smi >/dev/null || { log "FATAL: nvidia-smi missing"; exit 2; }
  nvidia-smi -L | tee -a "$LOG"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
fi

case "$FAMILY" in
  determinism)
    # Bit-identity of 1080p generations (docs/perf/determinism.md): the
    # oracle cells first when FV_ORACLE_URL serves references (so the
    # upstream pod can go early), then LTX-2.5 1080p 6 s T2V and I2V and H3
    # (h3-turbo) 1080p 5 s, each run twice in one process (prompt set
    # run1, run2) and once in a second process (run3), all with
    # FASTVIDEO_DIGEST=1 (stage digests in stderr, to find where two runs
    # part). The LTX first run is a conditioning-cache miss and the second a
    # hit; H3 runs without the cache, so its second run re-encodes after
    # `auto` released the resident encoder. A control process per model with
    # FASTVIDEO_SDPA_AUTO=timed (the old per-process timed kernel pick) gives
    # the speed baseline on the same pod. Frames are hashed (PNG bytes and
    # audio.wav) into <cell>/hashes.txt, then deleted but for 3 keyframes;
    # $RUNS/determinism.txt holds the verdicts. FV_DET_CELLS picks cells.
    if [[ -n "${FV_ORACLE_URL:-}" ]]; then
      FV_RUNS_DIR="$RUNS" bash "${BASH_SOURCE[0]}" oracle || true
    fi
    fx="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures"
    det_t2v="${FV_DET_T2V_PROMPT:-A red fox trots through fresh snow at dawn in a pine forest, its breath visible in the cold air, soft golden light, the camera tracks alongside at ground level, cinematic, shallow depth of field}"
    det_i2v="${FV_DET_I2V_PROMPT:-Aerial drone shot of a tropical beach: turquoise sea waves roll in and break into white foam on the sand, the camera glides slowly forward along the shoreline, bright sunny day.}"
    # det_set <file> <prompt> <seed> <name...>: a prompt set of one prompt
    # repeated under each name.
    det_set() {
      local file="$1" prompt="$2" seed="$3" first=1 n esc
      shift 3
      esc="$(printf '%s' "$prompt" | sed 's/\\/\\\\/g; s/"/\\"/g')"
      { printf '{"prompts": ['
        for n in "$@"; do
          (( first )) || printf ','
          first=0
          printf '{"name": "%s", "seed": %s, "prompt": "%s"}' "$n" "$seed" "$esc"
        done
        printf ']}\n'; } >"$file"
    }
    det_set "$RUNS/t2v-ab.json" "$det_t2v" 7 run1 run2
    det_set "$RUNS/t2v-c.json" "$det_t2v" 7 run3
    det_set "$RUNS/i2v-ab.json" "$det_i2v" 11 run1 run2
    det_set "$RUNS/i2v-c.json" "$det_i2v" 11 run3
    det_set "$RUNS/h3-ab.json" "$det_t2v" 7 run1 run2
    det_set "$RUNS/h3-c.json" "$det_t2v" 7 run3
    # det_hash <cell>: sha256 of every run's PNG frames (in frame order) and
    # audio.wav; keep frames 000 / 072 / 120 of each run.
    det_hash() {
      local cell="$RUNS/$1" d run
      # A dry run writes no arm directory, so there is nothing to hash.
      [[ "$DRY_RUN" == 1 ]] && return 0
      : >"$cell/hashes.txt"
      for d in "$cell"/frames/*/; do
        [[ -d "$d" ]] || continue
        run="$(basename "$d")"
        [[ "$run" == warmup ]] && continue
        printf '%s frames %s %s\n' "$run" "$(ls "$d" | grep -c '^frame-.*\.png$')" \
          "$(cat "$d"/frame-*.png 2>/dev/null | sha256sum | cut -d' ' -f1)" >>"$cell/hashes.txt"
        [[ -f "$d/audio.wav" ]] && printf '%s audio %s\n' "$run" "$(sha256sum <"$d/audio.wav" | cut -d' ' -f1)" >>"$cell/hashes.txt"
        find "$d" -name 'frame-*.png' ! -name 'frame-000.png' ! -name 'frame-072.png' ! -name 'frame-120.png' -delete
      done
      grep -h ' digest ' "$cell/stderr.log" 2>/dev/null | sed 's/^.*digest //' >"$cell/digests.txt" || true
      sed "s/^/[$1] /" "$cell/hashes.txt" | tee -a "$LOG"
    }
    det_on() { [[ -z "${FV_DET_CELLS:-}" || " $FV_DET_CELLS " == *" $1 "* ]]; }
    ltx_det=("$BIN" --mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25"
      --height 1088 --width 1920 --num-frames 145 --frame-rate 24 --two-stage --text streamed)
    for arm in t2v i2v; do
      img=()
      [[ "$arm" == i2v ]] && img=(--image "$fx/ti2v-beach-832x480.jpg")
      p="$det_t2v"; seed=7
      [[ "$arm" == i2v ]] && { p="$det_i2v"; seed=11; }
      for proc in ab c timed; do
        cell="ltx-$arm-$proc"
        det_on "$cell" || continue
        set_file="$RUNS/$arm-$proc.json"
        extra_env=()
        [[ "$proc" == timed ]] && { set_file="$RUNS/$arm-ab.json"; extra_env=(FASTVIDEO_SDPA_AUTO=timed); }
        gated_cell "$cell" ltx25-two-stage env FASTVIDEO_DIGEST=1 ${extra_env[@]+"${extra_env[@]}"} \
          "${ltx_det[@]}" ${img[@]+"${img[@]}"} --prompt "$p" --seed "$seed" --prompts "$set_file" \
          --clip "$RUNS/$cell/frames"
        det_hash "$cell"
      done
    done
    for proc in ab c timed; do
      cell="h3-$proc"
      det_on "$cell" || continue
      set_file="$RUNS/h3-$proc.json"
      extra_env=()
      [[ "$proc" == timed ]] && { set_file="$RUNS/h3-ab.json"; extra_env=(FASTVIDEO_SDPA_AUTO=timed); }
      gated_cell "$cell" fasth3-4step-vsa env FASTVIDEO_DIGEST=1 ${extra_env[@]+"${extra_env[@]}"} \
        "$BIN" --mode fast --techniques h3/fasth3_4step_vsa h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa \
          --height 1088 --width 1920 --oversize-canvas --seconds 5 --prompt "$det_t2v" --seed 7 \
          --prompts "$set_file" --text-encoder auto --no-text-cache --text-weights "$W/h3-base" \
          --adaln-cache "$RUNS/h3-det-adaln.cache" --clip-dir "$RUNS/$cell/frames"
      det_hash "$cell"
    done
    # Verdicts: every run of a model (run1, run2 in one process, run3 in the
    # other; the timed control's too, for reference) against run1.
    {
      for m in ltx-t2v ltx-i2v h3; do
        # (|| true: a model that did not run has no hashes.txt, and an -ab arm
        # that was skipped or crashed an empty one; pipefail + set -e ended
        # the family there, before `matrix done`. The family now goes on, and
        # a reference arm without run1 hashes says so in determinism.txt.)
        ref="$(grep -h '^run1 frames' "$RUNS/$m-ab/hashes.txt" 2>/dev/null | awk '{print $4}' || true)"
        refa="$(grep -h '^run1 audio' "$RUNS/$m-ab/hashes.txt" 2>/dev/null | awk '{print $3}' || true)"
        if [[ -f "$RUNS/$m-ab/hashes.txt" && -z "$ref" ]]; then
          echo "$m-ab: no run1 hashes"
        fi
        for f in "$RUNS/$m-ab/hashes.txt" "$RUNS/$m-c/hashes.txt" "$RUNS/$m-timed/hashes.txt"; do
          [[ -f "$f" ]] || continue
          c="$(basename "$(dirname "$f")")"
          while read -r run kind a b; do
            want="$ref"; got="$b"
            [[ "$kind" == audio ]] && { want="$refa"; got="$a"; }
            [[ "$kind" == frames ]] && [[ "$a" == 0 ]] && got=none
            printf '%s %s %s %s\n' "$c/$run" "$kind" "$([[ -n "$want" && "$got" == "$want" ]] && echo IDENTICAL || echo DIFFERENT)" "$got"
          done <"$f"
        done
      done
    } | tee "$RUNS/determinism.txt" | tee -a "$LOG"
    ;;
  det-short)
    # Short-config bit-identity check (docs/perf/determinism.md "Results"):
    # every arm runs FV_DET_REPEATS (default 3) times, each in a fresh
    # process with its own caches (nothing shared on disk between runs), and
    # the frames (PNG bytes, in path order) and audio.wav are hashed per run.
    # One more process per model with FASTVIDEO_SDPA_AUTO=timed (the old
    # per-process timed pick) is the speed baseline. Arms (FV_DET_CELLS
    # picks; default all but ltx-sol):
    #   h3      FastH3 4-step dense, 768x1344, 5 s (self-attention 56 x 37 710: rule -> cuDNN)
    #   wan     FastWan2.2 TI2V-5B, 3 steps, 704x1280x121 (24 x 27 280: rule -> cuDNN)
    #   ltx     LTX-2.5 distilled two-stage, dense stage 2, bf16 attention
    #           (FASTVIDEO_ATTN_SAGE=0), 1088x1920x97 (stage 2 32 x 26 520: rule -> cuDNN)
    #   ltx-sage  the same with SageAttention2 (FASTVIDEO_ATTN_SAGE=2: what
    #           ltx-pro serves on sm_120 by default)
    #   ltx-sol the ltx-turbo route (Sol stage 2), same canvas
    # $RUNS/determinism.txt holds the verdicts, each cell's hashes.json its hashes.
    reps="${FV_DET_REPEATS:-3}"
    det_p="${FV_DET_PROMPT:-A red fox trots through fresh snow at dawn in a pine forest, its breath visible in the cold air, soft golden light, the camera tracks alongside at ground level, cinematic, shallow depth of field}"
    det_on() { [[ " ${FV_DET_CELLS:-h3 wan ltx ltx-sage} " == *" $1 "* ]]; }
    # sh_hash <cell>: sha256 over every frame PNG under the cell (sorted by
    # path, warm-up excluded), and of the audio.wav if any; then the PNGs
    # are deleted but for frame-000.
    sh_hash() {
      local c="$RUNS/$1" fh ah n
      n="$(find "$c" -name 'frame-*.png' ! -path '*/warmup/*' | wc -l)"
      fh="$(find "$c" -name 'frame-*.png' ! -path '*/warmup/*' -print0 | sort -z | xargs -0 -r cat | sha256sum | cut -d' ' -f1)"
      ah="$(find "$c" -name 'audio.wav' ! -path '*/warmup/*' -print0 | sort -z | xargs -0 -r cat | sha256sum | cut -d' ' -f1)"
      (( n == 0 )) && fh=none
      [[ -z "$(find "$c" -name 'audio.wav' ! -path '*/warmup/*' | head -1)" ]] && ah=none
      write_json "$c/hashes.json" "$(printf '{"cell":"%s","frames":%s,"frames_sha256":"%s","audio_sha256":"%s"}' "$1" "$n" "$fh" "$ah")"
      printf '%s frames=%s frames_sha256=%s audio_sha256=%s\n' "$1" "$n" "$fh" "$ah" | tee -a "$RUNS/hashes.txt" | tee -a "$LOG"
      find "$c" -name 'frame-*.png' ! -name 'frame-000.png' -delete
    }
    # args_<arm> <cell dir>: that arm's command for one run, into ARGS, and
    # its own settings into ENVS (one flat `env` with the run's caches before
    # them: fv-lab's det-short cells, docs/adr/0019-fv-lab.md §15).
    args_h3() {
      ENVS=()
      ARGS=("$BIN" --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-dense --seconds 5
        --prompt "$det_p" --seed 7 --text-encoder resident-fp8 --text-weights "$W/h3-base" --no-text-cache
        --adaln-cache "$1/adaln.cache" --clip-dir "$1/frames")
    }
    args_wan() {
      ENVS=(FASTVIDEO_WAN_VAE=full)
      ARGS=("$BIN" --mode fast wan gen --weights "$W/fastwan22-ti2v-5b"
        --preset fast_wan_2_2_ti2v_5b --steps 3 --flow-shift 5.0 --fps 24 --seed 7 --height 704 --width 1280
        --num-frames 121 --prompt "$det_p" --no-text-cache --no-mp4 --clip-dir "$1/frames")
    }
    ltx_args() {
      local c="$1" sage="$2" route="$3"
      ENVS=(FASTVIDEO_ATTN_SAGE="$sage")
      ARGS=("$BIN" --mode fast --techniques "ltx2/ltx25_distill_$route" ltx2 gen
        --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25" --two-stage "--$route-stage2"
        --height 1088 --width 1920 --num-frames 97 --frame-rate 24 --text streamed --no-text-cache
        --prompt "$det_p" --seed 7 --clip "$c/frames")
    }
    args_ltx() { ltx_args "$1" 0 dense; }
    args_ltx-sage() { ltx_args "$1" 2 dense; }
    args_ltx-sol() { ltx_args "$1" 0 sol; }
    # sh_arm <arm> <verify cell>: runs r1..rN, then `timed` (not for ltx-sage:
    # Sage takes the long shapes, so the timed pick only moves cross-attention).
    sh_arm() {
      local arm="$1" wcell="$2" i cell timed
      det_on "$arm" || return 0
      for i in $(seq 1 "$reps") timed; do
        [[ "$i" == timed && "$arm" == ltx-sage ]] && continue
        cell="$arm-r$i" timed=()
        [[ "$i" == timed ]] && { cell="$arm-timed"; timed=(FASTVIDEO_SDPA_AUTO=timed); }
        mkdir -p "$RUNS/$cell"
        "args_$arm" "$RUNS/$cell"
        gated_cell "$cell" "$wcell" env FASTVIDEO_CACHE="$RUNS/$cell/cache" XDG_CACHE_HOME="$RUNS/$cell/cache" \
          ${timed[@]+"${timed[@]}"} ${ENVS[@]+"${ENVS[@]}"} "${ARGS[@]}"
        sh_hash "$cell"
      done
    }
    sh_arm h3 fasth3-4step-dense
    sh_arm wan fastwan22-ti2v-5b
    sh_arm ltx ltx25-two-stage
    sh_arm ltx-sage ltx25-two-stage
    sh_arm ltx-sol ltx25-two-stage
    # Verdicts: each run against r1 of its arm (the timed control too).
    {
      for arm in h3 wan ltx ltx-sage ltx-sol; do
        # (|| true: an arm that did not run has no lines, and pipefail + set -e
        # would end the family there.)
        ref="$(grep -h "^$arm-r1 " "$RUNS/hashes.txt" 2>/dev/null | sed 's/.*frames_sha256=\([^ ]*\).*/\1/' || true)"
        refa="$(grep -h "^$arm-r1 " "$RUNS/hashes.txt" 2>/dev/null | sed 's/.*audio_sha256=\([^ ]*\).*/\1/' || true)"
        [[ -n "$ref" ]] || continue
        { grep -hE "^$arm-(r[0-9]+|timed) " "$RUNS/hashes.txt" || true; } | while read -r cell fr fh ah; do
          fh="${fh#frames_sha256=}"; ah="${ah#audio_sha256=}"
          v=DIFFERENT; [[ "$fh" != none && "$fh" == "$ref" && "$ah" == "$refa" ]] && v=IDENTICAL
          printf '%s %s frames %s audio %s\n' "$cell" "$v" "$fh" "$ah"
        done
      done
    } | tee "$RUNS/determinism.txt" | tee -a "$LOG"
    ;;
  *)
    # A restored mode (fv-lab-retired.sh FV_LAB_RESTORED, docs/adr/0019-fv-lab.md
    # §12): asked for as its family plus its setting, or by its tier's name.
    restored_tier="" restored_why=""
    for r in $FV_LAB_RESTORED; do
      IFS=: read -r r_tier r_fam r_mode <<<"$r"
      r_key="${r_mode%%=*}"
      if [[ "$FAMILY" == "$r_fam" && "${!r_key:-}" == "${r_mode#*=}" ]]; then
        restored_tier="$r_tier" restored_why="$r_mode was a mode of $r_fam; it is the fv-lab tier $r_tier now"
      elif [[ "$FAMILY" == "$r_tier" ]]; then
        restored_tier="$r_tier" restored_why="$r_tier is an fv-lab tier (the $r_fam family's $r_mode mode), never a runpod-matrix.sh family"
      fi
    done
    if [[ -n "$restored_tier" ]]; then
      log "FATAL: $restored_why: fv-lab run $restored_tier on the box, FV_FAMILY=$restored_tier runpod-http.sh run or fv-lab pod up --family $restored_tier from a host"
    elif fv_lab_retired "$FAMILY"; then
      # docs/adr/0019-fv-lab.md §10: the tier runs only through fv-lab now.
      log "FATAL: $FAMILY is retired to fv-lab (fv-lab-retired.sh): fv-lab run $FAMILY on the box, runpod-http.sh run or fv-lab pod up --family $FAMILY from a host"
    else
      log "FATAL: unknown family $FAMILY"
    fi
    exit 2
    ;;
esac

log "matrix done"
write_json "$RUNS/done.json" "$(printf '{"family":"%s","ended":"%s"}' "$FAMILY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
