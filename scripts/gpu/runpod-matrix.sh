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
    sf_stream sfstream-main --parity ${FV_SFSTREAM_RUNS:---run reb-sink15-120s,seconds=120,rope=rebased,sink=15 \
      --run switch-keep-30s,seconds=30,switch_at=15,switch=keep --run switch-reset-20s,seconds=20,switch_at=10,switch=reset}
    sf_stream sfstream-10min --run reb-sink15-600s,seconds=600,rope=rebased,sink=15 --window-s 30
    ;;
  sfquality)
    # Long-rollout quality of the open-ended SF-Wan stream (design risk
    # R12): per 10 s window picture and latent statistics (top band against
    # the rest), a fresh-state TAEHV decode against the carried state, one
    # contact sheet per run (gpucheck-out/sheets/<run>.jpg), for the RoPE
    # policies, sink and window sizes, periodic KV re-cache, graph against
    # eager, and f32 activations. FV_SFQ_RUNS / FV_SFQ_F32_RUNS override.
    TAE_W="$TAE/taew2_1.safetensors"
    [[ -f "$TAE_W" ]] || bash "$(dirname "${BASH_SOURCE[0]}")/fetch_taehv.sh" "$TAE" >>"$RUNS/tae-fetch.log" 2>&1 \
      || log "WARN: taew2_1 fetch failed (tae-fetch.log)"
    export FASTVIDEO_TAE_DIR="$TAE"
    SF_PROMPT="${FV_SF_PROMPT:-A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed}"
    SF_SWITCH="${FV_SF_SWITCH:-A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed}"
    sfq() {
      # $2: a command prefix (e.g. "env FASTVIDEO_BF16_ACT=0"), word-split.
      local name="$1" pre="$2"; shift 2
      # shellcheck disable=SC2086
      gated_cell "$name" sfwan21-1.3b $pre "$BIN" --keep-going --mode fast wan stream --weights "$W/sfwan21-1.3b" \
        --prompt "$SF_PROMPT" --switch-prompt "$SF_SWITCH" --seed "$SEED" "$@"
    }
    SFQ_RUNS="${FV_SFQ_RUNS:-reb-s3-180,seconds=180,keep_s=60 reb-s3-eager-90,seconds=90,graphs=0
      rel-s3-120,seconds=120,rope=rel abs-s3-90,seconds=90,rope=abs reb-s1-90,seconds=90,sink=1
      reb-s6-120,seconds=120,sink=6 reb-s9-120,seconds=120,sink=9 reb-s12-120,seconds=120,sink=12
      reb-s3-w12-90,seconds=90,window=12 reb-s3-w27-90,seconds=90,window=27
      reb-s3-rc7k9-120,seconds=120,recache=7:9 reb-s3-rc14k18-120,seconds=120,recache=14:18}"
    # Runs are separated by spaces or "+" (so a list fits in FV_EXTRA_ENV).
    args=()
    for r in ${SFQ_RUNS//+/ }; do args+=(--run "$r"); done
    # shellcheck disable=SC2086
    sfq sfq-main "" ${FV_SFQ_PARITY---parity} "${args[@]}"
    # A second seed (FV_SFQ_SEED2_RUNS; skipped when unset).
    if [[ -n "${FV_SFQ_SEED2_RUNS:-}" ]]; then
      args=()
      for r in ${FV_SFQ_SEED2_RUNS//+/ }; do args+=(--run "$r"); done
      SEED="${FV_SFQ_SEED2:-7}" sfq sfq-seed2 "" "${args[@]}"
    fi
    args=()
    for r in ${FV_SFQ_F32_RUNS-reb-s3-f32-60,seconds=60,graphs=0}; do args+=(--run "$r"); done
    if (( ${#args[@]} )); then sfq sfq-f32 "env FASTVIDEO_BF16_ACT=0" "${args[@]}"; fi
    ;;
  headline)
    # The headline configurations on one pod (a new GPU type, one run):
    # cells borrowed from other families, all written into this run dir.
    # Its four source families are retired to fv-lab (fv-lab-retired.sh), so
    # `fv-lab run <tier> --cells …` runs them, into this run dir
    # (FV_RUNS_DIR), with this run's work, scratch and per-cell cap.
    lab="${FV_LAB_BIN:-$(command -v fv-lab || echo /opt/fastvideo-rs/target/release/fv-lab)}"
    if [[ "$DRY_RUN" != 1 ]]; then
      # A tier's cells carry their own prompt and seed; fv-lab only warns, so
      # refuse here rather than run something other than what was asked.
      for v in FV_PROMPTS FV_PROMPT FV_SEED; do
        if [[ -n "${!v:-}" ]]; then
          log "FATAL: $v is set, and headline's cells run through fv-lab, which takes prompts and seeds only from the cells (docs/adr/0019-fv-lab.md §7, §10)"
          exit 2
        fi
      done
      [[ -x "$lab" ]] || { log "FATAL: fv-lab missing at $lab (FV_LAB_BIN); headline's cells run through it"; exit 2; }
    fi
    sub() { local fam="$1" rc=0; shift
      # A dry run is fv-lab's (the tier's lines), or without fv-lab its command line.
      if [[ "$DRY_RUN" == 1 ]]; then
        if [[ -x "$lab" ]]; then
          FV_RUNS_DIR="$RUNS" FV_WORK="$WORK" FV_SCRATCH="$SCRATCH" "$lab" run "$fam" --cells "$*" --dry-run || true
        else
          log "dry-run fv-lab: $(printf '%q ' "$lab" run "$fam" --cells "$*")"
        fi
        return 0
      fi
      FV_RUNS_DIR="$RUNS" FV_WORK="$WORK" FV_SCRATCH="$SCRATCH" FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-300}" \
        "$lab" run "$fam" --cells "$*" || rc=$?
      # As the recursive `runpod-matrix.sh <family> || true` before it: a
      # failed tier is logged and the next one runs; its arms' summaries carry
      # their exits, and it never set headline's own exit (matrix_exit).
      (( rc == 0 )) || log "FAIL fv-lab run $fam exit=$rc"
      return 0; }
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
  sana-video)
    # SANA-Video 2B vs Sol-Engine's published ~2.77x (EasyCache 0.1 + QKV
    # merge + bf16 linear attention + compile, 1x GB200): 832x480, 81 frames,
    # 50 steps, cfg 6, warm process. Runs from the CI image's fv-gpucheck; the
    # tree must already be on the volume (weights-manifest.tsv
    # sana-video-2b-480p). docs/ports/sana-video.md has the cost and the gates.
    if ! "$BIN" sana-video --help >/dev/null 2>&1; then
      log "skip sana-video: fv-gpucheck in this image has no sana-video stage"
      write_json "$RUNS/skipped.json" '{"status":"skipped","reason":"fv-gpucheck sana-video not in image"}'
      exit 0
    fi
    SANA_W="${FV_SANA_WEIGHTS:-$W/sana-video-2b-480p}"
    SANA_PROMPT="${FV_SANA_PROMPT:-a corgi running on the beach}"
    # Two generations per cell (cold + timed warm) at an unmeasured speed.
    export FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-2400}"
    for arm in ${FV_SANA_ARMS:-baseline full}; do
      gated_cell "sana-$arm" sana-video-2b-480p \
        "$BIN" --mode fast sana-video gen \
          --weights "$SANA_W" \
          --prompt "$SANA_PROMPT" \
          --arm "$arm" \
          --seed "${FV_SANA_SEED:-42}" \
          --warm \
          --clip "$RUNS/sana-$arm/frames"
    done
    compare_cells sana-baseline sana-full
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

    # ---- Wan2.1 T2V-14B, Wan2.2 TI2V-5B, SF-Wan 1.3B (weights on the EU
    # volume fv-weights-h3-ltx-hy until 2026-10-13; the 500 GB EU volume has H3 + LTX only, so these cells need a
    # volume with the Wan trees or the trees fetched from R2). One prompt ($PROMPT), the upstream
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
    # ---- fal parity (docs/serve/fal-parity.md §3, P0 3-4) ----------------
    # FastWan2.2 TI2V-5B FullAttn, the wan-turbo tier (fal's
    # fal-ai/wan/v2.2-5b/text-to-video/fast-wan): module parity on its own
    # weights against Diffusers (the upstream pod's oracle:fastwan22-ti2v
    # dump, DiT at the DMD timestep 757), then its model card recipe: 3 DMD
    # steps (1000/757/522), shift 5, 704x1280x121 at 24 fps, the full Wan 2.2
    # VAE (fw22) and TAEHV taew2_2 (fw22-taehv, the wan-draft tier).
    if [[ -z "${FV_CELLS:-}" || " $FV_CELLS " == *" fw22-oracle "* ]]; then
      target=fastwan22-ti2v
      ref="$SCRATCH/oracle-ref/$target"
      if oracle_fetch "$target" "$ref"; then
        gated_cell fw22-oracle fastwan22-ti2v-5b \
          "$BIN" --mode fast --keep-going wan oracle --weights "$W/fastwan22-ti2v-5b" --reference "$ref/dump" \
            --dump-out "$RUNS/fw22-oracle-dump" --taehv "$TAE/taew2_2.safetensors"
        oracle_diff fw22-oracle-diff "$ref/dump" "$RUNS/fw22-oracle-dump"
        rm -rf "$ref" "$RUNS/fw22-oracle-dump"
      else
        mkdir -p "$RUNS/fw22-oracle"
        write_json "$RUNS/fw22-oracle/summary.json" '{"cell":"fw22-oracle","exit":null,"skipped":"reference dump unavailable"}'
      fi
    fi
    fw22=(--weights "$W/fastwan22-ti2v-5b" --preset fast_wan_2_2_ti2v_5b --steps 3 --flow-shift 5.0 --fps 24
      --seed "$SEED" --warm)
    gated_cell fw22 fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=full \
      "$BIN" --mode fast wan gen "${fw22[@]}" --height 704 --width 1280 --num-frames 121 \
        --prompt "$PROMPT" "${PROMPT_ARGS[@]}" --clip-dir "$RUNS/fw22/frames"
    gated_cell fw22-taehv fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=taehv \
      "$BIN" --mode fast wan gen "${fw22[@]}" --height 704 --width 1280 --num-frames 121 \
        --prompt "$PROMPT" "${PROMPT_ARGS[@]}" --clip-dir "$RUNS/fw22-taehv/frames"
    compare_cells fw22 fw22-taehv
    # The distilled model against the base checkpoint's 50-step recipe.
    compare_cells wan5b fw22
    gated_cell fw22-i2v fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=full \
      "$BIN" --mode fast wan gen "${fw22[@]}" --height 480 --width 832 --num-frames 121 \
        --image "$fixtures/ti2v-beach-832x480.jpg" --prompt "$ti2v_prompt" --clip-dir "$RUNS/fw22-i2v/frames"
    # fal's 580p (1024x576) and 161 frames on both 5B tiers; the 5B at fal's
    # defaults (40 UniPC steps, CFG 3.5, shift 5). One prompt each.
    fal5b=(--weights "$W/wan22-ti2v-5b" --preset wan_2_2_ti2v_5b --unipc --steps 40 --guidance 3.5
      --flow-shift 5.0 --fps 24 --negative "$wan_neg_cn" --seed "$SEED")
    gated_cell fw22-580p-161f fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=full \
      "$BIN" --mode fast wan gen "${fw22[@]}" --height 576 --width 1024 --num-frames 161 \
        --prompt "$PROMPT" --clip-dir "$RUNS/fw22-580p-161f/frames"
    gated_cell fw22-720p-161f fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=full \
      "$BIN" --mode fast wan gen "${fw22[@]}" --height 704 --width 1280 --num-frames 161 \
        --prompt "$PROMPT" --clip-dir "$RUNS/fw22-720p-161f/frames"
    gated_cell wan5b-580p-161f wan22-ti2v-5b \
      "$BIN" --mode fast wan gen "${fal5b[@]}" --height 576 --width 1024 --num-frames 161 \
        --prompt "$PROMPT" --clip-dir "$RUNS/wan5b-580p-161f/frames"
    gated_cell wan5b-720p-161f wan22-ti2v-5b \
      "$BIN" --mode fast wan gen "${fal5b[@]}" --height 704 --width 1280 --num-frames 161 \
        --prompt "$PROMPT" --clip-dir "$RUNS/wan5b-720p-161f/frames"
    # The catalog's wan-turbo through the engine service (`fv-gpucheck
    # engine`): the same frames as the CLI at the recipe's settings.
    fwgeo=(--height 704 --width 1280 --num-frames 121 --prompt "$PROMPT" --seed "$SEED")
    gated_cell cli-fw22 fastwan22-ti2v-5b env FASTVIDEO_WAN_VAE=full \
      "$BIN" --mode fast wan gen --weights "$W/fastwan22-ti2v-5b" --preset fast_wan_2_2_ti2v_5b --steps 3 \
        --flow-shift 5.0 --fps 24 "${fwgeo[@]}" --no-text-cache --no-mp4 --clip-dir "$RUNS/cli-fw22/frames"
    gated_cell engine-fw22 fastwan22-ti2v-5b \
      "$BIN" --keep-going --mode fast engine --model wan-turbo --weights-root "$W" --tae-dir "$TAE" "${fwgeo[@]}" \
        --reference "$RUNS/cli-fw22/frames" --cancel-after-step 1 --clip-out "$RUNS/engine-fw22/out"
    rm -rf "$RUNS"/cli-fw22/frames "$RUNS"/engine-fw22/out/*/frames
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
  b4b)
    # Plan B4b (docs/adr/0022-block-fusion.md): byte identity of the DiT
    # block fusions after the three families' copies became rt::fuse (one
    # rounding-policy table, one host reference, one switch, the plan's
    # value installed on the DiT and the text refiner / connectors). Each
    # cell runs the baseline binary (FV_BASELINE_SHA's runtime image,
    # fetch-baseline.sh) and this build with the same command line and
    # environment, then compare-clips --off-identity; run_cell writes each
    # arm's frames.sha256.json. Every arm encodes its prompt (--no-text-cache)
    # and is CUPTI-traced for one denoise step (FASTVIDEO_GPU_TRACE=1, the
    # second step): the trace's kernel table is the evidence of which fused
    # kernels ran (no log line names them). FASTVIDEO_GPU_TRACE_TOP lists
    # every kernel of the step in this build; the baseline predates it and
    # lists its 25 slowest, so the named-kernel checks rest on the PR arm and
    # the base arm must agree, call for call, on every fused / bf16-chain
    # kernel it lists, and on the step's total launches. The verdict
    # (b4b-<cell>.verdict.json, b4-verdict.sh b4b_verdict) also needs both
    # arms exit 0, --off-identity passed (not skipped), no dropped trace
    # record, the PR table complete, the named kernels (rule on: the
    # switch-gated fused kernels; rule off: none of those, and the unfused
    # path's kernels) and, for H3, the block structure in the call counts.
    # Only trees of the EU volume (H3, LTX; volumes.sh FV_EU_VOLUME_TREES):
    # Wan's fusions rest on the CPU fixtures (rt/fuse/equality.rs).
    # FV_B4B_CELLS narrows the cells (default: the four below; b4b-h3-fullopt,
    # the MXFP8 route, only when named), one per pod when the coordinator
    # runs them in parallel. Estimated wall time per cell on an RTX PRO 6000
    # (two cold-process runs plus the baseline fetch): H3 4-step ~7-9 min,
    # LTX ~8-10 min, H3 fullopt (49 forwards) ~15-20 min.
    base_sha="${FV_BASELINE_SHA:?b4b family needs FV_BASELINE_SHA (the commit before the fusion unification)}"
    BASE="$SCRATCH/baseline-$base_sha/fv-gpucheck"
    if [[ "$DRY_RUN" != 1 ]]; then
      if ! bash "$(dirname "${BASH_SOURCE[0]}")/fetch-baseline.sh" "$base_sha" "$(dirname "$BASE")" >>"$RUNS/baseline-fetch.log" 2>&1; then
        log "FATAL: baseline binary sha-$base_sha unavailable (baseline-fetch.log)"
        tee -a "$LOG" <"$RUNS/baseline-fetch.log"
        exit 2
      fi
      log "baseline $(tail -1 "$RUNS/baseline-fetch.log")"
    fi
    # A cold H3 load reads ~70 GB from the volume before its first step.
    export FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-900}"
    b4b_cells="${FV_B4B_CELLS:-b4b-h3-4stepvsa b4b-h3-4stepvsa-off b4b-ltx25-512p b4b-ltx25-512p-off}"
    # No verdict of an earlier run in this directory may stand in for this one.
    [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/b4b-*.verdict.json "$RUNS/b4b.verdict.json"
    # b4b_pair <cell> <weight cell> <rule on|off> <relations h3|-> <named
    # kernels> <env K=V ... or -> <clip flag> <fv-gpucheck args ...>:
    # <cell>-base (the baseline binary) and <cell>-pr (this build), each
    # traced with the whole kernel table requested, the clip flag and the
    # frames directory last; then the byte-identity compare and the cell's
    # verdict (b4b_verdict).
    b4b_pair() {
      local cell="$1" wcell="$2" rule="$3" rel="$4" named="$5" envs="$6" clip="$7" arm bin
      shift 7
      [[ " $b4b_cells " == *" $cell "* ]] || { log "skip $cell (not in FV_B4B_CELLS)"; return 0; }
      # shellcheck disable=SC2206
      local e=($envs) k=($named)
      [[ "$envs" == - ]] && e=()
      for arm in base pr; do
        bin="$BIN"
        [[ "$arm" == base ]] && bin="$BASE"
        if [[ "$DRY_RUN" == 1 ]]; then
          log "dry-run $cell-$arm: rm -rf $RUNS/$cell-$arm"
        else
          rm -rf "${RUNS:?}/$cell-$arm"
        fi
        gated_cell "$cell-$arm" "$wcell" \
          env FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_TOP=100000 ${e[@]+"${e[@]}"} \
          "$bin" "$@" "$clip" "$RUNS/$cell-$arm/frames"
      done
      [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS/compare/compare-clips-$cell-base--$cell-pr.json"
      compare_cells "$cell-base" "$cell-pr" --off-identity
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell: verdict b4b_verdict $cell $rule $rel ${k[*]}: both arms exit 0 (summary.json); --off-identity passed; both traces (stderr.log /gpu_trace) with a kernel table, dropped_records 0 and equal kernel_count; the PR table complete (calls sum to kernel_count); every fvf_* / h3_* / wan_* / mx_* kernel the base lists in the PR with the same calls; rule $rule: $([[ "$rule" == on ]] && echo "${k[*]} ran" || echo "no switch-gated fvf_* kernel ran, ${k[*]} did")$([[ "$rel" == h3 ]] && echo '; the H3 block structure in the call counts')"
        return 0
      fi
      log "$(b4b_verdict "$cell" "$rule" "$rel" ${k[@]+"${k[@]}"})"
      tee -a "$LOG" <"$RUNS/$cell-pr/fusion-evidence.log" >/dev/null || true
    }
    # H3: FastH3 4-step VSA at 480p / 5 s, the auto text encoder, each arm
    # encoding its prompt; fusions on (the default, set explicitly: the
    # deferred last residual and the q/k/v/gate heads in one launch), then
    # the control with FASTVIDEO_H3_FUSE=0 (the glue follows it off).
    h3_b4b=(--mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832
      --seconds 5 --prompt "$PROMPT" --seed "$SEED" --text-encoder auto --no-text-cache --no-mp4)
    b4b_pair b4b-h3-4stepvsa fasth3-4step-vsa on h3 "fvf_h3_gate_res_norm_mod fvf_h3_qkvg_heads" \
      "FASTVIDEO_H3_FUSE=1" --clip-dir "${h3_b4b[@]}"
    b4b_pair b4b-h3-4stepvsa-off fasth3-4step-vsa off h3 "h3_norm_mod h3_gate_residual h3_qk_norm_rope" \
      "FASTVIDEO_H3_FUSE=0" --clip-dir "${h3_b4b[@]}"
    # LTX-2.5 two-stage at 512p / 49 frames, streamed Gemma; fusions on
    # (modulated norms, q/k norm + RoPE, the per-head output gates), then the
    # control with FASTVIDEO_LTX_FUSE=0 (the bf16 op chains).
    ltx_b4b=(--mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25"
      --height 512 --width 768 --num-frames 49 --prompt "$PROMPT" --seed "$SEED"
      --two-stage --text streamed --no-text-cache)
    b4b_pair b4b-ltx25-512p ltx25-two-stage on - "fvf_ltx_res_norm_mod fvf_ltx_qk_norm_rope fvf_ltx_gate_merge" \
      "FASTVIDEO_LTX_FUSE=1" --clip "${ltx_b4b[@]}"
    b4b_pair b4b-ltx25-512p-off ltx25-two-stage off - "mx_rms_norm mx_rope_half" \
      "FASTVIDEO_LTX_FUSE=0" --clip "${ltx_b4b[@]}"
    # Named only: the RTX 5090 fullopt route (MXFP8 linears: the head merge
    # feeds to_out's MXFP8 activation), 49 forwards. Its TeaCache may read
    # block outputs, which keeps the last residual undeferred, so only the
    # merge is named and no block relation is checked.
    b4b_pair b4b-h3-fullopt h3-base on - "fvf_merge_heads_mx" "FASTVIDEO_H3_QUANT=mxfp8" --clip-dir \
      --mode fast h3 gen --weights "$W/h3-base" --techniques h3/rtx5090_fullopt --height 480 --width 832 \
      --seconds 5 --prompt "$PROMPT" --seed "$SEED" --text-encoder auto --no-text-cache --no-mp4
    # The family: every cell named in FV_B4B_CELLS must have passed.
    if [[ "$DRY_RUN" == 1 ]]; then
      log "dry-run b4b: family verdict b4_family_verdict b4b $b4b_cells -> $RUNS/b4b.verdict.json (a named cell without a passing verdict fails)"
    else
      # shellcheck disable=SC2086
      log "$(b4_family_verdict b4b $b4b_cells)"
    fi
    for f in "$RUNS"/compare/compare-clips-b4b-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
    ;;
  b4c)
    # Plan B4c (docs/adr/0021-prompt-cache.md): byte identity of the disk
    # prompt cache after the three per-family caches became the one
    # rt::prompt_cache. Each cell runs the baseline binary (FV_BASELINE_SHA's
    # runtime image, fetch-baseline.sh) and this build with the same command
    # line and environment, each arm twice over its own new cache directory
    # ($SCRATCH/b4c-cache/<cell>-<arm>, emptied first): pass -miss encodes
    # the prompt and writes the entry cold, pass -hit reads it. Then
    # compare-clips --off-identity base against pr for the miss pass and for
    # the hit pass (run_cell writes each pass's frames.sha256.json), and the
    # two arms' cache directories must hold byte-identical entries under the
    # same names (so this build reads a cache the baseline wrote exactly as
    # it reads its own). The verdict (b4c-<cell>.verdict.json,
    # b4-verdict.sh b4c_verdict) passes only when every pass exited 0, its
    # benchmark.json says "text_cache": "miss" / "hit" as planned (H3: and
    # "text_encoder" FP8 on the miss, "cache" on the hit), both base-vs-pr
    # compares passed --off-identity, and both caches hold the same entries
    # and no temporary file; b4c.verdict.json fails when a cell named in
    # FV_B4C_CELLS has no passing verdict. pr-miss against pr-hit is compared too, for the
    # record only: on LTX a hit feeds the stored f32 connector output (see
    # the techniques ltx25-512p-sol cell and ADR-0021); it gates nothing.
    # Only trees of the EU volume (H3, LTX); Wan's cache (same format as H3,
    # its own key) rests on the CPU fixtures (rt/prompt_cache/equality.rs).
    # FV_B4C_CELLS narrows the cells (default: both), one per pod when the
    # coordinator runs them in parallel. Estimated wall time per cell on an
    # RTX PRO 6000 (four cold-process runs plus the baseline fetch): H3
    # ~12-15 min, LTX ~14-18 min.
    base_sha="${FV_BASELINE_SHA:?b4c family needs FV_BASELINE_SHA (the commit before the prompt-cache unification)}"
    BASE="$SCRATCH/baseline-$base_sha/fv-gpucheck"
    if [[ "$DRY_RUN" != 1 ]]; then
      if ! bash "$(dirname "${BASH_SOURCE[0]}")/fetch-baseline.sh" "$base_sha" "$(dirname "$BASE")" >>"$RUNS/baseline-fetch.log" 2>&1; then
        log "FATAL: baseline binary sha-$base_sha unavailable (baseline-fetch.log)"
        tee -a "$LOG" <"$RUNS/baseline-fetch.log"
        exit 2
      fi
      log "baseline $(tail -1 "$RUNS/baseline-fetch.log")"
    fi
    # A cold H3 load reads ~70 GB from the volume before its first step.
    export FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-900}"
    b4c_cells="${FV_B4C_CELLS:-b4c-h3-4stepvsa b4c-ltx25-512p}"
    B4C_CACHE="$SCRATCH/b4c-cache"
    # No verdict of an earlier run in this directory may stand in for this one.
    [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/b4c-*.verdict.json "$RUNS/b4c.verdict.json"
    # b4c_cell <cell> <weight cell> <encoder rule: fp8 | -> <clip flag>
    # <fv-gpucheck args ...>: <cell>-<arm>-<pass> for arm base (the baseline
    # binary) and pr (this build), pass miss then hit over
    # $B4C_CACHE/<cell>-<arm>; the clip flag and the frames directory go
    # last. Each pass's directory and the cache start empty; the verdict is
    # b4c_verdict (scripts/gpu/b4-verdict.sh).
    b4c_cell() {
      local cell="$1" wcell="$2" rule="$3" clip="$4" arm bin pass dir
      shift 4
      [[ " $b4c_cells " == *" $cell "* ]] || { log "skip $cell (not in FV_B4C_CELLS)"; return 0; }
      for arm in base pr; do
        bin="$BIN"
        [[ "$arm" == base ]] && bin="$BASE"
        dir="$B4C_CACHE/$cell-$arm"
        if [[ "$DRY_RUN" == 1 ]]; then
          log "dry-run $cell-$arm: rm -rf $dir (a cold cache for the miss pass)"
        else
          rm -rf "$dir"
        fi
        for pass in miss hit; do
          if [[ "$DRY_RUN" == 1 ]]; then
            log "dry-run $cell-$arm-$pass: rm -rf $RUNS/$cell-$arm-$pass"
          else
            rm -rf "${RUNS:?}/$cell-$arm-$pass"
          fi
          gated_cell "$cell-$arm-$pass" "$wcell" \
            "$bin" "$@" --text-cache "$dir" "$clip" "$RUNS/$cell-$arm-$pass/frames"
          [[ "$DRY_RUN" == 1 ]] || log "[$cell-$arm-$pass] text_cache: $(b4_field "$RUNS/$cell-$arm-$pass/benchmark.json" text_cache) text_encoder: $(b4_field "$RUNS/$cell-$arm-$pass/benchmark.json" text_encoder)"
        done
      done
      for pass in miss hit; do
        [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS/compare/compare-clips-$cell-base-$pass--$cell-pr-$pass.json"
        compare_cells "$cell-base-$pass" "$cell-pr-$pass" --off-identity
      done
      compare_cells "$cell-pr-miss" "$cell-pr-hit"
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell: verdict b4c_verdict $cell $rule: every pass exit 0 (summary.json); benchmark.json \"text_cache\" miss / hit$([[ "$rule" == fp8 ]] && echo '; "text_encoder" contains fp8 on miss, is cache on hit'); --off-identity base vs pr passed for miss and hit; $B4C_CACHE/$cell-base and -pr hold the same entries (name, sha256), non-empty, no *.part / *.tmp*"
        return 0
      fi
      log "$(b4c_verdict "$cell" "$rule")"
      tee -a "$LOG" <"$RUNS/$cell.cache-pr.txt" >/dev/null
    }
    # 1. H3: FastH3 4-step VSA at 480p / 5 s, the auto text encoder, which
    #    must resolve to FP8 rows (resident FP8 on an RTX PRO 6000, so the
    #    "/fp8-rows" identity is exercised; rule fp8 fails the cell on a card
    #    where auto streams bf16), each arm's own cache.
    b4c_cell b4c-h3-4stepvsa fasth3-4step-vsa fp8 --clip-dir \
      --mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832 \
      --seconds 5 --prompt "$PROMPT" --seed "$SEED" --text-encoder auto --no-mp4
    # 2. LTX-2.5 two-stage at 512p / 49 frames, Gemma streamed on the miss
    #    pass and not opened on the hit pass.
    b4c_cell b4c-ltx25-512p ltx25-two-stage - --clip \
      --mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25" \
      --height 512 --width 768 --num-frames 49 --prompt "$PROMPT" --seed "$SEED" \
      --two-stage --text streamed
    # The family: every cell named in FV_B4C_CELLS must have passed.
    if [[ "$DRY_RUN" == 1 ]]; then
      log "dry-run b4c: family verdict b4_family_verdict b4c $b4c_cells -> $RUNS/b4c.verdict.json (a named cell without a passing verdict fails)"
    else
      # shellcheck disable=SC2086
      log "$(b4_family_verdict b4c $b4c_cells)"
    fi
    for f in "$RUNS"/compare/compare-clips-b4c-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
    done
    ;;
  b4c2)
    # Plan B4c2 (docs/adr/0021-prompt-cache.md "B4c2"): the prompt-cache keys
    # now carry every setting that changes the conditioning. FV_BASELINE_SHA
    # is main before B4c2. Two kinds of cell, both with the baseline binary
    # (fetch-baseline.sh) and this build, each arm over its own new cache
    # directory ($SCRATCH/b4c2-cache/<cell>-<arm>, emptied first):
    #  - a switch cell (b4c2_switch): pass -write runs one setting and stores
    #    (a miss), pass -read runs the other setting over the same cache, and
    #    this build also runs the read setting over a new cache (-fresh). The
    #    verdict (b4-verdict.sh b4c2_switch_verdict) passes only when the
    #    baseline's read *hits* (the gap, reproduced: main serves the other
    #    setting's conditioning), this build's read misses and its frames are
    #    byte-identical (--off-identity) to the fresh encode (the read setting
    #    over its own new cache, $SCRATCH/b4c2-cache/<cell>-pr-fresh), every
    #    pass exited 0, the caches hold one (base) and two (pr) entries and no
    #    temporary file, and the fresh encode stored exactly the entry (name
    #    and bytes) the read pass stored. -write against -read and base-read against pr-read
    #    are compared for the record only: they measure what the setting does
    #    to the output (for b4c2-ltx25-mode and b4c2-h3-tree, whether the two
    #    paths are bit-equal, which the key no longer assumes).
    #  - the no-change cell (b4c2_noop): default settings, -miss then -hit
    #    per arm, as b4c. Passes only when base and pr are byte-identical on
    #    the miss and on the hit, this build's hit equals its miss, and the
    #    two caches hold one entry each with the same bytes under different
    #    names (the same conditioning under the new key).
    # Only trees of the EU volume: H3 (with its text_encoder_fp8 tree) and
    # LTX-2.5 (no FP8 tree: its FP8 Gemma is quantised at load). The
    # recovered-8B weights (gap 2) and Wan (gap 3) are not on it; their
    # evidence is CPU-only (h3::text_cache, rt/prompt_cache/equality.rs).
    # FV_B4C2_CELLS narrows the cells (default: all five), one or more per
    # pod. Estimated wall time on an RTX PRO 6000, from the b4c gate's runs
    # (H3 cold miss ~115-205 s, hit ~33 s; LTX cold miss ~100-365 s, hit
    # ~30-130 s), after a ~2 min baseline fetch: each LTX switch cell (five
    # runs) ~12-16 min, each H3 switch cell (five runs) ~8-12 min, the
    # no-change cell (four runs) ~6-8 min; all five on one pod ~50-65 min.
    base_sha="${FV_BASELINE_SHA:?b4c2 family needs FV_BASELINE_SHA (main before B4c2)}"
    BASE="$SCRATCH/baseline-$base_sha/fv-gpucheck"
    if [[ "$DRY_RUN" != 1 ]]; then
      if ! bash "$(dirname "${BASH_SOURCE[0]}")/fetch-baseline.sh" "$base_sha" "$(dirname "$BASE")" >>"$RUNS/baseline-fetch.log" 2>&1; then
        log "FATAL: baseline binary sha-$base_sha unavailable (baseline-fetch.log)"
        tee -a "$LOG" <"$RUNS/baseline-fetch.log"
        exit 2
      fi
      log "baseline $(tail -1 "$RUNS/baseline-fetch.log")"
    fi
    # A cold H3 load reads ~70 GB from the volume before its first step.
    export FV_GEN_TIMEOUT_S="${FV_GEN_TIMEOUT_S:-900}"
    b4c2_cells="${FV_B4C2_CELLS:-b4c2-ltx25-fp8 b4c2-ltx25-mode b4c2-h3-numerics b4c2-h3-tree b4c2-h3-noop}"
    B4C_CACHE="$SCRATCH/b4c2-cache"
    # No verdict of an earlier run in this directory may stand in for this one.
    [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/b4c2-*.verdict.json "$RUNS/b4c2.verdict.json"
    # b4c2_pass <cell>-<arm>-<pass> <weight cell> <binary> <env or -> <flags or -> <cache flags> <clip flag> -- <fv-gpucheck args ...>:
    # one cold run in an empty pass directory; the env words go before the
    # binary, the flags after the shared arguments, then the cache flags,
    # the clip flag and the frames directory.
    b4c2_pass() {
      local name="$1" wcell="$2" bin="$3" envs="$4" flags="$5" cachef="$6" clip="$7"
      shift 8
      # shellcheck disable=SC2206
      local e=($envs) f=($flags) cf=($cachef)
      [[ "$envs" == - ]] && e=()
      [[ "$flags" == - ]] && f=()
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $name: rm -rf $RUNS/$name"
      else
        rm -rf "${RUNS:?}/$name"
      fi
      gated_cell "$name" "$wcell" \
        env ${e[@]+"${e[@]}"} "$bin" "$@" ${f[@]+"${f[@]}"} "${cf[@]}" "$clip" "$RUNS/$name/frames"
      [[ "$DRY_RUN" == 1 ]] || log "[$name] text_cache: $(b4_field "$RUNS/$name/benchmark.json" text_cache) text_encoder: $(b4_field "$RUNS/$name/benchmark.json" text_encoder)"
    }
    # b4c2_switch <cell> <weight cell> <clip flag> <write env> <write flags>
    # <read env> <read flags> <fv-gpucheck args ...> (env / flags: words, or
    # - for none).
    b4c2_switch() {
      local cell="$1" wcell="$2" clip="$3" wenv="$4" wflags="$5" renv="$6" rflags="$7" arm bin dir
      shift 7
      [[ " $b4c2_cells " == *" $cell "* ]] || { log "skip $cell (not in FV_B4C2_CELLS)"; return 0; }
      for arm in base pr; do
        bin="$BIN"
        [[ "$arm" == base ]] && bin="$BASE"
        dir="$B4C_CACHE/$cell-$arm"
        if [[ "$DRY_RUN" == 1 ]]; then
          log "dry-run $cell-$arm: rm -rf $dir (a cold cache for the write pass)"
        else
          rm -rf "$dir"
        fi
        b4c2_pass "$cell-$arm-write" "$wcell" "$bin" "$wenv" "$wflags" "--text-cache $dir" "$clip" -- "$@"
        b4c2_pass "$cell-$arm-read" "$wcell" "$bin" "$renv" "$rflags" "--text-cache $dir" "$clip" -- "$@"
      done
      # The reference: the read setting over a new, empty cache (a fresh
      # encode through the same path as the read pass's miss).
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell-pr-fresh: rm -rf $B4C_CACHE/$cell-pr-fresh (a cold cache)"
      else
        rm -rf "$B4C_CACHE/$cell-pr-fresh"
      fi
      b4c2_pass "$cell-pr-fresh" "$wcell" "$BIN" "$renv" "$rflags" "--text-cache $B4C_CACHE/$cell-pr-fresh" "$clip" -- "$@"
      [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/compare/compare-clips-"$cell"-*.json
      compare_cells "$cell-pr-fresh" "$cell-pr-read" --off-identity
      # For the record (not gating): what the setting does to the output,
      # and what main's wrong hit produced.
      compare_cells "$cell-pr-write" "$cell-pr-read" --off-identity
      compare_cells "$cell-base-read" "$cell-pr-read" --off-identity
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell: verdict b4c2_switch_verdict $cell: every pass exit 0 (summary.json); benchmark.json \"text_cache\" base-write / pr-write miss, base-read hit (main reproduces the gap), pr-read miss, pr-fresh miss; --off-identity pr-fresh vs pr-read passed; $B4C_CACHE/$cell-base holds 1 entry, -pr 2, -pr-fresh 1 that is also in -pr (name and sha256), no *.part / *.tmp*"
        return 0
      fi
      log "$(b4c2_switch_verdict "$cell")"
    }
    # b4c2_noop <cell> <weight cell> <clip flag> <fv-gpucheck args ...>.
    b4c2_noop() {
      local cell="$1" wcell="$2" clip="$3" arm bin dir pass
      shift 3
      [[ " $b4c2_cells " == *" $cell "* ]] || { log "skip $cell (not in FV_B4C2_CELLS)"; return 0; }
      for arm in base pr; do
        bin="$BIN"
        [[ "$arm" == base ]] && bin="$BASE"
        dir="$B4C_CACHE/$cell-$arm"
        if [[ "$DRY_RUN" == 1 ]]; then
          log "dry-run $cell-$arm: rm -rf $dir (a cold cache for the miss pass)"
        else
          rm -rf "$dir"
        fi
        for pass in miss hit; do
          b4c2_pass "$cell-$arm-$pass" "$wcell" "$bin" - - "--text-cache $dir" "$clip" -- "$@"
        done
      done
      [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/compare/compare-clips-"$cell"-*.json
      for pass in miss hit; do
        compare_cells "$cell-base-$pass" "$cell-pr-$pass" --off-identity
      done
      compare_cells "$cell-pr-miss" "$cell-pr-hit" --off-identity
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell: verdict b4c2_noop_verdict $cell: every pass exit 0 (summary.json); benchmark.json \"text_cache\" miss / hit; --off-identity base vs pr passed for miss and hit, and pr-miss vs pr-hit; $B4C_CACHE/$cell-base and -pr hold one entry each, same sha256, different names, no *.part / *.tmp*"
        return 0
      fi
      log "$(b4c2_noop_verdict "$cell")"
    }
    ltx_b4c2=(--mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$W/ltx25"
      --height 512 --width 768 --num-frames 49 --prompt "$PROMPT" --seed "$SEED" --two-stage)
    h3_b4c2=(--mode fast h3 gen --weights "$W/h3-base" --h3-recipe 4step-vsa --height 480 --width 832
      --seconds 5 --prompt "$PROMPT" --seed "$SEED" --text-encoder auto --no-mp4)
    # 1. Gaps 1 and 6 (LTX): FP8 Gemma (quantised at load: LTX-2.5 has no
    #    FP8 tree on the volume) writes, bf16 Gemma reads, both resident (FP8
    #    is always resident; two-stage drops Gemma before the DiT loads).
    b4c2_switch b4c2-ltx25-fp8 ltx25-two-stage --clip \
      "FASTVIDEO_LTX2_TEXT_FP8=1" "--text resident" - "--text resident" "${ltx_b4c2[@]}"
    # 2. Gap 5 (LTX): streamed Gemma writes, resident Gemma reads (bf16).
    #    -write vs -read measures whether the two modes are bit-equal.
    b4c2_switch b4c2-ltx25-mode ltx25-two-stage --clip \
      - "--text streamed" - "--text resident" "${ltx_b4c2[@]}"
    # 3. Gap 4 (H3; Wan's gap 3 is the same field): f32 activations write,
    #    the default bf16 activations read; auto encoder (resident FP8 rows
    #    from the text_encoder_fp8 tree on an RTX PRO 6000).
    b4c2_switch b4c2-h3-numerics fasth3-4step-vsa --clip-dir \
      "FASTVIDEO_BF16_ACT=0" - - - "${h3_b4c2[@]}"
    # 4. Gap 6 (H3): FP8 rows quantised at load (FASTVIDEO_TEXT_FP8_TREE=0)
    #    write, the pre-quantised tree reads, both with the resident FP8
    #    encoder (auto on this card). -write vs -read measures whether the
    #    tree is the load-time quantisation bit for bit on the device. No
    #    streamed-FP8 variant: no flag makes H3 stream FP8 (auto does only
    #    when its memory plan rejects the resident encoder for a larger
    #    request); the streamed tree decision is CPU-tested (h3::text
    #    EncoderRun).
    b4c2_switch b4c2-h3-tree fasth3-4step-vsa --clip-dir \
      "FASTVIDEO_TEXT_FP8_TREE=0" - - - "${h3_b4c2[@]}"
    # 5. No change: default settings, H3 (a hit there equals its miss).
    b4c2_noop b4c2-h3-noop fasth3-4step-vsa --clip-dir "${h3_b4c2[@]}"
    # The family: every cell named in FV_B4C2_CELLS must have passed.
    if [[ "$DRY_RUN" == 1 ]]; then
      log "dry-run b4c2: family verdict b4_family_verdict b4c2 $b4c2_cells -> $RUNS/b4c2.verdict.json (a named cell without a passing verdict fails)"
    else
      # shellcheck disable=SC2086
      log "$(b4_family_verdict b4c2 $b4c2_cells)"
    fi
    for f in "$RUNS"/compare/compare-clips-b4c2-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
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
    # them: fv-lab's det-short cells, docs/adr/0019-fv-lab.md §14).
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
    # The LTX reference-to-video prompt (upstream oracle.sh LTX_REF_PROMPT).
    LTX_REF_PROMPT="${FV_LTX_REF_PROMPT:-Reference sheet: Top Row Left (Setting): a rocky coastline at golden hour, dark boulders in the surf and green hills behind a sandy beach. Top Row Right (Setting): a closer view of the same boulders with waves breaking around them. Bottom Row Left (Prop): a red and white striped beach umbrella, shown twice. Bottom Row Right (Character): a cartoon orange crab with big claws and eyes on stalks, shown twice. Generated video: A bright 3D animated shot on the rocky beach at golden hour. The cheerful orange cartoon crab scuttles sideways across the wet sand in front of the dark boulders, waving its big claws, next to the red and white striped beach umbrella planted in the sand, while waves roll in and break into white foam behind it.}"
    oracle_targets="${FV_ORACLE_TARGETS:-fasth3-8step fasth3-4step-vsa ltx25-512p ltx25-512p-dense ltx25-4k ltx25-4k-dense}"
    # Every reference dump is fetched in the background as soon as it is
    # ready (FV_ORACLE_PREFETCH, default on), so an upstream pod that finishes
    # early can go (its idle watchdog) before our runs reach the last target.
    declare -A prefetch=()
    if [[ "${FV_ORACLE_PREFETCH:-1}" == 1 && "$DRY_RUN" != 1 ]]; then
      mkdir -p "$SCRATCH/oracle-ref"
      for target in $oracle_targets; do
        oracle_fetch "$target" "$SCRATCH/oracle-ref/$target" >"$SCRATCH/oracle-ref/$target.fetch.log" 2>&1 &
        prefetch[$target]=$!
      done
      # Every reference fetched and unpacked: publish ORACLE_REFS_DONE, on
      # which scripts/gpu/oracle.sh deletes the upstream pods instead of
      # keeping them until this pod ends.
      (
        t0=$(date +%s)
        while (( $(date +%s) - t0 < ${FV_ORACLE_WAIT_S:-10800} )); do
          all=1
          for target in $oracle_targets; do
            ref="$SCRATCH/oracle-ref/$target"
            [[ -d "$ref/dump" && ! -e "$ref/dump.tar" ]] || all=0
          done
          if (( all )); then
            date -u +%FT%TZ >"$RUNS/ORACLE_REFS_DONE"
            exit 0
          fi
          sleep 20
        done
      ) &
    fi
    for target in $oracle_targets; do
      ref="$SCRATCH/oracle-ref/$target"
      fetched=0
      if [[ -n "${prefetch[$target]:-}" ]] && wait "${prefetch[$target]}"; then
        fetched=1
        cat "$SCRATCH/oracle-ref/$target.fetch.log" >>"$LOG" 2>/dev/null || true
      fi
      if [[ "$fetched" != 1 ]] && ! oracle_fetch "$target" "$ref"; then
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
          # Image conditioning targets (upstream oracle.sh): 512p, dense stage 2.
          fx="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures"
          ltx_args=("${ltx_oracle[@]}")
          case "$target" in
            ltx25-i2v) arm=(--dense-stage2 --image "$fx/ti2v-beach-832x480.jpg") ;;
            ltx25-kf) arm=(--dense-stage2 --image "$fx/ti2v-beach-832x480.jpg"
              --cond-image "$fx/ti2v-beach-zoom-832x480.jpg@120@1.0") ;;
            ltx25-ref2v)
              # IC-LoRA reference-to-video (docs/ports/ltx-ref2v.md): the
              # Ingredients LoRA at stage 1 on the reference sheet, the
              # LoRA's 768x448 stage-1 bucket; prompt and sheet as
              # scripts/gpu/upstream/oracle.sh oracle_ltx_ref.
              wcell=ltx25-ref2v
              geo=(--height 896 --width 1536 --num-frames 121)
              arm=(--dense-stage2 --reference "$fx/ltx-ref-sheet-768x448.png"
                --ic-lora "$W/ltx25-ic-lora-ingredients/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors")
              ltx_args=(--prompt "$LTX_REF_PROMPT" --seed "$SEED" --two-stage --text streamed) ;;
            ltx25-a2v | ltx25-a2v-i2v)
              # Audio-to-video (docs/oracle.md "LTX-2.5 audio-to-video"): the
              # speech clip pinned as clean audio latents on both stages;
              # prompts as scripts/gpu/upstream/oracle.sh LTX_A2V_*.
              arm=(--dense-stage2 --audio "$fx/speech-flite-44k.flac")
              a2v_prompt="${FV_LTX_A2V_PROMPT:-A close-up of a woman with short dark hair talking directly to the camera in a bright living room, natural light, she speaks clearly and calmly, her lips moving with every word.}"
              if [[ "$target" == ltx25-a2v-i2v ]]; then
                arm+=(--image "$fx/ti2v-beach-832x480.jpg")
                a2v_prompt="${FV_LTX_A2V_I2V_PROMPT:-A calm beach at golden hour, gentle waves rolling in, while a narrator speaks.}"
              fi
              ltx_args=(--prompt "$a2v_prompt" --seed "$SEED" --two-stage --text streamed) ;;
            ltx25-retake | ltx25-retake-v | ltx25-retake-a | ltx25-extend)
              # Retake / extend (docs/oracle.md "LTX-2.5 retake and extend"):
              # one distilled stage at the source size on the beach push-in
              # fixture; specs and prompts as scripts/gpu/upstream/oracle.sh.
              src="$fx/beach-push-768x512-24fps.mp4"
              edit_prompt="${FV_LTX_RETAKE_PROMPT:-A huge wave crashes over the dark rocks at golden hour, white spray bursting high into the air, a narrator speaks calmly.}"
              case "$target" in
                ltx25-retake) spec="retake:$src:1.5:3.5:av" ;;
                ltx25-retake-v) spec="retake:$src:1.5:3.5:v" ;;
                ltx25-retake-a) spec="retake:$src:1.5:3.5:a" ;;
                ltx25-extend)
                  spec="extend:$src:48:end"
                  edit_prompt="${FV_LTX_EXTEND_PROMPT:-The camera keeps pushing in slowly over the rocky beach at golden hour, waves rolling onto the sand, a narrator speaks calmly.}" ;;
              esac
              arm=(--edit "$spec")
              ltx_args=(--prompt "$edit_prompt" --seed "$SEED" --text streamed) ;;
            ltx25-a2v-guided)
              # Guided audio-to-video (docs/oracle.md "LTX-2.5 guided
              # audio-to-video"): the dev transformer, the multimodal guider
              # at stage 1, the distilled LoRA at stage 2; the talking head.
              wcell=ltx25-a2v-guided
              arm=(--dense-stage2 --guided --audio "$fx/speech-flite-44k.flac")
              a2v_prompt="${FV_LTX_A2V_PROMPT:-A close-up of a woman with short dark hair talking directly to the camera in a bright living room, natural light, she speaks clearly and calmly, her lips moving with every word.}"
              ltx_args=(--prompt "$a2v_prompt" --seed "$SEED" --two-stage --text streamed) ;;
          esac
          ltx_dit="$W/ltx25"
          [[ "$target" == ltx25-a2v-guided ]] && ltx_dit="$W/ltx25-dev/transformer_full"
          cmd=("$BIN" --mode fast ltx2 gen --model-version 2.5 --weights "$W/ltx25" --dit "$ltx_dit"
            "${geo[@]}" "${arm[@]}" "${ltx_args[@]}") ;;
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
      # Ref2VA control: the reference's condition rows injected as well, so
      # only the DiT (and the text, when its token count matches) differs.
      if [[ "$target" == h3-ref2va-* ]]; then
        oracle_run "oracle-$target-cond" "$ours-cond" FASTVIDEO_INJECT_COND=1
        oracle_diff "oracle-$target-cond-diff" "$ref/dump" "$ours-cond"
      fi
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
      # Image conditioning (ltx25-i2v / ltx25-kf): the main run injects the
      # reference's conditioning latents. -ownenc encodes the reference's
      # preprocessed pixels with our encoder (the cond latent diff is the
      # encoder's alone); -ownimg preprocesses and encodes on our own (the
      # whole path end to end). Then frame fidelity: SSIM / PSNR of the
      # output's pinned frames against the conditioning images (cover +
      # center crop to the canvas), ours and the reference's.
      # Reference-to-video: -ownenc / -ownimg as for the image targets (the
      # reference latent, then the whole reference path, ours), then the
      # whole clip against the reference's (SSIM / PSNR over all frames).
      if [[ "$target" == ltx25-ref2v ]]; then
        oracle_run "oracle-$target-ownenc" "$ours-ownenc" FASTVIDEO_INJECT_COND=0
        oracle_diff "oracle-$target-ownenc-diff" "$ref/dump" "$ours-ownenc"
        oracle_run "oracle-$target-ownimg" "$ours-ownimg" FASTVIDEO_INJECT_COND=0 FASTVIDEO_INJECT_PIXELS=0
        oracle_diff "oracle-$target-ownimg-diff" "$ref/dump" "$ours-ownimg"
        fm="$RUNS/oracle-$target-frames"
        mkdir -p "$fm"
        for src in "ours:$RUNS/oracle-$target/frames/output.mp4" "ownimg:$RUNS/oracle-$target-ownimg/frames/output.mp4"; do
          mp4="${src#*:}"
          [[ -f "$mp4" && -f "$ref/dump/ref.mp4" ]] || { echo "${src%%:*} missing $mp4 or ref.mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          for m in ssim psnr; do
            r="$(ffmpeg -v info -nostats -i "$mp4" -i "$ref/dump/ref.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" \
              -f null - 2>&1 | grep -E "Parsed_$m" | tail -1)"
            echo "${src%%:*} vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        done
        cp "$ref/dump/ref.mp4" "$fm/reference.mp4" 2>/dev/null || true
        sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
      fi
      # Audio-to-video: -ownaud decodes and encodes the driving audio on our
      # own (no reference waveform or latent injected), then the clips (ours,
      # ownaud, the reference's) are kept for the lip-sync proxy
      # (scripts/gpu/lipsync_proxy.py, run where they are fetched) and the
      # output audio is checked against the input (the pass-through).
      if [[ "$target" == ltx25-a2v* ]]; then
        oracle_run "oracle-$target-ownaud" "$ours-ownaud" FASTVIDEO_INJECT_COND=0 FASTVIDEO_INJECT_PIXELS=0
        oracle_diff "oracle-$target-ownaud-diff" "$ref/dump" "$ours-ownaud"
        fm="$RUNS/oracle-$target-frames"
        mkdir -p "$fm"
        for src in "ours:$RUNS/oracle-$target/frames/output.mp4" "ownaud:$RUNS/oracle-$target-ownaud/frames/output.mp4" \
          "reference:$ref/dump/ref.mp4"; do
          mp4="${src#*:}"
          [[ -f "$mp4" ]] || { echo "${src%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,nb_frames,r_frame_rate,width,height \
            -of compact "$mp4" | sed "s/^/${src%%:*}: /" >>"$fm/metrics.txt" 2>&1
        done
        if [[ -f "$fm/ours.mp4" && -f "$fm/reference.mp4" ]]; then
          for m in ssim psnr; do
            r="$(ffmpeg -v info -nostats -i "$fm/ours.mp4" -i "$fm/reference.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" \
              -f null - 2>&1 | grep -E "Parsed_$m" | tail -1)"
            echo "ours vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        fi
        sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
      fi
      # Retake / extend: -ownenc decodes and encodes the source (video and
      # audio) on our own; then the clips (ours, ownenc, the reference's) and
      # their fidelity: every frame against the reference's, and the kept
      # frames (outside the window) against the source.
      if [[ "$target" == ltx25-retake* || "$target" == ltx25-extend ]]; then
        oracle_run "oracle-$target-ownenc" "$ours-ownenc" FASTVIDEO_INJECT_COND=0 FASTVIDEO_INJECT_PIXELS=0
        oracle_diff "oracle-$target-ownenc-diff" "$ref/dump" "$ours-ownenc"
        fm="$RUNS/oracle-$target-frames"
        mkdir -p "$fm"
        src="$fx/beach-push-768x512-24fps.mp4"
        for c in "ours:$RUNS/oracle-$target/frames/output.mp4" "ownenc:$RUNS/oracle-$target-ownenc/frames/output.mp4" \
          "reference:$ref/dump/ref.mp4"; do
          mp4="${c#*:}"
          [[ -f "$mp4" ]] || { echo "${c%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${c%%:*}.mp4" 2>/dev/null || true
          ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,nb_frames,r_frame_rate,width,height \
            -of compact "$mp4" | sed "s/^/${c%%:*}: /" >>"$fm/metrics.txt" 2>&1
          # Kept source frames: 0..32 and 89..120 for the retake of [1.5, 3.5) s
          # (the regenerated latent frames 5..11 cover pixel frames 33..88);
          # 0..104 for the extension (its 0.5 s feather regenerates latent
          # frames 14.., pixel frames 105..).
          case "$target" in
            ltx25-retake-a) keep="lt(n\,121)" ;;
            ltx25-retake*) keep="lt(n\,33)+gte(n\,89)" ;;
            *) keep="lt(n\,105)" ;;
          esac
          for m in ssim psnr; do
            r="$(ffmpeg -v info -nostats -i "$mp4" -i "$src" -lavfi "[0:v]select='$keep',setpts=N/24/TB,format=yuv420p[a];[1:v]select='$keep',setpts=N/24/TB,format=yuv420p[b];[a][b]$m" \
              -f null - 2>&1 | grep -E "Parsed_$m" | tail -1)"
            echo "${c%%:*} kept frames vs source $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        done
        if [[ -f "$fm/ours.mp4" && -f "$fm/reference.mp4" ]]; then
          for m in ssim psnr; do
            r="$(ffmpeg -v info -nostats -i "$fm/ours.mp4" -i "$fm/reference.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" \
              -f null - 2>&1 | grep -E "Parsed_$m" | tail -1)"
            echo "ours vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        fi
        sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
      fi
      if [[ "$target" == ltx25-i2v || "$target" == ltx25-kf || "$target" == ltx25-a2v-i2v ]]; then
        # (ltx25-a2v-i2v: its -ownaud run above already encodes the image on our own.)
        if [[ "$target" != ltx25-a2v-i2v ]]; then
          oracle_run "oracle-$target-ownenc" "$ours-ownenc" FASTVIDEO_INJECT_COND=0
          oracle_diff "oracle-$target-ownenc-diff" "$ref/dump" "$ours-ownenc"
          oracle_run "oracle-$target-ownimg" "$ours-ownimg" FASTVIDEO_INJECT_COND=0 FASTVIDEO_INJECT_PIXELS=0
          oracle_diff "oracle-$target-ownimg-diff" "$ref/dump" "$ours-ownimg"
        fi
        fm="$RUNS/oracle-$target-frames"
        mkdir -p "$fm"
        pins=("0 $fx/ti2v-beach-832x480.jpg")
        [[ "$target" == ltx25-kf ]] && pins+=("120 $fx/ti2v-beach-zoom-832x480.jpg")
        own="ownimg:$RUNS/oracle-$target-ownimg/frames/output.mp4"
        [[ "$target" == ltx25-a2v-i2v ]] && own="ownaud:$RUNS/oracle-$target-ownaud/frames/output.mp4"
        for src in "ours:$RUNS/oracle-$target/frames/output.mp4" "$own" "reference:$ref/dump/ref.mp4"; do
          mp4="${src#*:}"
          [[ -f "$mp4" ]] || { echo "${src%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          for pin in "${pins[@]}"; do
            idx="${pin%% *}" img="${pin#* }"
            for m in ssim psnr; do
              r="$(ffmpeg -v info -nostats -i "$mp4" -i "$img" -lavfi \
                "[0:v]select=eq(n\,$idx),setpts=N/TB[a];[1:v]scale=768:512:force_original_aspect_ratio=increase:flags=bilinear,crop=768:512,format=yuv420p[b];[a]format=yuv420p[c];[c][b]$m" \
                -frames:v 1 -f null - 2>&1 | grep -E "Parsed_$m" | tail -1)"
              echo "${src%%:*} frame $idx $m: ${r##*] }" >>"$fm/metrics.txt"
            done
          done
        done
        sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
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
      rm -rf "$ours" "$ours-f32" "$ours-owntext" "$ours-ownenc" "$ours-ownimg" "$ref"
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
  solbench)
    # sol-engine's published cells for Wan2.1-T2V-1.3B, Wan2.2-T2V-A14B and
    # LTX-2.3 HQ (NVlabs/Sana sol-engine models/{wan21_t2v_1_3b,wan22_t2v_a14b,
    # ltx23}.toml and config/<model>/*.toml), baseline vs optimized, on the
    # pod's one GPU. docs/ports/sol-wan-ltx23.md has the arms, what each
    # reproduces, the weight trees and the GPU-minute estimates. Cells skip
    # (summary.json "weights incomplete") unless verify-weights.sh passes:
    # wan21-t2v-1.3b, wan22-t2v-a14b and ltx23-hq (ltx23 + ltx23-dev), all on
    # the EU volume since 2026-10-06.
    #   FV_CELLS            subset (default below: base + optimized of each)
    #   FV_SOL_A14B_MOE     FASTVIDEO_WAN_MOE for the A14B cells (default auto:
    #                       swap on a 96 GB card, both resident on a B200)
    #   FV_SOL_PROMPTS=5    the 5 sol-engine prompts (default for 1.3B; A14B /
    #                       LTX default to prompt p0 / the ltx23 prompt only)
    : "${FV_CELLS:=wan13-sol-base wan13-sol-fullstack a14b-sol-base a14b-sol-fullopt ltx23-hq-base ltx23-hq-fullopt}"
    sol_wan_prompts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prompts-sol-wan-t2v5.json"
    wan_neg_cn="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    # p0 of that set (no python3 or jq in the runtime image).
    sol_p0="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."
    # Diffusers' single-shift UniPC sigmas: the sol-engine baselines run
    # Diffusers' WanPipeline (UniPCMultistepScheduler.from_config(flow_shift=...)).
    export FASTVIDEO_WAN_UNIPC_SIGMAS=diffusers
    # bf16 GEMMs and the full Wan VAE, as the reference arms run (on sm_100
    # FASTVIDEO_WAN_QUANT would otherwise default to mxfp8).
    export FASTVIDEO_WAN_QUANT=off FASTVIDEO_WAN_VAE=full

    # ---- Wan2.1-T2V-1.3B (models/wan21_t2v_1_3b.toml): 832x480, 81 f @ 16
    # fps, 50 UniPC steps, CFG 6, shift 3, seed 1024, 5 prompts after one warm
    # generation. No published number (the profile has none); the arms are
    # baseline.toml vs wan21_fullstack_sol.toml (EasyCache 0.036 + Sol-Attn
    # tau 1, 10 dense forwards, layer 0 dense, Morton3D), plus each half.
    w13=(--weights "$W/wan21-t2v-1.3b" --preset wan_t2v_1_3b --unipc --steps 50 --guidance 6.0
      --flow-shift 3.0 --fps 16 --height 480 --width 832 --num-frames 81 --seed 1024
      --negative "$wan_neg_cn" --warm)
    if [[ "${FV_SOL_PROMPTS:-5}" == 5 ]]; then
      w13+=(--prompts "$sol_wan_prompts")
    else
      w13+=(--prompt "$sol_p0")
    fi
    FV_GEN_TIMEOUT_S="${FV_SOL_W13_CAP_S:-1800}" gated_cell wan13-sol-base wan21-t2v-1.3b \
      "$BIN" --mode fast wan gen "${w13[@]}" --clip-dir "$RUNS/wan13-sol-base/frames"
    FV_GEN_TIMEOUT_S="${FV_SOL_W13_CAP_S:-1800}" gated_cell wan13-sol-easycache wan21-t2v-1.3b \
      env FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack \
      "$BIN" --mode fast wan gen "${w13[@]}" --clip-dir "$RUNS/wan13-sol-easycache/frames"
    FV_GEN_TIMEOUT_S="${FV_SOL_W13_CAP_S:-1800}" gated_cell wan13-sol-attn wan21-t2v-1.3b \
      env FASTVIDEO_WAN_SOL_ATTN=fullstack \
      "$BIN" --mode fast wan gen "${w13[@]}" --clip-dir "$RUNS/wan13-sol-attn/frames"
    FV_GEN_TIMEOUT_S="${FV_SOL_W13_CAP_S:-1800}" gated_cell wan13-sol-fullstack wan21-t2v-1.3b \
      env FASTVIDEO_WAN_SOL_ATTN=fullstack FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_EASYCACHE_PROFILE=fullstack \
      "$BIN" --mode fast wan gen "${w13[@]}" --clip-dir "$RUNS/wan13-sol-fullstack/frames"
    for arm in easycache attn fullstack; do
      compare_cells wan13-sol-base "wan13-sol-$arm"
      gate_cells wan13-sol-base "wan13-sol-$arm" lossy
    done

    # ---- Wan2.2-T2V-A14B (models/wan22_t2v_a14b.toml): 1280x720, 81 f @ 16
    # fps, 40 UniPC steps, shift 12, CFG 4.0 (high-noise expert) / 3.0
    # (low-noise), boundary 0.875 (26 / 14 steps), seed 1024. Published 1x
    # GB200: base 449.67 s (denoise 434.93), opt 207.01 s
    # (config/wan22_t2v_a14b/singlegpu_opt.toml: kernels + EasyCache 0.30 +
    # PISA 0.10). On a 96 GB card FASTVIDEO_WAN_MOE=auto parks both experts
    # in pinned host memory and swaps them at the boundary; on a B200 both
    # stay resident. One prompt, no warm-up generation (a 720p A14B
    # generation is ~20 min on an RTX PRO 6000; the cold overhead is small
    # next to it); FV_SOL_PROMPTS=5 runs sol-engine's five after a warm one.
    a14b=(--weights "$W/wan22-t2v-a14b" --preset wan_2_2_t2v_a14b --unipc --steps 40
      --guidance 4.0 --guidance-2 3.0 --flow-shift 12.0 --fps 16 --height 720 --width 1280
      --num-frames 81 --seed 1024 --negative "$wan_neg_cn")
    if [[ "${FV_SOL_PROMPTS:-1}" == 5 ]]; then
      a14b+=(--prompts "$sol_wan_prompts" --warm)
    else
      a14b+=(--prompt "$sol_p0")
    fi
    a14b_moe="FASTVIDEO_WAN_MOE=${FV_SOL_A14B_MOE:-auto}"
    FV_GEN_TIMEOUT_S="${FV_SOL_A14B_CAP_S:-5400}" gated_cell a14b-sol-base wan22-t2v-a14b \
      env "$a14b_moe" \
      "$BIN" --mode fast wan gen "${a14b[@]}" --clip-dir "$RUNS/a14b-sol-base/frames"
    FV_GEN_TIMEOUT_S="${FV_SOL_A14B_CAP_S:-5400}" gated_cell a14b-sol-fullopt wan22-t2v-a14b \
      env "$a14b_moe" FASTVIDEO_WAN_SOL_CACHE=easycache FASTVIDEO_WAN_PISA=1 \
      "$BIN" --mode fast wan gen "${a14b[@]}" --clip-dir "$RUNS/a14b-sol-fullopt/frames"
    # Opt-in: both experts resident at MXFP8 (the reference recipe on every
    # block linear; lossy) instead of the swap.
    FV_GEN_TIMEOUT_S="${FV_SOL_A14B_CAP_S:-5400}" gated_cell a14b-sol-mxfp8 wan22-t2v-a14b \
      env FASTVIDEO_WAN_MOE=both FASTVIDEO_WAN_QUANT=mxfp8 \
      "$BIN" --mode fast wan gen "${a14b[@]}" --clip-dir "$RUNS/a14b-sol-mxfp8/frames"
    for arm in fullopt mxfp8; do
      compare_cells a14b-sol-base "a14b-sol-$arm"
      gate_cells a14b-sol-base "a14b-sol-$arm" lossy
    done

    # ---- LTX-2.3 HQ (models/ltx23.toml [official_config]): the dev DiT with
    # the distilled LoRA 384 v1.1 fused at 0.25 (stage 1) / 0.5 (stage 2),
    # 1920x1088, 241 f @ 24 fps, 15-step res2s stage 1 at CFG 3, the 3-sigma
    # stage 2, seed 42, the profile's prompt and negative prompt, one warm
    # generation first (WARMUP=true). Published: 2.40x fullopt / baseline on
    # 1x GB200, no absolute baseline. fullopt = config/ltx23/fullopt.toml's
    # techniques we have: stage-1 SCSP (res2s calls 16-28), stage-2 PISA
    # (sparsity 0.9, block 64, layers 0-1 dense), NVFP4 video FFN, stage-2
    # midpoint prune (0.5, steps 1-2). Not ours: the KWL Triton fusions.
    ltx23_prompt="A cinematic 10 second aerial shot of an antique brass clockwork train crossing a snowy mountain bridge at sunrise, steam drifting through golden light, smooth camera movement, high detail"
    hq=(--model-version 2.3 --hq --weights "$W/ltx23" --dit "$W/ltx23-dev/ltx-2.3-22b-dev.safetensors"
      --prompt "$ltx23_prompt" --seed 42 --text streamed --dit-offload resident --warm)
    hq_up="FASTVIDEO_LTX2_UPSAMPLER=$W/ltx23-dev/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    FV_GEN_TIMEOUT_S="${FV_SOL_LTX_CAP_S:-2400}" gated_cell ltx23-hq-base ltx23-hq \
      env "$hq_up" "$BIN" --mode fast ltx2 gen "${hq[@]}" --dense-stage2 \
        --clip "$RUNS/ltx23-hq-base/frames"
    FV_GEN_TIMEOUT_S="${FV_SOL_LTX_CAP_S:-2400}" gated_cell ltx23-hq-fullopt ltx23-hq \
      env "$hq_up" FASTVIDEO_LTX2_STAGE1_CACHE=1 FASTVIDEO_LTX2_MIDPOINT_PRUNE=1 FASTVIDEO_NVFP4=1 \
      "$BIN" --mode fast ltx2 gen "${hq[@]}" --pisa-stage2 \
        --clip "$RUNS/ltx23-hq-fullopt/frames"
    compare_cells ltx23-hq-base ltx23-hq-fullopt
    gate_cells ltx23-hq-base ltx23-hq-fullopt lossy
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
    # the PNGs are deleted after the compares (fetch with FV_FETCH_TREE=1 and
    # FV_FETCH_SKIP='\.cache$'; a skip matching `frames/` also drops
    # `keyframes/`). FV_HD_POST_URL: a script fetched and run before the prune
    # with $RUNS (an upscaler benchmark; its weights must already be on the
    # volumes, see CLAUDE.md). turbo-1080p-10s / max-1080p-10s: the tier at
    # 10 s (one prompt) for time and memory. The 1080P serve gate runs
    # FV_PROMPTS=scripts/gpu/prompts-hd5.json FV_CELLS="turbo-768p
    # turbo-1080p max-768p max-1080p turbo-1080p-10s max-1080p-10s", then
    # scripts/gpu/lipsync_proxy.py on the speech prompts' MP4s.
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
    # The 1080P tier at 10 s, one prompt ($FV_PROMPT): time and peak memory
    # of a longer tier clip, against the serve engine's working-set estimate
    # (fastvideo_models::h3::memory::measured_working_bytes).
    hd_long() {
      local name="$1" wcell="$2" profile="$3" recipe="$4"
      gated_cell "$name" "$wcell" \
        "$BIN" --mode fast --techniques "$profile" h3 gen --weights "$W/h3-base" --h3-recipe "$recipe" \
          --height 1088 --width 1920 --oversize-canvas --seconds 10 --prompt "$PROMPT" --seed "$SEED" \
          --text-encoder streamed --text-cache "$SCRATCH/h3-text-cache" --text-weights "$W/h3-base" \
          --adaln-cache "$RUNS/$recipe-adaln.cache" --clip-dir "$RUNS/$name/frames"
      log "$name $(grep -oE '"(denoise_s|total_s|peak_memory_mb|peak_allocated_gib)": *[0-9.]+' "$RUNS/$name/benchmark.json" 2>/dev/null | head -8 | tr '\n' ' ')"
    }
    hd_long turbo-1080p-10s "${T[@]}"
    hd_long max-1080p-10s "${M[@]}"
    # 768p -> 1080p Lanczos baselines, then native 1080p against them.
    for pair in turbo-768p:turbo-1080p:1920:1088 turbo-768p-v:turbo-1080p-v:1088:1920 max-768p:max-1080p:1920:1088; do
      IFS=: read -r lo hi uw uh <<<"$pair"
      [[ -d "$RUNS/$lo/frames" && -d "$RUNS/$hi/frames" ]] || continue
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
        "$BIN" --keep-going --mode fast engine --model fastwan21-1.3b --weights-root "$W" --tae-dir "$TAE" "${wangeo[@]}" \
          --reference "$RUNS/cli-wan-turbo/frames" --cancel-after-step 1 --clip-out "$RUNS/engine-wan-turbo/out"
    fi
    # Keep the reports and MP4s; the PNG frames were compared on the box.
    rm -rf "$RUNS"/cli-*/frames "$RUNS"/engine-*/out/*/frames
    ;;
  sol-lingbot)
    # LingBot-Video MoE 30B-A3B vs sol-engine models/lingbot_video.toml
    # (docs/ports/lingbot.md "Comparison"): base 832x480x121 / 40 steps,
    # 1920x1088 refiner / 8 steps, guidance 3, seed 42, the three
    # t2v_val3 prompts. Published (4x GB200): baseline 375.53 s, fullopt
    # 144.36 s. FV_LINGBOT_RESIDENCY: swap (96 GB cards, default) or both
    # (B200). FV_LINGBOT_PROMPTS: prompts per arm (default 3).
    SOL_IN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sol"
    export FV_GEN_TIMEOUT_S="${FV_SOL_TIMEOUT_S:-14400}"
    run_cell lingbot-router "$BIN" --mode fast sol lingbot-router
    for arm in ${FV_LINGBOT_ARMS:-baseline fullopt}; do
      gated_cell "lingbot-$arm" lingbot-moe \
        "$BIN" --mode fast sol lingbot-gen --weights "$W/lingbot-video-moe-30b-a3b" \
          --prompts "$SOL_IN/lingbot-t2v-val3.txt" --num-prompts "${FV_LINGBOT_PROMPTS:-3}" \
          --arm "$arm" --residency "${FV_LINGBOT_RESIDENCY:-swap}" --seed 42 \
          --clip "$RUNS/lingbot-$arm/clips"
    done
    ;;
  sol-cosmos3)
    # Cosmos3-Super 64B T2V vs sol-engine models/cosmos3.toml (docs/ports/cosmos3.md
    # "Comparison"): 1280x720x189, 35 steps, guidance 6, seed 42, one warmup
    # request (WARMUP=true). Published (4x GB200): baseline 130.41 s, fullopt
    # 2.26x. FASTVIDEO_COSMOS3_UND=resident keeps the text tower on the GPU
    # (B200); unset streams it per prompt (96 GB cards).
    SOL_IN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sol"
    export FV_GEN_TIMEOUT_S="${FV_SOL_TIMEOUT_S:-10800}"
    c3=(--weights "$W/cosmos3-super" --prompt "$(cat "$SOL_IN/cosmos3-default.txt")"
      --negative-prompt "$(cat "$SOL_IN/cosmos3-negative.txt")" --seed 42 --warm)
    for arm in ${FV_COSMOS3_ARMS:-baseline teacache teacache-fp8}; do
      case "$arm" in
        teacache-fp8) gated_cell "cosmos3-$arm" cosmos3-super env FASTVIDEO_FP8=1 \
          "$BIN" --mode fast sol cosmos3-gen "${c3[@]}" --arm teacache --clip "$RUNS/cosmos3-$arm/clip" ;;
        *) gated_cell "cosmos3-$arm" cosmos3-super \
          "$BIN" --mode fast sol cosmos3-gen "${c3[@]}" --arm "$arm" --clip "$RUNS/cosmos3-$arm/clip" ;;
      esac
    done
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
