#!/usr/bin/env bash
# Runs ON a Runpod GPU box. Family: h3 | ltx | hunyuan | wan | b200 | ... | b4a
# One process per family. Continues to the next cell on failure.
# Spark joint LTX refine stays off (FASTVIDEO_LTX2_WEIGHTS unset) unless a
# cell sets it.
#
# Retired to fv-lab (docs/adr/0019-fv-lab.md §10): the families of
# fv-lab-retired.sh (rtx6000, techniques, b6, rtx5090, fastvideo, h3arms,
# trace, ltxfps, ltxvae, precision) passed their GPU gate and their branches
# are gone; they are tiers of lab/ (`fv-lab run <tier>` on the box,
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
# FV_BASELINE_SHA=<sha> bash scripts/gpu/runpod-matrix.sh b4a (or b4b, b4c, b4c2). A
# dry run does not fetch the baseline binary, and with
# FV_DRY_RUN_GATES=<file> it appends one line per gated cell ("arm", its
# verify-weights.sh cell, -) and per promotion gate ("gate", kind, OFF arm
# or -), for fv-lab's goldens (scripts/gpu/tests/fv-lab-goldens.sh) when a
# family moves to lab/.
set -euo pipefail
FAMILY="${1:?usage: runpod-matrix.sh solbench|sol-lingbot|sol-cosmos3|determinism|det-short|serve-engine|hd|headline|mmaudio|speechtest|h3|ltx|hunyuan|sana-video|wan|b200|precision-debug|fuse|oracle|writer|eval|ltxoffload|b4a|b4b|b4c|b4c2|h3attn (retired to fv-lab: fv-lab-retired.sh)}"
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
  b4a)
    # Plan B4a (docs/adr/0020-step-cache.md): byte identity of every step
    # cache a family on the EU volume ships, after the per-family controllers
    # became parameter sets of the one implementation per algorithm. Each
    # cell runs the baseline binary (FV_BASELINE_SHA's runtime image,
    # fetch-baseline.sh) and this build with the same command line and
    # environment, then compare-clips --off-identity; run_cell writes each
    # arm's frames.sha256.json. Every arm encodes its prompt
    # (--no-text-cache). A cell fails (b4a-<cell>.verdict.json) when either
    # arm did not exit 0 or reused no forward, the --off-identity compare did
    # not pass (b4-verdict.sh b4_identical), or the arms' cache evidence
    # lines differ; b4a.verdict.json fails when a cell named in
    # FV_B4A_CELLS has no passing verdict.
    # Only H3 TeaCache: Wan / SANA / LingBot / Cosmos caches have no tree
    # on the EU volume (volumes.sh FV_EU_VOLUME_TREES), and LTX-2.5 FBCache
    # has no reachable path (fbcache::scope admits only dev guided stage 1,
    # which no LTX-2.5 generate route runs today); their evidence is the CPU
    # equality fixtures (step_cache/equality.rs).
    # FV_B4A_CELLS narrows the cells (default: all three), one per pod when
    # the coordinator runs them in parallel.
    base_sha="${FV_BASELINE_SHA:?b4a family needs FV_BASELINE_SHA (the commit before the step-cache unification)}"
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
    fx="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fixtures"
    b4a_cells="${FV_B4A_CELLS:-b4a-h3-fullopt-env b4a-h3-fullopt-prof b4a-h3-4step-tea}"
    # No verdict of an earlier run in this directory may stand in for this one.
    [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS"/b4a-*.verdict.json "$RUNS/b4a.verdict.json"
    # b4a_evidence <cell>-<arm>: the step cache's lines of that arm's
    # stderr.log (what was installed, every H3 decision), without the
    # timestamp a log line starts with, into cache-evidence.log. On bytes
    # (LC_ALL=C, grep -a): a log line that is not valid UTF-8, or holds a NUL,
    # is still a line, whatever the pod's locale (fv-lab's [verdict] reads
    # the same bytes, docs/adr/0019-fv-lab.md §11.5).
    b4a_evidence() {
      LC_ALL=C grep -ah 'h3 sol teacache\|h3 teacache:\|h3 teacache step\|fuse step cache' "$RUNS/$1/stderr.log" \
        | LC_ALL=C sed -E 's/^(\[[^]]*\] *)+//' >"$RUNS/$1/cache-evidence.log" || true
    }
    # b4a_pair <cell> <weight cell> <env K=V ... or -> <fv-gpucheck args ...>:
    # <cell>-base (the baseline binary) and <cell>-pr (this build), the
    # frames under <cell>-<arm>/frames, then the byte-identity compare and
    # the cell's verdict.
    b4a_pair() {
      local cell="$1" wcell="$2" envs="$3" arm bin reuses verdict=pass why=""
      shift 3
      [[ " $b4a_cells " == *" $cell "* ]] || { log "skip $cell (not in FV_B4A_CELLS)"; return 0; }
      # shellcheck disable=SC2206
      local e=($envs)
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
          env ${e[@]+"${e[@]}"} "$bin" "$@" "$RUNS/$cell-$arm/frames"
        [[ "$DRY_RUN" == 1 ]] && continue
        b4_exit_ok "$cell-$arm" || { verdict=fail; why+=" $arm:exit"; }
        if [[ ! -f "$RUNS/$cell-$arm/stderr.log" ]]; then
          verdict=fail why+=" $arm:no-run"
          continue
        fi
        b4a_evidence "$cell-$arm"
        reuses="$(LC_ALL=C grep -ac 'h3 teacache step .*REUSE' "$RUNS/$cell-$arm/stderr.log" || true)"
        log "[$cell-$arm] teacache reused forwards: ${reuses:-0}"
        (( ${reuses:-0} > 0 )) || { verdict=fail; why+=" $arm:no-reuse"; }
      done
      [[ "$DRY_RUN" == 1 ]] || rm -f "$RUNS/compare/compare-clips-$cell-base--$cell-pr.json"
      compare_cells "$cell-base" "$cell-pr" --off-identity
      if [[ "$DRY_RUN" == 1 ]]; then
        log "dry-run $cell: verdict also needs both arms exit 0 (summary.json) and the --off-identity compare passed"
        return 0
      fi
      b4_identical "$cell-base" "$cell-pr" || { verdict=fail; why+=" not-identical"; }
      if [[ -f "$RUNS/$cell-base/cache-evidence.log" && -f "$RUNS/$cell-pr/cache-evidence.log" ]]; then
        # -a: evidence holding a NUL is still diffed as text, so the file
        # never names the run directory ("Binary files <path> and ...").
        if ! LC_ALL=C diff -a "$RUNS/$cell-base/cache-evidence.log" "$RUNS/$cell-pr/cache-evidence.log" \
          >"$RUNS/$cell.evidence.diff"; then
          verdict=fail why+=" evidence-differs"
        fi
      fi
      write_json "$RUNS/b4a-$cell.verdict.json" "$(printf '{"cell":"%s","verdict":"%s","why":"%s"}' "$cell" "$verdict" "${why# }")"
      log "b4a $cell: $verdict${why:+ (${why# })}"
    }
    # H3 at 480p / 5 s, the auto text encoder, each arm encoding its prompt.
    h3_b4a=(--mode fast h3 gen --weights "$W/h3-base" --height 480 --width 832 --seconds 5
      --prompt "$PROMPT" --seed "$SEED" --text-encoder auto --no-text-cache --no-mp4)
    # 1. The RTX 5090 fullopt route from the legacy flag: sol-h3-rtx over 49
    #    forwards with FASTVIDEO_H3_SOL_CACHE=teacache (rtx_teacache: 0.10 /
    #    retain 5 / cooldown 1), the `techniques` family's h3-480p-fullopt -env arm.
    b4a_pair b4a-h3-fullopt-env h3-base "FASTVIDEO_H3_SOL_CACHE=teacache" \
      "${h3_b4a[@]}" --h3-recipe sol-h3-rtx --clip-dir
    # 2. The same route from the profile (H3Techniques::teacache_state, the
    #    profile log line), MXFP8 linears as in the techniques -prof arm.
    b4a_pair b4a-h3-fullopt-prof h3-base "FASTVIDEO_H3_QUANT=mxfp8" \
      "${h3_b4a[@]}" --techniques h3/rtx5090_fullopt --clip-dir
    # 3. FastH3 4-step VSA with a loose TeaCache
    #    (fixtures/b4a-h3-4step-teacache.toml: threshold 4.0, retain 1,
    #    cooldown 1): forwards 1 and 2 are the only candidates and reuse
    #    while the accumulated rel-L1 of the block-0 probe stays under 4.0.
    #    Unmeasured on a GPU before this family; the reuse assertion says.
    b4a_pair b4a-h3-4step-tea fasth3-4step-vsa - \
      "${h3_b4a[@]}" --techniques "$fx/b4a-h3-4step-teacache.toml" --clip-dir
    # The family: every cell named in FV_B4A_CELLS must have passed.
    if [[ "$DRY_RUN" == 1 ]]; then
      log "dry-run b4a: family verdict b4_family_verdict b4a $b4a_cells -> $RUNS/b4a.verdict.json (a named cell without a passing verdict fails)"
    else
      # shellcheck disable=SC2086
      log "$(b4_family_verdict b4a $b4a_cells)"
    fi
    for f in "$RUNS"/compare/compare-clips-b4a-*.json; do
      [[ -f "$f" ]] && log "$(basename "$f"): $(grep -oE '"(status|off_identity|max_abs_diff_uint8)": *[^,}]*' "$f" | tr '\n' ' ')"
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
