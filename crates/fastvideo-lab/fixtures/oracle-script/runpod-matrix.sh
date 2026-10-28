#!/usr/bin/env bash
# Runs ON a Runpod GPU box. Family: sfwan | wan | oracle
# One process per family. Continues to the next cell on failure.
# Spark joint LTX refine stays off (FASTVIDEO_LTX2_WEIGHTS unset) unless a
# cell sets it.
#
# Retired to fv-lab (docs/adr/0019-fv-lab.md §10): the families of
# fv-lab-retired.sh (rtx6000, techniques, b6, rtx5090, fastvideo, h3arms,
# trace, ltxfps, ltxvae, precision; b4a, with its verdict step, §11.6;
# det-short and determinism, with their [repeats] hashes, §15.10; fuse, attn3
# and ltx, D1f's, §17.6; hd with its sibling hd5, serve-engine, sfstream and
# mmaudio, §22.19) passed their GPU gate and their branches are gone;
# they are tiers of lab/ (`fv-lab run <tier>` on the box, `runpod-http.sh run`
# / `fv-lab pod up` from a host). This script refuses them (exit 2). Two
# modes of them are fv-lab tiers too (§12, FV_LAB_RESTORED):
# FV_PRECISION_ARM=ltx-nvfp4 is `precision-nvfp4`, FV_VAE_GEN=1 `ltxvae-gen`;
# the refusal names the tier.
# Retired with no tier (§18, FV_SCRIPT_RETIRED): h3, hunyuan, sana-video, b200,
# writer, precision-debug, identity, b4b, b4c, b4c2, h3attn; and (§22, S1)
# sfquality, headline, speechtest, cold, eval, ltxoffload, solbench,
# sol-lingbot, sol-cosmos3. Their branches are gone too; this script refuses
# them (exit 2) and names what replaces each.
#
# Evaluation layer (the gen cells):
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
# bash scripts/gpu/runpod-matrix.sh wan (or any family). A dry run fetches
# no weights, references or baseline, and no TAE file: `wan` (since S6;
# `sfwan` through it) skips its fetch_taehv.sh in a dry run (ADR-0019
# §22.17), as `sfstream` and `mmaudio` did before their branches went. With
# FV_DRY_RUN_GATES=<file> it appends one line per gated cell ("arm", its
# verify-weights.sh cell, -) and per promotion gate ("gate", kind, OFF arm
# or -), for fv-lab's goldens (scripts/gpu/tests/fv-lab-goldens.sh) when a
# family moves to lab/.
set -euo pipefail
FAMILY="${1:?usage: runpod-matrix.sh sfwan|wan|oracle (retired to fv-lab or with no tier: fv-lab-retired.sh)}"
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
  # Families that call run_cell directly (wan, oracle, ...) honour FV_CELLS too.
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

# metric_dry <tag> <cmd...>: a derived-clip metric command (ffmpeg /
# ffprobe, the oracle's clip-metric targets): a dry run logs it and succeeds,
# so the caller skips it (fv-lab's goldens, docs/adr/0019-fv-lab.md §20);
# otherwise it fails and the caller runs the command.
metric_dry() {
  [[ "$DRY_RUN" == 1 ]] || return 1
  local tag="$1"
  shift
  log "dry-run metric $tag: $(printf '%q ' "$@")"
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
          [[ "$DRY_RUN" == 1 || ( -f "$mp4" && -f "$ref/dump/ref.mp4" ) ]] || { echo "${src%%:*} missing $mp4 or ref.mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          for m in ssim psnr; do
            cmd=(ffmpeg -v info -nostats -i "$mp4" -i "$ref/dump/ref.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" -f null -)
            metric_dry "$target-frames" "${cmd[@]}" && continue
            r="$("${cmd[@]}" 2>&1 | grep -E "Parsed_$m" | tail -1 || true)"
            echo "${src%%:*} vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        done
        cp "$ref/dump/ref.mp4" "$fm/reference.mp4" 2>/dev/null || true
        [[ ! -f "$fm/metrics.txt" ]] || sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
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
          [[ "$DRY_RUN" == 1 || -f "$mp4" ]] || { echo "${src%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          cmd=(ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,nb_frames,r_frame_rate,width,height -of compact "$mp4")
          metric_dry "$target-frames" "${cmd[@]}" || "${cmd[@]}" | sed "s/^/${src%%:*}: /" >>"$fm/metrics.txt" 2>&1 || true
        done
        if [[ "$DRY_RUN" == 1 || ( -f "$fm/ours.mp4" && -f "$fm/reference.mp4" ) ]]; then
          for m in ssim psnr; do
            cmd=(ffmpeg -v info -nostats -i "$fm/ours.mp4" -i "$fm/reference.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" -f null -)
            metric_dry "$target-frames" "${cmd[@]}" && continue
            r="$("${cmd[@]}" 2>&1 | grep -E "Parsed_$m" | tail -1 || true)"
            echo "ours vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        fi
        [[ ! -f "$fm/metrics.txt" ]] || sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
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
          [[ "$DRY_RUN" == 1 || -f "$mp4" ]] || { echo "${c%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${c%%:*}.mp4" 2>/dev/null || true
          cmd=(ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,nb_frames,r_frame_rate,width,height -of compact "$mp4")
          metric_dry "$target-frames" "${cmd[@]}" || "${cmd[@]}" | sed "s/^/${c%%:*}: /" >>"$fm/metrics.txt" 2>&1 || true
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
            cmd=(ffmpeg -v info -nostats -i "$mp4" -i "$src" -lavfi "[0:v]select='$keep',setpts=N/24/TB,format=yuv420p[a];[1:v]select='$keep',setpts=N/24/TB,format=yuv420p[b];[a][b]$m" -f null -)
            metric_dry "$target-frames" "${cmd[@]}" && continue
            r="$("${cmd[@]}" 2>&1 | grep -E "Parsed_$m" | tail -1 || true)"
            echo "${c%%:*} kept frames vs source $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        done
        if [[ "$DRY_RUN" == 1 || ( -f "$fm/ours.mp4" && -f "$fm/reference.mp4" ) ]]; then
          for m in ssim psnr; do
            cmd=(ffmpeg -v info -nostats -i "$fm/ours.mp4" -i "$fm/reference.mp4" -lavfi "[0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]$m" -f null -)
            metric_dry "$target-frames" "${cmd[@]}" && continue
            r="$("${cmd[@]}" 2>&1 | grep -E "Parsed_$m" | tail -1 || true)"
            echo "ours vs reference clip $m: ${r##*] }" >>"$fm/metrics.txt"
          done
        fi
        [[ ! -f "$fm/metrics.txt" ]] || sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
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
          [[ "$DRY_RUN" == 1 || -f "$mp4" ]] || { echo "${src%%:*} missing $mp4" >>"$fm/metrics.txt"; continue; }
          cp "$mp4" "$fm/${src%%:*}.mp4" 2>/dev/null || true
          for pin in "${pins[@]}"; do
            idx="${pin%% *}" img="${pin#* }"
            for m in ssim psnr; do
              cmd=(ffmpeg -v info -nostats -i "$mp4" -i "$img" -lavfi
                "[0:v]select=eq(n\,$idx),setpts=N/TB[a];[1:v]scale=768:512:force_original_aspect_ratio=increase:flags=bilinear,crop=768:512,format=yuv420p[b];[a]format=yuv420p[c];[c][b]$m"
                -frames:v 1 -f null -)
              metric_dry "$target-frames" "${cmd[@]}" && continue
              r="$("${cmd[@]}" 2>&1 | grep -E "Parsed_$m" | tail -1 || true)"
              echo "${src%%:*} frame $idx $m: ${r##*] }" >>"$fm/metrics.txt"
            done
          done
        done
        [[ ! -f "$fm/metrics.txt" ]] || sed "s/^/[$target-frames] /" "$fm/metrics.txt" | tee -a "$LOG" || true
      fi
      if [[ "$target" == sfwan* ]]; then
        # Frames: the reference's mp4 (its full VAE, bf16 decode, then its
        # encoder) against our PNG frames; our own mp4 against our PNGs is
        # the codec floor of that comparison.
        # The two decodes are metric commands (metric_dry: a dry run logs
        # them, fv-lab's goldens, docs/adr/0019-fv-lab.md §20.5, §21); our
        # clip is the first mp4 of our frames (`wan gen --clip-dir` writes
        # output.mp4 there, which a dry run names). Neither ends the family:
        # `ls` finding no mp4 under pipefail, or a failed decode, leaves the
        # compares below to log "frames missing". (`dec`, not `cmd`: the
        # -legacy and -composed runs below read the target's `cmd`.)
        for d in "$ref/frames" "$RUNS/oracle-$target/mp4frames"; do mkdir -p "$d"; done
        if [[ "$DRY_RUN" == 1 || -f "$ref/dump/ref.mp4" ]]; then
          dec=(ffmpeg -loglevel error -y -i "$ref/dump/ref.mp4" -start_number 0 "$ref/frames/frame-%03d.png")
          metric_dry "$target-frames" "${dec[@]}" || "${dec[@]}" || true
        fi
        m="$(ls "$RUNS/oracle-$target"/frames/*.mp4 2>/dev/null | head -1 || true)"
        [[ "$DRY_RUN" == 1 ]] && m="$RUNS/oracle-$target/frames/output.mp4"
        if [[ -n "$m" ]]; then
          dec=(ffmpeg -loglevel error -y -i "$m" -start_number 0 "$RUNS/oracle-$target/mp4frames/frame-%03d.png")
          metric_dry "$target-frames" "${dec[@]}" || "${dec[@]}" || true
        fi
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
      rm -rf "$ours" "$ours-f32" "$ours-owntext" "$ours-ownenc" "$ours-ownimg" "$ours-ownaud" "$ref"
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
    # A frozen sibling's mode (fv-lab-retired.sh FV_LAB_RETIRED_SIBLINGS,
    # docs/adr/0019-fv-lab.md §22.19): hd with FV_PROMPTS=…/prompts-hd5.json is
    # the hd5 tier, not hd (whose own prompt set has the same arm names).
    for r in $FV_LAB_RETIRED_SIBLINGS; do
      IFS=: read -r r_tier r_fam r_mode <<<"$r"
      r_key="${r_mode%%=*}"
      if [[ "$FAMILY" == "$r_fam" ]] && fv_lab_retired_sibling_mode "$FAMILY" "$r_key" "${!r_key:-}" >/dev/null; then
        restored_tier="$r_tier" restored_why="$r_key=${!r_key} with $r_fam is the fv-lab tier $r_tier now ($r_fam's $r_key mode, frozen with it: fv-lab-retired.sh FV_LAB_RETIRED_SIBLINGS, ADR-0019 §22.19; without it the run would be $r_fam's own)"
      fi
    done
    if [[ -n "$restored_tier" ]]; then
      log "FATAL: $restored_why: fv-lab run $restored_tier on the box, FV_FAMILY=$restored_tier runpod-http.sh run or fv-lab pod up --family $restored_tier from a host"
    elif fv_lab_retired "$FAMILY"; then
      # docs/adr/0019-fv-lab.md §10: the tier runs only through fv-lab now.
      log "FATAL: $FAMILY is retired to fv-lab (fv-lab-retired.sh): fv-lab run $FAMILY on the box, runpod-http.sh run or fv-lab pod up --family $FAMILY from a host"
    elif fv_script_retired "$FAMILY"; then
      # docs/adr/0019-fv-lab.md §18: retired with no tier; nothing runs it.
      log "FATAL: $FAMILY is retired with no tier (fv-lab-retired.sh FV_SCRIPT_RETIRED): $(fv_script_retired_why "$FAMILY")"
    else
      log "FATAL: unknown family $FAMILY"
    fi
    exit 2
    ;;
esac

log "matrix done"
write_json "$RUNS/done.json" "$(printf '{"family":"%s","ended":"%s"}' "$FAMILY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
