#!/bin/bash
# Phase 4 calibrated H3 gate (docs/perf/sage-attention.md section 8.2) on one
# RTX PRO 6000 pod; EU volume at /workspace (read only: nothing is written
# there). Everything written goes under /root/p4.
#   bash calib.sh gen <max|dense> [arms...]   arms: cud fw2 sage (default all)
#   bash calib.sh cmp <max|dense>             compare-clips + old gate
#   bash calib.sh repeat <max|dense>          cud again, one clip (determinism)
set -u
B=/e2e/p4/fv-gpucheck
W=/workspace/weights
P=/e2e/p4
WK=/root/p4
mkdir -p "$WK"
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)

prof_of() { case $1 in max) echo sol_h3_4step_engine_ladder ;; dense) echo sol_h3_4step ;; esac; }
env_of() {
  case $1 in
    cud|cud2) echo FASTVIDEO_FLASH_KERNEL=cudnn ;;
    fw2) echo FASTVIDEO_FLASH_KERNEL=v2 FASTVIDEO_CUDNN_SDPA_GRAPH=composite ;;
    sage) echo FASTVIDEO_ATTN_SAGE=2 FASTVIDEO_FLASH_KERNEL=cudnn ;;
  esac
}

# gen1 <recipe> <arm> <prompts.json>: one warm process over the prompt set.
gen1() {
  local r=$1 arm=$2 prompts=$3 c=$WK/$1-$2 t0 rc
  mkdir -p "$c"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 >"$c/smi.csv" 2>/dev/null &
  local smi=$!
  t0=$(date +%s)
  # shellcheck disable=SC2046
  ( cd "$c" && env $(env_of "$arm") timeout 2700 "$B" --mode fast --techniques "h3/$(prof_of "$r")" h3 gen \
      --weights $W/h3-base --h3-recipe sol-h3 --prompt x --prompts "$prompts" --seconds 5 --warm \
      --text-encoder resident-fp8 --dit-offload resident --text-cache "$c/text-cache" --text-weights $W/h3-base \
      --adaln-cache "$c/adaln.cache" --clip-dir "$c/frames" ) >"$c/log.txt" 2>&1
  rc=$?
  kill $smi 2>/dev/null
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":$(sort -n "$c/smi.csv" | tail -1)}" >"$c/run.json"
  log "cell $r-$arm rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(sort -n "$c/smi.csv" | tail -1)MiB sdpa=[$(grep -ho 'sdpa[^|]*->[^,]*' "$c/log.txt" | sort | uniq -c | tr '\n' ';' | cut -c1-300)] sage_lines=$(grep -c 'attn_sage' "$c/log.txt")"
}

# pair <recipe> <a> <b>: compare-clips per clip, then the old gate over all.
pair() {
  local r=$1 a=$WK/$1-$2 b=$WK/$1-$3 p tag cmp=()
  mkdir -p "$WK/compare" "$WK/gate"
  for p in "$a"/frames/*/; do
    p=$(basename "$p"); [[ $p == cold || $p == warmup ]] && continue
    [ -f "$b/frames/$p/frame-000.png" ] || continue
    tag="$r-$2--$3-$p"
    [ -f "$WK/compare/compare-clips-$tag.json" ] || \
      "$B" --out "$WK/compare" --tag "$tag" compare-clips --baseline "$a/frames/$p" --candidate "$b/frames/$p" "${LPIPS[@]}" \
        >"$WK/compare/$tag.out" 2>&1
    [ -f "$WK/compare/compare-clips-$tag.json" ] && cmp+=(--compare "$WK/compare/compare-clips-$tag.json")
  done
  tag="$r-$2--$3"
  "$B" --out "$WK/gate" --tag "$tag" gate --baseline "$a" --candidate "$b" --policy "$P/gate-policy.toml" --kind lossy "${cmp[@]}" \
    >"$WK/gate/$tag.out" 2>&1
  log "pairs $tag: ${#cmp[@]} compares; old gate: $(grep -o 'gate verdict: .*' "$WK/gate/$tag.out" | tail -1)"
}

case "${1:?stage}" in
gen)
  r=${2:?recipe}; shift 2
  for arm in ${*:-cud fw2 sage}; do gen1 "$r" "$arm" "$P/prompts-5x3.json"; done
  ;;
cmp)
  r=${2:?recipe}
  pair "$r" cud fw2; pair "$r" cud sage; pair "$r" fw2 sage
  ;;
repeat)
  r=${2:?recipe}
  python3 -c "
import json; d=json.load(open('$P/prompts-5x3.json')); d['prompts']=[p for p in d['prompts'] if p['name']=='spark-mountain-lake-s42']
json.dump(d, open('$WK/prompts-repeat.json','w'))"
  gen1 "$r" cud2 "$WK/prompts-repeat.json"
  a=$WK/$r-cud/frames/spark-mountain-lake-s42; b=$WK/$r-cud2/frames/spark-mountain-lake-s42
  n=0; d=0
  for f in "$b"/frame-*.png; do n=$((n+1)); cmp -s "$f" "$a/$(basename "$f")" || d=$((d+1)); done
  log "repeat $r cud vs cud2 (new process, spark-mountain-lake-s42): $d of $n frames differ"
  ;;
esac
