#!/bin/bash
# LongLive GPU check (docs/serve/research-longlive.md §9) on one RTX PRO 6000 pod,
# EU volume at /workspace (read only by convention: everything written goes to /root/ll).
#   bash ll.sh setup|sf|ll|ll240|metrics
set -u
B=/e2e/ll2/fv-gpucheck
W=/workspace/weights
WK=/root/ll
mkdir -p $WK
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache FASTVIDEO_TAE_DIR=$W/auxiliary/tae
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
P=$W/longlive-1.3b/prompts/interactive_example.jsonl
COMMON=(--weights $W/sfwan21-1.3b --height 480 --width 832 --fps 16 --switch-prompts $P)
run() {  # run <tag> <args...>: one fv-gpucheck process, nvidia-smi peak
  local J=$WK/$1; shift; mkdir -p $J
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 >$J/smi.csv 2>/dev/null &
  local smi=$! t0=$(date +%s)
  ( cd $J && timeout ${CELL_TIMEOUT:-3000} $B --mode fast ${KG:-} --out $J --tag $(basename $J) "$@" ) >$J/log.txt 2>&1
  local rc=$?; kill $smi 2>/dev/null
  log "cell $(basename $J) rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(cut -d, -f1 $J/smi.csv | sort -n | tail -1)MiB"
  return $rc
}
case "${1:?stage}" in
setup)
  ls -la $W/sfwan21-1.3b $W/auxiliary/tae $W/longlive-1.3b-safetensors $W/longlive-1.3b/prompts | tee -a $LOG
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $LOG
  { python3 -m venv $WK/venv && $WK/venv/bin/pip install -q numpy 'opencv-python-headless<5'; } >$WK/venv.log 2>&1
  log "venv exit=$?"
  ;;
sf)
  run sf wan stream "${COMMON[@]}" \
    --run sf10,seconds=10 --run sf60,seconds=60 \
    --run sf12_10,window=12,sink=3,rope=abs,seconds=10 --run sf12_60,window=12,sink=3,rope=abs,seconds=60
  ;;
ll)
  run ll wan stream "${COMMON[@]}" --longlive $W/longlive-1.3b-safetensors \
    --run ll10,longlive=1,seconds=10 --run ll60,longlive=1,seconds=60 \
    --run llsw,longlive=1,seconds=60,switch_at=15/30/45,dump=1 \
    --run llkeep,longlive=1,switch=keep,seconds=60,switch_at=15/30/45,dump=1 \
    --run llinf,longlive=1,rope=rebased,seconds=60,switch_at=15/30/45 \
    --run llabs20,longlive=1,seconds=20 --run llrb20,longlive=1,rope=rebased,seconds=20 --run llrel20,longlive=1,rope=rel,seconds=20 \
    --run llgraph,longlive=1,graphs=1,seconds=20,switch_at=10 --run lleager,longlive=1,graphs=0,seconds=20,switch_at=10
  ;;
rest)
  KG=--keep-going run llb wan stream "${COMMON[@]}" --longlive $W/longlive-1.3b-safetensors \
    --run llinf,longlive=1,rope=rebased,seconds=60,switch_at=15/30/45 \
    --run llabs20,longlive=1,seconds=20 --run llrb20,longlive=1,rope=rebased,seconds=20 --run llrel20,longlive=1,rope=rel,seconds=20 \
    --run llgraph,longlive=1,graphs=1,seconds=20,switch_at=10 --run lleager,longlive=1,graphs=0,seconds=20,switch_at=10 \
    --run llkeep2,longlive=1,switch=keep,seconds=60,switch_at=15/30/45 \
    --run ll240,longlive=1,seconds=240
  ;;
ll240)
  run ll240 wan stream "${COMMON[@]}" --longlive $W/longlive-1.3b-safetensors --run ll240,longlive=1,seconds=240
  ;;
metrics)
  F=$(find $WK/ll -type d -name frames | head -1); log "frames at $F"
  $WK/venv/bin/python /e2e/ll/drift.py $F > $WK/drift.json 2> $WK/drift.log
  log "drift metrics exit=$?"
  ;;
esac
