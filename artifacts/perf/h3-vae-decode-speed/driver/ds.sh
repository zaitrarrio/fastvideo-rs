#!/bin/bash
# H3 VAE decode speed: on-pod driver (one RTX PRO 6000 pod, the gpucheck
# runtime image, scripts/serve/e2e/pod.sh's sidecar), adapted from
# artifacts/perf/h3-vae-tile-batching/driver/vt.sh. The binary under test is
# this branch's fv-gpucheck, uploaded to /e2e/ds/bin (FV_BIN); every cell,
# the baseline included, runs that one binary. The EU volume at /workspace is
# only read; everything is written under /e2e/ds/work.
#   bash ds.sh prep
#   bash ds.sh bench                  kernels --groups h3vae_bench
#   bash ds.sh gen <cell> <res 768p|480p|1080p> <env K=V,K=V | -> <prompts json> [profile] [recipe] [weights]
#   bash ds.sh hash <cell>
#   bash ds.sh pair <a> <b>           compare-clips per common clip (LPIPS, PSNR, ...)
#   bash ds.sh results                out/results.tgz
#   bash ds.sh all <stage...>         run several "stage:args" in order (nohup'd by the caller)
set -u
B=${FV_BIN:-/e2e/ds/bin/fv-gpucheck}
W=/workspace/weights
WK=/e2e/ds/work
mkdir -p $WK/out $WK/compare $WK/libs
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
# cudarc dlopens the unversioned names (as scripts/gpu/runpod-matrix.sh links them).
. /opt/fastvideo-rs/scripts/gpu/cuda-13.pins
for spec in "cudnn:/lib/x86_64-linux-gnu/${CUDA_CUDNN_SONAME}" \
  "cublas:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLAS_SONAME}" \
  "cublasLt:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLASLT_SONAME}" \
  "nvrtc:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_NVRTC_SONAME}"; do
  n="${spec%%:*}"; s="${spec#*:}"
  [ -e "$WK/libs/lib$n.so" ] || { [ -e "$s" ] && ln -sf "$(readlink -f "$s")" "$WK/libs/lib$n.so"; }
done
# CUPTI for FASTVIDEO_GPU_TRACE_DECODE (as runpod-matrix.sh links it).
if [ ! -e "$WK/libs/libcupti.so" ]; then
  c="$(ldconfig -p 2>/dev/null | awk -v n="${CUDA_CUPTI_SONAME:-libcupti.so.13}" '$1 == n { print $NF; exit }')"
  [ -n "$c" ] && ln -sf "$(readlink -f "$c")" "$WK/libs/libcupti.so"
fi
export LD_LIBRARY_PATH="$WK/libs:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/fastvideo-rs/target/release:/usr/local/bin:$PATH"
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)
# Pinned kernel picks (as the tile-batching run): every cell deterministic.
PIN=(FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn FASTVIDEO_H3_VAE_CHECK=1)

geo() {
  case $1 in
    768p) echo "--height 768 --width 1344" ;;
    480p) echo "--height 480 --width 832" ;;
    1080p) echo "--height 1088 --width 1920 --oversize-canvas" ;;
    *) echo "bad res $1" >&2; exit 2 ;;
  esac
}

smi() {  # smi <file>: memory, SM clock, power once a second until killed
  nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 1000 >"$1" 2>/dev/null &
  echo $!
}

gen() {  # gen <cell> <res> <env|-> <prompts> [profile] [recipe] [weights]
  local c=$WK/$1 res=$2 ev=$3 ps=$4 prof=${5:-h3/fasth3_4step_vsa} recipe=${6:-4step-vsa} wts=${7:-h3-base}
  local envs=("${PIN[@]}")
  [ "$ev" != - ] && IFS=, read -r -a extra <<<"$ev" && envs+=("${extra[@]}")
  mkdir -p $c
  local pid t0=$(date +%s); pid=$(smi $c/smi.csv)
  # shellcheck disable=SC2046
  ( cd $c && env "${envs[@]}" timeout ${CELL_TIMEOUT:-2400} $B --techniques $prof --mode fast h3 gen \
      --weights $W/$wts --text-weights $W/h3-base --h3-recipe $recipe $(geo $res) --seconds 5 --prompt x \
      --prompts $ps --warm --text-encoder resident-fp8 --dit-offload resident \
      --adaln-cache $WK/adaln-$res-$recipe.cache --text-cache $WK/text-cache --clip-dir $c/frames ) >$c/log.txt 2>&1
  local rc=$?; kill $pid 2>/dev/null
  local peak; peak=$(cut -d, -f1 $c/smi.csv | sort -n | tail -1)
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":${peak:-0},\"res\":\"$res\",\"profile\":\"$prof\",\"env\":\"${envs[*]}\"}" >$c/run.json
  log "gen $1 rc=$rc wall=$(( $(date +%s) - t0 ))s peak=${peak}MiB res=$res env=$ev"
  grep -h "h3 vae check\|h3 vae fused ViT\|FASTVIDEO_H3_VAE_MXFP8" $c/log.txt | cut -c1-2500 | tee -a "$LOG"
  grep -h "h3 video decode + mp4" $c/log.txt | tee -a "$LOG"
  grep -h "h3/decode/gpu_trace" $c/log.txt | cut -c1-300 | tee -a "$LOG"
  grep -h "panicked\|Error\|error:" $c/log.txt | head -5 | tee -a "$LOG"
  hash_cell $1
}

hash_cell() {
  for d in $WK/$1/frames/*/; do
    [ -d "$d" ] || continue
    echo "$(basename $d) $(cat $d/frame-*.png 2>/dev/null | sha256sum | cut -c1-16)"
  done >$WK/$1/frames.sha
  sed "s/^/$1 /" $WK/$1/frames.sha | tee -a "$LOG"
}

pair() {  # every common (non-warm) clip of <a> and <b>
  local n=0 d p
  for d in $WK/$1/frames/*/; do
    p=$(basename $d); [[ $p == cold || $p == warmup ]] && continue
    [ -f $WK/$2/frames/$p/frame-000.png ] || continue
    $B --out $WK/compare --tag $1--$2-$p compare-clips --baseline $d --candidate $WK/$2/frames/$p "${LPIPS[@]}" \
      >$WK/compare/$1--$2-$p.out 2>&1 && n=$((n+1))
  done
  log "pair $1--$2: $n compares"
}

case "${1:?stage}" in
prep)
  [ -s /e2e/serve.pid ] && kill -TERM "$(cat /e2e/serve.pid)" 2>/dev/null
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.max.sm,power.limit --format=csv,noheader | tee -a $LOG
  chmod +x $B; $B --version 2>&1 | head -3 | tee -a $LOG
  sha256sum $B | tee -a $LOG
  ls -d $W/h3-base $W/h3-base/transformer_prequant_4step-vsa_mxfp8 $W/FastH3-4-step-Preview-v1-LoRA $W/auxiliary/lpips 2>&1 | tee -a $LOG
  df -h /root | tail -1 | tee -a $LOG
  ;;
bench)
  mkdir -p $WK/bench
  pid=$(smi $WK/bench/smi.csv)
  ( cd $WK/bench && timeout 1200 $B --out $WK/bench kernels --groups h3vae_bench ) >$WK/bench/log.txt 2>&1
  rc=$?; kill $pid 2>/dev/null
  log "bench rc=$rc"; tail -5 $WK/bench/log.txt | tee -a $LOG
  ;;
gen) shift; gen "$@" ;;
hash) hash_cell "$2" ;;
pair) pair "$2" "$3" ;;
results)
  ( cd $WK && tar czf out/results.tgz progress.log compare/*.json bench/*.json bench/log.txt bench/smi.csv \
      */benchmark.json */run.json */log.txt */frames.sha */smi.csv 2>/dev/null )
  ls -la $WK/out
  ;;
all)
  shift
  for st in "$@"; do
    IFS=: read -r -a a <<<"$st"
    log "== stage ${a[*]}"
    case ${a[0]} in
      gen) gen "${a[@]:1}" ;;
      pair) pair "${a[1]}" "${a[2]}" ;;
      *) bash "$0" "${a[@]}" ;;
    esac
  done
  log "== all done"
  ;;
esac
