#!/bin/bash
# H3 VAE decode tile batching: on-pod driver (one RTX PRO 6000 pod, the
# gpucheck runtime image, scripts/serve/e2e/pod.sh's sidecar). EU volume at
# /workspace is only read; everything is written under /e2e/vt/work.
#   bash vt.sh prep
#   bash vt.sh gen <cell> <res 768p|480p|1080p> <tile batch|default> <prompts json> [profile] [recipe] [weights]
#   bash vt.sh hash <cell>
#   bash vt.sh pair <a> <b>          compare-clips per common clip (LPIPS, PSNR, ...)
#   bash vt.sh results               out/results.tgz
#   bash vt.sh all <stage...>        run several "stage:args" in order (nohup'd by the caller)
set -u
B=${FV_BIN:-/opt/fastvideo-rs/target/release/fv-gpucheck}
W=/workspace/weights
WK=/e2e/vt/work
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
export LD_LIBRARY_PATH="$WK/libs:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/opt/fastvideo-rs/target/release:/usr/local/bin:$PATH"
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)
# Pinned kernel picks (as the plug-fast / sage-calibrated runs): every cell deterministic.
PIN=(FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn FASTVIDEO_H3_VAE_CHECK=1)

geo() {
  case $1 in
    768p) echo "--height 768 --width 1344" ;;
    480p) echo "--height 480 --width 832" ;;
    1080p) echo "--height 1088 --width 1920" ;;
    *) echo "bad res $1" >&2; exit 2 ;;
  esac
}

gen() {  # gen <cell> <res> <tile batch|default> <prompts> [profile] [recipe] [weights]
  local c=$WK/$1 res=$2 tb=$3 ps=$4 prof=${5:-h3/fasth3_4step_vsa} recipe=${6:-4step-vsa} wts=${7:-h3-base}
  local envs=("${PIN[@]}")
  [ "$tb" != default ] && envs+=(FASTVIDEO_H3_VAE_TILE_BATCH=$tb)
  mkdir -p $c
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 >$c/smi.csv 2>/dev/null &
  local smi=$! t0=$(date +%s)
  # shellcheck disable=SC2046
  ( cd $c && env "${envs[@]}" timeout ${CELL_TIMEOUT:-2400} $B --techniques $prof --mode fast h3 gen \
      --weights $W/$wts --text-weights $W/h3-base --h3-recipe $recipe $(geo $res) --seconds 5 --prompt x \
      --prompts $ps --warm --text-encoder resident-fp8 --dit-offload resident \
      --adaln-cache $c/adaln.cache --text-cache $c/text-cache --clip-dir $c/frames ) >$c/log.txt 2>&1
  local rc=$?; kill $smi 2>/dev/null
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":$(sort -n $c/smi.csv | tail -1),\"tile_batch\":\"$tb\",\"res\":\"$res\",\"profile\":\"$prof\",\"env\":\"${envs[*]}\"}" >$c/run.json
  log "gen $1 rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(sort -n $c/smi.csv | tail -1)MiB tb=$tb res=$res"
  grep -h "h3 vae check" $c/log.txt | cut -c1-2000 | tee -a "$LOG"
  grep -h "h3 video decode + mp4" $c/log.txt | tee -a "$LOG"
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
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | tee -a $LOG
  $B --version 2>&1 | head -3 | tee -a $LOG
  cat /opt/fastvideo-rs/target/release/fv-gpucheck.build-id 2>/dev/null | tee -a $LOG
  ls -d $W/h3-base $W/h3-base/transformer_prequant_4step-vsa_mxfp8 $W/FastH3-4-step-Preview-v1-LoRA $W/auxiliary/lpips 2>&1 | tee -a $LOG
  df -h /root | tail -1 | tee -a $LOG
  ;;
gen) shift; gen "$@" ;;
hash) hash_cell "$2" ;;
pair) pair "$2" "$3" ;;
results)
  ( cd $WK && tar czf out/results.tgz progress.log compare/*.json */benchmark.json */run.json */log.txt */frames.sha 2>/dev/null )
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
