#!/bin/bash
# wip/h3-plug-fast pod driver (docs/serve/research-longlive.md 12.8). EU volume
# at /workspace: weights read only; base frames kept under /workspace/scratch
# (new folder, temp name, sha256, rename); everything else under /root/pf.
#   bash pf.sh prep | h3 <arm...> | basecmp <arm...> | pair <a> <b> | hash <arm>
#              | keepbase | mp4 <arm> <name> <clip...> | sheet <name> <frame> <cols> <arm/clip...> | fsrv
set -u
P=/e2e/pf
B=/opt/fastvideo-rs/target/release/fv-gpucheck
W=/workspace/weights
WK=/root/pf
mkdir -p $WK $WK/out
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache FASTVIDEO_TAE_DIR=$W/auxiliary/tae
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)
PIN=(FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn)
cell() {  # cell <dir> <args...>: one fv-gpucheck process; nvidia-smi peak
  local c=$1; shift; mkdir -p $c
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 >$c/smi.csv 2>/dev/null &
  local smi=$! t0=$(date +%s)
  ( cd $c && env "${PIN[@]}" timeout ${CELL_TIMEOUT:-3000} $B "$@" ) >$c/log.txt 2>&1
  local rc=$?; kill $smi 2>/dev/null
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":$(sort -n $c/smi.csv | tail -1)}" >$c/run.json
  log "cell $(basename $c) rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(sort -n $c/smi.csv | tail -1)MiB sdpa_auto=$(grep -c 'sdpa auto' $c/log.txt) sage=$(grep -c 'attn_sage' $c/log.txt) sol=$(grep -c 'sol-attn' $c/log.txt) mxfp8=$(grep -ci 'mxfp8' $c/log.txt) frames_dirs=$(ls -d $c/frames/*/ 2>/dev/null | wc -l)"
}
cmp1() {  # cmp1 <tag> <baseline dir> <candidate dir>
  [ -f $WK/compare/compare-clips-$1.json ] || $B --out $WK/compare --tag $1 compare-clips --baseline $2 --candidate $3 "${LPIPS[@]}" >$WK/compare/$1.out 2>&1
  [ -f $WK/compare/compare-clips-$1.json ]
}
H3C=(h3 gen --weights $W/h3-base --text-weights $W/h3-base --prompt x --seconds 5 --text-encoder resident-fp8 --dit-offload resident)
case "${1:?stage}" in
prep)
  python3 - <<'PY'
import json
e=json.load(open('/e2e/pf/prompts-5x3.json'))
own=[p for p in e['prompts'] if p['name'] in ('h3-demo-s0','ltx-multishot-s42','ltx-newsbroadcast-s42','ltx-frogyoga-s42','spark-mountain-lake-s42')]
json.dump({'name':'h3-5-own-seed','prompts':own},open('/root/pf/prompts-5own.json','w'),indent=1)
print(len(e['prompts']), len(own))
PY
  [ -s /e2e/serve.pid ] && kill -TERM "$(cat /e2e/serve.pid)" 2>/dev/null; sleep 3
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader | tee -a $LOG
  ls -d $W/h3-base $W/longlive-plug $W/auxiliary/lpips $W/FastH3-4-step-Preview-v1-LoRA && log "weights present"
  df -h /workspace /root | tail -2; $B --version 2>&1 | head -2
  ;;
h3)
  shift; for arm in "$@"; do
    c=$WK/$arm; ps=$P/prompts-5x3.json; own=$WK/prompts-5own.json
    case $arm in
      pfast) a=(--techniques h3/sol_h3_4step_engine_ladder "${H3C[@]}" --prompts $ps --warm --h3-recipe h3-plug-4step --adaln-cache $c/adaln.cache) ;;
      max) a=(--techniques h3/sol_h3_4step_engine_ladder "${H3C[@]}" --prompts $ps --warm --h3-recipe sol-h3 --adaln-cache $c/adaln.cache) ;;
      base) a=("${H3C[@]}" --prompts $own --h3-recipe base --dense) ;;
      pdense) a=("${H3C[@]}" --prompts $own --warm --h3-recipe h3-plug-4step --dense) ;;
      pdense15) a=("${H3C[@]}" --prompts $ps --warm --h3-recipe h3-plug-4step --dense) ;;
    esac
    cell $c --mode fast "${a[@]}" --text-cache $c/text-cache --clip-dir $c/frames
  done
  ;;
basecmp)  # every clip of each arm vs its prompt's base clip (own seed)
  shift; mkdir -p $WK/compare
  for arm in "$@"; do n=0
    for d in $WK/$arm/frames/*/; do
      p=$(basename $d); [[ $p == cold || $p == warmup ]] && continue
      pr=${p%-s*}; bs=$(ls -d $WK/base/frames/$pr-s*/ 2>/dev/null | head -1); [ -n "$bs" ] || continue
      cmp1 base--$arm-$p $bs $d && n=$((n+1))
    done
    log "basecmp $arm: $n compares"
  done
  ;;
pair)  # pair <a> <b>: per common clip
  mkdir -p $WK/compare; n=0
  for d in $WK/$2/frames/*/; do
    p=$(basename $d); [[ $p == cold || $p == warmup ]] && continue
    [ -f $WK/$3/frames/$p/frame-000.png ] || continue
    cmp1 $2--$3-$p $d $WK/$3/frames/$p && n=$((n+1))
  done
  log "pair $2--$3: $n compares"
  ;;
hash)
  for d in $WK/$2/frames/*/; do echo "$(basename $d) $(cat $d/frame-*.png | sha256sum | cut -c1-16)"; done | tee $WK/$2/frames.sha
  ;;
keepbase)  # base frames onto the EU volume: new folder, temp name, sha256 verify, rename
  dst=/workspace/scratch/h3-plug-fast-base-768p-20261006; tmp=/workspace/scratch/.tmp-h3-plug-fast-base-768p-20261006
  [ -e $dst ] && { log "keepbase: $dst exists, untouched"; exit 0; }
  mkdir -p /workspace/scratch && rm -rf $tmp && mkdir -p $tmp
  ( cd $WK/base && find frames -type f | sort | xargs sha256sum ) > $WK/base/sha256.txt
  cp -r $WK/base/frames $tmp/ && cp $WK/base/sha256.txt $WK/base/benchmark.json $tmp/ 2>/dev/null
  ( cd $tmp && sha256sum --quiet -c sha256.txt ) && mv $tmp $dst && log "keepbase: $dst ($(du -sb $dst | cut -f1) bytes, $(wc -l <$dst/sha256.txt) files, sha256 ok)" || log "keepbase: verify FAILED, left at $tmp"
  ;;
mp4)  # mp4 <arm> <name> <clip...>: x264 crf 23 + aac
  c=$WK/$2; an=$3; shift 3
  for p in "$@"; do
    d=$c/frames/$p; pr=${p%-s*}; s=${p##*-s}; o=$WK/out/${an}__${pr}__s${s}.mp4
    src=$d/output.mp4; [ -f $src ] || src=$(ls $d/*.mp4 | head -1)
    if [ -f $d/audio.wav ]; then
      ffmpeg -nostdin -loglevel error -y -i $src -i $d/audio.wav -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k -shortest $o
    else
      ffmpeg -nostdin -loglevel error -y -i $src -map 0:v:0 -map 0:a? -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k $o
    fi
    echo "$(basename $o) $(stat -c %s $o) $(sha256sum $o | cut -c1-64)"
  done
  ;;
sheet)  # sheet <name> <frame> <cols> <arm/clip...>: tiled jpg (row-major) into out/
  n=$2; fr=$3; cols=$4; shift 4; t=$WK/sheet-$n; rm -rf $t; mkdir -p $t; i=0
  for x in "$@"; do cp $WK/${x%%/*}/frames/${x#*/}/frame-$fr.png $t/img-$(printf %03d $i).png; i=$((i+1)); done
  rows=$(( (i + cols - 1) / cols ))
  ffmpeg -nostdin -loglevel error -y -i $t/img-%03d.png -vf "scale=448:-2,tile=${cols}x${rows}" -frames:v 1 -q:v 3 $WK/out/$n.jpg && echo "$n.jpg $i tiles"
  ;;
results)  # results.tgz into out/: compare JSONs, benchmark/run/log per arm, hashes, progress
  ( cd $WK && tar czf out/results.tgz progress.log compare/*.json */benchmark.json */run.json */log.txt */frames.sha base/sha256.txt 2>/dev/null ); ls -la $WK/out
  ;;
fsrv)
  nohup python3 $P/fsrv.py >$WK/fsrv.log 2>&1 &
  echo $! >$WK/fsrv.pid; sleep 1; cat $WK/fsrv.pid
  ;;
esac
