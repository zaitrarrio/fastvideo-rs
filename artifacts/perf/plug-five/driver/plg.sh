#!/bin/bash
# wip/plug-ltx-gate pod driver (docs/perf/sage-attention.md 9.1,
# docs/serve/research-longlive.md 12.6). EU volume at /workspace: read only,
# everything written goes under /root/plg.
#   bash plg.sh prep | ltx [arms] | ltxcmp | h3 <arm...> | h3cmp <a> <b> | hash <arm> | mp4 <cell> <mp4-arm-name> <prompt-s...> | fsrv
set -u
P=/e2e/plg
B=/opt/fastvideo-rs/target/release/fv-gpucheck
W=/workspace/weights
WK=/root/plg
mkdir -p $WK $WK/out
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache FASTVIDEO_TAE_DIR=$W/auxiliary/tae
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
LPIPS=(); [ -f $W/auxiliary/lpips/.complete ] && LPIPS=(--lpips $W/auxiliary/lpips)
env_of() {
  case $1 in
    cud|plug|base|max|turbo) echo FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn ;;
    fw2|plugfw2) echo FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=v2 FASTVIDEO_CUDNN_SDPA_GRAPH=composite ;;
    sage) echo FASTVIDEO_ATTN_SAGE=2 FASTVIDEO_FLASH_KERNEL=cudnn ;;
  esac
}
# cell <dir> <arm> <args...>: one fv-gpucheck process; nvidia-smi peak.
cell() {
  local c=$1 arm=$2; shift 2; mkdir -p $c
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 >$c/smi.csv 2>/dev/null &
  local smi=$! t0=$(date +%s)
  # shellcheck disable=SC2046
  ( cd $c && env $(env_of $arm) timeout ${CELL_TIMEOUT:-3000} $B "$@" ) >$c/log.txt 2>&1
  local rc=$?; kill $smi 2>/dev/null
  echo "{\"rc\":$rc,\"wall_s\":$(( $(date +%s) - t0 )),\"peak_mib\":$(sort -n $c/smi.csv | tail -1)}" >$c/run.json
  log "cell $(basename $c) rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(sort -n $c/smi.csv | tail -1)MiB auto_lines=$(grep -c 'sdpa auto' $c/log.txt) sage_lines=$(grep -c 'attn_sage' $c/log.txt) frames_dirs=$(ls -d $c/frames/*/ 2>/dev/null | wc -l)"
}
pair() {  # pair <tagprefix> <dirA> <dirB>: compare-clips per common clip
  local t=$1 a=$2 b=$3 p n=0; mkdir -p $WK/compare
  for p in $a/frames/*/; do
    p=$(basename $p); [[ $p == cold || $p == warmup ]] && continue
    [ -f $b/frames/$p/frame-000.png ] || continue
    [ -f $WK/compare/compare-clips-$t-$p.json ] || $B --out $WK/compare --tag "$t-$p" compare-clips --baseline $a/frames/$p --candidate $b/frames/$p "${LPIPS[@]}" >$WK/compare/$t-$p.out 2>&1
    [ -f $WK/compare/compare-clips-$t-$p.json ] && n=$((n+1))
  done
  log "pairs $t: $n compares"
}
H3C=(h3 gen --weights $W/h3-base --text-weights $W/h3-base --prompt x --seconds 5 --text-encoder resident-fp8 --dit-offload resident)
case "${1:?stage}" in
prep)
  python3 - <<'PY'
import json
e=json.load(open('/e2e/plg/prompts-5x3.json'))
own=[p for p in e['prompts'] if p['name'] in ('h3-demo-s0','ltx-multishot-s42','ltx-newsbroadcast-s42','ltx-frogyoga-s42','spark-mountain-lake-s42')]
json.dump({'name':'h3-5-own-seed','prompts':own},open('/root/plg/prompts-5own.json','w'),indent=1)
l=json.load(open('/e2e/plg/prompts-ltx3.json'))
ps=[{'name':f"{p['name']}-s{s}",'seed':s,'prompt':p['prompt']} for p in l['prompts'] for s in (42,1042,2042)]
json.dump({'name':'ltx3x3','prompts':ps},open('/root/plg/prompts-ltx3x3.json','w'),indent=1)
print(len(own), len(ps))
PY
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $LOG
  ls $W/ltx25 $W/h3-base $W/auxiliary/lpips >/dev/null && log "weights present"; which ffmpeg; $B --version 2>&1 | head -2
  ;;
ltx)
  shift; for arm in ${*:-cud fw2 sage}; do
    c=$WK/ltx-$arm
    cell $c $arm --mode fast --techniques ltx2/ltx25_distill_dense ltx2 gen --model-version 2.5 --weights $W/ltx25 --dit $W/ltx25 \
      --two-stage --dense-stage2 --prompt x --prompts $WK/prompts-ltx3x3.json --width 1920 --height 1088 --num-frames 121 --frame-rate 24 \
      --warm --text resident --dit-offload resident --offload none --no-text-cache --clip $c/frames
  done
  ;;
ltxcmp)
  pair ltx-cud--fw2 $WK/ltx-cud $WK/ltx-fw2; pair ltx-cud--sage $WK/ltx-cud $WK/ltx-sage; pair ltx-fw2--sage $WK/ltx-fw2 $WK/ltx-sage
  ;;
h3)
  shift; for arm in "$@"; do
    c=$WK/$arm; ps=$P/prompts-5x3.json
    case $arm in
      base) a=(--prompts $WK/prompts-5own.json --h3-recipe base --dense) ;;
      plug|plugfw2) a=(--prompts $ps --warm --h3-recipe h3-plug-4step --dense) ;;
      max) a=(--prompts $ps --warm --h3-recipe sol-h3 --adaln-cache $c/adaln.cache) ;;
      turbo) a=(--prompts $ps --warm --h3-recipe 4step-vsa --adaln-cache $c/adaln.cache) ;;
    esac
    tech=(); [ $arm = max ] && tech=(--techniques h3/sol_h3_4step_engine_ladder); [ $arm = turbo ] && tech=(--techniques h3/fasth3_4step_vsa)
    cell $c $arm --mode fast "${tech[@]}" "${H3C[@]}" --text-cache $c/text-cache "${a[@]}" --clip-dir $c/frames
  done
  ;;
h3cmp)  # h3cmp <a> <b>
  pair $2--$3 $WK/$2 $WK/$3
  ;;
basecmp)  # every 4-step clip vs its prompt's base clip (own seed)
  shift; mkdir -p $WK/compare
  for arm in "$@"; do n=0
    for d in $WK/$arm/frames/*/; do
      p=$(basename $d); [[ $p == cold || $p == warmup ]] && continue
      pr=${p%-s*}; bs=$(ls -d $WK/base/frames/$pr-s*/ 2>/dev/null | head -1); [ -n "$bs" ] || continue
      t=base--$arm-$p
      [ -f $WK/compare/compare-clips-$t.json ] || $B --out $WK/compare --tag $t compare-clips --baseline $bs --candidate $d "${LPIPS[@]}" >$WK/compare/$t.out 2>&1
      [ -f $WK/compare/compare-clips-$t.json ] && n=$((n+1))
    done
    log "basecmp $arm: $n compares"
  done
  ;;
hash)  # hash <arm>: sha256 over each clip's frames
  for d in $WK/$2/frames/*/; do echo "$(basename $d) $(cat $d/frame-*.png | sha256sum | cut -c1-16)"; done | tee $WK/$2/frames.sha
  ;;
mp4)  # mp4 <cell> <arm-name> <clip...>: x264 crf 23 + aac from output.mp4 / audio.wav
  c=$WK/$2; an=$3; shift 3
  for p in "$@"; do
    d=$c/frames/$p; pr=${p%-s*}; s=${p##*-s}; o=$WK/out/${an}__${pr}__s${s}.mp4
    src=$d/output.mp4; [ -f $src ] || src=$(ls $d/*.mp4 | head -1)
    if [ -f $d/audio.wav ]; then
      ffmpeg -nostdin -loglevel error -y -i $src -i $d/audio.wav -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k -shortest $o
    else
      ffmpeg -nostdin -loglevel error -y -i $src -map 0:v:0 -map 0:a? -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k $o
    fi
    echo "$(basename $o) $(stat -c %s $o) $(sha256sum $o | cut -c1-64) audio=$(ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 $o)"
  done
  ;;
pack)  # pack <name> <paths relative to WK...>: a tgz under out/
  n=$2; shift 2; ( cd $WK && tar czf out/$n.tgz "$@" ) && echo "$n.tgz $(stat -c %s $WK/out/$n.tgz) $(sha256sum $WK/out/$n.tgz | cut -c1-64)"
  ;;
sheet)  # sheet <name> <frame> <cols> <cell/clip...>: tiled jpg (row-major) into out/
  n=$2; fr=$3; cols=$4; shift 4; t=$WK/sheet-$n; rm -rf $t; mkdir -p $t; i=0
  for x in "$@"; do cp $WK/${x%%/*}/frames/${x#*/}/frame-$fr.png $t/img-$(printf %03d $i).png; i=$((i+1)); done
  rows=$(( (i + cols - 1) / cols ))
  ffmpeg -nostdin -loglevel error -y -i $t/img-%03d.png -vf "scale=448:-2,tile=${cols}x${rows}" -frames:v 1 -q:v 3 $WK/out/$n.jpg && echo "$n.jpg $i tiles"
  ;;
fsrv)  # token-checked file server for /root/plg/out on :8000 (only while fv-serve is down)
  nohup python3 $P/fsrv.py >$WK/fsrv.log 2>&1 &
  echo $! >$WK/fsrv.pid; sleep 1; cat $WK/fsrv.pid
  ;;
esac
