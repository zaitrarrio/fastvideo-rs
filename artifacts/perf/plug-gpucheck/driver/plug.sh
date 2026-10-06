#!/bin/bash
# LongLive-Plug GPU check (docs/serve/research-longlive.md §12) on one RTX PRO 6000
# pod, EU volume at /workspace (read only by convention: all output under /root/plug).
#   bash plug.sh setup | h3a | h3b | h3base | w5 | w14 | cmp | sheets
set -u
D=/e2e/plug
B=$D/fv-gpucheck
W=/workspace/weights
WK=/root/plug
mkdir -p $WK
export RUST_LOG=info FASTVIDEO_CACHE=$WK/cache XDG_CACHE_HOME=$WK/cache FASTVIDEO_TAE_DIR=$W/auxiliary/tae
LOG=$WK/progress.log
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a "$LOG"; }
run() {  # run <cell> [K=V ...] -- <fv-gpucheck args>: one process, nvidia-smi peak
  local J=$WK/$1; shift; mkdir -p $J
  local envs=(); while [[ "$1" != -- ]]; do envs+=("$1"); shift; done; shift
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 >$J/smi.csv 2>/dev/null &
  local smi=$! t0=$(date +%s)
  ( cd $J && env "${envs[@]}" timeout ${CELL_TIMEOUT:-3600} $B --mode fast --out $J --tag $(basename $J) "$@" ) >$J/log.txt 2>&1
  local rc=$?; kill $smi 2>/dev/null
  log "cell $(basename $J) rc=$rc wall=$(( $(date +%s) - t0 ))s peak=$(cut -d, -f1 $J/smi.csv | sort -n | tail -1)MiB"
  grep -E 'plug lora|plug wan|plug [a-z0-9-]+:|h3 h3-plug|denoise=|fuse [0-9]+ pairs' $J/log.txt | head -8 | sed 's/^/    /' | tee -a "$LOG"
  return $rc
}
H3P=$WK/h3-prompts.json      # h3-demo, ltx-frogyoga, spark-mountain-lake
H3P2=$WK/h3-prompts2.json    # h3-demo, spark-mountain-lake (49-forward cells)
H3P1=$WK/h3-prompts1.json    # spark-mountain-lake (768p base)
WP=$WK/wan-prompts.json      # beach_dog, city_rain, spark-mountain-lake
WP2=$WK/wan-prompts2.json    # beach_dog, spark-mountain-lake (14B base)
H3C=(h3 gen --weights $W/h3-base --text-weights $W/h3-base --prompt unused --text-cache $WK/h3-text-cache)
H3=("${H3C[@]}" --prompts $H3P --warm)
case "${1:?stage}" in
setup)
  ls -la $W/h3-base $W/wan22-ti2v-5b $W/fastwan22-ti2v-5b $W/wan21-t2v-14b $W/longlive-plug/* $W/auxiliary/lpips $W/FastH3-4-step-Preview-v1-LoRA 2>&1 | tee -a $LOG
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $LOG
  free -g | tee -a $LOG; nproc | tee -a $LOG
  # H3: three of the five gate prompts (scripts/gpu/prompts-eval.json, their own seeds).
  python3 - <<'PY'
import json
e=json.load(open('/e2e/plug/prompts-eval.json'))
keep=['h3-demo','ltx-frogyoga','spark-mountain-lake']
json.dump({'prompts':[p for p in e['prompts'] if p['name'] in keep]},open('/root/plug/h3-prompts.json','w'),indent=1)
w=json.load(open('/e2e/plug/prompts.json'))
ps=[dict(p, seed=1024) for p in w['prompts']]+[p for p in e['prompts'] if p['name']=='spark-mountain-lake']
json.dump({'prompts':ps},open('/root/plug/wan-prompts.json','w'),indent=1)
sub=lambda ps,names: {'prompts':[p for p in ps if p['name'] in names]}
h3=[p for p in e['prompts'] if p['name'] in keep]
json.dump(sub(h3,['h3-demo','spark-mountain-lake']),open('/root/plug/h3-prompts2.json','w'),indent=1)
json.dump(sub(h3,['spark-mountain-lake']),open('/root/plug/h3-prompts1.json','w'),indent=1)
json.dump(sub(ps,['beach_dog','spark-mountain-lake']),open('/root/plug/wan-prompts2.json','w'),indent=1)
print('prompts ok')
PY
  { python3 -m venv $WK/venv && $WK/venv/bin/pip install -q numpy pillow; } >$WK/venv.log 2>&1
  log "venv exit=$?"
  ;;
h3a)  # 768p: the Plug 4-step, its bf16 control, FastH3 4-step (VSA = h3-turbo; dense)
  run h3-plug4 -- "${H3[@]}" --h3-recipe h3-plug-4step --dense --clip-dir $WK/h3-plug4/frames
  run h3-plug4-ctl FASTVIDEO_CUDNN_SDPA_GRAPH=composite -- "${H3[@]}" --h3-recipe h3-plug-4step --dense --clip-dir $WK/h3-plug4-ctl/frames
  run h3-turbo -- "${H3[@]}" --h3-recipe 4step-vsa --clip-dir $WK/h3-turbo/frames
  ;;
h3b)  # 480p, 49 forwards: the Plug CFG LoRA vs base, and a bf16 control of base
  G=(--prompts $H3P2 --height 480 --width 832 --dense)
  run h3-base480 -- "${H3C[@]}" "${G[@]}" --h3-recipe base --clip-dir $WK/h3-base480/frames
  run h3-plugcfg480 -- "${H3C[@]}" "${G[@]}" --h3-recipe h3-plug-cfg --clip-dir $WK/h3-plugcfg480/frames
  run h3-base480-ctl FASTVIDEO_CUDNN_SDPA_GRAPH=composite -- "${H3C[@]}" "${G[@]}" --h3-recipe base --clip-dir $WK/h3-base480-ctl/frames
  ;;
h3base)  # 768p base, 49 forwards (the reference the 4-step recipes distil)
  run h3-base768 -- "${H3C[@]}" --prompts $H3P1 --h3-recipe base --dense --clip-dir $WK/h3-base768/frames
  ;;
h3spark)  # pod 2: the 4-step recipes again on the 768p-base prompt, for the base comparison
  run h3-plug4s -- "${H3C[@]}" --prompts $H3P1 --h3-recipe h3-plug-4step --dense --clip-dir $WK/h3-plug4s/frames
  run h3-turbos -- "${H3C[@]}" --prompts $H3P1 --h3-recipe 4step-vsa --clip-dir $WK/h3-turbos/frames
  ;;
w5)
  for r in 720 480; do
    if [ $r = 720 ]; then G=(--height 704 --width 1280); else G=(--height 480 --width 832); fi
    C=(wan gen --prompts $WP --num-frames 121 --fps 24 --full-vae --text-cache $WK/wan-text-cache "${G[@]}")
    run w5-plug$r -- "${C[@]}" --warm --weights $W/wan22-ti2v-5b --preset wan_2_2_ti2v_5b --plug wan5b-plug-4step --clip-dir $WK/w5-plug$r/frames
    [ $r = 720 ] && run w5-plug$r-ctl FASTVIDEO_CUDNN_SDPA_GRAPH=composite -- "${C[@]}" --warm --weights $W/wan22-ti2v-5b --preset wan_2_2_ti2v_5b --plug wan5b-plug-4step --clip-dir $WK/w5-plug$r-ctl/frames
    run w5-turbo$r -- "${C[@]}" --warm --weights $W/fastwan22-ti2v-5b --preset fast_wan_2_2_ti2v_5b --steps 3 --flow-shift 5 --clip-dir $WK/w5-turbo$r/frames
    run w5-max$r -- "${C[@]}" --weights $W/wan22-ti2v-5b --preset wan_2_2_ti2v_5b --unipc --steps 50 --guidance 5 --flow-shift 5 --clip-dir $WK/w5-max$r/frames
  done
  ;;
w14)
  C=(wan gen --weights $W/wan21-t2v-14b --preset wan_t2v_14b --height 480 --width 832 --num-frames 81 --fps 16 --full-vae --text-cache $WK/wan-text-cache)
  run w14-plug -- "${C[@]}" --prompts $WP --warm --plug wan14b-plug-4step --clip-dir $WK/w14-plug/frames
  run w14-base -- "${C[@]}" --prompts $WP2 --unipc --steps 50 --guidance 5 --flow-shift 5 --clip-dir $WK/w14-base/frames
  ;;
cmp)  # cmp <baseline cell> <candidate cell>: compare-clips per prompt (LPIPS on the GPU)
  b=$2; c=$3; mkdir -p $WK/compare
  for p in $WK/$b/frames/*/; do
    p=$(basename $p); [ "$p" = cold ] || [ "$p" = warmup ] && continue
    [ -d $WK/$c/frames/$p ] || continue
    $B --out $WK/compare --tag "$b--$c-$p" compare-clips --baseline $WK/$b/frames/$p --candidate $WK/$c/frames/$p --lpips $W/auxiliary/lpips \
      >$WK/compare/$b--$c-$p.out 2>&1
    log "cmp $b--$c-$p rc=$?"
  done
  ;;
sheets)  # sheets <name> <frame> <cell...>: one row per prompt, one column per cell
  name=$2; fr=$3; shift 3
  $WK/venv/bin/python $D/sheet.py $WK $name $fr "$@" && log "sheet $name ok"
  ;;
esac
