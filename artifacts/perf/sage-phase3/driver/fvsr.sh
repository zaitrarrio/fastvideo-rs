#!/bin/bash
# FlashVSR x2 on the 384p stage-1 clips (the e stage's FlashVSR block, rerun after a missing dependency).
set -u
WK=/root/p3; UP=$WK/up; W=/workspace/weights; P=/e2e/p3; B=$P/fv-gpucheck
log() { echo "$(date -u +%FT%TZ) | $*" | tee -a $WK/progress.log; }
clips="ltx-multishot ltx-newsbroadcast ltx-frogyoga"
PATH=$UP/bin:$PATH uv pip install -q --python $UP/fvenv/bin/python modelscope >$WK/e/modelscope.log 2>&1
log "modelscope install exit=$?"
R=$WK/e/fvsr-runs; mkdir -p $R/turbo-768p/frames
for cl in $clips; do ln -sfn $WK/e/ltx384-s1/frames/$cl $R/turbo-768p/frames/$cl; done
mkdir -p $UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1
for f in $W/auxiliary/upscalers/flashvsr-v1.1/*; do ln -sf "$f" $UP/FlashVSR/examples/WanVSR/FlashVSR-v1.1/$(basename "$f"); done
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 >$WK/e/fvsr.smi 2>/dev/null &
smi=$!
timeout 1500 $UP/fvenv/bin/python $P/hd_flashvsr.py $UP/FlashVSR $R 2 $clips >$WK/e/flashvsr.jsonl 2>$WK/e/flashvsr.log
rc=$?; kill $smi 2>/dev/null
log "flashvsr exit=$rc peak=$(sort -n $WK/e/fvsr.smi | tail -1)MiB $(tail -1 $WK/e/flashvsr.jsonl)"
for cl in $clips; do
  d=$R/flashvsr-v1.1-x2/frames/$cl; [ -f $d/frame-000.png ] || continue
  ref=$WK/e/ref1280/$cl; mkdir -p $ref
  [ -f $ref/frame-000.png ] || ffmpeg -nostdin -loglevel error -y -start_number 0 -i $WK/e/ltx768-2s/frames/$cl/frame-%03d.png \
    -vf crop=1280:768 -start_number 0 $ref/frame-%03d.png
  $B --out $WK/compare --tag "ltx768-2s--fvsr-x2-$cl" compare-clips --baseline $ref --candidate $d --lpips $W/auxiliary/lpips \
    >$WK/compare/ltx768-2s--fvsr-x2-$cl.out 2>&1
  log "compare ltx768-2s--fvsr-x2-$cl exit=$?"
done
