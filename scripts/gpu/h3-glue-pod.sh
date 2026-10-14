#!/usr/bin/env bash
# Runs ON the H3 glue-fusion timing pod (scripts/gpu/h3-glue-validate.sh; docs/perf/h3-glue-fusion.md). One RTX PRO
# 6000 on the runtime image, two fv-gpucheck binaries in the bundle: `base` (the branch point) and `new` (the fusions).
#
#   1. fetch FastH3 V2 NVFP4's transformer/ from R2 (plan-nvfp4.json, presigned) in the background; in r2 mode first
#      the h3-turbo trees (plan-turbo.json: h3-base, its pre-quantized MXFP8 DiT, the LoRA); both models stream
#      the bf16 text encoder once per prompt (text cache), so the denoise timing has the GPU to itself;
#   2. `new kernels --groups dit_fusion nvfp4_linear`: every glue fusion against the ops it replaces, bit for bit;
#   3. timing, warm, one process per (model, binary), base and new back to back: h3-turbo (the served default,
#      MXFP8) and V2-NVFP4 (FASTVIDEO_H3_NVFP4_ATTN=1: NVFP4 attention + FFN), h3-demo at 832x480 and 1344x768,
#      124 frames (5 s), seed 1024; each process generates 480p cold first (--warm);
#   4. one CUPTI kernel trace (FASTVIDEO_GPU_TRACE=1) of a 480p generation per (model, binary);
#   5. parity: the clips base and new wrote, file by file (sha256), and compare-clips where any differs.
#
# Nothing is written to /workspace. /srv is served on :8000: log.txt, status.tsv (step, rc, seconds), the logs,
# benchmark JSONs, results.tgz and DONE (0 = every step ok).
# Env: FV_WEIGHTS_MODE (volume | r2), FV_SKIP (space-separated step names).
set -u
FV_WEIGHTS_MODE="${FV_WEIGHTS_MODE:-volume}"
SRV=/srv
L=/fvw
W=/workspace/weights
[[ "$FV_WEIGHTS_MODE" == r2 ]] && W=$L
mkdir -p "$SRV" "$L"
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
step() {
  local name="$1" t0=$SECONDS rc
  shift
  if [[ " ${FV_SKIP:-} " == *" $name "* ]]; then log "skip $name"; return 0; fi
  log "begin $name"
  "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>$SRV/status.tsv
  log "end $name rc=$rc $((SECONDS - t0))s"
  return "$rc"
}
finish() {
  tar -czf $SRV/results.tgz --exclude='*.png' --exclude='*.wav' --exclude=results.tgz -C $SRV . 2>/dev/null
  echo "$1" >$SRV/DONE
}
B=/opt/fv/base/fv-gpucheck
N=/opt/fv/new/fv-gpucheck
log "start: weights $W ($FV_WEIGHTS_MODE); base $($B --version 2>/dev/null | head -1); new $($N --version 2>/dev/null | head -1)"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
# A host whose GPU memory is already held (2026-10-13: 95.8 of 97.9 GiB used, no process of ours) cannot run
# anything: stop at once and delete this pod (Runpod's pod-scoped key).
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do printf '%s,' "$(date +%s.%N)"; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights ---------------------------------------------------------------------------------------------------
fetch() { "$PY" /opt/fv/r2-fetch.py --root "$L" --plan "$1" --status $SRV/fetch-status-$2.json >>$SRV/fetch-$2.log 2>&1; }
V2REV=0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0
# The small V2 configs no R2 tree carries (byte-identical to h3-base's on the Hub; docs/ports/fasth3-nvfp4.md).
small() {
  local f
  for f in fastvideo_inference.json modular_model_index.json checkpoint_content.json checkpoint_metadata.json \
           scheduler/scheduler_config.json audio_scheduler/scheduler_config.json processor/chat_template.json \
           processor/merges.txt processor/preprocessor_config.json processor/tokenizer.json processor/tokenizer_config.json \
           processor/video_preprocessor_config.json processor/vocab.json; do
    mkdir -p "$L/v2/$(dirname "$f")"
    curl -fsSL --retry 5 -o "$L/v2/$f" "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4/resolve/$V2REV/$f" \
      || { log "V2 $f: download failed"; return 1; }
  done
}
nvfp4_weights() {
  fetch /opt/fv/plan-nvfp4.json nvfp4 || return 1
  small || return 1
  # The ModelOpt transformer as published; text encoder, tokenizer, audio VAE and the official VAE are h3-base's.
  ln -sfn $L/fasth3-v2-nvfp4/transformer $L/v2/transformer
  for x in text_encoder tokenizer audio_vae vae; do ln -sfn "$W/h3-base/$x" "$L/v2/$x"; done
}
if [[ "$FV_WEIGHTS_MODE" == r2 ]]; then
  step fetch-turbo fetch /opt/fv/plan-turbo.json turbo & tpid=$!
else
  tpid=""
fi
# r2 mode: the NVFP4 tree after the turbo trees (one stream at a time is as fast as two).
( [[ -n "$tpid" ]] && wait "$tpid"; step fetch-nvfp4 nvfp4_weights ) & npid=$!

# ---- 2. kernels ---------------------------------------------------------------------------------------------------
step kernels "$N" --mode fast --out $SRV/gc --keep-going kernels --groups dit_fusion,nvfp4_linear --seed 17 \
  >$SRV/kernels.log 2>&1

# ---- prompts: h3-demo (scripts/gpu/prompts-hd5.json) at 480p and 768p, seed 1024 -----------------------------------
"$PY" - /opt/fv/prompts-hd5.json "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
p = {q["name"]: q["prompt"] for q in json.load(open(src))["prompts"]}["h3-demo"]
json.dump({"name": "h3-glue", "source": "scripts/gpu/prompts-hd5.json (h3-demo) at seed 1024",
           "prompts": [{"name": f"h3-demo-{c}", "prompt": p, "seed": 1024, "height": h, "width": w}
                       for c, (h, w) in (("480p", (480, 832)), ("768p", (768, 1344)))]},
          open(f"{out}/prompts.json", "w"), indent=1)
open(f"{out}/prompt.txt", "w").write(p)
PYEOF
P="$(cat $L/prompt.txt)"

# ---- 3./4. timing and traces --------------------------------------------------------------------------------------
turbo() { # <bin> <dir> <args...>
  local bin="$1" dir="$2"
  shift 2
  mkdir -p "$dir"
  (cd "$dir" && timeout 1500 "$bin" --mode fast --techniques h3/fasth3_4step_vsa h3 gen --weights "$W/h3-base" \
     --h3-recipe 4step-vsa --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 \
     --text-encoder streamed --text-cache $L/text-cache-turbo --clip-dir "$dir/clips" "$@") >"$dir/stdout.log" 2>"$dir/stderr.log"
}
nvfp4() { # <bin> <dir> <args...>
  local bin="$1" dir="$2"
  shift 2
  mkdir -p "$dir"
  (cd "$dir" && env FASTVIDEO_H3_NVFP4_ATTN=1 timeout 1500 "$bin" --mode fast h3 gen --weights $L/v2 \
     --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 --text-encoder streamed \
     --text-weights "$W/h3-base" --text-cache $L/text-cache-v2 --adaln-cache $L/adaln-v2.cache \
     --clip-dir "$dir/clips" "$@") >"$dir/stdout.log" 2>"$dir/stderr.log"
}
# A CUPTI kernel trace of one generation (benchmark.json `gpu_trace`).
traced() { (export FASTVIDEO_GPU_TRACE=1; "$@"); }
[[ -n "$tpid" ]] && wait "$tpid"
for bin in base new; do
  b=$B; [[ $bin == new ]] && b=$N
  step "turbo-$bin" turbo "$b" $SRV/turbo-$bin --prompts $L/prompts.json --warm
done
for bin in base new; do
  b=$B; [[ $bin == new ]] && b=$N
  step "trace-turbo-$bin" traced turbo "$b" $SRV/trace-turbo-$bin --no-mp4
done
wait "$npid"
for bin in base new; do
  b=$B; [[ $bin == new ]] && b=$N
  step "nvfp4-$bin" nvfp4 "$b" $SRV/nvfp4-$bin --prompts $L/prompts.json --warm
done
for bin in base new; do
  b=$B; [[ $bin == new ]] && b=$N
  step "trace-nvfp4-$bin" traced nvfp4 "$b" $SRV/trace-nvfp4-$bin --no-mp4
done

# ---- 5. parity: base and new clips, file by file --------------------------------------------------------------------
parity() {
  local m c
  for m in turbo nvfp4; do
    for c in h3-demo-480p h3-demo-768p; do
      [[ -d $SRV/$m-base/clips/$c && -d $SRV/$m-new/clips/$c ]] || { echo "$m $c: missing"; continue; }
      (cd $SRV/$m-base/clips/$c && find . -type f | sort | xargs sha256sum) >$SRV/sha-$m-base-$c.txt
      (cd $SRV/$m-new/clips/$c && find . -type f | sort | xargs sha256sum) >$SRV/sha-$m-new-$c.txt
      if cmp -s $SRV/sha-$m-base-$c.txt $SRV/sha-$m-new-$c.txt; then
        echo "$m $c: identical ($(wc -l <$SRV/sha-$m-base-$c.txt) files)"
      else
        echo "$m $c: DIFFERENT"
        "$N" --out $SRV/cmp-$m-$c --tag "base--new-$m-$c" compare-clips --baseline $SRV/$m-base/clips/$c \
          --candidate $SRV/$m-new/clips/$c >$SRV/cmp-$m-$c.log 2>&1
      fi
    done
  done
}
step parity parity >$SRV/parity.txt 2>&1

bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
finish $(( bad > 0 ? 1 : 0 ))
