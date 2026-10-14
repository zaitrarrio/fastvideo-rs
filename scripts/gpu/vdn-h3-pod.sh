#!/usr/bin/env bash
# Runs ON the VDN-H3 validation pod (scripts/gpu/vdn-h3-validate.sh; docs/ports/vdn-h3.md §5). One RTX PRO 6000 on
# the runtime image; h3-base from the EU weight volume at /workspace (or from R2 in r2 mode):
#
#   1. fetch vdn-h3-stage-b-2000 (+ h3-8step's DiT for the timing, in the background) from R2 (plan.json,
#      presigned) while uv builds the reference environment: the code repo's pins (torch 2.13.0+cu129, the patched
#      diffusers at its BASE.txt commit, transformers 5.15.0 ...);
#   2. `fv-gpucheck kernels --groups vdn`: every VDN kernel against the host reference;
#   3. parity: ours once with a dump (OUR text encoder's hidden states), the upstream diffusers component on those
#      same embeddings (scripts/gpu/upstream/oracle_vdn.py, dumps in our format, the decoded clip), then ours with
#      the reference's noise and text injected (FASTVIDEO_INJECT_DIR), bf16 linears and the served MXFP8, each
#      compared (`compare-dumps`, av_psnr.py);
#   4. timing, warm, served defaults: VDN-H3 / FastH3 V2 / h3-turbo at 480p and 768p 5 s, VDN-H3 at 768p 14.4 s,
#      nvidia-smi sampled every second; a 50-step VDN-H3 sample (the stage-B model's own schedule).
#
# Nothing is written to /workspace. /srv is served on :8000: log.txt, status.tsv (step, rc, seconds), *.log, the
# comparisons, the clips and DONE (0 = every step ok).
# Env: FV_PROMPT, FV_SEED, FV_H/FV_W/FV_FRAMES (parity geometry, default 832x480x124), FV_STEPS (8), FV_SKIP
# (space-separated step names), FV_UP_KERNELS / FV_UP_SOFTMAX (oracle_vdn.py, default eager / ref),
# FV_VDN_MODEL (stage-b, default: the stage-B checkpoint; 8step: the DMD model, vdn-h3-dmd-250-turbo over stage B),
# FV_INJ_ARMS (the injected runs, <name>:<FASTVIDEO_H3_QUANT or "default">, default "inj:off inj-served:default").
set -u
SRV=/srv
L=/fvw
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
STAGE=""
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
P="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
SEED="${FV_SEED:-1024}"
H="${FV_H:-480}"; WD="${FV_W:-832}"; NF="${FV_FRAMES:-124}"; STEPS="${FV_STEPS:-8}"
BIN=/opt/fv/fv-gpucheck
[[ -x "$BIN" ]] || BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
log "start: parity ${H}x${WD}x${NF} ${STEPS} steps seed $SEED; weights $W"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights and the reference environment --------------------------------------------------------------------
fetch() { "$PY" /opt/fv/r2-fetch.py --root "$L" --plan "$1" --status $SRV/fetch-status-$2.json >>$SRV/fetch-$2.log 2>&1; }
VPY=/opt/vdnpy/bin/python
vdn_setup() {
  (
    export UP=/opt/up
    . /opt/fv/upstream/setup.sh && ensure_base || exit 1
    uv venv -q --python "$PY" /opt/vdnpy || exit 1
    uv pip install -q --python $VPY "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu129 || exit 1
    git clone -q --filter=blob:none https://github.com/huggingface/diffusers.git /opt/diffusers \
      && git -C /opt/diffusers checkout -q 3a2f35d || exit 1
    # The code repo's diffusers_patches/ (one patch: the AdaLN SiLU pinned to fp32, a no-op at inference where
    # temb is already fp32), at the pinned code revision.
    curl -fsSL -o /opt/vdn.patch "https://raw.githubusercontent.com/OpenVDN/vdn-minimax-h3/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/diffusers_patches/0001-MiniMaxH3-pin-the-AdaLN-SiLU-to-fp32-explicitly.patch" \
      && git -C /opt/diffusers -c user.name=x -c user.email=x@x am -q /opt/vdn.patch || exit 1
    echo "diffusers $(git -C /opt/diffusers rev-parse HEAD) (+ patch: $(git -C /opt/diffusers log -1 --format=%s))"
    uv pip install -q --python $VPY /opt/diffusers "transformers==5.15.0" "accelerate==1.14.0" "safetensors==0.8.0" \
      "numpy==2.5.2" "av==18.1.0" "pillow==12.3.0" "huggingface_hub>=1.27.0" || exit 1
    $VPY -c "import torch, diffusers, triton; print('torch', torch.__version__, torch.version.cuda, 'diffusers', diffusers.__version__, 'triton', triton.__version__)"
  ) >>$SRV/setup.log 2>&1
}
step fetch fetch /opt/fv/plan.json vdn & fpid=$!
step vdn-setup vdn_setup & upid=$!
wait "$fpid"
STAGE="$(ls -d $L/vdn-h3-stage-b-2000/*/stage-b-step-2000 $L/vdn-h3-stage-b-2000/stage-b-step-2000 2>/dev/null | head -1)"
[[ -n "$STAGE" && -f "$STAGE/diffusers/config.json" ]] || { log "no VDN tree under $L"; ls -R $L | head -40 >>$SRV/log.txt; echo 1 >$SRV/DONE; exit 1; }
# What ours loads (FASTVIDEO_H3_VDN / _SHARED) and what the reference loads (one complete stage directory).
OURS_VDN="$STAGE"; OURS_SHARED=""; UP_STAGE="$STAGE"; RECIPE="vdn-h3-${STEPS}step"
if [[ "${FV_VDN_MODEL:-stage-b}" == 8step ]]; then
  TURBO="$(ls -d $L/vdn-h3-dmd-250-turbo/*/stage-dmd-step-250 $L/vdn-h3-dmd-250-turbo/stage-dmd-step-250 2>/dev/null | head -1)"
  [[ -n "$TURBO" && -f "$TURBO/diffusers/config.json" ]] || { log "no turbo tree under $L"; echo 1 >$SRV/DONE; exit 1; }
  OURS_VDN="$TURBO"; OURS_SHARED="$STAGE"; RECIPE=vdn-h3-8step
  # The reference reads stage-dmd-step-250 whole: the turbo tree's files plus stage B's byte-identical branch,
  # default adapter and remote code, linked (nothing copied).
  UP_STAGE=$L/vdn8/stage-dmd-step-250
  mkdir -p "$UP_STAGE"
  cp -rs "$TURBO/." "$UP_STAGE/"
  for f in linear_branch/model.safetensors adapters/default/adapter_model.safetensors diffusers/modeling_vdn_h3.py; do
    [[ -e "$UP_STAGE/$f" ]] || { mkdir -p "$(dirname "$UP_STAGE/$f")"; ln -s "$STAGE/$f" "$UP_STAGE/$f"; }
  done
fi
log "VDN checkpoint: ours $OURS_VDN (shared ${OURS_SHARED:-none}), reference $UP_STAGE, recipe $RECIPE"
step fetch-v2 fetch /opt/fv/plan-v2.json v2 & v2pid=$!

# ---- 2. the kernels ---------------------------------------------------------------------------------------------
step kernels "$BIN" --mode fast --out $SRV/gc --keep-going kernels --groups vdn --seed 17 >$SRV/kernels.log 2>&1

# ---- 3. parity -------------------------------------------------------------------------------------------------
GEN=("$BIN" --mode fast h3 gen --prompt "$P" --seed "$SEED" --text-cache $L/text-cache)
vdn() { env FASTVIDEO_H3_VDN="$OURS_VDN" FASTVIDEO_H3_VDN_SHARED="$OURS_SHARED" "$@"; }
# a. ours on our own noise, dumping our text encoder's hidden states (bf16 linears, the streamed bf16 encoder).
ours_text() {
  vdn env FASTVIDEO_H3_QUANT=off FASTVIDEO_DUMP_DIR=$SRV/ours/text-dump \
    "${GEN[@]}" --height "$H" --width "$WD" --num-frames "$NF" --weights "$W/h3-base" --h3-recipe "$RECIPE" \
    --text-encoder streamed --dit-offload resident --clip-dir $SRV/ours/seed-bf16 >>$SRV/ours-text.log 2>&1
}
step ours-text ours_text
wait "$upid"
# b. upstream on our text, its own noise
upstream() {
  mkdir -p $SRV/up
  $VPY /opt/fv/upstream/oracle_vdn.py --base-parent "$W" --stage "$UP_STAGE" --text-hidden $SRV/ours/text-dump \
    --height "$H" --width "$WD" --frames "$NF" --steps "$STEPS" --seed "$SEED" \
    --kernels "${FV_UP_KERNELS:-eager}" --softmax "${FV_UP_SOFTMAX:-ref}" --dump $SRV/up/dump --out $SRV/up \
    >>$SRV/upstream.log 2>&1
}
step upstream upstream
# c. ours on the reference's noise and text: bf16 linears (the reference's), then the served default (MXFP8)
ours_injected() { # <name> <quant | default: FASTVIDEO_H3_QUANT unset, the served default>
  local q=(env)
  [[ "$2" == default ]] || q=(env FASTVIDEO_H3_QUANT="$2")
  vdn "${q[@]}" FASTVIDEO_INJECT_DIR=$SRV/up/dump FASTVIDEO_DUMP_DIR=$SRV/ours/$1-dump \
    "${GEN[@]}" --height "$H" --width "$WD" --num-frames "$NF" --weights "$W/h3-base" --h3-recipe "$RECIPE" \
    --text-encoder streamed --dit-offload resident --clip-dir $SRV/ours/$1 >>$SRV/ours-$1.log 2>&1
}
for arm in ${FV_INJ_ARMS:-inj:off inj-served:default}; do
  name="${arm%%:*}"
  step "ours-$name" ours_injected "$name" "${arm#*:}"
  step "compare-dumps-$name" "$BIN" compare-dumps --baseline $SRV/up/dump --candidate $SRV/ours/$name-dump >$SRV/compare-dumps-$name.txt 2>&1
  step "av-psnr-$name" $VPY /opt/fv/upstream/av_psnr.py $SRV/up $SRV/ours/$name --json $SRV/av-psnr-$name.json >$SRV/av-psnr-$name.log 2>&1
done

# ---- 4. timing: served defaults, warm (one cold generation first), one process per model and geometry -------------
timed() { # <name> <height> <width> <frames> <args...>
  local name="$1" h="$2" w="$3" nf="$4"
  shift 4
  "$@" --height "$h" --width "$w" --num-frames "$nf" --clip-dir $SRV/ours/$name --warm >$SRV/time-$name.log 2>&1
}
for g in 480:832:124 768:1344:124; do
  IFS=: read -r gh gw gn <<<"$g"
  step "time-vdn-${gh}" timed "vdn-$gh" "$gh" "$gw" "$gn" vdn "${GEN[@]}" --weights "$W/h3-base" --h3-recipe vdn-h3-8step \
    --dit-offload resident
  step "time-turbo-${gh}" timed "turbo-$gh" "$gh" "$gw" "$gn" "$BIN" --mode fast --techniques h3/fasth3_4step_vsa h3 gen \
    --prompt "$P" --seed "$SEED" --text-cache $L/text-cache --weights "$W/h3-base" --h3-recipe 4step-vsa
done
wait "$v2pid"
for g in 480:832:124 768:1344:124; do
  IFS=: read -r gh gw gn <<<"$g"
  step "time-v2-${gh}" timed "v2-$gh" "$gh" "$gw" "$gn" "${GEN[@]}" --weights "$L/h3-8step" --text-weights "$W/h3-base" \
    --h3-recipe 8step --dense --adaln-cache $L/v2-adaln.cache
done
# The long clip (14.4 s, 345 frames, 102 latent frames): the hybrid needs the DiT resident, so the encoder streams.
# 768p may not fit 96 GB (upstream: 85-103 GB on an H200 in bf16); 480p is the fallback measurement.
step time-vdn-768-long timed "vdn-768-long" 768 1344 345 vdn "${GEN[@]}" --weights "$W/h3-base" --h3-recipe vdn-h3-8step \
  --text-encoder streamed --dit-offload resident
step time-vdn-480-long timed "vdn-480-long" 480 832 345 vdn "${GEN[@]}" --weights "$W/h3-base" --h3-recipe vdn-h3-8step \
  --text-encoder streamed --dit-offload resident
# The stage-B model on its own 50-step schedule: a quality sample, not a timing.
step sample-vdn-50 vdn "${GEN[@]}" --weights "$W/h3-base" --h3-recipe vdn-h3-50step --height 480 --width 832 --num-frames 124 \
  --clip-dir $SRV/ours/vdn50-480 >$SRV/sample-vdn-50.log 2>&1

tar -cf $SRV/dumps.tar -C $SRV up/dump ours/text-dump ours/inj-dump ours/inj-served-dump 2>/dev/null
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
