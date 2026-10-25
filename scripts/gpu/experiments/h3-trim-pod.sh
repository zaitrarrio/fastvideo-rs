#!/usr/bin/env bash
# Runs ON the FastH3 Trim validation pod (scripts/gpu/experiments/h3-trim-validate.sh; docs/ports/h3-trim.md §6). One RTX PRO
# 6000 on the runtime image with the EU weight volume (h3-base: text encoder, tokenizer, audio VAE) at /workspace:
#
#   1. fetch fasth3-trim-8step and h3-8step's transformer/ + vae/ from R2 (plan.json, presigned) onto the container
#      disk, while uv installs the upstream FastVideo stack (scripts/gpu/upstream/setup.sh, FV_REV = $FV_UP_REV);
#   2. upstream: FastVideo's FastH3 8-step example on the Trim snapshot (overlay: Trim transformer/vae/schedule +
#      h3-base text encoder/tokenizer/processor/audio VAE), strict eager profile, Triton VSA, with the oracle dump
#      hooks (scripts/gpu/upstream/oracle_dump.py);
#   3. ours injected with the reference's noise and text (FASTVIDEO_INJECT_DIR), dumped, `compare-dumps`, and the
#      decoded clips compared (av_psnr.py: video + audio PSNR);
#   4. timing: ours Trim and ours FastH3 V2 (h3-8step, the same VSA recipe), warm, same prompt / seed / geometry,
#      with nvidia-smi sampled every second for the VRAM peak.
#
# Nothing is written to /workspace (the shared volume is only read). /srv is served on :8000 (Runpod proxy):
# log.txt, status.tsv (step, rc, seconds), *.log, the dumps' comparison, the clips and DONE (0 = every step ok).
# Env: FV_UP_REV, FV_PROMPT, FV_SEED, FV_H/FV_W/FV_FRAMES, FV_DUMP_OPS, FV_SKIP (space-separated step names),
# FV_TEXT_ENCODER (h3 gen --text-encoder; default streamed: the bf16 encoder, as the reference).
set -u
SRV=/srv
L=/fvw
# volume: h3-base on the EU volume; r2: h3-base's text encoder, tokenizer and audio VAE fetched beside the rest.
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
O=$L/trim-overlay
UP=/opt/up
export UP
mkdir -p "$SRV" "$L"
cd "$SRV" || exit 1
# The in-pod self-delete (scripts/gpu/lib/self-delete.sh, the same text; checked by
# scripts/gpu/tests/self-delete.test.sh).
fv_self_delete() {
  # GraphQL podTerminate: the pod-scoped key may not DELETE through REST (HTTP 403,
  # docs/adr/0023-lab-report-and-ci-bridge.md §4). Success: HTTP 200 and no "errors".
  # The key reaches curl on stdin (-H @-), never in its argv; the answer is logged, then removed.
  _fvsd_code=$(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" | curl -sS --max-time 30 -o "$1.resp" \
    -w '%{http_code}' -X POST https://api.runpod.io/graphql -H 'Content-Type: application/json' -H @- \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" 2>>"$1")
  { cat "$1.resp"; echo " http=$_fvsd_code"; } >>"$1" 2>/dev/null
  _fvsd_ok=1
  [ "$_fvsd_code" = 200 ] && ! grep -q '"errors"' "$1.resp" 2>/dev/null && _fvsd_ok=0
  rm -f "$1.resp"
  return "$_fvsd_ok"
}
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
H="${FV_H:-480}"; WD="${FV_W:-832}"; NF="${FV_FRAMES:-124}"
OPS="${FV_DUMP_OPS:-0,1,20,41}"
BIN=/opt/fv/fv-gpucheck
[[ -x "$BIN" ]] || BIN=/opt/fastvideo-rs/target/release/fv-gpucheck   # the runtime image's own
log "start: ${H}x${WD}x${NF} seed $SEED, upstream FastVideo @ ${FV_UP_REV:-?}"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
# A host whose GPU memory is already held (2026-10-13: 95.8 of 97.9 GiB used, no process of ours) cannot run
# anything: stop at once and delete this pod (Runpod's pod-scoped key), so the launcher can retry elsewhere.
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
    for _ in 1 2 3 4 5; do fv_self_delete $SRV/log.txt && break; sleep 30; done
  fi
  exit 1
fi
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights from R2 and the upstream stack, concurrently -------------------------------------------------
fetch() { # <plan> <tag>: the Trim (and h3-base) trees first; FastH3 V2's while the Trim runs use the GPU
  "$PY" /opt/fv/r2-fetch.py --root "$L" --plan "$1" --status $SRV/fetch-status-$2.json >>$SRV/fetch-$2.log 2>&1
}
upstream_setup() {
  ( export FV_REV="$FV_UP_REV" UP_LOCAL=1; . /opt/fv/upstream/setup.sh && ensure_base && install_fastvideo ) >>$SRV/setup.log 2>&1
}
# FV_UP_DUMP_URL (+ _SHA256) and FV_UP_REF_URL: a previous pod's upstream dump (dumps.tar: up/dump) and reference
# clip, so this pod skips the upstream install and run; numpy for av_psnr.py then comes in a small venv.
reuse_upstream() {
  mkdir -p $SRV/up
  curl -fsSL --retry 5 -o /fvw/up-dump.tar "$FV_UP_DUMP_URL" && echo "$FV_UP_DUMP_SHA256  /fvw/up-dump.tar" | sha256sum -c - \
    && tar -xf /fvw/up-dump.tar -C $SRV up/dump && curl -fsSL --retry 5 -o $SRV/up/ref.mp4 "$FV_UP_REF_URL" \
    && uv venv -q --python "$PY" /opt/avpy && uv pip install -q --python /opt/avpy/bin/python numpy
} >>$SRV/setup.log 2>&1
step fetch fetch /opt/fv/plan.json trim & fpid=$!
if [[ -n "${FV_UP_DUMP_URL:-}" ]]; then
  step reuse-upstream reuse_upstream & upid=$!
  AVPY=/opt/avpy/bin/python
else
  step upstream-setup upstream_setup & upid=$!
  AVPY="$UP/fastvideo/bin/python"
fi
wait "$fpid"; wait "$upid"
[[ -d "$L/fasth3-trim-8step/transformer" ]] || { log "no Trim tree"; echo 1 >$SRV/DONE; exit 1; }
step fetch-v2 fetch /opt/fv/plan-v2.json v2 & v2pid=$!

mkdir -p "$O"
for x in transformer vae scheduler audio_scheduler fastvideo_inference.json model_index.json; do ln -sfn "$L/fasth3-trim-8step/$x" "$O/$x"; done
for x in text_encoder tokenizer audio_vae; do ln -sfn "$W/h3-base/$x" "$O/$x"; done
# FastVideo's H3 pipeline also loads processor/ (Qwen3VLProcessor: tokenizer files + preprocessor configs, 11 MB,
# byte-identical to h3-base's); neither weights tree carries it, so it comes from the Hub at the pinned revision.
mkdir -p "$O/processor"
for f in chat_template.json merges.txt preprocessor_config.json tokenizer.json tokenizer_config.json \
         video_preprocessor_config.json vocab.json; do
  curl -fsSL --retry 5 -o "$O/processor/$f" \
    "https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step/resolve/9edcb6511d5ff8dd1579e8eec8afee77e098b8d8/processor/$f" \
    || log "processor/$f: download failed"
done

# ---- 2. upstream FastVideo on the Trim snapshot, with the oracle dump hooks ------------------------------------
upstream() {
  mkdir -p $SRV/up
  env PYTHONPATH=/opt/fv/upstream/oracle_site FV_ORACLE_DUMP_DIR=$SRV/up/dump FASTVIDEO_DUMP_OPS="$OPS" \
    FASTVIDEO_STAGE_LOGGING=1 PYTHONUNBUFFERED=1 \
    "$UP/fastvideo/bin/python" /opt/fv/upstream/oracle_fastvideo.py --fastvideo-src "$UP/src/FastVideo" \
    --recipe 8step --out $SRV/up --repeats 1 -- \
    --prompt "$P" --seed "$SEED" --num-gpus 1 --vsa-kernel triton --no-fa4 --no-warmup --profile strict \
    --no-inference-torch-compile --no-compile-vae --model-path "$O" --height "$H" --width "$WD" --num-frames "$NF" \
    >>$SRV/upstream.log 2>&1
  local mp4
  mp4="$(find $SRV/up -maxdepth 2 -name 'run_01*.mp4' | head -1)"
  [[ -n "$mp4" ]] && cp "$mp4" $SRV/up/ref.mp4
  [[ -f $SRV/up/dump/oracle_meta.json ]]
}
[[ -n "${FV_UP_DUMP_URL:-}" ]] || step upstream upstream

# ---- 3. ours on the reference's noise and text -----------------------------------------------------------------
GEN=("$BIN" --mode fast h3 gen --prompt "$P" --seed "$SEED" --height "$H" --width "$WD" --num-frames "$NF"
     --text-encoder "${FV_TEXT_ENCODER:-streamed}" --text-weights "$W/h3-base" --text-cache $L/text-cache)
# Parity: bf16 linears (FASTVIDEO_H3_QUANT=off), as the reference's strict profile; then the served default (MXFP8
# blocks on sm_100+, docs/ports/h3-trim.md §6) on the same injected inputs.
ours_injected() { # <name> <quant>
  env FASTVIDEO_H3_QUANT="$2" FASTVIDEO_INJECT_DIR=$SRV/up/dump FASTVIDEO_DUMP_DIR=$SRV/ours/$1-dump FASTVIDEO_DUMP_OPS="$OPS" \
    "${GEN[@]}" --weights "$L/fasth3-trim-8step" --h3-recipe trim-8step --adaln-cache $L/trim-adaln.cache \
    --clip-dir $SRV/ours/$1 >>$SRV/ours-$1.log 2>&1
}
for arm in inj:off inj-served:mxfp8; do
  name="${arm%%:*}"
  step "ours-$name" ours_injected "$name" "${arm#*:}"
  step "compare-dumps-$name" "$BIN" compare-dumps --baseline $SRV/up/dump --candidate $SRV/ours/$name-dump >$SRV/compare-dumps-$name.txt 2>&1
  step "av-psnr-$name" "$AVPY" /opt/fv/upstream/av_psnr.py $SRV/up/ref.mp4 $SRV/ours/$name --json $SRV/av-psnr-$name.json >$SRV/av-psnr-$name.log 2>&1
done

# ---- 4. timing: Trim and FastH3 V2, warm, the same request -----------------------------------------------------
step time-trim "${GEN[@]}" --weights "$L/fasth3-trim-8step" --h3-recipe trim-8step --adaln-cache $L/trim-adaln.cache \
  --clip-dir $SRV/ours/trim --warm >$SRV/time-trim.log 2>&1
wait "$v2pid"
step time-v2 "${GEN[@]}" --weights "$L/h3-8step" --h3-recipe 8step --adaln-cache $L/v2-adaln.cache \
  --clip-dir $SRV/ours/v2 --warm >$SRV/time-v2.log 2>&1
# The two clips decoded by both: how far Trim moves from V2 on the same prompt and seed (not a gate).
step trim-vs-v2 "$AVPY" /opt/fv/upstream/av_psnr.py $SRV/ours/v2 $SRV/ours/trim --json $SRV/trim-vs-v2.json >$SRV/trim-vs-v2.log 2>&1

tar -cf $SRV/dumps.tar -C $SRV up/dump ours/inj-dump ours/inj-served-dump 2>/dev/null
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
