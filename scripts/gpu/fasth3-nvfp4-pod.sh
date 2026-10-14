#!/usr/bin/env bash
# Runs ON the FastH3 NVFP4 apples-to-apples pod (scripts/gpu/fasth3-nvfp4-validate.sh; docs/ports/fasth3-nvfp4.md).
# One RTX PRO 6000 on the upstream image `fastvideo-rs-upstream-fastvideo-nvfp4` (FastVideo @ d5253fd with its sm_120a
# FP4 attention kernels, CUDA 13.0 devel for FlashInfer's JIT, and the fastvideo-rs runtime libraries under
# /opt/fvrs-cuda); this branch's fv-gpucheck and scripts/gpu come in the staged bundle at /opt/fv.
#
#   1. fetch from R2 (plan.json, presigned, read only): the two ModelOpt NVFP4 trees, h3-8step's text encoder,
#      tokenizer, audio VAE and official VAE (and, for the dense control, its bf16 transformer), LPIPS; the small
#      configs (schedulers, fastvideo_inference.json, processor/) from the Hub at the pinned revisions;
#   2. upstream, as FastVideo's RTX PRO 6000 page runs it: the NVFP4 text-encoder conversion, the DiT repack
#      (--quantize-attention --quantize-gate, layer_profile h3_dit_vsa), then per checkpoint one process: 480p cold
#      (dumped for parity) and warm, the second prompt, 768p cold and warm, the second prompt, and (V2) a refused
#      8-grid-point request;
#   3. ours on the reference's noise and text (FASTVIDEO_INJECT_DIR), per-step latents dumped, `compare-dumps`,
#      clip PSNR / LPIPS against the reference clip;
#   4. ours timed, one process per arm, both canvases and both prompts, --warm: NVFP4 attention + FFN (the
#      h3_dit_vsa equivalent), NVFP4 FFN + MXFP8 attention, and (V2) the served dense MXFP8 recipe;
#   5. one profiled 480p generation per checkpoint (FASTVIDEO_PROFILE phases, FASTVIDEO_GPU_TRACE kernels).
#
# /srv is served on :8000 (Runpod proxy): log.txt, status.tsv (step, rc, seconds), results.tgz, DONE (0 = all ok).
# Nothing is written to R2 or any volume. Env: FV_SKIP (space-separated step names), FV_ONLY_V2=1 (skip the 4-step).
set -u
SRV=/srv
L=/fvw
FV=/opt/fv
G=$FV/gpu
UPV=/opt/upstream/fastvideo-nvfp4
UPSRC=/opt/upstream/src/FastVideo-nvfp4
BIN=$FV/fv-gpucheck
[[ -x "$BIN" ]] || BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
FVLIB=/opt/fvrs-cuda/lib64:/opt/fvrs-cuda/cudnn
PY=/opt/upstream/tools/bin/python
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
ours() { env LD_LIBRARY_PATH="$FVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$BIN" "$@"; }
log "start: $(ours --version 2>/dev/null | head -1); image FastVideo $(git -C $UPSRC rev-parse --short HEAD 2>/dev/null)"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
# A host whose GPU memory is already held (2026-10-13: 95.8 of 97.9 GiB used, no process of ours) cannot run
# anything: stop at once and delete this pod (Runpod's pod-scoped key), so the launcher can retry elsewhere.
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do printf '%s,' "$(date +%s.%N)"; nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights ---------------------------------------------------------------------------------------------
step fetch "$PY" $G/r2-fetch.py --root "$L" --plan $FV/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1
V2REV=0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0
S4REV=d7adcabeb19a81bb79cf9c4a04f5e366f1f2e90a
small() { # <repo> <rev> <dir>: the configs neither R2 tree carries (byte-identical to h3-base / h3-8step's on the Hub)
  local f
  for f in fastvideo_inference.json modular_model_index.json checkpoint_content.json checkpoint_metadata.json \
           scheduler/scheduler_config.json audio_scheduler/scheduler_config.json processor/chat_template.json \
           processor/merges.txt processor/preprocessor_config.json processor/tokenizer.json processor/tokenizer_config.json \
           processor/video_preprocessor_config.json processor/vocab.json; do
    mkdir -p "$3/$(dirname "$f")"
    curl -fsSL --retry 5 -o "$3/$f" "https://huggingface.co/$1/resolve/$2/$f" || { log "$1 $f: download failed"; return 1; }
  done
}
# A model folder: <dir> <small-config dir> <transformer> <text_encoder>; tokenizer, audio VAE and the official VAE
# are h3-8step's (= h3-base's).
folder() {
  local d="$1" s="$2"
  mkdir -p "$d"
  cp -r "$s"/. "$d"/
  ln -sfn "$3" "$d/transformer"
  ln -sfn "$4" "$d/text_encoder"
  for x in tokenizer audio_vae vae; do ln -sfn "$L/h3-8step/$x" "$d/$x"; done
}
prep() {
  small FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4 $V2REV $L/small-v2 \
    && small FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4 $S4REV $L/small-4s || return 1
  # The bf16 encoder's tokenizer files (the R2 tree carries its weights and JSON configs).
  for f in merges.txt vocab.json; do
    curl -fsSL --retry 5 -o "$L/h3-8step/text_encoder/$f" \
      "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/resolve/3da2ddfe1954d9cda4c05b643dc0f26007a655c5/text_encoder/$f" || return 1
  done
  # Ours: the ModelOpt transformer as published, the bf16 encoder (streamed once per prompt, then cached).
  folder $L/ours-v2 $L/small-v2 $L/fasth3-v2-nvfp4/transformer $L/h3-8step/text_encoder
  folder $L/ours-4s $L/small-4s $L/fasth3-4step-vsa-nvfp4/transformer $L/h3-8step/text_encoder
  # The served dense V2 (control): h3-8step as fetched, with its contract.
  [[ -d $L/h3-8step/transformer ]] && cp -n $L/small-v2/fastvideo_inference.json $L/h3-8step/ 2>/dev/null
  # The NVFP4 trees' own vae/ (ModelOpt NVFP4 decoder linears) is not read by either side: FastVideo d5253fd's H3
  # VAE has no ModelOpt path, ours neither; both decode with the official VAE.
  return 0
}
step prep prep >$SRV/prep.log 2>&1 || { log "prep failed"; echo 1 >$SRV/DONE; exit 1; }
step lpips bash $G/fetch-lpips.sh $L/lpips >$SRV/lpips.log 2>&1
step kernels env FV_NVFP4_BENCH=0 LD_LIBRARY_PATH="$FVLIB" "$BIN" --out $SRV/kernels kernels --groups nvfp4_linear >$SRV/kernels.log 2>&1

# ---- prompts: h3-demo and spark-mountain-lake (scripts/gpu/prompts-hd5.json) at seed 1024 ---------------------
"$PY" - "$G/prompts-hd5.json" "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
ps = {p["name"]: p["prompt"] for p in json.load(open(src))["prompts"]}
keep = ["h3-demo", "spark-mountain-lake"]
canv = {"480p": (480, 832), "768p": (768, 1344)}
# Ours: both canvases in one process per arm (the --warm generation is 480p h3-demo).
json.dump({"name": "fasth3-nvfp4", "source": "scripts/gpu/prompts-hd5.json (h3-demo, spark-mountain-lake) at seed 1024",
           "prompts": [{"name": f"{n}-{c}", "prompt": ps[n], "seed": 1024, "height": h, "width": w}
                       for c, (h, w) in canv.items() for n in keep]}, open(f"{out}/prompts-ours.json", "w"), indent=1)
for ck, steps in (("v2", 9), ("4s", 5)):
    runs = []
    for c, (h, w) in canv.items():
        runs.append({"tag": f"{ck}-{c}-h3-demo-cold", "prompt": ps["h3-demo"], "seed": 1024, "height": h, "width": w, "num_frames": 124, "steps": steps})
        runs.append({"tag": f"{ck}-{c}-h3-demo-warm", "prompt": ps["h3-demo"], "seed": 1024, "height": h, "width": w, "num_frames": 124, "steps": steps})
        runs.append({"tag": f"{ck}-{c}-spark-mountain-lake", "prompt": ps["spark-mountain-lake"], "seed": 1024, "height": h, "width": w, "num_frames": 124, "steps": steps})
    if ck == "v2":  # the owner's "also report 8": FastVideo checks grid points = rungs + 1 and refuses it
        runs.append({"tag": "v2-480p-h3-demo-8points", "prompt": ps["h3-demo"], "seed": 1024, "height": 480, "width": 832, "num_frames": 124, "steps": 8})
    json.dump(runs, open(f"{out}/runs-up-{ck}.json", "w"), indent=1)
open(f"{out}/prompt-h3-demo.txt", "w").write(ps["h3-demo"])
PYEOF
P="$(cat $L/prompt-h3-demo.txt)"

# ---- 2. upstream ----------------------------------------------------------------------------------------------
up() { env PYTHONUNBUFFERED=1 TORCHINDUCTOR_CACHE_DIR=$L/cache/inductor TRITON_CACHE_DIR=$L/cache/triton \
         FLASHINFER_WORKSPACE_BASE=$L/cache/flashinfer FLASHINFER_CUDA_ARCH_LIST=12.0a "$@"; }
convert_te() {
  (cd $UPSRC && up $UPV/bin/python scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py \
     --src $L/h3-8step/text_encoder --dst $L/conv/te-nvfp4)
}
convert_dit() { # <ck> <src transformer>
  (cd $UPSRC && up $UPV/bin/python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py \
     --src "$2" --dst $L/conv/$1-transformer-vsa --quantize-attention --quantize-gate)
}
upstream() { # <ck> <sparsity>
  local ck="$1"
  folder $L/up-$ck $L/small-$ck $L/conv/$ck-transformer-vsa $L/conv/te-nvfp4
  up env PYTHONPATH=$G/upstream/oracle_site FV_ORACLE_DUMP_DIR=$SRV/up-$ck/dump FASTVIDEO_DUMP_OPS=none \
    $UPV/bin/python $G/upstream/bench_fasth3_nvfp4.py --model $L/up-$ck --sparsity "$2" \
    --runs $L/runs-up-$ck.json --out $SRV/up-$ck
}
step convert-te convert_te >$SRV/convert-te.log 2>&1
step convert-v2 convert_dit v2 $L/fasth3-v2-nvfp4/transformer >$SRV/convert-v2.log 2>&1
step up-v2 upstream v2 0.8 >$SRV/up-v2.log 2>&1
CKS="v2"
[[ "${FV_ONLY_V2:-0}" == 1 ]] || CKS="v2 4s"

# ---- 3./4./5. ours, per checkpoint ----------------------------------------------------------------------------
COMMON=(--text-encoder streamed --text-weights $L/h3-8step --text-cache $L/text-cache)
parity() { # <ck>: ours (NVFP4 attention + FFN) on the reference's noise and text
  local ck="$1" ref
  ref="$(ls $SRV/up-$ck/00_*.mp4 2>/dev/null | head -1)"
  env FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_INJECT_DIR=$SRV/up-$ck/dump FASTVIDEO_DUMP_DIR=$SRV/ours-$ck/inj-dump \
    LD_LIBRARY_PATH="$FVLIB" "$BIN" --mode fast h3 gen --weights $L/ours-$ck --prompt "$P" --seed 1024 \
    --height 480 --width 832 --num-frames 124 "${COMMON[@]}" --adaln-cache $L/adaln-$ck.cache \
    --clip-dir $SRV/ours-$ck/inj >$SRV/ours-$ck-inj.log 2>&1 || return 1
  ours compare-dumps --baseline $SRV/up-$ck/dump --candidate $SRV/ours-$ck/inj-dump >$SRV/compare-dumps-$ck.txt 2>&1
  [[ -n "$ref" ]] && "$PY" $G/upstream/av_psnr.py "$ref" $SRV/ours-$ck/inj --json $SRV/av-psnr-$ck.json >$SRV/av-psnr-$ck.log 2>&1
  # LPIPS against the reference clip's frames.
  if [[ -n "$ref" ]]; then
    mkdir -p $L/refframes-$ck && ffmpeg -nostdin -loglevel error -y -i "$ref" $L/refframes-$ck/frame-%03d.png \
      && ours --out $SRV/ours-$ck/compare --tag "up--ours-$ck" compare-clips --baseline $L/refframes-$ck \
         --candidate $SRV/ours-$ck/inj --lpips $L/lpips --lpips-pairs all >$SRV/ours-$ck/compare.log 2>&1
  fi
  return 0
}
cell() { # <ck> <arm> <weights>: one process, both canvases and prompts, --warm
  local ck="$1" arm="$2" dir="$SRV/ours-$1/$2" rc
  mkdir -p "$dir"
  local vars=(FV_ARM="$arm")
  case "$arm" in
    nvfp4) vars+=(FASTVIDEO_H3_NVFP4_ATTN=1) ;;
    nvfp4ffn-mxfp8) vars+=(FASTVIDEO_H3_NVFP4_ATTN=0 FASTVIDEO_H3_QUANT=mxfp8) ;;
    dense-mxfp8) vars+=(FASTVIDEO_H3_QUANT=mxfp8) ;;
  esac
  local dflag=()
  [[ "$arm" == dense-mxfp8 ]] && dflag=(--dense)
  (cd "$dir" && env "${vars[@]}" LD_LIBRARY_PATH="$FVLIB" timeout 2400 "$BIN" --mode fast h3 gen --weights "$3" \
     --prompts $L/prompts-ours.json --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 --warm \
     "${dflag[@]}" "${COMMON[@]}" --adaln-cache $L/adaln-$ck-$arm.cache --clip-dir "$dir/frames") \
    >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  return $rc
}
profile() { # <ck>: phases (synchronizing) and a CUPTI kernel trace of one warm 480p generation
  local ck="$1" dir="$SRV/ours-$1/profile"
  mkdir -p "$dir"
  (cd "$dir" && env FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_PROFILE=1 LD_LIBRARY_PATH="$FVLIB" timeout 900 "$BIN" --mode fast \
     h3 gen --weights $L/ours-$ck --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 --warm \
     "${COMMON[@]}" --adaln-cache $L/adaln-$ck-nvfp4.cache --clip-dir "$dir/phases" --no-mp4) >"$dir/phases.log" 2>&1
  (cd "$dir" && env FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_GPU_TRACE=1 LD_LIBRARY_PATH="$FVLIB" timeout 900 "$BIN" --mode fast \
     h3 gen --weights $L/ours-$ck --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 \
     "${COMMON[@]}" --adaln-cache $L/adaln-$ck-nvfp4.cache --clip-dir "$dir/trace" --no-mp4) >"$dir/trace.log" 2>&1
}
for ck in $CKS; do
  if [[ "$ck" == 4s ]]; then
    step convert-4s convert_dit 4s $L/fasth3-4step-vsa-nvfp4/transformer >$SRV/convert-4s.log 2>&1
    step up-4s upstream 4s 0.9 >$SRV/up-4s.log 2>&1
  fi
  step "parity-$ck" parity "$ck"
  step "cell-$ck-nvfp4" cell "$ck" nvfp4 $L/ours-$ck
  step "cell-$ck-nvfp4ffn-mxfp8" cell "$ck" nvfp4ffn-mxfp8 $L/ours-$ck
  [[ "$ck" == v2 && -d $L/h3-8step/transformer ]] && step "cell-v2-dense-mxfp8" cell v2 dense-mxfp8 $L/h3-8step
  [[ "$ck" == v2 ]] && step "profile-$ck" profile "$ck"
done

# ---- results (no frames, no dumps: logs, reports, benchmark.json, result.json, the clips' mp4) --------------
tar -czf $SRV/results.tgz -C $SRV --exclude='*.png' --exclude='*-dump' --exclude='dump' --exclude='*.f32' \
  log.txt status.tsv nvidia-smi.txt nvsmi.csv fetch.log prep.log kernels kernels.log convert-*.log up-v2.log up-4s.log \
  up-v2 up-4s ours-v2 ours-4s compare-dumps-*.txt av-psnr-*.json ours-*-inj.log 2>/dev/null
tar -cf $SRV/dumps.tar -C $SRV up-v2/dump ours-v2/inj-dump up-4s/dump ours-4s/inj-dump 2>/dev/null
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
