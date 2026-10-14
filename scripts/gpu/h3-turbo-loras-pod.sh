#!/usr/bin/env bash
# Runs ON the Turbo LoRA validation pod (scripts/gpu/h3-turbo-loras-validate.sh; docs/ports/h3-turbo-loras.md). One
# RTX PRO 6000 on the runtime image; weights under $R (the EU volume's h3-base, FastH3 LoRA and LPIPS trees linked in,
# or all of them fetched from R2), the Turbo LoRA trees from R2:
#
#   1. fetch from R2 (plan.json) while uv installs the Space's Python stack (torch 2.11 cu130, transformers 5.8,
#      diffusers #14371 @ 665f5782) and the Space's own h3_lora.py / h3_split_blocks.py (Space revision 9a76d679);
#   2. parity (480p, 124 frames, the Space's default 6 grid points = larry-turbo-5step, bf16 linears):
#      ours once (text conditioning + merged weights dumped), ours with lightx2v (merged weights), the Space's stack
#      on our text (h3_turbo_lora_ref.py: its fold, its noise, every step dumped, the clip), ours again on its noise
#      and text, bf16 and served MXFP8; merged_parity.py, compare-dumps, av_psnr.py;
#   3. speed and quality: larry 6 and 4 forwards, lightx2v 4, h3-turbo, h3-max, each one warm process over two
#      prompts at 480p and 768p (5 s); the base model's 49 forwards at 480p for one prompt; compare-clips (LPIPS,
#      PSNR, sharpness) at 480p against the base and at 768p against h3-max.
#
# /srv is served on :8000 (Runpod proxy): log.txt, status.tsv (step, rc, seconds), *.log, the reports, the clips and
# DONE (0 = every step ok). Nothing is written to /workspace. Env: FV_WEIGHTS_MODE (volume | r2), FV_SKIP (step
# names), FV_SEED, FV_PARITY_PROMPT.
set -u
SRV=/srv
L=/fvw
R=$L/root
mkdir -p "$SRV" "$L" "$R"
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
BIN=/opt/fv/fv-gpucheck
SEED="${FV_SEED:-1024}"
P="${FV_PARITY_PROMPT:-A golden retriever runs along a sunny beach at low tide, waves rolling in, gulls calling overhead, cinematic tracking shot.}"
SPACE_REV=9a76d679e4197339c32b115d3eebe435d43b373a
H3_REV=42ed227ee7df40d41602854ae760620d6eb651fe
DIFFUSERS_REV=665f578278365ea4a3318cb8c9b66ce6c01204b9
log "start: weights $FV_WEIGHTS_MODE, seed $SEED"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights and the reference stack, concurrently ------------------------------------------------------------
if [[ "$FV_WEIGHTS_MODE" == volume ]]; then
  for t in h3-base FastH3-4-step-Preview-v1-LoRA auxiliary; do ln -sfn "/workspace/weights/$t" "$R/$t"; done
fi
fetch() { "$PY" /opt/fv/r2-fetch.py --root "$R" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >>$SRV/fetch.log 2>&1; }
ref_setup() {
  (
    set -e
    uv venv -q --python "$PY" /opt/refpy
    uv pip install -q --python /opt/refpy/bin/python --extra-index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0 transformers==5.8.0 accelerate==1.14.0 huggingface-hub==1.24.0 "safetensors>=0.8.0" av pillow numpy \
      "diffusers @ https://github.com/huggingface/diffusers/archive/$DIFFUSERS_REV.tar.gz"
    mkdir -p /opt/space
    for f in h3_lora.py h3_split_blocks.py; do
      curl -fsSL --retry 5 -o /opt/space/$f "https://huggingface.co/spaces/MiniMaxAI/MiniMax-H3-Turbo-Lora/resolve/$SPACE_REV/$f"
    done
    sha256sum /opt/space/*.py
    /opt/refpy/bin/python -c "import torch, diffusers, transformers; print(torch.__version__, torch.version.cuda, diffusers.__version__, transformers.__version__)"
  ) >>$SRV/ref-setup.log 2>&1
}
step fetch fetch & fpid=$!
step ref-setup ref_setup & rpid=$!
wait "$fpid"
for f in h3-turbo-loras/larry/minimax_h3_turbo_4step_ema_ckpt850.safetensors h3-turbo-loras/lightx2v/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors h3-base/transformer/config.json; do
  [[ -f "$R/$f" ]] || { log "missing $R/$f"; echo 1 >$SRV/DONE; exit 1; }
done
LARRY=$R/h3-turbo-loras/larry/minimax_h3_turbo_4step_ema_ckpt850.safetensors
LIGHTX=$R/h3-turbo-loras/lightx2v/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
export RUST_LOG=info FASTVIDEO_TAE_DIR=$R/auxiliary/tae
TC=$L/text-cache
GEN=("$BIN" --mode fast)
H3=(h3 gen --weights "$R/h3-base" --text-weights "$R/h3-base" --text-encoder streamed --text-cache "$TC")

# ---- 2. parity against the Space's stack --------------------------------------------------------------------------
PARAMS="transformer_blocks.0.attn.to_q,transformer_blocks.0.attn.to_k,transformer_blocks.0.attn.to_v,transformer_blocks.0.attn.to_out.0"
PARAMS+=",transformer_blocks.0.ff.net.0.proj,transformer_blocks.0.ff.net.2,transformer_blocks.25.attn.to_q,transformer_blocks.25.attn.to_v"
PARAMS+=",transformer_blocks.25.adaln_proj.linear,transformer_blocks.49.attn.to_k,transformer_blocks.49.attn.to_out.0"
PARAMS+=",transformer_blocks.49.ff.net.0.proj,transformer_blocks.49.ff.net.2,token_refiner.refiner_blocks.0.attn.to_q"
PARAMS+=",token_refiner.refiner_blocks.1.ff.net.0.proj,token_refiner.refiner_blocks.1.attn.to_out.0,norm_out.linear"
PAR=(--prompt "$P" --seed "$SEED" --height 480 --width 832 --num-frames 124)
ours() { # <name> <recipe> <profile> <quant> [env...]: one parity run of ours into $SRV/ours/<name>
  local name="$1" recipe="$2" prof="$3" q="$4"; shift 4
  env FASTVIDEO_H3_QUANT="$q" "$@" "${GEN[@]}" --techniques "$prof" --out $SRV/ours/$name-report --tag "$name" \
    "${H3[@]}" "${PAR[@]}" --h3-recipe "$recipe" --clip-dir $SRV/ours/$name >$SRV/ours-$name.log 2>&1
}
mkdir -p $SRV/ours $SRV/ref
step ours-pre ours pre larry-turbo-5step h3/larry_turbo_dense off FASTVIDEO_DUMP_DIR=$SRV/ours/pre-dump FV_H3_DUMP_MERGED="$PARAMS"
step ours-lightx-merged ours lightx lightx2v-turbo h3/lightx2v_turbo_dense off FASTVIDEO_DUMP_DIR=$SRV/ours/lightx-dump FV_H3_DUMP_MERGED="$PARAMS"
wait "$rpid"
O=$L/h3-diffusers
mkdir -p $O/scheduler $O/audio_scheduler
for x in transformer vae audio_vae; do ln -sfn "$R/h3-base/$x" "$O/$x"; done
for f in modular_model_index.json scheduler/scheduler_config.json audio_scheduler/scheduler_config.json; do
  curl -fsSL --retry 5 -o "$O/$f" "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/$H3_REV/$f" || log "$f: download failed"
done
# Every component of the index points at the Hub repo: make it this local overlay.
sed -i "s#\"MiniMaxAI/MiniMax-H3\"#\"$O\"#g" $O/modular_model_index.json
reference() {
  PYTHONUNBUFFERED=1 /opt/refpy/bin/python /opt/fv/upstream/h3_turbo_lora_ref.py --base $O --space /opt/space \
    --larry "$LARRY" --lightx "$LIGHTX" --text-hidden $SRV/ours/pre-dump/text_hidden --out $SRV/ref --clip $SRV/ref/clip \
    --height 480 --width 832 --num-frames 124 --steps 6 --seed "$SEED" --params "$PARAMS" >$SRV/ref.log 2>&1
}
step reference reference
AVPY=/opt/refpy/bin/python
step merged-larry $AVPY /opt/fv/upstream/merged_parity.py $SRV/ref/larry $SRV/ours/pre-dump --json $SRV/merged-larry.json >$SRV/merged-larry.txt 2>&1
step merged-lightx $AVPY /opt/fv/upstream/merged_parity.py $SRV/ref/lightx $SRV/ours/lightx-dump --base $SRV/ref/larry --json $SRV/merged-lightx.json >$SRV/merged-lightx.txt 2>&1
for arm in inj:off inj-served:mxfp8; do
  name="${arm%%:*}"
  step "ours-$name" ours "$name" larry-turbo-5step h3/larry_turbo_dense "${arm#*:}" FASTVIDEO_INJECT_DIR=$SRV/ref/larry FASTVIDEO_DUMP_DIR=$SRV/ours/$name-dump
  step "compare-dumps-$name" "$BIN" compare-dumps --baseline $SRV/ref/larry --candidate $SRV/ours/$name-dump >$SRV/compare-dumps-$name.txt 2>&1
  step "av-psnr-$name" $AVPY /opt/fv/upstream/av_psnr.py $SRV/ref/clip $SRV/ours/$name --json $SRV/av-psnr-$name.json >$SRV/av-psnr-$name.log 2>&1
done

# ---- 3. speed and quality -----------------------------------------------------------------------------------------
"$PY" - /opt/fv/prompts-eval.json $L/prompts.json $L/prompts-base.json <<'PYEOF'
import json, sys
e = {p["name"]: p for p in json.load(open(sys.argv[1]))["prompts"]}
keep = ["h3-demo", "spark-mountain-lake"]
rows = []
for size, (h, w) in (("480", (480, 832)), ("768", (768, 1344))):
    for k in keep:
        rows.append({"name": f"{k}-{size}p", "prompt": e[k]["prompt"], "seed": e[k]["seed"], "height": h, "width": w})
json.dump({"prompts": rows}, open(sys.argv[2], "w"), indent=1)
json.dump({"prompts": [r for r in rows if r["name"] == "h3-demo-480p"]}, open(sys.argv[3], "w"), indent=1)
PYEOF
perf() { # <cell> <recipe> <profile> [gen flags...]: one warm process over the prompt set
  local cell="$1" recipe="$2" prof="$3"; shift 3
  "${GEN[@]}" ${prof:+--techniques "$prof"} --out $SRV/perf/$cell --tag "$cell" "${H3[@]}" --prompt unused --prompts $L/prompts.json \
    --height 480 --width 832 --num-frames 107 --h3-recipe "$recipe" --clip-dir $SRV/perf/$cell/frames "$@" >$SRV/perf-$cell.log 2>&1
}
mkdir -p $SRV/perf
step perf-larry6 perf larry6 larry-turbo h3/larry_turbo_dense --warm
step perf-larry4 perf larry4 larry-turbo-4step h3/larry_turbo_dense --warm
step perf-lightx4 perf lightx4 lightx2v-turbo h3/lightx2v_turbo_dense --warm
step perf-turbo perf turbo 4step-vsa h3/fasth3_4step_vsa --warm
step perf-max perf max sol-h3 h3/sol_h3_4step_engine_ladder --warm
base() {
  "${GEN[@]}" --out $SRV/perf/base --tag base "${H3[@]}" --prompt unused --prompts $L/prompts-base.json \
    --h3-recipe base --dense --clip-dir $SRV/perf/base/frames >$SRV/perf-base.log 2>&1
}
step perf-base base
cmp() { # <baseline cell> <candidate cell> <prompt>
  mkdir -p $SRV/compare
  "$BIN" --out $SRV/compare --tag "$1--$2-$3" compare-clips --baseline $SRV/perf/$1/frames/$3 --candidate $SRV/perf/$2/frames/$3 \
    --lpips $R/auxiliary/lpips >$SRV/compare/$1--$2-$3.out 2>&1
}
for c in larry6 larry4 lightx4 turbo max; do
  step "cmp-base-$c" cmp base "$c" h3-demo-480p
  for p in h3-demo-768p spark-mountain-lake-768p; do [[ "$c" == max ]] || step "cmp-max-$c-$p" cmp max "$c" "$p"; done
done
step cmp-larry6-larry4 cmp larry6 larry4 spark-mountain-lake-480p

bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
