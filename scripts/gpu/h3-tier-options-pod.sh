#!/usr/bin/env bash
# Runs ON the H3 NVFP4 tier-options pod (scripts/gpu/h3-tier-options-validate.sh; docs/perf/h3-nvfp4-tier-options.md).
# One RTX PRO 6000 on a fastvideo-rs runtime image; this branch's fv-gpucheck, scripts/gpu and the R2 plan arrive
# in a bundle under /opt/fv. h3-base from the EU volume at /workspace (read only) or, in r2 mode, fetched.
#
# Steps (FV_STEPS, space-separated; default all, in this order):
#   fetch    the R2 trees of the plan (V2-NVFP4 transformer + its NVFP4 vae/, the Trim bf16 tree, the Trim NVFP4
#            file, TAEH3, LPIPS; h3-base in r2 mode)
#   prep     model folders and the prompt sets (the five prompts of prompts-hd5.json, their own seeds, 124 frames)
#   check    FASTVIDEO_H3_VAE_CHECK on the NVFP4 VAE (static and dynamic activation scales), with h3-base's bf16
#            decoder as the reference, one 480p generation each
#   cells    one process per arm and canvas (--warm, the prompt set), nvidia-smi sampled per cell
#   compare  compare-clips (PSNR, LPIPS, sharpness, jitter) per prompt and `gate` (gate-policy.toml, lossy) per pair
# Arms (FV_ARMS; default all): v2-mxvae (V2-NVFP4 + FP4 attention, the official VAE with its MXFP8 default),
# v2-bf16vae (FASTVIDEO_H3_VAE_MXFP8=0), v2-nvvae (FASTVIDEO_H3_VAE_NVFP4=1, the published NVFP4 vae/), v2-tae
# (TAEH3), trim-nv (Trim 8-step NVFP4, bf16 VSA), trim-nvfp4attn (+ FASTVIDEO_H3_VSA_FP4=1), trim-bf16
# (FASTVIDEO_H3_QUANT=off). Canvases (FV_RES): "480 768".
#
# /srv is served on :8000: log.txt, status.tsv (step, rc, seconds), results.tgz, DONE (0 = every step ok).
# Nothing is written to R2 or /workspace.
set -u
SRV=/srv
L=/fvw
B=/opt/fv
G=$B/gpu
BIN=$B/fv-gpucheck
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
STEPS=" ${FV_STEPS:-fetch prep check cells compare} "
ARMS="${FV_ARMS:-v2-mxvae v2-bf16vae v2-nvvae v2-tae trim-nv trim-nvfp4attn trim-bf16}"
RES="${FV_RES:-480 768}"
PROMPTS="talking-head spark-mountain-lake ltx-frogyoga ltx-newsbroadcast h3-demo"
mkdir -p "$SRV" "$L"
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
want() { [[ "$STEPS" == *" $1 "* ]]; }
step() {
  local name="$1" t0=$SECONDS rc
  shift
  log "begin $name"
  "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>$SRV/status.tsv
  log "end $name rc=$rc $((SECONDS - t0))s"
  return "$rc"
}
SAMPLER=""
start_sampler() { # <csv>: nvidia-smi every second; its PID in SAMPLER
  (while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits; done) >"$1" 2>/dev/null &
  SAMPLER=$!
}
stop_sampler() { # stop the sampler by its exact PID, so no file grows while tar reads it
  if [[ -n "$SAMPLER" ]]; then kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=""; fi
}
pack() {
  # Samplers stopped first (a growing file makes tar exit 1); then the archive is checked before it is published.
  stop_sampler
  (cd $SRV && tar --exclude='*.png' --exclude='*.raw' --exclude='*.f32' --exclude='*.bin' --exclude='*.safetensors' \
     --exclude='*-dump' --exclude='results.tgz*' --exclude='frames-mid.tgz*' --exclude='frames' -czf results.tgz.tmp .) \
    >/dev/null 2>&1
  # A few middle frames (480p, two prompts) for a visual check, in a second archive.
  (cd $SRV && find cells -path '*-480/frames/*' \( -path '*talking-head*' -o -path '*spark-mountain-lake*' \) -name 'frame-062.png' \
     2>/dev/null | tar -czf frames-mid.tgz.tmp -T - ) >/dev/null 2>&1
  if (cd $SRV && gzip -t results.tgz.tmp && tar -tzf results.tgz.tmp >/dev/null); then
    mv $SRV/results.tgz.tmp $SRV/results.tgz
    [[ -f $SRV/frames-mid.tgz.tmp ]] && mv $SRV/frames-mid.tgz.tmp $SRV/frames-mid.tgz
    log "packed results.tgz ($(du -h $SRV/results.tgz | cut -f1), $(tar -tzf $SRV/results.tgz | wc -l) entries)"
    return 0
  fi
  log "pack: results.tgz.tmp failed its check"
  return 1
}
finish() { pack || echo pack-failed >$SRV/PACK_FAILED; echo "$1" >$SRV/DONE; log "DONE $1"; }

log "start: mode ${FV_WEIGHTS_MODE:-volume}, steps${STEPS}, arms $ARMS, res $RES; $("$BIN" --version 2>/dev/null | head -1)"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi

# ---- fetch ------------------------------------------------------------------------------------------------------------
if want fetch; then
  step fetch "$PY" $G/r2-fetch.py --root "$L" --plan $B/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1 \
    || { finish 1; exit 1; }
fi

# ---- model folders and prompt sets ------------------------------------------------------------------------------------
V2REV=0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0
TRIM=$L/fasth3-trim-8step
TAE=$L/auxiliary/tae/taeh3.safetensors
LPIPS=$L/auxiliary/lpips
prep() {
  [[ -d "$W/h3-base/tokenizer" && -d "$W/h3-base/text_encoder" && -d "$W/h3-base/vae" ]] || { log "h3-base missing under $W"; return 1; }
  [[ -f "$TRIM/transformer/config.json" && -d "$L/fasth3-trim-8step-nvfp4/diffusion_models" && -f "$TAE" ]] || { log "R2 trees missing"; return 1; }
  local f x
  mkdir -p $L/small-v2
  for f in fastvideo_inference.json modular_model_index.json checkpoint_content.json checkpoint_metadata.json \
           scheduler/scheduler_config.json audio_scheduler/scheduler_config.json; do
    mkdir -p "$L/small-v2/$(dirname "$f")"
    curl -fsSL --retry 5 -o "$L/small-v2/$f" "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4/resolve/$V2REV/$f" \
      || { log "V2 $f: download failed"; return 1; }
  done
  # The published NVFP4 vae/: shards 1-2 and the configs from the V2-NVFP4 tree, shard 3 (byte-identical) from h3-base.
  mkdir -p $L/vae-nvfp4
  for f in config.json diffusion_pytorch_model.safetensors.index.json diffusion_pytorch_model-00001-of-00003.safetensors \
           diffusion_pytorch_model-00002-of-00003.safetensors; do
    ln -sfn "$L/fasth3-v2-nvfp4/vae/$f" "$L/vae-nvfp4/$f"
  done
  ln -sfn "$W/h3-base/vae/diffusion_pytorch_model-00003-of-00003.safetensors" "$L/vae-nvfp4/diffusion_pytorch_model-00003-of-00003.safetensors"
  for d in ours-v2 ours-v2nv; do
    mkdir -p $L/$d
    cp -r $L/small-v2/. $L/$d/
    ln -sfn $L/fasth3-v2-nvfp4/transformer $L/$d/transformer
    for x in tokenizer text_encoder audio_vae; do ln -sfn "$W/h3-base/$x" "$L/$d/$x"; done
  done
  ln -sfn "$W/h3-base/vae" $L/ours-v2/vae
  ln -sfn $L/vae-nvfp4 $L/ours-v2nv/vae
  "$PY" - "$G/prompts-hd5.json" "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
for res, (h, w) in {"480": (480, 832), "768": (768, 1344)}.items():
    ps = [dict(p, height=h, width=w) for p in d["prompts"]]
    json.dump({"name": f"h3-tier-options-{res}p", "source": f"scripts/gpu/prompts-hd5.json (all five, their own seeds) at {w}x{h}",
               "prompts": ps}, open(f"{out}/prompts-{res}.json", "w"), indent=1)
open(f"{out}/prompt-warm.txt", "w").write(next(p["prompt"] for p in d["prompts"] if p["name"] == "talking-head"))
PYEOF
}
if want prep; then step prep prep >$SRV/prep.log 2>&1 || { finish 1; exit 1; }; fi
P="$(cat $L/prompt-warm.txt 2>/dev/null)"
COMMON=(--text-encoder streamed --text-weights "$W/h3-base" --text-cache "$L/text-cache")

arm_args() { # <arm>: weights / recipe / decoder flags
  case "$1" in
    v2-*nvvae) echo "--weights $L/ours-v2nv --adaln-cache $L/adaln-v2.cache" ;;
    v2-tae) echo "--weights $L/ours-v2 --adaln-cache $L/adaln-v2.cache --taeh3-weights $TAE" ;;
    v2-*) echo "--weights $L/ours-v2 --adaln-cache $L/adaln-v2.cache" ;;
    trim-bf16) echo "--weights $TRIM --h3-recipe trim-8step --adaln-cache $L/adaln-trim.cache" ;;
    trim-nv*) echo "--weights $TRIM --h3-recipe trim-8step-nvfp4 --adaln-cache $L/adaln-trim.cache" ;;
  esac
}
arm_env() { # <arm>
  case "$1" in
    v2-bf16vae) echo "FASTVIDEO_H3_VAE_MXFP8=0" ;;
    v2-nvvae) echo "FASTVIDEO_H3_VAE_NVFP4=1" ;;
    trim-nvfp4attn) echo "FASTVIDEO_H3_VSA_FP4=1" ;;
    trim-bf16) echo "FASTVIDEO_H3_QUANT=off" ;;
    *) echo "FV_ARM=$1" ;;
  esac
}

# ---- check: the NVFP4 VAE against the bf16 decoder on the same tiles ---------------------------------------------------
check() { # <act: static|dynamic>
  local d=$SRV/check/$1
  mkdir -p "$d"
  # shellcheck disable=SC2046
  (cd "$d" && env FASTVIDEO_H3_VAE_NVFP4=1 FASTVIDEO_H3_VAE_CHECK=1 FASTVIDEO_H3_VAE_NVFP4_ACT="$1" \
     FASTVIDEO_H3_VAE_NVFP4_REF="$W/h3-base/vae" timeout 900 "$BIN" --mode fast h3 gen $(arm_args v2-nvvae) \
     --prompt "$P" --seed 1024 --height 480 --width 832 --num-frames 124 "${COMMON[@]}" --no-mp4 --clip-dir "$d/clip") \
    >"$d/stdout.log" 2>"$d/stderr.log"
  local rc=$?
  grep -h "h3 vae" "$d/stderr.log" >"$d/vae-lines.txt" 2>/dev/null
  rm -rf "$d/clip"
  return $rc
}
if want check; then
  for act in static dynamic; do step "check-$act" check "$act"; done
fi

# ---- cells ------------------------------------------------------------------------------------------------------------
cell() { # <arm> <res>
  local arm="$1" res="$2" dir="$SRV/cells/$1-$2" rc h=768 w=1344
  [[ "$res" == 480 ]] && { h=480; w=832; }
  mkdir -p "$dir"
  start_sampler "$dir/nvsmi.csv"
  # shellcheck disable=SC2046
  (cd "$dir" && env $(arm_env "$arm") timeout 2400 "$BIN" --mode fast h3 gen $(arm_args "$arm") \
     --prompts "$L/prompts-$res.json" --prompt "$P" --seed 1024 --height "$h" --width "$w" --num-frames 124 --warm \
     "${COMMON[@]}" --no-mp4 --clip-dir "$dir/frames") >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  stop_sampler
  grep -hE "h3 vae|h3: VSA|h3 nvfp4|taeh3" "$dir/stderr.log" | sort -u >"$dir/config-lines.txt" 2>/dev/null
  return $rc
}
if want cells; then
  for res in $RES; do
    for arm in $ARMS; do step "cell-$arm-$res" cell "$arm" "$res"; done
  done
fi

# ---- compare-clips and the gate ---------------------------------------------------------------------------------------
PAIRS="v2-mxvae:v2-tae v2-bf16vae:v2-tae v2-bf16vae:v2-mxvae v2-bf16vae:v2-nvvae v2-mxvae:v2-nvvae \
trim-bf16:trim-nv v2-mxvae:trim-nv trim-bf16:trim-nvfp4attn v2-mxvae:trim-nvfp4attn trim-nv:trim-nvfp4attn"
if want compare; then
  C=$SRV/compare
  mkdir -p $C/clips $C/gate
  for res in $RES; do
    for pair in $PAIRS; do
      b="${pair%%:*}-$res" c="${pair#*:}-$res"
      [[ -f "$SRV/cells/$b/benchmark.json" && -f "$SRV/cells/$c/benchmark.json" ]] || continue
      cmp=()
      for p in $PROMPTS; do
        [[ -d "$SRV/cells/$b/frames/$p" && -d "$SRV/cells/$c/frames/$p" ]] || continue
        step "compare-$b--$c-$p" "$BIN" --out $C/clips --tag "$b--$c-$p" compare-clips \
          --baseline "$SRV/cells/$b/frames/$p" --candidate "$SRV/cells/$c/frames/$p" --lpips "$LPIPS" \
          >"$C/clips/$b--$c-$p.stdout.log" 2>"$C/clips/$b--$c-$p.stderr.log"
        for f in "$C/clips/compare-clips-$b--$c-$p"*.json; do [[ -f "$f" ]] && cmp+=(--compare "$f"); done
      done
      # The gate's verdict is its report, not its exit status (a lossy FAIL exits non-zero): rc is recorded.
      step "gate-$b--$c" "$BIN" --out $C/gate --tag "$b--$c" gate --baseline "$SRV/cells/$b" --candidate "$SRV/cells/$c" \
        --policy $G/gate-policy.toml --kind lossy "${cmp[@]}" >"$C/gate/$b--$c.stdout.log" 2>"$C/gate/$b--$c.stderr.log"
    done
  done
fi
rc=0
awk -F'\t' '$2 != 0 && $1 !~ /^gate-/ {bad=1} END {exit bad}' $SRV/status.tsv || rc=1
finish "$rc"
