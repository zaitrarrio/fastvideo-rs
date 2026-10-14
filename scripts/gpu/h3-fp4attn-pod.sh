#!/usr/bin/env bash
# Runs ON the H3 sparse FP4 attention pod (scripts/gpu/h3-fp4attn-validate.sh; docs/ports/fasth3-sparse-fp4-attn.md).
# One RTX PRO 6000 on a fastvideo-rs runtime image (its fv-gpucheck and scripts/gpu); this script itself arrives in
# the pod's env (FV_POD_SCRIPT_B64), so it can run on an image that predates it. The R2 plan (presigned GETs from
# the read-only key) is /opt/fv/plan.json; h3-base from the EU volume at /workspace (read only) or, in r2 mode,
# fetched with the rest.
#
# Steps (FV_STEPS, space-separated; default all, in this order):
#   fetch     the NVFP4 trees' transformer/ (V2, and the 4-step with FV_WITH_4S=1), LPIPS; h3-base in r2 mode
#   prep      model folders: the NVFP4 transformer + h3-base's text encoder / tokenizer / VAEs + the Hub configs
#   kernels   `kernels --groups h3_fp4_attn` (unit checks on synthetic data; needs a binary that has the group)
#   trace768  CUPTI trace of one warm V2 step at 1344x768x124 (FASTVIDEO_GPU_TRACE) per arm, plus phases
#   trace480  the same at 832x480
#   parity    V2 480p h3-demo seed 1024, per-step latents dumped, per arm; compare-dumps / compare-clips fp4 vs bf16
#   cells     timed --warm runs, V2 (and 4-step) x {480p, 768p} x {h3-demo, spark-mountain-lake}, one process per arm
# Arms (FV_ARMS, default "bf16vsa fp4"): bf16vsa = #110's NVFP4 attention + FFN with bf16 VSA; fp4 = the same with
# FASTVIDEO_H3_VSA_FP4=1 (sparse FP4 VSA attention).
#
# /srv is served on :8000 (Runpod proxy): log.txt, status.tsv (step, rc, seconds), results.tgz, DONE (0 = all ok).
# Nothing is written to R2 or /workspace.
set -u
SRV=/srv
L=/fvw
G=/opt/fastvideo-rs/scripts/gpu
BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
STEPS=" ${FV_STEPS:-fetch prep kernels trace768 trace480 parity cells} "
ARMS="${FV_ARMS:-bf16vsa fp4}"
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
pack() {
  (cd $SRV && tar --exclude='*.png' --exclude='*.raw' --exclude='*.f32' --exclude='*.bin' --exclude='*.safetensors' \
     --exclude='*-dump' --exclude='results.tgz' -czf results.tgz.tmp . && mv results.tgz.tmp results.tgz)
}
finish() { pack; echo "$1" >$SRV/DONE; log "DONE $1"; }
log "start: mode ${FV_WEIGHTS_MODE:-volume}, steps${STEPS}, arms $ARMS; $("$BIN" --version 2>/dev/null | head -1)"
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
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- unit checks first: they need no weights, so they run while the weights arrive --------------------------------
fpid=""
if want fetch; then
  step fetch "$PY" $G/r2-fetch.py --root "$L" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1 &
  fpid=$!
fi
if want kernels; then
  step kernels "$BIN" --out $SRV/kernels kernels --groups h3_fp4_attn >$SRV/kernels.log 2>&1
  if [[ "${FV_KERNELS_GATE:-1}" == 1 ]] && ! grep -q '"status": *"pass"' $SRV/kernels/*.json 2>/dev/null; then
    log "kernel checks did not pass: stopping before any model run"
    [[ -n "$fpid" ]] && kill "$fpid" 2>/dev/null
    finish 1; exit 1
  fi
fi
want fetch && bash $G/fetch-lpips.sh $L/lpips >$SRV/lpips.log 2>&1
[[ -n "$fpid" ]] && wait "$fpid"

# ---- model folders ---------------------------------------------------------------------------------------------------
V2REV=0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0
S4REV=d7adcabeb19a81bb79cf9c4a04f5e366f1f2e90a
small() { # <repo> <rev> <dir>: the configs neither R2 tree carries (byte-identical to h3-base's on the Hub)
  local f
  for f in fastvideo_inference.json modular_model_index.json checkpoint_content.json checkpoint_metadata.json \
           scheduler/scheduler_config.json audio_scheduler/scheduler_config.json; do
    mkdir -p "$3/$(dirname "$f")"
    curl -fsSL --retry 5 -o "$3/$f" "https://huggingface.co/$1/resolve/$2/$f" || { log "$1 $f: download failed"; return 1; }
  done
}
folder() { # <dir> <small-config dir> <transformer>
  mkdir -p "$1"
  cp -r "$2"/. "$1"/
  ln -sfn "$3" "$1/transformer"
  local x
  for x in tokenizer text_encoder audio_vae vae; do ln -sfn "$W/h3-base/$x" "$1/$x"; done
}
prep() {
  [[ -d "$W/h3-base/tokenizer" && -d "$W/h3-base/text_encoder" ]] || { log "h3-base missing under $W"; return 1; }
  small FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4 $V2REV $L/small-v2 || return 1
  folder $L/ours-v2 $L/small-v2 $L/fasth3-v2-nvfp4/transformer
  if [[ -d $L/fasth3-4step-vsa-nvfp4/transformer ]]; then
    small FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4 $S4REV $L/small-4s || return 1
    folder $L/ours-4s $L/small-4s $L/fasth3-4step-vsa-nvfp4/transformer
  fi
  "$PY" - "$G/prompts-hd5.json" "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
ps = {p["name"]: p["prompt"] for p in json.load(open(src))["prompts"]}
keep = ["h3-demo", "spark-mountain-lake"]
canv = {"480p": (480, 832), "768p": (768, 1344)}
json.dump({"name": "h3-fp4attn", "source": "scripts/gpu/prompts-hd5.json (h3-demo, spark-mountain-lake) at seed 1024",
           "prompts": [{"name": f"{n}-{c}", "prompt": ps[n], "seed": 1024, "height": h, "width": w}
                       for c, (h, w) in canv.items() for n in keep]}, open(f"{out}/prompts-cells.json", "w"), indent=1)
open(f"{out}/prompt-h3-demo.txt", "w").write(ps["h3-demo"])
PYEOF
}
if want prep; then step prep prep >$SRV/prep.log 2>&1 || { finish 1; exit 1; }; fi
P="$(cat $L/prompt-h3-demo.txt 2>/dev/null)"
COMMON=(--text-encoder streamed --text-weights "$W/h3-base" --text-cache "$L/text-cache")
arm_env() { # <arm>
  case "$1" in
    bf16vsa) echo "FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=0" ;;
    fp4) echo "FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=1" ;;
  esac
}
gen() { # <ck> <arm> <dir> <height> <width> [extra env...] -- [extra args...]
  local ck="$1" arm="$2" dir="$3" h="$4" w="$5"
  shift 5
  local envs=() args=()
  while [[ $# -gt 0 && "$1" != -- ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == -- ]] && shift
  args=("$@")
  mkdir -p "$dir"
  # shellcheck disable=SC2046
  (cd "$dir" && env $(arm_env "$arm") "${envs[@]}" timeout 1500 "$BIN" --mode fast h3 gen --weights $L/ours-$ck \
     --prompt "$P" --seed 1024 --height "$h" --width "$w" --num-frames 124 "${COMMON[@]}" \
     --adaln-cache $L/adaln-$ck.cache --clip-dir "$dir/clip" "${args[@]}") >"$dir/stdout.log" 2>"$dir/stderr.log"
}
trace() { # <ck> <arm> <h> <w>
  local d=$SRV/trace/$1-$2-${3}p
  gen "$1" "$2" "$d" "$3" "$4" FASTVIDEO_GPU_TRACE=1 -- --no-mp4
}
for res in 768 480; do
  want "trace$res" || continue
  [[ $res == 768 ]] && wh="768 1344" || wh="480 832"
  for arm in $ARMS; do
    # shellcheck disable=SC2086
    step "trace$res-$arm" trace v2 "$arm" $wh
  done
done
if want parity; then
  for arm in $ARMS; do
    step "parity-$arm" gen v2 "$arm" $SRV/parity/$arm 480 832 FASTVIDEO_DUMP_DIR=$SRV/parity/$arm-dump --
  done
  if [[ " $ARMS " == *" bf16vsa "* && " $ARMS " == *" fp4 "* ]]; then
    step compare-dumps "$BIN" compare-dumps --baseline $SRV/parity/bf16vsa-dump --candidate $SRV/parity/fp4-dump \
      >$SRV/compare-dumps-bf16vsa-fp4.txt 2>&1
    step compare-clips "$BIN" --out $SRV/parity/compare --tag "bf16vsa--fp4" compare-clips \
      --baseline $SRV/parity/bf16vsa/clip --candidate $SRV/parity/fp4/clip --lpips $L/lpips --lpips-pairs all \
      >$SRV/parity/compare.log 2>&1
  fi
fi
if want cells; then
  CKS="v2"
  [[ -d $L/ours-4s ]] && CKS="v2 4s"
  for ck in $CKS; do
    for arm in $ARMS; do
      step "cell-$ck-$arm" gen "$ck" "$arm" $SRV/cells/$ck-$arm 480 832 -- --prompts $L/prompts-cells.json --warm --no-mp4
    done
  done
fi
rc=0
awk -F'\t' '$2 != 0 {bad=1} END {exit bad}' $SRV/status.tsv || rc=1
finish "$rc"
