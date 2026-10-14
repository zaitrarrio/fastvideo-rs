#!/usr/bin/env bash
# Runs ON the FastH3 Trim NVFP4 validation pod (scripts/gpu/h3-trim-nvfp4-validate.sh; docs/ports/h3-trim.md §5).
# One RTX PRO 6000 on this branch's runtime image (its fv-gpucheck and scripts/gpu), the R2 plan in
# /opt/fv/plan.json; h3-base from the EU volume at /workspace (read only) or, in r2 mode, fetched with the rest.
#
#   1. fetch the Trim trees (bf16 + the published NVFP4 file) from R2; LPIPS weights;
#   2. `kernels --groups nvfp4_linear`: the published-layout operand / stack against host math;
#   3. parity at 832x480, one prompt and seed, per-step latents dumped: bf16 linears (FASTVIDEO_H3_QUANT=off),
#      the served MXFP8 default and NVFP4, all on the same cached conditioning; `compare-dumps` bf16 -> each;
#   4. speed, memory and quality cells: {bf16, mxfp8, nvfp4} x {480p 832x480, 768p 1344x768}, 5 s, a three-prompt
#      set, `--warm` (one untimed generation first), one process per cell, nvidia-smi sampled every second;
#   5. `compare-clips` (PSNR, LPIPS, sharpness, jitter) and `gate` (scripts/gpu/gate-policy.toml) of NVFP4 against
#      the MXFP8 default and against bf16, per resolution.
#
# /srv is served on :8000 (Runpod proxy): log.txt, status.tsv (step, rc, seconds), the cells' logs, benchmark.json,
# compare / gate reports, the parity clips, and DONE (0 = every step ok). Nothing is written to /workspace.
# Env: FV_WEIGHTS_MODE (volume | r2), FV_SKIP (space-separated step names), FV_CELL_RES (default "480 768").
set -u
SRV=/srv
L=/fvw
G=/opt/fastvideo-rs/scripts/gpu
BIN=/opt/fastvideo-rs/target/release/fv-gpucheck
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
TRIM=$L/fasth3-trim-8step
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
log "start: mode ${FV_WEIGHTS_MODE:-volume}, $("$BIN" --version 2>/dev/null | head -1)"
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
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# ---- 1. weights, LPIPS ------------------------------------------------------------------------------------------
step fetch "$PY" $G/r2-fetch.py --root "$L" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1 &
fpid=$!
step lpips bash $G/fetch-lpips.sh $L/lpips >$SRV/lpips.log 2>&1
wait "$fpid"
[[ -f "$TRIM/transformer/config.json" && -d "$L/fasth3-trim-8step-nvfp4/diffusion_models" ]] || { log "weights missing"; echo 1 >$SRV/DONE; exit 1; }
[[ -d "$W/h3-base/tokenizer" ]] || { log "h3-base missing under $W"; echo 1 >$SRV/DONE; exit 1; }

# ---- 2. the published-layout NVFP4 linear on this GPU ---------------------------------------------------------
step kernels env FV_NVFP4_BENCH=0 "$BIN" --out $SRV/kernels kernels --groups nvfp4_linear >$SRV/kernels.log 2>&1

# ---- 3. parity: per-step latents, bf16 vs MXFP8 vs NVFP4 --------------------------------------------------------
P="${FV_PROMPT:-A man in his thirties talking to the camera in a bright living room, medium close-up, natural expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>}"
COMMON=(--text-encoder streamed --text-weights "$W/h3-base" --text-cache "$L/text-cache" --adaln-cache "$L/trim-adaln.cache")
recipe_of() { [[ "$1" == nvfp4 ]] && echo trim-8step-nvfp4 || echo trim-8step; }
quant_of() { case "$1" in bf16) echo off ;; mxfp8) echo mxfp8 ;; *) echo mxfp8 ;; esac; }
parity() { # <arm>
  env FASTVIDEO_H3_QUANT="$(quant_of "$1")" FASTVIDEO_DUMP_DIR=$SRV/parity/$1-dump \
    "$BIN" --mode fast h3 gen --weights "$TRIM" --h3-recipe "$(recipe_of "$1")" --prompt "$P" --seed 1024 \
    --height 480 --width 832 --seconds 5 "${COMMON[@]}" --clip-dir $SRV/parity/$1 >$SRV/parity-$1.log 2>&1
}
mkdir -p $SRV/parity
for arm in bf16 nvfp4 mxfp8; do step "parity-$arm" parity "$arm"; done
for arm in nvfp4 mxfp8; do
  step "compare-dumps-bf16-$arm" "$BIN" compare-dumps --baseline $SRV/parity/bf16-dump --candidate $SRV/parity/$arm-dump \
    >$SRV/compare-dumps-bf16-$arm.txt 2>&1
  step "compare-clips-parity-bf16-$arm" "$BIN" --out $SRV/parity/compare --tag "bf16--$arm" compare-clips \
    --baseline $SRV/parity/bf16 --candidate $SRV/parity/$arm --lpips $L/lpips --lpips-pairs all >$SRV/parity/compare-$arm.log 2>&1
done
step compare-dumps-mxfp8-nvfp4 "$BIN" compare-dumps --baseline $SRV/parity/mxfp8-dump --candidate $SRV/parity/nvfp4-dump \
  >$SRV/compare-dumps-mxfp8-nvfp4.txt 2>&1

# ---- 4. speed / memory / quality cells ------------------------------------------------------------------------
# Three prompts of scripts/gpu/prompts-hd5.json (talking head, landscape, broadcast; their own seeds) at one canvas.
RUNS=$SRV/runs
mkdir -p $RUNS
for res in ${FV_CELL_RES:-480 768}; do
  if [[ "$res" == 480 ]]; then h=480; w=832; else h=768; w=1344; fi
  "$PY" - "$G/prompts-hd5.json" "$RUNS/prompts-$res.json" "$h" "$w" <<'PYEOF'
import json, sys
src, out, h, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(src))
keep = ["talking-head", "spark-mountain-lake", "ltx-newsbroadcast"]
ps = [dict(p, height=h, width=w) for p in d["prompts"] if p["name"] in keep]
json.dump({"name": f"trim-nvfp4-{h}p", "source": f"scripts/gpu/prompts-hd5.json ({', '.join(keep)}) at {w}x{h}",
           "prompts": ps}, open(out, "w"), indent=1)
PYEOF
done
cell() { # <arm> <res>
  local arm="$1" res="$2" dir="$RUNS/$1-$2" rc h=768 w=1344
  [[ "$res" == 480 ]] && { h=480; w=832; }
  mkdir -p "$dir"
  # One process per cell. --warm: one untimed generation (the warm-up prompt, this canvas) before the set.
  (cd "$dir" && env FASTVIDEO_H3_QUANT="$(quant_of "$arm")" timeout 1500 \
    "$BIN" --mode fast h3 gen --weights "$TRIM" --h3-recipe "$(recipe_of "$arm")" --prompts "$RUNS/prompts-$res.json" \
    --prompt "$P" --seed 7 --height "$h" --width "$w" --seconds 5 --warm "${COMMON[@]}" --clip-dir "$dir/frames") \
    >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >"$dir/after-mib.txt" 2>&1
  return $rc
}
for res in ${FV_CELL_RES:-480 768}; do
  for arm in bf16 mxfp8 nvfp4; do step "cell-$arm-$res" cell "$arm" "$res"; done
done

# ---- 5. compare-clips and the gate ----------------------------------------------------------------------------
mkdir -p $RUNS/compare $RUNS/gate
for res in ${FV_CELL_RES:-480 768}; do
  for pair in mxfp8:nvfp4 bf16:nvfp4 bf16:mxfp8; do
    b="${pair%%:*}-$res" c="${pair#*:}-$res"
    for p in talking-head spark-mountain-lake ltx-newsbroadcast; do
      [[ -d "$RUNS/$b/frames/$p" && -d "$RUNS/$c/frames/$p" ]] || continue
      step "compare-$b--$c-$p" "$BIN" --out $RUNS/compare --tag "$b--$c-$p" compare-clips \
        --baseline "$RUNS/$b/frames/$p" --candidate "$RUNS/$c/frames/$p" --lpips $L/lpips \
        >"$RUNS/compare/$b--$c-$p.stdout.log" 2>"$RUNS/compare/$b--$c-$p.stderr.log"
    done
    cmp=()
    for f in "$RUNS/compare/compare-clips-$b--$c-"*.json; do [[ -f "$f" ]] && cmp+=(--compare "$f"); done
    [[ -f "$RUNS/$b/benchmark.json" && -f "$RUNS/$c/benchmark.json" ]] || continue
    step "gate-$b--$c" "$BIN" --out $RUNS/gate --tag "$b--$c" gate --baseline "$RUNS/$b" --candidate "$RUNS/$c" \
      --policy $G/gate-policy.toml --kind lossy "${cmp[@]}" >"$RUNS/gate/$b--$c.stdout.log" 2>"$RUNS/gate/$b--$c.stderr.log"
  done
done

# Small results bundle (no frames): logs, reports, benchmark.json, the parity clips' mp4.
tar -czf $SRV/results.tgz -C $SRV --exclude='*.png' --exclude='*-dump' log.txt status.tsv nvidia-smi.txt nvsmi.csv \
  fetch.log kernels kernels.log parity runs compare-dumps-*.txt parity-*.log 2>/dev/null
bad="$(awk -F'\t' '$2 != 0' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
