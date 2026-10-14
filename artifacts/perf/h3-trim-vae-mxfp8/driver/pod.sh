#!/usr/bin/env bash
# Runs ON the pod (launch.sh): MXFP8 against bf16 decoder linears on the FastH3 Trim 26-block video VAE.
# One RTX PRO 6000, the runtime image's CUDA libraries, this branch's fv-gpucheck from the bundle (/opt/fv), the
# Trim tree from R2 (plan.json in the bundle), h3-base and LPIPS from the EU volume at /workspace (read only).
#
#   ab480, ab768  h3-trim (recipe trim-8step, the served MXFP8 DiT), 5 s, prompts-5x3.json (5 prompts x 3 seeds),
#                 --warm, one process per resolution, FASTVIDEO_H3_VAE_MXFP8=1 FASTVIDEO_H3_VAE_AB=1: every clip
#                 decoded with the MXFP8 linears (<clip>/) and then, from the same latents, with the bf16 linears
#                 (<clip>/vae-bf16/). The DiT runs once per clip.
#   v480, v768    FASTVIDEO_H3_VAE_MXFP8=0 (the bf16 decoder as loaded today), one clip (h3-demo-s0): its frames
#                 must equal ab*'s vae-bf16 frames byte for byte (the in-process bf16 decode is the =0 decode), and
#                 its peak memory is the bf16 arm's.
#   compare-clips (LPIPS, PSNR, sharpness, jitter) bf16 -> MXFP8 per clip, then `gate` (scripts/gpu/gate-policy.toml,
#   kind lossy) on those reports.
# /srv is served on :8000: log.txt, status.tsv, the cells' logs and benchmark.json, results.tgz, DONE.
set -u
SRV=/srv
L=/fvw
W=/workspace/weights
BIN=/opt/fv/fv-gpucheck
TRIM=$L/fasth3-trim-8step
mkdir -p "$SRV" "$L"
cd "$SRV" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >>$SRV/log.txt; }
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
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
log "start: $("$BIN" --version 2>/dev/null | head -1); GPU memory used at boot: ${used:-?} MiB"
# A host whose GPU memory is already held (95.8 of 97.9 GiB used, no process of ours) cannot run anything.
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &

# cudarc dlopens the unversioned names (as scripts/gpu/runpod-matrix.sh links them).
mkdir -p $L/libs
. /opt/fastvideo-rs/scripts/gpu/cuda-13.pins
for spec in "cudnn:/lib/x86_64-linux-gnu/${CUDA_CUDNN_SONAME}" \
  "cublas:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLAS_SONAME}" \
  "cublasLt:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_CUBLASLT_SONAME}" \
  "nvrtc:/usr/local/cuda/targets/x86_64-linux/lib/${CUDA_NVRTC_SONAME}"; do
  n="${spec%%:*}"; s="${spec#*:}"
  [ -e "$L/libs/lib$n.so" ] || { [ -e "$s" ] && ln -sf "$(readlink -f "$s")" "$L/libs/lib$n.so"; }
done
export LD_LIBRARY_PATH="$L/libs:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export RUST_LOG=info FASTVIDEO_CACHE=$L/cache XDG_CACHE_HOME=$L/cache

step fetch "$PY" /opt/fv/r2-fetch.py --root "$L" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1
[[ -f "$TRIM/transformer/config.json" && -f "$TRIM/vae/config.json" ]] || { log "Trim tree missing"; echo 1 >$SRV/DONE; exit 1; }
[[ -d "$W/h3-base/tokenizer" && -d "$W/auxiliary/lpips" ]] || { log "h3-base / lpips missing under $W"; echo 1 >$SRV/DONE; exit 1; }

# Deterministic kernel picks (as the #106 gate run) and the VAE path check on the first decode.
PIN=(FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn FASTVIDEO_H3_VAE_CHECK=1)
RUNS=$SRV/runs
mkdir -p $RUNS
cell() { # <name> <480|768> <prompts json> <warm 0|1> <env K=V...>
  local name="$1" res="$2" ps="$3" dir="$RUNS/$1" h=768 w=1344 rc pid wf=()
  [[ "$4" == 1 ]] && wf=(--warm)
  shift 4
  [[ "$res" == 480 ]] && { h=480; w=832; }
  mkdir -p "$dir"
  (while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) >"$dir/smi.csv" 2>/dev/null &
  pid=$!
  (cd "$dir" && env "${PIN[@]}" "$@" timeout "${CELL_TIMEOUT:-1500}" \
    "$BIN" --mode fast h3 gen --weights "$TRIM" --h3-recipe trim-8step --text-weights "$W/h3-base" \
    --height "$h" --width "$w" --seconds 5 --prompt x --prompts "$ps" "${wf[@]}" \
    --text-encoder resident-fp8 --dit-offload resident --no-mp4 \
    --adaln-cache "$L/trim-adaln.cache" --text-cache "$L/text-cache" --clip-dir "$dir/frames") >"$dir/log.txt" 2>&1
  rc=$?
  kill $pid 2>/dev/null
  log "$name: peak $(sort -n "$dir/smi.csv" | tail -1) MiB (nvidia-smi)"
  grep -h "h3 vae check\|h3 vae:\|h3 vae a/b\|h3 video decode + mp4" "$dir/log.txt" | cut -c1-400 >"$dir/vae.log"
  for d in "$dir"/frames/*/; do
    [[ -d "$d" ]] || continue
    echo "$(basename "$d") $(cat "$d"/frame-*.png 2>/dev/null | sha256sum | cut -c1-16) $(cat "$d"/vae-bf16/frame-*.png 2>/dev/null | sha256sum | cut -c1-16)"
  done >"$dir/frames.sha"
  return $rc
}
"$PY" - /opt/fv/prompts-5x3.json $RUNS/prompts-1.json <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump(dict(d, name=d["name"] + "-first", prompts=d["prompts"][:1]), open(sys.argv[2], "w"), indent=1)
PYEOF
# compare-clips bf16 -> MXFP8 for every timed clip of ab<res>, one at a time (~7 s each), right after the cell, so
# a run cut short keeps what it measured. (A first version ran four at once and joined them with a bare `wait`,
# which also waited for the nvidia-smi sampler above and hung after four clips.)
mkdir -p $RUNS/compare $RUNS/gate
compares() { # <res>
  local d p
  for d in "$RUNS/ab$1"/frames/*/; do
    p="$(basename "$d")"
    [[ "$p" == warmup || "$p" == cold || ! -d "$d/vae-bf16" ]] && continue
    "$BIN" --out $RUNS/compare --tag "bf16-$1--mx-$1-$p" compare-clips --baseline "$d/vae-bf16" --candidate "$d" \
      --lpips "$W/auxiliary/lpips" >"$RUNS/compare/bf16-$1--mx-$1-$p.log" 2>&1
  done
}
AB=(FASTVIDEO_H3_VAE_MXFP8=1 FASTVIDEO_H3_VAE_AB=1)
step ab480 cell ab480 480 /opt/fv/prompts-5x3.json 1 "${AB[@]}"
step compares-480 compares 480
step ab768 cell ab768 768 /opt/fv/prompts-5x3.json 1 "${AB[@]}"
step compares-768 compares 768
CELL_TIMEOUT=600 step v768 cell v768 768 $RUNS/prompts-1.json 0 FASTVIDEO_H3_VAE_MXFP8=0
CELL_TIMEOUT=600 step v480 cell v480 480 $RUNS/prompts-1.json 0 FASTVIDEO_H3_VAE_MXFP8=0

# The in-process bf16 decode against the =0 process's decode of the same clip: byte-identical frames.
for r in 480 768; do
  a="$(cat $RUNS/ab$r/frames/h3-demo-s0/vae-bf16/frame-*.png 2>/dev/null | sha256sum | cut -c1-16)"
  b="$(cat $RUNS/v$r/frames/h3-demo-s0/frame-*.png 2>/dev/null | sha256sum | cut -c1-16)"
  log "identity $r: ab$r vae-bf16 $a, v$r (=0) $b: $([[ "$a" == "$b" ]] && echo IDENTICAL || echo DIFFERENT)"
done

for r in 480 768; do
  cmp=()
  for f in $RUNS/compare/compare-clips-bf16-$r--mx-$r-*.json; do [[ -f "$f" ]] && cmp+=(--compare "$f"); done
  step "gate-$r" "$BIN" --out $RUNS/gate --tag "bf16-$r--mx-$r" gate --baseline "$RUNS/ab$r" --candidate "$RUNS/ab$r" \
    --policy /opt/fastvideo-rs/scripts/gpu/gate-policy.toml --kind lossy "${cmp[@]}" >"$RUNS/gate/gate-$r.out" 2>&1
done

tar -czf $SRV/results.tgz -C $SRV --exclude='*.png' log.txt status.tsv nvidia-smi.txt nvsmi.csv fetch.log runs 2>/dev/null
bad="$(awk -F'\t' '$2 != 0 && $1 !~ /^gate-/' $SRV/status.tsv | wc -l)"
log "end: $bad step(s) failed (gate steps not counted: the performance gate fails by design, same cell both sides)"
echo $(( bad > 0 ? 1 : 0 )) >$SRV/DONE
