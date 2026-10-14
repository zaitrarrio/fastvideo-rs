#!/usr/bin/env bash
# Runs ON the H3 NVFP4 tier-ship smoke pod (scripts/gpu/h3-tier-ship-validate.sh; docs/perf/h3-nvfp4-tier-options.md
# "Shipped"). One RTX PRO 6000 on a fastvideo-rs runtime image; this branch's fv-gpucheck (fv-gpucheck), main's
# (fv-gpucheck-main), scripts/gpu and the R2 plan arrive in a bundle under /opt/fv. h3-base from the EU volume at
# /workspace (read only) or, in r2 mode, fetched.
#
# One prompt (talking-head, seed 1024), 832x480, 124 frames, --warm, one process per run:
#   main-default  main's binary, V2-NVFP4 as main ran it (a folder: the R2 tree's transformer/, h3-base's vae/):
#                 the official VAE with MXFP8 linears, the reference for the byte identity below
#   v2-default    this branch, the R2 tree as fetched (--h3-recipe v2-nvfp4): the published NVFP4 VAE by default,
#                 its third shard from h3-base's vae/
#   v2-vae0       this branch, FASTVIDEO_H3_VAE_NVFP4=0: h3-base's vae/ (MXFP8); must be byte-identical to main-default
#   v2-preview    this branch, --taeh3-weights (the fast-preview decoder)
#   trim-fast     this branch, --h3-recipe trim-8step-nvfp4-fp4attn (sparse FP4 VSA by the recipe)
# Steps (FV_STEPS; default "fetch prep runs identity"). /srv is served on :8000: log.txt, status.tsv, results.tgz, DONE.
# Nothing is written to R2 or /workspace.
set -u
SRV=/srv
L=/fvw
B=/opt/fv
BIN=$B/fv-gpucheck
MAIN=$B/fv-gpucheck-main
W=/workspace/weights
[[ "${FV_WEIGHTS_MODE:-volume}" == r2 ]] && W=$L
STEPS=" ${FV_STEPS:-fetch prep runs identity} "
RUNS="${FV_RUNS:-main-default v2-default v2-vae0 v2-preview trim-fast}"
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
stop_sampler() { # by its exact PID, so no file grows while tar reads it
  if [[ -n "$SAMPLER" ]]; then kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null; SAMPLER=""; fi
}
pack() {
  # Samplers stopped first (a growing file makes tar exit 1); the archive is checked before it is published.
  stop_sampler
  (cd $SRV && tar --exclude='*.png' --exclude='*.raw' --exclude='*.f32' --exclude='*.bin' --exclude='*.safetensors' \
     --exclude='results.tgz*' --exclude='frames' -czf results.tgz.tmp .) >/dev/null 2>&1
  # Two middle frames per run for a look.
  (cd $SRV && find runs -path '*/frames/*' -name 'frame-062.png' 2>/dev/null | tar -czf frames-mid.tgz.tmp -T -) >/dev/null 2>&1
  if (cd $SRV && gzip -t results.tgz.tmp && tar -tzf results.tgz.tmp >/dev/null); then
    mv $SRV/results.tgz.tmp $SRV/results.tgz
    [[ -f $SRV/frames-mid.tgz.tmp ]] && (cd $SRV && gzip -t frames-mid.tgz.tmp && tar -tzf frames-mid.tgz.tmp >/dev/null) \
      && mv $SRV/frames-mid.tgz.tmp $SRV/frames-mid.tgz
    log "packed results.tgz ($(du -h $SRV/results.tgz | cut -f1), $(tar -tzf $SRV/results.tgz | wc -l) entries)"
    return 0
  fi
  log "pack: results.tgz.tmp failed its check"
  return 1
}
finish() { pack || echo pack-failed >$SRV/PACK_FAILED; echo "$1" >$SRV/DONE; log "DONE $1"; }

log "start: mode ${FV_WEIGHTS_MODE:-volume}, steps${STEPS}, runs $RUNS; $("$BIN" --version 2>/dev/null | head -1); main $("$MAIN" --version 2>/dev/null | head -1)"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi

if want fetch; then
  step fetch "$PY" $B/gpu/r2-fetch.py --root "$L" --plan $B/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1 \
    || { finish 1; exit 1; }
fi

V2=$L/fasth3-v2-nvfp4
TRIM=$L/fasth3-trim-8step
TAE=$L/auxiliary/tae/taeh3.safetensors
prep() {
  [[ -d "$W/h3-base/tokenizer" && -d "$W/h3-base/text_encoder" && -f "$W/h3-base/vae/diffusion_pytorch_model-00003-of-00003.safetensors" ]] \
    || { log "h3-base missing under $W"; return 1; }
  [[ -f "$V2/transformer/config.json" && -f "$V2/vae/config.json" && -f "$TRIM/transformer/config.json" \
     && -d "$L/fasth3-trim-8step-nvfp4/diffusion_models" && -f "$TAE" ]] || { log "R2 trees missing"; return 1; }
  # The R2 tree as fetched carries vae/ shards 1-2 only: this branch completes it from h3-base by itself.
  ls -l "$V2/vae" >$SRV/v2-vae-listing.txt
  # main's V2-NVFP4: the folder main's tier-options ran (the R2 tree's transformer/, h3-base's vae/); main reads the
  # contract from fastvideo_inference.json or defaults to the 8-step V2 contract, the same as --h3-recipe v2-nvfp4.
  mkdir -p $L/main-v2
  ln -sfn "$V2/transformer" $L/main-v2/transformer
  ln -sfn "$W/h3-base/vae" $L/main-v2/vae
  "$PY" - "$B/gpu/prompts-hd5.json" "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
open(f"{out}/prompt.txt", "w").write(next(p["prompt"] for p in d["prompts"] if p["name"] == "talking-head"))
PYEOF
}
if want prep; then step prep prep >$SRV/prep.log 2>&1 || { finish 1; exit 1; }; fi
P="$(cat $L/prompt.txt 2>/dev/null)"
COMMON=(--text-encoder streamed --text-weights "$W/h3-base" --text-cache "$L/text-cache" --seed 1024 --height 480 --width 832 --num-frames 124 --warm --no-mp4)

run() { # <name>
  local name="$1" dir="$SRV/runs/$1" bin="$BIN" args=() envs=(FV_RUN="$1") rc
  case "$name" in
    main-default) bin="$MAIN"; args=(--weights $L/main-v2) ;;
    v2-default) args=(--weights "$V2" --h3-recipe v2-nvfp4) ;;
    v2-vae0) args=(--weights "$V2" --h3-recipe v2-nvfp4); envs+=(FASTVIDEO_H3_VAE_NVFP4=0) ;;
    v2-preview) args=(--weights "$V2" --h3-recipe v2-nvfp4 --taeh3-weights "$TAE") ;;
    trim-fast) args=(--weights "$TRIM" --h3-recipe trim-8step-nvfp4-fp4attn) ;;
    *) log "unknown run $name"; return 2 ;;
  esac
  mkdir -p "$dir"
  start_sampler "$dir/nvsmi.csv"
  (cd "$dir" && env "${envs[@]}" timeout 1500 "$bin" --mode fast h3 gen "${args[@]}" --prompt "$P" "${COMMON[@]}" \
     --clip-dir "$dir/frames") >"$dir/stdout.log" 2>"$dir/stderr.log"
  rc=$?
  stop_sampler
  grep -hE "h3 vae|h3: VSA|h3 nvfp4|taeh3|h3 recipe|h3 text encoder" "$dir/stderr.log" | sort -u >"$dir/config-lines.txt" 2>/dev/null
  # Every output file's SHA-256 (frames, audio), for the byte-identity check.
  (cd "$dir/frames" 2>/dev/null && find . -type f \( -name '*.png' -o -name '*.wav' \) | sort | xargs -r sha256sum) >"$dir/frames.sha256"
  return $rc
}
if want runs; then
  for r in $RUNS; do step "run-$r" run "$r"; done
fi

identity() {
  local a=$SRV/runs/main-default/frames.sha256 b=$SRV/runs/v2-vae0/frames.sha256
  [[ -s "$a" && -s "$b" ]] || { log "identity: a run is missing"; return 1; }
  if diff -q "$a" "$b" >/dev/null; then
    log "identity: v2-vae0 is byte-identical to main-default ($(wc -l <"$a") files)"
    echo "identical $(wc -l <"$a") files" >$SRV/identity.txt
  else
    log "identity: v2-vae0 DIFFERS from main-default"
    diff "$a" "$b" | head -20 >$SRV/identity.txt
    return 1
  fi
  # The NVFP4 default is a different decoder: its frames differ (said, not a failure).
  diff -q "$a" $SRV/runs/v2-default/frames.sha256 >/dev/null && echo "WARNING: v2-default equals main-default" >>$SRV/identity.txt
  return 0
}
if want identity; then step identity identity; fi
rc=0
awk -F'\t' '$2 != 0 {bad=1} END {exit bad}' $SRV/status.tsv || rc=1
finish "$rc"
