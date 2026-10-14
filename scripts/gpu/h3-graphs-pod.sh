#!/usr/bin/env bash
# Runs ON the H3 denoise-graphs pod (scripts/gpu/h3-graphs-validate.sh; docs/perf/h3-denoise-graphs.md).
# One RTX PRO 6000 on a fastvideo-rs runtime image (its fv-gpucheck); this script arrives in the pod's env
# (FV_POD_SCRIPT_B64). The R2 plan (presigned GETs from the read-only key) is PUT to /opt/fv/plan.json; h3-base, the
# h3-turbo MXFP8 tree and its LoRA from the EU volume at /workspace (read only) or, in r2 mode, h3-base's text encoder,
# tokenizer and VAEs from R2 (no h3-turbo cells then).
#
# Steps (FV_STEPS, space-separated; default all, in this order):
#   fetch   the NVFP4 trees' transformer/ (V2 and 4-step); h3-base in r2 mode
#   prep    model folders (NVFP4 transformer + h3-base's text encoder / tokenizer / VAEs + the Hub configs), prompts
#   trace   CUPTI trace of one step (FASTVIDEO_GPU_TRACE, step 1) with FASTVIDEO_H3_GRAPHS=1, h3-demo: an eager warm
#           request, then the capture, then a replay; V2 480p / 768p, h3-turbo 768p
#   cells   timed --warm processes per checkpoint (FV_CKS, default "v2 4s turbo") and arm (FV_ARMS, default
#           "sync lazy graphs"): h3-demo and spark-mountain-lake, three requests each, seed 1024, 124 frames, at
#           480p and 768p (h3-turbo: 768p)
#   hash    sha256 of every clip's PNG frames + WAV (then the PNGs are deleted): byte identity across arms and runs
# Arms: sync = FASTVIDEO_H3_STEP_SYNC=1 (a synchronize per step, the timing before this change); lazy = the default
# lazy step clock; graphs = FASTVIDEO_H3_GRAPHS=1 (per request shape: 1st eager, 2nd captured, later replayed).
# NVFP4 cells run the sparse FP4 tier (FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=1).
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
STEPS=" ${FV_STEPS:-fetch prep trace cells hash} "
ARMS="${FV_ARMS:-sync lazy graphs}"
CKS="${FV_CKS:-v2 4s turbo}"
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
pack() { # [final]
  # A file growing under tar makes it exit 1 (a previous run lost its results that way): intermediate packs leave the
  # nvidia-smi sampler's file out; the final one stops the sampler first. The archive is listed back before it
  # replaces the previous one.
  local skip=(--exclude=nvsmi.csv)
  if [[ "${1:-}" == final ]]; then
    [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null && wait "$SAMPLER" 2>/dev/null
    SAMPLER=""
    skip=()
  fi
  (cd $SRV && tar "${skip[@]}" --exclude='*.png' --exclude='*.raw' --exclude='*.f32' --exclude='*.bin' \
     --exclude='*.safetensors' --exclude='*-dump' --exclude='results.tgz*' -czf results.tgz.tmp . \
     && tar -tzf results.tgz.tmp >/dev/null && mv results.tgz.tmp results.tgz)
}
finish() { pack final || log "pack failed"; echo "$1" >$SRV/DONE; log "DONE $1"; }
# /srv on :8000, and PUT /plan (header X-FV-Token = $FV_UPLOAD_TOKEN) writes /opt/fv/plan.json.
cat >/opt/fv/serve.py <<'PYEOF'
import http.server, os, sys
TOKEN = os.environ.get("FV_UPLOAD_TOKEN", "")
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory="/srv", **k)
    def log_message(self, *a):
        pass
    def do_PUT(self):
        if not TOKEN or self.headers.get("X-FV-Token") != TOKEN or self.path != "/plan":
            self.send_response(403); self.end_headers(); return
        n = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(n)
        fd = os.open("/opt/fv/plan.json.tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace("/opt/fv/plan.json.tmp", "/opt/fv/plan.json")
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok\n")
http.server.ThreadingHTTPServer(("", 8000), H).serve_forever()
PYEOF
(env FV_UPLOAD_TOKEN="${FV_UPLOAD_TOKEN:-}" "$PY" /opt/fv/serve.py >/dev/null 2>&1 &)
unset FV_UPLOAD_TOKEN
log "start: mode ${FV_WEIGHTS_MODE:-volume}, steps${STEPS}, arms $ARMS, cks $CKS; $("$BIN" --version 2>/dev/null | head -1)"
nvidia-smi >$SRV/nvidia-smi.txt 2>&1
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [[ "${used:-0}" -gt 4096 ]]; then
  log "GPU busy before start: ${used} MiB used; deleting this pod"
  echo busy >$SRV/DONE
  [[ -n "${RUNPOD_POD_ID:-}" ]] && curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >>$SRV/log.txt 2>&1
  exit 1
fi
(while sleep 1; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits; done) >$SRV/nvsmi.csv 2>/dev/null &
SAMPLER=$!

if want fetch; then
  for _ in $(seq 180); do [[ -s /opt/fv/plan.json ]] && break; sleep 5; done
  [[ -s /opt/fv/plan.json ]] || { log "no plan uploaded in 15 min"; finish 1; exit 1; }
  log "plan received"
  step fetch "$PY" $G/r2-fetch.py --root "$L" --plan /opt/fv/plan.json --status $SRV/fetch-status.json >$SRV/fetch.log 2>&1 \
    || { finish 1; exit 1; }
fi

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
  small FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4 $S4REV $L/small-4s || return 1
  folder $L/ours-4s $L/small-4s $L/fasth3-4step-vsa-nvfp4/transformer
  "$PY" - "$G/prompts-hd5.json" "$L" <<'PYEOF'
import json, sys
src, out = sys.argv[1], sys.argv[2]
ps = {p["name"]: p["prompt"] for p in json.load(open(src))["prompts"]}
keep = ["h3-demo", "spark-mountain-lake"]
canv = {"480p": (480, 832), "768p": (768, 1344)}
def rows(cs, names=keep, runs="abc"):
    return [{"name": f"{n}-{c}-{r}", "prompt": ps[n], "seed": 1024, "height": canv[c][0], "width": canv[c][1]}
            for c in cs for n in names for r in runs]
src_note = "scripts/gpu/prompts-hd5.json (h3-demo, spark-mountain-lake) at seed 1024, three requests each"
json.dump({"name": "h3-graphs", "source": src_note, "prompts": rows(["480p", "768p"])},
          open(f"{out}/prompts-both.json", "w"), indent=1)
json.dump({"name": "h3-graphs-768", "source": src_note, "prompts": rows(["768p"])},
          open(f"{out}/prompts-768.json", "w"), indent=1)
# Trace: a warm-up request (spark), then h3-demo eager (warm), captured, replayed.
for c in canv:
    t = [{"name": f"spark-mountain-lake-{c}-w", "prompt": ps["spark-mountain-lake"], "seed": 1024,
          "height": canv[c][0], "width": canv[c][1]}] + rows([c], ["h3-demo"])
    json.dump({"name": f"h3-graphs-trace-{c}", "source": src_note, "prompts": t},
              open(f"{out}/prompts-trace-{c}.json", "w"), indent=1)
PYEOF
}
if want prep; then step prep prep >$SRV/prep.log 2>&1 || { finish 1; exit 1; }; fi
TURBO_OK=0
[[ -d "$W/h3-base/transformer_prequant_4step-vsa_mxfp8" && -d "$W/FastH3-4-step-Preview-v1-LoRA" ]] && TURBO_OK=1
[[ $TURBO_OK == 1 ]] || log "no h3-turbo tree under $W: h3-turbo cells skipped"
# The canvas only sizes --warm's untimed first request (prompts[0] at it); every listed prompt carries its own.
COMMON=(--text-encoder streamed --text-weights "$W/h3-base" --text-cache "$L/text-cache" --no-mp4 --num-frames 124
        --prompt unused)
arm_env() { # <arm>
  case "$1" in
    sync) echo "FASTVIDEO_H3_STEP_SYNC=1" ;;
    lazy) echo "" ;;
    graphs) echo "FASTVIDEO_H3_GRAPHS=1" ;;
  esac
}
ck_args() { # <ck>: env words, then -- , then gen args
  case "$1" in
    v2) echo "FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=1 -- --weights $L/ours-v2 --adaln-cache $L/adaln-v2.cache --height 480 --width 832" ;;
    4s) echo "FASTVIDEO_H3_NVFP4_ATTN=1 FASTVIDEO_H3_VSA_FP4=1 -- --weights $L/ours-4s --adaln-cache $L/adaln-4s.cache --height 480 --width 832" ;;
    turbo) echo "-- --weights $W/h3-base --h3-recipe 4step-vsa --adaln-cache $L/adaln-turbo.cache --height 768 --width 1344" ;;
  esac
}
gen() { # <ck> <dir> <prompts json> [extra env...] -- [extra args...]
  local ck="$1" dir="$2" prompts="$3"
  shift 3
  local envs=() args=() ckw pre=()
  while [[ $# -gt 0 && "$1" != -- ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == -- ]] && shift
  args=("$@")
  read -r -a ckw <<<"$(ck_args "$ck")"
  local cenv=() cargs=() seen=0 w
  for w in "${ckw[@]}"; do
    if [[ "$w" == -- ]]; then seen=1; continue; fi
    if [[ $seen == 0 ]]; then cenv+=("$w"); else cargs+=("$w"); fi
  done
  [[ "$ck" == turbo ]] && pre=(--techniques h3/fasth3_4step_vsa)
  mkdir -p "$dir"
  (cd "$dir" && env "${cenv[@]}" "${envs[@]}" timeout 2400 "$BIN" --mode fast "${pre[@]}" --out "$dir/report" h3 gen \
     "${cargs[@]}" "${COMMON[@]}" --prompts "$prompts" --clip-dir "$dir/clip" "${args[@]}") \
     >"$dir/stdout.log" 2>"$dir/stderr.log"
}
hash_dir() { # <cell dir>: one line per clip (sha256 of its PNG + WAV bytes, as fv-gpucheck's i2v parity), then rm PNGs
  "$PY" - "$1" "$SRV/hashes.tsv" <<'PYEOF'
import hashlib, os, sys
cell, out = sys.argv[1], sys.argv[2]
clip = os.path.join(cell, "clip")
rows = []
for root, dirs, files in sorted(os.walk(clip)):
    parts = sorted(f for f in files if f.endswith((".png", ".wav")))
    if not parts:
        continue
    h = hashlib.sha256()
    for f in parts:
        h.update(f.encode())
        h.update(open(os.path.join(root, f), "rb").read())
    rows.append(f"{os.path.relpath(cell, '/srv/cells')}\t{os.path.relpath(root, clip)}\t{len(parts)}\t{h.hexdigest()}\n")
with open(out, "a") as f:
    f.writelines(rows)
PYEOF
  find "$1/clip" -name '*.png' -delete 2>/dev/null
}

if want trace; then
  for t in "v2 480p" "v2 768p" "turbo 768p"; do
    read -r ck c <<<"$t"
    [[ $ck == turbo && $TURBO_OK != 1 ]] && continue
    d=$SRV/trace/$ck-$c
    step "trace-$ck-$c" gen "$ck" "$d" "$L/prompts-trace-$c.json" FASTVIDEO_H3_GRAPHS=1 FASTVIDEO_GPU_TRACE=1 --
    find "$d/clip" -name '*.png' -delete 2>/dev/null
    pack
  done
fi
if want cells; then
  for ck in $CKS; do
    [[ $ck == turbo && $TURBO_OK != 1 ]] && continue
    p=$L/prompts-both.json
    [[ $ck == turbo ]] && p=$L/prompts-768.json
    for arm in $ARMS; do
      d=$SRV/cells/$ck-$arm
      # shellcheck disable=SC2046
      step "cell-$ck-$arm" gen "$ck" "$d" "$p" $(arm_env "$arm") -- --warm
      want hash && hash_dir "$d"
      pack
    done
  done
fi
rc=0
awk -F'\t' '$2 != 0 {bad=1} END {exit bad}' $SRV/status.tsv || rc=1
finish "$rc"
