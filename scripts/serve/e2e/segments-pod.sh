#!/usr/bin/env bash
# GPU smoke for `/v1/segments` (docs/serve/segments.md §6) on one Runpod pod:
# fv-serve with the SF-Wan causal backend (`FV_ENGINE=cuda`,
# `FV_SFWAN_WEIGHTS`), then a client ON THE POD pulls one stream segment by
# segment the way the unreel-cloudflare control plane does (`continuity:
# "native"`): four 6 s segments, a prompt change (`recache`) at segment 2 and
# a scene cut (`reset`) at segment 3; every MP4 is downloaded, ffprobed, and
# the seam between segments 0 and 1 is measured (mean absolute pixel
# difference across the seam against the clip's own inter-frame difference).
# Like scripts/serve/runpod-sfwan-whip.sh, SF-Wan is fetched from Hugging
# Face onto the container disk: nothing is written to a volume.
#
#   RUNPOD_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:… \
#     bash scripts/serve/e2e/segments-pod.sh
#
# The image must be the `serve` target built with `cuda,http-client` from a
# branch that has the segments router. H100 has no NVENC: the segments
# encoder is `libx264` there and `nvenc` on L40S / RTX PRO 6000.
# Env: RUNPOD_API_KEY, RUNPOD_GPU_TYPES ('|'-separated preference list),
# RUNPOD_GPU_MAX_DPH (default 4.0), FV_POD_CAP_S (default 3600, the detached
# backstop that deletes the pod even if this shell dies), FV_MIN_BALANCE
# (default 8), FV_SEG_COUNT (default 4), FV_SEG_SECONDS (default 6).
# Results: artifacts/serve/segments/<tag>/; the ledger gets a row per pod.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=../../gpu/runpod-price.sh
source "$ROOT/scripts/gpu/runpod-price.sh"
# shellcheck source-path=SCRIPTDIR source=../../gpu/runpod-cuda.sh
source "$ROOT/scripts/gpu/runpod-cuda.sh"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
GQL="https://api.runpod.io/graphql"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
: "${RUNPOD_IMAGE:?RUNPOD_IMAGE (the fv-serve image with cuda,http-client) missing}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA H100 80GB HBM3|NVIDIA RTX PRO 6000 Blackwell Server Edition|NVIDIA H100 NVL|NVIDIA L40S}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-4.0}"
CAP_S="${FV_POD_CAP_S:-3600}"
MIN_BALANCE="${FV_MIN_BALANCE:-8}"
NSEG="${FV_SEG_COUNT:-4}"
SEGS="${FV_SEG_SECONDS:-6}"
TAG="$(date -u +%m%d%H%M)"
OUT="$ROOT/artifacts/serve/segments/$TAG"
LEDGER="$ROOT/artifacts/runpod/serve/ledger.tsv"
mkdir -p "$OUT" "$(dirname "$LEDGER")"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
ledger() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$*" >>"$LEDGER"; }
rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"
}
proxy() { curl -sS --max-time 30 --fail "https://$1-8000.proxy.runpod.net/$2"; }
balance() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' \
    -d '{"query":"{ myself { clientBalance } }"}' "$GQL" | jq -r '.data.myself.clientBalance // empty'
}

# The client on the pod: segments.py (python3 is in the image; no deps).
client_py() {
  cat <<'EOF'
import json, os, subprocess, sys, time, urllib.request
S = os.environ.get("S", "http://127.0.0.1:8080"); OUT = os.environ["OUT"]
N = int(os.environ.get("NSEG", "4")); SECS = float(os.environ.get("SEGS", "6"))
P = [
 "A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed",
 "A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed",
 "A drone shot gliding over the same river as mist rolls in and the light turns blue and cold, slow steady forward camera motion, highly detailed",
 "A busy night market street in the rain, neon signs reflected in puddles, handheld camera walking forward, highly detailed",
]
def call(method, path, body=None):
    req = urllib.request.Request(S + path, method=method, data=(json.dumps(body).encode() if body is not None else None),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read() or b"{}")
def log(m):
    print(f"[client {time.strftime('%H:%M:%S')}] {m}", flush=True)
summary = {"segments": [], "stream_id": "str_smoke" + str(int(time.time()))}
sid = summary["stream_id"]
for i in range(N):
    prompt = P[min(i, len(P) - 1)]
    body = {"stream_id": sid, "segment_index": i, "prompt": prompt, "duration_sec": SECS,
            "resolution": "480p", "aspect_ratio": "16:9", "seed": 42,
            "recache": i == 2, "reset": i == 3}
    t0 = time.time()
    st, seg = call("POST", "/v1/segments", body)
    log(f"POST seg {i} -> {st} {seg.get('id')} {seg.get('status')} warnings={seg.get('warnings')}")
    sid_seg = seg["id"]
    while seg["status"] in ("queued", "running"):
        time.sleep(0.5)
        _, seg = call("GET", f"/v1/segments/{sid_seg}")
    wall = time.time() - t0
    seg["client_wall_s"] = round(wall, 2)
    log(f"seg {i}: {seg['status']} wall={wall:.1f}s frames={seg.get('frames')} blocks={seg.get('blocks')} "
        f"session={seg.get('session')} metrics={seg.get('metrics')} err={seg.get('error')}")
    if seg["status"] == "succeeded":
        dst = f"{OUT}/seg-{i}.mp4"
        urllib.request.urlretrieve(seg["video_url"], dst)
        pr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=codec_name,width,height,nb_read_frames,avg_frame_rate",
                             "-of", "json", dst], capture_output=True, text=True)
        seg["ffprobe"] = json.loads(pr.stdout or "{}")
        seg["bytes"] = os.path.getsize(dst)
    summary["segments"].append(seg)
    _, stream = call("GET", f"/v1/segments/streams/{sid}")
    log(f"stream: {json.dumps(stream)[:300]}")
    summary["stream_after_%d" % i] = stream
_, summary["streams"] = call("GET", "/v1/segments/streams")
st, closed = call("DELETE", f"/v1/segments/streams/{sid}")
summary["delete"] = [st, closed]
json.dump(summary, open(f"{OUT}/segments.json", "w"), indent=1)
log("client done")
EOF
}

pod_cmd() {
  local enc="$1"
  cat <<EOF
set -u
R=/fvscratch/runs; OUT=\$R/segments/$TAG; mkdir -p "\$OUT"
/opt/fastvideo-rs/target/release/fv-gpucheck serve --dir \$R --port 8000 >"\$OUT/http.log" 2>&1 &
exec >"\$OUT/live.log" 2>&1
t0=\$(date +%s); el() { echo \$(( \$(date +%s) - t0 )); }
say() { echo "[\$(el)s] \$*"; }
finish() { say "done"; echo "done $TAG" >"\$OUT/DONE"; exec sleep infinity; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "caps=\$NVIDIA_DRIVER_CAPABILITIES"; cat /opt/fastvideo-rs/bin/fv-serve.features
  ffmpeg -hide_banner -encoders 2>/dev/null | grep -c h264_nvenc; echo "encoder=$enc"; } >"\$OUT/box.txt" 2>&1
say "weights"
W=/fvscratch/weights/sfwan; REPO=wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers
mkdir -p "\$W"
curl -fsSL "https://huggingface.co/api/models/\$REPO/tree/main?recursive=true" \
  | grep -oE '"path":"(model_index.json|scheduler/[^"]+|tokenizer/[^"]+|text_encoder/[^"]+|transformer/[^"]+|vae/[^"]+)"' \
  | sed 's/"path":"//;s/"\$//' >"\$OUT/files.txt"
while read -r f; do mkdir -p "\$W/\$(dirname "\$f")"; echo "\$f"; done <"\$OUT/files.txt" \
  | xargs -P 8 -I{} curl -fsSL --retry 5 -o "\$W/{}" "https://huggingface.co/\$REPO/resolve/main/{}"
du -sh "\$W" >>"\$OUT/box.txt"
bash /opt/fastvideo-rs/scripts/gpu/fetch_taehv.sh /fvscratch/tae >"\$OUT/tae.log" 2>&1 || say "tae fetch failed"
say "fv-serve"
FV_ENGINE=cuda FV_SFWAN_WEIGHTS=\$W FASTVIDEO_TAE_DIR=/fvscratch/tae FV_AUTH_MODE=none FV_BIND=0.0.0.0:8080 \
  FV_PUBLIC_BASE_URL=http://127.0.0.1:8080 FV_SEGMENTS_ENCODER=$enc \
  FV_STATE_DIR=/fvscratch/state FV_JOB_STORE=memory FV_ARTIFACTS=local FV_STREAM_STUN=none FV_OUTPUTS_ALLOW_PRIVATE=1 FV_LOG_FORMAT=text \
  /opt/fastvideo-rs/bin/fv-serve --config /etc/fv/runpod-fake.toml >"\$OUT/serve.log" 2>&1 &
S=http://127.0.0.1:8080
for i in \$(seq 1 900); do curl -fsS \$S/health >/dev/null 2>&1 && break; sleep 2; done
curl -sS \$S/health >"\$OUT/health.json"; say "ready: \$(cat "\$OUT/health.json")"
curl -sS \$S/fv/v1/capabilities >"\$OUT/capabilities.json"
curl -sS \$S/v1/segments/streams >"\$OUT/streams-empty.json"
echo "\$FV_SEG_PY_B64" | base64 -d >/segments.py
say "client"
S=\$S OUT=\$OUT NSEG=$NSEG SEGS=$SEGS python3 /segments.py >"\$OUT/client.log" 2>&1 || say "client failed"
tail -n 20 "\$OUT/client.log"
# Seam check: last frame of seg 0 vs first frame of seg 1, against the mean
# inter-frame difference inside seg 0 (both as ffmpeg psnr MSE of the pair).
if [ -f "\$OUT/seg-0.mp4" ] && [ -f "\$OUT/seg-1.mp4" ]; then
  ffmpeg -v error -sseof -0.07 -i "\$OUT/seg-0.mp4" -frames:v 1 -y "\$OUT/seg0-last.png"
  ffmpeg -v error -i "\$OUT/seg-1.mp4" -frames:v 1 -y "\$OUT/seg1-first.png"
  ffmpeg -v error -i "\$OUT/seg-0.mp4" -frames:v 1 -y "\$OUT/seg0-first.png"
  ffmpeg -v error -i "\$OUT/seg0-last.png" -i "\$OUT/seg1-first.png" -lavfi "psnr=stats_file=-" -f null - 2>/dev/null \
    | tail -1 >"\$OUT/seam-psnr.txt"
  ffmpeg -v error -i "\$OUT/seg-0.mp4" -vf "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" \
    -f null - 2>/dev/null | grep -oE 'YAVG=[0-9.]+' | awk -F= '{s+=\$2;n++} END{if(n) printf "intra_seg0_mean_ydiff=%.3f n=%d\n", s/n, n}' >"\$OUT/seam-intra.txt"
  ffmpeg -v error -i "\$OUT/seg0-last.png" -i "\$OUT/seg1-first.png" -lavfi "blend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" \
    -f null - 2>/dev/null | grep -oE 'YAVG=[0-9.]+' | sed 's/^/seam_ydiff_/' >>"\$OUT/seam-intra.txt"
  # the joined clip, for the control plane's own look
  printf "file '%s'\nfile '%s'\n" "\$OUT/seg-0.mp4" "\$OUT/seg-1.mp4" >"\$OUT/concat.txt"
  ffmpeg -v error -f concat -safe 0 -i "\$OUT/concat.txt" -c copy -y "\$OUT/seg-0-1-joined.mp4" || true
fi
for i in 1 2 3; do [ -f "\$OUT/seg-\$i.mp4" ] && ffmpeg -v error -i "\$OUT/seg-\$i.mp4" -frames:v 1 -y "\$OUT/seg\$i-first.png"; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv >"\$OUT/nvidia-smi.txt" 2>&1
curl -sS \$S/metrics >"\$OUT/metrics.txt" 2>/dev/null
grep -iE 'segment|causal|error|warn' "\$OUT/serve.log" | tail -n 200 >"\$OUT/serve-tail.log"
finish
EOF
}

create() {
  local gpu="$1" enc="$2" payload
  payload="$(jq -n --arg name "fv-segments-$TAG" --arg image "$RUNPOD_IMAGE" --arg gpu "$gpu" \
    --arg cmd "$(pod_cmd "$enc")" --arg py "$(client_py | base64 -w0)" \
    --arg cuda "${RUNPOD_ALLOWED_CUDA-$(fv_runpod_cuda)}" '{
      name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 80, volumeInGb: 0,
      env: {NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video", FV_SEG_PY_B64: $py},
      ports: ["8000/http"], dockerEntrypoint: ["/bin/bash", "-c"], dockerStartCmd: [$cmd]
    } + (if $cuda == "" then {} else {allowedCudaVersions: ($cuda | split(" "))} end)')"
  rest POST /pods "$payload"
}

b="$(balance)"; [[ -n "$b" ]] || { log "no balance"; exit 1; }
awk -v b="$b" -v m="$MIN_BALANCE" -v d="$MAX_DPH" -v c="$CAP_S" 'BEGIN{exit !(b - d*c/3600 >= m)}' \
  || { log "balance \$$b would drop below \$$MIN_BALANCE at the cap"; exit 1; }
log "balance \$$b"
id=""
IFS='|' read -r -a gpu_list <<<"$GPUS"
for g in "${gpu_list[@]}"; do
  fv_runpod_price_ok "$g" "$MAX_DPH" SECURE || continue
  enc=nvenc; [[ "$g" == *H100* ]] && enc=libx264
  log "create pod on $g (segments encoder $enc)"
  if resp="$(create "$g" "$enc" 2>&1)"; then
    id="$(jq -r '.id // empty' <<<"$resp")"
    dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    [[ -n "$id" ]] && break
  fi
  log "no pod on $g: ${resp:0:200}"
done
[[ -n "$id" ]] || { log "no pod"; exit 1; }
ledger "pod-created $id fv-segments-$TAG gpu=$g dph=$dph image=$RUNPOD_IMAGE"
cleanup() {
  log "delete pod $id"
  if rest DELETE "/pods/$id" >/dev/null; then ledger "pod-deleted $id"; else log "WARN: delete failed"; ledger "pod-delete-failed $id"; fi
}
trap cleanup EXIT
# Detached backstop: deletes the pod at the cap even if this shell dies.
# shellcheck disable=SC2016
setsid nohup env POD_ID="$id" CAP="$CAP_S" API="$API" bash -c \
  'sleep "$CAP"; curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "$API/pods/$POD_ID" >/dev/null' \
  >/dev/null 2>&1 < /dev/null &
if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then log "pod at \$$dph/hr exceeds \$$MAX_DPH"; exit 1; fi
log "pod $id on $g \$$dph/hr, cap ${CAP_S}s"
echo "$id $g $dph" >"$OUT/pod.txt"
t0=$(date +%s); last=""
until [[ "$(proxy "$id" "segments/$TAG/DONE" 2>/dev/null | head -1)" == "done $TAG" ]]; do
  (( $(date +%s) - t0 < CAP_S )) || { log "cap reached"; break; }
  cur="$(proxy "$id" "segments/$TAG/live.log" 2>/dev/null | tail -1 || true)"
  [[ -n "$cur" && "$cur" != "$last" ]] && { log "pod: $cur"; last="$cur"; }
  sleep 20
done
for f in live.log box.txt files.txt tae.log health.json capabilities.json streams-empty.json client.log segments.json \
  seam-psnr.txt seam-intra.txt nvidia-smi.txt metrics.txt serve-tail.log serve.log \
  seg-0.mp4 seg-1.mp4 seg-2.mp4 seg-3.mp4 seg-0-1-joined.mp4 \
  seg0-first.png seg0-last.png seg1-first.png seg2-first.png seg3-first.png; do
  curl -sS --max-time 300 --fail "https://$id-8000.proxy.runpod.net/segments/$TAG/$f" -o "$OUT/$f" 2>/dev/null || rm -f "$OUT/$f"
done
echo "elapsed_s=$(( $(date +%s) - t0 ))" >>"$OUT/pod.txt"
log "results → $OUT"
