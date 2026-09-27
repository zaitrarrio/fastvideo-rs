#!/usr/bin/env bash
# GPU acceptance for WP-15 (docs/serve/design.md §5.4, §5.8): one Runpod pod
# runs fv-serve with the SF-Wan causal backend (`engine.backend = cuda`,
# FV_SFWAN_WEIGHTS) and MediaMTX; `POST /fv/v1/streams` publishes the
# SF-Wan stream with WHIP (NVENC) to MediaMTX on the same pod; an RTSP
# reader (ffmpeg) records what MediaMTX serves, ffprobe counts the frames;
# the stream's status (TTFF phases, pacer stats) is collected before and
# after a prompt switch. Weight-free volume-wise: SF-Wan is fetched from
# Hugging Face onto the container disk (nothing is written to a volume).
#
#   RUNPOD_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:… \
#     bash scripts/serve/runpod-sfwan-whip.sh
#
# The image must be the `serve` target built with the `webrtc` feature
# (serve-image workflow, input features=cuda,http-client,webrtc).
# Env: RUNPOD_API_KEY, RUNPOD_GPU_TYPES (space-separated preference list,
# default L40S then RTX PRO 6000; both have NVENC, H100 has none),
# RUNPOD_GPU_MAX_DPH (default 2.5), FV_POD_CAP_S (default 3600),
# FV_STREAM_SECONDS (default 90). Results: artifacts/serve/sfwan-whip/<tag>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
API="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing}"
: "${RUNPOD_IMAGE:?RUNPOD_IMAGE (the fv-serve image with webrtc) missing}"
GPUS="${RUNPOD_GPU_TYPES:-NVIDIA L40S|NVIDIA RTX PRO 6000 Blackwell Server Edition}"
MAX_DPH="${RUNPOD_GPU_MAX_DPH:-2.5}"
CAP_S="${FV_POD_CAP_S:-3600}"
SECS="${FV_STREAM_SECONDS:-90}"
TAG="$(date -u +%m%d%H%M)"
OUT="$ROOT/artifacts/serve/sfwan-whip/$TAG"
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
rest() {
  curl -sS --fail-with-body -X "$1" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'content-type: application/json' ${3:+-d "$3"} "$API$2"
}
proxy() { curl -sS --max-time 30 --fail "https://$1-8000.proxy.runpod.net/$2"; }

pod_cmd() {
  cat <<EOF
set -u
R=/fvscratch/runs; OUT=\$R/sfwhip/$TAG; mkdir -p "\$OUT"
/opt/fastvideo-rs/target/release/fv-gpucheck serve --dir \$R --port 8000 >"\$OUT/http.log" 2>&1 &
exec >"\$OUT/live.log" 2>&1
t0=\$(date +%s); el() { echo \$(( \$(date +%s) - t0 )); }
say() { echo "[\$(el)s] \$*"; }
finish() { say "done"; echo "done $TAG" >"\$OUT/DONE"; exec sleep infinity; }
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "caps=\$NVIDIA_DRIVER_CAPABILITIES"; cat /opt/fastvideo-rs/bin/fv-serve.features
  ffmpeg -hide_banner -encoders 2>/dev/null | grep -c h264_nvenc; } >"\$OUT/box.txt" 2>&1
drv=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "\${drv:-0}" -ge 580 ] || { say "driver \$drv too old"; finish; }
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
say "mediamtx"
mkdir -p /fvscratch/mtx && curl -fsSL https://github.com/bluenviron/mediamtx/releases/download/v1.15.1/mediamtx_v1.15.1_linux_amd64.tar.gz \
  | tar xz -C /fvscratch/mtx
( cd /fvscratch/mtx && MTX_RTMP=no MTX_HLS=no MTX_SRT=no MTX_API=yes \
    MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 exec ./mediamtx mediamtx.yml ) >"\$OUT/mediamtx.log" 2>&1 &
say "fv-serve"
FV_ENGINE=cuda FV_SFWAN_WEIGHTS=\$W FASTVIDEO_TAE_DIR=/fvscratch/tae FV_AUTH_MODE=none FV_BIND=0.0.0.0:8080 \
  FV_STATE_DIR=/fvscratch/state FV_JOB_STORE=memory FV_ARTIFACTS=local FV_STREAM_STUN=none FV_LOG_FORMAT=text \
  /opt/fastvideo-rs/bin/fv-serve --config /etc/fv/runpod-fake.toml >"\$OUT/serve.log" 2>&1 &
S=http://127.0.0.1:8080
for i in \$(seq 1 900); do curl -fsS \$S/health >/dev/null 2>&1 && break; sleep 2; done
curl -sS \$S/health >"\$OUT/health.json"; say "ready: \$(cat "\$OUT/health.json")"
curl -sS \$S/fv/v1/capabilities >"\$OUT/capabilities.json"
P1="A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed"
P2="A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed"
curl -sS -X POST \$S/fv/v1/streams -H 'content-type: application/json' \
  -d "{\"model\":\"sf-wan\",\"whip_url\":\"http://127.0.0.1:8889/sfwan/whip\",\"prompt\":\"\$P1\",\"width\":832,\"height\":480,\"max_seconds\":$SECS}" \
  >"\$OUT/create.json"
ID=\$(grep -oE '"id":"fvstream_[^"]+"' "\$OUT/create.json" | head -1 | sed 's/"id":"//;s/"//')
say "stream \$ID"
[ -n "\$ID" ] || finish
for i in \$(seq 1 300); do
  curl -sS \$S/fv/v1/streams/\$ID >"\$OUT/status-now.json"
  grep -q '"state":"streaming"' "\$OUT/status-now.json" && break
  grep -q '"state":"closed"' "\$OUT/status-now.json" && break
  sleep 1
done
cp "\$OUT/status-now.json" "\$OUT/status-start.json"; say "streaming"
sleep 3
timeout 60 ffmpeg -hide_banner -rtsp_transport tcp -i rtsp://127.0.0.1:8554/sfwan -t 30 -c copy -y "\$OUT/rtsp-30s.mp4" >"\$OUT/ffmpeg.log" 2>&1 &
FF=\$!
sleep 12
curl -sS \$S/fv/v1/streams/\$ID >"\$OUT/status-before-switch.json"
curl -sS -X POST \$S/fv/v1/streams/\$ID/commands -H 'content-type: application/json' \
  -d "{\"type\":\"set_prompt\",\"data\":{\"prompt\":\"\$P2\"}}" >"\$OUT/switch.json"
say "prompt switched"
sleep 15
curl -sS \$S/fv/v1/streams/\$ID >"\$OUT/status-after-switch.json"
curl -sS http://127.0.0.1:9997/v3/paths/list >"\$OUT/mediamtx-paths.json"
nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.encoder --format=csv >"\$OUT/nvidia-smi.txt" 2>&1
wait \$FF
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,profile,width,height,nb_read_frames,avg_frame_rate,r_frame_rate \
  -of json "\$OUT/rtsp-30s.mp4" >"\$OUT/ffprobe.json" 2>&1
ffmpeg -hide_banner -v error -i "\$OUT/rtsp-30s.mp4" -vf "select=not(mod(n\,60))" -vsync vfr "\$OUT/frame-%02d.jpg" 2>>"\$OUT/ffmpeg.log"
curl -sS -X DELETE \$S/fv/v1/streams/\$ID >"\$OUT/status-final.json"
curl -sS \$S/metrics >"\$OUT/metrics.txt" 2>/dev/null
finish
EOF
}

create() {
  local gpu="$1" payload
  payload="$(jq -n --arg name "fv-sfwan-whip-$TAG" --arg image "$RUNPOD_IMAGE" --arg gpu "$gpu" \
    --arg cmd "$(pod_cmd)" '{
      name: $name, imageName: $image, cloudType: "SECURE", computeType: "GPU",
      gpuTypeIds: [$gpu], gpuCount: 1, containerDiskInGb: 80, volumeInGb: 0,
      allowedCudaVersions: ["13.0"],
      env: {NVIDIA_DRIVER_CAPABILITIES: "compute,utility,video"},
      ports: ["8000/http"], dockerStartCmd: ["/bin/bash", "-c", $cmd]
    }')"
  rest POST /pods "$payload"
}

id=""
IFS='|' read -r -a gpu_list <<<"$GPUS"
for g in "${gpu_list[@]}"; do
  log "create pod on $g"
  if resp="$(create "$g" 2>&1)"; then
    id="$(jq -r '.id // empty' <<<"$resp")"
    dph="$(jq -r '.costPerHr // 0' <<<"$resp")"
    [[ -n "$id" ]] && break
  fi
  log "no pod on $g: ${resp:0:200}"
done
[[ -n "$id" ]] || { log "no pod"; exit 1; }
cleanup() { log "delete pod $id"; rest DELETE "/pods/$id" >/dev/null || log "WARN: delete failed"; }
trap cleanup EXIT
if awk -v p="$dph" -v c="$MAX_DPH" 'BEGIN{exit !(p+0 > c+0)}'; then log "pod at \$$dph/hr exceeds \$$MAX_DPH"; exit 1; fi
log "pod $id \$$dph/hr, cap ${CAP_S}s"
echo "$id $dph" >"$OUT/pod.txt"
t0=$(date +%s); last=""
until [[ "$(proxy "$id" "sfwhip/$TAG/DONE" 2>/dev/null | head -1)" == "done $TAG" ]]; do
  (( $(date +%s) - t0 < CAP_S )) || { log "cap reached"; break; }
  cur="$(proxy "$id" "sfwhip/$TAG/live.log" 2>/dev/null | tail -1 || true)"
  [[ -n "$cur" && "$cur" != "$last" ]] && { log "pod: $cur"; last="$cur"; }
  sleep 20
done
for f in live.log box.txt files.txt tae.log mediamtx.log serve.log health.json capabilities.json create.json \
  status-start.json status-before-switch.json switch.json status-after-switch.json status-final.json \
  mediamtx-paths.json nvidia-smi.txt ffmpeg.log ffprobe.json metrics.txt rtsp-30s.mp4 \
  frame-01.jpg frame-02.jpg frame-03.jpg frame-04.jpg frame-05.jpg frame-06.jpg frame-07.jpg frame-08.jpg; do
  curl -sS --max-time 300 --fail "https://$id-8000.proxy.runpod.net/sfwhip/$TAG/$f" -o "$OUT/$f" 2>/dev/null || rm -f "$OUT/$f"
done
echo "elapsed_s=$(( $(date +%s) - t0 ))" >>"$OUT/pod.txt"
log "results → $OUT"
