#!/usr/bin/env bash
# Runs ON the E2E pod (started by wan-pod.sh; WP-18, design §7.6). A fixed
# script, no remote control: results are published read-only on port 8001
# (fv-gpucheck serve) under /fvscratch/runs/e2e/.
#
# Phase 1 (FastWan batch): fv-serve with /etc/fv/runpod-wan.toml (+ a fal
#   app for the model) on :8000; the driver (wan_batch.py, outside) calls the
#   public APIs, then signals the end of the phase with one
#   `DELETE /fv/v1/streams/e2e-phase1-done` (seen in /metrics), or the
#   phase times out (FV_E2E_PHASE1_MAX_S after ready).
# Phase 2 (SF-Wan live): MediaMTX + fv-serve serving `sfwan21-1.3b`;
#   a native /fv/v1/streams WHIP publish for FV_E2E_LIVE_S seconds with a
#   WHEP viewer (aiortc), an RTSP recorder, prompt switches every 60 s and
#   GPU memory sampling; then reactor_sdk 1.6.0 causal mode; then the raw
#   rollout throughput (fv-gpucheck wan stream, 60 s).
set -u
R=/fvscratch/runs/e2e
E=/fvscratch/e2e
mkdir -p "$R" "$E" /fvscratch/state1 /fvscratch/state2
/opt/fastvideo-rs/target/release/fv-gpucheck serve --dir /fvscratch/runs --port 8001 >/fvscratch/files.log 2>&1 &
exec >"$R/live.log" 2>&1
T0=$(date +%s)
say() { echo "[$(( $(date +%s) - T0 ))s $(date -u +%H:%M:%S)] $*"; }
phase() { echo "$1 $(date +%s.%N)" >>"$R/phases.txt"; say "phase $1"; }
finish() { phase "done"; echo "done ${FV_E2E_TAG:-}" >"$R/DONE"; exec sleep infinity; }
W=/workspace/weights
BIN=/opt/fastvideo-rs/bin/fv-serve
S=http://127.0.0.1:8000
P1="A drone shot gliding over a winding river through an autumn forest, golden afternoon light, slow steady forward camera motion, highly detailed"
P2="A drone shot gliding over snowy mountain peaks at dawn, pink sky, slow steady forward camera motion, highly detailed"
P3="A drone shot flying low over a turquoise tropical lagoon with white sand, bright midday sun, slow steady forward camera motion, highly detailed"

echo "$FV_E2E_PY_B64" | base64 -d | tar xz -C "$E"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "features=$(cat /opt/fastvideo-rs/bin/fv-serve.features)"
  echo "h264_nvenc_listed=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c h264_nvenc)"
  echo "public_ip_set=$([ -n "${RUNPOD_PUBLIC_IP:-}" ] && echo yes || echo no) tcp70000=${RUNPOD_TCP_PORT_70000:-none}"
  for d in fastwan21-1.3b sfwan21-1.3b auxiliary/tae; do echo "$d: $(ls "$W/$d" 2>&1 | tr '\n' ' ')"; done
  nproc; free -g | head -2; df -h /fvscratch | tail -1; } >"$R/box.txt" 2>&1

# The image needs a CUDA 13 driver (>= 580); the create call may run
# without the allowedCudaVersions filter when stock is short.
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "${drv:-0}" -ge 580 ] || { say "driver ${drv:-none} is older than 580"; finish; }

# A pod-local key for the in-pod clients (hash appended to FV_API_KEYS).
LKEY="fvk-$(openssl rand -hex 16)"
LHASH="$(printf '%s' "$LKEY" | sha256sum | cut -d' ' -f1)"
export FV_API_KEYS="${FV_API_KEYS},${LHASH}"
LAUTH="Authorization: Bearer $LKEY"
# fv-serve prints a generated admin token to its log, and the logs are
# published on :8001: choose one here instead (never written anywhere).
FV_ADMIN_TOKEN="$(openssl rand -hex 24)"
export FV_ADMIN_TOKEN

# Client setup in the background (python 3.12 venv, MediaMTX).
(
  set -e
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/fvscratch/uv sh
  /fvscratch/uv/uv venv -q -p 3.12 /fvscratch/venv
  VIRTUAL_ENV=/fvscratch/venv /fvscratch/uv/uv pip install -q reactor-sdk==1.6.0 aiortc aiohttp numpy
  mkdir -p /fvscratch/mtx
  curl -fsSL https://github.com/bluenviron/mediamtx/releases/download/v1.15.1/mediamtx_v1.15.1_linux_amd64.tar.gz | tar xz -C /fvscratch/mtx
  echo ok >/fvscratch/setup.ok
) >"$R/setup.log" 2>&1 &

gpu_sampler() { nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l "${2:-5}" >"$R/$1" 2>&1 & echo $!; }
wait_ready() { # $1 label -> seconds to /health 200
  local t
  t=$(date +%s)
  for _ in $(seq 1 600); do
    curl -fsS "$S/health" >/dev/null 2>&1 && { echo $(( $(date +%s) - t )); return 0; }
    kill -0 "$SERVE_PID" 2>/dev/null || { say "fv-serve ($1) exited"; return 1; }
    sleep 2
  done
  return 1
}
stop_serve() {
  kill "$SERVE_PID" 2>/dev/null
  for _ in $(seq 1 40); do kill -0 "$SERVE_PID" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$SERVE_PID" 2>/dev/null; sleep 2
}

# ---------------------------------------------------------------- phase 1
if [ "${FV_E2E_SKIP_PHASE1:-0}" != 1 ]; then
phase p1-start
sed -e 's#^state_dir = .*#state_dir = "/fvscratch/state1"#' \
    -e 's#^\[protocols\]#[protocols]\nfal_apps = ["fastvideo/fastwan21-1.3b"]#' \
    /etc/fv/runpod-wan.toml >"$E/wan.toml"
cp "$E/wan.toml" "$R/wan.toml"
GS=$(gpu_sampler phase1-gpu.csv 5)
FV_WEIGHTS=$W FV_STATE_DIR=/fvscratch/state1 "$BIN" --config "$E/wan.toml" >"$R/serve-wan.log" 2>&1 &
SERVE_PID=$!
if L=$(wait_ready wan); then
  echo "$L" >"$R/phase1-ready-s.txt"; phase p1-ready
  curl -sS "$S/health" >"$R/phase1-health.json"
  curl -sS -H "$LAUTH" -H 'content-type: application/json' -d '{"kind":"info","nvenc":true}' "$S/fv/v1/forward" >"$R/phase1-info.json"
  dl=$(( $(date +%s) + ${FV_E2E_PHASE1_MAX_S:-1500} ))
  while (( $(date +%s) < dl )); do
    curl -sS "$S/metrics" 2>/dev/null | grep 'fv_http_requests_total' | grep 'method="DELETE"' | grep -q 'route="/fv/v1/streams/{id}"' && break
    sleep 3
  done
  phase p1-end
  curl -sS "$S/metrics" >"$R/phase1-metrics.txt"
else
  say "phase 1: fv-serve not ready"; phase p1-failed
fi
stop_serve; kill "$GS" 2>/dev/null
fi

# ---------------------------------------------------------------- phase 2
phase p2-start
for _ in $(seq 1 300); do [ -f /fvscratch/setup.ok ] && break; sleep 2; done
[ -f /fvscratch/setup.ok ] || { say "client setup failed"; finish; }
( cd /fvscratch/mtx && MTX_RTMP=no MTX_HLS=no MTX_SRT=no MTX_API=yes MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 \
    exec ./mediamtx mediamtx.yml ) >"$R/mediamtx.log" 2>&1 &
sfwan_cfg() { # $1 public_ip
  cat <<EOF
[server]
bind = "0.0.0.0:8000"
state_dir = "/fvscratch/state2"
shutdown_grace_s = 5

[auth]
mode = "keys"

[artifacts]
backend = "auto"

[jobs]
backend = "auto"

[engine]
backend = "cuda"
post_encoder = "auto"

[[models]]
id = "sf-wan"
family = "wan"
recipe = "sfwan21-1.3b"
weights = "\${FV_WEIGHTS}/sfwan21-1.3b"
resident = true

[protocols]
openai_videos = false
fastwan = false
minimax = false
fal = false
fal_director = false
ltx = false
reactor = true
native = true

[reactor]
model = "sf-wan"

[webrtc]
public_ip = "$1"
EOF
}
start_sfwan() { # $1 public_ip, $2 log
  sfwan_cfg "$1" >"$E/sfwan.toml"; cp "$E/sfwan.toml" "$R/sfwan-$2.toml"
  FV_WEIGHTS=$W FV_STATE_DIR=/fvscratch/state2 FV_STREAM_STUN=none "$BIN" --config "$E/sfwan.toml" >"$R/serve-$2.log" 2>&1 &
  SERVE_PID=$!
}
start_sfwan auto sfwan
GS=$(gpu_sampler phase2-gpu.csv 5)
L=$(wait_ready sfwan) || { say "phase 2: fv-serve not ready"; finish; }
echo "$L" >"$R/phase2-ready-s.txt"; phase p2-ready
curl -sS -H "$LAUTH" "$S/fv/v1/capabilities" >"$R/phase2-capabilities.json"

# ---- native WHIP, FV_E2E_LIVE_S of video ----
LIVE=${FV_E2E_LIVE_S:-300}
echo "{\"t\":$(date +%s.%N)}" >"$R/live-post.json"
curl -sS -H "$LAUTH" -H 'content-type: application/json' -X POST "$S/fv/v1/streams" \
  -d "{\"model\":\"sf-wan\",\"whip_url\":\"http://127.0.0.1:8889/sfwan/whip\",\"prompt\":\"$P1\",\"width\":832,\"height\":480,\"max_seconds\":$((LIVE + 30))}" \
  >"$R/live-create.json"
ID=$(grep -oE '"id":"fvstream_[^"]+"' "$R/live-create.json" | head -1 | sed 's/"id":"//;s/"//')
say "stream $ID"
if [ -n "$ID" ]; then
  for _ in $(seq 1 600); do
    curl -sS -H "$LAUTH" "$S/fv/v1/streams/$ID" >"$R/live-status-now.json"
    grep -qE '"state":"(streaming|closed)"' "$R/live-status-now.json" && break
    sleep 0.1
  done
  echo "{\"t\":$(date +%s.%N)}" >"$R/live-streaming.json"
  cp "$R/live-status-now.json" "$R/live-status-start.json"; phase p2-live-streaming
  sleep 1
  VIEW_T0=$(date +%s.%N)
  /fvscratch/venv/bin/python "$E/whep_viewer.py" --url http://127.0.0.1:8889/sfwan/whep --seconds "$LIVE" \
    --out "$R/viewer-frames.jsonl" >"$R/viewer.json" 2>"$R/viewer.log" &
  VIEW=$!
  timeout $((LIVE + 30)) ffmpeg -hide_banner -loglevel error -rtsp_transport tcp -i rtsp://127.0.0.1:8554/sfwan \
    -t "$LIVE" -c copy -y "$R/rtsp.mkv" >"$R/rtsp.log" 2>&1 &
  REC=$!
  ST0=$(date +%s)
  n=0
  : >"$R/live-switches.jsonl"; : >"$R/live-status.jsonl"
  while (( $(date +%s) - ST0 < LIVE )); do
    el=$(( $(date +%s) - ST0 ))
    echo "{\"t\":$(date +%s.%N),\"s\":$(curl -sS -H "$LAUTH" "$S/fv/v1/streams/$ID")}" >>"$R/live-status.jsonl"
    want=$(( el / 60 ))
    if (( want > n && want <= 4 )); then
      n=$want
      case $n in 1) P=$P2 ;; 2) P=$P3 ;; 3) P=$P1 ;; *) P=$P2 ;; esac
      ts=$(date +%s.%N)
      rep=$(curl -sS -H "$LAUTH" -H 'content-type: application/json' -X POST "$S/fv/v1/streams/$ID/commands" \
        -d "{\"type\":\"set_prompt\",\"data\":{\"prompt\":\"$P\"}}")
      te=$(date +%s.%N)
      echo "{\"n\":$n,\"t\":$ts,\"t_reply\":$te,\"prompt\":\"${P:0:40}\",\"reply\":$rep}" >>"$R/live-switches.jsonl"
      say "switch $n"
    fi
    sleep 5
  done
  wait "$REC"; wait "$VIEW"
  curl -sS -H "$LAUTH" "$S/fv/v1/streams/$ID" >"$R/live-status-end.json"
  curl -sS -H "$LAUTH" -X DELETE "$S/fv/v1/streams/$ID" >"$R/live-status-final.json"
  curl -sS http://127.0.0.1:9997/v3/paths/list >"$R/mediamtx-paths.json"
  ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 "$R/rtsp.mkv" >"$R/rtsp-packets.csv" 2>&1
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,profile,level,width,height,avg_frame_rate,r_frame_rate:format=duration,size,bit_rate \
    -of json "$R/rtsp.mkv" >"$R/rtsp-ffprobe.json" 2>&1
  # A 16 s sample around the first switch (video time ~= wall time since the recorder started).
  ffmpeg -hide_banner -loglevel error -ss 52 -i "$R/rtsp.mkv" -t 16 -c:v libx264 -preset veryfast -crf 30 -an -y "$R/sample-switch.mp4" 2>>"$R/rtsp.log"
  echo "{\"viewer_t0\":$VIEW_T0,\"rec_started\":$ST0}" >"$R/live-clock.json"
  phase p2-live-end
fi
sleep 3

# ---- Reactor causal (reactor_sdk 1.6.0) ----
curl -sS "$S/session" >"$R/reactor-session-before.json" 2>&1
timeout 240 /fvscratch/venv/bin/python "$E/reactor_causal.py" --url "$S" --out "$R/reactor.json" --steady 30 >"$R/reactor.log" 2>&1
RC=$?
curl -sS "$S/session" >"$R/reactor-session-after.json" 2>&1
phase "p2-reactor-rc$RC"
if [ "$RC" != 0 ]; then
  # Retry with the ICE-TCP candidate on loopback (in case the pod cannot
  # reach its own public address).
  stop_serve
  start_sfwan 127.0.0.1 sfwan-loopback
  if wait_ready sfwan-loopback >"$R/phase2b-ready-s.txt"; then
    timeout 240 /fvscratch/venv/bin/python "$E/reactor_causal.py" --url "$S" --out "$R/reactor-loopback.json" --steady 30 >"$R/reactor-loopback.log" 2>&1
    phase "p2-reactor-loopback-rc$?"
  fi
fi
curl -sS "$S/metrics" >"$R/phase2-metrics.txt"
stop_serve; kill "$GS" 2>/dev/null

# ---- raw rollout throughput (no serving stack) ----
phase p2-gpucheck
FASTVIDEO_TAE_DIR=$W/auxiliary/tae timeout 420 /opt/fastvideo-rs/target/release/fv-gpucheck --out /fvscratch/gpucheck --mode fast \
  wan stream --weights "$W/sfwan21-1.3b" --prompt "$P1" --run g60,seconds=60,rope=rebased,sink=3 >"$R/gpucheck.log" 2>&1
echo "rc=$?" >>"$R/gpucheck.log"
find /fvscratch/gpucheck -maxdepth 2 -type f -printf '%s %p\n' >"$R/gpucheck-files.txt" 2>&1
mkdir -p "$R/gpucheck"; cp /fvscratch/gpucheck/*.json "$R/gpucheck/" 2>/dev/null
finish
