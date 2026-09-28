#!/usr/bin/env bash
# On-pod helper for the E2E re-verification (docs/serve/e2e/reverify.md), run
# through the pod.sh sidecar after `pod.sh bundle`. One pod, fv-serve
# restarted per config (/e2e/serve.sh from pod-boot.sh).
#
#   reverify-pod.sh cfgs          write /e2e/cfg-{h3max,h3draft,ltx,sfwan}.toml
#   reverify-pod.sh switch <cfg>  restart fv-serve on /e2e/cfg-<cfg>.toml (or
#                                 fv for the boot config); print seconds to /ping 200
#   reverify-pod.sh mtx           download and start MediaMTX (WHIP :8889, RTSP :8554)
#   reverify-pod.sh box           GPU, driver, weight dirs on the volume
set -uo pipefail
E=/e2e
WEBRTC=$'\n[webrtc]\nudp_port = 40010\ntcp_port = 40000\npublic_ip = "127.0.0.1"\n'
state() { sed 's#^state_dir = .*#state_dir = "/fvstate"#'; }

case "${1:-}" in
  cfgs)
    { state </etc/fv/runpod-h3-max.toml; printf '%s' "$WEBRTC"; } >$E/cfg-h3max.toml
    # h3-draft alone (a second H3 DiT does not fit next to turbo on 80 GB).
    state <$E/fv.toml | sed -e 's/^recipe = "h3-turbo"/recipe = "h3-draft"/' >$E/cfg-h3draft.toml
    { state </etc/fv/runpod-ltx.toml; printf '%s' "$WEBRTC"; } >$E/cfg-ltx.toml
    cat >$E/cfg-sfwan.toml <<EOF
[server]
bind = "0.0.0.0:8000"
state_dir = "/fvstate"
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
$WEBRTC
EOF
    grep -H -E '^(recipe|id|fal_apps|state_dir)' $E/cfg-*.toml
    ;;
  switch)
    cfg=$E/cfg-$2.toml; [ "$2" = fv ] && cfg=$E/fv.toml
    t0=$(date +%s)
    FV_STREAM_STUN=none $E/serve.sh "$cfg"
    sleep 3
    for _ in $(seq 1 450); do
      c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/ping)
      [ "$c" = 200 ] && { echo "ready_s=$(( $(date +%s) - t0 )) cfg=$cfg"; exit 0; }
      kill -0 "$(cat $E/serve.pid)" 2>/dev/null || { echo "fv-serve exited"; tail -30 $E/serve.log | grep -v -i token; exit 1; }
      sleep 2
    done
    echo "not ready after 900 s"; exit 1
    ;;
  mtx)
    mkdir -p $E/mtx
    [ -x $E/mtx/mediamtx ] || curl -fsSL https://github.com/bluenviron/mediamtx/releases/download/v1.15.1/mediamtx_v1.15.1_linux_amd64.tar.gz | tar xz -C $E/mtx
    ( cd $E/mtx && MTX_RTMP=no MTX_HLS=no MTX_SRT=no MTX_API=yes MTX_WEBRTCADDITIONALHOSTS=127.0.0.1 \
        setsid nohup ./mediamtx mediamtx.yml >$E/mediamtx.log 2>&1 & )
    sleep 2; tail -5 $E/mediamtx.log
    ;;
  box)
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "features=$(cat /opt/fastvideo-rs/bin/fv-serve.features 2>/dev/null)"
    echo "h264_nvenc_listed=$(ffmpeg -hide_banner -encoders 2>/dev/null | grep -c h264_nvenc)"
    for d in h3-base ltx25 sfwan21-1.3b auxiliary/tae; do echo "$d: $(ls /workspace/weights/$d 2>&1 | head -12 | tr '\n' ' ')"; done
    nproc; free -g | head -2
    ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
