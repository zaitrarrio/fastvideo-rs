# In-pod bootstrap for the WP-18 E2E pod (scripts/serve/e2e/pod.sh passes this
# file as `bash -c` start command; not run locally).
#
# 1. /e2e/serve.sh [config] (re)starts fv-serve in the background, logging to
#    /e2e/serve.log. The pod's Runpod identity is kept (public base URL,
#    worker id) but RUNPOD_POD_ID / RUNPOD_PUBLIC_IP are dropped from its
#    environment so the WebRTC host binds UDP 40010 + ICE-TCP 40000 and
#    advertises 127.0.0.1: the streaming clients (headless Chromium,
#    reactor_sdk) run on the pod itself.
# 2. Python 3 from apt, then the sidecar (scripts/serve/e2e/sidecar.py, from
#    FV_SIDECAR_B64) on :8001 in the foreground.
set -u
mkdir -p /e2e
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; nproc; } > /e2e/box.txt 2>&1
printf '%s' "$FV_SIDECAR_B64" | base64 -d > /e2e/sidecar.py
{ cat "${FV_E2E_BASE_CONFIG:-/etc/fv/runpod.toml}"; printf '\n[webrtc]\nudp_port = 40010\ntcp_port = 40000\npublic_ip = "127.0.0.1"\n'; } > /e2e/fv.toml
sed 's/^post_encoder = "auto"/post_encoder = "cpu-test-x264"/' /e2e/fv.toml > /e2e/fv-x264.toml
cat > /e2e/serve.sh <<'EOF'
#!/bin/bash
cfg="${1:-/e2e/fv.toml}"
if [ -s /e2e/serve.pid ]; then
  p=$(cat /e2e/serve.pid)
  kill -TERM "$p" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  kill -KILL "$p" 2>/dev/null
fi
echo "=== $(date -u +%FT%TZ) start $cfg" >> /e2e/serve.log
env -u RUNPOD_POD_ID -u RUNPOD_PUBLIC_IP \
  FV_PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net" FV_WORKER_ID="${RUNPOD_POD_ID}" \
  /opt/fastvideo-rs/bin/fv-serve --config "$cfg" >> /e2e/serve.log 2>&1 &
echo $! > /e2e/serve.pid
EOF
chmod +x /e2e/serve.sh
/e2e/serve.sh /e2e/fv.toml
date -u +%s > /e2e/boot.t0
export DEBIAN_FRONTEND=noninteractive
{ apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip xz-utils; } > /e2e/apt.log 2>&1
exec python3 /e2e/sidecar.py
