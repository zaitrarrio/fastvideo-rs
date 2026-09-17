#!/usr/bin/env bash
# Build cudarc CUDA and run a tiny / 1.3B GPU generate on the Vast box.
set -euo pipefail

INSTANCE_ID="${VAST_INSTANCE_ID:-50416610}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/fastvideo-rs}"
CARGO_HOME_DIR="${VAST_CARGO_HOME:-/workspace/.cargo}"
MODE="${1:-tiny}"

url="$(vastai ssh-url "$INSTANCE_ID")"
hostport="${url#ssh://root@}"
host="${hostport%:*}"
port="${hostport##*:}"

ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" bash -s -- "$REMOTE_DIR" "$MODE" "$CARGO_HOME_DIR" <<'EOF'
set -euo pipefail
REMOTE_DIR="$1"
MODE="$2"
export CARGO_HOME="$3"
mkdir -p "$CARGO_HOME"
# shellcheck disable=SC1091
if [ -f "$HOME/.cargo/env" ]; then
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
cd "$REMOTE_DIR"

GPU_SLUG="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="/workspace/artifacts/${STAMP}-${GPU_SLUG}-generate-${MODE}"
mkdir -p "$RUN_DIR"

echo "building fastvideo-cli --features cuda-cudarc (release)"
cargo build -p fastvideo-cli --release --features cuda-cudarc

if [ "$MODE" = "tiny" ]; then
  ./target/release/fastvideo generate \
    --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --backend cudarc \
    --tiny \
    --device cuda \
    --output "$RUN_DIR/tiny" \
    --prompt "A curious raccoon in a field of sunflowers."
  echo "artifacts $RUN_DIR"
  ls -la "$RUN_DIR/tiny" | head
  exit 0
fi

WEIGHTS="${WEIGHTS_DIR:-/workspace/weights/Wan2.1-T2V-1.3B-Diffusers}"
if [ ! -d "$WEIGHTS/transformer" ]; then
  echo "downloading Wan-AI/Wan2.1-T2V-1.3B-Diffusers to $WEIGHTS"
  mkdir -p "$WEIGHTS"
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$WEIGHTS"
fi

SMOKE_FLAGS=()
if [ "$MODE" = "1.3b-smoke" ]; then
  SMOKE_FLAGS=(--frames 9 --steps 2 --height 256 --width 256 --guidance 1.0)
fi

./target/release/fastvideo generate \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --backend cudarc \
  --device cuda \
  --weights "$WEIGHTS" \
  --output "$RUN_DIR/generate" \
  --prompt "A curious raccoon in a field of sunflowers." \
  "${SMOKE_FLAGS[@]}"
echo "artifacts $RUN_DIR"
ls -la "$RUN_DIR/generate" | head
EOF
