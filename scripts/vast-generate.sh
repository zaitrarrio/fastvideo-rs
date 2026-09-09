#!/usr/bin/env bash
# Build Candle CUDA and run a tiny GPU generate on the Vast box.
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

echo "building fastvideo-cli --features cuda (release)"
cargo build -p fastvideo-cli --release --features cuda

if [ "$MODE" = "tiny" ]; then
  ./target/release/fastvideo generate \
    --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --backend candle \
    --tiny \
    --device cuda \
    --dtype f32 \
    --output /workspace/fastvideo-tiny \
    --prompt "A curious raccoon in a field of sunflowers."
  ls -la /workspace/fastvideo-tiny | head
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
  --backend candle \
  --device cuda \
  --dtype bf16 \
  --weights "$WEIGHTS" \
  --output /workspace/fastvideo-out \
  --prompt "A curious raccoon in a field of sunflowers." \
  "${SMOKE_FLAGS[@]}"
ls -la /workspace/fastvideo-out | head
EOF
