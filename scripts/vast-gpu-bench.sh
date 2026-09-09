#!/usr/bin/env bash
# Sync, CUDA-test, and GPU-bench on the running Vast 4090. Not a Mac CPU job.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${VAST_INSTANCE_ID:-50416610}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/fastvideo-rs}"
CARGO_HOME_DIR="${VAST_CARGO_HOME:-/workspace/.cargo}"
WEIGHTS="${WEIGHTS_DIR:-/workspace/weights/Wan2.1-T2V-1.3B-Diffusers}"
CLIP_ROOT="${CLIP_WEIGHTS:-/workspace/weights/Wan2.1-I2V-14B-480P-Diffusers}"
MODE="${1:-all}"

"$ROOT/scripts/vast-sync.sh"

url="$(vastai ssh-url "$INSTANCE_ID")"
hostport="${url#ssh://root@}"
host="${hostport%:*}"
port="${hostport##*:}"

ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" bash -s -- \
  "$REMOTE_DIR" "$CARGO_HOME_DIR" "$WEIGHTS" "$CLIP_ROOT" "$MODE" <<'EOF'
set -euo pipefail
REMOTE_DIR="$1"
CARGO_HOME_DIR="$2"
WEIGHTS="$3"
CLIP_ROOT="$4"
MODE="$5"
export CARGO_HOME="$CARGO_HOME_DIR"
mkdir -p "$CARGO_HOME"
# shellcheck disable=SC1091
if [ -f "$HOME/.cargo/env" ]; then
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
cd "$REMOTE_DIR"

if ! command -v rustc >/dev/null 2>&1; then
  bash scripts/vast-setup-cuda.sh
  source "$HOME/.cargo/env"
fi

nvidia-smi -L
echo "building --features cuda (release)"
cargo test --workspace --features cuda --release -- --nocapture

if [ "$MODE" = "test" ]; then
  exit 0
fi

if [ ! -d "$WEIGHTS/transformer" ]; then
  echo "pulling 1.3B + CLIP weights"
  bash scripts/vast-pull-weights.sh
fi

export FASTVIDEO_GPU_SMOKE=1
cargo test -p fastvideo-core --features cuda --release cuda_1_3b_smoke_gated -- --nocapture || true

mkdir -p /workspace/fastvideo-bench /workspace/fastvideo-clip
python3 - <<'PY'
from pathlib import Path
from PIL import Image
p = Path("/workspace/fastvideo-clip/cond.png")
p.parent.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (512, 320), (40, 120, 200)).save(p)
print("wrote", p)
PY

echo "=== bench tiny CUDA ==="
./target/release/fastvideo generate \
  --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --tiny --device cuda --dtype f32 --output /workspace/fastvideo-tiny \
  --prompt "A curious raccoon in a field of sunflowers."

echo "=== bench 1.3B smoke ==="
./target/release/fastvideo bench \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --device cuda --dtype bf16 \
  --weights "$WEIGHTS" \
  --frames 9 --steps 2 --height 256 --width 256 --guidance 1.0 \
  --output /workspace/fastvideo-bench \
  --prompt "A curious raccoon in a field of sunflowers."

if [ -d "$CLIP_ROOT/image_encoder" ]; then
  echo "=== bench CLIP ViT-H ==="
  ./target/release/fastvideo bench \
    --model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --device cuda --dtype bf16 \
    --weights "$CLIP_ROOT" \
    --clip-only --image /workspace/fastvideo-clip/cond.png
else
  echo "skip CLIP bench: no $CLIP_ROOT/image_encoder"
fi

if [ "$MODE" = "full" ]; then
  echo "=== bench 1.3B 480p 8-step ==="
  ./target/release/fastvideo bench \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --device cuda --dtype bf16 \
    --weights "$WEIGHTS" \
    --frames 33 --steps 8 --height 480 --width 832 --guidance 3.0 \
    --output /workspace/fastvideo-bench-480 \
    --prompt "A curious raccoon in a field of sunflowers."
fi

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
ls -la /workspace/fastvideo-bench | head
EOF
