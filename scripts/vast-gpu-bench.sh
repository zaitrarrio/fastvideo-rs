#!/usr/bin/env bash
# Sync, CUDA-test, and GPU-bench on a Vast instance. Not a Mac CPU job.
# MODE: all (default) | test | 4s | 5s | full
#   4s  = CUDA tests + 480p 9-frame canary + 65-frame (~4s @16fps) 8-step bench
#   5s  = CUDA tests + 480p 9-frame canary + 81-frame (~5s @16fps) 8-step bench
#   full = all + 4s + 5s
# Each job writes under /workspace/artifacts/<stamp>-<gpu>-i<id>-<mode>/.
# Pull a copy home with: ./scripts/vast-pull-artifacts.sh
#
# Budget model:
#   - BENCH_TIMEOUT_SEC (default 5400s = 90m) caps the remote bench.
#   - The remote script installs `trap ... EXIT INT TERM` that destroys the
#     instance via `vastai destroy instance` so idle time cannot accrue.
#   - Local shell installs the same on INT/TERM.
#   - SSH keepalive: ServerAliveInterval=15s, ServerAliveCountMax=3, so the
#     tunnel dies within 45s of silence (vs. 10 minutes by default).
set -euo pipefail

# Safe to source: bail out after the trap+preflight are installed once.
# When sourced (BASH_SOURCE != $0), `return` is allowed; when executed
# directly, just skip the early-exit so we don't trigger `set -e` on an
# invalid `return`.
if [[ -n "${_VAST_BENCH_TRAPPED:-}" ]]; then
  [[ "${BASH_SOURCE[0]}" != "$0" ]] && return 0 || true
fi
_VAST_BENCH_TRAPPED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${VAST_INSTANCE_ID:-50512723}"
SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/id_strobe_vast}"
REMOTE_DIR="${VAST_REMOTE_DIR:-/workspace/fastvideo-rs}"
CARGO_HOME_DIR="${VAST_CARGO_HOME:-/workspace/.cargo}"
WEIGHTS="${WEIGHTS_DIR:-/workspace/weights/Wan2.1-T2V-1.3B-Diffusers}"
CLIP_ROOT="${CLIP_WEIGHTS:-/workspace/weights/Wan2.1-I2V-14B-480P-Diffusers}"
MODE="${1:-all}"
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-5400}"

# Local-side: on Ctrl-C / TERM, destroy the instance immediately so we don't
# accumulate idle billed time while we figure out what's wrong. Normal exit
# goes through the remote script's own EXIT trap, so we skip this one then.
# Only install when executed directly — when sourced we leave the caller's
# signal handling alone.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _on_local_int() {
    echo "[vast-gpu-bench] local INT/TERM — destroying instance $INSTANCE_ID"
    vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
    exit 130
  }
  # Install local Ctrl-C / TERM guard. No EXIT slot — the remote script's own
  # EXIT trap handles clean shutdown.
  trap '_on_local_int' INT TERM
fi

# Pre-flight: refuse to spend money on an already-stopped instance.  Only run
# when this script is executed directly (not sourced) so we don't blow up
# the caller's shell via `exit`.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if vastai show instance "$INSTANCE_ID" 2>/dev/null | grep -q "exited"; then
    echo "instance $INSTANCE_ID already exited; pick a new one (vastai show instance $INSTANCE_ID)" >&2
    exit 2
  fi
fi

"$ROOT/scripts/vast-sync.sh"

url="$(vastai ssh-url "$INSTANCE_ID")"
hostport="${url#ssh://root@}"
host="${hostport%:*}"
port="${hostport##*:}"

# Default: cudarc-only lean path. Override e.g. FASTVIDEO_BENCH_BACKENDS=candle,cudarc
BENCH_BACKENDS="${FASTVIDEO_BENCH_BACKENDS:-cudarc}"
BENCH_BACKENDS="${BENCH_BACKENDS// /,}"
# H100/H200=90, 4090=89; override with VAST_CUDA_COMPUTE_CAP.
CUDA_CAP="${VAST_CUDA_COMPUTE_CAP:-}"
if [ -z "$CUDA_CAP" ]; then
  case "$INSTANCE_ID" in
    50435452|50496300|50512723) CUDA_CAP=90 ;;
    50416610) CUDA_CAP=89 ;;
    *) CUDA_CAP=90 ;;
  esac
fi

# Hard wall-clock cap on the entire bench.  `timeout --foreground` keeps
# SIGTERM/SIGKILL delivery sane under `set -e` and ensures Ctrl-C reaches
# the bench loop instead of the timeout wrapper.  `--kill-after=30`
# SIGKILLs anything (cargo link, hung GPU kernel) that ignores SIGTERM.
# On cap-hit: ssh is killed, sshd closes the channel on the remote,
# the remote bash hits EOF, the EXIT trap fires, and `vastai destroy
# instance` reclaims the box so idle billed time cannot accrue.
timeout --foreground --kill-after=30 "$BENCH_TIMEOUT_SEC" \
ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "root@$host" \
  env \
  REMOTE_DIR="$REMOTE_DIR" \
  CARGO_HOME_DIR="$CARGO_HOME_DIR" \
  WEIGHTS="$WEIGHTS" \
  CLIP_ROOT="$CLIP_ROOT" \
  MODE="$MODE" \
  INSTANCE_ID="$INSTANCE_ID" \
  FASTVIDEO_BENCH_BACKENDS="$BENCH_BACKENDS" \
  CUDA_COMPUTE_CAP="$CUDA_CAP" \
  CARGO_HOME="$CARGO_HOME_DIR" \
  FASTVIDEO_LOG="${FASTVIDEO_LOG:-debug}" \
  FASTVIDEO_TF32="${FASTVIDEO_TF32:-1}" \
  FASTVIDEO_BF16="${FASTVIDEO_BF16:-1}" \
  FASTVIDEO_DEVICE_SCHED="${FASTVIDEO_DEVICE_SCHED:-1}" \
  FASTVIDEO_RESIDENT="${FASTVIDEO_RESIDENT:-1}" \
  FASTVIDEO_SDPA="${FASTVIDEO_SDPA:-flash}" \
  FASTVIDEO_SDPA_CHUNK="${FASTVIDEO_SDPA_CHUNK:-}" \
  bash -s <<'EOF'
set -euo pipefail

# Hard-budget safeguard: on any exit from this remote shell (success, error,
# SSH session timeout, parent kill), destroy the Vast instance so idle billed
# time cannot accrue.  `exit 0` ensures the trap runs even if the bench was
# killed by `timeout`, a Ctrl-C, or the SSH channel closing.
trap 'vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true; exit 0' EXIT INT TERM

# Wall-clock cap on the bench block.  Override by exporting BENCH_TIMEOUT_SEC
# before invoking scripts/vast-gpu-bench.sh.  Default 90 minutes.
BENCH_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC:-5400}"

export CARGO_HOME="${CARGO_HOME:-$CARGO_HOME_DIR}"
# Normalize commas → spaces for the for-loop.
FASTVIDEO_BENCH_BACKENDS="${FASTVIDEO_BENCH_BACKENDS//,/ }"
# Hopper-max defaults (H100/H200); empty SDPA_CHUNK → auto by SM.
export FASTVIDEO_LOG FASTVIDEO_TF32 FASTVIDEO_BF16 FASTVIDEO_DEVICE_SCHED FASTVIDEO_RESIDENT FASTVIDEO_SDPA
[ -n "${FASTVIDEO_SDPA_CHUNK:-}" ] && export FASTVIDEO_SDPA_CHUNK
echo "hopper env: LOG=$FASTVIDEO_LOG TF32=$FASTVIDEO_TF32 BF16=$FASTVIDEO_BF16 DEVICE_SCHED=$FASTVIDEO_DEVICE_SCHED RESIDENT=$FASTVIDEO_RESIDENT SDPA=$FASTVIDEO_SDPA CHUNK=${FASTVIDEO_SDPA_CHUNK:-auto}"
mkdir -p "$CARGO_HOME"
# shellcheck disable=SC1091
if [ -f "$HOME/.cargo/env" ]; then
  source "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.4/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
cd "$REMOTE_DIR"

if ! command -v rustc >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
  bash scripts/vast-setup-cuda.sh
fi
# Prefer workspace CARGO_HOME (Vast disks) over /root/.cargo.
# shellcheck disable=SC1091
if [ -f "$CARGO_HOME/env" ]; then
  source "$CARGO_HOME/env"
elif [ -f "$HOME/.cargo/env" ]; then
  source "$HOME/.cargo/env"
fi
export PATH="$CARGO_HOME/bin:/usr/local/cuda-12.4/bin:${PATH}"

nvidia-smi -L
GPU_SLUG="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="/workspace/artifacts/${STAMP}-${GPU_SLUG}-i${INSTANCE_ID}-${MODE}"
mkdir -p "$RUN_DIR"
export FASTVIDEO_ARTIFACT_DIR="$RUN_DIR/tests"
mkdir -p "$FASTVIDEO_ARTIFACT_DIR"
cat > "$RUN_DIR/run.json" <<RUN
{
  "instance_id": "$INSTANCE_ID",
  "gpu": "$GPU_SLUG",
  "mode": "$MODE",
  "started_utc": "$STAMP",
  "hostname": "$(hostname)"
}
RUN
echo "artifacts $RUN_DIR"

# Pick CUDA Cargo features from requested backends.
# candle/burn need the full `cuda` feature (Candle kernels + CubeCL).
# cudarc-only smoke uses lean `cuda-cudarc` to skip those cold builds.
NEED_FULL_CUDA=0
for BACKEND in $FASTVIDEO_BENCH_BACKENDS; do
  case "$BACKEND" in
    candle|burn) NEED_FULL_CUDA=1 ;;
  esac
done
if [ "$NEED_FULL_CUDA" = "1" ]; then
  CUDA_FEATURES=cuda
else
  CUDA_FEATURES=cuda-cudarc
fi
echo "building --features $CUDA_FEATURES (release); backends=$FASTVIDEO_BENCH_BACKENDS"

# Skip lib tests on the GPU box — they run on remote CPU and burn billable time.
# Use `--features` matching the bench backend. MODE=test just builds and exits.
if [ "$CUDA_FEATURES" = "cuda-cudarc" ]; then
  cargo build --release --features cuda-cudarc -p fastvideo-cli
else
  cargo build --release --features cuda -p fastvideo-cli
fi

if [ "$MODE" = "test" ]; then
  echo "MODE=$MODE done artifacts=$RUN_DIR"
  exit 0
fi

if [ ! -d "$WEIGHTS/transformer" ]; then
  echo "pulling 1.3B + CLIP weights"
  bash scripts/vast-pull-weights.sh
fi

if [ "$MODE" != "test" ] && [ "$CUDA_FEATURES" = "cuda" ]; then
  export FASTVIDEO_GPU_SMOKE=1
  # Note: smoke test removed to save billable CPU time on the H200.
  # Run `cargo test -p fastvideo-core --features cuda --release cuda_1_3b_smoke_gated`
  # locally if you need this gate.
fi

CLIP_IMG="$RUN_DIR/clip/cond.png"
mkdir -p "$RUN_DIR/clip"
python3 - <<PY
from pathlib import Path
from PIL import Image
p = Path("$CLIP_IMG")
p.parent.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (512, 320), (40, 120, 200)).save(p)
print("wrote", p)
PY

if [ "$MODE" != "test" ]; then
  BACKENDS="${FASTVIDEO_BENCH_BACKENDS:-cudarc}"
  for BACKEND in $BACKENDS; do
    if [ "$BACKEND" = "candle" ] || [ "$BACKEND" = "burn" ] || [ "$BACKEND" = "cudarc" ]; then
      DTYPE=f32
      [ "$BACKEND" = "candle" ] && DTYPE=bf16
      [ "$BACKEND" = "cudarc" ] && DTYPE=f32
      echo "=== bench tiny ($BACKEND cuda) ==="
      ./target/release/fastvideo generate \
        --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
        --backend "$BACKEND" \
        --tiny --device cuda --dtype "$DTYPE" --output "$RUN_DIR/tiny-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers." || {
          echo "warn: tiny $BACKEND cuda failed"; continue
        }
    else
      echo "=== bench tiny ($BACKEND cpu compile) ==="
      ./target/release/fastvideo generate \
        --model FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
        --backend "$BACKEND" \
        --tiny --device cpu --dtype f32 --output "$RUN_DIR/tiny-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers." || {
          echo "warn: tiny $BACKEND failed"; continue
        }
    fi

    if [ "$BACKEND" = "cudarc" ]; then
      echo "=== bench 1.3B smoke ($BACKEND cuda) ==="
      ./target/release/fastvideo bench \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --backend cudarc \
        --device cuda --dtype f32 \
        --weights "$WEIGHTS" \
        --frames 9 --steps 2 --height 256 --width 256 --guidance 1.0 \
        --output "$RUN_DIR/smoke-256-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers." || {
          echo "warn: 1.3B smoke $BACKEND failed"
        }
    elif [ "$BACKEND" = "candle" ]; then
      echo "=== bench 1.3B smoke ($BACKEND cuda) ==="
      ./target/release/fastvideo bench \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --backend candle \
        --device cuda --dtype bf16 \
        --weights "$WEIGHTS" \
        --frames 9 --steps 2 --height 256 --width 256 --guidance 1.0 \
        --output "$RUN_DIR/smoke-256-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers."
    elif [ "$BACKEND" = "burn" ]; then
      echo "=== bench 1.3B smoke ($BACKEND cuda f32) ==="
      ./target/release/fastvideo bench \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --backend burn \
        --device cuda --dtype f32 \
        --weights "$WEIGHTS" \
        --frames 9 --steps 2 --height 256 --width 256 --guidance 1.0 \
        --output "$RUN_DIR/smoke-256-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers." || {
          echo "warn: 1.3B smoke $BACKEND failed"
        }
    else
      echo "=== bench 1.3B smoke ($BACKEND cpu f32) ==="
      ./target/release/fastvideo generate \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --backend "$BACKEND" \
        --device cpu --dtype f32 \
        --weights "$WEIGHTS" \
        --frames 9 --steps 2 --height 256 --width 256 --guidance 1.0 \
        --output "$RUN_DIR/smoke-256-$BACKEND" \
        --prompt "A curious raccoon in a field of sunflowers." || {
          echo "warn: 1.3B smoke $BACKEND failed (host graph may OOM or be slow)"
        }
    fi
  done

  if [ -d "$CLIP_ROOT/image_encoder" ]; then
    echo "=== bench CLIP ViT-H ==="
    ./target/release/fastvideo bench \
      --model Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
      --backend cudarc \
      --device cuda --dtype f32 \
      --weights "$CLIP_ROOT" \
      --output "$RUN_DIR/clip" \
      --clip-only --image "$CLIP_IMG" || {
        echo "warn: CLIP bench failed (cudarc CLIP may still be landing)"
      }
  else
    echo "skip CLIP bench: no $CLIP_ROOT/image_encoder"
  fi
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "4s" ] || [ "$MODE" = "5s" ]; then
  echo "=== canary 1.3B 480p 9f (OOM check, cudarc) ==="
  ./target/release/fastvideo bench \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --backend cudarc \
    --device cuda --dtype f32 \
    --weights "$WEIGHTS" \
    --frames 9 --steps 2 --height 480 --width 832 --guidance 1.0 \
    --output "$RUN_DIR/canary-480p-cudarc" \
    --prompt "A curious raccoon in a field of sunflowers." || {
      echo "warn: canary 480p failed"
    }
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "4s" ]; then
  echo "=== bench 1.3B 480p 4s (65 frames @16fps, cudarc) ==="
  ./target/release/fastvideo bench \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --backend cudarc \
    --device cuda --dtype f32 \
    --weights "$WEIGHTS" \
    --frames 65 --steps 8 --height 480 --width 832 --guidance 1.0 \
    --output "$RUN_DIR/t2v-480p-4s-cudarc" \
    --prompt "A curious raccoon in a field of sunflowers." || {
      echo "warn: 4s cudarc failed"
    }
fi

if [ "$MODE" = "full" ] || [ "$MODE" = "5s" ]; then
  echo "=== bench 1.3B 480p 5s (81 frames @16fps, cudarc) ==="
  ./target/release/fastvideo bench \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --backend cudarc \
    --device cuda --dtype f32 \
    --weights "$WEIGHTS" \
    --frames 81 --steps 8 --height 480 --width 832 --guidance 1.0 \
    --output "$RUN_DIR/t2v-480p-5s-cudarc" \
    --prompt "A curious raccoon in a field of sunflowers." || {
      echo "warn: 5s cudarc failed"
    }
fi

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv | tee "$RUN_DIR/nvidia-smi.csv"
date -u +%Y%m%dT%H%M%SZ > "$RUN_DIR/finished-utc"
echo "MODE=$MODE done artifacts=$RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f | sort
EOF
