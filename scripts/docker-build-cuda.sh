#!/usr/bin/env bash
# Build the Linux CUDA fastvideo binary on this Mac via Docker (no NVIDIA GPU needed).
# Crates and incremental artifacts are cached in a Docker volume + ./target-linux.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${CUDA_BUILDER_IMAGE:-fastvideo-cuda-builder:12.4}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

cd "$ROOT"
mkdir -p target-linux

docker build \
  --platform "$PLATFORM" \
  -f docker/cuda-builder.Dockerfile \
  -t "$IMAGE" \
  docker/

docker run --rm \
  --platform "$PLATFORM" \
  -v "$ROOT":/src \
  -v "$ROOT/target-linux":/src/target-linux \
  -v fastvideo-rs-cargo-registry:/usr/local/cargo/registry \
  -v fastvideo-rs-cargo-git:/usr/local/cargo/git \
  -e CARGO_TARGET_DIR=/src/target-linux \
  -e CARGO_HOME=/usr/local/cargo \
  -w /src \
  "$IMAGE" \
  cargo build -p fastvideo-cli --release --features cuda

echo "linux cuda binary: $ROOT/target-linux/release/fastvideo"
file "$ROOT/target-linux/release/fastvideo" || true
