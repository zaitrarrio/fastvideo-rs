#!/usr/bin/env bash
# Build the vendored Rust-to-PTX packages in Docker. No NVIDIA GPU required.
# Docker Desktop on macOS cross-builds linux/amd64.
#
#   third_party/cuda-oxide  v0.2.1  → librustc_codegen_cuda.so
#   third_party/cutile-rs   v0.3.1  → default workspace members (not cuda-tile-rs)
#
# Image: docker/oxide.Dockerfile (Ubuntu 22.04, CUDA 13.0, LLVM 21,
# nightly-2026-04-03). Backend lands in artifacts/oxide/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${OXIDE_BUILDER_IMAGE:-fastvideo-oxide-builder:cu130}"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
OXIDE="$ROOT/third_party/cuda-oxide"
CUTILE="$ROOT/third_party/cutile-rs"

if [[ ! -f "$OXIDE/Cargo.toml" || ! -f "$CUTILE/Cargo.toml" ]]; then
  echo "oxide: vendored toolchain is not checked out." >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi
command -v docker >/dev/null 2>&1 || { echo "oxide: docker is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "oxide: Docker daemon is not running" >&2; exit 1; }

docker build \
  --platform "$PLATFORM" \
  -f "$ROOT/docker/oxide.Dockerfile" \
  -t "$IMAGE" \
  "$ROOT/docker"

mkdir -p "$ROOT/artifacts/oxide"

docker run --rm \
  --platform "$PLATFORM" \
  -v "$ROOT":/src \
  -v fastvideo-rs-cargo-registry:/usr/local/cargo/registry \
  -v fastvideo-rs-cargo-git:/usr/local/cargo/git \
  -v fastvideo-rs-oxide-target:/oxide-target \
  -e CARGO_HOME=/usr/local/cargo \
  -w /src \
  "$IMAGE" \
  bash -euo pipefail -c '
    export CARGO_TARGET_DIR=/oxide-target/cuda-oxide
    cd /src/third_party/cuda-oxide
    cargo oxide setup
    install -D "$CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so" /src/artifacts/oxide/librustc_codegen_cuda.so
    export CARGO_TARGET_DIR=/oxide-target/cutile
    cd /src/third_party/cutile-rs
    cargo build --locked --release
  '

echo "oxide: backend $ROOT/artifacts/oxide/librustc_codegen_cuda.so"
echo "oxide: cutile release libs are in the fastvideo-rs-oxide-target volume (/oxide-target/cutile)"
