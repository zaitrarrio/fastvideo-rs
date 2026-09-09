#!/usr/bin/env bash
# Install Rust + CUDA 12.4 nvcc on a Vast pytorch *runtime* image so Candle can compile kernels.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake pkg-config git curl ca-certificates wget \
  libssl-dev clang libclang-dev

if ! command -v rustc >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.82.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export CARGO_HOME="${CARGO_HOME:-/workspace/.cargo}"
mkdir -p "$CARGO_HOME"
# Keep the 1.82 toolchain; rust-toolchain.toml "stable" would otherwise auto-update.

if ! command -v nvcc >/dev/null 2>&1; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  rm -f cuda-keyring_1.1-1_all.deb
  apt-get update
  apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4 libcublas-dev-12-4 \
    cuda-nvrtc-dev-12-4 libcurand-dev-12-4
fi

export PATH="/usr/local/cuda-12.4/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
rustc --version
nvidia-smi -L
