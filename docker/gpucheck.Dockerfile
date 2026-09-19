# syntax=docker/dockerfile:1.7
# fv-gpucheck images. Built locally by scripts/gpu/docker.sh and in CI by
# .github/workflows/gpucheck-runtime-image.yml; see scripts/gpu/README.md.
#
# builder  Ubuntu 22.04 + Rust + CUDA 13.0 nvcc/NVRTC. Builds `fv-gpucheck
#          --features cuda` with per-SM cubins compiled ahead of time by
#          build.rs (cudarc still loads the CUDA *libraries* at run time), and
#          runs everything that needs no GPU: unit tests, the compile gates,
#          CPU-path reference dumps.
# build    Compiles the release binary from the repo (CI path).
# binary   The binary + build id. Locally overridden with
#          `--build-context binary=artifacts/gpucheck/dist` to reuse `docker.sh dist`.
# runtime  What a GPU box runs (ghcr.io/zaitrarrio/fastvideo-rs-runtime): Ubuntu
#          22.04 + only the CUDA 13.0 libraries cudarc loads (NVIDIA apt) +
#          rsync/ffmpeg/HF downloader + the binary and scripts. No PyTorch, no
#          toolkit: ~1.5GB compressed instead of ~9GB, so hosts boot quickly.

FROM ubuntu:22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev clang curl wget ca-certificates git \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-nvrtc-13-0 cuda-nvrtc-dev-13-0 \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# Matches rust-toolchain.toml (stable + rustfmt/clippy) so containers never
# download components at run time.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt,clippy \
 && rustc --version
ENV CUDARC_CUDA_VERSION=13000 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
    PATH=/usr/local/cuda-13.0/bin:/usr/local/cargo/bin:$PATH \
    NVCC=/usr/local/cuda-13.0/bin/nvcc \
    CARGO_TARGET_DIR=/target \
    CARGO_PROFILE_RELEASE_LTO=off \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    CARGO_PROFILE_RELEASE_PANIC=unwind
WORKDIR /src

FROM builder AS build
ARG BUILD_ID=unknown
COPY . /src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/target \
    cargo build --release -p fastvideo-gpucheck --features cuda \
 && mkdir -p /out \
 && cp /target/release/fv-gpucheck /out/fv-gpucheck \
 && echo "$BUILD_ID" > /out/fv-gpucheck.build-id

FROM scratch AS binary
COPY --from=build /out/ /

FROM ubuntu:22.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
# CUDA 13.0 runtime libraries from NVIDIA's apt repo (the PyPI `-cu13` wheels
# are placeholders). 13.0 needs a >= 580 driver; validate.sh's offer filter
# asks Vast for cuda_vers>=13.0 so an older box is never rented. cuDNN must be
# >= the version whose symbols cudarc 0.17 binds.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip rsync ffmpeg openssh-server ca-certificates curl wget \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvrtc-13-0 libcublas-13-0 libcudnn9-cuda-13 \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir huggingface_hub hf_transfer \
 && echo /usr/local/cuda-13.0/lib64 > /etc/ld.so.conf.d/fastvideo-nvidia.conf \
 && ldconfig \
 && ldconfig -p | grep -E 'libnvrtc\.so|libcublasLt\.so|libcublas\.so|libcudnn\.so' \
 && mkdir -p /run/sshd
# The NVIDIA container runtime injects the driver (libcuda) when these are set.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY scripts/gpu /opt/fastvideo-rs/scripts/gpu
COPY --from=binary /fv-gpucheck /fv-gpucheck.build-id /opt/fastvideo-rs/target/release/
LABEL org.opencontainers.image.source="https://github.com/zaitrarrio/fastvideo-rs" \
      org.opencontainers.image.description="fastvideo-rs cudarc GPU validation runtime (fv-gpucheck + CUDA runtime libraries)" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/fastvideo-rs
