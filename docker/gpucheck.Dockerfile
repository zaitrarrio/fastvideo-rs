# syntax=docker/dockerfile:1.7
# fv-gpucheck images. Built locally by scripts/gpu/docker.sh and in CI by
# .github/workflows/gpucheck-runtime-image.yml; see scripts/gpu/README.md.
#
# builder  Ubuntu 22.04 + Rust + CUDA 13.4 nvcc/NVRTC/tileiras. Builds `fv-gpucheck
#          --features cuda` with per-SM cubins compiled ahead of time by
#          build.rs (cudarc still loads the CUDA *libraries* at run time), and
#          runs everything that needs no GPU: unit tests, the compile gates,
#          CPU-path reference dumps.
# hf-fm    Builds the Rust HuggingFace downloader (hf-fetch-model --features cli).
# build    Compiles the release binary from the repo (CI path).
# binary   The binary + build id. Locally overridden with
#          `--build-context binary=artifacts/gpucheck/dist` to reuse `docker.sh dist`.
# runtime  What a GPU box runs (ghcr.io/zaitrarrio/fastvideo-rs-runtime): Ubuntu
#          22.04 + pinned CUDA 13.4 libraries (scripts/gpu/cuda-13.pins) +
#          tileiras + rsync/ffmpeg/hf-fm + the binary and scripts. No Python,
#          no PyTorch, no toolkit: hosts boot quickly.

FROM ubuntu:22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev clang curl wget ca-certificates git \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && . /etc/fastvideo/cuda-13.pins \
 && apt-get install -y --no-install-recommends --allow-downgrades \
      "$CUDA_NVCC_PKG" "$CUDA_NVRTC_PKG" "$CUDA_NVRTC_DEV_PKG" \
      "$CUDA_TILEIRAS_PKG" \
 && apt-mark hold cuda-nvcc-13-4 cuda-nvrtc-13-4 cuda-nvrtc-dev-13-4 cuda-tileiras-13-4 \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# Matches rust-toolchain.toml (stable + rustfmt/clippy) so containers never
# download components at run time.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt,clippy \
 && rustc --version
ENV CUDARC_CUDA_VERSION=13040 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64 \
    PATH=/usr/local/cuda-13.4/bin:/usr/local/cargo/bin:$PATH \
    NVCC=/usr/local/cuda-13.4/bin/nvcc \
    CARGO_TARGET_DIR=/target \
    CARGO_PROFILE_RELEASE_LTO=off \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
    CARGO_PROFILE_RELEASE_PANIC=unwind
WORKDIR /src

FROM builder AS hf-fm
RUN mkdir -p /out \
 && cargo install hf-fetch-model --features cli \
 && cp "$(command -v hf-fm)" /out/hf-fm \
 && cp "$(command -v hf-fetch-model)" /out/hf-fetch-model

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
# CUDA 13.4 runtime libraries from NVIDIA's apt repo, versions pinned in
# scripts/gpu/cuda-13.pins. 13.4 needs a >= 580 driver; validate.sh's offer
# filter asks Vast for cuda_vers>=13.0. No Python: weights come through hf-fm.
COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      rsync ffmpeg openssh-server ca-certificates curl wget binutils \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && . /etc/fastvideo/cuda-13.pins \
 && apt-get install -y --no-install-recommends --allow-downgrades \
      "$CUDA_NVRTC_PKG" "$CUDA_CUBLAS_PKG" "$CUDA_CUDNN_PKG" \
      "$CUDA_TILEIRAS_PKG" \
 && apt-mark hold cuda-nvrtc-13-4 libcublas-13-4 libcudnn9-cuda-13 cuda-tileiras-13-4 \
 && rm -rf /var/lib/apt/lists/* \
 && echo /usr/local/cuda-13.4/lib64 > /etc/ld.so.conf.d/fastvideo-nvidia.conf \
 && ldconfig \
 && . /etc/fastvideo/cuda-13.pins \
 && for soname in "$CUDA_NVRTC_SONAME" "$CUDA_CUBLAS_SONAME" "$CUDA_CUBLASLT_SONAME" "$CUDA_CUDNN_SONAME"; do
      src=$(ldconfig -p | awk -v n="$soname" '$1 == n { print $NF; exit }')
      test -n "$src" && test -e "$src"
      dir=$(dirname "$src")
      unversioned="${soname%.so.*}.so"
      if [ ! -e "$dir/$unversioned" ]; then ln -s "$soname" "$dir/$unversioned"; fi
    done \
 && ldconfig \
 && ldconfig -p | grep -E 'libnvrtc\.so|libcublasLt\.so|libcublas\.so|libcudnn\.so' \
 && mkdir -p /run/sshd
# The NVIDIA container runtime injects the driver (libcuda) when these are set.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=hf-fm /out/hf-fm /out/hf-fetch-model /usr/local/bin/
COPY scripts/gpu /opt/fastvideo-rs/scripts/gpu
COPY --from=binary /fv-gpucheck /fv-gpucheck.build-id /opt/fastvideo-rs/target/release/
LABEL org.opencontainers.image.source="https://github.com/zaitrarrio/fastvideo-rs" \
      org.opencontainers.image.description="fastvideo-rs cudarc GPU validation runtime (fv-gpucheck + CUDA runtime + hf-fm)" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/fastvideo-rs
