# syntax=docker/dockerfile:1.7
# fv-gpucheck images. Built locally by scripts/gpu/docker.sh and in CI by
# .github/workflows/gpucheck-runtime-image.yml; see scripts/gpu/README.md.
#
# builder  Ubuntu 22.04 + Rust + NVRTC 12.4. Builds `fv-gpucheck --features cuda`
#          without a CUDA toolkit (cudarc loads CUDA libraries at run time) and
#          runs everything that needs no GPU: unit tests, the NVRTC compile
#          gate, CPU-path reference dumps.
# build    Compiles the release binary from the repo (CI path).
# binary   The binary + build id. Locally overridden with
#          `--build-context binary=artifacts/gpucheck/dist` to reuse `docker.sh dist`.
# runtime  What a GPU box runs. Main publishes ghcr.io/<owner>/fastvideo-rs-runtime;
#          other branches publish a separate fastvideo-rs-runtime-<sanitized-ref>
#          package. Ubuntu 22.04 + only the CUDA libraries cudarc loads (pinned
#          NVIDIA wheels) + rsync/ffmpeg/HF downloader + the binary and scripts.
#          No PyTorch, no toolkit: ~1.5GB compressed instead of ~9GB, so hosts
#          boot quickly.

FROM ubuntu:22.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev clang curl wget ca-certificates git \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends cuda-nvrtc-12-4 \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
# Matches rust-toolchain.toml (stable + rustfmt/clippy) so containers never
# download components at run time.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --component rustfmt,clippy \
 && rustc --version
ENV CUDARC_CUDA_VERSION=12040 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 \
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
# Pinned to what the GPU ladder was validated against. cuDNN must be >= the
# version whose symbols cudarc 0.17 binds (the 9.1 in PyTorch/CUDA images is too old).
ARG NVRTC_VERSION=12.4.127
# cuBLAS 12.9: 12.4 predates Blackwell and runs generic FP32 kernels there
# (TF32 / bf16 compute ignored, FP32 2.6x slower; see `validate.sh run mathprobe`).
ARG CUBLAS_VERSION=12.9.1.4
ARG CUDNN_VERSION=9.26.0.51
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip rsync ffmpeg openssh-server ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir --no-deps --target /opt/nvidia-libs \
      "nvidia-cuda-nvrtc-cu12==${NVRTC_VERSION}" \
      "nvidia-cublas-cu12==${CUBLAS_VERSION}" \
      "nvidia-cudnn-cu12==${CUDNN_VERSION}" \
 && pip3 install --no-cache-dir huggingface_hub hf_transfer \
 && ls -d /opt/nvidia-libs/nvidia/*/lib > /etc/ld.so.conf.d/fastvideo-nvidia.conf \
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
