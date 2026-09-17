# syntax=docker/dockerfile:1.7
# fv-gpucheck images. Driven by scripts/gpu/docker.sh; see scripts/gpu/README.md.
#
# builder  Ubuntu 22.04 (same glibc as the Vast pytorch image the binary ships
#          to) + Rust + NVRTC 12.4. Builds `fv-gpucheck --features cuda`
#          without a CUDA toolkit (cudarc loads CUDA libraries at runtime) and
#          runs everything that needs no GPU: unit tests, the NVRTC compile
#          gate, CPU-path reference dumps.
# runtime  CUDA 12.4 + cuDNN runtime + the prebuilt binary. Runs every GPU
#          stage on any Linux host with the NVIDIA Container Toolkit:
#          `docker run --gpus all ...`. (No GPU passthrough on macOS.)

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

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg python3-pip ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --no-cache-dir 'huggingface_hub[hf_transfer]' \
 && pip3 install --no-cache-dir --no-deps --target /opt/fv-cudnn 'nvidia-cudnn-cu12==9.26.0.51' \
 && ln -sf /opt/fv-cudnn/nvidia/cudnn/lib/libcudnn.so.9 /opt/fv-cudnn/nvidia/cudnn/lib/libcudnn.so
# cudarc 0.17's cuDNN bindings need symbols newer than the base image's cuDNN 9.1.
ENV LD_LIBRARY_PATH=/opt/fv-cudnn/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
COPY fv-gpucheck fv-gpucheck.build-id /usr/local/bin/
ENV FV_BUILD_ID_FILE=/usr/local/bin/fv-gpucheck.build-id
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/fv-gpucheck"]
