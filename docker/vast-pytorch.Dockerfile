# syntax=docker/dockerfile:1.7
# Optional Vast images FROM vastai/pytorch — only for tiers that need Python+torch.
#
# Default rentals still use the slim image (docker/gpucheck.Dockerfile): Ubuntu +
# CUDA 13 libs + fv-gpucheck, no PyTorch. Use these images only when a tier runs
# transformers/diffusers/FastVideo in Python (oracle, compare, taehv, …).
#
# Targets:
#   runtime  vastai/pytorch + CUDA 13 for cudarc + fv-gpucheck + hf-fm + gcc/ffmpeg
#   oracle   runtime + transformers/diffusers in /venv/main (skips billed oracle-venv)
#
# Built by scripts/gpu/docker.sh and .github/workflows/vast-pytorch-image.yml.
# Do not override ENTRYPOINT/CMD — Vast's base image owns SSH / Jupyter supervisord.
#
# ARG VAST_PYTORCH_IMAGE pins the base. Prefer a cuda-13.* tag so it matches the
# CUDA 13.0 libraries cudarc loads (driver floor cuda_vers>=13.0).

ARG VAST_PYTORCH_IMAGE=vastai/pytorch:cuda-13.0.3-auto

# ---- compile stages (same shape as docker/gpucheck.Dockerfile) -------------------
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

# ---- Vast runtime ----------------------------------------------------------------
FROM ${VAST_PYTORCH_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_ID=unknown
# CUDA 13 libs for fv-gpucheck (cudarc). Torch keeps its own wheels under
# /venv/main; remote.sh clears LD_LIBRARY_PATH for python oracles so the two
# never mix. gcc is for Triton's first-import driver shim.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      rsync ffmpeg openssh-server ca-certificates curl wget binutils \
      gcc g++ git \
 && if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then \
      wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb \
      && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
      && apt-get update; \
    fi \
 && apt-get install -y --no-install-recommends \
      cuda-nvrtc-13-0 libcublas-13-0 libcudnn9-cuda-13 \
 && rm -rf /var/lib/apt/lists/* \
 && echo /usr/local/cuda-13.0/lib64 > /etc/ld.so.conf.d/fastvideo-nvidia.conf \
 && ldconfig \
 && ldconfig -p | grep -E 'libnvrtc\.so|libcublasLt\.so|libcublas\.so|libcudnn\.so' \
 && mkdir -p /run/sshd /opt/fastvideo-rs/target/release
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    FV_IMAGE_FLAVOR=pytorch
COPY --from=hf-fm /out/hf-fm /out/hf-fetch-model /usr/local/bin/
COPY scripts/gpu /opt/fastvideo-rs/scripts/gpu
COPY --from=binary /fv-gpucheck /fv-gpucheck.build-id /opt/fastvideo-rs/target/release/
LABEL org.opencontainers.image.source="https://github.com/zaitrarrio/fastvideo-rs" \
      org.opencontainers.image.description="fastvideo-rs on vastai/pytorch (fv-gpucheck + CUDA 13 + torch)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      dev.fastvideo.build-id="${BUILD_ID}" \
      dev.fastvideo.flavor="pytorch"
WORKDIR /opt/fastvideo-rs

# ---- Oracle (prebaked Python reference stack) ------------------------------------
FROM runtime AS oracle
# Install into the image's primary venv so oracles skip a billed uv create.
# Do not reinstall torch — vastai/pytorch already provides it for this CUDA line.
RUN /venv/main/bin/python -m pip install --no-cache-dir -U pip \
 && /venv/main/bin/python -m pip install --no-cache-dir \
      transformers accelerate safetensors sentencepiece protobuf pillow numpy \
      "diffusers @ https://github.com/huggingface/diffusers/archive/refs/heads/main.tar.gz" \
 && /venv/main/bin/python -c 'import torch, transformers, diffusers; print(torch.__version__, transformers.__version__, diffusers.__version__)'
ENV FV_IMAGE_FLAVOR=oracle \
    FV_ORACLE_PYTHON=/venv/main/bin/python
LABEL org.opencontainers.image.description="fastvideo-rs on vastai/pytorch with transformers/diffusers for oracles" \
      dev.fastvideo.flavor="oracle"
