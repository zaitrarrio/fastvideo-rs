# syntax=docker/dockerfile:1.7
# Upstream (Python) reference stacks for the RTX PRO 6000 comparison runs, one
# image per stack so the pods install nothing at boot:
#
#   fastvideo      hao-ai-lab/FastVideo @ FV_REV (torch 2.12 cu130, PyPI fastvideo-kernel:
#                  Triton VSA on sm_120; the sm_100a extension is datacenter-Blackwell only)
#   sol-h3         sol-engine MiniMax-H3 RTX5090 profile: pinned SGLang checkout
#                  (Rust extensions off) + Sol-Attn (techniques/sparse_backends), torch 2.11 cu130
#   sol-h3-4step   sol-engine Sol-H3 package requirements (torch 2.10 cu130, diffusers pin,
#                  cudnn-frontend[cutedsl])
#   sol-ltx25      Lightricks/LTX-2 @ fd4ded7 (uv sync; torch 2.13 cu132) + nvidia-cutlass-dsl
#                  + cuda-python + Sol-Attn
#
# Every stack is a uv venv under /opt/upstream built by scripts/gpu/upstream/setup.sh
# (the same installer pods used before), so the pins live in one place. The base
# is the public Runpod PyTorch image (CUDA 13.0 devel, Ubuntu 24.04) pinned by
# digest: Runpod hosts usually have its layers cached, and the devel toolchain
# covers the JIT paths (flashinfer, CuTe DSL). No weights are baked in.
# Built by .github/workflows/upstream-images.yml.

ARG BASE_IMAGE=runpod/pytorch:1.3.3-cu1300-torch2130-ubuntu2404@sha256:2add043aaa3f184bba3cf59fc3700ebcf02b3d1324eb53ce03719a01c912403d
ARG FVRS_RUNTIME=ghcr.io/zaitrarrio/fastvideo-rs-runtime:latest

FROM ${FVRS_RUNTIME} AS fvrs

FROM ${BASE_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    UP=/opt/upstream \
    UV_PYTHON_INSTALL_DIR=/opt/upstream/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    PATH=/opt/upstream/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates gcc g++ cmake ninja-build ffmpeg jq \
 && rm -rf /var/lib/apt/lists/*
# Only the installer: pins change rarely, the runner scripts often.
COPY scripts/gpu/upstream/setup.sh /opt/upstream-setup/setup.sh
RUN . /opt/upstream-setup/setup.sh && ensure_base \
 && uv venv -q --python 3.12 /opt/upstream/tools && uv pip install -q --python /opt/upstream/tools/bin/python numpy \
 && rm -rf /tmp/uv-cache
COPY --from=fvrs /opt/fastvideo-rs/target/release/fv-gpucheck /usr/local/bin/fv-gpucheck

FROM base AS fastvideo
RUN . /opt/upstream-setup/setup.sh && install_fastvideo && rm -rf /tmp/uv-cache
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="FastVideo upstream reference stack (fastvideo-rs benchmarks)"

FROM base AS sol-h3
RUN . /opt/upstream-setup/setup.sh && install_sol_h3_rtx5090 && rm -rf /tmp/uv-cache
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="sol-engine MiniMax-H3 RTX5090 profile (pinned SGLang + Sol-Attn)"

FROM base AS sol-h3-4step
RUN . /opt/upstream-setup/setup.sh && install_sol_h3_4step && rm -rf /tmp/uv-cache
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="sol-engine Sol-H3 four-step package"

FROM base AS sol-ltx25
RUN . /opt/upstream-setup/setup.sh && install_sol_ltx25 && rm -rf /tmp/uv-cache /root/.cache
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="sol-engine LTX-2.5 RTX5090 (LTX-2 fd4ded7 + Sol-Attn)"
