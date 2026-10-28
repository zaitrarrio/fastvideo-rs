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
#   sol-ltx25      Lightricks/LTX-2 @ fd4ded7 (uv sync; torch 2.13 cu132; PyAV pinned, setup.sh
#                  LTX25_AV_VERSION) + nvidia-cutlass-dsl + cuda-python + Sol-Attn
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

FROM ${BASE_IMAGE} AS base-os
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

FROM base-os AS base
COPY --from=fvrs /opt/fastvideo-rs/target/release/fv-gpucheck /usr/local/bin/fv-gpucheck

# FastH3 NVFP4 on sm_120 (docs/ports/fasth3-nvfp4.md): FastVideo @ FV_NVFP4_REV with fastvideo-kernel built for
# sm_120a (SageAttention3 dense / block-sparse FP4 attention), and the whole fastvideo-rs runtime so one pod runs
# both sides. The runtime's CUDA 13.4 libraries (cuBLASLt NVFP4, cuDNN, NVRTC) go to /opt/fvrs-cuda, off the
# default library path, so torch and FlashInfer keep the base image's CUDA 13.0: our runs set LD_LIBRARY_PATH.
# The runtime comes last, so a new FVRS_RUNTIME rebuilds only these layers, not the kernel build.
FROM base-os AS fastvideo-nvfp4
RUN . /opt/upstream-setup/setup.sh && install_fastvideo_nvfp4 && rm -rf /tmp/uv-cache /root/.cache
COPY --from=fvrs /usr/local/cuda-13.4/lib64/ /opt/fvrs-cuda/lib64/
COPY --from=fvrs /usr/lib/x86_64-linux-gnu/libcudnn* /opt/fvrs-cuda/cudnn/
COPY --from=fvrs /opt/fastvideo-rs /opt/fastvideo-rs
RUN ln -s /opt/fastvideo-rs/target/release/fv-gpucheck /usr/local/bin/fv-gpucheck
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="FastVideo FastH3 NVFP4 (sm_120 FP4 kernels) + the fastvideo-rs runtime"

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
# The retake fixture, bind-mounted for this step only (not kept in the image): install_sol_ltx25's
# check_ltx25_av decodes it with upstream's own decode_audio_from_file and fails the build unless PyAV is the
# pinned one (setup.sh LTX25_AV_VERSION) and the wave is [2, 222208].
RUN --mount=type=bind,source=scripts/gpu/fixtures/beach-push-768x512-24fps.mp4,target=/opt/fixtures/beach-push-768x512-24fps.mp4 \
    export LTX25_AUDIO_FIXTURE=/opt/fixtures/beach-push-768x512-24fps.mp4 \
 && . /opt/upstream-setup/setup.sh && install_sol_ltx25 && rm -rf /tmp/uv-cache /root/.cache
COPY scripts/gpu/upstream /opt/fvrs/scripts/gpu/upstream
LABEL org.opencontainers.image.description="sol-engine LTX-2.5 RTX5090 (LTX-2 fd4ded7 + Sol-Attn)"
