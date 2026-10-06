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
# oxide    Tile-IR NVFP4 GEMM cubins (cutile-rs + tileiras, no GPU) for sm_100/120.
# build    Compiles the release binary from the repo (CI path), embedding them.
# binary   The binary + build id. Locally overridden with
#          `--build-context binary=artifacts/gpucheck/dist` to reuse `docker.sh dist`.
# runtime  What a GPU box runs (ghcr.io/zaitrarrio/fastvideo-rs-runtime): the
#          shared base-cuda stack (pinned CUDA 13.4 libraries,
#          scripts/gpu/cuda-13.pins) + CUPTI + ssh/rsync/curl/hf-fm + the
#          binary and scripts, in thin layers. Layout: "Runtime images" below.
#
# Prebuilt binaries (CI default, docs/serve/images.md "Prebuilt binaries"):
# the build pod compiles every Rust binary and the oxide cubins
# (scripts/dev/build-pod.sh release-artifacts <sha> -> R2), and the workflows
# pass the downloaded directories as named build contexts that REPLACE the
# compile stages, so only the lean stages below run and nothing compiles:
#   --build-context oxide=<dir>          <dir>/out/oxide/        (stage oxide)
#   --build-context binary=<dir>         <dir>/fv-gpucheck{,.build-id} (binary)
#   --build-context hf-fm=<dir>          <dir>/out/hf-fm, hf-fetch-model (hf-fm)
#   --build-context serve-build=<dir>    <dir>/out/fv-serve{,.features} (serve-build)
#   --build-context cpu-build=<dir>      <dir>/out/fv-serve{,.features} (cpu-build)
# Without them (no R2 key, artifacts not built yet) the stages compile here.

# The base of every runtime image, pinned so the shared layers only change
# when this line does (scripts/ci/base-images.sh hashes it).
ARG UBUNTU=ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401

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
# Rust: install exactly what rust-toolchain.toml asks for (channel, components,
# targets) in this one layer, then pin later layers to that toolchain. Without
# this, the first `cargo` in a later layer sees a target the toolchain file
# lists but the image lacks (wasm32), syncs `stable` to the newest release and
# fails renaming files that live in a lower overlayfs layer ("Invalid
# cross-device link", os error 18). RUSTUP_TOOLCHAIN stops any later sync;
# RUSTUP_PERMIT_COPY_RENAME makes rustup copy instead of failing if one ever
# happens anyway.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUSTUP_PERMIT_COPY_RENAME=1
COPY rust-toolchain.toml /etc/fastvideo/rust/rust-toolchain.toml
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain none \
 && cd /etc/fastvideo/rust && rustup toolchain install \
 && rustup default stable \
 && rustc --version && cargo --version && rustup target list --installed
ENV RUSTUP_TOOLCHAIN=stable
ENV CUDARC_CUDA_VERSION=13000 \
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

# Tile-IR NVFP4 GEMM cubins for sm_100 and sm_120, compiled ahead of time
# with no GPU: fv-oxide-aot (crates/fastvideo-oxide-kernels) runs cutile-rs's
# compile-only KernelCompiler, then tileiras. Only the kernel crate and the
# vendored cutile-rs are copied in, so the registry cache keeps this layer
# until one of them changes. cutile's bindgen needs cuda.h and curand.h.
# Any failure fails the image build (build.rs also checks FV_REQUIRE_OXIDE).
FROM builder AS oxide
RUN apt-get update \
 && apt-get install -y --no-install-recommends libclang-dev \
      cuda-driver-dev-13-4 cuda-cudart-dev-13-4 libcurand-dev-13-4 \
 && rm -rf /var/lib/apt/lists/*
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda-13.4
COPY third_party/cutile-rs /oxide/third_party/cutile-rs
COPY crates/fastvideo-oxide-kernels /oxide/crates/fastvideo-oxide-kernels
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/oxide-target \
    test -f /oxide/third_party/cutile-rs/Cargo.toml \
 && cd /oxide/crates/fastvideo-oxide-kernels \
 && CARGO_TARGET_DIR=/oxide-target cargo build --release --locked \
 && /oxide-target/release/fv-oxide-aot /out/oxide --sm 100,120 \
 && test -s /out/oxide/manifest.tsv \
 && cat /out/oxide/manifest.tsv

# `docker buildx build --target oxide-out --output type=local,dest=artifacts/oxide`
FROM scratch AS oxide-out
COPY --from=oxide /out/oxide/ /

FROM builder AS build
ARG BUILD_ID=unknown
COPY . /src
COPY --from=oxide /out/oxide /oxide-cubins
ENV FV_OXIDE_CUBIN_DIR=/oxide-cubins \
    FV_REQUIRE_OXIDE=100,120
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/target \
    cargo build --release -p fastvideo-gpucheck --features cuda \
 && mkdir -p /out \
 && cp /target/release/fv-gpucheck /out/fv-gpucheck \
 && echo "$BUILD_ID" > /out/fv-gpucheck.build-id

FROM scratch AS binary
COPY --from=build /out/ /

# ============================================================================
# Runtime images: one shared layer stack (docs/serve/images.md "Layers").
#
#   ubuntu:22.04 (pinned digest)        every image
#   base-os: ca-certificates + x264/vpx/dav1d runtime libs (1 layer)
#            + the minimal ffmpeg/ffprobe build below (1 layer)
#   base-cuda: CUDA 13.4 runtime libraries, four stable layers, largest first:
#            cuBLASLt | cuDNN precompiled engines | cuDNN core | cuBLAS + NVRTC
#   ── base-os / base-cuda end here. CI publishes them as
#      fastvideo-rs-runtime:base-{os,cuda}-<hash> (scripts/ci/base-images.sh)
#      and passes them back as named build contexts, so every image below,
#      from any workflow, sits on the SAME layer digests and a host that
#      pulled one image already has ~95 % of the next.
#   serve-cuda-bin: fv-serve (one layer) -> serve-<variant>: config (KBs)
#   serve-cpu: base-os + fv-serve without CUDA
#   runtime (fv-gpucheck, benchmarks): base-cuda + debug tools (sshd, rsync,
#            curl) + CUPTI + hf-fm + scripts/gpu + oxide + fv-gpucheck
#   serve (`debug` flavour, fastvideo-rs-serve:latest/:sha-…): runtime +
#            fv-serve + every config + deploy/vast/worker.py
#
# Not shipped anywhere (measured in docs/serve/images.md): CUDA headers,
# static libraries, docs, dpkg metadata and the NVIDIA apt keyring;
# tileiras (only the AOT oxide build uses it; its package drags in nvcc,
# libnvvm, libnvjitlink and build-essential); Ubuntu's ffmpeg and its ~190
# shared-library packages; libcudnn_adv (legacy RNN / multi-head attention /
# CTC API) and libcudnn_ext (causal-conv1d / GNN ops); libnvblas; CUPTI's
# checkpoint / PC-sampling libraries; Python; compilers.
#
# >>> shared base: everything from here to "<<< shared base", the UBUNTU
# ARG and scripts/gpu/cuda-13.pins are hashed into the base image tag
# (scripts/ci/base-images.sh hash). Keep image-specific steps out of it.

# ---- ffmpeg: FFmpeg 4.4 (the release Ubuntu 22.04 ships, so the CLI behaves
# the same), every native component, external libraries limited to what the
# code asks for: libx264 (cpu-test-x264 fallback, LTX I2V conditioning),
# libvpx (VP8 for WebRTC peers), libdav1d (AV1 input decode) and NVENC
# (ffnvcodec: the nv-codec-headers Ubuntu built its ffmpeg with, 11.1: driver >=
# 470). Nothing autodetected, so no X11/SDL/VA-API/Pulse/... dependencies.
FROM ${UBUNTU} AS ffmpeg-build
ARG DEBIAN_FRONTEND=noninteractive
ARG FFMPEG_VERSION=4.4.5
ARG FFMPEG_SHA256=f9514e0d3515aee5a271283df71636e1d1ff7274b15853bcd84e144be416ab07
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config nasm curl ca-certificates xz-utils \
      libx264-dev libvpx-dev libdav1d-dev zlib1g-dev libffmpeg-nvenc-dev \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
 && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
 && mkdir /tmp/ffmpeg && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/ffmpeg --strip-components=1 \
 && cd /tmp/ffmpeg \
 && ./configure --prefix=/opt/ffmpeg --disable-autodetect --disable-debug --disable-doc --disable-ffplay \
      --disable-shared --enable-static --enable-gpl --enable-pthreads --enable-zlib \
      --enable-libx264 --enable-libvpx --enable-libdav1d \
      --enable-ffnvcodec --enable-nvenc \
 && make -j"$(nproc)" && make install \
 && strip /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe \
 && mkdir -p /out && cp /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe /out/ \
 && /out/ffmpeg -hide_banner -encoders | grep -E ' (h264_nvenc|libx264|libvpx|aac) ' \
 && rm -rf /tmp/ffmpeg /tmp/ffmpeg.tar.xz


# ---- CUDA runtime libraries, collected into /out/<layer>/ (copied into
# base-cuda without apt, the keyring or dpkg metadata). Only what fv-serve and
# fv-gpucheck load: cudarc dlopens libnvrtc, libcublas(+Lt), libcudnn and
# (FASTVIDEO_GPU_TRACE only) libcupti; libcudnn dlopens its graph/ops/cnn/
# heuristic/engine libraries, and the runtime-compiled and tensor-IR engines
# dlopen libnvrtc. (CUPTI: the cupti-libs stage, outside the shared base.)
FROM ${UBUNTU} AS cuda-libs
ARG DEBIAN_FRONTEND=noninteractive
COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && . /etc/fastvideo/cuda-13.pins \
 && apt-get install -y --no-install-recommends --allow-downgrades \
      "$CUDA_NVRTC_PKG" "$CUDA_CUBLAS_PKG" "$CUDA_CUDNN_PKG" \
 && rm -rf /var/lib/apt/lists/* \
 && L=/usr/local/cuda-13.4/targets/x86_64-linux/lib G=/usr/lib/x86_64-linux-gnu \
 && mkdir -p /out/cublaslt /out/cudnn-engines /out/cudnn /out/cublas-nvrtc \
 && cp -a "$L"/libcublasLt.so.* /out/cublaslt/ \
 && ln -s "$CUDA_CUBLASLT_SONAME" /out/cublaslt/libcublasLt.so \
 && cp -a "$G"/libcudnn_engines_precompiled.so.* /out/cudnn-engines/ \
 && for l in libcudnn libcudnn_graph libcudnn_ops libcudnn_cnn libcudnn_heuristic \
             libcudnn_engines_runtime_compiled libcudnn_engines_tensor_ir; do \
      cp -a "$G/$l".so.* /out/cudnn/; \
    done \
 && ln -s "$CUDA_CUDNN_SONAME" /out/cudnn/libcudnn.so \
 && cp -a "$L"/libcublas.so.* "$L"/libnvrtc.so.* "$L"/libnvrtc-builtins.so.* /out/cublas-nvrtc/ \
 && ln -s "$CUDA_CUBLAS_SONAME" /out/cublas-nvrtc/libcublas.so \
 && ln -s "$CUDA_NVRTC_SONAME" /out/cublas-nvrtc/libnvrtc.so \
 && ls -la /out/* \
 && du -sh /out/*

# ---- base-os: what every runtime image (CUDA or not) runs on.
FROM ${UBUNTU} AS base-os
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libx264-163 libvpx7 libdav1d5 \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/cache/debconf/*-old /var/lib/dpkg/*-old \
      /var/log/dpkg.log /var/log/apt /var/log/alternatives.log \
 && mkdir -p /fvstate /var/log
COPY --from=ffmpeg-build /out/ffmpeg /out/ffprobe /usr/local/bin/
RUN ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=black:s=64x48:r=24:d=0.1 -c:v libvpx -f ivf -y /dev/null \
 && ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=black:s=64x48:r=24:d=0.1 -f lavfi -i sine=d=0.1 \
      -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart -y /tmp/t.mp4 \
 && ffprobe -v error -show_entries stream=codec_name -of csv=p=0 /tmp/t.mp4 | grep -qx h264 \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q ' h264_nvenc ' \
 && rm /tmp/t.mp4
ENV FV_STATE_DIR=/fvstate \
    RUST_LOG=info
LABEL org.opencontainers.image.source="https://github.com/zaitrarrio/fastvideo-rs" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/fastvideo-rs

# ---- base-cuda: base-os + the CUDA 13.4 runtime libraries (not CUPTI, which
# only the runtime image adds). The pins file stays for provenance and
# scripts/gpu/remote.sh.
FROM base-os AS base-cuda
COPY --from=cuda-libs /out/cublaslt/ /usr/local/cuda-13.4/lib64/
COPY --from=cuda-libs /out/cudnn-engines/ /usr/lib/x86_64-linux-gnu/
COPY --from=cuda-libs /out/cudnn/ /usr/lib/x86_64-linux-gnu/
COPY --from=cuda-libs /out/cublas-nvrtc/ /usr/local/cuda-13.4/lib64/
COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
# /usr/local/cuda and targets/x86_64-linux/lib: the paths of NVIDIA's apt
# layout, which older scripts and boxes look in (symlinks, no data).
RUN echo /usr/local/cuda-13.4/lib64 > /etc/ld.so.conf.d/fastvideo-nvidia.conf \
 && ln -s cuda-13.4 /usr/local/cuda \
 && mkdir -p /usr/local/cuda-13.4/targets/x86_64-linux \
 && ln -s ../../lib64 /usr/local/cuda-13.4/targets/x86_64-linux/lib \
 && ldconfig \
 && ldconfig -p | grep -E 'libnvrtc\.so|libcublasLt\.so|libcublas\.so|libcudnn\.so'
# The NVIDIA container runtime injects the driver (libcuda, and libnvidia-encode
# for NVENC through the `video` capability) when these are set.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# <<< shared base

# ---- CUPTI (runtime image only), from its own small package, so building the
# runtime on a published base-cuda does not fetch the CUDA library packages.
# libcupti + the nvperf libraries it dlopens for the profiling APIs; not the
# static libraries, libcheckpoint or libpcsamplingutil.
FROM ${UBUNTU} AS cupti-libs
ARG DEBIAN_FRONTEND=noninteractive
COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && . /etc/fastvideo/cuda-13.pins \
 && apt-get install -y --no-install-recommends --allow-downgrades "$CUDA_CUPTI_PKG" \
 && rm -rf /var/lib/apt/lists/* \
 && L=/usr/local/cuda-13.4/targets/x86_64-linux/lib \
 && mkdir -p /out \
 && cp -a "$L"/libcupti.so* "$L"/libnvperf_host.so "$L"/libnvperf_target.so /out/ \
 && ls -la /out

# ---- runtime: what a GPU box runs for fv-gpucheck, benchmarks and remote
# debugging (ghcr.io/zaitrarrio/fastvideo-rs-runtime). base-cuda + thin layers,
# most stable first. No Python, no PyTorch, no toolkit: hosts boot quickly.
# scripts/gpu/remote.sh needs ssh/rsync (Vast boxes), curl (fetch-baseline);
# `ldd`, tar and gzip come with Ubuntu. NVENC needs the driver's `video`
# capability (base-cuda sets it).
FROM base-cuda AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server rsync curl \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/log/dpkg.log /var/log/apt \
 && mkdir -p /run/sshd
# CUPTI for FASTVIDEO_GPU_TRACE (cudarc `cupti`, loaded lazily; fv-serve's
# variant images go without it).
COPY --from=cupti-libs /out/ /usr/local/cuda-13.4/lib64/
RUN ldconfig && ldconfig -p | grep -q 'libcupti\.so\.13 '
COPY --from=hf-fm /out/hf-fm /out/hf-fetch-model /usr/local/bin/
COPY scripts/gpu /opt/fastvideo-rs/scripts/gpu
# Provenance only: the cubins are already embedded in fv-gpucheck.
COPY --from=oxide /out/oxide /opt/fastvideo-rs/oxide
COPY --from=binary /fv-gpucheck /fv-gpucheck.build-id /opt/fastvideo-rs/target/release/
LABEL org.opencontainers.image.description="fastvideo-rs cudarc GPU validation runtime (fv-gpucheck + CUDA runtime + hf-fm)"
WORKDIR /opt/fastvideo-rs

# ---- serve: fv-serve on the runtime image (docs/serve/design.md §6.1, WP-16) --
# serve-build compiles fv-serve with the CUDA backend and outbound HTTP (the
# Runpod queue worker, D1, R2) on top of the `build` stage, sharing its
# cargo caches. FV_SERVE_FEATURES never includes `encoders` (OpenH264 is a
# CPU-only test backend and is not shipped, design §0 decision 1).
FROM build AS serve-build
ARG FV_SERVE_FEATURES=cuda,http-client
# Build identity for `fv-serve --version` and /health (crates/fastvideo-serve/
# build.rs; .git is not in the context). CI passes the commit sha and time.
ARG BUILD_ID=unknown
ARG FV_GIT_SHA=unknown
ARG FV_BUILD_TIME=
# The Reactor adapter (a default feature) encodes Opus: audiopus_sys builds
# libopus statically with CMake.
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/target \
    cargo build --release -p fastvideo-serve --features "$FV_SERVE_FEATURES" \
 && mkdir -p /out \
 && cp /target/release/fv-serve /out/fv-serve \
 && echo "$FV_SERVE_FEATURES" > /out/fv-serve.features


# serve: the `debug` flavour (ghcr.io/zaitrarrio/fastvideo-rs-serve:latest /
# :sha-…; the legacy all-in-one tags): the runtime image (ssh, rsync,
# fv-gpucheck, scripts, CUPTI) + fv-serve + every config + the Vast worker.
# What Vast instances, CloudRift/GCP workers and profiling pods run; Runpod
# pods and serverless workers run the per-variant images below. It adds only
# its top layers to the shared stack (docs/serve/images.md).
# Ports: 8000/http. The ICE ports of design §6.1 (70000/tcp, 70010/udp) are
# symmetric platform requests above 65535, so they are published by the
# deploy scripts, not EXPOSEd.
FROM runtime AS serve
ENV FV_CONFIG=/etc/fv/runpod.toml
COPY configs/serve /etc/fv
COPY deploy/vast/worker.py /opt/fastvideo-rs/deploy/vast/worker.py
COPY --from=serve-build /out/fv-serve /out/fv-serve.features /opt/fastvideo-rs/bin/
LABEL org.opencontainers.image.description="fv-serve: FastVideo, MiniMax, fal and LTX APIs over the fastvideo-rs CUDA engines"
EXPOSE 8000
ENTRYPOINT ["/opt/fastvideo-rs/bin/fv-serve"]

# ---- serve-cuda-bin: the CUDA fv-serve (serve-build above) on base-cuda.
FROM base-cuda AS serve-cuda-bin
COPY --from=serve-build /out/fv-serve /out/fv-serve.features /opt/fastvideo-rs/bin/
COPY deploy/runpod/fv-entry.sh /opt/fastvideo-rs/bin/fv-entry
EXPOSE 8000
ENTRYPOINT ["/opt/fastvideo-rs/bin/fv-entry"]

# ---- one thin stage per variant: its config (plus the fake config for the
# CI smoke check) and the variant's identity for the entrypoint.
FROM serve-cuda-bin AS serve-h3-turbo
COPY configs/serve/runpod.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=h3-turbo FV_CONFIG=/etc/fv/runpod.toml
LABEL org.opencontainers.image.description="fv-serve h3-turbo (configs/serve/runpod.toml)"

FROM serve-cuda-bin AS serve-h3-max
COPY configs/serve/runpod-h3-max.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=h3-max FV_CONFIG=/etc/fv/runpod-h3-max.toml
LABEL org.opencontainers.image.description="fv-serve h3-max (configs/serve/runpod-h3-max.toml)"

FROM serve-cuda-bin AS serve-ltx
COPY configs/serve/runpod-ltx.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=ltx FV_CONFIG=/etc/fv/runpod-ltx.toml
LABEL org.opencontainers.image.description="fv-serve ltx (configs/serve/runpod-ltx.toml)"

FROM serve-cuda-bin AS serve-wan
COPY configs/serve/runpod-wan.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=wan FV_CONFIG=/etc/fv/runpod-wan.toml
LABEL org.opencontainers.image.description="fv-serve wan (configs/serve/runpod-wan.toml)"

# Also carries runpod-wan14b.toml (the Wan 14B fast tier, same binary):
# FV_CONFIG=/etc/fv/runpod-wan14b.toml selects it.
FROM serve-cuda-bin AS serve-wan5b
COPY configs/serve/runpod-wan5b.toml configs/serve/runpod-wan14b.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=wan5b FV_CONFIG=/etc/fv/runpod-wan5b.toml
LABEL org.opencontainers.image.description="fv-serve wan5b (configs/serve/runpod-wan5b.toml)"

FROM serve-cuda-bin AS serve-sfwan
COPY configs/serve/runpod-sfwan.toml configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=sfwan FV_CONFIG=/etc/fv/runpod-sfwan.toml
LABEL org.opencontainers.image.description="fv-serve sfwan (configs/serve/runpod-sfwan.toml)"


# ---- cpu: CPU only. fv-serve without `cuda` (no CUDA libraries at all) on
# base-os, with the fake engine: fv-control's fake-engine CPU workers
# (tiny-cpu) and tests. (It replaced the retired gateway image,
# docs/serve/edge-control-plane.md §9 stage 4.)
FROM build AS cpu-build
ARG FV_CPU_FEATURES=http-client
ARG BUILD_ID=unknown
ARG FV_GIT_SHA=unknown
ARG FV_BUILD_TIME=
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/target \
    cargo build --release -p fastvideo-serve --features "$FV_CPU_FEATURES" \
 && mkdir -p /out \
 && cp /target/release/fv-serve /out/fv-serve \
 && echo "$FV_CPU_FEATURES" > /out/fv-serve.features


FROM base-os AS serve-cpu
COPY --from=cpu-build /out/fv-serve /out/fv-serve.features /opt/fastvideo-rs/bin/
COPY deploy/runpod/fv-entry.sh /opt/fastvideo-rs/bin/fv-entry
COPY configs/serve/runpod-fake.toml /etc/fv/
ENV FV_VARIANT=cpu FV_CONFIG=/etc/fv/runpod-fake.toml
LABEL org.opencontainers.image.description="fv-serve cpu: fake engine, CPU only (configs/serve/runpod-fake.toml)"
EXPOSE 8000
ENTRYPOINT ["/opt/fastvideo-rs/bin/fv-entry"]
