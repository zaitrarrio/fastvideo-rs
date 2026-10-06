# syntax=docker/dockerfile:1.7
# fv-build base image: everything the shared CPU build pod (docs/dev/build-pod.md)
# needs, so the pod installs nothing when it boots.
#
#   ghcr.io/zaitrarrio/fastvideo-rs-build-base:<tag>
#
# <tag> is a content hash of this file and its inputs (scripts/dev/build-base-tag.sh:
# this Dockerfile, scripts/gpu/cuda-13.pins, rust-toolchain.toml). The workflow
# .github/workflows/build-base-image.yml builds and pushes it when that hash has no
# image yet; scripts/dev/build-pod.sh pins the tag (FV_BUILD_IMAGE), and the
# workflow fails when the pin and the hash differ. System packages change only
# here: edit, run `bash scripts/dev/build-base-tag.sh --pin`, push.
#
# Packages only: no crate of this repository is compiled here.
#
#   Ubuntu 22.04 (glibc 2.35, the same as the published runtime images, so the
#     release artifacts need no glibc downgrade check to pass)
#   CUDA 13.4.92 nvcc / NVRTC / tileiras (scripts/gpu/cuda-13.pins, the CI
#     builder's pins) + cudart/driver/cuRAND headers (oxide's bindgen)
#   Rust $RUST_VERSION with what rust-toolchain.toml lists (rustfmt, clippy,
#     wasm32-unknown-unknown); RUSTUP_TOOLCHAIN pins it, so `stable` in the
#     toolchain file never triggers a download on the pod
#   sccache, mold, clang/libclang, cmake, pkg-config
#   python3 (3.11) + venv, ffmpeg 6.0.1 (static: libx264, libvpx, libopus), Node, Playwright + its Chromium and
#     Chromium's system libraries (tests/compat, tests/console, FV_SERVE_UI=1)
#   jq, curl, binutils, zstd, xz (release-artifacts-pod.sh, the deps seeds)
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
# Pins. Bump one, then `bash scripts/dev/build-base-tag.sh --pin`.
ARG RUST_VERSION=1.99.0
ARG SCCACHE_VERSION=0.18.0
ARG SCCACHE_SHA256=45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89
ARG MOLD_VERSION=3.0.0
# mold publishes no checksum file; sha256 of the release tarball, 2026-10-06.
ARG MOLD_SHA256=6c90d4a474c7c0409dfb575be03a5345878ac14fdba18de8b40fa58c60121189
ARG NODE_VERSION=v22.23.3
ARG NODE_SHA256=df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de
# tests/compat/package.json pins the same Playwright.
ARG PLAYWRIGHT_VERSION=1.56.1
# ffmpeg: not Ubuntu 22.04's 4.4.2. Its decoder holds frames back under the
# low-latency decode flags fastvideo-media uses (fastvideo-media decode_pipe,
# fastvideo-webrtc ingest, fastvideo-reactor duplex, fastvideo-serve
# ingest_whip failed with it; they pass with Debian's 5.1, Ubuntu 24.04's 6.1
# and this static 6.0.1). Static build from johnvansickle.com (old-releases is
# a fixed URL); sha256 of the tarball, 2026-10-06.
ARG FFMPEG_VERSION=6.0.1
ARG FFMPEG_SHA256=28268bf402f1083833ea269331587f60a242848880073be8016501d864bd07a5
# python3 on PATH: CPython 3.11 (python-build-standalone; Ubuntu 22.04 has 3.10,
# and tests/compat/requirements.txt needs >= 3.11, e.g. websockets 17).
ARG PYTHON_BUILD=20251014
ARG PYTHON_VERSION=3.11.14
ARG PYTHON_SHA256=d0623c777fb89b904b56cd5aba51af29cbb34b1f9d45f0672f90f6dce30fa93e

COPY scripts/gpu/cuda-13.pins /etc/fastvideo/cuda-13.pins
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config git curl wget ca-certificates gnupg \
      libssl-dev clang libclang-dev \
      xz-utils zstd jq binutils file procps \
      python3 python3-venv python3-pip openssl \
 && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && . /etc/fastvideo/cuda-13.pins \
 && apt-get install -y --no-install-recommends --allow-downgrades \
      "$CUDA_NVCC_PKG" "$CUDA_NVRTC_PKG" "$CUDA_NVRTC_DEV_PKG" "$CUDA_TILEIRAS_PKG" \
      cuda-driver-dev-13-4 cuda-cudart-dev-13-4 libcurand-dev-13-4 \
 && apt-mark hold cuda-nvcc-13-4 cuda-nvrtc-13-4 cuda-nvrtc-dev-13-4 cuda-tileiras-13-4 \
 && rm -rf /var/lib/apt/lists/*

# ffmpeg + ffprobe 6.0.1 (static) in /usr/local/bin; there is no apt ffmpeg.
RUN cd /tmp \
 && curl -fsSL -o ff.tar.xz "https://johnvansickle.com/ffmpeg/old-releases/ffmpeg-${FFMPEG_VERSION}-amd64-static.tar.xz" \
 && echo "${FFMPEG_SHA256}  ff.tar.xz" | sha256sum -c - \
 && mkdir -p ff && tar -xJf ff.tar.xz -C ff --strip-components=1 \
 && install -m 755 ff/ffmpeg ff/ffprobe /usr/local/bin/ \
 && rm -rf ff ff.tar.xz \
 && ffmpeg -hide_banner -version | head -1 \
 && ffmpeg -hide_banner -encoders 2>/dev/null >/tmp/enc && grep -q ' libvpx ' /tmp/enc && grep -q ' libx264 ' /tmp/enc && grep -q ' libopus ' /tmp/enc \
 && ffmpeg -hide_banner -decoders 2>/dev/null >/tmp/dec && grep -q ' vp8 ' /tmp/dec && grep -q ' h264 ' /tmp/dec && rm /tmp/enc /tmp/dec

# sccache and mold: release binaries, sha256-checked.
RUN cd /tmp \
 && curl -fsSL -o sccache.tgz "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
 && echo "${SCCACHE_SHA256}  sccache.tgz" | sha256sum -c - \
 && tar -xzf sccache.tgz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache" \
 && curl -fsSL -o mold.tgz "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz" \
 && echo "${MOLD_SHA256}  mold.tgz" | sha256sum -c - \
 && tar -xzf mold.tgz --strip-components=1 -C /usr/local \
 && rm -f sccache.tgz mold.tgz \
 && sccache --version && mold --version

# Node + Playwright + Chromium (and, through install-deps, its system libraries).
ENV PATH=/opt/node/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    NODE_PATH=/opt/playwright/node_modules
RUN cd /tmp \
 && curl -fsSL -o node.tar.xz "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
 && echo "${NODE_SHA256}  node.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/node && tar -xJf node.tar.xz -C /opt/node --strip-components=1 && rm node.tar.xz \
 && mkdir -p /opt/playwright \
 && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install --prefix /opt/playwright --no-audit --no-fund --loglevel=error "playwright@${PLAYWRIGHT_VERSION}" \
 && /opt/playwright/node_modules/.bin/playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/* /root/.npm \
 && node -e "require('playwright')"

# CPython 3.11 in /opt/python, first on PATH as python3 (after every apt step:
# apt's maintainer scripts keep Ubuntu's /usr/bin/python3, 3.10). On PATH, not
# symlinked: a venv made through a symlink elsewhere has no working ensurepip.
RUN cd /tmp \
 && curl -fsSL -o py.tgz "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_BUILD}/cpython-${PYTHON_VERSION}%2B${PYTHON_BUILD}-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "${PYTHON_SHA256}  py.tgz" | sha256sum -c - \
 && mkdir -p /opt/python && tar -xzf py.tgz -C /opt/python --strip-components=1 && rm py.tgz
ENV PATH=/opt/python/bin:$PATH
RUN python3 -V && python3 -m venv /tmp/v && /tmp/v/bin/python -m pip --version && rm -rf /tmp/v

# Rust: what rust-toolchain.toml lists, at RUST_VERSION. RUSTUP_TOOLCHAIN makes
# every cargo/rustc use it, whatever the toolchain file's channel says (no
# rustup sync on the pod; a new Rust means a new image).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda-13.4/bin:$PATH \
    RUSTUP_PERMIT_COPY_RENAME=1
COPY rust-toolchain.toml /etc/fastvideo/rust/rust-toolchain.toml
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path \
 && cd /etc/fastvideo/rust \
 && sed -i "s/^channel = .*/channel = \"${RUST_VERSION}\"/" rust-toolchain.toml \
 && rustup toolchain install \
 && rustup default "${RUST_VERSION}" \
 && rm -rf /usr/local/cargo/registry /usr/local/rustup/downloads /usr/local/rustup/tmp \
 && rustc -vV && cargo -V && cargo clippy -V && rustfmt -V && rustup target list --installed
ENV RUSTUP_TOOLCHAIN=${RUST_VERSION}

# The build pod's environment (build-pod-server.py reads these).
ENV CUDA_HOME=/usr/local/cuda-13.4 \
    NVCC=/usr/local/cuda-13.4/bin/nvcc \
    CUDARC_CUDA_VERSION=13000 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64 \
    RUSTC_WRAPPER=/usr/local/bin/sccache

# What is in here, for `build-pod.sh status` and the release manifest.
RUN . /etc/fastvideo/cuda-13.pins \
 && jq -n --arg rust "$(rustc -V)" --arg cargo "$(cargo -V)" --arg sccache "$(sccache --version)" \
      --arg mold "$(mold --version | cut -d' ' -f1-2)" --arg clang "$(clang --version | head -1)" \
      --arg cmake "$(cmake --version | head -1)" --arg python "$(python3 -V)" \
      --arg node "$(node -v)" --arg playwright "${PLAYWRIGHT_VERSION}" \
      --arg ffmpeg "$(ffmpeg -version | head -1 | cut -d' ' -f1-3)" \
      --arg nvcc "$(nvcc --version | tail -1)" --arg tileiras "$(tileiras --version 2>&1 | tail -1)" \
      --arg glibc "$(ldd --version | head -1)" \
      '{rust: $rust, cargo: $cargo, sccache: $sccache, mold: $mold, clang: $clang, cmake: $cmake, \
        python: $python, node: $node, playwright: $playwright, ffmpeg: $ffmpeg, nvcc: $nvcc, \
        tileiras: $tileiras, glibc: $glibc}' >/etc/fastvideo/build-base.json \
 && cat /etc/fastvideo/build-base.json

WORKDIR /root
CMD ["bash"]
