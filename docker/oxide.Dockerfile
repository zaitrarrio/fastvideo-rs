# syntax=docker/dockerfile:1.7
# Rust-to-PTX images. Latest stable CUDA in NVIDIA's ubuntu2204 apt repo
# is cuda-toolkit 13.4.2 (packages cuda-*-13-4). Compile needs no GPU.
#
#   base     CUDA 13.4 runtime: cudart, nvrtc, curand, cublas, nvvm, tileiras
#   build    base + nvcc, headers, LLVM 21, nightly rustc; compiles the packages
#   runtime  base + llc-21 + that same nightly rustc + the compiled artifacts
#
#   docker build --target base    -t fastvideo-oxide-base:cu134    .
#   docker build --target build   -t fastvideo-oxide-build:cu134   .
#   docker build --target runtime -t fastvideo-oxide-runtime:cu134 .

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-13.4 \
    CUDA_PATH=/usr/local/cuda-13.4 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda-13.4 \
    CUDARC_CUDA_VERSION=13000 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-cudart-13-4 \
        cuda-nvrtc-13-4 \
        libcurand-13-4 \
        libcublas-13-4 \
        libnvvm-13-4 \
        cuda-tileiras-13-4 \
    && ln -sfn /usr/local/cuda-13.4 /usr/local/cuda \
    && echo /usr/local/cuda/lib64 > /etc/ld.so.conf.d/cuda.conf \
    && echo /usr/local/cuda/nvvm/lib64 >> /etc/ld.so.conf.d/cuda.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64

FROM base AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gnupg git pkg-config libssl-dev xz-utils \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-jammy-21.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-21 libclang-common-21-dev lld-21 llvm-21 llvm-21-dev \
        cuda-nvcc-13-4 \
        cuda-nvrtc-dev-13-4 \
        cuda-cudart-dev-13-4 \
        cuda-crt-13-4 \
        cuda-driver-dev-13-4 \
        libcurand-dev-13-4 \
    && ln -sfn /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
        /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 \
    && rm -rf /var/lib/apt/lists/*

# nightly-2026-04-03 matches third_party/cuda-oxide/rust-toolchain.toml.
# The driver stub is on LD_LIBRARY_PATH so cargo-oxide can start; this stage
# compiles, it does not run device code.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/lib/llvm-21/bin:/usr/local/cargo/bin:${PATH} \
    CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        --profile minimal \
        --default-toolchain nightly-2026-04-03 \
        --component rust-src \
        --component rustc-dev \
        --component rust-analyzer \
        --component clippy \
        --component llvm-tools \
    && rustc --version && nvcc --version && llc-21 --version

WORKDIR /src
COPY third_party/cuda-oxide /src/third_party/cuda-oxide
COPY third_party/cutile-rs /src/third_party/cutile-rs
COPY crates/fastvideo-oxide-kernels /src/crates/fastvideo-oxide-kernels

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/oxide-target \
    bash -euo pipefail -c '\
      export CARGO_TARGET_DIR=/oxide-target/cuda-oxide; \
      cd /src/third_party/cuda-oxide; \
      cargo oxide setup; \
      install -D "$CARGO_TARGET_DIR/debug/librustc_codegen_cuda.so" /opt/oxide/lib/librustc_codegen_cuda.so; \
      export CARGO_TARGET_DIR=/oxide-target/cutile; \
      cd /src/third_party/cutile-rs; \
      cargo build --locked --release; \
      mkdir -p /opt/oxide/cutile /opt/oxide/cubins; \
      find "$CARGO_TARGET_DIR/release" -maxdepth 1 -type f \
        ! -name "*.d" ! -name "*.rmeta" \
        -exec cp -a {} /opt/oxide/cutile/ \;; \
      export CARGO_TARGET_DIR=/oxide-target/fastvideo-oxide-kernels; \
      export CUDA_OXIDE_BACKEND=/opt/oxide/lib/librustc_codegen_cuda.so; \
      cd /src/crates/fastvideo-oxide-kernels; \
      echo "oxide: cargo oxide build --arch sm_100,sm_120"; \
      if cargo oxide build --arch sm_100,sm_120; then :; \
      else cargo oxide build --arch sm_100; cargo oxide build --arch sm_120; fi \
        || echo "oxide: Tile-IR cubin build skipped"; \
      find /oxide-target /src/crates/fastvideo-oxide-kernels -name "*.cubin" \
        -exec cp -a {} /opt/oxide/cubins/ \; || true; \
      ls -lh /opt/oxide/lib /opt/oxide/cutile /opt/oxide/cubins \
    '

FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils gnupg \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-jammy-21.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends llvm-21 \
    && rm -rf /var/lib/apt/lists/*

# Same nightly the backend was built against, without the rustc-dev headers.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/lib/llvm-21/bin:/usr/local/cargo/bin:${PATH} \
    CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    CUDA_OXIDE_BACKEND=/opt/oxide/lib/librustc_codegen_cuda.so

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        --profile minimal \
        --default-toolchain nightly-2026-04-03 \
    && rustc --version && llc-21 --version

COPY --from=build /opt/oxide /opt/oxide

WORKDIR /opt/oxide
CMD ["bash"]
