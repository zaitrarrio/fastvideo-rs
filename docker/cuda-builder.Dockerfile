# Linux + CUDA 13.0 builder. Compile only — no GPU required.
# Built from ubuntu:22.04 plus NVIDIA's apt packages rather than an
# nvidia/cuda image tag, so the CUDA version is pinned in one place we control.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl wget ca-certificates \
    libssl-dev clang libclang-dev \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-nvrtc-13-0 cuda-nvrtc-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda-13.0/bin:${PATH} \
    NVCC=/usr/local/cuda-13.0/bin/nvcc \
    CUDARC_CUDA_VERSION=13000 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustc --version && nvcc --version

WORKDIR /src
CMD ["bash"]
