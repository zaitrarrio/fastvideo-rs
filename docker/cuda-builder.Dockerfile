# Linux + CUDA 12.4 builder. Compile only — no GPU required.
# Matches Vast Ubuntu 22.04 / CUDA 12.4 so the binary can be copied onto the 4090.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl ca-certificates \
    libssl-dev clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:${PATH} \
    RUSTUP_TOOLCHAIN=1.82.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.82.0 \
    && rustc --version && nvcc --version

WORKDIR /src
CMD ["bash"]
