# Linux builder for the vendored Rust-to-PTX packages. Compile only — no GPU.
#
# Same Ubuntu 22.04 + CUDA 13.0 apt pin as docker/cuda-builder.Dockerfile, plus
# the LLVM 21 / libclang pair cuda-oxide's bindgen expects and the nightly
# rustc-codegen-cuda is pinned to (third_party/cuda-oxide/rust-toolchain.toml).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl gnupg git pkg-config \
        libssl-dev xz-utils \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" \
        > /etc/apt/sources.list.d/llvm-toolchain-jammy-21.list \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-21 libclang-common-21-dev lld-21 llvm-21 llvm-21-dev \
        cuda-nvcc-13-0 cuda-nvrtc-13-0 cuda-nvrtc-dev-13-0 \
        cuda-cudart-dev-13-0 cuda-crt-13-0 cuda-driver-dev-13-0 \
    && ln -sfn /usr/local/cuda-13.0 /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*

# nightly-2026-04-03 matches third_party/cuda-oxide/rust-toolchain.toml so a
# bind-mounted tree does not download the compiler again at run time.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/lib/llvm-21/bin:/usr/local/cuda/bin:/usr/local/cargo/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda-13.0 \
    CUDA_PATH=/usr/local/cuda-13.0 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 \
    CUDA_OXIDE_LLC=/usr/bin/llc-21 \
    LIBCLANG_PATH=/usr/lib/llvm-21/lib \
    LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
    CUDARC_CUDA_VERSION=13000
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
        --profile minimal \
        --default-toolchain nightly-2026-04-03 \
        --component rust-src \
        --component rustc-dev \
        --component rust-analyzer \
        --component clippy \
        --component llvm-tools \
    && rustc --version && nvcc --version && llc-21 --version

# cutile-rs cuda-bindings bindgen includes curand.h. It is not part of the
# nvcc/nvrtc set the other builders install.
RUN apt-get update && apt-get install -y --no-install-recommends libcurand-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

# cargo-oxide is linked against libcuda.so.1. The driver-dev package ships
# the link stub only (this image compiles; it does not run device code).
RUN ln -sfn /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so \
        /usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}

WORKDIR /src
CMD ["bash"]
