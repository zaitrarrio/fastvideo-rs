# ADR-0002: Real GPU inference on Vast CUDA hardware

- Status: Accepted
- Date: 2026-09-09
- Decision-id: FVID-2026-09-09-vast-gpu

## Context

Phase 1 proved the Wan graph on Candle CPU with `--tiny` zero weights. That is
not a product bar: 1.3B UMT5+DiT+VAE will not run in useful time on a laptop,
and this Mac cannot compile Candle's CUDA kernels.

The account already has Vast.ai GPU capacity. A running RTX 4090
(`loom-bench-rtx-4090`) is the bring-up host.

## Decision

- Target hardware for real inference is **Vast.ai NVIDIA CUDA** (24GB+ VRAM).
- Candle is built with `--features cuda`. GPU default dtype is **BF16**.
- Mac/CI stay on CPU (`cargo test` without the cuda feature).
- First GPU gate is tiny generate on CUDA; 1.3B Diffusers weights follow on the
  same box.

## Consequences

- CUDA toolkit (`nvcc`) must exist on the Vast image. Runtime PyTorch images
  need `scripts/vast-setup-cuda.sh`.
- A fourth backend or Metal path is a later choice, not a substitute for this
  gate.
- Weights live on the instance disk, not in this git repo.

Verified: tiny Candle CUDA generate (`--dtype f32`) wrote PNG frames on
RTX 4090 instance `50416610`.
