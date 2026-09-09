# ADR-0001: TensorBackend trait over Burn, Candle, and Luminal

- Status: Accepted
- Date: 2026-09-09
- Decision-id: FVID-2026-09-09-tensor-backend

## Context

FastVideo is a Python/PyTorch inference and post-training stack. This port must
run the same Wan/FastWan graphs on three Rust ML runtimes:

- Burn (CubeCL / Flex), designed around `Backend` + const-rank `Tensor<B, D>`
- Candle, eager dynamically ranked `Tensor`
- Luminal, a static graph compiler

Burn's own `Backend` trait cannot host Candle or Luminal. `burn-candle` is
deprecated as of Burn 0.21.

## Decision

Introduce `fastvideo_ops::TensorBackend` as the only surface model code may use.
Each runtime crate implements it.

Luminal must compile a **single DiT step** and a **single VAE decode**, then
execute those graphs from the Rust denoising loop. Do not unroll UniPC/DMD into
one static graph.

Attention in v0 is dense SDPA. FastVideo VSA/Sage CUDA kernels are deferred.

## Consequences

- Models are written once; backends can lag independently.
- Burn needs a dynamic-rank wrapper around Flex tensors (Phase 1).
- Luminal op coverage (conv3d, SDPA) may force a hybrid path (DiT on Luminal,
  VAE on Candle) if compilers lack primitives.
- A fourth backend is a new impl of the trait, not a rewrite of Wan.
