# ADR-0001: TensorBackend trait over Burn, Candle, Luminal, and cudarc

- Status: Superseded (amended 2026-09-10; cudarc-primary 2026-09-10; the trait retired 2026-10-24, plan B6)
- Date: 2026-09-09
- Decision-id: FVID-2026-09-09-tensor-backend

## Context

FastVideo is a Python/PyTorch inference and post-training stack. This port must
run the same Wan/FastWan graphs on Rust ML runtimes:

- Burn (CubeCL / Flex), designed around `Backend` + const-rank `Tensor<B, D>`
- Candle, eager dynamically ranked `Tensor`
- Luminal, a static graph compiler
- **cudarc**: direct CUDA via the `cudarc` crate (cuBLAS, NVRTC, cuDNN)

Burn's own `Backend` trait cannot host Candle or Luminal. `burn-candle` is
deprecated as of Burn 0.21.

## Decision

Introduce `fastvideo_ops::TensorBackend` as the only surface model code may use.
Each runtime crate implements it. Native Wan ports may also bypass the trait for
generate while the trait remains the long-term target.

**cudarc is the sole generate path we optimize.** Candle is a frozen behavioral
oracle for porting Wan features (I2V, MoE, causal). Burn and Luminal are frozen
— no new Wan features. CLI default `--backend` is `cudarc`; GPU builds prefer
`--features cuda-cudarc`.

Luminal must compile a **single DiT step** and a **single VAE decode**, then
execute those graphs from the Rust denoising loop. Do not unroll UniPC/DMD into
one static graph. (Frozen; historical requirement only.)

Attention in cudarc is **device-resident dense SDPA** (strided-batched cuBLAS
`Q@Kᵀ` + NVRTC softmax + `P@V`, query-chunked). Host flash/sparse remain as
fallbacks (`FASTVIDEO_SDPA=host` / `FASTVIDEO_VSA_FORCE_SPARSE=1`). FastVideo
vendor FA2/Sage CUDA extensions stay optional later.

`fastvideo-cudarc` is allowed `unsafe_code` (crate-local override of the
workspace forbid) because the `cudarc` driver/cuBLAS APIs require it.

## Consequences

- New Wan inference work lands only in `fastvideo-cudarc` (+ core/CLI wiring).
- Candle stays in-tree as copy-source for MoE / I2V / CLIP / causal logic.
- Burn / Luminal crates remain for CI compile and historical benches only.
- cudarc: Diffusers load + generate; with `--features cuda` and a live device,
  matmul (cuBLAS), same-shape elementwise / silu / gelu / last-axis softmax &
  rms_norm (NVRTC), and NCHW conv2d (cuDNN) run on GPU. Device residency is
  **on by default** (`FASTVIDEO_RESIDENT=0` disables): `CudaTensor` dual-storage
  keeps DiT activations on device; `Linear` pins weights at load and caches BF16
  once. Device UniPC order-1 axpy is default (`FASTVIDEO_DEVICE_SCHED=0` for full
  host UniPC). Hopper enables TF32 Tensor Core GEMMs (`FASTVIDEO_TF32=0` off),
  NVRTC `compute_90` + fast-math, and larger SDPA chunks (`FASTVIDEO_SDPA_CHUNK`).
  BF16 DiT GEMM is on by default (`FASTVIDEO_BF16=0`
  disables; falls back to F32 on unsupported devices). VAE causal conv3d uses
  device-resident GEMM/cuDNN windows on the decode hot path.

## Status 2026-10-23

Burn, Candle and Luminal are no longer in the tree: the workspace has no
crate for them, and `crates/fastvideo-core/src/backend_kind.rs` rejects the
names `candle`, `luminal` and `burn` (as well as `host`, `cpu` and `ref`)
with "was removed; use cudarc". `fastvideo_ops::TensorBackend` has no model
user: its only implementation is in `fastvideo-cudarc`, and the pipelines
reach the device through `cudarc/src/wan` directly. The `unsafe_code` scope
above is also out of date: the workspace `forbid` is overridden with `allow`
in `fastvideo-cudarc`, `fastvideo-loader` and `fastvideo-gpucheck` (each
crate's `Cargo.toml` `[lints.rust]`), not in cudarc only. The decision
record stays as the history of how cudarc became the sole path; the review
`docs/gaps/2026-10-22-architecture-review.md` §4.2 and the execution plan
`docs/plan/2026-10-23-execution-plan.md` (WP-B1) carry what happens next to
the runtime layer.

## Status 2026-10-24 (plan B6)

`fastvideo_ops::TensorBackend`, its host reference (`HostBackend`,
`HostTensor`, `Device`, `DType`, `OpsError`) and `fastvideo-cudarc`'s
`CudarcBackend` were retired (`docs/adr/0017-loader-stages.md`): nothing ran
through them. `fastvideo-ops` keeps only the FP8 E4M3 helpers (`fp8`) the
runtime and the model crate share. The Apple Silicon scaffold `fastvideo-mlx`
was retired with it. The decision this ADR records (one trait over several
runtimes) is superseded by its own amendment: cudarc is the only runtime, and
model code calls `fastvideo-cudarc`'s `rt` directly.
