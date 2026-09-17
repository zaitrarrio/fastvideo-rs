# ADR-0003: Close the device-residency gaps in the DiT hot path

- Status: Accepted
- Date: 2026-09-13
- Decision-id: FVID-2026-09-13-cudarc-perf-pass

## Context

ADR-0001's device-resident `CudaTensor` design (NVRTC elementwise/softmax/
rms_norm + cuBLAS GEMM, with a host `Vec<f32>` fallback for anything not yet
ported) was never fully closed out for two ops that sit directly in the DiT
block's per-block, per-step hot path:

- `layer_norm` had no device kernel at all — every call did a forced D2H
  sync, a scalar host loop, then H2D back.
- AdaLN's modulate/gate steps (`x * (1 + scale) + shift`, `x * gate`) have a
  different shape than `x` (`[B,1,dim]` broadcasting over `[B,L,dim]`), so
  they fell through `CudaTensor`'s generic `broadcast_bin` — a fully scalar,
  per-element `Vec`-allocating host path, also forcing D2H/H2D.

Both run twice or more per DiT block, every block, every denoising step. This
was large enough to matter more than any GEMM tuning, and it's also *why*
`FASTVIDEO_CUGRAPH` (CUDA graph capture) stayed off by default — capturing a
host-bouncing block body isn't safe.

Attention's chunked path (used when `Sq` exceeds the query-chunk size the
scores matrix fits at) gathered each chunk into a freshly-packed contiguous
buffer via `memcpy_dtod`, ran the chunk's GEMMs, then scattered the result
back — `2 * B*H` device-to-device copies per chunk for no compute benefit.
cuBLAS's strided-batched GEMM API already takes an inter-batch stride
independent of the per-call row count, so a chunk can be addressed in place.

Sequence-parallel dispatch (`FASTVIDEO_SP_WORLD>1`) had a `device_for_rank`
helper mapping a rank to a physical device index, but nothing ever called it
— every "rank" ran sequentially on the single global device, which is smaller
batched GEMMs than the unsharded call for zero benefit from extra GPUs.

## Decision

- Add real device kernels: `layer_norm_last` (optional affine, block-per-row
  reduction), `modulate_scale_shift_last` and `broadcast_mul_last` (fused
  AdaLN broadcast ops, no host round trip). `CudaTensor::layer_norm` gets a
  device branch; new `CudaTensor::modulate`/`gate_mul` methods replace the
  `broadcast_bin` call sites in `transformer.rs`.
- Rewrite `softmax_last`/`rms_norm_last` from one-thread-per-row (serial,
  uncoalesced) to one-block-per-row with a shared-memory tree reduction.
- Chunked SDPA addresses `Q`/the output buffer via `CudaView` offset +
  explicit stride (new `matmul_linear_wt_strided_batched_x_view` /
  `matmul_2d_strided_batched_out_view` in `device.rs`) instead of copying.
- Sequence-parallel dispatch spawns one OS thread per rank, each bound to its
  own `DeviceContext` via a thread-local override (`device::set_thread_device`)
  and a device registry (`device::device_for_index`); gather stays
  host-mediated (a `CudaSlice` belongs to the context that allocated it, so a
  shard's result is downloaded on its own rank thread before crossing back).
  NCCL P2P — skipping that host round trip — remains a follow-up.
- Batch CFG's cond/uncond into one batch=2 forward pass, scoped to the
  no-I2V/no-image-conditioning path (see decision log for why).
- Cache every hot-path `FASTVIDEO_*` flag read (`envflag::CachedBool`); add
  `[profile.release]` tuning; mmap + `rayon`-parallelize weight loading.

## Consequences

- `CudaTensor::modulate`/`gate_mul` require `[B,1,dim]`-shaped scale/shift/
  gate against a `[B,L,dim]` self — this is AdaLN's specific broadcast shape,
  not a general broadcast op; callers with a different shape still need
  `broadcast_bin`.
- `fastvideo-loader` now carries a crate-local `unsafe_code = allow` (for
  `Mmap::map`), matching `fastvideo-cudarc`'s existing CUDA-FFI override —
  the workspace default stays `forbid` everywhere else.
- None of this has run on a live GPU. `cargo check --features cuda` (via
  `CUDARC_CUDA_VERSION`, no nvcc/driver on this machine) type-checks the Rust
  call sites but cannot compile the NVRTC CUDA-C kernel strings or exercise
  anything gated on a live device. **First GPU run should specifically
  re-verify**: the new kernels' numerics (especially the block-reduction
  softmax/rms_norm and the chunked-SDPA offset math) against the
  previously-working code path, and the multi-GPU thread/context fan-out
  under real concurrent CUDA calls.
- Deliberately not attempted this pass (needs a GPU in the loop, not blind
  implementation): a Drop-based scratch-buffer allocator pool, and a true
  fused/tiled flash-attention kernel. See decision log for the reasoning.

Verified (host-side): `cargo test --workspace --lib` (113 tests) and
`CUDARC_CUDA_VERSION=12040 cargo check -p fastvideo-cudarc --features cuda`.
