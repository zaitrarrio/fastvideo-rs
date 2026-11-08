# ADR-0024: A driver-only runtime (no JIT, no cuDNN, static CUDA libraries)

- Status: Accepted for step 1 (D1, D3 decided 2026-11-07); steps 2–4
  Proposed (D2, D4 open)
- Date: 2026-11-07
- Decision-id: FVID-2026-11-07-driver-only-runtime
- Source: owner request, 2026-11-07 ("how could I make it even more
  encapsulated"); dependency audit in `docs/serve/images.md` §"CUDA runtime
  (libcudart) and the dependency audit"
- Code (this ADR's first increments): `crates/fastvideo-cudarc/build.rs`,
  `src/rt/ops.rs` (`moe_kernel_src`, `load_moe_kernels`), `src/rt/kernels.rs`,
  `src/rt/nvrtc_sass.rs`, `src/rt/attn_{fp8,sage,fp4,dc}.rs`, `src/h3/vdn.rs`
- Work orders: `docs/plan/2026-11-07-driver-only-runtime-wo.md`

## Context

A served `fastvideo` / `fv-serve` binary today needs, besides the host
driver (`libcuda.so.1`, R580+, checked by `fv-boot`):

| needs | why | size class |
|---|---|---|
| `libcublas` / `libcublasLt` 13.4 | every GEMM | large |
| `libcudnn` + engine sub-libraries | convolutions (`rt/conv.rs`, `ltx2/vae_fast.rs`), one SDPA candidate (`rt/cudnn_sdpa.rs`) | largest |
| `libnvrtc` + `libnvrtc-builtins` | our JIT fallbacks; also loaded by cuBLASLt / cuDNN engines (images.md) | medium |
| `libcudart` | CUPTI only (`FASTVIDEO_GPU_TRACE`) | small |

All of it is `dynamic-loading` through cudarc 0.17 (`Cargo.toml`
`[workspace.dependencies] cudarc`), shipped in the ~1 GB `base-cuda` layer.
The goal is a single binary whose only external dependency is the driver,
plus one verified weights bundle per variant.

Licence facts that bound the design (summarised from the documents as of
2026-11-07; not legal advice):

- CUDA Toolkit EULA, Attachment A (Linux) lists as distributable
  `libcublas_static.a`, `libcublasLt_static.a`, `libnvrtc_static.a`,
  `libnvrtc-builtins_static.a`, `libcudart_static.a`, and `libculibos.a` as
  required for static cuBLAS. Conditions: unmodified object code, the
  application adds material functionality, only our application may access
  them, no stand-alone redistribution of the SDK, and the SDK may not be made
  subject to a source-disclosing open-source licence.
- cuDNN SLA supplement: only the runtime `.so` / `.dll` files are
  distributable. Static cuDNN is not. A driver-only binary therefore requires
  removing cuDNN, not linking it.
- Neither document has an explicit hosting clause; both forbid renting "the
  SDK". The GPU-rental product's position is to be confirmed with
  `nvidia-compute-license-questions@nvidia.com` before it launches (D4).
- ffmpeg as built for the images is GPLv3 (`--enable-version3`). It must stay
  a separate process; nothing GPL may be linked into the binary that carries
  static NVIDIA libraries.

What already holds: every module in `KERNEL_PARTS` and the attn fp8 / sage /
fp4 / dc and H3 VDN modules embed nvcc cubins + PTX per SM (`aot.rs`
tables) and load them before NVRTC. The exception is the MoE module
(`rt/ops.rs` `load_moe_kernels`), which always NVRTC-compiles
`moe_kernel_src()` with fast math + FTZ, even though the same region is
already in the AOT main module (compiled there with the reference, non-fast
flags, so it is not a drop-in).

## Decision

Reach a driver-only runtime in four ordered steps, each its own work order
and each gated on byte identity of outputs where the plan's §1 rule applies:

1. **No runtime compilation in served builds.**
   1a. The MoE module gets its own AOT table (`AOT_MOE`), compiled by nvcc
       from the exact string `moe_kernel_src()` returns, with flags matching
       its NVRTC options, for every SM in `FV_CUBIN_SMS`. Loading follows the
       attn_fp8 order: exact cubin → same-major PTX ≤ device SM → NVRTC.
   1b. A cargo feature `jit` on `fastvideo-cudarc` gates every runtime NVRTC
       compile path. Without it, a module with no cubin or PTX for the
       device is a hard error naming the module, the device SM and the SMs
       the binary was built for; `FASTVIDEO_KERNELS=nvrtc` is refused naming
       the feature. `fastvideo-gpucheck` enables `jit`; served images do not.
       The cudarc `nvrtc` cargo feature stays on (its `Ptx` type loads our
       cubins); only our calls into NVRTC are gated.
2. **Remove the cuDNN SDPA candidate** from `sdpa_rule` once `flash_mma_fwd2`
   (or attn_dc) is measured within an owner-set margin of it on every tier
   GPU (D2).
3. **Replace cuDNN convolutions** with an in-tree implicit-GEMM conv kernel
   (2-D / 3-D, grouped incl. depthwise, transposed, f32 and bf16 NDHWC),
   then delete the cuDNN handles from `rt/device.rs` / `rt/graph.rs`.
   fv-gpucheck and fv-lab may keep cuDNN as references.
4. **Static CUDA libraries** via cudarc's `static-linking` feature: cuBLAS,
   cuBLASLt, cudart, `libculibos`; NVRTC static only if step 1b's measurement
   shows cuBLASLt still needs it with cuDNN gone.

## Owner decisions

- **D1 (step 1a flags). Decided 2026-11-07: fast math + FTZ, own `AOT_MOE`
  table** (FVID-2026-11-07-moe-aot-flags). Was recommended: MoE cubins compiled with flags
  equivalent to its NVRTC options (fast math + FTZ), proven bit-identical on
  GPU against the NVRTC module. Alternative: load the MoE entry points from
  the main AOT module (reference flags), accepting a numerics change and a
  new LingBot gold measurement.
- **D2 (step 2 margin).** The acceptable slowdown, per tier GPU, for dropping
  the cuDNN SDPA candidate.
- **D3 (step 1b CI type-check). Decided 2026-11-07: the recommendation**
  (FVID-2026-11-07-no-jit-build-gate). `task build:cuda` runs `cargo check` with
  no nvcc; without `jit` that binary would embed no kernels. Recommended:
  `build:cuda` checks with `--features cuda,jit`, and `build.rs` fails a
  `cuda` build without `jit` when no nvcc is found. Alternative: keep the
  warning and rely on the runtime hard error.
- **D4 (licensing).** Written confirmation from NVIDIA for the GPU-rental
  product before any image with static NVIDIA libraries serves rented
  capacity.

## Consequences

- Served behaviour changes only in failure: a device whose SM was not built
  now fails at module load instead of JIT-compiling. Each failure names the
  rebuild (`FV_CUBIN_SMS`).
- After step 3 the `base-cuda` layer loses cuDNN; after step 4 the runtime
  image needs no CUDA layer at all.
- Step 3 is the large one: it is new kernel work with its own parity gates
  (VAE decode PSNR per family against the cuDNN path) and is not started
  until steps 1–2 land.
- Rollback for steps 1–2 is cheap (feature flag, one rule entry); step 4 is a
  build-configuration change, reverted by switching cudarc back to
  `dynamic-loading`.
