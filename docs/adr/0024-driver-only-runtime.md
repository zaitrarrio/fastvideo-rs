# ADR-0024: A driver-only runtime (no JIT, no cuDNN, static CUDA libraries)

- Status: Step 1a accepted and done (PR #245, D1 revised 2026-11-12); step 1b
  deferred to step 4 (FVID-2026-11-12-defer-wo-1b, D3 suspended); step 2
  decided 2026-11-24 (D2: the cuDNN attention candidate is off; Sage / fwd2
  follow-ups in its work order); step 3 Proposed; step 4 partly done 2026-11-24 as an additional image
  (`cuda-static`, FVID-2026-11-24-ship-static-fv-serve, D5 below; D4 open)
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
   1a. The MoE module gets its own AOT table (`AOT_MOE`), compiled at build
       time by libnvrtc (not nvcc) from the exact string `moe_kernel_src()`
       returns, with the runtime's NVRTC option strings, for every SM in
       `FV_CUBIN_SMS`. Loading follows the
       attn_fp8 order: exact cubin → same-major PTX ≤ device SM → NVRTC.
   1b. *(Deferred 2026-11-12 to step 4, FVID-2026-11-12-defer-wo-1b: it removes
       no library while cuBLASLt / cuDNN still load libnvrtc, and it would drop
       the NVRTC fallback an unbuilt GPU generation relies on.)*
       A cargo feature `jit` on `fastvideo-cudarc` gates every runtime NVRTC
       compile path. Without it, a module with no cubin or PTX for the
       device is a hard error naming the module, the device SM and the SMs
       the binary was built for; `FASTVIDEO_KERNELS=nvrtc` is refused naming
       the feature. `fastvideo-gpucheck` enables `jit`; served images do not.
       The cudarc `nvrtc` cargo feature stays on (its `Ptx` type loads our
       cubins); only our calls into NVRTC are gated.
2. **Remove the cuDNN SDPA candidate** from `sdpa_rule` once `flash_mma_fwd2`
   (or attn_dc) is measured within an owner-set margin of it on every tier
   GPU (D2). *Measured and decided 2026-11-24 (D2 below): the candidate is
   off; the 3-5 % it bought on RTX PRO 6000 is taken back with SageAttention2
   where a model's gate allows it and with fwd2 tuning elsewhere (2b, 2c of
   the work order).*
3. **Replace cuDNN convolutions** with an in-tree implicit-GEMM conv kernel
   (2-D / 3-D, grouped incl. depthwise, transposed, f32 and bf16 NDHWC),
   then delete the cuDNN handles from `rt/device.rs` / `rt/graph.rs`.
   fv-gpucheck and fv-lab may keep cuDNN as references. *Work order
   written 2026-11-25 (`docs/plan/2026-11-25-adr24-step3-wo.md`: 3a routed
   spike, 3b implicit GEMM, 3c small kernels, 3d removal), with the
   inventory of every cuDNN call site on the served path.*
4. **Static CUDA libraries** via cudarc's `static-linking` feature: cuBLAS,
   cuBLASLt, cudart, `libculibos`; NVRTC static only if step 1b's measurement
   shows cuBLASLt still needs it with cuDNN gone.
   *Brought forward 2026-11-24 as a second image beside the dynamic one*
   (D5): fv-serve's `cuda-static` feature (`fastvideo-cudarc/link-static`)
   links cuBLAS, cuBLASLt, cudart, NVRTC (+ nvptxcompiler, builtins), CUPTI,
   `libculibos` and `libstdc++` statically; cuDNN stays the shared library
   (an empty `libcudnn.a` satisfies cudarc's directive, `dylib=cudnn`
   resolves the symbols), so the `serve-cuda-static` image stays on
   `base-cuda` until step 3. Every preset keeps running `cuda`;
   `cuda-static` is chosen by image tag. `docs/serve/images.md` §"The
   statically linked image".

## Owner decisions

- **D1 (step 1a compiler). Decided 2026-11-12: libnvrtc at build time**,
  with the runtime's own option strings (`rt/moe_nvrtc_opts.rs`:
  `--ftz=true --fmad=true`), FVID-2026-11-12-moe-aot-build-time-nvrtc.
  It supersedes the 2026-11-07 choice of nvcc `--use_fast_math --ftz=true`
  (FVID-2026-11-07-moe-aot-flags), which the GPU gate refuted: 4 of 12
  `moe_aot` checks differed on an RTX PRO 6000, because cudarc 0.17's
  `use_fast_math` only emits `--fmad=true` and the served MoE module was
  never fast math.
- **D2 (step 2 margin).** The acceptable slowdown, per tier GPU, for dropping
  the cuDNN SDPA candidate. **Measured 2026-11-24** on an RTX PRO 6000 (the
  only tier GPU where the rule picks cuDNN):
  `docs/perf/2026-11-24-sdpa-rule-cudnn-margins.md`. cuDNN is 4-8 % faster
  than `flash_mma_fwd2` on the long self-attention shapes (H3 768p +4.6 %,
  H3 1080p +4.1 %, LTX 1080p 20 s +4.1 %, Wan 720p +7.7 %), within noise or
  slower on cross attention, short sequences and 4K; at the denoise-step level
  that is about 3 % (up to about 5 % where the kernel margin is 7-8 %), 0 %
  on every other GPU. **Decided 2026-11-24 (FVID-2026-11-24-drop-cudnn-sdpa):
  drop the candidate now** (`sdpa_rule::RULES` empty, `CUDNN_DEFAULT_ON`
  false: `auto` never picks cuDNN), take the margin back with SageAttention2
  on sm_12x for the models whose calibrated quality gate allows it (already
  default for `ltx-pro`), and tune `flash_mma_fwd2` for every other case
  (H3, whose recipes failed the Sage gate; cross attention; short shapes).
  `FASTVIDEO_FLASH_KERNEL=cudnn` stays as a diagnostic override and
  fv-gpucheck keeps measuring cuDNN (`sdpa_rule`, `attn3_bench`); the
  library itself leaves with step 3. Work order:
  `docs/plan/2026-11-24-adr24-step2-wo.md`.
- **D3 (step 1b CI type-check). Decided 2026-11-07: the recommendation**
  (FVID-2026-11-07-no-jit-build-gate); suspended 2026-11-12 with step 1b. `task build:cuda` runs `cargo check` with
  no nvcc; without `jit` that binary would embed no kernels. Recommended:
  `build:cuda` checks with `--features cuda,jit`, and `build.rs` fails a
  `cuda` build without `jit` when no nvcc is found. Alternative: keep the
  warning and rely on the runtime hard error.
- **D4 (licensing).** Written confirmation from NVIDIA for the GPU-rental
  product before any image with static NVIDIA libraries serves rented
  capacity. Open as of 2026-11-24: the `cuda-static` image is built,
  smoke-checked and published, and serves no rented capacity until then.
- **D5 (a static image now). Decided 2026-11-24: ship a statically linked
  fv-serve in addition to the dynamically linked one**
  (FVID-2026-11-24-ship-static-fv-serve): a second CUDA image,
  `cuda-static`, not a replacement; `deploy/deploy.toml` `[[images]]` gains a
  `link` field (one dynamic CUDA image, at most one static, one CPU).

## Consequences

- Served behaviour changes only in failure: a device whose SM was not built
  now fails at module load instead of JIT-compiling. Each failure names the
  rebuild (`FV_CUBIN_SMS`).
- After step 3 the `base-cuda` layer loses cuDNN; after step 4 the runtime
  image needs no CUDA layer at all. The 2026-11-24 `cuda-static` image is
  the step-4 binary on today's `base-cuda`: its own CUDA calls no longer
  depend on the image's library versions, but cuDNN's engines still load the
  shared cuBLASLt / NVRTC, so the layer stays until step 3.
- Step 3 is the large one: it is new kernel work with its own parity gates
  (VAE decode PSNR per family against the cuDNN path) and is not started
  until steps 1–2 land.
- Rollback for steps 1–2 is cheap (feature flag, one rule entry); step 4 is a
  build-configuration change, reverted by switching cudarc back to
  `dynamic-loading`.
