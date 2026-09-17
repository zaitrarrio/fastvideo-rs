# Decision log

Project code: FVID

### FVID · 2026-09-13 · FVID-2026-09-13-strict-device-check
- Trigger: user asked how to verify GPU code paths never silently execute on CPU — every device op added in FVID-2026-09-13-cudarc-perf-pass is structured as "try the device kernel, silently compute on host if it returns `None` for any reason", which is correct for portability but means a real GPU run where something's subtly broken (bad shape guard, kernel launch failure, residency toggled off) still produces correct output, just slower, with nothing in the logs
- Options: no tooling (rely on manual profiling to notice a slowdown); log-only auditing; hard-fail strict mode; full distributed tracing
- Decision: added `tensor::strict_device_check(op, detail)` — when `FASTVIDEO_STRICT_DEVICE=1`, residency is enabled, and a CUDA device is live, a hot op (`layer_norm`, `modulate`, `gate_mul`, `attention`) falling back to the true host compute path returns a hard `Err` naming the op and shapes, instead of silently continuing; ops with no device kernel at all (`div`, `mean_keepdim`, `sqrt`, non-trailing-axis `narrow`/`chunk`/`cat`, sparse block-attention) are deliberately NOT wired to this — those are known, permanent gaps, not per-call regressions, and strict mode would just always fire there with no diagnostic value. Paired with a non-fatal counterpart: `tensor::device_path_stats()` (plain atomic hit/miss counters per op) and `FASTVIDEO_DEVICE_STATS=1`, which prints a device-vs-host dispatch summary after `generate()` without changing behavior — for auditing a run without risking a crash.
- Reason: user wants a concrete way to catch a silent GPU→CPU fallback regression before it just shows up as an unexplained slowdown
- Reversibility: cheap (both flags default off; zero behavior change unless explicitly set)
- Executed by: Executor
- ADR: ADR-0003 (amended — see "Consequences: first GPU run should re-verify")
- Verification: `cargo test --workspace --lib` (117 tests, 0 failures), incl. `strict_device_check_is_noop_without_a_live_device` (proves strict mode never fires on this host-only sandbox — the only half of its contract testable without a GPU) and `device_path_stats_record_and_reset` (counter bookkeeping). **The half that matters most — strict mode actually catching a live-device fallback — is untested here by construction** (no GPU in this sandbox); the intended workflow is `FASTVIDEO_STRICT_DEVICE=1` on a real Vast run as a smoke/CI gate.

### FVID · 2026-09-13 · FVID-2026-09-13-cudarc-perf-pass
- Trigger: performance review of `fastvideo-cudarc` found `layer_norm`/AdaLN-modulate host round trips defeating device residency every DiT block, one-thread-per-row softmax/rms_norm kernels, sequential (non-parallel) CFG and sequence-parallel dispatch, `Mmap`-free weight loading, and a chunked-SDPA path that gather/scatter-copied every chunk instead of using cuBLAS's native inter-batch stride
- Options: full rewrite incl. true flash-attention kernel + NCCL P2P + Drop-based scratch allocator; scoped fixes only where verifiable without a live GPU (this machine has no nvcc/driver — verification loop is `CUDARC_CUDA_VERSION=<pinned> cargo check --features cuda`, which type-checks cudarc API usage and NVRTC *Rust* call sites but cannot compile/run the NVRTC CUDA-C kernel strings themselves, and cannot exercise any code gated on a live `device::global_device()`)
- Decision: implemented and test-verified: device `layer_norm_last`/`modulate_scale_shift_last`/`broadcast_mul_last` NVRTC kernels (replacing AdaLN's `broadcast_bin` host path — see ADR-0003); block-reduction `softmax_last`/`rms_norm_last` (was one-thread-per-row, uncoalesced); cached every hot-path `FASTVIDEO_*` env-flag read behind `envflag::CachedBool`/`CachedString` (were re-parsing `std::env::var` per op); `[profile.release]` (LTO, codegen-units=1, panic=abort); batched cond/uncond CFG into one batch=2 forward pass (scoped off for I2V/image-conditioning — would need `mask`/`cond`/`image` also duplicated, not verifiable blind); real per-rank multi-GPU dispatch for `FASTVIDEO_SP_WORLD>1` (`device::device_for_index` + thread-local device override + `std::thread::scope` fan-out, host-mediated gather) — `sp::device_for_rank` existed but was dead code, every rank ran on one GPU sequentially before this; `Mmap`+`rayon`-parallel weight loading in `fastvideo-loader` (was `std::fs::read` + single-threaded scalar dtype conversion), needed a crate-local `unsafe_code = allow` override (same pattern as `fastvideo-cudarc`'s CUDA-FFI override) for `Mmap::map`; chunked SDPA now addresses `Q`/output via `CudaView` offset + explicit inter-batch stride (`matmul_linear_wt_strided_batched_x_view` / `matmul_2d_strided_batched_out_view`) instead of `memcpy_dtod`-gathering/scattering every chunk. Also fixed two pre-existing `--features cuda` compile breaks unrelated to the above (`Bf16Activation` missing `Debug`, wrong `CUgraphInstantiate_flags` variant name) found while building the verification loop — `cargo check --features cuda` did not compile before this pass, on any of these changes.
- Deliberately NOT done, scoped out as needing GPU-supervised verification rather than attempted blind: a generic Drop-based scratch-buffer pool for the remaining `alloc_zeros` churn (every elementwise/GEMM op still allocates fresh) — a pool needs to hook `CudaSlice` recycling into `Arc<DeviceBuffer>`'s drop path, and a wrong reference-count/lifetime edge case there is silent data corruption (a live tensor's buffer handed back to the pool and overwritten), not a crash; a fused/tiled flash-attention kernel (true O(N) memory attention) — writing a correct online-softmax CUDA kernel with shared-memory tiling blind, with no way to check its numerics against a reference on real hardware, is the wrong risk/verification tradeoff for this pass. Both remain real opportunities; do them with a GPU in the loop.
- Reason: user asked to integrate all recommendations from the perf review; prioritized changes verifiable via unit tests (host-fallback math, chunk-loop stride arithmetic, end-to-end tiny-model `generate()`) and `--features cuda` type-checking over ones that can only be checked by running real CUDA kernels
- Reversibility: medium (each item is an isolated function/module change; the two scoped-out items were never started, so nothing to revert there)
- Executed by: Executor
- ADR: ADR-0001 (amended), ADR-0003 (new — see below)
- Verification: `cargo test --workspace --lib` (113 tests, 0 failures, incl. new tests for `layer_norm`/`modulate`/`gate_mul` against naive references, a CFG-batched end-to-end tiny `generate()` run, a real mmap-backed multi-file/mixed-dtype loader round trip, and the chunked-SDPA offset/stride arithmetic across even/remainder/edge-case chunk sizes) + `CUDARC_CUDA_VERSION=12040 cargo check -p fastvideo-cudarc --features cuda` clean. **Still needed before trusting this on real workloads: an actual GPU run** — none of the above proves the NVRTC kernel *source* compiles under nvcc/NVRTC, that the new kernels' numerics match a reference on real hardware, or that the multi-GPU thread/context fan-out behaves correctly under real concurrent CUDA calls.

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-primary-parity
- Trigger: user chose cudarc-only focus; close Burn/Candle feature work; Wan inference then speed parity
- Options: keep multi-backend feature parity; freeze others and push cudarc
- Decision: CLI/scripts default to cudarc + `cuda-cudarc`; Candle/Burn/Luminal frozen; cudarc gains MoE, I2V+CLIP+VAE encode, causal mask, MP4 mux (`FASTVIDEO_SAVE_MP4`), TeaCache (`FASTVIDEO_TEACACHE`), chunked SDPA, `CachedLinear` residency (`FASTVIDEO_RESIDENT`), reject `num_gpus>1` until SP, warn on VSA ids
- Reason: plan locked cudarc as sole generate path to optimize
- Reversibility: medium (frozen backends remain compilable)
- Executed by: Executor
- ADR: ADR-0001 / ADR-0002 (amended cudarc-primary)
- Verification: `cargo test -p fastvideo-cudarc -p fastvideo-core --lib`

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-gpu-ops
- Trigger: cudarc v0 ran non-GEMM ops on host f32 after native weight load
- Options: host-only; NVRTC+cuDNN with H2D/D2H per op; full device-resident CudaTensor
- Decision: NVRTC elementwise/activations/softmax/rms_norm + cuDNN conv2d when global device is set; keep host Vec API; causal conv3d/broadcast stay host
- Reason: user asked to land the next speed step without blocking on full residency
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (amended)
- Verification: `cargo test -p fastvideo-cudarc --lib` + `fastvideo-core` lib (CUDA path compile/smoke pending Vast; Mac has no nvcc)

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-backend
- Trigger: Burn CubeCL CUDA ~2× slower than Candle on H100 smoke; need a thinner CUDA path
- Options: optimize Burn; cudarc fourth backend; Candle-only
- Decision: add `fastvideo-cudarc` as `--backend cudarc` with full Diffusers Wan load/generate; crate-local `unsafe_code = allow`; native BF16 loader path
- Reason: user chose cudarc as fourth generate backend with no stub bring-up
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (amended)
- Verification: `cargo test -p fastvideo-cudarc --lib` + workspace lib tests (CUDA smoke on Vast)

### FVID · 2026-09-10 · FVID-2026-09-10-h100-candle-vs-burn
- Trigger: user asked to benchmark candle vs burn on H100
- Options: 4090 only; H100 50435452 smoke 256² 9f/2step
- Decision: ran both on H100 SXM (`50435452`); candle CUDA BF16 77.3s vs burn CUDA F32 162.9s (load+generate)
- Reason: fair GPU compare of native Burn path vs proven Candle
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0002
- Verification: artifacts under `artifacts/20260910T010910Z-nvidia-h100-80gb-hbm3-i50435452-all/`

### FVID · 2026-09-09 · FVID-2026-09-09-burn-cuda-luminal-compile
- Trigger: native Wan ports were CPU-only; ADR-0001 needs Luminal compile-once DiT/VAE
- Options: Burn CubeCL cuda feature vs keep ndarray; real luminal Graph vs eager facade
- Decision: `fastvideo-burn` `--features cuda` → `Cuda<f32>` + device from CLI; luminal 0.2 `GenericCompiler`+`CPUCompiler` for tiny fixed-shape DiT step + VAE decode (eager fallback for full 1.3B)
- Reason: user asked to implement burn-cuda and luminal compile
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001
- Verification: `cargo test -p fastvideo-burn -p fastvideo-luminal -p fastvideo-core --lib` green (CUDA feature needs Vast)

### FVID · 2026-09-09 · FVID-2026-09-09-native-wan-ports
- Trigger: Burn/Luminal only wrote stub UniPC latents; need fair FastVideo/Wan benches
- Options: TensorBackend generic graph; native per-runtime Wan ports; Candle-only
- Decision: native UMT5+DiT+feat-cache VAE in `fastvideo-burn` (Burn 0.18 ndarray) and `fastvideo-luminal` (host NdTensor + compiled facades); no Candle fallback in generate; TensorBackend deferred
- Reason: user asked for native ports first, validate abstraction later
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (supersedes stub adapters for generate)
- Verification: `cargo test --workspace --lib` green; tiny PNG on burn/luminal/candle; Diffusers `WanPipeline::load` wired

### FVID · 2026-09-09 · FVID-2026-09-09-luminal
- Trigger: third Rust ML backend named “luminar”
- Options: Luminal (graph compiler) vs some other crate
- Decision: Luminal (https://github.com/luminal-ai/luminal)
- Reason: user confirmed that spelling
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0001
- Verification: checked against source

### FVID · 2026-09-09 · FVID-2026-09-09-wan-inference
- Trigger: FastVideo is a full post-training + inference monorepo
- Options: 1.3B-only inference; all Wan/FastWan inference; inference + training
- Decision: inference-first for every Wan/FastWan family in FastVideo’s registry
- Reason: user chose all Wan families, still no training
- Reversibility: cheap
- Executed by: Executor
- ADR: none
- Verification: checked against FastVideo `fastvideo/models/wan/definition.py`

### FVID · 2026-09-09 · FVID-2026-09-09-tensor-backend
- Trigger: need one model implementation for three runtimes
- Options: write models three times; Burn-only + export; custom TensorBackend trait
- Decision: custom `TensorBackend` trait; do not use deprecated `burn-candle`
- Reason: Burn, Candle, and Luminal are distinct runtimes; Luminal is graph-based
- Reversibility: costly
- Executed by: Executor
- ADR: ADR-0001
- Verification: pending

### FVID · 2026-09-09 · FVID-2026-09-09-vast-gpu
- Trigger: CPU `--tiny` is not a real inference bar; need GPU for Wan 1.3B
- Options: local Metal; wait for Luminal GPU; CUDA on Vast.ai
- Decision: real GPU runs on Vast CUDA hardware (Candle `--features cuda`, BF16)
- Reason: user directed GPU runs onto Vast
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0002
- Verification: gates green (tiny CUDA F32 generate on RTX 4090 instance 50416610)

