# Decision log

Project code: FVID

### FVID · 2026-09-18 · FVID-2026-09-18-bf16-conv3d
- Trigger: after the SiLU fusion and chunked decode, VAE convolutions were the remaining VAE cost, and the dominant one ran 1.11 TFLOP in 27.7 ms ≈ 40 TFLOPS on a 3090 Ti — this card's TF32 peak, so compute-bound with no memory headroom to recover
- Options: leave it; bf16 storage through the whole VAE (large refactor); bf16 operands per convolution with casts around it
- Decision: **bf16 cuDNN conv3d as a third auto-selected backend** (`ded1f0c`). bf16 tensor ops run at roughly twice TF32 on Ampere, and the casts around one convolution cost ~3.4 ms against ~13.8 ms of compute saved, so it wins even without chaining bf16 between ops. `conv3d`'s auto mode times three candidates per shape and keeps the winner; bf16 only competes in fast mode, since exact mode must stay comparable to the CPU path.
- Measured 1.3-2.1x on large shapes across **three architectures**: Ada 4090 `[1,192,10,224,416]` 32.7 -> 15.4 ms, `[1,96,10,448,832]` 41.1 -> 28.1 ms; Blackwell 5090 `[1,192,6,224,416]` 9.9 -> 5.6 ms, `[1,96,6,448,832]` 10.4 -> 7.2 ms. 13 of 14 full-resolution shapes chose bf16.
- **Per-shape selection is load-bearing, not tidiness.** On the 4090, `[1,96,3,128,128]` is 0.3 ms in cuDNN f32 and **6.6 ms in bf16** — 22x slower, presumably a poor cuDNN algorithm for that configuration — and one other shape also lost. A blanket bf16 switch, the obvious implementation, would have absorbed both regressions silently.
- Accuracy: VAE decode rel_l2 0.00287 at 64.0 dB against the fast-mode limit of 0.02, versus 0.00092 at 74 dB for F32. A real trade, of the same kind already accepted for bf16 linears and bf16 attention probabilities — not free like the SiLU fusion.
- Reversibility: cheap (`FASTVIDEO_CONV3D=cudnn` or `unfold` forces the old behaviour; exact mode never selects it)
- Executed by: Executor
- Verification: runs `20260918T095625Z-clip` (Ada) and `20260918T101337Z-clip` (Blackwell, all stages pass, 15 min, $0.120). 8s clip on the 5090: denoise 18.4 s, VAE decode 5.02 s, both clips within 40 ms of each other.
- Process note: a `clip-2s-exact` OOM during this work was **accumulated GPU memory on a reused instance**, not the card and not these changes — the stage passes on a fresh box. Rent a new instance per run; never trust a timing from a machine that has already done substantial work.

### FVID · 2026-09-18 · FVID-2026-09-18-fused-block-sparse-rejected
- Trigger: VSA's fine stage gathers each query tile's selected K/V into a dense buffer (~106 GB/layer written and read, plus a 26 GB score buffer) so cuBLAS can run the GEMMs on tensor cores. A fused kernel streaming K/V straight from the tiled layout would move ~53 GB/layer and materialise no scores.
- Options: keep the gather; fuse with scalar f32 math; fuse with tensor cores via inline PTX `mma.sync`
- Decision: **the fused kernel stays off.** Measured on one RTX 5090 against the gather path on the same card class: **8.0x slower at 1,456 tokens, 5.5x at 4,368, 8.6x at 13,104**. Trading ~80 GB/layer of traffic for scalar f32 math does not come close to paying for the loss of cuBLAS bf16 tensor cores, which run at roughly twice the scalar f32 rate before any efficiency gap.
- The structure was right this time and it still lost. One CUDA block per QUERY TILE amortises each K/V tile load over 64 queries — exactly what `flash_attn_f32` got wrong with one query per block — and that moved the result from 12-36x slower (FVID-2026-09-17-flash-sdpa-rejected) to ~8x. Structure was never the binding constraint; tensor cores are.
- Reason: this is the second hand-written attention kernel to lose to cuBLAS by a wide margin. The rule that generalises: **on this hardware, do not hand-write attention math that cuBLAS can express.** A fused kernel is only worth attempting with `mma.sync` tensor-core intrinsics, and even then it must beat a library that is already near peak.
- Reversibility: free (kept behind `FASTVIDEO_VSA_FUSED=1`, default off; correct, so it stays as a reference implementation and a place to add MMA later)
- Executed by: Executor
- Verification: correctness first — gpucheck holds it to the same host reference as the gather path across four grids in both precision modes, rel_l2 0.0026-0.0028, marginally *better* than the gather path's 0.0032 because the online softmax keeps probabilities in f32 registers instead of round-tripping bf16. Speed: run `20260918T104448Z-clip` (fused) vs `20260918T101337Z-clip` (gather), both RTX 5090 32607 MiB, fresh instances. Stopped after the probe: the clips would only have confirmed it more expensively.
- A dim=64 indexing bug (four output dims per lane hardcoded for dim=128) was found by the first hardware run and fixed in `6ea882c`; the d=128 grids the real model uses would never have caught it.

### FVID · 2026-09-18 · FVID-2026-09-18-vae-decode
- Trigger: after the VSA port, VAE decode (21.9 s) was the largest single cost in an 8s clip, bigger than any remaining attention win
- Options: bf16 storage through the VAE; faster convolutions; fuse elementwise ops; decode several latent frames per pass
- **Convolutions were ruled out by measurement, not skipped**: the dominant conv, `[1,96,6,448,832]` 3x3x3, runs 1.1 TFLOP in 27.7 ms ≈ 40 TFLOPS, which is this card's TF32 peak. Nothing to win there without bf16 storage — a much larger refactor.
- Decision: two changes, both verified equivalent rather than assumed.
  1. **SiLU folded into the channel RMS norm** (`0559ed1`). The decoder always pairs them and the norm kernel already reads x twice, so the activation rides along free and a whole read+write pass over a 859 MB tensor disappears. Three sites: both residual-block norms and each `norm_out`. **21.86 s -> 19.49 s (-10.8%)**, identical on both clips to 10 ms, gpucheck holds the fused path at rel_l2 8e-8 against norm-then-silu.
  2. **Chunked decode** (`a0cbb21`), `FASTVIDEO_VAE_CHUNK`. Decode ran one latent frame per pass — 33 sequential trips through the decoder. The causal conv cache makes a chunk equivalent to the same frames one at a time. **19.49 s -> 17.81 s (-8.6%) at chunk=2**, with 129 frames and a `clipped_fraction` identical to 18 digits, so the video is unchanged.
- **chunk=4 OOMs on 24 GB** (`CUDA_ERROR_OUT_OF_MEMORY`), so the default stays 1: an OOM on a smaller card is a far worse failure than a missed 9%. The flag is opt-in and documented.
- Latent frame 0 must stay its own pass: the temporal upsamplers detect the first pass through an empty cache slot and skip doubling, which is what makes the output 4n+1 rather than 4n. A unit test pins it — folding frame 0 into a chunk turns 17 frames into 14, and the test fails.
- Reason: chunking helped **less** than predicted. 33 sequential passes suggested launch overhead dominated, but halving the passes bought only 8.6%, so decode cost tracks the work rather than the pass count. The pure traffic reduction was the bigger lever.
- Reversibility: cheap (fusion is numerically identical; chunking is one env flag, default unchanged)
- Executed by: Executor
- Verification: runs `20260918T004528Z-clip` (fusion, pass, $0.110) and `20260918T011336Z-clip` (chunk=4 OOM, then chunk=2 pass). A 1.98 s "decode" at chunk=4 was an OOM part-way, not a 10x win — it exceeded what removing 3.7x of the passes could explain, which is what prompted checking.

### FVID · 2026-09-18 · FVID-2026-09-18-vsa-port
- Trigger: the head-to-head (FVID-2026-09-17-upstream-head-to-head) put upstream 1.96x ahead on the same GPU, and the gap was attributable to Video Sparse Attention, which we had not ported
- Options: leave dense; write FlashAttention-2 properly; port VSA with a fused block-sparse kernel; port VSA reusing cuBLAS via a gather
- Decision: **ported VSA**, reading upstream's implementation rather than inferring it. `(4,4,4)` tiles of 64 slots; coarse stage mean-pools Q/K/V per tile and attends over tiles; top-k of those scores picks the tiles the fine stage attends at full resolution; `out = coarse * to_gate_compress + sparse`. The fine stage **gathers** each query tile's selected K/V into a dense bf16 buffer and runs batched GEMMs, rather than a fused kernel: the gather moves ~53 GB/layer against dense attention's ~440 GB of score traffic, and it reuses the tensor-core path that is already fast here.
- **Coarse stage is pinned to F32** regardless of `GemmMath`. Tile selection is discrete: in fast mode bf16 rounding flipped a near-tie, a different tile was attended, and the largest test grid moved rel_l2 0.003 -> 0.043. The coarse stage is 819x819 and costs almost nothing, so selection no longer depends on the math mode.
- Result on an RTX 3090 Ti, 8s clip (448x832, 129 frames, 3 DMD steps): **denoise 82.7 s -> 45.6 s, 1.81x**, 27.6 -> 15.1 s/step, reproducible across both prompts to 15 ms, every quality gate passing. Scaling matches the algorithm: 2x SLOWER at 1,456 tokens (fixed overhead dominates), parity at 4,368, 16% faster at 13,104, 1.81x at 48,048.
- Quality: composition and subject are preserved under the same seed, but VSA is a different sample with visibly more saturated colour (`clipped_fraction` 0.0218 vs 0.008 dense). Not verified against upstream's own VSA output.
- Reversibility: cheap (opt-in; `FASTVIDEO_VSA=1` / gpucheck `--vsa`, and it needs a checkpoint carrying `to_gate_compress`)
- Executed by: Executor
- Verification: host reference equals dense attention exactly when every tile is selected (the load-bearing test); device kernels match that reference across four grids and two group sizes (run `20260917T231353Z-kernels`); end-to-end run `20260918T001353Z-clip` pass, 24 stages, $0.089. Four integration bugs found by live runs, none in the kernels: VSA leaking into dense-reference stages, `--vsa` parsed but never acted on, the legacy host-only sparse branch, and a missing head merge before `to_out`.
- Open: our generation is now ~71 s (45.6 denoise + 21.9 VAE + 3.8 write) against upstream's 55.3 s. VAE decode is the largest remaining single cost. A fused block-sparse kernel would remove the gather.

### FVID · 2026-09-17 · FVID-2026-09-17-upstream-head-to-head
- Trigger: user asked how our numbers compare with upstream FastVideo on the same hardware
- Options: compare against published figures (H100, different GPU); rent two boxes; run both implementations on ONE rented box
- Decision: added a **`compare` tier (T4)** that runs our clip stages and then installs upstream FastVideo in its own uv venv on the SAME instance and times the same 8s clip. Identical hardware by construction. The tier skips T1/T2 (it benchmarks, it does not re-validate). Model load is timed separately and excluded; one warm-up then median of 2.
- Result on an RTX 3090 Ti (448×832, 129 frames, 3 DMD steps, timesteps 1000/757/522, same FastWan2.1-T2V-1.3B weights): **upstream 55.30 s per generation vs ours ~108.4 s (denoise 82.74 + VAE 21.86 + write 3.80) — upstream ≈1.96× faster.** Upstream load 49.89 s, warm-up 95.85 s (Triton JIT; a single-shot benchmark would have called upstream SLOWER than us).
- **The dense comparison does not exist on upstream's side for these weights**: `TORCH_SDPA` cannot load the checkpoint — `Parameter blocks.0.to_gate_compress.bias not found in custom model state dict` — because FastWan ships VSA gate weights their dense model class does not define. Upstream on this model IS the VSA configuration.
- Reason: the gap is the optimization we deliberately have not ported. VSA cuts the quadratic attention term; our 48k-token forward is dominated by dense attention, which is what the flash experiment showed from the other side. ~2× is the measured price of no sparse attention.
- Reversibility: free (new tier, nothing in the shipped path changed)
- Executed by: Executor
- Verification: run `20260917T220533Z-compare`, upstream torch 2.12.0+cu126 / fastvideo 0.2.1. Nine attempts to get here; the failures were ours, not upstream's — see FVID-2026-09-17-exact-parity-24gb-oom and the harness fixes (bad-host recording on mid-run ssh death, disk precheck crediting fetched data, empty-array expansion under bash 3.2, shell lint).

### FVID · 2026-09-17 · FVID-2026-09-17-exact-parity-24gb-oom
- Trigger: the `compare` tier OOM'd in `parity-exact` on two different 24GB cards — a plain RTX 3090 (24124 MiB total, 23846 free, no other tenant) and an RTX 3090 Ti (24112 MiB, 23829 free) — failing on the FIRST `dit_forward_t999` right after `load`. The same stage passes on an RTX A5000 (24111 MiB) and on a larger 3090 Ti (24564 MiB)
- Options: rent only >24GB cards; cut exact-mode peak; give the mempool a finite release threshold; leave it and document
- Decision: **not fixed yet, recorded and routed around.** The `compare` tier now skips T1/T2 (it benchmarks, it does not re-validate) so it never runs exact-mode parity; the validation tiers still do, on cards that fit. Not diagnosed further under a benchmarking task — the suspects are the mempool release threshold (`u64::MAX`, so nothing returns to the driver) and cuDNN workspace from conv3d auto-selection, which times BOTH backends per shape and so allocates a cuDNN plan's workspace even where unfold wins.
- Reason: exact mode is F32 everywhere and is the memory-hungriest configuration; it is a validation path, not a production one, so a 24GB card failing it blocks validation but not use. Worth fixing, but not by guessing mid-benchmark
- Reversibility: free (no behaviour changed)
- Executed by: Executor
- Verification: runs `20260917T202914Z-compare` (3090) and `20260917T204412Z-compare` (3090 Ti), both rc=2 at `parity-exact`, $0.035 and ~$0.01. Next diagnostic: rerun the `parity` tier on a 24GB box with `FV_STAGE_ENV="FASTVIDEO_CONV3D=unfold"` — if it passes, cuDNN workspace is the cause; if not, the mempool is.

### FVID · 2026-09-17 · FVID-2026-09-17-bf16-attention-probs
- Trigger: after flash SDPA was rejected (FVID-2026-09-17-flash-sdpa-rejected), the dominant remaining cost in a clip-scale forward was dense attention's probability matrix (`bh*sq*sk`), written by softmax and read back by the `P@V` GEMM
- Options: leave it; store probabilities as bf16; also store pre-softmax scores as bf16; rewrite attention with two-level tiling; port VSA
- Decision: **probabilities are stored as bf16 under `GemmMath::Bf16`** (`FASTVIDEO_ATTN_PROBS_BF16=0` opts out; exact mode unchanged). New `softmax_last_bf16` kernel writes them directly — three passes over the f32 scores (max, sum, write) rather than keeping f32 exponentials, since a 48k-wide row cannot live in shared memory — `V` is cast once per call, and `gemm_raw` gained an A/B dtype parameter so `P@V` takes bf16 operands with an F32 result. Pre-softmax scores stay F32: bf16 there perturbs `exp()` by ~2% at score magnitudes around 10, far above the fast-mode error budget.
- Reason: under `CUBLAS_COMPUTE_32F_FAST_16BF` cuBLAS already rounds F32 operands to bf16 for the tensor-core op, so materialising the probabilities as bf16 is the same arithmetic over half the bytes — a pure traffic win, not a precision trade
- Reversibility: cheap (one env flag; exact mode never takes the path)
- Executed by: Executor
- Verification: same-instance A/B on 51338983 (sm_86), runs `20260917T190610Z-clip` (F32, via `FV_STAGE_ENV="FASTVIDEO_ATTN_PROBS_BF16=0"`) and `20260917T193716Z-clip` (bf16), 22 stages PASS each. **8s clip denoise 128.5 s → 82.7 s (-35.6%)**, 42.8 → 27.6 s/step; total clip 161.9 s → 115.8 s. Forward gain grows with sequence: 5.4% at 1,456 tokens, 15.3% at 4,368, 24.8% at 13,104. Peak denoise memory 9,859 → 9,539 MiB; probe peak 7,427 → 5,827 MiB. Accuracy unmoved: parity fast rel_l2 0.004515 → 0.004498, exact bit-identical at 2.355890268907824e-6; fast-vs-exact step-1 0.0166 → 0.0180 (gate 0.05), final 0.2035 → 0.2114 (gate 0.35), 24.06 → 23.69 dB (gate 20). New `softmax_bf16` kernel check: rel_l2 8e-4–1.9e-3 vs an f64 reference. Cost $0.183 for both halves.

### FVID · 2026-09-17 · FVID-2026-09-17-flash-sdpa-rejected
- Trigger: at 48k tokens (8s clip) a DiT forward costs ~43 s with the default dense SDPA, and attention dominates; the opt-in `flash_attn_f32` kernel was never measured at clip scale, so `FASTVIDEO_SDPA=flash` was A/B'd on a pinned RTX A5000 against run `20260917T175630Z-clip`
- Options: adopt flash as default; tune the existing kernel; rewrite it with query tiling + tensor cores; keep dense and cut its memory traffic; port upstream VSA sparse attention
- Decision: **flash stays opt-in and off**; dense remains the default. The kernel is not tunable into competitiveness — `cfg_flash` launches one block per query row (`grid = bh × sq`, `block = d`), so at 48k tokens 576,576 blocks of 128 threads each stream the WHOLE of K and V through shared memory. K/V traffic per attention call is `bh × sq × sk × d × 2 × 4 B` ≈ 28 TB, hundreds of TB per forward across 30 layers; at ~768 GB/s that is the measured cost. It trades away the materialized score matrix (110 GB/layer) for K/V re-reads three orders of magnitude larger, so the penalty GROWS with sequence length: measured 12.6× slower at 1,456 tokens, 21× at 4,368, 35.8× at 13,104, and the probe's budget gate aborted the run projecting 1,795 s per forward at 48,048 tokens (vs 43 s dense). It is also scalar FMA + warp-shuffle per key, forfeiting the tensor cores cuBLAS gets in the dense path.
- Reason: the hypothesis was that a flash-style kernel wins at long sequence because it never materializes scores; the measurement says this implementation loses for a different reason (no query tiling ⇒ no K/V reuse), and a real fix is FlashAttention-2 (query tiles, MMA on bf16, double-buffered loads), not tuning
- Reversibility: free (nothing changed in the default path; `FASTVIDEO_SDPA=flash` still selects it)
- Executed by: Executor
- ADR: none (no architecture change; negative result recorded)
- Verification: run `20260917T183200Z-clip` on a pinned RTX A5000, `FV_STAGE_ENV="FASTVIDEO_SDPA=flash"` — kernels/model/parity stages PASS (flash is numerically correct: parity exact rel_l2 2.3e-6), timings above, stage `probe-8s` rc=3 (budget). Cost $0.053. Next step recorded separately: cut dense's score-matrix traffic (bf16 probabilities), then evaluate the VSA port.

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

