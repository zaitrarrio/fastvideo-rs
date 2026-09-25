# Codebase review: GPU utilisation, sol-engine alignment, cuda-oxide / pliron

Date: 2026-09-24
Compares: working tree at `cfe7fd8` (main, 2 commits ahead of origin) against a clone of `NVlabs/Sana@sol-engine`. Read-only audit; numbers come from `decision-log.md` and the 2026-09-23 Runpod sweep (RTX PRO 6000, CUDA 13.0). Nothing was re-measured.

Verdict key: **MATCH**, **MISMATCH**, **NOT IMPL**, **INVENTED** (exists here, not upstream), **CTX** (same numbers, different pipeline context), **HOST** (runs on the CPU).

## Headline

| | |
|---|---|
| Kernels via cuda-oxide / cutile / pliron | **0** |
| Sol-Attn, PISA, SLA execution | **CPU** |
| Activation storage between every op | **f32** |
| C++ kernels in `kernels.cu` (nvcc AOT, 7 SMs) | 65 |

The GPU is well used where the code has a real device path (cuBLAS bf16
GEMMs, the `mma.sync` VSA fine stage, fused SwiGLU, TAEH3, cuDNN bf16 conv),
and the decision log shows several fashionable levers were measured and
correctly rejected (CUDA graphs 0.7%, FP8 linears, TMA, scalar flash). It is
**not** maximized: every sol-engine sparse-attention route (Sol-Attn, PISA,
SLA) copied Q/K/V to the host and ran scalar CPU math, activations lived in
f32 with a cast sandwich around every linear, dense SDPA materialised the
score matrix, and Cosmos and LingBot did RoPE / MoE routing on the host.
Parameter values matched upstream almost everywhere, but several pipelines
diverged structurally (Spark draft, LTX-2.5 stage-2 sampler and LoRA target,
FBCache scope, LTX-2.3 SCSP indexing, Wan A14B controller). cuda-oxide and
pliron had zero runtime footprint: vendored, toolchain built in Docker, no
fastvideo kernel compiled through them.

## 1. Are we maximizing the GPU?

H3 denoise seconds per step by attention route, RTX PRO 6000, 5 s clip, same DiT (source `artifacts/runpod/h3/*/stderr.log`, 2026-09-23):

| Route | s / step |
|---|---:|
| fasth3 8-step (VSA `mma.sync`) | 10.9 |
| sol-h3 4-step (dense cuBLAS SDPA) | 75.4 |

Dense is ~7x slower per step. Upstream runs one-GPU H3 with Sol-Attn (RTX 5090 4.52x, RTX 4090 4.44x) while ours was dense by decision.

### Ranked gaps

| # | Gap | Where | Evidence / impact |
|---|---|---|---|
| 1 | **Sol-Attn, PISA and SLA run on the CPU.** `host_cow()` on Q/K/V, scalar reference math in `fastvideo-models`, upload result. Logged as "sol-attn kernel". | `cudarc/src/sol_attn.rs:140-168`, `pisa_attn.rs:52-63`, `wan/sla.rs:63-95` | Hit by LTX-2.5 stage 2, LTX-2.3 PISA, H3 Spark/RTX routes, Wan SLA. GPU idle for the whole attention. Why `ltx25-two-stage` never produced a clip inside 5 min. |
| 2 | **Cosmos and LingBot do per-block host work.** Cosmos RoPE on host each attention call; LingBot router softmax/top-k on host and one `[1, d]` expert GEMM per token per expert. | `cosmos/transformer.rs:120-127`, `lingbot/transformer.rs:212-254` | Device->host->device sync per block per step, thousands of tiny launches. Unmeasured; structurally the worst code path in the tree. |
| 3 | **f32 activations, bf16 compute.** Every `Linear` is `cast_f32_bf16 -> gemm_ex -> cast_bf16_f32_bias_act`; norms, residuals, attention probs move 4 bytes/elem. | `wan/nn.rs:784-805`, `wan/device.rs:29-46, 523-576` | Upstream is bf16 end to end. Named as the next H3 lever in `decision-log.md:219`; FFN+QKVG GEMMs are ~60% of a step. |
| 4 | **Dense SDPA is not flash.** cuBLAS QK^T to an f32 score matrix, `softmax_last_bf16`, P@V, chunked by a 1 GiB budget. | `wan/attn.rs:110-214` | Default for every non-VSA self-attn, all cross-attn, LTX-2, Hunyuan 1.5, Cosmos, LLM encoders. H3 dense 87 s/step vs VSA 11.8 (`decision-log.md:216`). |
| 5 | **LTX-2 two-stage reloads the DiT from disk to change LoRA strength.** LoRA fused on the host at load time. | `ltx2/pipeline.rs:1143-1157`, `ltx2/lora.rs:38-60` | DiT load 138.6 s on the Runpod volume; the reload dominates two-stage wall time and caused the 95 GiB OOM with Gemma resident (now forced streamed). |
| 6 | **Text encoders once resident:** composed masked SDPA with f32 `[1,H,S,S]` scores, materialised GQA `repeat_kv`, FP8 rows dequantised to a full bf16 weight before every GEMM. | `llm.rs:646-712, 833-847`, `nn.rs:773-781, 1030` | Today disk-bound (89-91 s/prompt streamed on Runpod, 3.9 s warm on Vast). Becomes the term once weights are resident. |
| 7 | **Unfused glue.** H3 modulation re-uploaded per block per step via pageable memcpy; Hunyuan/LTX/Cosmos AdaLN and gates are 3-5 separate launches although `ln_adaln_e` / `residual_gate_add_e` exist (used by Wan only). LTX-2 Euler step round-trips through host memory. | `h3/transformer.rs:524-595`, `hunyuan15/transformer.rs:44-48`, `ltx2/pipeline.rs:381-416` | Small individually; all views (narrow/cat/permute) also copy. 3,112 launches per Wan step. |
| 8 | **Only one CUDA stream** (`default_stream()`, blocking sync). Second stream exists only for the LLM layer prefetcher. | `wan/device.rs:76-96`, `llm/prefetch.rs:81` | No overlap of VAE decode / text encode / H2D with denoise. Defensible for a single-model hot loop; costs on two-stage pipelines. |
| 9 | **NVFP4 is host fake-quant.** `dequant_beforehand` is host_cow -> CPU reconstruct -> upload per linear and per K/V. The `nvfp4_w4a4_gemm` kernel is a scalar 16x16 tile GEMM reached only from tests. | `wan/nvfp4.rs:244-293`, `kernels.cu:1799-1849` | Off by default. Not a device path; do not describe it as one. |

### Already good, do not redo

- AOT cubins for sm 75/80/86/89/90/100/120, NVRTC fallback, banner reports kernel origin.
- VSA fine stage on `mma.sync` bf16 + `cp.async`/TMA: 5.42x over gather; self-attn 74.7% -> 33% of DiT.
- bf16 attention probabilities (-35.6% denoise); bias/GELU fused into the widen; fused QKV(G) GEMMs on Wan/H3/Hunyuan.
- `swiglu_value_first` 7.0 -> 1.83 s; RMSNorm+SiLU and per-shape bf16 conv in the VAE (-10.8%).
- TAEH3 decode 28.97 -> 0.98 s; Wan VAE 16.18 -> 1.66 s; streaming frame writer.
- Mempool release threshold pinned so freed blocks stay cached; CFG batched into one forward; Wan scheduler on device.
- `FASTVIDEO_STRICT_DEVICE` catches silent CPU fallbacks for ops that have a kernel (Sol/PISA/SLA/NVFP4 are outside that guard because they were host by design).

### Measured and rejected, keep off

| Lever | Result |
|---|---|
| CUDA graph capture | 0.7% (3,112 launches ~ 16 ms of 2,237 ms/step) |
| `FASTVIDEO_FP8` E4M3 linears (Wan) | -4.8%, load 2x, 16.9 dB frames |
| H3 FFN-only E4M3 | -4.9%, 20.3 dB |
| MLX affine INT8 fused GEMM (scalar) | 38x slower |
| `flash_attn_f32` scalar SDPA | 12.6-35.8x slower |
| `vsa_fused_attn` scalar block-sparse | 5.5-8.6x slower than gather |
| TMA loads on `vsa_mma_attn` (sm90+) | noise: 123.5 vs 124.7 s |
| Fused QKVG GEMM (H3) | wash, +4 GiB peak |
| tcgen05 / UMMA for RTX PRO 6000 | not a valid target: SM12x has no TMEM; `mma.sync` is peak there |

## 2. Alignment with sol-engine spec and upstream

Compared against `/tmp/sol-engine` (NVlabs/Sana, branch `sol-engine`). Values cited to the upstream file that defines them. Parameter values matched in most places; the mismatches were structural.

### Sol-Attn backend contract

| Item | Verdict | Ours | Upstream |
|---|---|---|---|
| Block 64, diag threshold mean+tau*std in log2, eps 1e-6, +/-1 local window, sink->exact, exp2 compensation | MATCH | `models/sol_attn.rs` | `sparse_backends/sol_attn/common/selector.py` |
| Execution | HOST | CPU reference via `host_cow`; logs "sol-attn kernel" | CuTe SM89/90/100/120 + Triton GPU kernels |
| exact `thresh_type`; `kv_splits` 2/4; head-dim 128 / BF16 / BTHD validation | NOT IMPL | diag only, any dim, f32 | `interface.py`, `preprocess.py` |
| Multiple sink spans | INVENTED | `sol_attn_bhsd_sunk(sinks: &[(Option<usize>, usize)])` | one contiguous `sink_start` / `sink_tokens` |

### MiniMax-H3

| Item | Verdict | Ours | Upstream |
|---|---|---|---|
| One-GPU Sol-H3 dense (standing decision) | MATCH | `h3/config.rs:610-623 dense: true, vsa 0` | Sol-H3/README: "dense attention on 1 GPU". Note upstream's RTX 5090/4090/GB10 H3 numbers use Sol-Attn on one GPU; our comment "SOL/BSA is the multi-GPU profile" is not what upstream ships. |
| Sol-H3 sigmas (5-pt grid, shift 12/3) | MATCH | `h3/schedule.rs` | `Sol-H3-Spark/configs/default.json` |
| Spark stage-1 attention | MISMATCH | dense | VSA_cuDNN_BSA sparsity 0.9 tile 64 |
| Spark stage-1 LoRA | MISMATCH | dense-datafree alpha 64 | FastH3_VSA_DataFree strength 1.0 |
| Spark stage-1 DiT precision | MISMATCH | BF16 | W8A8_FP8_after_BF16_LoRA_merge |
| `FASTVIDEO_H3_SOL_ATTN=spark` route (layer0/step0 dense, tau 1/1.25/1.5) | CTX | `h3/sol.rs:100-117` as a T2V route | Ref2VA-only opt-in (`--ref-stage1-attn sol`) |
| Sink layout | MISMATCH | text+audio as multiple spans in native order | tokens permuted to [visual \| sinks], single suffix sink |
| RTX route: 10 dense steps, 2 dense layers, tau 1.0, 49 forwards | MATCH | `h3/sol.rs:121-131` | `models/minimax_h3.toml [rtx4090.policy]` |
| TeaCache 0.10 / retain 5 / cooldown 1 / coeffs [1,0] | MATCH | `h3/sol.rs:179-294` | `RTX4090/teacache.py` |
| Spark bridge: x2 upscale -> adapter -> crop 16 -> 3-step refine, LoRA 0.8, layer 0 dense | MATCH | `models/h3/spark.rs`, `cudarc/h3/spark.rs` | `runtime/stage2.py` |
| Refiner prompt | MISMATCH | request prompt, online Gemma | fixed "4K, refined, ..." with offline INT8 Gemma cache |
| Refiner sampler | MISMATCH | ancestral eta=1 | deterministic |
| H3 FirstBlockCache 0.08 (H100 profile) | NOT IMPL | - | `H100/first_block_cache.py` |
| GB200 super-acceleration v2 bridge | NOT IMPL | `scope.md` correctly says Spark, not this | `super_acceleration/STAGE2_CONTRACT.md` |

### LTX-2.3 and LTX-2.5

| Item | Verdict | Ours | Upstream |
|---|---|---|---|
| 2.3 HQ canvas 1920x1088x241, 15 steps, guidance 3.0, LoRA 0.25/0.5 | MATCH | `ltx2/hq.rs`, `ltx2/lora.rs` | `models/ltx23.toml` |
| 2.3 stage samplers | MISMATCH | Euler (`denoise_cfg`) | res2s both stages (`run_ltx23_common.sh:124-126`) |
| 2.3 PISA: layers 0-1 dense, sparsity 0.9, block 64; midpoint prune 0.5 at steps 1,2 | MATCH | `ltx2/pisa.rs` | `ltx23/optimized/env.sh:70-96` |
| 2.3 SCSP `8of15_last_29calls` | MISMATCH | skips steps 16-28; a 15-step stage 1 never reaches step 16, so the cache is inert | skips res2s calls 16-28 of 29 (`presets.py:29-33`) |
| 2.3 KWL fusions, NVFP4 video FFN | NOT IMPL | `scope.md` says so | `env.sh:29-90` |
| 2.5 stage-2 Sol: layer 0 dense, layers 1-47 tau 1/1.25/1.5, diag | MATCH | `ltx2/sol.rs:13-67` (but executes on CPU) | `ltx25/RTX5090/attention.py` |
| 2.5 stage-2 sampler | MISMATCH | `denoise_ancestral` eta=1 (`pipeline.rs:1484`) | "Stage 2 is always deterministic" (`distilled.py:206-209`) |
| 2.5 LoRA 0.8 target | MISMATCH | fused onto the distilled DiT whenever the file is present | applied to the dev BF16 DiT; distilled two-stage uses no LoRA |
| 2.5 FBCache 0.08 / warmup 1 / max 10, block-0 residual signal | MATCH | `ltx2/fbcache.rs` | `ltx_core/opt/step_cache.py` |
| 2.5 FBCache scope | MISMATCH | disarm at 1407 then `arm_requested` at 1439/1671; `begin_fbcache_step` sets armed=true -> active in stage 2 and Spark refiner | stage 1 only; Spark "cache-based reuse disabled" |
| 2.5 FBCache context | CTX | 8-step distilled ancestral CFG=1 | 30-step dev CFG+STG, 4 guidance passes |

### Wan, Cosmos, Hunyuan, LingBot, NVFP4

| Item | Verdict | Ours | Upstream |
|---|---|---|---|
| Wan EasyCache algorithm; TeaCache 0.12 / start 2; TaylorSeer lite | MATCH | `wan/sol_cache.rs` | `wan22_ti2v_5b/optimized/cache_runtime.py` |
| Wan EasyCache thresholds | CTX | 0.05 / 7 / 1 code defaults | delivered manifests use 0.036 (5B, 14B) or 0.10 retain 5 (14B tuned) |
| Wan A14B cache controller | MISMATCH | 5B whole-stack EasyCache, one state for both experts | block-0 fresh prefix + blocks 1-39 residual, per-expert accumulators, thr 0.30, start 5, tail 3 |
| Wan 14B Sol-Attn (tau 1.0, 10 dense steps, Morton3D); Wan 5B/A14B PISA 0.10 | NOT IMPL | no Sol/PISA route in `cudarc/src/wan` | `config/wan21_t2v_14b/fullstack*.toml` |
| Legacy `FASTVIDEO_TEACACHE` (TeaCache4Wan2.1 poly coeffs) | INVENTED | `wan/pipeline.rs:703-816` | not in sol-engine (different source) |
| Cosmos model family | MISMATCH | Cosmos-Predict2 2B/14B, EDM | `nvidia/Cosmos3-Super`, flow_shift 10 |
| Cosmos TeaCache 1.15 / start 10 / max 3; canvas 1280x720x189 / 35 / cfg 6 | MATCH | `cosmos/sol.rs` | `cosmos3/optimized/env.sh` |
| Cosmos NVFP4 on gate_up/down/qkv/out (skip first/last 3) | NOT IMPL | step selection only, not quantised (`scope.md` says so) | `env.sh:21-29` |
| Hunyuan 13B TeaCache | NOT IMPL | correctly gap-logged; upstream publishes no threshold | `hunyuan_video/optimized/step_cache_runtime.py` |
| LingBot refiner | NOT IMPL | GAP string accurate | upstream `optimized/` is a cuDNN baseline, no cache/PISA |
| NVFP4 recipe | MISMATCH | LongLive/FourOverSix MSE rule, Wan linears only, host fake-quant | TransformerEngine NVFP4BlockScaling W4A4 (~ our `static_6`), LTX-2.3 FFN + Cosmos3 |

### `scope.md` claims not backed by code

"Stage-2 Sol-Attn" and "Stage-2 PISA" as accelerations (both were CPU paths);
"FBCache stage 1 only" (it re-armed in stage 2 and the Spark refiner);
"Stage-1 SCSP skips steps 16-28" as a functioning cache on the 15-step HQ
contract (inert). Everything else in `scope.md` checked out.

## 3. Are we maximizing cuda-oxide and pliron?

**Footprint: zero.** No workspace crate depended on `cuda-device`,
`cuda-core`, `cutile`, or `pliron` (`Cargo.lock` had no such entries; only
`cudarc 0.17.8`). No `#[kernel]` or `#[cutile::module]` outside
`third_party/`. `scripts/oxide.sh` / `docker/oxide.Dockerfile` build the
toolchain itself (`librustc_codegen_cuda.so` and cutile rlibs into
`artifacts/oxide/`) and never run `cargo oxide build` on a fastvideo kernel.
Nothing in Taskfile, CI, or the decision log referenced them; the vendoring
commits (`09bd162`, `b110bd7`) had no decision-log entry.

### What is vendored

| Component | Version / pin | Notes |
|---|---|---|
| cuda-oxide | v0.2.1 (4514af2c), nightly-2026-04-03, LLVM 21 | `cargo oxide {run,build,pipeline,debug,setup}`; PTX/LTOIR/cubin artifacts; tcgen05, TMA, cluster, `gemm_sol` (58% of cublasLt SoL on B200) |
| cutile-rs | v0.3.1 (cdc69c1), MSRV 1.89 | Tile-IR kernels JIT'd via `tileiras` at runtime; flash attention (causal), gemm, nvfp4, mxfp8, `cudarc_interop` |
| pliron | transitive via cuda-oxide | MLIR-style IR framework the oxide backend lowers through; no direct use is possible or intended from fastvideo |
| oxide Docker | CUDA 13.4 + cuda-tileiras-13-4 | Runtime image and CI remain CUDA 13.0 |

### Blockers to using them

- Workspace is stable Rust 1.85 / edition 2021 with `unsafe_code = "forbid"`; the oxide backend is `rustc_private` on nightly; cutile needs 1.89.
- Runtime image is CUDA 13.0. cutile needs 13.1+ (sm_100), 13.2 (sm_8x), 13.3 (sm_90) plus the `tileiras` binary at runtime. Its JIT-at-first-use model conflicts with the AOT/no-JIT decision `FVID-2026-09-19-cuda13-aot-cubins-ci`.
- cudarc 0.17.8 loads PTX and cubin (`cuModuleLoadData`) but has no nvJitLink bindings, so LTOIR artifacts cannot be consumed without pulling in `cuda-host` and its nightly graph.
- Target hardware: RTX PRO 6000 is SM12x, no TMEM / tcgen05. Most of cuda-oxide's Blackwell-datacenter showcase (`gemm_sol`, `tcgen05_matmul`, CLC) does not apply to it.

### Where oxide / cutile plausibly pay off vs where cuBLAS / cuDNN already win

| Candidate | Verdict | Why |
|---|---|---|
| `nvfp4_w4a4_gemm` and `affine_w16_gemm` -> cutile Tile-IR GEMM | Best ratio | Both are scalar 16x16 FMA kernels measured 38x slower than bf16; cutile's `nvfp4.rs` / `mxfp8.rs` are the same layout on tensor cores. Both flags are off today, so purely additive. |
| sm_100 (B200/GB200) tensor-core VSA fine stage | Only if B200 enters the pool | `vsa_mma_attn` is `mma.sync` on every arch >= 80; on sm_100 it leaves tcgen05/TMEM idle. oxide's `tcgen05_matmul` / `gemm_sol` are the only working tcgen05 code available. |
| Non-causal flash SDPA in Tile IR | Real but needs writing | Would replace the materialised score matrix in `wan/attn.rs`. cutile's shipped attention is causal / sequence-shaped; the video variant does not exist. |
| Fused GEMM epilogues (bias+GELU, gate-residual, bf16 cast) | Modest | cuBLASLt epilogues cover part of this without a new toolchain; a Tile-IR GEMM with custom store fuses the rest. `h3_ffn_act` was 7 s of a 38 s FFN before swiglu fusion. |
| SIMT glue kernels rewritten in Rust | No | Memory-bound and identical in either language; costs a second toolchain and a per-SM artifact set. |
| Dense bf16/TF32 linears, FP8 linears | No | cublasLt is the baseline oxide's own `gemm_sol` benchmarks against and reaches 58% of. |
| cuDNN conv2d/3d | No | No Tile-IR conv exists in cutile-kernels. |

What it would take (not done, not recommended until Sol-Attn is on the
device): a nightly device crate outside `members` built only in the oxide
image; `scripts/oxide.sh` actually running `cargo oxide build --arch
sm_{80..120}` and exporting .ptx/.cubin; `fastvideo-cudarc/build.rs`
embedding them beside the nvcc cubins (no cudarc change needed for
PTX/cubin); for cutile, a runtime image bump to CUDA >= 13.3 with `tileiras`
and MSRV 1.89; and a decision-log entry. The vendoring is a bet on a compiler;
every measured gap in this review is fixable with the nvcc path that already
exists.

## 4. What to fix first (assessment only; nothing changed)

| # | Change | Unlocks |
|---|---|---|
| 1 | Device Sol-Attn: block-mean pooling + threshold selector as kernels, then reuse the VSA `mma.sync` fine stage with the selected tile list (same block-sparse shape; VSA already has gather/topk/combine kernels). PISA can share the selector. | LTX-2.5 stage 2, H3 Spark/RTX, Wan 14B Sol: the entire sparse-attention half of sol-engine becomes real instead of CPU. |
| 2 | LTX-2.5 stage 2: deterministic sampler; LoRA 0.8 only on the dev DiT; keep FBCache disarmed through stage 2 and the Spark refiner (drop the `arm_requested` at `pipeline.rs:1439/1671`). | Upstream parity for the two-stage and Spark bridge routes; quality of the 3-step refine. |
| 3 | Fix LTX-2.3 SCSP to index res2s calls (or remove the inert preset); pick res2s or document the Euler deviation. | Truthful `scope.md`; a stage-1 cache that actually skips work. |
| 4 | bf16 activations inside the DiT block (keep f32 residual stream if quality demands): removes the cast sandwich and halves memory traffic on every elementwise op. | H3 FFN+QKVG ~60% of step; largest remaining dense lever per decision log. |
| 5 | Move Cosmos RoPE, LingBot router/MoE, LTX Euler step, H3 BlockMods upload onto the device (existing `rope_half` / `lincomb3` / `index_select_rows` kernels cover most of it). | Removes per-block host syncs; makes Cosmos/LingBot measurable. |
| 6 | LoRA strength as a runtime scale (W + s*BA on device) instead of host-fused reload. | LTX two-stage wall time - ~140 s; no OOM window on strength change. |
| 7 | Spark draft: decide whether to follow upstream (VSA 0.9 + FastH3_VSA_DataFree + FP8) or keep dense; either way correct the config comment that calls SOL/BSA multi-GPU-only. | The Spark profile currently is not the Spark profile. |

## What happened after

Same day (2026-09-24, WS-A..WS-J): device Sol-Attn / PISA stages landed in
`wan/sol_ops.rs`, Wan 14B Sol and 5B/A14B PISA routes were added, the A14B
controller was ported, Spark stage 1 moved to VSA 0.9 + VSA-DataFree LoRA,
bf16 activations became an opt-in (`FASTVIDEO_BF16_ACT`), and
`stats::host_algorithm` began refusing CPU math under a live device. The
2026-09-25 code-level gap analysis re-audited the result.
