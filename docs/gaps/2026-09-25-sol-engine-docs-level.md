# Sol-Engine gap analysis: fastvideo-rs vs the published NVlabs Sol-Engine docs

Date: 2026-09-25 (first pass; superseded in detail by the code-level analysis the same day)
Compares: the published Sol-Engine docs (pipelines, techniques, agent workflow, as of 2026-09-25) against `docs/scope.md`, `crates/fastvideo-models/src/**/sol*.rs`, `crates/fastvideo-cudarc/src/wan/sol_ops.rs`, `fastvideo-gpucheck`, `scripts/gpu/runpod-matrix.sh`, and the 2026-09-24 decision-log entries (WS-A..WS-J, Phase 3, H200 warm suite).

Status key: **done** = wired + GPU-run; **partial** = wired, host-tested only;
**scaffold** = scaffold / constants; **missing** = not in tree.

## Headline

| | |
|---|---|
| Headline Sol-Engine pipelines with a GPU-measured optimised run here | **0 / 6** |
| Pipelines with contracts or constants ported (SANA-Video absent) | 5 / 6 |
| Technique entries wired and GPU-run | 4 / 23 |
| Sol-Engine quality gates implemented (LPIPS, VLM, authenticity) | **0 / 3** |

**The pattern across every row:** contracts are ported faithfully. Thresholds,
dense-layer sets, tau schedules, call indices all match the published tomls
and have host unit tests. What is missing is the second half of Sol-Engine's
loop: run it on the target GPU, gate the frames, record that the technique
actually engaged, and only then call it a speedup. Nine of the ten 2026-09-24
workstream entries close with "GPU untested here".

## A. Pipeline coverage

The six headline models plus the two legacy pipelines Sol-Engine still
documents. Status reflects the optimised line, not whether the base model
generates. Sol-Engine speedups are warmup-excluded medians on GB200.

| Model | Sol-Engine line | Sol-Engine result | Status | In fastvideo-rs | Gap |
|---|---|---|---|---|---|
| Wan2.2 TI2V-5B | lossless kernel fusion 1.52x; EasyCache 0.036 1.90x | 2.89x, 1x GB200, 704x1280, 121 f, 50 steps | partial | Preset registered. EasyCache `Fullstack` profile 0.036/7/1 + PISA-5B route (dense layers 0-3, 26-29; steps 0-3, 47-49) behind `FASTVIDEO_WAN_EASYCACHE_PROFILE` / `FASTVIDEO_WAN_PISA`. | Never run on GPU (Phase 3 matrix only ran Wan 1.3B). No QKV merge, no cross-attn K/V cache, no regional compile equivalent. Sol-Engine's line for this model is kernel-only + cache; PISA is optional here. |
| Wan2.2-A14B (MoE) | kernel fusion 1.13x; EasyCache 0.30 1.42x; PISA 0.10 1.35x | 2.17x, 1x GB200, 720x1280, 81 f, 40 steps, dual CFG | partial | A14B cache controller (block-0 fresh, blocks 1-39 residual, per-expert accumulators, thr 0.30 / start 5 / tail 3). PISA-A14B route (dense layers 0-3, 40-43; steps 0-3, 37-39). MoE `transformer_2` + dual CFG on cudarc. | GPU-unverified (WS-E logged "GPU kernels untested here"). A14B needs 48 GB+; no matrix cell exists. Sol-Engine batches CFG; here TeaCache CFG batching exists but EasyCache CFG pairing is unconfirmed. |
| LTX-2.3 | fusion; fixed-step cache; PISA; NVFP4 video FFN; midpoint token prune | 2.40x, 1x GB200, 1088x1920, 241 f | partial | Base + distilled presets. Stage-1 res2s (29 calls) with SCSP skipping calls 16-28. Stage-2 3-forward PISA (sparsity 0.9, block 64). Feature-norm midpoint prune (ratio 0.5, steps 1-2) with gather/scatter in the transformer. NVFP4 `scope_rule` for video FFN. | Every GPU attempt failed on loader contracts: Phase 3 `keyframes_abs_pos_embedding` -> `prompt_adaln`, H200 `decoder.mid_block...conv1` VAE key. The full 5-technique stack has never produced a frame. NVFP4 is dequant-to-bf16, so it contributes 0x speed. |
| Cosmos3-Super (64B) | TeaCache; NVFP4 (first/last steps dense) | 2.26x, 4x GB200, 1280x720, 189 f, 35 steps | scaffold | `cosmos3` host + cudarc scaffold. Official canvas, FlowMatch shift 10, TeaCache constants reused from Predict2 (1.15 / start 10 / max 3), `fp4_linear` step scoping. | DiT dims, Hub id, text encoder, VAE all `TODO(upstream)`. Not in the registry or CLI. No sequence-parallel path for a 64B DiT; the single-GPU plan (NVFP4 weights on a 96 GB card) has no real FP4 GEMM to run on. |
| SANA-Video 2B | EasyCache 0.1; linear-attn BF16; QKV merge; compile | 2.77x, 1x GB200, 832x480, 81 f, 50 steps | missing | Nothing. `docs/scope.md` records it as out of tree (SANA main branch is not implemented). | Whole pipeline: linear-attention DiT, DC-AE VAE, Gemma text. The Hub id `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` is public; the prior note that the profile wraps a private bundle is stale. |
| LingBot-Video 30B-A3B | cuDNN attn 1.79x; refiner PISA 0.10 1.12x; per-stage EasyCache 1.30x | 2.60x, 4x GB200 CP4+FSDP, 480p base -> 1088x1920 refiner | partial | Dense 1.3B and MoE 30B base presets, official 832x480 / 121 f / 40-step canvas, device MoE routing (`topk_last` + expert GEMM + `scatter_add_rows`). | No 1080p refiner or upsampler. `lingbot::sol::GAP` says the algorithms are "unspecified", but the published page now names them: base EasyCache 0.08, refiner EasyCache 0.25, refiner-only PISA density 0.10. No CP/FSDP topology. |
| Wan2.1-T2V-14B | compile; EasyCache; Sol-Attn tau 1.0 (first layer + first steps dense) | benchmark pending upstream (retired backend numbers withdrawn) | partial | `Sol14b` route: tau 1.0, 10 dense steps, layer 0 dense, Morton3D reorder -> `sol_attn::sol_attn`. EasyCache `Tuned14b` 0.10 / retain 5. 14B preset registered. | GPU-unverified for Sol-Attn on Wan. The only Sol-Attn kernel run to date (H3 on H200) was 8-14x slower than dense; see section D. |
| HunyuanVideo-13B | compile; TeaCache 0.15 / start 6 / max 2; Sol-Attn tau 1.0 with text sink | benchmark pending upstream | missing | Different family: HunyuanVideo 1.5 (480p/720p/1080p-SR). `FASTVIDEO_HUNYUAN15_SOL` logs the 13B gap and stays dense. | 13B is not a pipeline here. Even 1.5 fails on the Diffusers key layout (`Hunyuan15.double_blocks.*` vs `transformer_blocks.*.attn.to_q`) in both Phase 3 runs. |

## B. Technique coverage

Every entry Sol-Engine lists under its five acceleration methods, against
what exists in this tree. Kernel inventory from
`crates/fastvideo-cudarc/src/wan/kernels.cu` (`__global__` names) and
`wan/fused.rs`.

| Family | Method | Where Sol-Engine uses it | Status | In fastvideo-rs |
|---|---|---|---|---|
| Cache | TeaCache | Cosmos3-Super, HunyuanVideo-13B | done | Wan 2.1 polynomial (`FASTVIDEO_TEACACHE=1`, batched-CFG), H3 RTX residual controller, Cosmos Predict2/3 constants. Wan path GPU-run. |
| Cache | EasyCache | SANA, Wan-5B, Wan-14B, Wan-A14B, LingBot | partial | Three Wan profiles + A14B controller in `wan/sol_cache.rs`. Host tests only; no GPU cell has exercised it. |
| Cache | TaylorSeer | cache-dit adapter (baseline family) | partial | Wan lite: forecasts `proj_out`, interval 3 / warmup 3 / order 1 (`FASTVIDEO_WAN_SOL_CACHE=taylorseer`). Host-tested. |
| Cache | Fixed-step cache | LTX-2.3 stage-specific | partial | LTX-2.3 SCSP (`8of15_last_29calls`) + LTX-2.5 stage-1 FBCache (thr 0.08, max 10). Never completed on GPU because LTX-2.3 loads fail. |
| Cache | Cache-DiT / DBCache | related baseline | missing | - |
| Sparse attn | PISA | LTX-2.3 stage 2, Wan-A14B, LingBot refiner | partial | Device stages in `wan/sol_ops.rs` (tile means, top-k, first-order remainder, LSE merge). Routed for LTX-2.3 / Wan-5B / Wan-A14B. No GPU timing or quality number yet. |
| Sparse attn | Sol-Attn | Wan2.1-14B, HunyuanVideo-13B (CuTe sm90/sm100/sm120) | done | NVRTC kernels (`sol_diag_threshold`, `sol_exact_lists`, fine/coarse partials, LSE merge). Ran on H200 for sol-h3: 341-543 s/step vs 39.6 s dense. Correct-shaped, not fast. |
| Sparse attn | SpargeAttention | external adapter | missing | - |
| Sparse attn | Sparse VideoGen / SVG2 | local backend + adapter | missing | VSA (FastVideo's block-sparse, not a Sol-Engine entry) is the one sparse path that is GPU-proven: 1.81x on Wan 8 s, 7.0 s/step on FastH3. |
| Quantization | NVFP4 | Cosmos3-Super, LTX-2.3 video FFN (TransformerEngine, Blackwell) | scaffold | TE `static_6` scale rule, device `nvfp4_reconstruct` dequant -> bf16 GEMM. Tile-IR W4A4 GEMM exists but is off until it beats cuBLAS bf16 and hits >= 30 dB. Zero speedup today. |
| Quantization | ModelOpt / FP8 | practical family | partial | FP8 linears measured: -4.8% time at 16.9 dB (Wan) and 16-20 dB (H3 FFN). Rejected on quality; flag stays off. |
| Quantization | SageAttention | attention 8/4-bit family | missing | Named in FVID-2026-09-18 as the largest untested lever; nothing built. |
| Quantization | SVDQuant / Nunchaku | 4-bit diffusion | missing | - |
| Quantization | Diffusion PTQ (PTQ4DiT, Q-DiT, ViDiT-Q) | reference adapter | missing | QAD checkpoints are registered but run unquantised. |
| Kernel fusion | AdaLN + residual gate | Wan-14B compiled glue | done | `ln_adaln_e`, `residual_gate_add_e`, `ln_adaln_e_rope_half` NVRTC kernels; H3 `BlockMods` uploaded once. |
| Kernel fusion | QK-norm + RoPE | Wan-14B | done | `qk_norm_rope_bhsd`, `rope_half`, `rope_real`. |
| Kernel fusion | GEMM epilogues | CUTLASS / ByteTransformer style | partial | Post-GEMM kernels (`cast_bf16_f32_bias_act`, `bias_gelu_inplace`, `swiglu_value_first`); separate launches, not cuBLASLt epilogues. cuBLASLt is used only in `fp8.rs`. |
| Kernel fusion | QKV merge | SANA, Wan-5B, Wan-14B | partial | H3 fused QKVG `Linear::load_fused` measured as a wash (125.5 s vs 123.5 s). Not applied to Wan. |
| Kernel fusion | Attention output gate | output proj + gate glue | missing | Not identified as a fused path. |
| Kernel fusion | torch.compile / regional compile | SANA, Wan-5B, Wan-14B, Hunyuan | partial | No compiler tier in Rust. CUDA graphs measured at 0.7% of step time and rejected; hand-fused kernels are the substitute. |
| Kernel fusion | cuDNN attention backend | LingBot 1.79x, Wan-A14B `_native_cudnn` | missing | cuDNN is used for conv3d only. SDPA is cuBLAS GEMM + softmax with bf16 probabilities; flash-style and fused block-sparse were both measured and rejected. |
| Token pruning | Feature-norm pruning | LTX-2.3 midpoint | partial | `enable_midpoint_prune` in the LTX transformer (ratio 0.5, steps 1-2, prev-hidden reconstruction). Host-tested. |
| Token pruning | ToMe-SD | external adapter | missing | - |

## C. Approach: the agent-workflow page vs how this repo works

| Area | Sol-Engine | fastvideo-rs | Status |
|---|---|---|---|
| Orchestration | Master orchestrator + up to 3 executor sub-agents split by technique (kernel / cache / sparse). Executors run detached, watchdog-guarded, and return validated config; the master gates, dedupes, integrates. | Workstreams WS-A..WS-J merged 2026-09-24 through git worktrees with a decision-log entry each. Split is by model / feature, not by technique. No watchdog. Nine of ten entries end with "GPU untested here": executors return host-tested code, not validated configs. | partial |
| Config contract | One TOML per arm: `id`, `kind` (baseline / optimized / control), `model_profile`, `[runtime]`, `[env]`. `run.py --print` dry-runs paths; `--set` overrides; every run writes `launch.sh`, `manifest.resolved.toml`, `metadata.json`, `benchmark.json`. | `--config file.toml` overlays generate knobs (h/w/frames/steps). Techniques are env flags, same shape as Sol-Engine's `[env]`, and all-off is bit-identical (`exact` mode). But there is no `kind`, no resolved manifest, no dry-run, and the matrix cells are hard-coded shell cases in `runpod-matrix.sh`. | partial |
| Quality gate: LPIPS | LPIPS against frozen baseline frames. | rel_l2 / PSNR / cosine vs oracle tensors or vs the `exact` path; reference-free sanity gates (flat, frozen, saturated, luma flash). No perceptual metric. No frozen per-model baseline frame set. | missing |
| Quality gate: VLM rubric | Hosted VLM reviews baseline-vs-config side by side for snow, blur, mosaic, banding, ghosting, melting, flicker, motion regressions. | Contact sheets are written for human review. Nothing automated reads them. (VSA output noted as "more saturated" and never checked against upstream.) | missing |
| Quality gate: authenticity | Confirms the technique engaged (PISA wrote its stats, cache actually reused steps) so no-op optimisations cannot report a speedup. | Partial. `stats::host_algorithm` refuses to run scalar CPU math when a device is live (blocks the worst fake). `FASTVIDEO_DEVICE_STATS` prints per-op dispatch. `teacache hit #n` is a debug log. Nothing structured lands in the run JSON: no reuse count, no PISA density stats, no per-technique "engaged" field. | partial |
| Stated denominator | Baselines and same-topology controls are first-class; speedups name single-GPU / same-topology / vs-naive. | Strong in practice: same-box A/B (`FV_STAGE_ENV`), Phase 0 same-card denominators, upstream head-to-head on one rented box. Weak in form: the denominator lives in prose in `decision-log.md`, not in a machine-readable baseline id per run. | partial |
| Timing conventions | Exclude model load; medians over official validation prompts (5 for Wan, 3 for LingBot); same seed; warmup excluded. | Load / denoise / generate split out; `--warm` untimed pass; seed fixed at 1024. Single prompt (`FV_PROMPT`), single run per cell: no median, no official prompt set. | partial |
| Bit-exactness marking | Techniques that change FP reduction or sparsity are marked non-bit-exact. | `exact` vs `fast` mode is documented and gated (step-1 rel_l2 <= 0.05). No per-technique bit-exact flag in outputs; the LTX-2.3 fusion note that upstream turns some fusions off "because they change frames" is recorded only in `scope.md`. | partial |

## D. Evidence: the one Sol-Attn kernel run

MiniMax-H3 / FastH3 denoise seconds per step, 1x H200 SXM, image
`build-ec5a6cc0988aca26`, warm, same prompt and seed 1024, TAEH3 decode
(`FVID-2026-09-24-h200-warm-suite`). Sol-Attn steps are "sol-attn kernel"
per the run log.

| Cell / step | s / step |
|---|---:|
| fasth3 4-step VSA 0.9 | 7.0 |
| fasth3 8-step | 8.0 |
| fasth3 4-step dense | 39.6 |
| sol-h3 step 1 (dense by policy) | 39.6 |
| sol-h3 step 2 (Sol-Attn) | 543.3 |
| sol-h3 step 3 (Sol-Attn) | 433.0 |
| sol-h3 step 4 (Sol-Attn) | 341.0 |

**What this proves (authenticity: pass).** The kernel engaged: dense step 1
at 39.6 s matches the dense cell exactly, and steps 2-4 are a different code
path. That is the authenticity check Sol-Engine asks for, done by hand.

**What it does not prove (speed: fail).** Sol-Attn is 8.6-13.7x slower than
the dense path it replaces, and 49-78x slower than VSA. Sol-Engine ships CuTe
kernels for sm90/sm100/sm120; the NVRTC stages in `wan/sol_ops.rs` (tile
sums, exact lists, fine/coarse partials, LSE merge) are correct-shaped and
unprofiled. Two earlier hand-written attention kernels here (flash, fused
block-sparse) were also measured and rejected.

## E. Prioritised gaps

Ordered by how much of the Sol-Engine story each closes per unit of work.
S = a day, M = a few days on a rented GPU, L = a new port.

| # | Gap | Why it ranks here | Size |
|---|---|---|---|
| 1 | Profile and fix the Sol-Attn device path, or route Sol profiles to VSA until it is fast | The one Sol-Attn GPU run was 8-14x slower than dense; every Sol-tagged profile (Wan-14B, H3 4-step, LTX-2.5 stage 2) inherits that. VSA on the same card is 5.7x faster than dense. | M |
| 2 | Land one GPU cell for Wan2.2 TI2V-5B with EasyCache 0.036 on, and one with it off | Sol-Engine's simplest line (cache + fusion, no sparse), fits 24 GB, and would give the first GPU-measured EasyCache reuse count and speedup. Everything in `wan/sol_cache.rs` is host-only today. | S |
| 3 | Emit a per-run `benchmark.json` with `kind`, baseline id, technique-engaged counters, warmup-excluded step medians | Closes the config-contract, authenticity, and denominator gaps at once without new kernels. `perf.rs` already writes JSON; add fields. | S |
| 4 | Add LPIPS (or at minimum SSIM) against a frozen baseline frame set per model | Every speedup accepted so far was gated on rel_l2 / PSNR or eyeballed contact sheets. Sol-Engine accepts nothing without a perceptual gate. | M |
| 5 | Unblock LTX-2.3 loading (`prompt_adaln_single`, VAE decoder keys) and run the full stack once | Five techniques are wired for this model and none has produced a frame on GPU. The most-instrumented, least-verified pipeline in the tree. | S |
| 6 | Port the now-published LingBot numbers (EasyCache 0.08 / 0.25, refiner PISA 0.10) | `lingbot::sol::GAP` claims the algorithms are unspecified; the docs page now specifies them. The refiner and upsampler are the larger missing piece. | M-L |
| 7 | Decide NVFP4: real FP4 GEMM (Tile-IR bench + CUTLASS SM100/SM120) or drop the flag from the speedup story | Dequant-to-bf16 delivers 0x and misleads a reader of the env list into thinking NVFP4 is a live optimisation. | L |
| 8 | SANA-Video and Cosmos3-Super DiT | Two of Sol-Engine's six headline models are absent (one scaffold, one nothing). Both need new DiT ports; SANA's weights are public, Cosmos3-Super's dims are not. | L |

## Where this tree is ahead of the published Sol-Engine surface

Not part of the gap, but stated so the denominator is honest: an oracle tier
that attributes text-vs-DiT error against transformers/diffusers on
byte-identical tensors (found the UMT5 bias mirror); a `host_algorithm`
refusal that makes it impossible to log a kernel name while running CPU math;
per-shape timed conv3d backend selection; a cached CPU reference keyed on
source hash; and MiniMax-H3 / FastH3 / LTX-2.5 / Spark pipelines that
Sol-Engine's public docs do not cover at all.
