# Sol-Engine vs fastvideo-rs: code-level gap analysis

Date: 2026-09-25
Compares: NVlabs/Sana `sol-engine` branch source (HEAD `6c2f582`, 2026-09-21, cloned to `/tmp/sol-engine`) against the fastvideo-rs working tree. Every verdict cites lines on both sides; nothing here comes from the published docs pages. Compiled from five code-diff passes (Wan, H3, LTX, LingBot/Cosmos3/SANA/Hunyuan, Sol-Attn/PISA kernels) plus direct verification of the sink span, dense-step clock, and coarse-kernel launch shapes.

Rust paths are relative to `crates/fastvideo-cudarc/src` or
`crates/fastvideo-models/src`. Sol-Engine paths are relative to
`models/<model>/optimized` or `techniques/`.

Verdicts: **same** = identical rule and constants; **param drift** = same
algorithm, different constants, scope, or clock; **algorithm drift** = a
different computation; **missing** = no Rust counterpart; **n/a** = absent on
both sides.

## Headline

| | |
|---|---|
| Techniques compared | 60 |
| same | 12 |
| param drift | 14 |
| algorithm drift | 13 |
| missing in Rust | 20 |
| absent both sides | 1 |

The controllers are ported faithfully: EasyCache, TeaCache, TaylorSeer, A14B
guards, Cosmos TeaCache, H3 TeaCache, LTX FBCache, Sol diag threshold, coarse
term and PISA top-k all match the Python line for line. The drift is in what
surrounds them: step clocks, CFG batching, sink layout, sampler variants, and
the fused-kernel structure. And nothing in Sol-Engine's evaluation loop
(off-identity, paired pixel metrics, LPIPS/VLM gate, reuse counters, 5-prompt
medians) has a Rust counterpart, which is why two defects below survived two
GPU suites.

## 1. Verified defects

Read directly in both trees. The first two change every sol-h3 and Wan-14B
Sol measurement taken so far.

### H3 default sink span hits video rows

`h3/sol.rs:219-225` returns `(visual.len(), sinks)` as the sink span, which is
only correct after the Spark `[visual | text+audio]` permutation.
`h3/transformer.rs:817-819` passes the unpermuted sequence (text -> cond ->
audio -> video per `packing.rs:196-208`) straight to `sol_attn_sunk`; there is
no `index_select` or permutation anywhere in `cudarc/src/h3`. The kernel
therefore treats the *last* text+audio-many video tokens as exact sinks and
routes text/audio queries sparsely. Every sol-h3* cell on H200/B200 ran this
way. The log at `pipeline.rs:923-928` claims a permute that never happens.

### Wan 14B `dense_steps` counts denoise steps, upstream counts forwards

`sol_attn_backend.py:271-275` increments the step clock in
`sol_attn_begin_forward`, registered as a `transformer.forward` pre-hook
(`gpu_infer.py:259`). Under CFG that is two forwards per denoise step, so
`WAN22_SOL_DENSE_STEPS=10` means 5 dense denoise steps. `wan/sol.rs:11-16`
`SOL_14B_DENSE_STEPS=10` is compared against the pipeline's denoise-step
counter, giving 10 dense steps and roughly twice the dense work the manifest
specifies.

### Batched CFG couples TeaCache / TaylorSeer branch state

Sol-Engine never batches CFG; each branch owns its accumulator
(`cache_runtime.py:157-160, :392-393`). `wan/pipeline.rs:1059-1072` runs one
`[uncond, cond]` forward on the default T2V path, so `transformer.rs:1157-1160`
calls `note_computed` for both rows whenever either computes, and the
TaylorSeer step counter (`transformer.rs:1389`) advances per forward rather
than per branch on the unbatched paths.

### A14B controller signal is full-tensor, not the 64x128 probe

`cache_controller.py:148-155` subsamples 64 strided tokens and the first 128
channels for both the input delta and the residual norm, and all-reduces it.
`transformer.rs:1314` and `:1364-1369` feed `mean_abs_delta` over the whole
prefix and `mean_abs` over the whole residual to the same threshold 0.30. The
k estimate and the hit pattern will differ from the published run.

### sol-h3 T2V/I2V uses the Spark Ref2VA tau ladder

`h3/sol.rs:87-97` picks the Spark route for every sol-h3* recipe. Upstream
Sol-H3 T2V/I2V (`engine.py:278-285`) uses constant tau 1.0, dense_steps 1,
dense_layers 2 and a prefix sink; the escalating ladder with only layer 0
dense is guarded by `stage1.py:276-277` "this fixed Sol policy is Ref2VA
only". The doc comment at `sol.rs:5-7` misattributes it.

### LTX stage-2 prune compensation overwritten by uncond pass

`ltx2/transformer.rs:750` consumes the prune step on the first forward; when
`denoise_cfg` is on (`pipeline.rs:1686-1711`) the second forward runs unpruned
and `:791-799` replaces `prev` with the uncond hidden state. Python stage 2 is
`SimpleDenoiser` with one pass (`ti2vid_two_stages.py:287-311`).

## 2. Technique matrix

### Wan (18)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| EasyCache whole-stack (5B/14B/1.3B) | `cache_runtime.py:353-369` guards+accumulator, `:430-439` k refresh/payload; tomls 0.036/7/1, tuned 0.10/5 | `wan/sol_cache.rs:257-316` decide_cond/note_cond_computed/uncond_compute; `pipeline.rs:929-965` signals; profiles `:184-190` | same | |
| TeaCache (5B family, timestep_proj signal) | `cache_runtime.py:165-187` force order, `:241-267` per-branch state + whole-stack residual; never batched | `sol_cache.rs:472-596` same defaults/Horner; `transformer.rs:824-865` batched mode skips only if both rows reuse and resets both accumulators (`:1157-1160`) | algorithm drift | Separate-forward path is identical; the default batched T2V path couples cond/uncond decisions. |
| TaylorSeer lite (interval 3 / warmup 3 / order 1) | `cache_runtime.py:459-503` diffusers TaylorSeerCacheConfig, compute rule `(step-warmup-1)%interval==0` per branch | `sol_cache.rs:70-146` same formula, divided differences, forecast skips blocks+head (`transformer.rs:1563-1566`); step counter per forward_ctx (`:1389`) not per branch | algorithm drift | On unbatched paths (MoE, i2v, image) the counter advances twice per denoise step, halving warmup/interval. |
| A14B MoE cache controller (block-0 fresh, tail-39 residual) | `cache_controller.py:186-195` guards, `:148-155` probe = 64 strided tokens x first 128 channels, `:376-387` k from probe | `sol_cache.rs:715-850` same constants/guard order/threshold 0.30; `transformer.rs:1314,1364-1369` signal is full-tensor mean_abs_delta | param drift | Different k, estimate, and hit pattern than the probe. |
| A14B TeaCache (14B poly) / TaylorSeer (order 2, damping, forecast guard) | `cache_controller.py:105-121, :285-354` | only EasyCache variant exists (`sol_cache.rs:715-850`) | missing | |
| PISA routing (density 0.10, block 64, dense sets) | `wan_kernel_optimizations.py:127-166` env knobs + dense sets; kernel is an archived Sana `pisa_attention.py` NOT in repo (`:116-123`) | `wan/sol.rs:55-101` identical dense layer/step sets; `pisa_attn.rs:21-120` top-k of pooled QK + zeroth/first-order remainder | param drift | Schedule matches; mask/remainder math unverifiable. No DENSITY_RULES / PISA_LAYERS / KERNEL_NUM_STAGES / APPROX_REMAINDER knobs, no dense-fallback-on-exception. |
| Sol-Attn 14B routing (tau 1.0, diag, dense_steps 10, layer 0) | `sol_attn_backend.py:271-300` step clock increments per transformer.forward (`gpu_infer.py:259` pre-hook) => 10 forwards = 5 CFG denoise steps; reset only after warmup (`:429-433`) | `wan/sol.rs:11-16` SOL_14B_DENSE_STEPS=10 counted in denoise steps; `sol_attn.rs:118-175` threshold/rule match | param drift | Rust runs 2x the dense steps the manifest intends. No kv_splits, thresh_type=exact, SOL_ATTN_STRICT. |
| Sol-Attn on Wan2.1 1.3B | `wan21_fullstack_sol.toml`, `full.toml` (dense_steps 0) | `transformer.rs:1273` profile requires `num_layers >= 40 && !moe` => 1.3B (30 layers) gets `WanAttnProfile::Off` | missing | |
| Morton3D token reorder | `sol_attn_backend.py:209-232` one global permute before block 0, inverse after last block, RoPE tables permuted | `wan/sol.rs:108-139` identical bit-twiddle; `transformer.rs:178-201` + `sol_cache.rs:72-102` host_cow gather of q/k/v/out per attention call | param drift | Same output, D2H+H2D x4 per layer. |
| QKV / cross-KV projection fusion | `wan_kernel_runtime.py:217-226`; `wan_kernel_optimizations.py:402-463` | `transformer.rs:99-121` `Linear::load_fused` | same | |
| QK-norm + RoPE fusion | `wan_kernel_optimizations.py:872-1200` (Inductor compiled) | `fused.rs:162-235` `qk_norm_rope_bhsd` (one NVRTC kernel) | same | |
| Block glue (modulate, gated residual) | `wan_kernel_optimizations.py:503-515` | `fused.rs:20-58` `ln_adaln_e`, `:122-157` `residual_gate_add_e` (also fuses LN) | same | |
| bf16_block_glue (LN/AdaLN/gates in bf16) | `wan_kernel_runtime.py:49-87, :700-716` | f32 activations by default; `FASTVIDEO_BF16_ACT=1` opt-in (decision-log 2026-09-24-bf16-activations, GPU unmeasured) | param drift | |
| cross_kv_cache (packed KV + norm_k per block per prompt) | `wan_kernel_runtime.py:604-698`; `wan_kernel_optimizations.py:1809-1865` | `transformer.rs:339-341` recomputes cross K/V every block, every step | missing | |
| Invariant caches (text/time/patch embeddings) | `wan_kernel_runtime.py:523-602` invariant_cache_v2 with hit/miss/bypass stats | `transformer.rs:1577-1585` recomputed each forward; only rotary_cache (`:1536-1548`) is cached | missing | |
| cuDNN / flash native SDPA backend | `wan_kernel_runtime.py:742-764`; `wan_kernel_optimizations.py:1240-1274` | `attn.rs:98-215` cuBLAS QK^T + softmax + PV (bf16 probs opt) or `flash_attn_f32` opt-in (`:53-82`) | param drift | |
| Ulysses CP (packed/async A2A, ring2/ulysses2) | `wan_kernel_optimizations.py:1430-2033`; A14B `gpu_infer.py:153-185` | single GPU only | missing | |
| CFG execution | Diffusers runs cond then uncond; controllers pair by call parity (`cache_runtime.py:213-214, :392-393`) | `pipeline.rs:1059-1072` one batch-2 [uncond, cond] forward on default T2V; separate forwards on EasyCache/A14B paths (`:948-1010, :1073-1089`) | algorithm drift | Drives the TeaCache/TaylorSeer coupling above. |

### H3 (15)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| Sampler contracts (49 fwd shift 12/3; Sol-H3 4 fwd; Spark sigmas) | `minimax_h3.toml:29-31,40`; `engine.py:22,222-223,358`; Sol-H3-Spark `configs/default.json:10-11` | `h3/config.rs:556-679` recipes; `schedule.rs:150-164` step math; `:520-527` asserts Spark sigma values | same | |
| Sol-H3 T2V/I2V Sol-Attn policy | `engine.py:278-285` constant tau 1.0, dense_steps 1, dense_layers 2, sink_mode prefix (Ref2VA: 0/0/text_audio) | `h3/sol.rs:87-97` selects Spark route for ALL sol-h3* recipes: taus [1.0,1.25,1.5], step 0 dense, layer 0 dense (`:40,123-140`) | algorithm drift | Rust applies the Spark Ref2VA-only ladder (`stage1.py:276-277` raises for non-Ref2VA) to T2V/I2V. Doc comment `sol.rs:5-7` misattributes it. |
| RTX 4090/5090 route (step<10 \|\| layer<2, tau 1.0) | `rtx5090_fullopt.toml`; `RTX5090/adapter.py:452-461`, sink = text rows only (`:487-491`) | `h3/sol.rs:43-48,144-154` same numerics; sinks text + audio | param drift | |
| Sink span, default Suffix layout | Spark `stage1_ops/sol.py:21-37` permutes to [visual \| text+audio], index_select after RoPE, sink_start=len(visual), inverse permute | `h3/sol.rs:219-225` span (visual.len(), sinks) in permuted coords; `transformer.rs:817-819` calls `sol_attn_sunk` on the UNPERMUTED [text\|cond\|audio\|video] (`packing.rs:196-208`); no index_select in cudarc/src/h3 | algorithm drift | Verified bug: span [V, V+T+A) lands on the last T+A video rows; text/audio queries routed sparsely. Log at `pipeline.rs:923-928` claims a permute that never happens. |
| Sink span, native mode (`FASTVIDEO_H3_SOL_SINK=native`) | prefix sink = [0, video_start) = text + cond-video + audio (`sparse_attention.py:287-296`) | `h3/sol.rs:231-240` text span + audio span; cond-video rows excluded | param drift | |
| VSA 0.9 / tile 64 / gate weights | `stage1_ops/vsa.py:23-58` topk rule, prefix exempt, gate to_gate_compress nonzero; cuDNN BSA on SM121 (`:73-83`) | `vsa.rs:11-26,137-167` same selection; gate loaded when vsa_sparsity>0 (`pipeline.rs:446-447`); Wan VSA mma kernels | same | Kernel differs; no BSA warmup reference audit (`vsa.py:275-303`). |
| LoRA fuse rule (alpha 64/8, scale 1.0, fp32 diffs) | `lora.py:82-86,138-141,198-217` (rejects .set_weight); `engine.py:225-233` | `lora.rs:114-135` same multiplier; accepts hybrid .set_weight (`:182-198`); `spark.rs:23-41` silently falls back vsa-datafree -> dense-datafree | same | separate/fused LoRA modes (`lora_fusion.py:83-101`) missing. |
| W8A8 FP8 (312 linears) / MXFP8 blocks 2-46 / NVFP4 Qwen | `stage1.py:125-147`; `compute_quant.py:70-143`; `configs/default.json:6-7` | Intentionally off: `pipeline.rs:611` "W8A8 FP8 stays off (measured 16-20 dB)"; `FASTVIDEO_H3_FFN_FP8` is a different FFN-only scheme (`transformer.rs:836-905`) | missing | |
| TeaCache RTX (0.10 / retain 5 / cooldown 1, norm1 probe) | `teacache.py:44-78,125-192`; `RTX5090/model.py:1618-1670` | `h3/sol.rs:43-54,353-428`; `transformer.rs:1218-1273` (opt-in `FASTVIDEO_H3_SOL_CACHE=teacache`, no matrix cell) | same | |
| FirstBlockCache 0.08 (GB200/H100/A100/GB10) / EasyCache 0.30 | `GB200/cache_line.py:16-33`; `H100/profiles.py:60-84` | no FBCache/EasyCache for H3 (grep empty) | missing | |
| Spark bridge (upscaler, adapter, LTX refiner) | `h3_upscale.py:92-126` D(model(N(h3))) x2; `stage2.py:76-84` adapter -> latent[:, :, :16], joint AV refine, original PCM muxed | `h3/spark.rs:43-155,278-438` same geometry/normalisation; cudarc `spark.rs:2` forces upscaler attention OFF; `runpod-matrix.sh:146,321` unsets `FASTVIDEO_LTX2_WEIGHTS` => sol-h3-spark cell never refines | param drift | Stage-1 BF16 not W8A8+NVFP4; H3 video decoded in stage 1; no output tiling/NHWC/writer chunking. |
| Fusions (QKV, SwiGLU, AdaLN table) | `fusions.py:161-321`; `adaln.py:101-208` | `transformer.rs:22-23,773` fused QKV(G); `:827-828` `swiglu_value_first`; `:1109-1136` AdaLnTable cached to disk | same | Fused rmsnorm+modulate and qknorm+RoPE (`:776-794` separate ops) missing. |
| HyperFlow 8-step two-time LoRA | `HyperFlow/hyperflow_h3/schedule.py:31-35`, `embedder.py:13-25`, `sol_attn.py:58-63` | none | missing | |
| super_acceleration overlays (hybrid teacher/student, lowres sweep, handoff RPC, stage-2 server) | `stage1/*_overlay.py`, `handoff_protocol.py`, `stage2/sol_attention.py` | only TAEH3 decoder option (`pipeline.rs:201-204,644-648`) overlaps | missing | |
| Ulysses/CP + int8/FP8 comm quant, sol_bsa backend | `comm_quant.py`; `sparse_attention.py:571-620`; `sol_residual.py` | single GPU | missing | |

### LTX (14)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| Stage-1 distilled ancestral 8-step (2.5 RTX5090) | `distilled.py:62-73,170-183` EulerAncestral eta 1, seed+10000, DISTILLED_SIGMAS `constants.py:17` | `ltx2/pipeline.rs:589-637,1541-1567`; `schedule.rs:26-29,78-119,218-253` | same | |
| Stage-1 dev 30-step + 4-pass guidance (GB200) | `samplers.py:39-81` Euler; `denoisers.py:100-138` cond/uncond/STG(block 28)/modality; `guiders.py:261-272` rescale 0.7; `schedulers.py:21-57` anchor-4096 shift 2.05 | `pipeline.rs:399-406` 2-pass CFG on velocity; no perturb/rescale/modality (grep empty); `schedule.rs:145-183` uses actual token count | missing | |
| res2s sampler (2.3 HQ, stage 1 and 2) | `samplers.py:216-445` SDE eta 0.5, bongmath, legacy_mode, 0.0011 sigma injection, 2*15+1 = 31 evals; stage 2 also res2s (`ti2vid_two_stages_hq.py:315-336`) | `pipeline.rs:451-536` + `hq.rs:33-57` ODE only, 29 calls, last step snaps to x0; stage 2 stays 3-forward Euler | algorithm drift | |
| SCSP step cache (skip calls 16-28, delta 0) | `presets.py:29-33`; `step_cache.py:64-77` caches denoiser OUTPUT x0 | `ltx2/pisa.rs:37-62` same skip set; `transformer.rs:705-731` caches VELOCITY pair | param drift | x0 = x - sigma*v, so reuse at a different sigma differs. Also applied on Euler step index where 16 is unreachable at 8 steps (`pipeline.rs:301,355,607`). |
| FBCache (0.08 / warmup 1 / max 10, block-0 rel-L1) | `step_cache.py:44-49` rel_l1, `:149-159` accum/reset, `:259-277` residual reuse, `:119-126` per-pass state; `fullopt.toml:19-22` | `ltx2/fbcache.rs:11-17,101-129,151-166`; `transformer.rs:801-876,986-1018`; stage-1 only (`pipeline.rs:1610-1612`) | same | Rust applies it to the 8-step distilled loop (1 pass); Python to 30-step dev (4 passes). No teacache/easycache policies, no bypass counters. |
| Stage-2 sampler (Euler, SimpleDenoiser, stage-1 audio) | `ti2vid_two_stages.py:287-311` stage-2 audio discarded, stage-1 audio decoded; `constants.py:20` [0.909375,0.725,0.421875,0], `:25` GB200 [0.625,0.4,0] | `schedule.rs:126-140` same RTX sigmas; `pipeline.rs:1686-1711` denoise_cfg when guidance != 1; `:1713-1714` stage-2 audio KEPT | algorithm drift | Python never guides stage 2. GB200 2-forward sigmas absent. |
| PISA stage 2 (2.3, sparsity 0.9, block 64, dense L0-1) | `optimized/env.sh:70-82,102-103`; `sparse_attention.py:40-52`; kernel not in repo | `ltx2/pisa.rs:13-21`; `attention.rs:213-227` video-self only; `pisa.rs:17` FORWARDS=3 while HQ stage 2 is 7 res2s evals | param drift | |
| Sol-Attn 2.5 (taus 1/1.25/1.5, layer 0 dense, diag, no sink) | `RTX5090/attention.py:9-20,61-105`; `refiner/sol_attention.py:18-19,117-193`; `preprocess.py:143-186` | `ltx2/sol.rs:10-19,47-67`; `sol_attn.rs:118-175`; `attention.rs:221-223` | same | |
| Token pruning (feat_norm midpoint, ratio 0.5, steps 1-2) | `token_prune.py:638-653` score/keep, `:819-917` gather/scatter with prev compensation; `presets.py:55-61` | `ltx2/pisa.rs:83-121`; `transformer.rs:742-799` consumes step on first forward; second forward (uncond) runs unpruned and overwrites prev (`:791-799`) | algorithm drift | Only drifts when Rust stage-2 CFG is on; Python stage 2 has one pass. |
| NVFP4 (TE video-FFN only) | `nvfp4_ffn.py:39-58,100-155` TE NVFP4 GEMM, RHT/SR off, pad_m 16, bf16 fallback; `env.sh:85-90`; RTX5090 prequant checkpoint `gpu_infer.py:52-95` + `exact_adaln.py` | `wan/nn.rs:77-89,262-300` fake-quant EVERY eligible linear at load, reconstruct to f32, dense GEMM; `nvfp4.rs:281-319` K/V fake-quant incl. audio; `Scope::Ltx23VideoFfn` (`nvfp4.rs:229-301`) never called from ltx2/ | algorithm drift | |
| KWL fusion set (13 on / 11 off as frame-changing) | `kwl_fusions.py:22-38`; `optimized/env.sh:30-60` (FUSED_QKV explicitly OFF for exactness) | no fusion switches; Rust fused QKV stack (`nn.rs:262`) is a divergence from the validated set | missing | |
| Distilled LoRA strengths | 2.3: 0.25/0.5 (`run_ltx23_common.sh:64-65`); GB200 2.5 stage 2: 1.0 (`ti2vid_two_stages_mgpu.py:77`); refiner 0.8 (`refiner_head_cp.py:60`) | `ltx2/lora.rs:29-33` (0.25,0.5) V23 same; (0.0,0.8) V25; refine_joint uses `ltx2_5_22b_distilled` cfg => strength 0.0 (`h3/pipeline.rs:1117`, `pipeline.rs:1847`) | param drift | |
| 2.5 refiner (TAEHV, 3 Euler, video-only, source audio mux) | `refiner_head_cp.py:487-608`: mp4 -> crop 960x544 -> TAEHV encode -> upsample -> renoise 0.909375 -> 3 fwd SimpleDenoiser audio_state=None -> TAEHV decode -> mux source audio; manifest prompt | `pipeline.rs:1775-1938` refine_joint: H3 latent via upscaler+adapter, FIXED_PROMPT (`h3/spark.rs:19-21`), H3 PCM encoded and denoised jointly (`:1818-1878`), conv VAE decode, no LoRA | algorithm drift | |
| CFG-parallel / SP / 2x2 TDP / distributed VAE / FFN chunking | `ti2vid_two_stages_mgpu.py:113-187`; `memory.py:7-27` | single GPU | missing | |

### LingBot (7)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| Step-EasyCache base 0.08/4/2/2 and refiner 0.25/2/1/2 | `pipeline_lingbot_video.py:514-531` param sets, `:578-601` non-accumulating rel = mean\|x_i - x_ref\| / mean\|x_ref\| vs LAST COMPUTED input; caches batched CFG output | `lingbot/sol.rs:40-55` requested() + GAP string; `pipeline.rs:128-132` logs and stays dense | missing | Not the Wan EasyCache: no accumulator, no k estimate. |
| PISA on refiner (0.10, block 64, dense L0-3, head 2 / tail 1) | `transformer_lingbot_video.py:39-53,108-187` applied after CP4 gather per CFG segment; kernel importlib-loaded from archive (`:78-84`) | no refiner stage (`sol.rs:17-19`) | missing | |
| cuDNN SDPA varlen over cu_seqlens | `transformer_lingbot_video.py:190-222`, dispatch `:478-565` (PISA -> cudnn -> fa2) | `transformer.rs:85` dense masked SDPA | missing | |
| Scheduler | `pipeline_lingbot_video.py:108-113` FlowUniPCMultistep order 2 bh2 | `lingbot/schedule.rs:14-18` FlowMatch Euler | algorithm drift | |
| CFG | batched cond+uncond with encoded negative prompt (`--batch_cfg` `runner.py:1303`) | `pipeline.rs:152-170` two forwards, zero-tensor uncond | algorithm drift | |
| MoE routing (e_score_correction_bias, group top-k, shared experts) | `transformer_lingbot_video.py:600-634,668-672` | `transformer.rs:207-253` sigmoid -> top-k -> norm -> routed_scale; `config.rs:54-60` has no bias/group/shared fields | algorithm drift | |
| CP4 Ulysses + FSDP topology (golden 375.5 -> 144.4 s on 4 GPUs) | `runner.py:351,1433`; `evals/_golden/lingbot_opt/benchmark.json` | single GPU | missing | |

### Cosmos3 (3)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| TeaCache 1.15 / start 10 / max 3 (rel-L1, identity coeffs) | `optimized/env.sh:20` teacache_c115_s10_m3 parsed by EXTERNAL `run_cosmos3_cache_matrix.sh`; generic controller `techniques/methods/teacache.py:46-135` | `cosmos/sol.rs:54-63` relative_l1, `:88-101` `TeaCacheWindow::decide` (equivalent), `:122-131` follow(); `cosmos3/transformer.rs:292-307` block-residual reuse | param drift | Controller logic matches but `cosmos3/config.rs:79-81` `super_64b() -> None`; only exercised by Predict2 EDM path which skips flow_shift 10 (`cosmos/sol.rs:18-19`). |
| NVFP4 step-selective (gate_up/down/qkv/out, skip first 3 / last 3) | `optimized/env.sh:24-30` SGLANG_COSMOS3_FP4_LINEAR | `cosmos/sol.rs:49-51` `fp4_linear(step,n)` predicate only; no quantised linears | param drift | Constants only. |
| 4-GPU sequence parallel | `run_cosmos3_common.sh:46,131` NUM_GPUS=4 | single GPU | missing | |

### Hunyuan (2)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict | Note |
|---|---|---|---|---|
| TeaCache 0.15 / 6 / 2 on time_text_embed, whole-step cache | `step_cache_runtime.py:67-118` identity coeffs (`teacache.py:93`), wraps transformer.forward; `full.toml` | `hunyuan15/sol.rs:29-39` GAP string; `pipeline.rs:178-196` dense; time path is time_in only (`transformer.rs:236-283`) | missing | Rust ports HunyuanVideo 1.5, Sol-Engine runs 13B; controller portable via `cosmos::sol::TeaCacheWindow` with recalibration. |
| Sol-Attn tau 1.0 diag, exact text KV sink, dense text queries | `gpu_infer.py:134-185`; `sol_attn_backend.py:429-546` sink_start=video_len, text rows overwritten with SDPA | `transformer.rs:203` dense SDPA; text-sink kernel exists for Wan (`wan/sol_ops.rs:333-343`) but is not wired | missing | |

### SANA (1)

| Technique | Sol-Engine (file:line) | fastvideo-rs (file:line) | Verdict |
|---|---|---|---|
| SANA-Video 5B optimised arm | `config/sana_video/baseline.toml` FORWARD_CACHE_METHOD=none; DiT lives in external `sana_minimal_inference_zip` | `rg -i sana crates/` hits only PISA/Sol helpers; no model module | n/a |

## 3. Sol-Attn kernel: structure and the 10x

Measured sol-h3 step time vs dense and VSA (s/step, warm, TAEH3 decode).
Source: `FVID-2026-09-24-h200-warm-suite` and `FVID-2026-09-25-b200-warm-suite`.
Step 1 is dense by policy. Same kernel build `ec5a6cc0988aca26` on both GPUs;
the ratio (sol / dense = 8.6x to 13.7x) is the same on sm_90 and sm_100, so
the cause is structural, not an arch-specific miss.

| | dense 4-step (s/step) | sol step 1 | sol step 2 | sol step 3 | sol step 4 | VSA 0.9 (s/step) |
|---|---:|---:|---:|---:|---:|---:|
| H200 SXM | 39.6 | 39.6 | 543.3 | 433.0 | 341.0 | 7.0 |
| B200 | 29.1 | 28.9 | 324.4 | 261.7 | 209.4 | 5.33 |

### Structure

| Aspect | Sol-Engine (`techniques/sparse_backends/sol_attn`) | fastvideo-rs (`fastvideo-cudarc/src/wan/ops.rs`, `kernels.cu`) |
|---|---|---|
| Launches per call | 1 fused kernel (+ Triton preprocess for kc/vc/threshold) | ~12: sequential_plan (host) -> plan H2D -> vsa_tile_mean K -> sol_tile_sum V -> vsa_tile_mean Q -> cuBLAS f32 Q.Kc^T [bh,T,n] -> sol_diag_threshold -> sink flags H2D -> sol_exact_lists -> fine partials -> sol_coarse_partials -> sol_lse_merge (`ops.rs:2809-2846`) |
| Grid | one CTA per (64-query tile, b*h); sm90: 128 MMA + 32 producer threads, TMA (`sm90/kernel.py:24-27`, `mainloop.py:900-906`) | fine MMA: (n_tiles, bh) x 128 thr (`ops.rs:3231-3235`); coarse and scalar fine: one CTA PER QUERY TOKEN (T, bh) x 128 thr (`ops.rs:2757-2764`) |
| Route scores | WGMMA Q.KC^T in registers, column sums by shuffles, warp vote_ballot -> 2x32-bit mask (`mainloop.py:239-252, 540-541`) | materialised f32 score matrix [bh,T,n] (`ops.rs:2830-2834`); sol_exact_lists with block_dim (1,1,1) reduces 64 rows x n cols serially (`ops.rs:2980-2984`, `kernels.cu:2005-2028`) |
| Approximate term | reuse score tile: masked exp2, WGMMA P.VC, row_sum corrected by block length (`fwd.py:108-132`; `mainloop.py:771-836`) | per token, loop all n blocks; per block a linear scan of the exact list (`kernels.cu:2098-2104`) then a 128-lane product + 7-level `__syncthreads` tree reduction for ONE dot (`:2108-2113`); Kc/Vc re-read per token |
| Exact blocks | bfind over mask bits, TMA prefetch of next K during softmax/PV, RS-mode PV (`sm90/exact.py:124-186`) | sol_mma_attn_partials mma.sync m16n8k16 + cp.async 2-buffer, f32 (m,l,acc) epilogue (`kernels.cu:2264-2369`); scalar fallback per token per key (`:2052-2071`) |
| Softmax | single online softmax over route + exact; bf16 O + f32 LSE; kv_splits 2/4 on SM90 (`mainloop.py:1415-1432`; `fwd.py:11-108`) | two independent partial passes merged by sol_lse_merge (`kernels.cu:2152-2167`); no LSE output; no kv_splits |
| dtype | bf16 Q/K/V/O required, f32 accumulate (`interface.py:31-32`) | f32 everywhere; MMA fine converts Q/K/V to bf16 per call via 3x vsa_tile_qkv (`ops.rs:3225-3227`) |
| Exact set representation | two 32-bit words per 64-block group, bfind/popc (`selector.py:11-44`) | u32 list [bh, n, n] padded with 0xFFFFFFFF, scanned linearly (`ops.rs:2979`) |
| Host round trips per layer | none (hooks permute once per forward) | sequential_plan + 2 pageable memcpy_stod for plan (`ops.rs:737-752`) + sink flags memcpy_stod (`:2976-2978`); Wan route also host-gathers q/k/v/out (`sol_cache.rs:72-102`) |
| PISA / SLA fine stage | external sglang piecewise_attn (not in repo) | never reaches tensor cores: sol_fine_or_mma requires log2_space==1 (`ops.rs:3009`), PISA passes 0; sol_global_h_bar one thread per (e,d) looping over T (`kernels.cu:2179-2196`) |

### Ranked causes of the slowdown, from kernel code

1. `sol_coarse_partials` is O(T * n * n_exact) scalar work with a CTA per query token (`kernels.cu:2080-2130`). Sol-Engine does the approximate term for a 64-query tile as one WGMMA per block group plus one P.VC WGMMA: ~64x64 fewer scalar dots and 64x fewer Kc/Vc reads. Step times falling 543 -> 433 -> 341 s as routing changes exact counts is consistent with an n_exact-dependent scan.
2. Materialised f32 token x block score matrix plus a single-thread list build (`ops.rs:2830-2834, :2980-2984`). At T ~ 100k, n ~ 1.6k, 32 heads that is ~20 GB per layer, then one thread per q-block doing n*64 strided loads.
3. ~12 full-tensor f32 passes per call: three tile reductions, three bf16 conversions, two partial stages each writing acc[bh,T,128] f32 plus fill_device zeroing (`ops.rs:2793-2803`), then a merge reading both. Sol-Engine reads Q/K/V once via TMA and writes bf16 O once.
4. Fine stage is mma.sync + cp.async with no warp specialisation, no TMA, f32 partial epilogue (`kernels.cu:2219-2376`), and each CTA first scans a full n-length sentinel list (`:2238-2239`). At tau 1.0 Sol keeps far more blocks than VSA at 10%.
5. Per-layer synchronous host round trips (plan, sink flags), and on Wan a host gather of q/k/v/out for Morton.
6. n-entry exact list instead of a 2-word bitmask; scanned linearly inside the coarse loop.
7. PISA and SLA never use tensor cores (`ops.rs:3009, :2917-2919`).

Not implicated: selection rule, diag threshold, coarse-term math (all match
`preprocess.py` / `selector.py` / `fwd.py`), plan rebuild cost, Morton (Wan
only). Verified by reading `ops.rs:2757-2764, 2980-2984` and
`kernels.cu:2094-2116` directly.

## 4. Framework and evaluation layer

`techniques/`, `config/`, `scripts/`, `evals/` in Sol-Engine vs env flags,
`runpod-matrix.sh`, gpucheck, and `bench.json` in fastvideo-rs.

| Layer | Sol-Engine | fastvideo-rs | Verdict |
|---|---|---|---|
| Technique composition | `techniques/technique.py`: Phase {WRAP_ATTENTION, PRE_BLOCKS, IN_BLOCKS, POST_BLOCKS, ON_STEP}, Seam with EXCLUSIVE_SEAMS {ATTENTION_BACKEND, TOKEN_SET, STEP_OUTPUT, FFN_PRECISION}, Capability; `compose.py` checks capabilities and write/read seam conflicts and emits a Plan | Per-model env flags (`FASTVIDEO_WAN_SOL_ATTN`, `FASTVIDEO_WAN_PISA`, `FASTVIDEO_H3_SOL_CACHE`, ...). Exclusivity is whichever if/else arm wins in `wan/transformer.rs:239-264`; nothing declares that Sol-Attn and PISA write the same seam | missing |
| Step scheduling DSL | `techniques/schedule.py`: at_steps, before, by_stage, parse_steps ('0-3,47-49'), applied by the runtime per step index and stage | Hard-coded ranges per technique (`wan/sol.rs:55-69`, `ltx2/pisa.rs:37-62`, `h3/sol.rs:43-48`); each controller counts its own steps, which is how the dense_steps and TaylorSeer clock drifts arise | param drift |
| Config contract | `config/schema.md`: id, kind {baseline, env_only, patch, methodology}, purpose {control, delivery, frontier, evidence, blocker_probe, unsafe_probe}, [official_config], [env] UPPERCASE, [artifacts], [slurm], [patch].off_identity_required, [requirements].capabilities, [verification]. `scripts/run.py --print/--set` resolves to launch.sh + manifest.resolved.toml + metadata.json | `runpod-matrix.sh` cells (bash env + CLI flags) and `h3/config.rs` recipe structs. gpucheck `report.rs:155-222` dumps `FASTVIDEO_*` env into the report; no resolved manifest, no kind/purpose, no capability requirements | missing |
| Off-identity gate | `collect_run.py:813-843` build_off_identity: technique OFF run must be byte-identical to baseline frames (max_abs_diff_uint8 == 0) before ON is scored; determine_status `:513-542` | None. decision-log verifies "f32 path is bit-identical" by assertion, not by an artifact diff | missing |
| Pixel / temporal metrics | `collect_run.py:845-925` build_pixel_metrics: mse/mae/psnr vs baseline, sharpness_ratio, patch_boundary_ratio (multi-size), temporal_delta_error, temporal_jitter_ratio | gpucheck `quality.rs`: reference-free gates (frame stats, black/NaN, motion) plus PSNR vs f32 in host unit tests (e.g. bf16 act 57.46 dB). No paired baseline-vs-optimised frame metrics on GPU runs | param drift |
| Visual artifact gate | `tools/vision/lpips_judge.py` (AlexNet LPIPS) + `evals/rubrics/gemini_visual_artifact_gate.md` (12 artifact categories, JSON verdict promote\|tune\|reject\|rerun); `evals/tiers.toml` ranks by Gemini severity then LPIPS | None | missing |
| Speedup tiers | `evals/tiers.toml`: low >= 1.5x, medium >= 2.0x, high >= 3.0x over official baseline, quality gate required | Speedups quoted ad hoc in `decision-log.md` against Phase-0 or published numbers (e.g. 26.5 s vs FastH3 16.2 s = 1.64x); no tier file, no gate | missing |
| Benchmark artifact | `benchmark.json` schema 2 (golden wan5b_opt): baseline_class, timing_scope, 5-prompt median samples, warmup_samples, kernel_runtime.activation.stack, invariant_cache_stats hits/misses, cache_method per-generation compute/reuse + per-step trace, totals (700 calls / 380 compute / 320 reuse), memory | fastvideo-cli `bench.json`: model, backend, device, dtype, h/w/frames/steps, load_ms, generate_ms (`main.rs:444-471`); gpucheck `perf.rs:325-505` stage timings + device_stats; runpod-matrix `summary.json` pass/fail/cost. No reuse counters, no per-prompt median, no warmup separation | missing |
| Authenticity guards | `step_cache.py` comment documents the skip='16-28' const-wrap bug that skipped all 35 steps and produced a fake 6.4x; `benchmark.json` exposes compute/reuse counts so that class of bug is visible; SOL_ATTN_STRICT, refiner asserts 3/141 calls | `FASTVIDEO_STRICT_DEVICE` (`stats.rs:105-142`) refuses CPU fallback under a live device, and host_fallback counters land in device stats. Nothing counts cache reuse, route calls, or dense-guard hits, so a controller that skips every step would still report a clean run | missing |
| Unit test coverage of kernels | `sparse_backends/tests/test_mps_backend.py`: threshold vs independent torch stats for diag and exact, full-sink == dense SDPA in bf16 at 2e-2, mask vs einsum reference at tau=100, mixed exact/centroid output reference, q-block 32 vs 64 | Host oracle tests: sink range, pooling, diag formula, tau monotonicity, full-sink == dense in f32 at 40 tokens, device-shaped host twin == oracle (`sol_attn.rs:502-598`, `sol_ops.rs:697-766`). No test runs the CUDA kernels; all tests use <= 40 tokens (n <= 1 block) | param drift |

## 5. Priorities

| # | Change | Why | Where |
|---|---|---|---|
| 1 | Fix the H3 sink span (default layout) | Every sol-h3 number measured so far routed text/audio queries sparsely and sank video rows. Either permute q/k/v to [visual \| text+audio] before `sol_attn_sunk` or emit the span in native coordinates (the native mode already does, minus cond-video). | `h3/sol.rs:219-225`, `h3/transformer.rs:817-819` |
| 2 | Rewrite the Sol coarse pass and route as tile-level tensor-core work | Causes 1-3 account for the bulk of the 10x. Reuse the VSA plan-once structure (`vsa.rs:320-345`), compute route scores per 64-query tile with a bitmask, fold the approximate term into the fine kernel's online softmax, keep bf16 I/O. | `wan/ops.rs:2757-2846`, `kernels.cu:2080-2130, 2005-2028` |
| 3 | Make the step clock shared and forward-based where Sol-Engine's is | dense_steps (Wan Sol), TaylorSeer warmup/interval, and LTX SCSP call indexing all count differently from the manifests they cite. One pipeline-owned (step, forward, branch) clock passed to controllers removes the class. | `wan/sol.rs:11-16`, `wan/transformer.rs:1389`, `ltx2/pipeline.rs:301,355,607` |
| 4 | Persist reuse / route counters and an off-identity check | Sol-Engine found its fake 6.4x through compute/reuse counts in `benchmark.json`. Rust has host_fallback counters but nothing for cache hits, dense-guard calls, or sol/dense call counts, and no byte-identical OFF gate. | fastvideo-cli `main.rs:444-471`, gpucheck `perf.rs` / `report.rs`, `wan/stats.rs` |
| 5 | Route sol-h3 T2V/I2V through the Sol-H3 policy, not Spark's | tau 1.0 / dense_steps 1 / dense_layers 2 / prefix sink is the published one-GPU Sol-H3 contract; the Ref2VA ladder is explicitly rejected upstream for T2V. | `h3/sol.rs:87-97` |
| 6 | Un-batch CFG when a per-branch cache controller is armed, or make controllers row-aware | TeaCache and TaylorSeer semantics depend on independent branch state; the batched path resets both. | `wan/pipeline.rs:1059-1072`, `wan/transformer.rs:1157-1160` |
| 7 | Add a technique-oriented GPU smoke tier | The H200/B200 matrices run model cells only; EasyCache/TeaCache/TaylorSeer/A14B/PISA/NVFP4/BF16_ACT have never executed on a GPU. One short Wan or LTX cell per env flag with counters persisted would have caught the sink bug and the 10x before a $9 suite did. | `scripts/gpu/runpod-matrix.sh` |
