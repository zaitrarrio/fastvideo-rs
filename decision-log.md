# Decision log

Project code: FVID

### FVID · 2026-09-24 · FVID-2026-09-24-host-loops-off-device
- Trigger: Cosmos RoPE, LingBot MoE, and H3 AdaLN still bounced every attention / token / block through pageable H2D; FVID-2026-09-18 noted 911 H2D / 3.4 GiB on a 3-step Wan clip and asked whether uploads repeat per step.
- Decision: **device kernels + one upload, then slice.** Cosmos self-attn RoPE is `rope_real` (real-interleaved twin of `rope_half`) with host `apply_rope_real` behind `host_fallback`. LingBot MoE is `softmax_last` / `sigmoid_f` + `topk_last`, then `index_select_rows` + one expert GEMM + `scatter_add_rows` (no per-token `[1,d]` GEMM). H3 `BlockMods::upload` puts the whole AdaLN ladder + keyframe table on device once and `narrow`s `[1,6,hidden]` per segment.
- Reason: the host loops were correctness-shaped leftovers; the math already existed (`softmax_last`, cuBLAS, `narrow`). Keeping the CPU twins means unit tests still run without a GPU.
- **Wan 911 H2D finding (do not edit `wan/pipeline.rs`):** RoPE (`rotary_for`) and VSA plans are already pinned/cached. The remaining per-step uploads are (1) `dit_cfg` / `dit_cfg_easy` rebuilding `CudaTensor::from_vec(vec![t], [1])` every CFG call, (2) `nn::sinusoidal_timesteps` host-building `[B, freq_dim]` every `forward_ctx` so the first Linear re-uploads it, (3) causal `forward_ctx` rebuilding and `to_device()`ing a fresh `[1,1,S,S]` mask when `cfg.causal`. `CudaTensor::dev()` also re-uploads any still-host tensor on every use (`Owned` temporary), so an unpinned table inside a 30–40 layer loop multiplies. Average 3.4 GiB / 911 ≈ 3.9 MiB/transfer matches a mid-size table (sinusoid is tiny; a causal mask or a repeated activation upload is the plausible bulk).
- Reversibility: cheap — device path is opt-in on a live CUDA context; host fallbacks stay for CPU tests.
- Executed by: WS-D
- ADR: none
- Verification: `cargo test -p fastvideo-cudarc --lib cosmos lingbot`; H3 `block_mods_narrows_the_uploaded_table`. No GPU rented.

### FVID · 2026-09-24 · FVID-2026-09-24-phase0-strict-host-hunyuan-gen
- Trigger: review gaps — Sol/PISA/SLA/NVFP4 host paths were outside `host_fallback`; `fv-gpucheck hunyuan gen` missing from the published image; Spark VSA-DataFree LoRA not on the volume fetch list
- Options: leave host algorithms silent until WS-A; gate only under `FASTVIDEO_STRICT_DEVICE`; always refuse on a live device
- Decision: **`stats::host_algorithm`** — same refuse-on-device as `host_fallback`, plus `FASTVIDEO_STRICT_DEVICE=1` when a CUDA context is live. Wired at `sol_attn`, `pisa_attn`, `sla`, `nvfp4::dequant_beforehand`. **`fv-gpucheck hunyuan {info,gen}`** reports load/text/denoise/step/decode/write + peak MiB. Fetch row `FastH3-4-step-Preview-v1-VSA-DataFree`.
- Reason: a GPU sweep cannot log “sol-attn kernel” while running scalar CPU math; Hunyuan cells were skipping
- Reversibility: cheap — host oracles still run without a device; WS-A replaces the host call sites
- Executed by: Executor
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib hunyuan15`; `cargo test -p fastvideo-cudarc --lib` host_fallback / sol_attn / pisa_attn / nvfp4. GPU refuse path untested here (no nvcc).

### FVID · 2026-09-22 · FVID-2026-09-22-ltx25-diffvae
- Trigger: after distilled two-stage green — ship opt-in **DiffVAE** video decode on the validated 2.5 stack (conv VAE stays default; audio unchanged)
- Options: DiffVAE 1-step x0 untiled @ 768×512; defer tiling / NATTEN / multi-step stage-5 / two-stage+DiffVAE
- Decision: **`LTX2VideoDiffusionDecoderModel`** as opt-in path: denorm latents → NA stages 1–4 + PixelShuffle → stage-5 single x0 @ `t=1.0`; gather+SDPA NA (inward window); drop DiT before decode. `Ltx2Request::diff_vae` / `--diff-vae` / `FV_LTX2_DIFF_VAE=1`; fetch `diffusion_decoder/*` on 2.5 weight pull.
- Reason: peer quality decode path per Diffusers 2.5 pack; first green scoped to single-stage distilled without tiling
- Reversibility: cheap — flag off keeps conv VAE decode
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-22). `cargo test -p fastvideo-cudarc ltx2::diffusion_decoder`; `fv-gpucheck ltx2 gen --model-version 2.5 --diff-vae`. RTX PRO 6000 WS `51982652` (~$0.779 / 33 min): 8 ancestral ~18.1 s, **DiffVAE decode ~884.4 s** (host NA), audio decode ~0.4 s; **121** frames, wav 48 kHz / 240480 samples, peak **70469 MiB**, mp4 `artifacts/clips/20260922T001443Z-ltx2-gen/ltx25-diffvae.mp4`. Build `77cf645514609c38`.

### FVID · 2026-09-21 · FVID-2026-09-21-ltx25-two-stage
- Trigger: after stage-1 + BWE green — pick next 2.5 extension
- Options: distilled two-stage (half-res → spatial upsampler → 3-step stage-2); DiffVAE/diffusion decoder; duration head; prompt enhancer; 1536×1024 scale-up
- Decision: **distilled two-stage** on the validated stage-1 path. Same DiT; new `latent_upsampler/`; no stage-2 LoRA on distilled DiT. Validate at final **768×512** (stage-1 384×256) so stage-2 token count matches single-stage.
- Reason: official recipe; VRAM stays in existing `ltx2-gen` tier; other extras still out of scope
- Reversibility: cheap — `Ltx2Request::two_stage` / `--two-stage` / `FV_LTX2_TWO_STAGE=1`
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-21). `cargo test` latent_upsampler + request validation; `fv-gpucheck ltx2 gen --model-version 2.5 --two-stage`. RTX PRO 6000 WS `51970730` (~$0.560 / 25 min): stage-1 8 ancestral ~7.6 s, upsample 0.19 s → grid `[16,16,24]`, stage-2 3 ancestral ~11.6 s (~3.56 s/step), decode video ~3.9 s; **11** steps finite, **121** frames, wav 48 kHz / 240480 samples, peak **82117 MiB**, mp4 `artifacts/clips/20260921T220546Z-ltx2-gen/ltx25-two-stage.mp4`.

### FVID · 2026-09-21 · FVID-2026-09-21-ltx25-stage1
- Trigger: "let's implement LTX 2.5" after H3 matrix; user locked scope to distilled stage-1 + conv VAE, no extras unless they help perf
- Options: full 2.5 stack (DiffVAE/diffusion decoder, duration head, enhancer, two-stage); stage-1 only; Comfy-only vs Diffusers pack
- Decision: **distilled stage-1 T2AV** on the existing `ltx2` modules with `Ltx2ModelVersion::V25`. Weights: `Lightricks/LTX-2.5-Diffusers` (oracle) / Comfy split pack. Gemma4 unified TE, 22B distilled DiT (`ff_bias=false`, gated attn, cross_attn_mod, `use_prompt_embeddings=false`), conv video VAE, audio VAE + **BWE vocoder** (required by checkpoint). Ancestral Euler for `model_version≥2.5`. Keep 2.0 path. Spec: `docs/ports/ltx25.md`.
- Reason: matches the 2.0 milestone shape; official distilled stage-1 still needs gated/cross-attn-mod DiT + Gemma4 + BWE — those are checkpoint contracts, not optional product extras. Diffusion decoder / duration / stage-2 deferred.
- Reversibility: cheap — version enum + opt-in weight roots
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-21). Spec `docs/ports/ltx25.md`. `cargo test -p fastvideo-cudarc ltx2 --lib` / models ltx2 ok; `fv-gpucheck ltx2 gen --model-version 2.5`. Shipped: Gemma4 TE (`DecoderConfig::gemma4_12b_text`), gated DiT + `ff_bias=false` + cross_attn_mod AdaLN (9-row), per-modality connectors, ancestral Euler, conv VAE typed upsamplers (spatial d2s), vocoder **SnakeBeta AMP** + full **BWE** (Hann-sinc ×3 skip + MelSTFT + `bwe_generator` residual → 48 kHz; MelSTFT mel proj on host — CUDA `matmul` refuses `[M,F]@[N,F,T]`). **Remote (SnakeBeta+nearest):** Texas `51956651` (`20260921T193442Z-ltx2-gen`). **Remote (full BWE):** Switzerland `51967618` (`20260921T213121Z-ltx2-gen`), gen ~50 s after weights: BWE path logged, 8 ancestral ~2 s, **121** frames, wav 48 kHz / 240480 samples, mp4 `artifacts/clips/20260921T213121Z-ltx2-gen/ltx25-bwe.mp4`.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-matrix-followup
- Trigger: after encoder matrix pass — run remaining weight/SKU gaps (Synthetic Step1900, Preview on 80 GB, longer clip)
- Options: full matrix re-rent; serialize; skip Synthetic; raise res instead of duration
- Decision: **followup wave** (`FV_MATRIX_WAVE=followup`): skip V2 gens; H200 = DataFree baseline + Synthetic-Step1900 + DataFree **15s**; A100/H100 = Preview VSA stock + recovered-8b (`FV_MATRIX_PREVIEW_80` on). Same Vast parallel / reuse-box / TEST_ID / GPU-empty strategy. No LoRA-on-base, no v0.2, no dense+8b.
- Reason: quality A/B for Synthetic, 80 GB fit check, and a blog-scale duration sample without re-paying V2
- Reversibility: cheap — wave is env-gated; cells skip on clip reuse
- Executed by: Executor
- ADR: none
- Verification: **pass** (build `22bfa760236939c7`, relaunch after credit top-up). Drivers: `artifacts/gpucheck/matrix-logs/{a100,h100,h200}.followup.driver.log`. Shared run dir `…/20260921T013704Z-h3-matrix`. Boxes destroyed: A100 `51837932`, H100 `51837933`, H200 `51837935`. First attempt aborted (credit→$0).
  - **Preview-80 A100**: stock `021329Z` ~14.9 s/step; recovered-8b `021709Z`; PSNR stock-vs-8b **16.86 dB**. Run ~$0.75 / 44 min.
  - **Preview-80 H100**: stock `020132Z` ~10.6 s/step; recovered-8b `020416Z`; PSNR stock-vs-8b **16.37 dB**. Run ~$1.47 / 31 min.
  - **H200 DataFree** `021152Z`: warm ~7.7–8.3 s/step, peak **59780 MiB**.
  - **H200 Synthetic-Step1900** `022439Z`: warm ~7.7–10.0 s/step, peak **61028 MiB**. PSNR datafree-vs-synth1900 **15.38 dB**.
  - **H200 DataFree 15s** `022639Z`: warm **~33.2 s/step** ×4, denoise **133.0 s**, peak **91008 MiB**, 362 frames. Run ~$3.70 / 55 min.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-encoder-matrix
- Trigger: checkpoint × encoder × residency matrix (V2 / Preview VSA / Preview Dense × stock streamed / SearchingMan 8B / cache-hit / resident-fp8|bf16 on H200)
- Options: RunPod network volume; serialize SKUs; keep Python hub fetch; custom `fv-gpucheck fetch`
- Decision: **Vast, three parallel SKUs** (A100-80, H100-80 not NVL, H200). Slim runtime: **hf-fm**, no Python. Contracts `fasth3_4step_vsa` / `fasth3_4step_dense`; `--text-encoder recovered-8b`. Tier `h3-matrix`: reuse box, GPU-empty/restart between cells, nohup, `TEST_ID` on every log line, `FV_IMAGE_PULL_TIMEOUT=120`. Preview Hub ids: `…-4-step-Preview-v1-VSA-DataFree` / `…-Dense-DataFree`. Flag defaults unchanged.
- Reason: parallel CDN pulls beat serial volume copy; TAEH3 keeps decode off the critical path; PSNR (same box) gates 8B drift, not a host oracle.
- Reversibility: cheap — tier is opt-in; encoder choice is a CLI flag
- Executed by: Executor
- ADR: none
- Verification: **pass** (build `22bfa760236939c7`). Driver logs: `artifacts/gpucheck/matrix-logs/{a100,h100,h200}.driver.log`. Runs: `…/20260920T192357Z-h3-matrix` (A100), `…/192358Z` (H100+H200 V2), `…/194151Z` (H200 Preview). All three Vast boxes destroyed.
  - **V2 stock streamed** `--warm`: H200 ~9.0 s/step peak ~60 GiB; A100 ~16.2 s/step; H100 ~12.0 s/step.
  - **V2 recovered-8b** resident-bf16 PASS (same step times): H200 `20260920T192610Z-…` 198 s peak **68768 MiB**; A100 `20260920T192617Z-…` 393 s peak **68414 MiB**; H100 `20260920T192706Z-…` 277 s peak **69978 MiB**.
  - **V2 cache-hit** PASS on all three. H200 also **resident-fp8** `20260920T193237Z-…` peak **82816 MiB** + **resident-bf16** `20260920T193552Z-…`.
  - PSNR stock-vs-8b: A100 **15.23 dB**; H200 **15.03 dB**; H100 **14.95 dB**.
  - **Preview 4step-vsa** (H200): stock `20260920T195013Z-…` warm denoise **30.9 s** (~7.7 s×4) peak 60 GiB; 8b `20260920T195159Z-…` same ~7.7 s/step peak **69408 MiB**; cache-hit `20260920T195355Z-…`; PSNR stock-vs-8b **15.60 dB**.
  - **Preview 4step-dense** stock `20260920T195557Z-…` ~**42 s/step** ×4, warm denoise **168.8 s**, peak **57092 MiB** (~5.5× VSA).
  - Preview first attempt skipped: async `fetch` checked `.complete` before `wait-weights` — fixed in `validate.sh`.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-mlx-affine
- Trigger: "implement our own kernel and also port the MLX work"
- Options: 4-step Preview / 90% VSA A/B; community GGUF/nunchaku; process-wide `FASTVIDEO_FP8`; first-party affine W8A16 + official MLX recipe
- Decision: **first-party CUDA kernel** for FastVideo's MLX affine group-64 INT8/INT6/INT4 (weight-only, activations BF16/F32) and load `mlx_h3_dit.safetensors` / quantize-on-load from the official FastH3 DiT. Fused dequant-in-tile GEMM — do not materialize a full bf16 weight. Not nunchaku. FFN E4M3 stays off (`FVID-2026-09-20-h3-ffn-fp8-ab`).
- Reason: there is no official CUDA FP8/NVFP4 H3 DiT; MLX INT8 is the published weight-only grid (`mode=affine`, group 64, `w = scale * q + bias`). A whole-weight dequant would spend the VRAM we are trying to save.
- Reversibility: cheap — unset `FASTVIDEO_H3_AFFINE` / `--h3-affine`
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 147132 `20260920T151313Z-h3-gen` (`51767696`): bf16 warm denoise **125.6 s** / 15.0–16.0 s/step, peak 82482 MiB. INT8 affine fused GEMM (`--h3-affine int8`, quantize-on-load): step 1 **589.3 s**, step 2 **590.3 s** (~38×). Sole GPU process. Killed mid-step-3 — no PSNR, no warm affine. **Flag stays off.** Kernel is correctness-first (16×16 tiles, no tensor cores); do not enable until a fast GEMM exists.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-ffn-fp8-ab
- Trigger: "proceed" after ranking remaining H3 levers: FFN GEMMs 33 s, MMA 28 s, prefix 11 s
- Options: H3-only FP8 on `ff_in`/`ff_out`; fuse Q/K RMSNorm+RoPE (6.7 s); process-wide `FASTVIDEO_FP8`; prefix SDPA
- Decision: **H3-only E4M3 GEMM on the two FFN linears** (`FASTVIDEO_H3_FFN_FP8` / `--h3-ffn-fp8`). Same-box bf16 vs FP8 A/B, same prompt/seed. Do not set process-wide `FASTVIDEO_FP8`. Pad the token axis to a multiple of 16 (5s H3 is 37756).
- Reason: the cheap fuses are done; 33 s is the remaining FFN pile. Wan process-wide FP8 was −4.8% / 16.9 dB (`FVID-2026-09-18-fp8-linears-measured`) — this clip decides whether H3's larger GEMMs beat activation-quant overhead, and whether the quality is usable.
- Reversibility: cheap — unset the flag
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 132178 `20260920T114626Z-h3-gen`: warm denoise **122.5 → 116.5 s (−4.9%)**. FFN 34.5 → 28.1 s — `h3_ffn_in` 21.6 → 16.7, `h3_ffn_out` 11.1 → 9.6, act 1.8 unchanged. Load 82 → 114 s. Peak 82.5 → 71.0 GiB. Clip PSNR vs bf16 **20.3 dB** (Y 18.6). Same shape as Wan: a few percent, unusable frames. **Flag stays off.**

### FVID · 2026-09-20 · FVID-2026-09-20-fused-swiglu-next
- Trigger: "commit, push, then proceed to next steps" after unchunked FFN measured as a wash
- Options: fused value-first silu×mul kernel; process-wide `FASTVIDEO_FP8` on FFN; prefix SDPA; more MMA load work
- Decision: **fuse `h3_ffn_act`**. Last-dim `narrow` copies both `[S, ffn]` halves before a separate silu and mul. One `swiglu_value_first` pass. Then a profiled H3-gen+TAEH3.
- Reason: FFN is still 38.1 s; act is 7.0 s / 18% of it and the only new slice. `FASTVIDEO_FP8` stays off (`FVID-2026-09-18-fp8-linears-measured`).
- Reversibility: cheap — FeedForward can go back to `narrow` + `silu` + `mul`
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Max-Q machine 147132 `20260920T104313Z-h3-gen`: warm denoise **122.5 s** vs 125.9. FFN 34.6 s — `h3_ffn_in` 21.7 s, `h3_ffn_out` 11.1 s, **`h3_ffn_act` 1.83 s** (was 7.0 s). Peak 82.6 GiB. Act is the cut; GEMMs unchanged. Wall ~3 s on a different Max-Q, so treat E2E as noise-adjacent.

### FVID · 2026-09-20 · FVID-2026-09-20-ffn-unchunk-next
- Trigger: "commit, push and continue" after fused QKVG measured as a wash
- Options: process-wide `FASTVIDEO_FP8` on H3 FFN; keep FFN in bf16 through SwiGLU; unchunk the 8192-row FFN GEMM on 5s clips; fused silu×mul kernel
- Decision: **do not turn on `FASTVIDEO_FP8`.** Unchunk FFN when `[S, 2*ffn]` f32 fits in 6 GiB (5s = 4.0 GiB; 109k rows still chunks). Split `h3_ffn_in` / `h3_ffn_act` / `h3_ffn_out` so the next profile names the GEMMs. Then a profiled H3-gen+TAEH3.
- Reason: FFN is ~38 s / 31% and matches BF16 FLOPs; Wan FP8 linears were −4.8% with 16.9 dB frames (`FVID-2026-09-18-fp8-linears-measured`). Five M=8192 GEMMs are the cheap thing to stop doing
- Reversibility: cheap — `FFN_ROW_CHUNK` remains the fallback
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 103044 `20260920T102145Z-h3-gen`: warm denoise **125.9 s** vs 123.5–125.5. FFN still 38.1 s — `h3_ffn_in` 20.6 s, `h3_ffn_out` 10.5 s, **`h3_ffn_act` 7.0 s**. Peak 82 GiB unchanged. **Wash** on wall time; act is the only new slice worth a kernel.

### FVID · 2026-09-20 · FVID-2026-09-20-fused-qkvg-next
- Trigger: "commit and push then continue with next strategy" after TMA measured as noise on 5s H3
- Options: fused QKVG GEMM; FFN / SM12x FP8; prefix SDPA; more MMA load work
- Decision: **fused QKVG first** (one `Linear::load_fused` of `to_q`/`to_k`/`to_v`/`to_gate_compress`, same per-head RMSNorm+RoPE). Then a profiled H3-gen+TAEH3.
- Reason: QKVG+out is 36 s / 29% of denoise and currently rereads the 520 MB activation four times; FFN is the same size pile but a separate slice
- Reversibility: cheap — separate linears are a load-path revert; `FASTVIDEO_VSA_KERNEL` is untouched
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Max-Q 6000 `20260920T100022Z-h3-gen`: warm denoise **125.5 s** vs TMA-only 123.5 s. Attn still 36.5 s (fused GEMM 21.3 s, Q/K prep 6.7 s, out 6.6 s). Peak 82 GiB vs 78. **Wash.** FFN 38.6 s is the remaining pile.

### FVID · 2026-09-20 · FVID-2026-09-20-tma-kernels-first
- Trigger: "proceed" after ranking remaining H3 levers (TMA measure, fused QKVG, FP8 FFN, tiled QKV, prefix SDPA)
- Options: kernels A/B of TMA vs `mma` on sm90+; skip to fused QKVG/FFN; profiled H3-gen first
- Decision: **kernels A/B first** (`compute_cap>=900`, T1). Then a profiled H3-gen + TAEH3 only if TMA matches the host reference.
- Reason: TMA is coded and unmeasured; FFN/QKVG work is wasted if `vsa_h3_3_mma` is still load-bound
- Reversibility: cheap — `FASTVIDEO_VSA_KERNEL=mma` is the Ampere path
- Executed by: Executor
- ADR: none
- Verification: **pass** on 5060 Ti sm_120 after `shared::cta` + `__grid_constant__`. `vsa_tma_*` matches `vsa_mma_*` (rel_l2 0.00282). Profiled H3-gen+TAEH3 on Max-Q 6000: warm denoise **123.5 s** (Ampere was 124.7 s — TMA is noise). `vsa_mma_tile` 2.5 s / 6% of VSA; fine MMA 26.9 s; QKVG+out 36.0 s; FFN 37.3 s. Next lever is fused QKVG / FFN, not more MMA loads.

### FVID · 2026-09-19 · FVID-2026-09-19-sm120-is-not-umma
- Trigger: "continue" after the TAEH3 A/B, queued as Blackwell UMMA on `vsa_h3_3_mma`
- Options: port FastVideo's sm100a `tcgen05` kernel; write UMMA from scratch; TMA + existing `mma.sync` on SM12x; FP8 fine stage
- Decision: **do not write UMMA.** RTX PRO 6000 is SM12x: no TMEM, no `tcgen05`. FastVideo's sm100a body is `#if __CUDA_ARCH__ == 1000 && SM100_ALL` and ptxas rejects it for `sm_120`. Peak BF16 here is still `mma.sync`. Skip H3 prefix query tiles in the MMA grid (SDPA overwrites them). Next kernel work is TMA 128B-swizzled K/V loads around the current MMA, not a new ISA.
- Reason: Colfax / CUTLASS / Triton all document consumer Blackwell as MMAv2 + TMA; the 5s clip is already ~130 TFLOPS of BF16 MMA, so a wrong-ISA rewrite cannot land
- Reversibility: cheap — prefix skip is a grid offset; TMA stays unstarted
- Executed by: Executor
- ADR: none
- Verification: host VSA tests; prefix skip is 11/671 query tiles on the 5s layout (~1.6% of the MMA grid). TMA 128B-swizzle kernel (`vsa_mma_attn_tma`) added for sm90+; Ampere `cp.async` remains `FASTVIDEO_VSA_KERNEL=mma`. GPU parity pending (`vsa_tma_*` in the kernels tier).

### FVID · 2026-09-19 · FVID-2026-09-19-h3-profile-next
- Trigger: "proceed" after the H3 `--profile` gen on an RTX PRO 6000 WS (`20260919T223558Z-h3-gen`). Block: attn 62%, FFN 31%. VSA: MMA 78%, prefix_dense 17%. QKVG four-way even.
- Options: split-softmax / persistent prefix; fused QKVG; FFN fusion; Blackwell UMMA on `vsa_h3_3_mma`; TAEH3 A/B (already coded)
- Decision: **TAEH3 warm A/B first** (no `--profile`, same prompt/seed as the official 22.9 s decode). Then MMA for sm_120. Park split-softmax. Bootstrap fails if cuBLAS lacks `cublasGetEmulationSpecialValuesSupport` instead of aborting in `dlsym`.
- Reason: decode is 19% of E2E and the flag already exists; prefix is 5% of denoise; MMA is the VSA term we own
- Reversibility: cheap — TAEH3 is opt-in; the symbol check only refuses a known-bad `:latest`
- Executed by: Executor
- ADR: none
- Verification: same Max-Q, no `--profile`. Official (`20260919T225958Z-h3-gen`): warm decode **28.97 s**, denoise 124.7 s, E2E 154.2 s. TAEH3 (`20260919T231221Z-h3-gen`): warm decode **0.98 s** (~30×), denoise 125.0 s, E2E 126.6 s. Clip `artifacts/clips/20260919T231221Z-h3-gen/h3-taeh3.mp4`.

### FVID · 2026-09-19 · FVID-2026-09-19-remote-job-logs
- Trigger: "make a note to always show logs for remote processes including Vast and Runpod jobs" after an H3 profile rental where logs only appeared when asked
- Options: wait until asked; stream driver logs into chat for the life of the job
- Decision: **always paste remote-job logs in chat** (Vast, Runpod, rented build boxes, `validate.sh` / `build-remote`). Stage lines and step timings; skip per-block spam
- Reason: billed GPU time is invisible if the only output is a terminal file
- Reversibility: cheap — drop `.cursor/rules/remote-job-logs.mdc`
- Executed by: Executor
- ADR: none
- Verification: rule file present; H3 profile watcher still posting step lines

### FVID · 2026-09-19 · FVID-2026-09-19-taeh3-opt-in
- Trigger: "yes, let's implement taeh3" after the TensorRT-vs-TAEH3 recommendation for H3 decode (22.9 s / 19% of a warm clip)
- Options: TAEH3 opt-in (quality trade, same family as Wan TAEHV); TensorRT on the official ViT (~1.5× decode, new runtime); leave the official decoder
- Decision: **port TAEH3** (`taeh3.safetensors`) behind `--taeh3-weights` / `FASTVIDEO_TAEH3_WEIGHTS` / `FV_TAEH3=1`. Official ViT stays the default. TensorRT not started.
- Reason: decode is the only remaining non-DiT term worth cutting, and TAEH3 reuses the TAEHV port without adding TensorRT
- Reversibility: cheap — unset the flag and the official decoder loads
- Executed by: Executor
- ADR: none
- Verification: host tests green (`cargo test -p fastvideo-cudarc --lib taehv`: 13 passed). GPU A/B on Max-Q: TAEH3 warm decode **0.98 s** vs official **28.97 s**; denoise unchanged (~125 s); clip not flat (std 0.28).

### FVID · 2026-09-19 · FVID-2026-09-19-text-plan-measured
- Trigger: "lets run" — the text-encoding plan (oracle cache, conditioning cache + slim checkpoints, resident encoders, FP8 rows, prefetch) had only been checked on the host. All on RTX PRO 6000 96GB (user's choice; the one A100 80GB offer never booted, and 80 GB cannot hold H3 + a resident encoder or a float32 LTX-2 reference).
- Kernels tier (RTX 3060, <1 cent): FP8 row codes and dequantized weights equal the host exactly; the scales were one ulp off because the device compiler turns `a / 448` into a reciprocal multiply — both sides now spell `a * (1/448)` and are exact. Prefetched encode == plain streamed encode, bit for bit, for stored-bf16 and stored-f32 checkpoints.
- Text conditioning cost, same prompt and seed per model:

  | route | H3 (Qwen3-VL-32B tap 50) | LTX-2 (Gemma-3-12B + connectors) |
  | --- | ---: | ---: |
  | streamed + prefetch, cold disk | 13.3 s (compute starved 11.6 s) | 15.2 s from the f32 checkpoint (starved 7.9 s) |
  | streamed + prefetch, slim bf16 checkpoint | n/a (already bf16, unread shards never fetched) | 10.7 s (starved 4.7 s) |
  | streamed + prefetch, warm page cache | 3.9 s (starved 3.0 s) | — |
  | resident, new prompt | 0.33-0.41 s (FP8 rows, 22.7 GiB, 21-40 s one-time load, peak 83.4 GB) | 2.7 s (bf16, 20.0 GiB, 11.2 s load, peak 70.3 GB) |
  | conditioning cache hit | 0.19-0.22 s | 2.4 s |

  Reading: `compute starved` ~= `host read+convert` everywhere, so the streamed path is disk-bound and prefetch has nothing left to overlap on a cold cache; its win is the warm-cache case. The real levers are the resident encoder and the cache. LTX-2 has a ~2.4 s per-prompt floor that is neither Gemma nor the connectors (a cache hit pays it) — unexplained, tokenizer parse suspected. H3 loads the resident encoder (27 s) even when the prompt is a cache hit — wasted in a one-shot run.
- Parity, first hardware runs: H3 text vs transformers — ids exact, tap 50 rel 1.37e-2 streamed and 1.36e-2 resident FP8 (FP8 vs our native 4.2e-3; generation-time drift check 4.15e-3, cosine 0.99999). Gemma-3 vs transformers — all 49 hidden states pass, worst 1.65e-2 at the normed last state, ids exact; connectors vs float32 3.2e-7; end to end vs the bf16 reference 3.4e-3 / 3.7e-3. LTX-2 DiT judged against a float32 reference: ours is CLOSER to float32 than the reference's own bf16 pass at every loop step (step 7 video 0.212 vs 0.371); rope tables 10/10, forward 63/66, loop 17/22 — the misses are early audio-video cross-attention taps at 1.3-1.6x the bf16 floor (gate 1.25x), audio steps 6-7 at ~1.1x the gate, a 35 dB end-to-end frame PSNR gate that assumes non-diverging trajectories, and the vocoder at 40.9 dB SNR vs a 50 dB gate. Gates not yet re-set.
- All three reference dumps are now in the oracle cache (h3 text 19 MB, ltx2 text 849 MB, ltx2 dit 1.2 GB): these tiers no longer build a Python environment or load a reference model.
- Hosts: machine 111175 rented GPUs with ~60 GB held by another process (two LTX-2 OOMs and one reference OOM before free-memory logging showed it); `remote.sh env` now fails on a non-empty GPU, which marks the machine bad. Machine 150596 (38.146.30.19) downloads at 0-62 MiB/s; blacklisted. A chained tier idles ~5 min at hand-off because a leftover child keeps the log pipe open — not fixed yet.
- Cost: about $2.9 for everything above, of which ~$1.1 was lost to the two bad hosts.
- Reversibility: n/a (measurements); the harness changes are env-gated (`FV_TEXT_PLAN`, `FV_GPU_RAM_MIN`, `FV_DISK_GB`, `FV_MAX_GPU_USED_MIB`).

### FVID · 2026-09-19 · FVID-2026-09-19-text-encoder-prefetch
- Trigger: last step of the text-encoding plan the user approved ("let's follow your recommended order", then "proceed with step 5"). After the conditioning cache and the resident encoders, the streamed forward is still what runs on a card without room for a resident encoder and on every first prompt, and it was a strict sequence per layer: page weights in, convert, copy from pageable memory, then launch — the device idle for the first three.
- Decision: **the streamed encoder stages layer i+1 while layer i computes** (`llm/prefetch.rs`). A worker thread fills one layer-sized page-locked buffer straight from the mapped shards and uploads each tensor on a second, non-blocking stream; one staged layer waits in a bounded channel. Both ports get it without a change, since both stream through `llm::hidden_states`. `FASTVIDEO_LLM_PREFETCH=0` turns it off; it turns itself off (with a log line) when bf16 linears are off or the pinned allocation is refused.
- Synchronization is by hand, because the crate runs cudarc with event tracking off: the worker synchronizes the copy stream before handing a layer over, so compute never sees a partial weight; the weights live on the copy stream, and before a used layer is dropped the copy stream is made to wait on an event recorded on the compute stream, so the stream-ordered free cannot recycle memory that queued kernels still read. Cost: up to three layers on the device instead of one (~3 GB for Qwen3-VL-32B), one layer of pinned host memory.
- Same numbers by construction, and checked: one bf16 conversion (`fill_bf16`) now feeds both the plain load and the pinned buffer (host test: stored bf16 bit-exact incl. NaN payloads, f32/f16 rounded identically, across the chunk boundary); `Layer::assemble` builds a layer for both paths and a host test pins that the staged part list is exactly what it asks for, for both presets. The kernels tier gained `llm_prefetch_equals_streamed_{BF16,F32}`: a real safetensors file, encoded both ways on the device, every tap compared at tolerance zero.
- Not measured yet: nothing here has run on a GPU (rule: no GPU for what the host can check). The run log line `llm prefetch: N layers, host read+convert, upload wait, compute starved` says which side is the bottleneck; if `compute starved` is near the old text-encode time the disk is the limit and only the slim checkpoints / resident paths help further.
- Reversibility: trivial — one env flag, one module; the plain `Streamed` source is unchanged and is what the device check compares against.

### FVID · 2026-09-19 · FVID-2026-09-19-audio-video-ports-h3-ltx2
- Trigger: Wan / FastWan is silent ("I don't hear any audio"). The user asked for `FastVideo/FastVideo-FastH3-8-Step-V2` (DMD2-distilled MiniMax-H3) and LTX-2, both text-to-audio-video, ported in parallel onto the cudarc backend, dev card RTX PRO 6000 96 GB.
- Decision: **both ports exist and generate; specs in docs/ports/{h3,ltx2}.md.** Shared foundation first (lazy shard-aware loader with disk bf16 -> device bf16; `llm`, one streaming decoder-only encoder for Qwen3-VL-32B and Gemma-3-12B; rotate_half RoPE with explicit tables, GQA, erf-GELU, Snake, reflect/replicate pad, GroupNorm, cuDNN conv with dilation/groups/transposed, conv1d; WAV + AAC mux), then one agent per model writing only its own modules.
  - H3: Qwen3-VL tap 50 streamed (shards 12-14 never read); the ~26 GB of AdaLN weights never reside — each block's `adaln_proj` is streamed once and the `[8,50,3,6,5376]` table kept; VSA-H3 reuses the existing tensor-core fine kernel unchanged (forced prefix columns as a 1e30 bias before the ordinary top-k, prefix query rows recomputed dense).
  - LTX-2: distilled DiT + connectors from the official single file through a key-rename view (proved identical, to the last digit, to the community diffusers conversion); video VAE chunking that is exact (two-frame carry per 3x3x3 conv) where diffusers' tiling is a blend.
- Measured on one RTX PRO 6000, our binary alone (no Python on the box), first clip in the process:

  | | FastH3 8-Step V2 | LTX-2 19B distilled (stage 1) |
  | --- | ---: | ---: |
  | clip | 1344x768, 124 frames @ 24 fps, 32 kHz stereo | 768x512, 121 frames @ 24 fps, 24 kHz stereo |
  | text encode (streamed) | 10.6 s | 15.7 s |
  | denoise, 8 steps | 94.1 s (11.8 s/step, VSA-H3 80%) | 13.5 s (1.59 s/step) |
  | audio decode | 0.39 s | 0.43 s |
  | video decode | 22.9 s | 1.59 s |
  | write tail | 0.31 s | 0.24 s |
  | weight loads (once) | 15.0 s | 19.4 s |
  | wall | 145.5 s | 51.4 s |
  | warm, new prompt / cached embeddings | ~129 s / ~118 s | ~31 s / ~16 s |
  | peak VRAM | 53.6 GB | 39.9 GB |

  H3 dense attention is 87 s/step at 38k tokens, so VSA-H3 is 7.4x. The tiled NVRTC "flash" SDPA is ~20x SLOWER than chunked cuBLAS at that length (25 blocks in 15 minutes) — it is a memory tool, not a speed one.
- Parity against diffusers on hardware: H3 text tap 50 rel 1.37e-2 (bf16 both sides), token ids exact; H3 audio decoder 1.29e-5 and video decoder 1.47e-6 once the reference was pinned to real float32 (PyTorch's default TF32 convolutions had made a "float32" oracle 3e-3 wrong); H3 DiT forward 1.30e-2 video / 9.2e-3 audio with layout and both sigma ladders bitwise; VSA-H3 3e-3 vs f64 host loops; LTX-2 video VAE PSNR 91 dB, streamed == whole to 1.2e-6. Eight-step trajectories drift from a bf16 reference by ~2x per step in BOTH ports (H3 0.40, LTX-2 video 0.34 at step 7) while producing coherent, on-prompt clips: few-step sampling amplifies rounding; late-step trajectory parity against a bf16 reference is the wrong gate.
- Published comparison (FastH3 blog, warm E2E, 5 s at 1344x768): base H3 dense 132.5 s on 1x B200; FastH3 Preview v1 (4-step, 90% sparse) 16.2 s on 1x B200, 6.1 s on 4x. Ours is the 8-step / 80% checkpoint on a card ~2.5-3x slower than a B200, with f32 activations around bf16 GEMMs. Lightricks publishes no latency figure for LTX-2.
- Process rules the user set today, now in memory: never build/run/test on a GPU unless the step needs one (binary from CI via `task dist:ci`; key-manifest tests and CPU tiny-references against diffusers' own classes guard every loader and every structural property on a laptop); use published numbers rather than building oracle infrastructure. Costly lessons behind them: three oracle runs lost to venv environment bugs, an LTX-2 key mismatch found on a rented box, a reused box running a stale binary, a box idling through a git clone.
- Next levers, by measured share: H3 denoise is 80% of a warm clip (bf16 activations in the block, fused modulation + SwiGLU), H3 VAE decode 19% (larger tile batches); for LTX-2 a new prompt is half text encoding (Gemma ships as 47 GB of float32 — cache embeddings, or store a bf16 copy).
- Reversibility: cheap — new modules only; the Wan path is untouched and still passes its kernels and parity tiers.

### FVID · 2026-09-19 · FVID-2026-09-19-taehv-default-streaming-write
- Trigger: with exact-SM cubins and the tensor-core fine stage in, an 8s clip was 19.7s (H100 NVL) / 20.7s (RTX 5090) and only ~25% of it was the DiT. The Wan VAE decode and the serial PNG-then-ffmpeg write were the next two terms, and neither is a kernel problem.
- Decision: **the `gen` tier decodes with TAEHV by default (`FV_TAEHV=0` for the Wan VAE) and frames are written while the decoder runs.**
  - `remote.sh fetch-taehv` pulls and verifies `taew2_1.safetensors` behind the text encoding; the clip stage gets `--taehv-weights`. The UI passes `{"taehv": false}` through as `FV_TAEHV=0`.
  - `TaeHv::decode_streaming` / `AutoencoderKlWan::decode_streaming` hand each chunk's finished frames to a sink; `Pipeline::decode_latents_streaming` wraps both and `decode_latents` is the no-op-sink case, so nothing else changes. `pack_rgb_u8` turns a chunk into interleaved 8-bit RGB on the device (3 bytes/pixel down instead of 12, no host clamp loop); a `VideoWriter` thread encodes PNGs in parallel with rayon and streams rgb24 into ffmpeg's stdin, so the mux needs no PNG round trip. Byte-exactness against the old writer is a kernels-tier check (`pack_rgb_u8_matches_host`).
  - `--warm` (`FV_WARM=1`, `task gen WARM=1`) runs one untimed generation first; the timings block now separates `load_s` from `generate_s` (denoise + decode + write), which is what a served request costs.
- Measured, 129 frames 832x448, 3 DMD steps, VSA, same prompt and seed, **first clip in the process** (cold):

  | | RTX 5090 (`20260919T130240Z-gen`) | H100 NVL (`20260919T130210Z-gen`) |
  | --- | ---: | ---: |
  | weight load | 4.0s | 11.2s |
  | denoise (3 steps) | 5.50s (1.93 / 1.78 / 1.78) | 4.61s (1.73 / 1.42 / 1.42) |
  | decode (TAEHV, chunked, frames streamed) | 0.64s (`taehv.decode` 505ms) | 1.16s (`taehv.decode` 693ms) |
  | write tail after decode | 0.10s | 0.24s |
  | **generate (denoise + decode + write)** | **6.24s** | **6.01s** |
  | peak VRAM | 11.6 GB | 11.3 GB |

  Both gates pass (129 frames, video_quality), 1.06 MB mp4 each, contact sheets identical across cards, and the prompt is what is on screen. Both boxes booted from the CI image by build id (no upload).
- **Warm** (`--warm`: one untimed generation first, so this is a resident pipeline answering its next request), same clip:

  | | RTX 5090 (`20260919T131942Z-gen`, machine 149252) | H100 NVL (`20260919T131944Z-gen`) |
  | --- | ---: | ---: |
  | denoise (3 steps) | 5.71s (1.87 / 1.89 / 1.89) | 4.31s (1.42 / 1.42 / 1.42) |
  | decode (TAEHV; `taehv.decode` alone) | 0.59s (458ms) | 0.97s (606ms) |
  | write tail after decode | 0.09s | 0.23s |
  | **generate** | **6.43s** (0.80 s per video-second) | **5.60s** (0.69 s per video-second) |

  Warm vs cold: the H100 sheds its 0.3s first-step penalty and 0.2s of decode; the 5090 sheds nothing beyond the load (its first step was already within noise of the rest — this host's 5090 ran 1.88s/step against 1.78s on the cold run's host). So on either card the cost of a clip after the first is denoise + ~1s, and the first clip adds only the weight load (2.9–4.0s on the 5090 hosts, 11.3s on the H100 host, which is disk, not GPU).
- Reversibility: cheap — `FV_TAEHV=0` restores the Wan VAE; the streaming sink defaults to a no-op.

### FVID · 2026-09-19 · FVID-2026-09-19-cuda13-aot-cubins-ci
- Trigger: the Blackwell arch-mapping bug (`nvrtc_arch` → None → compute_52) showed that choosing the kernel target at run time is a class of failure, not an instance. And the user's point stood: NVRTC 12.4 with no nvcc, and Docker builds under x86 emulation, were facts about the laptop, not constraints on the project.
- Decision: **CUDA 13.0 everywhere, kernels compiled ahead of time by nvcc, builds off the Mac.**
  - `kernels.cu` is a file; `include_str!` feeds the NVRTC fallback from the same text nvcc compiles.
  - `build.rs` embeds a cubin + PTX per SM in `FV_CUBIN_SMS` (75, 80, 86, 89, 90, 100, 120). No nvcc → empty table and a warning; nvcc that fails → hard error.
  - `KernelFns::load_for`: exact-SM cubin → highest same-major embedded PTX → NVRTC. The device banner names which (`kernels=Cubin(86)`). `FASTVIDEO_KERNELS=nvrtc` keeps the source path exercised.
  - `nvrtc_arches` tries compute_100/120 natively (NVRTC 13 knows them) with compute_90 as the fallback for an older libnvrtc; an unknown future SM gets the newest PTX, never None. Pinned by a test.
  - Builder and runtime images from NVIDIA's apt packages (`cuda-nvcc-13-0`, `cuda-nvrtc-13-0`, `libcublas-13-0`, `libcudnn9-cuda-13`); the PyPI `-cu13` wheels are placeholders. `CUDARC_CUDA_VERSION=13000`.
  - Offer filter `cuda_vers>=13.0`: the 13.0 runtime libraries need a >= 580 driver and there is no minor-version compatibility across a major. Measured cost: **314 of 353 sm80+ offers (89%) qualify**, including 38 of 40 RTX 5090s.
  - CI builds the runtime image on **every branch**, tagged by build id — the tag `validate.sh` looks up — so a feature branch's boxes boot with the binary baked in. `:latest` only from main. The binary is also uploaded as a plain artifact. The T0 workflow fails if `build.rs` did not embed cubins.
  - `build-remote.sh` / `task build:remote`: rent the cheapest box with >= 8 cores, install the `cuda-builder.Dockerfile` recipe, build, pull the dist back. First real run: **GTX 1660 S at $0.048/hr, toolkit 100s, `cargo build --release` 60s, all seven cubins, 236s, ~$0.003.**
- **Verified end to end** (`20260919T120223Z-kernels`, RTX 3060, driver 580.126 / CUDA 13.0): image `build-d5f94b79e476c1e2` found by build id, **no upload — 6s from ssh to stages** against 8.5 minutes on one host the day before; nvrtc 13.0.88 / cuBLAS 13.1.1 / cuDNN 9.26 loaded; banner `kernels=Cubin(86)`; **T1 PASS, 0 failures**. The nvcc-compiled `mma.sync` kernel matches the host reference at rel_l2 0.0028 / 0.0031 — identical to NVRTC on the 5090 — at 3.0x / 2.7x / 4.3x over the gather path.
- **End-to-end effect of the tensor-core fine stage**, profiled on a 5090 (`20260919T114126Z-gen`, dispatched by `auto`): self-attention **74.7% → 33.0%** of a DiT block; the fine stage that was gather + QK/softmax + PV (~63% of the block) is now one phase at **13.0%**. The block is FFN 21.7%, self-attn 33.0%, cross-attn 13.8%; attention is no longer the dominant term. Remaining attention cost is mostly `tile_mean` (8.2%), the cheap fusion noted earlier.
- Two build-box lessons, both fixed: rsync creates only the last path component (a bare image has no `/workspace`, reported as receiver IO error 11), and validate.sh's inherited cleanup tried to pull a validation dir a build box does not have.
- Reversibility: medium — the driver floor and the 13.x soname are the coupling; dropping back to 12.x is a Dockerfile/env edit but re-admits the run-time target choice.
- Executed by: Executor
- Not done: a Hopper `wgmma`/TMA variant of the fine stage (`mma.sync` runs on sm90 but is not its peak path), and `tile_mean` fusion.

### FVID · 2026-09-19 · FVID-2026-09-19-vsa-tensor-core-fine-stage
- Trigger: the profile put the K/V gather at 48% of self-attention, ~21% of the DiT (FVID-2026-09-19-where-the-time-actually-goes), and a survey of every reference block-sparse kernel — FastVideo's Triton `_attn_fwd_sparse`, its ThunderKittens sm90 kernel, its sm100a kernel, FlashAttention-4's block sparsity — found that **none of them gather**. They permute Q/K/V into tile-contiguous order once, then one CUDA block per query tile re-points at each selected 64-row slab and folds it into an online softmax. The only FastVideo path that gathers is VMoBA, and only because it dispatches to `flash_attn_varlen`, which cannot take a block index. The gather is a consequence of calling cuBLAS per query tile, not of sparsity.
- Decision: **replace the fine stage with a tensor-core streaming kernel** (`vsa_mma_attn`): `mma.sync.m16n8k16` bf16, `ldmatrix` fragment loads over an XOR-swizzled tile, `cp.async` double buffering so tile i+1 loads while i computes, and P repacked from S's C-fragment layout straight into A-fragment layout so it never leaves registers. `vsa_tile_qkv` is the reference's `tile()`: O(padded × dim) once, against the gather's O(tiles × topk × 64 × dim) — 531 GB written per 8s clip.
- **Verified, RTX 5090** (`20260919T111737Z-kernels`, T1 PASS): against the host reference rel_l2 **0.0029 / 0.0031** — slightly better than the gather path's 0.0032 / 0.0033, since P stays in registers. Speed against the gather path, median of three:

  | tokens | topk | gather | mma | |
  | ---: | ---: | ---: | ---: | ---: |
  | 1,456 | 19 | 2.44 ms | 0.82 ms | 2.98x |
  | 4,368 | 19 | 2.50 ms | 0.88 ms | 2.85x |
  | 13,104 | 55 | 19.20 ms | 3.54 ms | **5.42x** |

  Reproduced in fast mode to within 1%. The speedup grows with token count; an 8s clip is 48,048 tokens.
- **This closes FVID-2026-09-18-fused-block-sparse-rejected the right way.** That kernel had the same structure and lost 8x on this same card class; this one wins 5.4x. The swing between the two is ~43x, and the only difference is scalar f32 math versus tensor cores. The idea was never wrong — the recorded rule ("do not hand-write attention math that cuBLAS can express") generalised one bad implementation into a closed door. The rule that survives is narrower: a hand-written kernel must use the tensor cores, or it will lose to a library that does.
- Hardware dispatch: `FASTVIDEO_VSA_KERNEL=auto|gather|fused|mma`. `auto` is now `mma` on sm80+ at dim 128, `gather` otherwise; the scalar kernel stays selectable as the negative result it is. The kernel body is guarded on `__CUDA_ARCH__ >= 800` and **traps** below it, so a wrong-arch build reports as a CUDA error rather than an uninitialised buffer.
- **Found on the way, and larger than this kernel**: `nvrtc_arch` returned `None` for sm_major 12, so on Blackwell NVRTC ran with no `-arch`, defaulted to compute_52, and compiled every `>= 800` guarded body out. The first run of this kernel was a no-op that reported garbage and a fake 3.04x — and *every kernel we had ever run on a 5090 or RTX PRO 4500* was JIT'd from compute_52 PTX. Hopper and later now target compute_90 (libnvrtc 12.4's ceiling; PTX is forward-compatible), pinned by a test. The per-arch compile gate never saw it because the gate names its targets; only the runtime path chose.
- Second lesson from that first run: `auto` had been pointed at the unverified kernel, so the "gather" baseline checks silently ran the stub too, and a "gather failure" was a second copy of the same failure. The baseline checks now pin the kernel by name, and a default may only point at a verified kernel — the flip is its own commit.
- Reversibility: cheap (`FASTVIDEO_VSA_KERNEL=gather`)
- Executed by: Executor
- Not yet measured: the end-to-end effect on a clip (`--profile` puts the whole fine stage under `vsa_4_mma`), and a Hopper `wgmma`/TMA variant — `mma.sync` runs on sm90 but is not its peak path.

### FVID · 2026-09-19 · FVID-2026-09-19-where-the-time-actually-goes
- Trigger: four hypotheses about the ~4x gap to FastWan-QAD were each built, measured and wrong — FP8 linears (4.8%), CUDA graphs (0.7%), a newer cuBLAS (0%), and fusing norms/residuals (predicted large). All four reasoned from upstream's description of their own work rather than from our stack. Profiling instead.
- **Measured, DiT block phases** (`--profile`, per-phase syncs, so proportions not absolutes): self-attention **74.7%**, FFN 14.2%, cross-attention 9.1%, and all five norms and gated residuals **2.0% combined**. The fusion targets FastWan-QAD lists first are worth 2% *in our implementation*.
- **Measured, VSA stages**: of self-attention, the **K/V gather is 48.2%** — **~21% of the whole DiT**, more than the FFN and cross-attention together. Fine QK+softmax 30.3%, tile means 8.7%, fine PV 6.9%, everything else 2.7%. The eight stages account for 50.04s of 51.67s, so nothing material is unattributed.
- Why FLOP arithmetic missed it: the gather is pure data movement. It materialises topk×tile_elems of K and V per query tile, 810 times per clip, and contributes no FLOPs at all — so an estimate built from GEMM throughput and attention FLOPs put attention at 8% of a step when it is 75%.
- **This reopens FVID-2026-09-18-fused-block-sparse-rejected.** That kernel existed to eliminate this exact gather, lost at 8x, and was recorded with a general rule: "do not hand-write attention math that cuBLAS can express." The measurement stands; the generalisation does not. One bad implementation of an idea was written up as if the idea were closed, and the idea's target turns out to be the largest single cost in the model. The rule should read as "a hand-written kernel must beat cuBLAS on the *math*" — it says nothing about a kernel whose purpose is to avoid materialising data.
- Second, smaller finding: `vsa_1_tile_mean` costs 8.7% of self-attention in **90 calls** — three separate mean-pools over full Q/K/V. Fusing them into one pass is contained and worth ~3.8% of the DiT.
- Reversibility: n/a (measurement only)
- Executed by: Executor
- Verification: `20260919T011154Z-gen` (block phases, RTX 3090) and `20260919T013314Z-gen` (VSA stages, RTX A5000). ~$0.20 for both. Absolute seconds are inflated by profiling syncs and the A5000 is slow; only the proportions are claimed.
- Process note: the profile cost ~$0.10 and was worth more than the four hypotheses that preceded it. Measure the stack before porting the other project's answers to it.

### FVID · 2026-09-18 · FVID-2026-09-18-cuda-graphs-not-worth-it
- Trigger: we sit ~4x behind FastWan-QAD's published 3.4s for a 5s 480p clip on a 4090, and the suspicion was launch overhead — our DiT launches a separate kernel per op with a fresh allocation, which is exactly what `torch.compile` and CUDA graphs remove. ADR-0003 had also deferred graph capture, so it looked like unclaimed ground.
- Two corrections on the way to the measurement. First, `FASTVIDEO_CUGRAPH` is **not implemented and not merely disabled**: commit `7566769` — the ADR-0003 pass itself — deleted `streams.rs` and rewrote `transformer.rs` from 994 lines to 237. The only surviving copy is an unmerged worktree 63 commits behind whose own commit is about schedulers. Second, the reason ADR-0003 gave for deferring capture ("capturing a host-bouncing block body isn't safe") was fixed by that same commit, which added the `layer_norm_last` / `modulate_scale_shift_last` kernels that removed the bounce. The conclusion outlived its premise.
- **Decision: do not build CUDA graph capture.** Measured on a 3090, 2s clip, VSA: **3,112 kernel launches per denoising step**, against a measured **2,237 ms** per step. At ~5 µs of launch overhead that is ~16 ms, or **0.7%**; at a pessimistic 10 µs, 1.4%. Graphs cannot be worth more than one or two percent here.
- The same run settles the other half: **zero host fallbacks**, and 6 device-to-host transfers for an entire clip (the frame readback). ADR-0003's blocker is gone — but it no longer matters, because the thing it was blocking is not worth doing.
- So the remaining gap to FastWan-QAD is **arithmetic, not overhead**. This also kills launch overhead as the explanation for the H100 dense result tying a 4090 (27.3s vs 27.5s), which needs a different cause — `mathprobe` on Hopper is the next measurement, since it reports which cuBLAS math mode the hardware actually honours.
- Of upstream's levers, FP8 linears are measured at 4.8% (FVID-2026-09-18-fp8-linears-measured) and TAEHV is adopted, leaving **SageAttention2++** as the only large untested one. Attention's share of the 2,237 ms per step should be estimated before building it.
- Incidental: 911 host-to-device transfers totalling 3.4 GiB during a 3-step clip. Not overhead-critical, but plausibly uploads repeated per step rather than once — worth a look.
- Reversibility: n/a (nothing built)
- Executed by: Executor
- Verification: `20260918T231508Z-gen`, ~$0.05. The counter lives in the one `launch!` macro every kernel goes through, so no call site can under-report, and clip reports now carry `device_stats` by default.

### FVID · 2026-09-18 · FVID-2026-09-18-taehv-adopted
- Trigger: the TAEHV port was verified against the reference (FVID-2026-09-18-taehv-port) but never run in the pipeline, so the quality and end-to-end time trade were both unmeasured.
- Decision: **TAEHV is worth using**, behind `FASTVIDEO_TAEHV_WEIGHTS=<dir>`. A/B on one RTX 3090, 8s clip (129 frames, 832x448), same prompt and seed, VSA on both sides, latents **bit-identical** (rel_l2 0.0) so the decoder is the only variable:

  | | Wan VAE | TAEHV |
  | --- | ---: | ---: |
  | VAE decode | 16.18s | **1.66s** (9.7x) |
  | total | 78.86s | **64.71s** (-18%) |
  | peak VRAM | 21.4 GB | **12.1 GB** (-44%) |

- Quality: **34.16 dB** PSNR between the two decodes. For scale, our own exact-vs-fast mode differs by 27.8 dB, so TAEHV sits closer to the Wan VAE than our fast path sits to our exact path. Visually the composition, motion and lighting are identical; TAEHV is slightly softer in fine detail (fur, foam).
- The memory drop matters as much as the time. 21.4 GB took an 8s clip to the edge of a 24GB card — the constraint behind this morning's OOM on a reused box and behind `--vae-chunk 2` existing at all. 12.1 GB is comfortable.
- **I had the structural trade backwards.** TAEHV being parallel over frames was described as an advantage over the Wan VAE's sequential feature cache; it is both. Every stage materialises every frame, and the deepest is ~12.6 GB for 129 frames, so the first A/B OOM'd. The fix is the reference's own sequential path: chunk the decode and carry each MemBlock's boundary frame across the seam. The oracle test ran 3 latent frames, where that liability cannot appear — a reminder that a correctness oracle at toy sizes says nothing about production shapes.
- Second failure worth recording: 33 latent frames at chunk 4 ends in a chunk of **one**, where there is no earlier frame to shift in. Only reachable at real clip lengths; the unit tests used frame counts that never produced a ragged tail. Both failures now have tests that fail without the fix.
- Reversibility: cheap — off unless the weights path is set, and a bad path errors rather than silently falling back to the Wan VAE.
- Executed by: Executor
- Verification: `20260918T195506Z-vaeab`. Three attempts, ~$0.16 total including the two failures.
- Not done: TAEHV is not on by default, and no clip tier enables it. The quality trade is real if small, so that should be a deliberate choice rather than a default.

### FVID · 2026-09-18 · FVID-2026-09-18-taehv-port
- Trigger: the Wan VAE is the largest component of a clip never attacked — 3.9s of a 23.7s 8-second clip on an H100 — and the only major block never checked against an external reference. FastWan-QAD's speedup leans on TAEHV for exactly this reason.
- Decision: **port the TAEHV decoder** (madebyollin/taehv, `taew2_1`) as an alternative to `AutoencoderKLWan`. Decoder only; text-to-video never encodes.
- Verified against the implementation it was read from, first hardware run: `video` rel_l2 **2.6e-4** (cosine 0.99999997) against a 2e-2 limit, output shape `[9, 3, 448, 832]` exact. `20260918T182450Z-taehv`, RTX 3060, **$0.008**.
- **Faster than the reference it copies**: decode 1.332s (reference PyTorch) → **0.443s** (ours), same card, same latent, same weights — 3.0x. Our port is parallel over frames because TAEHV's `past` is only a one-frame shift, so unlike the Wan VAE there is no sequential feature cache to serialise on.
- The measurement that could not be made by reading: TAEHV documents "~Gaussian" input, which had to mean **DiT-space** latents, *before* the per-channel `latents_mean`/`latents_std` un-normalisation the Wan VAE requires. Getting that backwards yields a plausible, wrongly-coloured video rather than an error — the same failure shape that hid the mirrored UMT5 bias. The oracle settles it.
- Every architectural constant inferred from the source was confirmed by the reference: `patch_size` 1, `latent_channels` 16, `t_upscale` 4, `frames_to_trim` 3, and T latent frames → 4T−3 output frames, which is Wan's 4n+1.
- Reversibility: cheap — additive, nothing existing changed.
- Executed by: Executor
- **Not yet answered**: what TAEHV costs in *quality* on a real clip, and what it saves end to end against the Wan VAE's 3.9s. Both need it wired into the pipeline and a clip run. Speed is established; the trade is not.

### FVID · 2026-09-18 · FVID-2026-09-18-fp8-linears-measured
- Trigger: FastWan-QAD (haoailab.com/blogs/fastwan-qad) claims 3.4s for a 5s 480p clip on a 4090 via FP8 linears + FP8 attention + TAEHV + full compile. The checkpoints turned out to be **unquantized** — F32 weights, a transformer config byte-identical to stock Wan2.1-1.3B, no quantization_config — so "FP8" names the precision the model was *trained to tolerate*, and every part of the speedup is ours to build.
- Built: `fastvideo-ops::fp8` (E4M3 conversion, exhaustively tested over all 256 codes), four CUDA kernels mirroring it, a cuBLASLt E4M3 GEMM, and FP8 linears behind `FASTVIDEO_FP8`.
- **The plumbing is correct.** On two sm89 cards (L40S and RTX 4090) the device quantizer matches the host reference on all 65,536 values with **zero** tolerance, and the GEMM lands at rel_l2 **1.39e-4** against a 2e-2 limit — bit-identical across both cards. Because both sides of that check see identical quantized operands against an f64 reference, 1.39e-4 is cuBLASLt's accumulation order alone; a wrong transpose would have been order-1 wrong.
- **Decision: FP8 linears stay off.** A/B on one 4090, FastWan-QAD-FP8-1.3B weights, same prompt and seed, 2s clip: denoise **4.207s → 4.007s (−4.8%)**, model load **6.49s → 12.76s**, total **14.36s → 20.19s (41% slower)**. Weight memory does fall 3607 → 2263 MiB (−37%).
- Accuracy cost: step-1 latents rel_l2 **0.108**, frames **16.9 dB** — against 4.4e-3 and 66 dB for our bf16 fast path. ~25x more error than bf16, on the checkpoint distilled specifically to tolerate per-tensor FP8.
- Why only 4.8%: the GEMMs really do run on FP8 tensor cores, but **activation quantization eats the win**. Every linear gains an amax reduction and a quantize pass over its input — two memory-bound passes over data the GEMM was about to read anyway. At Wan 1.3B's shapes, halving operand bandwidth does not outrun them. The load regression is separate and fixable (host-side element-by-element quantization of 1.3B parameters), but fixing it only removes a regression rather than creating a win.
- **Third instance of the same shape**, after FVID-2026-09-17-flash-sdpa-rejected and FVID-2026-09-18-fused-block-sparse-rejected: a correct implementation that loses to the simpler path. The cause differs — this one is not about tensor cores, it is about per-op overhead around them.
- What would have to change to revisit: fusing the quantization into the op that already writes the activation (the RMS norm, or the bias/activation epilogue), so it costs nothing extra. Even then the ceiling is the GEMM half of denoise, so expect 10-15%, against TAEHV which targets 3.9s of VAE out of a 23.7s H100 clip at far lower risk.
- Reversibility: cheap — the path is behind a flag that is off by default and never inferred from the device.
- Executed by: Executor
- Verification: `20260918T172402Z-kernels` (L40S, 405 checks, 0 fail) and `20260918T173147Z-fp8` (RTX 4090, A/B). $0.15 for both.

### FVID · 2026-09-18 · FVID-2026-09-18-umt5-bias-mirrored
- Trigger: user reported that generated clips do not follow the prompt. Confirmed visually: "a piper cub takes off" rendered a hand holding a green pepper; the benchmark prompt "A golden retriever sprints along the shoreline at sunset, waves breaking around its paws" rendered a static dog on grass. Upstream FastVideo, on the same GPU and the same weights, rendered the puppy on a beach with waves.
- Root cause: `relative_position_bucket` added the half-table offset when the key came **before** the query; HF adds it when `relative_position > 0`, i.e. after. The two halves of the learned 32-row relative attention bias were swapped. For a 12-token prompt **132 of 144 entries were wrong** — only the zero-distance diagonal survived.
- Why it presented as "keeps the subject, loses the scene": UMT5 carries no absolute or rotary positional encoding, so this bias, injected into all 24 encoder layers, is the model's *only* word-order signal. Mirroring it preserves token identity but binds modifiers, verbs and prepositional phrases to the wrong side.
- Decision: fix the predicate to `relative > 0` in all four backends (cudarc, models, burn, luminal — the function was copy-pasted, which is why every cross-backend comparison agreed with itself). Regression test takes its expectations from HF's formula rather than from our output, and fails against the old sign.
- Reversibility: cheap (one predicate)
- Executed by: Executor
- Verification: oracle tier `20260918T124220Z-oracle`, L40S 44GB, against transformers' UMT5 and diffusers' `WanTransformer3DModel` on identical inputs. Before the fix: text cosine **0.268** / rel_l2 **1.296** (FAIL), dit cosine **0.9999999** / rel_l2 **0.00045** (PASS), e2e cosine 0.823 / rel_l2 0.607 (FAIL).
- **The DiT port was exact the whole time.** Every attention, RoPE, patch-embed, adaLN, VSA and bf16 change was correct; the entire prompt-adherence failure was one inverted comparison in the text encoder.
- Process note: this survived 42 green validation runs because **every gate compared fastvideo-rs to fastvideo-rs** — `parity` is GPU vs our own CPU path (and feeds the DiT *random* embeddings, so it never touched the text encoder at all), and `compare` diffs two of our own clip dirs. The video-quality gates score luma, temporal MAD and clipping, so a coherent wrong video passes them. See FVID-2026-09-18-oracle-tier.

### FVID · 2026-09-18 · FVID-2026-09-18-oracle-tier
- Trigger: a prompt-adherence bug that no tier could see, because no tier compared us to anything but ourselves.
- Options: tighten the existing self-comparisons; add semantic scoring (CLIP similarity) to the clip gates; diff against the reference implementations on identical inputs
- Decision: **an `oracle` tier.** `scripts/gpu/upstream_oracle.py` runs transformers' UMT5 and diffusers' `WanTransformer3DModel` on the same weights and saves its inputs and outputs; `fv-gpucheck oracle` replays them through us. Three checks, on byte-identical tensors: `text` (our embedding vs the reference), `dit` (our DiT on the *reference* embedding), `e2e` (our DiT on *our* embedding).
- `dit` is the load-bearing design choice: feeding both sides the same conditioning removes the text encoder from the comparison, so text-vs-DiT is **decidable** rather than a single number that says only "something is wrong". On its first run it attributed the defect immediately — text failed at rel_l2 1.296 while dit passed at 0.00045.
- Float32 throughout: the oracle judges exact mode, and a looser judge cannot set a limit. Needs 40GB because UMT5-XXL in torch float32 is ~22GB before the DiT loads.
- Reversibility: cheap (additive; no existing tier changes)
- Executed by: Executor
- Verification: `20260918T124220Z-oracle` (baseline, pre-fix) — the tier failed exactly the two checks it should and passed the one that isolates the DiT. Cost $0.32, 36 min, most of it the upstream install.

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
- Reason: this is the second hand-written attention kernel to lose to cuBLAS by a wide margin. The rule that generalises: **on this hardware, do not hand-write attention math that cuBLAS can express.** **(Superseded 2026-09-19 — see FVID-2026-09-19-vsa-tensor-core-fine-stage. The 8x measurement stands; the same structure on tensor cores wins 5.4x on the same card class. Only this implementation was wrong.)** A fused kernel is only worth attempting with `mma.sync` tensor-core intrinsics, and even then it must beat a library that is already near peak.
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

