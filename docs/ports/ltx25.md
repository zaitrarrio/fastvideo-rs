# LTX-2.5 (22B, distilled) — port specification

**Distilled T2AV**: Gemma 4 + distilled DiT + conv video VAE + audio
VAE/vocoder (BWE @ 48 kHz), CFG=1, ancestral Euler. Optional **two-stage**
path: half-res stage-1 → spatial latent upsampler ×2 → 3-step stage-2 at full
res. Optional **DiffVAE** (diffusion video decoder) replaces the conv VAE
decode for higher fidelity. No duration head or prompt enhancer.

Sources (read 2026-09-21): `Lightricks/LTX-2.5-Diffusers` configs,
`Lightricks/LTX-2` `packages/ltx-pipelines` / `ltx-core`, diffusers
`transformer_ltx2.py` / `latent_upsampler.py` / `ltx2_diffusion_decoder.py`
on `main`.

Sibling: [ltx2.md](ltx2.md) documents the LTX-2.0 port this extends.

---

## Weights

Prefer the Diffusers pack for oracle parity (`Lightricks/LTX-2.5-Diffusers`):

| component | path |
|---|---|
| Distilled DiT | `transformer/` |
| Connectors | `connectors/` |
| Gemma4 TE | `text_encoder/` + `tokenizer/` |
| Conv video VAE | `vae/` (or Comfy `vae/ltx-2.5-video-vae-conv-bf16.safetensors`) |
| Audio VAE | `audio_vae/` |
| Vocoder | `vocoder/` → **`LTX2VocoderWithBWE`** @ 48 kHz |
| Spatial upsampler | `latent_upsampler/` (two-stage only) |
| Diffusion decoder | `diffusion_decoder/` (DiffVAE; ~0.83 GB) |

Comfy split pack (`Lightricks/LTX-2.5`): one `.safetensors` per component;
projections may live inside the Gemma4 file. Key remap lives in
`ltx-core` `gemma_assets` / connector ops.

Distilled DiT declares `model_version` ≥ 2.5 in safetensors metadata → denoise
uses **ancestral** Euler (`eta=1`, `s_noise=1`, noise seed = pipeline seed +
10000). Stage-1: 8-sigma list; stage-2: tail `[0.909375, 0.725, 0.421875]`.

---

## Two-stage distilled

Request `height`/`width` = **final** canvas (multiples of **64**). Stage 1 runs
at half resolution; audio tokens match the full clip length (unchanged by the
spatial upsampler).

1. Ancestral stage-1 at `H/2 × W/2` (8 steps) → DiT-normalized packed latents.
2. Unpack video → **de-normalize** (`ẑ·std/scaling + mean`) →
   `LTX2LatentUpsamplerModel` → **re-normalize** → pack. Audio passthrough.
3. Renoise video and audio: `x ← σ·ε + (1−σ)·x` with `σ = 0.909375`.
4. Ancestral stage-2 at full `H×W` (3 steps), same DiT (no stage-2 LoRA on the
   distilled transformer).
5. Decode once (conv VAE or DiffVAE + BWE vocoder).

Upsampler architecture (`latent_upsampler/config.json` on 2.5):
`in_channels=128`, `mid_channels=1024`, `num_blocks_per_stage=4`, `dims=3`,
`use_rational_resampler=false` → Conv3d stem + 4 ResBlocks (GroupNorm-32 +
SiLU) → per-frame Conv2d→PixelShuffle(2) → 4 ResBlocks → Conv3d head.

First validation canvas: final **768×512×121** (stage-1 **384×256**).

---

## Offload placement (`FASTVIDEO_LTX_OFFLOAD=cpu`)

sol-engine's BF16 RTX 5090 profile peaks at 30.36 GiB allocated / 31.40 GiB
reserved at 4k5s and 29.07 / 30.15 GiB at 1080p20s. It gets there with the
official pipeline's `--offload cpu` and nothing else of its own apart from
the FFN chunking. The sources are sol-engine `6c2f582` and Lightricks/LTX-2
`fd4ded7`. Paths below are relative to `models/ltx25/RTX5090/` and
`packages/ltx-*/src/`.

**What sol-engine places where**

| model | where it lives | when it is freed |
|---|---|---|
| Gemma 4 text encoder | Streamed layer by layer (`StreamingModelBuilder` over `model.model.language_model.layers`, `blocks.py:696-705`, chosen by `_text_encoder_ctx`, `blocks.py:741-746`) | End of encode (`_streaming_model` teardown + dispose + `cleanup_memory`, `blocks.py:188-209`) |
| Embeddings processor (connectors) | Whole on the device (`blocks.py:790-795`) | Right after `process_hidden_states` (`gpu_model`, `gpu_model.py:13-34`) |
| DiT | Streamed per block. The non-block weights are on the device (`_load_non_block_weights`, `builder.py:412-439`). The 48 blocks sit in pinned host buffers, one per block (`_build_pinned_source`, `builder.py:304-364`). Two device slots are used (`_DEFAULT_GPU_SLOTS = 2`, `builder.py:52`). The copy is issued in the block's own pre-hook, with no lookahead (`wrapper.py:54-75`, `provider.py:65-84`) | Rebuilt for **each** stage call and freed at its end (`DiffusionStage.__call__` → `_streaming_transformer_ctx`, `blocks.py:489-499, 501-582`). The registry does not cache weights (`cache_weights=False`, `blocks.py:334`), so each stage re-reads the checkpoint into pinned memory. That load is inside sol's stage times |
| Video encoder + latent upsampler | Whole on the device (`VideoUpsampler.__call__`, `blocks.py:1027-1045`) | End of the upsample |
| Video VAE decoder | Whole on the device, with `AUTO_TILING` (conv VAE 768/64 spatial, 80/24 frames, `helpers.py:60-97`) | After the last chunk (`_cleanup_iter`, `blocks.py:238-245`, `1124-1149`) |
| Audio VAE + vocoder | Whole on the device (`blocks.py:1180-1203`) | End of the audio decode |

The allocator is trimmed after every model (`AllocatorTrimStrategy.TRIM`,
which runs `gc` + `empty_cache`). The run sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`run_ltx25_gpu.sh:38`).
The only memory measure of its own is FFN chunking: 16 384-row pieces when
at least 65 536 rows (`memory.py:6-43`, installed at `gpu_infer.py:300-302`).
So nothing but activations and one model's working set share the device.
Stage 2 is two DiT block slots plus the 130 560-token activations.

**Ours.** `FASTVIDEO_LTX_OFFLOAD=cpu` (`ltx2 gen --offload cpu`,
`PipelineOptions::offload`) selects that placement in one switch
(`fastvideo_models::ltx2::memory::LtxOffload`):

- The DiT is `streamed` (`wan/offload.rs`). Linear weights sit in pinned host
  memory and go through a `lookahead + 1 = 2` slot device ring. Block
  `i + 1` is copied on a second stream while block `i` computes. The ring is
  released at the end of each stage.
- Gemma is streamed per layer, and the connectors are loaded for the encode
  and then dropped (`TextResidency::Streamed`).
- The upsampler is loaded only for its call. The video VAE (for its latent
  statistics) and the audio VAE + vocoder are loaded for the upsample and
  for the decode, and dropped after each. The pool is trimmed after every
  phase.
- FFN chunking is always on (`FeedForwardChunking::RTX5090`).

It refuses an explicit `resident` DiT (`FASTVIDEO_DIT_OFFLOAD` /
`--dit-offload`) or an explicit resident text encoder (`FASTVIDEO_LTX2_TEXT` /
`--text`). The default (`none`) leaves every model to its own `auto` policy,
so the resident run is unchanged.

There are two deliberate differences from the reference:

- The DiT's host copy is built once per process, not once per stage. Our
  stage times do not include reading the checkpoint, and a warm request pays
  no reload.
- The DiT's device skeleton stays loaded through the upsample and the decode:
  the non-block weights, the per-block modulation tables, and the biases.
  That is about 0.7 GiB (`DitPlacement::dit_bytes` less the ring), and
  neither phase is the peak.

Placement never changes a number. The streamed block runs the same kernels
on the same weight bits (`streamed_blocks_are_bit_identical_to_resident`),
so the frames are byte-identical to the resident run.

**Measured** on RTX PRO 6000 Blackwell Server Edition, with the Sol stage 2,
bf16 activations and FFN chunking (all defaults). These are the matrix family
`ltxoffload` (`scripts/gpu/runpod-matrix.sh`), runs `a73173e-09270009`,
`a2cd194-09270039` and `a2cd194-09270103`. The peaks are the pool's
high-water marks, the equivalent of `torch.cuda.max_memory_allocated` /
`_reserved`. "smi" is the device-wide peak.

| cell | peak alloc / reserved GiB (phase) | smi GiB | stage 1 s | upsample s | stage 2 s | video VAE s | e2e s |
|---|---|---|---|---|---|---|---|
| 512p resident (default), warm | 40.63 / 40.97 (upsample) | 44.3 | 2.13 | 0.47 | 2.76 | 0.62 | 7.17 |
| 512p `cpu`, warm | 5.72 / 6.31 (decode) | 7.8 | 7.26 | 1.08 | 4.32 | 0.64 | 15.32 |
| 4k5s `cpu`, warm | **20.41 / 21.19** (decode; stage 2 19.24 / 20.69) | 22.3 | 45.16 | 4.81 | 75.57 | 14.56 | 189.75 |
| 1080p20s `cpu`, cold | **18.95 / 20.19** (stage 2) | 20.9 | 42.55 | 5.43 | 71.42 | 14.23 | 178.93 |
| sol-engine 4k5s (Sol) | 30.36 / 31.40 | 32.3 | 55.88 | in stage 2 | 79.08 | 27.91 | 196.99 |
| sol-engine 1080p20s (Sol) | 29.07 / 30.15 | 31.0 | 54.85 | in stage 2 | 73.66 | 20.44 | 163.46 |

The 512p resident and `cpu` rows both read the prompt's contexts from the
text cache. Every other row of ours encodes the prompt. The 4k5s and
1080p20s e2e times include 46.5 s and 43.4 s of streamed Gemma encoding
with no text cache. A cache hit takes about 1 s.

Findings:

- **Byte identity.** In run `a2cd194-09270039` every cell encoded its
  prompt. The 512p PNG frames of resident, `none` and `cpu` hash the same
  (`a84cec39fd1c8f18`), and both compares report `off_identity` ok. The
  4k5s and 1080p20s `cpu` frames also hash the same on two different pods.
- **The wav is not deterministic run to run, in any mode.** Resident and
  `none` differ from each other just as `cpu` differs from both. Placement
  has no part in it.
- **The text cache does not replay a fresh encode bit for bit.** In run
  `a73173e-09270009` the resident cell missed the cache and the other two
  hit it. Their latents differed from stage-1 step 0, while `none` and `cpu`
  matched at every step. That is why the family now passes
  `--no-text-cache`.
- **The `exact` gate** passes `official_config`, `quantitative_quality` and
  (with every cell encoding) `off_identity`. It fails `performance`: at
  512p, `cpu` runs `denoise_s` at 0.38x of resident. At 512p stage 1 is
  PCIe-bound, because 8 forwards x 48 blocks = 277 GiB at 57 GB/s against
  2 s of compute. At 4K the same copies take 5.3 s beside a 45 s stage 1 and
  are fully hidden. For a placement mode, this failure is expected.
- **About 4.9 GiB stays live from the upsample onward.** This is the cuDNN
  conv workspace cache (`wan/conv.rs`, `ConvCache::workspace`, grown to the
  largest conv and never shrunk). At 4K it is most of the gap between the
  stage-2 activations and the phase peak. Releasing it in the phase trims is
  the next cut if a smaller card needs one.

---

## NVFP4 video FFN (`FASTVIDEO_NVFP4`, profile `ltx2/ltx25_distill_sol_nvfp4`)

sol-engine `transforms/nvfp4_ffn.py`: TransformerEngine's NVFP4 GEMM on the
video feed-forward only (`transformer_blocks.*.ff.net.{0.proj,2}`, 48 x 2
linears, 4096 → 16384 → 4096), RHT and stochastic rounding off, rows padded
to 16, bf16 fallback. Default off.

- **GEMM**: cuBLASLt block-scaled FP4 (`CUDA_R_4F_E2M1` A/B,
  `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, f32 compute, bf16 D, bias in
  the epilogue), TN with the weight as A. `alpha = decode(x) · decode(w)`
  (`amax / (6 · 448)` each) is computed on the device and read with
  `CUBLASLT_POINTER_MODE_DEVICE`, so a call never synchronizes. Descriptors
  and the heuristic's algorithm are cached per shape
  (`wan/nvfp4_linear.rs`). cudarc 0.17.8 already binds every enum needed.
- **Weights**: the loaded bf16 weights are quantized once with the TE
  `NVFP4BlockScaling` rule (`static_6`: tensor amax, E4M3 scale per 16
  elements) and the bf16 copies dropped: 72 MiB per block instead of
  256 MiB (about 8.6 GiB less on the DiT). The NVFP4 FFN stays resident
  when the other block weights stream.
- **Activations**: `nvfp4_amax_bf16` then `nvfp4_quant_bf16_sw` write packed
  E2M1 and the scales directly in cuBLASLt's 128x4 tiled layout
  (`to_blocked`: tile `(r/128)·(K/64) + c/4`, byte `(r%32)·16 + (r%128/32)·4
  + c%4`), no swizzle pass. The down projection's quantizer reads
  `bf16(gelu_tanh(h))`, so the GELU output is never written. The FFN input
  comes out of the fused norm/modulate kernel (`fuse::res_norm_mod`) in
  bf16; its amax needs one extra read (a fused amax would need a delayed
  scale, which TE's current-scaling recipe does not use).
- **Scope**: an LTX process takes `FASTVIDEO_NVFP4` as this scope only; the
  LongLive dequant-beforehand path (every eligible linear, K/V) is off in
  it. `mse` has no tensor-core form and keeps the bf16 FFN (logged).
- sol-engine's RTX5090 pre-quantized checkpoint (`gpu_infer.py`) is not on
  the weight volume (`fv-weights-h3-ltx-hy`), so it is not loaded; the same
  scope is quantized at load instead.

**Parity** (`fv-gpucheck kernels`, group `nvfp4_linear`, RTX PRO 6000, run
`600c38f-09271221`): the operand quantizer's codes and swizzled scales equal
the host TE reference bit for bit (plain and GELU; 300 rows, pads zero);
the linear matches the exact dequantized host math (f64) at rel_l2 1.66e-3
(bf16 output rounding; m = 300, bias in the epilogue); output vs the dense
bf16 linear 37.6-40.9 dB PSNR, 37.8 dB at the FFN up shape (rel_l2 0.145,
the FP4 quantization itself).

**Microbench** (ms, RTX PRO 6000, run `8f12d44-09271258`, the shapes the
pipeline runs: stage 1 whole, stage 2 in 16 384-row chunks). "linear" =
activation quantize + GEMM + bias; for the down projection it includes the
GELU (the bf16 FFN's bf16 GELU pass; fused into the NVFP4 quantizer). bf16
linear = cuBLASLt bf16 with the bias epilogue; FP8 = the W8A8 tensorwise
linear (`FASTVIDEO_FP8`, activation quantize included, no bias, no GELU).

| shape (M, K → N) | bf16 GEMM | bf16 linear | FP8 linear | NVFP4 GEMM | NVFP4 linear | vs bf16 linear |
|---|---|---|---|---|---|---|
| 4K stage 1 up (32640, 4096 → 16384) | 11.16 | 10.81 | 6.33 | 3.15 | 3.90 | 2.77x |
| 4K stage 1 down (32640, 16384 → 4096) | 10.35 | 12.62 | 8.69 | 2.99 | 5.63 | 2.24x |
| 1080p stage 1 up (32130, 4096 → 16384) | 10.37 | 10.59 | 6.27 | 3.09 | 3.86 | 2.75x |
| 1080p stage 1 down (32130, 16384 → 4096) | 10.32 | 12.58 | 8.61 | 2.93 | 5.56 | 2.26x |
| stage 2 chunk up (16384, 4096 → 16384) | 5.30 | 5.43 | 3.21 | 1.47 | 1.87 | 2.91x |
| stage 2 chunk down (16384, 16384 → 4096) | 5.16 | 6.37 | 4.37 | 1.34 | 2.72 | 2.34x |
| 4K stage 2 whole up (130560, 4096 → 16384) | 46.60 | 47.00 | 26.44 | 12.11 | 14.77 | 3.18x |

The down projection's NVFP4 linear spends about half its time in the two
activation passes (amax + quantize over the 16384-wide GELU input); a fused
amax in the up GEMM's epilogue is the next cut. An unaligned token count
(32130) runs unpadded.

**Generations** (RTX PRO 6000, run `51adbb3-09271342`, `runpod-matrix.sh
precision` with `FV_PRECISION_ARM=ltx-nvfp4`, `FV_LPIPS=1`): Sol stage 2,
resident, warm process, every run encoding its prompt (no text cache), one
prompt, seed 1024. bf16 = `ltx2/ltx25_distill_sol`, NVFP4 =
`ltx2/ltx25_distill_sol_nvfp4`. Seconds; peak = pool high-water mark
(allocated / reserved GiB), smi = device-wide peak.

| workload | arm | stage 1 | stage 2 | denoise | total (e2e) | peak GiB | smi GiB |
|---|---|---|---|---|---|---|---|
| 4k5s | bf16 | 45.75 | 79.92 | 125.67 | 209.86 | 55.23 / 56.66 | 57.3 |
| 4k5s | NVFP4 | 40.10 | 70.61 | 110.71 | 186.48 | 46.60 / 47.97 | 48.7 |
| 1080p20s | bf16 | 43.21 | 75.17 | 118.38 | 194.98 | 54.94 / 56.19 | 56.9 |
| 1080p20s | NVFP4 | 38.24 | 67.24 | 105.48 | 183.34 | 46.32 / 47.53 | 48.2 |

| workload | denoise speedup | total speedup | LPIPS mean / max | PSNR mean / min dB | sharpness ratio | jitter ratio | gate (`gate-policy.toml`, lossy) |
|---|---|---|---|---|---|---|---|
| 4k5s | 1.135x | 1.125x | 0.193 / 0.255 | 19.07 / 17.50 | **1.098** (limit 0.95-1.08) | 1.016 | **fail**: `quantitative_quality` (sharpness); performance pass |
| 1080p20s | 1.122x | 1.063x | 0.254 / 0.303 | 18.27 / 17.13 | 0.994 | 1.064 | **pass** (visual artifact deferred) |

The NVFP4 cells record 3072 `nvfp4_cublaslt` linear calls per request
(`benchmark.json` `quantized_linears`). The totals include
60-68 s of streamed Gemma encoding in both arms (noisy between runs). The
NVFP4 FFN saves 8.6 GiB of DiT weights, which shows as the ~8.6 GiB lower
peak in every phase. PSNR and LPIPS are telemetry in the policy (the
bf16 chaos floor); the 4K candidate fails only the hard sharpness range:
its frames are ~10% sharper than bf16's (FP4 noise reading as detail),
so the profile stays default off.

## DiffVAE (diffusion video decoder)

Opt-in replacement for the **video** conv VAE decoder
(`LTX2VideoDiffusionDecoderModel`). Encoding and audio stay on `vae/` /
`audio_vae/` + BWE. Driven like Diffusers’ `LTX2VideoDiffusionDecodePipeline`:
hand **de-normalized** latents; the decoder draws its own pixel noise.

1. Drop the DiT (VRAM) → load `diffusion_decoder/`.
2. Stages 1–4: deterministic neighborhood-attention upsample → context volume
   (PixelShuffle strides `(1,2,2)`, `(2,1,1)`, `(2,2,2)`, `(2,2,2)`).
3. Stage 5: patchify (`patch_size=4`), `x_t ~ N(0,1)`, **one** x0 step at
   `t=1.0` (`decoder_num_inference_steps=1`) → RGB.
4. Neighborhood attention: gather + SDPA with NATTEN’s inward-shifted fixed
   window (no Hub `kernels`/NATTEN). 3D RoPE as in
   `LTX2VideoVaeRotaryPosEmbed3D`.
5. Audio: existing packed-latent → audio VAE → BWE vocoder (unchanged).

Shipped defaults: `stage_channels=(2048,1024,512,512,256)`,
`stage_depths=(4,6,4,2,8)`, stage-5 kernel `(11,11,11)`, `head_dim=64`.
Tiling and multi-step stage-5 are deferred; first green is **untiled**
768×512×121 single-stage.

---

## Config deltas vs LTX-2.0

### DiT (`transformer/config.json`)

| key | 2.0 | 2.5 distilled |
|---|---|---|
| `ff_bias` | true | **false** (video FF only; `audio_ff_bias` stays true) |
| `gated_attn` / `audio_gated_attn` | false | **true** → `to_gate_logits` Linear → `2·sigmoid` per-head gate |
| `cross_attn_mod` / `audio_cross_attn_mod` | false | **true** → 9-row AdaLN + `prompt_*` tables + global `prompt_adaln` |
| `perturbed_attn` | false | **true** (STG processor; distilled CFG=1 leaves masks unset → no STG) |
| `use_prompt_embeddings` | true | **false** → no DiT `caption_projection*`; connectors emit stream widths |
| `use_keyframes_abs_pos_embedding` | false | **true** (tensor present; unused unless keyframe pipeline) |
| `rope_type` | split | split |
| `num_layers` / widths | 48 / 4096+2048 | same |

Block-0 keys include `attn*.to_gate_logits`, `prompt_scale_shift_table`,
`audio_prompt_scale_shift_table`; video `ff.net.*.weight` has **no bias**.

### Connectors

`per_modality_projections=true`, `proj_bias=true`, gated attn on both
connectors, separate `video_text_proj_in` / `audio_text_proj_in`
(`text_proj_in_factor=49`).

### Gemma 4 (`gemma4_unified`, `gemma_version=gemma4-12b-ltx-v1`)

Text config highlights vs Gemma-3-12B:

- `vocab_size` **262144** (was 262208)
- `hidden_size` 3840, 48 layers, sliding 1024 / full every 6th — same cadence
- Full-attention layers: `global_head_dim=512`, `rope_parameters.full_attention`
  `rope_type=proportional`, `partial_rotary_factor=0.25`,
  `num_global_key_value_heads=1`
- `attention_k_eq_v=true` (full-attention layers only)
- HF keys under `model.language_model.*` (or Comfy-flat `model.layers.*`)
- The layer is not Gemma 3's. Norms use `x·w`, the softmax scale is 1,
  V is normed without a weight, each layer ends with `*= layer_scalar`, and
  full-attention RoPE is `proportional`. See docs/oracle.md, "LTX-2.5 text
  path", for the parity numbers.

Encode-only for unified; vision/audio embedders exist but T2AV text path does
not need the towers.

### Video VAE (conv)

Not isomorphic to 2.0: `decoder_block_out_channels=[256,512,512,1024]`,
`decoder_layers_per_block=[4,6,4,2,2]`, `upsample_residual=false`,
`upsample_factor=[2,2,1,2]`, `upsample_type` mix, decoder padding `zeros`.

### Audio VAE + vocoder

- Audio VAE: `latent_channels=8`, `mel_bins=64`, `num_res_blocks=2`,
  `base_channels=128` (2.0 was latent 2 / mel 64 / fewer channels)
- Vocoder: **`LTX2VocoderWithBWE`**, SnakeBeta + antialias, **48 kHz** out

---

## Image conditioning: I2V and keyframes (serve E5 / E9)

Reference: `ltx_pipelines.distilled.DistilledPipeline(images=[(path, frame_idx,
strength, crf)])` at Lightricks/LTX-2 `fd4ded7` (the `--image PATH FRAME_IDX
STRENGTH [CRF]` of `gpu_infer.py`). Code: `ltx2/i2v_encode.rs`,
`ltx2/vae_encoder.rs`, `Ltx2Transformer::set_video_conditioning`, the `_cond`
samplers in `ltx2/pipeline.rs`; `Ltx2Request::images` (and the older
`image_path` = one image at frame 0).

| step | reference | ours |
|---|---|---|
| decode | PIL, EXIF rotations 3/6/8, RGBA→RGB, ICC → sRGB | `image` crate, same rotations and alpha drop; no ICC transform |
| CRF | one libx264 frame, `veryfast`, yuv420p, even crop, CRF 18 from 2.4 on (33 before), decoded back (PyAV) | same through the `ffmpeg` CLI (`FASTVIDEO_FFMPEG`), `-sws_flags bilinear` |
| resize | `resize_and_center_crop`: bilinear, `align_corners=False`, no antialias, cover then center crop; `x/127.5 − 1` → bf16 | same arithmetic on the host |
| encode | `VideoEncoder` in bf16 | `vae_encoder.rs` in f32 over the bf16 weights (Diffusers `encoder.*` keys; zero spatial padding; `SpaceToDepthDownsample` with its group-mean skip) |
| frame 0 | `VideoConditionByLatentIndex`: latent frame 0's tokens get the clean latent and mask `1 − strength` | same rows |
| frame k > 0 | `VideoConditionByKeyframeIndex`: tokens *appended*, RoPE time `[k, k+1)/fps`, no attention mask | appended block, `Ltx2RopeTables::with_keyframes` |
| noise | one draw over all tokens; `lerp(clean, lerp(init, ε, scale), mask)` | `initial_noise_with`, `StageConditioning::apply_initial` |
| DiT | per-token `timesteps = mask·σ`: video AdaLN, text-Q AdaLN/gate, a↔v scale/shift, head per token; a↔v gates and prompt AdaLN at the scalar σ | `forward_segmented`: runs of rows at one timestep, the modulation applied per run |
| x0 | `x − mask·σ·v`, then `x0·mask + clean·(1 − mask)` (`post_process_latent`) | `StageConditioning::{x0, post}` |
| samplers | stage 1 ancestral: blend x0, step, blend after the noise; stage 2 Euler: blend x0 | `ancestral_update_cond`, `euler_update_cond` |
| between stages | `clear_conditioning` drops the appended tokens; stage 2 re-encodes every image at full size | same |

Both stages condition (stage 1 at half size, stage 2 at full size). The
conditioned forwards run without FBCache, the midpoint prune and the stage-1
cache. `fv-gpucheck ltx2 gen --image PATH` (frame 0) and `--cond-image
PATH@FRAME[@STRENGTH[@CRF]]` (`FRAME` may be `last`). Serve maps `image_uri`
/ `image_url` to frame 0 and `last_frame_uri` / `last_image_url` to frame
`num_frames − 1`, strength 1, the checkpoint's CRF.

Oracle targets `ltx25-i2v` and `ltx25-kf` (scripts/gpu/upstream/oracle.sh,
runpod-matrix.sh `oracle`): 768×512×121, dense stage 2, the TI2V beach
fixture at frame 0 and (kf) its 1.35× zoom at frame 120. The reference also
dumps `s{n}_cond{i}_{pixels,latent}`; ours injects the reference's latents
(`FASTVIDEO_INJECT_COND=0` keeps ours) so the denoiser diff is the
conditioning's alone. Results: docs/oracle.md, "LTX-2.5 image conditioning".

## Audio-to-video (avatar P0, 2026-09-29)

Upstream ships audio-to-video as `A2VidPipelineTwoStage`
(`ltx_pipelines/a2vid_two_stage.py`, Lightricks/LTX-2 `fd4ded7`): the *dev*
transformer with CFG/STG/modality guidance at stage 1 and the distilled LoRA
at stage 2. That needs `transformer_full/` of `Lightricks/LTX-2.5-Diffusers`
(4 shards, 37 976 221 088 B ≈ 38.0 GB, LTX-2 community license, gated
auto-approval), which is not on the volumes, plus a guided sampler we do not
have. Not downloaded (large downloads need the owner's approval; any new tree
goes on both volumes, CLAUDE.md). What we serve instead is the same audio
mechanism on the distilled pipeline we already run: `DistilledPipeline`
with `a2vid_two_stage.py`'s audio handling put in line for line
(`scripts/gpu/upstream/ltx25_a2v.py` is that reference, built from upstream
blocks). Code: `ltx2/a2v.rs`, `Ltx2Pipeline::encode_driving_audio`,
`StageConditioning::with_frozen_audio`, `Ltx2Transformer::set_audio_frozen`,
`Ltx2Request::audio`.

| step | reference | ours |
|---|---|---|
| decode | `decode_audio_from_file(path, 0, num_frames / fps)`: PyAV, the file's rate and layout, float, `round(d·rate)` samples | ffprobe + ffmpeg `f32le` at the file's rate, same cut (Python rounding); a non-stereo file goes through `-ac 2` (the encoder takes 2 channels; upstream fails on mono) |
| encode | `AudioConditioner` → `encode_audio`: torchaudio resample to 16 kHz, slaney log-mel (1024 / 160 / 64), causal encoder, bf16 | `AudioEncoder::encode_waveform` (the refiner's encoder), f32 over the bf16 weights |
| cut | `[:, :, :AudioLatentShape.from_duration(num_frames / fps).frames]` (`round(d·25)`) | `conform_audio_time` to the clip's audio tokens; a shorter audio is refused |
| stage 1 / 2 | `ModalitySpec(frozen=True, noise_scale=0, initial_latent=…)`: the noiser still draws audio noise, `denoise_mask` 0, `Modality.sigma` 0 | the noise stream draws as for T2V, the audio stays the clean latent after every update (`StageConditioning::audio_after`) |
| DiT | audio per-token timesteps 0 (audio AdaLN, audio a↔v scale/shift, audio head); audio sigma 0 for the audio prompt AdaLN and the *video's* a→v gate (the cross modality's sigma); the audio's v→a gate and everything video keep the video sigma | `modulations()` with the freeze flag, both forward paths |
| image | the I2V / keyframe conditionings as `DistilledPipeline` | unchanged (combines with the frozen audio) |
| output | the decoded input waveform ("to preserve fidelity"), no vocoder | `DecodeOut::audio_passthrough`: `audio.wav` / the sink get the input PCM at its own rate |

`fv-gpucheck ltx2 gen --two-stage --audio FILE [--image …]`. Serve:
`Task::A2V` on the 2.5 models (see docs/serve/fal-parity.md,
docs/serve/e2e/ltx.md). Oracle: docs/oracle.md, "LTX-2.5 audio-to-video".

## Out of scope (this milestone)

DiffVAE tiling / multi-step stage-5 / two-stage+DiffVAE combo, duration head,
prompt enhancer, generated keyframes (`VideoGeneratedKeyframeSlots`), dev DiT
+ CFG/STG/modality guidance, Comfy int8/nvfp4, official 1536×1024 canvas.

---

## Verification

Host: config constructors, gated-attn unit test, ancestral step math vs
`EulerAncestralDiffusionStep`, key-layout detection; vocoder SnakeBeta + BWE
forward; latent upsampler geometry; DiffVAE NA window + 1-step shape.

GPU: `FV_LTX2_VERSION=2.5 scripts/gpu/validate.sh run ltx2-gen` (single-stage)
and `FV_LTX2_TWO_STAGE=1` for two-stage; `FV_LTX2_DIFF_VAE=1` for DiffVAE.
Stage-1 BWE clip
`artifacts/clips/20260921T213121Z-ltx2-gen/ltx25-bwe.mp4`. Two-stage remote
pass: `artifacts/clips/20260921T220546Z-ltx2-gen/ltx25-two-stage.mp4`
(RTX PRO 6000 WS `51970730`, 768×512×121, 11 ancestral steps). DiffVAE remote
pending. Oracle taps vs diffusers/`ltx-pipelines` remain optional.
