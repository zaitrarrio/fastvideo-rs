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
- `attention_k_eq_v=true`
- HF keys under `model.language_model.*` (or Comfy-flat `model.layers.*`)

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

## Out of scope (this milestone)

DiffVAE tiling / multi-step stage-5 / two-stage+DiffVAE combo, duration head,
prompt enhancer, multishot/keyframes, dev DiT + CFG/STG/modality guidance,
Comfy int8/nvfp4, official 1536×1024 canvas.

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
