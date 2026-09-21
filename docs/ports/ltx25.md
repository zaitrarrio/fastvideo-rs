# LTX-2.5 (22B, distilled) stage-1 — port specification

First ship: **distilled stage-1 T2AV only** (Gemma 4 + distilled DiT + conv
video VAE + audio VAE/vocoder), CFG=1, ancestral Euler. No diffusion decoder,
duration head, prompt enhancer, or stage-2 upsampler.

Sources (read 2026-09-21): `Lightricks/LTX-2.5-Diffusers` configs,
`Lightricks/LTX-2` `packages/ltx-pipelines` / `ltx-core`, diffusers
`transformer_ltx2.py` on `main`.

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

Comfy split pack (`Lightricks/LTX-2.5`): one `.safetensors` per component;
projections may live inside the Gemma4 file. Key remap lives in
`ltx-core` `gemma_assets` / connector ops.

Distilled DiT declares `model_version` ≥ 2.5 in safetensors metadata → stage-1
uses **ancestral** Euler (`eta=1`, `s_noise=1`, noise seed = pipeline seed +
10000). Same 8-sigma list as 2.0.

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

Diffusion decoder / DiffVAE, duration head, prompt enhancer, stage-2 spatial
upsampler, multishot/keyframes, dev DiT + CFG/STG/modality guidance, Comfy
int8/nvfp4.

---

## Verification

Host: config constructors, gated-attn unit test, ancestral step math vs
`EulerAncestralDiffusionStep`, key-layout detection.

GPU: `fv-gpucheck ltx2 … --model-version 2.5` after Diffusers weights are
local; oracle taps vs diffusers/`ltx-pipelines` distilled stage-1.
