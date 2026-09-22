# HunyuanVideo 1.5 — port specification

FastVideo family `hunyuan15`: MMDiT double-stream DiT, dual text
(Qwen2.5-VL + ByT5 glyphs), causal video VAE (f16t4d32), optional I2V and
1080p MeanFlow SR chain.

Sources (read 2026-09-22): FastVideo `main`
`configs/pipelines/hunyuan15.py`, `configs/models/dits/hunyuanvideo15.py`,
`configs/models/vaes/hunyuan15vae.py`, `models/dits/hunyuanvideo15.py`;
Hub packs under `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*`.

---

## Hub ids (registry)

| preset | Hub id | workload | notes |
|---|---|---|---|
| `hy15_480p_t2v` | `…-480p_t2v` | T2V | flow_shift **5** |
| `hy15_480p_i2v_distilled` | `…-480p_i2v_step_distilled` | I2V | flow_shift **7** |
| `hy15_720p_t2v` | `…-720p_t2v` | T2V | flow_shift **9** |
| `hy15_720p_i2v_distilled` | `…-720p_i2v_distilled` | I2V | flow_shift **7** |
| `hy15_1080p_sr` | `weizhou03/HunyuanVideo-1.5-Diffusers-1080p` (+ `-2SR`) | T2V/SR | MeanFlow + upsamplers |

Prefix all community ids with `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-`.

---

## DiT (`HunyuanVideo15Transformer3DModel`)

Diffusers / FastVideo custom key layout (`Hunyuan15.*`):

| piece | size |
|---|---|
| `hidden_size` | `16 × 128 = 2048` |
| `num_layers` | **54** double-stream blocks |
| `num_refiner_layers` | **2** (Qwen token refiner) |
| `in_channels` / `out_channels` | **65** / **32** (I2V packs cond in channel 0..) |
| `patch_size` / `patch_size_t` | **1** / **1** |
| `rope_theta` | **256** |
| `rope_axes_dim` | **(16, 56, 56)** on head_dim 128 |
| `text_embed_dim` | **3584** (Qwen2.5-VL) |
| `text_embed_2_dim` | **1472** (ByT5) |
| `image_embed_dim` | **1152** (I2V CLIP-style) |
| `mlp_ratio` | **4** |
| QK norm | RMSNorm per head (`eps=1e-6`) |

**Forward sketch**

1. Patch-embed latents → `img` tokens; optional image projection for I2V.
2. `txt_in`: SingleTokenRefiner on Qwen hidden states (`hidden_states[-3]`,
   crop first **108** template tokens).
3. `txt_in_2`: ByT5 glyph projection → concat onto text stream (+ cond type embeds).
4. `time_in` → shared `vec`; optional MeanFlow `timestep_r` for SR.
5. For each of 54 `MMDoubleStreamBlock`s: separate AdaLN (factor 6) for img/txt,
   joint attention over `[img|txt]` QKV, gated residuals + MLP.
6. `final_layer` AdaLN + linear → unpatchify to `out_channels`.

Weight remap lives in FastVideo `HunyuanVideo15ArchConfig.param_names_mapping`
(HF `transformer_blocks.*` ↔ custom `double_blocks.*`).

---

## Text

| encoder | role | precision (FV default) |
|---|---|---|
| Qwen2.5-VL | primary caption embeds; system template **108** tokens cropped | bf16 |
| ByT5 | glyph / quoted-text branch (`extract_glyph_texts`) | fp32 |

Max lengths: Qwen `1000+108`, ByT5 `256`.

---

## VAE

`Hunyuan15VAE`: latent **32** ch, spatial **16×**, temporal **4×**,
`scaling_factor ≈ 1.03682`, block outs `(128,256,512,1024,1024)`. Decode-only
at serve time (`load_encoder=False` in FV pipeline config).

---

## Denoise

Flow-match Euler with pipeline `flow_shift` (5 / 7 / 9). Distilled I2V packs
are short-step CFG=1 style; base T2V uses CFG as in the Hub recipe.

1080p SR: second stage with `flow_shift_sr=2` + MeanFlow `timestep_r` +
720p/1080p upsamplers (`SRTo720p` / `SRTo1080p`).

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::hunyuan15` config / rope / schedule / text helpers | landed |
| Registry `ModelFamily::Hunyuan15` + Hub ids | landed |
| cudarc DiT double-block + tiny forward | landed |
| Qwen2.5-VL mid-layer tap (`llm::qwen25_vl_7b_text`, crop 108) | landed (needs pack weights) |
| ByT5 glyph encode | landed via `Umt5Config::byt5_small` + byte tokenize |
| VAE decode (causal + DCAE upsample + mid attn) | landed (tiny host test; full weights unvalidated) |
| Denoise loop (FlowMatch Euler) → PNG frames | landed |
| Qwen `apply_chat_template` | approximate (system+user concat; refine later) |
| Oracle / gpucheck | not started |

Reuse: [`crate::wan::{tensor,nn,attn}`](../../crates/fastvideo-cudarc/src/wan/),
[`crate::llm`](../../crates/fastvideo-cudarc/src/llm.rs) (Qwen2.5-VL-7B config),
FlowMatch Euler scheduler.
