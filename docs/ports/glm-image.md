# GLM-Image — port specification

FastVideo family `glm_image`: `GlmImageTransformer2DModel` with GLM text/prior
path and AutoencoderKL. Hub `zai-org/GLM-Image`.

Sources (read 2026-09-22): Hub `transformer/config.json`, FastVideo issue #1030.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `glm_image` | `zai-org/GLM-Image` | T2I (+ I2I later) |

Default canvas **1024×1024**, FlowMatch Euler, steps **30**, guidance ≈ **3.5**.

---

## DiT (`GlmImageTransformer2DModel`)

| piece | value |
|---|---|
| `in/out_channels` | **16** / **16** |
| heads / head_dim / layers | **32** / **128** / **30** |
| `text_embed_dim` | **1472** |
| `condition_dim` | **256** |
| `time_embed_dim` | **512** |
| patch | **2** |
| prior VQ codebook | **16384** |

VAE: AutoencoderKL 16-ch (SD3-style).

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::glm_image` | landed |
| Registry + CLI T2I path | landed |
| cudarc DiT (tiny zeros + load hook) | landed |
| Generate scaffold | landed |
| GLM AR encoder / full weight parity | ByT5 glyph (`text_encoder/`) + AR text LM (`vision_language_encoder/`) when present; prior VQ `generate()` deferred |

### Text / AR notes (Diffusers `GlmImagePipeline`)

Hub layout (`zai-org/GLM-Image`):

| dir | class | role |
|---|---|---|
| `text_encoder/` | `T5EncoderModel` (ByT5-small, d_model **1472**) | glyph / prompt embeds |
| `vision_language_encoder/` | `GlmImageForConditionalGeneration` | AR text LM (hidden **4096**, 40 layers) → prior tokens |
| `tokenizer/` | `ByT5Tokenizer` | byte-level glyph ids |

This tree runs real ByT5 + AR **text** hidden-state graphs when those dirs exist.
CLIP-only is a legacy fallback, not treated as sufficient when ByT5/AR packs are
present. Full AR prior VQ token `generate()` + upsample remains external.
