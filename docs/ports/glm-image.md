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
| GLM AR encoder / full weight parity | external blocker |
