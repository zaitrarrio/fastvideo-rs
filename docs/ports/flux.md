# FLUX.1 — port specification

FastVideo family `flux`: dual/single-stream DiT (`FluxTransformer2DModel`) with
CLIP-L + T5-XXL text and AutoencoderKL. Hub `black-forest-labs/FLUX.1-dev`.

Sources (read 2026-09-22): FastVideo support matrix, Diffusers `FluxPipeline`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `flux1_dev` | `black-forest-labs/FLUX.1-dev` | T2I, guidance embeds |

Default canvas **1024×1024**, FlowMatch Euler, guidance ≈ **3.5**, steps **28**.
Latents packed: VAE 16-ch → DiT **64**-ch (2×2 pack).

---

## DiT (`FluxTransformer2DModel`)

| piece | value |
|---|---|
| `in_channels` (packed) | **64** |
| dual / single layers | **19** / **38** |
| heads / head_dim | **24** / **128** |
| `joint_attention_dim` | **4096** |
| `pooled_projection_dim` | **768** |
| RoPE axes | **(16, 56, 56)** |
| guidance_embeds | **true** |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::flux` | landed |
| Registry + CLI T2I path | landed |
| cudarc DiT (tiny zeros + load) | landed (`Flux1Transformer2D`) |
| Generate path | landed (`Flux1Pipeline` + `av_generate`) |
| CLIP+T5 encode / full weight parity | CLIP-L + T5-XXL `CudaTensor` encoders + tokenize helpers |
| Pack / RoPE ids / shift | landed (`flux::family`) |
| Weight-key maps + `transformer/config.json` | landed (`flux::weights`, `FLUX1_*_REQUIRED_KEYS`) |
| `flux1_schnell` registry | landed |
| DiT + device RoPE | landed (`Flux1Transformer2D`, `apply_rotary_bshd`) |
| SD3-style VAE decode | landed (shared `AutoencoderKlFlux2` with `Flux2VaeConfig::flux1`) |
