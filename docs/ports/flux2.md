# FLUX.2 — port specification

FastVideo family `flux2`: `Flux2Transformer2DModel` (dual + single stream) with
Mistral/LLM text path and AutoencoderKL. Hubs: Klein 4B/9B and FLUX.2-dev.

Sources (read 2026-09-22): Diffusers `Flux2Pipeline`, Hub configs for
`FLUX.2-klein-4B`, Diffusers defaults for FLUX.2-dev.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `flux2_klein_4b` | `black-forest-labs/FLUX.2-klein-4B` | distilled, no guidance embeds |
| `flux2_klein_9b` | `black-forest-labs/FLUX.2-klein-9B` | distilled |
| `flux2_dev` | `black-forest-labs/FLUX.2-dev` | full, guidance embeds |

Default canvas **1024×1024**. Packed latents: VAE 16-ch → DiT **128**-ch.

---

## DiT dims

| preset | dual / single | heads | joint_dim | guidance |
|---|---|---|---|---|
| klein_4b | 5 / 20 | 24 | 7680 | false |
| klein_9b | 8 / 24 | 32 | 10240 | false |
| dev | 8 / 48 | 48 | 15360 | true |

`in_channels=128`, `attention_head_dim=128`, `axes_dims_rope=(32,32,32,32)`,
`rope_theta=2000`.

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::flux2` | landed |
| Registry + CLI T2I path | landed |
| cudarc DiT (tiny zeros + load hook) | landed |
| Generate scaffold | landed |
| Text encode / full weight parity | Qwen3/Mistral3 `CudaTensor` encoders + chat wrap/tokenize |
| Pack / empirical μ / 4-axis ids | landed (`flux2::family`) |
| Weight-key maps + VAE config | landed (`flux2::weights`, `Flux2VaeConfig`, 32-ch FastVideo VAE) |
| 2D VAE decode | landed (`fastvideo-cudarc::flux2::vae`) |
| DiT + device RoPE | landed (`Flux2Transformer2D`, `apply_rotary_bshd`) |
| Generate-gap ranking | `docs/flux2-generate-gap.md` |
