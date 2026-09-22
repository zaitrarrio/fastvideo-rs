# Cosmos Predict2 — port specification

FastVideo family `cosmos`: Diffusers `Cosmos2VideoToWorldPipeline` /
`CosmosTransformer3DModel`. Video2World (image/video → future frames) with
T5 text, Wan VAE (`AutoencoderKLWan`), FlowMatch Euler + EDM-style σ
(`sigma_max=80`, `sigma_min=0.002`, `sigma_data=1.0`).

Sources (read 2026-09-22): Diffusers `transformer_cosmos.py` /
`pipeline_cosmos2_video2world.py`; NVIDIA `cosmos-predict2` Video2World 2B/14B
configs; FastVideo `configs/pipelines/cosmos.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `cosmos2_v2w_2b` | `nvidia/Cosmos-Predict2-2B-Video2World` | 2B DiT |
| `cosmos2_v2w_14b` | `nvidia/Cosmos-Predict2-14B-Video2World` | 14B DiT |

Default canvas ~720p, 16 fps, `state_t=24` latents (~81 frames with Wan VAE).

---

## DiT (`CosmosTransformer3DModel`) — Predict2-2B

From NVIDIA native net + Diffusers defaults (Hub `config.json` gated):

| piece | 2B | 14B |
|---|---|---|
| `num_attention_heads` | **16** | **40** |
| `attention_head_dim` | **128** | **128** |
| hidden | **2048** | **5120** |
| `num_layers` | **28** | **36** |
| `mlp_ratio` | **4** | **4** |
| `in_channels` / `out_channels` | **17** / **16** (cond ch + latents) | same |
| `patch_size` | **(1, 2, 2)** | same |
| `max_size` | **(128, 240, 240)** | same |
| `adaln_lora_dim` | **256** | **256** |
| `text_embed_dim` | **1024** (T5; may project) | same |
| `concat_padding_mask` | true (+1 ch into patch) | same |
| `extra_pos_embed_type` | `learnable` | same |
| `rope_scale` (t,h,w) | ~(1, 3, 3) native | ~(0.83, 2, 2) |

Block: AdaLN-Zero → self-attn (RoPE) → AdaLN-Zero → cross-attn (T5) →
AdaLN-Zero → GELU FF. Final AdaLN + linear unpatch.

---

## Text / VAE / schedule

| piece | notes |
|---|---|
| T5EncoderModel | Diffusers `text_encoder/`; max seq **512** |
| AutoencoderKLWan | Reuse Wan VAE graph (`vae/`) |
| Schedule | FlowMatch + Karras-ish σ; denoise uses `c_in/c_skip/c_out` EDM packing |
| Cond | `condition_mask` + first-frame latents (Video2World) |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::cosmos` config / rope / schedule / T5Config | landed |
| Registry + CLI path | landed |
| cudarc DiT + tiny forward | landed |
| T5 text (`T5Encoder` classic Relu, zeros fallback) | landed |
| Wan VAE decode + PNG frame dump | landed |
| Video2World cond (`condition_mask` + first-frame encode) | landed |

Host path: T5 → EDM Euler DiT → Wan VAE → `frame_*.png`. Optional `--image` packs
first-frame latents into cond channels per Diffusers `prepare_latents`.
