# GEN3C — port specification

FastVideo family `gen3c`: camera-controlled Video2World on a Cosmos-class DiT
with a 3D point-cloud cache (MoGe depth → warp buffers). Diffusers /
FastVideo `Gen3CPipeline`; Hub `FastVideo/GEN3C-Cosmos-7B-Diffusers`
(or convert `nvidia/GEN3C-Cosmos-7B` via FastVideo
`scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py`).

Sources (read 2026-09-22): FastVideo `configs/pipelines/gen3c.py`,
`configs/models/dits/gen3c.py`, `examples/inference/basic/basic_gen3c.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `gen3c_cosmos_7b` | `FastVideo/GEN3C-Cosmos-7B-Diffusers` | Diffusers layout |

Default canvas **704×1280**, **121** frames @ 24 fps, `state_t=16` latents,
`guidance_scale=1.0`, **35** EDM steps. Trajectories: `left` / `right` /
`zoom_in` / `zoom_out` / `clockwise` / `counterclockwise` (MoGe path optional).

---

## DiT (`Gen3C` / VideoExtendGeneralDIT) — 7B

Cosmos-shaped AdaLN-Zero blocks; patch embed takes **warped 3D cache**:

| piece | value |
|---|---|
| `num_attention_heads` × `attention_head_dim` | **32** × **128** → hidden **4096** |
| `num_layers` | **28** |
| `mlp_ratio` / `adaln_lora_dim` | **4** / **256** |
| VAE latent / out | **16** / **16** |
| `frame_buffer_max` | **2** |
| channels per buffer | **32** (16 warped + 16 mask) |
| DiT `in_channels` (pre-pad) | **81** = 16 + 1 cond mask + 64 buffers |
| `patch_size` / `max_size` | **(1,2,2)** / **(128,240,240)** |
| `rope_scale` (t,h,w) | **(2, 1, 1)** |
| `text_embed_dim` | **1024** (T5 Large, pad to max) |
| `extra_pos_embed_type` | `learnable` |
| `concat_padding_mask` | true (+1 into patch) |

Schedule matches Cosmos EDM packing but **`sigma_data=0.5`**,
`sigma_conditional=0.001` (official GEN3C defaults).

---

## Text / VAE / conditioning

| piece | notes |
|---|---|
| T5 Large | Reuse Cosmos T5 host path; pad to max length |
| AutoencoderKLWan | Reuse Wan VAE (`components-source` Cosmos-Predict2) |
| 3D cache | MoGe depth + forward warp → buffer channels (optional on host) |
| Cond | `condition_video_input_mask` + first-frame / cache frames |
| CFG | Legacy: dual pass when `guidance_scale > 1` |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::gen3c` config / schedule / trajectory ids | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse `CosmosTransformer` with GEN3C dims) | landed |
| Generate scaffold (T5 → EDM → Wan VAE; warp buffers packed) | landed |
| 3D cache trajectory + forward warp + buffer pack | landed |
| MoGe depth network weights | external blocker (hook: `depth_path` / synthetic depth) |

Host path: reuse Cosmos encode/denoise/VAE graphs with GEN3C channel layout and
`sigma_data=0.5`. No refuse/zeros stubs for core encode/denoise/VAE.
