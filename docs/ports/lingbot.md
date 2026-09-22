# LingBot-Video — port specification

FastVideo family `lingbot`: Robbyant LingBot-Video Dense (1.3B) / MoE T2V with
Qwen3-VL text, Wan VAE, FlowMatch (`flow_shift=3`).

Sources (read 2026-09-22): FastVideo `configs/pipelines/lingbot_video.py` +
`configs/models/dits/lingbot_video.py`; Hub `robbyant/lingbot-video-dense-1.3b`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `lingbot_dense_1_3b` | `robbyant/lingbot-video-dense-1.3b` | Dense T2V |
| `lingbot_moe_30b` | `robbyant/lingbot-video-moe-30b-a3b` | MoE (scaffold) |

Default canvas ~480×832, flow_shift **3**, guidance ~3.

---

## DiT (Dense 1.3B)

| piece | size |
|---|---|
| `hidden_size` | **2048** |
| `num_attention_heads` | **16** |
| `depth` | **24** |
| `intermediate_size` | **6144** |
| `in_channels` / `out_channels` | **16** / **16** |
| `patch_size` | **(1, 2, 2)** |
| `text_dim` | **2560** (Qwen3-VL) |
| `freq_dim` | **256** |
| `rope_theta` | **256** |
| `axes_dims` | **(32, 48, 48)** |

MoE pack adds `num_experts` / routed FFN (not required for Dense tiny forward).

---

## Text / VAE / schedule

| piece | notes |
|---|---|
| Qwen3-VL text | chat template + crop **140** (`PROMPT_CROP_START`) |
| AutoencoderKLWan | Reuse Wan VAE decode |
| Schedule | FlowMatch Euler, `flow_shift=3` |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::lingbot` config / rope / schedule | landed |
| Registry + CLI path | landed |
| cudarc DiT + tiny forward (Dense) | landed |
| Qwen3-VL text | zeros fallback (template/crop constants documented) |
| Wan VAE decode + PNG | landed |
| Denoise loop | landed |
| MoE routed FFN | pending |
