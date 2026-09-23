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
| `lingbot_moe_30b` | `robbyant/lingbot-video-moe-30b-a3b` | MoE 128 experts / top-8 |

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

### MoE 30B-A3B

| piece | size |
|---|---|
| `num_experts` | **128** |
| `num_experts_per_tok` | **8** |
| `moe_intermediate_size` | **512** |
| `score_func` | sigmoid |
| `norm_topk_prob` | true |

Routed FFN: gate logits → sigmoid → top-k → optional renorm → weighted SwiGLU
experts × `routed_scaling_factor`.

---

## Text / VAE / schedule

| piece | notes |
|---|---|
| Qwen3-VL text | FastVideo chat template + crop **140** (`PROMPT_CROP_START`); tap final layer |
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
| Qwen3-VL text encode (template + crop 140) | landed |
| Wan VAE decode + PNG | landed |
| Denoise loop | landed |
| MoE routed FFN | landed |
