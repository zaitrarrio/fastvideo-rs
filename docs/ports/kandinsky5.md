# Kandinsky 5.0 Video — port specification

FastVideo family `kandinsky5`: DiT with dual text (Qwen2.5-VL + CLIP),
HunyuanVideo VAE (16-ch), FlowMatch Euler, optional NABLA sparse attention and
DMD distilled schedules.

Sources (read 2026-09-22): Diffusers `Kandinsky5T2VPipeline` /
`Kandinsky5Transformer3DModel`; Hub `kandinskylab/Kandinsky-5.0-*`; FastVideo
`configs/pipelines/kandinsky5.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `k5_lite_t2v_5s` | `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` | Lite DiT |
| `k5_pro_t2v_5s` | `kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers` | 19B Pro |
| `k5_lite_i2v_5s` | `…-I2V-Lite-…` (when registered) | visual_cond |

flow_shift default **5**; DMD packs use steps `[1000, 750, 500, 250]`.

---

## DiT (`Kandinsky5Transformer3DModel`) — Lite

From Hub `transformer/config.json` (Lite-sft-5s):

| piece | size |
|---|---|
| `model_dim` | **1792** |
| `ff_dim` | **7168** |
| `num_visual_blocks` | **32** |
| `num_text_blocks` | **2** |
| `in_visual_dim` / `out_visual_dim` | **16** / **16** |
| `patch_size` | **(1, 2, 2)** |
| `axes_dims` (RoPE) | **(16, 24, 24)** on head layout |
| `in_text_dim` | **3584** (Qwen2.5-VL) |
| `in_text_dim2` | **768** (CLIP) |
| `time_dim` | **512** |
| `attention_type` | `regular` (Pro may use `nabla`) |
| `visual_cond` | true (I2V channel pack) |

---

## Text

| encoder | role |
|---|---|
| Qwen2.5-VL | primary; chat template crop **129** (`prompt_template_encode_start_idx`) |
| CLIP ViT-L/14 | pooled / sequence **768**, max **77** |

Reuse [`crate::llm::DecoderConfig::qwen25_vl_7b_text`](../../crates/fastvideo-cudarc/src/llm.rs)
with crop 129 (not Hunyuan's 108). CLIP text is new (or Wan CLIP path).

---

## VAE

`AutoencoderKLHunyuanVideo` (classic HunyuanVideo, **16** latent ch) — not the
1.5 VAE. Prefer reusing / adapting the existing Hunyuan VAE graph once ported;
until then load Refuse with a clear message.

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::kandinsky5` config | landed |
| Registry + CLI refuse path | landed |
| DiT / text / VAE / generate | not started |

Reuse Hunyuan15 Qwen mid-layer encode patterns; FlowMatch Euler schedule.
