# LongCat-Video — port specification

FastVideo family `longcat`: Meituan LongCat T2V DiT with optional **Block Sparse
Attention (BSA)**, UMT5 text, Wan VAE, FlowMatch Euler.

Sources (read 2026-09-22): Meituan `LongCatVideoTransformer3DModel` /
`LongCatVideoPipeline`; FastVideo `configs/pipelines/longcat.py` +
`configs/models/dits/longcat.py`; Hub `FastVideo/LongCat-Video-T2V-Diffusers`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `longcat_t2v_480p` | `FastVideo/LongCat-Video-T2V-Diffusers` | dense attn default |
| `longcat_t2v_720p` | same pack / refine LoRA path | BSA on for 720p refine |

Default canvas ~480×832, 93 frames, fps 15.

---

## DiT (`LongCatVideoTransformer3DModel`)

| piece | size |
|---|---|
| `hidden_size` | **4096** |
| `depth` | **48** |
| `num_heads` | **32** (head_dim 128) |
| `in_channels` / `out_channels` | **16** / **16** |
| `patch_size` | **(1, 2, 2)** |
| `caption_channels` | **4096** (UMT5) |
| `adaln_tembed_dim` | **512** |
| `mlp_ratio` | **4** (SwiGLU) |
| `enable_bsa` | false @480p; true @720p refine |

Block: AdaLN-6 → self-attn (optional BSA) → LayerNorm cross-attn → SwiGLU FFN.
BSA params default `sparsity=0.9375`, `chunk_3d_shape_{q,k}=[4,4,4]`.

### BSA algorithm (host reference)

Matches Meituan `flash_attn_bsa_3d`: rearrange THW → 3D chunks → mean-pool Q/K →
top-`(1-sparsity)` KV blocks per query block → sparse SDPA. Falls back to dense
SDPA when latent dims are not divisible by the chunk shape (tiny graphs).

---

## Text / VAE / schedule

| piece | notes |
|---|---|
| UMT5EncoderModel | Diffusers `text_encoder/`; pad embeds to **512** |
| AutoencoderKLWan | Reuse Wan VAE (`vae/`) |
| Schedule | FlowMatch Euler (`flow_shift` typically unset / 1.0) |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::longcat` config / schedule | landed |
| Registry + CLI path | landed |
| cudarc DiT + tiny forward (dense attn) | landed |
| BSA sparse path (`enable_bsa` / 720p) | landed |
| UMT5 text | encode when weights present; zeros dry-run without `text_encoder/` |
| Wan VAE decode + PNG | landed |
| Denoise loop | landed (FlowMatch Euler) |
