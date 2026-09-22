# Z-Image — port specification

FastVideo family `zimage`: text-to-image DiT (`ZImageTransformer2DModel`) with
Qwen3 text encoder and Diffusers `AutoencoderKL`. Hub
`Tongyi-MAI/Z-Image-Turbo` (Turbo distilled, 8-step, CFG=0).

Sources (read 2026-09-22): FastVideo `configs/pipelines/zimage.py`,
`configs/models/dits/zimage.py`, `examples/inference/basic/basic_zimage.py`,
`docs/cookbook/z-image.md`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `zimage_turbo` | `Tongyi-MAI/Z-Image-Turbo` | 8-step flow, CFG 0 |

Default canvas **1024×1024**, `num_frames=1`, `flow_shift=3.0`,
`max_sequence_length=512`.

---

## DiT (`ZImageTransformer2DModel`)

| piece | value |
|---|---|
| `in/out_channels` | **16** / **16** |
| `dim` / heads / layers | **3840** / **30** / **30** |
| refiner layers | **2** |
| `cap_feat_dim` (text) | **2560** (Qwen3 hidden[-2]) |
| RoPE `axes_dims` | **(32, 48, 48)** |
| `rope_theta` / `t_scale` | **256** / **1000** |
| patch | spatial **2**, temporal **1** |

Text: Qwen3 (+ thinking chat template). VAE: SD-style AutoencoderKL (16ch).
Schedule: FlowMatch Euler, `sigma_min=0`, reference discrete timesteps.

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::zimage` config / schedule | landed |
| Registry + CLI T2I path | landed |
| cudarc DiT (tiny zeros + load hook) | landed |
| Generate scaffold (noise → DiT → VAE → PNG) | landed |
| Qwen3 text encode | external blocker (zeros tokens until encoder weights) |
| AutoencoderKL decode path (structured upsample + scale) | landed |
| Full 30-layer DiT / VAE weight key parity | external blocker (needs Hub weights + key map) |

Host path: no refuse stubs for denoise/VAE decode when graphs are present.
