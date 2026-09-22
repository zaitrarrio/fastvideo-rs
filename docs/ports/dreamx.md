# DreamX-World — port specification

FastVideo family `dreamx_world`: Wan2.2 TI2V-5B world model with camera PRoPE
control adapter and action sequences (`w,d,w` + speeds).

Sources (read 2026-09-22): FastVideo `configs/pipelines/dreamx_world.py`,
`configs/models/dits/dreamx_world.py`, `examples/inference/basic/basic_dreamx_world.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `dreamx_5b_cam` | `FastVideo/DreamX-World-5B-Cam-Diffusers` | PRoPE cam, 480×832 |
| `dreamx_5b_ar` | `FastVideo/DreamX-World-5B-Diffusers` | causal AR / DMD |

---

## DiT (Wan TI2V-5B + cam adapter)

| piece | Cam / AR |
|---|---|
| heads × dim / layers / ffn | **24** × **128** / **30** / **14336** |
| `in/out_channels` | **48** / **48** |
| `cam_method` | `prope` |
| `add_control_adapter` | true |
| Cam AR extras | `local_attn_size=12`, `sink_size=3`, `attn_compress=4` |
| Text | UMT5-XXL (4096) |
| VAE | Wan2.2 48-ch (Lucy-Edit style) |
| Schedule | flow_shift 3.0 (cam) / 5.0 + DMD (AR) |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::dreamx` config | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse Wan TI2V-5B) + tiny forward | landed |
| Generate scaffold | landed |
| Action → viewmats / K (PRoPE camera pack) | landed |
| PRoPE self-attn / control-adapter fuse | external blocker (needs `add_control_adapter` weights in DiT) |

Host path reuses Wan 5B encode/denoise/VAE graphs.
