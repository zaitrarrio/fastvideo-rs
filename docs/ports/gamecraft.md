# HunyuanGameCraft — port specification

FastVideo family `gamecraft`: HunyuanVideo DiT + CameraNet (Plücker coordinates)
for action-controlled T2V/I2V. 33-channel concat (16 noise + 16 gt + 1 mask).

Sources (read 2026-09-22): FastVideo `configs/pipelines/hunyuangamecraft.py`,
`configs/models/dits/hunyuangamecraft.py`, `examples/inference/basic/basic_gamecraft.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `gamecraft_i2v` | `FastVideo/HunyuanGameCraft-Diffusers` | I2V default; T2V when no image |

---

## DiT / stack

| piece | notes |
|---|---|
| Base | HunyuanVideo double/single stream + CameraNet |
| `in_channels` | **33** (16 + 16 + 1) |
| Text | LLaMA-3-8B (LLaVA) + CLIP |
| VAE | GameCraft Hunyuan VAE (`mid_block_causal_attn`) |
| Schedule | flow_shift **5.0**, CFG **6.0**, ~50 steps |
| Canvas | **704×1280**, 33 frames |
| Actions | `forward` / strafe / rotations → Plücker `camera_states` |

This tree reuses **Hunyuan15** DiT/VAE host graphs with GameCraft channel
layout (same family lineage; GameCraft predates HY 1.5 but shares block shape).

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::gamecraft` config | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse Hunyuan15 with 33-ch) + tiny forward | landed |
| Generate scaffold | landed |
| Plücker trajectory builder (`create_camera_trajectory`) | landed |
| CameraNet CNN fuse into Hunyuan15 DiT | external blocker (needs `camera_net.*` weights) |

Host path reuses Hunyuan15 encode/denoise/VAE; no refuse stubs for those cores.
