# LingBot-World — port specification

FastVideo families `lingbotworld` / `lingbotworld2`: Wan2.2 I2V A14B-class
world models with camera injectors (`c2ws` MLP + per-block cam conditioner).

Sources (read 2026-09-22): FastVideo `configs/pipelines/lingbotworld.py` /
`lingbotworld2.py`, `configs/models/dits/lingbotworld.py`,
`examples/inference/basic/basic_lingbotworld_base_cam.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `lingbotworld_base_cam` | `FastVideo/LingBot-World-Base-Cam-Diffusers` | Wan2.2 I2V A14B + cam |
| `lingbotworld2_causal_fast` | `robbyant/lingbot-world-v2-14b-causal-fast` | causal-fast I2V |

---

## DiT

| piece | Base Cam |
|---|---|
| Base | Wan2.2 I2V A14B (40×128, 40 layers, 16→36 I2V concat typical) |
| Cam | `patch_embedding_wancamctrl`, `c2ws_mlp`, per-block cam injector |
| `flow_shift` | **10.0**, `boundary_ratio` **0.947** |
| Canvas | 480p I2V |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::lingbotworld` config | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse Wan I2V A14B) + tiny forward | landed |
| Generate scaffold | landed |
| Cam injector weights / c2ws packing | deferred |

Host path reuses Wan I2V encode/denoise/VAE.
