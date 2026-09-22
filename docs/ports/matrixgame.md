# Matrix-Game — port specification

FastVideo family `matrixgame`: interactive game world I2V on Wan DiTs with
keyboard/mouse (or camera) action modules.

Sources (read 2026-09-22): FastVideo `configs/pipelines/matrixgame2.py` /
`matrixgame3.py`, `configs/models/dits/matrixgame2.py` / `matrixgame3.py`,
`examples/inference/basic/basic_matrixgame2.py` / `basic_matrixgame3.py`.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `mg2_base_distilled` | `FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers` | universal, kb_dim=4 |
| `mg2_gta_distilled` | `FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers` | kb_dim=2 |
| `mg2_templerun_distilled` | `FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers` | kb_dim=7 |
| `mg2_base` | `FastVideo/Matrix-Game-2.0-Base-Diffusers` | non-distilled |
| `mg3_base_distilled` | `FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers` | 720×1280, 3-step |

Also recognized (fuzzy): GTA/TempleRun non-distilled Diffusers ids, Zelda
community checkpoints (`mignonjia/mg_*`).

---

## DiT

### Matrix-Game 2.0 (Wan I2V + action blocks)

| piece | value |
|---|---|
| Base | WanVideoArch (14B defaults: 40×128, 40 layers, 16ch) |
| `image_dim` | **1280** (CLIP vision) |
| `text_dim` | **0** (action-driven; empty prompt OK) |
| Action | keyboard/mouse injectors on first 15 blocks |
| Causal DMD | distilled: steps `[1000,666,333]`, `num_frames_per_block=3` |
| Canvas | **352×640**, long rollouts (~597 frames in examples) |
| `flow_shift` | **5.0** |

### Matrix-Game 3.0 (Wan2.2 TI2V-5B + action / cam)

| piece | value |
|---|---|
| `in/out_channels` | **48** / **48** |
| heads × dim / layers / ffn | **24** × **128** / **30** / **14336** |
| Canvas | **720×1280**, 57 frames, 3 distilled steps |
| Action | keyboard_dim_in=6, mouse, memory, camera patch embed |

VAE: Wan (MG2) / Wan2.2 light VAE (MG3).

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::matrixgame` config / action dims | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse `WanTransformer3D`) + tiny forward | landed |
| Generate scaffold (UMT5 optional → Wan denoise → VAE) | landed |
| Full action-module injectors / causal KV | deferred (metadata + zero action tensors) |

Host path reuses Wan encode/denoise/VAE; no refuse stubs for those cores.
