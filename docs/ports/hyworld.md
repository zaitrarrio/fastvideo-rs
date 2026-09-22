# HY-WorldPlay — port specification

FastVideo family `hyworld`: HunyuanVideo 1.5 bidirectional world model with
action embeddings, camera RoPE, and SigLIP image conditioning.

Sources (read 2026-09-22): FastVideo `configs/pipelines/hyworld.py`,
`configs/models/dits/hyworld.py`, `examples/inference/basic/basic_hyworld.py`,
PR #1027.

---

## Hub ids (registry)

| preset | Hub id | notes |
|---|---|---|
| `hyworld_bidirectional` | `FastVideo/HY-WorldPlay-Bidirectional-Diffusers` | 480p I2V |

---

## DiT

| piece | notes |
|---|---|
| Base | Hunyuan15 double-stream + `action_in` + camera RoPE |
| Image | SigLIP vision encoder |
| VAE | Hunyuan15 VAE (worldplay cache VAE deferred upstream) |
| Pose | strings like `w-31` → trajectory chunks |

---

## Port status (this tree)

| layer | status |
|---|---|
| Spec (this file) | landed |
| `fastvideo-models::hyworld` config | landed |
| Registry + CLI path | landed |
| cudarc DiT (reuse Hunyuan15) + tiny forward | landed |
| Generate scaffold | landed |
| SigLIP / action_in / camera RoPE | deferred (pose string stored) |

Host path reuses Hunyuan15 encode/denoise/VAE.
