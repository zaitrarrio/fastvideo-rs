# Registry port status (Phases 4–6)

Snapshot of FastVideo Hub families landed in this tree after Phase 4–6 scaffolding
(2026-09-22). Green = host config + cudarc tiny forward/generate + registry +
CLI `generate_av` path. Weight-parity / text encoders remain external unless noted.

| Phase | Family | Spec | Models | Cudarc | Registry | Generate | Notes |
|---|---|---|---|---|---|---|---|
| 4 | Z-Image | yes | yes | yes | yes | PNG | AutoencoderKL path; Qwen3 external |
| 4 | SD 3.5 | yes | yes | yes | yes | PNG | Triple text external |
| 4 | FLUX.1 | yes | yes | yes | yes | PNG | CLIP+T5 external |
| 4 | FLUX.2 | yes | yes | yes | yes | PNG | Klein 4B/9B + dev |
| 4 | GLM-Image | yes | yes | yes | yes | PNG | AR encoder external |
| 5 | Stable Audio | yes | yes | yes | yes | WAV | Audio VAE stub |
| 5 | MMAudio | yes | yes | yes | yes | WAV | Hub reserved; `MMAUDIO_MODEL_PATH` |
| 6 | FastMetal MLX | yes | mlx crate | n/a | separate | scaffold | Needs Apple Silicon + mlx feature |
| — | LTX I2V encode | yes | — | stub | — | — | Full VAE encoder external |

Deferred (not blocking Phase 4–5 green): Action/CameraNet/PRoPE/cam injector
fuses, MoGe/SigLIP weight-load hooks.
