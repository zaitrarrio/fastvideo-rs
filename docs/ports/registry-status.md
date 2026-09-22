# Registry port status (Phases 4–6)

Snapshot of FastVideo Hub families landed in this tree after Phase 4–6 scaffolding
(2026-09-22). Green = host config + cudarc tiny forward/generate + registry +
CLI `generate_av` path. Weight-parity remains Hub-gated unless noted.

| Phase | Family | Spec | Models | Cudarc | Registry | Generate | Notes |
|---|---|---|---|---|---|---|---|
| 4 | Z-Image | yes | yes | yes | yes | PNG | Qwen3 encode when `text_encoder/` present; Hub key probes |
| 4 | SD 3.5 | yes | yes | yes | yes | PNG | T5-XXL (`text_encoder_3`) when present; Hub key probes |
| 4 | FLUX.1 | yes | yes | yes | yes | PNG | T5-XXL (`text_encoder_2`) when present; Hub key probes |
| 4 | FLUX.2 | yes | yes | yes | yes | PNG | CLIP/T5 encode path; Hub key probes |
| 4 | GLM-Image | yes | yes | yes | yes | PNG | CLIP broadcast or clear GLM AR error; Hub key probes |
| 5 | Stable Audio | yes | yes | yes | yes | WAV | T5/CLIP text encode when present |
| 5 | MMAudio | yes | yes | yes | yes | WAV | Public `hkchengrex/MMAudio` + env path; T5/CLIP text |
| 6 | FastMetal MLX | yes | mlx crate | n/a | separate | scaffold | Host stubs + Metal gate; mlx-rs Apple Silicon only |
| — | LTX I2V encode | yes | — | partial | — | — | Diffusers `encoder.*` path when keys present; else stub |

## Landed this pass

- Text encoders: real CLIP / T5-XXL / T5-11B / Qwen3 graphs when encoder dirs exist; otherwise clear load error (tiny scaffolds may zeros).
- DiT/VAE Hub key maps: `hub_keys` probes for SD3.5, FLUX.1/2, Z-Image, GLM-Image, Stable Audio, AutoencoderKL, LTX encoder, world fuses.
- LTX VAE encoder: `ltx2/vae_encoder` + I2V prefers it over spatial stub.
- World DiT fuses: CameraNet / PRoPE / Action / cam injector / SigLIP probe+inject when keys present.
- MMAudio: registered `hkchengrex/MMAudio` (native `.pth`); Diffusers id still reserved; `MMAUDIO_MODEL_PATH` for converted packs.
- MLX: `MetalGate` + `MlxArrayStub` + weights layout validation; no invented mlx-rs APIs.

## Still external-only

- Full Hub weight download / auth for gated packs (SD3.5, FLUX, GLM, …).
- Full 30-layer DiT / VAE conv parity (scaffolds load probes + shape-correct forward).
- GLM AR text tower (non-CLIP) and MMAudio Synchformer frame encode.
- Apple Silicon Metal DiT/TAEHV via mlx-rs (target-specific dep; see `docs/ports/fastmetal-mlx.md`).
- Full LTX encoder ResNet/downsample stack (partial conv_in path landed).
