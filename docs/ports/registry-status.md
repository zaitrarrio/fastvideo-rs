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
| 4 | GLM-Image | yes | yes | yes | yes | PNG | ByT5 glyph + AR text tower when dirs present; Hub probes |
| 5 | Stable Audio | yes | yes | yes | yes | WAV | T5/CLIP text encode when present |
| 5 | MMAudio | yes | yes | yes | yes | WAV | Synchformer visual encode when `image_encoder/` has `vfeat_extractor.*` |
| 6 | FastMetal MLX | yes | mlx crate | n/a | separate | scaffold | Target-optional `mlx-rs` 0.25 on aarch64; host stubs elsewhere |
| — | LTX I2V encode | yes | — | yes | — | — | Full encoder ResNet/downsample when keys present; else stem/stub |

## Landed this pass

- Text encoders: real CLIP / T5-XXL / T5-11B / Qwen3 / ByT5 graphs when encoder dirs exist; otherwise clear load error (tiny scaffolds may zeros).
- DiT/VAE Hub key maps: `hub_keys` probes for SD3.5, FLUX.1/2, Z-Image, GLM-Image (+ AR/ByT5), Stable Audio, AutoencoderKL, LTX encoder, MMAudio Synchformer, world fuses.
- LTX VAE encoder: full causal ResNet + `LTXVideoDownsampler3d` space-to-depth stack when `encoder.down_blocks.*` present; stem `conv_in` fallback otherwise; I2V prefers real encode.
- World DiT fuses: CameraNet / PRoPE / Action / cam injector / SigLIP probe+inject when keys present.
- MMAudio: Synchformer visual path (`mmaudio/synchformer.rs`) — 16/8 clips → ~24 fps × 768-d sync features; clear error if `image_encoder/` has unrecognized keys.
- GLM-Image: ByT5 glyph encode from Hub `text_encoder/`; AR text LM from `vision_language_encoder/` (no silent CLIP-only when AR/ByT5 packs exist).
- MLX: `MetalGate` + optional target `mlx-rs` (`Array`/`ops::zeros`/`Device::gpu`); DiT/TAEHV still scaffold after gate.

## Still external-only

- Full Hub weight download / auth for gated packs (SD3.5, FLUX, GLM, …).
- Full 30-layer DiT / VAE decode parity (scaffolds load probes + shape-correct forward).
- GLM AR **prior VQ `generate()`** loop (text tower hidden states wired; token upsample deferred).
- Apple Silicon Metal DiT/TAEHV graphs (mlx-rs gate open on aarch64 + `--features mlx`).
