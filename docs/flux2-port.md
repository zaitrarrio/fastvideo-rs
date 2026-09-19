# Flux2 Rust port

FLUX.2 lands in this repo the same way Wan/FastWan did: Candle is the
behavioral oracle, cudarc is the generate path, Diffusers safetensors load
through `fastvideo-loader` / `WeightMap`, registry + CLI select models, and
`scripts/gpu/` benches rust vs upstream FastVideo on one Vast.ai box.

**Priority in this PR:** FLUX.2-dev + Klein. FLUX.1 is a follow-up (checklist
at the end).

## Inventory (upstream → rust)

Mapped from hao-ai-lab/FastVideo and published Diffusers layouts.

| Piece | Upstream | Rust |
| --- | --- | --- |
| DiT | `fastvideo/models/dits/flux_2.py` → `Flux2Transformer2DModel` | Candle `fastvideo_models::flux2::Flux2Transformer2D`; cudarc `fastvideo_cudarc::flux2::Flux2Transformer2D` |
| Double / single stream | `transformer_blocks.*` + `single_transformer_blocks.*` | Same Diffusers key names |
| MLP | SwiGLU (`linear_in` → split/silu* → `linear_out`) | `swiglu` in both backends |
| RoPE | 4-axis (`axes_dims_rope` [32,32,32,32], θ=2000) | `text_ids` / `image_ids` + pair-rotate |
| Packed latents | 128-ch 2×2 pack (`in_channels=128`) | `pack_latents_2x2` / `unpatchify_2x2`; packed seq = `(H/8/2)*(W/8/2)` |
| VAE | `AutoencoderKLFlux2` / `flux2vae` | Candle decode (ResNet + upsample); cudarc decode is a linear+upsample stand-in until the 2D stack is ported |
| Scheduler | FlowMatchEuler + empirical μ | `FlowMatchEulerDiscreteScheduler::set_timesteps_flux2` + `compute_empirical_mu` |
| Dev text | Mistral3 layers (10, 20, 30), joint 15360, guidance embeds, 50 steps | Layer-stack postprocess + dummy/tiny embeds. Full Mistral3 HF encoder is follow-up |
| Klein text | Qwen3 layers (9, 18, 27), joint 7680, no guidance, 4 steps | Candle `Qwen3Encoder` + stack; cudarc uses prompt-hash dummy embeds this PR |
| Dev HF id | `black-forest-labs/FLUX.2-dev` | Registry preset `flux2_dev` |
| Klein HF ids | `black-forest-labs/FLUX.2-klein-4B`, `…-9B` | Presets `flux2_klein_4b` (5+20 blocks) and `flux2_klein_9b` |
| Pipeline configs | `Flux2PipelineConfig`, `Flux2KleinPipelineConfig` | Same names on `WanModelDefinition.pipeline_config` |
| Weight keys | Diffusers `transformer/` + `vae/` | `FLUX2_TRANSFORMER_REQUIRED_KEYS` / `FLUX2_VAE_REQUIRED_KEYS`; arch from `transformer/config.json` |

Published Klein 4B `transformer/config.json` (used as the smoke target):
`num_layers=5`, `num_single_layers=20`, `in_channels=128`,
`joint_attention_dim=7680`, `guidance_embeds=false`. FastVideo’s Flux2
`in_channels=64` / VAE `latent_channels=16` were discarded in favour of the
HF/Diffusers 128-ch pack (`latent_channels=32` before 2×2 pack).

## How to select models

Same CLI as Wan:

```bash
cargo run -p fastvideo-cli -- list-models
# … includes black-forest-labs/FLUX.2-dev and FLUX.2-klein-{4B,9B}

# Zero-weight CI smoke (no Hub download)
cargo run -p fastvideo-cli -- generate \
  --model black-forest-labs/FLUX.2-dev --tiny --output /tmp/flux2-dev-tiny
cargo run -p fastvideo-cli -- generate \
  --model black-forest-labs/FLUX.2-klein-4B --tiny --backend candle \
  --output /tmp/flux2-klein-tiny

# GPU generate (Vast; Diffusers snapshot on disk or in HF cache)
cargo run -p fastvideo-cli --release --features cuda-cudarc -- generate \
  --model black-forest-labs/FLUX.2-klein-4B \
  --device cuda --weights /workspace/weights/flux2 \
  --height 1024 --width 1024 --frames 1 --steps 4 --guidance 1.0 \
  --output /workspace/flux2-out \
  --prompt "a photo of a banana on a wooden table, studio lighting"
```

`--backend candle` is the oracle. `--backend cudarc` (default) is generate.
Burn/Luminal stay frozen and refuse Flux2.

## Vast.ai API bench vs upstream

No new rental stack. Same `VAST_API_KEY` from `.env` (never committed) and
`scripts/gpu/{validate,remote,upstream_bench,lib}.sh`.

```bash
cp .env.example .env && chmod 600 .env   # set VAST_API_KEY
scripts/gpu/validate.sh offers compare-flux2
scripts/gpu/validate.sh run compare-flux2
```

What the tier does on one box:

1. Fetch `FV_FLUX2_REPO` (default `black-forest-labs/FLUX.2-klein-4B`) into
   `/workspace/weights/flux2` (`transformer/`, `vae/`, `text_encoder/`,
   `tokenizer/`, `scheduler/`).
2. `remote.sh upstream-install` then `upstream-bench TORCH_SDPA --workload t2i`
   with the same height/width/steps/guidance/prompt/seed.
3. If `artifacts/gpucheck/dist/fastvideo` was produced by `docker.sh dist`,
   upload it and run `remote.sh flux2-rust-bench` (`fastvideo bench --device cuda`).

Overrides: `FV_FLUX2_REPO`, `FV_FLUX2_STEPS`, `FV_FLUX2_GUIDANCE`,
`FV_FLUX2_HEIGHT`, `FV_FLUX2_WIDTH`, `FV_FLUX2_PROMPT`, `FV_UPSTREAM_RUNS`,
`FV_UPSTREAM_BACKENDS`. Prompts also live in `scripts/gpu/prompts-flux2.json`.

Artifacts (same run dir as Wan compare):

- `remote/upstream-TORCH_SDPA.json` — load seconds, warmup, median/min generate
- `remote/flux2-rust/bench.json` — rust `load_ms` / `generate_ms`
- PNG stills under `remote/flux2-rust/` and upstream’s video dir (num_frames=1)

`docker.sh dist` now ships both `fv-gpucheck` and the `fastvideo` CLI
(`--features cuda-cudarc`). `validate.sh local` also unit-tests
`fastvideo-models` and `fastvideo-core` so Flux2 oracle tests run in preflight.

## What is complete vs stubbed

**Done (reviewable increment):**

- Shared Flux2 scaffolding (config, family helpers, weight-key map, registry,
  sampling presets, CLI `list-models` / `generate` / `bench` / `schedule`)
- Candle oracle: DiT forward, 2D VAE decode, flow-match Euler + μ, tiny
  generate for both families, Qwen3 encoder for Klein
- cudarc generate: DiT load + forward, tiny + Diffusers `WeightMap` load,
  simplified VAE decode, wired through `VideoGenerator`
- End-to-end tiny smokes (cudarc dev, Candle Klein) and layout test that
  checks a local HF snapshot when present
- Vast `compare-flux2` tier + `upstream_bench.py --workload t2i`

**Not this PR:**

- Full Mistral3 text encoder (dev uses dummy/tiny embeds unless you feed
  pre-stacked hidden states)
- cudarc Qwen3 (Klein GPU generate uses prompt-hash dummy text this PR)
- Bit-exact DiT/VAE parity vs FastVideo / Diffusers
- Full cudarc 2D VAE (ResNet + upsample weights)
- FLUX.1 (see below)

## FLUX.1 follow-up checklist

Do this in a later PR; do not invent a third architecture.

1. **Inventory** Diffusers / FastVideo FLUX.1: `FluxTransformer2DModel`
   (double+single stream, but **3-axis RoPE**, `in_channels=64`,
   `joint_attention_dim=4096`, `guidance_embeds=true`), CLIP-L + T5-XXL text,
   `AutoencoderKL` (SD3-style, 16 latent channels, no 2×2 pack),
   FlowMatchEuler **without** Flux2 empirical μ.
2. **Reuse Flux2 DiT scaffolding** where the graph matches (double/single
   blocks, modulation, AdaLN). Split only the bits that differ: RoPE rank,
   pack/unpack, guidance default, text concat (CLIP pooled + T5 tokens).
3. **Candle oracle first:** CLIP + T5 (or load precomputed embeds), DiT
   forward, VAE decode, tiny generate. Registry key
   `black-forest-labs/FLUX.1-dev` (and schnell if the pipeline config exists).
4. **cudarc generate** after Candle tiny+layout tests pass. Same
   `VideoGenerator` / CLI flags (`--frames 1`).
5. **Loader:** Diffusers `transformer/`, `vae/`, `text_encoder/`,
   `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`. Required-key table +
   `transformer/config.json` override, mirroring `arch_from_transformer_config`.
6. **Tests:** FastVideo already compares FLUX.1 vs Diffusers — port the
   numerical checks that fit (timestep table, pack math, a tiny forward).
7. **Vast:** add `compare-flux1` or a `FV_FLUX_FAMILY=flux1` override on the
   existing `compare-flux2` path. Same `upstream_bench.py --workload t2i`
   (`--model-path black-forest-labs/FLUX.1-dev`, 50 steps, guidance 3.5).
8. **Do not** unfreeze Burn/Luminal for FLUX.1.

Suggested HF ids: `black-forest-labs/FLUX.1-dev`,
`black-forest-labs/FLUX.1-schnell`. Pipeline config lives at
FastVideo `configs/pipelines/flux.py`.
