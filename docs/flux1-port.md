# FLUX.1 Rust port

FLUX.1 follows the Flux2 port pattern: Candle is the behavioral oracle, cudarc
is the generate path, Diffusers safetensors load through `fastvideo-loader` /
`WeightMap`, registry + CLI select models, and `scripts/gpu/` benches rust vs
upstream FastVideo on one Vast.ai box.

This is **not** a third architecture. The graph is FluxTransformer2D
(double + single stream) with the published FLUX.1 splits: 3-axis RoPE, CLIP-L
+ T5-XXL, SD3 AutoencoderKL (16 latent channels), and FlowMatchEuler
`calculate_shift` (no Flux2 empirical μ).

See [flux2-port.md](flux2-port.md) for the shared scaffolding.

## Inventory (upstream → rust)

Mapped from FastVideo `configs/pipelines/flux.py` / `configs/models/dits/flux.py`
and published Diffusers `black-forest-labs/FLUX.1-dev` /
`FLUX.1-schnell`.

| Piece | Upstream | Rust |
| --- | --- | --- |
| DiT | `FluxTransformer2DModel` (19 double + 38 single) | Candle `fastvideo_models::flux1::Flux1Transformer2D`; cudarc `fastvideo_cudarc::flux1::Flux1Transformer2D` |
| Modulation | Per-block AdaLN-Zero (not Flux2 shared `Modulation`) | `AdaLayerNormZero` / `AdaLayerNormZeroSingle` / `AdaLayerNormContinuous` |
| MLP | GELU-tanh (`ff.net.0.proj` / `ff.net.2`) | `gelu_tanh` in both backends |
| RoPE | 3-axis `[16, 56, 56]`, θ=10000 | `text_ids` (zeros) / `image_ids` (`[0, h, w]`) + pair-rotate |
| Packed latents | Diffusers `_pack_latents`: 16ch → 64ch 2×2 (`in_channels=64`) | `pack_latents_flux1` / `unpack_latents_flux1`; packed seq = `(H/8/2)*(W/8/2)` |
| VAE | SD3-style `AutoencoderKL`, 16 latent channels, scale 0.3611, shift 0.1159 | Reuses Flux2 2D VAE decode with `Flux2VaeConfig::flux1()` |
| Scheduler | FlowMatchEuler + Diffusers `calculate_shift` | `set_timesteps_flux2` + `calculate_shift_flux1` (μ = seq·m + b) |
| Text | CLIP-L pooled (768) + T5-XXL tokens (4096) | Candle + cudarc `ClipTextEncoder` + `T5Encoder`. Tiny/CI stays dummy |
| Guidance | `guidance_embeds=true`; timestep/guidance/CLIP pooled via `time_text_embed` | Combined timestep + guidance + text proj |
| Dev HF id | `black-forest-labs/FLUX.1-dev` | Registry preset `flux1_dev` (50 steps, guidance 3.5, T5 pad 512) |
| Schnell HF id | `black-forest-labs/FLUX.1-schnell` | Preset `flux1_schnell` (4 steps, guidance 0, T5 pad 256) |
| Pipeline config | FastVideo `FluxPipelineConfig` | Same name on `WanModelDefinition.pipeline_config` |
| Weight keys | Diffusers `transformer/`, `vae/`, `text_encoder/`, `text_encoder_2/` | `FLUX1_TRANSFORMER_REQUIRED_KEYS` / `FLUX1_CLIP_REQUIRED_KEYS` / `FLUX1_T5_REQUIRED_KEYS`; arch from `transformer/config.json` |

The checklist in flux2-port.md said “no 2×2 pack”. Published FLUX.1 still
packs 16 VAE channels to 64 DiT channels with Diffusers’ 2×2 neighbourhood
(`c*4 + dy*2 + dx`). That is **not** Flux2’s 32→128 NCHW pack helper; the
pixel order is the same 2×2 but storage is seq-major `[seq, C*4]` on the
Diffusers side. Generate stores packed latents as NCHW `[64, ph, pw]` and
permutes to `[seq, 64]` for the DiT.

Published DiT widths (dev and schnell share the graph):

`num_layers=19`, `num_single_layers=38`, `in_channels=64`,
`num_attention_heads=24`, `attention_head_dim=128` (hidden 3072),
`joint_attention_dim=4096`, `pooled_projection_dim=768`,
`axes_dims_rope=[16, 56, 56]`, `guidance_embeds=true`.

## How to select models

```bash
cargo run -p fastvideo-cli -- list-models
# … includes black-forest-labs/FLUX.1-dev and FLUX.1-schnell
# plus FLUX.2-dev / FLUX.2-klein-{4B,9B}

# Zero-weight CI smoke (no Hub download)
cargo run -p fastvideo-cli -- generate \
  --model black-forest-labs/FLUX.1-dev --tiny --output /tmp/flux1-dev-tiny
cargo run -p fastvideo-cli -- generate \
  --model black-forest-labs/FLUX.1-schnell --tiny --backend candle \
  --output /tmp/flux1-schnell-tiny

# GPU generate (Vast; Diffusers snapshot on disk or in HF cache)
cargo run -p fastvideo-cli --release --features cuda-cudarc -- generate \
  --model black-forest-labs/FLUX.1-schnell \
  --device cuda --weights /workspace/weights/flux1 \
  --height 1024 --width 1024 --frames 1 --steps 4 --guidance 0 \
  --output /workspace/flux1-out \
  --prompt "a photo of a banana on a wooden table, studio lighting"
```

`--backend candle` is the oracle. `--backend cudarc` (default) is generate.
Burn/Luminal stay frozen and refuse new Flux work.

`FASTVIDEO_FLUX1_DUMMY_TEXT=1` keeps the prompt-hash stand-in (A/B vs real
CLIP+T5). `FASTVIDEO_FLUX1_TEXT_LEN` overrides the T5 pad (512 dev / 256 schnell).

Loader layout: `transformer/`, `vae/`, `text_encoder/` (CLIP-L),
`text_encoder_2/` (T5-XXL), `tokenizer/`, `tokenizer_2/`. I64 tensors are
skipped on ingest (same as Flux2).

## Vast.ai compare vs upstream

Same rental stack as Flux2 (`VAST_API_KEY` in `.env`).

```bash
scripts/gpu/validate.sh offers compare-flux1
scripts/gpu/validate.sh run compare-flux1
```

Default smoke is **FLUX.1-schnell** (4-step, guidance 0) so disk/VRAM stay
closer to the Klein 4B Flux2 compare. Override for the published FastVideo
FLUX.1-dev numbers:

```bash
FV_FLUX1_REPO=black-forest-labs/FLUX.1-dev \
FV_FLUX1_STEPS=50 FV_FLUX1_GUIDANCE=3.5 \
scripts/gpu/validate.sh run compare-flux1
```

`FV_FLUX_FAMILY=flux1` on `compare-flux2` selects the same FLUX.1 runner
(schnell defaults unless `FV_FLUX1_*` / `FV_FLUX2_*` override).

What the tier does on one box:

1. Fetch `FV_FLUX1_REPO` (default `black-forest-labs/FLUX.1-schnell`) into
   `/workspace/weights/flux1` (`transformer/`, `vae/`, `text_encoder/`,
   `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`, `scheduler/`).
2. `remote.sh upstream-install` then `upstream-bench TORCH_SDPA --workload t2i`
   with the same height/width/steps/guidance/prompt/seed.
3. If `artifacts/gpucheck/dist/fastvideo` was produced by `docker.sh dist`,
   upload it and run `remote.sh flux2-rust-bench` (`fastvideo bench --device cuda`).

Overrides: `FV_FLUX1_REPO`, `FV_FLUX1_STEPS`, `FV_FLUX1_GUIDANCE`,
`FV_FLUX1_HEIGHT`, `FV_FLUX1_WIDTH`, `FV_FLUX1_PROMPT`, `FV_UPSTREAM_RUNS`,
`FV_FLUX1_WARMUP` / `FV_FLUX1_RUNS` (fall back to `FV_FLUX2_*` then
`FV_UPSTREAM_RUNS`).

Manual upstream (no rust):

```bash
# FLUX.1-dev (FastVideo defaults)
python scripts/gpu/upstream_bench.py --backend TORCH_SDPA --workload t2i \
  --model-path black-forest-labs/FLUX.1-dev \
  --height 1024 --width 1024 --num-frames 1 --steps 50 --guidance 3.5 --fps 1 --seed 0

# FLUX.1-schnell
python scripts/gpu/upstream_bench.py --backend TORCH_SDPA --workload t2i \
  --model-path black-forest-labs/FLUX.1-schnell \
  --height 1024 --width 1024 --num-frames 1 --steps 4 --guidance 0 --fps 1 --seed 0
```

Full FLUX.1-dev (T5-XXL + 12B-class DiT) may OOM a 24GB card with real text;
use schnell, `FASTVIDEO_FLUX1_DUMMY_TEXT=1`, or a 40GB+ box for the 50-step
dev compare.

## What is complete vs residual

**Done:**

- Registry + sampling presets for `FLUX.1-dev` and `FLUX.1-schnell`
- Candle oracle: CLIP-L + T5-XXL (or dummy), DiT (AdaLN-Zero / GELU / 3-axis
  RoPE / CLIP pooled), SD3 VAE decode, flow-match Euler + `calculate_shift`,
  tiny generate
- cudarc generate: same graph, wired through `VideoGenerator` / CLI
  (`--frames 1`)
- Loader: Diffusers dirs + required-key tables + `transformer/config.json`
  override
- Tests: timestep/`calculate_shift` anchors, pack math, tiny DiT/CLIP/T5
  shapes, Candle + cudarc tiny PNG smokes
- Vast `compare-flux1` + `FV_FLUX_FAMILY=flux1`

**Still needs a GPU Vast run:**

- Real CLIP+T5 + full DiT vs upstream FastVideo on schnell (4-step) and
  optionally FLUX.1-dev (50-step, guidance 3.5)
- Bit-exact DiT/VAE vs Diffusers at published width (bf16 GEMM / SDPA will
  not be bitwise identical)

**Not in this port:**

- Burn / Luminal (stay frozen)
- VAE tiling / slicing
- FLUX.1 ControlNet / Fill / Redux / Canny (only T2I dev + schnell)
