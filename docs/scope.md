# Scope

Snapshot of what this tree registers and runs, as of 2026-09-23. Per-model
algorithms live in `docs/ports/`. This page is the index: families, Hub ids,
workloads, and which NVlabs/Sana `sol-engine` profiles are actually wired.

Host math is `crates/fastvideo-models`. The CUDA graph is
`crates/fastvideo-cudarc` under the same module name. `crates/fastvideo-core`
resolves a Hub id to one of those: every family is one `VideoFamily` in its
static `FAMILIES` list (`family/<fam>.rs`: the Hub-id table and the run), and
every generate takes the common `Request` (ADR-0013, plan B2). A Mac without `nvcc` can test the host
half. `--features cuda` needs a CUDA toolchain.

## Crates

| Crate | Role |
|---|---|
| `fastvideo-ops` | Shared numeric helpers |
| `fastvideo-models` | Configs, schedules, geometry, sol-engine contracts. No CUDA |
| `fastvideo-loader` | Safetensors, lazy shards, bf16 views. `unsafe` allowed (mmap) |
| `fastvideo-cudarc` | The runtime (`src/rt`: tensors, device, kernels, ops, weights) and the family pipelines. `unsafe` allowed (driver, cuBLAS, cuDNN FFI) |
| `fastvideo-core` | The `VideoFamily` trait and the static family list (Hub ids, resolution, each family's run), the common `Request` and `Request::validate(&ModelCaps)`, Wan sampling and `VideoGenerator` |
| `fastvideo-cli` | `fastvideo list-models` / `generate` / `bench` / `schedule` over the family list; behaviour pinned by goldens (`tests/cli_golden.rs`) |
| `fastvideo-gpucheck` | Staged GPU checks (`kernels`, `parity`, `clip`). `unsafe` allowed |
| `fastvideo-manifest` | The registry's public formats: model names, `model.toml` v1, the signed lineage manifest and its canonical BLAKE3 digest, JSON Schema export (ADR-0005..0007); layer 2, the family manifest `families/*.toml` (ADR-0008). Pure: no I/O, no CUDA (`fs` feature: the one directory reader) |
| `fastvideo-lab` | The lab's gold standards: `lab/gold/<variant>.toml` types, validation, directory reader and JSON Schema (ADR-0009), and the lint that keeps measurements out of `profiles/` comments. CPU only; plan D1 adds the `fv-lab` binary |

Shared device types (`CudaTensor`, conv, the weight map, the frame writer)
live under `cudarc/src/rt` (the runtime, extracted from `wan` in plan B1; `wan`
re-exports them under the old paths for now) and are used by every other model.

The served families are described once in `families/*.toml` (layer 2 of the
model family manifest, `docs/adr/0008-family-manifest-layer-2.md`): one file
per family (`h3`, `ltx2`, `wan`), one `[[variants]]` table per servable
configuration with its id, served and public names, tier, weights trees, GPU
needs, layer-1 `[caps]`, the runtime's `[serving]` caps and the engine
recipe. They are the source (plan A3): the engine has no hand-written
catalog, it embeds the files and projects them
(`crates/fastvideo-engine-service/src/cuda/families.rs`), the serve configs
name a variant and its residency (`[[models]] variant = ...`, no
`[aliases]`), and the per-model tables below are being turned into
projections of them (plan A4).

Workloads: `t2v`, `i2v`, `t2av`, `fl2va`, `ref2va`, `t2i`, `t2a`, `v2a`,
plus control/edit entries that carry an empty workload list in their family table.

## Video

### Wan

Samplers: UniPC, DMD, causal DMD, TurboDiffusion rCM.

| Preset | Sampler | Workload | Hub id |
|---|---|---|---|
| `wan_t2v_1_3b` | UniPC | t2v | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| `wan_t2v_14b` | UniPC | t2v | `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` |
| `wan_i2v_14b_480p` | UniPC | i2v | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
| `wan_i2v_14b_720p` | UniPC | i2v | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` |
| `wan_fun_1_3b_inp` | UniPC | i2v | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` |
| `wan_fun_1_3b_control` | UniPC | control | `IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers` |
| `turbo_t2v_1_3b` | rCM | t2v | `loayrashid/TurboWan2.1-T2V-1.3B-Diffusers` |
| `turbo_t2v_14b` | rCM | t2v | `loayrashid/TurboWan2.1-T2V-14B-Diffusers` |
| `turbo_i2v_a14b` | rCM | i2v | `loayrashid/TurboWan2.2-I2V-A14B-Diffusers` |
| `fast_wan_t2v_480p` | DMD | t2v | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, `FastVideo/FastWan2.1-T2V-14B-480P-Diffusers`, `FastVideo/FastWan-QAD-1.3B`, `FastVideo/FastWan-QAD-1.3B-SA2`, `FastVideo/FastWan-QAD-FP8-1.3B` |
| `wan_2_2_ti2v_5b` | UniPC | t2v, i2v | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| `fast_wan_2_2_ti2v_5b` | DMD | t2v, i2v | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`, `FastVideo/FastWan2.2-TI2V-5B-Diffusers` |
| `lucy_edit_dev` | UniPC | edit | `decart-ai/Lucy-Edit-Dev`, `decart-ai/Lucy-Edit-1.1-Dev` |
| `wan_2_2_t2v_a14b` | UniPC | t2v | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| `wan_2_2_i2v_a14b` | UniPC | i2v | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| `sf_wan_t2v_1_3b` | causal DMD | t2v | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers` |
| `sf_wan_2_2_t2v_a14b` | causal DMD | t2v | `rand0nmr/SFWan2.2-T2V-A14B-Diffusers` |
| `sf_wan_2_2_i2v_a14b` | causal DMD | i2v | `FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers` |

QAD checkpoints are the same architecture as FastWan 1.3B and ship
unquantized. They are listed so they do not fall through to UniPC.

The distilled presets (DMD, rCM, causal DMD) decode through TAEHV
(`taew2_1`) by default when its weights are found; `FASTVIDEO_WAN_VAE=full`
opts out to the Wan VAE (2 latent frames per pass by default). The mp4 is
written while the decode runs, PNG frames after it. Text K/V, text and time
embeddings are computed once per denoise (`FASTVIDEO_WAN_COND_CACHE=0` turns
it off, byte-identical), and UMT5 prompts are cached on disk. A VSA model id
needs `FASTVIDEO_VSA=1`, which enables the device VSA kernels in
self-attention only. Wan2.2 TI2V-5B runs the Wan 2.2 VAE (z 48, 16×
spatial through patchify, residual blocks; config from `vae/config.json`)
and, with `--image`, TI2V image-to-video: the encoded image pinned to latent
frame 0 at timestep 0 (per-frame DiT timesteps). `taew2_2` decodes it with
`FASTVIDEO_WAN_VAE=taehv`. Measurements and the upstream comparison:
[ports/wan.md](ports/wan.md).

### LTX-2 / 2.3 / 2.5

Joint text-to-audio-and-video. Gemma text, video VAE, audio VAE, vocoder.
2.5 uses Gemma 4, ancestral Euler, and a BWE vocoder. 2.3 distilled uses
Gemma 3 and the same DiT geometry as 2.5.

| Preset | Line | Workload | Hub id |
|---|---|---|---|
| `ltx2_distilled_20` | 2.0 distilled, 8-step, CFG 1 | t2av | `FastVideo/LTX2-Distilled-Diffusers`, `rootonchair/LTX-2-19b-distilled` |
| `ltx2_base_20` | 2.0 base, 40-step CFG | t2av | `Lightricks/LTX-2`, `FastVideo/LTX2-base`, `FastVideo/LTX2-Diffusers` |
| `ltx2_distilled_23` | 2.3 distilled, stage-1 res2s, stage-2 Euler, no LoRA | t2av, i2v | `FastVideo/LTX-2.3-Distilled-Diffusers`, `diffusers/LTX-2.3-Distilled-Diffusers` |
| `ltx2_base_23` | 2.3 base, 30-step CFG res2s. STG masks stay unset. Dev DiT LoRA 0.25 / 0.5 | t2av, i2v | `Lightricks/LTX-2.3`, `FastVideo/LTX2.3-Diffusers`, `diffusers/LTX-2.3-Diffusers` |
| `ltx2_distilled_25` | 2.5 distilled, ancestral stage 1, deterministic Euler stage 2, no LoRA | t2av | `Lightricks/LTX-2.5-Diffusers` |

Two-stage: half-resolution stage 1, spatial latent upsampler, then a 2- or
3-step stage-2 refine. Stage-2 sigmas are `0.909375`, `0.725`, `0.421875`, `0`.
`--diff-vae` on 2.5 swaps the conv video decoder for the diffusion decoder
(one x0 step, untiled). I2V encodes the first frame when `vae/encoder.*` is
present, and uses a spatial stub otherwise.

`FASTVIDEO_LTX2_HQ=1` selects the 2.3-base 15+3 HQ contract.

### MiniMax-H3 / FastH3

Joint audio and video. Recipes share the `sol_h3` registry row unless noted.
`sol-h3-spark` is a recipe name, not a separate Hub id.

| Preset / recipe | Workload | Hub id |
|---|---|---|
| `fasth3_8step` | t2av, fl2va | `FastVideo/FastVideo-FastH3-8-Step-V2`, `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` |
| `minimax_h3` | t2av, fl2va, ref2va | `MiniMaxAI/MiniMax-H3` |
| `sol_h3` | t2av, i2v, fl2va, ref2va | `FastVideo/FastH3-4-step-Preview-v1-LoRA` |
| `sol_h3_ref2va` | ref2va | `lightx2v/Minimax-h3-Turbo` |
| `4step-dense` / `4step-vsa` | named recipes on the H3 contract | same weights as the selected preset |
| `sol-h3` | 4 forwards, video shift 12, audio shift 3, Spark Sol-Attn | one-GPU Sol-H3 profile |
| `sol-h3-spark` | same draft, then the Spark bridge | draft canvas 672×384×124. Output target 1344×768×121 |

Spark, when `minimax_h3_latent_upscaler_3d_bf16.safetensors` and the
H3-to-LTX adapter are beside the weights: ×2 latent resize, adapter, crop to
`[1, 128, 16, 24, 42]`. If `FASTVIDEO_LTX2_WEIGHTS` is set, the 3-step LTX-2.5
refiner encodes the 32 kHz H3 PCM and muxes that same PCM into
`out_dir/refined/`. The H3 DiT stays resident for that load.

### HunyuanVideo 1.5

| Preset | Workload | Hub id |
|---|---|---|
| `hy15_480p_t2v` | t2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |
| `hy15_480p_i2v_distilled` | i2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled` |
| `hy15_720p_t2v` | t2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` |
| `hy15_720p_i2v_distilled` | i2v | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled` |
| `hy15_1080p_sr` | t2v | `weizhou03/HunyuanVideo-1.5-Diffusers-1080p`, `weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR` |

`FASTVIDEO_HUNYUAN15_OFFICIAL=1` uses 1280×720, 129 frames, 50 steps. Guidance
6 is recorded. This family has no CFG pair.

### Kandinsky 5.0

| Preset | Workload | Hub id |
|---|---|---|
| `k5_lite_t2v_5s` | t2v | `kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers` |
| `k5_pro_t2v_5s` | t2v | `kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers` |

### Cosmos Predict2

EDM Video2World, not the Cosmos3-Super FlowMatch profile.

| Preset | Workload | Hub id |
|---|---|---|
| `cosmos2_v2w_2b` | i2v | `nvidia/Cosmos-Predict2-2B-Video2World` |
| `cosmos2_v2w_14b` | i2v | `nvidia/Cosmos-Predict2-14B-Video2World` |

`FASTVIDEO_COSMOS3_OFFICIAL=1` selects 1280×720, 189 frames, 35 steps,
guidance 6. Flow-shift 10 is recorded and not applied.

### LongCat

| Preset | Workload | Hub id |
|---|---|---|
| `longcat_t2v_480p` | t2v | `FastVideo/LongCat-Video-T2V-Diffusers` |
| `longcat_t2v_720p` | t2v | same repo, 720p BSA preset |

### LingBot-Video

| Preset | Workload | Hub id |
|---|---|---|
| `lingbot_dense_1_3b` | t2v | `robbyant/lingbot-video-dense-1.3b` |
| `lingbot_moe_30b` | t2v | `robbyant/lingbot-video-moe-30b-a3b` |

`FASTVIDEO_LINGBOT_OFFICIAL=1` selects 832×480, 121 frames, 40 steps,
guidance 3, shift 3. The 1920×1088 refiner (8 steps, `t_thresh` 0.85) is
recorded and has no upsampler here.

## World and camera

All of these are image-to-video. Camera, action, and 3D-cache injectors run
when the matching weight keys are present.

| Preset | Hub id |
|---|---|
| `gen3c_cosmos_7b` | `FastVideo/GEN3C-Cosmos-7B-Diffusers` |
| `mg2_base_distilled` | `FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers` |
| `mg2_gta_distilled` | `FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers` |
| `mg2_templerun_distilled` | `FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers` |
| `mg2_base` | `FastVideo/Matrix-Game-2.0-Base-Diffusers`, GTA and TempleRun base packs |
| `mg3_base_distilled` | `FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers` |
| `dreamx_5b_cam` | `FastVideo/DreamX-World-5B-Cam-Diffusers` |
| `dreamx_5b_ar` | `FastVideo/DreamX-World-5B-Diffusers` |
| `lingbotworld_base_cam` | `FastVideo/LingBot-World-Base-Cam-Diffusers` |
| `lingbotworld2_causal_fast` | `robbyant/lingbot-world-v2-14b-causal-fast` |
| `gamecraft_i2v` | `FastVideo/HunyuanGameCraft-Diffusers` |
| `hyworld_bidirectional` | `FastVideo/HY-WorldPlay-Bidirectional-Diffusers` |

## Image and audio

These have a registry row, a cudarc generate path, and a text encoder when
the encoder directory is in the snapshot. Full-depth DiT and VAE parity
against Hub weights is still gated on those weights. See
`docs/ports/registry-status.md`.

| Preset | Workload | Hub id | Text |
|---|---|---|---|
| `zimage_turbo` | t2i | `Tongyi-MAI/Z-Image-Turbo` | Qwen3 |
| `sd35_medium` | t2i | `stabilityai/stable-diffusion-3.5-medium` | T5-XXL |
| `flux1_dev` | t2i | `black-forest-labs/FLUX.1-dev` | T5-XXL |
| `flux2_klein_4b` | t2i | `black-forest-labs/FLUX.2-klein-4B` | CLIP / T5 |
| `flux2_klein_9b` | t2i | `black-forest-labs/FLUX.2-klein-9B` | CLIP / T5 |
| `flux2_dev` | t2i | `black-forest-labs/FLUX.2-dev` | CLIP / T5 |
| `glm_image` | t2i | `zai-org/GLM-Image` | ByT5 glyph plus the AR tower when those dirs exist |
| `stable_audio_open_1_0` | t2a | `FastVideo/stable-audio-open-1.0-Diffusers` | T5 / CLIP |
| `stable_audio_open_small` | t2a | `FastVideo/stable-audio-open-small-Diffusers` | T5 / CLIP |
| `mmaudio_large_44k_v2` | v2a, t2a | `hkchengrex/MMAudio` (tree from `scripts/gpu/fetch-mmaudio.sh`) | CLIP DFN5B text; CLIP + Synchformer visual. Also the opt-in Wan soundtrack stage (`--audio mmaudio`). See [ports/mmaudio.md](ports/mmaudio.md) |

The Apple Silicon scaffold `fastvideo-mlx` (host stubs, `mlx-rs` behind a
feature, no generate path) was retired in plan B6
(`docs/adr/0017-loader-stages.md`). What stays is the CUDA runtime's reader of
FastVideo's MLX H3 export (`mlx_h3_dit.safetensors` + `mlx_h3_dit.json`, affine
INT8/6/4, `crates/fastvideo-cudarc/src/h3/mlx.rs`): an H3 load whose
`transformer/` (or root) holds that export runs it on the GPU with the fused
affine GEMM.

## Sol-engine profiles

From NVlabs/Sana branch `sol-engine` (`models/`). `main` on that repo is the
SANA image and video family and is not implemented here.

| Profile | What runs |
|---|---|
| Wan 2.1 1.3B / 14B, Wan 2.2 A14B and TI2V-5B | The matching presets above. `FASTVIDEO_WAN_SOL_CACHE=taylorseer` forecasts `proj_out` (interval 3, warmup 3, cooldown 2, order 1). Batched CFG TeaCache is on for the Wan CFG path |
| LTX-2.3 | Stage-1 ODE res2s (2 evals/step except last; 15-step HQ = 29 calls). Stage-2 Euler (3-forward Sol/PISA contract). Distilled checkpoints run without LoRA; the 0.25/0.5 pair fuses on the **dev** BF16 DiT only. Stage-1 SCSP skips **res2s calls** 16–28 of 29 when `FASTVIDEO_LTX2_STAGE1_CACHE=1` (an Euler 15-step run never reaches call 16). Stage-2 PISA at sparsity 0.9, block 64. Midpoint token prune when `FASTVIDEO_LTX2_MIDPOINT_PRUNE=1` |
| LTX-2.5 | Ancestral stage 1. Stage-2 and Spark refiner are deterministic Euler (`denoise_cfg` / `denoise`), not ancestral. Stage-2 Sol-Attn: layer 0 dense, layers 1–47 at tau 1 / 1.25 / 1.5. LoRA 0.8 on the **dev** BF16 DiT only — distilled two-stage does not fuse. GB200 first-block cache when `FASTVIDEO_LTX2_FBCACHE=1` (threshold 0.08, warmup 1, max 10 skips, **stage 1 only**; stage 2 / refiner stay disarmed) |
| LTX-2.5 refiner / Spark | H3×2 upscaler, H3-to-LTX adapter, `encode_audio`, 3-step deterministic joint refine, original PCM muxed |
| MiniMax-H3 Spark and RTX | 4-step `sol-h3` is Spark Sol-Attn (update 0 dense; later updates layer 0 dense + tau 1 / 1.25 / 1.5). `sol-h3-spark` is VSA 0.9 + Sol-Attn Off. RTX route (`sol-h3-rtx` / `FASTVIDEO_H3_SOL_ATTN=rtx`): first 10 steps and first 2 layers dense, tau 1.0, 49 forwards. `FASTVIDEO_H3_SOL_CACHE=teacache` is the RTX residual controller (threshold 0.10, retain 5, cooldown 1) |
| Cosmos3-Super | 64B MoT T2V port (`cosmos3`, docs/ports/cosmos3.md): text-tower K/V cached per prompt, gen tower per step, Hub UniPC Karras schedule. TeaCache (threshold 1.15, start step 10, max 3) on the gen stack; precision arm is `FASTVIDEO_FP8` W8A8, not NVFP4. `fv-gpucheck sol cosmos3-gen`; not in the CLI registry. Not GPU-run |
| HunyuanVideo | Official canvas only. The profile's TeaCache is HunyuanVideo-13B and is not applied |
| LingBot | MoE 30B-A3B per the Hub config, base + 1080p refiner (docs/ports/lingbot.md). `FASTVIDEO_LINGBOT_SOL=fullopt`: EasyCache (base 0.08 / refiner 0.25) + refiner PISA 0.10, layers 0–3 and refiner steps 0, 1, last dense. CP4 / FSDP topology not reproduced. Not GPU-run |
| Sana-Video 5B | Not in this tree. The sol-engine profile wraps a private bundle |
| SANA-Video 2B (public) | Ported (2026-10-06, GPU-unmeasured): `sana_video`, `fv-gpucheck sana-video gen --arm baseline\|full` (EasyCache 0.1, QKV merge, bf16 linear-attention operands). The published 2.77x is for this model; see docs/ports/sana-video.md |

4-step `sol-h3` uses Spark Sol-Attn, not the RTX first-10-dense window (that
window would make every 4-step forward dense). `sol-h3-rtx` is the 49-forward
RTX 5090 cell. SOL/BSA is not a multi-GPU-only profile. `to_gate_compress`
loads only when `vsa_sparsity > 0` (MiniMax-H3 has no gate).
`sol-h3-spark` Stage-1 is VSA 0.9 + FastH3_VSA_DataFree at strength 1.0, BF16
(upstream W8A8 FP8 after the LoRA merge is off: measured 16–20 dB). The
VSA-DataFree file is 50 `.set_weight` replacements; fuse still expects
dense-datafree LoRA A/B.

Phase 3 on one RTX PRO 6000 96 GB (5-min cell cap, image `af5edf649671bfb7`):
FastH3 8-step **10.7 s/step** (Phase 0 same card **75.4 s**); LTX-2.5 two-stage
0.42 s then 1.67 s/step, VAE OOM; LTX-2.0 distilled **1.47 s/step** ok;
Hunyuan Diffusers keys do not match `Hunyuan15.*`; Wan 1.3B not on the volume.

NVFP4 (`FASTVIDEO_NVFP4=1` → TE `static_6`; `mse` / `4o6` stay FourOverSix)
dequantizes W4A4 on device when a CUDA context is live (`nvfp4_reconstruct`).
**LTX-2 is the exception:** there `FASTVIDEO_NVFP4` (profile
`ltx2/ltx25_distill_sol_nvfp4`, default off) is sol-engine's `nvfp4_ffn.py`
scope, a real W4A4 GEMM on the 48 video FFNs only: cuBLASLt block-scaled
FP4 (`CUDA_R_4F_E2M1`, `VEC16_UE4M3` scales in the 128x4 tiled layout,
device alpha, bf16 out, bias epilogue; `wan/nvfp4_linear.rs`), weights
quantized at load and activations per call with the TE rule, the FFN's GELU
fused into the down projection's quantizer, RHT / stochastic rounding off,
rows padded to 16, bf16 fallback below sm_100. Audio FFN, attention, Gemma
and every other linear stay bf16, and the LongLive dequant / K/V path is off
in an LTX process. RTX PRO 6000 (`fv-gpucheck kernels`, group
`nvfp4_linear`): the up projection's whole linear (quantize + GEMM + bias)
is 2.7-3.2x the bf16 linear (down projection with its GELU 2.2-2.3x), the
GEMM alone ~3.5-4x; output PSNR vs bf16 37.8 dB. Generations (Sol stage 2,
warm): denoise 1.135x at 4k5s and 1.122x at 1080p20s, peak VRAM 8.6 GiB
lower; the promotion gate passes at 1080p20s (LPIPS 0.254) and fails at
4k5s on the sharpness range (1.098 > 1.08; LPIPS 0.193), so it stays off.
See [ports/ltx25.md](ports/ltx25.md#nvfp4-video-ffn-fastvideo_nvfp4-profile-ltx2ltx25_distill_sol_nvfp4).
sol-engine's pre-quantized RTX5090 checkpoint is not on our volume, so it is
not loaded.
The Tile-IR W4A4 GEMM stays off (`FASTVIDEO_NVFP4_OXIDE_GEMM`) until it beats
cuBLAS bf16 on the H3 FFN shape and PSNR ≥ 30 dB vs bf16. CUTLASS
SM100/SM120, Blackwell `to_blocked`, RHT, 2D block scales, and stochastic
rounding are not implemented. KWL operator fusion from the LTX-2.3 optimized
arm is not implemented. The upstream env turns several of those fusions off
because they change frames.

## Configurations

Weights: `--weights`, `FASTVIDEO_WEIGHTS`, or the Hugging Face snapshot.

Every `FASTVIDEO_*` setting the workspace reads, rendered from the declarative
list in `crates/fastvideo-models/src/techniques/settings/table.rs` (a test
fails when this table and the list differ; ADR-0010). *Read by*: `env, then
profile` settings take the environment variable, then the active technique
profile's value ([techniques.md](techniques.md#precedence)), then the default;
`env` settings take the process environment only, and a profile's `[env]`
entry for one has no effect. *Read*: when a change takes effect (once per
process: the first read is cached).

<!-- BEGIN GENERATED: FASTVIDEO_* settings (crates/fastvideo-models/src/techniques/settings/table.rs; do not edit) -->

| Variable | Read by | Type and parse | Default | Read | Family | Seam | Effect |
|---|---|---|---|---|---|---|---|
| `FASTVIDEO_ARTIFACT_DIR` | env | text: a directory | the system temp directory | per call | harness | `test` | Root of the artifact directories `fastvideo-core`'s GPU tests write. |
| `FASTVIDEO_ATTENTION_BACKEND` | env, then profile | keyword | empty | per call | runtime | `attention_backend` | `SLA_ATTN` / `sla` / `sagesla` / `SAGE_SLA_ATTN` (any case) select sparse-linear attention. |
| `FASTVIDEO_ATTN_FP8` | env, then profile | text: `0`, `1` / `all`, `dense` or `vsa` (H3 technique adapter) | the recipe's / profile's | per load | H3 | `attention` | FP8 attention ops (opt-in, lossy; H3). |
| `FASTVIDEO_ATTN_PROBS_BF16` | env, then profile | flag | `1` | once per process | runtime | `activation_precision` | Attention probabilities in bf16 (default on); `0` keeps them f32. |
| `FASTVIDEO_ATTN_SAGE` | env, then profile | text: `2` / `on` force, `0` / `off` disable, empty = unset (trimmed, any case); any other value turns it off with a one-time log line | unset | once per process | runtime | `attention_backend` | SageAttention-2 (INT8 QK + FP8 PV) for dense attention; unset leaves the dispatch rule. |
| `FASTVIDEO_ATTN_SAGE_MIN_SEQ` | env, then profile | usize | `6144` | once per process | runtime | `attention_backend` | Shortest sequence SageAttention runs on. |
| `FASTVIDEO_BF16` | env, then profile | flag | `1` | once per process | runtime | `ffn_precision` | cuBLAS tf32/bf16 compute. On unless set to `0`. |
| `FASTVIDEO_BF16_ACT` | env, then profile | flag: set: a default-on flag | on for H3, LTX-2 and Wan on a GPU, f32 on the CPU | once per process | runtime | `activation_precision` | DiT activations (and residual) as bf16, as the reference runs them. On by default for H3, LTX-2 and Wan on a GPU (Wan's UMT5 and VAE stay f32, FastVideo's fp32 precisions); `0` restores f32, `1` forces bf16 for every model. CPU runs keep f32. FastWan 1.3B 480x832x81 on RTX PRO 6000: denoise 2.50 → 1.96 s, LPIPS 0.107 vs f32 (see [ports/wan.md](ports/wan.md)). |
| `FASTVIDEO_CACHE` | env | text: a directory (empty = unset) | `$XDG_CACHE_HOME/fastvideo`, else `~/.cache/fastvideo` | per load | LTX-2 | `cache` | Root of the LTX text-conditioning cache (`<root>/ltx2-text`). |
| `FASTVIDEO_CONV3D` | env, then profile | keyword | `auto` | once per process | runtime | `kernel_choice` | 3D convolution kernel (profile `[kernels] conv3d`). |
| `FASTVIDEO_CONV3D_TIMED` | env, then profile | flag | `0` | once per process | runtime | `debug` | Time every 3D convolution and log it. |
| `FASTVIDEO_CONV_BWD_DATA_NONDETERMINISTIC` | env, then profile | flag | `0` | per call | runtime | `debug` | Allow cuDNN's non-deterministic backward-data convolution algorithms. |
| `FASTVIDEO_COSMOS3_GEN` | env | text: `auto` / empty, `resident`, `parked` / `park` (trimmed); anything else is an error | `auto` | per load | Cosmos3 | `residency` | Cosmos3 gen tower placement: `auto` parks it when the free memory does not cover it plus the decode headroom. |
| `FASTVIDEO_COSMOS3_MLP_CHUNK` | env, then profile | usize: at least 1 | `8192` | per call | Cosmos3 | `memory` | Cosmos3 MLP rows per chunk. |
| `FASTVIDEO_COSMOS3_OFFICIAL` | env | text: `1` or `official` (trimmed, any case) | `0` | per generate | Cosmos | `canvas` | Cosmos3 canvas: the published 1280×720 / 189f / 35-step / guidance-6 contract. |
| `FASTVIDEO_COSMOS3_UND` | env | text: `resident` (trimmed) keeps it resident | unset | per load | Cosmos3 | `residency` | Cosmos3 understanding tower kept resident. |
| `FASTVIDEO_COSMOS_SOL` | env | text: `1` or `teacache` (trimmed, any case) | `0` | per generate | Cosmos | `step_output` | Cosmos TeaCache. |
| `FASTVIDEO_CUDNN_SDPA_CFG` | env, then profile | usize | `0` | per call | runtime | `kernel_choice` | cuDNN SDPA engine config index (0: the heuristic's first). |
| `FASTVIDEO_CUDNN_SDPA_GRAPH` | env, then profile | keyword | `auto` | per call | runtime | `kernel_choice` | cuDNN SDPA graph capture: `auto`, or forced on / off. |
| `FASTVIDEO_CUDNN_SDPA_NONDETERMINISTIC` | env, then profile | flag | `0` | once per process | runtime | `debug` | Allow cuDNN SDPA's non-deterministic algorithms. |
| `FASTVIDEO_DC_SM103` | env, then profile | flag | `0` | per call | runtime | `kernel_choice` | Build the data-center attention kernels for `compute_103a` on sm_103. |
| `FASTVIDEO_DEVICE_BUDGET_GIB` | env | text: GiB, fractions allowed (trimmed); empty, `0` or `off` = no budget; an unparsable value is logged and ignored | unset | once per process | runtime | `memory` | Plan and run as if the card had this many GiB. A phase that peaks above it fails the run. |
| `FASTVIDEO_DEVICE_STATS` | env, then profile | flag | `0` | per generate | Wan | `trace` | Log device memory statistics after a Wan generate. |
| `FASTVIDEO_DIGEST` | env, then profile | flag | `0` | once per process | runtime | `debug` | Log a digest of every dumped tensor. |
| `FASTVIDEO_DIGEST_WEIGHTS` | env, then profile | flag | `0` | once per process | runtime | `debug` | Log a digest of every loaded weight (with `FASTVIDEO_DIGEST`). |
| `FASTVIDEO_DIT_OFFLOAD` | env, then profile | text: `auto`, `resident`, `streamed` (an unknown value is an error) | `auto` | per load | runtime | `residency` | LTX-2 / H3 DiT blocks: `auto` (default), `resident`, or `streamed`. Wan DiTs: `streamed` streams the blocks layer by layer (any other value: resident, or the A14B plan of `FASTVIDEO_WAN_MOE`). |
| `FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD` | env, then profile | usize: clamped to 1..=8 | `1` | per load | runtime | `residency` | Streamed DiT blocks copied ahead of the one computing. |
| `FASTVIDEO_DUMP_DIR` | env | text: a directory (empty = unset) | unset | once per process | runtime | `debug` | Dump named intermediate tensors here (parity runs). |
| `FASTVIDEO_DUMP_OPS` | env | text: block indices whose ops are dumped | `0` | once per process | runtime | `debug` | Blocks whose per-op tensors are dumped. |
| `FASTVIDEO_FFMPEG` | env | text: a program | `ffmpeg` | per call | LTX-2 | `io` | ffmpeg binary for the LTX image-conditioning recompression. |
| `FASTVIDEO_FLASH_KERNEL` | env, then profile | keyword | `auto` | per call | runtime | `kernel_choice` | Dense attention kernel (profile `[kernels] dense_attention`). |
| `FASTVIDEO_FLUX1_DUMMY_TEXT` | env | text: `1`, `true` or `TRUE` (exact) | `0` | per load | FLUX.1 | `text_encoder` | FLUX.1: zero text embeddings instead of the T5/CLIP encoders (smoke runs). |
| `FASTVIDEO_FLUX1_TEXT_LEN` | env | text: a `usize` (not trimmed), at least 1; unparsable = default | the model's T5 length | per load | FLUX.1 | `text_encoder` | FLUX.1 T5 sequence length. |
| `FASTVIDEO_FLUX2_DUMMY_TEXT` | env | text: `1`, `true` or `TRUE` (exact) | `0` | per load | FLUX.2 | `text_encoder` | FLUX.2: zero text embeddings instead of the text encoder (smoke runs). |
| `FASTVIDEO_FLUX2_TEXT_LEN` | env | text: a `usize` (not trimmed), at least 1; unparsable = default | the model's text length | per load | FLUX.2 | `text_encoder` | FLUX.2 text sequence length. |
| `FASTVIDEO_FP8` | env, then profile | flag | `0` | once per process | runtime | `ffn_precision` | FastVideo W8A8 tensorwise FP8 recipe on every linear (Cosmos3 reads it per load and per generate). |
| `FASTVIDEO_GPU_SMOKE` | env | text: `1` runs the test | unset | per call | harness | `test` | Opt in to `fastvideo-core`'s 1.3B CUDA smoke test. |
| `FASTVIDEO_GPU_TRACE` | env, then profile | flag | `0` | once per process | runtime | `trace` | CUPTI trace of one denoise step (kernel time by category, GPU idle). |
| `FASTVIDEO_GPU_TRACE_DECODE` | env, then profile | flag | `0` | once per process | runtime | `trace` | CUPTI trace of a whole video decode (kernel time by category, GPU idle). |
| `FASTVIDEO_GPU_TRACE_STEP` | env, then profile | usize | `1` | per call | runtime | `trace` | Denoise step `FASTVIDEO_GPU_TRACE` traces. |
| `FASTVIDEO_H2D_TRACE_BYTES` | env | text: a `usize` (not trimmed); unset or unparsable = off | unset | once per process | runtime | `trace` | Log host-to-device copies of at least this many bytes. |
| `FASTVIDEO_H2D_TRACE_SKIP` | env, then profile | usize | `0` | per call | runtime | `trace` | Host-to-device copies skipped before `FASTVIDEO_H2D_TRACE_BYTES` logs. |
| `FASTVIDEO_H3_AFFINE` | env, then profile | keyword: bit list | empty | once per process | H3 | `kernel_fusion` | H3 affine-fusion bits (`fv-gpucheck --h3-affine`). |
| `FASTVIDEO_H3_DIT_TREE` | env | text: `auto` / empty, `off` / `0` / `false` / `no`, `require` / `1` / `on` / `true` / `yes` (trimmed, any case); anything else is an error | `auto` | per load | H3 | `weights` | Pre-quantized H3 DiT tree: use it when valid, never, or require it. |
| `FASTVIDEO_H3_DIT_TREE_DIR` | env | text: a directory (trimmed; empty = default) | `<root>/transformer_prequant_<recipe>_<quant>` | per load | H3 | `weights` | Where the pre-quantized H3 DiT tree lives. |
| `FASTVIDEO_H3_FFN_FP8` | env, then profile | flag | `0` | per load | H3 | `ffn_precision` | Retired flag: set, the H3 load refuses and names `FASTVIDEO_H3_QUANT`. |
| `FASTVIDEO_H3_FUSE` | env, then profile | flag | `1` | once per process | H3 | `kernel_fusion` | H3 fused MXFP8 activation and residual-plus-norm kernels (bit-identical); `0` runs the separate ops. |
| `FASTVIDEO_H3_GLUE` | env, then profile | flag | `1` | once per process | H3 | `kernel_fusion` | H3 glue kernels between the fused ops (bit-identical); `0` runs the separate ops. |
| `FASTVIDEO_H3_GRAPHS` | env, then profile | flag | `0` | per call | H3 | `cuda_graphs` | CUDA-graph capture of the H3 denoise step. |
| `FASTVIDEO_H3_LORA_DEVICE` | env, then profile | flag | `1` | once per process | H3 | `lora` | Fuse H3 LoRAs on the device. |
| `FASTVIDEO_H3_LORA_VERIFY` | env, then profile | flag | `0` | once per process | H3 | `debug` | Check the device LoRA fusion against the host. |
| `FASTVIDEO_H3_LTX_ADAPTER` | env | text: a path | found beside the H3 weights | per load | H3 | `weights` | Spark bridge LTX adapter checkpoint. |
| `FASTVIDEO_H3_NVFP4_ACT` | env, then profile | text: `static` / empty or `dynamic` (trimmed); anything else is an error | `static` | per load | H3 | `ffn_precision` | ModelOpt NVFP4 DiT: activation scales from the checkpoint or measured per call. |
| `FASTVIDEO_H3_NVFP4_ATTN` | env, then profile | text: `1` / `on` / `true` / `nvfp4`, or `0` / `off` / `false` / `bf16` / empty (trimmed); anything else is an error | `0` | per load | H3 | `ffn_precision` | ModelOpt NVFP4 DiT: attention linears in NVFP4 too. |
| `FASTVIDEO_H3_QUANT` | env, then profile | text: `mxfp8`, `w8a8` or `off` (an unknown value is an error) | `mxfp8` on sm_100+ GPUs, the recipe's scoped default (VDN-H3: bf16) | per load | H3 | `ffn_precision` | H3 FP8 linear recipe: `mxfp8` (Sol-H3, the default on sm_100+ GPUs), `w8a8` (FastVideo) or `off` (bf16). |
| `FASTVIDEO_H3_SOL_ATTN` | env, then profile | text: `off`, `rtx`, `engine`, `spark` (H3 technique adapter) | the recipe's route | per load | H3 | `attention_backend` | H3 Sol-Attn route: `spark` or `rtx`. |
| `FASTVIDEO_H3_SOL_CACHE` | env, then profile | text: `teacache` or `0` (H3 technique adapter) | unset | per load | H3 | `step_output` | H3 RTX TeaCache. |
| `FASTVIDEO_H3_STEP_SYNC` | env, then profile | flag | `0` | per call | H3 | `debug` | Synchronize after every H3 denoise step (timing). |
| `FASTVIDEO_H3_UPSCALER` | env | text: a file | found beside the H3 weights | per load | H3 | `weights` | Spark bridge upscaler checkpoint. |
| `FASTVIDEO_H3_VAE_AB` | env, then profile | flag | `0` | once per process | H3 | `debug` | Re-decode with the bf16 H3 VAE and log the difference. |
| `FASTVIDEO_H3_VAE_CHECK` | env, then profile | flag | `0` | once per process | H3 | `debug` | Decode the first H3 VAE tile batch both ways and log the error. |
| `FASTVIDEO_H3_VAE_FUSED` | env, then profile | flag | `1` | once per process | H3 | `kernel_fusion` | H3 VAE fused kernels. |
| `FASTVIDEO_H3_VAE_GEMM` | env, then profile | keyword | `cublas` | once per process | H3 | `kernel_choice` | H3 VAE GEMM backend. |
| `FASTVIDEO_H3_VAE_MXFP8` | env, then profile | text: `0` or `1` (an unknown value is an error) | MXFP8 on sm_100+ for the measured decoder depths | per load | H3 | `video_decoder` | H3 VAE decoder linears as MXFP8. |
| `FASTVIDEO_H3_VAE_MXFP8_ALGO` | env, then profile | usize | `0` | per call | H3 | `kernel_choice` | H3 VAE MXFP8 GEMM algorithm index. |
| `FASTVIDEO_H3_VAE_NVFP4` | env, then profile | text: `0` or `1` (an unknown value is an error) | the tree's: NVFP4 when its `vae/` carries NVFP4 linears | per load | H3 | `video_decoder` | H3 VAE decoder linears as published ModelOpt NVFP4; `0` decodes with h3-base's bf16 `vae/`. |
| `FASTVIDEO_H3_VAE_NVFP4_ACT` | env, then profile | text: `static` / empty or `dynamic` (an unknown value is an error) | `static` | per load | H3 | `video_decoder` | H3 NVFP4 VAE activation scales: calibrated `input_scale` or measured per call. |
| `FASTVIDEO_H3_VAE_NVFP4_REF` | env, then profile | text: a `vae/` directory | unset | per load | H3 | `debug` | With `FASTVIDEO_H3_VAE_CHECK`, also decode through this bf16 decoder. |
| `FASTVIDEO_H3_VAE_SDPA` | env, then profile | keyword | `auto` | once per process | H3 | `kernel_choice` | H3 VAE attention kernel. |
| `FASTVIDEO_H3_VAE_TILE_BATCH` | env, then profile | usize: at least 1 | `8` | per load | H3 | `memory` | H3 VAE tiles decoded per batch. |
| `FASTVIDEO_H3_VAE_VEC` | env, then profile | flag | `1` | once per process | H3 | `kernel_choice` | H3 VAE vectorized kernels. |
| `FASTVIDEO_H3_VDN` | env | text: a path | unset | per load | H3 | `weights` | VDN-H3 checkpoint when `--vdn` is not given. |
| `FASTVIDEO_H3_VDN_SHARED` | env | text: a path (empty = unset) | unset | per load | H3 | `weights` | VDN-H3 shared checkpoint when the option is not given. |
| `FASTVIDEO_H3_VSA_FP4` | env, then profile | flag: set: a default-on flag | on where the NVFP4 tier has it | per load | H3 | `attention` | H3 VSA attention in NVFP4. |
| `FASTVIDEO_H3_VSA_FP4_P` | env, then profile | keyword | `two-level` | per call | H3 | `attention` | `single`: upstream's single-level P quantization (default two-level). |
| `FASTVIDEO_HUNYUAN15_OFFICIAL` | env | text: `1` or `official` (trimmed, any case) | `0` | per generate | HunyuanVideo 1.5 | `canvas` | 720p / 129f / 50-step canvas. |
| `FASTVIDEO_HUNYUAN15_SOL` | env | text: `1` or `teacache` (trimmed, any case) | `0` | per generate | HunyuanVideo 1.5 | `step_output` | Logs the 13B TeaCache gap and stays dense. |
| `FASTVIDEO_INJECT_COND` | env | text: H3: on only for `1`; LTX-2: on unless `0` (not trimmed) | off for H3, on for LTX-2 | per call | runtime | `debug` | With `FASTVIDEO_INJECT_DIR`: inject the reference's condition tensors. |
| `FASTVIDEO_INJECT_DIR` | env | text: a directory (empty = unset) | unset | once per process | runtime | `debug` | Inject reference tensors from this directory (parity runs). |
| `FASTVIDEO_INJECT_PIXELS` | env | text: on unless `0` (not trimmed) | `1` | per call | LTX-2 | `debug` | With `FASTVIDEO_INJECT_DIR`: inject the reference's conditioning pixels. |
| `FASTVIDEO_INJECT_STAGE2` | env | text: on unless `0` (not trimmed) | `1` | per call | LTX-2 | `debug` | With `FASTVIDEO_INJECT_DIR`: inject the reference's stage-2 entry latents. |
| `FASTVIDEO_INJECT_TEXT` | env | text: on unless `0` (not trimmed) | `1` | per call | runtime | `debug` | With `FASTVIDEO_INJECT_DIR`: inject the reference's text embeddings. |
| `FASTVIDEO_KERNELS` | env, then profile | keyword | `auto` | per load | runtime | `kernel_choice` | `nvrtc` compiles the kernels with NVRTC instead of the embedded cubins. |
| `FASTVIDEO_LINGBOT_MOE_CHUNK` | env, then profile | usize: at least 1 | `32768` | per call | LingBot | `memory` | LingBot MoE tokens per chunk. |
| `FASTVIDEO_LINGBOT_OFFICIAL` | env | text: `1` or `official` (trimmed, any case) | `0` | per generate | LingBot | `canvas` | 832×480 / 121f / 40-step canvas. |
| `FASTVIDEO_LINGBOT_REFINE` | env | text: `0` or `off` (exact) stop after the base stage | unset | per generate | LingBot | `canvas` | LingBot 1080p refiner stage. |
| `FASTVIDEO_LINGBOT_RESIDENCY` | env | text: `both` or `resident` (trimmed); anything else swaps | `swap` | per load | LingBot | `residency` | LingBot experts: both resident or swapped. |
| `FASTVIDEO_LINGBOT_ROUTER` | env, then profile | keyword | `device` | once per process | LingBot | `kernel_choice` | LingBot MoE router on the device or the host. |
| `FASTVIDEO_LINGBOT_ROUTER_CHECK` | env, then profile | flag | `0` | per call | LingBot | `debug` | Check the device router against the host. |
| `FASTVIDEO_LINGBOT_SOL` | env | text: `0` / `off` / `baseline`, `1` / `sol` / `fullopt`, `cache` / `easycache`, `pisa`, or a comma list (trimmed, any case) | `baseline` | per generate | LingBot | `step_output` | LingBot EasyCache and refiner PISA. |
| `FASTVIDEO_LLM_PREFETCH` | env | text: on unless `0` (not trimmed) | `1` | per call | text encoders | `loader` | Prefetch the next text-encoder layer while one computes. |
| `FASTVIDEO_LOG` | env | text: `0` / `off` / `false` / `quiet` / `error`, `2` / `debug` / `trace` / `verbose`, else info (trimmed, any case) | `info` | once per process | runtime | `log` | Runtime log level. |
| `FASTVIDEO_LTX2_AUDIO_VAE` | env | text: a file or directory | `audio_vae/` or `ltx-2.5-audio-vae-bf16.safetensors` | per load | LTX-2 | `weights` | Audio encoder file or directory. |
| `FASTVIDEO_LTX2_DIT` | env | text: a path | `$FASTVIDEO_LTX2_WEIGHTS/transformer` | per generate | H3 | `weights` | Override the Spark refiner's DiT path. |
| `FASTVIDEO_LTX2_FBCACHE` | env | text: `1` or `fbcache` (trimmed, any case) | `0` | per generate | LTX-2 | `step_output` | LTX-2.5 stage-1 first-block cache. |
| `FASTVIDEO_LTX2_HOST_ROPE` | env, then profile | flag | `0` | per call | LTX-2 | `debug` | Compute the LTX-2 RoPE tables on the host. |
| `FASTVIDEO_LTX2_HQ` | env | text: `1`, `hq` or `official` (trimmed, any case) | `0` | per generate | LTX-2 | `canvas` | 2.3-base 15+3 HQ contract (`fv-gpucheck ltx2 gen --model-version 2.3 --hq` runs it on the dev DiT). |
| `FASTVIDEO_LTX2_IC_LORA` | env | text: a file (empty = unset) | unset | per load | LTX-2 | `lora` | IC-LoRA file when the request names none. |
| `FASTVIDEO_LTX2_LORA` | env | text: a file | a known filename beside the weights | per load | LTX-2 | `lora` | Distilled LoRA file. |
| `FASTVIDEO_LTX2_LORA_BASE` | env | text: `host` or `device` (trimmed; another value is an error) | the load option | per load | LTX-2 | `residency` | Where the LoRA base weights stay during the fuse. |
| `FASTVIDEO_LTX2_MIDPOINT_PRUNE` | env | text: `1`, `feat_norm` or `prune` (trimmed, any case) | `0` | per generate | LTX-2 | `token_set` | LTX-2.3 stage-2 feature-norm prune. |
| `FASTVIDEO_LTX2_SAVE_LATENTS` | env | text: a path prefix (empty = unset) | unset | per generate | LTX-2 | `debug` | Save the decoded video latents to `<prefix>.f32` / `.shape` (for `ltx2 vae-bench --latents`). |
| `FASTVIDEO_LTX2_STAGE1_CACHE` | env | text: `1`, the preset name or `scsp` (trimmed, any case) | `0` | per generate | LTX-2 | `step_output` | LTX-2.3 stage-1 SCSP (res2s calls 16–28 of 29). |
| `FASTVIDEO_LTX2_TAE_WEIGHTS` | env, then profile | text: a path (empty = unset) | unset | per load | LTX-2 | `video_decoder` | LTX tiny decoder (TAEHV) weights. |
| `FASTVIDEO_LTX2_TEXT` | env | text: `resident`, `streamed` or `auto` (trimmed, any case; empty = unset); anything else is an error | the load option's text residency | per load | LTX-2 | `residency` | Gemma text encoder placement. |
| `FASTVIDEO_LTX2_TEXT_FP8` | env | text: on only for `1` (trimmed) | `0` | per load | LTX-2 | `text_encoder` | Gemma from the pre-quantized FP8 tree. |
| `FASTVIDEO_LTX2_TEXT_WARM` | env | text: on unless `0` (trimmed) | `1` | per load | LTX-2 | `text_encoder` | Open the Gemma weights ahead of the first encode. |
| `FASTVIDEO_LTX2_UPSAMPLER` | env | text: a file or directory | the pack's upsampler | per load | LTX-2 | `weights` | Spatial-upsampler file or folder in place of the pack's (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors` for the 2.3 HQ cell). |
| `FASTVIDEO_LTX2_VAE_CHUNK` | env, then profile | usize | `8` | per call | LTX-2 | `memory` | LTX-2 streaming VAE frames per chunk. |
| `FASTVIDEO_LTX2_VAE_LATENT_CHUNK` | env, then profile | usize: at least 1 | `1` | per call | LTX-2 | `memory` | LTX-2 VAE latent frames per decode chunk. |
| `FASTVIDEO_LTX2_WEIGHTS` | env | text: a directory | unset | per generate | H3 | `weights` | LTX-2.5 root for the Spark refiner. DiT defaults to `weights/transformer`. |
| `FASTVIDEO_LTX_FUSE` | env, then profile | flag | `1` | once per process | LTX-2 | `kernel_fusion` | LTX block fusions (bit-identical); `0` runs the separate ops. |
| `FASTVIDEO_LTX_OFFLOAD` | env, then profile | text: `none` / `off` / `0` / empty, `cpu` / `on` / `1` (trimmed, any case); `disk` and others are errors | `none` | per load | LTX-2 | `residency` | `cpu` gives sol-engine's `--offload cpu` placement: the DiT blocks stream from pinned host memory through 2 device slots, Gemma streams per layer, the connectors, upsampler and VAEs are on the device only around their calls. Byte-identical frames. `ltx2 gen --offload` overrides it. See [ports/ltx25.md](ports/ltx25.md#offload-placement-fastvideo_ltx_offloadcpu). |
| `FASTVIDEO_LTX_VAE_CHECK` | env, then profile | flag | `0` | once per process | LTX-2 | `debug` | Decode the process's first LTX VAE tile both ways and log rel_l2 and both times. |
| `FASTVIDEO_LTX_VAE_CHUNK_FRAMES` | env, then profile | usize: at least 1 | `16` | per call | LTX-2 | `memory` | Output frames per channels-last conv launch. |
| `FASTVIDEO_LTX_VAE_CHUNK_MB` | env, then profile | usize: at least 1 | `1024` | per call | LTX-2 | `memory` | MB per channels-last conv launch. |
| `FASTVIDEO_LTX_VAE_CONV_ALGO` | env, then profile | keyword | `heuristic` | per call | LTX-2 | `kernel_choice` | `tune`: time every cuDNN forward algorithm per conv shape (default: cuDNN's heuristic). |
| `FASTVIDEO_LTX_VAE_FAST` | env, then profile | flag | `1` | once per process | LTX-2 | `video_decoder` | LTX conv VAE tiled decode on the channels-last bf16 decoder (default on); `0` restores the f32 streaming decoder. |
| `FASTVIDEO_MMAUDIO_INJECT` | env | text: comma-separated stage names | unset | per call | MMAudio | `debug` | MMAudio stages whose reference tensors are injected. |
| `FASTVIDEO_MMAUDIO_WEIGHTS` | env | text: a directory | `MMAUDIO_MODEL_PATH`, else under `FV_WEIGHTS` | per load | MMAudio | `weights` | MMAudio weights root. |
| `FASTVIDEO_NVFP4` | env, then profile | text: `0` / `off` / `false` / `none` / `no`, `1` / `static_6` / `te` / ..., `static_4`, `mse` / `4o6` (any case); unknown = off | `0` | per load | runtime | `ffn_precision` | W4A4 dequant (`1` → `static_6`; `mse` for FourOverSix). LTX-2: the video FFN on NVFP4 tensor cores (`static_6` / `static_4`). |
| `FASTVIDEO_NVFP4_COSMOS_STEPS` | reserved (not read) | text: `1` / `true` / `on` / `yes` (trimmed, any case) | `0` | per load | Cosmos | `ffn_precision` | Reserved: restrict NVFP4 to the middle Cosmos steps (the parser exists, no reader yet). |
| `FASTVIDEO_NVFP4_LTX_VIDEO_FFN` | reserved (not read) | text: `1` / `true` / `on` / `yes` (trimmed, any case) | `0` | per load | LTX-2 | `ffn_precision` | Reserved: restrict NVFP4 to the LTX video FFN (the parser exists, no reader yet). |
| `FASTVIDEO_NVFP4_OXIDE_GEMM` | env, then profile | text: `1` / `true` / `on` / `yes` (trimmed, any case) | `0` | per call | runtime | `kernel_choice` | Tile-IR W4A4 GEMM (profile `[kernels] nvfp4_gemm`); off until it beats cuBLAS bf16. |
| `FASTVIDEO_NVRTC_PTX` | env, then profile | flag | `0` | per load | runtime | `kernel_choice` | Have NVRTC emit PTX instead of SASS. |
| `FASTVIDEO_PISA_FIRST_ORDER` | env, then profile | keyword | `1` | per call | runtime | `attention` | `0` drops PISA's first-order term. |
| `FASTVIDEO_PISA_KERNEL` | env, then profile | keyword | `fused` | per call | runtime | `kernel_choice` | `legacy` runs the unfused PISA kernel. |
| `FASTVIDEO_PLUG_ROOT` | env | text: a directory | `<base_root>/../longlive-plug` | per load | Wan | `lora` | Where LongLive-Plug adapters live. |
| `FASTVIDEO_PNG` | env | text: `inline`, `off` / `0` / `none`, else deferred (trimmed, any case) | `deferred` | per generate | runtime | `io` | PNG frame writing: deferred to a writer thread, inline, or off. |
| `FASTVIDEO_PNG_BUFFER_GIB` | env | text: GiB as `f64` (trimmed), negative = 0 | half the available host memory, else 8 GiB | per generate | runtime | `io` | Host RAM the deferred PNG frames may hold. |
| `FASTVIDEO_PNG_SYNC` | env, then profile | flag | `1` | per generate | runtime | `io` | Wait for the PNG writer before returning. |
| `FASTVIDEO_PREFETCH` | env | text: on unless `0` (trimmed) | `1` | per load | loader | `loader` | Weight read-ahead pool. |
| `FASTVIDEO_PREFETCH_CHUNK_MB` | env | text: a `usize` (trimmed), clamped to 1..=1024 | `16` | per load | loader | `loader` | Read-ahead chunk size. |
| `FASTVIDEO_PREFETCH_MODE` | env | text: `cache` (trimmed) forces page-cache mode | unset | once per process | loader | `loader` | Read-ahead into owned memory while `MemAvailable` allows, else the page cache. |
| `FASTVIDEO_PREFETCH_THREADS` | env | text: a `usize` (trimmed) | 16; `FASTVIDEO_STREAMING_PREFETCH_THREADS` while the weights stream | per load | loader | `loader` | Read-ahead threads. |
| `FASTVIDEO_PREFETCH_WINDOW_GB` | env | text: GB as `f64` (trimmed) | the smaller of 24 GB and 40% of `MemAvailable` | per load | loader | `loader` | Read-ahead window. |
| `FASTVIDEO_PROFILE` | env, then profile | flag | `0` | once per process | runtime | `trace` | Per-op timing statistics. |
| `FASTVIDEO_RESIDENT` | env, then profile | flag | `1` | once per process | runtime | `residency` | Keep tensors on the device between ops (default on); `0` is refused with a live CUDA device. |
| `FASTVIDEO_SANA_EASYCACHE` | env | text: an EasyCache threshold (a number) | unset | per load | SANA-Video | `step_output` | SANA-Video EasyCache threshold. |
| `FASTVIDEO_SANA_LINATTN_BF16` | env | text: a SANA-Video flag | the `FASTVIDEO_SANA_OPT` arm's | per load | SANA-Video | `activation_precision` | SANA-Video linear attention in bf16. |
| `FASTVIDEO_SANA_OPT` | env | text: `baseline` or `full` | `baseline` | per load | SANA-Video | `step_output` | SANA-Video optimization arm. |
| `FASTVIDEO_SANA_QKV_MERGE` | env | text: a SANA-Video flag | the `FASTVIDEO_SANA_OPT` arm's | per load | SANA-Video | `kernel_fusion` | SANA-Video merged QKV projection. |
| `FASTVIDEO_SAVE_MP4` | env | text: `1` or `true` (any case, not trimmed) | `0` | per generate | harness | `io` | Mux frames with ffmpeg. |
| `FASTVIDEO_SDPA` | env, then profile | keyword: `dense` (default), `cublas` or `flash`; `host` and `sparse` are retired and refused | `dense` | once per process | runtime | `attention_backend` | Dense attention backend for every family (`rt::sdpa_rule`): the dense rule's pick, or a forced `cublas` / `flash` route. |
| `FASTVIDEO_SDPA_AUTO` | env, then profile | keyword | `rule` | once per process | runtime | `kernel_choice` | How the dense attention backend is chosen. |
| `FASTVIDEO_SDPA_CHUNK` | env | text: a `usize` (not trimmed), at least 64 | by device | once per process | runtime | `memory` | Query rows per SDPA chunk. |
| `FASTVIDEO_SLA_BLKK` | env, then profile | usize: at least 1 | `64` | per call | runtime | `attention` | SLA key block. |
| `FASTVIDEO_SLA_BLKQ` | env, then profile | usize: at least 1 | `128` | per call | runtime | `attention` | SLA query block. |
| `FASTVIDEO_SLA_TOPK` | env, then profile | f32: clamped to 0.01..=1 | `0.1` | per call | runtime | `attention` | SLA top-k ratio. |
| `FASTVIDEO_SOL_KERNEL` | env, then profile | keyword | `auto` | per call | runtime | `kernel_choice` | Sol attention kernel (profile `[kernels] sol_attention`). |
| `FASTVIDEO_SOL_SPLITS` | env, then profile | keyword | `auto` | per call | runtime | `kernel_choice` | Sol attention split count. |
| `FASTVIDEO_SPLIT_ROWS` | env, then profile | flag | `1` | once per process | runtime | `kernel_fusion` | Row-split activation kernels (profile `kernel_fusion.split_rows`). |
| `FASTVIDEO_SP_WORLD` | env, then profile | usize: at least 1 | `1` | once per process | runtime | `parallel` | Sequence-parallel world size (`fastvideo --num-gpus` sets it). |
| `FASTVIDEO_STAGED_UPLOAD` | env, then profile | flag | `1` | once per process | runtime | `loader` | Upload weights through pinned staging buffers. |
| `FASTVIDEO_STREAMING_FETCH` | env | text: on only for `1` (trimmed) | `0` | per load | loader | `loader` | The boot started this process while the weights were still being fetched. |
| `FASTVIDEO_STREAMING_FETCH_STALE_S` | env | text: seconds as a positive finite `f64` (trimmed) | `120` | per load | loader | `loader` | A fetch whose status file is older than this is dead. |
| `FASTVIDEO_STREAMING_FETCH_STATUS` | env | text: a file (empty = unset) | unset | per load | loader | `loader` | The fetcher's status JSON. |
| `FASTVIDEO_STREAMING_FETCH_TIMEOUT_S` | env | text: seconds as a positive finite `f64` (trimmed) | `3600` | per load | loader | `loader` | The longest a single wait for streamed weights may take. |
| `FASTVIDEO_STREAMING_PREFETCH_THREADS` | env | text: a `usize` (trimmed) | `4` | per load | loader | `loader` | Read-ahead threads while the weights stream. |
| `FASTVIDEO_TAEH3_WEIGHTS` | env, then profile | text: a path (empty = unset) | unset | per load | H3 | `video_decoder` | H3 tiny decoder (TAEH3) weights. |
| `FASTVIDEO_TAEHV_CHUNK` | env, then profile | usize: at least 1 | the TAE architecture's decode chunk | per call | runtime | `memory` | TAEHV decode frames per chunk. |
| `FASTVIDEO_TAEHV_ENC_CHUNK` | env, then profile | usize | 2 × the temporal downscale | per call | runtime | `memory` | TAEHV encode frames per chunk. |
| `FASTVIDEO_TAEHV_WEIGHTS` | env | text: a path (empty = unset) | unset | per load | Wan | `video_decoder` | Wan tiny decoder (TAEHV) weights. |
| `FASTVIDEO_TAE_DIR` | env | text: a directory (empty = unset) | unset | per load | Wan | `video_decoder` | Where the distilled presets look for `taew2_1.safetensors` (`scripts/gpu/fetch_taehv.sh`). |
| `FASTVIDEO_TAE_TEST_DIR` | env | text: a directory | unset | per call | harness | `test` | Real TAE checkpoints for the reference test. |
| `FASTVIDEO_TEACACHE` | env | text: `1` or `true` (any case, not trimmed) | `0` | per load | Wan | `step_output` | Wan TeaCache (FastVideo's). |
| `FASTVIDEO_TEACACHE_RET` | env | text: `1` or `true` (any case, not trimmed) | `0` | per load | Wan | `step_output` | Wan TeaCache `use_ret_steps` coefficients. |
| `FASTVIDEO_TEACACHE_THRESH` | env | text: an `f64` (not trimmed); unparsable = default | `0.08` | per load | Wan | `step_output` | Wan TeaCache threshold. |
| `FASTVIDEO_TECHNIQUES` | env | text: a profile path or compiled-in name | unset | at start-up | runtime | `techniques` | Technique profile to install when no `--techniques` flag did (`docs/techniques.md`). |
| `FASTVIDEO_TEXT_FP8_TREE` | env | text: `0` (trimmed) skips the tree | `1` | per load | text encoders | `weights` | Text encoder from its pre-quantized FP8 tree when one is valid. |
| `FASTVIDEO_TF32` | env, then profile | flag | `1` | once per process | runtime | `ffn_precision` | TF32 tensor-core math for f32 GEMMs. |
| `FASTVIDEO_VAE_CHUNK` | env, then profile | usize: at least 1 | `2` | per call | Wan | `memory` | Latent frames per Wan VAE pass. |
| `FASTVIDEO_VERIFY_UPLOAD` | env, then profile | flag | `0` | once per process | runtime | `debug` | Read back every staged upload and compare. |
| `FASTVIDEO_VSA` | env, then profile | flag: the `fastvideo` CLI's VSA-id check reads `1` / `true` (any case, not trimmed) from the environment only | `0` | once per process | runtime | `attention_backend` | Wan block-sparse video attention. |
| `FASTVIDEO_VSA_BF16` | env, then profile | flag | `1` | per call | H3 | `activation_precision` | H3 VSA in bf16. |
| `FASTVIDEO_VSA_FUSED` | env, then profile | flag | `0` | per call | runtime | `kernel_choice` | Fused scalar VSA fine kernel. |
| `FASTVIDEO_VSA_GROUP` | env, then profile | usize: H3 technique adapter: its own parse | 8 for H3, 32 for Wan | per load | runtime | `attention` | VSA block group size. |
| `FASTVIDEO_VSA_KERNEL` | env, then profile | keyword | `auto` | per call | runtime | `kernel_choice` | VSA attention kernel (profile `[kernels] vsa_attention`). |
| `FASTVIDEO_VSA_SPARSITY` | env, then profile | f64: H3 technique adapter: a number in [0, 1) | 0.8 for Wan, the recipe's for H3 | per load | runtime | `attention` | VSA sparsity, one value everywhere. |
| `FASTVIDEO_VSA_TMA` | env, then profile | keyword | `1` | per call | runtime | `kernel_choice` | `0` turns the VSA TMA kernels off. |
| `FASTVIDEO_VSA_TOPK` | env, then profile | keyword | `v2` | per call | runtime | `kernel_choice` | `v1` runs the first VSA top-k kernel. |
| `FASTVIDEO_VSA_UNION_LOG` | env, then profile | flag | `0` | per call | H3 | `debug` | Log the VSA block-union statistics. |
| `FASTVIDEO_VSA_WINDOW` | retired (not read) | text: not read | unset | per call | runtime | `attention` | Retired with the host block-sparse prototype (`FASTVIDEO_SDPA=sparse`, plan B5): not read. `FASTVIDEO_SDPA=sparse` is refused, and the refusal names this variable. |
| `FASTVIDEO_WAN_AUDIO` | env | text: `mmaudio` or `1` (trimmed) | unset | per generate | MMAudio | `audio` | MMAudio soundtrack for Wan clips. |
| `FASTVIDEO_WAN_AUDIO_CFG` | env | text: parsed as the preset's number (not trimmed); unparsable = default | the MMAudio preset's | per generate | MMAudio | `audio` | MMAudio guidance. |
| `FASTVIDEO_WAN_AUDIO_PROMPT` | env | text: text | the video prompt | per generate | MMAudio | `audio` | MMAudio prompt. |
| `FASTVIDEO_WAN_AUDIO_SEED` | env | text: an integer (not trimmed); unparsable = default | the request's seed | per generate | MMAudio | `audio` | MMAudio seed. |
| `FASTVIDEO_WAN_AUDIO_STEPS` | env | text: a `usize` (not trimmed); unparsable = default | the MMAudio preset's | per generate | MMAudio | `audio` | MMAudio steps. |
| `FASTVIDEO_WAN_CAUSAL_AR` | env, then profile | flag | `1` | once per process | Wan | `attention` | SF-Wan autoregressive KV cache. |
| `FASTVIDEO_WAN_CAUSAL_FLASH` | env, then profile | flag | `1` | once per process | Wan | `kernel_choice` | SF-Wan block-causal self-attention on the masked flash kernel (default on); `0` materializes the `[S, S]` mask through `sdpa_composed`. |
| `FASTVIDEO_WAN_CAUSAL_FPB` | env, then profile | usize | the config's frames per block | per load | Wan | `attention` | SF-Wan frames per block. |
| `FASTVIDEO_WAN_COND_CACHE` | env, then profile | flag: `fv-gpucheck wan` reports the environment's value only | `1` | once per process | Wan | `cache` | `0`: recompute text K/V, text and time embeddings every forward. |
| `FASTVIDEO_WAN_DIT_LAZY` | env, then profile | flag | on for two experts or a streamed DiT | per load | Wan | `residency` | Open the Wan DiT weights lazily. |
| `FASTVIDEO_WAN_EASYCACHE_COOLDOWN` | env | text: a `usize` (not trimmed), else `WAN22_EASYCACHE_COOLDOWN_STEPS` | the EasyCache profile's | per generate | Wan | `step_output` | Wan EasyCache cooldown steps. |
| `FASTVIDEO_WAN_EASYCACHE_PROFILE` | env | text: a profile name, else `WAN22_EASYCACHE_PROFILE` | unset | per generate | Wan | `step_output` | Wan EasyCache parameter profile. |
| `FASTVIDEO_WAN_EASYCACHE_RETAIN` | env | text: a `usize` (not trimmed), else `WAN22_EASYCACHE_RETAIN_STEPS` | the EasyCache profile's | per generate | Wan | `step_output` | Wan EasyCache retained steps. |
| `FASTVIDEO_WAN_EASYCACHE_THRESH` | env | text: an `f64` (not trimmed), else `WAN22_EASYCACHE_THRESHOLD` | the EasyCache profile's | per generate | Wan | `step_output` | Wan EasyCache threshold. |
| `FASTVIDEO_WAN_FUSE` | env, then profile | flag | `1` | once per process | Wan | `kernel_fusion` | Wan block residual + LayerNorm (+ AdaLN) as one kernel under bf16 activations (default on); `0` runs the byte-identical two-kernel form. |
| `FASTVIDEO_WAN_GRAPH` | env, then profile | flag | `1` | once per process | Wan | `cuda_graphs` | CUDA-graph capture of the Wan step. |
| `FASTVIDEO_WAN_GRAPH_CUBLAS_WS_MIB` | env, then profile | usize | `32` | per call | Wan | `memory` | cuBLAS workspace for graph-captured GEMMs. |
| `FASTVIDEO_WAN_MOE` | env | text: `auto` / empty, `both` / `resident`, `swap` / `parked` (trimmed, any case); anything else is an error | `auto` | per load | Wan | `residency` | Wan 2.2 A14B experts: `auto` (both resident when the free memory after UMT5 covers two bf16 experts plus `FASTVIDEO_WAN_MOE_HEADROOM_GIB`, else `swap`), `both`, or `swap` (both parked in pinned host memory, swapped at the 0.875 boundary). See [ports/sol-wan-ltx23.md](ports/sol-wan-ltx23.md). |
| `FASTVIDEO_WAN_MOE_HEADROOM_GIB` | env | text: GiB as a finite `f64` ≥ 0 (trimmed) | `36` | per load | Wan | `residency` | Headroom `FASTVIDEO_WAN_MOE=auto` keeps beside two experts. |
| `FASTVIDEO_WAN_PISA` | env | text: `1`, `pisa`, `true` or `on` (trimmed), else `WAN22_PISA` | `0` | per load | Wan | `attention_backend` | Wan PISA sparse attention. |
| `FASTVIDEO_WAN_QUANT` | env, then profile | text: `mxfp8`, `w8a8` or `off` (an unknown value is an error) | `mxfp8` on sm_100, `off` elsewhere | per load | Wan | `ffn_precision` | Wan DiT FP8 recipe on every block's attention + FFN linears (on sm_120 both were slower than bf16 and doubled LPIPS). |
| `FASTVIDEO_WAN_SOL_ATTN` | env | text: `1`, `sol`, `true`, `on` or `fullstack` (trimmed), else `WAN22_SOL_ATTN` | `0` | per load | Wan | `attention_backend` | Sol-Attn route: 14B (10 dense forwards, layer 0 dense) and 1.3B (layer 0 dense), Morton3D on the device. `fullstack` on the 1.3B: the 14B guards (sol-engine `wan21_fullstack_sol.toml`, the 50-step base model). |
| `FASTVIDEO_WAN_SOL_CACHE` | env | text: `easycache`, `teacache` or `taylorseer`; empty / `0` / `off` / `none` / `false` = off (trimmed, any case) | unset | per generate | Wan | `step_output` | Wan Sol step cache (lossy; off by default). `taylorseer` forecasts `proj_out` (interval 3, warmup 3, cooldown 2, order 1). |
| `FASTVIDEO_WAN_TAYLOR_COOLDOWN` | env | text: a `usize` (not trimmed), else `WAN22_TAYLOR_COOLDOWN_START` | steps − 2 | per generate | Wan | `step_output` | TaylorSeer cooldown start. |
| `FASTVIDEO_WAN_TAYLOR_INTERVAL` | env | text: a `usize` (not trimmed), else `WAN22_TAYLOR_INTERVAL` | `3` | per generate | Wan | `step_output` | TaylorSeer interval. |
| `FASTVIDEO_WAN_TAYLOR_LITE` | env | text: `0` / `false` / `off` (trimmed) are refused, else `WAN22_TAYLOR_LITE` | `1` | per generate | Wan | `step_output` | TaylorSeer lite mode (the only mode implemented). |
| `FASTVIDEO_WAN_TAYLOR_ORDER` | env | text: a `usize` (not trimmed), else `WAN22_TAYLOR_ORDER` | `1` | per generate | Wan | `step_output` | TaylorSeer order. |
| `FASTVIDEO_WAN_TAYLOR_WARMUP` | env | text: a `usize` (not trimmed), else `WAN22_TAYLOR_WARMUP` | `3` | per generate | Wan | `step_output` | TaylorSeer warmup. |
| `FASTVIDEO_WAN_TEACACHE_COEFFS` | env | text: comma-separated `f64`s (unparsable parts dropped) | `1.0,0.0` | per generate | Wan | `step_output` | Wan Sol TeaCache polynomial. |
| `FASTVIDEO_WAN_TEACACHE_END` | env | text: a `usize` (not trimmed) | steps − 2 | per generate | Wan | `step_output` | Wan Sol TeaCache last step. |
| `FASTVIDEO_WAN_TEACACHE_MAX_HITS` | env | text: a `usize` (not trimmed) | `0` | per generate | Wan | `step_output` | Wan Sol TeaCache consecutive-skip cap (0: none). |
| `FASTVIDEO_WAN_TEACACHE_PERIODIC` | env | text: a `usize` (not trimmed) | `0` | per generate | Wan | `step_output` | Wan Sol TeaCache forced refresh period (0: none). |
| `FASTVIDEO_WAN_TEACACHE_START` | env | text: a `usize` (not trimmed) | `2` | per generate | Wan | `step_output` | Wan Sol TeaCache first step. |
| `FASTVIDEO_WAN_TEACACHE_THRESH` | env | text: an `f64` (not trimmed) | `0.12` | per generate | Wan | `step_output` | Wan Sol TeaCache threshold. |
| `FASTVIDEO_WAN_TEXT_CACHE` | env | text: a directory (trimmed), or `off` / `0` / `false` / `none` | `$XDG_CACHE_HOME/fastvideo/wan-text`, else `~/.cache/fastvideo/wan-text` | per load | Wan | `cache` | UMT5 prompt cache directory, or `off`. |
| `FASTVIDEO_WAN_UNIPC_SIGMAS` | env | text: `fastvideo` / `double` / empty, `diffusers` / `single` / `sol` (trimmed, any case); anything else is an error | `fastvideo` | per load | Wan | `scheduler` | `fastvideo` (FastVideo's double-shifted flow-UniPC sigmas) or `diffusers` (Diffusers' single shift, the sol-engine Wan baselines). |
| `FASTVIDEO_WAN_VAE` | env | text: `auto`, `full`, `taehv` (trimmed, any case) | `auto` | per load | Wan | `video_decoder` | `auto` (TAEHV for distilled presets when found), `full`, `taehv`. |
| `FASTVIDEO_WEIGHTS` | env | text: a directory | `--weights`, else the Hugging Face snapshot | per load | harness | `weights` | Weights root when `--weights` is not given. |
| `FASTVIDEO_WRITER_TRACE` | env, then profile | flag | `0` | per generate | runtime | `trace` | Log the frame writer's timing. |

<!-- END GENERATED: FASTVIDEO_* settings -->

Text-conditioning cache for LTX defaults under `~/.cache/fastvideo`
(`FASTVIDEO_CACHE` overrides the root).

## Not in this tree

NVlabs/Sana `main`: SANA image, SANA-1.5, SANA-Sprint, SANA-Video, SANA-Video
2.0, LongSANA, SANA-WM, SANA-Streaming, Sol-RL.

HunyuanVideo-13B as its own pipeline. The 1.5 family is what is registered.
