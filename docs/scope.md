# Scope

Snapshot of what this tree registers and runs, as of 2026-09-23. Per-model
algorithms live in `docs/ports/`. This page is the index: families, Hub ids,
workloads, and which NVlabs/Sana `sol-engine` profiles are actually wired.

Host math is `crates/fastvideo-models`. The CUDA graph is
`crates/fastvideo-cudarc` under the same module name. `crates/fastvideo-core`
resolves a Hub id to one of those. A Mac without `nvcc` can test the host
half. `--features cuda` needs a CUDA toolchain.

## Crates

| Crate | Role |
|---|---|
| `fastvideo-ops` | Shared numeric helpers |
| `fastvideo-models` | Configs, schedules, geometry, sol-engine contracts. No CUDA |
| `fastvideo-loader` | Safetensors, lazy shards, bf16 views |
| `fastvideo-cudarc` | Pipelines, weights, NVRTC kernels. `unsafe` allowed here only |
| `fastvideo-core` | Registry, workloads, `generate` dispatch |
| `fastvideo-cli` | `generate` / `generate_av` |
| `fastvideo-gpucheck` | Staged GPU checks (`kernels`, `parity`, `clip`) |
| `fastvideo-mlx` | Apple Silicon gate. Host stubs off that target |

Shared device types (`CudaTensor`, conv, the weight map, the frame writer)
live under `cudarc/src/wan` and are used by every other model.

Workloads: `t2v`, `i2v`, `t2av`, `fl2va`, `ref2va`, `t2i`, `t2a`, `v2a`,
plus control/edit entries that carry an empty workload list in the registry.

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
| `mmaudio_large_44k_v2` | v2a, t2a | `hkchengrex/MMAudio`, `FastVideo/MMAudio-large-44k-v2-Diffusers` | Synchformer when `vfeat_extractor.*` is present |

`fastvideo-mlx` is a separate Apple Silicon gate (`mlx-rs` on aarch64). The
DiT and TAEHV graphs behind that gate are still a scaffold.

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
| Cosmos3-Super | Canvas and TeaCache (threshold 1.15, start step 10, max 3). `fp4_linear` names the middle steps. Those linears are not quantized |
| HunyuanVideo | Official canvas only. The profile's TeaCache is HunyuanVideo-13B and is not applied |
| LingBot | Official base canvas only. Cache, PISA, and topology stay dense: the profile names them and does not specify the algorithms |
| Sana-Video 5B | Not in this tree. The sol-engine profile wraps a private bundle |

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
The Tile-IR W4A4 GEMM stays off (`FASTVIDEO_NVFP4_OXIDE_GEMM`) until it beats
cuBLAS bf16 on the H3 FFN shape and PSNR ≥ 30 dB vs bf16. CUTLASS
SM100/SM120, Blackwell `to_blocked`, RHT, 2D block scales, and stochastic
rounding are not implemented. KWL operator fusion from the LTX-2.3 optimized
arm is not implemented. The upstream env turns several of those fusions off
because they change frames.

## Configurations

Weights: `--weights`, `FASTVIDEO_WEIGHTS`, or the Hugging Face snapshot.

| Variable | Effect |
|---|---|
| `FASTVIDEO_LTX2_WEIGHTS` | LTX-2.5 root for the Spark refiner. DiT defaults to `weights/transformer` |
| `FASTVIDEO_LTX2_DIT` | Override that DiT path |
| `FASTVIDEO_LTX2_LORA` | Distilled LoRA file. Otherwise a known filename beside the weights |
| `FASTVIDEO_LTX2_AUDIO_VAE` | Audio encoder file or directory. Otherwise `audio_vae/` or `ltx-2.5-audio-vae-bf16.safetensors` |
| `FASTVIDEO_LTX2_TEXT` | `resident`, `streamed`, or `auto` |
| `FASTVIDEO_LTX2_HQ` | 2.3-base 15+3 HQ contract |
| `FASTVIDEO_LTX2_FBCACHE` | LTX-2.5 stage-1 first-block cache |
| `FASTVIDEO_LTX2_STAGE1_CACHE` | LTX-2.3 stage-1 SCSP (res2s calls 16–28 of 29) |
| `FASTVIDEO_LTX2_MIDPOINT_PRUNE` | LTX-2.3 stage-2 feature-norm prune |
| `FASTVIDEO_LTX_VAE_FAST` | LTX conv VAE tiled decode on the channels-last bf16 decoder (default on); `0` restores the f32 streaming decoder |
| `FASTVIDEO_LTX_VAE_CHECK` | Decode the process's first LTX VAE tile both ways and log rel_l2 and both times |
| `FASTVIDEO_LTX_VAE_CONV_ALGO` | `tune`: time every cuDNN forward algorithm per conv shape (default: cuDNN's heuristic) |
| `FASTVIDEO_LTX_VAE_CHUNK_FRAMES` / `FASTVIDEO_LTX_VAE_CHUNK_MB` | Output frames (16) / MB (1024) per channels-last conv launch |
| `FASTVIDEO_LTX2_SAVE_LATENTS` | Save the decoded video latents to `<prefix>.f32` / `.shape` (for `ltx2 vae-bench --latents`) |
| `FASTVIDEO_GPU_TRACE_DECODE` | CUPTI trace of a whole video decode (kernel time by category, GPU idle) |
| `FASTVIDEO_H3_SOL_ATTN` | `spark` or `rtx` |
| `FASTVIDEO_H3_SOL_CACHE` | RTX TeaCache |
| `FASTVIDEO_H3_UPSCALER` / `FASTVIDEO_H3_LTX_ADAPTER` | Spark bridge checkpoints, if not beside the H3 weights |
| `FASTVIDEO_WAN_SOL_CACHE` | `taylorseer` |
| `FASTVIDEO_COSMOS3_OFFICIAL` | Cosmos3 canvas |
| `FASTVIDEO_COSMOS_SOL` | Cosmos TeaCache |
| `FASTVIDEO_HUNYUAN15_OFFICIAL` | 720p / 129f / 50-step canvas |
| `FASTVIDEO_HUNYUAN15_SOL` | Logs the 13B TeaCache gap and stays dense |
| `FASTVIDEO_LINGBOT_OFFICIAL` | 832×480 / 121f / 40-step canvas |
| `FASTVIDEO_LINGBOT_SOL` | Logs the unspecified cache/PISA/topology gap and stays dense |
| `FASTVIDEO_NVFP4` | W4A4 dequant (`1` → `static_6`; `mse` for FourOverSix) |
| `FASTVIDEO_BF16` | cuBLAS tf32/bf16 compute. On unless set to `0` |
| `FASTVIDEO_BF16_ACT` | DiT activations (and residual) as bf16, as the reference runs them. On by default for H3 and LTX-2 on a GPU; `0` restores f32, `1` forces bf16 for every model. CPU runs keep f32 |
| `FASTVIDEO_VSA` | Wan block-sparse video attention |
| `FASTVIDEO_SP_WORLD` | Sequence-parallel world size |
| `FASTVIDEO_SAVE_MP4` | Mux frames with ffmpeg |

Text-conditioning cache for LTX defaults under `~/.cache/fastvideo`
(`FASTVIDEO_CACHE` overrides the root).

## Not in this tree

NVlabs/Sana `main`: SANA image, SANA-1.5, SANA-Sprint, SANA-Video, SANA-Video
2.0, LongSANA, SANA-WM, SANA-Streaming, Sol-RL.

HunyuanVideo-13B as its own pipeline. The 1.5 family is what is registered.
