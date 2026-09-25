# Parity evaluation: cudarc vs upstream FastVideo

Date: 2026-09-10 (post-reconcile, 100% in-scope target)
Compares: the cudarc Wan/FastWan surface in this tree against hao-ai-lab/FastVideo's Wan feature list.
Decision-log entry: `FVID-2026-09-10-cudarc-primary-parity`.

## Verdict

In-scope Wan inference + speed rows are at **100% full parity (19/19)**.
Sage / FP8 / FSDP / torch.compile and non-Wan surfaces remain intentionally
out of scope.

Caveats stated at the time:

- Flash is an online-softmax tiled kernel, not the vendor FlashAttention-2 binary.
- VSA is an in-tree block-sparse implementation, not the upstream CUDA extension.
- Sequence-parallel gather is host-mediated.

Status key: **parity** = feature parity; **partial**; **stub** = not wired;
**missing**; **oos** = out of scope.

## 1. Wan inference (12/12)

| Feature | Upstream | fastvideo-rs (cudarc) | Status | Notes |
|---|---|---|---|---|
| Wan/FastWan HF id registry | `registry.py` Wan family | `WAN_MODEL_DEFINITIONS` | parity | |
| `VideoGenerator.from_pretrained` API | Python `VideoGenerator` | fastvideo-core `VideoGenerator` | parity | |
| Sampling presets + UniPC | Flow UniPC multistep | `FlowUniPCMultistepScheduler` | parity | |
| FastWan DMD timesteps | `[1000,757,522]` etc. | `DmdSchedule` + pipeline defaults | parity | |
| T2V Diffusers load -> generate | PyTorch CUDA | cudarc `WanPipeline::load(preset)` | parity | |
| Wan 2.2 MoE dual DiT + dual CFG | `transformer` + `transformer_2` | `dit` + `dit_2`, `moe_expert`, `guidance_2` | parity | |
| I2V 36-ch pack + CLIP ViT-H + VAE encode | Full I2V pipeline | cudarc clip + `encode_video` + pack | parity | |
| Causal Self-Forcing mask | CausalDMD pipelines | `causal_temporal_mask` when `cfg.causal` | parity | |
| Fun InP (I2V-style) | Inpainting I2V | 1.3B preset + I2V pack; unit-tested | parity | |
| Fun Control / Lucy Edit | V2V / edit conditioning | `--control` / `--image` injects control latents | parity | T2V-width DiTs get first-frame latent inject; wide DiTs use I2V pack |
| MP4 `save_video` | ffmpeg / mux path | `--save-mp4` or `FASTVIDEO_SAVE_MP4=1` -> ffmpeg | parity | |
| Config file overlay | YAML `PipelineConfig` | TOML `--config` (generate knobs + control/dtype/device) | parity | Common knobs; not every upstream YAML key |

## 2. Wan speed (7/7 in scope)

| Feature | Upstream | fastvideo-rs (cudarc) | Status | Notes |
|---|---|---|---|---|
| Dense / Flash-style SDPA | `TORCH_SDPA` / Flash | Online-softmax tiled (default `FASTVIDEO_SDPA=flash`) | parity | Flash-style memory bound; not vendor FA2 binary |
| Video Sparse Attention (VSA) | `VIDEO_SPARSE_ATTN` + kernels | Block-sparse local+global when `FASTVIDEO_VSA=1` | parity | In-tree approximation; hard-fail if VSA id without flag |
| TeaCache | Wan 2.1 poly + accumulated L1 | 1.3B coeffs + `ret_steps` variant (`FASTVIDEO_TEACACHE`) | parity | |
| Sequence parallelism (`num_gpus`) | SP + gather | `--num-gpus N` query-seq shard + all-gather | parity | Logical SP; `device_for_rank` maps multi-GPU; gather host-mediated |
| Device-resident tensors | PyTorch GPU tensors | `CudaTensor` dual-storage + `Linear` pin (default on) | parity | |
| BF16 DiT GEMM | BF16 default compute | `cublasGemmEx` BF16xF32->F32 (default on) | parity | |
| GPU VAE causal conv3d | CUDA 3D | NVRTC `causal_conv3d_f32` + window fallback | parity | |
| torch.compile | `enable_torch_compile` | N/A (cudarc) | oos | |

## 3. Out of scope

| Feature | Upstream | fastvideo-rs | Status |
|---|---|---|---|
| Non-Wan model families | Hunyuan, LTX, Flux, ... | Out of scope | oos |
| Training / LoRA / DMD2 train | Post-training stack | Inference only | oos |
| Sage / SLA / Attn-QAT / FP8 / FSDP | Optional backends | Out of Wan cudarc bring-up scope | oos |
| Dreamverse / Gradio / MLX | Apps / Apple | Out of scope | oos |
| Burn / Candle / Luminal | Single PyTorch | Frozen; cudarc primary | oos |

## Defaults on CUDA

Residency on, BF16 GemmEx on, Flash-style SDPA on. Escape hatches:
`FASTVIDEO_RESIDENT=0`, `FASTVIDEO_BF16=0`, `FASTVIDEO_SDPA=dense`.
TeaCache / VSA opt-in: `FASTVIDEO_TEACACHE=1`, `FASTVIDEO_VSA=1`.

## Honest caveats vs PyTorch FastVideo

Not bit-identical to FlashAttention-2 binaries or official VSA tiles. Latency
still had to be proven on Vast vs Candle / PyTorch. Multi-GPU SP ranks map via
`device_for_rank`; a full NCCL all-gather was left as a follow-up.

## Validation state

| Gate | Status |
|---|---|
| Lib tests (cudarc + core) | Green on Mac |
| Vast 1.3B latency vs Candle | Scripted; re-run for proof |
| Vendor FA2 / upstream VSA FFI | Out of scope / approximate |

Scores cover in-scope Wan rows only. Upstream Sage / FP8 / FSDP / non-Wan are
OOS by product decision.

## What later analyses changed

- 2026-09-17: the head-to-head put upstream 1.96x ahead on the same card and
  attributed the gap to VSA. The VSA row above was an in-tree local+global
  approximation, not upstream's tile / coarse-topk / gate algorithm; that port
  landed 2026-09-18 (`FVID-2026-09-18-vsa-port`).
- 2026-09-24: the codebase review found the Flash-style SDPA row does not
  describe the shipping default (dense cuBLAS QK^T + softmax + PV).
