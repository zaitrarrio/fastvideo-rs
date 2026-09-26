# Upstream Python references on RTX PRO 6000

Every number below was measured on one **NVIDIA RTX PRO 6000 Blackwell Server Edition**
(97887 MiB, sm_120, driver 595.91.07, Runpod SECURE, EUR-IS-1).
The runs went through `scripts/gpu/runpod-http.sh upstream <runner sha>`, which calls `scripts/gpu/upstream/pod.sh`.
Weights were read from volume `jg48s6o1w0`, and everything written stayed on the container disk.
Raw cells are in `artifacts/runpod/upstream/<runner sha>-<MMDDHHMM>/<cell>/`
(`cell.json`, `result.json` / `benchmark.json`, logs). To re-tabulate them, run
`python3 scripts/gpu/upstream/summarize.py artifacts/runpod/upstream`.

**Prompt and seed** match our matrix (`scripts/gpu/runpod-matrix.sh` `PROMPT` / `SEED`).
The prompt is *"A man in his thirties talking to the camera in a bright living room, medium close-up, natural
expressions and hand gestures, soft window light. He says: <d>Hello, this was generated entirely in Rust.</d>"*
and the seed is 1024. Rows marked *demo* instead use sol-engine's own `models/minimax_h3/demo_prompt.json`
with seed 0 as a cross-check.

**Pinned upstream sources** (`scripts/gpu/upstream/setup.sh`):

| stack | source | runtime |
|---|---|---|
| FastVideo | hao-ai-lab/FastVideo `e90be598` | torch 2.12 cu130; Triton VSA (sm_100a kernels are datacenter-only); `--no-fa4`; `--num-gpus 1` |
| sol-engine (all rows) | NVlabs/Sana branch sol-engine `6c2f582b` | |
| sol-engine H3 RTX5090 | + SGLang `6fa3f9df` + Sol-Attn (`techniques/sparse_backends`) | torch 2.11 cu130 |
| sol-engine Sol-H3 4-step | `models/minimax_h3/Sol-H3/requirements.txt` | torch 2.10.0+cu130 (nvidia-cublas 13.1.0.3) |
| sol-engine LTX-2.5 RTX5090 | + Lightricks/LTX-2 `fd4ded7f` + nvidia-cutlass-dsl + cuda-python + apache-tvm-ffi + Sol-Attn | torch 2.13.0+cu132 |

**Images**: `ghcr.io/zaitrarrio/fastvideo-rs-upstream-<target>:sha-<7>` (`.github/workflows/upstream-images.yml`,
`docker/upstream.Dockerfile`, built with the same `setup.sh` installers). All of today's runs used
**`sha-78729d6`**. Run `297deb0-09252029` predates the baked images: it used `runpod/pytorch:1.3.3-cu1300-torch2130-ubuntu2404`
with the same installers run on the pod. For the other earlier runs, the artifacts record the venv stamp in `box.txt` but not the image tag.

**Units**: seconds per request. *Load* is model construction. *Warm* means a request after the load and at
least one warmup request, except for LTX-2.5 (see its methodology). *Peak torch* is the framework's own
allocator peak. *Peak smi* is the highest `nvidia-smi` reading over the whole process, including the CUDA context and caches.

## MiniMax-H3, 1344x768, 5 s (124 frames at 24 fps)

| implementation / route | steps (DiT forwards) | warm request | denoise | video decode | text enc | load | peak torch GiB | peak smi GiB | run |
|---|---|---|---|---|---|---|---|---|---|
| FastVideo FastH3-8-Step-V2, VSA 0.8 (Triton), inference torch.compile | 8 | **71.56** | 60.8 | 9.0 | 1.0 | 266.6 | 85.6 | 89.7 | 297deb0-09252029 |
| FastVideo FastH3 4-step LoRA (vsa-datafree), VSA 0.9 (Triton) | 4 | **43.66** | 26.3 | 9.0 | 1.0 | 301.6 | 77.8 | 82.9 | 297deb0-09252029 |
| FastVideo FastH3 4-step LoRA (dense-datafree), FLASH_ATTN | 4 | **51.65** | 39.0 | 9.0 | 1.0 | 272.1 | 74.2 | 78.3 | 297deb0-09252029 |
| FastVideo H3 base, strict profile, no compile | 50 | **541.58** | 521.2 | 10.9 | 9.2 | 236.6 | 73.0 | 77.6 | 0b015c4-09252140 |
| sol-engine Sol-H3 4-step, dense-datafree adapter, dense attention, TE offloaded | 4 | **60.91** | not split | not split | not split | 324.6 | 55.2 (62.6 reserved) | 74.4 | ec217cf-09261503 |
| sol-engine H3 RTX5090 `dense` | 50 | **802.05** | 719.7 | 74.3 | 7.6 | 198 | 16.4 | 31.1 | 7393e4a-09260036 |
| sol-engine H3 RTX5090 `sol` | 50 | **586.48** | 510.3 | 69.7 | 6.1 | 222 | 16.4 | 30.8 | 7393e4a-09260036 |
| sol-engine H3 RTX5090 `fullopt` | 50 steps, 14 computed | **168.27** | 149.7 | 11.4 | 6.8 | 249 | 25.1 | 36.0 | 7393e4a-09260036 |
| sol-engine H3 RTX5090 `dense`, demo prompt | 50 | 796.22 | 712.2 | 77.0 | 6.5 | 228 | 16.5 | 31.1 | 7393e4a-09260036 |
| sol-engine H3 RTX5090 `sol`, demo prompt | 50 | 610.95 | 524.1 | 77.7 | 8.8 | 196 | 16.5 | 30.9 | 7393e4a-09260036 |
| sol-engine H3 RTX5090 `fullopt`, demo prompt | 50 steps, 14 computed | 171.29 | 156.0 | 10.9 | 4.0 | 226 | 25.1 | 36.0 | 7393e4a-09260036 |

Methodology per stack:

- **FastVideo** (`scripts/gpu/upstream/bench_fastvideo.py`) uses FastVideo's own `examples/inference/basic/basic_fasth3.py` builders.
  It builds the generator, runs one excluded warmup at seed 999, then reports the median of 3 requests.
  Stage times come from `FASTVIDEO_STAGE_LOGGING`. The 4-step VSA runs were 44.0, 36.9 and 43.7 s; the median run
  includes a 6.9 s VideoSave. The 8-step 480p cell was also rerun in `0b015c4-09252140` (26.31 s), which agrees.
- **sol-engine H3 RTX5090** runs the reference's own `models/minimax_h3/RTX5090/run_minimax_h3_gpu.sh` with the env from
  `config/minimax_h3/rtx5090_<arm>.toml`. It uses the FL2VA partition, rebuilt byte-exactly from our weights plus the Hub
  (`reconstruct-h3-*.json`), and SGLang with layerwise DiT offload. That offload is why the process peaks near 31 to 36 GiB,
  and it makes the denoise PCIe-bound.
  The warm request is `benchmark.json` `measured.inference_time_s`, taken after one warmup request in the same process
  (5 steps for `dense`, 50 steps for `sol` / `fullopt`). Stage times come from SGLang's `[...Stage] finished in` lines.
  The peak torch column is `measured.peak_memory_mb`.
  - `sol`: Sol-Attn, tau 1.0, diag threshold, first 10 steps and 2 layers dense. The real-QKV correctness gate passed
    (max_abs 0.125, rel_l2 6.2e-4), and the route density was 0.179 over 590 blocks (37 756 live tokens).
    No TeaCache, so all 50 steps compute the DiT.
  - `dense`: no Sol, no TeaCache; all 50 steps compute the DiT.
  - `fullopt`: Sol-Attn as above, plus TeaCache (threshold 0.10, retain 5, cooldown 1, 49 decision calls,
    coefficients 1.0,0.0), regional torch.compile, and the full BF16 video VAE kept resident after denoise.
    TeaCache decisions in the measured request were **14 compute / 35 reuse** (reuse rate 0.714).
    The computed steps were 0, 1, 2, 3, 4, 13, 21, 28, 34, 39, 43, 45, 47 and 48, identical for our prompt and the demo prompt
    (`sol_events_rank0.jsonl`, `teacache_decision` events).
- **sol-engine Sol-H3 4-step** (`scripts/gpu/upstream/bench_sol_h3_4step.py`) follows the package's README methodology.
  It loads once (`MiniMaxH3Inference`, `attention_backend="dense"`, the only choice on 1 GPU) with
  `FastH3-4-step-Preview-v1-LoRA/dense-datafree` fused, runs one warmup (84.9 s), then reports the median of
  3 `engine.generate` calls. The runs were 60.91, 61.63 and 59.66 s. The timed span is text encoding, 4 DiT forwards,
  and video plus audio decode. The package does not split it into stages. There are two deviations, both recorded in `result.json`:
  (1) The Qwen3-VL text encoder stays on the host and is streamed per forward (`accelerate.cpu_offload`), because the
  released engine's full-GPU layout (TE 66 GB + DiT 66 GB) does not fit 96 GB. The 60.9 s therefore includes streaming the text encoder over PCIe once per request.
  (2) The process is re-executed with the wheel's `nvidia/cu13/lib` first on `LD_LIBRARY_PATH`. See the failures section below.

## MiniMax-H3, 832x480, 5 s (124 frames)

| implementation / route | steps | warm request | denoise | video decode | load | peak torch GiB | peak smi GiB | run |
|---|---|---|---|---|---|---|---|---|
| FastVideo FastH3-8-Step-V2, VSA 0.8 (Triton) | 8 | **26.21** | 20.7 | 4.0 | 266.6 | 75.9 | 77.6 | 297deb0-09252029 |
| FastVideo FastH3 4-step LoRA (vsa-datafree), VSA 0.9 (Triton) | 4 | **15.10** | 9.6 | 3.9 | 286.0 | 77.2 | 78.6 | 297deb0-09252029 |

sol-engine's H3 RTX5090 routes were **not run at 480p**. The reference driver (`models/minimax_h3/RTX5090/gpu_infer.py`)
hard-codes `short_edge: 768` / 1344x768 in its request and in `benchmark.json`, so a 480p cell would need the
reference itself modified. Sol-H3 4-step likewise hard-codes `WIDTH = 1344, HEIGHT = 768` (`h3_runtime/engine.py`).

## LTX-2.5 distilled two-stage (BF16), 3840x2176, 5 s (121 frames at 24 fps)

| arm | stage 1 | stage 2 | video_vae | e2e | peak alloc GiB | peak reserved GiB | peak smi GiB | stage-2 video attention calls | run |
|---|---|---|---|---|---|---|---|---|---|
| **Sol stage 2** | 55.88 | **79.08** | 27.91 | **196.99** | 30.36 | 31.40 | 32.3 | 144: 141 Sol kernel (47 each at tau 1.0 / 1.25 / 1.5) + 3 dense | ef75f72-09261430 |
| dense stage 2 | 62.83 | 155.55 | 18.43 | 257.03 | 30.36 | 31.40 | 32.3 | all dense | ef75f72-09261430 |
| dense stage 2 (earlier run) | 56.30 | 156.25 | 18.37 | 247.53 | 30.36 | 31.40 | 32.8 | all dense | e2ff318-09252354 |

## LTX-2.5 distilled two-stage (BF16), 1920x1088, 20 s (481 frames at 24 fps)

| arm | stage 1 | stage 2 | video_vae | e2e | peak alloc GiB | peak reserved GiB | peak smi GiB | stage-2 video attention calls | run |
|---|---|---|---|---|---|---|---|---|---|
| **Sol stage 2** | 54.85 | **73.66** | 20.44 | **163.46** | 29.07 | 30.15 | 31.0 | 144: 141 Sol kernel + 3 dense | ef75f72-09261430 |
| dense stage 2 | 53.69 | 146.07 | 20.31 | 234.35 | 29.07 | 30.15 | 31.6 | all dense | ef75f72-09261430 |
| dense stage 2 (earlier run) | 54.13 | 146.83 | 20.31 | 235.76 | 29.07 | 30.15 | 31.0 | all dense | e2ff318-09252354 |

LTX-2.5 methodology (`scripts/gpu/upstream/bench_ltx25.py`): the numbers are the reference driver's own
`benchmark.json` from `models/ltx25/RTX5090/gpu_infer.py --pipeline bf16 --offload cpu`, with the single-file packs rebuilt from
our Diffusers weights (`reconstruct-ltx-*.json`). The upsampler, video VAE and audio VAE rebuild byte-exactly; the text
encoder and the DiT embedding connectors carry the Diffusers numbers (`RECON_ACCEPT_MISMATCH=1`). The reference times
**one request end to end after the pipeline object is built** (build 0.05 s). Weights load lazily inside the stages
because of CPU offload, so stage 1 and e2e include loading from the container disk. There is no second warm request in
this methodology, so these rows are not "warm" in the FastVideo/H3 sense.
Stage 2 is the clean comparison between the arms. The dense arm keeps the same driver and instrumentation but routes every
stage-2 video self-attention to dense (the upstream README's "Dense Stage 2" column). The Sol arm is the unmodified driver.
Sol stage 2 is **1.97x** faster than dense at 4K 5 s (79.1 vs 155.6 s) and **1.98x** at 1080p 20 s (73.7 vs 146.1 s).
It saves 60 s of e2e at 4K and 71 s at 1080p 20 s. The two dense runs agree within 1 s on stage 2.

## Failures and fixes

| cell / step | run | cause | status |
|---|---|---|---|
| LTX Sol stage 2: `ModuleNotFoundError: tvm_ffi` | e2ff318-09252354 | older sol-ltx25 image without `apache-tvm-ffi` | fixed in the `sha-78729d6` image; measured above |
| LTX cells: `Path not found .../ltx-2.5-22b-distilled-transformer-bf16.safetensors` | 3979e17-09261332 | pod launched without `FV_EXTRA_ENV=RECON_ACCEPT_MISMATCH=1`, so the DiT pack (known connector mismatch) was not written | rerun in ef75f72-09261430 |
| Sol-H3 4-step: adapter fuse `CUBLAS_STATUS_INVALID_VALUE` (cublasGemmEx BF16) | 297deb0-09252029, 8774e51-09260024 | see the next row | fixed |
| Sol-H3 4-step: the FP32 fallback from 7393e4a also failed (`cublasSgemm` INVALID_VALUE) | 3979e17-09261332 | not a BF16 or shape problem: **every** cuBLAS GEMM failed in that venv, down to a 64x64 FP32 matmul, and cuBLASLt reported `NOT_INITIALIZED` (probe in ef75f72-09261430). torch 2.10.0+cu130 pins nvidia-cublas 13.1.0.3, but the base image's `LD_LIBRARY_PATH` (`/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64`, CUDA 13.0) shadows the wheel's libraries. With the wheel's `nvidia/cu13/lib` first, every probe passes and both `libcublas.so.13` and `libcublasLt.so.13` map from the wheel | commits 4b625ff and ec217cf: the bench probes a GEMM in a subprocess and re-execs under the first working library path (`result.json` `library_path_probe`). `addmm_fallbacks: 0`, so the adapter fused normally |
| Sol-H3 4-step: `No space left on device` | ef75f72-09261430 | Sol-H3's ModularPipeline downloads the full H3 from the Hub into `/opt/upstream/hf` (ignoring the local mirror), which did not fit beside the LTX packs on a 180 GB container disk | ran alone on a 240 GB disk in ec217cf-09261503 |
| sol-engine H3 RTX5090 at 480p | none | the reference hard-codes 1344x768 | not run |

## Pods (this session)

All RTX PRO 6000 at $2.09/hr on 2026-09-26, each deleted by the driver (a REST GET returns 404 for each):

| pod | image | runner | work | lifetime (UTC) |
|---|---|---|---|---|
| nr38ak52ssndd3 | upstream-sol-ltx25:sha-78729d6 | 3979e17 | LTX packs (DiT not written); Sol-H3 4-step (cuBLAS failure) | 13:32 to 14:28 (about 18 min image pull) |
| mmjelph8vuoetc | upstream-sol-ltx25:sha-78729d6 | ef75f72 | LTX Sol + dense, 4K 5 s + 1080p 20 s; Sol-H3 4-step (disk full) | 14:30 to 15:03 |
| 17cct81abi4jqx | upstream-sol-h3-4step:sha-78729d6 | ec217cf | Sol-H3 4-step | 15:03 to 15:33 |

The earlier runs (297deb0-09252029, 0b015c4-09252140, 0b015c4-09252233, e2ff318-09252354, 8774e51-09260024, 7393e4a-09260036) are from a previous session.
