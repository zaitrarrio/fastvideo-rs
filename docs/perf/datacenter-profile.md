# Datacenter GPU profile: H100 SXM (sm_90) and B200 (sm_100)

Run date: 2026-09-29. Before anyone writes datacenter kernels: where does
the time go on an H100 and a B200, kernel by kernel, for every serving
family? Measured with Nsight Systems on one **H100 80GB HBM3 (SXM)** and one
**B200**, same image, same configs as production. It answers the questions
in `docs/perf/datacenter-kernel-plan.md` §6 (see "Answers to the kernel
plan" below).

## Findings first

**B200**

1. **Sparse attention is the B200 bottleneck.** The VSA fine kernel
   (`vsa_mma_attn_tma2`) and the Sol kernel (`sol_mma_fwd_x4f`) are
   `mma.sync` kernels. On the B200 they run only **1.14–1.19x** faster than
   on the H100. The dense tcgen05 kernel (`fa_dc100_fwd_d128`) runs **1.97x**
   faster than its H100 twin, and the GEMMs run **~2x** faster.
   - VSA is 28 % of B200 device time at 768p and 41 % at 1080p. At 10 %
     density, one VSA call costs **63 %** of a dense layer on the tcgen05
     kernel: about 0.2 PFLOPS effective, 9 % of the BF16 peak.
   - A Sol call costs **70–71 %** of a dense layer on the B200, against
     41–42 % on the H100. On sm_100, Sol's sparsity buys almost nothing.
2. **The GEMMs are healthy.** cuBLASLt picks
   `nvjet_sm100_qqtst_128x256_…_Avec32UE8M0_Bvec32UE8M0` for MXFP8, which
   reaches **3.08–3.23 PFLOPS** (68–72 % of the dense FP8 peak). The BF16
   nvjet kernels reach 1.39–1.74 PFLOPS (62–78 %). A custom tcgen05 GEMM has
   ≤ 3 % end to end to win.
3. **H3 and Wan keep the B200 busy.** The GPU is busy 93–98 % of the GPU
   window (H3: 94 % of the engine's run time). Syncs and launch latency
   together cost ≤ 0.2 s per job. What remains is kernel efficiency:
   - the memory-bound glue (qk-norm/RoPE, SwiGLU, residual/gate/modulate,
     head split/merge, casts) runs at 5–15 % of HBM bandwidth. It is only
     1.1–1.3x faster than on the H100, against a 2.4x bandwidth ratio;
   - the **Wan VAE decode spends 46 % of its time in two copy kernels**
     (`gather_nd`, `block_copy`) and only 33 % in convolution;
   - on **SF-Wan, the MXFP8 glue (activation quantize, dequant epilogue,
     block copies) takes 37 % of block time**, against 6 % for the GEMMs it
     feeds. The H100 runs the same model in BF16 without it.
4. **LTX idles because of host work, not kernels.** (Fixed 2026-09-29:
   see "WP-F results" — −17 % T2V / −15 % I2V at 1080p on the PRO 6000.) The GPU is busy 63 % of
   the run. Stage 1 and stage 2 are each 95–97 % busy; per-step host idle is
   < 5 %.
   - The loss is one **3.6 s host-CPU gap before stage 2**, when the
     full-resolution RoPE tables are built on the CPU (`Ropes::with_keyframes`
     → `fastvideo_models::ltx2::rope`, f64 cos/sin per token and slot), then
     uploaded.
   - Then ~0.8 s of per-tile host round trips in the VAE decode, and CPU x264
     after the last kernel.

**H100 SXM**

1. **Sparse attention is again the largest target.**
   - VSA is 22 % of device time at 768p and 34 % at 1080p, at about 16–17 %
     of the BF16 peak effective.
   - Sol is 20 % of H3 max.
   - The dense kernel (`fa_dc90_fwd_d128`, FA3-style wgmma) is already at
     **61–62 %** of the BF16 peak. An "FA3 for sm_90" project has little left
     to win on dense attention.
2. **The "W8A8 fallback" GEMM is fast.** On sm_90 the MXFP8 recipe falls back
   to tensorwise W8A8 on all 50 H3 blocks (`FASTVIDEO_H3_QUANT=mxfp8 needs
   sm_100+ … running W8A8`), and cuBLASLt picks
   `nvjet_sm90_qqtst_128x160_128x5…`.
   - It reaches **1.45–1.50 PFLOPS**, 74–76 % of the FP8 peak.
   - The cost of the fallback is the separate activation pass:
     `w8a8_quantize` and `amax_abs_mixed` run at ~0.33 TB/s and take 6–7 % of
     H3 device time.
   - A block-scaled FP8 GEMM on sm_90 would buy accuracy parity with MXFP8,
     not speed.
3. **Wan runs in BF16 on the H100.** It has no quant line and uses
   `nvjet_sm90_tst_…_bias` at 83 % of the BF16 peak, so the MXFP8 glue never
   runs there.
4. **H3 on an 80 GB card streams its text encoder per new prompt.** The log
   says `h3 i2v encoder: stream`, with a 64.6 GiB resident plan. A new prompt
   costs 5.4 s: 49 GB of H2D with the GPU 19 % busy, 19 % of a cold-prompt
   768p job. The LTX stage-2 host gap is longer on this host's Xeon: 4.6 s
   plus 0.5 s.

**Ranked opportunities** (details and estimates in the last section):

1. VSA fine attention on tcgen05 / wgmma: B200 24–32 %, H100 13–22 % of H3
   turbo.
2. LTX host RoPE tables, cached or built on the device: ~20 % of LTX on both
   GPUs. Trivial next to a kernel.
3. Wan VAE decode copy kernels: ~22–24 % of Wan 5B.
4. SF-Wan on B200 without the MXFP8 glue: ~25 % of block time.
5. Vectorising and fusing the memory-bound DiT glue: 6–11 %.
6. Sol on tcgen05 / wgmma: B200 11 %, H100 6 % of H3 max.
7. Overlapping the job tail: 3–9 %.

## Setup

| | |
|---|---|
| GPUs | **B200** (`tz3d74y6u0ct7s`, US-CA-2 secure, $6.79/hr): sm_100, 183 359 MiB, 1000 W, SM max 1965 MHz, driver 595.91.07; host AMD EPYC 9555 (224 vCPU). **H100 80GB HBM3 (SXM)** (`52rgddmv8h5dvo`, US-CA-2 secure, $3.49/hr): sm_90, 81 559 MiB, 700 W, SM max 1980 MHz, driver 580.126.09; host Xeon Platinum 8462Y+ (128 vCPU). No H100 NVL/PCIe or H200 was in stock next to either weight volume. |
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:01c9fcf5…` (`:sha-522af72`). It is the image `docs/serve/bench/b200.md` used, so the numbers compare directly. It carries fv-serve, fv-gpucheck and all configs, and it was cached on the hosts. Main's newer green images (`5be5aa5`, `bd58c71`) change the avatar and controller paths, not the kernels profiled here. |
| Weights | US volume `s2k01690bi`, read only. Nothing was written to either volume. |
| Workloads | Native `/fv/v1/jobs`, fox prompt, seed 7, configs baked into the image: `runpod.toml` (h3-turbo), `runpod-h3-max.toml`, `runpod-ltx.toml`, `runpod-wan5b.toml`. SF-Wan: `fv-gpucheck --mode fast wan stream --weights sfwan21-1.3b --run g20,seconds=20,rope=rebased,sink=3`, the CUDA-graph path; the last 10 s of the 20 s rollout are analysed. |
| Warm-up | H3: none needed; the boot warm-up makes the first job warm. LTX: one untraced 720p job first, which pays the lazy text-encoder load and caches the prompt. Wan: one untraced 480p job first. |
| VAE decode | Each serve job includes its family's full VAE decode: the H3 ViT decoder, the LTX conv VAE and the Wan 2.2 VAE. SF-Wan's TAE decode runs inside every block. |
| Profiler | Nsight Systems CLI 2026.5.1, installed on the pod. fv-serve runs under `nsys launch --trace=cuda,cublas,cudnn --cuda-graph-trace=node`, and each traced job is bracketed by `nsys start --sample=none --cpuctxsw=none` / `nsys stop` (`scripts/gpu/nsys-pod.sh`). |
| What is traced | CUDA runtime and driver API, every kernel, memcpy and memset, and CUDA-graph nodes. **No CPU sampling**: the pods have no perf access. `RmProfilingAdminOnly=1` also rules out Nsight Compute counters, so TFLOPS and bandwidth are analytic (below). |
| Reproducibility | Every cell was measured twice, on two pairs of pods. The first round's files were lost in a container restart; its numbers were within 1–4 % of the committed second round. |

**How the numbers are made** (`scripts/gpu/nsys_profile.py`):

- The *GPU window* runs from the first to the last device activity of the
  job. *Busy* is the union of kernel, memcpy and memset intervals.
- Each idle gap is attributed through the correlation ID of the launch that
  ended it:
  - **launch latency**: the work was already queued;
  - **host sync**: a blocking `cuStreamSynchronize`, `cuEventSynchronize` or
    sync memcpy returned inside the gap;
  - **other CUDA API**: the host was inside another CUDA call
    (`cuMemAllocAsync`, `cuMemHostAlloc`, graph instantiate);
  - **host CPU**: no CUDA call; the host was computing.
- *Run* is the engine's `run_s`. Work before the first kernel and after the
  last one counts in run, not in the window: request setup, CPU x264
  encode.
- Stages are cut in 0.25 s bins by marker kernels, so each boundary is good
  to ±0.25 s:
  - H3: `vsa_`/`sol_`/`h3_*` for denoise, `h3v_*` (the ViT VAE decoder) for
    decode;
  - LTX: DiT kernels, the longest near-idle run between the two denoise
    stages, then `ltxv_*` for decode;
  - Wan: `fa_dc`/`wan_*` for denoise, conv and norm-channels for decode.
- Categories are by kernel name. The H3 VAE decoder is a ViT, so its time
  appears as GEMM, attention and norm in the *decode* stage rows.

**Perturbation.** Tracing did not visibly slow the jobs:

- B200 H3 turbo 768p: traced denoise 11.88 s, against 11.84–11.92 s untraced
  (`b200.md`).
- B200 SF-Wan: block p50 0.405 s traced, against 0.414 s untraced.
- H100 SF-Wan: 0.513 s traced, against the 23.8 fps H100 reference.

## GPU busy time against wall time

| GPU | Workload | Run s | GPU window s | GPU busy s | busy / window | busy / run | idle: host CPU | host sync | launch latency | other CUDA API | kernels (< 10 µs) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B200 | H3 turbo 768p (VSA) | 15.89 | 15.33 | 14.99 | 97.8 % | 94 % | 0.25 | 0.05 | 0.01 | 0.01 | 27319 (5325) |
| B200 | H3 max 768p (Sol-H3) | 15.08 | 14.54 | 14.23 | 97.9 % | 94 % | 0.24 | 0.05 | 0.01 | 0.01 | 20241 (2995) |
| B200 | H3 turbo 1080p (VSA) | 41.59 | 40.74 | 39.23 | 96.3 % | 94 % | 1.38 | 0.08 | 0.02 | 0.02 | 52264 (6709) |
| B200 | LTX-2.5 1080p 6 s | 18.93 | 17.23 | 12.00 | 69.7 % | 63 % | 5.02 | 0.13 | 0.06 | 0.01 | 88943 (47889) |
| B200 | Wan 2.2 5B 704p | 10.01 | 9.66 | 8.99 | 93.0 % | 90 % | 0.04 | 0.00 | 0.01 | 0.63 | 12317 (778) |
| B200 | SF-Wan blocks (10 s window) | — | 10.00 | 9.57 | 95.7 % | — | 0.03 | 0.12 | 0.12 | 0.16 | 132723 (6250) |
| H100 | H3 turbo 768p (VSA) | 28.48 | 27.73 | 23.05 | 83.1 % | 81 % | 2.34 | 1.53 | 0.36 | 0.45 | 29822 (4719) |
| H100 | H3 max 768p (Sol-H3) | 23.65 | 22.88 | 22.45 | 98.1 % | 95 % | 0.30 | 0.06 | 0.07 | 0.01 | 23124 (2388) |
| H100 | H3 turbo 1080p (VSA) | 58.56 | 57.21 | 55.99 | 97.9 % | 96 % | 0.93 | 0.08 | 0.20 | 0.01 | 60567 (5604) |
| H100 | LTX-2.5 1080p 6 s | 28.16 | 25.52 | 18.58 | 72.8 % | 66 % | 6.48 | 0.19 | 0.23 | 0.02 | 87577 (46247) |
| H100 | Wan 2.2 5B 704p | 12.83 | 12.30 | 12.25 | 99.6 % | 96 % | 0.02 | 0.00 | 0.03 | 0.00 | 11845 (1209) |
| H100 | SF-Wan blocks (10 s window) | — | 10.00 | 9.67 | 96.7 % | — | 0.04 | 0.11 | 0.11 | 0.08 | 91390 (19282) |

What the table says:

- **H3, Wan and SF-Wan are device-bound on both cards.**
  - Over 97 % of all gaps are shorter than 5 µs; syncs and launch latency sum
    to ≤ 0.2 s per job.
  - H3's idle is a host hand-off between VAE tiles (`memcpy DtoH →
    block_copy`) of 60–760 ms. At 1080p on the B200 these add up to 1.2 s,
    so that decode is only 86 % busy.
  - SF-Wan's idle is `cuGraphInstantiateWithFlags` (0.08–0.14 s per 10 s)
    and one sync per block.
  - The B200 Wan 5B run lost 0.62 s to `cuMemAllocAsync` in the decode (pool
    growth); the first round's run did not (99.4 % busy).
  - Host launch overhead is not worth a project for these families.
- **Run minus window** (0.6–0.9 s for H3/Wan, 1.7–2.6 s for LTX) is host work
  at the edges of the job, mostly CPU libx264: neither card has usable NVENC.
- **H100 H3 turbo 768p** was that prompt's first job on the process. Its
  first 5 s are the streamed text encoder: 49 GB H2D at 53 GB/s, GPU 19 %
  busy, a 1.5 s `cuStreamSynchronize` gap in the loader and a 0.44 s
  `cuMemHostAlloc`. The 1080p job reused the prompt: text 0.28 s.
- **LTX** is the outlier on both cards; see "Why LTX is only partly busy".

## Where the time goes: GEMM / attention / elementwise / VAE / copies

Shares of device time for the whole job, then seconds per stage.

- **VAE conv** covers convolution work: cuDNN and CUTLASS implicit-GEMM
  fprop, their NCHW↔NHWC transposes, group norm, upsample and channel
  norms.
- **layout / copy kernels** covers gathers, block copies and head
  split/merge; **memcpy / memset** covers copy-engine work.

| GPU | Workload | Stage | Wall s | Busy | GEMM | attention | norm / elementwise | cast / quant | layout / copy kernels | VAE conv | memcpy / memset |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B200 | H3 turbo 768p (VSA) | **whole job** | 15.33 | 98 % | 23 % | 44 % | 18 % | 4 % | 9 % | 0 % | 1 % |
| | | denoise | 12.00 | 98 % | 2.47 s | 5.97 s | 2.08 s | 0.66 s | 0.62 s | 0.00 s | 0.01 s |
| | | decode | 3.50 | 93 % | 1.03 s | 0.70 s | 0.66 s | 0.01 s | 0.73 s | 0.03 s | 0.09 s |
| B200 | H3 max 768p (Sol-H3) | **whole job** | 14.54 | 98 % | 21 % | 47 % | 20 % | 0 % | 10 % | 0 % | 1 % |
| | | denoise | 11.25 | 98 % | 2.03 s | 6.07 s | 2.20 s | 0.01 s | 0.71 s | 0.00 s | 0.01 s |
| | | decode | 3.50 | 93 % | 1.04 s | 0.69 s | 0.66 s | 0.01 s | 0.74 s | 0.03 s | 0.09 s |
| B200 | H3 turbo 1080p (VSA) | **whole job** | 40.74 | 96 % | 19 % | 54 % | 14 % | 4 % | 9 % | 0 % | 1 % |
| | | denoise | 32.75 | 99 % | 5.05 s | 19.88 s | 4.17 s | 1.37 s | 1.98 s | 0.00 s | 0.02 s |
| | | decode | 8.00 | 86 % | 2.22 s | 1.43 s | 1.39 s | 0.03 s | 1.57 s | 0.03 s | 0.18 s |
| B200 | LTX-2.5 1080p 6 s | **whole job** | 17.23 | 70 % | 26 % | 26 % | 26 % | 1 % | 1 % | 12 % | 9 % |
| | | stage 1 | 4.00 | 95 % | 1.28 s | 0.72 s | 1.28 s | 0.04 s | 0.05 s | 0.00 s | 0.42 s |
| | | stage 1 -> 2 (upsample, host) | 4.50 | 4 % | 0.00 s | 0.00 s | 0.00 s | 0.01 s | 0.01 s | 0.02 s | 0.13 s |
| | | stage 2 | 6.50 | 97 % | 1.81 s | 2.39 s | 1.78 s | 0.01 s | 0.07 s | 0.07 s | 0.18 s |
| | | decode | 2.25 | 79 % | 0.00 s | 0.00 s | 0.02 s | 0.01 s | 0.04 s | 1.39 s | 0.31 s |
| B200 | Wan 2.2 5B 704p | **whole job** | 9.66 | 93 % | 3 % | 7 % | 11 % | 16 % | 37 % | 26 % | 0 % |
| | | denoise | 2.00 | 98 % | 0.21 s | 0.65 s | 0.30 s | 0.69 s | 0.11 s | 0.00 s | 0.01 s |
| | | decode | 7.75 | 91 % | 0.03 s | 0.00 s | 0.69 s | 0.73 s | 3.24 s | 2.33 s | 0.02 s |
| B200 | SF-Wan blocks (10 s window) | **whole job** | 10.00 | 96 % | 6 % | 38 % | 14 % | 28 % | 11 % | 3 % | 1 % |
| H100 | H3 turbo 768p (VSA) | **whole job** | 27.73 | 83 % | 30 % | 34 % | 13 % | 10 % | 8 % | 0 % | 5 % |
| | | pre (text / setup) | 5.00 | 19 % | 0.02 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.91 s |
| | | denoise | 17.75 | 98 % | 4.64 s | 7.16 s | 2.36 s | 2.32 s | 0.85 s | 0.00 s | 0.02 s |
| | | decode | 5.00 | 99 % | 2.19 s | 0.80 s | 0.75 s | 0.02 s | 0.91 s | 0.07 s | 0.20 s |
| H100 | H3 max 768p (Sol-H3) | **whole job** | 22.88 | 98 % | 27 % | 42 % | 14 % | 8 % | 8 % | 0 % | 1 % |
| | | denoise | 18.00 | 98 % | 3.90 s | 8.78 s | 2.40 s | 1.72 s | 0.88 s | 0.03 s | 0.02 s |
| | | decode | 5.00 | 98 % | 2.19 s | 0.80 s | 0.74 s | 0.01 s | 0.90 s | 0.04 s | 0.20 s |
| H100 | H3 turbo 1080p (VSA) | **whole job** | 57.21 | 98 % | 26 % | 45 % | 12 % | 8 % | 8 % | 0 % | 1 % |
| | | denoise | 46.25 | 99 % | 9.75 s | 23.79 s | 5.01 s | 4.69 s | 2.54 s | 0.02 s | 0.03 s |
| | | decode | 11.00 | 94 % | 4.64 s | 1.72 s | 1.58 s | 0.03 s | 1.93 s | 0.05 s | 0.40 s |
| H100 | LTX-2.5 1080p 6 s | **whole job** | 25.52 | 73 % | 34 % | 22 % | 18 % | 0 % | 1 % | 14 % | 11 % |
| | | stage 1 | 6.50 | 94 % | 2.69 s | 1.13 s | 1.38 s | 0.04 s | 0.05 s | 0.00 s | 0.85 s |
| | | stage 1 -> 2 (upsample, host) | 6.00 | 8 % | 0.01 s | 0.04 s | 0.01 s | 0.01 s | 0.01 s | 0.04 s | 0.37 s |
| | | stage 2 | 9.25 | 98 % | 3.69 s | 2.92 s | 1.98 s | 0.01 s | 0.08 s | 0.06 s | 0.30 s |
| | | decode | 3.75 | 79 % | 0.00 s | 0.00 s | 0.03 s | 0.01 s | 0.04 s | 2.42 s | 0.46 s |
| H100 | Wan 2.2 5B 704p | **whole job** | 12.30 | 100 % | 8 % | 10 % | 11 % | 7 % | 32 % | 32 % | 1 % |
| | | denoise | 2.75 | 99 % | 0.85 s | 1.23 s | 0.51 s | 0.00 s | 0.13 s | 0.00 s | 0.01 s |
| | | decode | 9.75 | 98 % | 0.09 s | 0.05 s | 0.82 s | 0.84 s | 3.78 s | 3.93 s | 0.05 s |
| H100 | SF-Wan blocks (10 s window) | **whole job** | 10.00 | 97 % | 16 % | 51 % | 19 % | 1 % | 9 % | 4 % | 1 % |

**B200 over H100, per category** (H100 time / B200 time for the same job).
For SF-Wan the row compares equal 10 s windows. The B200 fits 24.7 blocks
in that window and the H100 19.5, so the per-block speedup is
0.513 / 0.405 = **1.27x**.

| Workload | GEMM | attention | norm / elementwise | cast / quant | layout / copy kernels | VAE conv | device time |
|---|---:|---:|---:|---:|---:|---:|---:|
| H3 turbo 768p (VSA) | 1.96x | 1.19x | 1.14x | 3.49x | 1.30x | — | 1.54x |
| H3 max 768p (Sol-H3) | 1.98x | 1.42x | 1.10x | — | 1.22x | — | 1.58x |
| H3 turbo 1080p (VSA) | 1.98x | 1.20x | 1.19x | 3.38x | 1.26x | — | 1.43x |
| LTX-2.5 1080p 6 s | 2.07x | 1.31x | 1.10x | 1.10x | 1.13x | 1.69x | 1.55x |
| Wan 2.2 5B 704p | 3.88x | 1.97x | 1.35x | 0.59x | 1.17x | 1.68x | 1.37x |
| SF-Wan blocks (10 s window) | 2.76x | 1.38x | 1.35x | 0.02x | 0.85x | 1.50x | 1.01x |

The B200 has 2.3x the dense BF16/FP8 math and 2.4x the HBM bandwidth of an
H100 SXM:

- GEMMs and dense attention collect that (~2x).
- The sparse attention (1.2–1.4x) and the memory-bound glue (1.1–1.35x) do
  not. They are bound by latency and occupancy, not bandwidth.
- On SF-Wan and Wan denoise, the B200 spends more time in cast/quant than
  the H100 spends on the whole BF16 path.

## Kernel implementations picked

From the kernel names and the serve logs
(`artifacts/perf/datacenter/*/kernel-selection-log.txt`):

| Role | B200 (sm_100) | H100 (sm_90) |
|---|---|---|
| H3 DiT linears | cuBLASLt MXFP8 block-scaled `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_…_Avec32UE8M0_Bvec32UE8M0` on blocks 2..=46 (180 linears); BF16 `nvjet_sm100_tst_128x256_64x6_2x1_2cta` elsewhere and for the VSA `to_gate_compress` | tensorwise **W8A8** on all 50 blocks and the refiner (312 linears): `nvjet_sm90_qqtst_128x160_128x5_2x1_v_bz_coopA_algo2` (plus `144x128` at 1080p), with separate `w8a8_quantize` + `amax_abs_mixed`; BF16 `nvjet_sm90_tst_256x160 / 192x208` for the gate and the VAE |
| LTX DiT linears | BF16 `nvjet_sm100_tst_128x256_…_bias` / `256x256` | BF16 `nvjet_sm90_tst_256x160 / 256x144_…_bias / 192x192` |
| Wan 5B / SF-Wan linears | MXFP8 `nvjet_sm100_qqtst_128x256 / 256x128 / 128x128`, with `mxfp8_quantize` before and `quant_linear_epilogue` after each GEMM | BF16 `nvjet_sm90_tst_192x192 / 192x144 …_bias` (no quantization) |
| Dense attention d=128 | ours, `fa_dc100_fwd_d128` (tcgen05 + TMEM, FA4-style) | ours, `fa_dc90_fwd_d128` (wgmma + TMA, FA3-style) |
| H3 turbo sparse attention | ours, `vsa_mma_attn_tma2` ("Tma ring (3 slots) (sm10 / sm9, 128B swizzle)"), `mma.sync` MMA; prep `vsa_tile_qkv`, `vsa_tile_mean`, `vsa_combine` | same kernels |
| H3 max / LTX stage-2 sparse | ours, `sol_mma_fwd_x4f` (`mma.sync`), `sol_prep_kv`; dense layers and steps on `fa_dc100` | `sol_mma_fwd_x4f`; dense on `fa_dc90` |
| H3 VAE decoder (ViT, d=64) | `flash_mma_fwd2_d64` (`mma.sync`), BF16 nvjet, `h3v_*` glue | same |
| Text-refiner / Qwen-VL vision | `sdpa: fused mma` (d=128), `sdpa: device dense` (d=72) | same |
| LTX / Wan VAE conv | cuDNN → CUTLASS 3 `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16` (plus TF32 `f32` variants and weight-stationary conv3d), `nchwToNhwc` / `nhwcToNchw` transposes | cuDNN → `sm90_xmma_fprop_implicit_gemm_bf16…256x128x64_warpgroupsize2x1x1` (plus a 12 ms TF32 `f32` variant), `sm80_xmma_fprop…indexed_wo_smem` |
| SF-Wan TAE decode | CUTLASS `conv3d_fprop_weight_stationary` (F32), `upsample_nearest` | `sm80_xmma_fprop_implicit_gemm_tf32` |

NVFP4 is off everywhere. The embedded oxide BF16 tiles are loaded on the B200
but no oxide kernel shows in any trace: cuBLASLt serves every GEMM.

## Achieved throughput against peak

The peaks are dense, with no sparsity:

| | BF16 | FP8 | HBM |
|---|---|---|---|
| B200 | 2250 TFLOPS | 4500 TFLOPS | 8 TB/s |
| H100 SXM | 989 TFLOPS | 1979 TFLOPS | 3.35 TB/s |

FLOPs are analytic, from the model configs in `crates/fastvideo-models/src/*/config.rs`:

- **H3**: 50 layers. Per layer, q/k/v 5376→7168, out 7168→5376, SwiGLU
  5376→28672→5376, plus `to_gate_compress` 5376→7168 on turbo. There are
  37 736 (768p) or 75 920 (1080p) joint rows, from the `vsa-h3 … rows` log
  line, over 4 steps.
- **LTX video stream**: 48 × (self 4·4096², text-cross q/o, a2v q/o, FF
  4096→16384→4096). Stage 1 has 9 690 tokens × 8 steps and stage 2 has
  38 760 × 3.
- **Wan 5B**: 30 × (6·3072² + 2·3072·14336), 26 598 tokens × 3 steps.
- **Attention**: 4·Sq·Sk·d·heads.

Times are the kernel sums in the denoise stage. Nsight Compute counters were
not available.

| Kernel group | Workload | B200 | % of B200 peak | H100 | % of H100 peak |
|---|---|---:|---:|---:|---:|
| FP8 DiT GEMMs (MXFP8 on B200, W8A8 on H100) | H3 turbo 768p | 3.08 PF | 69 % | 1.50 PF | 76 % |
| | H3 turbo 1080p | 3.13 PF | 70 % | 1.45 PF | 74 % |
| | Wan 5B 704p (B200 only) | 3.23 PF | 72 % | — (BF16) | — |
| BF16 DiT GEMMs | H3 (non-MX blocks + gate) | 1.39–1.74 PF | 62–78 % | 0.69–0.77 PF | 69–78 % |
| | LTX 1080p (all BF16) | 1.52 PF | 67 % | 0.73 PF | 74 % |
| | Wan 5B (H100 BF16) | — | — | 0.82 PF | 83 % |
| Dense attention `fa_dc*` d=128 | Wan 5B self-attention (90 × Sq=Sk=26 598, 24 heads) | 1.21 PF | 54 % | 0.61 PF | 62 % |
| | H3 max dense (56 dense layer-forwards, 39 748 rows) | 1.19 PF | 53 % | 0.60 PF | 61 % |
| VSA fine `vsa_mma_attn_tma2` (≈10 % density) | H3 turbo 768p: 21.3 / 25.4 ms per call | ≈ 0.19 PF effective, 1.9 PF dense-equivalent | ≈ 9 % | ≈ 0.16 PF effective, 1.6 PF dense-equivalent | ≈ 16 % |
| | H3 turbo 1080p: 80.9 / 96.2 ms per call | ≈ 0.20 PF effective | ≈ 9 % | ≈ 0.17 PF effective | ≈ 17 % |
| Sol `sol_mma_fwd_x4f` | H3 max: 27.0 / 30.8 ms per call | **71 %** of a dense `fa_dc100` layer (38 ms) | — | **41 %** of a dense `fa_dc90` layer (76 ms) | — |
| | LTX stage 2: 14.5 / 17.3 ms per call | **70 %** of a dense layer (20.7 ms) | — | **42 %** of a dense layer (41 ms) | — |

Memory-bound kernels, as bytes per call over time. The byte counts assume
bf16 activations (f32 where the name says `f32`), so they are estimates:

| Kernel (workload) | Bytes / call | B200 | % of 8 TB/s | H100 | % of 3.35 TB/s |
|---|---:|---:|---:|---:|---:|
| `h3_qk_norm_rope` (H3 768p; 270 M elements, read + write) | 1.08 GB | 2.52 ms → 0.43 TB/s | 5 % | 2.98 ms → 0.36 TB/s | 11 % |
| `h3_swiglu_mx` / `mx_swiglu` (H3 768p) | 2.7 / 3.2 GB | 2.61 ms → 1.0 TB/s | 13 % | 3.18 ms → 1.0 TB/s | 30 % |
| `w8a8_quantize` (H100 H3 768p; 203 M elements) | 0.61 GB | — | — | 1.83 ms → 0.33 TB/s | 10 % |
| `block_copy` (Wan VAE; 56 M elements, f32) | 0.45 GB | 0.64 ms → 0.70 TB/s | 9 % | 0.73 ms → 0.62 TB/s | 18 % |
| `quant_linear_epilogue` (SF-Wan; 7.2 M elements, f32 in, bf16 out) | 43 MB | 75 µs → 0.57 TB/s | 7 % | — | — |
| `mxfp8_quantize` (SF-Wan; 4680×1536) | 22 MB | 51 µs → 0.42 TB/s | 5 % | — | — |

`gather_nd` in the Wan VAE runs 5.6 ms (B200) / 6.6 ms (H100) per call over
a 7 M-thread grid, 315 calls. It is the single most expensive kernel of Wan
5B on both GPUs.

## Why LTX is only partly busy (B200 30–40 % util in `b200.md`)

The B200 1080p timeline, as 0.25 s bins
(`artifacts/perf/datacenter/b200/ltx-1080/gaps.csv`, `timeline.csv`):

| Interval (s from first kernel) | What runs | GPU busy |
|---|---|---:|
| 0.0 – 4.0 | stage 1: 8 ancestral steps at 9 690 tokens, 0.47 s each | 95 % |
| 4.0 – 4.4 | latent upsampler (cuDNN conv3d, 0.37 s), noise and renoise | ~40 % |
| **4.4 – 8.05** | **one 3.64 s host-CPU gap** (ends in `memcpy HtoD`) | **0 %** |
| 8.1 – 8.5 | one more 0.41 s host gap between H2D copies, then 4.9 GB H2D for stage-2 step 1 | ~10 % |
| 8.5 – 15.0 | stage 2: 3 Sol steps at 38 760 tokens, 1.95–2.23 s each | 97 % |
| 15.0 – 17.2 | VAE decode; ~0.4 s of host gaps (`memcpy DtoH → block_copy`: per-tile read-back) | 79 % |
| after the window | CPU x264 encode (0.81 s) | — |

The code shows what the gap is. In `ltx2/pipeline.rs` the stage-2 timer
starts, and `Ropes::with_keyframes(…, grid_full, …)` then builds the
video, audio and cross RoPE tables on the host. The builder is
`fastvideo_models::ltx2::rope`: one f64 `cos`/`sin` per token and slot, on
one thread, at 38 760 video tokens. `DeviceRope::upload` then produces the
H2D burst that ends the gap. On the H100 host, a Xeon at lower
single-thread speed, the same gap is **4.59 s + 0.50 s**.

So the answers are:

- LTX is **not** launch-bound or sync-bound: syncs sum to 0.13–0.19 s and
  launch latency to 0.06–0.23 s.
- Stage transitions and the upsampler cost ~0.5 s of device time.
- The text encoder is off the path while the prompt is cached (0.04–0.06
  s).
- CPU encode is 0.8–1.4 s after the window.
- The dominant idle is **host-side RoPE construction at the stage-2
  transition**, ~21 % of the B200 run and ~18 % of the H100 run.

That host gap is single-threaded CPU work, so it grows on slower hosts. That
fits `b200.md`'s LTX run on another B200 host (stage 2 14.8 s against 10.4 s
here) and its 30–40 % util. Stage 1 there was also slower (8.3 s against
4.9 s), which a single gap does not explain. Per-step host work, such as the
1.15 GB of H2D in 98 copies each step, is the likely suspect, but this
profile's host did not reproduce it.

## WP-F results: LTX host gaps removed (2026-09-29)

What changed (commits on `wip/ltx-rope`, merged to main):

- **RoPE tables built on the device.** A slot's angle depends only on a
  token's fraction on one axis and the slot's frequency, and an LTX grid
  has a few hundred distinct fractions (one per latent frame, row and
  column, plus keyframe / reference blocks). The host now evaluates cos/sin
  once per distinct (fraction, frequency) with the same f32-angle / f64
  cos-sin arithmetic (`SplitRopeLut`, `fastvideo_models::ltx2::rope`), and
  the kernel `ltx_split_rope` gathers the `[H·S, D]` tables on the device.
  Pure copies, so the tables are **bit-identical** to the host build: host
  test `factored_tables_expand_to_the_direct_ones_exactly` (T2V, I2V,
  keyframes, IC-LoRA reference, both divisions), GPU check `fv-gpucheck
  kernels --groups ltx_rope` (device vs host at the 1080p stage-2 geometry).
  Every `Ropes` constructor goes through it, so T2V, I2V/keyframes, ref2v,
  A2V, retake/extend and the refiner all do. `FASTVIDEO_LTX2_HOST_ROPE=1`
  restores the host build. This also removes the ~1.3 GB table upload.
- **CPU encode overlapped with the decode.** The "per-tile host round
  trips" in the decode were not the VAE: the served path wrote each chunk
  into ffmpeg's stdin *on the decode thread* (`Delivery::frames` via the
  sink pump), so the tiles waited for x264 / NVENC, and frames reached the
  encoder only at the decode's next report. `FrameSink::detach` now hands
  file jobs' ffmpeg feed to a relay thread fed straight from the writer's
  tap; the decode never waits on the encoder and the encoder works while
  later tiles decode. The VAE tiles already stayed on the device (blend,
  RGB8 pack on the GPU, one async copy per chunk on its own stream).

**Before / after, RTX PRO 6000 (EUR-IS-1, EU volume), same pod**, native
`/fv/v1/jobs`, `ltx-turbo`, 1920x1080 (1088 generated), 6 s, 24 fps, warm
(after one 720p job). Baseline image `sha-1dac7d9` (the merge base), new
`sha-a9fec55`; the control arm is the new image with
`FASTVIDEO_LTX2_HOST_ROPE=1`.

| Job | run_s | stage 1 | upsample | stage 2 | video decode | encode tail |
|---|---:|---:|---:|---:|---:|---:|
| T2V, baseline (2 runs) | **32.86 / 32.54** | 10.14 / 10.16 | 0.57 / 0.54 | 16.42 / 16.10 | 4.40 / 4.41 | 0.90 / 0.83 |
| T2V, new | **27.28 / 26.90** | 9.06 / 8.99 | 0.58 / 0.55 | 12.19 / 11.87 | 4.08 / 4.05 | 0.13 / 0.19 |
| T2V, new with host RoPE (control) | 33.36 | 10.15 | 0.65 | 16.93 | 4.12 | 0.14 |
| I2V, baseline (warm) | **38.00** | 12.15 | 0.54 | 19.09 | 4.40 | 0.91 |
| I2V, new (warm) | **32.34** | 11.08 | 0.62 | 14.65 | 4.03 | 0.12 |

- T2V **−17 %** (32.7 → 27.1 s), I2V **−15 %** (38.0 → 32.3 s) end to end.
- RoPE: stage 2 −4.3 s (the stage-2 table build and upload), stage 1
  −1.1 s (its smaller table). The control arm puts both back.
- Encode: the post-decode tail drops from 0.83–0.91 s to 0.12–0.19 s and
  the decode itself by ~0.35 s (no more stalls behind the encoder).
- First-job I2V (prompt load included): 84.1 → 76.4 s.
- Not run on the H100/B200 (budget); the profile's gap sizes there
  (3.6–4.6 s + 0.4–0.5 s before stage 2, 0.8–1.4 s encode tail) are the
  same work, so the profile's −18…−21 % estimate stands.

**Output identity.** The 720p warm-up job decodes to the same frames in all
three boots (baseline, new, control): frame MD5 `a5413900…`. At 1080p the
output is deterministic within a boot (both T2V runs identical in each arm)
but differs between boots *whatever the RoPE path*: baseline vs the
host-RoPE control 33.7 dB PSNR, baseline vs new 33.6 dB, new vs control
34.3 dB. That run-to-run spread predates this change (I2V differs even
between two runs of one boot: 35.6 dB) and is left for a separate look
(Sol stage-2 selection or a per-boot kernel choice are the suspects).
Raw records: `artifacts/perf/wp-f/rtxpro6000-ab.jsonl`.

**Oracle parity: not run to completion.** `scripts/gpu/oracle.sh 37faef4`
(targets `ltx25-512p ltx25-i2v`, `FV_ORACLE_RUN_MODE=all` so the runtime pod
would also run `fv-gpucheck kernels`, including `ltx_rope`) brought up the
upstream pod and produced the `ltx25-512p` reference dump, but no RTX PRO 6000
was free in EUR-IS-1 for the runtime pod, and the driver was lost in a
container restart; the upstream pod was deleted. So the GPU-side
`ltx_rope` bit-exact check and the oracle diff are **still to run**
(`ORACLE_TARGETS="ltx25-512p ltx25-i2v" FV_ORACLE_RUN_MODE=all
ORACLE_WAIT_FIRST=1 UP_IMAGE_TAG=latest scripts/gpu/oracle.sh <sha>`). What
covers the change meanwhile: the host test that the factored tables expand to
the direct ones bit for bit, the kernel being a pure gather of those values,
and the 720p output being bit-identical to the baseline.

Spend: A/B pod 18 min ($0.64); oracle upstream pod 100 min, mostly idle
while waiting for runtime stock ($3.48).

Still open from the profile's LTX list: the spatial upsampler is loaded
from disk every job (0.55 s warm, part of it host), and the stage-1 /
stage-2 per-step host work (< 5 %).

## Answers to the kernel plan (`datacenter-kernel-plan.md` §6)

| Plan question | Measured | Consequence |
|---|---|---|
| LTX host idle per step (B200) | Per step **< 5 %**: stage 1 is 95 % busy, stage 2 97 %. The idle is one 3.6 s + 0.4 s host gap at the stage-2 transition, plus decode round trips. | WP-F should target the stage-2 RoPE build and the decode tile round trips. Per-step sync removal and graph capture are worth < 3 %. |
| VSA share of H3 turbo denoise | B200 **51 %** (768p) and **61 %** (1080p); H100 **41 %** and **52 %** | ≥ 35 %: WP-D (VSA) rises to **#1** |
| VAE decode breakdown | Wan 5B (B200): layout copies 46 %, conv 33 %, cast 10 %, norm 10 %. LTX: conv 78 %, copies 17 %. H3 is a ViT: GEMM 32 %, copies 22 %, attention 21 %, norm 20 %. | For Wan, WP-G must cover the `gather_nd` / `block_copy` path, not only conv3d |
| Dense attention share | H3 max 15 % (B200) / 19 % (H100); Wan 5B 7 % / 10 %; kernels at 53–62 % of peak | The dense upgrades in WP-C/WP-B are worth ≤ 3 % end to end |
| GEMM share of H3 turbo on H100 | **28 %** of device time, already at 76 % of FP8 peak; plus 10 % cast/quant | < 35 %: WP-A should become "fuse the W8A8 activation quantize" (~5–6 %), not a new GEMM |
| Sol share (B200) | LTX stage-2 attention **38 %** (Sol 32 %); H3 max denoise attention **55 %** (Sol 35 %) | WP-C (Sol on tcgen05) comes after WP-F for LTX; for H3 max it is worth ~11 % |
| SF-Wan attention tail / re-rope share | attention 38 % (B200) / 51 % (H100); `wan_qk_norm_rope16` 5 %. On B200, **MXFP8 glue is 37 %**. | WP-E: turn off or fuse the MXFP8 glue on B200 first (~25 %), then tune causal attention |
| Per-step cuBLASLt host time | cuBLAS/cuDNN range tracing produced no events. Host launch overhead is invisible: no cuBLAS-related gaps; launch latency ≤ 0.2 % of the window in H3/Wan. | WP-0 (c) is a graph-capture prerequisite only, with no standalone gain |

## Ranked optimization opportunities

Estimated end-to-end gain is the saving over the measured run (`run_s`;
SF-Wan: block time). The assumed speedups are stated per item. The numbers
are estimates, not measurements.

| # | Opportunity | Where it acts | Est. gain, B200 | Est. gain, H100 | Type / effort |
|---:|---|---|---|---|---|
| 1 | **VSA fine attention on tcgen05/TMEM (sm_100) and wgmma (sm_90)**, with tile prep and combine fused in (`vsa_tile_qkv`, `vsa_tile_mean`, `vsa_combine`: another 1.5 s at 768p). Assumes 3.5x (B200) / 2.5x (H100) on `vsa_mma_attn_tma2`, i.e. ~40 % of peak at the executed density. | H3 turbo (the bulk of traffic) | **768p −24 %** (−3.8 s); **1080p −32 %** (−13 s) | **768p −13 %** (−15 % with a cached prompt); **1080p −22 %** | new kernel (WP-D) |
| 2 | **LTX: cache the RoPE tables per (grid, fps, audio tokens) or build them on the device** (the 3.6 s + 0.4 s host gap before stage 2, and the smaller one before stage 1) | LTX, every GPU | **−21 %** (−4.0 s) | **−18 %** (−5.1 s) | host code, small (WP-F) |
| 3 | **Wan VAE decode: fold causal-conv padding / cache into the conv** (implicit padding or a preallocated ring) instead of `gather_nd` + `block_copy`, and drop the f32↔bf16 casts. Assumes 75 % of 3.2 s (B200) / 3.8 s (H100) removed. | Wan 5B (and Wan 2.1 VAE users) | **−24 %** | **−22 %** | VAE rewrite (WP-G) |
| 4 | **SF-Wan on B200: BF16 linears (as the H100 already runs) or MXFP8 with the quantize fused into the producer and the dequant/bias in the GEMM epilogue.** Glue: 3.6 s of 9.6 s per 10 s window. | SF-Wan live | **−25 %** block time (0.405 → ~0.30 s, ≈ 38 fps) | 0 | config first, then fusion (WP-E) |
| 5 | **Vectorise and fuse the memory-bound DiT glue** (5–15 % of HBM BW): qk-norm + RoPE into the QKV epilogue; gate/residual/modulate into the out-proj and FF-out epilogues; head split/merge into the GEMM layouts; `cast_f32_bf16`. Assumes these halve. | H3, LTX, Wan | H3 −11 %, LTX −8 % | H3 −10 %, LTX −6 % | kernel work, incremental |
| 6 | **Sol attention on tcgen05 (sm_100)**, cutting a call from 71 % to ~41 % of a dense layer (the H100 ratio); **wgmma Sol on sm_90** at 1.5x | H3 max, LTX stage 2 | H3 max −11 %, LTX −4 % | H3 max −6 %, LTX −3 % | new kernel (WP-C / WP-B) |
| 7 | **Overlap the job tail**: stream frames into the (CPU, no NVENC) encoder during decode; keep LTX/H3 decode tiles on the device (no per-tile DtoH round trip) | all | LTX −8 %, Wan −4 %, H3 −3…6 % | LTX −9 %, Wan −6 %, H3 −3 % | serving code |
| 8 | **H3 VAE decoder (ViT)**: a d=64 `fa_dc` variant instead of `flash_mma_fwd2_d64` (`mma.sync`); fuse `h3v_qkv_heads` / `h3v_merge_heads` / `h3v_swiglu` into GEMM epilogues | H3 all | −6 % | −5 % | kernel work |
| 9 | **H100: fuse the W8A8 activation quantize** (`w8a8_quantize` + `amax_abs_mixed`, 1.5–3.0 s per job) into the producer kernels, as the sm_100 MXFP8 path already does in `h3_swiglu_mx` / `fvf_merge_heads_mx` | H3 on sm_90 | 0 | **−5…6 %** | fusion (WP-A, revised) |
| 10 | **Causal attention for SF-Wan** (Sq = 4 680 against a ≤ 32 760-token KV window): split-KV / 2-CTA tuning of `fa_dc100` / `fa_dc90` for short-query, long-KV shapes; assumes +30 % (B200), +15 % (H100) | SF-Wan | −9 % block time | −7 % | kernel tuning (WP-E) |
| 11 | **H100 text encoder residency** (not a kernel): a new prompt streams 49 GB of Qwen weights (5.4 s) | H3 on 80 GB cards | 0 | −19 % of a cold-prompt 768p job | memory planning |
| 12 | **Wan 5B on B200: MXFP8 glue in denoise** (cast/quant 0.69 s > GEMM 0.21 s); BF16 or a fused quant | Wan 5B | −5 % | 0 | config / fusion |
| 13 | **Pre-reserve the decode memory pool** (0.62 s in `cuMemAllocAsync` in one of the two B200 Wan runs) | Wan 5B | 0–6 % | ~0 | host code |
| 14 | Dense attention tuning (FA3/FA4-level): `fa_dc90` at 61–62 % and `fa_dc100` at 53–54 % of peak; assumes +25 % on B200 | H3 max dense steps, Wan | ≤ 3 % | ≤ 2 % | kernel tuning |
| 15 | Custom tcgen05 / wgmma GEMMs, or CUTLASS block-scaled FP8 on sm_90: cuBLASLt already at 62–78 % (BF16) / 69–76 % (FP8) | all | ≤ 3 % | ~0 (block-scaled FP8 on sm_90 is an accuracy item, not a speed one) | not worth it now |
| 16 | Host syncs / launch overhead / graph capture of LTX stages; caching SF-Wan graph execs (`cuGraphInstantiate` 1.4 %) | LTX, SF-Wan | LTX ≤ 3 %, SF-Wan ~1 % | same | host code |

Where the list differs from the task's hypotheses:

- **"FP8 GEMM on sm_90 via wgmma/TMA instead of the W8A8 fallback"**: the
  fallback's GEMM is already a wgmma/TMA cuBLASLt kernel at 74–76 % of FP8
  peak. The win on sm_90 is fusing its activation quantize (#9), not the
  GEMM.
- **"FA3-style attention for sm_90"**: dense `fa_dc90` is already FA3-style
  at 61–62 %. The sm_90 attention gains are in the sparse kernels (#1,
  #6).
- **"tcgen05/TMEM attention and GEMM for sm_100"**: yes for sparse
  attention (#1, #6). No for GEMMs (#15). Dense `fa_dc100` has modest
  headroom (#14).
- **"Removing host syncs and graph-capturing LTX stages"**: < 3 %. LTX
  loses its time to one host computation (#2) and the decode/encode tail
  (#7), not to syncs.
- **"Overlapping CPU encode"**: 3–9 % (#7), largest for LTX.

## WP-D results: VSA on the datacenter kernels (2026-09-29)

Opportunity #1 above, implemented as `fa_dc100_vsa` (tcgen05) /
`fa_dc90_vsa` (wgmma) with a KV-tile-list TMA producer, a fused prep
(tiles + means in one launch) and the H3 combine in the fine epilogue
(`docs/perf/datacenter-kernel-plan.md` WP-D). Same binary for both arms; the
baseline arm is `FASTVIDEO_VSA_KERNEL=tma2` (the incumbent path, unchanged).

**Kernel tier** (`fv-gpucheck kernels --groups vsa_dc,attn_dc`: 99/99 pass
on both GPUs; artifacts `artifacts/perf/wp-d/{b200,h100}/`).

| Check | B200 | H100 SXM |
|---|---|---|
| fine vs `vsa_mma_attn_tma2`, same selection (H3 480p / 768p / 1080p grids, sparsity 0.5-0.9, full and odd-base partial query ranges, smooth and white-noise q/k): rel-L2 (limit 1e-3) | 1.5e-5 - 5.7e-4 | 1.6e-5 - 4.7e-4 |
| fine vs an f64 block-masked dense reference on sampled query tiles (limit 3.5e-3) | 7.6e-5 - 1.58e-3; at most 1.011x tma2's own error | 7.6e-5 - 1.58e-3; at most 1.010x tma2's |
| fused prep vs `vsa_tile_qkv` + `vsa_tile_mean` (f32 round16, f32, bf16) | bit-identical | bit-identical |
| fused H3 combine vs `vsa_combine(round16)` / `_g16` (f32, bf16, no gate) | bit-identical | bit-identical |

Timings, 56 heads, synthetic smooth q/k (ms; the fine stage alone and
prep + fine + combine):

| Grid | Sparsity | B200 fine tma2 → dc | B200 stage | H100 fine tma2 → dc | H100 stage |
|---|---|---|---|---|---|
| 480p (280 tiles) | 0.8 | 7.13 → 5.15 (1.38x) | 1.55x | 7.77 → 4.23 (1.84x) | 1.70x |
| | 0.9 | 3.87 → 3.28 (1.18x) | 1.57x | 3.96 → 2.37 (1.67x) | 1.55x |
| 768p (660 tiles) | 0.8 | 36.3 → 24.7 (1.47x) | 1.58x | 41.6 → 22.0 (1.89x) | 1.92x |
| | 0.9 | 18.5 → 14.0 (1.33x) | 1.55x | 21.2 → 11.6 (1.83x) | 1.86x |
| 1080p (1350 tiles) | 0.8 | 149.2 → 97.9 (1.52x) | 1.59x | 171.7 → 105.7 (1.63x) | 1.82x |
| | 0.9 | 75.3 → 54.1 (1.39x) | 1.52x | 90.2 → 50.9 (1.77x) | 1.85x |

Prep (q/k/v tiling + means) alone: B200 5.87 → 1.49 ms at 768p, 11.40 →
2.49 ms at 1080p; H100 5.87 → 1.89 and 12.74 → 3.84 ms. Effective fine
rate: B200 366-437 TFLOPS, H100 405-464 (tma2: 276-287 / 237-250).

The sm_100 kernel runs two query tiles as one M = 128 tcgen05 tile over the
union of their selections. On these synthetic fields the union is 1.55-1.70x
one selection. **On real H3 turbo selections it is 1.25-1.42x (median 1.33,
200 calls, `FASTVIDEO_VSA_UNION_LOG=1`)**, so the real-data B200 gain is
larger than the synthetic table. The sm_90 kernel has no union but streams
K/V per warpgroup, which at ~450 TFLOPS is ~7 TB/s of L2 → SMEM traffic:
bandwidth, not wgmma, bounds it.

**End to end, H3 turbo (4step-vsa), B200** (US-CA-2, `fv-gpucheck --mode
fast h3 gen`, warm, seed 7, fox prompt, same pod and binary; the profile
column is the traced fv-serve job above):

| | profile (tma2) | tma2 (this run) | dc | change |
|---|---:|---:|---:|---:|
| 768p denoise | 11.88 s | 11.87 s | 9.18 s | **−22.6%** |
| 768p text + denoise + decode | — | 15.32 s | 12.62 s | −17.6% |
| 1080p denoise | 33.00 s | 32.73 s | 24.46 s | **−25.3%** |
| 1080p text + denoise + decode | — | 39.87 s | 31.59 s | −20.8% |
| peak allocated (768p / 1080p) | — | 65.2 / 76.5 GiB | 64.1 / 74.3 GiB | −1 / −2 GiB |

Against the profile's full job (run_s 15.33 s / 40.74 s, with x264 encode
and upload), the saving is −17.5% (768p) and −20.3% (1080p); the estimate
was −24% / −32%. The gap is the estimate's assumed 3.5x fine stage: the
measured real-data fine stage is ~2x (the union, and a raw tcgen05 rate of
~650-690 TFLOPS, below the dense kernel's 1.1 PFLOPS).

**Numerics, E2E.** Step-1 per-block rel-L2, dc vs tma2 (768p, same pod):
block 0 1.2e-4, 1 1.6e-3, 6 3.8e-3, 12 5.1e-3, 18 3.1e-2, 24 3.4e-2, 30
0.10, 36 0.21, 42 0.32, 48 0.56, 49 0.44; latents after steps 1 / 2 / 3:
1.0e-2 / 2.8e-2 / 8.6e-2. That is the order of the recorded Rust-vs-Python
oracle for this recipe (`docs/oracle.md`: 1.35e-2 at block 18, 0.36 at
block 49; latents 7.2e-3 / 2.5e-2 / 8.5e-2): bf16-floor early, then VSA's
top-k turning tiny differences into different tile choices. Clips (dc vs
tma2): 768p PSNR 17.2 dB, LPIPS 0.28, sharpness 0.98, jitter 0.96 (min
0.86); 1080p LPIPS 0.36. **Control**: the incumbent VSA with only the
dense-prefix SDPA kernel switched (`FASTVIDEO_FLASH_KERNEL=v2`) moves the
768p clip by LPIPS 0.32. The dc kernel's divergence is H3's sensitivity to
rounding, not a kernel defect. The Python oracle itself (an upstream
reference pod) was not re-run on B200.

**End to end, H3 turbo, H100 SXM** (US-CA-2, same method):

| | tma2 | dc | change |
|---|---:|---:|---:|
| 768p denoise | 19.46 s | 16.34 s | **−16.0%** |
| 768p text + denoise + decode | 24.58 s | 21.50 s | −12.5% |
| 1080p denoise | 49.62 s | 39.44 s | **−20.5%** |
| 1080p text + denoise + decode | 60.17 s | 49.96 s | −17.0% |
| peak allocated (768p / 1080p) | 57.0 / 68.3 GiB | 55.9 / 66.1 GiB | −1 / −2 GiB |

Against the profile's full H100 jobs (run_s 27.73 s / 57.21 s) the saving
is −11.1% (768p) and −17.8% (1080p); the estimate was −13% / −22%. Clips dc
vs tma2: LPIPS 0.22 (768p) / 0.23 (1080p), within the B200 control's 0.32.

## Top 20 kernels per workload

Full lists in `artifacts/perf/datacenter/<gpu>/<workload>/kernels.csv`,
with grid, block, registers, shared memory, and min/max duration.

### H3 turbo 768p (VSA)

B200, device time 15.06 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `vsa_mma_attn_tma2` | attention | 28.3 % | 4265 | 200 | 21326 |
| 2 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 11.3 % | 1697 | 720 | 2357 |
| 3 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | gemm | 9.3 % | 1399 | 3296 | 424 |
| 4 | `h3_qk_norm_rope` | norm_elementwise | 6.8 % | 1019 | 404 | 2523 |
| 5 | `vsa_tile_qkv` | attention | 6.0 % | 902 | 600 | 1503 |
| 6 | `flash_mma_fwd2_d64` | attention | 4.4 % | 667 | 1008 | 662 |
| 7 | `h3v_qkv_heads_bf16` | layout_copy | 3.3 % | 500 | 1008 | 496 |
| 8 | `h3_swiglu_mx` | norm_elementwise | 3.1 % | 469 | 180 | 2605 |
| 9 | `h3v_swiglu_bf16` | norm_elementwise | 2.8 % | 419 | 1008 | 415 |
| 10 | `cast_f32_bf16` | cast_quant | 2.8 % | 418 | 1438 | 291 |
| 11 | `vsa_combine` | attention | 2.4 % | 360 | 200 | 1800 |
| 12 | `h3_res_gate_norm_mod` | norm_elementwise | 2.0 % | 305 | 200 | 1524 |
| 13 | `nvjet_sm100_tst_256x240_64x4_2x1_2cta_v_bz_TNT` | gemm | 1.7 % | 255 | 1008 | 253 |
| 14 | `vsa_tile_mean` | attention | 1.7 % | 254 | 600 | 423 |
| 15 | `fvf_merge_heads_mx` | layout_copy | 1.7 % | 250 | 180 | 1391 |
| 16 | `h3v_residual_norm_bf16` | norm_elementwise | 1.4 % | 209 | 2044 | 102 |
| 17 | `fvf_h3_gate_res_norm_mod` | norm_elementwise | 1.3 % | 192 | 196 | 982 |
| 18 | `fill_f` | layout_copy | 1.1 % | 164 | 400 | 410 |
| 19 | `fvf_split_heads_rows` | layout_copy | 1.0 % | 153 | 402 | 380 |
| 20 | `mxfp8_quantize` | cast_quant | 0.9 % | 134 | 180 | 745 |

H100, device time 23.22 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `vsa_mma_attn_tma2` | attention | 21.9 % | 5076 | 200 | 25379 |
| 2 | `nvjet_sm90_qqtst_128x160_128x5_2x1_v_bz_coopA_algo2_TNT` | gemm | 16.7 % | 3885 | 1200 | 3238 |
| 3 | `w8a8_quantize` | cast_quant | 6.4 % | 1482 | 808 | 1834 |
| 4 | `h3_qk_norm_rope` | norm_elementwise | 5.2 % | 1205 | 404 | 2982 |
| 5 | `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_TNT` | gemm | 4.7 % | 1084 | 1008 | 1075 |
| 6 | `vsa_tile_qkv` | attention | 4.5 % | 1037 | 600 | 1728 |
| 7 | `flash_mma_fwd2_d64` | attention | 3.4 % | 793 | 1008 | 786 |
| 8 | `nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT` | gemm | 3.0 % | 692 | 200 | 3458 |
| 9 | `nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_TNT` | gemm | 2.9 % | 678 | 2016 | 336 |
| 10 | `cast_f32_bf16` | cast_quant | 2.8 % | 659 | 1618 | 407 |
| 11 | `mx_swiglu` | norm_elementwise | 2.7 % | 631 | 202 | 3122 |
| 12 | `h3v_qkv_heads_bf16` | layout_copy | 2.7 % | 620 | 1008 | 615 |
| 13 | `vsa_combine` | attention | 2.1 % | 483 | 200 | 2417 |
| 14 | `h3v_swiglu_bf16` | norm_elementwise | 1.9 % | 448 | 1008 | 444 |
| 15 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 1.8 % | 415 | 1040 | 399 |
| 16 | `mx_merge_heads` | layout_copy | 1.6 % | 369 | 202 | 1828 |
| 17 | `h3v_residual_norm_bf16` | norm_elementwise | 1.2 % | 273 | 2044 | 134 |
| 18 | `h3_res_gate_norm_mod` | norm_elementwise | 1.1 % | 252 | 200 | 1258 |
| 19 | `fvf_split_heads_rows` | layout_copy | 1.1 % | 247 | 402 | 613 |
| 20 | `vsa_tile_mean` | attention | 1.0 % | 241 | 600 | 402 |

### H3 max 768p (Sol-H3)

B200, device time 14.29 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `sol_mma_fwd_x4f` | attention | 27.2 % | 3886 | 144 | 26987 |
| 2 | `fa_dc100_fwd_d128` | attention | 14.9 % | 2136 | 202 | 10573 |
| 3 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 11.9 % | 1697 | 720 | 2357 |
| 4 | `h3_qk_norm_rope` | norm_elementwise | 7.3 % | 1050 | 404 | 2599 |
| 5 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | gemm | 7.3 % | 1042 | 3116 | 334 |
| 6 | `flash_mma_fwd2_d64` | attention | 4.7 % | 668 | 1008 | 663 |
| 7 | `h3v_qkv_heads_bf16` | layout_copy | 3.5 % | 501 | 1008 | 497 |
| 8 | `h3_swiglu_mx` | norm_elementwise | 3.4 % | 480 | 180 | 2669 |
| 9 | `h3v_swiglu_bf16` | norm_elementwise | 2.9 % | 416 | 1008 | 412 |
| 10 | `mx_block_copy` | layout_copy | 2.5 % | 358 | 596 | 600 |
| 11 | `h3_res_gate_norm_mod` | norm_elementwise | 2.2 % | 312 | 200 | 1558 |
| 12 | `fvf_h3_gate_res_norm_mod` | norm_elementwise | 2.1 % | 300 | 196 | 1529 |
| 13 | `fvf_merge_heads_mx` | layout_copy | 1.8 % | 257 | 180 | 1425 |
| 14 | `nvjet_sm100_tst_256x240_64x4_2x1_2cta_v_bz_TNT` | gemm | 1.8 % | 255 | 1008 | 253 |
| 15 | `h3v_residual_norm_bf16` | norm_elementwise | 1.5 % | 209 | 2044 | 102 |
| 16 | `h3v_merge_heads_bf16` | layout_copy | 0.8 % | 110 | 1008 | 109 |
| 17 | `nvjet_sm100_tst_256x256_64x4_2x1_2cta_v_bz_TNT` | gemm | 0.5 % | 75 | 20 | 3731 |
| 18 | `fvf_split_heads_rows` | layout_copy | 0.5 % | 67 | 202 | 332 |
| 19 | `gather_nd` | layout_copy | 0.5 % | 64 | 403 | 159 |
| 20 | `mx_swiglu` | norm_elementwise | 0.4 % | 60 | 22 | 2740 |

H100, device time 22.62 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `sol_mma_fwd_x4f` | attention | 19.6 % | 4436 | 144 | 30802 |
| 2 | `fa_dc90_fwd_d128` | attention | 18.7 % | 4237 | 202 | 20976 |
| 3 | `nvjet_sm90_qqtst_128x160_128x5_2x1_v_bz_coopA_algo2_TNT` | gemm | 17.2 % | 3902 | 1200 | 3251 |
| 4 | `w8a8_quantize` | cast_quant | 6.7 % | 1520 | 808 | 1881 |
| 5 | `h3_qk_norm_rope` | norm_elementwise | 5.5 % | 1252 | 404 | 3098 |
| 6 | `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_TNT` | gemm | 4.8 % | 1090 | 1008 | 1081 |
| 7 | `flash_mma_fwd2_d64` | attention | 3.5 % | 798 | 1008 | 792 |
| 8 | `nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_TNT` | gemm | 3.0 % | 682 | 2016 | 338 |
| 9 | `mx_swiglu` | norm_elementwise | 2.8 % | 643 | 202 | 3181 |
| 10 | `h3v_qkv_heads_bf16` | layout_copy | 2.8 % | 623 | 1008 | 618 |
| 11 | `h3v_swiglu_bf16` | norm_elementwise | 2.0 % | 450 | 1008 | 446 |
| 12 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN` | gemm | 1.8 % | 417 | 1040 | 401 |
| 13 | `mx_block_copy` | layout_copy | 1.8 % | 402 | 596 | 675 |
| 14 | `mx_merge_heads` | layout_copy | 1.7 % | 377 | 202 | 1868 |
| 15 | `h3v_residual_norm_bf16` | norm_elementwise | 1.2 % | 274 | 2044 | 134 |
| 16 | `h3_res_gate_norm_mod` | norm_elementwise | 1.1 % | 256 | 200 | 1281 |
| 17 | `fvf_h3_gate_res_norm_mod` | norm_elementwise | 1.0 % | 230 | 196 | 1172 |
| 18 | `amax_abs_mixed` | cast_quant | 0.8 % | 189 | 808 | 234 |
| 19 | `h3v_merge_heads_bf16` | layout_copy | 0.6 % | 137 | 1008 | 136 |
| 20 | `fvf_split_heads_rows` | layout_copy | 0.4 % | 89 | 202 | 440 |

### H3 turbo 1080p (VSA)

B200, device time 39.31 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `vsa_mma_attn_tma2` | attention | 41.2 % | 16183 | 200 | 80915 |
| 2 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | gemm | 6.8 % | 2681 | 4211 | 637 |
| 3 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 6.8 % | 2672 | 2160 | 1237 |
| 4 | `h3_qk_norm_rope` | norm_elementwise | 5.4 % | 2133 | 404 | 5279 |
| 5 | `vsa_tile_qkv` | attention | 4.8 % | 1908 | 600 | 3179 |
| 6 | `flash_mma_fwd2_d64` | attention | 3.6 % | 1427 | 2016 | 708 |
| 7 | `h3v_qkv_heads_bf16` | layout_copy | 2.7 % | 1078 | 2016 | 535 |
| 8 | `h3_swiglu_mx` | norm_elementwise | 2.5 % | 1001 | 1800 | 556 |
| 9 | `h3v_swiglu_bf16` | norm_elementwise | 2.3 % | 901 | 2016 | 447 |
| 10 | `cast_f32_bf16` | cast_quant | 2.1 % | 814 | 1222 | 666 |
| 11 | `mx_block_copy` | layout_copy | 1.9 % | 763 | 4020 | 190 |
| 12 | `vsa_combine` | attention | 1.9 % | 748 | 200 | 3739 |
| 13 | `nvjet_sm100_qqtst_256x128_128x5_2x2f_2cta_h_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 1.8 % | 696 | 1800 | 386 |
| 14 | `nvjet_sm100_tst_128x256_64x6_4x1f_2cta_v_bz_TNT` | gemm | 1.7 % | 684 | 3528 | 194 |
| 15 | `mxfp8_quantize` | cast_quant | 1.4 % | 549 | 1980 | 277 |
| 16 | `vsa_tile_mean` | attention | 1.3 % | 517 | 600 | 862 |
| 17 | `fvf_merge_heads_mx` | layout_copy | 1.2 % | 487 | 180 | 2703 |
| 18 | `h3v_residual_norm_bf16` | norm_elementwise | 1.1 % | 446 | 4088 | 109 |
| 19 | `h3_res_gate_norm_mod` | norm_elementwise | 1.1 % | 414 | 200 | 2068 |
| 20 | `fvf_h3_gate_res_norm_mod` | norm_elementwise | 1.0 % | 379 | 196 | 1933 |

H100, device time 56.18 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `vsa_mma_attn_tma2` | attention | 34.3 % | 19245 | 200 | 96223 |
| 2 | `nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT` | gemm | 6.3 % | 3538 | 1964 | 1801 |
| 3 | `nvjet_sm90_qqtst_128x160_128x5_1x2_h_bz_coopA_algo2_TNT` | gemm | 5.7 % | 3194 | 800 | 3992 |
| 4 | `nvjet_sm90_qqtst_144x128_128x6_1x2_h_bz_coopA_algo2_TNN` | gemm | 5.6 % | 3133 | 1800 | 1741 |
| 5 | `w8a8_quantize` | cast_quant | 5.3 % | 2971 | 4408 | 674 |
| 6 | `h3_qk_norm_rope` | norm_elementwise | 4.7 % | 2627 | 404 | 6502 |
| 7 | `vsa_tile_qkv` | attention | 4.1 % | 2315 | 600 | 3858 |
| 8 | `flash_mma_fwd2_d64` | attention | 3.1 % | 1721 | 2016 | 854 |
| 9 | `nvjet_sm90_qqtst_128x160_128x5_2x1_v_bz_coopA_algo2_TNT` | gemm | 3.0 % | 1667 | 2000 | 834 |
| 10 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT` | gemm | 2.4 % | 1346 | 3528 | 381 |
| 11 | `h3v_qkv_heads_bf16` | layout_copy | 2.4 % | 1337 | 2016 | 663 |
| 12 | `cast_f32_bf16` | cast_quant | 2.3 % | 1295 | 1402 | 924 |
| 13 | `mx_swiglu` | norm_elementwise | 2.3 % | 1271 | 2002 | 635 |
| 14 | `h3v_swiglu_bf16` | norm_elementwise | 1.7 % | 965 | 2016 | 478 |
| 15 | `vsa_combine` | attention | 1.7 % | 957 | 200 | 4787 |
| 16 | `mx_block_copy` | layout_copy | 1.5 % | 848 | 4020 | 211 |
| 17 | `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_TNT` | gemm | 1.5 % | 824 | 1764 | 467 |
| 18 | `mx_merge_heads` | layout_copy | 1.3 % | 714 | 202 | 3535 |
| 19 | `h3v_residual_norm_bf16` | norm_elementwise | 1.0 % | 578 | 4088 | 141 |
| 20 | `fvf_split_heads_rows` | layout_copy | 0.9 % | 500 | 402 | 1244 |

### LTX-2.5 1080p 6 s

B200, device time 12.05 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `sol_mma_fwd_x4f` | attention | 17.0 % | 2048 | 141 | 14523 |
| 2 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_bias_TNT` | gemm | 11.2 % | 1344 | 4131 | 325 |
| 3 | `fvf_ltx_qk_norm_rope` | norm_elementwise | 8.0 % | 966 | 6336 | 152 |
| 4 | `fvf_ltx_res_norm_mod` | norm_elementwise | 8.0 % | 965 | 5280 | 183 |
| 5 | `fa_dc100_fwd_d128` | attention | 6.5 % | 783 | 915 | 856 |
| 6 | `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT` | gemm | 6.1 % | 734 | 528 | 1389 |
| 7 | `mx_unary` | norm_elementwise | 5.2 % | 622 | 2288 | 272 |
| 8 | `ltxv_norm_silu` | vae_conv | 4.0 % | 480 | 3519 | 137 |
| 9 | `nvjet_sm100_tst_256x256_64x4_2x1_2cta_v_bz_TNT` | gemm | 3.7 % | 448 | 144 | 3112 |
| 10 | `nvjet_sm100_tst_256x224_64x4_2x1_2cta_v_bz_TNT` | gemm | 2.5 % | 299 | 384 | 778 |
| 11 | `mx_residual_gate` | norm_elementwise | 2.3 % | 279 | 1056 | 264 |
| 12 | `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 2.3 % | 279 | 623 | 448 |
| 13 | `flash_mma_fwd2_d64` | attention | 1.9 % | 235 | 2112 | 111 |
| 14 | `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 1.8 % | 219 | 1456 | 151 |
| 15 | `fvf_ltx_gate_merge` | norm_elementwise | 1.5 % | 179 | 3168 | 57 |
| 16 | `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 1.3 % | 161 | 1149 | 140 |
| 17 | `ltxv_d2s_bias` | vae_conv | 1.1 % | 129 | 270 | 479 |
| 18 | `nvjet_sm100_tst_128x192_64x7_2x1_2cta_v_bz_bias_TNT` | gemm | 1.0 % | 117 | 1152 | 101 |
| 19 | `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 0.9 % | 114 | 409 | 278 |
| 20 | `fvf_split_heads_rows` | layout_copy | 0.8 % | 91 | 3168 | 29 |

H100, device time 18.63 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `sol_mma_fwd_x4f` | attention | 13.1 % | 2433 | 141 | 17256 |
| 2 | `nvjet_sm90_tst_256x160_64x4_2x1_v_bz_coopA_TNT` | gemm | 12.8 % | 2385 | 672 | 3549 |
| 3 | `nvjet_sm90_tst_256x144_64x4_1x2_h_bz_coopA_bias_TNT` | gemm | 9.8 % | 1829 | 1443 | 1268 |
| 4 | `fa_dc90_fwd_d128` | attention | 7.2 % | 1337 | 915 | 1461 |
| 5 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256…` | vae_conv | 6.9 % | 1293 | 2503 | 517 |
| 6 | `fvf_ltx_res_norm_mod` | norm_elementwise | 5.6 % | 1050 | 5280 | 199 |
| 7 | `fvf_ltx_qk_norm_rope` | norm_elementwise | 5.6 % | 1036 | 6336 | 163 |
| 8 | `nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_bias_TNN` | gemm | 5.3 % | 994 | 2304 | 432 |
| 9 | `mx_unary` | norm_elementwise | 3.8 % | 708 | 2288 | 310 |
| 10 | `nvjet_sm90_tst_256x160_64x4_1x2_h_bz_coopA_TNT` | gemm | 3.3 % | 614 | 384 | 1600 |
| 11 | `ltxv_norm_silu` | vae_conv | 2.9 % | 538 | 3519 | 153 |
| 12 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize64x…` | vae_conv | 1.9 % | 360 | 1073 | 335 |
| 13 | `mx_residual_gate` | norm_elementwise | 1.6 % | 308 | 1056 | 291 |
| 14 | `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_bias_TNT` | gemm | 1.6 % | 298 | 2208 | 135 |
| 15 | `flash_mma_fwd2_d64` | attention | 1.4 % | 254 | 2112 | 120 |
| 16 | `fvf_ltx_gate_merge` | norm_elementwise | 1.1 % | 204 | 3168 | 64 |
| 17 | `ltxv_d2s_bias` | vae_conv | 0.8 % | 139 | 270 | 515 |
| 18 | `fvf_split_heads_rows` | layout_copy | 0.6 % | 104 | 3168 | 33 |
| 19 | `nvjet_sm90_tst_128x224_64x4_2x1_v_bz_coopA_bias_TNT` | gemm | 0.4 % | 82 | 392 | 209 |
| 20 | `cast_f32_bf16` | cast_quant | 0.3 % | 64 | 19551 | 3 |

### Wan 2.2 5B 704p

B200, device time 9.00 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `gather_nd` | layout_copy | 19.6 % | 1762 | 315 | 5593 |
| 2 | `block_copy` | layout_copy | 16.3 % | 1468 | 2302 | 638 |
| 3 | `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 7.5 % | 673 | 262 | 2567 |
| 4 | `fa_dc100_fwd_d128` | attention | 7.2 % | 647 | 180 | 3595 |
| 5 | `cast_f32_bf16` | cast_quant | 5.5 % | 492 | 1345 | 366 |
| 6 | `add_bias_inplace` | norm_elementwise | 5.3 % | 476 | 806 | 591 |
| 7 | `quant_linear_epilogue` | cast_quant | 4.8 % | 434 | 570 | 762 |
| 8 | `rms_norm_channels` | vae_conv | 4.7 % | 423 | 480 | 881 |
| 9 | `cutlass3x_sm100_tensorop_s256x256x16implicit_gemm_fprop_bf16_bf16_f32_void_b…` | vae_conv | 3.1 % | 282 | 155 | 1822 |
| 10 | `mxfp8_quantize` | cast_quant | 2.9 % | 259 | 570 | 454 |
| 11 | `cast_bf16_f32_bias_act` | cast_quant | 2.7 % | 239 | 663 | 361 |
| 12 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 2.4 % | 214 | 540 | 397 |
| 13 | `cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, f…` | vae_conv | 2.4 % | 213 | 1289 | 165 |
| 14 | `elem_add` | norm_elementwise | 2.3 % | 210 | 288 | 729 |
| 15 | `cutlass3x_sm100_tensorop_s256x256x8tf32implicit_gemm_fprop_f32_f32_f32_void_…` | vae_conv | 1.8 % | 162 | 31 | 5223 |
| 16 | `cutlass3x_sm100_tensorop_s256x256x8tf32implicit_gemm_fprop_f32_f32_f32_void_…` | vae_conv | 1.6 % | 146 | 34 | 4280 |
| 17 | `wan_qk_norm_rope16` | norm_elementwise | 1.4 % | 127 | 300 | 424 |
| 18 | `upsample_nearest` | vae_conv | 1.4 % | 125 | 63 | 1979 |
| 19 | `wan_res_ln` | norm_elementwise | 1.1 % | 99 | 180 | 550 |
| 20 | `cutlass3x_sm100_tensorop_s32x256x16_conv3d_fprop_weight_stationary_nq_2d_til…` | vae_conv | 1.1 % | 97 | 31 | 3141 |

H100, device time 12.29 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `gather_nd` | layout_copy | 17.0 % | 2086 | 315 | 6622 |
| 2 | `block_copy` | layout_copy | 13.7 % | 1680 | 2302 | 730 |
| 3 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize256…` | vae_conv | 12.6 % | 1552 | 255 | 6086 |
| 4 | `fa_dc90_fwd_d128` | attention | 10.4 % | 1275 | 180 | 7083 |
| 5 | `sm90_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize256x1…` | vae_conv | 4.6 % | 570 | 46 | 12397 |
| 6 | `cast_f32_bf16` | cast_quant | 4.6 % | 562 | 1915 | 294 |
| 7 | `add_bias_inplace` | norm_elementwise | 4.3 % | 529 | 806 | 657 |
| 8 | `rms_norm_channels` | vae_conv | 4.2 % | 520 | 480 | 1082 |
| 9 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128…` | vae_conv | 3.7 % | 456 | 374 | 1220 |
| 10 | `nvjet_sm90_tst_192x192_64x4_1x2_h_bz_coopB_bias_TNN` | gemm | 3.6 % | 437 | 360 | 1215 |
| 11 | `cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, f…` | vae_conv | 2.5 % | 309 | 1320 | 234 |
| 12 | `cast_bf16_f32_bias_act` | cast_quant | 2.3 % | 283 | 663 | 427 |
| 13 | `nvjet_sm90_tst_192x208_64x4_1x2_h_bz_coopB_bias_TNT` | gemm | 2.2 % | 269 | 93 | 2889 |
| 14 | `elem_add` | norm_elementwise | 2.2 % | 268 | 288 | 930 |
| 15 | `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_bias_TNT` | gemm | 1.4 % | 173 | 90 | 1925 |
| 16 | `mx_unary` | norm_elementwise | 1.4 % | 169 | 91 | 1858 |
| 17 | `wan_qk_norm_rope16` | norm_elementwise | 1.3 % | 155 | 300 | 516 |
| 18 | `upsample_nearest` | vae_conv | 1.2 % | 143 | 63 | 2272 |
| 19 | `cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, f…` | vae_conv | 1.1 % | 137 | 660 | 207 |
| 20 | `wan_res_ln` | norm_elementwise | 1.0 % | 124 | 180 | 690 |

### SF-Wan blocks (10 s window)

B200, device time 9.58 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `fa_dc100_fwd_d128` | attention | 37.7 % | 3611 | 7140 | 506 |
| 2 | `quant_linear_epilogue` | cast_quant | 16.8 % | 1609 | 21420 | 75 |
| 3 | `mxfp8_quantize` | cast_quant | 11.5 % | 1101 | 21420 | 51 |
| 4 | `mx_block_copy` | layout_copy | 5.8 % | 558 | 15133 | 37 |
| 5 | `wan_qk_norm_rope16` | norm_elementwise | 5.1 % | 491 | 10693 | 46 |
| 6 | `wan_res_ln` | norm_elementwise | 4.2 % | 406 | 7140 | 57 |
| 7 | `mx_merge_heads` | layout_copy | 3.2 % | 310 | 7140 | 43 |
| 8 | `nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 2.6 % | 247 | 7140 | 35 |
| 9 | `nvjet_sm100_qqtst_256x128_128x5_2x4f_2cta_h_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 1.9 % | 184 | 3570 | 51 |
| 10 | `wan_res_gate` | norm_elementwise | 1.6 % | 150 | 3570 | 42 |
| 11 | `mx_ln_adaln_e` | norm_elementwise | 1.2 % | 116 | 3689 | 31 |
| 12 | `nvjet_sm100_qqtst_128x128_128x8_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT` | gemm | 1.1 % | 110 | 10710 | 10 |
| 13 | `cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float, (bool)0, (…` | vae_conv | 0.8 % | 81 | 1647 | 49 |
| 14 | `block_copy` | layout_copy | 0.8 % | 78 | 1486 | 53 |
| 15 | `clamp_f` | norm_elementwise | 0.8 % | 75 | 810 | 93 |
| 16 | `add_bias_inplace` | norm_elementwise | 0.7 % | 67 | 901 | 74 |
| 17 | `fvf_split_heads_rows` | layout_copy | 0.5 % | 50 | 3570 | 14 |
| 18 | `cutlass3x_sm100_tensorop_s64x256x8_conv3d_fprop_weight_stationary_nq_2d_tile…` | vae_conv | 0.5 % | 48 | 327 | 148 |
| 19 | `cudnn::engines_precompiled::nhwcToNchwKernel<float, float, float, (bool)1, (…` | vae_conv | 0.4 % | 40 | 837 | 48 |
| 20 | `upsample_nearest` | vae_conv | 0.4 % | 35 | 81 | 429 |

H100, device time 9.67 s:

| # | Kernel | Category | Share | Total ms | Count | Mean µs |
|---:|---|---|---:|---:|---:|---:|
| 1 | `fa_dc90_fwd_d128` | attention | 51.4 % | 4971 | 5669 | 877 |
| 2 | `nvjet_sm90_tst_192x144_64x5_2x1_v_bz_coopB_bias_TNT` | gemm | 7.5 % | 723 | 11340 | 64 |
| 3 | `mx_unary` | norm_elementwise | 5.6 % | 544 | 2835 | 192 |
| 4 | `nvjet_sm90_tst_192x192_64x4_1x1_h_bz_coopB_bias_TNN` | gemm | 5.4 % | 521 | 2835 | 184 |
| 5 | `wan_qk_norm_rope16` | norm_elementwise | 4.8 % | 462 | 8503 | 54 |
| 6 | `mx_block_copy` | layout_copy | 4.6 % | 441 | 12163 | 36 |
| 7 | `wan_res_ln` | norm_elementwise | 4.0 % | 386 | 5670 | 68 |
| 8 | `mx_merge_heads` | layout_copy | 2.9 % | 283 | 5670 | 50 |
| 9 | `nvjet_sm90_tst_192x176_64x4_2x1_v_bz_coopB_bias_TNT` | gemm | 2.7 % | 262 | 2834 | 92 |
| 10 | `sm80_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize256x…` | vae_conv | 1.5 % | 142 | 242 | 587 |
| 11 | `wan_res_gate` | norm_elementwise | 1.4 % | 140 | 2835 | 49 |
| 12 | `mx_ln_adaln_e` | norm_elementwise | 1.1 % | 103 | 2929 | 35 |
| 13 | `cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float, (bool)0, (…` | vae_conv | 0.8 % | 79 | 528 | 150 |
| 14 | `clamp_f` | norm_elementwise | 0.8 % | 77 | 660 | 117 |
| 15 | `block_copy` | layout_copy | 0.8 % | 75 | 1211 | 62 |
| 16 | `add_bias_inplace` | norm_elementwise | 0.6 % | 62 | 732 | 84 |
| 17 | `cast_f32_bf16` | cast_quant | 0.5 % | 52 | 17943 | 3 |
| 18 | `fvf_split_heads_rows` | layout_copy | 0.4 % | 42 | 2834 | 15 |
| 19 | `upsample_nearest` | vae_conv | 0.3 % | 33 | 66 | 497 |
| 20 | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_tf32f32_tf32f32_f32_nhwckrsc_n…` | vae_conv | 0.3 % | 30 | 22 | 1354 |

## Spend

| Pod | GPU | Lifetime | Cost | Notes |
|---|---|---:|---:|---|
| `x05b4xj6vnzmky` | B200 | ≈ 11 min | $1.25 | round 1. `nsys launch` rejected `--cpuctxsw` (a start/profile-only switch), so fv-serve never started. The pod's 10-minute idle guard deleted it. |
| `df1338tu3n9spm` | L4 | 184 s | $0.03 | nsys option debugging, no volume |
| `aimjsaiw5sp3mt` | B200 | 735 s | $1.39 | round 1 data, lost in the container restart |
| `gwvhcx3usfmawd` | H100 SXM | 878 s | $0.85 | round 1 data, lost in the container restart |
| `52rgddmv8h5dvo` | H100 SXM | 661 s | $0.64 | round 2 (committed) |
| `tz3d74y6u0ct7s` | B200 | 675 s | $1.27 | round 2 (committed) |
| **Total** | | | **≈ $5.43** | budget $8 |

Guards on every pod:

- a wall-clock backstop of 1500–3000 s;
- an on-pod idle guard (`FV_E2E_IDLE_DELETE_MIN=10`);
- a local balance watchdog (delete below $12, then $15 in round 2).

Every pod was deleted and verified with a 404 on GET. The round-1 backstop
processes died with the container restart, after their pods had been
deleted. The price was checked before every create. Ledger:
`artifacts/perf/datacenter/ledger.tsv`.

## Reproduce

```bash
S=<scratch>; export FV_E2E_STATE=$S/b200.json RUNPOD_GPU_TYPES="NVIDIA B200" RUNPOD_GPU_MAX_DPH=6.8 \
  FV_POD_CAP_S=1500 RUNPOD_VOLUME_ID=s2k01690bi FV_E2E_NAME=fv-dcprof-b200- FV_MIN_BALANCE=15 \
  FV_E2E_ENV_JSON='{"FV_E2E_IDLE_DELETE_MIN":"10"}'
scripts/serve/e2e/pod.sh up ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:01c9fcf58d3b8504b1255879adcc1752366ecac92345d888eea87d98f1ecf895 /etc/fv/none.toml
# upload scripts/gpu/nsys_profile.py, scripts/gpu/nsys-pod.sh, scripts/serve/e2e/bench.py to /e2e (sidecar PUT /bundle)
scripts/serve/e2e/pod.sh run "FV_KEY=<key> bash /e2e/scripts/gpu/nsys-pod.sh key; bash /e2e/scripts/gpu/nsys-pod.sh install; \
  bash /e2e/scripts/gpu/nsys-pod.sh serve /etc/fv/runpod.toml; bash /e2e/scripts/gpu/nsys-pod.sh ready 400"
scripts/serve/e2e/pod.sh run "bash /e2e/scripts/gpu/nsys-pod.sh job h3t-768 '{\"model\":\"fasth3\",\"aspect_ratio\":\"16:9\",\"short_edge\":768,\"seconds\":5}'"
# ... per family: serve <config>, ready, [warm <label> <body>], job <label> <body>
# SF-Wan: FASTVIDEO_TAE_DIR=/workspace/weights/auxiliary/tae bash nsys-pod.sh gpucheck sfwan 10 --out /e2e/gc --mode fast \
#   wan stream --weights /workspace/weights/sfwan21-1.3b --prompt '<fox>' --run g20,seconds=20,rope=rebased,sink=3
# results: /e2e/prof/out/<label>/{summary.json,kernels.csv,timeline.csv,gaps.csv,kbins.csv}
scripts/serve/e2e/pod.sh down
```

Job bodies:

| Label | Body |
|---|---|
| `h3t-1080` | `short_edge: 1080` |
| `h3m-768` | `"model":"sol-h3"` |
| `ltx-1080` | `"model":"ltx25-distill-sol","short_edge":1080,"seconds":6`, after a 720p warm job |
| `wan5b-704` | `"model":"fastwan22-ti2v-5b","short_edge":704`, after a 480p warm job |

## Artifacts

`artifacts/perf/datacenter/`:

- `index.json`: every cell's window, busy, idle attribution, categories,
  stages and analytic throughput.
- `<gpu>/<workload>/`:
  - `summary.json`: the full analysis, plus stages and the job's engine
    metrics;
  - `kernels.csv`: every kernel;
  - `timeline.csv`: 0.5 s bins;
  - `gaps.csv`: every idle gap ≥ 20 µs with its attributed cause.
- `<gpu>/runs.jsonl` and `warm.jsonl`: the job records.
- `<gpu>/sfwan-run.txt`: the SF-Wan rollout report.
- `<gpu>/kernel-selection-log.txt`: serve-log lines naming the quant, VSA,
  Sol and SDPA paths.
- `ledger.tsv`: the pod ledger.

The 0.25 s per-kernel bins (`kbins.csv`) and full serve logs are on the
backup branch `wip/datacenter-profile` (commit `6768e9b`), not on main.
