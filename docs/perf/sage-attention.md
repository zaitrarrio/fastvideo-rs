# SageAttention2 / SageAttention3 against our attention kernels

Status (2026-10-01): Phases 1, 2 and 3 are done (Phase 3: section 6). The
kernel code (`wan/attn_sage.{cu,rs}`) and the harness (`scripts/gpu/sage/`:
`bench.py`, `capture.py`, `summarize.py`, `vast-run.sh`) stay on branch
`wip/sage-attn` (also merged into `wip/phase3-pro6000`), not on main: the
H3 five-prompt gate does not pass (section 6.2), so the port is not merged.
It is default-off either way; merging it is the owner's call (section 6.6).

**Summary**

* **SageAttention2 (INT8 QK, FP8 PV) clears the bar on sm_120.** On the
  RTX PRO 6000 it is 1.52-1.77x over our bf16 dense kernel at every shape
  >= 7k tokens. Accuracy is within the stated tolerance on peaked, outlier
  and real FastWan inputs: worst cosine 0.9990 / rel-L2 0.044 synthetic,
  0.9998 / 0.021 real.
* **Where it does not help.** LTX stage 1 (<= 4k tokens) gains nothing. On
  H100 it is only ~1.1-1.2x over our `fa_dc90`, so it does not clear there.
* **SageAttention3 (NVFP4) fails accuracy.** rel-L2 0.19-0.36 synthetic,
  up to 0.14 on real data, cosine down to 0.94, though it is the fastest
  (up to 2.2x on the PRO 6000). It exists for sm_120/121 only.
* **Ported** as `wan::attn_sage` (opt-in `FASTVIDEO_ATTN_SAGE=2`):
  upstream-level accuracy, 1.42-1.62x over fwd2 at >= 12k tokens (about
  10 % behind upstream's kernel). The Rust parity group passes on RTX PRO
  6000 (6/6 cases), and the kernel is 1.41-1.50x over our real dense
  default (cuDNN or fwd2, whichever is faster per shape).
* **End to end (Phase 3, RTX PRO 6000).** Denoise speedups where dense
  attention runs: Sol-H3 4-step dense 1.33x, H3 max (Sol-H3 engine ladder)
  1.13x, LTX-2.5 dense stage 2 at 1080p 1.15x. Nothing where the serving
  recipe is already sparse: h3-turbo (VSA) 1.00x (byte-identical), LTX
  turbo (Sol stage 2) 0.99-1.03x.
* **Gates.** LTX dense 1080p passes. Both H3 recipes fail on one prompt of
  five (sharpness 0.93 / 0.89 against a 0.95 floor). A bf16-only control
  (the dense kernel switched from cuDNN to fwd2) moves H3 just as far and
  fails the same gate on another prompt, so on H3 this gate cannot tell
  Sage from bf16 noise.
* **Spend:** $1.50 of Vast credit (Phases 1-2); Phase 3 about $4.36 on
  Runpod (section 6.7).

## 1. What we run today (Phase 0)

| arch | dense kernel (`attn.rs` `auto`) | measured |
|---|---|---|
| sm_120 (RTX PRO 6000, RTX 5090) | per shape, the faster of cuDNN's unified SDPA node and `flash_mma_fwd2_d128` (bf16 `mma.sync`, 128-query CTAs, double-buffered K/V) | RTX PRO 6000, H3 768p 56 x 37 966: fwd2 108 ms (375 TFLOPS), cuDNN 102-105 ms |
| sm_100 (B200) | `fa_dc100_fwd_d128` (tcgen05 + TMEM, FA4-style) | 1.15-1.21 PFLOPS, 53-54 % of peak |
| sm_90 (H100) | `fa_dc90_fwd_d128` (wgmma + TMA, FA3-style) | 0.61-0.65 PFLOPS, 61-62 % of peak |

The sparse paths (H3 turbo's VSA fine stage `vsa_mma_attn_tma2` / `vsa_dc`,
Sol-H3's `sol_mma_fwd_x4f`) are separate kernels with the same bf16
`mma.sync` inner loop as fwd2.

Our own "Sage-style" attempt, `attn_fp8` (FP8 E4M3 `Q K^T` with K smoothing,
per-16-row Q and per-64-key K scales, bf16 `P V`), was 1.19x (dense) and
1.33-1.40x (VSA fine stage) faster than bf16 on RTX PRO 6000 but failed the
stated tolerance (rel-L2 <= 5e-2 and cosine >= 0.998 against the bf16
kernel) on peaked scores: rel-L2 0.073-0.086 (docs/techniques.md, "FP8
attention"; docs/ports/h3.md section l). The note there names INT8 `Q K^T`
(SageAttention's choice) as the next candidate.

### Attention shapes we run (d = 128, self-attention, Sq = Sk)

| workload | heads | sequence | how |
|---|---|---|---|
| H3 turbo / max 480p (832x480, 5 s) | 56 | 15 100 | 256 text + 414 audio + 37 x 390 video (joint sequence) |
| H3 turbo / max 768p (1344x768, 5 s) | 56 | 37 966 | 256 + 414 + 37 x 1008 (docs/ports/h3.md section e) |
| LTX-2.5 480p (896x512 canvas), stage 1 / 2 | 32 | 1 792 / 7 168 | 16 latent frames x (14x8 / 28x16) |
| LTX-2.5 384p stage-1-only (672x384) | 32 | 4 032 | 16 x 12 x 21 |
| LTX-2.5 720p (1280x704), stage 1 / 2 | 32 | 3 520 / 14 080 | 16 x (11x20 / 22x40) |
| LTX-2.5 1080p (1920x1088) stage 2 (reference) | 32 | 32 640 | 16 x 34 x 60 |
| Wan2.2 TI2V-5B 480p (832x480, 121 frames) | 24 | 12 090 | 31 x 15 x 26 |
| FastWan2.1 1.3B 480p (832x480, 81 frames) | 12 | 32 760 | 21 x 30 x 52 |

H3 turbo runs these rows through VSA (top-k 132 / 66 of 660 video tiles at
768p), H3 max through Sol-H3, so dense is the upper bound of what a faster
dense kernel buys there; LTX, Wan 5B and dense H3 steps run dense. LTX's
audio stream (32 x 64, ~126 tokens) and all cross-attention (1 024 / 512 text
keys) are negligible and left out.

## 2. Upstream (thu-ml/SageAttention @ d1a57a5, 2026-01-17)

**SageAttention2 / 2++** (`sageattention` 2.2.0, `csrc/qattn`):
* `Q K^T` in INT8 (`mma.sync m16n8k32 s8`), K smoothed by its per-head mean
  (exact under softmax), Q/K scales per warp (`per_warp`: one per 32 query
  rows / 64 keys) or per thread (`per_thread`: the scale granularity of the
  MMA fragment each thread owns);
* `P V` in FP8 E4M3 (`m16n8k32 e4m3`), V per-channel scaled; accumulation
  `fp32` (really 22-bit on sm_89/120 FP8 MMA), `fp32+fp32` (FP22 in the MMA,
  flushed into an FP32 buffer every few tiles), or `fp32+fp16` (SageAttention2++:
  FP16-accumulating FP8 MMA, 2x the FP32-accumulate rate on consumer parts);
  or FP16 `P V` (`sageattn_qk_int8_pv_fp16_cuda`).
* `sageattn()` on sm_120/121 picks per-warp INT8 + FP8 PV + `fp32+fp16`.
  sm_90 has a wgmma kernel (`sageattn_qk_int8_pv_fp8_cuda_sm90`). sm_100 is
  compiled (the sm_89 `mma.sync` kernels) but `sageattn()` has no sm_100 case.
* Build: CUDA >= 12.8 for sm_120, torch >= 2.3, `TORCH_CUDA_ARCH_LIST`
  (`12.0` -> `sm_120a`), `pip install --no-build-isolation .`.

**SageAttention3** (`sageattention3_blackwell`, `sageattn3_blackwell()`):
* NVFP4 microscaling for both GEMMs: Q, K, V quantized to E2M1 with one E4M3
  scale per 16 elements; `P` quantized per 16 with a two-level scale.
  Preprocessing: K mean subtracted (in place on the caller's tensor), Q
  per-128-row block mean subtracted with the `qm K^T` correction added back
  (`delta_s`), sequences padded to 128.
* The MMA is `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
  (CUTLASS `SM120_16x32x64_TN_VS_NVFP4`), which PTX offers on sm_120a /
  sm_121a only. `setup.py` lists sm_100 too, but B200's block-scaled FP4 is
  tcgen05 (`UMMA`), not this warp-level MMA: **SageAttention3 is an sm_120 /
  sm_121 kernel** (RTX 5090, RTX PRO 6000, DGX Spark), not a B200 one.
* Build: CUDA >= 12.8, torch >= 2.8, a CUTLASS checkout in `csrc/cutlass`,
  built for the local GPU only. Upstream warns it is not lossless for all
  video models and suggests SageAttention2++ on the first / last steps.

## 3. Phase 1 results (kernel microbenchmark)

**Method.** `scripts/gpu/sage/bench.py` on Vast, image
`pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` (torch 2.9.1, CUDA 12.8, cuDNN
9.10), upstream SageAttention @ d1a57a5 built from source for the local GPU
(SageAttention3 with CUTLASS v4.2.0). Our kernels are the repo's sources
compiled with nvcc 12.8 (`flash_mma_fwd2_d128` with build.rs's options,
`attn_fp8.cu`, `attn_sage.cu`) and launched through libcuda on torch's
stream. Every timing is the median of 20 warm calls (CUDA events) of the
**whole call a model would make**: quantization, smoothing, padding and
transposes included. Inputs are bf16 `[1, H, S, 128]`, `Sq = Sk`.

Accuracy is measured against an FP32 reference built from the same
bf16-rounded inputs (TF32 off), as cosine similarity and rel-L2 over the
whole output. Three synthetic regimes are used:
* `normal`: q, k, v ~ N(0, 1);
* `peaked`: the `attn_fp8` gpucheck regime, q x 3 plus a per-(head,
  channel) K offset with std 2;
* `outlier`: 4 of 128 channels of q and k scaled x4 (they vary per token,
  so smoothing cannot remove them), plus the K offset.

The `real` regime is six captures of FastWan2.1 1.3B self-attention Q/K/V
after QK-norm and RoPE (832x480, 81 frames, 12 x 32 760; layers 0 / 15 / 29
at DMD steps 0 and 2; `capture.py`). These captures use a null text context:
the 22 GB UMT5 encoder was not downloaded.

On `normal` data every quantized kernel shows rel-L2 ~0.04. Flat scores
average V almost uniformly, so the output is tiny (~1/sqrt(S)), and any
absolute error is large relative to it. The `real` rows are the meaningful
accuracy numbers.

Tolerance, stated before the run (as for `attn_fp8`): cosine >= 0.998 and
rel-L2 <= 5e-2, here against FP32. The bar for a port: >= 1.3x over our
current dense kernel.

### 3.1 RTX PRO 6000 Blackwell Workstation (sm_120, 600 W): the target

Instance 53606078 (Serbia, $1.34/hr). Times in ms; in brackets, the speedup
over `fv_fwd2`. `sage2_warp32` is INT8 per-warp QK with FP8 PV and fp32+fp32
accumulation. `sage2pp` is upstream `sageattn()` on sm_120: the same, with
fp32+fp16 accumulation (SageAttention2++). `sage2_thr` uses per-thread INT8
scales. `sage2_f16pv` keeps P V in FP16. `sage3` is NVFP4.

| shape | heads x seq | fv_fwd2 | sdpa_cudnn | fv_fp8 | sage2_warp32 | sage2pp | sage2_thr | sage2_f16pv | sage3 |
|---|---|---|---|---|---|---|---|---|---|
| h3_480p | 56 x 15100 | 17.81 | 19.86 (0.90x) | 15.09 (1.18x) | 10.57 (1.69x) | 10.80 (1.65x) | 10.76 (1.65x) | 14.09 (1.26x) | 10.86 (1.64x) |
| h3_768p | 56 x 37966 | 113.85 | 128.45 (0.89x) | 94.57 (1.20x) | 64.37 (1.77x) | 66.10 (1.72x) | 64.79 (1.76x) | 86.38 (1.32x) | 59.14 (1.93x) |
| ltx_480p_s1 | 32 x 1792 | 0.23 | 0.22 (1.04x) | 0.40 (0.57x) | 0.28 (0.82x) | 0.30 (0.76x) | 0.40 (0.57x) | 0.40 (0.57x) | 0.41 (0.57x) |
| ltx_480p_s2 | 32 x 7168 | 2.31 | 2.52 (0.92x) | 2.11 (1.09x) | 1.51 (1.52x) | 1.56 (1.48x) | 1.54 (1.50x) | 1.95 (1.19x) | 1.43 (1.62x) |
| ltx_384p_1stage | 32 x 4032 | 0.81 | 0.82 (0.98x) | 0.89 (0.91x) | 0.60 (1.36x) | 0.62 (1.31x) | 0.74 (1.10x) | 0.83 (0.98x) | 0.79 (1.02x) |
| ltx_720p_s1 | 32 x 3520 | 0.61 | 0.66 (0.93x) | 0.72 (0.85x) | 0.50 (1.22x) | 0.52 (1.16x) | 0.60 (1.01x) | 0.68 (0.90x) | 0.71 (0.86x) |
| ltx_720p_s2 | 32 x 14080 | 8.82 | 9.94 (0.89x) | 7.35 (1.20x) | 5.14 (1.71x) | 5.30 (1.66x) | 5.27 (1.67x) | 6.83 (1.29x) | 4.72 (1.87x) |
| ltx_1080p_s2 | 32 x 32640 | 48.91 | 54.04 (0.91x) | 40.64 (1.20x) | 27.64 (1.77x) | 28.48 (1.72x) | 27.95 (1.75x) | 37.11 (1.32x) | 23.83 (2.05x) |
| wan5b_480p | 24 x 12090 | 5.04 | 5.41 (0.93x) | 4.40 (1.14x) | 3.06 (1.65x) | 3.15 (1.60x) | 3.11 (1.62x) | 4.04 (1.25x) | 3.22 (1.56x) |
| fastwan_480p | 12 x 32760 | 18.68 | 20.67 (0.90x) | 15.59 (1.20x) | 10.55 (1.77x) | 10.87 (1.72x) | 10.61 (1.76x) | 14.18 (1.32x) | 9.67 (1.93x) |

Worst case over all shapes, per input regime (cosine / rel-L2 against FP32):

| kernel | normal: worst cos / worst rel-L2 | peaked: worst cos / worst rel-L2 | outlier: worst cos / worst rel-L2 | real: worst cos / worst rel-L2 |
|---|---|---|---|---|
| fv_fwd2 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 |
| sdpa_cudnn | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 |
| fv_fp8 | 0.9992 / 0.039 | 0.9953 / 0.097 | 0.9922 / 0.125 | 0.9939 / 0.110 |
| sage2_warp32 | 0.9992 / 0.040 | 0.9992 / 0.041 | 0.9990 / 0.044 | 0.9998 / 0.021 |
| sage2pp | 0.9992 / 0.039 | 0.9991 / 0.041 | 0.9990 / 0.044 | 0.9998 / 0.021 |
| sage2_thr | 0.9992 / 0.039 | 0.9992 / 0.039 | 0.9991 / 0.042 | 0.9998 / 0.021 |
| sage2_f16pv | 0.9999 / 0.012 | 0.9996 / 0.027 | 0.9995 / 0.032 | 0.9999 / 0.016 |
| sage3 | 0.9814 / 0.193 | 0.9398 / 0.348 | 0.9386 / 0.354 | 0.9897 / 0.143 |
| sage3_nomean | 0.9813 / 0.193 | 0.9393 / 0.349 | 0.9380 / 0.356 | 0.9804 / 0.197 |

Real FastWan Q/K/V, rel-L2 / cosine against FP32:

| capture | fv_fwd2 | fv_fp8 | sage2_warp32 | sage2pp | sage2_f16pv | sage3 |
|---|---|---|---|---|---|---|
| fastwan_s0_l0 | 0.0018 / 1.00000 | 0.0622 / 0.99807 | 0.0214 / 0.99977 | 0.0215 / 0.99977 | 0.0104 / 0.99995 | 0.1284 / 0.99174 |
| fastwan_s0_l15 | 0.0017 / 1.00000 | 0.0102 / 0.99995 | 0.0056 / 0.99999 | 0.0055 / 0.99999 | 0.0046 / 0.99999 | 0.0381 / 0.99955 |
| fastwan_s0_l29 | 0.0016 / 1.00000 | 0.0602 / 0.99819 | 0.0194 / 0.99981 | 0.0193 / 0.99981 | 0.0165 / 0.99986 | 0.0860 / 0.99637 |
| fastwan_s2_l0 | 0.0017 / 1.00000 | 0.0424 / 0.99910 | 0.0151 / 0.99989 | 0.0152 / 0.99989 | 0.0069 / 0.99998 | 0.0872 / 0.99639 |
| fastwan_s2_l15 | 0.0017 / 1.00000 | 0.0099 / 0.99995 | 0.0053 / 0.99999 | 0.0053 / 0.99999 | 0.0044 / 0.99999 | 0.0380 / 0.99958 |
| fastwan_s2_l29 | 0.0016 / 1.00000 | 0.1103 / 0.99391 | 0.0191 / 0.99982 | 0.0191 / 0.99982 | 0.0134 / 0.99991 | 0.1430 / 0.98973 |

Notes:
* "Our dense" on sm_12x is whichever of cuDNN's SDPA node (cuDNN 9.26) and
  fwd2 is faster per shape. Here torch's cuDNN 9.10 is 8-11 % *slower* than
  fwd2, while our cuDNN 9.26 path is 3-4 % faster at H3 768p (102-105 ms
  vs 108-114 ms, docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md).
  Against our real default, take the fwd2 speedups minus ~4 %: SageAttention2
  is ~1.70x at H3 768p.
* Below ~4k tokens (LTX stage 1 at 480p / 720p) every quantized path loses
  or only ties. The extra quantization passes and launches cost about as much
  as the attention they save.

### 3.2 RTX 5090 (sm_120 GeForce, 400 W cap): for the record

Instance 53603769 (Romania, $0.47/hr). GeForce runs BF16/FP16 MMA with FP32
accumulation at half rate, and FP16-accumulating FP8 MMA (SageAttention2++)
at full rate. Its speedups therefore overstate the RTX PRO 6000's:
SageAttention2++ gets 2.5-2.7x here against 1.7x on the PRO 6000. The
accuracy numbers matched the PRO 6000's to the third digit.

| shape | heads x seq | fv_fwd2 | sdpa_cudnn | fv_fp8 | sage2pp | sage2_thr | sage2_f16pv | sage3 |
|---|---|---|---|---|---|---|---|---|
| h3_480p | 56 x 15100 | 34.32 | 33.16 (1.03x) | 27.56 (1.25x) | 13.21 (2.60x) | 14.69 (2.34x) | 22.38 (1.53x) | 12.76 (2.69x) |
| h3_768p | 56 x 37966 | 215.53 | 209.18 (1.03x) | 171.11 (1.26x) | 78.90 (2.73x) | 86.59 (2.49x) | 136.13 (1.58x) | 69.00 (3.12x) |
| ltx_480p_s1 | 32 x 1792 | 0.44 | 0.42 (1.05x) | 0.71 (0.61x) | 0.40 (1.09x) | 0.54 (0.81x) | 0.60 (0.73x) | 0.62 (0.70x) |
| ltx_480p_s2 | 32 x 7168 | 4.68 | 4.52 (1.04x) | 4.00 (1.17x) | 2.03 (2.31x) | 2.27 (2.06x) | 3.26 (1.44x) | 1.80 (2.60x) |
| ltx_384p_1stage | 32 x 4032 | 1.74 | 1.49 (1.17x) | 1.66 (1.05x) | 0.91 (1.91x) | 1.08 (1.60x) | 1.44 (1.21x) | 1.13 (1.54x) |
| ltx_720p_s1 | 32 x 3520 | 1.37 | 1.22 (1.12x) | 1.44 (0.95x) | 0.76 (1.79x) | 0.94 (1.46x) | 1.18 (1.16x) | 0.95 (1.45x) |
| ltx_720p_s2 | 32 x 14080 | 17.27 | 16.73 (1.03x) | 13.96 (1.24x) | 6.82 (2.53x) | 7.45 (2.32x) | 11.33 (1.52x) | 5.73 (3.01x) |
| ltx_1080p_s2 | 32 x 32640 | 91.00 | 87.62 (1.04x) | 72.46 (1.26x) | 33.89 (2.69x) | 37.22 (2.44x) | 57.79 (1.57x) | 27.68 (3.29x) |
| wan5b_480p | 24 x 12090 | 9.94 | 9.35 (1.06x) | 8.13 (1.22x) | 3.97 (2.51x) | 4.38 (2.27x) | 6.58 (1.51x) | 3.87 (2.57x) |
| fastwan_480p | 12 x 32760 | 36.14 | 34.19 (1.06x) | 28.86 (1.25x) | 13.21 (2.74x) | 14.55 (2.48x) | 22.97 (1.57x) | 11.35 (3.18x) |

### 3.3 Verdict per variant and arch

| variant | sm_120 (RTX PRO 6000 / 5090) | sm_100 (B200) | sm_90 (H100) |
|---|---|---|---|
| **SageAttention2 (INT8 QK, FP8 PV)** | **clears**: 1.52-1.77x over fwd2 at >= 7k tokens (H3 480p/768p, LTX stage 2 at 480p/720p/1080p, Wan 5B, FastWan); worst cosine 0.9990, worst rel-L2 0.044 on synthetic inputs, 0.9998 / 0.021 on real data. Not at LTX stage 1 (<= 4k tokens): 0.7-1.4x | not run: upstream has only its sm_89 `mma.sync` kernels there (no sm_100 path in `sageattn()`), and our tcgen05 dense kernel is already at 1.15-1.21 PFLOPS, above any `mma.sync` rate | does not clear: upstream's wgmma kernel is ~1.1-1.2x over our `fa_dc90` (section 5) |
| SageAttention2 FP16-PV | passes accuracy (rel-L2 <= 0.032), but 1.19-1.32x: under the bar | - | - |
| **SageAttention3 (NVFP4)** | **fails accuracy**: rel-L2 0.19 (normal), 0.33-0.36 (peaked / outlier), 0.04-0.14 on real FastWan Q/K/V, cosine down to 0.94. It is the fastest (1.6-2.2x on the PRO 6000, 2.5-3.3x on the 5090), but far outside the tolerance | does not run: its MMA is the sm_120a/121a warp-level block-scaled instruction | does not exist |
| our `attn_fp8` (FP8 QK, bf16 PV) | reconfirmed: 1.09-1.20x and rel-L2 0.09-0.13 (fails); 0.11 on real data | - | - |

SageAttention3 would need a mixed schedule (upstream suggests Sage2 on the
first and last steps) and an end-to-end quality gate before it could be
considered. At kernel level it is ~25x the bf16 error.

## 4. Phase 2: the port (`wan::attn_sage`, branch `wip/sage-attn`)

`crates/fastvideo-cudarc/src/wan/attn_sage.cu` / `attn_sage.rs`:

* **Contract and template.** The same contract as the dense fwd2 / `attn_fp8`
  kernels: bf16 `[b, h, s, 128]` in, f32 or bf16 out. The fwd2 schedule
  (128-query CTAs, 8 warps x 16 rows, double-buffered K/V stages, one
  barrier per key tile) is kept.
* **QK.** K is smoothed by its per-head mean and quantized to INT8 per
  64-key tile; Q to INT8 per 16 rows (finer than upstream's 32). `S` is
  computed by `mma.sync m16n8k32 s8` (exact int32), dequantized once into
  the log2 domain.
* **PV.** `P` is quantized to E4M3 (fixed scale 448) and V to E4M3 per
  channel. `P V` runs on `m16n8k32 e4m3` with f32 accumulation (sm_120 has a
  full f32 FP8 accumulator, so no two-level buffer is needed).
* **V layout.** V is stored transposed, `[bh, 128, sk_pad]`, with keys
  permuted inside each 16-key group. The softmax's accumulator fragment is
  then directly the A operand of the k32 MMA, and V is loaded with plain
  `ldmatrix`.
* **Footprint.** Shared memory is 32 KB (half of fwd2's). ptxas (sm_120):
  213 registers, no spills.
* **Knob.** `FASTVIDEO_ATTN_SAGE=2` (opt-in, default off) routes DiT
  self-attention through it in `nn::scaled_dot_product_attention` (H3,
  LTX-2.5, Wan; not the VAE). It applies when `Sq, Sk >=
  FASTVIDEO_ATTN_SAGE_MIN_SEQ` (default 6144); everything else keeps the
  bf16 kernels. `=3` is refused with a message.
* **Build.** `build.rs` embeds sm_89+ cubins, with NVRTC as the fallback (as
  for `attn_fp8`).
* **Parity group.** `fv-gpucheck kernels --groups attn_sage` covers flat /
  peaked parity against the bf16 kernel and f64 (the `attn_fp8` cases plus
  an odd 130-row case) and timing at the H3 768p / LTX 720p stage-2 /
  FastWan shapes. First run through the Rust binary in Phase 3: all six
  cases pass (section 6.1).

Measured with the harness on RTX PRO 6000 (instance 53608464; the kernel as
committed, 8 warps):

| shape | heads x seq | fv_fwd2 | fv_sage | sage2_warp32 |
|---|---|---|---|---|
| h3_480p | 56 x 15100 | 17.79 | 11.60 (1.53x) | 10.46 (1.70x) |
| h3_768p | 56 x 37966 | 113.60 | 70.47 (1.61x) | 64.13 (1.77x) |
| ltx_480p_s1 | 32 x 1792 | 0.24 | 0.60 (0.41x) | 0.36 (0.69x) |
| ltx_480p_s2 | 32 x 7168 | 2.32 | 1.86 (1.25x) | 1.52 (1.53x) |
| ltx_384p_1stage | 32 x 4032 | 0.82 | 0.97 (0.85x) | 0.68 (1.20x) |
| ltx_720p_s1 | 32 x 3520 | 0.62 | 0.82 (0.75x) | 0.57 (1.09x) |
| ltx_720p_s2 | 32 x 14080 | 8.87 | 5.68 (1.56x) | 5.11 (1.73x) |
| ltx_1080p_s2 | 32 x 32640 | 48.49 | 30.23 (1.60x) | 27.39 (1.77x) |
| wan5b_480p | 24 x 12090 | 5.00 | 3.51 (1.42x) | 3.03 (1.65x) |
| fastwan_480p | 12 x 32760 | 18.48 | 11.51 (1.61x) | 10.44 (1.77x) |

| kernel | normal: worst cos / worst rel-L2 | peaked: worst cos / worst rel-L2 | outlier: worst cos / worst rel-L2 | real: worst cos / worst rel-L2 |
|---|---|---|---|---|
| fv_fwd2 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 | 1.0000 / 0.002 |
| fv_sage | 0.9992 / 0.039 | 0.9992 / 0.041 | 0.9991 / 0.043 | 0.9998 / 0.021 |
| sage2_warp32 | 0.9992 / 0.040 | 0.9992 / 0.041 | 0.9990 / 0.044 | 0.9998 / 0.021 |

Real FastWan Q/K/V (rel-L2 / cosine against FP32), port vs upstream:

| capture | fv_sage | sage2_warp32 |
|---|---|---|
| fastwan_s0_l0 | 0.0214 / 0.99977 | 0.0214 / 0.99977 |
| fastwan_s0_l15 | 0.0055 / 0.99999 | 0.0056 / 0.99999 |
| fastwan_s0_l29 | 0.0193 / 0.99981 | 0.0194 / 0.99981 |
| fastwan_s2_l0 | 0.0151 / 0.99989 | 0.0151 / 0.99989 |
| fastwan_s2_l15 | 0.0052 / 0.99999 | 0.0053 / 0.99999 |
| fastwan_s2_l29 | 0.0190 / 0.99982 | 0.0191 / 0.99982 |

* **Accuracy:** equal to or slightly better than upstream on every regime,
  thanks to the finer Q scales.
* **Speed:** 1.42-1.62x over fwd2 at >= 12k tokens (H3 768p: 70.5 ms vs
  113.6 ms), about 10 % behind upstream's 64.1 ms.
* **What was tried:** a 32-row-warp variant (`attn_sage_fwd_w32_d128`, each
  K/V fragment feeding two MMAs as upstream does) ran at the same speed. It
  needs 255 registers and spills 100 B, so it stays an unused entry point.
  The next step is an `ncu` profile of the 8-warp kernel.
* **Small shapes:** below ~4k tokens the harness's per-launch Python
  overhead (5 launches) dominates. Those shapes are not routed (min-seq
  6144).

Not ported: the VSA fine-stage variant (H3 turbo's sparse kernel; the
`attn_fp8_vsa` template applies directly) and Sol-H3. Only the dense calls
benefit today: H3 dense steps and layers, LTX-2.5 stage 2, and Wan.

## 5. H100 SXM (sm_90)

Instance 53609815 (Czechia, $3.01/hr). `sage2pp` is upstream `sageattn()`,
which on sm_90 is the wgmma kernel (INT8 QK, FP8 PV, fp32+fp32).
`sage2_warp32` is upstream's sm_89 `mma.sync` kernel run on H100 (0.92-0.98x
of fwd2). `fv_sage` is our port (`mma.sync`). SageAttention3 does not exist
for sm_90.

| shape | heads x seq | fv_fwd2 | sdpa_cudnn | sdpa_flash | fv_sage | sage2pp | sage2_f16pv |
|---|---|---|---|---|---|---|---|
| h3_480p | 56 x 15100 | 21.18 | 12.41 (1.71x) | 19.57 (1.08x) | 21.00 (1.01x) | 9.83 (2.15x) | 15.90 (1.33x) |
| h3_768p | 56 x 37966 | 135.04 | 76.28 (1.77x) | 127.45 (1.06x) | 129.26 (1.04x) | 57.83 (2.34x) | 97.68 (1.38x) |
| ltx_480p_s1 | 32 x 1792 | 0.25 | 0.13 (1.91x) | 0.21 (1.20x) | 0.43 (0.60x) | 0.50 (0.51x) | 0.42 (0.60x) |
| ltx_480p_s2 | 32 x 7168 | 2.69 | 1.37 (1.97x) | 2.44 (1.11x) | 3.02 (0.89x) | 1.49 (1.81x) | 2.22 (1.21x) |
| ltx_384p_1stage | 32 x 4032 | 0.91 | 0.51 (1.79x) | 0.80 (1.13x) | 1.14 (0.80x) | 0.66 (1.38x) | 0.98 (0.93x) |
| ltx_720p_s1 | 32 x 3520 | 0.71 | 0.40 (1.79x) | 0.63 (1.13x) | 0.94 (0.76x) | 0.55 (1.29x) | 0.66 (1.08x) |
| ltx_720p_s2 | 32 x 14080 | 10.50 | 5.52 (1.90x) | 10.18 (1.03x) | 10.43 (1.01x) | 4.86 (2.16x) | 8.03 (1.31x) |
| ltx_1080p_s2 | 32 x 32640 | 57.38 | 29.44 (1.95x) | 54.35 (1.06x) | 54.52 (1.05x) | 25.00 (2.30x) | 42.10 (1.36x) |
| wan5b_480p | 24 x 12090 | 6.11 | 3.26 (1.87x) | 5.18 (1.18x) | 6.12 (1.00x) | 2.92 (2.09x) | 4.45 (1.37x) |
| fastwan_480p | 12 x 32760 | 21.84 | 12.25 (1.78x) | 20.30 (1.08x) | 21.20 (1.03x) | 9.30 (2.35x) | 17.35 (1.26x) |

Accuracy matches the sm_120 runs (worst cosine 0.9991, rel-L2 0.044
synthetic / 0.022 real).

**Verdict: SageAttention2 does not clear the bar on H100.** On sm_90 our
dense default is not fwd2 but `fa_dc90_fwd_d128` (wgmma + TMA,
FA3-style), measured at 0.61-0.65 PFLOPS
(docs/perf/datacenter-profile.md), i.e. ~64-68 ms at H3 768p. Upstream's sm_90
kernel at 57.8 ms (714 TOPS) is therefore about 1.1-1.2x over our kernel,
which is under 1.3x. It is 1.32x over torch's cuDNN 9.10 (our proxy here,
76.3 ms, 541 TFLOPS). Our `mma.sync` port is no faster than fwd2 on H100: a
wgmma port would be a separate kernel for a ~15 % gain. Not pursued.

## 6. Phase 3: end to end (Runpod RTX PRO 6000, 2026-10-01)

**Setup.** One RTX PRO 6000 Blackwell Server Edition pod (`9lnoscqp7y8v5k`,
EUR-IS-1, $2.09/hr, driver 595.91.07) on the EU weight volume `jg48s6o1w0`,
mounted read-only; everything written went to the container disk. The
binary is `fv-gpucheck` built on the shared build pod from
`wip/phase3-pro6000` (main + `wip/sage-attn` + `wip/ltx-director-res`,
`f6dcd6d`), uploaded to a pod running the `wip/sage-attn` runtime image
(`fastvideo-rs-runtime:sha-4eeb801`, the same pinned CUDA 13.4 libraries).
The branch's own CI image did not build: rustup inside the Docker build
fails to update `stable` to 1.99.0 ("Invalid cross-device link"), which
will hit main's next image build too. The pod was driven through
`scripts/serve/e2e/pod.sh` (sidecar exec). Driver, per-row summary and
frame montages: `artifacts/perf/sage-phase3/`.

Every A/B is two processes on the same pod, identical except for
`FASTVIDEO_ATTN_SAGE=2`; each arm has its own text and AdaLN caches, so
nothing is shared between arms. Quality is `fv-gpucheck compare-clips` (LPIPS
alex on 44 frames, PSNR, sharpness and temporal-jitter ratios) per prompt,
then `fv-gpucheck gate` with `scripts/gpu/gate-policy.toml` (`lossy`: hard
limits sharpness 0.95-1.08, jitter 0.85-1.20, frame count, LPIPS
available; promotion speedup >= 1.10 on denoise).

### 6.1 The Rust parity group (`kernels --groups attn_sage,attn_fp8,attn3_bench`)

First run through the Rust binary. `attn_sage` passes all six parity cases
against the bf16 kernel (rel-L2 0.035-0.039, cosine 0.9992-0.9994, peaked
S = 4097 included). The bench cases compare against `auto`, i.e. our real
dense default (cuDNN or fwd2, per shape):

| shape | heads x seq | dense `auto` ms | `attn_sage` ms | speedup | harness (Vast, vs fwd2) |
|---|---|---|---|---|---|
| H3 768p | 56 x 37 966 | 106.99 (386 TFLOPS) | 71.43 (579 TFLOPS) | 1.50x | 70.47 vs 113.60 (1.61x) |
| LTX 720p stage 2 | 32 x 14 080 | 8.75 | 6.00 | 1.46x | 5.68 vs 8.87 (1.56x) |
| FastWan 480p | 12 x 32 760 | 16.54 (cuDNN) | 11.77 | 1.41x | 11.51 vs 18.48 (1.61x) |

The port runs at the harness's speed; the lower ratio is the stronger
baseline: at the H3 shape `auto` picks cuDNN 9.26, about 6 % faster than
fwd2 there (106-110 ms against 112-117 ms in the H3 runs' logs). `attn_fp8`
reconfirms its known result: 5 peaked cases fail (rel-L2 0.067-0.086), 1.19x
dense.

### 6.2 H3 five-prompt gate (768p, `scripts/gpu/prompts-eval.json`)

Medians over the five prompts, warm process; denoise and total (text +
denoise + decode) per clip. LPIPS / PSNR / sharpness / jitter are the range
over the five prompts.

| recipe | dense video calls | bf16 denoise / total s | Sage denoise / total s | speedup denoise / total | LPIPS | PSNR dB | sharpness | jitter | gate |
|---|---|---|---|---|---|---|---|---|---|
| H3 max (`sol-h3`, `h3/sol_h3_4step_engine_ladder`) | 56 of 200 | 23.62 / 31.30 | 20.94 / 28.60 | 1.13x / 1.09x | 0.31-0.49 | 14.3-21.6 | 0.932-1.052 | 0.93-1.10 | **fail**: frogyoga sharpness 0.932 |
| Sol-H3 4-step dense (`h3/sol_h3_4step`) | 200 of 200 | 35.79 / 43.39 | 27.01 / 34.61 | **1.33x / 1.25x** | 0.31-0.51 | 13.6-20.8 | 0.891-1.031 | 0.78-1.06 | **fail**: frogyoga sharpness 0.891, jitter 0.780 |
| h3-turbo (`4step-vsa`) | 0 of 200 (all VSA) | 20.57 / 28.33 | 20.54 / 28.25 | 1.00x | 0 (byte-identical) | - | 1.000 | 1.000 | fail: no speedup |
| **control**: dense, bf16 with the dense kernel switched cuDNN -> fwd2 (`FASTVIDEO_CUDNN_SDPA_GRAPH=composite`) | 200 of 200 | 35.79 / 43.39 | 36.58 / 44.08 | 0.98x | 0.27-0.49 | 14.0-21.6 | 0.936-1.046 | 0.93-1.06 | fail: spark-mountain-lake sharpness 0.936 |

* **The gate is at H3's noise floor.** Switching only the bf16 dense kernel
  (cuDNN and fwd2 differ by rounding) moves every clip as far as Sage
  does: LPIPS 0.27-0.49 against Sage's 0.31-0.51, PSNR 14-22 dB in both.
  The control fails the same sharpness floor, on a different prompt. The
  montage `shots/h3dense-frogyoga-f060-bf16-sage-ctl.jpg` shows three
  different, equally clean compositions of the same prompt. Across
  processes the bf16 arms are deterministic (h3-turbo, where Sage never
  routes, is byte-identical), so the divergence comes from the changed
  kernel, and H3 amplifies any such change.
* **Sage is a little further out on one prompt.** frogyoga fails under both
  Sage arms (sharpness 0.932 and 0.891, jitter 0.780 dense) and sits at
  0.952 under the control. That is one prompt of five, so it is suggestive
  but not conclusive.
* **Lip sync (proxy, `scripts/gpu/lipsync_proxy.py`, pooled over h3-demo
  and ltx-newsbroadcast): not informative.** h3-demo never shows a stable
  face (29 of 124 face frames). On the one usable clip the bf16 arms are not
  in sync either (best lag +250 ms / +42 ms, |r| <= 0.3, negative speech
  contrast), so no arm can be told apart.
* **The `auto` choice is per process.** For the text encoder's d = 64 shape
  (224 x 1 797), one process picked cuDNN and the other fwd2 (0.69 vs 0.65-0.70
  ms). That flip did not change any output here, but it means `auto` is
  not reproducible across processes by construction.

**Verdict: fail for H3 under the current policy** (both recipes, one
prompt each). Sage stays off for H3. A fair re-test needs a gate that
measures against the bf16 spread: several bf16 control arms per prompt,
then Sage judged against their distribution. More prompts would also
settle whether frogyoga is a real Sage effect.

### 6.3 LTX-2.5 (two-stage, 121 frames, 3 prompts: ltx-multishot, -newsbroadcast, -frogyoga)

`--no-text-cache` in every arm (cache hits and misses differ on LTX,
docs/techniques.md). Stage 1 runs at half size: at 720p it is 3 520 tokens,
below the 6 144 routing floor, while at 1080p it is 8 160 tokens and is routed.

| cell | bf16 denoise s (stage 1 / 2) | Sage denoise s (stage 1 / 2) | speedup | LPIPS | PSNR dB | sharpness | gate |
|---|---|---|---|---|---|---|---|
| ltx-turbo 720p (1280x768, Sol stage 2) | 8.25 (3.16 / 4.32) | 8.20 (3.16 / 4.31) | 1.01x | 0.029-0.036 | 34.8-36.4 | 0.997-1.005 | fail: speed |
| ltx-turbo 1080p (1920x1088, Sol stage 2) | 18.92 (6.88 / 9.54) | 18.44 (6.54 / 9.48) | 1.03x | 0.20-0.22 | 22.3-24.4 | 0.986-1.006 | fail: speed |
| dense stage 2 1080p (`--dense-stage2`, the ltx-pro route) | 22.75 (6.96 / 14.99) | 19.84 (6.57 / 12.50) | **1.15x** (stage 2 1.20x) | 0.20-0.23 | 20.9-24.4 | 0.986-0.993 | **pass** |

The serving turbo route runs Sol on stage-2 layers 1-47, so Sage only
reaches layer 0 there (and stage 1 at 1080p). LTX is far less chaotic than
H3: at 720p Sage stays at 35 dB from bf16, so the 0.20 LPIPS at 1080p is
mostly the routed stage 1 moving the clip. Sage passes the gate on the
dense route (one LPIPS 0.86 outlier frame at a scene cut in multishot; the
mean is 0.23).

### 6.4 What it buys per request

| workload | denoise saved | per 5 s clip |
|---|---|---|
| Sol-H3 4-step dense 768p | 8.8 s of 35.8 | 43.4 -> 34.6 s total |
| H3 max 768p (serving) | 2.7 s of 23.6 | 31.3 -> 28.6 s |
| LTX dense stage 2 1080p | 2.9 s of 22.8 | |
| h3-turbo, ltx-turbo (serving) | none (sparse attention already) | |

### 6.5 Not covered

The VSA fine stage and Sol kernels have no Sage variant (`attn_fp8_vsa`
is the template), so the sparse serving recipes gain nothing yet. LTX
director tiers and the 384p upscale rows from the same session are in
docs/serve/e2e/ltx.md ("Director tiers", "384p stage 1 and upscale rows").

### 6.6 Decision

The port stays on `wip/sage-attn` / `wip/phase3-pro6000`, not on main. It is
opt-in (`FASTVIDEO_ATTN_SAGE=2`, default off), so merging it would change no
output; the merge condition was a passing gate, and H3's did not pass.
Options for the owner: (a) merge it default-off and enable it on the
ltx-pro dense route, the one recipe that passes (1.15x denoise at 1080p);
(b) re-run H3 with a control-calibrated gate first.

### 6.7 Spend (Runpod)

| item | time | cost |
|---|---|---|
| Build pod `dap2h1xyxyyy10` (cpu, EU-RO-1, $1.12/hr): merge build, unit tests, release `fv-gpucheck` / `fv-serve`; stopped | 24 min | ~$0.45 |
| GPU pod `9lnoscqp7y8v5k` (RTX PRO 6000, EUR-IS-1): items a-e; deleted, GET 404 | 106 min (6 357 s) | ~$3.69 |
| Build pod `9hndck5b8pgl4y` (cpu3c, $0.96/hr): tests of the main-bound tree; stopped | 14 min | ~$0.22 |
| **total** | | **~$4.36** |

The pod had a detached 3.5 h DELETE backstop, an on-pod idle guard (20 min
at 0 % GPU) and a local balance watchdog (delete below $9). The balance
went from $78.97 to $72.79 over the session; other agents' pods ran
at the same time.

## 7. Spend and instances (Vast)

| instance | GPU | what | wall | cost |
|---|---|---|---|---|
| 53603186 | RTX 5090 | first batch; setup bug (no `git` in the image), destroyed after the baseline rows | 4 min | ~$0.03 |
| 53603769 | RTX 5090 (400 W) | Phase 1 full run | 23 min | ~$0.18 |
| 53606078 | RTX PRO 6000 WS | Phase 1 full run | 10 min | ~$0.23 |
| 53607047 | RTX PRO 6000 WS | port v1 vs upstream | 10 min | ~$0.23 |
| 53608464 | RTX PRO 6000 WS | port v2 + w32 variant | 7 min | ~$0.16 |
| 53609815 | H100 SXM | sm_90 check | 14 min | ~$0.70 |
| | | **total** (Vast credit $33.27 -> $31.77, incl. storage and rounding) | | **$1.50** |

Every instance was labelled `fv-sage-*`, had a detached wall-clock backstop
(a DELETE after 1-2 h) and a container-side timeout, and was destroyed as
soon as its batch finished. Each destroy was checked: the instance list is
empty.
