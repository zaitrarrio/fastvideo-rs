# SageAttention2 / SageAttention3 against our attention kernels

Status: Phase 1 (kernel microbenchmark), 2026-09-30. Branch `wip/sage-attn`.
Harness: `scripts/gpu/sage/` (`bench.py`, `capture.py`, `vast-run.sh`).

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

## 3. Phase 1 results

(pending)
