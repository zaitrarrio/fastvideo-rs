# sm_120 dense attention vs cuDNN, and our VSA vs FastVideo's (2026-09-27)

Decision: `FVID-2026-09-27-attention-sm120` in `decision-log.md`. GPU: RTX PRO
6000 Blackwell Server Edition (sm_120, driver 595.91.07), runtime image with
cuDNN 9.26.0.51. Kernel numbers: `fv-gpucheck kernels` groups `attn3_parity`,
`attn3_bench`, `vsa_stages` (median of three synchronized calls after a
warm-up); FastVideo numbers: `scripts/gpu/upstream/bench_vsa.py` (median of
five CUDA-event-timed calls) on the plain PyTorch image (torch 2.13 cu130)
with the pinned `fastvideo-kernel==0.3.5` wheel.

## 1. cuDNN fused SDPA from Rust

cudarc 0.17.8 binds cuDNN's backend API but not the descriptors cuDNN added
for attention after 9.13. `wan/cudnn_sdpa.rs` now declares them itself
(`cudnn_sdpa::raw`, values from cuDNN 9.26 `cudnn_graph_v9.h`):
`CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR` (41) with
`CUDNN_ATTR_OPERATION_SDPA_FWD_{Q,K,V,O,STATS,SCALE}DESC` (2800-2805), and
`CUDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR` (45) with
`CUDNN_ATTR_OPERATION_SOFTMAX_{X,Y}DESC` (3100-3101). Every backend call goes
through cudarc's loaded `libcudnn` with enum arguments widened to `u32` and
statuses read as `i32`; failures carry `cudnnGetLastErrorString`.

Three graphs (`FASTVIDEO_CUDNN_SDPA_GRAPH=auto|unified|softmax|composite`):

| graph | what it is (cudnn-frontend equivalent) | cuDNN 9.26 on sm_120 |
|---|---|---|
| unified | one `OPERATION_SDPA_FWD` node: Q, K (literal, not K^T), V, O, by-value scale (`UnifiedSDPANode`, what `AttentionImplementation_t::AUTO` picks for bf16 inference) | heuristic mode A: engines 11 and 8; **all parity checks pass** (vs f32 reference rel_l2 <= 4.3e-3, vs fwd2 bf16 max_abs 1-2 bf16 ulp) |
| softmax | bmm, scale, one `OPERATION_SOFTMAX` node, bmm (`CompositeSDPANode` + `UnifiedSoftmaxNode`) | builds a plan on engine 1, but its output is garbage (inf / NaN, rel_l2 ~1e38 on every shape): **rejected** |
| composite | bmm, scale, max/sub/exp/sum/div, bmm (pre-9.21) | no engine ("non-flash composite MHA fprop is no longer supported") |

## 2. Dense timings (bf16 in/out, d=128, ms)

| shape | fwd2 (old default) | fwd3 | fwd3s | cuDNN unified cfg0 (engine 11) | cfg1 (engine 8) | cuDNN via torch (earlier) |
|---|---|---|---|---|---|---|
| H3 768p, 56 x 37 710 | 108.4 / 109.1 | 112.2 | 111.5 | **105.0 / 104.8** | 112.0 | 103.0 |
| LTX 768x512, 32 x 6 144 | 2.00 / 1.98 | 2.06 | 2.06 | 1.99 / 1.97 | 1.98 | - |
| LTX 1080p 20 s, 32 x 124 440 | 682.2 / 682.7 | 700.4 | 697.0 | **658.8 / 659.8** | 703.8 | 647.0 |
| LTX 4K 5 s, 32 x 130 560 | **759.9 / 757.2** | 778.2 | 776.0 | 769.0 / 766.6 | 763.7 / 759.8 | 704.0 |

Two pods (the second number is the second pod). cuDNN's heuristic first
config is 3-4% faster than fwd2 at H3 768p and LTX 1080p 20 s and ~1% slower
at 4K 5 s, where torch's cuDNN SDPA (a different cuDNN build) is faster still.

`flash_mma_fwd3` (S_{j+1} issued before softmax(S_j) so each warp has
independent HMMAs during its softmax, three-stage K/V ring, branch-free tail
masking) is bit-identical to fwd2 on every parity shape and real shape but
3% slower: at 255 registers (24 B spilled) ptxas keeps the softmax chain
mostly ahead of the QK HMMAs rather than interleaving them, and the extra
live S tile costs more than the overlap buys. `fwd3s` (skip the O rescale
when no row max rose; within 1 bf16 ulp of fwd2) recovers only ~0.5%. Both
stay opt-in (`FASTVIDEO_FLASH_KERNEL=v3|v3s`), not defaults.

Default on sm_12x (`attn.rs` `cudnn_default`, `auto_cudnn_or_v2`): for each
bf16-output shape, the first call times cuDNN's plan (after a warm-up
execution) against fwd2 on the real inputs and keeps the faster, so H3 768p
and LTX 1080p take cuDNN and a shape where fwd2 wins keeps fwd2.
`FASTVIDEO_FLASH_KERNEL=v2` / `=cudnn` fix the kernel.

## 3. VSA: ours vs FastVideo's `video_sparse_attn` (ms per call)

Tile (4, 4, 4), d=128, bf16. FastVideo times its op on tile-ordered padded
tensors (its attention backend's raster->tile gather of q/k/v/gate and the
tile->raster gather of the output are separate columns); ours times the Wan
f32 path (`vsa_attention_device`, whose tiling is inside `tile+fine`).

| workload | stage | FastVideo (Triton) | ours (before) |
|---|---|---|---|
| FastH3 768p, 56 h, 660 tiles, k=66 (0.9) | coarse | 1.89 | 2.73 |
| | top-k | **0.24** | 1.80 |
| | tile q/k/v(/gate) | 9.32 (4 tensors) | 5.39 (3 tensors, f32->bf16) |
| | fine | 22.65 (226 TFLOPS) | **15.79** (324 TFLOPS) |
| | combine (+ untile) | 2.44 + 1.94 | 2.28 (scatter included) |
| | op total (tile/untile excluded) | 27.81 | 28.33 (tile included) |
| FastH3 768p, k=132 (0.8) | fine | 43.92 | **31.01** |
| | op total | 48.40 | 43.81 |
| FastH3 480p, 56 h, 280 tiles, k=28 (0.9) | coarse / top-k / fine / combine | 0.81 / 0.11 / 5.09 / 1.04 | 1.05 / 0.66 / 2.92 / 0.91 |
| | op total | **6.42** | 7.89 |
| FastH3 480p, k=56 (0.8) | op total | **10.30** | 10.57 |
| FastWan 1.3B, 12 h, 624 tiles, k=125 (0.8) | coarse / top-k / fine / combine | 0.42 / 0.06 / 9.51 / 0.50 | 0.58 / 0.46 / 5.96 / 0.63 |
| | op total | 10.36 | **8.53** |

Our fine stage (`vsa_mma_attn_tma2`) is 1.4-1.7x faster everywhere. We lost
on top-k (4-8x slower: 32 bisection passes with block reductions, then one
thread writing the list), on coarse (f32 tile means + f32 GEMMs), and, in H3,
on widening bf16 q/k/v/gate to f32 before VSA (5.6 ms at 768p) and reading
f32 in tiling / combine. At 480p those made the whole op 23% slower.

Fixes (bit-exact):

* `vsa_topk2`: 8-bit radix select over the row held in shared memory (4
  histogram passes) + block-wide ordered compaction; the same indices in the
  same order as `vsa_topk` (checked on real and heavily tied scores). Default
  for rows up to 4096; `FASTVIDEO_VSA_TOPK=v1` restores the old kernel.
* H3 reads its bf16 activations in place (`vsa_tile_mean_b16`,
  `vsa_tile_qkv_b16`, `vsa_combine_g16`, `vsa_mma_attn_tiled_device`):
  no widening; tile means x3 1.17 ms, tiling x3 2.38 ms (was 5.37), combine
  1.87 ms (was 2.23) at 768p, all bit-identical to the f32 kernels.
  `FASTVIDEO_VSA_BF16=0` restores the widening path.
