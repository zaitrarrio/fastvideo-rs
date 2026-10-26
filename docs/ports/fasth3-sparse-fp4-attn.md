# FastH3 NVFP4: sparse FP4 VSA attention (the NVFP4 tier's default)

Owner-approved work order of 2026-10-14, following docs/ports/fasth3-nvfp4.md §8. The VSA-H3 fine stage on
block-scaled FP4 tensor cores, for the ModelOpt NVFP4 FastH3 checkpoints only.
**Default for the NVFP4 tier** (owner decision 2026-10-14, after the §3 results: upstream's speed, a different
sample than bf16 VSA): with `FASTVIDEO_H3_VSA_FP4` unset, an NVFP4 DiT on sm_120 takes it; `FASTVIDEO_H3_VSA_FP4=0`
keeps bf16 VSA. Every other DiT is unchanged.

| | |
|---|---|
| flag | `FASTVIDEO_H3_VSA_FP4`: unset = on for NVFP4 where available, `0` = bf16 VSA, `1` = insist (`FASTVIDEO_H3_VSA_FP4_P=single`: upstream's single-level P quantization) |
| taken when | the DiT is a ModelOpt NVFP4 `transformer/` (`h3::nvfp4::is_modelopt_dir`) **and** the device is sm_120 with the kernel loaded; otherwise bf16 VSA (logged). An explicit `=1` on any other DiT logs that it is ignored; on an NVFP4 DiT without the kernel it refuses to load |
| code | `crates/fastvideo-cudarc/src/wan/attn_fp4.{cu,rs}` (kernels, host twins), `h3/vsa.rs` (`H3Vsa::with_fp4`), `h3/pipeline.rs` (`vsa_fp4_choice`), `build.rs` (an sm_120a cubin) |
| checks | `fv-gpucheck kernels --groups h3_fp4_attn` (`crates/fastvideo-gpucheck/src/kernels_fp4.rs`) |
| GPU runs | `scripts/gpu/experiments/h3-fp4attn-validate.sh` + `h3-fp4attn-pod.sh` (deleted in D1e, docs/adr/0019-fv-lab.md §14; in git at `22a5fd7b`) |

## 1. The reference: what upstream actually does

Read from FastVideo `d5253fd` (the commit `ghcr.io/zaitrarrio/fastvideo-rs-upstream-fastvideo-nvfp4:sha-056aa71`
is built from, `scripts/gpu/upstream/setup.sh` `FV_NVFP4_REV`):
`fastvideo/models/dits/minimax_h3_vsa_fp4.py` (`vsa_fp4_attention`),
`fastvideo-kernel/attn_qat_infer/api.py` (`vsa_tile_mask_to_fp4_blocks`, `sageattn_blackwell_sparse_bshd`),
`attn_qat_infer/quantization/fp4_quantization_4d.cu` and `attn_qat_infer/blackwell/{mainloop_tma_ws,softmax_fused}.h`.

**Where it sits.** `FASTVIDEO_H3_VSA_FP4=1` keeps VSA-H3's own selection: tile pooling, the f32 pooled scores,
`_build_block_mask` (every prefix tile forced, top-`k_vid` video tiles, prefix query tiles dense), and the gated
compression branch (`softmax(pooled scores) @ pooled V`, times `to_gate_compress`). Only the block-sparse attention
itself runs on SageAttention3's FP4 kernel, for **every** query row, prefix rows included.

**Quantization granularity** (every scale is `amax / 6` rounded to E4M3 with RN + saturation, every element
`x / scale` rounded to E2M1 with RN + saturation, inputs bf16):

| operand | group of 16 | layout in the kernel | smoothing / global scale |
|---|---|---|---|
| Q | 16 consecutive head dims of a query row | row-major, `[B, H, L, D/2]` codes + `[.., D/16]` scales | none: `preprocess_qkv(per_block_mean)` defaults leave Q and K unsmoothed, and `delta_s` is a cached all-zero row read with `per_block_mean=False` |
| K | 16 consecutive head dims of a key row | rows permuted within each 32 (`scaled_fp4_quant_permute`) so the S accumulator is P's A fragment | none |
| V | 16 consecutive keys of one head dim | transposed, `[B, H, D, L/2]` | none |
| P | 16 consecutive keys of one query row, per KV block, inside the online softmax | registers | single-level by default (`single_level_p_quant=True`): `s_P = max(P~)/6` with `P~ = exp2(s - m_running)`; the row sum is the unquantized P~ |

Two details of P upstream: the codes are `P~ / s_P` with the **unrounded** f32 `s_P` while the MMA multiplies by
its E4M3 rounding (a per-group bias of up to 1/16), and single-level scales underflow E4M3 (min 2^-9) for any 16-key
group whose largest probability is below ~0.6 % of the running row max, which then contributes nothing to `P V`
while still counting in the row sum.

**Block / quadrant mask.** The kernel tiles 128 x 128 (BLOCK_M = BLOCK_N = 128). `vsa_tile_mask_to_fp4_blocks`
pairs VSA's 64-token tiles: block `(m, n)` is listed for query block `m` when any of its four 64 x 64 quadrants is
selected, and `q2k_quad` carries the 4 bits (bit `2 * row_half + col_half`). Masked quadrants score `-inf`; a warp
(16 rows, so inside one 64-row half) skips a block its half did not select and the `P V` chunk of a key half it did
not select; `kv_valid` holds each 64-key half's live count (VSA tile tails). Lists run in descending block order so
block 0 (the first text tile, always selected) is visited first. **A selected 64 x 64 quadrant is exactly one
(query tile, key tile) pair of VSA-H3's mask**, so the attended set equals the tile mask, token for token.

**The 9.5 GB `delta_s`.** `preprocess_qkv` used to build `delta_s = qm @ K^T` as `[B, H, L/128, KL]` f32 even with
Q smoothing off (zeros); at 73k tokens that was 9.5 GB, and its int32 batch stride overflowed the TMA descriptor.
Upstream now reads one cached zero row (`_zero_delta_s`, `per_block_mean=False`).

## 2. Our design

**Selection unchanged.** Our VSA-H3 already produces, per (head, query tile), the list of key tiles
(`vsa_topk_device` over pooled f32 scores with the prefix bias: every prefix tile, then the top `k_vid` video
tiles). The FP4 path consumes exactly that list; the compression branch, the bf16 combine
(`bf16(bf16(sparse) + bf16(bf16(coarse) * gate))`) and the selection are the bf16 path's code.

**Granularity: upstream's, element for element.** `attn_fp4_quant_tile` quantizes q / k / v straight from the
packed `[bh, seq, 128]` activations (f32 or bf16; f32 is rounded to bf16 first, as upstream's tensors are bf16) into
tile-slot order, with the scale rule above. Because VSA tiles are 64-aligned slots, our 16-key V groups and 16-key
P groups are the same keys as upstream's (pad slots are zeros on both sides). The device quantizer is checked bit
for bit against the host twin (`attn_fp4::quant16`).

**Mapping onto our tile lists (no 128-block pairing).** One CTA per 64-row query tile (4 warps x 16 rows) walks its
own list of 64-key tiles. That is the quadrant mask without the pairing: no quadrant is loaded for nothing, no warp
idles on a block its half did not select, and `topk` is uniform per tile so there are no ragged lists. Tile tails
(`block_sizes`) are masked to `-inf` like `kv_valid`. Prefix query tiles keep the existing path (bf16 SDPA against
all keys, then their compression term), as in the bf16 VSA path; upstream runs those few hundred rows through FP4
too. This is the one place our numerics are deliberately more exact than upstream's.

**P quantization: two-level by default.** `P' = P~ * 448 * 6` before the per-16 scale (upstream's own optional
two-level mode, `single_level_p_quant=False`), so a group's scale underflows only below ~4e-6 of the row max; the
factor is divided out with the row sum at the end, and the codes divide by the **rounded** scale. Upstream's
single-level rule is `FASTVIDEO_H3_VSA_FP4_P=single`; both are measured.

**The sm_120 instruction.**

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
    {d0..d3}, {a0..a3}, {b0, b1}, {c0..c3}, {sfa}, {0, 0}, {sfb}, {0, 0}
```

E2M1 operands packed 8 per register (low nibble first), f32 accumulate, one UE4M3 scale per 16 k (`scale_vec::4X`:
a 32-bit register holds the four scales of one row of A / column of B). Fragments (lane `l`, `g = l >> 2`,
`t = l & 3`): `a0` row `g`, k `8t..8t+7`; `a1` row `g + 8`; `a2` / `a3` the same at k + 32; `b0` column `g`, k
`8t..8t+7`, `b1` k + 32; `sfa` the scales of row `g + 8 (l & 1)`; `sfb` those of column `g`; `d` the usual
`D[g][2t, 2t+1]`, `D[g + 8][2t, 2t+1]`. The `mma_mxf4nvf4_layout` check runs one instruction on random fragments
against this statement (unit scales first, then random block scales).

Per 64-key tile and warp: `S = Q K^T` is 8 n-tiles x 2 k-chunks = 16 MMAs (Q's A fragments and scales live in
registers for the whole CTA), `O += P V` is 16 n-tiles x 1 k-chunk = 16 MMAs. K tiles are stored with rows permuted
within each 32 (`8n + 2t + j <-> 8t + 2n + j`, an involution, upstream's trick) so the S accumulator a thread holds
is P's A fragment (keys `8t..8t+7` and `32+8t..32+8t+7` of rows `g` and `g + 8`) with no shuffle; the P scales need
one `shfl.xor 1` (the other half of each 16-key group) and one `shfl.xor 2` (the row's other two groups). Shared
memory: Q 4.5 KB + three (K, V) stages of 9 KB (K codes 4 KB, K scales 512 B, V^T codes 4 KB, V scales 512 B),
`cp.async` double-buffered with one barrier per tile; ldmatrix rows swizzled (eight distinct 16-byte bank groups
per 8x8 matrix). 164 registers, no spills, 3 CTAs per SM.

**No fp32 delta buffers, nothing `[queries, keys]`-shaped.** There is no Q smoothing, so no `delta_s` at all. Per
call the FP4 operands are `bh * padded * (64 + 8)` bytes for q and for k and the same for V^T: about 0.6 GB at 768p
(56 heads, ~40k padded rows), against the bf16 path's three f32/bf16 tiled copies. The output is the fine stage's
usual f32 `[bh, padded, 128]`, consumed by the existing combine.

## 3. Results (one RTX PRO 6000 Blackwell Server Edition, EUR-IS-2, driver 580.159 / CUDA 13.0, image `fastvideo-rs-runtime:sha-b564268`, 2026-10-14)

### 3.1 Kernel checks (`kernels --groups h3_fp4_attn`, all PASS, 37 s)

| check | result |
|---|---|
| `mma_mxf4nvf4_layout`: one mma on random fragments vs the host statement of §2 | exact (0 wrong of 128, unit scales and random block scales) |
| `fp4_quantizer_bitwise`: q / k / v, f32 and bf16 inputs, vs the host twin of upstream's rule | 0 rows / columns differ; f32 and bf16 inputs give identical bytes |
| fine stage vs host emulation of the same FP4 algorithm (limit 2e-2) | 1.1e-6 (sparse), 5.2e-6 (full list) |
| **sparse 0.8, rel-L2 vs token-masked f64** (limit 0.19) | **0.160** two-level P, 0.165 single-level P (upstream's rule) |
| **full block list vs dense f64** (limit 0.19) | **0.183** two-level, 0.189 single-level |
| our bf16 VSA kernel vs the same references, for scale | 4.1e-4 / 4.7e-4 |

Problem: 2 heads, 4460 rows (6 prefix + 64 video tiles), q/k unit-RMS with a shared smooth positional component and
outlier channels. The emulation agrees with the kernel to f32 rounding, so the remaining 0.16-0.18 is the FP4
quantization itself, at upstream's measured floor (0.19); two-level P is slightly better than single-level, as
expected. There is no separate dense FP4 kernel: the full-list case is the dense FP4 attention.

**Fine-stage timing** (56 heads, random lists of the real length at sparsity 0.8, 5 s clips):

| canvas | rows | tiles / top-k | bf16 fine stage (tile gather + `vsa_mma_attn`) | FP4 quantize + fine | speedup | FP4 kernel |
|---|---:|---:|---:|---:|---:|---:|
| 832x480 | 15 100 | 291 / 67 | 9.75 ms | 1.16 + 3.01 = **4.17 ms** | 2.34x | 732 TFLOPS |
| 1344x768 | 37 966 | 671 / 143 | 40.2 ms | 3.06 + 13.2 = **16.3 ms** | 2.47x | 839 TFLOPS |

### 3.2 The 768p trace (phase 1) and the FP4 path in the model

CUPTI trace of one warm V2-NVFP4 step (`FASTVIDEO_GPU_TRACE=1`, NVFP4 attention + FFN as #110, 124 frames), per
block (step busy time / 50). Files: `artifacts/ports/fasth3-sparse-fp4-attn/run-2026-10-14/t{480,768}-{bf16,fp4}.json`.

| per block | 480p bf16 VSA | 480p FP4 | 768p bf16 VSA | 768p FP4 |
|---|---:|---:|---:|---:|
| **step busy / 50** | **33.6** | **27.9** | **98.3** | **74.1** |
| attention (all kernels) | 13.1 (40 %) | 7.35 | 48.6 (51 %) | 24.3 |
| - fine stage kernel (`vsa_mma_attn_tma2` / `attn_fp4_vsa`) | 7.16 | 2.66 | 34.1 | 12.6 |
| - tiling (`vsa_tile_qkv` x3 / `attn_fp4_quant_tile`) | 2.45 | 1.19 | 5.79 | 2.98 |
| - prefix rows SDPA, pooling, combine, top-k (unchanged) | 3.5 | 3.5 | 8.7 | 8.7 |
| NVFP4 GEMMs | 8.87 | 8.85 | 21.3 | 21.4 |
| norm / casts / layout (unchanged) | 11.6 | 11.7 | 28.4 | 28.4 |

- **Phase 1 answer: attention is the dominant gap at 768p.** It is 51 % of the step (40 % at 480p) and grows 3.7x
  for 2.5x the rows (k_vid grows with the video tiles), while everything else grows ~2.4x. Upstream's 768p step is
  ~73 ms per block (3.66 s / forward / 50): the whole 25 ms gap was attention.
- **GEMM algorithm at 768p rows.** cuBLASLt's heuristic changes with the row count. At 15.1k rows, 6 of the 7
  NVFP4 GEMMs per block run `cutlass3x_sm120 … 128x128x256` and 1 runs `nvjet … 256x128x128 vec16`. At 38k rows,
  2 run cutlass (5.97 ms each) and 5 run `nvjet … 128x256x128 vec16` (1.83 ms each). GEMM time grows 2.40x for 2.51x
  the rows, so the 768p choice is not a regression. Upstream's FlashInfer cutlass was not traced at 768p (its page
  has no such table), so this compares our choice against itself, not against upstream.
- With FP4 the fine stage is 2.7x faster in the model (7.16 -> 2.66 ms, 34.1 -> 12.6 ms); our 480p attention is
  now 7.35 ms per block against upstream's 7.9 ms (sparse FP4, its page).

### 3.3 Denoise timings (warm, 124 frames, seed 1024, one process per arm)

| checkpoint | canvas | prompt | bf16 VSA s | **FP4 VSA s** | speedup | upstream s (#110) | FP4 / upstream |
|---|---|---|---:|---:|---:|---:|---:|
| V2 (8 fwd) | 832x480 | h3-demo | 14.06 | **11.76** | 1.20x | 10.25 | 1.15x |
| V2 | 832x480 | spark-mountain-lake | 13.20 | **11.16** | 1.18x | 9.81 | 1.14x |
| V2 | 1344x768 | h3-demo | 39.99 | **30.29** | 1.32x | 29.28 | 1.03x |
| V2 | 1344x768 | spark-mountain-lake | 39.04 | **29.55** | 1.32x | 28.41 | 1.04x |

**Update (2026-10-14, docs/perf/h3-denoise-graphs.md):** the step loop now times steps with lazy CUDA events instead of
a synchronize per step, byte-identical. On the FP4 tier this gives V2 h3-demo 11.67 -> **11.14 s** at 480p and
29.96 -> **29.50 s** at 768p (upstream 10.25 / 29.28 s). Opt-in CUDA graphs of the steps (`FASTVIDEO_H3_GRAPHS=1`)
take off another 0.3-2% on a replayed request shape.

Median step 1.757 -> 1.472 s (480p) and 4.998 -> 3.789 s (768p). Peak device memory is unchanged or lower (480p
26.5 -> 26.0 GiB, 768p 34.2 -> 32.4 GiB on h3-demo). VAE decode is unchanged (2.1 / 4.8 s).

**4-step checkpoint: per-prompt numbers lost.** Its cells ran (both arms rc 0; process wall time with load, the warm-up
and four generations: 127 s bf16 VSA, 94 s FP4). The results archive was not written: `tar` exits 1 when `nvsmi.csv`
grows during packing, and the pod was deleted before this was noticed. The pack step and the launcher's fetch are
fixed. Expect the 4-step to follow the V2 ratios per forward.

### 3.4 Quality: FP4 attention vs #110's bf16 VSA NVFP4 arm (same checkpoint, seed, text and noise)

V2, 832x480x124, h3-demo, seed 1024; per-step latents dumped by both arms, `compare-dumps`, then `compare-clips`.

| | FP4 VSA vs bf16 VSA (this run) | for scale: ours vs upstream (#110) |
|---|---:|---:|
| step-1 block 0 / 24 / 48 output | 8.2e-3 / 9.1e-2 / 0.83 | 7.5e-2 / 0.13 / 0.84 |
| step-1 velocity | 0.51 | 0.47 |
| **video latents after step 1** / 2 / 4 | **1.1e-2** / 1.9e-2 / 5.1e-2 | 1.04e-2 / 1.8e-2 / 4.9e-2 |
| final video latents (step 8) | 0.65 | 0.61 |
| clip PSNR / LPIPS (alex, all frames) | 13.5 dB / 0.488 | 12.6 dB / 0.465 |
| sharpness ratio / jitter ratio (mean) | 0.916 / 0.838 | 0.938 / 1.097 |
| lossy gate hard checks (sharpness 0.95-1.08, jitter 0.85-1.20) | **FAIL** (both) | FAIL (sharpness) |

**Verdict: quality fails the lossy gate against our bf16-VSA NVFP4 arm.** The attention kernel itself sits at the
FP4 noise floor (0.16 rel-L2 sparse, §3.1). Through the first step the two arms agree to 1.1e-2, the same distance
#110 measured between our NVFP4 path and upstream's own FP4 output. The few-step sampler then amplifies it into a
different sample of the same scene, as it does between any two FP4 implementations. The FP4 arm is somewhat
softer (sharpness 0.92) and has less frame-to-frame change (jitter 0.84), each just outside the gate's window. One
prompt and one seed were measured; the five-prompt gate was not run within the budget. It stays an opt-in speed
tier for the NVFP4 checkpoints, which already fail the gate against bf16 and against upstream (docs/ports/fasth3-nvfp4.md
§6).

### 3.5 Cost

One RTX PRO 6000 pod (EUR-IS-2, $2.49/hr): 15:21:44-15:42:16, about $0.85. Every step of the session ran:
fetch of 185 GB from R2 at ~800 MB/s (234 s), kernel checks, four traces, parity and the timing cells. An earlier pod
was deleted 30 s after creation (~$0.02). Both were deleted and checked gone (GET 404); their backstops were stopped
by exact PID.
