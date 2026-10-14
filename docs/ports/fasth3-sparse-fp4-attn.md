# FastH3 NVFP4: sparse FP4 VSA attention (opt-in tier)

Owner-approved work order of 2026-10-14, following docs/ports/fasth3-nvfp4.md §8. The VSA-H3 fine stage on
block-scaled FP4 tensor cores, behind `FASTVIDEO_H3_VSA_FP4=1`, for the ModelOpt NVFP4 FastH3 checkpoints only.
**Never a default**: no preset, recipe, catalog entry or profile turns it on.

| | |
|---|---|
| flag | `FASTVIDEO_H3_VSA_FP4=1` (`FASTVIDEO_H3_VSA_FP4_P=single`: upstream's single-level P quantization) |
| taken when | the DiT is a ModelOpt NVFP4 `transformer/` (`h3::nvfp4::is_modelopt_dir`) **and** the device is sm_120 with the kernel loaded; any other DiT logs that the flag is ignored; an NVFP4 DiT on another device refuses it |
| code | `crates/fastvideo-cudarc/src/wan/attn_fp4.{cu,rs}` (kernels, host twins), `h3/vsa.rs` (`H3Vsa::with_fp4`), `h3/pipeline.rs` (`vsa_fp4_choice`), `build.rs` (an sm_120a cubin) |
| checks | `fv-gpucheck kernels --groups h3_fp4_attn` (`crates/fastvideo-gpucheck/src/kernels_fp4.rs`) |
| GPU runs | `scripts/gpu/h3-fp4attn-validate.sh` + `h3-fp4attn-pod.sh` |

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

## 3. Results

(filled in below by the GPU runs)
