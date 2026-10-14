# H3 DiT block: glue fusions (NVFP4, MXFP8, bf16)

Owner request 2026-10-14: fuse the non-GEMM, non-attention "glue" of the H3 DiT block, starting from the V2-NVFP4
block profile of docs/ports/fasth3-nvfp4.md §7, without changing quality. `FASTVIDEO_H3_GLUE` (default on, needs
`FASTVIDEO_H3_FUSE`) gates every change below; `=0` runs the ops they replace. No attention kernel changed.

## 1. What changed

| commit | glue it removes | where it applies | numerics |
|---|---|---|---|
| VSA fed bf16 in place | q/k/v/gate written widened to f32 for VSA, VSA's own f32 tiling / pooling / combine reads, and the dense prefix rows' f32 -> bf16 casts of the full k and v | every H3 VSA block with bf16 activations (h3-turbo MXFP8, the NVFP4 checkpoints, bf16) | bit-identical (VSA's bf16-in-place kernels, `vsa_*_b16`, are the f32 path's bits) |
| q/k/v/gate heads in one kernel | `h3_qk_norm_rope` x2 + `fvf_split_heads_rows` x2 | the same, and dense / Sol-layer attention | bit-identical (same reduction tree, same expressions) |
| SwiGLU writes the NVFP4 operand | `mx_swiglu` (bf16 `[S, ffn]` written) + `nvfp4_quant_bf16_sw` (read back) | published NVFP4 `ff_out` with a static activation scale (the ModelOpt checkpoints) | bit-identical operand bytes |
| head merge writes the NVFP4 operand | `cast_f32_bf16` + `mx_merge_heads` + `nvfp4_amax_bf16` + `nvfp4_quant_bf16_sw` | published NVFP4 `to_out` (`FASTVIDEO_H3_NVFP4_ATTN=1`) | bit-identical operand bytes and amax |

Kernels (`wan/kernels.cu`, DiT block fusions region): `fvf_h3_qkvg_heads`, `fvf_h3_swiglu_nvfp4`,
`fvf_amax_f32_r16`, `fvf_h3_merge_heads_nvfp4`, and the device helpers `fv_nvfp4_store_lane` (one NVFP4 block over
16 lanes: shuffle max, then `fv_nvfp4_scale_block`'s scale and E2M1 math, codes stored as 32-bit words, the scale
into cuBLASLt's `VEC16_UE4M3` tiled layout) and `fv_nvfp4_pad_row`.

## 2. The run (2026-10-14)

One RTX PRO 6000 Blackwell Server Edition (EU-CZ-1, driver 595.71.05, CUDA 13.2), runtime image `sha-8f20822`,
two `fv-gpucheck` builds on the same pod: **base** (the branch point, `94edde2`, = main after #110) and **new**
(this branch). `scripts/gpu/h3-glue-validate.sh up BASE NEW` + `scripts/gpu/h3-glue-pod.sh`; weights from R2 (no
EU-RO-1 stock: h3-base, its pre-quantized 4step-vsa MXFP8 DiT, the turbo LoRA, FastH3 V2 NVFP4's `transformer/`).
h3-demo prompt, seed 1024, 124 frames (5 s), bf16 text encoder streamed once per prompt (text cache). One process
per (model, binary), base and new back to back, each warm (`--warm`: a cold 480p generation first). V2-NVFP4 is
`FASTVIDEO_H3_NVFP4_ATTN=1` (NVFP4 attention + FFN, the arm #110 profiled). Results:
`artifacts/perf/h3-glue-fusion/run-2026-10-14/` (`summary.md`, the benchmark JSONs with the CUPTI traces,
`kernels.log`, `parity.txt`).

### Step time (warm, median step)

| model | canvas | base step s | new step s | step win | base denoise s | new denoise s |
|---|---|---:|---:|---:|---:|---:|
| V2-NVFP4 (8 fwd) | 832x480 | 1.711 | 1.431 | **-16.4%** | 13.70 | 11.44 |
| V2-NVFP4 | 1344x768 | 4.982 | 4.360 | **-12.5%** | 39.87 | 34.90 |
| h3-turbo, MXFP8 (4 fwd) | 832x480 | 1.913 | 1.753 | **-8.4%** | 7.66 | 7.02 |
| h3-turbo | 1344x768 | 5.060 | 4.658 | **-8.0%** | 20.24 | 18.63 |

Against FastVideo's sparse-FP4 path (docs/ports/fasth3-nvfp4.md §5: 1.28 / 3.66 s per forward), V2-NVFP4 goes from
1.34x / 1.36x behind to 1.12x / 1.19x (same hardware type, not the same pod).

### Per kernel, per block, 480p (one traced step / 50 blocks, ms)

V2-NVFP4 (block busy **33.81 -> 27.72 ms**, 2890 -> 2390 kernels per step):

| kernel | base | new | |
|---|---:|---:|---|
| `vsa_tile_qkv` -> `vsa_tile_qkv_b16` (x3) | 2.45 | 1.01 | VSA fed bf16 |
| `vsa_tile_mean` -> `vsa_tile_mean_b16` (x3) | 0.96 | 0.52 | VSA fed bf16 |
| `vsa_combine` -> `vsa_combine_g16` | 0.90 | 0.76 | VSA fed bf16 |
| `cast_f32_bf16` | 1.77 | 0.03 | VSA fed bf16 (prefix SDPA k/v casts), merge -> NVFP4 (VSA output cast) |
| `h3_qk_norm_rope` (x2) + `fvf_split_heads_rows` (x2) -> `fvf_h3_qkvg_heads` | 2.53 | 1.21 | heads in one kernel |
| `mx_swiglu` -> `fvf_h3_swiglu_nvfp4` | 1.40 | 1.27 | SwiGLU -> NVFP4 (and the next row) |
| `nvfp4_quant_bf16_sw` (4 -> 2 calls) | 1.74 | 0.61 | ff_out and to_out operands now written by their producers |
| `mx_merge_heads` -> `fvf_h3_merge_heads_nvfp4` | 0.79 | 0.67 | merge -> NVFP4 |
| `nvfp4_amax_bf16` (2 -> 1) + `fvf_amax_f32_r16` | 0.30 | 0.45 | merge -> NVFP4 (amax of the f32 VSA output) |
| `block_copy` -> `mx_block_copy` | 0.11 | 0.10 | |
| unchanged: `h3_res_gate_norm_mod` 0.48, `fvf_h3_gate_res_norm_mod` 0.46, `fill_f` 0.60, `mx_binary` 0.10 | | | |
| NVFP4 GEMMs (cutlass + nvjet) | 8.92 | 9.06 | not touched |
| `vsa_mma_attn_tma2` + `flash_mma_fwd_d128` | 8.72 | 8.78 | not touched |

By category: norm / modulation / QK-norm / SwiGLU 4.08 -> 1.08 (+ 1.21 heads), casts / quantization 3.85 -> 3.08,
layout 2.49 -> 0.78, attention (its glue kernels only) 13.17 -> 11.21.

h3-turbo (block busy **38.30 -> 34.36 ms**): the same VSA rows (`vsa_tile_qkv` 2.45 -> 1.01, `vsa_tile_mean` 0.96 ->
0.52, `vsa_combine` 0.90 -> 0.76), `cast_f32_bf16` 1.26 -> 0.09, `h3_qk_norm_rope` 1.59 + `fvf_split_heads_rows`
0.93 -> `fvf_h3_qkvg_heads` 1.18. Its MXFP8 producers (`h3_swiglu_mx`, `fvf_merge_heads_mx`) were already fused.

### Combined with sparse FP4 VSA (main `cf28c95`+, #113)

Once #113 made sparse FP4 VSA the default for the NVFP4 tier, the V2-NVFP4 cells were re-run with
`FASTVIDEO_H3_VSA_FP4` unset (on), `FASTVIDEO_H3_NVFP4_ATTN=1`:
- one RTX PRO 6000 Server Edition (EUR-IS-2, driver 580.173.02, CUDA 13.0), runtime image `sha-8f20822`;
- **base** = main `e873146`; **new** = this branch's head with main merged; **glue off** = the new build with
  `FASTVIDEO_H3_GLUE=0`;
- run with `FV_ONLY_NVFP4=1 scripts/gpu/h3-glue-validate.sh up BASE NEW`.

Results are in `artifacts/perf/h3-glue-fusion/run-2026-10-14-fp4vsa/`.

| V2-NVFP4, FP4 VSA on | canvas | base step s | glue off step s | new step s | step win |
|---|---|---:|---:|---:|---:|
| 8 fwd | 832x480 | 1.479 | 1.481 | 1.223 | **-17.3%** |
| 8 fwd | 1344x768 | 3.784 | 3.798 | 3.188 | **-15.7%** |

Denoise times:
- 480p: 11.81 s (base), 11.82 s (glue off), 9.74 s (new).
- 768p: 30.26 s (base), 30.36 s (glue off), 25.49 s (new).

Per block at 480p, block busy time went from **28.08 to 22.63 ms**, and kernels per step from 2790 to 2290:
- The glue rows are the same as in the bf16-VSA run.
- FP4 VSA's own quantizer, `attn_fp4_quant_tile`, now reads bf16 instead of f32: 1.18 to 0.66 ms.
- `attn_fp4_vsa` is unchanged: 2.71 to 2.70 ms.
- The GEMMs are unchanged: 9.00 ms in both.

Against FastVideo's sparse-FP4 path on its page (1.28 / 3.66 s per forward), this is 0.96x / 0.87x. The hardware
type is the same, but the pods differ.

**Parity:** the clips (frames, mp4, audio; 126 files each) are byte-identical between all three arms, at 480p and at
768p: base and new, glue off and new, base and glue off (`parity.txt`). The 33 `h3_glue_*` kernel checks pass again
on this pod.

## 3. Parity

- **Kernels**: `fv-gpucheck kernels --groups dit_fusion,nvfp4_linear` on the pod, all pass; the 33 new
  `h3_glue_*` checks are bit-exact (0 bf16 ulps; NVFP4 codes, scales and amax byte-equal), and every existing
  `dit_fusion` / `nvfp4_linear` check passes.
- **End to end**: every clip the new binary wrote (frames, mp4, audio; h3-turbo and V2-NVFP4, 480p and 768p) is
  **byte-identical** to the base binary's (`parity.txt`, `sha-*.txt`: 126 files each).
- Unit tests: `cargo test -p fastvideo-cudarc --features cuda --lib -- h3:: wan::nvfp4` (108 passed).

## 4. What is left, and where we stopped

Glue left in the V2-NVFP4 block at 480p (27.72 ms): `nvfp4_quant_bf16_sw` 0.61 (the qkvg and ff_in operands),
`fill_f` 0.60 (inside VSA: out of scope, attention is another work item), `h3_res_gate_norm_mod` 0.48 and
`fvf_h3_gate_res_norm_mod` 0.46 (already fused), `fvf_amax_f32_r16` 0.32, `nvfp4_amax_bf16` 0.13. The next
candidate, writing ff_in's NVFP4 operand from `h3_res_gate_norm_mod` (static scale), would remove about half of
0.61 ms plus a bf16 write, ~1% of a step: under the 2% bar, so not done. The qkvg operand needs its measured
amax first (dynamic scale), so it cannot be written on the fly.

h3-turbo: the norm + modulate that feeds the QKV (whose bf16 VSA gate rows keep it from taking an MXFP8 input)
is followed by `mxfp8_quantize`, **0.22 ms per block, 0.6% of the block**: a norm that writes both the bf16 rows and
the MXFP8 QKV input is under the 2% bar and was not done.

## 5. Cost

Three GPU pods, each deleted and checked gone (`GET /pods/<id>`: 404). Each staged bundle was removed, and each local
backstop was stopped by its PID after its command line was checked.

| pod | what it ran | time | cost |
|---|---|---|---:|
| 9f1ey4e9whjqvc | the main run | 10.5 min | $0.44 |
| djye3h0zc5lak1 | combined run, first try: the V2 cells started before h3-base's encoder and VAE had arrived (r2 mode); fixed in the pod script | 5 min | $0.21 |
| 7jb4gcsj93pqb9 | the combined run | 10 min | $0.42 |

Total: **$1.07**, at $2.49/hr.
