# FastH3 V2 / 4-step NVFP4: apples to apples with FastVideo on one RTX PRO 6000

Owner request 2026-10-14: compare against FastVideo's published RTX PRO 6000 numbers for FastH3
(`docs/inference/fasth3_rtx_pro_6000.md` in FastVideo @ `d5253fd`) on our own hardware, both checkpoints downloaded
with approval. Nothing here changes a default: the NVFP4 path is taken only for a `transformer/` in ModelOpt's layout,
and no catalog entry or preset was added.

## 1. The checkpoints (what they actually are)

| | V2-NVFP4 | 4-step VSA DataFree NVFP4 |
|---|---|---|
| Hub revision | `0aa5247ba4a5d3f9f4fdb99536d8cf55bdc5e2f0` | `d7adcabeb19a81bb79cf9c4a04f5e366f1f2e90a` |
| contract (`fastvideo_inference.json`) | 8 DMD rungs, 9 grid points, shifts 10 / 3, VSA 0.8, tile 64 | 4 rungs, 5 points, shifts 12 / 3 (h3-base schedulers), VSA 0.9, tile 64 |
| R2 tree | `fasth3-v2-nvfp4` (21 files, 55.19 GB) | `fasth3-4step-vsa-nvfp4` (26 files, 55.19 GB) |

- **Only the FFN is NVFP4.** The ModelOpt `quantization_config` quantizes `ff.net.0.proj` / `ff.net.2` of all 50
  blocks (W4A4: U8 E2M1 codes low nibble first, row-major F8_E4M3 per-16 scales, F32 `weight_scale_2`, and a
  calibrated static `input_scale`). Attention (`to_q/k/v/out`, the VSA `to_gate_compress`), the full-rank AdaLN
  projections, the refiner and the patch / output layers are bf16. Hence 53.5 GB of transformer, against 70.1 GB
  for V2 bf16.
- "Every linear NVFP4" (`layer_profile h3_dit_vsa`) is not in the checkpoint. FastVideo's
  `convert_minimax_h3_modelopt_nvfp4_dit.py --quantize-attention --quantize-gate` quantizes the bf16 attention and
  gate weights itself, with one global scale per projection (`448·6/amax`). Its activations then use a unit global
  scale. The FFN keeps ModelOpt's calibrated activation scale (`_nvfp4_input_global_sf = 1/input_scale`; the page's
  "drops the input_scale" note is out of date against the script at `d5253fd`).
- **The vae/ is H3's 36-block VAE with NVFP4 decoder-transformer linears** (shards 1-2 differ, shard 3 equals
  h3-base's). FastVideo `d5253fd`'s H3 VAE has no ModelOpt path, and neither does ours, so both sides here decode
  with the official VAE. The page's timings used a third-party 26-block "light INT8" VAE. Its VAE seconds are
  therefore not comparable to ours.
- Byte-identical to `h3-base` @ `42ed227` / `h3-8step` @ `3da2ddf` (Hub LFS SHA-256 / git blob), so not
  re-published: `text_encoder/`, `tokenizer/`, `processor/`, `audio_vae/`, vae shard 3, the V2 root JSONs and
  schedulers, and the 4-step's schedulers. docs/ops/r2-weights.md §15 has the record.
- The checkpoint's own README reports 11.92 dB PSNR / LPIPS 0.518 against same-seed bf16 references (five
  1344x768 samples, calibration-distribution prompts): FP4 trajectory drift, with no equivalence claim.

## 2. Why VSA "failed parity" before, and why it is used here

docs/oracle.md (2026-09-26) ran FastH3 8-step VSA 0.8 against FastVideo with injected noise and text. Step-1 latents
matched to 3.4e-3 (the dense control: 2.2e-3), and blocks 0-24 to 5e-3 to 9e-3. The difference then grows about 15%
per block from block ~25, to 0.25-0.33 at block 48, and the final latents part by 0.5. The same growth appears for
dense attention, and for our bf16 against our f32 run (0.48). So it is bf16 rounding amplified by depth, plus VSA
top-k tile flips triggered by that noise, not a VSA defect. The "failed" verdicts in docs/techniques.md
"H3 attention arms" are the lossy gate's sharpness / jitter windows on *raised* sparsities (0.85-0.95) against the
trained 0.8 / 0.9. They are not parity failures. At the trained sparsity our VSA is the reference function at the
model's noise floor, so the comparison here runs our VSA-H3 at 0.8 / 0.9, tile 64, with the exempt prefix and the
trained compression gate, matching FastVideo's mask semantics (tile-64 top-k over 64x64 tiles).

## 3. Our load (`crates/fastvideo-cudarc/src/h3/nvfp4.rs`)

| Where | What |
|---|---|
| `is_modelopt_dir` | `transformer/config.json` `quantization_config` = `modelopt` / `NVFP4`; any other kind keeps its old path |
| `PublishedLinear::read_modelopt` | codes as stored (already cuBLASLt's low-nibble-first), row-major scales moved into the 128x4 tiled layout, `weight_scale_2`, optional `input_scale` |
| `Nvfp4Dit::open_modelopt` | header checks of every block's FFN and bf16 attention; nothing read |
| `modelopt_ffn` | `ff_in` / `ff_out` on cuBLASLt `VEC16_UE4M3` (#105's operand and GEMM), the static activation scale by default (`FASTVIDEO_H3_NVFP4_ACT=dynamic` measures it) |
| `modelopt_attention` | `FASTVIDEO_H3_NVFP4_ATTN=1`: `to_q/k/v` (+ gate) and `to_out` quantized at load (TE `static_6`, one global scale per projection, as FastVideo's converter), activation amax measured per call. Unset: attention follows `FASTVIDEO_H3_QUANT` (MXFP8 blocks 2-46 on sm_120) |
| `wan/nvfp4_linear.rs` | `quantize_bf16_operand_with` (a fixed amax: no amax pass; past it the E4M3 scale clamps at 448 and the codes at ±6), `Nvfp4Stack::with_static_act_scale` |
| `h3/pipeline.rs` | the detection; no prequant fast-boot tree, resident only, no adapter |

No fp32 `delta_s`-style buffer exists on our path: VSA works on tile lists, and the NVFP4 activation operand is
`rows·k/2` codes plus `rows·k/16` scales per call.

Tests: `cargo test -p fastvideo-cudarc --features cuda --lib -- h3::nvfp4 wan::nvfp4` (11 passed; new: the ModelOpt
layout reads as our TE quantization of the same weight, with identical blocked scales; config detection). On the GPU:
`kernels --groups nvfp4_linear` `nvfp4_static_act_scale_matches_dynamic`.

## 4. Results

(filled from the pod run below)
