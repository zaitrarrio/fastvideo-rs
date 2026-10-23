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

## 4. The run (2026-10-14)

One RTX PRO 6000 Blackwell Server Edition (EU-NL-1, driver 595.91, CUDA 13.2), image
`ghcr.io/zaitrarrio/fastvideo-rs-upstream-fastvideo-nvfp4:sha-056aa71` (FastVideo `d5253fd`, fastvideo-kernel built for
sm_120a with the SageAttention3 FP4 kernels against CUTLASS `e67e63c`), and this branch's `fv-gpucheck` (`3d8e52b`)
with the runtime's CUDA 13.4 libraries. Both sides ran on the same pod:
`scripts/gpu/experiments/fasth3-nvfp4-validate.sh up <image> <bin>` and `scripts/gpu/experiments/fasth3-nvfp4-pod.sh`. Weights came from R2
(185 GB at ~180 MB/s from EU-NL-1 to the WNAM bucket, 17 min). Results are in `artifacts/ports/fasth3-nvfp4/run-2026-10-14/`
(`summary.md` from `scripts/gpu/experiments/fasth3-nvfp4-summary.py`). Every step passed (`status.tsv`).

**Upstream, exactly as its page runs it:**
- `convert_minimax_h3_text_encoder_nvfp4.py` (50 layers, probe 0.134, 170 s) and
  `convert_minimax_h3_modelopt_nvfp4_dit.py --quantize-attention --quantize-gate`: 350 linears, worst probe 0.140 (V2)
  and 0.139 (4-step), against the page's 0.14.
- `layer_profile h3_dit_vsa`, `VIDEO_SPARSE_ATTN_H3`, VSA 0.8 / 0.9 at tile 64, `FASTVIDEO_H3_VSA_FP4=1`,
  `FASTVIDEO_MINIMAX_H3_FUSIONS=all`, `FASTVIDEO_NVFP4_MM_BACKEND=cutlass`, `FASTVIDEO_H3_VAE_TILE_BATCH=28`.
- Every component resident, the VAE compiled, `num_inference_steps` 9 (V2) / 5 (4-step), 124 frames.
- Driven through `scripts/gpu/upstream/bench_fasth3_nvfp4.py`, which uses the page's own `VideoGenerator.from_config`
  settings.
- One process per checkpoint: 480p h3-demo cold, then warm, spark-mountain-lake, then the same at 768p, seed 1024.

**Ours:**
- Each NVFP4 tree's `transformer/` as published, the official VAE, the bf16 text encoder streamed once per prompt,
  then cached.
- VSA-H3 0.8 / 0.9 at tile 64, from the checkpoints' `fastvideo_inference.json`.
- One process per arm, with both canvases and both prompts, `--warm`: the first (cold) generation is 480p h3-demo.

**"8 steps".** FastVideo refuses `num_inference_steps` 8 for V2 ("8 DMD forwards require 9 grid points"): 9 grid
points *are* 8 forwards, so the page's 9 is the only valid setting, and every number below is 8 forwards (V2) / 4
forwards (4-step). s / forward = denoise / forwards upstream, and our median step.

**An upstream bug at 768p.** FastVideo's compiled VAE asserts (`torch._inductor.cudagraph_trees`
`assert torch._C._is_key_in_tls`) when a second shape is compiled in its forward thread. Every 768p request in the
480p-first process therefore failed *after* denoising. Their denoise times are the stage log's `denoising_stage`
lines; upstream 768p VAE times are missing. The page measured 768p as the first shape of its process.

## 5. Results

Warm, 124 frames (5 s), seed 1024. "Peak" is device memory from nvidia-smi over the request. Upstream holds every
component resident (the NVFP4 text encoder, the DiT and the compiled VAE). Ours streams the bf16 text encoder, so
its resident set is the DiT (11.1 GiB), the VAE and the refiner. The two peak columns are therefore not like for like.

| checkpoint | canvas | prompt | upstream denoise s | upstream s / fwd | ours NVFP4 denoise s | ours s / fwd | ours / upstream | ours FFN-NVFP4 + MXFP8 attn s | VAE up / ours s | peak up / ours GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 (8 fwd) | 832x480 | h3-demo | 10.25 | 1.28 | 13.84 | 1.73 | 1.35x | 15.18 | 2.85 / 2.88 | 70.2 / 24.4 |
| V2 | 832x480 | spark-mountain-lake | 9.81 | 1.23 | 13.17 | 1.64 | 1.34x | 14.21 | 2.99 / 2.95 | 70.2 / 23.8 |
| V2 | 1344x768 | h3-demo | 29.28 | 3.66 | 39.90 | 4.99 | 1.36x | 43.47 | — / 6.59 | 78.5 / 31.4 |
| V2 | 1344x768 | spark-mountain-lake | 28.41 | 3.55 | 38.77 | 4.85 | 1.36x | 41.99 | — / 6.60 | 78.5 / 31.3 |
| 4-step (4 fwd) | 832x480 | h3-demo | 4.78 | 1.20 | 6.42 | 1.60 | 1.34x | 7.01 | 2.89 / 2.90 | 70.2 / 24.0 |
| 4-step | 832x480 | spark-mountain-lake | 4.59 | 1.15 | 6.01 | 1.50 | 1.31x | 6.58 | 2.81 / 2.90 | 70.2 / 23.8 |
| 4-step | 1344x768 | h3-demo | 12.74 | 3.18 | 16.86 | 4.21 | 1.32x | 18.42 | — / 6.58 | 78.5 / 31.4 |
| 4-step | 1344x768 | spark-mountain-lake | 12.30 | 3.08 | 16.27 | 4.07 | 1.32x | 17.71 | — / 6.59 | 78.5 / 31.2 |

**Cold** (the first request of a process):
- **Upstream** V2 480p: 13.69 s denoise (the page: 12.5 s), then 35.5 s of VAE decode (its compile). 4-step: 6.84 s,
  and 8.8 s of decode, the inductor cache now warm. At 768p cold equals warm (29.30 s, 12.77 s).
- **Ours** V2 480p: 13.94 s denoise and 3.0 s of VAE decode. 4-step: 6.46 s. There is no compile, so cold is within 1%
  of warm.
- Load: upstream 34-37 s, ours 37 s (the DiT from local disk, AdaLN table computed).

**Against the page's claim.** Their 480p V2 cold denoise of 12.5 s reproduces at 13.69 s on our pod. Their 768p
cells are 243 frames (10 s) and were not rerun: the request was 5 s clips. **Ours is 1.31-1.36x behind upstream's
sparse-FP4 path at both canvases and on both checkpoints**, and level on the VAE where both decode the official VAE
(480p).

**How close the other routes get** (denoise, V2, 480p / 768p, against upstream's 10.0 / 28.8 s):

| route | 480p | 768p | vs upstream |
|---|---:|---:|---:|
| ours, NVFP4 attention + FFN, VSA 0.8 (this branch) | 13.5 | 39.3 | 1.35x / 1.36x |
| ours, NVFP4 FFN + MXFP8 attention, VSA 0.8 (this branch) | 14.7 | 42.7 | 1.47x / 1.48x |
| ours, served V2 dense MXFP8 (`fasth3-8step-dense`, docs/serve/e2e/h3-test-round-2026-10-14.md, another RTX PRO 6000) | 17.1-18.0 | 67.7-68.9 | 1.75x / 2.37x |

Dense bf16 V2 was not rerun. The Trim's bf16 is 1.30x slower than its MXFP8 (docs/ports/h3-trim.md §5.4), which puts
dense bf16 V2 at roughly 2.3x / 3x. The dense control (`FV_NO_DENSE=0`) was dropped from this run to fit the pod on a
smaller disk after two hosts failed to pull the image.

## 6. Parity against upstream's own output

Ours (NVFP4 attention + FFN, VSA) ran on the reference's noise and text (`FASTVIDEO_INJECT_DIR` from the oracle
dump of upstream's first request), 832x480x124, h3-demo, seed 1024. Rel-L2 of ours against upstream
(`compare-dumps-{v2,4s}.txt`); bf16 against bf16 (docs/oracle.md) for scale:

| | V2 NVFP4 | 4-step NVFP4 | bf16 vs bf16 (V2, 768p) |
|---|---:|---:|---:|
| injected noise, sigmas | 0 | 0 | 0 |
| refined text (upstream's NVFP4 encoder output injected) | 8.1e-3 | 8.2e-3 | 8.2e-3 |
| step-1 block 0 / 12 / 24 / 48 | 7.5e-2 / 4.8e-2 / 0.13 / 0.84 | 8.1e-2 / 5.8e-2 / 0.13 / 0.91 | 4.8e-3 / 5.3e-3 / 9.0e-3 / 0.33 |
| step-1 velocity | 0.47 | 0.46 | 0.17 |
| **video latents after step 1** / 2 / 4 | **1.04e-2** / 1.8e-2 / 4.9e-2 | **2.0e-2** / 4.5e-2 / 0.65 (final) | 3.4e-3 / 1.0e-2 / 3.6e-2 |
| final video latents | 0.61 (step 8) | 0.65 (step 4) | 0.53 |
| clip PSNR video / audio | 12.6 / 22.8 dB | 14.3 / 20.0 dB | (Trim 16.6 dB) |
| LPIPS (alex, all frames) | 0.465 | 0.528 | |
| sharpness / jitter ratio | 0.938 / 1.097 | 1.047 / 1.226 | |
| **lossy gate, hard quality checks** (sharpness 0.95-1.08, jitter 0.85-1.20) | **FAIL** (sharpness) | **FAIL** (jitter) | |

Reading:
- From block 0 on, the two FP4 implementations differ by 5-8e-2. That is ten times the bf16 pair's gap, and it is the
  FP4 noise floor of two different quantizations:
  - upstream runs FP4 Q·K and P·V (SageAttention3, 0.19 rel-L2 per call against fp32 on its own test) with a unit
    activation scale on the attention projections;
  - ours runs bf16 VSA attention with measured activation scales.
- Step-1 latents agree to 1.0e-2 (V2) and 2.0e-2 (4-step). The few-step sampler then amplifies the gap, as it does
  between two bf16 implementations, to a different sample of the same scene: 12.6 / 14.3 dB.
- The checkpoint's own README measures 11.9 dB for its NVFP4 output against its bf16 reference.

**Verdict:** parity at the FP4 floor through the first step; the decoded clips are different samples. Against its
own upstream output each checkpoint fails one hard check of our lossy gate, sharpness (V2) or jitter (4-step), as
every few-step recipe does against a different-noise reference (docs/serve/e2e/h3-test-round-2026-10-14.md).
NVFP4 V2 is opt-in only, like the Trim NVFP4 (§5.4 of docs/ports/h3-trim.md), which fails the same gate against bf16.

## 7. Where a block's time goes (ours)

CUPTI trace of one warm 480p V2 step (`FASTVIDEO_GPU_TRACE=1`, NVFP4 attention + FFN;
`run-2026-10-14/ours/v2-profile-gpu-trace-480p.json`): 1.69 s busy, 3.6% idle, so **33.9 ms per block** against
upstream's ~25 ms (1.25 s / 50).

| per block, 480p (15.1k tokens) | ours ms | upstream (its page) ms |
|---|---:|---:|
| attention: VSA kernel `vsa_mma_attn_tma2` 7.2, tile gather `vsa_tile_qkv` 2.4, pooling `vsa_tile_mean` 1.0, combine 0.9, compressed branch `flash_mma_fwd_d128` 1.5, top-k 0.1 | **13.1** | 7.9 (sparse FP4, quadrant masks); 14.8 VSA Triton bf16; 18.1 dense bf16 SDPA |
| NVFP4 GEMMs: cuBLASLt `cutlass3x_sm120 … ue4m3xe2m1 128x128x256` (6 per block) 7.4 + `nvjet … vec16` 1.5 | **9.0** | ~9.1 (its per-linear table at 15.1k rows: 4 x 0.88 + 0.92 + 3.01 + 1.65) |
| activation quantization: `nvfp4_quant_bf16_sw` 1.7, `nvfp4_amax_bf16` 0.3 | 2.0 | in its GEMM path |
| norm / modulation / QK-norm+RoPE / SwiGLU (`h3_qk_norm_rope` 1.6, `mx_swiglu` 1.4, norm-mod 1.0, …) | 4.1 | 3.4 (fused: 1.42 + 1.12 + 0.88) |
| casts (`cast_f32_bf16` 1.8 …) and layout (`fvf_split_heads_rows` 0.9, `mx_merge_heads` 0.8, `fill_f` 0.6, …) | 4.4 | (its fused route gathers once) |
| memcpy and idle | 1.4 | |

- **The GEMMs match.** cuBLASLt's choice at 15k rows is as fast as FlashInfer's cutlass. **Not checked: the GEMM
  algorithm at 768p row counts.** No 768p trace was taken. Our 768p step is 2.9x the 480p step for 2.4x the tokens,
  which says attention, not the GEMMs, grows faster.
- **The gap, 8.9 ms per block, is attention (+5.2 ms, 58%) and the non-GEMM glue (+~3.7 ms).** Our quantize, norm,
  cast and layout work comes to 10.5 ms, against the ~6.8 ms that upstream's total leaves for it.
- Our bf16 VSA (13.1 ms with its tiling) is already faster than upstream's own bf16 Triton VSA (14.8 ms), consistent
  with docs/techniques.md.

## 8. Recommendation: sparse FP4 attention

**Yes, as its own opt-in work order, gated on quality and preceded by a 768p trace.** No kernel was started here.
- **It is the larger lever.** Attention is 58% of our 8.9 ms per-block gap at 480p. A sparse FP4 kernel at upstream's
  7.9 ms would save ~5 ms of 33.9 per block, about 15% of a step. At 768p the share should grow: upstream's 768p /
  243-frame figures are bf16 VSA 180 ms against sparse FP4 83 ms with warp skip, and our 768p step grows faster than
  the token count. That 768p gain is unmeasured on ours.
- **The rest of the gap is a smaller item.** About 3.7 ms per block of glue: a separate amax + quantize pass,
  `cast_f32_bf16`, head split / merge, fills, and unfused SwiGLU / QK-norm. It costs no quality and helps every H3
  recipe, so it can go first or in parallel.
- **It must stay opt-in.** FP4 attention adds its own quantization (0.19 rel-L2 per call on the page's test), on top
  of an NVFP4 path that already fails our lossy gate against both our bf16 and upstream's own output. It belongs
  behind the NVFP4 checkpoints as a speed tier, never in a default.
- **Order:** take a 768p trace of the NVFP4 path (a few minutes of GPU), confirm attention's share, then write the
  work order with the page's design (tile-64 quadrant masks on 128-blocks, warp skip, no `delta_s` buffer) and a
  parity target of the dense-FP4 floor.

## 9. Cost and clean-up

- **GPU pods, $3.79 in all:**
  - Two EUR-IS-2 hosts never started the container within 16-20 min (a 16 GB image pull), and were deleted: $1.51.
  - The run itself: EU-NL-1, 55 min including a 12-min pull and the 17-min fetch, $2.28.
  - The fix that got a host: drop the launcher's `allowedCudaVersions ["13.0"]` (hosts now run CUDA 13.2 drivers),
    and a smaller disk.
- **The R2 publish CPU pod:** $0.17.
- **Total ≈ $3.96.**
- Every pod was deleted and checked gone (`GET /pods/<id>`: 404), and each local backstop was stopped by its exact
  PID after its command line was checked.
- Every staged bundle in `fv-weights-staging/tmp/fasth3-nvfp4-bundle-*` was removed, all of them this run's own. One,
  whose key was cut from the log, was removed by that run-unique prefix.

Not done for lack of need or budget: upstream 768p VAE times (its compile bug), the 768p / 243-frame cells, a 768p
trace, the dense bf16 / MXFP8 control on this pod, and the published NVFP4 VAE (neither side reads it).
