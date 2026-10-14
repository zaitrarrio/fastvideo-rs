# H3 VAE decode tile batching (work order, 2026-10-14)

**Verdict: stopped at Phase 1 under the work order's stop rule.** Batching
all of a chunk's tiles into one ViT forward, which is the existing
`FASTVIDEO_H3_VAE_TILE_BATCH=196` path, cuts the 768p 5 s video VAE by
**1.6 %** (6.292 s → 6.192 s, median of 3 seeds). At 480p it changes
nothing (2.666 s → 2.669 s). The target was ≥ 20 % and the stop threshold
< 10 %. Parity is exact: every frame is byte-identical to the default path at
both resolutions. No code changed, and no default changed.

Raw data: `artifacts/perf/h3-vae-tile-batching/` contains:

* `baseline/` and `after/`: `benchmark.json`, `run.json`, `log.txt` and
  `frames.sha` for each cell.
* `compare/`: the `compare-clips` reports.
* `progress.log`, `ledger.tsv`.
* `driver/`: `vt.sh` (the on-pod driver), `summarize.py`, and the prompt file.

## Setup

| | |
|---|---|
| Pod | `yn0t2zzi23v0ew`, Runpod SECURE, EU-RO-1, RTX PRO 6000 Blackwell Server Edition (97 887 MiB, **0 MiB used at boot**), driver 595.91.07, $2.49/hr |
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:efa9348a68b8…` (`sha-632742a`, main HEAD), `fv-gpucheck 0.1.15`, build id `384cb1da22579303`. **Every cell ran on this one pod with this one image and binary** |
| Weights | EU volume `bz0ofkfx45`, read only (`h3-base`, the 4-step VSA MXFP8 tree, the FastH3 LoRA, `auxiliary/lpips`). Nothing was written to it |
| Harness | `scripts/serve/e2e/pod.sh` (sidecar, `FV_E2E_NO_SECRETS=1`, 5400 s DELETE backstop, 20 min GPU-idle self-delete) + `driver/vt.sh` |
| Workload | `fv-gpucheck --techniques h3/fasth3_4step_vsa --mode fast h3 gen --h3-recipe 4step-vsa`, 5 s (124 frames), `--text-encoder resident-fp8 --dit-offload resident --warm`. The h3-demo prompt at seeds 0 / 1000 / 2000 (`prompts-5x3.json` of the calibrated gate). `FASTVIDEO_ATTN_SAGE=0 FASTVIDEO_FLASH_KERNEL=cudnn` (pinned picks: every cell is deterministic), `FASTVIDEO_H3_VAE_CHECK=1` in every cell |
| Order | 768p baseline → 768p `TILE_BATCH=196` → 768p baseline again (drift and determinism control) → 480p baseline → 480p `196` |
| Cost | 1652 s at $2.49/hr = **$1.14**. The pod was deleted and checked gone (404) |

The VAE check runs once per process, on the first tile batch. That batch
belongs to the untimed warm-up clip, so the timed clips are the plain fused
path.

## Phase 0 / Phase 1: per resolution (medians of 3 seeds)

"Video VAE" is `DecodeSplit::vae_s`: GPU time between chunk hand-offs, the
part comparable with FastVideo's `video_decoding_stage`. "Video decode" is the
whole decode through a finished mp4.

| resolution | path | tiles / forward | forwards / decode | denoise s | video VAE s | video decode s | generate s | VAE share of generate |
|---|---|---|---|---|---|---|---|---|
| 768p (1344x768) | baseline (default 8) | 7 | 28 | 20.08 | **6.292** | 6.408 | 26.88 | 23.4 % |
| 768p | baseline, repeat | 7 | 28 | 20.05 | 6.289 | 6.395 | 26.86 | 23.4 % |
| 768p | **Phase 1** (`=196`) | 28 | 7 | 20.06 | **6.192 (−1.6 %)** | 6.314 (−1.5 %) | 26.80 (−0.3 %) | 23.1 % |
| 480p (832x480) | baseline (default 8) | 6 | 14 | 7.64 | **2.666** | 2.720 | 10.71 | 24.9 % |
| 480p | **Phase 1** (`=196`) | 12 | 7 | 7.62 | **2.669 (+0.1 %)** | 2.733 | 10.68 | 25.0 % |
| — | Phase 2 | — | — | not run (stop rule; see below) | | | | |

Per-clip spread is about ±0.01 s, and the two 768p baselines agree within
0.003 s. The 0.10 s gain at 768p is therefore real, but it is small.

**Peak memory.** nvidia-smi showed 69 837 MiB at 768p and 61 773 MiB at
480p, identical for both paths. There is no OOM risk on 96 GB, so no bisection
was needed.

## Finding: `=196` is a per-chunk cap, not "all 196 tiles"

`decode_clip` batches within one temporal chunk:
`balanced_batch(tiles_in_chunk, tile_batch)`. Any value ≥ 28 therefore gives
28 tiles per forward and 7 forwards per 768p decode, one per chunk. A
whole-clip batch, meaning 196 tiles in a single forward, would need new code
that decodes several chunks before blending them. It was not built, for
three reasons:

1. The step from 28 to 7 forwards bought 1.6 %. The step from 7 to 1 would
   remove 6 more forwards, which by the same per-forward saving is about
   0.03 s (≈ 0.5 %).
2. It would raise the working set from one chunk to the whole clip.
3. The stop rule applies.

## Why batching cannot move this decoder much

* **FLOPs.** Per tile (1792 voxels + 5 suffix tokens = 1797 tokens, width
  2048, SwiGLU 8192, 36 blocks):
  * linears: 2 · 1797 · 2048 · 32768 ≈ 241 GFLOP per block
  * attention: 4 · 1797² · 2048 ≈ 26 GFLOP per block
  * total: 9.6 TFLOP per tile, **1.89 PFLOP per 768p decode** (196 tiles)

  At 6.29 s that is about **300 TFLOPS sustained**. The decode is GEMM-bound
  already at 7 tiles per forward (M = 12.6 k rows), not launch-bound.
* **Per-op profile** (`FASTVIDEO_H3_VAE_CHECK`, fused path, each op
  synchronized, one 768p forward). Batching changes the per-tile time by 5 %
  even with per-op syncs inflating the small ops. In the unsynchronized
  pipeline it is 1.6 %:

  | op | 7 tiles (s) | 28 tiles (s) | 28 tiles / (4 x 7 tiles) |
  |---|---|---|---|
  | GEMMs (ff_in, ff_out, qkv, out) | 0.155 | 0.584 | 0.94 |
  | SDPA | 0.023 | 0.089 | 0.97 |
  | element-wise (swiglu, residual+norm, qkv heads, merge heads) | 0.051 | 0.192 | 0.94 |
  | unpatchify + proj_in/out | 0.003 | 0.014 | 1.17 |
  | **forward** | **0.233** | **0.882** | **0.95** |

  At 480p (6 against 12 tiles per forward) the ratio is 0.98 synchronized and
  1.00 in the pipeline.
* **Launches.** Going from 28 to 7 forwards saves about 21 x ~370 ≈ 7.8 k
  launches. The measured 0.10 s is about 13 µs per launch saved, which is
  launch latency plus wave tails, and it is all there is to gain.
* **Decode wait.** The decode thread's hand-off wait rose from 0.27 s to
  0.88 s at 768p. Larger forwards let the CPU enqueue further ahead before it
  blocks on the 3-deep frame queue. This is CPU-side and off the GPU's
  critical path: video decode still fell by 0.094 s.

## Phase 3: parity and the quality gate

* **`FASTVIDEO_H3_VAE_CHECK`** (fused against unfused ViT, first tile batch):
  pass in all 5 processes. rel-L2 against the 1e-2 limit:

  | cell | rel-L2 |
  |---|---|
  | 768p, 7 tiles | 1.397e-3 |
  | 768p, 28 tiles | 1.651e-3 |
  | 480p, 6 tiles | 1.593e-3 |
  | 480p, 12 tiles | 1.767e-3 |

  The verdict was "fused kept" every time.
* **Pixel parity, Phase 1 against the default path.** The sha256 of the
  frame PNGs is **identical for every clip**:
  * 768p, 3 seeds + warm: `454cc626…`, `adc1faaf…`, `461f6abc…`
  * 480p, 3 seeds + warm: `2cb72331…`, `72e5e583…`, `f11dfef4…`

  `compare-clips` gives LPIPS 0.0 (mean and max) for all 6 pairs. The
  baseline repeat is also byte-identical to the first baseline. cuBLAS and
  the SDPA kernel give the same per-row results at M = 7 and M = 28 tiles
  here.
* **Quality gate (`h3/fasth3_4step_vsa` against the 49-forward base): not
  re-run. It cannot regress.** The candidate's frames are the baseline's
  frames byte for byte, so LPIPS, PSNR, sharpness and jitter against any
  reference are identical. Re-rendering the 49-forward base (≈ 7 min per
  clip x 5) would have measured nothing new.
* **TAEH3** (`[techniques.taeh3]`): untouched, since nothing in the code
  changed. Its numbers reproduce trivially.

## Phases 2, 4 and 5: not done (stop rule)

The work order says: "If measurement shows < 10 %, stop and report instead of
pushing further." Phase 1 measured 1.6 % at 768p and 0 % at 480p.

* **Phase 2** (memory-planned batch): skipped. The env path already reaches
  the per-chunk maximum without OOM, and it is worth 1.6 %.
* **Phase 4** (5 prompts x 3 seeds, 1080p, h3-max): skipped.
* **Phase 5**: skipped. The numbers already bound items 1, 3 and 4:
  * Unpatchify: 2 ms per forward, ≈ 0.06 s per 768p decode, < 1 %.
  * RGB pack + copy: 0.11 s in total.
  * Launch overhead, which a CUDA graph would hide: ≤ 0.1 s, per the Phase 1
    delta.

  A 20 % decode win has to come from the GEMMs, which are ~67 % of the
  forward, for example FP8 or MXFP8 decoder linears. That is a numerics
  change and outside this work order.

## Summary

Nothing changed in the code. The existing `FASTVIDEO_H3_VAE_TILE_BATCH=196`
path, which is effectively 28 tiles per forward (the per-chunk cap), was
measured on one RTX PRO 6000 against the default on the same pod and image:

* 768p 5 s: video VAE −1.6 % (6.29 → 6.19 s), generate −0.3 %
* 480p: no change

Output is byte-identical at both resolutions, so parity and the quality gate
pass by construction. The premise that the decode is launch-overhead-heavy
does not hold on this GPU. The 768p decode is ~1.9 PFLOP of mostly GEMM work
running at ~300 TFLOPS. Fewer, larger forwards remove only launch and
wave-tail time, about 0.1 s. A whole-clip (196-tile) batch would need new code
and project to about 0.5 % more. Per the stop rule, Phases 2-5 were not
started. The default stays at 8.
