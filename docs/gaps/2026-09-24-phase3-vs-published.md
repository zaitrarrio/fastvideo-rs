# Phase 3 gate vs published numbers

Date: 2026-09-24
Compares: the Phase 3 gate measured on RTX PRO 6000 96 GB (EUR-IS-1, $2.09/hr, driver 595.91.07, Cubin 120) and a Vast B200 T1 pass (Cubin 100) against published FastH3 / FastWan-QAD figures.
Decision-log entries: Phase 3 gate (2026-09-24), `FVID-2026-09-24-b200-arch-parity`.

Published FastH3 / FastWan-QAD figures are warm end-to-end on different cards
and recipes. This is **not** a same-box A/B.

## Headline

| | |
|---|---|
| FastH3 8-step, s/step (PRO 6000) | 10.8 s |
| Spark 4-step VSA 0.9, s/step (PRO 6000) | 9.8 s |
| Published FastH3 Preview E2E on 1x B200 | 16.2 s |
| Wan 1.3B DMD denoise (17 f) | 1.20 s |

**Do not treat 10.8 s/step as 16.2 s E2E.** The FastH3 blog reports warm wall
time for a finished 5 s 1344x768 clip (4-step, 90% sparse) on one B200. The
gate reported denoise step time on a PRO 6000, 8-step / 80% (or Spark 4-step
/ 90%), and hit the 300 s wall before decode. A B200 gen was not run; only
nvrtc / kernels / random-weight model.

## H3 denoise step time

Seconds per DiT step. The blog has no per-step figure, so the last row is E2E
wall / 4 as a lower bound (decode is inside that 16.2 s).

| Configuration | s / step |
|---|---:|
| Phase 0, PRO 6000 | 75.4 |
| Sep 19 8-step, PRO 6000 | 11.8 |
| Gate 8-step, PRO 6000 | 10.8 |
| Gate Spark 4-step, PRO 6000 | 9.8 |
| H200 Preview 4-step VSA (2026-09-20) | 7.7 |
| Blog 16.2 s E2E / 4 (B200) | 4.05 |

## Apples to apples (and where it is not)

| Workload | Published | Ours (this gate / prior) | Fairness |
|---|---|---|---|
| FastH3 Preview v1, 4-step 90% | 16.2 s warm E2E, 1x B200; 6.1 s on 4x | Spark 9.8 s/step x 3 then 300 s wall. No decode. PRO 6000. | Same sparsity class, different card. E2E unmeasured. |
| Base H3 dense | 132.5 s warm E2E, 1x B200 | Phase 0 75.4 s/step on PRO 6000 (not E2E). Dense attn historically 87 s/step at 38k tokens. | Different metric and card. Dense path is not the gate recipe. |
| FastH3 8-step V2, 5 s 1344x768 | No published 8-step E2E | Sep 19: denoise 94.1 s (11.8 s/step), VAE 22.9 s, wall 145.5 s. Gate: 10.8 s/step, no finish. | Best completed E2E we have. ~9x the blog 4-step B200 time, on a slower card and 2x steps. |
| LTX-2 distilled | Lightricks publishes no latency | 2.0: 1.47 s/step, ok 256 s. 2.5: ancestral 0.39 s, stage-2 1.67 s; decode not proven this pass. | No vendor number to beat. Internal: Sep 19 stage-1 1.59 s/step. |
| FastWan / QAD 1.3B, 5 s 480p | 3.4 s on RTX 4090 (FP8 + compile + TAEHV) | Gate: 17 frames (~1 s), denoise 1.20 s, VAE 1.12 s, generate 2.54 s. Prior 5090 + TAEHV 8 s clip: 6.24 s generate. | Different duration, decoder, and compile stack. 5090 TAEHV is ~0.78 s per video-second vs QAD 0.68. |
| B200 kernels | Blog assumes a working B200 stack | NVRTC sm100 pass, Cubin(100). Model random-weight pass. Same cast/affine bit-exact fails as PRO 6000. | Correctness on SM 100. No H3/Wan gen on B200. |

## If we scale PRO 6000 -> B200

The decision log already treats a PRO 6000 as ~2.5-3x slower than a B200.
Applying that only to denoise (an estimate, not a claim):

| Estimate | At 2.5x | At 3x |
|---|---:|---:|
| Spark 4-step denoise (9.8 x 4) | 15.7 s | 13.1 s |
| 8-step denoise (10.8 x 8) | 34.6 s | 28.8 s |
| Blog Preview E2E (measured) | 16.2 s | 16.2 s |

Spark denoise-only lands next to the blog E2E. Suggestive, not a result: we
still owed decode, a warm second clip, and a real B200 gen.

## Gate cell outcomes

| Cell | Outcome |
|---|---|
| Spark loads | pass |
| Wan 1.3B clip | pass |
| LTX-2.0 1.47 s/step | pass |
| H3 | 300 s wall, no decode |
| LTX-2.5 | no decode |
| Hunyuan | fail: Diffusers MMDiT key names |
| LTX-2.3 | fail: `prompt_adaln_single` |

Loader contracts that blocked Phase 3 (Spark `.set_weight`, H3 gate, LTX-2.3
keyframes, missing Wan weights) were fixed. Remaining misses were Hunyuan
Diffusers MMDiT names and LTX-2.3 `prompt_adaln_single`.

## What published sources actually claim

- FastH3 blog: warm E2E, 5 s at 1344x768. Base dense 132.5 s / 1x B200;
  Preview v1 16.2 s / 1x B200, 6.1 s / 4x.
- FastWan-QAD: 3.4 s for a 5 s 480p clip on a 4090 via FP8 linears, FP8
  attention, TAEHV, and compile.
- Lightricks: no LTX-2 latency.
- Sol-Engine: algorithm contracts (TeaCache, PISA, NVFP4), not a six-model
  B200 scoreboard we can quote.

## What closed it

The H200 (2026-09-24) and B200 (2026-09-25) warm suites ran the real thing:
fasth3-4step-vsa generate **35.0 s on H200 (7.0 s/step)** and **26.5 s on B200
(5.33 s/step, 21.4 s denoise, TAEH3 decode 2.76 s)** against the published
16.2 s, i.e. 2.16x and 1.64x of the blog number. See
`FVID-2026-09-24-h200-warm-suite` and `FVID-2026-09-25-b200-warm-suite`.
