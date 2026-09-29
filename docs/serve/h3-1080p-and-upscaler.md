# H3 at 1080p, and what an upscaler would take

Date: 2026-09-28. Two questions:

- **A.** Does H3 hold up if we generate natively at a 1080p canvas, above
  its trained pixel cap, and what does it cost?
- **B.** What would it take to add a video upscaler, and should we?

Raw run: `artifacts/runpod/hd/d46ca17-09282023/` (benchmark and compare
JSONs, logs). Frames: [`artifacts/serve/h3-1080p/`](../../artifacts/serve/h3-1080p/).

**Update (2026-09-28, later):** native 1080P now ships as an opt-in tier on
h3-turbo and h3-max; see [Shipped: the native 1080P tier](#shipped-the-native-1080p-tier)
for what is served, the 5-prompt gate and the lip-sync proxy.

**Update (2026-09-29):** the rest of the gate ran: h3-max passed 5/5
prompts, the 10 s cells confirmed the per-job memory estimate, and a fal
`resolution: "1080P"` request through fv-serve passed. See
[Gate run, part 2](#gate-run-part-2-2026-09-29-h3-max-5-prompts-10-s-cells-fal-1080p).

## Summary

- **Native 1080p works and adds real detail.** At 1920x1088 and 1088x1920,
  2.02x the trained pixel area, H3 gave 9 of 9 coherent clips over three
  prompts, two recipes and two orientations. We saw no duplicated subjects,
  tiling seams or broken frames, and every per-clip check passed. Compared
  with the same prompt at 768p Lanczos-upscaled to the same canvas, the
  native clips carry 5 to 16x more energy above the 768p band limit, and
  1:1 crops show real texture (waterfalls, tree lines, eyelashes, wood
  grain), not sharpened blur.
- **Cost.** h3-turbo takes 62 s end to end instead of 26 s (2.4x); denoise
  is 47.6 s instead of 19.2 s. h3-max takes 78 s instead of 29 s (2.7x);
  denoise is 63.9 s instead of 21.6 s. Peak memory is 57.5 GB (turbo) and
  41.8 GB (max). Both fit an RTX PRO 6000 or an 80 GB H100, but not a 32 GB
  card resident. On an RTX PRO 6000 at $2.09/hr, one 5 s clip is about
  $0.036 (turbo) or $0.046 (max) of GPU time.
- **Audio.** Container durations are unchanged: video 5.167 s (124 frames),
  audio 5.175 s, at both sizes. The audio latent count depends only on the
  frame count. A crude mouth-motion vs loudness correlation gives the same
  weak result at 768p and 1080p, so it neither shows nor rules out a sync
  change. A SyncNet check was not run.
- **Upscaler.** For H3's own outputs, native 1080p beats the plan
  "768p + a VSR model" on detail. It is also about as cheap as a self-hosted
  FlashVSR pass and cheaper than any hosted upscaler (fal SeedVR2 is $0.25
  per 5 s 1080p clip). I recommend **not porting an upscaler now**. Offer
  1080p as a native H3 canvas behind a gate (below). Revisit an upscaler
  only for 1440p/4K, where native generation stops being viable. If we need
  one, FlashVSR is the candidate to port, because it is Wan 2.1 1.3B plus
  small parts we mostly already have. Estimate: 7-9 agent-days and about
  $15-25 of GPU time.
- **Upscaler benchmark (measured, 2026-09-28 evening).** SeedVR2 3B fp16
  took our 768p H3 clips to 1080p on one RTX PRO 6000 in **150-198 s per
  5 s clip** (0.66-0.85 fps), with a 47 GB peak. To 1440p it took 250 s
  (0.50 fps) with a 77 GB peak. So 768p + SeedVR2 costs about 3x the GPU
  time of native 1080p (26 + ~160 s against 62 s). The output is much
  sharper than Lanczos: 15-66x the energy above the 768p band. On the
  metrics it is even sharper than native 1080p. It gets there by raising
  contrast, shifting tone (PSNR against its own 768p input is 30-35 dB) and
  inventing texture, and fine detail shimmers 1.5-1.9x more than with
  Lanczos. It shows no seams between its 33-frame batches. The weights
  (SeedVR2 3B + VAE, FlashVSR v1.1) are now on both volumes and verified.
  FlashVSR was not benchmarked: its sparse-attention extension built for
  Blackwell (sm_120) in 2 min 19 s, but a check-order bug in the setup
  script stopped the phase, and a coordinator budget stop ended the rerun.
  **The recommendation stands:** serve native 1080p and do not add an
  upscaler for 1080p. See
  [Upscaler benchmark](#upscaler-benchmark-seedvr2-3b-measured).

## Shipped: the native 1080P tier

Owner decision (2026-09-28): ship native H3 1080p as an opt-in size.
Commit `bc730e0`.

### What is served

| Surface | 1080P request | Canvas |
|---|---|---|
| fal `minimax/h3-max`, `minimax/h3-max-turbo`, `minimax/h3-turbo` (t2v, i2v) | `resolution: "1080P"` | 16:9 is generated at 1920x1088 and centre-cropped to 1920x1080; 9:16 1088x1920 → 1080x1920; 1:1 1088x1088 → 1080x1080; 4:3 1440x1088 → 1440x1080; 21:9 is capped by the 1088x1920 pixel budget (2176x960, not cropped) |
| fal director (`minimax/h3-max/director`) | `configure.resolution: "1080p"` | Chunks stream at the generation canvas (1920x1088). Each chunk takes about 2.5x as long to build as at 768p, so a session runs further behind real time: a 5 s chunk takes about 62 s on h3-turbo. Allowed, not recommended for live use |
| Native API | `short_edge: 1080` with `aspect_ratio`, or `size: "1920x1080"` | as fal; an exact size above the trained cap is padded to the multiple of 32 and cropped back |
| FastVideo `/v1/videos` | `size: "1920x1080"` / `width`+`height` | as native |
| Console | fal form (schema enum `1080P`); director form lists `1080p` | |
| MiniMax `/v2/video_generation` | not offered | MiniMax's H3 schema lists `768P`/`2K` (H3) and `480P`/`768P` (H3-Max); there is no `1080P` to map |

- **Models.** `h3-turbo` and `h3-max` carry the tier (`H3Recipe::hd_1080p`;
  `[[models]] h3_1080p = false` turns it off). `h3-draft`, the 8-step dense
  recipe and the Ref2VA models do not; 1080P there answers
  `Unsupported(H3Refine1080P)` (422 on `resolution`), as before.
- **Default.** 768P stays the default tier everywhere.
- **Memory gate, at start.** `CudaBackend::new` reads the device's total
  memory (without a context; `FASTVIDEO_DEVICE_BUDGET_GIB` caps it) and drops
  the tier from every H3 model when it is below the tier's plan
  (`plan_1080p_min_device_bytes`: resident-plan weights plus the
  measured-rate working set of 1920x1088x124, 71.2 GiB). 96 GB and 80 GB
  cards keep it (an H100 80 GB reports 79.6 GiB); 48 GB cards drop it and
  answer `H3Refine1080P`. The log says which.
- **Memory check, per job.** Before a 1080P-tier job starts, the engine
  compares its working set (`measured_working_bytes`: 320 KiB per packed row
  plus 2 GiB; 25.5 / 47.4 / 69.3 GiB at 5 / 10 / 15 s) with the free memory
  plus the pool's cached bytes. When it does not fit, the job is refused
  before any work with `H3Refine1080P` and a message naming both sizes. On
  a 96 GB card whose serve process keeps the FP8 text encoder resident
  (54 GiB of weights between jobs, docs/serve/e2e/i2v-resident.md), about
  40 GiB is free: 1080P up to about 7 s fits and 10-15 s is refused. On an
  80 GB card the text encoder is streamed (`auto`), which leaves room for
  longer clips. At 10 s the estimate covers the measured h3-turbo working
  set (42.9 GiB) with 10 % to spare, and h3-max needs less than half of it
  (22.4 GiB); see [Gate run, part 2](#gate-run-part-2-2026-09-29-h3-max-5-prompts-10-s-cells-fal-1080p).
- **Pricing hint.** GPU time is about 2.4x (turbo) to 2.7x (max) that of
  768P for the same clip; price 1080P at about 2.5x 768P (fal lists 2x).

### Gate run (2026-09-28): h3-turbo, 5 prompts

| Item | Value |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-bc730e0` |
| Command | `FV_FAMILY=hd FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd5.json FV_CELLS="turbo-768p turbo-1080p max-768p max-1080p turbo-1080p-10s max-1080p-10s" scripts/gpu/runpod-http.sh run bc730e0` |
| GPU | RTX PRO 6000 Blackwell Server Edition (97 887 MiB, driver 595.91.07), EUR-IS-1, $2.09/hr |
| Pod | `uv4bu766l4xhs2` (`fv-1080-hd-bc730e0-09282244`), 22:44:36 to 22:59:37 UTC (15 min, about $0.52); deleted and checked (API 404) |
| Prompts | `scripts/gpu/prompts-hd5.json`: talking-head, spark-mountain-lake, ltx-frogyoga (as above) plus ltx-newsbroadcast and h3-demo (two speech scenes) |
| Raw results | `artifacts/runpod/hd/bc730e0-09282244/` (benchmarks, sharpness, lip-sync JSON) |

The run was stopped by a coordinator budget stop after the two turbo cells:
the h3-max cells, the 10 s cells and the fv-serve E2E did not run (see
"Not done").

**Time and memory** (per prompt, warm process; one model load of about
100 s per cell not included):

| Prompt | 768p denoise / e2e | 1080p denoise / e2e | Peak (nvidia-smi) 768p → 1080p |
|---|---|---|---|
| talking-head | 19.0 s / —* | 47.2 s / 64.4 s | 44 294 → 57 412 MiB |
| spark-mountain-lake | 19.2 s / —* | 47.8 s / 62.0 s | 44 430 → 57 484 MiB |
| ltx-frogyoga | 19.5 s / 29.7 s | 48.1 s / 62.4 s | 44 462 → 57 548 MiB |
| ltx-newsbroadcast | 19.6 s / 29.9 s | 48.3 s / 62.6 s | 44 462 → 59 948 MiB |
| h3-demo | 20.2 s / 30.7 s | 49.3 s / 63.5 s | 44 558 → 60 012 MiB |

\*The first two 768p prompts streamed the text encoder into the cache (e2e
122 s and 43 s), so their e2e is not comparable; the other three are
29.7-30.7 s. Peak allocated: 41.4 GiB (768p), 52.7 GiB (1080p), as in the first
run. 1080p costs 2.45x the denoise and 2.1x the video decode.

**Coherence, duplication, tiling.** Frames 0/40/80/120 of every clip,
768p next to 1080p:
[`gate5-turbo-frame40.jpg`](../../artifacts/serve/h3-1080p/gate5-turbo-frame40.jpg),
[`gate5-turbo-frame120.jpg`](../../artifacts/serve/h3-1080p/gate5-turbo-frame120.jpg).
All five 1080p clips are coherent single scenes; the wider canvas shows
more scene (more bridge windows on the starship, more of the valley), not a
repeated one. No duplicated subjects and no tile seams or block grid. One
content note: the 1080p news clip has two "LIVE" bugs (top left and bottom
right), a plausible broadcast overlay rather than a tiling copy. PASS 5/5.

**Sharpness vs Lanczos-stretched 768p** (the method above, on the kept
PNG keyframes, measured at 1920x1088):

| Prompt | Laplacian var (Lanczos → native) | Gradient ratio | Energy above the 768p band |
|---|---|---:|---:|
| talking-head | 21.9 → 43.6 | 0.92 | 7.0x |
| spark-mountain-lake | 36.6 → 227.0 | 1.86 | 23.2x |
| ltx-frogyoga | 76.6 → 197.7 | 1.22 | 4.7x |
| ltx-newsbroadcast | 32.1 → 78.5 | 1.02 | 5.4x |
| h3-demo | 65.5 → 98.1 | 1.01 | 2.6x |

Native 1080p carries 2.6-23x the energy above what a 768p clip can hold,
and a higher Laplacian variance on every prompt. PASS 5/5 (criterion: more
detail above the 768p band limit than the Lanczos baseline).

### Lip sync

SyncNet was not run: it needs its own checkpoint, and new weights must go
on both volumes with the owner's approval (CLAUDE.md). A documented proxy,
`scripts/gpu/lipsync_proxy.py`, stands in:

- face: OpenCV Haar detector (opencv-python-headless 4.x, no extra weights);
- articulation: mean absolute change of the mouth region minus that of the
  upper face (head motion and camera moves cancel), per frame, on 96x48
  crops so resolutions are measured alike;
- audio: speech-band (300-3400 Hz) RMS per video frame;
- the lag (±6 frames, ±250 ms) with the best Pearson r, a confidence (best r
  minus the median over lags) and the speech contrast (articulation in the
  loudest 30 % of frames minus the quietest 30 %, in SD).

| Clip (h3-turbo) | Face frames | Best lag | r (lag 0) | Speech contrast |
|---|---|---|---|---|
| talking-head 768p | 124/124 | 0 | 0.169 (0.169) | 0.61 |
| talking-head 1080p | 124/124 | 0 | 0.190 (0.190) | 0.76 |
| ltx-newsbroadcast 768p | 75/124 | -5 | 0.187 (0.059) | 0.21 |
| ltx-newsbroadcast 1080p | 65/124 | -1 | 0.267 (0.176) | 0.40 |
| h3-demo (both) | 50 and 41/124 | n/a (the face is on screen in under half the frames) | | |
| **pooled 768p** | 2 clips | +6 | 0.157 (0.114) | 0.40 |
| **pooled 1080p** | 2 clips | **-1 (-42 ms)** | 0.192 (0.183) | 0.58 |

Reading: at 1080p the mouth works with the speech (positive r at lag 0,
positive speech contrast) and the best lag is within ±1 frame, inside the
±2-frame window; 768p is no better (its news clip peaks at -5 frames and
drags the pooled lag out). The earlier run's clips (d46ca17) give the same
picture: talking-head lag 0 at 768p and 1 at 1080p (turbo), 0 and 2 (max),
0 and 0 (turbo 9:16). The proxy's correlations are weak (r about 0.2), so
it rules out a gross sync break (a shifted or frozen mouth), not a subtle
one; a SyncNet score is still the stronger check.

### Not done in the first gate run (budget stop, 22:52 UTC)

The h3-max 5-prompt cells, the 10 s cells and the fv-serve fal request did
not run on 2026-09-28. They ran on 2026-09-29 (next section).

### Gate run, part 2 (2026-09-29): h3-max 5 prompts, 10 s cells, fal 1080P

| Item | Value |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-2cd1ba0` (main `2cd1ba0`, fv-gpucheck build id `e459d24cc5a25933`) |
| Command | `RUNPOD_VOLUME_ID=jg48s6o1w0 FV_POD_CAP_S=3000 FV_FAMILY=hd FV_PROMPTS=/opt/fastvideo-rs/scripts/gpu/prompts-hd5.json FV_CELLS="max-768p max-1080p turbo-1080p-10s max-1080p-10s" FV_FETCH_TREE=1 FV_FETCH_SKIP='\.cache$' FV_SKIP_TAE=1 scripts/gpu/runpod-http.sh run 2cd1ba0` |
| GPU | RTX PRO 6000 Blackwell Server Edition (97 887 MiB, driver 595.91.07), EUR-IS-1, $2.09/hr |
| Pod | `9t9awkj9iquyc7` (`fv-1080b-hd-2cd1ba0-09290122`), 01:22:29 to about 02:09:50 UTC (47 min, about $1.65); deleted, GET 404. The cells ended at 01:54:30; the last 15 min were the 1.5 GB tree fetch through the proxy (the 10 s cells keep every PNG frame, 600 MB each). Next time skip `frames/.*\.png$` for the `-10s` cells |
| Raw results | `artifacts/runpod/hd/2cd1ba0-09290122/` (benchmarks, compare-clips JSON, `report.txt`, lip-sync JSON) |

The driver exited 1 because the weight gate reports `sol-h3-spark` incomplete
(the latent upscaler file under `upscaler/` is missing on the EU volume).
No cell here uses it; every cell ran and passed.

**h3-max, 5 prompts** (warm process; model load 104-108 s per cell not
included):

| Prompt | 768p denoise / e2e | 1080p denoise / e2e | Peak (nvidia-smi) 768p → 1080p |
|---|---|---|---|
| talking-head | 21.3 s / —* | 64.3 s / 79.2 s | 35 008 → 41 056 MiB |
| spark-mountain-lake | 21.5 s / —* | 66.1 s / 81.2 s | 35 072 → 41 120 MiB |
| ltx-frogyoga | 21.8 s / —* | 68.5 s / 83.4 s | 35 104 → 41 216 MiB |
| ltx-newsbroadcast | 21.8 s / —* | 68.9 s / 84.0 s | 35 136 → 41 216 MiB |
| h3-demo | 22.1 s / —* | 68.8 s / 83.9 s | 35 136 → 41 280 MiB |

\*The 768p cell was the pod's first, so all five prompts streamed the text
encoder into the cache (e2e 88-118 s); denoise + decode is 28.0-28.8 s.
Steps: 8.3 / 4.8 / 4.5 / 4.2 s at 768p, 30.3 / 13.8 / 12.7 / 11.6 s at
1080p. Peak allocated 32.4 GiB (768p) and 38.2 GiB (1080p), as in the first
run. 1080p costs 3.1x the denoise (h3-max's dense first step scales about
quadratically) and 2.2x the video decode (6.7 → 14.5 s). This matches the
3-prompt run (78.4 s e2e, 41.8 GB).

**Coherence, duplication, tiling.** Keyframes 0 and 80 of every 1080p clip
(`max-1080p/keyframes/`): all five are coherent. There are no duplicated
subjects, no tile seams and no block grid. Two prompts describe several
shots (`ltx-newsbroadcast`: the reporter, then the site; `h3-demo`:
`[Shot 1]`…), and those clips change shot as the prompt asks. The
compare-clips patch-boundary ratio is 1.03-1.84 on average. PASS 5/5.

**Sharpness vs Lanczos-stretched 768p** (`hd_report.py`, lossless PNG
keyframes at 1920x1088):

| Prompt | Laplacian var (Lanczos → native) | Energy above the 768p band |
|---|---|---:|
| talking-head | 21.4 → 36.6 | 9.7x |
| spark-mountain-lake | 56.1 → 179.1 | 6.9x |
| ltx-frogyoga | 56.6 → 210.1 | 6.5x |
| ltx-newsbroadcast | 101.6 → 103.1 | 2.3x |
| h3-demo | 133.1 → 129.5 | 2.2x |

Every clip carries 2.2-9.7x the energy above the 768p band limit. PASS 5/5
on that criterion. The Laplacian variance is higher on 3 prompts and level
on the two multi-shot prompts, where the 1080p keyframes show other shots
than the 768p ones.

**Lip-sync proxy (h3-max).**

| Clip | Face frames | Best lag | r (lag 0) | Speech contrast |
|---|---|---|---|---|
| talking-head 768p / 1080p | 124 / 124 | -1 / +2 | 0.134 (0.115) / 0.154 (0.105) | 0.54 / 0.47 |
| ltx-newsbroadcast 768p / 1080p | 100 / 71 | -6 / +1 | -0.334 / -0.017 | -0.93 / -0.28 |

On talking-head the proxy reads the same at both sizes: best lag within
±2 frames and a positive r. The news clips give no usable signal at either
size (negative r and speech contrast; the reporter is on screen for only
part of the clip). That matches h3-max's first run (talking-head lag 0 and
2). The proxy does not show a sync change at 1080p.

**10 s at 1080p (1920x1088x244, `talking-head`, one warm process)** against
the per-job estimate (`measured_working_bytes`, 47.4 GiB at 10 s):

| Cell | Denoise | Video decode | e2e | Weights resident (text stage) | Peak allocated (denoise) | Working set measured | Estimate | nvidia-smi peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h3-turbo | 114.2 s | 28.9 s | 143.9 s | 30.1 GiB | 73.0 GiB | **42.9 GiB** | 47.4 GiB | 82 820 MiB |
| h3-max | 200.8 s | 29.0 s | 230.6 s | 26.5 GiB | 48.9 GiB | **22.4 GiB** | 47.4 GiB | 54 028 MiB |

- **The estimate holds.** On h3-turbo it covers the measured 10 s working
  set with 4.5 GiB (10 %) to spare. At 5 s it was 25.5 GiB estimated
  against 22.5 GiB measured. h3-max's Sol-Attn recipe needs less than
  half as much (22.4 GiB at 10 s, about 11.6 GiB at 5 s), so the one
  estimate is conservative for max. It is a single constant per packed row
  over both recipes; a per-recipe rate would admit longer h3-max clips on a
  card with resident text weights.
- On a 96 GB card with the text encoder resident (about 40 GiB free
  between jobs), a 10 s 1080P job is refused on both models, as designed.
  A streamed-encoder 80 GB card (about 49 GiB free beside the DiT) admits
  it. h3-turbo peaked at 80.2 GiB reserved (82.8 GB nvidia-smi), which
  fits an RTX PRO 6000 and is at the edge of an 80 GB H100 with the DiT
  resident.
- **Time.** 10 s costs 2.4x (turbo) and 3.0x (max) the 5 s denoise; max's
  dense first step is 98 s of its 201 s. At $2.09/hr a 10 s 1080p clip is
  about $0.08 (turbo) and $0.13 (max) of GPU time.
- **Frames.** Both clips have 244 frames and 243/243 face frames. Neither
  has a hard cut: ffmpeg scene scores are all below 0.1. h3-turbo is one
  stable scene. On **h3-max the room drifts**: over the 10 s the background
  (windows, sofa, shelves) morphs continuously into another room while the
  speaker stays the same person. It is not a cut and not tiling; it is
  long-clip background drift at 2x the trained area, and h3-max shows it at
  10 s, h3-turbo does not. Strip (every 20th frame):
  [`max-1080p-10s-strip.jpg`](../../artifacts/serve/h3-1080p/max-1080p-10s-strip.jpg).
  Lip-sync proxy: turbo lag 0, r 0.125, speech contrast 0.47; max lag -1,
  r 0.132, contrast -0.03.

**fal `resolution: "1080P"` through fv-serve.** The h3-turbo variant
image from its Runpod template (`fv-serve-h3-turbo-pod`,
`ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:c782eb37…`, tag
`h3-turbo-sha-2cd1ba0`) booted with its baked `runpod.toml` on an RTX PRO
6000 (EUR-IS-1, EU volume read only). Pod `bn56zvm3xeppih`, 532 s (about
$0.31); deleted, GET 404. Create to `/ping` 200 took 441 s, mostly the
first pull of the slim image on that host. Driver:
`scripts/serve/fal-queue-smoke.sh <base> minimax/h3-turbo text-to-video
'{"prompt": <fox>, "seed": 1, "resolution": "1080P"}'`. Raw:
`artifacts/serve/e2e/slim-images/h3-turbo/results.jsonl`.

| Request | Result | Wall (submit → result) | fal timings | MP4 |
|---|---|---:|---|---|
| `minimax/h3-turbo/text-to-video`, `resolution: "1080P"`, seed 1 | PASS, `COMPLETED` | 67.7 s | denoise 47.0 s, video decode 14.0 s, encode 0.65 s, total 62.4 s | **1920x1080** (generated 1920x1088, cropped), 124 frames @ 24, AAC 32 kHz; 10.7 MB, R2 |

The served denoise (47.0 s) matches the matrix cell (47.2-49.3 s), and the
clip is a coherent single scene
([`fal-1080p-turbo-frame60.jpg`](../../artifacts/serve/h3-1080p/fal-1080p-turbo-frame60.jpg)).
The 1080P tier is served end to end. The same pod ran the 480P request of
the h3-turbo E2E, and its output was byte-identical to the old image's
(docs/serve/images.md, GPU smoke).

**Spend (part 2):** hd pod $1.65 + fal pod $0.31 = **about $1.96**.

## Part A: native H3 1080p

### Setup

| Item | Value |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-d46ca17` (contains 49bd7c0 `--oversize-canvas` and the `hd` family) |
| Command | `FV_FAMILY=hd FV_LPIPS=1 FV_FETCH_TREE=1 FV_SKIP_TAE=1 scripts/gpu/runpod-http.sh run d46ca17` |
| GPU | RTX PRO 6000 Blackwell Server Edition (97 887 MiB, driver 595.91.07), EUR-IS-1, $2.09/hr |
| Pod | `awwqw8dm96u7h8` (`fv-hd-d46ca17-09282023`), 20:23:49 to 20:57:17 UTC (33.5 min, about $1.17), deleted and checked (API 404) |
| Weights | EU volume `jg48s6o1w0`, read only |
| Recipes | h3-turbo = `--h3-recipe 4step-vsa --techniques h3/fasth3_4step_vsa`; h3-max = `--h3-recipe sol-h3 --techniques h3/sol_h3_4step_engine_ladder` (the serve catalog's tiers) |
| Prompts | `scripts/gpu/prompts-hd.json`: `talking-head` (speech, seed 1024), `spark-mountain-lake` (landscape detail, seed 42), `ltx-frogyoga` (dialogue scene, seed 42) |
| Clip | 5 s = 124 frames at 24 fps, H3 audio at 32 kHz stereo |

**Canvas.** 1920x1088 is 60x34 x 32 px, so it is a valid H3 multiple.
Latents are 120x68 and tokens 60x34 per latent frame (2.02x the rows of
1344x768). The VSA tiles and the VAE tile planner handle partial tiles
already (768p is 42 tokens wide, not a multiple of 4 either), so no
kernel or layout change was needed. Only `check_canvas`'s 768x1344 pixel cap
refuses the canvas. `fv-gpucheck h3 gen --oversize-canvas` (test path only)
builds the request with `H3Geometry::new` instead of `H3Geometry::checked`.
Serving and `H3Request::sized` keep the cap.

### Time and memory (medians of 3 prompts, one warm process per cell)

| Cell | Canvas | Denoise | Step (s) | Video decode | End to end* | Peak GPU (nvidia-smi) | Peak allocated |
|---|---|---:|---|---:|---:|---:|---:|
| h3-turbo 768p | 1344x768 | 19.16 s | 4.8 x 4 | 6.43 s | 26.3 s | 44 430 MiB | 41.4 GiB |
| h3-turbo 1080p | 1920x1088 | **47.65 s** (2.49x) | 12.0 x 4 | 13.80 s (2.15x) | **61.9 s** (2.36x) | 57 484 MiB | 52.7 GiB |
| h3-turbo 768p 9:16 | 768x1344 | 19.36 s | 4.8 x 4 | 6.53 s | 26.3 s | 44 428 MiB | 41.4 GiB |
| h3-turbo 1080p 9:16 | 1088x1920 | 48.00 s (2.48x) | 12.0 x 4 | 13.77 s | 64.0 s (2.43x) | 57 484 MiB | 52.7 GiB |
| h3-max 768p | 1344x768 | 21.63 s | 8.4 / 4.8 / 4.5 / 4.2 | 6.53 s | 28.6 s | 35 916 MiB | 32.4 GiB |
| h3-max 1080p | 1920x1088 | **63.93 s** (2.96x) | 28.4 / 13.3 / 12.1 / 11.0 | 13.72 s | **78.4 s** (2.74x) | 41 836 MiB | 38.2 GiB |

\*`e2e_seconds` of a clip with the prompt's conditioning cached. The
turbo-768p cell was the pod's first, and its three prompts streamed the text
encoder (e2e about 101 s). Its figure here is denoise + decode + the other
cells' ~0.7 s overhead, which matches the turbo-768p-9:16 cell. Model load
is 84-98 s per process and is not included.

- **Scaling.** The VSA turbo steps scale 2.5x for 2.02x the tokens. h3-max's
  dense first step scales 3.4x (quadratic attention); its Sol-Attn steps
  scale about 2.7x. Decode scales about with pixels.
- **Memory.** The planner's `planned peak 74.11 GiB resident` for 1920x1088
  is conservative: the measured peak is 57.5 GB. FFN chunking switched on by
  itself at 1080p (`ffn_chunking.chunked_calls` 200/202). A 32 GB card would
  need the streamed plan (planned 33.5 GiB), which was not measured.

### Quality

Two instruments. The 768p clip Lanczos-upscaled to the 1080p canvas is the
"naive upscale" baseline.

1. **`fv-gpucheck compare-clips`** on the pod, on lossless PNGs (all 124
   frame pairs). The baseline is 768p Lanczos-upscaled with ffmpeg; the
   candidate is native 1080p. Same prompt and seed, but a different noise
   shape, so a different video. LPIPS and PSNR therefore measure "different
   content" (LPIPS 0.67-0.78, PSNR 7-12 dB) and are not a quality score.
   The repo's promotion gate (`gate-policy.toml`) assumes same-content arms
   and does not apply here. The useful columns are the ratios.
2. **No-reference sharpness** on frames 0/40/80/120 decoded from each MP4:
   Laplacian variance, and the share of spectral energy above 768p's band
   limit at the 1080p canvas (0.353 cycles/px). A Lanczos upscale has almost
   none there by construction. Native content has energy there only if the
   model drew detail at that scale.

| Pair (768p Lanczos → native) | Prompt | Gradient sharpness ratio | Temporal jitter ratio | Patch-boundary ratio | Laplacian var (Lanczos → native) | Energy above 768p band (x) |
|---|---|---:|---:|---:|---|---:|
| turbo 16:9 | talking-head | 0.995 | 0.98 | 1.16 | 20 → 38 | 6.3x |
| turbo 16:9 | mountain-lake | 2.013 | 1.59 | 2.00 | 35 → 220 | 15.7x |
| turbo 16:9 | frog-yoga | 1.291 | 1.24 | 1.48 | 75 → 195 | 4.6x |
| turbo 9:16 | talking-head | 1.401 | 1.13 | 1.56 | 23 → 128 | 55x |
| turbo 9:16 | mountain-lake | 1.560 | 0.95 | 1.63 | 50 → 213 | 8.1x |
| turbo 9:16 | frog-yoga | 1.102 | 1.47 | 1.29 | 103 → 185 | 6.7x |
| max 16:9 | talking-head | 1.007 | 1.08 | 1.24 | 20 → 35 | 8.9x |
| max 16:9 | mountain-lake | 1.800 | 1.73 | 1.84 | 50 → 177 | 6.4x |
| max 16:9 | frog-yoga | 1.259 | 1.31 | 1.36 | 56 → 200 | 5.7x |

How to read it:

- **Gradient sharpness ratio** (mean |∇| native / Lanczos): 1.0-2.0. For the
  16:9 talking head it is flat, because a medium close-up of a face against
  soft window light has little fine structure at either size.
- **Jitter and patch-boundary ratios above 1** follow the extra
  high-frequency content. More detail moves more between frames and crosses
  more 8/16/32-px boundaries. The frames show no block grid or boundary
  seams (crops below), so we read them as detail, not artifacts.
  `patch_boundary_ratio` is still the metric to watch in a proper
  same-canvas gate.
- **Energy above the 768p band** is 5-16x the Lanczos baseline (55x for one
  vertical face). This is the number that separates "new detail" from
  "interpolated 768p".

What we saw (frame 40 of every clip):

- [`overview-16x9-frame40.jpg`](../../artifacts/serve/h3-1080p/overview-16x9-frame40.jpg):
  columns turbo 768p, turbo 1080p, max 768p, max 1080p.
- [`overview-9x16-frame40.jpg`](../../artifacts/serve/h3-1080p/overview-9x16-frame40.jpg).
- Every 1080p composition is coherent. The wider canvas shows more scene,
  not a repeated one. max-1080p frog-yoga even follows the prompt better:
  the instructor is a frog.
- turbo-1080p's eagle has an odd dark wing in frame 40, a glitch of the kind
  4-step turbo also shows at 768p.

1:1 crops on the 1920x1088 canvas, with 768p Lanczos next to native:

- [`mountain-lake-1to1.jpg`](../../artifacts/serve/h3-1080p/mountain-lake-1to1.jpg):
  the clearest case. Native has a resolved waterfall, individual trees and
  rock strata; the upscale is soft mush.
- [`talking-head-face-1to1.jpg`](../../artifacts/serve/h3-1080p/talking-head-face-1to1.jpg):
  crisper eyes, brows and hair strands at 1080p. Skin stays natural, with no
  oversharpening halos.
- [`frogyoga-1to1.jpg`](../../artifacts/serve/h3-1080p/frogyoga-1to1.jpg).
- [`vertical-talking-head-1to1.jpg`](../../artifacts/serve/h3-1080p/vertical-talking-head-1to1.jpg).

Per-clip gates from `h3 gen` (frame count 124, middle frame not flat, MP4
with an audio track): **PASS on all 18 clips** (10 checks per cell, 0 FAIL).

**Verdict:** native 1080p beats naive upscaling clearly on landscapes and
dialogue scenes, and modestly on close-up faces. There is no sign of the
usual above-training-resolution failures (duplicated subjects, tiling).
Three prompts are a small sample, and the model was not trained at this
area, so ship it as an explicit canvas, not the default. First run a
5-prompt eval and a SyncNet lip-sync check.

### Audio

| | 768p | 1080p |
|---|---|---|
| MP4 video stream | 5.167 s, 124 frames | 5.167 s, 124 frames |
| MP4 audio stream (AAC) | 5.175 s | 5.175 s |
| `audio.wav` | 5.175 s, 32 kHz | 5.175 s, 32 kHz |
| Mouth-box motion vs audio RMS (talking-head), best correlation within ±6 frames | turbo +0.40, max +0.24 | turbo +0.29, max +0.32 |

The audio latent count follows the frame count only
(`audio_latent_num_frames`), so the audio track cannot drift in length. The
correlation check is crude: a hand-placed mouth box on MP4 frames. It is
weak at both sizes, so it does not settle lip sync either way. The speech
comes from the same joint denoise at both sizes, and listening and a SyncNet
score remain to be done.

### What shipping 1080p would take

1. A `1080p` short-edge tier in the H3 caps (`short_edges` gets 1088; the
   serve layer maps `1080p` to a 1088 short edge, and 1920x1080 output is a
   crop or pad from 1920x1088). `check_canvas` needs a second, explicit cap
   for that tier; the plain 768x1344 cap stays the default. The fal director
   already lists `1080p` for `minimax/h3-max/director` and currently answers
   `invalid_input` (docs/serve/fal-parity.md).
2. An eval gate on the same canvas: 5 prompts (`FV_PROMPTS=5`) at 1080p, with
   `compare-clips` between arms at 1080p (the existing policy applies there),
   plus SyncNet on the speech prompts.
3. A pricing note: about 2.4-2.7x the GPU time of 768p.

Effort: 1-2 agent-days, about $3 of GPU time.

## Part B: upscaler options

Sources: each project's README and model card, the FlashVSR paper
(arXiv 2510.12747, Table 2) and the fal model pages, all read on
2026-09-28. Speed figures labelled "paper" are the authors' numbers and
were not reproduced here.

### What we already have

| Piece | Where | What it does | Relevance to 1080p |
|---|---|---|---|
| LTX-2.5 spatial latent upsampler x2 | `crates/fastvideo-cudarc/src/ltx2/latent_upsampler.rs` | LTX latent x2, then a 3-step distilled refine | Already how LTX reaches 1080p: two-stage at 1920x1088, denoise 24.3 s for 5 s on RTX PRO 6000 (`artifacts/serve/e4-ltx-fps`). Works on LTX latents only (128 channels). |
| H3 x2 latent upscaler (3D conv, `LBH-123-AI/Minimax_h3_latent_Upscaler`) + H3-to-LTX adapter (`Efficient-Large-Model/H3-to-LTX-Latent-Adapter`) + LTX 3-step joint refine | `crates/fastvideo-cudarc/src/h3/spark.rs`, recipe `sol-h3-spark` | Spark: H3 draft at 672x384, x2 in latent space, LTX refine at 1344x768; H3's PCM is kept | The only H3-native super-resolution path we have. Its shapes are pinned to the Spark canvas (`fastvideo-models/src/h3/spark.rs` `H3_INPUT` .. `REFINER_INPUT`). A 1080p variant would draft H3 at 960x544 and refine with LTX at 1920x1088. |
| Wan 2.1 1.3B DiT (FastWan, VSA), Wan 2.1 VAE, TAEHV decoders, SF-Wan causal KV-cache streaming | `crates/fastvideo-cudarc/src/wan/`, `taehv` | T2V | FlashVSR is Wan 2.1 1.3B plus a small LQ projection and a TAEHV-style tiny causal decoder, so most of its graph is already here. |

### Survey

| Model | License (code / weights) | Weights | Architecture | Reported speed / memory | Temporal consistency | Fit for our 768p outputs |
|---|---|---|---|---|---|---|
| **FlashVSR v1.1** (OpenImagingLab) | Apache-2.0 / Apache-2.0 | `JunhaoZhuang/FlashVSR-v1.1`: Wan2.1-1.3B DiT (streaming DMD, LoRA r384 merged), `LQ_proj_in`, `TCDecoder`, Wan2.1 VAE; ~1.75 B params (~3.5 GB bf16) | One-step diffusion. Streaming: causal chunks with a KV cache and 8 frames of lookahead. Locality-constrained block-sparse attention (2x8x8 blocks, 10-20% density) | Paper, 101 frames at 768x1408 output on one A100: Tiny decoder 5.97 s (16.9 fps, 11.1 GB peak); Full 15.5 s (6.5 fps, 18.3 GB). About 12x faster than SeedVR2-3B | Streaming with causal cache; built for long videos | Trained for x4 (community ports run x2). 768p x4 is 3072p, too large. For 1080p either downscale the input first (loses detail) or run x2 to 1536p and resize. Official build needs Block-Sparse-Attention compiled (A100/A800 validated; the model card says RTX 40/50 and H800 are unknown). The GPL-3.0 ComfyUI port replaces it with Sparse_Sage and supports RTX 50. |
| **SeedVR2** 3B / 7B (ByteDance Seed) | Apache-2.0 / Apache-2.0 (commercial use allowed) | `numz/SeedVR2_comfyUI`: 3B fp16 6.78 GB (fp8 3.39 GB); 7B fp16 16.48 GB (fp8 8.24 GB; "sharp" variant); VAE 0.50 GB | One-step diffusion (adversarial post-training). NaDiT with adaptive window attention; its own causal video VAE | Paper table (FlashVSR): 3B, 101 frames at 768x1408 on A100: 70.6 s (1.43 fps), 52.9 GB peak. Official repo: 720p x 100 frames on one H100-80G; 1080p/2K needs 4x H100 sequence parallel. The numz port (Apache-2.0, SDPA fallback, VAE tiling, block swap) runs 1080p on one card | Batches of 4n+1 frames with overlap blending: consistent within a batch, seams possible between batches | Any target size (short side). Known to oversharpen lightly degraded input, which is exactly what a clean 768p generation is. |
| STAR (NJU) | MIT (I2VGen-XL variant); CogVideoX license (CogVideoX-5B variant) | ~2.5 B | Multi-step, text-guided diffusion | Paper table: 682 s per 101 frames (0.15 fps), 24.9 GB; README: about 39 GB for its 4x toy clip | Good | Far too slow to serve. |
| Upscale-A-Video (NTU) | NTU S-Lab License 1.0 (non-commercial) | ~1.1 B | Multi-step: SD x4 upscaler UNet + temporal layers + flow propagation | Paper table: 812 s per 101 frames (0.12 fps) | Good | Excluded (licence, speed). |
| VEnhancer (Vchitect) | No licence stated in the repo | `venhancer_v2.pth` | ControlNet on a video diffusion model; 15-step fast mode; also interpolates frames | README: one A100-80G required | Good | 1x-8x, up to 2K. Too slow, and the licence is unclear. |
| DOVE | Research code | ~10.5 B | One-step (CogVideoX based) | Paper table: 72.8 s per 101 frames, 25.4 GB | Good | No speed advantage over SeedVR2. |
| Real-ESRGAN | BSD-3-Clause | x4plus ~17 M params (64 MB) | GAN, per-frame RRDB CNN | Per-frame CNN, real-time class at 1080p on a modern GPU | None (per frame): flickers on fine texture | Any scale via resize. The cheap baseline; fal's `fal-ai/video-upscaler` runs it. |
| RealBasicVSR | Apache-2.0 | ~6 M params | GAN, recurrent (BasicVSR) with an input-cleaning module | Real-time class | Recurrent propagation, so stable | x4. Tends to paint over or smooth clean content. |
| HunyuanVideo-1.5 SR, 720p→1080p step-distilled | Tencent Hunyuan Community License (territory and scale restrictions; check before use) | Latent upsampler + 8.3 B DiT refine | 8-step latent refine | Not published separately | Good (latent refine) | Tied to the Hunyuan latent space; our Hunyuan 1.5 port is incomplete. |
| LTX-2 spatial / temporal upscalers x2 | LTX-2 Community License | `ltx-2-spatial-upscaler-x2-1.0` | Latent upsampler, then distilled refine steps | Measured in-house (LTX at 1080p, above) | Good (latent refine) | LTX latents only; H3 reaches it only through the Spark adapter. |
| Wan-based SDEdit refine | Wan: Apache-2.0 | Existing | Low-strength video-to-video at the target size with an existing Wan DiT | Not measured | Good (DiT) | No trained SR checkpoint, so quality is unproven. Not recommended. |

### Hosted prices (fal)

| fal endpoint | Model | Price | 5 s at 1080p (1920x1080x121 = 251 MP) |
|---|---|---|---|
| `fal-ai/flashvsr/upscale/video` | FlashVSR | $0.0005 / MP | ~$0.13 |
| `fal-ai/seedvr/upscale/video` | SeedVR2 | $0.001 / MP | ~$0.25 (fal's own example: 1920x1080x121 = $0.25) |
| `fal-ai/video-upscaler` | Real-ESRGAN, per frame | $0.0008 / MP | ~$0.20 |
| `fal-ai/topaz/upscale/video` | Topaz (closed; default Starlight Fast 2) | $0.01/s up to 720p, $0.02/s for 720p-1080p, $0.08/s above 1080p (x2 at 60 fps) | ~$0.10 |

Our own native H3 1080p clip costs $0.036 (turbo) to $0.046 (max) of RTX
PRO 6000 time, less than any of these on top of a 768p generation.

### Cost of the options for a 5 s 1080p H3 clip (RTX PRO 6000)

| Route | GPU seconds | Detail source | Extra resident memory | Status |
|---|---:|---|---|---|
| Native 1080p, turbo / max | 62 / 78 (measured) | The model itself | None | Works (this doc) |
| 768p + Lanczos | 26 / 29 | None | None | Measured baseline: soft |
| 768p + FlashVSR (x2 to 1536p, then resize) | 26 / 29 + ~15-30 (estimate: paper throughput scaled to 1536p output pixels) | VSR model hallucination from 768p | ~4 GB bf16 (+ Wan VAE) | Needs port or sidecar |
| 768p + SeedVR2-3B | 26 / 29 + **150-198 (measured**, numz CLI, SDPA, model load included) | VSR model: sharper than native, but tone-shifted and with invented texture | 47 GB peak (measured) | Measured below: about 3x the GPU time of native |
| Spark-1080p: H3 draft 960x544 → H3 x2 latent → adapter → LTX 3-step refine at 1920x1088 | ~9 (draft) + ~12-15 (refine) + ~4 (LTX decode) ≈ 25-30 (estimate) | LTX refiner at full resolution | LTX 22B resident beside H3 | Pieces exist, pinned to the 768p Spark canvas |

### Recommendation

1. **Now: native H3 1080p as an opt-in canvas.** This needs no new model and
   no new weights, and it is proven coherent on 9/9 clips. Next steps as in
   [What shipping 1080p would take](#what-shipping-1080p-would-take): 1-2
   agent-days, about $3 of GPU time.
2. **Do not port a VSR model for 1080p.** For H3's clean, generated 768p
   outputs, an upscaler would add a second model and about the same GPU time
   (FlashVSR) or much more (SeedVR2). It would only invent detail that
   native generation produces for real.
3. **If we want 1440p/4K later** (native H3 at 4x area is unlikely to hold
   up), port **FlashVSR v1.1**. It is Apache-2.0, the fastest, streaming, and
   its body is the Wan 2.1 1.3B DiT we already run.

   | FlashVSR port step | Effort (agent-days) |
   |---|---|
   | Weight loader for the streaming-DMD DiT, `LQ_proj_in` and `TCDecoder` | 1 |
   | `LQ_proj_in` conditioning into the Wan DiT | 0.5-1 |
   | Streaming causal chunks + KV cache (reuse the SF-Wan causal path) | 1.5-2 |
   | Locality-constrained block-sparse attention on our VSA kernels (2x8x8 tiles, top-k); dense first as the correctness path | 1.5-2 |
   | TCDecoder (TAEHV-style with LQ conditioning; reuse `taehv`) | 1 |
   | Python oracle parity on a pod, quality gate, serve post-process stage | 1.5-2 |
   | **Total** | **7-9 agent-days**; GPU about $15-25 (oracle and benchmark pods) |

   A Python sidecar with the resident-worker pattern would take 1-2
   agent-days plus about $3-5 of GPU, but it adds a PyTorch runtime to the
   serve image. The FlashVSR v1.1 weights are already on both volumes
   (`auxiliary/upscalers/flashvsr-v1.1/`).
4. **The highest-leverage experiment is Spark-1080p**, above all for h3-max.
   Generalise the Spark bridge shapes to a 960x544 draft and a 1920x1088 LTX
   refine. It reuses only ported kernels and weights already on both
   volumes, and could reach 1080p in about 25-30 s instead of 62-78 s.
   Effort: 3-5 agent-days, about $10-15. Risks: the LTX refiner changes H3's
   look, and both models must be resident (memory).
5. SeedVR2 is ruled out for serving on speed, memory and fidelity, and the
   benchmark below confirms it. It takes 150-198 s and 47 GB per 1080p clip,
   and it changes the look of the clip. It remains an option for an offline
   or premium 1440p pass: it runs on one 96 GB card in about 250 s ($0.15
   of GPU per clip).

### Upscaler benchmark: SeedVR2 3B (measured)

Run on 2026-09-28, with the owner's approval for the weights. Raw data:
`artifacts/runpod/hd/d46ca17-09282142/` (speed, memory, 1440p, and the
Lanczos and native metrics) and `artifacts/runpod/hd/d46ca17-09282245/`
(full-clip SeedVR2 1080p metrics).

**Weights, now on both volumes.** They were added add-only by
`fetch-hub-tree.sh`. Each file was downloaded into a temporary
`.<name>.partial-<stamp>` folder, SHA-256'd against the Hub (EU also against
the US list), and the folder was then renamed. A fresh CPU pod per volume
then ran `verify-weights.sh upscalers` (size and SHA-256 of every file):
**ok on US `s2k01690bi` and EU `jg48s6o1w0`**. Nothing else was touched.

| Tree | Repo @ revision | Files | Bytes |
|---|---|---|---:|
| `auxiliary/upscalers/seedvr2/` | `numz/SeedVR2_comfyUI` @ `09ced71` | `seedvr2_ema_3b_fp16.safetensors` (`2fd0e03a…2b304`), `ema_vae_fp16.safetensors` (`20678548…12ca1`) | 7 284 343 622 |
| `auxiliary/upscalers/flashvsr-v1.1/` | `JunhaoZhuang/FlashVSR-v1.1` @ `27561b1` | `diffusion_pytorch_model_streaming_dmd.safetensors`, `LQ_proj_in.ckpt`, `TCDecoder.ckpt`, `Wan2.1_VAE.pth`, `config.json`, `model_index.json` | 6 948 393 656 |

The rows are in `weights-manifest.tsv`, and the hashes are pinned in
`verify-weights.sh` (cell `upscalers`). Logs are in
`artifacts/runpod/fetch-auxiliary-upscalers-*` and
`artifacts/runpod/upscaler-weights/`. Together the two trees add 14.2 GB
per volume.

**Setup.** One RTX PRO 6000 (EUR-IS-1). The `hd` family ran with
`FV_CELLS="turbo-768p turbo-1080p upscaler"`. It regenerated the three turbo
clips at 768p and 1080p. H3 is deterministic here: the Lanczos metrics of
the two runs match to every digit. Then `scripts/gpu/hd-upscaler.sh`
ran SeedVR2 3B fp16 through the numz standalone CLI (pinned `4490bd1f`,
torch 2.11 cu128, SDPA, no compile) on the lossless 768p frames, encoded as
x264 CRF 8 yuv444. The settings were `--batch_size 33
--uniform_batch_size --temporal_overlap 3 --seed 42` with the default `lab`
colour correction. Each clip was one CLI process, so model load is
included. `scripts/gpu/hd_upscaler_metrics.py` scored every 124-frame clip.

**Speed and memory** (5 s clip = 124 frames):

| Target | Output | Wall per clip | fps | Phases (warm) | Peak GPU (nvidia-smi) |
|---|---|---:|---:|---|---:|
| 1088 short side | 1904x1088, resized to 1920x1088 | 150-157 s warm, 174 s cold (first run, hashes the weights); 176-198 s on a second host | 0.66-0.85 | VAE encode 25 s, DiT 39 s (about 20 s of it model load), **VAE decode 64 s**, post-processing and writes about 10 s | 46.7-47.3 GB |
| 1440 short side | 2520x1440 | 246-252 s | 0.50-0.51 | encode 55 s, DiT 55 s, decode 115 s, post 20 s | 76.5-77.0 GB |

For comparison, native H3 turbo at 1080p takes 62 s end to end. 768p
(26 s) + SeedVR2 takes about 176-224 s, so **about 3x the GPU time**, and
the upscale alone costs $0.087-0.115 per clip at $2.09/hr. The VAE, not
the one-step DiT, dominates. A resident worker with a faster VAE would help. Even
then, the DiT alone (about 20 s) plus any decode would at best match the
36 s that native 1080p adds.

**Quality** (luma, all 124 frames; spectra on every 4th frame). SeedVR2
1080p numbers are for talking-head and mountain-lake over the full clip.
For frog-yoga and all 1440p outputs only frames 0-3 were scored; see the
note below.

| Clip | Variant | Laplacian var | Energy above the 768p band (share, x Lanczos) | Warp error (all / high-pass) | Luma flicker | PSNR vs 768p input |
|---|---|---:|---|---|---:|---:|
| talking-head | 768p + Lanczos | 21 | 5.4e-5 (1x) | 1.35 / 0.89 | 0.447 | 51.2 dB |
| | 768p + SeedVR2 | 94 | 8.4e-4 (**15x**) | 1.94 / 1.30 (1.44x / 1.46x) | 0.456 | 34.9 dB |
| | native 1080p | 37 | 3.4e-4 (6x) | 1.23 / 0.85 | 0.547 | n/a |
| mountain-lake | 768p + Lanczos | 37 | 1.3e-4 (1x) | 2.36 / 1.86 | 0.350 | 49.6 dB |
| | 768p + SeedVR2 | 502 | 8.6e-3 (**66x**) | 4.39 / 3.60 (1.86x / 1.93x) | 0.357 | 30.1 dB |
| | native 1080p | 216 | 2.4e-3 (18x) | 3.20 / 2.83 | 0.272 | n/a |
| frog-yoga (frames 0-3) | Lanczos / SeedVR2 / native | 68 / 202 / 174 | 1.1e-4 / 6.6e-4 (6x) / 5.6e-4 | n/a | n/a | 47.0 / 31.7 dB |
| 1440p (frames 0-3), three clips | Lanczos / SeedVR2 at 2520x1440 | 9-27 / 42-122 | SeedVR2 7-12x Lanczos | n/a | n/a | SeedVR2 33.6-36.6 dB |

How to read it:

- **Detail.** SeedVR2 puts 15-66x Lanczos's energy above the 768p band.
  That is 2.5-3.6x even native 1080p's, and its Laplacian variance is also
  above native. It is not "more real detail than native": the crops show
  higher contrast, deeper shadows and synthetic texture, such as vertical
  striations on the cliff and crisp wing feathers.
- **Fidelity.** Area-downscaled back to 768p, SeedVR2 is 30-35 dB from its
  own input, against 50-51 dB for the Lanczos round trip. It changes tone
  and structure, not just adds detail, so for a generated clip it changes
  the look the user saw at 768p.
- **Temporal flicker.** The warp error is the mean |frame t − flow-warped
  frame t−1| over flow-consistent pixels, with flow from the 768p source.
  It rises 1.44-1.93x over Lanczos, and most of the rise is on the
  high-pass band: fine detail shimmers. Native 1080p also carries more
  high-pass detail on the landscape (2.83 vs 1.86), but less than SeedVR2
  (3.60) for less invented structure. Mean-luma flicker is unchanged. The
  per-pair series has **no spikes at the 33-frame batch boundaries**: the
  maxima (1.4-1.8x the median) are at the same motion frames as in
  Lanczos.
- **1440p** behaves the same way (7-12x Lanczos above the band) at 250 s and
  77 GB.

Crops (frame 40, 1:1 on the 1920x1088 canvas; the native column is a
different sample of the same prompt):

- [`upscaler-talking-head-1to1.jpg`](../../artifacts/serve/h3-1080p/upscaler-talking-head-1to1.jpg):
  SeedVR2 gives crisper eyes, brows and skin pores, but also stronger
  contrast and a "processed" look. Native is softer and more natural.
- [`upscaler-spark-mountain-lake-1to1.jpg`](../../artifacts/serve/h3-1080p/upscaler-spark-mountain-lake-1to1.jpg):
  SeedVR2 resolves the waterfall and trees far beyond Lanczos, but paints
  regular striations onto the cliff and darkens the scene.

**Caveats.** Three prompts, one seed each, one GPU type. The first full run
had a frame-collection bug: the CLI writes its PNGs into a subfolder, and an
ffmpeg concat list kept only 4 frames. For that run the SeedVR2 metrics,
the `compare-clips` reports against Lanczos and the keyframes cover frames
0-3 only, and they are not used above except where marked. Speed and memory
are unaffected. The bug is fixed in `hd-upscaler.sh`. A second, shorter run
re-collected full clips for talking-head and mountain-lake; its pod was
stopped by a coordinator budget stop before frog-yoga and 1440p, and its
metrics were computed locally from the lossless PNGs fetched from the pod.
No LPIPS against Lanczos over the full clip was taken.

**FlashVSR v1.1: not benchmarked.** The same hook (`FV_HD_FLASHVSR=1`,
`scripts/gpu/hd_flashvsr.py`) installed a CUDA 12.8 nvcc (apt), torch 2.7.1
cu128 and the official FlashVSR (`cf910c6`). It then built the
Block-Sparse-Attention extension (`49d6c39`) for sm_120 **in 2 min 19 s
without errors**. This is the first sign that the official sparse
attention builds on Blackwell. The script's import check then loaded the
extension before torch (`libc10.so` not found), so the phase stopped. That
is fixed (torch first). The coordinator's budget stop ruled out another GPU
pod. A FlashVSR run on this hook needs about 25 min of RTX PRO 6000 (about
$0.90): 12 min of setup, most of it apt, then the runs.

### Recommendation after the benchmark

1. **Serve native H3 1080p; do not add an upscaler for 1080p.** SeedVR2
   costs about 3x the GPU time of native 1080p, needs another 47 GB model
   resident (or a separate worker), and changes the clip's look (30-35 dB
   from its input, contrast and invented texture). Native 1080p adds
   detail that follows the prompt, at 62 s.
2. **Do not port SeedVR2.** Its time is spent in its VAE, not in the
   one-step DiT, and its output is not faithful to the 768p generation.
3. **For 1440p/4K, evaluate FlashVSR first**, as planned in item 3 above.
   The weights are on both volumes, and the sparse-attention extension
   builds on sm_120. Run `FV_HD_FLASHVSR=1` once (about $0.90) before
   committing to the 7-9 agent-day port. SeedVR2 at 1440p (250 s, 77 GB,
   about $0.15 of GPU per clip) is the fallback for an offline or premium
   tier.

## Spend

| Item | Cost |
|---|---|
| Earlier: pod `awwqw8dm96u7h8` (RTX PRO 6000, EUR-IS-1), 33.5 min | ~$1.17 |
| 1080P gate pod `uv4bu766l4xhs2` (RTX PRO 6000, EUR-IS-1), 15 min | ~$0.52 |
| Weight fetch and verify CPU pods, 7 pods (`b0j2pm5iey43cc`, `w1ukymy5rr6q9i`, `7z3ntxbtmetfb6` US; `yphft7ob6phlrg`, `j369wnd98qkwc8` EU; verify `8ie3j3ouvtnw8n` US, `1lu3s5qw2mcp91` EU), about 10 pod-minutes at $0.06-0.08/hr | ~$0.01 |
| Pod `v5vlk9khz9hm5n` (RTX PRO 6000): H3 cells only. `FV_CELLS` lacked `upscaler`, so the hook was skipped. 14.5 min | ~$0.50 |
| Pod `z40i7yclzakuse` (RTX PRO 6000): H3 + SeedVR2 1080p/1440p + FlashVSR setup + metrics, 60.5 min | ~$2.11 |
| Pod `r69atksckzoztw` (RTX PRO 6000): SeedVR2 1080p rerun with full frames, stopped at the coordinator's budget stop, 15.8 min | ~$0.55 |
| **Upscaler benchmark total** | **~$3.17** |
| 2026-09-29, gate part 2: pod `9t9awkj9iquyc7` (RTX PRO 6000, EUR-IS-1), 47 min | ~$1.65 |
| 2026-09-29, fal 1080P: pod `bn56zvm3xeppih` (RTX PRO 6000, EUR-IS-1), 8.9 min | ~$0.31 |

All pods were deleted, and a GET after each delete returned 404. The
Runpod balance was $34.49 before the weight pods and $20.26 after the last
GPU pod. The balance is shared: other agents' pods ran at the same time.

## Reproduce

```bash
# needs the image for a commit that contains the hd family
FV_FAMILY=hd FV_LPIPS=1 FV_FETCH_TREE=1 FV_SKIP_TAE=1 FV_FETCH_SKIP='\.cache$' \
  scripts/gpu/runpod-http.sh run <sha>
python3 scripts/gpu/hd_report.py artifacts/runpod/hd/<sha>-<time>

# the upscaler benchmark (weights already on both volumes: verify-weights.sh upscalers);
# the hook runs as the matrix cell `upscaler`, so FV_CELLS must list it
FV_FAMILY=hd FV_CELLS="turbo-768p turbo-1080p upscaler" FV_LPIPS=1 FV_FETCH_TREE=1 FV_SKIP_TAE=1 \
  FV_FETCH_SKIP='\.cache$' FV_POD_PREFIX=fv-upsc \
  FV_POD_FILES="scripts/gpu/hd-upscaler.sh scripts/gpu/hd_upscaler_metrics.py scripts/gpu/hd_flashvsr.py" \
  FV_EXTRA_ENV="FV_HD_POST_URL=file:///fvscratch/files/hd-upscaler.sh FV_HD_POST_TIMEOUT_S=5400 FV_HD_FLASHVSR=1" \
  scripts/gpu/runpod-http.sh run d46ca17
# FV_HD_UP_WARM=0 skips the warm repeat; FV_HD_UP_1440_CLIPS picks the 1440p clips;
# FV_HD_UP_CLIPS the clips overall.
```

The run behind this page used `FV_FETCH_SKIP='frames/[^/]+/frame-[0-9]+\.png$|\.cache$'`.
That pattern also matched `keyframes/`, so the PNG keyframes were not
fetched. The sharpness table above uses frames decoded from the fetched MP4s
(H.264, about 4.5 Mb/s at 1080p) instead. The on-pod `compare-clips` ratios
come from the lossless PNGs.
