# Our stack against sol-engine's published numbers

How fast our prebuilt binary runs each sol-engine model cell, next to the number
NVlabs published for it (NVlabs/Sana, branch `sol-engine`, `models/` at
670482d). This is **not apples to apples**: we run on the GPU the owner's
rules allow and report the hardware next to each number.

## Rules (owner, 2026-10-06)

- **GPU per published number.**
  - Where upstream published RTX 5090 numbers: an RTX 5090 if one can be had, else an RTX PRO 6000.
  - Where upstream published H100 numbers: an H100 if one can be had, else an RTX PRO 6000.
  - Everything else: RTX PRO 6000.
  - No other GPU type is ever rented.
- **Pods.**
  - At most 2 pods in parallel, each with at most 1 hour of benchmark wall clock.
  - Each pod is backstopped at 65 min: a local timer, plus a pod-side self-delete at 64 min.
  - Pods run in EUR-IS-1 with the EU weight volume `jg48s6o1w0` mounted.
- **No compile and no downloads on a pod.**
  - The image is the CI runtime image pinned by digest. The binary was built by CI from main.
  - The pod script comes from this branch: `scripts/gpu/sol-bench-pod.sh`, shipped through `runpod-http.sh FV_POD_SCRIPT`.
  - Weights are read from the volume only. TAE and LPIPS fetches are off (`FV_SKIP_TAE=1`, no `FV_LPIPS`).
- **Budget.** Stop before the Runpod balance would drop below $10. The account is shared with other agents.
- **Methodology.**
  - Warm numbers exclude load, as sol-engine's `benchmark.json` does. Load is reported separately.
  - A `--warm` cell runs one untimed generation first.
  - The two Wan2.1-14B 720p cells run one request after load: a warm-up would double a ~30-min cell.
  - Every cell keeps its `output.mp4` as a sample.

## Phase A run list (planned before renting)

Stock on 2026-10-06 13:2x UTC (GraphQL `lowestPrice`, secure, EUR-IS-1):

- RTX 5090: none.
- H100 (SXM, PCIe, NVL): none.
- RTX PRO 6000 Server: Low stock, $2.09/h.

Both instances therefore run on an **RTX PRO 6000 Blackwell Server (96 GB, sm_120)**.

Image: `ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:76aba91aac3a0c773778f1bd8c8bea386d668fd13a0893f1aa33e668ab7b20d4`. This is `sha-c74c5f5`, equal to `:latest` at the time. Main has since moved to cbff53d, but its crates differ from c74c5f5 only in `crates/fastvideo-serve/tests/edge_dispatch.rs`, and the cbff53d image was still building.

Estimates come from earlier PRO 6000 cell wall times: `artifacts/runpod/rtx5090/a8ecfec-09261537` and `6e651a6-09262349`, and `docs/ports/wan.md`. A cell whose estimate no longer fits the 55-min budget (container uptime) is recorded as skipped. Cells run in this order.

### Instance 1: MiniMax-H3 and LTX-2.5 (published on RTX 5090; ours on PRO 6000)

| # | Cell | Config | Theirs (RTX 5090) | Est. wall |
|---|---|---|---:|---:|
| 1 | `h3-768p-fullopt` | 1344x768, 124 f, 50 steps, `sol-h3-rtx` + TeaCache (= `rtx5090_fullopt.toml`), `--warm` | 231.2 s | 12 min |
| 2 | `ltx25-4k5s-sol-bf16` | distilled two-stage, 4k5s, Sol stage 2, BF16, `--warm` | 273.63 s | 8 min |
| 3 | `ltx25-1080p20s-sol-bf16` | 1080p20s, same | 261.65 s | 7 min |
| 4 | `ltx25-4k5s-sol-nvfp4` | 4k5s, NVFP4 video FFN (`ltx25_distill_sol_nvfp4`) | 171.82 s | 8 min |
| 5 | `ltx25-1080p20s-sol-nvfp4` | 1080p20s, same | 164.40 s | 7 min |
| 6 | `h3-768p-dense` | `rtx5090_dense.toml` (Sol-Attn off, no cache); one request after load | 1045.4 s | 11.5 min |

Boot (image pull, volume mount, tree listing) takes about 5 min. Cells 1–6 total about 54 min, so `h3-768p-dense` is the cell most likely to be skipped; it has the lowest priority.

### Instance 2: Wan (published on GB200; ours on PRO 6000)

| # | Cell | Config | Theirs (1x GB200) | Est. wall |
|---|---|---|---:|---:|
| 1 | `wan5b-base` | TI2V-5B T2V, 704x1280, 121 f, 50 UniPC steps, CFG 5, shift 5, `--warm` | 70.25 s | 8 min |
| 2 | `wan5b-easycache` | + EasyCache 0.036 / retain 7 / cooldown 1 | (cache 1.90x) | 5.5 min |
| 3 | `wan14-720p-fullstack` | T2V-14B, 1280x720, 81 f, 50 steps, CFG 5, shift 5; EasyCache 0.036 + Sol-Attn (`fullstack.toml`) | withdrawn | 15 min |
| 4 | `wan14-720p-base-s15` | the base, **15 of 50 steps**; 50-step denoise = 50/15 of this | withdrawn | 15 min |
| 5 | `wan5b-opt` | EasyCache + PISA 5B (`wan5b_kernel_easycache_pisa.toml`) | 24.35 s fullopt / 28.69 s golden | 5.5 min |

The full 720p base on PRO 6000 is about 30 min. That is about 3.8x the 480p denoise of 477 s, scaling for 2.3x the tokens with quadratic attention. A full base plus fullstack plus 5B would not fit in one hour, so the base is measured at 15 steps and extrapolated. Every base step costs the same: CFG runs every step and no cache is on. Load, text and decode are not scaled.

**Added during the run:** set `a3`. After instance 1's first cell showed MXFP8 linears (our sm_100+ default since c054278), `a3` re-ran the two H3 arms in BF16 as sol-engine's configs specify: `h3-768p-fullopt-bf16` (`--warm`) and `h3-768p-dense-bf16` (one request), both with `--no-text-cache`. It ran on its own pod after instance 1 was deleted, so there were still at most 2 in parallel.

**Cost bound:** 2 pods × 65 min × $2.09/h = **$4.53 at most**. Expected about $4.2.

Balance before renting: $29.57. Other agents were spending $3.27/h at the time.

## Phase A results (2026-10-06)

Every cell ran on an RTX PRO 6000 Blackwell Server (96 GB, sm_120, driver 595.91) in EUR-IS-1. The image was `fastvideo-rs-runtime@sha256:76aba91a…` (build id `109de7dd9ba02772`), with one prompt and seed 1024.

"Ours / theirs" below 1 means our run took less time. The hardware differs in every row, so the ratio is context, not a speedup claim. Raw data:

- `artifacts/runpod/sol-bench/2026-10-06/{a1-h3-ltx,a3-h3-bf16,a2-wan}/<cell>/benchmark.json`
- `results.json` and `RESULTS.md` in the same directory, made with `scripts/gpu/sol_bench_table.py`.

Sample clips: `<cell>/frames/output.mp4`. These are gitignored and kept with the run's scratch clips.

| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| MiniMax-H3 | 768p 124 f 50 st, dense (MXFP8 linears: our sm_100+ default) | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 395.36 | 0.38x | 95.03 / 0.28 / 388.38 / 6.56 | rtx5090_dense.toml is BF16; first request after load; text: cache hit |
| MiniMax-H3 | 768p 124 f 50 st, dense, BF16 | RTX 5090 | 1045.40 | RTX PRO 6000 (sm_120) | 472.11 | 0.45x | 92.25 / 0.48 / 464.94 / 6.52 | first request after load |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache) + MXFP8 linears | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 88.23 | 0.38x | 105.70 / 0.39 / 81.29 / 6.45 | rtx5090_fullopt.toml is BF16; text: cache hit |
| MiniMax-H3 | 768p 124 f 50 st, fullopt (Sol + TeaCache), BF16 | RTX 5090 | 231.20 | RTX PRO 6000 (sm_120) | 110.28 | 0.48x | 193.48 / 0.40 / 103.22 / 6.47 |  |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, BF16 | RTX 5090 | 273.63 | RTX PRO 6000 (sm_120) | 154.86 | 0.57x | 58.71 / 38.49 / 96.43 / 19.63 |  |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, BF16 | RTX 5090 | 261.65 | RTX PRO 6000 (sm_120) | 106.14 | 0.41x | 56.76 / 1.99 / 90.03 / 13.89 | text: cache hit |
| LTX-2.5 distilled | 4K 5 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 171.82 | RTX PRO 6000 (sm_120) | 134.62 | 0.78x | 56.42 / 2.17 / 83.18 / 49.00 | text: cache hit |
| LTX-2.5 distilled | 1080p 20 s, Sol stage 2, NVFP4 video FFN | RTX 5090 | 164.40 | RTX PRO 6000 (sm_120) | 93.28 | 0.57x | 56.36 / 1.96 / 77.19 / 13.89 | text: cache hit |
| Wan2.2 TI2V-5B | 704x1280x121, 50 st, CFG 5, base | 1x GB200 | 70.25 | RTX PRO 6000 (sm_120) | 164.89 | 2.35x | 84.43 / 0.01 / 151.14 / 13.71 | theirs: 5-prompt median; text: cache hit |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 | 1x GB200 | 24.35 | RTX PRO 6000 (sm_120) | 86.21 | 3.54x | 121.30 / 0.01 / 72.45 / 13.71 | theirs: fusion + EasyCache fullopt; text: cache hit |
| Wan2.2 TI2V-5B | same, EasyCache 0.036 + PISA | 1x GB200 | 28.69 | — | — | — | — / — / — / — | theirs: golden kernel+EasyCache+PISA run; not run: exit 124 |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, CFG 5, base (15 st measured, x50/15) | - | — | RTX PRO 6000 (sm_120) | 1797.48 | — | 167.51 / 0.96 / 1788.73 / 7.76 | absolute number withdrawn upstream; extrapolated: denoise x50/15; first request after load |
| Wan2.1 T2V-14B | 1280x720x81, 50 st, EasyCache + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 478.58 | — | 217.91 / 2.07 / 468.73 / 7.75 | absolute number withdrawn upstream; first request after load |

LTX-2.5 Sol stage 2 only (RTX 5090 published vs ours on RTX PRO 6000):

| Cell | Theirs stage 2 (s) | Ours stage 2 (s) | Ours / theirs | Ours stage 1 (s) |
|---|---:|---:|---:|---:|
| ltx25-4k5s-sol-bf16 | 130.32 | 54.18 | 0.42x | 39.45 |
| ltx25-1080p20s-sol-bf16 | 122.27 | 50.61 | 0.41x | 36.70 |
| ltx25-4k5s-sol-nvfp4 | 72.82 | 46.01 | 0.63x | 34.47 |
| ltx25-1080p20s-sol-nvfp4 | 66.78 | 42.73 | 0.64x | 31.76 |

Wan2.1 T2V-14B 720p, our fullstack speedup over base: **3.76x** (sol-engine's withdrawn README headline: ~3.48x).

Wan2.2 TI2V-5B wan5b-easycache speedup over our base: **1.91x** (theirs 2.885x fullopt, 2.45x golden).

### Reading the numbers

- **MiniMax-H3: MXFP8 versus BF16.**
  - Since c054278, our H3 DiT defaults to MXFP8 linears on any sm_100+ GPU (`FASTVIDEO_H3_QUANT` unset), and that includes the PRO 6000. sol-engine's `rtx5090_{dense,fullopt}.toml` are BF16.
  - Instance 1 therefore measured "our default stack" (MXFP8 rows).
  - A follow-up pod, `a3`, re-ran both arms with `FASTVIDEO_H3_QUANT=off` and the prompt encoded in the request (`--no-text-cache`). These are the like-for-like configs: dense **472.1 s** and fullopt **110.3 s**, against 1045.4 / 231.2 s published on RTX 5090.
  - Our dense-to-fullopt speedup is 4.28x (BF16) and 4.48x (MXFP8); theirs is 4.52x.
  - The text encoder is our pre-quantized FP8 Qwen3-VL resident on the card, at about 0.4 s per request.
- **The RTX PRO 6000 is not an RTX 5090.**
  - It has 96 GB and holds every model resident. A 5090 has 32 GB, so sol-engine runs layerwise offload there.
  - The earlier 32 GiB-budget emulation (`artifacts/runpod/rtx5090/6e651a6-09262350`, H3 fullopt 128.69 s) is the closest proxy we have for a real 5090. No 5090 was in stock in EUR-IS-1 on 2026-10-06.
- **LTX-2.5 text.**
  - Only `ltx25-4k5s-sol-bf16` encoded its prompt in the timed request: streamed Gemma, 38.5 s. Every later LTX cell hit the shared prompt cache (about 2 s).
  - sol-engine's E2E includes the encoder. Add about 36.5 s to those rows to compare E2E: 1080p20s BF16 ≈ 142.6 s, 4k5s NVFP4 ≈ 171.1 s, 1080p20s NVFP4 ≈ 129.8 s.
  - The Sol stage-2 table is the clean DiT-only comparison: 0.41–0.42x their time on BF16, 0.63–0.64x on NVFP4.
  - The 4k5s NVFP4 decode (49.0 s against 19.6 s on BF16) was a writer stall (`video_push` 41 s), not VAE compute.
- **Wan2.2 TI2V-5B.**
  - Base is 164.9 s against 70.25 s on one GB200. Our EasyCache 0.036 arm skips 26 of 50 steps: 1.91x over our base, matching their EasyCache factor of 1.90x.
  - Their 24.35 s also includes 1.52x of kernel fusion (`torch.compile`). We have no equivalent of that factor.
  - **EasyCache + PISA (`wan5b-opt`) did not finish.** The `Pisa5b` route (`pisa kernel: route=score sparsity=0.9`) ran at **63.9 s per step**, against about 3.0 s per dense step. It hit the budget stop at step 18/50 (exit 124). That is a performance bug in our PISA score route on sm_120, not a measurement.
- **Wan2.1 T2V-14B 720p.**
  - sol-engine withdrew its absolute number, so only our ratio is reported.
  - Base: 15 steps measured, at 35.8 s per step (536.6 s). Extrapolated ×50/15, the 50-step denoise is 1788.7 s and the E2E is **1797.5 s**.
  - Fullstack (EasyCache 0.036 + Sol-Attn, the `Sol14b` route): **478.6 s**, a **3.76x** speedup (the withdrawn README headline was ~3.48x).
  - Both cells are the first request after load; load (168–218 s) is reported separately.

### What happened on the pods

| Pod | Set | Host | Created → deleted (UTC) | Outcome |
|---|---|---|---|---|
| `da8ynhs5dg5ijd` | a1 | 9o33hyfazu6x | 13:33:43 → 14:10:40 (37 min) | 6/6 cells ok |
| `klo773wcpjthi5` | a2 | s3p8exc9lcvi | 13:34:03 → 13:49:16 | image never started; replaced |
| `zfwa6f1e0rpgrj` | a2 | s3p8exc9lcvi | 13:49:19 → 14:04:37 | same; the next create got HTTP 500 |
| `tuj1fc5f7bewgw` | a2 | s3p8exc9lcvi | 14:05:07 → 14:13:17 | same |
| `pwaljvzz0udwbn` | a2 | s3p8exc9lcvi | 14:13:20 → 14:21:28 | same; the next create got HTTP 500 |
| `72imbc65ufd0pr` | a3 | th4pa8uzi76t | 14:15:50 → 14:33:50 (18 min) | 2/2 cells ok |
| `6w7v3qx39n1ibk` | a2 | s3p8exc9lcvi | 14:22:05 → 14:22:5x | deleted on sight (bad host) |
| `6h46um7sne7l9d` | a2 | bwfzktulij0v | 14:22:54 → 15:24:01 (61 min) | 4/5 cells ok; `wan5b-opt` hit the budget stop |

- **Bad host.** Host `s3p8exc9lcvi` never started this image: there was no runtime after 8–15 min, five times. Runpod kept placing our EUR-IS-1 PRO 6000 pods on it. The fix was a wrapper that deletes a pod placed there immediately.
- **Pod time.** About 164 pod-minutes in all, an upper bound of about **$5.7** at $2.09/h. About 48 of those minutes (about $1.7) were the non-starting pods.
- **Parallelism and wall clock.** No more than 2 pods were up at any time. Every pod's cells ended within 55 min of its container start, and every pod was deleted by its driver. Neither the local 65-min backstop nor the pod-side 64-min self-delete fired.

## Phase B run list (ready; run once each model is merged and its weights are on EU)

The rules are the same as phase A.

- At most 2 pods in parallel, each with ≤ 1 h of wall clock and a 65-min backstop.
- EUR-IS-1 only, with `jg48s6o1w0` mounted.
- The CI runtime image pinned by digest, built from the main that contains the port.
- No compile and no downloads on the pod.
- Stop before the balance would drop below $10.

None of these models has an RTX 5090 or H100 published number, so every cell runs on an **RTX PRO 6000**.

Before each phase-B pod:

1. Check that the weight tree's `verify-weights.sh` cell exists and passes on EU.
2. Add the set (`b1`, `b2`, …) to `scripts/gpu/sol-bench-pod.sh`.
3. Update the estimates from a first-request timing if one exists.

H3 lesson from phase A: check what each model's sm_120 defaults switch on (quantized linears, FP8 text encoders, caches). Pin each arm to sol-engine's config explicitly, for example with `FASTVIDEO_H3_QUANT=off`, and record any extra technique as its own arm.

| Set | Model | Cells (arms) | Config (sol-engine) | Theirs | Est. wall | Prerequisite |
|---|---|---|---|---|---:|---|
| b1 | Wan2.1 T2V-1.3B | `wan13-base`, `wan13-fullstack` (EasyCache + Sol-Attn `Sol13b`) and `wan13-easycache` | 832x480, 81 f, 50 steps, CFG 5 (`models/wan21_t2v_1_3b.toml`), `--warm` | none published (profile only) | ~15 min | base weights `wan21-t2v-1.3b` (5.68 GB transformer) on EU |
| b1 | Wan2.2 T2V-A14B | `a14b-base`, `a14b-opt` (EasyCache + PISA-A14B) | 720x1280, 81 f, 40 steps, dual guidance 4/3, boundary switch; one request after load (no `--warm`) | 1x GB200: 449.67 / 207.01 s | ~40 min | merged A14B memory plan for 2x57 GB on 96 GB (expert swap, streamed offload or FP8); weights 114.3 GB |
| b2 | LTX-2.3 HQ | `ltx23-hq-base`, `ltx23-hq-opt` (SCSP stage-1 cache + PISA stage 2 + midpoint prune) | 1920x1088, 241 f, 15-step res2s stage 1 + 3-sigma stage 2, LoRA 0.25/0.5, `--warm` | ratio only: 2.40x (GB200) | ~35 min | 2.3 loader fix merged; dev DiT (46.15 GB) on EU |
| b2 | SANA-Video 2B | `sana-base`, `sana-easycache` | 832x480, 81 f, 50 steps, `--warm` | ratio only: 2.77x (GB200) | ~10 min | port merged; 14.0 GB weights |
| b3 | LingBot-Video MoE 30B-A3B | `lingbot-base`, `lingbot-opt` (EasyCache + refiner PISA) | base 832x480 → refiner 1920x1088, 121 f, 40 + 8 steps; one request after load | 4x GB200 (CP4): 375.53 / 144.36 s, a different topology | ~45 min | preset fixed to the Hub config; refiner stage; 130 GB weights |
| b4 | Cosmos3-Super 64B | `cosmos3-base`, `cosmos3-opt` (TeaCache 1.15/10/3 + step-selective NVFP4) | 1280x720, 189 f, 35 steps; FP8/NVFP4 weights or streamed offload on 96 GB | 4x GB200: 130.41 s, ratio 2.26x | ~50 min | DiT port; 132.7 GB weights; a memory plan |

Pairing (two pods in parallel):

- (b1 Wan-1.3B + SANA) with (b1 A14B).
- (b2 LTX-2.3) on its own.
- (b3 LingBot) with (b4 Cosmos3).

Each cell writes a `benchmark.json` (`total_s` excludes load) and an `output.mp4`. Generate the table with `scripts/gpu/sol_bench_table.py` after adding each cell's published number to its `PUBLISHED` map. Cost bound: about $2.26 per pod-hour cap.

Launch, one line per pod, from a host that reaches `*.runpod.io` over HTTPS. Look up the current main image digest with the GHCR manifest API first:

```bash
RUNPOD_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:<digest of sha-<main>> \
RUNPOD_VOLUME_ID=jg48s6o1w0 FV_FAMILY=sol-bench-b1 \
FV_POD_FILES=$PWD/scripts/gpu/sol-bench-pod.sh FV_POD_SCRIPT=sol-bench-pod.sh \
FV_EXTRA_ENV="FV_SOL_SET=b1" FV_SKIP_TAE=1 FV_POD_CAP_S=3900 FV_BOOT_WAIT_S=480 \
FV_AVOID_MACHINES="s3p8exc9lcvi lkyvy1sgj4rc" FV_FETCH_TREE=1 FV_FETCH_SKIP='\.png$|/cold/|\.cache$|text-cache|\.wav$' \
  scripts/gpu/runpod-http.sh run <main sha>
```

`FV_FETCH_SKIP` must not match a cell name. Phase A's pattern `cache` dropped the `wan5b-easycache` cell from the fetch; it was restored from a copy taken while the pod was still up.

Then:

```bash
python3 scripts/gpu/sol_bench_table.py artifacts/runpod/sol-bench/<date>/*
```

## Phase B run plan (2026-10-06, planned before renting)

The ports landed on main in #29, #30 and #31, and their weights reached EU in #35. Main is 0f7edc8.

**Image:** `ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:3ef20591e12597380766606fb33242e9e718858a0a29e80d959619ddcdcdd17e`.
- This is `sha-b70f57c`, which is also `:latest`.
- No runtime image was built for 0f7edc8. Nothing between b70f57c and 0f7edc8 touches a path the image workflow builds from: the diff is build-base, build-pod and serve-compat files only.
- The image's `fv-gpucheck` has `sana-video` and `sol` (`lingbot-gen`, `lingbot-router`, `cosmos3-gen`).
- Its `scripts/gpu` has the `solbench` family, `sol/` prompts and `prompts-sol-wan-t2v5.json`.

**GPU:** none of these models has a published RTX 5090 or H100 number, so every cell runs on an **RTX PRO 6000**.

**Pods:** six pods in three pairs, (b1, b2), then (b3, b4), then (b5, b6), so at most 2 are up at once. Each pod has 55 min of cell budget from container start, a 64-min self-delete and a 65-min local backstop. Every launch sets `FV_AVOID_MACHINES=s3p8exc9lcvi`.

**Arguments:** they mirror main's `solbench`, `sana-video`, `sol-lingbot` and `sol-cosmos3` families (`scripts/gpu/sol-bench-pod.sh` sets `b1`–`b6`).

**Precision is pinned to sol-engine's:**
- Wan arms run with `FASTVIDEO_WAN_QUANT=off`, the full Wan VAE and Diffusers UniPC sigmas.
- `FASTVIDEO_FP8` is off except in the extra `cosmos3-teacache-fp8` arm.
- LTX-2.3 fullopt uses NVFP4 because sol-engine's `config/ltx23/fullopt.toml` does.

**Skipped:** the Wan2.2-5B EasyCache + PISA arm (the Pisa5b slowdown found in phase A).

**Prompts:** LingBot and Cosmos3 run one prompt per arm. An arm that does not finish within the budget is reported as incomplete.

Estimates come from `docs/ports/sol-wan-ltx23.md` §"Estimated GPU time", `docs/ports/sana-video.md` and `docs/perf/sol-lingbot-cosmos3-plan.md` §3. The last are FLOP-based, not measured.

| Pod | Cells (in order) | Theirs | Est. cell wall |
|---|---|---|---|
| b1 | `sana-baseline`, `sana-full` (832x480x81, 50 steps, warm) | 2.77x ratio only (GB200) | 15 + 12 min |
|    | `wan13-sol-base`, `wan13-sol-fullstack` (832x480x81, 50 steps, CFG 6, 5 prompts after a warm one) | none published | 12 + 8 min |
| b2 | `a14b-sol-base`, `a14b-sol-fullopt` (720p, 81 f, 40 steps, expert swap; fullopt = EasyCache + PISA, `singlegpu_opt.toml`) | 449.67 / 207.01 s (1x GB200) | 30 + 15–20 min |
| b3 | `ltx23-hq-base`, `ltx23-hq-fullopt` (1920x1088x241 HQ, warm) | 2.40x ratio only (GB200) | 11 + 10 min |
|    | `lingbot-router` (no weights), `lingbot-fullopt` (1 prompt) | 144.36 s (4x GB200) | 1 + 20–25 min |
| b4 | `lingbot-baseline` (1 prompt) | 375.53 s (4x GB200) | ~40 min |
| b5 | `cosmos3-baseline` (1-step warm-up + 1 request) | 130.41 s (4x GB200) | ~40 min |
| b6 | `cosmos3-teacache` (BF16), then `cosmos3-teacache-fp8` (W8A8; theirs is NVFP4 on middle steps) | 2.26x ratio (4x GB200) | ~30 + ~20 min |

**Cost bound:** 6 pods × 65 min × $2.09/h ≈ **$13.6 at most**; expected about $11. The balance before renting was $35.26 (coordinator), with a $10 floor.


## Phase B results (2026-10-06)

Every pod ran on an RTX PRO 6000 Blackwell Server (96 GB, sm_120, driver 595.91, **188 GB container RAM**, 32 vCPU) in EUR-IS-1. The image was `sha256:3ef20591…` (`sha-b70f57c`, the binary for main 0f7edc8).

Raw data:

- `artifacts/runpod/sol-bench/2026-10-06-phaseB/<pod set>/<cell>/`
- `RESULTS.md` and `results.json` in the same directory.
- Sample clips are gitignored.

**Status: four of the twelve planned arms produced a number. Three models are blocked by bugs in the merged ports, not by measurement.**

| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| SANA-Video 2B | 832x480x81, 50 st, cfg 6, baseline | 1x GB200 | — | RTX PRO 6000 (sm_120) | 186.04 | — | 32.65 / 2.58 / 179.59 / 2.25 | theirs: ratio only (2.77x) |
| SANA-Video 2B | same, EasyCache 0.1 + QKV merge + bf16 linear attn | 1x GB200 | — | RTX PRO 6000 (sm_120) | 87.93 | — | 32.31 / 2.59 / 81.53 / 2.17 | theirs: ratio only (2.77x) |
| Wan2.1 T2V-1.3B | 832x480x81, 50 st, CFG 6, base (median of 5 prompts) | - | — | RTX PRO 6000 (sm_120) | 92.22 | — | 115.70 / 0.03 / 89.16 / 3.02 | no published number |
| Wan2.1 T2V-1.3B | same, EasyCache 0.036 + Sol-Attn | - | — | RTX PRO 6000 (sm_120) | 29.09 | — | 118.31 / 0.03 / 26.02 / 3.03 | no published number |
| Wan2.2 T2V-A14B | 1280x720x81, 40 st, CFG 4/3, base (expert swap) | 1x GB200 | 449.67 | — | — | — | — / — / — / — | not run: exit 137 |
| Wan2.2 T2V-A14B | same, EasyCache + PISA | 1x GB200 | 207.01 | — | — | — | — / — / — / — | theirs also: kernel fusion; not run: exit 137 |
| LTX-2.3 HQ | 1920x1088x241, res2s 15 + 3, dense stage 2 | 1x GB200 | — | — | — | — | — / — / — / — | theirs: ratio only (2.40x); not run: exit 2 |
| LTX-2.3 HQ | same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN | 1x GB200 | — | — | — | — | — / — / — / — | theirs: ratio only (2.40x); not run: exit 2 |
| LingBot-Video MoE | same, EasyCache + refiner PISA | 4x GB200 | 144.36 | — | — | — | — / — / — / — | theirs: 4 GPUs (CP4); not run: exit 2 |
| Cosmos3-Super 64B | 1280x720x189, 35 st, CFG 6 | 4x GB200 | 130.41 | — | — | — | — / — / — / — | theirs: 4 GPUs (SP); not run: exit 2 |
| Cosmos3-Super 64B | same, TeaCache + W8A8 FP8 | 4x GB200 | — | RTX PRO 6000 (sm_120) | 800.74 | — | 0.00 / 115.33 / 660.70 / 22.01 | theirs: 2.26x incl. NVFP4 |
| Cosmos3-Super 64B | same, no cache, W8A8 FP8 | 4x GB200 | 130.41 | — | — | — | — / — / — / — | theirs: BF16 on 4 GPUs (SP); not run: exit None |

Optimized-arm speedup over our own baseline (same GPU) vs theirs:

| Pair | Ours | Theirs |
|---|---:|---:|
| sana-baseline → sana-full | 2.12x | 2.77x |
| wan13-sol-base → wan13-sol-fullstack | 3.17x | — |

### Notes on the cells that ran

- **SANA-Video 2B.**
  - Baseline is 186.0 s (179.6 s denoise, 3.59 s per step). The full arm is 87.9 s: 28 of 50 steps reused, so 2.12x against their 2.77x.
  - Their ratio also counts `torch.compile` and the QKV merge. Ours has the merge and bf16 linear attention, but no compile equivalent.
  - The stage writes PNG frames only, with no MP4, so there is no sample clip.
- **Wan2.1 T2V-1.3B.**
  - Base is 92.2 s and fullstack (EasyCache 0.036 + Sol-Attn) is 29.1 s, a 3.17x speedup.
  - Both are medians of sol-engine's 5 prompts after one warm generation. sol-engine publishes no number for this model.
- **Cosmos3-Super 64B, FP8 only.**
  - The BF16 baseline (`cosmos3-baseline`, b5) finished its 1-step warm-up denoise at 45.6 s per step. It then hit **CUDA out of memory in the VAE decode**, with the DiT still resident.
  - W8A8 (`FASTVIDEO_FP8=1`) leaves enough room. **TeaCache + FP8 took 800.7 s per request:**
    - text tower 115.3 s, streamed from the volume each request (234 s in the cold warm-up);
    - denoise 660.7 s (16 of 35 steps computed, about 41.3 s per computed step);
    - decode 22.0 s;
    - load 241.5 s, outside the request.
  - The FP8 no-cache baseline could not finish in the pod budget, so I stopped it to save about 25 min of pod time. Extrapolated from the measured per-step cost: 115 + 35 × 41.3 + 22 ≈ **1582 s**, so TeaCache is about 1.98x on our FP8 path.
  - Theirs: 130.41 s BF16 on 4x GB200 (522 GPU-seconds), 2.26x for TeaCache + NVFP4. Our BF16 arms need the decode OOM fixed first.

### Blocked by bugs in the merged ports

These need code fixes, which can't happen on a pod: no compiling.

| Model | Cells | Failure | Where to look |
|---|---|---|---|
| Wan2.2 T2V-A14B | `a14b-sol-base`, `a14b-sol-fullopt` (b2, and b2r with `FASTVIDEO_PREFETCH=0`) | the process is SIGKILLed (exit 137) about 2.3 min into load, right after `wan moe: Auto -> both`; the container's 188 GB host-RAM limit OOMs during the two-expert load in all four attempts | `wan/pipeline.rs` loads `transformer` and `transformer_2` (57 GB f32 each on disk) with `UMT5` resident; the host staging likely peaks above 188 GB (not profiled) |
| LTX-2.3 HQ | `ltx23-hq-base`, `ltx23-hq-fullopt` (b3) | `error: no text projection in the checkpoint; looked for ["text_embedding_projection.aggregate_embed", "model.diffusion_model.text_embedding_projection.aggregate_embed"]`, 10–37 s in | the HQ path loads the dev single-file DiT, which has no text projection; the `ltx23` tree keeps it under `text_embedding_projection/` |
| LingBot-Video MoE | `lingbot-router`, `lingbot-fullopt` (b3); `lingbot-baseline` (b4) not launched | `DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION)` on the first MoE kernel | `wan/ops.rs` `load_moe_kernels` compiles the MoE kernels to **PTX** with the image's NVRTC 13.4 at run time; the 595.91 driver cannot JIT it. Fix: build them as cubins (AOT, like `kernels.cu`) or have NVRTC emit SASS for the device arch |

### Pods

All phase B pods were created and deleted by this run (UTC):

| Pod | Set | Host | Created → deleted | Outcome |
|---|---|---|---|---|
| `ouieg4qm26n2dr` | b2 | lkyvy1sgj4rc | 18:40:33 → 18:50:40 | image never started; replaced |
| `t9qemie0pqwhwy` | b2 | lkyvy1sgj4rc | 18:50:44 → 18:56:42 | same; driver stopped, pod deleted |
| `xt8o6n3gis0qtg`, `nknq6xzmayolqs` | b2 | lkyvy1sgj4rc | 18:57:05 / 18:57:53 | deleted on sight (`FV_AVOID_MACHINES`) |
| `kpv8c8q6xtorj2` | b2 | — | 18:58:39 → 19:08:15 | A14B OOM-killed at load (both arms) |
| `ny2eovsjco93oc` | b1 | qlfbxqg3mhpi | 18:51:37 → 19:29:05 | 4/4 ok; the container took about 9 min to start |
| `zfo2ztsbuo3a37` | b3 | th4pa8uzi76t | 19:08:43 → 19:13:48 | LTX-2.3 loader error; LingBot PTX error |
| `iqxss6zul6d9xb` | b5 | th4pa8uzi76t | 19:14:25 → 19:19:55 | Cosmos3 BF16 decode OOM |
| `0ybcuq5gwmlutq` | b6 | — | 19:23:00 → 19:55:12 | Cosmos3 TeaCache FP8 ok; FP8 baseline stopped (could not fit) |
| `9ox00xsl775stb` | b2r | — | 19:29:12 → 19:36:20 | A14B OOM again with prefetch off |

- **Pod time:** about 114 pod-minutes, so about **$4.0** at $2.09/h.
- **Parallelism and wall clock:** never more than 2 pods at once. Every pod was deleted within 38 min of its creation (b1 37.5 min, b6 32 min, the rest under 11 min).
- **Balance:** $35.26 before, $31.03 after.
- **Bad hosts:** host `lkyvy1sgj4rc` behaved like `s3p8exc9lcvi` (no container start after 10 min). Both are in `FV_AVOID_MACHINES` for future runs.
- **Result mirroring:** `scripts/gpu/sol_bench_mirror.py` copied each finished cell while its pod was still up. b6's results come from that copy, because the driver was stopped before its own fetch.

## Phase B2 (unblocked ports): run plan (2026-10-06, planned before renting)

The four ports that blocked phase B are fixed on `fix/phaseb-blocked-ports`:

| Block | Root cause | Fix |
|---|---|---|
| Wan2.2 T2V-A14B, exit 137 at load | `WanPipeline::load_with_dit` read `transformer/` and `transformer_2/` with the **eager** loader, which copies every tensor of a directory into host memory (2 × 57 GB f32), and held the first copy for the whole load while UMT5 (21 GB f32) and the second expert loaded. With the mapped shards that passes the 188 GB container limit. | A two-expert checkpoint opens its experts with the lazy store (`WeightMap::open`): tensors stay in the mapped shards and go to the device one at a time as bf16, the bits the eager path uploads. The load logs its host peak (`VmHWM`). `FASTVIDEO_WAN_DIT_LAZY=0/1` overrides. |
| LTX-2.3 HQ, `no text projection` | `TextConnectors::load` always probed for LTX-2.0's **shared** `text_embedding_projection.aggregate_embed`. 2.3 has per-modality projections; the dev single file holds them as `text_embedding_projection.{video,audio}_aggregate_embed` (manifest `ltx23_single_file.json`), so the probe failed although the weights were there. | The probe runs only for shared-projection configs. |
| LingBot MoE, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | `load_moe_kernels` NVRTC-compiled the MoE region of `kernels.cu` to PTX at run time; the image's NVRTC 13.4 writes PTX newer than driver 595.91 can JIT. | `wan::nvrtc_sass`: NVRTC emits SASS (`nvrtcGetCUBIN`) for the device's own SM, PTX only as the fallback. The other run-time NVRTC paths (kernels.cu, attn_fp8, attn_sage, attn_dc fallbacks) go through it too. |
| Cosmos3-Super BF16, decode OOM | The 62.4 GB bf16 gen tower stayed resident through the Wan 2.2 decode, with the pool still holding the denoise activations. | `FASTVIDEO_COSMOS3_GEN=auto` parks the bf16 gen tower in pinned host memory on a 96 GB card (the A14B swap's whole-ring offload). Before the decode the pipeline drops the text K/V and TeaCache residuals, releases the tower and trims the pool. The next request brings the tower back behind the first step's compute; that copy is inside the request time. |

**Image:** the CI runtime image for the branch head, pinned by digest (recorded in the results).

**GPU and region:** RTX PRO 6000 only, EUR-IS-1, the EU volume `jg48s6o1w0`. `FV_AVOID_MACHINES="s3p8exc9lcvi lkyvy1sgj4rc"`.

**Pins:** as phase B. Wan: `FASTVIDEO_WAN_QUANT=off`, Diffusers UniPC sigmas, the full VAE. `FASTVIDEO_FP8` off everywhere (Cosmos3 is BF16 this time). LTX-2.3 fullopt keeps NVFP4, as sol-engine's `fullopt.toml` does.

**Pods:** at most 2 up at once. Each has 55 min of cell budget, a 64-min self-delete and a 65-min local backstop; a cell whose estimate no longer fits is recorded as skipped.

| Pod set | Cells (in order) | Theirs | Est. cell wall |
|---|---|---|---|
| c1 | `a14b-sol-base`, `a14b-sol-fullopt` | 449.67 / 207.01 s (1x GB200) | 30 + 15–20 min |
| c2 | `ltx23-hq-base`, `ltx23-hq-fullopt`, `lingbot-router`, `lingbot-fullopt` | 2.40x ratio; 144.36 s (4x GB200) | 11 + 10 + 1 + 20–25 min |
| c3 | `lingbot-baseline` | 375.53 s (4x GB200) | ~40 min |
| c4 | `cosmos3-baseline` (BF16), then `cosmos3-teacache` (BF16) if it fits | 130.41 s (4x GB200), 2.26x | ~40 + ~25 min |

Order: (c1, c2) together, then (c3, c4).

**Money:** the balance was $30.55 before renting, with a $10 floor; it is checked before each pod. Four pods at ≤ 65 min × $2.09/h is $9.1 at most; expected about $6. The owner's aim is about $5, so c4's TeaCache arm is the first thing dropped, and a pod whose cells are done is deleted at once.

## Phase B2 results (2026-10-06)

Every pod was an RTX PRO 6000 Blackwell Server (96 GB, sm_120, driver 595.91, 188 GB container RAM) in EUR-IS-1. The images were the CI runtime images of `fix/phaseb-blocked-ports`:

| Image | Commit | Used by |
|---|---|---|
| `sha256:f4b14b07…` | ad94673 (the four fixes) | c1, c2, c3 |
| `sha256:da9e0244…` | a5e3675 (+ LTX-2.3 text-cache key) | c4 |
| `sha256:15d60d38…` | 679dfda (+ A14B swap headroom, LingBot pool trims) | c6 |

Raw data:

- `artifacts/runpod/sol-bench/2026-10-06-phaseB2/<pod set>/<cell>/` (each set's `driver.log` beside it).
- `RESULTS.md` and `results.json` in the same directory.
- Clips are gitignored.

**Status: three of the four blocked models now produce a number (A14B base, LTX-2.3 HQ base, Cosmos3 BF16 baseline).** LingBot gets past its PTX block and through the whole base stage, then runs out of memory entering the 1080p refiner; that fix is on the branch but has not been on a GPU. Both PISA arms that reached their sparse attention (LTX-2.3 fullopt stage 2, A14B fullopt) hit a slow PISA score-route kernel on sm_120 and were stopped.

| Model | Config | Their HW | Theirs (s) | Our HW | Ours (s) | Ours / theirs | Ours: load / text / denoise / decode (s) | Notes |
|---|---|---|---:|---|---:|---:|---|---|
| Wan2.2 T2V-A14B | 1280x720x81, 40 st, CFG 4/3, base (expert swap) | 1x GB200 | 449.67 | RTX PRO 6000 (sm_120) | 1441.74 | 3.21x | 314.66 / 1.15 / 1432.79 / 7.77 | first request after load; c6 |
| Wan2.2 T2V-A14B | same, EasyCache + PISA | 1x GB200 | 207.01 | — | — | — | — | not run: c1 stopped it (PISA route, and the decode OOM fixed later) |
| LTX-2.3 HQ | 1920x1088x241, res2s 15 + 3, dense stage 2 | 1x GB200 | — | RTX PRO 6000 (sm_120) | 213.39 | — | 15.41 / 49.03 / 153.11 / 9.87 | theirs: ratio only (2.40x); text: cache miss (streamed Gemma); c4 |
| LTX-2.3 HQ | same, SCSP + PISA s2 + midpoint prune + NVFP4 FFN | 1x GB200 | — | — | — | — | — | not run: stage-2 PISA step 1 over 10 min (dense: 13.3 s) |
| LingBot-Video MoE | 480p 121 f 40 st + 1080p refiner 8 st, 1 prompt | 4x GB200 | 375.53 | — | — | — | — | base stage done (40 steps, 514.6 s); OOM entering the refiner |
| LingBot-Video MoE | same, EasyCache + refiner PISA | 4x GB200 | 144.36 | — | — | — | — | base stage done (19 of 40 steps reused, 271.5 s); OOM entering the refiner |
| Cosmos3-Super 64B | 1280x720x189, 35 st, CFG 6, BF16 | 4x GB200 | 130.41 | RTX PRO 6000 (sm_120) | 1608.56 | 12.33x | 0.00 / 1.55 / 1584.09 / 21.97 | theirs: 4 GPUs (SP), 522 GPU-s; gen tower parked; c4 |
| Cosmos3-Super 64B | same, TeaCache 1.15/10/3 (BF16) | 4x GB200 | — | — | — | — | — | skipped by the budget gate (1500 s estimate, 815 s left) |

### Notes on the cells that ran

- **Wan2.2 T2V-A14B base, 1441.7 s (theirs 449.67 s on one GB200, 3.21x).**
  - The lazy expert load peaks at 59.2 GiB of host RSS with both experts on the device (c1). With the swap it is 111.2 GiB, which includes 52.3 GiB of pinned expert copies (c6). Phase B was killed above 188 GB.
  - c1 kept both experts resident (`auto`, 24 GiB headroom): 37.5 s per step, then the full-VAE decode ran out of memory with 32.5 GiB left. With a 36 GiB headroom `auto` swaps the experts on this card. c6 ran 35.8 s per step, one expert swap at t=865, and a 7.8 s decode.
  - Load is 314.7 s, outside the request, as in phase A's 14B cells. The prompt went through UMT5 in the request (1.15 s).
- **LTX-2.3 HQ base, 213.4 s.**
  - text 49.0 s (Gemma streamed, cache miss), stage 1 110.7 s (15 res2s steps at 7.5 s), stage 2 42.4 s (3 steps at 13.3 s), decode + mp4 9.6 s; peak 73.8 GiB allocated.
  - c2's first try ran the whole warm-up and then failed the timed request on the text-cache key (`text_encoder/` has its shards under `gemma/`), fixed in a5e3675.
  - sol-engine publishes only the 2.40x fullopt ratio, so there is no ratio without our fullopt.
- **Cosmos3-Super BF16 baseline, 1608.6 s (theirs 130.41 s on 4x GB200).**
  - The gen tower parks in pinned host memory before the decode and comes back behind the next request's first step: step 1 is 46.3 s against 45.25 s for the rest.
  - The decode (22.0 s) no longer runs out of memory.
  - BF16 is 45.3 s per step against 41.3 s for phase B's W8A8.
  - The text tower took 1.55 s because its weights were in the page cache after the warm-up (122 s cold).
- **LingBot-Video MoE.**
  - `lingbot-router` passes (0 differing expert sets, max weight error 0.0), and the MoE kernels load as `nvrtc sass sm_120`.
  - Both arms then finish the 480p base stage and write its clip. Baseline: 40 steps at 12.8 s, 514.6 s. Fullopt: EasyCache reuses 19 of 40 steps, 271.5 s, 1.90x on the base stage.
  - Both runs then hit CUDA out of memory entering the 1080p refiner. Under `swap` the base DiT was freed into the pool and the refiner loaded straight after, and the 1080p VAE encode ran with the refiner resident.
  - 679dfda trims the pool after each model leaves and encodes before the refiner loads. That is type-checked and unit-tested, not GPU-tested.

### Still blocked

| Cells | Error | Where |
|---|---|---|
| `lingbot-baseline`, `lingbot-fullopt` | `DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` entering the refiner (ad94673) | fix in 679dfda (`lingbot/pipeline.rs`), not yet run on a GPU |
| `ltx23-hq-fullopt`, `a14b-sol-fullopt` | no error: the stage-2 / sparse-step PISA attention (`pisa kernel: route=score sparsity=0.9 block=64`, 63 240 and 75 600 tokens) took over 10 min for one LTX step that is 13.3 s dense | `wan::ops::pisa_attn_device` on sm_120: the same slowdown as phase A's `wan5b-opt` (63.9 s per step) |
| `cosmos3-teacache` (BF16) | not run, budget | fix-free: needs a pod of its own (~25 min) |

### Pods

All phase B2 pods were created and deleted by this run (UTC):

| Pod | Set | Created → deleted | Minutes | Outcome |
|---|---|---|---:|---|
| `5kfyyhtm4m1ush` | c1 | 21:13:13 → 21:56:10 | 43.0 | A14B load fixed; base OOMed in the decode; deleted during fullopt |
| `kvh9t13pz5tcqo` | c2 | 21:13:15 → 21:34 | 20.8 | LTX-2.3 projection fixed; text-cache key error; fullopt stuck in PISA, deleted |
| `i1nhh8dyc7ib4o` | c3 | 21:34:37 → 22:13:45 | 39.1 | LingBot router ok; both arms OOM at the refiner |
| `yesv8pqin8xfl8` | c4 | 21:58:40 → 22:47:35 | 48.9 | LTX-2.3 HQ base ok; Cosmos3 BF16 baseline ok; TeaCache skipped |
| `zxrdpi7oiqwxz3` | c6 | 22:24:28 → 23:00:14 | 35.8 | A14B base ok |

- **Pod time:** 187.6 pod-minutes at $2.09/h = **$6.53**, over the ~$5 aim. The overrun went on two reruns that new bugs made necessary (c4's LTX, c6's A14B).
- **Parallelism and wall clock:** never more than 2 pods at once; every pod was deleted within 49 min of creation. No backstop fired.
- **Balance:** $30.55 before (20:55), $22.23 after (23:01). Other sessions' pods were also running in that window, so the drop is not all this run.
- **Hosts:** no pod landed on `s3p8exc9lcvi` or `lkyvy1sgj4rc`.

## Phase B3 (the remaining blocked cells): run plan (2026-10-07, planned before renting)

Phase B2 left five cells blocked: the three PISA arms (`wan5b-opt`, `a14b-sol-fullopt`, `ltx23-hq-fullopt`), both LingBot arms, and the Cosmos3 BF16 TeaCache arm (budget).

**Two images.**
- Main 3c66c10 (`sha256:fedcc8b5…`, also `:latest`) has the LingBot refiner memory fix (679dfda: pool trims, encode before the refiner loads). It runs the cells that need no code change (sets `d1`, `d2`).
- The CI runtime image of `fix/pisa-sm120-phaseb3` runs the PISA arms and LingBot fullopt, whose refiner uses the same PISA kernel (sets `e1`, `e2`). Recorded by digest in the results.

**GPU, region, pins:** as phase B2. RTX PRO 6000 only, EUR-IS-1, EU volume `jg48s6o1w0`, `FV_AVOID_MACHINES="s3p8exc9lcvi lkyvy1sgj4rc"`. Wan arms pin `FASTVIDEO_WAN_QUANT=off`, Diffusers UniPC sigmas and the full VAE (phase A's `wan5b-opt` did not; B3's does). `FASTVIDEO_FP8` off. LTX-2.3 fullopt keeps NVFP4 (sol-engine `fullopt.toml`).

**LingBot baseline does not fit one pod.** Phase B2 measured the base stage at 12.8 s per step (40 steps, 514.6 s) after about 550 s of load and text. The 1080p refiner runs 8 steps × CFG at 253k tokens with dense attention: at the base stage's attention throughput that is about 2.5k s, so load + base + refiner is about 3.6k s, over the 55-min cell budget. The baseline therefore runs the refiner with `--refiner-steps 1`, which gives 3 of the 8 official sigmas (0.85 plus the 2 tail steps). Every baseline refiner step costs the same (dense, CFG every step, no cache), so `sol_bench_table.py` scales the refiner denoise by 8/3, as phase A did for `wan14-720p-base-s15`. The cell is `lingbot-baseline-rs3`.

| Pod set | Image | Cells (in order) | Theirs | Est. cell wall |
|---|---|---|---|---|
| d1 | main 3c66c10 | `lingbot-router`, `lingbot-baseline-rs3` (base 40 st + refiner 3 of 8 st, ×8/3) | 375.53 s (4x GB200) | 1 + ~40 min |
| d2 | main 3c66c10 | `cosmos3-teacache` (BF16, TeaCache 1.15/10/3) | 2.26x ratio (4x GB200) | ~25 min |
| e1 | PISA branch | `pisa-parity` (device PISA vs the host oracle), `wan5b-opt` (EasyCache + PISA 5B), `lingbot-fullopt` (EasyCache + refiner PISA) | 28.69 s (1x GB200); 144.36 s (4x GB200) | 1 + 8 + ~35 min |
| e2 | PISA branch | `ltx23-hq-fullopt` (SCSP + PISA s2 + prune + NVFP4), `a14b-sol-fullopt` (EasyCache + PISA) | 2.40x ratio; 207.01 s (1x GB200) | 12 + ~20 min |

Order: d1 and d2 one after the other while the PISA fix is built and tested; then e1 and e2 (at most 2 pods of this run at once; another agent may run one more).

**Pods:** each has 55 min of cell budget from container start, a 64-min self-delete and a 65-min local backstop; a cell whose estimate no longer fits is recorded as skipped.

**Money:** the balance was $20.97 before renting (2026-10-07), shared with another agent, with an $8 floor; it is checked before each pod. Four pods at ≤ 65 min × $2.09/h is $9.1 at most; expected about $6–7. If a pod would take the balance near the floor, it is not started and the cell is reported as blocked.
