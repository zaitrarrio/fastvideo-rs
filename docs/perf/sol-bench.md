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
FV_AVOID_MACHINES=s3p8exc9lcvi FV_FETCH_TREE=1 FV_FETCH_SKIP='\.png$|/cold/|\.cache$|text-cache|\.wav$' \
  scripts/gpu/runpod-http.sh run <main sha>
```

`FV_FETCH_SKIP` must not match a cell name. Phase A's pattern `cache` dropped the `wan5b-easycache` cell from the fetch; it was restored from a copy taken while the pod was still up.

Then:

```bash
python3 scripts/gpu/sol_bench_table.py artifacts/runpod/sol-bench/<date>/*
```
