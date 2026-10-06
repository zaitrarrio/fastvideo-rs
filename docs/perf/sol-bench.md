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

**Cost bound:** 2 pods × 65 min × $2.09/h = **$4.53 at most**. Expected about $4.2.

Balance before renting: $29.57. Other agents were spending $3.27/h at the time.
