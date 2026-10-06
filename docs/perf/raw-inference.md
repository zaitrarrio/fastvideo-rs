# Raw generation time: every model at its smallest size (RTX PRO 6000)

Run date: 2026-09-29/30. How long does one clip take on the GPU alone? Each
model ran at its smallest serving size, straight from the CLI
(`fv-gpucheck`), with no fv-serve, HTTP or queue. Everything was resident on
the device and the text was encoded before timing. No quality was measured.

**No H100 or H200 could be rented.** From 23:43 to 00:49Z the harness
retried H100 SXM, NVL and PCIe in US-CA-2 every 5 min, and from 00:36Z also
H200 and H200 NVL in US-CA-2 and EUR-IS-1 every 4 min. Every create
answered "could not find any pods with required specifications". The only
fresh numbers are therefore from the RTX PRO 6000. The older H100 / H200
figures at the end are from earlier runs, at other sizes and through other
paths.

Headline (RTX PRO 6000, seconds of GPU time per 5 s clip, text excluded):

| Model (smallest size) | default recipe | max speed |
|---|---:|---:|
| H3 turbo 480p | 9.84 | 7.08 |
| H3 max 480p | 8.87 | 6.43 |
| LTX-2.5 turbo 720p | 10.04 | 7.38 |
| Wan 2.2 TI2V-5B turbo 480p | 7.11 | 1.49 |
| FastWan 1.3B 480p | 4.85 | 2.26 |
| SF-Wan 1.3B stream 480p | 15.2 fps (5.27 s per 5 s) | 15.2 fps |

## Setup

| | |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-3341394` = `@sha256:52d89026c58105fcb302a0ab618f3a5a0996333b750625f1dca1cec7b06e5491`. The `gpucheck-runtime-image` build of main `3341394`, green; fv-gpucheck build id `13a7b623b38af048`. No image was built for this run. |
| Build flags | The workspace release profile is fat LTO, `codegen-units = 1`, `opt-level = 3`. The gpucheck image overrides it: `docker/gpucheck.Dockerfile` sets `CARGO_PROFILE_RELEASE_LTO=off` and `CODEGEN_UNITS=16` (`opt-level` stays 3). This changes only host code. The GPU kernels are precompiled per arch: NVRTC cubins (`kernels=Cubin(120)` / `Cubin(90)` in the log) and embedded cuTile/oxide cubins for sm_100/sm_120. Rust flags were not tuned. |
| RTX PRO 6000 | Pod `5zbsx3omtnotmv`, RTX PRO 6000 Blackwell Server Edition (sm_120, 97 887 MiB), driver 595.91.07, EUR-IS-1 secure, $2.09/hr. Host: 256 vCPU, 1.5 TB RAM. Weights on the EU volume `jg48s6o1w0`. |
| H100 / H200 | Not run: out of stock (see above). |
| Weights | Read from `/workspace/weights` on the network volume; nothing on the volumes was written. All outputs, text caches and AdaLN caches went to the container disk. After the run, `find -newer` over the used trees came back empty. |
| Protocol | One process per model and arm: load, one untimed warm-up generation (`--warm`), then **5 timed generations** of a 5-prompt set (`--prompts`, all five the same prompt and seed 7). Each cell below gives the median (min–max) of those 5. |
| What is timed | **gen** = the pipeline's `total_s` minus `text_s`. `total_s` already leaves out the PNG writes. What remains: refine + denoise + decode (video and audio) + host glue. **No H.264/MP4** (`--no-mp4`). PNG frames are written after the timed region (`png_frames_s`, not counted). They could not be turned off: the H3 stage opens a frame for its sanity check. **Text** was encoded once. In timed runs it is a cache hit (0.03–0.26 s, subtracted). The one exception is LTX run 1, which encodes again after the uncached warm-up (4.7 s, subtracted). |
| Peak memory | Device memory used (`nvidia-smi`, 0.5 s samples, whole process), so it includes every resident weight. For LTX the timed-run peak is also given; it is lower because the process peak includes Gemma, which is dropped after encoding (see residency). |
| s per s of video | gen median ÷ clip length (H3 124 f / 24 fps = 5.17 s; LTX and Wan 5B 121 f / 24 = 5.04 s; FastWan 1.3B 81 f / 16 = 5.06 s). |

Sizes: H3 480p exists for both recipes (832x480, 124 frames, 5.17 s).
LTX-2.5's smallest tier is 720p: 1280x720, generated at **1280x768** (two-stage
canvas multiple of 64, padded and cropped as the server does), 121 frames at
24 fps. The Wan models ran at 832x480: TI2V-5B 121 frames at 24 fps, FastWan
1.3B 81 frames at 16 fps. SF-Wan streamed 832x480 at 16 fps.

### Arms and flags

| Model | Default (serving recipe) | Max-speed (no quality gate) |
|---|---|---|
| H3 turbo | `--techniques h3/fasth3_4step_vsa --h3-recipe 4step-vsa` (FastH3 Preview LoRA, VSA 0.9, MXFP8 linears, bf16 act), official ViT VAE | + `--taeh3-weights taeh3.safetensors`; two arms on one load (`--arm`): `tae` (the serving profile) and `tae-fp8attn` (`h3/fasth3_4step_vsa_fp8attn`, FP8 Q·Kᵀ in the VSA fine stage) |
| H3 max | `--techniques h3/sol_h3_4step_engine_ladder --h3-recipe sol-h3` (Sol engine route, tau 1.0/1.25/1.5, MXFP8) | + TAEH3 |
| LTX-2.5 turbo | `--techniques ltx2/ltx25_distill_sol --model-version 2.5 --two-stage` (8 + 3 steps, Sol stage 2), conv VAE, audio on | sm_120: `ltx2/ltx25_distill_sol_nvfp4` (the ltx-draft profile, NVFP4 video FFN) + `--ltx-tae-weights taeltx2_3_wide`; sm_90: `ltx2/ltx25_distill_sol_fp8` (W8A8 FP8; NVFP4 is Blackwell-only) + the same TAE |
| Wan 2.2 TI2V-5B turbo | `--preset fast_wan_2_2_ti2v_5b --steps 3 --flow-shift 5 --fps 24`, `FASTVIDEO_WAN_VAE=full` | `FASTVIDEO_WAN_VAE=taehv` (taew2_2) + `FASTVIDEO_WAN_QUANT=mxfp8` (sm_120) / `w8a8` (sm_90) |
| FastWan 1.3B | `--vsa wan gen` (fast_wan_t2v_480p, DMD 3 steps), `FASTVIDEO_WAN_VAE=full` | TAEHV (taew2_1) + the same FP8 switch |
| SF-Wan 1.3B | `wan stream --run r,seconds=10,rope=rebased,sink=3,sheet=0` (TAEHV each block, CUDA graphs) | + the same FP8 switch |

Common to every arm: `--mode fast`, `--dit-offload resident`. H3 adds
`--text-encoder resident-fp8`; LTX adds `--text resident --offload none`.
Attention is the per-arch default, which is already the fastest kernel. On
sm_120, dense SDPA times cuDNN against our `flash_mma_fwd2` per shape (log:
`sdpa auto (sm_12x) … -> flash_mma_fwd2`) and VSA runs the TMA-ring mma
kernel. On sm_90, the dc wgmma kernels (`fa_dc90_fwd_d128`, `fa_dc90_vsa`)
are the default by rule. The script is `artifacts/perf/raw-inference/bench.sh`.

### Residency (from the logs)

- **H3** (all four processes): `text_encoder chosen: resident-fp8, 22.72 GiB on device`,
  `h3 dit: 50 blocks resident`, `dit_residency: resident`. The per-phase ledger
  keeps `text_encoder 22.72 | text_refiner 1.49 | dit 23.87 | vae 4.52 | audio_vae 0.24`
  GiB live through every phase of every timed run.
- **LTX-2.5**: `ltx2 dit: 48 blocks resident` and `ltx2 text: resident, 20.3 GiB on
  device`. The DiT, conv VAE / TAE and connectors stay resident. Two exceptions,
  both built into the pipeline and not selectable from the CLI:
  1. After encoding, the pipeline **drops Gemma** (`dropped resident Gemma
     (contexts already encoded)`). This does not touch the timed numbers,
     because every timed run hits the text cache. Run 1 misses and reloads
     Gemma inside `text_s`, which is subtracted.
  2. The **latent upsampler** (1.85 GiB) is placed on the device only
     around its call (`upsampler 0.00 (max 1.85)` in the ledger). The
     upsample step takes 0.49–0.55 s per timed run, load included. That
     is the one piece of weight traffic inside the timed region.
- **Wan 5B / 1.3B / SF-Wan**: `resident=true`; the UMT5, DiT and decoder are
  loaded once. Timed runs are text-cache hits.

## RTX PRO 6000 (sm_120)

| Model | Size | Arm | gen median (min–max) s | denoise s | decode s | peak mem GiB | s compute / s video |
|---|---|---|---:|---:|---:|---:|---:|
| H3 turbo (FastH3 4-step VSA) | 832x480x124 | default | **9.84** (9.75–9.85) | 7.07 | 2.71 | 60.3 | 1.90 |
| | | max: TAEH3 | **7.35** (7.33–7.37) | 7.04 | 0.24 | 57.6 | 1.42 |
| | | max: TAEH3 + FP8 attn | **7.08** (7.08–7.09) | 6.78 | 0.24 | 57.6 | 1.37 |
| H3 max (Sol-H3 4-step) | 832x480x124 | default | **8.87** (8.82–8.89) | 6.13 | 2.69 | 54.7 | 1.72 |
| | | max: TAEH3 | **6.43** (6.40–6.45) | 6.14 | 0.23 | 52.0 | 1.24 |
| LTX-2.5 turbo (two-stage, Sol) | 1280x768x121 (720p) | default | **10.04** (9.69–10.08) | 8.53 | 1.46 | 66.0 (timed 43.3) | 1.99 |
| | | max: NVFP4 FFN + TAE | **7.38** (7.25–7.50) | 6.92 | 0.40 | 55.9 (timed 34.9) | 1.46 |
| Wan 2.2 TI2V-5B turbo | 832x480x121 | default | **7.11** (7.10–7.12) | 1.55 | 5.54 | 38.2 | 1.41 |
| | | max: TAEHV + MXFP8 | **1.49** (1.49–2.97) | 1.38 | 0.10 | 21.4 | 0.30 |
| FastWan 1.3B | 832x480x81 | default | **4.85** (4.85–4.86) | 1.92 | 2.91 | 27.1 | 0.96 |
| | | max: TAEHV + MXFP8 | **2.26** (2.26–2.29) | 1.99 | 0.26 | 21.5 | 0.45 |

Decode includes the audio decode for H3 (0.06 s) and LTX. For H3, denoise +
decode leave about 0.05 s of refine and glue out of gen. For LTX, denoise
covers stage 1, upsample and stage 2.

**SF-Wan 1.3B streaming, 832x480, 16 fps** (five 10 s rollouts after one warm-up):

| Arm | steady fps median (min–max) | s per 5 s of video | block p50 (12 frames) | TTFF | peak mem GiB |
|---|---:|---:|---:|---:|---:|
| default (TAEHV, bf16) | **15.18** (15.14–15.24) | 5.27 | 0.782 s | 0.43 s | 26.7 |
| max: + MXFP8 | **15.16** (15.12–15.18) | 5.28 | 0.780 s | 0.42 s | 26.1 |

On sm_120, SF-Wan runs about 5 % short of real time. This matches the 15.22
fps measured before (`docs/serve/e2e/wan.md`).

## H100 / H200: older figures only (not this run)

These come from earlier documents, on other images, at other sizes, and
mostly through fv-serve (x264 encode and job overhead included). They are
here for scale only and do not compare cell for cell with the table above.

| GPU | Workload | Figure | Source (date) |
|---|---|---|---|
| H100 80GB HBM3 (SXM) | H3 turbo 768p, CLI `h3 gen`, dc VSA | text + denoise + decode 21.50 s (denoise 16.34 s) | `datacenter-profile.md` (2026-09-29, `sha-522af72`) |
| H100 SXM | H3 turbo 768p, fv-serve job | run 28.48 s | `datacenter-profile.md` (2026-09-29) |
| H100 SXM | H3 max (Sol-H3) 768p, fv-serve job | run 23.65 s | `datacenter-profile.md` (2026-09-29) |
| H100 SXM | LTX-2.5 1080p 6 s, fv-serve job | run 28.16 s | `datacenter-profile.md` (2026-09-29) |
| H100 SXM | Wan 2.2 TI2V-5B turbo 704p, fv-serve job | run 12.83 s | `datacenter-profile.md` (2026-09-29) |
| H100 SXM | SF-Wan 1.3B `wan stream`, 832x480 | 23.8 frames/s (block p50 ≈ 0.51 s) → ≈ 3.4 s per 5 s of video | `datacenter-profile.md`, `docs/serve/e2e/wan.md` |
| H200 | FastH3 4-step VSA 768p (warm suite, TAEH3) | generate 35.0 s (7.0 s/step) | `docs/gaps/2026-09-24-phase3-vs-published.md` (2026-09-24, pre-dc kernels) |


## Notes

- **The decoder dominates Wan 5B.** Its full Wan 2.2 VAE takes 5.5 s of the
  7.1 s gen at 480p; TAEHV takes 0.10 s, which puts the model at 0.30 s of
  compute per second of video on the PRO 6000. On H3 and LTX the TAE saves
  2.5 s and 1.1 s.
- **At 480p, H3 max (Sol-H3) beats H3 turbo** by 0.9 s of denoise on the
  PRO 6000. At this small grid the VSA gather/gate overhead costs more than it
  saves. The serving tiers were tuned at 768p.
- **FP8 does not help everywhere.** MXFP8 made no difference for SF-Wan, cost
  3 % for FastWan 1.3B denoise (1.92 → 1.99 s, as `docs/ports/wan.md` found),
  and saved 11 % for Wan 5B denoise. The FP8 Q·Kᵀ VSA arm saved 4 % of H3 turbo
  denoise; its profile is marked as failing the kernel parity tolerance, and
  no quality was checked here.
- **Outliers.** One Wan 5B max run spent 1.58 s in the TAEHV decode against
  the usual 0.10 s (`taehv.decode done in 1539ms`). It shows as the 2.97 s max;
  the median is unaffected. LTX default run 1 had a text cache miss (the warm-up
  bypasses the cache); its 4.66 s encode is subtracted.
- **SF-Wan graphs.** A 10 s rollout replays CUDA graphs for only 4 of its 14
  blocks; graphs are captured per block position. The fps still matches the
  60 s figure from earlier runs.
- **Load times** (not timed, for reference; PRO 6000, EU volume, warm page
  cache after the first H3 load): H3 117–139 s, LTX 55–66 s, Wan 5B 55–74 s,
  FastWan 1.3B 37–68 s.
- **Not run:** every H100 / H200 cell (no capacity in US-CA-2 or EUR-IS-1 for 66 min). Everything planned on the RTX PRO 6000 ran; no job failed.

## Spend

| Pod | GPU | Lifetime | Cost |
|---|---|---:|---:|
| `5zbsx3omtnotmv` | RTX PRO 6000 (EUR-IS-1) | 1836 s | $1.07 |
| H100 / H200 retries | — (every create refused) | — | $0 |

Balance: $9.99 at the start, $8.03 at 00:49Z. The difference includes other
agents' and the build pod's spend over the same 66 min (≈ $0.60/hr standing).

Guards:

- the local balance watchdog (delete my pods below $3.50);
- a local wall-clock backstop (4500 s), plus an on-pod `sleep 4400; DELETE`;
- the on-pod idle guard (`FV_E2E_IDLE_DELETE_MIN=15`).

Every pod was deleted and checked with a 404 on GET. Only pods created by
this run were touched.

## Artifacts

`artifacts/perf/raw-inference/`:

- `bench.sh`: the on-pod driver;
- `summarize.py`: per-job medians from each `benchmark.json`;
- `<gpu>-summary.json`: per job and arm, the 5-run median/min/max, text, peaks
  and load;
- `<gpu>-loglines.txt`: the residency, technique and kernel lines per job;
- `progress.log`: the running log.
