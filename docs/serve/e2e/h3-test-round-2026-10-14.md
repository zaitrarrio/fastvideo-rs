# H3 test round + Cloudflare stream outputs (GPU check, 2026-10-14)

Owner request (2026-10-14): (A) a standard test round of every H3 model and recipe the repo serves, T2V plus the
image paths each one offers, at 480p and 768p, 5 s, with the standard prompts and seeds, compared against the last
recorded numbers (flag regressions over 5%); (B) the Cloudflare part of the GPU check that was skipped on
2026-10-13: stream outputs to the real Cloudflare Stream from a GPU-served model (WHIP H.264 / VP8 / VP9, RTMPS,
SRT) and the director `x_fv_output` copy with audio. Artifacts: `artifacts/serve/e2e/h3-round-2026-10-14/` (15 MB).

## Summary

| check | result |
|---|---|
| Code | main at `dcace95` (PR #99 merged during the run) / `33caa17`. Pod A (T2V): serve image `@sha256:17ef12ef…` (label `revision=dcace95`, binaries `tools-v0.1.17` = `3570bc2`: Trim, Turbo LoRAs, NCII guard, stream outputs). Pods B, C (serve jobs, Cloudflare): `@sha256:b98084e9…` (`revision=33caa17`, `tools-v0.1.18` = `dcace95`, **includes #99's copy-audio fix**) |
| A. T2V, 9 cells x 2 prompts x 480p/768p, 124 frames | **every cell ran** (base 49 forwards, h3-turbo, h3-max, h3-draft, FastH3 V2 dense, Trim MXFP8 and bf16, larry 6, lightx2v 4): 34 clips, all completed |
| A. Regressions | **none over 5% where the comparison is like for like**: the four recipes re-run at the Turbo LoRA round's 107 frames are 3.2% faster to 3.4% slower (denoise and total, all four prompts); base 480p -3.2%; turbo / max 768p within ±2.2% of the 2026-09-27 medians; Trim within 2% on `spark-mountain-lake`. Three ">5%" deltas are prompt effects, not regressions (below) |
| A. I2V / FL2VA / Ref2V (fv-serve) | **25 of 25 jobs succeeded**: I2V and FL2VA at 480p and 768p on h3-turbo, h3-max, larry, lightx2v, V2; h3-draft at 480p; Ref2V turbo 480p/768p; Ref2V max 480p. I2V turbo +2.6-3.0% vs 2026-09-28, Ref2V turbo 768p +0.3% vs its port run |
| A. Quality gate vs the 49-forward base | only **V2 dense at 768p passes** (both prompts). Every few-step recipe fails the `lossy` gate's sharpness / jitter windows against the base (LPIPS 0.47-0.73: different samples, as recorded for these recipes before) |
| B. Cloudflare, native streams (h3-turbo, 1344x768) | **all five kinds play**: WHIP H.264 / VP8 / VP9 at **1344x768** over WHEP in 29.7-30.8 s from the POST (first clip build ~26 s); RTMPS in 41.1-41.7 s and SRT in 63.7-87.3 s as HLS, top rendition **1260x720** (4 renditions); see [audio](#audio) |
| B. Director `x_fv_output` copy (PR #99 merged) | **video and audio**: WHEP 1344x768 H.264, 940 frames in 40 s, first frame 3.3 s after `output_ready`; decoded Opus has **real audio** (-30.7 to -41 dBFS RMS) exactly while each chunk plays, silence while the next chunk is late (`freeze_video_and_silence_audio_until_ready`) |
| B. Audio | **every kind delivers the model's audio**: decoded WHEP Opus up to -11.8 dBFS, RTMPS / SRT HLS AAC-LC 48 kHz stereo renditions at -23.9 / -26.6 dB mean (pod C: streams with queued clips, read from the start); the director copy's decoded audio -30.7 to -41 dBFS while chunks play |
| B. Clean-up | 15 `fv-test-` live inputs created, **15 verified gone (GET 404)**, 7 RTMPS/SRT recordings deleted; final listing: no `fv-test-` input, 1 other input (the pre-existing `divine-firefly-3665`, untouched) |
| Cost | GPU pods **$5.39** (A 3 438 s, B 3 550 s, C 805 s, at $2.49/hr); the shared build pod for two example builds; Cloudflare well under $0.10. All pods deleted and checked gone (404) |

## Setup

- **Pods** (Runpod SECURE, RTX PRO 6000 Blackwell **Server** Edition, 97 887 MiB, driver 595.91.07, $2.49/hr), each
  checked with `nvidia-smi` right after boot (0 MiB used on all three):
  - A `icpj4bitaiufgb` (T2V matrix, fv-gpucheck): EU-RO-1 had no RTX PRO 6000 capacity, so **R2 mode**: every tree on
    the container disk, 327 GB in 535 s (611 MB/s), 0 retries. 380 GB disk.
  - B `4nep0hs3ifiqt9` (fv-serve: image jobs, Cloudflare): R2 mode again, 380 GB in 482 s (790 MB/s).
  - C `vv1awkcuxvqjh4` (Cloudflare audio re-check): EU-RO-1 with the EU volume `bz0ofkfx45` mounted read only
    (symlinks only, nothing written).
- **Harness**: `scripts/serve/e2e/pod.sh` (fv-serve on the fake config at boot + the e2e sidecar; a detached
  wall-clock backstop per pod; a 15-20 min GPU-idle self-delete) and the on-pod driver
  `scripts/serve/e2e/h3-round.sh` (new). `pod.sh` gained `FV_E2E_SECRETS_ALLOW`: with `FV_E2E_NO_SECRETS=1`, only the
  named Runpod secret references stay in the pod env. Pod A had **no secret at all**; pods B and C had exactly
  `FV_CF_STREAM_API_TOKEN = {{ RUNPOD_SECRET_fv_cf_stream_api_token }}` and
  `FV_CF_ACCOUNT_ID = {{ RUNPOD_SECRET_fv_cf_account_id }}` (owner approval 2026-10-14), plus
  `FV_CF_STREAM_NAME_PREFIX=fv-test-`. The token was never read into the coordinating container, printed or logged
  (the pod check printed only `token-resolved-len-53`).
- **Clients** (built on the shared build pod, own worktree and target dir): the new
  `crates/fastvideo-serve/examples/cf_live_check.rs` (the `outputs_live` test's Cloudflare case over HTTP against a
  running fv-serve: WHEP read-back with a str0m peer, HLS segments, live-input clean-up limited to `fv-test-` inputs
  this run created) and `director_client`. `scripts/serve/e2e/whep_audio.py` (new, aiortc) decodes the WHEP audio.
- **T2V** (pod A): `fv-gpucheck h3 gen --prompts`, one warm process per cell (`--warm`: an untimed 480p generation
  first), 124 frames (5.17 s), text encoder streamed with a shared conditioning cache (as the 2026-10-13 Turbo LoRA
  round), each cell's served profile; prompts `h3-demo` (seed 0) and `spark-mountain-lake` (seed 42) of
  `scripts/gpu/prompts-eval.json` at 832x480 and 1344x768. Quality: `fv-gpucheck compare-clips` (LPIPS alex on the
  GPU, PSNR, sharpness and temporal-jitter ratios) of every clip against the base's clip of the same prompt and size;
  verdict by `scripts/gpu/gate-policy.toml` (`lossy`: frame counts equal, sharpness 0.95-1.08, jitter 0.85-1.20,
  LPIPS available).
- **Image paths** (pod B): fv-serve with each model as its shipped config defines it (`[[models]]` of
  `configs/serve/runpod-*.toml`, warmup off, one untimed 480p T2V job first), `POST /fv/v1/jobs`, seed 7, 124 frames;
  I2V: `scripts/gpu/fixtures/ti2v-beach-832x480.jpg` with the 2026-09-28 I2V prompt; FL2VA: that image first,
  `ti2v-beach-zoom-832x480.jpg` last; Ref2V: the beach image as the one reference with `scripts/serve/e2e/ref2v.py`'s
  prompt. `short_edge` 480 / 768 (canvas follows the 832x480 image: 832x480 / 1344x768). Text encoder `auto`
  (FP8 resident).

## A. T2V matrix (fv-gpucheck, 124 frames)

Load is the pipeline load in the warm process (weights from the container disk, page cache warm after the first
cell); denoise (median seconds per forward); decode = video VAE + mp4 + audio; generate = the whole request, text
cached; peak = the engine's device peak (`peak_mib`). Gate columns are against **base** (the first rows).

| model | prompt | load s | denoise s (s/fwd) | decode s | generate s | peak GiB | LPIPS | PSNR dB | sharp. | jitter | gate vs base |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base (49 fwd, dense) | h3-demo-480p | 14.9 | 102.52 (2.09) | 2.87 | 105.82 | 30.7 | - | - | - | - | (reference) |
|  | spark-mountain-lake-480p |  | 97.42 (1.99) | 2.84 | 104.48 | 30.1 | - | - | - | - | (reference) |
|  | h3-demo-768p |  | 404.08 (8.24) | 6.61 | 411.32 | 34.0 | - | - | - | - | (reference) |
|  | spark-mountain-lake-768p |  | 396.46 (8.09) | 6.56 | 403.58 | 34.6 | - | - | - | - | (reference) |
| h3-turbo (FastH3 4-step VSA) | h3-demo-480p | 10.1 | 7.79 (1.96) | 2.88 | 11.14 | 36.5 | 0.693 | 10.8 | 0.99 | 1.59 | fail (jitter) |
|  | spark-mountain-lake-480p |  | 7.18 (1.80) | 2.87 | 10.49 | 35.9 | 0.673 | 11.3 | 1.29 | 1.22 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 20.16 (5.05) | 6.59 | 27.40 | 43.3 | 0.693 | 9.6 | 1.03 | 2.58 | fail (jitter) |
|  | spark-mountain-lake-768p |  | 19.40 (4.86) | 6.55 | 26.50 | 43.2 | 0.616 | 13.1 | 1.31 | 1.95 | fail (sharpness, jitter) |
| h3-max (Sol-H3 4-step, tau ladder) | h3-demo-480p | 10.4 | 6.73 (1.57) | 2.84 | 10.05 | 30.3 | 0.619 | 11.8 | 1.23 | 2.24 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 6.33 (1.47) | 2.85 | 9.62 | 30.0 | 0.732 | 11.0 | 1.47 | 2.22 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 21.91 (4.68) | 6.56 | 29.11 | 34.4 | 0.644 | 10.6 | 1.60 | 2.33 | fail (sharpness, jitter) |
|  | spark-mountain-lake-768p |  | 21.33 (4.53) | 6.55 | 28.41 | 34.4 | 0.640 | 13.8 | 1.39 | 2.25 | fail (sharpness, jitter) |
| h3-draft (turbo + TAEH3, 480p tier only) | h3-demo-480p | 3.4 | 7.81 (1.96) | 0.53 | 8.86 | 32.8 | 0.696 | 10.9 | 0.87 | 1.47 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 7.25 (1.82) | 0.53 | 8.29 | 32.7 | 0.662 | 11.5 | 1.09 | 1.11 | fail (sharpness) |
| FastH3 V2 (8-step, dense, as served) | h3-demo-480p | 19.6 | 17.99 (2.25) | 2.85 | 21.34 | 33.8 | 0.497 | 13.2 | 1.02 | 0.79 | fail (jitter) |
|  | spark-mountain-lake-480p |  | 17.06 (2.13) | 2.85 | 20.39 | 33.5 | 0.538 | 14.6 | 0.70 | 0.80 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 68.88 (8.61) | 6.57 | 76.09 | 37.7 | 0.466 | 13.5 | 1.06 | 1.14 | **pass** |
|  | spark-mountain-lake-768p |  | 67.68 (8.46) | 6.57 | 74.81 | 37.5 | 0.479 | 16.0 | 0.97 | 1.00 | **pass** |
| FastH3 Trim, MXFP8 (served default) | h3-demo-480p | 11.2 | 14.16 (1.77) | 2.16 | 16.83 | 31.3 | 0.658 | 11.1 | 1.30 | 1.92 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 13.26 (1.66) | 2.14 | 15.87 | 31.2 | 0.691 | 11.7 | 1.43 | 1.92 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 39.60 (4.95) | 4.90 | 45.13 | 38.6 | 0.674 | 10.1 | 1.43 | 2.84 | fail (sharpness, jitter) |
|  | spark-mountain-lake-768p |  | 38.21 (4.77) | 4.86 | 43.68 | 39.6 | 0.639 | 13.7 | 1.42 | 2.58 | fail (sharpness, jitter) |
| FastH3 Trim, bf16 linears | h3-demo-480p | 7.1 | 18.13 (2.27) | 2.17 | 20.80 | 44.7 | 0.664 | 11.0 | 1.26 | 2.05 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 17.22 (2.16) | 2.13 | 19.83 | 43.9 | 0.690 | 11.3 | 1.34 | 1.95 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 49.96 (6.25) | 7.42 | 58.06 | 51.4 | 0.672 | 10.6 | 1.54 | 3.00 | fail (sharpness, jitter) |
|  | spark-mountain-lake-768p |  | 48.34 (6.05) | 4.86 | 53.82 | 51.3 | 0.650 | 13.5 | 1.47 | 2.68 | fail (sharpness, jitter) |
| larry-turbo (6 fwd) | h3-demo-480p | 16.3 | 12.39 (2.06) | 2.87 | 15.73 | 30.3 | 0.520 | 12.5 | 1.21 | 0.78 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 11.92 (2.00) | 2.88 | 15.28 | 30.5 | 0.513 | 15.1 | 0.88 | 0.78 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 49.39 (8.23) | 6.58 | 56.61 | 34.4 | 0.546 | 11.4 | 1.07 | 0.77 | fail (jitter) |
|  | spark-mountain-lake-768p |  | 48.52 (8.09) | 6.58 | 55.67 | 34.4 | 0.467 | 16.4 | 1.19 | 1.23 | fail (sharpness, jitter) |
| lightx2v-turbo (4 fwd) | h3-demo-480p | 13.8 | 8.25 (2.06) | 2.85 | 11.56 | 30.5 | 0.558 | 12.5 | 1.13 | 1.31 | fail (sharpness, jitter) |
|  | spark-mountain-lake-480p |  | 7.90 (1.97) | 2.86 | 11.22 | 30.7 | 0.510 | 15.3 | 0.81 | 0.74 | fail (sharpness, jitter) |
|  | h3-demo-768p |  | 32.86 (8.21) | 6.56 | 40.04 | 34.6 | 0.598 | 11.2 | 1.19 | 1.45 | fail (sharpness, jitter) |
|  | spark-mountain-lake-768p |  | 32.29 (8.07) | 6.54 | 39.38 | 34.6 | 0.504 | 16.5 | 0.86 | 0.84 | fail (sharpness, jitter) |

Frames: `samples/t2v-sheet-768p.jpg` (2.5 s into `spark-mountain-lake`; top row base, turbo, max, larry; bottom row
lightx2v, Trim MXFP8, V2, draft at 480p); clips `samples/larry-spark-mountain-lake-768p.mp4`,
`samples/trim-mx-spark-mountain-lake-768p.mp4`.

**Reading the gate.** The profiles record their gates against each recipe's own baseline (the h3attn arms against
FastH3 8-step / 4-step VSA / Sol-H3 4-step, docs/techniques.md "H3 attention arms"); against the 49-forward base,
a distilled few-step model is a different sample of the same prompt (LPIPS 0.47-0.73, PSNR 9.6-16.5 dB), so the
`lossy` gate's ratio windows judge style rather than damage. What the ratios say: h3-turbo, h3-max and the Trim are
**sharper and much busier** than the base (sharpness up to 1.60, temporal jitter 1.9-3.0x); larry and lightx2v
are closer to the base's motion (jitter 0.74-1.45) and the closest in LPIPS (0.47-0.60); **V2 dense is the closest
to the base** and passes at 768p on both prompts (LPIPS 0.47-0.48, sharpness 0.97-1.06, jitter 1.00-1.14). The
draft's TAEH3 decode is the softest (sharpness 0.87 on `h3-demo`).

**Trim MXFP8 vs bf16, same pod**: MXFP8 is 1.28x faster per forward at 480p (1.66-1.77 vs 2.16-2.27 s) and 1.26x at
768p (4.77-4.95 vs 6.05-6.25 s), and uses 13 GiB less at 768p, confirming 2026-10-13's recommendation (keep MXFP8).

### Against the last recorded numbers

**Like for like** (same pod type, same prompts and seeds, same 107-frame canvas as the 2026-10-13 Turbo LoRA round,
whose 5 s runs used 107 frames by mistake; `t2v/<cell>-f107/`):

| model | prompt | denoise now / then | delta | total now / then | delta | peak MiB now / then |
|---|---|---:|---:|---:|---:|---|
| larry-turbo | h3-demo-480p | 10.05 / 10.32 | -2.5% | 13.00 / 13.13 | -1.0% | 30948 / 30240 |
|  | spark-mountain-lake-480p | 9.68 / 9.89 | -2.1% | 12.62 / 17.00 | -25.8% (then: a cold text encode) | 30884 / 30144 |
|  | h3-demo-768p | 38.79 / 39.91 | -2.8% | 45.06 / 46.12 | -2.3% | 34380 / 33600 |
|  | spark-mountain-lake-768p | 38.20 / 39.34 | -2.9% | 44.41 / 45.52 | -2.4% | 34316 / 33600 |
| lightx2v-turbo | h3-demo-480p | 6.74 / 6.96 | -3.1% | 9.67 / 9.81 | -1.5% | 31108 / 29984 |
|  | spark-mountain-lake-480p | 6.56 / 6.65 | -1.4% | 9.46 / 9.51 | -0.5% | 31044 / 29888 |
|  | h3-demo-768p | 25.87 / 26.60 | -2.7% | 32.13 / 32.80 | -2.0% | 34604 / 33536 |
|  | spark-mountain-lake-768p | 25.43 / 26.24 | -3.1% | 31.64 / 32.40 | -2.3% | 34156 / 33472 |
| h3-turbo | h3-demo-480p | 6.56 / 6.57 | -0.2% | 9.51 / 9.46 | +0.4% | 36164 / 36064 |
|  | spark-mountain-lake-480p | 6.18 / 6.17 | +0.1% | 9.11 / 9.01 | +1.1% | 36036 / 35904 |
|  | h3-demo-768p | 16.56 / 16.82 | -1.5% | 22.90 / 23.05 | -0.7% | 42340 / 41888 |
|  | spark-mountain-lake-768p | 15.96 / 16.36 | -2.5% | 22.20 / 22.62 | -1.9% | 43172 / 41760 |
| h3-max | h3-demo-480p | 5.95 / 5.77 | +3.1% | 8.90 / 8.61 | +3.4% | 30771 / 30176 |
|  | spark-mountain-lake-480p | 5.32 / 5.48 | -3.0% | 8.25 / 8.33 | -1.0% | 30308 / 30144 |
|  | h3-demo-768p | 17.76 / 18.27 | -2.8% | 24.10 / 24.49 | -1.6% | 34252 / 34144 |
|  | spark-mountain-lake-768p | 17.20 / 17.77 | -3.2% | 23.48 / 23.95 | -1.9% | 34220 / 34080 |

Peak memory is 0.3-3.4% higher everywhere (the "then" column is the benchmark's `peak_memory_mb`, "now" the
per-prompt `peak_mib`; same allocator, same canvas): under the 5% bar.

**Other records** (124 frames; different harness or prompt, so indicative only):

| model | now | last recorded | delta | note |
|---|---|---|---:|---|
| base, h3-demo 480p | denoise 102.52 s | 105.9 s (2026-10-13 Turbo LoRA round, same prompt / seed / 124 frames) | -3.2% | like for like |
| h3-turbo 768p | denoise 20.16 / 19.40 s | 19.81 s (2026-09-27 h3attn, 5-prompt median) | +1.8% / -2.1% | |
| h3-max 768p | denoise 21.91 / 21.33 s | 21.80 s (2026-09-27 h3attn) | +0.5% / -2.2% | |
| Trim MXFP8 480p / 768p | denoise 14.16, 13.26 / 39.60, 38.21 s | 13.00 / 37.95 s (2026-10-13 GPU check, fv-serve job, another prompt, DiT streamed) | **+8.9%**, +2.0% / +4.3%, +0.7% | `h3-demo` only: see below |
| Trim bf16 480p / 768p | 18.13, 17.22 / 49.96, 48.34 s | 17.12 / 48.28 s (same) | **+5.9%**, +0.6% / +3.5%, +0.1% | `h3-demo` only |
| h3-draft 480p | generate 8.86 / 8.29 s | 8.1 s (catalog note, prompt not recorded) | **+9.4%** / +2.3% | `h3-demo` only |
| FastH3 V2 dense | 17.06-17.99 s (480p), 67.7-68.9 s (768p) | none: the only V2 record is the VSA recipe (24.40 s at 480p, a Workstation pod) | - | first record of the served dense recipe |
| h3-turbo I2V (fv-serve) | denoise 8.03 / 22.17 s | inference 7.8 / 21.6 s (2026-09-28, fal queue, same image / prompt / seed) | +3.0% / +2.6% | |
| Ref2V turbo 768p | denoise 33.71 s | 33.6 s (docs/ports/h3-ref2v.md, same image) | +0.3% | |

**The three ">5%" rows are the prompt, not a regression.** `h3-demo` is a long structured prompt (more text tokens
in every joint-attention call); it costs 5-7% more per forward than `spark-mountain-lake` in **every** cell at 480p
(base 2.09 vs 1.99 s, turbo 1.96 vs 1.80, Trim 1.77 vs 1.66), and the comparison rows were recorded with a short
prompt. On `spark-mountain-lake` the same cells are within +2.3%. Nothing was flagged as a real regression.

## A. Image paths through fv-serve (I2V, FL2VA, Ref2V)

`serve ready` is fv-serve start to `/ping 200` (load from the container disk / page cache, warmup off). Wall is the
client's submit-to-terminal time on the pod; peak is the job's `peak_memory_mb` (the FP8 text encoder and the vision
tower are resident, hence ~53-66 GiB).

| model | job | canvas | wall s | text s | denoise s | video decode s | run s | peak GiB | serve ready s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| h3-turbo | i2v-480 | 832x480x124 | 12.7 | 0.88 | 8.03 | 2.78 | 12.63 | 58.8 | 18.2 |
|  | i2v-768 | 1344x768x124 | 31.2 | 1.15 | 22.17 | 6.61 | 31.00 | 65.6 | |
|  | fl2va-480 | 832x480x124 | 14.3 | 1.33 | 8.80 | 2.81 | 13.97 | 59.1 | |
|  | fl2va-768 | 1344x768x124 | 34.8 | 1.31 | 24.89 | 6.64 | 34.35 | 66.0 | |
| h3-max | i2v-480 | 832x480x124 | 11.7 | 0.85 | 7.00 | 2.79 | 11.49 | 52.9 | 17.2 |
|  | i2v-768 | 1344x768x124 | 33.3 | 1.14 | 24.05 | 6.63 | 32.95 | 56.5 | |
|  | fl2va-480 | 832x480x124 | 13.8 | 2.11 | 7.69 | 2.80 | 13.66 | 53.1 | |
|  | fl2va-768 | 1344x768x124 | 37.0 | 2.36 | 26.24 | 6.56 | 36.73 | 56.7 | |
| larry-turbo | i2v-480 | 832x480x124 | 18.0 | 1.35 | 12.90 | 2.79 | 17.88 | 52.5 | 28.4 |
|  | i2v-768 | 1344x768x124 | 61.2 | 1.24 | 52.22 | 6.59 | 61.17 | 55.4 | |
|  | fl2va-480 | 832x480x124 | 19.6 | 1.34 | 13.83 | 2.82 | 18.99 | 52.6 | |
|  | fl2va-768 | 1344x768x124 | 65.0 | 1.37 | 55.35 | 6.66 | 64.80 | 55.6 | |
| lightx2v-turbo | i2v-480 | 832x480x124 | 13.2 | 0.88 | 8.58 | 2.80 | 13.09 | 52.5 | 30.3 |
|  | i2v-768 | 1344x768x124 | 43.4 | 0.74 | 34.78 | 6.60 | 43.19 | 55.4 | |
|  | fl2va-480 | 832x480x124 | 14.8 | 1.40 | 9.22 | 2.84 | 14.59 | 52.6 | |
|  | fl2va-768 | 1344x768x124 | 46.5 | 1.35 | 36.91 | 6.65 | 46.34 | 55.6 | |
| FastH3 V2 dense | i2v-480 | 832x480x124 | 23.3 | 0.84 | 18.45 | 2.79 | 23.10 | 56.3 | 34.4 |
|  | i2v-768 | 1344x768x124 | 81.8 | 1.07 | 72.84 | 6.61 | 81.65 | 59.5 | |
|  | fl2va-480 | 832x480x124 | 27.0 | 2.87 | 19.77 | 2.82 | 26.47 | 56.4 | |
|  | fl2va-768 | 1344x768x124 | 88.7 | 2.74 | 77.35 | 6.65 | 88.19 | 59.7 | |
| h3-draft | i2v-480 | 832x480x124 | 10.6 | 0.86 | 8.03 | 0.70 | 10.49 | 55.0 | 12.2 |
|  | fl2va-480 | 832x480x124 | 13.8 | 2.84 | 8.91 | 0.74 | 13.52 | 55.2 | |
| Ref2V turbo | ref2v-480 | 832x480x124 | 14.3 | 1.15 | 8.56 | 2.77 | 13.85 | 52.5 | 33.4 |
|  | ref2v-768 | 1344x768x124 | 42.8 | 0.80 | 33.71 | 6.59 | 42.33 | 55.3 | |
| Ref2V max (49 fwd) | ref2v-480 | 832x480x124 | 315.6 | 45.65 | 262.01 | 2.83 | 315.50 | 54.6 | 31.4 |

- Every model listed as image-capable in the catalog (`caps.rs`: all H3 entries but the Trim, T2V only, and the
  Ref2V pair, Ref2V only) served both I2V and FL2VA at both tiers; the LoRA recipes' FL2VA path ran on a GPU for the
  first time (docs/ports/h3-turbo-loras.md §9 listed it as untested). Sample: `samples/larry-fl2va-768.mp4`.
- The I2V and FL2VA jobs were not gated: there is no 49-forward image-conditioned reference in this round.

## B. Cloudflare Stream (live), from a GPU-served model

fv-serve on **h3-turbo** (`fasth3`, 1344x768 @ 24 fps, audio on), the Cloudflare default output (live inputs
`fv-test-<stream id>`), one native stream per kind (`POST /fv/v1/streams`, one prompt), read back from Cloudflare by
`cf_live_check` for 60 s. "Ready" is from the POST to the first decoded WHEP frame / the first HLS segment that
decodes; it includes the first clip's build (~26 s: an H3 clip stream shows nothing before its first clip).

| case | ready s | Cloudflare saw ingest | delivered (decoded) | frames decoded | encoder (fv-serve) | audio | input |
|---|---:|---:|---|---:|---|---|---|
| WHIP H.264 | 29.7 | - | WHEP **H.264 1344x768** | 918 in 40 s (39 keyframes) | `Nvenc` | Opus track, 1982 packets | deleted on stop (404) |
| WHIP VP8 | 30.2 | - | WHEP **VP8 1344x768** | 944 (40) | `libvpx` | Opus, 1986 packets | deleted on stop |
| WHIP VP9 | 30.8 | - | WHEP **VP9 1344x768** | 944 (40) | `libvpx-vp9` | Opus, 1983 packets | deleted on stop |
| RTMPS | 41.1 / 41.7 / 41.1 | 31.0-31.9 s | HLS H.264 **1260x720** top of 1260x720, 840x480, 630x360, 420x240 (DASH the same) | 144 in 3 segments | `ffmpeg h264_nvenc` | AAC-LC 48 kHz stereo rendition (`group_audio`), listed 57-58 s after the POST | disabled on stop, recording + input deleted (404) |
| SRT | 63.7 / 87.3 | 31.5-31.7 s | HLS H.264 **1260x720** top (4 renditions) | 96 in 2 segments | `ffmpeg h264_nvenc` | AAC-LC 48 kHz stereo rendition, listed 100 s after the POST | same |
| Director `x_fv_output` copy (WHIP) | 3.3 after `output_ready` (31.2 after session open) | - | WHEP **H.264 1344x768** | 940 in 40 s | `Nvenc` | **real audio** (decoded below) | deleted when the session ended (404) |

No picture was black (mean luma 84.5-90.1, deviation 48-57). SRT's HLS takes much longer than RTMPS (as in the
fake-engine runs: 68-72 s), and Cloudflare labels its SRT renditions 30 fps against 24 for RTMPS.
`samples/director-copy-frame200.jpg` is a frame of the copy as Cloudflare delivered it.

### Audio

A native clip stream with one prompt plays its single 5 s clip once and then **holds the last frame in silence**
until the stream ends; so a read that starts after that clip hears nothing. The first reads above were all late:
the WHEP reads began as the clip started (small Opus packets: 13-21 bytes on average), and the HLS audio renditions
became listable only after 57-100 s, long after the clip: `volumedetect` on 10 s and 60 s of the RTMPS / SRT audio
rendition read -91 dB (digital silence), and a decoded WHEP read that started 30 s after the POST heard 60 s of
silence. Pod C re-checked with streams that queue 8 clips and read from the start (below).

- **Director copy** (PR #99): `whep_audio.py` decoded the copy's Opus from Cloudflare for 60 s: RMS -30.7 /
  -32.7 dBFS in seconds 1-2 and -32.3 to -41.1 dBFS in seconds 31-36, silence in between: exactly the chunk
  playback pattern of this session (chunks built in 26-34 s, 5 s each, the gap frozen and silenced by the
  director's `deadline_missed` behaviour). The main WebRTC peer's own audio of the same session: mean -42.9 dB,
  max -17.9 dB. **The copy carries the session's audio** (2026-10-13: no audio track at all).

**Pod C: streams that keep playing, read from the start** (`h3-round.sh cf-audio2`; each stream queues 8 clips, so
a new 5 s clip plays every ~26 s; `artifacts/…/cloudflare/cf-audio2/`):

| kind | what was read | audio delivered by Cloudflare |
|---|---|---|
| WHIP H.264 (WHEP decoded by aiortc, 100 s from the first answer) | 4 919 Opus frames, 513 video frames at 1344x768 | **real audio**: 20 s above -60 dBFS RMS, loudest second -11.8 dBFS (silence while the next clip builds) |
| RTMPS (HLS audio rendition, 60 s from when it was listed, 60 s after the POST) | AAC-LC 48 kHz stereo | **real audio**: mean -23.9 dB, max -1.4 dB |
| SRT (HLS audio rendition, listed 162 s after the POST) | AAC-LC 48 kHz stereo, 129 kb/s | **real audio**: mean -26.6 dB, max -6.2 dB |

So **every kind carries the model's audio to Cloudflare**; pod B's -91 dB readings were reads of the silent hold
after a single clip. (fv-serve's counters: WHIP sent 5 064 Opus packets with 2 429 frames; the ffmpeg sinks report
`audio_packets_sent` 0, see findings.)

## What was skipped, and why

- **Ref2V max at 768p**: 49 forwards with the reference tokens took 681 s of denoise on 2026-10-04; its 480p run
  (262 s denoise, 45.6 s text) is here. Not worth ~$0.6 of a shared budget.
- **1080p** (h3-turbo / h3-max tier), 10-15 s clips, other step counts or strengths of the LoRAs: outside the
  standard round.
- **Quality gate of the image-conditioned jobs**: no base reference for them.
- **The 107-frame like-for-like rerun** covers the four recipes whose last numbers were measured at 107 frames; the
  Trim, V2, draft and base comparisons use other records (above).
- **Runpod EU-RO-1 capacity**: pods A and B could not get an RTX PRO 6000 next to the EU volume and fetched their
  trees from R2 (as `weights_source r2` does), which also makes their load times not comparable with volume runs.

## Findings

1. **No timing or memory regression over 5%** in any like-for-like comparison (16 prompt/size points at 107
   frames, base at 124 frames).
2. **All image paths work on every image-capable model**, including FL2VA on both Turbo LoRAs (first GPU run).
3. **Cloudflare: every kind plays from a GPU-served model**; WHIP keeps the native 1344x768, RTMPS/SRT come back as
   a 1260x720-topped ladder; the **director copy now carries audio** (#99 verified on a GPU).
4. **Cloudflare HLS audio is a separate rendition** (`#EXT-X-MEDIA TYPE=AUDIO`, `group_audio`): reading only the top
   video variant's segments (what `outputs_live` and `cf_live_check` do) shows no audio stream. The driver's
   `cf-hls-audio` step probes the master playlist instead. Worth teaching both readers.
5. Cosmetic, as on 2026-10-13: the ffmpeg sinks report `keyframes_sent` 0 and `audio_packets_sent` 0, and the
   stream status `fps` is null.
6. Harness: the serve image has no `jq` (installed per pod with apt); a `pkill -f` whose pattern also matches its own
   command line kills the sidecar's shell (use `pgrep` + PIDs).

## Cost and clean-up

| | |
|---|---|
| Pod A `icpj4bitaiufgb` | 3 438 s at $2.49/hr = **$2.38**; deleted, `GET /pods/…` 404; backstop process stopped |
| Pod B `4nep0hs3ifiqt9` | 3 550 s = **$2.46**; deleted, 404; backstop stopped |
| Pod C `vv1awkcuxvqjh4` | 805 s = **$0.56** (EU volume; turbo loaded in 232 s from it); deleted, 404; backstop stopped |
| Build pod | the shared `igkm3vyed88u9a` (own worktree and target dir) for two example builds and clippy, a few minutes; it stopped itself afterwards and could not be restarted (no EU-RO-1 CPU capacity), so a third build was dropped |
| Cloudflare | 15 `fv-test-` live inputs, all deleted and checked (404); 7 RTMPS/SRT recordings deleted; ~10 min WHEP + HLS viewing: cents. `divine-firefly-3665` untouched (still listed at the end) |
| Volume / R2 | read only; nothing written. R2 plans were presigned GETs minted locally; no key reached a pod |
| Balance | $49.25 at the start, $30.59 before pod C (other agents spending ~$8.6/hr in parallel) |

## Reproduce

```bash
# T2V matrix (fv-gpucheck), R2 mode; plan: scripts/gpu/r2-weights.py plan h3-base FastH3-4-step-Preview-v1-LoRA \
#   auxiliary/tae auxiliary/lpips h3-turbo-loras/{larry,lightx2v} h3-base/transformer_prequant_{4step-vsa,sol-h3}_mxfp8 \
#   fasth3-trim-8step h3-8step --ttl 14400 (filtered to the subtrees read), bundled as hr/plan.json with h3-round.sh
FV_E2E_STATE=… RUNPOD_GPU_MAX_DPH=2.6 FV_POD_CAP_S=5400 FV_E2E_DISK_GB=380 FV_E2E_NO_SECRETS=1 \
  scripts/serve/e2e/pod.sh up ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:17ef12ef1c176543a0beb5c205d48d2d044fd6d9fa50f6b09308ecbb1a510493 /etc/fv/runpod-fake.toml
pod.sh bundle a.tgz; pod.sh run 'bash /e2e/hr/h3-round.sh prep'; pod.sh run 'bash /e2e/hr/h3-round.sh t2v-all'
pod.sh run 'for c in larry lightx turbo max; do bash /e2e/hr/h3-round.sh t2v $c 107; done'
# Image jobs + Cloudflare (two secret references only)
FV_E2E_SECRETS_ALLOW="fv_cf_stream_api_token fv_cf_account_id" \
FV_E2E_ENV_JSON='{"FV_CF_STREAM_API_TOKEN":"{{ RUNPOD_SECRET_fv_cf_stream_api_token }}","FV_CF_STREAM_NAME_PREFIX":"fv-test-"}' \
  FV_E2E_NO_SECRETS=1 … pod.sh up ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:b98084e9dcf194ce711b87f225a2583d4220226010d5ba1238c479597d43db58 /etc/fv/runpod-fake.toml
pod.sh keyrun 'bash /e2e/hr/h3-round.sh cf; bash /e2e/hr/h3-round.sh cf-director; bash /e2e/hr/h3-round.sh cf-hls-audio; bash /e2e/hr/h3-round.sh cond-all'
pod.sh keyrun 'bash /e2e/hr/h3-round.sh cf-audio; bash /e2e/hr/h3-round.sh cf-audio2'
pod.sh down
```
