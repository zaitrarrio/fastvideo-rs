# GPU re-verification of the WP-18 E2E fixes

Run on 2026-09-28. The earlier E2E runs ([wan.md](wan.md), [h3-turbo.md](h3-turbo.md),
[ltx.md](ltx.md), [h3-max.md](h3-max.md)) fixed four things they could not
test on the right hardware. This run tested all four on one Hopper pod,
restarting fv-serve with a different config for each check.

| | |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:c2a7805f…297e` (`sha-7e82504` = `latest`). Main is `5d63ce2`, which only changes docs and artifacts after `7e82504` (`crates/` and `configs/` are identical). The serve-image workflow is path-filtered, so main has no `sha-5d63ce2` tag, and `7e82504` is the image of current main. It contains 450fb48 (pacer backpressure), 677bca6 (director cancel), cf7ac2c (fal default resolution) and c054278 (MXFP8 → W8A8) |
| GPU | 1x **H200** (143 GB, sm_90), driver 580.126.20, US-CA-2, $4.59/h secure. The first two types on the list (H100 80GB HBM3, H100 NVL) had no stock at create time. The create call had no `allowedCudaVersions` filter. The image was cached on the host, so `/ping` answered 204 within 2 min |
| Weights | US volume `s2k01690bi` at `/workspace`, only read (state in `/fvstate`) |
| Encoders | No NVENC on H200: MP4s use x264 and streams use `CpuTestX264` |
| Harness | `scripts/serve/e2e/pod.sh` (sidecar pod, `pod-boot.sh`, clients on the pod over loopback), plus two new files: `reverify-pod.sh` (configs, fv-serve restart, MediaMTX) and `reverify.py` (fal job, `/v1/videos/sync`, SF-Wan WHIP stream). `FV_ADMIN_TOKEN` was set at create time, and every key lived only in a mode-600 state file |
| Pod | `drfrs3nz8lf4hl` (`fv-reverify-0928184253`), 18:42:57 → 19:04:52 UTC, **1315 s = $1.68**. Deleted and verified (API 404). The detached backstop (3000 s) was stopped after the delete |
| Raw results | `artifacts/serve/e2e/reverify/` |

## Summary

| # | Fix | Result |
|---|---|---|
| 1 | SF-Wan pacer backpressure (450fb48) on a GPU faster than playout | **PASS**: 0 pacer drops (was 1265 in 5 min on H200), 0 underruns, 16.00 fps delivered, 30.3 IDR/min |
| 2 | Director close cancels in-flight chunks (677bca6) | **PASS with a caveat**: the stopped session's FL2VA chunk is cancelled at its first denoise step. The next session's chunk 0 took **24.2 s** with a warm page cache (was 88.8 s), and a fal job waited **7.3–9.0 s** in the queue (was 71 s). On a cold cache the next chunk 0 still took **85.2 s**, because the non-cancellable Qwen-VL text stage streams from the volume for 59 s (a known limit of the fix) |
| 3 | fal default resolution (cf7ac2c) | **PASS**: `fastvideo/ltx-turbo` `{prompt, seed}` → 200, 1920x1080. `minimax/h3-draft` `{prompt, seed}` → 200, 832x480 |
| 4 | h3-max MXFP8 → W8A8 on Hopper (c054278) | **PASS**: the log shows `mxfp8 needs sm_100+ (this GPU is sm_90); running W8A8`. One 768p T2V ran in 25.6 s wall, with 17.5 s denoise. The frames are clean |

## 1. SF-Wan live, `/fv/v1/streams` → WHIP → MediaMTX, 3 min

Config: `sf-wan` = `sfwan21-1.3b`, native + reactor only, WebRTC on
loopback (`reverify-pod.sh cfgs`). MediaMTX v1.15.1 ran on the pod. An RTSP
reader recorded what MediaMTX served (`ffmpeg -c copy`). The client sent
`set_prompt` at 60 s and 120 s, and polled the status every second.

| Metric | Value |
|---|---|
| fv-serve restart → `/ping` 200 | 133 s (first SF-Wan load from the volume) |
| TTFF | `first_frame_ms` 1826; `load` 1136 ms, `first_block` 690 ms, `transport` 369 ms, total 2.20 s; POST → `streaming` 2.0 s |
| Delivered (RTSP recording) | 2886 frames in 180.31 s = **16.00 fps**. The largest gap between packets was 0.063 s |
| Pacer | `effective_fps` 16.0, `unique_fps` 15.97, **`pacer_dropped` 0**, **underruns 0**, repeated 0, `video_send_errors` 0 |
| Backpressure | 248 blocks (≈ 2976 frames) were generated in 183 s: generation was held to playout. The same H200 generates 23.6 frames/s free-running (wan.md). Before the fix, 20.6 % of the frames were dropped here |
| IDRs | 91 key frames in 180.3 s = **30.3/min** (2 s GOP); `keyframe_requests` 92, of which 91 were covered by the periodic IDR; `forced_idrs` 1 |
| Prompt switch | Command reply **5.8 ms / 4.7 ms**. At the switch the session was at block 86 (≈ 1032 frames generated) against 979 frames played: about 53 frames (3.3 s) were queued ahead. That is near the ≤ 24-frame pacer mark plus the 2-block channel of the fix (the old bound was 39 frames + 1 block). In `sfwan-switches.jpg`, the lagoon colours of switch 2 show from about +4–6 s. Switch 1 (mountains) cannot be picked out within 10 s: the scene was already degraded (below) |
| GPU memory | 25705 MiB at the start → 26259 MiB (flat after one step) |

Quality: the stream degrades as the rerun in wan.md already found (the
open long-horizon item, design risk R12). `sfwan-longrun.jpg` shows frames
at 5, 60, 120 and 175 s: clean at 5 s, with colour banding over the top rows
by 60 s that grows over time. This run did not change or retest that item.

## 2. Director cancel (h3-turbo, `continuity = "anchor-last-frame"`)

Headless Chromium ran on the pod: `tests/compat/suites/fal_director.mjs`,
app `minimax/h3-turbo/director`, 480p. The first scenario (requestMiddleware)
stops after chunk 0 while the FL2VA chunk 1 builds. The second scenario
(proxyUrl) then opens a new session right away. When the suite exited (with
the second session also stopped mid-chunk), a fal T2V 480P job was submitted
at once. This sequence ran twice.

| | Run 1 (cold Qwen-VL page cache) | Run 2 (warm) | Before the fix (h3-turbo.md) |
|---|---|---|---|
| Session 1 `chunk 0` after ready | 18.8 s | 18.1 s | 20.5 s |
| Session 2 `chunk 0` after ready | 85.2 s | **24.2 s** | 88.8 s |
| fal job `IN_QUEUE` → `IN_PROGRESS` after the suite | **7.3 s** | **9.0 s** | 71 s |
| fal job submit → `COMPLETED` | 17.5 s | 18.7 s | — |
| Both scenarios | PASS, 24 fps VP8 832x480 | PASS | PASS |

Serve log, run 2 (`out/serve-director2.txt`):
1. Session 1 closes while its chunk 1 is in the Qwen-VL stage.
2. That stage finishes (`llm prefetch … 5.80s`) and `encode first
   (anchor-0)` runs.
3. Session 1's build then stops: no denoise step of it follows.
4. The next `memory text` and `step 4/4` belong to session 2's T2V chunk 0.

After session 2 stopped, its chunk 2 went the same way:
1. Qwen-VL ran for 5.9 s.
2. `encode first (anchor-1)` ran.
3. The build was cancelled.
4. The fal job ran (steps at 1.5 s, 480p 5 s).

So the fal job's 7–9 s queue time is the one stage that cannot be cancelled
(Qwen-VL text plus the first-frame encode), not a whole chunk.

In run 1 the same stage took 59 s (`host read+convert 58.87s`), because the
first FL2VA of the pod streamed Qwen-VL from the volume. So session 2
still waited 85 s. Even so, the cancel worked: without it, the full chunk
(+13 s denoise, +4 s decode) would have come on top. The fal job after run 1
waited only 7.3 s because by then the cache was warm. The text-stage limit
is recorded in h3-turbo.md finding 1, and nothing new was fixed here.
Cancelling inside the Qwen-VL layer loop would remove it.

## 3. fal default resolution (no `resolution` in the body)

| App (config) | Body | Result |
|---|---|---|
| `fastvideo/ltx-turbo` (`runpod-ltx.toml`, `ltx25-distill-sol`, profile `ltx2/ltx25_distill_sol` on sm_90) | `{prompt, seed}` | **PASS**: submit 200 → `COMPLETED`, result 200. MP4 1920x1080 H.264 High, 121 frames @ 24, AAC 48 kHz stereo. First job after the restart: 74.9 s wall, 24.3 s inference (was 422 "short edge 768 is not supported") |
| `minimax/h3-draft` (`runpod.toml` with `recipe = "h3-draft"`) | `{prompt, seed}` | **PASS**: 200 → `COMPLETED` in 9.6 s (inference 6.2 s). MP4 832x480, 124 frames @ 24, AAC 32 kHz stereo |

The restarts to `/ping` 200 took 40 s (h3-draft) and 27 s (LTX) with a warm
cache.

## 4. h3-max (Sol-H3 tau ladder) on Hopper

Config: `runpod-h3-max.toml` (`sol-h3`, profile
`h3/sol_h3_4step_engine_ladder`). The restart to `/ping` 200 took 65 s
(h3-base was in the page cache from item 2).

- **Load log.** `FASTVIDEO_H3_QUANT=mxfp8 needs sm_100+ (this GPU is sm_90);
  running W8A8`, then `h3 quant: w8a8 (312 reference linears incl. refiner;
  FastVideo tensorwise W8A8, all blocks + refiner), bf16 activations`. The
  same fallback also ran for h3-turbo in item 2.
- **Request.** One `POST /v1/videos/sync` with `h3-max`, 1344x768, 5 s,
  seed 7 → 302 to R2, **25.6 s wall**.
  - MP4: 1344x768 H.264 High, 124 frames @ 24, AAC 32 kHz stereo, 3.7 MB
    (x264).
  - `X-Stage-Durations`: text 0.94 s, refine 0.03 s, **denoise 17.51 s**
    (steps 5.5 / 4.3 / 4.0 / 3.7 s), audio_decode 0.36 s,
    video_decode 5.38 s, encode 1.53 s.
  - Peak memory: 53.4 GB.
  - For comparison, RTX PRO 6000 with native MXFP8 took 21.1 s denoise and
    28.7 s engine time (h3-max.md).
- **Quality** (`h3max-768p-grid.jpg`, frames 0/40/80/120, fox in a birch
  forest). The frames are sharp and coherent and follow the prompt: fur
  detail, birch bark, clean snow shadows. There is no noise, colour cast,
  banding or blockiness from the W8A8 path.

## Harness notes

- `pod.sh` has no file download. Results came back through the sidecar
  `exec` as base64 pieces. MediaMTX started through the sidecar kept the
  exec's pipe open, so that exec never returned; this did no harm.
- The fal director suite printed a 429 once per run while the previous
  session released. This is expected, and the suite retries.

## Files (`artifacts/serve/e2e/reverify/`)

- `out/sfwan.json` holds the stream's final status, the switches and the
  RTSP packet stats. The same folder has `out/sfwan/switches.json` and
  `out/sfwan-gpu.csv`.
- `out/director{1,2}.json` hold the suite outputs, with timelines.
  `out/fal-after-director{1,2}.json` hold the fal jobs submitted right after,
  and `out/serve-director2.txt` is the filtered serve log.
- `out/fal-ltx-nores.json` and `out/fal-h3draft-nores.json` are the fal jobs
  sent without `resolution`, and `out/h3max.json` is the 768p sync request.
- `h3max-768p-grid.jpg`, `sfwan-switches.jpg` (rows: switch 1 and switch 2 at
  +0/2/4/6/8/10 s) and `sfwan-longrun.jpg` are the contact sheets.
- `ledger.tsv` has the pod create and delete records.
