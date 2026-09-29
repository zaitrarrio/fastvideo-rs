# WP-18 GPU E2E: LTX-2.5 (pod C)

Run on 2026-09-28 (design §7.6, §8 WP-18). Real generations through every API
that serves LTX on one Runpod pod, first with `ltx-turbo`, then with the same
pod restarted as `ltx-pro`.

| | |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-9c42844` (main 9c42844) |
| GPU | 1x RTX PRO 6000 Blackwell Server Edition (96 GB), EUR-IS-1, $2.09/h secure |
| Weights | EU network volume at `/workspace` (read only; state in `/fvstate`) |
| Config | `configs/serve/runpod-ltx.toml` (with `fal_apps`, below), inlined by `scripts/serve/e2e/ltx-pod.sh`; stores R2 + D1; post encoder `auto` (NVENC) |
| Models | `ltx-turbo` = `ltx25-distill-two-stage-sol` (Sol stage 2); `ltx-pro` = `ltx25-distill-two-stage-dense` |
| Driver | `scripts/serve/e2e/ltx-pod.sh up / switch ltx-pro / down`, `scripts/serve/e2e/ltx_e2e.py <base> <out> <cases…>` |
| Raw results | `artifacts/serve/e2e/ltx/results.jsonl` (one line per case, ffprobe facts included) |
| Samples | `artifacts/serve/e2e/ltx/sample-turbo-720p24-first3s.mp4`, `sample-pro-1080p24-first2s.mp4` (stream-copied excerpts of served MP4s); frames `frame-*.jpg` |

**Result: every case passes.** One fal bug was found and fixed (below). Pod
lifetime 22.9 min (create 16:32:00, delete 16:54:53 UTC, deletion verified):
**$0.80**.

## Startup

| Phase | Time |
|---|---|
| Create → first `/ping` answer (204, loading) | < 2 min (image pull included) |
| Create → `/ping` 200 (`ltx-turbo` resident, fast loading) | 2 min 53 s |
| `PATCH` env (`ltx-pro`) → pro resident and ready | 80 s |

## Results

Times are client wall times over the Runpod HTTP proxy: sync = the request;
async = submit → terminal status (2-3 s poll granularity), before the
download. "Inference" is the denoise time the fal status reports
(`metrics.inference_time`). Every MP4 is H.264 at the requested
width×height (1088 generated, cropped to 1080); "audio" is AAC 48 kHz
stereo. Frame counts are ffprobe `nb_read_frames`, rates `r_frame_rate`.

### ltx-turbo

| # | Endpoint | Request | Result | Time | Frames @ rate | MP4 |
|---|---|---|---|---|---|---|
| 1 | LTX `POST /v2/text-to-video` + `GET /v2/text-to-video/{id}` + `result.video_url` (R2, no key) | `ltx-2-5-fast` 1280x720, 6 s, 24 fps (first job: warm-up) | PASS, `processing`→`completed` | 54.6 s | 145 @ 24 | 7.6 MB, audio |
| 2 | LTX `POST /v1/text-to-video` (sync, body = MP4) | `ltx-2-5-fast` 1920x1080, 6 s, 24 fps | PASS, 200 `video/mp4`, `x-request-id` | **40.8 s** | 145 @ 24 | 13.3 MB, audio |
| 3 | LTX v2 | 1920x1080, 6 s, **25 fps** | PASS | 41.5 s | 153 @ 25 | 13.2 MB, audio |
| 4 | LTX v2 | 1920x1080, 6 s, **48 fps** | PASS | 78.0 s | 289 @ 48 | 33.1 MB, audio |
| 5 | LTX v2 | 1920x1080, 6 s, **50 fps** | PASS | 83.7 s | 305 @ 50 | 31.7 MB, audio |
| 6 | LTX v2, video only | 1920x1080, 6 s, 24 fps, `generate_audio: false` (skip_audio_decode) | PASS, **no audio stream** | 43.4 s | 145 @ 24 | 13.3 MB |
| 7 | LTX v2 | 1920x1080, **20 s**, 24 fps | PASS | 144.2 s | 481 @ 24 | 41.7 MB, audio |
| 8 | LTX `POST /v1/upload` → `PUT upload_url` → `/v1/image-to-video` and `/v2/image-to-video` with `image_uri: ltx://uploads/…` | 1080p 6 s | PASS as designed at the time: upload 200 + `ltx://` URI, PUT 200; both i2v calls 400 (`Ltx25I2V`). **Served since E5, see "Image conditioning" below** | — | — | — |
| 9 | FastVideo `POST /v1/videos` → poll → `/content` | `model: ltx-turbo`, 1920x1080, `seconds: 5` | PASS, `completed` | 34.5 s | 121 @ 24 | 10.7 MB, audio |
| 10 | Native `POST /fv/v1/jobs` → poll → `output.url` | `ltx-turbo`, 1920x1080, `seconds: 5`, **fps 50** | PASS, `succeeded`; job `num_frames 257`, `tier turbo`, `recipe ltx25-distill-two-stage-sol` | 71.7 s (run 66.6 s, queue 0.3 s) | 257 @ 50 | 28.6 MB, audio |
| 11 | fal queue `POST /fastvideo/ltx-turbo/text-to-video` → status → result | `{prompt, seed}` (no `resolution`) | **FAIL (bug, fixed below)**: 422 "short edge 768 is not supported" | — | — | — |
| 12 | fal queue, same app | `resolution: "1080P"` | PASS, `COMPLETED`; result `{video, timings, expanded_prompt}` | 36.3 s (inference 27.1 s) | 121 @ 24 | 10.4 MB, audio |

### ltx-pro (same pod, restarted)

| # | Endpoint | Request | Result | Time | Frames @ rate | MP4 |
|---|---|---|---|---|---|---|
| 13 | LTX v2 | `ltx-2-5-pro` 1920x1080, 6 s, 24 fps (first job after restart) | PASS | 86.9 s | 145 @ 24 | 12.2 MB, audio |
| 14 | fal queue `fastvideo/ltx-pro` | `resolution: "1080P"`, 5 s | PASS | 39.6 s (inference 30.4 s) | 121 @ 24 | 10.5 MB, audio |
| 15 | Native `/fv/v1/jobs` | `ltx-pro`, 1920x1080, **20 s** | PASS, `tier max`, `recipe ltx25-distill-two-stage-dense` | 223.2 s (run 218.1 s) | 481 @ 24 | 41.9 MB, audio |

### Refusals (LTX error bodies `{"type":"error","error":{type,message}}`)

| Case | Expected | Got |
|---|---|---|
| `ltx-2-5-pro` on the turbo pod | 403 `permission_error` | PASS |
| `duration: 5` (the LTX matrix starts at 6 s) | 400 `invalid_request_error` | PASS, lists 6..20 |
| 20 s at 50 fps (fast: 6/8/10 s only at 48/50) | 400 | PASS |
| 20 s on `ltx-2-5-pro` (pro: 6/8/10 s) | 400 | PASS |
| `/v2/retake` | 403 `permission_error` | PASS |
| wrong key | 401 `authentication_error` | PASS |

## Observations

- The task's "1080p 5 s" is not an LTX API request: the LTX matrix starts at
  6 s (400 above). 5 s at 1080p ran through `/v1/videos` (24 fps, 121
  frames), the native API (50 fps, 257 frames) and fal (24 fps). Every rate
  24/25/48/50 ran at 1080p 6 s through the LTX API.
- Steady-state 1080p turbo: ~27 s denoise, ~35-41 s end to end for 121-153
  frames; 48/50 fps doubles the frames and the time (78-84 s for 289-305
  frames). 20 s at 1080p: 144 s turbo, 218 s run for pro (dense stage 2).
- The video-only request did not measure faster than the audio one (43.4 s
  vs 40.8 s sync; poll granularity is 2 s): the audio decode is small next
  to the video path at 1080p.
- All v1 sync requests finished under the Runpod proxy's ~100 s limit; a
  1080p 48/50 fps sync request (~80 s) is close to it, so long jobs belong on
  v2 over the proxy.
- `GET /fv/v1/jobs` lists native-protocol jobs only (LTX/fal/`/v1/videos`
  jobs are not in it), and the native job object carries no stage
  durations; per-stage timings on this pod come only from fal
  `metrics.inference_time`. A `metrics` field on the native job is a
  follow-up.
- The weight volume, `FV_WEIGHTS=/workspace/weights` and `ltx25/` layout
  worked unchanged on the EU volume; R2 media and D1 jobs worked.
- `RUNPOD_ALLOWED_CUDA=13.0` hid the only RTX PRO 6000 in stock ("no
  instances"); without the filter the pod came up and ran.

## Bug fixed

**fal apps on non-H3 models refused a body without `resolution`.** The fal
schema defaults `resolution` to `768P` (the MiniMax H3 contract), and
LTX (tiers 1080/720/1440/2160) — and `minimax/h3-draft` (480 only) — have no
768 tier, so `{prompt}` answered 422. Now an omitted `resolution` becomes the
resolved model's first tier when the model has no 768 tier
(`fastvideo_fal::schema::default_resolution_for`, called in
`queue::submit_job`; H3 turbo/max unchanged). Unit test
`omitted_resolution_follows_the_model_tiers`. Not yet re-run on a GPU (the
pod ran the 9c42844 image).

**Config:** `configs/serve/runpod-ltx.toml` mounted fal with the default
`minimax/h3-*` apps, which have no model on an LTX pod; it now sets
`fal_apps = ["fastvideo/ltx-turbo"]` (the run used `fastvideo/ltx-pro` for
the pro phase).

## Image conditioning (E5 / E9), 2026-09-28

Image `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-cb300fa` (main cb300fa),
`ltx-turbo`, 1x RTX PRO 6000 (EUR-IS-1, $2.09/h), pod `fv-ltxi-e2e-*`
created 21:40:43, deleted 21:49:07 (verified): **$0.30**. Boot to `/ping`
200: 74 s. Driver: `scripts/serve/e2e/ltx_e2e.py … i2v-v2-1080p kf-v2-1080p
i2v-upload i2v-v2-720p native-kf-720p v2-1440p-24` (key per run). Input: the
TI2V beach fixture (832x480 JPEG, data URI) and, for keyframes, its 1.35x
zoom as the last frame. Fidelity: SSIM / PSNR (ffmpeg) of the pinned output
frames against the images prepared as the engine prepares them (cover +
center crop to the 64-aligned generation canvas, then the output crop).

| Case | Request | Result | Time | Frames @ rate | Pinned-frame fidelity |
|---|---|---|---|---|---|
| `i2v-v2-1080p` | LTX `POST /v2/image-to-video`, `ltx-2-5-fast`, 1920x1080, 6 s, `image_uri` (data URI) | PASS, `processing` → `completed` | 80.3 s (first job after boot) | 145 @ 24, audio | frame 0: SSIM 0.9884, PSNR 43.6 dB |
| `kf-v2-1080p` | same + `last_frame_uri` (the zoom) | PASS | 44.1 s | 145 @ 24, audio | frame 0: 0.9887 / 43.7 dB; frame 144: 0.9884 / 43.3 dB |
| `i2v-upload` | `/v1/upload` → PUT → `/v2/image-to-video`, then `/v1/image-to-video` (sync), then `/v2` again, all with the same `ltx://` URI | first run: v1 PASS, the v2 leg after it answered 404 with an empty body; **rerun (pod `xgake526rrltxn`, 21:54-22:04, $0.33): all three PASS** (202 / 200 `video/mp4` 145 @ 24 with audio / 202, each with `X-Request-Id`) | — | 145 @ 24 | — |
| `i2v-v2-720p` | 1280x720 (generated 1280x768, cropped) | PASS | 23.6 s | 145 @ 24, audio | frame 0: 0.9804 / 41.8 dB |
| `native-kf-720p` | native `/fv/v1/jobs`, `image_url` + `last_image_url`, 1280x720, 5 s | PASS, `succeeded`, run 16.4 s | 20.2 s | 121 @ 24, audio | frame 0: 0.9805 / 41.9 dB; frame 120: 0.9807 / 42.1 dB |
| `v2-1440p-24` | text-to-video, 2560x1440, 6 s (the 1440p tier, untested before) | PASS | 111.0 s | 145 @ 24, audio, 2560x1440 | — |

Notes:

- The first run's v2 404 of `i2v-upload` did not come from fv-serve: it had
  no `X-Request-Id` and no LTX error body, the same sequence passes against
  the fake engine locally (`upload_then_image_to_video` now covers upload →
  v2 → v1 sync → v2), and the GPU rerun above passed every leg (the proxy
  answers carry `Server: cloudflare`). A transient Runpod-proxy answer.
- The first run's `server` block for LTX/fal/`/v1/videos` cases (e.g.
  `v2-1440p-24`: 121 frames, 1280x768) described **another job**: the
  script read the newest job of `GET /fv/v1/jobs`, which lists native jobs
  only (here `native-kf-720p`). The delivered 1440p file is 2560x1440, 145
  frames, 24 fps, as requested. `ltx_e2e.py` no longer attaches that block to
  non-native cases.
- The frame-0 numbers are the fidelity of the pinned frame to the image after
  the model's own VAE round trip and H.264; the GPU oracle (docs/oracle.md,
  "LTX-2.5 image conditioning") shows the reference upstream pipeline at the
  same level (512p: ours 0.9668, reference 0.9661 SSIM).
- 720p and 1440p text-to-video now both ran on GPU (720p in the first run,
  row 1; 1440p here).

## Reference-to-video (Ingredients IC-LoRA), 2026-09-28: setup

The serve path for the LTX reference mode (docs/ports/ltx-ref2v.md) is built
and tested on CPU (`scripts/serve/check.sh`: caps, routing, negotiate, the
native body and the fal `ingredient` schema), and the engine path passed the
GPU oracle on H100 (docs/oracle.md, "LTX-2.5 reference-to-video"). The serve
E2E on a GPU pod was held for budget on 2026-09-28 and **ran on 2026-09-29** (below).
Everything for it is in the repo:

- config `configs/serve/runpod-ltx-ref2v.toml`: only the `ltx25-ref2v`
  companion (the `ltx-pro` recipe with the IC-LoRA at stage 1), fal app
  `fal-ai/ltx-2.3-quality`; an 80 GB card suffices (64 GiB live, 70 GiB peak
  in the oracle run at 1536x896x121);
- pod: `RUNPOD_VOLUME_ID=s2k01690bi RUNPOD_GPU_TYPES="NVIDIA H100 80GB HBM3"
  RUNPOD_GPU_MAX_DPH=3.6 FV_SERVE_TOML=configs/serve/runpod-ltx-ref2v.toml
  bash scripts/serve/e2e/ltx-pod.sh up ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-<sha>`
  (US-CA-2 had no RTX PRO 6000 stock that day);
- cases (`ltx_e2e.py`): `probe`, `ref2v-fal-ingredient` (fal queue
  `fal-ai/ltx-2.3-quality/ingredient`, `image_url` = the oracle's sheet as a
  data URI, the oracle's prompt, seed 1024: expect 1536x896, 121 @ 24 with
  audio, plus frame 0 against the sheet), `ref2v-native` (`/fv/v1/jobs`,
  `model: ltx-pro`, `reference_urls`, `size: 1536x896`, `num_frames: 121`).

### GPU run, 2026-09-29

| | |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:680ffa2a…` = `ltx-sha-2cd1ba0`, the ltx variant image of main `2cd1ba0` (serve-image run 36503456884, green; contains `b38a408`). It is also the image of the `fv-serve-ltx-pod` template |
| Config | `configs/serve/runpod-ltx-ref2v.toml`, inlined by `ltx-pod.sh` (`FV_SERVE_TOML`); state in `/fvstate`, the volume read only |
| GPU | 1x H100 80GB HBM3, US-CA-2, US volume `s2k01690bi`, $3.49/hr. EUR-IS-1 had no RTX PRO 6000 for 6 minutes (6 tries, HTTP 500 "no instances"), so the driver fell back to the US H100 the setup above names |
| Pod | `668xunbciyho04` (`fv-ltx-ref2v-0929023958`), created 02:40:00, deleted 02:42:54 UTC and checked gone: 174 s, **about $0.17**. Create to `/ping` 200: 84 s |
| Driver | `ltx_e2e.py <base> artifacts/serve/e2e/ltx-ref2v probe ref2v-fal-ingredient ref2v-native` |
| Raw results | `artifacts/serve/e2e/ltx-ref2v/results.jsonl`, frames every 10th of the fal clip: `fal-ingredient-every10.jpg` |

**Result: all three cases pass.**

| Case | Request | Result | Time | Output |
|---|---|---|---|---|
| `probe` | `/healthz`, `/fv/v1/capabilities`, `/fal/schema` | PASS: one model `ltx25-ref2v` loaded, alias `ltx-pro`, tier `max` | — | — |
| `ref2v-fal-ingredient` | fal queue `fal-ai/ltx-2.3-quality/ingredient`, `image_url` = the reference sheet (data URI), oracle prompt, seed 1024 | PASS, `COMPLETED`; result keys `expanded_prompt, seed, timings, video` | 57.8 s (first job after boot), inference 21.2 s | 1536x896, 121 @ 24, AAC 48 kHz stereo, 2.36 MB |
| `ref2v-native` | `/fv/v1/jobs`, `model: ltx-pro`, `reference_urls: [sheet]`, `size: 1536x896`, `num_frames: 121`, seed 1024 | PASS, `succeeded`; `resolved_model ltx25-ref2v`, recipe `ltx25-ic-lora-ingredients-dense` | 25.4 s (run 22.5 s) | 1536x896, 121 @ 24, AAC 48 kHz stereo, 2.40 MB |

- Frame 0 against the sheet (both at 768x448): SSIM 0.777 (fal) and 0.771
  (native).
- **The clips open with the reference sheet.** Frames 0-41 (1.75 s) show
  the sheet's four panels: the coast panels move (waves), the prop and
  character panels stay still, and the sheet's labels come out as garbled
  text ("Stop Reffice Rep:", "Propp"). At frame 42 there is a hard cut
  (ffmpeg scene score 0.61 in both clips) to the generated shot: the
  orange cartoon crab under the red and white umbrella on the wet sand in
  front of the dark boulders, waves breaking, as the prompt describes. The
  two APIs produce the same video (they differ only in the encode).
  Frame 0's SSIM of 0.77 against the sheet comes from that opening; it is
  not the "new shot" the case note above expects.
- The engine path matched upstream `ICLoraPipeline` at the clip level
  (docs/oracle.md, SSIM 0.982), so this opening is probably the model's own
  behaviour for the `Reference sheet: … Generated video: …` prompt format,
  not a serve bug. The oracle run kept no frames to confirm it. **Open
  question for the owner:** should the ingredient endpoint trim the sheet
  segment (the first ~1.75 s), or match whatever fal's hosted
  `ltx-2.3-quality/ingredient` returns? Checking fal's hosted output for the
  same request would settle it.
- Warm run on H100: 22.5 s for 1536x896x121 with the IC-LoRA stage 1 (the
  oracle measured about 1.0 s per stage-1 step and 1.8-2.1 s per stage-2
  step on H100).

## Audio-to-video (avatar P0), 2026-09-29

Image `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-320e25b` (branch build of
the A2V commit), `ltx-turbo` (`configs/serve/runpod-ltx.toml`, fal apps
`fastvideo/ltx-turbo` and `lightricks/ltx-2.5`), 1x H100 80GB HBM3 on the US
volume `s2k01690bi` (US-CA-2; no RTX PRO 6000 in EUR-IS-1 that hour),
$3.49/h. Pod `1kfzrh1szqirmg` created 02:45:12, deleted 02:50:56 (verified,
404): **$0.33**. Boot to `/ping` 200: 219 s (image pull included). Driver:
`scripts/serve/e2e/ltx_e2e.py … probe a2v-v2-1080p a2v-native-i2v-720p
a2v-fal-fast a2v-errors`. Input: the oracle's speech fixture
(`scripts/gpu/fixtures/speech-flite-44k.flac`, 7.0 s, 44.1 kHz stereo FLAC)
as a data URI. Raw results: `artifacts/serve/e2e/ltx-a2v/results.jsonl`;
samples: `sample-a2v-v2-1080p-first3s.mp4`, `frame-*.jpg`.

| Case | Request | Result | Time | Frames @ rate | Audio |
|---|---|---|---|---|---|
| probe | `/fv/v1/capabilities` | PASS: `ltx25-ltx-turbo` tasks `t2v i2v keyframes a2v` | | | |
| `a2v-v2-1080p` | LTX `POST /v2/audio-to-video` `{audio_uri, prompt (talking head), model: ltx-2-5-fast}` | PASS, `processing` → `completed`, R2 URL | 37.5 s (first job) | 161 @ 24 (6.71 s: the longest 8k+1 clip in 7.0 s), 1920x1080 | AAC 44.1 kHz stereo; vs the input: corr 0.99999, lag 0 |
| `a2v-native-i2v-720p` | native `/fv/v1/jobs` `{audio_url, image_url (beach), size 1280x720}` | PASS, run 20.6 s | 25.2 s | 161 @ 24, 1280x720 | same; frame 0 vs the image: SSIM 0.9776, PSNR 41.5 dB |
| `a2v-fal-fast` | fal queue `lightricks/ltx-2.5/audio-to-video/fast` `{audio_url, prompt, seed}` | PASS, `COMPLETED`, `inference_time` 24.9 s | 34.5 s | 161 @ 24, 1920x1080 | same |
| `a2v-err-no-prompt` | no prompt, no image | 400 `invalid_request_error` "prompt is required if image_uri is not provided" | | | |
| `a2v-err-pro-unserved` | `model: ltx-2-5-pro` on the turbo pod | 403 `permission_error` | | | |
| `a2v-err-image-as-audio` | a JPEG as `audio_uri` | 400 `invalid_request_error` "expected audio input, got `image/jpeg`" | | | |

**Lip sync** (`scripts/gpu/lipsync_proxy.py`, the documented SyncNet
stand-in, see h3-1080p-and-upscaler.md "Lip sync"; the flite voice is
synthetic):

| Clip | Face frames | Best lag | r (lag 0) | Speech contrast |
|---|---|---|---|---|
| `a2v-v2-1080p` | 161/161 | -3 | 0.148 (-0.003) | 0.08 |
| `a2v-fal-fast` | 161/161 | -2 | 0.279 (0.172) | 0.49 |
| **pooled** | 2 clips | **-2 (-83 ms)** | 0.21 (0.085) | **0.29** |
| control: the same videos with the audio shifted by 2.3 s | 2 clips | -4 (-167 ms) | 0.22 (0.027) | **-0.02** |

Reading: pooled, the mouth moves with the speech (positive speech contrast,
best lag -2 frames, at the edge of the ±2-frame window: the mouth leads the
sound by about 80 ms); with the audio shifted the speech contrast drops to
zero and the lag leaves the window. The proxy's correlations are weak (r
about 0.2 even for the control), so this rules out a gross break (a frozen
or unrelated mouth), not a subtle offset; a SyncNet score is still the
stronger check.

Notes:

- Every output carries the input audio unchanged (AAC round trip, corr
  0.99999 at lag 0), as upstream returns the input waveform. The native
  job's `output.audio` said 48 kHz (the model's vocoder rate) while the file
  is 44.1 kHz; `negotiate` now records the driving audio's rate for A2V jobs.
- Warm A2V at 1080p for 6.7 s: 25 s of inference, as T2V of the same length.
