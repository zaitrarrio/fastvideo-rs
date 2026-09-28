# WP-18 GPU E2E: h3-turbo (pod A)

Run date: 2026-09-28. This is the design §7.6 E2E for `configs/serve/runpod.toml`
(FastH3 4-step VSA, `h3-turbo`). Every endpoint in this file ran a real
generation on one Runpod pod.

| | |
|---|---|
| Pod | `itxbg5fnspe3nz` (`fv-e2e-a-0928164345`), Runpod SECURE, EUR-IS-1 |
| GPU | RTX PRO 6000 Blackwell Server Edition, driver 595.91.07, 96 GB, 256 vCPU, NVENC |
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:3beb71e4…` (`sha-9c42844`) |
| Weights | EU network volume `jg48s6o1w0` at `/workspace`, read only (nothing was written to it) |
| Stores | D1 jobs, R2 artifacts (`auto`); media URLs are R2 presigned |
| Lifetime | 3222 s at $2.09/hr = **$1.87**; deleted and verified (404) |
| Harness | `scripts/serve/e2e/` (`pod.sh`, `pod-boot.sh`, `sidecar.py`, `batch.py`, `pod-clients.sh`, `fal_client_check.py`) |
| Raw results | `artifacts/serve/e2e/h3-turbo/` |

## How the run worked

- **Start command.** `pod.sh up` creates the pod with `pod-boot.sh` as its
  `bash -c` start command. That script starts `fv-serve --config /e2e/fv.toml`,
  which is `runpod.toml` plus a `[webrtc]` table, in the background. It then
  installs Python and runs `sidecar.py` on `:8001`, which the Runpod HTTP proxy
  also serves.
- **What the sidecar does.**
  - It receives webhooks and callbacks: `fv-serve` posts to
    `http://127.0.0.1:8001/hook/…` with `FV_CALLBACKS_ALLOW_PRIVATE=1`, and
    the test driver reads the deliveries back.
  - It takes a tar of the test files.
  - It runs commands on the pod. Those endpoints require a random token.
- **Why the streaming clients ran on the pod.** The driver's egress is an
  HTTPS proxy and the pod has no UDP, so there was no WebRTC path from the
  driver to the pod. The clients (headless Chromium, `reactor_sdk`) ran on the
  pod itself:
  - `fv-serve` ran without `RUNPOD_POD_ID`/`RUNPOD_PUBLIC_IP`. It kept the
    public base URL and worker id through `FV_PUBLIC_BASE_URL`/`FV_WORKER_ID`.
  - It bound UDP 40010 and ICE-TCP 40000 and advertised 127.0.0.1.
  - This checks the media and session stack, not the Runpod ICE-TCP public
    path.
- **Secrets.** The API key, the admin token (`FV_ADMIN_TOKEN`, set at create
  time) and the sidecar token were random per run. They lived only in a
  mode-600 state file and were never printed.
- **Guards.**
  - The balance floor was $8.
  - A detached 5400 s backstop would delete the pod; it was stopped after the
    manual delete.
  - The ledger is `artifacts/serve/e2e/h3-turbo/ledger.tsv`.
- **Timeline.**
  - Creation first failed for 10 min on RTX PRO 6000 ("There are no instances
    currently available", HTTP 500) while the CUDA 13.0 filter was on.
  - Without the filter, creation succeeded on the first try.
  - Pulling the image took about 10 min. `fv-serve` then went from start to
    `/ping` 200 in 3.0 min (model load from the volume).
  - A restart with a warm page cache took 2.7 min.

## Results

All generations were 5 s at 24 fps (124 frames) with AAC 32 kHz stereo, H.264
High profile. Resolution is 480p (832x480) unless the table says otherwise.

- **Wall** is measured by the client, through the Runpod proxy, from submit to
  terminal status.
- **Inference** is fal's `timings.inference`.

| Endpoint / flow | Result | Timing |
|---|---|---|
| `/health`, `/healthz`, `/ping`, `/metrics`, `/` | PASS (200) | — |
| fal queue T2V `minimax/h3-turbo/text-to-video` 480P: submit, status (+`logs=1`, 4 lines), result, R2 download | PASS | wall 15.1 s, inference 7.0 s, in progress after 0.44 s, MP4 2.9 MB |
| fal storage `/storage/upload/initiate` + `PUT upload_url` + `GET file_url` | PASS | 2.3 s round trip (76 KB JPEG) |
| fal queue I2V (`image_url` = uploaded file) 480P | PASS | **wall 114 s**, inference 7.8 s (see finding 3) |
| fal queue reference-to-video | PASS (clean refusal) | 422 `{"detail":[{"loc":["body"],"msg":"reference-to-video is not enabled on this server"}]}` |
| fal cancel: queued job | PASS | 202 `CANCELLATION_REQUESTED`; status `COMPLETED` + `error_type: client_cancelled`; result 499 |
| fal cancel: running job | PASS | 202; the job ended `client_cancelled` |
| fal cancel after completion | PASS | 400 `ALREADY_COMPLETED` |
| fal sync `/run/minimax/h3-turbo/text-to-video` 480P | PASS | 12.0 s (well under the 100 s proxy cap) |
| fal webhook (`?fal_webhook=`), Ed25519 over `rid\nuid\nts\nsha256(body)` checked against `/.well-known/jwks.json` | PASS | delivered 13.1 s after submit; signature valid; a tampered body is rejected |
| fal validation errors | PASS | bad resolution → 422 with fal `detail`; no key → 401 |
| fal app not served by this process (`minimax/h3-max`) | PASS | 404 `Application "minimax/h3-max" not found` |
| fal-client 1.0.3 (Python) `subscribe(with_logs)` + `run` | PASS | 13.4 s / 11.3 s |
| MiniMax V2 `POST /v2/video_generation` (`MiniMax-H3-Turbo`, 480P) → query loop → `content.url` download | PASS | 13.5 s to `succeeded` |
| MiniMax list `GET /v2/query/video_generation` | PASS | lists the task |
| MiniMax `callback_url`: challenge first, then task bodies | PASS | statuses `queued → running → succeeded` |
| MiniMax error envelope (empty content) | PASS | 400, `… (2013)` |
| FastVideo `/v1/models` (openai 3.6.0) | PASS | `fasth3`, `fasth3-4step-vsa`, `h3-turbo` |
| FastVideo `videos.create_and_poll` (832x480, 5 s) + `download_content` | PASS | 12.6 s + 2.3 s download |
| Native `GET /fv/v1/capabilities` | PASS | one model, tiers, readiness |
| Native `POST /fv/v1/jobs` (16:9, short edge 480) → poll → `/content` 302 → R2 | PASS | 13.1 s |
| fal queue T2V **768P** (1344x768) | PASS | wall 31.2 s, inference 19.1 s, MP4 6.7 MB |
| fal director (`@fal-ai/client` alpha, headless Chromium, `tests/compat/suites/fal_director.mjs`, app `minimax/h3-turbo/director`, 480p) | PASS (both scenarios) | see below |
| Reactor clip mode (`reactor_sdk` 1.6.0, `av`) | PASS | see below |
| Console smoke (`tests/console/smoke.cjs`, Chromium on the pod, public origin) | PASS | 341 s end to end |
| NVENC vs x264 post encoder | measured | see below |

### Streaming

**fal director, 480p (832x480), VP8.** The client offers no H.264, so the
server fell back to VP8 (`vp8_fallback`). Audio was Opus stereo.

- **requestMiddleware scenario:**
  - The peer connection was up in 19 ms.
  - The first audio RTP arrived at 0.10 s; this is the silence track.
  - `chunk 0` arrived at 20.5 s, with the first video RTP at 20.7 s. The
    chunk is 10 s.
  - Video decoded at 24.00 fps and was presented at 24.00 fps. Audio ran at
    47 996 samples/s.
  - Heartbeats held the session past 17 s.
  - A second prompt went through `prompt_pending → prompt_applied`, and a
    stale version was rejected.
  - `stop` gave `stream_exhausted`, followed by the final `session_metrics`.
- **proxyUrl scenario (legacy receive, video only):**
  - The session hit 429 once while the previous session released.
  - `chunk 0` arrived at **88.8 s**, then video ran at 24.00 fps.
  - The 88.8 s was the stopped first session's chunk 1 still building (finding
    1).

**Reactor clip mode (`av`, 1344x768).**

- **Session setup:**
  - `get_state`, `set_autoplay`, `enqueue` and `play` all worked.
  - `set_seed(-3)` raised `ReactorError`.
  - The client received `clip_queued`, `clip_generated`, `clip_started` and
    `clip_finished`.
- **Video:**
  - The first frame arrived 0.68 s after connect; this is the idle frame
    before the clip.
  - The clip was 107 frames; the client received 112 frames.
  - The median gap between frames during playout was 41.7 ms, which is
    24 fps. The largest gap was 3.1 s: playout waited while the clip
    generated.
- **Audio:** 436 frames at 48 kHz mono.

**Console (Chromium on the pod).** The console ran on the public origin with
the admin token set at create time. It covered:

- minting a key
- T2V and I2V with an upload
- the API snippets
- the history
- a live director session at 1344x768 with one video track and one audio
  track, including a second prompt and stop
- the r2v form at 390 px width
- revoking the key, after which the key is refused with 401

### NVENC vs x264 (post encoder)

Method:

- One fal T2V was run with each post encoder: the same prompt and seed,
  run sequentially with nothing else running.
- Between the two, `fv-serve` was restarted with `post_encoder =
  "cpu-test-x264"` and then back to `auto`, which resolves to NVENC.
- CPU is the container cgroup CPU delta. At idle it is 27 ms per 30 s.

| Post encoder | 480p wall / CPU / MP4 | 768p wall / CPU / MP4 | 768p decode+mp4 |
|---|---|---|---|
| NVENC (`p5 hq vbr`) | 16.3 s / 4.25 CPU-s / 2.9 MB | 30.7 s / 6.28 CPU-s / 6.7 MB | 6.48 s |
| libx264 (`veryfast`) | 16.6 s / 4.59 CPU-s / 1.5 MB | 32.2 s / 10.89 CPU-s / 3.6 MB | 6.92 s |

The encode runs in step with the VAE decode (`mp4 tail` is 0.02-0.04 s both
ways), so wall time barely moves. At 768p x264 costs **+4.6 CPU-s per clip
(+73%)**, about 3.7 cores for the length of the decode. At the same cq/crf
NVENC's files are about twice as large.

A standalone encode of the same 768p clip (124 frames), using the args of
`fastvideo-media` `FfmpegH264::file_args` at quality 19, gave these results
(`encode-ffmpeg.jsonl`):

| Encoder | Wall | CPU | Size |
|---|---|---|---|
| NVENC | 0.51 s | 0.36 CPU-s | 6.1 MB |
| x264 veryfast | 0.49 s | 3.5 CPU-s, 7.1 cores | 3.3 MB |
| x264 medium | 6.0 s | 14.9 CPU-s | 3.7 MB |

### Compat suites against the pod URL

`tests/compat/run.sh` starts its own fake `fv-serve` for every suite and has no
target-URL mode. Its API suites (`openai_videos.py`, `minimax.py`,
`fal_webhook.py`) take `--base`, but they check things only the fake engine
has:

- fake model ids (`fake-h3-turbo`, `fake-h3-max`)
- `[fake:fail]` fault injection
- `MiniMax-H3-Max`, which this h3-turbo process does not serve

So they were not run unchanged against the pod. `batch.py` covers the same
flows with real generations, and so do `fal_client_check.py` (the pinned
`fal-client`) and the two streaming suites. The streaming suites ran
unchanged except for new knobs:

- `FV_WMA_APP` in `fal_director.mjs`
- `FV_REACTOR_CLIP_TIMEOUT_S` in `reactor_sdk_compat.py`
- `FV_CONSOLE_ORIGIN` and `FV_CONSOLE_TIMEOUT_MS` in `smoke.cjs`

The local `tests/compat/run.sh fal-director` suite still passes with these
changes (58 s).

## Findings

1. **Fixed: stopping a director session did not cancel its in-flight chunk.**
   - **Symptom.** With `continuity = "anchor-last-frame"`, chunk 1 is an FL2VA
     build. After `stop`, it kept the GPU for about 85 s: 67 s of Qwen-VL
     multimodal text streamed from the volume on a cold cache, plus denoise and
     decode. The next director session's first chunk arrived 88.8 s after
     `ready` instead of about 20 s. The console run and the next queued fal job
     (71 s `IN_QUEUE`) hit the same thing.
   - **Cause.** `Clips::close` in `crates/fastvideo-serve/src/director.rs`
     released the executor slot but left the jobs it had queued running.
   - **Fix.** `Clips` now tracks the cancel token of each build and trips it on
     `close`. A queued build is dropped, and a running one ends at its next
     denoise step.
   - **Test.** `director::tests::close_cancels_the_inflight_build` uses the fake
     engine with 2 s steps; the build is cancelled in under 5 s. The test fails
     without the fix.
   - **Not fully covered.** The FL2VA text stage is not a denoise step, so a
     cancel during it still waits for that stage (about 1.4 s with a warm page
     cache, about 67 s cold).
2. **Recorded: `/uploads/{token}` and `/storage/upload/initiate` answer 405 to
   CORS preflights.**
   - A browser page on another origin (including the console opened at
     `http://127.0.0.1:8000` while `public_base_url` is the proxy URL) cannot
     upload.
   - Same-origin use (the console at its public URL) works.
   - fal's real storage endpoints allow cross-origin uploads, so a CORS layer on
     these two routes would match fal. It was left out as not a minimal fix.
3. **Recorded: I2V is slow (known, WP-11 notes).**
   - I2V wall was 114 s, while `timings.inference` reported 7.8 s. The
     multimodal text encoder streams from the volume on every request.
   - fal's `timings.inference` (and `metrics.inference_time`) covers only the
     denoise, so it hides about 100 s of the request. Consider reporting the
     whole engine time, or adding a `text_encode` timing.
4. **Recorded: some fields are missing or differ from the H3 output.**
   - The fal T2V result has `seed: null` even when the request sets `seed`.
   - The Reactor audio track is mono 48 kHz, while the MP4 audio is stereo
     32 kHz.
5. **Test fix: `tests/console/smoke.cjs` read `#video` too early.** It read the
   `src` right after `#result-status` turned `COMPLETED`, but the result fetch
   sets `src` later. Over a real network the test read nothing. The smoke now
   waits for the `src`.

## Files

- `artifacts/serve/e2e/h3-turbo/batch.json`: one record for every batch check,
  with ffprobe facts.
- `director-480p.json`, `reactor-av.json`: the streaming client outputs.
- `streaming-console-encode.json`: fal-client, the console, and the encoder
  legs.
- `encode-ffmpeg.jsonl`: the standalone encoder comparison.
- `serve-log-excerpt.txt`: a filtered `fv-serve` log with phase timings.
- `samples/fal-t2v-480p.mp4`: one 480p sample, 2.9 MB.
- `ledger.tsv`: the pod create and delete records.
