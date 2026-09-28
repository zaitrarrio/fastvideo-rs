# WP-18 GPU E2E: Wan family (pod D)

Run on 2026-09-28 (design §7.6, §8 WP-18). One Runpod pod: FastWan
(`wan-turbo`) batch through every API that serves it, then the same pod
restarted as SF-Wan live (native `/fv/v1/streams` over WHIP, Reactor causal
mode).

| | |
|---|---|
| Image | `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-9c42844` (main 9c42844), features `cuda,http-client` (+ default `reactor`/`webrtc`) |
| GPU | 1x H200 (143 GB), US-CA-2, driver 580.126, $4.59/h secure. No H100 / H100 NVL in stock for 5 min; H200 is Hopper with the same kernels (sm_90) |
| Weights | US network volume `s2k01690bi` at `/workspace` (read only; state in `/fvscratch`) |
| Phase 1 config | `configs/serve/runpod-wan.toml` + `fal_apps = ["fastvideo/fastwan21-1.3b"]` (now in the shipped file); stores R2 + D1 |
| Phase 2 config | `[[models]] sf-wan = sfwan21-1.3b` (WP-11 `CudaBackend`), `reactor` + `native` only, `[reactor] model = "sf-wan"` (inlined by the pod script) |
| Encoders | H200 has no NVENC: the startup probe resolved `streams = x264-test`, `post = cpu-test-x264`, `reactor = off` (VP8 via libvpx) |
| Driver | `scripts/serve/e2e/wan-pod.sh up / batch / fetch / down`; `wan_batch.py` (phase 1, from outside through the Runpod proxy); `wan-pod-run.sh` runs on the pod (phase 2: MediaMTX, WHIP stream, WHEP viewer, RTSP recorder, `reactor_causal.py`, `fv-gpucheck wan stream`) and publishes results read-only on :8001 |
| Raw results | `artifacts/serve/e2e/wan/batch/results.json`, `artifacts/serve/e2e/wan/live/` |
| Samples | `batch/fastwan-81f-a.mp4` (served MP4), `live2/sample-switch.mp4` (16 s of the rerun's WHIP stream around a prompt switch, re-encoded); contact sheets `live/relativistic-collapse-sheet.jpg`, `live2/rebased-longrun-sheet.jpg` |

**Result:** every batch endpoint passes. SF-Wan live streams at a steady
16 fps with 30 IDR/min, 2.1 s TTFF and 0 underruns, and Reactor causal mode
works end to end, but the run found three bugs in the live path (fixed
below) and one harness bug (the WHEP viewer). A second pod (phase 2 only,
image `sha-9d3f245` with the fixes, RTX PRO 6000) verified them and found
that long SF-Wan rollouts still degrade visually (open, below). Pods: H200
`3wow7t6n2sabhs` 14.5 min ($1.11) and RTX PRO 6000 `zuiko32mh86f3k` (EU
volume `jg48s6o1w0`; no US GPU of any type could be created for 18 min:
"create pod: There are no instances currently available") 11.5 min
($0.40), both deleted and verified gone: **$1.51**.

## Phase 1: FastWan batch

fv-serve start → `/health` 200 (FastWan resident, from the network
volume): **103 s**. Times are client wall times over the Runpod HTTP proxy
(submit → terminal status with 0.25 s polling, download separate). Every MP4
is H.264 High 832x480 @ 16 fps, faststart, no audio.

| # | Endpoint | Request | Result | Time |
|---|---|---|---|---|
| 1 | `GET /health`, `GET /`, `GET /fv/v1/capabilities` | | PASS: `model_loaded: true`, `{"model":"fastwan21-1.3b"}`, recipe `fastwan21-1.3b-dmd3-vsa` resident (the script's own caps check read the wrong JSON path; fixed, the data was right) | 1.4 s |
| 2 | FastWan `POST /generate` → `GET /status/{id}` → `GET /video/{id}` | 81 frames, first job | PASS, `queued`→`processing`→`completed`, 81 frames | 6.73 s (+1.04 s download) |
| 3 | FastWan `/generate` | 81 frames, warm | PASS | **5.00 s** |
| 4 | FastWan `/generate` | 121 frames | PASS, 121 frames | 7.39 s |
| 5 | FastWan `/generate` off grid | `num_frames: 50` | PASS: 400 `{"detail":"num_frames 50 must be one of 9..=129 … nearest valid: 53"}` | 0.35 s |
| 6 | FastWan `DELETE /video/{id}` | | PASS: 200, then `/status` 404 | 0.84 s |
| 7 | `openai` 3.6.0 `models.list` | | PASS: `fastwan21-1.3b`, `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, `wan-turbo` | 1.2 s |
| 8 | `openai` `videos.create` → `retrieve` → `download_content` → `delete` | `size="832x480"`, `seconds="5"` | PASS, 81 frames | 5.10 s |
| 9 | `openai` `videos.create_and_poll` | same | PASS | 5.03 s |
| 10 | native `POST /fv/v1/jobs` → `GET` → `/content` | 832x480, 81 frames | PASS: `/content` 302 to R2, 81 frames | 5.63 s |
| 11 | fal queue `POST /fastvideo/fastwan21-1.3b/text-to-video` → status → response → `video.url` | `duration 5`, `480P`, `16:9` | PASS: `COMPLETED`, `timings.inference` 1.90 s, 81 frames; but **848x480** (bug 3) and file name `…_minimax-h3.mp4` (the hosted H3 slug on every app; cosmetic, recorded) | 5.99 s |

Engine job durations (`fv_job_duration_seconds`): FastWan API 5.86 s mean
(3 jobs, first included), `/v1/videos` 4.63 s, native 6.60 s, fal 4.91 s.
GPU memory peak 29.4 GB. MMAudio: not exposed by fv-serve (only
`fv-gpucheck wan gen --audio mmaudio`), so there is nothing to test through
the APIs; the fal director has no causal mode (clip sessions only), so it
does not apply to SF-Wan.

## Phase 2: SF-Wan live

fv-serve restart as SF-Wan → `/health` 200: **86 s** (model load 81.7 s).

### Native `/fv/v1/streams` → WHIP → MediaMTX (same pod), 5 minutes

`POST /fv/v1/streams {model: sf-wan, whip_url: http://127.0.0.1:8889/sfwan/whip, 832x480}`,
an RTSP reader recording what MediaMTX serves (`ffmpeg -c copy`), `set_prompt`
every 60 s (4 switches), status every 5 s, `nvidia-smi` every 5 s.

| Metric | Value |
|---|---|
| TTFF (POST → first frame at the publisher) | **1.72 s**; phases `load` 1076 ms (session open, prompt encode), `first_block` 647 ms, `transport` 352 ms (WHIP offer, ICE/DTLS, first AU): total 2.07 s |
| Delivered (RTSP recording) | 4806 frames in 300.3 s = **16.00 fps**; H.264 Constrained Baseline L4.0, 832x480, 2.49 Mb/s |
| Pacer | `effective_fps` 16.0, `unique_fps` 15.97, **0 underruns**, 0 repeats, 0 send errors |
| Generation in fv-serve | 513 blocks (6153 frames) in 303 s = **20.3 frames/s** (block ≈ 0.59 s): the relativistic RoPE policy (bug 1) |
| Raw rollout, same pod (`fv-gpucheck wan stream`, rebased sink, graphs, 60 s) | **23.63 frames/s** steady, block p50 0.505 s / p90 0.509 s, memory flat at 26674 MiB (H100 reference: 23.8) |
| Frames dropped by the pacer | **1265 of 6153 (20.6 %)**: bug 2 |
| IDRs | 151 key frames in 300.3 s = **30.2/min** (2 s GOP); `keyframe_requests` 152 (MediaMTX PLI), 151 covered by the periodic IDR, `forced_idrs` 1 (start) |
| Prompt switch | command reply 20 ms (`state_update` with the new prompt); applied at the next block. Frames queued ahead of the switch: 39 (2.4 s) + one block → the first new-prompt frame plays ≈ 3 s after the command; the scene visibly changes 8.7 s (switch 1) and 9.5 s (switch 2) after it (KV kept: the rollout morphs over several blocks) |
| GPU memory | 25431 MiB at stream start; +384 MiB at 87 s, +352 MiB at 212 s, then flat at 26167 MiB to the end (two allocator steps, not a steady climb) |
| Quality | **collapses**: fine for ~30 s, stripes along the top by 60 s, colour bars over half the frame from ~100 s (`relativistic-collapse-sheet.jpg`, frames at 3/20/40/60/100/150/200/290 s): bug 1 |
| WHEP viewer | MediaMTX established the WHEP session (`is reading from path 'sfwan'`), but the aiortc viewer died on its first frame (`numpy` missing from the pod venv): harness bug, fixed |

### Reactor causal mode (`reactor_sdk` 1.6.0, local mode, on the pod)

The SDK reached the runtime through the Runpod ICE-TCP mapping of port 70000
(public IP + mapped port; no loopback fallback was needed). VP8 (the SDK
offers no H.264).

| Step | Result |
|---|---|
| connect (session + WebRTC) | 0.26 s; tracks `["main_video"]` |
| `set_prompt` → first frame | ack 2 ms (bodyless); first frame 0.46 s after the prompt (0.72 s after connect) |
| steady 30 s | 481 frames = **16.03 fps**; canvas 848x480 (bug 3) |
| `set_prompt` switch | ack 4 ms; `state_update` with the new prompt after 3 ms |
| `set_paused(true)` | `state_update{paused: true}`; 43 frames in the 5 s after the first second (the pacer's buffered frames play out; generation stopped) |
| `set_paused(false)` | 5 new frames within 1.06 s |
| `get_state` | `state_update{block_index: 57, paused: false, prompt, seed, unique_fps: 14.55}` |
| `reset` | bodyless ack; `block_index` 57 → 3 two seconds later; 80 frames in the next 5 s |
| messages seen | `state_update` only (no errors) |

### Rerun with the fixes (image `sha-9d3f245`, RTX PRO 6000, NVENC)

Same script, phase 2 only (`FV_E2E_SKIP_PHASE1=1`), results in
`artifacts/serve/e2e/wan/live2/`. fv-serve start → ready 69 s.

| Metric | Value |
|---|---|
| RoPE policy in the serve log | `rope=RebasedSink` (was `Relativistic`) |
| TTFF | first frame at the publisher 0.98 s; `load` 481 ms, `first_block` 500 ms, `transport` 649 ms, total 1.63 s |
| WHEP viewer (aiortc, WHEP from MediaMTX) | **4537 frames decoded in 299.2 s = 15.16 fps**, first frame 0.87 s after the WHEP POST, largest gap between frames 0.16 s |
| RTSP recording | 4555 frames / 300.3 s; 143 key frames = **28.6/min**; Constrained Baseline L4.0 832x480, 2.37 Mb/s, encoder `Nvenc` |
| Pacer | `effective_fps` 15.14 (adaptive: this GPU generates ~15.2 frames/s, below 16), 0 underruns, **0 dropped** (the backpressure path is not exercised on a GPU slower than playout; covered by the CPU test) |
| Raw rollout (`fv-gpucheck wan stream`, 60 s) | 15.22 frames/s, block p50 0.788 s (first SF-Wan number on sm_120) |
| GPU memory, 5 min | 27761 → 28026 MiB (+265 MiB in 32 MiB steps) |
| Reactor causal | all steps pass; canvas now **832x480**; steady 15.33 fps; first frame 0.18 s after `set_prompt`; `reset` block 45 → 3 |
| Quality | better than relativistic, but still **degrades**: a glowing artefact from ~30 s and horizontal banding along the top rows from ~45 s (before the first prompt switch at 60 s), covering the top third by 3-5 min (`live2/rebased-longrun-sheet.jpg`) |

The long-horizon degradation (design risk R12) is not fixed: the E6/E7
10-minute statistics rated the rebased sink stable for 2 minutes, but these
frames show visible artefacts well before that. It needs a visual long-run
study of the rollout (TAEHV carried state on the top rows is one suspect)
before `max_seconds` defaults above ~30 s are sensible.

**Follow-up (R12 study, `docs/ports/wan.md` "Long-run quality: the R12
study").** Reproduced without the server (`runpod-matrix.sh sfquality`).
The bands are in the DiT latents, not the decoder: a fresh-state TAEHV
decode matches the carried state to a few levels, and the latents' top
rows turn into stripes. Graph and eager runs are bitwise equal over 90 s,
and f32 degrades too. The cause is the one-block sink. The default sink is
now 15 frames (a deep sink), clean for 2 minutes on two seeds; to 5 minutes
the content stays coherent, with a transient artefact strip along the top
edge. Sheets: `r12/`. Live causal sessions are now capped on every
front-end: 120 s of video by default, up to 300 s on request, a `reset`
restarting the clock within the 300 s ceiling (`docs/serve/design.md`
§5.2).

## Bugs

1. **SF-Wan served with the relativistic RoPE policy** (`cuda/backend.rs`
   `rollout_base`, WP-11). The E6/E7 measurements chose the rebased sink as
   the default because relativistic flickers after ~60 s and is 0.1 s per
   block slower; the WP-11 `CudaBackend` (every `[[models]]` config)
   hard-coded `Relativistic`, while the older `FV_SFWAN_WEIGHTS` backend used
   the default. Seen here as 20.3 instead of 23.6 frames/s and the collapse
   above. **Fixed:** `RopePolicy::RebasedSink` (verified in the rerun log;
   the remaining degradation is the open item above).
2. **The causal pacer drops a fifth of the frames on a fast GPU**
   (`stream/pace.rs`). The pacer pulled every block as soon as it arrived,
   so any generator faster than 16 fps overflowed the 48-frame drop-oldest
   buffer (1265 frames skipped in 5 min: jumps and fast motion), and the
   executor never waited on the 4-deep channel the design describes.
   **Fixed:** the pacer takes a block only below a pull mark (24 frames),
   so a faster generator waits on the session channel; the adaptive rate is
   estimated from each block's generation time (`FramePacer::push_chunk_rate`)
   because arrivals now follow playout; `EngineConfig::causal_depth` 4 → 2
   so the frames queued ahead of a prompt switch stay near the old bound
   (≤ 24 in the pacer + 2 blocks). Test:
   `a_generator_faster_than_playout_is_held_back_not_dropped` (fails on the
   old pacer at tick 3). On GPU only the slower-than-playout case was
   rerun (0 drops, 0 underruns); a Hopper rerun is still owed.
3. **Aspect canvases can exceed the pixel budget** (`canvas_for_aspect`).
   Capping the area and then rounding both sides to the nearest multiple
   gave 848x480 (407040 px) for 16:9 at 480 on an 832x480 budget: fal
   `480P` on Wan and the Reactor canvas both generated 848x480. **Fixed:**
   when rounding overshoots, a side is rounded down (largest area within the
   budget, then the closest aspect): 832x480 (the rerun's Reactor canvas).
   H3's `resolve_canvas_size` parity test still passes.

Harness: the WHEP viewer needed `numpy` (added; the rerun decoded every frame); fv-serve prints a generated
admin token in its log and the pod script published the logs on :8001 for
the pod's lifetime (now it sets `FV_ADMIN_TOKEN` itself; the committed logs
are redacted excerpts). Recorded, not fixed: the fal output file name uses
the hosted `minimax-h3` slug for every app; the encoder startup WARN says
"falling back to openh264" even when the build has no OpenH264 and the
streams resolve to `x264-test`.
