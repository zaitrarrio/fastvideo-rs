# Streaming reference projects: strobe and infinite-livestream

Research input for the fastvideo-rs multi-protocol streaming server. That server
has to carry video+audio streams (H3 joint audio, LTX-2 audio VAE) and
video-only streams (Wan, FastWan, SF-Wan). This document reads two uploaded
reference projects end to end and ends with a recommendation. Status: research
only. No code in this repo changed.

Companion document: [`research-reactor.md`](research-reactor.md) covers the Reactor
Runtime wire protocol itself (protobuf, signalling routes, track declaration).
This document does not repeat it. It cites that document where the two meet.

## 0. Sources and citation convention

Citations are `path:line` or `path:start-end`, relative to the project root:

| prefix | tree |
|---|---|
| `strobe/…` | the strobe upload, checked out at `.refsrc/strobe/`. It is git-ignored and byte-identical to the uploaded `feb0880a-strobe-source`. |
| `infinite-livestream/…` | the infinite-livestream upload, checked out at `.refsrc/infinite-livestream/`. It is git-ignored. |
| anything else (`crates/…`, `docs/…`) | this repository (`fastvideo-rs`) |

When a claim is my own reading and neither project states it, it is marked
**INFERRED**. Numbers
are quoted as the sources record them, with the GPU named each time. Nothing
here was re-measured.

---

## 1. strobe: the pipeline from end to end

### 1.1 What strobe is

strobe generates causal video in near real time and publishes it outbound over
WebRTC with WHIP to an SFU (MediaMTX or Cloudflare). Viewers use WHEP or HLS
from that SFU. The GPU host needs no inbound media port. The shipped path is
Python (`strobe/src/strobe/*`). The Rust workspace (`strobe/rust/`) is dormant (see §2).

The live pipeline is: `FrameSource` (causal rollout) → `FramePacer` (jitter
buffer) → `GeneratedVideoTrack` (aiortc, libx264) → `WhipPublisher` → SFU.
Session docstring: `strobe/src/strobe/session.py:1-7`. Sources docstring:
`strobe/src/strobe/sources/__init__.py:1-17`.

**The live strobe path carries video only.** `StreamSession.start` adds exactly
one track, a `GeneratedVideoTrack` (`strobe/src/strobe/session.py:176`). No audio
track is ever added. The audio work is all batch or stub (§1.6).

### 1.2 Session lifecycle (`session.py`, `session_mode.py`)

1. **Construct** (`strobe/src/strobe/session.py:40-106`)
   - `apply_request_mode(has_image=…)` runs first and holds a process-wide lock
     for the whole session (`:54-61`). `strobe/src/strobe/session_mode.py` swaps
     settings for I2V:
     - Wan lane: the Causal Forcing frame-wise checkpoint and
       `num_frame_per_block=1` (`session_mode.py:63-99`).
     - LTX lane: only `ltx_num_frame_per_block=1` (`session_mode.py:48-61`).
     - The lock exists because the settings object is process-global
       (`session_mode.py:24-27`).
   - A `FramePacer(fps, max_buffer=buffer_frames, adaptive, min_fps)` is built
     (`:69-74`), then the source (`:75`).
   - Optional per-session overrides:
     - rolling-window attention (`configure_attn`, `:76-80`);
     - an I2V image (`configure_i2v`, `:81-91`).
   - An `RTCPeerConnection` and a `WhipPublisher` are created (`:92-98`).
2. **Start** (`:161-191`)
   - The TTFF clock starts (`ttff.begin`, `:166`) and the feed task is launched
     (`:167`).
   - `start` **waits for the first generated frame before the WHIP handshake**
     (`pacer.wait_first_frame(first_frame_timeout)`, `:170`). "Don't offer an
     empty track."
   - Then it adds the track (`:176`), publishes (`:177`), and sets
     `state="live"` (`:179`).
   - Any failure sets `state="error"`, reports the partial TTFF and calls
     `stop()` (`:182-190`).
3. **Feed loop** (`_feed`, `:114-159`)
   - `async for chunk in source.stream(prompt): pacer.push_chunk(chunk)`.
   - The duration deadline starts at the **first chunk**, not at session start.
     `duration_s` means seconds of video, not wall time, so a 263 s compile
     warm does not use up the stream (`:115-128`).
   - Every 5 s the loop logs `unique_fps = (served − underruns)/dt`
     (`:133-148`).
   - A source exception closes the pacer so `start()` fails fast (`:149-155`).
4. **Prompt steering**: `set_prompt` needs a live session and a source with
   `set_pending_prompt`. Otherwise it raises (`:198-210`).
5. **Stop** (`:212-313`)
   - Order: close the pacer, cancel the feed, `source.close()`, WHIP `DELETE`,
     `pc.close()` (`:222-233`).
   - It then builds a stats dict (`:234-310`): `pushed/served/dropped/underruns`,
     `video_s`, `unique_fps`, `gpu_name`, `vae_decode` (trt/eager), `rolling`,
     `local_attn_size`, `sink_size`, `num_frame_per_block`, `source`,
     `rollout`, `chunked_fallback`, `cache_resets`, `prompt_switches`, `i2v`,
     and `ttff`.
   - Silent fallbacks (chunked instead of causal, eager instead of TRT) are
     exposed on purpose (`:258-293`).

### 1.3 Control-plane API (`server.py`)

FastAPI. The server only starts and stops sessions, so its HTTP port is the only
inbound port (`strobe/src/strobe/server.py:1-3`).

| route | body | reply | errors |
|---|---|---|---|
| `GET /healthz` | – | `{"ok": true, "source": <name>, "sessions": n}` (`:83-85`) | – |
| `POST /sessions` (201) | `CreateSession {prompt (min 1), whip_url?, whip_token?, duration_s? (≥1), local_attn_size?, sink_size?, image_url?, image_b64?}` (`:45-57`) | `{"id", "state"}` (`:121`), sent **after first frame + WHIP** | 429 `max sessions (n) reached` (`:91-95`); 502 `failed to start stream: …` (`:110-111`) |
| `GET /sessions/{id}` | – | `{id, state, prompt, error, stats}` (`:124-130`) | 404 |
| `PATCH /sessions/{id}` | `{prompt}` (`:60-61`) | `{id, state, prompt}` (`:133-143`) | 404; 409 not live / non-interactive (`:140-141`) |
| `DELETE /sessions/{id}` | – | `{id, state, stats}` (`:146-152`) | 404 |

- **Auth.** If `STROBE_API_TOKEN` is set, the mutating routes require
  `Authorization: Bearer` (`:32-42`). A non-loopback bind with no token only
  logs a warning (`:70-75`).
- **The concurrency cap counts sessions that are still starting.** A second POST
  during a multi-minute LTX warm once loaded a second 13B model and OOMed the
  card (`:19-24`, `:91-114`). `max_sessions` defaults to 1
  (`strobe/src/strobe/config.py:149`).

### 1.4 Causal rollout sources

**Protocol** (`strobe/src/strobe/causal.py`)

- `DiTBackend` is the whole model surface (`causal.py:107-149`): `encode_text`,
  `init_caches`, `sample_noise`, `denoise_step(noisy, cond, t, kv_cache,
  crossattn_cache, current_start)`, `add_noise`, `write_context`, and
  `decode → (n,H,W,3) uint8`.
- Four optional audio hooks are declared as comments (`:145-149`).
- `run_causal_rollout` (`:179-374`) runs as follows:
  1. It encodes the text once. When prompt polling is off, it releases the text
     encoder, which reclaims about 6.5 GB (`:219-230`).
  2. For each block: sample noise, run a few-step DMD loop with re-noising
     (`:321-333`), then do the self-forcing clean-context KV rewrite
     `write_context` at `context_noise` (`:336-341`).
  3. It `emit`s the decoded pixels for that block (`:343`).
  4. In full-window mode (`local_attn_size == -1`) it resets the caches when
     `max_latent_frames` (21) would overflow (`:294-319`). The reset can
     KV-seed the previous tail (`motion_context_frames`, `:302-318`).
  5. With `cfg.audio` set, each block also runs a parallel audio loop:
     `sample_audio_noise` → `denoise_audio(…, video=x0)` →
     `write_audio_context` → `emit_audio(decode_audio(...))` (`:346-364`).
     `require_audio_hooks` refuses a backend without these hooks (`:212-215`,
     `strobe/src/strobe/audio_vae.py:24-39`).
- `CausalConfig` defaults (`:34-60`): 3-latent blocks; the step list
  `(1000,750,500,250)`; 1560 tokens per frame (30×52 at 832×480); VAE time
  factor 4; a 21-latent window (81 pixel frames, about 5 s at 16 fps).

**Wan / Self-Forcing** (`strobe/src/strobe/sources/selfforcing.py`, `causal_wan.py`, `causal_backend.py`)

- `causal_wan.py` adds two things to the stock diffusers `WanTransformer3DModel`
  (`causal_wan.py:1-18`):
  - RoPE offset by the block's start frame;
  - a per-layer KV cache, with an optional rolling window and sink tokens
    (`make_kv_caches`, `:28-55`; `evict_kv_cache`, `:58`).
- `CausalWanBackend` uses a flow-match scheduler with shift 5
  (`causal_backend.py:25-52`).
- `warmup()` compiles every block position before going live, because
  `torch.compile` specialises per position (`:208-244`).
- `SelfForcingSource.stream`:
  - Runs the blocking rollout in an executor thread. Chunks cross into asyncio
    through an `asyncio.Queue(maxsize=4)` using `run_coroutine_threadsafe`
    (`selfforcing.py:585-627`).
  - Holds a process-wide `_shared_busy` lock so a second session cannot share
    the warmed model (`:586-602`).
  - Falls back to chunked `WanPipeline` clips if the causal rollout fails,
    unless I2V is active or the fallback is disabled (`:632-652`,
    `:665-680`).
  - Reuses the shared backend across sessions. It rebuilds only when the
    attention geometry changes, and calls `backend.reset()` otherwise
    (`:698-726`).
  - For I2V, it encodes the image to a latent and seeds it at t=0
    (`:764-785`).
  - **The first session's cold start is 421.7 s. Every later session is 0.5 s**
    (H100). The cold start splits as weights 139 s, pruna smash 21 s, compile
    warm 259 s (`strobe/CLAUDE.md:113-123`).

**LTX** (`strobe/src/strobe/sources/ltx.py`, `causal_ltx.py`, `causal_backend_ltx.py`, `ltx_schedule.py`)

- These are the same protocol and transport with LTX geometry:
  - VAE time factor 8 and spatial factor 32 (`causal_backend_ltx.py:14-17`);
  - scheduler shift read from the checkpoint (`:28-29`);
  - RoPE applied by complex rotation, with no RoPE on cross-attention
    (`causal_ltx.py:23-35`).
- `ltx_schedule.py` is a torch-free port of the diffusers linear-quadratic
  sigma schedule. The hard-coded `(1000,750,500,250)` used earlier gave a
  "faceless lump" (`ltx_schedule.py:12-56`).
- **Identity correction.** The configured model is LTX-Video 0.9.7-distilled
  (about 13B, T5 encoder), **not LTX-2**, and **it has no audio**
  (`strobe/CLAUDE.md:497-504`, `strobe/src/strobe/config.py:169`).
- **The causal LTX rollout is an "unsupported premise".** The checkpoints are
  bidirectional and were never distilled for autoregressive continuation, so
  "blocks render fine and nothing advances between them"
  (`strobe/scripts/batch/ltx_av.py:15-18`, `strobe/docs/timeline.md:970-971`).
- Rolling KV on LTX collapses quality within 30-60 s (`strobe/CLAUDE.md:545`).
- LTX-2.5 (22B) with joint audio is **batch only**:
  - H200: RTF about 1.0 per 5 s clip;
  - RTX PRO 6000: RTF 1.87;
  - A100 int8wo: RTF 2.69;
  - no causal path (`strobe/CLAUDE.md:550`).

### 1.5 Chunked VAE decode (`trt_vae.py`, `trt_vae_ltx.py`, `OverlapWanDecode`)

- **The seam and the fix.**
  - `AutoencoderKLWan` clears its temporal cache on every `decode()`, so
    decoding one block at a time leaves a visible jump at each boundary.
  - `OverlapWanDecode` prepends the last `overlap` latent frames of the
    previous block and keeps only the last `4·n` pixel frames
    (`strobe/src/strobe/sources/causal_backend.py:327-441`).
  - `VAE_OVERLAP=0` also loses 25% of its frames (`strobe/CLAUDE.md:202-207`).
  - The LTX variant does the same bookkeeping with factor 8. It is optional
    there because the LTX VAE has `decoder_causal=False`
    (`causal_backend_ltx.py:395-418`).
- **TensorRT scope.** TRT replaces only the innermost `latents→pixels` call
  (`trt_vae.py:17-20`).
- **Engine shapes.** There is one plan per `(GPU arch, T)`, named
  `vae_t{T}.plan`. The rollout needs `T=block` and `T=block+min(overlap,block)`
  (`trt_vae.py:45-71`).
  - A missing shape is a hard error, not a fallback (`:22-29`).
- **Build and runtime rules.**
  - Build: about 10 min of CPU export plus about 11 min per shape.
  - It needs TRT 10.x (`strobe/CLAUDE.md:245-261`).
  - A dedicated CUDA stream for TRT hangs the pipeline (`trt_vae_ltx.py:27-30`).

### 1.6 Audio (`audio_vae.py`) and how LTX audio is actually produced

- **`audio_vae.py` is a guard, not an audio path.**
  - `MIMI_FRAME_RATE_HZ = 12.5` (`strobe/src/strobe/audio_vae.py:13`).
  - `decode_mimi` always raises "Refusing a silent mute mux" (`:16-21`).
  - `require_audio_hooks` checks for the four hooks (`:24-39`).
  - Mimi was dropped because it handles speech only (`strobe/decision-log.md:5-13`).
- **No strobe code decodes LTX audio live.** The live LTX lane is 0.9.7, which
  has no audio branch.
- **LTX-2.5 audio** exists only in batch scripts:
  - `strobe/scripts/batch/ltx25_eval.py:288-306` muxes the pipe's `audio` at
    `pipe.vocoder.config.output_sampling_rate`;
  - the recorded clips carry AAC 48 kHz stereo (`strobe/docs/timeline.md:756`);
  - the official pipeline needs `ltx-2.5-audio-vae` weights
    (`strobe/scripts/batch/ltx25_official_eval.py:50`).
- **LTX 0.9.8 A/V** is video-to-audio with an MMAudio sidecar, not joint audio
  (`strobe/scripts/batch/ltx_av.py:1-13`).
- **A/V sync in strobe** means "the batch mux has an audio stream":
  - ffmpeg `-c:v copy -c:a aac -shortest` (`strobe/scripts/batch/sidecar-audio.py:81-99`);
  - `ffprobe` refuses a silent mux (`:57-78`).
- **No live A/V clock exists in strobe.** The per-block audio chunking in
  `run_causal_rollout` (§1.4) is exercised only by a numpy mock.

The fastvideo-rs LTX-2 port targets the real LTX-2 audio chain:

- the vocoder emits **24 kHz stereo**, 240 samples per 10 ms mel frame
  (`docs/ports/ltx2.md:136-141`, `crates/fastvideo-cudarc/src/ltx2/vocoder.rs:9-11`);
- LTX-2.5 BWE produces 48 kHz (`docs/ports/ltx25.md:4`).

### 1.7 FramePacer (`pacing.py`)

The jitter buffer between bursty generation and a steady WebRTC track
(`strobe/src/strobe/pacing.py:1-11`):

- **Drop-oldest on overflow.** `push_chunk` appends the burst, then pops from
  the front while `len > max_buffer` and counts `dropped` (`:61-81`).
  Latency stays bounded.
- **Adaptive-rate clock.**
  - `gen_ema = 0.3·(n/dt) + 0.7·gen_ema` over burst arrivals (`:70-76`).
  - It starts at `min_fps` so the buffer builds before playout catches up
    (`:54-59`).
  - `effective_fps()` is the fixed `fps`, or clamps the EMA to
    `[min_fps, fps]` in adaptive mode (`:83-88`).
- **Freeze on underrun.** `next_frame()` never blocks. It returns the next
  buffered frame, or repeats the last frame and counts `underruns`. It raises
  only if nothing has ever been pushed (`:93-106`).
- **The smoothness metric is `unique_fps = served − underruns`**, not the RTP
  rate (`strobe/CLAUDE.md:217-218`, `strobe/src/strobe/loopback.py:10-17`).
- **Defaults.**
  - `fps=16` is a ceiling, `adaptive_fps=true`, `min_fps=4`.
  - `buffer_frames=48` (about 3 s at 16 fps).
  - (`strobe/src/strobe/config.py:28,53-54,128`)

### 1.8 Track and encoder (`track.py`, `rtc_h264.py`)

- **`GeneratedVideoTrack` paces at `effective_fps()`.**
  - It overrides aiortc's fixed-30 fps `next_timestamp` and advances the 90 kHz
    RTP timestamp by `VIDEO_CLOCK_RATE/fps`.
  - If it falls behind it re-syncs rather than bursting
    (`strobe/src/strobe/track.py:111-139`).
  - `recv()` pulls `pacer.next_frame()`, builds an rgb24 `av.VideoFrame`, and
    never blocks (`:147-155`).
- **`FrameRecorder`** writes an optional MP4 on a background thread. An inline
  encode once capped playout at 22.6 fps (`:17-96`).
- **Encoder settings.** libx264 inside aiortc; **not NVENC**.
  - `rtc_h264.install_h264_keyframe_interval` monkeypatches aiortc's
    `H264Encoder` to set:
    - `gop_size = keyint = min-keyint = round(fps·2s)`, `scenecut=0`;
    - `tune=zerolatency`, `profile=Baseline`, `level=31`, `yuv420p`;
    - `bit_rate = self.target_bitrate`
      (`strobe/src/strobe/rtc_h264.py:28-30`, `:66-83`).
  - Without the patch libx264 uses a GOP of about 250 frames (about 15 s), and
    Cloudflare HLS joiners sit black waiting for an IDR (`:1-12`).
  - The bitrate is aiortc's adaptive default. INFERRED: strobe never sets it.
- **NVENC was measured and rejected.** Software encode is 2.64 ms/frame, about
  6% of a 22.8 fps budget (`strobe/CLAUDE.md:181`, `strobe/rust/README.md:16`).

### 1.9 WHIP publish (`whip.py`)

- **Outbound only**: POST an SDP offer, receive the answer, DELETE the resource
  on teardown. "This is what makes the GPU host portable"
  (`strobe/src/strobe/whip.py:1-6`).
- **Headers** (`:93-100`):
  - `Content-Type: application/sdp`;
  - `Authorization: Basic base64(user:token)` for MediaMTX internal auth;
  - `Authorization: Bearer <token>` for Cloudflare Realtime.
- **Codec order.**
  - `_prefer_h264` reorders the capabilities to H.264, then RTX, then the rest
    (`:25-58`).
  - Cloudflare ingests only H.264. aiortc offers VP8 first, which negotiates
    but plays black (`:26-32`, `strobe/CLAUDE.md:213-214`).
- **ICE is not trickled.** aiortc gathers every candidate inside
  `setLocalDescription`, so one POST carries the complete offer (`:117-128`).
  - A 200 or 201 carries the answer SDP.
  - The optional `Location` header is resolved against the POST URL and kept
    for `DELETE` (`:126-135`).
  - There is no `PATCH` (trickle ICE) and no ICE restart.
- **Timeout** is `whip_timeout_s=30`. Teardown is best-effort (`:136-152`).
- **Deploy targets.**
  - MediaMTX: WHIP at `:8889/strobe/whip`, WHEP at `:8889/strobe/whep`, and
    low-latency HLS with 1 s segments and 200 ms parts
    (`strobe/deploy/mediamtx/mediamtx.yml:1-30`).
  - The React viewer uses WHEP recvonly for video **and audio**
    (`strobe/deploy/ui/src/client/hooks/use-whep-player.ts:52-53`).

### 1.10 Loopback, TTFF, motion context, image input

- **`loopback.py`** is a CPU bench that needs no GPU:
  - it wires the real pacer, track and H.264 into an in-process aiortc
    publisher→receiver pair;
  - it drives a synthetic source at a chosen generation rate and reports
    `rtp_fps` against `unique_fps` (`strobe/src/strobe/loopback.py:1-26`, `:86-192`);
  - "smooth" means underruns ≤ 10% of served frames (`:68-75`).
  - **fastvideo-rs should have the same harness.**
- **`ttff.py`** is a module-level phase recorder: `weights`, `smash`,
  `backend`, `compile_warm`, `first_block`, `publish`
  (`strobe/src/strobe/ttff.py:36-43`).
  - It records one session at a time (`:17-20`).
  - Each mark names the phase that ends there. The recorder emits one log line
    and a stats snapshot (`:61-105`).
  - `publish` (WHIP plus ICE/DTLS) adds to TTFF; it does not overlap the model
    load (`strobe/CLAUDE.md:138-139`).
- **`motion_context.py`** implements "last video (and optional audio) tail
  between blocks":
  - `MotionContext{video_latent, audio_latent}` and `latent_tail`
    (`strobe/src/strobe/motion_context.py:20-49`);
  - it is the carry that `run_causal_rollout` KV-seeds on a full-window reset;
  - for the bidirectional FastWan it falls back to "last RGB frame → next clip
    `image_path`" (`chain_image_plan`, `:52-69`).
- **`image_input.py`** loads exactly one of `image_url` or `image_b64`
  (`strobe/src/strobe/image_input.py:21-28`).
  - URLs must be http(s) (`:40-51`). This is an SSRF-relevant check, INFERRED.
  - `preprocess_for_wan` centre-crops to the target aspect, resizes bicubic and
    returns `(1,3,1,H,W)` in `[-1,1]` (`:63-95`).

### 1.11 Serverless (`runpod_handler.py`)

- **One queued RunPod job is one stream session.** The worker dials out to WHIP,
  so no inbound port is needed (`strobe/src/strobe/runpod_handler.py:1-6`).
- **Job input and output.**
  - Input: `{"input": {prompt, whip_url, whip_token, duration_s,
    local_attn_size, sink_size, image_url|image_b64}}` (`:7-11`).
  - Output: `{id, state, stats}` or `{error, stats}` (`:39-54`).
- **Endpoint requirements.**
  - The endpoint must be queue-based, not load-balancing.
  - `executionTimeout` must be at least the maximum stream length.
  - Keep `workersMin ≥ 1` so the warm model survives between jobs
    (`:21-25`, `strobe/CLAUDE.md:79-80`).
- **Viewer URL.** The job returns only when the stream ends. The viewer URL is
  therefore the SFU's WHEP/HLS URL, known ahead of time. INFERRED from the
  handler awaiting `session.wait()` (`:51`).

---

## 2. strobe's Rust workspace

`strobe/rust/Cargo.toml:3` lists five members. The README states the Python pipeline
is the reference and the Rust port is **dormant**: "the throughput case is
measured away" (`strobe/rust/README.md:1-35`).

| crate | what it does | maturity | overlap with fastvideo-rs |
|---|---|---|---|
| **strobe-core** | Pure logic with no CUDA or tokio (`strobe/rust/crates/strobe-core/src/lib.rs:1-22`). `pacing.rs`: a verbatim port of the Python FramePacer, generic over the frame type `T` (a device handle is possible) and without async signalling (`pacing.rs:1-19`, `:91-140`). `clock.rs`: `Clock` trait, `MonotonicClock`, `ManualClock` for deterministic tests (`clock.rs:1-66`). `frame_math.rs`: Wan overlap bookkeeping (`frame_math.rs:1-40`). `rollout.rs`: a pipelined denoise/decode scheduler on `std::thread` plus a bounded `sync_channel` (`rollout.rs:1-73`, `run_pipelined` `:90`). `conditioning.rs`: the precomputed-text-embedding contract, 512×4096 UMT5 with the encoder off-box (`conditioning.rs:1-40`). | 28 tests, about 1000 LOC, no TODOs (my count). README: "fully tested ... WORKS" (`strobe/rust/README.md:52-54`). | **Direct port candidates.** fastvideo-rs has no pacer. Its SF-Wan KV cache and rollout live in `crates/fastvideo-cudarc/src/wan/causal.rs:1-18`. |
| **strobe-engine** | Engine *planning*: which TRT/AOT shapes a rollout needs (one VAE engine per T, one DiT engine per block position) (`strobe/rust/crates/strobe-engine/src/plan.rs:1-18`). `Backend`/`Engine` traits with no implementors (`lib.rs:1-40`; `strobe/rust/README.md:74-78`). | 8 tests. The shape is right, but nothing runs. | None. fastvideo-rs runs its own kernels and has no TRT. |
| **strobe-h3** | MiniMax-H3 *host* layer ported from antirez/h3.c: `host` (geometry, sigma schedules, RNG), `layout` (packed sequence), `schedule` (reuse mask, layer thinning, token reduction), `api` (MiniMax v2 request/response and validation), `config`, `plan` (the $0 preflight), and `fastvideo.rs` (`strobe/rust/crates/strobe-h3/src/lib.rs:1-34`). `cuda` feature: a device/memory probe only. | 113 tests (README says 102), about 8000 LOC. "NOT AN ENGINE — no kernels, no forward pass" (`strobe/rust/README.md:57-61`). | Overlaps FV's own H3 port (`crates/fastvideo-cudarc/src/h3/`). The `api.rs` v2 validation is reusable (§5.4). |
| **strobe-h3-server** (`h3serve`) | axum server exposing the MiniMax v2 routes, the v1 file routes, FastVideo's OpenAI `/v1/videos`, a key store, a task queue, HMAC-signed URLs, SSRF-guarded callbacks, and a console. It runs **Python FastVideo as a subprocess** (resident or per task) and "never touches a GPU" (`strobe/rust/crates/strobe-h3-server/src/lib.rs:1-35`, `Cargo.toml:7`). | 25 `#[test]` + 45 `#[tokio::test]`, about 8900 LOC. "built + tested for $0; never run on a GPU" (`strobe/rust/crates/strobe-h3/src/lib.rs:18`). | The closest thing to our *batch* server. Our engine would replace its subprocess. |
| **strobe-h3-vsa** | VSA-H3 host oracle plus an Ampere MMA `.cu` fine stage. States that its algorithm "matches `fastvideo-rs` `h3/vsa.rs`" (`strobe/rust/crates/strobe-h3-vsa/src/lib.rs:19`). | 12 tests. Correct on H200 (rel_l2 0.0007), but no clip-level gain over Triton (17.41 s against 17.42 s denoise) (`strobe/docs/timeline.md:1229-1241`, `:1270-1282`). | Derived from our work. Nothing to take back. |

### 2.1 Does `strobe-h3/src/fastvideo.rs` link fastvideo-rs? No.

- **What it is.** `fastvideo.rs` lowers a resolved plan onto a **command line
  for hao-ai-lab's Python FastVideo**.
- **Its reasoning.** "FastVideo is the compute backend, this crate is the host
  layer". A cudarc port "would be the same mistake as the Burn experiment"
  (`strobe/rust/crates/strobe-h3/src/fastvideo.rs:1-17`).
- **What it encodes.**
  - Eight maintained FastVideo recipes (`:58-75`).
  - The FastH3 Preview DMD ladder `[999,749,500,250]` (`:119`) and the 8-Step
    V2 ladder (`:123`).
  - Three traps:
    - FastVideo's `--steps` counts sigma-grid points, which is forwards + 1;
    - distilled checkpoints pin their own ladder;
    - the video sigma shift is per checkpoint: 12 for Preview, 10 for 8-Step V2
      (`:19-32`).
- **No code-level dependency.** There is no Cargo dependency, `use`, or FFI on
  fastvideo-rs (`strobe/rust/crates/strobe-h3/Cargo.toml:9-21`). The only mentions
  of fastvideo-rs are comments:
  - `strobe-h3-server/src/wan.rs:20-21, 96-97`: fastvideo-rs keeps the text
    encoder resident; FastWan peaks at 21.4 GB with the Wan VAE and 12.1 GB
    with TAEHV;
  - `strobe-h3-vsa/src/lib.rs:19`;
  - `strobe/scripts/measure/vsa-mma-launch.cu:4`.
- **The workspace also pins** `cudarc 0.19.8` with dynamic loading and `str0m
  0.21.0` (sans-IO WebRTC) (`strobe/rust/Cargo.toml:50-60`). No crate uses `str0m`;
  `strobe-media` is "NOT STARTED" (`strobe/rust/README.md:62`).

### 2.2 What to port, what to depend on

**Port:**

- **`strobe-core::pacing` and `strobe-core::clock`**, as a small internal
  module:
  - they are generic over `T` and runtime-agnostic, with tests ported from
    Python;
  - copy under MIT with attribution rather than depend on them. INFERRED:
    the crate is not published and pins edition 2024 and rust 1.85
    (`strobe/rust/Cargo.toml:18-22`). Check against our toolchain.
  - Add the two things it leaves out:
    - an async `Notify` for "first frame" (`pacing.rs:16-19`);
    - an **audio lane** (see §6).
- **`strobe-core::frame_math`**: take the idea (frames kept per overlap), not
  the Wan-only constants.

**Take as a design, but do not build yet:**

- **`strobe-core::rollout::run_pipelined`**, a two-stage overlap.
  strobe **measured it at 11% of the theoretical overlap on an H100**:
  4.3 ms recovered of 38.5 ms. "Pipelining is not worth building"
  (`strobe/CLAUDE.md:152-169`). Its own docstring says the speedup is "projected,
  not measured" (`rollout.rs:36-49`).
- Our decode is TAEHV, about 66 ms per block (INFERRED: 0.46 s / 7 blocks).
  So the overlap would buy even less here.
- **The bounded channel for backpressure is still the right shape**, because
  it keeps the media runtime off the compute thread (`rollout.rs:23-34`).

**Do not take:**

- **strobe-engine**: no implementors; TRT-specific.
- **strobe-h3 compute**: it has none.

**Reference for batch adapters:**

- **strobe-h3-server**: route set, error envelope, signed URLs, callback SSRF
  guard.

---

## 3. strobe h3fast: what bought the performance

### 3.1 Scope

- h3fast implements FastH3's *recipe*, not its weights, on a Wan 2.2 base.
- The recipe is 4-step DMD, VSA sparsity, and Motion-Context chaining of video
  and audio (`strobe/docs/specs/h3fast/PRD.md:25-31`).
- Part A is a batch generator. Live WHIP is deferred (`PRD.md:38-53`).
- The live path stays Self-Forcing T2V plus TRT (`strobe/CLAUDE.md:444`).

### 3.2 Best measured results (GPU named on each)

| result | model / config | geometry | GPU | number | source |
|---|---|---|---|---|---|
| **Live SF-Wan** | Wan2.1-1.3B Self-Forcing DMD, 4-step, 3-latent blocks, `torch.compile`d DiT + **TRT VAE**, `VAE_OVERLAP=1` | 832×480 | H100 (two independent cards) | **22.8 fps** delivered (block 503 ms: denoise 252 + context 64 + decode 187); ceiling 23.9 | `strobe/CLAUDE.md:82-97`, `:166-169`; `strobe/src/strobe/trt_vae.py:6-13` |
| Live SF-Wan, better seam | same, `VAE_OVERLAP=2` | 832×480 | H100 SXM | about 19 fps; RunPod warm 19.7 unique_fps, 0 underruns (PCIe: 12.8) | `strobe/CLAUDE.md:73-90` |
| Live SF-Wan + rolling KV | `LOCAL_ATTN_SIZE` finite, sink | 832×480 | H100 SXM, TRT, overlap 2 | about 21 fps over about 90 s, 0 underruns (first sample; off by default) | `strobe/CLAUDE.md:98-102` |
| Live causal LTX 0.9.7 (video only) | eager VAE, overlap 2, 4-step | 832×480 | H100 SXM | 22.58 unique_fps, capped by `FPS=24` (about 29 f/s generated). No prompted motion; do not ship | `strobe/CLAUDE.md:541-545` |
| **#76 B** | FastWan2.2-TI2V-5B-FullAttn, 3-step DMD, `torch.compile` + TORCH_SDPA | 832×480 / 15.04 s / 361 f | H100 SXM | `generation_time` 16.94 s → **21.31 fps, RTF 1.13**, peak 12.4 GB, video only | `strobe/docs/timeline.md:275` |
| #77 | #76 B + FA2 | same | H100 SXM | 21.52 fps / RTF 1.115 (a wash, about 1%) | `strobe/docs/timeline.md:276` |
| **#83** | #76 B + MMAudio `large_44k_v2` V2A sidecar | 15 s / 361 f | H100 SXM | video 21.72 fps / video_rtf 1.105; audio 2.03 s → audio_rtf 0.135; **e2e_rtf 1.240** | `strobe/docs/timeline.md:282` |
| #81/#82 | Wan2.2-TI2V-5B teacher, 40 steps, I2V ref-lock, `--chain-clips 2` | 832×480 / 5 s×2 | H100 SXM | CLIP frame0/mid/last 0.996/0.988/0.994; hand-off 0.998; RTF 10.96 | `strobe/docs/timeline.md:280-281` |
| **#87** | TI2V-5B teacher, 8-step + compile + MMAudio | 5 s / 121 f; 15 s / 361 f | H100 SXM | 5 s: gen 11.71 s, video_rtf 2.32, audio_rtf 0.329, **e2e 2.65**. 15 s: gen 45.54 s, video_rtf 3.03, **e2e 3.19** | `strobe/docs/timeline.md:286` |
| **H3 Preview + taeh3** | FastVideo FastH3 Preview v0.2, 4 forwards, VSA 0.9 Triton tile 64, resident, eager (no sm100a, no regional compile) | 1344×768 / 124 f / 5.167 s | **H200 NVL** | median **e2e 24.13 s / RTF 4.67**; denoise 17.4 s; taeh3 decode 1.0-2.3 s; audio decode 0.5-0.8 s | `strobe/docs/timeline.md:1243-1261`; `strobe/CLAUDE.md:477-484` |
| H3 Preview (serving default) | same, taeh3 | same | RTX PRO 6000 96 GB | e2e 36.76 s / RTF 7.1 / denoise 26.95 s | `strobe/docs/timeline.md:1258-1259` |
| H3 Preview, h3-vae decode | eager, Triton VSA, h3-vae | same | RTX PRO 6000 96 GB | e2e 64.97 s (denoise 26.7, **video decode 31.6**, audio 1.31) | `strobe/docs/timeline.md:1114-1127` |
| H3 Preview, staged load | `--lazy-module-load --h3-sequential-load` | same | H100 80 GB | e2e 153.13 s: 83.8 s of that is reloading the conditioner on every clip | `strobe/docs/timeline.md:1157-1183` |
| FastH3 (for comparison, IL) | sm100a VSA from source, regional compile, FA4, H3 fusions, replicated DiT, prompt padded to 256 tokens | 1344×768 / 345 f / 14.375 s | **4× B200** (8× B200: 12.9 s per 15 s) | **14.4 s per clip, 1.0× realtime** | `infinite-livestream/fast-h3/README.md:242-251`, `:341-364`; `infinite-livestream/fast-h3/fasth3.yaml:98-103` |

`RTF` in h3fast is FastVideo's `generation_time / clip_s`
(`strobe/scripts/batch/h3fast.py:56-58`, `:882-912`). It excludes model load. Wall
`generate_s` is larger, for example #73 12.10 s against 13.90 s
(`strobe/docs/timeline.md:272`).

### 3.3 Techniques: what mattered, by how much, and what fastvideo-rs has

| technique | what strobe did | measured effect | fastvideo-rs equivalent / port cost |
|---|---|---|---|
| **`torch.compile` of the DiT** | Pruna `compiler=torch_compile` (module_list), `compile_dynamic=true` (`strobe/src/strobe/config.py:108-116`, `strobe/src/strobe/sources/selfforcing.py:536-581`), plus `torch.compile(causal_forward)` with a warmup over every block position (`causal_backend.py:208-244`). FastWan batch: `enable_torch_compile=True` (`strobe/scripts/batch/h3fast.py:817-818`); serve yaml `compile.enabled: true, vae_enabled: false` (`strobe/scripts/batch/serve/fastwan-5b.yaml:16-23`). | Live SF-Wan: denoise 397→253 ms, **1.19× of which is compile alone** (`strobe/CLAUDE.md:92-94`, `:111`). #73 against #71 (5B, 704×1280): RTF 2.96→2.40 (**1.23×**). Cold compile costs 259 s once per process; later sessions start in 0.5 s (`strobe/CLAUDE.md:113-126`). Every non-compile DiT path was slower: TRT 62 ms, AOTI 70 ms, eager 77 ms against 41 ms (`strobe/rust/README.md:31-35`). | **No equivalent by construction.** FV runs hand-written cudarc kernels. The gap to close is kernel fusion and launch overhead: our SF-Wan denoise is 3.83 s / 7 blocks ≈ 547 ms per block (INFERRED, assuming 7 three-latent blocks for 81 frames) against strobe's 316 ms (denoise 252 + context 64) on H100. **The whole fps gap (≈18 fps against 22.8) is in the DiT, not the decode.** Levers we own: CUDA Graphs per block position (none in `fastvideo-cudarc` today; grep found only a `SupportsCudaGraphProbe` trait in `crates/fastvideo-models/src/techniques/technique.rs:172`), fused norm/modulation/RoPE, FP8 linears (`FASTVIDEO_WAN_QUANT`, `docs/ports/wan.md:288-294`). |
| **Attention backend** | Live: Pruna `kernel=flash_attn3`. Batch: `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` (`strobe/scripts/batch/h3fast.py:436`, `:964`). | **FA3 contributes nothing measurable** on live Wan (`strobe/CLAUDE.md:111`). FA2 on 5B: about 2% (#74) and about 1% (#77). FA3 did not build (#75). | FV has its own flash and block-causal flash kernels (`docs/ports/wan.md:323-345`; `FASTVIDEO_WAN_CAUSAL_FLASH` default on, `docs/scope.md:312`). No port needed. |
| **TensorRT VAE** | ONNX export → TRT 10 fp16 plan per `(arch, T)`, preloaded, hard error on a missing shape (`strobe/src/strobe/trt_vae.py:1-71`); `vae_dtype=float16` (`config.py:55-59`). | VAE call 320→166 ms (**1.93×**), block 698→503 ms, **16.7→22.8 fps (1.37×)** on H100 SXM; RTX 4090 showed 2.51×, which does not carry over (`trt_vae.py:6-12`). | **Superseded in FV by TAEHV.** FV decodes SF-Wan and DMD with TAEHV `taew2_1` by default: 0.46 s for the 81-frame SF-Wan clip (about 66 ms per block, INFERRED), already below TRT's 187 ms. Full VAE 2.97-3.14 s → TAEHV 0.31-0.34 s at LPIPS 0.03 / 32.8 dB (`docs/ports/wan.md:33`, `:83-87`; `crates/fastvideo-cudarc/src/wan/taehv.rs:1-19`). Nothing to port. For the full VAE, per-shape plans are irrelevant to cudarc. |
| **Latent-overlap chunked decode** | `OverlapWanDecode`: prepend `overlap` latent frames, keep `4·n` (`causal_backend.py:327-441`). | overlap 1 against 2: 22.8 against about 19 fps with TRT; eager 15.9 against 14.2 (`strobe/CLAUDE.md:86-96`). overlap 0 loses 25% of frames (`:206-207`). | **Needed for per-block streaming decode in FV.** For the full Wan VAE the same bookkeeping applies, or the causal feat-cache can be kept across blocks. INFERRED: FV's streaming decoder already runs 2 latent frames per pass with a feat cache (`docs/ports/wan.md:182`), so keeping its state *across* block calls may beat overlap. For TAEHV: INFERRED that its temporal memory is the only seam risk; test whether decoding one block at a time with carried TAEHV state is identical to a whole-clip decode. |
| **Rolling KV + sink** | `local_attn_size` finite + `sink_size` (`config.py:95-105`; `causal_wan.py:28-58`). | Wan: about 21 fps, 0 underruns, no 5 s reset seam (first sample). LTX: quality collapses in 30-60 s (`strobe/CLAUDE.md:98-102`, `:545`). | **Exists**: FV's causal Wan implements `local_attn_size` eviction after `sink_size` sink frames, and the block-causal kernel takes sink tiles (`crates/fastvideo-cudarc/src/wan/causal.rs:1-18`, `docs/ports/wan.md:331`). The server needs to expose it per session and run until stopped, not stop at 81 frames. |
| **Windowed sparse KV probe** (`sparse_window`, not trained VSA) | Gather a KV window at inference (`causal_wan._sparse_gather`). | **+22%** eager fps (7.57→9.25) but mean abs pixel error 18, max 255: "not a free toggle" (#66, `strobe/docs/timeline.md:265`). | Do not port. |
| **VSA** | 5B FullAttn + VSA 0.8 → missing `to_gate_compress`, illegal instruction (#72). H3: VSA 0.9 tile 64; sm100a only on B200/B300, Triton elsewhere (`strobe/scripts/batch/serve/fasth3-preview.yaml:24-26`). | VSA must be trained in. On IL 4×B200 the sm100a route is about 2.5× faster than Triton (`infinite-livestream/fast-h3/README.md:349-357`). | FV has H3 VSA (`strobe-h3-vsa` says it matches FV `h3/vsa.rs`). Our FastH3 is 26.5 s on RTX PRO 6000 and 16.1 s on B200, against strobe's PRO 6000 serving default of 36.76 s and H200 24.13 s. **We are ahead of strobe on H3.** Clip lengths may differ, so INFERRED comparison. |
| **Motion-Context block chaining (video + audio)** | Rollout: `motion_context_frames` KV-seeds the prior latent tail (and the audio tail) on reset (`causal.py:302-318`). Batch: `--chain-clips N` saves the last RGB frame as the next clip's `image_path` (`strobe/scripts/batch/h3fast.py:870-909`). | Teacher chain-2: hand-off CLIP 0.998, RTF 10.96 (#82). FullAttn 5B ignores `image_path` (#79, FastVideo #711); the harness refuses it (`h3fast.py:519-521`, `:793-797`). | Rollout side: our KV cache is already continuous within a session. Clip side: FV has TI2V-5B I2V (`docs/ports/wan.md:188`), so last-frame → next-clip I2V chaining is a server-level loop. It is **not valid for FastWan FullAttn (T2V only)**. For H3, use the `fl2va` recipe. |
| **MMAudio sidecar** | V2A on a finished clip: `large_44k_v2` @`974010a`, Euler flow matching, `cfg_strength=4.5`, prompt needs `<S>…<E>` speech + `Audio:` line; FLAC at the model rate, muxed with `-c:v copy -c:a aac -shortest` (`strobe/scripts/batch/sidecar-audio.py:23-54`, `:81-99`, `:201`, `:232`, `:241`). | audio_rtf **0.135** on 15 s (2.03 s), 0.33 on 5 s; lifts #76 B e2e from 1.105 to 1.240 (#83, #84, #87). | **Exists**: `crates/fastvideo-cudarc/src/mmaudio/` (T2A/V2A, synchformer, `docs/ports/mmaudio.md`). The server work is running it per clip after video decode, or per window in streaming (§6.3). |
| **Chain-clips with ref-lock** | TI2V teacher `--image-path` → clip 0; last frame → clip k+1 (`h3fast.py:584-601`, `:888-897`). | ref-lock green only on the 40-step teacher (RTF about 11); 4- and 8-step truncations freeze motion (LPIPS 0.0006) (#86-#90). The distilled I2V student is not yet accepted (#93/#94). | Same as the Motion-Context row. |
| **Pruna** | Live: `torch_compile` + `flash_attn3`, optional quantizer (`config.py:107-116`). Batch: blocked on the torch 2.11 against 2.12 mismatch (#85). | Everything Pruna contributed live was the compile (`strobe/CLAUDE.md:111`). | N/A. |
| **Geometry choice** | #76 B at 832×480 against A at 1344×768 on the same 5B checkpoint. | RTF 1.13 against 2.83 (#76). The biggest single lever in h3fast. | Server policy: default the streaming canvas to 480p for real-time lanes. |
| **Process-resident model** | `REUSE_MODEL=true`; `h3fast-worker.py` keeps each arm warm (`strobe/scripts/batch/h3fast-worker.py:1-22`); strobe-h3-server `resident` gives about 40 s per task against about 250 s (`lib.rs:30`). | TTFF 421.7 s → 0.5 s (843×). | The FV server must keep engines resident and warm per canvas and frame count before a session is admitted. |
| **Prompt padding (IL)** | Pad prompts to 256 tokens so regional compile never recompiles (`infinite-livestream/fast-h3/README.md:358-364`). | Avoids about 23 s of recompile per clip. | Not needed for cudarc; we have no shape-keyed compile. INFERRED. |
| **Rejected** | DiT TRT, AOTI, C++ host, Burn, dedicated TRT stream, NVENC, pipelining. | `strobe/CLAUDE.md:171-188` | – |

**Comparison with our current numbers.** These rows mix geometry and GPU, so
treat them as INFERRED.

- **SF-Wan 1.3B, 81 frames.**
  - Ours: 4.41 s on H100 (denoise 3.83 + TAEHV 0.46), about 18.4 fps.
  - strobe: 22.8 fps on H100.
  - Our decode is already faster than their TRT. **The ~24% throughput gap is
    all in the denoise and context stages.**
- **FastWan.**
  - Ours: 1.3B, 480×832×81 in 2.32 s on RTX PRO 6000 (about 35 fps).
  - strobe #76 B: the 5B, 832×480 at 21.31 fps on H100.
  - Different checkpoints, not comparable. Wan2.2 TI2V-5B is ported in FV
    but its speed is not measured; #76 B is the bar to beat.
- **FastH3 4-step VSA 768p.**
  - Ours: 26.5 s on RTX PRO 6000; 16.1 s on B200.
  - strobe: 36.76 s on RTX PRO 6000; 24.13 s on H200.
  - IL: 14.4 s for a 14.375 s clip on 4× B200 with sequence parallelism.
  - Only IL reaches 1.0× realtime, and it takes 4-8 B200s.

### 3.4 Batch wire formats strobe implements

**FastVideo OpenAI `/v1/videos`**

- **Serving.** Upstream `fastvideo serve --config <yaml>` resolves the config
  from a catalog (`strobe/scripts/batch/fastvideo-serve.sh:20-50`,
  `strobe/scripts/batch/fastvideo-catalog.yaml:5-75`).
  - Catalog ids: `fastwan-5b` (default; t2v; 832x480, 361 frames, 3 steps),
    `wan22-ti2v-5b`, `ti2v-dmd-student`, `fasth3-preview` (t2va/fl2va,
    1344x768, 124 frames, 5 grid points), `fasth3-8step`.
  - H3 rows are `licensed: false` unless `STROBE_H3_LICENSE_OK`.
- **Routes** (reproduced in Rust in
  `strobe/rust/crates/strobe-h3-server/src/fastvideo_api.rs:1-24`, `:55-68`):
  - `GET /health`, `GET /v1/models`, `GET /v1/models/{model}`,
    `GET /v1/model_info`;
  - `POST /v1/videos` (also `/v1/videos/generations`), `POST /v1/videos/sync`,
    `GET /v1/videos`, `GET|DELETE /v1/videos/{id}`,
    `GET /v1/videos/{id}/content`.
- **Status mapping**: queued → `queued`, running → `in_progress`, succeeded →
  `completed` (`:743-745`).
- **Response body** (`:767-788`): `{id, object: "video", status, size: "WxH",
  seconds, file_name, completed_at, inference_time_s, …}`.
- **Request fields**: `model`, `prompt`, `seconds` (int or digit string),
  `size`, `num_frames`, `image_reference`, `extra_params` (`:78-150`,
  `:421-481`).
- **Deliberate differences from upstream.** Upstream accepts requests with no
  key; strobe requires a bearer key. `num_inference_steps` keeps FastVideo's
  grid-point meaning. Fields the engine cannot honour are *refused*, not
  dropped (`:13-24`).

**MiniMax v2** (Python adapter `strobe/scripts/batch/minimax-v2-server.py`; Rust `strobe-h3-server/src/routes.rs:51-74`)

- **Create.** `POST /v2/video_generation` with body
  `{model: "MiniMax-H3"|"MiniMax-H3-Max", content: [{type: text|image_url|video_url|audio_url, text?, image_url:{url}?, role?, seconds?}], resolution: "480P"|"768P"|"2K", duration: 4..15, ratio: "adaptive"|"21:9"|"16:9"|"4:3"|"1:1"|"3:4"|"9:16", extra?, callback_url?}`
  (`minimax-v2-server.py:58-75`).
  - The reply is `{task_id, base_resp: {status_code: 0, status_msg: "success"}}`
    (`:390-393`).
  - The Python adapter rejects `mm_file://`, `video_url` and `audio_url`
    (`:128-155`).
  - Image roles map to a FastVideo `task`: none → `t2va`; first+last or first →
    `fl2va`; `reference*` → `ref2va` (`:158-166`).
  - Size: short edge 480/768/1440, long edge rounded to a multiple of 32,
    `adaptive` → 16:9 (`:169-192`).
- **Lowering to FastVideo.** `POST {fastvideo}/videos` with
  `{model: <catalog id>, prompt, seconds: str(duration), size, task,
  image_reference: [{image_url}]}` (`:369-377`).
  - A non-H3 model returns 400 with `param: model` (`:304-317`).
  - H3 requires the license flag, or 403 (`:319-323`).
- **Query.** `GET /v2/query/video_generation/{task_id}` polls FastVideo
  `GET /videos/{id}`. The reply is `{task: {id, model, status:
  queued|running|succeeded|failed, created_at, updated_at, content: {url},
  resolution, duration, usage, ratio, task_type: "generation", modality:
  "video", error}, base_resp}` (`:395-440`).
- **The Rust server adds**: delete, list, the v1 files
  (`upload|list|retrieve|retrieve_content|delete`), and
  `/strobe/v1/{capabilities,tasks,files,admin/keys}` (`routes.rs:51-74`).
  - Supported surface: text-to-video on open H3 weights only. H3-Max, 2K,
    first/last frames and references are refused at create time (`lib.rs:11-16`).
- **Documented v2 limits**: 7000 characters of text, 4-15 s, at most 9
  reference images, 3 clips, 15 s of reference media, a 64 MB body
  (`strobe/rust/crates/strobe-h3/src/api.rs:12-14`).

---

## 4. infinite-livestream

### 4.1 The fast-h3 Reactor model contract

- **Model and hardware.** FastH3 Preview v1 (MiniMax-H3 35B distilled to 4
  forwards, 90% VSA) on the Reactor Runtime, 8× B200 (`infinite-livestream/README.md:12-34`;
  `infinite-livestream/fast-h3/reactor.yaml:44-45`).
- **The unit of work is a whole clip, not a frame.** That is why it subclasses
  `ReactorModel` with its own `run()` loop (`infinite-livestream/fast-h3/fasth3.py:15-19`).

**Tracks** (`infinite-livestream/fast-h3/fasth3_types.py:30-34`)

- `FastH3Output(Output)` declares `main_video: Video` and `main_audio: Audio`.
- Every emit carries both, sliced together (`fasth3.py:931-932`).
- How the runtime turns this into WebRTC m-lines: `research-reactor.md` §4.5.

**Commands** (`@event`)

| command | fields | notes |
|---|---|---|
| `enqueue` | `prompt` (≤800), `metadata` (≤2000), `seed?`, `seconds?`, `position?` | Adds to the generation queue. The reply is `clip_queued` (`fasth3.py:304-389`). |
| `play` | `clip_id?` | Takes the playout front or the named clip (`:403-441`). |
| `pop` | `clip_id` | Removes the clip from either queue. A build in flight for it is discarded (`:455-`). |
| `move` | `clip_id`, `position` | Reorders within the queue that holds the clip. |
| `stop` | – | Cuts to black. |
| `set_clip_seconds` | – | Snapped to 5.167-14.375 s. |
| `set_seed`, `set_autoplay` | – | – |
| `set_canvas` | `aspect` | 16:9, 1:1, 9:16 or 4:3. Only when both queues are empty and nothing plays. |
| `reset`, `get_queue`, `get_state` | – | – |

The command table is `infinite-livestream/fast-h3/README.md:166-178`.

**Messages** (`fasth3_types.py:67-307`)

- `state_update` carries the full snapshot, including the `valid_commands` list.
- `queue_update` carries both queues in full.
- Per-clip messages: `clip_queued`, `clip_generated`, `clip_moved`,
  `clip_started`, `clip_finished` (+`seconds_sent`), `clip_stopped`,
  `clip_popped`, `clip_failed` (+`reason`).
- Other replies: `clip_length_accepted`, `seed_accepted`, `autoplay_accepted`,
  `canvas_accepted`, `session_reset`, `command_error{command, reason}`.
- Every clip-referencing message embeds the whole `ClipInfo{clip_id, prompt,
  metadata, frames, seconds, seed, ready}` (`:37-64`).

**Queue-and-playout semantics** (`infinite-livestream/fast-h3/fasth3_queue.py`, `fasth3.py:750-933`)

- **Two bounded `ClipQueue`s**: generation (prompts) and playout (built clips,
  held in host RAM). Each supports insert-at-position and move
  (`fasth3_queue.py:88-176`).
- **The playout bound is also the host-memory budget.** About 1 GB per 14 s
  clip at 16:9; `queue_size: 10` (`infinite-livestream/fast-h3/fasth3.yaml:21-26`).
- **One build is in flight at a time.**
  - `_pump_builds` submits the front non-building entry only when the playout
    queue is not full. That is the submit-time reservation
    (`fasth3.py:796-854`).
  - A finished build whose entry was popped is discarded silently
    (`:813-814`).
  - Builds run only while a client is connected (`:761-768`).
- **Autoplay is a standing `play`** (`:772-785`).
- **`_emit_clip` is a strict 24 fps metronome** (`:886-933`):
  - it emits 3 frames per slice (`EMIT_FRAMES`, `:84-87`);
  - the audio slice is `samples[:, round(lo·spf) : round(hi·spf)]` with
    `spf = 48000/24 = 2000` (`:905`, `:916-917`);
  - it paces by frames and re-anchors instead of bursting to catch up
    (`:919-926`);
  - it calls `_pump_builds` on every slice so the next clip keeps building
    (`:910`);
  - the model pins `fps = 24` and `buffer_size = 48`, which is 2 s of
    transport tolerance (`:97-105`).
- **After every clip**: `output.flush()` holds black until the next play
  (`:866-869`).
- **Session rules.** `valid_commands` is derived purely from four state
  fields, e.g. `set_canvas` is allowed only when everything is empty
  (`infinite-livestream/fast-h3/fasth3_session_rules.py:12-46`).

**Audio production** (`infinite-livestream/fast-h3/fasth3_backend.py:31-34`, `:540-577`)

- The checkpoint's audio decoder runs at **32 kHz**.
- `_to_wire_audio` then:
  1. transposes to channel-major;
  2. resamples to **48 kHz** with `torchaudio.functional.resample`;
  3. **downmixes to mono**, because "the runtime recorder flattens two channels
     by concatenation";
  4. trims or pads to exactly `round(frames/24·48000)` samples;
  5. converts to int16.
- The result is A/V lock by construction: whole clips, sample count tied to
  frame count.

**Clip geometry** (`infinite-livestream/fast-h3/fasth3_clip_plan.py:18-45`)

- 24 fps only. Frame counts are `17n+5` in 5-15 s, so 124 to 345 frames:
  15 s aligns to 362 frames, which exceeds the cap (`fasth3.yaml:14-19`).
- The canvas short edge is 768 with max area 768×1344.
- Warmup builds every legal length (14 of them) at load, so no mid-session
  compile stall (`fasth3.yaml:76-88`).

**Hard cuts.** "Every clip is generated independently. There is no continuity of
subject, framing, or voice ... and the stream holds on black between plays"
(`infinite-livestream/fast-h3/README.md:263-269`).

**Timing on 4× B200**

- 14.4 s per 14.375 s clip (1.0×).
- Play-to-first-frame 0.22-0.25 s; stop-to-black about 0.13 s.
- Load-to-serving about 3.5 min (`infinite-livestream/fast-h3/README.md:341-347`).

### 4.2 The client: pacer, director, sinks

- **`ModelLink`** is the single abstraction (`infinite-livestream/streaming-client/model_link.py:1-25`).
  - It mirrors `state_update` and `queue_update`, fans out messages, and
    exposes `send_command` and the media path.
  - The broadcast timing is fixed at **24 fps / 48 kHz** (`:39-42`).
  - `generates_audio` tells the upsampler whether to write dialogue and a
    soundscape (`:60-62`).
- **`Pacer`** is a constant-rate A/V clock between the model and one sink
  (`infinite-livestream/streaming-client/pacer.py:1-27`).
  - **Video**: a `deque(maxlen=fps·2 s)` that drops the oldest frame and counts
    it (`:73-74`, `:92-99`). Frames of the wrong size are letterboxed, never
    resized (`:113-127`).
  - **Audio**: int16 chunks capped at 2 s, dropping the oldest (`:101-111`).
    Exactly `sample_rate/fps` samples per tick, padded with silence
    (`:129-149`). The rate must divide by fps (`:62-65`).
  - **Tick**: pop a frame or repeat the last one (black before any frame
    arrives), then send one tick of audio. It resnaps the clock after 8 missed
    periods rather than bursting (`:49`, `:153-204`).
  - **Sync is structural.** Both buffers have the same shallow cap, so while a
    clip plays both stay near empty. While idle, both run dry and emit
    repeats plus silence (`:17-22`).
  - The pacer outlives Reactor reconnects, so the platform stream never breaks
    (`:23-26`).
- **Director**: chat prompt → moderation → LLM upsampler → a *scene group* of
  1..N clips enqueued contiguously (`infinite-livestream/streaming-client/director.py:1-37`).
  - Group identity rides in the opaque `metadata` echo.
  - Viewer groups are inserted ahead of filler with `position`. An idle filler
    tops the queue up.
  - Playout order is curated with `move` while autoplay (set by the link)
    chains clips with millisecond gaps (`:282-292`,
    `infinite-livestream/streaming-client/reactor_link.py:144-149`).
- **Stitching and audio continuity across clips.** There is none beyond
  adjacency:
  - clips follow each other back to back through autoplay;
  - each clip's audio is sample-exact to its own frames;
  - any gap is filled by the pacer with the last frame and silence.
  - INFERRED: no crossfade, no carried audio state, no Motion-Context. At a
    clip boundary the audio simply steps from one waveform to the next.
- **Sinks** share one ffmpeg process design (`infinite-livestream/streaming-client/sinks/_ffmpeg.py:1-33`).
  - Input: rgb24 frames on stdin and s16le PCM on a second inherited pipe.
    Each pipe has its own writer thread and a bounded drop-oldest queue, so the
    event loop is never blocked.
  - Encode (`:173-198`):
    - `libx264 veryfast zerolatency yuv420p`, a GOP of `fps·keyframe_s`, CBR
      `-b:v/-maxrate/-bufsize`;
    - AAC 128k 44.1 kHz stereo.
  - The process is restarted lazily when it dies.
  - **RTMP**: FLV, 2 s keyframes. **An audio track is mandatory**; YouTube and
    Twitch reject video-only (`infinite-livestream/streaming-client/sinks/rtmp.py:1-24`).
  - **HLS**: 1 s keyframes and segments, `sc_threshold 0`, a rolling window,
    epoch numbering (`infinite-livestream/streaming-client/sinks/hls.py:9-16`, `:54-60`).
  - The pacer is built with `AudioFormat(48000, channels=1)`
    (`infinite-livestream/streaming-client/main.py:206-207`).

### 4.3 The three backend links: the exact calls

**`ReactorLink`** (`infinite-livestream/streaming-client/reactor_link.py`)

1. **Connect.**
   - Local: `Reactor(model, local=True, api_url=local_url)`.
   - Hosted: `Reactor(model, api_key=…)` (`:103-110`).
2. **Register callbacks.**
   - `reactor.on("message", …)` and `reactor.on_status(…)`.
   - Media callbacks are registered **before** `connect()`, by wire name:
     `reactor.track("main_video").on_frame(...)` and
     `reactor.track("main_audio").on_frame(...)` (`:113-120`).
3. **Start the session.**
   - `await reactor.connect()` (`:126`).
   - `send_command("get_state", {})` with a 30 s timeout (`:133`).
   - `send_command("set_autoplay", {"enabled": True})` (`:149`).
4. **Commands** go through `reactor.send_command(command, data)`. Replies are
   unwrapped from `{"type","data"}` (`:36-40`, `:60-75`).
5. **Media.** Video frames go to `pacer.submit_video`. Audio frames go to
   `pacer.submit_audio`, with a warning if the rate is not 48000 (`:211-222`).
6. **Reconnect** every 5 s. Queue contents die with the server-side session.
   - Local mode only: `POST {local_url}/stop_session` clears an orphaned session
     (`:79-100`, `:186-207`).

**`FalLink`** (`infinite-livestream/streaming-client/fal_link.py`), default model `minimax/h3-max/text-to-video` on `https://queue.fal.run`

- **Configuration** (`infinite-livestream/streaming-client/config.py:258-274`):
  - `FAL_RESOLUTION=768P`, `FAL_ASPECT_RATIO=16:9`;
  - 5-15 s whole seconds;
  - `FAL_CONCURRENCY=3`;
  - `FAL_PROMPT_EXPANSION=disabled`.
- **HTTP.** Every request carries `Authorization: Key <FAL_KEY>` (`:373`).
- **Submit.** `POST {base}/{model}` with body `{prompt, duration: <int seconds>,
  resolution, aspect_ratio, seed, prompt_expansion_mode?}` (`:197-198`,
  `:533-542`).
  - The reply carries `request_id`, `status_url`, `response_url` and
    `cancel_url`.
  - Only URLs on fal's own host are followed. Otherwise the link builds
    `{base}/{app}/requests/{id}[/status|/cancel]` (`:200-214`, `:543-554`).
- **Poll.** `GET status_url` every 2 s (`:88`, `:556-571`).
  - Pending states: `IN_QUEUE`, `IN_PROGRESS`. Done: `COMPLETED` (`:107-108`).
  - Timeout 900 s, then cancel (`:96`).
- **Fetch.** `GET response_url`, then take `video.url` (or `videos[]`) and
  download it from the CDN without the key (`:572-579`, `:787-800`).
- **Error classes.**
  - 400/413/415/422 fail the clip.
  - 401/403 are auth errors.
  - Anything else counts as "unreachable": the clip stays queued (`:102-104`,
    `:586-620`).
- **Order.** Builds run in parallel, but a clip crosses into playout in
  submission order (`:17-23`, `:476-481`).
- **Playout** (`:683-773`):
  - ffmpeg decodes each MP4 to `fps=24,scale…force_original_aspect_ratio=decrease,pad`
    rgb24;
  - the audio is decoded once to **48 kHz mono s16le** with `-ac 1 -ar 48000`;
  - 3-frame slices go to the pacer on a re-anchoring clock, with
    `2000·count` samples per slice;
  - a clip with no audio decodes to an empty array, and the pacer pads it with
    silence.

**`FastWanLink`** (`infinite-livestream/streaming-client/fastwan_link.py`)

The server is the FastWan Video API. Its docstring disagrees with
`model_link.py:22-24`, which calls it "FastVideo's OpenAI-compatible video-job
API". The code uses the routes below.

- **HTTP.** `Authorization: Bearer <key>` when a key is set (`:330`).
- **Health.** `GET /health` until `model_loaded` is true (`:356-361`).
- **Submit.** `POST /generate` with body `{prompt, width, height, num_frames,
  fps, seed}` (`:458-466`). The reply's `prompt_id` is the job id (`:467-469`).
- **Poll.** `GET /status/{id}` every 1 s while the status is `queued` or
  `processing`, until `completed` or `failed` (`:80`, `:91-92`, `:470-478`).
- **Fetch.** `GET /video/{id}` downloads the MP4 (`:480-483`), then
  `DELETE /video/{id}` (`:486`).
- **Frame counts** follow `4k+1` (`:74-75`), 49-121 frames at `FASTWAN_FPS=24`.
- **No audio.** `generates_audio = False` (INFERRED from `:25`); the pacer sends
  silence (`:22-25`).

---

## 5. Comparison of the three delivery models

| | **WHIP publish** (strobe) | **Reactor-style peer WebRTC** (fast-h3 on the Reactor Runtime) | **Clip-queue playout** (IL links to fal / FastWan / a Reactor model) |
|---|---|---|---|
| Topology | GPU → SFU (MediaMTX / Cloudflare) → WHEP/HLS viewers. Fan-out belongs to the SFU. | GPU ↔ one client over a peer connection; commands on the same session (`research-reactor.md` §3.3). One session is one audience. | Generator makes whole clips. A *playout* process runs pacer → encoder → RTMP/HLS/WebRTC. The generator can be remote and serverless. |
| Inbound ports on the GPU | **None**. WHIP is outbound POST/DELETE (`strobe/src/strobe/whip.py:1-6`). | The runtime serves HTTP signalling and ICE, so it needs a reachable host or a TURN relay (`research-reactor.md` §3). | None for fal and FastWan: the client polls a job API. |
| Serverless | **Yes**: one RunPod queue job is one stream (`strobe/src/strobe/runpod_handler.py:1-25`). Job timeout must be at least the stream length; cold start is about 7 min without a warm worker (`strobe/CLAUDE.md:79-80`). | Not serverless. A long-lived pod (`infinite-livestream/fast-h3/reactor.yaml`), a coordinator, and session reaping. | **Yes, naturally**: every build is a short job (fal queue, FastVideo `/v1/videos`). Playout is a cheap CPU box. |
| Time to first frame | Model TTFF (0.5 s warm, 422 s cold) + WHIP publish (`strobe/CLAUDE.md:113-139`). | Transport only for a pre-built clip (0.22-0.25 s play-to-first-frame); a new prompt waits for one clip build. | New content arrives after one full clip build plus queueing (about 14-15 s on 4× B200). The platform adds its own latency for RTMP/HLS. INFERRED: several seconds for HLS with 1 s segments. |
| Latency from prompt change to screen | Next block boundary (`PATCH /sessions`, `strobe/CLAUDE.md:103-105`). About 0.5-0.75 s per block plus pacer depth (INFERRED from the 503 ms block and a 48-frame cap). | Same as clip-queue for fast-h3 (clip granularity). A causal model could do block granularity over the same transport. | Clip granularity: ≥ 5 s clips, plus a queue ahead. |
| Throughput requirement | Generation ≥ playout fps. Otherwise the adaptive pacer lowers fps (freezes are avoided but motion slows). | fast-h3: build ≥ 1.0× realtime, or black gaps between clips (`infinite-livestream/fast-h3/fasth3.yaml:98-103`). | Same. The director's filler hides shortfalls with more clips. |
| Audio handling | **None live.** One video track (`strobe/src/strobe/session.py:176`). The WHEP viewer is ready to receive audio (`use-whep-player.ts:53`). | The model declares `main_video` + `main_audio` (`infinite-livestream/fast-h3/fasth3_types.py:30-34`). Audio is sample-sliced per video slice at 48 kHz mono int16 (`fasth3.py:905-932`). | Per-clip A/V from the MP4. The pacer holds A/V lockstep per tick; silence when there is no audio. RTMP requires an audio track. |
| Failure isolation | Media and generation share one asyncio loop. Starvation appears only during compile warm (2.4 s stalls), which is when ICE negotiates (`strobe/CLAUDE.md:192-198`). | The runtime separates media from model. The model loop "must survive anything" (`fasth3.py:750-795`). | The pacer and sink outlive generator reconnects (`infinite-livestream/streaming-client/pacer.py:23-26`). |
| Best fit | Real-time causal video (SF-Wan). | Interactive single-viewer sessions with commands (Reactor protocol parity). | Non-causal clip models with joint audio (H3, LTX-2.x, fal H3-max), and 24/7 broadcast. |

---

## 6. Recommendation for the fastvideo-rs server

### 6.0 Architecture in one picture

```
session (protocol front-end: Reactor | WHIP-publish | clip-queue API | fal-director)
   │ SessionSpec { model, tracks, canvas, fps, sample_rate, mode: causal|clips }
   ▼
Generator (dedicated OS thread, owns the CUDA context)
   ├─ CausalSource  : per-block frames (+ optional per-block audio)   [SF-Wan]
   └─ ClipSource    : whole clips (frames + waveform)                 [H3, LTX-2, FastWan]
   │ bounded channel (backpressure, never async on the GPU thread)
   ▼
AvPacer (strobe-core semantics + IL A/V lockstep)
   │ one tick = 1 video frame + sample_rate/fps audio samples
   ▼
Encoders (H.264 video; Opus for WebRTC / AAC for RTMP-HLS)
   ▼
Transports: WHIP publish | Reactor peer (str0m) | RTMP/HLS via ffmpeg | WHEP via SFU
```

### 6.1 (a) Real-time causal, video only (SF-Wan)

- **Source.**
  - Run the SF-Wan block loop open-ended: rolling KV with `local_attn_size` and
    `sink_size`, already in `crates/fastvideo-cudarc/src/wan/causal.rs:1-18`.
  - Emit **one block of frames at a time**, 12 pixel frames per 3 latent
    frames, instead of one 81-frame clip.
  - The TAEHV decode must carry state across block calls. Otherwise use
    strobe's latent overlap with the `4·n` keep rule
    (`strobe/src/strobe/sources/causal_backend.py:393-416`).
  - Add a CPU test that per-block decode equals whole-clip decode. This is the
    `overlap=0` trap strobe documents (`strobe/CLAUDE.md:202-207`).
- **Threading.**
  - Generation runs on its own thread. Blocks cross to the async side through a
    bounded channel of depth about 4 (`strobe/src/strobe/sources/selfforcing.py:606`,
    `strobe/rust/crates/strobe-core/src/rollout.rs:23-34`).
  - Keep tokio and the media stack off the CUDA thread.
- **Pacer.**
  - Port `strobe-core::pacing` and `clock`: drop-oldest, freeze on underrun,
    adaptive EMA 0.3/0.7 clamped to `[min_fps, fps]`, `unique_fps` stats.
  - Pace the RTP timestamp at `effective_fps()` the way `track.py:111-139`
    does.
  - Defaults: `fps` 16 for Wan2.1 with the ceiling raised when generation
    allows, `buffer_frames` about 3 s, `min_fps` 4
    (`strobe/src/strobe/config.py:28,53-54,128`).
- **Encoder.**
  - H.264 Constrained Baseline, level 3.1, `zerolatency`, **IDR every 2 s**, no
    scene-cut IDRs (`strobe/src/strobe/rtc_h264.py:66-83`).
  - Offer H.264 first (`strobe/src/strobe/whip.py:25-58`).
  - Software x264 or openh264 is enough: strobe measured 2.64 ms/frame
    (`strobe/CLAUDE.md:181`). NVENC is optional.
- **Admission and TTFF.**
  - Keep the engine resident and warm before accepting a session. Answer
    429 while a session is *starting* (`strobe/src/strobe/server.py:19-24`).
  - Do the transport handshake **after** the first block exists
    (`strobe/src/strobe/session.py:168-177`).
  - Start the duration clock at the first frame (`:115-128`).
  - Report a TTFF phase breakdown (`strobe/src/strobe/ttff.py:36-43`).
- **Performance work.**
  - Our 18.4 fps against strobe's 22.8 fps is a DiT gap, about 547 against
    316 ms per block (INFERRED, §3.3). Decode is already better.
  - Priorities: CUDA Graphs per block position, then fusion and FP8.
  - Resolution is the other lever: #76 B against A was 2.5× (§3.3).
- **Build a loopback bench** equal to `strobe/src/strobe/loopback.py`: synthetic
  source → pacer → encoder → local receiver, reporting `unique_fps` against a
  set generation rate. It needs no GPU.

### 6.2 (b) Video+audio, real time or near real time

**Evidence first**

- No reference project has a *causal* audio-video generator in production.
- strobe's causal LTX lane is video-only 0.9.7, and causal rollout on LTX is an
  unsupported premise (`strobe/scripts/batch/ltx_av.py:15-18`,
  `strobe/docs/timeline.md:970-971`).
- LTX-2.5 joint A/V is batch at about 1.0× RTF on an H200
  (`strobe/CLAUDE.md:550`).
- H3 is clip-based everywhere. It reaches 1.0× only on 4-8 B200s
  (`infinite-livestream/fast-h3/README.md:242-251`).

**Recommendation**

1. **Default: clip-queue playout for every A/V model (H3, LTX-2/2.5).**
   - Implement fast-h3's queue contract in-process as a Rust `ClipQueue` pair:
     - generation and playout queues, submit-time reservation;
     - `enqueue/play/pop/move/stop/reset/set_*`;
     - `ClipInfo` echoed on every message;
     - `valid_commands` derived from state
       (`infinite-livestream/fast-h3/fasth3_queue.py:88-176`, `infinite-livestream/fast-h3/fasth3_session_rules.py:12-46`).
   - This works behind the Reactor protocol (same command names; see
     `research-reactor.md` §4bis) and as a local director for RTMP/HLS.
2. **Audio normalisation per clip (wire format).**
   - Resample the model's native rate (H3 32 kHz; LTX-2 vocoder 24 kHz,
     `docs/ports/ltx2.md:136-141`; LTX-2.5 BWE 48 kHz) to **48 kHz**.
   - Trim or pad to `round(frames/fps·48000)` samples
     (`infinite-livestream/fast-h3/fasth3_backend.py:540-577`).
   - Keep **stereo** on our own wire. IL downmixed to mono only because of a
     Reactor-recorder quirk (`:543-547`). INFERRED: Opus stereo is standard in
     WebRTC. Downmix only for a sink that needs it.
3. **A/V lockstep emission.**
   - Emit on a metronome in small slices (3 frames), each with its exact audio
     slice, re-anchoring instead of bursting (`infinite-livestream/fast-h3/fasth3.py:886-933`).
   - The pacer holds **two FIFOs with the same shallow cap**. Each tick emits
     one frame (or a repeat) plus `sample_rate/fps` samples (or silence)
     (`infinite-livestream/streaming-client/pacer.py:129-204`).
   - `sample_rate % fps == 0` must hold: 48000/24 = 2000 and 48000/16 = 3000.
4. **Gapless chaining.** Keep autoplay model-side so clip-to-clip gaps are
   milliseconds (`infinite-livestream/streaming-client/reactor_link.py:144-149`).
   - INFERRED improvements over IL's hard cuts (`infinite-livestream/fast-h3/README.md:263-269`):
     - a 10-20 ms audio fade-out and fade-in at clip boundaries, to avoid
       clicks;
     - continuity on request: last frame → next clip's first-frame anchor. Use
       the H3 `fl2va` recipe (`strobe/scripts/batch/fastvideo-catalog.yaml:54`) or
       LTX-2 I2V. This is strobe's chain-clips pattern
       (`strobe/scripts/batch/h3fast.py:870-909`) with its ref-lock bar (hand-off
       CLIP ≥ 0.85, `strobe/docs/timeline.md:281`).
5. **Throughput honesty.** Publish `build_s / clip_s` per clip.
   - Below 1.0×, the stream has black and silent holds, or the director must
     queue ahead.
   - Size the playout buffer in host RAM, about 1 GB per 14 s of 768p
     (`infinite-livestream/fast-h3/fasth3.yaml:21-26`).
6. **Near-real-time A/V for video-only models.** Run MMAudio V2A per clip after
   decode.
   - audio_rtf is 0.135 on a 15 s clip on H100 (`strobe/docs/timeline.md:282`).
     FV already has the MMAudio port (`crates/fastvideo-cudarc/src/mmaudio/`).
   - For a *causal* SF-Wan stream, run V2A on sliding windows (for example 2-5 s
     of decoded frames) and play the audio one window behind the video. This is
     INFERRED and untested; it adds window-length latency to the audio track,
     so delay the video by the same amount to keep A/V aligned.
7. **Future causal A/V.** If a causal A/V student exists (none is public per
   the references), follow strobe's per-block audio hook shape:
   `sample_audio_noise → denoise_audio(video=x0) → write_audio_context →
   decode_audio` (`strobe/src/strobe/causal.py:145-149`, `:346-364`). The emitted
   audio for each block must equal `block_frames/fps·48000` samples.

### 6.3 (c) Declaring video-only or video+audio per session

- **Track set.** Derive it from the model capability and the request at session
  create, and fix it for the session's life. This matches Reactor, where
  `Output` declares tracks and the runtime builds m-lines from them
  (`research-reactor.md` §4.5; `infinite-livestream/fast-h3/fasth3_types.py:30-34`).

  ```text
  SessionSpec.tracks = [
    Video { width, height, fps },                   // always
    Audio { sample_rate: 48000, channels: 1|2 },    // iff model.generates_audio
                                                    //   || request.sidecar_audio
  ]
  ```

- **Per-transport rules**
  - **WebRTC (Reactor peer and WHIP)**: add an audio transceiver or m-line
    **only** when the spec has audio. Video-only sessions offer one video
    m-line, as strobe does (`strobe/src/strobe/session.py:176`).
    - Codec order: H.264 first for Cloudflare (`strobe/src/strobe/whip.py:25-42`);
      Opus 48 kHz for audio. INFERRED: `tests/test_whip.py:166` already expects
      `opus/48000/2` in answers.
  - **RTMP/HLS**: always mux an audio stream. The pacer synthesises silence for
    video-only models, because platforms reject video-only FLV
    (`infinite-livestream/streaming-client/sinks/rtmp.py:8-10`). Encode AAC; IL uses 128k,
    44.1 kHz stereo (`infinite-livestream/streaming-client/sinks/_ffmpeg.py:195`).
  - **Batch MP4**: mux audio only if present. Refuse an "A/V" job whose output
    has no audio stream (`strobe/scripts/batch/sidecar-audio.py:57-78`).
- **Capability surface.** Publish `generates_audio` per model id, following
  IL's `ModelLink.generates_audio`
  (`infinite-livestream/streaming-client/model_link.py:60-62`) and strobe-h3-server's
  `/strobe/v1/capabilities` (`strobe/rust/crates/strobe-h3-server/src/routes.rs:67`).
  - Wan, FastWan, SF-Wan: `false`.
  - H3, LTX-2, LTX-2.5: `true`.
  - MMAudio sidecar: an opt-in that turns `false` into `true`.
- **Refuse, do not degrade.**
  - A request for audio on a model without audio and without the sidecar
    returns 400. This follows strobe's hard-error rule
    (`strobe/src/strobe/audio_vae.py:16-21`) and strobe-h3-server's "a field that
    cannot be honoured is refused"
    (`strobe/rust/crates/strobe-h3-server/src/fastvideo_api.rs:20-24`).
  - A mid-session canvas change is refused while clips exist
    (`infinite-livestream/fast-h3/fasth3_session_rules.py:42-45`).

### 6.4 Protocol front-ends, in priority order

1. **Reactor protocol** (peer WebRTC plus commands): interactive sessions. SF-Wan
   uses block mode with prompt steering at block boundaries; H3 and LTX use the
   clip-queue contract. Details are in `research-reactor.md` §8.
2. **WHIP publish**: the serverless and fan-out mode for real-time causal video.
   - A RunPod or other queue job carries `{prompt, whip_url, whip_token,
     duration_s, …}` and dials out (`strobe/src/strobe/runpod_handler.py:7-11`).
   - Viewers use the SFU's WHEP or HLS.
   - Implementation: non-trickle offer, `Location` for DELETE, Basic or Bearer
     auth (`strobe/src/strobe/whip.py:93-152`).
3. **Clip-queue HTTP plus RTMP/HLS director** (the fal H3-max style): batch
   builds through our `/v1/videos` or `/v2/video_generation` adapters (§3.4),
   and a playout process that paces them into one broadcast
   (`infinite-livestream/streaming-client/fal_link.py:1-48`).
4. **Batch adapters**: FastVideo OpenAI `/v1/videos` and MiniMax v2, exactly as
   in §3.4. strobe-h3-server is the most complete reference for keys, signed
   URLs and callbacks.

### 6.5 What to port and what to reference

| item | action |
|---|---|
| `strobe-core::pacing` + `clock` | **Port** (MIT, attributed). Extend with an audio FIFO following IL's `Pacer` and an async first-frame notify. |
| `strobe-core::rollout` | Take the design (thread + bounded channel). **Do not** build decode/denoise overlap: it was measured at 11% of theory (`strobe/CLAUDE.md:152-169`). |
| `strobe-core::frame_math` | Port the idea as a generic `(time_factor, overlap)` helper with the keep-rule tests. |
| strobe `ttff.py`, `loopback.py` | Re-implement: a TTFF phase recorder and a CPU loopback smoothness bench. |
| strobe `whip.py`, `rtc_h264.py` | Re-implement in Rust (str0m or webrtc-rs). Keep H.264-first, the 2 s IDR, and non-trickle. |
| IL `fasth3_queue.py`, `fasth3_session_rules.py`, `_emit_clip`, `_to_wire_audio` | Re-implement as the clip-queue engine and wire audio normaliser. |
| IL `pacer.py`, `sinks/_ffmpeg.py` | Re-implement: A/V lockstep pacer; an ffmpeg sink with two pipes, writer threads, and restart. |
| strobe `api.rs` (strobe-h3) + strobe-h3-server routes | Reference for batch API validation and error envelopes. INFERRED: licence permits copying (workspace `license = "MIT"`, `strobe/rust/Cargo.toml:22`). |
| strobe-engine, strobe-h3 compute, strobe-h3-vsa | Skip. |

---

## 7. Open questions

1. **TAEHV per-block decode.** Does it carry temporal state cleanly across block
   calls, or does streaming need the latent-overlap bridge? A CPU/GPU parity
   test against the whole-clip decode would settle it.
2. **SF-Wan infinite rollout.** Our SF-Wan quality at long horizons with rolling
   KV is unmeasured. strobe has only a first, roughly 90 s sample
   (`strobe/CLAUDE.md:98-102`). LTX collapsed (`:545`), so measure before exposing
   an unbounded `duration_s`.
3. **MMAudio sliding-window V2A.** Latency and quality for a causal stream are
   unmeasured anywhere (§6.2 item 6).
4. **Audio at clip boundaries.** Whether a short crossfade at H3/LTX clip
   boundaries is acceptable, or whether chained first-frame anchors (`fl2va`)
   are required for continuity, is a product decision. IL ships hard cuts.
5. **Encoder.** Whether to run the H.264 encoder in-process (openh264/x264
   bindings) or as an ffmpeg subprocess as IL does. strobe's number (2.64 ms
   per frame, 480p) says either works. INFERRED.
