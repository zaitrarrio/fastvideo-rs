# `/v1/segments`: pull-based segments from one engine session

Status: implemented (`crates/fastvideo-serve/src/segments.rs`), mounted with
`protocols.native` when `[segments] enabled` (default on). CPU tests:
`crates/fastvideo-serve/tests/segments.rs`. GPU smoke: §6.

## 1. Why

An external control plane (the `unreel-cloudflare` stream engine) runs an
infinite live stream as a sequence of 5–10 s **segments**: it expands each
prompt, asks a render backend for the clip, and feeds the clip into its own
WHIP publisher. Its backend contract (`docs/render-backend.md` there) is
`POST /v1/segments` + `GET /v1/segments/{id}` with a `video_url` per segment.

On a causal model (SF-Wan, LongLive) the right answer is **one rollout for
the whole stream**, cut into segments after the fact: the KV cache, the
frame sink and the decoder state carry across segments and the seams do not
exist. That is what this router does over the engine's `CausalSession`
(design §5.4), without the live pacer and WHIP publisher of
`/fv/v1/streams` (the control plane publishes).

On a clip model (H3, LTX, Wan with image-to-video) a stream is **one clip
build per segment, chained on the last frame**, the way the fal director
chains its chunks (`director-chunks.md`): the router keeps the previous
segment's last frame, conditions the next build on it (`Task::I2V`) and
drops the duplicated frame. The control plane keeps using `continuity:
"native"`; nothing crosses the wire between segments. It may instead send
its own `context.image_url` (its `last_frame` mode) or an `end_image_url`
(keyframe planning, models with `Task::Keyframes`). This is the same
`ClipSession` (design §5.5) the director and the Reactor use, so a backend
serving any `runpod-h3-*.toml` / `runpod-ltx.toml` preset answers
`/v1/segments` with no further configuration.

## 2. Routes

| Route | Behaviour |
|---|---|
| `POST /v1/segments` | → 202 segment object. The first segment of a `stream_id` opens a session on an executor (exclusive lease): a causal rollout or a clip session, by the model; later segments continue it |
| `GET /v1/segments/{id}` | the segment object: `queued` → `running` → `succeeded` (`video_url`, a signed artifact URL) or `failed` (`error`) |
| `GET /v1/segments/streams` | the streams with a session here |
| `GET /v1/segments/streams/{stream_id}` | `path` (`causal` \| `clip`), model, canvas, sessions opened, blocks generated (causal) / `clips` built, queue depth, idle time, a causal session's `state_update` |
| `DELETE /v1/segments/streams/{stream_id}` | close the session now |

Also under `/fv/v1/segments`. Auth is the native API's (`Authorization:
Bearer <key>`); a busy executor answers 429 + `Retry-After` on the
*segment*, see §4.

### Request

```json
{
  "stream_id": "str_8fQ2kLmN0pRsTuVw",
  "segment_index": 7,
  "model": "sfwan",
  "prompt": "A rain-slick stone harbor at night … Camera: … Sound: …",
  "duration_sec": 6,
  "resolution": "480p",
  "aspect_ratio": "16:9",
  "seed": 42,
  "recache": true,
  "reset": false,
  "reference_image_urls": [],
  "context": {"image_url": null, "video_url": null, "overlap_sec": null},
  "callback_url": "https://unreel.example/internal/wake/str_…/token",
  "extra": {}
}
```

- `model`: a model served here, by id, served name or tier alias
  (`h3-turbo`, `ltx-pro`, …); default `[segments] model`, else the first
  causal model the engine serves, else the first clip model with
  image-to-video. A stream keeps the model it opened with. A model that can
  neither stream causally nor chain on a frame (text-only clip models) is
  refused.
- `duration_sec` 0.5–30. **Causal:** the segment is `round(duration_sec ×
  fps)` frames; the rollout's blocks (12 frames = 0.75 s at 16 fps) are
  consumed as needed and the remainder carries over to the next segment, so
  nothing is dropped and segment N+1 starts on the frame after segment N's
  last. 6 s = 96 frames = exactly 8 blocks. **Clip:** the count snaps up
  onto the model's frame grid within its range (H3 `17n+5` at 24 fps, 6 s →
  158 frames = 6.58 s; LTX `8k+1`, 6 s → 145; H3's minimum is 4.46 s, LTX's
  5 s on the two-stage recipes), and a continuation delivers one frame fewer
  (the duplicated anchor). The 202 object says `planned_frames` /
  `planned_duration_sec`; the control plane plans its timeline on it.
- `resolution` / `aspect_ratio`: a short-edge tier of the model (SF-Wan
  480p; H3 768p, 480p on the real recipes; LTX 480p…2160p). The model's
  default tier when absent; a tier it does not serve snaps to the nearest
  served one with a warning (a stream never fails on a tier name). LTX
  generates its padded canvas and the output is centre-cropped to the tier,
  the anchor frame uncropped, as the director does.
- `recache`: the direction changed. **Causal:** the prompt is set before the
  segment's first block; the model's prompt-switch policy applies (SF-Wan
  keeps the cache, LongLive re-caches once). A prompt equal to the previous
  segment's is not re-set, so an unchanged prompt never triggers a re-cache.
  **Clip:** nothing; every build is conditioned afresh on its prompt.
- `reset`: a scene cut. **Causal:** the KV cache is cleared and blocks the
  engine had already generated under the old scene are discarded
  (`session.discarded_blocks`); block numbering restarts at 0. **Clip:** the
  build starts from text instead of the previous last frame.
- `context.image_url`: **clip** models only (the control plane's
  `last_frame` mode): fetched (`http(s)`, feature `http-client`) and used as
  the build's first frame instead of the backend's own anchor. Refused on
  the causal path, which keeps its own state. `context.video_url` is refused
  on both (no extend on this path; the backend chains on the last frame).
- `end_image_url`: on clip models with `Task::Keyframes` (H3, LTX-2.5) the
  frame the segment ends on (keyframe planning); a warning elsewhere.
- `reference_image_urls`: accepted and ignored (`warnings`).
- `callback_url`: with the `http-client` feature, the finished segment object
  is POSTed there (3 attempts). Without it, poll.
- Resubmitting the same `(stream_id, segment_index)` returns the existing
  segment (idempotent).

### Segment object

```json
{
  "id": "fvseg_…", "object": "fv.segment",
  "stream_id": "str_…", "segment_index": 7,
  "status": "succeeded",
  "video_url": "https://…/files/…/str_…-7.mp4?…",
  "trim_head_sec": 0.0,
  "frames": 96, "fps": 16, "width": 832, "height": 480, "duration_sec": 6.0,
  "blocks": [56, 63],
  "planned_frames": 96, "planned_duration_sec": 6.0,
  "continuity": "native", "audio": false,
  "session": {"restarted": false, "stale_blocks": 1, "discarded_blocks": 0},
  "metrics": {"generation_ms": 3900.0, "encode_ms": 180.0},
  "error": null, "created_at": 1761086400, "finished_at": 1761086405
}
```

- `trim_head_sec` is always 0: nothing is re-rendered, and a clip
  continuation's duplicated first frame is dropped here (with one frame's
  worth of audio).
- `continuity`: how the segment was conditioned. `native` (a causal
  rollout), `anchor` (the previous segment's last frame), `image` (the
  request's `context.image_url`), `text` (a first segment, a `reset`, or an
  anchor the router could not keep; after a failed build the picture
  continues from the anchor it still has).
- `audio`: the MP4 has an audio track (H3 32 kHz stereo, LTX 48 kHz; AAC
  128 kb/s). The causal models are silent.
- `session.stale_blocks`: blocks at the head of the segment that were
  generated before this segment's prompt landed. The engine runs up to
  `causal_depth` (4) blocks ahead, so a direction shows after up to 3 s of
  the previous beat. The control plane can report `direction.accepted` at
  `segment_index` and know the picture turns within that segment.
- `session.restarted`: the session was reopened under this segment (its
  video-length limit, or an engine error): the seam before it is a cut.
  Causal sessions run `[streams] causal_default_max_s` (120 s) of video by
  default, `[segments] max_seconds` to raise it, never above
  `causal_hard_max_s` (300). The control plane should expect a restart at
  that cadence until the long-horizon study (docs/ports/wan.md R12) says
  otherwise.
- Output: MP4, H.264 at constant quality 19 (NVENC; `[segments] encoder =
  "libx264"` for CPU tests), `+faststart`, AAC audio when the model has it.

## 3. Session lifecycle

```
POST segment 0 ──► slot(stream_id) ──► worker task (by the model's path)
  causal:                               ├─ open_causal_session(spec)   (lease on an executor)
                                        ├─ set_prompt / reset per segment
                                        ├─ next_block() … until `need` frames (carry-over kept)
                                        ├─ write_mp4 → artifacts.put → signed url_for(ttl)
                                        └─ idle > idle_ttl (120 s) or DELETE → close, free the lease
  clip:                                 ├─ open_clip_session(spec)     (lease on an executor)
                                        ├─ per segment: first_frame = context.image_url | anchor | none (reset)
                                        │              last_frame  = end_image_url (Keyframes)
                                        │              ClipBuild{frames: planned} → frames + PCM
                                        ├─ anchor = last frame (PNG, uncropped); drop frame 0 of a continuation
                                        ├─ crop (LTX) → write_mp4 (+AAC) → artifacts.put → signed url_for(ttl)
                                        └─ idle / DELETE → close, free the lease, delete the anchor
```

A failed input (an image that does not fetch or decode) fails that segment
only; the session and the anchor stay, so the next segment continues the
picture. An engine failure (a busy executor, a failed build) drops the
session; the next segment reopens one (`session.restarted`), still on the
anchor.

The engine blocks the executor when the block channel is full, so a stream
whose control plane stops asking does not run away: it stalls after 4
blocks and resumes on the next segment (that lookahead is also why the first
blocks of a segment can be `stale`).

## 4. Capacity and errors

- One causal session per executor. A segment for a second stream on a busy
  executor is accepted (202) and then **fails** with the engine's busy error
  (`engine_failed`/`conflict` in `error`); the control plane retries later or
  routes the stream elsewhere. The control plane is expected to run one
  backend URL per GPU and pin a stream to it (`X-Stream-Id` is accepted but
  routing is the caller's).
- A stream keeps the model it opened with; a segment naming another model
  is refused.
- Clip builds run at `Priority::Stream` on the session's executor, ahead of
  batch jobs, like the director's chunks: a backend can keep serving its
  batch APIs between segments.
- `queue` (8) segments may wait per stream; more → 429.
- The session's `max_seconds` is enforced by the engine; when it ends under
  a segment the worker reopens once (`restarted: true`). Ending twice under
  one segment fails the segment.

## 5. Config

```toml
[segments]
enabled = true
model = "sfwan"        # FV_SEGMENTS_MODEL; default: first causal model, else first I2V clip model
idle_ttl_s = 120       # FV_SEGMENTS_IDLE_TTL_S
max_seconds = 300      # session video length; clamped to [streams] causal_hard_max_s
url_ttl_s = 3600
encoder = "nvenc"      # FV_SEGMENTS_ENCODER; "libx264" for CPU tests
keep = 256
queue = 8
```

## 6. GPU smoke (2026-10-21, H100 SXM, Runpod secure)

`scripts/serve/e2e/segments-pod.sh`: one pod with the `sfwan` variant image
(`sfwan-sha-b9da353`, features `cuda,http-client`), the `sfwan21-1.3b` tree
fetched from the Hub onto the container disk, `runpod-sfwan.toml`, segments
encoder `libx264` (H100 has no NVENC). An on-pod client pulled four 6 s
segments of one stream at 832×480 / 16 fps: the same prompt twice, a
prompt change (`recache`) at segment 2, a scene cut (`reset`) at segment 3.
Results in `artifacts/serve/segments/10212204/`.

| segment | blocks | generation_ms | encode_ms | wall (client) | session |
|---|---|---|---|---|---|
| 0 | 0–8 | 4911 (includes the session open) | 695 | 6.0 s | — |
| 1 | 9–16 | 2930 | 866 | 4.0 s | — |
| 2 (recache) | 17–24 | 2860 | 842 | 4.0 s | `stale_blocks: 3` |
| 3 (reset) | 0–8 | 4870 (includes the re-prefill) | 812 | 6.0 s | `discarded_blocks: 3` |

- Every segment: 96 frames, H.264 832×480 at 16 fps, 2.6–4.1 MB. The
  steady state is **2.9 s of generation per 6 s segment (≈2× realtime)** at
  30 GB of GPU memory; the libx264 encode adds 0.8 s and would be ~0.2 s on
  NVENC. A control plane that asks 1–2 segments ahead keeps the stream fed.
- **The seam is invisible.** The mean luma difference between the last frame
  of segment 0 and the first of segment 1 is 6.90, against 7.01 between
  consecutive frames inside segment 0 (`seam-intra.txt`); the pair's PSNR
  is 24.2 dB, an ordinary inter-frame value for this motion. Concatenating
  the two MP4s (`seg-0-1-joined.mp4`) shows one continuous shot.
- The direction at segment 2 landed after 3 blocks (2.25 s), exactly the
  engine's lookahead (`prompt switch: KV cache kept … block=20` for a
  segment starting at block 17); `stale_blocks` reported it.
- The reset at segment 3 discarded the 3 blocks the engine had run ahead
  under the old scene, restarted numbering at 0 and cost the first-segment
  price again (4.9 s); the picture is a clean cut to the new scene.
- fv-serve was ready 52 s after start (27 GB of weights loaded); the whole
  run took 4.4 pod-minutes ($0.29 at $3.99/h). An earlier attempt on the
  same image ran blind for 26 min because the variant images carry no
  `fv-gpucheck` (the result server); the script now uses python3.

## 7. Gaps

- The clip path (H3, LTX) is CPU-tested on the fake engine only
  (`tests/segments.rs`: an LTX chain with anchor, reset, image and keyframe
  inputs and audio in the MP4; the H3 tier alias and grid). Its GPU smoke
  is `segments-pod.sh` with `FV_SEG_MODEL=h3-turbo` / `ltx-turbo` and
  `FV_SEG_RES=768p` / `480p` on the matching variant image, with the H3 /
  LTX trees mounted from the EU volume or fetched from R2 (the script only
  fetches SF-Wan today); not run yet.
- No extend / tail prefill on either path (`context.video_url` is refused):
  LTX-2.5 `Task::Extend` exists in the engine but needs a staged source and
  an edit window, and its 2 s minimum extension and 8-frame rounding do not
  fit the contract's lengths. The control plane's `overlap` mode stays on
  clip providers.
- No reference images on either path (H3 ref2va is a separate recipe); no
  audio on the causal path; one stream per GPU.
- Not yet measured: a 5–10 minute causal stream (the `max_seconds` restart
  and the R12 drift question), NVENC on L40S / RTX PRO 6000, and the
  end-to-end join in the control plane's streamer (the MP4s above are what
  it gets).
- Artifacts are never deleted by this router (the store's retention applies).
