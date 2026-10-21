# `/v1/segments`: pull-based segments from a causal session

Status: implemented (`crates/fastvideo-serve/src/segments.rs`), mounted with
`protocols.native` when `[segments] enabled` (default on). CPU tests:
`crates/fastvideo-serve/tests/segments.rs`. Not yet run on a GPU.

## 1. Why

An external control plane (the `unreel-cloudflare` stream engine) runs an
infinite live stream as a sequence of 5–10 s **segments**: it expands each
prompt, asks a render backend for the clip, and feeds the clip into its own
WHIP publisher. Its backend contract (`docs/render-backend.md` there) is
`POST /v1/segments` + `GET /v1/segments/{id}` with a `video_url` per segment.

On a clip model that means one job per segment and a visible seam between
clips. On a causal model (SF-Wan, LongLive) the right answer is **one
rollout for the whole stream**, cut into segments after the fact: the KV
cache, the frame sink and the decoder state carry across segments and the
seams do not exist. That is what this router does, over the engine's
`CausalSession` (design §5.4), without the live pacer and WHIP publisher of
`/fv/v1/streams` (the control plane publishes).

## 2. Routes

| Route | Behaviour |
|---|---|
| `POST /v1/segments` | → 202 segment object. The first segment of a `stream_id` opens a causal session on an executor (exclusive lease); later segments continue it |
| `GET /v1/segments/{id}` | the segment object: `queued` → `running` → `succeeded` (`video_url`, a signed artifact URL) or `failed` (`error`) |
| `GET /v1/segments/streams` | the streams with a session here |
| `GET /v1/segments/streams/{stream_id}` | model, canvas, sessions opened, blocks generated, queue depth, idle time, the session's `state_update` |
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

- `model`: a causal model served here; default `[segments] model`, else the
  first causal model the engine serves. A stream keeps the model it opened with.
- `duration_sec` 0.5–30. The segment is `round(duration_sec × fps)` frames;
  the rollout's blocks (12 frames = 0.75 s at 16 fps) are consumed as needed
  and the remainder carries over to the next segment, so nothing is dropped
  and segment N+1 starts on the frame after segment N's last. 6 s = 96 frames
  = exactly 8 blocks.
- `resolution` / `aspect_ratio`: the causal canvas tier (480p; `width×height`
  from `canvas_for_aspect`, 832×480 at 16:9). Other tiers only when the
  model serves them.
- `recache`: the direction changed. The prompt is set before the segment's
  first block; the model's prompt-switch policy applies (SF-Wan keeps the
  cache, LongLive re-caches once). A prompt equal to the previous segment's
  is not re-set, so an unchanged prompt never triggers a re-cache.
- `reset`: a scene cut. The KV cache is cleared and blocks the engine had
  already generated under the old scene are discarded
  (`session.discarded_blocks`). Block numbering restarts at 0.
- `context.*`: refused (`invalid_param`). The causal path keeps its own state;
  the control plane uses `continuity: "native"`, which sends none.
- `reference_image_urls`: accepted and ignored on Wan (`warnings`).
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
  "session": {"restarted": false, "stale_blocks": 1, "discarded_blocks": 0},
  "metrics": {"generation_ms": 3900.0, "encode_ms": 180.0},
  "error": null, "created_at": 1761086400, "finished_at": 1761086405
}
```

- `trim_head_sec` is always 0: nothing is re-rendered.
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
- Output: video-only MP4 (Wan is silent), H.264 at constant quality 19
  (NVENC; `[segments] encoder = "libx264"` for CPU tests), `+faststart`.

## 3. Session lifecycle

```
POST segment 0 ──► slot(stream_id) ──► worker task
                                        ├─ open_causal_session(spec)   (lease on an executor)
                                        ├─ set_prompt / reset per segment
                                        ├─ next_block() … until `need` frames (carry-over kept)
                                        ├─ write_mp4 → artifacts.put → signed url_for(ttl)
                                        └─ idle > idle_ttl (120 s) or DELETE → close, free the lease
```

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
- `queue` (8) segments may wait per stream; more → 429.
- The session's `max_seconds` is enforced by the engine; when it ends under
  a segment the worker reopens once (`restarted: true`). Ending twice under
  one segment fails the segment.

## 5. Config

```toml
[segments]
enabled = true
model = "sfwan"        # FV_SEGMENTS_MODEL; default: first causal model served
idle_ttl_s = 120       # FV_SEGMENTS_IDLE_TTL_S
max_seconds = 300      # session video length; clamped to [streams] causal_hard_max_s
url_ttl_s = 3600
encoder = "nvenc"      # FV_SEGMENTS_ENCODER; "libx264" for CPU tests
keep = 256
queue = 8
```

## 6. Gaps

- Not run on a GPU yet: the CPU tests use the fake causal model. The first
  GPU smoke is `runpod-sfwan.toml` + two segments of 6 s, checking
  `generation_ms` ≈ 4 s (H100) and a seamless join in the control plane.
- No first-frame or tail prefill on the causal path (`context.*` is refused):
  the control plane's `last_frame` / `overlap` modes stay on clip providers.
- No audio; no reference images; one stream per GPU; 480p only.
- Artifacts are never deleted by this router (the store's retention applies).
