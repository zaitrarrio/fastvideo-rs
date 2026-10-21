# `/v1/segments`: pull-based segments from a causal session

Status: implemented (`crates/fastvideo-serve/src/segments.rs`), mounted with
`protocols.native` when `[segments] enabled` (default on). CPU tests:
`crates/fastvideo-serve/tests/segments.rs`. GPU smoke: §6.

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
- `reference_image_urls`, `end_image_url`: accepted and ignored on Wan (`warnings`).
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

- No first-frame or tail prefill on the causal path (`context.*` is refused):
  the control plane's `last_frame` / `overlap` modes stay on clip providers.
- No audio; no reference images; one stream per GPU; 480p only.
- Not yet measured: a 5–10 minute stream (the `max_seconds` restart and the
  R12 drift question), NVENC on L40S / RTX PRO 6000, and the end-to-end
  join in the control plane's streamer (the MP4s above are what it gets).
- Artifacts are never deleted by this router (the store's retention applies).
