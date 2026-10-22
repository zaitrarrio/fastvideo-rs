# ADR-0004: Tween — a keyframe storyboard app over fv-serve

- Status: Proposed
- Date: 2026-10-20
- Decision-id: FVID-2026-10-20-tween-storyboard

## Context

The owner wants animated storyboards: draw (or generate) panels, then have a
fast first-last-frame (FLF2V) model fill the motion between each pair of
panels, ideally faster than real time, and compare two model families side by
side before committing to one.

fv-serve already has everything a client needs:

- `POST /v1/videos` (`crates/fastvideo-openai-videos/src/videos.rs`) turns two
  `image_reference` entries into `Task::Keyframes` (first + last) and an
  explicit `task: "ref2va"` (or more than two images) into `Task::Ref2V`.
- LTX-2.5 recipes and every H3 tier declare `Task::Keyframes`
  (`fastvideo-engine-service/src/cuda/caps.rs`, `ModelCaps::h3`). LTX-2.3 does
  not (`GapId::LtxKeyframes`, E9). Ref2V needs a reference-loaded model
  (`h3-ref2v-*`, `ltx25-ref2v`).
- The job view carries `inference_time_s`, `seconds`, `created_at` and
  `completed_at`, so the real-time factor comes from the engine, not from a
  client stopwatch.
- The fake engine (`configs/serve/fake.toml`) serves `fake-ltx-turbo`,
  `fake-ltx-pro`, `fake-h3-turbo` and `fake-h3-max` on CPU.

Owner choices (2026-10-20): an end-to-end storyboard app; LTX and H3 side by
side (H3 `fl2va` / `ref2va` rather than a Wan VACE port); first + last panels
only (no mid-shot keyframes, the protocol `Anchor` stays `First | Last`); a
new app in this repository.

## Decision

Add **Tween** (`tween/`), a Cloudflare Worker (`fv-tween`) with its own D1
database and R2 bucket and a static UI, that drives fv-serve over the public
`/v1/videos` API. No Rust change.

1. **A board is an ordered list of panels; a shot is a pair of adjacent
   panels.** A board with N panels has N - 1 shots. Shot *i* runs from panel
   *i* to panel *i + 1*, so consecutive shots share their boundary frame by
   construction and the animatic stitches without seams. Shot settings
   (prompt, seconds, mode, seed) are keyed by the shot's start panel id;
   reordering panels re-derives the shots.
2. **Two shot modes.** `keyframes`: `image_reference: [first, last]`.
   `reference`: `task: "ref2va"` with the shot's panels plus any extra
   reference images; it runs only on a reference-loaded model and the
   engine's 4xx is surfaced as-is. The API has no combined FLRef2V task, so
   Tween does not fake one. **Reference mode is H3-only:** `/v1/videos`
   accepts `task` on H3 models only (`apply_model`), and without it one or
   two images on LTX always resolve to keyframes, so an LTX reference request
   is not expressible for a two-panel shot.
3. **Engines are configuration, not code.** Each engine (`ltx`, `h3`) has a
   draft and a final model id plus an optional reference model id, set by
   Worker vars (`TWEEN_LTX_DRAFT_MODEL`, …). Defaults point at the fake
   engine; `wrangler.toml` staging points at the production ids
   (`ltx25-distill-sol` / `ltx25-distill-dense`, `fasth3` / `sol-h3`).
4. **A bake-off submits the same shot to both engines** with the same seed,
   prompt, seconds and aspect, and records two renders side by side.
5. **The engine key stays server-side.** Unlike fv-app (browser-held key),
   Tween holds `TWEEN_ENGINE_KEY` as a Worker secret: panels and clips live in
   Tween's R2, and the browser never talks to the engine.
6. **Panels go to the engine as data URIs.** Two panels are a few MiB, well
   under `limits.body_max_mb`, and data URIs need no presigned URL lifetime
   or engine egress.
7. **Renders are pulled, not pushed.** The UI polls `GET /api/renders/:id`;
   the Worker polls the engine job and, on success, copies the MP4 into R2
   once and freezes the metrics in D1. No webhooks, no Durable Objects.
8. **Real-time factor = clip seconds / engine `inference_time_s`**, with the
   wall-clock factor (seconds / (completed_at - created_at)) recorded beside
   it so queueing and weight-swap time are visible rather than hidden.
9. **Draft then final.** Every render has a tier (`draft` | `final`); final
   re-renders reuse the shot's keyframes and seed so cut points do not move.

## Consequences

- Positive: the bake-off and the animatic run on today's engines; local
  development needs only Docker (the fake engine) and Node.
- Positive: switching engines or tiers is a vars change, not a deploy of
  fv-serve.
- Negative: the animatic is a client-side playlist, not a rendered MP4
  (no ffmpeg in a Worker). An export endpoint is follow-up work.
- Negative: mid-shot keyframes need a protocol change (`Anchor::Frame(u32)`)
  and are out of scope; LTX rope already positions arbitrary keyframes
  (`ltx2/rope.rs` `with_keyframes`), so the follow-up is protocol + negotiate
  work, not model work.
- Negative: polling costs one engine request per poll per running render;
  acceptable at storyboard scale.

## Alternatives considered

- **Browser-held engine key (fv-app's model).** Rejected: Tween stores panels
  and clips, so it needs server-side state anyway, and a shared studio tool
  should not hand every viewer the engine key.
- **Port Wan VACE as the second model.** Deferred by the owner: H3 already
  serves fl2va / ref2va, so the comparison needs no model work.
- **Signed R2 URLs instead of data URIs.** Rejected for now: needs engine
  egress to R2 and a URL lifetime longer than the queue; data URIs fit the
  body limit.
