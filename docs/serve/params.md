# Request parameters: `ParamSpec` and the surface rules

Plan E2, ADR-0012. Every request parameter of every submit endpoint is
described by a `ParamSpec` (`fastvideo_protocol::params`), one per
(surface, endpoint, model), derived from two sources only:

- the **surface rules**, `crates/fastvideo-protocol/src/params/rules.rs`:
  every limit or allowed set an upstream contract fixes, one constant each,
  citing its vendored file under `docs/serve/specs/` (checked against the
  files by `crates/fastvideo-protocol/tests/param_rules.rs`);
- the served model's **`ModelCaps`** (family manifests, projected by the
  engine): frame grid, rates, canvas tiers, knobs, reference limits.

Adapters validate with the rule constants and keep no numeric limit of
their own (`tests/param_scan.rs`); each adapter's tests check its parser
against its specs (`parse_agreement`).

## Reading the specs

`GET /fv/v1/params` (native auth; mounted only when the native API is,
like `/fv/v1/capabilities`) answers `{"object": "fv.param_specs",
"specs": [...]}` for every mounted submit endpoint and every served model:

| Surface | Endpoint | One spec per |
|---|---|---|
| `native` | `POST /fv/v1/jobs` | served model |
| `openai_videos` | `POST /v1/videos` | served model |
| `fastwan` | `POST /generate` | the FastWan model |
| `minimax` | `POST /v2/video_generation` | MiniMax model name that resolves here |
| `ltx` | `POST /v2/{text,image,audio}-to-video`, `/v2/{retake,extend}` (and `/v1/*`) | (endpoint, `model` id) whose target serves the task |
| `fal` | each configured `app/endpoint` | app endpoint whose model is served |

Each `Param` has `name` (dotted for nested fields), `type`, `required`,
`nullable`, `default`, `domain` (what this server accepts for the model),
`wire` (the wider domain the parser enforces, when the model narrows it:
inside `wire` but outside `domain` is refused by negotiation), `support`
(`honoured`, `no_op`, `refused`, or `gap` with its `GapId`), `maps_to` (the
normalized field), `upstream` and `note`. Domains: `int`, `number`,
`one_of`, `frames` (the model's grid), `text`, `items`, `canvas`, `aspect`,
`media`, `any`. `crates/fastvideo-serve/tests/goldens/param_specs.json`
holds the specs of every shipped config.

## Census (2026-10-24, before E2)

Every limit or allowed set enforced on the serve path, where it was, and
where it lives now. "Dup" marks a value restated in more than one place;
"Diff" a disagreement.

### Shared facts restated per surface (duplicates, now one source)

| Limit | Was (file:line at `f1007888`) | Surfaces / models | Upstream | Now |
|---|---|---|---|---|
| H3 clip 5..=15 s | `openai-videos/src/videos.rs:48` (`H3_MIN_SECONDS`), `fal/src/schema.rs:66-67`, `minimax/src/lib.rs:112-122` (4 / 5..15); the grid itself from `fastvideo_models::h3::config` (`FrameGrid::h3`) | `/v1/videos`, fal `minimax/h3*`, MiniMax; H3 | FastVideo 5-15 s; fal `minimax/h3` integer 5..15; MiniMax enum 4..15 (H3-Max 5) | `rules::h3::SECONDS` / `SECONDS_FROM_FOUR`, from `fastvideo_models::h3::config` (Dup x3) |
| H3 references 9 / 3 / 3 / 12 | `protocol/src/caps.rs:454` (`RefLimits::h3`), `minimax/src/create.rs:46`, `fal/src/schema.rs:69-72`, `fastvideo_models::h3::reference:29-32` | negotiation, MiniMax, fal r2v; H3 Ref2VA | fal `maxItems` 9 / 3 / 3, `reference_conditions` 12 | `rules::h3::REFERENCES` from `fastvideo_models::h3::reference` (Dup x4) |
| H3 aspect set | `minimax/src/create.rs:48` (`RATIOS`), `fal/src/schema.rs:349-366` (typed enum) | MiniMax, fal | MiniMax `ratio`, fal `aspect_ratio` | `rules::h3::ASPECT_RATIOS`, `rules::minimax::RATIOS`; fal's typed enum checked against them (Dup) |
| LTX duration matrix (fast: 20 s at <=1080p and 24/25 fps, else 10; 6..max step 2) | `ltxapi/src/models.rs:235-241`, `fal/src/schema/ltx.rs:84-92, 160-169` | LTX API, fal `lightricks/ltx-2.5` | LTX "Supported Models"; fal `duration` enums | `rules::ltx::{max_duration, durations}` (Dup) |
| LTX `camera_motion` set | `ltxapi/src/request.rs:62`, `fal/src/schema/ltx.rs:53` | LTX API, fal LTX | `TextToVideoRequestCameraMotion`; fal enum | `rules::ltx::CAMERA_MOTIONS` (Dup) |
| LTX prompt 5000 | `ltxapi/src/request.rs:59`, `fal/src/schema/ltx.rs:44` (used by fal ltx, a2v, edit, ingredient) | LTX API, fal LTX apps | `maxLength: 5000` | `rules::ltx::PROMPT_MAX_CHARS` (Dup) |
| LTX rates 24/25/48/50 | `ltxapi/src/models.rs:230`, `fal/src/schema/ltx.rs:48`; `engine-service/src/caps.rs:53` (`LTX_FPS`, the engine's) | LTX API, fal fast; the LTX caps | LTX docs; fal fast `fps` enum | wire set `rules::ltx::FPS` (Dup); the engine's `LTX_FPS` stays its own (a capability, equal by fact) |
| fal H3 prompt 50000 | `fal/src/schema.rs:64`, `fal/src/director/messages.rs:20` | fal queue, director | `maxLength: 50000` | `rules::fal_h3::PROMPT_MAX_CHARS` (Dup) |
| fast-h3 prompt 800 / metadata 2000 / 4 aspects | `reactor/src/commands.rs:290-293`, `engine-service/src/stream/player.rs:61-65`, `serve/src/segments.rs:72` | Reactor, stream player, `/v1/segments` | fast-h3 (research-reactor.md) | `rules::fast_h3` (Dup x3) |

### Per-surface limits (single place, moved into the rules)

| Surface | Limit | Was | Upstream | Now |
|---|---|---|---|---|
| `/v1/videos` | `num_inference_steps` 1..=200; `guidance_scale` 0..=20; `boundary_ratio` 0..=1; `n` / `num_outputs_per_prompt` 1..=10 (1 served); `quality` set; `fl2va` 1-2 images; `GET ?limit` 1..=100 default 20 | `openai-videos/src/videos.rs:385-500, 692, 711, 1169-1182` | `fastvideo/videos.schema.json`; OpenAI `limit` 0..100 | `rules::openai_videos` |
| `/v1/videos` | `guidance_scale_2` | only `negotiate` (>= 0) | `ge=0, le=20` | **Diff**: now 0..=20 (see below) |
| `/v1/videos`, FastWan | `seed` >= 0 | `videos.rs:492`, `fastwan.rs:120` | int64 (negative allowed) | kept; pinned in `param_rules` |
| MiniMax | `text` <= 7000; `extra.prompt_expansion_mode` set; `page_size` default 20, max 100 | `minimax/src/create.rs:44-50`, `query.rs:194` | MiniMax README / spec example | `rules::minimax` |
| LTX API | default fps 24; A2V audio 20 / 10 s; retake resolutions | `ltxapi/src/models.rs:232`, `request.rs:382-383, 494` | `ltx/openapi.json` | `rules::ltx` |
| fal LTX | default fps 25, default duration 6; pro rates 24/25/50; A2V guidance 1..=50 (ours), pro audio 10 s, 1080p, 24 fps; retake `start_time` 0..=20, duration default 5; ingredient frames 9..=481 (121), fps 1..=60 (24), strengths 0..=2 (1), canvas 1536x896; `LTX_FRAMES_MAX` 481 | `fal/src/schema/ltx.rs:46-51`, `a2v.rs:38-45`, `edit.rs:43-45`, `ingredient.rs:35-44` | fal OpenAPIs | `rules::ltx` |
| fal Wan | frames 17..=161 (81), fps 4..=60 (24), steps 2..=50 (40), guidance 1..=10 (3.5), shift 1..=10 (5), interpolated 0..=4, interpolators | `fal/src/schema/wan.rs:39-51` | fal Wan OpenAPIs | `rules::fal_wan` |
| fal director | `memory` 1..=50; `audio_bitrate` set; `protocol_version` 1; `prompt_version` <= 2^53-1; script beats 64, style 2000, entities 32 / 64 / 200 / 1000 (ours); `chunk_duration` 5 or 10 (ours) | `fal/src/director/messages.rs:18-40, 485-498` | director AsyncAPI | `rules::fal_director` |
| Reactor | `stop_session.reason` <= 64; uploads 64 MiB, 64 per session, id 1..=64 chars | `reactor/src/http.rs:116`, `uploads.rs:28-30, 60` | reactor_sdk | `rules::reactor` |
| `/v1/segments` | `duration_sec` 0.5..=30; `stream_id` 1..=128 chars; `aspect_ratio` sides 1..=64; prompt 800 (fast-h3) | `serve/src/segments.rs:417, 579, 590-596` | ours (docs/serve/segments.md) | `rules::segments` |

### Limits from the caps or the protocol (already one source, unchanged)

`negotiate.rs`: frame grid, rates (`check_fps`), canvas tiers and pixel
budget, the H3 1080P clip cap (`HdTier`, `FLAG_H3_1080P_LONG`), knobs
(steps > 0, guidance and `guidance_scale_2` >= 0, `flow_shift` > 0,
`boundary_ratio` 0..=1, `REFERENCE_STRENGTH_MAX` 1, `REFERENCE_LORA_STRENGTH_MAX`
2), audio-to-video 2..=20 s at 8-192 kHz, edits (`EDIT_*`, `EXTEND_*`: 505
frames, 1920x1088, 8-60 fps, 60 s source, 2-20 s window, 1-20 s context),
reference clip lengths (from `fastvideo_models::h3::reference`). The engine
(`engine-service/src/cuda/validate.rs`) re-checks the geometry per family
through `fastvideo_models` (H3 `H3Geometry`, LTX `check_geometry`, Wan
`check_geometry`) and the edit window. Service limits (body sizes, rate
limits, media byte limits, timeouts, the `[upsampler]` config's own
`memory` 1..50) are configuration, not request parameters.

### Disagreements with the vendored upstreams

| Where | Upstream | This server | Resolution |
|---|---|---|---|
| `/v1/videos` `guidance_scale_2` | 0..=20 | >= 0 (negotiation only) | **Changed** to 0..=20 at parse time, the upstream value (see "Message changes" below) |
| fal `minimax/h3-max*` `duration` | number 0.92..=15, default 5 (the vendored schema moved from integer 5..15) | integer 5..=15 | Kept (widening would accept requests refused today); pinned in `param_rules`; for the owner / E3 |
| fal `minimax/h3-max/reference-to-video` | `middle_image_url`, `middle_frame_time`, `reference_conditions` | not read (unknown fields are ignored) | Kept; E3 |
| FastVideo `seed` | any int64 | >= 0 | Kept (seeds are u64 inside); pinned |
| FastVideo `audio_start_in_s`, `audio_end_in_s` | new optional fields | unknown field: 400 (`extra="forbid"` mirror) | Kept; E3 |
| OpenAI `GET /videos?limit` | 0..=100 | 1..=100 | Kept; pinned |
| fal ingredient `reference_strength` | 0..=2 | parsed 0..=2, negotiation refuses > 1 | Kept (documented in the schema description) |
| MiniMax `model` | `MiniMax-H3`, `-Highspeed`, `-Lite`, `-Max` | `MiniMax-H3`, `-Max`, `-Turbo`, `-Draft` | Kept (design §0.3) |
| MiniMax `GET ?page_size` | example 20, no bounds | rule 1..=100 default 20; the parser refuses 0 but clamps values above 100 to 100 instead of refusing them | Kept (behaviour unchanged); the rule's maximum is the clamp |

### Message changes (the full list)

All on `/v1/videos` `guidance_scale_2`, all keeping status 400, type
`invalid_request_error` and param `guidance_scale_2`
(`tests/params.rs::guidance_scale_2_is_bounded_like_upstream`):

1. above 20: refused with `guidance_scale_2 must be within 0..=20` (the
   `guidance_scale` wording) where it was accepted on a model with the
   `guidance_2` knob;
2. negative: that range message instead of negotiation's
   `guidance_scale_2 must be a non-negative number`;
3. on a model without the `guidance_2` knob, an out-of-range value gets
   the range message instead of `` `guidance_scale_2` is not supported by
   model `<id>` `` (an in-range value still gets the latter);
4. order: the check now runs at parse time, so for a body with several
   faults it is reported before `boundary_ratio`, the task / image checks
   and every negotiation error.

### Found while testing the specs (for E3)

- The native API reports negotiation's errors under the normalized names
  (`duration`, `num_inference_steps`, `guidance_scale`) where its own fields
  are `seconds`, `steps`, `guidance`; `/v1/videos` renames `duration` to
  `seconds` (`fastvideo_param`) but not the others it shares.
- FastWan refuses a lone `height` (or `width: 0` with `height`) on `width`.
- The spec-negotiation agreement tests (`model_agreement`) cover
  `/v1/videos` and native; MiniMax, LTX and fal are covered at parse time
  (`parse_agreement`, and fal's catalog test); their caps-level checks are
  E3's matrix.
