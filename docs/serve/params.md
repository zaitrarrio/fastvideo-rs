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

All three are now rows of the certification matrix (below): the first two
as findings, the third as every MiniMax, LTX and fal parameter driven
through negotiation and the engine's job checks.

## Certification matrix (plan E3, ADR-0016)

`docs/serve/certification.tsv` is every parameter of every served spec,
driven through the real surface with a valid and an invalid value, on the
fake engine serving each shipped config's models with their real caps and
the CUDA pipelines' job checks. `crates/fastvideo-serve/tests/cert_matrix.rs`
generates it and fails when it drifts:

    cargo test -p fastvideo-serve --test cert_matrix                    # check
    FV_CERT_WRITE=1 cargo test -p fastvideo-serve --test cert_matrix    # regenerate
    FV_CERT_WRITE=<file> ...                                            # write elsewhere

The same file's `every_param_has_a_valid_and_an_invalid_case` is the CI
assertion: every parameter of `goldens/param_specs.json`, the director and
every Reactor mode has a valid and an invalid case, or an exemption (below).
Both run in adapter-tests (`scripts/serve/coverage.sh`) and
`scripts/serve/check.sh`; the matrix is about 8,000 requests, 10-25 s on
the build pod.

### Reading the TSV

One row per (section, surface, endpoint, model, engine model, parameter),
sorted. No cell is empty: where a value does not apply the cell says
`n/a (why)`. Several cases in one cell are separated by ` ; `, in the same
order across `invalid_value`, `invalid_status`, `invalid_code` and
`invalid_param`.

| Column | Meaning |
|---|---|
| `section` | `base` (the request every case of the spec starts from), `param` (one parameter), `unknown_field` (the surface's unknown-field policy, probed with `x_fv_cert_unknown`), `pair` (two fields valid alone, refused together) |
| `surface`, `endpoint`, `model`, `engine_model` | the spec (as `/fv/v1/params` serves it; `*` for the director, any app) |
| `configs` | the shipped config the spec was driven on, `(+N)` other configs serving the identical spec |
| `driver` | `http` (in-process `POST` through the assembled router), or `parse (... data channel)`: the director's `messages::parse`, Reactor's `CommandTable::validate` |
| `base` | the base candidate used and its fields; `(routed to X)` when every candidate ran on a tier companion |
| `param`, `type`, `required`, `support`, `domain` | from the spec; `domain` shows the parser's wider domain as `(parse: ...)` |
| `valid_value`, `valid_outcome` | the valid case and what happened (`accepted`, `accepted (routed to X)`, `accepted; engine refuses: ...`, `refused <status> <code> param=<p>`); `(first accepted of N tried)` when earlier set members were refused with the base's other fields; `n/a (refused parameter)` / `n/a (engine gap <code>)`: the parameter has no valid value on this model |
| `invalid_value` | the invalid cases: domain bounds and set misses, off-grid frames, too long text or lists, a malformed media URL (`not-a-url`; `""` for the director), a wrong JSON type `(wrong type)`; for a refused parameter, a value of it |
| `invalid_status`, `invalid_code` | the HTTP status (`n/a (data channel ...)` for the parse drivers) and the surface's code: native `error.kind`, `/v1/videos` / MiniMax / LTX `error.type`, fal `x-fal-error-type`, director `invalid_message`, Reactor `invalid_command`, FastWan none (`{detail}` only) |
| `invalid_param` | the field the refusal names: `error.param` (native, `/v1/videos`), the last `loc` element (fal), the parse error's field (director, Reactor); on FastWan, MiniMax and LTX, whose errors carry no field, `n/a (no param field; message names ...)`, and for FastWan's pairs also the parser's own param |
| `upstream` | the vendored contract (`ours` when none) |
| `finding` | `none`, or each disagreement with the spec: a valid value refused, an invalid one accepted, a refusal naming another field (`reported as X`), a message naming no field, a 5xx |

Media cases are data URIs (a 320x180 PNG, an 8 s WAV, a 21 s 768x512 MP4,
3 s reference clips), shown as `<data: ...>`; long strings as `"a"x5001`.

### Exemptions (parameters with no invalid case)

The cases are planned without this table; an exemption holds only for a
parameter whose plan has no invalid case (it takes any JSON value). An
exempt parameter that has one, or an exemption that matches no parameter,
fails the assertion as stale (`a_stale_exemption_fails_the_assertion`).

| Surface | Parameter | Reason |
|---|---|---|
| `fal_director` | `output`, `x_fv_output` | any JSON value: the output copy is checked by the session (`[director] allow_output`), not by the parser |

### Findings (2026-10-24) and candidate fixes

No request produced a 5xx or a panic. Every finding is recorded, none is
fixed here: each fix changes wire behaviour and is a separate decision.
Counts are TSV rows with the finding (a row can carry several kinds; 1,114 of the 4,630 rows carry at least one).

| # | Surface | Observed (request -> response) | Rows | Candidate fix |
|---|---|---|---|---|
| 1 | native | `seconds: 0.0` -> 400 `invalid_request` `param: "duration"` (also `extend_s`, `end_s` -> `duration`; `steps` -> `num_inference_steps`; `guidance` -> `guidance_scale`; `context_s` -> `context`; `start_s` -> `start_time`; `width` / `height` and width without height -> `size`; `short_edge` -> `resolution`; reference counts -> `references`) | 267 | rename negotiation's params to the native field names when rendering native errors (a native `fastvideo_param`) |
| 2 | native | an input the model's tasks lack (`audio_url`, `image_url`, `last_image_url`, `reference_urls`, image + references) -> 400 `param: "task"` | 81 | name the input field |
| 3 | native | body errors name no field: `seed: -1` (a u64), a wrong JSON type, an unknown field -> 400 `param: null` | 98 | `/v1/videos`' `body_error` (field from serde's unknown / missing field text) |
| 4 | `/v1/videos` | a wrong JSON type (`negative_prompt: 7`, `generate_sound: "x"`) -> 400 `param: null` | 124 | as 3 |
| 5 | `/v1/videos` | `width` / `height` / `video_params.width` / `.height` / `short_edge` over the canvas -> 400 `param: "size"`; reference counts -> `references`; refused reference lists -> `task` or `null` | 235 | report the field sent, or document `size` as the canvas group |
| 6 | `/v1/videos` | `input_reference` / `reference_url` / `image_reference` (one image) on a text-only model (13 models: Wan, FastWan, SF-Wan, FastH3, VDN-H3, Wan 14B plug) -> 400 `model ... does not support task i2v` | 39 | the spec's support follows the model's I2V / keyframe tasks |
| 7 | MiniMax | `content` first-frame image on a model without image-to-video -> 400 (2013) `does not support task i2v`; text-only `content` on `h3-ref2v-*` -> `does not support task t2v` | 33 | the spec's support follows the resolved model's tasks |
| 8 | MiniMax | wrong JSON types (`callback_url: 7`, `content: "x"`) -> 400 `invalid type ... expected a string`, no field named; reference items on a model without reference-to-video -> `reference-to-video is not enabled`, the item not named | 172 | name the field in the message (MiniMax has no param field) |
| 9 | LTX | `/v2/audio-to-video` with a 2 s audio (the spec's minimum) -> 400 `the audio is 2.00 s, shorter than the model's shortest clip (145 frames at 24 fps)`; audio length errors do not name `audio_uri` | 8 | derive the audio minimum from the caps (frames.min / fps) |
| 10 | fal | `fal-ai/ltx-2.3-quality/ingredient` `num_frames: 242` (off the 8k+1 grid) -> accepted (aligned up); Wan `interpolator_model: "none"`, `num_interpolated_frames: 0` -> accepted though `refused` | 7 | the spec says "aligned up" (wire range) and marks the default values `no_op` |
| 11 | native | `ltx25-a2v-guided`: `last_image_url` (spec: refused) accepted, `prompt: " "` accepted with audio + image (the note's exception), `seconds` 10.02 s refused against the 8 s audio; `ltx25-distill-dense`: `reference_urls` (spec: refused) accepted, routed to `ltx25-ref2v` | 4 | the spec reflects tier routing and the audio-to-video exceptions |
| 12 | `/v1/videos` | `ltx25-ref2v`, `ltx25-a2v-guided`: every base routes to `ltx25-distill-dense`; then `seconds: 11`, `num_frames: 249` are accepted (the companion's caps) | 6 | specs for routed names describe the companion, or the surface lists only models it can run |
| 13 | director | `configure` / `prompt` with `end_image_url: ""` or `audio_url: ""` -> accepted (only `image_url` and script beats are checked); enum and type errors name no field | 11 | `check_url` on those fields |
| 14 | FastWan | `{"prompt", "height": 480}` -> 400 `width and height must both be positive`: the parser's param is `width` (invisible on FastWan's wire, which has no param field); a canvas over the pixel budget names neither side | 6 | name `height` in the parser (no wire change) |
| 15 | MiniMax, LTX | `prompt_optimizer: "x"` (MiniMax), `enhance_prompt: "x"` (LTX) -> accepted: a non-boolean is ignored, as the notes say | 44 | none needed if intended (the notes document it); else refuse a non-boolean |

The pairs section also records the mutual exclusions that hold: MiniMax
first-frame + reference items, two text items; LTX audio-to-video
`last_frame_uri` without `image_uri`, no prompt and no image;
`/v1/videos` `input_reference` + `reference_url` / `image_reference`;
native retake + extend fields (named `video_url`, finding 1's kind).
