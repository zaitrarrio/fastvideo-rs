# fastvideo-openai-videos

fv-serve adapter for two batch APIs (design WP-06, §4.1-4.2):

| API | Routes |
|---|---|
| FastVideo `fastvideo serve` | `POST /v1/videos` (+ `/v1/videos/generations`), `POST /v1/videos/sync`, `GET /v1/videos`, `GET`/`DELETE /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `GET /v1/models`, `GET /v1/models/{model}`, `GET /v1/model_info` |
| FastWan Video API (`fastwan_link.py`) | `POST /generate`, `GET /status/{prompt_id}`, `GET`/`DELETE /video/{prompt_id}` |

`router(&ctx, VideosConfig)` mounts both. `GET /` and `GET /health` belong to
`fv-serve` (design §9): merge `fastwan::health_body` (`model_loaded`) and
`fastwan::root_body` (`model`) there, or mount `fastwan::service_routes` for a
standalone FastWan server.

Model names: every served name, plus the tier ids `h3-draft` / `h3-turbo` /
`h3-max`, `ltx-draft` / `ltx-turbo` / `ltx-pro`, `wan-*` (design §0.3, §0.6).
Responses carry `metadata: {tier, recipe, quality_gate}` when the model has a
tier or recipe (`quality_gate: false` marks draft results); sync replies carry
`X-FV-Tier` / `X-FV-Recipe`.

Where we differ from FastVideo on purpose:

- `DELETE` cancels a running generation (FastVideo lets it finish).
- `image_url` must be http(s) or a data URI; server-local paths are refused.
- `expires_at` is the real expiry (24 h retention); `file_path` is `null`.
- `lora`, `extra_params`, `video_path`/`video_url`, frame interpolation,
  TeaCache, `true_cfg_scale`, `max_sequence_length`, `sound_duration` and
  `start_time_seconds` answer 400 (no engine path takes them per request).

Tests: `tests/golden.rs` (pure normalize/view/error goldens in
`tests/golden/`, `FV_BLESS=1` rewrites the `*.out.json` files) and
`tests/e2e.rs` (the engine service's fake backend behind serve-kit, driven
over HTTP, including the FastWan flow as `fastwan_link.py` runs it).
