# fastvideo-ltxapi

fv-serve adapter for the LTX API (`api.ltx.io` wire format): `/v2/*` async
jobs, `/v1/*` sync generation, `/v1/upload` with `ltx://uploads/` refs, and
`403 permission_error` stubs for endpoints without an engine path. Design:
[docs/serve/design.md](../../docs/serve/design.md) §4.5 (WP-08); wire spec:
[docs/serve/research-ltx-api.md](../../docs/serve/research-ltx-api.md).

## Routes (`fastvideo_ltxapi::router(LtxConfig)`)

| Route | Behaviour |
|---|---|
| `POST /v2/{text,image}-to-video` | `202 {id, created_at}` |
| `GET /v2/{text,image}-to-video/{id}` | `pending` / `processing` / `completed` (`result.video_url`) / `failed` (`error`); a job from the other endpoint, another key, or `/v1` is `404` |
| `POST /v1/{text,image}-to-video` | sync: `200 video/mp4` bytes; over the sync timeout `504` (job cancelled); per-key concurrency limit `429 concurrency_limit_error` |
| `POST /v1/upload` | `200 {upload_url, storage_uri: "ltx://uploads/<token>", expires_at, required_headers: {}}` |
| `POST /v1\|v2/{audio-to-video,retake,extend,video-to-video-hdr,video-to-video-reframe}` | `403 permission_error` |
| `GET /v2/{those}/{id}` | `404 not_found_error` |

Every reply carries `x-request-id` (32 hex). Auth is `Authorization: Bearer`.
`PUT /uploads/{token}` and `GET /files/...` come from serve-kit
(`ServeCtx::routes`).

## Models

`ltx-2-5-pro`, `ltx-2-3-pro` → tier max; `ltx-2-5-fast`, `ltx-2-3-fast`,
`ltx-turbo` → tier turbo; `ltx-draft` → tier draft. Targets are configurable
(`LtxModels::retarget`). A known id with no served model is
`403 permission_error`; `ltx-2-fast` / `ltx-2-pro` and unknown ids are `400`.
The ltx §3 resolution × fps × duration matrix is enforced (`ltx-turbo` and
`ltx-draft` follow the `*-fast` rows). Responses carry `x-fv-tier`,
`x-fv-recipe` and, for draft, `x-fv-quality: draft`.

## Engine gaps (400 `invalid_request_error`, never approximated)

`fps` other than the engine's (24 until E4), `duration: null`,
`camera_motion`, `last_frame_uri` (until E9), `image_uri` on a model without
I2V. Canvases off the engine's multiple are generated larger and cropped
back (`1920x1080` → `1920x1088`, `1280x720` → `1280x768`, portrait
transposed).
