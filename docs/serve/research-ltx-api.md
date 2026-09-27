# LTX API (api.ltx.io): wire-compatibility research

Status: research note, 2026-09-27. Scope: the exact HTTP contract a Rust server
in front of the fastvideo-rs CUDA engine has to speak so that existing LTX API
clients (plain HTTP code written against the LTX docs, LTX-Desktop's API
client) can target it instead of `https://api.ltx.io`. Nothing here is
implemented yet.

## 0. Sources and conventions

Every fact carries a source tag. Anything I derived rather than read is marked
**INFERRED**. Anything the LTX docs do not specify is marked **UNDOCUMENTED**;
I did not fill those gaps with guesses. All pages were fetched on 2026-09-27
as Markdown (the docs serve clean Markdown when `.md` is appended to a page
URL, per https://docs.ltx.io/llms.txt).

| Tag | Source |
|---|---|
| **IDX** | https://docs.ltx.io/llms.txt (docs index, lists every API reference page) |
| **OAS** | https://docs.ltx.io/openapi.json (OpenAPI 3.1.0, `info.version` 1.0.0). The machine-readable contract; schemas below are copied from it |
| **WEL** | https://docs.ltx.io/welcome |
| **QS** | https://docs.ltx.io/quickstart |
| **AUTH** | https://docs.ltx.io/authentication |
| **MOD** | https://docs.ltx.io/models |
| **M25** | https://docs.ltx.io/models/ltx-2-5 |
| **M23** | https://docs.ltx.io/models/ltx-2-3 |
| **IN** | https://docs.ltx.io/input-formats |
| **MIG** | https://docs.ltx.io/migrate-v1-to-v2 |
| **ASY** | https://docs.ltx.io/async-jobs |
| **RL** | https://docs.ltx.io/rate-limits |
| **ERR** | https://docs.ltx.io/errors |
| **DBG** | https://docs.ltx.io/debugging |
| **PRC** | https://docs.ltx.io/pricing |
| **PG** | https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide |
| **R-T2V2 / R-I2V2 / R-A2V2 / R-RET2 / R-EXT2 / R-HDR2 / R-RF2 / R-JOB** | https://docs.ltx.io/api-documentation/api-reference/async-video-generation/{submit-text-to-video, submit-image-to-video, submit-audio-to-video, submit-retake, submit-extend, submit-video-to-video-hdr, submit-video-to-video-reframe, get-job-status} |
| **R-T2V1 / R-I2V1 / R-A2V1 / R-RET1 / R-EXT1** | https://docs.ltx.io/api-documentation/api-reference/video-generation/{text-to-video, image-to-video, audio-to-video, retake, extend} |
| **R-UP** | https://docs.ltx.io/api-documentation/api-reference/upload/create-upload |
| **CL-yyyy-mm-dd** | https://docs.ltx.io/api-changelog/yyyy/m/d (every entry listed at https://docs.ltx.io/api-changelog/llms.txt was read) |
| **DESK** | https://github.com/Lightricks/LTX-Desktop at `68cd86c` (2026-08-26): `backend/ltx2_server.py`, `backend/services/ltx_api_client/ltx_api_client_impl.py`, `backend/services/text_encoder/ltx_text_encoder.py`, `backend/tests/test_ltx_api_client.py` |
| **GHORG** | https://github.com/orgs/Lightricks/repositories |
| Local | Paths in this repo, cited inline |

---

## 1. Base URL, auth, versioning, limits, errors

### 1.1 Base URLs

- Production: `https://api.ltx.io` (OAS `servers[0]`, "Production server").
- Legacy host: `https://api.ltx.video`. It still works: "Existing integrations
  on `https://api.ltx.video` keep working unchanged" (CL-2026-07-21).
  LTX-Desktop still hardcodes it (DESK `ltx2_server.py`:
  `LTX_API_BASE_URL = "https://api.ltx.video"`).
- Paths are absolute from the host root, with the version as the first path
  segment (`/v1/...`, `/v2/...`) (OAS `paths`).

### 1.2 Authentication

- Every operation has `security: [{bearerAuth: []}]`; the scheme is
  `type: http, scheme: bearer`, "API key authentication" (OAS
  `components.securitySchemes.bearerAuth`).
- Header: `Authorization: Bearer YOUR_API_KEY` (AUTH).
- Keys come from the developer console at https://console.ltx.io (WEL, AUTH).
  Key format is **UNDOCUMENTED**.
- The docs suggest (not require) the env var `LTXV_API_KEY` in examples (AUTH);
  other examples use `$LTX_API_KEY` (M25). No client reads either
  automatically: there is no SDK (§4).
- Failure: `401` with
  `{"type":"error","error":{"type":"authentication_error","message":"Invalid API key"}}`.
  Listed causes: missing header, invalid/expired key, malformed header (not
  `Bearer <key>`) (AUTH).

### 1.3 Versioning

There is no version header; the version is the path prefix (OAS).

| Prefix | Style | Status |
|---|---|---|
| `/v1/{text-to-video,image-to-video,audio-to-video,retake,extend}` | sync: one request, `200` body is the MP4 | `deprecated: true` in OAS. "After October 26, 2026 at 11:59 PM UTC, requests to this endpoint will no longer work" (R-T2V1 and siblings; CL-2026-09-24) |
| `/v1/upload` | JSON, signed-URL issuance | Current; stays on `/v1` (MIG, CL-2026-09-24) |
| `/v2/{endpoint}` + `GET /v2/{endpoint}/{id}` | async jobs | Current, "recommended for production" (ASY) |

V2 "uses the same authentication and generation request parameters as V1"
(MIG); OAS confirms v1 and v2 of each endpoint `$ref` the same request schema.

### 1.4 Rate and concurrency limits

- Concurrency limit (sync API): default **2** concurrent generations per
  account; higher on request (RL).
- Rate limit (async API): a maximum number of V2 jobs **queued per
  organization**; exceeding it returns `429 rate_limit_error` (RL). The actual
  number is **UNDOCUMENTED**.
- Both return `429 Too Many Requests`. "Concurrency limit errors include a
  `Retry-After` header indicating seconds to wait" (RL). Clients are told to
  honour `Retry-After` on any `429` (ERR).
- Message strings shown: `"Too many concurrent requests. Please try again
  later."` (`concurrency_limit_error`) and `"Queue limit exceeded. Please try
  again later."` (`rate_limit_error`) (RL).

### 1.5 Error format

Body schema `Error` (OAS):

```json
{ "type": "error", "error": { "type": "<error_type>", "message": "<human text>" } }
```

- `type`: enum `["error"]`, required. `error.type`: string, required.
  `error.message`: string, required (OAS `Error`, `ErrorError`).
- The same `{type, message}` object appears as `error` inside a failed V2 job
  status (OAS `V2JobError`; ERR; ASY).

Error types (ERR):

| HTTP | `error.type` | Retry? | Meaning |
|---|---|---|---|
| 400 | `invalid_request_error` | No | Invalid parameters |
| 401 | `authentication_error` | No | Key missing or invalid |
| 402 | `insufficient_funds_error` | No | Not enough credits |
| 403 | `permission_error` | No | Endpoint not available for the account |
| 404 | `not_found_error` | No | Job doesn't exist or has expired |
| 422 | `content_filtered_error` | No | Rejected by safety filters |
| 429 | `concurrency_limit_error` | Yes | Too many concurrent requests (sync) |
| 429 | `rate_limit_error` | Yes | Queue limit exceeded (async) |
| 500 | `api_error` | Yes | Unexpected server error |
| 503 | `service_unavailable_error` | Yes | Temporarily unavailable |
| 529 | `overloaded_error` | Yes | Temporarily overloaded |

Per-operation declared responses (OAS):

- V2 submit: `202`, `400`, `401`, `402`, `422`, `429`, `500`, `503`.
- V2 job status: `200`, `401`, `404`, `500`.
- V1 generation: `200`, `400`, `401`, `422`, `429`, `500`, `503`, `504`
  ("Request timeout"). `402` is not listed on V1 in OAS although ERR lists it
  generally.
- `/v1/upload`: `200`, `401`, `500`, `503`.

Example validation message from the docs: omitting `duration` returns
`duration is required` (M25). Other message strings are **UNDOCUMENTED**.

### 1.6 Response headers

- `x-request-id`: unique id per request, e.g.
  `1234567890abcdef1234567890abcdef` (32 hex chars in the example) (DBG, QS).
- V1 success: QS says `Content-Type: video/mp4`; OAS declares the `200`
  content as `application/octet-stream` (`type: string, format: binary`).
  The two sources disagree; emitting `video/mp4` matches the prose docs and is
  what LTX-Desktop checks for a non-JSON body (DESK
  `_run_video_edit` reads `Content-Type` to decide between bytes and JSON)
  (**INFERRED**: `video/mp4` is the safer choice).

---

## 2. Endpoints

### 2.0 Shared rules

**Media inputs** (`image_uri`, `last_frame_uri`, `video_uri`, `audio_uri`)
accept three forms (IN):

| Form | Example | Max size |
|---|---|---|
| Upload URI from `/v1/upload` | `ltx://uploads/abc-123` | 200 MB upload cap; each endpoint applies its own limit at generation time (images 15 MB) |
| HTTPS URL | `https://...` | images 15 MB (10 s fetch timeout); video/audio 32 MB (30 s). HTTPS only, domain names only (no IPs), publicly accessible, **no redirects** |
| Data URI | `data:{mime};base64,{data}` | encoded: images 7 MB, video/audio 15 MB |

Per-endpoint video caps: `extend` 100 MB, `video-to-video-reframe` 200 MB
(upload or HTTPS only); data URIs keep their own lower limit (IN).

Accepted formats (IN):

- Images: PNG `image/png`, JPEG `image/jpeg`, WEBP `image/webp`.
- Video: MP4 `video/mp4`, MOV `video/quicktime`, MKV `video/x-matroska`;
  codecs H.264, H.265.
- Audio: WAV `audio/wav` (AAC-LC, MP3, Vorbis, FLAC), MP3 `audio/mpeg`, M4A
  `audio/mp4` (AAC-LC), OGG `audio/ogg` (Opus, Vorbis). AAC must be AAC-LC;
  HE-AAC/HE-AACv2 rejected. "PCM audio is not supported" (IN).

**`camera_motion` enum** (T2V, I2V, A2V; OAS):
`dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down`,
`static`, `focus_shift`. No default (optional, omitted = none) (OAS; CL-2026-01-01).

**`prompt`**: `maxLength: 5000` on every endpoint that has one (OAS).

**Audio in outputs**: "All endpoints return video with synchronized audio —
dialogue, music, and ambient sound are generated together with the visuals"
(WEL). T2V/I2V take `generate_audio` (default `true`); `false` gives silent
video (OAS). The **output audio codec, sample rate and channel count are
UNDOCUMENTED**, as are the output video codec, bitrate and container options
beyond "MP4" (QS).

**Resolution strings** are `WIDTHxHEIGHT` (OAS A2V/retake descriptions). T2V
and I2V `resolution` is a free `string` in OAS; allowed values come from the
model support matrix (§3).

### 2.1 Async (V2) job model

Lifecycle (ASY, R-JOB):

1. `POST /v2/{endpoint}` → `202 Accepted`, body `V2JobCreatedResponse`.
2. `GET /v2/{endpoint}/{id}` → `200`, body `V2JobStatusResponse`. The
   `{endpoint}` segment must be the one used at submit; path enum:
   `text-to-video`, `image-to-video`, `audio-to-video`, `retake`, `extend`,
   `video-to-video-hdr`, `video-to-video-reframe` (OAS
   `V2EndpointIdGetParametersEndpoint`).
3. On `completed`, download from the URL(s) in `result`.

`V2JobCreatedResponse` (OAS):

| Field | Type | Req | Notes |
|---|---|---|---|
| `id` | string | yes | Examples are UUIDs, e.g. `a1b2c3d4-e5f6-7890-abcd-ef1234567890` (ASY) |
| `created_at` | string, date-time | yes | ISO 8601, e.g. `2026-09-06T12:00:00.000Z` (MIG) |

`V2JobStatusResponse` is `oneOf` discriminated by `status` (OAS):

| `status` | Required fields | Extra |
|---|---|---|
| `pending` | `status`, `id`, `created_at` | "Job is queued" |
| `processing` | `status`, `id`, `created_at` | "Generation is running" |
| `completed` | `status`, `id`, `created_at`, `completed_at`, `result` | `result`: object, `additionalProperties: {type: string, format: uri}` |
| `failed` | `status`, `id`, `created_at`, `completed_at`, `error` | `error`: `{type, message}` with the HTTP error types |

Result keys: most endpoints return `video_url`; `video-to-video-hdr` returns
`exr_frames_url` (a ZIP of per-frame EXR images) (ASY, R-HDR2). Example:

```json
{ "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "status": "completed",
  "created_at": "2026-01-15T10:00:00.000Z", "completed_at": "2026-01-15T10:02:30.000Z",
  "result": { "video_url": "https://storage.googleapis.com/example/video.mp4" } }
```

Behavioural rules (ASY, R-JOB):

- Transitions `pending → processing → completed | failed`; the last two are
  terminal.
- Clients are told to poll every ≥ 5 s with jitter (5–6 s).
- Job status is kept up to **24 h** after the terminal state ("can be removed
  sooner"); afterwards `GET` returns `404` (`not_found_error`).
- Output URLs "expire independently of job status"; the expiry time is
  **UNDOCUMENTED**. The example host is `storage.googleapis.com` (a signed GCS
  URL, **INFERRED**). Download needs no `Authorization` header (every doc
  example fetches `result.video_url` without one; ASY).
- Webhooks / callbacks: **none documented**. OAS has no `callbacks` or
  `webhooks` section and no request field for a callback URL. Polling is the
  only completion mechanism.
- Cancellation / listing jobs: **no endpoint** (OAS has only submit and
  status).

### 2.2 Sync (V1) model

- `POST /v1/{endpoint}` with the same JSON body; `200` returns the file
  directly (MIG, R-T2V1). One connection stays open for the whole generation
  (MIG). Timeout surfaces as `504` (OAS). Failures are returned on the same
  request (MIG).
- Retired after 2026-10-26 23:59 UTC (CL-2026-09-24). Supporting it is cheap
  for us (**INFERRED**: same handler, block until done, stream the file) and
  keeps LTX-Desktop working, since DESK still calls `/v1/text-to-video`,
  `/v1/image-to-video`, `/v1/audio-to-video`, `/v1/retake` and `/v2/extend`
  (DESK tests).

### 2.3 `POST /v2/text-to-video` (and deprecated `POST /v1/text-to-video`)

Schema `TextToVideoRequest` (OAS, R-T2V2):

| Field | Type | Req | Default | Constraints / enum |
|---|---|---|---|---|
| `prompt` | string | yes | | maxLength 5000 |
| `model` | string enum | yes | | `ltx-2-3-fast`, `ltx-2-3-pro`, `ltx-2-5-fast`, `ltx-2-5-pro` |
| `duration` | integer \| null | yes (key must be present) | | per-model list (§3). `null` = automatic duration, 2.5 only |
| `fps` | integer | no | 24 | per-model/resolution list (§3) |
| `resolution` | string | yes | | per-model list (§3) |
| `generate_audio` | boolean | no | true | false = silent video |
| `camera_motion` | enum | no | | see §2.0 |

Billed per second of generated video (R-T2V2). Result key `video_url`.

### 2.4 `POST /v2/image-to-video` (and `/v1/image-to-video`)

Schema `ImageToVideoRequest` (OAS, R-I2V2): all T2V fields, plus:

| Field | Type | Req | Notes |
|---|---|---|---|
| `image_uri` | string | yes | First frame |
| `last_frame_uri` | string | no | Last frame; video interpolates first → last (CL-2026-03-05 added it on 2.3; M25 lists it on 2.5) |

Required: `image_uri`, `prompt`, `model`, `duration`, `resolution` (OAS).
`duration: null` "Cannot be combined with `last_frame_uri`" (OAS, M25). Model
enum is the same four ids. "The output preserves the visual identity of the
source image" (R-I2V2). How a source image whose aspect differs from
`resolution` is fitted (crop / pad / resize) is **UNDOCUMENTED**.

### 2.5 `POST /v2/audio-to-video` (and `/v1/audio-to-video`)

Schema `AudioToVideoRequest` (OAS, R-A2V2):

| Field | Type | Req | Default | Notes |
|---|---|---|---|---|
| `audio_uri` | string | **yes** | | Soundtrack; **sets the output length** |
| `image_uri` | string | conditional | | First frame. "Required if prompt is not provided" |
| `prompt` | string | conditional | | maxLength 5000. "Required if image_uri is not provided. Can be empty string when image_uri is provided" |
| `resolution` | string | no | from image orientation | Portrait image → `1080x1920`, landscape → `1920x1080`; no image → `1920x1080` |
| `model` | enum | no | `ltx-2-3-pro` | `ltx-2-3-pro`, `ltx-2-5-fast`, `ltx-2-5-pro` (no `ltx-2-3-fast`) |
| `fps` | integer | no | 24 | |
| `last_frame_uri` | string | no | | "Requires `image_uri`" |
| `camera_motion` | enum | no | | |

There is no `duration` and no `generate_audio` field. Maximum input audio
length (OAS `audio_uri` description):

| Model | 720p, 1080p | 1440p, 4K |
|---|---|---|
| `ltx-2-5-fast` | 20 s | 10 s |
| `ltx-2-5-pro` | 10 s | 10 s |
| `ltx-2-3-pro` | 20 s | 10 s |

Billed per second of input audio (R-A2V2). Whether the output track is the
input audio passed through or re-generated is **UNDOCUMENTED** (the welcome
page says the API "produces visuals synchronized to the audio", WEL).
`fps`, `last_frame_uri`, `camera_motion` on A2V date from CL-2026-08-19.

### 2.6 `POST /v2/retake` (and `/v1/retake`)

Schema `EditVideoRequest` (OAS, R-RET2):

| Field | Type | Req | Default | Constraints |
|---|---|---|---|---|
| `video_uri` | string | yes | | Max 3840x2160; min 73 frames (~3 s at 24 fps) |
| `start_time` | number (double) | yes | | minimum 0. Section clamped to video length |
| `duration` | number (double) | yes | | minimum 2. Clamped to video length |
| `prompt` | string | no | | maxLength 5000; what happens in the section |
| `mode` | enum | no | `replace_audio_and_video` | `replace_audio`, `replace_video`, `replace_audio_and_video` |
| `resolution` | enum | no | from input orientation | `1920x1080`, `1080x1920` |
| `model` | enum | no | `ltx-2-3-pro` | `ltx-2-3-pro` only |

Billed per second of the input video (R-RET2). Result `video_url`.

### 2.7 `POST /v2/extend` (and `/v1/extend`)

Schema `ExtendVideoRequest` (OAS, R-EXT2):

| Field | Type | Req | Default | Constraints |
|---|---|---|---|---|
| `video_uri` | string | yes | | Aspect 16:9 or 9:16; max 3840x2160; min 73 frames. Output keeps input resolution. Up to 100 MB (IN) |
| `duration` | number (double) | yes | | 2 ≤ d ≤ 20 s ("480 frames at 24fps") |
| `prompt` | string | no | | maxLength 5000 |
| `mode` | enum | no | `end` | `start`, `end` |
| `model` | enum | no | `ltx-2-3-pro` | `ltx-2-3-pro` only |
| `context` | number (double) | no | maximise within limits | 1 ≤ c ≤ 20 s of input used as context. `context + duration` in frames (at input fps) ≤ **505** |

"Audio is generated for the extended portion if the input video has audio"
(R-EXT1). Billed on extended portion + context frames, capped at 505 frames
(R-EXT2, PRC).

### 2.8 `POST /v2/video-to-video-hdr` (async only)

Schema `VideoToVideoHdrRequest` (OAS, R-HDR2): one field, `video_uri`
(string, required), "SDR source video in HTTPS URL or base64 data URI format".
Frame caps by input tier (snap to the smallest tier by pixel count):

| Tier | Max frames | ~Max at 24 fps |
|---|---|---|
| ≤ 1920×1080 | 181 | 7 s |
| ≤ 2560×1440 | 101 | 4 s |
| ≤ 3840×2160 | 41 | 2 s |

Output keeps input resolution. Result key `exr_frames_url`: ZIP of per-frame
EXR (R-HDR2, ASY). The docs index titles it "Upscale video to HDR" (IDX), but
it is dynamic-range conversion, not spatial upscale (R-HDR2). Pricing lists
it under `ltx-2-3-pro` (PRC). No `model` field.

### 2.9 `POST /v2/video-to-video-reframe` (async only)

Schema `VideoToVideoReframeRequest` (OAS, R-RF2):

| Field | Type | Req | Notes |
|---|---|---|---|
| `video_uri` | string | yes | Max 60 s and 1800 frames; up to 200 MB (IN) |
| `resolution` | enum | yes | `720x720`, `1080x1080` (1:1); `720x900`, `1080x1350` (4:5); `900x720`, `1350x1080` (5:4); `720x1280`, `1080x1920` (9:16); `1280x720`, `1920x1080` (16:9) |

Fills the missing area with generated content (outpainting) (R-RF2). A
`model` field, if sent, is ignored (CL-2026-09-23). Result `video_url`.

### 2.10 `POST /v1/upload`

No request body (OAS has no `requestBody`; the example posts with only the
auth header, R-UP). `200` body `UploadResponse` (OAS):

| Field | Type | Notes |
|---|---|---|
| `upload_url` | string | Pre-signed URL; client `PUT`s the file there; expires in 1 h |
| `storage_uri` | string | e.g. `ltx://...`; use as `image_uri` / `video_uri` / `audio_uri`; file lives 24 h |
| `expires_at` | date-time | Signed-URL expiry |
| `required_headers` | map string→string | Must be sent unchanged on the `PUT`; example `x-goog-content-length-range: 0,209715200`, `x-goog-if-generation-match: 0` |

The URL "creates a new object and cannot overwrite an existing object"
(R-UP). The client also sets `Content-Type` on the `PUT` in the example
(R-UP).

**INFERRED** for our server: issue a `upload_url` that points back at our own
host (e.g. `PUT /uploads/{token}`), return `ltx://uploads/{token}`, and accept
(ignore or enforce) the two `x-goog-*` headers, since clients copy
`required_headers` blindly.

### 2.11 Features that do not exist in the API

- No separate spatial upscale endpoint (the only "upscale" is HDR, §2.8).
- No keyframe list beyond `image_uri` + `last_frame_uri`.
- No `seed`, `negative_prompt`, `guidance_scale`, `steps` or `enhance_prompt`
  field on any endpoint (OAS). The prompting guide says the prompt enhancer
  flag is for local pipelines and "For direct API requests, do not use
  `--enhance-prompt`" (PG). Launch notes mention "built-in prompt
  enhancement" on the hosted API (CL-2025-10-29, CL-2025-12-09), which
  suggests the hosted service enhances server-side without a switch
  (**INFERRED**).
- Multi-shot is prompt-driven on 2.5, not a field (M25, PG).
- No webhooks, no job cancel, no job list, no streaming.

### 2.12 Undocumented endpoint seen in a first-party client

LTX-Desktop calls `POST {base}/v1/prompt-embedding` with the bearer key and a
JSON body including `model_id`, and unpickles the response as torch tensors
(video context 4096 wide, optional audio context after it) (DESK
`ltx_text_encoder.py`). It is not in OAS or the docs. **Not a compatibility
target** unless we want to serve LTX-Desktop's remote text encoding; the
response is a Python pickle.

---

## 3. Models and their limits

Model ids (OAS enums, MOD): `ltx-2-5-fast`, `ltx-2-5-pro`, `ltx-2-3-fast`,
`ltx-2-3-pro`. `ltx-2-fast` / `ltx-2-pro` were removed on 2026-08-16 and now
return an error (CL-2026-08-16). There is no `model` default on T2V/I2V
(required); A2V, retake and extend default to `ltx-2-3-pro` (OAS).

Endpoint support (MOD):

| Endpoint | 2-5-fast | 2-5-pro | 2-3-fast | 2-3-pro |
|---|---|---|---|---|
| text-to-video | ✓ | ✓ | ✓ | ✓ |
| image-to-video | ✓ | ✓ | ✓ | ✓ |
| audio-to-video | ✓ | ✓ | — | ✓ |
| retake | — | — | — | ✓ |
| extend | — | — | — | ✓ |

HDR and reframe have no model selection; pricing lists both as `ltx-2-3-pro`
(PRC).

Support matrix for T2V/I2V (identical for 2.5 and 2.3; M25, M23):

| Model | Resolution | FPS | Duration (s) |
|---|---|---|---|
| `*-fast` | 720p, 1080p | 24, 25 | 6, 8, 10, 12, 14, 16, 18, 20 |
| `*-fast` | 720p, 1080p | 48, 50 | 6, 8, 10 |
| `*-fast` | 1440p, 4K | 24, 25, 48, 50 | 6, 8, 10 |
| `*-pro` | 720p, 1080p, 1440p, 4K | 24, 25, 48, 50 | 6, 8, 10 |

Resolution strings (M25, M23):

| Tier | 16:9 | 9:16 |
|---|---|---|
| 720p | `1280x720` | `720x1280` |
| 1080p | `1920x1080` | `1080x1920` |
| 1440p | `2560x1440` | `1440x2560` |
| 4K | `3840x2160` | `2160x3840` |

Automatic duration (M25): `"duration": null` on `ltx-2-5-*` T2V/I2V; the
model picks the length, never above the longest duration allowed for that
resolution/fps; not with `last_frame_uri`. The key is still required.

Prices per second (PRC), for context on how "pro" and "fast" differ in cost:
T2V/I2V `ltx-2-5-fast` $0.09/0.13/0.19/0.30, `ltx-2-5-pro`
$0.12/0.17/0.25/0.39, `ltx-2-3-fast` $0.03/0.06/0.12/0.24, `ltx-2-3-pro`
$0.04/0.08/0.16/0.32 (720p/1080p/1440p/4K).

### 3.1 Mapping to our engine

What the engine runs (Local):

- `ltx2 gen` (`crates/fastvideo-gpucheck/src/ltx2_stage.rs`, `Gen`):
  `--model-version` `2.0` | `2.3` | `2.5`, all **distilled**; `--two-stage`
  (half-res stage 1 → ×2 latent upsampler → 3-step stage 2, needs 2.3 or 2.5);
  `--sol-stage2` (2.5, on by default for 2.5 two-stage) / `--dense-stage2` /
  `--pisa-stage2` (2.3); `--diff-vae` (2.5); `--image` (first-frame I2V);
  `--seed` (default 10); `--ltx-tae-weights`; geometry `--workload`
  `4k5s` (3840×2176, 121 f) | `1080p20s` (1920×1088, 481 f), both 24 fps,
  each field overridable with `--height/--width/--num-frames/--frame-rate`.
- 512p (768×512×121) is the oracle / validation canvas for 2.5 two-stage,
  dense and Sol stage 2 (`docs/oracle.md` §"LTX-2.5 distilled two-stage,
  512p"; `docs/ports/ltx25.md` §"Two-stage distilled").
- Presets (`docs/scope.md` §"LTX-2 / 2.3 / 2.5"): `ltx2_distilled_23`
  (t2av, i2v), `ltx2_base_23` (30-step CFG; t2av, i2v), `ltx2_distilled_25`
  (t2av only), plus 2.0 presets. `FASTVIDEO_LTX2_HQ=1` selects the 2.3-base
  15+3 HQ contract.
- Validation (`crates/fastvideo-cudarc/src/ltx2/pipeline.rs` `validate`):
  H and W multiples of 32 (64 when two-stage), `num_frames = 8k + 1`,
  positive frame rate, non-empty prompt.
- Output: MP4 via ffmpeg, H.264 (`libx264`, crf 19, `yuv420p`) + AAC 192 kb/s
  (`crates/fastvideo-cudarc/src/wan/writer.rs`), plus `audio.wav`. Audio is
  always generated (joint t2av). Vocoder rate: 24 kHz stereo on 2.0
  (`docs/ports/ltx2.md` §"Shapes"), 48 kHz BWE on 2.5
  (`docs/ports/ltx25.md`).
- Out of scope for 2.5 today: duration head, prompt enhancer,
  multishot/keyframes, dev DiT + CFG/STG, official 1536×1024 canvas
  (`docs/ports/ltx25.md` §"Out of scope").
- I2V: first-frame encode landed; per-token timesteps deferred; falls back to
  a spatial stub if `vae/` lacks encoder keys (`docs/ports/ltx2.md`
  §"I2V encode").

Proposed id mapping (all **INFERRED**; Lightricks does not publish what the
hosted `fast` / `pro` variants run):

| API id | Local route | Confidence / caveat |
|---|---|---|
| `ltx-2-5-fast` | `ltx2_distilled_25`, two-stage, Sol stage 2 (`--model-version 2.5 --two-stage`) | Good fit: "fast" = distilled is the natural reading. T2V only today (2.5 preset lists t2av only) |
| `ltx-2-5-pro` | `ltx2_distilled_25`, two-stage, `--dense-stage2` (optionally `--diff-vae`) | Weak: "pro" likely means the non-distilled dev DiT with guidance, which is out of scope for us. Serving distilled under the pro name is a quality downgrade we must document |
| `ltx-2-3-fast` | `ltx2_distilled_23`, two-stage | Good fit |
| `ltx-2-3-pro` | `ltx2_base_23` (30-step CFG) or `FASTVIDEO_LTX2_HQ=1` 15+3 | Plausible; `ltx2 gen` itself only exposes distilled versions, so the base route needs a non-`gen` entry point |
| `ltx-2-fast` / `ltx-2-pro` | `ltx2_distilled_20` | Removed upstream (CL-2026-08-16); return `400 invalid_request_error` to match |

Geometry mapping (**INFERRED** from the validation rules above):

| API request | Engine canvas | Note |
|---|---|---|
| `1920x1080`, 20 s, 24 fps (fast only) | 1920×1088×481 = `1080p20s` | Exactly our workload; crop 8 rows on output |
| `3840x2160`, 24 fps | 3840×2176, 8k+1 frames | `4k5s` is 5 s (121 f), but the API's shortest duration is **6 s**; 6/8/10 s at 4K = 145/193/241 frames, not measured |
| `1280x720` / `720x1280` | 1280×768 (pad to ×64) → crop 720 | 720 is not a multiple of 32 or 64 |
| `2560x1440` / `1440x2560` | native (both ×64) | Not measured |
| portrait (`1080x1920`, `2160x3840`) | transpose of the above | Engine accepts any H/W multiple; never validated portrait |
| `fps` 25/48/50 | `--frame-rate` | Accepted by validation; never measured. Frame count = `ceil(d·fps)` rounded up to 8k+1, then trim (e.g. 6 s @ 25 = 150 → 153) |
| 512p | not an API tier | Only useful as a non-standard extension |

Duration → frames: `d·fps` must be rounded to `8k+1` (6 s@24 → 145, 8 s → 193,
10 s → 241, 20 s → 481). The API does not say whether its own outputs are
exactly `d·fps` frames; **UNDOCUMENTED**.

---

## 4. SDKs and clients

- **No official SDK** for the hosted API is documented. The docs' "SDK Code"
  tabs are raw HTTP snippets (Python `requests`, TypeScript `fetch`, Go
  `net/http`, Java Unirest, cURL) with the URL hardcoded as
  `https://api.ltx.io/...` (R-T2V2 and every other reference page, ASY, MIG).
  Lightricks' GitHub org lists inference, ComfyUI, trainer and desktop repos
  but no API client (GHORG). Searches of PyPI (`ltx-api`) and npm (`ltx-api`)
  returned 404.
- So "pointing a client at our server" means changing the host string in the
  user's own code: `https://api.ltx.io` → our base URL. Nothing reads an env
  var for the base URL.
- **LTX-Desktop** (first-party app) has a real client,
  `LTXAPIClientImpl(http, ltx_api_base_url)`, which strips a trailing `/` and
  appends `/v1/...` or `/v2/...`. The base URL comes from
  `RuntimeConfig.ltx_api_base_url`, set from the module constant
  `LTX_API_BASE_URL = "https://api.ltx.video"` in `backend/ltx2_server.py`
  (DESK). No env var or setting overrides it; retargeting needs a one-line
  source patch (**INFERRED** from the code read). Behaviour worth matching:
  it accepts `200` or `202` on async submit, polls every 3 s by default,
  treats any status other than `pending`/`processing`/`completed` as failure
  and reads `error.message`, downloads `result.video_url` with no auth
  header, and on sync endpoints branches on the response `Content-Type`
  (DESK `ltx_api_client_impl.py`).
- ComfyUI's hosted API nodes and third-party resellers (fal, Replicate,
  Magic Hour) wrap LTX behind their own APIs; they are not LTX-API clients
  and were not examined.

---

## 5. Gaps: API features our engine lacks today

Ordered roughly by how often a client will hit them (**INFERRED** ordering).

1. **Async job service**: job store, `202` submit, status polling with the
   exact `oneOf` shapes, 24 h retention, `404 not_found_error` after expiry,
   output hosting with expiring download URLs. Pure server work; no engine
   change.
2. **Media ingestion**: HTTPS fetch (no redirects, size/timeouts), data URIs,
   `/v1/upload` signed-URL flow and `ltx://` resolution; image decode for
   PNG/JPEG/WEBP; video decode (H.264/H.265 in MP4/MOV/MKV); audio decode
   (AAC-LC/MP3/Vorbis/Opus/FLAC). The engine only reads a local PNG/JPEG
   (`--image`).
3. **Exact API canvases**: 1080/2160/720 heights need pad-to-64 + crop;
   6/8/10 s durations at 4K and 1440p are untested (only 4K 5 s and 1080p
   20 s are measured workloads); 25/48/50 fps never measured; portrait never
   validated.
4. **I2V on 2.5**: the `ltx2_distilled_25` preset is t2av only; I2V is
   validated on 2.3. Per-token timesteps for I2V are deferred.
5. **`last_frame_uri`** (first+last frame interpolation): keyframes are out of
   scope.
6. **`duration: null`** (automatic duration): needs the 2.5 duration head,
   out of scope. Could be rejected with `400` or mapped to a fixed length.
7. **`camera_motion`**: no local mechanism. The API does not say how it is
   implemented (LoRA, prompt, or conditioning); **UNDOCUMENTED**.
8. **`generate_audio: false`**: trivial (drop the AAC track); the engine
   always generates audio.
9. **"pro" quality tier**: the dev (non-distilled) 2.5 DiT with CFG/STG is out
   of scope; only distilled 2.5 runs. 2.3-pro may map to `ltx2_base_23`.
10. **Audio-to-video**: no A2V pipeline. The Spark refiner has an audio
    encoder path (`encode_audio`, `docs/scope.md`), but conditioning the joint
    DiT on fixed input audio is not implemented.
11. **Retake** (time-window regeneration, audio/video/both) and **extend**
    (start/end, context frames, 505-frame cap): need video encode of the
    input plus temporal masking/conditioning; not implemented.
12. **HDR** (`exr_frames_url`, EXR ZIP): needs the HDR IC-LoRA path and an EXR
    writer; not implemented.
13. **Reframe** (outpainting to 1:1, 4:5, 5:4, 9:16, 16:9): needs the
    outpainting IC-LoRA; not implemented.
14. **Safety filter** (`422 content_filtered_error`), billing
    (`402 insufficient_funds_error`), per-org queue limits
    (`429 rate_limit_error`) and concurrency limits with `Retry-After`: server
    policy, not engine work.
15. **Server-side prompt enhancement** (implied by CL-2025-10-29): the local
    prompt enhancer is out of scope for 2.5.

What we have that the API does not expose: `seed`, 512p canvases, arbitrary
`8k+1` lengths, Sol/dense/PISA stage-2 routes, DiffVAE. Exposing them would
need extension fields; LTX clients will never send them.

### Open questions (UNDOCUMENTED upstream)

- Output container details: audio codec, sample rate, channels, video bitrate.
- Output URL lifetime.
- Queue-limit size; whether `Retry-After` is sent on `rate_limit_error`.
- Behaviour on unknown JSON fields (reject vs ignore).
- How mismatched input image aspect is fitted to `resolution`.
- Exact frame count produced for a given `duration`/`fps`.
