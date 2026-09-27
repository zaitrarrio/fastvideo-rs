# Research: MiniMax H3 API and FastVideo API wire contracts

Status: research notes. They record the external contracts before any server
code exists. Gathered 2026-09-27 against fastvideo-rs `6ed09d1`.

Conventions:

- Every fact cites a URL or a file path. **INFERRED** marks a conclusion that
  no source states directly.
- `MM:` is short for `https://platform.minimax.io/docs/`. Each page is also
  served as raw Markdown at the same path plus `.md` (for example
  `MM:api-reference/video-generation-v2-create.md`). The index is
  `MM:llms.txt`, and the OpenAPI source for the V2 video API is
  `MM:api-reference/video/generation/api/v2-video-generation.json`.
- `FV@e90be59:` is short for `hao-ai-lab/FastVideo` at
  `e90be598e56138af5c82590f0d72f6fa2dfce400`.

---

## 0. Headline findings

1. **MiniMax H3 is served only by the V2 video API, not by
   `/v1/video_generation`.** The model ids are `MiniMax-H3` and
   `MiniMax-H3-Max`. The V2 API uses a different shape:
   - `POST /v2/video_generation` takes a multimodal `content[]` array plus
     `resolution`, `duration` and `ratio`.
   - Status is polled at `GET /v2/query/video_generation/{task_id}`, which
     returns `content.url` directly. There is no `file_id` exchange.
   - The status values are lowercase: `queued`, `running`, `succeeded`,
     `failed` and `cancelled`.
   - Errors use an OpenAI-style envelope instead of `base_resp`.

   The V1 API that the brief describes (`prompt`, `first_frame_image`,
   `subject_reference`, `Preparing/Queueing/Processing/Success/Fail`,
   `/v1/files/retrieve`, `base_resp`) still exists, but only for the Hailuo
   model family (`MiniMax-Hailuo-2.3`, `MiniMax-Hailuo-02`, `T2V-01`, `I2V-01*`,
   `S2V-01`). Its model enums do not list H3. Sections 1.2 to 1.7 cover V2 and
   section 1.8 covers V1 for completeness. Sources: `MM:guides/video-generation`,
   `MM:api-reference/video-generation-v2-create`,
   `MM:api-reference/video-generation-t2v`.
2. **FastVideo ships an HTTP server: `fastvideo serve`.** It is an
   OpenAI/vLLM-Omni-compatible `/v1/videos` job API built on FastAPI.
   - Source: `fastvideo/entrypoints/openai/` at the commit this repo pins
     (`scripts/gpu/upstream/setup.sh:24`, `FV_REV=e90be598…`).
   - That pin was also `main` HEAD when fetched on 2026-09-27, so pinned and
     latest are the same code.
   - It has H3-specific handling: `task` set to `t2va`, `fl2va` or `ref2va`,
     typed image, video and audio references, 24 fps only, frame counts on
     the `17n+5` grid, and a 768×1344 pixel cap.

   Section 2.1 covers it.
3. **The "FastWan Video API" is a different, non-`/v1` contract.** The
   user-supplied client `streaming-client/fastwan_link.py` calls
   `POST /generate`, `GET /status/{id}`, `GET /video/{id}`,
   `DELETE /video/{id}`, `GET /health` and `GET /`, with a `prompt_id` job key.
   - None of these routes is in FastVideo's OpenAI server. The FastVideo
     contract doc lists its routes in
     `FV@e90be59:docs/design/server_contracts/openai.md`.
   - I did not find the server that implements them (see 2.2). The output is
     silent video. H3 output carries a stereo audio track.
4. OpenAI's Sora `/v1/videos` is the upstream of FastVideo's shape. Section 2.3
   summarizes it so that a FastVideo-compatible server also works with
   OpenAI SDK clients.

---

## 1. MiniMax H3 API (MiniMax Open Platform, "Video Generation V2")

### 1.1 Hosts, auth and product naming

| Item | Value | Source |
| --- | --- | --- |
| Global base URL | `https://api.minimax.io` | `servers:` in `MM:api-reference/video-generation-v2-create.md` |
| China base URL | `https://api.minimax.cn`. `platform.minimaxi.com/docs/*` redirects to `platform.minimax.cn`, and the same V2 page lists `servers: - url: https://api.minimax.cn` | `https://platform.minimaxi.com/docs/api-reference/video-generation-v2-create.md` (fetched with redirects) |
| Auth | `Authorization: Bearer <API key>`. The security scheme is `bearerAuth` (`type: http, scheme: bearer`). Keys come from Account Management > API Keys | `securitySchemes` in the V2 OpenAPI |
| Request content type | `Content-Type: application/json` (a required header parameter) | V2 create `parameters` |
| Product names | "MiniMax H3" is `MiniMax-H3`. "MiniMax H3 Max" is `MiniMax-H3-Max`, which was post-trained by fal.ai on H3 for faster generation. The OpenAPI `info.description` still reads "MiniMax video generation V2 (Hailuo-03) API" | `MM:guides/video-generation`, V2 OpenAPI `info` |
| Plan | H3 and H3 Max need the Pay-as-you-go API plan | `MM:guides/video-generation` |

### 1.2 Endpoints (V2)

| Method and path | operationId | Purpose |
| --- | --- | --- |
| `POST /v2/video_generation` | `videoGenerationV2Create` | Create a generation task. Returns `{task_id}` |
| `GET /v2/query/video_generation/{task_id}` | `videoGenerationV2Query` | Query one task. Only tasks from the last 7 days can be queried |
| `GET /v2/query/video_generation` | `videoGenerationV2List` | List tasks from the last 7 days, paginated and filterable |
| `DELETE /v2/video_generation/{task_id}` | `videoGenerationV2Delete` | Cancel a `queued` task, or delete a `succeeded` or `failed` record |
| `POST /v2/h3_context_ir` | `h3ContextIRV2Create` | Prompt enhancement only. Outputs text, not video |
| `POST /v2/video_regeneration` | `videoRegenerationV2Create` | Upscale a 768P H3 output to 2K |

Sources: the `paths` of `MM:api-reference/video/generation/api/v2-video-generation.json`
and the per-endpoint pages `MM:api-reference/video-generation-v2-{create,query,list,delete,h3-context-ir,regeneration}`.

### 1.3 `POST /v2/video_generation` request (`VideoGenerationV2Req`)

The schema requires `model`, `content`, `resolution` and `duration`. Its full
property list is `model`, `content`, `resolution`, `duration`, `ratio`,
`extra` and `callback_url`. There is **no** `seed`, `prompt_optimizer`,
`negative_prompt`, `aigc_watermark`, `fps` or `num_frames`. Source: the V2
OpenAPI schema; `aigc_watermark` appears only on the regeneration request.

| Field | Type | Values and rules |
| --- | --- | --- |
| `model` | string enum | `MiniMax-H3` (T2V, I2V first/last frame, R2V; `768P`/`2K`; 4–15 s) or `MiniMax-H3-Max` (same modes; `480P`/`768P`, no 2K; 5–15 s) |
| `content` | array of `ContentItem` | Exactly one non-empty `text` item is required, otherwise HTTP 400 `(2013)`. See 1.3.1 |
| `resolution` | string enum | `480P`, `768P`, `2K`. H3: `768P`, `2K`. H3-Max: `480P`, `768P` (the description says it defaults to `768P`, but the schema still lists the field as required) |
| `duration` | integer enum 4–15 | H3: 4–15. H3-Max: 5–15 (4 is rejected) |
| `ratio` | string enum | `adaptive` (default), `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`. **t2va** (text only): required and must not be `adaptive`. **i2va** (`first_frame`/`last_frame`): always `adaptive`; another valid value is ignored without error. **r2va**: optional, defaults to `adaptive`. The actual ratio comes back in the query's `ratio` field |
| `extra` | object, `additionalProperties: false` | H3-Max only. Its only key is `prompt_expansion_mode`: `disabled`, `balanced` (default) or `quality`. Do not send `balance`, `""` or booleans |
| `callback_url` | string | See 1.6 |

#### 1.3.1 `content[]` items (`ContentItem`)

| Field | Values |
| --- | --- |
| `type` (required) | `text`, `image_url`, `video_url`, `audio_url` |
| `text` | The prompt, at most 7000 characters per `text` |
| `image_url.url` / `video_url.url` / `audio_url.url` | A public URL, `mm_file://{file_id}` (a platform file: an upload or an earlier output), or a data URI (`data:image/<fmt>;base64,…`, `data:video/mp4;base64,…`, `data:audio/<fmt>;base64,…`, with a lowercase format) |
| `role` | `first_frame`, `last_frame`, `reference_image`, `reference_video`, `reference_audio` |

Mode selection by content. Source: the `content` description in the V2 schema.

| Mode (MiniMax name) | Content |
| --- | --- |
| Text-to-video (t2va) | One `text` only |
| I2V, first frame | `text` + 1 `image_url` with role `first_frame`. The role may be omitted: a single image with no role defaults to `first_frame` |
| I2V, last frame | `text` + 1 `image_url` with role `last_frame` |
| I2V, first and last frame | `text` + 2 `image_url` items, `first_frame` and `last_frame` |
| Reference (r2va) | `text` + any mix of `reference_image` (at most 9), `reference_video` (at most 3) and `reference_audio` (at most 3) |

The first/last-frame roles and the reference roles cannot be mixed in one request.

Media limits. Sources: the V2 schema and `MM:guides/video-generation`.

- **Request body:** at most 64 MB.
- **Images:** JPG, JPEG, PNG, WEBP, HEIC or HEIF, at most 30 MB each.
  Each side 256–5760 px; aspect ratio (w/h) 0.4–2.5.
- **Videos:** MP4 or MOV; H.264 or H.265 video with AAC or MP3 audio.
  At most 50 MB each; each clip 2–15 s and at most 15 s in total.
  Frame rate 23.976–60; same size and aspect limits as images.
- **Audio:** WAV or MP3, at most 15 MB each; each clip 2–15 s and at most
  15 s in total.
- **Mixed references:** at most 12 files in total.

Examples, verbatim from the OpenAPI (URLs abridged):

```json
{"model":"MiniMax-H3","content":[{"type":"text","text":"Epic space-opera theatrical teaser: …"}],
 "resolution":"2K","duration":5,"ratio":"16:9"}
```
```json
{"model":"MiniMax-H3","content":[{"type":"text","text":"Pull focus to the people in the background…"},
 {"type":"image_url","image_url":{"url":"https://cdn.hailuoai.com/…png"},"role":"first_frame"}],
 "resolution":"2K","duration":5,"ratio":"adaptive"}
```
```json
{"model":"MiniMax-H3","content":[{"type":"text","text":"Character speaks: … Voice timbre follows reference audio 1."},
 {"type":"video_url","video_url":{"url":"https://…mp4"},"role":"reference_video"},
 {"type":"audio_url","audio_url":{"url":"https://…mp3"},"role":"reference_audio"}],
 "resolution":"2K","duration":5,"ratio":"adaptive"}
```

Response 200 (`VideoGenerationV2Resp`): `{"task_id": "424010985738629"}`. The
schema has only this field. The V2 create response has no `base_resp`.

### 1.4 Task object (query, list and callback body)

`GET /v2/query/video_generation/{task_id}` returns `{"task": VideoTask}`.
Source: `MM:api-reference/video-generation-v2-query`.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Task id |
| `model` | string | For example `MiniMax-H3` |
| `status` | enum | `queued`, `running`, `succeeded`, `failed`, `cancelled` |
| `error` | `{code: string, message: string}` | Only when the task failed, for example `{"code":"1026","message":"video description contains sensitive content"}` |
| `created_at`, `updated_at` | int | Unix seconds |
| `content` | `{url?, prompt?}` | `url` is a time-limited video download URL; query again for a fresh one. `prompt` is only for `task_type=h3_context_ir` |
| `resolution` | string | `2K` and so on |
| `duration` | int | Seconds |
| `usage` | object | Only on success. Video tasks: `total_seconds`, `input_seconds` (reference video), `output_seconds`, `input_image_count`, `input_audio_seconds` (omitted when there is no audio reference), `total_tokens`, `prompt_tokens`, `completion_tokens`. Context-IR tasks: the token fields only. On failure the example shows `usage: {}` |
| `ratio` | string | Can be `""` when it does not apply, for example on regeneration |
| `task_type` | enum | `generation`, `h3_context_ir`, `regeneration` |
| `modality` | enum | `video` or `text`. The schema's own example omits it, so treat it as optional |

Example of a succeeded query, verbatim from the OpenAPI:

```json
{"task":{"id":"424010985738629","model":"MiniMax-H3","status":"succeeded",
 "created_at":1785125529,"updated_at":1785125946,
 "content":{"url":"https://…/output.mp4"},"resolution":"2K","duration":5,
 "usage":{"total_seconds":5,"input_seconds":0,"output_seconds":5,"input_image_count":1,
          "input_audio_seconds":6,"total_tokens":273890,"prompt_tokens":13500,"completion_tokens":260390},
 "ratio":"16:9","task_type":"generation","modality":"video"}}
```

- **Polling:** MiniMax recommends polling about every 10 s
  (`MM:guides/video-generation`, code sample).
- **File retrieval:** V2 has no separate file-retrieval step. The client
  downloads `task.content.url` directly.

**List** (`GET /v2/query/video_generation`):

- Query parameters: `page_num` (starts at 1), `page_size`, `filter.status`,
  `filter.task_ids` (an array; may repeat), `filter.model` and
  `filter.task_type`.
- Response: `{"items":[VideoTask…],"total":int}`. `total` counts only the
  last 7 days.
- Source: `MM:api-reference/video-generation-v2-list`.

**Delete** (`DELETE /v2/video_generation/{task_id}`):

- `queued`: the task is cancelled (`action=cancelled`).
- `succeeded` or `failed`: the record is deleted (`action=deleted`).
- `running` or `cancelled`: an error is returned.
- Response: `{"task_id","action","status"}`, where `action` and `status` are
  each `cancelled` or `deleted`.
- Source: `MM:api-reference/video-generation-v2-delete`.

### 1.5 Errors (V2)

On error, the HTTP status is the real status and the body is an `OaiError`:

```json
{"type":"error","error":{"type":"bad_request_error",
 "message":"invalid params, content must include a non-empty text item (prompt is required) (2013)",
 "http_code":"400"},"request_id":"021785229015510a2c883cf675b9804d"}
```

| HTTP | `error.type` | Example message with its internal code in parentheses |
| --- | --- | --- |
| 400 | `bad_request_error` | `invalid params … (2013)` |
| 401 | `authorized_error` | `login fail: Please carry the API secret key in the 'Authorization' field of the request header (1004)` |
| 402 | `insufficient_balance_error` | `insufficient balance (1008)` |
| 422 | `unprocessable_entity_error` | `video description contains sensitive content (1026)` |
| 429 | `rate_limit_error` | `rate limit, please retry later (1002)` |
| 500 | `server_error` | `internal error (1000)` |
| 529 | `overloaded_error` | Listed in the `OaiErrorDetail.type` description only |

Source: `components.responses.Err*` and `OaiErrorDetail` in the V2 OpenAPI.
The codes inside the parentheses come from the platform-wide table in
`MM:api-reference/errorcode`, which lists 1000 unknown, 1001 timeout,
1002 rate limit, 1004 auth, 1008 balance, 1024 internal, 1026 and 1027
input/output sensitive, 1033 system, 1039 token limit, 1041 conn limit,
2013 invalid params, 2045 rate growth, 2049 invalid key, 2056 usage limit,
and others.

### 1.6 Callback protocol (V2)

The `callback_url` description, identical on create, Context-IR and
regeneration, says:

1. MiniMax first sends a **verification request containing a `challenge`
   field**. The receiver must **return the `challenge` unchanged within
   3 seconds**.
2. After verification, MiniMax POSTs to the URL **whenever the task status
   changes**. The body has the same structure as the Query Task response,
   that is `{"task": VideoTask}` (**INFERRED**: the docs say "same structure"
   but show no callback example).
3. The callback `status` values are `queued`, `running`, `succeeded`,
   `failed` and `cancelled`.

The docs do not say:

- the HTTP method of the verification request (V1 says `POST`; see 1.8),
- whether the echo body is `{"challenge": …}` (it is in V1's sample),
- the retry policy, or any signature header.

Treat those as **unspecified**. For our server as a callback *sender*,
**INFERRED**: mirror the V1 sample. POST `{"challenge": "<random>"}`, expect
`{"challenge": "<same>"}` within 3 s, then POST task bodies.

### 1.7 Rate limits and pricing

- **Rate limits:** "Video Generation V2 / MiniMax-H3" allows **300 RPM** and
  **30 in-flight tasks**. "Video Generation / Hailuo series" allows 20 RPM
  (`MM:guides/rate-limits`). H3-Max has no row there. **INFERRED**: it shares
  the V2 row.
- **Pricing** (`MM:guides/pricing-paygo`):
  - H3: 768P $0.08/s; 2K $0.13/s.
  - H3-Max: 480P $0.05/s; 768P $0.08/s.
  - Input extras: audio is free for both models. H3 images: first 5 free,
    then $0.04 each. H3-Max images: first 2 free, then $0.074 each.
    Reference video is billed by input seconds.
  - Regeneration (768P to 2K): $0.05/s.
  - Context-IR: $0.90 per million input tokens and $3.60 per million output
    tokens.

  Pricing only matters to us if we emit `usage`.

### 1.6b H3-Context-IR and regeneration (for completeness)

**`POST /v2/h3_context_ir`**

- Body: `model` (only `MiniMax-H3`), `content` (same rules as generation),
  `duration` (4–15), `ratio` and `callback_url`.
- The task is async. On success the query returns `task_type=h3_context_ir`,
  `modality=text` and `content.prompt`, which is a structured enhanced
  prompt with `[Shot N]` blocks, `overall_soundscape:` and
  `non_diegetic_music:`.
- Source: `MM:api-reference/video-generation-v2-h3-context-ir`.

**`POST /v2/video_regeneration`** accepts either of two bodies:

- `{model:"MiniMax-H3", source_task_id, resolution:"2K", callback_url?, aigc_watermark?=false}`.
  This form requires whitelist access.
- `{model, content:[…original inputs…, exactly one {type:"video_url", role:"base_video"}], resolution:"2K", callback_url?, aigc_watermark?}`.

The `base_video` must match the H3 768P output specification:

- an audio track is present,
- 24 fps,
- width and height are multiples of 32,
- area between 768×768 and 768×1344,
- **107–362 frames in steps of 17**.

Source: `MM:api-reference/video-generation-v2-regeneration`.

That frame grid is the `17n+5` grid our engine uses. However, **107 frames
(4 s) is below our 124-frame minimum** (see §3.4).

**Self-hosting notes:**

- MiniMax's self-host guide says Context-IR and 2K regeneration are "Not
  included in H3-Base". They need MiniMax Platform components
  (`MM:guides/local-deploy-h3`, capability matrix).
- The same guide documents the **self-hosted** H3 API as SGLang's (and
  vLLM-Omni's) OpenAI-style `/v1/videos`, not MiniMax's own V2 API.
- Its T2VA example body has SGLang-specific fields that FastVideo's schema
  does not have (FastVideo forbids unknown fields):
  - `"task":"t2va"`, `"conditions":[]`,
  - `"target":{"short_edge":768,"aspect_ratio":"16:9","duration_seconds":5.0}`,
  - `"num_inference_steps":50`, `"flow_shift":12.0`, `"audio_flow_shift":3.0`,
    `"seed":1101`.
- FL2VA images are sent as `conditions` with `role:"keyframe"`.
- Expected output: H.264 video at 24 fps plus AAC stereo audio at 32 kHz;
  16:9 at a 768 short edge resolves to 1344×768.

### 1.8 Legacy V1 API (Hailuo models; the shape named in the brief)

The four V1 variant pages (t2v, i2v, fl2v and s2v) and the query page were
checked for H3. None lists or mentions it (grep for "H3" found no matches).
Sources: `MM:api-reference/video/generation/api/{text-to-video,image-to-video,start-end-to-video,subject-reference-to-video,openapi}.json`
and `MM:api-reference/file/management/api/openapi.json`.

**`POST https://api.minimax.io/v1/video_generation`** (`Content-Type: application/json`)

The same endpoint serves four body variants:

| Variant | `model` enum | Required | Other fields |
| --- | --- | --- | --- |
| T2V | `MiniMax-Hailuo-2.3`, `MiniMax-Hailuo-02`, `T2V-01-Director`, `T2V-01` | `model`, `prompt` | `prompt` (at most 2000 characters; `[Truck left]`-style camera commands on 2.3, 02 and Director), `prompt_optimizer` (bool, default true), `fast_pretreatment` (bool, default false; 2.3 and 02 only), `duration` (int, default 6; 6 or 10 depending on model and resolution), `resolution` (`720P`, `768P`, `1080P`), `callback_url` |
| I2V | `MiniMax-Hailuo-2.3`, `MiniMax-Hailuo-2.3-Fast`, `MiniMax-Hailuo-02`, `I2V-01-Director`, `I2V-01-live`, `I2V-01` | `model`, `first_frame_image` | `first_frame_image`: a public URL or `data:image/jpeg;base64,…`; JPG, JPEG, PNG or WebP under 20 MB; short edge over 300 px; aspect 2:5–5:2. Otherwise as T2V, with `512P` also allowed |
| First and last frame | `MiniMax-Hailuo-02` | `model`, `last_frame_image` | `first_frame_image`, `last_frame_image`, `prompt`, `prompt_optimizer`, `duration`, `resolution` (`768P` or `1080P`; no 512P), `callback_url` |
| Subject reference | `S2V-01` | `model`, `subject_reference` | `subject_reference: [{type:"character", image:[<one url or data URL>]}]`, `prompt`, `prompt_optimizer`, `callback_url` |

- `aigc_watermark` is **not** in any V1 video schema fetched; in the V2 API
  it appears only on regeneration.
- Response: `{"task_id":"106916112212032","base_resp":{"status_code":0,"status_msg":"success"}}`.
- `base_resp.status_code` values: 0 ok, 1002 rate limit, 1004 auth,
  1008 balance, 1026 sensitive prompt, 2013 invalid params, 2049 invalid key.

**`GET /v1/query/video_generation?task_id=`**

- Response: `{task_id, status, file_id?, video_width?, video_height?, base_resp}`.
- `status` is one of `Preparing`, `Queueing`, `Processing`, `Success`, `Fail`.
- Example: `{"task_id":"176843862716480","status":"Success","file_id":"176844028768320","video_width":1920,"video_height":1080,"base_resp":{"status_code":0,"status_msg":"success"}}`.
- Query `base_resp` codes: 0, 1002, 1004, 1026, and 1027 (output sensitive).
- Source: `MM:api-reference/video-generation-query`.

**`GET /v1/files/retrieve?file_id=`** (`file_id` is an int64)

- Response: `{"file":{"file_id","bytes","created_at","filename","purpose","download_url"},"base_resp":{…}}`.
- `download_url` is "valid for 1 hour". The example has `filename:"output_aigc.mp4"` and `purpose:"video_generation"`.
- Retrieve codes: 1000, 1001, 1002, 1004, 1008, 1013, 1026, 1027, 1039, 2013.
- Source: `MM:api-reference/video-generation-download`.

Other file routes: `/v1/files/upload`, `/list`, `/retrieve_content` and `/delete`.

**V1 callback** (from the `callback_url` description and its sample FastAPI
receiver):

- A `POST` arrives with `{"challenge": …}`, and the receiver answers
  `{"challenge": <same>}` within 3 s.
- Status pushes then look like
  `{"task_id","status":"success","file_id","base_resp":{…}}`.
- The callback `status` values are **lowercase `processing`, `success` and
  `failed`**. They differ from the query's capitalized values.

**V1 rate limit:** Hailuo series, 20 RPM (`MM:guides/rate-limits`).

---

## 2. FastVideo API ("fastvideo-api")

### 2.1 FastVideo's own server: `fastvideo serve` (OpenAI/vLLM-Omni `/v1/videos`)

**Version:**

- This repo pins `FV_REV=e90be598e56138af5c82590f0d72f6fa2dfce400`
  ("FastVideo main, 2026-09-24"; `scripts/gpu/upstream/setup.sh:23-24`).
- A fetch of `origin/main` on 2026-09-27 returned the same commit
  (`e90be59 [perf] MiniMax H3: return uint8 frames from the decode worker (#1828)`),
  so pinned equals latest.

**Launch:**

- Command: `fastvideo serve --config <yaml> [--dotted.override V]`
  (`FV@e90be59:fastvideo/entrypoints/cli/serve.py`).
- The YAML has `generator:`, `server:` and `default_request:` blocks.
  `server` holds `host` (0.0.0.0), `port` (8000), `output_dir` (`outputs/`)
  and `served_model_name` (`fastvideo/api/schema.py:9-13`).
- A `streaming:` block switches to the WebSocket server instead (`serve.py:30-36`).
- The H3 example config is `examples/serving/openai_fasth3.yaml`:
  - `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2`, served as `fasth3`;
  - defaults 768×1344, 124 frames, 24 fps, 5 steps, guidance 1.0, seed 1000.

**Implementation:** FastAPI (`openai/api_server.py`).

- **Auth:** there is no authentication. CORS allows `*`. A grep found no
  `Authorization`/`api_key` in `entrypoints/openai/`, and `openai.md` says
  the playground "does not add authentication".
- **Engine concurrency:** a single `asyncio.Lock`, so the pipeline runs
  exactly one generation at a time (`openai/serving_engine.py`). A running
  CUDA call cannot be interrupted.
- **Job store:** in memory (`openai/stores.py`). Jobs are lost on restart.

#### Endpoints (`FV@e90be59:docs/design/server_contracts/openai.md`, `openai/video_api.py`, `openai/common_api.py`)

| Method and path | Behavior |
| --- | --- |
| `POST /v1/videos` | Create an async job. Returns `VideoResponse` with `status:"queued"`. `POST /v1/videos/generations` is a hidden alias |
| `POST /v1/videos/sync` | Generate and return `video/mp4` bytes. Headers: `X-Request-Id`, `X-Model`, `X-Inference-Time-S`, `X-Stage-Durations` (compact JSON), `X-Peak-Memory-MB`. The temporary MP4 is removed after sending |
| `GET /v1/videos?after=&limit=(1..100)&order=asc\|desc` | `{object:"list", data, first_id, last_id, has_more}`. Default order is `desc` by `created_at` |
| `GET /v1/videos/{id}` | `VideoResponse`. A failed job returns **HTTP 200** with `status:"failed"` |
| `GET /v1/videos/{id}/content?variant=video` | `FileResponse` `video/mp4`. Any other variant returns 400. A failed job returns 422. Not yet complete, or no file, returns 404 `"Generation is still in-progress"` |
| `DELETE /v1/videos/{id}` | `{id, deleted:true, object:"video.deleted"}`. The resource is removed at once. A running generation is left to finish, and its artifact is then deleted |
| `GET /v1/models`, `GET /v1/models/{model}` | Model cards: `{id, object:"model", created, owned_by:"fastvideo", root}` |
| `GET /v1/model_info` | `{model_path, served_model_name, lora}` |
| `GET /health` | `{"status":"ok"}`, or 503 when the engine or its workers are unhealthy |
| `POST /v1/images`, `/v1/images/generations`, `/v1/images/edits`, `GET /v1/images/{id}/content` | Image routes; not mounted on MLX |
| `GET /playground/`, `/playground/config` | Browser client |

Remix, extensions and characters are not implemented (`openai.md`).

#### Request: `VideoGenerationRequest` (`openai/protocol.py:112-201`)

- **Encoding:** `extra="forbid"`, so unknown top-level fields return 400.
- **Body formats:** JSON, `multipart/form-data` or form-urlencoded.
  - In a form, the fields `image_reference`, `video_reference`,
    `audio_reference`, `video_params`, `lora` and `extra_params` are JSON
    strings.
  - A multipart `input_reference` upload is saved. It becomes
    `video_reference` if its type is video, otherwise `input_reference`.
- **Merging:** `extra_body` and `extra_json` objects are merged into the top
  level (`video_api.py:_parse_video_request`).

| Field | Type and constraint | Notes |
| --- | --- | --- |
| `prompt` | str, required, non-blank | |
| `model` | str? | Must equal `served_model_name`, or the LoRA nickname when a LoRA is loaded. Otherwise 400 |
| `seconds` | int ≥ 1 or a string matching `^[1-9]\d*$` | Used as `num_frames = seconds × fps` when `num_frames` is absent |
| `size` | `^\d+x\d+$` | `WIDTHxHEIGHT`. Takes precedence over `width`/`height`, which take precedence over `video_params.*` |
| `width`, `height`, `fps`, `num_frames` | int ≥ 1 | `fps` defaults to 24 |
| `video_params` | `{width,height,num_frames,fps}` | vLLM-Omni block |
| `aspect_ratio` | `"W:H"` | For H3 this runs `resolve_canvas_size`: 768 short edge, capped at 768×1344, snapped to multiples of 32 |
| `short_edge` | int | Requires `aspect_ratio`. For H3 it must be 768 |
| `image_reference` | `{image_url}` \| `{file_id}` \| a list of these | `file_id` returns 400 (there is no Files store). `image_url` may be http(s), `data:image…`, or a local path on the server |
| `video_reference` | `{video_url}` \| `{file_id}` \| a list | Same rules |
| `audio_reference` | `{audio_url}` \| a list | |
| `input_reference`, `reference_url` | str | Legacy single image. At most one of the two, and not together with `image_reference` |
| `video_path`, `video_url` | str | SGLang legacy direct video |
| `task` | str | H3 only: `t2va`, `fl2va` or `ref2va` (see below) |
| `n`, `num_outputs_per_prompt` | 1–10 | Anything other than 1 returns 400: "exactly one video output per request" |
| `quality` | `auto`, `default`, `standard`, `hd` | Echoed only |
| `negative_prompt`, `num_inference_steps` (1–200), `guidance_scale` (0–20), `guidance_scale_2`, `boundary_ratio` (0–1), `flow_shift`, `true_cfg_scale`, `seed` (int64), `max_sequence_length`, `enable_teacache` | | Passed to sampling. Fields a model cannot accept fail at admission with 400 (`request_to_sampling_param`) |
| `generate_sound`, `sound_duration`, `start_time_seconds` | | `generate_sound` is ignored for H3, which always has audio |
| `enable_frame_interpolation`, `frame_interpolation_exp`, `_scale`, `_model_path` | | |
| `lora` | `{name\|lora_name\|adapter, path\|lora_path\|local_path, scale\|lora_scale}` | Must match the startup adapter, otherwise 400 |
| `extra_params` | dict | Only `ltx2_audio_latents`, `ltx2_audio_clean_latent`, `ltx2_audio_denoise_mask`, `audio_num_frames`, `video_position_offset_sec`, `vsa_mode`, `vsa_dense_first_n_steps`, `vsa_dense_layers` (`fastvideo/api/compat.py:47-56`) |
| `user` | str | Ignored |

Precedence: explicit request fields override operator `default_request`
fields, which override model preset defaults. The code uses
`model_fields_set`, so pydantic defaults do not count as client intent
(`request_adapter.py:build_generation_request`, `openai.md` "Defaults and
errors").

#### H3 rules in FastVideo's adapter (`openai/request_adapter.py`)

- **`task`** (`_apply_reference_inputs`):
  - `t2va` rejects any media.
  - `fl2va` needs 1–2 images: the first is `image_path`, and the second is
    `last_image`.
  - `ref2va` requires a server started with
    `override_pipeline_cls_name=MiniMaxH3Ref2VAModularPipeline`, and a
    ref2va server rejects the other tasks.
  - References keep the order images, then videos, then audio, whatever the
    order in the JSON.
- **Reference limits** (`fastvideo/pipelines/basic/minimax_h3/reference.py:30-33,77-101`):
  at most 9 images, 3 videos, 3 audio clips and 12 references in total.
  Audio-only reference sets are rejected.
- **Geometry and timing:**
  - `fps` must be 24.
  - Width and height must both be set, be multiples of 32, and have
    `w×h ≤ 768×1344`.
  - An explicit `num_frames` must already be on the `17n+5` grid, otherwise
    400 (with the next valid value in the message).
  - A frame count derived from `seconds` is silently aligned up, for example
    5 s gives 120, which becomes 124.
  - The range is 5–15 s (`packing.py:21-29`, `minimax_h3_input_preparation.py:94-103`).
- `openai.md` says FastH3 "requires guidance scale 1".

#### Response: `VideoResponse` (`openai/protocol.py:213-234`)

`{id:"video_gen_<32hex>", object:"video", model, prompt, status, progress, created_at, size:"WxH"|null, seconds:"<int>" (default "4"), quality, url:null, remixed_from_video_id:null, expires_at:null, file_path, file_name, media_type:"video/mp4", completed_at, error:{code,message}|null, peak_memory_mb, inference_time_s, stage_durations:{stage:seconds}}`.

- `status` is one of `queued`, `in_progress`, `completed`, `failed`.
- `progress` is only ever 0 or 100; there is no intermediate progress.
- A failed job has `error.code = "generation_failed"` (`video_api.py:_run_generation`).
- `file_path` is a server-local path, a FastVideo extension.

**Errors:** an OpenAI envelope, `{"error":{"message","type":"invalid_request_error"|"server_error","param":null,"code":<http int>}}`.
Validation errors return 400, not 422 (`api_server.py:124-151`).

#### Streaming

`fastvideo serve` with a `streaming:` block serves `WS /v1/stream` instead of
REST.

- It carries JSON control frames and binary fMP4 media chunks.
- Its messages include `session_init_v2`, `segment_prompt_source`,
  `media_init`, `media_segment_complete` and `step_complete`.
- Sources: `FV@e90be59:docs/design/server_contracts/streaming.md` and
  `fastvideo/entrypoints/streaming/protocol.py`.
- The REST `/v1/videos` API has no streaming and no SSE.

### 2.2 "FastWan Video API": the contract of the user's client

The only source is `/home/user/refsrc/d5fc0bdf-infinite-livestream-source/streaming-client/fastwan_link.py`,
with the README "Backends > FastWan on the FastWan Video API" section and
`config.py`. I did **not** read `.env.example` (see "Open items").

- **Base URL:** `FASTWAN_BASE_URL`, default `http://127.0.0.1:8000`. Routes sit
  at the root, with **no `/v1` prefix** (README "Backends"; `config.py:247-249`).
- **Auth:** when `FASTWAN_API_KEY` is set, every request carries
  `Authorization: Bearer <key>` (`fastwan_link.py:330`).

| Call | Request | Response the client relies on |
| --- | --- | --- |
| `GET /health` | | 200 JSON with a truthy `model_loaded`; otherwise the server counts as unreachable (`fastwan_link.py:359-361`) |
| `GET /` | | 200 JSON; the client reads `model` (the served model name) and compares it to `FASTWAN_MODEL` (`:362,370-381`) |
| `POST /generate` | JSON `{"prompt": str, "width": int, "height": int, "num_frames": int, "fps": int, "seed": int}` (`:458-465`) | 200 JSON with **`prompt_id`** (the job id) and `status`. The client checks `status` first, so the create reply can already be terminal (`:466-478`) |
| `GET /status/{prompt_id}` | | 200 JSON `{status, error?}`. `status` is one of `queued`, `processing`, `completed`, `failed`; any other value fails the clip (`:92, 471-478`) |
| `GET /video/{prompt_id}` | | 200 raw MP4 bytes; the client allows 300 s (`:480-483, 86`) |
| `DELETE /video/{prompt_id}` | | Best effort: deletes the job and its server-side MP4 (`:518-529`) |

- **Errors:** HTTP 400, 413, 415 or 422 means the request was rejected and
  the clip fails. Any other non-200 status, a connection error or a timeout
  means the server is unreachable: the clip stays queued, and the client
  re-probes `/health` every 5 s (`:88-89, 499-516`).
- **Error bodies:** the client parses FastAPI's `{"detail": str | [{msg…}]}`
  (`_error_detail`, `:632-648`). **INFERRED:** the server is a FastAPI app.
- **Semantics:** one prompt in, one MP4 out, with no server-side queue order,
  playout or media tracks. The server "generates one clip at a time per
  replica" (README; `fastwan_link.py:17-20`).
- **Client defaults** (`config.py:234-257`):
  - `FASTWAN_SIZE` 1280x704, even width and height required;
  - `FASTWAN_FPS` 24;
  - `FASTWAN_MIN_FRAMES` / `FASTWAN_MAX_FRAMES` 49 / 121, snapped to the Wan
    `4k+1` grid (2.04–5.04 s at 24 fps);
  - `FASTWAN_CONCURRENCY` 1.
- **Output:** a video-only MP4. `FastWanLink.generates_audio = False`, the
  client decodes with `-an`, and the pacer fills silence
  (`fastwan_link.py:142, 571`; README "FastWan has no audio"). H3 output, by
  contrast, has a stereo audio track: 32 kHz AAC in the MP4
  (`MM:guides/local-deploy-h3`; our `crates/fastvideo-models/src/h3/config.rs:334`
  sets `sampling_rate: 32000`).

**Server side: not located.**

- The routes are not in FastVideo's OpenAI server at `e90be59` (route table
  in `docs/design/server_contracts/openai.md`, confirmed by reading
  `video_api.py`, `common_api.py` and `api_server.py`).
- My attempt to grep the FastVideo checkout for `model_loaded` or
  `"/generate"` was blocked by the session's permission classifier, so it
  was not run.
- One unverified lead, **INFERRED** and not read: an earlier grep hit
  `examples/inference/gradio/serving/ray_serve_backend.py` in FastVideo,
  which exports `fastvideo_video_generation_seconds` Prometheus metrics. It
  might be, or neighbor, a Ray Serve backend with a `/generate` route.
  Someone should confirm by reading that file, or by asking the user where
  their FastWan Video API deployment comes from.

### 2.3 OpenAI Sora `/v1/videos` (reference shape; FastVideo's upstream)

Source: `openai/openai-openapi`, `master/openapi.yaml`, fetched 2026-09-27.

| Route | operationId |
| --- | --- |
| `POST /videos` (multipart or JSON) | `createVideo` |
| `GET /videos?after&limit(0..100)&order` | `ListVideos` |
| `GET /videos/{video_id}` | `GetVideo` |
| `DELETE /videos/{video_id}` | `DeleteVideo` |
| `GET /videos/{video_id}/content?variant=video\|thumbnail\|spritesheet` | `RetrieveVideoContent`; returns `video/mp4`, `image/webp` or JSON |
| `POST /videos/{video_id}/remix` | `CreateVideoRemix`, body `{prompt}` |
| `POST /videos/edits`, `/videos/extensions`, `/videos/characters`, `GET /videos/characters/{id}` | |

- **Create body:** `prompt` (required), `model` (`sora-2`, `sora-2-pro`, or a
  dated id; default `sora-2`), `seconds` (`"4"`, `"8"` or `"12"`; default 4),
  `size` (`720x1280`, `1280x720`, `1024x1792`, `1792x1024`; default
  720x1280) and `input_reference`. In multipart it is a file; in JSON it is
  `{image_url}` or `{file_id}`.
- **`VideoResource`:** `id`, `object:"video"`, `model`, `status`
  (`queued`, `in_progress`, `completed`, `failed`), `progress`, `created_at`,
  `completed_at`, `expires_at`, `prompt`, `size`, `seconds` (a string),
  `remixed_from_video_id`, and `error{code,message}`.
- **Delete:** returns `{id, object:"video.deleted", deleted}`.
- **List:** returns `{object:"list", data, first_id, last_id, has_more}`.

FastVideo's `VideoResponse` is a superset of this resource.

---

## 3. Our request surface and the mapping

### 3.1 Today's entry points

**`fastvideo generate`** (`crates/fastvideo-cli/src/main.rs:58-177`):

- `--model` (HF id), `--prompt`, `--negative`, `--seed`, `--steps`,
  `--frames`, `--height`, `--width`, `--guidance`, `--guidance-2` (audio
  guidance for AV models).
- `--image` (first frame for I2V or H3 FL2VA), `--last-image` (H3 FL2VA),
  `--ref` (repeatable, ordered; H3 Ref2VA; kind taken from the extension),
  `--control`.
- `--seconds` (H3, 5–15).
- LTX flags: `--two-stage`, `--refine-steps`, `--diff-vae`, `--sol-stage2`,
  `--dense-stage2`, `--pisa-stage2`, `--ltx23-hq`.
- `--h3-recipe`, `--h3-adapter`, `--dit-offload`, `--tae-weights`,
  `--save-mp4`, and a TOML `--config` overlay (`GenerateToml`,
  `main.rs:180-215`).
- There is no `fps` flag. The TOML has `fps` and `flow_shift` fields;
  **INFERRED** from the `AvGenerateOptions` construction at `main.rs:356-400`
  that they are not forwarded for the AV families.

**Core request structs:**

- Wan: `LoadOptions` / `VideoGenerator`
  (`crates/fastvideo-core/src/generator.rs:11-60`).
- LTX and H3: `AvGenerateOptions` (`crates/fastvideo-core/src/av_generate.rs:12-63`).
- H3 mapping in `generate_h3` (`av_generate.rs:285-360`):
  - FL2VA and Ref2VA are mutually exclusive.
  - Base MiniMax-H3 T2AV requires `--h3-recipe sol-h3-rtx`; FastH3 recipes
    handle T2AV (`h3_t2v_unwired`).
  - Geometry is checked by `H3Geometry::checked`.
  - The seed defaults to 1024 (`main.rs:361`).

**`H3Request`** (`crates/fastvideo-cudarc/src/h3/pipeline.rs:84-155`):

- Fields: `prompt`, `seed`, `height`, `width`, `num_frames` (aligned up to
  `17n+5`, 5–15 s at 24 fps), `mp4`, `first_image`, `last_image` and
  `references: Vec<H3ReferenceSpec{path, kind: Image|Video|Audio}>`
  (`crates/fastvideo-models/src/h3/reference.rs:34-57`).
- Helpers: `keyframe_anchors()` (First/Last), `is_ref2va()`, and
  constructors `seconds()` (16:9, 768×1344) and `sized()`.
- Pipeline-level options (`H3PipelineOptions`, `pipeline.rs:219-253`):
  `recipe`, `ref2va`, `adapter`, `reference_image_resize`, `dit_offload`,
  `text_encoder`, `taeh3` and `dense`. These are load-time settings, not
  per-request ones.
- Canvas helpers exist (`crates/fastvideo-models/src/h3/config.rs:740-800`):
  - `resolve_canvas_size(aw, ah)` is the same algorithm as FastVideo's
    `packing.py`: 768 short edge, capped at 768×1344, multiples of 32,
    aspect 1:4 to 4:1.
  - `snap_canvas` and `check_canvas` are also there.
- Reference limits are the same as FastVideo's: at most 9 images, 3 videos,
  3 audio clips and 12 in total (`reference.rs:29-32`).
- The Ref2VA pipeline encodes image, video and audio references
  (`pipeline.rs:1634-1650, 1806-1880`).
- Output: PNG frames, `audio.wav`, and `output.mp4` with H.264 and AAC when
  ffmpeg is present (`crates/fastvideo-cudarc/src/wan/pipeline.rs:2349-2400`,
  test `mp4_carries_the_audio_track`).

**`fv-gpucheck` generation commands:**

- **`h3 gen`** (`crates/fastvideo-gpucheck/src/h3_stage.rs:156-250`):
  `--weights`, `--prompt`, `--seconds` (default 5), `--num-frames`,
  `--height`/`--width`, `--seed` (1024), `--prompts` (a JSON set),
  `--h3-recipe`, `--dense`, `--no-mp4`, `--clip-dir`, `--text-encoder`,
  `--taeh3-weights`, `--dit-offload`, `--device-budget-gib`, `--warm`, and
  the cache flags. It has **no image or reference inputs**: it is T2VA only.
- **`ltx2 gen`** (`crates/fastvideo-gpucheck/src/ltx2_stage.rs:238-340, 415-443`):
  `--model-version` (2.0, 2.3 or 2.5), `--weights`, `--dit`, `--prompt`,
  `--seed` (10), `--workload` (`4k5s` or `1080p20s`) with
  `--height`/`--width`/`--num-frames`/`--frame-rate`, `--image` (I2V),
  `--two-stage`, `--diff-vae`, the Sol/PISA/dense stage-2 flags, `--offload`,
  `--dit-offload`, and the text flags.
- **`wan gen`** (`crates/fastvideo-gpucheck/src/wan_stage.rs:26-120`):
  `--weights`, `--preset`, `--prompt`, `--seed` (1024), `--negative`,
  `--height` (480), `--width` (832), `--num-frames` (81), `--steps` (3),
  `--unipc`, `--guidance` (1.0), `--flow-shift` (8.0), `--fps` (16),
  `--full-vae`, and `--image` (I2V for TI2V-5B and 2.1 I2V).
- The existing `fv-gpucheck serve` (`crates/fastvideo-gpucheck/src/serve.rs`)
  is a read-only static file server for run logs. It is not an inference API.

### 3.2 MiniMax V2 fields and our engine

| MiniMax V2 field | Maps to | Status |
| --- | --- | --- |
| `model: MiniMax-H3` | H3 pipeline with a recipe chosen at server start (`H3PipelineOptions.recipe`) | OK. Which recipe (base, FastH3 or Sol-H3) answers to `MiniMax-H3` is a server-config decision (**INFERRED**) |
| `model: MiniMax-H3-Max` | none | **Gap:** fal's post-trained H3-Max weights are not public in any source here. Alias to a fast FastH3 or Sol-H3 recipe, or return 400 |
| `content[text].text` (≤ 7000 characters) | `H3Request.prompt` | OK. Enforce the 7000-character limit at admission |
| `image_url` + role `first_frame`, or a single image with no role | `H3Request.first_image` (`KeyframeAnchor::First`) | OK once the server downloads or decodes the URL to a file |
| `image_url` + role `last_frame` | `H3Request.last_image` (`KeyframeAnchor::Last`) | OK; last-frame-only is allowed by both sides |
| `reference_image` / `reference_video` / `reference_audio` | `H3Request.references` with `ReferenceKind` from the role | OK. It needs the Ref2VA DiT (`transformer_ref/`, `H3PipelineOptions.ref2va`) loaded, possibly as a second pipeline. MiniMax keeps the `content` order; our pipeline keeps the given order. FastVideo reorders to images, videos, audio. **INFERRED:** keep MiniMax's order, since prompts say "reference video 1" |
| First/last frame mixed with references | 400 | We reject it too (`av_generate.rs:298-302`) |
| `url` forms: http(s), `data:` URI, `mm_file://{file_id}` | Local path | **Gap:** the engine takes paths only. The server must fetch, decode data URIs, and (for `mm_file://`) have a files store or reject with 400 |
| `resolution: 768P` + `ratio` | `resolve_canvas_size(ratio)` gives, for example, 768×1344 for 16:9 | OK for 16:9, 4:3, 1:1, 3:4 and 9:16. **Caution:** 21:9 at a 768 short edge exceeds the 768×1344 cap, and `resolve_canvas_size` scales it down to about 640×1504. MiniMax's actual 21:9 canvas is not documented, so **INFERRED** it is an equal-area rescale |
| `resolution: 2K` | none | **Gap:** 2K is MiniMax's regeneration stage (a separate H3 upscaler; "not included in H3-Base"). Our cap is 768×1344. Return 400, or generate at 768P and report it |
| `resolution: 480P` (H3-Max) | A smaller canvas such as 832×480 (the gpucheck "480p cell", `h3_stage.rs:168-170`) | Possible; the exact MiniMax 480P canvas per ratio is undocumented (**INFERRED**) |
| `ratio: adaptive` (i2va or r2va) | Derive from the first-frame or reference image aspect, then `resolve_canvas_size` | **Gap (server logic):** a small addition, since the snapping helper exists |
| `duration` 4–15 (integer) | `num_frames = align(duration × 24)` onto `17n+5` | **Gap for 4 s:** our geometry allows only 5–15 s, 124–362 frames (`H3Geometry`), while MiniMax's H3 grid starts at 107 frames. Either extend `H3Geometry` (needs checking against upstream: FastVideo's `MINIMAX_H3_MIN_DURATION=5.0`) or return 400 for 4 |
| `extra.prompt_expansion_mode`; H3-Context-IR | none | **Gap:** we have no prompt enhancer. Accept `disabled` and reject the others, or pass through unchanged |
| `callback_url` | Server job layer | **Gap (server):** challenge echo, then POST `{task}` on each status change |
| no seed field | `H3Request.seed` | The server picks a seed (random or fixed). **INFERRED:** expose it only in logs |
| `usage.*` | Server-computed from inputs and outputs | Server-only |
| Status `queued` → `running` → `succeeded`, `failed` or `cancelled` | Job queue in front of an engine that runs one job at a time | Server-only. `DELETE` must refuse while a task is `running`, which matches our non-interruptible CUDA run |
| 7-day retention; time-limited `content.url` | Server file store plus a signed URL or a `/files` route | Server-only |
| `/v2/video_regeneration` | none | **Gap:** no 2K regeneration model |

### 3.3 FastVideo `/v1/videos` fields and our engine

| FastVideo field | H3 (`H3Request` / `AvGenerateOptions`) | LTX-2.x (`AvGenerateOptions`) | Wan (`LoadOptions`) |
| --- | --- | --- | --- |
| `prompt` | `prompt` | `prompt` | `generate_video(prompt)` |
| `model` | Match against the server's served name | Same | Same |
| `size` / `width` / `height` / `video_params` / `aspect_ratio` + `short_edge` | `height`, `width` (checked); `resolve_canvas_size` for `aspect_ratio` | `height`, `width` | `height`, `width` |
| `fps` | Must be 24 (H3 is fixed) | **Gap:** no per-request fps in `AvGenerateOptions`; `ltx2 gen` has `--frame-rate` | **Gap:** `LoadOptions` has no fps; `wan gen --fps` (default 16) exists only in gpucheck |
| `num_frames` / `seconds` | `num_frames` must be on `17n+5`; `seconds × 24` aligned up | `num_frames` | `num_frames` (Wan `4k+1`; FastVideo does not snap it) |
| `seed` | `seed` | `seed` | `seed` |
| `negative_prompt` | Not used by H3 (CFG-distilled). **INFERRED:** accept and ignore, as FastVideo does not pass it for H3 | `negative_prompt` | `negative_prompt` |
| `guidance_scale` | FastH3 requires 1 | `guidance_scale` | `guidance_scale` |
| `guidance_scale_2` | none | `audio_guidance_scale` (our meaning; FastVideo's `guidance_scale_2` means the Wan 2.2 second expert) | `guidance_scale_2` |
| `num_inference_steps` | **Gap:** H3 steps are fixed by the recipe; `generate_h3` does not read `num_inference_steps` | `num_inference_steps` | `num_inference_steps` |
| `flow_shift` | none | none | **Gap:** CLI TOML only, not in `LoadOptions` |
| `task` = `t2va` / `fl2va` / `ref2va` | t2va: no images or refs. fl2va: `first_image` / `last_image` from `image_reference[0..2]`. ref2va: `references` from image, video and audio refs | 400 (FastVideo: "only defined for MiniMax-H3") | 400 |
| `image_reference` / `input_reference` | As above | `image_path` (LTX I2V, first frame) | `image_path` (I2V / TI2V) |
| `video_reference` / `audio_reference` | Ref2VA `references` (Video, Audio) | 400 | 400 |
| `n` / `num_outputs_per_prompt` ≠ 1 | 400 | 400 | 400 |
| `lora` | Must match the startup `h3_adapter`; per-request swap is unsafe | **Gap:** no LTX LoRA in our engine | None |
| `extra_params.vsa_*` | Recipe-level (`FASTVIDEO_VSA*` env, techniques profile); not per request | | |
| `generate_sound` | Always on | LTX always has audio | Wan is silent |
| `enable_frame_interpolation*`, `enable_teacache`, `true_cfg_scale`, `boundary_ratio`, `max_sequence_length`, `start_time_seconds`, `sound_duration` | **Gap:** not implemented; return 400 as FastVideo does when a model cannot take a field | | |
| Response `stage_durations`, `inference_time_s`, `peak_memory_mb` | Available from `H3Timings` / `H3Output.memory` (`pipeline.rs:335-380`) | Similar timing structs | `BenchStats` |

### 3.4 FastWan Video API and our Wan engine

| FastWan field | Our Wan surface | Notes |
| --- | --- | --- |
| `prompt` | `generate_video(prompt)` | |
| `width`, `height` (default 1280x704) | `LoadOptions.width` / `height` | FastWan 1280×704 is a Wan 2.2 TI2V-5B-style canvas (**INFERRED**); check that the chosen preset accepts it |
| `num_frames` (4k+1, 49–121) | `LoadOptions.num_frames` | |
| `fps` (24) | **Gap:** no fps in `LoadOptions`; `wan gen --fps` defaults to 16 | It affects only the MP4 container rate. Frame count is what matters to the client, which resamples to 24 fps with ffmpeg (`fastwan_link.py:571-573`) |
| `seed` | `LoadOptions.seed` | |
| Output | Silent H.264 MP4 | The client ignores audio (`-an`) |

### 3.5 Engine gaps, collected

1. **H3 durations below 5 s** (MiniMax `duration: 4` means 107 frames). Our
   `H3Geometry` minimum is 124 frames.
2. **2K output** (`resolution: 2K`, `/v2/video_regeneration`): no upscaler
   model.
3. **`MiniMax-H3-Max` weights:** not available. At best it is an alias to a
   FastH3 or Sol-H3 recipe.
4. **Prompt expansion** (`prompt_expansion_mode`, H3-Context-IR): none.
5. **Per-request H3 step count:** the recipe fixes it; `num_inference_steps`
   is ignored for H3 in `generate_h3`.
6. **Per-request fps** for Wan and LTX, and **`flow_shift`** for Wan, are not
   in the core request structs.
7. **One model per process today:** `VideoGenerator` and `generate_av` load
   per call. A server needs a resident, reusable pipeline handle. `H3Pipeline`
   supports this: `H3Pipeline::load` then `H3Pipeline::generate(&request, out_dir)`
   (`pipeline.rs:621, 1093`). It also needs FL2VA and Ref2VA DiTs side by
   side if one server offers both (as vLLM-Omni does, per
   `MM:guides/local-deploy-h3`).
8. **Media ingestion:** URL fetch, `data:` decode, `mm_file://` or `file_id`
   stores, and the MiniMax media limits (size, duration, aspect ratio 0.4–2.5,
   frame rate) all need to live in the server layer.
9. **Job layer:** queue, statuses, list/cancel/delete, 7-day retention,
   expiring download URLs, callbacks with a challenge, and `usage`
   accounting. The engine runs one generation at a time and cannot be
   interrupted, matching FastVideo's `asyncio.Lock` and MiniMax's refusal to
   delete a `running` task.
10. **LoRA per request:** unsafe with dense-merged adapters. Like FastVideo,
    accept only the startup adapter.

---

## Open items

- **FastWan Video API server:** not identified. The grep of the FastVideo
  checkout was blocked. Next steps: read
  `examples/inference/gradio/serving/ray_serve_backend.py` at `e90be59`, or
  ask the user for the deployment's source.
- **FastWan config file not read:** `streaming-client/.env.example` was not
  read, because the session's permission classifier blocked access. Only
  `fastwan_link.py`, `config.py` and the README were read.
- **MiniMax callback details:** the delivery method of the V2 verification
  request, the retry policy and any signature are undocumented.
- **MiniMax canvases:** the exact output canvas for `21:9` and for `480P` is
  undocumented.
