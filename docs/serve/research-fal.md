# fal.ai MiniMax H3 Max: wire-compatibility research

Status: research note, 2026-09-27. Scope: the exact contract a Rust server in
front of the fastvideo-rs CUDA engine has to speak so that unmodified fal
clients (`@fal-ai/client`, Python `fal_client`, plain HTTP) can use it in place
of fal's hosted `minimax/h3-max/*` endpoints. Nothing here is implemented yet.

## 0. Sources and conventions

Every fact carries a source. Anything I derived rather than read is marked
**INFERRED**. Anything fal does not document is marked **UNDOCUMENTED**, and I
did not fill those gaps with guesses.

| Tag | Source | Notes |
|---|---|---|
| **OA-T2V** | https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=minimax/h3-max/text-to-video | Queue OpenAPI 3.0.4, fetched 2026-09-27 |
| **OA-I2V** | https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=minimax/h3-max/image-to-video | same |
| **OA-R2V** | https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=minimax/h3-max/reference-to-video | same |
| **LLMS-T2V / -I2V / -R2V / -DIR** | https://fal.ai/models/minimax/h3-max/{text-to-video,image-to-video,reference-to-video,director}/llms.txt | fal's agent-readable model pages (schemas, pricing, examples). The link to them is on every model page |
| **ASYNC-DIR** | https://fal.ai/api/apps/fal-ai/minimax-h3-max-director/asyncapi.json | AsyncAPI 3.1.0 contract for the director's control channel |
| **API-DIR** | https://fal.ai/models/minimax/h3-max/director/api | The page embeds the director app's own FastAPI OpenAPI, with `/start-session`, `/info` and `/health`. The same document at `…/minimax-h3-max-director/openapi.json` returns 405/401 without a key |
| **PG-T2V / -I2V / -R2V** | https://fal.ai/models/minimax/h3-max/{text-to-video,image-to-video,reference-to-video} | Playground pages. Each shows an example result JSON and a sample MP4 |
| **D-QUEUE** | https://fal.ai/docs/documentation/model-apis/inference/queue | |
| **D-SYNC** | https://fal.ai/docs/documentation/model-apis/inference/synchronous | |
| **D-STREAM** | https://fal.ai/docs/documentation/model-apis/inference/streaming | |
| **D-RT** | https://fal.ai/docs/documentation/model-apis/inference/real-time | |
| **D-WS** | https://fal.ai/docs/documentation/model-apis/inference/websockets | |
| **D-HOOK** | https://fal.ai/docs/documentation/model-apis/inference/webhooks | |
| **D-REL** | https://fal.ai/docs/documentation/model-apis/inference/reliability | |
| **D-HDR** | https://fal.ai/docs/documentation/model-apis/common-parameters | |
| **D-ERR** | https://fal.ai/docs/documentation/model-apis/errors | |
| **D-REQERR** | https://fal.ai/docs/documentation/model-apis/request-errors | |
| **D-CONC** | https://fal.ai/docs/documentation/model-apis/concurrency-limits | |
| **D-CDN** | https://fal.ai/docs/documentation/model-apis/fal-cdn | |
| **D-PROXY** | https://fal.ai/docs/documentation/model-apis/inference/proxy-setup | |
| **D-WMA** | https://fal.ai/docs/documentation/development/wma | |
| **JS** | https://github.com/fal-ai/fal-js at `cf73f62385f5c89582f95371a3f4b51e312c5adc` (2026-09-24), `libs/client`, version `1.11.0-alpha.0` | npm dist-tags on 2026-09-27: `latest` 1.10.1, `alpha` 1.11.0-alpha.4 (https://registry.npmjs.org/@fal-ai/client) |
| **PY** | https://github.com/fal-ai/fal at `ec46b79562cce67e396c35fac4c3f62dcdf094da` (2026-09-22), `projects/fal_client` | PyPI `fal-client` latest 1.0.3 (https://pypi.org/pypi/fal-client/json) |
| **PROBE** | My own unauthenticated HTTP probes and MP4 box parsing, 2026-09-27 | Described where used |
| **FL** | `/home/user/refsrc/d5fc0bdf-infinite-livestream-source/streaming-client/fal_link.py` and its `README.md` "Backends" section | A working client supplied by the user (section 13) |

JS paths below are relative to `libs/client/src/`, and PY paths to
`projects/fal_client/src/fal_client/`. All docs pages were read through
https://fal.ai/docs/llms-full.txt, which inlines every page and marks each
one with its `Source:` URL.

---

## 1. Implementer's checklist

This is the short version. Each item is backed by a later section.

1. **Endpoint ids** (§2): `minimax/h3-max/text-to-video`, `minimax/h3-max/image-to-video`
   and `minimax/h3-max/reference-to-video` are queue/HTTP endpoints.
   `minimax/h3-max/director` is **not HTTP at all**. It is a WebRTC ("WMA")
   session (§8).
2. **Queue routes** (§9) must accept **both** path forms:
   - the full endpoint id: `/{owner}/{app}/{sub}/requests/{id}[/status|/cancel]`,
     which is what the published OpenAPI lists;
   - the app-only form: `/{owner}/{app}/requests/{id}[/status|/status/stream|/cancel]`.
     This is what `@fal-ai/client` always builds (it drops the sub-path), and
     what fal returns in `status_url` and friends (FL).
   The request id is unique, so routing can key on it alone.
3. **Status bodies** must always carry `queue_position` when `IN_QUEUE` and a
   `logs` key when `IN_PROGRESS` or `COMPLETED`. The Python client indexes
   `data["queue_position"]` and `data["logs"]` directly and raises `KeyError`
   otherwise (PY:client.py L711-726). Accept `?logs=1|0|true|false`, because
   Python sends an httpx bool (**INFERRED** from httpx serialization) and JS
   sends `"1"`/`"0"`.
4. **Result** `GET …/requests/{id}` returns the bare model output JSON (§3-§5),
   with an `x-fal-request-id` response header. JS reads it into
   `Result.requestId` (JS:response.ts).
5. **Output media** (§6): MP4 with H.264 video at the native size (1344×768 for
   768P 16:9) and 24 fps, plus **AAC-LC stereo 32 kHz audio**. The `moov` box
   comes before `mdat`, and a C2PA `uuid` box follows `ftyp`. The `File` object
   is `{url, content_type, file_name, file_size}`.
6. **Pointing clients at us** (§12): Python works with no code changes by
   setting `FAL_RUN_HOST` and optionally `FAL_QUEUE_RUN_HOST`, but it is
   https-only and uploads/tokens still go to fal. JS hard-codes `fal.run`, so the
   supported hook is `requestMiddleware` (rewrite the URL) or
   `proxyUrl: {url, when: "always"}`, where our server acts as a fal-style proxy
   that honours `x-fal-target-url`.
7. **Director** (§8) needs a WMA bridge: `POST /ice`, `POST /session` and
   `POST /session/heartbeat` on `wma.fal.run`, plus a WebRTC peer that
   receives video at 24 fps and audio, and a data channel labelled `control`
   carrying JSON text messages whose schemas are fully published.

---

## 2. Endpoint ids and what backs them

| Endpoint id | Kind | Backing fal app (PROBE) |
|---|---|---|
| `minimax/h3-max/text-to-video` | queue + sync HTTP | `POST https://queue.fal.run/minimax/h3-max/text-to-video` without a key returns 401 `Cannot access application "fal-ai/minimax-h3-turbo"`. `POST https://fal.run/…` returns 401 naming `github\|110602490/minimax-h3-turbo` |
| `minimax/h3-max/image-to-video` | queue + sync HTTP | same app family (OA-I2V `about`: "Image-text-to-video: `image_url` optional; text-only routes to t2v.") |
| `minimax/h3-max/reference-to-video` | queue + sync HTTP | OA-R2V `category: image-to-video` |
| `minimax/h3-max/director` | **realtime WebRTC session (WMA)** | App `fal-ai/minimax-h3-max-director` (ASYNC-DIR URL). `POST https://fal.run/minimax/h3-max/director` returns 404 `{"detail":"Application \"h3-max\" not found"}` (PROBE), even though LLMS-DIR lists "Endpoint: `https://fal.run/minimax/h3-max/director`" |

- `minimax/h3-max/*` is therefore an **alias** that fal resolves to the
  `minimax-h3-turbo` app. The schema titles agree: `TurboTextToVideoHailuo03Input`
  (OA-T2V), and the C2PA manifest inside every sample output names
  `fal-ai/minimax-h3-turbo` (PROBE, §6). The alias is transparent to clients.
- OpenAPI `x-fal-metadata` (OA-T2V/I2V/R2V):

| id | category | about | playgroundUrl / documentationUrl |
|---|---|---|---|
| text-to-video | `text-to-video` | "Text To Video Turbo" | https://fal.ai/models/minimax/h3-max/text-to-video, …/api |
| image-to-video | `image-to-video` | "Image-text-to-video: ``image_url`` optional; text-only routes to t2v." | …/image-to-video, …/api |
| reference-to-video | `image-to-video` | "Reference To Video Turbo" | …/reference-to-video, …/api |

- The director has **no queue OpenAPI**:
  `…/openapi.json?endpoint_id=minimax/h3-max/director` returns `null` with HTTP 404
  (PROBE). LLMS-DIR says: "do **not** call the session endpoint with `fal.run`,
  `fal.subscribe`, or the Queue API."
- Marketing text (LLMS-T2V): "fal's H3 Max is a post-trained variant of MiniMax
  H3, tuned for stronger prompt adherence and better aesthetics while
  co-optimized with our custom inference stack". Tags: `stylized, transform, lipsync`.

---

## 3. Shared types

### 3.1 `File` (output). Source: OA-T2V `components.schemas.File`

| field | type | required | description (verbatim) |
|---|---|---|---|
| `url` | string | **yes** | "The URL where the file can be downloaded from." |
| `content_type` | string \| null | no | "The mime type of the file." |
| `file_name` | string \| null | no | "The name of the file. It will be auto-generated if not provided." |
| `file_size` | integer \| null | no | "The size of the file in bytes." |

`x-fal-order-properties`: `url, content_type, file_name, file_size`. The live
H3 samples fill all four fields (PG-T2V):

```json
{"url":"https://v3b.fal.media/files/b/0aa7ecbd/cJvT63jq0mDi8-E8fYXHq_minimax-h3.mp4",
 "content_type":"video/mp4","file_name":"cJvT63jq0mDi8-E8fYXHq_minimax-h3.mp4","file_size":3554670}
```

The file name pattern is `<21-char nanoid>_minimax-h3.mp4` on all three H3
samples. The URL pattern is `https://v3b.fal.media/files/b/{8-hex prefix}/{file_name}`
(D-CDN: "CDN URL format `https://v3b.fal.media/files/b/{prefix}/{filename}`").

### 3.2 `QueueStatus`. Source: OA-T2V `components.schemas.QueueStatus`

| field | type | notes |
|---|---|---|
| `status` | enum `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED` | required |
| `request_id` | string | required |
| `response_url`, `status_url`, `cancel_url` | string | |
| `logs` | object (additionalProperties) | in practice an **array** of log objects. See §9.3 |
| `metrics` | object | |
| `queue_position` | integer | |

### 3.3 Fields common to the three HTTP endpoints

The same definitions appear in OA-T2V, OA-I2V and OA-R2V:

| field | type | default | constraints | description (verbatim) |
|---|---|---|---|---|
| `prompt` | string | none (**required**) | minLength 1, maxLength 50000 | "Text prompt for video generation" (R2V adds: "Refer to reference assets by their modality and order in the reference lists: Image 1, Image 2, Video 1, Audio 1, and so on.") |
| `duration` | integer | `5` | min 5, max 15 | "The duration of the video in seconds." |
| `resolution` | enum `"480P"`, `"768P"`, `"1080P"` | `"768P"` | | "The native generation resolution, or 1080P latent refinement from a native 768P source." |
| `seed` | integer \| null | none | | "Random seed. A random seed is selected when omitted." |
| `enable_safety_checker` | boolean | `true` | | "If set to true, the safety checker will be enabled." |
| `sync_mode` | boolean | `false` | | "Return the generated video as base64 instead of a CDN URL." |
| `prompt_expansion_mode` | string (**not** an enum in the schema, only `examples`) | `"balanced"` | listed in `required` | "How much effort to spend rewriting the prompt before generation. 'disabled' skips prompt expansion. 'balanced' returns in about a second. 'quality' spends up to ~30s on a richer prompt." Examples: `disabled`, `balanced`, `quality` |

Notes:
- `required: ["prompt", "prompt_expansion_mode"]` in all three schemas, but
  `prompt_expansion_mode` also has a default of `"balanced"`. FL omits it when
  unset and works, so the hosted server applies the default. Our server should
  treat it as optional with default `balanced` (**INFERRED** from FL L540-541
  plus the default).
- `sync_mode: true` returns "base64 instead of a CDN URL". The exact shape of
  the output (a `data:` URI in `video.url`?) is **UNDOCUMENTED**. By fal
  convention elsewhere ("Set `sync_mode: true` to receive base64 encoded
  responses", D-RT), `video.url` would hold a `data:video/mp4;base64,…` URI
  (**INFERRED**).
- Unknown fields: the OpenAPI does not set `additionalProperties: false` on
  these inputs. Whether the hosted app rejects extra keys is **UNDOCUMENTED**.
  The director messages, by contrast, explicitly forbid extra properties (§8).

### 3.4 Output fields common to all three

| field | type | required | description (verbatim) |
|---|---|---|---|
| `video` | `File` | **yes** | "The generated video" |
| `expanded_prompt` | string \| null | no | "The prompt after expansion, as sent to the model. Null when prompt expansion was disabled, left the prompt unchanged, or was performed internally by MiniMax's hosted API." |
| `timings` | object<string, number> \| null | no | "Timing breakdown in seconds. 'inference' is the DiT denoising time on the GPU backend. Null on routes that do not report backend timings." |

The live samples carry `"timings": {"inference": 2.5285604159580544}` (PG-T2V)
and `{"inference": 2.7652357418555766}` (PG-I2V). The expanded prompt is a
long structured text with sections such as `integrated_multimodal_description:`
and `overall_soundscape:`, inline dialogue tags `<d>[English] …</d>`, and
`<Picture 1>` references (PG-T2V, PG-I2V, PG-R2V).

---

## 4. `minimax/h3-max/text-to-video`

Source: OA-T2V, LLMS-T2V. Schema title `TurboTextToVideoHailuo03Input`.
Order (`x-fal-order-properties`): `prompt, duration, resolution, seed,
enable_safety_checker, sync_mode, prompt_expansion_mode, target_audio_url,
aspect_ratio`.

The §3.3 fields, plus:

| field | type | default | constraints | description (verbatim) |
|---|---|---|---|---|
| `aspect_ratio` | enum `"21:9"`, `"16:9"`, `"4:3"`, `"1:1"`, `"3:4"`, `"9:16"` | `"16:9"` | | "The aspect ratio of the generated video." |
| `target_audio_url` | string \| null | none | string: minLength 1, pattern `\S` | "Optional URL of an audio clip at least 2 seconds long (maximum 15 MB) to pin to the generated soundtrack. Longer clips are trimmed to the requested video duration, keeping the beginning. The original audio replaces the output soundtrack, padded with silence if shorter than the video, without changing playback speed. Exceptionally high sample rates may be resampled to 96 kHz. Accepts an HTTP(S) URL or a base64 data URI." |

Output (`TurboTextToVideoHailuo03Output`): `video` (required),
`expanded_prompt`, `timings`. Order: `video, expanded_prompt, timings`.

Examples (LLMS-T2V):

```json
// required-only request
{"prompt":"A white kitten chases a butterfly across a sunlit garden. Gentle camera tracking, natural movement, soft afternoon light filtering through the leaves.",
 "prompt_expansion_mode":"disabled"}
// full request
{"prompt":"…","duration":5,"resolution":"768P","enable_safety_checker":true,
 "prompt_expansion_mode":"disabled","aspect_ratio":"16:9"}
// schema example response
{"video":{"content_type":"video/mp4","file_size":6463396,
  "file_name":"--prs89fkHtWW406fmEs__NRhTqNku.mp4",
  "url":"https://v3b.fal.media/files/b/0aa46818/--prs89fkHtWW406fmEs__NRhTqNku.mp4"}}
```

Caveat (PROBE): the schema's example MP4 is **not** an H3 output. Its embedded
C2PA manifest names `fal-ai/minimax/hailuo-03/text-to-video`, and it is
2560×1440. Use the playground sample in §6 as the reference for H3 output.

cURL (LLMS-T2V):
```bash
curl --request POST --url https://fal.run/minimax/h3-max/text-to-video \
  --header "Authorization: Key $FAL_KEY" --header "Content-Type: application/json" \
  --data '{"prompt":"…","prompt_expansion_mode":"disabled"}'
```

---

## 5. `minimax/h3-max/image-to-video` and `minimax/h3-max/reference-to-video`

### 5.1 image-to-video

Source: OA-I2V, LLMS-I2V. Title `TurboImageToVideoHailuo03Input`. Order:
`prompt, duration, resolution, seed, enable_safety_checker, sync_mode,
prompt_expansion_mode, target_audio_url, image_url, end_image_url`.
**There is no `aspect_ratio` field.** The canvas follows the image.

The §3.3 fields, plus `target_audio_url` (same text as T2V), plus:

| field | type | default | description (verbatim) |
|---|---|---|---|
| `image_url` | string \| null | none | "Optional URL of the image to use as the first frame. When provided, the output canvas follows this image. If only end_image_url is provided, the canvas follows that last frame instead. If both images are omitted, the request is handled as text-to-video (16:9 by default)." Example `https://storage.googleapis.com/falserverless/example_inputs/hailuo23/pro_i2v_in.jpg` |
| `end_image_url` | string \| null | none | "Optional URL of the image to use as the last frame. It may be provided alone for end-only keyframe generation; in that case the output canvas follows this image." |

Output: same as T2V (`video`, `expanded_prompt`, `timings`).

Full example (LLMS-I2V): `{"prompt":"The camera slowly pulls back from the
scene, …","duration":5,"resolution":"768P","enable_safety_checker":true,
"prompt_expansion_mode":"disabled","image_url":"https://storage.googleapis.com/falserverless/example_inputs/hailuo23/pro_i2v_in.jpg"}`.

LLMS-I2V's "Example Response" is a generic placeholder (`"url": ""`,
`"content_type": "image/png"`). Ignore it.

### 5.2 reference-to-video

Source: OA-R2V, LLMS-R2V. Title `TurboReferenceToVideoHailuo03Input`. Order:
`prompt, duration, resolution, seed, enable_safety_checker, sync_mode,
prompt_expansion_mode, aspect_ratio, reference_image_urls,
reference_video_urls, reference_audio_urls`. **There is no `target_audio_url`.**

The §3.3 fields, plus:

| field | type | default | constraints | description (verbatim) |
|---|---|---|---|---|
| `aspect_ratio` | enum `"adaptive"`, `"21:9"`, `"16:9"`, `"4:3"`, `"1:1"`, `"3:4"`, `"9:16"` | `"adaptive"` | | "The aspect ratio of the generated video." |
| `reference_image_urls` | string[] | none | maxItems 9 | "URLs of subject/style reference images, referenced in the prompt as Image 1, Image 2, and so on. Reference images, videos, and audio clips must add up to at most 12 files." |
| `reference_video_urls` | string[] | none | maxItems 3 | "URLs of motion/reference video clips (2-15 seconds each, combined duration at most 15 seconds), referenced in the prompt as Video 1, Video 2, and so on. Reference images, videos, and audio clips must add up to at most 12 files." |
| `reference_audio_urls` | string[] | none | maxItems 3 | "URLs of reference audio clips (2-15 seconds each, combined duration at most 15 seconds), referenced in the prompt as Audio 1, Audio 2, and so on. Images, videos, and audio can be provided individually or together. Reference images, videos, and audio clips must add up to at most 12 files." |

Output (`TurboReferenceToVideoHailuo03Output`): `video` (required), **`seed`
(integer, required)**, "Base seed for reproducing the generation."; plus
`expanded_prompt` and `timings`. Order: `video, expanded_prompt, seed, timings`.
Live sample (PG-R2V): `…,"seed": 1851572118, "timings": {"inference": 4.423960474989144}}`.

Note the prompt convention difference. The schema speaks of "Image 1 / Video 1 /
Audio 1" (OA-R2V), while the model's expanded prompts use `<Picture 1>` and
`<Subject 1>` (PG-R2V).

---

## 6. Output media: container, codec, native audio

**H3 Max output includes native audio.** Evidence:

- The FL README (Backends table and "MiniMax H3 Max on fal.ai") says: "1344×768
  video with native sound, 5–15 s clips", and "fal's post-trained MiniMax H3,
  which renders video and its own sound together".
- `target_audio_url` "replaces the output soundtrack", so there is a
  soundtrack to replace (OA-T2V). The expanded prompts contain an
  `overall_soundscape:` section and `<d>` dialogue (PG-T2V/I2V). The tags
  include `lipsync` (LLMS-T2V).
- PROBE confirms it. I downloaded the three playground samples and parsed their
  ISO-BMFF boxes (no ffprobe available). All three are identical in layout:

| sample | from | video | audio |
|---|---|---|---|
| `…/0aa7ecbd/cJvT63jq0mDi8-E8fYXHq_minimax-h3.mp4` (3,554,670 B) | PG-T2V | `avc1` H.264 **High (profile_idc 100), level 4.1**, 1344×768, timescale 24000, 124 samples × 1000 ⇒ **24 fps, 5.1667 s** | `mp4a` **AAC-LC** (AOT 2), **2 ch, 32 000 Hz**, 163 frames × 1024 ⇒ 5.184 s |
| `…/0aa7ec74/bNpa9-5B0ZKqsrGfdqxZt_minimax-h3.mp4` (3,601,950 B) | PG-I2V | same, High 4.1, 1344×768, 24 fps, 124 frames | same AAC-LC 2 ch 32 kHz |
| `…/0aa91f5a/mMxt2Ga15hBB51PNsOdYo_minimax-h3.mp4` (5,252,129 B) | PG-R2V | H.264 **Baseline (profile_idc 66)**, level 4.1, 1344×768, 24 fps, 124 frames | same AAC-LC 2 ch 32 kHz |

More container facts from PROBE:
- Top-level box order is `ftyp(isom; isom,iso2,avc1,mp41)`, then `uuid` (~13.4 KB),
  `moov`, `free`, `mdat`. So the files are **faststart**.
- The muxer string is `Lavf58.76.100` (ffmpeg 4.4).
- The `uuid` box is a **C2PA manifest** (`c2pa.claim.v2`, `c2pa.hash.bmff.v3`,
  `c2pa.actions.v2`, an `ai.fal.info` assertion, IPTC
  `digitalSourceType trainedAlgorithmicMedia`, signer "fal.ai"). It records the
  endpoint (`fal-ai/minimax-h3-turbo`) and the request id. We can omit it, or
  emit our own; clients do not need it.
- CDN response headers: `content-type: video/mp4`,
  `cache-control: public, max-age=5184000, immutable`,
  `access-control-allow-origin: *`.

Duration: a 5 s request yields 124 video frames (5.167 s). The FL README says:
"The delivered file runs a frame or two past the request (5 s comes back as
5.18 s)". 768P at 16:9 renders 1344×768, and 480P at 16:9 renders 832×480
(FL README "Geometry"). Sizes for 1080P and the other aspect ratios are
**UNDOCUMENTED**.

The director's WebRTC path differs. Audio is 48 kHz (`session_info.audio_sample_rate: 48000`)
and conditioning audio is 32 kHz (`conditioning_audio_sample_rate: 32000`),
with an Opus bitrate target of 96/128/192 kb/s (§8.5).

---

## 7. Pricing (as displayed, 2026-09-27)

- **T2V and I2V** (LLMS-T2V, LLMS-I2V): "$0.025 per second at 480p, $0.04 per
  second at 768p, and $0.08 per second at 1080p". These are "promotional
  launch rates, 50% off … The discount ends September 30, after which 480p is
  $0.05/second, 768p is $0.08/second, and 1080p is $0.16/second."
- **R2V** (LLMS-R2V, PG-R2V): "Billing uses the requested output duration at
  $0.05 per second for 480p, $0.08 for 768p, and $0.16 for 1080p. Each request
  includes 4,096 reference tokens, shared across all reference images, videos,
  and audio clips; additional usage costs $0.02 per 1,000 tokens, prorated.
  Square reference images … contribute 1,024 tokens each." Reference audio is
  "approximately 80 tokens per second". "An audio track embedded in a
  reference-video file is not counted as a separately supplied audio reference."
- **Director** (LLMS-DIR): "Sessions cost $0.04 per second of video generated.
  The promotional price expires on Sep 30th, with list price being $0.08 per
  second of video. Each session is billed at a minimum of 60 seconds runtime."
- fal reports billing via the `X-Fal-Billable-Units` response header (D-HDR).

---

## 8. `minimax/h3-max/director` (realtime WebRTC, "WMA")

### 8.1 What it is

LLMS-DIR: "H3 Max Director generates continuous real-time video that can be
directed while it streams. Send new prompts during a session to evolve the
action while maintaining visual, character, and scene continuity."

**Transport.** "This endpoint publishes a WMA WebRTC contract. Open it with
`fal.realtime.open`; do **not** call the session endpoint with `fal.run`,
`fal.subscribe`, or the Queue API. The browser negotiates through fal and then
exchanges media with the model over WebRTC." (LLMS-DIR). It is **not** SSE,
**not** the msgpack `/realtime` WebSocket, and **not** `ws.fal.run`.

The ASYNC-DIR `servers.session` entry reads `protocol: "webrtc"`,
"Runtime-assigned WebRTC peer negotiated over HTTP", with
`x-fal-negotiated-by: start_session_start_session_post`. The app OpenAPI
(API-DIR) tags `/start-session` with
`x-fal-realtime: {asyncapi:{url:"./asyncapi.json"}, transport:{sessionProtocol:"wma", protocol:"webrtc", version:1}, schemaVersion:1}`.

**Media** (ASYNC-DIR `x-fal-media`, LLMS-DIR "Media contract"): the browser
sends **nothing**. It receives `video` (optional, settings `{"frameRate":24}`)
and `audio` (optional). The control plane is a WebRTC **data channel** named
`control` (JS:realtime/wma.ts L633 `pc.createDataChannel("control")`), and
ASYNC-DIR `channels.control.address: "fal"`. Messages are JSON text:
`send()` does `JSON.stringify(message)` (JS:realtime/wma.ts ~L1060), and
`onData(raw)` receives a raw string (LLMS-DIR example uses `JSON.parse(raw)`).

### 8.2 Signalling: the WMA bridge (`https://wma.fal.run`)

Sources: D-WMA ("Clients"), JS:realtime/wma.ts.

| call | request | response | source |
|---|---|---|---|
| `POST https://wma.fal.run/ice` | `{"app_id": "<endpointId>"}` | `{"ice_servers": RTCIceServer[], "status"?: string, "credential_age_seconds"?: number}`; `status: "app_managed"` means fall back | JS:wma.ts L178-206. Not in D-WMA. Without auth, PROBE gives 401 `{"error":"missing Authorization header"}`. A body without `app_id` gives 422 `text/plain` "missing field `app_id`" |
| fallback `POST https://fal.run/<endpointId>/ice` (via `context.run`) | `{}` | same `{ice_servers,…}` | JS:wma.ts L249-270 |
| `POST https://wma.fal.run/session` | `{"app_id": "minimax/h3-max/director", "sdp": "<complete offer SDP>", "type": "offer"}` | `{"session_id": "<uuid>", "sdp": "<answer SDP>", "type": "answer"}` | D-WMA; JS:wma.ts L927-938. Client timeout 120 s (`SESSION_NEGOTIATION_TIMEOUT_MS`) |
| `POST https://wma.fal.run/session/heartbeat` | `{"session_id": "…"}` | `{"alive": boolean}` | D-WMA: "Send heartbeats every 5 seconds." JS: 5 s interval, 4 s timeout. `alive:false` fails the session, and 3 consecutive non-OK or unparseable replies fail it |

Rules (D-WMA, JS:wma.ts header comment):
- **No trickle ICE.** The offer must contain all candidates.
- Auth is `Authorization: Key <key>` on every bridge call. JS attaches the
  configured credentials, or routes through the proxy (§12.1).
- "To end a session, stop sending heartbeats and close the peer connection."
  There is no DELETE call.
- "WMA sessions are not resumable. … Each call to `/session` spins up a new
  session on the runner." (D-WMA).
- The JS client queues up to 64 messages sent before the channel opens
  (`MAX_QUEUED_MESSAGES`). It also reserves the control types
  `wma.network-info.request` and `wma.network-info.response`, which carry a
  `request_id` and `path` (JS:wma.ts L37-41, L655-690). A compatible runner may
  answer `wma.network-info.request`. The exact shape is only in the client
  source (`normalizeRunnerPath`) and I did not specify it further.

**Runner side** (what the bridge calls; D-WMA "Using a raw `fal.App` with
`/start-session`", API-DIR):
- `POST /start-session` with body `StartSessionRequest` ("Offer forwarded by the
  WMA bridge."): `sdp` (string, required), `type` (const `"offer"`, default
  `"offer"`), `session_id` (string \| null), `ice_servers` (object[]),
  `ice_status` (string \| null), `credential_age_seconds` (number ≥ 0 \| null).
  Optional headers: `x-fal-caller-user-id`, `x-fal-request-id` (API-DIR).
- D-WMA: "the first SSE event you yield is your SDP answer, and the HTTP
  response stays open for the entire session". The first event is
  `data: {"sdp":…,"type":"answer","session_id":…}`. Keep-alive comments
  `: keepalive` go every 15 s in the example.
- The director app also exposes `POST /info`, which returns `DirectorInfo` (the
  same fields as the `session_info` message, §8.5), and `GET /health` (API-DIR).
- `x-fal-wma.requiredFeatures: ["configured-session/1","versioned-input/1"]`,
  `profileVersion: "0.1"`, `perspective: "client"` (ASYNC-DIR).

### 8.3 Session flow (LLMS-DIR "Session flow", ASYNC-DIR `x-fal-wma`)

1. Open the transport. "The transport being live does not mean the model is
   configured."
2. Client sends `configure` **once** with `prompt_version: 1`
   (`sequence: {scope:"session", field:"/prompt_version", initial:1, increment:1}`).
3. Wait for `configured` whose `prompt_version` matches. "Only that
   acknowledgement enables text updates. Pending, applied, or rejected input
   events do not confirm session readiness." (`configuredSession.ready`;
   `replay: "never"`).
4. Send `prompt` updates with `prompt_version` 2, 3, …. "Gaps are allowed; never
   reuse a version after uncertain delivery". Versions must stay within the JS
   safe-integer range.
5. Per-version outcomes: `prompt_pending`, then `prompt_applied` or
   `prompt_rejected`. `versionedInput.policy: "replace-pending"`: "Newer updates
   can replace older pending preparation, and the older input may never receive
   a final event."
6. The session ends on `stream_exhausted`, a session-failure `error`, or a
   transport close. "Later messages must not reopen it." Start a new session at
   version 1.

**Error taxonomy** (`x-fal-wma.errors`; `codeField: /code`,
`descriptionField: /error`):
- sessionFailure: `configuration_timeout`, `initialization_timeout`,
  `invalid_initial_image`, `invalid_initial_audio`, `invalid_initial_script`,
  `invalid_input`, `balance_unavailable`, `content_policy`,
  `generation_timeout`, `generation_failed`.
- inputFailure: `stale_prompt_version`.
- diagnostic: `invalid_message`, `not_configured`, `immutable_settings`.
- `unknownCode: "diagnostic"`, `uncorrelatedInput: "diagnostic"`.

### 8.4 Client → model messages

Source: LLMS-DIR, ASYNC-DIR. All messages have "Additional properties: not allowed".

**`configure`** (correlation `/prompt_version`)

| field | type | default | constraints / description (verbatim where quoted) |
|---|---|---|---|
| `type` | const `"configure"` | | required |
| `prompt_version` | integer | | required, ≥1 |
| `prompt` | string | | required, 1..50000 |
| `resolution` | `"480p"`, `"768p"`, `"1080p"` (**lowercase p**, unlike the HTTP endpoints) | `"768p"` | |
| `aspect_ratio` | `"16:9"`, `"9:16"`, `"1:1"` | `"16:9"` | |
| `image_url` | string (minLength 1) \| null | null | "URL of the image to use as the exact first frame. The opening prompt expansion also sees this image, so the first segment's prompt is grounded in it." |
| `end_image_url` | string \| null | null | "One-shot exact final frame for the first chunk. Director jointly plans that arrival and the following checkpoint continuation." |
| `audio_url` | string \| null | null | "Optional startup soundtrack. The stream's audio is pinned to this recording from the first chunk as FL2VA target audio, not a Ref2VA reference: every chunk is conditioned on the next window of it (plus the regenerated seam) until it ends, and the source PCM itself is what plays. Live `prompt` messages can replace or queue more audio at any time." |
| `memory` | integer | `12` | 1..50. "Number of prior segment prompts retained as context for future prompt expansion." |
| `audio_bitrate` | `96000`, `128000`, `192000` \| null | null | "Session audio target in bits/s … Explicit values use Opus audio mode on direct WebRTC; LiveKit uses its native mode with the same bitrate target. Null preserves transport defaults (WebRTC: 96000, voip). Immutable; reconnect to compare." |
| `seed` | integer \| null | null | |
| `script` | `ScriptBeat[]` (1..64) \| null | null | "Optional upfront script: beats at whole-second offsets from the first generated video. `prompt` stays the series premise; a beat prompt directs from its offset on. Cannot be combined with `end_image_url` or `audio_url` (place them in the script)." |
| `protocol_version` | const `1` | | optional |

**`ScriptBeat`** ("A direction on the associated video's clock, never the
stream clock."): `offset` (int ≥0, required; "Whole seconds from the start of
the first video generated under this script…"), `audio_url` (string \| null;
"starts playing exactly at this offset as FL2VA target audio; overlapping
sources are mixed"), `end_image_url` (string \| null; "Exact final frame of the
chunk that ends at this offset. Offsets of successive end images must be at
least three seconds apart."), and `prompt` (string 1..50000 \| null; "persists
until the next text beat"). No extra properties.

**`prompt`** (correlation `/prompt_version`)

| field | type | default | description |
|---|---|---|---|
| `type` | const `"prompt"` | | required |
| `prompt_version` | int ≥1 | | required |
| `prompt` | string 1..50000 \| null | null | |
| `end_image_url` | string \| null | null | |
| `audio_url` | string \| null | null | "FL2VA target audio for future chunks … With `audio_behavior` 'replace' (default) it starts at the next undispatched chunk and drops any queued audio; with 'queue' it plays after every previously accepted source ends, sample-exact." |
| `audio_behavior` | `"replace"`, `"queue"` | `"replace"` | |
| `replan` | boolean | `true` | "true (default) busts the planned prompt queue so the new direction applies at the next undispatched chunk; false appends the direction after the already-planned chunks." |
| `script` | `ScriptBeat[]` 1..64 \| null | null | "Exclusive with prompt/end_image_url/audio_url." With `script_mode` 'replace' it replaces "at the next undispatched chunk"; with 'append' it is queued behind the running script |
| `script_mode` | `"replace"`, `"append"` | `"replace"` | |

**`ping`**: `{"type":"ping","ts":<number>}`. **`stop`**: `{"type":"stop"}`.

LLMS-DIR example configure: `{"aspect_ratio":"16:9","protocol_version":1,
"memory":3,"prompt_version":1,"prompt":"A continuous original live-action
American sitcom produced in 1994, …","type":"configure","resolution":"768p"}`.
Example update: `{"type":"prompt","prompt":"They follow a narrow path down to
the harbor.","prompt_version":2}`.

### 8.5 Model → client messages

Source: LLMS-DIR, ASYNC-DIR. "req" means required.

| type | fields |
|---|---|
| `session_info` | See the constants below. `type` is the only required field |
| `configured` (corr.) | `prompt_version` req; `enable_safety_checker` bool req; `aspect_ratio` (16:9, 9:16, 1:1 \| null); `memory` (1..50 \| null); `chunk_duration` (5..15 \| null); `acceleration` (`none`, `regular` \| null); `has_initial_audio`, `has_initial_image` (bool \| null); `audio_bitrate` (96000/128000/192000 \| null); `resolution` (`480p/544p/640p/704p/768p` or `480p/768p/1080p` \| null) |
| `prompt_pending` (corr.) | `prompt_version` req |
| `prompt_applied` (corr.) | `prompt_version` req; `script_origin_chunk_index`, `script_queued`, `script_beats` (≥1), `script_mode` (all nullable) |
| `prompt_rejected` (corr.) | `prompt_version` req; `reason` req in `content_policy`, `preparation_failed`, `stale_prompt_version`, `invalid_script`, `infeasible_timing`, `invalid_audio`, `invalid_image`, `queue_full`; `error` string \| null |
| `audio_pending` | `behavior` (replace\|queue), `prompt_version` req |
| `audio_applied` | `behavior`, `source` (string), `duration_seconds` (>0), `transcribed` (bool), `remaining_seconds` (≥0), `queued_sources` (int ≥0), `prompt_version`, all req. "`starts_at_chunk_index` is the first chunk that can carry it", but that field is **not** in the schema's property list |
| `audio_rejected` | `error`, `prompt_version`, `reason` (`invalid_audio`, `content_policy`, `preparation_failed`, `queue_full`, `stale_prompt_version`), all req |
| `audio_exhausted` | `chunk_index` req, `silent_seconds` req, `source_version` (int ≥0 \| null). "Accepted audio ran out inside this chunk; the rest is silence." |
| `chunk` | req: `next_generation_estimate_seconds`, `buffer_depth_seconds`, `scheduling_slack_ms`, `generated_frame_count` (>0), `trimmed_context_frames`, `requested_duration_seconds` (5..15), `generation_seconds`, `scheduling_lead_ms`, `buffer_depth_chunks`, `chunk_index`, `playback_seconds` (>0), `route` (`gorgonea`, `betelgeuse`, `regulus`, `unknown`), `dispatch` `{overhead_ms, wall_ms, phases_ms{…:number}, classified_ms}`, `prompt_version`. Optional: `hard_cut` (false), `script_version`, `presented_frame_count`, `script_end_keyframe`, `script_offset_seconds`, `native_playable_frame_count` |
| `chunk_metrics` | `chunk_index`, `route`, `units` (const `"ms"`), `gauges` {…:number}, `phases_ms` {…:number} req; `chunk_consumable_ready_ms`, `chunk_consumable_interval_ms` (nullable) |
| `deadline_missed` | `chunk_index`, `late_by_seconds`, `behavior` const `"freeze_video_and_silence_audio_until_ready"` |
| `error` (corr.) | `error` string req, `code` req (the enum in §8.3 plus the diagnostics), `prompt_version` \| null, `detail` object[] \| null |
| `pong` | `client_ts` req |
| `session_metrics` | `units` "ms", `history_limit` (≥1), `session_wall_ms`, `history_size`, `gauges`, `phases` {name: {total_ms, p95_ms, p50_ms, count, max_ms}} req; `final` (bool, default false) |
| `stream_exhausted` | `chunks` (int ≥0), `reason` (`stopped`, `session_limit`) req |

**`session_info` / `DirectorInfo` constants** (LLMS-DIR, API-DIR), which document
the hosted engine's shape:
`app:"minimax-h3-max-director"`, `protocol_version:1`, `fps:24`,
`chunk_seconds:10`, `default_chunk_duration:10`, `min_chunk_duration:5`,
`max_chunk_duration:15`, `continuation_context_frames:39`,
`continuation_playback_seconds:8.5` (default), `audio_sample_rate:48000`,
`conditioning_audio_sample_rate:32000`, `audio_bitrates:[96000,128000,192000]`,
`default_audio_bitrate:null`, `aspect_ratios:["16:9","9:16","1:1"]`,
`resolutions:["480p","768p","1080p"]`, `max_session_seconds:null`,
`session_limit_scope:"configured"` (or `effective`), `prompt_expander:"fast"`,
`one_session_per_machine:true`, `controller_machine_type:"XL"`,
`backend_selection:"minimax-h3-turbo-balancer"`, `prompt_deck_size:6`,
`audio_conditioning:true`, `audio_behaviors:["replace","queue"]`,
`max_audio_source_seconds:600`, `prompt_context_segments:12`,
`default_memory:12`, `min_memory:1`, `max_memory:50`,
`default_acceleration:"regular"`, `accelerations:["none","regular"]`,
`scripts:true`, `script_modes:["replace","append"]`, `script_max_beats:64`,
`script_max_end_images:16`, `script_max_audio_beats:8`, `script_max_queued:4`,
`script_max_pending:4`, `script_max_decoded_audio_bytes:67108864`,
`script_session_max_decoded_audio_bytes:335544320`,
`script_min_end_image_spacing_seconds:3`, `script_min_chunk_seconds:3`,
`script_min_opening_chunk_seconds:5`,
`client_message_types:["configure","ping","prompt","stop"]`, and
`server_message_types` (the 16 types above).

### 8.6 What is and is not documented for the director

Documented:
- the full control-message JSON schemas both ways (ASYNC-DIR);
- the session state machine and error classes;
- the media directions and 24 fps video;
- the bridge's `/session` and `/session/heartbeat` (D-WMA);
- `/ice` (client source only);
- the runner's `/start-session` request schema (API-DIR) and SSE answer
  convention (D-WMA example);
- pricing.

UNDOCUMENTED:
- video codec and payload type on the WebRTC track;
- the video frame size per resolution/aspect (`configured.resolution` may be
  `544p/640p/704p`, which hints at internal tiers);
- the audio codec (the `audio_bitrate` text implies Opus on direct WebRTC);
- the order in which the runner emits `session_info` versus `configured`
  (**INFERRED**: `session_info` early, since it is listed in `server_message_types`);
- whether the runner or the client creates the data channel. The JS client
  creates `control`, while the D-WMA sample runner creates its own `ping`
  channel. A compatible runner should accept the client-created channel via
  `ondatachannel` (**INFERRED**).

The "LiveKit" mention in `audio_bitrate` implies a second transport exists. It
is not described anywhere public (API-DIR).

---

## 9. Queue API (`https://queue.fal.run`)

### 9.1 Submit

`POST https://queue.fal.run/{endpoint_id}` with a JSON body equal to the model
input (D-QUEUE, OA-T2V `paths./minimax/h3-max/text-to-video.post`). Optional
query parameters:
- `fal_webhook=<url>` (D-QUEUE, D-HOOK; JS:queue.ts L338-341; PY:client.py L1840);
- `fal_max_queue_length=<int>`: "Reject the request with `429` if the
  endpoint's queue already has more than this many requests waiting" (D-HDR).

Response (D-QUEUE):
```json
{"request_id":"764cabcf-b745-4b3e-ae38-1200304cf45b",
 "response_url":"https://queue.fal.run/fal-ai/flux/schnell/requests/764cabcf.../response",
 "status_url":"https://queue.fal.run/fal-ai/flux/schnell/requests/764cabcf.../status",
 "cancel_url":"https://queue.fal.run/fal-ai/flux/schnell/requests/764cabcf.../cancel",
 "queue_position":0}
```
The webhook page shows the submit response also carrying `gateway_request_id`
(D-HOOK).

**Doc/implementation mismatch, and which to follow:**
- D-QUEUE's example puts the full endpoint id in the URLs and a `/response`
  suffix on `response_url`.
- The OpenAPI defines the result at `GET …/requests/{id}` with no `/response`
  (OA-T2V).
- Every client builds the app-only form with no suffix: JS:queue.ts L359-552,
  and PY:client.py L1547-1552 (`base_url = f"{QUEUE_URL_FORMAT}{owner}/{alias}/requests/{request_id}"`,
  `response_url=base_url`).
- The real client in FL observed fal returning URLs of the form
  `…/minimax/h3-max/requests/{id}` ("without the endpoint's sub-path", FL README).

**Recommendation:** return app-only URLs without `/response`, and serve all of
these:
- `GET /{o}/{a}/requests/{id}`
- `GET /{o}/{a}/{sub}/requests/{id}`
- `…/response` as an alias (**INFERRED**, defensive).

Python follows `status_url`, `response_url` and `cancel_url` **from the submit
response** (PY:client.py L1873-1875). JS ignores them and rebuilds them
(JS:client.ts L137-143).

### 9.2 Headers accepted on submit

Sources: D-HDR; JS:headers.ts and queue.ts L313-358; PY:_headers.py.

| header | meaning | notes |
|---|---|---|
| `Authorization: Key <FAL_KEY>` | auth | `FAL_KEY` or `FAL_KEY_ID:FAL_KEY_SECRET` (JS:config.ts L135-146; PY:auth.py L107-111) |
| `X-Fal-Request-Timeout` | "time-to-start" deadline in seconds (> 1 in JS, > 0.1 per D-HDR). On expiry: 504 with `X-Fal-Request-Timeout-Type: user` | JS sends lowercase `x-fal-request-timeout` |
| `X-Fal-Runner-Hint` | routing affinity | |
| `X-Fal-Queue-Priority` | `normal` (default) or `low` | JS **always** sends it (`priority ?? "normal"`) |
| `X-Fal-Tags` | packed `key=value` tags; ≤10 pairs, key `^[a-z0-9._-]+$` ≤64, value ≤256, total ≤1024 B, `fal.` prefix reserved | JS:headers.ts |
| `X-Fal-Object-Lifecycle-Preference` | JSON `{"expiration_duration_seconds":…, "initial_acl":{…}}` | |
| `X-Fal-Store-IO` | `"0"` disables payload storage | |
| `X-Fal-No-Retry` | `1`/`true`/`yes` disables retries | |
| `X-Fal-Retry-Config` | JSON per-condition retry budgets (own apps only) | |
| `x-app-fal-disable-fallback` | disables model fallback | |

**Response headers** (D-HDR): `x-fal-request-id`, `X-Fal-Billable-Units`,
`X-Fal-Served-From`, `X-Fal-Request-Timeout-Type`, `X-Fal-Error-Type`,
`x-fal-runner-hints`. A compatible server should at least emit
`x-fal-request-id`.

### 9.3 Status

`GET …/requests/{id}/status?logs=1` (D-QUEUE; OA-T2V `logs` is a number, "`1`
… or not (`0`)").

```json
{"status":"IN_QUEUE","request_id":"764cabcf-...","queue_position":2,"response_url":"…"}
{"status":"IN_PROGRESS","request_id":"…","response_url":"…",
 "logs":[{"message":"Loading model weights...","timestamp":"2026-02-17T10:30:01.123Z"}]}
{"status":"COMPLETED","request_id":"…","response_url":"…",
 "logs":[{"message":"Done.","timestamp":"2026-02-17T10:30:05.789Z"}],
 "metrics":{"inference_time":3.42}}
```

On failure, `COMPLETED` also carries `error` (a human string) and `error_type`
(D-QUEUE table; D-REQERR). The JS type for a log entry is
`{message, level: "STDERR"|"STDOUT"|"ERROR"|"INFO"|"WARN"|"DEBUG", source: "USER", timestamp}`
(JS:types/common.ts L85-90). The JS `BaseQueueStatus` declares `request_id`,
`response_url`, `status_url` and `cancel_url` on every status. `isQueueStatus()`
tests `obj.status && obj.response_url` (JS:types/common.ts L96-127), so **always
include `response_url`**.

Unknown id: PROBE `GET https://queue.fal.run/minimax/h3-max/requests/00000000-…/status`
without a key returned 404 `{"status":"NOT_FOUND"}`. FL README: "fal answers
`404 NOT_FOUND` under a usable key and `401 invalid key credentials` under a bad one".

Client polling cadence: JS 500 ms (`DEFAULT_POLL_INTERVAL`, JS:queue.ts L31);
Python 0.1 s (`DEFAULT_QUEUE_POLL_INTERVAL`, PY:client.py L1155); FL 2 s. Status
must therefore be cheap.

**Status stream:** `GET …/requests/{id}/status/stream?logs=1` returns
`text/event-stream`. "Each event is a JSON status object in the same format as
the polling endpoint. The connection stays open until the status reaches
`COMPLETED`." (D-QUEUE; JS:queue.ts L380-400). JS uses it only when
`subscribe({mode:"streaming"})` is set. The default is polling.

### 9.4 Result

`GET …/requests/{id}` returns the model output JSON (OA-T2V). If it is fetched
before completion, the behaviour is **UNDOCUMENTED**. JS only calls it after
`COMPLETED`; Python's `get()` polls first.

### 9.5 Cancel

`PUT …/requests/{id}/cancel` (D-QUEUE):

| HTTP | body | meaning |
|---|---|---|
| 202 | `{"status":"CANCELLATION_REQUESTED"}` | accepted; may still complete if mid-processing |
| 400 | `{"status":"ALREADY_COMPLETED"}` | |
| 404 | `{"status":"NOT_FOUND"}` | |

The OpenAPI instead declares `200 {"success": boolean}` (OA-T2V). The docs table
is newer and more specific, so follow it. JS treats any 2xx as success.

### 9.6 Errors

- **Model and validation errors** (D-ERR): HTTP 4xx/5xx, header
  `X-Fal-Needs-Retry`, body `{"detail":[{loc, msg, type, url, ctx?, input?}]}`.
  The first-party types include `internal_server_error` (500),
  `generation_timeout` (504), `downstream_service_error` (500),
  `downstream_service_unavailable` (500), `content_policy_violation` (422),
  and others. Pydantic types pass through. JS raises `ValidationError` for any
  422 JSON body (JS:response.ts).
- **Infrastructure errors** (D-REQERR): `{"detail":"<string>","error_type":"<type>"}`
  plus the `X-Fal-Error-Type` header. Types are `request_timeout` (504),
  `startup_timeout` (504), `runner_scheduling_failure` (503),
  `runner_connection_timeout` / `runner_disconnected` /
  `runner_connection_refused` / `runner_connection_error` (503),
  `runner_incomplete_response` (502), `runner_server_error` (500),
  `client_disconnected` / `client_cancelled` (499), `bad_request` (400),
  `internal_error` (500).
- **Concurrency limit**: 429 with type `concurrent_requests_limit` and header
  `X-Fal-needs-retry: 1` (D-CONC).

### 9.7 Webhooks

Source: D-HOOK.
- Trigger: `?fal_webhook=<url>` on submit. On completion fal POSTs:
  `{"request_id","gateway_request_id","status":"OK"|"ERROR","payload":{…output…}}`.
  An error adds `"error":"Invalid status code: 422"` with the model's error body
  in `payload`. If the payload is not serializable: `"payload": null,
  "payload_error": "…"`.
- Delivery: a 2xx acknowledges. The first attempt has a 15 s timeout and
  retries have 120 s. Retries back off up to 31 times until the stored result
  expires (~1 h, or ~6 min for results ≥10 KB). A 3xx is a permanent failure,
  and so is a private/loopback target.
- Signing: headers `X-Fal-Webhook-Request-Id`, `X-Fal-Webhook-User-Id`,
  `X-Fal-Webhook-Timestamp` (unix seconds, ±300 s) and `X-Fal-Webhook-Signature`
  (hex ED25519 over `request_id\nuser_id\ntimestamp\nhex(sha256(body))`). Keys
  come from the JWKS at `https://rest.fal.ai/.well-known/jwks.json` (`x` =
  base64url public key). Our server cannot sign with fal's keys. Receivers that
  verify against fal's JWKS will reject our webhooks unless they are configured
  with our JWKS (**INFERRED**).
- Webhook source IPs: `https://api.fal.ai/v1/meta` returns `webhook_ip_ranges`.

### 9.8 Reliability semantics clients may rely on

D-QUEUE, D-REL: "Requests in the queue are never dropped." Runner failures
(503, 504, connection) are re-queued up to 10 times. Backup domains are
`falrun.com` and `queue.falrun.com`; the Python client "tries the corresponding
backup once if the primary connection fails or times out" (D-REL; PY:client.py
L91-96). This matters for a self-hosted server: with `FAL_RUN_HOST` overridden,
the backup map no longer matches, so no fallback happens (**INFERRED** from the
map keyed on the literal hostnames).

---

## 10. Other fal transports

### 10.1 Sync: `POST https://fal.run/{endpoint_id}`

The body is the model input and the response is the model output in the same
connection (D-SYNC). There is no queue and no server-side retries. The JS
client retries up to 3 times on retryable statuses (JS:client.ts L127-133). The
result's `requestId` comes from the `x-fal-request-id` header.

### 10.2 HTTP streaming: `POST https://fal.run/{endpoint_id}/stream`

D-STREAM: "Streaming is only supported by models that have a `/stream`
endpoint." The response is `text/event-stream`, "each line prefixed with
`data: `". The JS client JSON-parses each event's `data` and resolves `done()`
with **the last event** (JS:streaming.ts L291-331). A non-SSE content type is
treated as raw binary chunks. The default idle timeout is `EVENT_STREAM_TIMEOUT`.

The H3-max OpenAPIs expose **no `/stream` path** (OA-T2V/I2V/R2V), so whether
`…/text-to-video/stream` exists on fal is **UNDOCUMENTED**. If we add one, the
last SSE event should be the final output object.

In client-connection mode, JS appends `?fal_jwt_token=<token>` and sends no
Authorization header (JS:streaming.ts L200-215).

### 10.3 Realtime WebSocket: `wss://fal.run/{endpoint_id}/realtime`

This does **not** apply to H3 Max (D-RT: "Only models with an explicit
real-time endpoint are supported"; the listed ones are fast-lcm-diffusion and
fast-turbo-diffusion). For completeness:
- URL: `wss://fal.run/{appId}{path|/realtime}?fal_jwt_token=<t>[&max_buffering=1..60]`
  (JS:realtime/protocol.ts L32-48; PY:client.py L798-818).
- Framing is msgpack binary by default. Text frames are JSON. Errors arrive as
  `{"type":"x-fal-error","error":"…","reason":"…"}`, and `error:"TIMEOUT"` is
  ignored. `{"type":"x-fal-message"}` is meta and skipped. Unauthorized is
  `{"status":"error","error":"Unauthorized"}` (JS:realtime/protocol.ts L64-85;
  PY:client.py L880-905).
- Tokens: `POST https://rest.fal.ai/tokens/` with `{"allowed_apps":[<alias>],
  "token_expiration":120}` and `Authorization: Key …`. The response is a JSON
  string (or `{token}` / `{detail}`) (JS:auth.ts; PY:client.py L836-844,
  L1742). The docs' token-provider example instead uses
  `POST https://rest.fal.ai/tokens/realtime` with `{allowed_apps:[app], duration:120}`
  (D-RT). `https://rest.alpha.fal.ai/tokens/` also answers (401 without a key,
  PROBE), but the current clients use `rest.fal.ai`.

### 10.4 HTTP over WebSocket: `wss://ws.fal.run/{model_id}`

D-WS covers this. It carries any endpoint's normal HTTP contract. The client
sends the JSON payload. The server replies with
`{"type":"start","request_id","status":200,"headers":{…}}`, then the body as
binary or text frames, then
`{"type":"end","request_id","status":200,"time_to_first_byte_seconds":…}`.
Auth is `Authorization: Key` on the handshake. It is not used by
`@fal-ai/client` or `fal_client` (D-WS warning), so it is low priority.

---

## 11. Files: inputs and outputs

- Inputs are URLs. `data:` URIs are accepted ("Models also accept base64-encoded
  data URIs", D-CDN), and `target_audio_url` explicitly "Accepts an HTTP(S) URL
  or a base64 data URI" (OA-T2V). Our server must fetch `http(s)` URLs and
  decode `data:` URIs for every `*_url` field.
- JS `transformInput` uploads any `Blob`/`File` found anywhere in the input
  **before** submit and substitutes the URL (JS:storage.ts `transformInput`). The
  upload flow is:
  1. `POST https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3`
     with `{content_type, file_name}`. This goes through `dispatchRequest`, so
     it is proxy/middleware-rewritable.
  2. The response is `{upload_url, file_url}`.
  3. `PUT upload_url` with the raw bytes. This uses **plain fetch, with no
     middleware and no auth**.
  4. Files over 90 MB use `…/initiate-multipart`, then
     `PUT {upload_url}/{part}` and `POST {upload_url}/complete`
     `{parts:[{partNumber, etag}]}` (JS:storage.ts L308-510).
  So our server can serve `initiate` and hand back an `upload_url` on itself.
- Python `upload_file` / `upload`:
  1. Get a token with `POST https://rest.fal.ai/storage/auth/token?storage_type=fal-cdn-v3`,
     which returns `{token, token_type, base_url, expires_at}`.
  2. `POST https://v3.fal.media/files/upload`, which returns `{access_url}`.
  3. On failure, fall back to repository `"fal"`
     (`rest.fal.ai/storage/upload/initiate?storage_type=gcs`).
  **All of these hosts are hard-coded** (PY:client.py L88, L99, L176, L1404-1477).
- Outputs: `File.url` on fal's CDN, publicly readable by default (D-CDN).
  Expiry is controllable with `X-Fal-Object-Lifecycle-Preference`. FL downloads
  results **without** the key ("must not be sent one"), so our output URLs must
  be fetchable without auth, or carry a signature in the query.

---

## 12. How the clients build URLs, and how to point them at us

### 12.1 `@fal-ai/client` (JS/TS)

Base URLs are **hard-coded**:

| request | URL |
|---|---|
| `run` | `https://fal.run/{id}/{path}` |
| queue calls | `https://queue.fal.run/{id}` for submit, and `https://queue.fal.run/{owner}/{alias}/requests/{rid}[…]` for status/result/cancel (JS:request.ts L135-162, `buildUrl` with `subdomain:"queue"`; JS:queue.ts L313-552) |
| REST | `https://rest.fal.ai` (JS:config.ts L204) |
| WMA | `https://wma.fal.run` (JS:realtime/wma.ts L32) |
| realtime WebSocket | `wss://fal.run` (JS:realtime/protocol.ts L47) |

`parseEndpointId` splits `owner/alias/path…`. `workflows` and `comfy` are
namespaces that shift the split by one (JS:utils.ts L35-51).

A full URL passed as the endpoint id is accepted only if it is `https:`, uses
the default port, and has a host of `fal.ai`, `*.fal.ai`, `fal.run` or
`*.fal.run` (JS:utils.ts L77-94). Otherwise it throws "URLs must be https://
and point at a fal.run or fal.ai host".

There is **no base-URL or host option.** The override knobs are:
1. **`requestMiddleware`** in `createFalClient({requestMiddleware})` /
   `fal.config({...})`. This is an async `(req:{url, method, headers}) => req`
   applied to every `dispatchRequest`, which covers run, queue, stream, the
   token fetch and upload-initiate (JS:request.ts L57-61). It is also applied
   to WMA bridge calls **after** a host allow-list check (`wma.fal.run` or
   `*.fal.ai` must be the *original* URL; the middleware may then rewrite it)
   (JS:realtime.ts L50-70, L1519-1585). Rewriting
   `https://(queue\.)?fal\.run/` and `https://wma\.fal\.run/` to our origin is
   the cleanest way to point JS at a self-hosted server.
   Not covered: the raw `PUT` of uploads (which goes to whatever `upload_url`
   we return, so that is fine) and the realtime WebSocket URL (`wss://fal.run`,
   built outside the middleware).
2. **`proxyUrl`**, either a string (browser only) or
   `{url, when: "always" | "browser" | fn}` (JS:config.ts). It rewrites every
   request to `proxyUrl` and moves the real target into the **`x-fal-target-url`**
   header (JS:middleware.ts L65-100). A self-hosted server can act as that
   proxy: read `x-fal-target-url`, ignore the fal host, and route on its path.
   This matches fal's documented proxy protocol ("Accept all HTTP methods (GET,
   POST, PUT, DELETE); Read the target URL from the `x-fal-target-url` header;
   Add your API key", D-PROXY). The director docs require this path
   (`createFalClient({ proxyUrl: "/api/fal/proxy" })`, LLMS-DIR).
3. **`fetch`**: a custom fetch implementation, which could also redirect
   (**INFERRED**).
4. **Credentials**: `credentials` option, else env `FAL_KEY`, else
   `FAL_KEY_ID` + `FAL_KEY_SECRET` (JS:config.ts L125-146). The header is always
   `Authorization: Key <credentials>` (JS:request.ts L62-64).

Other JS behaviour our server sees:
- `Accept: application/json`, `Content-Type: application/json`, and
  `User-Agent` outside browsers.
- Requests are sent `mode: "cors"` except on Cloudflare Workers, so browser
  clients need CORS on our server (JS:request.ts L85).
- The JS client's `fal.realtime.open` / `wma()` API is `@experimental` and
  ships in `@fal-ai/client@alpha` (LLMS-DIR install line; npm `alpha` =
  1.11.0-alpha.4).

### 12.2 `fal_client` (Python)

Sources: PY:auth.py L84-85 and PY:client.py L85-99.

```python
FAL_RUN_HOST = os.environ.get("FAL_RUN_HOST", "fal.run")
FAL_QUEUE_RUN_HOST = os.environ.get("FAL_QUEUE_RUN_HOST", f"queue.{FAL_RUN_HOST}")
RUN_URL_FORMAT = f"https://{FAL_RUN_HOST}/"
QUEUE_URL_FORMAT = f"https://{FAL_QUEUE_RUN_HOST}/"
REALTIME_URL_FORMAT = f"wss://{FAL_RUN_HOST}/"
REST_URL = "https://rest.fal.ai"      # hard-coded
CDN_URL = "https://v3.fal.media"      # hard-coded
```

- **Host override is supported**: set `FAL_RUN_HOST=fal.example.com` (and
  optionally `FAL_QUEUE_RUN_HOST`). Both are read **at import time** and are
  **https/wss only**. A `host:port` value works, because the value is pasted
  into the URL verbatim (**INFERRED**). Plain-http local testing therefore needs
  TLS on our server, or a TLS-terminating proxy.
- Submit: `POST {QUEUE_URL_FORMAT}{application}[/{path}][?fal_webhook=…]`. It
  then follows `status_url`, `response_url` and `cancel_url` **from our
  response** (PY:client.py L1835-1876), so our server fully controls those.
  Handles rebuilt from a stored request id use
  `{QUEUE_URL_FORMAT}{owner}/{alias}/requests/{id}` (PY:client.py L1547, L1624).
- Status: `GET status_url` with `params={"logs": with_logs}`, a bool
  (PY:client.py L1561-1570). It is parsed strictly (PY:client.py L711-726, see
  §1 item 3).
- Stream: `POST {RUN_URL_FORMAT}{application}/stream` via `httpx_sse`, with
  `event.json()` per event (PY:client.py L1989-2023).
- Tokens: `POST https://rest.fal.ai/tokens/` (hard-coded). Uploads are
  hard-coded to `rest.fal.ai` and `v3.fal.media` (§11). **Python uploads and
  realtime tokens cannot be redirected** without patching module constants,
  for example `fal_client.client.REST_URL` / `CDN_URL`, which are module
  globals (**INFERRED**).
- Credentials: `FAL_KEY`, or `FAL_KEY_ID`/`FAL_KEY_SECRET`
  (PY:auth.py L107-111), or `SyncClient(key=…)`.

### 12.3 Plain HTTP

Any client that takes a base URL, such as FL's `FAL_BASE_URL`, just works with
the routes in §9, provided the app-only request paths are served.

---

## 13. Observed client usage (the user-supplied fal client)

Source: FL = `/home/user/refsrc/d5fc0bdf-infinite-livestream-source/streaming-client/fal_link.py`
plus `README.md` §"Backends" → "MiniMax H3 Max on fal.ai". I read neither
`config.py` nor `.env.example`; access to them was denied in this session, so
the default values of `FAL_BASE_URL` and the other settings are not recorded
here. That leaves open whether `FAL_BASE_URL` defaults to
`https://queue.fal.run` (**INFERRED**, because FL posts to
`{FAL_BASE_URL}/{FAL_MODEL}` and calls it "fal's queue REST API").

What this real client does on the wire:

| step | exact call | FL lines |
|---|---|---|
| Auth | every API call carries `Authorization: Key <FAL_KEY>` (one `aiohttp` session). CDN downloads use a **separate session with no auth** | L373-376, README "Auth" |
| Reachability/key probe | `GET {base}/{owner}/{app}/requests/{random-uuid4}/status`. 404 means reachable with a good key, and 401 means a bad key. Tolerated statuses are {400, 404, 405, 422} | L401-431, README "Failures" |
| Submit | `POST {base}/minimax/h3-max/text-to-video` with JSON `{"prompt", "duration":<int 5..15>, "resolution":<FAL_RESOLUTION>, "aspect_ratio":<FAL_ASPECT_RATIO>, "seed":<int>, ["prompt_expansion_mode"]}`. `prompt_expansion_mode` is sent only when configured. Needs `request_id`; uses `status_url`, `response_url` and `cancel_url` **only if they are on the same host as `base`**, else rebuilds `{base}/{owner}/{app}/requests/{id}[/status|/cancel]` | L196-214, L530-554 |
| App-only addressing | "A request is addressed by the app (`owner/app`) that built it, not by the endpoint's sub-path: `minimax/h3-max/text-to-video` submits, and `minimax/h3-max/requests/{id}` follows it." | L161-164 |
| Poll | `GET status_url` every **2 s**. `COMPLETED` ends the poll. `IN_QUEUE`/`IN_PROGRESS` continue. **Any other status** (e.g. a `FAILED` string) fails the clip with `status.error`. Gives up and cancels after 900 s | L88, L96, L555-571 |
| Result | `GET response_url`, then the video URL from `result.video.url`. It also tolerates `video` as a list or string, or a `videos` array | L572-577, L787-800 |
| Download | `GET <video.url>` unauthenticated, 300 s timeout. A 404 is fatal for the clip; anything else counts as "unreachable" | L622-637 |
| Cancel | `PUT cancel_url`, best effort. 400 and 404 are tolerated | L639-653 |
| Error mapping | HTTP 400/413/415/422 means rejected (clip fails). 401/403 means a bad key. Other non-200 responses, 5xx, 429 and timeouts mean "unreachable" (retry later). The error text comes from `detail` (string, or a list of `{msg}`) | L101-104, L586-620, L816-832 |
| Media handling | ffmpeg decodes the MP4 to 24 fps rgb24 scaled/letterboxed onto the canvas. **Audio is decoded to 48 kHz mono s16le**. A clip with no decodable audio becomes silence. `generates_audio = True` | L153, L683-773 |

Implications for our server:
- Submit must return **HTTP 200**, not 201 or 202, with `request_id`. FL treats
  only 200 as success (L608).
- The `status_url` host must equal the submit host, or FL rebuilds the URL
  (harmless if we serve the app-only form).
- Only `IN_QUEUE`, `IN_PROGRESS` and `COMPLETED` are non-fatal. A failed request
  may be `COMPLETED` with `error` (fal's documented form, D-QUEUE). In that
  case FL would then GET the result, and it must not contain `video`, so that FL
  fails the clip. A non-standard terminal status string also works with FL,
  but it breaks the Python SDK (`ValueError: Unknown status`, PY:client.py L726).
  So use `COMPLETED` + `error`/`error_type`, and make the result GET return an
  error status (4xx/5xx) with `detail` (**INFERRED**: the safest mapping for
  all three clients).
- The FL README also states that the output "run[s] a frame or two past the
  request (5 s comes back as 5.18 s)". That matches PROBE (§6).

---

## 14. Gaps and open questions

1. `sync_mode: true` output shape: **UNDOCUMENTED** (`data:` URI in
   `video.url` is INFERRED).
2. Output frame sizes for 1080P and non-16:9 aspect ratios, and I2V/R2V
   "adaptive" canvas rules beyond the prose: **UNDOCUMENTED**. Only 768P 16:9 →
   1344×768 was observed (PROBE) and 480P 16:9 → 832×480 is stated (FL README).
3. Whether H3 Max has a `/stream` SSE endpoint: not in the OpenAPI, so
   **UNDOCUMENTED**.
4. The exact failure body of `GET …/requests/{id}` for a failed request:
   **UNDOCUMENTED**. The status carries `error`/`error_type`.
5. Director: the WebRTC codecs, frame size per resolution, emission order of
   `session_info`, data-channel ownership, the `wma.network-info` reply shape,
   and the LiveKit alternative transport are all undocumented (§8.6). Driving a
   live session would settle them. That needs a funded `FAL_KEY` and costs at
   least 60 s of billing.
6. The cancel response code (202 per docs vs `200 {"success"}` per OpenAPI).
   Follow the docs. Clients accept any 2xx.
7. Webhook signatures cannot be fal-signed by a third-party server (§9.7).
8. `config.py` / `.env.example` in the user's client were not read (access
   denied), so its default `FAL_BASE_URL` and other defaults are unknown here.
