# fv-serve console

Browser pages served by `fv-serve` itself for minting API keys, trying the
fal-compatible endpoints (modelled on fal.ai model pages: Playground + API
tabs), live directing and streaming, and the native API's own fields. They are static files embedded in the binary
(`crates/fastvideo-serve/console/`, `include_str!`): no build step, no CDN,
scripts from the server's own origin only (strict CSP). Everything they do
goes through the public APIs, so anything the console does can also be done
with `curl`.

| Page | What it does |
|---|---|
| `/console` | Server URL (defaults to the page's origin) and API key (kept in `localStorage`), key check via `GET /fv/v1/capabilities`, the mounted APIs (its `protocols`), the **served models** with tier, recipe (attention, VAE, steps, profile), tasks and their live page (causal: Live stream, with the session-length rule; duplex: Live input), the tier bindings, and the mounted fal apps and endpoints (an app whose model is not served lists none). With `FV_AUTH_MODE=none` (capabilities report `auth.mode = "none"`) there is no key field, banner or check, and the console sends no `Authorization`; the admin page still needs the admin token |
| `/console/admin` | Admin token (kept in `sessionStorage`, this tab only); create, list and revoke API keys; **Experimental features** (§7) |
| `/console/models/{owner}/{alias}/{task}` | One endpoint, e.g. `minimax/h3-max/reference-to-video`: variant switcher (`h3-max`, `h3-turbo`, `h3-draft`), task tabs, Playground and API tabs |
| `/console/models/{owner}/{alias}/director` | Live director (WebRTC) page, clip or causal (§4) |
| `/console/stream` | Live stream: a causal model (SF-Wan, LongLive) over the Reactor runtime or a native `/fv/v1/streams` WHIP publish; prompt switches, pause, reset, stop, stats, licence (§4c) |
| `/console/native` | Native API: `POST /fv/v1/jobs` with the fields only it takes, on any served model or tier alias (§4d) |
| `/console/live` | Live input: publish the camera and microphone (`getUserMedia`) to a duplex model and watch its output, over native WHIP ingest or the Reactor runtime (§4b) |
| `/console/avatar` | Script avatar: photo, script, scene, speech rate, duration, seed (and an optional driving voice) into the Reactor runtime's avatar mode; the WebRTC stream and a per-window table (build time, real-time factor, when it started, how long playout waited) |

Disable the pages with `FV_CONSOLE=0` (or `server.console = false`); the
APIs stay up.

**Fake engine knobs** (tests, demos): `FV_FAKE_H3_1080P=1` gives the fake
H3 max / turbo models the 1080P tier; `FV_FAKE_DEVICE=a100` (or `l40s`,
`h100`, `b200`, `rtx-pro-6000`, `sm<NN>[:<GiB>]`) simulates a GPU for the
startup capability check, so a model it cannot run shows as failed with
its reason (the fake H3 models need FP8).

**Server status.** Every page's top bar has a status strip: one dot per
pool (green ready, amber busy / loading / draining, grey scaled to zero, red
unhealthy / down / failed). Click it for a panel with each pool's workers
(state, last seen, running and queued jobs), queue depth, the build it runs
(short sha and channel; flagged when its workers run different builds) and
models. Model
and director pages show the endpoint's pool next to Run / Start session and,
when the pool is loading, scaled to zero, draining or down, warn first: the
second click submits. The strip polls `GET /fv/v1/status` every 7 s while
the tab is visible (doubling up to 60 s on errors, paused while hidden).

`GET /fv/v1/status` is public in every auth mode and carries labels, states,
ages and counts only (no worker URLs, pod or endpoint ids, IPs, tokens or
probe errors; workers are `w1`, `w2`, … per pool). Single server: one
`local` pool from the engine (`loading` with `{done, total}`, `ready`,
`busy`, `draining`, `failed`). (The edge answers its own status from the
family queues, docs/serve/edge-control-plane.md; the gateway's per-pool
status went with the gateway.) A model that failed (its load, or the startup capability check:
a GPU that cannot run it) is `failed` with a `reason`, e.g. "model
`h3-turbo` cannot run on this GPU (NVIDIA A100 80GB, sm80): it needs FP8
tensor cores (Ada, Hopper or Blackwell: sm89 or newer)"; its pool lists it
in `failed_models`. Shape (see `crates/fastvideo-serve/src/status.rs`):

```json
{"object": "fv.status", "gateway": false, "state": "ready",
 "pools": [{"id": "h3", "kind": "pod", "state": "busy", "available": true,
            "models": ["h3-turbo"], "queued": 2, "running": 1, "last_seen_s": 1.4,
            "workers": [{"label": "w1", "state": "busy", "last_seen_s": 1.4,
                         "running": 1, "queued": 0, "sessions": 0}]}],
 "models": {"h3-turbo": {"state": "busy", "pools": ["h3"]}},
 "names": {"h3-turbo": "h3-turbo"}}
```

## 1. The admin token

Admin calls (`/fv/v1/admin/*`) need the admin token as
`Authorization: Bearer <token>`.

- Set it with `FV_ADMIN_TOKEN` (or `auth.admin_token` in the TOML). Use a
  long random value, e.g. `echo "fvadm_$(openssl rand -base64 32 | tr '+/' '-_' | tr -d =)"`.
- If it is unset, fv-serve makes one on its first start (`fvadm_` + 32
  bytes from the CSPRNG, base64url) and keeps it in
  **`<state_dir>/admin_token`** (mode 600). Later starts with the same state
  dir reuse it. The log names the file and shows only the token's first 4
  characters:

  ```
  INFO admin token: generated and stored (read it on the server; FV_ADMIN_TOKEN overrides) file=/workspace/fv-state/admin_token starts_with=fvad created=true
  ```

  Read it on the server (`cat <state_dir>/admin_token`). On a Runpod pod
  without a volume the state dir is container disk: the token survives a
  restart, not a re-creation. A remote operator can have it sealed to an
  X25519 key instead (`FV_ADMIN_TOKEN_RECIPIENT`,
  `GET /fv/v1/admin/token/sealed`).

The server keeps only the token's SHA-256 digest and compares digests in
constant time.

## 2. API keys

Minted keys look like `fv_` + 43 base64url characters. The key is shown once,
in the mint response; the store keeps its SHA-256 digest, a display prefix
(`fv_AbCdEf…`), a name and `created_at` / `last_used_at` / `revoked_at`.

A key works for **every** API and scheme the server authenticates: fal
`Authorization: Key <key>`, and `Authorization: Bearer <key>` for MiniMax, LTX,
the native `/fv/v1/*` API and the open FastVideo APIs (where it identifies the
owner). Jobs are owned by the key's id (`key_<first 12 hex of the digest>`),
the same form static `FV_API_KEYS` keys get. `FV_API_KEYS` (a list of SHA-256
hashes) keeps working alongside minted keys.

Where minted keys are stored (`FV_KEY_STORE` / `auth.key_store`):

| Value | Store |
|---|---|
| `auto` (default) | `d1` when the D1 settings are present (and fv-serve is built with `http-client`), else `file` |
| `d1` | Cloudflare D1 table `api_keys` (job-store migration 2), shared by every worker on the database |
| `file` | `<state_dir>/api_keys.json`, mode 0600, rewritten atomically |
| `memory` | This process only |

Lookups hit an in-memory cache, so auth adds no I/O. `last_used_at` changes
in memory at most once a minute per key and is written back every 30 s (and at
shutdown). With D1, each worker reloads the table every 30 s, so a key minted
or revoked on one worker applies on the others within 30 s; on the worker
that handled the call it applies at once.

### Admin API (native)

```bash
ADMIN=fvadm_…
# Mint (201; the only time the key is returned)
curl -s -X POST "$BASE/fv/v1/admin/keys" -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' -d '{"name":"laptop"}'
# → {"api_key":"fv_…","key":{"id":"key_…","name":"laptop","prefix":"fv_AbCdEf…","created_at":"…","last_used_at":null,"revoked_at":null,"revoked":false}}

# List (never includes keys or digests)
curl -s "$BASE/fv/v1/admin/keys" -H "Authorization: Bearer $ADMIN"
# → {"keys":[…],"backend":"file"}

# Revoke (idempotent; 404 for an unknown id)
curl -s -X DELETE "$BASE/fv/v1/admin/keys/key_…" -H "Authorization: Bearer $ADMIN"
```

Names are 1-64 characters without control characters. Without a valid admin
token every call answers `401 {"error":{"kind":"unauthorized",…}}`.

## 3. Model pages

The form is built from the endpoint's input JSON Schema,
`GET /fal/schema/{owner}/{alias}/{sub}` (catalog: `GET /fal/schema`). The
schema is generated from the same limits the fal adapter validates with
(`fastvideo-fal::catalog`, checked by tests against `FalInput::parse`), so
the form cannot drift from the API. Required fields and the main settings
come first; `seed`, `enable_safety_checker`, `sync_mode`,
`prompt_expansion_mode` and `target_audio_url` sit under "Additional
settings".

- **Attachments**: `image_url` / `end_image_url` (image-to-video),
  `reference_{image,video,audio}_urls` (reference-to-video) and
  `target_audio_url` take drag-and-drop or a file picker. Files upload
  through `POST /storage/upload/initiate` then `PUT` to the returned
  `upload_url`; the returned `file_url` goes into the input (the server reads
  its own uploads locally). A pasted `https://` URL or `data:` URI also works.
- **Run** submits to the queue (`POST /{app}/{task}`), polls
  `GET /{app}/requests/{id}/status?logs=1`, then fetches the result. The
  result pane shows the video, logs, timings (inference from the output;
  queue wait and total measured in the browser) and the output JSON; Cancel
  sends `PUT …/cancel`.
- **Requests**: the last 50 requests are kept in this browser; click one to
  reload its inputs and result.
- **Tier, quality, recipe**: the result's `x-fv-tier`, `x-fv-quality` and
  `x-fv-recipe` headers (`crates/fastvideo-fal/src/queue.rs`) are shown with
  the facts and kept in the history. A tier that does not pass the quality
  gate (`x-fv-quality: draft`) gets a "Draft quality" banner above the
  video and a pill in the history.
- **Reference limits**: on reference to video, `x-fv-min-references` /
  `x-fv-max-references` of the schema bound the image, video and audio
  lists together: a counter shows the total, the drop zones close at the
  maximum, and Run refuses fewer than the minimum.
- **API tab**: cURL (queue submit / status / result, and `/run` sync),
  Python `fal_client` (`FAL_RUN_HOST` / `FAL_QUEUE_RUN_HOST`; https only) and
  JavaScript `@fal-ai/client` (`requestMiddleware` rewriting fal's hosts to
  this server) for the current inputs; then the same request on every
  other API the server mounts (`protocols` of the capabilities) that can
  run the endpoint's model: native `/fv/v1/jobs`, OpenAI `/v1/videos`
  (text / image to video), MiniMax `/v2/video_generation` (H3 tiers),
  the LTX API `/v2/{endpoint}` (LTX tiers) and the Reactor runtime (stream
  models). Each body uses only the fields that API's parser takes; fal
  inputs with no counterpart are listed in a comment. A server that does
  not report `protocols` (older, or the gateway) gets native and OpenAI.

## 4. Director page

The director page implements the client side of design §5.6 in one module,
`console/director.js` (`DirectorClient`): `POST /wma/ice`, a client-created
`control` data channel with recv-only video and audio, a non-trickle offer to
`POST /wma/session`, heartbeats every 5 s, `configure` then versioned
`prompt` messages, `stop`. The server side is the fal director (WP-14,
`fastvideo-fal::director`), mounted when fv-serve is built with `webrtc`
and `protocols.fal_director` is on. Otherwise the signalling routes answer
404/405/501 and the page shows "Streaming is not available on this
server".

The form follows what the server advertises:

- **Schema** (`GET /fal/schema/{app}/director`): the resolutions (labelled
  with their cost next to 768p) and aspect ratios; nothing is assumed
  before it loads, and without it no resolution or aspect is sent (the
  session's defaults). Any other property it lists (an enum or a bounded
  integer) gets a control and is sent in `configure` under its name: the
  clip director's **chunk size** `chunk_duration` (5 s / 10 s, narrowed per
  resolution by `x-fv-options-by-resolution`; the `configured` echo is the
  length the session runs). `x-fv-director-mode: "causal"` (and the
  catalog's `director_mode`) switch to **causal mode**: text only (no image,
  end image or audio fields), the 480p / 16:9 form, the model's note.
  `x-fv-licence` (and the catalog's `licence`) shows a licence banner: the
  LongLive-1.3B weights are non-commercial.
- **Configure** (messages.rs): opening prompt, resolution, aspect ratio,
  seed, memory, first-frame image, last-frame image, driving audio, audio
  bitrate, and a **script** of beats (`offset` in whole seconds, a prompt, a
  keyframe end image and / or audio per beat). Images and audio take a URL
  or a file uploaded through the fal storage API.
- **session_info**, sent when the control channel opens, is checked before
  `configure` goes out (it waits up to 10 s for it; without one the configure
  goes out unchecked and the event log says so): a driving audio the session does not
  take (`audio_conditioning: false`), a resolution or aspect it does not
  serve, too many beats or keyframes are refused in the page. Its facts
  (fps, chunk, scripts, audio conditioning, causal block size and prompt
  switch) are under "Session info", and it hides the prompt-time fields the
  session does not take.
- **Direct**: the next prompt with replan, an end image, audio with its
  behaviour (`replace` / `queue`), or a script with `script_mode` (`replace`
  / `append`); a script is sent on its own (the server refuses it with a
  prompt or end image). The timeline shows pending / applied / rejected.
- **Stream**: the session's model, tier and recipe (`x-fv-model`,
  `x-fv-tier`, `x-fv-recipe` of `/wma/session`, draft flagged), the chunk
  length, chunks, buffer, generation time and (causal) the KV re-cache time.

## 4b. Script avatar page

`/console/avatar` is a Reactor client in one module, `console/avatar.js`,
for a server whose Reactor runtime runs in avatar mode (`[reactor] mode =
"avatar"`, design §5.7): `POST /start_session` (a running session is
joined), `ice_servers`, `POST connections`, recv-only video and audio with
client-created `data` and `control` channels, a non-trickle offer with the
track mapping, the answer polled; the v0 JSON wire, pings every 5 s,
`resume_track` for both tracks. **Start take** uploads the photo (and the
voice file) through `POST /sessions/{sid}/uploads` + `PUT`, sends the
setters and `start`; Pause / Resume / Stop / Reset send those commands and
End session calls `/stop_session`. The status line shows the window, the
seconds sent, the latency to the first frame and the stalls; the table
lists every window's build time and real-time factor. Every server message
is kept in `window.__avatar` for `tests/console/avatar.cjs` (fake engine in
`tests/console/run.sh`; live on a GPU pod through
`scripts/serve/e2e/pod-clients.sh avatar`, which also records the stream
and runs the lip-sync proxy).

## 4b. Live input page

`/console/live` (`console/live.js`) publishes the browser's camera and
microphone to a **duplex** model (design §5.11): the loopback echo
`fv-echo` (`FV_ECHO_MODEL=1`) today, real-time V2V and live avatars later.
The model list is every model of `GET /fv/v1/capabilities` with
`stream.duplex`; the facts line shows its input caps (codecs, maximum size
and fps, bitrate cap, audio) and session length.

- **Camera / Microphone** checkboxes and a **Camera resolution** choice
  (640×360, 1280×720, 320×240) go to `getUserMedia` (`frameRate.max` is the
  model's `max_fps`); the model scales whatever arrives to its input size
  and refuses pictures above its maximum.
- **Scene** and **Persona** are the session context; **Session length** is
  `max_seconds` (the causal rule: default 120 s, at most 300 s).
- **Transport**:
  - *Native WHIP ingest*: one send-receive video and audio transceiver
    each, a complete offer `POST`ed to `/fv/v1/streams/ingest?model=…`
    (`application/sdp`, `Authorization: Bearer <key>`), the answer from the
    201; the output comes back on the same peer. Stats poll the stream's
    `Location` every second; Stop sends `DELETE`.
  - *Reactor runtime* (the server's `[reactor] model` must be the duplex
    model, e.g. `FV_REACTOR_MODEL=fv-echo`): `/start_session` with the
    context, `connections`, recv-only `main_video`/`main_audio` and
    send-only `input_video`/`input_audio` transceivers with
    `track_mapping`, the `data` and `control` channels, `publish_track` and
    `resume_track` (v0 JSON), `get_state` every second for the stats, and
    `/stop_session` on Stop.
- **Pause model** sends `set_paused`. The page shows the output, your
  camera, the counters (frames out, input frames shown, decoded, dropped,
  refused, latency in the model queue) and an event log.

Duplex sessions need the API key on both transports (unless the server runs
with `FV_AUTH_MODE=none`); the page shows the usual banner without one.

**Busy engine.** A session that was just stopped (on this page or another)
releases its executor a moment later; until then the engine answers a new
session 409 (`Retry-After`), 429 or 503. The director, Live stream and
native stream starts retry for up to 20 s, showing "The engine is still
busy … retrying", instead of failing.

## 4c. Live stream page

`/console/stream` (`console/stream.js`, WebRTC helpers in `console/rtc.js`)
runs a **causal** model (`stream.causal` in `GET /fv/v1/capabilities`:
SF-Wan, LongLive) as one continuous rollout steered by prompt switches. The
facts line shows its block size and fps, canvas, recipe and the
session-length rule (`stream_limits`: 120 s by default, at most 300 s; a
reset restarts the clock); a fal app whose director runs the model in
causal mode is linked, and a licence the server advertises for it (the
catalog app's `licence`) is shown as a banner.

- **Transport**:
  - *Reactor runtime* (plays in the page), offered when the runtime streams
    this model (`GET /schema` `info.title`; `[reactor] model`, e.g.
    `FV_REACTOR_MODEL=fake-sfwan`): `/start_session` (`seed`,
    `max_seconds`), recv-only transceivers per output track, the `data` and
    `control` channels, `set_prompt` with the opening prompt, `get_state`
    every second, `/stop_session` on Stop.
  - *Native stream*: `POST /fv/v1/streams` (`model`, `whip_url`,
    `whip_token`, `whip_target`, `prompt`, `seed`, `max_seconds`) publishes
    H.264 to a WHIP endpoint (a relay: MediaMTX, Cloudflare Stream); the
    page plays the relay's **WHEP** URL when given. It polls
    `GET /fv/v1/streams/{id}` (state, output, pacer, TTFF, the session) and
    Stop sends `DELETE`. Offered when the build can publish
    (`protocols.streams`).
- **Direct**: Switch prompt (`set_prompt`, applied at the next block
  boundary; the timeline marks a switch applied when the session reports
  its prompt), Pause / Resume (`set_paused`), Reset (`reset`).

## 4d. Native API page

`/console/native` (`console/native.js`) submits `POST /fv/v1/jobs` with
the fields only the native API takes (NativeBody,
`crates/fastvideo-serve/src/native.rs`), on any served model, tier alias
(`tiers` of the capabilities, e.g. `ltx-draft` where a worker binds it:
tiers with no fal endpoint are usable here) or alias. The form follows the
model's caps: its tasks (text, image, keyframes, reference, audio to
video, retake, extend), canvas tiers (`aspect_ratio` + `short_edge`, or
`size`), frame range and fps (`seconds` or `num_frames`), reference limit,
and the knobs it honours (`negative_prompt`, `seed`, `steps`, `guidance`,
`reference_strength` / `reference_lora_strength`); retake (`video_url`,
`start_s`, `end_s`, `retake_mode`, optional `audio_url`) and extend
(`extend_s`, `extend_at`, `context_s`). The result shows the job's tier
(draft flagged), recipe, canvas, frames, seed and metrics.

`flow_shift` and `guidance_scale_2` are not native fields: with the
**OpenAI `/v1/videos`** API chosen, they appear on models whose knobs honour
them, and the job is submitted there. No API takes an `audio_out` or
`callback` field (audio follows the model; MiniMax's `callback_url` is in
its snippet on the model page).

## 5. Calling a server behind the Runpod proxy

These notes apply to the console's API snippets and to your own scripts
when fv-serve runs on a Runpod pod or load-balancer endpoint
(`https://<pod>-8000.proxy.runpod.net`, `https://<endpoint>.api.runpod.ai`).

- **Set a User-Agent.** Runpod's proxy sits behind Cloudflare, which
  answers **403 `error code: 1010`** to Python-urllib's default
  `User-Agent` (`Python-urllib/3.x`; seen on the pod proxy in the WP-18
  E2E, docs/serve/e2e/h3-max.md; treat the load-balancer URL the same
  way). The request never reaches fv-serve, so its logs show nothing. `requests`, `httpx`, `fal-client`, `openai`,
  `@fal-ai/client` and `curl` send their own agent and pass. With
  `urllib.request`, set one yourself:

  ```python
  import json, urllib.request
  req = urllib.request.Request(
      f"{base}/fv/v1/jobs",
      data=json.dumps({"model": "h3-turbo", "prompt": "a fox"}).encode(),
      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "User-Agent": "my-client/1.0"},
  )
  print(json.load(urllib.request.urlopen(req)))
  ```

- **~100 s per request.** The proxy closes a request after about 100 s.
  Long generations belong on the queue/async routes (fal queue,
  `/v1/videos` + poll, MiniMax, LTX v2, `/fv/v1/jobs`), not the sync ones.
- **Cross-origin pages.** A page on another origin (or the console opened
  on a different host than `public_base_url`) may call the API and upload
  files: CORS allows any origin by default, preflights included
  (`server.cors_origins` / `FV_CORS_ORIGINS` narrows it; design §9).

## 5a. Embedded console

Another server may serve these pages unchanged under a path prefix for one
backend: fv-control does, for a Runpod serverless endpoint at
`/serverless/<endpoint id>/console` (docs/control/serverless.md §5b), and
answers the API calls under the same prefix. It adds `<meta>` tags to each
page's `<head>`, which `common.js` reads (fv-serve itself sends none, so
nothing changes on a pod):

| tag | effect |
|---|---|
| `<meta name="fv-console-base" content="/serverless/<id>">` | the API base is `origin + prefix` (the Server URL field is fixed, `fv.base` is ignored); console links (`page()`, `modelHref`, the top bar) go under the prefix; the model page reads its endpoint after the prefix; the request history is kept per prefix (`fv.history:<prefix>`) |
| `<meta name="fv-console-off" content="stream,live,avatar,director,admin">` | those pages lose their links (`pageOn()`: top bar, the home page's Live column, the director task) |
| `<meta name="fv-console-note" content="…">` | a banner under the top bar |

The embedding server also rewrites the pages' static `href="/console…"` /
`src="/console…"` attributes under the prefix (the CSP's `base-uri 'none'`
rules out a `<base>` tag). fv-control bundles a copy of these files: after
changing one, run `node gen-configs.mjs` in `control/` (its unit tests fail
while the copy is stale).

## 6. Deployments page (removed)

The Deployments page (`/console/deployments`) was a gateway page and went
with the gateway (2026-10-06, [edge-control-plane.md](edge-control-plane.md)
§9). Release channels, promote and rollback are in fv-control's Releases
page and `scripts/serve/release.sh` ([releases.md](releases.md)).

## 7. Experimental features

The admin page's **Experimental features** section lists the server's
feature flags with a switch each (a confirm dialog first). The Deployments
page links to it. Every flag is off by default.

| Flag | Off (default) | On |
|---|---|---|
| `h3_1080p_long` | H3 native 1080P clips up to **5 s**; a longer 1080P request is a 4xx naming the 5 s limit and this flag | H3 1080P up to **10 s** (longer is still refused) |

Owner decision (2026-09-29): H3 native 1080P is 5 s at most by default;
10 s is experimental (about 47 GiB of working memory at 1080P,
docs/serve/h3-1080p-and-upscaler.md).

```bash
curl -s "$BASE/fv/v1/admin/flags" -H "Authorization: Bearer $ADMIN"
# → {"object":"fv.feature_flags","backend":"d1","flags":[{"name":"h3_1080p_long","enabled":false,"default":false,
#     "experimental":true,"description":"…","updated_at":null,"updated_by":null}]}
curl -s -X PUT "$BASE/fv/v1/admin/flags/h3_1080p_long" -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' -d '{"enabled": true}'
```

`PUT` answers the new list; an unknown flag is 404, a body without a boolean
`enabled` 400, a missing or wrong admin token 401.

**How a flag takes effect** (`crates/fastvideo-serve/src/flags.rs`):

- **Storage.** D1 table `feature_flags (name, enabled, updated_at,
  updated_by)`, created on first use (like the release registry tables),
  when the server has the D1 job store; otherwise
  `<state_dir>/feature_flags.json`. Every process caches the flags; a
  `PUT` applies at once on the process that took it, the others re-read D1
  every 30 s (the admin list re-reads it on every call).
- **Enforcement.** Flags act on the model caps
  (`fastvideo_protocol::apply_feature_flags`) through the engine gate every
  API negotiates against, so negotiation refuses what a flag does not
  allow, before a job exists or reaches a worker. The same
  caps feed `/fal/schema/…` (the console's forms) and
  `/fv/v1/capabilities`. Nothing is added to the dispatch: workers run
  jobs that were already negotiated. The fal director (whose sessions run
  on a worker) caps 1080p chunks from the same caps, with the worker
  reading the same D1 table.
- **What `h3_1080p_long` changes.** The H3 1080P tier's `canvas.hd`
  carries `max_frames` (124 = 5 s; 243 = 10 s with the flag) and, while
  the flag is off, `experimental_max_frames` (243) for the refusal message.
  The refusal (fal 422, native / `/v1/videos` 400, on `duration` or
  `num_frames`):

  > 1080P clips are limited to 5 s (at most 124 frames at 24 fps) on model
  > `h3-turbo`; this request asks for 243 frames (10.12 s). Longer 1080P
  > clips, up to 10 s, are an experimental feature (`h3_1080p_long`) that
  > is off on this server; an admin can enable it under Experimental
  > features in the console

  The fal form's `duration` gets `x-fv-max-by-resolution: {"1080P": 5}`
  (10 with the flag): the console narrows the duration slider when 1080P
  is chosen and clamps a longer value. Director sessions configured at
  1080p use 5 s chunks (10 s with the flag).

## 8. Tests

- Rust: `fastvideo-serve-kit` `keys` (mint / check / revoke, file
  persistence and digest-only storage, D1 over the SQLite mock shared by two
  workers, rate-limited `last_used_at`, admin API needs the token) and `auth`
  (minted keys on every API); `fastvideo-fal` `catalog` (schema agrees with
  validation); `fastvideo-serve` `tests/console.rs` (full router: admin
  token, keys on fal/native/MiniMax, revocation, restart persistence, pages
  and content types, `/fal/schema`, `auth.mode` in capabilities, the keyless
  flow under `FV_AUTH_MODE=none`, `/fv/v1/status` ready / busy / draining,
  the `protocols` of capabilities),
  `tests/flags.rs` (the `h3_1080p_long` flag through the admin API; 1080P
  over 5 s refused on fal, native and `/v1/videos` with the flag off and
  accepted up to 10 s with it on; the fal form's duration cap and
  `/fv/v1/capabilities` follow it; persistence across a restart and
  through D1; a fake engine on a simulated A100 failing its FP8 models with
  the reason in `/fv/v1/status`), `tests/dimension_sweep.rs` (the sweep
  runs with the flag off and on),
  `tests/direct_workers.rs` (a direct worker's keys and admin token),
  `admin_token` unit tests
  (stored once, reused, overridden, sealing) and `console` unit tests (every asset
  referenced is embedded; no inline scripts).
- Browser: `bash tests/console/run.sh` builds `fv-serve --features
  fake,encoders`, starts it with no config file and without
  `FV_ADMIN_TOKEN`, reads the token from `<state dir>/admin_token` (and
  checks the file is mode 600 and the token is not in the log) and drives
  headless Chromium through minting a key, the Experimental features
  section (with `FV_FAKE_H3_1080P=1`: `h3_1080p_long` off, the h3-turbo
  form's duration capped at 5 s at 1080P and 15 s at 768P, 10 s after
  enabling the flag, 5 s again after disabling it), text-to-video, image-to-video
  with an uploaded image, the API tab, history, a live director session
  (start, 1344x768 video with one video and one audio track playing, a
  second prompt applied, stop; the encoder is `auto`, i.e. OpenH264 on a
  machine without NVENC), a 390 px layout, the Deployments page (404 on a
  standalone server, then the admin API mocked with `page.route`: channels,
  drift, mixed versions, Promote's dry run → confirm → dispatch, a
  cancelled Rollback sending only its dry run) and revocation. `FV_SERVE_UI=1 bash
  scripts/serve/check.sh` runs it; `FV_CONSOLE_SHOTS=<dir>` saves
  screenshots. It needs `node`, the `playwright` npm package and a Chromium
  under `PLAYWRIGHT_BROWSERS_PATH` (default `/opt/pw-browsers`).
  After it, `tests/console/director_playback.cjs` (the director page with a
  fake engine slower than real time) and `tests/console/live_echo.cjs`: a
  server with `FV_ECHO_MODEL=1`, `FV_REACTOR_MODEL=fv-echo` and an API key;
  Chromium with `--use-fake-device-for-media-stream` publishes its fake
  camera and microphone from the Live input page over WHIP ingest, then
  over the Reactor runtime, and checks the page reaches streaming, the
  model shows input frames, the `<video>` shows the magenta overlay border
  around the camera's picture (centre colours match), the overlay counter
  advances, and Stop ends it; also that both transports refuse a request
  without the key. Then `tests/console/ui_gaps.cjs` (also in the CI
  `console` suite): served models with tier and recipe, the mounted APIs,
  no H3 tabs on an unserved app; a result's tier and recipe and a draft
  result flagged (`x-fv-quality` added in the browser); the reference limits
  and the per-API snippets; the clip director (schema resolutions, the
  chunk size narrowed per resolution (`FV_FAKE_H3_1080P=1`: 5 s only at
  1080p), sent in `configure` and echoed by `configured`; model, tier and recipe from the session headers;
  driving audio refused from `session_info`; a script-only prompt); the
  causal director (`fastvideo/fake-sfwan`: text-only 480p form, chunks, a
  prompt applied, a mocked licence banner); Live stream over the Reactor
  runtime (`FV_REACTOR_MODEL=fake-sfwan`: 832x480 video, a switch applied,
  stop) and the native `/fv/v1/streams` transport to a mock WHIP endpoint
  (create, `set_prompt` through `/commands`, stats, Stop → `DELETE`); the
  Native API page (tier aliases, a job with steps and guidance, retake
  fields, `flow_shift` through `/v1/videos`).
  `FV_CONSOLE_TESTS=live_echo bash tests/console/run.sh` runs one of them.

With the fake engine and no ffmpeg the "video" is a small placeholder file,
so the player shows the URL but cannot play it; with ffmpeg on `PATH` the fake
engine writes a real MP4.
