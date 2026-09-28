# fv-serve console

Browser pages served by `fv-serve` itself for minting API keys and trying the
fal-compatible endpoints, modelled on fal.ai model pages (Playground + API
tabs). They are static files embedded in the binary
(`crates/fastvideo-serve/console/`, `include_str!`): no build step, no CDN,
scripts from the server's own origin only (strict CSP). Everything they do
goes through the public APIs, so anything the console does can also be done
with `curl`.

| Page | What it does |
|---|---|
| `/console` | Server URL (defaults to the page's origin) and API key (kept in `localStorage`), key check via `GET /fv/v1/capabilities`, list of mounted fal apps and endpoints |
| `/console/admin` | Admin token (kept in `sessionStorage`, this tab only); create, list and revoke API keys |
| `/console/models/{owner}/{alias}/{task}` | One endpoint, e.g. `minimax/h3-max/reference-to-video`: variant switcher (`h3-max`, `h3-turbo`, `h3-draft`), task tabs, Playground and API tabs |
| `/console/models/{owner}/{alias}/director` | Live director (WebRTC) page |

Disable the pages with `FV_CONSOLE=0` (or `server.console = false`); the
APIs stay up.

## 1. The admin token

Admin calls (`/fv/v1/admin/*`) need the admin token as
`Authorization: Bearer <token>`.

- Set it with `FV_ADMIN_TOKEN` (or `auth.admin_token` in the TOML). Use a
  long random value, e.g. `echo "fvadm_$(openssl rand -base64 32 | tr '+/' '-_' | tr -d =)"`.
- If it is unset, fv-serve generates one at startup (`fvadm_` + 32 bytes from
  the CSPRNG, base64url) and logs it **once** at `WARN` in a banner:

  ```
  ==============================================================================
    fv-serve admin token (generated at startup; set FV_ADMIN_TOKEN to choose one):

        fvadm_…

    Mint API keys at /console/admin or POST /fv/v1/admin/keys with
    `Authorization: Bearer <admin token>`. It is not stored and not shown again.
  ==============================================================================
  ```

  A generated token changes on every restart. On Runpod/Vast, read it from the
  worker log, or set `FV_ADMIN_TOKEN` as a secret.

The server keeps only the token's SHA-256 digest and compares digests in
constant time. The token is never written to disk.

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
- **API tab**: cURL (queue submit / status / result, and `/run` sync),
  Python `fal_client` (`FAL_RUN_HOST` / `FAL_QUEUE_RUN_HOST`; https only) and
  JavaScript `@fal-ai/client` (`requestMiddleware` rewriting fal's hosts to
  this server) for the current inputs.

## 4. Director page

The director page (opening prompt, resolution, aspect ratio, seed, memory,
start image, Start/Stop, next-prompt box with replan, a prompt timeline with
pending/applied/rejected states, the WebRTC video element and an event log)
implements the client side of design §5.6 in one module,
`console/director.js` (`DirectorClient`): `POST /wma/ice`, a client-created
`control` data channel with recv-only video and audio, a non-trickle offer to
`POST /wma/session`, heartbeats every 5 s, `configure` then versioned
`prompt` messages, `stop`. The server side is the fal director (WP-14,
`fastvideo-fal::director`), mounted when fv-serve is built with `webrtc`
and `protocols.fal_director` is on. Otherwise the signalling routes answer
404/405/501 and the page shows "Streaming is not available on this server
yet".

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

## 6. Tests

- Rust: `fastvideo-serve-kit` `keys` (mint / check / revoke, file
  persistence and digest-only storage, D1 over the SQLite mock shared by two
  workers, rate-limited `last_used_at`, admin API needs the token) and `auth`
  (minted keys on every API); `fastvideo-fal` `catalog` (schema agrees with
  validation); `fastvideo-serve` `tests/console.rs` (full router: admin
  token, keys on fal/native/MiniMax, revocation, restart persistence, pages
  and content types, `/fal/schema`) and `console` unit tests (every asset
  referenced is embedded; no inline scripts).
- Browser: `bash tests/console/run.sh` builds `fv-serve --features
  fake,encoders`, starts it with no config file and without
  `FV_ADMIN_TOKEN`, reads the generated token from the log and drives
  headless Chromium through minting a key, text-to-video, image-to-video
  with an uploaded image, the API tab, history, a live director session
  (start, 1344x768 video with one video and one audio track playing, a
  second prompt applied, stop; the encoder is `auto`, i.e. OpenH264 on a
  machine without NVENC), a 390 px layout and revocation. `FV_SERVE_UI=1 bash
  scripts/serve/check.sh` runs it; `FV_CONSOLE_SHOTS=<dir>` saves
  screenshots. It needs `node`, the `playwright` npm package and a Chromium
  under `PLAYWRIGHT_BROWSERS_PATH` (default `/opt/pw-browsers`).

With the fake engine and no ffmpeg the "video" is a small placeholder file,
so the player shows the URL but cannot play it; with ffmpeg on `PATH` the fake
engine writes a real MP4.
