# fastvideo-fal

The fv-serve adapter for fal's queue and sync API (WP-09), and later the WMA
director (WP-14, `src/director/`). The design is in
[docs/serve/design.md](../../docs/serve/design.md) §4.4, and the wire contract
is in [docs/serve/research-fal.md](../../docs/serve/research-fal.md).

## Routes

Apps come from `FalConfig::apps`. The defaults are `minimax/h3-max`,
`minimax/h3-turbo` and `minimax/h3-draft`, which map to the H3 `Max`,
`Turbo` and `Draft` tiers. Routes are static, never wildcards. In the table,
`{sub}` is one of `text-to-video`, `image-to-video` or `reference-to-video`.

| Route | Answer |
|---|---|
| `POST /{app}/{sub}` | Queue submit. HTTP 200 with `{request_id, response_url, status_url, cancel_url, queue_position}` and app-only URLs. Accepts `?fal_webhook=` and `?fal_max_queue_length=` (429 when more requests than that are waiting). Sets `x-fal-request-id`. |
| `GET /{app}[/{sub}]/requests/{id}[/response]` | The bare output JSON. Still running: 400 `{"detail":"Request is still in progress"}`. Failed: the error's own status and body. Cancelled: 499 `client_cancelled`. |
| `GET …/requests/{id}/status?logs=1\|0\|true\|false` | `IN_QUEUE` always carries `queue_position`. `IN_PROGRESS` and `COMPLETED` always carry `logs`. A failed `COMPLETED` adds `error` and `error_type`. |
| `GET …/requests/{id}/status/stream` | SSE with one status object per change. It closes after `COMPLETED`. |
| `PUT …/requests/{id}/cancel` | 202 `CANCELLATION_REQUESTED`, 400 `ALREADY_COMPLETED` or 404 `NOT_FOUND` (see the note below). |
| `POST /run/{app}/{sub}` | Sync: the output on the same connection. Past `sync_timeout` the job is cancelled and the answer is 504. |
| `ANY /fal/proxy` | fal's proxy protocol. The request is routed by the host in `x-fal-target-url`: `queue.fal.run`, `fal.run` (goes to `/run`), `wma.fal.run` (goes to `/wma`), or `rest.fal.ai`. |
| `POST /storage/upload/initiate` | Returns `{upload_url, file_url}`. `upload_url` is serve-kit's `PUT /uploads/{token}`. |
| `GET /.well-known/jwks.json` | Our Ed25519 webhook key. |

Any unknown request id answers 404 `{"status":"NOT_FOUND"}`, and so does a
request id from another key or another app. A bad key answers 401
`{"detail":"invalid key credentials"}`. Tier and recipe metadata goes in the
`x-fv-tier` and `x-fv-recipe` response headers, and draft output also gets
`x-fv-quality: draft`. The fal output schema has no metadata field, so the
headers carry this instead.

**Cancel is 202, not 200.** The D-QUEUE table in the fal docs says 202
`{"status":"CANCELLATION_REQUESTED"}`, while the OpenAPI says
`200 {"success": bool}`. We follow the table because it is newer and more
specific, and every client accepts any 2xx. A queued job is cancelled at
once. A running job stops at its next denoise step.

## Wiring (fv-serve, WP-10)

```rust
let ctx = ServeCtx::builder(cfg, engine)
    .callbacks(Arc::new(CallbackSender::new(transport, Some(webhook_signer))))
    .renderer(ProtocolId::Fal, Arc::new(fastvideo_fal::FalWebhook::default()))
    .build().await?;
let app = fastvideo_fal::router(ctx.clone(), FalConfig::default()); // a Router<()>
```

Fal artifacts should be named with `fastvideo_fal::output_file_name(&job)`
(`<nanoid21>_<app>[-<tier>].mp4`, e.g. `…_minimax-h3-max.mp4`): hosted fal's
form, but named by the app and tier that made the file (hosted fal writes
`minimax-h3` for every H3 app).

## Tests

- `tests/queue_golden.rs` pins these fixtures in `tests/queue_golden/`:
  - every schema field, enum and default, plus the documented examples;
  - submit, status, result, error and webhook bodies.

  To regenerate them, run `FV_BLESS=1 cargo test -p fastvideo-fal --test queue_golden`.
- `tests/queue_http.rs` runs the router over HTTP. Most tests use a
  simulated engine. The main flow, cancel and MP4 checks use the real
  `EngineService` with `FakeBackend`. The MP4 check needs ffmpeg (on PATH,
  or set `FV_FFMPEG`) and compares the file against hosted H3: H.264, 24 fps,
  AAC-LC stereo 32 kHz, faststart.
- `tests/queue_compat.rs` runs the real clients against the fake engine:
  - Python `fal-client` 1.0.3 (set `FV_FAL_PYTHON`). The script stands up a
    local TLS endpoint, because the client is https-only.
  - `@fal-ai/client` 1.10.1 in `requestMiddleware` mode and in `proxyUrl`
    mode (set `FV_FAL_JS_DIR`).

  The module docs have the setup commands.

Not served: the JS multipart upload (`initiate-multipart`, for files over
90 MB), realtime tokens and HTTP `/stream`.
