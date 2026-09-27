# fastvideo-serve-kit

This crate holds the HTTP glue that every fv-serve API adapter shares. The
design is in [docs/serve/design.md](../../docs/serve/design.md) (WP-05).

| Module | What it provides |
|---|---|
| `ctx` | `ServeCtx` and its builder. `EngineGate` is the seam to the engine service: models, aliases, admission, submit, cancel. `SafetyFilter` is a moderation hook that accepts everything by default. |
| `auth` | Three modes: `none`, `keys` and `trust-gateway`. Each API has its own scheme: fal uses `Key`; MiniMax, LTX and native use `Bearer`; FastVideo, FastWan and Reactor are open (a valid key still sets the owner). Keys are configured as SHA-256 hashes (`FV_API_KEYS`). |
| `store` | `MemJobStore`: in memory, with optional JSON manifests for durability. On restart, unfinished jobs are marked failed. It also handles the expiry sweep and artifact and input cleanup. |
| `artifacts` | `LocalArtifactStore`, served at `GET /files/{id}/{name}?exp=&sig=` (HMAC-SHA256, CORS `*`, Range). `S3ArtifactStore` returns SigV4 presigned URLs for `runpod-queue`. |
| `uploads` | `UploadStore` and `PUT /uploads/{token}`. A token accepts one upload. These back LTX `/v1/upload` and fal storage initiate. |
| `ingest` | Stages media from HTTP(S) URLs, data URIs and upload ids. Includes the SSRF guard, per-API limits (`IngestPolicy::ltx()` and the default policy), MIME sniffing and a `Prober` hook. |
| `callback` | Sends MiniMax callbacks (challenge echo, then one POST per status change) and fal webhooks (Ed25519, fal header scheme, JWKS), with retries on a `RetrySchedule`. |
| `events` | `apply_event` applies engine events to a job, stores outputs as artifacts and fires callbacks. Also provides `cancel_job` and `wait_terminal`. |
| `sse` | Turns an `SseSpec` into an axum SSE response that follows job changes. |
| `handlers` | Converts `HttpReply` into an axum response. Provides the generic `submit`, `status` and `result` handlers, plus `submit_request` and `find_job`. |

Outbound HTTP (media fetch, callbacks and S3 upload) needs the `fetch`
feature. Without it, HTTP media refs are refused and callbacks are dropped.
