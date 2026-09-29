# Gateway on Cloudflare Workers + Durable Objects (workers-rs): assessment

Status: research and design, 2026-09-29. Nothing was deployed and no code
changed. The owner asked: "on queue handling, what if we use workers-rs to
implement the gateway and Durable Objects?" This document answers that
question against the gateway in [gateway.md](gateway.md)
(`crates/fastvideo-serve/src/gateway/`).

**Short answer.** Build a per-pool Durable Object (DO) **dispatcher** first,
and put the current fv-serve gateway in front of it as the API. Keep the API
adapters native until one wasm check passes (§6, phase 0). Do not use
a DO for serverless pools. After the other agent's fix (inline inputs and a
one-round-trip adopt), the DO wins about 0.5–1 s of queue time per job on
**pod** pools. It also removes the single CPU pod as the only place the queue
lives, and it pushes progress instead of polling D1. On serverless pools the
Runpod queue is still in the path, so the DO gains almost nothing there.
Moving every API to a Worker is possible, but it means splitting serve-kit
into a native part and a wasm part, about 5–8 agent-sessions. That is
worth doing only after phase 1 has proved the dispatcher.

## 1. Where the queue time goes today

Image-to-video queue time was measured at **3.9–5.2 s** (submit until the
worker starts):

| stage | today | after the fix in progress | with the DO dispatcher (pods) |
|---|---:|---:|---:|
| gateway R2 PUT of the input | 1–3 s | 0 (inline in the envelope) | 0 (inline); client presigned PUT for large files |
| worker R2 GET of the input | ~1 s | 0 (inline) | 0 (inline, ≤ 8 MiB) |
| D1 round trips (insert, dispatch row, adopt) | ~3 × 0.25 s ≈ 0.8 s | insert + adopt ≈ 0.5 s | 0 on the critical path (DO SQLite ~µs; D1 write-behind over the binding) |
| hop to the worker | Runpod pod proxy / `/run` | same | push over the worker's open WebSocket: one one-way trip, ~40–150 ms |
| **total, pod pool** | **3.9–5.2 s** | **≈ 0.8–1.5 s (estimate)** | **≈ 0.15–0.4 s (estimate)** |
| **total, serverless pool** | 3.9–5.2 s | ≈ 1–2 s + Runpod pickup | same as the fix: the Runpod queue is still in the path |

The last two columns are estimates from the measured parts. The per-stage
numbers are those given for the I2V run. The gateway's own submit round trip
was 0.7–1.2 s in [e2e/gateway.md](e2e/gateway.md).

- Most of the 3.9–5.2 s comes from the R2 transfers and the D1 round trips,
  and the fix already removes most of that. What the DO adds on top is
  **push** (no poll, no proxy hop) and **no D1 on the critical path**. Both
  cost about 0.5–1 s. Against a 5–40 s job (wan 4.3 s, h3 24 s, ltx 40 s
  run time), that is 2–15 % of the wall time.
- **Status latency** improves too, which the table does not show. The
  gateway answers SSE and sync waits by polling D1 every `watch_poll_ms =
  1000`. With the DO, the worker sends progress to it over the WebSocket, and
  the DO can fan it out at once. The Worker front then relays it to
  SSE clients, or D1 still carries it.
- The fv-serve gateway could get the same push without Cloudflare: workers
  would hold a WebSocket to the gateway, the queue would stay in memory,
  and D1 would be written behind. §5.4 compares the two. The DO is the
  better home because it is one writer per pool without leader election,
  it survives the loss of any one host, and it runs at the edge.

## 2. Platform facts (checked 2026-09-29)

### workers-rs

- Crate `worker` **0.8.7** (2026-09-25). Rust is a supported Workers language
  (not labelled experimental). RPC is still marked experimental.
- **Durable Objects**: `#[durable_object]` plus `impl DurableObject`, with
  `fetch`, `alarm`, `websocket_message`, `websocket_close` and
  `websocket_error`. `State` has `accept_web_socket`,
  `accept_websocket_with_tags`, `get_websockets(_with_tag)`, `get_tags`,
  `set_websocket_auto_response` and `block_concurrency_while`, so the
  **hibernation API is exposed**. `WebSocket` has `serialize_attachment` and
  `deserialize_attachment` (16 KiB per socket, which survives hibernation).
- **DO storage**: `storage().sql()` (SQLite-backed classes,
  `new_sqlite_classes` migration), KV `get`/`put`, `transaction`, and
  `set_alarm`/`get_alarm`/`delete_alarm`.
- **Bindings**: D1 (`d1` feature), R2, KV, Queues (`queue` feature), service
  bindings, Hyperdrive, rate limiting. `Fetch` handles outbound HTTP (the
  Runpod API, pod signalling). `#[event(scheduled)]` handles cron.
- **http/axum**: the `http` feature swaps in the `http` crate types, so
  an **axum `Router` runs inside a Worker**. Axum must be built without its
  `tokio`/`http1` server features.
- **Send**: JS-backed futures are `!Send`. axum and our `EngineGate`
  (`Send + Sync`, async-trait) need `Send`, so every JS call is wrapped in
  `worker::send::SendFuture` / `SendWrapper`. This is safe because the
  isolate is single-threaded.
- **Panics**: the default is `panic=abort`, which kills the wasm instance;
  it is reinitialised on the next request. `--panic-unwind` turns panics
  into JS exceptions. A DO must keep **no state that matters only in
  memory**.

### wasm32-unknown-unknown constraints

- There is no tokio runtime and there are no threads. `tokio::sync`
  compiles, but `tokio::{time, fs, net, process, spawn}` and `rayon` do
  not. `std::time::Instant::now()` panics; use `Date::now()`. Nothing can
  touch a filesystem.
- `uuid::new_v4` needs uuid's `js` feature. getrandom 0.3 needs the
  `wasm_js` backend (`--cfg getrandom_backend="wasm_js"`).
  `time::OffsetDateTime::now_utc` needs time's `wasm-bindgen` feature.
- C and C++ dependencies (`onig_sys`, `esaxx-rs`, `audiopus`,
  `openh264`) do not build.
- Limits: CPU per request is **30 s by default, up to 5 min**. Memory is
  **128 MB per isolate**. The global scope must start in **≤ 1 s**. The
  bundle may be **64 MiB uncompressed**; the old 3/10 MB compressed limit
  was removed on 2026-09-04. A Rust Worker is typically a few hundred KB
  to a few MB of wasm, and a cold start is a few ms.
- Subrequests: 10,000 per invocation on the paid plan, and **6 simultaneous
  open connections** per request. Cron, queue and alarm handlers can run
  for up to **15 min** of wall time.
- Outbound `fetch` goes **to host names, not IP addresses** (error 1003,
  "Direct IP access not allowed"). Some users report failures on
  non-standard ports.

### Durable Objects

- A SQLite DO holds up to **10 GB**. A row, string or BLOB is at most
  **2 MB**. The soft limit is **1,000 requests/s per object**.
- WebSocket messages the DO **receives** can be up to **32 MiB**.
- Hibernation: when idle, the DO is evicted while its clients stay
  connected. Ping frames get an automatic pong, which does not wake the DO,
  and no duration is billed while it hibernates. An *outgoing* WebSocket
  does not hibernate, so the GPU workers must dial **in** to the DO.
- **A deploy disconnects every WebSocket and restarts every DO.** Workers
  must reconnect and reconcile (§3.3).
- A DO lives where it is first created, or at a `locationHint` (`wnam`,
  `weur`, …). Put each pool's DO near its pods: US-CA-2 → `wnam`,
  EUR-IS-1 → `weur`.

### Bodies and uploads

- Request bodies may be **100 MB** on Free/Pro, 200 MB on Business and
  up to 5 GB on Enterprise. Response bodies have no limit (streamed).
  Headers are limited to 128 KB and URLs to 16 KB.
- Our ingest limits (`serve-kit/src/ingest.rs`) allow 64 MB data URIs and
  100 MB videos. **A 64 MB base64 body plus its decoded copy plus a
  `serde_json::Value` does not fit in 128 MB.** A Worker front must stream
  large bodies (R2 multipart), or cap inline JSON media at about 16 MB, or
  send clients to presigned uploads.
- **R2 presigned URLs**: GET, HEAD, PUT and DELETE, valid from 1 s to 7 days.
  They work only on the S3 host `<account>.r2.cloudflarestorage.com`, not
  on a custom domain. Browser uploads need CORS on the bucket. POST form
  uploads are not supported.

### Pricing (Workers Paid, $5/month base)

| item | included | beyond |
|---|---|---|
| Worker requests | 10 M/month | $0.30 / M |
| Worker CPU | 30 M ms/month | $0.02 / M ms |
| DO requests | 1 M/month | $0.15 / M. Incoming WS messages count at 20:1 |
| DO duration | 400 k GB-s/month | $12.50 / M GB-s (128 MB per DO; zero while hibernated) |
| DO SQLite | 25 B rows read, 50 M rows written, 5 GB-month | $0.001 / M read, $1.00 / M written, $0.20 / GB-month (billed since Jan 2026) |
| D1 | 25 B rows read, 50 M written, 5 GB | $0.001 / M, $1.00 / M, $0.75 / GB-month |
| R2 | none | $0.015 / GB-month, Class A $4.50 / M, Class B $0.36 / M, no egress fee |
| Queues | 1 M ops | $0.40 / M (per 64 KB) |

## 3. The architecture, piece by piece

```
client ─► Worker front (all APIs, auth, ingest)                     [phase 3]
            │   or: fv-serve gateway on its CPU pod as today        [phase 1]
            │  POST /enqueue (service binding / HTTPS + token)
            ▼
   PoolScheduler DO  (one per pool: "h3-turbo", "wan", "sfwan-live", …)
     SQLite: queue, attempts, worker registry, leases, metrics window
     alarm : reaper + autoscale tick (Runpod REST)
     D1    : write-behind of job rows (the durable, queryable record)
            ▲  ▲  ▲   outbound WebSocket per GPU worker (hibernatable server side)
            │  │  │   job push ▼ / ack, progress, done, heartbeat ▲
        pod workers (fv-serve role=worker)       media: client ◄──► worker (WebRTC)
```

### 3.1 Per-pool Durable Object as the scheduler

- Name the DO by the pool id (`idFromName("h3-turbo")`). That makes one
  single-threaded writer per pool, with input/output gates, so there is no
  lease row, no `gw_dispatch` race and no reaper election between replicas.
- SQLite tables mirror today's D1 tables, but hold the hot state: `queue
  (job_id, attempt, priority, envelope, state, worker, pushed_at,
  acked_at, heartbeat_at)`, `workers (worker_id, state, running,
  sessions, caps_json, connected)`, `leases` (sessions) and `metrics`
  (a finished-job window for `PoolMetrics`).
- Dispatch loop: on `/enqueue`, on each `websocket_message` (ack, done,
  or idle) and on each alarm, pick the least-loaded connected, non-draining
  worker, push `{"t":"job", envelope}`, and set an ack deadline alarm.
  If the ack does not arrive in time, requeue. The ack is **the adopt**:
  the worker takes the job on its own, with no D1 round trip on the
  critical path. It stays idempotent on `(job_id, attempt)`, with the same
  semantics as `D1JobStore::adopt`.
- Worker loss: the socket closes (`websocket_close`), or the heartbeat is
  older than `stale_after_s`, detected by an alarm. The job is then
  requeued while `attempt ≤ retries`, as §3 of gateway.md does, or failed.
- Capacity: pools see about 10³ jobs/h, far below 1,000 req/s per object.
  One DO per pool is enough. Nothing needs sharding.

### 3.2 GPU workers keep an outbound WebSocket to their pool's DO

- On the worker side, fv-serve `role = "worker"` gets `[gateway] dispatcher
  = "wss://…/pools/{pool}/connect"`, a reconnecting client
  (`tokio-tungstenite`), and the internal token in the upgrade. The worker
  announces `{worker_id, caps, running jobs, sessions, draining}`. That
  replaces self-registration in `gw_workers` (§5.3) and the caps probe (§4).
- The DO accepts with `accept_websocket_with_tags(ws, [worker_id])` and
  keeps the worker's identity in the socket attachment. An auto-response
  answers ping/pong, so an idle pool costs nothing.
- Progress: the worker sends ≤ 1 message/s per job. The DO updates SQLite,
  batches D1 writes, and forwards to any listeners. Pods no longer need a
  public HTTP port for dispatch; they still need one for ICE.
- **Serverless pools do not fit this model.** Runpod scales and reaps
  serverless workers from *its* queue. A job pushed around it is invisible
  to its scaler, and the worker would be idle-reaped. For those pools the
  DO calls Runpod `/run` as today (same envelope), and the gain is only the
  D1 round trips. That is why phase 1 targets pod pools, which the
  autoscaler already creates (`RunpodPods`).

### 3.3 Deploys and reconnects

Every deploy of the Worker script restarts the DOs and drops every
worker socket. The jobs keep running on the GPUs. On reconnect the worker
re-announces what it holds, and the DO reconciles: a held job stays
running, and a job the DO thought was pushed but the worker does not hold
is requeued. Buffered progress is flushed. This path has to be tested
explicitly (a deploy during a job), and `stale_after_s` has to exceed the
reconnect backoff.

### 3.4 D1 as the durable record, or DO storage only

Keep D1. The DO is the **authority for dispatch state**, and D1 is the
**record** the APIs read.

- Why not DO only: status, lists and results across pools and owners,
  `/fv/v1/admin/*`, the console, API keys (`KeyStore` on D1), and the
  existing `D1JobStore` and schema all query D1. Native fv-serve (the
  fallback, and single-host deployments) reaches D1 over HTTP but cannot
  reach a DO.
- The DO writes job rows through the **D1 binding**, a few ms from the
  edge, instead of the 250 ms HTTP API round trip that workers pay today.
  It writes behind the critical path, batched, at most 1/s per job, with
  the same columns as serve-kit's `jobs`. Workers stop writing D1
  themselves.
- Risk: D1 is single-primary. The pool DO and the D1 primary should sit in
  the same region.

### 3.5 R2 presigned uploads from clients

- This is independent of the move and can ship in the current gateway
  now. `S3ArtifactStore` already signs SigV4 URLs. fal `storage/upload/
  initiate`, `/uploads` and `/v1/upload` would answer a presigned R2
  `PUT` URL instead of a gateway disk token. That ends "uploads stay
  pinned" (gateway.md §6), and a gateway replica no longer carries the
  bytes.
- Presigned URLs use the S3 host only, not a custom domain, and they need
  bucket CORS for browsers. Our upload tokens therefore become URLs on
  another host, which fal and OpenAI clients already accept (they follow
  the returned URL).
- Worker path: small inputs (≤ ~8 MiB after decode) are inlined in the
  pushed envelope. Larger ones are presigned R2 `GET`s. The DO's 32 MiB
  limit applies to what it receives, so larger envelopes need a check
  before relying on it.

### 3.6 Autoscaling from DO alarms

- `fastvideo-autoscale::policy` is pure and deterministic ("time, signals,
  observation and balance in; one decision per pool out"). That is exactly
  what an alarm handler needs: read `PoolMetrics` from the DO's own SQLite,
  call the Runpod REST API (`Fetch`, key from a Worker secret), and set
  the next alarm (`interval_s`, 15 s).
- Leadership is free, because the pool DO is the only writer. The D1 lease
  (`fv_autoscale_lease`) stays for the global budget. Alternatively, a
  single "fleet" DO runs the cross-pool budget and balance-floor ($8) logic
  and the pool DOs report to it.
- Drain before delete becomes one message on the worker's socket, and
  the DO sees in-flight work directly, with no probe race.
- Alarm handlers get 15 min of wall time. A tick is a few Runpod calls, well
  inside that.
- Cost of a 15 s alarm: 172,800 DO requests/month per pool, inside the
  1 M included requests for up to ~5 pools. The DO does not hibernate while
  the tick runs, but a tick is milliseconds.

### 3.7 Streaming signalling

**Media stays on the GPU workers.** That covers the str0m peers (fal
director, Reactor) and the WHIP publisher (`fastvideo-webrtc`). A Worker
cannot terminate WebRTC. Signalling is HTTP and SSE, which a Worker handles
well:

- `/wma/session`, `/start_session` and `/fv/v1/streams`: the DO allocates a
  worker with a free session slot and records the lease in SQLite, replacing
  `gw_sessions`. The SDP offer and answer then go either:
  - over **Fetch** to the worker's public host name: the Runpod proxy
    `https://{pod}-{port}.proxy.runpod.net` or a DNS name, **never an IP
    address**, and preferably on 443; or
  - **relayed over the worker's WebSocket** (request id multiplexing). This
    is preferred: the pod needs no public HTTP port, only its ICE UDP/TCP
    port (70000/tcp today).
- The answer SDP still carries the worker's own public ICE candidates, so
  media flows client ↔ worker directly as today.
- Long SSE (`/start-session`, Reactor `/events`) streams through the Worker:
  wall time is unlimited while the client is connected, and CPU is billed
  only for bytes moved. Reactor's "lease by client address" fallback reads
  `CF-Connecting-IP`.
- Serverless WHIP streams (`kind:stream` Runpod job) are unchanged: the DO
  calls Runpod. Cloudflare Realtime (SFU) accepts WHIP and could be a WHIP
  target, but that is a separate decision.

## 4. Code reuse on wasm32-unknown-unknown

Method: `cargo tree --target wasm32-unknown-unknown -e normal` for each
crate. This reads metadata only and compiles nothing, as CLAUDE.md
requires. Blocking dependencies found:

| crate (src lines) | wasm blockers in its tree | cause | what it takes |
|---|---|---|---|
| `fastvideo-protocol` (3.8k) | `tokenizers` → `onig_sys` (C), `esaxx-rs` (C++), `rayon` | `fastvideo-models` dep, used only for `h3::config` (1.4k lines, no deps) in `caps.rs` and `negotiate.rs` | gate the text/tokenizer modules of `fastvideo-models` behind a default feature, or move `h3::config` into its own leaf crate. Plus uuid `js` and the getrandom `wasm_js` cfg on wasm. `tokio::sync` is fine. **~0.25 session** |
| `fastvideo-engine-service` `caps` (489 lines) | the whole engine: `image`, `rubato`, tokio `rt` | the gateway needs `CapabilityTable::build`, `tier_alias`, `parse_tier_alias` | move `caps.rs` (deps: protocol, `h3::lora` recipe predicates) into protocol or a `fastvideo-caps` crate. **~0.25 session** |
| `fastvideo-serve-kit` (8.3k) | axum with server features, `hyper`, `mio`, tokio `fs/net/process/time/rt`, `tower-http` fs, `reqwest` (`fetch`), `image` (ingest probing) | disk stores (`LocalArtifactStore`, `UploadStore`, `MemJobStore` manifests), `tokio::time` retries (callback, D1 client), reqwest, D1 over HTTP. The `engine-service` and `media` deps are unused in `src/` (tests only) | split features into `native` (default) and `edge`. Edge keeps auth, keys, handlers, SSE, callback signing, ingest policy and the S3 signer. It adds a D1-binding `JobStore`, an R2-binding `ArtifactStore`, `worker::Fetch` for callbacks and ingest, and `worker::Delay` for retries. Everything JS-backed gets `SendFuture`. **2–3 sessions** |
| `fastvideo-minimax` (1.6k), `fastvideo-ltxapi` (1.7k) | only through serve-kit, plus 2 `tokio::fs` uses in minimax | | once serve-kit builds for edge: **~0.25 session each** |
| `fastvideo-openai-videos` (2.0k) | serve-kit, engine-service (`tier_alias`), `tokio::fs`, axum `multipart` | | after the caps move: **~0.5 session** (multipart must stream to R2, §2 memory) |
| `fastvideo-fal` (7.2k) | serve-kit, `fastvideo-webrtc` and `fastvideo-media` **unconditionally** (the `director` module is always compiled; only str0m is gated) | the director media code | put the whole `director` module behind the feature, and keep the director **signalling** routes (gateway `proxy.rs`) separate. **~0.5–1 session** |
| `fastvideo-reactor`, `fastvideo-webrtc`, `fastvideo-media` | str0m, opus, openh264, tokio net | real media | **stay native** (GPU workers). Only the signalling proxy moves |
| `fastvideo-autoscale` (4.7k) | axum, tokio rt, `metrics`, serve-kit | the controller and admin routes | `policy` (and `gateway` mirror types) behind a `policy-only` feature. **~0.5 session** |
| `fastvideo-serve::gateway` (2.9k) | everything native | | not ported; **rewritten** as the DO (`dispatch`, `tick`, `store`, `proxy` map to §3.1–3.7). New crate `fastvideo-edge`, about 2.5–3.5k lines |

In short, **nothing compiles for wasm as-is**, because every crate reaches
`tokenizers` through `fastvideo-protocol` → `fastvideo-models`. Once that
one edge is cut, the protocol types (the envelope, `Job`, `ModelCaps`,
`ApiError`) are plain serde and should build. That is the check phase 0
runs. The adapters are an axum-on-serve-kit refactor, not a rewrite.

Bundle size is not a constraint now (64 MiB uncompressed). Keep
`image` out of the Worker anyway: it costs startup time within the 1 s
global-scope limit and CPU time. Probing moves to the GPU worker, which
already decodes the input.

## 5. Comparison

### 5.1 Latency

See §1. Pod pools: ≈ 0.15–0.4 s with the DO against ≈ 0.8–1.5 s after the
fix (both estimates). Serverless: about equal. Client → API: better on the
edge Worker than on one CPU pod in EUR-IS-1, especially for clients in the
US and for input uploads, whose bytes then land in R2 near the client.
Status: pushed rather than a 1 s D1 poll.

### 5.2 Cost (100 k jobs/month, 3 pools, estimate)

| | current gateway | Workers + DOs |
|---|---:|---:|
| front | CPU pod `cpu3c` $0.06/h ≈ **$44/month** (≈ $88 with 2 replicas for HA) | Workers Paid **$5** base. Requests ~5 API + ~30 status polls per job = 3.5 M, **included** |
| scheduler | in the pod | DO requests: ~30 progress msgs/job at 20:1 = 150 k, plus alarms 3 × 173 k = 520 k → **included**. Duration worst case (never hibernates): 3 × 128 MB × 2.6 M s ≈ 1 M GB-s → **≈ $7.5**; realistic with hibernation, well under the 400 k included |
| storage | D1 + R2 as now | D1 + R2 as now. DO SQLite: ~1 M rows written, **included** |
| **total** | **$44–88/month** | **≈ $5–13/month** |

Both totals are noise next to the GPUs (H100 serverless $4.18/h, ≈ $3,000
a month for one warm worker). Cost is not a reason for or against the
move.

### 5.3 Operations

| | current gateway | Workers + DOs |
|---|---|---|
| availability | one CPU pod. Image pull up to 19 min on (re)create (e2e/cluster.md); a pod loss takes every API down | multi-region edge; a DO is re-homed by Cloudflare on host failure |
| deploy | image build + pod restart | `wrangler deploy` in seconds, **but it drops every worker socket** (§3.3) |
| local test | cargo tests, `fv-d1-mock` | `wrangler dev` runs workerd with local DO/D1/R2 (no deploy); GPU e2e still needs pods |
| debugging | full Rust, tracing, gdb | `wrangler tail` / Workers Logs; wasm stack traces are poor; a panic aborts the instance |

### 5.4 The alternative without Cloudflare: push inside fv-serve

The gateway could accept worker WebSockets (axum has them), hold the queue
in memory, and write D1 behind. The latency would match the DO path. The
costs: with more than one replica, workers must connect to the replica
that owns their pool (leader election per pool through the D1 lease), a
restart loses the in-memory queue unless D1 is again on the critical path,
and the single CPU pod stays. It is a smaller change (~2 sessions) and a
reasonable fallback. The DO is the better tool: it is one durable
single-writer per pool, for free.

## 6. Risks

- **Vendor lock-in.** We already depend on D1 and R2. DOs add a runtime
  with no drop-in open equivalent; workerd is open source, but running it
  ourselves means operating it. Mitigations:
  - the worker ↔ dispatcher WebSocket protocol is **native** and lives in
    `fastvideo-protocol`, so the fv-serve gateway can implement the same
    protocol (§5.4) as a fallback;
  - adapters stay in shared crates that build both native and edge;
  - D1 stays the record.
- **Deploys drop worker sockets.** Reconcile on reconnect (§3.3). The
  failure mode is a double run or a lost job if reconcile is wrong, so it
  needs a deploy-during-job e2e test.
- **Memory (128 MB) and body limits (100 MB)** against our 64 MB data URIs
  and 100 MB videos: stream to R2, or keep big-body routes on the native
  gateway until they stream.
- **`!Send` everywhere.** `SendFuture` wrapping is mechanical but spreads
  through serve-kit's `EngineGate`, `JobStore` and `ArtifactStore`. If the
  edge build forces those traits to lose `Send`, the native build suffers.
  Keep the bounds and wrap on the edge side.
- **Fetch cannot target an IP address**, and non-standard ports are
  unreliable. Signalling goes over host names or over the worker socket
  (§3.7).
- **Serverless pools gain little.** The Runpod queue stays in the path.
  Whether moving hot pools to pods pays off is a question for the
  autoscaler and its cost model, not for the gateway.
- **Rust on Workers is less common than JS.** Some bindings lag (RPC
  experimental). Panic=abort resets the isolate. Fewer examples exist.
  It is still well supported for fetch, DO, SQLite, WebSocket hibernation,
  alarms, D1 and R2, which is all this design needs.
- **DO placement.** A pool DO far from its pods adds a WAN trip per
  message. Pin it with a `locationHint` per pool region. A pool spanning
  US and EU pods needs one DO per region-pool.

## 7. Recommendation and phased plan

Adopt the DO as the **dispatcher for pod pools**. Keep the fv-serve gateway
as the API front until phase 3 is justified. Effort is in agent-sessions
(one focused session is about one PR of this repo's usual size).

| phase | scope | exit criterion | effort |
|---|---|---|---|
| **0. Feasibility** (no deploy) | cut `protocol` → `models` → `tokenizers` (§4 row 1) and move `CapabilityTable` out of engine-service. `cargo check -p fastvideo-protocol --target wasm32-unknown-unknown` on the build pod. A hello `PoolScheduler` DO in `crates/fastvideo-edge` (workers-rs 0.8, SQLite, hibernated WebSocket, alarm), exercised with `wrangler dev` locally. Measure the queue time after the other agent's fix, to have the real baseline | protocol builds for wasm. Hello-DO push/ack works locally. Baseline numbers for pod and serverless pools | **1** |
| **1. DO dispatcher, pod pools** | `PoolScheduler` DO (§3.1–3.4): queue, registry, push/ack/adopt, reaper alarm, D1 write-behind. Worker-side WebSocket client in fv-serve (`gateway.dispatcher`). The gateway's `RemoteGate` gets pool `kind = "dispatcher"` and enqueues through the DO. Deploy (with the owner's go-ahead) to a staging Worker. E2E on one pod pool, including a deploy during a job | queue time ≤ 0.4 s p50 on a pod pool; no lost or double job across a redeploy | **3–4** |
| **2. Scaling and signalling in the DO** | autoscale `policy` in the DO alarm (policy-only feature, Runpod REST via `Fetch`, $8 floor in a fleet DO). Session leases and signalling relay over the worker socket (§3.7). R2 presigned uploads (§3.5, can land earlier in the current gateway) | autoscaler parity with `fv-autoscale-sim` decisions; director/Reactor session through the relay | **2–3** |
| **3. APIs on the Worker (optional)** | serve-kit `edge` feature, adapters on axum-in-Worker, streaming large bodies to R2, key store on the D1 binding. Retire the CPU pod | the e2e suite of every API against the Worker | **5–8** |

Total: 6–8 sessions to a working DO dispatcher with autoscaling
(phases 0–2), and 11–16 sessions including the API move. Decide on phase 3
after phase 1's numbers. If the fixed gateway lands near 1 s and the
single CPU pod has not been a problem, phases 0–2 are enough.

Do now, whatever is decided: R2 presigned uploads (§3.5) and progress
push to SSE. Both help the current gateway too.

## 8. Spike

No code was written. CLAUDE.md requires builds on the shared build pod, and
this task started no pods, so the wasm check stays in phase 0. The
dependency analysis in §4 comes from `cargo tree --target
wasm32-unknown-unknown` (metadata only), which shows that one edge
(`fastvideo-models` → `tokenizers`) is the blocker for every crate.

## Sources

- workers-rs: <https://github.com/cloudflare/workers-rs>; docs.rs `worker` 0.8.7:
  [DurableObject](https://docs.rs/worker/latest/worker/durable/trait.DurableObject.html),
  [State](https://docs.rs/worker/latest/worker/durable/struct.State.html),
  [Storage](https://docs.rs/worker/latest/worker/durable/struct.Storage.html),
  [WebSocket](https://docs.rs/worker/latest/worker/struct.WebSocket.html);
  `SendFuture` / axum `Send`: [workers-rs issue #485](https://github.com/cloudflare/workers-rs/issues/485)
- [Rust on Workers](https://developers.cloudflare.com/workers/languages/rust/)
- [Workers limits](https://developers.cloudflare.com/workers/platform/limits/),
  [64 MiB size changelog (2026-09-04)](https://developers.cloudflare.com/changelog/post/2026-09-04-increased-worker-size-limit/)
- [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/)
- [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/),
  [DO limits](https://developers.cloudflare.com/durable-objects/platform/limits/),
  [DO WebSockets and hibernation](https://developers.cloudflare.com/durable-objects/best-practices/websockets/)
- [R2 presigned URLs](https://developers.cloudflare.com/r2/api/s3/presigned-urls/)
- Direct IP fetch: [workerd #93](https://github.com/cloudflare/workerd/issues/93),
  [community thread](https://community.cloudflare.com/t/need-help-about-direct-ip-access-not-allowed/494422)
- Repo measurements: [e2e/gateway.md](e2e/gateway.md), [e2e/cluster.md](e2e/cluster.md),
  [gateway.md](gateway.md) §3–§8
