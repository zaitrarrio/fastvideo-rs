# The edge as the only entry point: retiring the fv-serve gateway pod

Status: **stage 1 (edge parity) built, 2026-10-06**; see "Stage 1 as
built" in §9 for what changed from the design. Stages and their exit
criteria are in §9; the owner's answers to §10 are recorded there.

The owner's decision: retire the fv-serve gateway pod. fv-control becomes
the only control plane for the model-family stacks, and one public
Cloudflare endpoint (the edge Worker) is the only entry point:

1. The edge authenticates API keys, enforces quotas, routes jobs to the
   per-family Durable Object (DO) queues of #13
   ([dispatch-do-family.md](dispatch-do-family.md)) and admits streaming
   sessions through the family DO, which hands the client to a GPU worker
   for media.
2. Protocol code stays on the workers. fal, MiniMax, OpenAI-style, LTX,
   native and Reactor are already served by fv-serve on every worker. The
   edge forwards by model id and family; it ports no protocol code.
3. Results go straight from workers to R2 (#13 direct uploads).
4. fv-control stays the control plane for clusters, scaling, keys and the
   admin token (#9, [gateway-less-auth](../control/gateway-less-auth.md)),
   the dashboard and the console.

This document reads the gateway code (`crates/fastvideo-serve/src/gateway/`,
the routes it mounts in `gateway::assemble`, `configs/serve/gateway*.toml`,
`control/src/cluster/*`) and lists every feature it provides, where each one
moves and how it is tested (§3). It assumes [gateway.md](gateway.md),
[gateway-cloudflare.md](gateway-cloudflare.md) and
[dispatch-do-family.md](dispatch-do-family.md).

## 1. Summary

```
client ──(one URL, API key)──► edge Worker (fv-edge, workers-rs)                         fv-control
                                │ 1 classify: (method, path) → protocol, model | job id | session id   │ clusters, pods, scale,
                                │ 2 auth: key digest → key id (D1 api_keys, cached) ; quotas           │ keys + admin token (via
                                │ 3 pick a front: a ready worker of the model's family                 │ the edge's admin routes),
                                │   (registry snapshot; sticky by job id)                              │ dashboard, family metrics
                                │ 4 forward: internal token + x-fv-edge-auth verdict                   │
                                ▼                                                                      │
        ┌──────── GPU worker = front + executor (fv-serve role=worker, auth.mode = trust-edge) ◄───────┘ drain / status
        │ front: the protocol adapters as today (normalize, ingest, negotiate, D1 row,                  (internal token)
        │        render); its EngineGate enqueues the envelope on its family DO (FrontGate)
        │ executor: ws per family ◄── offers / sessions / upload grants ── family DO ("h3", "ltx", …)
        │ output ── part PUTs ──► R2 (direct, #13)
        └─ media ◄══ WebRTC / WHIP ══► client (direct; signalling proxied by the edge)
```

- **Every GPU worker is also a front.** Today a worker already mounts every
  API (`app.rs`: `assemble` with the adapters), behind the internal token.
  In edge mode it keeps doing so; what changes is the engine seam: a submit
  does not run on the local engine but is enqueued on the model's family DO
  (`FrontGate`, the gateway's `edge_enqueue` path moved to the worker). The
  DO then offers it to whichever worker has a slot, maybe the same one.
  This keeps the queue in the DO (re-dispatch after a loss, the arbiter,
  credits) and keeps all protocol code on the workers.
- **The edge owns identity, not rendering.** It verifies the presented key
  and decides quotas, then forwards the request with a verdict header. The
  front's `Auth::authenticate` (new mode `trust-edge`) turns the verdict
  into `Ok(owner)` or an `ApiError`, which the protocol renders as it does
  today (a MiniMax `base_resp` code, a fal 401, …). So the edge never needs
  to know a protocol's error shape.
- **The registry.** Workers announce in `hello` the names that route to
  them (model ids, served names, aliases, tier aliases, fal app ids, the
  protocols they mount, their front URL). The family DOs fold that into one
  small **registry** object; each edge isolate caches its snapshot for a few
  seconds and routes without a DO round trip per request.
- **Nothing on the GPU hosts is public without the internal token.** In
  `trust-edge` mode every route except health and signed files needs
  `x-fv-internal-token` (as the gateway-worker role does today). The pods'
  Runpod proxy URLs stay reachable but answer 401.

What goes away: the CPU gateway pod and its watchdog, `gw_dispatch`,
`gw_workers`, `gw_sessions`, worker probes, the reaper, live-caps polling,
`RemoteGate` placement, the gateway's D1 view, the in-gateway autoscaler, and
serverless pools (§10 Q1).

## 2. The request path

### 2.1 Classification (edge)

The edge matches `(method, path)` against a route table generated from
`fastvideo_serve::router::route_table` (the §9 table every owner mounts;
`check_route_table` already rejects collisions). Generation happens at build
time into a JSON the edge embeds; a test fails when it drifts. Each entry
says how to find the routing key:

| key | routes | how the edge reads it |
|---|---|---|
| fal app | `/{owner}/{app}[/sub]`, `/run/{app}…`, `/{app}/requests/{id}/…`, director `/{app}/director/ice` | the longest registered fal app id that prefixes the path |
| body `model` | MiniMax `POST /v1/video_generation`, OpenAI `POST /v1/videos[/sync]` (JSON or multipart), LTX v1/v2 submits, native `POST /fv/v1/jobs`, `POST /fv/v1/streams` | a streaming scan for the top-level `model` key (JSON) or the `model` part (multipart); the bytes read are kept and replayed, the rest is streamed, the body is never parsed whole |
| job id | status, result, content, cancel, delete of every API whose path has no model | the id from the path or query (`task_id`); routed to any front that mounts the protocol, sticky (§2.3) |
| session id | `/wma/session/heartbeat`, Reactor `/session`, `/events`, `/stop_session`, `/sessions/{sid}/…`, `/fv/v1/streams/{id}*` | the family DO's session table (owner → session for Reactor's lease-by-caller) |
| none | lists, `/v1/models`, `/fal/schema`, JWKS, `/console*` | any ready front that mounts the protocol |
| edge-owned | `/ping`, `/health`, `/healthz`, `/`, `/fv/v1/status`, `/fv/v1/capabilities`, `/metrics`, `/fv/v1/admin/keys*`, `/uploads/{token}` | answered by the edge (§3) |

A body without a `model` key routes to the family of the protocol's default
model, which the fronts announce (`registry.defaults[protocol]`). A model the
registry does not know goes to any front of that protocol with the verdict
`unknown_model`, so the 400/404 "model not served" keeps each API's shape.

### 2.2 Identity and quotas (edge)

- Key check: SHA-256 of the presented key (either scheme; fal's
  `Key id:secret` hashed as one token, as `parse_authorization` does), looked
  up in `api_keys` (D1 binding) and in the `FV_API_KEYS` hash list (Worker
  secret). Hits are cached per isolate for 15 s, misses for 5 s. The key id
  is `key_<12 hex>` as everywhere else.
- The verdict header, set only by the edge (any client-supplied
  `x-fv-edge-*` header is dropped):
  `x-fv-edge-auth: {"v":1,"key":"key_…"|null,"presented":bool,"valid":bool,"scheme":"key"|"bearer"|null,"deny":null|{"kind":"rate_limited","retry_after":s}}`.
  It rides with `x-fv-internal-token`, which is what makes it trusted.
- The front applies the per-API policy (`AuthPolicy::Require` / `Open`) to
  the verdict instead of to the key ring, so fal still wants `Key`, MiniMax
  `Bearer`, and OpenAI-style stays open but identified.
- Quotas, decided at the edge and rendered by the front through `deny`:
  - per key, requests per minute: the Workers rate-limiting binding (per
    location, approximate) for the coarse cap;
  - per key, jobs in flight: counted by the family DOs (`owner` on the
    enqueue, new), summed over families by the registry; the MiniMax limits
    (300/min, 30 in flight per key, `fastvideo-minimax/src/limits.rs`) become
    edge limits, because a per-process limiter on N fronts allows N times
    the limit;
  - per model, queue admission (`max_queued` of the pool, 429 `QueueFull`):
    enforced by the family DO at enqueue (new `max_queued` on `EnqueueReq`),
    returned to the front, which renders it as today;
  - invalid keys: requests with an invalid key are rate-limited per client
    address at the edge with a plain 429, so a key-guessing flood never
    reaches a GPU host.

### 2.3 Picking a front (edge)

From the registry snapshot: connected, ready, non-draining workers that
serve the model (or mount the protocol). Submits go to the least-held front,
idle workers first (a front's CPU work is ingest and a D1 insert, but a
worker in a model load is slower). Requests by job id go to the front picked
by rendezvous hashing of the id over the protocol's fronts, so every status
poll of one job lands on one front and is answered from its in-memory view
(§3.2, status freshness). When the registry has no front for a family the
edge forwards the request to any front with the verdict `unavailable`
(rendered 503 + `Retry-After` in the API's shape); with no front at all it
answers a plain JSON 503.

Forwarding: `fetch` to the front's Runpod proxy host
(`https://{pod}-8000.proxy.runpod.net`, a host name on 443, never an IP),
streaming both ways; `x-forwarded-for` and `cf-connecting-ip` are passed
for Reactor's lease by client address.

### 2.4 The front (worker)

New worker mode, set by fv-control in edge clusters:

| setting | env | meaning |
|---|---|---|
| `auth.mode = "trust-edge"` | `FV_AUTH_MODE` | identity from `x-fv-edge-auth`; every route but health, `/metrics` and signed files needs the internal token |
| `dispatch.front = true` | `FV_DISPATCH_FRONT` | the engine seam is `FrontGate`: `submit` builds the envelope (inline / source / store inputs, the gateway's `plan_inputs`, moved) and enqueues it on the model's family DO; `cancel` marks the row and sends the DO cancel; `models()` are this worker's caps |
| `dispatch.endpoint` | `FV_DISPATCH_ENDPOINT` | this pod's own URL (front and session endpoint). Today `edge_link` announces `public_base`, which on a gateway worker is the **gateway's** URL: a gap (§4) |
| `server.public_base_url` | `FV_PUBLIC_BASE_URL` | the edge URL: every URL a front renders (fal `status_url`, webhook bodies, upload tickets) points at the edge |
| `dispatch.families`, `direct_upload`, `do_url` | as #13 | `do_url` = the edge URL |

The front's job store is the worker's `D1JobStore` plus the gateway's
read-through view (`GatewayJobStore`, moved): jobs this worker executes are
authoritative in memory; jobs it fronted or reads by id come from D1 and the
view; `watch()` (fal `status/stream`, sync endpoints) is woken by the family
DO (a long poll `GET /families/{f}/jobs/{id}/wait?since=` answered at each
`ack` / `done`, new) and falls back to the D1 poll every `watch_poll_ms`.

## 3. Parity checklist

Every feature of the gateway, found in `gateway/{mod,dispatch,edge,proxy,
routes,tick,store,scale,runpod,schema}.rs`, the routes `gateway::assemble`
mounts (`releases.rs`, `flags.rs`, `admin_token.rs`, `console.rs`,
`multiworker.rs`, serve-kit `keys.rs` / `uploads.rs`), `configs/serve/
gateway*.toml` and fv-control's `control/src/cluster/*`. Columns: **today**
(where it is), **moves to** (E = edge Worker, D = family DO, W = worker /
front, C = fv-control, — = dropped), **how**, **test** (U unit, I native
integration in `crates/fastvideo-serve/tests`, X compat suites
`tests/compat/run.sh` in a new `FV_COMPAT_EDGE=1` mode, S staging Worker, G
the live GPU test), **stage** (§9).

The native integration host is the one #13 already uses for the DO (a
native stand-in that runs the same pure scheduler): the edge's routing,
auth and quota decisions go into a pure module
(`fastvideo-dispatch-proto::front`), so the same code runs in the wasm
Worker and in a native stand-in the tests and the compat mode start.

### 3.1 Auth, keys, quotas

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| `auth.mode = keys`: per-API policy (fal `Key`, MiniMax/LTX/native `Bearer`, OpenAI/FastWan/Reactor open but identified) | serve-kit `auth.rs` on the gateway | E (verify) + W (policy) | §2.2: verdict header, `trust-edge` mode applies the policy | U (`auth.rs` new cases), I, X (every suite with and without a key) | 1 |
| `auth.mode = none` (demo) | gateway | E | the edge skips key checks when the cluster says `auth: none`; verdict `key: null, valid: false` | I | 1 |
| static keys `FV_API_KEYS` (SHA-256 list) | gateway env | E | Worker secret `FV_API_KEYS`, same format | U | 1 |
| minted keys: `POST/GET /fv/v1/admin/keys`, `DELETE /fv/v1/admin/keys/{id}` | serve-kit `admin_routes` on the gateway (D1 `api_keys`) | E | same routes and shapes on the edge over the D1 binding (same table, same `fv_` + 32 bytes, digest only); fv-control's mint / list / revoke call the edge with the admin token | U, I, S | 1 |
| key revocation latency | immediate on the replica that revokes, ≤ 30 s on others (D1 reload) | E | ≤ 15 s (isolate cache TTL); an edge-wide revocation epoch in the registry DO can make it immediate if wanted (§10 Q4) | I (revoke → refused within the TTL) | 1 |
| `last_used_at` | key store flush | E | written behind (`waitUntil`), at most once per minute per key per isolate | U | 1 |
| admin token: `/fv/v1/admin/*`, `/fv/v1/gateway/pools`; `FV_ADMIN_TOKEN` or generated, sealed at `/fv/v1/admin/token/sealed` | `admin_token.rs`, gateway | C + E | fv-control makes the token (as for gateway-less clusters, #9) and writes its SHA-256 to the edge (D1 `edge_admin`, read by the edge, cached 30 s); the sealed route is not needed (fv-control holds the token) | U, I, S | 1 (edge), 2 (C) |
| per-pool admission `max_queued` → 429 `QueueFull` + `Retry-After` | `dispatch.rs` | D + W | `EnqueueReq.max_queued` (new) per model; the DO answers 429, the front renders it | U (sched), I | 1 |
| MiniMax 300/min and 30 in flight per key | in the adapter, per process | E (+ D count) | §2.2; the fronts' own limiter stays as a per-front backstop | U, I | 1 |
| worker queue limit `limits.queue_max` | worker | W / D | unchanged on the worker; the DO's credits make it rarely hit | — | — |

### 3.2 Batch jobs and fal queue semantics

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| submit pipeline: normalize → admission → resolve against caps → ingest → negotiate → D1 row → dispatch | serve-kit handlers on the gateway over `RemoteGate` | W (front) | the same handlers on the front; `FrontGate::submit` enqueues on the DO | I, X | 1 |
| request ids (fal uuid, MiniMax 18 digits, `video_gen_…`, native) | adapters' `new_external_id` on the gateway | W | unchanged (fronts mint them); D1 `jobs (protocol, external_id)` is the lookup every front shares | X | 1 |
| fal queue: `POST /{app}`, `GET …/requests/{id}/status`, `…/status/stream` (SSE), `GET …/requests/{id}` (result), `PUT …/cancel`, `/run/{app}` sync | `fastvideo-fal` on the gateway; status from `GatewayJobStore` | W | routed by fal app (§2.1); status / result from D1 via the front's view; sticky front per id | X fal-py, fal-js (queue, status, subscribe, run, cancel, errors) | 1 |
| **stable result URLs (#12)**: one URL per finished request across `get()`, `result()` by id and the webhook | `fal::queue::url_issued_at` + `UrlSigner::url_issued` (S3: `X-Amz-Date = issued`) | W | two fronts must sign the same URL: same R2 credentials, endpoint and bucket on every worker (the S3 presign is deterministic). **Gap:** `DirectStore` (direct uploads, #13) implements only `url_for`, so `url_issued` falls back to the trait default (re-signed per read, not stable): it must delegate (§4) | U (`DirectStore` delegates), X fal-py "result() by id" with the 1.1 s wait **through two different fronts** | 1 |
| status reads from memory (`watch_poll_ms` view, finished jobs 60 s) | `gateway/store.rs` | W | the view moves to the front's store; stickiness keeps one job's polls on one front | I (`job_overhead` budget through the edge stand-in) | 1 |
| completion pushed to the dispatching replica (`follow`: `GET /fv/v1/internal/jobs/{id}?wait_s=25&since=`) | `dispatch.rs::follow` | D + W | the family DO's job wait (§2.4) wakes the front's watcher; then one D1 read | I (`job_overhead`: poll / SSE / webhook within work + 400 ms at 0 ms D1) | 1 |
| cancel (queued → cancelled at once; running → `cancel_requested`, forwarded) | `dispatch.rs::cancel_job` | W + D | the front marks the row as the gateway did and sends `POST /families/{f}/cancel/{job}`; the DO cancels a queued job or sends `cancel` to the holder, whose write is authoritative | I, X (fal cancel, OpenAI delete) | 1 |
| delete (OpenAI `DELETE /v1/videos/{id}`, native, MiniMax) | gateway via `gw_dispatch` | W | as cancel, then the row and artifacts are removed by the front (D1 + R2 shared) | X | 1 |
| inputs: inline ≤ 8 MiB, `source` passthrough for large client URLs, store (R2) otherwise | `dispatch.rs::plan_inputs` | W | moved to `FrontGate` unchanged | I (`inline_inputs_take_the_store_off_the_dispatch_path` on the front) | 1 |
| 424 restage (worker cannot fetch a `source`) | `edge.rs::edge_restage` in the gateway's tick | W + D | no tick: the front puts a **fallback URL** in the envelope (the R2 key its background `stage_for_retry` copy will land at, signed); a worker that gets a 424 fetches the fallback (retrying until it exists, ≤ 3 min) instead of nacking 424 | I (`a_424_restages_inputs_through_the_store`, rewritten) | 1 |
| inputs staged for a re-dispatch after a loss | `stage_for_retry` + `gw_dispatch` row | W + D | the copy is made by the front (as today) and referenced by the envelope the DO keeps; the DO re-dispatches the same envelope | I | 1 |
| worker loss → re-dispatch once, then fail; fail the D1 row | gateway reaper (`tick.rs`), DO for DO pools | D (+ executor or front for the D1 row) | the DO re-dispatches; a job the DO fails is failed in D1 by the DO (`edge_failed` moves into the DO: it writes the terminal row through the D1 binding with the lease fence) | U (sched), I | 1 |
| `dispatched_at` / `timings.queue` split (`dispatch` + `wait`) | job fields set by worker | W | unchanged | X (fal timings present) | — |
| D1 insert behind for DO-only models (phase 2) | `gateway/store.rs::set_insert_behind` | W | moved with the store | I | 1 |
| sync endpoints: `/v1/videos/sync`, fal `/run/{app}`, LTX v1 | gateway waits via `watch()` | W | the front waits (same code). The 100 s Runpod proxy cap applies to the edge → front hop exactly as it applied to client → gateway pod (design.md §8: 524 after 100 s); unchanged | X (openai sync, LTX v1, fal run) | 1 |
| lists (`GET /v1/videos`, `/fv/v1/jobs`, MiniMax query) by owner | gateway D1 | W | any front of the protocol; D1 `list` | X | 1 |
| request echo, callbacks stored on the row | gateway insert | W | unchanged | X | — |

### 3.3 Webhooks

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| fal webhooks (Ed25519, `FV_WEBHOOK_ED25519_KEY`), MiniMax `callback_url`, OpenAI-style | sent by the worker that ran the job | W | unchanged; the body's URLs point at the edge (`FV_PUBLIC_BASE_URL`) | X fal-webhook (fal-client + @fal-ai/client, JWKS verify), minimax | 1 |
| JWKS `/.well-known/jwks.json` | gateway | W (via E) | any front; one shared key across workers | X fal-webhook | 1 |
| MiniMax callback challenge at submit | gateway (`challenge_done_elsewhere` on workers) | W | the front runs it; the executor of a DO envelope keeps `challenge_done_elsewhere` | X minimax | 1 |

### 3.4 Multi-pool routing, aliases, capabilities

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| model → pools (first available with room) | `Catalog::pools_of`, `pools_for_name` | E (model → family) + D (model → worker) | the registry maps every name to a family; the DO offers only to workers that announce the model | U (registry), I (two families, two pools of one family) | 1 |
| aliases: `[aliases]`, `pools[].aliases`, tier aliases (`h3-turbo`, `ltx-max`), `engine.tier_overrides` | `Catalog::build` on the gateway | W (resolve) + E (route) | fv-control renders the spec's `gateway.aliases` into every worker's `[aliases]`; workers resolve and announce the alias names in `hello`; the registry routes them | U, X (MiniMax `MiniMax-H3-Turbo`) | 1 (W, E), 2 (C) |
| `/fv/v1/capabilities` (aggregated `models`, `tiers`, `aliases`, `pools`, `readiness`, `auth.mode`) | `routes::capabilities` | E | the edge merges one front's capabilities per family (`models`, `tiers`, `aliases` concatenated by id) and adds `pools` and `readiness` from the registry; the console reads it | I (shape test against the gateway's), X console | 1 |
| fal apps (`[protocols] fal_apps`: the union) | gateway base | W + E | each worker mounts its config's apps and announces them; the edge routes `/{app}` by them; an app nobody serves answers 404 from any fal front | X fal-py | 1 |
| Reactor model (`[gateway] reactor_model`) | gateway | W + E | workers announce `reactor: true` for their Reactor model; the edge routes Reactor routes to that family (the spec's `reactor_model` picks which when several do) | X reactor | 1 |
| `/v1/models`, `/fal/schema` | gateway (local) | W | any front (a list of its own models: §10 Q6 on whether to merge) | X openai | 1 |
| experimental feature flags (D1 `feature_flags`) applied to caps before negotiation; `/fv/v1/admin/flags` | `flags.rs` (`FlaggedGate`) on the gateway | W (+ E route) | fronts already wrap their caps with the flags (every worker runs `FlaggedGate`); the admin routes are forwarded by the edge to any front with the admin verdict | I (`tests/flags.rs` through the edge), X console | 1 |

### 3.5 Streaming: director, Reactor, WHIP, leases

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| fal director: `/wma/ice`, `/{app}/director/ice`, `/run/{app}/director/ice`, `/wma/session` (lease on `session_id`), `/wma/session/heartbeat`, `/start-session` (SSE), `/info` | `proxy.rs::director_routes` + `edge_admit` for family pools | E + D | the edge admits `POST /families/{f}/sessions {model, kind: director, owner}`, proxies the signalling to the granted endpoint (SSE streamed), records the session id; heartbeats renew the DO lease; media client ↔ GPU | X fal-director (Chromium `fal.realtime.open`), G | 1 |
| Reactor local runtime: `/start_session` (lease to the key owner, else the client address), `/session`, `/stop_session`, `/events` (SSE), `/schema`, `/sessions/{sid}/…` (incl. uploads) | `proxy.rs::reactor_routes` | E + D | admission as above (`kind: reactor`, owner = key id or `ip:<cf-connecting-ip>`); follow-up calls by the DO's owner → session map; `/stop_session` releases | X reactor (A/V, video-only, causal), G | 1 |
| native streams `POST/GET/DELETE /fv/v1/streams`, `…/{id}/commands` (worker publishes over WHIP to the client's `whip_url`) | `proxy.rs::stream_routes` | E + D | admitted (`kind: stream`), proxied, lease = stream id | I (`streams_whip`), X console stream page | 1 |
| native WHIP ingest `POST /fv/v1/streams/ingest` (SDP offer → answer, `Location`), `GET/DELETE …/{id}`, `…/commands` | **not proxied by the gateway today** (workers only) | E + D | new: admitted (`kind: ingest`), the offer proxied and `Location` rewritten to the edge; or, per the owner's "hand the client to a worker over direct WHIP", a **307** to the worker with a short-lived session capability (§10 Q3) | I (`ingest_whip` through the stand-in), G (console live page) | 1 |
| WHEP playback | SFU (Cloudflare Realtime), not the gateway | — | unchanged | — | — |
| session leases: `gw_sessions` (D1), lease expiry in the tick, release on stop | `proxy.rs`, `tick.rs::expire_leases` | D | the family DO's `sessions` table (#13) is the only lease: admission, renew, expiry alarm, release, re-announce after a reconnect | U (sched, exists), I | 1 |
| `max_streams` per pool | pool config | W / D | worker `dispatch.sessions`; the DO answers 429 when no GPU has room | U | 1 |
| serverless `kind:stream` (Runpod job publishes over WHIP) | `proxy.rs` | — | dropped with serverless pools (§10 Q1) | — | — |

### 3.6 Uploads and files

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| fal `POST /storage/upload/initiate`, `/uploads/{token}` PUT, LTX `/v1/upload`, `ltx://uploads/<token>` | serve-kit `UploadStore` on gateway-local disk (pinned to one replica) | W (ticket) + E (bytes) + R2 | the front mints the ticket (token signed with a key fronts and edge share, `FV_UPLOAD_TICKET_KEY`); the `upload_url` is the edge's `/uploads/{token}`; the edge streams the PUT into R2 (`uploads/<token>/<file>`) through a binding; ingestion on any worker resolves the token from R2. Uploads stop being pinned | U (ticket), I, X fal-js storage upload, ltx upload | 1 |
| Reactor session uploads `/sessions/{sid}/uploads` | worker (session-local) | W via E | proxied with the session (it is the session's worker that needs them) | X reactor | 1 |
| `/files/{id}/{name}` (local artifacts, signed) | gateway | — | edge clusters require R2 artifacts (presigned URLs on the R2 S3 host); `/files` stays for single-host use | X (result download) | 1 |
| `/fal/proxy` (fal-js `proxyUrl`) | gateway (pinned: re-enters the router) | W via E | proxied to any fal front; its inner request goes to the same front, which serves it | X fal-js (`requestMiddleware` + `proxyUrl`) | 1 |

### 3.7 Console and admin APIs

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| `/console`, `/console/{admin,deployments,avatar,live,stream,native}`, `/console/models/…`, `/console/assets/*` | `console.rs` on the gateway | E → W | the edge serves `/console*` from any front (static, cacheable at the edge) so its same-origin API calls keep working; fv-control's dashboard links to `<edge>/console` (§10 Q5: or fv-control hosts the pages) | X console (Playwright smoke + `ui_gaps`) | 1 |
| keys admin page | console + `/fv/v1/admin/keys` | E | edge routes (§3.1) | X console | 1 |
| flags admin | `/fv/v1/admin/flags` | W via E | §3.4 | I | 1 |
| releases / deployments: `GET /fv/v1/admin/releases`, `/deployments`, `POST …/promote`, `…/rollback` (dispatches `release.yml` with `FV_GITHUB_TOKEN` on the gateway) | `releases.rs` | C | fv-control already reads the `releases` table and dispatches `release.yml` (`/api/releases`, `/api/github/release`); the console's Deployments page links there. Live builds and drift come from the registry (`hello.version`, `sha`) | control unit tests, C UI smoke | 2 |
| `/fv/v1/gateway/pools` (admin: `PoolMetrics`, per-worker detail incl. full `build`) | `routes.rs` | E (+ D) | `GET /fv/v1/edge/families` (admin): every family's `status` + `metrics` (#13), workers with `build`; a compatibility alias at the old path for fv-control during the switch | I, C | 1 |

### 3.8 Metrics and logs

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| `/metrics` (Prometheus): `fv_pool_{queued,running,oldest_queued_seconds,streams,workers,available,submitted_total}`, `fv_gateway_submit_phase_seconds`, request metrics | gateway | E (fleet) + W (per worker) | the edge renders the per-family gauges from DO `metrics` (admin token); submit-phase timings move to the front (`fv_front_submit_phase_seconds`); each worker's own `/metrics` stays (internal token) | U (render), I | 1 |
| `PoolMetrics` for the autoscaler (`GET /fv/v1/gateway/pools`) | gateway tick | D | `GET /families/{f}/metrics` (#13) | U | — |
| fv-control collector: jobs per worker, health from the gateway's admin view | `collector.ts` | C | read the edge's families view (held, sessions, ready per worker) | control unit + integration | 2 |
| logs: each pod ships to fv-control `/ingest` (`log_ship.rs`) | gateway and workers | W + E | workers unchanged; the edge logs one structured line per request (route, family, front, status, ms; no keys) to Workers Logs and ships a batch to fv-control's ingest through a service binding (`waitUntil`) | S (lines visible in fv-control) | 1 (E), 2 (C) |
| per-request tracing (`TraceLayer`) | gateway | E + W | `x-request-id` minted at the edge, passed to the front | I | 1 |

### 3.9 Health, drain, rolling deploys

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| `/ping` (200 when a pool can take work, 503 draining), `/health`, `/healthz` (pool summary), `/` | `routes.rs` | E | from the registry: ready if any family has a ready front; the same JSON shapes | I | 1 |
| `/fv/v1/status` (public: pool and worker states, `versions`, `mixed_versions`, failed models with reasons) | `status.rs` over the tick's probes | E | rendered by the edge from the registry in the same `fv.status` shape (pools = families, workers `w1…`, states from `hello` / `status` frames, `failed_models` announced by workers) | I (shape test against the gateway's), X console status strip | 1 |
| worker probes (`/fv/v1/internal/status` every `tick_s`), live caps every `caps_refresh_s`, `gw_workers` registration | `tick.rs`, `worker::Registration` | D | the worker sockets replace them (hello, status, slots) | U, I | 1 |
| a worker on a GPU that cannot run its model (`readiness: failed`) | tick → worker `failed`, never dispatched | W + D | the worker announces `failed_models`; the DO never offers those models to it; the registry shows it `failed` | I | 1 |
| drain: `POST /fv/v1/internal/drain` / `undrain`, SIGTERM | worker; gateway skips draining workers | W + D + E | the worker reports `draining` on its sockets (exists); the edge stops picking it as a front | I | 1 |
| gateway drain on shutdown (503, admission stops) | gateway | — | no process to drain; an edge deploy is atomic per request | — | — |
| rolling redeploy (fv-control `roll`: new workers, wait ready, drain old, swap, `patchGateway` with the new URLs, delete) | `do.ts::roll` | C | the same steps without `patchGateway`: new workers join the DOs and the registry by themselves; readiness from the families view | control integration, G (optional) | 2 |
| edge deploys | — | E | a deploy drops every worker socket; workers reconnect in 0.25–0.3 s and re-announce (#13, measured); in-flight proxied requests finish on the old version | S (deploy during a burst) | 1 |
| gateway pod watchdog (deadline, balance floor; deletes the cluster's pods and itself) | `GATEWAY_BOOT` | C + W | fv-control's cron already enforces both (`collector.ts`); the second, independent backstop moves into `WORKER_BOOT` in edge mode: each pod deletes **itself** at the deadline or below `min_balance` | control unit (payload), G (a pod's own backstop fires on a short deadline) | 2 |

### 3.10 Serverless pools and autoscale

| feature | today | moves to | how | test | stage |
|---|---|---|---|---|---|
| `kind = "runpod-serverless"` pools: `/run` with the queue envelope, Runpod `/health` probes, `/status` reaper, `/cancel` | `runpod.rs`, `dispatch.rs`, `tick.rs` | — | dropped (§10 Q1). fv-control never launches them; single-model serverless endpoints (`runpod-endpoint.sh`) do not use the gateway and keep working; the worker's Runpod queue handler stays | — | 4 |
| in-gateway autoscaler (`[autoscale]`, `PoolScaler`, D1 lease) | `autoscale.rs`, `fastvideo-autoscale` | C (later) | fv-control clusters set `[autoscale] enabled = false` today, so they lose nothing: manual scale and the idle auto-stop policy stay in fv-control. A family-metrics signal source for `fastvideo-autoscale` (or a port into fv-control) is a follow-up, not part of this migration | — | follow-up |
| `fv-autoscale` standalone with `HttpGatewayPools` | autoscale crate | — | that source is removed with the gateway; the Runpod-health source stays | — | 4 |

## 4. Gaps found while reading the code

1. **Stable result URLs regress with direct uploads.** `DirectStore`
   (`crates/fastvideo-serve/src/upload.rs`) implements `UrlSigner::url_for`
   but not `url_issued`, so fal results stored through it get the trait's
   default (a fresh signature per read). Fix: delegate `url_issued` to the
   inner store; test `get()` / `result()` 1.1 s apart. This affects any
   cluster with `direct_upload` today, not only edge mode.
2. **The worker's session endpoint is its public base URL** (`app.rs`:
   `endpoint: base`), and on a gateway worker `FV_PUBLIC_BASE_URL` is the
   gateway's URL. Edge mode needs both: rendered URLs on the edge, the
   endpoint on the pod (`dispatch.endpoint`, §2.4).
3. **Uploads are local disk** (`UploadStore`), pinned to one gateway
   replica. With several fronts they must be shared (§3.6).
4. **The MiniMax rate limiter is per process** (§3.1).
5. **No `owner` and no `max_queued` on `EnqueueReq`**, so the DO can neither
   count jobs per key nor apply pool admission (§3.1).
6. **The `gateway` image variant is also the CPU fake-engine worker** of
   fv-control's `tiny-cpu` template (`FAKE_CPU_WORKER_TOML`). Deleting the
   `serve-gateway` stage needs a `cpu` (fake engine, no CUDA) variant kept
   for workers (§9 stage 4).
7. **The staging edge writes its own D1** (`fv-edge-staging`), and its
   `OUTPUTS` bucket is not the workers' artifact bucket. Edge mode needs the
   edge, the fronts and the executors on one D1 (jobs, keys, flags) and one
   output bucket the fronts can sign (§10 Q2).
8. **The worker's internal routes take the internal token, but the token is
   per cluster** (`ClusterSecrets.internal_token`). One edge serving a
   cluster needs that cluster's token; fv-control registers its SHA-256
   with the edge (D1 `edge_tokens`), so no Worker redeploy per cluster.

## 5. fv-control

### 5.1 Spec

```ts
control_plane: "gateway" | "edge" | "both";   // default "gateway" until stage 4, then "edge" only
```

- `edge`: no gateway pod. `up` phases become `init → workers → register →
  wait`: `register` writes the cluster's internal-token digest, admin-token
  digest and settings (aliases, Reactor model, `auth`) to the edge's D1;
  `wait` reads the edge's families view until each pool has a ready worker.
- `both` (migration and the latency comparison): the gateway pod as today,
  with every pool `dispatch = "durable-object"` and `family` set, so the
  gateway enqueues on the same family DOs the edge uses; the workers run in
  front mode and also keep their internal job route. Both entry points serve
  the same GPUs.
- `gateway.*` settings that still mean something move to `edge.*`
  (`auth`, `aliases`, `reactor_model`, `fal_apps`, `protocols`); the rest
  (`cpu_flavors`, `vcpu`, `base`, `github_token`) apply to `gateway` / `both`
  only. Validation refuses a second running `edge` cluster on the same edge
  in v1 (§10 Q2).

### 5.2 Worker env in edge mode

`FV_SERVE_ROLE=worker`, `FV_AUTH_MODE=trust-edge`, `FV_DISPATCH_FRONT=1`,
`FV_DISPATCH_DO_URL=<edge>`, `FV_DISPATCH_FAMILIES=<from the pool's
models>` (`h3` → `h3`, `ltx2` → `ltx`, causal `wan` recipes → `sfwan`, other
`wan` → `wan`; a pool may override), `FV_DISPATCH_DIRECT_UPLOAD=1`,
`FV_MP4_FRAGMENTED=1`, `FV_DISPATCH_ENDPOINT=https://${RUNPOD_POD_ID}-8000.proxy.runpod.net`,
`FV_PUBLIC_BASE_URL=<edge>`, `FV_INTERNAL_TOKEN`, `FV_UPLOAD_TICKET_KEY`, the
spec's aliases as `[aliases]`, plus the backstop variables for the
self-delete watchdog (`FV_CLUSTER_DEADLINE`, `FV_MIN_BALANCE`,
`FV_BACKSTOP_API_KEY`, now on workers). All new keys are reserved.

### 5.3 Dashboard

The cluster card shows the edge URL, the admin token (reveal, as #9), keys
(mint, list, revoke through the edge), the families view (workers per
family, queue depth and age, sessions, versions), and links to
`<edge>/console`. The collector reads the families view instead of the
gateway's pools view.

## 6. Migration

1. **Stages 1–2 land with the gateway untouched**: `control_plane` defaults
   to `gateway`; edge mode is opt-in per cluster.
2. **Side by side** (`both`), on staging: one cluster, both URLs, the same
   workers; compat suites and the latency comparison run against both
   (stage 3).
3. **Switch**: fv-control's default becomes `edge`. A running gateway
   cluster moves with two operations: `control-plane both` (re-renders the
   worker env: front mode and families; workers restart and reload models,
   minutes per pool, rolled pool by pool with the existing roll machinery so
   one pool keeps serving), then clients move to the edge URL, then
   `control-plane edge` (stops the gateway pod). Clusters are short-lived
   (`cap_s`, default 6000 s), so in practice most are simply relaunched in
   edge mode.
4. **Delete** (stage 4) only after the live test passes: the gateway code,
   configs, image stage, scripts, CI steps and doc references (§9).

Rollback before stage 4: set `control_plane: gateway` and relaunch (the
gateway path is untouched until then). After stage 4: revert the stage-4
PR.

## 7. Failure modes

| failure | effect | handling |
|---|---|---|
| Cloudflare (edge) outage | every API down; jobs already queued or running finish on the GPUs, write D1/R2 and send their webhooks | same blast radius as the gateway pod today, on a multi-region platform instead of one CPU pod |
| no ready front for a family (all loading, draining, scaled to zero) | submits for that family get 503 + `Retry-After` (rendered by another family's front, else plain JSON) | as the gateway did for an unavailable pod pool; there is no scale-from-zero for pods in either design |
| a front dies mid-request | the client gets 502/504; a submit may already have its D1 row and DO job (it runs; the client never saw the id) | as a gateway pod crash today; the job is not lost, only its id. Fal and OpenAI have no idempotency key; MiniMax neither |
| a front is busy (ingest of a 64 MiB data URI) | that submit is slower; the GPU job on the same host is unaffected (engine thread) | least-held front first; large inputs are rare and capped (`body_max_mb`) |
| the executor is lost mid-job | DO re-dispatches once, then fails the row | #13, unchanged |
| a front's status view is stale | at most `watch_poll_ms` (1 s), as on gateway replicas; DO job waits push completion | stickiness keeps one job's polls on one front |
| sticky front leaves (drain, roll) | polls re-hash to another front, one D1 read each until its view warms | — |
| edge deploy | worker sockets drop and reconnect (0.25–0.3 s measured); a job pushed in the gap waits for its ack timeout (10 s) | deploy runbook (#13 §9.9) |
| D1 unavailable | key checks continue from the isolate cache (15 s, then last-known-good up to 10 min for hits); new submits fail on the fronts (row insert) | as today |
| R2 unavailable | uploads and results fail; dispatch unaffected | as today |
| leaked pod URL | 401 without the internal token | `trust-edge` |
| forged verdict header | dropped at the edge; a worker needs the internal token to trust it | — |
| key revoked | refused within 15 s | §10 Q4 |
| request > 100 s on a sync endpoint | 524 from the Runpod proxy on the edge → front hop | unchanged from today; the async APIs are the documented path |
| family DO overload | > ~1,000 req/s per family | the edge never asks the DO per request (registry snapshot); a DO sees enqueues, sessions, waits; shard by (family, region) if ever needed (#13 §11) |

## 8. Cost and latency

**Running cost** (estimates; prices as in gateway-cloudflare.md §2):

| | gateway | edge |
|---|---:|---:|
| front | one CPU pod `cpu3c` $0.06/h ≈ **$44/month per cluster** (≈ $88 with a second replica), plus its pod disk | Workers Paid **$5/month** base for the account (already paid for fv-control and the staging edge); 100k jobs/month at ~35 requests per job (submit, ~30 status polls, result, download redirect) ≈ 3.5 M requests, inside the 10 M included |
| CPU | in the pod | ~1–3 ms per routed request, ~3.5–10 M ms/month, inside the 30 M included; upload PUTs stream (I/O, little CPU) |
| DO | — (gateway in-process) | enqueue, session, job-wait and registry requests ≈ 5–10 per job, ~1 M/month: at or just over the included 1 M ($0.15/M beyond) |
| D1 | the gateway's reads and writes | the same rows written by the fronts instead; the edge adds key reads (cached) and `last_used_at` writes (≤ 1/min/key) |
| total | **$44–88/month per cluster** | **≈ $0–2/month** on top of the existing $5 |

Both are noise next to the GPUs (one RTX PRO 6000 ≈ $1,500/month). Cost is
not the reason for the move; one fewer pod to start (the gateway pod's image
pull and boot are on every cluster's critical path), one fewer hop and one
fewer process to operate are.

**Latency** (to be measured in stage 3):

- Submit: client → edge → Runpod proxy → front, against client → Runpod
  proxy → gateway pod. Then front → DO enqueue (~20–30 ms from EU, #13) and
  the push, against gateway → Runpod proxy → worker (40–150 ms, §1 of
  gateway-cloudflare.md). Expected: queue time at or below the DO path's
  measured 0.49 s p50 (gateway-cloudflare.md §9.8, which still had the
  gateway's hop in it).
- Status polls: from the sticky front's memory, one edge hop plus one Runpod
  proxy hop, as the gateway pod.
- Completion: pushed through the DO job wait, as the gateway's `follow`.
- Session admission: one DO round trip (0.2 s measured on staging, #13
  §14.2), then signalling through the edge.

## 9. Stages

One PR per stage, each green before the next.

### Stage 1 — edge parity

- `fastvideo-dispatch-proto`: `front` module (classification from the
  generated route table, key verdict, quota decisions, front choice with
  rendezvous hashing, registry types); `EnqueueReq.{owner, max_queued}`;
  `hello` routes (names, fal apps, protocols, defaults, Reactor flag, front
  URL, failed models); the job wait; DO-side failing of D1 rows.
- `fastvideo-edge`: the public front (routing, auth, quotas, forwarding,
  sessions and signalling proxy, `/uploads` to R2, admin keys over the D1
  binding, `/fv/v1/status`, `/fv/v1/capabilities`, health, `/metrics`, the
  families view); a `Registry` DO; D1 access behind a small trait (binding in
  the Worker, HTTP to `fv-d1-mock` in the native stand-in).
- `fastvideo-serve`: `auth.mode = trust-edge`; `dispatch.front` with
  `FrontGate` (the gateway's `plan_inputs`, `stage_for_retry`,
  `edge_enqueue`, cancel, insert-behind and read-through view moved out of
  `gateway/`, which keeps working on top of them); `dispatch.endpoint`;
  the upload ticket key and R2 upload resolution; the 424 fallback URL; the
  `DirectStore::url_issued` fix.
- Tests: units; `tests/edge_front.rs` (native edge stand-in + fake workers
  in front mode + `fv-d1-mock` + the S3 mock: every row of §3 marked I);
  `tests/compat/run.sh` with `FV_COMPAT_EDGE=1` (every suite); the same
  integration tests against `wrangler dev` where they need no stand-in
  internals, then against the staging Worker. `FV_SERVE_HEAVY=1
  scripts/serve/check.sh` on the build pod.
- Exit: all of the above green; the edge staging deploy from the PR head
  only if the owner asks (otherwise after merge).

### Stage 1 as built

Code: `fastvideo_dispatch_proto::front` (classification, verdicts, the
registry view and front choice, quotas, session bindings, merges,
capabilities); `fastvideo_edge::{front, keys, registry}` (the Worker's
public front, admin keys over D1, the `Registry` DO) next to the family DO
in `edge.rs`; `fastvideo_serve::front` (`FrontGate`, the envelope and
store moved out of `gateway/`, which re-exports them); the native stand-in
`fastvideo_serve::edge_host` and its binary `fv-edge-local`. Deploy:
`scripts/serve/cf-edge.sh` adds the `REGISTRY` binding (migration `v2`) and
the edge's vars.

Answers to §10 (each a switch the owner can flip):

| Q | Default | Switch |
|---|---|---|
| 1 serverless pools | dropped (fronts must be reachable workers) | — (stage 4 removes them) |
| 2 clusters | one edge per cluster; staging uses the edge's D1 `fv-edge-staging` and its outputs bucket, shared with the workers (never `fv-jobs`) | fv-control (stage 2) |
| 3 WHIP | the offer is proxied, `Location` rewritten to the edge | `FV_EDGE_WHIP=redirect`: 307 to the admitted worker with `?fv_cap=` (HMAC of the caller's verdict keyed by the internal token, 6 h); the worker's token layer takes it on `/fv/v1/streams/ingest*` only and carries it on the answer's `Location` |
| 4 revocation | 15 s key cache per isolate; a revoke at the edge (or `POST /fv/v1/admin/keys/invalidate`, admin token, for keys revoked elsewhere) bumps the registry's `key_epoch`, and every isolate drops its key cache on its next registry read (≤ 2 s); no per-request DO round trip | — |
| 5 console | served at the edge URL (forwarded to a front) | — |
| 6 `/v1/models`, `/fal/schema` | merged across families at the edge | — |

Where the build differs from the design above:

- **Uploads stay on the front that issued them.** Upload tokens carry the
  issuing worker's tag (`{tag}.…`); the edge routes `PUT /uploads`,
  `/files` and `/fv/v1/internal/uploads` by that tag, and a job that lands on
  another front fetches the input through the edge
  (`GET /fv/v1/internal/uploads/{token}`, internal token). No R2 staging
  of client uploads in stage 1; results still go straight to R2.
- **The per-key rate counts submits only** (`key_rpm`, default 300/min),
  not polls: the compat suites poll faster than any sane submit limit.
  `key_in_flight` (default 30) comes from the family objects' `owners`
  counts; per-model `max_queued` is enforced in the DO (`EnqueueReq`).
  Limits are per isolate (a fixed one-minute window), so a key spread over
  many colos gets more; good enough for abuse control, not billing.
- **Sessions**: admitted through the family DO and bound in the registry by
  the id clients use (`director:{sid}`, `reactor:{key|ip}`, `stream:{id}`,
  `ingest:{id}`); follow-ups renew the lease, stop/DELETE releases it. When
  admission is refused the edge first asks the workers whether their bound
  director sessions are still alive (`/wma/session/heartbeat`) and
  reclaims dead ones. In `FV_EDGE_WHIP=redirect` mode the edge never sees
  the ingest id, so that lease ends on its TTL or when the worker reports
  the session ended.
- **Revocation** is the epoch above, not a per-request check.
- `wrangler dev` covers the Worker's own paths (keys, registry, a job
  through a front); the protocol suites run against the native stand-in
  (`FV_COMPAT_EDGE=1`), which shares every decision with the Worker through
  `fastvideo_dispatch_proto::front`.

### Stage 2 — fv-control

- `control_plane: "gateway" | "edge" | "both"` (§5); `up`, `down`, `scale`,
  `roll`, `restart` without a gateway; the `register` phase; worker env
  (§5.2) and the self-delete watchdog in `WORKER_BOOT`; mint / list / revoke
  keys and the admin token through the edge; the collector on the families
  view; the dashboard card (§5.3); releases on fv-control.
- Tests: `control/test/unit` (payloads, spec validation, env for each
  mode), `control/test/integration/run.mjs` (an `edge` cluster on the
  fake-engine CPU worker: start → register → keys → scale → roll → stop,
  against a local edge stand-in), the UI smoke.
- Exit: green; fv-control staging deploy from main (or the PR head for
  stage 3, said in the report).

### Stage 3 — live test (owner-approved)

- fv-control staging launches one `edge` cluster in EUR-IS-1 (EU volume
  `jg48s6o1w0`): pools `h3-turbo` (fasth3) and `ltx` (ltx25-distill-sol),
  one RTX PRO 6000 each, `cap_s` 5400, both pods with their own backstop
  (§3.9) and fv-control's deadline.
- Against the edge URL: fal-py, fal-js, openai, minimax, native (compat
  suites, or their representative subsets pointed at a remote URL: queue,
  status, SSE, result by id with the 1.1 s wait across two fronts, cancel,
  webhook, upload), the director over WebRTC on h3-turbo (`fal.realtime`
  `wma`) and a Reactor session (fasth3 is the Reactor model).
- Latency: switch the cluster to `both` for 20–30 minutes (one CPU gateway
  pod on the same family DOs) and run the same 10 text-to-video jobs and 30
  status polls per path; record submit time, queue time (`timings.queue`),
  time to `COMPLETED` (poll and SSE).
- Budget: 2 × RTX PRO 6000 at ≈ $2.09/h for ≤ 1.5 h ≈ **$6.3**, plus the
  CPU gateway pod ≈ $0.05: **≤ $7**, under the $10 cap. Start only at a
  balance ≥ $15; stop before it would drop below $8. Nothing left idle;
  everything deleted at the end (`GET /pods/<id>` → 404 for each).
- Exit: every listed check passes; numbers recorded in this document.

### Stage 4 — remove the gateway

Only after stage 3 passes:

- Delete `crates/fastvideo-serve/src/gateway/*` (what the fronts need has
  moved in stage 1), `engine.backend = "remote"` and its config
  (`[gateway]` keys that only the gateway reads, `[[pools]]`), `autoscale.rs`
  (the in-gateway hook) and the `HttpGatewayPools` source, the gateway
  routes in `multiworker.rs`, `configs/serve/gateway*.toml`, the
  `serve-gateway` / `gateway-build` image stages (keeping a CPU fake-engine
  `cpu` variant for workers, gap 6) and the `gateway` variant in CI and
  `release.sh`, `scripts/serve/runpod-gateway.sh`, `runpod-cluster.sh`
  (fv-control is the launcher), `edge-gpu-test.sh`, the gateway parts of
  `do-family-pod.sh`, `tests/{gateway,gateway_burst,gateway_bases,
  edge_dispatch,job_overhead (gateway rows)}.rs` (rewritten on the edge
  stand-in in stage 1), the `FV_COMPAT_GATEWAY` mode, and fv-control's
  `gateway-base.ts`, `createGateway` / `patchGateway`, the `gateway` /
  `both` modes and their fixtures.
- Keep: the worker's internal routes and Runpod queue handler, `gw_*` D1
  tables (left in place, unused; dropping tables is a data change the owner
  should decide), the per-pool DO routes (`/pools/…`) for one release.
- Docs: gateway.md and gateway-cloudflare.md get a "retired" banner and
  point here; references in design.md, console.md, releases.md, images.md,
  e2e/*.md and docs/control are updated.

## 10. Questions for the owner

1. **Serverless pools.** The edge design has no serverless pools (a front
   must be a reachable worker; Runpod's own queue cannot hold our DO's
   jobs). fv-control never launches them. Drop them with the gateway
   (recommended), or keep a DO → Runpod `/run` path for them as a later
   stage?
2. **One edge, how many clusters?** Recommended for v1: one edge deployment
   serves one fleet (staging: one `edge` cluster at a time; production: one),
   fv-control refuses a second; per-cluster namespaces (DO names and a path
   or host prefix) later if needed. Also: staging should use its own D1 and
   output bucket shared by the staging edge and the staging cluster's
   workers (fv-control env overrides for `FV_D1_DATABASE_ID`,
   `FV_R2_BUCKET`), not `fv-jobs`. OK?
3. **WHIP hand-off**: proxy the SDP offer through the edge (one origin, works
   for every client), or answer 307 to the worker with a session capability
   (media and signalling both direct; WHIP clients follow redirects, but
   browsers then need CORS on the worker)? Recommended: proxy by default,
   307 behind a flag.
4. **Key revocation**: ≤ 15 s (cache TTL) acceptable, or immediate (a
   revocation epoch in the registry DO checked on every request, one DO
   round trip per request from far colos)?
5. **The console**: served at the edge URL from the fronts (recommended:
   same-origin API calls keep working), or hosted by fv-control calling the
   edge cross-origin?
6. **`/v1/models` and `/fal/schema`**: answered by one front (its models),
   or merged across families at the edge like `/fv/v1/capabilities`?

## Sources

Code read for §3: `crates/fastvideo-serve/src/gateway/{mod,dispatch,edge,
proxy,routes,tick,store,scale,runpod,schema}.rs`, `app.rs`, `worker.rs`,
`edge_link.rs`, `upload.rs`, `multiworker.rs`, `router.rs`, `releases.rs`,
`flags.rs`, `admin_token.rs`, `console.rs`, `streams.rs`, `ingest.rs`;
`crates/fastvideo-serve-kit/src/{auth,keys,uploads}.rs`, `d1/store.rs`;
`crates/fastvideo-fal/src/queue.rs`; `crates/fastvideo-protocol/src/http.rs`;
`crates/fastvideo-minimax/src/limits.rs`; `crates/fastvideo-edge/src/edge.rs`;
`crates/fastvideo-dispatch-proto/src/lib.rs`; `configs/serve/gateway*.toml`;
`control/src/cluster/{spec,payloads,ops,do,gateway-base}.ts`,
`control/src/collector.ts`, `control/src/releases.ts`;
`docker/gpucheck.Dockerfile`; `tests/compat/run.sh`.
