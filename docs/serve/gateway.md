# fv-serve gateway: one entry point in front of per-family GPU pools

Status: design + implementation (2026-09-28). Owner of the gateway code:
`crates/fastvideo-serve/src/gateway/`. The autoscaler lives in
`crates/fastvideo-autoscale/` (another agent) and plugs in through the
interface in §7, which is the contract between the two.

Everything here is **native** (our own shapes); no external protocol field
is added. The public APIs (FastVideo, MiniMax, fal, LTX, Reactor, native)
are exactly those of design.md §4-§5: the gateway mounts the same adapter
crates and only replaces the engine behind them.

## 1. Shape

```
client ──(one URL, one API key)──► fv-serve gateway ×N (CPU, stateless)
                                     │  auth, normalize, ingest, negotiate,
                                     │  job row in D1, dispatch, status/result
                                     │  from D1, signalling proxy
                     ┌───────────────┼──────────────────────┐
             pool h3-turbo      pool wan-turbo        pool sfwan-live …
         (Runpod serverless)  (Runpod serverless)       (pods)
          /run kind:http        /run kind:http     POST /fv/v1/internal/jobs
                     └───────────────┴──────────────────────┘
                 workers: ordinary fv-serve (`server.role = "worker"`)
                 write progress/results to the same D1 + R2
media (WebRTC) ◄──────────────── direct client ↔ worker (pods) / WHIP → SFU
```

- The gateway is `fv-serve` with `[engine] backend = "remote"`: no GPU, no
  `cuda` feature, needs `http-client`. It runs the full HTTP surface
  (every API, `/console`, `/fv/v1/*`, uploads, `/files`) over a
  `RemoteGate` (serve-kit's `EngineGate` seam, design §3.5-§3.6) instead of
  the in-process `EngineService`.
- One **pool** per model family / tier (`[[pools]]`): a Runpod serverless
  queue endpoint, or a list of pods (static URLs plus self-registered
  workers, §5.3).
- **State lives in D1 and R2 only** (design §0 decision 7). The gateway
  holds caches (caps, pool health, metrics) but no authoritative state, so
  several gateway replicas can run behind one load balancer.

## 2. Configuration

Gateway (`configs/serve/gateway.toml`):

```toml
[engine]
backend = "remote"                 # gateway mode

[gateway]
# internal_token: FV_INTERNAL_TOKEN (secret shared with every worker; never a user key)
caps_refresh_s = 60                # live caps refresh per pool
tick_s = 5                         # metrics + reaper tick
runpod_api_base = "https://api.runpod.ai/v2"   # tests: the local simulator
# runpod_api_key: FV_RUNPOD_API_KEY (else RUNPOD_API_KEY)
watch_poll_ms = 1000               # D1 poll for SSE / sync waits

[[pools]]
id = "h3-turbo"
kind = "runpod-serverless"         # | "pod"
endpoint_id = "…"                  # serverless (FV_POOL_H3_TURBO_ENDPOINT overrides)
urls = []                          # pod: worker base URLs (plus registered workers)
max_queued = 32                    # admission: queued jobs in D1 for this pool
max_streams = 0                    # 0: streams not served by this pool
dispatch_timeout_s = 30            # one dispatch request
job_timeout_s = 3600               # envelope wait (serverless) / reaper cap
stale_after_s = 90                 # running job without a worker heartbeat → lost
retries = 1                        # re-dispatch after worker loss (0: fail at once)
aliases = { "MiniMax-H3-Turbo" = "fasth3" }
# Static caps (the fallback while the pool is scaled to zero): the same
# entries a worker's [[models]] has, resolved through the CUDA catalog
# (built on CPU), or fake model ids for tests.
[[pools.models]]
id = "fasth3"
family = "h3"
recipe = "h3-turbo"
```

Worker (any existing config plus):

```toml
[server]
role = "worker"                    # FV_SERVE_ROLE=worker
# public_base_url = the gateway's public URL, so any URL a worker renders
# (webhook bodies, fal URLs) points at the gateway.
[gateway]
pool = "h3-turbo"                  # FV_GATEWAY_POOL (self-registration, §5.3)
# internal_token: FV_INTERNAL_TOKEN
[jobs]
heartbeat_s = 10                   # D1 heartbeat of unfinished jobs (worker-loss detection)
```

Shared secrets between gateway and workers: `FV_INTERNAL_TOKEN`, the D1 and
R2 set (`FV_CF_*`, `FV_D1_DATABASE_ID`, `FV_R2_*`), `FV_URL_SIGNING_KEY`
and `FV_WEBHOOK_ED25519_KEY` (workers send the webhooks; receivers verify
them against the gateway's `/.well-known/jwks.json`). User API keys
(`FV_API_KEYS`, minted keys) live only on the gateway.

## 3. Batch jobs

Submit on the gateway (the generic serve-kit pipeline, unchanged):
auth → normalize → admission → model resolution against the **aggregated
caps** → ingestion (inputs staged on the gateway) → `negotiate` → job row
inserted in D1 (`queued`, owner, callback, request echo) → `RemoteGate::submit`:

1. **Pool choice**: the model's entry in the aggregated `CapabilityTable`
   lists the pools serving it (the "executors" of the entry); the first
   available one with admission room is used.
   - No pool config serves the model → the usual 400/404 "model not
     served" of that API.
   - Every pool serving it is unavailable (pods unreachable, endpoint
     errors) → `503` + `Retry-After` (`ApiError::loading`).
   - Pool over `max_queued` → `429 QueueFull` + `Retry-After`.
2. **Inputs**: staged files are put into the artifact store (R2 in
   production) under a fresh id and passed as signed URLs; the worker
   downloads them into its own inputs dir and deletes them when the job
   ends.
3. **Envelope** (native), the same for both pool kinds:

   ```jsonc
   POST /fv/v1/internal/jobs
   {"job": <Job>, "inputs": {"<gateway path>": "<signed url>", …},
    "attempt": 1, "pool": "h3-turbo"}
   ```

   - `pod`: sent directly to a worker (`x-fv-internal-token`), tried in
     order of least in-flight work; a refused/unreachable worker moves on
     to the next.
   - `runpod-serverless`: `POST {runpod_api_base}/{endpoint}/run` with the
     existing queue envelope `{"input":{"kind":"http","method":"POST",
     "path":"/fv/v1/internal/jobs","body":…,"wait":true,
     "poll_path":"/fv/v1/internal/jobs/{job id}","cancel_path":…}}`. The
     queue worker dispatches it in-process (queue-originated requests are
     already authenticated by Runpod, so the token is added by the worker
     itself and never appears in the job input); `wait` keeps the Runpod
     job `IN_PROGRESS` until the engine job ends, so Runpod's own
     concurrency and scaling see a busy worker.
4. **Dispatch record**: `gw_dispatch` row (job id, pool, kind, target =
   pod URL or endpoint id, ref = Runpod job id or worker id, attempt,
   state). Any gateway replica can cancel or reap from it.
5. **Worker side**: `POST /fv/v1/internal/jobs` adopts the job
   (`D1JobStore::adopt`: the worker's cache becomes authoritative and the
   row's `worker` column its id), stages the inputs, submits to its engine.
   From then on it is an ordinary job on that worker: progress (≤ 1 write/s),
   logs, artifacts (R2), terminal state and callbacks/webhooks all come
   from the worker. Idempotent: a duplicate delivery to the same worker
   answers the held job; a job already adopted by another live worker is
   refused (409).

Status, results, lists, fal `status/stream` (SSE), sync endpoints: the
gateway answers them from D1 (`GatewayJobStore`: read-through, `watch()`
polls D1 every `watch_poll_ms`). Signed artifact URLs come from the shared
R2 store (or a shared local directory in single-host tests).

**Cancel** (`DELETE`/`cancel` routes of every API): the gateway marks the
row (queued → cancelled at once; running → `cancel_requested`) and forwards
it: pod → `DELETE {url}/fv/v1/internal/jobs/{id}`; serverless → Runpod
`/cancel/{ref}`: a queued Runpod job is dropped (the gateway marks the job
cancelled), a running one reaches the worker as job-stop, whose dispatcher
calls the envelope's `cancel_path` in-process.

**Worker loss** (the reaper, every `tick_s`, any replica, idempotent):

- A `queued` job still in a Runpod queue has no worker heartbeat; the reaper
  asks Runpod `/status/{ref}` and touches the row while it is `IN_QUEUE` /
  `IN_PROGRESS` (before adoption).
- A job whose row heartbeat (`updated_at`, bumped by the worker every
  `jobs.heartbeat_s`) is older than the pool's `stale_after_s`, or whose
  Runpod job ended (`FAILED`, `COMPLETED`, `CANCELLED`, `TIMED_OUT`, not
  found) while the row is unfinished, is **lost**: with `attempt ≤ retries`
  it is reset to `queued` (worker cleared, log line) and dispatched again
  (another pod, or a new Runpod job; the old ref is cancelled best effort);
  otherwise it is failed `"the worker running this job was lost"` (which
  fires its webhook through the gateway).

## 4. Capabilities

- Per pool: **static** caps from `[[pools.models]]` (CUDA catalog, same
  resolution as a worker's `[[models]]`) or `fake_models`; **live** caps
  from `GET /fv/v1/internal/status` of a worker (pods; serverless only when
  Runpod `/health` reports an idle ready worker, via a `kind:http` `/runsync`,
  so a caps refresh never wakes a scaled-to-zero pool). Live caps replace
  static ones when fetched; cached per replica for `caps_refresh_s`.
- The gateway's table is `CapabilityTable::build` over the pools (one
  "executor" per pool), so tier aliases (`h3-turbo`, `ltx-max`, …) bind as
  on a single server, `[aliases]` and `pools[].aliases` merge on top.
- `/fv/v1/capabilities` adds `pools` (id, kind, available, models, workers)
  and each model's `pools`; the console reads the same endpoint, so it
  lists every model of every pool.

## 5. Streaming

WebRTC media never passes through the gateway.

### 5.1 Peer sessions (fal director, Reactor): pods only

The gateway authenticates the caller, picks a worker of the model's pod
pool with a free session slot, proxies the **signalling** HTTP to it (with
the internal token, the user's `Authorization` stripped) and records a
**lease** in D1 (`gw_sessions`: session id → pool, worker URL, owner).
The worker's answer SDP carries its own ICE candidates (public IP / ICE-TCP
port), so media flows client ↔ worker directly.

- fal director: `/wma/ice`, `/{app}/director/ice` → any worker of the
  app's pool; `/wma/session` → allocate + proxy + lease on the returned
  `session_id`; `/wma/session/heartbeat` → by lease; `/start-session`
  (SSE) → allocate + streamed proxy; `/info` → any worker.
- Reactor local runtime: `/start_session` allocates a worker (the pool of
  `[gateway] reactor_model`) and leases it to the caller (API key owner,
  else the client address); `/session`, `/stop_session`, `/events` (SSE)
  and `/schema` follow the caller's lease; `/sessions/{sid}/…` follow the
  lease of `sid`. `/stop_session` ends the lease.
- Serverless pools cannot take inbound WebRTC: peer sessions on them answer
  `503` with the reason.

### 5.2 Native WHIP streams (`/fv/v1/streams`)

- pod pool: proxied to a worker (lease on the stream id), media goes
  worker → WHIP endpoint (SFU).
- serverless pool: a `kind:stream` Runpod job (the worker publishes over
  WHIP); the gateway answers `201` with its own stream id at once
  (`state: "starting"`), `GET` reads the Runpod job's progress/output
  (`live` → session stats), `DELETE` cancels the Runpod job.

### 5.3 Pod workers register themselves

A worker with `server.role = "worker"`, `gateway.pool` and a public base URL
upserts `gw_workers (pool, worker_id, url, state, sessions, running,
updated_at)` every 10 s (`draining` on shutdown). Gateways use static
`urls` plus registered workers seen in the last 45 s. This is what lets an
autoscaler add or remove pods without touching the gateway.

## 6. Replicas, health, route filter

- `/ping`: 200 when at least one pool is available (or has static caps and
  can scale from zero), 503 while draining; `/healthz` and `/health`
  include per-pool state. `/fv/v1/gateway/pools` (admin token) returns the
  metrics of §7.
- Several replicas: every gateway route reads D1/R2, so with
  `server.workers_max > 1` (design §6.5) the gateway serves jobs, cancel
  and delete (routed through `gw_dispatch`), fal `status/stream` (D1 poll)
  and streaming signalling (leases in D1). Uploads (`/uploads`,
  `/v1/upload`, fal storage initiate) and `/fal/proxy` stay pinned (the
  upload store is replica-local disk), and `/files` needs R2.
- Workers behind a gateway always run `workers_max = 1` semantics per
  worker (the gateway addresses each worker, or Runpod's queue does).

## 7. Autoscaler interface (contract with `crates/fastvideo-autoscale`)

In `fastvideo_serve::gateway::scale` (no other gateway type needed):

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PoolKind { RunpodServerless, Pod }

/// Summary of durations (seconds) over a window.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DurationStats { pub count: u32, pub mean_s: f64, pub p50_s: f64, pub p90_s: f64, pub max_s: f64 }

/// Workers of a pool as the gateway sees them (serverless: Runpod /health;
/// pods: registered + static workers and their probes).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerCounts { pub total: u32, pub ready: u32, pub busy: u32, pub idle: u32,
                          pub initializing: u32, pub unhealthy: u32 }

/// One pool at one instant (all numbers from D1 unless noted).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolMetrics {
    pub pool: String,
    pub kind: PoolKind,
    pub endpoint_id: Option<String>,        // serverless
    pub at_unix_ms: i64,
    pub queued: u32,                        // dispatched to this pool, not started
    pub running: u32,                       // started, not finished
    pub oldest_queued_age_s: f64,           // 0 when none queued
    pub streams: u32,                       // live stream / peer-session leases
    pub run_time: DurationStats,            // started→finished, jobs finished in `window_s`
    pub queue_wait: DurationStats,          // created→started, same jobs
    pub window_s: u64,
    pub workers: WorkerCounts,
    pub available: bool,                    // the gateway would dispatch now
    pub max_queued: u32,                    // admission limit (0 = none)
    pub max_streams: u32,
}

/// Implemented by the autoscaler; called after every metrics tick
/// (`gateway.tick_s`) with every pool, on one replica at a time
/// best effort (each replica calls its own hooks).
#[async_trait::async_trait]
pub trait PoolScaler: Send + Sync + 'static {
    async fn observe(&self, pools: &[PoolMetrics]);
}
```

- In process: `Overrides { scalers: vec![Arc<dyn PoolScaler>], .. }` or
  `app.gateway().unwrap().add_scaler(..)`.
- Out of process: `GET /fv/v1/gateway/pools` (admin token,
  `Authorization: Bearer <FV_ADMIN_TOKEN>`) →
  `{"object":"fv.gateway.pools","pools":[PoolMetrics…]}`, and Prometheus
  gauges on `/metrics`: `fv_pool_queued`, `fv_pool_running`,
  `fv_pool_oldest_queued_seconds`, `fv_pool_streams`, `fv_pool_workers`
  (label `state`), `fv_pool_available`, all labelled `pool`.
- Pods the autoscaler starts join the pool by themselves (§5.3); a pod
  going away should first be set `draining` (the worker does it on
  SIGTERM) so no new work is dispatched to it.
- D1 tables (gateway-owned, read-only for the autoscaler): `gw_dispatch`,
  `gw_sessions`, `gw_workers` (schema in `gateway/schema.rs`).
