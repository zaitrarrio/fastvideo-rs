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
    #[serde(default)]
    pub submitted_total: u64,               // jobs ever dispatched here (monotonic, D1)
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
  (label `state`), `fv_pool_available`, `fv_pool_submitted_total`, all
  labelled `pool`.
- Pods the autoscaler starts join the pool by themselves (§5.3); a pod
  going away should first be set `draining` so no new work is dispatched
  to it: the worker does it on SIGTERM, and on
  `POST {worker}/fv/v1/internal/drain` (internal token; `…/undrain`
  reverses it). A draining worker refuses new jobs and sessions (503),
  finishes what it holds, reports `draining` in `/fv/v1/internal/status`
  and its `gw_workers` row; gateways skip it when dispatching. The reply
  is `{worker_id, draining, running, queued, sessions}`.
- Arrivals: `submitted_total` counts every job dispatched to the pool
  (one per job, re-dispatches excluded), from `gw_dispatch`, so every
  replica reports the same monotonic value.
- D1 tables (gateway-owned, read-only for the autoscaler): `gw_dispatch`,
  `gw_sessions`, `gw_workers` (schema in `gateway/schema.rs`).

<!-- BEGIN §8 Autoscaling: owned by crates/fastvideo-autoscale -->

## 8. Autoscaling (`crates/fastvideo-autoscale`)

One controller steers every pool from the §7 metrics. It runs inside the
gateway process (`[autoscale]`, in-process `PoolScaler`) or as its own
process (`fv-autoscale`, feature `runpod`), and only the holder of a D1
lease row acts, so several gateway replicas do not fight.

### 8.1 Pieces

| Module | Role |
|---|---|
| `policy` | Pure and deterministic: time, signals, observation and balance in; one decision per pool out. Unit-tested on a simulated clock. |
| `provider::runpod` | `RunpodServerless` (primary path: Runpod scales, we steer the endpoint), `RunpodPods` (pods from a template), `RunpodBalance`, `RunpodHealthSignals` (signals from `/health`, no gateway), `HttpGatewayPools` (`GET /fv/v1/gateway/pools`), `GatewayWorkers` (per-pod load from `gw_workers`). |
| `sim` | The simulated world and provider, traces, and the harness behind `fv-autoscale-sim`. |
| `gateway` | Mirror of §7 `PoolMetrics` (same JSON) and `GatewaySignals`, the sink `PoolScaler::observe` feeds. |
| `controller`, `lease`, `admin` | The loop, the D1 lease (`fv_autoscale_lease`, one conditional upsert), and `/fv/v1/admin/autoscale`. |

Gateway wiring (`crates/fastvideo-serve/src/autoscale.rs`, feature
`http-client`): with `[autoscale] enabled` in gateway mode, `App::build`
registers a `PoolScaler` that turns each tick's `PoolMetrics` into
`GatewaySignals`, fills an empty `serverless.endpoint_id` from the matching
`[[pools]]` entry (every autoscale pool must name a gateway pool), uses the
gateway's D1 connection for the lease and for `gw_workers`, the Runpod key
of `gateway.runpod_api_key`, and merges `/fv/v1/admin/autoscale` behind the
admin token. `fv-autoscale` runs the same controller outside the gateway
(signals from `GET /fv/v1/gateway/pools`, or from Runpod `/health`).

### 8.2 Policy (per pool, every `interval_s`)

1. **Demand** in workers: the larger of
   - the **queue need**: busy workers plus what starts every queued job
     within the SLO once capacity is ready (a busy worker takes
     `floor(SLO/D)` more jobs, any other `floor(SLO/D)+1`, `D` = recent job
     duration, EWMA of the gateway's `run_time.mean_s`);
   - the **steady state** `λ·D / (0.8 · jobs_per_worker)` plus streams
     (`λ` measured over at least 60 s, so a burst counts as queue).
2. **Cold-start prediction**: the queue when a worker started now is
   ready is `Q + max(λ − μ, dQ/dt) · cold_start` (cold start: configured,
   then the EWMA of measured starts). If that queue would miss the SLO, the
   demand covers it now.
3. **Queue-age breach**: the oldest job waited more than
   `slo_breach_fraction · SLO` and no idle or booting worker will take it →
   one more worker, cooldown bypassed.
4. **Floors and caps**: `max(min_workers, warm_min, schedule)` up to
   `min(max_workers, pool budget / price)`; never below the busy workers.
5. **Steps**: rate- and prediction-driven growth by at most
   `scale_up_step` per `scale_up_cooldown_s`; queued jobs and floors are met
   at once. Scale-down only after the work in hand fits one worker fewer
   and the steady demand leaves `hysteresis` headroom for `idle_timeout_s`,
   then one step per `scale_down_cooldown_s`.
6. **Budgets**: global `$/hr` across pools (busy workers first, then floors,
   then the rest by `priority` and queue urgency). **Balance floor** ($8):
   below it the controller is in hard stop: serverless `workersMin = 0`,
   `workersMax = busy`; pods: idle ones drained, booting ones deleted, no
   new ones. An unknown balance blocks scale-ups.
7. **Actions**:
   - serverless: `workersMin = target`, `workersMax = cap` (never below the
     busy workers), plus the pool's `scalerType`/`scalerValue`/`idleTimeout`,
     PATCHed only when different. Runpod starts and reaps the workers: min
     raised ahead of load, max capped by budget, min back to 0 when idle.
     Recommended scaler: `QUEUE_DELAY` at SLO/2, so Runpod's own scaler is a
     backstop and does not start a worker per short job;
   - pods: `create` (GPU-type list × region+volume placements, e.g. US-CA-2
     with `s2k01690bi`, then EUR-IS-1 with `jg48s6o1w0`; out of stock →
     next; no `allowedCudaVersions` filter; a pod above 1.25 × the price table
     is deleted at once), `drain` (idle ready workers, oldest first), `delete`
     (draining with zero in-flight, re-checked right before `DELETE`;
     booting past `boot_timeout_s`), `undrain` (instead of a new pod). Pods
     past `max_lifetime_s` are replaced first and drained when the
     replacement is ready. A worker holding a job or a live stream is never
     deleted.

### 8.3 Configuration

`configs/serve/autoscale.toml` holds the full example; `gateway.toml` ends
with a delimited `[autoscale] enabled = false` block to replace with its
tables (or pass the file to `fv-autoscale --config`). Pool names are the §2
`[[pools]] id`s (`h3-turbo`, `wan`, `ltx`, `sfwan-live` in the example);
`kind = "runpod-serverless"` is accepted as `serverless`. Start with
`dry_run = true`, read the decisions in the log and on the admin route, then
`POST /fv/v1/admin/autoscale {"dry_run": false}`.

| Key | Default | |
|---|---|---|
| `enabled`, `dry_run` | false, true | dry run decides and logs, never writes |
| `interval_s` | 15 | tick |
| `budget_usd_per_hr` | 0 (none) | global cap |
| `balance_floor_usd` | 8 | hard stop |
| `prices` | table as of 2026-09-28 | pods: secure price; `serverless:<gpu>`: flex price |
| `lease` | memory | `d1` for replicas (`ttl_s` 60) |
| pool `min_workers` / `max_workers` / `warm_min` | 0 / 2 / 0 | |
| pool `schedule` | [] | `{days, start_hour, end_hour, min_workers}`, clock `schedule_utc_offset_min` |
| pool `jobs_per_worker`, `streams_per_worker` | 1, 1 | one batch job / live stream per GPU |
| pool `slo_queue_wait_s`, `slo_breach_fraction` | 60, 0.5 | |
| pool `scale_up_step`, `scale_up_cooldown_s` | 2, 30 | |
| pool `idle_timeout_s`, `scale_down_cooldown_s`, `scale_down_step`, `hysteresis` | 300, 120, 1, 0.2 | |
| pool `cold_start_s`, `default_job_s` | 130, 30 | until measured |
| pool `budget_usd_per_hr`, `price_usd_per_hr`, `priority` | 0, table, 0 | |
| `serverless` | | `endpoint_id`, `gpu_type`, `scaler_type`, `scaler_value`, `idle_timeout_s` |
| `pod` | | `template_id`, `gpu_types`, `placements`, `cloud_type`, `max_lifetime_s` (12 h), `boot_timeout_s` (1200), `url_template`, `ready_path`, `max_usd_per_hr` |

### 8.4 Observability

- Log line `autoscale decision` per change (pool, action, current, target,
  busy, demand, floor, cap, endpoint min/max, create/drain/delete,
  leader, dry run, reasons such as `queue: 4 queued, oldest 12s`,
  `predictive: queue 9.0 in 120s cold start`, `pool budget $9.00/h caps 2`).
- `GET /fv/v1/admin/autoscale` (admin token): config summary, leader,
  balance, the last tick (decision + observation + apply report per pool),
  per-pool estimates, the last `history` decisions. `POST` with
  `{"dry_run": false}` switches dry run off (`{"tick": true}` runs a tick).
- `/metrics`: `fv_autoscale_{target,demand,busy,floor,cap}_workers{pool}`,
  `fv_autoscale_workers{pool,state}`, `fv_autoscale_target_usd_per_hour{pool}`,
  `fv_autoscale_queued{pool}`, `fv_autoscale_oldest_queued_seconds{pool}`,
  `fv_autoscale_decisions_total{pool,action}`,
  `fv_autoscale_apply_errors_total{pool}`, `fv_autoscale_leader`,
  `fv_autoscale_dry_run`, `fv_autoscale_balance_usd`.

### 8.5 Requests to the gateway

- **Drain a pod**: §7 says a pod leaving should be `draining` first, and
  `gw_workers` is read-only for the autoscaler. `GatewayWorkers::drain`
  calls `POST {worker}/fv/v1/internal/drain` (internal token) best effort;
  the worker should set its `gw_workers` row to `draining` and keep it so
  (and `…/undrain`). Until then deletion relies on zero in-flight work
  (`running + sessions`) read right before `DELETE`.
- **Arrivals**: `PoolMetrics` has no arrival counter; the rate is estimated
  as `run_time.count / window_s` plus the growth of queued + running. A
  monotonic `submitted_total` per pool would make it exact.

### 8.6 Simulation

`cargo run -p fastvideo-autoscale --bin fv-autoscale-sim` (5 seeds pooled;
`--timeline <trace> <family>` prints one run per minute). Per-family
timings from the WP-18/19 runs: wan cold start 49 + 70 s, job 6.4 s; h3
80 + 52 s, job 24.4 s (+45 s on a worker's first job); ltx 79 + 33 s, job
40 s (+15 s first); 10 % of cold starts land on a host without the image
(458 s start). Serverless H100 at $4.18/h, max 4 workers. Average load per
family (assumption; the business plan's shape): wan 120, h3 60, ltx 40
jobs/h; *diurnal* peaks at 14:00 with the busiest hour 2.5× the daily
average; *spike* is 10× for 10 min; *idle→burst* is 20 jobs in 2 min after
2 h idle. Strategies: **autoscaler** (the pools of
`sim::harness::pool_for`: SLO 60 s wan / 120 s h3, ltx; Runpod scaler
`QUEUE_DELAY` at SLO/2, idle 60 s; diurnal adds a 10:00-19:00 floor of 1),
**runpod-only** (the endpoint at min 0 / max 4 with Runpod's console
defaults, `QUEUE_DELAY` 4 s, idle 5 s), **always-on N** (min = max = N).

| trace | family | strategy | jobs/run | wait p50 s | wait p95 s | ≤ SLO | GPU-h/run | $/run | cold starts/run (uncached) | max workers |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steady | wan | autoscaler | 713 | 0 | 7 | 99% (SLO 60s) | 6.40 | 26.74 | 4.0 (0.4) | 4 |
| steady | wan | runpod-only | 713 | 59 | 132 | 51% (SLO 60s) | 14.69 | 61.40 | 296.4 (26.6) | 4 |
| diurnal 2.5x | wan | autoscaler | 2905 | 0 | 13 | 100% (SLO 60s) | 20.80 | 86.94 | 13.6 (1.4) | 3 |
| diurnal 2.5x | wan | runpod-only | 2905 | 61 | 131 | 50% (SLO 60s) | 44.09 | 184.30 | 839.2 (81.6) | 4 |
| spike 10x/10min | wan | autoscaler | 641 | 0 | 67 | 94% (SLO 60s) | 5.23 | 21.86 | 7.0 (0.4) | 4 |
| spike 10x/10min | wan | runpod-only | 641 | 50 | 131 | 57% (SLO 60s) | 10.11 | 42.26 | 195.2 (19.0) | 4 |
| idle→burst 20 | wan | autoscaler | 20 | 115 | 138 | 0% (SLO 60s) | 0.72 | 2.99 | 4.0 (0.4) | 4 |
| idle→burst 20 | wan | runpod-only | 20 | 97 | 124 | 18% (SLO 60s) | 0.22 | 0.93 | 4.0 (0.4) | 4 |
| steady | h3 | autoscaler | 349 | 0 | 43 | 99% (SLO 120s) | 7.57 | 31.64 | 9.6 (0.4) | 4 |
| steady | h3 | runpod-only | 349 | 91 | 183 | 61% (SLO 120s) | 13.21 | 55.23 | 187.4 (17.4) | 4 |
| steady | h3 | always-on 2 | 349 | 0 | 4 | 100% (SLO 120s) | 12.00 | 50.18 | 2.0 (0.2) | 2 |
| diurnal 2.5x | h3 | autoscaler | 1469 | 0 | 26 | 99% (SLO 120s) | 29.75 | 124.36 | 27.4 (2.6) | 4 |
| diurnal 2.5x | h3 | runpod-only | 1469 | 72 | 169 | 68% (SLO 120s) | 40.82 | 170.65 | 517.0 (50.2) | 4 |
| diurnal 2.5x | h3 | always-on 2 | 1469 | 0 | 18 | 100% (SLO 120s) | 48.00 | 200.64 | 2.0 (0.2) | 2 |
| spike 10x/10min | h3 | autoscaler | 340 | 6 | 248 | 73% (SLO 120s) | 6.68 | 27.93 | 10.6 (1.0) | 4 |
| spike 10x/10min | h3 | runpod-only | 340 | 112 | 279 | 52% (SLO 120s) | 9.61 | 40.18 | 127.8 (10.6) | 4 |
| spike 10x/10min | h3 | always-on 2 | 340 | 0 | 570 | 72% (SLO 120s) | 8.01 | 33.47 | 2.0 (0.2) | 2 |
| idle→burst 20 | h3 | autoscaler | 20 | 214 | 245 | 0% (SLO 120s) | 0.87 | 3.65 | 4.0 (0.4) | 4 |
| idle→burst 20 | h3 | autoscaler, warm 1 | 20 | 162 | 197 | 21% (SLO 120s) | 3.61 | 15.09 | 4.0 (0.4) | 4 |
| idle→burst 20 | h3 | runpod-only | 20 | 189 | 222 | 0% (SLO 120s) | 0.38 | 1.58 | 4.0 (0.4) | 4 |
| idle→burst 20 | h3 | always-on 2 | 20 | 107 | 164 | 61% (SLO 120s) | 6.07 | 25.36 | 2.0 (0.2) | 2 |
| steady | ltx | autoscaler | 231 | 0 | 59 | 99% (SLO 120s) | 8.68 | 36.28 | 16.2 (1.4) | 4 |
| steady | ltx | runpod-only | 231 | 98 | 167 | 88% (SLO 120s) | 9.63 | 40.25 | 157.8 (13.6) | 4 |
| diurnal 2.5x | ltx | autoscaler | 944 | 0 | 48 | 99% (SLO 120s) | 30.64 | 128.09 | 40.8 (3.8) | 4 |
| diurnal 2.5x | ltx | runpod-only | 944 | 73 | 159 | 88% (SLO 120s) | 32.29 | 134.96 | 470.8 (45.2) | 4 |
| spike 10x/10min | ltx | autoscaler | 220 | 8 | 244 | 78% (SLO 120s) | 7.05 | 29.48 | 14.2 (1.2) | 4 |
| spike 10x/10min | ltx | runpod-only | 220 | 116 | 301 | 63% (SLO 120s) | 7.13 | 29.81 | 102.4 (9.4) | 4 |
| idle→burst 20 | ltx | autoscaler | 20 | 191 | 275 | 2% (SLO 120s) | 0.91 | 3.80 | 4.0 (0.4) | 4 |
| idle→burst 20 | ltx | runpod-only | 20 | 173 | 257 | 17% (SLO 120s) | 0.41 | 1.71 | 4.0 (0.4) | 4 |

Reading it:

- Steady and diurnal load: the autoscaler meets the SLO for ≥ 99 % of jobs
  (p95 7-48 s) at 44-95 % of runpod-only's cost, which pays a cold start
  (and a first-job penalty) per burst of arrivals and misses the SLO for
  a third to a half of the jobs. Against always-on sized for the peak it
  saves 25-40 % on h3 at a similar p95.
- Spike 10× for 10 min: both are capped by `max_workers = 4`; the
  autoscaler holds the workers it raised instead of dropping them between
  arrivals, so p50 stays near 0.
- Idle→burst: every strategy without a warm worker pays the cold start
  (p50 ≈ cold start + queue); the autoscaler costs more than runpod-only
  there (it keeps workers for `idle_timeout_s` after the burst). The lever
  is `warm_min` or a `schedule` floor (`autoscaler, warm 1`: p50 214 → 162 s
  for $15 over 3 h).

### 8.7 Live validation

2026-09-28, provider path (`fv-autoscale` standalone, signals from Runpod
`/health`; the gateway was not on main yet when the run started).

- Endpoint `fv-as-q-0928193752` (`5plvepwya58f25`, template `86oapybs4p`)
  from `runpod-endpoint.sh up` with `FV_ENDPOINT_PREFIX=fv-as`,
  `/etc/fv/runpod-wan.toml` (wan-turbo, `fastwan21-1.3b`), image
  `sha-7e82504`, GPUs H100 80GB HBM3 / H100 NVL / H200, US volume
  `s2k01690bi` (read only), created at min 0 / max 1, `QUEUE_DELAY` 1, idle 30 s.
- Pool: `max_workers = 2`, pool budget $9/h (cap 2), SLO 60 s, idle timeout
  120 s, scale-down cooldown 60 s, Runpod scaler `QUEUE_DELAY` 30, idle 30 s.

| step | time (UTC) | what happened (Runpod REST `GET /endpoints/{id}` and `/health`) |
|---|---|---|
| dry run, 4 jobs | 19:38:33 | decisions logged (`ScaleUp 0 → 2`: queue 4, prediction, queue age), **no PATCH**: the endpoint stayed min 0 / max 1 / scaler 1; Runpod's own scaler ran the 4 jobs on one worker (all `succeeded`, delay 135 s, execution 5.3 s) |
| live, controller starts | 19:41:59 | PATCH `{"scalerValue":30,"workersMax":2}` (max from the pool budget) |
| live, burst of 6 jobs | 19:42:03 | 19:42:08 PATCH `{"workersMin":2}` (queue 6 + prediction); two workers came up; all 6 `succeeded` (delay 77-93 s from zero, execution 5.3-6.3 s) |
| idle | 19:43:53 | last job done; `/health` kept `workers.running: 2` with `jobs.inProgress: 0` for minutes, which the first controller counted as busy (min pinned at 2). Fixed (busy = `min(running, inProgress)`, `a970484`) and restarted at 19:44:56 |
| scale down | 19:46:57 | PATCH `{"workersMin":1}` after 120 s idle |
| scale down | 19:47:57 | PATCH `{"workersMin":0}` after the 60 s cooldown |
| at zero | 19:50:32 | min 0 / max 2, workers idle 0, initializing 0, running 0 (2 `throttled` = unallocated) |
| cleanup | 19:51 | endpoint and template deleted; `GET` of both → 404; no `fv-as*` endpoint, template or pod left; no volume written |

Spend: ≈ 17 H100 worker-minutes (≈ 2.5 dry run, ≈ 2 × 7.5 live) ≈ $1.2 at
$4.18/h; the account balance went $48.03 → $46.74 over the run, which
includes other agents' pods.

<!-- END §8 Autoscaling -->
