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
watch_poll_ms = 1000               # D1 poll for SSE / sync waits; freshness of the status view (§3.3)
inline_inputs_max_bytes = 8388608  # inputs up to 8 MiB per job ride in the dispatch (§3.1; serverless ≤ 6 MiB)
input_passthrough = true           # large video/audio given as a public URL: the worker fetches it (§3.1)
stage_inputs_for_retry = true      # copy inputs to R2 after the dispatch, for a re-dispatch (§3.1)

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
dispatch = "gateway"               # pod pools: or "durable-object" + do_url (push through a
                                   # Cloudflare Durable Object; gateway-cloudflare.md §10)
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
# [dispatch] do_url = "https://…"  # a pool with dispatch = "durable-object": the worker's
                                   # socket to the pool's DO (gateway-cloudflare.md §10)
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
2. **Inputs** (§3.1): small ones travel inside the envelope (base64), large
   video/audio a client gave as a public URL are fetched by the worker from
   there, anything else goes through the artifact store (R2) with a signed
   URL. The worker puts them in its own inputs dir, all at once.
3. **Envelope** (native), the same for both pool kinds:

   ```jsonc
   POST /fv/v1/internal/jobs
   {"job": <Job>,
    "inputs": [{"path": "<gateway path>", "kind": "image", "bytes": 1769472,
                "inline": "<base64>"},                     // or
               {"path": …, "source": "https://…", "sha256": "…"}, // or
               {"path": …, "url": "<signed url>", "artifact": {…}}],
    "attempt": 1, "pool": "h3-turbo"}
   ```

   The worker accepts envelopes up to 64 MiB.

   - `pod`: sent directly to a worker (`x-fv-internal-token`), the one
     where the job starts soonest, reserved before the call (§3.4); a
     refused/unreachable worker gives its slot back and the next is tried.
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
5. **Worker side**: `POST /fv/v1/internal/jobs` puts the inputs in place,
   sets `Job.dispatched_at`, adopts the job (`D1JobStore::adopt`: the
   worker's cache becomes authoritative and the row's `worker` column its
   id) and submits to its engine. The adoption is **one D1 statement**: an
   upsert whose `DO UPDATE … WHERE` only applies when the row is queued or
   running, has no cancel request, and has no other worker with a fresh
   heartbeat, `RETURNING job`; no row back means refused (409). It keeps
   the row's fields and sets the worker's input paths and `dispatched_at`
   (`json_set`), so the race between two workers is decided inside SQLite.
   From then on it is an ordinary job on that worker: progress (≤ 1 write/s),
   logs, artifacts (R2), terminal state and callbacks/webhooks all come
   from the worker. Idempotent: a duplicate delivery to the same worker
   answers the held job; a job already adopted by another live worker is
   refused (409).

Status, results, lists, fal `status/stream` (SSE), sync endpoints: the
gateway answers them from D1 (`GatewayJobStore`: read-through, `watch()`
polls D1 every `watch_poll_ms`), through the in-memory view of §3.3. The
replica that dispatched a job to a pod worker also *follows* it: the
worker reports each status change as soon as D1 has it (§3.5).
Signed artifact URLs come from the shared R2 store (or a shared local
directory in single-host tests).

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

### 3.1 Input path

Measured on the cluster (2026-09-28, image-to-video): the gateway's R2 PUT
of the input took 1-3 s (once 11 s) and the worker's R2 GET about 1 s, all
before the GPU saw the job. The input does not need to be stored for the
job to run, so the store is off the submit path now:

| input | how it reaches the worker |
|---|---|
| all of a job's inputs together up to `inline_inputs_max_bytes` (8 MiB; serverless pools 6 MiB, a Runpod `/run` body is ≤ 10 MB) | **inline**: base64 in the envelope; the worker writes the bytes |
| a larger video or audio input the client gave as an `http(s)` URL (`Job.input_sources`; images are left out because ingestion may rewrite them upright) | **source**: the worker fetches the URL itself with ingestion's SSRF guard, redirect rules and size/time limits, and checks the gateway's SHA-256; if that fails the worker answers **424** and the gateway sends that input through the store and dispatches again |
| anything else (large images, uploads, data URIs over the budget) | **store**: R2 PUT (all inputs at once) and a signed URL; the worker reads the object (all at once) |

**Owner decision (2026-09-29): inline inputs are the default for
serverless pools too.** An image-to-video (or any job whose inputs fit
`inline_inputs_max_bytes`) rides in the Runpod `/run` body, capped at 6 MiB
for serverless pools so the body stays under Runpod's 10 MB; only larger
inputs go through a URL or the store. This is what the gateway does today;
the setting stays for tests and for turning it off (`0`).

After a dispatch the gateway copies the inputs that skipped the store into
it **in the background** (`stage_inputs_for_retry`, pools with
`retries > 0`) and points the `gw_dispatch` row at the copies, so a
re-dispatch after a worker loss (§3, any replica) finds them. A re-dispatch
that comes before the copy is done stores the input from the replica's
staged file then. The row never holds the inline bytes. The copies go when
the tick closes the row (the job finished) or the job is failed as lost;
a copy that finishes after the row moved on is deleted at once. None of the
mounted APIs returns a job's input URL, so nothing else needs a stored
input (uploads, `/uploads` and fal storage keep their own store).

### 3.2 Queue timings

fal `timings.queue` is `created_at` → the worker's `started_at`. Behind a
gateway it splits into **`dispatch`** (`created_at` → `Job.dispatched_at`:
the job row insert, input transfer, the hop to the worker, which sets
`dispatched_at` once the inputs are in place) and **`wait`**
(`dispatched_at` → `started_at`: the adoption write and the wait for the
GPU), with `dispatch + wait == queue`. `dispatched_at` is omitted from the
job JSON when unset (a job that never left its process), and then `timings`
has no split.

Where the time goes, per dispatch:

- log line `gateway: dispatched` (job, pool, inputs, and how many went
  `inline` / `source` / `store`, `stage_inputs_ms`, `dispatch_ms`,
  `record_ms`) and on the worker `worker: dispatched inputs in place`
  (`fetch_ms`);
- `/metrics`: `fv_gateway_submit_phase_seconds{pool,phase}` (`phase` =
  `stage_inputs`, `dispatch`, `record`), `fv_worker_input_fetch_seconds`,
  `fv_worker_adopt_seconds`, `fv_gateway_retry_stage_seconds` (the
  background copy), `fv_gateway_job_reads_total{source}` (`memory` | `d1`,
  §3.3).

Dispatch is immediate on submit (not bound to the tick). The worker adopts
in one D1 round trip (§3 step 5, was a read then an upsert), and puts all
inputs in place concurrently, as the gateway stores them concurrently.

Before / after on the fake engine
(`tests/gateway.rs::inline_inputs_take_the_store_off_the_dispatch_path`):
fal image-to-video with a 1024x576 noise PNG (1.7 MB), the shared
artifacts directory behind R2-like latency (PUT 1.5 s, GET 1.0 s) and
250 ms per D1 call; mean of 3 jobs, seconds:

| input path | submit call | `dispatch` | `wait` | `queue` |
|---|---:|---:|---:|---:|
| before: through the store (`inline_inputs_max_bytes = 0`) | 3.97 | 2.93 | 0.38 | 3.30 |
| after: inline (default) | 1.58 | 0.46 | 0.33 | 0.79 |

The 2.5 s gone are the PUT and the GET; what is left of `dispatch` is the
job row insert and the hop (with 250 ms D1 calls), and `wait` is the
adoption write plus the fake engine picking the job up. Text-to-video
(no inputs) was already about 1.1 s of `queue` on the cluster.

### 3.3 Status reads from memory

Each fal status poll cost one D1 read (~0.28 s). `GatewayJobStore` keeps a
view of the jobs it recently inserted, updated or read, and answers `get` /
by-external-id reads from it where the state is known: a job seen within
`watch_poll_ms` (the latency SSE already has), or a finished job seen within
the last 60 s (finished jobs only change by deletion). Anything else reads
D1 and refreshes the view; the `watch()` pollers refresh it too. With
several replicas, a change made through another replica shows up here at
most `watch_poll_ms` later (60 s for deleting a finished job).

### 3.4 Pod placement and bursts

A burst of submits used to land on one pod worker: the gateway picked the
least-loaded worker from its last probe and counted the job against it
only when the dispatch call *returned* (the worker's adopt alone is a D1
round trip), so every concurrent submit saw the same idle worker. 5 jobs
on 3 idle workers queued on one GPU (numbers below;
[gateway-cloudflare.md](gateway-cloudflare.md) §9.4 found 5–11 s mean on
5 workers).

Now, per pod pool, under the pool's lock:

- **Pick and reserve at once.** A worker's load is what it last reported
  (`running` + `queued`) plus the jobs this replica placed there since that
  report was computed, plus this replica's dispatch calls to it in progress
  (`reserved`). The pick takes usable workers (healthy, not draining or
  failed, not already tried for this job), ready ones first, then the fewest
  job-times of wait (`load / capacity`: 0 = a free slot), then the lowest
  load; equal workers are ordered by a per-replica seed, so two replicas do
  not start their bursts on the same worker. The slot is reserved before
  the call.
- **Capacity and queue limit.** Workers report `capacity` (their engine's
  executors) and `queue_max` (`limits.queue_max`) in
  `GET /fv/v1/internal/status`; a worker whose waiting jobs
  (`load − capacity`) reach `queue_max` is skipped. When every usable
  worker is full, `429 QueueFull` + `Retry-After` (as a worker's own 429).
- **No free slot anywhere**: the job goes to the worker where it starts
  soonest and queues there. The gateway keeps no queue of its own (a
  held job would add a poll interval and a second dispatch round trip,
  and the D1 rows already are the durable queue).
- **Release.** A failed call (refusal, 5xx, connection error) or a dropped
  submit gives the slot back (a guard, so no path leaks one). A taken job
  becomes a *placement*: the worker's 202 answer carries its load after
  taking it (`load: {running, queued, capacity, queue_max}`), which is
  applied at once.
- **Reconciliation with the tick.** Every report (probe or dispatch answer)
  carries the instant after which the worker computed it; placements
  acknowledged before that instant are in it and are dropped, later ones
  still count; an older report never overwrites a newer one. Nothing is
  reset wholesale at the tick, and a finished job leaves the count with the
  next report.
- **Replicas.** Each replica reserves for its own calls; other replicas'
  jobs show up in the next report (probe every `tick_s`, or any dispatch
  answer from that worker). Dispatch-row claims are unchanged: a
  re-dispatch claims the row by `attempt`/`state` (one replica wins), and a
  row upsert never overwrites a later attempt.
- **Serverless pools** are unchanged: Runpod queues and places. Their
  `pending` count (for `max_queued` and the pool order) now keeps the
  dispatches the tick's D1 count missed (recorded after it was read)
  instead of resetting to zero, and counts nothing twice.

Measured with the fake engine (`crates/fastvideo-serve/tests/gateway_burst.rs`,
`burst_queue_times_one_worker_vs_three`: 5 jobs submitted at once, 4 × 250 ms
steps per job — about 10 s end to end on the build pod, D1 at 250 ms per call,
most of it the fake engine's own libx264 encode; see §3.5;
queue = `created_at` → `started_at`):

| workers | before: mean / max queue | jobs per worker | after: mean / max queue | jobs per worker |
|---|---:|---|---:|---|
| 1 | 14.3 s / 27.6 s | 5 | 14.8 s / 28.4 s | 5 |
| 3 | 16.7 s / 30.6 s | 5 on one | 3.5 s / 8.2 s | 2, 1, 2 |

After the fix three jobs start at once (0.5 s: the insert and adopt round
trips) and the other two wait one job time.

### 3.5 Per-job overhead

One job end to end on the fake engine, measured point by point
(`crates/fastvideo-serve/tests/job_overhead.rs`; `cargo test -p
fastvideo-serve --features http-client --test job_overhead --
--include-ignored --nocapture` prints the waterfall): a fal
`minimax/h3-turbo` job with a webhook, a status poll every 20 ms and an
SSE stream; 4 fake steps (1 s of simulated work); D1 mocked at 0 or
250 ms per call; R2 PUT mocked; debug events at submit/record, dispatch,
worker envelope/adopt, every engine event, finalize, the artifact PUT and
the terminal write, and every D1 call. Build pod, debug build, 2026-09-29;
milliseconds from the submit call to the client seeing `COMPLETED`
(poll / SSE / webhook):

| setup | before | after |
|---|---:|---:|
| single server, D1 0 ms | 1013 / 1010 / 1042 | 1020 / 1007 / 1036 |
| single server, D1 250 ms | 1261 / 1259 / 1561 | 1279 / 1257 / 1540 |
| gateway + 1 worker, D1 0 ms, 4 × 250 ms | 1043 / 1022 / 1074 | 1042 / 1021 / 1048 |
| gateway + 1 worker, D1 0 ms, 4 × 300 ms | 2037 / 2020 / 1243 | 1236 / 1215 / 1242 |
| gateway + 1 worker, D1 250 ms | 2282 / 2271 / 1821 | 1781 / 1764 / 1790 |
| gateway + 1 worker, D1 250 ms, 4 × 300 ms | 2273 / 2272 / 1995 | 1982 / 1965 / 1999 |
| gateway, D1 250 ms, R2 PUT 1 s | 3574 / 3542 / 2806 | 2792 / 2769 / 2796 |
| single server, D1 0 ms, fake x264 on | 9691 / 9675 / 9723 | 10005 / 9983 / 10013 |
| gateway, the burst test's settings, fake x264 on | 12077 / 11906 / 11824 | 6845 / 6714 / 6740 |

(With 4 × 250 ms the job ended right at a tick of the gateway's 1 s watch
poll, which hid the wait; 4 × 300 ms shows it. The x264 rows vary by
seconds with the pod's load: the fake engine's own encode.)

Every gap over 100 ms, and what it was:

| gap | size | kind | now |
|---|---:|---|---|
| fake steps | 4 × step | simulated work | — |
| fake `decode` → `mux` → `finished` (renders 121 frames of 1344x768 in a debug build and pipes them through libx264) | 5-7 s | **test artefact**: the fake encoder; `engine.fake.mp4 = false` turns it off | unchanged |
| engine `finished` → finalize start: `ffmpeg -version` run per finished job, blocking an async worker thread | 0.7-1.2 s | **real** (every worker with ffmpeg; less on an idle host) | probed once per process, at startup, off the runtime |
| finalize: `-c copy +faststart` remux of an MP4 already faststart | 0.9-1.4 s | **real** | skipped when the file is faststart, one video track, only audio beside it, and no crop/`-an` (`mp4::finalize_is_noop`) |
| gateway: terminal (or running) status in D1 → client | up to `watch_poll_ms` (1 s) + one D1 read (0.25 s) | **real** (the `watch()` poll and the view's freshness window) | the dispatching replica follows the job on its pod worker: `GET /fv/v1/internal/jobs/{id}?wait_s=25&since=<status>` answers at the next status change once D1 has it (`D1JobStore::settle`), with the job; the gateway puts it in its view and wakes the job's poller (a report during the poller's D1 read wins); ~1 ms after the worker's write. A status never moves back in the view (a read started before a change does not undo it). Other replicas still poll. |
| gateway insert, worker adopt, terminal write | 1 D1 round trip each | real, **required** (the id is durable before the 202; the adopt is the fencing; the row says succeeded only after the artifact is stored) | unchanged |
| R2 PUT → terminal write | the PUT | real, **required** (the row carries the artifact; nothing correct to overlap it with) | unchanged |
| worker `Started` write | 1 round trip | real, off the critical path (the pump waits, the engine does not) | unchanged |
| gateway `gw_dispatch` write before the 202 | 1 round trip on the submit call | real, required (cancel and the reaper find the worker through it) | unchanged |
| webhook after the terminal write | 1 round trip | real, intentional (a receiver that reads the job gets the finished state) | unchanged |
| submit read-back from D1 when the view went stale during the dispatch (`watch_poll_ms` ≤ the dispatch time) | 1 round trip | real | the view counts the dispatched job as seen (`GatewayJobStore::dispatched`) |
| client poll (100 ms in the burst test) | ≤ the interval | **test artefact** | — |
| `progress_interval_ms = 100`, `heartbeat_s = 1`, `watch_poll_ms = 100` in the burst test | continuous D1 writes/reads | **test artefact** (defaults 1000 / 60 / 1000) | — |

D1 writes per job (defaults): the insert, the adopt, the `running` write,
at most one coalesced progress/log write per second, the terminal write,
and the `gw_dispatch` insert and close; the flusher now sends one tick's
due writes of all jobs as **one D1 batch** (a worker with queued jobs
wrote one row per job per tick, serially; per-job throttling is
unchanged, and a failed batch falls back to one write per job). The D1
poll behind an SSE stream (`watch_poll_ms`) is unchanged for jobs this
replica did not dispatch.

`overhead_budget_on_the_fake_engine` (feature `http-client`: run by
`FV_SERVE_HEAVY=1 scripts/serve/check.sh`) holds it: at 0 ms D1 the poll, SSE and webhook see the job finish
within the simulated work + 400 ms on both paths; at 250 ms through the
gateway within work + 3 round trips + 400 ms. The code before these fixes
fails it (2031 ms against 1600 ms).

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
- `/fv/v1/capabilities` adds `pools` and each model's `pools`; the console
  reads the same endpoint, so it lists every model of every pool. `pools`
  is the safe summary of `/fv/v1/status` (id, kind, state, available,
  models, queue depth, running jobs, last seen, workers as `w1`, `w2`, …
  with state and load): never worker URLs, worker / pod / endpoint ids, IPs
  or probe error texts. Like a single server's, it also reports the
  gateway's `auth.mode` (`none` | `keys` | `trust-gateway`).
- `GET /fv/v1/status` (public, no secrets; docs/serve/console.md): each
  pool's and worker's state from the tick's probes (`ready`, `busy`,
  `loading`, `scaled_to_zero`, `draining`, `unhealthy`, `down`, `failed`),
  last-seen age, queue depth and running jobs; the console's status strip
  polls it.
- **A worker on a GPU that cannot run its model** (the startup capability
  check, `crates/fastvideo-engine-service/src/device.rs`: compute
  capability, FP8 / NVFP4 tensor cores, the DiT's size against the
  device's memory) fails that model instead of loading it: its internal
  status says `readiness: failed` with `failed_models: {model: reason}`,
  the gateway shows the worker `failed` and never dispatches to it, and
  `/fv/v1/status` gives the model `state: failed` with the `reason` (e.g.
  "model `h3-turbo` cannot run on this GPU (NVIDIA A100 80GB, sm80): it
  needs FP8 tensor cores (Ada, Hopper or Blackwell: sm89 or newer)"). This
  replaces the failure mode where an A100 reported ready and every H3 job
  then failed with "no tensorwise FP8 algorithm on sm80".
- **Experimental feature flags** (docs/serve/console.md §7) act on the
  caps at negotiation: the gateway wraps its pools' aggregated caps with
  the flags (D1 `feature_flags`, cached, re-read every 30 s) before any
  API negotiates, so a request the flags do not allow is refused on the
  gateway and never dispatched. Workers need no flag state for batch jobs
  (they run already-negotiated jobs); a worker's fal director reads the same
  D1 table for its 1080p chunk cap.

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
  can scale from zero), 503 while draining; `/healthz` includes the same
  safe per-pool summary as `/fv/v1/status`. The details (worker URLs and
  ids, endpoint ids, probe errors) are only on `/fv/v1/gateway/pools`
  (admin token: the metrics of §7 plus `state`, each pool's full view,
  including each worker's full `build` (git sha, build time, variant, image
  digest and tag, release channel) and `gateway_build`).
- **Versions** (docs/serve/releases.md): workers report their build in
  `GET /fv/v1/internal/status` (`build`), and the tick keeps it per worker.
  The public views (`/fv/v1/status`, `/healthz`, capabilities' `pools`)
  show per pod pool only `versions: [{sha, channel, workers}]` (the
  7-character sha and the channel of the answering workers) and
  `mixed_versions` when they run more than one sha (a rolling redeploy in
  progress, or drift); `/fv/v1/status` adds the gateway's own `version`
  (`{sha, channel}`) and a top-level `mixed_versions`. Serverless pools
  have no per-worker version (their workers follow the Runpod template).
- **Releases** (admin token; `crates/fastvideo-serve/src/releases.rs`):
  `GET /fv/v1/admin/releases`, `GET /fv/v1/admin/deployments` (the D1
  history and registry, live builds, drift) and `POST
  /fv/v1/admin/releases/{promote,rollback}` (`dry_run` for the plan;
  otherwise they dispatch `release.yml`, which needs `FV_GITHUB_TOKEN` on
  the gateway, else 503). The console's Deployments page uses them.
  Errors returned to API clients name the pool only (a `503` for a pool
  that cannot take work, a dispatch nobody took); the causes go to the
  log.
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
   - pods: `create` (GPU-type list × region+volume placements, e.g.
     EUR-IS-1 with `jg48s6o1w0` — EU only since the US volume `s2k01690bi`
     was deleted 2026-10; a rebuilt US volume is another placement; out of stock →
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

## 9. Auth and the admin token

- The gateway runs `auth.mode = "keys"` (the default, and what
  `configs/serve/gateway*.toml` set): users call it with API keys, from
  `FV_API_KEYS` (SHA-256 list) or minted with the admin token
  (`/console/admin`, `POST /fv/v1/admin/keys`, stored in D1 so every
  replica sees them). `FV_AUTH_MODE=none` still exists for a local demo;
  the cluster script no longer uses it. Workers are always
  `trust-gateway` behind the internal token.
- **Admin token** (`/fv/v1/admin/*`, `/fv/v1/gateway/pools`, key minting):
  `FV_ADMIN_TOKEN` when set; otherwise the server makes one on its first
  start and keeps it in **`<state_dir>/admin_token`** (mode 600, written
  under a temporary name and renamed), and every later start with the same
  state dir reuses it. It is never logged: the log line names the file and
  the token's first 4 characters. The process keeps only its SHA-256.
  Workers keep no file (their admin routes sit behind the internal token).
- **On a Runpod pod without a volume** the state dir is on the container
  disk: the token survives a restart of the pod (an env PATCH, `extend`),
  not its re-creation (a new pod makes a new token). Put `state_dir` on a
  volume, or set `FV_ADMIN_TOKEN` from a Runpod secret, to keep it across
  re-creations.
- **Fetching it remotely**: with `FV_ADMIN_TOKEN_RECIPIENT` (an X25519
  public key, 32 bytes base64) the server publishes the token sealed to
  that key at `GET /fv/v1/admin/token/sealed` (public; 404 without a
  recipient): `{"alg": "X25519-SHA512-AES256CTR-HMACSHA256", "epk", "iv",
  "ct", "tag"}` (ephemeral X25519, SHA-512 key derivation, AES-256-CTR,
  HMAC-SHA256 over `iv ‖ ct`; `crates/fastvideo-serve/src/admin_token.rs`).
  Only the private key's holder can open it, with a stock `openssl`.
- **The cluster** (`scripts/serve/runpod-cluster.sh`): `up` makes an X25519
  key pair next to the state file (`cluster.json.admin-key.pem`, mode 600;
  the private half never leaves the machine) and gives the gateway the
  public half; nothing secret about the admin token is in the pod's env.
  The owner reads the token with

  ```bash
  scripts/serve/runpod-cluster.sh admin-token
  ```

  which fetches the sealed token on first use, opens it with `openssl`,
  keeps a copy in the state file (`.admin_token`, mode 600) and prints it.
  `wait`, `status`, `mint` and `smoke` use the same copy through a header
  file (never argv): `wait` / `status` read the admin route
  `/fv/v1/gateway/pools`, `mint <name> <file>` mints a user key, and
  `smoke` uses `FV_CLUSTER_KEY_FILE` or mints one key once
  (`.smoke_api_key`). If the gateway pod was re-created, the copy stops
  working (401) and the script fetches the new token by itself. A state
  file from an older script (an `.admin_token` it generated, no key pair)
  keeps passing that token as `FV_ADMIN_TOKEN`, so a running cluster is
  not cut off.
- **GitHub token for Promote / Rollback** (optional): when
  `/root/.config/fv/github_token` (or `$FV_GITHUB_TOKEN_FILE`) exists with
  mode 600, `runpod-cluster.sh` (`up`, and every gateway env update:
  `roll`, `extend`, the pool URLs) puts its content in the gateway pod's
  env as `FV_GITHUB_TOKEN`, so the console's Deployments page can dispatch
  `release.yml` (docs/serve/releases.md). jq reads it from the file: it is
  never printed and never written to the state file, the ledger or the
  repo. Another mode is refused with a warning; without the file the
  Promote / Rollback routes answer 503 and their dry runs still work.
- **Price cap before create**: every script that creates a GPU pod
  (`runpod-cluster.sh`, `runpod-pod.sh`, `scripts/gpu/runpod-http.sh`,
  `runpod-sfwan-whip.sh`, `e2e/{pod,wan-pod,ltx-pod}.sh`) checks Runpod's
  price quote for the GPU type (`gpuTypes` `securePrice` /
  `communityPrice`, and the datacenter's lowest uninterruptible price when
  it pins one) against `RUNPOD_GPU_MAX_DPH` **before** `POST /pods`, with
  `scripts/gpu/runpod-price.sh`, and skips a type over the cap or without a
  quote. Before, `runpod-pod.sh` created the pod, read `costPerHr` and
  deleted it when over the cap: a director-debug run created and deleted 18
  H100 pods above its cap that way. The post-create check stays as a
  second guard.
