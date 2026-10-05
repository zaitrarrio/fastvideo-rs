# Family Durable Objects: one queue per model family, GPU hosts as clients

Status: **design 2026-10-02; implemented 2026-10-02/05** behind the existing
opt-in (`dispatch = "durable-object"` + `family`); the classic gateway path
stays the default. Results: §14.
This document extends the per-pool `PoolScheduler` design and its phases 0–2
in [gateway-cloudflare.md](gateway-cloudflare.md) (§3, §9). Read that first:
everything there (push dispatch, ack = adopt, leases as fencing tokens,
reconnect reconcile, spill of large envelopes, D1 write-behind) still holds
and is reused. Results and the production checklist are in §12–§13.

The owner's ask: "Document the DO based design and then implement it."

## 1. Summary

- **One Durable Object per model family** (`h3`, `ltx`, `wan`, `sfwan`, …),
  not per pool. Its SQLite storage *is* the queue: jobs, attempts, leases,
  the worker registry, session leases, uploads and a metrics window. One
  writer per family; alarms drive lease expiry, the reaper and upload
  cleanup.
- **GPU hosts are clients.** A worker holds one outbound WebSocket per family
  it serves (hibernatable on the DO side). Dispatch is **credit-based**: the
  worker advertises its free slots, the DO offers a job only to a worker it
  believes has one, the worker's **ack is the adopt** (idempotent on
  `(job_id, attempt)`, fenced by the lease), a **nack** sends the job to
  another worker at once. Cancel and drain go down the same socket.
- **Per-host capacity arbiter.** A GPU that serves several families gets
  offers from several DOs. A local arbiter on the worker accepts or refuses
  every offer against *one* slot budget, so two DOs can never overfill one
  GPU (§6).
- **Outputs go straight to R2.** A Rust thread on the worker uploads the
  output while the engine is still writing it, part by part, through
  per-job, short-lived URLs the DO mints. No R2 credentials ever reach a GPU
  host. With a fragmented MP4 writer the "MP4 tail" (the remux and upload
  after the last frame) shrinks to the last part. On completion the worker
  reports `{job, key, bytes, sha256}`; the DO completes the upload and only
  then may the job succeed (§7).
- **Inputs**: inline in the envelope up to 8 MiB as today, larger ones as
  presigned R2 `GET`s from the store (§7.6).
- **Streaming sessions** (director, Reactor, WHIP) are **not** queue jobs.
  The family DO does admission and placement: it picks a GPU with free
  session capacity, the worker's arbiter reserves it, and the DO returns a
  session lease plus the GPU's own endpoint. Signalling and media then go
  client ↔ GPU directly; the DO only renews, expires and reclaims the lease
  (§8).
- **D1 stays the cross-family record** (job list, status, billing,
  results). The DO writes its view of each job (dispatch timings, attempt,
  result key/bytes/sha256) to D1 behind the critical path; clients keep
  using SSE, webhooks and polling (§9).
- **Robustness**: at-least-once with idempotent acks and lease fencing;
  lease-expiry alarms requeue; a deploy drops every socket and workers
  re-announce their in-flight jobs and sessions without duplicate retries;
  incomplete multipart uploads are aborted by the DO and by an R2 lifecycle
  rule; each family exports its queue depth as the autoscaler signal (§10).

## 2. What changes against the per-pool design

| | per pool (phases 0–2) | per family (this design) |
|---|---|---|
| DO key | pool id (`h3-turbo`, `gpu-h3`) | family (`wan`, `ltx`, `sfwan`, `h3`); several pools may share one family DO |
| routes | `/pools/{pool}/…` | `/families/{family}/…` (the `/pools/` routes stay, same class, for rollback) |
| worker sockets | one, to its pool | one per family it serves |
| placement | DO computes `capacity − held` from its own view | worker **credits** (`slots` frame) minus offers in flight; DO view is advisory, the worker arbiter decides |
| multi-family GPU | not possible (one pool per worker) | local arbiter; nack → re-offer elsewhere, no backoff |
| outputs | worker uploads with its own R2 credentials after the job (whole file) | DO-minted part URLs, no credentials on GPU hosts, upload overlapped with the encode |
| sessions | DO pools answer 503 | DO admission + session lease; GPU endpoint returned; media direct |
| metrics | `/pools/{pool}/status` | plus `/families/{family}/metrics` (queue depth and age, slots, sessions) for fv-control and the autoscaler |
| protocol version | `proto = 1` | `proto = 2` (v1 workers keep working: capacity-based placement, no uploads, no sessions) |

Unchanged: the envelope (opaque to the DO), the lease/fencing rules, spill to
R2 above 1 MiB, the 424 restage path, the reconnect grace and stale timers,
D1 write-behind, the gateway as the API front.

## 3. Architecture

```
client ─► fv-serve gateway (auth, adapters, D1 job rows)          [as today]
            │  POST /families/{f}/enqueue | /sessions | /cancel
            ▼
   FamilyScheduler DO  ("wan", "ltx", "sfwan", "h3" …)        ◄── fv-control: GET /families/{f}/metrics
     SQLite: jobs, envelopes, workers, sessions, uploads, meta
     alarm : ack deadlines, lost workers, session expiry, upload cleanup
     D1    : edge_jobs write-behind (dispatch facts + result key/bytes/sha256)
     R2    : envelopes > 1 MiB; output multipart uploads (create / complete / abort)
            ▲            ▲
   ws "wan" │            │ ws "sfwan"      one socket per family, credits ▲ offers ▼
        ┌───┴────────────┴───┐
        │ GPU worker (fv-serve role=worker)                         media: client ◄──► GPU (WebRTC / WHIP)
        │  arbiter: 1 slot budget across families and sessions
        │  upload thread ──── PUT parts ───────────────► R2 (edge capability URL or S3 presigned URL)
        └────────────────────┘
```

## 4. Families and keying

- A **dispatch family** is a string key, `[A-Za-z0-9._-]`. Defaults follow
  the protocol family of the models (`h3`, `ltx2` → `ltx`, `wan`); streaming
  causal models (SF-Wan, LongLive) form `sfwan`. Config can override it on
  both sides:
  - gateway: `[[pools]] family = "wan"` (`FV_POOL_<ID>_FAMILY`) on a pool
    with `dispatch = "durable-object"`; without it the pool keeps the per-pool
    routes (phase 0–2 behaviour);
  - worker: `[dispatch] families = ["wan", "sfwan"]` (`FV_DISPATCH_FAMILIES`);
    without it, `[gateway] pool` keeps the single per-pool socket.
- The DO name is `family:{family}` (`idFromName`), so family objects and
  pool objects of the same class never collide.
- Within a family the DO still filters by model: a worker announces the model
  ids it serves (`hello.models`), a job carries its model, and a job is only
  offered to a worker that serves it (wan 1.3B vs wan 14B in one family).
- `POOL_LOCATIONS` becomes `LOCATIONS` keyed by DO name (`family:wan` →
  `weur`). A DO lives in one place (§11).

## 5. Data model (DO SQLite)

All rows are JSON records of the pure scheduler (`fastvideo-dispatch-proto::sched`)
keyed by id, so the schema follows the Rust types; a migration is additive
(`CREATE TABLE IF NOT EXISTS`, new fields `#[serde(default)]`).

| table | key | what |
|---|---|---|
| `meta` | `k` | `name` (`family:wan` or a pool id), schema version |
| `jobs` | `job_id` | `JobRec`: phase (`queued`/`pushed`/`running`/finished), attempt, max attempts, worker, model, seq, timestamps (enqueued, pushed, acked, finished), ack deadline, backoff, lease, takeover, restage, spill key, cancel flag, error, and (new) `result {key, bytes, sha256}` |
| `envelopes` | `(job_id, part)` | the envelope in ≤ 1 MB chunks (above `SPILL_BYTES` it lives in R2 instead) |
| `workers` | `worker_id` | `WorkerRec`: connected, draining, capacity, version/sha, models, caps, last seen, disconnected at, and (new) `credits` (free slots last reported), `offers_sent`, `offers_seen`, `session_free`, `endpoint`, `proto` |
| `sessions` (new) | `session_id` | `SessionRec`: model, kind, owner, worker, lease, state (`offered`/`live`/`ended`), offered at, deadline (offer ack), expires at, tried workers, endpoint, end reason |
| `uploads` (new) | `upload_id` | `UploadRec`: job, attempt, lease, worker, key, state (`creating`/`open`/`completing`/`done`/`aborting`), parts granted, created at, expires at |

The metrics window is the last 256 dispatches (`queue`, `ack`, worker take
time) plus finished jobs kept for an hour (`keep_finished_ms`), both already
in the scheduler; `metrics` derives queue depth and age from `jobs`.

## 6. Wire protocol v2 and the arbiter

### 6.1 Frames

JSON text frames tagged by `"t"` (as v1). New in v2 in **bold**.

Worker → DO:

| frame | fields | meaning |
|---|---|---|
| `hello` | worker_id, pool (= the DO name), proto, version, sha, capacity, draining, models, caps, jobs (held/finished), **sessions** (held), **endpoint**, **slots** | first frame; re-announces everything in flight after a reconnect |
| **`slots`** | free, session_free, offers_seen | credits: what the arbiter can still take, counted after the `offers_seen` offers it has received from this DO |
| `ack` | job_id, attempt, lease, worker_ms | taken = adopted (idempotent on `(job_id, attempt)`) |
| `nack` | job_id, attempt, retry, code, message | not taken; **429** = the arbiter is full (another family took the slot): offered elsewhere at once, this worker gets no offer until its next `slots`; 409/503 back off; 424 restage; `retry = false` fails |
| `done` | job_id, attempt, state | finished (frees the slot) |
| `status` | running, draining, capacity | heartbeat |
| **`upload_init`** | req, job_id, attempt, lease, name, content_type, parts | open a multipart upload for the job's output |
| **`upload_more`** | req, job_id, upload_id, from, count | more part URLs |
| **`upload_done`** | req, job_id, attempt, lease, upload_id, parts `[{n, etag}]`, bytes, sha256 | complete it |
| **`upload_abort`** | job_id, upload_id | give it up |
| **`session_ack`** | session_id, lease, endpoint | the arbiter reserved the GPU for this session |
| **`session_nack`** | session_id, lease, code, message | refused (busy); the DO tries the next worker |
| **`session_end`** | session_id, lease | the session ended on the GPU |

DO → worker:

| frame | fields | meaning |
|---|---|---|
| `welcome` | worker_id, pool, cancel, **end_sessions** | answer to hello: jobs and sessions to drop here |
| `job` | job_id, attempt, envelope, lease, takeover | an **offer** |
| `cancel` / `drain` | | as v1 |
| **`upload_grant`** | req, job_id, upload_id, key, bucket, part_urls `[{n, url}]`, expires_ms, error | URLs for parts `from..from+count` |
| **`upload_committed`** | req, job_id, key, bytes, ok, error | the object exists (or why not) |
| **`session_offer`** | session_id, lease, model, kind, ttl_ms | reserve the GPU for a session |
| **`session_revoke`** | session_id, reason | the lease ended (released, expired, replaced) |

HTTP (gateway, fv-control → Worker → DO; internal token unless noted):

| route | body / answer |
|---|---|
| `GET /families/{f}/connect` | WebSocket upgrade (`x-fv-worker-id`) |
| `POST /families/{f}/enqueue` | `EnqueueReq` → `EnqueueResp` (as v1) |
| `POST /families/{f}/cancel/{job}` | as v1 |
| `GET /families/{f}/status` | `PoolStatus` + sessions (internal or admin token) |
| `GET /families/{f}/metrics` | `FamilyMetrics` (internal or admin token) |
| `POST /families/{f}/sessions` | `SessionReq {session_id?, model, kind, owner, ttl_ms}` → 200 `SessionGrant {session_id, lease, worker_id, endpoint, expires_ms}`, 429 no capacity |
| `POST /families/{f}/sessions/{id}/renew` | → `{expires_ms}`; 404 if gone |
| `POST /families/{f}/sessions/{id}/release` | → `{state}` |
| `PUT /up/{token}` | **no token header**: the capability token in the path is the authorization (edge upload mode, §7.3) |
| `GET /dl/{token}` | same, for a result object (tests, operators) |

### 6.2 Credits

The DO keeps, per worker, the last `slots.free` it was told, and how many
offers it has sent (`offers_sent`); the worker reports how many it has
received (`offers_seen`). The DO's estimate of free slots is

```
free_estimate = slots.free − (offers_sent − offers_seen)
```

so an offer sent after the worker computed its report is never counted
twice, and a `slots` frame that crosses an offer on the wire cannot make the
DO overfill the worker. v1 workers (no `slots`) keep the v1 rule
`capacity − held`. The estimate only decides *whom to offer to*; the arbiter
on the worker decides *whether to take it*, so a stale estimate costs one
nack and an immediate re-offer, never a double booking.

### 6.3 The per-host arbiter (the rule)

One worker process owns one GPU slot budget `capacity` (`[dispatch]
capacity`, default 1 when it serves several families, else 2 as today:
running + the next one waiting in the engine queue). The engine runs one
job at a time per GPU in any case; the budget bounds what is *held*.

1. An offer is taken only if `held_jobs < capacity` **and** no session holds
   the GPU. Taking it and counting it is one atomic step (one mutex), so two
   offers from two family DOs that arrive together cannot both pass.
2. Otherwise the worker nacks **429** (`retry`). The DO re-offers the job to
   another worker immediately and treats this worker as full until its next
   `slots` frame.
3. A session offer is taken only if nothing is held (`held_jobs == 0`) and no
   other session is live: sessions are exclusive (`[dispatch]
   session_exclusive = true`). While a session holds the GPU, every family
   socket reports `free = 0`.
4. Every change of the budget (take, done, session start/end, drain) sends a
   `slots` frame on every family socket, so each DO learns about the other
   families' use within one message.
5. Draining reports `free = 0` everywhere; held jobs finish.

## 7. Outputs: direct, overlapped upload to R2

### 7.1 Flow

```
worker                                   family DO                         R2
  │ ack(job)                                 │                               │
  │ upload_init(job, attempt, lease) ───────►│ check lease = job's lease     │
  │                                          │ create multipart ────────────►│
  │◄──────────── upload_grant(part URLs, exp)│ uploads row                   │
  │ upload thread: tail output.mp4           │                               │
  │   PUT part 1..n while frames encode ─────┼──────────────────────────────►│
  │ engine finished; finalize (no-op on fMP4)│                               │
  │   verify parts, PUT the last part ───────┼──────────────────────────────►│
  │ upload_done(parts, bytes, sha256) ──────►│ check lease; complete ───────►│
  │◄─────────── upload_committed(key, bytes) │ result into jobs + D1         │
  │ job row: succeeded, artifact {bucket, key} (lease-fenced write)          │
  │ done ───────────────────────────────────►│                               │
```

- The grant is requested right after the ack, in parallel with the engine's
  start, so it is off the critical path.
- Part URLs expire with the job's upload TTL (default 1 h, at most the job
  timeout). `upload_more` mints more (and fresh ones) when needed.
- **Fencing**: `upload_init` and `upload_done` carry the lease. A worker the
  DO gave up on (its job moved to another worker under a newer lease) gets
  `error` and cannot commit; its upload is aborted. A result is committed at
  most once per job (idempotent on `upload_id`).

### 7.2 The upload thread and the MP4 tail

- One `std::thread` per job (`fv-serve/src/upload.rs`), started at the ack.
  It follows the engine's output file (`<output_dir>/<job>/output.mp4`) while
  it grows and PUTs every complete part (8 MiB by default; S3 allows 5 MiB to
  5 GiB, the last part smaller) as soon as it is on disk, hashing as it goes.
- When the engine finishes, the gate hands the final file to the thread. One
  local read pass compares each uploaded part's SHA-256 with the file and
  re-PUTs only the parts that differ (a writer that seeks back, e.g. to
  patch an `mdat` size, changes part 1 only), then uploads the rest. The
  whole-file SHA-256 comes from the same pass. This makes the overlap safe for
  any writer; it only *pays off* for writers that append.
- **Fragmented MP4** (`[engine] mp4_fragmented = true`, `FV_MP4_FRAGMENTED`):
  the MP4 writer uses `-movflags +frag_keyframe+empty_moov+default_base_moof`
  instead of `+faststart`. The file is append-only, plays progressively in
  browsers, and needs no faststart remux (the `moov` is first), so the gate
  skips `finalize` unless the job needs a crop or `-an`. What is left after
  the last frame is ffmpeg's flush plus one part upload and the complete.
  A job that needs post-processing falls back to "finalize, then upload the
  finalized file through the same grant" (no overlap, still no R2
  credentials on the host).
- Stats per job (`fv_worker_upload_*`, the `upload_done` frame): bytes sent
  before the engine finished, bytes re-sent, time from engine finish to commit.

### 7.3 URL modes

| mode | when | part URL | create / complete / abort |
|---|---|---|---|
| **edge** (default) | the Worker has the output bucket bound (`OUTPUTS`) and `FV_UPLOAD_SIGNING_KEY` | `https://<worker>/up/<token>`: `token = b64url({k, u, n, e}) . hex(HMAC-SHA256(key, payload))`; the Worker front checks it (no DO hop) and calls `upload_part` on the binding | R2 binding (`create_multipart_upload`, `resume_multipart_upload(..).complete/abort`) |
| **s3** | additionally `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_S3_ENDPOINT` secrets | SigV4 query-presigned `PUT …?partNumber=n&uploadId=u` on `<account>.r2.cloudflarestorage.com` (bytes do not pass through the Worker) | binding as above (R2 upload ids are shared between the binding and the S3 API) |

Both are "presigned" in the sense that matters: per part, per job, expiring,
minted by the DO, and the GPU host holds no R2 credentials. The edge mode
needs no R2 API token (the deploy token cannot mint one, gateway-cloudflare.md
§9.6), so staging uses it; production can switch to s3 to keep video bytes
off the Worker (Workers bill CPU, not bandwidth; a 20 MB part costs a few ms).
The SigV4 presigner lives in `fastvideo-dispatch-proto::presign` (pure Rust,
`hmac` + `sha2`, builds for wasm32) and is tested against the AWS example
vector and against serve-kit's `S3Config::presign`.

### 7.4 Cleanup

- The DO aborts an upload when its job fails, is cancelled, is lost to
  another worker, or when the upload outlives `expires_ms` without
  `upload_done` (alarm).
- The output bucket gets an R2 lifecycle rule: **abort incomplete multipart
  uploads after 1 day** (and, on staging, delete objects after 7 days). That
  covers a DO that is itself lost mid-upload.

### 7.5 What the API sees

The job's artifact is `{location: Object {bucket, key}, bytes}` as with the
current S3 artifact store; download URLs are signed by whoever serves the
API with its own credentials for that bucket. In production the Worker's
`OUTPUTS` binding must therefore be the gateway's artifact bucket. The
staging Worker binds its own bucket (`fv-edge-staging-outputs`) and serves
`GET /dl/<token>` for checks.

### 7.6 Inputs

Unchanged and documented here for completeness: the gateway inlines inputs up
to 8 MiB (base64) in the envelope; larger ones go to the store and the
envelope carries a presigned `GET` (`InputRef.url`); client URLs are fetched
by the worker (`source`, sha256-checked) with the 424 restage fallback.
Envelopes above 1 MiB are spilled by the DO to R2 while queued.

## 8. Streaming sessions (director, Reactor, WHIP)

Sessions are not jobs: they are admitted, placed and leased, never queued.

```
gateway                      family DO                          GPU worker
  │ POST /sessions ─────────────►│ pick: connected, serves model,    │
  │   {model, kind, ttl}         │ session_free > 0, fewest held     │
  │                              │ session_offer(lease) ────────────►│ arbiter: exclusive reserve
  │                              │◄──────────── session_ack(endpoint) │ (nack → next worker)
  │◄── {session_id, lease, endpoint, expires}                         │
  │ signalling (SDP/ICE, /wma/session, /start_session, WHIP) ─────────►│ endpoint
  │ client ◄═══════════════ media (WebRTC / WHIP) ════════════════════► GPU
  │ heartbeats → POST /sessions/{id}/renew (extends expires)          │
  │ end → POST /sessions/{id}/release ─►│ session_revoke ───────────►│ slot back; slots frame
```

- Admission is synchronous: the HTTP call waits for the ack (default 5 s per
  worker, at most 3 workers) and answers 429 with `retry-after` when no GPU
  has room. The lease is the fencing token for renew/release.
- `endpoint` is the worker's public base URL (`FV_PUBLIC_BASE_URL`, the
  Runpod proxy host). The gateway proxies the signalling HTTP to it as for
  gateway pools today; the answer SDP carries the GPU's own ICE candidates,
  so media goes client ↔ GPU directly. (Relaying the signalling over the
  socket, gateway-cloudflare.md §3.7, stays a later option.)
- Expiry: a session not renewed within `ttl_ms` (default 60 s; the gateway
  renews on each director heartbeat and on Reactor/stream calls) is ended by
  the alarm, and the worker gets `session_revoke`. A worker that ends a
  session itself sends `session_end`. Either way the arbiter frees the GPU
  and every family socket gets a `slots` frame.
- A worker that loses its socket keeps its session (media does not depend on
  the DO); on reconnect `hello.sessions` re-announces it; the DO keeps it if
  the lease is current, else answers `welcome.end_sessions`.

### State machine: session lease

```
            session_offer                session_ack
  (admit) ───────────────► offered ─────────────────────► live ──┬── release / expiry / session_end ──► ended
                              │  nack, offer timeout              │
                              └──► next worker, or ended(no_capacity)
```

## 9. D1

- The gateway still inserts the job row the APIs read (behind the response
  for DO-only models, phase 2) and the worker writes progress and the result
  into it, fenced by the lease (unchanged).
- The family DO writes `edge_jobs` (per job: family, state, attempt, worker,
  enqueued/pushed/acked/finished timestamps, error, and now `result_key`,
  `result_bytes`, `result_sha256`) through the D1 binding behind the critical
  path, on every state change. On production that binding is the gateway's
  database (`fv-jobs`), so billing and audits can join `jobs` and
  `edge_jobs`.
- Making the DO the only writer of `jobs` needs the serve-kit job model in
  wasm (blocked on `protocol → models → tokenizers`, gateway-cloudflare.md
  §4); it is a follow-up, not part of this change.

## 10. Robustness and failure modes

| failure | what happens |
|---|---|
| offer lost on the wire / worker never acks | ack deadline (10 s) → queued again, same attempt, next lease; the worker that acks late gets `cancel` |
| two DOs offer to one GPU at once | arbiter takes one, nacks 429 the other → re-offered elsewhere at once |
| worker crash mid-job | socket closes → grace (20 s) → re-dispatch once (`attempt + 1`, takeover, newer lease), then fail; its upload is aborted |
| network partition (worker alive, socket gone) | as a crash for the DO; the old holder's D1 writes and `upload_done` are refused by the lease, it stops the job when it learns (fenced writes / welcome cancel) |
| **Worker deploy** | every socket drops, every DO restarts from SQLite; workers reconnect (0.25 s backoff, ≤ 2 s) and re-announce held jobs (with leases), finished-but-unreported jobs and live sessions; the DO keeps what is current, requeues what it pushed but the worker never got, cancels stale copies. No job is retried because of a deploy |
| upload part PUT fails | the thread retries the part (3×, backoff); a grant that expired is refreshed with `upload_more` |
| worker dies mid-upload | the DO aborts the multipart upload when the job is lost; the R2 lifecycle rule aborts anything older than a day |
| DO evicted while idle | hibernation keeps sockets; state reloads from SQLite on the next event |
| session holder disappears | lease expires (no renew) → slot reclaimed; the worker ends the session when told, or on its own media timeout |
| D1 unavailable | dispatch is unaffected (D1 is behind the critical path); `edge_jobs` writes are retried on the next change |

Semantics: at-least-once delivery of offers, exactly-once effect per lease
(acks idempotent on `(job_id, attempt)`, writes conditional on the lease,
one committed upload per job).

**Autoscaler signal.** `GET /families/{f}/metrics` answers

```json
{"family": "wan", "now_ms": …, "queued": 3, "oldest_queued_ms": 4200,
 "pushed": 1, "running": 2, "workers": 2, "slots_total": 2, "slots_free": 0,
 "sessions_live": 1, "session_capacity": 1, "failed_1h": 0,
 "queue_p50_ms": 21, "ack_p50_ms": 15}
```

fv-control reads it per configured family (admin token) and shows queue
depth and age per family; the autoscaler policy takes `queued`,
`oldest_queued_ms` and `slots_free` as its per-family demand signal.

## 11. Region

A DO lives in one location (where first used, or its `locationHint`). A
family DO in `weur` with workers in EUR-IS-1 and US-CA-2 costs the US workers
about 100–150 ms per message (offer, ack, credit, upload grant). That is
acceptable: one job costs a handful of messages against a 5–40 s run, and
part PUTs go to R2 directly, not through the DO. Media never touches the DO.
If one family's traffic is split evenly across continents, run one DO per
(family, region) (`wan-eu`, `wan-us`) and route by the pool's region; that
needs no protocol change.

## 12. Migration path

1. **Now (opt-in)**: a pool with `dispatch = "durable-object"` and **`family =
   "<f>"`** uses the family routes; without `family` it keeps the pool routes.
   Workers opt in with `[dispatch] families = [...]` (`FV_DISPATCH_FAMILIES`)
   and `direct_upload = true` (`FV_DISPATCH_DIRECT_UPLOAD`). Old workers
   (proto 1) can connect to a family DO; they just get no sessions and no
   upload grants.
2. **Staging**: deploy the class with the family routes next to the pool
   routes (migration `v2`: no class rename, new tables only), bind the outputs
   bucket and the upload key, run §13.
3. **Default switch criteria** (all on a production-like setup, not staging):
   - queue time p50 ≤ the gateway path's, and no lost or duplicate job, over
     ≥ 1,000 jobs across ≥ 2 families and ≥ 2 regions;
   - ≥ 3 redeploys during load with every job finished once;
   - one multi-family GPU worker for ≥ 24 h with no arbiter violation
     (`fv_worker_arbiter_overbooked_total == 0`);
   - direct upload: time from last frame to result ≤ the classic path's for
     every model, and no orphaned multipart uploads after the lifecycle period;
   - sessions: director and Reactor sessions admitted through the DO with the
     same success rate as the gateway path;
   - the production checklist in gateway-cloudflare.md §9.9 (production
     Worker on `fv-jobs`, least-privilege token, alerts, runbook) done.
4. **Switch**: the config default for pod pools becomes `durable-object` with
   `family` derived; serverless pools stay on the gateway path. Rollback is
   per pool (`dispatch = "gateway"`).

## 13. Test plan

| level | test | where |
|---|---|---|
| unit | scheduler: credits and in-flight offers, arbiter nack → immediate re-offer elsewhere, lease-expiry requeue, re-announce after a restore (no duplicate), session admit/ack/nack/expiry/release/re-announce, upload init/grant/fencing/commit/abort-on-loss, metrics | `fastvideo-dispatch-proto` (`cargo test`) |
| unit | SigV4 presign: AWS example vector, multipart part query, equality with serve-kit's signer; edge capability token sign/verify/expiry | `fastvideo-dispatch-proto`, `fastvideo-serve` |
| unit | worker arbiter: concurrent offers from two families never exceed the budget; session exclusivity | `fastvideo-serve` |
| unit | tail upload: a file that grows while uploading (and one that is patched at the head) uploads every part once (part 1 again when patched), sha256 matches | `fastvideo-serve` against a local S3 mock |
| integration | fake-engine worker serving two families through the native family DO: arbiter (never two jobs held over budget), nack → re-offer, lease expiry requeue, deploy → reconnect → re-announce without duplicates, direct multipart upload to the local S3 mock with sha256 check, session admission (gateway → DO → worker endpoint) | `crates/fastvideo-serve/tests/edge_family.rs` |
| integration | the same tests against `wrangler dev` / staging (`FV_EDGE_URL`) where the stand-in is not needed | build pod / this container |
| regression | `FV_SERVE_HEAVY=1 scripts/serve/check.sh` (`gateway_burst::two_gateway_replicas_share_the_pool` is a known flake: rerun alone) | build pod |
| real | one RTX PRO 6000 in EUR-IS-1 serving two families: burst on both (arbiter), queue time and time-to-result vs the classic path, kill/reconnect mid-job, one director session through the DO | §14 |

## 14. Results

### 14.1 What was built

| component | where | what |
|---|---|---|
| protocol 2 + scheduler | `crates/fastvideo-dispatch-proto` (`lib.rs`, `sched.rs`) | `Scope` (pool / family routes), the frames of §6.1, credit placement (§6.2), 429 arbiter nack → immediate re-offer without backoff or bounce limit, session admission / ack / nack / offer timeout / renew / expiry / release / re-announce, lease-fenced uploads (create → grant → complete, abort on loss / fence / cancel / TTL), `metrics` |
| presigner | `fastvideo-dispatch-proto::presign` | SigV4 query presigning with extra signed params (`UploadPart`), edge capability URLs (`/up`, `/dl`, HMAC + expiry); pure Rust, builds for wasm32 |
| Durable Object | `crates/fastvideo-edge/src/edge.rs` | `/families/{f}/…` next to `/pools/{p}/…` (same class, object `family:{f}`); SQLite tables `sessions`, `uploads` (schema 2, additive); R2 multipart through the `OUTPUTS` binding; `/up` and `/dl` checked at the Worker front (no DO hop); S3-presigned mode when `R2_*` secrets exist; admission waits for the worker's answer (20 s); `edge_results` in D1 |
| worker | `crates/fastvideo-serve/src/{arbiter,edge_link,upload}.rs` | one socket per family, one arbiter per GPU (§6.3), `slots` on every budget change, session offers, re-announce; per-job upload thread that follows the growing output, verify pass, commit through the object; `DirectStore` in front of the artifact store |
| fragmented MP4 | `fastvideo-media::mp4` (`Mp4Spec::fragmented`), engine-service CUDA delivery, `engine.mp4_fragmented` | append-only MP4; the gate skips the faststart remux unless the job needs a crop or `-an` |
| gateway | `crates/fastvideo-serve/src/gateway/{edge,proxy,tick}.rs` | `[[pools]] family`: family routes; director / Reactor / native-stream sessions admitted by the object, signalling proxied to the returned endpoint, the object's session id kept in `gw_sessions.body`, renewed by the tick and released when the gateway's lease ends; a family pool keeps only its own models' live caps |
| deploy | `scripts/serve/cf-edge.sh` | R2 `fv-edge-staging-outputs` (lifecycle: abort incomplete multipart uploads after 1 day, delete objects after 7), `FV_UPLOAD_SIGNING_KEY` secret, `status family:<f>` |
| GPU driver | `scripts/serve/e2e/do-family-pod.sh` | one worker on two families + a gateway on the e2e pod: burst, kill drill, director session |

fv-control does **not** show the per-family queue depth yet (follow-up): the
signal is `GET /families/{f}/metrics` (and `cf-edge.sh status family:<f>`);
adding it to fv-control needs a new Worker secret on its staging deploy,
which is another service's change.

### 14.2 Tests

- `fastvideo-dispatch-proto`: 29 unit tests (16 new: credits, arbiter nack,
  v2 ack timeout, v2 redeploy re-announce, sessions ×3, uploads ×2, metrics,
  presigner ×5).
- `fastvideo-serve`: `arbiter` (8 threads × 50 rounds of concurrent offers
  from two families: never more than one taken), `upload` (presigner equal to
  serve-kit's `S3Config::presign` on R2 / AWS / path-style endpoints).
- `fastvideo-media`: `fragmented_mp4_is_append_only` (real ffmpeg: every byte
  on disk while frames are written is unchanged at the end; `ftyp`, `moov`,
  `moof…`; 96 frames decode).
- `tests/edge_family.rs` (native family objects + an S3 mock that checks
  every SigV4 signature): **7/7** — two-family burst (12 jobs, 19–21 offers,
  7–9 arbiter 429s, no worker ever ran two jobs at once, each job through its
  own family's pool), 429 → re-offered elsewhere (queue 2 ms), ack timeout →
  requeue to another worker under a new lease, object restart during jobs →
  workers back in 0.26 s, every job offered once and finished where it ran,
  direct upload committed (key, bytes, sha256), tail upload (6 of 7 parts sent
  while the file grew; a header patch re-sent part 1 only; 31 ms from finish
  to commit), session admission (Reactor through the gateway; the GPU's batch
  job waits while the session holds it; a second session gets 429; stop frees
  the GPU and the job runs).
- **The same tests against the staging Worker** (`FV_EDGE_URL`, real wasm
  object, R2 binding): burst, direct upload (object read back through `/dl`,
  24,721 bytes) and session admission (0.2 s) pass; the four that need the
  stand-in's internals skip.
- `FV_SERVE_HEAVY=1 scripts/serve/check.sh` (build pod): check, clippy
  (native and wasm32) and every test pass except
  `ingest_whip::whip_ingest_round_trips_the_camera_through_the_echo`. That test
  needs the VP8 echo encoder to keep real time (it failed with 24 frames sent,
  612 dropped, in 29 s; the passing run on main took 22 s). It failed only on
  a loaded shared build pod. The diff touches no ingest, WebRTC or echo code,
  and every new path in `App::build` is gated on `dispatch.*` settings that
  test does not set. A re-run on an idle pod is pending: the Runpod balance
  went negative before it could run.

### 14.3 Staging and one GPU (2026-10-02)

- Staging Worker `fv-edge-staging` version `9e1c237-20261002T115751Z`, family
  objects hinted to `weur`.
- One RTX PRO 6000 (EUR-IS-1, EU volume read-only, pod `32l8nou003heyf`,
  runtime image + this branch's `fv-serve`): one worker with
  `fastwan22-ti2v-5b` (family `wan`) and `longlive-1.3b` (family `sfwan`) in
  one process (both non-VSA; `fastwan21-1.3b` cannot share a process with
  SF-Wan), loaded in 250 s, holding sockets to `family:wan` and
  `family:sfwan`. A gateway on the same pod.
- **Burst, 3 + 3 jobs at once**: all 6 succeeded, one after another (capacity
  1), every output committed through the object (direct upload, fragmented
  MP4, no remux). From the engine's last frame to the stored result:
  **1.19–1.67 s** (outputs of 1.0–3.2 MB, a single part each: overlap needs
  outputs above the 5 MiB part size, so `early` was 0 here).
- **Bug found and fixed** (fb1bb40): the gateway took a family pool's live
  caps from its workers' hellos, and a two-family worker reports both
  families' models, so `p-wan` claimed LongLive and all six jobs went through
  `family:wan`. A family pool now keeps only its own models; the burst test
  asserts each job's pool.
- **Not run on the GPU** (the pod's backstop ended the window, then the
  Runpod balance went negative): the classic-path baseline, the kill
  drill, and the LongLive director session through the object. The
  native and staging tests cover these paths (§14.2).
- Spend: one pod for 60 min at $2.09/h ≈ **$2.09**; deleted by its backstop
  (`GET /pods/32l8nou003heyf` → 404).

### 14.4 Before this becomes the default

The criteria of §12.3, plus: the GPU numbers above (classic vs DO time to
result on outputs larger than a part, a kill drill, a director session), the
`ingest_whip` re-run on an idle pod, per-family queue depth in fv-control,
and the production items of gateway-cloudflare.md §9.9 (production Worker on
`fv-jobs`, the gateway's artifact bucket bound as `OUTPUTS`, a least-privilege
token, alerts, a deploy runbook).
