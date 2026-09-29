# fv-control: the cluster controller

Status: staging, 2026-09-29. Code: `control/` (a Cloudflare Worker), the CLI
`scripts/serve/fv-control.sh`, and log shipping in fv-serve
(`crates/fastvideo-serve/src/log_ship.rs`).

**Staging:** <https://fv-control-staging.maximalize.workers.dev> (Worker
`fv-control-staging` on the account's workers.dev).

fv-control starts and stops gateway clusters on Runpod. It also:

- keeps env vars at three levels: account, cluster and pod;
- starts and stops a cluster's gateway on its own;
- shows each pod's logs;
- tracks what every pod in the account costs;
- gives a dashboard of balance, burn, idle time, excessive spend and
  CPU/GPU utilisation, with alerts.

Everything the UI does is also a JSON API.

## 1. Architecture

```
browser / fv-control.sh / agents ──► Worker fv-control-staging (Hono, TypeScript)
                                       │  /api/*  JSON API (auth below)
                                       │  /ingest/v1/logs  (per-cluster ingest token)
                                       │  static dashboard (public/, Workers static assets)
         cron * * * * * ──────────────►│  collector.ts: snapshot every pod, costs, alerts, backstops
                                       ├─► D1 fv-control   clusters, pods, env, costs, alerts, audit, log tail
                                       ├─► D1 fv-jobs      releases, deployments (read only)
                                       ├─► R2 fv-control-logs   log archive (NDJSON, 30-day lifecycle)
                                       ├─► Analytics Engine fv_control_metrics   per-minute samples
                                       └─► Durable Object ClusterOps (one per cluster)
                                             an operation at a time, a step machine on alarms
                                             (up, down, extend, scale, roll, restart, gateway start/stop),
                                             and live-tail WebSockets (hibernation API)
   Runpod REST + GraphQL + pod log endpoint, GHCR, GitHub, the gateways' admin routes
```

**Language: TypeScript (Hono), not workers-rs.** The reasons:

- The controller is an HTTP/JSON and CRUD application. It shares no code
  with the Rust crates. What it ports from `runpod-cluster.sh` is
  shell logic: payloads, env and the boot commands.
- WebCrypto does everything the controller needs, with no dependencies.
  That covers X25519 for the sealed admin token, AES-GCM, PBKDF2 and HMAC.
  Its unit tests run the same code under Node.
- The iteration loop is `tsc` + vitest + `wrangler dev` in this container
  in seconds. A Rust build on the build pod takes minutes. The Worker's
  bundle is 54 KiB gzip and it starts in 2 ms.
- The workers-rs dispatcher in `crates/fastvideo-edge` is a hot-path
  scheduler, where Rust pays off. The controller is a separate Worker with
  its own bindings and deploys. The two share no code or state.

**Why one Durable Object per cluster.** A rolling redeploy waits up to 30
min for a new worker and 15 min for a drain. The DO keeps the operation in
its storage and advances it one step per alarm (10 ms to 20 s apart), and
it runs at most one operation per cluster. D1 holds the cluster's spec and
state, and each operation's log (`operations`), so the UI and the cron see
the same thing.

### Files

| file | |
|---|---|
| `control/src/index.ts` | routes (the API) and the cron entry |
| `control/src/auth.ts` | Access JWT, passphrase sessions, CSRF, API tokens |
| `control/src/cluster/spec.ts` | the cluster definition, templates `standard` and `tiny-cpu` |
| `control/src/cluster/payloads.ts` | the port of runpod-cluster.sh: boot commands, env, pod payloads, gateway TOML |
| `control/src/cluster/ops.ts` | create / patch / delete, price projection, admin token, probes |
| `control/src/cluster/do.ts` | `ClusterOps`: the operations |
| `control/src/collector.ts` | the per-minute snapshot, costs, alerts, backstops, retention |
| `control/src/envvars.ts` | env layers |
| `control/src/logs.ts` | ingest, search, download |
| `control/src/metrics.ts` | Analytics Engine writes and SQL, the Prometheus parser |
| `control/src/github.ts`, `ghcr.ts`, `releases.ts` | release dispatch, CI, image tags, drift |
| `control/migrations/0001_init.sql` | the D1 schema |
| `control/public/` | the dashboard (no framework, SVG charts) |

## 2. Auth

The controller holds the account's Runpod key, so every route but
`/healthz`, `/api/auth/*` and the ingest endpoint needs auth. It supports
two modes.

**Cloudflare Access (preferred).** This mode is on when the vars
`ACCESS_TEAM_DOMAIN` (`https://<team>.cloudflareaccess.com`) and
`ACCESS_AUD` (the application's AUD tag) are set:

- Every `/api` request needs a valid `Cf-Access-Jwt-Assertion` (or
  `CF_Authorization` cookie). The Worker checks it: RS256 against the
  team's JWKS, `aud`, `iss` and `exp`.
- The identity must be in `OWNER_EMAILS` when that is set.
- Scripts use an Access service token, which Access turns into a JWT.
- The Access application must bypass `/ingest/*`: pods authenticate
  there with their cluster's ingest token.

**Owner passphrase (the fallback, and what staging uses).**

- `OWNER_PASSPHRASE_HASH` is a Worker secret:
  `pbkdf2-sha256$100000$salt$hash`. The passphrase is first HMAC'ed with
  the `SESSION_SECRET` pepper, then run through PBKDF2-SHA256 (Workers
  allow at most 100 000 iterations).
- The passphrase is generated with about 125 bits of entropy, which is
  what makes it hard to guess. The KDF slows down guessing if the hash
  and pepper ever leak.
- Login sets `__Host-fvc_session` (HttpOnly, Secure, SameSite=Strict,
  12 h). The cookie is an HMAC-signed session id, backed by the `sessions`
  table, so logout revokes it.
- Every mutating request on a session needs `x-csrf-token` (an HMAC of the
  session id, returned by login and `/api/auth/me`) and a same-origin
  `Origin`.
- Rate limits (in D1): 5 login attempts per IP per 15 min, and 30 per
  hour in total.

**API tokens** (Settings → API tokens, or `POST /api/tokens` from a
session) are for `fv-control.sh` and agents:

- A token is `fvc_…`, stored as SHA-256, with scope `read` or `admin` and
  a 90-day expiry. It can be revoked.
- A read token cannot mutate, and no token can mint tokens.
- Tokens work in passphrase mode. Behind Access they also need a service
  token.

**Audit.** Every mutating action writes a row to `audit`: who, what, when,
before and after, ok, detail and IP. That includes failed logins, token
mints, spec and env changes (secret values masked), every operation,
admin-token reveals, key mints, release dispatches, policy changes and
the policy engine's own actions (`policy:*`).

**Choice for staging: passphrase.** Zero Trust is not enabled on the
account (`access/apps` answers "Access is not enabled"), and the deploy
token has no Access permissions. To switch to Access:

1. Enable Zero Trust in the dashboard.
2. Create a self-hosted Access application for
   `fv-control-staging.maximalize.workers.dev`, with an allow policy for
   the owner's email and a bypass for `/ingest/*`.
3. Set `ACCESS_TEAM_DOMAIN`, `ACCESS_AUD` and `OWNER_EMAILS` as vars and
   redeploy.

**How the owner logs in.** Open the URL and enter the passphrase stored in
`/root/.config/fv/fv-control-staging-passphrase` (mode 600, on the agent
container). To rotate it: hash a new passphrase with `hashPassphrase` in
`control/test/harness.mjs` (using the same `SESSION_SECRET`), then run
`wrangler secret put OWNER_PASSPHRASE_HASH --env staging`.

### Secrets

These are Worker secrets (`wrangler secret put/bulk --env staging`). They
are never in code, D1, logs or responses. `scrub()` removes them from every
upstream error message, audit detail and ingested log line.

| secret | use |
|---|---|
| `RUNPOD_API_KEY` | Runpod REST, GraphQL and the pod log endpoint. It is also passed to the gateway pod as `FV_BACKSTOP_API_KEY` for its watchdog, as the script does |
| `CLOUDFLARE_API_KEY` | the Analytics Engine SQL API (charts) |
| `GITHUB_PAT` | `release.yml` dispatch, CI status. Given to gateways as `FV_GITHUB_TOKEN` when the spec says `gateway.github_token` (standard template: on; tiny-cpu: off) |
| `CONTROL_KEK` | 32 bytes; AES-256-GCM for cluster secrets and secret env values in D1 (associated data = the row) |
| `SESSION_SECRET` | 32 bytes; session and CSRF HMACs, passphrase pepper |
| `OWNER_PASSPHRASE_HASH` | see above |

A cluster's own secrets are generated by the controller and sealed in D1:
internal token, URL-signing key, X25519 admin key pair, the gateway's admin
token once opened, and the log ingest token (also stored as a SHA-256).
Local copies of the staging values are in `/root/.config/fv/` (mode 600):
`fv-control-staging-secrets.json`, `fv-control-staging-passphrase` and
`fv-control-token` (the CLI's API token).

### Cloudflare credentials

The owner's token (`/root/.config/fv/cf_api_token`) deploys the Worker
and is also the Worker's `CLOUDFLARE_API_KEY`. It **cannot mint tokens**:
`/user/tokens` and `/accounts/<id>/tokens` answer 9109 "Unauthorized". So
no narrower `fv-control-staging-*` token was created.

To let agents create least-privilege tokens, add **User → API Tokens:
Edit** (or **Account → Account API Tokens: Edit**) to the base token. Then
create:

| name | permissions | purpose |
|---|---|---|
| `fv-control-staging-deploy` | Account: Workers Scripts Edit, D1 Edit, Workers R2 Storage Edit, Account Analytics Read | deploys (shareable with the fastvideo-edge DO agent: same scopes) |
| `fv-control-staging-runtime` | Account Analytics Read | the Worker's `CLOUDFLARE_API_KEY` (read-only AE SQL) |

To enable Cloudflare Access, the token also needs **Access: Apps and
Policies Edit**, **Access: Organizations, Identity Providers, and Groups
Edit** and **Access: Service Tokens Edit**, and Zero Trust must be
enabled once in the dashboard.

## 3. Deployment

```bash
scripts/serve/fv-control.sh deploy staging     # npm ci, tsc, unit tests, D1 migrations, wrangler deploy
```

Resources, created once:

- D1 `fv-control` (`b883b6a2-…`).
- R2 `fv-control-logs`, with lifecycle rule `expire-logs` (prefix
  `logs/`, 30 days).
- The Analytics Engine dataset `fv_control_metrics`, created on the first
  write.
- DO class `ClusterOps` (SQLite-backed).
- Cron `* * * * *`.

`wrangler.toml` binds D1 `fv-jobs` read-only as `JOBS_DB`, for releases
and the deployment registry. `PUBLIC_URL` is the address pods ship logs
to.

A production controller would be an `[env.production]` block with its own
D1, R2, secrets and URL. It also needs Access first.

## 4. Clusters

A **spec** (`control/src/cluster/spec.ts`) holds:

- `image`: exactly one of `channel` (`stable`, `latest`, …), `sha`, or
  `ref`. A channel or sha gives per-variant images (`<variant>-<channel>`,
  `<variant>-sha-<sha>`). A ref gives one all-in-one image. Images are
  resolved to digests at start, and `:stable` falls back to `:latest` as
  in the scripts.
- `regions`: `eu` is volume `jg48s6o1w0` in EUR-IS-1 (RTX PRO 6000); `us`
  is `s2k01690bi` in US-CA-2 (H100/H200).
- `gateway`: CPU flavors, vCPU, disk, TOML base (`pods` is
  `gateway-pods.toml`; `minimal` drops the reactor and fal apps), auth
  mode, and `github_token`.
- `pools`: `id`, `variant`, `count`, `compute` GPU/CPU, `config` (a path in
  the image) or `config_toml` (inline, sent as `FV_WORKER_TOML_B64`), GPU
  types, regions, static caps (`models` or `fake_models`), and queue and
  timeout limits.
- `cap_s` (backstop), `min_balance` (the pod watchdog's floor),
  `balance_floor` (the controller's floor, at least $8), `min_start`,
  `max_gpu_dph`, `auto_stop_idle_min`, `log_shipping`.

The `standard` template is the script's cluster: a cpu3c gateway and one
worker each in h3-turbo, h3-max, ltx and wan (wan5b). The `tiny-cpu`
template is a CPU gateway plus one CPU worker on the gateway image with the
fake engine.

**Operations** (`POST /api/clusters/<id>/<op>`, one at a time per cluster;
each op's log is in `/api/ops/<id>`):

| op | what it does |
|---|---|
| `start` | Price check (below). Resolve the digests and set the deadline (now + `cap_s`). Create the gateway: preferred DC first, then any DC, trying each CPU flavor. Create the workers: each region × GPU type in order; a pod over `max_gpu_dph` is deleted at once. PATCH the gateway with `FV_POOL_<ID>_URLS` and `FV_CLUSTER_PODS`. Wait until every pool has a ready worker (the admin pools view, ≤ 30 min) |
| `stop` | Delete every pod, verify that each is gone (retrying up to 5 times), then clear the state |
| `extend {minutes}` | Refused if the account's burn would take the balance below the floor before the new deadline. PATCHes the gateway (its watchdog holds the deadline) |
| `scale {pool, count}` | Up: projection, create, PATCH the gateway. Down: drain (`POST /fv/v1/internal/drain`), wait until idle (≤ 15 min), drop the pod from the gateway, delete it |
| `roll {target, pools?, gateway?}` | The 7-step rolling redeploy of `release.sh redeploy`: new workers, gateway sees both, wait for `/health` AVAILABLE with the target digest, drain the old ones, wait until idle, gateway sees only the new ones (plus the image when `gateway`), delete the old ones. Aborts by deleting the new pods and pointing the gateway back. Needs at least 40 min before the deadline and `min_start` |
| `restart {pods?}` | Rolling env apply: PATCH workers one at a time, then the gateway, each only after the previous one answers again |
| `gateway/stop`, `gateway/start` | Runpod stop/start of the gateway pod (same id and URL; container disk and admin token are new). With no gateway, `start` creates one and re-points the workers |

**Parity with the script.** Unit tests check these byte for byte against
`runpod-cluster.sh`:

- the gateway boot command, with its watchdog;
- the worker boot command, plus the inline-config branch;
- the embedded gateway TOML base, against `configs/serve/gateway-pods.toml`.

The payloads are the script's. Workers expose `8000/http` and
`70000/tcp` (GPU); the network volume is mounted at `/workspace`. The env
is the script's, including the Runpod secret references, `FV_IMAGE_*`,
`FV_BACKSTOP_API_KEY`, `FV_ADMIN_TOKEN_RECIPIENT` and the admin token
flow. The admin token comes from `GET /fv/v1/admin/token/sealed`, opened
with WebCrypto X25519; a test checks it against the script's own openssl
`open_sealed`. On a 401 the controller fetches the token again.

**Importing a script cluster.** `POST /api/clusters/import {state,
admin_key_pem?, name?}`, or `fv-control.sh import cluster.json
[cluster.json.admin-key.pem]`:

- It takes the script's `cluster.json` as is, including `workers`,
  `rolling` and `retired`.
- A legacy state (its own `FV_ADMIN_TOKEN`, no key pair) keeps passing
  that token.
- The imported pods show "needs restart" until their env is re-applied,
  because log shipping is new to them.
- The script keeps working and its backstops are unaffected. Use only one
  tool per cluster after the import.

**Price check and floor.** `POST /api/clusters/<id>/price` and every
`start`:

- For each pod, the $/hr of the most expensive GPU type it may land on
  (Runpod `securePrice`), or the CPU flavor's estimate.
- The account's current burn plus the cluster's, times the hours to the
  deadline.
- Refused if the balance is below `min_start`, or the projection is below
  `max(balance_floor, BALANCE_FLOOR=8)`.
- Scale-up and extend apply the same check.

**Backstops**, three and independent:

1. The gateway pod's watchdog (the script's) deletes the workers and
   itself at the deadline, or below `min_balance`.
2. The controller's cron: a cluster past its deadline gets a `stop`,
   cancelling any running operation. If the DO fails, the pods are
   deleted directly.
3. The balance floor rule: below the floor, the cron stops every
   controller cluster.

## 5. Environment variables

| level | scope id | applies to |
|---|---|---|
| account ("Runpod level") | – | every controller cluster's pods |
| cluster | cluster id | that cluster's pods |
| pod | Runpod pod id | that pod |

- **Resolution:** pod > cluster > account > system. The system keys
  (`RESERVED_KEYS` in `payloads.ts`) are the controller's own: tokens,
  deadline, pod list, pool URLs, boot config, `FV_IMAGE_*`,
  `FV_GITHUB_TOKEN`, log shipping. Setting them is refused with 400.
- A secret variable (`secret: true`) is sealed in D1 and shows as
  `••••••••` in every view and in the audit log. System secrets are masked
  the same way. Runpod secret references (`{{ RUNPOD_SECRET_… }}`) are
  shown and flagged as references.
- `GET /api/clusters/<id>/env` returns each pod's effective env: masked,
  with its source and what it overrides. It also returns `needs_restart`:
  the pods whose applied env hash (recorded at create or PATCH) differs from
  the desired one. Runpod applies an env change only by a PATCH, which
  restarts the container.
- "Apply with a rolling restart" (Env page, or `POST …/restart`) runs the
  `restart` operation.

## 6. Logs

**What Runpod exposes:** the public REST API has no log route
(`/v1/pods/<id>/logs` is 400). The console's endpoint
`https://hapi.runpod.net/v1/pod/<id>/logs` answers with the API key:
`{container: [...], system: [...]}`, the tail only (about 70 lines each),
with no history or search. It is undocumented. The controller shows it as
"Runpod container log" for any pod, including external ones, and scrubs
secrets from it.

That is not enough, so **fv-serve ships its logs**
(`crates/fastvideo-serve/src/log_ship.rs`, off by default):

- A tracing layer turns each event into
  `{ts, level, target, fields: {message, …, plus the enclosing spans'
  fields such as job_id and session_id}}`.
- It `try_send`s into a 10 000-line queue and drops when the queue is
  full. It never blocks.
- A tokio task POSTs `{pod, lines}` batches to `FV_LOG_SHIP_URL` with
  `Authorization: Bearer $FV_LOG_SHIP_TOKEN`, every `FV_LOG_SHIP_INTERVAL_MS`
  (2 s) or every `FV_LOG_SHIP_BATCH` (200) lines.
- It stops at `FV_LOG_SHIP_MAX_PER_MIN` (6000) lines per minute. Dropped
  lines are counted and reported in a WARN line.
- It retries 5xx and 429 errors twice with backoff.
- It never ships the HTTP client's own events.
- It is filtered by `FV_LOG_SHIP_LEVEL` (info) after `RUST_LOG`.
- The controller sets these variables on every pod when the spec has
  `log_shipping` (the default). Images built before this change ignore
  them.

**Ingest** (`POST /ingest/v1/logs`):

- It needs the cluster's ingest token, and the pod must belong to that
  cluster.
- Limits: 1 MiB and 2000 lines per batch, 600 batches per minute per
  cluster.
- Each batch goes to R2 at `logs/<cluster>/<pod>/<day>/<hour>/<ts>.ndjson`,
  kept 30 days (bucket lifecycle).
- The lines also go to D1 `log_lines`, the searchable 24 h tail, and to
  the cluster's DO for live-tail WebSockets.

**UI** (Logs page):

- choose a pod;
- filter by level and search text (message and fields, so a job id finds
  its lines);
- live tail over a WebSocket (`/api/logs/tail?pod=`);
- download a day's NDJSON from R2 (`/api/logs/download?pod=&day=`);
- the Runpod tab.

## 7. Cost model

The collector runs every minute (`collector.ts`):

- One GraphQL call gets every pod of the account: `desiredStatus`,
  `costPerHr`, GPU count and type, DC, image, uptime, per-GPU utilisation
  and memory, and container CPU and memory, plus `clientBalance` and
  `currentSpendPerHr`.
- For each running controller cluster, the gateway's admin
  `/fv/v1/gateway/pools` gives per-worker running and queued jobs, readiness
  and build. Its `/metrics` is parsed for a whitelist of series only.
- **Cost accrues for RUNNING pods**: `costPerHr × elapsed` since the last
  sample, capped at 5 min, into `cost_daily(day, pod)`, along with minutes
  and idle minutes. Volume storage of stopped pods is not counted.
- **Attribution:** a controller pod gets `cluster:<name>`. Anything else
  is external, matched by name prefix (`attribution` policy: `fv-build`
  → `external:build-pod`, `fv-cluster-` → `external:runpod-cluster.sh`,
  `fv-b200` → `external:b200-bench`, `loom-` → `external:loom`, …). Failing
  that, the name's first two dash parts, else `external`. External pods
  are counted everywhere and never touched.
- **Dashboard numbers:**
  - balance, and its change over the last hour;
  - burn in $/hr (Runpod's);
  - time to floor: (balance − floor) / burn;
  - spend today;
  - idle burn;
  - per-cluster $/hr and cost today;
  - spend per day by owner (7 days).

## 8. Alerts and policies

These are the settings (`/api/policies`, Settings page). Alerts open, refresh
and resolve every minute.

| alert | when | default | auto-action |
|---|---|---|---|
| `pod_idle` | a GPU pod with GPU < `idle_gpu_pct` (5 %) and no running jobs for `idle_min` (30 min) | on | `auto_stop_idle` (off): after `auto_stop_idle_min` (60) the idle worker of a **controller** cluster is drained and removed (a `scale` op); per cluster `auto_stop_idle_min` in the spec |
| `cluster_dph` | a cluster's $/hr > `cluster_dph_max` (15) | on | – |
| `daily_spend` | today's spend > `daily_spend_max` (150) | on | – |
| `balance_margin` | balance < floor + `balance_margin` (10) | on | – |
| `balance_floor` | balance < floor ($8), or < a cluster's own `balance_floor` | on | `stop_on_floor` (**on**): stop controller clusters |
| `deadline` | < 15 min to a cluster's deadline (info); passed (critical) | on | **always**: stop the cluster |
| `pod_down` | a controller pod not answering for `pod_down_min` (10) | on | – |

Everything is notify-only except the two rules the scripts already had:
the deadline backstop and the balance floor. Auto-actions never touch
external pods. Pod health in the pod views comes from the gateway's view
(ready, loading or down).

## 9. Observability

This follows the owner's "lean" guidance.

- **Metrics:** fv-serve's Prometheus `/metrics` remains the only metrics
  source. There is no OpenTelemetry SDK, collector, node_exporter,
  dcgm-exporter or sidecar. GPU, CPU and memory come from the Runpod
  runtime metrics API.
- **Collection is a pull.** The cron reads Runpod's GraphQL and each
  gateway's admin `/metrics`, and parses only `PROM_WHITELIST`
  (`fv_pool_{queued,running,workers,available,streams,oldest_queued_seconds,submitted_total}`,
  `fv_gateway_{dispatched,lost,redispatched}_total`,
  `fv_jobs_{submitted,finished}_total`, `fv_ready`).
- **Storage:**
  - High-frequency series go to **Workers Analytics Engine**, dataset
    `fv_control_metrics`:
    - pod rows: blobs `pod, pod_id, owner, cluster, name, gpu, status`;
      doubles `$/hr, gpu%, gpu mem%, cpu%, mem%, uptime, jobs running,
      jobs queued, idle`;
    - pool rows (per gateway pool);
    - account rows.
  - AE is cheap, keeps 3 months automatically, and is read through its SQL
    API (`metrics.ts`, `avgIf` per bucket).
  - D1 holds rollups (`cost_daily`), the latest pod view (`pods`), balance
    samples (30 days), alerts and the audit log.
  - Without the AE binding (local dev, tests), samples go to D1
    `pod_samples` (24 h) and the charts read that instead.
- **Logs:** structured JSON from fv-serve, shipped to R2 (section 6). This
  is the only new fv-serve code, and it is off by default.
- **Traces:** none for now. The `otel` cargo feature of fastvideo-serve is
  an empty seam, off by default: a `tracing-opentelemetry` layer would go
  next to the log-shipping layer in `main.rs`'s `init_tracing` (the
  registry already takes optional layers).
- **Grafana (not built):** if Grafana is wanted later, fv-control's cron
  can push the same whitelisted samples to a Grafana Cloud Prometheus
  `remote_write` endpoint. That needs snappy-compressed protobuf, a
  `GRAFANA_REMOTE_WRITE_URL` and a `GRAFANA_API_KEY` secret. Or Grafana
  can query AE's SQL API directly with the ClickHouse-compatible data
  source. The export belongs in the controller, not in the pods.

## 10. API

All responses are JSON. Auth is a session cookie plus `x-csrf-token`, or
`Authorization: Bearer fvc_…`.

| route | |
|---|---|
| `GET /api/overview` | the dashboard numbers, clusters, open alerts, idle pods |
| `GET /api/pods`, `/api/pods/<id>`, `/api/pods/<id>/runpod-logs` | account pods with owner, health, utilisation, jobs, cost |
| `GET /api/metrics/series?hours=&pod=`, `/api/balance?hours=`, `/api/costs?days=` | time series and cost breakdowns |
| `GET/POST /api/clusters`, `GET /api/templates`, `GET/DELETE /api/clusters/<id>`, `PUT …/spec` | definitions |
| `POST /api/clusters/<id>/{price,start,stop,extend,scale,roll,restart,gateway/start,gateway/stop,cancel}` | operations (202 + operation id) |
| `GET /api/clusters/<id>/ops`, `/api/ops/<id>` | operation logs |
| `GET /api/clusters/<id>/env` | effective env per pod (masked), `needs_restart` |
| `GET /api/clusters/<id>/gateway`, `POST …/admin-token`, `POST …/mint-key` | the gateway's status and pools view; reveal the admin token (audited); mint a user API key |
| `POST /api/clusters/import` | adopt a runpod-cluster.sh state |
| `GET /api/env/account`, `GET /api/env/<scope>/<id>`, `PUT/DELETE /api/env/<scope>/<id>/<KEY>` (`PUT /api/env/account/<KEY>`) | env layers |
| `GET /api/alerts`, `POST /api/alerts/<id>/resolve`, `GET/PUT /api/policies` | alerts and policies |
| `GET /api/logs?pod=&q=&level=&since=`, `/api/logs/download?pod=&day=`, `/api/logs/tail?pod=` (WebSocket) | logs |
| `GET /api/releases`, `/api/images/tags?filter=`, `/api/github/ci`, `POST /api/github/release` | channels, drift, registry, GHCR tags, CI on main, promote/rollback dispatch |
| `GET/POST /api/tokens`, `DELETE /api/tokens/<id>`, `GET /api/audit` | tokens and audit |
| `POST /api/collect` | run the collector now |
| `POST /ingest/v1/logs` | log ingest (cluster ingest token) |

The CLI `scripts/serve/fv-control.sh` wraps these; run it without
arguments for help.

## 11. Tests

```bash
cd control
npm test                      # vitest (unit)
npm run test:integration      # wrangler dev + mocked upstreams
npm run test:ui               # headless Chromium (npx playwright-core install chromium-headless-shell)
```

**Unit tests** (`test/unit`, a node:sqlite D1 shim):

- sealing at rest;
- the passphrase KDF and pepper;
- the sealed admin token: round trip and tamper, plus interop with an
  openssl key and the script's `open_sealed`;
- byte parity with the script's boot commands and the gateway TOML base;
- env and payload shapes;
- spec validation;
- env resolution and masking;
- the Prometheus parser;
- AE SQL injection safety;
- attribution, alert lifecycle and rate limits;
- secret scrubbing;
- release dispatch validation;
- drift.

**Integration** (`test/integration/run.mjs`, 18 steps) runs the Worker
under `wrangler dev` (workerd with local D1, R2 and DO) against
`test/harness.mjs`. The harness mocks Runpod REST, GraphQL and hapi, GHCR,
GitHub and the Cloudflare API, and simulates gateway and worker pods,
including the sealed token. The steps cover:

- auth, CSRF and Origin, forged cookies, token scopes, and the login rate
  limit;
- the floor refusing a start;
- a full start with every payload and env checked;
- the collector, attribution, costs and the idle alert;
- env layers and a rolling restart (workers first, then the gateway);
- ingest, search, level filter, live tail, download, auth and scrubbing;
- scale up and down, with a drain;
- a roll with the gateway, checking digests;
- extend, gateway stop and start, admin token and key mint;
- GitHub dispatch and CI;
- the deadline backstop and the balance-floor stop;
- importing a script state;
- the audit log, and that no secret appears in any response.

**UI:** login, the dashboard with its charts and tooltip, every page, a
secret env var set through the UI and never rendered, dark mode, and no
horizontal scroll at 390 px. Screenshots go to `control/test-results/`.

**fv-serve:** the `log_ship` unit tests (config from env, JSON lines with
span fields, dropping when full), run on the build pod.

## 12. Live test (2026-09-29, staging, real Runpod account)

**Read-only first.** The staging cron snapshots the account every minute:

- 15 to 19 pods, all attributed. Examples: `fv-build` → `external:build-pod`,
  `fv-a2vg-…` → `external:fv-a2vg`, `loom-wan-test-pod` → `external:loom`.
  The daydreamlive and vidu pods, which are stopped, show as `external`.
- Balance, burn, per-minute costs, and GPU/CPU series from Analytics
  Engine.
- The Runpod log tail of the build pod.
- Channel heads and the registry from fv-jobs (`stable` = `2cd1ba0`).
- CI on main and GHCR tags.

**One lifecycle** with the `tiny-cpu` template (cluster `live-tiny`,
`cap_s` 2400):

1. The price check projected $0.12/hr and a balance of $30.33 at the
   deadline.
2. `start` created a cpu3c gateway on `gateway-stable` (sha256:00d85291…)
   and one cpu3c fake-engine worker on the same image, then PATCHed the
   gateway with `FV_POOL_FAKE_URLS`.
3. Setting a cluster env var marked both pods "needs restart". The rolling
   `restart` did the worker first (back in 36 s), then the gateway (20 s).
4. The controller minted a user key through the gateway, and a
   `fake-wan` job went through the gateway to the worker and `succeeded`
   (401 without the key).
5. `extend 10` moved the gateway's `FV_CLUSTER_DEADLINE`.
6. `gateway/stop` and `gateway/start` worked; the pools view came back.
7. `stop` deleted both pods, and Runpod answered 404 for each. The
   cluster's `cost_daily` shows $0.04 over 44 pod-minutes.

The test found two version-skew problems between main's configs and the
promoted `stable` image. Both are fixed:

- **The gateway TOML.** `gateway-pods.toml` on main has
  `inline_inputs_max_bytes`, `input_passthrough` and
  `stage_inputs_for_retry`. The `2cd1ba0` gateway rejects them as unknown
  fields and crash-loops. The `minimal` base now leaves them out, so their
  defaults apply. `runpod-cluster.sh up stable` has the same problem, since
  it ships main's file to an older image.
- **The sealed admin token.** The `2cd1ba0` image has no
  `/fv/v1/admin/token/sealed` route. When the gateway is healthy and that
  route answers 404, the controller switches the cluster to a token it
  makes itself. That token is passed as `FV_ADMIN_TOKEN`, sealed in D1 and
  masked in every view. The gateway then shows "needs restart", and one
  rolling restart applies it.

Log shipping could not be exercised live: `2cd1ba0` predates
`log_ship.rs`. The pods ignored the `FV_LOG_SHIP_*` variables, and the
Runpod log tab showed their output. Shipping is covered by the fv-serve
unit tests and the integration test's ingest path. It turns on for a
cluster started from an image built after this change.
