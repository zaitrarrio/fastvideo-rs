# fv-control: the cluster controller

Status: staging, 2026-09-29. Code: `control/` (a Cloudflare Worker), the CLI
`scripts/serve/fv-control.sh`, and log shipping in fv-serve
(`crates/fastvideo-serve/src/log_ship.rs`).

**Staging:** <https://fv-control-staging.maximalize.workers.dev> (Worker
`fv-control-staging` on the account's workers.dev).

fv-control starts and stops clusters of fv-serve workers on Runpod, behind
the edge Worker (the only entry point; the gateway pod was retired
2026-10-06, [edge-control-plane.md](../serve/edge-control-plane.md)) or
called directly. It also:

- keeps env vars at four levels: account, cluster, pool and pod;
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
                                             (up, down, extend, scale, roll, restart),
                                             and live-tail WebSockets (hibernation API)
   Runpod REST + GraphQL + pod log endpoint, GHCR, GitHub, the edge (service binding EDGE), direct workers' admin routes
```

**Language: TypeScript (Hono), not workers-rs.** The reasons:

- The controller is an HTTP/JSON and CRUD application. It shares no code
  with the Rust crates. What it took from the retired `runpod-cluster.sh`
  is shell logic: payloads, env and the boot commands.
- WebCrypto does everything the controller needs, with no dependencies.
  That covers AES-GCM, PBKDF2 and HMAC.
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
| `control/src/cluster/spec.ts` | the cluster definition, templates and pool presets |
| `control/src/cluster/catalog.json` | editor suggestions: recipes, model ids, fal apps, engine env keys |
| `control/gen-configs.mjs` | generates `worker-configs.ts` from `configs/serve` |
| `control/src/cluster/payloads.ts` | boot commands, env, pod payloads (edge and direct workers) |
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
| `RUNPOD_API_KEY` | Runpod REST, GraphQL and the pod log endpoint. It is also passed to each edge worker as `FV_BACKSTOP_API_KEY` for its watchdog |
| `CLOUDFLARE_API_KEY` | the Analytics Engine SQL API (charts) |
| `GITHUB_PAT` | `release.yml` dispatch, CI status |
| `GITHUB_RUNNER_PAT` | optional, build pods (§8a): fine-grained, Administration read/write + Actions read on the repository; mints runner registration tokens, lists and removes runners, reads queued jobs and `scripts/dev/build-pod-server.py` / `build-pod.sh` at `main`. Unset: `GITHUB_PAT` is used |
| `BUILD_CACHE_R2_ACCESS_KEY_ID`, `BUILD_CACHE_R2_SECRET_ACCESS_KEY` | optional, build pods (§8a): an R2 key scoped to the bucket `fv-build-cache`, put into each build pod's env for sccache and the deps seeds |
| `CONTROL_KEK` | 32 bytes; AES-256-GCM for cluster secrets and secret env values in D1 (associated data = the row) |
| `SESSION_SECRET` | 32 bytes; session and CSRF HMACs, passphrase pepper |
| `OWNER_PASSPHRASE_HASH` | see above |
| `EDGE_INTERNAL_TOKEN`, `EDGE_ADMIN_TOKEN` | the edge Worker's `FV_INTERNAL_TOKEN` and `FV_ADMIN_TOKEN` (`control_plane: "edge"` clusters, below). With `EDGE_URL`, `EDGE_D1_DATABASE_ID` and `EDGE_OUTPUTS_BUCKET`, set by `scripts/serve/fv-control.sh edge-link staging` from `scripts/serve/cf-edge.sh`'s state |

A cluster's own secrets are generated by the controller and sealed in D1:
internal token, URL-signing key, a direct cluster's admin token, and the
log ingest token (also stored as a SHA-256).
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

- `control_plane`: `edge` (default) or `direct`. An **edge cluster**: the
  edge Worker (`EDGE_URL`) is its only entry point and
  every worker is an API front behind it
  ([edge-control-plane.md](../serve/edge-control-plane.md)). The workers
  get the edge's internal token and URL, `FV_DISPATCH_FRONT=1`, their
  families (from each model's family: `h3`, `ltx2` → `ltx`, causal `wan`
  recipes → `sfwan`, other `wan` → `wan`, fake models → `fake`; a pool's
  `family` overrides), direct uploads through the edge, the edge's D1
  (`EDGE_D1_DATABASE_ID`) instead of the account's job store, and the
  edge's outputs bucket (`EDGE_OUTPUTS_BUCKET`) as `FV_R2_BUCKET`, where
  their results land and their result URLs are presigned (never the
  account's production bucket; without it they get no R2). fv-control
  reaches the edge through the `EDGE` service binding: a Worker cannot
  fetch another `workers.dev` Worker of its account (error 1042). Their boot (`EDGE_WORKER_BOOT`) exports the pod's own proxy
  URL as `FV_DISPATCH_ENDPOINT` and runs a backstop watchdog: at the
  deadline or below `min_balance` each worker deletes its own pod (the
  cron's deadline backstop holds as well). `up` checks the edge (`register`)
  before any pod is made and waits until each pool has a ready front in the
  edge's families view. Keys, the admin token and the pools view are the
  edge's. One edge cluster runs at a time; a second one's `up` fails.
  An `extend` moves fv-control's deadline; the
  workers keep their launch deadline until they are restarted
  (`Env: apply`).

- `image`: exactly one of `channel` (`stable`, `latest`, …), `sha`, or
  `ref`. A channel or sha gives per-variant images (`<variant>-<channel>`,
  `<variant>-sha-<sha>`). A ref gives one all-in-one image. Images are
  resolved to digests at start, and `:stable` falls back to `:latest` as
  in the scripts.
- `regions`: `eu` is volume `jg48s6o1w0` in EUR-IS-1 (RTX PRO 6000). The
  default and the only available region is `["eu"]`. `us` (US-CA-2) is
  unavailable: Runpod deleted its weights volume (`s2k01690bi`) on about
  2026-10-05, and the owner chose EU only (2026-10-06). A spec that names
  `us` is rejected with a 400, not silently trimmed, so a saved cluster never
  changes placement behind the owner's back. A stored spec that still names
  `us` cannot start, scale, roll or restart (409) until it is edited, and
  placement skips `us` in any case. When US is rebuilt, setting
  `US_VOLUME_ID` in `control/src/cluster/regions.ts` brings the region back
  (`docs/ops/runpod-volumes.md`).
- `auth`: `keys` (default) or `none`: a direct cluster's client auth
  (`FV_AUTH_MODE`). An edge cluster uses the edge's own setting.
- A spec stored before the gateway was retired (a `gateway` block,
  `control_plane: "gateway"`) is read as today's: `enabled: false` gives
  `direct`, anything else `edge`; `gateway.auth` becomes `auth`; the pool
  variant `gateway` becomes `cpu`.
- `pools`: `id`, `variant`, `count`, `compute` GPU/CPU, `config` (a path in
  the image) or `config_toml` (inline, sent as `FV_WORKER_TOML_B64`), GPU
  types, regions, static caps (`models` or `fake_models`), and queue and
  timeout limits.
- `cap_s` (backstop), `min_balance` (the edge workers' watchdog floor),
  `balance_floor` (the controller's floor, at least $8), `min_start`,
  `max_gpu_dph`, `auto_stop_idle_min`, `log_shipping`.

The `standard` template is one worker each in h3-turbo, h3-max, ltx and
wan (wan5b). The `tiny-cpu` template is one CPU worker on the `cpu` image
with the fake engine. The `ltx`, `h3`, `wan` and `longlive` templates group
the pool presets below.

**Pool presets** (`POOL_PRESETS` in `spec.ts`; the dashboard's "Add pool"):
the four standard pools plus `ltx-pro` (`ltx25-distill-dense`, recipe
`ltx-pro`; Sage attention on by default on sm_120), `ltx-a2v`, `ltx-ref2v`,
`h3-ref2v`, `fastwan21` (the `wan` image, FastWan 2.1 1.3B), `sfwan` (SF-Wan
streaming) and `longlive` (LongLive-1.3B on the `sfwan` image;
**non-commercial** weights). They reuse the image variants CI builds; a
config the variant's image does not carry rides inline (`config_toml`,
generated from `configs/serve` by `node gen-configs.mjs`). A pool entry
with a preset's id and nothing else gets the preset. Recipes outside the
fv-serve catalog (the LongLive-Plug `*-plug-*` recipes) are refused: a
worker resolves every pool model against the catalog at start.

**Operations** (`POST /api/clusters/<id>/<op>`, one at a time per cluster;
each op's log is in `/api/ops/<id>`):

| op | what it does |
|---|---|
| `start` | Price check (below). Resolve the digests and set the deadline (now + `cap_s`). Edge: check the edge (`register`) first. Create the workers: each region × GPU type in order; a pod over `max_gpu_dph` is deleted at once. Wait until every pool has a ready worker (edge: a ready front in the edge's families view; direct: each worker's `/health`; ≤ 30 min) |
| `stop` | Delete every pod, verify that each is gone (retrying up to 5 times), then clear the state |
| `extend {minutes}` | Refused if the account's burn would take the balance below the floor before the new deadline. Moves fv-control's deadline; edge workers keep their launch deadline until restarted |
| `scale {pool, count}` | Up: projection, create, wait until ready. Down: drain (`POST /fv/v1/internal/drain`), wait until idle (≤ 15 min), delete it |
| `roll {target, pools?}` | Rolling redeploy: new workers, wait for `/health` AVAILABLE with the target digest (edge: ready at the edge), drain the old ones, wait until idle, delete them. Aborts by deleting the new pods. Needs at least 40 min before the deadline and `min_start` |
| `restart {pods?, pools?}` | Rolling env apply: PATCH workers one at a time, each only after the previous one answers again. With `pods` and/or `pools`: just those, whether or not their env changed; the dashboard's "Restart…" picks them |

**Direct clusters** (`control_plane: "direct"`): `start` creates only
the workers, and clients call each one at its pod URL. The controller makes
the cluster's admin token and passes it to every worker as `FV_ADMIN_TOKEN`
with `FV_WORKER_DIRECT=1`, so the workers check API keys themselves.
Minted keys live in the shared D1 `api_keys` table. Minting goes to one
worker; a revocation goes to all of them. See
[gateway-less-auth.md](gateway-less-auth.md).

**Payloads.** Workers expose `8000/http` and `70000/tcp` (GPU); the
network volume is mounted at `/workspace`. The env carries the Runpod
secret references, `FV_IMAGE_*`, the log-shipping variables and, on edge
workers, the backstop (`FV_CLUSTER_DEADLINE`, `FV_MIN_BALANCE`,
`FV_BACKSTOP_API_KEY`). Unit tests check the boot commands
(`EDGE_WORKER_BOOT`, the direct worker's) and the generated configs.

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

1. Each edge worker's watchdog deletes its own pod at the deadline, or
   below `min_balance`.
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
| pool | `<cluster id>:<pool id>` | every worker of that pool, including new ones (scale-up, roll) and after restarts |
| pod | Runpod pod id | that pod |

- **Resolution:** pod > pool > cluster > account > system. The system keys
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
- For each running edge cluster, the edge's families view
  (`/fv/v1/edge/families`) gives per-worker readiness, held jobs and build.
- **Cost accrues for RUNNING pods**: `costPerHr × elapsed` since the last
  sample, capped at 5 min, into `cost_daily(day, pod)`, along with minutes
  and idle minutes. Volume storage of stopped pods is not counted.
- **Attribution:** a controller pod gets `cluster:<name>`. Anything else
  is external, matched by name prefix (`attribution` policy: `fv-build`
  → `external:build-pod`, `fv-cluster-` → `external:runpod-cluster.sh` (retired script),
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
- **CloudRift (second provider, docs/ops/cloudrift.md §7).** With
  `CLOUDRIFT_API_KEY` set, the same cron lists CloudRift rentals. Their rows
  in `pods` have `provider = 'cloudrift'` and owner `cloudrift:<fv-kind>`
  (ours, tag `fv-owner:fastvideo-rs`) or `external:cloudrift`, and costs go
  to `cost_daily`. The cron terminates our rentals past their
  `fv-deadline:<unix>` tag or below `CLOUDRIFT_BALANCE_FLOOR`. The CloudRift
  balance is in the overview's `cloudrift` block, separate from Runpod's.
  CloudRift answers money in cents (the balance and `cost_per_hour`, seen
  live 2026-10-06); the client converts to dollars (`CLOUDRIFT_COST_UNIT`
  defaults to `cents`). Only RTX PRO 6000 and RTX 5090 (`rtxpro6000-*`,
  `rtx59-*`) are allowed: the price route refuses other GPUs (400), and the
  cron terminates our live rental on any other type (`cloudrift_type`).

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
| `build_pod` | the shared build pod (`external:build-pod`) up ≥ `build_pod_max_h` (9 h), or idle (no jobs, per its public `/healthz`) `build_pod_idle_grace_min` (15) past its own idle stop | on | `build_pod_backstop` (**on**): stop it, terminate if the stop is refused (`src/buildpod.ts`, docs/dev/build-pod.md) |

Everything is notify-only except the two rules the scripts already had:
the deadline backstop and the balance floor, plus the build pod backstop
the owner asked for on 2026-10-02. Otherwise auto-actions never touch
external pods. Pod health in the pod views comes from the edge's families
view, or a direct worker's `/health` (ready, loading or down).

The dashboard lists the open alerts with a **Resolve** button
(`POST /api/alerts/<id>/resolve`, audited); an alert whose condition still
holds opens again at the next collector pass. Its **Build pod** card is
read only: the pod's own `/healthz` timers (uptime, idle, time to its idle
and cap stops), its last self-stop attempt, the running jobs (no command
lines), and how far the controller's backstop is.

## 8a. Build pods

fv-control manages the CPU build pods (`src/buildpods.ts`; design and the
numbers: docs/dev/build-pods-fv-control.md). One or more pods, in any Runpod
datacenter, placed by CPU stock and price (`GET /api/build-pods/plan`); each
runs `scripts/dev/build-pod-server.py` and the base image pinned in
`scripts/dev/build-pod.sh` **at `main`** (fetched from GitHub at create time),
so a branch cannot put its server on a shared pod. The policy (settings key
`build_pods`, `GET/PUT /api/build-pods/policy`, the dashboard's **Build pods**
card) is off by default.

- **`up`** (`POST /api/build-pods/up`; `fv-control.sh build-pod up`;
  `build-pod.sh up`): reuse the least busy running pod, else start a stopped
  current one, else create one (deleting stopped *outdated* ones first; a
  running outdated pod is never touched). A D1 lease serialises concurrent
  calls. Refused below floor + `balance_margin`, above `daily_usd_max` of
  build-pod spend today, at `max_pods`. Admin callers get the pod token
  (sealed in D1, audited on every read).
- **stop / delete** refuse while the pod runs jobs or its GitHub runner is
  busy, unless `force`; both remove the runner from GitHub first.
- **Cron:** states from Runpod; per-pod backstop (its own idle stop / cap +
  `backstop_margin_min`; alert `build_pod`); stop on the balance floor;
  registers each ready shared pod as a runner (labels `fv-build`,
  `fv-build-<region>`; a 1-h registration token to the pod's `POST
  /v1/runner`, never the PAT) and removes the runners of stopped pods and
  offline `fv-build-<pod>` runners; wakes a pod for queued jobs that need
  `fv-build` (`wake_on_queue`) or for runs of `wake_workflows`.
- **Costs:** owner `build-pod:<name>` in `pods`, `cost_daily` and every spend
  view; alert `build_pod_spend` above `daily_usd_max`; runner or wake errors
  open `build_pod_runner`.
- **CI** (`POST /api/ci/build-runner`, a token with scope `ci`): `pod` when
  an idle `fv-build` runner is online; `wait` (+ `retry_after_s`) while a pod
  it woke comes up; `github` with the reason otherwise.
- The legacy pod named `fv-build` (owner `external:build-pod`, created by the
  old script) keeps the `build_pod_*` policies of §8 until it is gone.

| route | |
|---|---|
| `GET /api/build-pods` | every managed pod: state, placement, $/hr, spend today, `/healthz` timers, runner, outdated; the policy |
| `GET/PUT /api/build-pods/policy`, `GET /api/build-pods/plan?region=` | policy; placement candidates and the server/image at `main` |
| `POST /api/build-pods/up {region?}`, `POST /api/build-pods {region?, purpose: test, server_ref?}` | a pod (+ its token); a separate test pod |
| `GET /api/build-pods/<id>`, `GET …/token` (admin), `POST …/start`, `POST …/stop {force?}`, `DELETE …?force=1`, `POST …/runner` | one pod |
| `POST /api/ci/build-runner {label?, region?, wake?}` | the CI builder choice |

## 9. Observability

This follows the owner's "lean" guidance.

- **Metrics:** fv-serve's Prometheus `/metrics` remains the only metrics
  source. There is no OpenTelemetry SDK, collector, node_exporter,
  dcgm-exporter or sidecar. GPU, CPU and memory come from the Runpod
  runtime metrics API.
- **Collection is a pull.** The cron reads Runpod's GraphQL and the
  edge's families view. (The gateway's `/metrics` pool series went with the
  gateway.)
- **Storage:**
  - High-frequency series go to **Workers Analytics Engine**, dataset
    `fv_control_metrics`:
    - pod rows: blobs `pod, pod_id, owner, cluster, name, gpu, status`;
      doubles `$/hr, gpu%, gpu mem%, cpu%, mem%, uptime, jobs running,
      jobs queued, idle`;
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
| `POST /api/clusters/<id>/{price,start,stop,extend,scale,roll,restart,cancel}` | operations (202 + operation id) |
| `GET /api/clusters/<id>/ops`, `/api/ops/<id>` | operation logs |
| `GET /api/clusters/<id>/env` | effective env per pod (masked), `needs_restart` |
| `GET /api/clusters/<id>/front` (alias `…/gateway`), `POST …/admin-token`, `POST …/mint-key`, `GET …/keys`, `DELETE …/keys/<key_id>` | the cluster's front: the edge's status and families view, or each direct worker's URL and health; reveal the admin token (the edge's, or a direct cluster's with the worker URLs; audited); mint a user API key; list keys; revoke one (at the edge, or on every direct worker; audited) |
| `GET /api/buildpod` | the shared build pod: its `/healthz` timers (up, idle, idle / cap stop), last self-stop attempt, running jobs, and the backstop's distance (read only) |
| `GET /api/env/account`, `GET /api/env/<scope>/<id>`, `PUT/DELETE /api/env/<scope>/<id>/<KEY>` (`PUT /api/env/account/<KEY>`; scope `pool`: id `<cluster>:<pool>`) | env layers |
| `GET /api/alerts`, `POST /api/alerts/<id>/resolve`, `GET/PUT /api/policies` | alerts and policies |
| `GET /api/logs?pod=&q=&level=&since=`, `/api/logs/download?pod=&day=`, `/api/logs/tail?pod=` (WebSocket) | logs |
| `GET /api/releases`, `/api/images/tags?filter=`, `/api/github/ci`, `POST /api/github/release` | channels, drift, registry, GHCR tags, CI on main, promote/rollback dispatch |
| `GET/POST /api/tokens`, `DELETE /api/tokens/<id>`, `GET /api/audit` | tokens and audit |
| `GET /api/schemas`, `/api/schemas/<name>`, `/api/schemas/dynamic?cluster=` | JSON Schemas and live values (section 13) |
| `GET/PUT /api/docs/<kind>/<id>`, `POST …/validate`, `POST …/plan`, `GET …/history`, `POST …/restore` | editable documents with versions (section 13) |
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
- the boot commands and generated configs;
- legacy spec migration (a stored `gateway` block);
- env and payload shapes;
- spec validation;
- env resolution and masking;
- AE SQL injection safety;
- attribution, alert lifecycle and rate limits;
- secret scrubbing;
- release dispatch validation;
- drift.

**Integration** (`test/integration/run.mjs`, 24 steps) runs the Worker
under `wrangler dev` (workerd with local D1, R2 and DO) against
`test/harness.mjs`. The harness mocks Runpod REST, GraphQL and hapi, GHCR,
GitHub and the Cloudflare API, and simulates worker pods and the edge
Worker (families view, keys, admin token). The steps cover:

- auth, CSRF and Origin, forged cookies, token scopes, and the login rate
  limit;
- the floor refusing a start;
- a full start of an edge cluster with every payload and env checked;
- the collector, attribution, costs and the idle alert;
- env layers (account, cluster, pool, pod) and rolling restarts;
- ingest, search, level filter, live tail, download, auth and scrubbing;
- scale up and down, with a drain;
- a roll, checking digests;
- extend, the edge's admin token, and keys minted, listed and revoked at
  the edge;
- a direct cluster: the controller's admin token on every worker
  (also after a scale-up), the workers view, and minting, listing and
  revoking a key on the workers;
- GitHub dispatch and CI;
- the deadline backstop and the balance-floor stop;
- a second edge cluster: register, scale, roll, one edge cluster at a time;
- the audit log, and that no secret appears in any response.

**UI:** login, the dashboard with its charts and tooltip, every page, a
secret env var set through the UI and never rendered, dark mode, and no
horizontal scroll at 390 px. Screenshots go to `control/test-results/`.

**fv-serve:** the `log_ship` unit tests (config from env, JSON lines with
span fields, dropping when full), run on the build pod.

## 12. Live test (2026-09-29, staging, real Runpod account)

> Historical: this ran with the gateway pod, which has since been retired.
> The edge lifecycle's live test is in
> [edge-control-plane.md](../serve/edge-control-plane.md) §9, "Stage 3
> results".

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

## 13. Editors (specs and JSON)

Every JSON document the controller edits has a schema, a smart editor, a
generated form, and the same safety path: validate, review the diff (and
the plan for a cluster spec), save against the loaded version, then
history and restore.

### Schemas

`control/src/schemas.ts` defines the schemas in zod. They are the one
source for:

- **server-side validation** on every write path: the document API, the
  older `PUT /api/clusters/<id>/spec`, `PUT /api/policies`, `POST /api/tokens`
  and `POST /api/github/release`;
- **JSON Schema** (draft 2020-12, `z.toJSONSchema`) served at
  `GET /api/schemas` and `GET /api/schemas/<name>`;
- **compile-time checks** against the TypeScript types. The `_check*`
  assignments at the bottom of the file break `tsc` if `ClusterSpec`,
  `PoolSpec` or `Policies` drift from their schemas.

| schema | document |
|---|---|
| `cluster-spec` (with `pool`) | a cluster definition: pools, GPU types, regions and volumes, image channel / sha / ref, counts, backstop (`cap_s`), floors, auto-stop, log shipping |
| `env` | env vars at one level: `KEY → {value, secret, set?}` |
| `policies` (with `attribution`) | alert thresholds, auto-actions, attribution rules |
| `token-create` | API token name, scope (`read` / `admin` / `ci`: only `/api/ci/*`) and expiry |
| `release-dispatch` | `release.yml` promote / rollback inputs |

Every field carries a `description`, which the editor shows on hover and
the form shows as help. Two custom keywords drive the editor:

- `x-dynamic: <source>` marks a field that takes live values (GPU types,
  channels, shas, variants, fake models, env keys).
- `x-secret` marks the write-only `set` of an env var.

The browser validates as you type with a small validator
(`control/ui/schema.ts`). The server then validates with zod, debounced,
which adds the checks JSON Schema cannot express: "exactly one of
channel, sha, ref"; "config or config_toml"; "models or fake_models"; a
duplicate or reserved pool id; the fixed cluster name; secret rules; and
the controller's reserved env keys. A unit test runs both validators on
the same valid and invalid documents and checks that they agree.

**Live values** come from `GET /api/schemas/dynamic?cluster=`
(`control/src/dynamic.ts`):

- Runpod GPU types with secure and community $/hr, memory and stock,
  cached 5 min;
- regions with their DCs and network volumes;
- CPU flavors with $/vCPU;
- release channels with their head sha and per-variant digests (from
  fv-jobs), plus recent shas;
- image variants and fake models;
- the cluster's pools;
- env keys in use, and the reserved keys.

### Documents API

| route | |
|---|---|
| `GET /api/docs/<kind>/<id>` | `{doc, version, schema}` |
| `POST /api/docs/<kind>/<id>/validate {doc}` | `{ok, issues: [{path, message}]}` |
| `POST /api/docs/cluster-spec/<id>/plan {doc}` | what the change means for the running cluster, and the projection |
| `PUT /api/docs/<kind>/<id> {doc, version}` | save; **409** `{current_version}` when someone saved in between |
| `GET /api/docs/<kind>/<id>/history` | the saves, from the audit log (`doc.save` / `doc.restore`, before and after) |
| `POST /api/docs/<kind>/<id>/restore {audit_id, which, version}` | save a snapshot again, as a new version |

The kinds and ids are:

- `cluster-spec/<cluster id or name>`
- `policies/default`
- `attribution/default` (a slice of the policies)
- `env/account`, `env/cluster:<id>`, `env/pool:<cluster id or name>:<pool id>` and `env/pod:<pod id>`

Versions live in D1 `doc_versions` (migration `0002`). The check is one
conditional `UPDATE … WHERE version = ?`. Every other write path bumps
the same version: per-key env routes, the older spec and policies routes,
and a `scale` operation (which changes a pool's count). So an editor
opened before any of these is refused on save instead of overwriting.

**The plan** of a cluster spec compares the proposed spec with the
cluster's state:

- workers to create or drain per pool (count changes and new or removed
  pools);
- a roll when the image source or a pool's variant or image changes;
- each running pod whose env would differ (the same env hash as the
  "needs restart" view).

It also shows $/hr now and after, and the price projection against the
floor; for a running cluster, that projection takes the account's burn
without the cluster's own pods. Saving changes only the definition.
Scale, Roll and the env restart apply it, and the plan names the one
needed.

**Secrets:**

- An env document returns a secret as `{value: null, secret: true}`,
  never its value.
- A new value goes only in the write-only `set`. In the form this is a
  password field; the JSON tab shows a placeholder and refuses a typed
  value.
- History snapshots store secrets masked. Restoring a snapshot keeps
  secrets that still exist. It cannot bring back the value of one that
  was deleted: that key is reported as `skipped`, and the owner sets it
  again.

### The editor (`control/ui/`, built to `public/editor.js`)

The editor is CodeMirror 6: `@codemirror/{state,view,commands,language,lang-json,lint,autocomplete,search}`,
with no basic-setup bundle. It adds:

- **Inline diagnostics:** local schema errors, plus the server's issues
  placed at their JSON path; JSON syntax errors at their position.
- **Hover docs:** type, description, range or pattern, and for a live
  value its price and stock or its sha. An unknown GPU type or channel is
  flagged.
- **Completion:**
  - missing keys, required ones first, inserted as `"key": `;
  - enum values, `true` / `false` / `null`;
  - live values with details (e.g. `"NVIDIA H200"  $3.59/hr · Medium stock`);
  - env keys in use.
- Folding, search (Ctrl-F), bracket matching, history, format
  (pretty-print).
- Light and dark from the page's CSS tokens. Line wrapping and a capped
  height keep it usable on a phone.

**The form** (`control/ui/form.ts`) is generated from the same schema and
kept in sync with the JSON tab. A change on either side updates the other;
the form refuses to open while the JSON does not parse. It renders:

- objects as a grid of fields with help text;
- enums as selects, booleans as toggles, numbers with their bounds;
- live strings with a datalist;
- live and enum arrays (GPU types, regions, CPU flavors) as ordered chip
  pickers, where the order is the placement order;
- arrays of objects (pools, attribution rules) as a table with the main
  columns (id, variant, count, compute), each item's full form behind
  "more", and add / remove;
- an env set as a key / value / secret table: secret inputs are
  write-only password fields, and new keys complete from the keys in use.

**The panel** (`control/ui/panel.ts`) adds:

- Form / JSON tabs, a version badge and an unsaved-changes marker;
- Format, Validate and Revert;
- Review: a line diff of current → proposed and, for a spec, the plan;
  Save is refused if anything is invalid;
- on a 409, "Compare with the saved version" and "Reload";
- History: each save with its diff, "Restore this version" and "Restore
  the version before it".

**Where it is used:**

- the cluster page: spec editor, outside the page's 10 s refresh;
- the clusters page: the define editor, with schema, completion and live
  values;
- the Env page: account, cluster and pod documents;
- Settings: policies and attribution.

**Read-only JSON trees** (`renderTree`) are collapsible, render lazily,
and have search plus copy-path (`$.pools[0].count`) and copy-value. They
show:

- pod snapshots and their 24 h metric samples (pod page);
- each audit entry's before / after (Settings);
- release heads and history (Releases).

The **Env page** shows the effective env per pod with the source level of
each key as a badge:

- a value that overrides a lower level is highlighted;
- a conflict (two of the owner's levels setting different values) is
  marked;
- "needs restart" and "Apply with a rolling restart" work as before, and
  the view refreshes after a save.

**Bundle:** the dashboard (`app.js`, `app.css`) stays dependency-free and
small. `editor.js` (425 KiB minified, 139 KiB gzip; mostly
`@codemirror/view`) loads on first use, only on pages that edit or view
JSON. Wrangler builds it (`[build] command = "node build-ui.mjs"`) before
`dev` and `deploy`; it is not committed.

**CSP:** CodeMirror injects its styles with a `<style>` element, so
`style-src` is `'self' 'unsafe-inline'`. Scripts stay `'self'` only, and
every API value reaches the DOM through `textContent`.

### Tests

- **Unit** (`test/unit/schemas.test.ts`):
  - the JSON Schemas (required keys, descriptions, `x-dynamic`,
    `x-secret`);
  - zod validation with paths;
  - the browser validator against zod;
  - `schemaAt` / `unwrap`;
  - editor path ↔ position mapping on a real CodeMirror state;
  - the line diff;
  - documents: versions and 409, invalid saves, the fixed name, history
    and restore, the plan, env secrets (write-only, never read back or put
    in history, restore skips lost values, reserved keys), and policies
    and attribution sharing one setting.
- **Integration**, the step "schemas, dynamic values and the document
  API":
  - schemas and live values;
  - validation issues with paths, including a server-only refinement;
  - 400 on save, the plan, save → v+1, a stale save → 409;
  - version required, history, restore;
  - an env secret never in any response;
  - per-key routes bumping the version;
  - token scope checked by the schema.
- **UI smoke** (headless Chromium), in order:
  1. the spec editor's JSON tab: an invalid `cap_s` shows an inline lint
     error and an issue;
  2. hover docs, and key completion (`log_level`);
  3. fix the value, change a pool count in the form;
  4. Review shows the diff and the plan (create a worker, within the
     floor); Save → v1; a stale PUT is refused;
  5. History → restore → v2, checked by reading the document back;
  6. the env form with a write-only secret: masked in the JSON tab,
     saved, never in the page, and the effective view flags the restart;
  7. JSON tree search.
