# fv-control: Runpod serverless endpoints

Status: 2026-10-06, branch `feat/fvc-serverless`. Code: `control/src/serverless/`
(`spec.ts`, `payloads.ts`, `runpod-sls.ts`, `ops.ts`, `routes.ts`), migration
`control/migrations/0006_serverless.sql`, the **Serverless** page
(`control/public/serverless.js`), the CLI `scripts/serve/fv-control.sh endpoint …`.

fv-control creates, updates, scales, invokes and deletes Runpod serverless
endpoints of fv-serve: one image variant behind Runpod's **queue** (`/run`,
`/runsync`; the worker runs `FV_SERVE_MODE=runpod-queue` and takes the native
job envelope, `kind: info | http | stream`) or its **load balancer** (fv-serve's
HTTP on port 8000, `/ping`; GPU only). It is the API form of
`scripts/serve/runpod-endpoint.sh` (docs/serve/e2e/serverless.md), with
ownership, money guards, cost and health on top.

## 1. The spec

A JSON document, validated by one zod schema (`EndpointSpecZ`, served as
`GET /api/schemas/serverless-endpoint`, the same framework as cluster specs).
Missing fields take the variant's defaults (`GET /api/serverless/defaults?variant=`).

| field | default | |
|---|---|---|
| `name` | | `[a-z][a-z0-9-]{0,30}`; the Runpod endpoint is `fvc-<name>` |
| `mode` | `queue` | `queue` or `lb` (GPU, `scaler_type: REQUEST_COUNT`) |
| `image` | `{channel: stable}` | one of `channel`, `sha`, `ref`; resolved to a digest like cluster images (`<variant>-<channel>`, `:stable` → `:latest` before the first promotion) |
| `variant`, `compute` | `cpu` → `CPU` | `cpu` (fake engine) runs on CPU workers; the CUDA variants on GPU |
| `config` / `config_toml` | the variant's baked config | a config file in the image (`FV_CONFIG`) or inline (`FV_WORKER_TOML_B64`) |
| `env` | | plain extra env; keys fv-control sets (serve mode, image identity, the `{{ RUNPOD_SECRET_fv_* }}` references, …) are refused |
| `gpu_types`, `gpu_count`, `allowed_cuda` | RTX PRO 6000 Server, 1, `["13.0"]` | GPU types in priority order |
| `cpu_flavors`, `vcpu` | `cpu3c, cpu5c`, 2 | CPU workers |
| `network_volume`, `data_centers` | GPU: `jg48s6o1w0` in `EUR-IS-1`; CPU: none | only the EU weights volume is accepted (CLAUDE.md); a volume pins its data center |
| `workers_min`, `workers_max` | 0, 1 | 0–4, 0–8 |
| `idle_timeout_s`, `flashboot`, `execution_timeout_s` | 5, false, 1800 | |
| `scaler_type`, `scaler_value` | `QUEUE_DELAY`, 4 | |
| `container_disk_gb` | 20 | |
| `deadline_min`, `deadline_action` | 120, `delete` | the backstop: `delete` (endpoint + template) or `scale0` (workers 0/0) that long after create (or after the last extend); `null` for none |

An update merges a partial spec over the stored one. Image, env, config and
disk patch the **template** (Runpod rolls the workers); scaling, timeouts,
GPU types and data centers patch the **endpoint**; `mode`, `compute`,
`network_volume` and a CPU endpoint's flavors / vCPUs need a new endpoint (409).

## 2. Runpod API (checked live 2026-10-06)

| call | used for |
|---|---|
| REST v1 `POST /templates` + `POST /endpoints` | GPU queue endpoints (`runpod-endpoint.sh`'s path) |
| REST v2 `POST https://api.runpod.io/v2/serverless` | CPU queue endpoints (`type: QUEUE`, `cpu: [{id, vcpuCount}]`) and load balancers (`type: LOAD_BALANCER`, `gpu.pools` from `GET /v2/catalog/gpus`). v2 makes the endpoint's template and Runpod deletes it with the endpoint |
| REST v1 `PATCH /endpoints/<id>` | every update and scale (also of v2-made endpoints), and the reconcile after create |
| REST v1 `PATCH /templates/<id>` | image / env / config updates (also of v2-made templates) |
| REST v1 `DELETE /endpoints/<id>`, `DELETE /templates/<id>` | delete |
| REST v1 `GET /endpoints/<id>?includeWorkers=true` | the endpoint and its workers (never its template's env, which Runpod returns in clear: views pick named fields) |
| REST v1 `GET /billing/endpoints?bucketSize=day&grouping=endpointId` | spend per endpoint and day: `amount` ($), `timeBilledMs` |
| `https://api.runpod.ai/v2/<id>/health`, `/runsync`, `/run`, `/status/<job>` | health (jobs, workers idle/running/initializing/throttled/unhealthy), test invokes |
| `https://<id>.api.runpod.ai/<path>` | load-balancer test invokes |
| GraphQL `myself { clientBalance endpoints { id name type pods { id desiredStatus costPerHr } } }` | the balance and live workers with $/hr, one call per tick |
| `hapi.runpod.net/v1/pod/<worker>/logs` | a worker's container and system log tail (the pods' log endpoint works for serverless workers) |

Runpod quirks found live, and handled:

- **REST v1 ignores `computeType: CPU`**: the endpoint gets GPU workers (the
  first live run's "CPU" worker was an RTX 2000 Ada). CPU endpoints go through v2.
- **REST v1 create drops `flashboot: false` and `workersMax: 0`** (FlashBoot
  on, max 3). Every create is followed by a `PATCH` with the spec's scaling
  fields.
- **`computeType` is not an update field**: `PATCH` refuses it ("Extra input keys").
- A v2-made template is deleted with its endpoint; its `DELETE` then answers
  400 "template not found", which counts as gone.
- Serverless workers are **not** in `myself { pods }`, so the per-minute pod
  collector never double-counts them; their cost comes only from billing.

## 3. Ownership and money

- **Only endpoints fv-control made.** Every one is a row in
  `serverless_endpoints` (migration 0006; the row stays after delete, for the
  ledger). Routes resolve `:id` as the row id (`se_…`), the Runpod endpoint id or
  the live name, and answer 404 for anything else. Before acting on Runpod
  they also check its name is still `fvc-…` (403 otherwise). `GET
  /api/serverless?external=1` lists the account's other endpoints by id and name only.
- **Balance floor.** Create, scale-up (a higher `workers_max` or `workers_min`)
  and invoke need balance ≥ `BALANCE_FLOOR` + `serverless.balance_margin`
  (default $2): 402 otherwise. Below the floor the tick scales every endpoint
  to 0/0 (when the `stop_on_floor` policy and `serverless.scale0_on_floor` are
  on; actor `policy:balance_floor`).
- **Limits** (settings key `serverless`, `GET/PUT /api/serverless/policy`):
  `max_endpoints` 4 live, `max_workers` 8 summed `workers_max`.
- **Backstop.** `deadline_min` after create, the tick deletes or scales to 0
  (actor `policy:serverless_deadline`; alert `serverless`). `extend` moves it.
- **Create is all or nothing**: a failure deletes what was made (endpoint,
  then template) and leaves the row `failed`.
- **Delete** scales to 0/0, deletes the endpoint, then the template; a refused
  delete (a worker still running) leaves the row `deleting` and the tick
  retries every minute.
- **Audit:** `serverless.create`, `.update`, `.scale`, `.extend`, `.invoke`,
  `.delete`, `.deleted`, `.policy`, with before/after; failures with `ok = 0`.

## 4. Cost, health, logs (the tick)

`serverlessTick` runs from the cron next to the collector (and by hand:
`POST /api/serverless/tick`). With no endpoint it makes no Runpod call.

- **Health**: `/health` per endpoint, plus GraphQL's live workers and their
  summed $/hr (`workers`, `live_dph`).
- **Cost**: every 10 min, Runpod's billing per endpoint and UTC day goes into
  `cost_daily` as `pod_id = sls:<endpoint id>`, `owner = serverless:<name>`
  (an upsert: billing is the truth, re-read for two days). So serverless
  spend shows in the Costs page (by owner, by day), the dashboard's cost
  today and the daily-spend alert, next to pods and clusters. Runpod's billing lags
  (minutes to hours); `live_dph` is the running rate meanwhile.
- **Job stats and cold start**: every test invoke is a `serverless_jobs` row
  (Runpod `delayTime` and `executionTime`, fv-control's end-to-end time, the
  worker, output truncated and scrubbed). It is `cold` when `/health` showed
  no worker up at submit; the cold start is the median `delayTime` of cold
  jobs.
- **Logs**: each live worker's Runpod log tail (≤ 200 container + 50 system
  lines, ANSI stripped) replaces that worker's rows in `log_lines`, under
  `cluster_id = serverless:<row id>` and `pod_id = <worker id>`. These are the
  same table and fields cluster logs use, so the log search (`/api/logs?pod=<worker>`)
  and a log explorer over (source, pod_id) see them. `GET /api/serverless/<id>/logs?worker=`
  fetches one on demand. Retention is the log store's 24 h.

## 5. API, CLI, UI

| route | |
|---|---|
| `GET /api/serverless[?all=1&external=1]` | endpoints (live and deleted in the last day; `all`: every one), cost today, the policy |
| `POST /api/serverless {spec}` | create (201) |
| `POST /api/serverless/validate {spec}` | validate; the spec with defaults filled |
| `GET /api/serverless/defaults?name=&variant=`, `GET/PUT /api/serverless/policy` | |
| `GET /api/serverless/<id>` | the row, Runpod's view (workers), `/health`, recent jobs and their stats, cost per day, audit |
| `PUT /api/serverless/<id> {spec}` | update (merged over the stored spec) |
| `POST /api/serverless/<id>/scale {workers_min?, workers_max?}`, `POST …/extend {minutes}` | |
| `DELETE /api/serverless/<id>` | 200 deleted, 202 deleting (the tick retries) |
| `POST /api/serverless/<id>/invoke {input?, sync?}` (queue) / `{method, path, body?}` (lb) | a test request; `GET …/jobs`, `GET …/jobs/<job>` polls one |
| `GET /api/serverless/<id>/logs?worker=` | a worker's log tail (also stored) |
| `POST /api/serverless/tick` | the tick now, billing included |

```bash
fv-control.sh endpoint create fake-test                    # cpu variant, CPU workers, 0..1, delete after 2 h
fv-control.sh endpoint create spec.json                     # or a full spec
fv-control.sh endpoint list | show fake-test
fv-control.sh endpoint invoke fake-test                     # {"kind":"info"}; polls when /runsync returns IN_QUEUE
fv-control.sh endpoint invoke fake-test '{"kind":"http","method":"POST","path":"/fv/v1/jobs","body":{"model":"fake-wan","prompt":"a fox","seed":1},"wait":true}'
fv-control.sh endpoint scale fake-test 0 | extend fake-test 60 | logs fake-test | delete fake-test
```

The **Serverless** page (`#/serverless`) lists the endpoints (status, workers,
queue, $/hr, cost today and billed, backstop). It creates one from a form or the spec
JSON (validate first), and per endpoint shows health, cold start and warm
queue wait, scale / scale to 0 / backstop +30 min / delete, a test invoke with
presets (info, capabilities, a fake job; LB: ping, capabilities), workers and
their logs, the spec editor, recent invokes, cost per day and audit.

## 6. The edge

The edge control plane cannot route a model family to a serverless endpoint.
Its design dropped serverless pools on purpose
([edge-control-plane.md](../serve/edge-control-plane.md) §3.10, §10 Q1):

- a front must be a reachable worker holding a WebSocket to its family
  Durable Object;
- Runpod's queue cannot hold the family DO's jobs.

Wiring it needs either:

- a DO → Runpod `/run` path (a "serverless front" in the family DO that
  forwards a job envelope and polls `/status`); or
- a load-balancer endpoint whose workers dial the edge as fronts.

LB workers run `FV_SERVE_MODE=http`, so they could take the edge env
(`FV_DISPATCH_FRONT`, the internal token, `FV_DISPATCH_ENDPOINT=https://<id>.api.runpod.ai`).
But Runpod's LB answers only while a worker is up, and its gateway needs the
Runpod API key, which the edge would have to send. Neither path is built.
Serverless endpoints are standalone today: clients call Runpod's URLs with the
Runpod key, as with `runpod-endpoint.sh`.

## 7. Live test (2026-10-06)

Driver: `control/test/live/serverless-live.ts` (`npx vite-node …`). It runs
the same `ops.ts` / `payloads.ts` against the real Runpod API, with an
in-memory D1 (staging's D1 untouched). It also starts a detached 45-min backstop.

**Run 1 (the `fvc-sls-live` endpoint, `yjy0sfsbk1o9l8`, REST v1 path).**

- Create took 2.3 s.
- Cold info job: queue wait 14.3 s, exec 0.18 s.
- Warm: 0.15 s wait.
- Fake job (`fake-wan`, `wait: true`): 2.7 s exec, `succeeded`.
- The worker was an RTX 2000 Ada ($0.24/hr): the REST v1 `computeType` quirk above.
- Scale to 0 failed on `computeType` in the PATCH (fixed).
- Delete worked; endpoint and template verified absent.

**Run 2 (`ebm69ru5aqjwyn`, CPU via v2; cpu3c/cpu5c in EU-CZ-1, $0.072/hr; image `cpu-latest` @ `sha256:a4471ad5…`, since `cpu-stable` does not exist yet).**

- Create took 2.6 s.
- Cold info job:
  - `/runsync` returned IN_QUEUE after 90 s, then the poll path ran;
  - queue wait 97.3 s: Runpod started three workers, and logged about 86 s between "worker is ready" and the container start;
  - fv-serve was ready 7.1 s after its process start.
- Warm info job: 0.14 s wait, 0.13 s exec.
- Fake job: `succeeded`, 2.7 s exec, 3.0 s end to end.
- The tick stored 64 log lines, including the worker's `FV-SERVE READY`.
- Scale to 0: 0.9 s.
- Delete:
  - the endpoint was gone at once;
  - the row stayed `deleting` on the v2 template's "not found" answer, which is now handled.
- Endpoint and template verified absent.

Worker time in both runs was under 3 min (≈ $0.01 at the observed rates).
Between runs, four throw-away probes (`fvc-probe*`, `workersMax` ≤ 1, no jobs,
no workers) mapped the API quirks above. All were deleted, and their templates
verified absent.
