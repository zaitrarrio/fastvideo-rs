# Standalone pods

Status: built 2026-10-06 (branch `feat/standalone-pods`), not deployed yet.
Code: `control/src/standalone.ts`, the routes in `control/src/index.ts`,
`control/public/standalone.js`, `scripts/serve/fv-control.sh pod …`.

A standalone pod is one fv-serve pod launched on its own, not as part of a
cluster: one GPU (or CPU) worker from a variant and an image, in a region,
with a deadline, which the owner can stop, start and delete.

## Design: a one-pool direct cluster

A standalone pod is stored as a cluster row with
`clusters.source = 'standalone'`. Its spec has one pool, `pod`, with
`count: 1`, and `control_plane: "direct"`. Everything a cluster pod goes
through, it goes through by the same code, not a copy:

| concern | how a standalone pod gets it |
|---|---|
| validation | `normalizeSpec` (the cluster spec schema), after `standaloneSpec` turns the launch request into a spec |
| create | the ClusterOps `up` operation: price check against the floor, image preflight (`cluster/preflight.ts`), placement (`workerPlacements`), payload (`workerCreatePayload`), the `max_gpu_dph` cap, wait until ready with the early verdict (`cluster/upwait.ts`) |
| stop / delete | the `down` operation (delete, verify, retry); delete with a pod passes `delete_definition`, and the definition goes once the pod is gone |
| deadline | the cron's deadline backstop (`collector.ts`), and the pod's own watchdog: `WATCHDOG_WORKER_BOOT` (the edge fronts' watchdog on a direct worker), with `FV_CLUSTER_DEADLINE`, `FV_MIN_BALANCE`, `FV_BACKSTOP_API_KEY` |
| balance floor | the cron's floor rule stops it like any cluster; the price check refuses a start under the floor |
| idle stop | `idle_stop_min` (the spec's `auto_stop_idle_min`): after that many idle minutes the cron stops it (a cluster drains one worker instead; a standalone pod has nothing to drain to) |
| costs | `cost_daily` and every spend view, owner `pod:<name>`, `cluster_id` = its id |
| logs | log shipping with its own ingest token, and the Runpod container / system log captured from boot (`podlogs.ts`) |
| boot timeline | `cluster_pods.boot` (`boottime.ts`), like every controller pod |
| env | the launch's `env` is stored at the definition's cluster level (it survives stop / start); the Env page and `/api/env/cluster/<id>` edit it; reserved keys are refused |
| admin token, keys | the direct-cluster routes (`/api/clusters/<id>/admin-token`, `mint-key`, `keys`) work on its id |
| audit | `standalone.launch`, `.start`, `.stop`, `.extend`, `.delete` (secret env values masked) |

Stop deletes the pod rather than pausing it: a GPU pod with a network volume
cannot be paused on Runpod, and a deleted pod costs nothing. The definition,
its env, logs, timeline and costs stay; start makes a new pod.

A standalone pod is direct only. An edge front needs the one edge cluster
that fronts the edge (docs/serve/edge-control-plane.md §10 Q2), so a
standalone pod is not put behind the edge. Its URL is its pod's proxy URL;
`auth: keys` (default) with the controller's admin token, as for a direct
cluster. `scale` past one is refused (400); `roll` and `restart` work.

## API

Mutations need admin (an admin token, or a session with `x-csrf-token` and
a same-origin `Origin`); a read token can list and read.

| route | |
|---|---|
| `POST /api/standalone` | launch; 201 `{pod, operation}`; `start: false` only defines it |
| `GET /api/standalone` | every standalone pod: definition, pod, status, boot diagnosis and timeline, cost today / total, operation |
| `GET /api/standalone/<id or name>` | one |
| `POST …/start {skip_image_check?}`, `POST …/stop`, `POST …/extend {minutes}` | operations (202) |
| `DELETE …` | 200 when it has no pod; else 202, and it is deleted once the pod is gone |
| `GET /api/pods/<pod>/status` | any controller pod: Runpod view, the controller's record, boot diagnosis, boot phase and timeline, cost, log count |
| `GET /api/pods/<pod>/boot` | the boot timeline |
| `GET /api/pods/<pod>/logs?after_id=&source=runpod\|serve\|control&level=&q=&since=&until=&limit=` | log lines, oldest first; `next_after_id` to follow |

The launch request:

```json
{
  "name": "h3-solo",
  "preset": "h3-turbo",              // or "variant" + "config" / "config_toml" + "models" / "fake_models"
  "channel": "latest",               // or "sha", or "image" (a ref or a digest); default: stable
  "compute": "GPU",                  // or "CPU" (+ "cpu_flavors", "vcpu")
  "gpu_types": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
  "region": "eu",                    // or "dc": "EUR-IS-1"; EU only (CLAUDE.md)
  "volume": true,                    // the region's weights volume at /workspace (GPU default)
  "env": {"FOO": "bar", "HF_TOKEN": {"value": "…", "secret": true}},
  "deadline_min": 60,                // 5-10080
  "idle_stop_min": 30,               // optional
  "max_gpu_dph": 3.6,
  "start": true
}
```

## CLI

```bash
fv-control.sh pod launch h3-solo --preset h3-turbo --channel latest --deadline-min 45 --wait
fv-control.sh pod launch fake1 --variant cpu --cpu cpu3c --config /etc/fv/runpod-fake.toml --json extra.json
fv-control.sh pod list | status <name> | stop <name> | start <name> | extend <name> 30 | delete <name> | wait <name>
fv-control.sh pod logs <name | pod id> [--source runpod|serve] [--follow]
fv-control.sh boot <pod id | name>            # the boot timeline
```

`--secret-env K=@FILE` reads a secret value from a file (not argv).

## UI

The **Standalone** page (`#/standalone`, `public/standalone.js`): the list
with status, boot phase, pod, image, GPU, DC, $/hr, cost today and total,
deadline, and Start / Stop / Extend / Delete; a launch form (preset or custom
variant, image source, compute, GPU type, region, volume, deadline, idle
stop, env). `#/standalone?id=<name>` shows one pod: its pod and definition,
cost, the boot timeline, the last operation's log and the last 40 log lines,
with a link to the Logs page. The page adds its own route to the dashboard's
router; the shared files only gain a nav link and a script tag
(`index.html`), and the pod page one line for the boot timeline card.

## Tests

`control/test/unit/standalone.test.ts` (spec building and refusals, payload
and watchdog boot, the routes with auth and CSRF, delete with and without a
pod, pod status and logs JSON), the integration step "standalone pod" (the
real Durable Object: launch, up, the watchdog boot, owner `pod:<name>`,
status, boot timeline, logs, stop, start, delete), and the UI smoke (launch
from the form, the pod's page, phone width).
