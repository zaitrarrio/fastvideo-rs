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
  "region": "eu",                    // or "dc": "EUR-IS-1"; the volume is EU only (CLAUDE.md). With "weights_source": "r2" (no volume) any Runpod "dc", e.g. "US-TX-3"
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

## GMI Cloud and NVIDIA Brev

`provider: gmi | brev` (`--provider` on the CLI) launches the same pod on a
GMI Cloud container or a Brev VM (docs/serve/deploy-gmi-brev.md): `provider_gpu`
is the provider's product / instance type, there is no weights volume
(`weights_source: none` runs the fake engine; `hub` downloads at boot after
the owner's approval), the pod reports its HTTPS tunnel URL to
`/ingest/v1/endpoint` (a named tunnel fv-control made on the owner's zone,
URL known at create, or a quick tunnel without those settings;
deploy-gmi-brev.md §6.4), and a monthly budget (`GMI_BUDGET_USD`,
`BREV_BUDGET_USD`) replaces the balance floor. Both are off until the owner
sets their secrets; the form and `fv-control.sh providers` say why.

**Brev keep-on-stop** (docs/serve/deploy-gmi-brev.md §3.4). Stopping a
Brev pod of a stoppable instance type, whose Hub weights completed, stops
the VM instead of deleting it (owner rule 2026-10-11: "stopping a brev vm
should NOT delete it. brev VM's should be re-used or manually deleted"):
it is *parked* with its weights under `~/workspace/weights`, or *held*
with a note when its boot never became ready (a failed boot: partial
weights stay and the next boot resumes them). This covers Stop, the
deadline, the idle stop and a budget stop. `fv-control.sh pod delete
<name>` of a pod whose VM is live deletes that VM (your explicit delete);
the delete of a definition with no pod leaves any parked VM alone. The
next launch of the same instance type reuses a parked or held instance
before creating one: a *warm* boot when it holds the trees, else the
trees download onto it. A stop Brev refuses leaves the VM running with the
alert `brev_stop_failed` (its $/hr); fv-control never deletes it.

Non-stoppable types (e.g. `massedcompute_*`) are launched like any other
(owner decision 2026-10-13). Brev cannot stop them: a stop answers HTTP
500 "does not support stop" and the VM keeps running. fv-control then
releases the pod (state `running_unstoppable`), raises one critical alert
`brev_unstoppable` with its $/hr and $/day, tries the stop again hourly,
and bills it in the ledger and the Brev budget until you delete it
(`fv-control.sh brev delete-parked <id>`, Delete on the Pods page, or
`pod delete <name>` instead of a stop). The launch's budget check counts
such a pod 24 h past its deadline (`BREV_NOSTOP_MARGIN_H`).

A parked instance bills storage only. That cost counts in the Brev
budget and the cost views (owner `brev:parked`); the policies
`brev_park_max` (2) and `brev_park_max_days` (7) only raise the alert
`brev_parked_over_limit`. A restart that fails (no capacity) or does not
come up is *held* (reused by a later launch after 15 minutes, or at once
after Unpark), and another instance or a fresh pod is used.

The launch form lists the stoppable instance types first, then the others
with "WARNING: can't be stopped: a stop leaves it running at $X/hr", and shows "warm:
restarts parked …" or "cold: downloads … (~N GB): boot ~M min" at the
instance-type field. The Pods page lists parked and held instances with
their weights, disk, $/day and reuse status, with Delete (yours only) and
Unpark for a held one. The Stop confirmation of a Brev pod says its VM is
stopped and kept, with its storage $/day.

```bash
fv-control.sh brev parked                    # policy line; parked / held instances: weights, disk, $/day, reuse
fv-control.sh brev unpark <workspace id>     # a held one reusable at once
fv-control.sh brev delete-parked <workspace id>
```

```bash
fv-control.sh providers
fv-control.sh pod launch smoke-gmi --provider gmi --gpu container.h200.x1 --fake --deadline-min 20 --wait
fv-control.sh pod launch smoke-brev --provider brev --gpu <instanceType> --fake --deadline-min 20 --wait
```

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
