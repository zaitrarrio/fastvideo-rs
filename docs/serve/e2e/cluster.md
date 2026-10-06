# Gateway cluster: a CPU gateway and four GPU pod pools (manual-testing deployment)

> **Historical (2026-10-06).** The gateway, `scripts/serve/runpod-cluster.sh` and
> `configs/serve/gateway-pods.toml` were removed; fv-control runs clusters
> behind the edge ([edge-control-plane.md](../edge-control-plane.md) §9, "Stage 4 as built"). This
> is the record of a past run.

Date: 2026-09-28. `scripts/serve/runpod-cluster.sh up` with image
`ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:33ec255cc547a5d28a938c9873cdb301f45cb263ac6d6dbb457b6dabff48348f`
(tag `sha-6ce80bd`: the newest green serve image when this ran. Main was then
`cb87b2b`, whose diff from `6ce80bd` is CLAUDE.md plus one fetch script, so
the serve code is the same). Raw smoke result:
`artifacts/serve/e2e/cluster/smoke-*.json`. It holds no URLs, keys or tokens.

## Shape

- **Gateway**: a Runpod CPU pod (`cpu3c`, 2 vCPU, $0.06/hr, EUR-IS-1) that
  runs `fv-serve` with `configs/serve/gateway-pods.toml`. That config sets
  `[engine] backend = "remote"` and four `kind = "pod"` pools. The script
  passes the config as base64 in the env, because the image predates the
  file. `FV_AUTH_MODE=none`: no API key is needed. (Since 2026-09-29 the
  script runs the gateway with `keys` and the gateway keeps its own admin
  token, fetched sealed by `runpod-cluster.sh admin-token`: gateway.md §9.)
- **Workers**: one pod per pool, all on the EU volume `jg48s6o1w0`
  (EUR-IS-1). Each is an RTX PRO 6000 Blackwell Server Edition (96 GB) at
  $2.09/hr. Every worker runs with `FV_SERVE_ROLE=worker`, the shared
  internal token, the shared URL-signing key, and `FV_PUBLIC_BASE_URL` set
  to the gateway URL.

  | pool | config in the image | models |
  |---|---|---|
  | h3-turbo | `/etc/fv/runpod.toml` (warmup on, I2V encoder auto) | `fasth3` |
  | h3-max | `/etc/fv/runpod-h3-max.toml` | `sol-h3` |
  | ltx | `/etc/fv/runpod-ltx.toml` | `ltx25-distill-sol` (fal `fastvideo/ltx-turbo`, `lightricks/ltx-2.5`) |
  | wan | `/etc/fv/runpod-wan5b.toml` | `fastwan22-ti2v-5b` (resident), `wan22-ti2v-5b` (swap) |

- **Worker discovery**: the gateway lists the worker pods in
  `FV_POOL_<ID>_URLS`. Self-registration is turned off on each worker
  (`[gateway] register = false`, appended by the start command).
  Registration writes the worker's `server.public_base_url` into
  `gw_workers`, and here that URL is the gateway's. The gateway would then
  probe itself as a worker. Ordering: the script creates the gateway first
  (its URL goes to the workers), then the workers, then PATCHes the
  gateway's env with the worker URLs. The PATCH restarts the gateway's
  container.

## Auth checks (no key)

| route | answer |
|---|---|
| `/console`, `/console/admin` (static pages) | 200 |
| `POST /fv/v1/admin/keys` (key minting), `GET /fv/v1/admin/keys` | **401** without the admin token |
| `/fv/v1/gateway/pools` | **401** without the admin token, 200 with it |
| `/fv/v1/admin/autoscale` | 404 (the autoscaler is off) |
| any worker: `/fv/v1/capabilities`, `/fv/v1/jobs`, `/fv/v1/internal/status` | **401** (internal token required) |
| any worker: `/ping` | 200 (open by design) |

`AuthMode::None` affects only the user-API check (`Auth::authenticate`). The
admin routes check the admin token on their own, whatever the auth mode.
The cluster generates its own `FV_ADMIN_TOKEN` and keeps it in the state
file, so minting keys and the pools view stay protected.

## Boot (pods created 22:39:01-22:39:08 UTC)

| pod | first `/ping` 204 (loading) | ready (`/ping` or `/health` 200) |
|---|---|---|
| wan | – | ≤ 22:49:47 (≤ 10.7 min; container up at ~22:39:10) |
| h3-turbo (warmup) | 22:49:47 | 22:50:04 (11.0 min) |
| gateway (CPU) | – | 22:58:24 (19.4 min) |
| ltx | 22:57:48 | 23:00:19 (21.2 min) |
| h3-max | 22:57:48 | between 23:00 and 23:11 (the poll stopped at 23:00; all 4 were ready at 23:11) |

Nearly all of the boot time was image pulls: the gateway, ltx and h3-max
showed no container (uptime 0, proxy 404) for 15-19 min, then came up
without intervention. Model load after the pull took about 20 s (wan and
h3-turbo) to 2.5 min (ltx). The PATCH to the gateway came 8 s after its
create, before the pull ended. It did not break the gateway, and did not
visibly delay it compared with the two GPU pods pulled on other hosts.

## Smoke: one text-to-video per pool through the gateway, no API key

Native `POST /fv/v1/jobs`, `aspect_ratio: "16:9"` at the model's smallest
advertised short edge, default length, seed 1. Wall time runs from submit to
the gateway reporting `succeeded`, with status polled every 1 s.

| pool | model | short edge | wall (s) | run on worker (s) | denoise (s) | queue (s) |
|---|---|---:|---:|---:|---:|---:|
| h3-turbo | fasth3 | 480 | 14.4 | 10.7 | 7.2 | 1.0 |
| h3-max | sol-h3 | 480 | 14.3 | 9.9 | 6.2 | 1.0 |
| ltx | ltx25-distill-sol | 720 | 47.0 | 42.4 | 14.6 | 2.7 |
| wan | fastwan22-ti2v-5b | 480 | 12.3 | 8.8 | 1.7 | 1.2 |

The ltx run includes 24.5 s of text encoding on the first job after boot.

Other APIs, one call each without a key, all completed:

- fal: `minimax/h3-turbo/text-to-video` returned COMPLETED.
- MiniMax: `/v2/video_generation` with `MiniMax-H3-Max` at 768P returned succeeded.
- OpenAI-style: `/v1/videos` with `fastwan22-ti2v-5b` returned completed; `/content` returned 200 and 6.5 MB.
- LTX: `/v2/text-to-video` with `ltx-2-5-fast` at 1920x1080 for 6 s returned 202, then completed.

## Backstops and money

- Deadline: create + 6000 s, which is **2026-09-29T00:19:00Z**. Three
  mechanisms enforce it:
  - The gateway pod's watchdog deletes the workers and itself at the
    deadline. It also deletes them if the balance drops below $8.25, checked
    every 60 s.
  - A detached local loop deletes every pod in the state file at the
    deadline.
  - A one-shot Routine at 00:20Z runs in a fresh session and deletes the five
    pods, unless the gateway's `FV_CLUSTER_DEADLINE` was extended.
- A local balance watchdog (every 60 s, floor $8.50) was added at 23:11.
- Cluster cost: $8.42/hr. The account's spend was $12.8/hr, including other
  agents' pods, and the balance was $17.96 at 23:15.

## Teardown and extension

```sh
scripts/serve/runpod-cluster.sh down           # delete all five pods, verify
scripts/serve/runpod-cluster.sh extend 30      # move the deadline 30 min (restarts the gateway pod only)
scripts/serve/runpod-cluster.sh status         # pods, deadline, pools
```

`extend` does not move the Routine. The Routine checks the gateway's
`FV_CLUSTER_DEADLINE` and does nothing while it is in the future.

## Run 2 (2026-09-29): per-variant images, auth on, WebRTC ports

`FV_CLUSTER_CAP_S=10800 FV_MIN_BALANCE=15 runpod-cluster.sh up sha-2cd1ba0`.
A `sha-<commit>` argument deploys the per-variant images of that commit
(docs/serve/images.md). `2cd1ba0` was the newest green serve image, and main
(`8bcf5bd`) differed from it only in docs.

| pod | image tag | digest |
|---|---|---|
| gateway (cpu3c, $0.06/hr) | `gateway-sha-2cd1ba0` | `sha256:00d85291…` |
| h3-turbo | `h3-turbo-sha-2cd1ba0` | `sha256:c782eb37…` |
| h3-max | `h3-max-sha-2cd1ba0` | `sha256:066a5547…` |
| ltx | `ltx-sha-2cd1ba0` | `sha256:680ffa2a…` |
| wan | `wan5b-sha-2cd1ba0` | `sha256:0441e76f…` |

- All workers ran on RTX PRO 6000 in EUR-IS-1 at $2.09/hr. Each worker has
  `8000/http` plus `70000/tcp`: Runpod gives each one a public IP and a
  symmetric TCP port (`RUNPOD_TCP_PORT_70000`), which the WebRTC host
  advertises for ICE-TCP.
- Auth is `keys`. A call without a key gets 401. One user key was minted
  with `runpod-cluster.sh mint`, which uses the admin route and stores the
  key in D1. The admin token exists only in the state file.
- The slim gateway image has no `curl`, so the gateway watchdog installs it
  with apt at boot.

Boot, measured from create at 02:01:57-02:02:02. The slim images pull in
seconds, where run 1's all-in-one image took 10-19 min:

| pod | ready |
|---|---|
| gateway | 02:02:40 (43 s, including the restart from the env PATCH) |
| ltx | 02:03:26 (1.4 min) |
| wan | 02:03:38 (1.6 min) |
| h3-turbo (warmup) | 02:06:23 (4.4 min) |
| h3-max | 02:07:07 (5.1 min) |

Smoke with the minted key (native API, 16:9, smallest short edge):

| pool | short edge | wall (s) | worker run (s) | denoise (s) |
|---|---:|---:|---:|---:|
| h3-turbo | 480 | 14.7 | 10.5 | 7.1 |
| h3-max | 480 | 18.8 | 9.8 | 6.3 |
| ltx | 720 | 40.0 | 35.8 | 13.5 |
| wan | 480 | 13.0 | 8.9 | 1.7 |

`/fv/v1/status` (public) lists the four pools as `ready`, each with one
worker. The console's `common.js`, which contains the status strip, reads
that endpoint. `/console` returned 200.

Backstop: 2026-09-29T05:01:56Z (3 h). It is enforced by the gateway
watchdog (which also deletes at a balance below $15), the local loop, and a
Routine at 05:03Z. A local balance watchdog checks every 60 s with a floor
of $15.
