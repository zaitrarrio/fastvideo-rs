# CloudRift: a third GPU provider for tests and deployments

Date: 2026-10-06. CloudRift (cloudrift.ai) rents GPU servers as Docker
containers or VMs through a REST API. This page covers what the API offers,
how it compares with Runpod and Vast, and what the repo now has for it:
`scripts/gpu/cloudrift.sh` for GPU checks, `scripts/serve/cloudrift-worker.sh`
for fv-serve workers, and a CloudRift provider in fv-control.

**Nothing has run on CloudRift yet.** Everything below comes from the
official docs, the published OpenAPI spec and unauthenticated reads of the
public catalog. It has also been tested against a mock of the API. Any claim
not backed by one of those sources is marked **UNVERIFIED**. Section 9 lists
what the owner must provide for the first live test.

Sources (all read on 2026-10-06):

| Ref | URL |
|---|---|
| [docs] | https://docs.cloudrift.ai/ (sitemap: `/sitemap.xml`) |
| [rest] | https://docs.cloudrift.ai/extras/rest-api |
| [spec] | https://api.cloudrift.ai/api-docs/openapi.json (rift-server **0.62.1**, OpenAPI 3.1; the Swagger UI is https://api.cloudrift.ai/swagger-ui/) |
| [inst] | https://docs.cloudrift.ai/cli-interface/instance-management |
| [vol] | https://docs.cloudrift.ai/features/volumes, https://docs.cloudrift.ai/troubleshooting/volumes-and-reservations |
| [chg] | https://docs.cloudrift.ai/changelog/2026, https://docs.cloudrift.ai/changelog/2025 |
| [price] | https://www.cloudrift.ai/pricing |
| [comfy] | https://docs.cloudrift.ai/tutorials/image-generation/comfyui-tutorial |
| [dstack] | dstack's CloudRift backend, `src/dstack/_internal/core/backends/cloudrift/{api_client,compute}.py` on github.com/dstackai/dstack (third-party: how a working client calls the API) |
| [emmy] | github.com/cloudrift-ai/emmy PR #809 "pin the API version to 2026-09-08" (CloudRift's own repo) |

## 1. The API

### Shape and auth

- Base URL `https://api.cloudrift.ai`. Every operation is a **POST** to
  `/api/v1/<path>` with the body `{"version": "<date>", "data": {...}}`. The
  answer is `{"version": "<date>", "data": {...}}` [spec].
- Auth: the header **`X-API-Key: <key>`** (security scheme `api-key`) or a
  JWT bearer from `/api/v1/auth/login` (scheme `token`) [spec]. The docs say
  user API keys and team API keys both work [rest]. dstack sends
  `X-API-Key` [dstack]. Our code sends only `X-API-Key` and never a bearer.
- Check a key with `POST /api/v1/auth/me`, which returns `{email, id, name,
  provider, totp_enabled}` [spec].
- **Protocol versions.** In v0.62.0 (2026-09-09), `instances/rent` began
  accepting only the v062 protocol. Older pins are rejected [chg]. The date
  that maps to v062 is **`2026-09-08`**, the pin CloudRift's own client moved
  to [emmy]. `instances/list` and `instances/terminate` have required v061
  since v0.61.0 and accept the newer date too [chg][emmy]. Our default is
  `2026-09-08`, which `CLOUDRIFT_API_VERSION` overrides. On the public
  listing the server ignores the pin and answers `"version": "2025-01-29"`
  (observed).
- Which endpoints take an API key, per the spec's `security`:
  - **API key or JWT:** `instances/{rent,list,terminate,stop,start,pause,resume,reset,metrics}`, `account/info` and `auth/me`.
  - **JWT only:** `volumes/{create,list,update}`, `ssh-keys/*` and `account/transactions/list`. Whether an API key works there is **UNVERIFIED**.
  - **`instance-types/list`:** answers without any credentials (observed).

### Endpoints we use

| Need | Call | Fields (from [spec]) |
|---|---|---|
| GPU catalog and prices | `instance-types/list` `{selector: "All" \| {ByServiceAndLocation: {services: ["docker"\|"vm"], datacenters?}} \| {ByName: [...]}}` | `instance_types[]`: `name`, `brand_short`, `datacenters[]`, `variants[]`. Each variant has `name`, `gpu_count`, `vram`, `cpu_count`, `dram`, `disk`, **`cost_per_hour` in cents**, `available_nodes`, `available_nodes_per_dc`, `ip_availability_per_dc`, `volume_types_per_dc` (`cost_per_gb_per_month` in cents) |
| Balance | `account/info` `{}` | `balance` (USD, double) |
| Rent | `instances/rent` (version ≥ 2026-09-08) | `selector: {ByInstanceTypeAndLocation: {instance_type: "<variant name>", datacenters?: [...]}}`, `with_public_ip`, `config` (below), `name?`, `tags?` (free-form strings), `cluster_name?`, `reservation?`, `team_id?`. The answer is `{instance_ids: [...]}` (HTTP 201) |
| Status | `instances/list` `{selector: {ById \| ByStatus: {statuses} \| ByTags: {all?, any?} \| ByClusterName}, mask?}` | `instances[]`: `id`, `status` (`Initializing`, `Active`, `Deactivating`, `Inactive`, `Failed`), `instance_name`, `tags`, `host_address`, `port_mappings` (pairs of ints), `failure.user_message`, `resource_info.cost_per_hour`, `gpus[]`, `created_at`. The `mask` flags: `with_connection_info`, `with_usage_info`, `with_hardware_info`, `with_credentials` (we never set the last one) |
| GPU metrics | `instances/metrics` `{selector: {ById}}` | `metrics[].gpus[]`: `gpu_utilization_percent`, `fb_used_mib`, `power_usage_watts`, `temperature_celsius`, … |
| Terminate | `instances/terminate` `{selector: {ById \| ByTags \| …}}` | `terminated[]` (HTTP 201). Terminating a `Failed` rental dismisses it [inst] |
| Stop / start | `instances/stop`, `instances/start` | Stop is a graceful shutdown. For a reserved instance, stopping **terminates** it and its ephemeral disk [vol] |
| Volumes | `volumes/create` `{datacenter, name, size_gb, volume_type_name, team_id?}`, `volumes/list` `{selector: "All" \| ById \| ByName}` | `VolumeInfo`: `id`, `size_gb`, `cost_per_month`, `datacenter_name`, `datacenters[]`, `status`. The public spec has no delete; that is done in the console [vol] |

The Docker config of a rental:
`config: {Docker: {image, command?: [string], env?: [[name, value], ...], ports?: ["<host>:<container>/<tcp|udp|sctp>"], registry_auth?: {UsernamePassword: {username, password}}, volumes?: {Mounts: [{volume: {ById|ByName}, mount_path}]}}}`
[spec]. VMs use `config: {VirtualMachine: {image_url, cloudinit_commands?, ssh_key?, ports?, volumes?}}`, which is the path dstack takes.

### What the API does not have

- **Logs:** there is no container-log endpoint in the public API (none in
  [spec]). Results and logs come back over SSH (gpucheck) or HTTP (fv-serve).
- **Public HTTPS:** no managed HTTPS proxy is documented, unlike Runpod's
  `https://<pod>-8000.proxy.runpod.net`. A Docker rental with
  `with_public_ip: true` exposes its mapped ports on `host_address` over
  plain TCP (the ComfyUI tutorial opens `http://<node IP>:8188` [comfy]).
  TLS takes a tunnel such as cloudflared in the container, or a TLS
  terminator in front. Some datacenters report
  `ip_availability_per_dc.public_ips: false` (observed for
  `ap-east-tw-kn-2`). Rentals there would have no reachable port, so our
  scripts only pick a datacenter that has free stock.
- **Secrets:** there is no secret store, so env values are stored in the
  rental config as given (**UNVERIFIED** who at CloudRift can read them).
- **A spend cap or idle stop on the server side:** none in [spec] or [docs].
  CloudRift sends low-balance e-mails to team owners [chg]. **Volumes are
  deleted when the balance is depleted** [vol].
- **Serverless or queue endpoints:** none. CloudRift is rentals only (plus an
  LLM inference API that does not apply here).

### Billing

Billing is per second of active runtime with no minimum, and there are no
egress, ingress or API-call charges [price]. A rental that fails to start is
not charged [chg]. Persistent volumes are billed until deleted, even with no
instance attached [vol].

## 2. CloudRift compared with Runpod and Vast

1-GPU on-demand prices from the public catalog (`cloudrift.sh catalog`,
2026-10-06, in $/hr) and from [price]:

| GPU | CloudRift | Free nodes (public listing) | Runpod (repo figures) | Vast |
|---|---|---|---|---|
| RTX PRO 6000 96 GB | 1.34-1.39 (three host types); "upon request" on [price] | 0 | 2.09 (EUR-IS-1, recorded 2026-09-24 in docs/gaps/2026-09-24-phase3-vs-published.md) | marketplace |
| RTX PRO 6000 Max-Q | 1.55 | 0 | - | marketplace |
| RTX 5090 32 GB | 0.62-0.65 | 0 | listed | marketplace |
| RTX 4090 24 GB | 0.39-0.48 | 0 | listed | marketplace |
| L40S 48 GB | 0.63 | 0 | listed | marketplace |
| A100 SXM4 80 GB | 1.05 | 0 | listed | marketplace |
| H100 80 GB / H200 141 GB | "upon request" [price]; **not in the API catalog** | - | our US pools (`REGIONS.us`) | marketplace |
| B200 | not offered | - | `runpod.sh b200` | marketplace |
| V100 SXM2/SXM3 | 0.25-0.28 | 1-3 | - | - |
| AMD MI350X 288 GB | 4.00 | 0 | - | - |

- **Stock.** The unauthenticated listing showed free nodes only for V100,
  which CUDA 13 does not support, so our images cannot use it. Whether a
  logged-in account sees more stock is **UNVERIFIED** (the CLI's
  `instance-type list` does the same call with a key). RTX PRO 6000, H100
  and H200 at "upon request" suggest capacity by arrangement.
- **Regions.** The datacenters in the listing are `ustx1a_a01` and
  `usny01_a01` (USA), `eu-central-it-gv-1` (Italy),
  `ap_northeast_kr_se_1` (Korea) and `ap-east-tw-kn-2` (Taiwan). Most
  types had no datacenter listed at the time.
- **What fits us.** RTX PRO 6000 at about $1.34-1.39/hr would be about a
  third cheaper than the $2.09/hr recorded on Runpod, if there is stock.
  RTX 4090/5090 are cheap for CUDA smoke tests (sm89/sm120). There is no
  B200 and no listed H100/H200. Per-second billing and free egress beat
  Runpod's per-minute billing.
- **What CloudRift lacks against Runpod.** There is no HTTPS proxy, no
  serverless queue, no secret store, no logs API and no API to patch a
  rental's env (env changes need a new rental). Network volumes exist only
  at some partners, and the API manages them with a JWT only.
- **Against Vast.** CloudRift has the same "rent a container with a public
  port" model. It has no `ssh_direct` runtype or onstart (we run sshd
  ourselves), but it has tags, a balance endpoint and per-instance GPU
  metrics. Vast's money guards (`VAST_MAX_DPH`, backstop, ledger) map
  one-to-one.

## 3. Weights (proposal: needs the owner's approval)

Runpod network volumes cannot be mounted on CloudRift. The options:

| Option | How | Cost | Time | Notes |
|---|---|---|---|---|
| **A. CloudRift persistent volume, filled once from the Hub** (recommended once a pool runs there) | `volumes/create` in a datacenter with volume support, then one rental with the volume mounted that runs `hf-fm` per manifest row (`scripts/gpu/rebuild-volume.sh` logic) and checks `verify-weights.sh` / `weights-sha256.tsv` | Storage: `volume_types_per_dc` was empty on the public listing, so the $/GB-month is **UNVERIFIED**. The fill rental is about $0.70 (RTX PRO 6000 for about 30 min) | Hub rate on Runpod was 150-260 MB/s per 8-vCPU pod (runpod-volumes.md §7). One pool's trees, e.g. h3-turbo `h3-8step` ~148 GB, take about 10-17 min. CloudRift's network rate is **UNVERIFIED** | Volumes exist only at some partners, and the docs name RTX PRO 6000. Volumes attach at rent time only. **They are deleted when the balance runs out**, so never keep the only copy there. The volume API takes a JWT, not an API key (spec): first creation in the console |
| B. R2 mirror of the trees | Copy each tree once from a Runpod volume to an R2 bucket (S3 API), and have workers sync the trees they need at boot | R2 storage is $0.015/GB-month with free egress (Cloudflare's published price, not re-checked today). 1.2 TB is about $18/month | Each boot pulls 25-150 GB per pool. Speed from CloudRift to R2 is **UNVERIFIED** | Serves Vast and any other provider too. Costs every boot time unless paired with A |
| C. Hub at every boot | `hf-fm` into the container disk | none | Like A, at every start | Only for small trees (wan5b ~24 GB) or one-off tests |

**Recommendation.** For testing, use **no weights** (gpucheck's random-weight
cells and the fake engine). Before the first real pool, ask the owner to
approve **A** for that pool's trees only, in one datacenter with RTX PRO 6000
volume support, and size it per pool (e.g. 200 GB for h3-turbo). Treat it as
a cache: the Runpod volumes stay the record of truth (CLAUDE.md), every tree
is verified against `weights-sha256.tsv`, and nothing new lands only on
CloudRift. Consider **B** when a third provider needs the same trees, or if
CloudRift volumes prove unavailable where the stock is. CLAUDE.md's rules
cover only the two Runpod volumes, so a CloudRift volume needs an owner
decision. Suggested rule: caches of manifest trees, add-only, verified,
listed in a new `scripts/gpu/weights-cloudrift.tsv`.

## 4. Cost safety

| Guard | gpucheck (`cloudrift.sh`) | worker (`cloudrift-worker.sh`) | fv-control |
|---|---|---|---|
| Balance floor (Runpod: $8) | `CLOUDRIFT_MIN_BALANCE`, default 8: `account/info` before every rent, and no rental if the balance is unknown | same | `CLOUDRIFT_BALANCE_FLOOR` (default `BALANCE_FLOOR`): critical alert; with `stop_on_floor`, terminates **our** rentals |
| Price cap | `CLOUDRIFT_MAX_DPH` (1.5), checked on the catalog price **before** the rent | `CLOUDRIFT_MAX_DPH` (1.0) | `GET /api/providers/cloudrift/price?gpu=` |
| Wall-clock backstop | detached `sleep cap; terminate` (`CLOUDRIFT_CAP_S`, 1800 s), the key only in its env | smoke 1800 s, up 3600 s | the `fv-deadline:<unix>` tag on every rental: the cron terminates our rentals past it |
| Idle guard | while waiting: GPU under `CLOUDRIFT_IDLE_GPU_PCT` (5%) with no new output for `CLOUDRIFT_IDLE_MIN` (15) minutes means terminate. The container also exits `FV_IDLE_S` (1200 s) after it finishes | - | `pod_idle` alert from `instances/metrics` |
| Exit trap | terminate on EXIT/INT/TERM, three tries, confirmed through `instances/list` | smoke: same | - |
| Ownership | tags `fv`, `fv-owner:fastvideo-rs`, `fv-kind:<kind>`; `reap` and fv-control touch **only** these | same | `POST /api/providers/cloudrift/instances/:id/terminate` refuses others (403) |
| Ledger | `artifacts/cloudrift/ledger.tsv` | same | audit log `cloudrift.terminate` |

Whether a container that exits leaves the rental `Active` and billed is
**UNVERIFIED**. The scripts never rely on the exit; they terminate.

## 5. Access: images, build pod and CI

- The CI images on GHCR are **public**. Anonymous pulls of
  `ghcr.io/zaitrarrio/fastvideo-rs-runtime` and `…-serve` returned 200 (checked
  2026-10-06). Compressed size is about 1.57 GB (runtime) and 1.60 GB
  (serve) for `:latest`. CloudRift pulls them with no `registry_auth`. A
  private image would need `registry_auth.UsernamePassword` with a
  read-only GHCR token, which CloudRift stores in the rental.
- **Pin digests:** `cloudrift-worker.sh up` refuses a tag, and the other
  commands warn.
- The build pod stays on Runpod (CLAUDE.md). CloudRift only runs what CI
  published. `scripts/gpu/build-remote.sh`-style rsync of a fresh binary
  works over the gpucheck container's sshd (`cloudrift.sh launch`, then
  ssh to `root@<host>:<port>`), as on Runpod.
- `fv-gpucheck`'s runtime image needs a CUDA 13.4 driver (≥ 580). Which
  driver CloudRift hosts run is **UNVERIFIED**. The smoke writes
  `nvidia-smi.txt` first, so it shows the version.

## 6. The scripts

### `scripts/gpu/cloudrift.sh`: checks and benchmarks

```
cloudrift.sh plan                    # the rent payload, no key
cloudrift.sh catalog                 # public catalog with free stock
cloudrift.sh smoke [image]           # nvrtc + fast kernels, results to artifacts/cloudrift/<id>/
cloudrift.sh run [image] -- --mode fast kernels     # one custom fv-gpucheck step
cloudrift.sh launch | wait <id> | fetch <id> | down <id> | status | reap
```

The container is the runtime image with
`command: ["bash", "-c", BOOT]`. The boot installs the public key from
`FV_SSH_PUBKEY`, starts sshd on 22 (published as `2222:22/tcp`) and runs
each fv-gpucheck step from `FV_STEPS_B64`. It writes
`/workspace/gpucheck-out/{nvidia-smi.txt,run.log,DONE}`. The driver waits
for `DONE`, rsyncs the directory, writes `summary.json` (rent-to-ssh time,
wall time, estimated cost) and terminates.

### `scripts/serve/cloudrift-worker.sh`: fv-serve

```
cloudrift-worker.sh plan                       # payload, secrets masked
cloudrift-worker.sh smoke [image]              # standalone fake engine: /healthz on the public port, one job
cloudrift-worker.sh up <pool> <image@digest>   # a worker for the gateway's pod pool
cloudrift-worker.sh down <id>
```

It joins a gateway's pod pool (`kind = "pod"`, docs/serve/gateway.md §5.3):

- **`CLOUDRIFT_CMD_MODE=args` (default).** `command` is fv-serve's arguments
  (`--config …`), and the image ENTRYPOINT (`fv-serve` or `fv-entry`) stays.
  The worker runs `FV_SERVE_ROLE=worker`, `FV_GATEWAY_POOL`, the gateway's
  `FV_INTERNAL_TOKEN` and the D1/R2 secrets. `up` prints
  `http://<host>:<port>`, which goes in the gateway's
  `FV_POOL_<POOL>_URLS` (a static pod pool, like `runpod-cluster.sh`).
- **`CLOUDRIFT_CMD_MODE=exec`.** This mode works only if `command` replaces
  the entrypoint. A shell boot finds the public IP, sets
  `FV_PUBLIC_BASE_URL=http://<ip>:<port>` and starts fv-serve. The worker
  then **registers itself** in `gw_workers` (D1), and gateways pick it up
  within 45 s with no gateway change.

**Plain HTTP.** Gateway-to-worker traffic carries `x-fv-internal-token` in
clear. Before production, put a tunnel (cloudflared) or a TLS terminator in
front of the worker, and use scoped, revocable D1/R2 tokens, because
CloudRift keeps the rental env.

### Tests

`bash scripts/gpu/tests/cloudrift.test.sh` (25 checks) runs both scripts
against `scripts/gpu/tests/cloudrift_mock.py`, a fake API that also answers
for the rented fv-serve container, with ssh and rsync stubs. It covers:

- plan without a key, and the public catalog;
- the balance floor and the price cap (no rent);
- smoke, results and termination;
- the idle guard and the detached backstop;
- the worker smoke and `up`;
- `reap` sparing a foreign rental;
- the key in no output or ledger line.

## 7. fv-control

fv-control's cluster engine (`src/cluster/*`) is Runpod-specific. It builds
Runpod pod payloads, PATCHes env, uses the Runpod proxy URL and runs the
gateway watchdog against the Runpod API. This change adds CloudRift as a
**second provider for visibility and cost safety**, and leaves the cluster
engine alone:

- `src/cloudrift.ts`: the client (balance, live rentals, metrics,
  terminate, catalog price). `X-API-Key` only, and errors are scrubbed of
  the key.
- `src/collector-cloudrift.ts`: run by the per-minute cron when
  `CLOUDRIFT_API_KEY` is set. It writes rentals to `pods` with
  `provider = 'cloudrift'` (migration `0003_providers.sql`), accrues
  `cost_daily`, tracks idle time from GPU metrics, and attributes owners
  (`cloudrift:<fv-kind>` for ours, `external:cloudrift` for others). It
  terminates our rentals past their `fv-deadline` tag, dismisses our
  `Failed` ones, and enforces the CloudRift balance floor. A CloudRift
  outage raises a `cloudrift_api` alert and never disturbs the Runpod half.
- API: `GET /api/providers`, `GET /api/providers/cloudrift/price?gpu=`,
  `POST /api/providers/cloudrift/instances/:id/terminate` (ours only). The
  overview carries a `cloudrift` block (balance, floor, $/hr, hours to
  floor).
- Config: `wrangler secret put CLOUDRIFT_API_KEY`, plus the optional vars
  `CLOUDRIFT_API`, `CLOUDRIFT_API_VERSION`, `CLOUDRIFT_BALANCE_FLOOR` and
  `CLOUDRIFT_COST_UNIT`. The last one exists because the spec types
  `resource_info.cost_per_hour` as "currency units" while the catalog uses
  cents; the default is `usd`, **UNVERIFIED** until the first live listing.
- Tests: `test/unit/cloudrift.test.ts` (12) and one integration step against
  a CloudRift mock in `test/harness.mjs`.

**Plan for CloudRift pools in clusters (not done here).**

1. Add `provider?: "runpod" | "cloudrift"` to `PoolSpec`, plus CloudRift
   placements (`gpu_brands`, `datacenters`).
2. Write a `Provider` interface (`create`, `remove`, `get`, `account`,
   `podUrl`) with `runpod.ts` and `cloudrift.ts` behind it. `createWorker`
   and `deletePod` go through it, and `cluster_pods` gets a `provider`
   column.
3. CloudRift workers can't be PATCHed: `restart` and `roll` re-create them,
   which `roll` already does.
4. The gateway reaches CloudRift workers at `http://<host>:<port>` (static
   pool URLs). This needs the TLS decision in section 6 first.
5. Teach the gateway watchdog's `kill_all` to terminate CloudRift workers
   (a `FV_CLOUDRIFT_PODS` list and the CloudRift key on the gateway).
   Alternatively, rely on the `fv-deadline` tag and the controller cron.
6. The price check sums CloudRift catalog prices. The balance projection
   checks both accounts.

Steps 1-6 come to about 400-600 lines plus tests. The live API facts in
section 8 should be confirmed first.

## 8. UNVERIFIED items

The first live smoke test settles these:

1. Whether Docker `command` replaces the image ENTRYPOINT or only CMD. The
   spec has no entrypoint field, which suggests CMD. gpucheck works either
   way (its image has no ENTRYPOINT). For fv-serve, `args` mode assumes CMD
   and `exec` mode assumes replace.
2. The order inside `port_mappings` (we read `[container, host]` as dstack
   does), and whether the requested host port is honoured.
3. Whether an exited container stops billing.
4. The unit of `resource_info.cost_per_hour` (dollars or cents).
5. Stock and datacenters as seen with a key. The public listing showed only
   V100 free.
6. Host NVIDIA driver versions: CUDA 13.4 needs a ≥ 580 driver.
7. Volume prices, which datacenters support volumes, and whether the volume
   API takes an API key.
8. Whether tags with `:` are accepted in rent. The spec's example is
   `["env:prod", "team:ml"]`, so this is likely fine.

## 9. First live test: what the owner provides, and a ≤ $3 plan

**Provide:**

- **A CloudRift account with a user API key**, readable as
  `CLOUDRIFT_API_KEY` or in `/root/.config/fv/cloudrift_api_key` (mode
  600). A team key works too.
- **Funding of at least $10.** The scripts refuse to rent below $8, and
  volumes are deleted at zero.
- Optionally, CloudRift support's word on RTX PRO 6000 availability
  ("upon request").

**Plan (worst case about $2):**

1. **Read-only ($0).** Run `cloudrift.sh balance` and `cloudrift.sh catalog`
   with the key, then confirm items 4, 5 and 7 through `auth/me` and
   `instances/list`.
2. **gpucheck smoke (about $0.20-0.70).**
   `CLOUDRIFT_GPUS="RTX 4090,RTX 5090,RTX PRO 6000" CLOUDRIFT_MAX_DPH=1.5 CLOUDRIFT_CAP_S=1500 cloudrift.sh smoke ghcr.io/zaitrarrio/fastvideo-rs-runtime@sha256:<digest>`.
   At most 25 minutes at $0.39-1.39/hr. It records the rent-to-ssh time
   (boot plus a 1.6 GB pull), driver version and kernel results, and
   settles items 1, 2 and 6.
3. **Serve smoke (about $0.20-0.50).**
   `CLOUDRIFT_CAP_S=1200 cloudrift-worker.sh smoke ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:<digest>`
   with the fake engine. It records rent-to-Active, Active-to-`/healthz`
   (the public port) and one job, and checks the CMD semantics (item 1).
4. **Termination check ($0).** Run `cloudrift.sh status` (expect nothing
   live), `instances/list ById` (expect `Inactive`), and confirm the
   balance delta matches the summaries (item 3).

Every step has a backstop of 25 minutes or less. Step 3 runs only if step 2
passed. No weights are downloaded.
