# Build pods managed by fv-control (design)

**Status (2026-10-06):** implemented on branch `wip/fvc-build-pods`
(control/src/buildpods.ts, the pod server's R2 cache, `fv-control.sh
build-pod`, `build-pod.sh` as a client); tested against mocks (unit,
integration, UI), not yet on staging or a live pod. Owner decision:
fv-control manages the whole lifecycle of the CPU build pods, as it does for
GPU pods. Today `scripts/dev/build-pod.sh` creates, starts, stops and deletes
one shared pod pinned to the `fv-build` volume (`pxy4hlsnwq`, EU-RO-1), and
fv-control only has a backstop (`control/src/buildpod.ts`). The pod's HTTP
service (`scripts/dev/build-pod-server.py`: sync, jobs, artifacts, self-stop,
eviction, deps seeds, and, from PR #32, `POST /v1/runner`) does not change
its contract.

## 1. Goals and non-goals

Goals:

- fv-control creates, starts, stops, replaces and deletes **one or more**
  build pods in **any** Runpod datacenter. It places them by CPU stock and
  price, and each pod has its own idle stop and hard cap. The existing
  backstop becomes the per-pod backstop.
- Build pods are ordinary controller pods for **costs**: the per-minute
  ledger (`cost_daily`), spend per owner, alerts and the console.
- Each pod registers as a **GitHub self-hosted runner** (labels `fv-build`
  and `fv-build-<region>`). fv-control holds the repo-admin PAT, mints the
  1-hour registration token and sends it to the pod's `/v1/runner` (PR #32's
  contract). It removes the runner when the pod stops or is deleted.
- **Wake on demand**: from the cron (queued GitHub jobs that need
  `fv-build`), from the API and console button, from
  `fv-control.sh build-pod up|down|status`, and from CI before it picks a
  builder (§6).
- **Caches that follow the pod**: a per-region network volume is optional.
  sccache and the deps seeds are shared through R2 (§7).
- `build-pod.sh` keeps sync/run/fetch/status/seed/evict/release-artifacts
  as a client of the pod's HTTP API. Its lifecycle commands call
  fv-control, and it can no longer delete a pod.

Not goals: changing the pod's job API, the allowlist or the per-agent
worktree and `CARGO_TARGET_DIR` model (all unchanged); GPU builds; other
providers (CloudRift/GCP have no CPU build path today, so the design keeps a
`provider` column but implements Runpod only).

## 2. The rule that removes the incident

Today `build-pod.sh up`, run from any branch, may **delete** a stopped shared
pod whose server or image differs from that branch's checkout, and create
one from the branch's `build-pod-server.py`. With managed pods:

- **Only fv-control creates, replaces or deletes build pods.** `build-pod.sh`
  has no Runpod calls left, and the Runpod key is not needed to build.
- The server code and image come from **`main`** by default (fetched from
  GitHub by fv-control at create time: `scripts/dev/build-pod-server.py` and
  the `BASE_IMAGE_TAG` pin in `scripts/dev/build-pod.sh`, at a `server_ref`
  that defaults to `main`). A branch can no longer put its server on the
  shared pod.
- A test pod with another ref (`POST /api/build-pods {server_ref, purpose:
  "test"}`, admin only) is a **separate** pod. Its purpose is `test`, it has
  no runner, and `up` never hands it out.
- **Replace only when idle.** A pod whose server sha or image differs from
  the current one is marked `outdated`. A running outdated pod is never
  touched. When it is stopped (its own idle stop, or `down`), the next `up`
  deletes it and creates a new pod instead of starting it; a stopped pod
  runs no jobs. `POST …/replace` refuses while `/healthz` shows active jobs
  or GitHub shows its runner `busy`.
- `down` (stop) refuses while jobs or the runner are busy, unless the caller
  passes `force` (audited).

## 3. Data model (D1 migration `0004_build_pods.sql`)

`build_pods`: one row per managed pod (it stays after deletion for the
ledger):

| column | |
|---|---|
| `id` | `bp_<hex>` |
| `name` | `fv-build-<region>-<n>` (Runpod pod name) |
| `pod_id` | Runpod id (null before the create call returns) |
| `purpose` | `shared` (handed out by `up`) or `test` |
| `state` | `creating`, `running`, `stopping`, `stopped`, `deleted`, `failed`. Runpod is the source of truth; the cron reconciles. |
| `dc`, `region`, `flavor`, `vcpu`, `disk_gb`, `cost_per_hr`, `volume_id` | placement |
| `image`, `server_ref`, `server_sha` | what it runs (`outdated` is computed against the current ones) |
| `token_sealed`, `token_sha` | the pod's bearer token, AES-GCM under `CONTROL_KEK` (AAD = row id), and its sha256 (what the pod gets) |
| `idle_min`, `max_h`, `max_grace_min` | per-pod limits, sent to the pod's env |
| `runner_name`, `runner_id`, `runner_state`, `runner_error`, `labels` | GitHub runner |
| `created_at/by`, `started_at`, `stopped_at`, `deleted_at`, `last_error` | |

`locks`: `(key, holder, expires_at)`, a compare-and-set lease, so that two
concurrent `up` calls (agents in parallel) do not create two pods.

**Pool policy** (settings key `build_pods`, the Settings page, `GET/PUT
/api/build-pods/policy`): `enabled`; `max_pods` (2); `max_dph_per_pod`
(1.5); `daily_usd_max` (20; above it, `up` refuses and an alert opens);
`balance_margin` (2 above the account floor `BALANCE_FLOOR`); `flavors`
(`cpu5c cpu3c`); `vcpus` (`32 16`); `disk_gb` (200) / `disk_gb_fallback`
(80); `regions` (preferred DC prefixes, default `EU`, then any listed DC);
`volumes` (DC → network volume id, default empty; §7); `idle_min` (20),
`max_h` (8), `max_grace_min` (30); `backstop_margin_min` (30); runner
`labels` (`fv-build`); `wake_on_queue` (on); `wake_workflows` (empty);
`server_ref` (`main`); `image` (empty = the pin at `server_ref`); and the
R2 cache (`cache.r2_bucket`, `cache.r2_endpoint`, on when the Worker has the
R2 key secrets).

## 4. Lifecycle

All lifecycle actions run in the Worker. Every Runpod call is short (one
REST create/start/stop/delete); the Durable Object is not needed. Waiting
for readiness is the caller's polling plus the cron.

**Placement** (`placeBuildPod`): two GraphQL reads. The first is
`dataCenters { id listed storageSupport }`. The second is one aliased
query of `cpuFlavors { specifics(input: {dataCenterId, instanceId:
"<flavor>-<vcpu>-<ram>"}) { stockStatus securePrice } }` per candidate.
Seen live on 2026-10-06: CPU prices are the same in every DC (cpu3c-32
$0.96, cpu5c-32 $1.12, cpu3c-16 $0.48, cpu5c-16 $0.56); stock varies
(cpu5c-32 had none anywhere; cpu3c-32 High in EU-RO-1 and EUR-IS-1;
cpu3c-16 also in EU-NL-1 and US-CA-2). Candidates are ordered by:

1. a DC with a configured volume;
2. the preferred regions;
3. size (`vcpus`, largest first);
4. the `flavors` order;
5. stock (High > Medium > Low);
6. price.

Candidates without stock or above `max_dph_per_pod` are dropped. fv-control
then creates the pod on the first candidate (`dataCenterIds: [dc]`,
`cpuFlavorIds: [flavor]`). It falls through to the next on "no instances
available", and retries once with Runpod's container-disk cap when that is
the error (the logic of `build-pod.sh create_pod`). A create answer above
the $/hr cap is deleted at once.

**Payload**: the start command of `build-pod.sh` (the server from
`FV_BUILD_SERVER_B64` and the curl watchdog), ported to TypeScript. The env
is `FV_BUILD_TOKEN_SHA256`, `FV_BUILD_SERVER_B64` (gzip+base64 of the
server at `server_ref`), the limits, eviction, `FV_BUILD_IMAGE`,
`FV_BUILD_POD_NAME`, and, without a volume, `FV_BUILD_ROOT=/root/fvb-cache`
(container disk) plus the R2 cache settings (§7). It is never returned by
the API.

**`up`** (`POST /api/build-pods/up {region?, wait?}`, under the lock):

1. A running `shared` pod (in `region` if given): the one with the fewest
   active jobs. It is returned as is, outdated or not.
2. Else a stopped, not outdated `shared` pod: Runpod `start`. If the start
   is refused (the host has no free CPU), the pod is deleted and step 3
   runs.
3. Else a stopped outdated pod is deleted, and a new pod is placed and
   created.

Each step first checks: the policy is enabled, the balance is ≥ floor +
margin, build-pod spend today is < `daily_usd_max`, and pods < `max_pods`.
The answer has the pod view (`id`, `pod_id`, `url`, `state`, `phase` from
`/healthz`) and the **pod token** (admin callers only; the read is
audited). The client polls `GET /api/build-pods/<id>` until `phase =
ready`.

**`down`** (`POST /api/build-pods/<id>/stop {force?}`) and **delete**
(`DELETE /api/build-pods/<id> {force?}`): both refuse while jobs are active
or the runner is busy, unless `force`. A stop that Runpod refuses becomes a
delete, as today. Before either, the runner is removed from GitHub.

**Cron** (inside `collect`, which already has every pod from one GraphQL
call):

- Reconcile states from Runpod: `RUNNING` → running, `EXITED` → stopped,
  missing → deleted. Set `cost_per_hr`.
- **Backstop per pod** (replaces the global `build_pod_*` check for managed
  pods): stop once Runpod's uptime ≥ `max_h + max_grace_min + backstop
  margin`, or once `/healthz` shows no jobs and idle ≥ `idle_min +
  backstop margin`. Alert kind `build_pod`. Pods named `fv-build` that are
  not managed (the legacy pod) keep the old global policy until they are
  gone.
- **Balance floor**: below the floor, managed build pods are stopped like
  controller clusters (`stop_on_floor`).
- **Runner**: register a running, ready pod that has no runner yet (§5).
  Remove the runner of stopped or deleted pods, and offline runners named
  `fv-build-*` whose pod is not running.
- **Wake** (§6).

**Costs**: the collector names managed pods' owner `build-pod:<name>` (looked
up by `pod_id` before name attribution). So `cost_daily`, spend by owner,
the overview and the AE samples include them like cluster pods. The alert
`build_pod_spend` fires when today's build-pod spend exceeds
`daily_usd_max`. The console's **Build pods** card lists each pod (region,
flavor, $/hr, state, `/healthz` timers, jobs, runner, spend today) with
**Up**, **Stop**, **Delete** buttons and the policy.

## 5. GitHub runner

- Worker secret `GITHUB_RUNNER_PAT`: a fine-grained PAT on
  `zaitrarrio/fastvideo-rs` with **Administration: read and write** (and
  Actions: read for the queue check). It is separate from `GITHUB_PAT`,
  which dispatches releases, so each token has one job. The code falls back
  to `GITHUB_PAT` if that token was given Administration too.
- Register: `POST /repos/{repo}/actions/runners/registration-token` → `POST
  <pod>/v1/runner {token, repo, labels: "fv-build,fv-build-<region>", name:
  "fv-build-<pod_id>"}` with the pod token. The registration token is never
  stored or logged; it expires in 1 h and is single use. fv-control then
  polls `GET /v1/runner` from the cron. A server without `/v1/runner`
  (404, before PR #32) gives `runner_state = unsupported`.
- Deregister: `GET /repos/{repo}/actions/runners?per_page=100` → find by
  name → `DELETE /repos/{repo}/actions/runners/{id}`. This runs before a
  stop or delete fv-control does, and in the cron for pods that stopped
  themselves. It needs nothing from the pod (the pod may be gone). It never
  deletes a `busy` runner; a busy runner also blocks `down` without
  `force`.
- A stop loses the container disk, and the runner's credentials with it.
  So each start registers again (`--replace` takes over the name).

## 6. Wake on demand (and the CI builder choice)

**Queued jobs** (`wake_on_queue`, every cron minute): list runs with
`status=queued` and `status=in_progress` (a run is `in_progress` while
its plan job ran and its `fv-build` job is queued). Then list each run's
jobs and count the `queued` ones whose `labels` include `fv-build`. If
there are any and no online, idle runner carries all their labels, `up`
runs with the region from an `fv-build-<region>` label. That is at most
1 + (runs) GitHub calls a minute, far under the 5000/h limit. Runs older
than 24 h are ignored.

**`wake_workflows`** (opt-in, empty by default): a queued or running run of
a listed workflow (e.g. `tools-release.yml`) also wakes a pod. This saves
the ~2 min boot, and costs one idle period (~$0.30) when the plan then
skips.

**The CI builder choice.** In PR #32, `tools-release.yml`'s plan job runs
`tools-release.sh pick-runner`. It lists the runners with a repo secret
`FV_RUNNER_READ_TOKEN` (a GitHub PAT, Administration: read). It builds on
the pod only when an idle `fv-build` runner is online; otherwise it builds
on GitHub-hosted. With fv-control managing pods, a stopped pod would
always lose to GitHub-hosted. The proposal is an fv-control endpoint the
plan job calls instead:

```
POST /api/ci/build-runner {workflow, run_id, sha, wait_s?: 0..240}
  → {builder: "pod"|"github", reason, pod?: {id, region, runner}}
```

- Auth: an fv-control API token with the new scope **`ci`**, stored as the
  repo secret `FV_CONTROL_CI_TOKEN`. A `ci` token can call only
  `/api/ci/*`. It can wake a pod but cannot read pod tokens, stop pods or
  change anything else. The GitHub PAT stays only in fv-control, and the
  repo needs no Administration-read secret.
- Logic: if an idle online `fv-build` runner exists → `pod`. Else, if
  build pods are enabled and allowed (floor, budget, `max_pods`), run `up`
  and answer `{builder: "wait", retry_after_s}` until the runner is online,
  then `pod`. If `wait_s` runs out, or `up` is refused → `github`, with
  the reason. The plan step polls for at most `wait_s` (default 180 s: a
  boot takes 80 s, plus the runner registration). Polling, not one long
  request: Workers' request time, and the plan job's own timeout, stay
  short.
- `pick-runner` change (on #32's side, after both merge): if
  `FV_CONTROL_CI_TOKEN` is set, call the endpoint, else keep the read-token
  path. It is about 15 lines of bash. This branch adds the endpoint; the
  read-token path keeps working unchanged.

Workflows on `pull_request` must still never use `fv-build` (#32's safety
rule). fv-control only wakes for jobs; it does not decide which code runs.

## 7. Caches when pods live anywhere

Today everything slow to rebuild lives on the `fv-build` volume in EU-RO-1:

- sccache (40 GB cap; 92 % hit rate measured on a second pod over the same
  volume, 2026-10-06);
- deps seeds (`<key>.tar.zst`, 0.88 GB each, 3.6 GB unpacked, extracted in
  4–6 s);
- the cargo registry cache;
- `release-cache/`.

A pod in another DC cannot mount that volume (volumes are per DC), and a
200 GB volume per region costs $14/month each.

**Design: R2 is the shared cache; a volume is an optional local
accelerator.**

| cache | without a volume | with a volume in the pod's DC |
|---|---|---|
| sccache | **S3 backend on R2** (`SCCACHE_BUCKET=fv-build-cache`, `SCCACHE_ENDPOINT=https://<acct>.r2.cloudflarestorage.com`, `SCCACHE_REGION=auto`, `SCCACHE_S3_KEY_PREFIX=sccache/`), the same keys from every region | the same R2 backend (sccache 0.18 has one backend; R2 keeps hits across regions) |
| deps seeds | R2 `deps-seed/<key>.tar.zst` (+ `.json`): download if not local, upload after a seed build | the volume first, then R2; seeds built here are uploaded too |
| cargo registry | downloaded from crates.io per fresh pod (CDN) | the volume, as today |
| release-cache | rebuilt (oxide ~3 min) | the volume, as today |

- **Credentials**: an R2 API token scoped to the one bucket `fv-build-cache`,
  with Object Read & Write. It is held as Worker secrets
  (`BUILD_CACHE_R2_ACCESS_KEY_ID`, `BUILD_CACHE_R2_SECRET_ACCESS_KEY`) and
  put into the pod's env at create. The server passes it **only** to the
  sccache server process and its own seed upload/download, never to jobs
  (`job_env` drops `FV_BUILD_R2_*` and `AWS_*`). Jobs talk to the local
  sccache daemon.
- **Server changes** (`build-pod-server.py`): start sccache with the R2
  env when `FV_BUILD_R2_BUCKET` is set; a small stdlib SigV4 client (head,
  get to file, streamed put with `UNSIGNED-PAYLOAD`) for seeds; `status`
  shows the sccache backend. Without the R2 settings, nothing changes
  (today's pod on its volume).
- **Retention**: a bucket lifecycle rule deletes objects 30 days after
  upload. sccache on S3 has no size cap and does not refresh objects on a
  hit, so a dependency unchanged for 30 days compiles once again.
- **The `fv-build` volume** is never deleted. Once the legacy pod is gone,
  the owner may set `volumes: {"EU-RO-1": "pxy4hlsnwq"}`. A managed pod in
  EU-RO-1 then mounts it, and only one at a time (`up` does not attach a
  volume that another running pod uses; sccache's local LRU and the seed
  writer assume one writer).

**Expected hit rates.** Measured numbers are from 2026-10-06
(docs/dev/build-pod.md); the R2 ones are estimates until measured.

- Dependencies: the same toolchain and `Cargo.lock` give the same sccache
  keys, so a new pod in any region should hit like the measured 92 %
  second pod. The deps seed covers the first job of each agent (one
  zstd stream). Workspace crates do not hit across agents (path-keyed, see
  "Caches" in docs/dev/build-pod.md), as today.
- A cold R2 fill (first pod after a toolchain or lock change) is ~7 k PUTs.
  A warm new pod is ~7 k GETs at 32-way parallelism. At 50–200 ms per
  cross-region GET that is ~10–45 s of lookups, against minutes of
  compile.
- Seed download from R2 at 50–150 MB/s: 6–18 s for 0.9 GB, against 4–6 s
  from the volume. Without any cache, a new agent's release fv-serve is
  447 s; with a seed, 187–206 s.
- Cargo registry without a volume: about 700 crates from the crates.io
  CDN, estimated 20–60 s per fresh pod (not measured). If that matters, a
  `cargo-home/<lockhash>.tar.zst` in R2 is the next step.

**Cost** (R2 list prices: $0.015/GB-month, Class A $4.50/M, Class B $0.36/M,
no egress; free tier 10 GB, 1 M A, 10 M B a month):

| | per month |
|---|---|
| sccache, ~10–30 GB of live objects | $0.15–0.45 (partly in the free tier) |
| deps seeds, 4 × 0.9 GB | $0.05 |
| operations: ~50 cold fills + ~300 warm pods a month | ~$1 Class A + ~$0.75 Class B, inside the free tier in practice |
| **R2 total** | **< $1/month**, against **$14/month per regional 200 GB volume** |

GHCR "cache images" were considered and rejected. The pod has no Docker
daemon to build them, a 1.6 GB base image is pulled already, and layer
granularity is too coarse for per-crate caching.

## 8. Clients

- `fv-control.sh build-pod up [--region R] [--no-wait]` waits until the pod
  is ready. It writes the pod id and token into `~/.config/fv-build/`
  (`pod`, `auth-header`, mode 600), the files `build-pod.sh` already reads.
  The other commands are `build-pod down|stop [--force] [<id>]`, `build-pod
  delete [--force] <id>`, `build-pod status`, `build-pod list`, `build-pod
  policy`, and `build-pod token <id>` (writes the auth files).
- `build-pod.sh up|stop|down` call `fv-control.sh build-pod …`. `plan` and
  `volume-create` are gone. `status` adds fv-control's view. `run`, `sync`,
  `fetch`, `seed`, `clean`, `evict`, `log`, `cancel` and
  `release-artifacts` are unchanged. The Runpod key is no longer read.
  `FV_BUILD_URL` (a local test server) still works.
- PR #32's `build-pod.sh runner` stays as a manual fallback.
  `up`'s automatic runner step is replaced by fv-control's.

## 9. What the owner provides

1. **Worker secret `GITHUB_RUNNER_PAT`** (staging, later production):
   a fine-grained PAT, resource owner `zaitrarrio`, repository
   `fastvideo-rs` only, with Administration: **Read and write** and
   Actions: **Read** (Metadata: read is implied).
   `wrangler secret put GITHUB_RUNNER_PAT --env staging`.
2. **R2 cache** (optional, for pods outside EU-RO-1 to hit the caches):
   a bucket `fv-build-cache` (the bucket can be created from here with the
   Cloudflare API token), an R2 API token with Object Read & Write on that
   bucket only (dashboard → R2 → Manage R2 API tokens; the API token here
   cannot mint R2 keys), then `wrangler secret put
   BUILD_CACHE_R2_ACCESS_KEY_ID` / `BUILD_CACHE_R2_SECRET_ACCESS_KEY`.
   Also a lifecycle rule "expire after 30 days" on the bucket.
3. **CI** (after #32 merges): an fv-control API token with scope `ci` as
   repo secret `FV_CONTROL_CI_TOKEN`. Then `FV_RUNNER_READ_TOKEN` can go.
   Behind Cloudflare Access, also an Access service token for it.
4. **Agents**: their `~/.config/fv/fv-control-token` must be an **admin**
   token: `up` returns the pod token only to admin callers.
5. Runpod: nothing new. The account key the Worker has already creates
   pods.

## 10. Rollout

1. Merge; `fv-control.sh deploy staging` (migration 0004).
2. With `build_pods.enabled = false` (the default), nothing changes: the
   legacy pod keeps the old backstop.
3. The owner sets the secrets and enables the policy. The next
   `build-pod.sh up` (now via fv-control) creates a managed pod; the legacy
   pod stops itself when idle and is left to the backstop. Deleting the
   legacy pod is the owner's call.
4. After #32: the `pick-runner` call (§6), and `wake_on_queue` covers
   anything that still queues.
