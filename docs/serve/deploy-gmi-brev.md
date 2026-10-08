# Deploying fv-serve on GMI Cloud and NVIDIA Brev

Status, 2026-10-08: **research + fv-control adapters tested against
simulated APIs only.** No GMI Cloud or Brev account, key or token exists in
this repo or its Worker secrets. Nothing here has been called live; §8 is
what the owner provides and the first live smoke test. Every statement is
tagged:

- **DOC**: stated in the provider's official docs (URL cited);
- **SRC**: read in the provider's own open-source client (brevdev/brev-cli),
  not in its docs: an implementation detail that can change without notice;
- **UNVERIFIED**: our inference, or the docs are silent. A live smoke test
  (§8) must confirm it before it is relied on.

Owner request: "let's add deployment support for GMI Cloud and NVIDIA Brev."
The fv-control side (§6) launches standalone pods (and cluster pools) with
`provider: gmi | brev` next to `runpod`; CloudRift stays as it is (on hold).

## 1. Summary

| | GMI Cloud | NVIDIA Brev |
|---|---|---|
| What we rent | **GPU container** (Kubernetes-backed, from a "template" = image) [g-create] [g-ce-containers]; bare metal also exists [g-bm] | **GPU VM** on an underlying cloud, aggregated by Brev [b-gpu]; "container mode" exists in the console and Launchables [b-containers] [b-launch] |
| Public API | REST, `https://console.gmicloud.ai/api/v1`, OpenAPI-documented [g-intro] | **No documented REST API**; the documented interface is the `brev` CLI [b-cli]. The CLI talks to `https://brevapi.us-west-2-prod.control-plane.brev.dev` (SRC [b-src-config]) |
| Auth | `Authorization: Bearer <API key>`; key made in Organization Settings → API Keys, shown once [g-intro] [g-keys] | Personal API key bound to one org, Read or Read & Write, with an expiry; `brev login --api-key` or `BREV_API_KEY` [b-keys]. Whether the key works as a bearer on the REST API directly is **UNVERIFIED** (the CLI may exchange it) |
| Runs our ghcr image | Yes via a template whose `path` is the image (`POST /v1/templates`) [g-tmpl-create]; `path` format for a ghcr digest ref **UNVERIFIED** | Yes: VM + startup script that runs `docker run --gpus all …` (Docker and the NVIDIA Container Toolkit are preinstalled [b-containers]); `startupScript` / `vmOnlyMode` fields (SRC [b-src-ws]) |
| Env, command | `envs[]`, `command`, `args` on create [g-create] | in the startup script (SRC) |
| GPUs (relevant) | H100, H200, B200, GB200 (GB300 pre-order) on the pricing page [g-price]; product ids like `container.h200.x1` [g-intro]; RTX PRO 6000 not listed (**UNVERIFIED**) | B200, H200, H100, A100 80GB, L40S, RTX PRO Server 6000, RTX 5090, … [b-gputypes] |
| Regions | IDCs from `GET /v1/idcs` (example `us-denver-1`, status `available/full/maintenance`) [g-idcs] | per instance type (`location`, `provider`, SRC [b-src-it]); `brev search --provider` filter [b-search] |
| Prices | "from" $2.60 H100, $3.20 H200, $5.00 B200, $8.00 GB200 per GPU-hour [g-price]; `GET /v1/containers/products` has an integer `price` with **no unit** [g-products] | `brev search --json` shows `$/HR` per type [b-search]; no API doc; "bills per hour based on GPU type" [b-gpu] |
| Persistent storage | Not documented for containers; "reconfiguration … data … permanently LOST" [g-ce-containers]; the container object lists `storages[]` [g-list] but there is no storage API page | `/home/ubuntu/workspace` persists across stop (storage billed), lost on delete [b-gpu] [b-mgmt] |
| Networking | Ports on create (max 5, TCP/UDP) [g-create]; inbound needs an **Elastic IP + firewall** ("without a firewall, the container will not accept any inbound connections even if it has an EIP") [g-ce-containers] [g-eip] [g-fw]. No HTTPS | `brev port-forward` (SSH) [b-conn]; Cloudflare-authenticated tunnels / Secure Links (browser login; not for APIs) [b-conn] [b-launch]; Launchables can "expose a port … to all IP addresses" [b-launch]; `--flex-ports` filter for types with a configurable firewall [b-create]. No API-level port opening documented |
| Balance / billing API | **None** documented (Billing / Credits pages are console only) [g-billing] [g-credits] | **None** documented. "If your organization runs out of credits, Brev may stop stoppable running instances and delete non-stoppable resources" [b-gpu] |
| Labels / tags | **None** in the container schema [g-create] | none (SRC) |
| Logs | `GET /v1/containers/{id}/logs` (text/plain) [g-logs] | none via API (`brev exec`/`shell` over SSH) [b-cli] |
| Stop vs delete | `DELETE /v1/containers/{id}`; pay-as-you-go only can be terminated (prepaid plans run to term) [g-ce-containers]. **Terminate does not release an EIP** [g-ce-containers] | stop (data in `~/workspace` kept, no compute charge) / delete [b-mgmt] |
| Serverless / inference | GMI Inference Engine (serverless + dedicated endpoints) serves models from its own catalog [g-ie]; not a host for our image: not used | none relevant |

**Bottom line.** GMI Cloud is a straight fit for fv-control (documented REST,
containers, env, logs), missing only a balance API, labels and HTTPS.
Brev is VM-shaped and its API is the CLI's private one: fv-control can drive
it (it is what the open-source CLI does), but every Brev call is UNVERIFIED
until a live smoke test, and a change on Brev's side can break it without
notice.

## 2. GMI Cloud

Docs index: <https://docs.gmicloud.ai/llms.txt>.

### 2.1 Auth

Bearer API key in `Authorization` [g-intro]; organization API keys
(`/v1/organizations/.../api-keys`, Organization Settings → API Keys) [g-keys].
A separate key type with "Inference" scope exists for the inference API;
for the Cluster Engine (containers) a key with Cluster Engine scope is needed
(**UNVERIFIED** which scope names exist). Worker secret: **`GMI_API_KEY`**.

### 2.2 Containers

| Call | Endpoint | Notes |
|---|---|---|
| list | `GET /v1/containers` | fields: `id`, `name`, `status` (`unknown creating running terminating stopped error zombie`), `reason`, `product`, `idc`, `templateId`, `createdAt`, `publicIP`/`inboundIP`/`outboundIP` (`ipAddress`, `status`), `eipAddress`, `ports[]`, `storages[]`, `envs[]` [g-list] |
| create | `POST /v1/containers` | required `name` (`^([A-Za-z0-9][A-Za-z0-9_\-. ]*)?[A-Za-z0-9]$`, ≤255), `templateId`, `count`, `product`, `idc`; optional `command`, `args[]`, `envs[{name,value}]`, `ports[{containerPort,port?,protocol}]` (≤5), `sshKeyIdList`; answers `[{id}]` [g-create] |
| get | `GET /v1/containers/{id}` | [g-get] |
| update | `PUT /v1/containers/{id}` | name, templateId, envs, command, args, ports; **restarts and wipes data** [g-update] [g-ce-containers] |
| delete | `DELETE /v1/containers/{id}` | [g-delete] |
| restart | `POST …/restart` | [g-restart] |
| logs | `GET /v1/containers/{id}/logs` → text/plain [g-logs] |
| products | `GET /v1/containers/products?idc=` → `name idc type price valid spec gpuModel productLine` [g-products] |
| templates | `GET/POST /v1/templates` (`name`, `path` = image URI, `credential{username,secret}` for a private registry, `status` default `published`) [g-tmpl-create] |
| IDCs | `GET /v1/idcs` (no auth listed) → `idcId name country countrySubdivision status` [g-idcs] |

Product ids are per account ("contact the sales team to confirm the correct
product ID" [g-intro]); containers are gated by an entitlement ("No
Container Instances … Contact Support") [g-ce-containers]. **The owner must
ask GMI to enable containers and confirm the product ids for the GPUs we want.**

UNVERIFIED: the unit of `price` (fv-control assumes cents per hour and makes
it configurable, `GMI_PRICE_DIVISOR`); whether `valid` means in stock;
whether a GPU count lives in `spec`; whether a template `path` may be a
digest reference (`ghcr.io/…@sha256:…`); whether container env values are
visible to other org members in the console (assume yes, as on Runpod).

### 2.3 Networking

A container gets no inbound traffic without an Elastic IP **and** a
firewall [g-ce-containers]: `POST /v1/elastic-ips` (`idc`, `count`, `name`,
`product`) → `POST /v1/elastic-ips/{id}/associate` (`instanceID`,
`instanceType: container`) → `POST /v1/firewalls` (`inboundRules[{protocol,
portRange{min,max}, remoteIpPrefix}]`) → `POST /v1/firewalls/{id}/associate`
[g-eip] [g-eip-assoc] [g-fw] [g-fw-assoc]. That gives plain HTTP on an IP: no
TLS. fv-control's pods carry bearer keys, so plain HTTP is not acceptable
for clients, and the edge Worker cannot fetch a bare IP either.

**Design (§6.4):** the pod opens an **outbound Cloudflare Tunnel** and
serves on its `https://…` hostname. Outbound connections need no EIP,
firewall or open port. EIP + firewall stay a documented option for WebRTC
(ICE over UDP/TCP needs a public IP), not implemented.

### 2.4 Storage for weights

No volume API is documented for containers, and a reconfigure wipes the
container's data [g-ce-containers]. The container object's `storages[]`
(`id`, `containerPath`) [g-list] suggests attachable storage exists; how to
create it is undocumented (**UNVERIFIED**, ask GMI). So v1 uses
**populate-at-boot from the Hub** (§7), behind the owner's approval.

### 2.5 Billing

Pay-as-you-go or prepaid plans; prepaid containers "cannot be terminated
before their rental period ends" [g-ce-containers]. Credits with an
auto-top-up option are shown in the console [g-credits]; **no API returns
the balance**. fv-control therefore enforces a **configured budget**
(§6.5), the deadline backstop and the per-pod $/hr cap.

## 3. NVIDIA Brev

Docs index: <https://docs.nvidia.com/brev/llms.txt>.

### 3.1 Auth

Personal API keys, one org each, Read or Read & Write, with an expiry,
shown once [b-keys]. CI uses `brev login --api-key "$BREV_API_KEY"` then
`brev refresh` [b-cicd]. fv-control runs in a Cloudflare Worker and cannot
run the CLI, so it calls the REST API the CLI uses (SRC):

| Call (SRC [b-src-ws]) | Endpoint |
|---|---|
| create | `POST api/organizations/{orgId}/workspaces` with `name`, `instanceType`, `vmOnlyMode`, `startupScript`, `portMappings`, `baseImage`, `files`, `launchJupyterOnStart` |
| list | `GET api/organizations/{orgId}/workspaces` |
| get | `GET api/workspaces/{id}` |
| stop / start | `PUT api/workspaces/{id}/stop` / `…/start` |
| delete | `DELETE api/workspaces/{id}` |

Base `BREV_API_URL`, default `https://brevapi.us-west-2-prod.control-plane.brev.dev`
(SRC [b-src-config]). Workspace fields: `id name instanceType dns status
healthStatus tunnel{applications[{port}]}`; statuses `RUNNING STARTING
STOPPING DEPLOYING STOPPED DELETING FAILURE` (SRC [b-src-entity]).

**UNVERIFIED**: that an API key is accepted as `Authorization: Bearer` by
this API (the CLI may exchange it for a token via NGC auth,
`BREV_AUTH_URL=https://api.ngc.nvidia.com` SRC [b-src-config]); whether
`startupScript` runs as root and how long it may run; whether
`instanceType` names are the ones `brev search` prints. Worker secrets:
**`BREV_API_TOKEN`** (the key or a token, see §8) and **`BREV_ORG_ID`**.

### 3.2 Instances, GPUs, prices

`brev search gpu` lists types with `TYPE GPU COUNT VRAM TOTAL $/HR BOOT
FEATURES` (S stoppable, R rebootable, P flex ports) and `--json` [b-search];
`brev create --type a,b,c` is a fallback chain, `--startup-script @file`
[b-create]. The search data comes from `ListPublicInstanceType` on
`https://api.brev.dev` (gRPC/connect, SRC [b-src-it]); its JSON shape is not
documented, so fv-control does not call it: **prices come from a configured
table** (`BREV_PRICES` JSON, `{"<instanceType>": usd_per_hr}`), and with no
entry the planner uses the spec's `max_gpu_dph` (the most it may cost).

GPU types listed in the docs include B200 (192 GB), H200 (141 GB), H100,
A100 80GB, L40S, RTX PRO Server 6000 (96 GB), RTX 5090 [b-gputypes].

### 3.3 Networking

Documented options are SSH port-forward and browser-authenticated tunnels
/ Secure Links ("for direct API access without browser authentication, use
`brev port-forward`") [b-conn] [b-launch]. Neither serves API clients. A
Launchable can expose a port to all IPs [b-launch], and the
`portMappings` create field (SRC) may do the same over the API
(**UNVERIFIED**), plain HTTP again. **Design:** the same outbound
Cloudflare Tunnel as GMI (§6.4).

### 3.4 Storage

`/home/ubuntu/workspace` persists across stop, is deleted with the instance
[b-gpu] [b-mgmt]. Weights: populate-at-boot into
`/home/ubuntu/workspace/weights` (§7); a *stopped* instance keeps them, so
a stop/start cycle avoids a second download (storage is billed while
stopped [b-gpu]). fv-control v1 deletes on stop (simpler, no idle storage
bill); keep-on-stop is a follow-up.

### 3.5 Billing

Hourly by GPU type; stopped instances pay only storage [b-gpu]. Credits
with no documented API; at zero credits Brev "may stop stoppable running
instances and delete non-stoppable resources" [b-gpu]. Same guard as GMI:
configured budget + deadline + $/hr cap.

### 3.6 Launchables

Shareable environment configs (compute, container, ports, setup script),
deployed from a link in the console; **no CLI or API to deploy one** is
documented [b-launch]. Useful for the owner to try fv-serve by hand
(a Launchable with our image, port 8000 exposed); not usable by fv-control.

## 4. What does not change

- Runpod stays the primary provider; its code paths are untouched when
  `provider` is absent or `runpod`.
- CloudRift: no behaviour change (on hold by the owner).
- Volumes: the only weights volume remains Runpod EU `jg48s6o1w0`
  (CLAUDE.md). GMI and Brev never mount it.

## 5. Gaps that block "production" on these providers

1. **No balance APIs** → a configured budget is the only money guard
   besides the deadline; the owner must also set a spend limit / no
   auto-top-up in each provider's console.
2. **No labels** → "ours" is decided by fv-control's own D1 records (pods it
   created) and the `fv-` name prefix; nothing else on the account is touched.
3. **HTTPS** → v1 uses Cloudflare *Quick* Tunnels (`trycloudflare.com`),
   which Cloudflare documents as "for testing and development", 200
   in-flight requests, no SSE, no uptime guarantee [cf-quick]. fv-serve
   uses no SSE (WebSockets work). Production needs a named tunnel per pod
   (Cloudflare API + a DNS zone, §9).
4. **Weights** → a Hub download per pod boot (tens to hundreds of GB) until
   a provider volume exists. Needs the owner's approval per launch (§7).
5. **Brev API is undocumented** → a contract risk; confirm with NVIDIA
   whether a public API exists for org automation.

## 6. fv-control: what is built

All of this runs against simulated GMI / Brev APIs in the tests (§10); none
of it has met the real APIs.

### 6.1 Provider abstraction

| File | What |
|---|---|
| `control/src/gmi.ts` | GMI client: containers (list / create / delete / logs), templates, products, IDC default; named fields only (a container's `envs` are never returned) |
| `control/src/brev.ts` | Brev client: workspaces (list / create with `vmOnlyMode` + `startupScript` / delete) under `BREV_ORG_ID`; `BREV_PRICES` |
| `control/src/providers.ts` | the `ComputeProvider` interface and its two implementations, pod keys, the pod env and start command, the weights plan, the budget, `providerIssues` (every refusal, path-anchored), the endpoint report |
| `control/src/collector-providers.ts` | the cron's GMI / Brev half: pods table, ledger, orphans, budget stop |
| `control/src/weights-sources.ts` | generated by `gen-configs.mjs` from `weights-manifest.tsv` + `weights-revisions.tsv` (`--check` keeps it in sync) |

```ts
interface ComputeProvider {
  id: "gmi" | "brev"; title: string;
  off(env): string | null;                 // why it is off ("GMI_API_KEY is not set")
  gpus(env): string[];                      // GMI_PRODUCTS / BREV_INSTANCE_TYPES
  offer(env, gpu, region?): Promise<Offer>; // $/hr + stock for the planner (null: unknown)
  list(env): Promise<ProviderInstance[]>;   // fv-pod-* / fv-ctl-* names only
  create(env, req): Promise<{ id }>;
  remove(env, inst): Promise<boolean>;
  logs?(env, inst): Promise<string[]>;
  budget(env): number | null;
}
```

Runpod keeps `runpod.ts` and stays the default: a pool without `provider`
(or `runpod`) takes exactly the old code path. CloudRift is untouched (its
collector and routes are as before; it never launched from fv-control).
A GMI / Brev pod's id is `gmi:<name>` / `brev:<name>`: the name is
fv-control's (`fv-pod-<pod>-<stamp>`, `fv-ctl-<cluster>-<pool>-<stamp>`),
known before the create call, so the env carries the pod's own id
(`FV_WORKER_ID`, `FV_LOG_SHIP_POD`). `deletePod`, the `up` boot watch, the
`down` verify, `workerHealth` / `workerInternal` and the cron's direct
delete dispatch on the prefix; the provider's own id is looked up by name.
The Runpod collector ignores prefixed ids, and the Runpod balance floor
stops only clusters with a Runpod pod.

### 6.2 Config (zod, `/api/schemas`)

| Field | Where | Rule |
|---|---|---|
| `provider` | pool, standalone launch | `runpod` (default) \| `gmi` \| `brev` |
| `provider_gpu` | pool, launch | required on gmi / brev; one of `GMI_PRODUCTS` / `BREV_INSTANCE_TYPES` (live check) |
| `provider_region` | pool, launch | GMI IDC (default `GMI_DEFAULT_IDC`); refused on brev |
| `weights_source` | pool, launch | `volume` (Runpod only) \| `hub` \| `none` (default on gmi / brev); a pool with `models` needs `hub` |
| `hub_download_approved` / `weights_download_approved` | pool / launch | required with `hub`, and the Worker's `FV_HUB_DOWNLOADS_APPROVED=1` |

On gmi / brev the Runpod fields (`gpu_types`, `cpu_flavors`, `vcpu`,
`volume`, `regions` / `region` / `dc`, `compute: CPU`) are refused at their
paths; the `cpu` image (fake engine) may run on a provider GPU as a smoke
test. `/api/schemas/dynamic` gains `providers` (an off one says why),
`provider_gpus` and `provider_regions`; the standalone form and the cluster
page's pool card offer them and hide the Runpod fields.

### 6.3 Launch

Same image (the variant image resolved to a digest; the image preflight
runs), same env layers (system < account < cluster < pool < pod), same
fv-serve config handling, same auth (`keys` direct, `trust-edge` behind the
edge), same log shipping. What differs (`providerEnv`, `PROVIDER_BOOT`):

- `{{ RUNPOD_SECRET_… }}` references are replaced from the Worker secret
  `FV_PROVIDER_SECRET_ENV` (JSON `{KEY: value}`) or left out with a
  warning in the operation log (the worker then has no D1 keys / R2).
- The Runpod key (`FV_BACKSTOP_API_KEY`) and `FV_MIN_BALANCE` are removed.
  A GMI pod gets the GMI key as its own `FV_BACKSTOP_API_KEY` (its
  watchdog deletes its container at the deadline); a Brev VM gets none.
- Added: `FV_PROVIDER`, `FV_POD_ID`, `FV_POD_NAME`, `FV_WORKER_ID`,
  `FV_LOG_SHIP_POD`, `FV_WEIGHTS=/workspace/weights`, `FV_WEIGHTS_SOURCE`,
  `FV_CLUSTER_DEADLINE`, `FV_ENDPOINT_REPORT_URL`
  (`$PUBLIC_URL/ingest/v1/endpoint`), `FV_ENDPOINT_REPORT_TOKEN` (the
  cluster's ingest token; masked like the other system secrets); with
  `hub`: `FV_WEIGHTS_TREES_B64`, `FV_SCRIPTS_URL`. All are reserved keys.
- GMI: template `fv-img-<sha256(image)[:12]>` (created once, reused), then
  `POST /v1/containers` with `envs`, `command: "bash"`,
  `args: ["-c", PROVIDER_BOOT]`, `ports: [{containerPort: 8000, protocol: TCP}]`.
- Brev: `POST api/organizations/<org>/workspaces` with `vmOnlyMode: true`
  and a startup script that writes the env file (umask 077) and the boot,
  runs `docker run -d --gpus all --network host --env-file … -v
  /home/ubuntu/workspace/weights:/workspace/weights <image>` on the boot,
  and starts a host watchdog (`docker rm -f` + `shutdown -h now` at the
  deadline). Env values with a line break are refused.
- `patchWorker` (env apply) is refused for these pods (GMI wipes a
  container on update; a Brev startup script runs once): roll, or stop and
  start.

### 6.4 HTTPS: the tunnel

`PROVIDER_BOOT` downloads cloudflared 2026.9.0 (linux-amd64, SHA-256
`53b7a7a5…ac6134`, checked before it runs), starts
`cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000`, reads the
`https://<x>.trycloudflare.com` URL, exports it as `FV_DISPATCH_ENDPOINT`
for edge fronts, and `POST`s `{pod, phase: "tunnel", url}` to
`/ingest/v1/endpoint` with the ingest token. fv-control accepts only a
quick-tunnel URL, for a live pod of that token's cluster (rate-limited
60/min), stores it as the pod's URL, and the `up` wait probes `/health`
there. `phase: "weights"` shows progress; `phase: "failed"` (a weights,
tunnel or cloudflared error) fails the boot at the next poll and the pod is
deleted. No inbound port, EIP or firewall is needed.

### 6.5 Money guards

| Guard | Runpod | GMI / Brev |
|---|---|---|
| per-pod $/hr cap | create answer's `costPerHr` > `max_gpu_dph` → delete | offer price (GMI products ÷ `GMI_PRICE_DIVISOR`, Brev `BREV_PRICES`) > `max_gpu_dph` → refused before create; unknown → planned and booked at `max_gpu_dph` |
| balance floor | account balance (API) | **no API**: `GMI_BUDGET_USD` / `BREV_BUDGET_USD` per UTC month: this month's ledger + (running + new $/hr) × hours to the deadline must fit, at start, scale-up and extend; unset → refused. The cron warns at 80 % and stops that provider's clusters at 100 % (policy `stop_on_floor`) |
| deadline | cron + pod watchdog | cron (`stopCluster`) + the pod's watchdog (GMI: API delete; Brev: host `shutdown`) |
| ours only | listed by id | fv- names **and** recorded in `cluster_pods`; an fv-named instance fv-control did not record raises `<p>_orphan` and is never touched; anything else is not even listed |

## 7. Weights on GMI / Brev

No provider volume yet (§2.4, §3.4), so `weights_source: hub`:

1. `poolTrees`: the `${FV_WEIGHTS}/<tree>` paths of the pool's worker
   config (inline, or the image's baked config) plus its preset's
   `weights`. Each must be a plain Hub tree in `weights-sources.ts` (one
   repo, fetcher `hub`, 40-hex revision); trees needing several repos or a
   conversion (`h3-ref2va`, `mmaudio-44k-v2`) are refused at their field.
2. `FV_WEIGHTS_TREES_B64`: `tree<TAB>dest<TAB>repo<TAB>revision<TAB>globs`
   rows, plus `aux` rows (pinned URL, SHA-256, size) for `auxiliary/`.
3. The boot installs a venv with `huggingface_hub`, fetches
   `scripts/gpu/fetch-hub-tree.py` at the image's commit (the spec's `sha`,
   else `main`), and runs it per tree: download into a `.partial-<stamp>`
   folder, every file checked against the Hub listing at that revision
   (LFS SHA-256, git blob SHA-1, sizes), `.complete`, then rename. A tree
   with `.complete` is skipped (a Brev VM keeps them under
   `~/workspace/weights`). Aux files are checked by SHA-256 and size. Any
   failure reports `failed` and the pod is deleted.
4. **Approval**: refused unless the launch has
   `weights_download_approved: true` **and** the Worker has
   `FV_HUB_DOWNLOADS_APPROVED=1`. Both default off. No test performs a
   download (the simulated pods never run the boot script).

Rough sizes (`scripts/gcp/weights.sh plan`): H3 ~120 GB, LTX-2.5 ~70 GB,
Wan 5B ~35 GB: 5–15 min per boot at the GPU's rate. A GMI storage volume
or a stopped Brev VM would remove it (§9).

## 8. Owner to-do (nothing works live until these exist)

1. **GMI Cloud**
   1. Create an account / org; ask GMI support to **enable containers**,
      and for the **product ids** of 1× H100 / H200 / B200 (and RTX PRO 6000
      if offered), the IDCs, and the unit of `price` in
      `/v1/containers/products`.
   2. Create an API key with container (Cluster Engine) scope:
      `wrangler secret put GMI_API_KEY` on fv-control staging first.
   3. Vars: `GMI_PRODUCTS` (comma list), `GMI_DEFAULT_IDC`,
      `GMI_BUDGET_USD` (e.g. `20`), `GMI_PRICE_DIVISOR` if `price` is not
      cents.
   4. In the GMI console: no auto top-up, a spend cap if offered.
2. **NVIDIA Brev**
   1. Create an org and a **Read & Write** API key with a short expiry.
   2. `wrangler secret put BREV_API_TOKEN`; vars `BREV_ORG_ID`,
      `BREV_INSTANCE_TYPES` and `BREV_PRICES` (from `brev search gpu
      --json`), `BREV_BUDGET_USD`.
   3. Ask NVIDIA whether the workspaces REST API may be automated and
      whether the API key is a bearer token for it (if not, set
      `BREV_API_TOKEN` to whatever token the CLI stores after
      `brev login --api-key`, UNVERIFIED).
3. **Both**: `PUBLIC_URL` must be set on fv-control (pods report there).
   Optionally `FV_PROVIDER_SECRET_ENV` (JSON of the `FV_CF_*` / `FV_R2_*`
   values) for D1 keys and R2 results on these pods.
4. **Weights**: decide whether Hub downloads at boot are approved
   (`FV_HUB_DOWNLOADS_APPROVED=1`) and for which presets; until then only
   the fake engine runs there.
5. **Production HTTPS**: a Cloudflare API token with Tunnel + DNS edit on
   one zone, for a named tunnel per pod (§9).

### First live smoke test (no weights, about 20 minutes, under $2 each)

```sh
scripts/serve/fv-control.sh providers                       # gmi / brev: on, budget
scripts/serve/fv-control.sh pod launch smoke-gmi --provider gmi --gpu <product-id> \
  --fake --deadline-min 20 --max-dph 4 --wait
scripts/serve/fv-control.sh pod status smoke-gmi | jq .pod.url   # https://….trycloudflare.com
curl -s "<url>/health"                                      # state AVAILABLE
scripts/serve/fv-control.sh api GET /api/pods/gmi:<name>/provider-logs | jq .
scripts/serve/fv-control.sh pod delete smoke-gmi            # then check the GMI console: gone
# Brev: the same with --provider brev --gpu <instanceType>
```

It confirms auth, template + create (the `path` format of a digest ref),
`command` / `args` / `envs`, apt + curl + cloudflared inside our image, the
tunnel report, health, logs (GMI), delete, and on Brev that the REST paths,
the bearer token and the startup script (root? Docker + toolkit present?)
work as the CLI source suggests.

### Verified vs UNVERIFIED

| | Status |
|---|---|
| GMI endpoints, fields, statuses, auth header (§2) | DOC |
| GMI `price` unit, `valid` meaning, template `path` format, `command`/`args` overriding the image entrypoint, billing in `creating`, inbound needing EIP + firewall (we avoid it) | UNVERIFIED |
| GMI list shape of `/v1/templates` (array assumed) | UNVERIFIED |
| Brev REST paths and fields (§3.1) | SRC (brev-cli), not a documented contract |
| Brev bearer = API key, startup script as root, `shutdown` stopping billing | UNVERIFIED |
| Brev GPU list, stop/delete semantics, credits behaviour | DOC |
| Quick-tunnel limits (200 in flight, no SSE, no uptime guarantee) | DOC (Cloudflare) |
| cloudflared 2026.9.0 linux-amd64 SHA-256 | computed here from the release asset |
| everything in fv-control | simulated only: unit + integration tests (§10) |

## 9. Follow-ups

- Named Cloudflare Tunnels per pod (API-created, DNS on our zone) for
  production HTTPS and SSE-safe streaming.
- GMI storage (ask GMI how to create what `storages[]` mounts) → a
  weights volume, filled once (add-only, verified, like the EU volume).
- Brev stop/start instead of delete to keep `~/workspace/weights`.
- GMI EIP + firewall for WebRTC (ICE) if the Reactor runs there.
- Brev prices from `ListPublicInstanceType` once its JSON is confirmed.

## 10. Tests

- `control/test/unit/providers.test.ts`: pod keys; the schema rules (pool
  and launch); off / not allowed / Hub approval refusals; the weights plan
  (pinned 40-hex revisions, aux SHA-256, multi-repo trees refused); the env
  (secret refs, no Runpod key, GMI self-delete key only on GMI); the boot
  (pinned cloudflared, report, fv-serve last); the Brev startup script; the
  endpoint URL rule; budget check and decision; both clients against a
  fetch mock (bearer, named fields only, errors scrubbed, fv- names only).
- `control/test/harness.mjs`: simulated GMI (`/gmi/v1`: containers,
  templates, products, logs) and Brev (`/brev/api`: org workspaces) whose
  creates boot as simulated pods that report a tunnel URL (or a failure).
- `control/test/integration/run.mjs`, step "GMI / Brev standalone pods":
  the path-anchored launch checks; GMI create → tunnel report → health →
  ready → collector (owner, cost, orphan reported, foreign untouched) →
  provider logs (scrubbed) → offers → stop deletes only ours → delete; a
  failed boot deleted at once; a Brev launch refused by the budget before
  any VM; a Brev launch through `fv-control.sh pod launch --provider brev
  --fake` end to end; and the existing "no secret in any response" check
  with the GMI / Brev keys added.
- `control/test/ui/config-validation.mjs`: the provider select, the
  Runpod fields hidden, an off provider's reason, Launch off, the coverage
  audit.

## Sources

GMI Cloud (all retrieved 2026-10-08):
- [g-intro] https://docs.gmicloud.ai/api-reference/introduction
- [g-keys] https://docs.gmicloud.ai/cluster-engine/user-management/api-keys
- [g-create] https://docs.gmicloud.ai/api-reference/containers/create-containers
- [g-list] https://docs.gmicloud.ai/api-reference/containers/list-container-information
- [g-get] https://docs.gmicloud.ai/api-reference/containers/get-container-info-by-id
- [g-update] https://docs.gmicloud.ai/api-reference/containers/update-container
- [g-delete] https://docs.gmicloud.ai/api-reference/containers/delete-container
- [g-restart] https://docs.gmicloud.ai/api-reference/containers/restart-container
- [g-logs] https://docs.gmicloud.ai/api-reference/containers/download-container-logs
- [g-products] https://docs.gmicloud.ai/api-reference/containers/get-container-products
- [g-tmpl-create] https://docs.gmicloud.ai/api-reference/templates/create-template
- [g-idcs] https://docs.gmicloud.ai/api-reference/idcs/list-all-idcs
- [g-eip] https://docs.gmicloud.ai/api-reference/elastic-ips/allocate-elastic-ip-for-organization
- [g-eip-assoc] https://docs.gmicloud.ai/api-reference/elastic-ips/associate-elastic-ip-with-instance
- [g-fw] https://docs.gmicloud.ai/api-reference/firewalls/create-firewall
- [g-fw-assoc] https://docs.gmicloud.ai/api-reference/firewalls/associate-firewall
- [g-bm] https://docs.gmicloud.ai/api-reference/baremetals/create-baremetal-servers
- [g-ce-containers] https://docs.gmicloud.ai/cluster-engine/resources/containers
- [g-billing] https://docs.gmicloud.ai/cluster-engine/user-management/billing
- [g-credits] https://docs.gmicloud.ai/cluster-engine/user-management/credits-coupons
- [g-ie] https://docs.gmicloud.ai/inference-engine/ie-intro
- [g-price] https://www.gmicloud.ai/pricing

NVIDIA Brev (docs retrieved 2026-10-08; source at brevdev/brev-cli `main`):
- [b-cli] https://docs.nvidia.com/brev/cli/cli-overview
- [b-create] https://docs.nvidia.com/brev/cli/instance-creation
- [b-mgmt] https://docs.nvidia.com/brev/cli/instance-management
- [b-conn] https://docs.nvidia.com/brev/cli/connectivity
- [b-search] https://docs.nvidia.com/brev/cli/search-discovery
- [b-keys] https://docs.nvidia.com/brev/guides/api-keys
- [b-cicd] https://docs.nvidia.com/brev/guides/ci-cd
- [b-containers] https://docs.nvidia.com/brev/guides/development-tools/custom-containers
- [b-launch] https://docs.nvidia.com/brev/concepts/launchables
- [b-gpu] https://docs.nvidia.com/brev/concepts/gpu-instances
- [b-gputypes] https://docs.nvidia.com/brev/reference/gpu-types
- [b-src-config] https://github.com/brevdev/brev-cli/blob/main/pkg/config/config.go
- [b-src-ws] https://github.com/brevdev/brev-cli/blob/main/pkg/store/workspace.go
- [b-src-entity] https://github.com/brevdev/brev-cli/blob/main/pkg/entity/entity.go
- [b-src-it] https://github.com/brevdev/brev-cli/blob/main/pkg/store/instancetypes.go

Cloudflare:
- [cf-quick] https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
