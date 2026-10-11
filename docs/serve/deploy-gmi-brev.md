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
[b-create]. **Live prices:** `GET api/instances/alltypesavailable/{org}`
(bearer key; read live 2026-10-10, GET only) lists every type the org may
rent with `type`, `supported_gpus`, `base_price.amount` ($/hr at its
default `location`), `location_prices`, `stoppable` (`true`, or `null` on
the Shadeform types such as `massedcompute_*`, `hyperstack_*`),
`elastic_root_volume`, `supported_storage[]` (`price_per_gb_hr`, e.g. AWS
gp3 $0.000132/GB-hr, or a fixed `size` such as `850GiB` without a price on
Shadeform), `estimated_deploy_time` (`7m0s`), `is_available`, `provider`
and `cloud_cred_id` (the create needs it). fv-control reads it (`brev.ts`
`brev.types`, cached 1 h per isolate) for the planner's $/hr, the stoppable
flag, the storage price of a parked instance and the deploy-time estimate;
`BREV_PRICES` (JSON `{"<instanceType>": usd_per_hr}`) still overrides a
price, and `BREV_INSTANCE_TYPES` stays the allow-list. With neither, the
planner uses the spec's `max_gpu_dph` (the most it may cost). Of 321 GPU
types listed on 2026-10-10, 258 were stoppable (GCP, AWS, Nebius, Crusoe)
and 63 not (Shadeform, Lambda, SF Compute).

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
stopped [b-gpu]; a restart "fails if no capacity in the same
provider/region, data remains inaccessible" [b-mgmt]).

**Keep-on-stop** (`control/src/brev-park.ts`, D1 migration 0008
`brev_instances`):

- **Every Brev workspace fv-control creates is recorded**: id, name / pod
  key, instance type, location, stoppable, disk GB, storage $/GB-hr, the
  launch's trees at their revisions, state (`live`, `restarting`,
  `parked`, `held`, `stop_failed`, `deleting`, `deleted`, `gone`), a
  boot-token hash and the sealed run script.
- **Disk**: `diskStorage` at create = the trees × 1.3 + 40 GB, at least
  200 GiB, in 50s (e.g. H3 250, H3 + LTX-2.5 400). Fixed-disk types record
  their listed size.
- **Every boot fetches its launch.** The startup script (given once at
  create, `vmBuild.lifeCycleScriptAttr`) only writes the boot URL, a
  per-instance token (a header file) and a systemd unit `fv-boot.service`
  that runs at every boot: it fetches the VM's current run script from
  `GET /ingest/v1/brev-boot` (Bearer token; served only while the instance
  is live, sealed at rest, 20/min) and runs it. The run script is the host
  script (§6.3: host reports, env file, `docker pull` / `docker run` of the
  image, heartbeat, deadline watchdog). A restarted VM therefore runs the
  new launch's image, env, deadline and ingest token; a parked one started
  outside fv-control gets 409 and runs nothing.
- **Owner rule (2026-10-11):** "stopping a brev vm should NOT delete it.
  brev VM's should be re-used or manually deleted". fv-control never
  deletes a Brev VM on its own. The only deletes are the owner's:
  `fv-control.sh pod delete <name>` (a standalone pod whose VM is live:
  the explicit delete of that pod), `fv-control.sh brev delete-parked
  <id>`, Delete on the Pods page's parked list, and
  `POST /api/providers/brev/parked/<id>/delete`.
- **Every stop stops the VM** (`PUT api/workspaces/{id}/stop`): user stop,
  deadline backstop, budget stop (`stop_on_floor`), idle stop, scale-down,
  roll / aborted roll, a failed or never-ready boot, a failed or timed-out
  warm restart, the `down`'s orphan sweep, the collector's direct fallback
  when the operations DO fails. The instance is then recorded:
  - **parked** when its weights completed (the pod became ready, or
    reported its tunnel, which the boot does after the weights): the trees
    it holds (union of the launches that completed) with `parked_at`;
  - **held** otherwise, with a note why (`stopped before its boot became
    ready (<pod FAILED phase>): weights may be incomplete`, `restart
    failed: …`, `no keep-on-stop record` for a VM made before #64).
    Partially downloaded trees stay on its disk; the next boot's weights
    loop removes `.partial-*` leftovers and replaces any tree that is not
    complete at the pinned revision (§7).
  A fake-engine pod (no trees) that became ready parks with no trees.
- **A stop Brev refuses** (a type that turned out not stoppable, an API
  error) leaves the VM **running**: state `stop_failed`, a log warning
  with its $/hr, and the critical alert `brev_stop_failed` (its $/hr and
  $/day) each cron pass while it runs. The cron retries the stop; once it
  takes (or Brev shows it stopped) the instance is held. Nothing deletes
  it; the owner may delete it (`brev delete-parked` accepts it).
- **Non-stoppable types are refused** (rule `brev_stoppable_only`): a type
  the live list marks not stoppable (e.g. the Shadeform `massedcompute_*`
  / `hyperstack_*` types) could only be deleted, so the validator, launch,
  start, scale and roll refuse it with the stoppable allowed types from the
  live list, and `createOtherWorker` / the provider's create refuse it
  again. The launch form lists stoppable types only.
- **Reuse before create**: a new launch on Brev claims (atomically) a
  parked or held instance of the requested type before creating a VM: the
  one holding the most requested trees at the pinned revisions first, then
  parked before held, then the newest; otherwise any of the type whose
  disk fits (its complete trees ∪ its last launch's trees ∪ the requested
  ones, by the §7 sizes, + 40 GB headroom ≤ its disk; a tree replaced at
  another revision counts once). It stores the new run script and calls
  `PUT …/start`; the missing trees download onto it. It keeps its first
  name, so its pod key `brev:<name>`; its `cluster_pods` row starts over
  for the new cluster. A held instance is reused too, once
  `BREV_HELD_REUSE_AFTER_S` (default 900) passed since it was held (so the
  re-placement right after a failed boot does not land on the VM that just
  failed; Unpark makes it reusable at once); the operation log names it
  and its note. Only if none fits is a new VM created.
- **Fallbacks**: a start that fails (e.g. no capacity), or a restarted
  instance not running (and not ready) within `BREV_RESTART_TIMEOUT_S`
  (default 900), is stopped again and **held** (reused later, see above),
  and the next candidate or a fresh instance is tried. The former policy
  `brev_park_delete_failed` is **removed** (2026-10-11): a stored value is
  dropped and ignored.
- **Money**: a parked or held instance costs disk GB × the type's highest
  listed $/GB-hr (unknown: $0.000263). The cron books it each minute in
  the ledger (owner `brev:parked`, pods table `PARKED` / `HELD`), and
  `budgetCheck` adds it to the projection (`… parked storage $x/hr
  included`). `/api/providers` shows `parked_dph`. A `stop_failed` VM is
  still billed at its GPU $/hr under its cluster.
- **Thresholds are alerts only** (policies): more than `brev_park_max`
  (default 2) parked or held instances, or one parked longer than
  `brev_park_max_days` (default 7), raises `brev_parked_over_limit`
  (warn) listing each instance with its state, type, age and storage
  $/day, at park time (in the operation log) and every cron pass. Nothing
  is deleted; `0` means any parked instance alerts (it no longer means
  "delete on stop"). Only rows fv-control recorded with fv- names are
  touched; an fv- workspace it did not record is reported (`brev_orphan`),
  never stopped, started or deleted. A parked instance found running
  outside fv-control raises `brev_parked`.
- **Owner**:
  - UI: the Pods page lists parked, held and stop-refused instances with
    their weights, disk, $/day, age, reuse status and note, the policy
    line, an over-threshold banner, Delete (manual) and Unpark (a held
    one). Stop confirmations say a Brev VM is stopped and kept (storage
    $/day), not deleted.
  - CLI: `fv-control.sh brev parked` (the policy line, then one row per
    instance with its reuse status), `brev unpark <id>` (a held one
    reusable at once), `brev delete-parked <id>`.
  - API: `GET /api/providers/brev/parked` (`policy.line`,
    `policy.auto_delete: false`, `over_limit`, per instance `reuse`),
    `POST /api/providers/brev/parked/<id>/delete | unpark`.
- **Planner**: the launch form lists the allowed stoppable instance types,
  each with its live $/hr, "stoppable: stopped and kept on stop (reused by
  the next launch)", the deploy time and how many are parked / held. With
  Hub weights, the validator shows at the instance-type field "warm:
  restarts parked|held <name> (holds …): boot ~N min", "reuse: restarts
  … (none of the trees …; downloads … onto it)" or "cold: downloads …
  (~X GB): boot ~N min". Estimate: the list's deploy time + the missing GB
  at 0.15 GB/s.

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
3. **HTTPS** → a named Cloudflare Tunnel per pod on the owner's zone,
   made and deleted by fv-control (§6.4), once `CF_TUNNEL_API_TOKEN` and
   the host template are set. Without them the fallback is a Cloudflare
   *Quick* Tunnel (`trycloudflare.com`), which Cloudflare documents as "for
   testing and development", 200 in-flight requests, no SSE, no uptime
   guarantee [cf-quick] (WebSockets work).
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
- Brev: `POST api/organizations/<org>/workspaces` with brev-cli's v1 body
  (the type's `cloudCredId`, `diskStorage` sized for the trees,
  `isStoppable` true for a stoppable type) and a startup script
  (`vmBuild.lifeCycleScriptAttr`) that installs `fv-boot.service`. At every
  boot that unit fetches the VM's run script for its current launch from
  `GET /ingest/v1/brev-boot` (§3.4). The run script is the host script:
  - it writes the env file (umask 077) and the boot;
  - it runs `docker run -d --gpus all --network host --env-file … -v
    /home/ubuntu/workspace/weights:/workspace/weights <image>` on the boot;
  - it starts a host watchdog (`docker rm -f` at the deadline,
    `shutdown -h now`).

  It also appends one line per boot (time, hostname, pod) to
  `~/workspace/fv/boots.log`, and the `host` report carries `boots=<lines>`
  (2 after one stop / start). Env values with a line break are refused.
- Brev host reports (`control/src/brev-host.ts`): the startup script logs
  to `~/workspace/fv/host.log` and reports to `/ingest/v1/endpoint` with
  the ingest token before anything else, so a VM whose host part fails is
  no longer silent: `host` (whoami, uid, `sudo -n` ok, `docker`,
  `docker info`, `nvidia-smi -L`, the NVIDIA container toolkit, disk free,
  kernel, OS), `host_failed` (the failing step + the last 20 log lines;
  fails the boot at once), `container_started` (after `docker run`), and a
  detached `heartbeat` every 30 s (container state + its last log line)
  until the boot logs its tunnel. After the tunnel the same watcher polls
  `127.0.0.1:8000/health` every 10 s: `serve_ready` at the first 200,
  `container_exited` (exit code, OOMKilled, the last 60 lines of `docker
  logs`) if the container exits, `serve_timeout` (the last 60 lines) after
  10 min. Both fail the boot at once. The log lines have every env value
  (8+ chars) replaced by `[redacted]` and are kept in
  `~/workspace/fv/container.log`; the container is not removed when it
  exits (`--restart no`, no `--rm`). Docker missing: installed (docker.io
  + the NVIDIA container toolkit) only with `FV_BREV_INSTALL_DOCKER=1` in
  the pod env. The reports show in `provider-logs` (`[report …]` lines),
  the boot timeline (`host script (Brev VM)`, `container start`, `tunnel
  reported`, `fv-serve READY`) and the `up` wait's pool status. Verified
  live 2026-10-10 on g7e.2xlarge: Brev runs the `lifeCycleScriptAttr`
  script as `ubuntu` with passwordless sudo, Docker 29 with the NVIDIA
  runtime, toolkit 1.20.1.
- **Preflight in the container** (`PROVIDER_BOOT`, before the weights and
  the tunnel): `/opt/fastvideo-rs/bin/fv-serve` must exist, the worker
  config is made as `WORKER_BOOT` does it, and `fv-serve --print-config`
  must accept it with the env. Otherwise `failed` carries fv-serve's own
  message. Before `exec fv-serve`, the boot logs the config path and
  `fv-serve --version`.
- **Key store**: a direct worker gets `FV_KEY_STORE=d1`, and fv-serve
  refuses to start without the D1 settings ("auth.key_store = d1 needs the
  D1 settings"). That is what the first live Brev smoke hit: the container
  exited right after its tunnel. Without `FV_PROVIDER_SECRET_ENV`'s
  `FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN` and `FV_D1_DATABASE_ID`, a GMI /
  Brev pod now gets `FV_KEY_STORE=memory`: keys minted on it with its
  admin token live in its memory. The operation log says so (`NOTE:`).
- `patchWorker` (env apply) is refused for these pods (GMI wipes a
  container on update; a Brev startup script runs once): roll, or stop and
  start.

### 6.4 HTTPS: the tunnel

A GMI / Brev pod has no HTTPS proxy, so it runs an outbound Cloudflare
tunnel to `127.0.0.1:8000` (no inbound port, EIP or firewall). There are
two kinds; fv-control picks one per launch from its settings.

**Named tunnels (the default once configured; `src/tunnels.ts`).** With
`FV_TUNNEL_HOST_TEMPLATE` (e.g. `{pod}-fv-pods.maximal.llc`),
`FV_TUNNEL_ZONE` (`maximal.llc`), `CF_ACCOUNT_ID` and the secret
`CF_TUNNEL_API_TOKEN` (Tunnel edit on the account, DNS edit on the zone)
set, fv-control makes, **before the provider's create call**:

1. a row of `pod_tunnels` (migration 0009: name, pod, host, URL, zone,
   tunnel id, DNS record id, state), written first so a crash leaves a
   record to clean up;
2. `POST /accounts/{acct}/cfd_tunnel` `{name, config_src: "cloudflare"}`,
   name `fv-<pod name slug>-<6 hex>`;
3. `PUT …/cfd_tunnel/{id}/configurations` with the ingress
   `[{hostname: <host>, service: "http://127.0.0.1:8000"}, {service:
   "http_status:404"}]` and no `originRequest` (cloudflared's defaults keep
   websockets, SSE and the WebRTC signalling working);
4. `POST /zones/{zone}/dns_records`: CNAME `<host>` →
   `<id>.cfargotunnel.com`, proxied, comment `fv-control <pod id>`
   (the zone id is looked up by name once and cached in D1);
5. `GET …/cfd_tunnel/{id}/token`, passed to the pod as `FV_TUNNEL_TOKEN`
   (GMI: the create call's env; Brev: the run script, sealed in D1) with
   `FV_TUNNEL_URL = https://<host>`. The token is never stored by
   fv-control, logged or returned; `scrub` removes any tunnel token
   (`eyJhIjoi…`) from provider logs and errors.

`{pod}` is a DNS-safe slug of the pod name plus a random 6-hex suffix (≤ 63
characters per label). If any step fails, what was made is deleted and the
launch says why (`no <gpu> in <provider>: named tunnel …`). The operation
log names the URL at create (`named tunnel …: <pod> will answer at
https://<host>`), before the pod is up.

The pod's boot (`TUNNEL_SH` in `PROVIDER_BOOT`, the same pinned
cloudflared) runs `cloudflared tunnel --no-autoupdate run` with the token in
cloudflared's environment (`TUNNEL_TOKEN`, not its argv), unsets
`FV_TUNNEL_TOKEN` before fv-serve starts, waits for `Registered tunnel
connection`, and reports `{phase: "tunnel", url: $FV_TUNNEL_URL}` as
before. For a pod with a named tunnel, `/ingest/v1/endpoint` accepts
**only that exact URL** (no quick-tunnel URL, no other host).

Delete: when the pod is deleted (stop, scale-down, failed boot, the
provider's create failing, the deadline), fv-control deletes the DNS record,
then the tunnel's connections (`DELETE …/connections`: a deleted pod's
connections linger), then the tunnel. A **parked Brev instance keeps its
tunnel and record** (§3.4): its warm restart reuses them (a fresh token
read, no new tunnel); they go when the parked instance is deleted (only by
the owner since 2026-10-11) or Brev no longer lists it. A delete
that fails leaves the row `delete_failed` and the cron retries.

Only ours: fv-control deletes a tunnel only by the id it recorded, after
checking its name is the recorded `fv-…` one, and a DNS record only by its
recorded id, after checking its comment starts with `fv-control` and its
name is the recorded host. The cron (`tunnelSweep`, every minute; the
Cloudflare listings every 10 minutes; `POST /api/tunnels/sweep` or
`fv-control.sh tunnels sweep` now) deletes recorded tunnels whose pod is
gone (no live `cluster_pods` row and no live / parked Brev record, after a
10-minute grace for a create in flight), records the id of a tunnel or
record made just before a crash (found by its recorded name / host) so the
next pass deletes it, and raises `tunnel_orphan` for an `fv-` tunnel or an
`fv-control` record it did not record, **never touching it**.

**One level below the zone.** Cloudflare's Universal SSL certificate
covers `maximal.llc` and `*.maximal.llc` only. A live check (2026-10-11)
showed `x.fv-pods.maximal.llc` failing the edge's TLS handshake while
`x-fv-pods.maximal.llc` works. So the template should put `{pod}` one
level below the zone (`{pod}-fv-pods.maximal.llc`). A deeper template is
allowed but warned about loudly (the Pods page's tunnels card, the launch
form at `provider`, the `up` log, `fv-control.sh providers` on stderr,
`GET /api/providers` `tunnel.warning`): it needs an Advanced Certificate
Manager certificate for `*.fv-pods.maximal.llc`. A template without
`{pod}`, outside the zone or with a bad label is an error that refuses GMI
/ Brev launches until fixed.

**Quick tunnels (the fallback).** Without those settings (or with some of
them missing, which is warned about), `PROVIDER_BOOT` starts
`cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000`, reads
the `https://<x>.trycloudflare.com` URL from its log and reports it;
fv-control accepts only a quick-tunnel URL from such a pod. Runpod pods are
unaffected either way (Runpod's proxy).

Common to both: the report is accepted only for a live pod of the ingest
token's cluster (rate-limited 60/min); fv-control stores the URL as the
pod's, the `up` wait probes `/health` there, edge fronts export it as
`FV_DISPATCH_ENDPOINT`. `phase: "weights"` shows progress; `phase:
"failed"` (a weights, tunnel or cloudflared error) fails the boot at the
next poll and the pod is deleted (with its tunnel).

| | Quick tunnel | Named tunnel |
|---|---|---|
| URL | random `*.trycloudflare.com`, known only after the pod reports it | `https://<slug>-fv-pods.<zone>`, known at create |
| Concurrency | 200 in-flight requests | no tunnel-specific cap (the zone's plan limits apply) |
| SSE (`text/event-stream`) | not supported | supported (proxied; an idle stream needs a byte within the proxy's 100 s read timeout) |
| Websockets (WebRTC signalling) | yes | yes |
| Uptime | "testing only", no guarantee | a normal proxied hostname |
| Upload size | the zone's plan limit (100 MB Free / Pro) | the same |
| Setup | none | the four settings above |
| Cleanup | none (dies with the pod) | DNS + tunnel deleted by fv-control; leaks swept by the cron |

### 6.5 Money guards

| Guard | Runpod | GMI / Brev |
|---|---|---|
| per-pod $/hr cap | create answer's `costPerHr` > `max_gpu_dph` → delete | offer price (GMI products ÷ `GMI_PRICE_DIVISOR`, Brev's live instance-type list or the `BREV_PRICES` override) > `max_gpu_dph` → refused before create; unknown → planned and booked at `max_gpu_dph` |
| balance floor | account balance (API) | **no API**: `GMI_BUDGET_USD` / `BREV_BUDGET_USD` per UTC month: this month's ledger + (running + Brev parked storage + new $/hr) × hours to the deadline must fit, at start, scale-up and extend; unset → refused. The cron warns at 80 % and stops that provider's clusters at 100 % (policy `stop_on_floor`; Brev VMs are stopped and parked / held, never deleted, §3.4) |
| idle storage | none | Brev parked / held instances: kept until the owner deletes them (owner rule 2026-10-11); alert `brev_parked_over_limit` past `brev_park_max` instances or `brev_park_max_days` days, with their age and $/day (§3.4) |
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
   (LFS SHA-256, git blob SHA-1, sizes), `.complete`, then rename; the
   boot then writes `.fv-revision` (the pinned revision) into the tree. A
   tree is skipped only with `.complete` **and** `.fv-revision` equal to
   its pinned revision: a Brev VM restarted from a stop keeps
   `~/workspace/weights` (§3.4), so its trees are not downloaded again.
   A tree at another revision, or a leftover without both marks, is moved
   aside and removed (the VM's own disk, never a shared volume), then
   fetched again; `.partial` leftovers of an interrupted download are
   removed first. The fetcher (apt, venv, `fetch-hub-tree.py`) is set up
   only when a tree is missing. Aux files are checked by SHA-256 and size.
   Any failure reports `failed` and the pod is deleted (a warm restart
   that fails is held instead, §3.4). The loop is `WEIGHTS_SH` in
   `providers.ts`, unit-tested in bash with a stub fetcher.
4. **Approval**: refused unless the launch has
   `weights_download_approved: true` **and** the Worker has
   `FV_HUB_DOWNLOADS_APPROVED=1`. Both default off. No test performs a
   download (the simulated pods never run the boot script).

Rough sizes (`scripts/gcp/weights.sh plan`): H3 ~120 GB, LTX-2.5 ~70 GB,
Wan 5B ~35 GB: 5–15 min per boot at the GPU's rate. On a stoppable Brev
type it is paid once: later launches restart the parked VM (§3.4). A GMI
storage volume would do the same there (§9).

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
      `BREV_INSTANCE_TYPES` (the allow-list; put **stoppable** types first
      for presets with weights, e.g. AWS / GCP / Nebius / Crusoe types,
      not `massedcompute_*` / `hyperstack_*`), `BREV_BUDGET_USD`.
      `BREV_PRICES` is optional now (an override of the live list).
   3. Keep-on-stop (§3.4): every stop keeps the VM (owner rule
      2026-10-11: fv-control never deletes a Brev VM on its own). Set the
      alert thresholds `brev_park_max` (default 2) and
      `brev_park_max_days` (default 7) in the policies. Check parked /
      held instances with `fv-control.sh brev parked` or on the Pods page;
      they bill storage (~$0.8/day for 250 GB on AWS gp3) until you delete
      them. Allow only stoppable types in `BREV_INSTANCE_TYPES` (others
      are refused at launch).
   4. Ask NVIDIA whether the workspaces REST API may be automated and
      whether the API key is a bearer token for it (if not, set
      `BREV_API_TOKEN` to whatever token the CLI stores after
      `brev login --api-key`, UNVERIFIED).
3. **Both**: `PUBLIC_URL` must be set on fv-control (pods report there).
   Optionally `FV_PROVIDER_SECRET_ENV` (JSON of the `FV_CF_*` / `FV_R2_*`
   values) for D1 keys and R2 results on these pods.
4. **Weights**: decide whether Hub downloads at boot are approved
   (`FV_HUB_DOWNLOADS_APPROVED=1`) and for which presets; until then only
   the fake engine runs there.
5. **Named pod tunnels (§6.4)**: `wrangler secret put CF_TUNNEL_API_TOKEN`
   (a token with Cloudflare Tunnel edit on the account and DNS edit on the
   zone; the account token was checked live to have both on `maximal.llc`)
   on fv-control staging; vars `FV_TUNNEL_ZONE = "maximal.llc"` and
   `FV_TUNNEL_HOST_TEMPLATE = "{pod}-fv-pods.maximal.llc"` (one level below
   the zone: Universal SSL covers it; staging's `wrangler.toml` has both)
   and `CF_ACCOUNT_ID` (already set). Until the secret is set, pods use
   quick tunnels. Check with `fv-control.sh providers` (the `tunnels` line;
   a warning on stderr) and `fv-control.sh tunnels`.

### First live smoke test (no weights, about 20 minutes, under $2 each)

```sh
scripts/serve/fv-control.sh providers                       # gmi / brev: on, budget
scripts/serve/fv-control.sh pod launch smoke-gmi --provider gmi --gpu <product-id> \
  --fake --deadline-min 20 --max-dph 4 --wait
scripts/serve/fv-control.sh pod status smoke-gmi | jq .pod.url   # https://fv-pod-smoke-gmi-…-fv-pods.maximal.llc (named; quick: https://….trycloudflare.com)
scripts/serve/fv-control.sh tunnels                        # its tunnel, host and state
curl -s "<url>/health"                                      # state AVAILABLE
scripts/serve/fv-control.sh api GET /api/pods/gmi:<name>/provider-logs | jq .
scripts/serve/fv-control.sh pod delete smoke-gmi            # then check the GMI console: gone
scripts/serve/fv-control.sh tunnels --all                  # its tunnel: deleted (and gone from the Cloudflare dashboard)
# Brev: the same with --provider brev --gpu <instanceType>
```

It confirms auth, template + create (the `path` format of a digest ref),
`command` / `args` / `envs`, apt + curl + cloudflared inside our image, the
tunnel report, health, logs (GMI), delete, and on Brev that the REST paths,
the bearer token and the startup script (root? Docker + toolkit present?)
work as the CLI source suggests.

### Live keep-on-stop test (a stoppable type, e.g. g7e.2xlarge; under $5)

1. `fv-control.sh brev parked` is empty. Launch a pod with a small Hub tree
   (approved, e.g. preset `fastwan21`, ~30 GB; the launch form's outlook
   names the trees and their size). Expect `host` with `boots=1`, the
   weights downloaded, ready.
2. Stop it: the operation log says `parked (stopped, not deleted; … kept
   on its … GB disk, $x/day storage)`; `brev parked` lists it; the Brev console shows
   it STOPPED, not deleted.
3. Launch the same preset and type again. The log says `warm restart of
   parked …`. The `host` report says `boots=2`; `provider-logs` shows
   `run_fetched` for boot 2; the boot logs `tree …: present at <rev>` and
   `weights ready: N present, 0 fetched`; then ready.
4. Clean up: `brev delete-parked <id>`, and check the Brev console.

If step 3's restart fails or times out, the instance is held
(`brev parked` shows the note) and a fresh one is created; the held one is
reused by a later launch (after `BREV_HELD_REUSE_AFTER_S`) or deleted by
hand. fv-control never deletes either on its own (owner rule 2026-10-11).

### Verified vs UNVERIFIED

| | Status |
|---|---|
| GMI endpoints, fields, statuses, auth header (§2) | DOC |
| GMI `price` unit, `valid` meaning, template `path` format, `command`/`args` overriding the image entrypoint, billing in `creating`, inbound needing EIP + firewall (we avoid it) | UNVERIFIED |
| GMI list shape of `/v1/templates` (array assumed) | UNVERIFIED |
| Brev REST paths and fields (§3.1) | SRC (brev-cli), not a documented contract |
| Brev bearer = API key, the instance-type listing and its fields (§3.2) | LIVE (GET, 2026-10-10) |
| Brev v1 create body; the `lifeCycleScriptAttr` script run as `ubuntu` with passwordless sudo, Docker + NVIDIA runtime; a fake-engine pod serving through its tunnel (g7e.2xlarge) | LIVE (coordinator's smoke, 2026-10-10) |
| Brev `PUT …/stop` / `…/start` request shapes and states (STOPPING → STOPPED, STARTING → RUNNING) | SRC (brev-cli `StopWorkspace` / `StartWorkspace`), not called live |
| Whether Brev re-runs the `lifeCycleScriptAttr` script on start | UNVERIFIED: not needed, the systemd unit `fv-boot.service` runs the host part at every boot (and a re-run only `start`s the active unit) |
| Whether the server's stop needs `isStoppable: true` (brev-cli: "used for autostop only"; its `brev stop` checks the type's `stoppable`), and whether `isStoppable: true` enables an idle autostop | UNVERIFIED (sent true for stoppable types; `BREV_IS_STOPPABLE=0` turns it off) |
| `diskStorage` honoured on every provider (fixed-disk Shadeform types list a size) | UNVERIFIED |
| `/home/ubuntu/workspace` kept across stop / start with the same disk; start failing without capacity | DOC [b-gpu] [b-mgmt]; the persistence test is the owner's next live run |
| Storage billed while stopped at the listed $/GB-hr (fv-control books the highest storage class) | DOC + listing; the class Brev picks is UNVERIFIED |
| `shutdown` (the host watchdog) stopping billing | UNVERIFIED |
| Brev GPU list, stop/delete semantics, credits behaviour | DOC |
| Quick-tunnel limits (200 in flight, no SSE, no uptime guarantee) | DOC (Cloudflare) |
| Named tunnels: one subdomain level works through Universal SSL, two levels fail the TLS handshake | LIVE (coordinator, 2026-10-11) |
| The account token has Tunnel + DNS edit on `maximal.llc` | LIVE (coordinator) |
| fv-control's tunnel / DNS calls (create body, ingress, CNAME, token, delete order, `comment.startswith` listing), a pod's cloudflared connecting with the token, SSE through a named tunnel | UNVERIFIED live: simulated only (`harness.mjs` `m.cf`); first live launch with the secret set confirms |
| cloudflared 2026.9.0 linux-amd64 SHA-256 | computed here from the release asset |
| everything in fv-control | simulated only: unit + integration tests (§10) |

## 9. Follow-ups

- Named tunnels (§6.4): a live check of SSE and WebRTC signalling
  through one; Cloudflare Access (service tokens) in front of the pod
  hosts if they should not be public.
- GMI storage (ask GMI how to create what `storages[]` mounts) → a
  weights volume, filled once (add-only, verified, like the EU volume).
- GMI EIP + firewall for WebRTC (ICE) if the Reactor runs there.
- Brev keep-on-stop for scale-down and roll victims (only cluster stops
  park today), and a cross-type warm pool (a parked disk serves only its
  own instance type).
- Brev `location_prices`: the planner uses `base_price` (the type's default
  location); fv-control sends no `location`.

## 10. Tests

- Named tunnels (§6.4): `control/test/unit/tunnels.test.ts` (settings,
  the two-level warning, template errors refusing launches, the host slug
  ≤ 63, create order / rollback / reuse / only-ours delete against a fetch
  mock, `TUNNEL_SH` in bash with a stub cloudflared: the token in its
  environment and not its argv, unset before fv-serve; the quick path; the
  token scrub); the harness's simulated Cloudflare API (`m.cf`: tunnels,
  configurations, tokens, connections, zones, DNS records) and edge
  (`/tunnel/<host>/…` routed DNS → tunnel → ingress → the connected pod);
  integration step "named pod tunnels" (GMI create → token in the
  container's env, URL logged up front, served through the tunnel → only
  its own URL accepted → stop deletes record, connections, tunnel; a
  failed provider create removes them; Brev park keeps them, the warm
  restart reuses them, deleting the parked instance deletes them; leak
  cleanup incl. a crash before the id was recorded; strays alerted, foreign
  ones untouched; no token in any response or the Worker's log; the CLI);
  the GMI step asserts the quick-tunnel fallback; `control/test/ui/tunnels.mjs`
  (the Pods page's tunnels card and the launch form's warning for a
  two-level template).

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
  failed boot deleted at once (GMI) or stopped and held (Brev, owner rule
  2026-10-11); a Brev launch refused by the budget before
  any VM; a Brev launch through `fv-control.sh pod launch --provider brev
  --fake` end to end; and the existing "no secret in any response" check
  with the GMI / Brev keys added.
- `control/test/ui/config-validation.mjs`: the provider select, the
  Runpod fields hidden, an off provider's reason, Launch off, the coverage
  audit.
- Keep-on-stop (§3.4):
  - `control/test/unit/brev-park.test.ts`: `WEIGHTS_SH` run in bash with
    a stub fetcher (fetch missing, skip complete at the pinned revision,
    replace another revision or a leftover, `.partial` cleanup); the
    instance-type parsing; park vs delete (non-stoppable, no weights,
    never ready, policy off); warm candidates and the atomic claim; held /
    unpark / delete-by-policy; the restart timeout; the limits; the budget
    with parked storage; the boot endpoint (live only, sealed run script);
    the warm / cold outlook.
  - The Brev simulator (`harness.mjs`): the instance-type listing,
    `PUT …/stop` / `…/start` (no capacity: `startFail`; hanging:
    `startHang`), a per-workspace disk that keeps trees across stops, and
    VM boots that fetch the run script from `/ingest/v1/brev-boot`.
  - Integration step "Brev keep-on-stop" (a second Worker with Hub
    downloads approved): cold launch → stop parks (not deleted) → ledger,
    pods row, budget and no orphan alert → warm restart (no create, no
    download, the new env) → park again; restart failure → held + fresh;
    restart timeout → held + fresh; the `brev_park_max` threshold an
    alert (nothing deleted); a held instance reused (no create); a stop
    Brev refuses left running with `brev_stop_failed`, then held by the
    cron's retry; a non-stoppable type refused at validate and launch; the
    CLI (`brev parked`, `unpark`, `delete-parked`); the age threshold an
    alert; the only DELETE the owner's; foreign and unrecorded fv-
    workspaces never touched. The step "GMI / Brev standalone pods": a
    failed boot (host failure, container exit) stops and holds the VM;
    `pod delete` of a live VM deletes it.
  - Unit (`brev-park.test.ts`): "owner rule 2026-10-11: no automatic path
    deletes a Brev VM" stubs `brev.remove` and runs stop, deadline,
    scale-down, roll, orphan, failed boot, failed / timed-out restart, no
    record, refused stop, the direct backstop / budget release and the
    cron (limits, refused-stop retry, budget floor): it fails if any of
    them calls it; only `deleteParked` and the manual `pod delete` do.
  - `control/test/ui/brev-parked.mjs`: the Pods page's parked card
    (weights, disk, $/day, held, Unpark, Delete), and the launch form's
    instance types (stoppable first, live price) with the cold outlook.

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
