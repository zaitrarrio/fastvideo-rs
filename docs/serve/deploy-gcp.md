# Google Cloud: fv-serve on Compute Engine GPU VMs

Date: 2026-10-06. This page covers running fv-serve on Google Cloud GPU VMs.
A VM is either a standalone server or a **worker that joins a family Durable
Object queue**, as a Runpod or CloudRift worker does
([dispatch-do-family.md](dispatch-do-family.md)). It also covers the weights,
networking, cost safety, how fv-control would manage the VMs, and what the
owner must provide for a first live test.

**Nothing has run on Google Cloud yet.** The repo has no GCP credentials. The
scripts have been tested offline: dry runs, an offline JWT test, and a mock of
the OAuth and Compute Engine APIs. The configs are checked by fv-serve's own
config tests. Facts come from Google's docs, read on 2026-10-06. Prices
come from the billing catalog as mirrored by gcloud-compute.com, because
Google's pricing pages render client-side. Anything not backed by one of
these is marked **UNVERIFIED**. §10 lists what the first live run settles,
and §11 what the owner provides.

Sources (read 2026-10-06):

| Ref | URL |
|---|---|
| [gpu-zones] | https://docs.cloud.google.com/compute/docs/gpus/gpu-regions-zones |
| [accel] | https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines (G4, A3, G2 machine types, disks, provisioning models) |
| [hdml] | https://docs.cloud.google.com/compute/docs/disks/hd-types/hyperdisk-ml |
| [quota] | https://docs.cloud.google.com/compute/resource-usage (GPU and Hyperdisk ML quotas) |
| [iam] | https://docs.cloud.google.com/compute/docs/access/iam (Compute Engine roles) |
| [disk-price] | https://cloud.google.com/compute/disks-image-pricing (Hyperdisk ML $0.000109589/GiB-hr and $0.000164384 per MiB/s-hr in us-central1, as quoted by the page's search snippet; the page itself renders client-side) |
| [gpu-price] | https://cloud.google.com/compute/gpus-pricing and https://cloud.google.com/compute/vm-instance-pricing (render client-side; not readable here) |
| [mirror] | https://gcloud-compute.com/g4-standard-48.html, `/a3-highgpu-1g.html`, `/g2-standard-8.html` (Cloud Billing catalog mirror, pages updated 2026-10-04) |
| [jwt] | https://developers.google.com/identity/protocols/oauth2/service-account (the JWT-bearer grant that `auth.sh` implements) |
| [maxrun] | https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime (`maxRunDuration` + `instanceTerminationAction`) |

## 1. What is in the repo

| File | What |
|---|---|
| `scripts/gcp/auth.sh` | An OAuth access token from a service-account key, using openssl, curl and jq only (no gcloud): an RS256 JWT exchanged at the key's `token_uri`. `check` mints a token and reads the project; `self-test` signs and verifies a JWT offline. The token is cached per key (mode 600) and passed to curl through a config fd, so it never appears on a command line |
| `scripts/gcp/lib.sh` | REST helpers (`gce`, `gce_wait`), the dry-run mode, labels, the price table, and the family table reader |
| `scripts/gcp/families.tsv` | One row per family (`h3-turbo`, `h3-max`, `ltx`, `wan`, `fake`): config, Runpod twin, dispatch family, verify cells, weight trees, default machine, and API names for the e2e run. The scripts and a Rust test read it |
| `scripts/gcp/vm.sh` | `preflight`, `up` (standalone), `worker` (family DO worker), `wait`, `down`, `reap`, `list`, `ssh-free-logs`, `plan`, `secrets-push`, `gc` |
| `scripts/gcp/startup.sh` | The VM startup script. It checks the driver and NVENC, installs Docker and the NVIDIA toolkit, mounts the weight disk read-only, runs `verify-weights.sh` on the host, then runs the pinned image on the host network with the config and env from metadata. It also starts optional Caddy TLS, the idle watchdog, and the serial-console log. Two more roles: `populate` (weights from the Hub) and `quantize` (FP8 text encoders) |
| `scripts/gcp/weights.sh` | `plan`, `bucket`, `populate` (needs `FV_GCP_WEIGHTS_APPROVED=1`), `quantize`, `image`, `disk-up`/`disk-down` (Hyperdisk ML), `work-down` |
| `scripts/gcp/e2e.sh` | The GPU end-to-end run: one standalone VM per family and the API matrix (fal, MiniMax, LTX, OpenAI videos, native, live, NVENC vs x264), with a budget cap and delete-on-exit |
| `scripts/gcp/selftest.sh` | Offline: `bash -n` + shellcheck, the JWT self-test, the mock test, every command in dry-run mode with sentinel secrets (none may leak), and the guards |
| `scripts/gcp/tests/{gcp_mock.py,vm.test.sh}` | A mock of the token endpoint and the Compute Engine API. 38 checks of `vm.sh` create, worker, delete and reap (§9) |
| `configs/serve/gcp-{h3-turbo,h3-max,ltx,wan}.toml` | One family per VM. `[[models]]`, aliases and `[protocols]` are identical to `runpod.toml`, `runpod-h3-max.toml`, `runpod-ltx.toml` and `runpod-wan.toml` (tested). State is in `/fvstate`, WebRTC uses real ports 40010/udp and 40000/tcp, and director chunks are 5 s |

Changes against the 2026-09-28 draft (rescued in the first commit of this
branch):

- `ltx-turbo` and `wan-turbo` became `ltx` and `wan`, named after the Runpod
  pools. The old `gcp-wan-turbo.toml` had `recipe = "wan-turbo"` with the 1.3B
  weights. Since `wan-turbo` now names FastWan2.2 5B (`runpod-wan5b.toml`), that
  would have loaded the 5B model from the 1.3B tree. `gcp-wan.toml` uses
  `recipe = "fastwan21-1.3b"` as `runpod-wan.toml` does.
- The worker role, the default EU region, the label-based reaper, the idle
  stop and optional TLS were added. Weights are verified on the host, because
  the per-variant serve images ship no scripts. Weight revisions come from
  `weights-revisions.tsv`. The VMs now run as their own service account.

## 2. Auth: a service-account key and least-privilege roles

The scripts read the key from `GCP_SA_KEY_JSON` (raw or base64 JSON), from
`GCP_SA_KEY_FILE`, or from `/root/.config/fv/gcp_sa_key.json` (mode 600). The
project comes from `GCP_PROJECT`, else the key's `project_id`. Neither the key
nor the token is ever printed. `bash scripts/gcp/auth.sh check` prints the
project, the account, and the token's remaining lifetime.

There are two service accounts:

| Account | Used by | Roles (predefined, [iam]) |
|---|---|---|
| **`fv-deploy@<project>`** (the key) | `vm.sh`, `weights.sh`, `e2e.sh`, later fv-control | `roles/compute.instanceAdmin.v1` (instances: create, delete, setMetadata, serial port output) · `roles/compute.securityAdmin` (firewall rules) · `roles/compute.storageAdmin` (disks and images: the weight disks) · `roles/iam.serviceAccountUser` **on `fv-vm` only**, needed to create VMs that run as it · `roles/secretmanager.admin` (for `secrets-push`; drop it if the owner creates secrets by hand) · `roles/storage.admin` **on the weights bucket only** (`weights.sh bucket`) |
| **`fv-vm@<project>`** (`FV_GCP_SA_EMAIL`) | the VMs (metadata-server token) | `roles/secretmanager.secretAccessor` on the `fv_*` secrets (Secret Manager mode) · a custom role with `compute.instances.delete` and `compute.instances.get`, with an IAM condition `resource.name.startsWith("projects/<p>/zones/")` and an `fv-` instance name prefix (UNVERIFIED condition syntax), for the idle self-delete · `roles/storage.objectAdmin` on the weights bucket (populate VM only; optional) |

Tighter alternative for `fv-deploy`: one custom role with exactly the calls
the scripts make: `compute.instances.{create,delete,get,list,setMetadata,setLabels,getSerialPortOutput}`,
`compute.disks.{create,delete,get,list,use,useReadOnly,setLabels}`,
`compute.images.{create,get,list,useReadOnly}`, `compute.firewalls.{create,delete,get,list}`,
`compute.networks.{get,updatePolicy}`, `compute.subnetworks.use`,
`compute.subnetworks.useExternalIp`, `compute.instances.setServiceAccount`,
`compute.zoneOperations.get`, `compute.globalOperations.get`,
`compute.regions.get`, `compute.projects.get`, `compute.machineTypes.get`,
plus `iam.serviceAccounts.actAs` on `fv-vm`. This list was derived from the
REST calls in `vm.sh`/`weights.sh` and is UNVERIFIED until the first live run
(a missing one shows up as a 403 naming it).

The VMs never run as the deploy account. A process on a VM can read its
service account's token from the metadata server, and with the host network
that includes the container. So `fv-vm` must not be able to create anything.

## 3. GPU types, regions, quotas and prices

| GPU | Machine type (1 GPU) | vCPU / RAM | Provisioning (1 GPU) | Boot disk | EU zones [gpu-zones] |
|---|---|---|---|---|---|
| **RTX PRO 6000** 96 GB (G4) | `g4-standard-48` (also -96, -192, -384 for 2/4/8 GPUs) | 48 / 180 GB, 1.5 TiB local SSD [accel] | on-demand, Spot, Flex-start | **Hyperdisk Balanced only** [accel] | europe-west4-a/b/c, europe-north1-a/b/c, europe-west1-b/c, europe-west2-b/c, europe-west8-b/c, europe-west10-b (us-central1-b/c/f) |
| **H100** 80 GB (A3 High) | `a3-highgpu-1g` (2g, 4g, 8g) | 26 / 234 GB [mirror] | **Spot or Flex-start only** below 8 GPUs ("You must create instances by using Spot VMs or Flex-start VMs" [accel]) | Hyperdisk Balanced | europe-west1-c, europe-west3-c, europe-west4-b/c, with limited capacity: "contact your account team" [gpu-zones] |
| **L4** 24 GB (G2) | `g2-standard-8` / `-16` / `-24` | 8 / 32, 16 / 64, 24 / 96 GB [accel] | on-demand, Spot | `pd-balanced` (no Hyperdisk Balanced; Hyperdisk ML is fine as a data disk) [accel] | europe-west4-a/b/c, europe-west1-b/c, europe-west2-a/b, europe-west3-a/b, europe-west6-b/c |

**Default zone: `europe-west4-b`.** It is the one zone with all three GPU types,
and it is in Europe, like the EU weights volume (Iceland) and the family DOs
(`weur`). `europe-north1` (Finland) has G4 too, and is cheaper on Spot.

Prices per hour for the **whole VM** (GPU + vCPU + RAM, Linux, no disks),
from [mirror] (Cloud Billing catalog, 2026-10-04):

| Machine | us-central1 on-demand / Spot | europe-west4 | europe-north1 | europe-west1 |
|---|---|---|---|---|
| g4-standard-48 (RTX PRO 6000) | $4.50 / $1.77 | $4.95 / $2.21 | $4.95 / $2.11 | $4.95 / $2.35 |
| a3-highgpu-1g (H100) | ($11.06) / $6.62 | ($14.07) / $7.69 | - | ($12.17) / - |
| g2-standard-8 (L4) | $0.85 / $0.51 | $0.90 / $0.54 | - | $0.94 / $0.55 |

On-demand A3 1g prices are listed but cannot be bought (Spot/Flex only).
For comparison, Runpod's RTX PRO 6000 is $2.09/hr (EUR-IS-1, recorded
2026-09-24). A G4 costs about 2.4× that on-demand, and about the same on
Spot, though Spot VMs can be pre-empted at any time. The G4 also brings 48
vCPUs and 180 GB of RAM. These rows live in `lib.sh` (`GCP_PRICES`).
`vm.sh` refuses a machine with no price row, and anything over
`FV_GCP_MAX_DPH` (default $6/hr).

**Quotas** [quota]. A new project has a global GPU quota ("GPUs (all
regions)") that must be raised, plus a regional quota per GPU family. The docs
name them `GPU_FAMILY:NVIDIA_RTX_PRO_6000`, `GPU_FAMILY:NVIDIA_L4` and
`GPU_FAMILY:NVIDIA_H100`, with older-style names such as
`PREEMPTIBLE_NVIDIA_L4_GPUS` for Spot. Hyperdisk ML has its own quotas,
`HDML-TOTAL-GB` and `HDML-TOTAL-THROUGHPUT`. Which exact metric names
`regions.get` returns is UNVERIFIED, and `vm.sh preflight` accepts both
spellings. Ask for:

| Quota | Region | Value |
|---|---|---|
| GPUs (all regions) | global | 2 |
| RTX PRO 6000 (on-demand and Spot / preemptible) | europe-west4 | 1 each |
| L4 (on-demand and Spot) | europe-west4 | 1 each (smoke tests) |
| Hyperdisk ML capacity / throughput | europe-west4 | 500 GB / 2,000 MB/s |
| Hyperdisk Balanced capacity | europe-west4 | 1,000 GB (boot disks + the work disk) |
| H100 (preemptible) | europe-west4 | optional; capacity is by arrangement |

## 4. Weights

**Plan.** The serving families need about 344 GB (`weights.sh plan`):

| Tree | GB | Source |
|---|---|---|
| `h3-base` | 144.0 | MiniMaxAI/MiniMax-H3 @ 42ed227 |
| `FastH3-4-step-Preview-v1-LoRA` | 6.8 | @ f509e62 |
| `ltx25` | 125.1 | Lightricks/LTX-2.5-Diffusers @ 426936f (gated: accept the terms with the `HF_TOKEN` account) |
| `fastwan21-1.3b` | 29.2 | FastVideo/FastWan2.1-T2V-1.3B-Diffusers @ 25e7ed7 |
| `text_encoder_fp8` (h3-base + ltx25) | 38.9 | derived (`fv-gpucheck quantize-text-encoder`) or copied from EU |
| `auxiliary/` | 0.4 | pinned URLs + SHA-256 |

Revisions come from `scripts/gpu/weights-revisions.tsv`. Extra trees can be
added with `FV_GCP_WEIGHTS_EXTRA` (e.g. `mmaudio-44k-v2 fastwan22-ti2v-5b`).

**Layout on GCP.**

1. `weights.sh populate`: a CPU VM (`c3-standard-8`, $0.42/hr) writes the trees
   onto a 450 GB Hyperdisk Balanced **work disk** from the Hub, then runs
   `verify-weights.sh` for every family cell. It optionally copies the trees to a
   regional GCS bucket. Hub to GCP ingress is free.
2. `weights.sh quantize`: a G4 writes `text_encoder_fp8`. Alternatively, copy
   it from EU (below).
3. `weights.sh image`: a **disk image** of the work disk (family `fv-weights`),
   then `work-down`.
4. Per test campaign, `weights.sh disk-up` creates a **Hyperdisk ML** volume
   from the image in the serving zone, in `READ_ONLY_MANY` mode. Every serve VM
   in that zone attaches it read-only (up to 2,500 VMs for volumes of 512 GiB or
   less [hdml]). It is deleted with `disk-down` when the campaign ends.

Hyperdisk ML facts [hdml]: 4 GiB to 64 TiB. Throughput is 400 MiB/s to
2 TiB/s, defaulting to MAX(24 × GiB, 400) MiB/s. Size can change every 4 hours
and throughput every 6 hours. It is zonal, cannot be a boot disk, has no
multi-writer mode, and cannot be created in read-write-single mode from an
image or snapshot. Our `disk-up` creates it `READ_ONLY_MANY` from the image.

**Cost** (us-central1 list prices [disk-price]; EU rates are UNVERIFIED and
expected to be about 10% higher):

| Item | Price | Our size | Cost |
|---|---|---|---|
| Hyperdisk ML capacity | $0.08/GiB-month | 450 GiB | $0.049/hr |
| Hyperdisk ML throughput | $0.12 per MiB/s-month | **1,200 MiB/s** (set explicitly) | $0.197/hr |
| → Hyperdisk ML while it exists | | | **$0.25/hr ≈ $180/month** |
| the default throughput (24 × 450 = 10,800 MiB/s) | | | $1,332/month: always set it |
| disk image (`fv-weights`) | $0.05/GiB-month | ≤ 450 GiB | ≤ $22.50/month |
| GCS Standard (optional durable copy) | ~$0.02/GiB-month | 321 GiB | ~$6.40/month |
| work disk while populating | $0.08/GiB-month | 450 GiB | $0.05/hr |
| populate VM | $0.42/hr | 1-2 h | ~$1 |
| quantize VM (G4) | $4.95/hr | 0.5-1 h | ~$2.50-5 |

1,200 MiB/s loads a 41 GB H3 DiT in about 35 s (UNVERIFIED: the per-VM read
limit of a G4 on Hyperdisk ML has not been measured). The image costs about
$22/month and is the thing to keep; the Hyperdisk ML volume should exist only
while VMs run.

**Filling from the EU Runpod volume instead of the Hub.** The Hub trees come
from the Hub at the same pinned revisions, the method
`docs/ops/runpod-volumes.md` §5.4 uses to rebuild a volume. It is free,
150-260 MB/s per pod, and needs nothing from Runpod. The only tree worth
copying from EU is the derived **`text_encoder_fp8`** (38.9 GB). The copy
would use the 2026-09-27 method (runpod-volumes.md §3):

- a Runpod CPU pod on `jg48s6o1w0` serves the two folders read-only over the
  Runpod HTTPS proxy, with Range support, under a random path (≈ $0.24/hr);
- the populate VM pulls them into `<root>/.text_encoder_fp8.partial-<stamp>`,
  compares every file's SHA-256 with the source's, and renames;
- then `FV_VERIFY_FP8_SHA=1 verify-weights.sh text-fp8` runs.

That took 7 min at about 93 MB/s between Runpod regions. It would replace the
G4 `quantize` hour (about $5 against cents). Runpod egress charges are
UNVERIFIED (none recorded in the repo). This copy is **not implemented**: it
needs a Runpod pod on the EU volume, which is a separate approval. Until then,
use `weights.sh quantize`, or run without the FP8 trees (the loader then
quantizes at load).

**Approval.** Populating downloads about 321 GiB. Per CLAUDE.md, large
downloads need the owner's approval, so `weights.sh populate` refuses to run
without `FV_GCP_WEIGHTS_APPROVED=1`. **Proposal:** approve populate for the four
families in europe-west4 when the first real-model test is wanted. Until then,
no weights are needed: the §11 smoke test uses the fake engine. Suggested rule
for CLAUDE.md, as for CloudRift: GCP weight disks and images are **caches** of
manifest trees. They are add-only, verified with `verify-weights.sh`, and listed
in `scripts/gpu/weights-manifest.tsv` like any tree; nothing new lands only on
GCP, and the EU Runpod volume stays the record of truth.

## 5. Networking, WebRTC, TLS and secrets

- **Host network, public IP.** The VM gets an ephemeral external IP (Premium
  tier, gVNIC). The container runs with `--network host`, so the ICE ports are
  real ports: 40010/udp and 40000/tcp (ICE-TCP). fv-serve advertises
  `FV_PUBLIC_IP` from the metadata server. That is the "plain VM" case in
  `fastvideo-webrtc::ice::resolve_ports`: identity mapping, and UDP works,
  unlike on Runpod pods.
- **Firewall.** One rule per VM (`fv-serve-<vm>`, target tag = the VM name,
  description `fv-owner=fastvideo-rs …`):
  - standalone: tcp 8000 + 40000 and udp 40010, from `FV_GCP_SOURCE_CIDR`
    (default: this machine's public IPs as /32);
  - worker: tcp 80, 443 and 40000 and udp 40010, from `0.0.0.0/0`, because
    its clients are the public.

  `down`, `reap` and `gc` delete the rule with the VM.
- **TLS.** A family-DO worker's `endpoint` (its `FV_PUBLIC_BASE_URL`) is where
  clients send session signalling. The console and fal's director run on
  HTTPS pages, which cannot call plain `http://<ip>:8000` (mixed content). So
  workers default to `FV_GCP_TLS=sslip`: Caddy on 443 (`caddy reverse-proxy
  --from <a-b-c-d>.sslip.io --to 127.0.0.1:8000`) gets a Let's Encrypt
  certificate for the sslip.io name, which resolves to the VM's own IP. Port
  8000 is then closed to the world. This is UNVERIFIED live. Before
  production:
  - pin the Caddy image digest (`FV_GCP_CADDY_IMAGE`; default `caddy:2`);
  - consider a domain of ours instead of sslip.io (Let's Encrypt rate limits
    per registered domain apply to sslip.io as a whole);
  - or use a Cloudflare Tunnel, which needs no open 80/443 but adds a hop.
- **Secrets** (`FV_CF_*`, `FV_D1_DATABASE_ID`, `FV_R2_*`,
  `FV_WEBHOOK_ED25519_KEY`, `FV_URL_SIGNING_KEY`, `FV_INTERNAL_TOKEN`,
  `FV_ADMIN_TOKEN`) are read from the environment or from
  `FV_GCP_SECRETS_FILE` (`KEY=VALUE`, only these names). They are never
  printed, and dry runs redact them. Two transports:
  - `FV_GCP_SECRETS=secret-manager` (**recommended**): `vm.sh secrets-push`
    stores each one as a Secret Manager secret with a lower-case id, the
    Runpod secret spelling. The VM reads the `:access` of the latest version
    with its own token and writes a mode-600 env file. Only the names travel
    in metadata.
  - `metadata` (the default when any secret is set): the values travel as
    `fv-secret-*` instance metadata. `e2e.sh` scrubs them once the container
    is up. Anyone with `compute.instances.get` on the project can read
    metadata, so use this only for tests.

## 6. Running

```bash
export GCP_SA_KEY_FILE=/root/.config/fv/gcp_sa_key.json GCP_ZONE=europe-west4-b
bash scripts/gcp/auth.sh check
bash scripts/gcp/vm.sh preflight                  # quotas, image, machine types, weight disk

# Standalone (one family, a per-run API key; e2e.sh does this per family)
read -r VM IP KEY < <(FV_SERVE_IMAGE=ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:<digest> bash scripts/gcp/vm.sh up h3-turbo)
bash scripts/gcp/vm.sh wait "$VM" "$IP"
bash scripts/gcp/vm.sh down "$VM"

# A worker on the h3 family Durable Object (gateway behind the DO, or gateway-less)
FV_DISPATCH_DO_URL=https://<fv-edge worker> FV_GCP_SECRETS=secret-manager \
  bash scripts/gcp/vm.sh worker h3-turbo ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:<digest>
```

What a worker gets. These are the same variables fv-control's
`workerSystemEnv` and `scripts/serve/edge-gpu-test.sh` give a Runpod worker,
and that the CloudRift worker would need for a DO pool:

| Variable | Value |
|---|---|
| `FV_SERVE_ROLE` | `worker` |
| `FV_INTERNAL_TOKEN` | the cluster's token (secret) |
| `FV_DISPATCH_DO_URL` | the fv-edge Worker (https; `vm.sh` refuses http) |
| `FV_DISPATCH_FAMILIES` | from families.tsv: `h3` (h3-turbo, h3-max), `ltx`, `wan`; override with the variable |
| `FV_DISPATCH_DIRECT_UPLOAD` | `1`: outputs go straight to R2 through DO-minted part URLs; no R2 key is needed on the VM |
| `FV_DISPATCH_CAPACITY` / `FV_DISPATCH_SESSIONS` | 2 / 1 (the arbiter's budget) |
| `FV_JOBS_HEARTBEAT_S` | 10 |
| `FV_WORKER_ID` | `gce-<vm name>` |
| `FV_PUBLIC_BASE_URL` | `https://<a-b-c-d>.sslip.io` (the session `endpoint`) |
| `FV_IMAGE_REF` / `FV_IMAGE_DIGEST` | the pinned image (`worker` refuses a tag) |
| with `FV_GCP_DIRECT=1` | `FV_WORKER_DIRECT=1`, `FV_AUTH_MODE=keys`, `FV_KEY_STORE=d1`, `FV_ADMIN_TOKEN`, and the D1 secrets (docs/control/gateway-less-auth.md); refused without them |

`crates/fastvideo-serve/tests/e2e.rs` (`gcp_configs_mirror_runpod_and_join_family_dos`)
applies exactly this environment to every `gcp-*.toml` and runs fv-serve's
`validate()`, both as a DO worker and as a direct worker. It also checks:

- the models match their Runpod twin;
- the dispatch family is the models' family (`ltx2` → `ltx`).

The configs set no `[gateway] pool`, so the worker opens one socket per
family, never the per-pool socket.

## 7. Cost safety

| Guard | Where | Default |
|---|---|---|
| $/hr cap against the price table, before any call | `vm.sh` (`FV_GCP_MAX_DPH`) | $6/hr: G4 and G2 pass, A3 does not |
| **Deadline**: `scheduling.maxRunDuration` with `instanceTerminationAction=DELETE` [maxrun]. Compute Engine deletes the VM even if this container is gone | every VM (`FV_GCP_CAP_S`) | 5,400 s standalone, 14,400 s worker, 14,400 s populate |
| The same deadline as the `fv-deadline` label | every VM | - |
| **Label-based reaper**: `vm.sh reap` lists every zone's instances with `fv-owner=fastvideo-rs`. It deletes those past `fv-deadline` or stopped (`TERMINATED`/`STOPPED`/`SUSPENDED`), then our orphan firewall rules. Others are never touched; `--dry-run` lists | run by hand, from cron, or by fv-control (§8) | - |
| **Idle stop**: a watchdog on the VM (its own systemd unit) deletes the VM after `FV_GCP_IDLE_S` at 0% GPU, through the Compute API with `fv-vm`'s token. If that is refused, it powers off: the GPU stops billing, and `reap` deletes the stopped VM | `startup.sh` | 1,800 s; 0 = off |
| No restart: `onHostMaintenance=TERMINATE`, `automaticRestart=false`; the container has `--restart no` | every VM | - |
| Ownership: `fv-owner`, `fv-kind`, `fv-run`, `fv-family`, `fv-role` and `fv-deadline` labels. `down`/`reap`/`gc`/`disk-down` refuse resources without `fv-owner=fastvideo-rs` | all scripts | - |
| Budget: projected worst case (each VM's $/hr × cap + Hyperdisk ML hours) before anything is created, and the running estimate before each VM | `e2e.sh` (`FV_GCP_BUDGET_USD`) | $40 |
| Delete-on-exit trap | `e2e.sh`, `weights.sh` helper VMs | - |
| Ledger | `artifacts/gcp/ledger.tsv` | - |
| Account-level budget alert | Cloud Billing budget (owner, console) | **ask the owner to set one**, e.g. $50/month with alerts at 50/90/100%. A budget alerts but does not stop spend |

There is no balance floor like Runpod's $8: GCP bills a card or invoice
afterwards. The billing budget is the closest equivalent.

## 8. fv-control and GCP workers (plan, not done here)

fv-control's cluster engine (`control/src/cluster/*`) is Runpod-only. Main
has no provider seam yet. The CloudRift PR (#15, open as a draft) adds
`pods.provider` (migration `0003_providers.sql`), a CloudRift collector and a
written plan for a `Provider` interface (its docs/ops/cloudrift.md §7). GCP
should come in through that same seam, after #15 merges:

1. **Provider interface** (#15's plan, step 2): `create`, `remove`, `get`,
   `account`, `podUrl`. `control/src/gcp.ts` would implement it with the same
   JWT signing in WebCrypto (`crypto.subtle.importKey("pkcs8", …,
   {name: "RSASSA-PKCS1-v1_5", hash: "SHA-256"})`). The key would be a Worker
   secret `GCP_SA_KEY_JSON`, the token cached in KV for 55 min, and errors
   scrubbed of the key.
2. **Placements.** `PoolSpec.provider = "gcp"`, with `gcp: {zone, machine,
   provisioning}`. `workerCreatePayload` gets a GCP branch that builds the same
   instance payload as `vm.sh instance_payload`: labels, `maxRunDuration` from
   the cluster deadline, the `fv-vm` service account, and metadata (startup
   script, config, `fv-env` = `workerSystemEnv` plus the family DO variables,
   and Secret Manager names). One source of truth: move the payload into a small
   JSON template under `scripts/gcp/` that both read, or port and test it like
   `payloads.test.ts`.
3. **Pools on family DOs.** A GCP pool is always `dispatch = "durable-object"`
   with `family`. The worker dials out, so there is no `FV_POOL_<X>_URLS`, no
   registration in `gw_workers`, and no proxy URL. That is simpler than the
   CloudRift static-pool path. The DO status endpoints (`/families/{f}/status`)
   already give fv-control queue depth and worker counts per family.
4. **Collector.** Each minute, an aggregated list with `labels.fv-owner=…`:
   - write rows to `pods` with `provider = 'gcp'`, and accrue `cost_daily`
     from the price table;
   - run the reaper rule in Workers (past `fv-deadline`, stopped → delete);
   - raise a `gcp_api` alert on errors without disturbing the Runpod half;
   - attribute owners: `gcp:<fv-kind>` for ours, `external:gcp` for others,
     which are never touched.
5. **Restart/roll.** GCP VMs cannot change env in place (metadata can, but the
   container reads it at boot), so `restart` and `roll` re-create them, as for
   CloudRift.
6. **Spend.** The Cloud Billing API gives no live balance. Use the price table
   × hours for the projection, and the billing budget's Pub/Sub notifications
   if a hard stop is wanted (a Pub/Sub push to an fv-control route that runs
   `stop_on_floor`).

Estimate: about 500-700 lines plus tests on top of #15. It is worth doing only
once a GCP pool is wanted beyond tests.

## 9. Tests

| Test | What | Where it runs |
|---|---|---|
| `bash -n`, `shellcheck -x` | every `scripts/gcp/*.sh` and `tests/*.sh` | here (in `selftest.sh`) |
| `auth.sh self-test` | signs a JWT with a throwaway RSA key and verifies the RS256 signature, header, claims and the base64 key form | here |
| `scripts/gcp/tests/vm.test.sh` (38 checks) | against `gcp_mock.py`: no key → nothing called; `auth.sh check`; price cap and unknown family refused before any call; `up` payload (labels, deadline, maxRunDuration + DELETE, Hyperdisk Balanced boot, READ_ONLY weight disk, the `fv-vm` account, the key's hash, secrets as items, verify scripts shipped, the config byte-equal); firewall rule; `worker`: needs the DO URL and internal token, refuses a tag, family env (`ltx`, direct uploads), token only as a secret item, cap/idle/TLS, open ports; direct worker env, and refused without the admin token; secret-manager mode; secrets-file whitelist; `down` refuses a foreign VM; `reap` (dry run, past deadline in another zone, stopped, keeps live and foreign, orphan firewall rules); no token, private-key line or secret in any output or the ledger; every call carried the bearer token; token cache mode 600 | here |
| `scripts/gcp/selftest.sh` | all of the above, plus every command in dry-run mode with sentinel secrets (none may leak), and the approval and cap guards | here |
| `gcp_configs_mirror_runpod_and_join_family_dos` (`crates/fastvideo-serve/tests/e2e.rs`) | §6 | build pod / CI |
| `shipped_configs_parse` (`tests/e2e.rs`) | every `configs/serve/*.toml`, the GCP ones included: parse, validate, the CUDA catalog resolves | build pod / CI |

## 10. UNVERIFIED (the first live run settles these)

1. The exact quota metric names returned by `regions.get`
   (`GPU_FAMILY:…` vs `…_GPUS`), and the default quotas of a new project.
2. The minimal custom-role permission list (§2), and the IAM condition for
   `fv-vm`'s self-delete.
3. That the `ubuntu-accelerator-2404-amd64-with-nvidia-580` image family boots
   on G4 with driver ≥ 580 and ships `libnvidia-encode` (startup installs it when
   missing).
4. Hyperdisk ML read rate on one G4 at 1,200 MiB/s provisioned, and model
   load times.
5. Caddy + sslip.io certificate issuance from a fresh VM (HTTP-01 on port 80).
6. UDP reachability of 40010 through the VPC firewall for WebRTC (expected to
   work: a plain VM with a public IP).
7. EU prices for Hyperdisk ML, images and GCS.
8. Spot G4 availability in europe-west4 / europe-north1.

## 11. First live test: what the owner provides, and a ≤ $3 plan

**Provide:**

- **A GCP project** with billing linked and the **Compute Engine API** and
  **Secret Manager API** enabled.
- **A service-account key** for `fv-deploy@<project>` with the §2 roles, saved
  as `/root/.config/fv/gcp_sa_key.json` (mode 600) or passed in
  `GCP_SA_KEY_JSON`.
- **A second account `fv-vm@<project>`** with no key, and the §2 VM roles
  (for the smoke test, only `roles/secretmanager.secretAccessor`; self-delete
  can wait). Grant `fv-deploy` `roles/iam.serviceAccountUser` on it.
- **Quota in europe-west4:**
  - "GPUs (all regions)" ≥ 1;
  - L4 ≥ 1 (on-demand);
  - RTX PRO 6000 ≥ 1 (on-demand or preemptible);
  - Hyperdisk Balanced ≥ 300 GB.

  No Hyperdisk ML is needed for the smoke test.
- **A billing budget** of, say, $50/month with e-mail alerts.
- No weights, and no approval for a large download: the smoke test uses the
  fake engine.

**Plan (worst case about $2.30, every VM with a hard deadline):**

1. **Read-only ($0).** Run `auth.sh check` and `vm.sh preflight` in
   europe-west4-b. This settles items 1 and 3 in part.
2. **Fake standalone on an L4 (≤ $0.47).**
   `FV_GCP_CAP_S=1800 vm.sh up fake <serve image@digest>` (g2-standard-8,
   $0.90/hr on-demand), then `vm.sh wait`, `/healthz`, one fake job and
   `vm.sh down`. This records boot, driver and Docker+toolkit time and the
   image pull. It settles items 3 and 6 (a WebRTC probe via the Reactor
   check) and checks the idle watchdog's start line.
3. **Fake worker on the staging family DO (≤ $0.47).**
   `FV_GCP_CAP_S=1800 FV_DISPATCH_FAMILIES=h3 FV_DISPATCH_DO_URL=<fv-edge-staging> vm.sh worker fake <digest>`,
   with the staging internal token in `FV_GCP_SECRETS_FILE`. Submit a fake job
   through the staging family route and check that it lands on
   `gce-fv-w-fake-…`. Read `https://<ip>.sslip.io/healthz`, which settles item
   5. Then `vm.sh down`.
4. **G4 on Spot, fake engine (≤ $0.75).**
   `FV_GCP_SPOT=1 FV_GCP_MACHINE=g4-standard-48 FV_GCP_CAP_S=1200 vm.sh up fake <digest>`
   ($2.21/hr Spot in europe-west4, or `GCP_ZONE=europe-north1-b` at $2.11),
   with `FV_GCP_ENCODE_BENCH=1`. This gives the driver/NVENC line on an RTX PRO
   6000 and the encode benchmark, and settles items 3 and 8.
5. **Teardown ($0).** Run `vm.sh reap --dry-run` (expect nothing) and
   `vm.sh list` (expect no instances or firewall rules). Compare the ledger
   with the billing report the next day.

Boot disks (100 GB Hyperdisk Balanced or pd-balanced) for under an hour cost
cents. Step 4 runs only if steps 2-3 pass. A real-model test (h3-turbo on
G4) comes after the weights approval in §4: about $2.2 per VM-hour on Spot
plus $0.25/hr of Hyperdisk ML.
