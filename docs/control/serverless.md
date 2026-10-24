# fv-control: Runpod serverless endpoints

Status: 2026-10-06, branch `feat/fvc-serverless`. Code: `control/src/serverless/`
(`spec.ts`, `payloads.ts`, `runpod-sls.ts`, `ops.ts`, `routes.ts`, `cancel.ts`, `console.ts`; since
2026-10-07, branch `claude/serverless-presets`, model-first: `presets.ts`,
`serves.ts`, `examples.ts`, with `control/src/presets.ts`, `gpus.ts`,
`volumes.ts`, §1a), migration
`control/migrations/0006_serverless.sql`, the **Serverless** page
(`control/public/serverless.js`, `control/ui/forms/serverless.ts`,
`serves.ts`), the CLI `scripts/serve/fv-control.sh endpoint …`.

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
Missing fields take the preset's defaults (`GET /api/serverless/defaults?preset=`), or for a
custom endpoint the variant's (`?variant=`). §1a says what a preset is and what is checked.

| field | default | |
|---|---|---|
| `name` | | `[a-z][a-z0-9-]{0,30}`; the Runpod endpoint is `fvc-<name>` |
| `mode` | `queue` | `queue` or `lb` (GPU, `scaler_type: REQUEST_COUNT`, the default for `lb`) |
| `preset` | none (custom) | what it serves (§1a): sets `variant`, `compute`, `config` / `config_toml`, disk and timeout; `cpu` is the fake engine |
| `image` | `{channel: stable}` | one of `channel`, `sha`, `ref`; resolved to a digest like cluster images (`<variant>-<channel>`, `:stable` → `:latest` before the first promotion) |
| `variant`, `compute` | the preset's; custom: `cpu` → `CPU` | `cpu` (fake engine) runs on CPU workers; the CUDA variants on GPU |
| `config` / `config_toml` | the preset's; custom: the variant's baked config | a config file the image carries (`FV_CONFIG`) or inline (`FV_WORKER_TOML_B64`), in both modes |
| `env` | | plain extra env; keys fv-control sets (serve mode, image identity, the `{{ RUNPOD_SECRET_fv_* }}` references, …) are refused |
| `gpu_types`, `gpu_count`, `allowed_cuda` | RTX PRO 6000 Server, 1; `allowed_cuda` `[]` | GPU types in priority order; `allowed_cuda` is an exact list for Runpod's host filter. Leave it empty: every GPU endpoint gets `minCudaVersion: "13.0"`, and a list hides the 13.1 / 13.2 hosts (§1b) |
| `cpu_flavors`, `vcpu` | `cpu3c, cpu5c`, 2 | CPU workers |
| `weights_source` | GPU: `volume` (`DEFAULT_SLS_WEIGHTS_SOURCE`, spec.ts); CPU: none | `volume`: the EU weights volume at `/runpod-volume`; `r2`: the R2 weights store, fetched onto the container disk at boot, no volume, any data centre (§1b) |
| `network_volume`, `data_centers` | GPU on `volume`: `bz0ofkfx45` in `EU-RO-1` (H3 + LTX trees only; other presets default to `r2`); on `r2`: none, any DC; CPU: none | only the EU weights volume is accepted (CLAUDE.md); a volume pins its data center. With `r2`, `network_volume` is null or an R2 cache volume the owner listed (§1b, off by default) |
| `workers_min`, `workers_max` | 0, 1 | 0–4, 0–8 |
| `idle_timeout_s`, `flashboot`, `execution_timeout_s` | 5, false, 1800 | |
| `scaler_type`, `scaler_value` | `QUEUE_DELAY`, 4 | |
| `container_disk_gb` | the preset's (20); `r2`: the trees' bytes × 1.15 + 30 GB | 5-200; up to 500 with `r2` (Runpod's serverless maximum is **UNVERIFIED**) |
| `deadline_min`, `deadline_action` | 120, `delete` | the backstop: `delete` (endpoint + template) or `scale0` (workers 0/0) that long after create (or after the last extend); `null` for none |

An update merges a partial spec over the stored one. Image, env, config and
disk patch the **template** (Runpod rolls the workers); scaling, timeouts,
GPU types and data centers patch the **endpoint**; `mode`, `compute`,
`network_volume` and a CPU endpoint's flavors / vCPUs need a new endpoint (409).
Switching `weights_source` changes `network_volume`, so it needs a new endpoint too.

## 1b. Weights: the volume or the R2 weights store (`weights_source`)

The owner decided (2026-10-11) that the R2 bucket `fv-weights` replaces the EU network volume as the weights
store (docs/ops/r2-weights.md). A GPU endpoint takes `weights_source`:

- **`volume`** (the default until the coordinator flips `DEFAULT_SLS_WEIGHTS_SOURCE` in `src/serverless/spec.ts`
  after the benchmarks; one line): the EU volume `bz0ofkfx45` (H3 + LTX trees, since 2026-10-13) mounted at `/runpod-volume`, workers in
  `EU-RO-1` only. A preset whose trees are not on the volume (wan, fastwan21, sfwan, longlive) defaults to `r2`.
- **`r2`**: no network volume, so **no data-centre restriction**. Leave `data_centers` out for any DC, or list
  DCs anywhere. The validator then asks Runpod's stock for every data centre and shows where each GPU type is in
  stock (a warning line, `stock` in `POST /api/serverless/validate`, "Cold start" in the form). Before any worker
  boots, fv-control requires:
  - every tree is published in R2 (`scripts/gpu/weights-r2.tsv`);
  - the store is configured (`WEIGHTS_R2_*`, `PUBLIC_URL`);
  - the container disk holds the trees (bytes × 1.15 + 30 GB; the default is sized from the manifests).

  Passing `weights_source: r2` to a preset brings that placement with it (`GET /api/serverless/defaults?preset=…&weights_source=r2`).

  **The CUDA filter means "at least 13.0", never "exactly 13.0".** The images ship CUDA 13 runtime libraries,
  which run on any driver offering CUDA 13.0 or newer (≥ 580). On 2026-10-12 an `r2` worker landed on an older host
  whose `libcuda` lacked `cuCtxGetDevice_v2`, and fv-serve panicked and crash-looped
  (docs/ops/r2-bench-2026-10-12.md finding 2). Runpod labels each host by its driver's CUDA version. On 2026-10-14
  (GraphQL `lowestPrice` by `allowedCudaVersions`) every Secure RTX PRO 6000 host was labelled `"13.2"`, H100 hosts
  `"13.0"`, and H200 hosts `"13.0"` or `"13.2"`. So the old default `allowed_cuda: ["13.0"]` hid every RTX PRO 6000
  ("no capacity", or workers that never started). REST v1 cannot list the newer hosts either: it refuses `"13.1"`
  and `"13.2"` (400, enum ends at `"13.0"`). Hence:
  - every GPU endpoint gets **`minCudaVersion: "13.0"`** (create and every PATCH; `MIN_CUDA` in `src/enums.ts`).
    Runpod applies it as a minimum, so a `"13.2"` host passes and a 12.x host does not;
  - **`allowed_cuda` defaults to `[]`** (no exact list) for every weights source (`DEFAULT_ALLOWED_CUDA`). It becomes
    the full `IMAGE_CUDA` list (`["13.0", "13.1", "13.2"]`) on its own once Runpod's REST enum (`CUDA_VERSIONS`)
    names every label. A new label goes into `IMAGE_CUDA` and `scripts/gpu/runpod-cuda.sh`, which a unit test
    keeps equal;
  - an update that empties `allowed_cuda` sends `allowedCudaVersions: []` ("not set: any version"). It used to send
    every REST version, which also stops at 13.0 and so hid the same hosts;
  - the boot's driver check (below) still refuses a pre-13 host before it fetches anything.

**The worker's boot** (`SLS_R2_SH` in `src/serverless/weights.ts`, put before fv-serve by `payloads.ts`; the
template's start is `bash -c`):

0. **Driver check** (`DRIVER_CHECK_SH`, `src/driver-check.ts`; every serverless boot with a config and every pod
   boot run it first): the driver version from `nvidia-smi` must be ≥ 580, and the `libcuda.so.1` the loader opens
   must have `cuCtxGetDevice_v2`. Otherwise the boot prints `driver_too_old: …`, reports it (`failed`, so the boot
   row and the outlook name the host), sleeps 20 s and exits: no fetch and no fv-serve panic. A pod's boot
   diagnosis treats the line (or cudarc's `undefined symbol: cu…`) as fatal and places a new pod.
   `FV_SKIP_DRIVER_CHECK=1` skips it.
1. Reports `boot` to `POST /ingest/v1/sls-report`. Every later step is logged with the time of day and the seconds
   since the container started (`[fv-boot] 05:01:02 +13.4s step fetcher: 0.2s (from the image)`) and reported
   as a `step` (`tools`, `fetcher`, `fetch`, `aux`; streaming adds `stream_ready` and `load_overlap`). The boot row keeps them (`steps`, migration 0011) and the
   outlook's `last_boot.breakdown` shows `<step>_s`, so a slow boot says where its time went
   (docs/ops/r2-bench-2026-10-12.md findings 3 and 4).
2. **Cache check.** For every tree in `FV_R2_TREE_REVS` (`dest@rev …`) it looks for `$FV_WEIGHTS/<dest>/.complete`
   and `.fv-revision` = the pinned revision. If all are there it is a **cache hit**: no plan request, no fetch, and
   fv-control does not even have to be reachable.
3. On a **miss**:
   - `python3` and `curl`: the serve images carry both (`serve-cuda-bin` stage); an older image gets them with
     `apt-get` (timeouts 15 s, 3 retries). That apt step is the likely cause of the unexplained ~70 s of finding 3.
   - The fetcher: the image's `/opt/fastvideo-rs/bin/r2-fetch.py` when its SHA-256 is the one fv-control pins
     (`FV_R2_FETCH_SHA256`), else `/ingest/v1/r2-fetch.py` with `--connect-timeout 5 --max-time 30 --retry 4
     --retry-all-errors` (it used `--max-time 60` and stalled 60 s once).
   - `r2-fetch.py` asks `POST /ingest/v1/weights-plan` with the endpoint's weights token. The fetcher is the same
     one pods use: 64 parallel ranged GETs of 128 MiB (two manifest parts each, each part checked on its own),
     a straggler watchdog, and resume (docs/ops/r2-weights.md §4 and §5.4). It keeps complete trees and resumes
     partial files.
   - Below sm_100 (H100, H200: `nvidia-smi --query-gpu=compute_cap`) it skips the `*_mxfp8` pre-quantized DiTs,
     which the engine cannot use there (`FV_R2_SKIP_TREES`).
   - **Streaming** (2026-10-13; docs/ops/r2-weights.md §11). When the image's fv-serve carries the marker
     `fv-streaming-fetch/1`, and the worker has no cache volume, and `FV_R2_STREAM` is not `0`, the fetcher runs
     in the background with `--stream`. The boot goes on as soon as the files fv-serve reads unguarded and the aux
     trees are in place (step `stream_ready`). fv-serve then starts, reniced by 5, with
     `FASTVIDEO_STREAMING_FETCH=1`, and its loader waits on each file's progress map. The fetch's end reports
     `fetch` and `load_overlap`.
   - A fetch that fails after fv-serve started reports `failed` and stops fv-serve (SIGTERM). Runpod restarts the
     container and the fetch resumes in place. Health stays not-ready until loading completes.
4. The aux files (TAE, LPIPS) come **from R2 with the trees** (`auxiliary/tae`, `auxiliary/lpips`; `slsFetchTrees`)
   and are then checked by SHA-256 against their rows. A missing or different one still comes from its pinned
   URL (an add-only cache volume does that, and `WEIGHTS_R2_AUX=0` turns the R2 path off). Then it reports
   `serve` and starts fv-serve with the config, as before.

The trees land in `/workspace/weights` (`FV_WEIGHTS`) on the container disk. A failure exits after 30 s, and
Runpod restarts the container; the fetch resumes.

**Auth: a weights token per endpoint.**

- At create, fv-control mints `fvw_<64 hex>` and puts it in the template env as `FV_WEIGHTS_TOKEN`. D1 keeps only
  its SHA-256 (`serverless_endpoints.weights_token_hash`, migration 0010).
- The token is good for exactly two things:
  - the plan of **that endpoint's** trees, derived from its stored spec (`slsTrees`), never from the request. A
    request naming another endpoint gets 403. An endpoint not on `r2` gets 409.
  - that endpoint's boot reports.
- It never reaches an R2 key. The presigned URLs expire after `WEIGHTS_R2_URL_TTL_S`, and each URL is for one
  object.
- **Rotation**: every template update writes a new token. The previous hash stays valid for 30 min
  (`SLS_TOKEN_GRACE_MS`) while Runpod rolls the workers.
- **Revocation**: a delete (or a failed create) clears both hashes at once.
- A cluster's ingest token still works for its own pods. The two token kinds are told apart by the `fvw_` prefix.
- **Exposure**: the token sits in clear in the Runpod template env, which anyone with the Runpod account can read.
  It is scoped to presigned reads of this endpoint's published trees. A Runpod secret reference per endpoint would
  hide it, at the cost of creating and deleting one Runpod secret per endpoint and rotation (not done).

**Cold-start breakdown.** Each boot is a row in `serverless_boots`: container start, cache `hit` / `miss` (and the
trees fetched), fetch start / end, GB, s, MB/s, and fv-serve start. The fetcher's 15 s progress lines and its
summary arrive through `--report-url`. Every report is also a log line (target `fv.boot`) in the log store under
the worker's id, next to Runpod's container log; the tick's log capture keeps those lines.

`GET /api/serverless/:id` returns:

- `boots`: the last 10;
- `outlook`: the estimate:
  - **start** ~60 s with the image on the host (measured +49 to +62 s; up to 458 s on a host without the image,
    about 10 % of cold starts, docs/serve/gateway.md §8.6);
  - **+ fetch** = the manifests' GB at `R2_DOWNLOAD_GBPS` (0.55, the mean of two H100 workers with the fixed
    fetcher, 2026-10-12 round 2; docs/ops/r2-weights.md §5.4), or at the last measured MB/s once a boot fetched more than 5 GB;
  - **+ load** (volume measurements: wan 70, h3 52, ltx 33 s; from a local disk **UNVERIFIED**);
  - the cache state of the last boot.

The endpoint page shows a "Cold start" card with the boots table. The CLI's `endpoint show` prints `outlook` and
`boots`.

**Caching across cold starts.**

| Situation | What happens |
|---|---|
| Fresh worker, fresh host | Full fetch. Container disk is lost when a worker stops (Runpod docs). |
| Same worker restarted (crash, OOM, a failed boot) | Cache hit, or a resume of the partial files. |
| FlashBoot revival | Cache hit **if** the revival keeps the container disk. Runpod says FlashBoot "retain[s] worker state after spin-down"; whether that includes the container disk is **UNVERIFIED**, and our 2026-10 run showed fv-serve restarting from scratch (docs/serve/images.md §FlashBoot). The cache check makes it free when it does. |
| R2 cache volume (off by default) | Below. |

**Optional: an R2 cache volume per DC (off by default).**

- The owner creates a small network volume in a data centre and lists it in `FV_SLS_R2_CACHE_VOLUMES`
  (`{"<id>": "<DC>"}`). An r2 endpoint may then name it as `network_volume`, which pins its workers to that DC.
- The boot fills `/runpod-volume/fv-r2-cache/` from R2 with `r2-fetch.py --add-only`:
  - each worker stages under its own name;
  - a tree another worker finished first wins;
  - nothing there is ever moved aside or removed;
  - a tree at another revision is an error, not replaced (CLAUDE.md's add-only rule).
- `$FV_WEIGHTS` links to the cache.
- **Trade-off**: storage costs $0.07/GB-month (h3-base 144 GB: ~$10/month per DC). It brings back the one-DC
  restriction (one volume per endpoint here; Runpod allows one volume per DC on an endpoint, not wired). Loads
  then read the network volume again (200-400 MB/s typical per Runpod), which was the slow part of today's cold
  starts.
- Use it only if the benchmarks show R2 fetches slower than volume loads.

**Runpod's cached models (checked 2026-10-11).** The feature takes **one Hugging Face model per endpoint** (public,
gated or private with a token). It has no S3 or R2 source. It mounts at `/runpod-volume/huggingface-cache/hub/`,
schedules workers on hosts that already hold the model, and does not bill the download time
(docs.runpod.io/serverless/endpoints/model-caching). It **does not fit as is**:

- our trees are R2 objects at pinned revisions, several per family (DiT, text encoder, VAE, upscaler), and some
  derived (FP8, fused) trees exist only on the volume.

It could fit with one **private HF repo per preset** bundling exactly that preset's files, read with an HF token.
That means a new large upload (owner approval, CLAUDE.md), a token on the endpoint, and fv-serve or the boot
pointing `FV_WEIGHTS` at the snapshot path. It is worth evaluating for H3 (largest fetch) once the R2 numbers are in.

Examples:

```bash
# r2, any data centre, GPU types in priority order
fv-control.sh endpoint create h3r2 --preset h3-turbo --weights r2 --gpu "NVIDIA H100 80GB HBM3" --gpu "NVIDIA H200"
# r2, pinned to two DCs
fv-control.sh endpoint create wanr2 --preset wan --weights r2 --dc US-TX-3 --dc EU-RO-1
# as a document
fv-control.sh endpoint create - <<'EOF'
{"name": "wanr2", "preset": "wan", "weights_source": "r2", "workers_max": 1, "deadline_min": 60}
EOF
```

## 1c. Weights from the Hub with FlashBoot (`weights_source: hub`)

Owner request 2026-10-14 ("get the weights and use flashboot"). Pods have had `hub` since PR #95
(docs/control/standalone-pods.md); a serverless endpoint takes it the same way:

- **Two-key approval, as for pods.** The spec says `weights_download_approved: true` (the owner approves the
  download at every cold start of this endpoint's workers), **and** fv-control runs with
  `FV_HUB_DOWNLOADS_APPROVED=1`. Without the first the schema refuses the spec at `weights_download_approved`;
  without the second the validator, the image preflight (`POST /api/preflight {endpoint}`) and the create refuse
  it (`slsWeightsIssues`). CLAUDE.md: large downloads need the owner's approval.
- **Placement.** No network volume (`network_volume: null`; a volume is refused), so any data centre, and
  `minCudaVersion: "13.0"` with no exact CUDA list, as with r2 (§1b). The container disk is sized from the trees (bytes × 1.15 + 30 GB,
  `slsHubDiskGb`; up to Runpod's 500 GB, over 200 GB allowed). `GET /api/serverless/defaults?preset=…&weights_source=hub`
  brings that placement.
- **FlashBoot is on** (`flashboot: true`, set by the placement; `false` is refused): a revival keeps the worker, so it
  downloads nothing.
- **What is downloaded** (`slsHubPlan`): the preset's trees and the trees its config names, each from its one pinned
  Hub repo at the revision of `scripts/gpu/weights-revisions.tsv` (`HUB_TREES`). A preset's pre-quantized trees are
  R2-only: the Hub has no source for them, and the engine quantizes at load. A tree with no plain Hub source (a
  conversion such as `longlive-1.3b-safetensors`, several repos) is refused: use r2 or the volume. The aux files
  (TAE, LPIPS) come from their pinned URLs, checked by SHA-256.

**The worker's boot** (`SLS_HUB_SH` in `src/serverless/weights.ts`, before fv-serve; the template's start is
`bash -c`, after the driver check):

1. Reports `boot`, then the **cache check**: every tree at `$FV_WEIGHTS/<dest>` (`/workspace/weights`, the container
   disk) with `.complete` and `.fv-revision` = the pinned revision. All there (a restarted container on the same
   disk): a cache hit, no download.
2. On a miss, the shared Hub fetcher (`HUB_FETCH_TREE_SH`, the one GMI / Brev / Runpod pods use) sets up a venv with
   `huggingface_hub` (step `hub_tools`) and runs `scripts/gpu/fetch-hub-tree.py` at the image's commit (`main` for a
   channel; `FV_SCRIPTS_URL`) once per tree. It checks every file against the Hub listing at that revision (size;
   LFS files by SHA-256, the others by their git blob SHA-1) and renames the tree into place only then. Its `DONE`
   file must say 0.
3. **Recorded hashes.** `FV_HUB_SHA256_B64` carries `scripts/gpu/weights-sha256.tsv`'s rows of the endpoint's trees
   (`HUB_SHA256`, generated by `gen-configs.mjs`). Every recorded file must match the SHA-256 the fetcher computed and
   the recorded size, or the tree is removed (the worker's own disk) and the boot fails before fv-serve. A tree with
   no recorded rows says so in the log (`checked against the Hub listing only`): `wan22-ti2v-5b` today, for
   example, while `fastwan22-ti2v-5b`'s 15 files are recorded.
4. Each tree is a step (`hub_<dest>`). The end reports `weights ready: hub <GB> GB fetched in <s>s (<MB/s> MB/s)`, which
   the boot row keeps as the fetch's GB, s and MB/s, then the `weights` step and `serve`. fv-serve starts.

The weights token is minted as for r2, but only the boot reports use it: the plan route refuses a hub endpoint (409).
The outlook estimates start + download at `HUB_DOWNLOAD_GBPS` (0.2 GB/s, runpod-volumes.md §7) + ~40 s of tool setup
+ load, or the last measured rate.

```bash
fv-control.sh endpoint create wanhub --preset wan --weights hub --approve-hub-download --gpu "NVIDIA H100 80GB HBM3"
```

**Live check (2026-10-14).** Preset `wan` (`fastwan22-ti2v-5b` + `wan22-ti2v-5b`, 58.4 GB), image
`wan5b-stable`, `workers_max` 1, `idle_timeout_s` 5, FlashBoot on, any DC, the template and endpoint built by this
code (`templateCreatePayload` / `endpointCreatePayload`) and posted to Runpod directly. There was no fv-control
deployment, so no boot reports. The Runpod secret references were left out (no R2 or D1 credentials on the worker).
Times come from the Runpod job status and the worker's container log.

| | Worker | Container start → weights ready | Of which | fv-serve load | Job delay / exec |
|---|---|---:|---|---:|---|
| Cold, `info` | H100 80GB (sm_90, driver 580.126, 84.4 GB GPU free at load) | 194 s | tools 49 s; fastwan22 58 s (~415 MB/s), 15/15 files = weights-sha256.tsv; wan22 62 s (Hub listing only); aux 13 s | 93 s | **335.7 s** / 0.39 s |
| Cold again (the first worker had idled out), a generation | H200 (driver 580.178, 149.5 GB free) | 285 s | tools 16 s; fastwan22 103 s; wan22 152 s (~220 MB/s) | 74 s | 360.7 s / **13.2 s** (fastwan22, defaults) |
| 3 min idle, then `info` | the same H200 worker | none: no new boot, no download | — | none (resident) | **1.5 s** / 0.11 s |

- The Hub rate varied about 2× between hosts (≈ 415 vs 220 MB/s), against R2's 0.33-0.77 GB/s
  (docs/ops/r2-bench-2026-10-12.md). Cold from zero was 336-361 s with Hub against R2's 279 s on the same preset.
- **The revival.** The third job was taken after 1.5 s by the worker that had served the second one, with no new
  container start. Its first log line on wake was `job-take failed error=error sending request`, which fits a suspended
  process whose long-poll died. Runpod's worker list showed it `RUNNING` and health `running 1` throughout, so whether
  it was suspended (FlashBoot) or kept up, and how it was billed meanwhile, is **UNVERIFIED**.
- With `idle_timeout_s` 5, a job sent about 15 s after the previous one ended still went to a new worker: a full
  cold start, not a revival. A longer `idle_timeout_s` keeps the worker warm between closely spaced jobs.

## 1a. Model-first: presets, what an endpoint serves, the checks

**The incident (2026-10-07).** An endpoint `ref2va` (lb, variant `h3-max`, no
config) was meant to serve H3 reference-to-video. Its workers ran the h3-max
image's baked config (`runpod-h3-max.toml`: `sol-h3` only, alias
`MiniMax-H3-Max`), so a MiniMax ref2v request naming `MiniMax-H3-Turbo` got
"model `MiniMax-H3-Turbo` is not served here". The spec was image-first;
nothing said what it would serve, the ref2v config exists only inline, and
load balancers refused inline configs.

**Presets.** `preset` takes an id from the one catalog cluster pools use
(`control/src/presets.ts` `POOL_PRESETS`, re-exported by `cluster/spec.ts`;
`serverless/presets.ts` adds `cpu`, the fake engine): `h3-turbo`, `h3-max`,
`h3-ref2v`, `fasth3-trim`, `h3-larry-turbo`, `h3-lightx2v-turbo`, `ltx`, `ltx-pro`, `ltx-a2v`, `ltx-ref2v`, `wan`, `fastwan21`,
`sfwan`, `longlive`, `cpu`. A preset sets the variant, compute and config (a
config file the image carries, left unset when it is the image's baked default
so the image's own entrypoint runs it; otherwise inline), the container disk
(`ltx-a2v`: 60 GB) and the execution timeout (the pool's job timeout). An input
may repeat those fields but not change them (`variant: set by preset
h3-ref2v …`). An update that names another preset drops the old preset's
config; one that names a variant or config without a preset becomes custom
(`preset: null` too). `GET /api/serverless/presets` lists each preset with
what it serves, its weight trees and least GPU memory. Raw `variant` /
`config` / `config_toml` remain the custom path ("Advanced / custom" in the
form), checked the same way.

**Inline configs on load balancers.** The refusal was not a Runpod limit:
`FV_WORKER_TOML_B64` is an env var, and the template's start (`SLS_BOOT`,
`payloads.ts`) decodes it. Runpod's v2 create (the only API that makes a
load balancer) has no entrypoint field, so after create fv-control patches the
v2-made template with that start (`v2TemplateBoot`), as it already did for CPU
queue endpoints; a template update sets it too. A load balancer's config
*file* still rides v2's `args` (`--config <path>` to the image's entrypoint,
fv-boot, or `fv-entry` on images from before it).
With `workers_min: 0` no worker runs before the patch; with a warm worker
Runpod rolls it onto the patched template.

**What it serves** (`serverless/serves.ts`), derived before any worker boots
from the config the spec runs: the preset's, the inline one, a file of the
image or the variant's baked default. The image contents are generated from
`docker/gpucheck.Dockerfile`'s `serve-<variant>` stages
(`control/src/cluster/image-configs.ts`, `node gen-configs.mjs`; the unit
tests fail when stale). It lists the models (recipe, tier, tasks, resident,
weight trees), the API names (config `[aliases]` and the MiniMax names, which
resolve as the MiniMax adapter does: a config alias, else the canonical tier
alias, else the H3 model of that tier), the mounted APIs (fv-serve's
`[protocols]` defaults filled in) and fal apps. Tasks and tiers per recipe
mirror `crates/fastvideo-engine-service/src/cuda/caps.rs`. A spec without a
preset whose variant and config match one shows that preset as **inferred**
(the `ref2va` spec reads as `h3-max`, serving `sol-h3`, no `MiniMax-H3-Turbo`);
otherwise "custom". Stored specs are not migrated: those without `preset` keep
working unchanged. `POST /api/serverless/<id>/serves/check` compares the
derived models with a running worker's `/fv/v1/capabilities` (a recorded
invoke; it refuses when no worker is up rather than start one) and names what
is missing or extra.

**Checks before create and before an update that changes what it serves**
(`servingIssues`; field-level issues in the form and the 400's `issues`; a
scale never runs them, so a scale to 0 always works):

| check | data source |
|---|---|
| a `config` path is a file the variant's image carries (else: which preset sends it inline) | `image-configs.ts`, generated from the Dockerfile; skipped (a warning) for an `image.ref` |
| GPU presets have a network volume; a custom config that loads weights too | the preset, the config's `weights = "${FV_WEIGHTS}/<tree>"` |
| every weight tree is on the volume | a static list per volume (`control/src/volumes.ts`): the EU volume's trees from `docs/ops/runpod-volumes.md` (per-volume table) and `scripts/gpu/weights-manifest.tsv`; a unit test checks every manifest row and every preset's trees are in it. fv-control cannot list a volume without a pod. A new tree is added there with its manifest row |
| each GPU type has the memory the preset needs (`h3-*`, `ltx*`: 80 GB, so no RTX 5090 / 48 GB cards; `wan`: 32; 1.3B models: 24) | `presets.ts` `min_vram_gb` (conservative), `control/src/gpus.ts` (Runpod's `memoryInGb` per GPU type id; a unit test covers every type in the enum) |
| the data centres are the volume's (EU-RO-1 for the EU volume) | the schema (`spec.ts`), as before |
| licence (LongLive: non-commercial) | a warning |

**Test invokes** (`serverless/examples.ts`): the endpoint page's Test invoke
offers ready-made requests per model × task × API, built from what it serves
and shaped for the mode: queue, the native envelope `{"kind": "http", method,
path, body, "wait": true}` (the worker waits on the job it creates:
`crates/fastvideo-deploy/src/runpod/mod.rs` `HttpJob`); lb, `{method, path,
body}`, whose reply is the job's id (the example says where to poll). The
bodies are the ones the e2e scripts proved (`scripts/serve/e2e/ref2v.py`
`t_minimax` and `t_fal_turbo`, `ltx_e2e.py`): e.g. "MiniMax V2 · ref2v · Turbo
(h3-ref2v-turbo) · MiniMax-H3-Turbo, 768P 5 s". A task that needs media (i2v,
ref2v, a2v) gets a URL field the worker must be able to fetch. The JSON stays
editable. Streams (SF-Wan, LongLive) have no job examples (capabilities only).

**Not done.** An `h3-all` preset (one worker, `[engine] swap = true`, FL2VA and
Ref2VA): it needs a new worker config whose three-DiT swap (memory on a 96 GB
card, swap latency) was never measured on a GPU, so it is left for a GPU run.
`h3-ref2v` already swaps its two Ref2VA DiTs.

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
| `POST /api/serverless/validate {spec, id?}` | validate (schema, §1a checks, GPU stock); the spec with defaults filled, `serving` (what it serves, the preset given or inferred, the examples) |
| `GET /api/serverless/defaults?name=&preset=` (or `&variant=`), `GET/PUT /api/serverless/policy` | |
| `GET /api/serverless/presets` | every preset: variant, config, weights, least GPU memory, what it serves |
| `GET /api/serverless/<id>/serves`, `POST …/serves/check` | what it serves and its test invokes; the check against a running worker's capabilities |
| `GET /api/serverless/<id>` | the row, Runpod's view (workers), `/health`, recent jobs and their stats, cost per day, audit |
| `PUT /api/serverless/<id> {spec}` | update (merged over the stored spec) |
| `POST /api/serverless/<id>/scale {workers_min?, workers_max?}`, `POST …/extend {minutes}` | |
| `DELETE /api/serverless/<id>` | 200 deleted, 202 deleting (the tick retries) |
| `POST /api/serverless/<id>/invoke {input?, sync?}` (queue) / `{method, path, body?}` (lb) | a test request; `GET …/jobs`, `GET …/jobs/<job>` polls one |
| `POST /api/serverless/<id>/jobs/<job>/cancel {fv_job?, fv_api?, stop_fv_job?}` | cancel one job: a Runpod job id (any job of the endpoint, also one fv-control did not submit) or an invoke's number (§5a) |
| `GET /api/serverless/<id>/queue` | queued and running now (`/health`) |
| `POST /api/serverless/<id>/purge {confirm, expected?}` | drop every queued job; `confirm` is the endpoint's name, `expected` the count you saw (409 when the queue grew past it) |
| `GET /api/serverless/<id>/logs?worker=` | a worker's log tail (also stored) |
| `DELETE /api/serverless/<id>/console-cache` | forget the console's cached capabilities and schemas (§5b) |
| `/serverless/<endpoint id>/console…` and the API under it | fv-serve's console for the endpoint (§5b) |
| `POST /api/serverless/tick` | the tick now, billing included |

```bash
fv-control.sh endpoint presets                             # what each preset serves, GPU memory, weights
fv-control.sh endpoint create r2v --preset h3-ref2v --mode lb   # model-first: H3 reference-to-video behind a load balancer
fv-control.sh endpoint serves r2v                           # models, API names (MiniMax-H3-Turbo -> h3-ref2v-turbo), examples
fv-control.sh endpoint create fake-test                    # custom: cpu variant, CPU workers, 0..1, delete after 2 h
fv-control.sh endpoint create spec.json                     # or a full spec
fv-control.sh endpoint list | show fake-test
fv-control.sh endpoint invoke fake-test                     # {"kind":"info"}; polls when /runsync returns IN_QUEUE
fv-control.sh endpoint invoke fake-test '{"kind":"http","method":"POST","path":"/fv/v1/jobs","body":{"model":"fake-wan","prompt":"a fox","seed":1},"wait":true}'
fv-control.sh endpoint scale fake-test 0 | extend fake-test 60 | logs fake-test | delete fake-test
fv-control.sh endpoint cancel fake-test <runpod job id | invoke number> [--fv-job ID [--fv-api API]] [--no-fv]
fv-control.sh endpoint purge fake-test [--yes]             # shows queued / running, asks for the name
```

The **Serverless** page (`#/serverless`) lists the endpoints (status, workers,
queue, $/hr, cost today and billed, backstop). It creates one from a form or the spec
JSON (validate first): the preset comes first, and a **Serves** panel shows
what the endpoint will serve, live, as you pick (variant and config sit under
"Advanced / custom"). Per endpoint it shows health, cold start and warm
queue wait, what it serves (preset or inferred, with the check against a
running worker), scale / scale to 0 / backstop +30 min / delete, a test invoke
with a picker of the examples it serves (§1a) and the JSON, workers and
their logs, the spec editor, recent invokes, cost per day and audit. Its
**Queue** card cancels a job (a pasted Runpod job id, or Cancel on an
unfinished invoke) and purges the queue: the dialog shows the queued and
running counts and takes the endpoint's name; the outcome follows the job's
status (and the fv-serve cancel's queue job) until both finish. **Open
console** opens fv-serve's console for the endpoint in a new tab, and
**Refresh console cache** forgets its cached capabilities and schemas (§5b).

## 5a. Cancel and purge

Admin only (a read token gets 403; a session needs the CSRF header), every
input checked by the `serverless-cancel` / `serverless-purge` schemas,
audited as `serverless.cancel` / `serverless.purge`.

**One job.** `POST /v2/<id>/cancel/<job>` (`src/serverless/cancel.ts`):

| the job was | what happens |
|---|---|
| `IN_QUEUE` | Runpod drops it. No worker saw it, so no fv-serve job exists. |
| `IN_PROGRESS` | Runpod lists it on the worker's job-stop long poll; the worker cancels the queue job (`crates/fastvideo-deploy/src/runpod/worker.rs`). An http job waiting on the fv-serve job it created `DELETE`s its `cancel_path` then (`dispatch.rs`), and that fv-serve job stops at its next denoise step. fv-control's invokes get the owning API's route as `cancel_path` whenever they wait (`withCancelPath`: `/fv/v1/jobs/{id}`, `/v1/videos/{id}`, `/video/{id}`, `/v2/video_generation/{id}`). |
| `COMPLETED` / `FAILED` / … | nothing to cancel on Runpod. A fire-and-forget submit (no `wait`) still has its fv-serve job running in the worker. |

**The fv-serve job.** fv-control finds its id in the queue job's output
(the submit reply; with `wait`, `submit`; while waiting, the progress
output's `poll_path`), or takes `fv_job` (+ `fv_api`) for a job submitted
elsewhere. When the job reached a worker, the worker did not stop it itself
(no `cancel_path`, or the queue job already finished) and the output does
not show it finished, fv-control sends the owning API's cancel as one more
queue job, `/run {"kind": "http", "method": "DELETE", "path": "/fv/v1/jobs/<id>"}`,
recorded as an invoke (route `cancel:<api>`) whose status the page polls.
Two limits, said in the reply:

- fv-serve jobs live in the worker's process. With no worker up the job
  ended with it, so nothing is sent (and nothing scales a worker up).
- With several workers up the DELETE can land on another one, which answers
  404. There is no way to address one queue worker; with `workers_max: 1`
  it is always the right one.

fal ids (the cancel route needs the app) and LTX (no cancel route) are not
offered here; `fv_api` is `native`, `openai_videos`, `fastwan` or
`minimax_v2`.

A job id fv-control did not submit is recorded as an invoke (route
`external`), so its status shows on the page and polls like the others.

**Purge.** `POST /v2/<id>/purge-queue` answers `{"removed": N, "status":
"completed"}` and drops queued jobs only; running ones finish (cancel them
one by one). fv-control reads `/health` before (refusing when the queue grew
past `expected`) and after, and polls its own waiting invokes. Load-balancer
endpoints have no queue: both calls answer 409.

Live check (2026-10-07, staging endpoint `h3-max2`, `lp85qdnkl6dqtz`, scaled
to 0): `POST /v2/lp85qdnkl6dqtz/purge-queue` answered
`{"removed":0,"status":"completed"}`, `/health` after it
`inQueue: 0, inProgress: 0`. The 4 stuck jobs were already gone: fv-control's
last health of the endpoint (03:00 UTC, row marked deleted) said `inQueue: 0`.

## 5b. Console

The endpoint page's **Open console** opens fv-serve's own browser console
(docs/serve/console.md: the model pages with their Playground and API tabs,
the Native API page, the status strip) for that endpoint at
`/serverless/<endpoint id>/console`, served by fv-control. A queue endpoint
has no HTTP server the browser can reach, so fv-control answers every call
the pages make under the same prefix and turns the work into Runpod jobs.
Code: `control/src/serverless/console.ts`; tests: `test/unit/console.test.ts`,
the integration step "serverless console", `test/ui/console.mjs`.

**The pages** are the files `crates/fastvideo-serve/console/` holds,
bundled unchanged (`node gen-configs.mjs` writes
`src/serverless/console-assets.ts`; a unit test fails when it is stale).
fv-control only adds three `<meta>` tags to each page's head and moves its
`/console` links under the prefix; the console's `common.js` reads them
(docs/serve/console.md "Embedded console"): `fv-console-base` (the prefix:
the API base and every page link, the server URL fixed, its own request
history), `fv-console-off` (the pages not served, so their links go) and
`fv-console-note` (a banner saying where the requests run). The pages keep
fv-serve's CSP.

**Auth.** Behind fv-control's login like `/api` (Access, the session cookie,
an `fvc_` token; a read token only reads). The console sends no CSRF
token, so a cookie session's POST / PUT / DELETE needs an `Origin` header
naming fv-control (the pages' own fetches send it; the cookie is
`SameSite=Strict` as well). The Runpod key never leaves fv-control. The
cached capabilities say `auth.mode: "none"`, so the console asks for no API
key. A page opened without a session goes to the dashboard's login.

| the console calls | fv-control |
|---|---|
| pages, `/console/assets/*` | the bundled files |
| `GET /fv/v1/capabilities`, `GET /fal/schema`, `GET /fal/schema/<endpoint>` | a cached reply (D1 `settings`, key `slsc:<row>:<image>:<path>`, 30 min). A miss runs one queue job `{"kind":"http","method":"GET","path":…}` and waits up to 25 s for it; past that the page gets 503 "a worker is starting (cold start)" and the job stays pending, so a reload a minute later finds it (never a second job). Opening the home page starts the capabilities and catalog jobs at once. A stale entry is served as is and refreshed in the background only while a worker is up: opening the console never wakes a worker once the cache is filled. **Refresh console cache** on the endpoint page drops it. A new image (another digest) starts a new cache. |
| `GET /fv/v1/status` | synthesised (`crates/fastvideo-serve/src/status.rs` shape): one pool named after the endpoint, `kind: runpod-serverless`, its state from Runpod's `/health` worker counts (idle / ready → `ready`, running → `busy`, initializing → `loading`, none → `scaled_to_zero`; a scaled-down or deleted endpoint is `down`), queued and running jobs, the models and names from the cached capabilities. The model page then warns before a cold start as it does for any scaled-to-zero pool. No job. |
| a submit: `POST /fv/v1/jobs`, `/v1/videos`, `/v2/video_generation`, a fal `POST /<app>/<endpoint>` | one `/run` job `{"kind":"http","method":"POST","path","headers":{"content-type"},"body" (JSON) or "body_b64","wait":true,"timeout_s":<execution_timeout_s>}` plus the API's `cancel_path` (native, OpenAI, MiniMax). The reply comes at once, in the API's shape, with the **Runpod job id as the job's id** (`id`, `request_id`, `task_id`); every later call of the page names that id. With a worker up fv-control waits up to 3 s first, so a request fv-serve refuses (422 …) comes back as fv-serve's own answer. Recorded as an invoke (route `console:<api>`, the input with long strings elided), audited `serverless.console`, refused below the balance floor + margin and on an endpoint that is not `active`. |
| status and results: `GET /fv/v1/jobs/<id>`, `/v1/videos/<id>`, `/v2/query/video_generation?task_id=`, `/<app>/requests/<id>[/status]` | Runpod's `/status` of the queue job (no job is started) and the shared job store: `JOBS_DB`, fv-serve's D1 `jobs` row of the fv-serve job (its id is in the waiting job's progress, `{state, poll_path}`), for the state, progress, queue position and logs while it runs. The finished reply is the worker's own: the waiting job's output is the last status body fv-serve gave (fal: the result body), shown with the id swapped. Without a store row (no `JOBS_DB`, or the worker keeps jobs in memory) the in-flight view comes from Runpod's state alone (queued / running, no progress or logs). A failed or timed-out queue job, or one Runpod no longer has (about 30 min after it ended), shows as failed with Runpod's message. |
| `GET /v1/videos/<id>/content` | the finished video's URL fetched by fv-control (a presigned R2 URL sends no CORS headers to the page's `fetch`) |
| cancel: `DELETE /fv/v1/jobs/<id>`, `/v1/videos/<id>`, `/v2/video_generation/<id>`, `PUT /<app>/requests/<id>/cancel` | §5a's cancel of the queue job. A running native / OpenAI / MiniMax job stops through its `cancel_path`; a fal job's cancel route is a `PUT` under its app (a `cancel_path` is a `DELETE`), so fv-control sends it as one more queue job after Runpod stopped the waiting one (the same one-worker limit as §5a). |
| `POST /storage/upload/initiate`, then `PUT` | uploads go to R2 (the `LOGS` bucket, `console-uploads/<row>/…`, at most 64 MB; swept after a day), through an fv-control upload URL (15 min). The `file_url` the job gets is `<PUBLIC_URL>/serverless-uploads/<signed token>/<name>`: public, the HMAC-signed token (24 h) is the capability, so the worker can fetch it. Nothing lands on one worker's disk. Request bodies themselves are at most 8 MB (a Runpod job input is at most 10 MB): send files as URLs. |
| other API GETs (`/v1/…`, `/v2/…`, `/fv/v1/…`, e.g. MiniMax's `/v1/files/retrieve`) | not faithful from the store: one queue job, waited up to 25 s, its reply as is |
| live pages and routes (`/console/stream`, `live`, `avatar`, a model's `director`, `admin`; `/fv/v1/streams`, `/wma/*`, Reactor's `/schema`, `/fv/v1/admin/*`) | off in v1, with a note: WebRTC / WHIP sessions need a server the browser reaches, and API keys are fv-control's. |

**Why submits wait.** A queue worker runs one job at a time
(`crates/fastvideo-deploy/src/runpod/worker.rs`, concurrency 1), and Runpod
stops a worker that has no job after `idle_timeout_s`. A fire-and-forget
submit (no `wait`) would leave the fv-serve job running on a worker Runpod
counts as idle, so it could be stopped under the job. Waiting keeps the
queue job (and the worker) busy until the fv-serve job ends, gives the final
reply as the job's output, and lets a Runpod cancel reach the fv-serve job.
The cost: while one generation runs, other queue jobs of the console (a
schema not cached yet, a fal cancel) wait for it, or for a second worker
when `workers_max` allows one. The fv-serve id is not needed promptly: the
page uses the Runpod job id.

**Media.** The finished replies carry the worker's own URLs. With the
endpoint's R2 secrets set (`FV_R2_*`, `SECRET_ENV_REFS`; the default), they
are presigned R2 URLs the browser plays. Without them (local artifacts)
the URLs point at the worker itself and do not play.

Not in v1: the result headers fv-serve sets beside a fal result
(`x-fv-tier`, `x-fv-quality`, …) are not in a queue job's output, so the
model page shows no tier or draft pill; MiniMax's file download is the
fallback GET above.

**Load-balancer endpoints.** The same pages, the cached capabilities and
schemas (fetched from the load balancer), the synthesised status (from
fv-control's worker count: polling the worker would keep it awake) and the
R2 uploads; every other call goes on to `https://<id>.api.runpod.ai` with
the Runpod key, its own URLs in JSON replies pointing back to the prefix.

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
