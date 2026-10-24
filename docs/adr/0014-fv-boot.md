# ADR-0014: `fv-boot`, the one boot sequence of a serve container

- Status: Accepted (plan C2, first step: fv-boot in the images and preferred by every boot fv-control sends; the
  shell boots' deletion and "fv-control sends env only" wait for the image promotion, §6)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-boot
- Code: `crates/fastvideo-deploy/src/boot/` (the sequence, typed steps, the seams), `src/bin/fv-boot.rs` (feature
  `boot`); fixture `crates/fastvideo-deploy/fixtures/boot-protocol.json` (shared with
  `control/test/unit/fv-boot-protocol.test.ts`); staging `control/src/fv-boot.ts`; GPU gate
  `scripts/gpu/fv-boot-smoke-pod.sh`
- Plan: C2 (`docs/plan/2026-10-23-execution-plan.md`); review `docs/gaps/2026-10-22-architecture-review.md` §4.5
  ("Three languages for one concern", "Scripts and fv-control disagree"), §6 item 6, §7.4 C2

## Context

The review counted the boot sequence of a serve container (driver check, config materialization, weights, sidecar,
tunnel and reports, watchdog, exec) ten times, in three languages, mostly bash inside TypeScript strings that no test
could run as a whole. Scripts and fv-control disagreed: fv-control replaced the image entrypoint with its own boot,
the scripts let the entrypoint run; the scripts knew only volume weights, fv-control knew volume, r2, hub and none.

### 1. Inventory (the code on `main` at bbadc543)

Ten texts, by where they run. "In container" means the process the serve image runs; "host" means a VM script that
installs Docker and starts the image (it is the provider layer, plan C3, not a container boot).

| # | Boot | Where | Used by |
|---|---|---|---|
| 1 | `deploy/runpod/fv-entry.sh` | in container (image `ENTRYPOINT`) | serverless templates without a boot override, v2 load balancers (`args: --config …`), GCP and CloudRift `docker run`, the scripts' Runpod pods |
| 2 | `WORKER_BOOT` (cluster/payloads.ts) + `bootFor` inserts (`UPSAMPLER_START_SH`, `R2_RUNPOD_SH`, `HUB_RUNPOD_SH`, `UPSAMPLER_WAIT_SH`) | in container (`dockerStartCmd`, `bash -c`) | fv-control cluster workers (direct) |
| 3 | `EDGE_WORKER_BOOT` = 2 + `FV_DISPATCH_ENDPOINT` + `WATCHDOG` | in container | edge fronts |
| 4 | `WATCHDOG_WORKER_BOOT` = 2 + `WATCHDOG` | in container | standalone pods |
| 5 | `PROVIDER_BOOT` (providers.ts: `report`, `fail`, `DRIVER_CHECK_SH`, preflight, `UPSAMPLER_*_SH`, `HUB_FETCH_TREE_SH`, `WEIGHTS_SH` = `R2_FETCH_SH` + `WEIGHTS_LOOP_SH`, `TUNNEL_SH`, deadline loop) | in container (GMI `bash -c`; Brev `docker run --entrypoint bash … /fv-boot.sh`) | GMI and Brev pods |
| 6 | `SLS_BOOT` (serverless/payloads.ts) | in container (`/bin/sh -c`) | serverless endpoints with a config |
| 7 | `SLS_R2_BOOT` / `SLS_HUB_BOOT` = 6 + `SLS_R2_SH` / `SLS_HUB_SH` (serverless/weights.ts) | in container (`bash -c`) | serverless endpoints with weights_source r2 / hub |
| 8 | `BREV_BOOTSTRAP` + `brevHostScript` (brev-park.ts, brev-host.ts) | host (systemd unit, then `docker run` with 5) | Brev VMs |
| 9 | `scripts/gcp/startup.sh` `role_serve` | host (Compute Engine startup script; container: 1) | GCP VMs (never run live) |
| 10 | `scripts/serve/cloudrift-worker.sh` `vm_boot` | host (cloud-init; container: 1) | CloudRift VMs (on hold) |
| – | `crates/fastvideo-deploy/src/vast.rs` `onstart` (and its copies in `scripts/serve/vast.sh`, `vast-serverless.sh`) | in container (sshd is PID 1: `nohup fv-serve`) | Vast instances |
| – | `scripts/gpu/serverless-worker.sh` | deleted in F1 (#139) | – |

What each step did per boot (✓ does it, – does not):

| Step | 1 fv-entry | 2-4 cluster | 5 provider | 6-7 serverless | 8-10 host | Vast |
|---|---|---|---|---|---|---|
| start marker `[fv-boot] start` | – | ✓ | ✓ (with provider, pod) | – | – | – |
| driver check (≥ 580, `cuCtxGetDevice_v2`) | – | ✓ | ✓, reported | ✓, reported (serverless id) | GCP: host check | – |
| `/runpod-volume` → `/workspace/weights` link, `FV_WEIGHTS` default | ✓ | – | – | ✓ (link only) | – | – |
| volume marker `[fv-boot] volume: …` | – | ✓ (fixed `/workspace/weights`) | – | – | – | – |
| config: inline `FV_WORKER_TOML_B64` | – | ✓ → `/fv-worker.toml` | ✓ | ✓ (no gateway edit) | – | – |
| config: named file | args `--config` | `FV_WORKER_CONFIG` copied | `FV_WORKER_CONFIG` copied | `FV_CONFIG` | args | args |
| config: baked `FV_CONFIG` | ✓ (fv-serve reads it) | – | – | ✓ | via 1 | – |
| `[gateway] register = false` | – | ✓ | ✓ | – | – | – |
| `fv-serve --print-config` preflight | – | – | ✓ | – | – | – |
| missing-tree warning (old `weights =` form) | ✓ | – | – | – | GCP: verify cells | – |
| weights volume | ✓ | ✓ | – | ✓ | GCP disk, CloudRift volume | – |
| weights r2 (r2-fetch.py, streaming, aux) | – | ✓ (`R2_RUNPOD_SH`) | ✓ | ✓ (cache check, cache volume) | – | – |
| weights hub (fetch-hub-tree.py, SHA-256) | – | ✓ (`HUB_RUNPOD_SH`) | ✓ | ✓ | GCP populate role | – |
| upsampler sidecar start / wait | ✓ | ✓ | ✓, reported | ✓, reported | – | – |
| tunnel (cloudflared) + report | – | – | ✓ | – | CloudRift: own front | – |
| reports to fv-control | – | – (no URL on Runpod pods) | pod protocol | serverless protocol | Brev host: own phases | – |
| watchdog | – | 3, 4: deadline + balance, pod deletes itself | deadline (GMI delete, pkill) | – | GCP: idle watchdog | – |
| `FV_WORKER_ID` | – | `RUNPOD_POD_ID` | `FV_POD_ID` | – | GCP / CloudRift: in env | – |
| fail behaviour | `set -e` exit | r2/hub: sleep 30, exit 1 | report, sleep 600, exit 1 | report, sleep 30, exit 1 | – | – |
| exec | `fv-serve "$@"` | `--config /fv-worker.toml` | same | `--config $cfg` | – | `nohup fv-serve` |

Where they disagreed: the driver check ran on fv-control's boots only (the image entrypoint and Vast let a pre-CUDA-13
host panic); only the provider boot checked the config before downloading weights; the volume marker named a fixed
path; the serverless boot reported its driver failure under a second timestamp; hub reports were retried five times
on GMI / Brev and once on Runpod; r2's aux rows were handled three ways (pod, provider loop, serverless add-only); the
image entrypoint knew none of r2, hub, tunnel or reports, so fv-control had to replace it.

## Decision

### 2. One sequence, in Rust, chosen by the environment

`fv-boot` is a binary of `fastvideo-deploy` (feature `boot`), in every serve image beside fv-serve, and the images'
`ENTRYPOINT`. It runs the union of the steps above, each a typed function on four seams (`Spawner`, `Http`, `Clock`,
`LogSink`, `boot/sys.rs`), so the whole sequence runs in tests against a fake provider:

1. `[fv-boot] start (<kind>; fv-boot <version>)`; the **driver check** (`FV_SKIP_DRIVER_CHECK=1` skips it);
2. the **volume**: the serverless link, `FV_WEIGHTS` defaulted to the root that exists, `[fv-boot] volume: <FV_WEIGHTS>
   (N trees)` or `… missing`; the state directory;
3. **config materialization** (§3.2) and `fv-serve <args> --print-config` (a refused config fails the boot before any
   download, everywhere);
4. the upsampler **sidecar** start (`fv-upsampler start`, unchanged);
5. the **weights plan** (§3.3) behind the `WeightsSource` trait: `volume`, `r2`, `hub`, `none`, and `registry` typed
   and refused until the registry's `hydrate` exists (model-registry-rs WO-002);
6. the **tunnel** of a GMI / Brev pod (cloudflared pinned by SHA-256; named or quick) and its `tunnel` report; an edge
   front's `FV_DISPATCH_ENDPOINT`;
7. the **watchdog**, `fv-boot watchdog`, a process of its own that outlives the exec;
8. the sidecar **wait** (its exports go to fv-serve), the serverless `serve` report, then **exec** fv-serve (same pid,
   the environment exactly as computed, `FV_TUNNEL_TOKEN` removed).

The kind of host (`Kind`) comes from the environment fv-control already sends, never from a new switch. Precedence,
first match wins:

1. `FV_PROVIDER` set and not `runpod` → **provider pod** (GMI / Brev);
2. `RUNPOD_ENDPOINT_ID` (Runpod's own) or `FV_CONTROL_ENDPOINT` (fv-control's) → **serverless worker**, even with
   `FV_SERVE_ROLE=worker` (`scripts/serve/runpod-endpoint.sh` lets a queue endpoint carry the worker role env);
3. `FV_SERVE_ROLE=worker` **and** `RUNPOD_POD_ID` → **cluster worker** (fv-control's cluster and standalone pods);
4. else **the image on its own**: GCP and CloudRift workers (they set `FV_SERVE_ROLE=worker` but are not Runpod pods),
   Vast, the scripts.

It selects only what the legacy boots did differently: the `/fv-worker.toml` handling of controller workers (cluster,
provider), `FV_WORKER_ID`, the tunnel and the fail sleep of provider pods, the Runpod watchdog. GCP / CloudRift
workers therefore boot as the old `fv-entry.sh` did for them (their `--config` argument, no gateway edit, no Runpod
watchdog), and a role endpoint as any serverless worker.

`fv-boot --version`, `--help` and `--print-config` exec fv-serve directly (CI's image smoke, `docker run <image>
--version`). `fv-boot version` prints fv-boot's own version. Arguments are fv-serve's, as the entrypoint passed them.

The r2 source keeps `scripts/gpu/r2-fetch.py` (the image's copy when its SHA-256 is fv-control's pin, else
fv-control's download) until `hydrate` replaces every caller. A streaming fetch runs under `fv-boot fetch-guard`, a
process of its own: fv-serve starts once the files its loader does not gate are in place; the guard times the fetch
(`fetch`, `load_overlap` steps) and, if the fetch fails after fv-serve started, stops fv-serve (TERM, the grace,
KILL), as the shell subshell did.

### 3. The environment contract

Every variable fv-boot reads, with its source today. No new name was invented except `FV_BOOT_LEGACY` (§6), the
rollback switch fv-control's shell preamble and fv-boot itself read.

#### 3.1 Identity, kind, paths

| Variable | Read for | Set by today |
|---|---|---|
| `RUNPOD_POD_ID` | `FV_WORKER_ID` (cluster), the serverless boot id, the Runpod watchdog, the edge front's endpoint, r2 `--pod` | Runpod |
| `RUNPOD_ENDPOINT_ID` | kind: serverless | Runpod serverless |
| `FV_CONTROL_ENDPOINT` | kind: serverless | fv-control (`serverless/payloads.ts workerEnv`) |
| `FV_SERVE_ROLE` | kind: cluster worker (`worker`) | fv-control (`workerSystemEnv`) |
| `FV_PROVIDER`, `FV_POD_ID`, `FV_POD_NAME` | kind: provider; report pod id; GMI watchdog | fv-control (`providerEnv`) |
| `FV_DISPATCH_FRONT` | `FV_DISPATCH_ENDPOINT` export (Runpod proxy URL, or the tunnel URL) | fv-control (`edgeEnv`) |
| `FV_WEIGHTS` | the weights root (default: `/runpod-volume/weights` or `/workspace/weights`, whichever exists) | fv-control, images |
| `FV_STATE_DIR` | state directory (default `/fvstate`) | fv-control |
| `FV_SERVE_BIN` | the serve binary (default `/opt/fastvideo-rs/bin/fv-serve`) | the shell boots' convention |
| `FV_VARIANT` | the missing-tree warning | images (`ENV`) |
| `FV_SKIP_DRIVER_CHECK`, `FV_LIBCUDA`, `FV_DRIVER_FAIL_SLEEP_S` | driver check | operators, tests (driver-check.ts) |
| `FV_FAIL_SLEEP_S` | pause before a failed boot exits (default 600 provider, 30 elsewhere) | serverless boots |
| `PATH` | `python3` lookup | the image |

#### 3.2 Config

| Variable | Form | Set by today |
|---|---|---|
| `FV_WORKER_TOML_B64` | inline (1st): decoded to `/fv-worker.toml` | fv-control (pool / spec `config_toml`) |
| `--config <path>` argument | named (2nd): used as given | GCP, CloudRift, Vast, v2 load balancer `args` |
| `FV_WORKER_CONFIG` | named, controller workers only: copied to `/fv-worker.toml` | fv-control (`pool.config`) |
| `FV_CONFIG` | baked (3rd) | images (`ENV`), fv-control (serverless `spec.config`) |

A controller worker's (cluster or provider) inline or `FV_WORKER_CONFIG` copy gets `register = false` under
`[gateway]` (idempotent). A controller worker with neither (and no `--config` argument) fails, as the shell boots did
(`set -u` on the unset `FV_WORKER_CONFIG`); it never falls back to the baked `FV_CONFIG` without the edit.

#### 3.3 Weights

| Variable | Read for | Set by today |
|---|---|---|
| `FV_WEIGHTS_SOURCE` | `volume` (default) / `r2` / `hub` / `none` / `registry` | fv-control (`runpodR2Env`, `runpodHubEnv`, `providerEnv`, `slsWeightsEnv`, `slsHubEnv`) |
| `FV_WEIGHTS_TREES_B64` | `tree` rows (hub) and `aux` rows (all) | fv-control (`weightsPlan`) |
| `FV_WEIGHTS_PLAN_URL`, `FV_R2_FETCH_URL`, `FV_R2_FETCH_SHA256` | r2: the plan, the fetcher and its pin | fv-control (`r2PodEnv`) |
| `FV_WEIGHTS_TOKEN` (serverless) else `FV_ENDPOINT_REPORT_TOKEN` | r2 `--token-env` | fv-control |
| `FV_R2_TREE_REVS`, `FV_R2_CACHE_ROOT` | r2 serverless cache check, add-only cache volume | fv-control (`slsWeightsEnv`) |
| `FV_R2_SKIP_TREES` | r2 trees not fetched (default: `*_mxfp8` below sm_100; `auxiliary/*` on an add-only cache) | operators |
| `FV_R2_STREAM`, `FV_SERVE_NICE`, `FV_STREAM_KILL_GRACE_S` | streaming | operators |
| `FV_R2_STATE`, `FV_R2_TOOLS`, `FV_R2_FETCH_IMAGE` | r2 paths | operators, tests |
| `FV_R2_CONCURRENCY`, `FV_R2_MIN_FREE_GB`, … | passed through to r2-fetch.py | operators |
| `FV_HUB_SHA256_B64`, `FV_SCRIPTS_URL` | hub: recorded hashes, fetch-hub-tree.py | fv-control (`slsHubEnv`, `runpodHubEnv`, `providerEnv`) |

#### 3.4 Sidecar, tunnel, watchdog

| Variable | Read for | Set by today |
|---|---|---|
| `FV_UPSAMPLER_LOCAL` (+ the launcher's own `FV_UPSAMPLER_*`) | the sidecar | fv-control (`upsamplerPodEnv`) |
| `FV_TUNNEL_TOKEN`, `FV_TUNNEL_URL` | named tunnel (token to cloudflared's env only; removed from fv-serve's) | fv-control (tunnels.ts) |
| `FV_CLUSTER_DEADLINE`, `FV_BACKSTOP_API_KEY`, `FV_MIN_BALANCE`, `FV_BACKSTOP_API` | the watchdog | fv-control (`edgeEnv`, `directEnv`, `providerEnv`) |

fv-boot exports to fv-serve: `FV_WORKER_ID`, `FV_DISPATCH_ENDPOINT`, `FV_WEIGHTS` (when defaulted), `FV_POD_ID` (r2 on
a Runpod pod), `FV_R2_SKIP_TREES`, `FASTVIDEO_STREAMING_FETCH`, `FASTVIDEO_STREAMING_FETCH_STATUS`, and the sidecar's
`FV_UPSAMPLER_BASE_URL` / `_MODEL` / `_VISION` / `_LOCAL`.

### 4. The phase-report protocol (kept byte-compatible)

Two protocols exist and fv-control parses both; fv-boot speaks the one its environment names:

- **pod** (`FV_ENDPOINT_REPORT_URL`, `Bearer $FV_ENDPOINT_REPORT_TOKEN`; providers.ts `endpointReport`): body
  `{"pod":"$FV_POD_ID","phase":…,"url":…,"detail":…}`, phases `weights` (`checking`, `downloading <tree>`), `tunnel`
  (with the URL), `failed`, `upsampler`; up to 5 × 20 s, retried only when no HTTP answer came (`curl -sS` in the
  shell: a 401 / 404 / 429 is an answer and ends the report).
- **serverless** (`FV_SLS_REPORT_URL`, `Bearer $FV_WEIGHTS_TOKEN`; serverless/ingest.ts `slsReport`): the same body
  with `pod = "<RUNPOD_POD_ID>@<boot start, unix s>"`, `url = ""`, phases `boot`, `cache` (`hit` / `miss: <trees>`),
  `weights` (hub's `weights ready: hub X GB fetched in Ys (Z MB/s)`), `step` (`<name> <secs>s (<note>)`: `tools`,
  `fetcher`, `fetch`, `stream_ready`, `load_overlap`, `aux`, `hub_tools`, `hub_<tree>`, `weights`, `upsampler`),
  `serve`, `failed`; best effort, once.

Details are cleaned as the shell did (no `"` or `\`, newlines as spaces, ≤ 280 chars); the body is serialized JSON
with the four keys in that order (identical bytes to the shell's for every detail the shell could send validly).
r2-fetch.py keeps posting its own `weights` progress with `--report-url`.

Log lines: `[fv-boot] HH:MM:SS +E.Es <msg>`, plus the untimed markers fv-control and the Brev host match exactly:
`[fv-boot] start…` (boottime.ts `boot_script`), `[fv-boot] volume: …` (`volume`), `[fv-boot] tunnel <url>` (the Brev
host watcher's grep), and the `driver_too_old: ` failure (podlogs.ts). The fixture
`crates/fastvideo-deploy/fixtures/boot-protocol.json` holds the bodies, markers and pinned constants (driver floor,
symbol, cloudflared URL and SHA-256, MXFP8 glob, streaming marker, image paths); `boot/tests.rs` proves fv-boot
produces them, `control/test/unit/fv-boot-protocol.test.ts` proves fv-control accepts and parses them.

### 5. Failure behaviour

A failed step logs `FAILED: <why>`, reports `failed` (either protocol, when a sink is configured), sleeps
(`FV_DRIVER_FAIL_SLEEP_S`, 20 s, for the driver; else `FV_FAIL_SLEEP_S`, default 600 s on a GMI / Brev pod, 30 s
elsewhere) and exits 1: Runpod restarts the container and the fetch resumes (complete trees are kept); fv-control's
boot diagnosis reads the line. The messages keep the shell boots' wording where something reads them (`fv-serve
config: …`, `no worker config X in the image`, `r2 weights: <the fetcher's FAILED line>`, `tree X: …`, `aux X`,
`r2-fetch.py sha256`, `cloudflared sha256`, `named tunnel did not connect: <redacted tail>`, `no tunnel URL: …`).
Never fatal: the sidecar (fv-serve starts and upsampling falls back), the watchdog's own start, the missing-tree
warning. A `registry` source or an unknown `FV_WEIGHTS_SOURCE` fails loudly instead of serving without weights.

Behaviour changes against the shell boots, all on the side of failing earlier or saying more: the driver check and
the `--print-config` preflight now run on every path (the image entrypoint included); the volume marker names
`FV_WEIGHTS`; the serverless driver failure carries the boot's own id; pod reports are retried on Runpod too (Runpod
pods get no report URL from fv-control, so nothing changes there today). A refused config's `fv-serve: …` lines are
copied to the log verbatim before the `FAILED:` line, so fv-control's boot diagnosis (podlogs.ts `CONTAINER_FATAL`)
sees a restarting container as a crash loop. The streaming guard and the watchdog stop fv-serve by pid through the
shell's `kill` builtin (`sh -c 'kill -s SIG "$1"'`): the images have no procps, so the shell boots' `kill` / `pkill`
binaries cannot be assumed. `FV_BACKSTOP_API_KEY` is removed from fv-serve's environment and every child's but the
watchdog's, as `FV_TUNNEL_TOKEN` is. A GMI / Brev pod on r2 cleans `.*.partial-*` / `.stale-*` leftovers and reports
`weights checking` after the fetch, as PROVIDER_BOOT's weights loop did; an unset `FV_R2_SKIP_TREES` that resolves to
nothing stays unset.

### 6. Staging (deploy compatibility) and what is deleted when

Images already deployed have no fv-boot, and fv-control is deployed by hand, separately from the images (plan A3
caveat). Every combination keeps working:

| fv-control | image | what runs |
|---|---|---|
| old | old | the shell boots (today) |
| old | new (fv-boot) | the shell boots: they still find `fv-upsampler`, `r2-fetch.py`, `fv-serve` at the same paths; a serverless template without an override runs the new `ENTRYPOINT` (fv-boot, a superset of fv-entry) |
| new | old | each shell boot starts with `PREFER_FV_BOOT_SH` (`control/src/fv-boot.ts`): `fv-boot` is not there, so the rest of the script runs as before |
| new | new | the preamble execs fv-boot, which reads the env fv-control sent |

`FV_BOOT_LEGACY=1` (any env layer of a cluster, pool, pod or endpoint, or a template's / VM's env) is the rollback
switch, on every path:

- fv-control's shell boots (cluster, standalone, edge, GMI / Brev, serverless with a config or r2 / hub): the preamble
  skips fv-boot and the shell body runs, exactly as on an old image;
- the image entrypoint (serverless without a boot override, v2 load balancers' `args`, GCP / CloudRift `docker run`,
  Vast's onstart, the scripts): fv-boot itself sees it and runs only what `fv-entry.sh` did (the serverless volume
  link, the `FV_WEIGHTS` default, the upsampler sidecar), then execs fv-serve with its arguments (no driver check,
  config step, weights plan, tunnel, watchdog or report).

It needs no new fv-control or image, but it does need the env to reach the container: a new pod, worker or VM (an
fv-control env change rolls a pool's pods; a serverless template update rolls its workers). A running container keeps
the boot it started with. Vast's onstart prefers fv-boot the same way.

Deleted in this change: `deploy/runpod/fv-entry.sh` (the images' `ENTRYPOINT` is fv-boot; nothing named the path),
the Vast onstart's direct `fv-serve` start (one fallback line remains). Already gone: `scripts/gpu/serverless-worker.sh`
(F1).

Kept until the images with fv-boot are promoted and no pool, endpoint, channel or pinned spec can start an older
image (the follow-up, C2b): `WORKER_BOOT`, `EDGE_WORKER_BOOT`, `WATCHDOG_WORKER_BOOT` and `WATCHDOG`; `PROVIDER_BOOT`;
`SLS_BOOT`, `SLS_R2_BOOT`, `SLS_HUB_BOOT`, `SLS_R2_SH`, `SLS_HUB_SH`; `DRIVER_CHECK_SH`; `UPSAMPLER_START_SH`,
`UPSAMPLER_WAIT_SH`; `R2_FETCH_SH`, `R2_RUNPOD_SH`, `HUB_RUNPOD_SH`, `HUB_FETCH_TREE_SH`, `WEIGHTS_LOOP_SH`,
`TUNNEL_SH`; the Vast fallback line. Then fv-control sends env only: the cluster and standalone payloads drop
`dockerEntrypoint` / `dockerStartCmd`, serverless templates drop their boot override, GMI's create drops its
`bash -c` command and the Brev host script runs the image without `--entrypoint bash`. **"fv-control sends env only"
is therefore not met by this change; it is met by C2b after the promotion.**

Kept for other reasons: `deploy/runpod/fv-upsampler.sh` (the sidecar's supervisor of llama-server; fv-boot drives it
through `start` / `wait` and the shell boots still call it; a Rust port needs its own GPU check and is deferred);
`scripts/gpu/r2-fetch.py` (behind the `r2` source until `hydrate`); the host halves `BREV_BOOTSTRAP` /
`brevHostScript`, `scripts/gcp/startup.sh` and CloudRift's `vm_boot` (they install Docker, mount disks, front TLS and
start the image, whose entrypoint is now fv-boot: the provider layer of C3, not a container boot).

**The image build.** fv-boot has its own Dockerfile stage, `boot-build` (`FROM builder`, `cargo build --release -p
fastvideo-deploy --features boot --bin fv-boot`), which every serve stage copies from. Tools releases built after this
change carry `out/fv-boot` in their `serve-cuda` and `serve-cpu` sets (`release-artifacts-pod.sh build_serve`,
`tools-deps.sh`, the `tools-release.sh` gates), and `scripts/ci/prebuilt.sh` then passes `boot-build=<that set>` as a
named context, so nothing compiles. With a release from before fv-boot (the newest today, `tools-v0.1.37`) the
`serve-build` / `cpu-build` contexts stay prebuilt and only `boot-build` compiles on the runner (one crate and its
dependencies), with a notice in the job. `serve-image.yml` therefore builds with both kinds of release.

**Deploy order (recommended).** The GPU gate covered one kind (the image on its own, `:ltx`, volume weights). So:

1. merge; CI builds every serve image with fv-boot as the entrypoint (`ci-images.sh` and the debug smoke assert it);
2. publish a tools release from the merged commit (it carries fv-boot; optional, it only saves the runner compile);
3. deploy fv-control with `FV_BOOT_LEGACY=1` in the account env layer (every boot it sends keeps the shell path);
4. promote the images (`release.sh promote`): pods and endpoints fv-control starts still run the shell boots; image
   entrypoints run fv-boot (GPU-checked kind);
5. lift `FV_BOOT_LEGACY` per pool / endpoint after a check of that kind (the open gates below);
6. C2b once every pool runs fv-boot and no channel, pinned spec or running pod can start a pre-fv-boot image.

Open per-kind gates (each one pod or worker, its boot log, timeline and reports checked against the shell boot's):

- a cluster worker (inline config, `[gateway]` edit, `FV_WORKER_ID`, volume weights);
- a serverless worker with weights_source r2 and streaming (cache miss, `fetch-guard`, `load_overlap`, the
  serverless reports in the endpoint's boots view);
- an edge front (its `FV_DISPATCH_ENDPOINT`, the watchdog's balance check and self-delete at a short deadline);
- a GMI pod and a Brev pod (hub or r2 weights, the tunnel and its report, the Brev host watcher's `tunnel` grep, the
  GMI watchdog);
- the upsampler sidecar on any of these (`FV_UPSAMPLER_LOCAL=1`).

### 7. Not in this change

- the `registry` source (`hydrate`, model-registry-rs WO-002): typed and refused;
- the upsampler launcher in Rust (above);
- the host halves (C3 `Provider`);
- the deletion of the shell boots and "env only" (C2b, after the promotion).

### 8. Validation

CPU, on the build pod: `cargo test --locked -p fastvideo-deploy --features boot` (the sequence against a fake
provider per kind and source, the config forms, driver verdicts, report bodies, watchdog loops, the guard; and
`tests/boot_r2.rs`: the real `r2-fetch.py` against an in-process fake chunk source, cold, warm and streaming through
the real `fv-boot fetch-guard` binary); control: `fv-boot-protocol.test.ts` and the existing boot tests (the shell
boots still run in bash in CI).

GPU (coordinator, done 2026-10-24): one pod boot on the EU volume (`bz0ofkfx45`, EU-RO-1) with the PR's release
fv-boot on the promoted `fastvideo-rs-serve:ltx` image (fv-serve 0.1.37), RTX PRO 6000:
`scripts/gpu/fv-boot-smoke-pod.sh`: driver 595.91 ok, volume (9 trees), `--print-config` ok, exec fv-serve, ready,
one native job (480p 16:9) succeeded in 108 s. Evidence: `artifacts/runpod/c2-boot/`. Only the image kind with volume
weights; the other kinds are the open gates of §6.

## Consequences

- One boot, testable without a GPU; fv-control and the scripts run the same steps from the same env.
- The images' entrypoint gains the driver check and the config preflight: a bad host or config fails in seconds with a
  line fv-control already understands.
- Until C2b, the shell boots and fv-boot coexist; a change to the boot must be made in fv-boot (the new images) and,
  only if it matters for the old images still running, in the shell text. The fixture keeps the protocol in step.
