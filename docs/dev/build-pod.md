# Shared build pod

Agents compile and test on one shared Runpod **CPU pod** with a 200 GB
**network volume** (`fv-build`), not in their own containers, whose disks are
small. `scripts/dev/build-pod.sh` drives it over the pod's HTTPS proxy
(`https://<pod>-8000.proxy.runpod.net`); there is no SSH. Each agent gets its
own source snapshot and `CARGO_TARGET_DIR` on the pod's container disk, so
agents build in parallel without sharing a cargo lock or clobbering each
other's artifacts; the toolchains and the compile cache live on the volume.

**Status (2026-09-28):** in use. The `fv-build` volume (`pxy4hlsnwq`,
EU-RO-1) exists; pods were created, validated and stopped on it — see
[Measured](#measured-2026-09-28) for build times, sizes and the stop/start
behaviour.

## One-liners

```bash
B=scripts/dev/build-pod.sh
A=$(basename "$PWD")          # agent name: your worktree's directory ("." works too)

$B up                                     # reuse / start / create; waits until ready
$B run $A -- cargo check -p fastvideo-serve --all-targets
$B run $A -- bash scripts/serve/check.sh
$B run $A -- cargo check -p fastvideo-cudarc -p fastvideo-gpucheck -p fastvideo-cli \
             --features fastvideo-cudarc/cuda,fastvideo-gpucheck/cuda   # CUDA type-check
$B run $A -- cargo build --release -p fastvideo-serve --features cuda,http-client
$B fetch $A release/fv-serve              # -> artifacts/build-pod/$A/release/fv-serve
$B run $A -- cargo build --release -p fastvideo-gpucheck --features cuda
$B fetch $A release/fv-gpucheck
$B run $A -- bash tests/compat/run.sh      # client-compat suites (Node + Chromium on the pod)
$B status                                 # pod, $/hr, setup, jobs, disk, self-stop timers, per-agent sizes + eviction
$B cancel <job>                           # cancel a job whose client died
$B clean $A target                        # drop your target dir when you are done
$B evict                                  # run the eviction pass now (see Limits)
$B stop                                   # when nobody needs it (it also stops itself)
```

`run` syncs first (only files whose size or mtime changed; deletions are
mirrored), then streams the log and exits with the command's exit status.
`--no-sync` skips the sync. Leading `K=V` words set allowlisted environment
variables: `run $A -- FV_SERVE_HEAVY=1 bash scripts/serve/check.sh`.
Ctrl-C cancels the job on the pod; `log <job>` re-attaches after a dropped
connection.

## What runs where

| Path | What |
|---|---|
| container disk `/root/fvb/worktrees/<agent>/` | the agent's snapshot: tracked + untracked, non-ignored files (submodules included) |
| container disk `/root/fvb/target/<agent>/` | that agent's `CARGO_TARGET_DIR` (incremental while the pod runs; evicted when unused, see Limits) |
| container disk `/root/fvb/.last-use/<agent>` | stamp of the agent's last sync, job or fetch (what eviction goes by) |
| container disk `/root/fvb/cargo/` | `CARGO_HOME`; its `registry/cache` and `git/db` link to the volume |
| volume `cargo/registry/cache`, `cargo/git/db` | downloaded crates and git checkouts (shared) |
| volume `rustup/` | shared `RUSTUP_HOME` (stable + rustfmt + clippy, per `rust-toolchain.toml`) |
| `sccache/` | shared sccache (40 GB cap): registry crates compile once for all agents |
| `cuda-13.4/` | CUDA 13.4.92 nvcc, crt, cudart, NVRTC, libnvvm, CCCL, tileiras (NVIDIA redist tarballs, sha256-checked) — the same toolkit the CI builder image pins |
| `node-v22.23.3/`, `playwright-1.56.1/`, `pw-browsers/` | Node, the Playwright package and its Chromium, for `tests/compat/run.sh`, `tests/console/run.sh` and `FV_SERVE_UI=1` |
| `jobs/`, `logs/`, `ledger.tsv` | job logs, service log, pod-side ledger (self-stops) |

**Why targets are not on the volume:** the first pod kept them there. Cargo
on the network filesystem took 18 min for a cold `cargo check` of the serve
crates and 30 s for a no-op one (fingerprint stats), and a full snapshot sync
took 63 s to extract. On the container disk with the volume's sccache: 3.5 min,
1.4 s and 5 s. The price is that a stopped or recreated pod starts with empty
target dirs, which sccache refills (release builds of fv-serve / fv-gpucheck
in about 6.5 min each). `FV_BUILD_TARGETS=volume` in the pod env restores
the old layout.

The first boot on a fresh volume installs the toolchain onto the volume
(CUDA redist in ~90 s, the Rust toolchain, sccache); later boots only
`apt-get install` cmake/clang/mold/pkg-config into the container and are
ready in 30–60 s. A second phase (in the background; build jobs do not wait
for it) installs ffmpeg and Chromium's system libraries into the container
and, once, Node and Playwright's Chromium onto the volume; compat, console
and `FV_SERVE_UI=1` jobs wait for it. The image is stock `rust:1-bookworm`;
nothing is baked for us.

**apt on the pod is serialized.** The image's `docker-clean` apt hook deletes
`/var/cache/apt/archives/*.deb` after every dpkg run and `apt-get update`, so
a release-artifacts job's `apt-get` running while the extras unpacked ffmpeg
deleted that install's packages (`extras: failed`, `apt-get install -y exited
100`, no ffmpeg / libvpx, 2026-10-06). The server and
`release-artifacts-pod.sh` now take one lock (`flock /var/lock/fv-apt.lock`),
the server clears the hook (`/etc/apt/apt.conf.d/zz-fv-keep-debs`), and a
failed install is repaired (`dpkg --configure -a`, `apt-get -f install`) and
retried up to 3 times. `status` shows `extras.ffmpeg_libvpx`. Anything else
that runs apt on the pod should take the same lock.

Jobs get: `CUDARC_CUDA_VERSION=13000`, `NVCC` and `CUDA_HOME` pointing at the
volume's toolkit (so `--features cuda` builds compile the AOT cubins),
`RUSTC_WRAPPER=sccache`, mold as the linker, and the CI builder's release
overrides (`CARGO_PROFILE_RELEASE_LTO=off`, `CODEGEN_UNITS=16`,
`PANIC=unwind`) so release builds match the published images and take minutes,
not the fat-LTO hour. Pass `CARGO_PROFILE_RELEASE_LTO=fat
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort` for a
workspace-profile build.

## Release artifacts (prebuilt binaries for the image workflows)

Owner decision 2026-10-06 (option 1b): **the build pod compiles, R2 hands
off, GitHub assembles.** Every Rust binary and the oxide cubins the images
and CI jobs need are compiled here; the GitHub workflows download them and
only run the lean Docker stages (and the tests), no `cargo`. The pod has no
Docker daemon, so the images themselves are still assembled on GitHub.

```bash
B=scripts/dev/build-pod.sh
$B release-artifacts HEAD          # or a sha / origin/main; wakes the pod (`up`)
$B release-artifacts <sha> --force # rebuild and re-upload
$B release-artifacts <sha> --no-upload --keep   # build + verify only, into artifacts/release/<sha>/
$B release-artifacts <sha> --sets "serve-fake gpucheck-tests"   # a subset
```

What it does:

1. Resolves the revision to a full sha; skips if `artifacts/<sha>/manifest.json`
   is already in R2 (`--force` rebuilds). One run per container at a time
   (`flock` on `~/.config/fv-build/release.lock`).
2. Checks the sha out into a temporary local worktree (plus the
   `third_party/cutile-rs` submodule), computes the build identity the
   workflows compute (`scripts/gpu/docker.sh build-id`, the commit time),
   `up`s the pod and syncs the snapshot to the fixed agent dir **`fv-release`**
   (one target dir reused across shas, so consecutive commits build
   incrementally). Eviction (see Limits) never takes `fv-release` while its
   job runs or within an hour of its last sync, job or fetch, and never
   touches `release-cache/` on the volume.
3. Runs `bash scripts/dev/release-artifacts-pod.sh` (allowlisted) on the pod.
   The recipe is the driving checkout's copy (its hash is in the manifest), so
   older shas build too.
4. Fetches the tarballs and `manifest.json`, checks every tarball's sha256,
   uploads them to R2, `manifest.json` last (its presence means "complete").

The sets, each a tarball whose root is laid out like the Docker stage it
replaces (so GitHub uses the extracted directory as a named build context):

| set | built as | replaces / used by |
|---|---|---|
| `oxide` | `fv-oxide-aot --sm 100,120` (cached on the volume by its inputs, like the image's layer cache) | stage `oxide` (gpucheck.Dockerfile) |
| `gpucheck` | `cargo build --release -p fastvideo-gpucheck --features cuda`, oxide embedded (`FV_REQUIRE_OXIDE=100,120`); checked with `fv-gpucheck nvrtc` | stage `binary`; gpucheck-t0 nvrtc gates |
| `gpucheck-vast` | the same without nvcc and oxide, exactly as `vast-pytorch.Dockerfile` builds it today (its builder's `NVCC=/usr/local/cuda-13.0/bin/nvcc` does not exist, so its binary NVRTC-compiles at run time) | stage `binary` (vast-pytorch.Dockerfile) |
| `hf-fm` | `cargo install hf-fetch-model --features cli` (unpinned, as in the image; version in the manifest) | stage `hf-fm` (both Dockerfiles) |
| `serve-cuda` | `cargo build --release -p fastvideo-serve --features cuda,http-client` | stage `serve-build` |
| `serve-gateway` | `… --features http-client` | stage `gateway-build` |
| `serve-fake` | serve-compat's debug `--features fake,full` build, same `--config` opt-levels | serve-compat `build` job |
| `gpucheck-tests` | `cargo test -p fastvideo-gpucheck -p fastvideo-cudarc --lib --bins --no-run` + `tests.tsv` | gpucheck-t0 `unit-tests` job |

**Same settings as the image builder:** `CARGO_PROFILE_RELEASE_LTO=off`,
`CODEGEN_UNITS=16`, `PANIC=unwind`, `CUDARC_CUDA_VERSION=13000`, CUDA
13.4.92 (the redist the pod installs, same version the apt pins name), the
`stable` toolchain of `rust-toolchain.toml`, **no RUSTFLAGS** (the script
drops the pod's mold link-arg, so the system linker links, as in the image),
and `FV_GIT_SHA` / `FV_BUILD_TIME` / `BUILD_ID` as the workflows pass them,
so `fv-serve --version` reports the same build. The pod is Debian bookworm
(glibc 2.36) and the images Ubuntu 22.04 (glibc 2.35): the script fails if
any ELF needs a `GLIBC_` symbol newer than 2.35. The oxide build needs
libclang and cuRAND headers the volume's toolkit lacks: the script installs
`libclang-dev` and unpacks the `libcurand` redist headers (sha256-checked)
into `release-cache/` on the volume and a private toolkit overlay.

**R2 layout** (bucket `fv-build-artifacts`, account `Maximal`):

```
artifacts/<sha>/manifest.json        sha, build_id, build_time, run_id, builder (rustc -vV,
                                     cargo, nvcc, tileiras, glibc, recipe hash), settings,
                                     per set: tarball, sha256, size, features, NEEDED libs,
                                     and every file's sha256/size/mode
artifacts/<sha>/<set>.tar.gz         one per set (gzip, reproducible tar: sorted, owner 0,
                                     mtime = commit time)
```

**Retention:** a lifecycle rule (`expire-artifacts-30d`) deletes every object
30 days after upload and aborts incomplete multipart uploads after 1 day. A
commit older than that falls back to compiling (or is rebuilt with
`release-artifacts <sha>`).

**Credentials:** the upload uses an S3 (SigV4) key scoped to this one bucket,
read from `~/.config/fv/r2-build-artifacts-rw.env` (mode 600, never printed,
never on argv; `FV_R2_ARTIFACTS_ENV_FILE` overrides):

```
FV_R2_ARTIFACTS_ENDPOINT=https://0cd06fd37ee4e07de370821cb3852a8a.r2.cloudflarestorage.com
FV_R2_ARTIFACTS_ACCESS_KEY_ID=…
FV_R2_ARTIFACTS_SECRET_ACCESS_KEY=…
```

`scripts/dev/r2.py` (stdlib only: `head|get|put|ls`) is the client on both
sides; `python3 scripts/dev/test_r2.py` checks its signer against AWS's
published SigV4 examples. **Creating the keys (owner, once):** the
Cloudflare API token in `/root/.config/fv/cf_api_token` can manage buckets but
not create API tokens, so R2 S3 keys cannot be minted from here. In the
Cloudflare dashboard → R2 → *Manage R2 API tokens* → *Create API token*:

1. `fv-build-artifacts-rw`: permission **Object Read & Write**, *Apply to
   specific buckets only* → `fv-build-artifacts`, TTL forever. Put its
   Access Key ID / Secret Access Key and the account's S3 endpoint into
   `~/.config/fv/r2-build-artifacts-rw.env` on the machine that runs
   `release-artifacts` (the coordinator's container), `chmod 600`.
2. `fv-build-artifacts-ro`: permission **Object Read only**, same bucket.
   Add it to the GitHub repository (Settings → Secrets and variables →
   Actions) as `FV_R2_ARTIFACTS_ENDPOINT`, `FV_R2_ARTIFACTS_ACCESS_KEY_ID`,
   `FV_R2_ARTIFACTS_SECRET_ACCESS_KEY`.

**Trigger: the coordinator builds before it pushes.** Whoever pushes a
commit that CI will build runs `release-artifacts` for it first:
the coordinating session after a merge, before `git push origin main`
(`build-pod.sh release-artifacts HEAD && git push origin HEAD:main`), and an
agent before pushing a PR head whose workflows it wants fast. It wakes the
pod (`up`: start, create, or fail with the reason, e.g. the balance floor)
and takes ~1–3 min when only a few crates changed, longer after a pod stop
(empty target dir; sccache refills it). Chosen over the alternatives because
it needs no new moving part: a GitHub workflow cannot wake the pod or reach
its token (it lives only in this container), and fv-control polling GitHub
would need the pod token and the R2 write key in the Worker and a job queue
on the pod for commits nobody waits for. If the build was skipped or is
still running, the workflows **fall back** to today's in-image compile with
a `::warning::` (CI never deadlocks); the repository variable
`FV_PREBUILT_WAIT_MIN` makes them poll R2 that long first, and
`FV_PREBUILT_DISABLE=1` turns the download off. Pull-request jobs use the
PR **head** sha (`github.event.pull_request.head.sha`), not the merge ref.

**Measured (2026-10-06, pod cpu3c 32 vCPU, $0.96/hr):** first run on a fresh
pod (empty target dir, cold sccache for the no-mold flags) **28 min**
(oxide 3 min, gpucheck 7, serve-cuda 4, gateway 1.5, fake 4, tests 2,
vast 3.5, hf-fm 2.7); the next commit (main `0f64377`, after `89fa23b`)
**8.8 min** with oxide from the volume cache; fetch of all 8 sets (201 MB
gzip) 20 s. About $0.15 of pod time per incremental commit. Output for
`0f64377`: serve-cuda 31 MB gz (fv-serve 115.5 MB), serve-gateway 18 MB,
serve-fake 31 MB, gpucheck 18 MB, gpucheck-vast 10 MB, hf-fm 18 MB (0.12.1),
gpucheck-tests 84 MB, oxide 0.2 MB; every ELF needs only libc / libm /
libgcc_s (libstdc++ for the fake build), max `GLIBC_2.35`. Against the image
CI built for the same commit: `fv-serve --version` identical (git sha, build
time, build id `6359813b8b741bf5`, profile, features) for the CUDA and the
gateway binary, `fv-serve.features` identical, sizes within 0.1 % (115.50
vs 115.39 MB, 58.59 vs 58.66 MB; rustc 1.98.1 on the pod vs the image
builder's cached stable). The gpucheck-tests binaries pass outside the pod
(40 + 516 tests, `prebuilt.sh run-tests`).

## Allowlist

The service runs no shell. Allowed: `cargo check|build|test|clippy|fmt|doc|
tree|metadata` (not `--target-dir` / `--manifest-path`), `bash
scripts/serve/check.sh`, `bash scripts/gpu/lint.sh`, `bash
tests/compat/run.sh`, `bash tests/console/run.sh`, `bash
scripts/dev/release-artifacts-pod.sh` (release artifacts, above). Jobs see no `RUNPOD_*`
variables (they would switch fv-serve's ICE into "serving on a Runpod pod"
mode in local tests). Environment overrides:
`CUDARC_CUDA_VERSION`, `FV_*`, `RUST_LOG`, `RUST_BACKTRACE`, `RUSTFLAGS`,
`RUSTDOCFLAGS`, `CARGO_PROFILE_*`, `CARGO_INCREMENTAL`, `CARGO_BUILD_JOBS`,
`CARGO_TERM_COLOR`. Add to `CARGO_SUBCOMMANDS` / `SCRIPTS` / `ENV_ALLOW` in
`scripts/dev/build-pod-server.py` when a new build path needs it. Build
scripts and tests still execute code, so the token is the real boundary.

## Auth

`up` generates a fresh random token per pod and keeps it in
`~/.config/fv-build/` (mode 600, never printed). The pod only receives its
SHA-256. Every worktree in this container shares that directory, so all local
agents can use the pod. A session in another container has no token: it must
not recreate a pod another session is using; ask, or wait for it to stop.

## Costs and money guards

| Item | Price (Runpod secure cloud, 2026-09-28) |
|---|---|
| **Pod (default): cpu5c, 32 vCPU / 64 GB** | **$1.12/hr** (never allocated on 2026-09-28) |
| fallback: cpu3c, 32 vCPU / 64 GB | $0.96/hr (what `up` got every time) |
| cpu3g, 32 vCPU / 128 GB (`FV_BUILD_FLAVORS=cpu3g`) | $1.28/hr |
| cpu5c / cpu3c, 16 vCPU / 32 GB (automatic fallback, `FV_BUILD_VCPUS_FALLBACK`) | $0.56 / $0.48/hr |
| Volume `fv-build`, 200 GB in EU-RO-1 | $0.07/GB/month = **$14/month** |

32 vCPUs keep parallel rustc and the per-SM nvcc cubin compiles busy; 64 GB
is plenty for 32 rustc processes with LTO off. cpu5c is the newer generation
(faster per core) for 17 % more per hour; `FV_BUILD_FLAVORS` falls back to
cpu3c when cpu5c has no stock. The volume pins the pod to EU-RO-1, whose CPU
stock comes and goes: on 2026-09-28 there was no 32-vCPU cpu5c/cpu3c/cpu3g/
cpu5g for 15+ minutes while 16 vCPU had stock. `up` therefore tries 32, then
16 vCPU, and retries both every 30 s for `FV_BUILD_STOCK_WAIT_S` (900).

- **Self-stop, three layers** (`status` prints a `self-stop:` line: uptime,
  idle time, time to the idle and the cap stop, and the last failed attempt):
  1. *The pod's server* (`build-pod-server.py`, `StopPolicy` / `Stopper`).
     **Idle:** `FV_BUILD_IDLE_MIN` (default 20) minutes with no queued or
     running job and no work request (sync, job submit, artifact fetch,
     clean). Status, health, agent listings, manifests and job-log polls do
     **not** count, so a monitor polling `status` cannot keep the pod alive.
     **Cap:** `FV_BUILD_MAX_HOURS` (default 8) after boot it stops at once if
     no job is active; otherwise it refuses new jobs (HTTP 503), writes a
     warning into the active jobs' logs and stops `FV_BUILD_MAX_GRACE_MIN`
     (default 30) later even if they still run. The stop tries REST stop,
     REST terminate, GraphQL `podStop`, GraphQL `podTerminate` with the
     pod-scoped `RUNPOD_API_KEY`, logs every failure with its HTTP status and
     body (`GET /v1/log`, and Runpod's container log), and retries 1, 2, 4 …
     10 min apart; 10 min after an accepted call it tries again if the pod
     still runs.
  2. *A curl watchdog in the pod's start command*, a separate process: 15 min
     after the server's hard stop (cap + grace + 15 min = 8 h 45 min) it
     stops / terminates the pod every 5 min until it is gone.
  3. *fv-control's per-minute cron* (`control/src/buildpod.ts`, independent
     of the pod and of this container): a `RUNNING` pod attributed
     `external:build-pod` is stopped (terminated if the stop is refused)
     once Runpod reports it up ≥ 9 h (`build_pod_max_h`), or once its public
     `/healthz` shows no jobs and idle ≥ its idle stop + 15 min
     (`build_pod_idle_grace_min`). Alert kind `build_pod`, audited.
     Its dashboard shows the same timers read only (Dashboard → Build pod):
     the public `/healthz` carries the timers, the last self-stop attempt
     (`self_stop`) and the active jobs (`jobs`: id, agent, state, seconds;
     no command lines).
  The volume keeps everything worth keeping, so a terminate costs nothing
  but the container disk (which a stop loses too); `up` recreates.
- **Incident 2026-10-02 (why the layers exist).** Pod `jactz9o1k6x58u` ran
  9 h+ ($0.96/hr): idle from ~05:15 to ~10:30 UTC and past its 8 h cap,
  without stopping. Its container log shows the watchdog *did* fire, 65
  times from 04:24 on (`self-stop (idle 20 min)`, later `self-stop
  (wall-clock cap 8h)`), and every call failed: `POST /pods/<id>/stop
  failed: HTTP Error 403` and the same for `DELETE`. Cause: Cloudflare in
  front of `rest.runpod.io` and `api.runpod.io` answers Python's default
  `User-Agent: Python-urllib/3.x` with 403 `error code: 1010` (reproduced
  with the account key: 403 with the default agent, 200 with any explicit
  one). The old code did not log the response body, so the 403 looked like
  a permission problem. No self-stop had ever worked; earlier pods were
  stopped by hand. Second, lesser flaw: every authenticated request,
  `status` polls included, reset the idle timer. Fixed as above; the
  server's unit tests are `python3 scripts/dev/test_build_pod_server.py`.
- `up` refuses to run below a $8 balance (`FV_MIN_BALANCE`) and deletes a pod
  created above `FV_BUILD_MAX_DPH` (default $1.50/hr).
- Ledger: `~/.config/fv-build/ledger.tsv` (local) and `ledger.tsv` on the
  volume (pod-side stops).

## Limits

- **Disk:** target dirs share the 200 GB container disk (`FV_BUILD_CONTAINER_GB`).
  Measured: `FV_SERVE_HEAVY=1 FV_SERVE_UI=1 check.sh` leaves **~45 GB**
  (debug, every feature combination); release fv-serve + fv-gpucheck + the
  CUDA check **2.6 GB**; other agents' test targets 13–21 GB. Three heavy
  agents fill it (on 2026-09-29 nine finished agents had left 185 GB and
  27 GB free), so `clean <agent> target` when done. `status` shows
  `local_disk`, the volume's usage (a du every 15 min) and a per-agent table:
  idle hours, target/snapshot sizes (a du every 5 min), and hours until
  eviction.
- **Eviction** (automatic, on the pod): every minute and after each job, the
  server removes the target dir and snapshot of every agent unused (no sync,
  job or fetch) for more than `FV_BUILD_EVICT_HOURS` (6), then, while the
  container disk has less than `FV_BUILD_EVICT_FREE_GB` (40) free, target dirs
  in least-recently-used order. An agent with a job running or queued is never
  touched. The release agent **`fv-release`** (`FV_BUILD_EVICT_PROTECT`; its
  `release-artifacts` run syncs, builds, then fetches the tarballs from its
  target dir) is also kept for `FV_BUILD_EVICT_HOLD_MIN` (60) after its last
  sync, job or fetch, and goes last under disk pressure (its target dir is
  the incremental cache of consecutive release builds). Only
  `target/<agent>` and `worktrees/<agent>` on the container disk are ever
  evicted: the volume (`release-cache/`, sccache, toolchains, crates) never
  is. Eviction is not activity: it never delays the idle self-stop. Each
  eviction goes to `logs/pod.log` and `status`
  (`eviction.recent`). An evicted target dir only costs a cold build (sccache
  refills it); an evicted snapshot is re-sent by the next `run`. `sync` and
  `run` that still find less than `FV_BUILD_MIN_FREE_GB` (2) free after a pass
  fail with `build pod disk full: …` (HTTP 507). The eviction settings are
  sent at pod creation (`up`); the logic has unit tests,
  `python3 scripts/dev/test_build_pod_server.py`.
- **Network volume I/O** is slow for many small files, which is why only
  large-file caches (sccache, `.crate`s, toolchains) live there.
- **Concurrency:** 4 jobs at once (`FV_BUILD_MAX_JOBS`), one per agent; more
  queue. Two agents building release CUDA binaries at once share 32 vCPUs.
- **Transfers** go through the Runpod HTTPS proxy: uploads up to 512 MiB per
  sync (a full snapshot of this repo is ~40 MB, later syncs are deltas);
  artifacts up to 2 GiB, gzip-compressed in transit.
- **No GPU:** CUDA code compiles (nvcc cubins, NVRTC sources, type-checks)
  but never runs here. Kernel and model runs stay on GPU pods
  (`scripts/gpu/runpod-http.sh`).
- **Server updates:** the service code is sent at pod creation (so is the
  start command with its curl watchdog). The self-stop fix of 2026-10-02
  (server `957ae128b12e` → the next sha) reaches the build pod at its next
  creation: `up` recreates a stopped pod whose server is older. After
  changing `build-pod-server.py`, `down` then `up` (the volume keeps all
  caches); `up` prints a note when the pod runs an older server, and
  recreates (rather than starts) a *stopped* pod whose server is older.
  `down` kills other agents' running jobs: check `status` first.

## Measured (2026-09-28)

Pods on `fv-build` in EU-RO-1: `up` asked for cpu5c then cpu3c at 32 vCPU
and always got **cpu3c, 32 vCPU / 64 GB, $0.96/hr** (once, after 15 min
without 32-vCPU stock, cpu3c 16 vCPU / 32 GB at $0.48/hr). The create
payload is accepted as written. The container sees the host's 192 CPUs in
`os.cpu_count()`, but its affinity (what cargo sizes `-j` by) is the pod's
vCPUs.

**Stop / start:** `stop` on a network-volume pod is accepted (`EXITED`);
`up` then starts it again: setup ready ~30 s after the container starts
(CUDA, rustup and Node markers on the volume, nothing reinstalled), the
volume contents intact (15.8 GB used before and after), the container disk
empty (target dirs and snapshots gone, as expected).

Boot: first boot on the empty volume ~2 min to `ready` (CUDA redist ~90 s),
later creates 30–60 s; the browser extras (ffmpeg, Chromium libraries, and
once Node + Chromium onto the volume) follow in the background in 1–2 min.

| Job (agents `validate`, `validate2` in parallel) | Volume targets, 32 vCPU | Container-disk targets |
|---|---|---|
| full snapshot sync (1403 files, 29 MB gz) | 63 s | 4–5 s |
| `cargo check` 13 serve crates `--all-targets`, cold | 18 min 20 s (cold registry, cold sccache) | 3 min 32 s (16 vCPU, sccache warm) |
| same, no-op | 30 s | 1.4 s |
| CUDA type-check (`fastvideo-cudarc/-gpucheck/-cli`, `cuda`) | 18 min 14 s | 4 min 31 s (16 vCPU) |
| release `fv-gpucheck --features cuda`, cold | — | 6 min 33 s (16 vCPU, next to check.sh) |
| release `fv-serve --features cuda,http-client`, cold | — | 6 min 34 s (16 vCPU, next to tests) |
| same two, no-op, before the build.rs fix | — | 3 min 25 s / 3 min 42 s |
| same two, no-op, after it | — | 1.6 s / 1.3 s |
| `FV_SERVE_HEAVY=1 FV_SERVE_UI=1 check.sh` up to its director_e2e step | — | 4 min 40 s |
| `tests/compat/run.sh` (all 10 suites, incl. clients + fv-serve build) | — | 25 min 34 s (32 vCPU, 3 other agents' jobs running) |
| fetch fv-gpucheck (56 MB) / fv-serve (77 MB) | — | 3 s each; both run `--help` here |

The no-op release rebuilds re-ran `fastvideo-cudarc`'s build script (every
nvcc cubin) each time because it watched the absent `artifacts/oxide`;
fixed in `crates/fastvideo-cudarc/build.rs`.

Target dir sizes: `check.sh` heavy + UI ~45 GB; release fv-serve +
fv-gpucheck + the CUDA check 2.6 GB; other agents' test targets 13–21 GB.

Results on main as of this run: the serve crates' check/clippy/tests pass,
and so do the heavy steps except `fastvideo-fal --test director_e2e`
(`av_session_end_to_end`, `video_only_session`: video at 0–1 fps instead of
24, also with no other job running and without `RUNPOD_*` in the env). The
console smoke failed once under `check.sh` ("video URL does not serve") and
passed as the compat `console` suite. Compat: openai, fastwan, minimax, ltx,
fal-py, fal-js, fal-webhook, console PASS; **fal-director** (video decoded at
0.25 fps) and **reactor** (only 4–12 video frames in av / video / causal)
FAIL, the same real-time video symptom as director_e2e.

Spend for these runs: about $2.25 of pod time (four pods; other agents used
them too), plus the volume at $14/month.
