# Shared build pod

Agents compile and test on one shared Runpod **CPU pod** with a 200 GB
**network volume** (`fv-build`), not in their own containers, whose disks are
small. `scripts/dev/build-pod.sh` drives it over the pod's HTTPS proxy
(`https://<pod>-8000.proxy.runpod.net`); there is no SSH. Each agent gets its
own source snapshot and `CARGO_TARGET_DIR` on the pod's container disk, so
agents build in parallel without sharing a cargo lock or clobbering each
other's artifacts; the toolchains and the compile cache live on the volume.

**Status (2026-10-06):** in use; the pod now runs the prebuilt base image
([Base image](#base-image)) with persistent dependency caches ([Caches](#caches)).
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
$B status                                 # pod, $/hr, image, sccache hit rate, cache sizes, deps seeds, jobs, disk,
                                          # self-stop timers, per-agent sizes + eviction
$B seed $A                                # build the deps seed of your Cargo.lock now (jobs start one on their own)
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
| image `ghcr.io/zaitrarrio/fastvideo-rs-build-base:<tag>` | toolchains and tools (see [Base image](#base-image)): Rust, CUDA 13.4, sccache, mold, clang, ffmpeg, Node + Playwright + Chromium |
| container disk `/root/fvb/worktrees/<agent>/` | the agent's snapshot: tracked + untracked, non-ignored files (submodules included) |
| container disk `/root/fvb/target/<agent>/` | that agent's `CARGO_TARGET_DIR` (incremental while the pod runs; seeded from a deps seed when created; evicted when unused, see Limits) |
| container disk `/root/fvb/.last-use/<agent>` | stamp of the agent's last sync, job or fetch (what eviction goes by) |
| container disk `/root/fvb/cargo/` | `CARGO_HOME`; its `registry/cache` and `git/db` link to the volume |
| volume `cargo/registry/cache`, `cargo/git/db` | downloaded crates and git checkouts (shared) |
| volume `sccache/` | shared sccache (40 GB cap), every rustc, cc-rs and CMake compile goes through it |
| volume `deps-seed/<key>.tar.zst` | prebuilt dependencies per Cargo.lock + toolchain + image (see [Caches](#caches)) |
| volume `release-cache/` | `release-artifacts` oxide cubins and hf-fm |
| volume `jobs/`, `logs/`, `ledger.tsv` | job logs, service log, pod-side ledger (self-stops) |
| volume `rustup/`, `cuda-13.4/`, `tools/`, `node-v22.23.3/`, `playwright-1.56.1/`, `pw-browsers/` | what pods on `rust:1-bookworm` installed at boot; unused since the base image (the owner may delete them) |

**Why targets are not on the volume:** the first pod kept them there. Cargo
on the network filesystem took 18 min for a cold `cargo check` of the serve
crates and 30 s for a no-op one (fingerprint stats), and a full snapshot sync
took 63 s to extract. On the container disk with the volume's sccache: 3.5 min,
1.4 s and 5 s. A stopped or recreated pod starts with empty target dirs; the
deps seed refills the dependencies in one sequential read and sccache the
rest. `FV_BUILD_TARGETS=volume` in the pod env restores the old layout.

## Base image

The pod runs `ghcr.io/zaitrarrio/fastvideo-rs-build-base:<tag>`
(`docker/build-base.Dockerfile`) and **installs nothing when it boots**: no
apt, no rustup, no downloads. The server checks that the tools are there and,
if one is missing, the setup fails with `missing from the image <image>:
<tools>` (browser tools: compat / console / `FV_SERVE_UI=1` jobs fail with
`browser extras unavailable: …`; build jobs still run).

| In the image | Version (pinned in the Dockerfile) |
|---|---|
| Ubuntu 22.04 | glibc 2.35, the same as the published runtime images |
| Rust | `RUST_VERSION` (1.99.0) + what `rust-toolchain.toml` lists (rustfmt, clippy, wasm32); `RUSTUP_TOOLCHAIN` pins it, so `channel = "stable"` never makes the pod download a toolchain |
| CUDA | 13.4.92 nvcc / NVRTC / tileiras from `scripts/gpu/cuda-13.pins` (the CI builder's pins) + cudart / driver / cuRAND headers for oxide's bindgen |
| sccache, mold | 0.18.0, 3.0.0 (release binaries, sha256-checked) |
| build tools | build-essential, clang + libclang, cmake, pkg-config, libssl-dev, git, jq, binutils, zstd, xz |
| tests | python3 + venv, ffmpeg 4.4 with libvpx, Node v22.23.3, Playwright 1.56.1 + its Chromium and system libraries (`PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`, `NODE_PATH=/opt/playwright/node_modules`) |

`/etc/fastvideo/build-base.json` lists the versions; `status` prints it.

**Tag = content hash.** `scripts/dev/build-base-tag.sh` hashes the
Dockerfile, `scripts/gpu/cuda-13.pins` and `rust-toolchain.toml` into
`bb-<16 hex>`; `build-pod.sh` pins it (`BASE_IMAGE_TAG`). The
`build-base-image` workflow runs when one of those files (or the pin) changes,
builds and pushes the tag only if GHCR has no image with it (packages only,
nothing of ours compiles), smoke-tests it (every tool, `rustup target list`,
CUDA headers, libvpx, a headless Chromium launch) and fails if the pin and the
hash differ. To change the image: edit the Dockerfile (e.g. bump
`RUST_VERSION`), `bash scripts/dev/build-base-tag.sh --pin`, push, wait for
the workflow, then `down` + `up` the pod (`up` recreates a stopped pod whose
image differs from the pin and notes a running one).

The image is not used as the CI builder stage (`docker/gpucheck.Dockerfile`,
`cuda-builder.Dockerfile`): with prebuilt R2 binaries those stages rarely
run, pulling a 1.6 GB (compressed) image to replace a cached layer is not
faster, and
coupling CI's toolchain to the pod's pin is a separate decision.

## Caches

Rust dependencies are rebuilt only when `Cargo.lock`, the toolchain (image)
or the job flags change:

1. **sccache** (`RUSTC_WRAPPER`, `sccache/` on the volume, 40 GB). The server
   starts the sccache daemon itself at boot.
   *Our own crates do not hit across agents* (tried 2026-10-06): sccache
   0.18 hashes a rustc call's cwd, its arguments and every `CARGO_*` variable
   (`CARGO_MANIFEST_DIR`, `CARGO_MANIFEST_PATH`) verbatim, and its
   `SCCACHE_BASEDIRS` path stripping applies to C/C++ only (`src/compiler/rust.rs`
   has no basedirs). One daemon per agent with `SCCACHE_BASEDIRS=<worktree>:<target>`
   gave 0 hits for the 18 workspace crates of a new agent's release fv-serve
   (210 s, against 201 s with one shared daemon), so it was dropped. A stable
   path per agent would need a mount namespace per job (`CAP_SYS_ADMIN`),
   which a Runpod container does not have; `--remap-path-prefix` does not
   change cwd or `CARGO_MANIFEST_DIR`. The deps seed covers the
   dependencies; our crates compile once per agent, then incrementally.
   Before, the first job's rustc spawned it inside that job's process group, so cancelling that job
   (`killpg`) killed the cache server under everyone else's build. cc-rs
   (OpenH264, Opus, libwebp …) uses sccache because `RUSTC_WRAPPER` is
   sccache, and CMake builds get `CMAKE_{C,CXX}_COMPILER_LAUNCHER`. `status`
   prints `sccache: <hits> hits / <n> cacheable compiles (hit rate …)` since
   boot. Not cacheable by design: incremental
   (debug workspace) crates, workspace crates of another agent (above),
   proc-macros/bins/dylibs, build-script runs, nvcc in `fastvideo-cudarc`'s
   build script.
   *Audit of the old setup:* `RUSTC_WRAPPER` was set only if the volume's
   `tools/sccache` existed (a failed download disabled it silently, "sccache
   is optional"); `status` showed only that boolean, never a hit rate; the
   server lived in a job's process group; C/CMake compiles were not
   explicitly routed. It was used for rustc, but nothing reported how well.
2. **Deps seeds** (`deps-seed/<key>.tar.zst`, key = sha256 of `Cargo.lock`,
   `rustc -vV`, the image, the seed recipe and the job flags). A seed is a
   target dir in which every registry / git dependency of the common
   commands is built (`SEED_RECIPE` in `build-pod-server.py`: `cargo check
   --all-targets` and `cargo test --no-run` of `check.sh`'s crates, the CUDA
   type-check, release `fv-serve --features cuda,http-client` and
   `fv-gpucheck --features cuda`). Every **path** package (workspace members,
   vendored crates) is stripped out, fingerprints, outputs and incremental
   state included: cargo judges their freshness by mtimes, which a copied
   snapshot cannot be trusted with. Registry and git units are keyed by
   version, features, profile and flags, so a seeded unit is only reused when
   cargo would have produced the same one.
   - *Use:* when a job starts and its agent has **no target dir** (new agent,
     pod restart, eviction, `clean`), the server extracts the seed of that
     worktree's `Cargo.lock` into it (local container disk; one sequential
     zstd read from the volume) and notes `[deps seed] …` in the job log.
     Without a seed, or without the disk space, the job just builds (sccache).
     Existing target dirs are never touched.
   - *Build:* the first successful job on a `Cargo.lock` with no seed starts
     one in the background (job agent `fv-seed`, `nice 15`, half the vCPUs,
     one at a time; `build-pod.sh seed <agent> [--force]` starts it by hand).
     It copies that agent's snapshot, runs the recipe (`--keep-going`:
     failing workspace crates do not matter), strips, packs to
     `<key>.tar.zst.part-…` and renames. The newest `FV_BUILD_SEED_KEEP` (4)
     seeds by last use are kept. `FV_BUILD_SEED=0` turns seeds off.
   - Release-artifacts builds (`fv-release`) run without mold's RUSTFLAGS, so
     their units differ from the seed's; they rely on their own target dir
     and sccache as before.
3. **Cargo registry** (`cargo/registry/cache`, `git/db` on the volume):
   crates download once; the per-pod `registry/src` extraction is local.

Eviction (#22) only ever removes `target/<agent>` and `worktrees/<agent>` on
the container disk; `sccache/`, `deps-seed/`, `cargo/` and `release-cache/`
on the volume are never evicted (seeds are only replaced by the keep-4 rule).
A seed build's agent `fv-seed` is busy while it runs (never evicted) and
removes its own dirs when done. `status` shows `caches_gb` (sccache,
cargo_registry, deps_seed, release_cache, local_registry_src; a du every 15
min) and the seeds.

Jobs get: `CUDARC_CUDA_VERSION=13000`, `NVCC` and `CUDA_HOME` pointing at the
image's toolkit (so `--features cuda` builds compile the AOT cubins),
`RUSTC_WRAPPER=sccache`, mold as the linker, and the CI builder's release
overrides (`CARGO_PROFILE_RELEASE_LTO=off`, `CODEGEN_UNITS=16`,
`PANIC=unwind`) so release builds match the published images and take minutes,
not the fat-LTO hour. Pass `CARGO_PROFILE_RELEASE_LTO=fat
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort` for a
workspace-profile build.

## Release artifacts (prebuilt binaries for the image workflows)

Owner decisions 2026-10-06: **the build pod compiles and tests, GitHub
Releases hand off, GitHub assembles** (R2 is no longer used). Every Rust
binary and the oxide cubins the images and CI jobs need are compiled here,
tested, and published as a SemVer **tools release** (`tools-v<X.Y.Z>`); the
GitHub workflows download it and only run the lean Docker stages (and the
tests), no `cargo`. The pod has no Docker daemon, so the images themselves
are still assembled on GitHub. Publishing, versioning, retention and how the
workflows pick a release: **docs/dev/tools-releases.md**.

```bash
bash scripts/ci/tools-release.sh status            # does HEAD need a release / a bump?
bash scripts/ci/tools-release.sh publish origin/main   # build + test + publish (coordinator)
B=scripts/dev/build-pod.sh
$B release-artifacts <sha>         # build + verify only, into artifacts/release/<sha>/ ($FV_RELEASE_OUT)
$B release-artifacts <sha> --sets "serve-fake gpucheck-tests"   # a subset
```

What `release-artifacts` does:

1. Resolves the revision to a full sha. One run per container at a time
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
   older shas build too; `tools-release.sh publish` insists it is the
   commit's own copy (the recipe is one of the release's inputs).
4. Fetches the tarballs and `manifest.json` and checks every tarball's
   sha256. `tools-release.sh publish` then runs the tests and uploads them.

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
13.4.92 (the base image's, from the same apt pins as the CI builder), the
Rust of the base image (`RUST_VERSION`; CI's builder takes the `stable` of
the day), **no RUSTFLAGS** (the script drops the pod's mold link-arg, so the
system linker links, as in the image), and `FV_GIT_SHA` / `FV_BUILD_TIME` /
`BUILD_ID` as the workflows pass them, so `fv-serve --version` reports the
same build. The base image is Ubuntu 22.04 (glibc 2.35) like the images; the
script still fails if any ELF needs a `GLIBC_` symbol newer than 2.35 (pods
on `rust:1-bookworm` had glibc 2.36). The oxide build's libclang and
cuda.h / curand.h come from the image; the script installs nothing and fails
naming a missing tool (on a redist toolkit without cuRAND it still unpacks
the `libcurand` redist headers into `release-cache/` and an overlay).
`builder.host` in the manifest names the image.

**Manifest** (`manifest.json`, the release's copy adds the version, tag,
input hash, source commit and test summary; docs/dev/tools-releases.md):
sha, build_id, build_time, run_id, builder (rustc -vV, cargo, nvcc, tileiras,
glibc, recipe hash), settings, and per set: tarball, sha256, size, features,
NEEDED libs, and every file's sha256/size/mode. Each `<set>.tar.gz` is a
reproducible gzip tar (sorted, owner 0, mtime = commit time).

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
| cpu5c / cpu3c, 16 vCPU / 32 GB (automatic fallback, `FV_BUILD_VCPUS_FALLBACK`) | $0.56 / $0.48/hr; container disk at most 120 / 80 GB, so the fallback asks for `FV_BUILD_CONTAINER_GB_FALLBACK` (80) and, if Runpod names a lower cap, retries with it |
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
  container disk has less than `FV_BUILD_EVICT_FREE_GB` free (default a fifth
  of the container disk, at most 40 GB: 40 on 200 GB, 16 on an 80 GB fallback
  pod; the server applies the same cap to the actual disk), target dirs
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
  large-file caches (sccache, `.crate`s, deps seeds) live there.
- **Concurrency:** 4 jobs at once (`FV_BUILD_MAX_JOBS`), one per agent; more
  queue. Two agents building release CUDA binaries at once share 32 vCPUs.
- **Transfers** go through the Runpod HTTPS proxy: uploads up to 512 MiB per
  sync (a full snapshot of this repo is ~40 MB, later syncs are deltas);
  artifacts up to 2 GiB, gzip-compressed in transit.
- **No GPU:** CUDA code compiles (nvcc cubins, NVRTC sources, type-checks)
  but never runs here. Kernel and model runs stay on GPU pods
  (`scripts/gpu/runpod-http.sh`).
- **Image updates:** a new base image tag reaches the pod at its next
  creation, like the server: `up` recreates a *stopped* pod whose image
  differs from the pin and prints a note for a running one (`down` + `up`
  when no jobs run; the volume keeps every cache, but seeds and sccache
  entries are keyed by the toolchain, so a Rust bump rebuilds once).
- **Server updates:** the service code is sent at pod creation (so is the
  start command with its curl watchdog). The self-stop fix of 2026-10-02
  (server `957ae128b12e` → the next sha) reaches the build pod at its next
  creation: `up` recreates a stopped pod whose server is older. After
  changing `build-pod-server.py`, `down` then `up` (the volume keeps all
  caches); `up` prints a note when the pod runs an older server, and
  recreates (rather than starts) a *stopped* pod whose server is older.
  `down` kills other agents' running jobs: check `status` first.

## Measured: base image and caches (2026-10-06)

Two throwaway pods on the `fv-build` volume, each with its own root
(`FV_BUILD_POD_ENV='{"FV_BUILD_ROOT": "/workspace/fvb-test-…"}'`, deleted
afterwards; the live pod was not touched), 16 vCPU / 32 GB, 80 GB container
disk, same commit (main `cbff53d`). *Before*: `rust:1-bookworm` + main's
server. *After*: `fastvideo-rs-build-base:bb-b49b814e45f9f7d2` (1.6 GB
compressed) + this server. Both cold builds start from an empty sccache
and registry. Times are client wall clock (`build-pod.sh run`, sync
included).

| | before | after |
|---|---|---|
| pod create → `ready` (first boot on an empty root) | 122 s, then the browser extras (apt ffmpeg, Node, Chromium) in the background | 79 s, nothing after (pull 1.6 GB + start; the server's setup takes 0 s) |
| pod create → `ready` (warm root / image cached elsewhere) | 47 s + extras 1–2 min | 63–74 s (image pull on a new host each time) |
| cold release `fv-serve --features cuda,http-client` | 416 s | 220 s |
| cold release `fv-gpucheck --features cuda` (after fv-serve) | 172 s | 91 s |
| incremental (touch `main.rs`), fv-serve / fv-gpucheck | 15 s / 16 s | 10 s / 11 s |
| new agent (empty target dir, as after a pod restart or eviction), fv-serve / fv-gpucheck | 447 s / 153 s (warm sccache) | 187–206 s / 80–82 s (deps seed: 3.6 GB extracted in 4–6 s, then 18 crates compiled instead of 60+) |
| after adding one dependency (`humansize` to fastvideo-serve) | 38 s | 33 s |
| new agent, one sccache daemon per agent with `SCCACHE_BASEDIRS` (tried, dropped) | — | 210 s, 0 sccache hits on the 18 workspace crates (shared daemon: 201 s) |
| deps seed build (background, `nice`, 8 jobs) | — | 326 s; 3.6 GB → 0.88 GB zstd |
| sccache hit rate | not reported | 0 % on the cold pod; 92 % (6265 / 6836) on a second pod over the same volume |

Notes. The cold "after" builds are faster mainly because the before pod
compiled while its extras phase ran apt and downloads, and mold/rustc differ
(1.98 vs 1.99). With a seed, what remains for a new agent is the workspace
crates themselves (release fv-serve's ~3 min at 16 vCPU): sccache keys them
by the agent's absolute path (`CARGO_MANIFEST_DIR`), so they do not hit
across agents. In the "before" new-agent run the warm sccache on the network
volume did not beat the cold build at all. Two of the "after" runs were on a
cpu5c pod (16 vCPU, $0.56/hr) because cpu3c had no stock. The first seed run
showed a stripping bug (a test target `tests/serde.rs` named like the
registry crate `serde` took serde's rlib out of the seed, and 60 crates
recompiled); stripping now goes by the unit hashes of the path packages'
fingerprints, with a unit test. Spend: about $1.1 of pod time over both rounds
(the `SCCACHE_BASEDIRS` trial included); the test roots were removed afterwards
(checked: only `fv-build/` is left at the volume root).

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
