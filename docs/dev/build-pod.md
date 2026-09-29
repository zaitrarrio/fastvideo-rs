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
$B status                                 # pod, $/hr, setup, jobs, disk, per-agent sizes + eviction
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

Jobs get: `CUDARC_CUDA_VERSION=13000`, `NVCC` and `CUDA_HOME` pointing at the
volume's toolkit (so `--features cuda` builds compile the AOT cubins),
`RUSTC_WRAPPER=sccache`, mold as the linker, and the CI builder's release
overrides (`CARGO_PROFILE_RELEASE_LTO=off`, `CODEGEN_UNITS=16`,
`PANIC=unwind`) so release builds match the published images and take minutes,
not the fat-LTO hour. Pass `CARGO_PROFILE_RELEASE_LTO=fat
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_PANIC=abort` for a
workspace-profile build.

## Allowlist

The service runs no shell. Allowed: `cargo check|build|test|clippy|fmt|doc|
tree|metadata` (not `--target-dir` / `--manifest-path`), `bash
scripts/serve/check.sh`, `bash scripts/gpu/lint.sh`, `bash
tests/compat/run.sh`, `bash tests/console/run.sh`. Jobs see no `RUNPOD_*`
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

- The pod stops itself after `FV_BUILD_IDLE_MIN` (default 20) minutes with no
  request and no running job, and `FV_BUILD_MAX_HOURS` (default 8) after boot
  regardless. Both run on the pod, so they fire even if every agent container
  is gone. It uses the pod-scoped `RUNPOD_API_KEY` Runpod injects (verified:
  every pod had `RUNPOD_API_KEY` and `RUNPOD_POD_ID`, `status` shows
  `self_stop_key: true`; the idle stop itself has not fired on a real pod
  yet, since the pod was never idle for 20 min). Runpod does *stop* a pod
  with a network volume (verified); the terminate fallback stays for safety.
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
  touched. Each eviction goes to `logs/pod.log` and `status`
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
- **Server updates:** the service code is sent at pod creation. After
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
