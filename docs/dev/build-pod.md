# Shared build pod

Agents compile and test on one shared Runpod **CPU pod** with a 200 GB
**network volume** (`fv-build`), not in their own containers, whose disks are
small. `scripts/dev/build-pod.sh` drives it over the pod's HTTPS proxy
(`https://<pod>-8000.proxy.runpod.net`); there is no SSH. Each agent gets its
own source snapshot and `CARGO_TARGET_DIR` on the volume, so agents build in
parallel without sharing a cargo lock or clobbering each other's artifacts.

**Status (2026-09-28):** the service and client are tested locally (sync and
delta sync, allowlist, exit codes, two agents in parallel, cancel, artifact
fetch, path-escape rejection, idle self-stop trigger), and the volume toolkit
install was checked (the redist nvcc 13.4.92 compiles an sm_90 cubin). The
`fv-build` volume has not been created yet, so there are no measured pod
build times; run `build-pod.sh volume-create` once, then `up`.

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
$B status                                 # pod, $/hr, setup, running jobs, disk
$B clean $A target                        # drop your target dir when you are done
$B stop                                   # when nobody needs it (it also stops itself)
```

`run` syncs first (only files whose size or mtime changed; deletions are
mirrored), then streams the log and exits with the command's exit status.
`--no-sync` skips the sync. Leading `K=V` words set allowlisted environment
variables: `run $A -- FV_SERVE_HEAVY=1 bash scripts/serve/check.sh`.
Ctrl-C cancels the job on the pod; `log <job>` re-attaches after a dropped
connection.

## What runs where

| Path on the volume (`/workspace/fv-build`) | What |
|---|---|
| `worktrees/<agent>/` | the agent's snapshot: tracked + untracked, non-ignored files (submodules included) |
| `target/<agent>/` | that agent's `CARGO_TARGET_DIR` (incremental, kept between runs) |
| `cargo/` | shared `CARGO_HOME` (registry, git checkouts) |
| `rustup/` | shared `RUSTUP_HOME` (stable + rustfmt + clippy, per `rust-toolchain.toml`) |
| `sccache/` | shared sccache (40 GB cap): registry crates compile once for all agents |
| `cuda-13.4/` | CUDA 13.4.92 nvcc, crt, cudart, NVRTC, libnvvm, CCCL, tileiras (NVIDIA redist tarballs, sha256-checked) — the same toolkit the CI builder image pins |
| `jobs/`, `logs/`, `ledger.tsv` | job logs, service log, pod-side ledger (self-stops) |

The first boot on a fresh volume installs the toolchain onto the volume
(~200 MB of CUDA downloads, the Rust toolchain, sccache); later boots only
`apt-get install` cmake/clang/mold/pkg-config into the container (about a
minute). The image is stock `rust:1-bookworm`; nothing is baked for us.

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
scripts/serve/check.sh`, `bash scripts/gpu/lint.sh`. Environment overrides:
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
| **Pod (default): cpu5c, 32 vCPU / 64 GB** | **$1.12/hr** |
| fallback: cpu3c, 32 vCPU / 64 GB | $0.96/hr |
| cpu3g, 32 vCPU / 128 GB (`FV_BUILD_FLAVORS=cpu3g`) | $1.28/hr |
| cpu5c / cpu3c, 16 vCPU / 32 GB (`FV_BUILD_VCPUS=16`) | $0.56 / $0.48/hr |
| Volume `fv-build`, 200 GB in EU-RO-1 | $0.07/GB/month = **$14/month** |

32 vCPUs keep parallel rustc and the per-SM nvcc cubin compiles busy; 64 GB
is plenty for 32 rustc processes with LTO off. cpu5c is the newer generation
(faster per core) for 17 % more per hour; `FV_BUILD_FLAVORS` falls back to
cpu3c when cpu5c has no stock. EU-RO-1 was the datacenter reporting stock for
both at 32 vCPU.

- The pod stops itself after `FV_BUILD_IDLE_MIN` (default 20) minutes with no
  request and no running job, and `FV_BUILD_MAX_HOURS` (default 8) after boot
  regardless. Both run on the pod, so they fire even if every agent container
  is gone. It uses the pod-scoped `RUNPOD_API_KEY` Runpod injects; `status`
  shows `self_stop_key`. If Runpod refuses to *stop* a pod with a network
  volume, the pod terminates itself instead; nothing is lost, since all state
  is on the volume, and the next `up` creates a new pod.
- `up` refuses to run below a $8 balance (`FV_MIN_BALANCE`) and deletes a pod
  created above `FV_BUILD_MAX_DPH` (default $1.50/hr).
- Ledger: `~/.config/fv-build/ledger.tsv` (local) and `ledger.tsv` on the
  volume (pod-side stops).

## Limits

- **Disk:** 200 GB total. Expect (not yet measured) 10–20 GB per CUDA release
  target dir and a few GB per debug `check.sh` target, so roughly 8–10 active
  agents next to the caches. Target dirs untouched for 14 days (`FV_BUILD_TARGET_TTL_DAYS`) are
  pruned at boot; `clean <agent>` frees one now; `agents --sizes` shows usage.
- **Network volume I/O** is slower than local NVMe; incremental rebuilds are
  still far cheaper than cold ones, and sccache absorbs most dependency cost
  for new agents.
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
  caches); `up` prints a note when the pod runs an older server.
- `FV_SERVE_UI=1` (headless Chromium) is not supported on the pod.
