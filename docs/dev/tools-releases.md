# Tools releases (prebuilt binaries on GitHub Releases)

Owner decisions 2026-10-06: the prebuilt tool binaries are distributed as
**GitHub Releases** (R2 bucket `fv-build-artifacts` retired), versioned with
**SemVer**, and **released automatically when a push to main changed them**,
after they compiled and passed their tests **on the build pod** (a
self-hosted GitHub runner): `.github/workflows/tools-release.yml`. Image
workflows assemble images from a release instead of compiling.

```bash
T=scripts/ci/tools-release.sh
$T status                 # version, input hash, newest release, the next tag
$T plan origin/main       # what tools-release.yml would do: skip, or the version/tag
$T bump feature           # only for a MINOR/MAJOR change: Cargo.toml + Cargo.lock, in the PR
$T publish origin/main    # by hand from the coordinator (plan + build --pod + upload)
$T list                   # releases, highest version first
$T resolve [--version 0.1.3] [--input-hash H] [--exact]
$T fetch tools-v0.1.0 /tmp/t serve-cpu    # download + verify sha256
$T prune --dry-run        # retention (publish runs it)
```

## What a release is

- **Tag** `tools-v<MAJOR.MINOR.PATCH>` on the source commit (the `tools-v`
  prefix keeps them apart from any other release; prune and resolve only
  touch tags with it). Prereleases from a branch head (`publish
  --prerelease`): `tools-v<version>-pre.<sha12>`; they are only ever picked
  by an exact input-hash match, never as "latest".
- **Assets**: one gzip tarball per set (`oxide gpucheck gpucheck-vast hf-fm
  serve-cuda serve-cpu serve-fake gpucheck-tests`, about 200 MB in all;
  the sets and their build settings are in docs/dev/build-pod.md "Release
  artifacts") and `manifest.json`: version, tag, input hash, inputs, source
  commit, build id/time, builder (rustc, cargo, nvcc, tileiras, glibc,
  recipe hash), settings, test summary, and per set the tarball's sha256,
  size, features, NEEDED libraries and every file's sha256/size/mode.
- **Body**: release notes (commits that touched the inputs since the previous
  release) and three machine-read lines: `input-hash:`, `source-commit:`,
  `manifest-sha256:`. Consumers check the manifest against the body, each
  tarball against the manifest, and each extracted file (and nothing else)
  against the manifest.

The release is never GitHub's "Latest" (`make_latest: false`): consumers
pick by SemVer, not by GitHub's flag or by date.

## Versioning

Every binary reports the release version: `fv-serve -V` / `--version`, the
`version` of fv-serve's `/health` (and the worker and edge routes),
`fv-gpucheck -V`. The build passes it as `FV_RELEASE_VERSION`
(crates/fastvideo-serve/build.rs, `option_env!` in fv-gpucheck); a local
build without it reports the workspace version (`[workspace.package]
version` in `Cargo.toml`). hf-fm is third-party (its version is in the
manifest).

| change | 1.0 and later | 0.x (now) |
|---|---|---|
| breaking: a tool's CLI/flags, config file format, wire/API protocol between components (fv-serve ↔ edge/fv-control, dispatch protocol version), on-disk/weights layout the tools need | MAJOR | MINOR |
| backwards-compatible feature: new model/recipe/arm, flag, endpoint | MINOR | PATCH |
| fix or perf, no interface change | PATCH | PATCH |

0.x follows Cargo's rule (`0.y` is the compatibility line).

**The release version needs no human step** (`tools-release.sh plan`):

- inputs unchanged since a release (same input hash) → **skip**: nothing
  builds;
- else the **workspace version** if it is above the highest `tools-v*`
  release (someone ran `tools-release.sh bump breaking|feature` in their PR
  for a MINOR/MAJOR change);
- else **the highest release's PATCH + 1** (0.x too).

Nothing is committed back to main: the version lives in the tag, the
manifest and the binaries. So a PATCH needs nothing; a feature or a breaking
change needs the `bump` (Cargo.toml + Cargo.lock) merged with it, and if it is
forgotten the release is a PATCH (fix by bumping in a follow-up: the next
release takes the bumped version). Release notes list the commits that
touched the inputs since the previous tag. A tag that already exists with
other inputs stops the run (a race); prereleases (`--prerelease`) are
`tools-v<next>-pre.<sha12>`.

## Inputs (when the tools "changed")

The input hash is sha256 over `git ls-tree -r` (mode, blob id, path) of
`crates/ profiles/ Cargo.toml Cargo.lock rust-toolchain.toml
scripts/gpu/cuda-13.pins third_party/cutile-rs` (the submodule commit)
`scripts/dev/release-artifacts-pod.sh` (the recipe: flags, features,
settings) and `docker/build-base.Dockerfile` (the build pod's base image,
once that lands), at a commit, so it needs no checkout. The Dockerfiles are
not inputs: with releases they only assemble images. Not covered: the
`stable` toolchain moving under an unchanged `rust-toolchain.toml` and the
unpinned hf-fetch-model; both are recorded in the manifest.

## Publishing (tools-release.yml: build, test, then publish)

On every push to main (and by hand, on main only), one run at a time; a
newer push replaces a queued run, the running one finishes:

1. **plan** (GitHub-hosted, seconds): `tools-release.sh plan $GITHUB_SHA`.
   Skip when the input hash is released; else the version above. Then the
   **builder** (`tools-release.sh pick-runner`, owner decision: build pods
   first, GitHub-hosted as the fallback):
   - `pod` when at least one self-hosted runner labelled `fv-build` is
     **online and idle**. Listing runners needs *Administration: read*,
     which a workflow's `GITHUB_TOKEN` cannot have, so it uses the
     repository secret **`FV_RUNNER_READ_TOKEN`** (a fine-grained PAT for
     this repository with only *Administration: read*); without it the
     choice is always `github`;
   - else `github`;
   - the repository variable **`FV_BUILD_RUNNER`** = `pod` | `github`
     forces it (`auto` by default).
2. **build**, the same command either way,
   `tools-release.sh build $GITHUB_SHA --local --version V --tag tools-vV`:
   - **build-pod** (`runs-on: [self-hosted, fv-build]`): on a build pod's
     runner, with its toolchain, sccache and volume caches (the fast path:
     ~8 min incremental + the gate);
   - **build-hosted** (`runs-on: vars.FV_TOOLS_HOSTED_RUNNER || ubuntu-latest`;
     set the variable to a larger runner's label to speed it up): frees the
     host's disk, then `docker run`s the command in the **build-base image**
     (`scripts/dev/build-base-tag.sh --image`, the image the pods run, so
     the same toolchain, CUDA and tools), with cold caches (expect well over
     an hour on a 4-vCPU runner). Same recipe and the same gate, so the
     release is the same either way (only absolute source paths inside the
     binaries differ).

   Then, in both:
   - every set in release mode (scripts/dev/release-artifacts-pod.sh, as
     `build-pod.sh release-artifacts` runs it) with `FV_RELEASE_VERSION=V`,
     the `fv-gpucheck nvrtc` gate (AOT cubins + oxide cubins for sm_100/120
     embedded), the glibc ≤ 2.35 check, target dir `target/gh-runner` (pod)
     or `/work/target` (hosted);
   - **test gate** (any failure: nothing is published):
     `scripts/serve/check.sh` (check + clippy + tests of the serve crates,
     target `target/gh-runner-test`, no debuginfo); the **shipped**
     gpucheck/cudarc unit-test binaries (`gpucheck-tests`) against the
     commit's sources; `-V` of the shipped fv-serve (cpu, fake) and
     fv-gpucheck must print V;
   - the staged release (tarballs, `manifest.json`, `body.md`) becomes a
     workflow artifact (3 days).
3. **publish** (whichever build ran must have passed; GitHub-hosted, the job's `GITHUB_TOKEN` with
   `contents: write`, `actions: write`): `tools-release.sh upload stage
   --dispatch`: a draft release (no tag yet) → upload → publish (creates the
   tag; a failure before that deletes the draft) → download every set and
   check every sha256 → prune → dispatch serve-image,
   gpucheck-runtime-image and vast-pytorch-image (`tools_version` = V) on
   main. A release created with `GITHUB_TOKEN` fires no `release` event for
   other workflows, so dispatching is explicit; vast-pytorch-image also
   listens to `release: published` (`tools-v*`) for releases made with a PAT
   or from a runner.

**Tokens**: none is stored on the shared pod. The build job has a read-only
`GITHUB_TOKEN` and checks out with `persist-credentials: false`; the publish
job (with write access) runs on GitHub's hosted runner. The pod's runner
holds only its own runner credentials (below).

**By hand** (`tools-release.sh publish <rev>`, the coordinator): the same
plan, `build --pod` (`build-pod.sh release-artifacts` + jobs on the pod, the
unit-test binaries run in the coordinator's container), then `upload`, with
`~/.config/fv/github_token` or `FV_GITHUB_TOKEN_FILE` (contents:write,
actions:write for `--dispatch`; read into a mode-600 header file, never on a
command line, never printed). The coordinator session's GitHub proxy refuses
release writes (2026-10-06), so this path needs another session type or
machine; `--no-upload` stops after the gate.

## Build pod runner

The build pod is a **self-hosted runner** with the label `fv-build`,
registered **once per pod** (not ephemeral: a pod lives hours and runs one
tools release at a time; `--replace` takes over the name
`fv-build-<pod id>` from an earlier pod; GitHub drops offline runners after
14 days). build-pod-server.py (`POST /v1/runner`) downloads the newest
actions/runner, checks its sha256 against the release notes, configures it
with the registration token and keeps `run.sh` running; steps see the pod's
job environment (toolchain, CUDA, sccache; no `RUNPOD_*`). A running
workflow job counts as pod activity (no idle stop; the 8 h cap still
applies) and its `target/gh-runner*` dirs are not evicted meanwhile. The
runner's credentials live in `<local>/actions-runner` on the container disk
and go with the pod. `GET /v1/runner` (and `build-pod.sh status`) shows it.

**Pod-side contract** (for whatever manages the pods: today `build-pod.sh`,
next fv-control, which will own the pods' lifecycle and keep the PAT as a
Worker secret). The pod's server (build-pod-server.py) is reached over the
pod's HTTPS proxy with the pod token (`Authorization: Bearer <pod token>`,
the same as every `/v1/*` call):

- `POST /v1/runner` with JSON `{"token": "<registration token>", "repo":
  "<owner>/<name>", "labels": "fv-build" (optional, default fv-build;
  `[A-Za-z0-9_.,-]`), "name": "<runner name>" (optional, default
  `fv-build-<RUNPOD_POD_ID>`)}` → `202 {"registering": name, "labels"}`;
  `400` on a missing token or a bad repo/labels/name. Registration runs in
  the background: download (sha256-checked), `config.sh --unattended
  --replace --disableupdate`, then `run.sh`. Posting again re-registers
  (the running runner is stopped first).
- `GET /v1/runner` → `{phase: absent|installing|configuring|running|failed|exited <rc>,
  error, name, labels, version, repo, busy}`; also under `runner` in
  `GET /v1/status`. Poll until `running` (or `failed`).
- **The token**: a *registration* token (`POST
  /repos/{owner}/{repo}/actions/runners/registration-token` with the PAT;
  valid 1 h, single use). Send that, never the PAT: the PAT stays with the
  caller. The pod passes it to `config.sh` once, keeps it out of its logs
  and error messages (replaced by `<token>`) and never writes it to disk;
  only the runner's own credentials stay in `<local>/actions-runner` on the
  container disk, gone with the pod.
- Labels: `fv-build` is what tools-release.yml targets; give pods in other
  regions or sizes extra labels if workflows should pick them, but keep
  `fv-build` on every pod that may build releases. Names must be unique per
  pod (the default is).
- While a workflow job runs on the runner the pod counts as busy (no idle
  stop; its own 8 h cap still applies). A manager that stops pods should
  check `GET /v1/runner` → `busy` (or the runner's `busy` in the GitHub
  API) first.

**Registering** (`build-pod.sh runner`; `build-pod.sh up` does it too when a
token source exists, `FV_BUILD_RUNNER=0` skips): the coordinator sends a
**registration token** (valid 1 h, used once, never written on the pod) from

- `~/.config/fv/gh-runner-token`: a token from the repository's Settings →
  Actions → Runners → *New self-hosted runner* (the `--token` value), or
- `~/.config/fv/gh-runner-pat`: a fine-grained PAT for this repository with
  **Administration: read and write**, with which `build-pod.sh` mints one
  (`POST /repos/{repo}/actions/runners/registration-token`) at every `up`.

Without either, `build-pod.sh runner` fails and says so. The build job waits
in GitHub's queue while no runner is online (up to 24 h): start the pod
(`build-pod.sh up`) after merging a tools change, or let the next `up` pick
it up.

**Safety**: only `tools-release.yml` (push to main, dispatch on main) runs on
`fv-build`; no workflow on `pull_request` may use it (a fork's code would
run on the shared pod). Keep "Require approval for all outside
collaborators" on for Actions.

## Consuming (the workflows)

`scripts/ci/prebuilt.sh fetch <sets>` picks, for the checked-out commit:

1. the repository variable **`FV_TOOLS_VERSION`** (e.g. `0.1.3`): a pin for
   rollbacks, image workflows only;
2. else the release built from **exactly this commit's tools inputs**;
3. else (the commit changed the tools since the last release):
   - image workflows **on main** (serve-image, gpucheck-runtime-image):
     the **highest SemVer** release, with a warning;
     the image carries `dev.fastvideo.tools=<tag>`, and gpucheck images
     use the release's build id for `:build-<id>` and the binary's
     `.build-id`, so validate.sh never mistakes it for this commit's binary;
   - test workflows (gpucheck-t0, serve-compat) and image workflows on
     branches, which must run their own code: **compile on the GitHub
     runner** as before (with a warning), unless a prerelease for exactly
     these inputs exists (`publish --prerelease <branch head>`).

**vast-pytorch-image is different (owner decision 2026-10-06):** it always
ships the **newest** tools release (`FV_PREBUILT_SELECT=newest`: highest
SemVer `tools-v*`, even when an older release matches the commit), or the
`tools_version` dispatch input / `FV_TOOLS_VERSION` pin, sha256-checked, and
**never compiles**: with no usable release the job fails with
`::error title=Prebuilt tools::no usable tools release …`
(`FV_PREBUILT_REQUIRE=1`). It runs when `docker/vast-pytorch.Dockerfile`
changes, by hand (`workflow_dispatch`, optional `tools_version`), and for
every new tools release (dispatch from the publisher, or the `release`
event). Tags: `:build-<release build id>`, `:tools-v<X.Y.Z>`,
`:sha-<commit>` and `:latest` (main or a release); labels
`dev.fastvideo.tools` and `dev.fastvideo.tools-commit`.

**Trade-off** (chosen for simplicity): main's image workflows never wait for
the pod and never compile, but between a tools change landing and its
release (~30–40 min, longer while the pod is down), main's images carry the
previous release's binaries (labelled). tools-release.yml dispatches the
image workflows as soon as the release is out, so they then rebuild with
it. The alternative (images wait/poll for their exact release) blocks
main's images on the pod and was not chosen.
Pull requests that change the tools compile on the runner, as they did
before (R2 artifacts for PR heads existed only when someone built them).

## Retention

`prune` (run by every stable upload) keeps the newest `FV_TOOLS_KEEP`
(default 10) stable releases plus the pinned `FV_TOOLS_VERSION`, and deletes
prereleases older than 14 days or whose version has been released, each with
its tag. It refuses anything not tagged `tools-v…`.

## Measured (2026-10-06, build pod cpu3c 32 vCPU, $0.96/hr)

`publish cbff53d --no-upload` (main, would be `tools-v0.1.0`): build 8.1 min
(incremental: oxide from the volume cache, target dir warm; first build on a
cold target dir 28 min), gate 1 `check.sh` 14.8 min (cold target dir; it is
cleaned after to spare the shared disk), gate 2 the 2 shipped test binaries
(40 + 520 tests) 7.9 min here, gate 3 seconds: about 31 min in all, ~$0.40 of
pod time. Assets: 8 tarballs, 202 MB, plus a 14 KB manifest. The release
itself could not be created from the coordinator session: its GitHub proxy
refuses release writes (see the PR). Runner time saved per workflow: docs/serve/images.md.
