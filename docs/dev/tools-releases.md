# Tools releases (prebuilt binaries on GitHub Releases)

Owner decisions 2026-10-06: the prebuilt tool binaries are distributed as
**GitHub Releases** (R2 bucket `fv-build-artifacts` retired), versioned with
**SemVer**, and published only after they compiled and passed their tests on
the build pod. Image workflows assemble images from a release instead of
compiling.

```bash
T=scripts/ci/tools-release.sh
$T status                 # version, input hash, newest release; "ready" / "bump first" / "released"
$T bump fix               # or feature / breaking: Cargo.toml + Cargo.lock (commit + merge it)
$T publish origin/main    # coordinator: build on the pod, test, publish tools-v<version>
$T list                   # releases, highest version first
$T resolve [--version 0.1.3] [--input-hash H] [--exact]
$T fetch tools-v0.1.0 /tmp/t serve-gateway    # download + verify sha256
$T prune --dry-run        # retention (publish runs it)
```

## What a release is

- **Tag** `tools-v<MAJOR.MINOR.PATCH>` on the source commit (the `tools-v`
  prefix keeps them apart from any other release; prune and resolve only
  touch tags with it). Prereleases from a branch head (`publish
  --prerelease`): `tools-v<version>-pre.<sha12>`; they are only ever picked
  by an exact input-hash match, never as "latest".
- **Assets**: one gzip tarball per set (`oxide gpucheck gpucheck-vast hf-fm
  serve-cuda serve-gateway serve-fake gpucheck-tests`, about 200 MB in all;
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

The tools version is the workspace version (`[workspace.package] version` in
`Cargo.toml`), so every binary already reports it: `fv-serve -V` /
`--version`, `fv-gpucheck -V`, and `version` in fv-serve's `/health`. hf-fm is
third-party (its own version is in the manifest).

| change | 1.0 and later | 0.x (now) |
|---|---|---|
| breaking: a tool's CLI/flags, config file format, wire/API protocol between components (fv-serve ↔ edge/fv-control, dispatch protocol version), on-disk/weights layout the tools need | MAJOR | MINOR |
| backwards-compatible feature: new model/recipe/arm, flag, endpoint | MINOR | PATCH |
| fix or perf, no interface change | PATCH | PATCH |

0.x follows Cargo's rule (`0.y` is the compatibility line). The first release
is `tools-v0.1.0`. Commit subjects here are not Conventional Commits, so the
bump is explicit: whoever lands a change to the tools' inputs (or the
coordinator before publishing) runs `tools-release.sh bump
breaking|feature|fix` and commits `Cargo.toml` + `Cargo.lock`; the release
notes list the commits since the previous tag. `publish` checks it:

- inputs unchanged since a release (same input hash) → nothing to publish;
- inputs changed and the version is not above the highest release →
  **refused** ("bump it");
- the tag exists → **refused**.

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

## Publishing (build, test, then publish)

`tools-release.sh publish <rev>` runs in the coordinator's container, which
already drives the build pod and holds the token; the pod never sees it.

1. Checks: stable releases only from commits on `origin/main` (`--prerelease`
   for others); the local recipe equals the commit's; the version/tag rules
   above.
2. **Build**: `build-pod.sh release-artifacts <sha>` (agent `fv-release`, or
   `$FV_RELEASE_AGENT`): every set in release mode on the pod, the
   `fv-gpucheck nvrtc` gate (AOT cubins + oxide cubins for sm_100/120
   embedded), the glibc ≤ 2.35 check; fetched and sha256-checked.
3. **Test gate** (any failure: nothing is uploaded):
   `scripts/serve/check.sh` (check + clippy + tests of the serve crates) on the
   pod in its own target dir (agent `<release agent>-test`); the **shipped**
   gpucheck/cudarc unit-test binaries (`gpucheck-tests` set) run here against
   the commit's sources (`prebuilt.sh run-tests`); `-V` of the shipped
   fv-serve (gateway, fake) and fv-gpucheck must print the version.
4. **Publish**: a draft release (no tag yet) → upload the tarballs and
   `manifest.json` → publish (creates the tag). Any failure deletes the
   draft, so a half-uploaded set is never visible.
5. **Verify**: download every set as a consumer would and check every sha256.
6. **Prune** (below), and with `--dispatch` start serve-image,
   gpucheck-runtime-image and vast-pytorch-image (with `tools_version` =
   the new version) on main (needs `actions:write`), so the images pick up
   the release right away. Releases created with a workflow's
   `GITHUB_TOKEN` fire no `release` event for other workflows, so every
   publisher (this script, and any future publish workflow) must dispatch
   explicitly; vast-pytorch-image also listens to `release: published`
   (`tools-v*` only) for releases created with a PAT or from a runner.

**Token**: `~/.config/fv/github_token` (mode 600) in the coordinator
container, or `FV_GITHUB_TOKEN_FILE`; it needs `contents:write` on this
repository (and `actions:write` for `--dispatch`). The script reads it into
a mode-600 header file that curl reads (never on a command line, never
printed, removed on exit) and sends it only to `api.github.com` /
`uploads.github.com`. It is not copied to the build pod or anywhere else.
The workflows only read, with their own `GITHUB_TOKEN`.

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
     branches, which must run their own code: **compile on the runner** as
     before (with a warning), unless a prerelease for exactly these inputs
     exists (`publish --prerelease <branch head>` from the coordinator).

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

**Trade-off** (chosen for simplicity): main never waits for the pod and never
compiles, but between a tools change landing and its release, main's images
carry the previous release's binaries (labelled). `publish --dispatch`
rebuilds them as soon as the release is out; the coordinator runs `publish`
after merging a tools change. The alternative (images wait/poll for their
exact release) blocks main's images on the coordinator and was not chosen.
Pull requests that change the tools compile on the runner, as they did
before (R2 artifacts for PR heads existed only when someone built them).

## Retention

`prune` (run by every stable publish) keeps the newest `FV_TOOLS_KEEP`
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
