# Releases, channels and deployments

What runs where, which build it is, and how to move it: build identity in
the binary, release channels in GHCR, the release history and a deployment
registry in D1, and `scripts/serve/release.sh` over all of it.

Code: `crates/fastvideo-serve/build.rs` and `src/build_info.rs` (identity),
`.github/workflows/release.yml` (promote / rollback), `scripts/serve/release.sh`
(CLI), `scripts/serve/lib/{d1,ghcr,registry}.sh` (helpers),
`deploy/d1/registry.sql` (tables), `scripts/serve/tests/release.test.sh`
(tests against a mocked API).

## Build identity

`fv-serve --version`:

```
fv-serve 0.1.0
git:      2cd1ba0c61c9b8a4a5d6f0b8a9e0e2f6c1d3b4a5
built:    2026-09-29T01:05:12Z
build-id: 5d0c3f0e9a4b7c21
profile:  release
features: cuda,http-client,fal,ltxapi,minimax,openai-videos,reactor,webrtc
variant:  h3-turbo
channel:  stable
image:    ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:c782eb37…
tag:      h3-turbo-sha-2cd1ba0
digest:   sha256:c782eb37…
```

(`-V` prints only `fv-serve 0.1.0`.) `/health`, `/healthz` carry the same as
`build` (`/` adds `git_sha`); the startup log line has it too.

| field | from |
|---|---|
| `git_sha`, `built`, `build-id`, `features`, `profile` | compiled in by `build.rs`: `FV_GIT_SHA` / `GITHUB_SHA` / `git rev-parse HEAD`; `FV_BUILD_TIME` / the commit time (not the wall clock, so one commit builds identically); `FV_BUILD_ID` / `BUILD_ID`. The image build passes them as build args (`.git` is not in the Docker context) |
| `variant` | `FV_VARIANT` at run time (baked into each variant image; the CUDA variants share one binary, so it cannot be compiled in); `FV_BUILD_VARIANT` at compile time otherwise |
| `image.ref`, `image.tag`, `image.digest`, `channel` | `FV_IMAGE_REF`, `FV_IMAGE_TAG`, `FV_IMAGE_DIGEST`, `FV_RELEASE_CHANNEL` at run time. An image cannot know its own digest: the Runpod templates and every deploy script set them (`fv_image_env_json`, scripts/serve/variants.sh) |

## Channels

| channel | tags | moved by | who follows it |
|---|---|---|---|
| `latest` | `:latest`, `:<variant>`, `:<variant>-latest` | every green main build (serve-image.yml), recorded as a `build` release | nothing by default: for trying the newest build |
| `stable` | `:stable`, `:<variant>-stable` | `release.sh promote <sha> stable` / `rollback` (release.yml) | the Runpod templates `fv-serve-<variant>-{sls,pod}`, so every template-based endpoint and pod, and the deploy scripts' default image |

Decision: `:latest` stays "the newest green main build" (what it has always
meant, and what people and older scripts expect); promotion is a separate,
explicit `:stable` channel. Main moves every few minutes, and a template
update rolls every endpoint on it, so production must not follow main.
Other channels (`canary`, …) work the same way (any lower-case word that is
not a variant name or a build tag); the templates follow exactly one,
`FV_TEMPLATE_CHANNEL` (repository variable, default `stable`; setting it to
`latest` restores the old "CI syncs the templates after each main build").

Promotion never rebuilds: `docker buildx imagetools create --tag <repo>:<channel> <repo>@<digest>`
copies the manifest index as is, so the channel tag has the build's digest
(checked after each retag). Everything references digests; tags are for
people.

Until the first promotion there is no `:stable`, and the deploy scripts
fall back to `:latest` (with a note on stderr).

## Release history (D1)

Table `releases` in the `fv-jobs` D1 database (`deploy/d1/registry.sql`,
applied idempotently by the scripts before their first write): one row per
promotion, rollback or recorded main build: `channel`, `git_sha`,
`digests` (JSON: `debug` = the all-in-one image, and one per variant),
`action` (`build` | `promote` | `rollback`), `promoted_at`, `promoted_by`
(`ci:release#<run> by <actor>`, `user:…`, `agent:…`), `notes`, `run_url`,
`templates_updated`, `source_release` (a rollback's target) and
`rolled_back_at`. The current release of a channel is its newest row. D1 is
the source of truth; there are no git tags.

**Rollback** re-promotes the newest earlier release of the channel whose sha
differs from the current one and that was not itself rolled back; the
releases after it are marked `rolled_back_at`. So successive rollbacks walk
back through history (an undo stack) instead of flip-flopping, and a new
promotion starts a new head. `--to <id>` picks a release explicitly. The
recorded digests must still exist in GHCR (checked first).

## Deployment registry (D1)

Table `deployments`: one row per Runpod resource a deploy script created,
keyed `<kind>:<runpod id>` (`kind`: `pod`, `endpoint`, `gateway`), with pool,
variant, image and digest, git sha and channel (looked up from `releases`
by digest when the script did not name them), region / DC, GPU, $/hr,
`created_at` / `ready_at` / `deleted_at`, `created_by` (`<who>/<script>`),
`status` (`creating` → `ready` → `draining` → `deleted`; `gone` when
reconcile found it missing) and `meta` (template, config, cluster).

Writers (best effort: a D1 problem logs one warning and never fails a
deploy; `FV_REGISTRY=0` turns it off): `runpod-pod.sh` (create, ready after
`/ping` 200, delete), `runpod-endpoint.sh` (create, ready after the first
job or LB `/ping`, delete), `runpod-gateway.sh` (the gateway pod; its pools
through runpod-endpoint.sh with `FV_DEPLOY_POOL`), `runpod-cluster.sh`
(gateway and workers; ready in `wait`; `roll`; `down`). Detached wall-clock
backstops do not write; reconcile catches what they deleted.

## CLI: scripts/serve/release.sh

```bash
R=scripts/serve/release.sh
$R list [--verify]                    # head of each channel + digests (--verify: GHCR tags agree)
$R history [--channel stable] [--limit 20]
$R deployed [--probe] [--json]        # live fv-serve pods/endpoints: image, sha, age, channels, drift
$R promote 2cd1ba0 stable             # also: a digest, or a tag (`promote latest stable`)
$R promote 2cd1ba0 stable --dry-run   # the plan: retags, template changes; nothing happens
$R rollback stable [--to 12]
$R redeploy h3-turbo stable           # rolling, the standing cluster (below)
$R redeploy all 3f9e2aa [--dry-run]
$R reconcile [--dry-run] [--adopt] [--fix]
$R resolve 2cd1ba0                    # the build's image set (JSON)
```

`promote` and `rollback` run where they are asked: in GitHub Actions, or
with `--local`, they retag (needs a GHCR login with `packages:write`), sync
the templates and write D1 right there; anywhere else they dispatch
`release.yml` through the GitHub API (`GH_TOKEN` with `actions:write`).
`--dry-run` always runs locally and changes nothing (GHCR, Runpod and D1 are
only read). Other flags: `--notes`, `--no-templates`, `--allow-partial` (a
build some variant images are missing from; refused by default), `--force`
(re-apply a promotion that is already current).

`deployed` shows, per live resource (pods running an `fastvideo-rs-serve`
image or named `fv-serve-*` / `fv-cluster-*` / `fv-gw-*`; endpoints whose
template runs one or named `fv-*`; nothing else in the account): which
release key its digest is (`h3-turbo`, `debug`, …), the git sha, age, the
channels whose current release contains the digest, and drift against the
channel it follows (its row's channel, its env's `FV_RELEASE_CHANNEL`, else
the templates' channel). `--probe` asks each running pod's `/health` for the
sha the binary reports. It never prints a resource's env.

`reconcile` lists the account's pods, endpoints and templates, marks rows
whose resource is gone (`status = gone`, `deleted_at`), flags live fv-serve
resources nobody recorded (`--adopt` records them, `created_by = reconcile`)
and rows whose live image differs from the recorded one (`--fix` records the
live one), and flags `fv-serve-*` templates that are not on the digest of
the channel they follow.

### Rolling redeploy

`redeploy <pool|all|gateway> [channel|sha]` (default `stable`) rolls the
standing gateway cluster of `runpod-cluster.sh` (state
`artifacts/runpod/serve/cluster.json`) through `runpod-cluster.sh roll`:

1. per pool, a second worker on the target image (single-worker pools briefly
   have two workers);
2. the gateway gets both URLs (`FV_POOL_<ID>_URLS`; its container restarts);
3. wait until the new worker's `/health` is `AVAILABLE` and reports the
   target digest (`FV_ROLL_WAIT_S`, default 1800 s; otherwise the new pods
   are deleted and the gateway is pointed back);
4. drain the old worker (`POST /fv/v1/internal/drain`, internal token):
   running work finishes, nothing new is taken;
5. wait until it is idle (`/fv/v1/internal/status`: nothing running, queued
   or in a session; `FV_DRAIN_WAIT_S`, default 900 s, then it goes anyway);
6. the gateway gets only the new URLs (second restart); `gateway` in the
   selection moves the gateway pod to the new image in the same PATCH (same
   pod id, so its URL stays);
7. delete the old worker.

A cluster started on per-variant images (`runpod-cluster.sh up sha-<commit>`
or `up stable`) rolls each pod onto the target's image of its variant
(`wan` → `wan5b`, the gateway → `gateway`) and keeps the state's `images`
in step; one started on an image reference rolls onto the target's
all-in-one image (release key `debug`). `FV_CLUSTER_IMAGE_KIND=variant|debug`
overrides.
The balance guard (`FV_CLUSTER_MIN_START`) and a deadline check (at least
`FV_ROLL_MIN_LEFT_S`, 40 min, left) apply. Serverless endpoints are not
rolled here: an endpoint on a shared template follows the template, which
`promote … stable` updates, and Runpod rolls its workers.

## Workflow and secrets

`.github/workflows/release.yml` (workflow_dispatch: `action`, `target`,
`channel`, `to`, `notes`, `templates`, `allow_partial`, `dry_run`) runs
`release.sh … --local` with the GHCR login of `GITHUB_TOKEN`. It needs:

| secret | for |
|---|---|
| `RUNPOD_API_KEY` | the template sync (already set) |
| `FV_CF_API_TOKEN` (or `CLOUDFLARE_API_TOKEN`) | D1: a Cloudflare API token with D1 edit on the account |
| `FV_CF_ACCOUNT_ID`, `FV_D1_DATABASE_ID` | optional (looked up from the token: first account, database `fv-jobs`) |

serve-image.yml records each main build as a `latest` release with the same
token (and only warns without it).

Locally the same variables come from the environment or `.env`
(`CLOUDFLARE_API_KEY` works as the token). No script prints a secret: tokens
reach curl through header file descriptors, and only named fields of Runpod
objects are ever shown.

## Tests

`bash scripts/serve/tests/release.test.sh` (run by `scripts/serve/check.sh`;
needs curl, jq, python3; no network) starts `mock_api.py` (D1 on SQLite,
Runpod REST and GraphQL, a registry, GitHub dispatch, fv-serve pods) and a
`docker` stub, then checks resolve, promote (dry run, dispatch, local,
partial builds, latest), list / history, rollback (undo stack, `--to`),
record-build, the registry writes of runpod-pod.sh, deployed, reconcile, a
rolling redeploy and a gateway move, and that no token appears in any
output or the ledger. The Rust side: `build_info` unit tests,
`tests/version.rs` (`--version` output) and the `/health` assertions in
`tests/e2e.rs`.
