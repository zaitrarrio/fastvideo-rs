# fv-serve images: one CUDA image and one CPU image, published to GHCR and Runpod

Date: 2026-09-28; shared layer stack 2026-10-06 ([Lean runtime
images](#lean-runtime-images-2026-10-06)); one image with boot-time config
2026-10-25 ([below](#one-image-with-boot-time-config-2026-10-25); before it,
CI built seven variant images, one per baked config). Build:
`docker/gpucheck.Dockerfile` (stages `base-os`, `base-cuda` … `serve-cuda`,
`serve-cpu`, `runtime`, `serve`), CI: `.github/workflows/serve-image.yml` +
`scripts/serve/ci-images.sh` + `scripts/ci/base-images.sh`, Runpod
publishing: `scripts/serve/runpod-templates.sh`, image table:
`scripts/serve/variants.sh` (generated from `deploy/deploy.toml` `[[images]]`).

## Images

All tags live in the one public package `ghcr.io/zaitrarrio/fastvideo-rs-serve`
(a new package per variant would start private on GHCR, and Runpod and the
deploy scripts pull anonymously).

<!-- BEGIN GENERATED from deploy/deploy.toml by control/gen-configs.mjs; do not edit -->

| image | fv-serve | carries under `/etc/fv` (`FV_WORKER_CONFIG`) | GHCR tags | Runpod templates |
|---|---|---|---|---|
| cuda | CUDA (every preset) | `runpod-fake.toml` | `:cuda-sha-<sha>`, `:cuda` / `:cuda-latest`, `:cuda-stable` | `fv-serve-cuda-sls`, `fv-serve-cuda-pod` |
| cpu (CPU only, fake engine; `gateway` until 2026-10-06) | CPU only (fake engine) | `runpod-fake.toml` | `:cpu-sha-<sha>`, `:cpu` / `:cpu-latest`, `:cpu-stable` | `fv-serve-cpu-pod` |

<!-- END GENERATED -->

Neither image bakes a worker config: every launch sends one at boot
(`FV_WORKER_TOML_B64`, the config inline, or `FV_WORKER_CONFIG`, a file in the
image), and an image started without one refuses to boot. The only config
each carries is `/etc/fv/runpod-fake.toml` (fake engine), for CI's no-GPU
smoke check, which names it explicitly. `<sha>` tags are immutable. Two release channels move over them
([releases.md](releases.md)): `latest` (`:latest`, `:<image>`,
`:<image>-latest`) is the newest green main build, moved by CI;
`stable` (`:stable`, `:<image>-stable`) moves only when a build is
promoted (`scripts/serve/release.sh promote <sha>`), and is what deploys
and the Runpod templates follow. Deploy tooling pins digests. Every image's
`fv-serve --version` and `/health` (`build`) report its git sha and build
time; the templates' env adds the image digest and channel. A worker
reports the same in its internal status and `/health`; fv-control's
Releases page shows each cluster pod's build against its channel.

### Layers

Every runtime image is one stack; the two lower parts are published once per
content hash and reused by every build (see [Lean runtime
images](#lean-runtime-images-2026-10-06)):

```
ubuntu:22.04 (pinned by digest)                     every image
base-os:   ca-certificates + libssl3 + libx264/libvpx/libdav1d  every image
           ffmpeg + ffprobe (minimal FFmpeg 4.4; TLS + libsrt for RTMPS/SRT outputs)
base-cuda: cuBLASLt                                  every CUDA image
           cuDNN precompiled engines                 (4 layers, pulled in parallel,
           cuDNN core (graph, ops, cnn, heuristic,    unchanged until a CUDA pin
             runtime-compiled + tensor-IR engines)    or the ffmpeg build changes)
           cuBLAS + NVRTC (+ builtins) + cudart
           ld.so config
 ├─ serve-cuda: fv-serve --features cuda,http-client, python3 + curl + scripts/gpu/r2-fetch.py (weights_source r2), runpod-fake.toml, FV_VARIANT=cuda
 │    + llama-server (llama.cpp, CUDA; fastvideo-rs-runtime:llama-server-<hash>, 82 MB) + fv-upsampler: the local
 │      prompt upsampler sidecar, docs/serve/upsampler-local.md
 ├─ runtime: sshd/rsync/curl, CUPTI, hf-fm, scripts/gpu, oxide, fv-gpucheck
 │   └─ serve (debug): + configs, deploy/vast/worker.py, fv-serve
base-os ─ serve-cpu: fv-serve --features http-client (no CUDA library)
```

One CUDA image serves every preset, so Runpod's host image cache covers
every family at once (before 2026-10-25 six CUDA variants differed only in
a last layer of a few KB); `fastvideo-rs-runtime` and the
debug image add only their thin top layers to the same base. The `cpu` image (fv-control's
fake-engine workers) shares the OS and ffmpeg layers and has no CUDA
library at all (the smoke check asserts it).

**Per-family binaries.** `fastvideo-serve` has no per-family cargo features
(`cuda` pulls every `fastvideo-cudarc` pipeline: Wan, LTX-2, H3, MMAudio,
upscalers, shared kernels). Splitting it would mean cfg-gating the engine
service and the pipelines; the binary is ~30 MB compressed against ~1 GB of
CUDA libraries, so the win would be small and one shared binary layer is
cached once. The worker config, the only thing the variant images
differed in, is now sent at boot.

**Entrypoint** (`/opt/fastvideo-rs/bin/fv-boot`, `crates/fastvideo-deploy`
`boot`, [ADR-0014](../adr/0014-fv-boot.md); it replaced
`deploy/runpod/fv-entry.sh`): the host driver check, links
`/workspace/weights` to `/runpod-volume/weights` on serverless workers (the
volume's HF-cache trees link absolutely into `/workspace/weights`), defaults
`FV_WEIGHTS` to the mounted root, materializes the config (inline
`FV_WORKER_TOML_B64`, else `--config`, else `FV_WORKER_CONFIG`; with none of
them it **refuses to boot**, exit code non-zero, naming the remedy; a baked
`FV_CONFIG` from an old image is not read) and
checks it with `fv-serve --print-config`, runs the weights plan
(`FV_WEIGHTS_SOURCE`: volume, r2, hub, none), the upsampler sidecar, a GMI /
Brev pod's tunnel and the watchdog, and `exec`s fv-serve. Arguments pass
through (`--config …` still overrides; `--version`, `--help` and
`--print-config` go straight to fv-serve). The `serve` (debug) and `cpu`
images use it too.

### Weights are not in the images

Weights stay on the network volume (EU `bz0ofkfx45`, `fv-weights-h3-ltx-500`,
EU-RO-1, H3 + LTX trees only, `scripts/gpu/volumes.sh`; everything else comes
from the R2 bucket `fv-weights`; the US volume `s2k01690bi` was deleted
2026-10, EU only, docs/ops/runpod-volumes.md):

- Size. The H3 load views ~72 GB of DiT weights next to a 26 GB FP8 text
  encoder tree, LTX-2.5 has a 13 GB FP8 Gemma tree
  (docs/gaps/2026-09-27-cold-start.md); even the 1.3B Wan tree carries its
  UMT5 text encoder (~11 GB). Runpod caps a GitHub-built image at 80 GB, and
  a pull of tens of GB from GHCR onto a fresh host (the ~6 GB runtime image
  took up to 458 s end to end on a fresh H200 host) takes far longer than the
  volume loads measured with E12 (LTX 33 s, H3 57 s).
- Every GPU host that runs any variant would have to pull its weights before
  the first job, and each weight change would republish multi-GB layers.
- Runpod's **cached models** (the one mechanism that puts weights on the host
  outside the image) only takes a Hugging Face model id, one per endpoint,
  mounted at `/runpod-volume/huggingface-cache/hub/`. Our trees are not all
  on the Hub as served (the E13 FP8 text-encoder trees, the fused/converted
  trees, the family's multi-repo layout), and one model per endpoint does
  not cover a family (DiT + text encoder + VAE + upscaler repos). It also
  mounts where the network volume mounts. It is worth revisiting for a
  single-repo family once the trees are published on the Hub.

## One image with boot-time config (2026-10-25)

Owner decision 2026-10-25 ([ADR-0011](../adr/0011-deploy-section.md) "One
image with boot-time config"): CI builds one CUDA serve image (`cuda`) and
the CPU image (`cpu`) instead of seven variant images whose only difference
was a baked config. Every launch path sends the worker config at boot:

| launch path | how the config is sent |
|---|---|
| fv-control clusters (Runpod pods, GMI / Brev) | `FV_WORKER_TOML_B64`, the pool's config text (`control/src/worker-config.ts` `workerConfigEnv`, via `cluster/payloads.ts` and `providers.ts`) |
| fv-control serverless endpoints | `FV_WORKER_TOML_B64` in the endpoint's own template; what the endpoint serves derives from the spec's config (`serverless/serves.ts`) |
| fv-control standalone launches | a preset's config, a named `configs/serve` file or the given text, inline |
| `fv-deploy` (`crates/fastvideo-deploy` `launch`) | `FV_WORKER_TOML_B64` of the preset's config, for every preset (`inline_config`) |
| autoscaler pod pools | `[autoscale.pools.pod] worker_config` (a `configs/serve` name), read at start and sent as `FV_WORKER_TOML_B64` in the pod's env |
| `runpod-pod.sh`, `runpod-endpoint.sh` | `variants.sh` `fv_config_env_json`: `FV_WORKER_CONFIG=/etc/fv/runpod-fake.toml`, else the file inline |
| `vast.sh`, `cloudrift-worker.sh`, GCP `vm.sh` | `--config` (unchanged) |
| CI's smoke check (`ci-images.sh smoke`) | `FV_WORKER_CONFIG=/etc/fv/runpod-fake.toml`, explicitly; a second check asserts the image refuses to boot without a config |

An image started without a config exits non-zero with `no worker config:
the image serves no default; send FV_WORKER_TOML_B64 (inline) or
FV_WORKER_CONFIG (a file in the image), or pass --config <path>`
(`crates/fastvideo-deploy` `boot::config::missing`).

**Size of a config in the env.** Runpod documents no limit on one
environment variable's value (the pod API allows up to 50 variables; the
v2 template body is capped at 1 MiB). The largest worker config,
`runpod-family-h3.toml`, is 4199 bytes, 5600 bytes as base64. fv-control
(`worker-config.ts` `MAX_WORKER_TOML_B64`) and fv-deploy
(`launch.rs` `MAX_WORKER_TOML_B64`) refuse an encoded config over 16 KiB, a
conservative guard; tests assert the two agree and that every committed
config fits within half of it.

**Stored specs.** Cluster specs, pool specs, standalone launches and
serverless specs saved with a `variant` still load: the field is dropped,
`cpu` becomes `compute = "CPU"`, and a pool or spec with no config takes
the config its variant image used to bake (`enums.ts` `legacyVariant`,
`LEGACY_VARIANT_CONFIGS`; tests in `payloads.test.ts` and
`serverless.test.ts`). A preset owns its config **by name** (`config =
"runpod.toml"`, its text read at each launch), so an edited
`configs/serve` file reaches the next launch and never makes a stored
preset spec invalid; a stored serverless spec naming a preset reads with the
preset's compute and config, whatever an older fv-control stored (none, a
path in a variant image, an inline snapshot). A scale (or an update that
changes only the worker counts) keeps the stored spec and never touches the
template or the image. CPU workers run the fake engine only: a CPU pool,
standalone pod or serverless endpoint whose config's `[engine] backend` is
not `fake` is refused (a custom serverless endpoint still defaults to CPU, as
before, so a CUDA config there needs `compute: GPU`). A stored cluster spec
whose pool config cannot be sent (a path outside `configs/serve`, an inline
config over 12 KiB, a CUDA config on CPU) loads, is named by the editor's
check, and up / scale / roll / start refuse it with a 409 naming the field.

### One image: the live steps

The change ships as code; the live resources move only when the owner runs
these, in order (none of them was run by the change itself).

**What the old fv-control resolves, and the window.** The deployed (old)
fv-control resolves a pool's or endpoint's image as `<variant>-<channel>`:
`h3-turbo-stable`, `ltx-latest`, …, and `cpu-<channel>` for CPU workers.
The merge-time CI build moves `:cpu-latest` / `:cpu` (and `:latest`, the
debug image), never a CUDA variant tag; the promotion (step 4) moves
`:cpu-stable`. The new CPU image bakes no config, and the old fv-control
creates CPU **serverless** endpoints without one (its CPU cluster pools and
standalone pods send theirs: `config_toml` inline, or `FV_WORKER_CONFIG` of
the fake config, which the image carries). So from the merge build until the
new fv-control is deployed, a CPU serverless endpoint **created or updated**
on channel `latest` would refuse to boot; from the promotion, the same on
`stable`. Endpoints and pods already running pin a digest and are not
affected. Hence: deploy fv-control right after the merge, before the
promotion, and create or update no CPU serverless endpoint in between.

1. **Merge and build.** The push to main runs `serve-image.yml`, which pushes
   `:cuda-sha-<sha>`, `:cuda`, `:cuda-latest`, `:cpu-sha-<sha>`, `:cpu`,
   `:cpu-latest` and the debug image's `:sha-<sha>` / `:latest`, and records
   the `latest` release with the keys `debug cuda cpu`. The old CUDA variant
   tags (`:h3-turbo-stable`, `:ltx-latest`, …) stay where they are; nothing
   moves them again. The templates move only if the repository variable
   `FV_TEMPLATE_CHANNEL` is `latest` (default: `stable`, so they do not);
   check it before merging.
2. **Deploy fv-control** (`control/`, wrangler deploy) from that main, as
   soon as the build is green. From then on every cluster, serverless
   endpoint and standalone launch resolves `:cuda-<channel>` /
   `:cpu-<channel>` and sends its config inline. Until the promotion,
   `:cuda-stable` does not exist and a `stable` launch falls back to
   `:cuda-latest` (`ghcr.ts`, as before the first promotion of any image).
   Running pods and endpoints keep their old images and configs until
   relaunched or rolled.
3. **Verify on something small first**: a CPU cluster (`tiny-cpu`), then
   one GPU cluster pool and one serverless endpoint (both on `latest`);
   check `/health` reports the new image and the expected served models.
   Optionally a hand smoke on one GPU pod:
   `FV_SERVE_CONFIG=runpod.toml scripts/serve/runpod-pod.sh up` and a job,
   then `down` (a few cents).
4. **Promote.** `scripts/serve/release.sh promote <sha> stable`. It creates
   `:cuda-stable` / `:cpu-stable` and, because the templates follow
   `stable`, runs `runpod-templates.sh sync` for `cuda` and `cpu`: this
   **creates** `fv-serve-cuda-sls` and `fv-serve-cuda-pod` and moves
   `fv-serve-cpu-pod` to the new CPU image. `release.sh list --verify` must
   pass (it fails when a channel head lacks one of `debug cuda cpu`).
   `fv-serve-cpu-pod` and `:cpu-stable` now name an image with no baked
   config: a hand-started pod needs `FV_WORKER_CONFIG=/etc/fv/runpod-fake.toml`
   or `--config`.
5. **Roll what is running.** Existing clusters: roll each to its channel
   (Releases page, or the cluster's roll action); a stored spec with
   `variant` loads through the read-compat mapping, and one whose pool
   config the new fv-control cannot send (a path outside `configs/serve`, an
   inline config over 12 KiB) is named by the editor's check and refused by
   up / scale / roll (409) until edited. Existing serverless endpoints: a
   scale never touches their template or image; update or relaunch each
   from fv-control to move it to the new image (a template edit alone no
   longer changes what an endpoint serves).
6. **Autoscaler.** Put the id of `fv-serve-cuda-pod`
   (`scripts/serve/runpod-templates.sh id cuda pod`) in `deploy/deploy.toml`
   `sfwan-live` `template_id` (replacing `REPLACE_FV_SERVE_CUDA_POD_TEMPLATE_ID`)
   and run `node control/gen-configs.mjs`. The autoscaler reads the
   template's env (`GET /templates/{id}`) and sends it with the worker config
   on top, because REST v1 `POST /pods` documents no rule for a body `env`
   next to a `templateId` (the v2 create documents a per-key merge, body
   values winning). Before leaving `dry_run`, start one pod through it and
   check its env has both the `{{ RUNPOD_SECRET_fv_* }}` references and
   `FV_WORKER_TOML_B64`.
7. **Retire the old resources** only after 2-6 hold for a while: the
   `fv-serve-<variant>-sls` / `-pod` templates of the six CUDA variants (no
   code names them any more; `runpod-templates.sh list` shows them). Leave
   the old GHCR tags: they are immutable history and what the previous
   fv-control deployment boots.

**Rollback.** Before step 2, only CPU serverless creates on `latest` are
affected (above). After it, roll fv-control back to the previous
deployment: it boots the old CUDA variant tags, which still exist; its CPU
serverless endpoints need a `:cpu-<channel>` from before the change
(`release.sh promote <old sha> latest --allow-partial` moves `:cpu-latest`
back). `release.sh rollback` cannot cross this change: an older release has
no `cuda` digest, so it refuses and names the tags that would stay, unless
`--allow-partial`. From here on a rollback or pin is per image, no longer per
variant.

## Prebuilt binaries (compile on the build pod, assemble on GitHub)

Since 2026-10-06 (owner decisions) no image workflow needs to compile Rust
or CUDA. When a push to main changes the tools, `tools-release.yml` builds
and tests them on the shared build pod (a self-hosted runner) and publishes
a SemVer GitHub release, `tools-v<X.Y.Z>`, then dispatches the image
workflows (**docs/dev/tools-releases.md**).
Each workflow then:

1. **Resolves and downloads** (`scripts/ci/prebuilt.sh fetch <sets>`, with
   the job's `GITHUB_TOKEN`): the release pinned by the repository variable
   `FV_TOOLS_VERSION`, else the one built from exactly this checkout's tools
   inputs (input hash), else, for the image workflows on main only, the
   highest SemVer release (warning; the image is labelled
   `dev.fastvideo.tools=<tag>`). It checks the manifest against the release
   body, the tarballs' and every file's sha256 against the manifest, and the
   fv-serve features against the requested ones.
2. **Assembles** with the extracted directories as **named build contexts
   that replace the compile stages** (`--build-context serve-build=<dir>`,
   `cpu-build`, `binary`, `oxide`, `hf-fm`). The Dockerfiles' `COPY
   --from=<stage>` lines take the prebuilt files unchanged, so the lean
   runtime stages (`runtime`, `serve-os`, `serve-cuda-base`,
   `serve-cuda`, `serve-cpu`, `serve`) are exactly
   as before: no toolkit, no compiler in any runtime image. BuildKit never
   runs a replaced stage or its `builder` parent, so the log shows no
   `cargo` step (the C ffmpeg build stays, from the registry cache).
   fv-boot (the entrypoint, [ADR-0014](../adr/0014-fv-boot.md)) has its own
   `boot-build` stage: a release whose `serve-cuda` / `serve-cpu` set carries
   `out/fv-boot` replaces it too (`boot-build=<dir>`); a release from
   before fv-boot does not, and only that stage compiles (fv-boot's one
   crate, from `builder`), with a notice in the job. Nor does a release
   older than the commit's tools inputs (main's `FV_PREBUILT_STALE=latest`,
   a `FV_TOOLS_VERSION` pin): fv-boot enforces the image contract of the
   Dockerfile it ships in (the refusal without a config, ADR-0011), so it is
   always this commit's (`prebuilt.sh contexts --stale`; serve-image run 286
   hung on an image whose stale fv-boot served fv-serve's default). fv-lab (the runtime
   image's validation driver, [ADR-0019](../adr/0019-fv-lab.md)) rides in the
   `gpucheck` set the same way (`out/fv-lab`, `lab-build=<dir>`).
3. **Falls back** when there is no usable release (test workflows and
   branch images whose tools inputs have no release, other features
   requested on dispatch, a bad checksum): a `::warning::` names the reason,
   and the job compiles exactly as before, so CI never deadlocks on the pod;
   `FV_PREBUILT_REQUIRE=1` would fail instead.

| workflow | sets downloaded | what no longer compiles on the runner |
|---|---|---|
| serve-image | `oxide gpucheck hf-fm serve-cuda serve-cpu` | every stage with `cargo` / `nvcc` / `tileiras` (debug, `cuda`, `cpu`) |
| gpucheck-runtime-image | `oxide gpucheck hf-fm` | builder, oxide, build, hf-fm |
| vast-pytorch-image | `gpucheck-vast hf-fm`, always from the **newest** tools release (or the pin); no release: the job fails, it never compiles | builder, build, hf-fm |
| serve-compat (`build` job) | `serve-fake` | the debug `fake,full` fv-serve (toolchain + rust-cache skipped) |
| gpucheck-t0 | `gpucheck-tests gpucheck` | `cargo test` (runs the pod's test binaries), the CUDA type-check (covered by the release `--features cuda` build), the nvrtc job's release build (runs the prebuilt `fv-gpucheck nvrtc`) |
| upstream-images, release | none | nothing compiled before either (upstream copies fv-gpucheck from the runtime image; release retags) |

Runner time (2026-10-06 runs, all compiling, since no prebuilt set was ever
served from R2) and the compile share a release removes: serve-image 23 min
(debug-image build 12.2 min, most of it cargo/nvcc, plus 1 min freeing
disk), gpucheck-runtime-image 9.5–11 min (build 7.1 min + 2 min freeing
disk), vast-pytorch-image 10 min (build 6.3 min), serve-compat build job
2.8 min (with a warm rust-cache), gpucheck-t0 unit-tests 14.4 min (of which
running the tests ~8 min) and nvrtc-compile 6.5 min. Expected with an exact
release: roughly 10, 7–8, 4–5, 2, 6 and 3 min less respectively (estimates
from the step times; not yet measured on a release, see
docs/dev/tools-releases.md "Measured").

Pull-request jobs hash the checked-out merge commit. Repository variables:
`FV_TOOLS_VERSION` (pin the image workflows to one release, e.g. `0.1.3`, for
a rollback), `FV_PREBUILT_DISABLE=1` (always compile).

## Publishing to Runpod

Options checked (Runpod docs, 2026-09-28):

| option | what it gives | verdict |
|---|---|---|
| Runpod-hosted registry you push to | does not exist: Runpod pulls from Docker Hub, GHCR, ECR, …; the API only stores **credentials** for external registries (`/containerregistryauth`, v2 `/registries`) | n/a |
| **Deploy from GitHub** (serverless) | Runpod builds the repo's Dockerfile and stores the image in its own registry (`registry.runpod.net/…`, visible in `GET /v2/serverless/{id}/builds`), then rolls the endpoint | the only way to get a Runpod-hosted image, but: set up per endpoint in the console (GitHub OAuth; no API to create one), triggered by GitHub **releases**, serverless only (no pods), 30 min limit on the Docker build step (our Rust + CUDA build is ~45 min cold), no private base images, and the built image is bound to that endpoint. Our deploy scripts create and delete endpoints per run and pods need the same image. Not used; a thin `FROM ghcr.io/…:<variant>-sha-…` Dockerfile would fit the 30 min limit if a permanent GitHub-built endpoint is ever wanted |
| Runpod Hub | public listing of a repo (hub.json, tests.json, `handler.py`), manual review | not a private deployment path |
| **Templates pointing at the image** | persistent `fv-serve-<image>-sls` / `-pod` templates the endpoints and pods start from; updating a template rolls every endpoint using it (rolling release) | **chosen** |
| Cached models | weights on the host, see above | not applicable to the image |

What `runpod-templates.sh sync` does, per image (run by
`release.sh promote … stable` / `rollback`, see [releases.md](releases.md);
CI's `Register the Runpod templates` step runs `sync-all` after main builds
only when the repository variable `FV_TEMPLATE_CHANNEL` is `latest`):
it creates or updates the serverless template (`isServerless`,
`FV_SERVE_MODE=runpod-queue`, `FV_WEIGHTS=/runpod-volume/weights`,
trust-gateway auth, 20 GB disk) and the pod template (HTTP mode, ports
8000/http + 70000/tcp, `/workspace` volume mount, 30 GB disk); `cpu`
gets a CPU pod template only. Images are referenced **by digest**. Secrets are
Runpod secret references (`{{ RUNPOD_SECRET_fv_* }}`), never values. If the
registry ever goes private, create a Runpod registry auth for GHCR once and set
its id as the `RUNPOD_REGISTRY_AUTH_ID` secret; the script adds it to every
template. The env also names the image (`FV_IMAGE_REF`, `FV_IMAGE_DIGEST`)
and the channel (`FV_RELEASE_CHANNEL`), which fv-serve reports. The
templates carry **no worker config** (since 2026-10-25): whatever starts an
endpoint or pod from one adds it to the env, so a template edit changes the
image, never what a running endpoint serves.

Deploy scripts boot what the templates name:

- `scripts/serve/runpod-endpoint.sh`: every endpoint gets a per-run
  template that boots the digest `fv-serve-cuda-sls` names, with the config
  in its env (`FV_SERVE_CONFIG`: `/etc/fv/runpod-fake.toml` as
  `FV_WORKER_CONFIG`, a `configs/serve` name or a local file inline as
  `FV_WORKER_TOML_B64`; `variants.sh` `fv_config_env_json`). `[image]` /
  `FV_SERVE_IMAGE` still override.
- `scripts/serve/runpod-pod.sh`: the pod boots the image of
  `fv-serve-cuda-pod`, with the config in its env the same way.
- Without templates (no API key, not yet synced) the scripts fall back to the
  GHCR tag `:cuda`.
- Both start the image's entrypoint, fv-boot, with the config in the env and
  no `--config` argument, so they need an image with fv-boot (plan C2,
  #159, 2026-10-24, and later). A hand-passed `FV_SERVE_IMAGE` / `[image]`
  from before it does not serve the config sent (its `fv-entry.sh` reads only
  the baked `FV_CONFIG`, not `FV_WORKER_TOML_B64` / `FV_WORKER_CONFIG`); use
  an older checkout of the scripts for such an image.

### Required GitHub secrets

| secret | used by | required |
|---|---|---|
| `GITHUB_TOKEN` (automatic) | push to GHCR, registry build cache | yes (built in) |
| `RUNPOD_API_KEY` | `release.yml` template sync; `Register the Runpod templates` when the templates follow `latest` (REST v1 `/templates`) | for Runpod publishing; without it the step warns and skips |
| `FV_CF_API_TOKEN` (or `CLOUDFLARE_API_TOKEN`) | release history in D1: `Record the latest release`, `release.yml` | for release history; without it serve-image warns and release.yml fails |
| `FV_CF_ACCOUNT_ID`, `FV_D1_DATABASE_ID` | the same | no (looked up from the token: first account, database `fv-jobs`) |
| `RUNPOD_REGISTRY_AUTH_ID` | added to the templates as `containerRegistryAuthId` | only if the GHCR package becomes private |

The Runpod account needs the `fv_*` Runpod secrets the templates reference
(they already exist for the deploy scripts).

## Size reduction

(2026-09-28; the variant images measured here were replaced by the one
`cuda` image on 2026-10-25, which is any one of them without the config
layer.) Compressed sizes from the registry manifests (gzip layers, linux/amd64).
**Before**: every variant ran the one image `fastvideo-rs-serve:latest`
(`sha256:c0ce4f45…`, main before this change). **After**: the variant images
of CI run 36491613203 (branch `claude/serve-images`, `a3292bf`).

| variant | before | after | change |
|---|---:|---:|---:|
| h3-turbo | 1600.3 MB, 12 layers | **996.8 MB**, 12 layers (`sha256:bdc7a2b1…`) | −603.5 MB (−38 %) |
| h3-max | 1600.3 | **996.8** (`sha256:705aeac2…`) | −38 % |
| ltx | 1600.3 | **996.8** (`sha256:f24dc8a0…`) | −38 % |
| wan | 1600.3 | **996.8** (`sha256:7ad81c50…`) | −38 % |
| wan5b | 1600.3 | **996.8** (`sha256:8dc96b42…`) | −38 % |
| sfwan | 1600.3 | **996.8** (`sha256:900c9856…`) | −38 % |
| gateway (CPU) | 1600.3 | **68.4 MB**, 8 layers (`sha256:330c15c1…`) | −1531.9 MB (−96 %) |
| debug (legacy all-in-one) | 1600.3 | unchanged (`:latest`, `:sha-…`) | – |

The six CUDA variants share every layer but the last (config + entrypoint,
< 10 KB), so a host holding one of them pulls another in a few KB; before,
there was nothing to share, but also only one image.

Layers (MB compressed):

| layer | before (`:latest`) | after (CUDA variant) | after (gateway) |
|---|---:|---:|---:|
| ubuntu:22.04 | 29.8 | 29.8 | 29.8 |
| apt: CUDA libs + tileiras (→ nvcc, libnvjitlink, build-essential) + CUPTI + ffmpeg (193 packages) + openssh-server + rsync + curl/wget/binutils | **1506.1** | – | – |
| apt: ca-certificates + libx264/libvpx/libdav1d | – | 3.7 | 3.7 |
| ffmpeg + ffprobe (FFmpeg 4.4.5, static except x264/vpx/dav1d) | – | 19.1 | 19.1 |
| … since 2026-10-13: + OpenSSL TLS (`rtmps://`) and static libsrt (`srt://`), [Stream outputs](#stream-outputs-ffmpeg-2026-10-13) | – | 19.9 | 19.9 |
| cuBLAS + cuBLASLt 13.8 | (in the apt layer) | 439.5 | – |
| cuDNN 9.26 without `libcudnn_adv` | (in the apt layer) | 421.9 | – |
| NVRTC 13.4 + builtins | (in the apt layer) | 54.4 | – |
| hf-fm / scripts/gpu / oxide cubins / fv-gpucheck | 17.9 / 0.4 / 0.2 / 17.4 | – | – |
| fv-serve (+ configs before) | 28.4 | 28.4 (`cuda,http-client`) | 15.9 (`http-client`, no CUDA) |

What each removal is worth: the per-layer numbers above are measured; the
split of the old 1.5 GB apt layer is not (it was one layer). Dependency
closures from the Ubuntu 22.04 / NVIDIA indexes (`.deb` sizes, xz; gzip layers
come out larger), on top of the base image and the CUDA libraries we keep:

| removed | packages | .deb (xz) | installed |
|---|---:|---:|---:|
| `cuda-tileiras-13-4` (depends on `cuda-nvcc-13-4` → build-essential, gcc, libnvvm, CCCL headers, `libnvjitlink`) | 58 | 231 MB | 866 MB |
| Ubuntu `ffmpeg` (`--no-install-recommends`) | 193 | 117 MB | 396 MB |
| `cuda-cupti-13-4` | 1 | 16 MB | 153 MB |
| `libcudnn_adv.so.9.26.0` (one file of the cuDNN package) | – | – | 107 MB |
| openssh-server + rsync | 10 | 2 MB | 7 MB |
| replacement: libx264-163, libvpx7, libdav1d5 + the ffmpeg build | 3 | 2 MB | 6 MB + 19.1 MB layer |

### Verification of each removal

- **Smoke check (CI, no GPU)**, per variant (`ci-images.sh smoke`, run
  36491613203: all seven `smoke ok`): every dynamic library of fv-serve,
  ffmpeg and ffprobe resolves (`ldd`); `fv-serve --version`; the baked
  config and the fake config parse; ffmpeg lists `h264_nvenc`, `libx264`,
  `libvpx`, `aac` and muxes an H.264 + AAC MP4 that ffprobe reads back (the
  build stage also encodes VP8/IVF); sshd, rsync, fv-gpucheck, hf-fm,
  tileiras and nvcc are absent, as are `scripts/`, `oxide/`, CUPTI and
  `libcudnn_adv`; CUDA variants have every NVRTC/cuBLAS/cuDNN soname fv-serve
  or cuDNN dlopens (graph, ops, cnn, heuristic, the three engine libraries,
  nvrtc-builtins); the gateway has no CUDA library and no `cuda` feature;
  HTTP mode on the fake engine answers `/healthz` through the new entrypoint.
- **GPU smoke (wan variant)**: `sha256:7ad81c50…` booted from the published
  template `fv-serve-wan-sls` on a Runpod queue endpoint (US `s2k01690bi`,
  H100 80GB HBM3, driver 580.126.09): two `fastwan21-1.3b` jobs `succeeded`
  (seed 1 → 1 089 835 B MP4, the same size as seed 1 on the old image),
  model load 47.6 s / 46.2 s (old image 47-58 s). NVENC: `h264_nvenc` present,
  encode not possible on H100 (no NVENC hardware, as before). The other CUDA
  variants carry exactly the same base, CUDA and binary layers. The H3 and
  LTX variants ran on a GPU on 2026-09-29 (next item).
- **GPU smoke (h3-turbo, h3-max, ltx variants), 2026-09-29.** Each image was
  taken from its published pod template (`fv-serve-<variant>-pod`; all
  three are the images of main `2cd1ba0`, serve-image run 36503456884). Each
  booted with its baked config (`scripts/serve/runpod-pod.sh up` with
  `FV_SERVE_CONFIG=/etc/fv/<config>`, `RUNPOD_ALLOWED_CUDA=""`) on 1x RTX PRO
  6000 Blackwell (EUR-IS-1, $2.09/hr, EU volume `jg48s6o1w0` read only, R2 +
  D1 stores) and served one fal queue job. "Old image" is the legacy
  all-in-one `fastvideo-rs-serve:sha-9c42844` (1.6 GB) of the WP-18 E2E runs
  (docs/serve/e2e/h3-turbo.md, h3-max.md, ltx.md). Raw results:
  `artifacts/serve/e2e/slim-images/<variant>/`.

  | variant | image (template digest) | pod, lifetime | create → `/ping` 200 | job | result | against the old image |
  |---|---|---|---:|---|---|---|
  | h3-turbo | `sha256:c782eb37…` | `bn56zvm3xeppih`, 532 s ($0.31) | 441 s | fal `minimax/h3-turbo` 480P, fox prompt, seed 1; then 1080P (docs/serve/h3-1080p-and-upscaler.md) | PASS: 832x480, 124 frames, AAC; wall 12.5 s, denoise 7.05 s, encode 0.51 s | **byte-identical** MP4 (2 916 017 B, `cmp` equal, SSIM 1.0) to `artifacts/serve/e2e/h3-turbo/samples/fal-t2v-480p.mp4` |
  | h3-max | `sha256:066a5547…` | `14siooczdtjlkp`, 387 s ($0.22) | 342 s | fal `minimax/h3-max` default 768P, fox prompt, seed 1 | PASS: 1344x768, 124 frames, AAC; wall 36.6 s, denoise 22.7 s, encode 0.90 s | same MP4 size as the old run's identical request (8 286 812 B, `artifacts/serve/e2e/h3-max/results.jsonl`); against the kept 960-px re-encode of it, SSIM 0.962 / PSNR 37.0 dB, which is the re-encode's limit |
  | ltx | `sha256:680ffa2a…` | `sn8tfngz26mcua`, 147 s ($0.09) | 64 s | `ltx_e2e.py probe fal-turbo-1080p` (fal `fastvideo/ltx-turbo`, 1080P, seed 3) | PASS: 1920x1080, 121 frames @ 24, AAC 48 kHz; first job after boot 71.5 s, inference 26.2 s (old 27.1 s) | 10 554 814 B against 10 436 599 B (+1.1 %): not bit-identical, as expected, because the LTX path changed between `9c42844` and `2cd1ba0` (image conditioning E5/E9, the IC-LoRA wiring). The clip is a coherent, sharp shot of the prompt. The old MP4 was not kept, so no frame metric |

  The LTX job ran the two-stage path on the slim image on sm_120 (RTX PRO
  6000), including cuDNN conv3d in the latent upsampler. The same ltx image
  also served the LTX reference-to-video E2E on an H100 (sm_90,
  docs/serve/e2e/ltx.md). The
  identical H3 outputs show that the image slimming (cuDNN without `adv`,
  no CUPTI, the minimal FFmpeg) does not change what fv-serve computes or
  how it encodes. The h3-turbo and h3-max pulls (342-441 s to ready) were
  first pulls of these images on the host; ltx came up in 64 s on a host
  that already had the shared layers.
- **cuDNN pruning**: only `libcudnn_adv` is dropped. `libcudnn.so` loads it
  only for the legacy RNN / multi-head-attention / CTC API, which fv-serve
  does not call (it uses the legacy convolution API → `libcudnn_cnn`,
  `cudnnReduceTensor` → `libcudnn_ops`, and the backend graph API for SDPA).
  `libcudnn_graph` dlopens the three engine libraries (`precompiled`,
  `runtime_compiled`, `tensor_ir`) and the heuristic library; which engine
  serves the unified SDPA node (engine ids 11 and 8,
  docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md) is not mapped to a
  library, so all engine libraries stay. (2026-10-06: `tensor_ir` measured
  as required, and `libcudnn_ext` dropped; see [Lean runtime
  images](#lean-runtime-images-2026-10-06).) Checked: no cuDNN library needs
  `libnvJitLink` (DT_NEEDED and dlopen strings), so dropping tileiras's
  `libnvjitlink` is safe.
- **CUPTI**: loaded lazily by `cudarc/cupti` only for `FASTVIDEO_GPU_TRACE`;
  the debug image keeps it.
- **ffmpeg**: every codec, format and filter the code names (`h264_nvenc`
  with p4/p5 presets, `libx264`, `libvpx`, `libvpx-vp9`, native AAC,
  mp4/ivf/h264/rawvideo/f32le/s16le/wav/flv/mpegts, lavfi `color`/`sine`,
  `crop`/`fps`/scale, the `rtmps`/`tls`/`https` and `srt` protocols) is
  either native (all native components are built) or one of the external
  libraries (x264, vpx, dav1d; OpenSSL and libsrt since 2026-10-13).
  FFmpeg 4.4 is the release Ubuntu 22.04 packages, so the command lines behave
  as before. Not built: the other external libraries of Ubuntu's build (x265,
  aom, openh264, libass, …), which no code path uses.
- **zstd**: built, not switched on. The workflow has a `compression: zstd`
  input (tags get a `-zstd-` infix: `:wan-zstd-sha-…`, never the bare
  variant tag, and no template update). CI run 36495373046 (zstd level 3,
  smoke checks passed): wan **959.7 MB** (`sha256:f6f45408…`, −37.1 MB /
  −3.7 % against gzip), gateway **64.5 MB** (`sha256:c97ab8f0…`). The size
  gain is small; zstd's gain is faster decompression on the host. Runpod
  does not document which container runtime its hosts run or whether they
  pull zstd layers, a worker that cannot pull its image sits in the queue
  without an error, and the GPU budget was stopped before a zstd pull test,
  so gzip stays the default until one queue job on a `-zstd-` tag passes.
- **Base image**: `ubuntu:22.04` stays (29.8 MB): the binary is built against
  its glibc and the NVIDIA apt pins target it; a distroless/Debian-slim base
  would save ~0-10 MB compressed for a glibc mismatch risk.

## Lean runtime images (2026-10-06)

Owner decision: make the runtime images multi-layer and lean, and remove
unused components. Branch `wip/lean-runtime-images`; CI runs
gpucheck-runtime-image 37509025431 (`f692ed2`) and serve-image 37511210976
(`7472073`, dispatched on the branch). Sizes are compressed (gzip) from the
registry manifests, linux/amd64.

### Before / after

| image | before | after | change |
|---|---:|---:|---:|
| `fastvideo-rs-runtime` (fv-gpucheck) | 1567.3 MB, 8 layers (one 1498.7 MB apt layer) | **1019.8 MB**, 19 layers | −547.5 MB (−35 %) |
| `fastvideo-rs-serve:latest` / `:sha-…` (debug, the old all-in-one) | 1597.7 MB, 12 layers | **1050.1 MB**, 22 layers | −547.6 MB (−34 %) |
| `fastvideo-rs-serve:<variant>` ×6 (CUDA) | 998.7 MB, 12 layers | **994.7 MB**, 14 layers | −4.0 MB (`libcudnn_ext`) |
| `fastvideo-rs-serve:cpu` | 68.4 MB | **69.0 MB**, 8 layers | +0.6 MB (Ubuntu digest pin) |

Shared layers (identical digests), measured on the pushed images:

| pair | before | after |
|---|---:|---:|
| runtime ↔ any CUDA variant | 29.8 MB (Ubuntu only) | **964.3 MB** (all of base-cuda) |
| runtime ↔ debug | 1546.4 MB | **964.5 MB** guaranteed (base-cuda); the runtime-only layers (≈56 MB) match too when the two workflows hit the same registry cache (all 1019.8 MB on `f692ed2`/`7472073`, base-cuda only on `d940b41`) |
| CUDA variant ↔ CUDA variant | 998.7 MB (all but the config layer) | **994.7 MB** (all but the config layer) |
| cpu ↔ everything | 52.3 MB | 52.3 MB (base-os) |
| **a host pulling runtime + one variant** | 2536 MB | **1050 MB** (−59 %) |
| unique bytes of runtime + debug + all six variants (`:latest` tags before, this branch after) | 2587.5 MB | **1080.5 MB** (−58 %) |

Layers of the CUDA stack (MB): Ubuntu 29.8 · codec libs 3.5 · ffmpeg 19.1 ·
cuBLASLt 388.2 · cuDNN precompiled engines 204.9 · cuDNN core 213.1 · cuBLAS
+ NVRTC 105.8 (+ cudart since the follow-up, see below) · ld.so config 0.0. Runtime adds sshd/rsync/curl 4.4 · CUPTI
12.2 · hf-fm 18.0 · scripts 1.4 · oxide 0.2 · fv-gpucheck 19.3; debug adds
configs + worker.py + fv-serve 30.4; a variant adds fv-serve 30.4 + config.

**Pull time.** Both GPU smokes below were the first pull of these images on
their hosts: create → first `/ping` 30 s (debug) and 27 s (ltx variant),
against 64-441 s for the previous variant images in the 2026-09-29 smokes.
These are two samples on hosts Runpod chose, not a controlled measurement.

### How the layers stay shared across workflows

BuildKit only reuses a layer blob when it gets a cache hit, and the two
image workflows run on different runners with different caches, so before
this change the runtime image and the variants each had their own CUDA
layers. Now `scripts/ci/base-images.sh ensure` (a step in both workflows)
hashes the `ARG UBUNTU=` line, the Dockerfile between `# >>> shared base`
and `# <<< shared base`, `scripts/gpu/cuda-13.pins` and `cuda-13-runtime.pins`, and publishes
`fastvideo-rs-runtime:base-os-<hash>` and `:base-cuda-<hash>` once (7 min,
first run only; later runs resolve the tags in 1 s). The digests go back
into the build as named build contexts (`base-os=docker-image://…@sha256:…`,
the same mechanism the prebuilt binaries use), which replace those stages.
gzip pushes keep the base blobs as they are (`force-compression` is only
set for zstd). Ubuntu is pinned by digest, so the base changes only when
that line, a CUDA pin or the ffmpeg build changes. `FV_BASE_DISABLE=1`
(repository variable) builds the stages inline instead. If both workflows
build a new hash at the same moment, the later push wins the tag and the
earlier images keep their own (valid) base until their next build.

### What was removed, and how it was checked

From the runtime and debug images (dependency closures as measured above,
in "Size reduction"):

| removed | why it is unused | size |
|---|---|---|
| `cuda-tileiras-13-4` + its `cuda-nvcc-13-4` → build-essential, gcc, libnvvm, CCCL headers, libnvjitlink | only the AOT oxide build (`builder`/`oxide` stages) runs tileiras; the cubins are embedded; no runtime JIT (grep: no tileiras call outside `fastvideo-oxide-kernels`) | 231 MB .deb / 866 MB installed |
| Ubuntu `ffmpeg` (193 packages) | replaced by the shared minimal ffmpeg (x264, vpx, dav1d, NVENC, native AAC), which covers fv-gpucheck's mp4 writer, `ffprobe` and `hd-upscaler.sh` | 117 MB .deb / 396 MB installed |
| `libcudnn_adv` | legacy RNN / MHA / CTC API, never called (as for the variants since 2026-09-28) | 100.2 MB |
| `libcudnn_ext` (all images) | exports only `cudnnCausalConv1d*`, `cudnnFFTCausalConv1d*`, `cudnnGnnAgg*` (subquadratic ops), which nothing calls | 3.8 MB |
| CUPTI `libcheckpoint`, `libpcsamplingutil`, static libraries | the activity API (`FASTVIDEO_GPU_TRACE`) needs `libcupti` (+ `libnvperf_*`, kept) | ~1 MB (+105 MB of `.a` that were already deleted) |
| `libnvblas`, dpkg/apt metadata, the NVIDIA keyring, `wget`, `binutils` | not loaded; `remote.sh` probes symbols with `grep -a` first, `nsys-pod.sh` now downloads with curl | small |

Kept, because something loads it: cuBLAS + cuBLASLt (`CudaBlas::new` in
every device), NVRTC + builtins (kernels; cuDNN runtime-compiled and
tensor-IR engines), cuDNN graph/ops/cnn/heuristic and all three engine
libraries, CUPTI (runtime/debug only), sshd/rsync/curl (`remote.sh`, Vast).
No variant drops cuDNN: every CUDA device creates a `Cudnn` handle
(`wan/device.rs`) and every family's VAE runs cuDNN convolutions.

**`libcudnn_engines_tensor_ir` is required (measured, 2026-10-06).** It is
the second-largest candidate (75.8 MB). With it moved away on an RTX PRO
6000 (sm_120), `fv-gpucheck kernels --groups conv,attn3_parity` fails:
convolutions return `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` and the SDPA
graphs get no plan (`status 1008: ptrDesc->finalize()`). It stays.

### CUDA runtime (libcudart) and the dependency audit

Owner follow-up (2026-10-06): base-cuda also ships the CUDA runtime,
`cuda-cudart-13-4` (pinned as `CUDA_CUDART_PKG` / `CUDA_CUDART_SONAME` in
`scripts/gpu/cuda-13-runtime.pins`, a separate file because `cuda-13.pins`
is an input of the build-base image tag and of the tools-release hash, which
should not move for a package only the runtime images ship): `libcudart.so.13.4.92` + `libcudart.so.13` +
`libcudart.so` in the cuBLAS + NVRTC layer (0.8 MB uncompressed). cudarc
uses the driver API and does not load it, but CUPTI dlopens `libcudart.so`
and anything linked against the runtime finds it. Its package's other
dependencies (`cuda-toolkit-*-config-common`) only carry ld.so/alternatives
configuration, which base-cuda writes itself.

Audit of every shipped library (DT_NEEDED from `readelf -d`, dlopen names
from the binaries' strings):

| needs | from |
|---|---|
| glibc (`libc`, `libm`, `libdl`, `librt`, `libpthread`, `libutil`, `ld-linux`), `libstdc++.so.6`, `libgcc_s.so.1`, `libz.so.1` | Ubuntu 22.04 base |
| `libcublasLt.so.13` (cuBLAS, cuDNN precompiled engines), `libnvrtc.so.13` + `libnvrtc-builtins` (cuBLASLt, cuDNN engines), the cuDNN sub-libraries, `libcudart.so` (CUPTI) | base-cuda (CUPTI: runtime image) |
| `libcuda.so.1` | the host driver (NVIDIA container runtime) |
| `libcudnn_adv` / `libcudnn_ext` (dlopened by `libcudnn` only for their APIs) | intentionally absent (unused) |
| `libcask_profile_interface.so` (cuDNN precompiled engines) | an optional profiling hook, not part of any NVIDIA package |
| EGL/GL/X11/OpenCL/OptiX/Vulkan-SC libraries (CUPTI / nvperf graphics-interop profiling) | the host driver when present; not used by the activity API |

With cudart (CI runs 37516701535 and 37517976275 on `d940b41`): base-cuda
**964.5 MB** (+0.2 MB), runtime 1020.0 MB, debug 1050.4 MB, CUDA variants
994.9 MB, cpu 69.0 MB; all of them on the same base-cuda digests.

### Stream outputs ffmpeg (2026-10-13)

The stream outputs (docs/serve/stream-outputs.md) publish RTMPS and SRT
with ffmpeg, which the minimal build could not (`--disable-autodetect`: no
TLS, no libsrt). The ffmpeg-build stage now adds `--enable-openssl
--enable-version3 --enable-libsrt` (OpenSSL 3 is Apache-2.0, so the binary
is GPLv3) with SRT 1.5.4 built from source as a static library, and
base-os installs `libssl3` (one shared copy for ffmpeg and ffprobe; static
OpenSSL would sit in both binaries). base-os fails its build unless
`ffmpeg -protocols` lists `rtmps` and `srt`, `libvpx-vp9` encodes, and `ldd`
resolves every library of ffmpeg and ffprobe.

| layer (compressed, GHCR) | before (`base-os-ea8bf35926f522ad`) | after (`base-os-67056bc7cf6a3849`) | change |
|---|---:|---:|---:|
| apt: ca-certificates, codec libs (+ libssl3: libssl.so.3, libcrypto.so.3, its engines, 5.7 MB installed) | 3.5 MB | 5.8 MB | +2.3 MB |
| ffmpeg + ffprobe | 19.1 MB | 19.9 MB | +0.8 MB |
| **base-os** | **52.3 MB** | **55.4 MB** | **+3.1 MB** |

Every runtime image sits on base-os, so each grows by 3.1 MB (base-cuda
964.5 → 967.6 MB). `fastvideo-rs-serve:cpu`: 69.5 → 74.5 MB (serve-image
run 38471722784 on `05b282c`; fv-serve itself +2.0 MB with the outputs code
and the console's hls.js).

Nothing else is missing. base-cuda now fails its build if any library in
`/usr/local/cuda-13.4/lib64` or any `libcudnn*` has an unresolved `ldd`
dependency, or if `libnvrtc.so`, `libcublas.so`, `libcublasLt.so`,
`libcudnn.so` or `libcudart.so` is not in the linker cache.

### GPU smoke (2026-10-06, EUR-IS-1, EU volume `jg48s6o1w0` read only)

Two RTX PRO 6000 Blackwell pods (driver 595.91.07, $2.09/hr) on the CI-built
images, each with a backstop and deleted after: 144 s + 143 s, about
**$0.17** in total.

1. **debug** `fastvideo-rs-serve@sha256:879910cf…` (= runtime + fv-serve):
   fv-serve with `runpod-ltx.toml` ready (`ltx25-distill-sol` loaded) 80 s
   after create; `fv-gpucheck nvrtc` PASS (154 kernels each for sm_90/100/120);
   `fv-gpucheck kernels --groups conv,attn3_parity,gemm` PASS, 0 failures
   (cuDNN conv3d, cuDNN SDPA unified graph on engines 8 and 11); CUPTI
   resolvable; the tensor-IR removal test above.
2. **ltx variant** `fastvideo-rs-serve@sha256:6ebadbc4…`: ready 67 s after
   create; one native job (`ltx25-distill-sol`, fox prompt, seed 1)
   `succeeded`: 1920x1080, 121 frames, H.264 + AAC 48 kHz, 8 465 068 B,
   inference 19.5 s, R2 + D1 stores.

The other variants differ from ltx only in their config layer.

### The legacy all-in-one target

Not retired, because it is still the default of several tools; it is now
the `runtime` image + fv-serve and costs a host that has any other CUDA image
only its top layers (30 MB, plus 56 MB of runtime layers over a variant). It
is used by: `scripts/serve/vast.sh` and `vast-serverless.sh` (they need
`deploy/vast/worker.py` and sshd), `cloudrift-worker.sh`, `scripts/gcp/vm.sh`
(`:latest` defaults), `runpod-pod.sh` / `runpod-endpoint.sh` for the fake
config (`:stable`), `scripts/serve/e2e/*.sh` and the profiling pods (they
need fv-serve and fv-gpucheck in one image), `release.sh record-build`
(the debug ref) and fv-control's all-in-one clusters (`spec.image.ref`, rolled
by `resolveTarget`). No Runpod template references it: the 13 `fv-serve-*`
templates all point at variant digests (one, `fv-serve-gateway-pod`, still
points at the retired gateway image; the owner may delete it). Retiring the
tags would mean moving each of those defaults to a variant image and giving
Vast/CloudRift/GCP a variant-plus-worker image; not done here.

## FlashBoot

`scripts/serve/runpod-endpoint.sh` keeps FlashBoot **off** by default;
`FV_FLASHBOOT=1` turns it on (queue endpoints: REST v1 `flashboot: true`;
load balancer: REST v2 `"FLASHBOOT"`; `FV_FLASHBOOT=priority` asks for
`PRIORITY_FLASHBOOT` on LB endpoints). Driver:
`scripts/serve/e2e/flashboot.sh`; raw results (one JSON line per cycle, the
driver log, `/health` + v2 worker list every 10 s, the ledger):
`artifacts/serve/e2e/flashboot/`.

Setup: queue endpoint, wan-turbo (`/etc/fv/runpod-wan.toml`,
`fastwan21-1.3b`), US volume `s2k01690bi` (US-CA-2), GPU list H100 NVL, H100
80GB HBM3, H200, RTX PRO 6000 (every worker was an **H100 80GB HBM3**),
workers min 0 / max 1, idle timeout 5 s. Per cycle: one native
`POST /fv/v1/jobs` (848x480, 81 frames, `wait: true`), then an info job on
the same worker (process start, `ready_after_s`). Between cycles the driver
waits until the worker is down (the v2 worker list's uptime stops advancing;
94-251 s after the job), then 30 s more. Every idle cycle's fv-serve process
started **after** the submit, i.e. the worker really had stopped.
Images: the current `:latest` (`sha256:c0ce4f45…`, 1.6 GB) for the on/off
comparison (runs 1-2); the slim wan variant (`sha256:7ad81c50…`, 997 MB,
booted from the published template) for run 3.

Seconds from `/run` submit:

| run / cycle | FlashBoot | image | process start | load (`ready_after_s`) | job taken (`delayTime`) | execution | **end to end** |
|---|---|---|---:|---:|---:|---:|---:|
| reference, WP-18 (serverless.md) scale from 0 | off | `sha-9c42844` | +49.3 | 70.4 | +119.6 | 6.4 | **126.6** |
| 1 / 1 scale from 0 | on | `:latest` | +62.5 | 51.8 | +114.2 | 6.4 | **122.0** |
| 1 / 2 after idle-out | on | `:latest` | (worker throttled, then started) | | +105 | 6 | ≈ **111** (log timestamps; the info job was lost) |
| 2 / 1 scale from 0 | on | `:latest` | +57.0 | 53.6 | +110.3 | 6.7 | **119.1** |
| 2 / 2 after idle-out | on | `:latest` | +3.9 | 57.7 | +61.5 | 6.3 | **68.9** |
| 2 / 3 after idle-out | on | `:latest` | +4.7 | 52.1 | +56.7 | 6.4 | **63.8** |
| 2 / 4 after idle-out | on | `:latest` | +27.0 | 52.5 | +79.3 | 7.3 | **88.7** |
| 2 / 5 after idle-out (endpoint PATCHed to off) | **off** | `:latest` | +6.0 | 47.1 | +53.1 | 6.4 | **61.8** |
| 2 / 6 after idle-out | **off** | `:latest` | +7.6 | 50.5 | +57.8 | 7.1 | **66.5** |
| 3 / 1 scale from 0, new image layers not on the host | on | slim `wan` | +80.2 | 47.6 | +127.6 | 6.3 | **134.7** |
| 3 / 2 after idle-out | on | slim `wan` | +8.5 | 46.2 | +54.6 | 7.3 | **62.2** |

Findings:

1. **FlashBoot does not keep fv-serve's state.** After an idle-out the
   process starts fresh and loads the model again (46-58 s every time), so
   the model load dominates every restart with FlashBoot on or off.
2. **What FlashBoot on/off changes here is not measurable.** The restart
   after idle-out took +3.9 / +4.7 / +27.0 s to the process start with
   FlashBoot on and +6.0 / +7.6 s with it off on the same endpoint (same
   hosts, image cached). End to end 64-89 s (on) against 62-67 s (off).
   Samples are small (3 vs 2) and the off cycles ran after the on cycles on
   hosts that had just run the worker, which is the situation FlashBoot is
   meant to speed up; with a ~50 s model load on top, the container-start
   difference it could make is a few seconds at most here. The v2 worker list
   showed three workers for this max-1 endpoint (only one ever had uptime); `/health` kept reporting a stopped
   FlashBoot worker as `idle: 1, ready: 1`, so `/health` cannot tell an
   idle-out from a live idle worker.
3. **Idle restart vs scale from 0**: 62-89 s against 119-135 s. The saving
   (~55 s) is the scheduling + container start of a fresh worker, and it
   shows up with FlashBoot off as well once the hosts have the image.
4. **Slim image, first pull**: scale from 0 on a host without the new
   layers took +80 s to the process start (+57-62 s for the cached old
   image); after that the slim image restarts like the old one (+8.5 s) and
   loads the model 5-10 s faster in these samples (46-48 s vs 47-58 s; not a
   controlled comparison).
5. Recommendation: leave FlashBoot off by default (no measured gain, and it
   keeps extra workers provisioned); the lever for cold starts is the model
   load (keep-warm `workersMin` for latency-critical pools, or faster Wan
   text-encoder loading, cold-start doc finding 3). Re-measure FlashBoot
   when fv-serve can checkpoint a loaded model, or on an endpoint with
   steadier traffic.

Cost: ≈ 20 min of H100 worker time over the three runs (≈ $1.5 at the flex
rate). The runs were cut after run 3 cycle 2 (coordinator budget stop);
endpoints `efvt0e92qjfnm6`, `flno5pyjhpq9mn`, `7rvc47m5qtbfxi` and their
per-run templates are deleted (REST 404); the published `fv-serve-*`
templates stay (they are the Runpod publication).
