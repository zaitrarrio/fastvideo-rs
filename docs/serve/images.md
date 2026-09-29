# fv-serve images: one per pod variant, published to GHCR and Runpod

Date: 2026-09-28. Build: `docker/gpucheck.Dockerfile` (stages `serve-os` …
`serve-<variant>`), CI: `.github/workflows/serve-image.yml` +
`scripts/serve/ci-images.sh`, Runpod publishing:
`scripts/serve/runpod-templates.sh`, variant table: `scripts/serve/variants.sh`.

## Images

All tags live in the one public package `ghcr.io/zaitrarrio/fastvideo-rs-serve`
(a new package per variant would start private on GHCR, and Runpod and the
deploy scripts pull anonymously).

| variant | config baked as `FV_CONFIG` | GHCR tags | Runpod templates |
|---|---|---|---|
| h3-turbo | `runpod.toml` | `:h3-turbo-sha-<sha>`, `:h3-turbo` / `:h3-turbo-latest`, `:h3-turbo-stable` | `fv-serve-h3-turbo-sls`, `fv-serve-h3-turbo-pod` |
| h3-max | `runpod-h3-max.toml` | `:h3-max`, … | `fv-serve-h3-max-sls` / `-pod` |
| ltx | `runpod-ltx.toml` | `:ltx`, … | `fv-serve-ltx-sls` / `-pod` |
| wan (wan-turbo) | `runpod-wan.toml` | `:wan`, … | `fv-serve-wan-sls` / `-pod` |
| wan5b | `runpod-wan5b.toml` | `:wan5b`, … | `fv-serve-wan5b-sls` / `-pod` |
| sfwan | `runpod-sfwan.toml` | `:sfwan`, … | `fv-serve-sfwan-sls` / `-pod` |
| gateway (CPU only) | `gateway.toml` | `:gateway`, … | `fv-serve-gateway-pod` |
| debug (legacy all-in-one) | every config, `runpod.toml` default | `:sha-<sha>`, `:latest`, `:stable` | none |

Each image also carries `runpod-fake.toml` (CI smoke check, fake engine).
`<sha>` tags are immutable. Two release channels move over them
([releases.md](releases.md)): `latest` (`:latest`, `:<variant>`,
`:<variant>-latest`) is the newest green main build, moved by CI;
`stable` (`:stable`, `:<variant>-stable`) moves only when a build is
promoted (`scripts/serve/release.sh promote <sha>`), and is what deploys
and the Runpod templates follow. Deploy tooling pins digests. Every image's
`fv-serve --version` and `/health` (`build`) report its git sha and build
time; the templates' env adds the image digest and channel.

### Layers

```
ubuntu:22.04                                   all images
serve-os: ca-certificates, libx264/libvpx/libdav1d runtime libs
ffmpeg + ffprobe (minimal FFmpeg 4.4 build)    all images
cuBLAS / cuBLASLt  | cuDNN (no adv) | NVRTC      CUDA variants (3 layers: pulled in parallel)
fv-serve --features cuda,http-client            CUDA variants (identical binary)
config + fv-entry + FV_VARIANT/FV_CONFIG        one small layer per variant
```

The CUDA variants differ only in their last layer (a few KB), so a host that
has pulled any one of them pulls the next in seconds, and Runpod's host
image cache covers every family at once. The gateway shares the OS and
ffmpeg layers and has no CUDA library at all (the smoke check asserts it).

**Per-family binaries.** `fastvideo-serve` has no per-family cargo features
(`cuda` pulls every `fastvideo-cudarc` pipeline: Wan, LTX-2, H3, MMAudio,
upscalers, shared kernels). Splitting it would mean cfg-gating the engine
service and the pipelines; the binary is ~30 MB compressed against ~1 GB of
CUDA libraries, so the win would be small and one shared binary layer is
cached once across variants instead of seven. The variant images therefore
share the binary and differ in config and entrypoint.

**Entrypoint** (`deploy/runpod/fv-entry.sh`): links `/workspace/weights` to
`/runpod-volume/weights` on serverless workers (the volume's HF-cache trees
link absolutely into `/workspace/weights`), defaults `FV_WEIGHTS` to the
mounted root, warns about any weights tree of the baked config that is
missing, and `exec`s fv-serve (which reads `FV_CONFIG`). Arguments pass
through (`--config …` still overrides).

### Weights are not in the images

Weights stay on the network volumes (US `s2k01690bi`, EU `jg48s6o1w0`):

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

## Publishing to Runpod

Options checked (Runpod docs, 2026-09-28):

| option | what it gives | verdict |
|---|---|---|
| Runpod-hosted registry you push to | does not exist: Runpod pulls from Docker Hub, GHCR, ECR, …; the API only stores **credentials** for external registries (`/containerregistryauth`, v2 `/registries`) | n/a |
| **Deploy from GitHub** (serverless) | Runpod builds the repo's Dockerfile and stores the image in its own registry (`registry.runpod.net/…`, visible in `GET /v2/serverless/{id}/builds`), then rolls the endpoint | the only way to get a Runpod-hosted image, but: set up per endpoint in the console (GitHub OAuth; no API to create one), triggered by GitHub **releases**, serverless only (no pods), 30 min limit on the Docker build step (our Rust + CUDA build is ~45 min cold), no private base images, and the built image is bound to that endpoint. Our deploy scripts create and delete endpoints per run and pods need the same image. Not used; a thin `FROM ghcr.io/…:<variant>-sha-…` Dockerfile would fit the 30 min limit if a permanent GitHub-built endpoint is ever wanted |
| Runpod Hub | public listing of a repo (hub.json, tests.json, `handler.py`), manual review | not a private deployment path |
| **Templates pointing at the image** | persistent `fv-serve-<variant>-sls` / `-pod` templates the endpoints and pods start from; updating a template rolls every endpoint using it (rolling release) | **chosen** |
| Cached models | weights on the host, see above | not applicable to the image |

What `runpod-templates.sh sync` does, per variant (run by
`release.sh promote … stable` / `rollback`, see [releases.md](releases.md);
CI's `Register the Runpod templates` step runs `sync-all` after main builds
only when the repository variable `FV_TEMPLATE_CHANNEL` is `latest`):
it creates or updates the serverless template (`isServerless`,
`FV_SERVE_MODE=runpod-queue`, `FV_WEIGHTS=/runpod-volume/weights`,
trust-gateway auth, 20 GB disk) and the pod template (HTTP mode, ports
8000/http + 70000/tcp, `/workspace` volume mount, 30 GB disk); the gateway
gets a CPU pod template only. Images are referenced **by digest**. Secrets are
Runpod secret references (`{{ RUNPOD_SECRET_fv_* }}`), never values. If the
registry ever goes private, create a Runpod registry auth for GHCR once and set
its id as the `RUNPOD_REGISTRY_AUTH_ID` secret; the script adds it to every
template. The env also names the image (`FV_IMAGE_REF`, `FV_IMAGE_DIGEST`)
and the channel (`FV_RELEASE_CHANNEL`), which fv-serve reports.

Deploy scripts boot what the templates name:

- `scripts/serve/runpod-endpoint.sh`: for a variant config
  (`FV_SERVE_CONFIG=/etc/fv/runpod-wan.toml`, or `FV_VARIANT=wan`) with no
  explicit image, the queue endpoint is created **on the published template**
  itself (`down` never deletes it); with `FV_EXTRA_ENV_JSON` (gateway worker
  role) or for load-balancer endpoints, a per-run template/endpoint boots the
  template's image digest. `[image]` / `FV_SERVE_IMAGE` still override; the
  fake config boots the all-in-one `:stable` (`:latest` until the first
  promotion).
- `scripts/serve/runpod-pod.sh`: the pod boots the image of
  `fv-serve-<variant>-pod`.
- `scripts/serve/runpod-gateway.sh validate` without an image: each pool its
  variant image, the gateway pod the CPU `gateway` image.
- Without templates (no API key, not yet synced) the scripts fall back to the
  GHCR variant tag (`:wan`, …).

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

Compressed sizes from the registry manifests (gzip layers, linux/amd64).
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
  variants carry exactly the same base, CUDA and binary layers; they were not
  run on a GPU here (the coordinator stopped GPU spend), so the H3 and LTX
  code paths (cuDNN conv3d in the LTX upsampler, cuDNN SDPA on sm_12x) are
  covered only by the soname checks above.
- **cuDNN pruning**: only `libcudnn_adv` is dropped. `libcudnn.so` loads it
  only for the legacy RNN / multi-head-attention / CTC API, which fv-serve
  does not call (it uses the legacy convolution API → `libcudnn_cnn`,
  `cudnnReduceTensor` → `libcudnn_ops`, and the backend graph API for SDPA).
  `libcudnn_graph` dlopens the three engine libraries (`precompiled`,
  `runtime_compiled`, `tensor_ir`) and the heuristic library; which engine
  serves the unified SDPA node (engine ids 11 and 8,
  docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md) is not mapped to a
  library, so all engine libraries stay. Checked: no cuDNN library needs
  `libnvJitLink` (DT_NEEDED and dlopen strings), so dropping tileiras's
  `libnvjitlink` is safe.
- **CUPTI**: loaded lazily by `cudarc/cupti` only for `FASTVIDEO_GPU_TRACE`;
  the debug image keeps it.
- **ffmpeg**: every codec, format and filter the code names (`h264_nvenc`
  with p4/p5 presets, `libx264`, `libvpx`, native AAC, mp4/ivf/h264/rawvideo/
  f32le/s16le/wav, lavfi `color`/`sine`, `crop`/`fps`/scale) is either native
  (all native components are built) or one of the three external libraries.
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
