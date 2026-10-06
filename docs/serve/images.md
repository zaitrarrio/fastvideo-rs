# fv-serve images: one per pod variant, published to GHCR and Runpod

Date: 2026-09-28; shared layer stack 2026-10-06 ([Lean runtime
images](#lean-runtime-images-2026-10-06)). Build: `docker/gpucheck.Dockerfile`
(stages `base-os`, `base-cuda` … `serve-<variant>`, `runtime`, `serve`), CI:
`.github/workflows/serve-image.yml` + `scripts/serve/ci-images.sh` +
`scripts/ci/base-images.sh`, Runpod publishing:
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
| wan5b | `runpod-wan5b.toml` (also carries `runpod-wan14b.toml`, the `wan14b-turbo` tier: set `FV_CONFIG=/etc/fv/runpod-wan14b.toml`) | `:wan5b`, … | `fv-serve-wan5b-sls` / `-pod` |
| sfwan | `runpod-sfwan.toml` | `:sfwan`, … | `fv-serve-sfwan-sls` / `-pod` |
| cpu (CPU only, fake engine; `gateway` until 2026-10-06) | `runpod-fake.toml` | `:cpu`, … | `fv-serve-cpu-pod` |
| debug (the old all-in-one tags; since 2026-10-06 the `runtime` image + fv-serve, on the shared layers) | every config, `runpod.toml` default | `:sha-<sha>`, `:latest`, `:stable` | none |

Each image also carries `runpod-fake.toml` (CI smoke check, fake engine).
`<sha>` tags are immutable. Two release channels move over them
([releases.md](releases.md)): `latest` (`:latest`, `:<variant>`,
`:<variant>-latest`) is the newest green main build, moved by CI;
`stable` (`:stable`, `:<variant>-stable`) moves only when a build is
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
base-os:   ca-certificates + libx264/libvpx/libdav1d  every image
           ffmpeg + ffprobe (minimal FFmpeg 4.4)
base-cuda: cuBLASLt                                  every CUDA image
           cuDNN precompiled engines                 (4 layers, pulled in parallel,
           cuDNN core (graph, ops, cnn, heuristic,    unchanged until a CUDA pin
             runtime-compiled + tensor-IR engines)    or the ffmpeg build changes)
           cuBLAS + NVRTC (+ builtins)
           ld.so config
 ├─ serve-cuda-bin: fv-serve --features cuda,http-client  -> serve-<variant>: config + FV_VARIANT
 ├─ runtime: sshd/rsync/curl, CUPTI, hf-fm, scripts/gpu, oxide, fv-gpucheck
 │   └─ serve (debug): + configs, deploy/vast/worker.py, fv-serve
base-os ─ serve-cpu: fv-serve --features http-client (no CUDA library)
```

The CUDA variants differ only in their last layer (a few KB), so a host that
has pulled any one of them pulls the next in seconds, and Runpod's host
image cache covers every family at once; `fastvideo-rs-runtime` and the
debug image add only their thin top layers to the same base. The `cpu` image (fv-control's
fake-engine workers) shares the OS and ffmpeg layers and has no CUDA
library at all (the smoke check asserts it).

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

Weights stay on the network volume (EU `jg48s6o1w0`; the US volume
`s2k01690bi` was deleted 2026-10, EU only, docs/ops/runpod-volumes.md):

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

## Prebuilt binaries (compile on the build pod, assemble on GitHub)

Since 2026-10-06 (owner decision, option 1b) no image workflow needs to
compile Rust or CUDA. The shared build pod builds a commit's binaries
(`scripts/dev/build-pod.sh release-artifacts <sha>`, docs/dev/build-pod.md
"Release artifacts") and uploads them to the R2 bucket
`fv-build-artifacts` under `artifacts/<sha>/` (`manifest.json` + one
tarball per set, deleted after 30 days). Each workflow then:

1. **Downloads** (`scripts/ci/prebuilt.sh fetch <sha> <sets>`, read-only R2
   key from repository secrets) the manifest and the sets it needs, checks
   the tarballs' and every file's sha256 against the manifest, the build id
   against `scripts/gpu/docker.sh build-id` and the fv-serve features against
   the requested ones.
2. **Assembles** with the extracted directories as **named build contexts
   that replace the compile stages** (`--build-context serve-build=<dir>`,
   `cpu-build`, `binary`, `oxide`, `hf-fm`). The Dockerfiles' `COPY
   --from=<stage>` lines take the prebuilt files unchanged, so the lean
   runtime stages (`runtime`, `serve-os`, `serve-cuda-base`,
   `serve-cuda-bin`, `serve-<variant>`, `serve-cpu`, `serve`) are exactly
   as before: no toolkit, no compiler in any runtime image. BuildKit never
   runs a replaced stage or its `builder` parent, so the log shows no
   `cargo` step (the C ffmpeg build stays, from the registry cache).
3. **Falls back** when anything is missing or does not match (no secrets,
   not built yet, older than 30 days, other features requested on dispatch):
   a `::warning::` names the reason and the command to build them, and the
   job compiles exactly as before. Recommended and implemented as the
   default so CI never deadlocks on the pod; `FV_PREBUILT_REQUIRE=1` would
   fail instead.

| workflow | sets downloaded | what no longer compiles on the runner |
|---|---|---|
| serve-image | `oxide gpucheck hf-fm serve-cuda serve-cpu` | every stage with `cargo` / `nvcc` / `tileiras` (debug + 7 variants) |
| gpucheck-runtime-image | `oxide gpucheck hf-fm` | builder, oxide, build, hf-fm |
| vast-pytorch-image | `gpucheck-vast hf-fm` | builder, build, hf-fm |
| serve-compat (`build` job) | `serve-fake` | the debug `fake,full` fv-serve (toolchain + rust-cache skipped) |
| gpucheck-t0 | `gpucheck-tests gpucheck` | `cargo test` (runs the pod's test binaries), the CUDA type-check (covered by the release `--features cuda` build), the nvrtc job's release build (runs the prebuilt `fv-gpucheck nvrtc`) |
| upstream-images, release | none | nothing compiled before either (upstream copies fv-gpucheck from the runtime image; release retags) |

Pull-request jobs use the PR head sha. Repository variables:
`FV_PREBUILT_WAIT_MIN` (poll R2 that many minutes before falling back,
default 0), `FV_PREBUILT_DISABLE=1` (always compile).

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
8000/http + 70000/tcp, `/workspace` volume mount, 30 GB disk); `cpu`
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
  itself (`down` never deletes it); with `FV_EXTRA_ENV_JSON` (worker role)
  or for load-balancer endpoints, a per-run template/endpoint boots the
  template's image digest. `[image]` / `FV_SERVE_IMAGE` still override; the
  fake config boots the all-in-one `:stable` (`:latest` until the first
  promotion).
- `scripts/serve/runpod-pod.sh`: the pod boots the image of
  `fv-serve-<variant>-pod`.
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
| `FV_R2_ARTIFACTS_ENDPOINT` | prebuilt binaries: `https://<account id>.r2.cloudflarestorage.com` | for prebuilt binaries; without the three, every workflow warns and compiles |
| `FV_R2_ARTIFACTS_ACCESS_KEY_ID`, `FV_R2_ARTIFACTS_SECRET_ACCESS_KEY` | an R2 API token with **Object Read only** on `fv-build-artifacts` (docs/dev/build-pod.md "Release artifacts") | the same |

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
| runtime ↔ debug | 1546.4 MB | **1019.8 MB** (all of runtime) |
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
and `# <<< shared base` and `scripts/gpu/cuda-13.pins`, and publishes
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
`scripts/gpu/cuda-13.pins`): `libcudart.so.13.4.92` + `libcudart.so.13` +
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
