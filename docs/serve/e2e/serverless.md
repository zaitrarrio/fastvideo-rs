# WP-18 test E: Runpod serverless with the real CUDA backend

Date: 2026-09-28. Earlier serverless runs (WP-16) used the fake engine; these
run `fv-serve` with `[engine] backend = "cuda"` from the `serve` image.
Raw results (one JSON per endpoint, driver log, `/health` samples every
10 s): `artifacts/serve/e2e/serverless/`. Signed R2 query strings are removed
from the committed JSON.

## Setup

- Image `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-9c42844`
  (`sha256:3beb71e4…`) for the queue runs; `sha-4c8c773` (`sha256:7b9f2796…`,
  the weights-link fix below) for the load-balancer run. Digest-pinned by
  `scripts/serve/runpod-endpoint.sh`.
- Endpoints and templates `fv-e2e-e-*` from `runpod-endpoint.sh up` /
  `up-lb` (`FV_ENDPOINT_PREFIX=fv-e2e-e`, `FV_IDLE_TIMEOUT_S=30`): workers
  min 0 / max 1, FlashBoot off, execution timeout 1 800 s, wall-clock
  backstop, secrets as `{{ RUNPOD_SECRET_fv_* }}` references.
- Volume: US `s2k01690bi` (US-CA-2), GPU list H200, H100 NVL, H100 80GB HBM3
  (plus RTX PRO 6000 for wan). The worker was an **H100 80GB HBM3** (driver
  580.126.09) every time.
- Configs: `/etc/fv/runpod.toml` (h3-turbo, resident `fasth3`) and
  `/etc/fv/runpod-wan.toml` (wan-turbo, resident `fastwan21-1.3b`).
- Jobs: queue `/run` with the native envelope, `wait: true`:
  - h3-turbo: fal `POST /minimax/h3-turbo/text-to-video`
    `{"prompt":"A red fox trots through fresh snow at dawn…","seed":1}`
    (seed 2 for the warm job); 1344x768, 124 frames + AAC.
  - wan-turbo: native `POST /fv/v1/jobs` `{"model":"fastwan21-1.3b",…}`;
    848x480, 81 frames.
  - then `{"kind":"info","nvenc":true}` on the same worker, which reports
    `process_start_unix` and `ready_after_s`.
- Driver: a small wrapper over `runpod-endpoint.sh up|job|down` (cold job,
  warm job, info job, fetch each MP4 from R2 and ffprobe it, delete).

## Timeline (seconds from `/run` submit)

| | worker initializing (`/health`) | process start | models loaded (`ready_after_s`) | job taken (`delayTime`) | output (job COMPLETED) | execution |
|---|---:|---:|---:|---:|---:|---:|
| h3-turbo cold (scale from 0) | +21 | **+80.3** | +132.6 (load 52.3) | +132.4 | **+202.4** | 69.5 (fal `inference` 17.5) |
| h3-turbo warm (next job) | – | – | – | +0.0 | **+25.1** | 24.4 (inference 17.5) |
| wan-turbo cold (scale from 0) | (not sampled) | **+49.3** | +119.7 (load 70.4) | +119.6 | **+126.6** | 6.4 |
| wan-turbo warm | – | – | – | +0.0 | **+7.0** | 6.4 |
| info job (either) | – | – | – | +0.0 | +0.7-0.8 | 0.7 |

Load balancer (wan-turbo, `sha-4c8c773`): endpoint create → `/ping` 200 in
**415 s** (every earlier `/ping` hung up to curl's 150 s limit with no
answer, so no 204 was seen); then `POST /v1/videos/sync`
`{"model":"fastwan21-1.3b",…}` → **302** (Location = the artifact URL) in
**6.8 s**.

Image pull: the queue workers started 49-80 s after submit, which matches a
host that already had the image layers (a fresh H200 host took 458 s in
docs/gaps/2026-09-27-cold-start.md); the ~6 GB image was not pulled cold
here, so a not-cached number is still the older 458 s one.

## Results

| check | result |
|---|---|
| queue h3-turbo cold job, fal T2V, `wait: true` | **PASS**: COMPLETED, fal status 200, MP4 in R2 (`<account>.r2.cloudflarestorage.com`, 200, 3 234 357 B), h264 1344x768 124 frames + AAC |
| queue h3-turbo warm job | **PASS**: 25.1 s end to end, 3 668 582 B MP4 from R2 |
| queue wan-turbo cold + warm | **PASS**: `succeeded`, recipe `fastwan21-1.3b-dmd3-vsa`, h264 848x480 81 frames from R2 (1 064 742 / 840 159 B) |
| `kind: info` | **PASS**: engine `cuda`, jobs `d1`, artifacts `s3` (R2), webhook key set, weights root `/runpod-volume/weights` lists all 19 trees |
| NVENC | **not available on H100**: `h264_nvenc` built, `OpenEncodeSessionEx failed: unsupported device (2)` / `No capable devices found` (H100 has no NVENC); the post encoder fell back to `cpu-test-x264`, the director to OpenH264 |
| LB endpoint `/ping` + one sync job | **PASS** (302 to the MP4 as designed; the MP4 itself was not fetched) |
| EU (`jg48s6o1w0`, RTX PRO 6000 Server/Workstation) | **no worker in 22 min**: `/health` stayed at 0 workers (not initializing, not throttled); the GraphQL endpoint view showed no pods; the endpoint was deleted unused. Widening `allowedCudaVersions` to 13.0/12.9/12.8 did not help |

## Findings

1. **Weights links (fixed).** The volume's HF-cache trees link absolutely
   into `/workspace/weights` (where pods mount the volume); serverless
   workers mount it at `/runpod-volume`. `scripts/gpu/serverless-worker.sh`
   linked the path, `fv-serve` did not. Now `fv-serve` links
   `/workspace/weights` → `$FV_WEIGHTS` at startup when that path is free
   (`deploy::link_pod_weights`, `4c8c773`), and the queue template's
   entrypoint does the same for older images. The queue runs above used the
   entrypoint link; the LB run used the image fix.
2. **First H3 job pays ~45 s extra.** The cold h3-turbo job ran 69.5 s against
   24.4 s warm with the same 17.5 s denoise: first-request setup outside the
   denoise (not seen for wan-turbo, 6.4 s both). `warmup = true` in the
   config (design §6.3) would move it before readiness.
3. **FastWan does not cold start faster than H3 here.** Loading
   `fastwan21-1.3b` took 70 s against 52 s for h3-turbo (E12/E13 fast
   loading covers H3; the Wan text encoder path does not have it).
   Submit → first output was still shorter (127 s vs 202 s) because of the
   H3 first-job overhead.
4. **H3 load via fv-serve on the US volume is 52 s** (cold host, H100), in
   line with the 57 s device-merge number of the cold-start doc, not the
   ~6 min WP-11 pod note.
5. **Hopper (H100, H200) has no NVENC**, so on this GPU list the post
   encoder is the CPU x264 fallback; list NVENC-capable GPUs (RTX PRO 6000,
   L40S, RTX 6000 Ada) when hardware encoding matters.
6. The LB endpoint did not answer `/ping` at all while the worker booted
   (no 204): the Runpod gateway holds the request until a worker is up.

## Spend and cleanup

Worker time billed (H100 80GB): h3 ~4 min, wan ~2 min, LB ~7 min ≈ 13 min,
**≈ $0.9** (the account balance also moves with the other agents' pods).
Deleted and verified absent (REST `/endpoints`, `/templates`): endpoints
`1bumthyr4t8tt5` (EU, never got a worker), `w9nun9n2evt009`,
`102wh2sgqet2i7`, `qmoctm732zz5qs` (LB); templates `5xubi545dl`,
`kqb3xwrdwx`, `d0i9dap671`. No volume was written.
