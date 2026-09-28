# Gateway on Runpod: one CPU gateway in front of two GPU pools

Date: 2026-09-28. `scripts/serve/runpod-gateway.sh validate` with image
`ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-bd80065`. Raw result:
`artifacts/serve/e2e/gateway/validate-0928203631.json` (no URLs, keys or
tokens in it).

## Setup

- Gateway: a Runpod **CPU pod** (`cpu3c`, 2 vCPU) running
  `fv-serve --config /etc/fv/gateway.toml` (the CUDA image; no GPU needed),
  D1 + R2 from the account's Runpod secrets, a per-run user key (only its
  SHA-256 on the pod) and internal token, the Runpod API key for the pools.
  Ready 77 s after pod create (image pull on a CPU host).
- Pools: two queue endpoints on the US volume `s2k01690bi` (US-CA-2),
  workers min 0 / max 1, idle 30 s, H100/H200 list, `FV_SERVE_ROLE=worker`:
  `h3-turbo` (`/etc/fv/runpod.toml`, `fasth3`) and `wan`
  (`/etc/fv/runpod-wan.toml`, `fastwan21-1.3b`). `h3-max`, `ltx` and
  `sfwan-live` are configured in `gateway.toml` but were not deployed.
- Every request went to the gateway's public URL with the one user key.

## Results

| API → pool | pool state | submit (HTTP, s) | submit → done via the gateway |
|---|---|---:|---:|
| native `/fv/v1/jobs` → wan | cold (scale from 0) | 202, 0.96 | 158.3 s, succeeded |
| FastVideo `/v1/videos` → wan | warm | 200, 0.72 | 5.1 s, completed |
| fal `minimax/h3-turbo/text-to-video` → h3-turbo | cold | 200, 1.16 | 212.3 s, COMPLETED |
| MiniMax `/v2/video_generation` (`MiniMax-H3-Turbo`) → h3-turbo | warm | 200, 0.91 | 27.8 s, succeeded |
| LTX `/v2/text-to-video` → `ltx` (not deployed) | endpoint missing | **503** + `Retry-After`, 0.65 | – |

Added latency (same warm wan request, native API): through the gateway
(submit, then poll the gateway's status at 0.5 s) versus straight to the
pool's endpoint (queue `/run` with the `kind: http` envelope and
`wait: true`, polling Runpod `/status` at 0.5 s):

| sample | gateway end to end | direct to the endpoint | note |
|---|---:|---:|---|
| 1 | 128.7 s | 5.9 s | the wan worker had idled out (30 s) during the h3 jobs: a cold start, not overhead |
| 2 | 5.0 s | 6.3 s | |
| 3 | 5.1 s | 6.1 s | |

- The gateway's own cost is the submit round trip: **0.7-1.2 s** (auth,
  D1 insert, Runpod `/run`; through the Runpod pod proxy).
- End to end it is **~1 s faster** than polling Runpod: the gateway reads
  the job's terminal state from D1 as soon as the worker writes it, while
  a Runpod `/status` poll waits for the worker's job-done report.
- Pool metrics after the run (`GET /fv/v1/gateway/pools`): wan run time
  mean 4.3 s (5 jobs), h3-turbo 26.5 s (2 jobs); queue wait includes the
  cold starts (wan max 153 s, h3 185 s).

## Spend and cleanup

A first attempt the same hour was stopped before any GPU job ran (the
driver did not print the refused submit; the script now logs each submit
and stops on a refusal). GPU time of the run: ~4 min wan + ~5 min h3 on
H100 serverless (≈ $0.7), CPU pod ~25 min in total (< $0.05). The shared
account balance moved from $42.97 to $40.93 over the run (other agents'
pods included). `runpod-gateway.sh down` afterwards: no `fv-gw-*` pod,
endpoint or template left (checked with the REST lists). No volume was
written.
