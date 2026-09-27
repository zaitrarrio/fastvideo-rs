# fastvideo-deploy

fv-serve deploy (WP-16, [docs/serve/design.md](../../docs/serve/design.md) §6):

- `runpod`: the Rust Runpod serverless queue worker (research-deploy §1.1):
  job-take (204/400/429), ping, job-stop, progress, stream chunks, job-done
  with Fibonacci retries, shutdown drain, prestart failure. `sim` is an
  in-memory Runpod queue (worker and client URL shapes); `fv-runpod-sim`
  serves it on TCP for local runs.
- `dispatch`: the **native** job envelope run against the fv-serve router
  in-process — `{"kind":"http","method","path","headers","body","wait"}`,
  `{"kind":"stream","model","prompt","whip_url",…}`, `{"kind":"info","nvenc"}` —
  and the Vast forwarder route `POST /fv/v1/forward`.
- `env`: `RUNPOD_*` / `VAST_*` discovery (platform, public IP, mapped ports,
  ICE candidates, public base URL).
- `vast`: the Vast REST `env` object, the onstart script and the PyWorker
  forwarder (`deploy/vast/worker.py`).

Local end-to-end (queue mode on the fake engine):

```sh
cargo run -p fastvideo-deploy --bin fv-runpod-sim -- 127.0.0.1:8765 > /tmp/sim.env &
set -a; . /tmp/sim.env; set +a
FV_SERVE_MODE=runpod-queue FV_AUTH_MODE=trust-gateway FV_JOB_STORE=memory FV_ARTIFACTS=local \
  cargo run -p fastvideo-serve --features http-client -- --config configs/serve/runpod-fake.toml &
curl -s localhost:8765/v2/sim-ep/run -d '{"input":{"kind":"http","path":"/fv/v1/jobs","body":{"model":"fake-wan","prompt":"a fox"},"wait":true}}'
curl -s localhost:8765/v2/sim-ep/status/sim-1
```

Deploy: `scripts/serve/runpod-pod.sh smoke`, `scripts/serve/runpod-endpoint.sh
smoke|smoke-lb`, `scripts/serve/vast.sh plan|smoke`,
`scripts/serve/vast-serverless.sh plan|up` (image:
`ghcr.io/zaitrarrio/fastvideo-rs-serve`, built by
`.github/workflows/serve-image.yml`, always digest-pinned).
