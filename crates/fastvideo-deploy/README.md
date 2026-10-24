# fastvideo-deploy

fv-serve deploy (WP-16, [docs/serve/design.md](../../docs/serve/design.md) §6):

- `provider`, `launch`, `guard`, `lifecycle` and the `fv-deploy` binary (feature
  `deploy`; [ADR-0018](../../docs/adr/0018-fv-deploy.md), usage
  [docs/ops/fv-deploy.md](../../docs/ops/fv-deploy.md)): the `Provider` trait
  (Runpod pods and serverless queue endpoints), the `Launch` / `Instance` /
  `Offer` / `Account` types with their JSON Schemas (`schemas/*.v1.json`), a
  launch built from a `deploy/deploy.toml` preset, the shared guards (EU-volume
  region guard, balance floor, price cap, detached wall-clock backstop,
  readiness with replace) and `plan | up | status | logs | smoke | down`. The
  payloads equal the scripts' (`fixtures/deploy/*.json`, written by
  `scripts/serve/tests/fv-deploy-goldens.sh`).

  ```sh
  cargo build --release --locked -p fastvideo-deploy --features deploy --bin fv-deploy
  cargo test --locked -p fastvideo-deploy --all-features
  ```

- `boot` and the `fv-boot` binary (feature `boot`;
  [ADR-0014](../../docs/adr/0014-fv-boot.md)): the serve images' entrypoint,
  one boot sequence for every host (driver check, config materialization,
  weights plan `volume` / `r2` / `hub` / `none` / `registry` (pending
  `hydrate`), the upsampler sidecar, a GMI / Brev pod's tunnel and phase
  reports, the watchdog, exec fv-serve). Every step runs on the seams in
  `boot/sys.rs`; `boot/tests.rs` drives the sequence against a fake provider,
  `tests/boot_r2.rs` runs the real `scripts/gpu/r2-fetch.py` against a fake
  chunk source, and `fixtures/boot-protocol.json` is the report / log
  protocol fv-control's vitest checks too.

  ```sh
  cargo build --release --locked -p fastvideo-deploy --features boot --bin fv-boot
  cargo test --locked -p fastvideo-deploy --features boot
  ```

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
- `vast`: the Vast REST `env` object, the onstart script (fv-boot, else
  fv-serve on older images) and the PyWorker forwarder (`deploy/vast/worker.py`).

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
