# fv-deploy: one preset on a provider, with the money guards

`fv-deploy` launches one instance of a `deploy/deploy.toml` preset, waits until it is ready, runs the entry smoke and
deletes it, with the shared guards (balance floor, price cap, EU-volume region guard, detached wall-clock backstop,
readiness with replace). Design: [ADR-0018](../adr/0018-fv-deploy.md). Providers today: Runpod pods (`--kind pod`,
the default) and Runpod serverless queue endpoints (`--kind serverless`); Vast and GCP follow (plan C3b).

## Build

On the build pod (CLAUDE.md "Builds"), then fetch the binary:

```sh
bash scripts/dev/build-pod.sh run . -- cargo build --release --locked -p fastvideo-deploy --features deploy --bin fv-deploy
bash scripts/dev/build-pod.sh fetch . release/fv-deploy ./fv-deploy
```

Run it from the repository root (it reads `deploy/deploy.toml`, `families/`, `configs/serve/` and
`scripts/gpu/volumes.sh`; `--root DIR` elsewhere).

## Commands

```sh
# What would be created: the Launch and the exact Runpod request(s). No network, no key needed.
./fv-deploy plan --family h3 --preset h3-turbo --region eu

# Create, wait for /ping 200, smoke, delete. Without --yes this prints the plan.
RUNPOD_API_KEY=… ./fv-deploy up --family h3 --preset h3-turbo --region eu --channel latest --yes --cap-s 1800

# Keep it running after the smoke (the backstop still deletes it at --cap-s); then use it:
RUNPOD_API_KEY=… ./fv-deploy up --family ltx2 --preset ltx --channel latest --yes --keep
RUNPOD_API_KEY=… ./fv-deploy status <pod id>
RUNPOD_API_KEY=… ./fv-deploy logs <pod id> --follow
RUNPOD_API_KEY=… ./fv-deploy smoke <pod id> [--model M] [--params '{"num_frames": 9}']
RUNPOD_API_KEY=… ./fv-deploy down <pod id>

# A serverless queue endpoint of the same preset (smoke through native queue jobs):
RUNPOD_API_KEY=… ./fv-deploy up --family h3 --preset h3-turbo --kind serverless --channel latest --yes

# The JSON Schemas (fv-control's types are generated from them, plan C3c):
./fv-deploy schema launch      # also: instance, offer, account
```

Options of `plan` / `up`: `--region eu` (the only region with a weights volume; `us` is refused), `--kind pod|serverless`,
`--image REF` (instead of `ghcr.io/zaitrarrio/fastvideo-rs-serve:<variant>-<channel>`), `--channel stable|latest`
(default `stable`; the preset launch runs fv-boot, so use `latest` or a digest built after #159 until `stable` is
promoted), `--gpu TYPE` (repeatable, fallback order; default `NVIDIA RTX PRO 6000 Blackwell Server Edition`),
`--cap-s` (1800), `--max-usd-per-hr` (2.5), `--min-balance` (8, the minimum), `--ready-timeout-s` (600),
`--ready-cycles` (3). `up` only: `--yes`, `--no-smoke`, `--keep`, `--model`, `--params`, `--job-wait-s` (600),
`--state-dir`.

## Environment

| Variable | Use |
|---|---|
| `RUNPOD_API_KEY` | required for everything but `plan` / `schema`; only ever an `Authorization` header, never printed, never in a launch |
| `RUNPOD_API_BASE`, `RUNPOD_GRAPHQL`, `RUNPOD_QUEUE_API`, `RUNPOD_HAPI`, `FV_RUNPOD_PROXY` | API overrides (tests, fakes) |

## Guards (all on by default; the numbers are flags)

1. **Region**: Runpod GPU instances mount the EU volume `bz0ofkfx45` in `EU-RO-1`, nothing else; the deleted US volume
   `s2k01690bi` and region `us` are refused. The preset's weight trees must be on the EU volume
   (`scripts/gpu/volumes.sh FV_EU_VOLUME_TREES`): H3 and LTX presets launch, Wan presets are refused (r2 / hub weights
   are not in fv-deploy yet).
2. **Balance floor**: the Runpod balance is read before every create; below $8 nothing is created.
3. **Price cap**: Runpod's quote for the GPU type is checked before the create; no quote or a quote over the cap skips
   the type; a pod billed over the cap is deleted at once.
4. **Backstop**: after each create, `fv-deploy backstop` runs detached (its own process group, SIGHUP ignored, the key
   only in its environment) and deletes the instance `--cap-s` seconds after its creation even if `up` is killed; it
   exits early once the instance is gone. Its log is `artifacts/fv-deploy/backstop-<stamp>.log`.
5. **Readiness**: an instance not answering `/ping` 200 within 600 s of its creation is deleted and replaced, at most
   3 instances, then `up` stops and reports.
6. **Delete on exit**: a failure after the create, a failed smoke, Ctrl-C or SIGTERM delete what `up` created.

## Files

`artifacts/fv-deploy/` (not committed): `ledger.tsv` (every create, ready and delete; the scripts' ledger format),
`up-<stamp>.json` (the report: cycles, readiness time, smoke), `instances/<id>.json`, `keys/<id>.key` (mode 0600, with
`--keep`: the run key `smoke` and your own calls use as `Authorization: Bearer`), `backstop-<stamp>.log`.

## Replacing the scripts

`fv-deploy` replaces `scripts/serve/runpod-pod.sh`, the queue half of `runpod-endpoint.sh` and the create / cap half
of `scripts/gpu/runpod-http.sh`; the scripts stay until the coordinator's live gate (ADR-0018 §6 lists which script
each subcommand replaces and when it is deleted). `scripts/serve/tests/fv-deploy-goldens.sh --check` keeps the Rust
payloads equal to what those scripts send.
