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

# Vast (ADR-0018 §8.2): the fake-engine smoke only (no weights volume on Vast; r2 not in fv-deploy yet).
VAST_API_KEY=… ./fv-deploy up --kind vast --family fake --preset fake --channel latest --yes

# GCP (ADR-0018 §8.3): a deploy/deploy.toml [[gcp]] row; the floor is checked on the run's budget, less $4.95
# (a G4 hour) per VM created. A cold GCP boot outlasts the shared 600 s readiness bound: hence --force-guards.
GCP_SA_KEY_FILE=… FV_GCP_BUDGET_USD=25 ./fv-deploy up --kind gcp --family h3 --preset h3-turbo --channel latest \
  --cap-s 3600 --force-guards --ready-timeout-s 2700 --yes
./fv-deploy down <vm name> --kind gcp [--zone europe-west4-b]   # the zone up recorded, else --zone

# The JSON Schemas (fv-control's types are generated from them, plan C3c):
./fv-deploy schema launch      # also: instance, offer, account
```

Options of `plan` / `up`: `--region eu` (the only region with a weights volume; `us` is refused), `--kind pod|serverless|vast|gcp`,
`--image REF` (instead of `ghcr.io/zaitrarrio/fastvideo-rs-serve:<variant>-<channel>`), `--channel stable|latest`
(default `stable`; the preset launch runs fv-boot, so use `latest` or a digest built after #159 until `stable` is
promoted), `--gpu TYPE` (repeatable, fallback order; default `NVIDIA RTX PRO 6000 Blackwell Server Edition`),
`--cap-s` (1800, ≤ 14400), `--max-usd-per-hr` (Runpod 2.5 for one GPU, Vast 0.60, GCP 6.0 for the whole VM),
`--min-balance` (8, the minimum), `--ready-timeout-s` (600, ≤ 600),
`--ready-cycles` (3, ≤ 3), `--force-guards` (lifts those upper bounds). `up` only: `--yes`, `--no-smoke`, `--keep`, `--model`, `--params`, `--job-wait-s` (600),
`--state-dir`. GCP only: `--source-cidr` (repeatable; default `FV_GCP_SOURCE_CIDR`, else this machine's public
addresses as /32), `--spot`, `--idle-s` (1800; 0 off), `--secret-manager` (the VM reads the `FV_*` secrets from
Secret Manager by name; else none), `--budget-usd` (default `FV_GCP_BUDGET_USD`). On Vast and GCP `--gpu` names
provider GPU names / machine types (Vast: any GPU with a CUDA 13 driver when none is given; GCP: the row's machine).

## Environment

| Variable | Use |
|---|---|
| `RUNPOD_API_KEY` | required for everything but `plan` / `schema`; only ever an `Authorization` header, never printed, never in a launch |
| `RUNPOD_API_BASE`, `RUNPOD_GRAPHQL`, `RUNPOD_QUEUE_API`, `RUNPOD_HAPI`, `FV_RUNPOD_PROXY` | API overrides (tests, fakes) |
| `VAST_API_KEY` | Vast (`--kind vast`): as `RUNPOD_API_KEY`; secrets are Vast account env vars, never in a launch |
| `VAST_API_BASE` | Vast API override |
| `GCP_SA_KEY_JSON` / `GCP_SA_KEY_FILE`, or `GCP_ACCESS_TOKEN` | GCP (`--kind gcp`): the token is minted by `scripts/gcp/auth.sh` from the key (or given), asked before every request, only a header; a given token (about an hour of life) allows `--cap-s` up to 3000 |
| `GCP_PROJECT`, `GCP_ZONE` | the project (default: the key's `project_id`); `GCP_ZONE` is not used by commands on an existing VM (they take the recorded zone or `--zone`) |
| `FV_GCP_BUDGET_USD` | GCP's "balance": the budget of one `up`, less each created VM's price × the cap, that the $8 floor and the projection are checked on; at least $8 + price × cap (G4, 1800 s: $10.48); without it `up` creates nothing |
| `FV_GCP_COMPUTE_API` | Compute API override (tests, fakes) |

## Guards (all on by default; the numbers are flags)

1. **Region**: Runpod GPU instances mount the EU volume `bz0ofkfx45` in `EU-RO-1`, nothing else; the deleted US volume
   `s2k01690bi` and region `us` are refused. The preset's weight trees must be on the EU volume
   (`scripts/gpu/volumes.sh FV_EU_VOLUME_TREES`): H3 and LTX presets launch, Wan presets are refused (r2 / hub weights
   are not in fv-deploy yet).
2. **Balance floor over the cap**: the Runpod balance is read before every create; it must be at least $8 and stay at
   least $8 over the whole cap: `balance − (quote + current spend/hr) × cap_s/3600 ≥ 8`.
3. **Price cap**: Runpod's quote is checked before the create; no quote or a quote over the cap skips the type; a pod
   billed over the cap, or whose billed price is missing or unreadable, is deleted at once. Serverless: the pod quote
   of one worker is the only price checked (no serverless quote), so `workers_max > 1` is refused (ADR-0018 §5.3).
4. **Backstop**: after each create, before anything else, `fv-deploy backstop` runs detached (its own process group,
   SIGHUP ignored, the key only in its environment) and deletes the instance `--cap-s` seconds after its creation even
   if `up` is killed. `up` waits for it to report "armed"; without that the instance is deleted and `up` stops. It
   stands down early only after two "gone" readings a minute apart. Its log is
   `artifacts/fv-deploy/backstop-<stamp>.log`. **It runs on the host that ran `up` and dies with it**: run the live
   gate from a machine that stays up for `--cap-s`.
5. **Lost answers**: a create whose answer was lost (a network error, a 5xx, no id) stops `up`; the run's pods /
   endpoints / templates are found by name and deleted. Every exit of `up` does the same sweep (but a `--keep` pod).
6. **Readiness**: an instance not answering `/ping` 200 within 600 s of its creation is deleted and replaced, at most
   3 instances per `up` (every create counts); if a delete fails, `up` stops instead of creating another.
7. **Bounds**: `--cap-s` ≤ 4 h, `--ready-cycles` ≤ 3, `--ready-timeout-s` ≤ 600 unless `--force-guards` (never the $8
   floor).
8. **Delete on exit**: a failure after the create, a failed smoke, Ctrl-C or SIGTERM delete what `up` created (with
   retries; the ledger records `deleted` only when it was). A second Ctrl-C quits at once; the backstop still deletes
   at the cap.
9. **Serverless**: `down` deletes the endpoint, then its template; a template left behind costs nothing (the exit
   sweep retries it).

`status`, `logs`, `smoke` and `down` need `--kind` for an id `up` did not record, unless the id is a serverless
`<endpoint>/<template>` (a pod `down` of an endpoint id would report success on its 404 and leave the endpoint).

Worst case of one `up` at the defaults: three pods × 600 s of readiness, then a smoke of at most 120 s (`/ping` wait)
plus 600 s (the job; on serverless its POST is one queue job of at most 600 s): about 42 GPU-minutes, about $1.75 at
the $2.50/hr cap.

## Files

`artifacts/fv-deploy/` (not committed): `ledger.tsv` (every create, ready and delete; the scripts' ledger format),
`up-<stamp>.json` (the report: cycles, readiness time, smoke), `instances/<id>.json`, `keys/<id>.key` (mode 0600, with
`--keep`: the run key `smoke` and your own calls use as `Authorization: Bearer`), `backstop-<stamp>.log`.

## Replacing the scripts

`fv-deploy` replaces `scripts/serve/runpod-pod.sh`, the queue half of `runpod-endpoint.sh` and the create / cap half
of `scripts/gpu/runpod-http.sh`; with `--kind vast` / `--kind gcp` it does what `scripts/serve/vast.sh smoke` and
`scripts/gcp/vm.sh up` + `wait` do (their live gates are the owner's, ADR-0018 §8.4); the scripts stay until the live gate (ADR-0018 §6 lists which script
each subcommand replaces and when it is deleted). `scripts/serve/tests/fv-deploy-goldens.sh --check` keeps the Rust
payloads equal to what those scripts send.
