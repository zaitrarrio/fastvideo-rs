# fv-lab: run a tier's cells on a GPU pod

`fv-lab` runs the lab's run plan (`lab/tiers.toml`, `lab/cells/*.toml`) on a GPU box and makes that box: a Runpod pod
on fv-deploy's guards (price cap, $8 floor, EU volume, detached backstop, ready within 600 s in at most 3 cycles).
Design: [ADR-0019](../adr/0019-fv-lab.md). Today it carries ten tiers, each one a `runpod-matrix.sh` family. Each is
held equal to the script's dry run by a golden ([The families](#the-families)). The other families move in later
slices. An eleventh, `gold`, is fv-lab's own: every gold variant's canonical cell (ADR-0023).

## Build

On the build pod (CLAUDE.md "Builds"), then fetch the binary:

```sh
bash scripts/dev/build-pod.sh run . -- cargo build --release --locked -p fastvideo-lab --features cli --bin fv-lab
bash scripts/dev/build-pod.sh fetch . release/fv-lab ./fv-lab
```

The tools release carries it in its `gpucheck` set, and the runtime image `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>`
at
`/opt/fastvideo-rs/target/release/fv-lab` with `lab/` at `/opt/fastvideo-rs/lab` (and the image workflow uploads it
beside `fv-gpucheck` as an artifact). `fv-lab` finds `lab/tiers.toml` above the current directory, then above its own
binary; `--root DIR` (or `FV_LAB_ROOT`) elsewhere.

## The run plan

```sh
./fv-lab check                    # parse and validate lab/; one line per tier
./fv-lab schema tiers             # the JSON Schemas (crates/fastvideo-lab/schemas/{tiers,cell}.v1.json)
./fv-lab run rtx6000 --dry-run --work /workspace --scratch /fvscratch --tag test
```

`--dry-run` prints every arm's command line exactly as `FV_DRY_RUN=1 runpod-matrix.sh rtx6000` logs it, and writes
nothing. A tier lists cells; a cell (`lab/cells/<name>.toml`) is one workload (prompt, seed, length, canvas) and its
arms; an arm is one run directory, one fv-gpucheck command (with `{weights}`, `{runs}`, `{prompt}`, … placeholders),
gated on a `verify-weights.sh` cell and naming the weight trees it reads. Edit a cell, then `fv-lab check` and the
dry run show what will run.

## The families

| tier | cells | arms | what it runs | notes |
|---|---|---|---|---|
| `rtx6000` | `rtx6000-{h3,ltx25}` | 7 | the RTX PRO 6000 parity baseline: H3, FastH3, LTX-2.5 | D1a; `runpod-http.sh`'s default |
| `techniques` | `techniques-*` (4) | 13 | technique profiles against the baseline binary and this build, byte identity | needs `--baseline-sha` |
| `b6` | `b6-*` (8) | 16 | the loader split's load paths, baseline binary vs this build, byte identity | needs `--baseline-sha` |
| `rtx5090` | `rtx5090-*` (5) | 17 | the sol-engine RTX 5090 suite (dense, Sol, fullopt, TAE decoders) | resident on the PRO 6000 |
| `fastvideo` | `fastvideo-*` (6) | 17 | our stack without sol-engine techniques, TAE twins, lossy gates | |
| `h3arms` | `h3arms-*` (3) | 13 | FastH3 Sol / TeaCache arms gated against the plain recipe | 11 arms skip on the EU volume (`h3-8step`) |
| `trace` | `trace-{h3,ltx25}` | 5 | one CUPTI-traced warm step per workload | |
| `ltxfps` | `ltxfps-{ltx25,h3}` | 5 | LTX-2.5 at 25 / 48 / 50 fps and silent, the frame-sink checks | |
| `ltxvae` | `ltxvae-*` (4) | 12 | the LTX-2.5 VAE decode alone (`vae-bench`) | |
| `precision` | `precision-{h3,ltx25}` | 8 | precision switches against f32 activations | the script's default mode only |
| `gold` | `gold-*` (32) | 32 | every gold variant's canonical cell: the served variant through the serve engine at its gold canvas | fv-lab's own, generated from `lab/gold/` ([below](#the-gold-files-ci-and-the-tier-table)) |

The script keeps the other families. ADR-0019 §9.1 says why for each and when its branch retires. An arm whose tree is
not on the EU volume (`h3-8step`) is recorded as skipped (`weights incomplete`), as the script records it.

Cell fields beyond D1a ([ADR-0019 §9.2](../adr/0019-fv-lab.md)):

```toml
[[arm]]
name = "b6-solh3-tree-base"
bin = "baseline"            # FV_BASELINE_SHA's fv-gpucheck, fetched into <scratch>/baseline-<sha>/
env = []                    # a bare `env` before the binary, as the script writes it
tae = "taeh3.safetensors"   # read as {tae}/taeh3.safetensors; fetched once, else the arm is skipped
...

[[compare]]
baseline = "a"
candidate = "b"
gate = { kind = "lossy" }   # this pair's gate, right after the compare
```

```sh
./fv-lab run b6 --dry-run --baseline-sha <7>      # the baseline tiers need the commit
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family b6 --baseline-sha <base7> --yes --wait
./fv-lab run rtx5090 --dry-run --cells h3-480p-fullopt,h3-480p-fullopt-taeh3   # a subset, by arm
```

## On a GPU box: `fv-lab run`

```sh
FV_WORK=/workspace FV_SCRATCH=/fvscratch FV_RUN_TAG=<tag> fv-lab run rtx6000      # every arm
fv-lab run rtx6000 --cells sol-h3,ltx25-two-stage                                 # a subset (FV_CELLS)
fv-lab run rtx6000-ltx25                                                          # one cell
```

The pod's start command does this. It writes `runpod-matrix.sh`'s run directory (`<scratch>/runs/<tier>/<tag>/`:
`live.log`, per arm `summary.json`, `stdout.log`, `stderr.log`, `nvidia-smi-{start,end}.txt`, frames,
`benchmark.json`, `compare/`, `gate/`, `done.json`). Settings, as flags or the script's variables: `--work`
(`FV_WORK`), `--scratch` (`FV_SCRATCH`), `--tag` (`FV_RUN_TAG`), `--cells` (`FV_CELLS`), `--gpucheck`
(`FV_GPUCHECK`), `--cell-timeout-s` (`FV_GEN_TIMEOUT_S`, default the tier's), `FV_LPIPS=1`, `FV_GATE_POLICY`. Exit 2
when the box cannot run anything (no fv-gpucheck, no nvidia-smi, no cuDNN), as the script's FATAL.

## From a laptop or the session container: `fv-lab pod`

```sh
# The plan: the Launch and the exact POST /pods body. No key, no network.
./fv-lab pod up --sha <7> --family rtx6000

# Create the pod (runtime image at <7>, pinned to its digest), wait until it serves its run directory:
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family rtx6000 --yes
# … it prints the pod id and the tag; the pod runs `fv-lab run rtx6000` on its own.

RUNPOD_API_KEY=… ./fv-lab pod status <pod id>           # provider state, server, last log lines
RUNPOD_API_KEY=… ./fv-lab pod fetch <pod id> --wait --down   # wait for DONE, fetch, delete
RUNPOD_API_KEY=… ./fv-lab pod down <pod id>             # delete now (no pod record needed)
RUNPOD_API_KEY=… ./fv-lab pod down --from-ledger        # every pod the ledger shows created and not deleted
RUNPOD_API_KEY=… ./fv-lab pod reap [--yes]              # pods recorded here, past their cap + 10 min
RUNPOD_API_KEY=… ./fv-lab pod reap --account [--yes]    # any fv-lab-<tier>-<sha7>-<MMDDHHMMSS>-<hex4> pod past its tier's cap

# Or all of it in one command (as runpod-http.sh run):
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family rtx6000 --yes --wait
```

Options of `up`: `--cells a b` (only these arms), `--extra-env K=V …` (`FV_EXTRA_ENV`), `--lpips`, `--tag`,
`--state-dir`. The run tag is `<sha>-<MMDDHHMMSS>-<4 random hex>`, and `up` refuses to create when a live pod already
has the name. Results land in `artifacts/runpod/<tier>/<tag>/` (the same place and files as
`runpod-http.sh`, plus `pod.json`); the ledger, pod records and backstop logs in `artifacts/fv-lab/`. Run `up` from a
host that outlives the tier's wall cap (4 h for `rtx6000`): the backstop process runs there. The pod also deletes
itself after 10 idle GPU minutes (the start command's idle watchdog, with the pod-scoped key Runpod injects),
and at the tier's wall cap after its container starts (the cap watchdog, ADR-0023 §4), whatever happens to that host.
Both use GraphQL `podTerminate`: the pod-scoped key gets HTTP 403 on a REST DELETE.
After an interrupted `up` (its pod record is only written once the pod is ready), `pod down --from-ledger` deletes
what the ledger shows created and not deleted.

`fetch` (and `up --wait`) fail (exit 1) unless the fetched `DONE` says `done <tag>` and `matrix.out` holds
`matrix_exit=0`, and when the host's driver was too old; the files fetched so far are kept. `--down` and `up --wait`
delete the pod whatever happened, then report.

Every arm leaves `<arm>/frames.sha256.json` (each frame's SHA-256, written by `fv-lab run` and by `runpod-matrix.sh`
alike, fetched by both drivers). Two runs of one commit compare with

```sh
bash scripts/gpu/fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>
```

(same arms, same exits and skips, byte-identical frames records, and for the D1b families the same compare verdicts
and the same gate reports); the D1a GPU gate is that command after one `fv-lab
pod up --sha <7> --family rtx6000 --yes --wait` and one `runpod-http.sh run <7>` of the same commit
([ADR-0019 §8](../adr/0019-fv-lab.md)). The commit must be one whose runtime image carries fv-lab and the frames record:
the final head of PR #171 or later. ADR-0019 §9.5 has the D1b families' runs: the cheapest cells per family and their
cost.

## The gold files: CI and the tier table

Design: [ADR-0023](../adr/0023-lab-report-and-ci-bridge.md) (plan D3, D4).

```sh
./fv-lab gold check                         # parse, resolve against families/, validated commits exist, the age table
./fv-lab gold check --max-age-days 30       # also fail a validated variant older than 30 days (default: warn only)
./fv-lab gold cells --write                 # regenerate lab/cells/gold-*.toml after a gold or family file changes
./fv-lab gold cells --check                 # CI: the committed gold cells are current
./fv-lab run gold --all --dry-run           # the gold tier: one serve-engine arm per gold variant
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family gold --yes --wait   # the gold runs -> artifacts/runpod/gold/<tag>/
./fv-lab report tiers --runs artifacts/runpod/gold/<tag>                # docs/perf/tiers.md from that run set
./fv-lab report tiers --check               # CI: docs/perf/tiers.md is what the run set its header names renders to
```

`gold check` prints Markdown (the table CI appends to the job summary) and exits 1 on a gold file that does not parse
or resolve, or a `[last_validated].commit` that is not a commit here (outside a git work tree the commits are not
checked, and it says so). A variant never validated is a warning.

`report tiers` writes one row per gold file. A variant is measured only by the arm of its name in the run set (the gold
tier names its arms by variant id), with every value read from that arm's `summary.json`, `benchmark.json` or
`gpucheck-out/engine.json` and the run's `pod.json`; anything else is "not measured" with the reason. Options:
`--gold DIR` (other gold files), `--out FILE`, `--stdout`.

Workflows: `gold-check.yml` runs the three checks on lab, family and lab-crate changes; `gold-run.yml`
(`workflow_dispatch`, input `confirm: spend`) runs `pod up --family gold --yes --wait` and renders the table as an
artifact. It needs the `RUNPOD_API_KEY` repository secret, which is the owner's to add; without it the job refuses.

## Keeping it equal to the scripts

```sh
bash scripts/gpu/tests/fv-lab-goldens.sh           # regenerate crates/fastvideo-lab/fixtures/ from the scripts
bash scripts/gpu/tests/fv-lab-goldens.sh --check   # CI (control.yml)
```

For every tier of `lab/tiers.toml`, the fixtures are the script's dry run (`<tier>-dry-run.txt`), its weight gates
(`<tier>-gates.tsv`: the cells the dry run gated, with their `verify-weights.sh --list` needs) and `runpod-http.sh
plan`'s pod body (`runpod-http-<tier>.json`). The `gold` tier is no script family: only its pod body is written there, and its dry run is fv-lab's own golden
(`fixtures/gold-dry-run.txt`, `tests/gold_tier.rs`). Once, there is also its `frames_digest` of a sample. A new tier fails
`--check` until its fixtures are written; `tests/{dry_run,payload_golden,frames_digest,fv_lab_bin}.rs` hold fv-lab to
them. The script runs every script under `env -i`, so your `FV_*` / `RUNPOD_*` settings do not leak in.
