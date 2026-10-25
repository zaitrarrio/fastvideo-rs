# fv-lab: run a tier's cells on a GPU pod

`fv-lab` runs the lab's run plan (`lab/tiers.toml`, `lab/cells/*.toml`) on a GPU box and makes that box: a Runpod pod
on fv-deploy's guards (price cap, $8 floor, EU volume, detached backstop, ready within 600 s in at most 3 cycles).
Design: [ADR-0019](../adr/0019-fv-lab.md). Today it carries ten tiers that were `runpod-matrix.sh` families. Each
passed its GPU gate (fv-lab's run and the script's of one commit, frames byte-identical), so its script branch is
deleted and the tier is fv-lab only ([The families](#the-families), [ADR-0019 §10](../adr/0019-fv-lab.md)). Each is
held to the script's last dry run by a frozen golden ([below](#keeping-it-equal-to-the-scripts)). The other families
stay in the script and move in later slices. An eleventh, `gold`, is fv-lab's own: every gold variant's canonical
cell (ADR-0023). A twelfth, `b4a` (D1d), is moved but not yet retired: its cells carry a verdict step, and its
script branch stays until its GPU gate ([Verdicts and `fv-lab gate`](#verdicts-and-fv-lab-gate)).

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

`--dry-run` prints every arm's command line exactly as `FV_DRY_RUN=1 runpod-matrix.sh rtx6000` logged it before the
branch was deleted (the frozen golden), and writes nothing. A tier lists cells; a cell (`lab/cells/<name>.toml`) is one workload (prompt, seed, length, canvas) and its
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
| `b4a` | `b4a-h3-*` (3) | 6 | the H3 TeaCache step cache, baseline binary vs this build: byte identity and equal cache decisions, a verdict per cell | D1d; needs `--baseline-sha`; **not retired** (script branch kept until its GPU gate) |
| `gold` | `gold-*` (32) | 32 | every gold variant's canonical cell: the served variant through the serve engine at its gold canvas | fv-lab's own, generated from `lab/gold/` ([below](#the-gold-files-ci-and-the-tier-table)) |

All ten above `b4a` are **retired from the script** (`scripts/gpu/fv-lab-retired.sh`, ADR-0019 §10): their `runpod-matrix.sh`
branches are deleted, and `runpod-matrix.sh <tier>` refuses them (exit 2). The script keeps the other families.
ADR-0019 §9.1 says why for each and when its branch retires. An arm whose tree is not on the EU volume (`h3-8step`)
is recorded as skipped (`weights incomplete`), as the script recorded it.

What the retired branches had and the tiers do not, dropped with the branches: the `precision` family's
`FV_PRECISION_ARM=ltx-nvfp4` mode, `ltxvae`'s `FV_VAE_GEN=1` arms (and the `FV_VAE_*`, `FV_LTXFPS*`,
`FV_TECH_CELLS`, `FV_B6_CELLS` knobs, ADR-0019 §9.4), and the summaries the branches copied into `live.log`.
Either variant comes back as its own tier when wanted (git history has the commands).

### `runpod-http.sh` for a retired tier

`runpod-http.sh run [sha]` with `FV_FAMILY` (default `rtx6000`) set to a retired tier runs
`fv-lab pod up --sha <sha> --family <tier> --root <repo> [--cells …] [--baseline-sha …] [--extra-env …] [--lpips] --yes --wait`
instead of its own pod, mapping `FV_CELLS`, `FV_BASELINE_SHA`, `FV_EXTRA_ENV` and `FV_LPIPS=1` across; `plan [sha]
[tag]` prints fv-lab's plan (no `--yes`). The other families keep the script path. `FV_POD_CAP_S` below the tier's
`wall_cap_s` maps to `pod up --wall-cap-s` (a cap is lowered, never raised). Every other setting of the script path
that fv-lab would not honour is refused, not dropped, unless fv-lab does the same thing anyway:

- no equivalent, refused when set: `FV_POD_FILES`, `FV_POD_SCRIPT`, `FV_POD_FILE`, `FV_PROMPTS`, `FV_PROMPT`,
  `FV_SEED`, `FV_AVOID_MACHINES`, `FV_CREATE_WAIT_S`, `FV_FETCH_SKIP`, `FV_FETCH_MAX_MB`, `RUNPOD_ALLOWED_CUDA`,
  `FV_KEEP_POD=1`, `FV_FETCH_TREE=1`, `RUNPOD_NO_VOLUME=1`;
- refused unless equal to the tier's value in `lab/tiers.toml`: `FV_GEN_TIMEOUT_S` (`cell_timeout_s`),
  `FV_CONTAINER_DISK_GB`, `RUNPOD_CLOUD_TYPE`, `RUNPOD_GPU_TYPE`, `FV_SKIP_TAE=1` on a `tae = true` tier, and
  `RUNPOD_GPU_MAX_DPH` below the tier's `max_usd_per_hr` (above it is moot: fv-deploy holds the tier's);
- refused unless equal to fv-lab's fixed value: `FV_IDLE_GPU_MIN` 10, `FV_IDLE_GPU_GRACE_MIN` 30,
  `FV_IDLE_NET_BYTES` 262144, `FV_MIN_DRIVER` 580, `FV_BOOT_WAIT_S` 600, `RUNPOD_VOLUME_ID` / `RUNPOD_VOLUME_NAME` /
  `FV_PLAN_VOLUME` the EU volume, `RUNPOD_IMAGE` the commit's runtime image, `FV_POD_PREFIX` `fv-lab`;
- `FV_EXTRA_ENV` entries no tier reads: the dropped modes' knobs (`FV_PRECISION_ARM`, `FV_NVFP4_WORKLOADS`,
  `FV_VAE_*`, `FV_LTXFPS*`, `FV_TECH_CELLS`, `FV_B6_CELLS`), `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED`, and what the
  start command sets after them (`FV_WORK`, `FV_SCRATCH`, `FV_RUN_TAG`, `FV_CELLS`, `FV_GEN_TIMEOUT_S`).

`FV_CELLS` and `FV_EXTRA_ENV` are split on spaces (and `FV_CELLS` on commas) without globbing. `all` mode is refused
(run `kernels`, then `run`), and `FV_FAMILY=gold` is refused for `run` / `all` (no script family: `fv-lab pod up
--family gold`; its `plan` stays the script's body, the payload golden). The binary is `FV_LAB_BIN`, else `fv-lab`
on `PATH`, else `./fv-lab` at the repository root ([Build](#build)); without one the driver says how to build it.
`scripts/gpu/tests/fv-lab-route.test.sh` (CI: `control.yml`) holds all of this, with a fake `FV_LAB_BIN`.

```sh
FV_DRY_RUN=1 FV_FAMILY=b6 FV_BASELINE_SHA=e93e357 bash scripts/gpu/runpod-http.sh run <7>   # prints the fv-lab command
FV_DRY_RUN=1 FV_FAMILY=b4a bash scripts/gpu/runpod-http.sh run <7>   # a script family: says what it would run
```

`FV_DRY_RUN=1` never creates a pod and needs no key. `runpod.sh rtx6000` is retired with the branch (the Taskfile's
`rtx6000` task runs `runpod-http.sh run`). The script's `headline` family runs its cells of `fastvideo`, `rtx6000`,
`rtx5090` and `precision` through `fv-lab run <tier> --cells …` into its own run directory (`--runs-dir` /
`FV_RUNS_DIR`). Outside a dry run it stops (exit 2) when fv-lab is missing or `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED`
is set (the cells carry their own), and logs `FAIL fv-lab run <tier> exit=<n>` for a failed tier and goes on, as it
did with the recursive script.

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

## Verdicts and `fv-lab gate`

A cell may carry a `[verdict]` (D1d, [ADR-0019 §11](../adr/0019-fv-lab.md)): what the script's `b4a` family did
around its arms, as data. The verdict's arms start from an empty directory; right after each, its exit (`summary.json`),
its evidence files (the `stderr.log` lines matching patterns, the log's `[…]` prefixes stripped) and its line counts
are checked; after the cell's compares, its `--off-identity` pairs and its evidence diffs. The run directory then
holds `<family>-<cell>.verdict.json` (`{"cell","verdict":"pass|fail","why"}`) and, at the end, `<family>.verdict.json`
over the cells that ran; the log says `<family> <cell>: pass|fail (<reasons>)`.

```toml
[verdict]
family = "b4a"
arms = ["b4a-h3-4step-tea-base", "b4a-h3-4step-tea-pr"]     # <cell>-<label>; reasons say "pr:exit"
describe = "verdict also needs both arms exit 0 (summary.json) and the --off-identity compare passed"

[[verdict.evidence]]
file = "cache-evidence.log"
lines = ["h3 sol teacache", "h3 teacache:", "h3 teacache step", "fuse step cache"]   # substrings, * = anything
strip = "bracket-prefixes"
diff = "evidence.diff"            # the two arms' files must be equal

[[verdict.count]]
what = "teacache reused forwards"
lines = ["h3 teacache step *REUSE"]
min = 1
why = "no-reuse"

[[verdict.identical]]
baseline = "b4a-h3-4step-tea-base"
candidate = "b4a-h3-4step-tea-pr"
```

A verdict cell runs only when one of its verdict arms is selected (`--cells`); `fv-lab run b4a --cells
b4a-h3-4step-tea-base,b4a-h3-4step-tea-pr` is the script's `FV_B4A_CELLS=b4a-h3-4step-tea`.

```sh
./fv-lab run b4a --dry-run --baseline-sha <7>          # the script's dry run, verdict lines included
./fv-lab gate artifacts/runpod/b4a/<tag>               # verdicts and gates again, offline, into the run directory
./fv-lab gate artifacts/runpod/fastvideo/<tag> --gpucheck ./fv-gpucheck --out /tmp/regate   # a tier with gates
```

`fv-lab gate <run dir>` evaluates a finished run's promotion gates (`fv-gpucheck gate`, CPU only, so any build of
fv-gpucheck) and verdicts again from its files, with no GPU and without running an arm or a compare: the tier from
`--family` or the directory's parent, the arms those with a `summary.json` (or `--cells`), the output into the run
directory or `--out`. Exit 0 when all passed, 1 when one failed or nothing could be evaluated. `pod fetch` takes the
verdict files (`*.verdict.json`, `*.diff`, `<arm>/*-evidence.log`); `runpod-http.sh` only with `FV_FETCH_TREE=1`, and
`fv-lab-compare-runs.sh` compares them byte for byte when either run has them.

## On a GPU box: `fv-lab run`

```sh
FV_WORK=/workspace FV_SCRATCH=/fvscratch FV_RUN_TAG=<tag> fv-lab run rtx6000      # every arm
fv-lab run rtx6000 --cells sol-h3,ltx25-two-stage                                 # a subset (FV_CELLS)
fv-lab run rtx6000-ltx25                                                          # one cell
```

The pod's start command does this. It writes `runpod-matrix.sh`'s run directory (`<scratch>/runs/<tier>/<tag>/`:
`live.log`, per arm `summary.json`, `stdout.log`, `stderr.log`, `nvidia-smi-{start,end}.txt`, frames,
`benchmark.json`, `compare/`, `gate/`, `done.json`). Settings, as flags or the script's variables: `--work`
(`FV_WORK`), `--scratch` (`FV_SCRATCH`), `--tag` (`FV_RUN_TAG`), `--runs-dir` (`FV_RUNS_DIR`, the run directory
itself instead of `<scratch>/runs/<tier>/<tag>`), `--cells` (`FV_CELLS`), `--gpucheck`
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
plan`'s pod body (`runpod-http-<tier>.json`).

**Frozen since the GPU gate.** Once a tier passed its GPU gate and its script branch is deleted (it is listed in
`scripts/gpu/fv-lab-retired.sh`), the scripts can no longer produce its fixtures. They stay as they were written
before the branch went and are fv-lab's own goldens from then on: the goldens script does not regenerate them, only
checks that they exist, that the tier is in `lab/tiers.toml` and that its branch is gone (and that every other script
tier still has one); `tests/dry_run.rs` checks the same. The Rust tests still compare fv-lab's dry run, gates and pod
body with them, so a change to a frozen tier's commands or pod is deliberate: edit the cell (or `pod-start.sh.in`)
and the fixture in the same PR, and say why in the PR. The ten retired tiers are frozen; `b4a`'s fixtures are still written from its script branch until its GPU gate; `gold` was never a
script family (only its pod body comes from `runpod-http.sh plan`). The `gold` tier is no script family: only its pod body is written there, and its dry run is fv-lab's own golden
(`fixtures/gold-dry-run.txt`, `tests/gold_tier.rs`). Once, there is also its `frames_digest` of a sample. A new tier fails
`--check` until its fixtures are written; `tests/{dry_run,payload_golden,frames_digest,fv_lab_bin}.rs` hold fv-lab to
them. The script runs every script under `env -i`, so your `FV_*` / `RUNPOD_*` settings do not leak in.
