# fv-lab: run a tier's cells on a GPU pod

`fv-lab` runs the lab's run plan (`lab/tiers.toml`, `lab/cells/*.toml`) on a GPU box and makes that box: a Runpod pod
on fv-deploy's guards (price cap, $8 floor, EU volume, detached backstop, ready within 600 s in at most 3 cycles).
Design: [ADR-0019](../adr/0019-fv-lab.md). Today it carries one tier, `rtx6000` (the RTX PRO 6000 parity baseline that
`scripts/gpu/runpod-http.sh` runs by default); the other `runpod-matrix.sh` families move in later slices.

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
RUNPOD_API_KEY=… ./fv-lab pod down <pod id>             # delete now
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
itself after 10 idle GPU minutes (the start command's idle watchdog, with the pod-scoped key Runpod injects).

`fetch` (and `up --wait`) fail (exit 1) unless the fetched `DONE` says `done <tag>` and `matrix.out` holds
`matrix_exit=0`, and when the host's driver was too old; the files fetched so far are kept. `--down` and `up --wait`
delete the pod whatever happened, then report.

Every arm leaves `<arm>/frames.sha256.json` (each frame's SHA-256, written by `fv-lab run` and by `runpod-matrix.sh`
alike, fetched by both drivers). Two runs of one commit compare with

```sh
bash scripts/gpu/fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>
```

(same arms, same exits and skips, byte-identical frames records); the D1a GPU gate is that command after one `fv-lab
pod up --sha <7> --family rtx6000 --yes --wait` and one `runpod-http.sh run <7>` of the same commit
([ADR-0019 §8](../adr/0019-fv-lab.md)). The commit must be one whose runtime image carries fv-lab and the frames record:
the final head of PR #171 or later.

## Keeping it equal to the scripts

```sh
bash scripts/gpu/tests/fv-lab-goldens.sh           # regenerate crates/fastvideo-lab/fixtures/ from the scripts
bash scripts/gpu/tests/fv-lab-goldens.sh --check   # CI (control.yml)
```

The fixtures are the script's dry run, its weight gates (`verify-weights.sh --list`), `runpod-http.sh plan`'s pod
body and its `frames_digest` of a sample; `tests/{dry_run,payload_golden,frames_digest,fv_lab_bin}.rs` hold fv-lab to
them. The script runs every script under `env -i`, so your `FV_*` / `RUNPOD_*` settings do not leak in.
