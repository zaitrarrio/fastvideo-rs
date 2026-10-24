# ADR-0019: `fv-lab`, the run plan in `lab/`, and the pod on fv-deploy's guards

- Status: Accepted for D1a (`fv-lab run` and `fv-lab pod` for the `rtx6000` family, the `lab/tiers.toml` and
  `lab/cells/*.toml` schemas, the dry-run and payload goldens); the GPU run that shows `fv-lab run rtx6000`
  reproduces the family is the coordinator's; D1b (the other families), D1c (`report`), D1d (`gate` / `oracle`
  moved) and the retirement of `runpod-matrix.sh`, `runpod-http.sh` and the port one-offs follow (§8)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-lab
- Code: `crates/fastvideo-lab/src/{plan,run,pod,quote}.rs`, `src/bin/fv-lab.rs` (feature `cli`), `src/pod-start.sh.in`;
  `scripts/gpu/fv-lab-compare-runs.sh` (the GPU gate's comparison); fv-lab in the tools release's `gpucheck` set;
  schemas `crates/fastvideo-lab/schemas/{tiers,cell}.v1.json`; data `lab/tiers.toml`, `lab/cells/rtx6000-*.toml`;
  goldens `crates/fastvideo-lab/fixtures/*` (written by `scripts/gpu/tests/fv-lab-goldens.sh`); one seam in
  `fastvideo-deploy` (`runpod::Pod::list`, §6)
- Usage: `docs/dev/fv-lab.md`
- Plan: D1 (`docs/plan/2026-10-23-execution-plan.md`); review `docs/gaps/2026-10-22-architecture-review.md` §7.5

## Context

A GPU validation run today is two bash scripts. `scripts/gpu/runpod-http.sh` creates a Runpod pod on the runtime
image, whose start command serves `/fvscratch/runs` on port 8000, checks the driver, writes `box.txt`, `tree.txt`,
`weights.log`, fetches the tiny autoencoders and runs `scripts/gpu/runpod-matrix.sh <family>`; the driver polls
`DONE` through the pod's HTTPS proxy, fetches the summaries into `artifacts/runpod/<family>/<tag>/` and deletes the
pod. `runpod-matrix.sh` (3,200 lines) holds 30-odd families as `case` branches: each a list of `gated_cell` /
`run_cell` / `compare_cells` / `gate_cells` calls with the commands spelled out inline. The money guards of the pod are
the script's own (`runpod-price.sh`, a `nohup sleep; curl -X DELETE` backstop, three readiness cycles of 1,200 s).

The review (§7.5) asks for `fv-lab`, a Rust binary with `pod`, `run`, `report` and `gate` / `oracle`, the run plan
as data (`lab/tiers.toml`, `lab/cells/*.toml`), the pod on the `Provider` trait of C3. Its exit criterion: `fv-lab run
rtx6000` reproduces today's `rtx6000` family. D1 as a whole is weeks; this ADR decides the shape and lands **D1a**:
`run` and `pod` for that one family, proven equal to the scripts, so the later slices move the other families and
`report` onto a shape that is already validated.

## Decision

### 1. Scope of D1a

`fv-lab run <tier|cell> [--dry-run]`, `fv-lab pod {up,status,fetch,down,reap}`, `fv-lab check`, `fv-lab schema`;
`lab/tiers.toml` with the `rtx6000` tier and `lab/cells/rtx6000-{h3,ltx25}.toml` with its seven arms; the dry-run
equality and the payload golden as tests with script-generated fixtures; `fv-lab` in the runtime image. No script is
retired (§8).

### 2. Where `fv-lab` lives: a binary of `fastvideo-lab`

`fv-lab` is `crates/fastvideo-lab/src/bin/fv-lab.rs`, behind the feature `cli`, not a sibling binary of
`fv-gpucheck`:

- **No CUDA.** `fastvideo-gpucheck` links cudarc and the CUDA kernels (its image build needs nvcc, its tests the NVRTC
  gate). `fv-lab` runs on the GPU box *and* on a laptop or CI host (`pod up`, `pod fetch`, `--dry-run`, `check`), so it
  must build anywhere. It calls `fv-gpucheck` (generation, `compare-clips`, `gate`) and `verify-weights.sh` as
  processes, exactly as the script does: nothing measured is reimplemented.
- **One crate for the lab's data.** The gold files (D2, ADR-0009) are already `fastvideo-lab`'s; D3's `report tiers`
  joins gold files with run records, so the run plan, the runner and the reports share one crate.
- **fv-deploy below it.** `fastvideo-lab` depends on `fastvideo-deploy` (library half, no features) for the
  `Provider` trait, `Launch`, the guards and the lifecycle; the `cli` feature adds clap, tokio's signals and
  `fastvideo-deploy/boot`'s reqwest client. All are in `Cargo.lock`; no new crate.

The review's "`gate` / `oracle` moved or aliased" stays in `fv-gpucheck` for D1a (D1d decides moving versus aliasing).

### 3. The run plan: `lab/tiers.toml` and `lab/cells/*.toml`

serde with `deny_unknown_fields` everywhere, schemars JSON Schemas committed as `schemas/tiers.v1.json` and
`schemas/cell.v1.json` with a drift test (`tests/schema.rs`), in the style of `family.v1.json` and `gold.v1.json`.
`fv-lab check` validates both; `Lab::load` is the one reader.

**A tier** (`[[tier]]`, what `runpod-matrix.sh` calls a family) is a run plan on one GPU class:

| field | meaning | `rtx6000` (from) |
|---|---|---|
| `name` | `fv-lab run <name>`, `runs/<name>/<tag>`, `artifacts/runpod/<name>/<tag>`, pod `fv-lab-<name>-<tag>` | `rtx6000` |
| `provider` | `runpod-pod` (Vast / GCP with C3b) | `runpod-pod` |
| `gpu`, `gpu_count` (1), `cloud` | the provider's GPU type | RTX PRO 6000 Blackwell Server Edition, 1, SECURE (`RUNPOD_GPU_TYPE`, `RUNPOD_CLOUD_TYPE`) |
| `region` | `eu` only (CLAUDE.md) | `eu` |
| `image` | flavour: `runtime` = `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>` | `runtime` |
| `max_usd_per_hr` | fv-deploy's price cap | 2.5 (§7) |
| `wall_cap_s` | the backstop's deadline (≤ 4 h, fv-deploy's bound) | 14400 (`FV_POD_CAP_S`) |
| `cell_timeout_s` | `timeout --kill-after=15` per arm; 0 none | 3600 (`FV_GEN_TIMEOUT_S` as runpod-http passes it) |
| `container_disk_gb` | | 120 (`FV_CONTAINER_DISK_GB`) |
| `weights` | `volume` today, `registry` later (§4) | `volume` |
| `preflight` | `verify-weights.sh` cells for `weights.log` at pod start | the six cells runpod-http.sh checks |
| `tae` | `fetch-tae.sh` at pod start | true (`FV_SKIP_TAE` unset) |
| `cells` | cell files, in run order | `rtx6000-h3`, `rtx6000-ltx25` |

**A cell** (`lab/cells/<name>.toml`) is one workload and its arms, as the review defines it ("a profile plus a
workload plus arms"):

- `[workload]`: `model` (the fv-gpucheck pipeline), `prompt` or `prompt_set`, `seed`, `seconds`, `frames`, `width`,
  `height`, `steps`, `note`. Every value is *used*: an arm's command reads it as a placeholder (`{prompt}`, `{seed}`,
  `{seconds}`, …), and a value no arm reads is a validation error, so the declared workload is the one that runs (an
  absent value is the pipeline's or recipe's default, and the note says so).
- `[lists]`: named argument lists an arm splices with a whole element `"{@name}"` (the script's `h3_common`).
- `[[arm]]`: `name` (the run directory), `gate` (the `verify-weights.sh` cell that must pass first), `trees` (§4),
  optional `steps` (descriptive: the recipe flag sets them) and `note`, `env` (ordered `K=V`, rendered as the
  script's `env K=V … fv-gpucheck …` prefix), `argv` (fv-gpucheck's arguments; the binary is the runner's).
- `[[compare]]` (`baseline`, `candidate`, `args`) and `[[gate]]` (`baseline`, `candidate`, `kind` lossy | exact,
  `off`): `compare_cells` / `gate_cells`, after the arms. `rtx6000` has none; the runner implements both (tested on
  fakes) because the other families use them.

Placeholders: `{weights}` (the weight root), `{scratch}`, `{runs}` (the run directory), `{root}` (repository or image
root), `{arm}`, and the workload's. An unknown placeholder, a list nobody splices, a duplicate arm (per tier: arms are
run directories), a compare naming a foreign arm, or a non-`K=V` env entry fails validation.

Why argv templates and not a per-pipeline command builder: the equality with today's script is byte for byte only if
the argument order is the script's, and the families spell their commands in different orders; a template keeps the
order and the structured workload stays the single source of the values. D1b may add builders where families agree.

### 4. Weights by tree: the volume today, the registry's `hydrate` later

An arm names the trees it reads by their `scripts/gpu/weights-manifest.tsv` `dest` (`h3-base`, `ltx25`, `upscaler`, …),
and its `gate` names the `verify-weights.sh` cell that checks them. Two tests hold the names to the repository:
`trees` equals the roots of the gate's needs (`verify-weights.sh --list`, recorded in
`fixtures/rtx6000-gates.tsv`), and every `{weights}/<x>` the command reads is a listed tree (validation); every tree
is a manifest row and every gate a `verify-weights.sh` cell (`tests/lab_files.rs`).

Where the trees are is the tier's `weights`: `volume` mounts the EU volume `bz0ofkfx45` at `/workspace` and
`{weights}` is `/workspace/weights` (the only value D1a accepts). When the registry's `hydrate` lands (plan D1 / registry
WO-002), `weights = "registry"` makes the runner stage each arm's `trees` onto the container disk with `hydrate` before
its gate, and `{weights}` points there; the cells do not change. A tree not on the EU volume is not refused: the arm is
gated, recorded as skipped (`weights incomplete`), exactly as the script does; today that is `fasth3-8step` (its
`h3-8step` tree is R2-only since 2026-10-13), which `tests/lab_files.rs` pins.

### 5. `fv-lab run`: the script's frame, the same files

`run` ports `runpod-matrix.sh`'s frame, not its families: the CUDA library links in `<scratch>/fv-libs` (from
`cuda-13.pins`), the children's `PATH` / `LD_LIBRARY_PATH` / `FV_AUX_DIR`, `FASTVIDEO_LTX2_WEIGHTS` removed except
where an arm sets it, `matrix start image=<build id>`, the binary and `nvidia-smi` checks (exit 2 as the script's
FATAL), then per arm `gated_cell` (`FV_CELLS`, `FV_WEIGHTS=$W bash verify-weights.sh <gate>`) and `run_cell` (an empty
GPU first, the command in the arm directory under `timeout --signal=TERM --kill-after=15 <cap>`, its logs, the
failure excerpt, the peak line, `summary.json`), `compare_cells` and `gate_cells` with `fv-gpucheck compare-clips` /
`gate` and the gate policy, `matrix done` and `done.json`. The run directory is the script's, file for file:

```text
<scratch>/runs/<tier>/<tag>/  live.log  done.json  <arm>.weights.log  lpips-fetch.log
  <arm>/summary.json stdout.log stderr.log nvidia-smi-{start,end}.txt frames/ benchmark.json gpucheck-out/
  <arm>/frames.sha256.json
  compare/compare-clips-<b>--<c>[-<prompt>].json  gate/gate-<b>--<c>.json
```

`frames.sha256.json` is new on both sides (review of #171): after every arm that ran, both runners write every `*.png`
under `<arm>/frames` (recursive, byte order of the path, symlinks not followed) with its SHA-256, and the count:

```json
{"arm":"sol-h3","frames":[
{"path":"frame-000.png","sha256":"…"},
…
],"count":124}
```

`runpod-matrix.sh` gains `frames_digest` (called at the end of `run_cell`; it merges cleanly with #170) and `fv-lab
run` writes the same bytes (`run::frames_digest`; `tests/frames_digest.rs` compares it with the script's own function
run by the goldens script on `fixtures/frames-sample`). Both fetches already take every JSON directly in an arm
directory, so the record reaches `artifacts/` without the frames (which neither fetch takes), and two runs are compared
byte for byte after their pods are gone (§8). The runner's `nvidia-smi -L` and query lines are fatal when they fail,
as under the script's `set -e -o pipefail`.

and `pod fetch` mirrors it into `artifacts/runpod/<tier>/<tag>/` the way `runpod-http.sh fetch_results` does (plus a
`pod.json` record), so every existing summary script, `compare` and evidence reader keeps working on fv-lab runs.
The environment is the script's too (`FV_WORK`, `FV_SCRATCH`, `FV_RUN_TAG`, `FV_CELLS`, `FV_GEN_TIMEOUT_S`, `FV_LPIPS`,
`FV_GATE_POLICY`, `FV_GPUCHECK`, as flags with those defaults), so the pod's start command changes in one word.

**Equality.** `FV_DRY_RUN=1` (B6's switch, #170; main lacked it, so this PR carries the same hunks verbatim,
merge-checked against #170 to resolve to #170's file) makes the script log each arm's command line as
`dry-run <arm>: $(printf '%q ' "$@")`. `fv-lab run --dry-run` prints the same lines (`quote.rs` is bash 5.2's `printf
%q`, tested character by character). `scripts/gpu/tests/fv-lab-goldens.sh` runs the script's dry run for `rtx6000`
(volume `/workspace`, disk `/fvscratch`, tag `1234567-10241200`) into `fixtures/rtx6000-dry-run.txt`; `tests/dry_run.rs`
and `tests/fv_lab_bin.rs` assert fv-lab's lines equal it, all nine (`matrix start`, seven arms, `matrix done`);
`--check` regenerates and diffs (CI: `control.yml`). Every script runs under `env -i` there (PATH, HOME, `LC_ALL=C`),
so the caller's `FV_*` / `RUNPOD_*` cannot change a fixture.

### 6. `fv-lab pod`: fv-deploy's lifecycle, reused

`pod up --sha <7> --family <tier> [--yes]` names the run `<sha>-<MMDDHHMMSS>-<4 random hex>` (the pod
`fv-lab-<tier>-<tag>`; `runpod-http.sh`'s minute stamp repeats, and fv-deploy's exit sweep deletes by exact name, so two
runs in one minute would delete each other's pod), refuses to create when a live pod already has that name, and builds
a `Launch` (the runtime image at that commit, the tier's GPU, disk and
caps, the EU volume in `EU-RO-1`, ports `8000/http`, `dockerStartCmd ["/bin/bash", "-c", <start command>]`) and runs
`fastvideo_deploy::lifecycle::Deployer::up` with `keep` and no smoke: the region guard, the $8 floor projected over
the cap, the price cap on the quote and on the billed price, the detached backstop (this binary as `fv-lab backstop`,
the same handshake), and the 600 s / 3-cycle readiness rule, all fv-deploy's code. Without `--yes` it prints the plan.
Two things are fv-lab's:

- **Readiness** is the run directory's server answering `GET /<tier>/` 200 (what `runpod-http.sh wait_up` polls); a
  lab pod serves files, not fv-serve's `/ping`. `LabPod` wraps `runpod::Pod` and overrides only `Provider::reach`.
- **The start command** is `runpod-http.sh start_cmd` (idle watchdog with the pod-scoped key Runpod injects, results
  server, `box.txt`, driver check, `tree.txt`, `weights.log`, `fetch-tae.sh`, `DONE`), generated from the script into
  `src/pod-start.sh.in` with its variables as markers, `bash …/runpod-matrix.sh <family>` replaced by
  `/opt/fastvideo-rs/target/release/fv-lab run <family>`.

**Payload golden.** `runpod-http.sh plan <sha> <tag>` (added here: the body `run` would send, no create, no price
check; `pod_payload` factored out of `create_pod` unchanged) with `FV_POD_PREFIX=fv-lab` writes
`fixtures/runpod-http-rtx6000.json`; `tests/payload_golden.rs` asserts fv-lab's `POST /pods` body equals it except:

1. the runner line (`fv-lab run` for `bash runpod-matrix.sh`);
2. `env: {}` (fv-deploy's `create_body` always sends `env`; the script only with `FV_POD_FILES`, which fv-lab lacks);
3. at `up` only, the image pinned to its digest (`ghcr_resolve`), as fv-deploy does.

`pod fetch <id> [--wait] [--down]` is `wait_done` (the `DONE` marker carries the tag; the pod's state; new `live.log`
lines) plus `fetch_results` (and `DONE` itself). A run is a success only when the fetched `DONE` says `done <tag>`,
`matrix.out` holds `matrix_exit=0` and no other exit, and the driver was not too old (`pod::judge_run`); anything else
fails the command. `--down` (and `up --wait`) delete the pod whatever the wait or the fetch did, then report both.
`pod status` reads the provider state, the server and the last log lines; `pod down` deletes with fv-deploy's retries
and ledger (`artifacts/fv-lab/ledger.tsv`).

`pod reap [--yes]` takes only the pods this machine's `pod up` recorded (`artifacts/fv-lab/pods/<id>.json`) that are
live and older than their recorded cap plus 10 minutes (CLAUDE.md "only touch pods you created"). The account-wide
sweep is explicit, `--account`, and takes only names of the full pattern `fv-lab-<known tier>-<sha7>-<MMDDHHMMSS>-<4
hex>`, past that tier's cap: a `runpod-http.sh` pod made with `FV_POD_PREFIX=fv-lab` (minute stamp, any cap) never
matches.

The one seam in `fastvideo-deploy`: `runpod::Pod::list()` (`GET /pods` as `PodListing { id, name, state, created_at,
usd_per_hr }`), because `reap` selects by id or name pattern and age, `up` checks that its name is free, and the trait's
`sweep` deletes by exact name only.
`RunpodApi::rest` stays crate-private; the method is 40 lines with a unit test.

### 7. Differences from the scripts, and why

| | scripts | fv-lab | why |
|---|---|---|---|
| price cap | $5/hr (`RUNPOD_GPU_MAX_DPH`) | $2.50/hr (tier) | fv-deploy's guard (ADR-0018 §5); the PRO 6000 quote is $2.49 |
| readiness | 1,200 s × 3 (`FV_BOOT_WAIT_S`) | 600 s × 3 | the owner's rule (plan §1) |
| balance | none | $8 floor projected over the cap | CLAUDE.md |
| backstop | `nohup sleep; curl` | fv-deploy's detached backstop | handshake, early stand-down only on two 404s |
| image | tag | digest-pinned at `up` | the run names the binary it ran |
| `remote.sh env` | called, output unused (`/tmp/fv-env.json`) | not called | its only effect is a JSON nobody reads |
| `FV_PROMPTS`, `FV_PROMPT`, `FV_SEED` | override the family's commands | ignored, each with a warning | a cell's prompts and seeds are its workload |
| run tag | `<sha>-<MMDDHHMM>` | `<sha>-<MMDDHHMMSS>-<4 hex>`, name checked free | unique pod names (§6) |
| run verdict | `matrix_exit≠0` or an old driver fail | also no `DONE` or no `matrix_exit=0` fail | a run that did not finish is not a success |
| `FV_POD_FILES`, `FV_POD_SCRIPT`, `kernels` / `all` modes | yes | no | not used by `rtx6000`; later slices if needed |
| `FV_FETCH_TREE` mirror | yes | no | D1c with `report` |

### 8. The GPU gate, and what retires when

Nothing is deleted in D1a. The gate is the coordinator's: one `fv-lab pod up` run and one `runpod-http.sh run` of the
same commit, compared by `scripts/gpu/fv-lab-compare-runs.sh <fv-lab run dir> <runpod-http run dir>`: the same arm
directories, the same `summary.json` `exit` / `skipped`, `benchmark.json` in both or neither, and byte-identical
`frames.sha256.json` (which `docs/perf/determinism.md` promises for the same request and seed on the same GPU class).
`rtx6000` has no compares, so there are no compare verdicts to match.

The commit must be one whose runtime image carries `fv-lab` **and** whose `runpod-matrix.sh` writes the frames record:
the head of PR #171 after the review fixes (or its merge commit on main), once `gpucheck-runtime-image` has published
`fastvideo-rs-runtime:sha-<7>` for it. From the repository root, with `fv-lab` built on the build pod:

```sh
SHA=<7>                                                            # that commit
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family rtx6000 --yes --wait
#   -> artifacts/runpod/rtx6000/$SHA-<MMDDHHMMSS>-<hex>/   (the pod is deleted at the end)
RUNPOD_API_KEY=… bash scripts/gpu/runpod-http.sh run $SHA       # FV_FAMILY defaults to rtx6000
#   -> artifacts/runpod/rtx6000/$SHA-<MMDDHHMM>/
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/rtx6000/$SHA-<MMDDHHMMSS>-<hex> artifacts/runpod/rtx6000/$SHA-<MMDDHHMM>
```

Pass: exit 0, every arm `same`, `fasth3-8step` skipped (`weights incomplete`) in both. Then:

- **D1b**: the other families as tiers and cells (`rtx5090`, `fastvideo`, `techniques`, `b6`, `h3arms`, … one PR per
  few families, each with its dry-run golden; TAE gating and prompt sets join the cell schema); `runpod-http.sh run`
  then calls `fv-lab pod up --wait`; `runpod-matrix.sh`'s branches are deleted as they move; when the last moves, the
  script is deleted, and `runpod.sh cmd_rtx6000` with it.
- **D1c**: `fv-lab report {summary,table,tiers}` replacing the `*-summary.py`, `hd_report.py`, `sol_bench_table.py`,
  `timings.sh`; the `FV_FETCH_TREE` mirror.
- **D1d**: `gate` / `oracle` moved into or aliased by `fv-lab` (`oracle.sh`, the `oracle` family).
- **D1e**: the 24 port one-offs in `scripts/gpu/experiments/` (F1) become cells or are deleted; `runpod-http.sh`
  is deleted once nothing calls it (`upstream` mode moves with D1d).

## Open decisions

1. **The tier price cap**: $2.50/hr (fv-deploy's guard) instead of the script's $5. A PRO 6000 quote above $2.50 refuses
   the run; raising `max_usd_per_hr` is one line.
2. **Until the next tools release**: `fv-lab` rides in the tools release's `gpucheck` set (`out/fv-lab`; built by
   `release-artifacts-pod.sh build_gpucheck`, cooked by `tools-deps.sh gpucheck`, its `-V` checked by
   `tools-release.sh`'s gate, handed to the image as `lab-build=<set>` by `prebuilt.sh`), as fv-boot rides in the
   serve sets. Releases from before this one lack it, so the first runtime (and debug serve) images after the merge
   still compile the `lab-build` stage (a notice in the job says so) until a tools release is published from a commit
   with fv-lab; branch images, which compile everything anyway, are unaffected.
3. **Gate placement in a cell**: compares and gates run after a cell's arms; a family that interleaves them (rtx5090
   per resolution) splits into one cell per loop iteration. Acceptable until D1b shows otherwise.

## Consequences

- One reader of the run plan, with schemas and validation; the `rtx6000` family exists once, as data.
- `fv-lab run` and `runpod-matrix.sh` produce the same commands (tested), the same files and logs (tested on fakes),
  and their pods differ only as listed (tested); the GPU run decides the retirement.
- The pod's money and readiness guards are fv-deploy's, so a fix there covers fv-lab.
- The runtime image grows by one CPU binary and `lab/`; CI checks the goldens (`control.yml`, which also self-tests
  `fv-lab-compare-runs.sh`), the crate's tests with `cli` (`coverage.sh`), and the image's `fv-lab --version`, `check`
  and a dry run. With a tools release that carries fv-lab, no runtime image compiles it.
- Every `rtx6000` run (either runner) now carries per-arm frame hashes, so byte identity is checkable after the pod is
  gone.

## Validation

CPU, on the build pod: `cargo test --locked -p fastvideo-lab --all-features`, `cargo test --locked -p fastvideo-deploy
--all-features`, `cargo clippy --locked --workspace --all-targets -- -D warnings` (and with `--features
fastvideo-lab/cli`), `cargo fmt --all --check`, `bash scripts/gpu/lint.sh`, `bash scripts/gpu/tests/fv-lab-goldens.sh
--check`.

GPU (the coordinator's): the procedure of §8, on a commit whose runtime image is published with `fv-lab` in it and
whose `runpod-matrix.sh` writes `frames.sha256.json`. Expected: the six arms whose trees are on the EU volume run, `fasth3-8step` is recorded as skipped
(`weights incomplete`), each run directory holds the files of §5, and `fv-lab-compare-runs.sh` exits 0.
