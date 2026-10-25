# ADR-0019: `fv-lab`, the run plan in `lab/`, and the pod on fv-deploy's guards

- Status: Accepted for D1a (`fv-lab run` and `fv-lab pod` for the `rtx6000` family, the `lab/tiers.toml` and
  `lab/cells/*.toml` schemas, the dry-run and payload goldens; GPU gate passed) and D1b (nine more families as tiers,
  four additive cell-schema fields, a golden per tier, §9; GPU gate passed for all nine). The ten families' script
  branches are deleted and the tiers are fv-lab only, with frozen goldens (§10, 2026-10-25). Two dropped modes are
  tiers again, frozen from the script at `5987e6a4` (§11, proposed; GPU check pending). D1c (`report`), D1d
  (`gate` / `oracle` moved) and the retirement of the rest of `runpod-matrix.sh`, `runpod-http.sh` and the port
  one-offs follow (§8)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-lab
- Code: `crates/fastvideo-lab/src/{plan,run,pod,quote}.rs`, `src/bin/fv-lab.rs` (feature `cli`), `src/pod-start.sh.in`;
  `scripts/gpu/fv-lab-compare-runs.sh` (the GPU gate's comparison); fv-lab in the tools release's `gpucheck` set;
  schemas `crates/fastvideo-lab/schemas/{tiers,cell}.v1.json`; data `lab/tiers.toml`, `lab/cells/*.toml` (D1a
  `rtx6000-*`, D1b `techniques-*`, `b6-*`, `rtx5090-*`, `fastvideo-*`, `h3arms-*`, `trace-*`, `ltxfps-*`, `ltxvae-*`,
  `precision-*`);
  goldens `crates/fastvideo-lab/fixtures/*` (written by `scripts/gpu/tests/fv-lab-goldens.sh`, frozen for the tiers of
  `scripts/gpu/fv-lab-retired.sh`, §10); one seam in
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

- **D1b** (§9): the other families as tiers and cells (`techniques`, `b6`, `rtx5090`, `fastvideo`, `h3arms`, `trace`,
  `ltxfps`, `ltxvae`, `precision` in the first D1b PR, each with its dry-run golden; TAE gating and the baseline
  binary joined the cell schema, prompt sets wait for a family that needs them); a family's `runpod-matrix.sh`
  branch is deleted once its GPU run through both runners matches (§9.5), and `runpod-http.sh run` then calls
  `fv-lab pod up --wait` for it (done for `rtx6000` and the nine, §10, with `runpod.sh cmd_rtx6000`); when the last
  family moves, the script is deleted.
- **D1c**: `fv-lab report {summary,table,tiers}` replacing the `*-summary.py`, `hd_report.py`, `sol_bench_table.py`,
  `timings.sh`; the `FV_FETCH_TREE` mirror.
- **D1d**: `gate` / `oracle` moved into or aliased by `fv-lab` (`oracle.sh`, the `oracle` family).
- **D1e**: the 24 port one-offs in `scripts/gpu/experiments/` (F1) become cells or are deleted; `runpod-http.sh`
  is deleted once nothing calls it (`upstream` mode moves with D1d).

### 9. D1b: nine more families, four schema additions

D1b moves the `runpod-matrix.sh` families whose branches are lists of arms, compares and gates, the byte-identity
families with a baseline binary first. Each one becomes a tier in `lab/tiers.toml` and cell files in `lab/cells/`.
A golden made from the script holds it equal (§9.3). No branch is deleted yet: the coordinator's GPU run of a family
through both runners (§9.5) retires its branch.

**Moved** (tier: cells / arms): `techniques` (4 / 13), `b6` (8 / 16), `rtx5090` (5 / 17), `fastvideo` (6 / 17),
`h3arms` (3 / 13), `trace` (2 / 5), `ltxfps` (2 / 5), `ltxvae` (4 / 12), `precision` (2 / 8, the default mode only).
With `rtx6000` that makes 10 tiers, 38 cell files and 113 arms. Every tier is the `rtx6000` tier's pod, because
`runpod-http.sh` gives every `FV_FAMILY` the same pod: the PRO 6000, its six preflight cells, `FV_GEN_TIMEOUT_S`
3600, the TAE fetch and a 4 h cap.

#### 9.1 Inventory of `runpod-matrix.sh`

Trees on the EU volume (`bz0ofkfx45`, volumes.sh `FV_EU_VOLUME_TREES`): `h3-base`, `FastH3-4-step-Preview-v1-LoRA`,
`h3-ref2va`, `ltx25`, `ltx25-dev`, `ltx25-ic-lora-ingredients`, `upscaler`, `h3-to-ltx`, `auxiliary/{tae,lpips}`.
"Steps between arms" means steps other than compares and gates that run between arms, or that read what an arm
wrote. ADR-0019 open decision 3 covers compares and gates placed between arms.

| family | arms (compares / gates) | trees; on the EU volume | steps between arms | schema | D1b | retire-when |
|---|---|---|---|---|---|---|
| `rtx6000` | 7 | H3, LoRA, upscaler, h3-to-ltx, ltx25 yes; `h3-8step` no (1 arm skipped) | none | as is | moved (D1a) | **retired (script branch deleted, fv-lab only)**, §10: D1a gate on `492e28a`, `sol-h3` (248) and `ltx25-two-stage` (242 frames) byte-identical; `artifacts/runpod/rtx6000/492e28a-*` |
| `techniques` | 13 in 4 cells (9 / 0) | yes, but `h3-8step` (3 arms) | a compare pair after each `tech_h3` call: one cell per call | `bin = "baseline"`, `env = []` | moved | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, base `e93e357`, `h3-480p-fullopt` base/env/prof, 3 arms × 248 frames byte-identical, 9 reports the same; `d1b-compare/techniques.txt` |
| `b6` | 16 in 8 cells (8 / 0) | yes (h3-base, LoRA, h3-ref2va, ltx25, ltx25-dev, ic-lora) | a compare after each pair: one cell per pair; per-arm stage-log lines in live.log (log only) | `bin`, `env = []` | moved | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, base `e93e357`, `b6-solh3-tree` base/pr, 2 arms × 124 frames byte-identical, 8 reports the same; `d1b-compare/b6.txt` |
| `b4a` | 6 (3 / 0) | yes | after each arm: TeaCache reuse count and `cache-evidence.log`; after each pair: the evidence diff, `b4a-<cell>.verdict.json` | no: a verdict step that is neither a compare nor a gate | stays | a typed evidence/verdict step (D1d, with `gate`) |
| `rtx5090` | 17 in 5 cells (14 / 0) | yes (h3-base, ltx25, TAE) | compares after each resolution and workload: one cell per loop iteration | `tae` | moved | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, `h3-480p` dense/sol/fullopt/fullopt-taeh3, 4 arms × 248 frames byte-identical, 14 reports the same; `d1b-compare/rtx5090.txt` |
| `fastvideo` | 17 in 6 cells (7 / 7) | yes, but `h3-8step` (4 arms) | per iteration: one cell each | `tae`, paired gate | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 4 arms the same (2 ran, 248 frames each byte-identical; 2 `h3-8step` skipped in both), 8 reports and the gate report the same; `d1b-compare/fastvideo.txt` |
| `h3arms` | 13 in 3 cells (13 / 10) | `h3-8step` for 11 arms (skipped); 4-step VSA yes | a compare followed by its gate; per-arm reuse counts in live.log (log only) | paired gate, `tae` | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 13 arms the same (2 ran byte-identical, 11 `h3-8step` skipped in both), 14 reports the same; `d1b-compare/h3arms.txt` |
| `trace` | 5 in 2 cells | yes, but `h3-8step` (1 arm) | `/gpu_trace` lines copied into live.log after the arms (log only) | as is | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 2 arms the same (1 ran, 248 frames byte-identical; 1 skipped in both); `d1b-compare/trace.txt` |
| `ltxfps` | 5 in 2 cells | yes, but `h3-8step` (1 arm) | PASS/FAIL lines copied into live.log (log only) | as is | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 2 arms the same (1 ran, 121 frames byte-identical; 1 skipped in both); `d1b-compare/ltxfps.txt`; `FV_LTXFPS*` dropped |
| `ltxvae` | 12 in 4 cells (2 / 0) | yes (ltx25) | a compare after the first two arms of each workload: two cells per workload; `FV_VAE_GEN=1` arms (off by default) | as is | moved (default arms) | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, `ltxvae-4k5s` fast/streaming, 2 arms × 121 frames byte-identical, the compare the same; `d1b-compare/ltxvae.txt`; `FV_VAE_GEN` dropped |
| `precision` | 8 in 2 cells (8 / 6) | ltx25 yes; `h3-8step` (4 arms) | compare + gate pairs | paired gate | moved (default mode) | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, `ltx25-4k5s-sol` f32act/bf16act, 2 arms × 242 frames byte-identical, 9 reports and the gate report the same; `d1b-compare/precision.txt`; the `ltx-nvfp4` mode dropped |
| `fuse` | 10 (3 / 3) | yes, but `h3-8step` | an ungated `kernels` run_cell; one compare block across H3 and LTX arms | needs ungated arms | stays | a later slice |
| `attn3` | 6 (3 / 0) | yes | one compare block across H3 and LTX arms (one cell, two pipelines) | almost (`model` is one pipeline) | stays | a later slice |
| `ltx` | 3 | ltx25 yes; ltx23, ltx2 no | ungated `run_cell` | needs ungated arms | stays | a later slice |
| `eval` | 5 (3 / 3) | yes, but `h3-8step` | a prompt set and LPIPS by default; an LPIPS self-test only when LPIPS loaded | no | stays | prompt sets join the schema |
| `h3attn` | 4 processes, ~30 `--arm`s | partly | one process runs many arms (`{arm}` in its clip dir); FP8 arms dropped on a kernel FAIL | no | stays | D1e or retired |
| `hd` | 8 | yes | ffmpeg Lanczos upscale between the arms and their compares; keyframe prune; post script | no | stays | D1c (`report`) or a cell type for derived clips |
| `ltxoffload` | 3 to 6 | yes | 4K cells run only when the 512p frames hash equal (data-dependent); `env -u` | no | stays | a later slice |
| `cold` | 10 | yes, but `h3-8step` | page-cache drop before each LTX arm, PNG deletion, `env -u` | no | stays | a later slice |
| `serve-engine` | 6 | H3, LTX yes; Wan no | frames deleted after the run; `FV_ENGINE_CELLS`; `--tae-dir` without a gate | almost | stays | a later slice |
| `writer` | 7 | ltx25 yes | sysinfo, a sampler process, `dd` throughput | no | stays | D1e or retired |
| `determinism`, `det-short` | 9 to 19 | partly | prompt-set files written at run time, hashing, verdict files, frame deletion | no | stay | determinism checks move with D1c / D1d |
| `oracle`, `sfwan` | many | partly | reference-dump fetches from upstream pods, compare-dumps; `sfwan` recurses into `oracle` and `wan` | no | stay | D1d |
| `wan` | ~35 | no (Wan trees R2-only) | oracle fetches, ungated kernels, `rm` | no | stays | D1d / a Wan volume |
| `b200` | 8 | partly | TAE and Spark file probes at run time; ungated | no | stays | retired with the US volume, or D1e |
| `h3` | 3 | `h3-8step` no | Spark file probes; ungated | no | stays | retired (rtx6000 covers it) |
| `headline` | the others' | partly | runs other families with `FV_CELLS` subsets; since §10 its four sources (`fastvideo`, `rtx6000`, `rtx5090`, `precision`) through `fv-lab run <tier> --cells … ` into its run directory (`FV_RUNS_DIR`) | no: a tier of other tiers' arms | stays | a tier with an arm filter |
| `mmaudio`, `speechtest` | 10+ | no | Python venvs, upstream clones, Whisper | no | stay | D1e |
| `sfstream`, `sfquality` | 2 to 4 | no (sfwan R2-only) | TAEHV fetch, run lists from env | no | stay | D1e |
| `precision-debug` | 7 to 10 | partly | compare-dumps only when dump dirs exist; dump deletion | no | stays | D1e or retired |
| `identity` | from `FV_ID_CASES` | partly | cases from the environment, hashing, frame deletion | no | stays | D1e or retired |
| `solbench`, `sol-lingbot`, `sol-cosmos3` | 3 to 9 | no | per-arm timeouts, exported settings, ungated routers, prompts read from files | no | stay | D1e |
| `hunyuan`, `sana-video` | 1 to 2 | no | a `--help` probe that ends the family | no | stay | D1e or retired |

`kernels` is not a family. It is a `runpod-http.sh` mode (`kernels`, `all`) that runs `fv-gpucheck kernels` before or
instead of the family (ADR-0019 §7: not in fv-lab). Its in-family uses (`kernels-wan`, `kernels-fuse`,
`attn-fp8-kernels`) are ungated `run_cell`s. They move once an arm may have no weight gate. That step is small, but
every family that uses one also needs something more (above).

#### 9.2 The schema additions (cell v1, additive, `deny_unknown_fields` kept)

All four are optional fields. Every D1a cell parses unchanged. `schemas/cell.v1.json` is regenerated and its drift
test (`tests/schema.rs`) also names the new keys.

1. **`[[arm]] bin = "gpucheck" | "baseline"`** (`ArmBin`, default `gpucheck`). A `baseline` arm runs
   `<scratch>/baseline-<sha>/fv-gpucheck`: the binary of `FV_BASELINE_SHA`'s runtime image, which
   `scripts/gpu/fetch-baseline.sh` fetches into the same path as the script's `techniques` / `b6` / `b4a` preamble. The
   runner fetches it after `matrix start` and the GPU checks, once, when any selected cell has a baseline arm. It logs
   `baseline <the fetch's last line>`. A failed fetch is the script's FATAL (exit 2, before any arm). The commit
   comes from `fv-lab run --baseline-sha` / `FV_BASELINE_SHA`. `fv-lab pod up --baseline-sha <7>` puts it in the pod's
   run environment (`FV_EXTRA_ENV`), and refuses a tier with baseline arms without it.

   The baseline fails closed (review of #176):
   - `pod up` refuses a baseline equal to `--sha`. A byte-identity gate of a build against itself passes whatever
     the build does.
   - `pod up` refuses a `FV_BASELINE_SHA` in `--extra-env` that is not 7 hex digits, including an empty value.
   - `pod up` refuses the commit given twice (`pod::baseline_for_pod`).
   - The runner logs the baseline's `fv-gpucheck.build-id` beside the image's, and stops (FATAL) when they are the
     same build. The script has no such check. The arm is the same file as its
   `-pr` / `-env` twin but for `bin`, so `--off-identity` compares stay a byte-identity check of the two binaries.
   Why: the script's identity families have a baseline binary, and a cell arm needs a way to name it.
2. **`env = []`** (`env: Option<Vec<String>>`). If `env` is absent, the command line has no `env` prefix. If it is
   present but empty, the command line starts with a bare `env`. The script's `env ${e[@]+"${e[@]}"} "$bin"` writes
   that bare `env` (`techniques`, `b6`), and the dry-run equality is word for word. Why: without it the commands differ
   in one word.
3. **`[[arm]] tae = "<file>"` and the `{tae}` placeholder.** `{tae}` is the script's `$TAE`: `FV_TAE_DIR`; else the
   volume's `auxiliary/tae` when its `.complete` marker exists; else `<scratch>/tae`, where the pod's start command
   fetches the files. An arm with `tae` is `tae_gated_cell`: `fetch-tae.sh` runs once per run if the file is missing,
   and if the file is still missing the arm is recorded as skipped (`tae weights missing`). Then the weight gate runs.
   Validation: such an arm must read `{tae}/<file>`, and `{tae}` may not appear in an arm without `tae`. Every `tae`
   file is one that `fetch-tae.sh` fetches, and every tier with TAE arms has `tae = true`
   (`tests/lab_files.rs`). Why: the TAE-decoded twins of `rtx5090`, `fastvideo` and `h3arms`.
4. **`[[compare]] gate = { kind, off }`** (`PairedGate`). The gate of the compare's own pair runs right after that
   compare. A `[[gate]]` still runs after all of the cell's compares. Why: `h3arms` and `precision` loop
   `compare_cells b c; gate_cells b c`, `fastvideo` loops `compare; compare; gate; gate`, and the dry run (and
   `live.log`) keeps the script's order either way. Gate placement (open decision 3) is otherwise settled as
   proposed: one cell per loop iteration (`rtx5090`, `fastvideo`, `ltxvae`), one per helper call (`techniques`,
   `b6`).

`Workload` gains no field. A named sol-engine workload (`--workload 4k5s`) stays a literal in the argv, and the
cell's note names it.

#### 9.3 Equality: one golden per tier, from the script

`scripts/gpu/tests/fv-lab-goldens.sh` now reads the tier names from `lab/tiers.toml`, so a new tier with no fixture
fails `--check`. It writes `<tier>-dry-run.txt`, `<tier>-gates.tsv` and `runpod-http-<tier>.json` for every tier,
under `env -i` as before, with `FV_BASELINE_SHA=7654321`. The repository path is written as `/opt/fastvideo-rs`, the
image's root, which is what `{root}` renders on a pod (the b6 fixtures under `scripts/gpu/fixtures/`). The fake
volume's `auxiliary/tae` is marked complete, as on `bz0ofkfx45`, so `$TAE` is `/workspace/weights/auxiliary/tae`.
A cell that hard-coded `{scratch}/tae` would differ.

`<tier>-gates.tsv` comes from the dry run itself, in run order, through `FV_DRY_RUN_GATES=<file>`:
- each `gated_cell` writes `arm <name> <verify-weights.sh cell> <TAE file or ->`, and the goldens add that cell's
  needs;
- each `gate_cells` writes `gate <baseline> <candidate> <kind> <OFF arm or ->`, which the dry run's log never
  showed.

So a gate's `kind` / `off`, and whether an arm is TAE-gated, are checked too. A cell that wrote the TAE path into
its argv without `tae` would match the dry run and fail here. This replaces D1a's awk over the branch, which cannot
expand loops.

Script changes, dry run only: `FV_DRY_RUN_GATES`; a dry `tae_gated_cell` neither fetches nor checks the file, it logs
the command (before this change a dry `rtx5090` would download the TAE files); and `techniques` skips the baseline
fetch when dry, as `b6` / `b4a` already did. No family's real run changes.

`tests/dry_run.rs` asserts `dry_run_lines` equal to each tier's fixture, line for line: 10 tiers, 217 lines, one
`dry-run` line per arm. It also asserts that every arm gates on the script's cell with that cell's trees, and that the
baseline arms run `/fvscratch/baseline-7654321/fv-gpucheck` and do not render without the commit.
`tests/payload_golden.rs` asserts every tier's pod body equal to `FV_FAMILY=<tier> runpod-http.sh plan`'s (the
runner line and `env {}` excepted, §6). `tests/fv_lab_bin.rs` runs `fv-lab run b6 --dry-run --baseline-sha` against
its fixture, and `pod up --family b6` with and without `--baseline-sha`. `tests/runner.rs` runs a baseline pair, a
fetched TAE arm, a missing TAE arm (one fetch) and a paired gate on fakes, and also a failing baseline fetch (FATAL,
no arm run). `tests/lab_files.rs` checks every tree against weights-manifest.tsv or the EU volume's list
(`h3-ref2va` is a volume tree with no hub row). It also checks that the only tree any tier needs and the volume lacks
is `h3-8step`, and that a cell belongs to exactly one tier.

#### 9.4 Differences from the script, and why

| | script | fv-lab | why |
|---|---|---|---|
| summaries copied into `live.log` after arms (b6 stage lines, techniques / b6 compare fields, h3arms reuse counts and gate verdicts, trace `/gpu_trace`, ltxfps PASS/FAIL, ltxvae decode lines) | yes | no | log only: each is a `grep` of a file both runners write and both fetches take (`stderr.log`, `compare/*.json`, `gate/*.json`). Nothing reads them from `live.log`. D1c's `report` reads the files |
| `FV_TECH_CELLS`, `FV_B6_CELLS` (narrow by pair) | yes | `--cells` / `FV_CELLS` by arm (`<pair>-base,<pair>-pr`), or `fv-lab run <cell> --family <tier>` | arms are the unit of a run directory |
| `techniques` H3 caches | shared by the three arms until this PR | per arm, both sides | see below |
| `FV_GEN_TIMEOUT_S` default 900 (`b6`) | when run by hand | the tier's 3600 | `runpod-http.sh` passes 3600 to every family, so 900 never applied on a pod |
| `FV_PROMPTS`, `FV_LPIPS` defaults meant for `h3arms` | honoured | `FV_PROMPTS` ignored (§7), `--lpips` | a prompt set is a cell's workload; a `h3arms-eval` tier with `prompt_set` is one cell file each when wanted |
| `FV_VAE_*` (`ltxvae`), `FV_LTXFPS*`, `FV_PRECISION_ARM` | change the arms | not read | the cells are the defaults; another variant is another tier (`FV_PRECISION_ARM=ltx-nvfp4` and `FV_VAE_GEN=1` are, since §11) |

**`techniques`: the H3 caches were shared, and the identity missed two code paths (review of #176).**
- **What was wrong:** the three H3 arms of a `tech_h3` cell shared `--text-cache $SCRATCH/h3-text-cache` (keyed by
  prompt, tokenizer and encoder weights, with no code version: fastvideo-cudarc `h3/text_cache.rs`) and
  `--adaln-cache $RUNS/<cell>-adaln.cache` (it skips the AdaLN projections when the file exists). `-base` ran first
  and wrote both, and the candidate build's `-env` / `-prof` read them.
- **Consequence:** `techniques` gates before this fix did not cover the candidate build's H3 text encoder or its
  AdaLN path. A regression in either would have passed `--off-identity`.
- **Fix, in the script and the cells in lockstep:** every H3 arm has its own `h3-text-cache-<arm>` and
  `<arm>-adaln.cache`. Its warm-up encodes and projects with its own binary, and its timed pass (the compared one)
  reads them back. The cache-on path is still exercised, now within each arm. The family's intent (one warm process
  per arm, timings comparable with earlier `techniques` runs) is kept.
- **Not chosen:** `--no-text-cache` and no AdaLN cache, as `b6` runs. That would change what `-prof`'s timings
  measure.
- **Not affected:** the LTX cell already encodes per arm.
- This is the one real-run change to `runpod-matrix.sh` in D1b. The goldens are regenerated from the changed script.
  `docs/techniques.md` notes the gap.

#### 9.5 The GPU gate per family (the coordinator's)

The procedure is §8's. Use one commit for both sides: this PR's head or its merge, once
`fastvideo-rs-runtime:sha-<7>` is published (the image carries the new `lab/` and fv-lab). Use the cheapest cells
that exercise the new parts. Both runners honour `FV_CELLS`. An unselected arm is logged `skip`. A compare whose
frames are missing writes the same `skipped` report on both sides.

`fv-lab-compare-runs.sh` now also checks:
- each compare report (`compare/compare-clips-*.json`) agrees on `status` and the off-identity status;
- each gate report exists in both runs (a gate's verdict weighs timings, which differ between pods, so only its
  presence is compared);
- any unreadable summary, frames record or report is a difference;
- it fails unless at least one arm ran (exit 0) with byte-identical frames in both runs, because two all-skipped or
  all-crashed runs are equal but prove nothing. It prints how many arms ran, were skipped and failed alike. The baseline families need `BASE`, a commit whose runtime image is published (e.g. `e93e357`, the B4a gate's
baseline).

```sh
SHA=<7>; BASE=e93e357
# fv-lab side (FAMILY and CELLS per row below; --baseline-sha only for techniques / b6)
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family $FAMILY --cells $CELLS [--baseline-sha $BASE] --yes --wait
# script side, the same commit and cells
RUNPOD_API_KEY=… FV_FAMILY=$FAMILY FV_CELLS="$CELLS" [FV_EXTRA_ENV="FV_BASELINE_SHA=$BASE"] \
  bash scripts/gpu/runpod-http.sh run $SHA
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/$FAMILY/$SHA-<MMDDHHMMSS>-<hex> artifacts/runpod/$FAMILY/$SHA-<MMDDHHMM>
```

| family | `CELLS` (cheapest that exercises it) | what it proves | est. per side | est. total |
|---|---|---|---|---|
| `techniques` | `h3-480p-fullopt-base h3-480p-fullopt-env h3-480p-fullopt-prof` | baseline fetch, bare `env`, the profile arm, per-arm text and AdaLN caches (each arm encodes with its own binary), 2 `--off-identity` reports | ~22 min | ~$1.85 |
| `b6` | `b6-solh3-tree-base b6-solh3-tree-pr` | a baseline pair, 1 `--off-identity` report | ~20 min | ~$1.70 |
| `rtx5090` | `h3-480p-dense h3-480p-sol h3-480p-fullopt h3-480p-fullopt-taeh3` | TAE arm via `{tae}`, 4 compares | ~25 min | ~$2.10 |
| `fastvideo` | `fasth3-4step-vsa-480p fasth3-4step-vsa-480p-taeh3 fasth3-8step-480p fasth3-8step-480p-taeh3` | a standalone `[[gate]]`, TAE, 2 skipped arms | ~15 min | ~$1.25 |
| `h3arms` | all (11 arms skip on `h3-8step`) | paired gates, skipped arms, 4-step VSA + TAEH3 | ~15 min | ~$1.25 |
| `trace` | `fasth3-4step-vsa-768p fasth3-8step-768p` | env arm, one skip | ~12 min | ~$1.00 |
| `ltxfps` | `ltx25-1080p-25fps fasth3-8step-sink` | 1080p 121-frame sink check, one skip | ~15 min | ~$1.25 |
| `ltxvae` | `ltxvae-4k5s-fast ltxvae-4k5s-streaming` | vae-bench, a compare | ~15 min | ~$1.25 |
| `precision` | `ltx25-4k5s-sol-f32act ltx25-4k5s-sol-bf16act` | a paired compare + gate at 4k5s | ~22 min | ~$1.85 |

The estimates are for the PRO 6000 at about $2.49/hr. They come from the D1a gate (two arms, ~15 min per side), b6
(~7 min per arm) and B4a (one 480p pair per pod, ~12 min). Pod boot and the preflight add ~5 min per side. All nine
come to about $13. Run both sides of a family in parallel. Pass, for each family: `fv-lab-compare-runs.sh` exits 0 (at least one arm ran with byte-identical frames),
both `matrix_exit=0`. That family's `case` branch is then deleted in the next D1 PR. Once a family's branch is gone,
`runpod-http.sh run` for it calls `fv-lab pod up --wait` (§8 D1b). All nine passed (`c21d801` for `techniques`, `b6`,
`precision`, `rtx5090`; `b0d0bb8` for `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae`; evidence in
`artifacts/runpod/<family>/` and `artifacts/runpod/d1b-compare/`), and their branches are deleted (§10).

### 10. The gated families leave the script: branches deleted, goldens frozen

Status: accepted 2026-10-25. The ten families whose GPU gate passed (§8 for `rtx6000`, §9.5 for the nine; the §9.1
rows give each one's evidence) are fv-lab only.

1. **The branches.** Their `case` branches are deleted from `runpod-matrix.sh` (696 lines of branches; 739 lines out, 47 in, comments included), with the one helper only
   they used (`tae_gated_cell` and its `TAEH3` / `TAELTX` paths; `FV_DRY_RUN_GATES` keeps its four columns, the
   TAE one now always `-`). The shared frame (`run_cell`, `gated_cell`, `compare_cells`, `gate_cells`,
   `frames_digest`, the TAE and LPIPS directories) stays for the remaining families, whose dry runs are unchanged
   (`FV_DRY_RUN=1` output and `FV_DRY_RUN_GATES` records compared before and after for every remaining family but
   `oracle`, whose dry run waits on a reference URL; `identity` and `determinism` differ only in the line numbers of
   their bash error messages; `headline`, item 5). The list is `scripts/gpu/fv-lab-retired.sh`
   (`FV_LAB_RETIRED`), the one place a family is marked retired; `runpod-matrix.sh <retired>` exits 2 and names the
   fv-lab commands.
2. **`runpod-http.sh`.** `run` (`FV_FAMILY`, default `rtx6000`) for a retired family runs `fv-lab pod up --sha
   <sha> --family <f> --root <repo> [--tag …] [--cells $FV_CELLS] [--baseline-sha $FV_BASELINE_SHA] [--extra-env
   $FV_EXTRA_ENV] [--lpips] [--wall-cap-s $FV_POD_CAP_S] --yes --wait`; `plan` runs the same without `--yes --wait`
   (fv-lab's plan). `pod up --wall-cap-s` (new) lowers the tier's cap, never raises it. Every other setting of the
   script path that fv-lab would not honour is refused rather than dropped, unless fv-lab does the same anyway (the
   tier's value read from `lab/tiers.toml`, such as `cell_timeout_s` for `FV_GEN_TIMEOUT_S`, or fv-lab's fixed one);
   so are `FV_EXTRA_ENV` entries no tier reads (§9.4's knobs, the prompts, what the start command sets after them).
   docs/dev/fv-lab.md lists them. `all` is refused for a retired family, and `run` / `all` for `gold` (never a
   script family; its `plan` stays the payload golden). Lists are split without globbing. The binary is `FV_LAB_BIN`, else `fv-lab` on `PATH`, else `<repo>/fv-lab`, else an error
   that says how to build it. `FV_DRY_RUN=1` prints the route and the fv-lab command line and creates nothing (and
   needs no key); for a script family it says what it would run. `kernels`, `weights`, `upstream`, `attach`,
   `status`, `down` are unchanged. `runpod.sh rtx6000` (`cmd_rtx6000`, the SSH driver of the `rtx6000` branch) is
   deleted and points at fv-lab; the Taskfile's `rtx6000` task calls `runpod-http.sh run`.
3. **Frozen since the GPU gate.** The scripts cannot write these tiers' fixtures any more. Their
   `<tier>-dry-run.txt`, `<tier>-gates.tsv` and `runpod-http-<tier>.json` stay as the script last wrote them and are
   fv-lab's own goldens: `fv-lab-goldens.sh` no longer regenerates them, and checks that they exist, that the tier is
   in `lab/tiers.toml` with no branch left, and that every other script tier still has its branch;
   `tests/dry_run.rs` checks the same. `tests/{dry_run,payload_golden,fv_lab_bin}.rs` still compare fv-lab's dry run,
   gates and pod body with them, so a change to a frozen tier's commands or pod is made on purpose: the cell (or
   `pod-start.sh.in`) and the fixture change in the same PR, with the reason in the PR. A family moved later joins
   `fv-lab-retired.sh` in the PR that deletes its branch, after its own GPU gate.
4. **Dropped with the branches**, not moved (§9.4 already left them out of the tiers): `precision`'s
   `FV_PRECISION_ARM=ltx-nvfp4` mode, `ltxvae`'s `FV_VAE_GEN=1` arms, the `FV_VAE_*`, `FV_LTXFPS*`, `FV_TECH_CELLS`,
   `FV_B6_CELLS` knobs, and the summaries copied into `live.log`. Either variant returns as its own tier if wanted
   (the commands are in git history). Both did, on the owner's decision of 2026-10-25: `precision-nvfp4` and
   `ltxvae-gen` (§11).
5. **`headline`** (stays in the script) borrowed cells from `fastvideo`, `rtx6000`, `rtx5090` and `precision` by
   running the script recursively with `FV_RUNS_DIR`. It now runs `fv-lab run <tier> --cells …` into its run
   directory, so `fv-lab run` gained `--runs-dir` / `FV_RUNS_DIR` (the script's switch; default
   `<scratch>/runs/<tier>/<tag>`), with this run's `FV_WORK`, `FV_SCRATCH` and per-cell cap (the script's 300 s
   default when unset). Its dry run is fv-lab's when the binary is there, else the fv-lab command line. Outside a
   dry run a missing fv-lab, or `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED` (fv-lab would only warn), is FATAL (exit
   2); a failed tier is logged (`FAIL fv-lab run <tier> exit=<n>`) and the next runs, as the recursive
   `runpod-matrix.sh <family> || true` did (it never set `headline`'s own exit).

`scripts/gpu/tests/fv-lab-route.test.sh` (CI: `control.yml`) checks items 2 and 5 without a network, a key or a GPU:
the mapping, every refusal, `gold`, a fake `FV_LAB_BIN` for `runpod-http.sh` and for `headline` on a fake box.

Why one list file and not "no branch means fv-lab": a typo in `FV_FAMILY` must not create an fv-lab pod, and the
`gold` tier (no branch, never a script family) keeps its `runpod-http.sh plan` golden.

### 11. Two dropped modes restored as tiers: `precision-nvfp4`, `ltxvae-gen`

Status: proposed 2026-10-25 (owner decision "restore dropped test modes"); CPU part done, GPU check pending (the
coordinator's, below). §10 item 4 dropped two modes of retired families with their branches. They come back as
tiers of their own, each reproducing what the script ran in that mode at `5987e6a4` (main before #185), the last
commit with the branches.

**What the modes ran** (`git show 5987e6a4:scripts/gpu/runpod-matrix.sh`, the `precision)` and `ltxvae)` branches):

| mode | arms, in order | compares / gates | knobs |
|---|---|---|---|
| `FV_PRECISION_ARM=ltx-nvfp4 runpod-matrix.sh precision` | per workload of `FV_NVFP4_WORKLOADS` (default `4k5s 1080p20s`): `ltx25-<wl>-sol-bf16` (`--techniques ltx2/ltx25_distill_sol`), `ltx25-<wl>-sol-nvfp4` (`ltx2/ltx25_distill_sol_nvfp4`); Sol stage 2, warm, `--no-text-cache`. **Only** these: the mode exited before the default arms | per workload a compare, then its lossy gate; then `rm -rf` of both arms' frames | `FV_NVFP4_WORKLOADS` |
| `FV_VAE_GEN=1 runpod-matrix.sh ltxvae` | the default 12 arms, and per workload after its untraced arms: `ltx25-<wl>-gen` (cold `ltx2 gen`, dense stage 2, `FASTVIDEO_LTX2_SAVE_LATENTS=$RUNS/latents-<wl>`, `FASTVIDEO_GPU_TRACE_DECODE=$FV_VAE_TRACE`), `ltx25-<wl>-gen-streaming` (`vae-bench --latents` those, streaming decoder, warm) | the default two, and per workload `compare gen-streaming gen` | `FV_VAE_TRACE` (default 1), `FV_VAE_UNTRACED` (default 1), `FV_VAE_WORKLOADS` (default both) |

**Design.**

1. **Tiers, cells.** `precision-nvfp4`: `precision-nvfp4-ltx25-{4k5s,1080p20s}`, one cell per workload (two arms, a
   compare with its paired lossy gate, §9.2 item 4): 4 arms. `ltxvae-gen`: the `ltxvae` tier's four cells with a
   `ltxvae-gen-<wl>` cell after each workload's two, in the script's order: 6 cells, 16 arms. The gen cell's second
   arm reads `{runs}/latents-<wl>`, which its first arm writes; arms run in cell order, so that holds. Both tiers are
   the `rtx6000` tier's pod (§9), as `runpod-http.sh` gave the script's every `FV_FAMILY`. **No schema change**: the
   cell v1 fields of D1a/D1b cover both modes (an `env` entry with `{runs}`, a `--techniques` argv, paired gates), so
   nothing collides with D1d's `[verdict]` (#188).
2. **Shared cells.** `ltxvae-gen` lists `ltxvae`'s cells instead of copying 12 arms. This is the one exception to
   "a cell belongs to exactly one tier" (`tests/lab_files.rs` pins it: `ltxvae-gen` shares exactly `ltxvae`'s cells,
   in order, and no other tier shares any). `fv-lab run ltxvae-4k5s` still resolves to `ltxvae` (the first tier
   listing it); `--family ltxvae-gen` writes it into `ltxvae-gen`'s run directory. Not chosen: a gen-only tier (its
   dry run is not the mode's, and a run of the mode would need two pods); copied cells (two sources for 12 arms).
3. **Knobs.** The workload knobs (`FV_NVFP4_WORKLOADS`, `FV_VAE_WORKLOADS`) and `FV_VAE_UNTRACED=0` become `--cells`
   / `FV_CELLS` by arm (one cell per workload; the untraced arms are their own cells). `FV_VAE_TRACE=0` has no
   equivalent and stays refused (a `trace = 0` twin is another tier when wanted). Only the defaults are restored.
4. **What the tiers do not do** (§9.4's kind, log or disk only): the `precision` mode's `rm -rf` of the four arms'
   frames after each gate (fv-lab keeps them on the pod's 120 GB container disk, about 8 GB for the four arms; the
   frames records are written before it on both sides and no fetch takes frames) and `ltxvae`'s `vae-bench` /
   `video decode` lines copied into `live.log` (both runners write the arms' `stderr.log`).
5. **Frozen from the start.** No script branch can write these tiers' fixtures, so they are what the scripts at
   `5987e6a4` wrote in each mode, under the inputs of §9.3, and are listed in `scripts/gpu/fv-lab-retired.sh`
   (`FV_LAB_RETIRED`): `fv-lab-goldens.sh --check` checks they exist and that no branch has their names,
   `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch` the same and the table below. Retired is
   the category that fits: "fv-lab only, frozen goldens, no branch", and `runpod-http.sh run` / `plan` then route
   `FV_FAMILY=<tier>` to `fv-lab pod up` like the ten. `FV_LAB_RESTORED` (same file) maps each tier to its mode:
   `precision-nvfp4:precision:FV_PRECISION_ARM=ltx-nvfp4`, `ltxvae-gen:ltxvae:FV_VAE_GEN=1`.
   `scripts/gpu/tests/fv-lab-restored-goldens.sh [--check]` reproduces the six fixtures from that commit (`git
   archive 5987e6a4 scripts/gpu`, `env -i`, the mode's `K=V` beside `FV_DRY_RUN=1`, and `FV_EXTRA_ENV=<K=V>` for
   `runpod-http.sh plan`). It needs the history, so it is a manual check, not CI's. The same steps for the default
   modes reproduce the frozen `precision` and `ltxvae` fixtures byte for byte (the plan body aside, which then
   carries the dummy setting), which checks the method.
6. **Equality, and the only differences.** The fixtures are the script's bytes, so they name the family where
   fv-lab names the tier, and the pod body carries the mode's setting. `tests/common/mod.rs` maps exactly that and
   nothing else before comparing:
   - dry run (`tests/dry_run.rs`, `tests/fv_lab_bin.rs::restored_modes_dry_run_as_the_script_did`): the `[<family>] `
     prefix and `/runs/<family>/` read as the tier's;
   - pod body (`tests/payload_golden.rs`): the pod name `fv-lab-<family>-<tag>`, `OUT=$SCRATCH/runs/<family>/…` and
     the runner's `runpod-matrix.sh <family>` read as the tier's, and `env <K=V> ` on the kernels and run lines read
     as `env  ` (fv-lab passes no setting: the tier is the mode). The mapped body must equal fv-lab's with §6's
     differences, as for every tier.
   Gates (`<tier>-gates.tsv`) carry no family name and compare unchanged. On a real run the same renaming holds:
   the run directory is `runs/<tier>/<tag>` (and `artifacts/runpod/<tier>/<tag>`), `live.log` lines and
   `done.json`'s `family` say the tier, the pod is `fv-lab-<tier>-…`; arm directories, commands and files are the
   script's.
7. **`runpod-http.sh`.** `FV_EXTRA_ENV` with a mode's exact `K=V` is refused on any retired family, naming the tier:
   `use FV_FAMILY=precision-nvfp4` / `use FV_FAMILY=ltxvae-gen`. Not routed: routing would put the run under another
   run directory and pod name than the `FV_FAMILY` asked for, and would need its own translation of the companion
   knobs, so a refusal that names the one-word fix is less surprising. `FV_NVFP4_WORKLOADS`, `FV_VAE_WORKLOADS` and
   `FV_VAE_UNTRACED` are refused with "narrow by arm with `FV_CELLS`"; any other value of `FV_PRECISION_ARM` /
   `FV_VAE_GEN`, and `FV_VAE_TRACE`, keep §10's refusal. `fv-lab-route.test.sh` holds it.

**Weights and GPU.** Every arm reads only `ltx25` (gate `ltx25-two-stage`), on the EU volume `bz0ofkfx45`
(`volumes.sh` `FV_EU_VOLUME_TREES`); no new tree, no download. The NVFP4 profile quantizes the bf16 FFN weights at
load. NVFP4 runs on sm_100 / sm_120 (`profiles/ltx2/ltx25_distill_sol_nvfp4.toml`; below sm_100 the FFN stays bf16
with a log line, which would make the pair a bf16-vs-bf16 check); the script's mode ran on the RTX PRO 6000
(sm_120: run `51adbb3-09271342`, FVID-2026-09-27-ltx-nvfp4-ffn-cublaslt), and the tier keeps that GPU.

**The GPU check (the coordinator's).** No script branch is left at main, so the script side runs at `5987e6a4` from
a worktree of that commit, on its published runtime image: `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-5987e6a`
**exists** (gpucheck-runtime-image run 40220450426, `workflow_dispatch` on main, 2026-10-25 18:17Z, digest
`sha256:9a835991908f766bac21b812df2b44e96c3303a366ef022c11cb351860694e94`, build id `2327ed48ab74bffe`, fv-gpucheck
from `tools-v0.1.49` "built from this commit's tools inputs"; the registry answers 200 for the tag). The fv-lab side
runs at `SHA`, this PR's head or its merge once its runtime image is published (it must carry these `lab/` cells).
The cheapest cells that exercise each mode:

```sh
SHA=<7>                                   # this PR's head or merge, runtime image published
git worktree add --detach /tmp/fv-5987e6a 5987e6a4
# precision-nvfp4 at 4k5s: the NVFP4 FFN arm, its bf16 twin, a compare and a lossy gate
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family precision-nvfp4 \
  --cells ltx25-4k5s-sol-bf16 ltx25-4k5s-sol-nvfp4 --yes --wait
(cd /tmp/fv-5987e6a && RUNPOD_API_KEY=… FV_FAMILY=precision FV_EXTRA_ENV="FV_PRECISION_ARM=ltx-nvfp4" \
  FV_CELLS="ltx25-4k5s-sol-bf16 ltx25-4k5s-sol-nvfp4" bash scripts/gpu/runpod-http.sh run 5987e6a)
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/precision-nvfp4/$SHA-<MMDDHHMMSS>-<hex> \
  /tmp/fv-5987e6a/artifacts/runpod/precision/5987e6a-<MMDDHHMM>
# ltxvae-gen at 4k5s: the cold gen that saves latents, the streaming decode of them, a compare
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family ltxvae-gen \
  --cells ltx25-4k5s-gen ltx25-4k5s-gen-streaming --yes --wait
(cd /tmp/fv-5987e6a && RUNPOD_API_KEY=… FV_FAMILY=ltxvae FV_EXTRA_ENV="FV_VAE_GEN=1" \
  FV_CELLS="ltx25-4k5s-gen ltx25-4k5s-gen-streaming" bash scripts/gpu/runpod-http.sh run 5987e6a)
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/ltxvae-gen/$SHA-<MMDDHHMMSS>-<hex> \
  /tmp/fv-5987e6a/artifacts/runpod/ltxvae/5987e6a-<MMDDHHMM>
git worktree remove /tmp/fv-5987e6a
```

The script side is `5987e6a4`'s `runpod-http.sh` with its own guards (its $5/hr cap, its backstop, the GraphQL
self-delete of #183); the fv-lab side has fv-deploy's (§7). Both on the RTX PRO 6000 Blackwell Server Edition, EU,
about $2.49/hr. Estimates from D1b (`precision` 4k5s pair ~22 min per side; one 4k5s gen plus a decode ~18 min per
side, pod boot and preflight included): `precision-nvfp4` ~$1.85, `ltxvae-gen` ~$1.50, about **$3.35** for the
four pods. Run the two sides of a tier in parallel.

**Byte identity is expected.** Between `5987e6a4` and this branch (at its base `2258aec2`) no file that makes a frame
changed: `git diff --stat 5987e6a4 HEAD` touches no fv-gpucheck crate (`fastvideo-{gpucheck,cudarc,models,loader,
ops,oxide-kernels,core,media,trace,upsampler,engine-service,protocol}`), not `Cargo.lock`, not `profiles/`, and in
`docker/gpucheck.Dockerfile` only the serve stages (the `runtime` stage is unchanged). What changed is fv-lab,
fv-deploy, fv-manifest's deploy section, autoscale, control and scripts, none of them in a generation. Pass: each
`fv-lab-compare-runs.sh` exits 0 (both arms ran with byte-identical frames records, the compare report agrees, the
gate report exists in both for `precision-nvfp4`) and both sides end `matrix_exit=0`. A lossy gate *verdict* of
fail (the 2026-09-27 run failed 4k5s on sharpness) is no run failure on either side. Before running, recheck at the
SHA you test: `git diff --stat 5987e6a4 $SHA -- crates/fastvideo-{gpucheck,cudarc,models,loader,ops,oxide-kernels,
core,media,trace,upsampler,engine-service,protocol} Cargo.lock profiles rust-toolchain.toml` (other agents work in
`crates/fastvideo-engine-service`'s tests), and the fv-lab image's `dev.fastvideo.tools` label: a stale tools
release (`FV_PREBUILT_STALE=latest`) built from another commit's gpucheck inputs is another binary. If either shows
a change in generation code, byte identity is no longer promised; then require only a successful run on both sides
(`matrix_exit=0`, both arms `exit 0`, the compare and gate reports present) and record which files differed. The
same holds if the bf16 arm is byte-identical and only the NVFP4 arm differs: cuBLASLt picks its FP4 algorithm per
driver, so compare the two pods' `box.txt` driver versions before calling it a regression.

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
   per resolution) splits into one cell per loop iteration. Settled by D1b (§9.2 item 4): one cell per loop iteration
   or helper call, plus a compare's optional paired `gate` for `compare; gate` loops. A family that also interleaves
   other steps (`b4a`'s verdict, `hd`'s upscale) stays in the script (§9.1).

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

D1b, GPU (the coordinator's): §9.5 per moved family, on one commit whose runtime image carries D1b's `lab/` and
fv-lab; expected `fv-lab-compare-runs.sh` exit 0 (arms, frames, compare reports) and `matrix_exit=0` on both sides.
Passed for all nine (§9.5).

§10, CPU: `bash scripts/gpu/tests/fv-lab-goldens.sh --check` (frozen fixtures present, branches gone),
`tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `tests/fv_lab_bin.rs::runs_dir_replaces_the_run_directory`,
`FV_DRY_RUN=1 runpod-http.sh run` for a retired and a script family, and every remaining family's
`FV_DRY_RUN=1 runpod-matrix.sh` output equal before and after the deletion (`headline` aside, which now prints fv-lab's).
