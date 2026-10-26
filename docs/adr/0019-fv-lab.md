# ADR-0019: `fv-lab`, the run plan in `lab/`, and the pod on fv-deploy's guards

- Status: Accepted for D1a (`fv-lab run` and `fv-lab pod` for the `rtx6000` family, the `lab/tiers.toml` and
  `lab/cells/*.toml` schemas, the dry-run and payload goldens; GPU gate passed) and D1b (nine more families as tiers,
  four additive cell-schema fields, a golden per tier, §9; GPU gate passed for all nine). The ten families' script
  branches are deleted and the tiers are fv-lab only, with frozen goldens (§10, 2026-10-25). D1d's first slice
  (§11: a typed verdict step in the cell schema, `b4a` as a tier, `fv-lab gate`) is accepted: its GPU gate passed on
  `e82b6b7` and `b4a` is retired from the script like the ten (§11.6). Two dropped modes are tiers again, frozen
  from the script at `5987e6a4` (§12, accepted; GPU check passed on `8b18d89`).
  D1c (§13: `fv-lab report summary` and `report table`, the `FV_FETCH_TREE` mirror; CPU only, no GPU gate) is
  accepted. D1d's second slice (§14: `det-short`, `determinism` and `oracle`'s tensor-diff targets as tiers, the
  typed `[repeats]` frame-hash check, references, ungated arms, prompt files) is proposed, its GPU checks pending; the
  oracle's clip-metric targets, `runpod-http.sh upstream`, the rest of `runpod-matrix.sh` and the port one-offs
  follow (§8, §14.6)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-lab
- Code: `crates/fastvideo-lab/src/{plan,run,pod,quote}.rs`, D1c `src/{summary,table,experiments,pyfmt}.rs` (§13), `src/bin/fv-lab.rs` (feature `cli`), `src/pod-start.sh.in`;
  `scripts/gpu/fv-lab-compare-runs.sh` (the GPU gate's comparison); fv-lab in the tools release's `gpucheck` set;
  schemas `crates/fastvideo-lab/schemas/{tiers,cell}.v1.json`; data `lab/tiers.toml`, `lab/cells/*.toml` (D1a
  `rtx6000-*`, D1b `techniques-*`, `b6-*`, `rtx5090-*`, `fastvideo-*`, `h3arms-*`, `trace-*`, `ltxfps-*`, `ltxvae-*`,
  `precision-*`; D1d `b4a-*`, §11; `det-short-*`, `determinism-*`, `oracle-*`, §14, with `src/repeats.rs`);
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
| `FV_FETCH_TREE` mirror | yes | `pod up --wait` / `pod fetch` `--fetch-tree [--fetch-skip] [--fetch-max-mb]` (D1c, §13.3) | |

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
  `timings.sh`; the `FV_FETCH_TREE` mirror. Done (§13): `report summary` and `report table`, goldens from the scripts
  on committed run directories; `sol_bench_table.py` and the four `*-summary.py` deleted; `hd_report.py` and
  `timings.sh` stay until their callers move (§13.5).
- **D1d**: `gate` / `oracle` moved into or aliased by `fv-lab` (`oracle.sh`, the `oracle` family). First slice (§11):
  the verdict step, `b4a` as a tier, `fv-lab gate` (the promotion gate aliased: `fv-gpucheck gate` driven offline);
  `b4a`'s branch deleted after its GPU gate (§11.6).
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
| `b4a` | 6 in 3 cells (3 / 0) | yes | after each arm: TeaCache reuse count and `cache-evidence.log`; after each pair: the evidence diff, `b4a-<cell>.verdict.json` | `[verdict]` (D1d, §11) | moved in D1d (§11) | **retired (script branch deleted, fv-lab only)**, §11.6: `e82b6b7`, base `e93e357`, `b4a-h3-4step-tea` base/pr, 2 arms × 124 frames byte-identical, the compare and 5 verdict files the same (cell and family verdict `pass` on both), `fv-lab gate --check` 0 differences on both runs; `artifacts/runpod/b4a/e82b6b7-*`, `b4a/compare-e82b6b7.txt`; `FV_B4A_CELLS` dropped |
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
| `determinism`, `det-short` | 9 to 19 | partly | prompt-set files written at run time, hashing, verdict files, frame deletion | `[[prompt_file]]`, `[repeats]` (§14) | moved in D1d (§14) | after each one's GPU check (§14.7); `det-short`'s opt-in `ltx-sol` group and other `FV_DET_REPEATS` stay script-only |
| `oracle` | 9 tensor-diff targets moved (32 arms); 10 clip-metric targets and `sfwan13` stay | partly | reference-dump fetches from upstream pods, compare-dumps, ffmpeg metrics | `[reference]`, ungated arms, `fresh` / `requires` / `echo` (§14) | tensor-diff targets moved in D1d (§14) | after its GPU check (§14.7); the clip-metric targets with a derived-clip step (§14.6) |
| `sfwan` | many | no | recurses into `oracle` and `wan` | no | stays | D1d / a Wan volume |
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
| summaries copied into `live.log` after arms (b6 stage lines, techniques / b6 compare fields, h3arms reuse counts and gate verdicts, trace `/gpu_trace`, ltxfps PASS/FAIL, ltxvae decode lines) | yes | no | log only: each is a `grep` of a file both runners write and both fetches take (`stderr.log`, `compare/*.json`, `gate/*.json`). Nothing reads them from `live.log`. D1c's `report` reads the files: `fv-lab report summary <run dir>` prints them (§13.1) |
| `FV_TECH_CELLS`, `FV_B6_CELLS` (narrow by pair) | yes | `--cells` / `FV_CELLS` by arm (`<pair>-base,<pair>-pr`), or `fv-lab run <cell> --family <tier>` | arms are the unit of a run directory |
| `techniques` H3 caches | shared by the three arms until this PR | per arm, both sides | see below |
| `FV_GEN_TIMEOUT_S` default 900 (`b6`) | when run by hand | the tier's 3600 | `runpod-http.sh` passes 3600 to every family, so 900 never applied on a pod |
| `FV_PROMPTS`, `FV_LPIPS` defaults meant for `h3arms` | honoured | `FV_PROMPTS` ignored (§7), `--lpips` | a prompt set is a cell's workload; a `h3arms-eval` tier with `prompt_set` is one cell file each when wanted |
| `FV_VAE_*` (`ltxvae`), `FV_LTXFPS*`, `FV_PRECISION_ARM` | change the arms | not read | the cells are the defaults; another variant is another tier (`FV_PRECISION_ARM=ltx-nvfp4` and `FV_VAE_GEN=1` are, since §12) |

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
rows give each one's evidence) are fv-lab only. `b4a` followed the same way after its own gate (§11.6), in the next
D1 PR.

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
   `ltxvae-gen` (§12).
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

### 11. D1d, first slice: a typed verdict step, `b4a` as a tier, `fv-lab gate`

Status: accepted 2026-10-25. The GPU gate (§11.6) passed on `e82b6b7`; `b4a` is in `fv-lab-retired.sh`, its
`runpod-matrix.sh` branch is deleted and its fixtures are frozen (§11.6, "Result").

`b4a` (plan B4a, ADR-0020) is the first family whose branch does more than arms, compares and gates: after each arm
it counts the TeaCache reuses and writes `cache-evidence.log`, and after each pair it diffs the two evidence files
and writes `b4a-<cell>.verdict.json`; at the end `b4a.verdict.json`. That is the "verdict step" of §9.1. D1d gives
it a type.

#### 11.1 The verdict step (cell v1, schema addition 5)

One optional table per cell, additive (`deny_unknown_fields` kept, every cell file before it parses unchanged,
`schemas/cell.v1.json` regenerated, `tests/schema.rs` names the new keys):

```toml
[verdict]
family = "b4a"                  # b4a-<cell>.verdict.json, b4a.verdict.json; must be the tier's name
arms = ["b4a-h3-4step-tea-base", "b4a-h3-4step-tea-pr"]   # each <cell>-<label>; the label names it in reasons
describe = "verdict also needs both arms exit 0 (summary.json) and the --off-identity compare passed"

[[verdict.evidence]]            # per arm: matching stderr.log lines into <arm>/<file>
file = "cache-evidence.log"     # ends in -evidence.log (what `pod fetch` takes)
lines = ["h3 sol teacache", "h3 teacache:", "h3 teacache step", "fuse step cache"]
strip = "bracket-prefixes"      # sed -E 's/^(\[[^]]*\] *)+//'
diff = "evidence.diff"          # the two arms' files equal: diff into <runs>/<cell>.evidence.diff, else evidence-differs

[[verdict.count]]               # per arm: grep -c, logged "[<arm>] <what>: <n>"; under min: <label>:<why>
what = "teacache reused forwards"
lines = ["h3 teacache step *REUSE"]
min = 1
why = "no-reuse"

[[verdict.identical]]           # the --off-identity compare passed (b4-verdict.sh b4_identical), else why
baseline = "b4a-h3-4step-tea-base"
candidate = "b4a-h3-4step-tea-pr"
```

What runs, in the script's order (`b4a_pair`, `b4_family_verdict`):

1. Before the first arm, every `<family>-*.verdict.json` and `<family>.verdict.json` of the run directory is removed
   (no earlier run's verdict may stand in).
2. A verdict cell runs only when one of its verdict arms is selected (`--cells` / `FV_CELLS`); otherwise it is logged
   `skip <cell> (no arm in FV_CELLS)` and nothing of it runs. When it runs, its identity reports are removed, then
   each verdict arm's directory is removed before the arm (selected or not, as the script does), the arm runs, and
   right after it: `summary.json` exit 0 or `<label>:exit`; no `stderr.log` gives `<label>:no-run` and ends that arm's
   checks; else each evidence file is written and each count logged and checked. Then the cell's compares (and
   gates), then each `identical` pair, then each evidence `diff`.
3. `<runs>/<family>-<cell>.verdict.json` is `{"cell":…,"verdict":"pass|fail","why":"<reasons>"}` and the log line
   `<family> <cell>: pass|fail (<reasons>)`, as the script writes them.
4. After every other step, one `<runs>/<family>.verdict.json` per family over the verdict cells that ran:
   `{"family":…,"verdict":…,"cells":"<cells>","why":…}`, failing on a cell without a verdict file or with a failing one
   (`b4_family_verdict`'s rule, ported). No tier field is needed: a run with verdict cells ends with their family's.
5. The dry run prints the script's lines: `dry-run <arm>: rm -rf <runs>/<arm>` before each verdict arm, the cell's
   lines, `dry-run <cell>: <describe>`, and the family line (which names `b4_family_verdict`, as the script's does).

Validation: the verdict's arms are arms of the cell named `<cell>-<label>`; `family` is the name of the tier that
lists the cell; evidence files end in `-evidence.log` and are unique; a `diff` needs exactly two arms and its name is
unique (two evidence files diffed into one file would hide the first's difference); patterns are
one line and not only `*`; reasons are tokens; every `identical` pair has a `[[compare]]` with `--off-identity`.

**Line patterns** are substrings with `*` for any run of characters (pieces matched in order), not regular
expressions: the script's two patterns (`a\|b\|c\|d`, `h3 teacache step .*REUSE`) are exactly that, and a regex
engine would be a new dependency of the crate. Matching is on bytes, line by line as grep reads them.

**The `diff` file** is written by `diff` itself (a process, like `verify-weights.sh` and `fv-gpucheck`): its format is
the script's by construction. Both sides run `LC_ALL=C diff -a`: evidence holding a NUL would otherwise make `diff`
write `Binary files <path> and <path> differ`, naming the run directory, which differs between two runs' tags and
between a pod and `fv-lab gate` (review of #188).

**Bytes, not a locale (review of #188, F6).** The runtime image (`docker/gpucheck.Dockerfile`, `ubuntu:22.04`) sets no
`LANG` / `LC_ALL` and installs no locales, and neither pod start command sets one, so pods run in the C locale. All
the same, the script's evidence `grep` / `sed` and its reuse count now run `LC_ALL=C grep -a` / `LC_ALL=C sed`
(real runs only; the dry run is unchanged): under a UTF-8 locale GNU grep reads a line that is not valid UTF-8 (or
holds a NUL) as binary and writes no evidence line at all, while fv-lab matches bytes. `tests/verdict.rs` runs the
script under `C.UTF-8` with CRLF lines, bytes that are not UTF-8, a NUL, an unterminated last line and an empty
`stderr.log`, and holds both runners byte-identical.

**Why typed and not a hook** (a shell snippet per cell): the equality with the script holds only if fv-lab writes the
same files, which a typed step does by construction and a test can hold (below); `fv-lab check` validates it; and
`fv-lab gate` can evaluate it again offline (§11.3). A shell hook would move the script's grep and sed into data.

**For the next families.** The frame (fresh arms, per-arm checks after each arm, per-cell checks after the compares,
reason tokens, the two verdict files and the family verdict) is general; what `determinism` and `oracle` need is new
typed lists under `[verdict]` (for example a frames-hash equality across arms, a `benchmark.json` field rule as
`b4c` checks `text_cache`, a cache-directory listing), not a new step. None is built here.

#### 11.2 `b4a` as a tier

`lab/tiers.toml` `b4a` (the `rtx6000` pod, as every tier: `runpod-http.sh` gives every family the same pod), cells
`lab/cells/b4a-h3-{fullopt-env,fullopt-prof,4step-tea}.toml`, one per `b4a_pair`, two arms each (`bin = "baseline"`
and this build), one `--off-identity` compare, the verdict above. 3 cells, 6 arms. The goldens come from the script
as in D1b (§9.3): `fixtures/b4a-dry-run.txt` (21 lines), `b4a-gates.tsv`, `runpod-http-b4a.json`, written by
`fv-lab-goldens.sh` while the branch existed and frozen since (§11.6); `tests/{dry_run,payload_golden,fv_lab_bin}.rs`
hold fv-lab to them.
(`fv-lab-goldens.sh`'s `has_branch` read every branch as missing under `pipefail`, `grep -q` cutting `sed` off with
SIGPIPE; it never showed because no tier had a live branch before `b4a`. Fixed.)

**Equality beyond the dry run.** `tests/verdict.rs` copies `runpod-matrix.sh` (and what it sources) into a fake root
(since §11.6 the frozen copy of the script's last `b4a` branch, `fixtures/b4a-script/`) and runs its real `b4a` branch
and `fv-lab run b4a` (the committed cells) on the same fake box: a fake fv-gpucheck
whose arms log step-cache lines and whose compare writes an off-identity report, a fake `nvidia-smi`,
`verify-weights.sh` and `fetch-baseline.sh`. One cell passes, one fails on its evidence only, one fails on
everything. Every verdict file, evidence file and evidence diff is byte-identical between the two runners, and so
are the verdict lines of `live.log`; with `FV_B4A_CELLS` against `--cells` the same subset runs on both sides; a stale
verdict or arm directory never stands in. `fv-lab gate` over either run directory, with the selection it infers
itself (`run::gate_selection`), reproduces the script's files and `--check` finds them equal. The review of #188
added the cases the first fake left out, each script against fv-lab byte for byte: an empty `stderr.log`, CRLF,
non-UTF-8 and NUL bytes in a UTF-8 locale, a missing compare report, arms skipped on their weight gate, a one-arm
selection (its compare a `frames missing` skip), and a run cut short after its first cell (F1, below).

#### 11.3 `fv-lab gate <run dir>`

`gate` was not an fv-lab subcommand. It is now: `fv-lab gate <run dir> [--family T] [--cells …] [--out D] [--check]
[--gpucheck B] [--policy F]` evaluates a finished run's promotion gates and verdicts again from its files, with the
runner's own code and no GPU. No arm or compare runs and nothing in the run directory is written or removed.

- **The tier** is `--family` or the run directory's parent (`artifacts/runpod/<tier>/<tag>`).
- **The arms** (review of #188, F1) are `--cells`; else the run's selection record `<tier>.arms.json`, which `fv-lab
  run` writes before its first arm (the arms it selected, in run order) and `pod fetch` takes; else, for a run without
  one (a script run, or fv-lab before the record), every arm of the tier when the run holds them all, and otherwise an
  error asking for `--cells`. The first version took the arms holding a `summary.json` as "the arms the run selected":
  a run cut short after its first cell then passed with a family verdict of that one cell, where the script fails the
  cells it never finished (`:no-verdict`). With the record, those cells fail (`:exit :no-run`); without it, fv-lab
  does not guess. Unknown arm names are refused.
- **The output** (F2) goes to `--out`, default a sibling `<run dir>.regate`, and never into the run directory or below
  it (refused before anything is created): overwritten, the run's own verdict files would be fv-lab's, and a later
  comparison would compare fv-lab with itself. What an earlier evaluation left in `--out` is removed first (F5), so an
  arm without a `stderr.log` cannot keep an old evidence file that is then diffed.
- **`--check`** compares each file the evaluation writes with the run's own (verdicts, evidence files and diffs byte
  for byte; a gate report by its `status`, as it also holds timings): one `DIFF` line each, exit 1 on any. That is the
  judge's question (does the run hold what fv-lab derives from it?), independent of whether the verdicts pass.
- **Gates** (F3): each `[[gate]]` / paired gate is `fv-gpucheck gate` (CPU only: `benchmark.json` and the compare
  reports under the gate policy) into `<out>/gate/`. A gate whose arms did not both run (skipped or not selected) is
  skipped and counted; one whose arms both ran (`summary.json` records an exit) without a `benchmark.json` fails,
  where the first version counted it as neither and could hide a crashed candidate.
- **Verdicts** are §11.1 into `<out>`.

It prints one line per gate and verdict and `<n> evaluated, <f> failed, <s> skipped`; exit 0 when everything
evaluated passed, 1 when anything failed or nothing could be evaluated. A tier with gates needs an fv-gpucheck
(`--gpucheck` / `FV_GPUCHECK`, any build).

**Moved or aliased (the review's question, §8):** aliased. The promotion rule stays `fv-gpucheck gate`, the measured
code with its own tests, which `fv-lab run` already calls as a process; moving it into `fastvideo-lab` would mean
pulling it out of a crate that links CUDA. fv-lab owns what is fv-lab's: driving it (on the pod, and now offline)
and the verdict step. The weight gates (`verify-weights.sh`) are not re-evaluated: they need the weights.

#### 11.4 The verdict files reach `artifacts/`

Neither fetch took them: `runpod-http.sh fetch_results` and `fv-lab pod fetch` take fixed run files, and per
directory `summary.json`, the logs and JSONs. `fv-lab pod fetch` now also takes the run directory's `*.verdict.json`,
`*.diff` and `<tier>.arms.json` (§11.3) and each directory's `*-evidence.log` (`pod::is_verdict_file`). `runpod-http.sh` is unchanged: its
`FV_FETCH_TREE=1` (with `FV_FETCH_SKIP=/frames/`) already mirrors them. `fv-lab-compare-runs.sh` compares those files
byte for byte whenever either run has them (another verdict, or a file in one run only, is a difference), and
`--expect-verdicts N` fails unless at least N of them compared the same (F7: two runs whose verdict files never
arrived would otherwise pass with none); its self-test in `fv-lab-goldens.sh` covers each case. Without the files, `fv-lab gate` rebuilds them from what both fetches
take (`summary.json`, `stderr.log`, the compare reports).

#### 11.5 Differences from the script, and why

| | script | fv-lab | why |
|---|---|---|---|
| selecting pairs | `FV_B4A_CELLS` (by pair) | `--cells` / `FV_CELLS` by arm; a verdict cell runs when any of its verdict arms is selected | arms are the unit of a run directory (§9.4); the GPU gate pairs `FV_B4A_CELLS=<cell>` with `--cells <cell>-base <cell>-pr` (`tests/verdict.rs`) |
| `FV_CELLS` naming some arms of `b4a` | every pair still runs its checks and fails the unselected ones | the pairs with no selected verdict arm are skipped whole | the script's knob for that is `FV_B4A_CELLS`; failing a pair nobody asked for adds nothing |
| `FV_GEN_TIMEOUT_S` default 900 | when run by hand | the tier's 3600 | `runpod-http.sh` passes 3600 (§9.4) |
| the compare fields copied into `live.log` after the family | yes | no | log only (§9.4) |
| the verdict files in `artifacts/` | only with `FV_FETCH_TREE=1` | `pod fetch` takes them | §11.4 |
| the family verdict's `cells` (and so its `why`) | in `FV_B4A_CELLS`'s order; a name matching no cell fails (`<name>:no-verdict`) | in the tier's order; `pod up` refuses an arm name the tier lacks, `gate` too | chosen (review of #188, F4): document it, not reproduce it, since fv-lab has no ordered cell list to follow; `fv-lab-compare-runs.sh` compares a family verdict's `cells` and `why` as sets |
| `FV_B4A_CELLS` handed to fv-lab | (the script's knob) | refused by `pod up --extra-env` and by `run` (F8); since §11.6 also by `runpod-http.sh`'s `FV_EXTRA_ENV` (`FV_LAB_NOT_READ`) | fv-lab would ignore it and run all three cells at three times the cost |
| the arm selection | not recorded | `<tier>.arms.json` in the run directory | `fv-lab gate` reads it (§11.3) |
| `fv-lab gate` on a fetched run | n/a | writes beside it (`<run>.regate`), never into it | §11.3 |

#### 11.6 The GPU gate for `b4a` (the coordinator's)

The procedure is §8's and §9.5's: one commit for both sides, this PR's head or its merge, once
`fastvideo-rs-runtime:sha-<7>` is published (its image carries the new `lab/` and fv-lab). The cheapest cell that
exercises every part is `b4a-h3-4step-tea` (4 steps; the baseline fetch, bare `env`, fresh arm directories, 5
evidence lines and 2 reuses per arm, the identity compare, the evidence diff, the cell and family verdicts). `BASE`
is a commit whose runtime image is published, the B4a gate's baseline.

```sh
SHA=<7>; BASE=e93e357
# fv-lab side (fetch takes the verdict files and b4a.arms.json)
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family b4a --cells b4a-h3-4step-tea-base b4a-h3-4step-tea-pr \
  --baseline-sha $BASE --yes --wait
# script side, the same commit and cell; the tree fetch brings the verdict files
RUNPOD_API_KEY=… FV_FAMILY=b4a FV_EXTRA_ENV="FV_BASELINE_SHA=$BASE FV_B4A_CELLS=b4a-h3-4step-tea" \
  FV_FETCH_TREE=1 FV_FETCH_SKIP=/frames/ bash scripts/gpu/runpod-http.sh run $SHA
L=artifacts/runpod/b4a/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/b4a/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh --expect-verdicts 5 $L $S
# each pod's verdict files are what fv-lab derives from that run's own files (written elsewhere, compared)
./fv-lab gate $L --check --out /tmp/regate-fvlab      # the arms from $L/b4a.arms.json
./fv-lab gate $S --check --out /tmp/regate-script --cells b4a-h3-4step-tea-base,b4a-h3-4step-tea-pr   # the script writes no record
```

Pass: `fv-lab-compare-runs.sh --expect-verdicts 5` exits 0 with 2 arms the same (both ran, frames byte-identical),
the compare report the same and 5 verdict files the same (`b4a.verdict.json`, `b4a-b4a-h3-4step-tea.verdict.json`,
`b4a-h3-4step-tea.evidence.diff`, both `cache-evidence.log`); both `matrix_exit=0`; both `fv-lab gate --check` exit 0
(`0 difference(s)`). The verdict itself may pass or fail; what is gated is that both runners reach the same one and
that it is what fv-lab derives. Estimate: ~8 min of arms (the B4a
runs: 280 s and 175 s per arm) plus ~5 min boot and preflight per side, ~13 min, about $0.55 per side, $1.10 in all
at $2.49/hr; both sides in parallel. The other two cells (49 forwards, ~6 to 8 min per arm) add about $1 per side
each if wanted (`--expect-verdicts` is then 1 + 4 per cell, 13 for all three: the family file, and each cell's
verdict, diff and two evidence files).

Then, in the next D1 PR: `b4a` joins `fv-lab-retired.sh`, its branch is deleted, its three fixtures freeze (§10), and
`runpod-http.sh`'s route refuses `FV_B4A_CELLS` in `FV_EXTRA_ENV` as it refuses the other dropped knobs.

**Result (2026-10-25): pass.** `SHA=e82b6b7`, `BASE=e93e357`, the cell `b4a-h3-4step-tea`, both on the RTX PRO 6000:

- fv-lab side: `fv-lab pod up --sha e82b6b7 --family b4a --cells b4a-h3-4step-tea-base b4a-h3-4step-tea-pr
  --baseline-sha e93e357 --yes --wait` (pod `j25pn9a2xc1ll8`) → `artifacts/runpod/b4a/e82b6b7-1025211328-a2f3`;
- script side: `FV_FAMILY=b4a FV_EXTRA_ENV="FV_BASELINE_SHA=e93e357 FV_B4A_CELLS=b4a-h3-4step-tea" FV_FETCH_TREE=1
  FV_FETCH_SKIP=/frames/ runpod-http.sh run e82b6b7` (pod `ik1hd9tz9cay82`) → `artifacts/runpod/b4a/e82b6b7-10252113`;
- `fv-lab-compare-runs.sh --expect-verdicts 5` (`artifacts/runpod/b4a/compare-e82b6b7.txt`): exit 0, "2 arm(s) the
  same (2 ran with byte-identical frames, 0 skipped in both, 0 failed in both), 1 report(s) the same, 5 verdict
  file(s) the same, 0 difference(s)"; 124 frames per arm; the `--off-identity` compare `pass` / `ok`;
- both `matrix_exit=0`; per arm 5 evidence lines and 2 reused forwards on both sides; the cell verdict and the family
  verdict `pass` on both;
- `fv-lab gate --check`: 0 differences on each run (fv-lab's arms from its `b4a.arms.json`, the script's with
  `--cells b4a-h3-4step-tea-base,b4a-h3-4step-tea-pr`).

The image: `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-e82b6b7` at
`sha256:c5a683f55f24a9400f7010114109db589da459ad53419e0d35e1f1a32f5ee344` (labels `dev.fastvideo.tools=compiled`,
revision `e82b6b78`). The first attempt ran on the tag's earlier digest (`sha256:bd1761d2…`), built from a prebuilt
tools release whose fv-lab (0.1.51) predates `[verdict]`: it refused the cells ("unknown field `verdict`").
`gpucheck-runtime-image` (run 40233480444) had caught that in its fv-lab check, but only after it had pushed
`sha-e82b6b7`, so the tag named an image that could not run the commit's `lab/`. **Follow-up (open):** the image job
runs its checks (fv-lab `--version`, `check`, the dry run) before it pushes `sha-<7>`, or pushes to a staging tag and
retags only after they pass.

Done in that PR: `b4a` is in `FV_LAB_RETIRED`; the `b4a)` branch (its `b4a_evidence` / `b4a_pair` helpers lived inside
it) is deleted from `runpod-matrix.sh`, and the remaining families' dry runs are unchanged (`FV_DRY_RUN=1` output and
`FV_DRY_RUN_GATES` records before and after for all 32: 30 byte-identical, `determinism` differing only in a bash
error message's line number and `writer` only in its `dd` timings, which differ run to run). `b4-verdict.sh` stays
whole: `b4_exit_ok`, `b4_identical` and `b4_family_verdict` are also `b4b` / `b4c` / `b4c2`'s, and
`tests/b4-verdict.test.sh` keeps testing them. `fv-lab-goldens.sh` no longer writes `b4a`'s three fixtures (§10 item
3). `tests/verdict.rs` ran the script's live `b4a` branch, which now refuses `b4a` (exit 2); it runs a frozen copy
instead, `crates/fastvideo-lab/fixtures/b4a-script/`: `runpod-matrix.sh` as of main `8b18d89f`, cut to its shared
frame, the `b4a` branch and the fallback branch with every kept line unchanged (`git show
8b18d89f:scripts/gpu/runpod-matrix.sh | sed -n '1,446p;1618,1743p;3197,3223p'`), and that commit's `b4-verdict.sh`,
`cuda-13.pins` and `fv-lab-retired.sh` whole, so changes under `scripts/gpu/` no longer move the reference. What
the frame reads from fixed host paths is kept out where the test can do it without editing the cut: the CUDA
library links under `<scratch>/fv-libs` are faked (so the image's `remote.sh` never runs and no host library is
linked), and the `matrix start image=` line, read from the hard-coded
`/opt/fastvideo-rs/target/release/fv-gpucheck.build-id`, reads as `unknown` when a host has that file. One host
dependency is left, as it was before the freeze: the frame's `PATH` puts `/opt/fastvideo-rs/target/release` and
`/usr/local/bin` before the fake `bin/`, so a host `nvidia-smi` there would shadow the fake one. A new test runs the
frozen copy's dry run and finds it equal to `fixtures/b4a-dry-run.txt`, line for line. `runpod-http.sh run` / `plan` route
`FV_FAMILY=b4a` to `fv-lab pod up` (§10 item 2), and refuse `FV_B4A_CELLS` in `FV_EXTRA_ENV` with "b4a selects by arm,
as every tier: `FV_CELLS="<cell>-base <cell>-pr"`" (for another tier, a message without `b4a`). The review of #195
added: the pair selectors `FV_B4A_CELLS`, `FV_TECH_CELLS` and `FV_B6_CELLS` set in the host environment (not
`FV_EXTRA_ENV`) are refused too for every retired tier, where they were ignored and the run covered every cell
(`fv-lab-route.test.sh`). The gate's script-side command is now
refused (it sets `FV_FETCH_SKIP`, `FV_FETCH_TREE=1` and `FV_B4A_CELLS`, none of which fv-lab honours; the first
refusal answers); its fv-lab equivalent is
`FV_FAMILY=b4a FV_CELLS="b4a-h3-4step-tea-base b4a-h3-4step-tea-pr" FV_BASELINE_SHA=e93e357 runpod-http.sh run <7>`.

#### 11.7 Open decisions

1. **Selection by arm** (§11.5): `--cells` takes arms (and `FV_B4A_CELLS` is refused); a cell-name form (`--cells b4a-h3-4step-tea` meaning its
   verdict arms) would read closer to `FV_B4A_CELLS`. Proposed: arms only, as every tier.
2. **`runpod-http.sh`'s fetch** stays without the verdict files (the script side of the gate uses `FV_FETCH_TREE=1`).
   Settled by §11.6: `runpod-http.sh run` routes `b4a` to `fv-lab pod up`, whose `pod fetch` takes them; the script
   fetch was not changed.
3. **Settled (review of #188, F2):** `fv-lab gate` never writes into the run directory; `--out` defaults to
   `<run>.regate`, and `--check` compares.
4. **`oracle`** (the `oracle` family, `oracle.sh`, `fv-gpucheck compare-dumps`) is the second D1d slice; the
   `determinism` verdicts become new typed checks under `[verdict]` (§11.1) with D1c / D1d. Done in §14: a `[repeats]`
   table beside `[verdict]` rather than under it (§14.3 says why), the oracle's tensor-diff targets as a tier.

### 12. Two dropped modes restored as tiers: `precision-nvfp4`, `ltxvae-gen`

Status: accepted 2026-10-25 (owner decision "restore dropped test modes"); CPU part done, GPU check passed (the
coordinator's, "Result" below). §10 item 4 dropped two modes of retired families with their branches. They come back as
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

**Byte identity is expected.** The crates in fv-gpucheck's build are `cargo tree --locked -p fastvideo-gpucheck
--features cuda -e normal,build`'s workspace members: `fastvideo-{gpucheck,cudarc,engine-service,loader,manifest,
media,models,ops,protocol,trace}`. Add `fastvideo-oxide-kernels` for the oxide cubins (the tools release's `oxide`
set), `profiles/`, `Cargo.toml`, `Cargo.lock` and `rust-toolchain.toml`. Between `5987e6a4` and this branch (main at
`6c13e195` merged in), the only change among these is in `fastvideo-manifest`: the deploy section
(`deploy.rs`, `schemas/deploy.v1.json`) and a `SMOKE_CONFIG` re-export in `lib.rs`. fv-gpucheck links it through
`fastvideo-engine-service` (the `cuda` feature), but no generation reads the deploy section. `family.rs` is
unchanged. The other tools inputs (`third_party/cutile-rs`, `scripts/gpu/cuda-13.pins`, `docker/build-base.Dockerfile`,
`scripts/dev/release-artifacts-pod.sh`) are unchanged, and so is the `runtime` stage of `docker/gpucheck.Dockerfile`
(only its serve stages changed). Fv-lab, fv-deploy, autoscale, control and scripts changed too, and none of them is in
a generation.

The binary is not the same build. The tools release's inputs are all of `crates/` (and the paths above;
`tools-release.sh` `INPUTS`), and `crates/` changed. So the image of the PR head carries another tools build than
`tools-v0.1.49`: one released from the head's own inputs, else the job compiles, else (on main, with
`FV_PREBUILT_STALE=latest`) the newest release with a warning. What is promised is equal frames from equal generation
source, not an equal binary.

Pass: each `fv-lab-compare-runs.sh` exits 0, meaning:
- both arms ran with byte-identical frames records;
- the compare report agrees;
- for `precision-nvfp4`, the gate report exists in both runs;
- both sides end `matrix_exit=0`.

A lossy gate *verdict* of fail (the 2026-09-27 run failed 4k5s on sharpness) is no run failure on either side.

Before running, recheck at the `SHA` you test (other agents work in `crates/fastvideo-engine-service`'s tests):

```sh
git diff --stat 5987e6a4 $SHA -- crates/fastvideo-{gpucheck,cudarc,engine-service,loader,manifest,media,models,ops,protocol,trace,oxide-kernels} \
  profiles Cargo.toml Cargo.lock rust-toolchain.toml third_party/cutile-rs scripts/gpu/cuda-13.pins
cargo tree --locked -p fastvideo-gpucheck --features cuda -e normal,build --prefix none | grep '(/'   # the list above, unchanged?
```

Then read the fv-lab image's `dev.fastvideo.tools` label and the image job's notice. It must name a release built
from `SHA`'s own inputs ("built from this commit's tools inputs") or a compile. A stale release (`FV_PREBUILT_STALE=latest`
with "no release matches") is another commit's gpucheck.

- **Byte identity no longer promised.** If the diff shows a change outside `fastvideo-manifest`'s deploy section, or
  the binary is stale, require only a successful run on both sides: `matrix_exit=0`, both arms `exit 0`, the compare
  and gate reports present. Record which files differed.
- **Only the NVFP4 arm differs** (the bf16 arm is byte-identical): cuBLASLt picks its FP4 algorithm per driver, so
  compare the two pods' `box.txt` driver versions before calling it a regression.

**Result (2026-10-25): pass, both tiers.** fv-lab at main `8b18d89` (#191's merge; image `sha-8b18d89`, tools
compiled, build id `1fb053d232522a86`), the script at `5987e6a4` (image `sha-5987e6a`, build id `2327ed48ab74bffe`),
the 4k5s cells of each mode; four RTX PRO 6000 pods in EU-RO-1 on the volume `bz0ofkfx45`, driver 595.91.07, all
deleted:

- `ltxvae-gen`:
  - fv-lab: `fv-lab pod up --sha 8b18d89 --family ltxvae-gen --cells ltx25-4k5s-gen ltx25-4k5s-gen-streaming --yes
    --wait` (pod `wgzwvt5mjcn1nv`) → `artifacts/runpod/ltxvae-gen/8b18d89-1025215640-5305`.
  - The script: `FV_FAMILY=ltxvae FV_EXTRA_ENV="FV_VAE_GEN=1" FV_CELLS="ltx25-4k5s-gen ltx25-4k5s-gen-streaming"
    runpod-http.sh run 5987e6a` → `artifacts/runpod/ltxvae/5987e6a-10252157`.
  - `fv-lab-compare-runs.sh` (`artifacts/runpod/ltxvae-gen/compare-8b18d89.txt`): exit 0, "2 arm(s) the same (2 ran
    with byte-identical frames, 0 skipped in both, 0 failed in both), 4 report(s) the same, … 0 difference(s)": 121
    frames per arm; the gen / streaming compare `pass` on both, the three unselected compares `skipped` on both.
- `precision-nvfp4`:
  - fv-lab: `fv-lab pod up --sha 8b18d89 --family precision-nvfp4 --cells ltx25-4k5s-sol-bf16 ltx25-4k5s-sol-nvfp4
    --yes --wait` (pod `ymrchzi0z3eixy`) → `artifacts/runpod/precision-nvfp4/8b18d89-1025215713-20dd`.
  - The script: `FV_FAMILY=precision FV_EXTRA_ENV="FV_PRECISION_ARM=ltx-nvfp4" FV_CELLS="ltx25-4k5s-sol-bf16
    ltx25-4k5s-sol-nvfp4" runpod-http.sh run 5987e6a` → `artifacts/runpod/precision/5987e6a-10252158`.
  - `fv-lab-compare-runs.sh` (`artifacts/runpod/precision-nvfp4/compare-8b18d89.txt`): exit 0, 2 arms the same, 242
    frames per arm byte-identical (the NVFP4 arm too), 3 reports the same (the 4k5s compare, the unselected
    1080p20s compare skipped on both, and the gate report present on both).
  - The lossy gate's verdict is `fail` on both sides, as the 2026-09-27 run's 4k5s gate was: no run failure.
- Both script pods (`64opofps1163h8`, `kxhdp1v19w151y`) and both fv-lab pods end `matrix_exit=0`.

Byte identity held, so the generation source between `5987e6a4` and `8b18d89` (the deploy section of
`fastvideo-manifest` aside) and the other tools build left these arms' frames unchanged.

**Refused, not ignored (review of #191).**
- `fv-lab` refuses the settings no tier reads (`crates/fastvideo-lab/src/knobs.rs`):
  - `fv-lab run` refuses the retired families' mode knobs (§9.4) in its environment.
  - `pod up --extra-env` refuses those knobs and what the start command sets after `--extra-env`.
  - This is the same set as `runpod-http.sh`'s `FV_LAB_NOT_READ` (`tests/knobs.rs` holds them equal, and
    `RESTORED_MODES` equal to `FV_LAB_RESTORED`).
  - D1d's `FV_B4A_CELLS` refusal (§11.5) is one more entry of the same module, with its message unchanged. It was
    refused by fv-lab only (`SCRIPT_KNOBS`) while `b4a` had its script branch; since `b4a` retired (§11.6) it is a
    `MODE_KNOBS` entry and in `FV_LAB_NOT_READ`, and `SCRIPT_KNOBS` is gone.
- A restored mode's setting is answered first and names its tier. Each refusal lists every refused key at once.
- `runpod-http.sh` does the same: the mode check comes before every other refusal of the call, and one message lists
  all the refused `FV_EXTRA_ENV` keys.
- `runpod-matrix.sh` names the tier for `FV_PRECISION_ARM=ltx-nvfp4 … precision` / `FV_VAE_GEN=1 … ltxvae`, and says
  that `precision-nvfp4` / `ltxvae-gen` are fv-lab tiers, never script families (`fv-lab-route.test.sh`).

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
   other steps (`hd`'s upscale) stays in the script (§9.1); `b4a`'s verdict became a typed step in D1d (§11).

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

§11 (D1d), CPU: `tests/verdict.rs` (the frozen copy of the script's last `b4a` branch and `fv-lab run b4a` on one fake box: verdicts,
evidence and diffs byte-identical; `fv-lab gate` reproduces them), `tests/fv_lab_bin.rs` (`b4a` dry run,
`gate_evaluates_a_finished_runs_verdicts_offline`), `tests/dry_run.rs` and `tests/payload_golden.rs` with the `b4a`
fixtures, `fv-lab-goldens.sh --check` (with compare-runs' verdict-file cases); GPU: §11.6.

§14 (D1d, second slice), CPU: `tests/repeats_oracle.rs` (the live `det-short`, `determinism` and `oracle` branches and
`fv-lab run` on one fake box: every output file byte-identical, `fv-lab gate --check` clean, the typed verdicts),
`tests/{dry_run,payload_golden}.rs` with the three tiers' fixtures, `plan::tests::references_ungated_arms_prompt_files_and_repeats`,
`tests/fv_lab_bin.rs::the_oracle_pod_needs_its_references_and_the_d1d_knobs_are_refused`, `fv-lab-goldens.sh --check`
(with compare-runs' hash and dump-line cases); GPU: §14.7.

§13 (D1c), CPU: `tests/report.rs` (every fixture of `fixtures/report/`, byte for byte), `tests/fetch_tree.rs` (the
script's `fetch_tree` on a fake pod), `tests/fv_lab_bin.rs::report_summary_and_table_print_the_scripts_output`,
`::the_tree_fetch_is_checked_before_any_pod_exists`, `bash scripts/gpu/tests/fv-lab-report-goldens.sh --check`,
`fv-lab-route.test.sh`; no GPU.

§10, CPU: `bash scripts/gpu/tests/fv-lab-goldens.sh --check` (frozen fixtures present, branches gone),
`tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `tests/fv_lab_bin.rs::runs_dir_replaces_the_run_directory`,
`FV_DRY_RUN=1 runpod-http.sh run` for a retired and a script family, and every remaining family's
`FV_DRY_RUN=1 runpod-matrix.sh` output equal before and after the deletion (`headline` aside, which now prints fv-lab's).

### 13. D1c: `fv-lab report summary` and `report table`, and the tree fetch

Status: accepted 2026-10-26. CPU only: every report reads files a run directory already holds, so there is no GPU gate;
the goldens are the scripts' own output on committed run directories. Decision-id FVID-2026-10-26-fv-lab-report.

#### 13.1 What replaces what

| replaced | by | reads |
|---|---|---|
| the summary lines the retired families' `runpod-matrix.sh` branches copied into `live.log` (§9.4, §10 item 4) | `fv-lab report summary <run dir>` (`--kind tier`, the default; the tier is `--family` or the directory's parent) | `techniques`: the `compare/compare-clips-*.json` fields and each `*-base` / `*-env` / `*-prof` arm's last `denoise_s` / `total_s`; `b6`: each arm's load-stage lines of `stderr.log`, then the `compare-clips-b6-*` fields; `b4a`: the `compare-clips-b4a-*` fields; `h3arms`: the six technique arms' TeaCache reuse counts, then each `gate/gate-*.json` verdict; `trace`: the last `/gpu_trace` line of five arms (400 bytes); `ltxfps`: the sink / mp4 / frames PASS and FAIL lines (300 bytes); `ltxvae`, `ltxvae-gen`: the last three decode lines per arm (400 bytes). Every other tier's branch printed none: nothing, and a note on stderr. Not ported: b4a's per-arm `[<arm>] teacache reused forwards: <n>` and per-cell `b4a <cell>: pass\|fail (…)` lines and the family verdict line, because they are the verdict step's (§11.1): `fv-lab run` writes them into `live.log` itself, and `fv-lab gate` re-derives them offline |
| `scripts/gpu/hd_report.py` | `report summary --kind hd` | the cells' `benchmark.json`, the compare reports, `ffprobe` (a process, as in the script) on the kept mp4s; not the keyframe sharpness, and no `--jpg-out` (§13.4) |
| `experiments/h3-graphs-summary.py` | `report summary --kind h3-graphs` | `cells/`, `trace/` (`stderr.log` or `events.log`), `hashes.tsv` |
| `experiments/h3-tier-options-summary.py` | `report summary --kind h3-tier-options [--json OUT]` | `cells/`, `compare/clips`, `compare/gate`, `check/` |
| `experiments/h3-trim-nvfp4-summary.py` | `report summary --kind h3-trim-nvfp4 [--json OUT]` | `runs/`, the parity dumps |
| `experiments/fasth3-nvfp4-summary.py` | `report summary --kind fasth3-nvfp4 [--json OUT]` | `up-*`, `ours-*`, the parity dumps, the phase profile |
| `scripts/gpu/sol_bench_table.py` | `report table --kind sol-bench <run dir>… [--json F]` | `<cell>/benchmark.json` (or `frames/`), the `sol` stage report's `timings` (LingBot, Cosmos3), `summary.json`; writes `results.json` where the script did (the first run directory's parent) unless `--json` / `--stdout` |
| `scripts/gpu/timings.sh` | `report table --kind timings <run dir> [--out F]` | a `validate.sh` run's `remote/*.json`, `summary.json`, `cached-refs`; writes `timings.md` and prints it |

`report tiers` is ADR-0023's. The ports are line for line, including each script's quirks (a `.get` default that
applies only to a missing key, `x or y`, dict insertion order, `statistics.median` keeping an int, Python's float
`repr` and `.2e` spelling, jq's `//` taking the right side on `null`, `false` or an error); the regular expressions are
matched by hand with the scripts' backtracking order (no regex engine is a dependency of the crate). Python's
`json.dumps` output is reproduced by `pyfmt.rs` (an order-keeping JSON reader and writer), since `serde_json` sorts
keys in this workspace and spells floats differently (`1e-5`).

#### 13.2 Equality: the scripts' output on committed runs

`scripts/gpu/tests/fv-lab-report-goldens.sh [--check]` writes `crates/fastvideo-lab/fixtures/report/` from the scripts,
under `env -i` with `LC_ALL=C`, Python as `python3 -I` (stdlib only; no network, no download), and
`tests/report.rs` holds fv-lab to every fixture byte for byte (53 files):

- **Tier summaries**: all 31 committed run directories of the eight tiers. The deleted branches' lines are frozen in
  `fixtures/report/summary-script.sh`, verbatim between markers (`runpod-matrix.sh` at `5987e6a4`, `b4a`'s from
  `fixtures/b4a-script/`), in a frame that only drops the `log` stamp and supplies the loops that stood around the
  per-arm lines; `--check` compares each block with its source whenever the history holds `5987e6a4`. The 31 runs:
  - 18 script-side runs of the frozen version (`LIVE`): every fixture line is in that run's own `live.log`, and,
    for 16 of them, the lines of `live.log` shaped as that family's summary lines are exactly the fixture's, in order
    (the reverse check). The other two (`techniques/1de80f1-10240618`, `93696e5-10240625`) were committed with only
    `live.log` and `compare/`, so their arm-timing lines cannot be rebuilt and only the forward check holds;
  - 3 script-side runs of earlier versions of the summaries (`techniques/51898b8-10240107`, `90dd3f8-10240259`,
    `b6/a2ca568-10241101`): fixtures only, their `live.log` is another version's;
  - 2 runs whose files hold no summary line: `b6/a2ca568-10241022` (no arm directory was committed) and
    `ltxfps/3df276d-09272000` (no `stderr.log` was committed): empty fixtures;
  - 8 fv-lab-side runs (tags `-<MMDDHHMMSS>-<hex>`), whose `live.log` has none of these lines: the gap this closes.
- **hd**: the five `artifacts/runpod/hd/*` runs, `hd_report.py` run with `numpy` and `PIL` stood in by empty modules:
  none of the committed runs keeps keyframes, so neither is called and the output is exact. None keeps an mp4
  either, so `fixtures/report/hd-run/` (synthetic: mp4s under `frames/`, a hidden directory, a failing probe, non-ASCII
  and `-0.0` in a `benchmark.json`) covers `## streams`, with `fixtures/report/bin/ffprobe` (a stand-in that answers from
  the file name) first on `PATH` for the script and passed to fv-lab.
- **Experiments**: `h3-graphs` and `h3-tier-options` on their committed run directories (both equal the summaries
  committed beside them); `h3-trim-nvfp4` and `fasth3-nvfp4` on their archives laid out as the summaries read them
  (`fv-lab-report-goldens.sh layout trim|fasth3`, the one mapping the test also uses; the compare-clips reports, the
  `stderr.log` and the profile were not archived, so those parts are covered by unit tests of the matchers).
- **sol-bench**: the three committed run sets, and `fixtures/report/sol-bench-cells/` (synthetic, every `PUBLISHED`
  row, the LingBot and Cosmos3 paths, `-s15`, `-rs3`, a skipped cell, a later run of a cell, a missing and a
  trailing-slash argument), table and `results.json`.
- **timings**: `fixtures/report/timings-run/` (three committed stage reports and synthetic model / parity / clip / probe
  reports, a timeout report, a broken file, a cached CPU reference). Its numbers are spelled alike by jq 1.6 and 1.7.

`tests/fv_lab_bin.rs` runs the commands (summary from the parent's tier, `--json`, `--out`, refusals of another kind's
options and of an unknown tier).

#### 13.3 The tree fetch: `FV_FETCH_TREE=1`

`runpod-http.sh fetch_results` with `FV_FETCH_TREE=1` mirrors the whole run directory instead of the summaries:
`fetch_tree` walks the pod's listing (`<family>/<tag>/`) recursively, passes over `../`, absolute and query hrefs,
leaves out each path (`<family>/<tag>/…` as the listing spells it) that `grep -qE "$FV_FETCH_SKIP"` matches, keeps a
subdirectory's spelling and decodes a file's name (`printf '%b'` of `%HH`), and drops a file over `FV_FETCH_MAX_MB`
MiB (default 400; `curl --max-filesize`, 0 no limit) or whose download fails. fv-lab: `pod up --wait` and `pod fetch`
take `--fetch-tree [--fetch-skip ERE] [--fetch-max-mb N]` (`pod::fetch_tree`); the summaries' fetch is unchanged
without it. `runpod-http.sh run` maps `FV_FETCH_TREE=1`, `FV_FETCH_SKIP`, `FV_FETCH_MAX_MB` across (it refused them
before); `FV_FETCH_SKIP` / `FV_FETCH_MAX_MB` without `FV_FETCH_TREE=1` stay refused (the script ignores them); `plan`
fetches nothing and passes none.

`--fetch-skip` is an ERE subset fv-lab reads without a regex engine: alternatives of literal characters, `.`,
backslash-escaped punctuation, `^` at an alternative's start and `$` at its end. Anything else (brackets, groups,
repetition, `\d`-style escapes, GNU's anchors `\<`, `\>`, `` \` ``, `\'`, an empty alternative) is refused before a
pod exists. A pattern starting with `-` is refused by the command line (clap reads it as an option); the script handed
it to grep, which read it as one too.

What fv-lab's tree fetch adds to the script's (review of #198), because a pod's listing is not trusted and `up --wait`
deletes the pod after the fetch:

- **Names.** An entry is refused (logged, not fetched) unless its spelling and its decoded name are each a single
  component: not empty, `.` or `..`, and holding no `/`, `\` or NUL; a `%` not followed by two hex digits, and a name
  that is not UTF-8 once decoded, are refused too. Otherwise `href="%2Fhome%2Fuser%2F.bashrc"` decodes to an absolute
  path that `Path::join` would put in place of the run directory, `..%2F..%2Fx` and `../../x/` climb out of it, and
  `./` recurses forever. Every joined path is checked to be a child of the directory it is written into, and the walk
  stops at `TREE_DEPTH` (16) levels.
- **Size.** Files are fetched with `Http::download_capped` (fastvideo-deploy, additive: a provided method on the
  trait, overridden by the reqwest client): refused on a larger `Content-Length` before any byte is written, stopped
  mid-stream as soon as the bytes pass `--fetch-max-mb`, written under a temporary name (`.<name>.fv-lab-part`) and
  renamed; a failure removes both, as the script's `|| rm -f`.
- **No lost run.** The top-level listing is retried 5 times, 10 s apart (as `fetch_run`'s); the files `judge_run`
  reads (`RUN_FILES`, `DONE`) are fetched whatever `--fetch-skip` matches; a listing that stays empty is an error after
  those are fetched.
- `runpod-http.sh` refuses an `FV_FETCH_MAX_MB` that is not a decimal number (bash's `$(( … << 20 ))` reads `010` as
  octal and `0x10` as hex). Every skip pattern the
repository uses is in it (`/frames/`, `\.cache$`, `png`, `oracle-dump\.tar|/dump/|\.mp4$`, sol-bench's
`\.png$|/cold/|\.cache$|text-cache|\.wav$`). `tests/fetch_tree.rs` cuts `proxy` and `fetch_tree` out of
`runpod-http.sh`, runs them against a fake `curl` serving a fake pod, runs fv-lab's against the same files, and finds
the two trees byte-identical in four cases (no skip; `/frames/`; an alternation with a 1 MiB limit; anchors with no
limit); `fv-lab-route.test.sh` covers the mapping and the refusals.

#### 13.4 Differences from the scripts, and why

| | script | fv-lab | why |
|---|---|---|---|
| tier summary lines that went through the frame's `log` | `[HH:MM:SS] [family] ` in front | the line alone | the stamp is the frame's, not the summary's; the per-arm lines that went through `tee` had none either |
| a `stderr.log` holding a NUL byte | GNU grep (C locale) prints `binary file matches` and none of its lines | its lines | fv-lab reads bytes; a NUL in a stage log is not a reason to hide its lines |
| hd keyframe sharpness (Laplacian variance, high-frequency share of each kept PNG) | computed with numpy and Pillow | one row per keyframe set: `not computed: fv-lab reads no PNG (scripts/gpu/hd_report.py)` | no image decoder or FFT among the crate's dependencies; `hd_report.py` stays for it (§13.5) |
| a run directory the script would raise on (a missing key, a value of the wrong type, an unparsable number) | a traceback after the lines printed so far | an error naming the file and the key, exit 1, nothing printed | a partial report reads as a complete one |
| `timings` numbers | jq 1.7 echoes a number read from its input as spelled (`1.500`, `0.00001`); jq 1.6 its shortest form | the shortest form (jq 1.6, and 1.7 after arithmetic) | one spelling whatever jq the caller has; the fixture holds only numbers both spell alike |
| `timings` report file holding several JSON values | each processed | the file read as one value (else no rows, as a parse error) | `fv-gpucheck` writes one |
| file order | `validate.sh`'s glob in the caller's locale | byte order (C) | the goldens run under `LC_ALL=C` |
| `--fetch-skip` | any ERE | the subset of §13.3, else refused before the pod | no regex engine; a pattern either means what grep matches or fails early |
| `--fetch-tree` on `pod up` | n/a | only with `--wait` | otherwise nothing would fetch; `pod fetch --fetch-tree` later |
| a listing entry that is not a single name (absolute, `..`, `.`, a `/`, `\` or NUL once decoded) | followed | refused and logged | it would write outside the run directory or loop (§13.3) |
| a malformed `%` escape in an entry (`%2` at the end, `%Ag`), a name not UTF-8 once decoded | `printf '%b'` rules (`%2` kept, `\xA` then `g`), bytes as they are | refused and logged | no server's listing holds one (`urllib.parse.quote`); a name fv-lab cannot spell is not guessed |
| a file over `--fetch-max-mb` | curl stops on `Content-Length` or mid-stream | the same, then a temporary file removed | |
| the top-level listing | one read | 5 tries; empty after them: an error once `RUN_FILES` and `DONE` are fetched | `up --wait` deletes the pod next |
| `RUN_FILES` and `DONE` matched by `--fetch-skip` | skipped | fetched | `judge_run` reads them |
| `FV_FETCH_MAX_MB=010`, `0x10` | 8, 16 MiB (bash arithmetic) | refused by the route | fv-lab reads a decimal |
| a skip pattern starting with `-` | an option to grep | refused (clap) | |
| `hd_report.py --jpg-out DIR` | makes DIR, prints `## crops -> DIR`, writes no crop | no such option | it wrote nothing; an empty header is not a report |
| JSON the reports read with a leading zero, `1.`, a raw control character in a string, a lone surrogate | `json.loads` refuses all but the last, which it keeps in a `str` | refused (an error) | a Rust `String` cannot hold a lone surrogate; the rest is Python's own rule |
| `FV_FETCH_SKIP` / `FV_FETCH_MAX_MB` without `FV_FETCH_TREE=1` | ignored | refused (`runpod-http.sh` route) | refused rather than dropped (§10) |

#### 13.5 Retired, and what stays

- **Deleted**: `scripts/gpu/sol_bench_table.py` and `scripts/gpu/experiments/{fasth3-nvfp4,h3-graphs,
  h3-tier-options,h3-trim-nvfp4}-summary.py`: nothing called them (docs named them; the docs now name the fv-lab
  commands). Their fixtures are frozen: `fv-lab-report-goldens.sh` runs each from `f9e57ea2` (the last commit with them)
  when the history holds it, and otherwise keeps the fixtures and says so (CI's shallow clone). The sol-bench
  `PUBLISHED` map is now `crates/fastvideo-lab/src/table.rs`.
- **Stays** `scripts/gpu/hd_report.py` until the `hd` family moves (§9.1: a cell type for derived clips, or `hd`
  retired) or the keyframe sharpness lands in `fv-gpucheck` (it reads PNGs); `report summary --kind hd` already covers
  the rest.
- **Stays** `scripts/gpu/timings.sh` while `validate.sh` (the Vast driver, outside fv-lab) calls it; it goes when
  `validate.sh` is retired (C3 / D1e) or calls `fv-lab report table --kind timings`, which needs fv-lab on the
  driver's machine.
- The frozen summary lines (`fixtures/report/summary-script.sh`) are a test reference only; nothing runs them on a pod.

#### 13.6 For D1d: reports that are verdicts

Some reports decide pass / fail rather than describe, and belong in the typed `[verdict]` step (§11.1) rather than in
`report`: the `determinism` / `det-short` families' frame-hash equality across runs (`determinism.txt`), h3-graphs'
byte identity of every checkpoint / prompt / canvas group (`hashes.tsv`, `MISMATCH`), fasth3-nvfp4's lossy-gate hard
quality windows applied to the parity clip (`gate_hard_quality`), and h3arms' reuse count (already a `[[verdict.count]]`
in b4a). `report` prints them as the scripts did; D1d gives them a type (a frames-hash equality list and a compare-
report window rule under `[verdict]`), so `fv-lab gate` can re-evaluate them offline.

### 14. D1d, second slice: `det-short`, `determinism` and `oracle` as tiers

Status: proposed 2026-10-26 (CPU part done in the draft PR; the GPU checks, §14.7, are the coordinator's). No script
branch is deleted here: each family leaves `runpod-matrix.sh` only after its own check through both runners (§8).
Decision-id FVID-2026-10-26-d1d-oracle-determinism.

#### 14.1 What moved

| tier | cells | arms | the script it is |
|---|---|---|---|
| `det-short` | `det-short-{h3,wan,ltx,ltx-sage}` | 15 | `runpod-matrix.sh det-short` at its defaults (`FV_DET_CELLS="h3 wan ltx ltx-sage"`, `FV_DET_REPEATS=3`): three fresh processes per group and the timed-pick control (not for `ltx-sage`), every arm's frames hashed after it, `determinism.txt` |
| `determinism` | `determinism-{ltx-t2v,ltx-i2v,h3}` | 9 | `runpod-matrix.sh determinism` without `FV_ORACLE_URL` (its oracle prelude is the `oracle` tier): per model `-ab` (two runs in one process), `-c` (one in another), `-timed`; the prompt sets written into the run directory; hashes per run, stage digests, `determinism.txt` |
| `oracle` | `oracle-<target>` for `fasth3-8step`, `fasth3-8step-vsa0`, `fasth3-4step-vsa`, `fasth3-4step-dense`, `h3-ref2va-4step`, `ltx25-512p`, `ltx25-512p-dense`, `ltx25-4k`, `ltx25-4k-dense` | 32 (14 gated, 18 `compare-dumps`) | `runpod-matrix.sh oracle` with `FV_ORACLE_TARGETS` naming those nine (the tensor-diff targets: the reference fetched, our run on its injected inputs, `compare-dumps`; the f32 noise floor for the H3 targets, the condition-row control for Ref2VA) and `FV_GEN_TIMEOUT_S=5400` (what `oracle.sh` passes) |

Every tier is the `rtx6000` tier's pod (the PRO 6000, the six preflight cells, the TAE fetch, a 4 h cap), as
`runpod-http.sh` gives every `FV_FAMILY`; `oracle`'s per-arm cap is 5400 s. Weights: every tree is on the EU volume
`bz0ofkfx45` except `h3-8step` (the two `fasth3-8step*` oracle cells) and `fastwan22-ti2v-5b` (det-short's `wan`
group), whose arms both runners record as skipped (`weights incomplete`), as before; `tests/lab_files.rs` pins those
two as the only trees any tier lacks there. No new tree, no download.

#### 14.2 Cell schema additions (v1, additive, `deny_unknown_fields` kept)

Every cell file before them parses unchanged, the gold cells' generated text is unchanged, `schemas/cell.v1.json` is
regenerated (`tests/schema.rs`).

6. **Ungated arms.** `[[arm]] gate` is optional. Without it the arm is the script's bare `run_cell` (no
   `verify-weights.sh`, no `FV_DRY_RUN_GATES` row); it may read no weight tree (`trees` empty, no `{weights}`, no
   `tae`). Used by the oracle's `compare-dumps` arms. (§9.1 named this step for `kernels`-style arms; those families
   still need more, §9.1.)
7. **`[[arm]] fresh`, `requires`, `echo`.** `fresh`: paths below `{runs}/` removed before the arm, selected or not
   (`oracle_run`'s `rm -rf "$dump"`). `requires`: directories that must all exist, else the arm is passed over without a
   trace (`oracle_diff`: no dumps, nothing to compare; a dry run lists it all the same). `echo`: `stderr.log` line
   patterns (the verdict step's matcher) written into the log after the arm as `[<arm>] <line>` (`oracle_diff`'s
   `compare-dumps` lines).
8. **`[reference] target`** and the `{ref}` placeholder. The upstream reference of an oracle target: the runner fetches
   it as the script's `oracle_fetch` does (poll every base URL of `FV_ORACLE_URL` for `oracle-<target>/ORACLE_DONE` every
   30 s up to `FV_ORACLE_WAIT_S`, default 10800; then `oracle-dump.tar` with curl, three attempts, `tar -xf`, a `dump/`
   directory) into `<scratch>/oracle-ref/<target>/`, and `{ref}` is its `dump/`. All selected references are fetched in
   the background from the start of the run (the script's prefetch), and `<runs>/ORACLE_REFS_DONE` is written once all
   are here (what `scripts/gpu/oracle.sh` waits for to delete the upstream pods). An unavailable reference skips the
   cell, recorded on its first arm (`reference dump unavailable`); after the cell the reference and every arm's `fresh`
   path are removed. `FV_ORACLE_URL` takes the script's spaces or commas (what `pod up --extra-env` can carry).
9. **`[[prompt_file]] file, names`.** A prompt set of the workload's prompt and seed under each name, written to
   `{runs}/<file>` before the cell's first arm, byte for byte as `det_set` prints it (`\` and `"` escaped). A file name
   is used once per tier; some arm must read `{runs}/<file>`.
10. **`[repeats] hash, reference, judged, controls, keep`.** Frame-hash equality across repeats (§13.6's first
    report): after each arm of the cell its frames are hashed and pruned to `keep`, and after every cell
    `<runs>/determinism.txt` reports each hash against the reference's, `IDENTICAL` or `DIFFERENT`, in the script's
    words. `hash = "arm"` is `det-short`'s `sh_hash` (every `frame-*.png` under the arm but `*/warmup/*`, in path-byte
    order, concatenated, and every `audio.wav`; `<arm>/hashes.json` and a line of `<runs>/hashes.txt`); `hash = "run"`
    is `determinism`'s `det_hash` (per run directory `frames/<run>/` but `warmup`, its frames in name order and its
    `audio.wav`, `<arm>/hashes.txt`; the stage digests of `stderr.log` into `<arm>/digests.txt`). `reference` is an arm,
    or `<arm>/<run>`; `judged` ∪ `controls` are the cell's arms, the reference arm judged; one hash kind per tier (one
    `determinism.txt`). Ported in `src/repeats.rs` (no regex engine: the two `sed`s and `read -r` are matched by hand).

#### 14.3 The typed verdict, and why it writes no file

A `[repeats]` cell **passes** when its reference has frames and every hash of its `judged` arms equals the reference's
(frames and audio); `controls` (the timed-pick runs, which differ by design: docs/perf/determinism.md) are reported,
not judged; a reference arm that did not run (skipped on its weights) leaves the cell undecided. The runner logs
`repeats <cell>: pass|fail (<reasons>)` (reasons `<arm>:different`, `<arm>/<run>:frames-different`, `<ref>:no-frames`,
`<arm>:missing`, `<arm>:no-runs`) and `fv-lab gate` decides it again offline: it rebuilds `hashes.txt` from each arm's
`hashes.json` and `determinism.txt` from the per-arm hashes, which both fetches take, so frames are never needed
(`--check` compares the two files byte for byte with the run's own).

It is not a list under `[verdict]`, and it writes no `*.verdict.json`: `[verdict]`'s frame is `b4a`'s script behaviour
(fresh arm directories, the `rm -rf` dry-run lines, a verdict file per cell and per family), and the determinism
families' script does none of it. Written files would exist in fv-lab's run only, and the GPU comparison is file for
file (`fv-lab-compare-runs.sh` counts a verdict file in one run as a difference). The decision lives where it can be
checked against the script: `determinism.txt` (byte-identical) and `fv-lab gate`'s outcome and exit status.

The other two reports of §13.6 are not moved here: h3-graphs' byte identity (`hashes.tsv`, `MISMATCH`) and
fasth3-nvfp4's lossy-gate windows belong to `scripts/gpu/experiments/`, which D1e moves (a `[repeats]`-like group check
and a compare-report window rule when those experiments become cells); h3arms' reuse count already is a
`[[verdict.count]]`.

#### 14.4 Equality

- **Goldens.** `fv-lab-goldens.sh` writes the three tiers' fixtures from the live branches like every script tier
  (`<tier>-dry-run.txt`, `<tier>-gates.tsv`, `runpod-http-<tier>.json`; 17 / 11 / 43 dry-run lines). One tier is the
  script under fixed settings: `oracle` is generated with `FV_ORACLE_TARGETS` set to its nine targets and
  `FV_GEN_TIMEOUT_S=5400` (`fam_env` in the script). `tests/{dry_run,payload_golden}.rs` hold fv-lab to them; the gates
  test skips ungated arms as `FV_DRY_RUN_GATES` does.
- **Script changes.** They change no generation, hash or report a finished run writes; one changes how a run with a
  failed reference arm ends (the second item):
  - dry run only: a dry `oracle_fetch` logs `dry-run reference <target>: <dump>` and fetches nothing (before, the dry
    run needed a URL and, without one, skipped every target); the prefetch does not start in a dry run; a dry
    `oracle_diff` logs its command; a dry `det_hash` returns (before, it failed under `set -e` on the arm directory a
    dry run does not make, after the first arm, so the dry run had one arm);
  - `determinism`'s verdict reads `|| true` like `det-short`'s. **A real-run change** (review of #202): on main a model
    that did not run (`FV_DET_CELLS`), or whose `-ab` arm was skipped or crashed (an empty `hashes.txt`), killed the
    verdict subshell under `pipefail` and the family with it, so `matrix_exit` was non-zero and `determinism.txt` cut
    short. Now the family goes on to `matrix done` (`matrix_exit=0`), and a reference arm that ran without run1 hashes
    says so: `<model>-ab: no run1 hashes` in `determinism.txt` and `live.log` (its other runs follow as `DIFFERENT`).
    fv-lab writes the same line (`<ref arm>: no <run> hashes`), and its typed verdict leaves the cell undecided when
    the reference arm did not run, a fail when it ran without frames; `tests/repeats_oracle.rs` (`noh3`) holds both
    byte-equal;
  - `det-short`'s arms pass one flat `env` (`env FASTVIDEO_CACHE=… XDG_CACHE_HOME=… [FASTVIDEO_SDPA_AUTO=timed]
    FASTVIDEO_WAN_VAE=full …`) where they nested two (`env … env FASTVIDEO_WAN_VAE=full …`): the same environment in the
    same order, a cell's one `env` list.
- **The other families' dry runs** (`FV_DRY_RUN=1` output and `FV_DRY_RUN_GATES` records, before and after, for every
  `case` branch): the same, but `identity` (a bash error message's line number), `writer` (its `dd` timings, which
  differ run to run) and `wan`, whose branch calls the same `oracle_fetch` / `oracle_diff`: its dry run now lists its
  oracle cells instead of `oracle: FV_ORACLE_URL unset`. (The old `oracle` dry run never ended: its prefetch watcher
  waited `FV_ORACLE_WAIT_S` for dumps a dry run never fetches.)
- **Fakes, both runners** (`tests/repeats_oracle.rs`): the live branches copied with what they source into a fake root
  and `fv-lab run` of the committed tiers on one fake box (a fake fv-gpucheck writing frames per prompt-set run with a
  warm-up directory, audio, stage digests and dumps, and answering `compare-dumps`; the upstream reference served as
  a `file://` URL to the real curl). Every output file is the same, byte for byte (hashes, `determinism.txt`, digests,
  prompt sets, kept frames, frames records), exits and skips agree, the `tee`'d report and `compare-dumps` lines of
  `live.log` are equal, the dumps and the reference are gone after a target, and a missing reference skips its cell
  on both sides; a `flaky` mode makes repeats differ and both reports and the verdict say so; `fv-lab gate --check`
  finds no difference on either run directory and decides the verdicts. `fv-lab-compare-runs.sh` compares the hashes
  files and each arm's `compare-dumps` lines (self-test in `fv-lab-goldens.sh`).

#### 14.5 Differences from the script, and why

| | script | fv-lab | why |
|---|---|---|---|
| `FV_DET_CELLS`, `FV_ORACLE_TARGETS` | pick groups / targets | refused by `fv-lab run` and `pod up --extra-env`; `--cells` by arm (a group's or target's arms) | arms are the unit (§9.4); one cell per group / target |
| `FV_DET_REPEATS`, `FV_DET_PROMPT`, `FV_DET_T2V_PROMPT`, `FV_DET_I2V_PROMPT`, `FV_REF2VA_PROMPT`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT` | change the arms | refused (`knobs::SCRIPT_KNOBS`) | the tiers are the defaults; a variant is another tier (§12's way) |
| `FV_ORACLE_PREFETCH=0` | sequential fetches | refused: always in the background | the result is the same; only the upstream pods' lifetime differs |
| `FASTVIDEO_DUMP_OPS` other than `0,1,24,47` for `oracle` | the dumped blocks | refused | every arm sets it; another value would be overridden |
| `ORACLE_REFS_DONE` | written by a watcher once every target's `dump/` exists at the same moment (polled every 20 s up to `FV_ORACLE_WAIT_S`); the loop's cleanup removes each target's dump after its cell, so in a run of several targets an early target's dump can be gone before the last arrives and the file is never written (a script bug for multi-target runs, which then keep the upstream pods to the end) | written when every selected reference has been fetched (a count of successful fetches) | the meaning `oracle.sh` waits for: every reference is here, the upstream pods can go |
| a reference whose prefetch failed | fetched again inline (another `FV_ORACLE_WAIT_S`) | the cell skipped | the prefetch already waited and tried three downloads; a second 3 h wait on a GPU pod buys nothing |
| `FV_CELLS` naming some arms of an oracle target | the target's reference fetched all the same | fetched only when one of its arms is selected; other targets `skip <cell> (no arm in FV_CELLS)` | selection by arm |
| `oracle-h3-ref2va-*`'s `-dump-cond` after the cell | left on the container disk | removed with the other dumps | disk only |
| `determinism` with an unselected arm (`FV_CELLS`) | `det_hash` fails, the family ends | the arm is not hashed | a run that cannot finish is no reference |
| `determinism` with `FV_ORACLE_URL` | runs the `oracle` family first | not read by `determinism` | the oracle is its own tier |
| a `stderr.log` holding a NUL | `grep` in the C locale writes `Binary file … matches` into `digests.txt` / drops the echo | the lines | as §13.4: bytes, not a locale |
| `pod up` of `oracle` | (the script ran whatever it was given) | needs `--extra-env FV_ORACLE_URL=…` and `--cells` | otherwise every cell skips, or a reference never made upstream holds a GPU pod for `FV_ORACLE_WAIT_S` |

#### 14.6 What stays in the script, and when it moves

- **The oracle's clip-metric targets** (`ltx25-{i2v,kf,ref2v,a2v,a2v-i2v,a2v-guided,retake,retake-v,retake-a,extend}`)
  and **`sfwan13`**: after their runs and diffs they compute ffmpeg SSIM / PSNR and ffprobe lines into
  `oracle-<target>-frames/metrics.txt` (select filters, pinned frames, the source's kept frames) and, for `sfwan13`,
  decode the reference mp4 into frames for `compare-clips`. That is a derived-clip step, the one `hd` waits for (§9.1);
  they move with it. `FV_ORACLE_OWN_TEXT=1`'s `-owntext` arms are another tier when wanted.
- **`runpod-http.sh upstream`** is not moved or aliased. It boots another image (`fastvideo-rs-upstream-<target>`, the
  Python references, no fv-gpucheck and no fv-lab), runs `scripts/gpu/upstream/pod.sh` (`UP_STEPS`), serves results
  with its own start command and idle watchdog (`net`: busy while fetched from), and is kept alive and deleted by
  `oracle.sh` (`FV_KEEP_POD`, `ORACLE_REFS_DONE`, the keepalive fetches). An fv-lab tier is a list of fv-gpucheck arms
  on the runtime image, so the upstream pod is a second `ImageFlavour` and start command, and the orchestration an
  `fv-lab oracle` driver (upstream pods up, the runtime tier with `FV_ORACLE_URL`, release on `ORACLE_REFS_DONE`).
  It moves with that driver, after the `oracle` tier's GPU check and branch retirement, and before `runpod-http.sh` is
  deleted (D1e's end). Until then `oracle.sh` drives `runpod-http.sh upstream` and, for the runtime side, either
  runner: `runpod-http.sh run` with `FV_FAMILY=oracle`, and once the branch is retired `FV_EXTRA_ENV`'s
  `FV_ORACLE_TARGETS` becomes `FV_CELLS` (oracle.sh's change in the retirement PR).
- **`det-short`'s `ltx-sol` group** (off by default) and other `FV_DET_REPEATS`: tiers when wanted.
- **`identity`** and **`precision-debug`** (§9.1): cases from the environment, compare-dumps only when dump directories
  exist, frame deletion. Their verdicts are hash equalities (`identity`) and dump diffs (`precision-debug`), which
  `[repeats]` and the ungated `requires` / `echo` arms now cover; what is left is that their arm lists come from the
  environment (`FV_ID_CASES`). They stay "D1e or retired" (§9.1).

#### 14.7 The GPU checks (the coordinator's)

The procedure is §8's: one commit `SHA` for both sides (this PR's head or its merge, once
`fastvideo-rs-runtime:sha-<7>` is published with its `lab/` and fv-lab), the cheapest cells that exercise the new
parts, both sides in parallel, then `fv-lab-compare-runs.sh`. RTX PRO 6000, EU, on `bz0ofkfx45`, about $2.49/hr; pod
boot and preflight ~5 min per side.

```sh
SHA=<7>
# det-short, the h3 group: three fresh processes and the timed control, hashes and determinism.txt
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family det-short --cells h3-r1,h3-r2,h3-r3,h3-timed --yes --wait
RUNPOD_API_KEY=… FV_FAMILY=det-short FV_EXTRA_ENV="FV_DET_CELLS=h3" FV_FETCH_TREE=1 FV_FETCH_SKIP='/frames/|/cache/|adaln\.cache$' \
  bash scripts/gpu/runpod-http.sh run $SHA
L=artifacts/runpod/det-short/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/det-short/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh --expect-verdicts 6 $L $S     # determinism.txt, hashes.txt, 4 hashes.json
./fv-lab gate $L --check --out /tmp/regate-ds-lab
./fv-lab gate $S --check --out /tmp/regate-ds-script --cells h3-r1,h3-r2,h3-r3,h3-timed

# determinism, the h3 model: two runs in one process, one in another, the timed control (1088x1920 5 s)
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family determinism --cells h3-ab,h3-c,h3-timed --yes --wait
# (FV_DET_CELLS is space separated; runpod-http.sh writes FV_EXTRA_ENV into the start command as shell text,
#  so the inner quotes hold the list together, as oracle.sh's FV_ORACLE_URL='…' does)
RUNPOD_API_KEY=… FV_FAMILY=determinism FV_EXTRA_ENV="FV_DET_CELLS='h3-ab h3-c h3-timed'" \
  FV_FETCH_TREE=1 FV_FETCH_SKIP='/frames/' bash scripts/gpu/runpod-http.sh run $SHA
L=artifacts/runpod/determinism/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/determinism/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh --expect-verdicts 4 $L $S     # determinism.txt, 3 hashes.txt
./fv-lab gate $L --check --out /tmp/regate-det-lab
./fv-lab gate $S --check --out /tmp/regate-det-script --cells h3-ab,h3-c,h3-timed

# oracle, fasth3-4step-vsa: one upstream pod makes the reference, both runtime sides fetch it
RUNPOD_API_KEY=… FV_KEEP_POD=1 FV_POD_FILE=/tmp/up.pod UP_IMAGE_TARGET=fastvideo UP_IMAGE_TAG=latest \
  UP_STEPS="info:box weights:h3-diffusers oracle:fasth3-4step-vsa" UP_CELL_TIMEOUT_S=7200 \
  FV_EXTRA_ENV="FASTVIDEO_DUMP_OPS=0,1,24,47" FV_FETCH_SKIP='oracle-dump\.tar|/dump/|\.mp4$' FV_CONTAINER_DISK_GB=150 \
  bash scripts/gpu/runpod-http.sh upstream $SHA & UPD=$!   # keeps the pod; writes "<id> <tag>" to /tmp/up.pod
until [ -s /tmp/up.pod ]; do sleep 20; done; read -r UP UPTAG </tmp/up.pod
URL=https://$UP-8000.proxy.runpod.net/upstream/$UPTAG
until curl -sf -o /dev/null $URL/oracle-fasth3-4step-vsa/ORACLE_DONE; do sleep 30; done   # the reference is served
# oracle.sh's keepalive (its traffic keeps the upstream pod's net watchdog from deleting it), bounded to 90 min
# whatever happens below, so a stuck runtime side cannot hold the upstream pod forever:
timeout 5400 bash -c "while sleep 60; do curl -s -o /dev/null https://$UP-8000.proxy.runpod.net/keepalive.bin; done" & KA=$!
CELLS=oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family oracle --cells $CELLS --extra-env FV_ORACLE_URL=$URL --yes --wait & LAB=$!
RUNPOD_API_KEY=… FV_FAMILY=oracle FV_GEN_TIMEOUT_S=5400 \
  FV_EXTRA_ENV="FV_ORACLE_URL=$URL FV_ORACLE_TARGETS=fasth3-4step-vsa FASTVIDEO_DUMP_OPS=0,1,24,47" \
  bash scripts/gpu/runpod-http.sh run $SHA & SCR=$!
wait $LAB $SCR                                             # both runtime pods done (each fetched the dump first)
kill $KA; RUNPOD_API_KEY=… bash scripts/gpu/runpod-http.sh down $UP; wait $UPD   # the upstream pod: only after both
L=artifacts/runpod/oracle/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/oracle/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh $L $S
```

Pass, for each family: `fv-lab-compare-runs.sh` exits 0 and both pods end `matrix_exit=0`:

- `det-short`: 4 arms the same (4 ran, frames byte-identical, about 124 frames each), 6 verdict files the same, and
  both `fv-lab gate --check` with 0 differences. The verdict (`repeats det-short-h3`) is pass on both when r1-r3 are
  identical (the 2026-10-06 run's were); a fail on both sides is no runner difference.
- `determinism`: 3 arms the same, 4 verdict files the same, both `--check` clean.
- `oracle`: 5 arms the same (2 ran with byte-identical frames, the 3 `compare-dumps` arms with the same
  `compare-dumps` lines: "3 with the same compare-dumps lines"), both sides with `ORACLE_REFS_DONE` reached (fv-lab's
  `live.log`: `oracle: fasth3-4step-vsa reference <size>`).

**Estimates** ($2.49/hr): `det-short` h3 group: 4 arms of 160-190 s (run `6ec6c64-10061138`), ~12 min + 5 min per
side, ~$0.70 per side, **~$1.40**. `determinism` h3: 1088x1920 5 s 4-step VSA, ~2 min load and ~1-1.5 min per run, 5
runs in 3 processes, ~13 min + 5 min per side, **~$1.50**. `oracle` fasth3-4step-vsa: the upstream pod ~45 min (15 min
image pull, the `h3-diffusers` view, the reference; docs/oracle.md: 35 min for an LTX reference), ~$1.90; each
runtime side two H3 runs with dumps and three `compare-dumps`, ~10 min + 5 min, ~$0.60 per side; **~$3.10**. All
three about **$6** (and the `ltx` group of det-short, ~10 min more per side, ~$0.85, if wanted). Stop before the
balance would fall below $8 (CLAUDE.md).

Byte identity is expected for the frames and hashes on one GPU class (docs/perf/determinism.md), with the caveats of
§12 (the same generation source and tools build on both sides: one `SHA` here). The `compare-dumps` lines depend on
both runtime dumps and the one reference, so they too are the same when the frames are.

#### 14.8 Then

After a family's check passes, in the next D1 PR: it joins `FV_LAB_RETIRED`, its `runpod-matrix.sh` branch is
deleted (and, for `oracle`, only its moved targets: the clip-metric targets keep a reduced branch until §14.6's step,
or the branch stays whole and the tier is the moved part's equal; to be decided with the result), its fixtures
freeze, `knobs::SCRIPT_KNOBS` joins `FV_LAB_NOT_READ`, and `oracle.sh`'s runtime side passes `FV_CELLS` instead of
`FV_ORACLE_TARGETS`.

#### 14.9 Review of #202

- The oracle check's commands (§14.7) wait on the two runtime sides by PID, and the keepalive loop has its own
  90-minute bound, so the upstream pod is always deleted after them.
- `determinism`'s `|| true` is a real-run change, stated in §14.4, and a reference arm without run1 hashes is reported
  (`<model>-ab: no run1 hashes`) by both runners.
- `FV_ORACLE_URL` is `http(s)://` only: `fv-lab run` and `pod up --extra-env` refuse anything else (a `file://` URL, a
  value that would read as a curl option), and every curl call passes `--proto =http,https --proto-redir =http,https
  --url <u>`. The tests' fake upstream is a `file://` URL through `RunConfig::allow_file_urls`, which no flag or
  variable of the binary sets.
- Validation: a cell with both `[reference]` and `[verdict]` is refused (its verdict would never run), and so are two
  cells of one tier with the same `reference.target` (one `oracle-ref` directory, fetched twice at once). A `fresh`
  path must name something below `{runs}/`: at least one component, none empty, `.` or `..` (`{runs}/` or `{runs}/.`
  would have removed the run directory).
- §14.5 gains the `ORACLE_REFS_DONE` row; the det-short check's script side skips the AdaLN caches
  (`adaln\.cache$`).
