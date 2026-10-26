# ADR-0019: `fv-lab`, the run plan in `lab/`, and the pod on fv-deploy's guards

- Status: Accepted for D1a (`fv-lab run` and `fv-lab pod` for the `rtx6000` family, the `lab/tiers.toml` and
  `lab/cells/*.toml` schemas, the dry-run and payload goldens; GPU gate passed) and D1b (nine more families as tiers,
  four additive cell-schema fields, a golden per tier, §9; GPU gate passed for all nine). The ten families' script
  branches are deleted and the tiers are fv-lab only, with frozen goldens (§10, 2026-10-25). D1d's first slice
  (§11: a typed verdict step in the cell schema, `b4a` as a tier, `fv-lab gate`) is accepted: its GPU gate passed on
  `e82b6b7` and `b4a` is retired from the script like the ten (§11.6). Two dropped modes are tiers again, frozen
  from the script at `5987e6a4` (§12, accepted; GPU check passed on `8b18d89`).
  D1c (§13: `fv-lab report summary` and `report table`, the `FV_FETCH_TREE` mirror; CPU only, no GPU gate) is
  accepted. D1e (§14: the port one-offs, one kept as the tier `h3-graphs`, nine deleted; `runpod-http.sh`'s callers
  listed) is accepted: the `h3-graphs` tier's GPU check passed on `f93a961` and its two experiment scripts are deleted
  (§14.4, "Result"). D1d's second slice (§15: `det-short`, `determinism` and `oracle`'s tensor-diff targets as tiers,
  the typed `[repeats]` frame-hash check, references, ungated arms, prompt files) is accepted: all three GPU checks
  passed on `f93a961` (§15.7), `det-short` and `determinism` are retired from the script, and `oracle`'s branch stays
  whole, the tier its tensor-diff part's equal (§15.10). The oracle's clip-metric targets and `oracle.sh`'s runtime
  side, `runpod-http.sh upstream`, and the retirement of the rest of `runpod-matrix.sh` and `runpod-http.sh` follow
  (§8, §14.6, §15.6, §15.10). §16 (`weights = "volume+r2"`: the trees the EU volume lacks fetched from the R2
  weights store per arm; the `gold` tier on it) is proposed, CPU-tested; its GPU check is the full gold run (§16.7).
  §17 (D1f: `fuse`, `attn3` and `ltx` as tiers, no new cell field, the script's `ltx` arms gated) is accepted for
  those three: their GPU checks passed on `762ce4d` and they are retired from the script (§17.5 "Results", §17.6).
  `b4b` / `b4c` / `b4c2` did not move (§17.3). §18 (D1g, coordinator decision 2026-10-26) is accepted: those three,
  `h3` (§17.4) and §14.7's `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`, `identity` and `h3attn` are
  retired from the script with no tier (`FV_SCRIPT_RETIRED`, refused with a pointer). §19 (D1h: `fv-lab oracle`, the driver of §15.6: the upstream reference pods of
  `lab/upstream.toml` on fv-deploy's guards, the `oracle` tier's runtime pod with `FV_ORACLE_URL`, their release on
  `ORACLE_REFS_DONE`, a bounded wait for it, a stock wait or any failure; payload goldens from `runpod-http.sh plan-upstream`) is proposed,
  CPU-tested; its GPU check is §19.7. §20 (D1i, the coordinator's decisions on §19.9: a typed `[[metric]]` step in
  the cell, `metrics.txt` a report, the ten LTX clip-metric targets as the `oracle` tier's `extra_cells`) is proposed,
  CPU-tested; its GPU check is §20.7. `sfwan13` is a follow-up (§20.5), so `oracle`'s branch stays whole
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-fv-lab
- Code: `crates/fastvideo-lab/src/{plan,run,pod,quote}.rs`, D1c `src/{summary,table,experiments,pyfmt}.rs` (§13), `src/bin/fv-lab.rs` (feature `cli`), `src/pod-start.sh.in`;
  `scripts/gpu/fv-lab-compare-runs.sh` (the GPU gate's comparison); fv-lab in the tools release's `gpucheck` set;
  schemas `crates/fastvideo-lab/schemas/{tiers,cell}.v1.json`; data `lab/tiers.toml`, `lab/cells/*.toml` (D1a
  `rtx6000-*`, D1b `techniques-*`, `b6-*`, `rtx5090-*`, `fastvideo-*`, `h3arms-*`, `trace-*`, `ltxfps-*`, `ltxvae-*`,
  `precision-*`; D1d `b4a-*`, §11; D1e `h3-graphs-*`, §14; D1d `det-short-*`, `determinism-*`, `oracle-*`, §15, with
  `src/repeats.rs`; D1f `fuse-*`, `attn3-*`, `ltx-*`, §17);
  goldens `crates/fastvideo-lab/fixtures/*` (written by `scripts/gpu/tests/fv-lab-goldens.sh`, frozen for the tiers of
  `scripts/gpu/fv-lab-retired.sh`, §10); one seam in
  `fastvideo-deploy` (`runpod::Pod::list`, §6)
- Usage: `docs/dev/fv-lab.md`
- §16: `crates/fastvideo-lab/src/weights.rs`, `run.rs` (`stage_r2`), `pod.rs` (`launch` env); tests
  `tests/r2_runner.rs`, `tests/gold_tier.rs`
- §19: `crates/fastvideo-lab/src/oracle.rs`, `src/oracle_live.rs`, `src/upstream-start.sh.in`, `src/bin/fv-lab.rs` (`oracle`),
  `lab/upstream.toml`, `schemas/upstream.v1.json`, `scripts/gpu/runpod-http.sh plan-upstream`; tests
  `tests/oracle_driver.rs`, `tests/oracle_live.rs`, fixtures `runpod-http-upstream-*.json`
- §20: `plan.rs` (`Metric`, `Measure`, `Step::Metric`, `Tier::extra_cells`), `run.rs` (`Runner::metric`), `oracle.rs`
  (`runtime_fetch`), `lab/cells/oracle-ltx25-{i2v,kf,ref2v,a2v,a2v-i2v,retake,retake-v,retake-a,extend,a2v-guided}.toml`;
  tests `tests/dry_run.rs`, `tests/repeats_oracle.rs`; fixtures `oracle-extra-{dry-run.txt,gates.tsv}`
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
| `min_ram_gb` | optional: the pod's host-RAM floor, whole GB per GPU (Runpod `minRAMPerGPU`, positive); absent, none is asked for (§16.10) | absent (`FV_MIN_RAM_GB` unset); the gold tier's is 256 (192 until the run at `7e580e5`, §16.10) |
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
`h3-8step` tree is R2-only since 2026-10-13), which `tests/lab_files.rs` pins. Since §16 a tier may also say
`weights = "volume+r2"`: the volume's trees as here, and every other tree fetched from the R2 weights store onto the
container disk before the arms that read it (the `gold` tier); `volume` is unchanged.

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
  is deleted once nothing calls it (`upstream` mode moves with D1d). Done (§14): `h3-graphs` (its h3-turbo cells) is a tier,
  its scripts kept until its GPU check (§14.4); the other nine experiments are deleted; what still calls
  `runpod-http.sh` is §14.6.

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
| `b4b` | 10 in 5 cells (5 / 0) | yes | after each pair: the CUPTI kernel tables of both arms, `fusion-evidence.log`, `trace-meta.txt`, `<cell>.fusion-evidence.diff`, the cell verdict; the family verdict | a trace-evidence verdict (kernel table, named kernels, block arithmetic): large | not moved (D1f, §17.3) | **retired, no tier** (§18, §17.3 option (b)): a finished gate (passed 2026-10-24, #179; `artifacts/runpod/b4b/`) |
| `b4c` | 8 in 2 cells (6 / 0) | yes | per pass a `benchmark.json` field line; per arm group a shared cache directory under `$SCRATCH`, its listing and temporary-file check; the listings' diff; the cell and family verdicts | `benchmark.json` field rules, cache listings, per-group fresh dirs under `{scratch}`, the script's reason order: large | not moved (D1f, §17.3) | **retired, no tier** (§18, §17.3 option (b)): passed 2026-10-24, #177; `artifacts/runpod/b4c/` |
| `b4c2` | 24 in 5 cells (15 / 0) | yes | as `b4c`, plus hit/miss state sets, entry counts, a subset and a same-bytes rule, three caches per switch cell | as `b4c`, and more: large | not moved (D1f, §17.3) | **retired, no tier** (§18, §17.3 option (b)): passed 2026-10-24, #178; `artifacts/runpod/b4c2/` |
| `rtx5090` | 17 in 5 cells (14 / 0) | yes (h3-base, ltx25, TAE) | compares after each resolution and workload: one cell per loop iteration | `tae` | moved | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, `h3-480p` dense/sol/fullopt/fullopt-taeh3, 4 arms × 248 frames byte-identical, 14 reports the same; `d1b-compare/rtx5090.txt` |
| `fastvideo` | 17 in 6 cells (7 / 7) | yes, but `h3-8step` (4 arms) | per iteration: one cell each | `tae`, paired gate | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 4 arms the same (2 ran, 248 frames each byte-identical; 2 `h3-8step` skipped in both), 8 reports and the gate report the same; `d1b-compare/fastvideo.txt` |
| `h3arms` | 13 in 3 cells (13 / 10) | `h3-8step` for 11 arms (skipped); 4-step VSA yes | a compare followed by its gate; per-arm reuse counts in live.log (log only) | paired gate, `tae` | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 13 arms the same (2 ran byte-identical, 11 `h3-8step` skipped in both), 14 reports the same; `d1b-compare/h3arms.txt` |
| `trace` | 5 in 2 cells | yes, but `h3-8step` (1 arm) | `/gpu_trace` lines copied into live.log after the arms (log only) | as is | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 2 arms the same (1 ran, 248 frames byte-identical; 1 skipped in both); `d1b-compare/trace.txt` |
| `ltxfps` | 5 in 2 cells | yes, but `h3-8step` (1 arm) | PASS/FAIL lines copied into live.log (log only) | as is | moved | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, 2 arms the same (1 ran, 121 frames byte-identical; 1 skipped in both); `d1b-compare/ltxfps.txt`; `FV_LTXFPS*` dropped |
| `ltxvae` | 12 in 4 cells (2 / 0) | yes (ltx25) | a compare after the first two arms of each workload: two cells per workload; `FV_VAE_GEN=1` arms (off by default) | as is | moved (default arms) | **retired (script branch deleted, fv-lab only)**, §10: `b0d0bb8`, `ltxvae-4k5s` fast/streaming, 2 arms × 121 frames byte-identical, the compare the same; `d1b-compare/ltxvae.txt`; `FV_VAE_GEN` dropped |
| `precision` | 8 in 2 cells (8 / 6) | ltx25 yes; `h3-8step` (4 arms) | compare + gate pairs | paired gate | moved (default mode) | **retired (script branch deleted, fv-lab only)**, §10: `c21d801`, `ltx25-4k5s-sol` f32act/bf16act, 2 arms × 242 frames byte-identical, 9 reports and the gate report the same; `d1b-compare/precision.txt`; the `ltx-nvfp4` mode dropped |
| `fuse` | 10 in 3 cells (3 / 3) | yes, but `h3-8step` (3 arms) | an ungated `kernels` run_cell; one compare block across H3 and LTX arms; `[PASS]`/`[FAIL]` and `/gpu_trace` lines in live.log (log only) | ungated arms (§15); `model` names both pipelines (§17.2) | moved (D1f, §17) | **retired (script branch deleted, fv-lab only)**, §17.6: `762ce4d`, `kernels-fuse` exit 0 (90 `[PASS]`), the 4-step VSA off / on pair 248 frames each byte-identical (the off / on compare `pass`, off-identity `ok`), the 8-step pair skipped on both; `artifacts/runpod/fuse/762ce4d-*`, `compare-762ce4d.txt` |
| `attn3` | 6 in 1 cell (3 / 0) | yes | one compare block across H3 and LTX arms (one cell, two pipelines) | as is: `model` is descriptive and names both (§17.2); `FV_ATTN3_*` refused | moved (D1f, §17) | **retired (script branch deleted, fv-lab only)**, §17.6: `762ce4d`, the VSA bf16 pair (b160 / b161) 248 frames each byte-identical (their compare `pass`, off-identity `ok`); `artifacts/runpod/attn3/762ce4d-*`, `compare-762ce4d.txt`; `FV_ATTN3_*` dropped (select by arm) |
| `ltx` | 3 in 1 cell | ltx25 yes; ltx23, ltx2 no (2 arms skipped) | ungated `run_cell`s that read weights (the two arms without their trees failed) | as is: the script's branch gates its arms since D1f, a real-run change in lockstep (§17.2) | moved (D1f, §17) | **retired (script branch deleted, fv-lab only)**, §17.6: `762ce4d`, `ltx25-two-stage` 242 frames byte-identical, the LTX-2.3 and LTX-2.0 arms skipped on both; `artifacts/runpod/ltx/762ce4d-*`, `compare-762ce4d.txt` |
| `eval` | 5 (3 / 3) | yes, but `h3-8step` | a prompt set and LPIPS by default; an LPIPS self-test only when LPIPS loaded | no | stays | prompt sets join the schema |
| `h3attn` | 4 processes, ~30 `--arm`s | partly | one process runs many arms (`{arm}` in its clip dir); FP8 arms dropped on a kernel FAIL | no | no tier | **retired, no tier** (§18): no attention arm of it is an open item: its arms are in `docs/techniques.md` "H3 attention arms", its FP8 follow-up closed by SageAttention (§18) |
| `hd` | 8 | yes | ffmpeg Lanczos upscale between the arms and their compares; keyframe prune; post script | no | stays | D1c (`report`) or a cell type for derived clips |
| `ltxoffload` | 3 to 6 | yes | 4K cells run only when the 512p frames hash equal (data-dependent); `env -u` | no | stays | a later slice |
| `cold` | 10 | yes, but `h3-8step` | page-cache drop before each LTX arm, PNG deletion, `env -u` | no | stays | a later slice |
| `serve-engine` | 6 | H3, LTX yes; Wan no | frames deleted after the run; `FV_ENGINE_CELLS`; `--tae-dir` without a gate | almost | stays | a later slice |
| `writer` | 7 | ltx25 yes | sysinfo, a sampler process, `dd` throughput | no | no tier | **retired, no tier** (§18): a one-time measurement (commit `a8ecfec6`); no result in `docs/perf`, no run committed (§18) |
| `determinism`, `det-short` | 9 to 19 | partly | prompt-set files written at run time, hashing, verdict files, frame deletion | `[[prompt_file]]`, `[repeats]` (§15) | moved in D1d (§15) | **retired (script branches deleted, fv-lab only)**, §15.10: `f93a961`, `det-short` h3 group 3 arms × 124 frames byte-identical, `h3-timed` a control (reported), 5 verdict files the same; `determinism` h3 2 arms (248 / 124 frames) byte-identical, `h3-timed` a control, 3 verdict files the same; `fv-lab gate --check` 0 differences on all four runs; `artifacts/runpod/{det-short,determinism}/f93a961-*`, `compare-f93a961.txt`, `gate-f93a961.txt`; `FV_DET_*` dropped (`det-short`'s opt-in `ltx-sol` group and other `FV_DET_REPEATS` went with the branch: tiers when wanted, §15.6) |
| `oracle` | 9 tensor-diff targets moved (32 arms); 10 clip-metric targets and `sfwan13` stay | partly | reference-dump fetches from upstream pods, compare-dumps, ffmpeg metrics | `[reference]`, ungated arms, `fresh` / `requires` / `echo` (§15) | tensor-diff targets moved in D1d (§15) | GPU check passed on `f93a961` (`fasth3-4step-vsa`: 5 arms the same, 2 × 124 frames byte-identical, 3 with the same `compare-dumps` lines; `artifacts/runpod/oracle/f93a961-*`, `compare-f93a961.txt`); **branch kept whole** (§15.10): the clip-metric targets, `sfwan13` and `oracle.sh`'s runtime side still run there; it retires with the derived-clip step and the `fv-lab oracle` driver (§15.6) |
| `sfwan` | many | no | recurses into `oracle` and `wan` | no | stays | D1d / a Wan volume |
| `wan` | ~35 | no (Wan trees R2-only) | oracle fetches, ungated kernels, `rm` | no | stays | D1d / a Wan volume |
| `b200` | 8 | partly | TAE and Spark file probes at run time; ungated | no | no tier | **retired, no tier** (§18): the US volume is gone; `runpod.sh b200` refused too |
| `h3` | 3 | `h3-8step` no | Spark file probes; ungated | no | no tier | **retired, no tier** (§18): §17.4's proposal accepted: `rtx6000` runs every arm but the refiner-less Spark fallback (`FASTVIDEO_LTX2_WEIGHTS` unset), which nothing serves, and the cold streamed-encoder runs; no `sol-h3-spark-no-refiner` arm |
| `headline` | the others' | partly | runs other families with `FV_CELLS` subsets; since §10 its four sources (`fastvideo`, `rtx6000`, `rtx5090`, `precision`) through `fv-lab run <tier> --cells … ` into its run directory (`FV_RUNS_DIR`) | no: a tier of other tiers' arms | stays | a tier with an arm filter |
| `mmaudio`, `speechtest` | 10+ | no | Python venvs, upstream clones, Whisper | no | stay | D1e |
| `sfstream`, `sfquality` | 2 to 4 | no (sfwan R2-only) | TAEHV fetch, run lists from env | no | stay | D1e |
| `precision-debug` | 7 to 10 | partly | compare-dumps only when dump dirs exist; dump deletion | no | no tier | **retired, no tier** (§18): the `precision` tiers run the same variants, not its per-step CUPTI trace or its dump diffs |
| `identity` | from `FV_ID_CASES` | partly | cases from the environment, hashing, frame deletion | no | no tier | **retired, no tier** (§18): the `det-short` / `determinism` tiers cover it |
| `solbench`, `sol-lingbot`, `sol-cosmos3` | 3 to 9 | no | per-arm timeouts, exported settings, ungated routers, prompts read from files | no | stay | D1e |
| `hunyuan`, `sana-video` | 1 to 2 | no | a `--help` probe that ends the family | no | no tier | **retired, no tier** (§18): nothing runs them (no variant or gold file serves them) |

`kernels` is not a family. It is a `runpod-http.sh` mode (`kernels`, `all`) that runs `fv-gpucheck kernels` before or
instead of the family (ADR-0019 §7: not in fv-lab). Its in-family uses (`kernels-wan`, `kernels-fuse`,
`attn-fp8-kernels`) are ungated `run_cell`s. An arm may have no weight gate since §15 (item 6), and `kernels-fuse`
moved with `fuse` as the ungated arm of the cell `fuse-kernels` (D1f, §17); `kernels-wan` stays with its family
(`wan`), which needs more (above), and `attn-fp8-kernels` went with `h3attn` (§18).

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
   `determinism` verdicts become new typed checks under `[verdict]` (§11.1) with D1c / D1d. Done in §15: a `[repeats]`
   table beside `[verdict]` rather than under it (§15.3 says why), the oracle's tensor-diff targets as a tier.

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
4. **The R2 key of `volume+r2` pods** (§16.5): the serve containers' `fv_r2_*` Runpod secrets today (no new secret, no
   owner action, UNVERIFIED for `fv-weights`). Least privilege would be a dedicated read-only pair scoped to
   `fv-weights` (the key `r2-fill-volume.sh` already uses) as two new Runpod secrets, and the two names in
   `weights::R2_KEY_ENV`; the owner's call.

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

§15 (D1d, second slice), CPU: `tests/repeats_oracle.rs` (the `det-short` and `determinism` branches, frozen at
`0f63c56b` since §15.10 with a test that their dry runs equal the frozen goldens, the live `oracle` branch, and
`fv-lab run` on one fake box: every output file byte-identical, `fv-lab gate --check` clean, the typed verdicts),
`tests/{dry_run,payload_golden}.rs` with the three tiers' fixtures, `plan::tests::references_ungated_arms_prompt_files_and_repeats`,
`tests/fv_lab_bin.rs::the_oracle_pod_needs_its_references_and_the_d1d_knobs_are_refused`, `fv-lab-goldens.sh --check`
(with compare-runs' hash and dump-line cases), `fv-lab-route.test.sh` (§15.10's routing and refusals); GPU: §15.7
(passed on `f93a961`).

§17 (D1f), CPU: `tests/dry_run.rs` and `tests/payload_golden.rs` with the `fuse`, `attn3` and `ltx` fixtures (written
from the live branches by `fv-lab-goldens.sh`), `tests/lab_files.rs` (the trees the volume lacks: `h3-8step`,
`fastwan22-ti2v-5b`, `ltx23`, `ltx2`), `knobs::tests::the_retired_attn3_knobs_are_not_read` (`FV_ATTN3_*`, in
`MODE_KNOBS` since §17.6), `tests/knobs.rs`, `tests/schema.rs` (`model`'s description), `fv-lab-goldens.sh --check`
(the three fixtures frozen since §17.6), `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`,
`fv-lab-route.test.sh` (§17.6's routing and refusals); GPU: §17.5 (passed on `762ce4d`).

§13 (D1c), CPU: `tests/report.rs` (every fixture of `fixtures/report/`, byte for byte), `tests/fetch_tree.rs` (the
script's `fetch_tree` on a fake pod), `tests/fv_lab_bin.rs::report_summary_and_table_print_the_scripts_output`,
`::the_tree_fetch_is_checked_before_any_pod_exists`, `bash scripts/gpu/tests/fv-lab-report-goldens.sh --check`,
`fv-lab-route.test.sh`; no GPU.

§14 (D1e), CPU: `tests/experiment_tiers.rs` (the `h3-graphs` arms equal the pod script's turbo commands under the §14.3
mapping; its dry run equals its golden; the compares), `bash scripts/gpu/tests/fv-lab-experiment-goldens.sh --check`
(the script's commands and prompt sets), `tests/dry_run.rs`, `tests/payload_golden.rs`, `fv-lab-goldens.sh --check`,
`fv-lab-route.test.sh`, `self-delete.test.sh`; GPU: §14.4 (passed on `f93a961`; the experiment goldens read the
deleted pod script from the history since).

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

### 14. D1e: the port one-offs, kept as a tier or deleted

Status: accepted 2026-10-26. The kept tier's GPU check (§14.4) passed on `f93a961`, and
`scripts/gpu/experiments/` is gone (§14.4, "Result"). Decision-id
FVID-2026-10-26-fv-lab-one-offs. §8 left D1e two things: the port one-offs F1 moved to `scripts/gpu/experiments/`
(ten experiments: twenty pod / launcher scripts, two `*-plan.py`; their four `*-summary.py` went in D1c, §13.5), and
`runpod-http.sh`, deleted once nothing calls it.

#### 14.1 The rule

An experiment is **kept** as fv-lab cells and a tier only when it still measures something a gold file
(`lab/gold/`), a served variant (`families/*.toml`) or an open item depends on, and fv-lab can run it: every tree on
the EU volume (`weights = "volume"`, §4; the registry's `hydrate` is not there yet), no upstream Python stack (that
is the oracle's, D1d), no step the cell schema lacks. Otherwise it is **deleted**: a finished experiment whose result
is in `docs/ports/*.md` / `docs/perf/*.md`, its evidence in `artifacts/` (both stay). What the deleted ones measured
for the served variants is re-measured by other means: latency by the `gold` tier (one serve-engine cell per gold
file, ADR-0023 §3, once its trees are reachable), byte identity of a change by `techniques` / `b6` / `b4a`
(baseline vs this build), kernels by `fv-gpucheck kernels`, upstream parity by the oracle (D1d). A deleted
script stays readable in git: `git show 22a5fd7b:scripts/gpu/experiments/<file>`.

#### 14.2 The decisions

| experiment (scripts) | what it measured; where it is recorded | trees / stack | decision, and why |
|---|---|---|---|
| `fasth3-nvfp4` (`-validate.sh`, `-pod.sh`, `-plan.py`) | FastH3 V2 / 4-step NVFP4 apples to apples with upstream FastVideo on one pod: upstream's timings, parity on injected noise, our NVFP4 arms, phase profile (2026-10-14). `docs/ports/fasth3-nvfp4.md` §4-§8, `artifacts/ports/fasth3-nvfp4/run-2026-10-14/`, `fv-lab report summary --kind fasth3-nvfp4` | NVFP4 trees R2-only; the upstream image `fastvideo-rs-upstream-fastvideo-nvfp4` | **delete**: finished port proof; its recommendation (sparse FP4 attention) was carried out (`fasth3-sparse-fp4-attn.md`). Upstream parity is the oracle's (D1d); the served variants' latency is the `gold` tier's (`gold-fasth3-v2-nvfp4`, `gold-fasth3-4step-vsa-nvfp4`) |
| `h3-fp4attn` (`-validate.sh`, `-pod.sh`) | sparse FP4 VSA attention against bf16 VSA on the NVFP4 checkpoints: kernel checks, 768p / 480p traces, parity, timings (2026-10-14). `docs/ports/fasth3-sparse-fp4-attn.md` §3, `artifacts/ports/fasth3-sparse-fp4-attn/` | NVFP4 trees R2-only | **delete**: the owner decided the default on these numbers (2026-10-14); the kernel checks are `fv-gpucheck kernels --groups h3_fp4_attn` (`runpod-http.sh kernels`); the served NVFP4 tier is the `gold` tier's. The five-prompt gate it did not run is a measurement of a served variant, i.e. a gold run, not a reason to keep a one-off. Its numbers (`fasth3-sparse-fp4-attn.md` §3.3-§3.4) are on R2-only trees: until PR #201 (`claude/lab-r2-weights`, the gold tier on `volume+r2` weights) is merged and the gold run is done, the reproduction of these numbers is `git show 22a5fd7b:scripts/gpu/experiments/h3-fp4attn-{validate,pod}.sh` (the doc says so) |
| `h3-glue` (`-validate.sh`, `-pod.sh`) | the H3 block glue fusions: base vs new binary, `FASTVIDEO_H3_GLUE=0`, kernel checks, byte-identical clips (2026-10-14). `docs/perf/h3-glue-fusion.md`, `artifacts/perf/h3-glue-fusion/` | V2-NVFP4 R2-only; h3-turbo on the volume | **delete**: merged and proven bit-exact; §4 there closes the remaining candidates (under the 2 % bar). A later change to the block is a `techniques` / `b6` byte-identity run (baseline vs this build), kernels `fv-gpucheck kernels --groups dit_fusion,nvfp4_linear` |
| `h3-graphs` (`-validate.sh`, `-pod.sh`) | lazy step events (default) and the opt-in CUDA graphs of the H3 denoise steps: CUPTI traces, sync / lazy / graphs timings, byte identity over 72 clips (2026-10-14, V2-NVFP4 and 4-step NVFP4). `docs/perf/h3-denoise-graphs.md`, `artifacts/perf/h3-denoise-graphs/run-2026-10-14/`, `fv-lab report summary --kind h3-graphs` | NVFP4 trees R2-only; **h3-turbo all on the EU volume** | **keep, as the tier `h3-graphs` (h3-turbo cells only)**: §5 there leaves an open item, "h3-turbo (MXFP8, 768p) was not measured" (no EU-RO-1 stock then), on the default served H3 tier (`fasth3-4step-vsa`, served `h3-turbo`), whose lazy step clock is the default; the graph cache's `MAX_KEYS` was "set after this run and not re-measured". Its NVFP4 cells stay out until `hydrate` (§14.3) |
| `h3-tier-options` (`-validate.sh`, `-pod.sh`) | three options for the H3 NVFP4 tier (TAEH3, the published NVFP4 VAE, Trim NVFP4 ± FP4 attention), five prompts, compares and gates (2026-10-14). `docs/perf/h3-nvfp4-tier-options.md`, `artifacts/perf/h3-nvfp4-tier-options/`, `report summary --kind h3-tier-options` | V2-NVFP4, Trim, Trim NVFP4 R2-only | **delete**: the owner decided on these numbers and they shipped ("Decisions (2026-10-14)"). Six gold files take latency or quality numbers from the doc (`fasth3-v2-nvfp4{,-taeh3}`, `fasth3-trim-nvfp4{,-fast}`, `fasth3-4step-vsa{,-480p}-taeh3`; 32 cite its price): their latency is what the `gold` tier re-measures, their `quality.*` values are dated sources (ADR-0009), and no open decision rests on re-running the comparison. Their trees are R2-only: the `gold` tier on the EU volume skips them. Until PR #201 (`claude/lab-r2-weights`, the gold tier on `volume+r2` weights) is merged and the gold run is done, the reproduction of these numbers is `git show 22a5fd7b:scripts/gpu/experiments/h3-tier-options-{validate,pod}.sh` |
| `h3-tier-ship` (`-validate.sh`, `-pod.sh`) | smoke of what shipped: main vs branch, the NVFP4 VAE default, `FASTVIDEO_H3_VAE_NVFP4=0` byte-identical to main, preview, Trim fastest (2026-10-14). `docs/perf/h3-nvfp4-tier-options.md` "GPU smoke", `artifacts/perf/h3-nvfp4-tier-ship/` | R2-only | **delete**: a one-time smoke of a merged change; the served variants are the `gold` tier's. `fasth3-trim-nvfp4-fast`'s latency comes from this smoke (its gold file cites the doc's "GPU smoke" table, `artifacts/perf/h3-nvfp4-tier-ship/`); its trees are R2-only. Until PR #201 (`claude/lab-r2-weights`, the gold tier on `volume+r2` weights) is merged and the gold run is done, the reproduction of these numbers is `git show 22a5fd7b:scripts/gpu/experiments/h3-tier-ship-{validate,pod}.sh` |
| `h3-trim` (`-validate.sh`, `-pod.sh`) | the Trim port against upstream FastVideo (injected parity, `av_psnr.py`) and its timing against V2 (2026-10-13). `docs/ports/h3-trim.md` §6, `artifacts/ports/h3-trim/` | `fasth3-trim-8step`, `h3-8step` R2-only; an upstream uv stack | **delete**: finished port proof; parity is the oracle's (D1d), `gold-fasth3-trim-8step` the served variant's latency. Its §7 open item "MXFP8 versus bf16 speed would decide the default" was measured on one pod by `h3-trim-nvfp4` (§5.4 there: MXFP8 1.30x / 1.27x faster, gate pass); `h3-trim.md` §7 now says so |
| `h3-trim-nvfp4` (`-validate.sh`, `-pod.sh`, `-plan.py`) | the published Trim NVFP4 file against bf16 / MXFP8: kernel check, parity, speed, compares and gates (2026-10-14). `docs/ports/h3-trim.md` §5.4, `artifacts/ports/h3-trim/nvfp4/`, `report summary --kind h3-trim-nvfp4` | R2-only | **delete**: finished; served opt-in (`fasth3-trim-nvfp4`, its gold file cites the tier-options run) |
| `h3-turbo-loras` (`-validate.sh`, `-pod.sh`) | larry / lightx2v LoRAs against the Space's own stack (merged-weight and injected parity), speed and quality (2026-10-14, two runs). `docs/ports/h3-turbo-loras.md` §6-§7, `artifacts/ports/h3-turbo-loras/` | LoRA trees R2-only; the Space's Python stack | **delete**: finished port proof; parity is the oracle's (D1d), the three served variants' latency the `gold` tier's |
| `vdn-h3` (`-validate.sh`, `-pod.sh`) | VDN-H3 (50-step and 8-step) against the upstream diffusers component (`oracle_vdn.py`), kernel checks, timings (2026-10-14). `docs/ports/vdn-h3.md` §5, §6, §8, `artifacts/ports/vdn-h3/` | VDN trees R2-only; an upstream uv stack | **delete**: finished port proof; `kernels --groups vdn` stays in fv-gpucheck, parity is the oracle's (D1d), `gold-vdn-h3` / `gold-vdn-h3-8step` the latency. Its trees are R2-only: until PR #201 (`claude/lab-r2-weights`, the gold tier on `volume+r2` weights) is merged and the gold run is done, the reproduction of these numbers is `git show 22a5fd7b:scripts/gpu/experiments/vdn-h3-{validate,pod}.sh` (`vdn-h3.md` §7 says so) |

One kept (two scripts, deleted once §14.4 passed), nine deleted (eighteen scripts and two `*-plan.py`).

#### 14.3 `h3-graphs` as a tier

`lab/tiers.toml` `h3-graphs` (the `rtx6000` tier's pod: PRO 6000, EU volume, 4 h cap, 3600 s per arm, the TAE fetch)
with two cells, in the pod script's order:

| cell | arm | env | what |
|---|---|---|---|
| `h3-graphs-turbo-trace` | `h3-graphs-turbo-trace-768p` | `FASTVIDEO_H3_GRAPHS=1 FASTVIDEO_GPU_TRACE=1` | the script's `trace` step for `turbo 768p`: a spark-mountain-lake warm-up request, then h3-demo eager / captured / replayed, one traced step each (no `--warm`) |
| `h3-graphs-turbo` | `h3-graphs-turbo-sync` | `FASTVIDEO_H3_STEP_SYNC=1` | the `cells` step, arm `sync`: a synchronize per step |
| | `h3-graphs-turbo-lazy` | none (`env = []`) | arm `lazy`: the default |
| | `h3-graphs-turbo-graphs` | `FASTVIDEO_H3_GRAPHS=1` | arm `graphs` |
| | compares | | `lazy -> sync` and `lazy -> graphs`, `--off-identity`, one report per request directory (`compare_cells`), 6 each |

Every arm: `--mode fast --techniques h3/fasth3_4step_vsa --out <arm>/report h3 gen --weights {weights}/h3-base
--h3-recipe 4step-vsa --adaln-cache … --height 768 --width 1344 --text-encoder streamed --text-weights
{weights}/h3-base --text-cache … --no-mp4 --num-frames 124 --prompt unused --prompts <set> --clip-dir <arm>/frames`,
the three timed arms with `--warm`; gate `fasth3-4step-vsa` (h3-base, its `transformer_prequant_4step-vsa_mxfp8`, the
LoRA: the script's `TURBO_OK`), all on `bz0ofkfx45`. The prompt sets are the script's own prep output:
`scripts/gpu/prompts-h3-graphs-768.json` (h3-demo and spark-mountain-lake at 768p, three requests each, seed 1024)
and `scripts/gpu/prompts-h3-graphs-trace-768p.json`, both from `scripts/gpu/prompts-hd5.json`.

**Equality, from the pod script itself.** The script cannot dry-run, so `scripts/gpu/tests/fv-lab-experiment-goldens.sh
[--check]` (CI: `control.yml`) cuts its `ARMS`, `CKS`, `COMMON`, `arm_env`, `ck_args`, `gen`, the prep Python and the
`trace` / `cells` blocks out of `h3-graphs-pod.sh` and runs them under `env -i` with `env(1)` stood in by a stub that
records each process's directory and argv: `fixtures/experiments/h3-graphs-script.txt`, every process of every
checkpoint (v2, 4s, turbo; 12 lines). It runs the prep Python on `prompts-hd5.json` and holds the two prompt files to
its output byte for byte. `tests/experiment_tiers.rs::h3_graphs_runs_the_pod_scripts_commands` takes the script's
four turbo lines and finds each equal to the arm's command word for word after this mapping, and only it:

| the pod script | the tier | why |
|---|---|---|
| `timeout 2400` before the binary | the runner's `timeout … 3600` (the tier's `cell_timeout_s`) | every tier's per-arm cap; 2400 s was the script's own |
| `/srv/cells/turbo-<arm>`, `/srv/trace/turbo-768p` (cwd, `--out …/report`) | `{runs}/h3-graphs-turbo-<arm>`, `{runs}/h3-graphs-turbo-trace-768p` | the run directory |
| `--clip-dir …/clip` | `--clip-dir …/frames` | where the frames record and `compare_cells` look |
| `/fvw/adaln-turbo.cache` | `{runs}/h3-graphs-turbo-adaln.cache` | the script's scratch file, shared by the four processes as there |
| `/fvw/text-cache` | `{scratch}/h3-graphs-text-cache` | likewise |
| `/fvw/prompts-768.json`, `/fvw/prompts-trace-768p.json` | `{root}/scripts/gpu/prompts-h3-graphs-768.json`, `…-trace-768p.json` | the same bytes, committed |

The tier's own dry run is its golden (`fixtures/h3-graphs-dry-run.txt`, `h3_graphs_dry_run_is_its_golden`), frozen from
the start as §12's were; `tests/dry_run.rs` leaves the experiment tiers out of the script families, as it leaves
`gold`. `FV_LAB_EXPERIMENTS` (`scripts/gpu/fv-lab-retired.sh`) lists them: `fv-lab-goldens.sh` writes only their pod
body (`runpod-http-h3-graphs.json`, held by `tests/payload_golden.rs` like every tier's) and checks that each is a
tier, not retired and without a `runpod-matrix.sh` branch; `runpod-http.sh run` / `all` refuse them and name the
fv-lab command, as for `gold` (`fv-lab-route.test.sh`). When the pod script is deleted (§14.4), the goldens script
reads it from the parent of the deleting commit, and keeps the fixtures in a shallow clone.

**Differences from the experiment, and why.**

| | the experiment | the tier | why |
|---|---|---|---|
| checkpoints | `v2`, `4s` (NVFP4, from R2, composed folders with Hub configs) and `turbo` | `turbo` only | the NVFP4 trees are R2-only and the folders are built at run time; they come back as cells with `hydrate` (§4), `--h3-recipe v2-nvfp4` on the tree as `h3-tier-ship` ran it |
| canvases | V2 / 4-step at 480p and 768p | 768p | h3-turbo ran 768p only in the script too |
| the `--warm` cold clip | hashed with the requests (`cold` in `hashes.tsv`) | not compared: `compare_cells` skips the `cold` / `warmup` directories; its frames are in each arm's `frames.sha256.json` | the runner's rule for the untimed pass; the cold request is h3-demo `a`'s twin, which is compared |
| byte identity | sha256 of each request's PNGs and WAV (`hashes.tsv`), then the PNGs deleted | `--off-identity` compares (pixels, frame count) per request, and each arm's `frames.sha256.json` | the runner's identity tools; the WAV is not compared (no fv-lab tool reads audio); the frames stay on the pod's 120 GB disk (about 8-10 GB) |
| the pod | its own launcher: uv Python, a presigned R2 plan PUT to the pod, 160 GB disk, `nvsmi.csv` sampled every second, a busy-GPU self-delete | fv-lab's (§6): the runtime image, the EU volume, `nvidia-smi` at each arm's start and end, an empty GPU checked before each arm | fv-deploy's guards; the peak is `benchmark.json`'s device maximum |
| report | `fv-lab report summary --kind h3-graphs` on `cells/`, `trace/`, `hashes.tsv` | the run directory's layout (§5); the timings in each arm's `benchmark.json`, the graph lines (`h3 graphs: N steps: R replayed, C captured, E eager`) in its `stderr.log` | the summary reads the experiment's layout; a tier kind for it is a later `report` change |

#### 14.4 The GPU check (the coordinator's)

On one commit whose runtime image is published (`fastvideo-rs-runtime:sha-<7>`: this PR's head or its merge; the
image carries `lab/cells/h3-graphs-*` and `scripts/gpu/prompts-h3-graphs-*.json`), from the repository root with
`fv-lab` built on the build pod:

```sh
SHA=<7>
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family h3-graphs --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '/frames/|\.cache$' --yes --wait
#   -> artifacts/runpod/h3-graphs/$SHA-<MMDDHHMMSS>-<hex>/   (the pod is deleted at the end)
R=artifacts/runpod/h3-graphs/$SHA-<MMDDHHMMSS>-<hex>
for a in h3-graphs-turbo-trace-768p h3-graphs-turbo-sync h3-graphs-turbo-lazy h3-graphs-turbo-graphs; do
  jq -e '.exit == 0' "$R/$a/summary.json" >/dev/null || { echo "FAIL $a"; exit 1; }; done
n=$(ls "$R"/compare/compare-clips-h3-graphs-turbo-lazy--h3-graphs-turbo-{sync,graphs}-*.json | wc -l)
[ "$n" -eq 12 ] || { echo "FAIL: $n compare reports, want 12"; exit 1; }
for f in "$R"/compare/compare-clips-h3-graphs-turbo-lazy--*.json; do
  jq -e '.status != "skipped" and ([.checks[] | select(.name | startswith("off_identity")) | .ok] | length == 2 and all)' \
    "$f" >/dev/null || { echo "FAIL $f"; exit 1; }
done
grep -qE 'h3 graphs: [0-9]+ steps: [1-9][0-9]* replayed' "$R/h3-graphs-turbo-graphs/stderr.log" \
  || { echo "FAIL: the graphs arm replayed no request"; exit 1; }
echo PASS
```

Pass: `pod up --wait` exits 0 (`DONE`, `matrix_exit=0`) and the checks print `PASS`: the four arms exit 0, the twelve compare reports exist and
are byte-identical (sync = lazy = graphs on every request, as on the NVFP4 checkpoints), and the graphs arm replayed
at least one request (h3-demo's third, spark's third). Record in `docs/perf/h3-denoise-graphs.md` §5 the h3-turbo
768p denoise per arm (each arm's `benchmark.json`) against the expectation there (lazy about 1.4 % under sync, a
replay 0.6-0.7 % more) and the graph pool high water; then delete `h3-graphs-{validate,pod}.sh` and
`scripts/gpu/experiments/` with them (the goldens script then reads the history). The same commit must drop
`scripts/gpu/experiments/h3-graphs-validate.sh` from `LOCAL` in `scripts/gpu/tests/self-delete.test.sh` and lower
its launcher floor from 3 to 2 (`family-pool-validate.sh` and `ltx-pod.sh` are left), or that test fails ("LOCAL
names …h3-graphs-validate.sh", "only 2 launchers"). A failing compare is a finding
about the lazy clock or the graphs on h3-turbo, not a run failure: report it and keep the scripts.

Cost: one RTX PRO 6000 Blackwell Server Edition in EU-RO-1 on `bz0ofkfx45` at about $2.49/hr. Boot and preflight
~5 min; the trace arm streams the bf16 text encoder for its two prompts (up to ~4 min each from the volume, as
`h3-tier-options` measured) and makes four 768p requests; each timed arm loads (~16 s) and makes seven (~28 s each,
`docs/ports/h3-turbo-loras.md` §7); twelve frame compares: **~35 min, about $1.45**, bounded at $2.49 by
`--wall-cap-s 3600`. No other pod, no new tree, no download.

**Result (2026-10-26): pass.** `SHA=f93a961` (image `fastvideo-rs-runtime:sha-f93a961`, build id
`7431ed604fd282ff`), `fv-lab pod up --sha f93a961 --family h3-graphs --wall-cap-s 3600 --fetch-tree --fetch-skip
'/frames/|\.cache$' --yes --wait` (pod `8zx5w6h0766g35`, RTX PRO 6000, EU-RO-1, $2.49/hr; `matrix start` 04:38:34 to
`matrix done` 05:04:01, about 25 min of arms) → `artifacts/runpod/h3-graphs/f93a961-1026043809-da7b` (`DONE`,
`matrix_exit=0`). The checks above print `PASS` (`artifacts/runpod/h3-graphs/check-f93a961.txt`): the four arms exit
0, the twelve compare reports exist and are byte-identical (sync = lazy = graphs on every request), and the graphs
arm replayed 4 of 4 steps on three requests (h3-demo `b` and `c`, spark `c`; h3-demo `a` and spark `b` captured,
spark `a` and the cold request ran eager, a new key's first sight). The h3-turbo 768p denoise (each arm's
`benchmark.json` `denoise_s`, the median of its six timed requests): sync 17.98 s, lazy 17.88 s (−0.56 %), graphs
17.93 s (−0.30 % vs sync); per replayed request graphs is 0.19-0.21 % under lazy (0.69-0.79 % under sync); the
graph pool's high water 10.34 GiB (`stderr.log`: `graph pool high 10.34 GiB` at every capture). Against
`docs/perf/h3-denoise-graphs.md` §5's expectation (lazy about 1.4 % under sync, a replay 0.6-0.7 % more), both
gains are about a third of the NVFP4 cells' on h3-turbo, and neither changes a bit; §5 there records it, and its
verdicts stand (lazy default, graphs opt-in).

Done in that PR, as above: `scripts/gpu/experiments/{h3-graphs-validate.sh,h3-graphs-pod.sh,README.md}` deleted (the
directory with them); `self-delete.test.sh` drops the launcher from `LOCAL` and its launcher floor is 2
(`family-pool-validate.sh`, `ltx-pod.sh`); `fv-lab-experiment-goldens.sh` reads `h3-graphs-pod.sh` and its input `prompts-hd5.json`
from the parent of the deleting commit (`source_of`), so a later edit of the live prompts file cannot pair with the
old script, and in CI's shallow clone keeps the fixtures as they are (so both agree); `control.yml` drops its
`scripts/gpu/experiments/**` path filters. The experiment's own
NVFP4 cells stay readable in git (`git show <deleting commit>^:scripts/gpu/experiments/h3-graphs-{validate,pod}.sh`,
`docs/perf/h3-denoise-graphs.md` §6).

#### 14.5 Deleted, and the references

Deleted: `scripts/gpu/experiments/{fasth3-nvfp4,h3-fp4attn,h3-glue,h3-tier-options,h3-tier-ship,h3-trim-nvfp4,h3-trim,
h3-turbo-loras,vdn-h3}-{validate,pod}.sh`, `{fasth3-nvfp4,h3-trim-nvfp4}-plan.py`. References fixed: each doc that
named one says it was deleted here and where it is in git (`docs/perf/{h3-glue-fusion,h3-nvfp4-tier-options}.md`,
`docs/ports/{fasth3-nvfp4,fasth3-sparse-fp4-attn,h3-trim,h3-turbo-loras,vdn-h3}.md`, `docs/ops/r2-weights.md`,
ADR-0023 §4), `docs/perf/h3-denoise-graphs.md` §5-§6 points at the tier, `README.md` and
`scripts/gpu/experiments/README.md` describe the one left, `scripts/gpu/tests/self-delete.test.sh` drops the nine
launchers (its floors now 34 inline copies and 3 launchers), `docs/ports/CHECKLIST.md` sends a port's GPU runs to
fv-lab cells. No Taskfile task or workflow named a deleted file. The committed run directories, the
`report summary --kind fasth3-nvfp4|h3-tier-options|h3-trim-nvfp4|h3-graphs` ports and their fixtures stay (they
read `artifacts/`). `artifacts/perf/h3-trim-vae-mxfp8/driver/launch.sh` names the Trim scripts; it is evidence and
stays as committed.

Left without a caller by the deletion, and **not** deleted here (the upstream references are the oracle's, D1d):
`scripts/gpu/upstream/{oracle_vdn,bench_fasth3_nvfp4,h3_turbo_lora_ref,merged_parity,av_psnr}.py` (the parity
references of `vdn-h3`, `fasth3-nvfp4`, `h3-turbo-loras`, `h3-trim`) and `docker/upstream.Dockerfile`'s
`fastvideo-nvfp4` target (built on dispatch only by `upstream-images.yml`).

#### 14.6 `runpod-http.sh`: what still calls it

Not deleted: it still has callers. At this commit:

| mode | callers | after |
|---|---|---|
| `run` (script path) | every family with a `runpod-matrix.sh` branch: `mmaudio`, `speechtest`, `sfwan`, `sfstream`, `sfquality`, `headline`, `cold`, `wan`, `eval`, `oracle`, `ltxoffload`, `solbench`, `hd`, `serve-engine`, `sol-lingbot`, `sol-cosmos3` (16; `determinism` and `det-short` left with §15.10, `fuse`, `attn3` and `ltx` with §17.6, and eleven retired with no tier with §18: `h3`, `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`, `identity`, `b4b`, `b4c`, `b4c2`, `h3attn`); `oracle.sh` (`FV_FAMILY=oracle`, `FV_ORACLE_RUN_MODE`, `FV_KEEP_POD`, `FV_POD_FILE`); `FV_POD_FILES` / `FV_POD_SCRIPT=sol-bench-pod.sh` (`docs/perf/sol-bench.md`; `hd-upscaler.sh` ships by it) | goes with `runpod-matrix.sh` (each family a tier or retired) and the sol-bench pod script |
| `run` (fv-lab route, §10) | the 18 retired tiers (13, then `det-short` and `determinism`, §15.10, then `fuse`, `attn3` and `ltx`, §17.6); the Taskfile's `rtx6000` task | the Taskfile calls `fv-lab pod up` directly |
| `run` / `all` refused | `gold`, the experiment tiers (§14.3); `run` / `all` / `plan` (exit 2) the families retired with no tier (§18) | — |
| `plan` | `fv-lab-goldens.sh` (every script family's, `gold`'s and `h3-graphs`'s pod body: `fixtures/runpod-http-*.json`), `fv-lab-restored-goldens.sh` (at `5987e6a4`), the route test | the bodies freeze as the families' fixtures did (§10 item 3) |
| `kernels`, `all` | by hand (`README.md`; `kernels.rs`'s `FV_EXTRA_ENV` groups) | an fv-lab kernels tier once an arm may have no weight gate (§9.1) |
| `weights` | by hand | `fv-lab pod up` of any tier writes `weights.log` |
| `upstream` | `oracle.sh` (the reference pods, `UP_IMAGE_TARGET`), `scripts/gpu/upstream/pod.sh` (its start command) | moves with D1d's `oracle` |
| `attach`, `status`, `down` | `oracle.sh` (`down` of its reference pods); by hand | `fv-lab pod status` / `down` for fv-lab pods |

Its functions are read by tests: `tests/fetch_tree.rs` (cuts `proxy` and `fetch_tree`), `tests/knobs.rs`
(`FV_LAB_NOT_READ`), `self-delete.test.sh` (its start command's self-delete), `fv-lab-route.test.sh`; its usage text by
the route test. The scripts that only name it in a message: `runpod.sh`, `runpod-matrix.sh`. It is deleted when the
last row above has moved; the tests that cut it then freeze their copy, as `tests/verdict.rs` froze `b4a`'s branch.

#### 14.7 The `runpod-matrix.sh` families §9.1 marks "D1e or retired"

They are script families, not port one-offs. D1e left them in `runpod-matrix.sh` (D1d edited that file in
parallel, and a branch is deleted only with the remaining families' dry runs compared before and after, §10 item 1)
and proposed, for the slice that retires the script, what follows. §18 (D1g) decided `hunyuan`, `sana-video`, `b200`,
`h3attn`, `writer`, `precision-debug` and `identity`: all retired with no tier, their branches deleted; the other rows
are still proposals:

| family | what keeps it from a tier today | proposal |
|---|---|---|
| `hunyuan`, `sana-video` | a `--help` probe that ends the family; trees not on the EU volume | retire: the gold files and served variants decide what runs |
| `b200` | the US volume is gone; TAE / Spark probes | retire with the US volume (CLAUDE.md) |
| `mmaudio`, `speechtest` | Python venvs, upstream clones, Whisper | the oracle's (D1d) or retired |
| `sfstream`, `sfquality` | `sfwan` trees R2-only; TAEHV fetch; run lists from the environment | a tier with `hydrate` |
| `h3attn` | one process runs many `--arm`s; FP8 arms dropped on a kernel FAIL | retire unless an attention arm is an open item (`docs/techniques.md`) |
| `writer` | sysinfo, a sampler, `dd` throughput | retire (a one-time measurement, `docs/perf`) |
| `precision-debug`, `identity` | dump-dependent compares, cases from the environment | retire; `precision` and the identity tiers cover them |
| `solbench`, `sol-lingbot`, `sol-cosmos3` | per-arm timeouts, routers, prompts from files; trees not on the volume | tiers with `hydrate` and a prompt-file arm, or retired |

### 15. D1d, second slice: `det-short`, `determinism` and `oracle` as tiers

Status: accepted 2026-10-26. The three GPU checks (§15.7) passed on `f93a961`; `det-short` and `determinism` are
retired from the script (in `FV_LAB_RETIRED`, branches deleted, fixtures frozen), and `oracle`'s branch stays whole,
its tier the tensor-diff part's equal, until the `fv-lab oracle` driver and the clip-metric cells exist (§15.10).
No script branch was deleted in the PR that made the tiers: each family left `runpod-matrix.sh` only after its own
check through both runners (§8). Decision-id FVID-2026-10-26-d1d-oracle-determinism.

#### 15.1 What moved

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

#### 15.2 Cell schema additions (v1, additive, `deny_unknown_fields` kept)

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

#### 15.3 The typed verdict, and why it writes no file

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

#### 15.4 Equality

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

#### 15.5 Differences from the script, and why

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

#### 15.6 What stays in the script, and when it moves

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
  environment (`FV_ID_CASES`). D1e (§14.7) proposes retiring them; this slice does not change that. (Both retired
  with no tier in §18.)

#### 15.7 The GPU checks (the coordinator's)

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
bash scripts/gpu/fv-lab-compare-runs.sh --expect-verdicts 5 $L $S     # determinism.txt, hashes.txt, 3 judged hashes.json
./fv-lab gate $L --check --out /tmp/regate-ds-lab
./fv-lab gate $S --check --out /tmp/regate-ds-script --cells h3-r1,h3-r2,h3-r3,h3-timed

# determinism, the h3 model: two runs in one process, one in another, the timed control (1088x1920 5 s)
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family determinism --cells h3-ab,h3-c,h3-timed --yes --wait
# (FV_DET_CELLS is space separated; runpod-http.sh writes FV_EXTRA_ENV into the start command as shell text,
#  so the inner quotes hold the list together, as oracle.sh's FV_ORACLE_URL='…' does)
RUNPOD_API_KEY=… FV_FAMILY=determinism FV_EXTRA_ENV="FV_DET_CELLS='h3-ab h3-c h3-timed'" \
  FV_FETCH_TREE=1 FV_FETCH_SKIP='/frames/' bash scripts/gpu/runpod-http.sh run $SHA
L=artifacts/runpod/determinism/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/determinism/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh --expect-verdicts 3 $L $S     # determinism.txt, 2 judged hashes.txt
./fv-lab gate $L --check --out /tmp/regate-det-lab
./fv-lab gate $S --check --out /tmp/regate-det-script --cells h3-ab,h3-c,h3-timed

# oracle, fasth3-4step-vsa: one upstream pod makes the reference, both runtime sides fetch it
# (FV_MIN_RAM_GB=192: the reference's LoRA conversion needs host RAM a default pod may not get;
#  docs/oracle.md "Upstream cells: host RAM")
RUNPOD_API_KEY=… FV_MIN_RAM_GB=192 FV_KEEP_POD=1 FV_POD_FILE=/tmp/up.pod UP_IMAGE_TARGET=fastvideo UP_IMAGE_TAG=latest \
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

**Control arms in the compare** (added after the first `det-short` check, run `f93a961`): the timed-pick arm
(`FASTVIDEO_SDPA_AUTO=timed`) is not reproducible across processes (docs/perf/determinism.md), and it differed
between the two sides of that run while r1-r3 were byte-identical and both `fv-lab gate --check` passed. So
`fv-lab-compare-runs.sh` treats every arm a cell lists in `[repeats] controls` as fv-lab's verdict does: reported, not
judged. It reads the controls from the checkout's `lab/cells/*.toml` (`--cells DIR` for another directory; a name
another cell declares as an arm it does not list as a control stays judged; a name outside fv-lab's arm-name grammar
exits 2). A control arm must still exist in both runs, agree on `summary.json` `exit` / `skipped`, and have the same
frame count in both (the timed pick changes bytes, not counts); its frame bytes, `compare-dumps` lines and own
`hashes.txt`, `hashes.json` and evidence logs are printed on a `control <arm>:` line and are neither differences nor
verdict files. The run's `determinism.txt` and `hashes.txt` are compared without the control arms' lines and still
count, unless only control lines are left in both (then neither compared nor counted); an unreadable one is a
difference. Hence 5 and 3 for `--expect-verdicts` above, not 6 and 4.

Pass, for each family: `fv-lab-compare-runs.sh` exits 0 and both pods end `matrix_exit=0`:

- `det-short`: 3 arms the same (3 ran, frames byte-identical, about 124 frames each), 1 control arm reported
  (`h3-timed`), 5 verdict files the same, and both `fv-lab gate --check` with 0 differences. The verdict (`repeats det-short-h3`) is pass on both when r1-r3 are
  identical (the 2026-10-06 run's were); a fail on both sides is no runner difference.
- `determinism`: 2 arms the same, 1 control arm reported (`h3-timed`), 3 verdict files the same, both `--check`
  clean.
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

**Results (2026-10-26): all three pass.** `SHA=f93a961` on both sides of each check (image
`fastvideo-rs-runtime:sha-f93a961`, build id `7431ed604fd282ff`), RTX PRO 6000 in EU-RO-1 on `bz0ofkfx45`, the
commands above. Each compare and `fv-lab gate --check` was run again from the committed run directories (fv-lab
built at `f93a961`; `fv-lab-compare-runs.sh` of main after #204) and its output saved next to them.

- **`det-short`** (h3 group): fv-lab `artifacts/runpod/det-short/f93a961-1026045808-a9a4` (pod `s9zflnm1e3nszd`),
  script `artifacts/runpod/det-short/f93a961-10260508`; both `matrix_exit=0`. `fv-lab-compare-runs.sh
  --expect-verdicts 5` exit 0 (`det-short/compare-f93a961.txt`): "3 arm(s) the same (3 ran with byte-identical frames,
  …), 1 control arm(s) reported, … 5 verdict file(s) the same, 0 difference(s)"; 124 frames per arm. The control
  `h3-timed` was identical to r1 on the fv-lab side and differed on the script side (frames
  `83266888…` against `4cfba8be…`, the audio the same): the timed pick, reported, not judged (the compare's control
  rule, added after this run's first comparison). `repeats det-short-h3: pass` on both; `fv-lab gate --check` 0
  differences on both (`det-short/gate-f93a961.txt`; the script's run with `--cells h3-r1,h3-r2,h3-r3,h3-timed`).
  The first attempt's pods died during `h3-r3` (2026-10-26; a host failure suspected; fv-lab pod `t37wlzbywqp3re`,
  script pod `0uwxdgiddf6fn2`) and were retried on new pods; its fv-lab stub
  (`det-short/f93a961-1026043809-0c44`, a `pod.json` only) is an infrastructure retry and not committed.
- **`determinism`** (h3 model): fv-lab `artifacts/runpod/determinism/f93a961-1026043809-9e04` (pod
  `r7z1ddrfki2dvv`), script `artifacts/runpod/determinism/f93a961-10260439`; both `matrix_exit=0`.
  `--expect-verdicts 3` exit 0 (`determinism/compare-f93a961.txt`): "2 arm(s) the same (2 ran with byte-identical
  frames, …), 1 control arm(s) reported, … 3 verdict file(s) the same, 0 difference(s)"; `h3-ab` 248 frames (run1,
  run2), `h3-c` 124; the control `h3-timed` byte-identical on both sides this time. `repeats determinism-h3: pass`
  on both; `fv-lab gate --check` 0 differences on both (`determinism/gate-f93a961.txt`).
- **`oracle`** (`fasth3-4step-vsa`): the reference `artifacts/runpod/upstream/55fa19a-10260558` (the `fastvideo`
  upstream image, `UP_STEPS="info:box weights:h3-diffusers oracle:fasth3-4step-vsa"`, `FV_MIN_RAM_GB=192`:
  docs/oracle.md "Upstream cells: host RAM"); fv-lab `artifacts/runpod/oracle/f93a961-1026061114-8c80` (pod
  `70egna7udfp0fr`), script `artifacts/runpod/oracle/f93a961-10260609`; both `matrix_exit=0`, both fetched the
  1.3 G reference (`oracle: fasth3-4step-vsa reference 1.3G`). `fv-lab-compare-runs.sh` exit 0
  (`oracle/compare-f93a961.txt`): "5 arm(s) the same (2 ran with byte-identical frames, …; 3 with the same
  compare-dumps lines), … 0 difference(s)"; 124 frames for the bf16 and the f32 run. The first attempt (runtime
  directories `oracle/f93a961-10260544` and `oracle/f93a961-1026054453-3225`, upstream `upstream/f93a961-10260534`)
  fetched no reference: the upstream worker was OOM-killed converting the LoRA on a pod without enough host RAM,
  which `FV_MIN_RAM_GB=192` then gave (#205); those directories are not committed (as no failed attempt of an
  earlier gate was).

#### 15.8 Then

After a family's check passes, in the next D1 PR: it joins `FV_LAB_RETIRED`, its `runpod-matrix.sh` branch is
deleted (and, for `oracle`, only its moved targets: the clip-metric targets keep a reduced branch until §15.6's step,
or the branch stays whole and the tier is the moved part's equal; to be decided with the result), its fixtures
freeze, `knobs::SCRIPT_KNOBS` joins `FV_LAB_NOT_READ`, and `oracle.sh`'s runtime side passes `FV_CELLS` instead of
`FV_ORACLE_TARGETS`. Done for `det-short` and `determinism`; for `oracle`, decided with the result: the branch stays
whole (§15.10).

#### 15.9 Review of #202

- The oracle check's commands (§15.7) wait on the two runtime sides by PID, and the keepalive loop has its own
  90-minute bound, so the upstream pod is always deleted after them.
- `determinism`'s `|| true` is a real-run change, stated in §15.4, and a reference arm without run1 hashes is reported
  (`<model>-ab: no run1 hashes`) by both runners.
- `FV_ORACLE_URL` is `http(s)://` only: `fv-lab run` and `pod up --extra-env` refuse anything else (a `file://` URL, a
  value that would read as a curl option), and every curl call passes `--proto =http,https --proto-redir =http,https
  --url <u>`. The tests' fake upstream is a `file://` URL through `RunConfig::allow_file_urls`, which no flag or
  variable of the binary sets.
- Validation: a cell with both `[reference]` and `[verdict]` is refused (its verdict would never run), and so are two
  cells of one tier with the same `reference.target` (one `oracle-ref` directory, fetched twice at once). A `fresh`
  path must name something below `{runs}/`: at least one component, none empty, `.` or `..` (`{runs}/` or `{runs}/.`
  would have removed the run directory).
- §15.5 gains the `ORACLE_REFS_DONE` row; the det-short check's script side skips the AdaLN caches
  (`adaln\.cache$`).

#### 15.10 The retirement: `det-short` and `determinism` go, `oracle`'s branch stays whole

Done in the PR after the checks (2026-10-26), as `b4a` was retired (§11.6):

- `det-short` and `determinism` are in `FV_LAB_RETIRED`; their `case` branches are deleted from `runpod-matrix.sh`
  (`determinism`'s `FV_ORACLE_URL` prelude, which ran the `oracle` family first, with it). The 30 remaining families'
  dry runs are unchanged (`FV_DRY_RUN=1` output and `FV_DRY_RUN_GATES` records before and after): 26 byte-identical,
  `identity` differing in a bash error message's line number, `mmaudio` and `speechtest` in an error message naming
  the dry run's temporary directory, `writer` in its `dd` timings.
- Their six fixtures are frozen (`fv-lab-goldens.sh` no longer writes them).
- `knobs.rs`: the five `FV_DET_*` knobs move from `SCRIPT_KNOBS` into `MODE_KNOBS`, so into `not_read()` and
  `runpod-http.sh`'s `FV_LAB_NOT_READ` (`tests/knobs.rs` holds them equal). `runpod-http.sh run` / `plan` route the two
  families to `fv-lab pod up`, refuse `FV_DET_CELLS` (in `FV_EXTRA_ENV` and in the host environment) with a pointer
  to `FV_CELLS`, and refuse `FV_ORACLE_URL` for `determinism` (the script ran the oracle first with it; the tier
  does not, §15.5, so it is refused, not dropped; `fv-lab pod up --extra-env` and `fv-lab run` refuse it for that tier too), and refuse an `FV_EXTRA_ENV` holding a newline (`read` takes its first line only, so later entries, refusals included, were dropped). Every
  other tier passes `FV_ORACLE_URL` on.
- `tests/repeats_oracle.rs` ran the live branches. For `det-short` and `determinism` it runs a frozen copy,
  `crates/fastvideo-lab/fixtures/det-script/`: `runpod-matrix.sh` at main `0f63c56b` cut to its frame, the two
  branches and the fallback branch, every kept line unchanged (`git show 0f63c56b:scripts/gpu/runpod-matrix.sh |
  sed -n '1,454p;2114,2327p;3093,3119p'`), and that commit's `b4-verdict.sh`, `cuda-13.pins` and
  `fv-lab-retired.sh` whole. A new test there runs the frozen copy's dry run of each and finds it equal to the
  frozen `<family>-dry-run.txt`, line for line. The oracle tests still run the live branch.
- `fv-lab-route.test.sh`: both routed by arm, the `FV_DET_*` refused, `FV_ORACLE_URL` refused for `determinism`
  only, `runpod-matrix.sh` refusing both (exit 2), `oracle` still on the script path.

**`oracle`: the branch stays whole** (§15.8's second option; decided with the result and accepted by the coordinator
2026-10-26, FVID-2026-10-26-oracle-branch-stays-whole), out of `FV_LAB_RETIRED`, its fixtures still written
from the live branch, `FV_ORACLE_TARGETS` / `FV_ORACLE_F32` / `FV_ORACLE_OWN_TEXT` / `FV_ORACLE_PREFETCH` /
`FV_REF2VA_PROMPT` still `SCRIPT_KNOBS` (refused by fv-lab only), and `oracle.sh` unchanged. The tier is the
tensor-diff part's equal, held to the live branch by `tests/repeats_oracle.rs` and now by a GPU check. Neither
other reading of §15.8 keeps what the branch does:

- **Retire it whole**: a retired family has no branch (`fv-lab-goldens.sh`, `tests/dry_run.rs`) and
  `runpod-matrix.sh` refuses it, so the clip-metric targets (`ltx25-{i2v,kf,ref2v,a2v,a2v-i2v,a2v-guided,retake,
  retake-v,retake-a,extend}`, which `oracle.sh` runs on request through `ORACLE_TARGETS`) and `sfwan13` (the `sfwan`
  family's prelude calls `runpod-matrix.sh oracle` with `FV_ORACLE_TARGETS=sfwan13`) would have no runner.
- **A reduced branch with the moved targets deleted**: in `FV_LAB_RETIRED` it would be refused all the same; out of
  it, `FV_FAMILY=oracle` stays on the script path, which would then refuse `oracle.sh`'s default targets (all
  moved). `oracle.sh` cannot send its runtime side through fv-lab today: it sets `FV_POD_FILE` (the runtime pod's id
  and tag, which it polls for `ORACLE_REFS_DONE` to release the upstream pods) and `FV_CREATE_WAIT_S` (the stock wait
  after which it deletes them), and `fv_lab_route` refuses both because `fv-lab pod up` has neither; nor `FV_ORACLE_RUN_MODE=all`
  (`all` is refused for an fv-lab family), nor `FV_ORACLE_F32` / `FV_ORACLE_OWN_TEXT` (no tier reads them).

**What has to exist before it retires (open):** the `fv-lab oracle` driver of §15.6 (the upstream pods up as a
second image flavour, the runtime tier with `FV_ORACLE_URL`, the release on `ORACLE_REFS_DONE`, a stock wait: what
`oracle.sh` and §15.7's hand-run commands do), and the clip-metric targets and `sfwan13` as cells (the derived-clip
step, §9.1 / §15.6). Then `oracle` joins `FV_LAB_RETIRED`, its branch goes (the `wan` family keeps
`oracle_fetch` / `oracle_diff`), its fixtures freeze, its `SCRIPT_KNOBS` join `FV_LAB_NOT_READ`, and `oracle.sh` is
the driver or passes `FV_CELLS`.

### 16. Weights by tree from two places: `weights = "volume+r2"`

Status: proposed 2026-10-26 (draft PR from `claude/lab-r2-weights`), CPU-tested; the GPU gate is the full gold run of
§16.7, the coordinator's. Decision-id FVID-2026-10-26-fv-lab-volume-r2.

#### 16.1 Why

Since the owner's 2026-10-13 decision the 500 GB EU volume `bz0ofkfx45` holds only the H3 and LTX trees
(`scripts/gpu/volumes.sh` `FV_EU_VOLUME_TREES`); every other tree is in the R2 weights store (`scripts/gpu/weights-r2.tsv`,
`docs/ops/r2-weights.md`). §4 accepted only `volume`, so the `gold` tier (ADR-0023 §3) recorded about two thirds of its
arms as skipped (`weights incomplete`): Wan, LongLive, the NVFP4, trim, VDN and turbo-LoRA trees. fv-boot and
fv-control already fetch R2 trees for serve pods (`weights_source r2`), but fv-lab's pods are not fv-control pods, so
neither path served them.

#### 16.2 Decision: a new tier mode, not a fallback inside `volume`

A tier's `weights` is now fv-lab's own `TierWeights` (`plan.rs`): `volume` or **`volume+r2`**. `tiers.v1.json` is
regenerated and is not purely additive: it gains `volume+r2` and loses the four values of fv-deploy's `WeightsSource`
(`r2`, `hub`, `none`, `registry`) that it listed but §4's validation always refused, so every file valid before is valid
now and `volume` reads as before. In `volume+r2` each tree an arm names (§4) is classified by
`weights::TreeSources::source`: a tree an EU-volume tree covers (`tree_covers`, so `auxiliary/tae/taeh3.safetensors` is
the volume's `auxiliary/tae`) is read from the volume; else the longest `weights-r2.tsv` dest covering it is fetched from
R2; else it is in neither, nothing is fetched, and the arm's gate records it as skipped (`weights incomplete`), as on a
`volume` tier. Why a mode, not "`volume` falls back to R2":

- **Every other tier stays what its GPU gate measured.** The thirteen `volume` tiers are frozen against the scripts (§10,
  §12); `rtx6000`'s `fasth3-8step` skip is part of that record (`tests/lab_files.rs`). A silent fallback would turn it
  into a 148 GB fetch and change those tiers' dry runs, pods and timings. With a mode their dry-run and payload goldens
  are byte-identical (`fv-lab-goldens.sh --check`, `fv-lab-restored-goldens.sh --check`).
- **The cost is visible in the run plan.** A `volume+r2` tier fetches hundreds of GB and needs a container disk and an R2
  key; a tier says so in one word, and its dry run lists every fetch and drop.
- **The registry's `hydrate` slots in the same way** (§4: `registry` later) as a third value.

#### 16.3 Where `{weights}` points: one merged root on the container disk

`{weights}` (and `FV_WEIGHTS` for `verify-weights.sh`, `FV_AUX_DIR`) is `<scratch>/weights` (`/fvscratch/weights`):

- one **symlink per volume tree** into `<work>/weights` (`/workspace/weights`), for the trees no other volume tree holds
  (`TreeSources::link_roots`: `h3-base` covers its nested prequant trees; `auxiliary/tae` and `auxiliary/lpips`, not
  `auxiliary`, so a fetched `auxiliary/<x>` lands on the container disk). Made once at the run's start; the volume is
  only read (CLAUDE.md: never written).
- the **fetched trees as plain directories** beside them, where `r2-fetch.py --root` puts them (`.complete`,
  `.fv-revision`, `sha256.txt`, `.fv-manifest.json`).

So the layout is the volume's, a tree per `<root>/<dest>`, which is what `fv-gpucheck --weights-root`, `fv-gpucheck engine`
(the gold cells) and `verify-weights.sh` read (`find -L`, `-f` / `-d` follow links; fv-serve read a weights root of symlinks into the volume in the
2026-10-14 H3 round, pod C, `docs/serve/e2e/h3-test-round-2026-10-14.md`). One root for every arm, not one per arm: an arm's reads stay a superset of
the volume layout, so the volume-only arms run exactly as on a `volume` tier. An R2 tree that would hold a volume tree,
or a nested R2 tree whose parent the arm does not read, is refused before any arm (`TreeSources::arm_r2`; `FATAL`, exit
2); no gold arm has either.

**Never on the volume** (CLAUDE.md: the volume is add-only and never written here). `fv-lab run` refuses a
`volume+r2` tier, exit 2 before anything is written, when the scratch disk is the work directory (the default without
`--scratch` / `FV_SCRATCH`) or when the merged root, resolved through symlinks, is inside or above the work directory or
the volume mount `/workspace` (`weights::check_off_volume`). The runner checks it again when it links the root, before
every `r2-fetch.py`, in every removal of a fetched tree (`remove_fetched`) and before the staging clean-up, so a later
caller that skips the entry check still cannot write or remove on the volume (`tests/fv_lab_bin.rs`
`a_volume_r2_root_on_the_volume_is_refused`, `weights::tests::the_merged_root_stays_off_the_volume`). A tree under a
volume tree is the volume's even when R2 publishes it on its own (`ltx25/text_encoder_fp8` under `ltx25`): it is never
fetched, so nothing is written through a volume symlink (`weights::tests::a_tree_under_a_volume_tree_is_the_volumes_…`).
`FV_EU_VOLUME_TREES` names are checked like `weights-r2.tsv`'s (relative, no `.` / `..`, `[A-Za-z0-9._/-]`).

#### 16.4 The fetch: the repository's tools, per arm

Before an arm's gate (`Runner::stage_r2`; `gated_cell` and so `tae_gated_cell` and verdict arms too):

1. `weights::step(held, need)`: when every R2 tree the arm reads is already on the disk, nothing happens. Otherwise every
   fetched tree the arm does not read (and that holds none it reads) is **removed**, then the missing ones are fetched.
   An arm that fetches nothing changes nothing, so a tree stays across volume-only arms until an arm that must fetch
   needs the room. Only directories the run fetched are removed; a symlink is never removed (`remove_fetched`).
2. `python3`: the runtime image has none, and stays so. It is Python-free by design (`docker/gpucheck.Dockerfile`
   `runtime`: "No Python, no PyTorch, no toolkit: hosts boot quickly"), and the image job fails on a `python3` in it
   (`gpucheck-runtime-image.yml`); only the `volume+r2` runs need one, so they bring it rather than every runtime pod
   carrying it. At the first fetch of a run the runner checks the tools' imports (`weights::PYTHON_CHECK`) and, when
   they fail, runs `apt-get update` and `apt-get install -y --no-install-recommends python3` (`-o
   DPkg::Lock::Timeout=120`) under one `timeout 600` (`APT_CAP_S`, the bound the idle limit counts, §16.6), into
   `<runs>/r2-apt.log`, as fv-boot does on an image without it. Only a success is remembered: after a failure the next
   fetching arm tries again (`tests/r2_runner.rs` `a_failed_python3_install_is_tried_again_…`), and
   `tests/lab_files.rs` holds the runtime stage Python-free.
3. `python3 scripts/gpu/r2-weights.py plan <dest>@<rev>… --pin scripts/gpu/weights-r2.tsv --ttl <cap + 600> --out
   <scratch>/r2/plan.json`: the presigned plan for exactly the missing trees, at the revisions `weights-r2.tsv` records
   (no listing). `--pin` (new) refuses a manifest whose SHA-256 is not that table's column 5, as fv-control's
   `r2Manifest` does (`test_r2_weights.py` `test_plan_pin_…`). The only child that gets the R2 key. `<scratch>/r2` is
   mode 700 and outside `<scratch>/runs`, which the pod serves over HTTP; the plan (bearer URLs) is removed after the
   fetch whatever it did, and at the run's start (an interrupted run's).
4. `timeout --signal=TERM --kill-after=15 <cap> python3 scripts/gpu/r2-fetch.py --root <scratch>/weights --plan …
   --skip-trees '' --status <arm>/r2-fetch.json` (and `FV_R2_SKIP_TREES` removed from its env: every tree of the plan),
   `<cap>` the tier's `cell_timeout_s` (3600 s; 3600 when 0), never past the run's deadline (§16.7). Its log is
   `<arm>/r2-fetch.log`; `pod fetch` takes `r2-fetch.json` with every arm JSON. After a failed fetch, each tree it
   completed stays (it is held, as a success would have it) and each it did not is removed, with the staging
   (`.r2-partial`, `.stale-*`), so the disk holds only held trees; the arm is recorded as skipped (`r2 fetch failed`),
   `live.log` gets `SKIP <arm>: r2 fetch failed: <the fetcher's FAILED line>`, and the run goes on.

The children share `fv-lab`'s process group, so an interrupt of the group (a terminal's Ctrl-C) stops a fetch too; a
run killed alone leaves at most `plan.json`, which the next run removes first.

Not a third fetcher: `r2-fetch.py` is the fetcher fv-boot's R2 source runs (ranged GETs, every 64 MiB part checked
against the manifest's SHA-256, stragglers cut, resume), and `r2-weights.py plan` is the presigner the experiment and
volume-fill scripts use. fv-boot's own R2 source (`boot::weights::r2`) is not called: it asks fv-control for the plan
(`FV_WEIGHTS_PLAN_URL` with a cluster pod's token), and an fv-lab pod is no fv-control pod. Its helpers are reused
(`tree_present`, `clean_leftovers`, `failed_line`, `ready_line`).

`--dry-run` prints the plan with every fetch assumed to succeed: `dry-run weights <root>: …` after `matrix start`, and
`dry-run r2 <arm>: drop …; fetch <dest>@<rev> (<GB>) into <root>` before each arm that fetches (`fixtures/gold-dry-run.txt`).

#### 16.5 Credentials: Runpod secret references, as every serve pod

The pod's env holds `FV_WEIGHTS_R2_ACCESS_KEY_ID = {{ RUNPOD_SECRET_fv_r2_access_key_id }}` and
`FV_WEIGHTS_R2_SECRET_ACCESS_KEY = {{ RUNPOD_SECRET_fv_r2_secret_access_key }}` (`weights::r2_env_refs`, set by `pod::launch`
for a `volume+r2` tier only). Runpod fills them in inside the container; the values are never in a `Launch`, the
`pod up` plan, a golden, the ledger, a file, a log or an argv (the mechanism of fv-deploy's `secret_env_refs`, the serve
templates and `segments-pod.sh`'s R2 mode; fv-deploy's `check_env` still refuses the API key). The endpoint and bucket are
`r2-weights.py`'s defaults (the account's endpoint, `fv-weights`). `fv-lab run` reads the key once into `weights::R2Key`
(whose `Debug` prints `<redacted>`), removes both names from every child's environment (`Runner::cmd`) and sets them on
the plan step alone; a test checks that an arm sees neither and that no file of the run directory holds either value.
Without the key (or with an unfilled reference) the run logs a warning and every arm that must fetch is recorded as
skipped (`r2 fetch failed`); volume arms still run.

UNVERIFIED: that the `fv_r2_*` secrets (the serve containers' R2 key, made for `fv-media`) can read `fv-weights`.
`segments-pod.sh` uses them for exactly this, but no run of that mode is recorded. If they cannot, the plan step answers
403, the R2 arms are skipped and the volume arms still produce their rows; the remedy is the owner's (Runpod secrets
from the read-only weights key `r2-fill-volume.sh` uses, and the two names in `R2_KEY_ENV`). Least privilege argues for
that dedicated read-only pair anyway (open decision 4 below).

#### 16.6 The container disk and the gold tier

The gold tier is `weights = "volume+r2"` and `container_disk_gb = 300`. From `weights-r2.tsv` and the tier's order
(`tests/gold_tier.rs` pins both numbers):

| | |
|---|---|
| R2 trees the 32 gold arms read | 18 distinct, 530.74 GB, each fetched once (an arm's trees are kept while the next arms read them) |
| largest single arm | `fasth3-8step-dense`: `h3-8step`, 147.86 GB (then `wan14b-plug-4step`, 84.09 GB) |
| most on the disk at once | 147.86 GB (`WeightsPlan::peak_bytes`: after a fetch the disk holds exactly what that arm reads) |
| container disk | 300 GB = 147.86 GB + the 120 GB every `volume` tier has for runs, frames, TAE files and library links (`DISK_RESERVE_GB`), rounded up; `r2-fetch.py` refuses a fetch that would leave under 5 GB free |

Keeping every tree instead (no drops) would need about 650 GB of container disk for no gain: each tree is fetched once
in this order anyway. 11 of the 32 arms read the volume only; 14 fetch, and 7 more read trees an earlier arm fetched.

**The idle watchdog.** The pod's idle watchdog deletes it after `FV_IDLE_GPU_MIN` minutes at 0 % GPU (10 on a volume
tier, 30 before the GPU's first use), and an R2 fetch uses no GPU: a slow 148 GB fetch would delete the pod with every
result. A `volume+r2` tier's limit, before and after the first use, is `weights::r2_idle_min`: a fallback `apt-get`
(600 s) plus one fetch at its cap, in minutes, plus 15, at least 30: **85 min** for the gold tier. The watchdog itself
is unchanged (a heartbeat would change every tier's start command, which the scripts' goldens freeze). Two fetches
back to back with no GPU arm between (a fetching arm whose gate then fails) at the cap each could still reach it; at
the measured rates the largest fetch is about 4 min.

The pod body golden (`runpod-http-gold.json`) is `runpod-http.sh plan` with `FV_CONTAINER_DISK_GB=300`,
`FV_IDLE_GPU_MIN` = `FV_IDLE_GPU_GRACE_MIN` = 85 and `FV_MIN_RAM_GB=256` (the tier's `min_ram_gb`, §16.10;
`fv-lab-goldens.sh`), and its env (the R2 references) is the one allowed difference (`tests/payload_golden.rs`).

#### 16.7 The GPU check (the coordinator's)

On a commit whose runtime image `ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-<7>` is published with this change in it
(the image carries `lab/`, `scripts/gpu/r2-*.py`, `weights-r2.tsv` and `volumes.sh`), from the repository root with
`fv-lab` built on the build pod, and the Runpod secrets `fv_r2_access_key_id` / `fv_r2_secret_access_key` present:

```sh
SHA=<7>
# Optional first, about 20 min and under $1: two R2 arms (one fetch each) and one volume arm; proves the key reads fv-weights.
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family gold --cells h3-larry-turbo,vdn-h3,sol-h3 --yes --wait
# The full gold run: all 32 arms.
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family gold --yes --wait
#   -> artifacts/runpod/gold/$SHA-<MMDDHHMMSS>-<hex>/   (the pod is deleted at the end)
./fv-lab report tiers --runs artifacts/runpod/gold/$SHA-<MMDDHHMMSS>-<hex>
# Or split in two (below: when one run would not fit the wall cap), two pods at the same $SHA:
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family gold --cells <the 21 arms that read R2 trees> --yes --wait
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family gold --cells <the 11 others> --yes --wait
#   -> two run directories; one table from both (also `--runs <first> --runs <second>`):
./fv-lab report tiers --runs artifacts/runpod/gold/$SHA-<first>,artifacts/runpod/gold/$SHA-<second>
```

Pass: `DONE` and `matrix_exit=0`; `live.log` has the line `weights /fvscratch/weights: 10 volume trees linked from
/workspace/weights; 18 R2 trees (530.74 GB) …` and one `r2 <arm>: weights ready: …` line per fetching arm (14); no arm skipped for `weights incomplete` or
`r2 fetch failed`; each arm `exit 0` or a recorded failure of the engine (ADR-0023 §3: the reference- and
audio-to-video variants cannot run from a text request and fail, which is a finding).

Cost (RTX PRO 6000 at about $2.49/hr, the tier's 4 h wall cap): boot and readiness 5-10 min; `apt-get install python3`
about 1 min; 530.74 GB from R2 at the 0.61-0.79 GB/s two R2-mode PRO 6000 pods measured on 2026-10-14
(`docs/serve/e2e/h3-test-round-2026-10-14.md`; not in EU-RO-1, so UNVERIFIED there), 11-15 min; 32 cold arms at 2-5 min
each (the gold files' recorded load times are 36-139 s, generation seconds to about two minutes), 64-160 min. In all about 1.5-3.1 h,
**$3.70-$7.70**, at most **$9.96** at the cap; fv-deploy's floor needs a balance of at least $17.96 ($8 + 4 h x $2.49)
before the create. R2 egress is free.

**Near the wall cap a run ends early and is still fetched** (any tier). `pod up` puts `FV_RUN_DEADLINE` (unix
seconds: its create time plus the cap less `RUN_DEADLINE_RESERVE_S`, 20 min) in the pod's env, not its start command,
so the payload goldens stay the scripts'. Before each arm the runner skips it as `wall cap near` when less than 600 s
(`MIN_ARM_S`) would be left after its R2 fetch at a 0.25 GB/s floor; an arm that runs, and its fetch, get at most the
time left as their cap. So the runner ends and the start command writes `DONE` before the deadline. The driver (`pod up
--wait`, `pod fetch --wait`) stops waiting at the cap less `FETCH_RESERVE_S` (15 min, `pod::wait_limit_s`) and fetches
whatever the run holds (judged failed when there is no `DONE`), while the pod still lives. If the smoke run's per-arm
times project past about 3.5 h, splitting the full run in two with `--cells` (the 21 arms that read R2 trees, then the
11 others; about 10 min of boot more) still measures every arm, and `report tiers` renders the two run directories
as one run set: `--runs` takes several, comma separated or repeated (a run directory must not contain a comma). It
refuses runs of different tiers, a run whose commit is not known (no `pod.json` sha and no tag head of at least 7 hex
digits), runs of different commits, the same directory twice (however written), and an arm that passed (exit 0) in
more than one run. An arm in several runs is shown from the copy that passed, else the last copy that failed, else
the last skipped copy (in `--runs` order), so a failed or `wall cap near` arm rerun in a later run shows the rerun;
the run it was not shown from lists it. Each row reads its arm's own run (its `pod.json` GPU and price, and the end
its validation age counts to); the header names every run, and a single run renders as before. `report tiers` writes
no gold file: `[last_validated]` is written by hand (ADR-0023 "What a gold run then does"), and its `run` is the
arm's `summary.json` in the directory that variant's arm came from (`tests/gold_files.rs` requires the run record to
be a file).

**Result (2026-10-26): pass, split in two.** `SHA=7e580e5` (image `fastvideo-rs-runtime:sha-7e580e5`, digest
`sha256:76a74c1b…`, build id `a968fe04b59a01ab`), the two `--cells` commands above, both pods an RTX PRO 6000
Blackwell Server Edition in EU-RO-1 at $2.49/hr, created with the tier's host-RAM floor (`min_ram_gb = 192`,
§16.10); Runpod gave each 283 GB (`pod.json` `memory_gb`; cgroup v2 limit 269,889 MiB in `box.txt`).

- *The 21 arms that read R2 trees*: pod `4s5wqhitndx3hi` (wall cap 7200 s) →
  `artifacts/runpod/gold/7e580e5-1026090916-e804` (`DONE`, `matrix_exit=0`; `matrix start` 09:09:50 to `matrix
  done` 10:15:45, about 66 min of arms and 68 min of pod). `live.log` has `weights /fvscratch/weights: 10 volume
  trees linked from /workspace/weights; 18 R2 trees (530.74 GB) fetched before the arms that read them, at most
  147.86 GB on the container disk at once` and 14 `r2 <arm>: weights ready` lines (0.52-0.96 GB/s for the trees
  over 10 GB; `h3-8step`, 147.86 GB, in 162 s; no retry); no arm skipped for `weights incomplete` or `r2 fetch
  failed`. All 21 arms exit 0.
- *The 11 volume-only arms*: pod `68yj5g2i9dczyd` (wall cap 3600 s) →
  `artifacts/runpod/gold/7e580e5-1026091001-7421` (`DONE`, `matrix_exit=0`; 09:10:34 to 09:30:09, about 20 min of
  arms and 22 min of pod). 7 arms exit 0; 4 exit 2 with `negotiate: … model `<id>` does not support task t2v`
  (`h3-ref2v-max`, `h3-ref2v-turbo`, `ltx25-a2v-guided`, `ltx25-ref2v`): the recorded engine failure ADR-0023 §3
  expects of a reference- or audio-to-video variant, a finding and not a number. The two H3 ones loaded for 199 s and
  132 s first (`load/io h3 total`), the LTX ones for 5 s and 34 s (`load/io ltx2 load`; plan D1, follow-up 4: negotiate before load).

Pass: every criterion above holds on both pods. Cost: about 68 + 22 min at $2.49/hr, **about $3.74** (estimate
$3.70-$7.70). `fv-lab report tiers --runs artifacts/runpod/gold/7e580e5-1026090916-e804,artifacts/runpod/gold/7e580e5-1026091001-7421`
renders `docs/perf/tiers.md` from the two (committed, as the run set): 28 variants measured, every one `same` as its
gold file's canvas, frames and steps; the 4 above `not measured (failed: exit 2)`.

*The gold files* (ADR-0023 "What a gold run then does", ADR-0009). Each of the 28 measured variants has
`[last_validated]` `commit = "7e580e5"`, `run = "<its run directory>/<arm>/summary.json"`, `date = "2026-10-26"`, and
its arm's `gpucheck-out/engine.json` as a `[[source]]` (heading `checks`: the `timings` check, with the host-RAM
record in the note). Gaps are closed only for fields the run measured, and one `[latency]` table holds one kind of
measurement, from one process:

- the 10 files with no latency measurement before (`fasth3-4step-vsa-nvfp4{,-taeh3}`, `fasth3-8step-dense`,
  `h3-larry-turbo`, `h3-lightx2v-turbo{,-v01}`, `longlive-1.3b`, `vdn-h3{,-8step}`, `wan14b-plug-4step`) take the
  run's job: `measures = "serve_job"`, `p50_s` = `submit_to_mp4_s` (the text encode of the uncached prompt included),
  `denoise_s`, `decode_s` = `video_decode` + `encode` (as the table), `peak_gib` where the engine reports one (not for
  the two Wan pipelines), `usd_per_clip` derived; `min_s` / `max_s` stay gaps (one request). `load_s` is closed only
  where no earlier arm of the same pod had read the arm's model trees (auxiliary TAE files, a few MB, aside):
  `fasth3-4step-vsa-nvfp4-taeh3` (88.1 s), `longlive-1.3b` (35.9 s) and `wan14b-plug-4step` (155.5 s). In the other
  7 it stays a gap whose reason says the load was page-cache warm and names the cold sibling where there is one
  (`fasth3-4step-vsa-nvfp4` 11.7 s right after `fasth3-4step-vsa-nvfp4-taeh3` read the same trees in 88.1 s; every
  other H3 arm of that pod after the first had `h3-base` warm); the measured value stays in the source's note. Their
  `prompt_set` loses "no run of this variant yet", so their cells are regenerated (the note only);
- the 18 files with a measurement of another kind (`gen`, `end_to_end`, `stream`: CLI medians, a cached prompt, often
  an older setting in `workload.env`) are unchanged in their numbers: a serve job is not that quantity, and `load_s`
  and `peak_gib` are of "the process that measured" (`gold.rs`), so no gap closes; the run's numbers are in its
  source's note (`fields = []`), and the gap reasons for `load_s` / `peak_gib` (and `fasth3-trim-8step`'s `p50_s`)
  point to it;
- the 4 refusals are not validated and their gold files are unchanged.

`[last_validated]` here means the variant ran to exit 0 at this commit on its gold cell; it does not mean every number
already in the file was re-confirmed, and files whose recorded `peak_gib` differs from the run's (for example
`fasth3-trim-nvfp4`, 20.3 GiB against 43.6) measure another setting or process: see their notes.

A `load_s` of the gold run is the engine's load in the run's arm order: a tree an earlier arm read, or that the arm's
own R2 fetch had just written, can load from the page cache (`fasth3-4step-vsa-taeh3` 7.0 s after
`fasth3-4step-vsa-480p-taeh3`'s 166.0 s); each source's note says so.

*Host RAM.* Every arm that ran recorded `peak_ram_mib`, `ram_limit_mib`, `pod_peak_mib_cumulative` and `oom_kills`;
`oom_kills` is 0 for every arm. `wan14b-plug-4step` peaked at 156,004 MiB of process memory (152.3 GiB, 163.6 GB),
about twice the ~80 GB of §16.10's estimate; its cumulative peak reached the 269,889 MiB limit (the page cache of
its 84.09 GB fetch, reclaimable) with no OOM kill. The next are `h3-ref2v-turbo` 79,933 MiB, `fasth3-8step-dense`
78,470 MiB and `h3-ref2v-max` 71,543 MiB; every other arm stays under 70,000 MiB. Runpod's GB are 10^9 bytes (283 GB
is the 269,889 MiB limit), so the 192 GB floor is 183,105 MiB and a pod given exactly the floor would leave the
14B plug load about 27,100 MiB (26.5 GiB) by this sample. That does not show such a pod survives: the peak is a
once-a-second sample, and pages not yet written back (up to the 84 GB the arm had just fetched) are charged to the
same cgroup (§16.10), so the floor **may not hold**. Raising the gold tier's `min_ram_gb` (or cutting the loader's host
copies) is a prerequisite of the next gold run (plan D3, follow-up a): raised to 256 GB, §16.10 "Raised to 256". The peak supports host-RAM starvation as the cause of `e9b9435`'s exit 137 (below), which
that pod, with no floor, could not record.

*Before the fixes: the run at `e9b9435`* (the same split; RTX PRO 6000, EU-RO-1, no host-RAM floor; not committed,
as no failed attempt of an earlier gate was, §15.7). Smoke `e9b9435-1026053846-0a25`: `h3-larry-turbo`, `vdn-h3`,
`sol-h3` exit 0 (after #203; the smoke `f93a961-1026044005-eb60` had failed the first two on the technique profile,
ADR-0023 §3). Volume half `e9b9435-1026060611-28c6` (pod `v284zu5hlj0omr`): 7 exit 0, the same 4 t2v refusals. R2
half `e9b9435-1026062720-e2e4` (pod `v2umu0dbeazjfb`): 14 exit 0 and three findings, each fixed before `7e580e5`:

- the 4 NVFP4 arms skipped `weights incomplete` (their R2 trees leave out vae shard 3, byte-identical to h3-base's):
  #208 (`verify-weights.sh` `shared_file`, size and SHA-256 checked); at `7e580e5` all four exit 0;
- `longlive-1.3b` and `sfwan21-1.3b` failed `engine/progress` alone (28 events against 4 steps): #209 (the check
  expects every causal block's steps, ADR-0023 §3); at `7e580e5` both exit 0;
- `wan14b-plug-4step` exit 137 18 s into its load, right after its 84.09 GB fetch: #210 (the 192 GB floor and the
  pod's memory on record, §16.10); at `7e580e5` it exits 0 at the 156,004 MiB peak above.

#### 16.8 Differences, and what stays

- A nested R2 tree is fetched only with its parent (refused otherwise, §16.3); no gold arm reads one.
- The preflight `weights.log` and `tree.txt` at pod start still describe the volume only.
- `gold-run.yml` needs nothing new: the key is a Runpod secret, not a GitHub one.

#### 16.9 Review of #201 (2026-10-26)

The coordinator's review, all fixed on the branch: the merged root may never be on the volume (§16.3, the blocking
finding: without `--scratch` it was `/workspace/weights`, and fetches, removals and the staging clean-up would have
written and deleted there); the idle watchdog's limit for `volume+r2` pods (§16.6); a run deadline and the driver's
fetch reserve, so a run near the cap ends and is fetched (§16.7); `plan --pin` (§16.4); a partly failed fetch keeps
what it completed and removes the rest (§16.4); python3 by a capped apt install retried at the next fetching arm
(§16.4: the review's suggestion to put python3 in the runtime image was dropped, because that image is Python-free by
design and its CI check fails on it); the TAE directory read from the volume's marker when the command lines are rendered (`RunConfig::tae`);
`FV_EU_VOLUME_TREES` names checked; the nested-tree classification pinned by a test; `--skip-trees ''`; `plan.json`
removed at the run's start; the schema wording (§16.2).

#### 16.10 The host-RAM floor: `min_ram_gb` (2026-10-26)

**Why.** In the full gold run at e9b9435 (RTX PRO 6000, EU-RO-1) the arm `wan14b-plug-4step` died 18 s into its load
with exit 137, right after its 84.09 GB of R2 trees were fetched, and printed nothing after the device line
(`artifacts/runpod/gold/e9b9435-1026062720-e2e4/`, gitignored). A SIGKILL during a load with no stack dump is what the
pod's cgroup OOM killer leaves, as with the upstream oracle cell (docs/oracle.md "Upstream cells: host RAM"). fv-lab's
pods asked for no RAM floor, so they got Runpod's default (`minRAMPerGPU` 8 GB per GPU) and whatever share the host
gave; and `box.txt` did not record that share, so the OOM kill could not be proved.

**Decision.**

1. *Schema.* A tier may set `min_ram_gb` (tiers v1, additive: optional, `deny_unknown_fields` kept): the pod's
   host-RAM floor in whole GB per GPU. Validation refuses 0 (`Tiers::validate`); serde refuses a negative, fractional,
   string or out-of-range value. Absent, nothing changes: no tier's pod body changes but the gold tier's.
2. *fv-deploy.* `GpuSpec::min_ram_gb` (`launch.v1.json`, additive, omitted when unset) is Runpod's `PodCreateInput`
   `minRAMPerGPU` in `pod::create_body`, sent only when set, so every existing payload golden is byte-identical.
   `guard::check_min_ram` (in `Deployer::preflight`, which every fv-deploy `up` runs) refuses 0 and a floor on any
   launch but a Runpod pod (no other create body carries one, so a serverless launch's would otherwise be dropped).
   The schemas say `minimum: 1` (`#[schemars(range(min = 1))]`), so fv-control's zod validator refuses 0 too.
   fv-control's `runpodPodBody` mirrors the field; its endpoint bodies have none, and it builds no launch with a
   floor; `launch-gen.ts` is regenerated (`gen-configs.mjs` now carries an integer's `minimum` into zod).
   `Instance::memory_gb` / `vcpus` (`instance.v1.json`, additive) are the create answer's `memoryInGb` / `vcpuCount`;
   `up` logs them with the price (`runpod-pod <id> on <gpu> at $x/hr ram=<n>GB vcpu=<n>`), as `runpod-http.sh` does,
   and `pod up` records them, with the floor asked for, in `pod.json` and `<state>/pods/<id>.json`. `PodRecord` keeps
   `deny_unknown_fields` (a misspelt field is an error), so `pod reap` skips a record it cannot read (another fv-lab
   version's, a torn write) with a warning naming the file, and reaps the rest (`pod::read_records`).
3. *fv-lab pod.* `pod::launch` copies the tier's floor into the launch. The pod body golden passes it to the script as
   `FV_MIN_RAM_GB` (§16.6), so both bodies carry `minRAMPerGPU` and `tests/payload_golden.rs` pins it to the tier's
   value (and that only the gold tier has one). On the script route of a retired family, `FV_MIN_RAM_GB` for a tier
   without a floor is refused naming `min_ram_gb` (`fv-lab-route.test.sh`). The branch that passes a value equal to
   the tier's floor is defensive: today only `gold` has a floor, and `gold` never takes that route (`run` / `all`
   are refused for it first).
4. *The runner records the pod's memory* (`podmem`, the method of `scripts/gpu/upstream/pod.sh` `cg_dir` / `cg_mem`):
   the memory cgroup from `/proc/self/cgroup` (v2 `0::<path>`, else v1's `memory` controller) under `/sys/fs/cgroup`,
   else the mount's root. At start it appends to `box.txt` `cpus=<n> mem=<host GiB>G` and `cgroup=<v2|v1> <dir>
   pod_ram_limit_mib=<n|null> oom_kills=<n>` and logs `pod memory: …`. Each arm that runs is sampled every second
   (anon + shmem on v2, total_rss + total_shmem on v1: process memory without the page cache), and its `summary.json`
   gains, before `ended`, `peak_ram_mib` (the whole cgroup's sample: the runner and anything beside the arm count
   too), `ram_limit_mib` (the runner's own cgroup's limit; an ancestor's is not read), `pod_peak_mib_cumulative`
   (v2 `memory.peak`, v1 `memory.max_usage_in_bytes`: since the pod started, page cache included; `null` on v2
   before Linux 5.19) and `oom_kills` (during the arm); `null` when unreadable, and none of the four without a
   memory cgroup. New fields only: every reader reads by key
   (`report tiers`, `fv-lab-compare-runs.sh`), by `"exit":…` (`summary_exit_ok`) or carries the object whole (`report
   table`'s rows). `live.log` gets `ram <arm>: ram <peak> MiB of <limit> MiB, oom_kills <n>` per arm, and `OOM <arm>: the kernel OOM-killed <n> process(es) in this pod
   during the arm (host RAM, not GPU …)` when the count rose, after the arm's `FAIL` line.
5. *The gold tier's floor: 192 GB.* The largest host load of a gold arm is `wan14b-plug-4step`'s:
   `wan::plug::load_merged_transformer` maps the DiT shards but copies every tensor into an owned host buffer in its
   on-disk dtype (`load_raw_tensors_native`; the 14B DiT is f32 on disk, about 57 GB of the 80.41 GB tree), merges
   the two LoRA trees (3.68 GB) into it, and holds it while `WanPipeline` loads the f32 UMT5 encoder the same way
   (`WeightMap::from_dir`, about 23 GB): about 80 GB of process memory at the peak. 192 GB is that with more than
   100 GB of headroom, and the floor the upstream cells' rerun asked for on a PRO 6000 in EU-RO-1, where Runpod gave
   the pod 283 GB (docs/oracle.md "Upstream cells: host RAM"). The page cache the fetch and the copy fill is charged
   to the same cgroup but reclaimable, so it is no OOM risk; the one caveat is pages not yet written back, at most
   the 84 GB the arm just fetched. Every other gold arm reads smaller trees, or `h3-8step` (147.86 GB) through a loader whose host
   footprint the H3 tiers run without a floor today; no other tier gets one without evidence.

**Consequences.** A pod with a floor can only land on a host with that much RAM to give: fewer offers, and a create
with none is fv-deploy's `no capacity` (nothing created, nothing billed). The floor is a guess from the loader, not a
measurement; the first gold run with this change records each arm's peak and the pod's limit, from which the floor can
be lowered (or the 14B loader made to stream instead of copying). UNVERIFIED: that the 137 at e9b9435 was the cgroup
OOM killer (that run recorded no cgroup), and that Runpod honours `minRAMPerGPU` on PRO 6000 hosts in EU-RO-1 as on
the upstream pods.

**Raised to 256 (2026-10-26, after the gold run at `7e580e5`, §16.7 "Result").** The measurement replaced the guess
of item 5: `wan14b-plug-4step` peaked at 156,004 MiB of process memory (`peak_ram_mib`, anon + shmem sampled each
second; its own log line `wan load: host peak RSS 171.1 GiB (eager dit)`, which also counts the mapped shards), about
twice the 80 GB above. The loader holds every f32 tensor twice: `fastvideo_loader::RawTensor` keeps an f32 tensor's
little-endian `data` bytes and a parallel `values: Vec<f32>` (`from_native`, `from_f32`), so the merged DiT is about
2 x 57 GB and the UMT5 map, copied while the DiT map is held, about 2 x 23 GB: about 150 GiB with the VAE, the
sample's 152.3 GiB. Runpod's GB are 10^9 bytes, so 192 GB is 183,105 MiB, 27,101 MiB above that peak; pages of the
arm's 84.09 GB fetch not yet written back are charged to the same cgroup (item 5), so the floor may not hold.

- *Decision:* the gold tier's `min_ram_gb` is **256**: 244,140 MiB, 88,136 MiB above the peak, which is more than the
  whole 84.09 GB fetch (80,195 MiB) even if none of it were written back yet. The guards allow it (`min_ram_gb` is a
  positive `u32`, `Tiers::validate` and `guard::check_min_ram` refuse only 0, the schemas say `minimum: 1`, and
  `runpod-http.sh` takes any whole positive number), and it is below the 283 GB Runpod gave both pods of the run.
  The pod body golden (`runpod-http-gold.json`, `FV_MIN_RAM_GB=256`) and `tests/payload_golden.rs` pin it.
- *Not chosen:* 283 (what the hosts gave: no headroom over today's offers, more `no capacity`); cutting the loader's
  host copies instead. Either cut is free of numerics but not of review: storing an f32 `RawTensor` once (about
  76 GiB at the peak) changes the loader every model shares, and uploading the DiT before the text encoder when it
  comes in as an eager map (about 107 GiB: the DiT map is dropped before UMT5 is copied) changes the device's
  allocation order and, for a two-expert checkpoint, the free memory the MoE plan reads "after the text encoder".
  Both are left to their own change with a GPU byte-identity check; with both, about 53 GiB.
- *Consequences:* fewer hosts qualify: a PRO 6000 host with less than 256 GB to give is no longer offered to the gold
  pod, and a create no host can place is fv-deploy's `no capacity` (nothing created, nothing billed); the driver
  then fails the `pod up` as before. UNVERIFIED: how many EU-RO-1 PRO 6000 hosts give 256 GB or more (both gold pods
  got 283 GB at a 192 floor).

### 17. D1f: `fuse`, `attn3` and `ltx` as tiers; the B4 families stay; `h3` proposed for retirement

Status: accepted 2026-10-26 for the three moved families: their GPU checks (§17.5) passed on `762ce4d`, and `fuse`,
`attn3` and `ltx` are retired from the script (in `FV_LAB_RETIRED`, branches deleted, fixtures frozen, §17.6). The B4
families (§17.3 option (b)) and `h3` (§17.4) are retired with no tier since §18 (coordinator decision 2026-10-26). No script branch was deleted in
the PR that made the tiers (#212): each family left `runpod-matrix.sh` only after its own check through both runners
(§8), as §10, §11.6 and §15.10 did. Decision-id FVID-2026-10-26-d1f-tiers.

This slice took seven families of §9.1: `b4b`, `b4c` and `b4c2` (the plan-B4 byte-identity families with verdicts),
`fuse`, `attn3` and `ltx` (the ones §9.1 said "need ungated arms", which §15 added), and `h3` (§9.1: "retired, rtx6000
covers it"). Three moved, the three B4 families did not (§17.3), and `h3` is a retirement proposal (§17.4).

#### 17.1 What moved

| tier | cells | arms (compares / gates) | the script it is |
|---|---|---|---|
| `fuse` | `fuse-kernels`, `fuse-trace`, `fuse-switch` | 10 (3 / 3) | `runpod-matrix.sh fuse`: the ungated `kernels-fuse` (`fv-gpucheck kernels --groups dit_fusion,bf16_act`), three CUPTI-traced warm steps (`trace-*`), then six off / on arms and per pair a `--off-identity` compare with its `exact` gate (`off` = the candidate itself) |
| `attn3` | `attn3-h3-ltx25` | 6 (3 / 0) | `runpod-matrix.sh attn3` at its defaults (`FV_ATTN3_ARMS="v2 cudnn"`, `FV_ATTN3_VSA="0 1"`): the dense attention A/B on FastH3 4-step dense and LTX-2.5 1080p20s dense, the VSA bf16 pair, then three compares |
| `ltx` | `ltx-versions` | 3 | `runpod-matrix.sh ltx`: LTX-2.5 two-stage (warm), LTX-2.3 two-stage, LTX-2.0 19B distilled |

All three are the `rtx6000` tier's pod (PRO 6000, EU, `bz0ofkfx45`, the six preflight cells, the TAE fetch, a 4 h cap,
3600 s per arm), as `runpod-http.sh` gives every `FV_FAMILY`. The goldens come from the live branches as every script
tier's (§9.3): `fixtures/{fuse,attn3,ltx}-dry-run.txt` (18 / 11 / 5 lines), `-gates.tsv` (12 / 6 / 3 rows),
`runpod-http-{fuse,attn3,ltx}.json`; `tests/dry_run.rs` and `tests/payload_golden.rs` hold fv-lab to them unchanged
(the gate count there is 28 now: `fuse`'s three exact gates). With them `lab/tiers.toml` has 21 tiers (19 of them
`runpod-matrix.sh` families or modes, retired or not; besides those, `gold` and `h3-graphs`).

Weights: every tree an arm reads is on the EU volume (`h3-base`, `FastH3-4-step-Preview-v1-LoRA`, `ltx25`) except
`h3-8step` (`fuse`'s three 8-step arms, as elsewhere) and `ltx23` / `ltx2` (`ltx`'s LTX-2.3 and LTX-2.0 arms: R2 only,
71.60 GB and 94.74 GB in `weights-r2.tsv`). Those arms are recorded as skipped (`weights incomplete`) by both runners;
`tests/lab_files.rs` now pins those four trees as the only ones any script tier lacks on the volume, and the two `ltx`
arms by name. No new tree, no download; the tiers are `weights = "volume"` (a `volume+r2` `ltx` would fetch 166 GB per
run: the owner's call, CLAUDE.md, not proposed).

#### 17.2 What each needed: no new cell field

**`fuse`.** The `kernels-fuse` arm is §15.2's ungated arm (no `gate`, no tree, `--out {runs}/{arm}/gpucheck-out`): it
reads no weights, so §15's rule fits it as written. The script's three compares and gates follow all six switch arms,
H3 and LTX-2.5 alike (`for wl in …; do compare_cells; gate_cells; done` after the arms), so `fuse-switch` is one cell
holding both pipelines' arms with three `[[compare]]`s and their paired gates (§9.2 item 4: `gate = { kind = "exact",
off = "<candidate>" }`, the script's `gate_cells "$wl-off" "$wl-on" exact "$wl-on"`). The kernel check and the traced
steps are cells of their own (`fuse-kernels`, `fuse-trace`): no compare follows them, so the split leaves the run
order, and the dry run, as the script's.

**`attn3`.** One cell for the same reason: its compares follow all six arms. "Almost (`model` is one pipeline)" in
§9.1 was the only gap, and it is descriptive: `Workload::model` was never read by the runner, only required non-empty.
Its doc (and so `schemas/cell.v1.json`'s description) now says a cell whose arms run two pipelines names both,
space-separated (`model = "h3 ltx2"`, also `fuse-trace` and `fuse-switch`), and a cell without a generation names what
it runs (`fuse-kernels`: `model = "kernels"`). No field, no validation and no type changed: every cell parses as
before. Not chosen: a tier-level "compares after all cells" switch (a new field for what one cell already says) and
per-pipeline cells (the compares would move between the arms, and the dry run with them).

**`ltx`.** §9.1's "ungated `run_cell`" read weights: the script ran its three arms with bare `run_cell` on
`$W/ltx25`, `$W/ltx23` and `$W/ltx2`, so on `bz0ofkfx45` the two arms whose trees are not there failed on the missing
files (a non-zero exit) instead of being skipped. An ungated fv-lab arm may read no weight tree (§15.2 item 6), so
`ltx` would have needed that rule lifted, an arm that reads `{weights}` without a gate. Instead the script's branch
gates its arms (`gated_cell <arm> ltx25-two-stage | ltx23 | ltx2`, the `verify-weights.sh` cells of those trees), in
lockstep with the cell, as §9.4 (`techniques`' caches) and §15.4 (`determinism`'s `|| true`) changed real runs:

- **A real-run change** of `runpod-matrix.sh ltx`: on a volume with all three trees nothing changes (each gate passes,
  the same commands run); on `bz0ofkfx45` the LTX-2.3 and LTX-2.0 arms are `skipped: weights incomplete` (with their
  `<arm>.weights.log`) instead of a failed run of a missing file. `matrix_exit` is 0 either way (`run_cell` never
  failed the family). The dry run is unchanged; its gates record gains three `arm` rows.
- **Why not the rule lifted:** the gate is what §4 promises for every arm (a tree that is not there is a skip, not a
  crash), and an unverified read of `{weights}` would be the first; the script's crash was no measurement anybody
  relied on (the family has no compare and no report).

`FV_ATTN3_ARMS` and `FV_ATTN3_VSA` (attn3's arm pickers) are refused by `fv-lab run` and `pod up --extra-env`
(`knobs::SCRIPT_KNOBS`: the script still reads them, so `runpod-http.sh`'s `FV_LAB_NOT_READ` gains them when the
branch is deleted, as the oracle's did not yet); `--cells` by arm replaces them. `fuse` and `ltx` read no knob.

**Differences from the script, and why** (§9.4's kind; the arms, commands and files are the script's):

| | script | fv-lab | why |
|---|---|---|---|
| `fuse`: `kernels-fuse`'s `[PASS]` / `[FAIL]` lines and each trace arm's last `/gpu_trace` line copied into `live.log` | yes | no | log only: both are in the arm's `stderr.log`, which both fetches take (§9.4) |
| `FV_ATTN3_ARMS`, `FV_ATTN3_VSA` | pick arms | refused; `--cells` by arm | arms are the unit (§9.4) |
| `ltx`'s arms without their tree | failed (exit ≠ 0) | skipped | the script gates them too since this PR (above): both runners skip |

#### 17.3 Did not move: `b4b`, `b4c`, `b4c2` (each needs a new kind of verdict)

What the three share with `b4a` is in fv-lab already: `b4_exit_ok` (`run::summary_exit_ok`), `b4_identical`
(`run::compare_identical`), `b4_family_verdict` (`Runner::family_verdict`), the baseline binary (`bin = "baseline"`),
fresh arm directories and the `<family>-<cell>.verdict.json` frame (§11.1). What each cell's own verdict function reads
is not, and none of it is one of §11.1's lists (`evidence`, `count`, `identical`):

- **`b4c`** (2 cells, 8 arms, 6 compares; `b4c_verdict`). The "`text_cache` check" of §11.1 is two things:
  1. *A `benchmark.json` field rule per arm*: `"text_cache"` equal to the pass (`miss`, `hit`) and, for the H3 cell,
     `"text_encoder"` containing `fp8` on the miss and equal to `cache` on the hit (`b4_field`: the first string value
     of the key in the file with its whitespace removed), each failure a reason naming the whole arm and the value
     (`<cell>-base-miss:text_cache=none`), where §11.1 names an arm by its label (`base:exit`).
  2. *A cache-directory listing per arm group*: both arms of a group (`-miss`, `-hit`) share
     `{scratch}/b4c-cache/<cell>-<arm>`, removed once before the group (§15.2's `fresh` is per arm and below `{runs}/`
     only), listed after it into `<runs>/<cell>.cache-<arm>.txt` (`sha256 name` per file, name order, `find -maxdepth
     1 -type f`), checked for `*.part` / `*.tmp*` (`<arm>:temporary-files`), and at the end both listings non-empty
     (`no-cache-entries`) and equal (`diff` into `<runs>/<cell>.cache.diff`, `cache-entries-differ`).
  Besides those: the dry run's `dry-run <cell>-<arm>: rm -rf <cache> (a cold cache for the miss pass)` line names no
  arm; the reasons are ordered per arm group (base's passes, base's temporary files, pr's, then the identities, then
  the listings), not per arm then identities then diffs as §11.1; the listings and `.cache.diff` would have to reach
  `artifacts/` (`pod fetch`, `fv-lab-compare-runs.sh`), and `fv-lab gate` could not list a cache directory left on the
  pod, so it would re-read the fetched listings (an offline rule unlike `b4a`'s).
- **`b4c2`** (5 cells, 24 arms; `b4c2_switch_verdict`, `b4c2_noop_verdict`). All of `b4c`'s, plus state sets per arm
  (the baseline's read must *hit*), entry counts per cache (1, 2 and 1 files), a subset rule (the fresh cache's entry is
  in this build's listing, `grep -qxF -f`), one entry with the same bytes under different names (the no-change cell),
  a third cache used by one arm (`-pr-fresh`), and another reason order (every exit first, then the states).
- **`b4b`** (5 cells, 10 arms; `b4b_verdict`). A trace-evidence step: the last `/gpu_trace ` line of each arm's
  `stderr.log`, its `top_kernels` table scraped from JSON (`kernel_count`, `dropped_records`, per kernel `name` and
  `calls`), `fusion-evidence.log` (the `fvf_` / `h3_` / `wan_` / `mx_` kernels, name order) and `trace-meta.txt` per
  arm, `comm -23` of the two arms' evidence into `<cell>.fusion-evidence.diff`, per-cell named-kernel rules (`on`:
  each ran; `off`: no switch-gated `fvf_*` kernel but `fvf_split_heads_rows` ran, the named unfused kernels did), the
  PR table's completeness (its calls sum to `kernel_count`), equal launch counts, and the H3 block-structure
  arithmetic over four kernels' calls (`b4b_relations`). Also `b4b-h3-fullopt`, a cell that runs only when named.

Each is a typed verdict of its own with its tests against the script on fakes (`tests/verdict.rs`'s method), about
the size of §11 for `b4c` / `b4c2` together and again for `b4b`: more than this slice's other families together, so
they stop here (the task's rule for "something large"), and §9.1 says so per row. Their GPU gates passed when their
changes merged (2026-10-24: `b4b` #179, `b4c` #177, `b4c2` #178; `artifacts/runpod/{b4b,b4c,b4c2}/`), so nothing waits
on them. **Proposed for the coordinator** (not decided here): either
(a) a later slice adds one ordered `[[verdict.check]]` list (each check typed, `arm` named explicitly, evaluated in
file order after the compares, its reason text as the script's: `exit`, `field` with `equals` / `contains` / `one_of`,
`cache` with `dir` under `{scratch}`, `first` / `last` arms, `count`, `same_as`, `subset_of`, `same_bytes`, and for
`b4b` a `trace` rule), which reproduces both reason orders by listing checks in the script's order; or
(b) retire them as §14.1 retired finished experiments: their changes are merged and proven, their baselines are
pinned to the commit before each change, and a later change to the fusions or the prompt cache gets its own
byte-identity run (`techniques` / `b6` / `b4a`'s baseline-vs-this-build shape) and, for the cache format, the CPU
fixtures (`rt/prompt_cache/equality.rs`, `rt/fuse/equality.rs`). `b4-verdict.sh` and `tests/b4-verdict.test.sh` stay
either way until the branches go. **Decided** (coordinator, 2026-10-26): option (b), done in §18.

#### 17.4 `h3`: proposed for retirement (rtx6000 covers it, but for one fallback)

§9.1's claim, checked arm by arm against `lab/cells/rtx6000-h3.toml` (and the other tiers):

| `h3` arm (`runpod-matrix.sh h3`, every arm a bare `run_cell`) | the tier arm that covers it | what differs |
|---|---|---|
| `fasth3-8step-warm`: `--weights $W/h3-8step`, no `--h3-recipe`, `--text-encoder streamed`, cold | `rtx6000` `fasth3-8step` (`--h3-recipe 8step`, `auto`, `--warm`, gated) | `h3-8step` is not on the EU volume: the `h3` arm fails there (ungated), the tier's is skipped; neither measures anything on `bz0ofkfx45` |
| `sol-h3-4step`: `h3-base`, `--h3-recipe sol-h3`, `streamed`, cold | `rtx6000` `sol-h3` (`auto`, `--warm`); `b6-solh3-*` | the text encoder mode (streamed H3 encoding: `b6-h3-streamed`, `h3-graphs`, the oracle cells) and a cold process (every `--warm` arm runs the same request once untimed first) |
| `sol-h3-spark` (only when its upscaler and adapter files are found): `env -u FASTVIDEO_LTX2_WEIGHTS` W8A8, bf16 act, upscaler, adapter, `--h3-recipe sol-h3-spark`, `streamed`, cold | `rtx6000` `sol-h3-spark` (the same env and recipe **plus** `FASTVIDEO_LTX2_WEIGHTS={weights}/ltx25`, gated on `sol-h3-spark`) | **not covered**: without `FASTVIDEO_LTX2_WEIGHTS` the Spark recipe skips the 3-step LTX refiner and decodes with H3 (`h3/pipeline.rs`: "Set FASTVIDEO_LTX2_WEIGHTS to run the 3-step LTX refiner. H3 decode continues."). No tier runs that fallback; no family variant or gold file serves Spark (`families/`, `lab/gold/`), and the full bridge is `rtx6000`'s |

So the claim holds for what the family measures, with one exception: the refiner-less Spark fallback, a warning path
of a recipe nothing serves. **Proposed**: retire `h3` with no tier (branch deleted, `runpod-http.sh run` refusing it
with a pointer to `rtx6000`), and if the fallback should keep a check, one arm `sol-h3-spark-no-refiner` (the
`rtx6000` arm without `FASTVIDEO_LTX2_WEIGHTS`, gated on `sol-h3`'s trees plus the upscaler and adapter) in the
`rtx6000` tier, whose fixtures are frozen, so it would be a deliberate fixture change (§10 item 3). Not done in this
PR: the coordinator decides. The same reading applies to `ltx` once its check passes: its `ltx25-two-stage` arm is
`rtx6000`'s word for word, and its other two arms skip on the volume; it stays a tier because it is the only one
that names the LTX-2.3 and LTX-2.0 commands. **Decided** (coordinator, 2026-10-26): retired with no tier, and no
`sol-h3-spark-no-refiner` arm (a warning path of a recipe nothing serves); done in §18.

#### 17.5 The GPU checks (the coordinator's)

The procedure is §8's: one commit `SHA` for both sides (this PR's head or its merge, once
`fastvideo-rs-runtime:sha-<7>` is published with its `lab/` and fv-lab: the `ltx` side needs this PR's
`runpod-matrix.sh`, whose arms are gated), the cheapest cells that exercise the new parts, both sides in parallel,
then `fv-lab-compare-runs.sh`. One RTX PRO 6000 Blackwell Server Edition per side, EU-RO-1 on `bz0ofkfx45`, about
$2.49/hr; pod boot and preflight ~5 min per side. Every tree these cells run on is on the volume (`h3-base`,
`FastH3-4-step-Preview-v1-LoRA`, `ltx25`); the `h3-8step`, `ltx23` and `ltx2` arms are skipped on both sides (their
trees are R2 only, and nothing is fetched).

```sh
SHA=<7>
# fuse: the ungated kernel check, the 4-step VSA off / on pair (--off-identity compare and its exact gate),
# and the 8-step pair, skipped on the volume on both sides
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family fuse \
  --cells kernels-fuse,fasth3-4step-vsa-768p-off,fasth3-4step-vsa-768p-on,fasth3-8step-768p-off,fasth3-8step-768p-on --yes --wait
RUNPOD_API_KEY=… FV_FAMILY=fuse \
  FV_CELLS="kernels-fuse fasth3-4step-vsa-768p-off fasth3-4step-vsa-768p-on fasth3-8step-768p-off fasth3-8step-768p-on" \
  bash scripts/gpu/runpod-http.sh run $SHA
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/fuse/$SHA-<MMDDHHMMSS>-<hex> artifacts/runpod/fuse/$SHA-<MMDDHHMM>

# attn3: the VSA bf16-in-place pair of the cross-pipeline cell, its compare; the other two compares skip on both sides
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family attn3 \
  --cells fasth3-4step-vsa-768p-b160,fasth3-4step-vsa-768p-b161 --yes --wait
RUNPOD_API_KEY=… FV_FAMILY=attn3 FV_CELLS="fasth3-4step-vsa-768p-b160 fasth3-4step-vsa-768p-b161" \
  bash scripts/gpu/runpod-http.sh run $SHA
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/attn3/$SHA-<MMDDHHMMSS>-<hex> artifacts/runpod/attn3/$SHA-<MMDDHHMM>

# ltx: all three arms; LTX-2.5 runs, LTX-2.3 and LTX-2.0 skip on their gates on both sides
RUNPOD_API_KEY=… ./fv-lab pod up --sha $SHA --family ltx --yes --wait
RUNPOD_API_KEY=… FV_FAMILY=ltx bash scripts/gpu/runpod-http.sh run $SHA
bash scripts/gpu/fv-lab-compare-runs.sh artifacts/runpod/ltx/$SHA-<MMDDHHMMSS>-<hex> artifacts/runpod/ltx/$SHA-<MMDDHHMM>
```

Pass, for each family: `fv-lab-compare-runs.sh` exits 0 and both pods end `matrix_exit=0`. No tier here has a verdict
step, so no `--expect-verdicts` (0 verdict files on both sides), and no `FV_FETCH_TREE`: both fetches take every file
compared. Expected summary lines:

- `fuse`: "5 arm(s) the same (2 ran with byte-identical frames, 2 skipped in both, 0 failed in both; 0 with the same
  compare-dumps lines), 0 control arm(s) reported, 4 report(s) the same, 0 verdict file(s) the same, 0
  difference(s)": `kernels-fuse` the same with `exit` 0 and no frames (it writes none; a kernel `[FAIL]` exits
  non-zero: `kernels.rs` stops at the first failed `report.check(…)?` without `--keep-going`, and the report's finish
  exits `EXIT_CHECK_FAILED` = 1, so a failing kernel reads `"exit":1` in both `summary.json`, counted as "1 failed in
  both", and the compare still exits 0 because the VSA arms ran: read the line, not only the exit status), the two VSA arms
  ran (248 frames each, as `trace`'s `fasth3-4step-vsa-768p` at `b0d0bb8`), the two 8-step arms `skipped: weights incomplete`; the reports are the VSA compare (both
  `pass`, off-identity `ok` when the fusions keep every bit, as the family requires), the 8-step and the LTX compares
  `skipped` (frames missing) on both, and the VSA gate report present in both.
- `attn3`: "2 arm(s) the same (2 ran with byte-identical frames, 0 skipped in both, 0 failed in both; …), … 3
  report(s) the same, 0 verdict file(s) the same, 0 difference(s)": the b160 / b161 compare (no `--off-identity`: its
  `status` agrees; the family's own bar is bit-equal frames, readable in that report's PSNR / max difference) and the
  two unselected compares `skipped` on both.
- `ltx`: "3 arm(s) the same (1 ran with byte-identical frames, 2 skipped in both, 0 failed in both; …), … 0 report(s)
  the same, 0 verdict file(s) the same, 0 difference(s)": `ltx25-two-stage` about 242 frames (as the D1a gate's), the
  other two `skipped: weights incomplete` on both (the `SKIP … weights incomplete` lines in both `live.log`; neither
  fetch keeps the pod's `<arm>.weights.log`). The LTX-2.5 arm cannot be left out to save its ~9 min: without it no
  arm runs, and `fv-lab-compare-runs.sh` exits 1 ("no arm ran with frames in both runs: nothing was compared").

A compare verdict that fails on both sides (a fused path no longer bit-equal, cuDNN off by a bit) is a finding about
the build, not a runner difference: report it and keep the branch until it is understood.

**Estimates** ($2.49/hr; per side, boot and preflight included): `fuse` ~20 min (the kernel groups ~5 min,
UNVERIFIED: no committed run of them; two 4-step VSA arms at the recipe canvas ~4 min each, `trace`'s
`fasth3-4step-vsa-768p` took 231 s at `b0d0bb8`; the skips seconds), ~$0.85, **~$1.70** both sides; `attn3` ~14 min,
~$0.60, **~$1.20**; `ltx` ~15 min (`ltx25-two-stage` 535-542 s in the D1a gate), ~$0.65, **~$1.30**. All three about
**$4.20** for six pods, bounded by each tier's 4 h cap (pass `--wall-cap-s 3600` to the fv-lab side and
`FV_POD_CAP_S=3600` to the script side for $2.49 a pod at most). Optional, if wanted: `attn3`'s H3 dense pair
(`fasth3-4step-dense-768p-v2,…-cudnn`, the A/B the family is for, ~8 min and ~$0.35 more per side) and `fuse`'s LTX
pair (`ltx25-4k5s-sol-off,…-on`, 4k5s two-stage twice, ~20 min and ~$0.85 more per side). Stop before the balance
would fall below $8 (CLAUDE.md).

Byte identity is expected for the frames on one GPU class from one `SHA` (docs/perf/determinism.md, §12's caveats).

**Results (2026-10-26): all three pass.** `SHA=762ce4d` (main after #213) on both sides of each check: image
`ghcr.io/zaitrarrio/fastvideo-rs-runtime:sha-762ce4d` (digest `sha256:ae688277d596110669173d39ccb6458768246f31ba2a237b27cebb0712508d0e`,
build id `c7d4230782e731e1`, tools `tools-v0.1.70`), fv-lab 0.1.70 on the host, one RTX PRO 6000 Blackwell Server
Edition per side in EU-RO-1 on `bz0ofkfx45` at $2.49/hr, the commands above with `--wall-cap-s 3600` on the fv-lab
side and `FV_POD_CAP_S=3600` on the script side. The six pods ran in parallel, 12:40 to 12:51Z; every side's
`matrix.out` ends `matrix_exit=0` (the fv-lab sides' `DONE` fetched too), and all six pods were deleted (fv-lab's at its fetch, the script's at its `destroy
pod`). Both `ltx` creates found no capacity at 12:40 and were retried once, a minute later. Each preflight's
`weights.log` ends `exit=1` (`h3-8step` incomplete), as on every run on this volume. Each compare was run again from
the committed run directories and its output saved next to them with its command and exit status
(`artifacts/runpod/<family>/compare-762ce4d.txt`); each is §17.5's expected line.

- **`fuse`**: fv-lab `artifacts/runpod/fuse/762ce4d-1026124004-44af` (pod `ktao6reo6e4p1e`), script
  `artifacts/runpod/fuse/762ce4d-10261240` (pod `l83gwsp4q0vrt3`). `fv-lab-compare-runs.sh` exit 0: "5 arm(s) the same
  (2 ran with byte-identical frames, 2 skipped in both, 0 failed in both; 0 with the same compare-dumps lines), 0
  control arm(s) reported, 4 report(s) the same, 0 verdict file(s) the same, 0 difference(s)". `kernels-fuse` exit 0
  on both with 90 `[PASS]` and no `[FAIL]` lines (4 s each: the ~5 min estimate was far over). The VSA off and on
  arms 248 frames each, byte-identical across the runners; their compare `pass` with off-identity `ok` (124 pairs,
  max difference 0) on both, so the fusions keep every bit, as the family requires. The 8-step pair `skipped:
  weights incomplete` on both, its compare and the LTX compare `skipped` (frames missing). The VSA gate report is
  present on both, and its own status is `fail` on both, for the same two checks: `performance/speedup` (the warm
  denoise, fusions on against off, 18.54 to 17.13 s on the fv-lab side and 18.74 to 17.36 s on the script side,
  1.08×: over the gate policy's 1.03 experimental floor, under its 1.10 promotion bar) and `lpips_available` (no
  `--lpips` in §17.5's commands). The gate is the fusion knob's promotion decision (should the switch be promoted
  on speed and quality), not a parity or correctness verdict: parity is the compare's off-identity `ok` (and the
  gate's own `off_identity` check passed), both runners agree on the gate, and it is not a finding against the
  retirement (accepted by the coordinator, 2026-10-26).
- **`attn3`**: fv-lab `artifacts/runpod/attn3/762ce4d-1026124004-04fa` (pod `2euds1fn4gfyt8`), script
  `artifacts/runpod/attn3/762ce4d-10261240` (pod `vir27fw4n6g0vp`). Exit 0: "2 arm(s) the same (2 ran with
  byte-identical frames, 0 skipped in both, 0 failed in both; 0 with the same compare-dumps lines), 0 control arm(s)
  reported, 3 report(s) the same, 0 verdict file(s) the same, 0 difference(s)". b160 and b161 248 frames each; their
  compare `pass` with off-identity `ok` (max difference 0: VSA on bf16 activations read in place equals the f32
  widening path bit for bit, the family's own bar) on both; the two dense compares `skipped` on both.
- **`ltx`**: fv-lab `artifacts/runpod/ltx/762ce4d-1026124138-4dd8` (pod `zsz3k16tggu58v`), script
  `artifacts/runpod/ltx/762ce4d-10261241` (pod `yis61qgkwwo71c`). Exit 0: "3 arm(s) the same (1 ran with
  byte-identical frames, 2 skipped in both, 0 failed in both; 0 with the same compare-dumps lines), 0 control arm(s)
  reported, 0 report(s) the same, 0 verdict file(s) the same, 0 difference(s)". `ltx25-two-stage` 242 frames,
  byte-identical (549 s and 540 s); `ltx23-two-stage` and `ltx20-distilled-8step` `skipped: weights incomplete` on
  both (`SKIP … weights incomplete` in both `live.log`): the script's gates of §17.2 skip what fv-lab skips.

Cost: about $2.2 for the six pods (the Runpod balance $23.11 before, $20.90 after), against the ~$4.20 estimate.

#### 17.6 Then

After a family's check passes, in the next D1 PR (as §15.10): it joins `FV_LAB_RETIRED`, its `runpod-matrix.sh` branch
is deleted (the remaining families' dry runs compared before and after), its three fixtures freeze, `runpod-http.sh
run` / `plan` route it to `fv-lab pod up`, and for `attn3` `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` move from `SCRIPT_KNOBS`
to `MODE_KNOBS` (so into `runpod-http.sh`'s `FV_LAB_NOT_READ`). `h3` goes in the same PR if the coordinator accepts
§17.4; the B4 families wait for §17.3's decision. (Both went with no tier in the next PR: §18.)

**Done** for `fuse`, `attn3` and `ltx` in the PR after the checks (2026-10-26, branch `claude/d1f-retire-fuse-attn3-ltx`),
as §15.10 did for `det-short` and `determinism`:

- The evidence is committed: the six run directories and the three `compare-762ce4d.txt` (§17.5 "Results").
- All three are in `FV_LAB_RETIRED`; their `case` branches are deleted from `runpod-matrix.sh` (with them the
  `fuse_h3` / `fuse_ltx` helpers, the `[PASS]` / `/gpu_trace` log copies and the last reads of `FV_ATTN3_*`). The 27
  remaining families' dry runs are unchanged (`FV_DRY_RUN=1` output and `FV_DRY_RUN_GATES` records before and after,
  every top-level `case` label): 26 byte-identical, `writer` differing only in its `dd` timings.
- Their nine fixtures are frozen (`fv-lab-goldens.sh` no longer writes them; `tests/dry_run.rs` lists the three among
  the retired). No crate test ran these branches live (§15.10's frozen copy was for `tests/repeats_oracle.rs`), so no
  frozen copy of the script is needed: the fixtures are the reference.
- `knobs.rs`: `FV_ATTN3_ARMS` and `FV_ATTN3_VSA` move from `SCRIPT_KNOBS` (now the oracle's five only) into
  `MODE_KNOBS`, so into `not_read()` and `runpod-http.sh`'s `FV_LAB_NOT_READ` (`tests/knobs.rs` holds them equal);
  fv-lab's refusal text is unchanged (the tier is their defaults, narrow it by arm).
- `runpod-http.sh run` / `plan` route the three to `fv-lab pod up`; `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` are refused in
  `FV_EXTRA_ENV` (with a pointer to `FV_CELLS` and the default arms) and in the host environment, as
  `FV_DET_CELLS` is.
- `fv-lab-route.test.sh`: the three routed (by arm, and `ltx` whole), `FV_ATTN3_*` refused both ways, `all`
  refused, `plan` fv-lab's, `runpod-matrix.sh` refusing each (exit 2), and `h3`, `b4c`, `b4c2` still on the script
  path.
- `lab/tiers.toml` and the five cells say "was runpod-matrix.sh …".
- `scripts/gpu/runpod.sh` (`gpu` and `matrix`, the SSH drivers of the first volume fill) started `runpod-matrix.sh
  ltx` on a paid pod, which would now refuse it (coordinator, 2026-10-26: fixed here). Each family's pod command is
  `rp_family_cmd`'s: a family of `FV_LAB_RETIRED` runs its tier with the image's fv-lab (`FV_WORK=/workspace
  /opt/fastvideo-rs/target/release/fv-lab run ltx`; the runtime image ships it with `lab/` beside it,
  `docker/gpucheck.Dockerfile`), into the same `/workspace/runs/ltx` with the same `driver.log` / `pid`; without
  fv-lab in the image the pod's line stops with exit 2 and names `fv-lab pod up --family ltx`. A family with a
  `runpod-matrix.sh` branch (`h3`, `hunyuan`, `wan`) runs the script as before; anything else (an experiment tier,
  `gold`, an unknown name) stops `gpu` / `matrix` before the key is read or a pod is created
  (`rp_check_families` at the top of both). The family lists are constants (`RP_GPU_FAMILIES`, `RP_MATRIX_FAMILIES`):
  `gpu` creates and starts one pod per family of its list, and `runpod.sh plan gpu|matrix` prints each pod's command
  from the same lists (and fails on a family nothing can run). `scripts/gpu/tests/runpod-plan.test.sh` (in
  `control.yml`) sources the script (`FV_SOURCE_ONLY=1`, as `docker.sh` and `validate.sh`) with its Runpod, ssh and
  rsync calls stubbed and drives the real `cmd_gpu` and `cmd_matrix`: `ltx`'s command is `fv-lab run ltx` and no
  recorded command, over ssh or local, starts `runpod-matrix.sh` for a retired family, while `h3`, `hunyuan` and
  `wan` start it as before. *Per-arm cap:* `gpu`'s `ltx` arms ran under `run_cell`'s default 300 s (no
  `FV_GEN_TIMEOUT_S` on that path) and now under the tier's `cell_timeout_s = 3600`, a fix: `ltx25-two-stage` takes
  about 540 s. `matrix` exports `FV_GEN_TIMEOUT_S=300`, which fv-lab reads as its per-arm cap and which would kill
  `ltx25-two-stage`, so `matrix` leaves it out for a retired family (the tier's cap applies) and keeps it for the
  script families.

### 18. D1g: the finished families retired with no tier

Status: accepted 2026-10-26 (coordinator decision of 2026-10-26, executed in the PR after #214). Decision-id
FVID-2026-10-26-d1g-retire.

Eleven `runpod-matrix.sh` families leave the script without becoming tiers: their GPU evidence is committed or
recorded, another tier covers them, or what they ran on is gone. Nothing runs them afterwards.

#### 18.1 The decisions (coordinator, 2026-10-26)

| family | decision | why, and what replaces it |
|---|---|---|
| `b4b`, `b4c`, `b4c2` | §17.3 option (b): retired as finished experiments | Their GPU gates passed when their changes merged (`b4b` #179, `b4c` #177, `b4c2` #178; `artifacts/runpod/{b4b,b4c,b4c2}/`). A later change to the fusions or the prompt cache gets its own byte-identity run (the `b4a` / `techniques` tiers' baseline-vs-this-build shape) plus the CPU fixtures (`rt/prompt_cache/equality.rs`, `rt/fuse/equality.rs`). No `[[verdict.check]]` slice (option (a)). |
| `h3` | §17.4's proposal: retired with no tier | `rtx6000` runs every arm but the refiner-less Spark fallback (a warning path of a recipe nothing serves) and the cold streamed-encoder runs (§17.4's table). **No** `sol-h3-spark-no-refiner` arm, so `rtx6000`'s frozen fixtures are unchanged. The refusal points at the `rtx6000` tier and names those gaps; the `h3` commands are in `git show b019ccb2:scripts/gpu/runpod-matrix.sh`. |
| `hunyuan`, `sana-video` | §14.7: retired with no tier | Nothing runs them: their trees are not on the EU volume and no variant or gold file serves them. Their arms by hand: `docs/ports/sana-video.md` keeps the `fv-gpucheck` command per arm, and both branches are in `git show b019ccb2:scripts/gpu/runpod-matrix.sh`. |
| `b200` | §14.7: retired with no tier | The US volume it ran on is gone (CLAUDE.md, owner decision 2026-10-06). `runpod.sh b200`, which created a US volume and ran `runpod-matrix.sh b200` on a B200 pod, is refused too (exit 2) and `cmd_b200` is deleted with the three helpers only it used (`rp_pick_us_b200_dc`, `rp_create_us_volume`, `rp_log_box_image`) and its two settings (`RP_B200_VOL_NAME`, `RP_B200_DESTS`); `runpod.sh offers` keeps `RP_B200_TYPE` / `RP_US_DCS`. |
| `writer` | §14.7: retired with no tier | A one-time measurement (the frame writer behind a timed decode, commit `a8ecfec6`). Checked: `docs/perf` records no result of it and no run directory is committed (no `artifacts/runpod/writer/`), so there is no line to point at; the change it measured is described in `a8ecfec6`'s message, and the family's commands are in `git show b019ccb2:scripts/gpu/runpod-matrix.sh`. |
| `precision-debug` | §14.7: retired with no tier | The `precision` tiers run the same precision variants (§14.7), but not its per-step CUPTI trace (`FASTVIDEO_GPU_TRACE_STEP`) or its `FASTVIDEO_DUMP_DIR` + `compare-dumps` diffs; those by hand from `git show b019ccb2:scripts/gpu/runpod-matrix.sh`, which the refusal names. |
| `identity` | §14.7: retired with no tier | The `det-short` / `determinism` tiers cover byte identity; its E12 results stay in `artifacts/runpod/identity/` (`docs/gaps/2026-09-27-cold-start.md`). |
| `h3attn` | §14.7's rule: "retire unless an attention arm is an open item (`docs/techniques.md`)": **retired** | `docs/techniques.md` "H3 attention arms" records every arm's numbers and verdict (the served variants' in their gold files) and lists no open arm. Its one forward item, "the next step is INT8 `Q K^T` … and then the end-to-end gate" after the FP8 arms were dropped, is closed in the same section: SageAttention2 was ported (`wan::attn_sage`), merged default-off, and its calibrated H3 gate failed both H3 recipes, so H3 stays bf16 (Phase 4, `docs/perf/sage-attention.md` §8). Neither the execution plan nor this ADR lists an `h3attn` arm as open. The `profiles/h3/*` arms stay (built-in profiles, `crates/fastvideo-models/src/techniques/builtin.rs`); `attn-fp8-kernels` (`fv-gpucheck kernels --groups attn_fp8`) is a kernels group anyone can run. |

#### 18.2 What was done

As §15.10 and §17.6, with one difference: these families have no tier to route to.

- **A separate list.** `FV_LAB_RETIRED` means "fv-lab only": `runpod-http.sh run` / `plan` call `fv-lab pod up
  --family <f>`, `runpod.sh gpu` / `matrix` run `fv-lab run <f>`, and `tests/dry_run.rs` / `fv-lab-goldens.sh`
  require each entry to be a tier with frozen fixtures. A family with no tier there would be routed to a tier that
  does not exist. So `scripts/gpu/fv-lab-retired.sh` gains `FV_SCRIPT_RETIRED` (the eleven, the one place they are
  marked) and `fv_script_retired_why <family>` (what replaces each, the text of every refusal).
- **Refused, exit 2, never routed**: `runpod-matrix.sh <f>` (its fallback branch, after the restored-mode and
  fv-lab checks); `runpod-http.sh run` / `all` / `plan` with `FV_FAMILY=<f>` (before the key check, a lookup or a
  pod, with or without `FV_DRY_RUN=1`; `kernels` / `weights` / `upstream` are family-independent and unchanged);
  `runpod.sh gpu` / `matrix` / `plan` for a list holding one (`rp_family_cmd`, before the key or a pod); `runpod.sh
  b200`.
- **The branches.** The eleven `case` branches are deleted from `runpod-matrix.sh` (907 lines; with the
  `b4-verdict.sh` source line and the header, 914 out, 11 in), with the helpers only they used, all local to the
  branches (`id_hash`, `arms_cell`, the B4 cache and baseline-fetch setup, `writer`'s sampler). The shared frame
  (`run_cell`, `gated_cell`, `compare_cells`, `gate_cells`, `frames_digest`, the TAE / LPIPS directories, the oracle
  helpers) stays: the remaining families use every part of it.
- **`b4-verdict.sh` and `tests/b4-verdict.test.sh` deleted.** After the branches nothing reads them: `tests/verdict.rs`
  runs `b4a`'s verdict against its own frozen copy (`fixtures/b4a-script/b4-verdict.sh`, §11.6), and the `det-short`
  / `determinism` copy in `fixtures/det-script/` carries one too (§15.10). `tests/repeats_oracle.rs` copied the live
  file only because the live script sourced it: it now copies three files for the live `oracle` branch and four for
  the frozen copy. No crate test ran the eleven branches live, so no frozen copy of them is needed.
- **Fixtures.** None: no family here was a tier, so none had a dry-run, gates or payload golden to freeze. The GPU
  evidence stays where it is (`artifacts/runpod/{b4b,b4c,b4c2,identity}/`).
- **Knobs.** None moved. `knobs::SCRIPT_KNOBS` holds only the oracle's five (no tier ran these families, so their
  knobs were never in fv-lab's lists), and `knobs.rs` and `runpod-http.sh`'s `FV_LAB_NOT_READ` are unchanged and
  still equal (`tests/knobs.rs`). The knobs read only by the deleted branches (`FV_B4B_CELLS`, `FV_B4C_CELLS`,
  `FV_B4C2_CELLS`, `FV_ID_CASES`, `FV_ID_EVICT`, `FV_SANA_*`, `FV_WRITER_4K`, `FV_H3ATTN_*`, `FV_ARM_TIMEOUT_S`) are read
  by nothing now; a run that would set them is refused by its family.
- **`runpod.sh gpu` / `matrix`** (the SSH drivers of the first volume fill): `RP_GPU_FAMILIES=(ltx)` and
  `RP_MATRIX_FAMILIES=(ltx wan)` (were `h3 ltx hunyuan` and `h3 ltx hunyuan wan`), the pod-per-family behaviour kept for
  what remains; the dead `hunyuan`-to-`hy` pod-name mapping and `matrix`'s `h3`-only `aot_sms` probe went with them.
  `tests/runpod-plan.test.sh` drives the real `cmd_gpu` / `cmd_matrix` on stubs as before: one `ltx` pod (7200 s
  backstop) running `fv-lab run ltx`, `matrix` running `ltx` through fv-lab and `wan` through the script, no recorded
  command starting `runpod-matrix.sh` for any family of `FV_LAB_RETIRED` or `FV_SCRIPT_RETIRED`, each of the eleven
  refused before the key or a pod in either list, and `runpod.sh b200` refused without a key.
- **Dry runs compared** (§10 item 1): `FV_DRY_RUN=1` output (time stamps stripped, the temporary directory replaced by
  a placeholder), `FV_DRY_RUN_GATES` records and exit status, every remaining top-level `case` label, before
  (`b019ccb2`) and after, the same environment (`FV_BASELINE_SHA`, `FV_ORACLE_URL`, `FV_ID_CASES` set so every
  family's dry run reaches its commands): all **16 byte-identical** (`mmaudio`, `speechtest`, `sfwan`, `sfstream`,
  `sfquality`, `headline`, `cold`, `wan`, `eval`, `oracle`, `ltxoffload`, `solbench`, `hd`, `serve-engine`,
  `sol-lingbot`, `sol-cosmos3`; `mmaudio` and `speechtest` exit 1 at the same `find` of a dry run's missing
  directory on both sides, as before). `writer`, which differed in its `dd` timings in §15.10 and §17.6, is gone.
- **Tests.** `fv-lab-route.test.sh`: `FV_SCRIPT_RETIRED` is the eleven; each has a pointer, is no tier and in no
  other list; `runpod-http.sh run` (dry run and without a key), `all` and `plan` refuse each (exit 2) with its
  pointer; `runpod-matrix.sh` refuses each (exit 2); a fake `FV_LAB_BIN` is never called for one. `fv-lab-goldens.sh`
  checks the list too (no tier, no branch, in no other list). `tests/dry_run.rs`
  (`families_retired_with_no_tier_have_no_branch_and_no_tier`) holds the same from Rust: the list, no branch, no
  tier, no fixture, in neither `FV_LAB_RETIRED` nor `FV_LAB_EXPERIMENTS`.
- **Review of #215** (independent, nothing blocking): the pointers of `h3` and `precision-debug` name what `rtx6000`
  and the `precision` tiers do not run (the refiner-less Spark fallback and the cold streamed-encoder runs; the
  per-step CUPTI trace and the dump diffs), and `hunyuan` / `sana-video`'s say nothing runs them and how to run their
  arms by hand; the `all` refusal test runs without a key. `runpod-http.sh` now removes its private copy
  (`$TMPDIR/runpod-http.XXXXXX`, which every run left behind) at exit and before `exec fv-lab`, only the copy it was
  started as (`FV_HTTP_COPY` holds its path; run in place, it removes nothing); `fv-lab-route.test.sh` checks both.

#### 18.3 What is left in the script

Sixteen families: `mmaudio`, `speechtest`, `sfwan`, `sfstream`, `sfquality`, `headline`, `cold`, `wan`, `eval`,
`oracle` (whole, §15.10), `ltxoffload`, `solbench`, `hd`, `serve-engine`, `sol-lingbot`, `sol-cosmos3`. Each is a later
slice's (a tier with `hydrate`, a prompt-file arm, a derived-clip step, the `fv-lab oracle` driver) or §14.7's open
proposals (`mmaudio`, `speechtest`, `sfstream`, `sfquality`, the `sol*` families).

### 19. D1h: the `fv-lab oracle` driver

Status: proposed 2026-10-26 (draft PR from `claude/d1h-oracle-driver`), CPU-tested; its GPU check is the
coordinator's (§19.7). Decision-id FVID-2026-10-26-fv-lab-oracle-driver. This is §15.6's driver ("upstream pods up, the
runtime tier with `FV_ORACLE_URL`, release on `ORACLE_REFS_DONE`") and the first of the two things §15.10 says must
exist before `oracle`'s branch retires. The second, the clip-metric targets and `sfwan13` as cells, is **not** in
this slice: it needs a derived-clip / metric step the ADR has not designed (§19.9). Nothing is retired here.

#### 19.1 What it does

`fv-lab oracle --sha <7> --cells <arms…> [--yes]` is what `scripts/gpu/oracle.sh` and §15.7's hand-run commands do,
on fv-deploy's guards (`oracle::drive` on the `OracleOps` trait; the binary's world is `oracle_live::LiveOps`):

1. **The upstream pods up**, one per stack of `lab/upstream.toml` (§19.2) that makes a selected target, in the
   file's order and one after the other, each with fv-deploy's `Deployer::up` (kept), as `fv-lab pod up` does: the
   price cap on the quote and the billed price, the $8 floor projected over the cap, the EU volume `bz0ofkfx45` in
   EU-RO-1 only, the detached backstop at the stack's wall cap, ready (its file server answering `GET /upstream/`)
   within 600 s in at most 3 cycles, the stack's host-RAM floor (`minRAMPerGPU`; 192 GB for the FastVideo stack,
   docs/oracle.md "Upstream cells: host RAM"), a live pod of the same name refused, the image pinned to its digest.
   Each is recorded (`artifacts/fv-lab/pods/<id>.json`, `artifacts/runpod/upstream/<tag>/pod.json`) and written to
   fv-deploy's ledger; a record that cannot be written deletes the pod just created before the error
   (`pod::or_delete`, `fv-lab pod up` too). A pod that does not come up releases those already up and fails the
   run.
2. **The references waited for**, polling every 60 s each selected target's `<pod>/upstream/<tag>/oracle-<target>/ORACLE_DONE`
   (the hand-run check's step 2) and fetching each pod's `keepalive.bin` meanwhile (its idle watchdog counts that
   traffic as busy, `runpod-http.sh idle_watchdog net`), until every target is served, or its pod's run wrote `DONE`
   without it, or the pod is gone, or `--refs-wait-s` passed **since the first create** (default: what the upstream
   cap leaves after the stock wait and the 30-minute fetch reserve, 10,500 s at the 4 h cap; counted from before
   the first create, so no pod's cap, counted from its own later create, can pass first). No reference served:
   every pod released, the run fails, no runtime pod is made. A pod none of whose targets was served is released at
   once; the runtime side runs the served targets' arms only (a missing reference still fails the run).
3. **The runtime side**: `fv-lab pod up --sha <sha> --family oracle --tag <tag> --cells <the served targets' arms>
   --extra-env FV_ORACLE_URL=<the serving pods' bases, comma separated> [--extra-env …] --wall-cap-s <cap> --yes
   --wait` as a child process in its own process group (its stdout to stderr): the tier's pod, run, fetch, verdict
   and delete are `pod up`'s, unchanged (and gated in §15.7). `--url-file` writes the bases first. A tag whose run
   directory exists is refused (another run's `pod.json` would be read as this one's).
4. **The release**, polling every 60 s with the keepalive. The upstream pods are released (fetched, then deleted):
   - once the runtime pod (its `pod.json`, written when it is ready) serves `ORACLE_REFS_DONE` (every reference
     fetched, §15.2 item 8);
   - 1,800 s after the runtime pod was ready without it (`REF_FETCH_RESERVE_S`: the runner's ~5-minute preflight,
     then the downloads; §15.7's 1.3 GB H3 reference took under a minute), so a run that never publishes it does not
     keep them;
   - when no runtime pod is ready within `--stock-wait-s` (default 2,100 s: fv-deploy's 3 × 600 s readiness and 5
     minutes): the child is sent SIGTERM (it deletes what it made), the upstream pods are released, then the child
     is waited for (at most `ABORT_WAIT_S`, 600 s; past it, SIGKILL and its pod deleted by the driver);
   - when the child ends.
   An interrupt (SIGINT / SIGTERM) sends the child SIGTERM, **deletes** the upstream pods without fetching them (a
   fetch may take 900 s a file), then waits for the child as above. A child that ends in failure while its pod (its
   `pod.json`) is still live, killed say, has that pod deleted by the driver.
5. **Release** is: fetch the upstream run (`pod fetch --fetch-tree`'s tree fetch with `oracle.sh`'s skip
   `oracle-dump\.tar|/dump/|\.mp4$`) into `artifacts/runpod/upstream/<tag>/`, then delete the pod with fv-deploy's
   retries and ledger. A failed fetch is reported and does not stop the delete; a failed delete fails the run (the
   backstop still deletes at the cap). Every path through the driver ends with every upstream pod it created deleted
   (`tests/oracle_driver.rs`, `tests/oracle_live.rs`).

`--hold-upstream FILE` keeps the upstream pods, after the first two release triggers of step 4 or after a
**successful** runtime side, until FILE exists (at most until their cap less 15 minutes after the first create), for
another reader of the same references: the GPU check's script side (§19.7). The stock wait, a failed start, a failed
runtime side and an interrupt release at once.

The report (stdout, JSON) names the served and missing targets, each upstream pod (stack, id, tag, why released,
fetch and delete), the runtime run's tag, its pod once ready, its result and exit code; `fv-lab oracle` exits 0 on a
clean run, the runtime side's own exit code when it failed with one, else 1.

Before anything is created the driver refuses (`oracle::plan_oracle`, then `pod up`'s own checks of the runtime
launch): a tier with no `[reference]` cell, no `--cells`, an arm not of the tier, a selected cell without a reference,
a target no stack makes, `FV_ORACLE_URL` (or any knob `pod up --extra-env` refuses) in `--extra-env`, a cap raised
above the tier's or the stack's, a stock wait of 0 or not below the runtime cap, a refs wait that does not fit the
upstream cap, a `--sha` that is not 7 hex, an upstream commit that does not resolve to 40 hex, an image tag other
than `latest` / `sha-<7>`, a reused tag, a launch fv-deploy's guards or region check refuse. With `--yes`, before the
first create, the account's balance less what its pods already spend over the run's longest cap less every pod of the
run at its price cap over its wall cap and two replaced readiness cycles (2 × 600 s) must stay at $8 or above
(`oracle::check_balance`; fv-deploy's per-create projection counts the upstream pods' spend when the runtime pod is
created too).

Without `--yes` it prints the plan and needs no key: each upstream pod's `Launch` and `POST /pods` body, the runtime
pod's (with `upstream-pod` standing for the upstream pods' ids, known once they exist) and its `pod up` command, the
waits and the money projection.

#### 19.2 `lab/upstream.toml`: the upstream stacks (a schema of its own, v1)

```toml
schema = 1
[[stack]]
name = "fastvideo"            # the pod's stack
image = "fastvideo"           # ghcr.io/zaitrarrio/fastvideo-rs-upstream-<image>:<tag>
gpu = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
gpu_count = 1
cloud = "SECURE"
region = "eu"
max_usd_per_hr = 2.5
wall_cap_s = 14400
container_disk_gb = 150
min_ram_gb = 192
cell_timeout_s = 7200         # UP_CELL_TIMEOUT_S
env = ["FASTVIDEO_DUMP_OPS=0,1,24,47"]   # the pod's FV_EXTRA_ENV
[[stack.target]]
name = "fasth3-4step-vsa"     # oracle-<target> on the pod; a cell's [reference] target
steps = ["weights:h3-diffusers"]
```

Two stacks, the values of `oracle.sh`'s `start_up` calls: `fastvideo` (the five H3 targets; 150 GB; the 192 GB
floor) and `sol-ltx25` (the four LTX targets; 250 GB; `RECON_ACCEPT_MISMATCH=1`; no floor, none measured). Per
target, its weight steps: `weights:fasth3-8step weights:h3-diffusers` for the two 8-step targets (oracle.sh's union
for any `fasth3-*` target), `weights:h3-diffusers` for the two LoRA targets (§15.7's `UP_STEPS`), none for
`h3-ref2va-4step` (`oracle_ref2va` builds the view itself), `weights:ltx25` for the LTX ones. `UP_STEPS` is
`info:box`, the selected targets' steps once each in the stack's target order, `oracle:<targets>`
(`oracle::up_steps`).

Why a file of its own and not tiers: an upstream stack has no cells and does not run `fv-lab run` (its pod runs
`scripts/gpu/upstream/pod.sh` from a clone of the repository), and `lab/tiers.toml` has two line-based readers that
would take a stack's `name =` for a tier's (`fv-lab-goldens.sh`'s tier list, `runpod-http.sh tier_value`). serde with `deny_unknown_fields`, the schema
`schemas/upstream.v1.json` (drift test `tests/schema.rs`), `fv-lab check` and `fv-lab schema upstream`; validation as a
tier's (EU only, one GPU, a cap within 4 h, a positive floor), plain `K=V` env values, `weights:<view>` steps, a
target in one stack only, and against the cells (`Stacks::check_against`): every stack target is some cell's
`[reference] target` and every cell's reference has a stack. No tier may be named `upstream` (the upstream pods'
family: their pod names, run directories and `pod reap --account`'s pattern; `Tiers::validate`). No existing file
changes; the tiers and cell schemas are untouched.

#### 19.3 The upstream pod: `runpod-http.sh upstream`'s, and its payload golden

`oracle::upstream_launch` is the stack's GPU, cloud, disk, floor and caps, the EU volume at `/workspace` (the
references read `/workspace/weights`), ports `8000/http`, `dockerStartCmd ["/bin/bash", "-c", <start command>]`. The
start command is `src/upstream-start.sh.in`: `runpod-http.sh upstream_start_cmd` with `idle_watchdog net`, generated
from the script with its variables as markers (`@TAG@`, `@UP_SHA@`, `@UP_STEPS@`, `@UP_CELL_TIMEOUT@`,
`@EXTRA_ENV@`, the idle settings, `@CAP_S@`). `UP_SHA` is the full commit the pod clones (`--up-sha`, default `--sha`,
resolved with `git rev-parse`); the image tag is `--upstream-image-tag`, default `latest` (what §15.7 used: the
upstream images are built only when `scripts/gpu/upstream/**` or `docker/upstream*` change, so
`sha-<runtime sha>` usually does not exist), pinned to its digest at create.

`runpod-http.sh plan-upstream [sha] [tag]` (added here: the body `upstream` would send, nothing created, `UP_SHA`
settable; the image choice factored out of `cmd_run` into `upstream_image`, unchanged) writes, through
`fv-lab-goldens.sh`, `fixtures/runpod-http-upstream-{fastvideo,sol-ltx25}.json` with `oracle.sh`'s settings for one
selection per stack (`fasth3-4step-vsa`; `ltx25-512p,ltx25-512p-dense`). `tests/oracle_driver.rs` holds fv-lab's body
to them, field by field, except:

1. the start command carries the pod's own wall-cap watchdog, the block every fv-lab pod has (`pod-start.sh.in`,
   ADR-0023 §4), right after `set -u`; the script's upstream pod has only the driver's `nohup sleep; curl`
   backstop;
2. `env: {}` (fv-deploy's `create_body` always sends it);
3. at create only, the image pinned to its digest.

The name is equal (`fv-lab-upstream-<tag>`, the golden made with `FV_POD_PREFIX=fv-lab`); fv-lab's tag is
`<sha>-<MMDDHHMMSS>-<4 hex>` per pod, so two stacks never share a run directory. `pod reap --account` takes
`fv-lab-upstream-<sha7>-<stamp>-<hex>` pods past the longest stack cap.

#### 19.4 Differences from `oracle.sh`, and why

| | `oracle.sh` | `fv-lab oracle` | why |
|---|---|---|---|
| targets | `ORACLE_TARGETS` (default the six H3 / LTX tensor-diff targets) | `--cells`, by arm | arms are the unit (§9.4, §15.5) |
| upstream pods' guards | `runpod-http.sh`'s: $5/hr, 1,200 s × 3 readiness, `nohup` backstop, no balance check | fv-deploy's: $2.50/hr, 600 s × 3, the detached backstop, the $8 floor, a combined projection of every pod | §7; the owner's rules |
| upstream pod cap | the driver's backstop only | also the pod's own cap watchdog | ADR-0023 §4 |
| image tag | `sha-<sha>` unless `UP_IMAGE_TAG` | `latest` unless `--upstream-image-tag`, pinned to a digest | the tag that exists; the run names the image it ran |
| `FV_MIN_RAM_GB` | the caller's environment, on every pod it starts | the stack's `min_ram_gb` (the runtime pod: the tier's, none) | a floor where the evidence is (docs/oracle.md) |
| weight steps | `weights:fasth3-8step weights:h3-diffusers` for any `fasth3-*` target | per target (the LoRA targets: `h3-diffusers` only) | the 8-step view reads `h3-8step`, which the EU volume lacks; §15.7 ran without it |
| runtime pod start | at once (or `ORACLE_WAIT_FIRST=1`: after the first target) | after every selected reference is served (or failed) | §15.7's order: the runtime pod never idles while references are made |
| a reference not served | its target's cell waits `FV_ORACLE_WAIT_S` on the runtime pod, then skips | its arms are not run; the run fails | no GPU pod waits for a reference that will not come |
| stock wait | `FV_RUNTIME_STOCK_WAIT_S` 900 s: the upstream pods go, the runtime driver goes on | `--stock-wait-s` 2,100 s (fv-deploy's 3 × 600 s readiness and 5 min): the runtime side is stopped too | fv-deploy fails a create without stock at once, so the wait bounds readiness only |
| bring-up | the stacks' pods in parallel (background drivers, 3 create attempts each) | one after the other (up to 3 × 600 s each), the references' deadline counted from the first create | fv-deploy's lifecycle is one blocking `up` per pod; the deadline keeps every pod inside its cap |
| a stack's pod does not come up | that stack's targets dropped, the others run | every pod already up released, the run fails | a partial oracle run is a failed one; a rerun with fewer `--cells` is explicit |
| readiness of a cold image pull | 1,200 s × 3 | 600 s × 3 (the owner's rule) | **risk:** a host pulling the upstream image cold (tens of GB; §15.7's estimate said up to 15 min) may not be ready in 600 s; fv-deploy then replaces the pod (a few cents each) and, after 3 cycles, fails the run. §15.7's pod was ready 3.5 min after create (the image cached on the host); kept, not raised |
| `ORACLE_REFS_DONE` never published | the upstream pods kept (keepalive) until the runtime pod ends | released 1,800 s after the runtime pod was ready | a money bound |
| interrupt | the trap deletes the upstream pods; the runtime driver is left to its own | SIGTERM to the runtime side, the upstream pods deleted without a fetch, the runtime side waited for (600 s, then killed and its pod deleted) | nothing outlives the driver but what a backstop deletes |
| upstream fetch | when the upstream run wrote `DONE` (before the release) | at the release | the dumps are served until then; the same tree and skip |
| `UP_AFTER`, `FV_ORACLE_RUN_MODE=all`, `FV_ORACLE_FAMILY`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT`, `FV_ORACLE_EXTRA_ENV` | yes | no (`--extra-env` for the runtime run, its refusals `pod up`'s) | not used by the tier's cells; `FV_ORACLE_F32` / `OWN_TEXT` are the tier's defaults (§15.5) |
| clip-metric targets, `sfwan13` | yes (`runpod-matrix.sh oracle`) | the ten LTX clip-metric targets since §20 (the tier's extra cells); `sfwan13` not yet (§20.5) | |

#### 19.5 Code and tests

`src/oracle.rs` (stacks, the upstream launch and start command, `plan_oracle`, `check_balance`, `drive` on the
`OracleOps` trait), `src/oracle_live.rs` (`LiveOps`: fv-deploy's `Deployer` on a `Provider`, the pods through an
`Http`, the runtime side as a child `pod up`, every seam a parameter), `src/upstream-start.sh.in`,
`src/bin/fv-lab.rs` (`fv-lab oracle` on Runpod and reqwest; `check` and `schema upstream`; `reap --account`, which
fails on an unreadable `lab/upstream.toml`), `pod.rs` (`write_record`, `or_delete`), `lab/upstream.toml`, `schemas/upstream.v1.json`; `scripts/gpu/runpod-http.sh plan-upstream`,
`scripts/gpu/tests/fv-lab-goldens.sh` (the two upstream goldens). The runtime image ships `lab/upstream.toml` beside
`lab/tiers.toml` and `lab/cells` (`docker/gpucheck.Dockerfile`; its smoke step's `fv-lab check` reads it, and the
image and `control.yml` workflows' path filters name it); `tests/fv_lab_bin.rs::the_runtime_images_lab_subset_passes_check`
copies exactly what that stage of the Dockerfile copies into a root of its own and runs `check` and a dry run there.

`tests/oracle_driver.rs` (21): the payload golden and the start command; the stacks file against the cells, its
refusals and no tier named `upstream`; the plan (one pod per stack, the tier's order, the caps, the waits, the
projection) and its 14 refusals; the balance guard; and the driver on a fake world whose clock is its sleeps: the
order (up, keepalive while the reference is made, the bases published, the runtime side with them, keepalive, fetch
then delete, nothing after), the release on `ORACLE_REFS_DONE` before the runtime side ends, at the runtime side's
end without it, 1,800 s after the runtime pod was ready without it, on the stock wait (stop, release, then wait), on
the refs deadline counted from the first create (two stacks of 1,500 s bring-up each), on an upstream run that ends
without its reference (that pod released before the runtime side, which runs the other stack's arms against the
other pod only), on a gone pod, on a second stack that does not come up (the first released; the first failing
creates nothing more), on a start failure, on an interrupt (stop, delete without fetch, wait; during the wait and
during the run), with a failed fetch (the delete still runs) and a failed delete (the run fails), and the hold
(lifted after `ORACLE_REFS_DONE`, after the runtime side's end, never: its limit; not on the stock wait; not after a
failed runtime side). `tests/oracle_live.rs` (7): `LiveOps` against a fake provider (fv-deploy's `Deployer` creates
and deletes through it), a fake HTTP and a fake `fv-lab` script: an upstream pod recorded (ledger, records, digest);
one whose record cannot be written deleted; a runtime side stopped with SIGTERM and waited for, its still-live pod
deleted after its failure, its arguments carrying the run's root and state; one that ends well left alone, one that
fails (exit 3) with its pod deleted; a reused tag refused; the URL and hold files; the reference and run state
through the proxy. `pod.rs`: `or_delete`. `tests/fv_lab_bin.rs`: the plan through the binary without a key, five
refusals, `--yes` without a key, `check`'s stack lines, and the runtime image's `lab/` subset (the Dockerfile's
`COPY` lines of the stage that ships fv-lab, in a root of their own) passing `check` and the smoke's dry run.

Mutations each caught by a test (applied one at a time, run, reverted; review of #216): no SIGTERM in
`runtime_stop` (the fake child never sees it, is killed, the test fails); the record error returned without the
delete; the stock wait waiting for the runtime side before the release; a hold kept after a failed runtime side;
the references' deadline counted after the bring-up; a failed runtime side's pod not deleted; `or_delete`
deleting nothing; an interrupt fetching before the delete.

#### 19.6 What does not change

The `oracle` tier, its cells and fixtures, `fv-lab run`, `fv-lab pod up` (the driver calls it), `oracle.sh`,
`runpod-matrix.sh` (the branch stays whole, §15.10) and every other tier's pod body.

#### 19.7 The GPU check (the coordinator's)

Both runtime runners on **one** reference, as §15.7, with the upstream pod now `fv-lab oracle`'s: the driver brings
it up, publishes its base (`--url-file`) and holds it (`--hold-upstream`) until the script side, reading the same
base, is done. That exercises every part of the driver (the upstream pod on fv-deploy's guards and its floor, the
wait for `ORACLE_DONE`, the keepalive, the runtime side with `FV_ORACLE_URL`, the release on `ORACLE_REFS_DONE` and
the hold, both fetches, the deletes) on the cheapest target, `fasth3-4step-vsa`, whose runtime side §15.7 already
proved equal. No clip-metric target is in it: none is a cell yet (§19.9); its check comes with them.

The commit `SHA` must be on GitHub (the upstream pod clones it) with a published runtime image
`fastvideo-rs-runtime:sha-<7>` (any commit after #202 carries the oracle cells); `fv-lab` is this PR's, built on the
build pod (docs/dev/fv-lab.md "Build"). RTX PRO 6000, EU-RO-1, on `bz0ofkfx45`.

```sh
SHA=<7>
CELLS=oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32
D=/tmp/fv-oracle-check; rm -rf $D; mkdir -p $D
# fv-lab side: the upstream pod (fastvideo stack, 192 GB floor), the reference, the runtime pod, the release
RUNPOD_API_KEY=… ./fv-lab oracle --sha $SHA --cells $CELLS --upstream-wall-cap-s 7200 --wall-cap-s 5400 \
  --url-file $D/urls --hold-upstream $D/hold --yes >$D/report.json 2>$D/lab.log & LAB=$!
# script side: the same reference, runpod-http.sh's oracle family (as §15.7), then lift the hold whatever happened
until [ -s $D/urls ] || ! kill -0 $LAB 2>/dev/null; do sleep 20; done
if [ -s $D/urls ]; then
  RUNPOD_API_KEY=… FV_FAMILY=oracle FV_GEN_TIMEOUT_S=5400 FV_POD_CAP_S=5400 \
    FV_EXTRA_ENV="FV_ORACLE_URL=$(cat $D/urls) FV_ORACLE_TARGETS=fasth3-4step-vsa FASTVIDEO_DUMP_OPS=0,1,24,47" \
    bash scripts/gpu/runpod-http.sh run $SHA >$D/script.log 2>&1
fi
touch $D/hold; wait $LAB; echo "fv-lab oracle rc=$?"
L=artifacts/runpod/oracle/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/oracle/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh $L $S
jq '{served, missing, runtime_ok, runtime_exit_code, runtime_pod, problems, upstream: [.upstream[] | {id, tag, released, deleted}]}' $D/report.json
U=artifacts/runpod/upstream/<the upstream tag in report.json>
jq '{exit, oom_kills, peak_ram_mib}' $U/oracle-fasth3-4step-vsa/cell.json; grep pod_ram_limit_mib $U/box.txt; jq .min_ram_gb $U/pod.json
grep -E 'runpod-pod-(created|deleted)' artifacts/fv-lab/ledger.tsv | tail -4   # both fv-lab pods created and deleted
```

**Pass:**

- `fv-lab oracle rc=0`; the report: `"served": ["fasth3-4step-vsa"]`, `"missing": []`, `"runtime_ok": true`,
  `"runtime_exit_code": 0`, `"runtime_pod": {"id": …}`, `"problems": []`, one upstream entry whose `released` is
  `"the runtime side ended; the hold is lifted"` (or `"the runtime pod has every reference; the hold is lifted"` if
  the script side finished first) with `"deleted": {"Ok": null}`; `lab.log` has `reference fasth3-4step-vsa served by <id>` and `FV_ORACLE_URL=https://<id>-8000…`.
- `fv-lab-compare-runs.sh` exit 0 with §15.7's line: "5 arm(s) the same (2 ran with byte-identical frames, …; 3 with
  the same compare-dumps lines), … 0 difference(s)"; both runtime pods `matrix_exit=0`, both `live.log`s
  `oracle: fasth3-4step-vsa reference 1.3G`.
- The upstream run fetched to `artifacts/runpod/upstream/<tag>/` without dumps or clips: `cell.json` exit 0 and
  `oom_kills` 0, `box.txt` a `pod_ram_limit_mib` of at least 183,105 (192 GB), `pod.json` `"min_ram_gb": 192`.
- The ledger shows the upstream and the fv-lab runtime pod created and deleted; `fv-lab pod reap --account` lists
  nothing.

**Estimate** ($2.49/hr; the timings of §15.7's run): the upstream pod ~30-45 min (ready 3.5 min after create in
§15.7's run; the `h3-diffusers` view and the reference ~10 min; then held while both runtime sides fetch and run,
~20 min), ~$1.90 at most; each runtime pod ~20 min (5 min boot and preflight, two H3 runs with dumps, three
`compare-dumps`), ~$0.85 each; **about $3.60** in all. Readiness is fv-deploy's 600 s per cycle: a host that pulls
the upstream image cold may need longer, and fv-deploy then replaces the pod (each replaced cycle up to 600 s, about
$0.40) and fails after 3 (§19.4); rerun then. fv-lab's balance guard needs the balance at $18.42 or more for its two
pods at these caps with their replaced cycles ($5.83 + $4.58 + the $8 floor), plus what the account already spends
over the 2 h cap (a running build pod at $0.96/hr adds $1.92); with the script side's pod, start at $25 or more. Stop
before the balance would fall below $8 (CLAUDE.md).

#### 19.8 Then: `oracle.sh` on the driver, and the retirement of `oracle`

After the check passes, in the next D1 PR:

1. `scripts/gpu/oracle.sh` runs `fv-lab oracle` for targets that are cells (the six of its default
   `ORACLE_TARGETS` are), mapping `ORACLE_TARGETS` to the targets' arms (`--cells`, every arm of `oracle-<target>`),
   `FV_RUNTIME_STOCK_WAIT_S` to `--stock-wait-s`, `FV_POD_CAP_S` to the caps; it refuses `UP_AFTER`,
   `FV_ORACLE_RUN_MODE=all`, `FV_ORACLE_FAMILY`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT` and `FV_ORACLE_EXTRA_ENV` for
   them (§19.4), and keeps its own path for the clip-metric targets and `sfwan13` until §19.9.
2. `oracle` retires only when no target is left on the script path, i.e. after §19.9's cells pass their own check:
   then it joins `FV_LAB_RETIRED`, its `case` branch is deleted (the `wan` family keeps `oracle_fetch` /
   `oracle_diff`; the other families' dry runs compared before and after), its three fixtures freeze,
   `FV_ORACLE_TARGETS`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT`, `FV_ORACLE_PREFETCH` and `FV_REF2VA_PROMPT` move from
   `SCRIPT_KNOBS` to `MODE_KNOBS` (so into `runpod-http.sh`'s `FV_LAB_NOT_READ`), `runpod-http.sh run` / `plan` route
   `oracle` to `fv-lab pod up` (refusing `FV_ORACLE_TARGETS` with a pointer to `FV_CELLS`), `oracle.sh` becomes
   `fv-lab oracle` (or is deleted), `tests/repeats_oracle.rs`'s oracle tests run a frozen copy of the branch (as
   §15.10's `det-script/`), and the `sfwan` family's prelude (`FV_ORACLE_TARGETS=sfwan13 runpod-matrix.sh oracle`)
   calls `fv-lab run oracle --cells <sfwan13's arms> --runs-dir "$RUNS"`.
3. `runpod-http.sh upstream` stays for the upstream benchmarks (`UP_STEPS=cells:…`, `bench:*`) until those move;
   `plan-upstream` stays the upstream pods' golden.

#### 19.9 Not in this slice: the clip-metric targets and `sfwan13` (a decision first)

Decided by the coordinator on 2026-10-26 and done in §20 (the ten LTX targets; `sfwan13` a follow-up, §20.5).

`ltx25-{i2v,kf,ref2v,a2v,a2v-i2v,a2v-guided,retake,retake-v,retake-a,extend}` and `sfwan13` add, after their runs and
`compare-dumps`, steps the cell schema has no field for: ffmpeg SSIM / PSNR of whole clips, of `select`ed kept frames
against the source clip, of pinned frames against scaled and cropped conditioning images, `ffprobe` stream lines, the
reference's `ref.mp4` and the clips copied into `oracle-<target>-frames/`, all into `metrics.txt` and the log; for
`sfwan13` the reference mp4 decoded into frames for two `compare-clips` against frame directories that are not arms'.
These are §9.1's derived-clip step (what `hd` waits for, §15.6); the ADR has not designed it, and the choices are
architectural (CLAUDE.md "stop and ask"):

- *Where the step lives*: (a) a typed `[[metric]]` list in the cell (kinds `clip-vs-clip`, `kept-frames-vs-source`
  with a frame `select`, `frame-vs-image` with a canvas, `probe`), the runner composing the exact ffmpeg / ffprobe
  argv the script does, so the dry run lists them and the goldens hold them byte for byte; or (b) an ungated arm
  running a repository script (`scripts/gpu/oracle-metrics.sh <target>`, cut from the branch), the cell only naming
  it; (b) is smaller, (a) is what `hd` would reuse.
- *What `metrics.txt` is judged by*: today nothing reads it but people; a typed step could carry a threshold (a
  verdict) or stay a report.
- *`sfwan13`'s inputs*: its weights tree `sfwan21-1.3b` is R2-only since 2026-10-13, so on a `volume` tier its arms
  skip; it needs `weights = "volume+r2"` (a second oracle tier, or the oracle tier moved to `volume+r2`, which changes
  its frozen-to-be pod body), and the upstream `fastvideo` stack's `oracle_sfwan` reads `/workspace/weights/sfwan21-1.3b`,
  which the EU volume lacks too.

Until then they run as they do (`oracle.sh` with `ORACLE_TARGETS`, the `sfwan` prelude), and `oracle`'s branch stays
whole.

### 20. D1i: the clip-metric targets as cells: `[[metric]]` and extra cells

Status: proposed 2026-10-26 (draft PR from `claude/d1i-oracle-metrics`), CPU-tested; its GPU check is the
coordinator's (§20.7). Decision-id FVID-2026-10-26-d1i-oracle-metrics. This is §19.9's slice: the ten LTX
clip-metric targets of `runpod-matrix.sh oracle` become cells of the `oracle` tier. `sfwan13` does not (§20.5).
Nothing is retired here.

#### 20.1 The coordinator's decisions on §19.9 (2026-10-26)

1. **Where the step lives: option (a)**, a typed `[[metric]]` list in the cell. The runner composes exactly the
   ffmpeg / ffprobe argv the script does, the dry run lists them, and the goldens hold them byte for byte. The schema
   change is additive (cell v1 keeps `deny_unknown_fields`; `schemas/cell.v1.json` regenerated). `hd` is to reuse it
   later, so nothing in it is oracle-only.
2. **`metrics.txt` stays a report**, as the script writes it: no threshold, no verdict.
3. **`sfwan13`**: a sibling tier (`oracle-r2`, `weights = "volume+r2"`), so `oracle`'s pod body and fixtures stay; its
   upstream side reads `/workspace/weights/sfwan21-1.3b`, which the EU volume lacks. If getting that tree from R2
   onto the upstream pod's container disk is a small, typed addition, do it; else the ten LTX targets now and
   `sfwan13` in a follow-up, saying exactly what it needs. It is not small (§20.5).

#### 20.2 `[[metric]]` (cell v1, additive)

A metric is one **pass over clips**, the script's `for src in "<label>:<path>" …` loops, into a report directory:

```toml
[[metric]]
report = "{runs}/oracle-ltx25-retake-frames"   # below {runs}/: metrics.txt and the clip copies
tag = "ltx25-retake-frames"                    # the echoed lines' prefix, the dry-run lines' name
clips = [
  { label = "ours", path = "{runs}/oracle-ltx25-retake/frames/output.mp4" },
  { label = "ownenc", path = "{runs}/oracle-ltx25-retake-ownenc/frames/output.mp4" },
  { label = "reference", path = "{ref}/ref.mp4" },
]
requires = []          # more files every clip's measures need (ltx25-ref2v: ["{ref}/ref.mp4"])
missing = true         # a clip not there writes `<label> missing <path>` (+ ` or <file name>` per requires)
copy = true            # each present clip copied to <report>/<label>.mp4 first
measure = [
  { kind = "probe" },
  { kind = "kept-frames-vs-source", source = "{root}/scripts/gpu/fixtures/beach-push-768x512-24fps.mp4", select = 'lt(n\,33)+gte(n\,89)', fps = 24 },
]
echo = false           # after the pass: metrics.txt, all of it, into the log as `[<tag>] <line>`
```

For each clip in order: the clip and every `requires` file must be regular files, else (with `missing`) the line and
the next clip; with `copy`, the copy; then every measure in order. The kinds, and the argv each composes (the
script's, word for word; `<m>` is `ssim` then `psnr`, the script's `for m in ssim psnr`):

| kind | argv | line written |
|---|---|---|
| `probe` | `ffprobe -v error -show_entries stream=codec_type,codec_name,sample_rate,channels,nb_frames,r_frame_rate,width,height -of compact <clip>` | each stdout line as `<label>: <line>` |
| `clip-vs-clip` (`against = {label, path}`) | `ffmpeg -v info -nostats -i <clip> -i <against> -lavfi [0:v]format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]<m> -f null -` | `<label> vs <against label> clip <m>: <r>` |
| `kept-frames-vs-source` (`source`, `select`, `fps`) | `ffmpeg -v info -nostats -i <clip> -i <source> -lavfi [0:v]select='<select>',setpts=N/<fps>/TB,format=yuv420p[a];[1:v]select='<select>',setpts=N/<fps>/TB,format=yuv420p[b];[a][b]<m> -f null -` | `<label> kept frames vs source <m>: <r>` |
| `frame-vs-image` (`frame`, `image`, `width`, `height`) | `ffmpeg -v info -nostats -i <clip> -i <image> -lavfi [0:v]select=eq(n\,<frame>),setpts=N/TB[a];[1:v]scale=<w>:<h>:force_original_aspect_ratio=increase:flags=bilinear,crop=<w>:<h>,format=yuv420p[b];[a]format=yuv420p[c];[c][b]<m> -frames:v 1 -f null -` | `<label> frame <frame> <m>: <r>` |

`<r>` is the script's `$(… 2>&1 | grep -E "Parsed_<m>" | tail -1)` then `${r##*] }`: the last line of ffmpeg's
output (stdout and stderr together; `-f null -` writes nothing to stdout) holding `Parsed_<m>`, after its last `] `,
empty without one (`run::metric_value`). The runner runs `ffmpeg` / `ffprobe` from the children's `PATH` (the runtime
image's, as the script's; `RunConfig::ffmpeg` / `ffprobe`, which the tests point at fakes), each under `timeout` at
the arm cap. A dry run prints every command as `dry-run metric <tag>: <printf %q argv>`, each clip assumed present.

**Adjusted from §19.9's sketch, and why.** §19.9 named four kinds of step; here they are four kinds of *measure*
inside one pass over clips. The script's loops interleave the measures per clip (`ltx25-retake*`: clip 1's probe and
kept frames, then clip 2's, …), a missing clip writes one line and skips all its measures, and `metrics.txt`'s line
order is that interleaving, which the byte-for-byte equality must keep. One-kind-per-entry would reorder the lines or
need the missing rule twice. The trailing comparisons of the copies (`if [[ -f ours.mp4 && -f reference.mp4 ]]`) are a
second pass whose `requires` names the reference copy and which writes nothing when it is absent (`missing = false`);
`ltx25-ref2v`'s clip check includes the reference (`requires = ["{ref}/ref.mp4"]`, hence `… or ref.mp4`); and
`a2v-i2v`'s two blocks are three passes, each block's last with `echo` (the script echoes the whole file after each
block, so the first block's lines appear twice in the log, as in the script).

Validation: `report` a directory below `{runs}/`; `tag` and labels `[a-z0-9][a-z0-9._-]*`, labels unique per metric;
every path one placeholder of `{runs}` / `{scratch}` / `{ref}` / `{root}` then `/…`, no `.` / `..`, no quotes (`{ref}`
only in a cell with a `[reference]`); `select` printable without quotes, braces, `;` or brackets (it sits in a
quoted filter graph); `fps`, `width`, `height` positive; a metric measures or copies something; an unknown key or
kind is refused (`deny_unknown_fields`, `probe` a struct variant so a stray key on it is refused too). Nothing
names the oracle: `hd`'s keyframe metrics are passes over its arms' clips with these kinds or new ones.

#### 20.3 Extra cells (tiers v1, additive): the default run unchanged

**How the tier selects cells today.** A tier's `cells` is its default run: `fv-lab run oracle` without `--cells`
runs all nine targets, and that run is the dry-run golden (`oracle-dry-run.txt`, 43 lines) the GPU checks were made
against. `--cells` / `FV_CELLS` narrows it by arm (the unselected cells log `skip <cell> (no arm in FV_CELLS)`), and
for this tier `pod up` and `fv-lab oracle` refuse to run without `--cells` (§15.5, §19.1).

So the ten targets are **`extra_cells`**, a new optional tier field (`schemas/tiers.v1.json` regenerated,
`deny_unknown_fields` kept, left out when empty): cells of the tier run after `cells`, in their order, and only when
the selection names one of their arms (`Lab::resolve_selected`; `Lab::resolve` is the default run). Every reader
of a tier's arms takes them (`Tier::all_cells`): `fv-lab run` (selection from `--cells` / `FV_CELLS`), `pod up`'s
`--cells` check and the cells its pod runs, `fv-lab oracle`'s plan, `fv-lab gate` (the run's selection record picks
them), the lab's validation (unique arms and reference targets over all of the tier's cells; a cell listed once),
`fv-lab check` (`oracle: 9 cell(s), 32 arm(s) (+10 extra cell(s), 46 arm(s), on request)`). The default run is
byte-identical: `oracle-dry-run.txt`, `oracle-gates.tsv` and `runpod-http-oracle.json` unchanged
(`fv-lab-goldens.sh --check`), and `tests/dry_run.rs` holds `fv-lab run oracle` to the first. Why not the nine
plus ten in `cells`: the default run would be nineteen targets, its golden another, and a `fv-lab run oracle`
without `--cells` (a box, not a pod) four times as long. Why not a sibling tier: the coordinator's decision (they
are the `oracle` tier's cells), and `fv-lab oracle`'s `--family oracle` default then covers every LTX target.

#### 20.4 The ten cells and their references

`lab/cells/oracle-ltx25-<target>.toml`, each the script's arms for its target (the reference injected as for T2V,
`FASTVIDEO_DUMP_OPS=0,1,24,47`, `FASTVIDEO_H3_QUANT=off`, `--clip {runs}/{arm}/frames`; the prompts the script's
`FV_LTX_*_PROMPT` defaults, the fixtures under `{root}/scripts/gpu/fixtures`), then its metrics:

| target | arms (46 in all) | gate (trees) | metrics |
|---|---|---|---|
| `ltx25-i2v`, `ltx25-kf` | main, `-diff`, `-ownenc` (`INJECT_COND=0`), `-ownenc-diff`, `-ownimg` (`+ INJECT_PIXELS=0`), `-ownimg-diff` | `ltx25-two-stage` (`ltx25`) | ours / ownimg / reference: frame 0 vs the beach image (kf: and frame 120 vs its zoom), cover + crop to 768x512 |
| `ltx25-ref2v` (1536x896) | as i2v | `ltx25-ref2v` (`ltx25`, `ltx25-ic-lora-ingredients`) | ours / ownimg vs `ref.mp4`, then `ref.mp4` copied |
| `ltx25-a2v`, `ltx25-a2v-guided` (`--dit ltx25-dev/transformer_full`) | main, `-diff`, `-ownaud` (`INJECT_COND=0 INJECT_PIXELS=0`), `-ownaud-diff` | `ltx25-two-stage`; guided `ltx25-a2v-guided` (`ltx25`, `ltx25-dev`) | ours / ownaud / reference probed; ours vs reference |
| `ltx25-a2v-i2v` | as a2v | `ltx25-two-stage` | as a2v; then frame 0 of ours / ownaud / reference vs the beach image |
| `ltx25-retake`, `-retake-v`, `-retake-a`, `ltx25-extend` | main, `-diff`, `-ownenc` (`INJECT_COND=0 INJECT_PIXELS=0`), `-ownenc-diff` | `ltx25-two-stage` | ours / ownenc / reference probed and their kept frames vs the source (`lt(n\,33)+gte(n\,89)`; retake-a `lt(n\,121)`; extend `lt(n\,105)`, at 24 fps); ours vs reference |

Every tree is on the EU volume (`tests/lab_files.rs`). `lab/upstream.toml`'s `sol-ltx25` stack gains the ten
targets, in `scripts/gpu/upstream/oracle.sh run_oracle`'s order, with `oracle.sh`'s weight steps:
`weights:ltx25-dev` for `ltx25-a2v-guided` (the dev DiT and the distilled LoRA), `weights:ltx25` for the others
(ref2v's IC-LoRA is read from the volume as is). `fv-lab oracle --cells <the arms of ltx25-retake and ltx25-a2v-i2v>`
plans one `sol-ltx25` pod with `UP_STEPS="info:box weights:ltx25 oracle:ltx25-a2v-i2v,ltx25-retake"`
(`tests/oracle_driver.rs`).

`knobs::SCRIPT_KNOBS` gains the five prompts these targets read (`FV_LTX_REF_PROMPT`, `FV_LTX_A2V_PROMPT`,
`FV_LTX_A2V_I2V_PROMPT`, `FV_LTX_RETAKE_PROMPT`, `FV_LTX_EXTEND_PROMPT`): refused by fv-lab, read by the script.

#### 20.5 `sfwan13`: a follow-up, and what it needs

Not small, so not here (decision 3). What it needs:

1. **Upstream, the tree.** `oracle_sfwan` (`scripts/gpu/upstream/oracle.sh`) reads `$W/sfwan21-1.3b`
   (`/workspace/weights`, the volume; never written, CLAUDE.md), and the tree (28.93 GB, `weights-r2.tsv`) is R2-only
   since 2026-10-13. It needs (a) a typed upstream step to fetch an R2 tree onto the container disk: a stack
   target's `r2 = ["sfwan21-1.3b"]` in `lab/upstream.toml` (schema `upstream.v1.json`, additive), rendered as a
   `UP_STEPS` step `r2:sfwan21-1.3b` that `scripts/gpu/upstream/pod.sh` runs as fv-lab's runner does (§16.4:
   `r2-weights.py plan --pin weights-r2.tsv`, then `r2-fetch.py --root $UW`; the upstream image has a Python, so no
   apt step); (b) the R2 key on that pod as Runpod secret references (`weights::r2_env_refs`, §16.5) in the upstream
   launch, only when a selected target has an `r2` step, and handed to the plan step alone (a change to
   `upstream-start.sh.in` and to `runpod-http.sh plan-upstream`'s golden for that selection); (c) `oracle_sfwan`
   reading the tree from `$UW` when it is there (a script change on the upstream side); (d) the `fastvideo` stack's
   container disk raised by about 30 GB for such a run (150 → 180 GB), or a per-target disk. Four changes over a
   schema, a launch body, `pod.sh` and the upstream `oracle.sh`.
2. **Runtime, the tier.** `oracle-r2`: the `oracle` tier's pod with `weights = "volume+r2"` (the 28.93 GB tree on
   the container disk: 150 GB, the idle limit `weights::r2_idle_min`), one cell `oracle-sfwan13`: the main arm
   (`FASTVIDEO_WAN_VAE=full`, `--clip-dir`), `-diff`, the f32 control (`FV_ORACLE_F32`'s default for `sfwan*`: `-f32`,
   `-f32-diff`, `-bf16-vs-f32`), `-legacy` (`FASTVIDEO_WAN_CAUSAL_AR=0 FASTVIDEO_WAN_CAUSAL_FPB=1`) and its diff,
   `-composed` (`FASTVIDEO_WAN_CAUSAL_FLASH=0`), its diff and `-flash-vs-composed`. And two steps the schema lacks:
   a **`decode`** measure (`ffmpeg -loglevel error -y -i <clip> -start_number 0 <dir>/frame-%03d.png`: the
   reference's `ref.mp4` into `<scratch>/oracle-ref/sfwan13/frames`, our mp4 into `{runs}/oracle-sfwan13/mp4frames`;
   our clip is `ls frames/*.mp4 | head -1` in the script, so a named clip or a glob rule; the reference's frames sit
   beside `{ref}`, so a placeholder for the reference directory), and a **`[[compare]]` of two frame directories**
   that are not arms' (`compare_one oracle-sfwan13-frames-vs-ref <ref frames> <our frames>`, `-codec-floor`,
   `-legacy-frames-vs-ref`). Its goldens: the script's dry run with `FV_ORACLE_TARGETS=sfwan13` (a second `fam_env`
   of `fv-lab-goldens.sh`), and the pod body `runpod-http.sh plan` with the tier's disk and idle limit, its R2 env the
   one allowed difference (as `gold`, §16.6).
3. **The `sfwan` family's prelude** (`FV_ORACLE_TARGETS=sfwan13 bash runpod-matrix.sh oracle` with `FV_RUNS_DIR`)
   becomes `fv-lab run oracle-r2 --cells <its arms> --runs-dir "$RUNS"` (that pod then needs the R2 key, or the
   prelude is dropped and `sfwan13` runs on its own through `fv-lab oracle --family oracle-r2`).

Until then `sfwan13` runs as it does (`oracle.sh` with `ORACLE_TARGETS=sfwan13`, the `sfwan` prelude).

#### 20.6 Equality: goldens, script changes, fakes, the compare

- **Goldens.** `fv-lab-goldens.sh` writes, for a tier with extra cells, `<tier>-extra-dry-run.txt` and
  `<tier>-extra-gates.tsv` from the script under the extra cells' settings: `oracle` with `FV_ORACLE_TARGETS`
  naming the ten (`FV_GEN_TIMEOUT_S=5400`). 145 lines, 87 of them metric commands. The pod body is the tier's (one
  fixture). `tests/dry_run.rs::the_extra_cells_dry_run_as_the_script_with_their_targets`: `fv-lab run oracle
  --dry-run` selecting the ten targets' arms is that file line for line after one `skip <cell> (no arm in
  FV_CELLS)` per default cell (the script, given its targets, names no other), and its gated arms are the gates
  file's.
- **Script changes** (`runpod-matrix.sh oracle`):
  - *dry run only*: each metric command is built as an array and passed to `metric_dry <tag> <cmd…>`, which in a dry
    run logs `dry-run metric <tag>: <printf %q>` and skips it, and the clip checks pass in a dry run (before, the
    dry run wrote `missing` lines into a scratch `metrics.txt` and logged no command); a real run runs the same argv
    as before;
  - *a real-run change, failures only*: the metric pipelines end in `|| true`. Under `set -euo pipefail` a
    comparison whose ffmpeg printed no `Parsed_` line (`grep` exits 1) or a failing `ffprobe` ended the whole family
    (`matrix_exit` non-zero, the targets after it never run). Now the value is empty and the run goes on; fv-lab
    does the same;
  - *a real-run change, disk only*: `$ours-ownaud` (the a2v targets' third dump, hundreds of MB, which the
    `/dump/` fetch skip does not match) is removed with the others after the target; fv-lab removes it as a `fresh`
    path.
- **Fakes, both runners** (`tests/repeats_oracle.rs::clip_metrics_report_as_the_script`): the live branch and
  `fv-lab run` on one fake box with fake `ffmpeg` / `ffprobe` (a checksum of their inputs' bytes and of the filter
  graph, behind `] `), the fixtures, and references for `ltx25-kf`, `ltx25-ref2v`, `ltx25-a2v-i2v`, `ltx25-retake`
  and `ltx25-retake-a` (whose dump has no `ref.mp4`). Every `metrics.txt` and clip copy is byte-identical (paths of
  either side's scratch read alike), the log's echoed reports equal, the `missing` and `or ref.mp4` lines, the
  double echo of a2v-i2v, the dumps (`-ownaud` too) gone. The script copy's `PATH` line puts the fakes first (it
  prepends `/usr/local/bin`, where a real ffmpeg would shadow them); `ORACLE_REFS_DONE` is fv-lab's alone in a run of
  several targets (§15.5).
- **A fix found by that test:** the log's lines are now written in one `write` each (`run::raw_line`). With five
  references fetched at once, the fetch threads' `writeln!` (text, then newline) interleaved and split a line.
- **`fv-lab-compare-runs.sh`** compares every `*/metrics.txt` of either run: byte-identical, or, when only the
  values differ (each line cut at its last `: ` is the same in both), reported (`metrics <file>: the same
  measurements, values differ …`) and no difference: the values read the clips' mp4 encodes, and those depend on
  more than the frames (`frames.sha256.json` covers the PNGs; libx264's output depends on its thread count, which
  follows the pod's CPUs: UNVERIFIED whether the two pods' encodes are byte-identical). A missing line, another
  order or a report in one run only is a difference. The summary gains `<n> metrics report(s)`. Self-test in
  `fv-lab-goldens.sh`.
- **`fv-lab oracle --fetch-tree [--fetch-skip ERE] [--fetch-max-mb N]`** passes `pod up`'s tree fetch to the
  runtime side (`OraclePlan::runtime_fetch`): `metrics.txt` and the copies are files of the tree, not of the default
  fetch's summaries (`runpod-http.sh` needs `FV_FETCH_TREE=1` for them too). Without it nothing changes.

| | script | fv-lab | why |
|---|---|---|---|
| a metric command | runs on `PATH`, no cap | the same tool on the same `PATH`, under `timeout` at the arm cap | a hung ffmpeg would hold the pod to its backstop |
| `FV_LTX_*_PROMPT` | change the arms | refused | the cells are the defaults (§15.5) |

#### 20.7 The GPU check (the coordinator's)

The fewest targets that run every kind: **`ltx25-a2v-i2v`** (probe, clip-vs-clip, frame-vs-image) and
**`ltx25-retake`** (probe, kept-frames-vs-source, clip-vs-clip), 8 arms, one `sol-ltx25` upstream pod (two
references), the cheapest of their kinds at 512p (retake is one distilled stage; ref2v at 1536x896 and the guided
a2v on the dev DiT cost more and add no kind). Both runtime runners on one reference, as §19.7; the commit `SHA` on
GitHub with `fastvideo-rs-runtime:sha-<7>` published (with this PR's `lab/` and fv-lab), `fv-lab` built on the build
pod. RTX PRO 6000, EU-RO-1, on `bz0ofkfx45`.

```sh
SHA=<7>
CELLS=oracle-ltx25-a2v-i2v,oracle-ltx25-a2v-i2v-diff,oracle-ltx25-a2v-i2v-ownaud,oracle-ltx25-a2v-i2v-ownaud-diff,oracle-ltx25-retake,oracle-ltx25-retake-diff,oracle-ltx25-retake-ownenc,oracle-ltx25-retake-ownenc-diff
D=/tmp/fv-oracle-metrics; rm -rf $D; mkdir -p $D
./fv-lab oracle --sha $SHA --cells $CELLS --upstream-wall-cap-s 7200 --wall-cap-s 3600 --fetch-tree --fetch-skip '/frames/'   # the plan: check it
# fv-lab side: the sol-ltx25 upstream pod, both references, the runtime pod (tree fetch, no frame PNGs), the release
RUNPOD_API_KEY=… ./fv-lab oracle --sha $SHA --cells $CELLS --upstream-wall-cap-s 7200 --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '/frames/' --url-file $D/urls --hold-upstream $D/hold --yes >$D/report.json 2>$D/lab.log & LAB=$!
# script side: the same references (as §19.7), then lift the hold whatever happened
until [ -s $D/urls ] || ! kill -0 $LAB 2>/dev/null; do sleep 20; done
if [ -s $D/urls ]; then
  RUNPOD_API_KEY=… FV_FAMILY=oracle FV_GEN_TIMEOUT_S=5400 FV_POD_CAP_S=3600 FV_FETCH_TREE=1 FV_FETCH_SKIP='/frames/' \
    FV_EXTRA_ENV="FV_ORACLE_URL=$(cat $D/urls) FV_ORACLE_TARGETS='ltx25-a2v-i2v ltx25-retake' FASTVIDEO_DUMP_OPS=0,1,24,47" \
    bash scripts/gpu/runpod-http.sh run $SHA >$D/script.log 2>&1
fi
touch $D/hold; wait $LAB; echo "fv-lab oracle rc=$?"
L=artifacts/runpod/oracle/$SHA-<MMDDHHMMSS>-<hex>; S=artifacts/runpod/oracle/$SHA-<MMDDHHMM>
bash scripts/gpu/fv-lab-compare-runs.sh $L $S
jq '{served, missing, runtime_ok, runtime_exit_code, problems, upstream: [.upstream[] | {id, tag, released, deleted}]}' $D/report.json
for r in $L $S; do grep -c '^\[ltx25-\(a2v-i2v\|retake\)-frames\] ' $r/live.log; done   # 36 each: a2v-i2v 8 + 14 (echoed after each part), retake 14
grep -E 'runpod-pod-(created|deleted)' artifacts/fv-lab/ledger.tsv | tail -4    # both fv-lab pods created and deleted
```

**Pass:**

- `fv-lab oracle rc=0`; the report `"served": ["ltx25-a2v-i2v", "ltx25-retake"]`, `"missing": []`, `"runtime_ok":
  true`, one upstream entry released (`… the hold is lifted`) with `"deleted": {"Ok": null}`.
- `fv-lab-compare-runs.sh` exit 0 with `same oracle-ltx25-a2v-i2v-frames/metrics.txt` and `same
  oracle-ltx25-retake-frames/metrics.txt` (or, for either, `metrics <file>: the same measurements, values differ
  (reported, not judged: the clips' encodes)`, §20.6) and the line "8 arm(s) the same (4 ran with byte-identical
  frames, 0 skipped in both, 0 failed in both; 4 with the same compare-dumps lines), 0 control arm(s) reported, 0
  report(s) the same, 0 verdict file(s) the same, 2 metrics report(s), 0 difference(s)". (If the two a2v-i2v
  generation arms exit 1 on both sides, as on 2026-09-29 before `ltx2/gen.wav`'s size check was fixed, they count as
  "failed in both" and the line reads 2 ran, 2 failed; the compare still passes on the retake arms.)
- Each `metrics.txt`: a2v-i2v 6 probe lines (2 streams x ours, ownaud, reference), 2 `ours vs reference clip`, 6
  `<label> frame 0 <m>`; retake 6 probe lines, 6 `kept frames vs source`, 2 `ours vs reference clip`; no `missing`
  line; values not empty. Both `live.log`s carry them prefixed (a2v-i2v's first 8 twice).
- Both runtime pods `matrix_exit=0`; the ledger shows the upstream and the fv-lab runtime pod created and deleted;
  `fv-lab pod reap --account` lists nothing.

**Estimate** ($2.49/hr): the upstream pod about 45-60 min (readiness, up to 15 min of a cold image pull on a new
host, docs/oracle.md "LTX-2.5 audio-to-video"; `weights:ltx25`'s pack rebuild; two references of about 1-3 min each;
then held while both runtime sides run), about $1.90-$2.50; each runtime pod about 20 min (5 min boot and preflight,
two 512p two-stage and two one-stage LTX runs with dumps, four `compare-dumps`, 22 ffmpeg / ffprobe commands in
seconds), about $0.85 each: **about $3.60-$4.20** in all. fv-lab's balance guard (§19.1) needs, at these caps, $8 +
the upstream pod at $2.50/hr over 7,200 + 1,200 s ($5.83) + the runtime pod over 3,600 + 1,200 s ($3.33) = **$17.17**,
plus what the account already spends over the 2 h cap (a running build pod at $0.96/hr adds $1.92: stop it); with
the script side's pod (`FV_POD_CAP_S=3600`, at most about $2.50) start at **$20 or more**. The upstream cap stays at
7,200 s: the references' wait is what the cap leaves after the stock wait and the fetch reserve (§19.1), 3,300 s at
7,200, and a cold pull plus the pack rebuild may need most of it. The readiness risk of §19.4 (600 s per cycle)
applies. Stop before the balance would fall below $8 (CLAUDE.md).

#### 20.8 Then: the retirement of `oracle` (§15.10, §19.8)

After §19.7's and this check pass, in the next D1 PR, `scripts/gpu/oracle.sh` runs `fv-lab oracle` for every
target that is a cell (now nineteen: every default target and the ten here), mapping `ORACLE_TARGETS` to `--cells`
(every arm of `oracle-<target>`), `FV_RUNTIME_STOCK_WAIT_S` to `--stock-wait-s`, `FV_POD_CAP_S` to the caps, and
passing `--fetch-tree --fetch-skip '/frames/'`; it refuses `UP_AFTER`, `FV_ORACLE_RUN_MODE=all`,
`FV_ORACLE_FAMILY`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT` and `FV_ORACLE_EXTRA_ENV` for them, and keeps its own path
for `sfwan13` alone. `oracle` retires when `sfwan13` leaves the script path too (§20.5, its own check), in one PR:

1. **Knobs.** `oracle` joins `FV_LAB_RETIRED`; `FV_ORACLE_TARGETS`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT`,
   `FV_ORACLE_PREFETCH`, `FV_REF2VA_PROMPT` and the five `FV_LTX_*_PROMPT` move from `SCRIPT_KNOBS` to `MODE_KNOBS`
   (so into `runpod-http.sh`'s `FV_LAB_NOT_READ`, `tests/knobs.rs`); `runpod-http.sh run` / `plan` route `oracle`
   (and `oracle-r2`) to `fv-lab pod up`, refusing `FV_ORACLE_TARGETS` with a pointer to `FV_CELLS`.
2. **Fixtures freeze**: `oracle-dry-run.txt`, `oracle-gates.tsv`, `runpod-http-oracle.json`,
   `oracle-extra-dry-run.txt`, `oracle-extra-gates.tsv` (and `oracle-r2`'s); `fv-lab-goldens.sh` stops writing them.
3. **The branch goes**: the `oracle)` case, `metric_dry` and `oracle_run`; the `wan` family keeps `oracle_fetch` /
   `oracle_diff`; the other families' dry runs compared before and after. `tests/repeats_oracle.rs`'s oracle tests run
   a frozen copy of the branch (`fixtures/oracle-script/`, as §15.10's `det-script/`), with its dry runs held to the
   frozen goldens.
4. **`oracle.sh`** becomes `fv-lab oracle` (a wrapper with the mapping above, or deleted with a pointer in
   docs/oracle.md).
5. **The `sfwan` family's prelude** becomes `fv-lab run oracle-r2 --cells <sfwan13's arms> --runs-dir "$RUNS"`
   (§20.5 item 3).

`runpod-http.sh upstream` stays for the upstream benchmarks until they move (§19.8 item 3).

#### 20.9 What does not change

The `oracle` tier's default run, its three fixtures and pod body; every other tier; `fv-lab pod up`, `fv-lab oracle`
without `--fetch-tree`; `oracle.sh`; the script's real runs of the tensor-diff targets; `runpod-http.sh`. Code:
`plan.rs` (`Metric`, `Clip`, `Measure`, `Step::Metric`, `Tier::extra_cells`, `Lab::resolve_selected`), `run.rs`
(`Runner::metric`, `metric_value`, the dry-run lines, `raw_line`), `oracle.rs` (`runtime_fetch`), `bin/fv-lab.rs`;
tests in `plan.rs`, `run.rs`, `tests/{dry_run,repeats_oracle,oracle_driver,fv_lab_bin,lab_files}.rs`.
