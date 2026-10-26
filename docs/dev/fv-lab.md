# fv-lab: run a tier's cells on a GPU pod

`fv-lab` runs the lab's run plan (`lab/tiers.toml`, `lab/cells/*.toml`) on a GPU box and makes that box: a Runpod pod
on fv-deploy's guards (price cap, $8 floor, EU volume, detached backstop, ready within 600 s in at most 3 cycles).
Design: [ADR-0019](../adr/0019-fv-lab.md). Today it carries eleven tiers that were `runpod-matrix.sh` families. Each
passed its GPU gate (fv-lab's run and the script's of one commit, frames byte-identical), so its script branch is
deleted and the tier is fv-lab only ([The families](#the-families), [ADR-0019 §10](../adr/0019-fv-lab.md)). Each is
held to the script's last dry run by a frozen golden ([below](#keeping-it-equal-to-the-scripts)). The other families
stay in the script and move in later slices. The last of the eleven, `b4a` (D1d), is the first whose cells carry a
verdict step ([Verdicts and `fv-lab gate`](#verdicts-and-fv-lab-gate); its GPU gate passed on `e82b6b7`, ADR-0019
§11.6). A twelfth tier, `gold`, is fv-lab's own: every gold variant's canonical cell (ADR-0023).
`precision-nvfp4` and `ltxvae-gen` are two modes of retired families, restored as tiers ([ADR-0019 §12](../adr/0019-fv-lab.md)).

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
| `b4a` | `b4a-h3-*` (3) | 6 | the H3 TeaCache step cache, baseline binary vs this build: byte identity and equal cache decisions, a verdict per cell | D1d; needs `--baseline-sha`; retired after its GPU gate (§11.6) |
| `precision-nvfp4` | `precision-nvfp4-ltx25-{4k5s,1080p20s}` | 4 | NVFP4 video FFN against bf16, LTX-2.5 Sol, compare + lossy gate | was `precision`'s `FV_PRECISION_ARM=ltx-nvfp4` mode; restored (§12) |
| `ltxvae-gen` | `ltxvae`'s 4 + `ltxvae-gen-{4k5s,1080p20s}` | 16 | `ltxvae`, plus per workload a cold gen saving its latents and the streaming decoder on them | was `ltxvae`'s `FV_VAE_GEN=1`; restored (§12); shares `ltxvae`'s cells |
| `det-short` | `det-short-{h3,wan,ltx,ltx-sage}` | 15 | byte identity across fresh processes at short configs, three runs per group and the timed-pick control; `determinism.txt` | D1d (§15); its script branch stays until its GPU check; `wan` skips on the EU volume |
| `determinism` | `determinism-{ltx-t2v,ltx-i2v,h3}` | 9 | byte identity of 1080p runs within one process and across two; prompt sets as files, hashes per run, stage digests | D1d (§15); branch stays until its GPU check |
| `oracle` | `oracle-<target>` (9) | 32 | the GPU oracle diff of the nine tensor-diff targets against the upstream references (`FV_ORACLE_URL`) | D1d (§15); needs `--extra-env FV_ORACLE_URL=…` and `--cells`; the clip-metric targets stay in the script |
| `gold` | `gold-*` (32) | 32 | every gold variant's canonical cell: the served variant through the serve engine at its gold canvas | fv-lab's own, generated from `lab/gold/` ([below](#the-gold-files-ci-and-the-tier-table)); weights `volume+r2` ([below](#weights-from-the-volume-and-r2-volumer2)) |
| `h3-graphs` | `h3-graphs-turbo{-trace,}` | 4 | h3-turbo at 768p: a traced process with the step graphs on, then per-step sync, the lazy step clock and CUDA graphs of the steps, timed warm, `--off-identity` against lazy | D1e: was `scripts/gpu/experiments/h3-graphs-*.sh` (`FV_CKS=turbo`), [below](#port-one-offs-the-experiment-tiers) |

All eleven above `precision-nvfp4` are **retired from the script** (`scripts/gpu/fv-lab-retired.sh`, ADR-0019 §10; `b4a`
§11.6): their `runpod-matrix.sh` branches are deleted, and `runpod-matrix.sh <tier>` refuses them (exit 2). The script keeps the other families.
ADR-0019 §9.1 says why for each and when its branch retires. An arm whose tree is not on the EU volume (`h3-8step`)
is recorded as skipped (`weights incomplete`), as the script recorded it.

What the retired branches had and the tiers do not, dropped with the branches: the `FV_VAE_*`, `FV_LTXFPS*`,
`FV_TECH_CELLS`, `FV_B6_CELLS`, `FV_B4A_CELLS` knobs (ADR-0019 §9.4, §11.5) and the summaries the branches copied into `live.log`. The two
modes dropped with them are tiers again (owner decision 2026-10-25, [ADR-0019 §12](../adr/0019-fv-lab.md)):

```sh
./fv-lab run precision-nvfp4 --dry-run      # was FV_PRECISION_ARM=ltx-nvfp4 runpod-matrix.sh precision
./fv-lab run ltxvae-gen --dry-run           # was FV_VAE_GEN=1 runpod-matrix.sh ltxvae (all 16 arms)
./fv-lab run ltxvae-gen --cells ltx25-4k5s-gen,ltx25-4k5s-gen-streaming   # only the 4k5s gen pair (no knob did this)
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family precision-nvfp4 --lpips --yes --wait
```

The old knobs pick arms, and `--cells` (`FV_CELLS`) names the same arms. Each list below assumes the mode's other knobs
at their defaults:

| old setting | `--cells` |
|---|---|
| `FV_NVFP4_WORKLOADS=4k5s` | `ltx25-4k5s-sol-bf16,ltx25-4k5s-sol-nvfp4` |
| `FV_NVFP4_WORKLOADS=1080p20s` | `ltx25-1080p20s-sol-bf16,ltx25-1080p20s-sol-nvfp4` |
| `FV_VAE_WORKLOADS=4k5s` (the 8 arms of 4k5s) | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltxvae-4k5s-fast-untraced,ltxvae-4k5s-streaming-untraced,ltxvae-4k5s-fast-mp4,ltxvae-4k5s-fast-tune,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming` |
| `FV_VAE_WORKLOADS=1080p20s` | the same eight with `1080p20s` |
| `FV_VAE_UNTRACED=0` (traced and gen arms, both workloads) | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming,ltxvae-1080p20s-fast,ltxvae-1080p20s-streaming,ltx25-1080p20s-gen,ltx25-1080p20s-gen-streaming` |
| `FV_VAE_UNTRACED=0 FV_VAE_WORKLOADS=4k5s` | `ltxvae-4k5s-fast,ltxvae-4k5s-streaming,ltx25-4k5s-gen,ltx25-4k5s-gen-streaming` |

`--cells` runs the same arms, but the run is not identical. The script never reached the arms its knobs left out. fv-lab
logs each unselected arm as `skip <arm> (not in FV_CELLS)` and a compare of two unselected arms as `skip compare
(frames missing)`. The arm directories, frames and reports are the same.

Their goldens are what the script wrote in that mode at `5987e6a4` (`scripts/gpu/tests/fv-lab-restored-goldens.sh
--check` reproduces them from git history), frozen from the start; fv-lab's dry run and pod equal them with the
family named by the tier and without the mode's `FV_EXTRA_ENV` setting. `FV_VAE_TRACE=0` has no tier. NVFP4 needs
sm_100 / sm_120: the tier is the PRO 6000, as the mode was.

### Port one-offs: the experiment tiers

D1e ([ADR-0019 §14](../adr/0019-fv-lab.md)) decided the ten port experiments F1 moved to `scripts/gpu/experiments/`:
nine are deleted (finished; results in `docs/ports/*.md`, `docs/perf/*.md`, evidence in `artifacts/`, scripts in git at
`22a5fd7b`), and `h3-graphs` is the tier above, for the open item of `docs/perf/h3-denoise-graphs.md` §5 (h3-turbo
was never measured). Such a tier is listed in `FV_LAB_EXPERIMENTS` (`scripts/gpu/fv-lab-retired.sh`): never a
`runpod-matrix.sh` family, so `runpod-http.sh run` / `all` refuse it as they refuse `gold`. Its commands are the
experiment pod script's own:

```sh
bash scripts/gpu/tests/fv-lab-experiment-goldens.sh [--check]   # cut from h3-graphs-pod.sh: fixtures/experiments/h3-graphs-script.txt, the prompt sets
./fv-lab run h3-graphs --dry-run                                 # fv-lab's own golden: fixtures/h3-graphs-dry-run.txt
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family h3-graphs --wall-cap-s 3600 \
  --fetch-tree --fetch-skip '/frames/|\.cache$' --yes --wait    # the GPU check, ADR-0019 §14.4 (~35 min, ~$1.45)
```

`tests/experiment_tiers.rs` holds each arm to the script's command word for word after the path mapping of §14.3
(the run directory for `/srv/…`, `frames/` for `clip/`, the tier's cache paths for `/fvw/…`, the committed prompt sets,
the runner's cap for `timeout 2400`). The experiment's own scripts stay until that GPU check passes.

### `runpod-http.sh` for a retired tier

`runpod-http.sh run [sha]` with `FV_FAMILY` (default `rtx6000`) set to a retired tier runs
`fv-lab pod up --sha <sha> --family <tier> --root <repo> [--cells …] [--baseline-sha …] [--extra-env …] [--lpips] --yes --wait`
instead of its own pod, mapping `FV_CELLS`, `FV_BASELINE_SHA`, `FV_EXTRA_ENV`, `FV_LPIPS=1` and the tree fetch
(`FV_FETCH_TREE=1` → `--fetch-tree`, `FV_FETCH_SKIP` → `--fetch-skip`, `FV_FETCH_MAX_MB` → `--fetch-max-mb`; `run` only) across; `plan [sha]
[tag]` prints fv-lab's plan (no `--yes`). The other families keep the script path. `FV_POD_CAP_S` below the tier's
`wall_cap_s` maps to `pod up --wall-cap-s` (a cap is lowered, never raised). Every other setting of the script path
that fv-lab would not honour is refused, not dropped, unless fv-lab does the same thing anyway:

- no equivalent, refused when set: `FV_POD_FILES`, `FV_POD_SCRIPT`, `FV_POD_FILE`, `FV_PROMPTS`, `FV_PROMPT`,
  `FV_SEED`, `FV_AVOID_MACHINES`, `FV_CREATE_WAIT_S`, `RUNPOD_ALLOWED_CUDA`,
  `FV_KEEP_POD=1`, `RUNPOD_NO_VOLUME=1`, `FV_FETCH_SKIP` / `FV_FETCH_MAX_MB` without `FV_FETCH_TREE=1`, and the retired pair selectors `FV_TECH_CELLS`,
  `FV_B6_CELLS`, `FV_B4A_CELLS` (select by arm with `FV_CELLS`);
- refused unless equal to the tier's value in `lab/tiers.toml`: `FV_GEN_TIMEOUT_S` (`cell_timeout_s`),
  `FV_CONTAINER_DISK_GB`, `RUNPOD_CLOUD_TYPE`, `RUNPOD_GPU_TYPE`, `FV_SKIP_TAE=1` on a `tae = true` tier, and
  `RUNPOD_GPU_MAX_DPH` below the tier's `max_usd_per_hr` (above it is moot: fv-deploy holds the tier's);
- refused unless equal to fv-lab's fixed value: `FV_IDLE_GPU_MIN` 10, `FV_IDLE_GPU_GRACE_MIN` 30,
  `FV_IDLE_NET_BYTES` 262144, `FV_MIN_DRIVER` 580, `FV_BOOT_WAIT_S` 600, `RUNPOD_VOLUME_ID` / `RUNPOD_VOLUME_NAME` /
  `FV_PLAN_VOLUME` the EU volume, `RUNPOD_IMAGE` the commit's runtime image, `FV_POD_PREFIX` `fv-lab`;
- `FV_EXTRA_ENV` entries no tier reads: the dropped modes' knobs (`FV_PRECISION_ARM`, `FV_NVFP4_WORKLOADS`,
  `FV_VAE_*`, `FV_LTXFPS*`, `FV_TECH_CELLS`, `FV_B6_CELLS`, `FV_B4A_CELLS`), `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED`, and what the
  start command sets after them (`FV_WORK`, `FV_SCRATCH`, `FV_RUN_TAG`, `FV_CELLS`, `FV_GEN_TIMEOUT_S`). The two
  restored modes' settings name their tier: `FV_PRECISION_ARM=ltx-nvfp4` says `use FV_FAMILY=precision-nvfp4`,
  `FV_VAE_GEN=1` says `use FV_FAMILY=ltxvae-gen` (refused, not routed: the run would land under another run
  directory and pod name than `FV_FAMILY`'s); `FV_NVFP4_WORKLOADS`, `FV_VAE_WORKLOADS`, `FV_VAE_UNTRACED` say to
  narrow by arm with `FV_CELLS`, and `FV_B4A_CELLS` to select `b4a`'s pair by its arms (`FV_CELLS="<cell>-base
  <cell>-pr"`). A mode setting is answered before any other refusal, and one message lists every
  refused `FV_EXTRA_ENV` key. `fv-lab` itself refuses the same: `pod up --extra-env` this whole set, `fv-lab run`
  the mode knobs in its environment (its own inputs `FV_WORK`, … are not refused, and `FV_PROMPTS` / `FV_PROMPT` /
  `FV_SEED` keep their warning); `runpod-matrix.sh precision` with `FV_PRECISION_ARM=ltx-nvfp4` (or `ltxvae` with
  `FV_VAE_GEN=1`, or the tier's own name) names the tier.

`FV_CELLS` and `FV_EXTRA_ENV` are split on spaces (and `FV_CELLS` on commas) without globbing. `all` mode is refused
(run `kernels`, then `run`), and `FV_FAMILY=gold` is refused for `run` / `all` (no script family: `fv-lab pod up
--family gold`; its `plan` stays the script's body, the payload golden). The binary is `FV_LAB_BIN`, else `fv-lab`
on `PATH`, else `./fv-lab` at the repository root ([Build](#build)); without one the driver says how to build it.
`scripts/gpu/tests/fv-lab-route.test.sh` (CI: `control.yml`) holds all of this, with a fake `FV_LAB_BIN`.

```sh
FV_DRY_RUN=1 FV_FAMILY=b6 FV_BASELINE_SHA=e93e357 bash scripts/gpu/runpod-http.sh run <7>   # prints the fv-lab command
FV_DRY_RUN=1 FV_FAMILY=b4a FV_CELLS="b4a-h3-4step-tea-base b4a-h3-4step-tea-pr" FV_BASELINE_SHA=e93e357 \
  bash scripts/gpu/runpod-http.sh run <7>                           # b4a's GPU gate cell, through fv-lab
FV_DRY_RUN=1 FV_FAMILY=b4b bash scripts/gpu/runpod-http.sh run <7>   # a script family: says what it would run
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
around its arms, as data (the branch is retired; `crates/fastvideo-lab/tests/verdict.rs` holds fv-lab to a frozen copy
of it, `fixtures/b4a-script/`). The verdict's arms start from an empty directory; right after each, its exit (`summary.json`),
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
b4a-h3-4step-tea-base,b4a-h3-4step-tea-pr` is what the retired script ran as `FV_B4A_CELLS=b4a-h3-4step-tea`.

```sh
./fv-lab run b4a --dry-run --baseline-sha <7>          # the script's dry run, verdict lines included
./fv-lab gate artifacts/runpod/b4a/<tag>               # verdicts and gates again, offline, into <tag>.regate
./fv-lab gate artifacts/runpod/b4a/<tag> --check --out /tmp/regate   # and compare with the run's own files
./fv-lab gate artifacts/runpod/fastvideo/<tag> --gpucheck ./fv-gpucheck --out /tmp/regate   # a tier with gates
```

`fv-lab gate <run dir>` evaluates a finished run's promotion gates (`fv-gpucheck gate`, CPU only, so any build of
fv-gpucheck) and verdicts again from its files, with no GPU and without running an arm or a compare
([ADR-0019 §11.3](../adr/0019-fv-lab.md)):

- the tier is `--family` or the directory's parent;
- the arms are `--cells`, else the run's `<tier>.arms.json` (the selection `fv-lab run` records), else all of the
  tier's when the run holds them all; a run without the record that holds fewer (a script run with `FV_B4A_CELLS` from
  before the branch went, or one cut short) is refused until `--cells` names its selection, because a guess would pass the cells it never
  finished;
- the output goes to `--out`, default `<run dir>.regate`, never into the run directory;
- `--check` compares each derived file with the run's own (gate reports by status) and exits 1 on any difference,
  whether or not the verdicts pass;
- a gate whose arms did not both run is skipped and counted; one whose arms ran without a `benchmark.json` fails.

Exit 0 when all passed, 1 when one failed or nothing could be evaluated. `pod fetch` takes the verdict files
(`*.verdict.json`, `*.diff`, `<tier>.arms.json`, `<arm>/*-evidence.log`); `runpod-http.sh` only with
`FV_FETCH_TREE=1`. `fv-lab-compare-runs.sh` compares them byte for byte when either run has them (a family verdict's
cell list as a set), and `--expect-verdicts N` fails unless at least N compared the same. `FV_B4A_CELLS` is refused
by `fv-lab run`, `pod up --extra-env` and `runpod-http.sh`'s `FV_EXTRA_ENV`: select a pair by its arms (`--cells
<cell>-base <cell>-pr`, `FV_CELLS`).

## Determinism and the oracle (D1d, second slice)

[ADR-0019 §15](../adr/0019-fv-lab.md). Three tiers keep their `runpod-matrix.sh` branches until each one's GPU check
through both runners (§15.7); their goldens come from the live branches (`fv-lab-goldens.sh`).

- **`[repeats]`** (det-short, determinism): after each arm its frames are hashed (`hash = "arm"`: one hash per arm,
  `<arm>/hashes.json` and `<runs>/hashes.txt`; `hash = "run"`: one per prompt-set run, `<arm>/hashes.txt` and the
  stage digests in `<arm>/digests.txt`) and pruned to `keep`; after every cell `<runs>/determinism.txt` reports each
  hash against the reference (`IDENTICAL` / `DIFFERENT`), as the script writes it. The typed verdict: a cell passes
  when every `judged` arm's hashes equal the reference's (the timed-pick `controls` are reported only); the log says
  `repeats <cell>: pass|fail (…)` and `fv-lab gate <run dir>` decides it again from the fetched hashes (no frames
  needed). No verdict file is written (the script writes none; the GPU comparison is file for file).
- **`[[prompt_file]]`** (determinism): the prompt set an arm reads as `{runs}/<file>`, written as the script's `det_set`.
- **`[reference]`** (oracle): the upstream dump of `oracle-<target>`, fetched from `FV_ORACLE_URL` (space or comma
  separated `http(s)://` base URLs, anything else refused; `FV_ORACLE_WAIT_S`, default 3 h) in the background from the start, read as `{ref}`; the run
  directory gets `ORACLE_REFS_DONE` once every selected reference is here. Unavailable: the cell's first arm is
  recorded as skipped (`reference dump unavailable`). After the cell the dumps (`fresh`) and the reference are removed.
- **Ungated arms** (no `gate`): `compare-dumps` of two dumps, run only when both exist (`requires`), their
  `compare-dumps` lines copied into the log (`echo`).

```sh
./fv-lab run det-short --dry-run                         # the script's dry run
./fv-lab run det-short --cells h3-r1,h3-r2,h3-r3,h3-timed   # was FV_DET_CELLS=h3
./fv-lab run determinism --cells h3-ab,h3-c,h3-timed     # was FV_DET_CELLS="h3-ab h3-c h3-timed"
./fv-lab gate artifacts/runpod/det-short/<tag> --check --out /tmp/regate   # determinism.txt again, and the verdicts
# oracle: the upstream pods first (scripts/gpu/oracle.sh, or runpod-http.sh upstream with FV_KEEP_POD=1), then
RUNPOD_API_KEY=… ./fv-lab pod up --sha <7> --family oracle \
  --cells oracle-fasth3-4step-vsa,oracle-fasth3-4step-vsa-diff,oracle-fasth3-4step-vsa-f32,oracle-fasth3-4step-vsa-f32-diff,oracle-fasth3-4step-vsa-bf16-vs-f32 \
  --extra-env FV_ORACLE_URL=https://<upstream pod>-8000.proxy.runpod.net/upstream/<tag> --yes --wait
```

`FV_ORACLE_TARGETS=<t…>` is `--cells` with the arms of `oracle-<t>` (each target's main arm, its `-diff`, and for the
H3 targets `-f32`, `-f32-diff`, `-bf16-vs-f32`; for Ref2VA `-cond`, `-cond-diff`). Refused by `fv-lab run` and `pod
up --extra-env` (the script still reads them): `FV_DET_CELLS`, `FV_DET_REPEATS`, `FV_DET_PROMPT`,
`FV_DET_T2V_PROMPT`, `FV_DET_I2V_PROMPT`, `FV_ORACLE_TARGETS`, `FV_ORACLE_F32`, `FV_ORACLE_OWN_TEXT`,
`FV_ORACLE_PREFETCH`, `FV_REF2VA_PROMPT`, and for `oracle` a `FASTVIDEO_DUMP_OPS` other than `0,1,24,47`. `pod fetch`
takes `determinism.txt`, `hashes.txt` and each arm's `hashes.txt` / `digests.txt`; `fv-lab-compare-runs.sh` compares
the hashes files byte for byte and each arm's `compare-dumps` lines. Still in the script (§15.6): the oracle's ffmpeg
clip-metric targets (`ltx25-{i2v,kf,ref2v,a2v*,retake*,extend}`, `sfwan13`), `FV_ORACLE_OWN_TEXT=1`, det-short's
`ltx-sol`, and `runpod-http.sh upstream` (the reference pods; it moves with an `fv-lab oracle` driver).

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

`fetch` (and `up --wait`) take the summaries as `runpod-http.sh` does; with `--fetch-tree` the whole run directory
instead (`FV_FETCH_TREE=1`: frames, mp4s, wavs, every file), less what `--fetch-skip` matches (paths
`<tier>/<tag>/…`; an ERE of alternatives of literal text, `.`, `\`-escaped punctuation, `^` and `$`, e.g.
`'/frames/'` or `'\.png$|\.cache$'`; anything else is refused before the pod is created) and files above
`--fetch-max-mb` MiB (default 400, 0 no limit; checked while the file streams). `up` takes them only with `--wait`.
A pattern starting with `-` is refused (clap reads it as an option). The pod's listing is not trusted: an entry that
is not a single name inside the run directory (absolute, `..`, `.`, a `/`, `\` or NUL once decoded, a malformed `%`
escape) is refused and logged, and the walk stops 16 levels down. The top-level listing is retried 5 times, and
`RUN_FILES` and `DONE` (what the run is judged by) are fetched whatever the skip says (ADR-0019 §13.3).

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

## Reports: `fv-lab report summary` and `report table`

What the report scripts printed, read from a fetched run directory's files (ADR-0019 §13); no GPU, no network.

```sh
./fv-lab report summary artifacts/runpod/b6/<tag>        # the lines b6's script branch copied into live.log
./fv-lab report summary artifacts/runpod/<dir> --family h3arms   # the tier when the parent is not its name
./fv-lab report summary --kind hd artifacts/runpod/hd/<tag>   # hd_report.py, without keyframe sharpness (no --jpg-out)
./fv-lab report summary --kind h3-graphs DIR             # was experiments/h3-graphs-summary.py
./fv-lab report summary --kind h3-tier-options DIR --json OUT   # was h3-tier-options-summary.py
./fv-lab report summary --kind h3-trim-nvfp4 DIR --json OUT     # was h3-trim-nvfp4-summary.py
./fv-lab report summary --kind fasth3-nvfp4 DIR --json OUT      # was fasth3-nvfp4-summary.py
./fv-lab report table --kind sol-bench artifacts/runpod/sol-bench/<date>/*   # was sol_bench_table.py; results.json beside them (--json F, --stdout)
./fv-lab report table --kind timings <validate.sh run dir>     # timings.sh: <run>/timings.md, printed (--out F, --stdout)
```

`--kind tier` (the default) prints, for `techniques`, `b6`, `b4a`, `h3arms`, `trace`, `ltxfps`, `ltxvae` and
`ltxvae-gen`, the summary lines their `runpod-matrix.sh` branches wrote into `live.log` (compare fields, arm timings,
load-stage lines, reuse counts, gate verdicts, the `/gpu_trace` line, sink checks, decode lines), without the
`[HH:MM:SS] [family] ` stamp, for runs of either runner. The other tiers' branches printed none. `hd_report.py`
(the keyframe sharpness needs numpy) and `timings.sh` (`validate.sh` calls it) stay; `sol_bench_table.py` and the four
experiment `*-summary.py` are deleted (ADR-0019 §13.5).

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
./fv-lab report tiers --runs artifacts/runpod/gold/<tag1>,artifacts/runpod/gold/<tag2>   # a gold run split with --cells, as one
./fv-lab report tiers --check               # CI: docs/perf/tiers.md is what the run set its header names renders to
```

`gold check` prints Markdown (the table CI appends to the job summary) and exits 1 on a gold file that does not parse
or resolve, or a `[last_validated].commit` that is not a commit here (outside a git work tree the commits are not
checked, and it says so). A variant never validated is a warning.

`report tiers` writes one row per gold file. A variant is measured only by the arm of its name in the run set (the gold
tier names its arms by variant id), with every value read from that arm's `summary.json`, `benchmark.json` or
`gpucheck-out/engine.json` and the run's `pod.json`; anything else is "not measured" with the reason. Options:
`--gold DIR` (other gold files), `--out FILE`, `--stdout`.

`--runs` takes several run directories (comma separated or repeated) of one tier and one commit, e.g. a gold run
split in two with `--cells` ([ADR-0019 §16.7](../adr/0019-fv-lab.md#167-the-gpu-check-the-coordinators)), and
renders them as one run set: the arms are merged, each row reads its own run's `pod.json`, and the header names every
run (its tag, end, GPU and price). Refused: runs of different tiers or commits (where `pod.json` or the tag names
one), the same directory twice, and an arm that ran in more than one run. An arm may be in several runs only when
every copy but at most one was skipped; the copy that ran is shown (all skipped: the first in `--runs` order).
`--check` reads the runs back from the header (`` Run set: `a`, `b` ``). One run renders exactly as before.

Workflows: `gold-check.yml` runs the three checks on lab, family and lab-crate changes; `gold-run.yml`
(`workflow_dispatch`, input `confirm: spend`) runs `pod up --family gold --yes --wait` and renders the table as an
artifact. It needs the `RUNPOD_API_KEY` repository secret, which is the owner's to add; without it the job refuses.

## Weights from the volume and R2: `volume+r2`

Design: [ADR-0019 §16](../adr/0019-fv-lab.md#16-weights-by-tree-from-two-places-weights--volumer2). A tier's
`weights = "volume+r2"` (the `gold` tier) reads every tree the EU volume holds from the volume and fetches every other
tree from the R2 weights store onto the container disk, right before the first arm that reads it:

- `{weights}` is `/fvscratch/weights`: a symlink per volume tree into `/workspace/weights`, the fetched trees beside
  them. The volume is never written.
- Per arm, when a tree it reads is not on the disk yet: the fetched trees it does not read are removed, then
  `r2-weights.py plan <dest>@<rev> --pin weights-r2.tsv` (the revisions and manifest SHA-256s of
  `scripts/gpu/weights-r2.tsv`) and `r2-fetch.py` fetch the missing ones under the arm's cap. Logs: `<arm>/r2-fetch.log`,
  `<arm>/r2-fetch.json`, the `r2 <arm>: …` lines of `live.log`. The runtime image has no Python (by design), so the
  first fetch installs `python3` with apt (`r2-apt.log`, at most 10 min; a failure is retried at the next fetch).
- A failed fetch records the arm as skipped (`r2 fetch failed`); a tree in neither place as `weights incomplete`.
- The root is never on the volume: run by hand, pass `--scratch` (or `FV_SCRATCH`) on the container disk, e.g.
  `fv-lab run gold --all --work /workspace --scratch /fvscratch`; without it (the scratch disk is then the work
  directory), or with a scratch under `/workspace`, the run is refused (exit 2).
- The idle watchdog of a `volume+r2` pod waits 85 minutes at 0 % GPU (a fetch uses none), not 10.
- Any pod's run gets `FV_RUN_DEADLINE` from `pod up` (the cap less 20 min): arms that could not finish by then are
  skipped as `wall cap near`, so `DONE` is written, and `pod up --wait` stops waiting 15 min before the cap and
  fetches what is there.
- The R2 key is two Runpod secret references in the pod's env (`fv_r2_access_key_id`, `fv_r2_secret_access_key`, filled
  in by Runpod); `fv-lab run` gives it to the plan step only. Nothing to set locally.
- `fv-lab run gold --all --dry-run …` prints the plan: one `dry-run weights …` line (18 trees, 530.74 GB, at most
  147.86 GB at once) and a `dry-run r2 <arm>: drop …; fetch …` line before each arm that fetches. The gold tier's
  container disk (300 GB) holds the largest arm (`h3-8step`, 147.86 GB) plus the 120 GB every volume tier has
  (`tests/gold_tier.rs` checks it).

The full gold run, its smoke run and the cost: ADR-0019 §16.7 (about 1.5-3.1 h, $3.70-$7.70, at most $9.96 at the 4 h
cap).

## Keeping it equal to the scripts

```sh
bash scripts/gpu/tests/fv-lab-goldens.sh           # regenerate crates/fastvideo-lab/fixtures/ from the scripts
bash scripts/gpu/tests/fv-lab-goldens.sh --check   # CI (control.yml)
bash scripts/gpu/tests/fv-lab-report-goldens.sh [--check]   # the reports' fixtures (fixtures/report/, ADR-0019 §13.2)
bash scripts/gpu/tests/fv-lab-experiment-goldens.sh [--check]   # the experiment tiers' commands (ADR-0019 §14.3)
```

The report goldens are the report scripts' output on committed run directories (`python3 -I`, `env -i`,
`LC_ALL=C`); `tests/report.rs` compares fv-lab with each byte for byte. The retired scripts run from `f9e57ea2` when
the history has it (a full clone); in a shallow clone their fixtures are kept as frozen. The deleted branches' summary
lines are frozen in `fixtures/report/summary-script.sh`, each block checked against `runpod-matrix.sh` at `5987e6a4`
when the history has it, and every line of the script-side runs is checked to be in that run's `live.log`.

For every tier of `lab/tiers.toml`, the fixtures are the script's dry run (`<tier>-dry-run.txt`), its weight gates
(`<tier>-gates.tsv`: the cells the dry run gated, with their `verify-weights.sh --list` needs) and `runpod-http.sh
plan`'s pod body (`runpod-http-<tier>.json`).

**Frozen since the GPU gate.** Once a tier passed its GPU gate and its script branch is deleted (it is listed in
`scripts/gpu/fv-lab-retired.sh`), the scripts can no longer produce its fixtures. They stay as they were written
before the branch went and are fv-lab's own goldens from then on: the goldens script does not regenerate them, only
checks that they exist, that the tier is in `lab/tiers.toml` and that its branch is gone (and that every other script
tier still has one); `tests/dry_run.rs` checks the same. The Rust tests still compare fv-lab's dry run, gates and pod
body with them, so a change to a frozen tier's commands or pod is deliberate: edit the cell (or `pod-start.sh.in`)
and the fixture in the same PR, and say why in the PR. The eleven retired tiers are frozen (`b4a`'s since its GPU
gate, ADR-0019 §11.6), and so are the two
restored modes (`precision-nvfp4`, `ltxvae-gen`), whose fixtures come from the script at `5987e6a4` in that mode
(`FV_LAB_RESTORED` in `fv-lab-retired.sh`; `fv-lab-restored-goldens.sh --check` reproduces them); `gold` was never a
script family (only its pod body comes from `runpod-http.sh plan`). The `gold` tier is no script family: only its pod body is written there, and its dry run is fv-lab's own golden
(`fixtures/gold-dry-run.txt`, `tests/gold_tier.rs`). The experiment tiers (`FV_LAB_EXPERIMENTS`, D1e) are the same: only
their pod body is written there, their dry run is fv-lab's own golden (`fixtures/<tier>-dry-run.txt`) and
`fv-lab-experiment-goldens.sh` holds their commands to the experiment's pod script (`tests/experiment_tiers.rs`). Once, there is also its `frames_digest` of a sample. A new tier fails
`--check` until its fixtures are written; `tests/{dry_run,payload_golden,frames_digest,fv_lab_bin}.rs` hold fv-lab to
them. The script runs every script under `env -i`, so your `FV_*` / `RUNPOD_*` settings do not leak in.
