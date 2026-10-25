# ADR-0023: The tier table and the gold CI bridge (plan D3, D4)

- Status: Accepted for the CPU parts of D3 and D4 (`fv-lab report tiers`, `docs/perf/tiers.md` with no run set,
  `fv-lab gold check`, the `gold` tier and its cells, the two workflows). The GPU gold runs that fill the table are
  not run here (budget); the `gold-run` workflow needs the owner's `RUNPOD_API_KEY` secret (§4)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-lab-report-ci-bridge
- Code: `crates/fastvideo-lab/src/{report,goldcheck,goldcell}.rs`, `src/bin/fv-lab.rs` (`report tiers`, `gold check`,
  `gold cells`, `run --all`, `pod down --from-ledger`, `pod up --cells` checked); `src/pod.rs` and `src/pod-start.sh.in`
  with `scripts/gpu/runpod-http.sh start_cmd` (the pod's own wall cap, §4); data `lab/tiers.toml` (`gold` tier), `lab/cells/gold-*.toml` (generated);
  `docs/perf/tiers.md` (generated); `.github/workflows/{gold-check,gold-run}.yml`; fixtures
  `crates/fastvideo-lab/fixtures/{tiers/,gold-dry-run.txt,runpod-http-gold.json}`; tests
  `tests/{tiers_report,gold_tier}.rs`
- Builds on: ADR-0009 (gold files), ADR-0019 (fv-lab, its run sets, tiers and cells)
- Plan: D3, D4 (`docs/plan/2026-10-23-execution-plan.md`)

## Context

ADR-0009 gave every served variant a gold file whose numbers are cited or declared gaps, with an empty
`[last_validated]`, and named D3's tier table and D4's CI bridge ("every file parses and `last_validated.commit` is an
ancestor of HEAD no older than N days"). ADR-0019 gave the run plan (tiers, cells) and the run set a pod leaves in
`artifacts/runpod/<tier>/<tag>/` (`summary.json` per arm, fv-gpucheck's `benchmark.json` or `gpucheck-out/*.json`,
`pod.json`). Nothing joined the two: no table was generated from a run set, no CI step read the gold files, and there
was no command that runs every gold variant. The gold runs themselves are GPU work that the budget does not cover now,
so this decides and builds everything they need, so that one run set fills the table.

## Decision

### 1. `fv-lab report tiers`: one run set, rendered, never invented

`fv-lab report tiers --runs artifacts/runpod/<tier>/<tag>` writes `docs/perf/tiers.md`: one section per family, one row
per gold file (gold-file order). A variant is measured by the arm of the same name in the run set, and only when that
arm is its gold cell: the run set is the `gold` tier's (which names its arms by variant id, §3), or the arm's engine
report ran that variant (`context.model.id`). A same-named CLI arm of another tier (`rtx6000`'s `sol-h3`) is shown as
"not its gold cell", never as the variant's measurement. The columns come from the gold schema and what the runs record:

| column | from |
|---|---|
| variant, tier | the gold file |
| status | the arm: `measured`, or `not measured (<why>)`: no run set, no arm, skipped (the runner's reason), failed (its exit), no timing record |
| GPU | `benchmark.json` `gpu`, else `pod.json` `gpu` |
| cell, `vs gold` | measured width x height x frames and steps (`benchmark.json` `workload`, or the engine report's `mp4` / `job` / `progress`), and whether that is the gold file's `[workload]` |
| wall s and what it measures | `gen`: `benchmark.json` `total_s` (one warm `generate()`); `job`: `engine.json` `submit_to_mp4_s` |
| text / denoise / decode / load s, peak GiB | the same record's stage times, `load_s`, peak device memory; decode is the decode through the finished MP4 in both kinds (`benchmark.json` `decode_s`; the engine's `video_decode` plus its `encode`) |
| `$/clip` | the run's own `pod.json` `usd_per_hr` x its own wall time; `-` when the run recorded no price |
| last validated | the gold file's `[last_validated]` date and commit, and its age in days before the run set ended |

Nothing is copied from elsewhere: a row that is not measured shows `-`, not the gold file's cited number (those stay in
the gold file and its sources). `$/clip` is the only derived value, from two numbers of the same run.

The rendering is deterministic, and its header line `Run set: <path>` (or `none`) names its input, so `fv-lab report
tiers --check` re-renders from the run set the committed file names and fails on any difference (first differing line
in the message). The golden is a fixture run set (`fixtures/tiers/runs/gold/…`: a serve-engine arm, a CLI arm at another
canvas, a skipped arm, a failed arm, an arm of no gold variant, a variant with no arm) and fixture gold files, rendered
to `fixtures/tiers/tiers.md` (`tests/tiers_report.rs`), which also checks the committed `docs/perf/tiers.md`.

**Seeded with no run set.** No committed run set measured a gold variant's cell: the only arm named like a variant is
`sol-h3` in `artifacts/runpod/rtx6000/492e28a-*`, and that is the `rtx6000` family's CLI arm (`h3 gen --h3-recipe
sol-h3` without the variant's technique profile, at 1344x768, not the gold 832x480), not the served variant. Showing it
as `sol-h3`'s measurement would mislabel it. So `docs/perf/tiers.md` is rendered with `Run set: none`, every variant
"not measured (no run set)"; the first gold run replaces it.

### 2. `fv-lab gold check`: parse, resolve, commit, age

- **Parse**: `lab/gold/*.toml` through `gold::load_dir` (ADR-0009's validation), and every family variant has a gold
  file.
- **Resolve** against `families/*.toml` (`fastvideo_manifest::family::load_dir`, A2's loader): the variant exists as
  a variant id in the family the file names, with the same tier and profile; a `registry_name` equals the family
  file's (both without `@rev`, which family files never carry); a `rev` needs a `registry_name`. A family variant with
  a registry name the gold file does not carry is shown `(not certified)`, not failed. The rev itself is the published
  manifest's: no file in this repository holds it, so the check cannot confirm it beyond its shape (ADR-0009).
- **Commit**: `git rev-parse --verify --quiet <commit>^{commit}` through `std::process::Command` (no dependency); when
  that fails, `git rev-parse --disambiguate=<commit>` tells an ambiguous short sha (its own failure: write more digits)
  from a missing one. Outside a git work tree (`git rev-parse --is-inside-work-tree` fails, or no `git`), commits are
  not checked and the report says why.
- **Age**: days from `last_validated.date` to today (UTC, or `--today`), and commits since
  (`git rev-list --count <c>..HEAD`, `-` when git cannot say, e.g. a shallow clone), marked "not an ancestor of HEAD"
  when `git merge-base --is-ancestor` says so (then the count is HEAD's commits the validated commit lacks).

Output: a Markdown table (variant, family, tier, registry, last validated, age, commit, commits since, status) on
stdout, for `$GITHUB_STEP_SUMMARY`; warnings and errors on stderr. Exit 1 on a parse or resolution failure, a validated
commit that is not in the repository or is an ambiguous short sha, or a date after today. Never validated is a warning (every file today). Age is
warn-only by default; `--max-age-days N` fails a validated variant older than N days. "Served" is every family variant:
ADR-0009 gives each one a gold file. ADR-0009's "ancestor of HEAD" is weakened to "exists in the repository" for now,
because a validation run of a branch commit merged with `--no-ff` is not an ancestor of the branch's later heads until
merged; `commits since` makes the distance visible instead.

### 3. The `gold` tier: each variant's canonical cell, generated

`fv-lab run gold --all` runs one arm per gold file. The arm is the served variant through the serve engine:
`fv-gpucheck --keep-going --mode fast engine --model <variant> --weights-root {weights} --prompt … --seed …
--width/--height[/--num-frames] … --clip-out {runs}/{arm}/out`. `fv-gpucheck engine` loads the variant from the family
files (the engine's one catalog) with its recipe and technique profile, as fv-serve does, so the cell needs no
per-variant flags and cannot disagree with what is served. It gates on `verify-weights.sh variant:<id>` and names the
family variant's trees. The canvas and length are the gold file's `[workload]`; the seed its first seed (else 1024);
the prompt `runpod-matrix.sh`'s default (the engine runs one request; the gold prompt set is named in the cell's note,
not run); the gold file's `workload.env`, which reproduces an older measurement, is not applied: the gold run measures
the served default, and the report's `vs gold` column shows where the canvas moved.

The cells are ordinary cell files (`lab/cells/gold-<variant>.toml`, no schema change) in an ordinary tier (`gold`, the
`rtx6000` tier's pod: PRO 6000, $2.50/hr cap, 4 h wall cap, 3600 s per arm, the same preflight and TAE fetch), so the
runner, the pod, `pod fetch` and the report need nothing new. They are generated by `gold::gold_cells`
(`fv-lab gold cells --write`), and `tests/gold_tier.rs` holds the committed files equal to the generation, the tier's
cell list equal to the gold files in order, each arm equal to its variant (name, `variant:<id>` gate, trees, canvas),
and `fv-lab run gold --all --dry-run` equal to `fixtures/gold-dry-run.txt`. That dry-run golden is fv-lab's own: the
gold tier has no `runpod-matrix.sh` family to be equal to, so `fv-lab-goldens.sh` writes only its pod body
(`runpod-http-gold.json`, from `runpod-http.sh plan` as for every tier, which `tests/payload_golden.rs` checks), and
the script-equality tests (`dry_run.rs`, `lab_files.rs`) leave the gold tier out. `--all` names the whole tier
explicitly (the default for a tier; refused for a single cell and with `--cells`).

On the EU volume about eleven arms run (the H3 and LTX-2.5 variants whose trees are there); the Wan, LongLive, trim,
NVFP4, VDN and turbo-LoRA trees are R2-only, so those arms are recorded as skipped (`weights incomplete`) and the
table says so. Variants the engine cannot run from a text request (reference-to-video, audio-to-video) fail and are
shown as failed: that is a finding, not a number.

### 4. CI

- **`gold-check.yml`** (pull requests and pushes touching `lab/**`, `families/**`, `crates/fastvideo-lab/**`,
  `docs/perf/tiers.md`, `Cargo.lock`, the workflow; and `workflow_dispatch`): builds fv-lab, runs `fv-lab gold check`
  with its table appended to `$GITHUB_STEP_SUMMARY` (the visible per-model validation age), `fv-lab gold cells --check`
  (which also catches an orphan `gold-*.toml` and a `gold` tier list that is not the gold files in order) and `fv-lab
  report tiers --check`. It also runs on `artifacts/runpod/gold/**` (a committed run set the table may name), the root
  `Cargo.toml` and `rust-toolchain.toml`. Full history (`fetch-depth: 0`) for the commit checks. No secrets.
- **`gold-run.yml`** (`workflow_dispatch` only): inputs `sha` (the runtime image), optional `cells`, and `confirm`,
  which must be `spend`. It refuses to start when `RUNPOD_API_KEY` is absent, before checking out or building
  anything. Each `cells` word must match `^[a-z0-9][a-z0-9._-]*$` and be an arm of the gold tier
  (`lab/cells/gold-<id>.toml`), and `pod up --cells` itself refuses a name that is no arm of the tier, so no input can
  pass a flag (`--state-dir` would have moved the records the cleanup reads). It runs `fv-lab pod up --sha <sha>
  --family gold --tag <sha>-<MMDDHHMMSS>-<run id hex> --state-dir artifacts/fv-lab --yes --wait`: fv-deploy's guards
  (price cap on quote and bill, $8 floor projected over the cap, detached backstop at the 4 h cap, 600 s x 3
  readiness), `fv-lab run gold` on the pod, fetch into `artifacts/runpod/gold/<tag>/`, delete. The tag is unique per
  workflow run, and the table is rendered from exactly that run set into the job summary; the run set, the ledger and
  the table are uploaded. It commits nothing.
- **No pod outlives the job unguarded.** Three layers, because the detached backstop lives on the runner and the
  runner's orphan cleanup kills it, and a cancel sends SIGINT then SIGTERM about 7.5 s later, whose second signal
  ends `pod up` (`exit(130)`) before its own cleanup; the pod record (`<state>/pods/<id>.json`) is only written
  once the pod is ready, up to about 30 min after the create, and not at all when `up` fails to delete:
  1. fv-deploy's ledger line `runpod-pod-created <id> …` is written right after the create, before the backstop and
     readiness. An `always()` step runs `fv-lab pod down --from-ledger`, which deletes every pod the ledger shows
     created and not deleted (`pod::undeleted_in_ledger`; a `delete-failed` line keeps it listed; a 404 counts as
     deleted). `pod down <id>` no longer needs a pod record either.
  2. Every fv-lab pod deletes itself at its tier's wall cap: the start command (`src/pod-start.sh.in`, and
     `runpod-http.sh start_cmd` in lockstep, so the payload goldens stay equal) starts a cap watchdog that sleeps
     `wall_cap_s` (`FV_POD_CAP_S` on the script side) after the container starts, then terminates the pod with the
     pod-scoped key Runpod injects (GraphQL `podTerminate`, below), retrying each minute until it succeeds. It counts from the container start, so it fires up to the
     boot time after the driver-side backstop would have; it is the guard for a dead runner, not a tighter cap. A
     `StartOptions` without a cap is refused. (The script's `upstream` mode start command is unchanged.)
  3. The idle watchdog (10 GPU-idle minutes), now through the same GraphQL call, so it actually works (below).

  **The pod-scoped key cannot delete through REST (live evidence, coordinator, 2026-10-24 21:16–21:45Z, EU-RO-1 with
  `bz0ofkfx45`).** (1) RTX PRO 6000 pod `pq69qpoh5b0mxy` via `runpod-http.sh run` with `FV_POD_CAP_S=240`, the driver
  and its backstop killed right after the create: the REST cap watchdog of the first version of this change did not
  fire; the pod was still running nine minutes later and was deleted by hand. (2) A CPU probe pod (python image):
  `RUNPOD_API_KEY` and `RUNPOD_POD_ID` are set inside, but `DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID`
  with `Authorization: Bearer $RUNPOD_API_KEY` answers HTTP 403 every time. (3) A second CPU probe pod: `POST
  https://api.runpod.io/graphql`, `Content-Type: application/json`, the same Bearer key, body
  `{"query":"mutation { podTerminate(input: {podId: \"<RUNPOD_POD_ID>\"}) }"}` answered 200
  `{"data":{"podTerminate":null}}` and the pod was gone within a second. So every in-pod self-delete through REST
  with the pod-scoped key never worked, the existing idle watchdogs of `runpod-http.sh` and fv-lab's start command
  included (their documented "the pod deletes itself after 10 idle GPU minutes" was not true until now).

  Both watchdogs now call `fv_self_delete`: curl `POST https://api.runpod.io/graphql` with that body and the Bearer
  key (never printed), the response into `<log>.resp` and the log; success is HTTP 200 with no `"errors"` in the
  body, anything else is retried (the cap watchdog every 60 s until success; the idle watchdog each minute while
  idle). No REST DELETE remains in either start command. `pod::tests::every_pod_deletes_itself_at_its_wall_cap` pins
  the call (endpoint, mutation, success rule, one definition per watchdog) and the absence of `rest.runpod.io` and
  `-X DELETE`; the payload goldens are regenerated from the script and equal. The script's `upstream` mode shares
  `idle_watchdog`, so its pods get the fix too.

  **Follow-up, not changed here** (in-pod REST self-deletes with the pod-scoped `RUNPOD_API_KEY`, the same defect):
  `scripts/serve/e2e/pod-boot.sh:49`, `scripts/serve/e2e/ltx-pod.sh:49`, `scripts/gpu/family-pool-pod.sh:40`,
  `scripts/gpu/family-pool-validate.sh:82`, `scripts/gpu/sol-bench-pod.sh:41`, and the one-offs under
  `scripts/gpu/experiments/` (`h3-tier-ship-validate.sh:126`, `h3-fp4attn-validate.sh:110`,
  `h3-tier-options-pod.sh:81`, `h3-turbo-loras-validate.sh:96`, `vdn-h3-validate.sh:120`, `h3-glue-pod.sh:52`,
  `h3-glue-validate.sh:113`, `vdn-h3-pod.sh:53`, `h3-graphs-validate.sh:107`, `h3-graphs-pod.sh:92`,
  `h3-fp4attn-pod.sh:82`, `h3-tier-ship-pod.sh:74`, `h3-trim-pod.sh:56`, `h3-trim-validate.sh:120`,
  `h3-tier-options-validate.sh:126`, `fasth3-nvfp4-validate.sh:93`, `h3-trim-nvfp4-pod.sh:48`,
  `fasth3-nvfp4-pod.sh:56`, `h3-turbo-loras-pod.sh:49`, `h3-trim-nvfp4-validate.sh:83`; plan D1e retires them). With
  another key, to be checked: `control/src/cluster/payloads.ts:54` and `crates/fastvideo-deploy/src/boot/watchdog.rs:139`
  (`delete_runpod`, REST with `FV_BACKSTOP_API_KEY`, which also counts any HTTP answer, a 403 included, as done).
  `scripts/dev/build-pod-server.py:73` already falls back to GraphQL. `artifacts/perf/h3-trim-vae-mxfp8/driver/`
  is committed evidence and stays as run.
- **The secret's scope.** `RUNPOD_API_KEY` is a step-level env of the three steps that call Runpod (the refusal check,
  `pod up`, `pod down`), never the job's: the cargo build scripts and the third-party actions never see it. The
  third-party actions of both workflows are pinned by commit, read from their tags with `git ls-remote` on 2026-10-24:
  `actions/checkout` v4 `11d5960a…`, `actions/upload-artifact` v4 `ea165f8d…`, `Swatinem/rust-cache` v2 (v2.9.3)
  `d1527c8f…`, and `dtolnay/rust-toolchain` at its `stable` branch head `b725eda5…` (with `toolchain: stable`).
- **Re-running spends again.** "Re-run jobs" (or "Re-run failed jobs") on a `gold-run` run creates a new pod and
  costs another pass; the inputs, `confirm: spend` included, are reused without asking.

**Owner decision needed:** the repository has no `RUNPOD_API_KEY` secret, and this change does not add or request
one. Recommended: store it as an environment secret of the `gold-runs` environment (Settings, Environments), with
deployment branches restricted to `main` and required reviewers, so a run waits for a person's approval and a branch
cannot use the key. The job already names `environment: gold-runs`; until the owner configures it GitHub creates it
unprotected on the first run, and without the secret the job refuses anyway. Until the owner adds it (and accepts about $2.49/hr for roughly an hour per gold run), `gold-run` refuses, and the
gold runs go through `fv-lab pod up --family gold` by hand as every GPU run does today.

### What a gold run then does (not automated here)

Commit the fetched run set, `fv-lab report tiers --runs artifacts/runpod/gold/<tag>` and the table; then, per measured
variant, update the gold file by ADR-0009's rules (the run record as the `[[source]]`, `[last_validated]` with the
commit, run and date, gaps closed only for fields the run measured). Writing gold files automatically is a later step
with its own review: it changes the files' numbers.

## Consequences

- One run set fills `docs/perf/tiers.md` with one command; `--check` keeps it from drifting.
- CI shows every variant's validation age on every lab or family change, and fails on an unresolvable gold file or a
  validated commit that does not exist.
- A gold file change regenerates its cell (`fv-lab gold cells --write`), or CI fails.
- No new dependency, no schema change (tiers v1 and cell v1 are unchanged; the `gold` tier is data).
- Not done: the GPU gold runs, the automatic gold-file update, a hard age limit (`--max-age-days` exists, CI does not
  pass it until variants are validated), and confirming `@rev` against published manifests (the registry's).

## Validation

CPU, on the build pod: `cargo fmt --all --check`, `cargo clippy --locked --workspace --all-targets -- -D warnings`,
`cargo clippy --locked -p fastvideo-lab --all-features --all-targets -- -D warnings`, `cargo test --locked -p
fastvideo-lab --all-features`, `cargo test --workspace --lib --locked`, `bash scripts/gpu/lint.sh`; locally
`bash scripts/gpu/tests/fv-lab-goldens.sh --check`, `fv-lab gold check`, `fv-lab gold cells --check`, `fv-lab report
tiers --check`. GPU: none here.
