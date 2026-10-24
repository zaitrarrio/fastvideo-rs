# Execution plan for the architecture proposal

Date: 2026-10-23. Source: `docs/gaps/2026-10-22-architecture-review.md` §7
(revised 2026-10-23 for the registry split, §7.11) and the owner's
instruction to plan and begin execution. This file is the live plan: the
status table is updated by every PR that completes a work package, and the
execution log at the end records what ran where and what it cost.

## 1. Operating rules

- **Where code compiles and tests run.** Never in an agent's container
  (CLAUDE.md "Builds"): on a CPU build pod on the `fv-build` network volume
  (`pxy4hlsnwq`, EU-RO-1), driven by `scripts/dev/build-pod.sh run <agent>
  -- <cmd>` with each agent's own worktree and `CARGO_TARGET_DIR`. The
  coordinating session provisions the pod (fv-control when a token is
  available, else directly on the Runpod REST API with the same payload
  `control/src/buildpods.ts` builds: base image pin, `build-pod-server.py`
  from main, a session token, idle stop, hard cap and backstop) and records
  it in the execution log.
- **GPU pods (owner instruction 2026-10-23).** Only pods that mount the EU
  weights volume `bz0ofkfx45` (`fv-weights-h3-ltx-500`, EU-RO-1); RTX PRO
  6000 by default (the gold-standard GPU; the volume's trees cover H3 and
  LTX). A pod that is not ready (container running, `/ping` or the stage
  service answering) within **10 minutes** of creation is deleted and a new
  one acquired; after three cycles, stop and report. Every pod has a
  wall-clock backstop and is deleted when its cells are done. Stop before
  the Runpod balance would fall below $8 (CLAUDE.md). Balance at plan time:
  $22.61, so GPU time in this round is limited to a few hours; the plan
  defers GPU cells to the points marked **GPU** and runs CPU gates first.
- **Branches and merges.** One branch and one PR per work package
  (`claude/<wp-id>-<short>`), opened as a draft by the agent that did the
  work; the coordinating session reviews, merges main into the branch, waits
  for CI green on that head, and merges with `--no-ff` (CLAUDE.md "Merging
  to main"). Sub-agents never merge.
- **Byte identity first (review §7.1).** Every inference-core structure
  change carries a `techniques` byte-identity cell per family before merge
  (**GPU**); kernel changes never ride along with structure changes.
- **Pinned public formats.** `fastvideo-manifest` layer 1 is the registry's
  contract; widenings happen before the first publish, with ADRs and
  fixtures; afterwards a change is a versioned schema.
- **Owner decisions still open** (review §7.10): Vast serving, Runpod LB
  mode (`multiworker.rs`), one image with boot-time config versus seven
  variant images, GMI/Brev in Rust or TS. Work packages that depend on one
  are marked and wait.

## 2. Work packages

Status: `todo`, `active`, `pr` (draft PR open), `merged`, `blocked`.
Effort is one engineer-week unless noted; the calendar is shorter with
parallel agents.

### Phase 0: cleanup and gates (no design risk)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| F1 | Delete the six `scripts/vast-*.sh`, `scripts/docker-build-cuda.sh`, `scripts/gpu/serverless-worker.sh`, `scripts/gpu/upstream_bench.py`, `scripts/gpu/build-remote.sh`, `scripts/gpu/prompts-flux2.json`, `scripts/gpu/sage/{run,vast-run,setup}.sh`; move the 24 port one-offs and `runpod-matrix.sh` to `scripts/gpu/experiments/`; remove the `h3-*`/`ltx2-*` tiers from `validate.sh` and the README Vast sections; fix `docs/serve/images.md:87`, `scripts/gcp/weights.sh:3-4`, ADR-0001 and `docs/scope.md` (removed backends, `unsafe` scope); add `docs/ports/CHECKLIST.md` | no reference to a deleted path remains (`grep`), `scripts/gpu/lint.sh` passes, README describes the Runpod path | `bash scripts/gpu/lint.sh`; `task check` lint part on the pod | none | merged (#139) |
| F2 | One formatting commit; fix the 27 clippy lints; `task check` runs `cargo fmt --check` and `clippy -D warnings` as real gates; `gpucheck-t0` runs them | `cargo fmt --check` and `cargo clippy --workspace --all-targets -D warnings` green | pod | F1 (to avoid churn) | merged (#142) |
| F3a | Serving hardening from review §7.7: per-job deadline in `StepControl`, bounded engine event channel, per-model duration and queue-wait histograms, residency gauges, `otel` feature implemented | tests in `fastvideo-engine-service` and `fastvideo-serve`; `scripts/serve/check.sh` green | pod | none | merged (#140; `otel` deferred to F3c) |
| F3c | Implement the `otel` feature (OpenTelemetry export of the trace recorder's spans) | spans visible in an OTLP collector | pod | F3a | merged (#150; live-collector check pending) |
| F3b | Delete `multiworker.rs` and `server.workers_max`; rename `[gateway]` to `[cluster]`, drop `gw_workers` | config round-trip tests | pod | owner decision (Runpod LB mode) | blocked |

### Phase 1: the manifest (Workstream A)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| A1 | `crates/fastvideo-manifest` layer 1: modules `name`, `model_toml`, `manifest`, `lineage`, `sig`, `schema`; the registry's WO-001 contracts verbatim plus the two widenings (`[variant]`, `[runtime]` recipe/profile, `[caps]`); fixtures under `crates/fastvideo-manifest/fixtures/` (valid and ≥25 invalid names with error kinds, model.toml valid/invalid, manifests with `.digest` siblings, test keys, lineage chains); proptest round-trip; `schemas/*.json` export with a drift test; ADR-0005 naming, ADR-0006 model.toml, ADR-0007 manifest/digest/lineage/signatures | all WO-001 acceptance tests green on CPU; no I/O, no CUDA, `unsafe` forbidden | `cargo test -p fastvideo-manifest` on the pod; clippy -D warnings | none | merged (#143) |
| A2 | Layer 2 types (`family`, `variant`, `tiers`) and the first three family files `families/{h3,ltx2,wan}.toml` authored from `engine-service/src/cuda/caps.rs`; an equality test proving `catalog()` equals the manifest projection row for row | equality test green for every catalog row | pod | A1 | merged (#147) |
| A3 | Engine projection: `model_from_config` resolves through the manifest; `validate.rs` and `protocol::ModelCaps` ranges come from it; `[aliases]`, `served_names`, `[protocols]` derived; the 35 serve TOMLs reduced to variant ids and residency; the fake backend serves the manifest; `catalog()` deleted | zero hand-written catalog rows; zero `[aliases]` blocks; `dimension_sweep.rs` and `tests/e2e.rs` green; `/fv/v1/capabilities` unchanged for every config (golden) | pod; serve-compat CI | A2 | pr (#152) |
| A4 | Control-plane and script generation from the families: `control/gen-configs.mjs` emits presets, `enums.ts` ids, `recipe-trees.ts`, `r2-trees.ts`, `weights-sources.ts`, `volumes.ts`; `variants.sh` tables, `volumes.sh`, `gcp/families.tsv`, `autoscale.toml` pools, `verify-weights.sh` cells and `docs/serve/images.md` table generated with `--check` in CI | `gen-configs.mjs --check` green; control unit tests green | pod (node + vitest) | A2 | merged (#149) |
| A5 | Registry hand-off: publish layer 1 (git dependency at a pinned rev); the registry's WO-001 scaffolds against it | registry WO-001 green | registry repo | A1 | active |

### Phase 2: inference core structure (Workstream B, first half)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| B1 | Extract the runtime: `cudarc/src/wan/{tensor,device,ops,kernels,nn,weights,offload,quant,log,dump,ledger,envflag,resident,stage_upload,hostmem,writer}` to `cudarc/src/rt/`; `kernels.cu` split by region and concatenated at build time; `gpucheck` and `engine-service` import `rt` | byte-identical SASS and clips | CPU tests + nvrtc gate on the pod; **GPU** `techniques` byte-identity cell for H3, LTX, Wan | F1 | merged (#146) |
| B2 | `VideoFamily` trait, common `Request`/`Conditioning`, `Request::validate(&ModelCaps)`; the 21 registry tables, `resolve()`, `all_registered_ids()` and the 20 `generate_*` functions replaced by the manifest and a static family list; `AvGenerateOptions` reduced to common fields | CLI behaviour unchanged for every preset (golden `--help` and tiny CPU generations); 83 Hub ids resolve | pod; **GPU** one tiny generation per served family | A2, B1 | todo |
| B3 | Settings layer: the `FASTVIDEO_*` table generated from one declarative list; the 111 direct reads routed through `settings::var`; a `Settings` snapshot passed to `load`; `model_spec` for Wan and the small families; `docs/scope.md` table generated | zero direct `env::var("FASTVIDEO_` outside `settings`; off-identity invariant tests | pod; **GPU** byte-identity cell | B1 | pr (B3a #153; B3b todo) |

### Phase 3 and 4: deployment (Workstream C)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| C1 | `[deploy]` section of the family manifest; presets, variant tables, `families.tsv`, autoscale pools and the Dockerfile variant stages generated from it | `--check` green; one image with boot-time config if the owner decides so | pod | A4 | todo |
| C2 | `fv-boot` in `fastvideo-deploy`: driver check, config materialization, weights plan (volume / registry `hydrate` / hub / none), sidecar, tunnel and phase reports, watchdog, exec; fake provider and fake chunk source tests; the ten boot variants deleted | one boot script in the tree; fv-control sends env only | pod; **GPU** one pod boot on the EU volume | C1, registry WO-002 for the `hydrate` source (volume and hub sources first) | todo |
| C3 | `fv-deploy` with the `Provider` trait (from `fastvideo-autoscale`): `runpod::{Pod,Serverless}`, `vast`, `gcp`; shared money guards, backstop, ledger, templates, release promote/rollback; fv-control on the shared `Launch` schema | `fv-deploy up` per provider with the same smoke | pod; **GPU** one `fv-deploy up` on Runpod EU | C2 | todo |

### Phase 5: validation driver and gold standards (Workstream D)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| D1 | `fv-lab` binary: `pod`, `run`, `report`, `gate`/`oracle` moved; `lab/tiers.toml`, `lab/cells/*.toml`; the 24 port one-offs and `runpod-matrix.sh` retired | `fv-lab run rtx6000` reproduces today's `rtx6000` family | pod; **GPU** one `rtx6000` run | C3 (`Provider`) | todo |
| D2 | `lab/gold/<variant>.toml` for every served variant from the numbers in `docs/perf`, `docs/oracle.md` and the profile comments; keyed by registry name and `@rev` | every served variant has a gold file; comments in `profiles/` no longer carry numbers | pod | A2 | merged (#151) |
| D3 | `fv-lab report tiers` and the generated `docs/perf/tiers.md` | table generated from one run set | pod; **GPU** the gold runs | D1, D2 | todo |
| D4 | CI bridge: gold files parse and resolve; `last_validated.commit` age check; an on-demand `fv-lab run gold --all` workflow | visible per-model validation age in CI | CI | D3 | todo |

### Phase 6: API conformance (Workstream E)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| E1 | Vendored upstream specs under `docs/serve/specs/` with a drift script; `schemars` schemas for native and director | drift script green | CI | none | merged (#144; schemars schemas deferred to E2) |
| E2 | `ParamSpec` per (surface, model) derived from the manifest caps; adapters validate from it; duplicated constants deleted; LTX body typed | one source for every limit | pod; serve-compat | A3 | todo |
| E3 | Certification matrix harness over every surface, parameter x surface x model TSV, CI assertion that every parameter has a valid and an invalid case | TSV with no empty cell | pod | E2 | todo |
| E4 | `cargo-llvm-cov` on the pod; adapter `cargo test` in a GitHub workflow | coverage artifact in CI | CI | none | merged (#148) |
| E5 | Upstream captures as fixtures | depends on owner keys | manual | owner | blocked |

### Phase 7: inference core unification (Workstream B, second half)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| B4 | One `fuse.rs` with a rounding policy; one `StepCache` trait with `EasyCache`/`TeaCache`/`FbCache`/`Taylor`; one disk prompt cache | byte identity per family | **GPU** byte-identity cells | B1 | todo |
| B5 | One `sdpa(q, k, v, Plan)` entry with `sdpa_rule.rs` as the chooser; host attention twins retired after `no_host_fallback`; `nvfp4_gemm.rs` and the oxide AOT path behind features | `no_host_fallback` green | **GPU** `kernels` tier | B1 | todo |
| B6 | `H3Pipeline::load` and `Ltx2Pipeline::load` split into plan, open, place, fuse; dead code (`TensorBackend`, mlx scaffold) retired | engine lifecycle testable on CPU | pod | B1 | todo |

## 3. Order of execution

1. F1 and A1 start together (no dependency between them; A1 is the
   keystone and ships alone when green).
2. F2 after F1 merges; F3a in parallel with A1.
3. A2 after A1; A3 and A4 after A2; A5 as soon as A1 is on main.
4. B1 after F1; B2 and B3 after B1 and A2 (first **GPU** cells).
5. C1 after A4; C2 after C1; C3 after C2; D1 after C3; D2 after A2 at any
   time; D3 and D4 after D1 and D2.
6. E1 and E4 at any time; E2 after A3; E3 after E2.
7. B4, B5, B6 after B1, interleaved with phases 3-6 as GPU budget allows.

## 4. Execution log

Appended as work runs. Each entry: date, WP, agent or session, where it
ran (pod id, $/hr), result, cost.

- 2026-10-23 21:57Z: PR #138 (review revision, decision log, this plan) and
  `model-registry-rs#2` (registry rescope) merged by the coordinating
  session (docs only, CI green where it runs).
- 2026-10-23 22:12Z: build pod `i8tu0lpdb6nvov` created directly on the
  Runpod REST API (fv-control has no token in this environment): cpu3c,
  16 vCPU, $0.48/hr, EU-RO-1, volume `fv-build` (`pxy4hlsnwq`) at
  `/workspace`, base image pin `bb-d124c1066ca42695`, session token, idle
  stop 30 min, hard cap 6 h + 30 min grace with the in-pod backstop. Ready
  in about 2.5 min; no readiness cycle needed. Balance at start $22.61.
- 2026-10-23 22:37Z: F1 opened #139 (69 files, +311/-2348; lint and
  reference grep green; no Rust change). Two scope extensions accepted by
  the coordinator: the Vast `compare` tier (it only ran the deleted
  `upstream_bench.py`) and `serverless-coldstart.sh` (it only pushed the
  deleted `serverless-worker.sh` into a template). `runpod-matrix.sh` stays
  until D1. Merged 22:55Z after gpucheck-t0 (unit tests, NVRTC gate) and
  the image build passed.
- 2026-10-23 22:52Z: F3a opened #140 (13 code files, +902/-32): 338 crate
  tests and `scripts/serve/check.sh` (986 passed) green on the pod. `otel`
  deferred to a new F3c. Awaiting serve-compat CI.
- 2026-10-24 00:10Z: A1 opened draft PR #143 (`claude/a1-fastvideo-manifest`):
  `crates/fastvideo-manifest` layer 1 (modules `name`, `model_toml`,
  `manifest`, `canonical`, `sig`, `lineage`, `schema`), ADR-0005..0007,
  fixtures from an independent Python oracle (`fixtures/gen_fixtures.py`).
  On build pod `i8tu0lpdb6nvov`: `cargo test -p fastvideo-manifest` 52 passed
  (15 unit, 37 integration incl. 2 proptests x 512 cases), `cargo clippy
  --all-targets -D warnings` and `cargo fmt --check` green for the crate.
  Lockfile adds 17 packages (blake3 1.8.7, schemars 1.2.3, proptest 1.12.0
  and their deps). Open questions in the PR: `seg-dot` and `--`,
  `[defaults]` keys, `verify_lineage` for the model.toml-side checks.
- 2026-10-24 01:05Z: A2 opened draft PR #147 (`claude/a2-family-manifests`):
  `fastvideo-manifest::family` (layer 2: `FamilyManifest`, `VariantManifest`,
  `FamilySet`, `schemas/family.v1.json`, the `fs` feature for `load_dir`),
  `families/{h3,ltx2,wan}.toml` generated from `catalog()` (18 + 5 + 7 = 30
  variants, every catalog row), `cuda::families` (`catalog_from_manifests`,
  the `${weights}` / `${tae}` substitution, the generator), ADR-0008. On
  build pod `i8tu0lpdb6nvov`: `cargo test --locked -p fastvideo-manifest -p
  fastvideo-engine-service` 59 + 136 passed (1 ignored: the generator),
  `tests/families.rs` proves projection == catalog row for row under two
  layouts; `cargo fmt --check`, `clippy --workspace --all-targets -D
  warnings` and `scripts/serve/check.sh` green. No new lockfile package.
  `registry_name` empty everywhere (nothing published); A3 blockers listed
  in the PR.
- Merge rule applied by the coordinator: when the base moved only by
  documentation since the branch point, the PR's own green CI stands;
  otherwise main is merged into the branch and CI re-run on that head.
- 2026-10-24 00:30Z: E1 opened as a draft PR (docs and scripts only, no
  Rust, no pod). Vendored under `docs/serve/specs/`: LTX `openapi.json`;
  21 fal queue OpenAPIs (H3 max/max-turbo/base t2v, i2v, r2v; LTX-2.5
  t2v/i2v fast and pro; LTX-2.5 a2v fast and pro; LTX-2.3 extend and
  retake; LTX-2.3-quality ingredient; LTX-2.3-22b t2v; Wan 2.2-5B t2v, i2v,
  fast-wan) plus the director `asyncapi.json`; MiniMax
  `v2-video-generation.json`; the OpenAI `openapi.yaml` `/videos*` trim
  (openai-openapi `main` 0938b0f0); FastVideo `protocol.py` and
  `api_server.py` verbatim at `main` 8db2d71e plus a DERIVED
  `videos.schema.json`. Unavailable (recorded, re-probed weekly): the
  director's queue OpenAPI (404 `null`) and the director app's own
  OpenAPI (405 without a key). `scripts/serve/specs-drift.sh` green (30
  specs); `.github/workflows/specs-drift.yml` runs it weekly and files
  one issue on drift.
- 2026-10-24 00:18Z: F2 merged (#142): 438 files formatted, about 370
  clippy diagnostics fixed (the Taskfile's "27" was stale), fmt and clippy
  are gates in the Taskfile, both PR workflows and `scripts/serve/check.sh`.
  Two decisions taken by the coordinator and recorded here: workspace
  `rust-version` 1.85 -> 1.88 (`fastvideo-models` and `fastvideo-loader`
  already used 1.87/1.88 APIs, so 1.85 was never true); a crate-wide
  `allow(clippy::needless_range_loop)` in `fastvideo-cudarc` and
  `fastvideo-models` (49 index loops that mirror the Python references).
  Pod: 1807 lib tests, `check.sh` 986 tests, all green.
- 2026-10-24 00:35Z: A1 merged (#143): `crates/fastvideo-manifest` layer 1,
  52 tests, fixtures from an independent Python oracle, ADR-0005..0007.
  Decisions recorded: `verify_chain(&[Manifest])` stays as pinned and the
  checks that need `model.toml` (variant, licence, name, config digest) are
  `verify_lineage(&[LineageEntry])`; the model-id grammar is read literally
  (`--` allowed inside a model id, forbidden in a namespace); `[defaults]`
  keeps WO-001's six keys. Open for the owner before the first publish:
  whether `--` should be forbidden in model ids too, and whether
  `[defaults]` should carry `guidance_2`, `flow_shift`, `negative`, `seed`.
- 2026-10-24 00:39Z: E1 merged (#144): 30 upstream specs vendored under
  `docs/serve/specs/` (LTX, 22 fal endpoints, the director AsyncAPI,
  MiniMax, the trimmed OpenAI `/videos` paths, FastVideo `protocol.py` /
  `api_server.py` at `8db2d71e` plus a derived JSON Schema), drift script
  and weekly workflow; two director documents unavailable without a fal
  key. The oracle pin `FV_REV` in `scripts/gpu/upstream/setup.sh` stays
  (it pins the parity reference, a separate concern).
- 2026-10-24 00:36Z: A2, A5 (registry WO-001 against the merged crate, built
  in the coordinator's container since that workspace is small) and B1
  started. Balance $21.6, pod `i8tu0lpdb6nvov` at 4.4 h uptime.
- 2026-10-24 01:20Z: E4 opened as a draft PR (`claude/e4-adapter-ci`): the
  `adapter-tests` workflow runs `scripts/serve/coverage.sh` (new) in the build
  base image on pull requests and main pushes touching the serve crates:
  `cargo test --locked` of 11 crates (the five adapters, `fastvideo-serve`,
  protocol, trace, engine-service, serve-kit, upsampler) under
  `cargo-llvm-cov` 0.9.1, with `lcov.info`, an HTML report and the totals as
  the `adapter-coverage` artifact and job summary. Before: no workflow ran
  these crates' Rust tests on pull requests (serve-compat runs the client
  libraries against the binary; tools-release runs `check.sh` on main only).
  `task coverage` / `task coverage:tests` added; `coverage.sh` allowlisted
  for new build pods (the running pod's server from main refuses it until
  merge). On pod `i8tu0lpdb6nvov`: `cargo test --locked` of the set, 711
  passed, 0 failed, 10 ignored, 171 s; no Rust change. Not done:
  `cargo-llvm-cov` in `docker/build-base.Dockerfile` (re-pins the image).
- 2026-10-24 01:25Z: E4's `adapter-tests` run on the PR head (PR #148,
  run 40049075337) green on the first attempt, 8 min wall clock (container
  pull 55 s, the instrumented `cargo test` 6.5 min cold, no rust-cache
  hit yet): 69 test binaries, 720 passed, 0 failed; coverage over the instrumented
  workspace files: lines 62.42% (54115 / 86700), regions 62.96%, functions
  59.31%; `adapter-coverage` artifact 4.5 MB (lcov, HTML, summaries).
- 2026-10-24 00:40Z: B1 opened as a draft PR (`claude/b1-runtime-extract`)
  from the build pod `i8tu0lpdb6nvov`: 45 `wan/` modules moved to
  `cudarc/src/rt/` with `wan::` re-export shims, `kernels.cu` split into 14
  region files concatenated at build time (sha256 pinned by two tests). On
  the pod: fmt and clippy `-D warnings` green, `--features cuda` type-check
  green (nvcc embedded cubins for sm 75–120 from the concatenation), 1809
  workspace lib tests passed, 0 failed (cudarc 606, models 544,
  engine-service 75), `scripts/serve/check.sh` green. GPU byte-identity cell
  pending (coordinator, before merge).
- 2026-10-24 01:41Z: B1 GPU byte-identity gate green on one RTX PRO 6000
  (EU-RO-1, volume `bz0ofkfx45`, pod `b4vz9cljstlvbc`, 34 min at $2.49/hr,
  deleted): `techniques` family, baseline `42da257` against the PR image
  `51898b8`; cells fasth3-4step-vsa-768p, h3-480p-fullopt and
  ltx25-512p-sol, each base vs env and base vs prof: 124/124 (H3) and
  121/121 (LTX) frames identical, max abs diff 0; the LTX cache arm
  identical too. fasth3-8step-768p skipped (`h3-8step` is not on the 500 GB
  EU volume, owner decision); no Wan cell (no Wan trees on the EU volume;
  Wan is covered by the kernel-source sha256 pins and the shared runtime).
  Evidence: `artifacts/runpod/techniques/51898b8-10240107/compare/`. Main
  (A2, E4) merged into the branch: no cudarc or gpucheck file changed since
  the baseline, so the GPU evidence stands for the merged head.
- 2026-10-24 01:40Z: A4 opened a draft PR (`claude/a4-control-gen`): the
  family manifests are parsed by `control/gen-configs.mjs` (smol-toml 1.9.1,
  the one new devDependency) and generate `control/src/families-gen.ts`, the
  `EU_TREES` region of `volumes.ts` (from `volumes.sh`), the `weight_trees` /
  `served` columns of `scripts/gcp/families.tsv` and the `variant_trees`
  region of `verify-weights.sh` (new `variant:<id>` cells); `recipe-trees.ts`
  and every model preset's `weights` / `min_vram_gb` derive from them;
  cross-checks tie `catalog.json`, `autoscale.toml` and `volumes.sh` to the
  manifests. Byte-identical except the two H3 `weight_trees` cells of
  `families.tsv` (order). Found: `longlive-1.3b` and the lightx2v v0.1 recipe
  have no variant in the manifests (recorded as exceptions, for A3). In this
  container (node 22, no Rust): `gen-configs.mjs --check`, `npm run
  typecheck`, 43 vitest files (one pre-existing failure fixed: a test grepped
  a `caps.rs` line F2 reformatted), `scripts/gpu/lint.sh` green. New workflow
  `.github/workflows/control.yml` runs them. Deployment data stays hand-written
  for C1 (presets, image variants, `variants.sh`, autoscale pools, the
  component cells, the `images.md` table).
- 2026-10-24 01:37Z: F3c opened a draft PR (`claude/f3c-otel`): the `otel`
  feature of `fastvideo-serve` implemented (`src/otel.rs`, `[otel]` config
  section with `FV_OTEL*` overrides, off by default; OTLP over
  http/protobuf; one trace per job: a `job` span with id / model / api /
  status / queue-wait / inference durations and the recorder's events as
  spans under the request's trace id; `tracing-opentelemetry` layer in
  `init_tracing`). Crates `opentelemetry*` 0.33.1 and
  `tracing-opentelemetry` 0.34.0 (6 new lockfile entries, nothing bumped).
  On build pod `i8tu0lpdb6nvov`: `cargo test -p fastvideo-serve` 83 unit
  tests and `--features otel` 88 (5 `otel::tests`, the config round-trip),
  every integration suite green both ways; `cargo clippy --workspace
  --all-targets -D warnings` and the `otel` clippy green; `cargo fmt
  --check` green; `scripts/serve/check.sh` 987 cargo tests + 45
  release.test + 2 node, exit 0 in 245 s. Not done: a run against a live
  OTLP collector (needs a pod with a collector; recorded as pending in
  docs/serve/tracing.md).
- 2026-10-24 02:10Z: D2 opened a draft PR (`claude/d2-gold-files`):
  `crates/fastvideo-lab` (lib: `gold` types, `validate`, `load_dir`,
  `schemas/gold.v1.json`, the `profiles` comment lint), 30
  `lab/gold/<variant>.toml` (one per family variant id), `lab/README.md`,
  ADR-0009; the measurement lines of 20 `profiles/h3/*.toml` comments
  replaced by pointers (comment-only: every profile parses to the same TOML).
  Every number in a gold file is cited by a `[[source]]` and every absent one
  declared by a `[[gap]]`, both enforced by the parser: 14 of 30 variants
  carry a clip time, 3 a PSNR against upstream, 0 a `[last_validated]`; 10
  variants carry no measurement but the class price (the D3 work list). On
  build pod `i8tu0lpdb6nvov`: `cargo test --locked -p fastvideo-lab` 19
  passed (10 unit, 6 gold-files, 1 profile lint, 2 schema), `cargo fmt --all
  --check` and `cargo clippy --locked --workspace --all-targets -D warnings`
  green. Lockfile: the new crate only. `lab/tiers.toml` and
  `lab/cells/*.toml` left to D1 (wall caps, stages and cells are not in any
  document).
- 2026-10-24 02:25Z: D2 review fixes on #151 (coordinator decisions in the
  decision log, FVID-2026-10-24-gold-tier-gpu-and-settings): quality numbers
  measured on an H100 are gaps with the values in source notes
  (`fasth3-4step-vsa`, `ltx25-ref2v`); numbers measured under a non-default
  setting stay only with the setting named in `workload.note` and the new
  `workload.env` (`FASTVIDEO_H3_VAE_MXFP8=0` on the pre-2026-10-14 H3 runs,
  `FASTVIDEO_WAN_QUANT=mxfp8` on the Wan TAEHV arms; `fasth3-trim-nvfp4`
  un-gapped under the same rule); `quality.cell` on the Trim and V2 TAEH3
  files; `[[source]]` heading and date required, every heading checked
  against its file; the profile lint catches ranges with a unit, `×` / `X`
  / spaced speed-ups, more units and metric names up to three words before
  the value, and lets scale factors and clip lengths pass;
  `fastvideo-lab`, `lab/**`, `profiles/**`, `families/**` in the
  adapter-tests workflow and `scripts/serve/coverage.sh`. Coverage now: p50
  17/30, PSNR vs upstream 1/30, LPIPS 0/30, 11 variants with the price only.
  On `i8tu0lpdb6nvov`: `cargo test --locked -p fastvideo-lab` 28 passed (17
  unit, 8 gold-files, 1 profile lint, 2 schema); the coverage crate set
  (`cargo test --locked` of its 12 crates) 748 passed, 0 failed; fmt, clippy
  `-D warnings` and `scripts/gpu/lint.sh` green.
- 2026-10-24 02:50Z (coordinator): merged since 01:15Z, each on a head
  with main merged in and CI green: A2 (#147, after an independent review),
  E4 (#148), B1 (#146, after the GPU byte-identity gate: 9 compare reports,
  every frame identical), A4 (#149, after review fixes: `MODEL_NAMES` holds
  only names the engine resolves), F3c (#150, after review fixes; the live
  OTLP collector check, an `[otel] headers` key and an `--features otel` CI
  build stay open), D2 (#151, after review fixes: tier-GPU and exact-setting
  rules). Coordinator decisions: gold quality numbers measured on a GPU
  other than the tier's are gaps (values kept in cited notes); a number
  measured under a non-default setting stays only when `workload.note`
  names the setting. A lab test's made-up setting name became
  `FASTVIDEO_TEST_X` in B3a's merge (the settings scan exempts that prefix).
  Main's `adapter-tests` is intermittently red in `fastvideo-fal`
  `tests/director_e2e.rs` (a duplicate SDP `a=mid` in the test offer; a
  playout-depth assertion that depends on machine speed), first run on every
  PR since E4; a fix is being root-caused on `claude/fix-fal-director-flakes`.
  Three agents stopped on a usage-credit error at 01:51Z and were resumed
  from their worktrees. Runpod balance $58.6 after a top-up. Open: A3 (#152,
  merging main and regenerating control), B3a (#153, GPU byte-identity run
  pending), A5 (blocked on the `FASTVIDEO_RS_TOKEN` secret). Next: B2, C1, E2
  after A3; B3b after B3a.
