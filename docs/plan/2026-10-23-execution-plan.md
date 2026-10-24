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
| B2 | `VideoFamily` trait, common `Request`/`Conditioning`, `Request::validate(&ModelCaps)`; the 21 registry tables, `resolve()`, `all_registered_ids()` and the 20 `generate_*` functions replaced by the manifest and a static family list; `AvGenerateOptions` reduced to common fields | CLI behaviour unchanged for every preset (golden `--help` and tiny CPU generations); 83 Hub ids resolve | pod; **GPU** one tiny generation per served family | A2, B1 | merged (#157) |
| B3 | Settings layer: the `FASTVIDEO_*` table generated from one declarative list; the 111 direct reads routed through `settings::var`; a `Settings` snapshot passed to `load`; `model_spec` for Wan and the small families; `docs/scope.md` table generated | zero direct `env::var("FASTVIDEO_` outside `settings`; off-identity invariant tests | pod; **GPU** byte-identity cell | B1 | merged (B3a #153; B3b #162) |

### Phase 3 and 4: deployment (Workstream C)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| C1 | `[deploy]` section of the family manifest; presets, variant tables, `families.tsv`, autoscale pools and the Dockerfile variant stages generated from it | `--check` green; one image with boot-time config if the owner decides so | pod | A4 | merged (#156; single image: owner decision, ADR-0011) |
| C2 | `fv-boot` in `fastvideo-deploy`: driver check, config materialization, weights plan (volume / registry `hydrate` / hub / none), sidecar, tunnel and phase reports, watchdog, exec; fake provider and fake chunk source tests; the ten boot variants deleted | one boot script in the tree; fv-control sends env only | pod; **GPU** one pod boot on the EU volume | C1, registry WO-002 for the `hydrate` source (volume and hub sources first) | merged (#159: fv-boot is the images' entrypoint and every fv-control boot prefers it, ADR-0014; GPU gate green for the image kind on the EU volume (ltx); open per-kind gates before lifting `FV_BOOT_LEGACY` (cluster worker, serverless r2 streaming, edge watchdog, GMI / Brev, upsampler; ADR-0014 §6); remaining: C2b deletes the shell boots and sends env only after the image promotion, the `registry` source after WO-002, the upsampler launcher in Rust) |
| C3 | `fv-deploy` with the `Provider` trait (from `fastvideo-autoscale`): `runpod::{Pod,Serverless}`, `vast`, `gcp`; shared money guards, backstop, ledger, templates, release promote/rollback; fv-control on the shared `Launch` schema | `fv-deploy up` per provider with the same smoke | pod; **GPU** one `fv-deploy up` on Runpod EU | C2 | todo (C3a pr #169: Runpod pods and serverless queue endpoints, the `Provider` trait in `fastvideo-deploy`, the `Launch` schema, the shared guards and backstop, `fv-deploy plan|up|status|logs|smoke|down`, ADR-0018; live gate passed on Runpod EU (h3-turbo smoke); next: C3b Vast + GCP, C3c fv-control on the generated `Launch` types, C3d templates / release / D1 ledger) |

### Phase 5: validation driver and gold standards (Workstream D)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| D1 | `fv-lab` binary: `pod`, `run`, `report`, `gate`/`oracle` moved; `lab/tiers.toml`, `lab/cells/*.toml`; the 24 port one-offs and `runpod-matrix.sh` retired | `fv-lab run rtx6000` reproduces today's `rtx6000` family | pod; **GPU** one `rtx6000` run | C3 (`Provider`) | todo (D1a merged #171: `fv-lab run` + `pod` on fv-deploy's guards, `lab/tiers.toml` and `lab/cells/rtx6000-*.toml`, dry-run and payload goldens from the scripts, ADR-0019; GPU gate passed (fv-lab = script, sol-h3 and ltx25-two-stage frames byte-identical); next: D1b other families, D1c `report`, D1d `gate`/`oracle`, D1e one-offs) |
| D2 | `lab/gold/<variant>.toml` for every served variant from the numbers in `docs/perf`, `docs/oracle.md` and the profile comments; keyed by registry name and `@rev` | every served variant has a gold file; comments in `profiles/` no longer carry numbers | pod | A2 | merged (#151) |
| D3 | `fv-lab report tiers` and the generated `docs/perf/tiers.md` | table generated from one run set | pod; **GPU** the gold runs | D1, D2 | todo |
| D4 | CI bridge: gold files parse and resolve; `last_validated.commit` age check; an on-demand `fv-lab run gold --all` workflow | visible per-model validation age in CI | CI | D3 | todo |

### Phase 6: API conformance (Workstream E)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| E1 | Vendored upstream specs under `docs/serve/specs/` with a drift script; `schemars` schemas for native and director | drift script green | CI | none | merged (#144; schemars schemas deferred to E2) |
| E2 | `ParamSpec` per (surface, model) derived from the manifest caps; adapters validate from it; duplicated constants deleted; LTX body typed | one source for every limit | pod; serve-compat | A3 | merged (#158) |
| E3 | Certification matrix harness over every surface, parameter x surface x model TSV, CI assertion that every parameter has a valid and an invalid case | TSV with no empty cell | pod | E2 | merged (#164) |
| E4 | `cargo-llvm-cov` on the pod; adapter `cargo test` in a GitHub workflow | coverage artifact in CI | CI | none | merged (#148) |
| E5 | Upstream captures as fixtures | depends on owner keys | manual | owner | blocked |

### Phase 7: inference core unification (Workstream B, second half)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| B4 | One `fuse.rs` with a rounding policy; one `StepCache` trait with `EasyCache`/`TeaCache`/`FbCache`/`Taylor`; one disk prompt cache | byte identity per family | **GPU** byte-identity cells | B1 | active (B4a pr #173: step caches, ADR-0020, GPU `b4a` cells pending; then B4b `fuse.rs`, B4c disk prompt cache) |
| B5 | One `sdpa(q, k, v, Plan)` entry with `sdpa_rule.rs` as the chooser; host attention twins retired after `no_host_fallback`; `nvfp4_gemm.rs` and the oxide AOT path behind features | `no_host_fallback` green | **GPU** `kernels` tier | B1 | merged (#161) |
| B6 | `H3Pipeline::load` and `Ltx2Pipeline::load` split into plan, open, place, fuse; dead code (`TensorBackend`, mlx scaffold) retired | engine lifecycle testable on CPU | pod | B1 | merged (#170: plan/open/place/fuse, ADR-0017; `techniques` and `b6` GPU identity cells all byte-identical vs main; TensorBackend and the mlx scaffold retired) |

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
- 2026-10-24 02:26Z: A3 opened a draft PR (`claude/a3-engine-projection`):
  the engine embeds `families/*.toml` (`FV_FAMILIES_DIR` overrides) and
  `catalog()` is their projection; the hand-written rows and recipe-derived
  caps are deleted; `[variants.serving]` and `public_names` added to layer 2;
  `longlive-1.3b` and `h3-lightx2v-turbo-v01` are variants; the 35 serve
  TOMLs and the compat config name `variant` + residency (zero `[aliases]`).
  Golden `/fv/v1/capabilities` + `/v1/models` for 36 configs taken on the
  unchanged tree first: identical for every CUDA config; one listed
  exception (`fake.toml`, `tests/compat/serve.toml` gain `fastwan-1.3b`).
  On build pod `i8tu0lpdb6nvov`, merged head: manifest + engine-service +
  protocol 292 tests, `scripts/serve/check.sh` 999 passed / 0 failed in 106
  binaries (dimension_sweep 8, e2e 14, capabilities_golden 1), fmt, clippy
  `-D warnings`, cuda-feature check green. Control-side follow-up for A4:
  `gen-configs.mjs --check` and `serverless/serves.ts` must read the new
  `[[models]]` form.
- 2026-10-24 02:55Z: A3 merged main (A4 #149, F3c #150) and brought the
  control plane in line: `gen-configs.mjs` regenerated (worker and image
  configs, `families-gen.ts` with `public_names`, `tasks`, `main_tree` and
  the two new variants), `CATALOG_EXCEPTIONS`, `ENGINE_RECIPE_ALIASES`, the
  `recipe-trees.ts` override and the LongLive `weights_extra` removed,
  `serverless/serves.ts` reads `[[models]] variant` and public names. Review
  fixes: e2e scripts write and edit the variant form (with guards), retired
  config forms get named refusals, `[[models]]` resolve against one manifest
  snapshot. Control: `gen-configs.mjs --check`, typecheck, vitest 580/580;
  pod: `check.sh` 1000 passed / 0 failed, serve `--features otel` 143
  passed, manifest + engine + protocol 292, fmt, clippy, cuda check green.
- 2026-10-24 02:55Z: B3a opened as draft PR #153 (`claude/b3-settings`): the
  declarative `FASTVIDEO_*` list (`techniques::settings::SETTINGS`, 222 rows:
  112 layered, 108 env-only, 2 reserved), the 112 direct and 27 indirect
  reads routed through `settings::env_var` / `env_var_os` at the same call
  sites, the runtime's typed getters on the shared `settings::parse`,
  `docs/scope.md`'s table generated, ADR-0010. On build pod
  `i8tu0lpdb6nvov`: fmt and clippy `-D warnings` green, `cargo test
  --workspace --lib` 1847 passed, `-p fastvideo-models -p fastvideo-cudarc
  -p fastvideo-gpucheck` 1211 passed, `--features cuda` type-check green,
  `scripts/serve/check.sh` 995 passed. One pre-existing race seen once and
  not reproduced: `wan/vae.rs`'s chunking test sets `FASTVIDEO_VAE_CHUNK`
  while `hooks.rs`'s Wan end-to-end test decodes. B3b (`load` snapshot,
  `model_spec`s) is a second PR. GPU byte-identity cell pending
  (coordinator).
- 2026-10-24 03:40Z: B3a GPU byte-identity gate green on one RTX PRO 6000
  (EU-RO-1, volume `bz0ofkfx45`, pod `3nbse01c0zav0e`, deleted): the
  `techniques` family, baseline `1ea846f` (main) against the PR image
  `90dd3f8`; fasth3-4step-vsa-768p and h3-480p-fullopt 124/124 frames, and
  ltx25-512p-sol 121/121, identical for base vs env and base vs prof (max
  abs diff 0); the LTX cache arm identical too. fasth3-8step-768p skipped
  (`h3-8step` is not on the 500 GB EU volume); no Wan cell (no Wan trees on
  the volume; Wan's reads are `std::env::var` at the same call sites,
  checked by script in the review). Evidence:
  `artifacts/runpod/techniques/90dd3f8-10240259/compare/`. Review: no
  blocking finding; three documentation-only table rows corrected. Main
  merged in since then changed no inference crate, so the evidence stands.
- 2026-10-24 04:30Z: C1 opened a draft PR (`claude/c1-deploy-section`):
  `deploy/deploy.toml` (ADR-0011) holds the deployment data A4 left
  hand-written, value for value: 8 images (one legacy), 24 model presets, 4
  family presets, 5 GCP rows, the dispatch map, 4 autoscale pools.
  `fastvideo_manifest::deploy` types and validates it against the families
  (variant ids, recipe names, family-preset card class >= the variants'
  floor, prequant nesting, EU-only volume), `schemas/deploy.v1.json` with a
  drift test. `gen-configs.mjs` generates `deploy-gen.ts` (presets.ts reads
  it), the `enums.ts` `VARIANTS` / `POOL_PRESET_IDS` regions, the inline
  worker configs, `variants.sh`, the thin Dockerfile stages, the autoscale
  pools, the `images.md` table and `families.tsv`; `deployChecks` ties the
  cpu stage, `serve-image.yml`'s lists, the GCP cells and the autoscale
  prices to it. Byte identity: fv-control's preset values identical
  (JSON dump before and after), autoscale.toml and the Dockerfile build lines
  identical but markers, every families.tsv row identical; intended: one
  preset order for the three id lists, uniform doc tables. Single image with
  boot-time config: not implemented, analysed in ADR-0011 for the owner.
  Control (this container): `--check`, typecheck, vitest 45 files / 595
  tests, `scripts/gpu/lint.sh`. Build pod `pzocv1keasg3xn`: `cargo test
  --locked -p fastvideo-manifest` green (3 deploy unit + 2 committed-file +
  7 schema tests among them), workspace clippy `-D warnings`, fmt,
  `coverage.sh --tests` 13 crates / 819 passed (fastvideo-manifest joins the
  set); `fastvideo-autoscale` `committed_example_parses` and the GCP e2e
  test green on the generated files.

- 2026-10-24 04:30Z: B2 opened draft PR #157 (`claude/b2-video-family`):
  `fastvideo-core::family` (`VideoFamily`, `Pipeline`, the static `FAMILIES`
  list of 21 families, one module each with its Hub-id table and run),
  `fastvideo-core::request` (`Request`, `Conditioning`,
  `Request::validate(&ModelCaps)`), `AvGenerateOptions` reduced to the
  common fields plus `Wan/Ltx2/H3Options`; `registry.rs` and
  `av_generate.rs` deleted; ADR-0013. Goldens taken first on the unchanged
  tree (CLI `--help`, `list-models` 83 ids, generate/schedule for every id
  and preset, tiny CPU frames, a 459-input resolution table) pass
  unmodified. On build pod `pzocv1keasg3xn`: workspace lib tests 1881
  passed, models + cli + core 595 passed, fmt, clippy `-D warnings`, the
  cuda check and a `--features cuda` check of core and cli green,
  `scripts/serve/check.sh` 1004 passed. GPU cells pending (coordinator):
  CLI A/B frame identity for H3 (`h3-base` + LoRA), LTX-2.5 (`ltx25`), Wan
  (R2 `fastwan21-1.3b`, or `--tiny --device cuda`).
- 2026-10-24 05:00Z: B2 GPU gate (coordinator): CLI A/B frame identity,
  release CUDA `fastvideo` built on build pod `pzocv1keasg3xn` at main
  `6d5cf6a6` (base) and the PR at `9c56b15e` (head), the same command run
  with both binaries on one RTX PRO 6000 pod (EU-RO-1, volume `bz0ofkfx45`,
  runtime image `sha-6d5cf6a` for the CUDA libraries only), sha256 of every
  output file compared. `ltx25` (LTX-2.5 T2V, seed 7): identical, 122
  files; `wan-tiny-cuda`: identical, 2 files (pod `o6u0poa0uwoj38`);
  `h3-4step` (FastH3 4-step dense LoRA on `h3-base`, 5 s, seed 7):
  identical, 125 files (pod `txi9ybyb3uj24k`; two earlier H3 attempts on
  `h6o3z35thydpey` and the first pod gave the LoRA directory instead of its
  file, exit 2 on both arms, a script error). Evidence:
  `artifacts/runpod/b2-cli/6d5cf6a-1024{0441,0453}/`. The review follow-up
  `c898d2c4` changed only LTX's `--guidance-2` routing (same value, not
  passed by the cells) and test-only `Request::validate`; the evidence
  stands for it. Three pods, about 17 GPU-minutes at $2.49/hr, deleted.
- 2026-10-24 05:10Z: E2 opened a draft PR (`claude/e2-paramspec`):
  `fastvideo-protocol::params` (`ParamSpec`, the surface rules: every
  upstream limit once, cited, H3 facts derived from `fastvideo_models`), the
  five adapters and the native API validate with them and build their specs
  (`params.rs`), the LTX bodies typed against `ltx/openapi.json`, `GET
  /fv/v1/params` with a golden for all 36 configs (182 distinct specs).
  Census in docs/serve/params.md: 9 duplicated limits (H3 5-15 s x3, H3
  references x4, LTX matrix / motions / prompt / rates, fal prompt, fast-h3
  x3) now one source; one disagreement resolved (`guidance_scale_2` 0..=20,
  FastVideo's bound); 8 upstream divergences kept and pinned. On build pod
  `pzocv1keasg3xn` (main #153 merged in): `scripts/serve/check.sh` 1031
  passed / 0 failed in 110 binaries (capabilities golden unchanged),
  protocol + engine-service + manifest 308 passed, `tests/compat/run.sh`
  10/10 suites, fmt and clippy `-D warnings` green. Left for E3: director /
  Reactor control-channel specs, the matrix and TSV, `schemars`.
- 2026-10-24 05:45Z (coordinator): merged C1 (#156, 04:43Z), B2 (#157,
  05:08Z) and E2 (#158, 05:41Z), each with main merged in, CI green on the
  head and an independent review; review fixes landed on the branches
  (B2: unused `Conditioning::task` dropped, the frames-wins divergence
  named; E2: `/fv/v1/params` mounted only with the native API,
  `param_scan` over `segments.rs` and `player.rs`, the full
  `guidance_scale_2` message list). Started B3b, B5 and E3; C2 opened
  #159, under review, its GPU boot smoke running (serve image `:ltx`, PR
  `fv-boot` from the build pod, EU volume).
- 2026-10-24 06:30Z: E3 opened a draft PR (`claude/e3-cert-matrix`): the
  certification matrix `crates/fastvideo-serve/tests/cert_matrix.rs`
  (ADR-0016) drives every spec the 36 shipped configs serve at
  `/fv/v1/params`, the director `configure` / `prompt` specs
  (`fastvideo_fal::params::director_specs`) and the Reactor command tables
  (`CommandTable::param_specs`) with a valid and every invalid case through
  the real router on the fake engine (8,065 requests, 9-24 s on the pod);
  `docs/serve/certification.tsv`: 4,630 rows (290 base, 3,863 parameter,
  290 unknown-field, 187 pair), 8 surfaces, 290 distinct specs, 36 engine
  models, no empty cell, drift-checked; CI assertion
  `every_param_has_a_valid_and_an_invalid_case` with 2 reasoned exemptions.
  No 5xx or panic; 1,114 rows carry findings (15 kinds, listed with
  candidate fixes in docs/serve/params.md; no wire change). On build pod
  `pzocv1keasg3xn`: protocol + serve + fal + reactor 389 passed,
  `scripts/serve/check.sh` 1035 passed / 0 failed in 113 binaries,
  `tests/compat/run.sh` 10/10, fmt and clippy `-D warnings` green.

- 2026-10-24 06:10Z: B5 opened as draft PR #161 (`claude/b5-sdpa`, ADR-0015):
  `rt::sdpa::sdpa(q, k, v, &Plan)` is the one dense-attention entry (every
  family; nn's six SDPA functions, the ops wrapper, H3's FP8 chooser, H3
  VAE's kernel parse and FLUX.2's `text_sdpa` copy removed) and
  `rt::sdpa_rule` the one chooser (`dense_route`, `mma_kernel`, the masked
  routes, over a `Probe`). Identity: `sdpa_rule_identity.rs` transcribes the
  merge-base choosers; identical kernel / argument / effectful-probe traces
  over 19 200 fused-kernel picks and 2.8 M entry cases (every setting, device
  fact, production shape, output type and set of declining kernels).
  Retired: `FASTVIDEO_SDPA=host|sparse` (host-only, already refused on a GPU
  run); the sparse routes' host oracles and the composed SDPA stay. Features
  `fastvideo-cudarc/nvfp4-gemm` and `oxide`, off by default (no served config
  sets them), on in `fastvideo-gpucheck/cuda`; the serve binaries no longer
  embed the oxide cubins. On build pod `pzocv1keasg3xn`: fmt, workspace
  clippy `-D warnings` green; `-p fastvideo-cudarc -p fastvideo-models -p
  fastvideo-gpucheck` 1220 passed (base 1211; +11 new, 2 removed with the
  host twin); workspace lib 1891 passed; `scripts/serve/check.sh` 1004
  passed; `--features cuda` clippy of cudarc (features off, `nvfp4-gemm`,
  `oxide`), engine-service and gpucheck: the base's warnings and no new one
  (cudarc 62 = 62). GPU pending (coordinator): `kernels` tier
  (`no_host_fallback` now runs every `Plan` on device operands) and the
  `techniques` byte-identity cells for H3 and LTX.
- 2026-10-24 06:40Z: B5 review follow-ups on #161 (no dispatch change; GPU
  evidence on `1de80f1`): main merged (B2 #157, E2 #158, plan #160);
  `gpucheck-t0` job `serve-cuda-check` type-checks the serve feature set
  (`-p fastvideo-cudarc --features cuda`, `-p fastvideo-engine-service
  --features cuda`; fv-gpucheck's own check pulls `oxide` in);
  `release-artifacts-pod.sh` builds serve without the oxide environment;
  stale `nn::` SDPA pointers in five docs fixed; `device_mma_sdpa_pick`
  private. The `sdpa_rule` tests (the 2.8 M-case enumeration included) take
  2.0 s in a debug build, so all stay in the default run. On build pod
  `pzocv1keasg3xn`: fmt, workspace clippy `-D warnings`, `scripts/gpu/lint.sh`
  green; cudarc + models + gpucheck 1220 passed, 0 failed; both new checks
  green with no warning; `scripts/serve/check.sh` 1033 passed, 0 failed.
- 2026-10-24 06:50Z: B5 GPU gates (coordinator), runtime image `sha-1de80f1`
  (the PR's code; later commits change no dispatch), RTX PRO 6000 in
  EU-RO-1 with volume `bz0ofkfx45`, pods deleted. `kernels` tier (pod
  `kkm6prv7a2q7a2`): every `no_host_fallback` check passes on device (48
  new `sdpa_*_on_device` checks over d 128/64/80, self and cross, f32 and
  bf16: zero host fallbacks, device-resident output); 889 PASS, 9 FAIL. The
  same 9 fail on main's `sha-326c879` (pod `b7qm5l9f2kip2a`, 841 PASS):
  the five FP8 "peaked" accuracy checks (known, docs/techniques.md), the
  three `sdpa_rule` timing margins (cuDNN 7-8 % faster than the rule's
  fwd2 pick on LTX 1080p/720p and the 24k edge) and one Wan norm at 2 ULP;
  none is B5's. `techniques` byte identity vs `326c879` (pod
  `syol3vmb2k5qfd` failed at once without FV_BASELINE_SHA on the pod, a
  coordinator error; rerun on a new pod): fasth3-4step-vsa-768p and
  h3-480p-fullopt 124/124 frames, ltx25-512p-sol 121/121, env and prof
  arms and the LTX cache arm, max abs diff 0; fasth3-8step skipped (not
  on the EU volume). Evidence: `artifacts/runpod/techniques/1de80f1-10240618/`,
  `artifacts/runpod/kernels/{1de80f1-10240616,326c879-10240623}/`.

- 2026-10-24 06:20Z: B3b opened draft PR #162 (`claude/b3b-settings-load`):
  `settings::Snapshot` captured at load start and passed to new
  `load_with_settings` entries of H3, LTX-2 and Wan (old entries capture
  one); load bodies and load-only helpers read it through
  `settings::Source`, the unchanged callees through `Snapshot::scope`;
  the MLX tree's in-load `FASTVIDEO_H3_AFFINE` write via `settings::set_env`.
  `ModelSpec` with its techniques-layer reads; `family_spec` for H3, LTX-2,
  Wan and the 18 small families; profiles still H3 / LTX-2 only (minimal
  spec, ADR-0010 amendment with the per-site classification). On build pod
  (agent `b3b-settings`): fmt, clippy `-D warnings`, models + cudarc +
  gpucheck + core 1264 passed, workspace lib 1903 passed, `--features cuda`
  check green, `scripts/serve/check.sh` 1004 passed (a timing test in
  `fastvideo-media` failed once on the shared pod, green on rerun). GPU
  `techniques` byte-identity family pending (coordinator).
- 2026-10-24 06:55Z: B3b GPU gate (coordinator): `techniques` byte identity,
  runtime image `sha-93696e5` (the PR's code; later commits change only the
  spec scan, the `!Send` guard and docs) against main `sha-9a3c18b` (same
  inference code as the merge base `71b658b`), one RTX PRO 6000 in EU-RO-1
  with volume `bz0ofkfx45` (deleted): fasth3-4step-vsa-768p and
  h3-480p-fullopt 124/124 frames, ltx25-512p-sol 121/121, env and prof
  arms and the LTX cache arm, max abs diff 0; every H3 and LTX cell loads
  through `load_with_settings`; fasth3-8step skipped (not on the EU
  volume). Evidence: `artifacts/runpod/techniques/93696e5-10240625/`.

- 2026-10-24 05:40Z: C2 opened draft PR #159 (`claude/c2-fv-boot`): `fv-boot`
  in `fastvideo-deploy` (feature `boot`), the `ENTRYPOINT` of every serve
  image (`fv-entry.sh` deleted), ADR-0014 with the inventory of the ten boots,
  the env contract and the byte-compatible report protocol (fixture shared
  with a control vitest). Staging: every fv-control shell boot starts with
  `PREFER_FV_BOOT_SH` (fv-boot when the image has it, `FV_BOOT_LEGACY=1`
  rolls back); the shell bodies and "env only" wait for the promotion (C2b).
  On build pod `pzocv1keasg3xn`: `cargo test --locked -p fastvideo-deploy
  --features boot` 61 + 2 (real `r2-fetch.py` against a fake chunk source,
  streaming through the real `fetch-guard`) + 16 passed; workspace clippy
  `-D warnings`, fmt, `coverage.sh --tests` (14 crates, 895 passed),
  `check.sh` (exit 0), `scripts/gpu/lint.sh` green. Control: `--check`,
  typecheck, vitest 46 files / 602 tests. No new lockfile package. GPU gate
  (`scripts/gpu/fv-boot-smoke-pod.sh` on the EU volume) pending.
- 2026-10-24 05:51Z: C2 GPU gate (coordinator): one boot of the promoted
  serve image `fastvideo-rs-serve:ltx` (fv-serve 0.1.37) through the PR's
  release `fv-boot` (built on build pod `pzocv1keasg3xn`), on an RTX PRO
  6000 in EU-RO-1 with volume `bz0ofkfx45`: driver check (595.91) ok,
  volume (9 trees), config `/etc/fv/runpod-ltx.toml` checked with
  `--print-config`, exec fv-serve; ready, `ltx25-distill-sol` loaded; one
  native job (480p 16:9, seed 1) succeeded in 108 s (pod `0qayz3a464wd7g`).
  The first attempt (pod `1b7hdj1l6cwik2`) booted to ready in 122 s, but
  the smoke script read the model id from the wrong key, so no job was
  submitted; fixed in `4d65003f`. Evidence:
  `artifacts/runpod/c2-boot/{bcde06c-10240544,4d65003-10240548}/`. Two
  pods, about 7 GPU-minutes, deleted.
- 2026-10-24 06:30Z: C2 review fixes on #159 (main with B2 #157, E2 #158 and
  #160 merged in): fv-boot gets its own `boot-build` Dockerfile stage, which
  `prebuilt.sh` replaces when a tools release's serve set carries `out/fv-boot`
  (`release-artifacts-pod.sh`, `tools-deps.sh` and the `tools-release.sh`
  gates ship and check it from now on); with today's newest release
  (tools-v0.1.37, checked with a real `prebuilt.sh fetch`) only that stage
  compiles. Also: signals by pid through the shell builtin (no procps in the
  images), fv-serve's config refusal logged verbatim (a crash loop in the
  diagnosis), `FV_BOOT_LEGACY=1` honoured by fv-boot itself, kind precedence
  fixed for GCP / CloudRift workers and role endpoints, a controller worker
  without a config fails, reports end on any HTTP answer, the backstop key
  only in the watchdog's env, provider r2 parity, `--features boot` clippy
  and tests in `check.sh`, the image entrypoint asserted in CI. Recommended
  rollout: fv-control with `FV_BOOT_LEGACY=1`, lifted per kind after the
  open per-kind gates (ADR-0014 §6).
- 2026-10-24 06:35Z: C2 tools-release fix on #159: the hosted serve-cpu
  dry run (run 40069726472) failed in tools-deps' oxide stage, which runs
  only when `release-artifacts-pod.sh` changes (its layer is otherwise
  cached): `ldconfig -p | grep -q libclang` under `pipefail` lost a race
  (grep exits at the match, ldconfig's next write dies of SIGPIPE), although
  build-base's ld cache lists libclang (checked in the image's layers and on
  the build pod); now `grep >/dev/null`. fv-boot reports
  `FV_RELEASE_VERSION` as fv-serve does (the gates check `fv-boot version`;
  a mismatch case added to `test_tools_release.sh`).
- 2026-10-24 06:46Z: C2 real-entrypoint gate (coordinator): the branch's
  serve image `fastvideo-rs-serve:ltx-sha-0975761` (built by serve-image.yml
  dispatched on the branch: sha tags only, no `latest`, no templates),
  started with its own ENTRYPOINT `fv-boot` and no start-command override,
  on an RTX PRO 6000 in EU-RO-1 with volume `bz0ofkfx45` (pod
  `gnpypqun6ueyv1`, deleted): `/ping` after 42 s, `/healthz` ready after 94
  s more with `ltx25-distill-sol` loaded (build `0975761`, variant `ltx`);
  auth on (a throwaway key, hash only in `FV_API_KEYS`): an unauthenticated
  submit got 401; one native job (480p 16:9) succeeded in 100 s. The head
  since then changes `fv-boot --version` (release version), the libclang
  check in `release-artifacts-pod.sh`, docs, and merges main (#163); the
  boot path is unchanged. Evidence:
  `artifacts/runpod/c2-boot/entry-0975761-10240642/`.

- 2026-10-24 09:15Z, coordinator: merged C2 #159 (fv-boot; head e38bf05,
  all 37 checks green on main merged in) and the engine fix #167 (parked
  events: one reserved tail permit per parkable kind, so a lagging
  consumer still gets the latest queued/started/stage/progress event and
  the terminal event; head
  a1fc2d3, 19 checks green). Earlier today, also merged: B3b #162, B5
  #161, E3 #164, and the main-health fixes #163 (kernel_src test embeds
  the generated source), #165 (`FASTVIDEO_FLASH_KERNEL` doc line the
  catalog test reads) and #166 (decode pipe `-threads 1` on the output).
  Started C3a (`fv-deploy`: `Provider` trait, `runpod::{Pod,Serverless}`,
  shared price cap / balance floor / backstop / readiness-replace guards;
  ADR-0018 reserved); B6 (ADR-0017) still in progress. No GPU time in
  this step.

- 2026-10-24 09:30Z: C3a (`claude/c3a-fv-deploy`, draft PR #169): `fv-deploy` with
  the `Provider` trait in `fastvideo-deploy` (`runpod::{Pod, Serverless}`;
  `fastvideo-autoscale` now depends on it), the `Launch` / `Instance` /
  `Offer` / `Account` schemas, a launch from a `deploy/deploy.toml` preset,
  the shared guards (EU-volume region guard, $8 floor, price cap, detached
  backstop, 600 s / 3-cycle readiness) and `plan|up|status|logs|smoke|down`;
  payload goldens from the scripts' own `plan` (ADR-0018). CPU only, on the
  shared build pod (`pzocv1keasg3xn`, no pod created): `cargo test -p
  fastvideo-deploy --all-features` (95 unit + 8 goldens + 2 schema + 4
  binary + the existing suites), autoscale tests, workspace clippy, fmt,
  lint green. No Runpod call, no GPU. Live gate for the coordinator:
  `RUNPOD_API_KEY=… fv-deploy up --family h3 --preset h3-turbo --region eu
  --channel latest --yes --cap-s 1800`.
- 2026-10-24 09:50Z: C3a review fixes on #169 (coordinator's review): the
  backstop is armed (with an armed-file handshake) right after every create,
  before the billed check; deletes retried, a failed delete stops `up`; a
  lost create answer stops and sweeps by the run's name, as does every exit;
  the floor holds over the cap; a string or missing `costPerHr` handled;
  serverless `workers_max > 1` refused (no serverless quote, ADR-0018 §5.3);
  the backstop stands down only on two 404s a minute apart; `down` needs
  `--kind` for a bare unrecorded id; interrupts reach the smoke; upper bounds
  unless `--force-guards`; CI builds `--features deploy`. CPU only on the
  build pod; no Runpod call.

- 2026-10-24 09:46Z, coordinator: C3a live gate on Runpod EU with
  `fv-deploy` built from `23e1500` (the review fixes) on the build pod:
  `fv-deploy up --family h3 --preset h3-turbo --region eu --channel latest
  --yes --cap-s 1800` from the session container. Balance $49.39 (floor
  projection passed), image `serve:h3-turbo-latest` pinned to its digest
  `sha256:54acd68a…` (built on main `8178a7a` from tools-v0.1.38, so fv-boot
  is the entrypoint), one RTX PRO 6000 in EU-RO-1 on `bz0ofkfx45` at
  $2.49/hr (cap $2.50), detached backstop armed (handshake), `/ping` at 49 s,
  ready at 219 s (cycle 1 of 3), `/healthz` ready with `fasth3`, one native
  job (480p 16:9) succeeded in 17.5 s, pod deleted at 09:50:40; the backstop
  saw it gone and exited by itself at 09:51:52. About 4 GPU-minutes,
  ~$0.17. Evidence: `artifacts/runpod/c3a-deploy/23e1500-10240946/`.
  Side finding: main's serve images had not been rebuilt since #152
  (tools-release on main kept queueing behind a `check-test` run on
  `c7a6bb5` that hung for over an hour, before the #166/#167 fixes); the
  coordinator cancelled it, tools-v0.1.38 published from `8178a7a` and
  `serve-image` rebuilt every variant.

- 2026-10-24 09:20Z: B6 opened as a draft PR (`claude/b6-loader-split`,
  ADR-0017): `H3Pipeline::load_with_settings` and
  `Ltx2Pipeline::load_with_settings` are the composition of four typed
  stages (`h3/pipeline/load.rs`, `ltx2/pipeline/load.rs`): the plan (pure
  over the options, the settings snapshot, configs and manifests and a
  `rt::load_stages::DeviceFacts` measured once), open (maps and read-ahead),
  place (the device; adapter merge per linear) and fuse (TeaCache, the merge
  report; LTX-2's LoRAs fuse per request). Same tensors, order, precisions,
  placements, `load/io` lines and ledger bookings; only non-evidence log
  lines and failure timing moved (listed in the ADR). CPU lifecycle:
  plan goldens for 9 H3 and 7 LTX-2 served / `techniques` configurations,
  H3 plan + open on a tiny snapshot, the full LTX-2 plan → open → place →
  fuse on the tiny fixtures (H3 place needs production dims; neither
  family's generate runs on the CPU). Retired: `fastvideo-ops::TensorBackend`
  and its host twin, `CudarcBackend`, the `fastvideo-mlx` crate (and the
  `mlx-rs` lock tree); kept: `fastvideo-ops::fp8`, the CUDA reader of the
  MLX H3 export (`h3/mlx.rs`, chosen by the checkpoint). On build pod
  (agent `b6-loader`): models + cudarc + core + engine-service 1378 passed,
  0 failed; workspace lib 1916 passed (1971 after merging main at
  `8178a7a`); fmt and workspace clippy `-D
  warnings` green; `--features cuda` check of cudarc, engine-service,
  gpucheck and core green (the base's warnings only); `scripts/serve/check.sh`
  1037 passed; `scripts/gpu/lint.sh` green. GPU `techniques` byte-identity
  family pending (coordinator).
- 2026-10-24 10:10Z: B6 review follow-ups on the PR: ADR-0017 corrected
  (h3-480p-fullopt is `sol-h3-rtx`, no adapter merge) and extended (the
  `load/io` wall seconds exclude tree selection; `dit_tree_identity_for`
  resolves the quant mode eagerly; NVFP4, VDN and MLX loads have no EU tree,
  so no GPU cell); new `b6` family in `scripts/gpu/runpod-matrix.sh` (8 cells,
  baseline vs this build, `--off-identity`: sol-h3 from its prequant tree,
  sol-h3 and 4step-vsa merged at load, H3 under a 32 GiB budget, H3 Ref2VA,
  LTX-2.5 IC-LoRA, dev a2v guided, `--offload cpu`; about 40 min on one RTX
  PRO 6000) and `FV_DRY_RUN=1` for the matrix; the H3 plan goldens grew to
  17 cases on a synthetic EU-volume layout (the served prequant trees
  selected, B200, H100 with W8A8, Trim NVFP4, ModelOpt NVFP4, VDN), the
  H3 placement line labelled, `FV_GOLDEN_BLESS=1` writes the goldens, the
  plan resolves the techniques under the `active` it is given. On build pod
  (agent `b6-loader`): models + cudarc + core + engine-service 1379 passed;
  workspace lib 1971 passed; fmt, clippy `-D warnings`, the `--features
  cuda` check, `scripts/serve/check.sh` (1177 passed) and
  `scripts/gpu/lint.sh` green.

- 2026-10-24 10:22-12:10Z, coordinator: B6 GPU gate, runtime image
  `sha-a2ca568` (the PR head) against main `sha-d061c68`, RTX PRO 6000s in
  EU-RO-1 with `bz0ofkfx45`, every comparison `compare-clips
  --off-identity` (max abs diff 0):
  `techniques` (`artifacts/runpod/techniques/a2ca568-10241022/`):
  fasth3-4step-vsa-768p and h3-480p-fullopt base vs env and prof,
  ltx25-512p-sol base vs env and prof, env vs cache: all pass
  (fasth3-8step skipped, not on the EU volume). `b6` family (ADR-0017):
  b6-solh3-tree (loads `transformer_prequant_sol-h3_mxfp8`, 180
  pre-quantized linears), b6-solh3-merge (tree off, Sol-H3 LoRA fused at
  load), b6-h3-streamed (32 GiB budget), b6-ltx25-ref2v (IC-LoRA),
  b6-ltx25-a2v-guided (49 frames), b6-ltx25-offload-cpu (121 frames): all
  pass, evidence `artifacts/runpod/b6/a2ca568-{10241101,10241146,10241147,10241148,10241149}/`;
  b6-4stepvsa-merge and b6-h3-ref2va passed on the first `b6` pod
  (`compare … exit=0` in its driver log) but that pod reached its cap and
  deleted itself before the driver fetched their JSON
  (`artifacts/runpod/b6/a2ca568-10241022/driver.log`), so those two are
  recorded from the log only. The first `b6` pod ran the eight cells in
  series at ~7 min per arm (longer than the PR's estimate); the missing
  cells were rerun one per pod. About 3.7 GPU-hours in all, ~$9.

- 2026-10-24 10:50Z: D1a (`claude/d1a-fv-lab`, draft PR #171): `fv-lab`
  (binary of `fastvideo-lab`, feature `cli`) with `run <tier|cell>
  [--dry-run]` (runpod-matrix.sh's frame, the same run directory) and `pod
  {up,status,fetch,down,reap}` on fv-deploy's `Provider`, guards and
  lifecycle; `lab/tiers.toml` (`rtx6000`) and `lab/cells/rtx6000-{h3,ltx25}.toml`
  (seven arms) with schemas; B6's `FV_DRY_RUN=1` hunks and a new
  `runpod-http.sh plan`; goldens from the scripts (`fv-lab-goldens.sh`): the
  dry run equals runpod-matrix.sh's, the pod body equals runpod-http.sh's but
  the runner line and `env {}`; `fv-lab` in the runtime image; one seam in
  fastvideo-deploy (`runpod::Pod::list`); ADR-0019. CPU only, on the shared
  build pod (`pzocv1keasg3xn`, no pod created): `cargo test -p fastvideo-lab
  --all-features` (33 unit + 25 integration), fastvideo-deploy tests,
  workspace clippy (with and without `cli`), fmt, lint, goldens `--check`
  green. No Runpod call, no GPU. GPU run for the coordinator:
  `RUNPOD_API_KEY=… fv-lab pod up --sha <7> --family rtx6000 --yes`, then
  `fv-lab pod fetch <id> --wait --down`.
- 2026-10-24 11:35Z: D1a review fixes on #171 (coordinator's review): both
  runners write `<arm>/frames.sha256.json` (byte-identical, fetched by both)
  and `scripts/gpu/fv-lab-compare-runs.sh` is the GPU gate; `pod reap` takes
  only this machine's recorded pods (`--account` for full-pattern names);
  run tags `<sha>-<MMDDHHMMSS>-<hex4>` with a live-name check; fv-lab in the
  tools release's `gpucheck` set (`lab-build=` context, `-V` gated); a
  failing nvidia-smi is fatal; success needs `DONE` and `matrix_exit=0`; the
  pod is deleted before errors return; goldens under `env -i`. The first
  image run on 9a9dc55 failed (lab-build and build raced on one cargo
  registry cache mount); lab-build has its own cache ids since 492e28a,
  whose checks are all green (image 12 m 53 s, a branch compile; the
  PR's tools-release assemble gated `fv-lab 0.1.40`). CPU only, build pod;
  no Runpod call. GPU gate: ADR-0019 §8 on `492e28a` or later.

- 2026-10-24 11:44Z, coordinator: D1a GPU gate on `492e28a` (runtime
  image carrying `fv-lab`), cells `sol-h3` and `ltx25-two-stage` of the
  `rtx6000` family, each side on its own RTX PRO 6000 in EU-RO-1 with
  `bz0ofkfx45`: `fv-lab pod up --sha 492e28a --family rtx6000 --cells
  sol-h3 ltx25-two-stage --yes --wait` (ready 32 s, fv-deploy backstop,
  pod deleted by `--wait`) against `runpod-http.sh run 492e28a` with
  `FV_CELLS="sol-h3 ltx25-two-stage"`. `scripts/gpu/fv-lab-compare-runs.sh`:
  2 arms the same, 0 differences; `sol-h3` 248 and `ltx25-two-stage` 242
  frames byte-identical (`frames.sha256.json` equal), both
  `matrix_exit=0`. About 15 GPU-minutes per side, ~$1.25 in all.
  Evidence: `artifacts/runpod/rtx6000/492e28a-1024114432-9d08/` (fv-lab,
  with `compare-vs-script.txt`) and `artifacts/runpod/rtx6000/492e28a-10241144/`
  (script). The other `rtx6000` arms (fasth3-4step-vsa, fasth3-4step-dense,
  sol-h3-spark; fasth3-8step is skipped on the EU volume) were not run on
  GPU here; their command lines are covered by the dry-run golden.

- 2026-10-24 12:58Z, coordinator: merged C3a #169 (10:05Z), B6 #170
  (12:31Z) and D1a #171 (12:58Z), each with main merged in and CI green
  on that head. Started B4a (step caches, ADR-0020). The build pod
  `pzocv1keasg3xn` stopped idle at 11:30Z and could not restart (no free
  vCPU on its host); replaced by `l3d75u8a8zhw47` (same image, env, start
  command and `fv-build` volume, cpu3c 16 vCPU, EU-RO-1) and the stopped
  pod deleted. Main's images compile `fv-lab` (`lab-build`) until the
  tools release from `20cb60c` publishes; then it comes prebuilt.
- 2026-10-24 13:25Z: B4a opened draft PR #173 (`claude/b4a-stepcache`):
  `fastvideo_models::techniques::step_cache` (ADR-0020), one `StepCache`
  trait and one `EasyCache` / `TeaCache` / `FbCache` / `Taylor`; the nine
  per-family controllers (Wan / SANA / A14B / LingBot EasyCache, H3 / Wan
  Sol / Cosmos(3) / TeaCache4Wan2.1 TeaCache, LTX FBCache, Wan Taylor) are
  parameter sets, their arithmetic differences named parameters. CPU
  evidence: `step_cache/equality.rs` against the verbatim pre-B4a
  controllers (`legacy.rs`), decisions and latents bit for bit. On build
  pod `l3d75u8a8zhw47` (agent `b4a-stepcache`): fmt, workspace clippy
  `-D warnings`, `--features cuda` check, `scripts/gpu/lint.sh`,
  `scripts/serve/check.sh` green; cudarc + models + engine-service + core
  1396 passed, 1 failed: `settings::workspace_check` on fastvideo-lab's
  `FASTVIDEO_X` test names, failing identically on main `3da5cab4` (D1a).
  GPU gate pending (coordinator): `runpod-matrix.sh b4a`, three H3
  TeaCache cells (fullopt env / profile, a 4-step reuse fixture),
  `FV_B4A_CELLS` for one cell per pod; Wan / SANA / LingBot / Cosmos caches
  (no EU-volume tree) and LTX-2.5 FBCache (no reachable path) rest on the
  CPU fixtures.
- 2026-10-24 14:10Z: B4a review fixes on #173: the unreachable LTX FBCache
  cell dropped (ADR-0020 and decision log corrected); b4a cells fail on
  zero reuses or differing cache evidence; exact threshold-tie fixtures per
  comparison family and the TeaCache4Wan2.1 f32 accumulator pinned (five
  comparison / precision mutations now caught); `dit_cfg_easy` without
  `expect`; `Forced` enum, `WrongRefresh`; docs.
