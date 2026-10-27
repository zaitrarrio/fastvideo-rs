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
- **Owner decisions still open** (review §7.10): Vast serving, GMI/Brev in
  Rust or TS. Decided: one image with boot-time config (2026-10-25); Runpod
  LB mode kept (2026-10-25, F3b reduced to the renames). Work packages that depend on one
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
| F3b | ~~Delete `multiworker.rs` and `server.workers_max`~~ (owner 2026-10-25: keep the Runpod LB endpoints, so both stay); rename `[gateway]` to `[cluster]` (`[gateway]` read as a deprecated alias; the shipped configs keep it until every image reads `[cluster]`, ADR-0011 "Renaming a worker-config section"); `gw_workers` kept (fv-autoscale's pod pools read it), the dead `gw_dispatch` / `gw_sessions` no longer created | config round-trip tests | pod | owner decision (Runpod LB mode: keep) | merged (#192: `[cluster]` read with `[gateway]` as its deprecated name, stage 1 of ADR-0011 "Renaming a worker-config section"; nothing writes `[cluster]` yet; LB endpoints kept; stage 2 once every image is past #192) |

### Phase 1: the manifest (Workstream A)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| A1 | `crates/fastvideo-manifest` layer 1: modules `name`, `model_toml`, `manifest`, `lineage`, `sig`, `schema`; the registry's WO-001 contracts verbatim plus the two widenings (`[variant]`, `[runtime]` recipe/profile, `[caps]`); fixtures under `crates/fastvideo-manifest/fixtures/` (valid and ≥25 invalid names with error kinds, model.toml valid/invalid, manifests with `.digest` siblings, test keys, lineage chains); proptest round-trip; `schemas/*.json` export with a drift test; ADR-0005 naming, ADR-0006 model.toml, ADR-0007 manifest/digest/lineage/signatures | all WO-001 acceptance tests green on CPU; no I/O, no CUDA, `unsafe` forbidden | `cargo test -p fastvideo-manifest` on the pod; clippy -D warnings | none | merged (#143) |
| A2 | Layer 2 types (`family`, `variant`, `tiers`) and the first three family files `families/{h3,ltx2,wan}.toml` authored from `engine-service/src/cuda/caps.rs`; an equality test proving `catalog()` equals the manifest projection row for row | equality test green for every catalog row | pod | A1 | merged (#147) |
| A3 | Engine projection: `model_from_config` resolves through the manifest; `validate.rs` and `protocol::ModelCaps` ranges come from it; `[aliases]`, `served_names`, `[protocols]` derived; the 35 serve TOMLs reduced to variant ids and residency; the fake backend serves the manifest; `catalog()` deleted | zero hand-written catalog rows; zero `[aliases]` blocks; `dimension_sweep.rs` and `tests/e2e.rs` green; `/fv/v1/capabilities` unchanged for every config (golden) | pod; serve-compat CI | A2 | merged (#152) |
| A4 | Control-plane and script generation from the families: `control/gen-configs.mjs` emits presets, `enums.ts` ids, `recipe-trees.ts`, `r2-trees.ts`, `weights-sources.ts`, `volumes.ts`; `variants.sh` tables, `volumes.sh`, `gcp/families.tsv`, `autoscale.toml` pools, `verify-weights.sh` cells and `docs/serve/images.md` table generated with `--check` in CI | `gen-configs.mjs --check` green; control unit tests green | pod (node + vitest) | A2 | merged (#149) |
| A5 | Registry hand-off: publish layer 1 (git dependency at a pinned rev); the registry's WO-001 scaffolds against it | registry WO-001 green | registry repo | A1 | merged (model-registry-rs #3, e3d0d333: WO-001 scaffold against `fastvideo-manifest` at a pinned rev; CI green with `FASTVIDEO_RS_TOKEN`; #1 closed) |

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
| C3 | `fv-deploy` with the `Provider` trait (from `fastvideo-autoscale`): `runpod::{Pod,Serverless}`, `vast`, `gcp`; shared money guards, backstop, ledger, templates, release promote/rollback; fv-control on the shared `Launch` schema | `fv-deploy up` per provider with the same smoke | pod; **GPU** one `fv-deploy up` on Runpod EU | C2 | todo (C3a pr #169: Runpod pods and serverless queue endpoints, the `Provider` trait in `fastvideo-deploy`, the `Launch` schema, the shared guards and backstop, `fv-deploy plan|up|status|logs|smoke|down`, ADR-0018; live gate passed on Runpod EU (h3-turbo smoke); C3c: fv-control on the generated `Launch` types: `control/src/launch-gen.ts` (types + zod from the committed schemas, `gen-configs.mjs --check`), every Runpod GPU pod and GPU serverless endpoint built as a `Launch` with bodies by the Rust builders' rules, payloads unchanged, fv-deploy goldens field by field with the intended differences, ADR-0018 §7; CPU / GMI / Brev outside Launch v1 until the schema gains them (after C3b); C3b merged (#193): Vast (fake-engine smoke) and GCP (`[[gcp]]` rows) on the same trait and guards, payload goldens from `vast.sh` / `vm.sh`, `vm.sh`'s E2BIG fixed, the autoscaler's Runpod requests / answers / template pod shared with fv-deploy (ADR-0018 §8, §9; the full move listed there); `Launch` schema change additive (two `ProviderKind` values, `launch-gen.ts` regenerated); Vast and GCP live gates the owner's; C3d pr #199 (ADR-0018 §10): `fv-deploy templates sync | sync-all | list | id | plan`, `release promote | rollback | resolve`, `reconcile` and the D1 `deployments` ledger written by `up` / `down` (the scripts' statements and columns), request goldens from `release.sh` / `runpod-templates.sh` / `registry.sh` against `mock_api.py` (28 scenarios with the order of their effects and two partial failures, `fixtures/deploy/release.json`; review fixes §10.7), the guards' `null` = "not checked by this launcher" (`Launch` schema additive, `launch-gen.ts` regenerated, fv-control's serverless guards `null`); scripts kept until the owner's live gates (§10.6); remaining: `release.sh list / history / deployed / record-build`, fv-control writing its own launches into `deployments`) |
| C4 | One image with boot-time config (owner decision 2026-10-25, ADR-0011): one CUDA serve image plus the CPU image instead of seven variant images; every launch path sends the worker config at boot; an image started without one refuses to boot; `variant` fields, `VARIANTS`, the thin Dockerfile stages, `image-configs.ts` and `fv_variant_for_config` removed, stored specs read through a mapping | gates green; no live resource changed; the owner's live steps written (`docs/serve/images.md` "One image: the live steps") | pod; CI image build on main; live steps by the owner | C1, C2, C3a | merged (#186, c3ceb96c; no live resource changed; the owner's ordered live steps are in `docs/serve/images.md` "One image: the live steps") |

### Phase 5: validation driver and gold standards (Workstream D)

| WP | Scope | Exit criterion | Validation | Depends | Status |
|---|---|---|---|---|---|
| D1 | `fv-lab` binary: `pod`, `run`, `report`, `gate`/`oracle` moved; `lab/tiers.toml`, `lab/cells/*.toml`; the 24 port one-offs and `runpod-matrix.sh` retired | `fv-lab run rtx6000` reproduces today's `rtx6000` family | pod; **GPU** one `rtx6000` run | C3 (`Provider`) | active (D1a merged #171: `fv-lab run` + `pod` on fv-deploy's guards, `lab/tiers.toml` and `lab/cells/rtx6000-*.toml`, dry-run and payload goldens from the scripts, ADR-0019; GPU gate passed (fv-lab = script, sol-h3 and ltx25-two-stage frames byte-identical); D1b merged #176: `techniques`, `b6`, `rtx5090`, `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae`, `precision` as tiers (cell schema: baseline arms, TAE arms, bare `env`, paired gates), a golden per tier, the family inventory in ADR-0019 §9; GPU fv-lab = script for all nine moved tiers (`techniques`, `b6`, `precision`, `rtx5090` 2026-10-24; `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae` 2026-10-25, #184; every arm that ran byte-identical); the ten tiers retired from the script in #185 (ADR-0019 §10: branches deleted, `runpod-http.sh run`/`plan` call `fv-lab pod up`, goldens frozen, `scripts/gpu/fv-lab-retired.sh`); D1d first slice merged #188 (ADR-0019 §11): a typed `[verdict]` step in the cell schema, `b4a` as a tier, `fv-lab gate <run dir>` (gates and verdicts re-evaluated offline); `b4a` GPU gate passed on `e82b6b7` 2026-10-25 (fv-lab = script: frames byte-identical, 5 verdict files the same, both `fv-lab gate --check` clean) and its script branch retired (merged #195 (§11.6: `fv-lab-retired.sh`, goldens frozen, `FV_B4A_CELLS` refused, `tests/verdict.rs` on a frozen copy of the branch, host-env pair selectors refused); two dropped modes restored as tiers `precision-nvfp4` and `ltxvae-gen` (owner decision 2026-10-25, merged #191, ADR-0019 §12: goldens from the script at `5987e6a4`, frozen; GPU check passed 2026-10-25, fv-lab at `8b18d89` vs the script at `5987e6a4`, both tiers' 4k5s arms byte-identical, evidence in #195); D1c (ADR-0019 §13, draft PR): `fv-lab report summary` (the retired families' `live.log` lines, `hd_report.py` less the keyframe sharpness, the four experiment summaries) and `report table` (`sol_bench_table.py`, `timings.sh`), 52 goldens from the scripts on committed run directories, `pod fetch --fetch-tree` (the `FV_FETCH_TREE` mirror, `runpod-http.sh` maps it), `sol_bench_table.py` and the `*-summary.py` deleted, `hd_report.py` / `timings.sh` kept until their callers move; no GPU; D1e (ADR-0019 §14, draft PR): the ten port one-offs decided, `h3-graphs` (its h3-turbo cells, the open item of `docs/perf/h3-denoise-graphs.md` §5) a tier whose commands are cut from its own pod script (`fv-lab-experiment-goldens.sh`, `tests/experiment_tiers.rs`), its **GPU** check passed on `f93a961` 2026-10-26 (sync = lazy = graphs byte-identical, h3-turbo 768p denoise sync 17.98 s, lazy 17.88 s (−0.56 %), graphs 17.93 s (−0.30 %), replays 0.2 % under lazy, graph pool 10.34 GiB; recorded in `docs/perf/h3-denoise-graphs.md` §5) and its two scripts deleted with `scripts/gpu/experiments/` (PR `claude/d1-retire-det-oracle-graphs`), the other nine deleted, `runpod-http.sh`'s remaining callers listed (§14.6); D1d second slice (ADR-0019 §15, draft PR): `det-short`, `determinism` and the oracle's nine tensor-diff targets as tiers (15 / 9 / 32 arms) with goldens from the live branches, cell schema: ungated arms, `fresh` / `requires` / `echo`, `[reference]` (background fetch from `FV_ORACLE_URL`, `ORACLE_REFS_DONE`), `[[prompt_file]]`, `[repeats]` (the scripts' hashes and `determinism.txt` byte for byte, a typed verdict `fv-lab gate` re-decides offline); GPU checks passed on `f93a961` 2026-10-26 for all three (fv-lab = script: `det-short` h3 group and `determinism` h3 byte-identical with the timed controls reported, `oracle` `fasth3-4step-vsa` frames and `compare-dumps` lines the same, every `fv-lab gate --check` clean); `det-short` and `determinism` retired from the script in PR `claude/d1-retire-det-oracle-graphs` (ADR-0019 §15.10: `fv-lab-retired.sh`, branches deleted, goldens frozen, `FV_DET_*` refused, `tests/repeats_oracle.rs` on a frozen copy at `0f63c56b`); `oracle`'s branch kept whole (§15.10: the clip-metric targets, `sfwan13` and `oracle.sh`'s runtime side, which needs `FV_POD_FILE` / `FV_CREATE_WAIT_S`, still run there); D1f (ADR-0019 §17, merged #212): `fuse` (10 arms in 3 cells), `attn3` (6, one cell over H3 and LTX-2.5) and `ltx` (3) as tiers with goldens from the live branches, no new cell field (`fuse`'s kernel check is an ungated arm; `model` is descriptive and names both pipelines), the script's `ltx` arms gated (a real-run change in lockstep: its LTX-2.3 / LTX-2.0 arms skip on the volume instead of failing), `FV_ATTN3_*` refused by fv-lab; GPU checks passed on `762ce4d` 2026-10-26 for all three (fv-lab = script, six pods in EU-RO-1, about $2.2: `fuse` `kernels-fuse` exit 0 with 90 `[PASS]`, the 4-step VSA off / on pair 248 frames each byte-identical and off-identity `ok`; `attn3` the VSA bf16 pair byte-identical, off-identity `ok`; `ltx` `ltx25-two-stage` 242 frames byte-identical, the LTX-2.3 / LTX-2.0 arms skipped on both) and `fuse`, `attn3`, `ltx` retired from the script in #214, merged (ADR-0019 §17.5 "Results", §17.6: `fv-lab-retired.sh`, branches deleted, goldens frozen, `FV_ATTN3_*` into `FV_LAB_NOT_READ`; `runpod.sh gpu` / `matrix` run `ltx` with the image's fv-lab, `runpod-plan.test.sh`); `b4b` / `b4c` / `b4c2` not moved (each needs a new kind of verdict: `benchmark.json` field rules, cache listings, CUPTI kernel tables, §17.3); D1g (ADR-0019 §18, coordinator decision 2026-10-26, draft PR `claude/d1g-retire-finished`): eleven families retired from the script with no tier, `b4b` / `b4c` / `b4c2` (§17.3 option (b), finished experiments), `h3` (§17.4, no `sol-h3-spark-no-refiner` arm), `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`, `identity` and `h3attn` (§14.7; no `h3attn` attention arm is open): `FV_SCRIPT_RETIRED` in `fv-lab-retired.sh`, refused (exit 2, with a pointer) by `runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan` and `runpod.sh gpu` / `matrix` / `b200`, branches deleted (the 16 remaining families' dry runs byte-identical), `b4-verdict.sh` and its test deleted, `runpod.sh` lists `ltx` / `ltx wan`; D1h (ADR-0019 §19, draft PR `claude/d1h-oracle-driver`): `fv-lab oracle`, the driver of §15.6: the upstream reference pods of `lab/upstream.toml` (two stacks, schema `upstream.v1.json`) on fv-deploy's guards (the 192 GB floor for the FastVideo stack), the wait for `ORACLE_DONE` with the keepalive, the `oracle` tier's runtime pod (`pod up --wait`) with `FV_ORACLE_URL`, the upstream pods fetched and deleted on `ORACLE_REFS_DONE`, on a stock wait or on any failure, a combined balance guard, `--url-file` / `--hold-upstream` for a second reader; payload goldens from `runpod-http.sh plan-upstream` (new), 21 driver tests on fakes and 7 on its live world (fake provider, HTTP and executable); CPU only, its **GPU** check (§19.7, about $3.60) the coordinator's; not in it: the clip-metric targets and `sfwan13` as cells, which need a derived-clip step the ADR has not designed (§19.9, a decision); D1i (ADR-0019 §20, coordinator decisions on §19.9, draft PR `claude/d1i-oracle-metrics`): a typed `[[metric]]` step in the cell (a pass over clips: `probe`, `clip-vs-clip`, `kept-frames-vs-source`, `frame-vs-image`; the script's ffmpeg / ffprobe argv word for word, listed by the dry run), `metrics.txt` a report, the ten LTX clip-metric targets as the `oracle` tier's `extra_cells` (run only when `--cells` names their arms: the default run and its three fixtures unchanged), their `sol-ltx25` stack targets, goldens `oracle-extra-{dry-run.txt,gates.tsv}` from the live branch (whose dry run now logs the metric commands), both runners on fakes equal, `fv-lab oracle --fetch-tree`, `fv-lab-compare-runs.sh` compares `metrics.txt`; CPU only, its **GPU** check (§20.7: `ltx25-a2v-i2v` and `ltx25-retake`, about $3.60-$4.20, the guard needs $17.17 with the build pod stopped, the worst case of all three pods $12.50, start at $20.50; after a balance top-up and the merge, with §19.7) the coordinator's; `sfwan13` a follow-up (§20.5); §19.7's GPU check passed on `664473b` 2026-10-26 and D1h is accepted (coordinator, 2026-10-26) (served `fasth3-4step-vsa`, runtime exit 0, compare exit 0: 5 arms the same, 2 with 124 frames byte-identical, 3 with the same `compare-dumps` lines; about $2.63; ADR-0019 §19.7 "Results", PR `claude/oracle-checks-664473b`); §20.7's first attempt on `664473b` compared nothing (the `sol-ltx25` pod had no Hub token for the gated `Lightricks/LTX-2.5`, 401, both cells without a dump; the driver counted `ORACLE_DONE` as served), so in the same PR a reference is served only with its `oracle-dump.tar` (missing at once with the cell's exit otherwise, no runtime pod when nothing is served) and the `sol-ltx25` pod gets `HF_TOKEN` as the Runpod secret reference `{{ RUNPOD_SECRET_fv_hf_token }}` (`hub_token`, ADR-0019 §20.7 "First attempt"); next: the owner creates the Runpod secret `fv_hf_token` (docs/ops/runpod-secrets.md), §20.7 re-run, then `oracle.sh` on the driver (§20.8), `sfwan13` as `oracle-r2` (§20.5; done in D1j, ADR-0019 §21, merged #219: the `oracle-r2` tier and its cell, the upstream `r2:<tree>` step with the R2 key by secret reference, the `decode` measure / directory compares / `after` anchors, the script's `sfwan13` dry-run fixes, and `runpod-http.sh run` `FV_R2_TREES` for the script side, §21.8 decided A; its **GPU** check §21.7 passed on `4b4980b` 2026-10-26 on every criterion and §21 is accepted (served `sfwan13`, runtime exit 0, compare exit 0: 10 arms the same, 4 with 81 frames byte-identical, 6 with the same `compare-dumps` lines, 3 `compare-clips` reports `pass` on both sides; 28.93 GB from R2 on each of the three pods, no key or presigned URL in any fetched file; about $0.86; ADR-0019 §21.7 "Results", PR `claude/oracle-r2-check-4b4980b`, which also sends `r2-stage.sh`'s apt output to `r2-apt.log`)), then `oracle` retires (ADR-0019 §21.9; **not before every cell family it covers has passed its GPU check**: the tensor-diff targets (checked on `fasth3-4step-vsa`) and `sfwan13` have, the clip-metric targets wait for §20.7's re-run, blocked on the owner creating the Runpod secret `fv_hf_token`), `runpod-http.sh upstream` (§15.6), the remaining script families; follow-ups: (1) the runtime image job checks with the previous tools release's fv-lab, so a tier-schema change fails main's first image run until its tools release exists (**done** in #217: a stale release's fv-lab is not used, `prebuilt.sh contexts --stale`, so `lab-build` compiles the commit's, ADR-0019 §20.6); (2) det-short's first pods died during `h3-r3` on 2026-10-26 (host failure suspected; fv-lab `t37wlzbywqp3re`, script `0uwxdgiddf6fn2`), retried; (3) the gold cells' technique bug was fixed by #203; (4) the gold `ref2v` / `a2v` arms load the model for 4-5 min before `negotiate` refuses t2v: negotiate before load (at `7e580e5`: H3 199 s and 132 s, LTX 5 s and 34 s, before the refusal; D3 follow-up (b); **done** in #213: `EngineService::start_negotiated`, `fv-gpucheck engine` negotiates against the table `start` serves before the model loads); (5) the gold run at `e9b9435` skipped the four NVFP4 arms as `MISSING shard` (their R2 trees leave out vae shard 3, byte-identical to h3-base's; fixed in `verify-weights.sh` `shared_file`, #208; at `7e580e5` all four ran, 2026-10-26): a `gen-configs.mjs` cross-check that every variant reading a tree with a `shared_file` entry has `text_weights` = the tree that holds the copy (h3-base) (**done** in #213: `sharedFileChecks` in `crossChecks`, reading the script's table); (6) the engine's `completed_vae_files` (crates/fastvideo-cudarc/src/h3/pipeline.rs) fills ANY index shard missing from a tree's `vae/` from h3-base's, not only the shared ones: limit it to the same list (**done** in #213: `SHARED_FILES`, held to `shared_file` by a test, the copy's length checked)); the remaining fifteen script families (all but `oracle`): ADR-0019 §22, accepted (coordinator, 2026-10-26, FVID-2026-10-26-d1-remaining): eleven retired with no tier (`sfwan` with `oracle`), `hd` and `serve-engine` tiers now, `wan`, `sfstream` and a small `mmaudio` tier after `{tae}` without a TAE gate; slices S1-S8, GPU checks about $10.2 (about $12.8 with `wan`'s oracle cells); S1 (draft PR #222, ADR-0019 §22.8): `sfquality`, `headline`, `speechtest`, `cold`, `eval`, `ltxoffload`, `solbench`, `sol-lingbot`, `sol-cosmos3` in `FV_SCRIPT_RETIRED` with pointers naming what is lost, refused (exit 2) by `runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan` and `runpod.sh gpu` / `matrix` / `plan`, their branches deleted (495 lines, and `eval`'s defaults), the 12 remaining dry runs byte-identical, the stale docs pointed at the retirement; seven families left (`mmaudio`, `sfwan`, `sfstream`, `wan`, `oracle`, `hd`, `serve-engine`); CPU only; S2 (draft PR `claude/d1-s2-hd`, ADR-0019 §22.9): `hd` and its 1080P-gate sibling `hd5` (`prompts-hd5.json` in a cell) as tiers with goldens from the live branch, the derived step (Lanczos compares, post script, keyframe prune) dropped from the script in lockstep, `FV_HD_POST_*` refused; its **GPU** check (§22.10: `turbo-768p turbo-1080p max-1080p`, ~$2.00) passed on `d53ea62` 2026-10-27 (fv-lab = script, two pods in EU-RO-1, both `matrix_exit=0`, compare exit 0: 3 arms the same, 372 frames each byte-identical; about $1.20 with an aborted first attempt; ADR-0019 §22.10 "Result", PR `claude/hd-check-d53ea62`), so S2 is done; next for `hd`: S7 (its branch deleted, `hd` and `hd5` frozen); S3 (draft PR `claude/d1-s3-serve-engine`, ADR-0019 §22.11): `serve-engine` as a tier (three cells, the branch's six arms, `volume+r2` for `fastwan21-1.3b`, 150 GB) with goldens from the live branch, `--tae-dir "$TAE"` dropped from the script's three engine arms in lockstep (`fv-gpucheck engine` then takes `<weights root>/auxiliary/tae`; none of the three recipes reads a tiny decoder, so the frames do not change), `FV_ENGINE_CELLS` refused by fv-lab (select by arm), the other families' dry runs byte-identical; its **GPU** check (§22.12: all six arms, the script side with `FV_R2_TREES=fastwan21-1.3b`, both sides a tree fetch without PNGs, ~$1.35-$2.25) the coordinator's; S4 (draft PR `claude/d1-s4-sfstream`, ADR-0019 §22.13): §22.4's Addition 16 (an arm may read `{tae}` without a `tae` file: not TAE-gated, gated on its trees only, allowed only in a `tae = true` tier; cell v1 unchanged) and `sfstream` as a tier (one cell, the branch's two arms, `volume+r2` for `sfwan21-1.3b`, 150 GB) with goldens from the live branch, the script's `export FASTVIDEO_TAE_DIR` moved into each arm's env in lockstep (and its taew2_1 fetch skipped in a dry run), `FV_SFSTREAM_RUNS` / `FV_SF_PROMPT` / `FV_SF_SWITCH` refused by fv-lab, the other families' dry runs byte-identical; its **GPU** check (§22.14: `sfstream-main`, the script side with `FV_R2_TREES=sfwan21-1.3b`, both sides a tree fetch without PNGs; the compare exits 1 by construction, no arm writes frames, so the runners' identity is the arm's own records, `wan.json`'s checks, run notes and `parity/absolute` note and the window sheets, equal across the pods, per the coordinator's decision of 2026-10-27) the coordinator's |
| D2 | `lab/gold/<variant>.toml` for every served variant from the numbers in `docs/perf`, `docs/oracle.md` and the profile comments; keyed by registry name and `@rev` | every served variant has a gold file; comments in `profiles/` no longer carry numbers | pod | A2 | merged (#151) |
| D3 | `fv-lab report tiers` and the generated `docs/perf/tiers.md` | table generated from one run set | pod; **GPU** the gold runs | D1, D2 | done 2026-10-26: the full gold run at `7e580e5` (ADR-0019 §16.7 "Result", PR `claude/gold-run-7e580e5`), split in two pods (`artifacts/runpod/gold/7e580e5-1026090916-e804`, the 21 R2 arms, all exit 0; `7e580e5-1026091001-7421`, the 11 volume arms, 7 exit 0 and the 4 expected t2v refusals of ADR-0023 §3), about 68 + 22 min, about $3.74; `docs/perf/tiers.md` rendered from both (28 measured, 4 failed); `[last_validated]` written for the 28 measured variants (2026-10-26, `7e580e5`, the arm's `summary.json`: ran to exit 0 on the gold cell, not every number re-confirmed), the arm's `engine.json` their `[[source]]`; follow-ups: (a) `wan14b-plug-4step` peaked at 156,004 MiB of host process memory (152.3 GiB), about twice the ~80 GB of ADR-0019 §16.10, and the 192 GB floor is 183,105 MiB (Runpod's GB are 10^9 bytes), so a pod given exactly the floor leaves it about 26.5 GiB by a once-a-second sample, with pages not yet written back charged to the same cgroup: the floor may not hold. **Prerequisite of the next gold run:** raise the gold tier's `min_ram_gb` (to 256, say), or cut the Wan plug loader's host copies (it holds an f32 DiT copy, the LoRA merge and the f32 UMT5 at once); **done** in #213: `min_ram_gb = 256` (ADR-0019 §16.10 "Raised to 256": the peak is `RawTensor` holding each f32 tensor twice); still open, two future PRs, each with a GPU byte-identity check: (a1) store an f32 `RawTensor` once (about 76 GiB at the peak; the shared loader), (a2) upload an eager Wan DiT map before the text encoder and drop it (about 107 GiB; device allocation order, the MoE plan's free memory); (b) the `ref2v` / `a2v` arms load before `negotiate` refuses t2v (H3 about 2-3 min each): the D1 row's follow-up (4), negotiate before load (**done** in #213). Before: CPU part merged #181, ADR-0023: `fv-lab report tiers [--runs <run set>] [--check]` renders one run set, a fixture run-set golden; `docs/perf/tiers.md` committed with `Run set: none`, every variant not measured, because no committed run set measured a gold cell; the gold tier is `weights = "volume+r2"` since ADR-0019 §16 (draft PR `claude/lab-r2-weights`): the 18 R2-only trees, 530.74 GB, fetched per arm at a 147.86 GB peak, so all 32 variants run; command and cost in ADR-0019 §16.7, about 1.5-3.1 h, $3.70-$7.70, at most $9.96) |
| D4 | CI bridge: gold files parse and resolve; `last_validated.commit` age check; an on-demand `fv-lab run gold --all` workflow | visible per-model validation age in CI | CI | D3 | merged (#181, ADR-0023: `fv-lab gold check` (parse, resolve against `families/`, `git cat-file -e`, the age table; warn-only, `--max-age-days` opt-in) in `gold-check.yml` with the table in the job summary; the `gold` tier (one serve-engine cell per gold file, generated by `fv-lab gold cells`, drift-tested) and `fv-lab run gold --all`; `gold-run.yml`, `workflow_dispatch` only, refuses without `RUNPOD_API_KEY`; owner 2026-10-25: no secret, gold runs are on demand; the first gold run, 2026-10-26 at `7e580e5` (D3), validated 28 of the 32 variants, so `gold check` now shows their age) |

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
| B4 | One `fuse.rs` with a rounding policy; one `StepCache` trait with `EasyCache`/`TeaCache`/`FbCache`/`Taylor`; one disk prompt cache | byte identity per family | **GPU** byte-identity cells | B1 | merged (B4a #173: one `StepCache` trait, ADR-0020; B4b #179: one `rt::fuse` with rounding policies selecting the existing kernels, one host reference, one switch, the plan's value installed by the fuse stage, ADR-0022; B4c #177: one disk prompt cache `rt::prompt_cache`, ADR-0021; B4c2 #178: the six pre-existing prompt-cache key gaps closed. GPU byte-identity cells passed for each (`b4a`, `b4b`, `b4c`, `b4c2`); Wan and the recovered 8B are CPU evidence only (not on the EU volume). Follow-ups: ADR-0021 rows 7-8, `FASTVIDEO_FP8` linears and `FASTVIDEO_SDPA` in the text encoders, unverified) |
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

- 2026-10-24 14:24Z, coordinator: B4a GPU gate, runtime image
  `sha-abccc38` (the PR head) against main `sha-e93e357` (pre-B4a step
  caches), three single-cell RTX PRO 6000 pods in EU-RO-1 with
  `bz0ofkfx45` (`FV_B4A_CELLS=<cell>`): b4a-h3-fullopt-env, b4a-h3-fullopt-prof
  and b4a-h3-4step-tea all `compare-clips --off-identity` pass, 124 of 124
  frame digests equal; TeaCache reused 35 steps in both arms of each
  fullopt cell and 2 forwards in both arms of the 4-step cell (the 4-step
  fixture's threshold 4.0 does reuse), and the cache evidence lines
  (`h3 teacache …`, `fuse step cache`) are identical between arms (49, 49
  and 5 lines). The pods' fetch takes only JSON and stdout/stderr, so the
  cells' own `verdict.json` / `cache-evidence.log` stayed on the pods; the
  coordinator recomputed both from the fetched stderr
  (`<cell>-<arm>/cache-evidence.txt`, `<cell>.evidence.diff`, empty).
  About 35 GPU-minutes, ~$1.45. Evidence:
  `artifacts/runpod/b4a/abccc38-{10241423,10241424,10241425}/`.
- 2026-10-24 15:30Z: D1b (`claude/d1b-fv-lab-families`, draft PR #176): nine more
  `runpod-matrix.sh` families as fv-lab tiers and cells (`techniques`, `b6`,
  `rtx5090`, `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae`, `precision`
  default mode; 10 tiers, 38 cells, 113 arms with `rtx6000`). Four additive
  cell fields: `bin = "baseline"` (fetch-baseline.sh, `--baseline-sha`), `tae`
  with `{tae}`, `env = []`, and a compare's paired `gate`. The script changes
  only its dry run (`FV_DRY_RUN_GATES`, no TAE / baseline fetch when dry). The
  goldens cover every tier: dry run, gate cells, pod body.
  `fv-lab-compare-runs.sh` also compares compare verdicts and gate reports.
  ADR-0019 §9 has the 40-family inventory with a retire-when for each; `b4a`
  stays (its evidence verdict), as do `hd`, `ltxoffload` and the rest, each
  with a reason. No branch was deleted. CPU only, on build pod
  `l3d75u8a8zhw47` (agent `d1b-lab`): fastvideo-lab tests (all features),
  workspace clippy, fmt, lint and goldens `--check` all green. No Runpod call.
  GPU runs (coordinator), one per family, cheapest cells, about $13 in all:
  ADR-0019 §9.5.
- 2026-10-24 15:40Z: B4c opened draft PR #177 (`claude/b4c-prompt-cache`):
  `fastvideo_cudarc::rt::prompt_cache` (ADR-0021), one key derivation
  (`KeyRecipe`: domain and prompt trimming; H3 / Wan / LTX recipes) and one
  store (`Format`: two instances, `H3` for H3 and Wan, `LTX2`; layout, temp
  name, fsync, cleanup, finite check, error text and log line as named
  parameters); keys, file names, bytes and log lines unchanged, so existing
  cache directories are read as before. CPU evidence:
  `prompt_cache/equality.rs` against the verbatim pre-B4c caches
  (`legacy.rs`): keys (pinned values recomputed in Python), identities,
  byte-equal files, cross reads, identical accept / reject of every
  truncation and bit flip, hit / miss sequences, failure leftovers,
  concurrent writers; 18 key / format mutations all caught. On build pod
  `l3d75u8a8zhw47` (agent `b4c-prompt-cache`): fmt, workspace clippy
  `-D warnings`, `--features cuda` check (cudarc, engine-service, gpucheck,
  core), cudarc + models + engine-service + core tests (1414 passed),
  `cargo test --workspace --lib` (2071 passed), `scripts/serve/check.sh`,
  `scripts/gpu/lint.sh` green. GPU gate pending (coordinator):
  `runpod-matrix.sh b4c` with `FV_BASELINE_SHA=e18ec010`, cells
  `b4c-h3-4stepvsa` and `b4c-ltx25-512p` (`FV_B4C_CELLS`, one per pod),
  each base and PR arm a cold miss pass then a hit pass over its own cache.
- 2026-10-24 16:20Z: D1b review fixes on #176 (main merged in):
  - `fv-lab-compare-runs.sh` fails unless an arm ran with byte-identical frames, treats unreadable files as
    differences and prints ran / skipped / failed counts;
  - `pod up` refuses a baseline equal to `--sha` and a malformed or empty `FV_BASELINE_SHA` in `--extra-env`, and the
    runner refuses a baseline that is this build;
  - the goldens record gate kind / OFF arm and each arm's TAE file, with the volume's TAE copy complete;
  - the `techniques` H3 arms get their own text and AdaLN caches in the script and the cells. Before this, those
    gates did not cover the candidate's H3 text-encoder or AdaLN paths (docs/techniques.md);
  - CI path filters and the control.yml step name fixed.
  CPU only, build pod `l3d75u8a8zhw47`: lab tests, workspace clippy, fmt, lint and goldens all green.
- 2026-10-24 16:25Z, coordinator: B4c GPU gate, runtime image
  `sha-da9dbac` (the PR head after the review fixes) against main
  `sha-e18ec01` (pre-B4c prompt caches), two single-cell RTX PRO 6000 pods
  in EU-RO-1 with `bz0ofkfx45` (`FV_B4C_CELLS=<cell>`). `b4c-h3-4stepvsa`
  and `b4c-ltx25-512p` both pass: base-miss against pr-miss, base-hit
  against pr-hit and pr-miss against pr-hit are each `compare-clips
  --off-identity` pass with max abs diff 0 over all frames; every miss arm
  logged `text_cache: miss` and every hit arm `text_cache: hit`; the H3
  miss arms ran the text encoder on the FP8 path (`text_encoder:
  resident-fp8`), the hit arms read the cache (`text_encoder: cache`).
  Wall times (miss / hit): H3 base 204 s / 33 s, PR 114 s / 33 s; LTX base
  363 s / 130 s, PR 100 s / 27 s (the base arms run first and pay the cold
  volume reads). About 19 GPU-minutes, ~$0.80. Evidence:
  `artifacts/runpod/b4c/da9dbac-{10241612,10241613}/`.
- 2026-10-24 17:21Z, coordinator: D1b GPU comparisons, runtime image
  `sha-c21d801` (#176's head before the main merge), baseline `e93e357` for
  the `-base` arms. RTX PRO 6000 pods in EU-RO-1 with `bz0ofkfx45`: each family
  ran once through `fv-lab pod up --wait` and once through `runpod-http.sh`
  (the script) with the same cells, then `fv-lab-compare-runs.sh` (fails
  closed):
  - `b6` (b6-solh3-tree base/pr): 2 arms, 124 frames each byte-identical; 8 reports the same;
  - `techniques` (h3-480p-fullopt base/env/prof): 3 arms, 248 frames each byte-identical; 9 reports the same;
  - `precision` (ltx25-4k5s-sol f32act/bf16act): 2 arms, 242 frames each byte-identical; 9 reports and the
    gate report the same;
  - `rtx5090` (h3-480p dense/sol/fullopt/fullopt-taeh3): 4 arms, 248 frames each byte-identical; 14 reports
    the same.
  Within each run the gates also hold: b6 and techniques base against the
  candidate arms pass `--off-identity` (identical), so the techniques arms
  now cover the candidate's own H3 text encoder and AdaLN caches. The fv-lab
  `rtx5090` and `precision` pods were first refused by fv-deploy's floor
  projection (4 h cap × (quote + the account's current spend) would cross
  $8 with five pods up), as designed; they ran once nothing else was up.
  The first script-side launch failed before creating any pod (a launcher
  quoting error on my side). About 133 GPU-minutes, ~$5.50. Evidence:
  `artifacts/runpod/{b6,techniques,precision,rtx5090}/c21d801-*` and the
  comparison outputs in `artifacts/runpod/d1b-compare/`.
- 2026-10-24 17:40Z: B4c2 opened draft PR #178 (`claude/b4c2-prompt-key-gaps`):
  the six key gaps of ADR-0021 closed. New key fields, each the value the
  run resolved: `Numerics` (GEMM math, compute capability, bf16
  activations; `Numerics::live` at the lookup), `WeightsRead` plus the FP8
  tree's identity (H3 `ResidentDecoder::prequantized`, the streamed tree;
  LTX `gemma_read`), LTX `TextMode` (`planned_mode`, taken before the
  lookup and run by the miss), recovered-8B file identities of base, LoRA
  and adapter. New domains (`fastvideo h3 text conditioning v2`,
  `WANTEXT2`, `fastvideo ltx2 text conditioning v2`); layout and magics
  unchanged; every existing entry misses once. CPU evidence: pinned keys
  recomputed in Python, the families' key tests, `numerics` tests,
  `b4c2_keys_never_name_a_legacy_entry` (no new key names a legacy entry;
  an entry legacy wrote is a miss for the new code), the v1 derivations
  still equal to `legacy.rs`; 20 mutations (each new field dropped, each
  recovered file dropped, each domain reverted, `Numerics` parts dropped,
  `prequantized` never set, tree keyed as at-load, one mode field) all
  caught. On build pod `l3d75u8a8zhw47` (agent `b4c2-keys`): fmt, workspace
  clippy `-D warnings`, `--features cuda` check (cudarc, engine-service,
  gpucheck, core), cudarc + models + engine-service + core tests (1424
  passed), `cargo test --workspace --lib` (2081 passed),
  `scripts/gpu/lint.sh`, `b4-verdict.test.sh`, `scripts/serve/check.sh`
  green. Found and recorded, not closed: `FASTVIDEO_FP8` linears and
  `FASTVIDEO_SDPA` (ADR-0021 "B4c2" rows 7, 8). GPU gate pending
  (coordinator): `runpod-matrix.sh b4c2` with `FV_BASELINE_SHA=17e4a1dc`,
  cells `b4c2-ltx25-fp8`, `b4c2-ltx25-mode`, `b4c2-h3-numerics`,
  `b4c2-h3-tree`, `b4c2-h3-noop` (`FV_B4C2_CELLS`); recovered 8B and Wan are
  CPU evidence only (not on the EU volume).
- 2026-10-24 17:56Z: B4c2 review fixes on #178 (coordinator's review: all
  six gaps closed, nothing blocking): the H3 key gets the resident /
  streamed mode (`EncodeSettings`, shared with LTX); `EncoderRun`
  (`h3/text.rs`) is the pure choice of weights read and mode, its table
  tested; Wan reads `Numerics::live` inside UMT5's `with_bf16_act(false, ..)`
  scope; LTX `plan_mode`, `gemma_tree`, `gemma_weights_read` pure and tested,
  FP8 with an explicit streamed residency keyed as the bf16 checkpoint it
  streams (computation unchanged; ADR notes the tree-map case); H3 tree and
  recovered identities through `files_head_tail_cached`; ADR: in LTX `auto`
  the key depends on free memory at the lookup (spurious misses only). New
  pins (Python): H3 `29d517fd…` / tree `9556f05a…` / streamed tree
  `59ddcb14…`, recovered `1af19d5f…`, Wan `1f09144a…` (LTX unchanged). 31
  mutations all caught. Gates on the build pod green (cudarc + models +
  engine-service + core tests 1430, `--workspace --lib` 2087). No H3
  streamed-FP8 GPU cell: no flag makes H3 stream FP8. Merged main
  `1100c35f` (D1b).
- 2026-10-24 18:30Z: B4b opened draft PR #179 (`claude/b4b-fuse`, from main
  `17e4a1dc`, merged with `1100c35f`): `rt::fuse` with ADR-0022. The
  inventory found one computation per op (residual + modulated norm, q/k
  norm + RoPE) whose families differ in rounding points, contraction,
  normalizer, reduction order and RoPE pairing; the kernels stay as they
  are (their source is pinned) and a typed `RoundingPolicy` / `QkPolicy`
  per family op selects the kernel and its mode arguments; Wan's and H3's
  CPU twins run one host reference under their policies; the four switches
  are one implementation, and the H3 / LTX-2 plans' `kernel_fusions` (an
  `rt::fuse::BlockFusion`) is installed on the DiT by the fuse stage
  (ADR-0017). CPU evidence: `rt/fuse/equality.rs` against the verbatim
  pre-B4b code, bit for bit, and ten mutations each caught. GPU gate
  prepared, not run: `runpod-matrix.sh b4b` (`b4b-h3-4stepvsa`,
  `-off`, `b4b-ltx25-512p`, `-off`; `b4b-h3-fullopt` when named;
  `FV_B4B_CELLS`), every arm CUPTI-traced so the verdict sees the fused
  kernels run in both arms. Wan: CPU evidence only (no EU-volume tree).
- 2026-10-24 18:58Z, coordinator: B4c2 GPU gate, runtime image
  `sha-072f0b3` (#178's head after the review fixes and the main merge)
  against main `sha-17e4a1d` (B4c, before B4c2). Five single-cell RTX PRO 6000
  pods in EU-RO-1 with `bz0ofkfx45` (`FV_B4C2_CELLS=<cell>`), all `pass`:
  - `b4c2-ltx25-fp8` (gap 1/6): main's bf16 read hits the FP8 entry and its
    frames differ from a fresh bf16 encode (PSNR 24.7 dB, max abs diff 218):
    the gap gave silently wrong output. The PR's read misses and is
    byte-identical to the fresh encode.
  - `b4c2-h3-numerics` (gap 4): written under `FASTVIDEO_BF16_ACT=0` (the
    encoder ran `streamed-fp8`), main's default read hits it, PSNR 18.2 dB
    against the correct output; the PR misses and equals the fresh encode.
  - `b4c2-ltx25-mode` (gap 5): main hits, the PR misses and equals the fresh
    encode; streamed and resident Gemma were also bit-equal on this card.
  - `b4c2-h3-tree` (gap 6): main hits, the PR misses and equals the fresh
    encode; the FP8 tree and quantise-at-load were also bit-equal.
  - `b4c2-h3-noop`: base and PR byte-identical on the miss and the hit, the
    PR's hit equals its miss.
  Gaps 2 (recovered 8B) and 3 (Wan) are CPU evidence only (no trees on the
  EU volume). About 56 GPU-minutes, ~$2.30. Evidence:
  `artifacts/runpod/b4c2/072f0b3-1024184{0,1,2,3}/`, `072f0b3-10241839/`.
- 2026-10-24 19:15Z: B4b review fixes on #179: the GPU gate's kernel
  evidence was cut to the trace's 25 slowest kernels; `FASTVIDEO_GPU_TRACE_TOP`
  (report only) now lists every kernel in the PR arm (the baseline keeps
  25), and `b4b_verdict` requires a complete PR table, no dropped records,
  equal launch counts, the base's listed fused / bf16-chain kernels equal
  call for call, every named gated kernel (`on`) or the unfused path's
  kernels and no gated one (`off`), and H3's block structure in the call
  counts; the plan's value now also covers the H3 text refiner and the
  LTX text connectors; the switch scopes restore through drop guards; H3's
  host `gate_residual` rejects partial rows; `QkPolicy` documented as
  descriptive on the device; the mutations are a script
  (`scripts/dev/b4b-fuse-mutations.sh`, eleven, all caught).
- 2026-10-24 19:20Z, coordinator: B4b GPU gate, runtime image
  `sha-c493bd2` (#179's head after the review fixes) against main
  `sha-17e4a1d` (before B4b), four single-cell RTX PRO 6000 pods in EU-RO-1
  with `bz0ofkfx45` (`FV_B4B_CELLS=<cell>`), every arm under
  `FASTVIDEO_GPU_TRACE=1 FASTVIDEO_GPU_TRACE_TOP=100000`. All four `pass`:
  base against PR `compare-clips --off-identity` identical on every frame,
  `dropped_records` 0 and the same step kernel count in both arms, the PR's
  table complete:
  - `b4b-h3-4stepvsa` (fuse on): 1785 launches per traced step;
    `fvf_h3_gate_res_norm_mod` 49 + `h3_norm_mod` 1 = `h3_gate_residual` 1 +
    49 = `h3_res_gate_norm_mod` 50 (the block structure), `fvf_h3_qkvg_heads`
    50, `fvf_merge_heads_mx` 45, the same in both arms.
  - `b4b-h3-4stepvsa-off`: 2074 launches; no gated `fvf_*` kernel;
    `h3_norm_mod` and `h3_gate_residual` 50 each, `h3_qk_norm_rope` 100.
  - `b4b-ltx25-512p` (fuse on): 7464 launches; `fvf_ltx_res_norm_mod` 480,
    `fvf_ltx_qk_norm_rope` 576, `fvf_ltx_gate_merge` 288 in both arms.
  - `b4b-ltx25-512p-off`: 12072 launches; no gated kernel; `mx_rms_norm`
    1056, `mx_rope_half` 384.
  The baseline lists its 25 slowest kernels only (it predates
  `FASTVIDEO_GPU_TRACE_TOP`); every fused kernel it lists has the PR's call
  count. Each cell's `fusion-evidence.log` / `trace-meta.txt` were
  recomputed from the fetched stderr with `b4b_evidence` (the fetch takes
  only JSON and stdout/stderr). Wan: CPU evidence only. About 32
  GPU-minutes, ~$1.35. Evidence: `artifacts/runpod/b4b/c493bd2-1024191{0,1,2,3}/`.
- 2026-10-24 19:45Z, coordinator: B4 complete. B4b #179 merged (38d74e5d)
  after its GPU gate (19:20Z) and the main merge; B4c2 #178 merged
  (2812b27d); D1b #176 merged (1100c35f). Status rows corrected: A3 merged
  (#152), D1 active (D1a #171, D1b #176 merged; the `fastvideo`, `h3arms`,
  `trace`, `ltxfps` and `ltxvae` tiers still to compare on GPU before their
  script branches go), B4 merged. Spend today since 15:55Z about $13 (D1b
  $5.50, B4c $0.80, B4c2 $2.30, B4b $1.35, build pod); balance about $21.
- 2026-10-24 21:30Z: D3 / D4 CPU parts on `claude/d3-d4-lab-report` (draft
  PR, ADR-0023): `fv-lab report tiers` and `docs/perf/tiers.md` (run set:
  none, every variant not measured), `fv-lab gold check` (32 gold files
  resolve, 0 validated, warn-only), the `gold` tier (32 generated cells,
  about 11 run on the EU volume), `gold-check.yml` and `gold-run.yml`.
  Build pod only; no GPU. Open: the owner's `RUNPOD_API_KEY` secret for
  `gold-run.yml`, and the GPU gold runs (about $2.50 for one pass).
- 2026-10-24 21:20Z: #181 review fixes (coordinator's review): no pod
  outlives a cancelled `gold-run` (`pod down --from-ledger` from fv-deploy's
  create line; every fv-lab and `runpod-http.sh run` pod deletes itself at its
  wall cap), `cells` checked as arms, the key step-scoped and the actions
  pinned, a `gold-runs` environment recommended (owner), a same-named arm of
  another tier never counts as a gold measurement, `git rev-parse --verify`
  with ambiguity and ancestry. Build pod only.
- 2026-10-24 21:54Z, coordinator: pod-side wall cap verified live on
  #181's head 83c5d6e1 (GraphQL `podTerminate`). `runpod-http.sh run` with
  `FV_POD_CAP_S=240` (b4b cell, image `sha-c493bd2`); the driver and its
  detached backstop were killed right after create, so only the pod could
  delete itself. Pod `qtqbwuyi2ypsab` created 21:49:29, gone (404) at
  21:53:58: the cap after container start. Before the fix the same test
  left pod `pq69qpoh5b0mxy` running (deleted by hand after 9 min); the probes
  showed REST DELETE with the pod key is 403 and GraphQL `podTerminate` is
  200 (ADR-0023 §4). The idle watchdogs use the same call now. Cost of the
  tests about $0.80. Cargo gates on 83c5d6e1 (build pod): fmt, workspace
  clippy, fastvideo-lab clippy (all features), fastvideo-lab tests (100),
  `cargo test --workspace --lib` (2121) green.
- 2026-10-24, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): `stream_causal`
  session-limit hang root-caused and fixed on `claude/fix-causal-limit-hang`
  (draft PR): `Metronome::poll` could ask for a sub-nanosecond wait that
  tokio's paused clock never gets past (first block at a wheel slot
  boundary, then f64 deadline a few ulps after the clock); it now ticks
  within 1 µs of the deadline, and the paused-time tests run under a 60 s
  real-time guard. Verified by the coordinator on the replacement build
  pod `awavskcvtbs0ub` (`l3d75u8a8zhw47`'s host had no free vCPU; cloned onto
  the same `fv-build` volume and the old pod deleted): the regression test
  passes and fails with the slack set to 0 (anchor 3 ms: a 2.2e-16 s wait);
  20 of 20 repeated runs of the session-limit tests pass; fmt, workspace
  clippy, `cargo test -p fastvideo-engine-service` (141),
  `cargo test --workspace --lib` (2109) and `scripts/serve/check.sh` green.
  No GPU cost.
- 2026-10-25 17:20Z, coordinator: D1b GPU comparisons for the remaining
  five tiers, runtime image `sha-b0d0bb8` (main), fv-lab built from main.
  RTX PRO 6000 pods in EU-RO-1 with `bz0ofkfx45`, each family once through
  `fv-lab pod up --wait` and once through `runpod-http.sh run`, same cells
  (ADR-0019 §9.5), then `fv-lab-compare-runs.sh`, all exit 0:
  - `fastvideo`: 4 arms the same (2 ran, 248 frames each byte-identical; 2 skipped in both, `h3-8step` not on
    the EU volume); 8 reports and the gate report the same;
  - `h3arms`: 13 arms the same (2 ran byte-identical, 11 `h3-8step` arms skipped in both); 14 reports the same;
  - `trace`: 2 arms the same (1 ran, 248 frames byte-identical; 1 skipped in both);
  - `ltxfps`: 2 arms the same (1 ran, 121 frames byte-identical; 1 skipped in both);
  - `ltxvae`: 2 arms the same (both ran, 121 frames each byte-identical); the compare the same.
  All nine D1b tiers now match the script, so their `runpod-matrix.sh`
  branches can go (ADR-0019 §9.5). fv-deploy's floor projection refused one
  launch while five pods were up, as designed. About 75 GPU-minutes, ~$3.
  Evidence: `artifacts/runpod/{fastvideo,h3arms,trace,ltxfps,ltxvae}/b0d0bb8-*`,
  comparison outputs in `artifacts/runpod/d1b-compare/`.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): every remaining
  in-pod REST self-delete (ADR-0023 §4's 25 sites: scripts/serve/e2e,
  family-pool, sol-bench, the experiments) now calls GraphQL `podTerminate`
  through one `fv_self_delete` (`scripts/gpu/lib/self-delete.sh`, embedded by
  the launchers, inline and checked in the pod scripts,
  `scripts/gpu/tests/self-delete.test.sh` in `control.yml`); fv-control's and
  fv-boot's watchdogs keep REST with the account key but retry anything but
  a 2xx or 404. Branch `claude/pod-self-delete-graphql` (draft PR); build
  pod gates green (`cargo test --workspace --lib` 2123, fastvideo-deploy
  138); no GPU cost. Live check by the coordinator pending.
- 2026-10-25, coordinator: #183's self-delete verified live on a CPU pod
  (`inuqqmjr86ah47`, the library then `sleep 60; until fv_self_delete …`:
  gone on the first attempt, 17:01:57Z). Review fixes, Claude
  (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): launchers render through
  `start_cmd` / `print-start`, the check runs each rendered self-delete and
  checks order, aliases and every path (`self-delete.yml`); the key goes to
  curl on stdin, `.resp` removed, `_fvsd_code`; goldens regenerated. Build
  pod: fmt, clippy, fastvideo-deploy (138), fastvideo-lab all features (100),
  `cargo test --workspace --lib` (2123) green. No GPU cost.
- 2026-10-25 17:32Z, coordinator: the shipped `fv_self_delete` (c22aef8a,
  key on stdin through `-H @-`) re-checked live on a CPU pod
  (`xtsg3gdqwf4gny`, python:3.12-slim with Debian curl): gone on the first
  attempt, 17:31:57Z. No GPU cost.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): the ten gated
  families leave `runpod-matrix.sh` on `claude/d1c-retire-script-branches`
  (draft PR, ADR-0019 §10): their `case` branches and `tae_gated_cell`
  deleted, `scripts/gpu/fv-lab-retired.sh` lists them, `runpod-http.sh
  run`/`plan` hand them to `fv-lab pod up` (`FV_LAB_BIN`, `FV_DRY_RUN=1`
  prints the command), their fixtures frozen as fv-lab's own goldens,
  `runpod.sh rtx6000` deleted, `headline` runs its cells of them through
  `fv-lab run --runs-dir`; `precision`'s `ltx-nvfp4` mode and `ltxvae`'s
  `FV_VAE_GEN` arms dropped. Every remaining family's dry run equal before
  and after. No GPU cost.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): C4 (one image with
  boot-time config, owner decision 2026-10-25) on `claude/c4-one-image`
  (draft PR): `[[images]]` is `cuda` + `cpu`, every preset on `cuda`;
  fv-boot refuses to boot without `FV_WORKER_TOML_B64` / `--config` /
  `FV_WORKER_CONFIG`; fv-control, fv-deploy, the autoscaler and the Runpod
  scripts send every config at boot (largest 5600 base64 bytes, guard
  16 KiB); stored `variant` specs read through `legacyVariant`. No live
  template, endpoint or GHCR tag changed; the owner's ordered live steps are
  in `docs/serve/images.md`. Gates on build pod `pgozx2aeruu9jx`: fmt,
  workspace clippy, crate tests, `cargo test --workspace --lib`,
  `scripts/serve/check.sh`; control gen `--check`, typecheck, unit tests;
  script lint and goldens. No GPU cost.
- 2026-10-25 19:35Z, coordinator: merged #183 (in-pod self-delete through
  GraphQL `podTerminate`, re-checked live on CPU pod `xtsg3gdqwf4gny`),
  #184 (D1b evidence for the last five tiers), #185 (the ten fv-lab tiers'
  `runpod-matrix.sh` branches retired; independent review, its findings
  fixed: unsupported settings refused, `--wall-cap-s`, `gold` run refused,
  `fv-lab-route.test.sh`) and #186 (C4 one image with boot-time config;
  independent review, its four bugs and six further items fixed; Rust gates
  re-run on the new build pod `k66zz15an4sktc`, a clone of `pgozx2aeruu9jx`
  after a host vCPU shortage). No GPU cost. Open with the owner: LB keep or
  retire; whether `precision`'s `ltx-nvfp4` mode or `ltxvae`'s `FV_VAE_GEN`
  arms come back as tiers; C4's live steps.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1d first slice on
  `claude/d1d-gate-verdict` (draft PR, ADR-0019 §11): the cell schema's
  `[verdict]` (fresh verdict arms, evidence files, line counts, identity
  pairs, evidence diffs, per-cell and family verdict files as the script
  writes them), `b4a` as a tier with goldens from the script (its branch
  stays until its GPU gate), `fv-lab gate <run dir>` (promotion gates via
  `fv-gpucheck gate` and verdicts, offline), `pod fetch` and
  `fv-lab-compare-runs.sh` cover the verdict files; `fv-lab-goldens.sh`'s
  `has_branch` pipefail bug fixed. The script's own `b4a` branch and
  `fv-lab run b4a` write byte-identical verdicts on one fake box
  (`tests/verdict.rs`). Gates on build pod `k66zz15an4sktc`. No GPU cost;
  the `b4a` GPU gate (one 4-step pair per side, ~$1.10) is the
  coordinator's.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): #188 review fixes
  (F1-F8): `fv-lab run` records its arm selection and `fv-lab gate` reads it
  (no guessing a cut-short run), writes beside the run (`<run>.regate`),
  `--check` compares with the run's files, fails a gate whose arms ran
  without `benchmark.json`; the script's evidence grep / sed / diff pinned
  to bytes; `fv-lab-compare-runs.sh --expect-verdicts`; `FV_B4A_CELLS`
  refused by fv-lab. New script-vs-fv-lab cases (UTF-8 locale, CRLF, NUL,
  empty stderr, missing compare, skipped arms, one-arm selection,
  interrupted run). b4a dry run unchanged. No GPU cost.
- 2026-10-25 20:30Z: C3c (`claude/c3c-control-launch`, draft PR): fv-control
  on fv-deploy's `Launch` (ADR-0018 §7). `gen-configs.mjs` generates
  `control/src/launch-gen.ts` (TypeScript + zod) from
  `crates/fastvideo-deploy/schemas/*.v1.json` (`--check` in `control.yml`,
  which now runs on schema changes); cluster workers, edge fronts, standalone
  pods and GPU serverless endpoints are schema-checked `Launch` values whose
  Runpod bodies follow `pod.rs` / `serverless.rs`; payloads unchanged (a
  committed corpus of 1024 bodies from main's builders, `payload-corpus.test.ts`); `launch-goldens.test.ts` holds the TS builders to the C3a
  goldens and h3-turbo / h3-max / ltx / ltx-pro to fv-deploy field by field
  (intended differences listed). CPU pods / endpoints, GMI / Brev and the v2
  envelope stay outside Launch v1 (proposal: a CPU section and `gmi` /
  `brev` kinds after C3b). No Rust change, no live resource. No GPU cost.
- 2026-10-25 21:00Z: C3c review fixes on #189: the payload corpus committed
  (`control/test/unit/fixtures/payload-corpus.json`, 1024 cases written from
  main's builders at 2258aec2, regenerate command in the test); intended
  differences pinned per context (standalone / direct / edge / serverless)
  and checked for staleness per test; `gen-configs.mjs` refuses any JSON
  Schema keyword it does not translate; serverless guards fail-closed and
  named (`SLS_GUARDS_NOT_CHECKED`, proposal for `null` in ADR-0018 §7.3); a
  Launch that fails the schema is a 400 / 409 before the op and a clean
  "no pod" inside it; schema errors never carry a value (tested).
- 2026-10-25 20:42Z, C3b agent (`claude/c3b-vast-gcp`): Vast and GCP
  providers on fv-deploy's `Provider` trait under the shared guards
  (`--kind vast|gcp`), goldens from `vast.sh plan` and `vm.sh`'s payload
  functions; `vm.sh` fixed (its metadata items exceeded the 128 KiB
  argument limit, so `up` failed on main; `vm.test.sh` 21 failed → 38
  passed); the autoscaler's Runpod requests, answer mapping, balance and
  template pod now fv-deploy's (ADR-0018 §8, §9). CPU gates on build pod
  `k66zz15an4sktc`. No live resource created, no GPU cost; the Vast and GCP
  live gates are the owner's (§8.4).
- 2026-10-25, F3b (sub-agent of the coordinating session, worktree
  `claude/f3b-cluster-renames`): owner decision "keep LB endpoints; F3b
  reduced to the renames". `[gateway]` → `[cluster]` in fv-serve (old name
  read with a warning, both refused, round-trip tests over every shipped
  config), fv-boot's and the shell boots' registration edit under either
  name; shipped configs unchanged (stage 1 of ADR-0011 "Renaming a
  worker-config section": images built before it refuse `[cluster]`);
  `gw_workers` kept. Merged main with C3c (#189): the payload corpus
  regenerated for the one intended change, the shell boots' registration
  edit (253 of 1024 cases, `pod/*` cluster, edge and standalone bodies,
  `dockerStartCmd` only; no serverless case). Gates on build pod
  `k66zz15an4sktc`; control checks local. No GPU cost.
- 2026-10-25, F3b review fixes on #192 (coordinator's independent review,
  nothing blocking): fv-boot appends `[gateway]` too (nothing writes
  `[cluster]` before stage 2; `FV_SERVE_BIN` can name an older fv-serve);
  fv-control refuses an inline `config_toml` that sets `[cluster]`; the
  guard renames through a TOML parse; ADR-0011's stage 2 lists the writers
  to flip. Merged main at e82b6b78 (#188). Gates on build pod
  `k66zz15an4sktc`; control checks local. No GPU cost.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): two dropped script
  modes restored as fv-lab tiers (owner decision 2026-10-25, "restore dropped
  test modes") on `claude/restore-nvfp4-vaegen` (draft PR, ADR-0019 §12):
  `precision-nvfp4` (was `FV_PRECISION_ARM=ltx-nvfp4 runpod-matrix.sh
  precision`, 4 arms) and `ltxvae-gen` (was `FV_VAE_GEN=1 ... ltxvae`, 16
  arms, `ltxvae`'s cells plus a gen cell per workload); no schema change;
  fixtures from the script at `5987e6a4` in each mode, frozen
  (`fv-lab-restored-goldens.sh` reproduces them); `runpod-http.sh` names the
  tier for the modes' settings. GPU check pending (coordinator: fv-lab at the
  PR's commit vs `runpod-http.sh` at `5987e6a4`, image `sha-5987e6a`
  published; about $3.35). No GPU cost.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): #191 review fixes
  (main at `6c13e195` merged in): the byte-identity check's crate list from
  `cargo tree -p fastvideo-gpucheck --features cuda` (adds
  `fastvideo-manifest`, changed only in its deploy section) and a note that
  the PR head's image carries another tools build than `tools-v0.1.49`; the
  old knobs' exact `--cells` lists in `docs/dev/fv-lab.md`; `fv-lab run` /
  `pod up --extra-env` refuse what no tier reads (the same keys as
  `runpod-http.sh`), the mode's setting first, all keys in one message;
  `runpod-matrix.sh` names the restored tier. No GPU cost.
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): `b4a` retired from
  the script on `claude/d1d-retire-b4a` (draft PR, ADR-0019 §11.6), after its
  GPU gate on `e82b6b7` (base `e93e357`, cell `b4a-h3-4step-tea`; fv-lab pod
  `j25pn9a2xc1ll8`, script pod `ik1hd9tz9cay82`; `fv-lab-compare-runs.sh
  --expect-verdicts 5` exit 0, 124 frames per arm byte-identical, both
  `matrix_exit=0`, both `fv-lab gate --check` 0 differences; evidence in
  `artifacts/runpod/b4a/`): `b4a` in `FV_LAB_RETIRED`, its branch deleted
  (the other 32 families' dry runs unchanged), its fixtures frozen,
  `runpod-http.sh` routes it to `fv-lab pod up` and refuses `FV_B4A_CELLS`
  (now in `FV_LAB_NOT_READ` / `knobs::MODE_KNOBS`), `tests/verdict.rs` runs a
  frozen copy of the branch (`fixtures/b4a-script/`). Follow-up recorded: the
  runtime image job pushed `sha-e82b6b7` before its fv-lab check failed (the
  first digest's fv-lab predated `[verdict]`). No GPU cost in this PR (the
  gate's two pods were the coordinator's).
- 2026-10-25, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): review fixes on the
  `b4a` retirement PR (#195, main `f74c7761` merged in): the frozen b4a
  script copy now vendors everything it sources and fakes the frame's host
  paths (CUDA links, the image build id); `runpod-http.sh` names `b4a` in the
  `FV_B4A_CELLS` refusal only for `b4a`, and refuses the pair selectors
  `FV_TECH_CELLS` / `FV_B6_CELLS` / `FV_B4A_CELLS` from the host environment
  for every retired tier. Also recorded there: the coordinator's #191 GPU
  check (ADR-0019 §12 "Result"): `ltxvae-gen` and `precision-nvfp4` at
  `8b18d89` against the script at `5987e6a4`, 4k5s arms byte-identical (121
  and 242 frames per arm), `fv-lab-compare-runs.sh` exit 0 and
  `matrix_exit=0` on all four pods (fv-lab `wgzwvt5mjcn1nv`,
  `ymrchzi0z3eixy`; script `64opofps1163h8`, `kxhdp1v19w151y`; all deleted);
  the NVFP4 pair's lossy gate verdict `fail` on both sides, as in 2026-09.
  No GPU cost in this PR.
- 2026-10-25 21:45Z, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): serve-image
  run 286 (c3ceb96c, C4) hung in "Smoke-check the serve images" on the
  refusal check. Root cause: main's `FV_PREBUILT_STALE=latest` took the
  newest tools release, tools-v0.1.50 (d428ff46, before C4), whose fv-boot
  replaced the image's `boot-build` stage (images labelled
  `dev.fastvideo.tools=tools-v0.1.50`); that fv-boot boots fv-serve's
  default config when none is given, so `docker run --rm` served forever
  (reproduced with the release's binaries: 0.1.50 execs fv-serve, 0.1.51
  refuses in 0 s). Fix (`claude/fix-serve-smoke-hang`, draft PR): fv-boot
  comes from a release only when it is built from the commit's tools
  inputs (`prebuilt.sh contexts --stale`); `ci-images.sh smoke` bounds every
  `docker run` (`FV_SMOKE_TIMEOUT_S`, 120 s, removes the container, names
  the check); step `timeout-minutes` on both smoke steps; PR CI
  `serve-boot.yml` runs the fv-boot binary as the smoke does
  (`tests/fv_boot_bin.rs`, fails on d428ff46's fv-boot), the bounded-smoke
  test and the tools-release selection tests. Build pod k66zz15an4sktc;
  no Runpod resource created, no run cancelled or re-run.
- 2026-10-25 23:30Z, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), coordinator:
  merged #191 (restored tiers), #192 (F3b stage 1), #193 (C3b), #194 (the
  serve smoke fix, with review fixes: the timeout is always named, the
  fv-boot test's pipes go to files, a pinned `FV_TOOLS_VERSION` keeps its
  release's fv-boot, the image label `dev.fastvideo.fv-boot`), #195 (D1d:
  `b4a` retired, the #191 GPU evidence) and #196 (a fv-control Brev host
  script race: a failing step's `host_failed` report read `host.log`
  through the async `tee` before the step's stderr reached it; it now
  tails the step's own output; a test with a lagging `tee` reproduces it).
  Serve-image run 286 (stuck since 19:34Z) cancelled; run 296 replaced
  `:cuda-latest` / `:cpu-latest` at 22:37Z from `feae3a14` with
  tools-v0.1.55 (an fv-boot that refuses a boot without a config). The
  owner still deploys fv-control before promoting `stable` (the deployed
  fv-control creates CPU endpoints without a worker config). #191 GPU
  check: 4 pods, about $2.1 (balance $51.94 to $49.85, build pod included). Follow-ups: the runtime
  image job pushes `sha-<7>` before its fv-lab check; an fv-lab fetch once
  missed `tree.txt` / `matrix.out`; C3b: Vast has no region constraint
  (owner question) and the Vast and GCP live gates are the owner's; next
  packages C3d, D1c / D1d `oracle` / D1e, the D3 GPU gold runs; E5 blocked.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1c on `claude/d1c-lab-report`
  (draft PR, ADR-0019 §13): `fv-lab report summary` and `report table`, ports of
  the retired families' `live.log` summary greps, `hd_report.py` (less the
  keyframe sharpness), the four experiment `*-summary.py`, `sol_bench_table.py`
  and `timings.sh`, held byte for byte to 52 fixtures the scripts write on
  committed run directories (`fv-lab-report-goldens.sh`; the deleted branches'
  lines frozen verbatim, each script-side run's lines found in its `live.log`);
  `pod up --wait` / `pod fetch --fetch-tree` (the `FV_FETCH_TREE` mirror,
  equal to the script's `fetch_tree` on a fake pod; `runpod-http.sh` maps it);
  `sol_bench_table.py` and the `*-summary.py` deleted, `hd_report.py` and
  `timings.sh` kept until their callers move. Gates on build pod
  `8gg7uz30x3nesu` (the coordinator's clone of `k66zz15an4sktc`). No GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): #198 review fixes
  (ADR-0019 §13.3, §13.4): the tree fetch refuses listing entries that are not
  a single name inside the run directory (absolute, `..`, `./`, `/` `\` NUL
  once decoded, malformed `%`), caps the depth at 16, streams each file with a
  size cap (`Http::download_capped`, additive in fastvideo-deploy, the reqwest
  client stops on `Content-Length` or mid-stream) into a temporary name,
  retries the top-level listing 5 times and always fetches `RUN_FILES` / `DONE`;
  GNU `\<` `\>` `` \` `` `\'` refused in `--fetch-skip`; `FV_FETCH_MAX_MB` must be
  decimal; strict JSON (Python's grammar, no lone surrogate); `sol_timings`
  globs dotfiles as `Path.glob`; jq's `-0` and `basename .`; no `--jpg-out`; a
  synthetic hd run covers `## streams`; a reverse `live.log` check; CI path
  filters widened. No GPU cost.
- 2026-10-26 01:00Z, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), C3d agent: `fv-deploy templates`,
  `release promote | rollback | resolve`, `reconcile` and the D1 `deployments` ledger (ADR-0018 §10), the
  guards' `null` ("not checked by this launcher", the `Launch` schema change additive); draft PR #199 from
  `claude/c3d-templates-release`. Built and tested on the build pod (`k66zz15an4sktc` had stopped; the
  coordinator replaced it with `8gg7uz30x3nesu` on the same `fv-build` volume); control gates locally; the
  goldens and `release.test.sh` also under jq 1.6. No Runpod template, endpoint or pod, GHCR tag, D1 row
  or R2 object touched; the live gates are the owner's (§10.6). No GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), D1e agent: the port one-offs (ADR-0019 §14) on
  `claude/d1e-one-offs` (draft PR): `h3-graphs` kept as the fv-lab tier `h3-graphs` (h3-turbo at 768p: trace, sync /
  lazy / graphs, `--off-identity` compares; every tree on the EU volume), its commands cut from `h3-graphs-pod.sh` and
  held word for word after a documented path mapping, its prompt sets the script's own output; the other nine
  experiments deleted (eighteen scripts, two `*-plan.py`), their docs pointing at git `22a5fd7b`;
  `FV_LAB_EXPERIMENTS` (`runpod-http.sh run` / `all` refuse such a tier); `runpod-http.sh`'s callers and modes listed
  for its retirement. Gates on build pods `8gg7uz30x3nesu` and (after the coordinator's switch) `dbf6gok3crgj9x`;
  the shell goldens locally. No Runpod resource created; the
  `h3-graphs` GPU check (one PRO 6000, ~35 min, ~$1.45) is the coordinator's.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), D1e agent: #200 fixes. CI's gawk read the `-v` patterns of
  `fv-lab-experiment-goldens.sh` without their backslashes (`^COMMON=(`, an invalid regexp; mawk kept them): the
  block cuts are plain-string matches through `ENVIRON` now, checked under gawk, mawk and busybox awk. Then the
  coordinator's review fixes (ADR-0019 §14.2-§14.4: the self-delete floor when the h3-graphs scripts go, the
  reproduction of the deleted experiments' gold numbers until PR #201's gold run, the `--fetch-tree` command, loud
  checks, nits). Build pod `dbf6gok3crgj9x`. No GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1d second slice on `claude/d1d-oracle` (draft PR,
  ADR-0019 §15): `det-short`, `determinism` and `oracle` (its nine tensor-diff targets) as fv-lab tiers, goldens from
  the live branches (`oracle` under `FV_ORACLE_TARGETS` and `FV_GEN_TIMEOUT_S=5400`), additive cell-schema fields
  (ungated arms, `fresh` / `requires` / `echo`, `[reference]`, `[[prompt_file]]`, `[repeats]`) and `src/repeats.rs`;
  `tests/repeats_oracle.rs` runs the live branches and `fv-lab run` on one fake box, outputs byte-identical; script
  changes dry-run only but `determinism`'s verdict `|| true` and `det-short`'s flat `env`; branches kept until the
  coordinator's GPU checks (§15.7, about $6); `runpod-http.sh upstream` stays with `oracle.sh` until an `fv-lab oracle`
  driver (§15.6). Gates on build pods `8gg7uz30x3nesu` and `dbf6gok3crgj9x`. No GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): fv-lab tiers read R2 trees, on
  `claude/lab-r2-weights` (draft PR, ADR-0019 §16): a tier mode `weights = "volume+r2"` (the EU
  volume's trees linked into `<scratch>/weights`, every other tree fetched per arm with
  `r2-weights.py plan` and `r2-fetch.py`, the R2 key as Runpod secret references), the `gold` tier on
  it with a 300 GB container disk; every other tier's goldens unchanged. Gates on build pods
  `8gg7uz30x3nesu`, then `dbf6gok3crgj9x` (the coordinator's switch). No GPU, no R2 or volume
  write; the full gold run is the coordinator's (ADR-0019 §16.7).
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): #201 review fixes on
  `claude/lab-r2-weights` (ADR-0019 §16.9): the merged weights root refused on or above the volume
  (exit 2), an 85 min idle limit for `volume+r2` pods, a run deadline in the pod's env and the
  driver's 15 min fetch reserve, `r2-weights.py plan --pin`, partial-fetch clean-up, python3 by a capped,
  retried apt install (the runtime image stays Python-free). Build pod `dbf6gok3crgj9x`. No GPU, no R2 or volume write.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): gold smoke fix on `claude/gold-cells-techniques`
  (ADR-0023 §3): the gold smoke on `sha-f93a961` failed `h3-larry-turbo` and `sol-h3` at once (exit 2, the
  engine refused fv-gpucheck's empty start-up profile); the generated gold cells now pass the family entry's
  `--techniques <engine.profile>` before `engine` (14 of 32 arms), `ProcessPlan::check_installed` is the engine's
  check and a CPU test runs it over every gold cell; `gold-dry-run.txt` regenerated. Build pods `dbf6gok3crgj9x`
  then `dzid0vrox9rjoa`. No GPU, no R2 or volume write; the smoke rerun is the coordinator's.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1 retirements after the coordinator's GPU checks on
  `f93a961` (image `sha-f93a961`), on `claude/d1-retire-det-oracle-graphs` (draft PR). `h3-graphs` (ADR-0019 §14.4,
  pod `8zx5w6h0766g35`): the check prints `PASS`, the graphs arm replayed 4 of 4 steps on three requests; h3-turbo
  768p denoise sync 17.98 s, lazy 17.88 s (−0.56 %), graphs 17.93 s (−0.30 %), graph pool high water 10.34 GiB,
  recorded in `docs/perf/h3-denoise-graphs.md` §5 against its expectation (lazy ~1.4 %, a replay 0.6-0.7 % more:
  both about a third to 40 % of it, every bit unchanged); `h3-graphs-{validate,pod}.sh` and `scripts/gpu/experiments/`
  deleted, `self-delete.test.sh`'s launcher floor 3 → 2. `det-short` (fv-lab pod `s9zflnm1e3nszd`) and `determinism`
  (fv-lab pod `r7z1ddrfki2dvv`): `fv-lab-compare-runs.sh --expect-verdicts 5` / `3` exit 0 (3 / 2 arms the same, 1
  control each, 5 / 3 verdict files), `fv-lab gate --check` 0 differences on all four runs, all `matrix_exit=0`;
  retired (§15.10: `FV_LAB_RETIRED`, branches deleted with `determinism`'s oracle prelude, the 30 other families' dry
  runs unchanged but for error-message line numbers / temporary paths and `writer`'s `dd` timings, fixtures frozen,
  `FV_DET_*` into `FV_LAB_NOT_READ`, `FV_ORACLE_URL` refused for `determinism`, `tests/repeats_oracle.rs` on a frozen
  copy at `0f63c56b`). `oracle` (`fasth3-4step-vsa`, fv-lab pod `70egna7udfp0fr`, reference
  `upstream/55fa19a-10260558` with `FV_MIN_RAM_GB=192`): compare exit 0, 5 arms the same; its branch kept whole
  (the agent's decision under §15.8, accepted by the coordinator, §15.10). Infrastructure retries not committed: det-short's first
  pods (fv-lab `t37wlzbywqp3re`, script `0uwxdgiddf6fn2`) died during `h3-r3` (host failure suspected; the fv-lab
  stub `det-short/f93a961-1026043809-0c44`); the first oracle attempt's runtime directories
  (`oracle/f93a961-10260544`, `oracle/f93a961-1026054453-3225`) fetched no reference after the upstream worker was
  OOM-killed (#205). Follow-ups recorded in the D1 row: the runtime image job and the previous tools release's
  fv-lab; the gold cells' technique bug (fixed by #203); the gold `ref2v` / `a2v` arms loading 4-5 min before
  `negotiate` refuses t2v. Gates on build pod `keragz68b14ees`. No GPU cost in this PR (the
  checks' pods were the coordinator's).
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): review fixes on #207 after merging main (`2390a4d`, #206):
  `fv-lab run determinism` refuses `FV_ORACLE_URL` (it was dropped), `runpod-http.sh`'s fv-lab route refuses an
  `FV_EXTRA_ENV` with a newline (later entries were dropped), the experiment goldens read `prompts-hd5.json` from the
  deleted pod script's commit, `control.yml`'s dead `scripts/gpu/experiments/**` filters dropped, lazy vs sync −0.56 %,
  the frozen dry-run test also compares the gates. Build pod `keragz68b14ees`. No GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): the full gold run's results, on `claude/gold-run-7e580e5`
  (draft PR; ADR-0019 §16.7 "Result"). The coordinator ran the gold tier at `7e580e5` in two pods with the 192 GB
  host-RAM floor (Runpod gave 283 GB each): the 21 R2 arms (`7e580e5-1026090916-e804`, 18 trees, 530.74 GB fetched,
  all exit 0) and the 11 volume arms (`7e580e5-1026091001-7421`, 7 exit 0, 4 t2v refusals as ADR-0023 §3 expects),
  `oom_kills` 0 everywhere, about $3.74. Committed: both run directories (no frames), `docs/perf/tiers.md` from them
  (28 measured, 4 failed), `[last_validated]` and a run `[[source]]` in the 28 measured gold files (the 10 without a
  latency measurement take the run's serve job, `load_s` only where no earlier arm of the pod had read the trees (3);
  the 18 with another kind keep their numbers and close no gap), the 10 regenerated gold cells (note only). The `e9b9435` attempt (NVFP4 skips, causal progress,
  the 14B plug's exit 137; fixed by #208, #209, #210) is described there and not committed, as earlier failed
  attempts were not. Follow-ups in the D3 row: the 14B plug's 156,004 MiB host peak against the 192 GB floor (raising the floor
  is a prerequisite of the next gold run); negotiate before load. Review of #211 applied: warm `load_s` and
  other-kind closures reverted to gaps, the floor "may not hold", what `[last_validated]` means. No Rust change; no GPU cost in this PR (the pods were the coordinator's).
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1f retirements after the coordinator's GPU checks on
  `762ce4d` (image `sha-762ce4d`, tools-v0.1.70, fv-lab 0.1.70), on `claude/d1f-retire-fuse-attn3-ltx` (draft PR;
  ADR-0019 §17.5 "Results", §17.6). Six RTX PRO 6000 pods in EU-RO-1 on `bz0ofkfx45`, 12:40-12:51Z, all deleted;
  the `ltx` pair retried once for capacity; about $2.2 ($23.11 to $20.90). `fuse` (fv-lab `ktao6reo6e4p1e`, script
  `l83gwsp4q0vrt3`), `attn3` (`2euds1fn4gfyt8`, `vir27fw4n6g0vp`) and `ltx` (`zsz3k16tggu58v`, `yis61qgkwwo71c`):
  `fv-lab-compare-runs.sh` exit 0 with §17.5's expected lines, every `matrix_exit=0`. The `fuse` VSA gate reads `fail`
  on both sides for its promotion bar (fusions 1.08× on the warm denoise, under 1.10) and the missing LPIPS, with
  its off-identity check passed: the fusion knob's promotion decision, not a parity verdict, and not a finding
  against the retirement. Retired: `FV_LAB_RETIRED`, the three branches
  deleted (the 27 other families' dry runs unchanged but for `writer`'s `dd` timings), fixtures frozen,
  `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` from `SCRIPT_KNOBS` into `MODE_KNOBS` and `FV_LAB_NOT_READ`, refused in
  `FV_EXTRA_ENV` and the host environment; `runpod.sh gpu` / `matrix` start a retired family's tier with the
  image's fv-lab instead of `runpod-matrix.sh ltx` and stop before any pod on a family nothing can run
  (`runpod.sh plan`, `runpod-plan.test.sh`); `h3` and the B4 families unchanged. Tests on build pod `keragz68b14ees`.
  No GPU cost in this PR (the checks' pods were the coordinator's).
- 2026-10-26, agent (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1g on `claude/d1g-retire-finished` (draft PR; ADR-0019
  §18, coordinator decision 2026-10-26): `b4b`, `b4c`, `b4c2` (§17.3 option (b)), `h3` (§17.4, no
  `sol-h3-spark-no-refiner` arm) and §14.7's `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`,
  `identity`, `h3attn` retired from `runpod-matrix.sh` with no tier. `h3attn` by §14.7's rule: `docs/techniques.md`
  lists no open attention arm (its INT8 `Q K^T` next step closed with SageAttention, Phase 4). A new list
  `FV_SCRIPT_RETIRED` (`fv-lab-retired.sh`, with a pointer per family) because `FV_LAB_RETIRED` routes to an fv-lab
  tier; `runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan` and `runpod.sh gpu` / `matrix` / `plan` / `b200`
  refuse them (exit 2). The eleven branches deleted (907 lines); the 16 remaining families' `FV_DRY_RUN=1` output
  and `FV_DRY_RUN_GATES` records byte-identical before and after. `b4-verdict.sh` and `tests/b4-verdict.test.sh`
  deleted (their readers use frozen copies). `runpod.sh`: `gpu` starts `ltx`, `matrix` runs `ltx wan`, `cmd_b200`
  deleted. No knob moved, no fixture (none was a tier). `writer`: no result in `docs/perf` and no run committed
  (recorded in §18, open). Tests on build pod `14fr0ls707hywf`; no GPU.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): D1h on `claude/d1h-oracle-driver` (draft PR; ADR-0019
  §19): `fv-lab oracle`, the driver of an oracle run (§15.6): `lab/upstream.toml` (the `fastvideo` and `sol-ltx25`
  reference stacks and each target's weight steps, schema `upstream.v1.json`, checked against the oracle cells),
  the upstream pods on fv-deploy's guards (price cap, $8 floor, EU volume, backstop and own cap watchdog, 600 s
  readiness, `minRAMPerGPU` 192 for the FastVideo stack), the wait for each reference's `ORACLE_DONE` with the
  keepalive, the runtime side as `fv-lab pod up --wait` with `FV_ORACLE_URL`, the release (tree fetch, then delete)
  on the runtime pod's `ORACLE_REFS_DONE` (or 1,800 s after it is ready without it), on a 2,100 s stock wait, on any failure or interrupt; a combined balance
  guard over every pod of the run; `--url-file` / `--hold-upstream` so the GPU check's script side reads the same
  reference. Payload goldens from the new `runpod-http.sh plan-upstream` (`fixtures/runpod-http-upstream-*.json`;
  intended differences: the cap watchdog, `env {}`, the digest at create). `tests/oracle_driver.rs` (21 on fakes),
  `tests/oracle_live.rs` (7: the live world on a fake provider, HTTP and executable),
  `tests/fv_lab_bin.rs` (the plan without a key). Not in it: the clip-metric targets and `sfwan13` (a derived-clip step
  to decide first, §19.9). Nothing retired; `oracle.sh` and the branch unchanged. The independent review's fixes are in:
  the image ships `lab/upstream.toml`; a pod whose records fail after its create is deleted (`pod::or_delete`, here
  and in `pod up`); stop, release, then wait on an interrupt or the stock wait; a failed runtime side's pod deleted;
  the references' deadline from the first create; eight mutations each caught. Merged with main after #215. Tests on
  build pod `14fr0ls707hywf`; no GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), D1i agent: the coordinator's decisions on ADR-0019 §19.9 on
  `claude/d1i-oracle-metrics` (draft PR; ADR-0019 §20, FVID-2026-10-26-d1i-oracle-metrics). A typed `[[metric]]` list
  in the cell schema (v1, additive: a pass over clips into a report directory, `missing` / `copy` / `echo`, four
  measures composing the script's ffmpeg / ffprobe argv; `cell.v1.json` regenerated); `extra_cells` in the tier schema
  (v1, additive; `tiers.v1.json` regenerated): run when `--cells` names their arms, read by `run`, `pod up`, `oracle`,
  `gate` and `check`; the ten LTX clip-metric targets as the `oracle` tier's extra cells (46 arms) and `sol-ltx25`
  stack targets (`weights:ltx25`, the guided one `weights:ltx25-dev`); the script's dry run lists its metric commands
  (`metric_dry`), its metric pipelines end in `|| true` (a failure no longer ends the family under `set -e`), the
  a2v targets' `-dump-ownaud` is removed; goldens `oracle-extra-*` (the `oracle` tier's own fixtures unchanged);
  `tests/repeats_oracle.rs` runs both runners on fake ffmpeg / ffprobe and five targets (reports and copies byte-identical);
  a split-line race in the log fixed (`raw_line`, one write per line); `fv-lab oracle --fetch-tree`;
  `fv-lab-compare-runs.sh` compares `metrics.txt` (values alone differing reported). `sfwan13` left for a follow-up
  (§20.5: its upstream R2 fetch is not small). Tests on build pod `68kv8dgygusqq0`; no GPU cost.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), D1i agent: #217 review fixes. Blocking: a stale tools
  release's fv-lab is no longer the runtime image's `lab-build` (`scripts/ci/prebuilt.sh contexts --stale`, test in
  `test_tools_release.sh`), so main's image compiles the commit's fv-lab and its smoke `fv-lab check` reads the new
  `lab/` (D1 follow-up (1) done). `fv-lab-compare-runs.sh` reads `metrics.txt` with the run tag normalised and lets
  only ffmpeg SSIM / PSNR values differ (probe lines byte for byte; empty or failed values only when the same), with
  self-tests for each case; ADR §20.5 names the `sfwan13` branch's two dry-run fixes its golden needs first; §20.7
  the build pod stopped first, the image label check and the worst case ($12.50, start at $20.50); the dry run's
  echo guarded; all ten targets on fakes, under different run tags; a bare `,` in `select` refused.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): the coordinator's GPU checks at `664473b` (fv-lab 0.1.74,
  image `sha-664473b`, tools `tools-v0.1.74`) on `claude/oracle-checks-664473b` (draft PR). ADR-0019 §19.7 passed:
  `fv-lab oracle` served `fasth3-4step-vsa` (upstream `g0e1srbtwgor6r`, 192 GB floor, the reference 13 min after its
  create), runtime exit 0, compare exit 0 (5 arms the same), every pod deleted, about $2.63. §20.7 compared nothing:
  the `sol-ltx25` pod's `weights:ltx25` got 401 from the gated `Lightricks/LTX-2.5` (no Hub token: the EU volume has no
  `hf/token` and nothing passed `HF_TOKEN`), both cells exited 2 and wrote `ORACLE_DONE` without a dump, and the driver
  started a runtime pod whose arms skipped on both sides; the balance $26.35 to $25.78. Fixed here: served means
  `ORACLE_DONE` and `oracle-dump.tar` (the listing; missing with the cell's exit otherwise; no runtime pod when nothing
  is served), `hub_token = true` on `sol-ltx25` and `runpod-http.sh upstream` for `weights:ltx25*` pass `HF_TOKEN` as
  the Runpod secret reference `fv_hf_token`, `upstream/pod.sh` takes an unfilled reference as no token and records the
  token's source in `box.txt`. The six run directories, both compare outputs and both reports committed (no secret in
  them). Tests on the build pod; no GPU cost. Owner: create the Runpod secret `fv_hf_token`, then §20.7 re-runs.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): #218 review fixes (independent review, nothing blocking;
  the coordinator accepted ADR-0019 §19 on §19.7's pass). A pod whose run ends in the poll window its reference is
  written has its pending references asked 3 more times, 5 s apart, before they fail (one proxy error on the listing
  no longer drops a finished reference); `pod.sh`'s token block is `hub_token_init` (whitespace dropped, an empty file
  no token, the env's note kept when the file wins) with `scripts/gpu/tests/hub-token.test.sh` in CI; `runpod-http.sh`
  normalises the step list's whitespace and refuses a `UP_HUB_TOKEN` other than 0 / 1; `lab/upstream.toml` refuses
  secret-like env names and a `weights:ltx25*` step without `hub_token`; a malformed `cell.json` is reported as such;
  both checks' `lab.log` and the four pods' ledger lines committed (`oracle-lab-664473b-*.log`,
  `oracle-ledger-664473b.tsv`).
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), D1j agent: `sfwan13` as fv-lab cells on `claude/d1j-sfwan13`
  (draft PR #219; ADR-0019 §21, FVID-2026-10-26-d1j-sfwan13). The tier `oracle-r2` (`weights = "volume+r2"`, 150 GB)
  with `oracle-sfwan13` (ten arms in the script's order), a sibling of the `oracle` family (`FV_LAB_SIBLINGS`, goldens
  from the live branch with `FV_ORACLE_TARGETS=sfwan13`); additive cell schema (`decode`, optional `report`,
  `{ref_dir}`, `[[compare]] tag`, `after`); `lab/upstream.toml` targets' `r2` and `container_disk_gb` (`sfwan13` on
  the `fastvideo` stack, 180 GB, `pod.sh` `r2:<tree>` with the key to its plan step alone); the script's `sfwan13`
  dry-run fixes (the other 15 families' dry runs byte-identical). The coordinator then chose §21.8 option A:
  `runpod-http.sh run` `FV_R2_TREES` (fv-lab's merged root on the script runner's pod through `scripts/gpu/r2-stage.sh`,
  the key by secret reference, unset a no-op), golden `runpod-http-oracle-sfwan13-r2.json`, `runpod-r2.test.sh` in CI.
  Tests on the build pod; no GPU cost. Next: §21.7's GPU check (the coordinator's), then `oracle` retires (§21.9).

- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): the coordinator's GPU check ADR-0019 §21.7 (`sfwan13` as
  `oracle-r2`) at main `4b4980b` (image `sha-4b4980b`, tools `tools-v0.1.76`, fv-lab 0.1.76 built from `4b4980b`)
  passed on every criterion; evidence and results on `claude/oracle-r2-check-4b4980b` (draft PR). `fv-lab oracle`
  served `sfwan13` (upstream `beelfsmgjfn15i`: `r2_key=env`, `r2:sfwan21-1.3b` 0, the reference 44 s, peak RAM
  12,174 MiB, 170 MB; released and deleted), runtime exit 0 on both sides (`matrix_exit=0`), compare exit 0 with
  §21.7's expected line (10 arms the same; `oracle-sfwan13`, `-f32`, `-legacy`, `-composed` 81 frames byte-identical;
  the three directory compares `pass` on both sides). The R2 tree on each pod: upstream 29.7 s, fv-lab 27.4 s (1057
  MB/s), the script side 40.6 s (712 MB/s). No key value or presigned URL in any fetched or committed file. The
  balance $24.27 to $23.41 (about $0.86). §21 accepted. `oracle` still does not retire: §20.7's clip-metric re-run
  waits for the owner's Runpod secret `fv_hf_token`, and the retirement waits for every cell family's check (ADR-0019
  §21.9). Small fix: `r2-stage.sh` writes apt's output to `r2-apt.log` (57 lines of dpkg progress were in
  `r2-stage.log`), `runpod-r2.test.sh` extended; the fix needs no GPU run; no Rust change.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), S1 agent: ADR-0019 §22's slice S1 on
  `claude/d1-s1-retire-nine` (draft PR #222, from `4f8822a0`; FVID-2026-10-26-d1-remaining). `sfquality`, `headline`,
  `speechtest`, `cold`, `eval`, `ltxoffload`, `solbench`, `sol-lingbot` and `sol-cosmos3` retired from
  `runpod-matrix.sh` with no tier, §18's way: `FV_SCRIPT_RETIRED` (twenty families) with a pointer each (what replaces
  it, what is lost, `git show 4f8822a0:scripts/gpu/runpod-matrix.sh`); every entry point refuses them (exit 2, before
  a key or a pod) through the existing checks. The nine branches deleted (495 lines; §22.6 said 567) and `eval`'s
  three default lines; no shared helper had no other use. Dry runs of the seven remaining families (and `oracle` with
  `FV_ORACLE_TARGETS=sfwan13`, its tensor-diff and clip-metric settings, `wan` / `sfwan` with `FV_ORACLE_URL`): 12
  runs byte-identical before and after (output, gate records, exit). No knob moved (none was a tier). Tests name the
  twenty families; mutations caught. Stale docs updated (§22.8 lists them); `sol-bench-pod.sh` untouched. Tests on the
  build pod; no GPU.
- 2026-10-26, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), S2 agent: ADR-0019 §22's slice S2 on `claude/d1-s2-hd`
  (draft PR #223, from `4f8822a0`, main with S1 merged in; FVID-2026-10-26-d1-s2-hd). `hd` as an fv-lab tier
  (`hd-canvases`, `hd-10s`: the branch's eight arms, goldens from the live branch) and the 1080P serve gate's
  `FV_PROMPTS=…/prompts-hd5.json` as the tier `hd5` (a prompt-set cell plus `hd-10s`, a sibling of the `hd` family,
  §22.9: a tier keeps the arm names the hd reports read). In lockstep the branch loses its derived step (the Lanczos
  upscale and compares, the `FV_HD_POST_URL` post script, the keyframe prune); `FV_HD_POST_URL` /
  `FV_HD_POST_TIMEOUT_S` refused by `runpod-http.sh` (both paths), fv-lab and `runpod-matrix.sh`. Dry runs of every
  other family byte-identical before and after; `hd` lost only its eight empty dry-run timing lines. GPU check §22.10
  (`turbo-768p turbo-1080p max-1080p`, ~$2.00) the coordinator's. Tests on the build pod.
- 2026-10-27, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ): the coordinator's GPU check ADR-0019 §22.10 (S2, `hd`:
  `turbo-768p turbo-1080p max-1080p`) at main `d53ea62` (after #223; image `sha-d53ea62`, tools `tools-v0.1.78`,
  built at `d53ea62`; fv-lab 0.1.78) passed; evidence and the result on `claude/hd-check-d53ea62` (draft PR). Both
  sides in parallel, 00:34Z to 00:51Z, an RTX PRO 6000 each in EU-RO-1 on `bz0ofkfx45`: fv-lab `pod up` (pod
  `frtucp564l3vsi`, ready 22.9 s, deleted `done`; `artifacts/runpod/hd/d53ea62-1027003410-8701`) and `runpod-http.sh
  run` (pod `urtrmxowjpl29h`; `artifacts/runpod/hd/d53ea62-10270034`), both `matrix_exit=0`, each `weights.log`
  `exit=1` (`h3-8step` incomplete, as expected). `fv-lab-compare-runs.sh` exit 0 with §22.10's expected line (3 arms
  the same, 372 frames each byte-identical; `compare-d53ea62.txt`). `benchmark.json` `total_s` fv-lab / script:
  `turbo-768p` 37.52 / 36.51 s, `turbo-1080p` 54.89 / 54.53 s, `max-1080p` 73.24 / 72.76 s. A first attempt (about
  00:31Z) was aborted when a container restart killed both local drivers about 2 min after the creates; both pods
  were deleted by hand. The balance $22.64 to $21.44 (about $1.20, that attempt included; estimate ~$2.00). S2 is
  done GPU-side; S7 retires `hd` (and freezes `hd5`). No code change; no Rust compile.
- 2026-10-27, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), S3 agent: ADR-0019 §22's slice S3 on
  `claude/d1-s3-serve-engine` (from `d53ea62e`, main with S2 merged; FVID-2026-10-27-d1-s3-serve-engine).
  `serve-engine` as an fv-lab tier (`serve-engine-h3-turbo`, `-ltx-turbo`, `-fastwan21-1.3b`: the branch's six arms
  in its order, goldens from the live branch; `weights = "volume+r2"`, 150 GB, since `fastwan21-1.3b` is R2-only). In
  the same commit the branch's three engine arms lose `--tae-dir "$TAE"` (`fv-gpucheck engine` then takes
  `<weights root>/auxiliary/tae`, `$TAE` only when the volume's TAE copy is complete; none of the three recipes reads
  a tiny decoder, so the frames do not change); `wan`'s `engine-fw22` keeps it (S6). `FV_ENGINE_CELLS`
  refused by fv-lab (`SCRIPT_KNOBS`; select by arm). A golden of the GPU check's script-side body
  (`FV_R2_TREES=fastwan21-1.3b`). Dry runs of every other family byte-identical before and after; `serve-engine`'s
  differ only by the three dropped `--tae-dir` arguments. GPU check §22.12 (all six arms, ~$1.35-$2.25) the
  coordinator's. Tests on the build pod.
- 2026-10-27, Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), S4 agent: ADR-0019 §22's slice S4 on
  `claude/d1-s4-sfstream` (from `3bc6d033`, main with S3 merged; FVID-2026-10-27-d1-s4-sfstream). §22.4's
  Addition 16: `plan.rs`'s refusal of `{tae}` without `tae` became a tier check (allowed in a gated arm, in a tier
  with `tae = true`; such an arm is not TAE-gated and gates on its trees only), cell v1 additive, every existing cell
  and golden unchanged. `sfstream` as an fv-lab tier (`sfstream-rollout`: `sfstream-main`, `sfstream-10min`, goldens
  from the live branch; `weights = "volume+r2"`, 150 GB, since `sfwan21-1.3b` is R2-only). In the same commit the
  branch's `export FASTVIDEO_TAE_DIR="$TAE"` became each arm's `env FASTVIDEO_TAE_DIR="$TAE"` (the same value reaches
  the same two commands), and its taew2_1 fetch is skipped in a dry run (it had downloaded into the dry run's TAE
  directory). `FV_SFSTREAM_RUNS`, `FV_SF_PROMPT`, `FV_SF_SWITCH` refused by fv-lab (`SCRIPT_KNOBS`). A golden of the
  GPU check's script-side body (`FV_R2_TREES=sfwan21-1.3b`). Dry runs of every other family byte-identical before
  and after; `sfstream`'s differ only by the env word in each arm. GPU check §22.14 (`sfstream-main`) the
  coordinator's. Tests on the build pod.
