# fastvideo-rs: architecture and implementation review, with a proposal

Date: 2026-10-22. Scope: every element of the tree at `fb1bce3` (the
inference crates, the serving stack, the API adapters, deployment and the
control plane, the scripts, the validation ladder, CI and the docs). Method:
a read-only survey of the sources, five parallel deep dives (one per layer),
and counts taken with `find`, `wc` and `grep`; nothing was compiled or run on
a GPU, and every GPU number quoted below is a number the repo itself
records. Section 9 lists what was not verified.

The owner's questions, answered in order: strengths and weaknesses (§3, §4),
opportunities for performance, less complexity, more modularity (§5), what a
model-family deployment needs today (§6), fewer scripts and first-class
cross-provider deployment (§7.3, §7.4), API parity and parameter
certification (§7.6), a gold standard and cost / latency / quality tiers
(§7.5), and the proposal itself (§7, §8).

## 1. Summary

The system works and is unusually well measured: a Rust port of a dozen
video, audio and image diffusion families with its own CUDA kernels, a
serving stack that is wire-compatible with four public APIs, a Cloudflare
edge, a Runpod cluster controller, and a validation ladder that compares
against upstream with injected noise and an f32 noise-floor control.
Decisions are logged with numbers. That is the strength, and it is real.

The weakness is structural and it is the same weakness in every layer:
**the knowledge of what a model is, what it needs and what it is allowed to
do lives in code and prose, in several languages, instead of in one piece
of data.** A single served model id appears in about 45 non-doc source files
across Rust, TypeScript, bash, Python, TOML and TSV; the weights tree name
`h3-base` appears in 113. The engine's capability catalog is a 3,051-line
Rust file. The model-to-(weights, GPU, image, config) mapping is encoded in
at least 14 places. The boot sequence of a worker exists in 10 variants in
three languages. 233 `FASTVIDEO_*` environment names are read through two
mechanisms plus 111 direct reads; the scope doc lists 54 of them. 240 files
under `scripts/` (50k lines) hold two generations of GPU harness, 24
one-off port experiments that each reimplement pod lifecycle, and the only
record of most performance and quality results is prose copied into 22 docs
and into TOML comments.

None of that is a defect of design intent. The serve design doc and the
techniques layer both show the right instincts (pure protocol core, fake
engine in CI, typed technique composition with exclusive seams). The gap is
that those instincts stopped at the layer boundary each time, so the same
fact is restated at every boundary and nothing checks the restatements
against each other except a growing set of "keep X equal to Y" unit tests.

The proposal (§7) is therefore one idea applied five times: **make the model
family manifest the single source of truth, and make every other layer a
projection of it** (the engine catalog, the API parameter schemas, the
deployment presets, the weights manifests, the validation cells and the
gold standards). Around that: extract the runtime out of `cudarc/src/wan`,
replace the per-family copy-paste in dispatch with one trait, put the ten
boot scripts into one Rust `fv-boot`, put the provider logic into one Rust
`fv-deploy` with a `Provider` trait, and put the GPU harness into one Rust
`fv-lab` that reads checked-in tier, cell and gold-standard data and emits
the tier table. Script count goes from 240 files to roughly 95; shell from
about 27k lines to about 8k.

Headline numbers:

| Element | Size | Note |
|---|---|---|
| Rust workspace | 26 crates, 340k lines of `.rs` (+1 excluded crate) | `fastvideo-cudarc` 134.9k incl. 11.1k lines of CUDA (216 kernels) |
| Control plane | 28.9k lines TS (`control/src`), 7.2k (`control/ui`), 2.5k (`control/public`) | plus 742 lines in `app/` |
| Scripts | 240 files, 213 code (126 sh, 83 py, 4 mjs), 50.4k lines | 24 port one-offs, 6 legacy Vast scripts, 2 harness generations |
| Configs | 35 serve TOMLs, 43 technique profiles, 1 autoscale TOML | 65 distinct serve keys; 18 identical in all 35 files |
| Docs | 127 Markdown files, 48.4k lines | `design.md` 2,395 lines; `decision-log.md` 134 KB |
| CI | 9 workflows; 2 run on pull requests | no workflow touches a GPU |
| Model identity | one id in ~45 source files; ~14 places map model to artifacts | §4.1 |
| Configuration | 233 `FASTVIDEO_*` names, 174 `FV_*` names in crates | 2 readers + 111 direct `env::var` reads |

## 2. System map

```
                    clients (fal / MiniMax / LTX / OpenAI SDKs, browsers, Reactor)
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             │ Cloudflare: edge Worker + per-family Durable Object     │  crates/fastvideo-edge (wasm)
             │ (fastvideo-dispatch-proto: pure scheduler, shared)     │  crates/fastvideo-dispatch-proto
             └───────────────────────────┬───────────────────────────┘
                                         │ WebSocket job dispatch
  fv-control (TS Worker, Hono) ──launch/roll/stop──► GPU workers: fv-serve (one process, one GPU)
  control/src, control/ui                            ├─ adapters: openai-videos, minimax, ltxapi, fal, reactor
  D1 + R2 + Analytics Engine                         ├─ serve-kit (axum glue, stores, ingestion, callbacks)
  providers: Runpod (pods, serverless), GMI, Brev    ├─ protocol (normalized request, caps, negotiate, errors)
  observes CloudRift; GCP/Vast are bash only         ├─ engine-service (scheduler, executor thread, caps catalog)
                                                     │     └─ cuda/ (feature): H3Model, Ltx2Model, WanModel
                                                     └─ media / webrtc / trace / upsampler
                                                               │
                                         fastvideo-cudarc: pipelines + kernels (wan/ = the runtime)
                                         fastvideo-models (host math, techniques), loader, ops, core (CLI registry)
                                                               │
                      weights: R2 `fv-weights` (57 trees) / EU volume (13 trees) / Hub; 4 TSV manifests
```

Two control paths exist side by side: fv-control (TypeScript, the intended
one) and the bash deploy scripts (`scripts/serve/runpod-*.sh`, `vast.sh`,
`cloudrift-worker.sh`, `scripts/gcp/*`), which disagree with fv-control on
who runs the entrypoint, which weights sources exist, and where the deploy
ledger is (§4.3).

## 3. Strengths

Listed because the proposal must preserve them.

1. **Numerics and parity discipline.** NVRTC options mirror PyTorch
   (`crates/fastvideo-cudarc/src/wan/kernels.rs:245-255`); determinism was
   engineered, not assumed (`docs/perf/determinism.md`, fixed attention rule
   in `sdpa_rule.rs`, ordered reductions, byte-identity cells); the oracle
   injects the reference's noise and conditioning and judges against an f32
   control arm (`docs/oracle.md`); safetensors-header manifest tests catch a
   wrong key or shape without hardware (`h3/manifest_tests.rs`).
2. **Memory planning as a first-class object.** Per-model memory planners
   (`crates/fastvideo-models/src/h3/memory.rs`, `ltx2/memory.rs`), an
   allocation ledger, pinned-host block streaming through device slots,
   `FASTVIDEO_DEVICE_BUDGET_GIB` to plan for a smaller card.
3. **The techniques layer** (`crates/fastvideo-models/src/techniques/`,
   `docs/techniques.md`): typed techniques with phases, exclusive seams,
   conflict checks, unknown keys as errors, a logged resolved plan, and the
   off-identity invariant. This is the right shape for every other knob in
   the system.
4. **A clean CPU/GPU split in serving.** Only `fastvideo-engine-service`
   sees CUDA, behind one feature; the whole HTTP stack, including ten real
   client SDK suites over TLS, runs on `FakeBackend` in CI
   (`.github/workflows/serve-compat.yml`, `tests/compat/`). The protocol
   core is framework-free, so adapter methods are pure and golden-testable
   (`crates/fastvideo-protocol/src/http.rs`).
5. **Pure schedulers.** `engine-service/src/scheduler.rs` and
   `dispatch-proto/src/sched.rs` have no runtime dependency and are shared
   between wasm and native; the dimension sweep
   (`crates/fastvideo-serve/tests/dimension_sweep.rs`) proves "accepted at
   submit implies the engine accepts" with the real CUDA catalog on the
   fake backend.
6. **Observability that does not perturb.** `fastvideo-trace`: opt-in,
   non-blocking recorder, W3C `traceparent`, CUDA-event GPU timelines,
   browser and edge beacons, measured overhead within noise.
7. **Cost discipline everywhere.** Every renting path has a price cap, a
   wall-clock backstop, a reaper and a balance floor; costs are quoted per
   tier (`scripts/gpu/validate.sh`, `Taskfile.yml`, fv-control policies).
8. **Release identity.** Digest-pinned images, explicit `stable` promotion
   with a D1 history and an undo-stack rollback (`docs/serve/releases.md`).
9. **Decision hygiene.** `decision-log.md`, `docs/ops/open-items.md` and
   `docs/gaps/` record who decided what, with the measurement that
   motivated it.
10. **The best single test in the repo** is the LTX model matrix: more
    than 5,000 model x resolution x fps x duration combinations against an
    independently written `spec_allows`
    (`crates/fastvideo-ltxapi/tests/golden.rs:230-257`). It is the template
    for §7.6.

## 4. Weaknesses

### 4.1 Model identity is scattered (the root cause)

A model, to this system, is: a Hub id, a CLI preset, an engine catalog row,
a recipe string, a technique profile, a serve `[[models]]` entry with
aliases, a served name per API, a fal app id, a MiniMax model name, an LTX
model id, a weights tree (or several), an R2 prefix, a volume tree, a
Docker image variant, a control preset, a GPU requirement, a validation
cell, a gold number in a doc. Each of those is defined where it is used:

| Where | What it defines | Evidence |
|---|---|---|
| `crates/fastvideo-core/src/registry.rs` | 21 `ModelFamily` variants, 21 separate static tables, a 22-branch `resolve()`, 83 Hub ids | `:287-309`, `:453-870`, `:893-969` |
| `crates/fastvideo-core/src/av_generate.rs` | 20 copy-pasted `generate_<family>` functions; `AvGenerateOptions` with 35 fields, several family-specific | `:135-1700` |
| `crates/fastvideo-engine-service/src/cuda/caps.rs` | the served catalog: 3,051 lines, `catalog()` is ~370 lines of constructor calls, recipe names resolved by a string `match` | `:1221-1592`, `:1625-1650` |
| `crates/fastvideo-engine-service/src/cuda/validate.rs`, `crates/fastvideo-protocol/src/caps.rs` | per-family limits, again | |
| `crates/fastvideo-{openai-videos,minimax,ltxapi,fal}` | per-surface copies of limits: LTX matrix in `ltxapi/src/models.rs:21-26` and `fal/src/schema/ltx.rs:44-51`; H3 duration in three crates; aspect enums in three places; knob ranges that differ by surface | §4.4 |
| `configs/serve/*.toml` (35) | `[[models]]` rows and an `[aliases]` matrix repeated in 22 files; `[protocols]` identical in all 35 | |
| `profiles/**/*.toml` (43) | technique switches per recipe, measured numbers in comments | |
| `control/src/{presets,enums,recipe-trees,r2-trees,weights-sources,volumes}.ts`, `cluster/{catalog.json,worker-configs.ts,image-configs.ts}` | presets, variants, tree lists, generated copies of the TOMLs | |
| `scripts/serve/variants.sh`, `scripts/gpu/volumes.sh`, `scripts/gcp/families.tsv`, `configs/autoscale.toml` | hand-kept bash and TSV copies of the same lists | kept equal by unit tests, e.g. `serverless-presets.test.ts` |
| `scripts/gpu/weights-{manifest,r2,revisions,sha256}.tsv`, `verify-weights.sh` | trees, revisions, hashes, verify cells | |
| `docker/gpucheck.Dockerfile:494-523`, `docs/serve/images.md` | image variants, each baking one config | |

Consequences: adding a variant means editing six to nine of these; a
recipe string typo is a runtime error; the control plane's `fv-entry.sh`
preflight cannot see trees the engine finds beside `weights`, so
`control/src/recipe-trees.ts` exists to restate them; GPU requirements are
prose plus one `min_vram_gb` (NVFP4-only variants "fail to load" on a
non-Blackwell card, `control/src/presets.ts:118`); `configs/autoscale.toml`
pins H100 while fv-control defaults to RTX PRO 6000.

### 4.2 Inference core

- **`cudarc/src/wan` is the runtime, misnamed as a family.** All 24 other
  family directories import `crate::wan::` (log 231 sites, tensor 145,
  weights 117, dump 96, nn 83, device 82, envflag 67, ops 58). The leak runs
  both ways: 23 `wan/*.rs` files name H3 or LTX types
  (`wan/quant.rs:448 H3QuantPlan`, `wan/nvfp4_linear.rs` for the LTX FFN),
  and `wan/kernels.cu` (7,413 lines) holds the `h3`, `ltx2`, `moe` and
  `sol` regions. `gpucheck` and `engine-service` depend on "wan" to reach
  the runtime.
- **No family trait.** `grep 'pub trait .*Pipeline'` finds nothing; 25
  family-specific `Request`/`Options` structs; `SamplingParam` is Wan-only
  and has no `validate`. `H3Pipeline::load` is ~830 lines in one function
  (`h3/pipeline.rs:1248-2074`).
- **Configuration sprawl.** 233 `FASTVIDEO_*` names, two readers
  (`techniques/settings.rs::var` and `wan/envflag.rs` with first-read
  caching) and 111 direct `env::var` reads that bypass both; the caching
  makes settings order-dependent process globals (`gpucheck/src/mode.rs`
  must run "at process start"). `techniques::model_spec` covers only H3 and
  LTX-2; Wan reads the techniques layer in 11 files without a spec.
- **Duplication on the hot path.** Seven or more dense-attention backends
  (25 `sdpa*`/`attn*` functions); norms in six places; the block fusion
  written three times (`wan/fuse.rs` 619, `h3/fused16.rs` 1,009,
  `ltx2/fuse.rs` 321); five NVFP4 modules plus the oxide crate for a
  feature that is off; step caches reimplemented per family (EasyCache x3,
  TeaCache x3, FbCache, Taylor), three separate disk prompt caches.
- **Host twins.** 105 `*_device` and 39 `*_host` functions, 804
  `#[cfg(feature = "cuda")]` gates, a dual-storage `CudaTensor`. Every op
  has a CPU path that doubles the test surface and makes a silent host
  fallback possible (gpucheck's `no_host_fallback` group exists to police
  it).
- **Dead or stale.** `fastvideo-ops::TensorBackend` has no user (the only
  impl delegates to the host); ADR-0001 and `docs/scope.md` still describe
  Candle/Luminal as frozen and `unsafe` as cudarc-only, while
  `backend_kind.rs` rejects those names and `loader` and `gpucheck` also
  allow `unsafe`. `fastvideo-mlx` is a scaffold. `nvfp4_gemm.rs` has no
  caller.

### 4.3 Serving stack

- **The capability catalog is code** (§4.1), and the `[[models]]` entry
  does not declare capabilities; it selects a catalog row by a recipe
  alias and overrides it with free-form `extra` strings
  (`crates/fastvideo-serve/src/app.rs:245-270`).
- **One process, one GPU, no fault isolation.** `CudaBackend` is built for
  device 0 only (`app.rs:276-279`); a CUDA abort takes HTTP, sessions and
  the D1 flush with it; the scheduler supports several executors but the
  config cannot reach them.
- **Swap pools thrash.** The family pool reloads a 40-66 GB DiT from disk
  on every model switch; the scheduler is FIFO and residency-unaware
  (`engine-service/src/scheduler.rs::next_for`), so alternating requests
  for two variants pay the swap each time (about 75 s per switch on an
  RTX PRO 6000 per `docs/ops/open-items.md`).
- **No per-job wall-clock limit in the engine**; a hung kernel is caught by
  the D1 stale heartbeat (900 s) or by shutdown.
- **Residue of retired designs.** `[gateway]` and `gw_workers` survive
  under the retired gateway's name; `multiworker.rs` (378 lines) serves the
  Runpod load-balancer mode the edge superseded; `edge_host.rs` (1,339
  lines) reimplements the Worker natively for tests; `config.rs` is 1,969
  lines with a hand-written env overlay for ~76 keys.
- **Observability gaps.** The `otel` feature is empty; there are no
  per-model latency histograms, no load/swap/residency metrics; the edge
  path trace was never run end to end.

### 4.4 API surfaces

- **No machine-readable upstream spec is in the repo.** The fal, LTX,
  MiniMax, OpenAI and director contracts are prose snapshots from
  2026-09-27/28 in `docs/serve/research-*.md`; there is no drift check. The
  one self-authored schema (`/fal/schema`, JSON Schema 2020-12) has a
  schema-vs-parser agreement test; nothing equivalent exists for the other
  four surfaces.
- **Three validation styles coexist**: serde strict with
  `deny_unknown_fields` (OpenAI, native, director), serde permissive with
  raw `Value` (MiniMax, FastWan), hand-rolled from `Value` (LTX, fal). The
  LTX request has no typed struct.
- **Limits are duplicated and inconsistent across surfaces**: OpenAI
  refuses steps outside 1..=200 and guidance outside 0..=20
  (`openai-videos/src/videos.rs:467-480`) while `negotiate` only checks
  `steps > 0`; the H3 5-15 s duration rule is in three crates.
- **Certification is uneven.** About 230 parameter slots. LTX has the
  5,000-combination matrix; OpenAI has invalid cases for ~30 of 42 fields;
  the dimension sweep covers native, fal, director `configure`, MiniMax and
  LTX but not `/v1/videos` or FastWan; streams, WHIP ingest and director
  `Configure` have no per-parameter tests; the compat suites are mostly
  happy-path (2-5 invalid cases each). Goldens are self-generated; only one
  or two fixtures were captured from upstream, and no upstream error bodies.
- **Adapter unit tests are not in GitHub CI**; they run only on the build
  pod through `scripts/serve/check.sh`.

### 4.5 Deployment and control plane

- **Three languages for one concern.** The boot sequence (driver check,
  config materialization, weights, sidecar, tunnel/report, watchdog, exec)
  exists ten times: `deploy/runpod/fv-entry.sh`, three `*_BOOT` strings in
  `control/src/cluster/payloads.ts`, `PROVIDER_BOOT`, `BREV_BOOTSTRAP`,
  `SLS_BOOT`/`SLS_HUB_BOOT`, `scripts/gcp/startup.sh`,
  `scripts/serve/cloudrift-worker.sh`, `crates/fastvideo-deploy/src/vast.rs`.
  Bash embedded in TypeScript strings is untestable shell. Runpod pod
  creation is written four times (TS, two bash stacks, Rust). The
  serverless queue worker exists in Rust and in bash.
- **Scripts and fv-control disagree.** fv-control injects its own boot
  script and bypasses the image entrypoint; the scripts let the entrypoint
  run. The scripts only know volume weights; fv-control knows
  volume/r2/hub/none. Deploy ledgers are split between D1 `deployments`
  (scripts) and `cluster_pods` (fv-control).
- **Provider coverage is uneven by construction.** Runpod is everywhere;
  GMI/Brev exist only in TS (simulated); GCP exists only in bash (never run
  live); Vast exists in Rust glue, bash and six Candle-era legacy scripts;
  CloudRift is bash plus observe-only TS, on hold. There is no single
  interface a new provider implements.
- **`fastvideo-autoscale` is orphaned.** No crate depends on it; its host
  was retired; fv-control reimplements its serverless scaling. It also
  holds the cleanest `Provider` trait in the tree
  (`crates/fastvideo-autoscale/src/provider/mod.rs:40-47`), which is the
  seed for §7.4.
- **Serve TOML is already provider-neutral** (the GCP files differ from
  their Runpod twins only in WebRTC ports, chunk seconds and `state_dir`);
  the provider knowledge is all in env and boot scripts. That is good news
  for the proposal.

### 4.6 Scripts, validation and CI

- **Two harness generations plus a legacy one.** The Vast ladder
  (`validate.sh` 1,566 lines, `remote.sh`, `scripts/ui/server.py`, and the
  Taskfile's `kernels|parity|clip|...` tasks) and the Runpod matrix
  (`runpod-matrix.sh` 3,202 lines with 30 families as bash `case` arms,
  `runpod-http.sh`, `runpod.sh`). All work since late September runs on
  Runpod; the README still documents the Candle-era `scripts/vast-*.sh`.
- **Forks instead of parameters.** 24 port-specific files (~3,900 lines):
  each `<port>-validate.sh` reimplements create/price/backstop/poll/fetch/
  delete, each `<port>-pod.sh` reimplements fetch-from-plan, LPIPS, cell
  loop and `status.tsv`.
- **Thresholds and workloads are not data.** Offer floors, $/hr and wall
  caps are bash `case` tables; parity limits are Rust constants; the only
  TOML policy (`scripts/gpu/gate-policy.toml`) is relative (candidate vs
  baseline), and its only hard quality gates are frame count, sharpness
  ratio and jitter ratio. PSNR and LPIPS are telemetry. There is no
  absolute per-model quality bound anywhere.
- **Results are prose.** 22 docs hold numeric LPIPS values, 26 hold
  s/step figures, profiles hold measurements in comments, and nothing joins
  `benchmark.json`, the gate verdict and the pod price. A cost / latency /
  quality tier table cannot be produced without a human.
- **CI proves compile and CPU behaviour only.** No workflow references a
  GPU. The gap between "CI green" and "validated on a GPU" is unbounded and
  unrecorded per commit. `cargo fmt` and `clippy` on the workspace are
  advisory (60 files of drift, 27 lints).
- Unreferenced files: `scripts/ci/test_tools_release.sh`,
  `scripts/gpu/lpips_fixtures.py`, `scripts/gpu/ncii-check.sh`,
  `scripts/gpu/prompts-flux2.json`, `scripts/serve/fv-app.sh`,
  `scripts/gpu/sage/{run,vast-run}.sh`; `validate.sh` carries `h3-*` and
  `ltx2-*` tiers that nothing invokes.

## 5. Opportunities

### 5.1 Performance

Measured by the repo, unexploited or exploitable at low risk:

- **Residency-aware scheduling** in the family pool: batch queued jobs by
  resident model with bounded starvation before swapping. Removes the
  ~75 s swap from alternating traffic; pure scheduler change
  (`engine-service/src/scheduler.rs`).
- **Pre-stage the next model** while a DiT runs: the text encoders are
  already family-cached; the swap could start the next DiT's host staging
  during the current decode.
- **Warm-request latency items already parked** in `docs/ops/open-items.md`:
  upload the MP4 while it encodes (0.7-1.2 s), push completion instead of
  700 ms polling (0.3-0.6 s), take the two ~280 ms D1 round trips off the
  critical path, let a real request preempt background warm-up (4.7 s seen).
- **Host-built per-step AdaLN table (~51 ms per step on H3,
  `docs/perf/h3-denoise-graphs.md`)** moved on-device; and the host
  attention paths still reachable in the sparse routes (`h3/vsa.rs
  attention_host`, `vsa_attention_host`, `sla_attention_host`) retired once
  `no_host_fallback` proves them unreachable.
- **Sequence-parallel gather is host-mediated** (ADR-0003, `wan/sp.rs`);
  multi-GPU per process is unreachable from serve config. Both are the
  same work item: real per-device backends.
- **Cold start** is where the published E2E numbers are lost (H3 cold load
  measured at 5.4-8.3 min before E12/E13, `design.md` §0.4). The lazy Wan
  DiT under R2 streaming halved cold start (106.6 s vs 212.6 s,
  `docs/ops/r2-bench-2026-10-13-wan-lazy.md`); applying the same lazy
  store to H3 and LTX is the obvious next step, and the gold standard
  (§7.5) should record cold start as a first-class number.

### 5.2 Complexity reduction without functional or performance loss

Each item is byte-identical or behaviour-preserving by construction:

- Extract the runtime out of `wan/` (mechanical rename, zero perf change).
- Replace 21 registry tables, the 22-branch `resolve`, the 20
  `generate_<family>` functions and the 25 request structs with one trait
  and one manifest (§7.2).
- One `fuse.rs` parameterized by rounding policy; one `StepCache` trait
  with `EasyCache`/`TeaCache`/`FbCache`/`Taylor` implementations in
  `techniques/`; one `sdpa(q, k, v, Plan)` entry with `sdpa_rule.rs` as the
  only chooser.
- Delete `multiworker.rs`, the `[gateway]`/`gw_workers` names, the dead
  `TensorBackend`, `nvfp4_gemm.rs` and the oxide AOT path until a model
  path needs them (or gate them behind features); delete the six legacy
  Vast scripts, `docker-build-cuda.sh`, `serverless-worker.sh`,
  `upstream_bench.py`, the unreferenced files above.
- Derive `[aliases]`, `served_names` and `[protocols]` from the manifest
  instead of repeating them in 35 TOMLs; derive the serve env overlay from
  the config struct instead of hand-writing 76 branches.
- Collapse the 24 port one-offs and the three pod drivers into one
  parameterized runner (§7.5).

### 5.3 Modularity and extensibility

- **A `VideoFamily` trait plus a declarative family manifest** makes a
  port a directory plus a manifest row instead of nine edits.
- **A `Provider` trait plus a family deployment manifest** makes a provider
  an implementation of six methods instead of a bash stack.
- **An `EngineBackend` factory** (keyed by family, optionally
  out-of-process) gives crash isolation and real multi-GPU without touching
  `app.rs`.
- **A `ParamSpec` per (surface, model) derived from `ModelCaps`** makes a
  new API surface a renderer of one schema instead of a fourth copy of the
  limits.

## 6. What it takes to deploy a model family today

Traced for the H3 family on Runpod, which is the most complete path. Every
line is a separate artifact that someone edits by hand or that is generated
from another hand-edited artifact.

1. **Weights.** 13 trees (`h3-base`, `FastH3-4-step-Preview-v1-LoRA`,
   `h3-ref2va`, `fasth3-trim-8step`, `fasth3-trim-8step-nvfp4`,
   `fasth3-v2-nvfp4`, `fasth3-4step-vsa-nvfp4`, three `h3-turbo-loras/*`,
   two `vdn-h3-*`, `auxiliary/tae`; ~396 GB, 485 GB of disk) published to
   R2 by the fetch scripts, recorded in four TSVs
   (`weights-manifest.tsv`, `weights-r2.tsv`, `weights-revisions.tsv`,
   `weights-sha256.tsv`), verified by cells in `verify-weights.sh`, and
   listed again in `control/src/{weights-sources,r2-trees,recipe-trees,volumes}.ts`
   and `scripts/gpu/volumes.sh`.
2. **Engine catalog.** A `CudaModel` per variant in
   `engine-service/src/cuda/caps.rs::catalog()` (19 for H3), a recipe
   alias in `model_from_config`, per-family limits in `validate.rs`, and a
   technique profile under `profiles/h3/` (37 files for H3).
3. **Serve config.** `configs/serve/runpod-family-h3.toml`: 19 `[[models]]`
   rows, `engine.swap = true`, a 16-row `[aliases]` matrix, `[protocols]`,
   `[gateway] pool = "h3-family"`; the GPU requirement is a comment.
4. **Image.** `docker/gpucheck.Dockerfile` stage `serve-h3-max` bakes that
   TOML (`scripts/serve/variants.sh`, `control/src/enums.ts VARIANTS`,
   generated `image-configs.ts`); CI publishes
   `ghcr.io/zaitrarrio/fastvideo-rs-serve:h3-max-sha-<7>`;
   `release.sh promote` moves `:h3-max-stable` and syncs the Runpod
   templates `fv-serve-h3-max-{sls,pod}`.
5. **Control preset.** `control/src/presets.ts` `h3-family` (union of 15
   member presets, `min_vram_gb: 96`, weights source `r2` because the
   volume lacks the NVFP4, Trim, LoRA and VDN trees), the cluster template
   `h3` in `cluster/spec.ts`, `POOL_PRESET_IDS` in `enums.ts`, the region
   default (RTX PRO 6000, EU-RO-1) in `cluster/regions.ts`.
6. **Boot.** fv-control's `WORKER_BOOT` script (not `fv-entry.sh`): driver
   check, `FV_WORKER_TOML_B64` materialization, `r2-fetch.py` with a
   presigned plan from `weights-plan.ts`, optional upsampler sidecar,
   Cloudflare tunnel, phase reports to `/ingest/v1/endpoint`, `exec
   fv-serve`.
7. **Secrets and env.** Seven `RUNPOD_SECRET_fv_*` mapped to `FV_*`
   (`configs/serve/runpod.toml`), plus `FV_WEIGHTS_SOURCE`, `FV_SERVE_ROLE`,
   `FV_DISPATCH_*`, and the engine's `FASTVIDEO_*` process plan, which
   every model in the process must agree on (`ProcessPlan::for_models`).
8. **Edge.** A family Durable Object pool name matching `gateway.pool`,
   API keys and the served names the edge routes on.
9. **Validation.** `task rtx6000` or `runpod-http.sh run <sha>` by hand;
   the `rtx6000` family in `runpod-matrix.sh`; `family-pool-validate.sh`
   for the swap timing; numbers transcribed into `docs/serve/e2e/*.md`.
10. **Docs.** `docs/ports/h3.md` (1,414 lines), `docs/scope.md`,
    `docs/serve/images.md`, `docs/control/README.md`.

For LTX the list is the same with 4 models, 3 trees (~165 GB) on the EU
volume, the `ltx` variant and preset, `min_vram_gb: 80`. For GCP, only
`h3-turbo`, `h3-max`, `ltx`, `wan` and `fake` exist (`scripts/gcp/families.tsv`)
and the path has never run live. For Vast, GMI and Brev there is no family
deployment at all, only single-config launches.

The owner's question has a short answer: **what a family needs is a
manifest of about 15 fields (members, recipes, trees, image, GPU class and
features, disk, residency plan, served names, tiers, gold cells, licence),
and today those 15 fields are spread over roughly 14 files in five
formats.**

## 7. Proposal

### 7.1 Principles

1. **One source of truth per fact, and that source is data.** Code
   projects the data; prose cites it. Everything that is a copy today
   becomes generated or deleted.
2. **Rust for anything that runs on a pod, in CI, or that a provider
   calls; TypeScript only where it already is (the Workers); Python only on
   the reference side (torch, diffusers, upstream oracles); bash only as a
   one-line wrapper.**
3. **Byte-identical first.** Every refactor in the inference core lands
   behind the existing `techniques` byte-identity cells and the
   `determinism` tests. No kernel changes ride along with structure changes.
4. **CI must be able to say which commit was last validated on which GPU
   for which model.** Not "run everything on a GPU per PR", but a recorded,
   bounded gap.
5. **Preserve the strengths in §3.** The fake engine, the oracle method,
   the gate policy, the release channels and the cost guards stay as they
   are; they are moved, not rewritten.

### 7.2 Workstream A: the model family manifest (the keystone)

Introduce `families/<family>.toml` (or one directory per family with
`manifest.toml`), owned by a new small crate `fastvideo-manifest` (serde
types, validation, no CUDA) that every other crate and the control plane
read. Sketch, with the fields drawn from what §6 shows is needed (the numeric
values are illustrative; the real ones come from `cuda/caps.rs` and the
profile):

```toml
[family]
id = "h3"
display = "MiniMax-H3 / FastH3"
licence = "minimax-community+owner-licence"   # docs/ports/h3-ref2v.md §1.1
port_doc = "docs/ports/h3.md"

[[variants]]
id = "fasth3-4step-vsa"            # the one id everything else derives from
tier = "turbo"                      # draft | turbo | max
recipe = "4step-vsa"
profile = "h3/fasth3_4step_vsa"
trees = ["h3-base", "FastH3-4-step-Preview-v1-LoRA", "auxiliary/tae"]
served_names = ["fasth3", "h3-turbo"]
aliases = { minimax = ["MiniMax-H3", "MiniMax-H3-Turbo"], fal = ["minimax/h3-turbo"], ltx = [], openai = ["h3-turbo"] }
tasks = ["t2av", "i2v", "fl2va"]
canvas = { multiple = 32, max_area = 1032192, aspect = [0.25, 4.0], short_edges = [480, 768], hd_1080p = true }
frames = { step = 17, offset = 5, min = 73, max = 260 }
fps = { allowed = [24], default = 24 }
knobs = { seed = true, negative = true, steps = false, guidance = false }
refs = { images = 9, videos = 3, audios = 3, total = 12 }
gpu = { min_vram_gib = 80, min_sm = 89, needs = [] }   # machine-readable, not prose; an NVFP4 variant says min_sm = 100, needs = ["nvfp4"]
dit_gib = 45
gold = "lab/gold/fasth3-4step-vsa.toml"                          # §7.5
```

What it generates or replaces (each with a `--check` mode in CI so the
generated file cannot drift):

| Today | After |
|---|---|
| `engine-service/src/cuda/caps.rs::catalog()` and the recipe `match`; `validate.rs` limits; `protocol/caps.rs` helper constructors; the fake model set | `CudaModel` and `ModelCaps` built from the manifest; `validate.rs` reads ranges from it; the fake backend serves the same manifest |
| `[aliases]`, `served_names`, `[protocols]` repeated in 35 serve TOMLs | derived; a serve TOML lists variant ids and residency only |
| `core/src/registry.rs` 21 tables and `resolve()` | the CLI resolves Hub ids through the manifest (Hub ids are a field) |
| `control/src/presets.ts`, `enums.ts` ids and `VARIANTS`, `recipe-trees.ts`, `r2-trees.ts`, `weights-sources.ts`, `volumes.ts`, `cluster/catalog.json` | generated TS from the manifest (extend `control/gen-configs.mjs`, which already generates `worker-configs.ts` from `configs/serve`) |
| `scripts/serve/variants.sh` tables, `scripts/gpu/volumes.sh`, `scripts/gcp/families.tsv`, `configs/autoscale.toml` pools, `verify-weights.sh` cell lists | generated, then most of the scripts are deleted by §7.4 and §7.5 |
| `docs/serve/images.md` table, `docs/scope.md` preset tables | generated sections |
| per-surface limit constants in the four adapter crates | `ParamSpec` derived from the manifest (§7.6) |

The four weights TSVs stay as the tree-level record (dest, Hub id,
revision, sha256), because they are already data; the manifest references
trees by the TSV's dest name and CI checks that every referenced tree
exists in `weights-r2.tsv`.

Effort: about 2 weeks for the crate, the engine projection and the serve
TOML simplification; about 1 week for the control-plane generation. This
is the prerequisite for everything below and should land first.

### 7.3 Workstream B: inference core structure

1. **Extract the runtime.** Move `wan/{tensor,device,ops,kernels,nn,
   weights,offload,quant,log,dump,ledger,envflag,resident,stage_upload,
   hostmem,writer}` to `fastvideo-cudarc::rt` (a module first, a crate
   later if build times justify it); `wan/` keeps only the Wan family.
   Mechanical (`crate::wan::` to `crate::rt::`), byte-identical, and it
   removes the dependency of `gpucheck` and `engine-service` on a family
   name. Split `kernels.cu` by region into `rt/kernels/*.cu` and
   `<family>/kernels.cu` concatenated at build time; same SASS.
2. **One `VideoFamily` trait**:
   `resolve(hub_id) -> Option<Variant>`, `load(root, &Settings, &Plan) ->
   Box<dyn Pipeline>`, `Pipeline::generate(&Request, &Hooks) -> Output`,
   with a common `Request { canvas, timing, sampler, conditioning: enum
   Conditioning { Text, Image, Keyframes, Refs, Audio, ... } }` validated
   once by `Request::validate(&ModelCaps)`. The manifest supplies the
   variant table; a static `&[&dyn VideoFamily]` replaces the 21 tables and
   the 20 `generate_*` functions. `AvGenerateOptions` shrinks to the common
   fields; family-specific options move into the manifest or the profile.
3. **Finish the settings layer.** Generate the `FASTVIDEO_*` table (name,
   type, default, family, seam, doc) from one declarative list in
   `techniques/`; route the 111 direct reads through `settings::var`; pass
   a `Settings` snapshot into `load` instead of process globals; give Wan
   (and the small families) a `model_spec`. `docs/scope.md`'s table becomes
   generated and complete.
4. **Unify the duplicated hot-path pieces** under the byte-identity cells:
   one `fuse.rs` with a rounding-point policy; one `StepCache` trait; one
   `sdpa` entry with `sdpa_rule.rs` as the chooser; one disk prompt cache.
5. **Split the big loaders** (`H3Pipeline::load`, `Ltx2Pipeline::load`)
   into plan, open, place, fuse stages. The planners already exist in
   `fastvideo-models`; the split makes the engine lifecycle testable on CPU
   and is what a lazy/streamed cold start needs anyway.
6. **Retire what is dead**: `TensorBackend`, the mlx scaffold (or move it
   out of the workspace like oxide), `nvfp4_gemm.rs` and the oxide AOT path
   behind a feature, the host attention twins once `no_host_fallback`
   passes without them; fix ADR-0001 and `docs/scope.md` (removed backends,
   `unsafe` scope); add a port checklist (`docs/ports/CHECKLIST.md`) that
   is the §6 list reduced to "manifest + family directory + oracle".

Effort: step 1 about 1 week; steps 2 and 3 about 3 weeks together; step 4
about 2 weeks; step 5 about 1 week per big family; step 6 days. All of it
is CPU-testable except the byte-identity confirmation, which is one
`techniques` cell per family per step (about $2 per family).

### 7.4 Workstream C: first-class cross-provider deployment

Three pieces, in dependency order.

**C1. `deploy/families/<family>.toml`**: the deployment half of the family
manifest (it can be a `[deploy]` table inside §7.2's file): worker config,
image variant, residency plan (`resident`, `swap`), weights source default
per region, disk, GPU class with `needs`, timeouts, pool name, upsampler.
Generated from it: the presets, the variant tables, `families.tsv`, the
autoscale pools, the Dockerfile variant stages (or a single image with the
config supplied at boot, which the inline `FV_WORKER_TOML_B64` path already
proves works, making the seven variant images unnecessary).

**C2. `fv-boot`**, a Rust binary in the image beside `fv-serve`
(`crates/fastvideo-deploy`): the one boot sequence. Driver check, config
materialization (`FV_WORKER_TOML_B64` / `FV_WORKER_CONFIG` / baked), the
weights plan (volume / r2 / hub / none, via the plan API or an embedded
port of `r2-fetch.py`), sidecar start, tunnel and phase reports, watchdog,
`exec fv-serve`. Testable with a fake provider and a fake R2
(`scripts/gpu/tests/fake_r2.py` already exists). Deletes `fv-entry.sh`,
`fv-upsampler.sh`, the three `*_BOOT` strings, `PROVIDER_BOOT`,
`BREV_BOOTSTRAP`, `SLS_BOOT`/`SLS_R2_SH`/`SLS_HUB_SH`, `DRIVER_CHECK_SH`,
`UPSAMPLER_*_SH`, most of `scripts/gcp/startup.sh`, the `vm_boot` of
`cloudrift-worker.sh`, the `onstart` body of `vast.rs`, and
`scripts/gpu/serverless-worker.sh`. fv-control then sends env only; a
GCP or Brev cloud-init becomes `docker run <image> fv-boot`.

**C3. `fv-deploy`**, a Rust CLI with a `Provider` trait, promoted from
`fastvideo-autoscale::provider::Provider` and `fastvideo-deploy`:

```rust
trait Provider {
    fn offers(&self, q: &OfferQuery) -> Result<Vec<Offer>>;
    fn create(&self, launch: &Launch) -> Result<Instance>;   // Launch built from the family manifest + a provider section
    fn status(&self, id: &InstanceId) -> Result<InstanceState>;
    fn logs(&self, id: &InstanceId, since: Option<Cursor>) -> Result<LogPage>;
    fn destroy(&self, id: &InstanceId) -> Result<()>;
    fn account(&self) -> Result<Account>;                     // balance or budget, for the shared money guard
}
```

Implementations: `runpod::Pod`, `runpod::Serverless` (reuse
`provider/runpod.rs`), `vast`, `cloudrift`, `gcp` (port `vm.sh`/`auth.sh`),
`gmi`, `brev` (port from TS, or keep those two in TS behind the same JSON
`Launch`/`Instance` schema until the Rust versions are proven). Shared
once: price cap, balance floor, detached wall-clock backstop, ledger to D1
`deployments`, readiness poll on `/ping`, template sync. Subcommands:
`fv-deploy plan|up|smoke|down|templates sync|release promote|rollback|reconcile`.
fv-control keeps the orchestration it is good at (the Durable Object step
machine, env layers, dashboards, alerts) and calls `fv-deploy` semantics
through the same `Launch` schema (generated TS types from the Rust structs
via `schemars`, the way `schemas.ts` already feeds the UI).

Collapses: `runpod-pod.sh`, `runpod-endpoint.sh`, `runpod-templates.sh`,
`vast.sh`, `vast-serverless.sh`, `cloudrift-worker.sh`,
`cloudrift.sh`+`cloudrift-lib.sh`, `scripts/gcp/{auth,lib,vm,e2e}.sh`,
`serverless-coldstart.sh`, `runpod-price.sh`, `runpod-cuda.sh`,
`scripts/serve/lib/{registry,d1,ghcr}.sh`, and finally `release.sh`.

Effort: C1 about 1-2 weeks (mostly generation and tests); C2 about 2-3
weeks; C3 about 4-5 weeks for Runpod + Vast + GCP with the shared guards,
plus about 1 week per further provider. The first live GCP run and the
first live GMI/Brev run become `fv-deploy up` invocations with the same
smoke test, which is the point.

### 7.5 Workstream D: one validation driver, gold standards, tiers

**D1. `fv-lab`**, a Rust binary (a sibling of `fv-gpucheck`, which already
owns `gate`, `compare-clips`, `compare-dumps`, `lpips`, `benchmark.rs`,
`quality.rs`, `serve.rs`, `coldstart.rs`) with subcommands:
`pod {up,status,fetch,down,reap}` (on the `Provider` trait of C3),
`run <family|cells>` (the on-pod cell runner replacing `runpod-matrix.sh`,
`remote.sh`, every `*-pod.sh`), `weights {plan,fetch,verify,publish,fill,
copy}` (replacing the `fetch-*`, `r2-*`, `rebuild-volume`, `verify-*`
scripts; `r2-weights.py`/`r2-fetch.py` stay Python until C2 embeds them),
`report {summary,table,tiers}` (replacing every `*-summary.py`,
`hd_report.py`, `sol_bench_table.py`, `timings.sh`), and the existing
`gate`/`oracle` moved or aliased.

**D2. Data read by `fv-lab`, checked in:**
- `lab/tiers.toml`: per tier and family: GPU class, offer query, `$/hr`
  cap, wall cap, disk, image flavour, stages, cells (replaces the `case`
  tables in `validate.sh` and `runpod-matrix.sh`).
- `lab/cells/*.toml`: a cell is a profile plus a workload (prompt set,
  seeds, size, frames, steps) plus arms.
- `lab/gold/<variant>.toml`: **the gold standard per served model**:

```toml
[id]           variant = "fasth3-4step-vsa"  profile = "h3/fasth3_4step_vsa"  serve_config = "configs/serve/runpod-family-h3.toml"
[reference]    prompts = "scripts/gpu/prompts-eval.json"  seeds = [1024]  size = [832, 480]  frames = 124  steps = 4
               upstream = { image = "fastvideo-rs-upstream-fastvideo", target = "fasth3-4step-vsa", rev = "e90be59" }
[gpu]          type = "RTX PRO 6000 Blackwell"  sm = 120  price_usd_hr = 2.09
[quality]      policy = "scripts/gpu/gate-policy.toml"            # relative gate vs the baseline cell (kept)
               oracle = "bf16-closer-to-ref-than-own-f32"           # docs/oracle.md method (kept)
               lpips_vs_upstream_max = 0.25  psnr_vs_upstream_min_db = 19.0   # absolute, from docs/oracle.md
               determinism = "byte-identical"
[performance]  denoise_s_max = 10.5  total_s_max = 14.5  peak_gib_max = 34  cold_start_s_max = 120
[cost]         usd_per_clip_max = 0.0085
[last_validated] commit = "fb1bce3"  run = "artifacts/gold/fasth3-4step-vsa/fb1bce3.json"  date = "2026-10-14"
```

The relative gate policy and the oracle method are kept exactly; what is
new is the absolute bound per model, the cost bound, cold start as a gated
number, and the `last_validated` record. The numbers for the first gold
files come from `docs/perf/raw-inference.md`, `docs/perf/sol-bench.md`,
`docs/perf/h3-nvfp4-tier-options.md`, `docs/oracle.md` and the profile
comments, which then stop carrying numbers.

**D3. The tier table.** `fv-lab report tiers` joins `benchmark.json`
(`total_s`, `denoise_s`, `decode_s`, peak memory, per-prompt medians), the
gate JSON (speedup, LPIPS, PSNR, sharpness, jitter, verdict), the pod
record (`$/hr`, GPU, image digest) and the gold bounds, and emits one row
per (variant, GPU): `$ per clip`, seconds per second of video, LPIPS vs
upstream, gate verdict, speed tier from `gate-policy.toml [tiers]`,
validated commit. Rendered to `docs/perf/tiers.md`, it is the owner's
"clear delineation of performance tiers between cost / latency / quality";
the public `draft < turbo < max` tiers of `design.md` §0.3 and §0.6 become
rows that must satisfy their gold file, instead of names chosen by hand.

**D4. The CI bridge.** `gpucheck-t0` additionally checks that every
`lab/gold/*.toml` parses, references an existing cell and tree, and that
`last_validated.commit` is an ancestor of HEAD not older than N days; a
nightly or on-demand `fv-lab run gold --all` workflow on one Runpod pod
(about $2/hour, the `rtx6000` family today) updates the records. The gap
between CI green and GPU-validated becomes a visible number per model.

Effort: D1 about 3-4 weeks (the pod lifecycle and cell runner are the
bulk; `gate`/`oracle`/`lpips` already exist); D2 about 1 week of data
entry from the docs; D3 about 1 week; D4 days.

### 7.6 Workstream E: API parity and parameter certification

1. **Vendor the upstream specs** under `docs/serve/specs/`: the LTX
   `openapi.json`, the fal queue OpenAPI for the 15 endpoints, the director
   `asyncapi.json`, the MiniMax `v2-video-generation.json`, the OpenAI
   `openapi.yaml` `/videos` subset, a JSON Schema derived from FastVideo's
   `protocol.py`; generate the native and director schemas from the Rust
   types with `schemars`. Add a drift script (re-fetch, diff, fail) run
   weekly. About 8-12 person-days including the conformance generator:
   per request schema, emit boundary-valid bodies and one invalid body per
   constraint (type, enum, bound, required, pattern, unknown field), and
   validate every reply and golden against the response schema on the fake
   server.
2. **`ParamSpec` as the single source of truth** per (surface, model):
   derived from the manifest's `ModelCaps` plus per-surface constants
   (names, enums, defaults); it drives adapter validation, `/fal/schema`,
   `/fv/v1/capabilities`, the doc tables and the LTX/fal model matrices;
   the duplicated constants are deleted and the LTX body gets a typed
   struct. Unknown-field policy and knob ranges become one rule per surface
   declared in the spec. About 10-15 person-days.
3. **The certification matrix**: generalize `dimension_sweep.rs` into a
   surface-agnostic harness over `/v1/videos`, FastWan, MiniMax, LTX v1/v2,
   every fal app, director `configure`/`prompt`, native jobs, streams and
   WHIP ingest. Per parameter: boundary valid, below, above, wrong type,
   unknown field, mutual exclusion; the expected status, error code and
   `param`/`loc` per surface. Emit a parameter x surface x model TSV, and
   make "every parameter has a valid and an invalid case" a CI assertion.
   About 8-10 person-days.
4. **Coverage and CI**: `cargo-llvm-cov` for the adapter crates on the
   build pod, the parameter-coverage report from (3) as a CI artifact, and
   the adapter crates' `cargo test` added to a GitHub workflow (today they
   run only on the build pod). About 3-5 person-days.
5. **Upstream captures**: record real upstream responses and error bodies
   (with keys the owner holds) as fixtures, so goldens stop being
   self-generated. Size depends on account access.

Total about 30-42 person-days. This workstream depends on A for the
manifest-derived caps, but (1) and (4) can start immediately.

### 7.7 Workstream F: immediate deletions and doc fixes (no design risk)

- Delete `scripts/vast-{generate,gpu-bench,pull-artifacts,pull-weights,setup-cuda,sync}.sh`,
  `scripts/docker-build-cuda.sh`, `scripts/gpu/serverless-worker.sh`,
  `scripts/gpu/upstream_bench.py`, `scripts/gpu/sage/{run,vast-run,setup}.sh`
  (fold into an upstream cell), `scripts/gpu/build-remote.sh`,
  `scripts/gpu/prompts-flux2.json`, `scripts/gpu/lpips_fixtures.py` (or
  commit its output as a fixture), `scripts/ci/test_tools_release.sh` (or
  wire it into CI), `scripts/serve/fv-app.sh` if `app/` deploys itself.
  Owner decision needed for `scripts/serve/vast.sh` and
  `vast-serverless.sh` (is Vast serving coming back?).
- Move the 24 port one-offs and `runpod-matrix.sh` to
  `scripts/gpu/experiments/` until D1 absorbs them, so `scripts/gpu/`
  reads as tooling again.
- Delete the `h3-*`/`ltx2-*` Vast tiers from `validate.sh` and the README
  §Vast sections; make `README.md` describe the Runpod path.
- Fix `docs/serve/images.md:87` (names the deleted volume),
  `scripts/gcp/weights.sh:3-4` (calls the volume the record of truth),
  ADR-0001 and `docs/scope.md` (frozen backends, `unsafe` scope).
- In serving: delete `multiworker.rs` and `server.workers_max` if the
  Runpod LB mode is retired (owner to confirm), rename `[gateway]` to
  `[cluster]`, drop `gw_workers`, add a per-job deadline to `StepControl`,
  bound the engine event channel, add per-model duration and queue-wait
  histograms and residency gauges, and implement the `otel` feature.
- Make `cargo fmt --check` and `clippy -D warnings` real gates after one
  formatting commit (the drift is 60 files and 27 lints, a day's work).

### 7.8 Script count, before and after

| Group | Today | After A-F | Where it goes |
|---|---:|---:|---|
| Provider ops (Runpod, Vast, CloudRift, GCP, serverless, release, lib) | 44 | ~8 | `fv-deploy` (C3), `fv-boot` (C2); wrangler wrappers and `release.sh` thin wrappers stay |
| Weights (fetch, r2, verify, rebuild, TSVs) | 30 | ~9 | `fv-lab weights` (D1); the 4 TSVs and `r2-weights.py`/`r2-fetch.py` stay |
| Validation tiers (validate, remote, matrix, family-pool, prompts, fixtures) | 17 | ~9 | `fv-lab pod/run` (D1); prompts, fixtures, `gate-policy.toml` stay as data |
| Port one-offs (`*-pod.sh`, `*-validate.sh`, `*-summary.py`, `*-plan.py`) | 24 | 0 | cells in `lab/cells/*.toml` + `fv-lab report` |
| Oracle / parity (Python reference side) | 38 | ~25 | stays Python; eight oracle generators behind one `oracle_dump.py --target`, four `fetch-*.py` folded into `fetch-hub-tree.py` |
| Perf bench | 27 | ~15 | upstream `bench_*.py` stay; summaries move to `fv-lab report` |
| E2E serve (clients, pods) | 24 | ~16 | client-side Python/JS stays; the pod scripts go to `fv-deploy`/`fv-lab` |
| Build / CI / dev | 19 | ~16 | stays (GitHub-runner and Docker specific) |
| UI, tests, misc | 17 | ~10 | `scripts/ui` retired with the Vast ladder; shell tests move next to the Rust tools |
| **Total files** | **240** | **~95** (3 new Rust binaries, ~12 new TOML data files) | shell lines ~27k to ~8k |

### 7.9 Sequencing

| Phase | Weeks (one engineer, excluding GPU validation wall time) | Delivers |
|---|---|---|
| 0 | 1 | F: deletions, doc fixes, fmt/clippy gates, experiments folder |
| 1 | 3 | A: manifest crate, engine catalog and serve TOML as projections, control-plane generation |
| 2 | 3 | B1-B3: runtime extraction, `VideoFamily` trait, settings layer (byte-identical cells after each) |
| 3 | 4 | C1-C2: deployment manifest, `fv-boot`; first live GCP and Brev boots through it |
| 4 | 5 | C3: `fv-deploy` with Runpod, Vast, GCP; fv-control on the shared `Launch` schema |
| 5 | 5 | D: `fv-lab`, gold files for every served variant, `tiers.md`, the CI bridge |
| 6 | 6 | E: specs, `ParamSpec`, certification matrix, coverage |
| 7 | 3 | B4-B6: fusion/cache/attention unification, loader split, dead code |

Phases 2, 4 and 6 are independent of each other once phase 1 has landed,
so with the repository's usual parallel agents the calendar is closer to
12 weeks than 30. Each phase has a measurable exit: phase 1, zero
hand-written catalog rows and zero `[aliases]` blocks; phase 2, byte
identity on the `techniques` cells and `no_host_fallback` green; phase 3,
one boot script in the tree; phase 4, one `fv-deploy up` per provider with
the same smoke; phase 5, a `last_validated` commit per served variant and
a generated `tiers.md`; phase 6, every parameter with a valid and an
invalid certified case; phase 7, one fusion, one cache trait, one attention
entry.

### 7.10 Risks

- **The manifest can become a second catalog** if the engine keeps
  hand-written rows "for now". The exit criterion for phase 1 is zero
  hand-written rows; the `--check` generators enforce it.
- **Byte identity across the runtime extraction** is cheap to verify but
  must actually be run per family; budget about $2 per family per step.
- **Rewriting bash-in-TS into Rust changes the failure modes** on real
  providers (the TS code carries hard-won provider quirks: Brev stop
  semantics, Runpod quota errors). Port the quirks as tests first, from
  `control/test/unit/*` and the mock providers.
- **`r2-fetch.py` is embedded verbatim by fv-control**; keep it Python
  until `fv-boot` has replaced every caller, then port.
- **Owner decisions needed**: Vast serving (keep or drop), Runpod LB mode
  (keep or drop `multiworker.rs`), single image with boot-time config
  versus seven variant images, and whether GMI/Brev move to Rust or stay
  TS behind the shared schema.

## 8. Recommendation

Do phase 0 and phase 1 now. Phase 1 (the manifest) is the one change that
makes every later workstream smaller: the engine catalog, the serve
configs, the control presets, the deployment presets, the parameter
schemas and the gold standards all become projections of it, and the
answer to "what does it take to deploy model X" becomes "its manifest
row". Then run B, C and E in parallel, and D as soon as C's `Provider`
trait exists. The three Rust binaries (`fv-boot`, `fv-deploy`, `fv-lab`)
are where the scripts go; the owner's two stated goals, fewer scripts and
first-class cross-provider deployment, are the same deliverable.

## 9. What was not verified

- Nothing was compiled or run; test counts are `grep` counts of
  `#[test]`/`#[tokio::test]`, not `cargo test` output.
- GPU timings, LPIPS and PSNR figures are quoted from the repo's own docs
  and artifacts, not re-measured.
- Whether `validate.sh`'s `h3-*`/`ltx2-*` tiers, `scripts/gpu/sage/*` and
  the Vast ladder still run (no callers, no recent artifacts in this
  checkout).
- Upstream specs were not re-fetched; drift since 2026-09-27/28 is unknown.
- Director per-field test coverage was not read in detail.
- Durable Object behaviour on live Cloudflare, and the live GMI/Brev/GCP
  paths, which the repo itself marks as simulated or unverified.
- The git history in this clone starts 2026-10-07 (316 commits), so file
  age could not be used; obsolescence was inferred from content.

Source material for this review: the five layer reports are summarized in
this file; their evidence is the file and line citations above.
