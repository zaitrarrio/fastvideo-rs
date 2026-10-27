# Decision log

Project code: FVID

### FVID · 2026-10-24 · FVID-2026-10-24-metronome-deadline-slack
- Trigger: `stream_causal`'s paused-time session-limit tests hung twice (CI job 120998352681, `check.sh` on the build pod). Root cause: when the first block reaches the causal pacer while tokio's paused clock stands at a timer-wheel slot boundary (e.g. 1.984 s) rather than a whole second, the metronome schedule is anchored on a millisecond that f64 cannot represent; later the clock lands exactly on a deadline that the f64 schedule places a few ulps later (64.109 s vs 64.10900000000001 s), `Metronome::poll` asks for a 7e-15 s wait, `Duration::from_secs_f64` makes it 0 ns, tokio fires that sleep at once without advancing the paused clock, and the pacer spins forever without ticking or closing `ticks`
- Options: a tokio timeout or retry in the test; round the wait up in the causal pacer only; let `Metronome::poll` treat a deadline within a small slack as reached
- Decision: `Metronome::poll` ticks when `now` is within `DEADLINE_SLACK_S` (1 µs) of the deadline, so every wait it returns moves any clock; the two paused-time tests run on their own thread under a 60 s real-time guard (`paused_time`) that fails with a clear message instead of hanging. No ADR: in real time a tick due within 1 µs now fires at that poll instead of after a zero-length sleep, which nothing can observe
- Reason: the defect is in the scheduler's contract (a wait shorter than any timer's resolution), shared by the clip pacer and the fal director; the guard only bounds a future hang
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none
- Verification: `metronome_waits_always_move_a_millisecond_clock` (replays the paused clock for 1003 anchors; fails before the fix, e.g. at the 1.984 s anchor); a Python replay of tokio 1.53.1's millisecond wheel and Rust's float conversions: 6370 of 20000 whole-ms anchors spun before, 0 after, 0 of 20000 slow-first-block runs after; on the build pod the test fails with the slack set to 0 and passes with it, 20 of 20 repeated runs of the session-limit tests pass, and fmt, workspace clippy, the engine-service tests, the workspace lib tests and `scripts/serve/check.sh` are green

### FVID · 2026-10-24 · FVID-2026-10-24-gold-tier-gpu-and-settings
- Trigger: review of PR #151 (plan D2): two gold files carried oracle numbers measured on an H100 for an RTX PRO 6000 tier (`fasth3-4step-vsa` PSNR / LPIPS, `ltx25-ref2v` PSNR), and H3 latencies measured before the MXFP8 video-decoder default (2026-10-14) sat in the fields while `fasth3-trim-nvfp4` gapped its numbers for the same reason
- Options: keep the numbers with a note; gap every number not measured in the served configuration; keep a number measured under a non-default setting only when that exact setting is named
- Decision: **(coordinator) a number measured on a GPU other than the tier's is a gap, its values kept in a cited `[[source]]` note and the gap reason ("measured on H100 only; D3 re-measures on the tier GPU"); a number measured under a non-default setting stays in the fields only when `workload.note` names the exact setting that reproduces it (here `FASTVIDEO_H3_VAE_MXFP8=0`, the bf16 video decode that was the default before 2026-10-14, and `FASTVIDEO_WAN_QUANT=mxfp8` for the Wan TAEHV arms), with the environment also in a new `workload.env` the parser checks the note names; every `[[source]]` carries a heading (checked to exist in the file) and a date.** ADR-0009 Choice 1 unchanged; Choice 5 added
- Reason: one rule for every file; `fv-lab` can rerun a cell exactly; a gold number is only comparable on the tier's GPU
- Reversibility: cheap (this repository's format)
- Executed by: Executor (plan D2, review fixes)
- ADR: docs/adr/0009-gold-standard-files.md (Choice 5)
- Verification: `cargo test -p fastvideo-lab` (`sources_need_a_heading_and_a_date`, `workload_env_is_named_in_the_note`, `every_cited_heading_exists_in_its_file`)

### FVID · 2026-10-24 · FVID-2026-10-24-gold-standard-files
- Trigger: plan D2 (docs/plan/2026-10-23-execution-plan.md): `lab/gold/<variant>.toml` for every served variant from the numbers in `docs/perf`, `docs/oracle.md` and the profile comments, keyed by registry name and `@rev`; the profile comments stop carrying numbers
- Options: bounds with margins as the review sketches them; the measured values as the sources report them, with the gate's tolerances applied by `fv-lab`; numbers copied from the nearest measured configuration when the variant itself has none; only cited numbers, with every absence declared
- Decision: **a new library crate `fastvideo-lab` defines the format (`Gold`, `validate`, `load_dir`, `schemas/gold.v1.json`); every numeric field is optional, a present one must be cited by a `[[source]]` (path, heading, date) and an absent one declared by a `[[gap]]` with its reason, both enforced by the parser; the first 30 files hold the sources' measured values (no margins, no rescaling, no borrowing from nearby configurations, which are cited in notes); the most recent dated source wins and both are cited; `cost.usd_per_hr` is the class's most recent dated price ($2.49/hr, RTX PRO 6000, EU-RO-1, 2026-10-14) and `usd_per_clip` the one derived value; profile comments point to the gold files and `docs/techniques.md` and a CPU test rejects measurement-looking numbers in them; `lab/tiers.toml` is left to D1**
- Reason: a gold standard that invents is worse than none; the gaps are D3's GPU work list; the gate's tolerances are already the repository's rule
- Reversibility: cheap (this repository's format, versioned by `schema`; nothing published depends on it)
- Executed by: Executor (plan D2)
- ADR: docs/adr/0009-gold-standard-files.md
- Verification: `cargo test -p fastvideo-lab` (unit tests; every gold file parses and validates; gold files == family variant ids both ways; family, tier and profile agree; cited paths exist; no measurement in a profile comment; schema drift); profiles' TOML values unchanged (comment-only diff)

### FVID · 2026-10-24 · FVID-2026-10-24-family-files-are-the-source
- Trigger: plan A3 (docs/plan/2026-10-23-execution-plan.md): the engine projection of the family manifests; `catalog()` deleted, the serve TOMLs reduced to variants and residency
- Options: load `families/` from disk at startup; embed the files at build time; embed with a run-time directory override
- Decision: **the engine embeds `families/*.toml` (`include_str!`) and `FV_FAMILIES_DIR=<dir>` overrides them at run time**; a test keeps the embedded copies equal to the committed files
- Reason: the binary and the serve images stay self-contained (no new file to ship or mount), and a lab run can still try an unreleased variant without a rebuild
- Reversibility: cheap (one function, `cuda::families::family_set`)
- Executed by: Executor (plan A3)
- ADR: docs/adr/0008-family-manifest-layer-2.md (amendment, decision 1)
- Verification: `crates/fastvideo-engine-service/tests/families.rs` `the_embedded_files_are_the_committed_ones`; `cuda::families` unit test of the override

### FVID · 2026-10-24 · FVID-2026-10-24-serving-caps-in-layer-2
- Trigger: plan A3 / A2 review follow-up (b): the parts of `ModelCaps` layer 1's `Caps` does not carry (tasks, default frame count, audio, container-only fps, pad-and-crop, the H3 1080P tier, the reference-strength knobs)
- Options: widen layer-1 `[caps]` (a pinned public format, before the first publish); a `[variants.serving]` table in layer 2 beside `caps`; keep them derived by the engine from the recipe
- Decision: **`[variants.serving]` in layer 2**; only the 1080P on/off switch and the stream shape stay recipe-derived (they change per deployment)
- Reason: these fields describe how this runtime serves a variant, which a registry clone does not need; widening layer 1 later is a versioned change made only if the registry wants them
- Reversibility: cheap (layer 2 is versioned by `schema`)
- Executed by: Executor (plan A3)
- ADR: docs/adr/0008-family-manifest-layer-2.md (amendment, decision 2)
- Verification: `crates/fastvideo-serve/tests/capabilities_golden.rs` (every shipped config's `/fv/v1/capabilities` unchanged); `tests/families.rs` `serving_caps_project_onto_model_caps`

### FVID · 2026-10-24 · FVID-2026-10-24-public-names-replace-aliases
- Trigger: plan A3: zero `[aliases]` blocks in the serve configs
- Options: keep `[aliases]` per config; derive the API names from tier rules in the adapters; a `public_names` list per variant carried by the engine
- Decision: **`public_names` per variant, carried by `EngineBackend::public_names` into `CapabilityTable::public_aliases`; `[aliases]`, `[[models]] family` and `[[models]] recipe` are refused**; a name claimed by a tier and its task companion goes to the model that serves text-to-video
- Reason: the names are facts about the variant, not the deployment; one place instead of 23 hand-kept tables
- Reversibility: cheap before any external config depends on the new form
- Executed by: Executor (plan A3)
- ADR: docs/adr/0008-family-manifest-layer-2.md (amendment, decision 3)
- Verification: the capabilities golden; listed exception: `fake.toml` and `tests/compat/serve.toml` gain `fastwan-1.3b -> fake-wan` (the fake model set now declares it, as `runpod-fake.toml` did by hand)

### FVID · 2026-10-24 · FVID-2026-10-24-trees-explicit-tae-rules
- Trigger: A2 review follow-ups (a), (c), (e): `implicit_trees` was an engine rule, `${tae}/<file>` was not checked against `trees`, the ADR called every tree a `weights-manifest.tsv` row
- Options: keep the rule in the engine; write the trees in the files and check them by test
- Decision: **trees are data in the files (rule deleted, a test checks them against the recipes); `${tae}/<file>` needs `auxiliary/tae/<file>` in `trees`, a bare `${tae}` is a search path, a bare `${weights}` is refused; the ADR and README name non-Hub trees (`h3-ref2va`, `longlive-1.3b-safetensors`) as weights-store trees**
- Reason: A4's generators and the registry read `trees` without the engine
- Reversibility: cheap
- Executed by: Executor (plan A3)
- ADR: docs/adr/0008-family-manifest-layer-2.md (amendment, decisions 4 and 5)
- Verification: `fastvideo-manifest` unit tests (`rejects_bad_files_with_paths`); `tests/families.rs` `trees_name_what_the_recipes_read`

### FVID · 2026-10-24 · FVID-2026-10-24-longlive-and-lightx2v-v01-variants
- Trigger: plan A4 (#149) found two served models with no variant: `longlive-1.3b` (an SF-Wan entry with a `longlive` path) and `h3-lightx2v-turbo-v01` (a `model_from_config` special case of `h3-lightx2v-turbo`)
- Options: keep them as serve-config special cases; make them variants
- Decision: **both are variants** (`families/wan.toml` `longlive-1.3b`, tree `longlive-1.3b-safetensors`; `families/h3.toml` `h3-lightx2v-turbo-v01`, tree `h3-turbo-loras/lightx2v-v0.1`); their configs keep the old id of the base variant as a compatibility served name so the API is unchanged
- Reason: the file is the source; A4's `CATALOG_EXCEPTIONS` and the `recipe-trees.ts` override can be removed
- Reversibility: cheap
- Executed by: Executor (plan A3)
- ADR: docs/adr/0008-family-manifest-layer-2.md (amendment)
- Verification: the capabilities golden for `runpod-longlive`, `runpod-family-streaming`, `runpod-h3-lightx2v-turbo-v01`, `runpod-family-h3`; `cuda::caps` tests `longlive_variant_and_its_overrides`, `turbo_lora_models`

### FVID · 2026-10-24 · FVID-2026-10-24-settings-layer
- Trigger: plan B3 (docs/plan/2026-10-23-execution-plan.md): the `FASTVIDEO_*` table from one declarative list, the direct reads routed through the settings layer, the scope doc generated, byte-identical
- Options: a typed key per setting at every call site; string literals at the call sites with the list made authoritative by workspace scan tests; the list and readers in a new dependency-free crate or in `fastvideo-models`; route the direct reads through the layered reader (`settings::var`) or through an env-only reader
- Decision: **one sorted `SETTINGS` table in `fastvideo-models::techniques::settings` (222 rows: reader, parse, default, family, seam, read timing, doc); call sites keep their literals; every direct and indirect read calls `settings::env_var` / `env_var_os` (exactly `std::env::var` / `var_os`) at the same call site, so no name changes reader, parse or read time; the runtime's typed getters call the shared `settings::parse` functions; `docs/scope.md`'s table is rendered from the rows; B3 lands in two PRs (this one: list, routing, doc, tests; then the `Settings` snapshot for `load` and the Wan / small-family `model_spec`s)**
- Reason: `fastvideo-models` sits below every reader without a cycle; promoting the 108 env-only names to the layered reader would let profiles set them, a behaviour change; typed keys would rewrite ~400 sites in one structure change for what the scan tests already enforce
- Reversibility: cheap (moving a name to the layered reader, or to a typed key, is a local change plus its row)
- Executed by: Executor (plan B3)
- ADR: docs/adr/0010-settings-layer.md
- Verification: `crates/fastvideo-models/src/techniques/settings/{table,parse,workspace_check}.rs` on the build pod (no direct read outside the module, no undeclared name, getters agree with rows, doc not stale, off-identity per typed row and per in-crate text parser, env > profile > default for H3 / LTX keys); GPU `techniques` byte-identity cell by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-family-manifest-layer-2
- Trigger: plan A2 (docs/plan/2026-10-23-execution-plan.md): layer 2 of the model family manifest and the first three family files, with the equality proof against the hand-written engine catalog
- Options: the pure crate also owning the engine's recipe structs (one schema, the crate depends on the inference core); the engine's recipe as an opaque table the crate validates only around (ids, trees, GPU, caps); a separate crate for layer 2
- Decision: **layer 2 lives in `fastvideo-manifest` as `family` with `engine` an opaque TOML table the engine interprets; trees are `weights-manifest.tsv` names and every path in `engine` is written `${weights}/<tree>/…` or `${tae}/<file>` and substituted with the engine's layout at load; the one directory reader is the default-off feature `fs`.** The files are generated from `catalog()` until A3 inverts the direction; `registry_name` stays empty until a model is published
- Reason: the registry, `fv-boot`, `fv-lab` and the control-plane generator read family files without the inference core; one file serves every machine; the crate's no-I/O contract holds by default
- Reversibility: cheap (layer 2 is this repository's format, versioned by `schema`; nothing published depends on it)
- Executed by: Executor (plan A2)
- ADR: docs/adr/0008-family-manifest-layer-2.md
- Verification: `crates/fastvideo-engine-service/tests/families.rs` (projection equals the catalog row for row under two layouts, files equal the generator, id sets equal, every tier alias covered); `cargo test -p fastvideo-manifest` (unit tests, `schemas/family.v1.json` drift)

### FVID · 2026-10-23 · FVID-2026-10-23-manifest-crate-registry-formats
- Trigger: the architecture review (docs/gaps/2026-10-22-architecture-review.md, merged as #137) proposed one family manifest as the source of truth; the sibling `zaitrarrio/model-registry-rs` had specified the same public formats as its own `registry-core` (WO-001: naming grammar, `model.toml`, signed file-and-chunk manifest, lineage, signatures); the owner asked (2026-10-23) to fold that work into the review and rescope the registry
- Options: two schemas in two repositories kept equal by tests; the formats in the registry with the runtime depending on it; the formats with the runtime and the registry depending on it; everything in one repository
- Decision: **the core formats are implemented here as the pure crate `fastvideo-manifest` (layer 1: name, model_toml, manifest, lineage, sig, schema export; layer 2: the runtime family manifest) and the registry depends on this repository for them.** Before the first publish, `model.toml` gains `[variant]`, a `[runtime]` recipe/profile pointer and `[caps]`; the name grammar is unchanged. `fv-lab run gold` is the registry's verification runner and writes the registry's benchmark record; the registry's `hydrate` is the pod weights fetch of `fv-boot` and `fv-lab`; the chunk store, API, `gia` CLI, licence gate and model pages stay in the registry repository. The four weights TSVs are generated from manifests and then retired
- Reason: the runtime is the thing that guarantees a published model loads, so it owns the format it loads; two copies would recreate the duplication the review found, now across repositories; one benchmark runner and one weights fetcher instead of two
- Reversibility: cheap until the first model is published; costly after (public formats)
- Executed by: Executor (docs only; the crate is Workstream A, phase 1)
- ADR: the registry's ADR-0001..0003 are written here with the crate
- Verification: review §7.2, §7.4, §7.5, §7.11 revised; the registry's CLAUDE.md, BACKGROUND, work orders and decision log rescoped in the same decision

### FVID · 2026-10-20 · FVID-2026-10-20-tween-storyboard
- Trigger: owner asked for a fast FLF2V / FLRef2V model for animated storyboards and chose (2026-10-20) an end-to-end storyboard app, LTX and H3 side by side, first + last panels only, a new app in this repository
- Options: second model H3 fl2va/ref2va, Wan Fun-InP 1.3B, a Wan VACE port; mid-shot keyframes now or later; app in this repo, a separate repo, or a view in app/
- Decision: **Tween** (`tween/`, Worker `fv-tween`, own D1 + R2) drives fv-serve over `/v1/videos`; shots are adjacent panel pairs (shared boundary frames); `keyframes` and `reference` shot modes; engine model ids are Worker vars (draft / final / reference per engine); engine key server-side; real-time factor from the engine's `inference_time_s`. No Rust change; `Anchor` stays `First | Last`
- Reason: every capability exists in fv-serve today (LTX-2.5 and H3 declare `Task::Keyframes`); the fake engine gives a GPU-free local stack
- Reversibility: cheap (a separate Worker; nothing in fv-serve depends on it)
- Executed by: Executor
- ADR: docs/adr/0004-tween-storyboard-app.md
- Verification: `tween/` typecheck, unit tests and `wrangler deploy --dry-run`; `task tween:e2e` against the fake engine (owner-run, needs Docker)

### FVID · 2026-09-29 · FVID-2026-09-29-ltx-device-rope
- Trigger: datacenter profile (docs/perf/datacenter-profile.md): LTX idles 3.6-4.6 s before stage 2 on a single-threaded host RoPE build, and the VAE decode stalls behind the CPU x264 feed; kernel plan WP-F, owner-approved
- Options: cache the host tables per geometry; build them on the device; keep the host build
- Decision: **the LTX RoPE tables are gathered on the device** (`ltx_split_rope`) from a factored form (cos/sin of each distinct fraction x frequency plus an index), and file jobs feed ffmpeg from a relay thread (`FrameSink::detach`). Escape hatch: `FASTVIDEO_LTX2_HOST_ROPE=1` restores the host build
- Reason: the gather is bit-identical to the host tables (pure copies), needs no per-geometry cache, and covers every conditioning layout
- Reversibility: cheap (env flag; the detach is one trait method)
- Executed by: Executor
- ADR: none
- Verification: `factored_tables_expand_to_the_direct_ones_exactly` (host), `fv-gpucheck kernels` group `ltx_rope` (device vs host, 1080p), sink/output tests; timings in docs/perf/datacenter-profile.md "WP-F results"
### FVID · 2026-09-29 · FVID-2026-09-29-vsa-datacenter
- Trigger: owner approved WP-D; the datacenter profile put VSA's mma.sync fine kernel at 51-61% of H3 turbo denoise on B200 (41-52% on H100)
- Options: KV-tile-list producer in `attn_dc.cu` with tcgen05 / wgmma consumers; port FastVideo's ThunderKittens sm_90a / sm100a kernels; keep mma.sync
- Decision: `attn_dc.cu` gains a KV-tile-list producer, `fa_dc100_vsa` (M = 128 over the union of two query tiles' selections, per-row masks) and `fa_dc90_vsa` (one query tile per wgmma warpgroup), a fused prep (tiles + means, bit for bit the old kernels, so the selection is unchanged) and H3's combine in the fine epilogue. Seam value `vsa_attention = dc`, `auto` on 9.0 / 10.0 only; `FASTVIDEO_VSA_KERNEL=tma2` restores the mma.sync path; sm_120 never loads the module. sm_103a cubin built, loaded on 10.3 only with `FASTVIDEO_DC_SM103=1`
- Reason: `vsa_dc` kernels group 99/99 on B200 and H100 (fine vs tma2 rel-L2 <= 5.7e-4, vs an f64 block-masked reference within 1.011x tma2's error, prep and combine bit-identical); B200 H3 turbo denoise 11.87 -> 9.18 s (768p), 32.73 -> 24.46 s (1080p) on one pod, H100 19.46 -> 16.34 s / 49.62 -> 39.44 s; step-1 block dumps at the order of the Rust-vs-Python oracle; clip divergence (LPIPS 0.28) below a dense-kernel-switch control (0.32)
- Reversibility: cheap (`FASTVIDEO_VSA_KERNEL=tma2`)
- Executed by: Executor
- ADR: none
- Oracle (owner-approved, $2.06): FastVideo reference (RTX PRO 6000; the upstream image dies on sm_90) injected into ours on one H100, tma2 vs dc on the same pod: final latents 0.387 vs 0.375, clip PSNR 19.17 vs 19.73 dB, SSIM 0.691 vs 0.699, LPIPS 0.238 vs 0.236 against the reference; dc no worse at any block, so the default stays dc. Pods 9rz0air1ze26xy (failed H100 reference), uomkr05e6boddd (reference), 7nnmf9cciynj8j (H100 runtime), all deleted
- Verification: pods 1i22qqo8592p38 (H100 kernels), 9qx6yvvqsm4fm5 (B200 kernels), xlei1f70cz6zuw (B200 E2E), 3l1v20bt2op1ej (H100 E2E), both US-CA-2 on the US weights volume, all deleted; GPU spend $5.08; artifacts/perf/wp-d/; docs/perf/datacenter-profile.md "WP-D results"

### FVID · 2026-09-29 · FVID-2026-09-29-h3-1080p-length
- Trigger: owner decision list (2026-09-29), item 1
- Options: serve native 1080P over the whole 4..15 s grid; cap it at 5 s; cap at 5 s with 10 s behind a switch
- Decision: **H3 native 1080P is 5 s at most by default; up to 10 s is the experimental feature `h3_1080p_long`**, toggled on the console's admin page (Experimental features), stored in D1 `feature_flags`, applied to the caps at negotiation (gateway or single server), reflected in the fal form (`x-fv-max-by-resolution`) and the director's 1080p chunks
- Reason: 10 s at 1080P needs ~47 GiB of working memory and was validated on two cells only
- Reversibility: cheap (flag; the caps' `max_frames`)
- Executed by: Executor
- ADR: none
- Verification: `crates/fastvideo-protocol/tests/negotiate.rs`, `crates/fastvideo-serve/tests/flags.rs`, `tests/dimension_sweep.rs` (flag off and on), console smoke; docs/serve/console.md §7, docs/serve/h3-1080p-and-upscaler.md

### FVID · 2026-09-29 · FVID-2026-09-29-owner-decisions
- Trigger: owner decision list (2026-09-29), items 2, 3, 5
- Decision: (2) **over-length LTX clips run shorter** (the longest clip that fits, `Snap::Nearest`, as since `401a089`), not refused; (3) **inline inputs are the default for serverless pools** (`inline_inputs_max_bytes`, 6 MiB there); (5) **a worker refuses to report ready on a GPU that cannot run its model** (startup capability check: compute capability, FP8 / NVFP4, memory; the model is failed with the reason in `/fv/v1/status`)
- Reversibility: cheap
- Executed by: Executor
- ADR: none
- Verification: docs/serve/fal-parity.md, docs/serve/gateway.md §3.1 and §4; `crates/fastvideo-engine-service/src/device.rs` tests, `crates/fastvideo-serve/tests/flags.rs` (simulated A100)

### FVID · 2026-09-28 · FVID-2026-09-28-sfwan-stream-cap
- Trigger: the R12 study (`docs/ports/wan.md`) rates a single-prompt SF-Wan stream (sink 15) clean for 2 min and usable to 5 min with a transient top-edge strip; owner: "yes cap SF-Wan streams"
- Options: leave `max_seconds` to each request; cap per front-end; one shared cap for every live causal session, with or without a reset restarting the clock
- Decision: **every live causal session has a limit**: request `max_seconds`, else `[streams] causal_default_max_s` = 120, clamped to `causal_hard_max_s` = 300 (env `FV_CAUSAL_DEFAULT_MAX_S` / `FV_CAUSAL_HARD_MAX_S`; 0 refused). Video time from the first frame; **a `reset` restarts the clock, the session never exceeds the hard ceiling in all**; a kept prompt switch does not restart it. Native streams end `session_limit`, Reactor `session_ended{reason}`; the fal director is clip-only (causal models refused), so it is unaffected. Advertised in `/fv/v1/capabilities` (`stream_limits`), the Reactor schema (`x-reactor.session_limits`) and the console
- Reason: the study's numbers; the reset rule follows from "a reset renews the anchor" while the ceiling bounds the total session simply
- Reversibility: cheap (config)
- Executed by: Executor
- ADR: none
- Verification: fake-engine tests (paused tokio clock: default ends at 120 s, 30/200 s honoured, 3600 clamped to 300; real time: resets extend to the ceiling; native end_reason and Reactor session_ended); `docs/serve/design.md` §5.2/§5.4

### FVID · 2026-09-27 · FVID-2026-09-27-attention-sm120
- Trigger: dense attention 5-6% behind cuDNN-via-torch on sm_120 (H3 768p 109.3 vs 103.0 ms, LTX 1080p20s 680.1 vs 647.0, 4K 752.1 vs 704.0); our cuDNN path never ran (cudarc 0.17.8 has no unified SDPA / softmax bindings); our VSA never compared with FastVideo's
- Options: bind cuDNN's unified SDPA node ourselves; tune flash_mma_fwd2 (software-pipelined QK, 3-stage ring, rescale skipping); both, winner as sm_120 default with escape hatches
- Decision: **sm_12x `auto` = cuDNN unified SDPA node, picked per shape by a one-time timing against flash_mma_fwd2** (`attn.rs` `auto_cudnn_or_v2`; bf16 out only; `FASTVIDEO_FLASH_KERNEL=v2|cudnn` fix the kernel). Raw FFI in `cudnn_sdpa::raw` (OPERATION_SDPA_FWD=41, attrs 2800-2805; OPERATION_SOFTMAX=45, attrs 3100-3101; cuDNN 9.26 header values). `flash_mma_fwd3` / `fwd3s` kept opt-in (`v3|v3s`). VSA: `vsa_topk2` (radix select) default (`FASTVIDEO_VSA_TOPK=v1` escape); H3 bf16-in-place VSA inputs (`FASTVIDEO_VSA_BF16`, bit-identical)
- Reason: `attn3_bench` (2 pods): cuDNN unified cfg0 (engine 11) 104.8-105.0 vs fwd2 108.4-109.1 ms at H3 768p, 658.8-659.8 vs 682.2-682.7 at LTX 1080p20s, but 766.6-769.0 vs 757.2-759.9 at 4K 5s (hence per-shape). fwd3 bit-identical but 3% slower (255 regs, ptxas does not interleave), fwd3s 0.5% better than fwd3 only. Composite-softmax graph plans on engine 1 but returns inf/NaN: rejected. VSA fine stage ours 1.4-1.7x faster than FastVideo's Triton; top-k was 4-8x slower (bisection), coarse ~1.4x slower
- Reversibility: cheap (env flags; fwd2 / vsa_topk kept)
- Executed by: Executor
- ADR: none
- Verification: `attn3_parity` on GPU: fwd3 bit-exact vs fwd2 (11 shapes, f32+bf16 out), fwd3s within 1 bf16 ulp, cuDNN unified vs f32 reference rel_l2 <= 4.3e-3 on every shape; `vsa_b16_*` bit-exact; generation A/B (attn3 family, RTX PRO 6000): FastH3 4-step dense 768p denoise 33.23 -> 32.28 s (8.30 -> 8.07 s/step; frames PSNR 20.4 dB vs fwd2, the known H3 sensitivity to 1-ulp bf16 changes), LTX-2.5 1080p20s dense stage 2 44.0 -> 42.7 s/step (PSNR 32.7 dB); FastH3 VSA frames identical with FASTVIDEO_VSA_BF16=0/1 (path not engaged there: inputs arrive f32); `vsa_topk2` bit-exact vs `vsa_topk` on real and tied scores, 1.78 -> 0.27 ms at FastH3 768p; `auto` pod: H3 768p 103.7 vs fwd2 106.6 ms, LTX 1080p20s 650.3 vs 672.1, 4K picks fwd2. VSA op vs FastVideo (ms, 0.9 / 0.8): 768p 26.8 / 41.8 vs 27.8 / 48.4, 480p 7.3 / 10.1 vs 6.4 / 10.3 (theirs without its 3.7 ms tiling), FastWan 8.2 vs 10.4. Spend ~$4.6 (7 pods, all deleted). Details: `docs/gaps/2026-09-27-attention-sm120-cudnn-vsa.md`; artifacts/runpod/{rtx6000,attn3,upstream}

### FVID · 2026-09-27 · FVID-2026-09-27-attention-datacenter-sm90
- Trigger: balance topped up; run the Rust-path `attn_dc` group on B200, the same plus `bench:attn_dc bench:attn` on H100, make the sm_90 kernel automatic if it passes and beats V2
- Options: keep dc90 opt-in; make it `auto` on 9.0; flip B200 back to V2 if its group failed
- Decision: `auto` takes the attn_dc kernel on 9.0 and 10.0 (d = 128); B200 default kept; the gpucheck f32 bound becomes "1e-2, or no worse than V2 +5%"
- Reason: `fv-gpucheck kernels --groups attn_dc,attn_bench` (image sha-4bc3ec3, cubins loaded) passes every check but one on both GPUs: B200 and H100 fail only `attn_dc_1x1x513x1000_vs_f32` at rel L2 1.053e-2 (limit 1e-2), a case with 4x-scaled inputs whose error is bf16 input rounding against the unrounded f32 reference (dc vs V2 there 6.6e-5, identical on both GPUs), so the bound, not the kernel, was wrong. attn_bench dense (TFLOPS, V2 -> dc): B200 369 -> 1173 H3 768p, 372 -> 1148 1080p20s, 373 -> 1152 4K; H100 SXM 303 -> 616, 326 -> 565, 321 -> 570. H100 harness: dc90 26/26 parity (same numbers as dc100), 642 / 649 / 645 / 619 TFLOPS vs V2 336 / 315 / 322 / 323; torch cuDNN SDPA 598 / 675 / 587 / 632 (dc90 ahead at H3 768p and 1080p20s); sol-engine sm90 Sol tau 1.0 15.4 / 0.52 / 120.7 / 139.0 ms (kv_splits 4: 99.4 ms at 1080p20s, 106.9 at 4K). Our Sol x4f on B200 is 146 ms at 1080p20s tau 1.0 vs sol-engine 87.8: the next target
- Reversibility: cheap (`FASTVIDEO_FLASH_KERNEL=v2`)
- Executed by: Executor
- ADR: none
- Verification: pods havhdoj7e5cv4c (B200 kernels), la1etadov0yz8c (H100 harness + bench_attn), 18rfe79oxg4f4d (H100 kernels), all volume-less secure and deleted; artifacts/runpod/dc/, artifacts/runpod/upstream/attn-dc-h100/
  - B200 generation check (pod feuynzq6ifdjb4, US-CA-2, fv-weights-b200-us, image sha-4bc3ec3 with dc as the B200 default, warm, TAEH3; stopped after four cells, ~19 min): fasth3-8step denoise 31.7 s (earlier B200 suite 49.1 s), fasth3-4step-vsa 12.3 s (21.4 s), fasth3-4step-dense 12.3 s / 3.08 s per step (116.2 s), sol-h3 12.3 s / 3.08 s per step (824 s). The sol-h3 recipe in this build logs `techniques: none ... dense=true`: it no longer routes through the Sol kernel, so it is not a Sol-H3 measurement, and the earlier suite ran an older image, so these ratios are not dc alone. `runpod-matrix.sh` b200 cells ignored FV_CELLS (run_cell now honours it). artifacts/runpod/b200-dc/

### FVID · 2026-09-27 · FVID-2026-09-27-ltx-nvfp4-ffn-cublaslt
- Trigger: `FASTVIDEO_NVFP4` dequantized to bf16 at load (no speedup); our scalar W4A4 kernel was 38x slower than bf16; sol-engine runs TE's NVFP4 GEMM on the LTX video FFN (`nvfp4_ffn.py`)
- Options: oxide Tile-IR cubins (1.4-2.3x bf16 at the FFN shapes); cuBLASLt block-scaled FP4 (`VEC16_UE4M3`); a hand-written CUTLASS-style kernel; load sol-engine's pre-quantized RTX5090 checkpoint
- Decision: **cuBLASLt block-scaled FP4 on the LTX-2 video FFN only** (`wan/nvfp4_linear.rs`): weights quantized at load and activations per call with the TE `static_6` rule straight into cuBLASLt's layout (scales pre-swizzled, rows padded to 16), GELU fused into the down projection's quantizer, alpha on the device (pointer mode device), bias epilogue, RHT / SR off, bf16 fallback below sm_100 or for `mse`. An LTX process takes `FASTVIDEO_NVFP4` as this scope only (LongLive dequant / K/V path off there). Profile `ltx2/ltx25_distill_sol_nvfp4`, default off. Pre-quantized checkpoint not loaded: it is not on `fv-weights-h3-ltx-hy`.
- Reason: RTX PRO 6000 cuBLASLt NVFP4 GEMM 3.5-4.0x bf16 (1.48-1.66 PFLOPS) vs oxide 1.4-2.3x; whole linear incl. activation quantize 2.7-3.2x the bf16 linear; codes / scales bit-identical to the host TE reference, GEMM at bf16-rounding distance from the exact dequantized math
- Reversibility: cheap — profile / env opt-in; bf16 path unchanged when off
- Executed by: Executor
- ADR: none
- Verification: **kernels pass** (`fv-gpucheck kernels`, groups `nvfp4_gemm` / `nvfp4_linear`, runs `36ce5a2-09271149`, `600c38f-09271221`): operand quantizer 0 mismatches (plain, GELU), linear rel_l2 1.66e-3 vs exact, 37.8 dB vs bf16 at the FFN up shape. Generations (`runpod-matrix.sh precision`, `FV_PRECISION_ARM=ltx-nvfp4`, run `51adbb3-09271342`, RTX PRO 6000, warm, Sol stage 2): 4k5s denoise 125.7 -> 110.7 s (1.135x; stage 1 45.8 -> 40.1, stage 2 79.9 -> 70.6), total 209.9 -> 186.5 s, peak 55.2 -> 46.6 GiB, LPIPS 0.193, PSNR 19.1 dB, sharpness 1.098 -> gate **fail** (quantitative_quality, sharpness > 1.08); 1080p20s denoise 118.4 -> 105.5 s (1.122x; stage 1 43.2 -> 38.2, stage 2 75.2 -> 67.2), total 195.0 -> 183.3 s, peak 54.9 -> 46.3 GiB, LPIPS 0.254, PSNR 18.3 dB, sharpness 0.994 -> gate **pass**. Stays default off. Down-projection linear incl. GELU 2.24-2.34x bf16 (run `8f12d44-09271258`). Fix found by the first generation: the LTX-2.5 video FFN has no bias (562ec95).

### FVID · 2026-09-27 · FVID-2026-09-27-attention-datacenter
- Trigger: Sol-H3 on B200 gained only 1.30x over RTX PRO 6000 (cuBLAS / cuDNN parts ~2x): every attention kernel is mma.sync; asked for tcgen05 / TMEM / TMA (sm_100) and wgmma / TMA (sm_90) kernels, dense first, then Sol, VSA, block-causal
- Options: hand-written tcgen05 + TMEM FA4-style kernel in a new file with arch-specific cubins; CUTLASS / CuTe C++ (new dependency, not NVRTC-able); cuDNN SDPA only (binding still rejects the graph, other agent's work); keep mma.sync on B200
- Decision: new `attn_dc.cu` (separate module; build.rs compiles it for sm_90a and sm_100a only; loaded only on 9.0 / 10.0), `FlashKernel::Dc` / seam value `dc`; `auto` takes it on 10.0 only (sm_90 wgmma kernel built but unmeasured -> opt-in); `FASTVIDEO_FLASH_KERNEL=v2` restores mma.sync; sm_120 path unchanged. Sol / VSA / block-causal datacenter kernels not started
- Reason: B200 `attn_dc_bench` (pod-built harness): dc100 passes all 13 parity cases (vs flash_mma_fwd2 rel L2 <= 7.7e-4, error vs f64 equal to V2's, bf16 out = RNE(f32 out), component probes exact) and runs 1092 / 964 / 1102 / 1109 TFLOPS at H3 768p / LTX 512p / 1080p20s / 4K (V2: 372 / 347 / 374 / 374; cuDNN SDPA 1490 / 1350 / 1401 / 1405; the N=64 P.V variant ~8% slower). Runpod balance fell to ~$10 (other agents' pods at $17-24/hr) and the coordinator froze pod creation, so the H100 run, the Rust-path `attn_dc` group, and the B200 generation check did not run
- Reversibility: cheap (env / profile seam; old kernels kept; module isolated)
- Executed by: Executor
- ADR: none
- Verification: B200 pod 3dxiud5gyc8hnr (secure $6.79/hr, 12:18-12:36Z, ~$2.0, deleted): harness parity + bench, `bench_attn.py` (torch 2.13 cuDNN, sol-engine 6c2f582 sm100 Sol: tau 1.0 15.3 ms H3 768p, 87.8 ms 1080p20s, 95.7 ms 4K). Host: `cargo check` cuda features, techniques / attn unit tests green. Pending on hardware: `fv-gpucheck kernels --groups attn_dc,attn_bench` on B200 and H100 (`RUNPOD_NO_VOLUME=1 FV_EXTRA_ENV=FV_KERNEL_GROUPS=attn_dc,attn_bench runpod-http.sh kernels <sha>`), dc90 parity / timing (`UP_STEPS=bench:attn_dc bench:attn`), FastH3 8-step + Sol-H3 on B200
  - Results: artifacts/runpod/upstream/attn-dc-b200/

### FVID · 2026-09-26 · FVID-2026-09-26-attention-phase3b
- Trigger: LTX-2.5 1080p 20 s Sol stage 2 1.15x slower than sol-engine; dense flash the largest H3 kernel; VSA 1.34x vs FastVideo's 1.49x
- Options: warp-specialised Sol + KV splits (as asked); mirror sol-engine's sm120 kernel (`sol_attn/sm120/mainloop.py`: 4 MMA warps, warp 0 issues TMA, STAGES=1, kv_splits=1 — splits are sm90-only, `interface.py:114-116`); cuDNN fused SDPA for dense; FA2-style 128-query dense; smaller VSA smem
- Decision: Sol default `sol_mma_fwd_x4f` (ldmatrix.x4 + `ex2.approx.ftz`, CuTe's fastmath exp2); warp-specialised `sol_mma_fwd2` with KV splits kept opt-in (`FASTVIDEO_SOL_KERNEL=ws`, `FASTVIDEO_SOL_SPLITS`); dense default `flash_mma_fwd2` where its 128-query grid fills >= 16 waves; VSA default `vsa_mma_attn_tma2` (3-slot ring, 48 KB); cuDNN SDPA opt-in (`FASTVIDEO_FLASH_KERNEL=cudnn`) but falls back
- Reason: RTX PRO 6000 `attn_bench`: x4f bit-identical to v1 and 7-8% faster (1080p20s tau 1.0 136.9 vs 147.8 ms; sol-engine 135.5 ms); ws bit-identical but ~10% slower (fifth warp caps registers at 200), splits slower still (grid is already 165 waves); dense v2 bit-identical, 373 vs 358 TFLOPS (cuDNN via torch 392-397); VSA ring bit-identical, 1.25x; cuDNN 9.26 rejects the composite SDPA graph ("non-flash composite MHA fprop is no longer supported"), needs the unified softmax node cudarc does not bind
- Reversibility: cheap (env flags; old kernels kept)
- Executed by: Executor
- ADR: none
- Verification: `attn2_parity` bit-exact on GPU; generations (traced): LTX 1080p20s Sol stage 2 75.9 -> 66.3 s (sol-engine 73.7 s, incl. concurrent DiT fusions), FastH3 4-step dense denoise 34.6 -> 33.1 s; artifacts/runpod/{trace,rtx6000,upstream}

### FVID · 2026-09-25 · FVID-2026-09-25-b200-warm-suite
- Trigger: warm H3 / FastH3 / LTX architectural-parity suite on 1× B200 NOW; existing US-CA-2 volume only; pinned image `build-ec5a6cc0988aca26`
- Options: US-CA-2 B200 + volume `s2k01690bi` (528G TAEH3 + Preview LoRA); fall back to H200 / PRO 6000 if stock gone; wait / invent numbers
- Decision: **rent 1× NVIDIA B200** in **US-CA-2 only**, attach existing `s2k01690bi`. Pin DC. Do not create or delete volumes (`jg48s6o1w0` unused/kept). Run `runpod-matrix.sh b200` (`--warm`, TAEH3, 3600s cell cap). Oxide GEMM unset. Reap GPU after pull. No H200/PRO 6000 fallback.
- Reason: stock just probed US-CA-2 B200 Low $5.98/hr under $20; published FastH3 Preview v1 is **16.2s warm E2E on 1× B200**; H200 same-volume 4step-vsa was **35.0s / 7.0 s/step**
- Reversibility: cheap — GPU destroyed; both volumes kept; no script commit (rsync `runpod-matrix.sh` only)
- Executed by: Executor
- ADR: none
- Verification: driver **595.91.07**, `box_build_id=ec5a6cc0988aca26`, **Cubin(100)** / sm_100 / compute_cap 10.0. Image stayed `ghcr.io/zaitrarrio/fastvideo-rs-runtime:build-ec5a6cc0988aca26` (newer main CI 36080171264 rebuilt the same build-id). Artifacts `artifacts/runpod/b200/runs/`. GPU reaped; volumes `s2k01690bi` + `jg48s6o1w0` still present.
  - Offer: `NVIDIA B200` US-CA-2 Low **$5.98/hr**; billed **$6.79/hr**. First pod `l5h1a35e4bdala` (05:42:16Z) was destroyed by a local cost-watchdog using a 2025 epoch; re-probe still Low $5.98; recreated `sd0nc9o8atcwdy` (`fv-gpu-b200-20260925054906`, same host 38.80.152.146:30236). Cost ≈ **$10.2** (first ~6.4 min + second 05:49:07–07:13:20Z × $6.79/hr). EUR volume unused.
  - **fasth3-4step-vsa** (headline): ok 462s. load 404.6s. denoise **21.4s**. generate **26.5s**. steps **5.33 / 5.34 / 5.33 / 5.41**. decode 2.76s. peak 108173 MiB. TAEH3. vs published FastH3 **16.2s B200 E2E**: 26.5s is **1.64×**. vs H200 same-volume **35.0s / 7.0 s/step**: **0.76×** (1.32× faster).
  - **fasth3-8step**: ok 356s. load 243s. denoise 49.1s. generate **54.3s**. steps **~6.14s ×8**. peak 107661 MiB. vs H200 70.0s / 8.02 s/step.
  - **fasth3-4step-dense**: ok 534s. load 289s. denoise 116.2s. generate **120.8s**. steps **~29.1s ×4**. peak 104941 MiB. vs H200 164.4s / 39.6 s/step.
  - **sol-h3**: ok 2080s. load 415s. denoise 824.4s. generate **830.0s**. steps **28.9 / 324.4 / 261.7 / 209.4** (sol-attn kernel). peak 107615 MiB. vs H200 1364s (39.6 / 543 / 433 / 341).
  - **sol-h3-spark**: FAIL 433s exit=2. VSA denoise **5.5 / 5.4 / 5.3 / 5.3 s/step** then `h3 adapter: only model_type "tiny" is published, got null`.
  - **ltx25-two-stage**: ok 522s. warmup 423s. measured ancestral **~0.66–0.69s**, refine **1.20 / 1.20 / 1.16s**. 121 frames + wav + mp4.
  - **ltx23-two-stage**: FAIL 200s exit=2. `missing weight key: decoder.mid_block.resnets.0.conv1.conv.weight`.
  - **ltx20-distilled-8step**: ok 369s. warmup 129s. warm steps **0.67s ×8**. vs H200 0.96 / 0.87×7; vs Phase 3 PRO 6000 **1.47 s/step**.
  - Skipped: none of TAEH3 / Preview LoRA (both on US volume). Oxide GEMM off (`gemm=Bf16`). No family restart (b200 driver is one process).

### FVID · 2026-09-24 · FVID-2026-09-24-h200-warm-suite
- Trigger: warm H3 / FastH3 / LTX architectural-parity suite on 1× H200; existing volume only; pinned image `build-ec5a6cc0988aca26`
- Options: US-CA-2 H200 SXM + volume `s2k01690bi` (TAEH3 + Preview LoRA); EUR-IS-1 H200 + `jg48s6o1w0` (no H200 stock); wait / invent numbers / rent B200 or PRO 6000
- Decision: **rent 1× NVIDIA H200** (`hr6x1l507yzk1x`, US-CA-2, offer $3.59 / billed **$4.59/hr**) on existing volume `s2k01690bi`. Pin DC. Do not create or delete volumes (`jg48s6o1w0` kept). Run `runpod-matrix.sh b200` (`--warm`, TAEH3, 3600s cell cap). Oxide GEMM unset. Reap GPU after pull.
- Reason: first in-stock H200 under $20 was US-CA-2 Low; that DC already has the preferred 528G tree
- Reversibility: cheap — GPU destroyed; both volumes kept; no script commit (rsync only)
- Executed by: Executor
- ADR: none
- Verification: driver **580.126.20** (CUDA 13.0), `box_build_id=ec5a6cc0988aca26`, **Cubin(90)** / sm_90. Artifacts `artifacts/runpod/h200/runs/`. Cost ≈ **$9.10** (02:57:28–04:56Z × $4.59/hr). GPU reaped; volumes `s2k01690bi` + `jg48s6o1w0` still present.
  - Offers: `NVIDIA H200` US-CA-2 Low $3.59; EUR-IS-1 none; H200 NVL both DCs none. Did not rent B200 / PRO 6000.
  - **fasth3-4step-vsa**: ok 675s. load 601.8s. denoise **28.0s**. generate **35.0s**. steps **7.01 / 7.00 / 7.00 / 7.02**. peak 106654 MiB. TAEH3. vs published FastH3 **16.2s B200 E2E**: 35.0s is **2.16×**. vs prior H200 Preview **7.7 s/step**: **7.0 s/step** (~0.91×).
  - **fasth3-8step**: ok 394s. load 251s. denoise 64.2s. generate 70.0s. steps **~8.02s ×8**. peak 107742 MiB.
  - **fasth3-4step-dense**: ok 683s. load 352s. denoise 158.4s. generate 164.4s. steps **~39.6s ×4**. peak 100080 MiB.
  - **sol-h3**: ok 3128s. load 373s. denoise 1356.9s. generate 1363.9s. steps **39.6 / 543.3 / 433.0 / 341.0** (sol-attn kernel). peak 106814 MiB. Fused Preview LoRA dense-datafree 362 pairs.
  - **sol-h3-spark**: FAIL 426s exit=2. VSA 0.9 denoise **7.0–7.1 s/step** then `h3 adapter: only model_type "tiny" is published, got null`.
  - **ltx25-two-stage**: ok 527s. warmup 417s. measured ancestral **~0.67–0.83s**, refine **~1.39–1.44s**. 121 frames + wav + mp4.
  - **ltx23-two-stage**: FAIL 174s exit=2. `missing weight key: decoder.mid_block.resnets.0.conv1.conv.weight`.
  - **ltx20-distilled-8step**: ok 357s. warmup 127s. steps **0.96 / 0.87×7**. vs Phase 3 PRO 6000 **1.47 s/step**.
  - Skipped: none of TAEH3 / Preview LoRA (both on US volume). No family restart (b200 driver is one process).

### FVID · 2026-09-24 · FVID-2026-09-24-b200-arch-parity
- Trigger: user wants architectural parity on B200 for H3 / FastH3 / LTX, then later fine-tune RTX 6000
- Options: raise `RUNPOD_GPU_MAX_DPH` to $20 and create a new US volume for B200; reuse EUR-IS-1 `jg48s6o1w0` (cannot attach to US B200); discard prior PRO 6000 / 8-step / official-VAE work
- Decision: raise `RUNPOD_GPU_MAX_DPH` to $20; create a NEW US network volume (do NOT reuse EUR-IS-1 `jg48s6o1w0`; keep that volume); implement B200-aligned recipes/architecture; run warm B200 gens; do not discard prior PRO 6000 / 8-step / official-VAE work
- Reason: published FastH3 16.2 s is 1× B200 warm Preview v1 4-step 90%; EUR volume cannot attach to US B200
- Reversibility: cheap
- Executed by: Executor
- ADR: none
- Verification: host gates green (`fastvideo-models` 299 / `fastvideo-cudarc` 352, 0 ignored). Runtime image `build-ec5a6cc0988aca26` (`sha256:428e3c779f81be2e29aebcfc17dbfccd5911a81e9d7bc2aa2e849f718f93abbf`, CI 36077955888). US volume `s2k01690bi` fetched. B200 warm numbers pending — Runpod stock none under $20 after US DC probe.

### FVID · 2026-09-24 · FVID-2026-09-24-phase3-failure-gate
- Trigger: Phase 3 cells that failed on `build-af5edf649671bfb7` (Spark `.set_weight`, H3 gate, LTX-2.5 decode OOM, LTX-2.3 keyframes, Hunyuan `Hunyuan15.*`, Wan 1.3B absent) plus host contracts and a weights manifest
- Options: skip failing cells / delete tests to go green; one image then one cheap smoke then one PRO 6000 with family restarts; invent SANA-Video / Super-64B / enable oxide GEMM
- Decision: **host tests first, then image `build-44a9e9db1321e7dc`, then fetch/verify on `jg48s6o1w0` (incl. `fastwan21-1.3b`), then Blackwell smoke, then PRO 6000 matrix with destroy/recreate between H3 → LTX → Hunyuan → Wan.** Oxide GEMM stays off (no H3 FFN bench + PSNR ≥ 30 dB this pass). pliron stays out. Shared H3 `--text-cache` / `--text-weights`; no `--warm` / `--no-text-cache` on H3.
- Measured vs Phase 0 **75.4 s/step** and Phase 3 **10.7 s** (H3) / **1.47 s** (LTX-2.0) on the same-volume PRO 6000 (driver 595.91.07, $2.09/hr):
  - **fasth3-8step**: load 142 s (Gemma streamed 89 s), steps 1–6 at **10.8 s** (matches 10.7 s). Wall 300 s / exit 124 before decode.
  - **sol-h3-4step**: `vsa=0 dense=false` (gate off — load contract fixed). Load 243 s. Wall 301 s / exit 124 (no step times; streamed Gemma + DiT still exceed 5 min).
  - **sol-h3-spark**: **loads**. Recipe `vsa=0.9` + FastH3_VSA_DataFree (no more 50-replacement reject). Load 265 s, steps 1–3 at **9.8 s**. Wall 300 s / exit 124.
  - **ltx25-two-stage**: ancestral **0.39 s**, stage-2 **1.67 s** (was 1.67 s then OOM). This pass: denoise finished, **no OOM** in the cell log; `--warm` + 300 s wall (exit 124) before decode proof.
  - **ltx23-two-stage**: keyframes miss **gone** (DiT 48/48). New miss: `prompt_adaln` vs LtxCore `prompt_adaln_single`. Rename landed in tree; not in this image.
  - **ltx20-distilled-8step**: **ok 256 s**, **1.47 s/step** (matches Phase 3).
  - **hy15-480-t2v**: still `Hunyuan15.double_blocks.0.img_mod.linear.weight`. Volume is Diffusers MMDiT (`transformer_blocks.0.attn.to_q`, no `img_mod`). Prefix/block aliases are not enough.
  - **wan13-dmd-3step**: **ok**. Embed 65 s; clip 17 s; denoise **1.20 s** (510/322/321 ms); decode 1.12 s; 17 frames 480×832; quality gates pass. First matrix attempt used a directory as `--embeds` (fixed in `runpod-matrix.sh`).
- Smoke (same image, PRO 6000): NVRTC all SMs PASS; kernels `Cubin(120)`.
- Reason: measure the loader fixes on GPU without skipping cells; keep the 5-min cap; do not enable unpublished kernels
- Reversibility: cheap — volume kept; GPU pods destroyed; oxide flag still off
- Executed by: Executor
- ADR: none
- Verification: `artifacts/runpod/phase3-gate/runs/`. Image CI 36042414508. Host `cargo test -p fastvideo-models --lib` 299 / `fastvideo-cudarc --lib` 352, 0 ignored. Cost ≈ $2.5 matrix + smoke. Volume `jg48s6o1w0` stays.

### FVID · 2026-09-24 · FVID-2026-09-24-phase3-pro6000-sweep
- Trigger: Phase 3 — rebuild the CUDA 13.4 runtime image and run the H3 / LTX / Hunyuan / Wan matrix (5-min cell cap) against Phase 0 numbers
- Options: wait for three PRO 6000s in EUR-IS-1; fall back to A100 80GB on the ready volume; sequential families on one Blackwell
- Decision: **one RTX PRO 6000 96 GB** (`r009p7bcmwfu8k`, EUR-IS-1, $2.09/hr, driver 595.91.07) on volume `jg48s6o1w0`. Image `build-af5edf649671bfb7` (CUDA 13.4 libs, `CUDARC_CUDA_VERSION=13000` — cudarc 0.17.8 rejects 13040). A100 570.195 panicked on `cuCtxGetDevice_v2` and was destroyed. Wan skipped (no 1.3B tree). Families ran sequentially.
- Measured vs Phase 0 same-volume PRO 6000 (`sol-h3-4step` 01:33Z: **75.4 s/step**, peak 60181 MiB, 615 s wall):
  - **fasth3-8step-warm**: steps 1–7 at **10.7 s** (~7×). Load 130.5 s. Wall-cap 301 s / exit 124 during step 8 / decode. Peak ~63.8 GiB mid-run.
  - **sol-h3-4step**: load refused `to_gate_compress` on MiniMax-H3 (recipe now `dense=false` / Sol-Attn). `--dense` fused 362-pair dense-datafree and loaded (dit 176.6 s, total load 217 s) then hit the 300 s cap during streamed text (no step times). Gate load now keys off `vsa_sparsity > 0`.
  - **sol-h3-spark**: refuse. FastH3_VSA_DataFree is 50 `.set_weight` replacements; `plan_from_keys` still requires dense-datafree LoRA.
  - **ltx25-two-stage**: stage-1 ancestral **0.42 s/step**, stage-2 Euler **1.67 s/step**, then VAE **OOM**. 246 s.
  - **ltx23-two-stage**: missing `keyframes_abs_pos_embedding` (2.5 key on 2.3 weights). 109 s.
  - **ltx20-distilled-8step**: **ok 236 s**, **1.47 s/step**, mp4 356 KiB.
  - **hy15-480-t2v**: Diffusers tree; loader wants remapped `Hunyuan15.double_blocks.*`.
- Reason: 5-minute cells measure denoise when load is short; cold H3+streamed Gemma does not finish. FastH3 step time is the speedup the kernels were for. Remaining fails are loader contracts, not kernel panics.
- Reversibility: cheap — image `:latest` smokes after `-INFINITY` → `-1e30f`; volume kept; pod destroyed after the sweep
- Executed by: Executor
- ADR: none
- Verification: `artifacts/runpod/phase3/runs/`. Smoke `gpucheck-runtime-image` 36034088206 success. Cost ≈ $1.1.

### FVID · 2026-09-24 · FVID-2026-09-24-bf16-activations
- Trigger: WS-I — DiT block activations still live as f32; H3 denoise is ~80% of a warm clip and the reference stores the stream as bf16
- Options: keep residual + activations f32; bf16 activations with residual f32; bf16 activations and residual
- Decision: **opt-in `FASTVIDEO_BF16_ACT=1`** (default off). `CudaTensor` is `f32 | bf16` (host oracle stays `Vec<f32>` rounded through `half::bf16`). Linears take/return bf16; rms/layer-norm and SDPA run on bf16 values with f32 accumulation. **Residual is bf16** — host one-block PSNR vs the f32 path: H3 **57.46 dB**, LTX-2.5 **51.64 dB**, Wan **51.12 dB** (gate ≥ 35 dB). `FASTVIDEO_BF16` (cuBLAS compute type) is unchanged.
- Reason: residual-as-bf16 cleared the 35 dB host gate on all three tiny blocks, so the stream can stay in the activation dtype
- Reversibility: cheap — flag defaults off; today's f32 path is bit-identical
- Executed by: WS-I
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-cudarc --lib --offline` — 342 passed. GPU H3 step time **unmeasured** (no GPU / no nvcc on this Mac).

### FVID · 2026-09-24 · FVID-2026-09-24-cosmos3-super-scaffold
- Trigger: WS-J — 64B T2V Cosmos3-Super has no crate; Predict2 is EDM Video2World; Super needs FlowMatch + single-GPU NVFP4/FP8; upstream serves 4-GPU sequence parallel
- Options: extend the Predict2 `cosmos` crate; new `cosmos3` scaffold with published canvas/TeaCache and `TODO(upstream)` DiT dims; invent Super widths from unpublished Predict2.5 notes
- Decision: **new `cosmos3` host + cudarc scaffold**. Reuse Predict2 TeaCache 1.15 / start 10 / max 3 (`cosmos::sol`). Official canvas 1280×720 / 189f / 35 / guidance 6 / fps 24. FlowMatch applies recorded shift 10 (Predict2 EDM still does not). Super DiT / Hub / text / VAE marked `TODO(upstream)`. Multi-GPU SP out of scope. Registry/CLI/gpucheck skipped (no published Hub id). Single-GPU plan is `FASTVIDEO_NVFP4=1` → TE `static_6` plus `FASTVIDEO_NVFP4_COSMOS_STEPS`.
- Reason: do not invent unpublished 64B widths; one 96 GB PRO 6000 is a quantized-weight path, not 4-way SP
- Reversibility: cheap — new modules only; Predict2 files untouched
- Executed by: WS-J
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib --offline cosmos3`; `cargo test -p fastvideo-cudarc --lib --offline cosmos3`. No GPU rented.

### FVID · 2026-09-24 · FVID-2026-09-24-nvfp4-te-static6-oxide
- Trigger: default NVFP4 rule was FourOverSix MSE; host fake-quant in `dequant_beforehand`; no Tile-IR GEMM crate; runtime image still on CUDA 13.0
- Options: keep MSE as `FASTVIDEO_NVFP4=1`; switch the on-switch to TransformerEngine `NVFP4BlockScaling` (`static_6`) and leave MSE as an explicit opt-in; ship an ungated oxide GEMM
- Decision: **`FASTVIDEO_NVFP4=1` → `static_6`**. `mse` / `4o6` / `fouroversix` stay FourOverSix. Device `nvfp4_reconstruct` replaces host fake-quant when a CUDA context is live; `host_algorithm` remains on the CPU path. Models-side [`scope_rule`] + `FASTVIDEO_NVFP4_LTX_VIDEO_FFN` / `FASTVIDEO_NVFP4_COSMOS_STEPS` for LTX-2.3 video FFN and Cosmos middle steps (transformers not edited). New excluded crate `fastvideo-oxide-kernels` (nightly-2026-04-03 / cutile nvfp4 pattern). `scripts/oxide.sh` runs `cargo oxide build --arch sm_100,sm_120`. `build.rs` embeds cubins if present. Runtime image + `cuda-13.pins` → CUDA 13.4 + tileiras. Tile-IR GEMM **off** (`FASTVIDEO_NVFP4_OXIDE_GEMM`) until it beats cuBLAS bf16 on the H3 FFN shape (K=5376, N=14336) and PSNR ≥ 30 dB vs bf16.
- Reason: TE static `amax/6` is the published NVFP4BlockScaling rule; MSE is a LongLive training heuristic. Oxide GEMM is unmeasured on this Mac.
- Reversibility: cheap — env still selects MSE; oxide cubins are optional; GEMM flag defaults off
- Executed by: WS-H
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib nvfp4`; `cargo test -p fastvideo-cudarc --lib nvfp4`. Oxide cubins not emitted on this Mac (no cargo-oxide / tileiras). GPU reconstruct untested here (no nvcc).

### FVID · 2026-09-24 · FVID-2026-09-24-llm-resident-compute
- Trigger: resident encoder compute still paid GQA `repeat_kv`, a full composed `[1,H,S,S]` score matrix, per-GEMM FP8 dequant, and a host DeepStack add
- Options: rewrite `wan/attn.rs` / `Linear`; keep streamed host fallbacks; cache dequant in the decoder layer
- Decision: **GQA via strided batched GEMM over KV-head groups** (host still `repeat_kv`). **Chunked dense SDPA** in `llm/attn.rs` reusing Wan GEMMs + mask add (no `wan/attn.rs` rewrite). **Resident FP8 dequants once** via `host::fp8_rows_*` + `Linear::from_tensors`. **DeepStack `index_select` + `add`** so `x` stays on device.
- Reason: resident prompt-after-prompt path; streamed/disk loaders unchanged; `Linear` / Wan attention stay owned elsewhere
- Reversibility: cheap — host fallbacks remain; precision enum still reports Fp8Rows
- Executed by: Executor
- ADR: none
- Verification: **gates green**. `cargo test -p fastvideo-cudarc --lib llm` — 11 passed (host).

### FVID · 2026-09-24 · FVID-2026-09-24-solh3-spark-vsa-bf16
- Trigger: sol-h3-spark Stage-1 still mapped to the dense 4-step Sol-H3 contract; one-GPU Sol-H3 was documented as dense-only / SOL-BSA multi-GPU; Spark LoRA preferred dense-datafree; refiner used the user prompt
- Options: copy upstream W8A8 FP8 after the VSA-DataFree merge; keep dense 4-step; ship Spark as VSA 0.9 / tile 64 / BF16
- Decision: **Spark Stage-1 is VSA 0.9 tile 64 + FastH3_VSA_DataFree strength 1.0 in BF16.** W8A8 FP8 stays off — the same quant measured 16–20 dB here, so it is a different picture, not a free encode. Sink tokens permute to `[visual | sinks]` with one suffix sink; `FASTVIDEO_H3_SOL_SINK=native` keeps the old multi-span. One-GPU sol-h3 is no longer dense-locked: RTX 5090 policy (10 dense steps, 2 dense layers, tau 1.0, 49 forwards on `sol-h3-rtx`). Refiner uses the upstream fixed prompt with cached Gemma; sampler is unchanged.
- Reason: match Sol-H3-Spark `configs/default.json` Stage-1 / `prompt_cache.py` without taking the quality hit we already measured on W8A8
- Reversibility: cheap — recipe names and `FASTVIDEO_H3_SOL_SINK` / `FASTVIDEO_H3_SOL_ATTN` select the old spans or the 4-step distilled path
- Executed by: WS-F
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib h3`; `cargo test -p fastvideo-cudarc --lib h3::sol h3::spark`

### FVID · 2026-09-24 · FVID-2026-09-24-wan-sol-pisa-a14b-cache
- Trigger: WS-E — A14B reused the 5B whole-stack EasyCache; EasyCache knobs ignored delivered manifests; Wan Sol/PISA routes were not wired
- Options: keep one EasyCache for every Wan SKU; split A14B to the published block-0 / tail-39 controller; silently change `official()` from 0.05/7/1 to 0.036
- Decision: **A14B cache controller** (block-0 fresh, blocks 1–39 residual, per-expert accumulators, thr 0.30 / start 5 / tail 3 / max_reuse 1). EasyCache profiles: fullstack 0.036/7/1, 14B-tuned 0.10 retain 5, **code-default 0.05/7/1** so old tests stay. Wan 14B Sol: tau 1.0, 10 dense steps, layer 0 dense, Morton3D reorder → `crate::sol_attn::sol_attn`. 5B/A14B PISA density 0.10 with the published dense layer/step sets → `crate::pisa_attn::pisa_attn`.
- Reason: match sol-engine `cache_controller.py` and the delivered tomls without silently changing existing EasyCache tests
- Reversibility: cheap — env-gated (`FASTVIDEO_WAN_SOL_CACHE`, `FASTVIDEO_WAN_EASYCACHE_PROFILE`, `FASTVIDEO_WAN_SOL_ATTN`, `FASTVIDEO_WAN_PISA`)
- Executed by: Executor
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-cudarc --lib sol_cache` (3 passed); `cargo test -p fastvideo-models --lib wan` (35 passed). GPU kernels untested here (WS-A owns the device path).

### FVID · 2026-09-24 · FVID-2026-09-24-host-loops-off-device
- Trigger: Cosmos RoPE, LingBot MoE, and H3 AdaLN still bounced every attention / token / block through pageable H2D; FVID-2026-09-18 noted 911 H2D / 3.4 GiB on a 3-step Wan clip and asked whether uploads repeat per step.
- Decision: **device kernels + one upload, then slice.** Cosmos self-attn RoPE is `rope_real` (real-interleaved twin of `rope_half`) with host `apply_rope_real` behind `host_fallback`. LingBot MoE is `softmax_last` / `sigmoid_f` + `topk_last`, then `index_select_rows` + one expert GEMM + `scatter_add_rows` (no per-token `[1,d]` GEMM). H3 `BlockMods::upload` puts the whole AdaLN ladder + keyframe table on device once and `narrow`s `[1,6,hidden]` per segment.
- Reason: the host loops were correctness-shaped leftovers; the math already existed (`softmax_last`, cuBLAS, `narrow`). Keeping the CPU twins means unit tests still run without a GPU.
- **Wan 911 H2D finding (do not edit `wan/pipeline.rs`):** RoPE (`rotary_for`) and VSA plans are already pinned/cached. The remaining per-step uploads are (1) `dit_cfg` / `dit_cfg_easy` rebuilding `CudaTensor::from_vec(vec![t], [1])` every CFG call, (2) `nn::sinusoidal_timesteps` host-building `[B, freq_dim]` every `forward_ctx` so the first Linear re-uploads it, (3) causal `forward_ctx` rebuilding and `to_device()`ing a fresh `[1,1,S,S]` mask when `cfg.causal`. `CudaTensor::dev()` also re-uploads any still-host tensor on every use (`Owned` temporary), so an unpinned table inside a 30–40 layer loop multiplies. Average 3.4 GiB / 911 ≈ 3.9 MiB/transfer matches a mid-size table (sinusoid is tiny; a causal mask or a repeated activation upload is the plausible bulk).
- Reversibility: cheap — device path is opt-in on a live CUDA context; host fallbacks stay for CPU tests.
- Executed by: WS-D
- ADR: none
- Verification: `cargo test -p fastvideo-cudarc --lib cosmos lingbot`; H3 `block_mods_narrows_the_uploaded_table`. No GPU rented.

### FVID · 2026-09-24 · FVID-2026-09-24-sol-device-partials
- Trigger: WS-A — replace Sol/PISA/SLA `host_cow` CPU paths with a device implementation (unselected blocks keep a coarse term, not block-skip)
- Options: extend `vsa_mma_attn` in place; host-only staging; **pool + cuBLAS coarse + sentinel lists + fine/coarse flash partials + LSE merge**
- Decision: device-shaped stages in `wan/sol_ops.rs` (host twins + CUDA launchers). Means reuse `vsa_tile_mean`; new `sol_tile_sum`, diag-threshold exact lists (`0xFFFFFFFF` pad), `sol_fine_partials` / `sol_mma_attn_partials` (m,l,acc), coarse remainder with `block_len` multiplicity, LSE merge. PISA reuses 1–3 with `vsa_topk` + first-order remainder kernel. SLA sparse maps when `blk_k=64` (K-smooth + top-k exact); otherwise `host_algorithm`. `host_algorithm` stays only on the no-device fallback. Multiple sink spans kept.
- Reason: Sol is training-free threshold routing plus zeroth-order reuse; VSA top-k+gate is the wrong contract
- Reversibility: cheap — host oracles unchanged; device is opt-in on a live resident CUDA context
- Executed by: Executor
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-cudarc --lib sol_attn pisa_attn sla sol_ops`. GPU kernels unrun here (no nvcc). Experiment: dense SDPA via the mma fine stage with all tiles selected is **unlikely to beat `wan/attn.rs` at this crate's typical seq lengths** (H3/LTX hundreds–few thousand tokens) — flash/mma comments already record tiled online-softmax losing to cuBLAS GEMM+softmax there. At 8s Wan (~48k) not materialising S×S scores would win on memory and maybe time; WS-K not implemented.
- Remaining host fallbacks: Sol/PISA/SLA when no live resident CUDA device; SLA when `blk_k != 64`; mma fine partials only at dim=128 / sm80+ (scalar partials otherwise)

### FVID · 2026-09-24 · FVID-2026-09-24-ltx2-sampler-correctness
- Trigger: LTX-2.5 stage 2 / Spark used ancestral Euler; distilled two-stage fused LoRA 0.8; FBCache re-armed on stage 2; SCSP claimed “steps 16–28” on a 15-step Euler loop that never reaches step 16
- Options: keep ancestral stage 2; fuse 0.8 on distilled; leave SCSP as a no-op and document Euler; implement ODE res2s (no SDE / bongmath) and index **calls**
- Decision: **stage 2 + Spark refiner = deterministic Euler** (`denoise_cfg` / `denoise`). **LoRA 0.8 (and 2.3 0.25/0.5) only on the dev BF16 DiT** — distilled two-stage does not fuse. **FBCache stage-1 only**: drop `arm_requested` on stage 2 / refiner; `begin_fbcache_step` does not re-arm after `disarm_fbcache`. **ODE res2s** on LTX-2.3 stage 1 (`lincomb3`); 15 steps → 29 calls so SCSP skips calls 16–28. Stage 2 stays 3-forward Euler for the Sol/PISA contract. Ancestral step is device `lincomb3`.
- Reason: matches upstream `distilled.py` (“Stage 2 is always deterministic”), `techniques/presets.py` call indexing, and the GB200 “stage 1 only” FBCache row
- Reversibility: cheap — sampler choice is version-gated; LoRA fuse is `use_dynamic_shifting`; SCSP still default-off
- Executed by: WS-B
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib ltx2`; `cargo test -p fastvideo-cudarc --lib ltx2`. GPU untested here.

### FVID · 2026-09-24 · FVID-2026-09-24-lora-device-refuse
- Trigger: LTX-2 / H3 strength change reloaded the DiT from disk (host `apply_bf16` / `H3LoraFuse::fuse` only)
- Options: keep host fuse + reload; keep unfused `W0` plus `(A, B)` and re-fuse on device
- Decision: **`Linear::{attach_lora,set_lora_strength}`** — snapshot `W0` on first attach; `W = W0 + s·B·A` via one GEMM + add. Host **`apply_bf16` / `fuse` stay the load-time default**. `ltx2::lora::set_strength(s)` and `H3LoraFuse::set_lora_strength` call through to attached linears. No disk, no DiT reload.
- Reason: a strength change must finish in seconds on a resident DiT
- Reversibility: cheap — loaders that never attach keep today's host fuse
- Executed by: Executor
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-cudarc --lib nn lora`. Device GEMM path untested here (no nvcc).

### FVID · 2026-09-24 · FVID-2026-09-24-phase0-strict-host-hunyuan-gen
- Trigger: review gaps — Sol/PISA/SLA/NVFP4 host paths were outside `host_fallback`; `fv-gpucheck hunyuan gen` missing from the published image; Spark VSA-DataFree LoRA not on the volume fetch list
- Options: leave host algorithms silent until WS-A; gate only under `FASTVIDEO_STRICT_DEVICE`; always refuse on a live device
- Decision: **`stats::host_algorithm`** — same refuse-on-device as `host_fallback`, plus `FASTVIDEO_STRICT_DEVICE=1` when a CUDA context is live. Wired at `sol_attn`, `pisa_attn`, `sla`, `nvfp4::dequant_beforehand`. **`fv-gpucheck hunyuan {info,gen}`** reports load/text/denoise/step/decode/write + peak MiB. Fetch row `FastH3-4-step-Preview-v1-VSA-DataFree`.
- Reason: a GPU sweep cannot log “sol-attn kernel” while running scalar CPU math; Hunyuan cells were skipping
- Reversibility: cheap — host oracles still run without a device; WS-A replaces the host call sites
- Executed by: Executor
- ADR: none
- Verification: **host pass**. `cargo test -p fastvideo-models --lib hunyuan15`; `cargo test -p fastvideo-cudarc --lib` host_fallback / sol_attn / pisa_attn / nvfp4. GPU refuse path untested here (no nvcc).

### FVID · 2026-09-22 · FVID-2026-09-22-ltx25-diffvae
- Trigger: after distilled two-stage green — ship opt-in **DiffVAE** video decode on the validated 2.5 stack (conv VAE stays default; audio unchanged)
- Options: DiffVAE 1-step x0 untiled @ 768×512; defer tiling / NATTEN / multi-step stage-5 / two-stage+DiffVAE
- Decision: **`LTX2VideoDiffusionDecoderModel`** as opt-in path: denorm latents → NA stages 1–4 + PixelShuffle → stage-5 single x0 @ `t=1.0`; gather+SDPA NA (inward window); drop DiT before decode. `Ltx2Request::diff_vae` / `--diff-vae` / `FV_LTX2_DIFF_VAE=1`; fetch `diffusion_decoder/*` on 2.5 weight pull.
- Reason: peer quality decode path per Diffusers 2.5 pack; first green scoped to single-stage distilled without tiling
- Reversibility: cheap — flag off keeps conv VAE decode
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-22). `cargo test -p fastvideo-cudarc ltx2::diffusion_decoder`; `fv-gpucheck ltx2 gen --model-version 2.5 --diff-vae`. RTX PRO 6000 WS `51982652` (~$0.779 / 33 min): 8 ancestral ~18.1 s, **DiffVAE decode ~884.4 s** (host NA), audio decode ~0.4 s; **121** frames, wav 48 kHz / 240480 samples, peak **70469 MiB**, mp4 `artifacts/clips/20260922T001443Z-ltx2-gen/ltx25-diffvae.mp4`. Build `77cf645514609c38`.

### FVID · 2026-09-21 · FVID-2026-09-21-ltx25-two-stage
- Trigger: after stage-1 + BWE green — pick next 2.5 extension
- Options: distilled two-stage (half-res → spatial upsampler → 3-step stage-2); DiffVAE/diffusion decoder; duration head; prompt enhancer; 1536×1024 scale-up
- Decision: **distilled two-stage** on the validated stage-1 path. Same DiT; new `latent_upsampler/`; no stage-2 LoRA on distilled DiT. Validate at final **768×512** (stage-1 384×256) so stage-2 token count matches single-stage.
- Reason: official recipe; VRAM stays in existing `ltx2-gen` tier; other extras still out of scope
- Reversibility: cheap — `Ltx2Request::two_stage` / `--two-stage` / `FV_LTX2_TWO_STAGE=1`
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-21). `cargo test` latent_upsampler + request validation; `fv-gpucheck ltx2 gen --model-version 2.5 --two-stage`. RTX PRO 6000 WS `51970730` (~$0.560 / 25 min): stage-1 8 ancestral ~7.6 s, upsample 0.19 s → grid `[16,16,24]`, stage-2 3 ancestral ~11.6 s (~3.56 s/step), decode video ~3.9 s; **11** steps finite, **121** frames, wav 48 kHz / 240480 samples, peak **82117 MiB**, mp4 `artifacts/clips/20260921T220546Z-ltx2-gen/ltx25-two-stage.mp4`.

### FVID · 2026-09-21 · FVID-2026-09-21-ltx25-stage1
- Trigger: "let's implement LTX 2.5" after H3 matrix; user locked scope to distilled stage-1 + conv VAE, no extras unless they help perf
- Options: full 2.5 stack (DiffVAE/diffusion decoder, duration head, enhancer, two-stage); stage-1 only; Comfy-only vs Diffusers pack
- Decision: **distilled stage-1 T2AV** on the existing `ltx2` modules with `Ltx2ModelVersion::V25`. Weights: `Lightricks/LTX-2.5-Diffusers` (oracle) / Comfy split pack. Gemma4 unified TE, 22B distilled DiT (`ff_bias=false`, gated attn, cross_attn_mod, `use_prompt_embeddings=false`), conv video VAE, audio VAE + **BWE vocoder** (required by checkpoint). Ancestral Euler for `model_version≥2.5`. Keep 2.0 path. Spec: `docs/ports/ltx25.md`.
- Reason: matches the 2.0 milestone shape; official distilled stage-1 still needs gated/cross-attn-mod DiT + Gemma4 + BWE — those are checkpoint contracts, not optional product extras. Diffusion decoder / duration / stage-2 deferred.
- Reversibility: cheap — version enum + opt-in weight roots
- Executed by: Executor
- ADR: none
- Verification: **host pass + remote GPU gen pass** (2026-09-21). Spec `docs/ports/ltx25.md`. `cargo test -p fastvideo-cudarc ltx2 --lib` / models ltx2 ok; `fv-gpucheck ltx2 gen --model-version 2.5`. Shipped: Gemma4 TE (`DecoderConfig::gemma4_12b_text`), gated DiT + `ff_bias=false` + cross_attn_mod AdaLN (9-row), per-modality connectors, ancestral Euler, conv VAE typed upsamplers (spatial d2s), vocoder **SnakeBeta AMP** + full **BWE** (Hann-sinc ×3 skip + MelSTFT + `bwe_generator` residual → 48 kHz; MelSTFT mel proj on host — CUDA `matmul` refuses `[M,F]@[N,F,T]`). **Remote (SnakeBeta+nearest):** Texas `51956651` (`20260921T193442Z-ltx2-gen`). **Remote (full BWE):** Switzerland `51967618` (`20260921T213121Z-ltx2-gen`), gen ~50 s after weights: BWE path logged, 8 ancestral ~2 s, **121** frames, wav 48 kHz / 240480 samples, mp4 `artifacts/clips/20260921T213121Z-ltx2-gen/ltx25-bwe.mp4`.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-matrix-followup
- Trigger: after encoder matrix pass — run remaining weight/SKU gaps (Synthetic Step1900, Preview on 80 GB, longer clip)
- Options: full matrix re-rent; serialize; skip Synthetic; raise res instead of duration
- Decision: **followup wave** (`FV_MATRIX_WAVE=followup`): skip V2 gens; H200 = DataFree baseline + Synthetic-Step1900 + DataFree **15s**; A100/H100 = Preview VSA stock + recovered-8b (`FV_MATRIX_PREVIEW_80` on). Same Vast parallel / reuse-box / TEST_ID / GPU-empty strategy. No LoRA-on-base, no v0.2, no dense+8b.
- Reason: quality A/B for Synthetic, 80 GB fit check, and a blog-scale duration sample without re-paying V2
- Reversibility: cheap — wave is env-gated; cells skip on clip reuse
- Executed by: Executor
- ADR: none
- Verification: **pass** (build `22bfa760236939c7`, relaunch after credit top-up). Drivers: `artifacts/gpucheck/matrix-logs/{a100,h100,h200}.followup.driver.log`. Shared run dir `…/20260921T013704Z-h3-matrix`. Boxes destroyed: A100 `51837932`, H100 `51837933`, H200 `51837935`. First attempt aborted (credit→$0).
  - **Preview-80 A100**: stock `021329Z` ~14.9 s/step; recovered-8b `021709Z`; PSNR stock-vs-8b **16.86 dB**. Run ~$0.75 / 44 min.
  - **Preview-80 H100**: stock `020132Z` ~10.6 s/step; recovered-8b `020416Z`; PSNR stock-vs-8b **16.37 dB**. Run ~$1.47 / 31 min.
  - **H200 DataFree** `021152Z`: warm ~7.7–8.3 s/step, peak **59780 MiB**.
  - **H200 Synthetic-Step1900** `022439Z`: warm ~7.7–10.0 s/step, peak **61028 MiB**. PSNR datafree-vs-synth1900 **15.38 dB**.
  - **H200 DataFree 15s** `022639Z`: warm **~33.2 s/step** ×4, denoise **133.0 s**, peak **91008 MiB**, 362 frames. Run ~$3.70 / 55 min.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-encoder-matrix
- Trigger: checkpoint × encoder × residency matrix (V2 / Preview VSA / Preview Dense × stock streamed / SearchingMan 8B / cache-hit / resident-fp8|bf16 on H200)
- Options: RunPod network volume; serialize SKUs; keep Python hub fetch; custom `fv-gpucheck fetch`
- Decision: **Vast, three parallel SKUs** (A100-80, H100-80 not NVL, H200). Slim runtime: **hf-fm**, no Python. Contracts `fasth3_4step_vsa` / `fasth3_4step_dense`; `--text-encoder recovered-8b`. Tier `h3-matrix`: reuse box, GPU-empty/restart between cells, nohup, `TEST_ID` on every log line, `FV_IMAGE_PULL_TIMEOUT=120`. Preview Hub ids: `…-4-step-Preview-v1-VSA-DataFree` / `…-Dense-DataFree`. Flag defaults unchanged.
- Reason: parallel CDN pulls beat serial volume copy; TAEH3 keeps decode off the critical path; PSNR (same box) gates 8B drift, not a host oracle.
- Reversibility: cheap — tier is opt-in; encoder choice is a CLI flag
- Executed by: Executor
- ADR: none
- Verification: **pass** (build `22bfa760236939c7`). Driver logs: `artifacts/gpucheck/matrix-logs/{a100,h100,h200}.driver.log`. Runs: `…/20260920T192357Z-h3-matrix` (A100), `…/192358Z` (H100+H200 V2), `…/194151Z` (H200 Preview). All three Vast boxes destroyed.
  - **V2 stock streamed** `--warm`: H200 ~9.0 s/step peak ~60 GiB; A100 ~16.2 s/step; H100 ~12.0 s/step.
  - **V2 recovered-8b** resident-bf16 PASS (same step times): H200 `20260920T192610Z-…` 198 s peak **68768 MiB**; A100 `20260920T192617Z-…` 393 s peak **68414 MiB**; H100 `20260920T192706Z-…` 277 s peak **69978 MiB**.
  - **V2 cache-hit** PASS on all three. H200 also **resident-fp8** `20260920T193237Z-…` peak **82816 MiB** + **resident-bf16** `20260920T193552Z-…`.
  - PSNR stock-vs-8b: A100 **15.23 dB**; H200 **15.03 dB**; H100 **14.95 dB**.
  - **Preview 4step-vsa** (H200): stock `20260920T195013Z-…` warm denoise **30.9 s** (~7.7 s×4) peak 60 GiB; 8b `20260920T195159Z-…` same ~7.7 s/step peak **69408 MiB**; cache-hit `20260920T195355Z-…`; PSNR stock-vs-8b **15.60 dB**.
  - **Preview 4step-dense** stock `20260920T195557Z-…` ~**42 s/step** ×4, warm denoise **168.8 s**, peak **57092 MiB** (~5.5× VSA).
  - Preview first attempt skipped: async `fetch` checked `.complete` before `wait-weights` — fixed in `validate.sh`.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-mlx-affine
- Trigger: "implement our own kernel and also port the MLX work"
- Options: 4-step Preview / 90% VSA A/B; community GGUF/nunchaku; process-wide `FASTVIDEO_FP8`; first-party affine W8A16 + official MLX recipe
- Decision: **first-party CUDA kernel** for FastVideo's MLX affine group-64 INT8/INT6/INT4 (weight-only, activations BF16/F32) and load `mlx_h3_dit.safetensors` / quantize-on-load from the official FastH3 DiT. Fused dequant-in-tile GEMM — do not materialize a full bf16 weight. Not nunchaku. FFN E4M3 stays off (`FVID-2026-09-20-h3-ffn-fp8-ab`).
- Reason: there is no official CUDA FP8/NVFP4 H3 DiT; MLX INT8 is the published weight-only grid (`mode=affine`, group 64, `w = scale * q + bias`). A whole-weight dequant would spend the VRAM we are trying to save.
- Reversibility: cheap — unset `FASTVIDEO_H3_AFFINE` / `--h3-affine`
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 147132 `20260920T151313Z-h3-gen` (`51767696`): bf16 warm denoise **125.6 s** / 15.0–16.0 s/step, peak 82482 MiB. INT8 affine fused GEMM (`--h3-affine int8`, quantize-on-load): step 1 **589.3 s**, step 2 **590.3 s** (~38×). Sole GPU process. Killed mid-step-3 — no PSNR, no warm affine. **Flag stays off.** Kernel is correctness-first (16×16 tiles, no tensor cores); do not enable until a fast GEMM exists.

### FVID · 2026-09-20 · FVID-2026-09-20-h3-ffn-fp8-ab
- Trigger: "proceed" after ranking remaining H3 levers: FFN GEMMs 33 s, MMA 28 s, prefix 11 s
- Options: H3-only FP8 on `ff_in`/`ff_out`; fuse Q/K RMSNorm+RoPE (6.7 s); process-wide `FASTVIDEO_FP8`; prefix SDPA
- Decision: **H3-only E4M3 GEMM on the two FFN linears** (`FASTVIDEO_H3_FFN_FP8` / `--h3-ffn-fp8`). Same-box bf16 vs FP8 A/B, same prompt/seed. Do not set process-wide `FASTVIDEO_FP8`. Pad the token axis to a multiple of 16 (5s H3 is 37756).
- Reason: the cheap fuses are done; 33 s is the remaining FFN pile. Wan process-wide FP8 was −4.8% / 16.9 dB (`FVID-2026-09-18-fp8-linears-measured`) — this clip decides whether H3's larger GEMMs beat activation-quant overhead, and whether the quality is usable.
- Reversibility: cheap — unset the flag
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 132178 `20260920T114626Z-h3-gen`: warm denoise **122.5 → 116.5 s (−4.9%)**. FFN 34.5 → 28.1 s — `h3_ffn_in` 21.6 → 16.7, `h3_ffn_out` 11.1 → 9.6, act 1.8 unchanged. Load 82 → 114 s. Peak 82.5 → 71.0 GiB. Clip PSNR vs bf16 **20.3 dB** (Y 18.6). Same shape as Wan: a few percent, unusable frames. **Flag stays off.**

### FVID · 2026-09-20 · FVID-2026-09-20-fused-swiglu-next
- Trigger: "commit, push, then proceed to next steps" after unchunked FFN measured as a wash
- Options: fused value-first silu×mul kernel; process-wide `FASTVIDEO_FP8` on FFN; prefix SDPA; more MMA load work
- Decision: **fuse `h3_ffn_act`**. Last-dim `narrow` copies both `[S, ffn]` halves before a separate silu and mul. One `swiglu_value_first` pass. Then a profiled H3-gen+TAEH3.
- Reason: FFN is still 38.1 s; act is 7.0 s / 18% of it and the only new slice. `FASTVIDEO_FP8` stays off (`FVID-2026-09-18-fp8-linears-measured`).
- Reversibility: cheap — FeedForward can go back to `narrow` + `silu` + `mul`
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Max-Q machine 147132 `20260920T104313Z-h3-gen`: warm denoise **122.5 s** vs 125.9. FFN 34.6 s — `h3_ffn_in` 21.7 s, `h3_ffn_out` 11.1 s, **`h3_ffn_act` 1.83 s** (was 7.0 s). Peak 82.6 GiB. Act is the cut; GEMMs unchanged. Wall ~3 s on a different Max-Q, so treat E2E as noise-adjacent.

### FVID · 2026-09-20 · FVID-2026-09-20-ffn-unchunk-next
- Trigger: "commit, push and continue" after fused QKVG measured as a wash
- Options: process-wide `FASTVIDEO_FP8` on H3 FFN; keep FFN in bf16 through SwiGLU; unchunk the 8192-row FFN GEMM on 5s clips; fused silu×mul kernel
- Decision: **do not turn on `FASTVIDEO_FP8`.** Unchunk FFN when `[S, 2*ffn]` f32 fits in 6 GiB (5s = 4.0 GiB; 109k rows still chunks). Split `h3_ffn_in` / `h3_ffn_act` / `h3_ffn_out` so the next profile names the GEMMs. Then a profiled H3-gen+TAEH3.
- Reason: FFN is ~38 s / 31% and matches BF16 FLOPs; Wan FP8 linears were −4.8% with 16.9 dB frames (`FVID-2026-09-18-fp8-linears-measured`). Five M=8192 GEMMs are the cheap thing to stop doing
- Reversibility: cheap — `FFN_ROW_CHUNK` remains the fallback
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Same Max-Q machine 103044 `20260920T102145Z-h3-gen`: warm denoise **125.9 s** vs 123.5–125.5. FFN still 38.1 s — `h3_ffn_in` 20.6 s, `h3_ffn_out` 10.5 s, **`h3_ffn_act` 7.0 s**. Peak 82 GiB unchanged. **Wash** on wall time; act is the only new slice worth a kernel.

### FVID · 2026-09-20 · FVID-2026-09-20-fused-qkvg-next
- Trigger: "commit and push then continue with next strategy" after TMA measured as noise on 5s H3
- Options: fused QKVG GEMM; FFN / SM12x FP8; prefix SDPA; more MMA load work
- Decision: **fused QKVG first** (one `Linear::load_fused` of `to_q`/`to_k`/`to_v`/`to_gate_compress`, same per-head RMSNorm+RoPE). Then a profiled H3-gen+TAEH3.
- Reason: QKVG+out is 36 s / 29% of denoise and currently rereads the 520 MB activation four times; FFN is the same size pile but a separate slice
- Reversibility: cheap — separate linears are a load-path revert; `FASTVIDEO_VSA_KERNEL` is untouched
- Executed by: Executor
- ADR: none
- Verification: host tests pass. Max-Q 6000 `20260920T100022Z-h3-gen`: warm denoise **125.5 s** vs TMA-only 123.5 s. Attn still 36.5 s (fused GEMM 21.3 s, Q/K prep 6.7 s, out 6.6 s). Peak 82 GiB vs 78. **Wash.** FFN 38.6 s is the remaining pile.

### FVID · 2026-09-20 · FVID-2026-09-20-tma-kernels-first
- Trigger: "proceed" after ranking remaining H3 levers (TMA measure, fused QKVG, FP8 FFN, tiled QKV, prefix SDPA)
- Options: kernels A/B of TMA vs `mma` on sm90+; skip to fused QKVG/FFN; profiled H3-gen first
- Decision: **kernels A/B first** (`compute_cap>=900`, T1). Then a profiled H3-gen + TAEH3 only if TMA matches the host reference.
- Reason: TMA is coded and unmeasured; FFN/QKVG work is wasted if `vsa_h3_3_mma` is still load-bound
- Reversibility: cheap — `FASTVIDEO_VSA_KERNEL=mma` is the Ampere path
- Executed by: Executor
- ADR: none
- Verification: **pass** on 5060 Ti sm_120 after `shared::cta` + `__grid_constant__`. `vsa_tma_*` matches `vsa_mma_*` (rel_l2 0.00282). Profiled H3-gen+TAEH3 on Max-Q 6000: warm denoise **123.5 s** (Ampere was 124.7 s — TMA is noise). `vsa_mma_tile` 2.5 s / 6% of VSA; fine MMA 26.9 s; QKVG+out 36.0 s; FFN 37.3 s. Next lever is fused QKVG / FFN, not more MMA loads.

### FVID · 2026-09-19 · FVID-2026-09-19-sm120-is-not-umma
- Trigger: "continue" after the TAEH3 A/B, queued as Blackwell UMMA on `vsa_h3_3_mma`
- Options: port FastVideo's sm100a `tcgen05` kernel; write UMMA from scratch; TMA + existing `mma.sync` on SM12x; FP8 fine stage
- Decision: **do not write UMMA.** RTX PRO 6000 is SM12x: no TMEM, no `tcgen05`. FastVideo's sm100a body is `#if __CUDA_ARCH__ == 1000 && SM100_ALL` and ptxas rejects it for `sm_120`. Peak BF16 here is still `mma.sync`. Skip H3 prefix query tiles in the MMA grid (SDPA overwrites them). Next kernel work is TMA 128B-swizzled K/V loads around the current MMA, not a new ISA.
- Reason: Colfax / CUTLASS / Triton all document consumer Blackwell as MMAv2 + TMA; the 5s clip is already ~130 TFLOPS of BF16 MMA, so a wrong-ISA rewrite cannot land
- Reversibility: cheap — prefix skip is a grid offset; TMA stays unstarted
- Executed by: Executor
- ADR: none
- Verification: host VSA tests; prefix skip is 11/671 query tiles on the 5s layout (~1.6% of the MMA grid). TMA 128B-swizzle kernel (`vsa_mma_attn_tma`) added for sm90+; Ampere `cp.async` remains `FASTVIDEO_VSA_KERNEL=mma`. GPU parity pending (`vsa_tma_*` in the kernels tier).

### FVID · 2026-09-19 · FVID-2026-09-19-h3-profile-next
- Trigger: "proceed" after the H3 `--profile` gen on an RTX PRO 6000 WS (`20260919T223558Z-h3-gen`). Block: attn 62%, FFN 31%. VSA: MMA 78%, prefix_dense 17%. QKVG four-way even.
- Options: split-softmax / persistent prefix; fused QKVG; FFN fusion; Blackwell UMMA on `vsa_h3_3_mma`; TAEH3 A/B (already coded)
- Decision: **TAEH3 warm A/B first** (no `--profile`, same prompt/seed as the official 22.9 s decode). Then MMA for sm_120. Park split-softmax. Bootstrap fails if cuBLAS lacks `cublasGetEmulationSpecialValuesSupport` instead of aborting in `dlsym`.
- Reason: decode is 19% of E2E and the flag already exists; prefix is 5% of denoise; MMA is the VSA term we own
- Reversibility: cheap — TAEH3 is opt-in; the symbol check only refuses a known-bad `:latest`
- Executed by: Executor
- ADR: none
- Verification: same Max-Q, no `--profile`. Official (`20260919T225958Z-h3-gen`): warm decode **28.97 s**, denoise 124.7 s, E2E 154.2 s. TAEH3 (`20260919T231221Z-h3-gen`): warm decode **0.98 s** (~30×), denoise 125.0 s, E2E 126.6 s. Clip `artifacts/clips/20260919T231221Z-h3-gen/h3-taeh3.mp4`.

### FVID · 2026-09-19 · FVID-2026-09-19-remote-job-logs
- Trigger: "make a note to always show logs for remote processes including Vast and Runpod jobs" after an H3 profile rental where logs only appeared when asked
- Options: wait until asked; stream driver logs into chat for the life of the job
- Decision: **always paste remote-job logs in chat** (Vast, Runpod, rented build boxes, `validate.sh` / `build-remote`). Stage lines and step timings; skip per-block spam
- Reason: billed GPU time is invisible if the only output is a terminal file
- Reversibility: cheap — drop `.cursor/rules/remote-job-logs.mdc`
- Executed by: Executor
- ADR: none
- Verification: rule file present; H3 profile watcher still posting step lines

### FVID · 2026-09-19 · FVID-2026-09-19-taeh3-opt-in
- Trigger: "yes, let's implement taeh3" after the TensorRT-vs-TAEH3 recommendation for H3 decode (22.9 s / 19% of a warm clip)
- Options: TAEH3 opt-in (quality trade, same family as Wan TAEHV); TensorRT on the official ViT (~1.5× decode, new runtime); leave the official decoder
- Decision: **port TAEH3** (`taeh3.safetensors`) behind `--taeh3-weights` / `FASTVIDEO_TAEH3_WEIGHTS` / `FV_TAEH3=1`. Official ViT stays the default. TensorRT not started.
- Reason: decode is the only remaining non-DiT term worth cutting, and TAEH3 reuses the TAEHV port without adding TensorRT
- Reversibility: cheap — unset the flag and the official decoder loads
- Executed by: Executor
- ADR: none
- Verification: host tests green (`cargo test -p fastvideo-cudarc --lib taehv`: 13 passed). GPU A/B on Max-Q: TAEH3 warm decode **0.98 s** vs official **28.97 s**; denoise unchanged (~125 s); clip not flat (std 0.28).

### FVID · 2026-09-19 · FVID-2026-09-19-text-plan-measured
- Trigger: "lets run" — the text-encoding plan (oracle cache, conditioning cache + slim checkpoints, resident encoders, FP8 rows, prefetch) had only been checked on the host. All on RTX PRO 6000 96GB (user's choice; the one A100 80GB offer never booted, and 80 GB cannot hold H3 + a resident encoder or a float32 LTX-2 reference).
- Kernels tier (RTX 3060, <1 cent): FP8 row codes and dequantized weights equal the host exactly; the scales were one ulp off because the device compiler turns `a / 448` into a reciprocal multiply — both sides now spell `a * (1/448)` and are exact. Prefetched encode == plain streamed encode, bit for bit, for stored-bf16 and stored-f32 checkpoints.
- Text conditioning cost, same prompt and seed per model:

  | route | H3 (Qwen3-VL-32B tap 50) | LTX-2 (Gemma-3-12B + connectors) |
  | --- | ---: | ---: |
  | streamed + prefetch, cold disk | 13.3 s (compute starved 11.6 s) | 15.2 s from the f32 checkpoint (starved 7.9 s) |
  | streamed + prefetch, slim bf16 checkpoint | n/a (already bf16, unread shards never fetched) | 10.7 s (starved 4.7 s) |
  | streamed + prefetch, warm page cache | 3.9 s (starved 3.0 s) | — |
  | resident, new prompt | 0.33-0.41 s (FP8 rows, 22.7 GiB, 21-40 s one-time load, peak 83.4 GB) | 2.7 s (bf16, 20.0 GiB, 11.2 s load, peak 70.3 GB) |
  | conditioning cache hit | 0.19-0.22 s | 2.4 s |

  Reading: `compute starved` ~= `host read+convert` everywhere, so the streamed path is disk-bound and prefetch has nothing left to overlap on a cold cache; its win is the warm-cache case. The real levers are the resident encoder and the cache. LTX-2 has a ~2.4 s per-prompt floor that is neither Gemma nor the connectors (a cache hit pays it) — unexplained, tokenizer parse suspected. H3 loads the resident encoder (27 s) even when the prompt is a cache hit — wasted in a one-shot run.
- Parity, first hardware runs: H3 text vs transformers — ids exact, tap 50 rel 1.37e-2 streamed and 1.36e-2 resident FP8 (FP8 vs our native 4.2e-3; generation-time drift check 4.15e-3, cosine 0.99999). Gemma-3 vs transformers — all 49 hidden states pass, worst 1.65e-2 at the normed last state, ids exact; connectors vs float32 3.2e-7; end to end vs the bf16 reference 3.4e-3 / 3.7e-3. LTX-2 DiT judged against a float32 reference: ours is CLOSER to float32 than the reference's own bf16 pass at every loop step (step 7 video 0.212 vs 0.371); rope tables 10/10, forward 63/66, loop 17/22 — the misses are early audio-video cross-attention taps at 1.3-1.6x the bf16 floor (gate 1.25x), audio steps 6-7 at ~1.1x the gate, a 35 dB end-to-end frame PSNR gate that assumes non-diverging trajectories, and the vocoder at 40.9 dB SNR vs a 50 dB gate. Gates not yet re-set.
- All three reference dumps are now in the oracle cache (h3 text 19 MB, ltx2 text 849 MB, ltx2 dit 1.2 GB): these tiers no longer build a Python environment or load a reference model.
- Hosts: machine 111175 rented GPUs with ~60 GB held by another process (two LTX-2 OOMs and one reference OOM before free-memory logging showed it); `remote.sh env` now fails on a non-empty GPU, which marks the machine bad. Machine 150596 (38.146.30.19) downloads at 0-62 MiB/s; blacklisted. A chained tier idles ~5 min at hand-off because a leftover child keeps the log pipe open — not fixed yet.
- Cost: about $2.9 for everything above, of which ~$1.1 was lost to the two bad hosts.
- Reversibility: n/a (measurements); the harness changes are env-gated (`FV_TEXT_PLAN`, `FV_GPU_RAM_MIN`, `FV_DISK_GB`, `FV_MAX_GPU_USED_MIB`).

### FVID · 2026-09-19 · FVID-2026-09-19-text-encoder-prefetch
- Trigger: last step of the text-encoding plan the user approved ("let's follow your recommended order", then "proceed with step 5"). After the conditioning cache and the resident encoders, the streamed forward is still what runs on a card without room for a resident encoder and on every first prompt, and it was a strict sequence per layer: page weights in, convert, copy from pageable memory, then launch — the device idle for the first three.
- Decision: **the streamed encoder stages layer i+1 while layer i computes** (`llm/prefetch.rs`). A worker thread fills one layer-sized page-locked buffer straight from the mapped shards and uploads each tensor on a second, non-blocking stream; one staged layer waits in a bounded channel. Both ports get it without a change, since both stream through `llm::hidden_states`. `FASTVIDEO_LLM_PREFETCH=0` turns it off; it turns itself off (with a log line) when bf16 linears are off or the pinned allocation is refused.
- Synchronization is by hand, because the crate runs cudarc with event tracking off: the worker synchronizes the copy stream before handing a layer over, so compute never sees a partial weight; the weights live on the copy stream, and before a used layer is dropped the copy stream is made to wait on an event recorded on the compute stream, so the stream-ordered free cannot recycle memory that queued kernels still read. Cost: up to three layers on the device instead of one (~3 GB for Qwen3-VL-32B), one layer of pinned host memory.
- Same numbers by construction, and checked: one bf16 conversion (`fill_bf16`) now feeds both the plain load and the pinned buffer (host test: stored bf16 bit-exact incl. NaN payloads, f32/f16 rounded identically, across the chunk boundary); `Layer::assemble` builds a layer for both paths and a host test pins that the staged part list is exactly what it asks for, for both presets. The kernels tier gained `llm_prefetch_equals_streamed_{BF16,F32}`: a real safetensors file, encoded both ways on the device, every tap compared at tolerance zero.
- Not measured yet: nothing here has run on a GPU (rule: no GPU for what the host can check). The run log line `llm prefetch: N layers, host read+convert, upload wait, compute starved` says which side is the bottleneck; if `compute starved` is near the old text-encode time the disk is the limit and only the slim checkpoints / resident paths help further.
- Reversibility: trivial — one env flag, one module; the plain `Streamed` source is unchanged and is what the device check compares against.

### FVID · 2026-09-19 · FVID-2026-09-19-audio-video-ports-h3-ltx2
- Trigger: Wan / FastWan is silent ("I don't hear any audio"). The user asked for `FastVideo/FastVideo-FastH3-8-Step-V2` (DMD2-distilled MiniMax-H3) and LTX-2, both text-to-audio-video, ported in parallel onto the cudarc backend, dev card RTX PRO 6000 96 GB.
- Decision: **both ports exist and generate; specs in docs/ports/{h3,ltx2}.md.** Shared foundation first (lazy shard-aware loader with disk bf16 -> device bf16; `llm`, one streaming decoder-only encoder for Qwen3-VL-32B and Gemma-3-12B; rotate_half RoPE with explicit tables, GQA, erf-GELU, Snake, reflect/replicate pad, GroupNorm, cuDNN conv with dilation/groups/transposed, conv1d; WAV + AAC mux), then one agent per model writing only its own modules.
  - H3: Qwen3-VL tap 50 streamed (shards 12-14 never read); the ~26 GB of AdaLN weights never reside — each block's `adaln_proj` is streamed once and the `[8,50,3,6,5376]` table kept; VSA-H3 reuses the existing tensor-core fine kernel unchanged (forced prefix columns as a 1e30 bias before the ordinary top-k, prefix query rows recomputed dense).
  - LTX-2: distilled DiT + connectors from the official single file through a key-rename view (proved identical, to the last digit, to the community diffusers conversion); video VAE chunking that is exact (two-frame carry per 3x3x3 conv) where diffusers' tiling is a blend.
- Measured on one RTX PRO 6000, our binary alone (no Python on the box), first clip in the process:

  | | FastH3 8-Step V2 | LTX-2 19B distilled (stage 1) |
  | --- | ---: | ---: |
  | clip | 1344x768, 124 frames @ 24 fps, 32 kHz stereo | 768x512, 121 frames @ 24 fps, 24 kHz stereo |
  | text encode (streamed) | 10.6 s | 15.7 s |
  | denoise, 8 steps | 94.1 s (11.8 s/step, VSA-H3 80%) | 13.5 s (1.59 s/step) |
  | audio decode | 0.39 s | 0.43 s |
  | video decode | 22.9 s | 1.59 s |
  | write tail | 0.31 s | 0.24 s |
  | weight loads (once) | 15.0 s | 19.4 s |
  | wall | 145.5 s | 51.4 s |
  | warm, new prompt / cached embeddings | ~129 s / ~118 s | ~31 s / ~16 s |
  | peak VRAM | 53.6 GB | 39.9 GB |

  H3 dense attention is 87 s/step at 38k tokens, so VSA-H3 is 7.4x. The tiled NVRTC "flash" SDPA is ~20x SLOWER than chunked cuBLAS at that length (25 blocks in 15 minutes) — it is a memory tool, not a speed one.
- Parity against diffusers on hardware: H3 text tap 50 rel 1.37e-2 (bf16 both sides), token ids exact; H3 audio decoder 1.29e-5 and video decoder 1.47e-6 once the reference was pinned to real float32 (PyTorch's default TF32 convolutions had made a "float32" oracle 3e-3 wrong); H3 DiT forward 1.30e-2 video / 9.2e-3 audio with layout and both sigma ladders bitwise; VSA-H3 3e-3 vs f64 host loops; LTX-2 video VAE PSNR 91 dB, streamed == whole to 1.2e-6. Eight-step trajectories drift from a bf16 reference by ~2x per step in BOTH ports (H3 0.40, LTX-2 video 0.34 at step 7) while producing coherent, on-prompt clips: few-step sampling amplifies rounding; late-step trajectory parity against a bf16 reference is the wrong gate.
- Published comparison (FastH3 blog, warm E2E, 5 s at 1344x768): base H3 dense 132.5 s on 1x B200; FastH3 Preview v1 (4-step, 90% sparse) 16.2 s on 1x B200, 6.1 s on 4x. Ours is the 8-step / 80% checkpoint on a card ~2.5-3x slower than a B200, with f32 activations around bf16 GEMMs. Lightricks publishes no latency figure for LTX-2.
- Process rules the user set today, now in memory: never build/run/test on a GPU unless the step needs one (binary from CI via `task dist:ci`; key-manifest tests and CPU tiny-references against diffusers' own classes guard every loader and every structural property on a laptop); use published numbers rather than building oracle infrastructure. Costly lessons behind them: three oracle runs lost to venv environment bugs, an LTX-2 key mismatch found on a rented box, a reused box running a stale binary, a box idling through a git clone.
- Next levers, by measured share: H3 denoise is 80% of a warm clip (bf16 activations in the block, fused modulation + SwiGLU), H3 VAE decode 19% (larger tile batches); for LTX-2 a new prompt is half text encoding (Gemma ships as 47 GB of float32 — cache embeddings, or store a bf16 copy).
- Reversibility: cheap — new modules only; the Wan path is untouched and still passes its kernels and parity tiers.

### FVID · 2026-09-19 · FVID-2026-09-19-taehv-default-streaming-write
- Trigger: with exact-SM cubins and the tensor-core fine stage in, an 8s clip was 19.7s (H100 NVL) / 20.7s (RTX 5090) and only ~25% of it was the DiT. The Wan VAE decode and the serial PNG-then-ffmpeg write were the next two terms, and neither is a kernel problem.
- Decision: **the `gen` tier decodes with TAEHV by default (`FV_TAEHV=0` for the Wan VAE) and frames are written while the decoder runs.**
  - `remote.sh fetch-taehv` pulls and verifies `taew2_1.safetensors` behind the text encoding; the clip stage gets `--taehv-weights`. The UI passes `{"taehv": false}` through as `FV_TAEHV=0`.
  - `TaeHv::decode_streaming` / `AutoencoderKlWan::decode_streaming` hand each chunk's finished frames to a sink; `Pipeline::decode_latents_streaming` wraps both and `decode_latents` is the no-op-sink case, so nothing else changes. `pack_rgb_u8` turns a chunk into interleaved 8-bit RGB on the device (3 bytes/pixel down instead of 12, no host clamp loop); a `VideoWriter` thread encodes PNGs in parallel with rayon and streams rgb24 into ffmpeg's stdin, so the mux needs no PNG round trip. Byte-exactness against the old writer is a kernels-tier check (`pack_rgb_u8_matches_host`).
  - `--warm` (`FV_WARM=1`, `task gen WARM=1`) runs one untimed generation first; the timings block now separates `load_s` from `generate_s` (denoise + decode + write), which is what a served request costs.
- Measured, 129 frames 832x448, 3 DMD steps, VSA, same prompt and seed, **first clip in the process** (cold):

  | | RTX 5090 (`20260919T130240Z-gen`) | H100 NVL (`20260919T130210Z-gen`) |
  | --- | ---: | ---: |
  | weight load | 4.0s | 11.2s |
  | denoise (3 steps) | 5.50s (1.93 / 1.78 / 1.78) | 4.61s (1.73 / 1.42 / 1.42) |
  | decode (TAEHV, chunked, frames streamed) | 0.64s (`taehv.decode` 505ms) | 1.16s (`taehv.decode` 693ms) |
  | write tail after decode | 0.10s | 0.24s |
  | **generate (denoise + decode + write)** | **6.24s** | **6.01s** |
  | peak VRAM | 11.6 GB | 11.3 GB |

  Both gates pass (129 frames, video_quality), 1.06 MB mp4 each, contact sheets identical across cards, and the prompt is what is on screen. Both boxes booted from the CI image by build id (no upload).
- **Warm** (`--warm`: one untimed generation first, so this is a resident pipeline answering its next request), same clip:

  | | RTX 5090 (`20260919T131942Z-gen`, machine 149252) | H100 NVL (`20260919T131944Z-gen`) |
  | --- | ---: | ---: |
  | denoise (3 steps) | 5.71s (1.87 / 1.89 / 1.89) | 4.31s (1.42 / 1.42 / 1.42) |
  | decode (TAEHV; `taehv.decode` alone) | 0.59s (458ms) | 0.97s (606ms) |
  | write tail after decode | 0.09s | 0.23s |
  | **generate** | **6.43s** (0.80 s per video-second) | **5.60s** (0.69 s per video-second) |

  Warm vs cold: the H100 sheds its 0.3s first-step penalty and 0.2s of decode; the 5090 sheds nothing beyond the load (its first step was already within noise of the rest — this host's 5090 ran 1.88s/step against 1.78s on the cold run's host). So on either card the cost of a clip after the first is denoise + ~1s, and the first clip adds only the weight load (2.9–4.0s on the 5090 hosts, 11.3s on the H100 host, which is disk, not GPU).
- Reversibility: cheap — `FV_TAEHV=0` restores the Wan VAE; the streaming sink defaults to a no-op.

### FVID · 2026-09-19 · FVID-2026-09-19-cuda13-aot-cubins-ci
- Trigger: the Blackwell arch-mapping bug (`nvrtc_arch` → None → compute_52) showed that choosing the kernel target at run time is a class of failure, not an instance. And the user's point stood: NVRTC 12.4 with no nvcc, and Docker builds under x86 emulation, were facts about the laptop, not constraints on the project.
- Decision: **CUDA 13.0 everywhere, kernels compiled ahead of time by nvcc, builds off the Mac.**
  - `kernels.cu` is a file; `include_str!` feeds the NVRTC fallback from the same text nvcc compiles.
  - `build.rs` embeds a cubin + PTX per SM in `FV_CUBIN_SMS` (75, 80, 86, 89, 90, 100, 120). No nvcc → empty table and a warning; nvcc that fails → hard error.
  - `KernelFns::load_for`: exact-SM cubin → highest same-major embedded PTX → NVRTC. The device banner names which (`kernels=Cubin(86)`). `FASTVIDEO_KERNELS=nvrtc` keeps the source path exercised.
  - `nvrtc_arches` tries compute_100/120 natively (NVRTC 13 knows them) with compute_90 as the fallback for an older libnvrtc; an unknown future SM gets the newest PTX, never None. Pinned by a test.
  - Builder and runtime images from NVIDIA's apt packages (`cuda-nvcc-13-0`, `cuda-nvrtc-13-0`, `libcublas-13-0`, `libcudnn9-cuda-13`); the PyPI `-cu13` wheels are placeholders. `CUDARC_CUDA_VERSION=13000`.
  - Offer filter `cuda_vers>=13.0`: the 13.0 runtime libraries need a >= 580 driver and there is no minor-version compatibility across a major. Measured cost: **314 of 353 sm80+ offers (89%) qualify**, including 38 of 40 RTX 5090s.
  - CI builds the runtime image on **every branch**, tagged by build id — the tag `validate.sh` looks up — so a feature branch's boxes boot with the binary baked in. `:latest` only from main. The binary is also uploaded as a plain artifact. The T0 workflow fails if `build.rs` did not embed cubins.
  - `build-remote.sh` / `task build:remote`: rent the cheapest box with >= 8 cores, install the `cuda-builder.Dockerfile` recipe, build, pull the dist back. First real run: **GTX 1660 S at $0.048/hr, toolkit 100s, `cargo build --release` 60s, all seven cubins, 236s, ~$0.003.**
- **Verified end to end** (`20260919T120223Z-kernels`, RTX 3060, driver 580.126 / CUDA 13.0): image `build-d5f94b79e476c1e2` found by build id, **no upload — 6s from ssh to stages** against 8.5 minutes on one host the day before; nvrtc 13.0.88 / cuBLAS 13.1.1 / cuDNN 9.26 loaded; banner `kernels=Cubin(86)`; **T1 PASS, 0 failures**. The nvcc-compiled `mma.sync` kernel matches the host reference at rel_l2 0.0028 / 0.0031 — identical to NVRTC on the 5090 — at 3.0x / 2.7x / 4.3x over the gather path.
- **End-to-end effect of the tensor-core fine stage**, profiled on a 5090 (`20260919T114126Z-gen`, dispatched by `auto`): self-attention **74.7% → 33.0%** of a DiT block; the fine stage that was gather + QK/softmax + PV (~63% of the block) is now one phase at **13.0%**. The block is FFN 21.7%, self-attn 33.0%, cross-attn 13.8%; attention is no longer the dominant term. Remaining attention cost is mostly `tile_mean` (8.2%), the cheap fusion noted earlier.
- Two build-box lessons, both fixed: rsync creates only the last path component (a bare image has no `/workspace`, reported as receiver IO error 11), and validate.sh's inherited cleanup tried to pull a validation dir a build box does not have.
- Reversibility: medium — the driver floor and the 13.x soname are the coupling; dropping back to 12.x is a Dockerfile/env edit but re-admits the run-time target choice.
- Executed by: Executor
- Not done: a Hopper `wgmma`/TMA variant of the fine stage (`mma.sync` runs on sm90 but is not its peak path), and `tile_mean` fusion.

### FVID · 2026-09-19 · FVID-2026-09-19-vsa-tensor-core-fine-stage
- Trigger: the profile put the K/V gather at 48% of self-attention, ~21% of the DiT (FVID-2026-09-19-where-the-time-actually-goes), and a survey of every reference block-sparse kernel — FastVideo's Triton `_attn_fwd_sparse`, its ThunderKittens sm90 kernel, its sm100a kernel, FlashAttention-4's block sparsity — found that **none of them gather**. They permute Q/K/V into tile-contiguous order once, then one CUDA block per query tile re-points at each selected 64-row slab and folds it into an online softmax. The only FastVideo path that gathers is VMoBA, and only because it dispatches to `flash_attn_varlen`, which cannot take a block index. The gather is a consequence of calling cuBLAS per query tile, not of sparsity.
- Decision: **replace the fine stage with a tensor-core streaming kernel** (`vsa_mma_attn`): `mma.sync.m16n8k16` bf16, `ldmatrix` fragment loads over an XOR-swizzled tile, `cp.async` double buffering so tile i+1 loads while i computes, and P repacked from S's C-fragment layout straight into A-fragment layout so it never leaves registers. `vsa_tile_qkv` is the reference's `tile()`: O(padded × dim) once, against the gather's O(tiles × topk × 64 × dim) — 531 GB written per 8s clip.
- **Verified, RTX 5090** (`20260919T111737Z-kernels`, T1 PASS): against the host reference rel_l2 **0.0029 / 0.0031** — slightly better than the gather path's 0.0032 / 0.0033, since P stays in registers. Speed against the gather path, median of three:

  | tokens | topk | gather | mma | |
  | ---: | ---: | ---: | ---: | ---: |
  | 1,456 | 19 | 2.44 ms | 0.82 ms | 2.98x |
  | 4,368 | 19 | 2.50 ms | 0.88 ms | 2.85x |
  | 13,104 | 55 | 19.20 ms | 3.54 ms | **5.42x** |

  Reproduced in fast mode to within 1%. The speedup grows with token count; an 8s clip is 48,048 tokens.
- **This closes FVID-2026-09-18-fused-block-sparse-rejected the right way.** That kernel had the same structure and lost 8x on this same card class; this one wins 5.4x. The swing between the two is ~43x, and the only difference is scalar f32 math versus tensor cores. The idea was never wrong — the recorded rule ("do not hand-write attention math that cuBLAS can express") generalised one bad implementation into a closed door. The rule that survives is narrower: a hand-written kernel must use the tensor cores, or it will lose to a library that does.
- Hardware dispatch: `FASTVIDEO_VSA_KERNEL=auto|gather|fused|mma`. `auto` is now `mma` on sm80+ at dim 128, `gather` otherwise; the scalar kernel stays selectable as the negative result it is. The kernel body is guarded on `__CUDA_ARCH__ >= 800` and **traps** below it, so a wrong-arch build reports as a CUDA error rather than an uninitialised buffer.
- **Found on the way, and larger than this kernel**: `nvrtc_arch` returned `None` for sm_major 12, so on Blackwell NVRTC ran with no `-arch`, defaulted to compute_52, and compiled every `>= 800` guarded body out. The first run of this kernel was a no-op that reported garbage and a fake 3.04x — and *every kernel we had ever run on a 5090 or RTX PRO 4500* was JIT'd from compute_52 PTX. Hopper and later now target compute_90 (libnvrtc 12.4's ceiling; PTX is forward-compatible), pinned by a test. The per-arch compile gate never saw it because the gate names its targets; only the runtime path chose.
- Second lesson from that first run: `auto` had been pointed at the unverified kernel, so the "gather" baseline checks silently ran the stub too, and a "gather failure" was a second copy of the same failure. The baseline checks now pin the kernel by name, and a default may only point at a verified kernel — the flip is its own commit.
- Reversibility: cheap (`FASTVIDEO_VSA_KERNEL=gather`)
- Executed by: Executor
- Not yet measured: the end-to-end effect on a clip (`--profile` puts the whole fine stage under `vsa_4_mma`), and a Hopper `wgmma`/TMA variant — `mma.sync` runs on sm90 but is not its peak path.

### FVID · 2026-09-19 · FVID-2026-09-19-where-the-time-actually-goes
- Trigger: four hypotheses about the ~4x gap to FastWan-QAD were each built, measured and wrong — FP8 linears (4.8%), CUDA graphs (0.7%), a newer cuBLAS (0%), and fusing norms/residuals (predicted large). All four reasoned from upstream's description of their own work rather than from our stack. Profiling instead.
- **Measured, DiT block phases** (`--profile`, per-phase syncs, so proportions not absolutes): self-attention **74.7%**, FFN 14.2%, cross-attention 9.1%, and all five norms and gated residuals **2.0% combined**. The fusion targets FastWan-QAD lists first are worth 2% *in our implementation*.
- **Measured, VSA stages**: of self-attention, the **K/V gather is 48.2%** — **~21% of the whole DiT**, more than the FFN and cross-attention together. Fine QK+softmax 30.3%, tile means 8.7%, fine PV 6.9%, everything else 2.7%. The eight stages account for 50.04s of 51.67s, so nothing material is unattributed.
- Why FLOP arithmetic missed it: the gather is pure data movement. It materialises topk×tile_elems of K and V per query tile, 810 times per clip, and contributes no FLOPs at all — so an estimate built from GEMM throughput and attention FLOPs put attention at 8% of a step when it is 75%.
- **This reopens FVID-2026-09-18-fused-block-sparse-rejected.** That kernel existed to eliminate this exact gather, lost at 8x, and was recorded with a general rule: "do not hand-write attention math that cuBLAS can express." The measurement stands; the generalisation does not. One bad implementation of an idea was written up as if the idea were closed, and the idea's target turns out to be the largest single cost in the model. The rule should read as "a hand-written kernel must beat cuBLAS on the *math*" — it says nothing about a kernel whose purpose is to avoid materialising data.
- Second, smaller finding: `vsa_1_tile_mean` costs 8.7% of self-attention in **90 calls** — three separate mean-pools over full Q/K/V. Fusing them into one pass is contained and worth ~3.8% of the DiT.
- Reversibility: n/a (measurement only)
- Executed by: Executor
- Verification: `20260919T011154Z-gen` (block phases, RTX 3090) and `20260919T013314Z-gen` (VSA stages, RTX A5000). ~$0.20 for both. Absolute seconds are inflated by profiling syncs and the A5000 is slow; only the proportions are claimed.
- Process note: the profile cost ~$0.10 and was worth more than the four hypotheses that preceded it. Measure the stack before porting the other project's answers to it.

### FVID · 2026-09-18 · FVID-2026-09-18-cuda-graphs-not-worth-it
- Trigger: we sit ~4x behind FastWan-QAD's published 3.4s for a 5s 480p clip on a 4090, and the suspicion was launch overhead — our DiT launches a separate kernel per op with a fresh allocation, which is exactly what `torch.compile` and CUDA graphs remove. ADR-0003 had also deferred graph capture, so it looked like unclaimed ground.
- Two corrections on the way to the measurement. First, `FASTVIDEO_CUGRAPH` is **not implemented and not merely disabled**: commit `7566769` — the ADR-0003 pass itself — deleted `streams.rs` and rewrote `transformer.rs` from 994 lines to 237. The only surviving copy is an unmerged worktree 63 commits behind whose own commit is about schedulers. Second, the reason ADR-0003 gave for deferring capture ("capturing a host-bouncing block body isn't safe") was fixed by that same commit, which added the `layer_norm_last` / `modulate_scale_shift_last` kernels that removed the bounce. The conclusion outlived its premise.
- **Decision: do not build CUDA graph capture.** Measured on a 3090, 2s clip, VSA: **3,112 kernel launches per denoising step**, against a measured **2,237 ms** per step. At ~5 µs of launch overhead that is ~16 ms, or **0.7%**; at a pessimistic 10 µs, 1.4%. Graphs cannot be worth more than one or two percent here.
- The same run settles the other half: **zero host fallbacks**, and 6 device-to-host transfers for an entire clip (the frame readback). ADR-0003's blocker is gone — but it no longer matters, because the thing it was blocking is not worth doing.
- So the remaining gap to FastWan-QAD is **arithmetic, not overhead**. This also kills launch overhead as the explanation for the H100 dense result tying a 4090 (27.3s vs 27.5s), which needs a different cause — `mathprobe` on Hopper is the next measurement, since it reports which cuBLAS math mode the hardware actually honours.
- Of upstream's levers, FP8 linears are measured at 4.8% (FVID-2026-09-18-fp8-linears-measured) and TAEHV is adopted, leaving **SageAttention2++** as the only large untested one. Attention's share of the 2,237 ms per step should be estimated before building it.
- Incidental: 911 host-to-device transfers totalling 3.4 GiB during a 3-step clip. Not overhead-critical, but plausibly uploads repeated per step rather than once — worth a look.
- Reversibility: n/a (nothing built)
- Executed by: Executor
- Verification: `20260918T231508Z-gen`, ~$0.05. The counter lives in the one `launch!` macro every kernel goes through, so no call site can under-report, and clip reports now carry `device_stats` by default.

### FVID · 2026-09-18 · FVID-2026-09-18-taehv-adopted
- Trigger: the TAEHV port was verified against the reference (FVID-2026-09-18-taehv-port) but never run in the pipeline, so the quality and end-to-end time trade were both unmeasured.
- Decision: **TAEHV is worth using**, behind `FASTVIDEO_TAEHV_WEIGHTS=<dir>`. A/B on one RTX 3090, 8s clip (129 frames, 832x448), same prompt and seed, VSA on both sides, latents **bit-identical** (rel_l2 0.0) so the decoder is the only variable:

  | | Wan VAE | TAEHV |
  | --- | ---: | ---: |
  | VAE decode | 16.18s | **1.66s** (9.7x) |
  | total | 78.86s | **64.71s** (-18%) |
  | peak VRAM | 21.4 GB | **12.1 GB** (-44%) |

- Quality: **34.16 dB** PSNR between the two decodes. For scale, our own exact-vs-fast mode differs by 27.8 dB, so TAEHV sits closer to the Wan VAE than our fast path sits to our exact path. Visually the composition, motion and lighting are identical; TAEHV is slightly softer in fine detail (fur, foam).
- The memory drop matters as much as the time. 21.4 GB took an 8s clip to the edge of a 24GB card — the constraint behind this morning's OOM on a reused box and behind `--vae-chunk 2` existing at all. 12.1 GB is comfortable.
- **I had the structural trade backwards.** TAEHV being parallel over frames was described as an advantage over the Wan VAE's sequential feature cache; it is both. Every stage materialises every frame, and the deepest is ~12.6 GB for 129 frames, so the first A/B OOM'd. The fix is the reference's own sequential path: chunk the decode and carry each MemBlock's boundary frame across the seam. The oracle test ran 3 latent frames, where that liability cannot appear — a reminder that a correctness oracle at toy sizes says nothing about production shapes.
- Second failure worth recording: 33 latent frames at chunk 4 ends in a chunk of **one**, where there is no earlier frame to shift in. Only reachable at real clip lengths; the unit tests used frame counts that never produced a ragged tail. Both failures now have tests that fail without the fix.
- Reversibility: cheap — off unless the weights path is set, and a bad path errors rather than silently falling back to the Wan VAE.
- Executed by: Executor
- Verification: `20260918T195506Z-vaeab`. Three attempts, ~$0.16 total including the two failures.
- Not done: TAEHV is not on by default, and no clip tier enables it. The quality trade is real if small, so that should be a deliberate choice rather than a default.

### FVID · 2026-09-18 · FVID-2026-09-18-taehv-port
- Trigger: the Wan VAE is the largest component of a clip never attacked — 3.9s of a 23.7s 8-second clip on an H100 — and the only major block never checked against an external reference. FastWan-QAD's speedup leans on TAEHV for exactly this reason.
- Decision: **port the TAEHV decoder** (madebyollin/taehv, `taew2_1`) as an alternative to `AutoencoderKLWan`. Decoder only; text-to-video never encodes.
- Verified against the implementation it was read from, first hardware run: `video` rel_l2 **2.6e-4** (cosine 0.99999997) against a 2e-2 limit, output shape `[9, 3, 448, 832]` exact. `20260918T182450Z-taehv`, RTX 3060, **$0.008**.
- **Faster than the reference it copies**: decode 1.332s (reference PyTorch) → **0.443s** (ours), same card, same latent, same weights — 3.0x. Our port is parallel over frames because TAEHV's `past` is only a one-frame shift, so unlike the Wan VAE there is no sequential feature cache to serialise on.
- The measurement that could not be made by reading: TAEHV documents "~Gaussian" input, which had to mean **DiT-space** latents, *before* the per-channel `latents_mean`/`latents_std` un-normalisation the Wan VAE requires. Getting that backwards yields a plausible, wrongly-coloured video rather than an error — the same failure shape that hid the mirrored UMT5 bias. The oracle settles it.
- Every architectural constant inferred from the source was confirmed by the reference: `patch_size` 1, `latent_channels` 16, `t_upscale` 4, `frames_to_trim` 3, and T latent frames → 4T−3 output frames, which is Wan's 4n+1.
- Reversibility: cheap — additive, nothing existing changed.
- Executed by: Executor
- **Not yet answered**: what TAEHV costs in *quality* on a real clip, and what it saves end to end against the Wan VAE's 3.9s. Both need it wired into the pipeline and a clip run. Speed is established; the trade is not.

### FVID · 2026-09-18 · FVID-2026-09-18-fp8-linears-measured
- Trigger: FastWan-QAD (haoailab.com/blogs/fastwan-qad) claims 3.4s for a 5s 480p clip on a 4090 via FP8 linears + FP8 attention + TAEHV + full compile. The checkpoints turned out to be **unquantized** — F32 weights, a transformer config byte-identical to stock Wan2.1-1.3B, no quantization_config — so "FP8" names the precision the model was *trained to tolerate*, and every part of the speedup is ours to build.
- Built: `fastvideo-ops::fp8` (E4M3 conversion, exhaustively tested over all 256 codes), four CUDA kernels mirroring it, a cuBLASLt E4M3 GEMM, and FP8 linears behind `FASTVIDEO_FP8`.
- **The plumbing is correct.** On two sm89 cards (L40S and RTX 4090) the device quantizer matches the host reference on all 65,536 values with **zero** tolerance, and the GEMM lands at rel_l2 **1.39e-4** against a 2e-2 limit — bit-identical across both cards. Because both sides of that check see identical quantized operands against an f64 reference, 1.39e-4 is cuBLASLt's accumulation order alone; a wrong transpose would have been order-1 wrong.
- **Decision: FP8 linears stay off.** A/B on one 4090, FastWan-QAD-FP8-1.3B weights, same prompt and seed, 2s clip: denoise **4.207s → 4.007s (−4.8%)**, model load **6.49s → 12.76s**, total **14.36s → 20.19s (41% slower)**. Weight memory does fall 3607 → 2263 MiB (−37%).
- Accuracy cost: step-1 latents rel_l2 **0.108**, frames **16.9 dB** — against 4.4e-3 and 66 dB for our bf16 fast path. ~25x more error than bf16, on the checkpoint distilled specifically to tolerate per-tensor FP8.
- Why only 4.8%: the GEMMs really do run on FP8 tensor cores, but **activation quantization eats the win**. Every linear gains an amax reduction and a quantize pass over its input — two memory-bound passes over data the GEMM was about to read anyway. At Wan 1.3B's shapes, halving operand bandwidth does not outrun them. The load regression is separate and fixable (host-side element-by-element quantization of 1.3B parameters), but fixing it only removes a regression rather than creating a win.
- **Third instance of the same shape**, after FVID-2026-09-17-flash-sdpa-rejected and FVID-2026-09-18-fused-block-sparse-rejected: a correct implementation that loses to the simpler path. The cause differs — this one is not about tensor cores, it is about per-op overhead around them.
- What would have to change to revisit: fusing the quantization into the op that already writes the activation (the RMS norm, or the bias/activation epilogue), so it costs nothing extra. Even then the ceiling is the GEMM half of denoise, so expect 10-15%, against TAEHV which targets 3.9s of VAE out of a 23.7s H100 clip at far lower risk.
- Reversibility: cheap — the path is behind a flag that is off by default and never inferred from the device.
- Executed by: Executor
- Verification: `20260918T172402Z-kernels` (L40S, 405 checks, 0 fail) and `20260918T173147Z-fp8` (RTX 4090, A/B). $0.15 for both.

### FVID · 2026-09-18 · FVID-2026-09-18-umt5-bias-mirrored
- Trigger: user reported that generated clips do not follow the prompt. Confirmed visually: "a piper cub takes off" rendered a hand holding a green pepper; the benchmark prompt "A golden retriever sprints along the shoreline at sunset, waves breaking around its paws" rendered a static dog on grass. Upstream FastVideo, on the same GPU and the same weights, rendered the puppy on a beach with waves.
- Root cause: `relative_position_bucket` added the half-table offset when the key came **before** the query; HF adds it when `relative_position > 0`, i.e. after. The two halves of the learned 32-row relative attention bias were swapped. For a 12-token prompt **132 of 144 entries were wrong** — only the zero-distance diagonal survived.
- Why it presented as "keeps the subject, loses the scene": UMT5 carries no absolute or rotary positional encoding, so this bias, injected into all 24 encoder layers, is the model's *only* word-order signal. Mirroring it preserves token identity but binds modifiers, verbs and prepositional phrases to the wrong side.
- Decision: fix the predicate to `relative > 0` in all four backends (cudarc, models, burn, luminal — the function was copy-pasted, which is why every cross-backend comparison agreed with itself). Regression test takes its expectations from HF's formula rather than from our output, and fails against the old sign.
- Reversibility: cheap (one predicate)
- Executed by: Executor
- Verification: oracle tier `20260918T124220Z-oracle`, L40S 44GB, against transformers' UMT5 and diffusers' `WanTransformer3DModel` on identical inputs. Before the fix: text cosine **0.268** / rel_l2 **1.296** (FAIL), dit cosine **0.9999999** / rel_l2 **0.00045** (PASS), e2e cosine 0.823 / rel_l2 0.607 (FAIL).
- **The DiT port was exact the whole time.** Every attention, RoPE, patch-embed, adaLN, VSA and bf16 change was correct; the entire prompt-adherence failure was one inverted comparison in the text encoder.
- Process note: this survived 42 green validation runs because **every gate compared fastvideo-rs to fastvideo-rs** — `parity` is GPU vs our own CPU path (and feeds the DiT *random* embeddings, so it never touched the text encoder at all), and `compare` diffs two of our own clip dirs. The video-quality gates score luma, temporal MAD and clipping, so a coherent wrong video passes them. See FVID-2026-09-18-oracle-tier.

### FVID · 2026-09-18 · FVID-2026-09-18-oracle-tier
- Trigger: a prompt-adherence bug that no tier could see, because no tier compared us to anything but ourselves.
- Options: tighten the existing self-comparisons; add semantic scoring (CLIP similarity) to the clip gates; diff against the reference implementations on identical inputs
- Decision: **an `oracle` tier.** `scripts/gpu/upstream_oracle.py` runs transformers' UMT5 and diffusers' `WanTransformer3DModel` on the same weights and saves its inputs and outputs; `fv-gpucheck oracle` replays them through us. Three checks, on byte-identical tensors: `text` (our embedding vs the reference), `dit` (our DiT on the *reference* embedding), `e2e` (our DiT on *our* embedding).
- `dit` is the load-bearing design choice: feeding both sides the same conditioning removes the text encoder from the comparison, so text-vs-DiT is **decidable** rather than a single number that says only "something is wrong". On its first run it attributed the defect immediately — text failed at rel_l2 1.296 while dit passed at 0.00045.
- Float32 throughout: the oracle judges exact mode, and a looser judge cannot set a limit. Needs 40GB because UMT5-XXL in torch float32 is ~22GB before the DiT loads.
- Reversibility: cheap (additive; no existing tier changes)
- Executed by: Executor
- Verification: `20260918T124220Z-oracle` (baseline, pre-fix) — the tier failed exactly the two checks it should and passed the one that isolates the DiT. Cost $0.32, 36 min, most of it the upstream install.

### FVID · 2026-09-18 · FVID-2026-09-18-bf16-conv3d
- Trigger: after the SiLU fusion and chunked decode, VAE convolutions were the remaining VAE cost, and the dominant one ran 1.11 TFLOP in 27.7 ms ≈ 40 TFLOPS on a 3090 Ti — this card's TF32 peak, so compute-bound with no memory headroom to recover
- Options: leave it; bf16 storage through the whole VAE (large refactor); bf16 operands per convolution with casts around it
- Decision: **bf16 cuDNN conv3d as a third auto-selected backend** (`ded1f0c`). bf16 tensor ops run at roughly twice TF32 on Ampere, and the casts around one convolution cost ~3.4 ms against ~13.8 ms of compute saved, so it wins even without chaining bf16 between ops. `conv3d`'s auto mode times three candidates per shape and keeps the winner; bf16 only competes in fast mode, since exact mode must stay comparable to the CPU path.
- Measured 1.3-2.1x on large shapes across **three architectures**: Ada 4090 `[1,192,10,224,416]` 32.7 -> 15.4 ms, `[1,96,10,448,832]` 41.1 -> 28.1 ms; Blackwell 5090 `[1,192,6,224,416]` 9.9 -> 5.6 ms, `[1,96,6,448,832]` 10.4 -> 7.2 ms. 13 of 14 full-resolution shapes chose bf16.
- **Per-shape selection is load-bearing, not tidiness.** On the 4090, `[1,96,3,128,128]` is 0.3 ms in cuDNN f32 and **6.6 ms in bf16** — 22x slower, presumably a poor cuDNN algorithm for that configuration — and one other shape also lost. A blanket bf16 switch, the obvious implementation, would have absorbed both regressions silently.
- Accuracy: VAE decode rel_l2 0.00287 at 64.0 dB against the fast-mode limit of 0.02, versus 0.00092 at 74 dB for F32. A real trade, of the same kind already accepted for bf16 linears and bf16 attention probabilities — not free like the SiLU fusion.
- Reversibility: cheap (`FASTVIDEO_CONV3D=cudnn` or `unfold` forces the old behaviour; exact mode never selects it)
- Executed by: Executor
- Verification: runs `20260918T095625Z-clip` (Ada) and `20260918T101337Z-clip` (Blackwell, all stages pass, 15 min, $0.120). 8s clip on the 5090: denoise 18.4 s, VAE decode 5.02 s, both clips within 40 ms of each other.
- Process note: a `clip-2s-exact` OOM during this work was **accumulated GPU memory on a reused instance**, not the card and not these changes — the stage passes on a fresh box. Rent a new instance per run; never trust a timing from a machine that has already done substantial work.

### FVID · 2026-09-18 · FVID-2026-09-18-fused-block-sparse-rejected
- Trigger: VSA's fine stage gathers each query tile's selected K/V into a dense buffer (~106 GB/layer written and read, plus a 26 GB score buffer) so cuBLAS can run the GEMMs on tensor cores. A fused kernel streaming K/V straight from the tiled layout would move ~53 GB/layer and materialise no scores.
- Options: keep the gather; fuse with scalar f32 math; fuse with tensor cores via inline PTX `mma.sync`
- Decision: **the fused kernel stays off.** Measured on one RTX 5090 against the gather path on the same card class: **8.0x slower at 1,456 tokens, 5.5x at 4,368, 8.6x at 13,104**. Trading ~80 GB/layer of traffic for scalar f32 math does not come close to paying for the loss of cuBLAS bf16 tensor cores, which run at roughly twice the scalar f32 rate before any efficiency gap.
- The structure was right this time and it still lost. One CUDA block per QUERY TILE amortises each K/V tile load over 64 queries — exactly what `flash_attn_f32` got wrong with one query per block — and that moved the result from 12-36x slower (FVID-2026-09-17-flash-sdpa-rejected) to ~8x. Structure was never the binding constraint; tensor cores are.
- Reason: this is the second hand-written attention kernel to lose to cuBLAS by a wide margin. The rule that generalises: **on this hardware, do not hand-write attention math that cuBLAS can express.** **(Superseded 2026-09-19 — see FVID-2026-09-19-vsa-tensor-core-fine-stage. The 8x measurement stands; the same structure on tensor cores wins 5.4x on the same card class. Only this implementation was wrong.)** A fused kernel is only worth attempting with `mma.sync` tensor-core intrinsics, and even then it must beat a library that is already near peak.
- Reversibility: free (kept behind `FASTVIDEO_VSA_FUSED=1`, default off; correct, so it stays as a reference implementation and a place to add MMA later)
- Executed by: Executor
- Verification: correctness first — gpucheck holds it to the same host reference as the gather path across four grids in both precision modes, rel_l2 0.0026-0.0028, marginally *better* than the gather path's 0.0032 because the online softmax keeps probabilities in f32 registers instead of round-tripping bf16. Speed: run `20260918T104448Z-clip` (fused) vs `20260918T101337Z-clip` (gather), both RTX 5090 32607 MiB, fresh instances. Stopped after the probe: the clips would only have confirmed it more expensively.
- A dim=64 indexing bug (four output dims per lane hardcoded for dim=128) was found by the first hardware run and fixed in `6ea882c`; the d=128 grids the real model uses would never have caught it.

### FVID · 2026-09-18 · FVID-2026-09-18-vae-decode
- Trigger: after the VSA port, VAE decode (21.9 s) was the largest single cost in an 8s clip, bigger than any remaining attention win
- Options: bf16 storage through the VAE; faster convolutions; fuse elementwise ops; decode several latent frames per pass
- **Convolutions were ruled out by measurement, not skipped**: the dominant conv, `[1,96,6,448,832]` 3x3x3, runs 1.1 TFLOP in 27.7 ms ≈ 40 TFLOPS, which is this card's TF32 peak. Nothing to win there without bf16 storage — a much larger refactor.
- Decision: two changes, both verified equivalent rather than assumed.
  1. **SiLU folded into the channel RMS norm** (`0559ed1`). The decoder always pairs them and the norm kernel already reads x twice, so the activation rides along free and a whole read+write pass over a 859 MB tensor disappears. Three sites: both residual-block norms and each `norm_out`. **21.86 s -> 19.49 s (-10.8%)**, identical on both clips to 10 ms, gpucheck holds the fused path at rel_l2 8e-8 against norm-then-silu.
  2. **Chunked decode** (`a0cbb21`), `FASTVIDEO_VAE_CHUNK`. Decode ran one latent frame per pass — 33 sequential trips through the decoder. The causal conv cache makes a chunk equivalent to the same frames one at a time. **19.49 s -> 17.81 s (-8.6%) at chunk=2**, with 129 frames and a `clipped_fraction` identical to 18 digits, so the video is unchanged.
- **chunk=4 OOMs on 24 GB** (`CUDA_ERROR_OUT_OF_MEMORY`), so the default stays 1: an OOM on a smaller card is a far worse failure than a missed 9%. The flag is opt-in and documented.
- Latent frame 0 must stay its own pass: the temporal upsamplers detect the first pass through an empty cache slot and skip doubling, which is what makes the output 4n+1 rather than 4n. A unit test pins it — folding frame 0 into a chunk turns 17 frames into 14, and the test fails.
- Reason: chunking helped **less** than predicted. 33 sequential passes suggested launch overhead dominated, but halving the passes bought only 8.6%, so decode cost tracks the work rather than the pass count. The pure traffic reduction was the bigger lever.
- Reversibility: cheap (fusion is numerically identical; chunking is one env flag, default unchanged)
- Executed by: Executor
- Verification: runs `20260918T004528Z-clip` (fusion, pass, $0.110) and `20260918T011336Z-clip` (chunk=4 OOM, then chunk=2 pass). A 1.98 s "decode" at chunk=4 was an OOM part-way, not a 10x win — it exceeded what removing 3.7x of the passes could explain, which is what prompted checking.

### FVID · 2026-09-18 · FVID-2026-09-18-vsa-port
- Trigger: the head-to-head (FVID-2026-09-17-upstream-head-to-head) put upstream 1.96x ahead on the same GPU, and the gap was attributable to Video Sparse Attention, which we had not ported
- Options: leave dense; write FlashAttention-2 properly; port VSA with a fused block-sparse kernel; port VSA reusing cuBLAS via a gather
- Decision: **ported VSA**, reading upstream's implementation rather than inferring it. `(4,4,4)` tiles of 64 slots; coarse stage mean-pools Q/K/V per tile and attends over tiles; top-k of those scores picks the tiles the fine stage attends at full resolution; `out = coarse * to_gate_compress + sparse`. The fine stage **gathers** each query tile's selected K/V into a dense bf16 buffer and runs batched GEMMs, rather than a fused kernel: the gather moves ~53 GB/layer against dense attention's ~440 GB of score traffic, and it reuses the tensor-core path that is already fast here.
- **Coarse stage is pinned to F32** regardless of `GemmMath`. Tile selection is discrete: in fast mode bf16 rounding flipped a near-tie, a different tile was attended, and the largest test grid moved rel_l2 0.003 -> 0.043. The coarse stage is 819x819 and costs almost nothing, so selection no longer depends on the math mode.
- Result on an RTX 3090 Ti, 8s clip (448x832, 129 frames, 3 DMD steps): **denoise 82.7 s -> 45.6 s, 1.81x**, 27.6 -> 15.1 s/step, reproducible across both prompts to 15 ms, every quality gate passing. Scaling matches the algorithm: 2x SLOWER at 1,456 tokens (fixed overhead dominates), parity at 4,368, 16% faster at 13,104, 1.81x at 48,048.
- Quality: composition and subject are preserved under the same seed, but VSA is a different sample with visibly more saturated colour (`clipped_fraction` 0.0218 vs 0.008 dense). Not verified against upstream's own VSA output.
- Reversibility: cheap (opt-in; `FASTVIDEO_VSA=1` / gpucheck `--vsa`, and it needs a checkpoint carrying `to_gate_compress`)
- Executed by: Executor
- Verification: host reference equals dense attention exactly when every tile is selected (the load-bearing test); device kernels match that reference across four grids and two group sizes (run `20260917T231353Z-kernels`); end-to-end run `20260918T001353Z-clip` pass, 24 stages, $0.089. Four integration bugs found by live runs, none in the kernels: VSA leaking into dense-reference stages, `--vsa` parsed but never acted on, the legacy host-only sparse branch, and a missing head merge before `to_out`.
- Open: our generation is now ~71 s (45.6 denoise + 21.9 VAE + 3.8 write) against upstream's 55.3 s. VAE decode is the largest remaining single cost. A fused block-sparse kernel would remove the gather.

### FVID · 2026-09-17 · FVID-2026-09-17-upstream-head-to-head
- Trigger: user asked how our numbers compare with upstream FastVideo on the same hardware
- Options: compare against published figures (H100, different GPU); rent two boxes; run both implementations on ONE rented box
- Decision: added a **`compare` tier (T4)** that runs our clip stages and then installs upstream FastVideo in its own uv venv on the SAME instance and times the same 8s clip. Identical hardware by construction. The tier skips T1/T2 (it benchmarks, it does not re-validate). Model load is timed separately and excluded; one warm-up then median of 2.
- Result on an RTX 3090 Ti (448×832, 129 frames, 3 DMD steps, timesteps 1000/757/522, same FastWan2.1-T2V-1.3B weights): **upstream 55.30 s per generation vs ours ~108.4 s (denoise 82.74 + VAE 21.86 + write 3.80) — upstream ≈1.96× faster.** Upstream load 49.89 s, warm-up 95.85 s (Triton JIT; a single-shot benchmark would have called upstream SLOWER than us).
- **The dense comparison does not exist on upstream's side for these weights**: `TORCH_SDPA` cannot load the checkpoint — `Parameter blocks.0.to_gate_compress.bias not found in custom model state dict` — because FastWan ships VSA gate weights their dense model class does not define. Upstream on this model IS the VSA configuration.
- Reason: the gap is the optimization we deliberately have not ported. VSA cuts the quadratic attention term; our 48k-token forward is dominated by dense attention, which is what the flash experiment showed from the other side. ~2× is the measured price of no sparse attention.
- Reversibility: free (new tier, nothing in the shipped path changed)
- Executed by: Executor
- Verification: run `20260917T220533Z-compare`, upstream torch 2.12.0+cu126 / fastvideo 0.2.1. Nine attempts to get here; the failures were ours, not upstream's — see FVID-2026-09-17-exact-parity-24gb-oom and the harness fixes (bad-host recording on mid-run ssh death, disk precheck crediting fetched data, empty-array expansion under bash 3.2, shell lint).

### FVID · 2026-09-17 · FVID-2026-09-17-exact-parity-24gb-oom
- Trigger: the `compare` tier OOM'd in `parity-exact` on two different 24GB cards — a plain RTX 3090 (24124 MiB total, 23846 free, no other tenant) and an RTX 3090 Ti (24112 MiB, 23829 free) — failing on the FIRST `dit_forward_t999` right after `load`. The same stage passes on an RTX A5000 (24111 MiB) and on a larger 3090 Ti (24564 MiB)
- Options: rent only >24GB cards; cut exact-mode peak; give the mempool a finite release threshold; leave it and document
- Decision: **not fixed yet, recorded and routed around.** The `compare` tier now skips T1/T2 (it benchmarks, it does not re-validate) so it never runs exact-mode parity; the validation tiers still do, on cards that fit. Not diagnosed further under a benchmarking task — the suspects are the mempool release threshold (`u64::MAX`, so nothing returns to the driver) and cuDNN workspace from conv3d auto-selection, which times BOTH backends per shape and so allocates a cuDNN plan's workspace even where unfold wins.
- Reason: exact mode is F32 everywhere and is the memory-hungriest configuration; it is a validation path, not a production one, so a 24GB card failing it blocks validation but not use. Worth fixing, but not by guessing mid-benchmark
- Reversibility: free (no behaviour changed)
- Executed by: Executor
- Verification: runs `20260917T202914Z-compare` (3090) and `20260917T204412Z-compare` (3090 Ti), both rc=2 at `parity-exact`, $0.035 and ~$0.01. Next diagnostic: rerun the `parity` tier on a 24GB box with `FV_STAGE_ENV="FASTVIDEO_CONV3D=unfold"` — if it passes, cuDNN workspace is the cause; if not, the mempool is.

### FVID · 2026-09-17 · FVID-2026-09-17-bf16-attention-probs
- Trigger: after flash SDPA was rejected (FVID-2026-09-17-flash-sdpa-rejected), the dominant remaining cost in a clip-scale forward was dense attention's probability matrix (`bh*sq*sk`), written by softmax and read back by the `P@V` GEMM
- Options: leave it; store probabilities as bf16; also store pre-softmax scores as bf16; rewrite attention with two-level tiling; port VSA
- Decision: **probabilities are stored as bf16 under `GemmMath::Bf16`** (`FASTVIDEO_ATTN_PROBS_BF16=0` opts out; exact mode unchanged). New `softmax_last_bf16` kernel writes them directly — three passes over the f32 scores (max, sum, write) rather than keeping f32 exponentials, since a 48k-wide row cannot live in shared memory — `V` is cast once per call, and `gemm_raw` gained an A/B dtype parameter so `P@V` takes bf16 operands with an F32 result. Pre-softmax scores stay F32: bf16 there perturbs `exp()` by ~2% at score magnitudes around 10, far above the fast-mode error budget.
- Reason: under `CUBLAS_COMPUTE_32F_FAST_16BF` cuBLAS already rounds F32 operands to bf16 for the tensor-core op, so materialising the probabilities as bf16 is the same arithmetic over half the bytes — a pure traffic win, not a precision trade
- Reversibility: cheap (one env flag; exact mode never takes the path)
- Executed by: Executor
- Verification: same-instance A/B on 51338983 (sm_86), runs `20260917T190610Z-clip` (F32, via `FV_STAGE_ENV="FASTVIDEO_ATTN_PROBS_BF16=0"`) and `20260917T193716Z-clip` (bf16), 22 stages PASS each. **8s clip denoise 128.5 s → 82.7 s (-35.6%)**, 42.8 → 27.6 s/step; total clip 161.9 s → 115.8 s. Forward gain grows with sequence: 5.4% at 1,456 tokens, 15.3% at 4,368, 24.8% at 13,104. Peak denoise memory 9,859 → 9,539 MiB; probe peak 7,427 → 5,827 MiB. Accuracy unmoved: parity fast rel_l2 0.004515 → 0.004498, exact bit-identical at 2.355890268907824e-6; fast-vs-exact step-1 0.0166 → 0.0180 (gate 0.05), final 0.2035 → 0.2114 (gate 0.35), 24.06 → 23.69 dB (gate 20). New `softmax_bf16` kernel check: rel_l2 8e-4–1.9e-3 vs an f64 reference. Cost $0.183 for both halves.

### FVID · 2026-09-17 · FVID-2026-09-17-flash-sdpa-rejected
- Trigger: at 48k tokens (8s clip) a DiT forward costs ~43 s with the default dense SDPA, and attention dominates; the opt-in `flash_attn_f32` kernel was never measured at clip scale, so `FASTVIDEO_SDPA=flash` was A/B'd on a pinned RTX A5000 against run `20260917T175630Z-clip`
- Options: adopt flash as default; tune the existing kernel; rewrite it with query tiling + tensor cores; keep dense and cut its memory traffic; port upstream VSA sparse attention
- Decision: **flash stays opt-in and off**; dense remains the default. The kernel is not tunable into competitiveness — `cfg_flash` launches one block per query row (`grid = bh × sq`, `block = d`), so at 48k tokens 576,576 blocks of 128 threads each stream the WHOLE of K and V through shared memory. K/V traffic per attention call is `bh × sq × sk × d × 2 × 4 B` ≈ 28 TB, hundreds of TB per forward across 30 layers; at ~768 GB/s that is the measured cost. It trades away the materialized score matrix (110 GB/layer) for K/V re-reads three orders of magnitude larger, so the penalty GROWS with sequence length: measured 12.6× slower at 1,456 tokens, 21× at 4,368, 35.8× at 13,104, and the probe's budget gate aborted the run projecting 1,795 s per forward at 48,048 tokens (vs 43 s dense). It is also scalar FMA + warp-shuffle per key, forfeiting the tensor cores cuBLAS gets in the dense path.
- Reason: the hypothesis was that a flash-style kernel wins at long sequence because it never materializes scores; the measurement says this implementation loses for a different reason (no query tiling ⇒ no K/V reuse), and a real fix is FlashAttention-2 (query tiles, MMA on bf16, double-buffered loads), not tuning
- Reversibility: free (nothing changed in the default path; `FASTVIDEO_SDPA=flash` still selects it)
- Executed by: Executor
- ADR: none (no architecture change; negative result recorded)
- Verification: run `20260917T183200Z-clip` on a pinned RTX A5000, `FV_STAGE_ENV="FASTVIDEO_SDPA=flash"` — kernels/model/parity stages PASS (flash is numerically correct: parity exact rel_l2 2.3e-6), timings above, stage `probe-8s` rc=3 (budget). Cost $0.053. Next step recorded separately: cut dense's score-matrix traffic (bf16 probabilities), then evaluate the VSA port.

### FVID · 2026-09-13 · FVID-2026-09-13-strict-device-check
- Trigger: user asked how to verify GPU code paths never silently execute on CPU — every device op added in FVID-2026-09-13-cudarc-perf-pass is structured as "try the device kernel, silently compute on host if it returns `None` for any reason", which is correct for portability but means a real GPU run where something's subtly broken (bad shape guard, kernel launch failure, residency toggled off) still produces correct output, just slower, with nothing in the logs
- Options: no tooling (rely on manual profiling to notice a slowdown); log-only auditing; hard-fail strict mode; full distributed tracing
- Decision: added `tensor::strict_device_check(op, detail)` — when `FASTVIDEO_STRICT_DEVICE=1`, residency is enabled, and a CUDA device is live, a hot op (`layer_norm`, `modulate`, `gate_mul`, `attention`) falling back to the true host compute path returns a hard `Err` naming the op and shapes, instead of silently continuing; ops with no device kernel at all (`div`, `mean_keepdim`, `sqrt`, non-trailing-axis `narrow`/`chunk`/`cat`, sparse block-attention) are deliberately NOT wired to this — those are known, permanent gaps, not per-call regressions, and strict mode would just always fire there with no diagnostic value. Paired with a non-fatal counterpart: `tensor::device_path_stats()` (plain atomic hit/miss counters per op) and `FASTVIDEO_DEVICE_STATS=1`, which prints a device-vs-host dispatch summary after `generate()` without changing behavior — for auditing a run without risking a crash.
- Reason: user wants a concrete way to catch a silent GPU→CPU fallback regression before it just shows up as an unexplained slowdown
- Reversibility: cheap (both flags default off; zero behavior change unless explicitly set)
- Executed by: Executor
- ADR: ADR-0003 (amended — see "Consequences: first GPU run should re-verify")
- Verification: `cargo test --workspace --lib` (117 tests, 0 failures), incl. `strict_device_check_is_noop_without_a_live_device` (proves strict mode never fires on this host-only sandbox — the only half of its contract testable without a GPU) and `device_path_stats_record_and_reset` (counter bookkeeping). **The half that matters most — strict mode actually catching a live-device fallback — is untested here by construction** (no GPU in this sandbox); the intended workflow is `FASTVIDEO_STRICT_DEVICE=1` on a real Vast run as a smoke/CI gate.

### FVID · 2026-09-13 · FVID-2026-09-13-cudarc-perf-pass
- Trigger: performance review of `fastvideo-cudarc` found `layer_norm`/AdaLN-modulate host round trips defeating device residency every DiT block, one-thread-per-row softmax/rms_norm kernels, sequential (non-parallel) CFG and sequence-parallel dispatch, `Mmap`-free weight loading, and a chunked-SDPA path that gather/scatter-copied every chunk instead of using cuBLAS's native inter-batch stride
- Options: full rewrite incl. true flash-attention kernel + NCCL P2P + Drop-based scratch allocator; scoped fixes only where verifiable without a live GPU (this machine has no nvcc/driver — verification loop is `CUDARC_CUDA_VERSION=<pinned> cargo check --features cuda`, which type-checks cudarc API usage and NVRTC *Rust* call sites but cannot compile/run the NVRTC CUDA-C kernel strings themselves, and cannot exercise any code gated on a live `device::global_device()`)
- Decision: implemented and test-verified: device `layer_norm_last`/`modulate_scale_shift_last`/`broadcast_mul_last` NVRTC kernels (replacing AdaLN's `broadcast_bin` host path — see ADR-0003); block-reduction `softmax_last`/`rms_norm_last` (was one-thread-per-row, uncoalesced); cached every hot-path `FASTVIDEO_*` env-flag read behind `envflag::CachedBool`/`CachedString` (were re-parsing `std::env::var` per op); `[profile.release]` (LTO, codegen-units=1, panic=abort); batched cond/uncond CFG into one batch=2 forward pass (scoped off for I2V/image-conditioning — would need `mask`/`cond`/`image` also duplicated, not verifiable blind); real per-rank multi-GPU dispatch for `FASTVIDEO_SP_WORLD>1` (`device::device_for_index` + thread-local device override + `std::thread::scope` fan-out, host-mediated gather) — `sp::device_for_rank` existed but was dead code, every rank ran on one GPU sequentially before this; `Mmap`+`rayon`-parallel weight loading in `fastvideo-loader` (was `std::fs::read` + single-threaded scalar dtype conversion), needed a crate-local `unsafe_code = allow` override (same pattern as `fastvideo-cudarc`'s CUDA-FFI override) for `Mmap::map`; chunked SDPA now addresses `Q`/output via `CudaView` offset + explicit inter-batch stride (`matmul_linear_wt_strided_batched_x_view` / `matmul_2d_strided_batched_out_view`) instead of `memcpy_dtod`-gathering/scattering every chunk. Also fixed two pre-existing `--features cuda` compile breaks unrelated to the above (`Bf16Activation` missing `Debug`, wrong `CUgraphInstantiate_flags` variant name) found while building the verification loop — `cargo check --features cuda` did not compile before this pass, on any of these changes.
- Deliberately NOT done, scoped out as needing GPU-supervised verification rather than attempted blind: a generic Drop-based scratch-buffer pool for the remaining `alloc_zeros` churn (every elementwise/GEMM op still allocates fresh) — a pool needs to hook `CudaSlice` recycling into `Arc<DeviceBuffer>`'s drop path, and a wrong reference-count/lifetime edge case there is silent data corruption (a live tensor's buffer handed back to the pool and overwritten), not a crash; a fused/tiled flash-attention kernel (true O(N) memory attention) — writing a correct online-softmax CUDA kernel with shared-memory tiling blind, with no way to check its numerics against a reference on real hardware, is the wrong risk/verification tradeoff for this pass. Both remain real opportunities; do them with a GPU in the loop.
- Reason: user asked to integrate all recommendations from the perf review; prioritized changes verifiable via unit tests (host-fallback math, chunk-loop stride arithmetic, end-to-end tiny-model `generate()`) and `--features cuda` type-checking over ones that can only be checked by running real CUDA kernels
- Reversibility: medium (each item is an isolated function/module change; the two scoped-out items were never started, so nothing to revert there)
- Executed by: Executor
- ADR: ADR-0001 (amended), ADR-0003 (new — see below)
- Verification: `cargo test --workspace --lib` (113 tests, 0 failures, incl. new tests for `layer_norm`/`modulate`/`gate_mul` against naive references, a CFG-batched end-to-end tiny `generate()` run, a real mmap-backed multi-file/mixed-dtype loader round trip, and the chunked-SDPA offset/stride arithmetic across even/remainder/edge-case chunk sizes) + `CUDARC_CUDA_VERSION=12040 cargo check -p fastvideo-cudarc --features cuda` clean. **Still needed before trusting this on real workloads: an actual GPU run** — none of the above proves the NVRTC kernel *source* compiles under nvcc/NVRTC, that the new kernels' numerics match a reference on real hardware, or that the multi-GPU thread/context fan-out behaves correctly under real concurrent CUDA calls.

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-primary-parity
- Trigger: user chose cudarc-only focus; close Burn/Candle feature work; Wan inference then speed parity
- Options: keep multi-backend feature parity; freeze others and push cudarc
- Decision: CLI/scripts default to cudarc + `cuda-cudarc`; Candle/Burn/Luminal frozen; cudarc gains MoE, I2V+CLIP+VAE encode, causal mask, MP4 mux (`FASTVIDEO_SAVE_MP4`), TeaCache (`FASTVIDEO_TEACACHE`), chunked SDPA, `CachedLinear` residency (`FASTVIDEO_RESIDENT`), reject `num_gpus>1` until SP, warn on VSA ids
- Reason: plan locked cudarc as sole generate path to optimize
- Reversibility: medium (frozen backends remain compilable)
- Executed by: Executor
- ADR: ADR-0001 / ADR-0002 (amended cudarc-primary)
- Verification: `cargo test -p fastvideo-cudarc -p fastvideo-core --lib`

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-gpu-ops
- Trigger: cudarc v0 ran non-GEMM ops on host f32 after native weight load
- Options: host-only; NVRTC+cuDNN with H2D/D2H per op; full device-resident CudaTensor
- Decision: NVRTC elementwise/activations/softmax/rms_norm + cuDNN conv2d when global device is set; keep host Vec API; causal conv3d/broadcast stay host
- Reason: user asked to land the next speed step without blocking on full residency
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (amended)
- Verification: `cargo test -p fastvideo-cudarc --lib` + `fastvideo-core` lib (CUDA path compile/smoke pending Vast; Mac has no nvcc)

### FVID · 2026-09-10 · FVID-2026-09-10-cudarc-backend
- Trigger: Burn CubeCL CUDA ~2× slower than Candle on H100 smoke; need a thinner CUDA path
- Options: optimize Burn; cudarc fourth backend; Candle-only
- Decision: add `fastvideo-cudarc` as `--backend cudarc` with full Diffusers Wan load/generate; crate-local `unsafe_code = allow`; native BF16 loader path
- Reason: user chose cudarc as fourth generate backend with no stub bring-up
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (amended)
- Verification: `cargo test -p fastvideo-cudarc --lib` + workspace lib tests (CUDA smoke on Vast)

### FVID · 2026-09-10 · FVID-2026-09-10-h100-candle-vs-burn
- Trigger: user asked to benchmark candle vs burn on H100
- Options: 4090 only; H100 50435452 smoke 256² 9f/2step
- Decision: ran both on H100 SXM (`50435452`); candle CUDA BF16 77.3s vs burn CUDA F32 162.9s (load+generate)
- Reason: fair GPU compare of native Burn path vs proven Candle
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0002
- Verification: artifacts under `artifacts/20260910T010910Z-nvidia-h100-80gb-hbm3-i50435452-all/`

### FVID · 2026-09-09 · FVID-2026-09-09-burn-cuda-luminal-compile
- Trigger: native Wan ports were CPU-only; ADR-0001 needs Luminal compile-once DiT/VAE
- Options: Burn CubeCL cuda feature vs keep ndarray; real luminal Graph vs eager facade
- Decision: `fastvideo-burn` `--features cuda` → `Cuda<f32>` + device from CLI; luminal 0.2 `GenericCompiler`+`CPUCompiler` for tiny fixed-shape DiT step + VAE decode (eager fallback for full 1.3B)
- Reason: user asked to implement burn-cuda and luminal compile
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001
- Verification: `cargo test -p fastvideo-burn -p fastvideo-luminal -p fastvideo-core --lib` green (CUDA feature needs Vast)

### FVID · 2026-09-09 · FVID-2026-09-09-native-wan-ports
- Trigger: Burn/Luminal only wrote stub UniPC latents; need fair FastVideo/Wan benches
- Options: TensorBackend generic graph; native per-runtime Wan ports; Candle-only
- Decision: native UMT5+DiT+feat-cache VAE in `fastvideo-burn` (Burn 0.18 ndarray) and `fastvideo-luminal` (host NdTensor + compiled facades); no Candle fallback in generate; TensorBackend deferred
- Reason: user asked for native ports first, validate abstraction later
- Reversibility: medium
- Executed by: Executor
- ADR: ADR-0001 (supersedes stub adapters for generate)
- Verification: `cargo test --workspace --lib` green; tiny PNG on burn/luminal/candle; Diffusers `WanPipeline::load` wired

### FVID · 2026-09-09 · FVID-2026-09-09-luminal
- Trigger: third Rust ML backend named “luminar”
- Options: Luminal (graph compiler) vs some other crate
- Decision: Luminal (https://github.com/luminal-ai/luminal)
- Reason: user confirmed that spelling
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0001
- Verification: checked against source

### FVID · 2026-09-09 · FVID-2026-09-09-wan-inference
- Trigger: FastVideo is a full post-training + inference monorepo
- Options: 1.3B-only inference; all Wan/FastWan inference; inference + training
- Decision: inference-first for every Wan/FastWan family in FastVideo’s registry
- Reason: user chose all Wan families, still no training
- Reversibility: cheap
- Executed by: Executor
- ADR: none
- Verification: checked against FastVideo `fastvideo/models/wan/definition.py`

### FVID · 2026-09-09 · FVID-2026-09-09-tensor-backend
- Trigger: need one model implementation for three runtimes
- Options: write models three times; Burn-only + export; custom TensorBackend trait
- Decision: custom `TensorBackend` trait; do not use deprecated `burn-candle`
- Reason: Burn, Candle, and Luminal are distinct runtimes; Luminal is graph-based
- Reversibility: costly
- Executed by: Executor
- ADR: ADR-0001
- Verification: pending

### FVID · 2026-09-09 · FVID-2026-09-09-vast-gpu
- Trigger: CPU `--tiny` is not a real inference bar; need GPU for Wan 1.3B
- Options: local Metal; wait for Luminal GPU; CUDA on Vast.ai
- Decision: real GPU runs on Vast CUDA hardware (Candle `--features cuda`, BF16)
- Reason: user directed GPU runs onto Vast
- Reversibility: cheap
- Executed by: Executor
- ADR: ADR-0002
- Verification: gates green (tiny CUDA F32 generate on RTX 4090 instance 50416610)

### FVID · 2026-10-21 · FVID-2026-10-21-segments-clip-path
- Trigger: the owner: "we need this to work with h3 and ltx as well" (`/v1/segments` served only the causal SF-Wan path)
- Options: one engine clip job per segment with the control plane chaining on `context.image_url` (its `last_frame` mode, a frame over the wire per segment); LTX extend from the previous segment's tail (not reachable through `ClipSession`, 2 s minimum extension, 8-frame rounding); the backend chaining on its own last frame over `ClipSession` the way the fal director chains chunks
- Decision: the director's way: a clip session per stream, one build per segment anchored on the previous segment's last frame (`Task::I2V`), the duplicated frame and its audio dropped, LTX generated padded and cropped to the tier with the anchor uncropped; `context.image_url` and `end_image_url` accepted on clip models, `context.video_url` refused on both paths; clip lengths snap onto the model's grid and the 202 object reports `planned_frames` / `planned_duration_sec`, which the control plane's custom provider uses as the segment's length
- Reason: it is the continuity the engine already has for H3 and LTX, proven by the director; `continuity: "native"` keeps working unchanged for the control plane; extend does not fit the contract's lengths
- Reversibility: cheap (a second worker in the same router; the engine is unchanged)
- Executed by: Claude (session_013xBkLaNmFvEuC4BKWpjuHc)
- ADR: none
- Verification: `tests/segments.rs` on the fake LTX and H3 models (anchor, reset, image and keyframe inputs, audio track, tier alias, grid); GPU smoke 2026-10-21 on an RTX PRO 6000 with the EU volume (docs/serve/segments.md §6): LTX-2.5 turbo chains seamlessly (seam 4.4 vs 3.0 inside the clip), 10 s per 5 s segment; H3 turbo and H3 max do not reproduce the anchor at all (`seam.anchor_diff` 36–44, the level of two unrelated scenes), a first-frame conditioning finding for the H3 port that the router now measures (`seam`)

### FVID · 2026-10-21 · FVID-2026-10-21-segments-adapter
- Trigger: the unreel-cloudflare control plane needs a render backend with native continuity (its `custom` provider contract: `POST /v1/segments`, `GET /v1/segments/{id}`, `video_url` per segment)
- Options: one clip job per segment (last-frame anchoring, VAE round trip per seam); a live `/fv/v1/streams` session the control plane consumes over WHEP; one causal session per `stream_id` cut into segments after the fact
- Decision: one causal session per `stream_id`, segments cut from the continuous rollout with frame carry-over (`crates/fastvideo-serve/src/segments.rs`, docs/serve/segments.md); `reset` for scene cuts, `context.*` refused on the causal path, the control plane publishes
- Reason: the seam only exists for scheduling; the KV cache, sink and decoder state carry across segments; the control plane keeps its own WHIP publisher and buffer
- Reversibility: cheap (a router over existing engine sessions; nothing in the engine changed)
- Executed by: Claude (session_013xBkLaNmFvEuC4BKWpjuHc)
- ADR: none
- Verification: CPU tests on the fake causal model (`tests/segments.rs`); GPU smoke 2026-10-21 on an H100 (docs/serve/segments.md §6): four 6 s segments, 2.9 s generation per segment in steady state, the seam between segments 0 and 1 indistinguishable from an inter-frame difference, `stale_blocks`/`discarded_blocks` as designed

### FVID · 2026-10-24 · FVID-2026-10-24-a4-toml-parser
- Trigger: plan A4 needs `control/gen-configs.mjs` (plain Node) to read `families/*.toml`; control had no TOML parser (wrangler bundles its own, not importable) and Node 22 has none
- Options: a hand-written TOML-subset parser in the generator; `smol-toml`; `@iarna/toml` (TOML 0.5, unmaintained)
- Decision: `smol-toml` 1.9.1 as a pinned devDependency of `control/` (TOML 1.0, zero dependencies); the generator validates the shape it reads (`parseFamily`) and tests pin a fixture
- Reason: the family files use tables, arrays of tables and typed arrays a subset parser would get subtly wrong; the dependency is small, well known and dev-only
- Reversibility: cheap (one import)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none (ADR-0008 covers the format)
- Verification: `node gen-configs.mjs --check`, `test/unit/families-gen.test.ts`

### FVID · 2026-10-24 · FVID-2026-10-24-a4-generator-scope
- Trigger: plan A4 lists presets, `enums.ts`, `variants.sh`, `families.tsv`, autoscale pools, `verify-weights.sh` cells and the `images.md` table as generated; the family manifests carry no deployment data (image variant, config, pool sizes, machine types, cells by component)
- Options: invent a `[deploy]` section now; generate only what the manifests derive and cross-check the rest; leave everything hand-written until C1
- Decision: the generator owns the identity-derived parts only (`families-gen.ts`, `volumes.ts` `EU_TREES` from `volumes.sh`, `families.tsv` `weight_trees` / `served`, `verify-weights.sh` `variant_trees`), consumers derive from the tables (`recipe-trees.ts`, every model preset's `weights` / `min_vram_gb`, the `family` enum) and `--check` cross-checks the hand-written copies (`catalog.json`, `autoscale.toml`, `volumes.sh`); deployment data stays hand-written for C1. Names the manifests lack (`longlive-1.3b`, `lightx2v-turbo-v0.1`) are recorded exceptions, not invented variants
- Reason: no invented data; the family files are A3's to change; a generated region with a hand-written source is still one source
- Reversibility: cheap (C1 extends the same generator)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none
- Verification: `gen-configs.mjs --check` and the control unit tests in `.github/workflows/control.yml`

### FVID · 2026-10-24 · FVID-2026-10-24-a4-control-workflow
- Trigger: A4 needs `gen-configs.mjs --check` and the control unit tests in CI; no workflow ran them (control's vitest suite was not in CI, which let one test go stale after F2)
- Options: a job in `serve-compat.yml` (its `paths` would then run the ten compat suites on every control change); a new node-only workflow
- Decision: `.github/workflows/control.yml`: `npm ci`, `--check`, typecheck, vitest, `scripts/gpu/lint.sh`, on changes to the generator's inputs and outputs
- Reason: minutes instead of a Rust build and a browser matrix; the same pinned-by-tag action style
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none
- Verification: the workflow on the A4 PR

### FVID · 2026-10-24 · FVID-2026-10-24-deploy-section
- Trigger: plan C1 asks for the deployment half of the family manifest (review §7.4) and the deployment tables A4 left hand-written (presets, `enums.ts` `VARIANTS` / `POOL_PRESET_IDS`, `variants.sh`, `families.tsv` rows, autoscale pools, the `images.md` table, the Dockerfile variant stages) generated from it
- Options: a `[deploy]` table per variant in `families/*.toml`; one `deploy/<family>.toml` per family; one `deploy/deploy.toml`; types in `fastvideo-manifest` or in `fastvideo-deploy`
- Decision: one `deploy/deploy.toml` (images, model presets = one pool each, family presets, GCP rows, dispatch map, autoscale pools), typed and validated in `fastvideo_manifest::deploy` against the `FamilySet` (variant ids exist, recipe names resolve to them, family-preset card class meets the variants' floor, only the EU volume `bz0ofkfx45` in `EU-RO-1`), schema `deploy.v1.json` with a drift test; `control/gen-configs.mjs` generates every table from it with `--check`; the single CUDA image with boot-time config is analysed in the ADR and left to the owner
- Reason: the engine embeds the family files, so deployment edits there would rebuild it; the deployment unit is a pool, not a variant; the dashboard and the autoscaler keep cross-family orders and some entries have no family; validation needs the family types and no server stack (`fastvideo-deploy` is the fv-serve queue worker)
- Reversibility: cheap (this repository's format, versioned by `schema`; not a registry contract)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0011-deploy-section.md
- Verification: `cargo test -p fastvideo-manifest` (`deploy::tests`, `tests/deploy.rs`, `tests/schema.rs`), `gen-configs.mjs --check`, `control/test/unit/deploy-gen.test.ts`

### FVID · 2026-10-24 · FVID-2026-10-24-manifest-tests-in-ci
- Trigger: no workflow ran `fastvideo-manifest`'s tests, so the deployment file's Rust validation and the schema drift would only run on the build pod
- Options: a new workflow; add the crate to the adapter-tests set (`scripts/serve/coverage.sh`) with `crates/fastvideo-manifest/**` and `deploy/deploy.toml` in its paths
- Decision: the adapter-tests set (one more crate, no new workflow), as D2 did for `fastvideo-lab`
- Reason: one place for the CPU crates' tests and coverage; `families/**` already triggers it
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none
- Verification: `bash scripts/serve/coverage.sh --tests` on the build pod (13 crates, 819 passed)

### FVID · 2026-10-24 · FVID-2026-10-24-b2-family-trait
- Trigger: plan B2: replace the 21 registry tables, `resolve()`, `all_registered_ids()` and the 20 `generate_*` functions with one trait and a static family list, CLI behaviour unchanged
- Options: (a) the review's sketch verbatim (`Conditioning` one enum value, `load` places weights, Hub ids from the manifest); (b) the same trait with `conditioning: Vec<Conditioning>`, `load` that plans and `generate` that loads, the Hub-id tables in one Rust module per family
- Decision: (b), all 21 families moved in one PR (no stack needed: the moves are verbatim and the CPU goldens pass unmodified)
- Reason: the CLI takes `--image`, `--last-image`, `--ref`, `--control` together and each family has always picked its own (LTX with `--image --ref` runs I2V, H3 refuses FL2VA with Ref2VA); one enum value would drop inputs and change those behaviours. Every family but Wan loads and generates in one call today (B6 splits the loaders). The family files describe the 30 served variants, not the CLI's 63 Hub presets, and `family.rs` / `families/*.toml` are C1's this round
- Reversibility: cheap (the trait does not change when `variants()` reads Hub ids from a widened manifest)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0013
- Verification: `crates/fastvideo-cli/tests/cli_golden.rs` and `crates/fastvideo-core/tests/resolve_golden.rs`, taken on the unchanged tree, pass unmodified; `family::tests` (83 Hub ids resolve to their family)

### FVID · 2026-10-24 · FVID-2026-10-24-b2-validate-not-in-cli
- Trigger: `Request::validate(&ModelCaps)` is the one check the review asks for; the CLI's presets have no `ModelCaps`
- Options: validate CLI requests against the nearest served variant's caps; leave the CLI on each family's own checks and make validate the serve path's
- Decision: the CLI does not call `validate`; `engine-service` adopting it is left to E2 (it owns `validate.rs` and `fastvideo-protocol` this round)
- Reason: caps on the CLI would refuse what it accepts today (the tiny goldens run at 64x64x5, below every served grid)
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0013
- Verification: `request::tests` (tasks, conflicts, canvas, frames, seconds, fps, refs, knobs)

### FVID · 2026-10-24 · FVID-2026-10-24-param-spec
- Trigger: plan E2: request limits duplicated across the adapters and disagreeing between surfaces (review §4.4, §7.6 item 2); E3 needs a machine-readable parameter list
- Options: per-adapter spec types; one `ParamSpec` type in `fastvideo-protocol` with the limits in each adapter; the type and every upstream limit in `fastvideo-protocol::params` (adapters keep their wire enums and wording)
- Decision: the last: `ParamSpec` / `Param` / `Domain` / `Support` and `params::rules` (one cited constant per upstream limit, model facts derived from `fastvideo_models`) in the protocol crate; each adapter builds its specs from the rules and the caps; `GET /fv/v1/params` serves them; the LTX bodies are typed; a scan test refuses numeric limits in adapter validation code
- Reason: one source per limit with no dependency cycle; adapters' error texts stay theirs
- Reversibility: moderate (the endpoint and the golden are new public surface)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: ADR-0012
- Verification: `param_scan`, `param_rules`, the adapters' `params` agreement tests, `ltxapi::body` tests, `param_spec_golden`, `capabilities_golden` unchanged

### FVID · 2026-10-24 · FVID-2026-10-24-guidance-scale-2-bound
- Trigger: E2 census: `/v1/videos` bounds `guidance_scale` to 0..=20 but `guidance_scale_2` only to >= 0 (negotiation); FastVideo's `protocol.py` gives both `ge=0, le=20`
- Options: keep the gap; bound `guidance_scale_2` to 0..=20 like upstream
- Decision: bound it at parse time (400 `guidance_scale_2 must be within 0..=20`, the `guidance_scale` wording). Full list of message changes, same status and code in each: above 20 refused (was accepted); a negative value gets the range message instead of "must be a non-negative number"; on a model without the `guidance_2` knob an out-of-range value gets the range message instead of "not supported by model"; reported before `boundary_ratio`, the task / image checks and negotiation errors
- Reason: the vendored upstream contract wins where two places disagree (specs README rule); only upstream-invalid requests change
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: ADR-0012
- Verification: `fastvideo-openai-videos/tests/params.rs` (agreement), `param_rules::fastvideo_videos_rules`

### FVID · 2026-10-24 · FVID-2026-10-24-upstream-divergences-kept
- Trigger: E2 census against the vendored specs: fal `minimax/h3-max*` `duration` is now a number 0.92..=15 (ours integer 5..=15); FastVideo seeds may be negative (ours >= 0); OpenAI `limit` minimum 0 (ours 1); FastVideo's new `audio_start_in_s` / `audio_end_in_s` and fal h3-max r2v's `middle_*` / `reference_conditions` are not read
- Options: follow upstream now (accepts requests refused today, and needs engine work for sub-5 s H3 on fal); keep and pin
- Decision: keep this server's behaviour, pin each upstream value in `param_rules` next to ours, list them in docs/serve/params.md for the owner and E3
- Reason: E2 must not change behaviour for valid requests or existing errors; widening is a product decision
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: ADR-0012
- Verification: `param_rules` (fails if either side moves)

### FVID · 2026-10-24 · FVID-2026-10-24-cert-matrix
- Trigger: plan E3: a parameter x surface x model certification matrix over every surface, as a TSV, with a CI assertion that every parameter has a valid and an invalid case (review §7.6 item 3)
- Options: a new test crate; a test in `fastvideo-serve` over the assembled router; per-adapter tests at parse time only
- Decision: a `fastvideo-serve` integration test (`tests/cert_matrix.rs`) driving the specs each shipped config serves at `/fv/v1/params` through the real router on the fake engine (job checks of the CUDA pipelines, accepted jobs cancelled); the director and Reactor channels through their parsers, with `ParamSpec`s built by their crates; cases from `Domain::valid_example` / `invalid_examples` plus explicit plans for composite fields; output `docs/serve/certification.tsv` with a drift check; the CI assertion plans the golden's specs without a server and requires a reasoned exemption for a parameter with no invalid case (planned without the exemptions, so an exempt parameter with a case is stale)
- Reason: reuses the e2e machinery and the CI that already runs serve tests (no new crate, workflow or dependency); the served specs, not hand-written lists, decide the matrix
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: ADR-0016
- Verification: `cert_matrix::certification_matrix_is_golden`, `cert_matrix::every_param_has_a_valid_and_an_invalid_case`

### FVID · 2026-10-24 · FVID-2026-10-24-cert-findings-recorded
- Trigger: the matrix: 1,114 of 4,630 rows disagree with their spec (naming, spec support, routing, director empty URLs, the two census findings); no 5xx or panic
- Options: fix the wire behaviour in the E3 PR; record each finding in the TSV and list the candidate fixes
- Decision: record and list (docs/serve/params.md "Findings and candidate fixes"); no wire behaviour changes in E3
- Reason: each fix changes what clients see and is the coordinator's / owner's decision; the TSV shows the change when one lands
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: ADR-0016
- Verification: `docs/serve/certification.tsv` `finding` column

### FVID · 2026-10-24 · FVID-2026-10-24-b5-sdpa-entry
- Trigger: plan B5: one `sdpa(q, k, v, Plan)` entry with `sdpa_rule.rs` as the chooser, byte-identical dispatch
- Options: (a) `Plan` = the resolved kernel, built by each caller from the settings and the device; (b) `Plan` = what the call site computes (dense / masked / FP8-first / block-causal / VDN window / KV window / composed), the kernel chosen inside the entry by `sdpa_rule` per query shard, over a `Probe` of the settings and the device
- Decision: (b); `sdpa_rule::dense_route` / `mma_kernel` and the masked-route predicates are the only choosers, the kernels keep declining what they cannot run and `run_route` falls through as the old `if let Some` chain did; the H3 ViT decoder keeps launching the fused kernel (the rule's pick) in its fused chain
- Reason: the kernel depends on the shard's query length and on the calling thread's device (sequence-parallel shards run on other threads), so it cannot be fixed at the call site; the effectful probes (attn_dc load, cuDNN plan build) stay lazy, in the old order
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0015
- Verification: `rt::sdpa_rule::identity` (the merge-base choosers transcribed; identical kernel / argument / probe traces over every setting, device fact, shape, output type and set of declining kernels)

### FVID · 2026-10-24 · FVID-2026-10-24-b5-host-twins
- Trigger: plan B5: retire the host attention twins once `no_host_fallback` proves them unreachable on CUDA
- Options: retire every host attention path; retire only the host-only dense backends; keep all
- Decision: retire `flash_style_sdpa_host` (`FASTVIDEO_SDPA=host`) and `block_sparse_sdpa` (`FASTVIDEO_SDPA=sparse`), refused with a named replacement; keep the composed SDPA (CPU / tiny and masked path) and the sparse routes' host oracles (`vsa_attention_host`, `h3::vsa::attention_host`, `sla_attention_host`, Sol / PISA), which gpucheck and the CPU tests judge the kernels against and which already error with a live device
- Reason: both retired backends were host-only (refused with a live device), duplicated the composed reference on a CPU run, and no config sets them; the oracles are needed
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0015
- Verification: `identity::only_unmasked_host_and_sparse_calls_are_refused`; GPU: `fv-gpucheck kernels` group `no_host_fallback` (every `Plan` on device operands, zero host fallbacks)

### FVID · 2026-10-24 · FVID-2026-10-24-b5-nvfp4-oxide-features
- Trigger: plan B5: `nvfp4_gemm.rs` and the oxide AOT path behind features
- Options: features on by default; off by default with every build that compiles the code today turning them on; off by default, on only where the code is measured
- Decision: `fastvideo-cudarc/nvfp4-gemm` and `fastvideo-cudarc/oxide` (implies it), off by default; `fastvideo-gpucheck/cuda` enables `oxide`, the serve binaries do not
- Reason: no served config, profile or family turns `FASTVIDEO_NVFP4_OXIDE_GEMM` or `[kernels] nvfp4_gemm` on; fv-gpucheck's nvfp4 groups and its nvrtc report (checked by the release pod and the gpucheck image workflow) need them, so every fv-gpucheck build keeps compiling and embedding them and no CI matrix changes; serve without the cubins behaves as any build without them did (the in-tree W4A4 kernel)
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0015
- Verification: build pod: workspace clippy (features off), `-p fastvideo-cudarc --features cuda`, `cuda,nvfp4-gemm`, `cuda,oxide`, `-p fastvideo-gpucheck --features cuda`

### FVID · 2026-10-24 · FVID-2026-10-24-b3b-load-snapshot
- Trigger: plan B3, second half: a `Settings` snapshot passed to the H3 / LTX-2 / Wan `load` instead of process globals, byte-identical, without restructuring the loaders (B6's)
- Options: thread a snapshot parameter through every helper a load calls (the shared runtime included, B5's attention files among them); a snapshot parameter for the load bodies and their load-only helpers plus a thread-local scope that serves the snapshot to the unchanged callees; a scope alone
- Decision: **`settings::Snapshot` (both readers' sources for every row, captured at load start) passed to new `load_with_settings` entries (the old entries capture one); the load bodies and load-only helpers read it as a parameter through the `settings::Source` trait (`from_settings` twins of `QuantMode`, `DitOffload`, `LtxOffload`, `ModelOptOptions`, `MoeResidency`, `nvfp4`, `H3Techniques`, `WanVaeChoice`); `Snapshot::scope` serves it to the four readers on the load thread for the callees; process-cached getters stay as they are; the one in-load `set_var` (MLX `FASTVIDEO_H3_AFFINE`) goes through `settings::set_env`, which updates the scope**
- Reason: identical values by construction (the accessors are the readers' definitions over the captured pair), identical read results because nothing changes the environment, profile or model scope during a load, worker threads unchanged (the scope is per thread like the model scope), and no signature change below the load bodies; B6 makes the remaining callees explicit
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0010-settings-layer.md (amendment B3b)
- Verification: `settings::snapshot::tests` (empty snapshot = today's defaults row for row, captured = live readers, non-UTF-8, scope freezes the capture, nested and cross-thread, `set_env`), per-family resolved-config identity (`h3::techniques`, `h3::pipeline`, `ltx2::pipeline`, `wan::pipeline` `load_settings_tests`), `rt::envflag` cached-getter order test; GPU `techniques` byte-identity family by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-b3b-model-spec
- Trigger: plan B3: "`model_spec` for Wan and the small families"; ADR-0010 noted that adding `wan` to `[pipeline].model` makes Wan profiles loadable
- Options: the minimal spec (what the runtime exposes and which techniques-layer settings and kernels it reads, no profile loadability); the full spec (Wan profiles loadable, with a Wan technique adapter and settings mapping)
- Decision: **the minimal spec: `ModelSpec` gains `profiles`, `families`, `extra_settings`, `kernels`; `registry::family_spec` covers H3, LTX-2, Wan and the 18 small families; `model_spec` (profile loadability) stays H3 and LTX-2; a workspace scan requires every layered read and kernel op in a family's sources and in the review's 11 shared runtime files to be declared by its spec**
- Reason: the full version needs a Wan profile adapter and a settings mapping (`taehv` and the precision recipes map to H3 / LTX names; Wan's TAEHV, step-cache and route settings are env-only rows, so promoting them is a reviewed behaviour change) and a Wan GPU cell, and the EU volume has no Wan tree beyond `--tiny`
- Reversibility: cheap (setting `profiles: true` with an adapter later)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0010-settings-layer.md (amendment B3b)
- Verification: `techniques::registry::tests` (a Wan profile is refused as before, small families compose nothing), `workspace_check::family_specs_declare_their_techniques_reads`, `fastvideo-core` `family::tests::every_family_has_a_model_spec`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-boot
- Trigger: plan C2 asks for one boot sequence (review §7.4) instead of the ten in bash, TypeScript strings and Rust (`fv-entry.sh`, fv-control's `*_BOOT` / `PROVIDER_BOOT` / `SLS_*`, the host scripts, Vast's onstart)
- Options: keep bash and share snippets; one shell script in the image; a Rust binary beside fv-serve driven by the env fv-control already sends
- Decision: `fv-boot` in `fastvideo-deploy` (feature `boot`), the `ENTRYPOINT` of every serve image (`serve`, `serve-<variant>`, `serve-cpu`); typed steps on four seams (processes, HTTP, clock, log); the host kind from existing env (`FV_PROVIDER`, `FV_SERVE_ROLE`, `RUNPOD_ENDPOINT_ID` / `FV_CONTROL_ENDPOINT`), no new switch; r2 stays on `r2-fetch.py` (streaming under `fv-boot fetch-guard`); `registry` typed and refused until `hydrate`; the upsampler launcher (`fv-upsampler.sh`) kept as the sidecar's supervisor; the report protocols and log markers byte-compatible, pinned by a fixture both the Rust test and a control vitest read; the driver check and `--print-config` preflight on every path; `deploy/runpod/fv-entry.sh` deleted
- Reason: CPU-testable end to end (fake provider, real fetcher against a fake chunk source); one place for every provider's boot; the env contract is already what fv-control sends
- Reversibility: cheap (the shell boots remain until C2b; `FV_BOOT_LEGACY=1` selects them)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0014-fv-boot.md
- Verification: `cargo test --locked -p fastvideo-deploy --features boot`, `control/test/unit/fv-boot-protocol.test.ts`; GPU: `scripts/gpu/fv-boot-smoke-pod.sh` on the EU volume (pending, coordinator)

### FVID · 2026-10-24 · FVID-2026-10-24-fv-boot-staging
- Trigger: images already deployed have no fv-boot, and fv-control is deployed by hand separately from the images (plan A3 caveat), so fv-control cannot require fv-boot yet
- Options: wait with fv-control until every image is promoted; payloads that require fv-boot; payloads that prefer fv-boot when the image has it
- Decision: every shell boot fv-control sends starts with `PREFER_FV_BOOT_SH` (`exec fv-boot` when it is executable and `FV_BOOT_LEGACY` is not 1, else the shell body as before); Vast's onstart prefers fv-boot the same way; the shell bodies are deleted and fv-control sends env only in a follow-up (C2b) once no pool, endpoint, channel or pinned spec can start a pre-fv-boot image; deploy order: merge, GPU gate, promote, fv-control at any time, C2b
- Reason: all four combinations of old / new fv-control and old / new image keep working; one env switch rolls back without a deploy
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0014-fv-boot.md §6
- Verification: `fv-boot-protocol.test.ts` (the preamble in every boot; fv-boot runs when present, the shell boot otherwise or with `FV_BOOT_LEGACY=1`); the existing bash boot tests unchanged

### FVID · 2026-10-24 · FVID-2026-10-24-deploy-tests-in-ci
- Trigger: no workflow ran `fastvideo-deploy`'s tests (fv-boot's fake-provider tests, the Runpod queue simulator)
- Options: a new workflow; the adapter-tests set
- Decision: the adapter-tests set (`scripts/serve/coverage.sh`, `crates/fastvideo-deploy/**` in its paths); control.yml also runs on `crates/fastvideo-deploy/fixtures/**` (its vitest reads the fixture)
- Reason: as for `fastvideo-manifest` and `fastvideo-lab`
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: none
- Verification: `bash scripts/serve/coverage.sh --tests` on the build pod

### FVID · 2026-10-24 · FVID-2026-10-24-fv-boot-review
- Trigger: the independent review of #159: serve-image.yml on main replaces serve-build / cpu-build with a tools release that has no fv-boot; the streaming guard and the watchdog used `kill` / `pkill` binaries the images lack; a refused config hid fv-serve's own line from the boot diagnosis; the rollback switch did not reach the image entrypoint; GCP / CloudRift workers and role endpoints were detected as cluster workers
- Options: (image build) compile fv-boot inside serve-build / cpu-build and fall back to compiling them whole with an old release; a fv-boot-only `boot-build` stage that a release carrying fv-boot replaces through prebuilt.sh; an optional COPY (an image without its entrypoint)
- Decision: a `boot-build` stage (`FROM builder`, fv-boot only) that every serve stage copies from, replaced by `boot-build=<serve set>` when the tools release has `out/fv-boot` (release-artifacts-pod.sh, tools-deps.sh and the tools-release.sh gates ship and check it from now on); signals by pid through `sh -c 'kill -s SIG "$1"'`; fv-serve's stderr logged verbatim; `FV_BOOT_LEGACY=1` honoured by fv-boot with fv-entry.sh's steps only; kind precedence provider > serverless (`RUNPOD_ENDPOINT_ID` / `FV_CONTROL_ENDPOINT`) > cluster worker (`FV_SERVE_ROLE=worker` and `RUNPOD_POD_ID`) > image; a controller worker without a config fails; pod reports end on any HTTP answer; `FV_BACKSTOP_API_KEY` only in the watchdog's env; recommended rollout: fv-control deployed with `FV_BOOT_LEGACY=1`, lifted per kind after a per-kind check
- Reason: serve-image.yml stays green with today's newest release (only one small crate compiles) and with every later one (nothing compiles); the other fixes restore the shell boots' behaviour where fv-boot had drifted
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0014-fv-boot.md (§2, §3.2, §4, §5, §6, §8)
- Verification: `cargo test --locked -p fastvideo-deploy --features boot` (a real process stopped through the shell builtin, the legacy path, the kinds, the reports), `scripts/ci/test_tools_release.sh` (the `prebuilt.sh contexts` cases), `prebuilt.sh fetch` against tools-v0.1.37, `fv-boot-protocol.test.ts` (a refused config is a crash loop)

### FVID · 2026-10-24 · FVID-2026-10-24-fv-deploy
- Trigger: plan C3 asks for one `Provider` trait promoted from `fastvideo-autoscale` (review §7.4) and the money guards written once; C3a is Runpod plus everything shared
- Options: the trait in `fastvideo-autoscale` with `fastvideo-deploy` depending on it; the trait in `fastvideo-deploy` with the autoscaler depending on it; a new crate; async or blocking
- Decision: the trait in `fastvideo-deploy::provider` (offers / payload / create / status / logs / destroy / account / reach), blocking on fv-boot's `Http` / `Clock` seams; `fastvideo-autoscale` depends on `fastvideo-deploy` (shared Runpod constants and `parse_runpod_time`), its pool-steering `Provider` stays until its pod / endpoint providers move onto the new trait; `runpod::{Pod, Serverless}` implemented; the binary `fv-deploy` behind feature `deploy` (clap, already in the lockfile)
- Reason: the instance lifecycle is the lower layer (fv-deploy, fv-lab, fv-control's schema and the autoscaler consume it); every caller is sequential, and blocking seams run the whole lifecycle in tests in milliseconds
- Reversibility: cheap (no format published; the autoscaler's API kept by re-export)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §2
- Verification: `cargo test --locked -p fastvideo-deploy --all-features`, `cargo test --locked -p fastvideo-autoscale` (and `--all-features`), workspace clippy `-D warnings`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-deploy-launch
- Trigger: C3 needs a provider-neutral `Launch` / `Instance` schema that fv-control can generate types from (C3c), built from the deployment data of C1
- Options: provider payloads straight from the presets; a `Launch` type with a schema, turned into payloads by each provider
- Decision: `Launch` (schema 1), `Instance`, `InstanceState`, `Offer`, `Account` (serde + schemars, `deny_unknown_fields`) with JSON Schemas in `crates/fastvideo-deploy/schemas/*.v1.json` under a drift test; a launch = a `deploy/deploy.toml` preset (image variant, config baked as `--config /etc/fv/…` or inline `FV_WORKER_TOML_B64`, disk, variants and their trees) plus a provider section (region `eu` → `EU-RO-1`, volume `bz0ofkfx45`, RTX PRO 6000; channel; guards); fv-boot as the entrypoint; volume weights only, a preset whose trees are not all on the EU volume is refused; never a secret value in a launch (Runpod secret references, the run key's hash)
- Reason: one schema for the CLI and fv-control; the presets stay the single source (ADR-0011)
- Reversibility: cheap until C3c generates TS from the schema; then a versioned schema change
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §3
- Verification: `tests/schema.rs`, `tests/payload_goldens.rs` (`every_preset_builds_or_is_refused_for_its_trees`, `wrong_family_unknown_preset_and_us_are_refused`)

### FVID · 2026-10-24 · FVID-2026-10-24-fv-deploy-guards
- Trigger: the price check, balance floor, wall-clock backstop, readiness wait and EU-only rule were repeated in three scripts, fv-control and the autoscaler, with different numbers
- Options: keep them per tool; one implementation in the lifecycle over the trait
- Decision: written once (`guard.rs`, `lifecycle.rs`): region guard (Runpod GPU instances only on `bz0ofkfx45` in `EU-RO-1`; the US volume and region refused), balance floor before every create (default and minimum $8), price cap on the quote before the create and on the billed price after (default $2.50/hr, a guard choice), the env guard (the provider key never in a launch), the detached backstop (this binary as `fv-deploy backstop` in its own process group, SIGHUP ignored, key only in its env, early exit when the instance is gone), readiness `/ping` with replace after 600 s and at most 3 cycles (the owner's rule), delete on failure / smoke failure / SIGINT unless `--keep`; a local TSV ledger (D1 is C3d); the in-pod watchdog not used (it would put the account key in the pod's env: owner decision)
- Reason: one tested set of rules for every provider; the backstop survives the CLI as the scripts' `nohup … curl -X DELETE` did
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §5
- Verification: `lifecycle/tests.rs` (provider-neutral fake), `provider/tests.rs` (fake Runpod), `tests/fv_deploy_bin.rs` (the real detached backstop deletes on a local fake Runpod; `up --yes` below the floor creates nothing)

### FVID · 2026-10-24 · FVID-2026-10-24-fv-deploy-goldens
- Trigger: C3a must prove its Runpod payloads equal what the scripts send before any script is retired
- Options: hand-written expected JSON; the scripts' own `plan` output as the fixture
- Decision: `scripts/serve/tests/fv-deploy-goldens.sh` runs `runpod-pod.sh plan` and `runpod-endpoint.sh plan` offline (a fake `curl` answers the volume lookup only) into `crates/fastvideo-deploy/fixtures/deploy/*.json`; `tests/payload_goldens.rs` asserts the Rust bodies equal them, and that a preset pod differs only in name, fv-boot entrypoint, 40 GB disk, `FV_WEIGHTS_SOURCE` and `FV_RELEASE_CHANNEL`; `control.yml` runs the script with `--check`; no script is deleted until the coordinator's live gate (ADR-0018 §6 lists which and when)
- Reason: the fixture is the script's behaviour, not a copy of it
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §4, §6
- Verification: `bash scripts/serve/tests/fv-deploy-goldens.sh --check`, `cargo test --locked -p fastvideo-deploy --test payload_goldens`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-deploy-review
- Trigger: the independent review of PR #169 found leak and guard-bypass paths before the live run (a billed-over-cap pod deleted before its backstop existed; a lost create answer treated as "no instance"; a failed arm or replace-delete followed by more creates; `down <endpoint>` deleting nothing; a string `costPerHr`; the floor not projected over the cap; serverless prices unchecked; the backstop trusting a spawn and a single "gone" reading; interrupts not reaching the smoke; no upper bounds; CI not building `deploy`)
- Options: fix each in the lifecycle and providers with a test; defer to C3b
- Decision: backstop armed (with an `armed` file handshake, 15 s) right after every create, before the billed check; deletes retried 3 × 5 s, the ledger says deleted only on success, a failed delete stops `up`; a create error other than 4xx / no capacity stops the fallback and sweeps by the run's unique name, and every exit of `up` sweeps it (but a `--keep` instance); a missing / unreadable billed price fails the cap (numbers and numeric strings accepted); the floor holds over the cap (`balance − (quote + spend) × cap_s/3600 ≥ floor`); serverless `workers_max > 1` refused (no serverless quote: a documented gap); the backstop stands down only on two 404 readings ≥ 60 s apart; `down` needs `--kind` for an unrecorded bare id and infers serverless from `/`; interrupts polled in the smoke and queue calls, a second signal exits; `cap_s ≤ 4 h`, `ready_cycles ≤ 3`, `ready_timeout_s ≤ 600` unless `--force-guards`, every create counts against the cycles; CI builds and tests `--features deploy` (check.sh, coverage.sh) and the goldens' path filter covers what the scripts source
- Reason: no path may leave a billed instance without a backstop or create more after a delete failed; the floor rule is "would drop", not "is"
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §2, §5
- Verification: the tests named in ADR-0018 "Consequences"; `cargo test --locked -p fastvideo-deploy --all-features`

### FVID · 2026-10-24 · FVID-2026-10-24-loader-stages
- Trigger: plan B6: split `H3Pipeline::load` (one ~880-line function) and `Ltx2Pipeline::load` into plan, open, place, fuse; "engine lifecycle testable on CPU"
- Options: extract helper functions inside the existing bodies; four typed stages with the device read once into a facts value the plan decides on; four stages with the plan reading the device itself
- Decision: **four typed stages per family (`H3LoadPlan` / `Ltx2LoadPlan::new` → `open` → `place` → `fuse`), `load_with_settings` their composition; the plan is pure over the options, the settings snapshot, configs and manifests and a `DeviceFacts` value (`free`, capability, device-expected, bf16 activations, linear route) measured once before it; pure forms of the runtime's device-dependent precision rules (`QuantMode::resolve_on`, `mxfp8_wanted_on`, `planned_linears_on`, `require_tensor_cores_on`) with the live functions defined as them; an adapter merge stays applied per linear in place (merge before quantization, no second copy) with the plan deciding it and fuse holding the passes over the finished model (TeaCache, the merge report); LTX-2's fuse records strength 0 (its LoRAs fuse per request)**
- Reason: the plan becomes a value that can be pinned in goldens for served configurations on given cards without a GPU or weights; the stage order is the statement order, so tensors, precisions, placements, `load/io` lines and ledger bookings are unchanged; a post-placement merge would change numerics or double the DiT's memory
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0017
- Verification: build pod: plan goldens (`tests/goldens/{h3,ltx2}_load_plan.txt`, 17 H3 and 7 LTX-2 configurations), H3 open on a tiny snapshot, the LTX-2 plan → open → place → fuse on the tiny fixtures, `QuantMode::resolve_on` against the live reader; GPU `techniques` byte-identity family by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-b6-retire-dead-code
- Trigger: plan B6 / review §7.3 item 6: retire `TensorBackend` and the mlx scaffold, only what is unreachable
- Options: delete; move out of the workspace (as oxide); keep
- Decision: **delete `fastvideo-ops::TensorBackend` with its host reference (`HostBackend`, `HostTensor`, `Device`, `DType`, `OpsError`), `fastvideo-cudarc::CudarcBackend`, the unused `fastvideo-ops` dependency of `fastvideo-core` and `fastvideo-gpucheck`, and the `fastvideo-mlx` crate (its `mlx-rs` lock entries go with it); keep `fastvideo-ops::fp8` and the CUDA runtime's reader of FastVideo's MLX H3 export (`h3/mlx.rs`, the MLX key aliases, `rt::affine`)**
- Reason: nothing called the trait but its own tests, and nothing depended on the scaffold (host stubs, no graph); the MLX H3 export reader is a live load path, chosen by the checkpoint (any H3 tree holding `mlx_h3_dit.json` loads through it, from the CLI or a serve config), even though no served variant or EU-volume tree ships one today; the scaffold stays readable in git history
- Reversibility: cheap (git history)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: 0017 (and 0001's status)
- Verification: workspace `cargo check` / clippy `-D warnings` with `--locked` on the regenerated lockfile, workspace lib tests

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab
- Trigger: plan D1 asks for `fv-lab` (review §7.5); D1a is `run` + `pod` for the `rtx6000` family, proven equal to the scripts
- Options: a binary beside `fv-gpucheck` (CUDA crate); a binary of `fastvideo-lab`; a new crate
- Decision: `fv-lab` is a binary of `fastvideo-lab` behind feature `cli` (clap, tokio signals, `fastvideo-deploy/boot`; all in Cargo.lock), CPU only, calling `fv-gpucheck` and `verify-weights.sh` as processes; `fastvideo-lab` depends on `fastvideo-deploy`'s library half
- Reason: it runs on the GPU box and on hosts without CUDA (`pod`, `--dry-run`, `check`); the gold files, run plan and reports share one crate
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §2
- Verification: `cargo test --locked -p fastvideo-lab --all-features`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-plan
- Trigger: D1 needs the run plan as data (`lab/tiers.toml`, `lab/cells/*.toml`) instead of `runpod-matrix.sh` `case` branches
- Options: per-pipeline command builders from a structured workload; argv templates with placeholders over a structured workload
- Decision: `[[tier]]` (GPU, count 1, cloud, region `eu`, image flavour `runtime`, `max_usd_per_hr`, `wall_cap_s` ≤ 4 h, `cell_timeout_s`, disk, weights `volume`, `preflight`, `tae`, `cells`) and cell files (`[workload]` whose every value an arm must read as a placeholder, `[lists]` spliced by `"{@name}"`, `[[arm]]` name / gate / trees / steps / env / argv, `[[compare]]`, `[[gate]]`), `deny_unknown_fields`, schemas `tiers.v1.json` / `cell.v1.json` with a drift test; `rtx6000` ported value for value from runpod-http.sh's defaults and the script's branch (two cells, seven arms)
- Reason: argv templates keep each family's argument order (byte equality with the script) while the workload values have one source
- Reversibility: cheap until D3 reads the cells; then a versioned schema
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §3
- Verification: `tests/schema.rs`, `plan::tests`, `tests/lab_files.rs`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-weights
- Trigger: a cell must name its weights so the volume works today and the registry's `hydrate` later
- Options: paths in commands only; tree names per arm plus the tier's weights source
- Decision: each arm lists its trees by weights-manifest.tsv `dest` and its `verify-weights.sh` gate; tests hold `trees` equal to the gate's roots (`verify-weights.sh --list`) and to every `{weights}/<tree>` the command reads; the tier's `weights = "volume"` maps `{weights}` to `/workspace/weights` on `bz0ofkfx45` (the only value accepted in D1a); `registry` will stage the same trees with `hydrate`; an arm whose tree is not on the volume is gated and recorded as skipped, as the script does (`fasth3-8step`: `h3-8step` is R2-only)
- Reason: the cells do not change when the weights source does
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §4
- Verification: `tests/dry_run.rs` (`arms_gate_on_the_scripts_cells_and_name_their_trees`), `tests/lab_files.rs`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-equality
- Trigger: D1's exit is "`fv-lab run rtx6000` reproduces today's family"; CPU evidence is needed before the GPU run
- Options: hand-written expected commands; the script's own dry run as the fixture
- Decision: `runpod-matrix.sh` gets B6's `FV_DRY_RUN=1` hunks verbatim (the usage line too, so a merge with #170 resolves to #170's file; checked with `git merge-file`); `scripts/gpu/tests/fv-lab-goldens.sh` writes the script's dry run, its gate pairs and `runpod-http.sh plan`'s pod body (new `plan` mode, `pod_payload` factored out of `create_pod`) into `crates/fastvideo-lab/fixtures/`; `fv-lab run --dry-run` prints the same lines (bash 5.2 `printf %q` in `quote.rs`); `--check` in `control.yml`; the runner's files and logs are tested on a fake fv-gpucheck / verify-weights / nvidia-smi / timeout
- Reason: the fixture is the script's behaviour, not a copy of it
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §5
- Verification: `bash scripts/gpu/tests/fv-lab-goldens.sh --check`, `tests/dry_run.rs`, `tests/fv_lab_bin.rs`, `tests/runner.rs`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-pod
- Trigger: D1's `pod` must run on C3's `Provider` and guards, not copy them
- Options: copy runpod-http.sh's create/wait/fetch/delete into Rust; reuse `fastvideo_deploy::lifecycle::Deployer::up` with `keep` and no smoke
- Decision: reuse: fv-deploy's region guard, $8 floor over the cap, price cap on quote and bill, detached backstop (`fv-lab backstop`, same handshake) and 600 s / 3-cycle readiness; `LabPod` wraps `runpod::Pod` and overrides only `reach` (ready = `GET /<tier>/` 200, the run directory's server, as `wait_up`); the start command is runpod-http.sh's (template generated from the script) with `bash runpod-matrix.sh` → `fv-lab run`; payload golden equal but that line and `env: {}` (image digest-pinned at `up`); `pod fetch` ports `wait_done` / `fetch_results` into `artifacts/runpod/<tier>/<tag>/`; `pod reap` deletes `fv-lab-*` pods past their tier's cap + 10 min, which needs one seam in fastvideo-deploy, `runpod::Pod::list()`; the tier's price cap is fv-deploy's $2.50/hr, not the script's $5 (open decision)
- Reason: one tested set of money guards for every launcher; the pod body stays the script's
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §6, §7
- Verification: `tests/payload_golden.rs`, `pod::tests`, `runpod::pod::tests::list_reads_names_states_ages_and_prices`; GPU run by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-review
- Trigger: the independent review of PR #171: the GPU byte-identity evidence could not be produced (no fetch takes frames, the pod is deleted); `pod reap` could delete pods this session did not create; two `pod up` in one UTC minute shared a pod name (fv-deploy's exit sweep deletes by exact name); the runtime image build went from about 2 to about 10 minutes on main; a failing nvidia-smi was not fatal; a fetch without `matrix.out` / `DONE` counted as success; early returns skipped the pod delete; CI path filters; the goldens read the caller's environment
- Options: fix each with a test; defer to D1b
- Decision: both runners write `<arm>/frames.sha256.json` (every frame's SHA-256 by path; `frames_digest` in runpod-matrix.sh's `run_cell`, byte-identical in fv-lab, fixture from the script's own function) and both fetches take it; the gate is `scripts/gpu/fv-lab-compare-runs.sh` between an `fv-lab pod up` run and a `runpod-http.sh run` of one commit; `reap` takes only this machine's recorded pods, `--account` only full `fv-lab-<known tier>-<sha7>-<MMDDHHMMSS>-<hex4>` names; the run tag is `<sha>-<MMDDHHMMSS>-<4 random hex>` and `up` refuses a live name; fv-lab rides in the tools release's `gpucheck` set (`out/fv-lab`, `-V` gated, `lab-build=<set>` context) like fv-boot, the stage compiling only as a fallback; nvidia-smi failures fatal; success needs the fetched `DONE` and `matrix_exit=0`; the pod is deleted before any error returns; adapter-tests watches the scripts lab_files.rs reads; the goldens run under `env -i`
- Reason: the GPU gate must be checkable after the pods are gone; only touch pods you created; one name per run; images on main must not compile fv-lab once a release carries it
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §5, §6, §7, §8, open decision 2
- Verification: `tests/frames_digest.rs`, `tests/runner.rs` (`a_failing_nvidia_smi_is_fatal`, the digest of an arm), `pod::tests` (tags, `name_taken`, `reap_recorded`, `reap_account`, `judge_run`, the fetch), the bin's `after_delete` test, `scripts/ci/test_tools_release.sh` (fv-lab version gate, lab-build contexts), `fv-lab-goldens.sh --check` (compare-runs self-test)

### FVID · 2026-10-24 · FVID-2026-10-24-step-cache-home
- Trigger: plan B4a; nine step-cache controllers in five families (review §3, §5.2)
- Options: a `StepCache` module in fastvideo-cudarc (`rt/`); `fastvideo_models::techniques::step_cache`
- Decision: `fastvideo_models::techniques::step_cache`: the trait and one `EasyCache` / `TeaCache` / `FbCache` / `Taylor`, scalars only; signals, residuals, factors and the device reductions stay in each family's cudarc runtime
- Reason: every copy already lived in fastvideo-models and holds only scalars, so it tests without CUDA; the techniques layer owns the `step_output` seam and the settings; cudarc depends on models
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0020-step-cache.md
- Verification: `cargo test -p fastvideo-models step_cache`, cudarc/models/engine-service/core tests

### FVID · 2026-10-24 · FVID-2026-10-24-step-cache-parameters
- Trigger: the per-family copies differ in arithmetic (accumulator reset point, `>=` vs `<`, f32 accumulator, Horner form, forced steps, zero-guarded caps, EasyCache estimators)
- Options: normalise to one rule; keep each behaviour as a named parameter
- Decision: named parameters (`ResetOn`, `Accumulate`, `Rescale`, `Comparison`, `Cooldown`, `Cap`, `Estimator`, `EasyLabels`), each family a parameter set; two uniform writings only where provably the same numbers (Wan EasyCache's `initialize` clear; Cosmos counting a reuse in `note_reused`)
- Reason: byte identity first (plan §1); a normalisation changes clips
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0020-step-cache.md "The arithmetic each implementation preserves"
- Verification: `step_cache/equality.rs` against the pre-B4a controllers in `step_cache/legacy.rs` (verbatim; TeaCache4Wan2.1 a hand reduction of its decision): decisions and latents bit for bit, one exact threshold tie per comparison family, the TeaCache4Wan2.1 f32 accumulator bits and an f32-vs-f64 split; mutations caught: H3 `ResetOn`, A14B `Estimator`, f64 accumulation, `>=` → `>`, the `!(acc < thr)` tie flip

### FVID · 2026-10-24 · FVID-2026-10-24-step-cache-gpu-cells
- Trigger: B4a's exit criterion (byte identity per family) needs GPU cells; only H3 and LTX trees are on the EU volume
- Options: cells for every family (R2 trees, large downloads); EU-volume families only, CPU fixtures for the rest
- Decision: `runpod-matrix.sh b4a`, three H3 TeaCache cells (fullopt from the flag, fullopt from the profile, a 4-step VSA reuse fixture), base vs PR with `--off-identity` and each arm's `frames.sha256.json`; a cell fails when either arm reused no forward or the arms' cache evidence lines differ; `FV_B4A_CELLS` narrows. No LTX-2.5 FBCache cell: `fbcache::scope` admits only dev guided stage 1 and no LTX-2.5 generate route reaches it (guided A2V runs with `use_cfg = false`, other guided requests fail `check_generate_contract`), and `scope` is not widened, so FBCache rests on its CPU fixture like Wan / SANA / LingBot / Cosmos / Cosmos3 (no EU-volume tree)
- Reason: the owner's EU-volume-only GPU rule (plan §1); new weight downloads need approval; widening `scope` would change outputs
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0020-step-cache.md "Evidence"
- Verification: `FV_DRY_RUN=1 bash scripts/gpu/runpod-matrix.sh b4a`; GPU run by the coordinator; `h3::techniques::tests::b4a_gpu_cell_profile_reuses_the_middle_forwards`; review of PR #173 (the FBCache path)

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-d1b-families
- Trigger: plan D1b moves more `runpod-matrix.sh` families into fv-lab so that their `case` branches can be deleted one at a time; the byte-identity families come first
- Options: every family at once with a richer schema (steps, probes, hashing); only the families whose branches are arms, compares and gates, with minimal typed additions, and the rest left in the script with a reason
- Decision: move `techniques`, `b6`, `rtx5090`, `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae` and `precision` (its default mode) as tiers and cells (10 tiers, 38 cells, 113 arms with `rtx6000`), each on the `rtx6000` tier's pod, as `runpod-http.sh` gives every family; every other family stays, its reason and retire-when recorded in ADR-0019 §9.1; no branch is deleted until that family's GPU run through both runners matches (§9.5)
- Reason: the moved families are lists of arms, compares and gates, so they fit the schema; the others interleave steps that the schema should not grow for one family (Lanczos upscales, verdict files, data-dependent branches, probes, venvs)
- Reversibility: cheap (the script keeps every branch until its family's GPU run)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9, §9.1
- Verification: `tests/dry_run.rs` (every tier against its script fixture), `bash scripts/gpu/tests/fv-lab-goldens.sh --check`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-baseline-arm
- Trigger: `techniques` and `b6` (and `b4a`) run every cell with a baseline commit's fv-gpucheck and with this build, then `compare-clips --off-identity`
- Options: a second tier binary setting; a per-arm `bin` naming the baseline binary; the baseline path as a placeholder in the argv
- Decision: `[[arm]] bin = "baseline"` (`ArmBin`, default `gpucheck`) runs `<scratch>/baseline-<sha>/fv-gpucheck`, fetched by `fetch-baseline.sh` into the script's path once per run when a cell has a baseline arm; the commit comes from `--baseline-sha` / `FV_BASELINE_SHA`; a failed fetch is FATAL (exit 2) as in the script; `pod up --baseline-sha` passes it to the pod (`FV_EXTRA_ENV`), and `pod up` refuses a baseline tier without it
- Reason: the pair differs only in `bin`, so the off-identity compare stays a check of the two binaries; the fetch and the path stay the script's
- Reversibility: cheap (additive optional field)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.2 (1)
- Verification: `plan::tests::baseline_arms_bare_env_tae_and_paired_gates`, `tests/dry_run.rs::baseline_arms_run_the_fetched_binary`, `tests/runner.rs` (baseline pair, failed fetch), `tests/fv_lab_bin.rs::a_baseline_tier_needs_the_baseline_commit`

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-tae-and-bare-env
- Trigger: `rtx5090`, `fastvideo` and `h3arms` have TAE-decoded twins (the script's `tae_gated_cell`), and `techniques` / `b6` write a bare `env` before the binary
- Options: TAE paths hard-coded to `/fvscratch/tae`; a `{tae}` placeholder resolved like the script's `$TAE` with a per-arm file gate; for `env`, accept one word of difference or make the prefix explicit
- Decision: `[[arm]] tae = "<file>"` and `{tae}`: `FV_TAE_DIR`, else the volume's `auxiliary/tae` when `.complete`, else `<scratch>/tae`; one `fetch-tae.sh` per run for a missing file, still missing → skipped (`tae weights missing`), then the weight gate; such an arm must read `{tae}/<file>`, and `{tae}` may not appear in an arm without `tae`. `env` becomes optional: absent means no prefix, `env = []` renders a bare `env`
- Reason: the command lines stay equal word for word with the script, and the TAE location follows the volume as the script's does
- Reversibility: cheap (additive)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.2 (2), (3)
- Verification: `tests/runner.rs::baseline_arms_and_tae_arms`, `tests/lab_files.rs::every_tae_file_is_one_fetch_tae_sh_fetches`, the `rtx5090` / `fastvideo` / `h3arms` dry-run goldens

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-gate-placement
- Trigger: ADR-0019 open decision 3; `h3arms` and `precision` loop `compare; gate` per pair, `fastvideo` loops `compare; compare; gate; gate`, and `rtx5090` compares after each resolution
- Options: a general ordered step list in a cell; one cell per loop iteration plus a compare's optional paired gate
- Decision: one cell per loop iteration or helper call (`rtx5090`, `fastvideo`, `ltxvae`, `techniques`, `b6`), and `[[compare]] gate = { kind, off }` runs that pair's gate right after its compare; `[[gate]]` keeps running after all compares
- Reason: both script orders are expressible with one small typed field, and the dry run and `live.log` keep the script's order
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.2 (4), open decision 3
- Verification: `plan::tests::baseline_arms_bare_env_tae_and_paired_gates`, the `h3arms` / `precision` / `fastvideo` dry-run goldens

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-goldens-per-tier
- Trigger: D1a's goldens covered `rtx6000` only, and took the gate pairs from an awk over the branch, which cannot expand loops
- Options: per-family hand-written fixtures; the script's own dry run for every tier, with the gated cells reported by the script
- Decision: `fv-lab-goldens.sh` reads the tier names from `lab/tiers.toml` and writes `<tier>-dry-run.txt`, `<tier>-gates.tsv` and `runpod-http-<tier>.json` for each (`FV_BASELINE_SHA=7654321`, the repository path written as `/opt/fastvideo-rs`); `runpod-matrix.sh` gains dry-run-only behaviour: `FV_DRY_RUN_GATES=<file>`, no TAE fetch and no `techniques` baseline fetch when dry; `fv-lab-compare-runs.sh` also requires equal compare-report verdicts (status, off-identity) and gate reports in both runs (a gate verdict weighs timings, so only its presence)
- Reason: the fixture is the script's behaviour for every family that moves, and a new tier without fixtures fails CI
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.3, §9.5
- Verification: `bash scripts/gpu/tests/fv-lab-goldens.sh --check` (rtx6000 fixtures unchanged; compare-runs self-test with reports), `tests/{dry_run,payload_golden}.rs` over every tier

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-b4a-stays
- Trigger: `b4a` is a byte-identity family like `b6`, but each pair also computes a verdict from the arms' TeaCache evidence (reuse count > 0 per arm, equal `cache-evidence.log`, `b4a-<cell>.verdict.json`)
- Options: move the arms and compares and drop the verdict; add a typed evidence/verdict step now; leave the family in the script
- Decision: `b4a` stays in `runpod-matrix.sh`; the evidence step is designed with D1d (`gate` moved into fv-lab)
- Reason: without the verdict, fv-lab would pass a run the script fails; a verdict step is more than a minimal schema addition, and B4a's GPU gate has already passed (#173)
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.1
- Verification: n/a (no change)

### FVID · 2026-10-24 · FVID-2026-10-24-techniques-per-arm-caches
- Trigger: review of #176. The `techniques` H3 arms shared `--text-cache $SCRATCH/h3-text-cache` (no code version in the key) and `--adaln-cache $RUNS/<cell>-adaln.cache` (skips the projections when the file exists), so the candidate build's `-env` / `-prof` read what the baseline binary wrote
- Options: `--no-text-cache` and no AdaLN cache (as `b6`); a text cache and an AdaLN cache per arm
- Decision: per arm (`h3-text-cache-<arm>`, `<arm>-adaln.cache`), in `runpod-matrix.sh` and the `techniques-*` cells in lockstep. Each arm's warm-up computes both with its own binary, and its timed pass reads them; the goldens are regenerated from the changed script. Before this, `techniques` gates did not cover the candidate's H3 text-encoder or AdaLN paths (noted in docs/techniques.md and ADR-0019 §9.4)
- Reason: keeps the family's intent (one warm process per arm, cache-on timings comparable with earlier runs) while the compared pass uses only the candidate's own encoder and AdaLN output
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.4
- Verification: the `techniques` dry-run golden (per-arm paths on both runners); the coordinator's `techniques` GPU run

### FVID · 2026-10-24 · FVID-2026-10-24-fv-lab-gates-fail-closed
- Trigger: review of #176. `fv-lab-compare-runs.sh` passed two all-skipped runs, and two unreadable reports compared equal. `pod up` accepted a baseline equal to the run's commit, and an unchecked `FV_BASELINE_SHA` in `--extra-env`. The goldens could not see a gate's kind / OFF arm or an arm's TAE gating
- Options: document the gaps; close them
- Decision: compare-runs fails unless at least one arm ran (exit 0) with byte-identical frames in both runs, treats anything unreadable as a difference, and prints the ran / skipped / failed counts. `pod::baseline_for_pod` refuses a baseline equal to `--sha`, a malformed or empty `FV_BASELINE_SHA` in `--extra-env`, and the commit given twice; the runner stops when the baseline's build-id is the image's. `FV_DRY_RUN_GATES` records `arm … <TAE file or ->` and `gate … <kind> <off or ->` in run order, and the golden volume's `auxiliary/tae` is complete, as on `bz0ofkfx45`
- Reason: a gate must fail when it did not compare anything, and a hard-coded path or a changed gate kind must fail the goldens
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.2, §9.3, §9.5
- Verification: `fv-lab-goldens.sh --check` (negative cases: all-skipped, all-crashed, unreadable summaries and reports), `pod::tests::the_baseline_commit_of_a_pod_fails_closed`, `tests/fv_lab_bin.rs::a_baseline_tier_needs_the_baseline_commit`, `tests/runner.rs::a_baseline_that_is_this_build_is_fatal`, `tests/dry_run.rs::arms_and_gates_are_the_scripts_and_arms_name_their_trees`
### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-home
- Trigger: plan B4c; three disk prompt caches (H3 `h3/text_cache.rs`, LTX-2 `ltx2/text_cache.rs`, Wan's key in `wan/pipeline.rs` over H3's store; review §3, §7.3 item 4)
- Options: `fastvideo_models` beside B4a's `step_cache`; `fastvideo_cudarc::rt::prompt_cache`
- Decision: `fastvideo_cudarc::rt::prompt_cache` (key, file identities, format, atomic store, load, get-or-compute); the families keep only their identity inputs, default directories and value types
- Reason: keys and seals are SHA-256 and `sha2` is a cudarc dependency only (no new dependency); the identities read weight files and lazy checkpoints and the values are device tensors read back; every caller is a cudarc pipeline; `rt` is the shared runtime (B1) and builds without CUDA
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md
- Verification: `cargo test -p fastvideo-cudarc prompt_cache`, cudarc/models/engine-service/core tests

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-parameters
- Trigger: the copies differ in what is on disk: key domain and prompt trimming, file identities (framed head+tail MiB vs unframed 256 KiB head vs tensor fingerprint), two entry layouts, finite check, temporary names, fsync, cleanup, error text and log line
- Options: normalise to one key and one format (orphaning or re-versioning every existing cache); keep each difference as a named parameter
- Decision: named parameters: `KeyRecipe { domain, PromptNorm }` with the length and identity fields as arguments (one derivation for all three), `Format { magic, extension, Layout, TempName, Durability, TempCleanup, Finite, ErrorText, StoreLog }` with two instances, `H3` (H3 and Wan) and `LTX2`; keys, files and log lines unchanged
- Reason: byte identity first (plan §1); existing cache directories must keep working
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "Key", "On-disk format"
- Verification: `rt/prompt_cache/equality.rs` against the pre-B4c caches in `legacy.rs` (verbatim): keys (pinned H3 / LTX / Wan values recomputed in Python), identities, byte-equal files, cross reads, identical accept / reject of every truncation and bit flip, hit / miss sequences, failure leftovers, concurrent writers; 18 mutations of key and format all caught

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-compatibility
- Trigger: B4c brief: existing cache directories must be read exactly as before or ignored safely, never misread
- Options: version the unified format (ignore old entries); read old entries as before
- Decision: read as before, every family: same names, same format, same checks; nothing is versioned, so nothing is ignored by version; damaged or foreign files stay misses
- Reason: the unification changes no byte on disk, so a version bump would only throw away warm caches
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "Compatibility rule"
- Verification: `equality.rs` cross reads both ways and byte-equal files; GPU `b4c` cells require the base and PR cache directories to hold byte-identical entries

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-gaps-kept
- Trigger: B4c review of the keys (and the independent review of PR #177): six inputs the conditioning depends on are not in the keys: LTX FP8 Gemma `FASTVIDEO_LTX2_TEXT_FP8` (silent wrong output, medium); the H3 recovered-8B identity is a constant, so a changed `ara.safetensors` LoRA, conditioning adapter or base serves stale conditioning (silent wrong output, medium); Wan UMT5 GEMM compute type (low); `FASTVIDEO_BF16_ACT`, the global GEMM compute type and GPU arch for H3 / LTX (small numeric differences); LTX resident vs streamed Gemma (expected equal, unproven); FP8 tree vs quantise-at-load (fine while bit-equal). Also: no size cap; threads of one process share a temporary name
- Options: fix in B4c; keep and record as follow-ups
- Decision: **(coordinator) keep unchanged in B4c, recorded with file:line and severity in ADR-0021 "Follow-ups: key gaps carried over"; the coordinator fixes them in a follow-up PR (B4c2) right after B4c merges**
- Reason: each fix changes keys or behaviour, which B4c (byte identity) must not; each is its own reviewed decision
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "Follow-ups: key gaps carried over", "Eviction and size", "Concurrency"
- Verification: n/a (recorded gaps; B4c2)

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-gpu-cells
- Trigger: B4c's exit criterion (byte identity per family) needs GPU cells; only H3 and LTX trees are on the EU volume
- Options: one cell per family with a cache, base vs PR, a cold pass and a hit pass per arm; a shared cache across arms
- Decision: `runpod-matrix.sh b4c`: `b4c-h3-4stepvsa` and `b4c-ltx25-512p`, each arm (baseline binary, PR build; same argv / env) over its own new cache directory, pass `-miss` then `-hit` (each pass directory emptied first); `--off-identity` base vs PR per pass; the verdict (`scripts/gpu/b4-verdict.sh`) passes only on positive evidence: every pass `summary.json` exit 0, `benchmark.json` `miss` / `hit` as planned, on H3 `text_encoder` containing `fp8` on the misses and `cache` on the hits, both compares' `off_identity` checks passed and not skipped, both cache directories non-empty, byte-identical and free of `*.part` / `*.tmp*`; `b4c.verdict.json` fails when a named cell has no passing verdict (review of PR #177; `b4a` got the same exit / compare / family fold); `pr-miss` vs `pr-hit` compared for the record only; `FV_B4C_CELLS` narrows; Wan rests on the CPU fixtures (no EU-volume tree)
- Reason: separate caches per arm compare like with like (a shared cache would compare a hit with a miss, the techniques LTX lesson); byte-equal entries prove the PR reads the baseline's cache as its own
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "Evidence"
- Verification: `FV_DRY_RUN=1 FV_BASELINE_SHA=<sha> bash scripts/gpu/runpod-matrix.sh b4c`; GPU run by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-block-fusion
- Trigger: plan B4b: the DiT block fusion written three times (`wan/fuse.rs`, `h3/fused16.rs`, `ltx2/fuse.rs`); one module parameterized by rounding policy, byte-identical
- Options: one templated kernel per op; one Rust module whose typed rounding policy selects the existing kernels, with one host reference and one switch; stop after the inventory
- Decision: `rt::fuse`: `RoundingPolicy` / `QkPolicy` (one named constant per family op) selecting the kernel and its mode arguments, one host reference (`rt::fuse::host`) the Wan and H3 CPU twins run, one switch implementation; the kernels, the launchers' operand and table handling and the one-family ops (LTX head gates, H3 MXFP8 / NVFP4 producers, q/k/v/gate split) stay with their families
- Reason: the residual + modulated norm and the q/k norm + RoPE are one computation per op with different rounding points, but the kernels also differ in normalizer, reduction order and rsqrt, and their source is pinned (`KERNELS_CU_SHA256`): one templated kernel would move bits
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0022-block-fusion.md
- Verification: `rt/fuse/equality.rs` against the verbatim pre-B4b code (`legacy.rs`), bit for bit, and ten mutations each caught; GPU `b4b` cells

### FVID · 2026-10-24 · FVID-2026-10-24-block-fusion-plan-value
- Trigger: ADR-0017 "How B4 and a lazy cold start plug in": the fusion switch becomes a value the plan decides and the fuse stage installs, instead of the process-wide cached getters
- Options: thread the value through every fused call; install it on the DiT and run its forwards under it (a scoped value, as `with_quant_act` does for H3's activations); keep the getters
- Decision: `kernel_fusions` is an `rt::fuse::BlockFusion` (read from the snapshot); the H3 fuse stage installs it on the DiT and the text refiner, the LTX-2 fuse stage on the placed DiT and the pipeline on every DiT it loads later and around its text encodes (the connectors); `forward` / `forward_probed` / the refiner and the encodes run under it, restored by a drop guard; forced values (parity checks) still win; the cached process value remains for DiTs built outside a plan and for Wan (no loader stages)
- Reason: no fused call signature changes; the installed value equals the cached one in every process the tooling permits (settings set before a load, fv-serve's `ProcessPlan` requires one value per process), so outputs do not move
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0022-block-fusion.md "The plan's value installed on the DiT"
- Verification: `the_plans_read_what_the_old_plans_read`, `the_switches_resolve_as_the_old_getters`, `the_family_getters_are_the_switches`; the load-plan goldens unchanged

### FVID · 2026-10-24 · FVID-2026-10-24-block-fusion-gpu-cells
- Trigger: B4b's exit criterion (byte identity per family) needs GPU cells that also prove the fused kernels ran in both arms; no log line names them, and a new line would exist in the PR arm only
- Options: a new log line (absent from the baseline); the plan description (not logged at run time); the CUPTI step trace both builds have (`FASTVIDEO_GPU_TRACE=1`)
- Decision: `runpod-matrix.sh b4b`: `b4b-h3-4stepvsa`, `b4b-h3-4stepvsa-off`, `b4b-ltx25-512p`, `b4b-ltx25-512p-off` (and `b4b-h3-fullopt` when named), baseline binary against this build with the same argv / env, every arm traced; a new `FASTVIDEO_GPU_TRACE_TOP` (report only, default 25) lists every kernel of the step in the PR arm (the baseline predates it and lists 25), so after review: `b4b_verdict` passes only when both arms exited 0, `--off-identity` passed (not skipped), both traces have a kernel table, `dropped_records` 0 and equal `kernel_count`, the PR table is complete, every fused / bf16-chain kernel the base lists matches the PR call for call, rule `on` finds every named gated kernel in the PR arm (H3 `fvf_h3_gate_res_norm_mod`, `fvf_h3_qkvg_heads`; LTX `fvf_ltx_res_norm_mod`, `fvf_ltx_qk_norm_rope`, `fvf_ltx_gate_merge`), rule `off` finds none and finds the unfused path's kernels, and H3's call counts keep the block structure; `FV_B4B_CELLS` narrows; Wan rests on the CPU fixtures (no EU-volume tree)
- Reason: positive evidence from both arms with one mechanism; the fusion-off controls show the switch reaches the kernels the same way in both builds
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0022-block-fusion.md "Evidence"
- Verification: `bash scripts/gpu/tests/b4-verdict.test.sh`; `FV_DRY_RUN=1 FV_BASELINE_SHA=17e4a1d bash scripts/gpu/runpod-matrix.sh b4b`; GPU run by the coordinator
### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-fields
- Trigger: B4c2 brief: close the six key gaps of ADR-0021 so a hit never returns conditioning a fresh encode under the current settings would not
- Options: key the requested settings (environment, CLI); key the values the run resolved; full hashes of the recovered-8B files
- Decision: resolved values, one field each: `Numerics` (the device context's GEMM math and compute capability, the resolved bf16 activations, `host` without a device) read by `Numerics::live()` at the lookup, not by threading the device through signatures; `WeightsRead` (checkpoint, FP8 rows at load, FP8 tree) plus the tree's `files_head_tail`, from what the encoder read (`ResidentDecoder::prequantized`, the streamed tree, LTX `gemma_read`); LTX `TextMode` from `planned_mode`; recovered 8B: `files_head_tail` of base, LoRA and adapter (head and tail MiB, as LTX, not full hashes)
- Reason: a spurious miss costs one encode, a wrong hit is silent wrong output; the device context is the one the encode runs on, so reading it at the lookup is exact without a signature change
- Reversibility: cheap (another domain)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "B4c2"
- Verification: the families' key tests (pinned keys recomputed in Python), `numerics` tests, `equality.rs` `b4c2_keys_never_name_a_legacy_entry`, mutations; GPU `b4c2` cells (coordinator)

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-versioning
- Trigger: every key derivation changes; ADR-0021 "Versioning" said a changed key gets a new magic and a new domain, the B4c2 brief says a new domain and no magic change unless necessary
- Options: new magics and domains; new domains only
- Decision: new domains only (`fastvideo h3 text conditioning v2`, `WANTEXT2`, `fastvideo ltx2 text conditioning v2`); magics and layout unchanged; the pre-B4c2 recipes stay as `*_KEY_V1` for the tests; ADR "Versioning" amended (a changed layout: magic and domain; a changed key alone: domain)
- Reason: the layout did not change; every name moves with the domain, so an old entry is never looked up; only a SHA-256 collision could name one, which a magic bump would also turn into a miss (accepted, recorded in the ADR)
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "On-disk format" (Versioning), "B4c2"
- Verification: `b4c2_keys_never_name_a_legacy_entry`, the `the_new_domain_alone_keeps_old_entries_out` tests; every existing entry misses once

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-ltx-decisions
- Trigger: LTX's mode (resident / streamed) was decided at the miss, after the lookup, and the FP8 tree at every open of Gemma; a key field must be known before the lookup and match what the miss runs
- Options: decide at the lookup and pass the decision to the miss; key the requested residency
- Decision: `planned_mode` (the decision `ensure_backend` made, without loading) before the lookup, passed to `compute`; `gemma_read` decided once per `TextEncoder`, used by the key and by every open
- Reason: the key then names exactly what a miss runs; a hit queries free memory once more and an invalid `FASTVIDEO_LTX2_TEXT` now fails a hit as it failed a miss
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "B4c2"
- Verification: `ltx2::pipeline` cache tests; GPU `b4c2-ltx25-fp8`, `b4c2-ltx25-mode`

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-gaps-found
- Trigger: closing the six gaps showed two more settings that may change text conditioning: `FASTVIDEO_FP8` (FP8 `Linear`s; Wan UMT5, possibly LTX connectors) and `FASTVIDEO_SDPA` (attention backend; routing of the text encoders unverified)
- Options: close them in B4c2; record them
- Decision: recorded as ADR-0021 "B4c2" rows 7 and 8, not closed (the brief: a gap larger than the table is reported, not widened into the PR)
- Reason: each needs its own verification of what the encoders actually do
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "B4c2"
- Verification: n/a (recorded)

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-gpu-cells
- Trigger: B4c2 needs GPU evidence that each fix changes a wrong hit into a correct miss, and that default settings move no output bit
- Options: extend `b4c`; a `b4c2` family beside it
- Decision: `runpod-matrix.sh b4c2` (`FV_B4C2_CELLS`): switch cells `b4c2-ltx25-fp8`, `b4c2-ltx25-mode`, `b4c2-h3-numerics`, `b4c2-h3-tree` (write under one setting, read under another, plus a fresh encode over a new cache; verdict `b4c2_switch_verdict`: main's read hits, the PR's misses, equals the fresh encode in frames and stored entry) and `b4c2-h3-noop` (`b4c2_noop_verdict`: base and PR byte-identical on miss and hit, PR hit equals its miss, same entry bytes under a new name); the mode and tree cells also record whether those paths are bit-equal on the device
- Reason: positive evidence on both sides (the gap reproduced on main, closed here); only EU-volume trees (no recovered 8B, no Wan)
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "B4c2"
- Verification: `bash scripts/gpu/tests/b4-verdict.test.sh`; `FV_DRY_RUN=1 FV_BASELINE_SHA=17e4a1dc bash scripts/gpu/runpod-matrix.sh b4c2`; GPU run by the coordinator

### FVID · 2026-10-24 · FVID-2026-10-24-prompt-cache-b4c2-review
- Trigger: independent review of PR #178 (coordinator): H3 resident and streamed FP8 shared a key (the same unproven equality as LTX gap 5); Wan's `act` field was read outside UMT5's f32 scope; the field choices had no tests; LTX FP8 with an explicit streamed residency was keyed as FP8 though it streams bf16; identities re-read per prompt
- Options: fix on the branch; record
- Decision: (coordinator) fix on the branch, domain-only versioning and rows 7 / 8 as follow-ups confirmed. H3 key gets the `TextMode` field (`EncodeSettings`, shared with LTX); `EncoderRun` (`h3/text.rs`) is the pure choice of weights and mode, table-tested; Wan reads `Numerics::live` inside `with_bf16_act(false, ..)`; LTX `plan_mode`, `gemma_tree`, `gemma_weights_read` are pure and tested, FP8 + explicit streamed is keyed `weights=checkpoint` (computation unchanged); H3 tree and recovered identities through `identity::files_head_tail_cached` (re-read when a file's size, modification time or the file set changes); no H3 streamed-FP8 GPU cell (no flag forces it)
- Reason: the key names what ran; a spurious miss costs one encode, a wrong hit is silent wrong output
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0021-prompt-cache.md "B4c2"
- Verification: pins recomputed in Python (H3 `29d517fd…`, `9556f05a…`, `59ddcb14…`, recovered `1af19d5f…`, Wan `1f09144a…`); 31 mutations; gates on the build pod


### FVID · 2026-10-24 · FVID-2026-10-24-lab-report-ci-bridge
- Trigger: plan D3 / D4 (CPU parts): `fv-lab report tiers` and `docs/perf/tiers.md`; the CI bridge (gold files parse and resolve, `last_validated.commit` age, an on-demand `fv-lab run gold --all`); the GPU gold runs are out of budget now
- Options: table from the gold files' cited numbers; table from one run set only. Gold runs: per-variant CLI commands written by hand; one serve-engine command per variant (`fv-gpucheck engine --model <variant>`). Cells: generated in memory; committed and drift-tested. Gold tier marker: a new tiers-schema field; the reserved tier name `gold`
- Decision: the table renders one run set only (arm named by variant id; skipped, failed or unrecorded = "not measured"; `$/clip` = the run's price x its wall time) and names the run set in its header for `--check`; committed with `Run set: none` (no committed run set measured a gold cell: `rtx6000`'s `sol-h3` arm is the CLI recipe at another canvas). `fv-lab gold check` resolves against `families/` (variant, family, tier, profile, registry name; a rev needs a registry name), checks validated commits with `git cat-file -e`, prints the age table; warn-only, `--max-age-days` opt-in. The `gold` tier runs each variant through the serve engine at its gold canvas, cells committed under `lab/cells/gold-*.toml`, generated by `fv-lab gold cells` and drift-tested; no schema change; the script-equality tests skip the `gold` tier, whose dry run is fv-lab's own golden
- Reason: never invent (a table that mixes cited and measured numbers hides which is which); the engine is the served configuration, so the canonical cell needs no hand-written flags; committed cells keep the runner, pod and fetch unchanged
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md
- Verification: `tests/tiers_report.rs` (fixture run-set golden, committed table), `tests/gold_tier.rs`, `goldcheck` unit tests; gates on the build pod

### FVID · 2026-10-24 · FVID-2026-10-24-gold-run-secret
- Trigger: D4's on-demand `fv-lab run gold --all` workflow spends money and needs a Runpod API key in CI
- Options: add the secret; leave it to the owner
- Decision: `gold-run.yml` is `workflow_dispatch` only, refuses without `RUNPOD_API_KEY` (and without `confirm = spend`), and runs on fv-deploy's guards; no secret is added or requested: adding `RUNPOD_API_KEY` to the repository is the owner's decision (open)
- Reason: CLAUDE.md money rules; secrets are the owner's
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §4
- Verification: the workflow's first step; owner decision pending

### FVID · 2026-10-24 · FVID-2026-10-24-lab-report-ci-bridge-review
- Trigger: independent review of PR #181 (coordinator): a pod could leak on a cancelled `gold-run` (record written only after readiness, second signal exits without cleanup, backstop killed with the runner); the `cells` input passed flags; `RUNPOD_API_KEY` was job-wide; a same-named arm of another tier counted as a variant's measurement; the table step needed exactly one run dir; gold-check paths; `git cat-file` could not tell an ambiguous short sha; nits (decode column, orphan cells, `|` in reasons, re-runs spend)
- Options: fix in this PR; record as follow-ups
- Decision: (coordinator) fix in this PR. `pod down --from-ledger` deletes every `runpod-pod-created` id with no `-deleted` line (the ledger line is written right after the create), `pod down <id>` needs no record, and gold-run's always() step uses it; every fv-lab pod (and every `runpod-http.sh run` pod, in lockstep for the payload goldens) deletes itself at the tier's wall cap after its container starts; `pod up --cells` must name arms of the tier and the workflow checks each word; the key is step-level env of the Runpod steps, the actions pinned by commit read from their tags, a `gold-runs` environment named and recommended (main only, required reviewers); the report counts an arm only from a `gold` run set or an engine report of that variant; a unique per-run tag; `git rev-parse --verify`, an ambiguous sha fails, `merge-base --is-ancestor` marks non-ancestors; decode through the MP4 in both kinds; `gold cells --check` catches orphans and the tier list
- Reason: the guards must hold before the owner adds the secret
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §1, §2, §4
- Verification: `pod::tests::{every_pod_deletes_itself_at_its_wall_cap, the_ledger_lists_pods_created_and_not_deleted, cells_must_be_arms_of_the_tier}`, `tests/fv_lab_bin.rs` (`pod down --from-ledger`, `pod up --cells`), `tests/payload_golden.rs` against the regenerated `runpod-http-*.json`, `goldcheck` and `goldcell::drift` unit tests, `tests/tiers_report.rs`; gates on the build pod

### FVID · 2026-10-24 · FVID-2026-10-24-pod-self-delete-graphql
- Trigger: the coordinator's live test of #181's pod-side wall cap (2026-10-24 21:16–21:45Z, EU-RO-1): pod `pq69qpoh5b0mxy` (`FV_POD_CAP_S=240`, driver and backstop killed) was still running after nine minutes; on a CPU probe pod the pod-scoped `RUNPOD_API_KEY` gets HTTP 403 on `DELETE rest.runpod.io/v1/pods/$RUNPOD_POD_ID`, while GraphQL `podTerminate` with the same key answered 200 and the pod was gone within a second
- Options: GraphQL `podTerminate`; REST with another key in the pod (a full account key on every pod: no)
- Decision: every in-pod self-delete of the GPU test pods (the cap and idle watchdogs of `runpod-http.sh` `idle_watchdog` / `start_cmd` and fv-lab's `src/pod-start.sh.in`) calls GraphQL `podTerminate` with the pod-scoped key; success = HTTP 200 without `"errors"`, retried each minute; no REST DELETE left there; the other REST self-deletes are listed in ADR-0023 §4 as a follow-up (scripts/serve, experiments, control, fv-boot's watchdog), not changed in this PR
- Reason: the idle watchdogs documented since D1a never deleted a pod; only the driver-side backstop did
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ); probes by the coordinator
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §4
- Verification: `pod::tests::every_pod_deletes_itself_at_its_wall_cap`; the rendered function run locally against a fake curl (200 + data: success; 200 + errors and 403: retry; the request body is exactly the probe's); payload goldens regenerated and equal; the coordinator's live cap test on the pushed commit

### FVID · 2026-10-25 · FVID-2026-10-25-pod-self-delete-everywhere
- Trigger: ADR-0023 §4's follow-up list (owner-approved): 25 more in-pod self-deletes still DELETE `rest.runpod.io/v1/pods/$RUNPOD_POD_ID` with the pod-scoped key (HTTP 403); fv-control's edge/standalone watchdog and fv-boot's `delete_runpod` use `FV_BACKSTOP_API_KEY` and count any HTTP answer, a 403 included, as done
- Options: 25 hand copies; one library sourced on the pod (the pod scripts are shipped alone and cannot source the repository); one library embedded by the local launchers plus checked inline copies in the pod scripts
- Decision: `scripts/gpu/lib/self-delete.sh` holds the one `fv_self_delete` (#181's text); the launchers embed it into the start commands they send, the pod scripts carry the same text inline, each retries until success (the busy-GPU exits 5 times, then the start command's own retry); `scripts/gpu/tests/self-delete.test.sh` (in `control.yml`) fails on any REST self-delete with the pod key and on any copy that differs from the library, and runs the function against a stand-in curl. `FV_BACKSTOP_API_KEY` is fv-control's account key (not pod-scoped), so payloads.ts and watchdog.rs keep REST, but only a 2xx or a 404 now counts as deleted; anything else is retried
- Reason: the in-pod idle and cap deletes never worked; one text keeps them the same everywhere
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §4
- Verification: `scripts/gpu/tests/self-delete.test.sh` (37 inline copies + `ltx-pod.sh entry`; fails with a site reverted (tried on sol-bench-pod.sh, h3-glue-validate.sh, ltx-pod.sh, pod-boot.sh), a changed copy, a wrong library path and a broken success rule), `boot::tests::watchdog_retries_a_refused_runpod_delete`, `edge.test.ts` "counts only a 2xx or a 404 as deleted"; `lint.sh`, `fv-lab-goldens.sh --check` (fixtures unchanged), control's gen-check, typecheck and unit tests; fmt, clippy, `cargo test -p fastvideo-deploy`, `cargo test --workspace --lib` on the build pod; the coordinator's live check pending

### FVID · 2026-10-25 · FVID-2026-10-25-pod-self-delete-review
- Trigger: the coordinator's live check of #183 passed (CPU pod `inuqqmjr86ah47`, gone on the first attempt at 17:01:57Z); its review found the check blind to order and to what is sent (the library appended, read but unsent, or the inline definition below the call all passed), the CI paths narrower than the scan, REST deletes through an alias (`$FV_POD_ID`, any variable, multi-line Python) not caught, a pause after the last watchdog attempt, a global `code` and a `.resp` left in served log dirs
- Options: fix in #183; follow-up PR
- Decision: (coordinator) fix in #183. The `*-validate.sh` build their command in `start_cmd` (`up` sends it, `print-start` prints it); the check renders every launcher (`print-start`, `ltx-pod.sh entry`), requires one copy defined before the call, runs its self-delete with stand-in curl/sleep (1 GraphQL call on 200, 3 on 403), checks definition-before-call in every inline copy, flags REST deletes of `RUNPOD_POD_ID` / `FV_POD_ID` anywhere and any REST pod delete outside a `LOCAL` list, and runs in its own `self-delete.yml` on every path it scans; `fv_self_delete` passes the key on curl's stdin (`-H @-`), uses `_fvsd_code` / `_fvsd_ok`, removes `<log>.resp` (every copy, runpod-http.sh / pod-start.sh.in and their goldens included); the payloads.ts watchdog skips the pause after its last attempt
- Reason: the review's mutations must fail the check; the key and the answers stay out of argv and served directories
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ); review and live check by the coordinator
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §4
- Verification: the check fails on each of the review's mutations and eight more (ADR-0023 §4 lists them) and passes on the head; `pod::tests::every_pod_deletes_itself_at_its_wall_cap` (stdin key, `rm -f "$1.resp"`, no `Bearer $RUNPOD_API_KEY`); `edge.test.ts` (no pause after the last attempt); goldens regenerated and `--check` green; build-pod gates. The changed function text is not yet live-checked

### FVID · 2026-10-25 · FVID-2026-10-25-retire-gated-script-families
- Trigger: the coordinator's GPU gates (ADR-0019 §8, §9.5) passed for all ten fv-lab tiers: `rtx6000` (D1a, #171), `techniques`, `b6`, `precision`, `rtx5090` (#176) and `fastvideo`, `h3arms`, `trace`, `ltxfps`, `ltxvae` (#184); every arm that ran was byte-identical; the coordinator asked for the branches' retirement in the next D1 PR
- Options: delete the ten `case` branches and route the families to fv-lab, with their fixtures frozen; keep the branches until the whole script goes
- Decision: (coordinator) delete them. `scripts/gpu/fv-lab-retired.sh` lists the retired families; `runpod-http.sh run` / `plan` for one runs `fv-lab pod up` (FV_CELLS, FV_BASELINE_SHA, FV_EXTRA_ENV, FV_LPIPS mapped; settings fv-lab lacks refused; `FV_LAB_BIN` / PATH / `./fv-lab`; `FV_DRY_RUN=1` prints the command); their goldens are frozen (fv-lab's own since the gate, changed only on purpose with the cell in one PR); `runpod.sh rtx6000` deleted; `headline` runs its cells of the retired tiers through `fv-lab run --runs-dir`; `precision`'s `ltx-nvfp4` mode and `ltxvae`'s `FV_VAE_GEN` arms are dropped with the branches (a tier each if wanted)
- Reason: two runners for one tier drift; the gate showed they are equal, so the script copy only costs upkeep
- Reversibility: cheap (git history has the branches; a family returns by restoring its branch and leaving the list)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §10
- Verification: `fv-lab-goldens.sh --check` (frozen fixtures present, branches gone, the guards fail when not), `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `tests/fv_lab_bin.rs::runs_dir_replaces_the_run_directory`, every remaining family's dry run equal before and after, `runpod-http.sh` dry runs and a fake `FV_LAB_BIN`; gates on the build pod

### FVID · 2026-10-25 · FVID-2026-10-25-retire-gated-script-families-review
- Trigger: independent review of PR #185 (coordinator): host settings fv-lab ignores were dropped silently (worst: `FV_POD_CAP_S=1800` became the tier's 14400 s), `FV_EXTRA_ENV` passed the dropped modes' knobs to a pod that ignores them, `headline` ignored `FV_PROMPTS` / `FV_PROMPT` / `FV_SEED` and swallowed a missing fv-lab, host-side globbing of `FV_CELLS` / `FV_EXTRA_ENV`, truncated usage texts, stale lab comments, a CI path filter, `FV_FAMILY=gold runpod-http.sh run` creating a pod that runs nothing, `FV_GEN_TIMEOUT_S` compared with a literal
- Options: fix in #185; follow-ups
- Decision: (coordinator) fix in #185. `fv-lab pod up --wall-cap-s` lowers a tier's cap and `FV_POD_CAP_S` maps to it; every other script-path setting is refused unless fv-lab does the same (the tier's value from `lab/tiers.toml`, or fv-lab's fixed one); `FV_EXTRA_ENV` keys no tier reads are refused; `headline` exits 2 on a missing fv-lab or a prompt / seed override and logs a failed tier as the recursive run did; `gold` refused for `run` / `all`; lists read with `read -ra`; usage printed to the end of the header; `has_branch` reads the top-level `case "$FAMILY"` labels as bash does
- Reason: "refused, not dropped" must hold for every setting, or a run silently differs from what was paid for
- Reversibility: cheap
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §10
- Verification: `scripts/gpu/tests/fv-lab-route.test.sh` (67 checks; each fix's mutation fails it), `tests/fv_lab_bin.rs::wall_cap_can_only_be_lowered`, `tests/dry_run.rs::branch_labels_are_read_as_bash_reads_them`; gates on the build pod

### FVID · 2026-10-25 · FVID-2026-10-25-one-image-boot-config
- Trigger: ADR-0011's open owner decision (one CUDA image whose worker config is always supplied at boot, review §7.4)
- Options: keep seven variant images with baked configs; one CUDA image + the CPU image, config sent at boot
- Decision: (owner, 2026-10-25) one image with boot-time config. `[[images]]` is `cuda` (carries only `runpod-fake.toml`) and `cpu`; every preset runs on `cuda`; every launch path sends `FV_WORKER_TOML_B64` (or `FV_WORKER_CONFIG` for a file the image carries); fv-boot refuses to boot without a config; stored specs naming a `variant` load through a mapping; an encoded config over 16 KiB is refused (Runpod documents no per-value limit; the largest config is 5600 bytes encoded). Live templates, endpoints and GHCR tags are moved by the owner's ordered steps, not by the change
- Reason: the variants differed only in a few-KB config layer; 22 of 28 presets already sent their config inline; one image, one CI build, one template pair
- Reversibility: medium (code: revert the PR; live: roll fv-control back to the previous deployment, whose variant tags remain; `release.sh rollback` cannot cross the change)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0011-deploy-section.md "One image with boot-time config"; docs/serve/images.md "One image: the live steps"
- Verification: `fastvideo-deploy` `an_image_started_without_a_config_refuses_to_boot`, `boot::config::tests::no_config_refuses_to_boot`, `payload_goldens::every_worker_config_fits_the_env_size_guard`; `fastvideo-manifest` deploy tests (one GPU image, smoke config carried); fv-control `payloads.test.ts` (read-compat, size guard), `gen-configs.mjs --check`; `ci-images.sh smoke` asserts the refusal (runs in CI on the image build); live steps pending the owner

### FVID · 2026-10-25 · FVID-2026-10-25-fv-lab-verdict-step
- Trigger: plan D1d (first slice): `b4a` stayed in `runpod-matrix.sh` because its verdict (reuse counts, cache evidence, the evidence diff, per-cell and family verdict files) was neither a compare nor a gate (ADR-0019 §9.1); `gate` was not an fv-lab subcommand
- Options: a shell hook per cell; a typed `[verdict]` in the cell schema; keep `b4a` in the script
- Decision: a typed, optional `[verdict]` table (cell v1, additive): fresh verdict arms, per-arm exit / evidence files / line counts, per-cell `--off-identity` pairs and evidence diffs, `<family>-<cell>.verdict.json` and `<family>.verdict.json` as the script writes them; `b4a` moved as a tier (3 cells, goldens from the script; the branch stays until its GPU gate); `fv-lab gate <run dir>` re-evaluates a finished run's promotion gates and verdicts offline; the promotion rule stays `fv-gpucheck gate` (aliased, not moved); `fv-lab pod fetch` also takes the verdict files; line patterns are substrings with `*`, no regex dependency
- Reason: equality with the script needs the same files, which a typed step gives by construction and a test can hold (the script's own branch and fv-lab on one fake box); typed data is validated by `fv-lab check` and can be re-evaluated offline; the frame generalises to `determinism` / `oracle` by adding checks, not steps
- Reversibility: high (an optional table; the script branch is untouched until the GPU gate)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §12
- Verification: `tests/verdict.rs` (script vs fv-lab verdicts, evidence and diffs byte-identical on fakes; FV_B4A_CELLS vs --cells; stale verdicts; `fv-lab gate` offline), `tests/{dry_run,payload_golden,fv_lab_bin,schema}.rs` with the `b4a` fixtures, `fv-lab-goldens.sh --check`; GPU gate (§11.6) pending, the coordinator's

### FVID · 2026-10-25 · FVID-2026-10-25-fv-lab-verdict-step-review
- Trigger: independent review of PR #188 (D1d first slice): nothing blocking in the on-pod verdict step; `fv-lab gate` and its tooling needed eight fixes
- Options: per finding, as the review proposed or documented instead; F4 either reproduce the script's cell order or document it
- Decision: `fv-lab run` records its arm selection (`<tier>.arms.json`, fetched) and `fv-lab gate` reads it, refusing to guess without it unless the run holds every arm (F1); `gate` never writes into the run directory (`--out` default `<run>.regate`) and `--check` compares the derived files with the run's (F2); a gate whose arms both ran without `benchmark.json` fails and skipped gates are counted (F3); F4 documented, not reproduced: fv-lab lists a family verdict's cells in the tier's order and `fv-lab-compare-runs.sh` compares `cells` / `why` as sets; a reused `--out` is cleared first (F5); the script's evidence `grep` / `sed` / `diff` pinned to bytes (`LC_ALL=C`, `-a`) as fv-lab reads them, the runtime image being in the C locale anyway (F6; `diff -a` also keeps run paths out of an evidence diff); `fv-lab-compare-runs.sh --expect-verdicts N` (F7); `FV_B4A_CELLS` refused by `run` and `pod up`, a duplicate evidence `diff` name refused (F8)
- Reason: an offline judge must not pass a run it cannot see whole, nor overwrite the evidence it judges; the rest closes gaps where a check could pass on nothing
- Reversibility: high (fv-lab and two script helpers; the b4a dry run is unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §11.1, §11.3, §11.4, §11.5, §11.6
- Verification: `tests/verdict.rs` (edge cases in a UTF-8 locale, one-arm selection, interrupted run, `--out` refusal, `--check`, gates skipped / failed, script vs fv-lab byte for byte), `tests/fv_lab_bin.rs` (`gate` record / `--cells` / `--check` / `--out`, `FV_B4A_CELLS` refused), `plan` validation test, `fv-lab-goldens.sh --check` (compare-runs `--expect-verdicts`, family verdict order)

### FVID · 2026-10-25 · FVID-2026-10-25-c3c-control-launch
- Trigger: plan C3 slice C3c (fv-control on the shared `Launch` schema), coordinator's work package
- Options: hand-written TS types kept in sync by review; TS types + zod validators generated from fv-deploy's committed JSON Schemas, with fv-control's Runpod bodies derived from a `Launch` by the Rust builders' rules; or extending the Rust schema first for CPU / GMI / Brev
- Decision: generate `control/src/launch-gen.ts` (types + zod) from `crates/fastvideo-deploy/schemas/*.v1.json` in `gen-configs.mjs` (`--check` in CI); every Runpod GPU pod and GPU serverless endpoint of fv-control is a schema-checked `Launch` whose bodies come from `control/src/launch.ts` (the port of `pod.rs create_body`, `serverless.rs template_body` / `endpoint_body`); CPU pods / endpoints, GMI / Brev, the REST v2 envelope and build pods stay outside Launch v1 (no CPU compute or provider kind in it); no Rust change (the schema extension is proposed for after C3b, which edits the same types)
- Reason: one source for the launch shape, drift fails CI, no new dependency (zod was already fv-control's); payloads unchanged (`control/test/unit/payload-corpus.test.ts`: 1024 cases against a snapshot of main's builders at 2258aec2)
- Reversibility: high (code only; no live resource, stored shape or API response changed)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §7
- Verification: `control/test/unit/payload-corpus.test.ts` (fv-control's bodies = main's at 2258aec2), `control/test/unit/launch-goldens.test.ts` (the C3a payload goldens from TS; h3-turbo, h3-max, ltx, ltx-pro field by field against fv-deploy with the intended differences of ADR-0018 §7.4; schema drift fails `--check` on a copy of the tree; stored specs launch), `node gen-configs.mjs --check`, `npm run typecheck`, `npx vitest run test/unit`; `cargo test --locked -p fastvideo-deploy` (schemas current)

### FVID · 2026-10-25 · FVID-2026-10-25-fv-deploy-vast-gcp
- Trigger: plan C3b (ADR-0018 "Open decisions"): Vast and GCP on fv-deploy's `Provider` trait; the autoscaler's Runpod code on fv-deploy's
- Options: (a) Vast / GCP behind the trait with the shared guards, the autoscaler's Runpod code fully on the trait; (b) the same for Vast / GCP, and for the autoscaler only the shared pieces (requests, answer mapping, balance, billed price, template pod body) written once, sans I/O, with the rest listed; (c) leave the scripts
- Decision: (agent, proposed for the coordinator's review) (b). `ProviderKind` gains `vast` and `gcp` (the only, additive, `Launch` schema change; GCP's own settings are provider options, not `Launch` fields). Vast launches the fake-engine smoke only (no weights volume, r2 not in fv-deploy yet); GCP launches `[[gcp]]` rows, quoted from `scripts/gcp/lib.sh`'s price table (no live price), the floor checked on the run's budget `FV_GCP_BUDGET_USD` (no budget, no create), the token minted by `scripts/gcp/auth.sh`; Secret Manager by name or no secrets (never secret values in a request). `vm.sh`'s E2BIG on the metadata items fixed. No live resource created; the Vast and GCP live gates are the owner's
- Reason: one lifecycle and one set of money guards for every provider, payloads pinned to the scripts by goldens; the autoscaler's full move needs a template-capable `Launch` and an async adapter (ADR-0018 §9 "What remains"), larger than C3b and touching its controller
- Reversibility: high (code only; the scripts stay until the live gates)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §8 (Vast, GCP), §9 (autoscaler)
- Verification: `payload_goldens` (`vast_search_and_create_equal_vast_sh`, `gcp_firewall_and_instance_equal_vm_sh`, the preset-diff goldens, `every_gcp_row_builds`), `provider/tests_c3b.rs` (21), `runpod::shared_tests`, the autoscaler's `the_async_client_sends_the_shared_requests`, `fv_deploy_bin` (`plan_vast_and_gcp_make_no_request_and_hide_the_keys`, `down_on_vast_and_a_foreign_gcp_vm`), `fv-deploy-goldens.sh --check`, `scripts/gcp/tests/vm.test.sh`; live gates pending the owner

### FVID · 2026-10-25 · FVID-2026-10-25-fv-deploy-vast-gcp-review
- Trigger: the independent review of #193 (C3b) and its CI failure on jq 1.6
- Options: fix each confirmed and plausible finding with a test; or defer them to C3d
- Decision: (coordinator, executed by the C3b agent) fix them in #193. The GCP budget is per `up` and shrinks by each created VM's table price × the cap, with live `fv-owner` VMs as the current spend; GCP commands on an existing VM take the recorded zone or `--zone`, never `GCP_ZONE`, and `down` of nothing is "not found"; a fixed `GCP_ACCESS_TOKEN` allows a cap of at most 50 min; floors and caps fail closed on NaN; the Vast sweep refuses a listing it cannot read and a lost create answer is swept 6 times 18 s apart; Vast never sends the UNVERIFIED `gpu_name` filter without `--gpu`; a Vast debt fails the floor; an empty source range is refused (Rust and vm.sh); vm.sh fails the payload when its metadata items fail; vast.sh and the goldens work on jq 1.6 (`$label` is a jq 1.6 keyword) and the goldens print a failing step's stderr
- Reason: each closes a path where a guard could pass on a wrong reading (a fixed budget, a default zone, NaN, an empty listing or source list, an expired token)
- Reversibility: high (code only)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §8 (§8.1 table, §8.2, §8.3, §8.4), §9
- Verification: `provider/tests_c3b.rs` review tests (`gcp_budget_shrinks_with_each_create`, `down_of_an_unknown_instance_is_not_found`, `vast_sweep_refuses_a_listing_it_cannot_read`, `vast_lost_answer_listed_late_is_swept_by_a_later_sweep`, `vast_names_gpus_only_when_asked`, `vast_billed_price_debt_and_refused_deletes`, `nan_fails_closed_and_an_empty_source_range_is_refused`), `fv_deploy_bin` (`down_on_vast_and_a_foreign_gcp_vm`, `gcp_cli_refuses_nan_budgets_empty_ranges_and_long_static_tokens`), `boot::sys` `request_debug_redacts_the_authorization_header`, `vm.test.sh` (42, jq 1.6 and 1.7), `fv-deploy-goldens.sh --check` on jq 1.6 and 1.7

### FVID · 2026-10-25 · FVID-2026-10-25-keep-lb-endpoints
- Trigger: plan F3b ("delete `multiworker.rs` and `server.workers_max`; rename `[gateway]` to `[cluster]`, drop `gw_workers`") was blocked on the owner's decision on the Runpod load-balancer (LB) mode: the edge Worker replaced the CPU gateway pod and its serverless pools, not standalone Runpod serverless endpoints in LB mode, which rely on `multiworker.rs` / `server.workers_max` (pinned routes filtered when `workers_max > 1`)
- Options: A, keep the LB endpoints (keep `multiworker.rs` and `server.workers_max`; F3b does only the renames); B, retire LB mode and delete both
- Decision: **(owner) A: keep LB endpoints; F3b reduced to the renames.** Executed as: fv-serve's `[gateway]` table is `[cluster]`, `[gateway]` read as its deprecated name (a startup warning) and a file with both refused; the rename is staged readers first (ADR-0011 "Renaming a worker-config section"): fv-serve, fv-boot (`register_off`) and fv-control's shell boots (`REGISTER_OFF_SH`) read both names now, and every shipped `configs/serve` file keeps `[gateway]` until every image a launcher can boot reads `[cluster]`, because an older image's fv-serve refuses the unknown `[cluster]` table and would fail every new worker of its pool. `gw_workers` is kept under its name (fv-autoscale's pod pools read a pod's in-flight work from it before deleting it, and workers of every image write it); the retired gateway's `gw_dispatch` / `gw_sessions` are no longer created (existing tables stay). The env names (`FV_GATEWAY_POOL`, ...) are unchanged. Nothing in `multiworker.rs` was specific to the gateway pod (its `Owner::Gateway` routes are the worker's internal routes, used by the edge, fv-control and fv-autoscale), so it is unchanged
- Reason: LB endpoints are a live deployment path the edge does not cover; a config the deployed images refuse would brick their pools at the next scale-up
- Reversibility: cheap (stage 2, the shipped configs' rename, is a later change; the alias costs a few lines)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), sub-agent of the coordinating session
- ADR: docs/adr/0011-deploy-section.md "Renaming a worker-config section"; docs/adr/0014-fv-boot.md (the registration edit)
- Verification: `fastvideo-serve` `config::tests::cluster_and_its_old_name_gateway_round_trip`, `the_old_name_reads_every_key_and_refuses_unknown_ones`, `cluster_and_gateway_together_are_refused`, `shipped_configs_round_trip_under_both_names` (every shipped config round-trips and reads the same under `[cluster]`; none says `[cluster]` yet); the LB tests (`multiworker::tests::*`, `multi_worker_*`) unchanged and green; `fastvideo-deploy` `boot::config::tests::register_edit`; fv-control `payloads.test.ts` (a `[cluster]` config edited in place), `providers.test.ts`; `gen-configs.mjs --check` and the fv-deploy goldens unchanged (the worker configs sent are byte-identical); the C3c payload corpus changes only in the shell boots' registration edit (253 `pod/*` cases, `dockerStartCmd`; the corpus with the old snippet substituted equals the previous snapshot); stage 2 pending the image roll

### FVID · 2026-10-25 · FVID-2026-10-25-keep-lb-endpoints-review
- Trigger: independent review of PR #192 (coordinator): nothing blocking (the shell boots' edit equals main's on every config without `[cluster]`; the 253 changed corpus bodies differ only in `dockerStartCmd`); three minor items: (A) fv-boot appended `[cluster]` to a config with neither table, but `FV_SERVE_BIN` (the fv-boot smoke pod's `FV_SMOKE_SERVE_BIN`) can point it at an older fv-serve; (B) fv-control accepted an inline `config_toml` with `[cluster]`, which a pre-#192 image (a pinned sha, `stable`) refuses at the `--print-config` preflight; (C) the guard test renamed only an exact `[gateway]` line
- Options: fix in #192; follow-ups
- Decision: (coordinator) fix in #192. (A) `register_off` appends `[gateway]`, as the shell boots do: nothing writes `[cluster]` in stage 1, asserted in `register_edit`; ADR-0011's stage 2 lists the four writers to flip together (shipped configs and their guard, `register_off`, `REGISTER_OFF_SH` and the payload corpus, the `config_toml` refusal). (B) `setsClusterTable` (`control/src/worker-config.ts`, a line reader: `[cluster]` / `[cluster.x]` / `[[cluster]]` / quoted headers, a top-level `cluster` key) refuses such a `config_toml` in the cluster-pool, standalone and serverless schemas and in `workerConfigProblem` (launch and stored specs), naming ADR-0011. (C) the guard renames the table through a TOML parse; the alias test covers spaced, quoted, commented and dotted spellings
- Reason: stage 1 must be safe whatever fv-serve a boot reaches; refusing early names the problem instead of a failed boot
- Reversibility: cheap (stage 2 removes the refusal)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0011-deploy-section.md "Renaming a worker-config section"; docs/adr/0014-fv-boot.md
- Verification: `boot::config::tests::register_edit` (never introduces `[cluster]`), boot tests expecting `[gateway]` appended; fv-control `payloads.test.ts` (the detector's positive and negative spellings, the message, every carried config clean) and `config-validation.test.ts` (cluster spec and serverless endpoint refuse `[cluster]`, accept `[gateway]`); `fastvideo-serve` `shipped_configs_round_trip_under_both_names` (TOML-level rename, at least one shipped `[gateway]`) and `cluster_and_its_old_name_gateway_round_trip` (header spellings); the payload corpus unchanged by these fixes

### FVID · 2026-10-25 · FVID-2026-10-25-restore-dropped-test-modes
- Trigger: #185 (ADR-0019 §10.4) dropped two modes of retired families with their `runpod-matrix.sh` branches: `precision`'s `FV_PRECISION_ARM=ltx-nvfp4` (NVFP4 video FFN against bf16, LTX-2.5 Sol) and `ltxvae`'s `FV_VAE_GEN=1` arms (a cold gen saving its latents, the streaming decoder on them); the coordinator asked the owner whether they come back
- Options: leave them in git history; restore each as an fv-lab tier; restore the script branches
- Decision: (owner, 2026-10-25, "restore dropped test modes") restore them as fv-lab tiers `precision-nvfp4` (4 arms, its own cells) and `ltxvae-gen` (`ltxvae`'s cells plus a gen cell per workload, 16 arms; the one pair of tiers sharing cells), on the `rtx6000` tier's pod (the PRO 6000, sm_120, which NVFP4 needs); no cell-schema change; fixtures are the script's dry run, gates and plan body in that mode at `5987e6a4`, frozen from the start (`FV_LAB_RETIRED`, `FV_LAB_RESTORED` in `scripts/gpu/fv-lab-retired.sh`, `fv-lab-restored-goldens.sh`); `runpod-http.sh` refuses the modes' settings in `FV_EXTRA_ENV` and names the tier (not routed: another run directory and pod name than `FV_FAMILY`'s)
- Reason: both modes produced recorded evidence (FVID-2026-09-27-ltx-nvfp4-ffn-cublaslt; the VAE's real-content PSNR) that no tier could re-measure after #185
- Reversibility: cheap (delete the two tiers, their cells and fixtures)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §12
- Verification: CPU: `tests/dry_run.rs` (dry run and gates equal the 5987e6a4 script's in the mode, the family read as the tier), `tests/payload_golden.rs` (pod body likewise, the mode's `FV_EXTRA_ENV` gone), `tests/fv_lab_bin.rs::restored_modes_dry_run_as_the_script_did`, `tests/lab_files.rs::every_cell_file_belongs_to_exactly_one_tier` (the one shared set pinned), `fv-lab-restored-goldens.sh --check`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`; GPU: pending, the coordinator's check of ADR-0019 §12 (fv-lab at the PR's commit vs the script at `5987e6a4`, image `sha-5987e6a` published, about $3.35); review of #191: `fv-lab run` / `pod up --extra-env` refuse the settings no tier reads (`knobs.rs`, the set of `runpod-http.sh`'s `FV_LAB_NOT_READ`, `tests/knobs.rs`), the mode's setting answered first with its tier, every refused key in one message; `runpod-matrix.sh` names the restored tier

### FVID · 2026-10-25 · FVID-2026-10-25-retire-b4a-script-branch
- Trigger: the `b4a` GPU gate (ADR-0019 §11.6) passed on `e82b6b7`: fv-lab and `runpod-matrix.sh b4a` byte-identical on `b4a-h3-4step-tea` (frames, the compare, 5 verdict files), both `matrix_exit=0`, both `fv-lab gate --check` clean
- Options: keep the branch as a second runner; retire it as §10 retired the ten (branch deleted, fixtures frozen, `runpod-http.sh` routing); for `tests/verdict.rs`, which ran the live branch: compare with frozen output files, or run a frozen copy of the branch
- Decision: retire it: `b4a` joins `FV_LAB_RETIRED`, the `b4a)` branch is deleted (`b4-verdict.sh` stays whole, `b4b` / `b4c` / `b4c2` use the same helpers), its three fixtures freeze, `runpod-http.sh run` / `plan` route it to `fv-lab pod up`, `FV_B4A_CELLS` moves into `FV_LAB_NOT_READ` (`knobs::MODE_KNOBS`, `SCRIPT_KNOBS` removed) and is refused with a pointer to `FV_CELLS` / `--cells`; `tests/verdict.rs` runs a frozen copy (`crates/fastvideo-lab/fixtures/b4a-script/`: the script at `8b18d89f` cut to its frame and the `b4a` branch, lines unchanged, and that commit's `b4-verdict.sh`), which keeps every script-vs-fv-lab case (UTF-8 locale, CRLF, NUL, subsets, interrupted runs) instead of freezing their outputs
- Reason: the gate is the retirement rule of §8 / §10; a frozen copy keeps the byte-for-byte differential with the fake box, where frozen outputs would pin only today's fake
- Reversibility: high (the branch is in git history and in the frozen copy)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.1, §11, §11.6, §12
- Verification: `tests/verdict.rs` (the frozen script, with everything it sources vendored and its host paths faked, against fv-lab, and the frozen script's dry run equal to `fixtures/b4a-dry-run.txt`), `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `tests/knobs.rs`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh` (b4a routed, `FV_B4A_CELLS` refused in `FV_EXTRA_ENV` and, with `FV_TECH_CELLS` / `FV_B6_CELLS`, in the host environment, review of #195; the matrix refuses `b4a`), `b4-verdict.test.sh`, `FV_DRY_RUN=1 runpod-matrix.sh <family>` before and after for the 32 remaining families; open follow-up: `gpucheck-runtime-image` should run its checks before pushing `sha-<7>` (run 40233480444 pushed a `sha-e82b6b7` whose fv-lab could not parse `[verdict]`)

### FVID · 2026-10-25 · FVID-2026-10-25-restored-tiers-gpu-check
- Trigger: FVID-2026-10-25-restore-dropped-test-modes left its GPU verification pending (ADR-0019 §12, proposed)
- Options: accept §12 on the coordinator's GPU check; keep it proposed
- Decision: §12 accepted: the coordinator's check passed for both restored tiers, fv-lab at main `8b18d89` (image `sha-8b18d89`, tools compiled) against `runpod-http.sh` at `5987e6a4` in each mode, on the 4k5s cells
- Reason: `ltxvae-gen` (`ltx25-4k5s-gen`, `-gen-streaming`) 121 frames per arm byte-identical, 4 reports the same; `precision-nvfp4` (`ltx25-4k5s-sol-bf16`, `-nvfp4`) 242 frames per arm byte-identical, 3 reports the same including the gate report (lossy verdict `fail` on both, as in 2026-09); `fv-lab-compare-runs.sh` exit 0 and `matrix_exit=0` on all four pods
- Reversibility: high (a record; the tiers are unchanged)
- Executed by: coordinator (GPU run, pods `wgzwvt5mjcn1nv`, `ymrchzi0z3eixy`, `64opofps1163h8`, `kxhdp1v19w151y`, all deleted); recorded by Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §12 "Result"
- Verification: `artifacts/runpod/ltxvae-gen/compare-8b18d89.txt`, `artifacts/runpod/precision-nvfp4/compare-8b18d89.txt` and the four run directories under `artifacts/runpod/{ltxvae-gen,ltxvae,precision-nvfp4,precision}/`

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-report
- Trigger: plan D1c (ADR-0019 §8): `fv-lab report {summary,table}` replacing the `*-summary.py`, `hd_report.py`, `sol_bench_table.py`, `timings.sh`, and the `FV_FETCH_TREE` mirror; the retired families' `live.log` summary lines had no reader since §10
- Options: reports as new Rust tables of our own choosing; ports equal to the scripts line for line with goldens from the scripts; for `--fetch-skip`: a regex dependency, `grep -E` as a process, or an ERE subset refused outside it; for the replaced scripts: delete all, keep all, delete those nothing calls
- Decision: line-for-line ports (`report summary --kind tier|hd|h3-graphs|h3-tier-options|h3-trim-nvfp4|fasth3-nvfp4`, `report table --kind sol-bench|timings`) held to fixtures the scripts write on committed run directories (`fv-lab-report-goldens.sh`, `tests/report.rs`), Python's JSON and number spelling in `pyfmt.rs` (no dependency); the deleted branches' summary lines frozen verbatim in `fixtures/report/summary-script.sh`; `pod up --wait` / `pod fetch --fetch-tree [--fetch-skip] [--fetch-max-mb]` with an ERE subset and `runpod-http.sh` mapping `FV_FETCH_TREE=1` across; `sol_bench_table.py` and the four experiment `*-summary.py` deleted (no caller), `hd_report.py` (keyframe sharpness needs numpy) and `timings.sh` (`validate.sh` calls it) kept until their callers move
- Reason: equality with the scripts is what lets a script be deleted; the fixtures are the scripts' bytes, and every summary line of the frozen version's script runs is in that run's `live.log`; every skip pattern the repository uses is in the subset, and a pattern outside it fails before a pod exists
- Reversibility: high (the scripts are in git history at `f9e57ea2`; the reports are additive)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §13
- Verification: CPU: `tests/report.rs` (52 fixtures byte for byte), `tests/fetch_tree.rs` (the script's own `fetch_tree` against a fake `curl`, four cases, trees byte-identical), `tests/fv_lab_bin.rs` (the commands, refusals), `fv-lab-report-goldens.sh --check` (CI: `control.yml`), `fv-lab-route.test.sh`; no GPU (the reports read files) Review of #198: the tree fetch treats the pod's listing as untrusted (single-name entries only, depth 16, a size cap enforced while streaming through an additive `Http::download_capped`, 5 listing tries, `RUN_FILES` / `DONE` always fetched), and those cases are tested on fake listings (`pod::tests::the_tree_fetch_stays_inside_the_run_directory` and two more) and the reqwest cap on a local server.

### FVID · 2026-10-26 · FVID-2026-10-26-fv-deploy-c3d
- Trigger: plan C3d (ADR-0018 "Open decisions", §6 "Kept: … `runpod-templates.sh` and `release.sh` (C3d)", §7.3's proposal)
- Options: (a) templates, release promote / rollback, reconcile and the D1 ledger in fv-deploy with request goldens from the scripts, and the guards' numbers as `Option` (null = not checked by this launcher); (b) slices 1 + 2 only (templates, release), the ledger and the schema change later; (c) leave the scripts
- Decision: (agent, proposed for the coordinator's review) (a). The serve templates are a `Launch` built by the Runpod template builder; promote / rollback / reconcile send the scripts' requests (the same SQL, the deployments upsert's column order, the `digests` key order); `up` / `down` / the sweeps write the same `deployments` rows as the deploy scripts through a `Recorder` (Runpod kinds only; differences: `created_by …/fv-deploy`, `region`, `meta.guards`); `Guards.max_usd_per_hr`, `cap_s`, `ready_timeout_s`, `ready_cycles` become required-nullable (`null` refused by fv-deploy's `up`, shown as "not checked"; fv-control's serverless endpoints write `null`). `mock_api.py` logs requests and serves digests that are the SHA-256 of the manifest bytes. No new dependency; no live resource touched
- Reason: the last two scripts ADR-0018 kept, on the shared seams and pinned to the scripts by goldens; one ledger for the scripts, fv-deploy and fv-control; the fail-closed 0 that fv-control wrote for unchecked guards could be read as a cap by a consumer
- Reversibility: high (code; the schema change is additive; the scripts stay until the live gates)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §10
- Verification: `tests/release_goldens.rs` (28 scenarios against `mock_api.py` with the effects' order and two partial failures, mutation-checked), `payload_goldens` `serve_templates_equal_runpod_templates_sh`, `fv_deploy_bin` `templates_plan_and_release_refusals_make_no_request`, `lifecycle::tests` `not_checked_guards_are_refused_and_shown`, `the_recorder_sees_create_ready_and_delete`, the `release` unit tests, `fv-deploy-goldens.sh --check` (jq 1.6 and 1.7), `release.test.sh` (50), control `launch-goldens.test.ts`; live gates pending the owner (§10.6)

### FVID · 2026-10-26 · FVID-2026-10-26-fv-deploy-c3d-review
- Trigger: the coordinator's review of #199 (one blocking finding, nine more)
- Options: fix each in #199 with a test; or defer the non-blocking ones
- Decision: (coordinator, executed by the C3d agent) fix all in #199. The D1 `created` write comes after the backstop and the billed-price check (a pod deleted at once gets no row); a Runpod listing that is not one fails closed; the goldens pin the effects' order (retags logged in the mock's sequence) and two partial failures (template PATCH 500, release INSERT 500, after the retags); a guard key missing from a `Launch` fails at parse; the python-backed goldens fail in CI without python3; the row's channel is the launch's; the detached backstop records its delete; one `registry.sql` lookup; fv-control names `cap_s` as unchecked without a deadline; ADR §7.2 / §7.3 marked superseded by §10.4
- Reason: the backstop must never wait on a ledger write; an unreadable listing must never read as "nothing"; the goldens must catch reordering and partial failure
- Reversibility: high (code only)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0018-fv-deploy.md §10.3, §10.4, §10.7
- Verification: `the_ledger_hears_of_a_create_only_after_its_backstop_and_price_check`, `the_backstop_records_its_delete`, `a_listing_that_is_not_one_is_an_error`, `reconcile_and_sync_stop_on_an_answer_that_is_not_a_listing`, `not_checked_guards_are_refused_and_shown` (missing key), `release_goldens` (28 scenarios, order, the launch channel), `fv-deploy-goldens.sh --check` (jq 1.6 and 1.7), control `launch-goldens.test.ts`

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-one-offs
- Trigger: plan D1, slice D1e (ADR-0019 §8: "the 24 port one-offs in `scripts/gpu/experiments/` become cells or are deleted; `runpod-http.sh` is deleted once nothing calls it")
- Options: (a) every experiment as cells; (b) keep only what a gold file, a served variant or an open item still depends on and fv-lab can run (EU-volume trees, no upstream stack), delete the finished rest; (c) leave the directory as it is
- Decision: (agent, proposed for the coordinator's review) (b). `h3-graphs` becomes the tier `h3-graphs` (its h3-turbo cells: the open item of `docs/perf/h3-denoise-graphs.md` §5 on the default H3 tier, all trees on the EU volume), its commands cut from `h3-graphs-pod.sh` itself by `fv-lab-experiment-goldens.sh` and held word for word after a documented path mapping (`tests/experiment_tiers.rs`); its scripts stay until its GPU check (§14.4, ~$1.45). The other nine (`fasth3-nvfp4`, `h3-fp4attn`, `h3-glue`, `h3-tier-options`, `h3-tier-ship`, `h3-trim`, `h3-trim-nvfp4`, `h3-turbo-loras`, `vdn-h3`) are deleted: finished, recorded in `docs/ports` / `docs/perf` with their evidence in `artifacts/`, their trees R2-only and half of them upstream-parity runs (the oracle's, D1d). `FV_LAB_EXPERIMENTS` marks tiers made from a one-off (never script families: `runpod-http.sh run` / `all` refuse them, as `gold`). `runpod-http.sh` stays: its callers are listed (§14.6). The `runpod-matrix.sh` families §9.1 marks "D1e or retired" get a proposal only (§14.7): D1d edits that script in parallel. No new dependency; no live resource touched
- Reason: a one-off kept as data must run on what fv-lab reaches today and measure something still open; the gold tier, the byte-identity tiers, `fv-gpucheck kernels` and the oracle cover what the deleted ones proved, and their results and evidence stay
- Reversibility: high (the deleted scripts are in git at `22a5fd7b`; the tier is additive)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §14
- Verification: CPU: `tests/experiment_tiers.rs` (the four arms equal the pod script's turbo commands under the §14.3 mapping, the tier's dry run equals its golden, the compares), `fv-lab-experiment-goldens.sh --check` (the script's 12 commands, its prompt sets byte for byte; CI `control.yml`), `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`, `self-delete.test.sh`, the fastvideo-lab and workspace tests; GPU: §14.4 pending (the coordinator's)

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-one-offs-review
- Trigger: the coordinator's review of #200 at `bfa9a1dc` (nothing blocking; the gawk fix confirmed)
- Options: fix each in #200; or defer
- Decision: (coordinator, executed by the D1e agent) fix all in #200. ADR-0019 §14.4 and `scripts/gpu/experiments/README.md` say the commit deleting `h3-graphs-{validate,pod}.sh` must drop its `LOCAL` entry in `self-delete.test.sh` and lower the launcher floor to 2; the §14.2 rows of `h3-tier-options`, `h3-tier-ship`, `h3-fp4attn`, `vdn-h3` (and those docs) say that, until PR #201 (the gold tier on `volume+r2` weights) is merged and the gold run done, their numbers' reproduction is `git show 22a5fd7b:…`; `docs/perf/h3-denoise-graphs.md` uses the §14.4 command (`--wall-cap-s 3600 --fetch-tree --fetch-skip …`); the §14.4 checks fail loudly; the weak "vdn-h3 gold cites the doc" keep reason dropped; the `--warm` cold clip's non-comparison listed among the differences; `--no-renames` on the goldens' deletion lookup; the frames estimate 8-10 GB
- Reason: the deletion of the kept scripts must not break a test; a deleted experiment's gold numbers must say how to reproduce them until the gold tier can
- Reversibility: high (docs and a test script)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §14.2-§14.4
- Verification: `fv-lab-experiment-goldens.sh --check` under gawk, mawk and busybox awk; the shell goldens and route / self-delete tests; the fastvideo-lab tests on the build pod

### FVID · 2026-10-26 · FVID-2026-10-26-d1d-oracle-determinism
- Trigger: plan D1d (ADR-0019 §8, §11.7 item 4, §13.6): the `oracle` family and the `determinism` / `det-short` families still lived only in `runpod-matrix.sh`; their frame-hash equality was a report a reader judged
- Options: hash checks as a list under `[verdict]` (with its fresh arms and verdict files) or a `[repeats]` table beside it that writes only the script's files; the whole oracle (with its ffmpeg clip metrics) or its tensor-diff targets now; `runpod-http.sh upstream` moved into fv-lab now, aliased, or left with `oracle.sh` until an `fv-lab oracle` driver
- Decision: (agent, proposed for the coordinator's review) three tiers (`det-short` 15 arms, `determinism` 9, `oracle` 32 for its nine tensor-diff targets) with goldens from the live branches (`oracle` under `FV_ORACLE_TARGETS` = those nine and `FV_GEN_TIMEOUT_S=5400`); additive cell-schema fields: optional `gate` (ungated arms), `fresh` / `requires` / `echo`, `[reference]` (`{ref}`, background fetch, `ORACLE_REFS_DONE`), `[[prompt_file]]`, `[repeats]` (hashes, `determinism.txt` and a typed verdict that writes no file); branches kept until each family's GPU check; the clip-metric targets, `sfwan13`, det-short's `ltx-sol` and `runpod-http.sh upstream` stay (ADR §15.6); fv-lab refuses the families' script knobs (`knobs::SCRIPT_KNOBS`); script changes: dry-run only for `oracle_fetch` / `oracle_diff` / prefetch / `det_hash`, `determinism`'s verdict `|| true`, `det-short`'s one flat `env`
- Reason: the GPU comparison is file for file, so fv-lab writes what the script writes and nothing more, and the verdict is decided where it can be checked (`determinism.txt`, `fv-lab gate`); the clip metrics are a derived-clip step `hd` needs too, and the upstream pod is another image and lifecycle than a tier's
- Reversibility: high (additive schema fields, new tiers; the script branches are untouched in what a real run computes)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §15
- Verification: CPU: `tests/repeats_oracle.rs` (live branches vs `fv-lab run` on one fake box, every output byte-identical, `fv-lab gate --check` clean), dry-run and payload goldens, `fv-lab-goldens.sh --check`; GPU: pending, the coordinator's (§15.7: det-short h3 group, determinism h3, oracle fasth3-4step-vsa with one upstream pod; about $6) Review of #202 (ADR §15.9): `determinism`'s `|| true` recorded as a real-run change (the family no longer dies on a missing reference; both runners report `<model>-ab: no run1 hashes`), `FV_ORACLE_URL` http(s) only with curl `--proto`/`--url`, `[reference]`+`[verdict]` and duplicate targets refused, `fresh` limited to paths below `{runs}/`, the GPU check's waits by PID with a bounded keepalive.

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-volume-r2
- Trigger: the coordinator's work order for plan D3's GPU gold runs: the `gold` tier recorded the arms whose trees are not on the 500 GB EU volume (owner decision 2026-10-13) as skipped, so about 21 of 32 variants could never be measured
- Options: (a) `volume` falls back to R2 per tree in every tier; (b) a new tier mode `volume+r2`, only the gold tier on it; (c) wait for the registry's `hydrate`; for the fetch: fv-boot's R2 source (needs fv-control's presigned plan), `r2-weights.py plan` + `r2-fetch.py` on the pod, or a new fetcher; for the key: Runpod secret references, a key pair in the pod env, or a plan presigned on the operator's machine (too large for the env, or staged through an R2 write); for the disk: keep every tree (about 650 GB) or drop what the next fetching arm does not read
- Decision: (agent, proposed for the coordinator's review) (b): `TierWeights` `volume` | `volume+r2` (tiers schema regenerated: `volume+r2` added, the four `WeightsSource` values it listed but always refused removed); `{weights}` is one merged root `<scratch>/weights` (a symlink per volume tree, fetched trees beside them); per arm, before its gate, `r2-weights.py plan <dest>@<rev>` (the only child with the key) then `r2-fetch.py` under the arm's cap, keeping fetched trees until an arm that must fetch drops what it does not read; the key as `{{ RUNPOD_SECRET_fv_r2_access_key_id }}` / `{{ RUNPOD_SECRET_fv_r2_secret_access_key }}` in the pod env; the gold tier `volume+r2` with `container_disk_gb = 300` (largest arm 147.86 GB + the 120 GB every volume tier has)
- Reason: every other tier stays byte-identical to the run its GPU gate measured (a fallback would turn `rtx6000`'s pinned `fasth3-8step` skip into a 148 GB fetch); no third fetcher (r2-fetch.py is what fv-boot runs, fv-boot's own source needs fv-control); no secret value anywhere fv-lab writes; 18 trees, 530.74 GB, each fetched once at a 147.86 GB peak
- Reversibility: high (data and code; `weights = "volume"` puts the gold tier back)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16; docs/adr/0023-lab-report-and-ci-bridge.md §3
- Verification: CPU: `tests/r2_runner.rs` (a fake python: links, drops, skips, the key on the plan step only and in no run file, apt once, a fatal layout; and the real `r2-weights.py` + `r2-fetch.py` against `fake_r2.py`), `tests/gold_tier.rs` (every gold tree on the volume or in R2, the 147.86 GB peak and 530.74 GB total, the disk, the dry-run golden), `tests/payload_golden.rs` (the env refs the one difference), `weights::tests`, `fv-lab-goldens.sh --check` and `fv-lab-restored-goldens.sh --check` (only `runpod-http-gold.json` changed: the disk); GPU: the full gold run of ADR-0019 §16.7 (the coordinator's; UNVERIFIED that the `fv_r2_*` secrets read `fv-weights`, open decision 4)

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-volume-r2-review
- Trigger: the coordinator's review of #201 (one blocking finding: the merged weights root defaulted onto the volume without `--scratch`)
- Options: fix each in #201 with a test; for the idle watchdog, a heartbeat file in every tier's start command (changes the scripts' frozen pod goldens) or a higher limit for `volume+r2` pods; for the run's deadline, a start-command variable (same objection) or the pod's env
- Decision: (coordinator, executed by the agent) fix all in #201: `weights::check_off_volume` refuses (exit 2) a scratch disk equal to the work directory and a root inside or above it or `/workspace`, checked again before every fetch, removal and clean-up; the `volume+r2` idle limit `r2_idle_min` (85 min for the gold tier); `FV_RUN_DEADLINE` in the pod's env from `pod up` (cap less 20 min), arms skipped as `wall cap near` or capped to the time left, the driver waiting until the cap less 15 min and then fetching; `r2-weights.py plan --pin <weights-r2.tsv>`; a partly failed fetch keeps its complete trees and removes the rest; `python3 by an apt install at the first fetch (update and install under one `timeout 600`, a 120 s dpkg lock wait, retried at the next fetching arm; not in the runtime image, which is Python-free by design and whose CI job fails on python3: the review's image suggestion dropped); the smaller fixes of ADR-0019 §16.9
- Reason: CLAUDE.md's add-only volume rule must hold for any invocation, not only the pod's; a fetch must not look idle to the watchdog; a long run must leave a record; the pod must load the manifest the repository recorded
- Reversibility: high (code and data)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16.3, §16.4, §16.6, §16.7, §16.9
- Verification: `tests/fv_lab_bin.rs` `a_volume_r2_root_on_the_volume_is_refused`, `weights::tests` (`the_merged_root_stays_off_the_volume`, `a_tree_under_a_volume_tree_is_the_volumes_…`, `tree_names_are_checked`), `tests/r2_runner.rs` (`a_partly_failed_fetch_…`, `near_the_deadline_…`, the pin refusal against `fake_r2.py`), `pod::tests` (idle limits, `FV_RUN_DEADLINE`, `wait_limit_s`), `tests/lab_files.rs` `the_runtime_image_stays_python_free_…`, `tests/r2_runner.rs` `a_failed_python3_install_is_tried_again_…`, `test_r2_weights.py` `test_plan_pin_…`; `fv-lab-goldens.sh --check` (only `runpod-http-gold.json` changed: disk and idle limits)

### FVID · 2026-10-26 · FVID-2026-10-26-gold-cells-techniques
- Trigger: the gold smoke run on `sha-f93a961` (`--cells h3-larry-turbo,vdn-h3,sol-h3`) failed `h3-larry-turbo` and `sol-h3` at once with exit 2: "technique settings already installed (None: {}) differ from what the served models need (Some(\"h3/larry_turbo_dense\"): …); start the process with that profile or none"; `vdn-h3`, which has no profile, passed
- Root cause: `fv-gpucheck` installs its process's technique profile at start-up (`settings::install_from_env`, none without `--techniques`), and a profile is installed once per process; `CudaBackend::new` then cannot install its model's profile and refuses the empty one. The generated gold cells ran `fv-gpucheck … engine --model <variant>` with no `--techniques`, on ADR-0023 §3's assumption that the engine installs it as in fv-serve (which installs nothing first)
- Options: (a) the gold cell generator passes the variant's profile (`--techniques <engine.profile>` before `engine`) from the family file; (b) fv-gpucheck's `engine` skips its start-up install so the backend installs the plan's profile
- Decision: (agent, proposed for the coordinator's review) (a): `goldcell::technique_profile` reads the family variant's `engine.profile`, every gold cell is regenerated with it (14 of 32 arms gain `--techniques`; the 18 whose family entry names no profile, e.g. Wan, LongLive, VDN-H3 and the NVFP4 releases, start with none, as before); the engine's comparison moves into `ProcessPlan::check_installed` (same message), which `install_process_plan` calls and a CPU test runs over every committed gold cell
- Reason: it is how every other `fv-gpucheck engine` invocation already runs (`runpod-matrix.sh` serve family, `quantize-dit`'s documented usage); (b) changes the start-up order of a shared binary for one stage, and a settings read before the backend would freeze the empty profile again; the generator stays the one source and the family file the one place a profile is named
- Reversibility: high (generated data and tests)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §3
- Verification: `crates/fastvideo-lab/tests/gold_tier.rs` `every_gold_arm_starts_with_its_technique_profile`; `crates/fastvideo-engine-service/tests/gold_cells.rs` (the engine's `ProcessPlan::for_models` + `check_installed` over every gold cell, the smoke run's refusal reproduced on the CPU, every variant's `engine.profile` equal to its plan's profile); `fixtures/gold-dry-run.txt` regenerated; GPU: pending, the coordinator's rerun of the smoke

### FVID · 2026-10-26 · FVID-2026-10-26-compare-runs-controls
- Trigger: the first `det-short` check of ADR-0019 §15.7 (commit `f93a961`): `fv-lab-compare-runs.sh --expect-verdicts 6` reported 4 differences, all from `h3-timed`, the timed-pick control, while `h3-r1`-`h3-r3` were byte-identical and both `fv-lab gate --check` passed
- Options: hard-code the control names in the script; an explicit `--controls a,b` flag; read `[repeats] controls` from the checkout's `lab/cells/*.toml`
- Decision: (coordinator, executed by the agent) read the cells (bash only, a line reader of the cells' shape; `--cells DIR` for tests); a control arm must exist in both runs, agree on `exit` / `skipped` and have the same frame count in both, its frame bytes, dump lines and own verdict files are reported (`control <arm>:`), neither differences nor verdict files; `determinism.txt` and `hashes.txt` are compared without its lines and still count (not counted when only control lines are left in both; unreadable is a difference); a name another cell judges stays judged; a name outside fv-lab's arm-name grammar exits 2 (review of #204); `--expect-verdicts` 5 for det-short's h3 group and 3 for determinism's h3 model
- Reason: fv-lab's verdict judges only `judged` (`repeats.rs`); the timed pick is not reproducible across processes (docs/perf/determinism.md), so a byte comparison of it fails legitimate pairs of runs; the cells are the source of truth at the compared commit
- Reversibility: high (script)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §15.7
- Verification: `fv-lab-goldens.sh --check` (a control's bytes differing passes with its lines out of the counted files; a judged arm differing, a control in one run only, another control exit, or a name another cell judges differing fail; `h3-timed` of the checkout's cells is a control; an unclosed controls array or a control named `a b` exits 2; a judged `tcx` beside the control `tc` differing fails; a control's frame count differing fails; control lines only are not counted; unreadable reports differ); on the real runs: det-short `f93a961-1026045808-a9a4` / `f93a961-10260508` exit 0, "1 control arm(s) reported, 5 verdict file(s) the same, 0 difference(s)"; determinism `f93a961-1026043809-9e04` / `f93a961-10260439` exit 0 with `--expect-verdicts 3`

### FVID · 2026-10-26 · FVID-2026-10-26-report-tiers-runs
- Trigger: ADR-0019 §16.7 allows the full `gold` run (32 arms) to be split in two with `--cells` (the 21 arms that read R2 trees, then the 11 others) when one run would not fit the tier's 4 h wall cap, and the coordinator runs it so on two pods at one commit; `fv-lab report tiers --runs` took one run directory, so the split run could not give one `docs/perf/tiers.md`
- Options: (a) `--runs` takes several run directories of one tier and commit, merged into one run set; (b) merge the two fetched directories into one by hand before rendering (loses which pod, GPU and price measured each arm)
- Decision: (coordinator's request, executed by the agent; the duplicate and commit rules changed by the review, FVID-2026-10-26-report-tiers-runs-review) (a): `--runs` comma separated, as fv-lab's `--cells a,b`, or repeated (clap `value_delimiter = ','`); `report::Runs::new` refuses runs of different tiers, of different commits where known (`pod.json` sha, else the tag's 7-digit head; a prefix agrees), the same directory twice, and an arm that ran (no `skipped`, passed or failed) in more than one run; an arm in several runs shows the copy that ran, or the first in `--runs` order when all were skipped; each row reads its own run's `pod.json` (GPU fallback, `$/clip` price); the header is `` Run set: `<a>`, `<b>` `` plus a list of the runs (tag, arms, end, GPU, price) under the shared commit and last end; one run renders byte for byte as before. `[last_validated]` needs no change: fv-lab writes no gold file (ADR-0023 "What a gold run then does": by hand), and `report tiers` only reads it, its age counted to the last end over all runs
- Reason: the table must stay one command and `--check`-able for the split run, and say which run (pod, price) measured each arm; refusing a twice-run arm keeps one number per variant without a hidden choice between two measurements
- Reversibility: high (code, tests and docs; the single-run output and its goldens are unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16.7; docs/adr/0023-lab-report-and-ci-bridge.md §1
- Verification: `report::tests` (`runs_of_one_tier_and_commit_merge_their_arms`, `runs_that_are_not_one_run_set_are_refused`, `one_run_renders_as_before_and_several_name_every_run`, `run_set_line_round_trips`); `tests/tiers_report.rs` `a_split_run_set_renders_as_one` (the fixture run split in two renders the single run's rows byte for byte; another price in the second run reaches its rows); `tests/fv_lab_bin.rs` `report_tiers_renders_several_runs_as_one` (comma and repeated forms, `--out` then `--check`, refusals exit 1); `fixtures/tiers/tiers.md` and `docs/perf/tiers.md` unchanged

### FVID · 2026-10-26 · FVID-2026-10-26-report-tiers-runs-review
- Trigger: the independent review of #206: a failed arm rerun in another run made the set unrenderable; an empty or short sha, or a run of unknown commit, joined any set; `--runs ""` rendered `Run set: none` and exited 0; validation ages counted to the latest run; `./a` and `a` were two directories; the duplicate sentence and the per-run arm count were imprecise
- Decision: (coordinator, executed by the agent) all fixed in #206: refuse an arm only when more than one copy passed (`exit` 0, not skipped); show the passing copy, else the last failed copy, else the last skipped copy, in `--runs` order, and list each copy not shown under its run (`Also here, shown from another run: …`); `report::known_commit` (pod.json sha, else the tag head, at least 7 hex digits, lowercased), and with several runs a run of unknown commit is refused; `--runs` naming no directory exits 1; a row's validation age counts to its arm's run's end (no arm: the run set's last end); run directories compared by path components; the duplicate sentence only when an arm is in more than one run; the per-run count is the arms shown from it; `docs/dev/fv-lab.md`: run directories must not contain commas
- Reversibility: high (code, tests and docs; the single-run output and its goldens are unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16.7; docs/adr/0023-lab-report-and-ci-bridge.md §1
- Verification: `report::tests` (`runs_of_one_tier_and_commit_merge_their_arms`: failed then passed shows the pass, failed twice the last, skipped twice the last; `runs_that_are_not_one_run_set_are_refused`: two passes, unknown commits, `./a/` and `a`; `commits_are_known_only_as_seven_or_more_hex_digits`; `one_run_renders_as_before_and_several_name_every_run`); `tests/tiers_report.rs` `a_split_run_set_renders_as_one` (per-run ages, a failed arm rerun renders the pass); `tests/fv_lab_bin.rs` `report_tiers_renders_several_runs_as_one` (`--runs ""`, `,` and ` , ` exit 1 and leave the table)

### FVID · 2026-10-26 · FVID-2026-10-26-retire-h3-graphs-experiment
- Trigger: the `h3-graphs` tier's GPU check (ADR-0019 §14.4) passed on `f93a961` (pod `8zx5w6h0766g35`): the check prints `PASS`, the twelve `--off-identity` compares byte-identical (sync = lazy = graphs), the graphs arm replayed 4 of 4 steps on three requests
- Options: keep `h3-graphs-{validate,pod}.sh` as a second runner; delete them as §14.4 says, the goldens script reading the pod script from the history
- Decision: (coordinator, executed by the agent) delete them and `scripts/gpu/experiments/` (its README with them); in the same commit `self-delete.test.sh` drops `h3-graphs-validate.sh` from `LOCAL` and lowers its launcher floor from 3 to 2; `docs/perf/h3-denoise-graphs.md` §5 records the h3-turbo 768p denoise (sync 17.98 s, lazy 17.88 s −0.56 %, graphs 17.93 s −0.30 %; replays 0.19-0.21 % under lazy; graph pool high water 10.34 GiB) against its expectation (lazy ~1.4 %, a replay 0.6-0.7 % more), the verdicts unchanged (lazy default, graphs opt-in)
- Reason: §14.4's pass rule; the tier is the experiment's h3-turbo part word for word (§14.3), and the NVFP4 part stays readable in git until `hydrate`
- Reversibility: high (the scripts are in git at the parent of the deleting commit)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §14, §14.4 "Result"
- Verification: the §14.4 check rerun on the committed run directory (`artifacts/runpod/h3-graphs/check-f93a961.txt`), `self-delete.test.sh`, `fv-lab-experiment-goldens.sh --check` (after the commit: the pod script from the history), `fv-lab-route.test.sh`, `tests/experiment_tiers.rs`

### FVID · 2026-10-26 · FVID-2026-10-26-retire-det-short-determinism
- Trigger: the `det-short` and `determinism` GPU checks (ADR-0019 §15.7) passed on `f93a961`: `fv-lab-compare-runs.sh --expect-verdicts 5` / `--expect-verdicts 3` exit 0 (3 / 2 arms byte-identical, the `h3-timed` control reported, 5 / 3 verdict files the same), both sides `matrix_exit=0`, `fv-lab gate --check` 0 differences on all four runs
- Options: keep the branches as second runners; retire them as `b4a` was (§11.6); for `tests/repeats_oracle.rs`, which ran the live branches: frozen outputs, or a frozen copy of the branches
- Decision: (coordinator, executed by the agent) retire both: in `FV_LAB_RETIRED`, both `case` branches deleted (`determinism`'s `FV_ORACLE_URL` prelude with it), their six fixtures frozen, the five `FV_DET_*` knobs from `knobs::SCRIPT_KNOBS` into `MODE_KNOBS` and `runpod-http.sh`'s `FV_LAB_NOT_READ`, `FV_DET_CELLS` refused with a pointer to `FV_CELLS` (in `FV_EXTRA_ENV` and the host environment), `FV_ORACLE_URL` refused for `determinism` (the tier does not run the oracle first: refused, not dropped); `tests/repeats_oracle.rs` runs a frozen copy (`crates/fastvideo-lab/fixtures/det-script/`: the script at `0f63c56b` cut to its frame, the two branches and the fallback, lines unchanged, and that commit's sourced files), plus a test that its dry runs equal the frozen goldens
- Reason: the retirement rule of §8 / §15.8; a frozen copy keeps the byte-for-byte differential with the fake box (as `tests/verdict.rs` for `b4a`)
- Reversibility: high (the branches are in git history and in the frozen copy)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.1, §15.7 "Results", §15.10
- Verification: `tests/repeats_oracle.rs` (frozen script against fv-lab, and its dry runs equal to the fixtures), `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `tests/knobs.rs`, `knobs::tests`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`, `FV_DRY_RUN=1 runpod-matrix.sh <family>` before and after for the 30 remaining families (26 byte-identical; `identity`, `mmaudio`, `speechtest` only in an error message's line number or temporary path, `writer` in its `dd` timings); evidence `artifacts/runpod/{det-short,determinism}/` with `compare-f93a961.txt` and `gate-f93a961.txt`

### FVID · 2026-10-26 · FVID-2026-10-26-oracle-branch-stays-whole
- Trigger: the `oracle` GPU check (ADR-0019 §15.7, `fasth3-4step-vsa`) passed on `f93a961`: `fv-lab-compare-runs.sh` exit 0, 5 arms the same (2 with byte-identical frames, 3 with the same `compare-dumps` lines), both `matrix_exit=0`, both fetched the 1.3 G reference (`upstream/55fa19a-10260558`, `FV_MIN_RAM_GB=192`); §15.8 left the branch's fate "to be decided with the result"
- Options: (a) retire the branch whole; (b) a reduced branch without the nine moved targets; (c) keep the branch whole, the tier the tensor-diff part's equal
- Decision: (agent, under §15.8; accepted by the coordinator) (c): `oracle` stays out of `FV_LAB_RETIRED`, its fixtures are still written from the live branch, its five knobs stay `SCRIPT_KNOBS` (refused by fv-lab only), `oracle.sh` is unchanged; the evidence is committed (`artifacts/runpod/oracle/f93a961-*`, `compare-f93a961.txt`)
- Reason: (a) loses the clip-metric targets (`ltx25-{i2v,kf,ref2v,a2v*,retake*,extend}`, run by `oracle.sh` on request) and `sfwan13` (the `sfwan` family's prelude calls `runpod-matrix.sh oracle`): a retired family has no branch and the script refuses it; (b) inside `FV_LAB_RETIRED` is refused the same way, and outside it would refuse `oracle.sh`'s default targets on the script path, while `oracle.sh` cannot use the fv-lab route: it sets `FV_POD_FILE` (the runtime pod id and tag it polls for `ORACLE_REFS_DONE`) and `FV_CREATE_WAIT_S`, which `fv_lab_route` refuses because `fv-lab pod up` has neither, and `FV_ORACLE_RUN_MODE=all` and `FV_ORACLE_F32` / `FV_ORACLE_OWN_TEXT` have no fv-lab equivalent
- Open item: before `oracle` can retire, the `fv-lab oracle` driver of §15.6 / §15.7 (upstream pods, the runtime tier with `FV_ORACLE_URL`, release on `ORACLE_REFS_DONE`, a stock wait) and the clip-metric targets and `sfwan13` as cells (the derived-clip step) must exist
- Reversibility: high (nothing deleted)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §15.7 "Results", §15.8, §15.10
- Verification: `tests/repeats_oracle.rs` (the live branch against fv-lab), `fv-lab-goldens.sh --check` (oracle's fixtures from the live branch), `fv-lab-route.test.sh` (`oracle` still on the script path)

### FVID · 2026-10-26 · FVID-2026-10-26-d1-retire-review
- Trigger: the independent review of #207 (nothing blocking): `fv-lab run determinism` with `FV_ORACLE_URL` in its environment dropped it silently, though the docs said it is refused; `runpod-http.sh` read `FV_EXTRA_ENV`'s first line only, so a newline dropped later entries (the `FV_DET_CELLS` / `FV_ORACLE_URL` refusals and `--extra-env` included); with full history the experiment goldens paired the deleted pod script with the live `prompts-hd5.json`, while CI's shallow clone only checks presence; nits
- Decision: (coordinator, executed by the agent) `knobs::check_run_env` also checks `FV_ORACLE_URL` (`why()` scopes the refusal to `determinism`; `oracle` reads it); `fv_lab_route` refuses an `FV_EXTRA_ENV` holding a newline; `fv-lab-experiment-goldens.sh` reads `prompts-hd5.json` from the same commit as the pod script (the deleting commit's parent); `control.yml` drops the dead `scripts/gpu/experiments/**` filters; lazy vs sync written −0.56 % (17.8792 / 17.9808); `docs/perf/determinism.md` names the tiers; the frozen dry-run test also compares `FV_DRY_RUN_GATES` with the first four columns of `<family>-gates.tsv`
- Reason: refused, not dropped (ADR-0019 §10, §15.10); local and CI regeneration must agree
- Reversibility: high
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §14.4 "Result", §15.10
- Verification: `knobs::tests::the_retired_determinism_families_knobs_are_not_read` (`check_run_env` refuses `FV_ORACLE_URL` for `determinism`, not for `oracle` / `det-short`), `fv-lab-route.test.sh` (a newline in `FV_EXTRA_ENV` refused), `fv-lab-experiment-goldens.sh --check` (passes with the live `prompts-hd5.json` edited: the history's copy is read), `tests/repeats_oracle.rs::the_frozen_script_dry_runs_as_the_frozen_goldens`

### FVID · 2026-10-26 · FVID-2026-10-26-gold-causal-progress
- Trigger: the full gold run on `e9b9435` (ADR-0019 §16.7, `artifacts/runpod/gold/e9b9435-1026062720-e2e4`) failed `longlive-1.3b` and `sfwan21-1.3b` on `engine/progress` alone: `{"events":28,"last":[28,28]}` vs `{"steps":4}`, the MP4 and its 81 frames passing
- Options: (a) the engine reports the recipe's 4 steps (or 1..4 per block) for a causal clip; (b) the check expects what the engine reports, every causal block's steps, derived from the variant's recipe
- Decision: (coordinator's request, executed by the agent) (b): `CudaModel::progress_steps(num_frames)` = the recipe's steps x `SfWanRecipe::blocks` (`num_frames.div_ceil(block_frames)`, from the family file) on a causal recipe, else the recipe's steps; `fv-gpucheck engine` checks the event count against it (`limits.events`, `limits.steps` kept as the recipe's per-block steps that `report tiers` compares with the gold `[workload] steps`), assuming the default causal settings as the gold cells run; the engine's `planned` total stays the per-block steps (review of #209: `FASTVIDEO_WAN_CAUSAL_FPB` / `_AR` change the pipeline's block count, not the family file's, and `deliver` already takes the pipeline's total from its first event); no cell or gold file changes
- Reason: a job's `Progress { step, total }` counts its denoise forwards across its stepping stages (`output::deliver`, LTX 8 + 3 = 11), the cudarc SF-Wan rollout counts `index / total` across blocks, and every API surface serves `step / total` as one 0..1 fraction; (a) would end a 28-forward job at 4 / 4 or restart the fraction each block
- Reversibility: high (code and docs)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md §3
- Verification: `cuda::caps::tests::causal_clip_progress_counts_every_block` (both causal variants: `block_frames` = temporal compression x `num_frames_per_block` of the preset's `WanVideoArchConfig`, `blocks` = the pipeline's `latent_frames / num_frames_per_block` over the whole frame grid, `knobs.steps` off, 81 frames 28 events with 4 recipe steps; a bidirectional Wan reports its recipe's steps); the CUDA type-check of `fastvideo-engine-service` and `fastvideo-gpucheck`; GPU: the next gold run's two causal arms

### FVID · 2026-10-26 · FVID-2026-10-26-lab-min-ram
- Trigger: the full gold run at `e9b9435` (RTX PRO 6000, EU-RO-1): the arm `wan14b-plug-4step` died with exit 137 18 s into its load, right after fetching its 84.09 GB of R2 trees, with nothing printed but the device line; fv-lab pods asked for no host-RAM floor (Runpod's default `minRAMPerGPU` 8 GB per GPU) and `box.txt` did not record the pod's RAM limit, so an OOM kill could not be proved (the upstream cells' case, #205)
- Options: a floor in the script only (`FV_MIN_RAM_GB`, which fv-lab routes refuse); a per-tier floor in `lab/tiers.toml` carried by fv-deploy's launch; record the pod's memory only
- Decision: (coordinator, executed by the agent) a per-tier `min_ram_gb` (tiers v1, optional, positive) carried as `GpuSpec::min_ram_gb` into the Runpod pod body's `minRAMPerGPU`, sent only when set, refused at 0 and off a Runpod pod; `Instance::memory_gb` / `vcpus` from the create answer, logged and kept in `pod.json`; the runner records the pod's memory cgroup (`box.txt`: limit and OOM kills; per arm `summary.json`: peak process memory, limit, cumulative peak, OOM kills during the arm; `live.log`: `ram <arm>` and an `OOM <arm>` line), as `upstream/pod.sh` does; the gold tier's floor 192 GB (the 14B plug loader copies the f32 DiT, about 57 GB, and holds it while it copies the f32 text encoder, about 23 GB: about 80 GB of process memory, with more than 100 GB of headroom; the upstream cells' rerun asked for 192 on a PRO 6000 in EU-RO-1 and got 283 GB; the page cache is reclaimable, pages not yet written back the one caveat); no other tier gets one
- Reason: the floor is the pod's, so it belongs to the tier, as the disk does; recording the cgroup turns the next 137 into evidence and lets the floor be lowered from measurements
- Reversibility: high (drop `min_ram_gb` from the tier; the field and the records are additive)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §3 (the tier table), §16.6, §16.10
- Verification: `plan::tests::tiers_take_an_optional_positive_host_ram_floor`, `pod::tests::a_tiers_host_ram_floor_is_the_pods_min_ram_per_gpu`, `tests/payload_golden.rs` (gold's body carries `minRAMPerGPU: 192`, equal to `FV_MIN_RAM_GB=192 runpod-http.sh plan`; every other tier's unchanged), fv-deploy `provider::tests::{a_host_ram_floor_is_min_ram_per_gpu_and_only_when_set, a_host_ram_floor_of_zero_or_off_a_runpod_pod_is_refused, the_pods_ram_and_vcpus_are_read_from_the_create_answer_and_logged}`, `tests/payload_goldens.rs` (unchanged fixtures), `podmem::tests`, `tests/runner.rs::{the_pods_ram_limit_and_an_arms_oom_kill_are_recorded, without_a_memory_cgroup_the_summary_is_unchanged}`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`, `control/test/unit/launch-goldens.test.ts` (a zero floor refused by zod: `minimum: 1`), `podmem::tests::a_dropped_sampler_stops_its_thread`, `pod::tests::read_records_skips_what_it_cannot_read` (review of #210); UNVERIFIED on a GPU pod until the next gold run

### FVID · 2026-10-26 · FVID-2026-10-26-gold-run-7e580e5
- Trigger: the full gold run at `7e580e5` (ADR-0019 §16.7), split in two pods with the 192 GB floor: `artifacts/runpod/gold/7e580e5-1026090916-e804` (the 21 R2 arms, 18 trees, 530.74 GB, all exit 0) and `7e580e5-1026091001-7421` (the 11 volume arms, 7 exit 0, 4 exit 2 "does not support task t2v"); both `DONE`, `matrix_exit=0`, `oom_kills` 0 for every arm; `wan14b-plug-4step` peaked at 156,004 MiB of host process memory
- Options: for the gold files of variants whose `[latency]` already held a measurement of another kind (`gen`, `end_to_end`, `stream`): (a) replace it with the run's serve job; (b) keep it and cite the run in a note; for `load_s`: close it from any arm, or only from an arm whose model trees no earlier arm of the pod had read; for the e9b9435 attempt: commit it as findings evidence, or describe it only
- Decision: (coordinator's request, executed by the agent) commit both `7e580e5` run directories as the run set and render `docs/perf/tiers.md` from them; per measured variant (28) `[last_validated]` = `7e580e5`, the arm's `summary.json`, 2026-10-26, and the arm's `gpucheck-out/engine.json` as a `[[source]]`; the 10 files with no latency measurement take the run's job (`measures = "serve_job"`, `min_s` / `max_s` gaps), `load_s` only where no earlier arm of the pod had read the arm's model trees (3 of 10; in the other 7 a gap: page-cache warm, the value in the source's note); for the other 18, (b) with no gap closed (`load_s` and `peak_gib` are of the process that measured; one kind of measurement per `[latency]` table), the run's numbers in the source's note; `[last_validated]` means the variant ran to exit 0 at this commit on its gold cell, not that every number already in the file was re-confirmed; the 4 refusals not validated, their files unchanged; the `e9b9435` runs described in §16.7 and not committed
- Reason: ADR-0023 "What a gold run then does" and ADR-0009 (gaps closed only for fields the run measured; a number from another setting or measurement is cited in a note, not copied); a warm load is not the variant's load (review of #211); a serve job (one request, the uncached prompt's text encode included, the served default) is not a CLI median with a cached prompt, so one `[latency]` table keeps one kind of measurement; failed attempts of earlier gates were not committed (§15.7)
- Open items: the 192 GB floor is 183,105 MiB, about 26.5 GiB above the 14B plug's sampled peak, and pages not yet written back count against the same cgroup (§16.10), so the floor may not hold on a pod given exactly 192 GB: raising the gold tier's `min_ram_gb` (or cutting the Wan plug loader's host copies) is a prerequisite of the next gold run (plan D3, follow-up a); `negotiate` before load for the ref2v / a2v variants (plan D1, follow-up 4)
- Reversibility: high (evidence and documents; the gold files' former numbers stay in their cited sources)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16.7 "Result", §16.10; docs/adr/0023-lab-report-and-ci-bridge.md "What a gold run then does"
- Verification: `fv-lab report tiers --check`, `fv-lab gold check` (28 validated, 0 errors), `fv-lab gold cells --check` (fv-lab 0.1.68 built at `7e580e5`), every cited path, JSON key and run record present (the checks of `tests/gold_files.rs`), `fv-lab-goldens.sh`, `fv-lab-experiment-goldens.sh`, `fv-lab-report-goldens.sh`, `fv-lab-restored-goldens.sh` with `--check`, `scripts/gpu/lint.sh`

### FVID · 2026-10-26 · FVID-2026-10-26-gold-min-ram-256
- Trigger: the gold run at `7e580e5` (ADR-0019 §16.7 "Result"): `wan14b-plug-4step` peaked at 156,004 MiB of process memory against the gold tier's 192 GB floor, which is 183,105 MiB (Runpod's GB are 10^9 bytes), with pages of its 84.09 GB fetch not yet written back charged to the same cgroup; §16.7 made raising the floor (or cutting the loader's host copies) a prerequisite of the next gold run (plan D3, follow-up a)
- Options: 256 GB; 283 GB (what Runpod gave both pods); keep 192 and cut the Wan plug loader's host copies (store an f32 `RawTensor` once; upload an eager DiT map before the text encoder)
- Decision: (coordinator's request, executed by the agent) `min_ram_gb = 256` for the gold tier: 244,140 MiB, 88,136 MiB above the measured peak, more than the whole fetch (80,195 MiB) even if none of it were written back; the loader is unchanged here, its two cuts described (ADR-0019 §16.10 "Raised to 256") and left to their own change with a GPU byte-identity check
- Reason: the peak is about twice item 5's estimate because `fastvideo_loader::RawTensor` holds an f32 tensor as both bytes and an f32 vector (the merged DiT about 2 x 57 GB, the UMT5 map about 2 x 23 GB at once); a floor below 283 keeps the hosts the run landed on; each loader cut touches code every model or the MoE plan shares
- Reversibility: high (one tier value and its golden)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §16.10 "Raised to 256"
- Verification: `tests/payload_golden.rs` (gold's body `minRAMPerGPU: 256`, equal to `FV_MIN_RAM_GB=256 runpod-http.sh plan`), `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`; UNVERIFIED: how many EU-RO-1 PRO 6000 hosts give 256 GB (both gold pods got 283 GB)

### FVID · 2026-10-26 · FVID-2026-10-26-negotiate-before-load
- Trigger: the gold run at `7e580e5` (ADR-0019 §16.7 "Result"): `h3-ref2v-max`, `h3-ref2v-turbo`, `ltx25-ref2v` and `ltx25-a2v-guided` loaded their model for 199, 132, 34 and 5 s before `fv-gpucheck engine` negotiated the t2v request and `negotiate` refused it (plan D1 follow-up 4, D3 follow-up b)
- Options: negotiate against `CudaModel::caps()` from the catalog before the backend is built; negotiate against the started engine's table before waiting for readiness (the executor is already loading); build the table `start` serves without starting it, negotiate, then start
- Decision: (coordinator's request, executed by the agent) the third: `EngineService::caps_table` builds the table from the backends' declarations by the code `start` now uses (`declare` / `into_table`), and `EngineService::start_negotiated` negotiates against it and starts only an accepted request (`StartNegotiatedError::{UnknownModel, Negotiate, Start}`, the `negotiate: …` text unchanged); `fv-gpucheck engine` calls it after `CudaBackend::new` (device checks and the H3 1080P gate, no model load). fv-serve is unchanged: it loads its resident models at start whatever the requests, and its submit path negotiates before `engine.submit` (so before a swap load); while it loads, `admit` answers `loading` before negotiating, which costs no load
- Reason: the catalog's `CudaModel::caps()` is not the served table (the table rebinds tiers across models and `CudaBackend::new` may drop the 1080P tier on the device), so only the table `start` builds keeps every acceptance and refusal as it was
- Reversibility: high (code and tests)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0023-lab-report-and-ci-bridge.md (the gold tier's refusals)
- Verification: `crates/fastvideo-engine-service/tests/start_negotiated.rs` (the four gold refusals from the real catalog with a backend that counts `load` calls: 0; an unknown model refused with 0 loads; an accepted request starts and loads, its caps and job equal to `negotiate` against the started engine's table; `caps_table` equal to the started engine's table and public aliases for the whole catalog); `cargo check -p fastvideo-gpucheck --features cuda`; UNVERIFIED on a GPU until the next gold run

### FVID · 2026-10-26 · FVID-2026-10-26-h3-vae-shared-files
- Trigger: plan D1 follow-up 6: `completed_vae_files` (crates/fastvideo-cudarc/src/h3/pipeline.rs) filled any shard a tree's `vae/` index names but the tree lacks from h3-base's `vae/`, while only the NVFP4 trees' shard 3 is known byte-identical (`verify-weights.sh` `shared_file`, #208)
- Options: one table in the family manifest (a format change); a Rust constant held to `verify-weights.sh`'s table by a test; match by file name only, whatever the tree
- Decision: (coordinator's request, executed by the agent) `h3::pipeline::SHARED_FILES` (file, copy, bytes, SHA-256; the two NVFP4 trees' `vae/diffusion_pytorch_model-00003-of-00003.safetensors` from `h3-base/vae/`), keyed by the tree directory's name as the script is; the engine fills only a listed file, only from a `base` whose tree is the listed copy's, only when the copy has the listed length (398,539,336 bytes); any other missing shard is an error naming the table; the SHA-256 stays the script's check (no 0.4 GB hash at every load); `verify-weights.sh` stays the source the test and gen-configs.mjs read
- Reason: the script is what the lab runs before every arm and gen-configs.mjs can read it, so a format change for one row is not needed; a file of the same name in another revision of h3-base would decode wrong silently
- Reversibility: high (code and tests)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/ops/r2-weights.md §15
- Verification: `h3::pipeline::tests::{only_the_listed_shared_vae_shard_is_read_from_h3_base, shared_files_mirror_verify_weights_shared_file, the_nvfp4_trees_vae_is_its_default_decoder_completed_from_h3_base}`, `verify-weights.test.sh`

### FVID · 2026-10-26 · FVID-2026-10-26-gen-configs-shared-file-check
- Trigger: plan D1 follow-up 5: the NVFP4 trees leave out vae shard 3 (`verify-weights.sh` `shared_file`, #208), which the engine reads from its text weights' `vae/`; nothing checked that a variant reading such a tree names that tree as its `text_weights` and in its `trees`
- Options: a hand list of the NVFP4 variants in gen-configs.mjs; read `verify-weights.sh`'s `shared_file` table and check every variant whose trees include a tree it names
- Decision: (coordinator's request, executed by the agent) the second: `sharedFiles()` reads the table (plain `NAME=value` assignments, the `"<pattern>" | …)` and `echo "<copy> <bytes> <sha256>" ;;` arms; refuses a script without the function or a malformed arm), `sharedFileChecks()` requires the copy's tree in `trees` and `engine.text_weights = "${weights}/<copy tree>"`; part of `crossChecks`, so `node gen-configs.mjs --check` and `families-gen.test.ts` run it; `parseFamily` carries `text_weights` (not written to any generated file)
- Reason: one table (the script) for the lab gate, the engine's mirror test and this check; a new tree with a shared file is checked without editing gen-configs.mjs
- Reversibility: high (a check)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/ops/r2-weights.md §15
- Verification: `control/test/unit/families-gen.test.ts` (the table's two rows; the four NVFP4 variants pass; another `text_weights`, none, or h3-base missing from `trees` refused naming the variant and the copy; a script without the table or with a bad arm throws), `node gen-configs.mjs --check`, vitest unit, tsc

### FVID · 2026-10-26 · FVID-2026-10-26-gold-followups-review
- Trigger: the independent review of #213 (nothing blocking): both `shared_file` table parsers (the Rust mirror test, gen-configs.mjs `sharedFiles`) dropped an arm written in another form while another arm parsed; an unknown variable stayed in an expanded field; `caps_table_is_the_started_engines_table` compared models and aliases, not the whole table; an empty backend list gave `UnknownModel`
- Decision: (coordinator, executed by the agent) both parsers strict: only `case "$1" in`, quoted arms each followed by one `echo "<copy> <bytes> <sha256>" ;;`, `*) return 1 ;;` and `esac`; an echo with no arm, an arm with no echo, any other line, or a field still holding `$` is an error (tests: an unquoted pattern, a one-line arm and five more forms in both); the table compared whole (`CapabilityTable: PartialEq`); `start_negotiated` refuses no backend with `start`'s error; the plan's D1 follow-ups 4-6 and D3 (a)/(b) marked done, the two loader cuts left open as future PRs; `--text-weights` help and r2-weights.md §15 say the NVFP4 and h3-base tree directories keep their R2 names
- Reversibility: high (tests and docs)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- Verification: `h3::pipeline::tests::the_shared_file_parser_refuses_unexpected_forms`, `families-gen.test.ts` "sharedFiles is strict", `start_negotiated.rs` (`caps_table_is_the_started_engines_table`, `no_backend_is_the_starts_error`)

### FVID · 2026-10-26 · FVID-2026-10-26-d1f-tiers
- Trigger: the coordinator's D1f slice: move `b4b`, `b4c`, `b4c2`, `fuse`, `attn3` and `ltx` as fv-lab tiers without retiring a branch, and check §9.1's "h3: retired (rtx6000 covers it)"
- Options: per family, a tier with no schema change, an additive schema field, or stop (a large addition) and say why; for `ltx`, whose ungated `run_cell`s read weights: lift the "an ungated arm reads no tree" rule (§15.2) or gate the script's arms in lockstep; for `h3`: a tier, retirement, or retirement with a new `rtx6000` arm for what it alone ran
- Decision: (agent, proposed for the coordinator's review) `fuse` (3 cells, 10 arms), `attn3` (1 cell, 6 arms) and `ltx` (1 cell, 3 arms) are tiers with goldens from the live branches and no new cell field: `fuse`'s kernel check is §15's ungated arm, and `Workload::model` (never read by the runner) is documented as descriptive, naming both pipelines of a cell whose compares follow arms of each (`h3 ltx2`); the script's `ltx` branch gates its arms (`gated_cell`, a real-run change: on `bz0ofkfx45` its LTX-2.3 and LTX-2.0 arms are skipped instead of failing on the missing trees); `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` refused by fv-lab (`SCRIPT_KNOBS`); `b4b`, `b4c`, `b4c2` stop: each verdict reads what no `[verdict]` list types (`benchmark.json` field rules, cache-directory listings under `$SCRATCH` with per-group fresh dirs, CUPTI kernel tables, other reason orders), §11's size or more, and their gates passed when their changes merged; `h3`: retirement with no tier proposed (every arm covered by `rtx6000` but the refiner-less Spark fallback, which nothing serves), not executed
- Reason: §4 promises a skip, not a crash, for a tree that is not there, and an unverified `{weights}` read would be the first; a field added only to say "two pipelines" is a field for a comment; the B4 verdicts are bespoke and would double the slice for families with no pending gate
- Reversibility: high (cells and fixtures; the script's `ltx` change is three words)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §17 (§9.1 rows)
- Verification: `tests/dry_run.rs` (the three tiers' dry runs and gates equal the script's), `tests/payload_golden.rs`, `tests/lab_files.rs` (`ltx23`, `ltx2` the new trees the volume lacks), `knobs::tests`, `tests/schema.rs`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`; GPU: §17.5 (the coordinator's; ~$4.20)
- Open: §17.3 (port the B4 verdicts as one ordered `[[verdict.check]]` list, or retire the three as finished gates) and §17.4 (`h3`'s retirement, with or without a `sol-h3-spark-no-refiner` arm), the coordinator's

### FVID · 2026-10-26 · FVID-2026-10-26-retire-fuse-attn3-ltx
- Trigger: the `fuse`, `attn3` and `ltx` GPU checks (ADR-0019 §17.5) passed on `762ce4d`: `fv-lab-compare-runs.sh` exit 0 with §17.5's expected lines (5 / 2 / 3 arms the same; 2 / 2 / 1 ran with byte-identical frames, 2 / 0 / 2 skipped in both; 4 / 3 / 0 reports the same; 0 differences), every side `matrix_exit=0`, the six pods deleted; `kernels-fuse` 90 `[PASS]`, the `fuse` VSA off / on and the `attn3` b160 / b161 compares `pass` with off-identity `ok`, `ltx25-two-stage` 242 frames byte-identical
- Options: keep the branches as second runners; retire the three as §17.6 says (as `det-short` and `determinism` in §15.10)
- Decision: (coordinator, executed by the agent) retire all three: in `FV_LAB_RETIRED`, their `case` branches deleted, their nine fixtures frozen, `FV_ATTN3_ARMS` / `FV_ATTN3_VSA` from `knobs::SCRIPT_KNOBS` into `MODE_KNOBS` and `runpod-http.sh`'s `FV_LAB_NOT_READ`, refused in `FV_EXTRA_ENV` (pointing at `FV_CELLS`) and in the host environment; no frozen copy of the script (no crate test ran these branches live); `h3` (§17.4) and `b4b` / `b4c` / `b4c2` (§17.3) unchanged; `runpod.sh gpu` / `matrix` (coordinator: fix in this PR) run a retired family's tier with the image's fv-lab instead of `runpod-matrix.sh ltx`, and stop before any pod on a family nothing can run. The `fuse` VSA gate's `fail` on both sides (promotion speedup 1.08× under the 1.10 bar, no LPIPS; its off-identity check passed) is the fusion knob's promotion decision, not a parity or correctness verdict and not a finding against the retirement (accepted by the coordinator)
- Reason: §8 / §17.6's retirement rule; every arm that ran is byte-identical across the runners and the skips agree
- Reversibility: high (the branches are in git history at the parent of the deleting commit; the fixtures are what they wrote)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §9.1, §17.5 "Results", §17.6
- Verification: `knobs::tests::the_retired_attn3_knobs_are_not_read`, `tests/knobs.rs`, `tests/dry_run.rs::retired_families_are_frozen_tiers_without_a_branch`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`, `runpod-plan.test.sh`, `FV_DRY_RUN=1 runpod-matrix.sh <family>` before and after for the 27 remaining families (26 byte-identical, `writer` in its `dd` timings); evidence `artifacts/runpod/{fuse,attn3,ltx}/762ce4d-*` with `compare-762ce4d.txt`

### FVID · 2026-10-26 · FVID-2026-10-26-d1g-retire
- Trigger: D1g: the `runpod-matrix.sh` families that get no fv-lab tier: `b4b` / `b4c` / `b4c2` (ADR-0019 §17.3), `h3` (§17.4), and §14.7's "retire" rows `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`, `identity`, `h3attn`
- Options: B4: (a) a typed `[[verdict.check]]` slice or (b) retire as finished experiments; `h3`: retire with or without a `sol-h3-spark-no-refiner` arm in `rtx6000`; the §14.7 rows: tiers or retired; mechanics: add them to `FV_LAB_RETIRED` (which routes to an fv-lab tier) or a separate "retired with no tier" list
- Decision: (coordinator decision 2026-10-26, executed by the agent) B4 option (b); `h3` retired with no tier and no `sol-h3-spark-no-refiner` arm; `hunyuan`, `sana-video`, `b200`, `writer`, `precision-debug`, `identity` retired with no tier; `h3attn` retired by §14.7's rule (no attention arm of it is an open item: `docs/techniques.md`'s one forward item, INT8 `Q K^T`, closed with SageAttention, Phase 4). Mechanics (agent): a new list `FV_SCRIPT_RETIRED` in `scripts/gpu/fv-lab-retired.sh` with `fv_script_retired_why` (the pointer per family); `runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan` and `runpod.sh gpu` / `matrix` / `plan` / `b200` refuse them (exit 2), never route to fv-lab; the eleven branches deleted (907 lines); `b4-verdict.sh` and `tests/b4-verdict.test.sh` deleted (`tests/verdict.rs` and `tests/repeats_oracle.rs` read their own frozen copies in `fixtures/b4a-script/` and `fixtures/det-script/`); `runpod.sh` lists `gpu` = `ltx`, `matrix` = `ltx wan`, `cmd_b200` deleted; no knob moved (none was in `SCRIPT_KNOBS`); no fixture (none was a tier)
- Reason: their GPU evidence is committed (`artifacts/runpod/{b4b,b4c,b4c2,identity}/`) or recorded (`docs/techniques.md`), another tier covers them (`rtx6000`, `precision`, `det-short` / `determinism`), or what they ran on is gone (the US volume, trees not on the EU volume); `FV_LAB_RETIRED` would route them to tiers that do not exist
- Reversibility: high (the branches are in `git show b019ccb2:scripts/gpu/runpod-matrix.sh`; a family returns as a tier when wanted)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §18 (§9.1, §14.6, §14.7, §17.3, §17.4)
- Verification: `fv-lab-route.test.sh`, `runpod-plan.test.sh`, `fv-lab-goldens.sh --check`, `tests/dry_run.rs::families_retired_with_no_tier_have_no_branch_and_no_tier`, `tests/repeats_oracle.rs`; `FV_DRY_RUN=1` output and `FV_DRY_RUN_GATES` of the 16 remaining families before (`b019ccb2`) and after: byte-identical
- Open: `writer`'s measurement has no result in `docs/perf` and no committed run (only commit `a8ecfec6`'s description)

### FVID · 2026-10-26 · FVID-2026-10-26-fv-lab-oracle-driver
- Trigger: the coordinator's D1h: the `fv-lab oracle` driver of ADR-0019 §15.6 / §15.10 (what `oracle.sh` and §15.7's hand-run commands do) and the remaining oracle targets as cells
- Decision: (agent, for the coordinator's review) `fv-lab oracle` brings up the upstream reference pods as fv-lab pods on fv-deploy's guards from a new `lab/upstream.toml` (its own schema v1, additive: one `[[stack]]` per upstream image with its targets' weight steps; not tiers, which run cells), waits for every selected reference before the runtime pod, runs the runtime side as `fv-lab pod up --wait`, and releases the upstream pods on `ORACLE_REFS_DONE` (or 1,800 s after the runtime pod is ready without it), a 2,100 s stock wait or any failure (review of #216: a pod is never left behind by a failed step after its create, the runtime side is stopped, the upstream pods released, then the runtime side waited for); the clip-metric targets and `sfwan13` are not moved (they need a derived-clip step the ADR has not designed: a decision, §19.9), so `oracle`'s branch stays whole
- Reason: the ADR names the driver's shape (a second image and start command, the release on `ORACLE_REFS_DONE`); the stacks are data like the tiers; the runtime side is reused, not copied
- Reversibility: high (a new subcommand, file and schema; `oracle.sh`, the branch and every existing pod body unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §19
- Verification: `tests/oracle_driver.rs` (payload goldens from `runpod-http.sh plan-upstream`, the plan's refusals, the balance guard, the driver's order and releases on fakes), `tests/fv_lab_bin.rs::oracle_without_yes_is_the_plan_and_needs_no_key`, `tests/schema.rs`, `fv-lab-goldens.sh --check`; GPU check §19.7 pending (the coordinator's)

### FVID · 2026-10-26 · FVID-2026-10-26-d1i-oracle-metrics
- Trigger: ADR-0019 §19.9: the oracle's clip-metric targets and `sfwan13` need a derived-clip step the cell schema lacks (the coordinator's D1i)
- Options: where the step lives: (a) a typed `[[metric]]` list in the cell, or (b) an ungated arm running a repository script; what `metrics.txt` is judged by: a threshold (a verdict) or nothing (a report); `sfwan13`: the `oracle` tier moved to `volume+r2`, or a sibling tier
- Decision: (coordinator decision 2026-10-26, executed by the agent) (a), additive, the script's exact ffmpeg / ffprobe argv, listed by the dry run and held by goldens, not oracle-only (`hd` later); `metrics.txt` a report, no verdict; `sfwan13` a sibling tier `oracle-r2` with `weights = "volume+r2"` if its upstream R2 fetch is small, else a follow-up. As executed: a metric is a pass over clips with four kinds of measure (`probe`, `clip-vs-clip`, `kept-frames-vs-source`, `frame-vs-image`), since the script interleaves measures per clip; the ten LTX targets are the `oracle` tier's new `extra_cells` (tiers v1, additive), run only when `--cells` names their arms, so the default run and its goldens are unchanged; `sfwan13` is the follow-up (its upstream R2 fetch needs a schema step, the R2 key on the upstream pod, `pod.sh` and the upstream `oracle.sh`; its runtime cell needs a `decode` measure and a compare of frame directories)
- Reason: §19.9; decision 2 keeps the script's meaning; the extra cells keep the GPU-checked default run byte-identical
- Reversibility: high (additive fields; the script branch, `oracle.sh` and every pod body unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §20
- Review (2026-10-26, coordinator): the compare rule (values alone differing reported, not a failure) accepted, tightened to ffmpeg SSIM / PSNR values with probe lines byte for byte and the run tag normalised; the GPU check waits for a balance top-up and the merge
- Verification: `tests/dry_run.rs::the_extra_cells_dry_run_as_the_script_with_their_targets` (145 golden lines, 87 metric commands), `tests/repeats_oracle.rs::clip_metrics_report_as_the_script`, `plan::tests::metrics_*`, `plan::tests::extra_cells_run_on_request`, `tests/oracle_driver.rs::the_plan_takes_the_extra_cells_by_arm`, `tests/schema.rs`, `fv-lab-goldens.sh --check` (the existing `oracle` fixtures unchanged); GPU check §20.7 pending (the coordinator's)

### FVID · 2026-10-26 · FVID-2026-10-26-oracle-served-means-dump
- Trigger: the coordinator's GPU checks at `664473b`: ADR-0019 §19.7 passed; §20.7 compared nothing (the `sol-ltx25` upstream pod had no Hub token for the gated `Lightricks/LTX-2.5`, both cells wrote `ORACLE_DONE` without a dump, and the driver counted that as served)
- Decision: (coordinator's task, executed by the agent) a reference is served only when `ORACLE_DONE` and `oracle-dump.tar` exist (the directory listing); `ORACLE_DONE` alone is missing at once with the cell's `cell.json` exit, and with nothing served the upstream pods are released without a runtime pod and the run fails; `oracle.sh`'s `ORACLE_DONE` unchanged. The upstream pod's Hub token is the owner's Runpod secret `fv_hf_token`, passed as `HF_TOKEN = {{ RUNPOD_SECRET_fv_hf_token }}`, never a value: `hub_token = true` per stack in `lab/upstream.toml` (additive, only `sol-ltx25`) and `runpod-http.sh upstream` for `weights:ltx25*` steps (`UP_HUB_TOKEN` forces it); `upstream/pod.sh` treats a blank or unfilled reference as no token
- Reason: a run that "passed" on references that did not exist; the token must not be in the repository or on the volume, and fv-lab already passes the R2 key by secret reference
- Reversibility: high (additive field; the `fastvideo` stack's body and golden unchanged; `sol-ltx25`'s golden gains the reference)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §19.1, §19.2, §19.3, §19.7 "Results", §20.7 "First attempt"
- Verification: `tests/oracle_driver.rs::{oracle_done_without_a_dump_is_missing_and_no_runtime_pod_starts, a_reference_without_its_dump_drops_its_arms_and_fails_the_run, the_upstream_pod_body_equals_runpod_http_sh_plan_upstream_but_the_cap_watchdog, the_stacks_file_refuses_what_it_cannot_run}`, `tests/oracle_live.rs::references_and_the_upstream_run_through_the_proxy`, `tests/schema.rs`, `fv-lab-goldens.sh --check`; seven mutations each caught (ADR §19.5)
- Open: the owner creates the Runpod secret `fv_hf_token` (a Hub token that accepted the LTX-2.5 licence, docs/ops/runpod-secrets.md); what Runpod does with a reference to a secret that does not exist is UNVERIFIED (create refused or the value empty / left as text; both handled); then §20.7 is re-run
- Review (2026-10-26, coordinator, #218): nothing blocking; fixed: a finished pod's pending references asked 3 more times before they fail, `pod.sh`'s `hub_token_init` tested in CI (`hub-token.test.sh`), secret-like env names and a `weights:ltx25*` step without `hub_token` refused, the lab logs and ledger lines committed. **ADR-0019 §19 (the `fv-lab oracle` driver) accepted** (coordinator decision 2026-10-26): its GPU check passed on `664473b` (§19.7 "Results")

### FVID · 2026-10-26 · FVID-2026-10-26-d1j-sfwan13
- Trigger: ADR-0019 §20.5 (`sfwan13`, the last `runpod-matrix.sh oracle` target that was not a cell; its tree `sfwan21-1.3b` R2-only since 2026-10-13)
- Decision: (coordinator's decisions, executed by the agent) the upstream tree by a typed per-target `r2` in `lab/upstream.toml` (a `pod.sh` step `r2:<tree>`: `r2-weights.py plan --pin`, `r2-fetch.py --root $UW`; the R2 key as Runpod secret references only when a selected target has one, handed to the plan step alone) and a per-target `container_disk_gb` (the `fastvideo` body unchanged without `sfwan13`); the runtime tier `oracle-r2` (`weights = "volume+r2"`, cell `oracle-sfwan13`, a sibling of the `oracle` family whose goldens are the live branch with `FV_ORACLE_TARGETS=sfwan13`); additive cell schema: a `decode` measure, `report` optional for decode-only passes, `{ref_dir}`, `[[compare]] tag` (two frame directories), `after` anchors; the script's `sfwan13` dry-run fixes first; the `sfwan` prelude unchanged
- Reason: §20.1 decision 3 (a sibling tier keeps `oracle`'s pod body and fixtures); generic schema pieces `hd` can reuse
- Reversibility: high (additive schema and tier; every existing fixture and pod body unchanged)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), PR #219
- ADR: docs/adr/0019-fv-lab.md §21
- Verification: `tests/dry_run.rs::the_oracle_r2_tier_dry_runs_as_the_oracle_family_with_sfwan13`, `tests/repeats_oracle.rs::sfwan13_decodes_compares_and_cleans_up_as_the_script`, `tests/oracle_driver.rs::the_plan_of_sfwan13_fetches_its_tree_upstream_and_runs_on_oracle_r2`, `scripts/gpu/tests/upstream-r2.test.sh`, `fv-lab-goldens.sh --check`; the 15 other families' dry runs byte-identical before and after; GPU check §21.7 pending (the coordinator's)
- Result (2026-10-26, coordinator): §21.7's GPU check passed on `4b4980b` on every criterion (served `sfwan13`, runtime exit 0 on both runners, compare exit 0: 10 arms the same, 4 with 81 frames byte-identical, 3 `compare-clips` reports `pass` on both sides; no key or presigned URL in any fetched file; about $0.86; ADR-0019 §21.7 "Results", PR `claude/oracle-r2-check-4b4980b`); §21 accepted; `oracle` retires only after every cell family has passed its check, so it still waits for §20.7's re-run (blocked on the owner's Runpod secret `fv_hf_token`, §21.9)

### FVID · 2026-10-26 · FVID-2026-10-26-script-r2-trees
- Trigger: ADR-0019 §21.8: the script runner of §21.7's GPU check mounts the EU volume as `$W`, which lacks `sfwan21-1.3b`, and had no R2 key, so its `sfwan13` arms would skip and the two-runner compare would compare nothing
- Options: A `FV_R2_TREES` on `runpod-http.sh run` (fv-lab's merged root on the script runner's pod); B a pod script fed a host-minted plan (bearer URLs in the pod body); C fv-lab's side alone (no runner equality)
- Decision: A (coordinator, 2026-10-26). `FV_R2_TREES` (run / plan only): trees checked before the key against `weights-r2.tsv` and the EU volume's trees; the container disk refused (not raised) below the trees and 120 GB; idle limits default to `weights::r2_idle_min`; the body's `env` the `weights::r2_env_refs` references; the start command unsets the key on its first line and hands it to `scripts/gpu/r2-stage.sh` alone, which links fv-lab's link roots and fetches into `$SCRATCH/weights` (the plan 700 outside the served runs, removed whatever happened, never on the volume); the matrix runs with `FV_WORK=$SCRATCH`, so its command lines are fv-lab's word for word. `scripts/gpu/oracle.sh` passes it for `sfwan13`
- Reason: §8's equality needs both runners on the same tree; no value of the key in the repository, a body, an argv or a served file
- Reversibility: high (unset: every pod body and golden byte-identical)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), PR #219
- ADR: docs/adr/0019-fv-lab.md §21.8, §21.7
- Verification: `scripts/gpu/tests/runpod-r2.test.sh` (refusals, unset = the `oracle` golden, no key leakage, `r2-stage.sh` on fakes; seven mutations caught), `tests/payload_golden.rs::the_script_side_r2_pod_reads_its_trees_as_the_oracle_r2_tier`, golden `runpod-http-oracle-sfwan13-r2.json`, `fv-lab-goldens.sh --check`

### FVID · 2026-10-26 · FVID-2026-10-26-d1-remaining
- Trigger: ADR-0019 §22 (#221) proposed what happens to the fifteen `runpod-matrix.sh` families left besides `oracle`. An independent review of #221 checked every claim: the dry-run counts, tree sizes, the `{tae}` refusal and six families held, and it corrected `mmaudio`, `hd`, `serve-engine`, `cold`, `eval`, `ltxoffload`, the sol families and the costs
- Options: per family, (a) a tier now, (b) a tier after a schema addition, (c) retire with no tier, (d) an owner decision. For the TAE paths: `{tae}` without a TAE gate, a cell-level `export`, or `{weights}/auxiliary/tae`
- Decision (coordinator, 2026-10-26):
  - (c), retire with no tier: `sfquality`, `headline`, `speechtest`, `cold`, `eval`, `ltxoffload`, `solbench`, `sol-lingbot`, `sol-cosmos3`. `sfwan` retires with `oracle`. `sol-bench-pod.sh` stays as it is.
  - (a), tier now: `hd`, with the Lanczos / post-script / keyframe step dropped, its orphans updated, and a GPU check including `max-1080p`; `serve-engine`, with `--tae-dir` dropped in lockstep.
  - (b), tier after §22.4 (`{tae}` without a TAE gate, with per-arm `FASTVIDEO_TAE_DIR` `env` entries): `wan`, `sfstream`, and a small `mmaudio` tier (`wan13-audio`, `sfwan13-audio`, and `ours-v2a` if it fits; no upstream parity).
  - `wan`'s two oracle cells are in its GPU check only if the balance allows it at run time.
  - Slice order: S1 the nine retirements, with stale docs updated; S2 `hd`; S3 `serve-engine`; S4 §22.4 + `sfstream`; S5 `mmaudio`; S6 `wan`; S7 delete the branches of S2-S6 after their checks; S8 `sfwan` with `oracle` (blocked on the owner's `fv_hf_token`).
- Reason:
  - The retired families are finished measurements, have no recorded result, or are covered by another tier or script. What each loses is stated in §22.3.
  - `hd`, `serve-engine`, `wan`, `sfstream` and `mmaudio` are the only pod checks of a served or user-reachable path: the 1080P `sol-h3` canvas and its time and memory; engine-vs-CLI byte identity; the Wan kernels and parity; `wan stream` parity; the `--audio mmaudio` sidecar.
  - `{tae}` without a gate is the smallest addition that keeps the dry runs, and so the goldens, holding the TAE paths.
- Reversibility: high. Nothing is done yet; each slice is its own PR, and a branch leaves the script only after its own GPU check (§8).
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ), PR #221 (docs only)
- ADR: docs/adr/0019-fv-lab.md §22
- Verification: the review of #221. GPU checks in S2-S6, about $10.2 in all, or about $12.8 with `wan`'s oracle cells (§22.5); all estimates.
- Open (owner):
  - whether the `mmaudio` knob stays offered (catalog `FASTVIDEO_WAN_AUDIO`, the CLI's `--audio mmaudio`, GCP staging);
  - whether `sol-bench-pod.sh` moves to fv-lab;
  - the total spend, within the standing "provision as needed" permission.
- Result (2026-10-26, S1, draft PR #222): the nine (c) families retired with no tier: in `FV_SCRIPT_RETIRED` with pointers naming what is lost, refused (exit 2) by `runpod-matrix.sh`, `runpod-http.sh run` / `all` / `plan` and `runpod.sh gpu` / `matrix` / `plan`, their branches deleted (495 lines, not §22.6's 567; plus `eval`'s defaults), 12 dry runs of the remaining families byte-identical before and after, no knob moved, stale docs updated; tests and mutations in ADR-0019 §22.8; CPU only

### FVID · 2026-10-26 · FVID-2026-10-26-d1-s2-hd
- Trigger: the coordinator's slice S2 of FVID-2026-10-26-d1-remaining (ADR-0019 §22.6 item 2): `hd` as an fv-lab tier, its derived step dropped from the script in lockstep, its three orphans handled
- Options:
  - `FV_HD_POST_URL` (`runpod-http.sh:341`): remove the comment, or refuse the variable;
  - the 1080P gate's `FV_PROMPTS=…/prompts-hd5.json`: `extra_cells` of `hd` (arms renamed), or a prompt-set cell in a tier of its own (the script's arm names)
- Decision: (agent, within the coordinator's decision)
  - the `hd` tier: `hd-canvases` (six arms over `prompts-hd.json`) and `hd-10s` (two arms), goldens from the live branch; the Lanczos upscale and its compares, the post script and the keyframe prune leave `runpod-matrix.sh hd` in the same PR;
  - `FV_HD_POST_URL` and `FV_HD_POST_TIMEOUT_S` refused (`runpod-http.sh` both paths, fv-lab `knobs::DROPPED_KNOBS`, `runpod-matrix.sh`), the comment left: it sits in the pod start command, so removing it would rewrite every pod golden, frozen and restored ones included;
  - the gate as the tier `hd5` (`hd5-canvases` with `prompt_set = "scripts/gpu/prompts-hd5.json"`, sharing `hd-10s`), a sibling of the `hd` family (`FV_LAB_SIBLINGS`), its golden the script's dry run with that `FV_PROMPTS`
- Reason: `hd_report.py` and `report summary --kind hd` read the arms by the script's names, which extra cells could not keep (arm names are unique per tier); the dry runs of every other family are unchanged, and `hd`'s lost only its eight empty dry-run timing lines (log only)
- Reversibility: high (cells, fixtures; the dropped commands are in `git show 4f8822a0:scripts/gpu/runpod-matrix.sh`)
- Executed by: Claude (session_01JVv1ZmGZ9cVZ8hArQVp3MZ)
- ADR: docs/adr/0019-fv-lab.md §22.9, §22.10
- Verification: `tests/dry_run.rs` (the two tiers' dry runs and gates equal the script's), `tests/payload_golden.rs`, `tests/lab_files.rs`, `knobs::tests`, `fv-lab-goldens.sh --check`, `fv-lab-route.test.sh`, every family's dry run before and after; GPU: §22.10 (the coordinator's; `turbo-768p turbo-1080p max-1080p`, ~$2.00)
- Open: S7 (after §22.10 passes): `hd` into `FV_LAB_RETIRED`, its branch deleted, `hd5` frozen with it
