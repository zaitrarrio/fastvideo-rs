# ADR-0010: The settings layer: one declarative list of `FASTVIDEO_*` settings

- Status: Accepted; plan B3 complete (first half B3a #153: the list, the
  routing, `docs/scope.md`; second half B3b: the `Settings` snapshot passed
  to `load` and the `model_spec`s of Wan and the small families, see
  "Amendment (B3b)")
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-settings-layer
- Code: `crates/fastvideo-models/src/techniques/settings/` (`mod.rs` the
  readers, `table.rs` the list, `parse.rs` the shared parse rules, `doc.rs`
  the renderer, `snapshot.rs` the load snapshot and the `Source` trait,
  `workspace_check.rs` the acceptance tests); re-exported as
  `fastvideo_models::settings`; the model specs in
  `crates/fastvideo-models/src/techniques/registry.rs`
- Doc: `docs/scope.md` "Configurations" (generated region)

## Context

The architecture review (`docs/gaps/2026-10-22-architecture-review.md` §4.2,
§7.3 item 3) counted 233 `FASTVIDEO_*` names in the crates, two readers
(`techniques::settings::var`, the env > profile > default reader, and the
runtime's `envflag` getters on top of it) and 111 direct `std::env::var`
reads that bypass both. `docs/scope.md` documented about fifty of them by
hand. A name could be added, renamed or given a second parse anywhere, and
nothing listed what exists.

The non-negotiable for this change is byte identity: every variable keeps
its name, its parse (truthy set, number format, trimming, default when unset
or unparsable), its read time (once per process, per load, per generate, per
call) and its precedence against technique profiles.

The census (the PR description has it per name, with every read site):
222 real names (the other 11 of the review's 233 are test scaffolding,
prefixes used in `starts_with`, and Rust identifiers that contain the
string). 112 direct literal reads in 42 files, plus 27 indirect ones: a
`const` name passed to `std::env::var` (`BUDGET_ENV`, `dit_tree::ENV`,
`PLUG_ROOT_ENV`, `UPSAMPLER_ENV`, `MoeResidency`'s `ENV`, ...), helpers that
take a name (`env_usize`, `env_secs`, `env_or`, `env_f64`), and lookup
closures (`|k| std::env::var(k).ok()` in SANA-Video and `fv-gpucheck`). No
`option_env!` and no `format!`-built names.

## Decision

### Where the list lives

In `fastvideo-models` (`techniques::settings`), not in a new crate.
`fastvideo-models` depends only on `fastvideo-ops`, and every crate that
reads a setting (`fastvideo-cudarc`, `-core`, `-loader`, `-gpucheck`,
`-engine-service`) already depends on it, so the list and both readers sit
below all of them without a cycle. A separate dependency-free crate would add
a workspace member for no gain; it becomes worth it only if `fastvideo-ops`
ever needs a setting.

### One row per name

`SETTINGS` is a sorted `const` array of `Setting { name, reader, parse,
rule, default, family, seam, timing, doc }`:

- `reader`: `Layered` (`settings::var` / `is_set`: environment, then the
  profile table, then the caller's default), `Env` (`settings::env_var` /
  `env_var_os`: exactly `std::env::var` / `var_os`), or `Reserved` (a parser
  and a `const` exist, nothing reads it: `FASTVIDEO_NVFP4_COSMOS_STEPS`,
  `FASTVIDEO_NVFP4_LTX_VIDEO_FFN`);
- `parse`: `Flag`, `Usize`, `F32`, `F64`, `Keyword` (the shared rules of the
  runtime's typed getters) or `Text` (parsed at the call site, as `rule`
  says);
- `default`: `Unset`, `Value(v)` (setting `v` explicitly changes nothing) or
  `Computed(how)` (device, model, request or another setting decides);
- `family`, `seam` (a technique seam from `docs/techniques.md`, or the
  subsystem), `timing` (`Start`, `Process`, `Load`, `Request`, `Call`), and
  the doc string `docs/scope.md` shows.

The rows describe the readers; they do not drive them. Call sites keep
their string literals (`bool_flag("FASTVIDEO_WAN_FUSE", true)`,
`settings::env_var("FASTVIDEO_WAN_VAE")`): searching for a name still finds
its readers, and moving every read behind a typed key would have rewritten
about 400 sites in one structure change. The tests below make the list
authoritative instead.

### Routing: two readers, each name keeps its own

Every direct read became `settings::env_var(name)` / `env_var_os(name)` at
the same call site; the indirect forms call the same functions inside their
helpers and closures. These functions are `std::env::var` / `var_os`, so the
value, the parse that follows, the call time and the precedence are the ones
the code had. None of these names was moved to the layered reader: that
would let a profile's `[env]` entry (or a technique) set it, a behaviour
change that gets its own byte-identity cell when someone wants it.

The layered reads were already `settings::var` / `is_set` and the runtime's
`envflag` getters; they are unchanged except that `bool_flag`, `usize_flag`,
`f32_flag`, `f64_flag` and `string_flag` now call `settings::parse::{flag,
usize_or, f32_or, f64_or, lower_or}`, the same expressions moved into
`fastvideo-models` so the table's tests check the code the getters run.

### What the tests enforce (the acceptance criterion)

`workspace_check.rs` scans every `.rs` file under `crates/` (comments
stripped, string literals located) and fails on:

1. `std::env::var("FASTVIDEO_...")` / `var_os` anywhere outside the settings
   module;
2. in the inference crates, a `std::env::var` / `var_os` whose argument is
   not a string literal, except ten reviewed sites, none of which reads a
   `FASTVIDEO_*` setting (`INDIRECT_ALLOWED`: an env save/restore guard, the
   process-plan check that compares before it sets, legacy `WAN22_*`
   aliases, `FV_*`, `FASTH3_*` and `XDG` / `HOME` names);
3. a `FASTVIDEO_*` name in any string literal that is not a row
   (`FASTVIDEO_TEST_*` exempt), and a row no source names;
4. a typed getter whose parse, literal default or reader disagrees with its
   row, `settings::var` on an `Env` row, `env_var` on a `Layered` row
   (except six reviewed sites: tests that save and restore a variable, and
   two readers that have always read the environment only);
5. `docs/scope.md`'s generated region differing from `render_markdown()`
   (the failure prints the region to paste).

Off-identity tests (`table.rs`): for every typed row with a `Value`
default, unset and set-to-the-default parse equal through the layered reader;
the same for the twenty text rows whose parser lives in `fastvideo-models`
(`ltx2::hq::requested`, `nvfp4::requested`, `MoeResidency::parse`, ...);
truth tables of `flag` (both polarities), the number parsers and `lower_or`;
env > profile > default for `FASTVIDEO_H3_QUANT`, `BF16_ACT`, `DIT_OFFLOAD`,
`DIT_OFFLOAD_LOOKAHEAD`, `FLASH_KERNEL` (H3 profile) and `LTX_OFFLOAD` (LTX
profile); every name a compiled-in profile or a `[kernels]` op installs is a
`Layered` row (an `Env` row there would make the profile entry silently do
nothing); an `Env` row does not see a model scope's table.

### Read timing

Unchanged by construction: no read moved out of its function, no cached
value was introduced or removed, and `env_var` does not touch the profile
table (so it never initializes the lazily loaded profile, which a
`settings::var` call would). The `timing` column records what the code does.
(B3b: a `Load` read now returns the value the load's snapshot captured when
the load started, see the amendment.)

## Consequences

- A new setting is a row first; a reader of an undeclared name fails
  `cargo test -p fastvideo-models`. `docs/scope.md` lists all 222.
- The census surfaced three quirks, kept as they are and now documented in
  the rows: `FASTVIDEO_INJECT_COND` is on only for `1` in H3 and on unless
  `0` in LTX-2; `FASTVIDEO_VSA` is a layered default-off flag in the runtime
  but the `fastvideo` CLI's VSA-id check reads `1` / `true` from the
  environment only; `fv-gpucheck wan` reports `FASTVIDEO_WAN_COND_CACHE` from
  the environment even when a profile set it. Two names are reserved
  (declared, never read).
- 108 rows are `Env` (112 `Layered`, 2 `Reserved`): a technique profile
  cannot set them. Promoting one is a reviewed behaviour change.

## Amendment (B3b): the snapshot passed to `load`, and `model_spec`

The first PR left two items of plan B3 (the review's §7.3 item 3): "pass a
`Settings` snapshot into `load` instead of process globals" and "give Wan
(and the small families) a `model_spec`". Both are below. The constraint is
the one above: byte identity (same values, precedence, defaults, and reads at
points that give the same result for a process that sets its environment
before it starts).

### The snapshot passed to `load`

`settings::Snapshot` (`snapshot.rs`; the plan calls it the `Settings`
snapshot, the name `Settings` was already the profile table's) holds, for
every row of `SETTINGS`, the pair both readers consult: the process
environment's value (`std::env::var_os`) and the table's (the model scope's,
else the active profile's). `Snapshot::capture()` takes it; the four
accessors are the readers' definitions over the pair (`var`: the environment
when it is valid UTF-8, else the table; `is_set`: either present; `env_var`
/ `env_var_os`: the environment only, including `NotUnicode`), and the typed
accessors (`bool_flag`, `usize_flag`, `f32_flag`, `f64_flag`,
`string_flag`) are the runtime getters' `settings::parse` rules over `var`.
The trait `settings::Source` has these methods; `Snapshot` and `Live` (the
live readers) implement it, so a load-time helper written once against
`&dyn Source` serves the load (the snapshot) and its other callers (live).

Each loader gained an entry that takes the snapshot, and its old entry
captures one and calls it (no caller changed; B6 owns the loaders' shape):

| loader | takes the snapshot | captures it |
|---|---|---|
| H3 | `H3Pipeline::load_with_settings(root, options, &Snapshot)` (`crates/fastvideo-cudarc/src/h3/pipeline.rs:1317`) | `H3Pipeline::load` (:1308) |
| LTX-2 | `Ltx2Pipeline::load_with_settings(paths, cfg, options, &Snapshot)` (`ltx2/pipeline.rs:2930`) | `Ltx2Pipeline::load` (:2921) |
| Wan | `WanPipeline::load_with_settings(root, preset, parts, dit, &Snapshot)` (`wan/pipeline.rs:455`) | `WanPipeline::load_with_dit` (:439), and through it `load` / `load_with` |

Inside, two mechanisms, chosen so that no loader is restructured:

1. **Explicit**: the reads in the three load bodies and in the load-only
   helpers they call take the snapshot as a parameter (`settings.var(..)`,
   `X::from_settings(settings)`); the helpers that other code also calls got
   a `from_settings` / `from_settings_or` twin and their `from_env` became
   `from_settings(&Live)` (`QuantMode`, `DitOffload`, `LtxOffload`,
   `ModelOptOptions`, `MoeResidency`, `nvfp4::from_env`,
   `H3Techniques::from_process`, `WanVaeChoice`, `streaming_fetch_env`,
   `headroom_bytes`).
2. **Scoped**: the first statement of each `load_with_settings` is
   `let _load_scope = settings.scope();`. While the guard lives, this thread's
   `settings::var`, `is_set`, `env_var` and `env_var_os` return the
   snapshot's values for every row. The shared code a load calls (the DiT and
   decoder constructors, quantization, weight loading, the text encoders)
   keeps its signatures and reads the same values. A name that is not a row
   is read live (none is: `workspace_check` enforces it).

Why this is the same behaviour:

- *Values and precedence*: by construction, the accessors are the readers'
  definitions over the captured pair, and the typed accessors call the same
  `settings::parse` functions as the runtime getters. Tested over every row
  against the live readers, including a model scope and a non-UTF-8 value
  (`snapshot::tests::a_captured_snapshot_equals_the_live_readers`,
  `a_non_utf8_environment_value_is_what_std_env_makes_of_it`,
  `accessors_follow_the_readers_precedence`).
- *Read time*: a read inside a load now returns the value at load start
  instead of the value at its own line. These differ only if the
  environment, the profile or the model scope changes during a load. None
  does: the profile is installed once per process (`install` refuses a
  second), the model scope (`scope_model`, fv-serve) is entered by the caller
  around the whole load, and every `set_var` of a `FASTVIDEO_*` name outside
  tests runs before a load starts (`fv-gpucheck --mode` and its CLI flags,
  the `fastvideo` CLI's `FASTVIDEO_SP_WORLD` / `FASTVIDEO_WAN_AUDIO`,
  fv-serve's process plan and its Wan decoder `EnvGuard`). The one write
  inside a load, the MLX H3 tree's `FASTVIDEO_H3_AFFINE`
  (`rt::affine::apply_env`, called by `h3::mlx::open_map` in the H3 load),
  now goes through `settings::set_env`, which sets the environment and the
  running load's scoped snapshot, so the linear loads after it read the value
  as before (`snapshot::tests::set_env_inside_a_load_is_seen_by_the_later_reads`).
- *Threads*: the scope is thread-local, like the model scope. A worker thread
  a loader spawns reads the live readers, as it did before (it never saw the
  model scope either), so its values are unchanged too.
- *Nested loads* (the H3 Spark refiner loads LTX-2 inside an H3 generate,
  not inside the H3 load; a load started inside another load would capture
  the outer snapshot, which is what the live readers return there).
- *Profile initialisation*: `capture` reads the table for every row, which
  initialises the lazily loaded profile (`FASTVIDEO_TECHNIQUES`) if nothing
  did before. Each of the three loads already did at its first layered read
  (H3: `settings::active()`; LTX-2: `active()`; Wan: `FASTVIDEO_WAN_DIT_LAZY`
  among its first statements), and every binary installs the profile at
  start-up, so this moves nothing observable.

Off-identity: with no `FASTVIDEO_*` set and no profile, every captured pair
is empty and every accessor returns what the reader returns today (no value;
for each of the typed rows with a documented default, that default)
(`an_empty_snapshot_is_todays_defaults_field_for_field`).

Per family, the resolved load configuration from a snapshot built from given
sources equals what the pre-snapshot code resolved from the same sources (the
old bodies written out in the tests: `settings::var` = env, else table;
`env_var` = env), over env and profile-table arms of each setting:

- H3: `H3Techniques` for 8 recipes x {no profile, a profile} x 15 arms
  (`h3::techniques::tests::a_snapshot_resolves_what_the_live_readers_resolved`);
  `QuantMode`, `planned_dit_weights`, `DitOffload` (with and without the CLI
  flag), `ModelOptOptions`, `resolve_taeh3`, `vdn_checkpoint`, the NVFP4
  rule (`h3::pipeline::load_settings_tests`);
- LTX-2: `LtxOffload`, `DitOffload`, the text residency and LoRA-base reads,
  `resolve_ic_lora`, `resolve_tae` (`ltx2::pipeline::load_settings_tests`);
- Wan: the causal frames per block, `FASTVIDEO_WAN_DIT_LAZY` with both
  defaults, the streaming-fetch gate, `DitOffload`, `MoeResidency` and its
  headroom, `WanVaeChoice`, the TAEHV weights and search directories
  (`wan::pipeline::load_settings_tests`);
- the model crate's helpers on their own (`snapshot::tests::model_load_helpers_resolve_the_same_from_a_snapshot`).

### Classification of the reads in the three load paths

Every settings read the H3, LTX-2 and Wan loads make, in the load functions
and the helpers they call, and the reads in the same files that run later.
Lines are on this PR's head; `crates/fastvideo-cudarc/src/` is omitted.

- **a1**: read inside the load, now from the snapshot passed as a parameter;
- **a2**: read inside the load by a callee whose code is unchanged, served
  from the snapshot by the load scope (when the same function also runs
  after the load, e.g. a decoder loaded per decode under streaming, that
  later call reads live, as before);
- **b**: cached once per process on its first read (`OnceLock`,
  `CachedBool`, `CachedString`, `settings::active()`); kept. When the first
  read happens inside a load it caches the snapshot's value, which is the
  value it would have cached;
- **c**: per generate, per step or per call; stays where it is and reads
  live (inside a load, a per-call op used by the load is served by the
  scope).

**H3** (`H3Pipeline::load_with_settings`, `h3/pipeline.rs:1317`)

| site | function | setting | class |
|---|---|---|---|
| `h3/pipeline.rs:423`, `:447` | `vdn_checkpoint` | `FASTVIDEO_H3_VDN`, `FASTVIDEO_H3_VDN_SHARED` | a1 |
| `h3/pipeline.rs:864` | `dit_tree_identity` | `FASTVIDEO_NVFP4` (`nvfp4::from_settings`) | a1 |
| `h3/pipeline.rs:863` | `dit_tree_identity` | `FASTVIDEO_H3_AFFINE` (`affine::bits_from_env`, `rt/affine.rs:61`) | b |
| `h3/pipeline.rs:925` | `dit_tree_identity` | `FASTVIDEO_H3_QUANT` (`QuantMode::from_settings`) | a1 |
| `h3/pipeline.rs:990` | `resolve_taeh3` (from the load: `:1390`, `:1651`) | `FASTVIDEO_TAEH3_WEIGHTS` | a1 |
| `h3/pipeline.rs:3105`, `:3277` | `resolve_taeh3(.., &Live)` in `load_video_decoder_from`, `planned_linears` | `FASTVIDEO_TAEH3_WEIGHTS` | a2 in the load, c per decode when streamed |
| `h3/pipeline.rs:1254` | `planned_dit_weights` | `FASTVIDEO_H3_QUANT` | a1 |
| `h3/pipeline.rs:1327` | load | `settings::active()` (the profile) | b |
| `h3/pipeline.rs:1371`, `:1382` (`models h3/techniques.rs:329`) | load: `H3Techniques::from_settings` / `resolve` | `FASTVIDEO_ATTN_FP8`, `FASTVIDEO_H3_SOL_ATTN`, `FASTVIDEO_H3_SOL_CACHE`, `FASTVIDEO_VSA_GROUP`, `FASTVIDEO_VSA_SPARSITY` | a1 |
| `h3/pipeline.rs:1412` | load (log) | each profile setting's name (`env_var_os`) | a1 |
| `h3/pipeline.rs:1443`, `:1946`, `:2039` | load | `FASTVIDEO_DIT_OFFLOAD` (`DitOffload::from_settings_or`) | a1 |
| `h3/pipeline.rs:1449`, `:1143` | load, `denoise_reserve_bytes` | `FASTVIDEO_BF16_ACT` (`tensor::bf16_activations`, `rt/tensor.rs:66`) | b |
| `h3/pipeline.rs:1556` (`h3/nvfp4.rs:306`, `:311`) | load: `ModelOptOptions::from_settings` | `FASTVIDEO_H3_NVFP4_ATTN`, `FASTVIDEO_H3_NVFP4_ACT` | a1 |
| `h3/pipeline.rs:1832` | load (Spark log) | `FASTVIDEO_H3_QUANT` | a1 |
| `h3/pipeline.rs:1056`, `:1112` (`rt/attn_fp4.rs:33`, `:39`) | `vsa_fp4_choice` | `FASTVIDEO_H3_VSA_FP4`, `FASTVIDEO_H3_VSA_FP4_P` | a2 (B5's file, unchanged) |
| `h3/pipeline.rs:3366` (`h3/vae.rs:219`) | `resolve_video_vae` -> `nvfp4_setting` | `FASTVIDEO_H3_VAE_NVFP4` | a2 |
| `h3/vae.rs:140` | `mxfp8_resolved` (`planned_linears`, `load_with`) | `FASTVIDEO_H3_VAE_MXFP8` | a2; c per decode when streamed |
| `h3/vae.rs:292`, `:793`, `:893`, `:936`, `:958` | `nvfp4_static_act`, `load_choice`, `load_with` | `FASTVIDEO_H3_VAE_NVFP4_ACT`, `FASTVIDEO_H3_VAE_NVFP4_REF`, `FASTVIDEO_H3_VAE_NVFP4`, `FASTVIDEO_H3_VAE_MXFP8`, `FASTVIDEO_H3_VAE_TILE_BATCH` | a2; c per decode when streamed |
| `h3/vae.rs:50`, `:56`, `:337`, `:362`, `:375`, `:405` | decoder getters | `FASTVIDEO_H3_VAE_FUSED`, `_CHECK`, `_AB`, `_VEC`, `_GEMM`, `_SDPA` | b |
| `h3/vae.rs:383` | `mx_algo` | `FASTVIDEO_H3_VAE_MXFP8_ALGO` | c |
| `h3/dit_tree.rs:89`, `:292` | `select` (from the load) | `FASTVIDEO_H3_DIT_TREE`, `FASTVIDEO_H3_DIT_TREE_DIR` | a2 |
| `h3/lora.rs:264` | `H3LoraFuse::open_plug` | `FASTVIDEO_PLUG_ROOT` | a2 |
| `h3/transformer.rs:2718`, `:2772`, `:2998`, `:3013`, `:3160` | refiner / DiT loads | `FASTVIDEO_H3_QUANT` (`QuantMode::from_env`) | a2 |
| `h3/transformer.rs:2118` | `refuse_retired_ffn_fp8` (`FeedForward::load`) | `FASTVIDEO_H3_FFN_FP8` | a2 |
| `h3/transformer.rs:1658`, `:2154` | `Attention::load`, `FeedForward::load` | `FASTVIDEO_H3_AFFINE` | b |
| `h3/transformer.rs:198`, `:338` | `lora_device::enabled`, `verify_enabled` | `FASTVIDEO_H3_LORA_DEVICE`, `_LORA_VERIFY` | b |
| `h3/transformer.rs:634`, `:1478`, `:1763`, `:1879`, `:2067`, `:2078`, `:2091`, `:2224`, `:3537`, `:3693`; `h3/vdn.rs:242`, `:472`; `h3/vae.rs:1042` | forwards | `FASTVIDEO_BF16_ACT` | b |
| `h3/fused16.rs:308`, `:336` | `fuse_enabled`, `glue_enabled` | `FASTVIDEO_H3_FUSE`, `FASTVIDEO_H3_GLUE` | b |
| `h3/vdn.rs:1183` | `Small::load_embedded` (VDN branch attach) | `FASTVIDEO_KERNELS` | a2 |
| `llm/prefetch.rs:47`, `llm/prequant.rs:151` | resident text encoder open | `FASTVIDEO_LLM_PREFETCH`, `FASTVIDEO_TEXT_FP8_TREE` | a2; c when the encoder streams per request |
| `rt/offload.rs:139` (`BlockWeights::new`, `:227`) | streamed blocks | `FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD` | a2 |
| `rt/nn.rs:185` (`nvfp4_rule_for`), `models nvfp4.rs:233`, `:297` | linear loads | `FASTVIDEO_NVFP4` | a2; c per call |
| `rt/nn.rs:221`; `rt/stage_upload.rs:43`, `:152`; `rt/dump.rs:211`, `:220`; `rt/device.rs:352`; `rt/bf16_gemm.rs:14`; `rt/hopper.rs:16`; `rt/resident.rs:16`; `rt/conv.rs:576`; `rt/act16.rs:335` | runtime getters | `FASTVIDEO_FP8`, `_STAGED_UPLOAD`, `_VERIFY_UPLOAD`, `_DIGEST`, `_DIGEST_WEIGHTS`, `_DEVICE_BUDGET_GIB`, `_BF16`, `_TF32`, `_RESIDENT`, `_CONV3D_TIMED`, `_SPLIT_ROWS` | b |
| `loader prefetch.rs:104` (`:67`, `:73`, `:84`), `:502`; `loader stream.rs:70`, `:73`, `:74` (`PartsGate::open`) | weight maps | `FASTVIDEO_PREFETCH`, `_STREAMING_PREFETCH_THREADS`, `_PREFETCH_WINDOW_GB`, `_PREFETCH_MODE`; `_STREAMING_FETCH_STATUS`, `_STALE_S`, `_TIMEOUT_S` | b (prefetch config, mode); a2 (gate options) |
| `rt/kernels.rs:408` | `KernelFns::load_for` (device init) | `FASTVIDEO_KERNELS` | b (once per device) |
| `h3/pipeline.rs:611`, `:2350`, `:2386`, `:2448`, `:3593`, `:3607`; `h3/spark.rs:559`, `:572`; `h3/step_graph.rs:29`, `:153`; `h3/vsa.rs:67`, `:642`, `:891`, `:1063`, `:1073`, `:1259`, `:1262` | generate, `set_arm`, Spark bridge and refiner, steps, VSA | `FASTVIDEO_INJECT_COND`, the profile and arm settings, `FASTVIDEO_LTX2_WEIGHTS`, `_LTX2_DIT`, `_H3_UPSCALER`, `_H3_LTX_ADAPTER`, `_H3_STEP_SYNC`, `_H3_GRAPHS`, `_VSA_GROUP`, `_VSA_KERNEL`, `_VSA_UNION_LOG`, `_VSA_BF16` | c |

**LTX-2** (`Ltx2Pipeline::load_with_settings`, `ltx2/pipeline.rs:2930`)

| site | function | setting | class |
|---|---|---|---|
| `ltx2/pipeline.rs:2725` | `resolve_ic_lora` | `FASTVIDEO_LTX2_IC_LORA` | a1 |
| `ltx2/pipeline.rs:2737` | `resolve_tae` | `FASTVIDEO_LTX2_TAE_WEIGHTS` | a1 |
| `ltx2/pipeline.rs:2980` | load | `settings::active()` | b |
| `ltx2/pipeline.rs:2986` | load (log) | each profile setting's name (`env_var_os`) | a1 |
| `ltx2/pipeline.rs:3006` (`models ltx2/memory.rs:599`) | load | `FASTVIDEO_LTX_OFFLOAD` (`LtxOffload::from_settings_or`) | a1 |
| `ltx2/pipeline.rs:3010` | load | `FASTVIDEO_LTX2_TEXT` | a1 |
| `ltx2/pipeline.rs:3015` | load | `FASTVIDEO_DIT_OFFLOAD` | a1 |
| `ltx2/pipeline.rs:3118` | load | `FASTVIDEO_LTX2_LORA_BASE` | a1 |
| `ltx2/pipeline.rs:3020` (`rt/offload.rs:139`) | load (log) | `FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD` | a2 |
| `ltx2/lora.rs:99` (from `ltx2/pipeline.rs:2946`) | `lora::resolve` | `FASTVIDEO_LTX2_LORA` | a2 |
| `ltx2/pipeline.rs:1735` (from `:3078`, `:1745`) | `upsampler_dir` | `FASTVIDEO_LTX2_UPSAMPLER` | a2 in the load; c when the upsampler loads per generate (`:4237`) |
| `ltx2/pipeline.rs:2285`, `:2320` | `TextEncoder::new`, `warm` | `FASTVIDEO_LTX2_TEXT_FP8`, `_LTX2_TEXT_WARM` | a2 |
| `llm/prequant.rs:151`, `llm/prefetch.rs:47` | Gemma open | `FASTVIDEO_TEXT_FP8_TREE`, `_LLM_PREFETCH` | a2; c per request when streamed |
| `rt/nvfp4.rs:43`, `models nvfp4.rs:233` | video FFN linear loads | `FASTVIDEO_NVFP4` | a2; c per call |
| `ltx2/fuse.rs:39`; `ltx2/vae_fast.rs:258`, `:281`; `rt/tensor.rs:66` (`ltx2/fuse.rs:81`, `ltx2/transformer.rs:759`) | getters | `FASTVIDEO_LTX_FUSE`, `_LTX_VAE_FAST`, `_LTX_VAE_CHECK`, `_BF16_ACT` | b |
| the runtime and loader getters listed under H3 | | | b |
| `ltx2/pipeline.rs:1935`, `:2475`, `:2570`, `:3393`, `:3401`, `:3409`, `:3446`, `:3459`, `:3500`, `:3536`, `:3571`, `:3669`, `:3947`, `:4352`, `:4753`; `ltx2/text_cache.rs:144`; `ltx2/i2v_encode.rs:121`; `ltx2/transformer.rs:463`; `ltx2/v2v.rs:586`, `:746`; `ltx2/vae.rs:526`, `:569`; `ltx2/vae_fast.rs:782`, `:783`, `:1057` | generate, encode, decode | `FASTVIDEO_LTX2_SAVE_LATENTS`, `_LTX2_TEXT`, `_LTX2_FBCACHE`, `_LTX2_STAGE1_CACHE`, `_LTX2_MIDPOINT_PRUNE`, `_INJECT_*`, `_LTX2_HQ`, `_LTX2_AUDIO_VAE`, `_CACHE`, `_FFMPEG`, `_LTX2_HOST_ROPE`, `_LTX2_VAE_CHUNK`, `_LTX2_VAE_LATENT_CHUNK`, `_LTX_VAE_CHUNK_FRAMES`, `_LTX_VAE_CHUNK_MB`, `_LTX_VAE_CONV_ALGO` | c |

**Wan** (`WanPipeline::load_with_settings`, `wan/pipeline.rs:455`)

| site | function | setting | class |
|---|---|---|---|
| `wan/pipeline.rs:474` | load | `FASTVIDEO_WAN_CAUSAL_FPB` | a1 |
| `wan/pipeline.rs:491` (`loader stream.rs:89`) | load | `FASTVIDEO_STREAMING_FETCH` (`streaming_fetch_from`) | a1 |
| `wan/pipeline.rs:492` | load | `FASTVIDEO_WAN_DIT_LAZY` | a1 |
| `wan/pipeline.rs:547` | load | `FASTVIDEO_DIT_OFFLOAD` | a1 |
| `wan/pipeline.rs:553` (`models wan/moe.rs:84`) | load | `FASTVIDEO_WAN_MOE` | a1 |
| `wan/pipeline.rs:556` (`models wan/moe.rs:110`) | load | `FASTVIDEO_WAN_MOE_HEADROOM_GIB` | a1 |
| `wan/pipeline.rs:593` (`:126`) | load: `WanVaeChoice::from_settings` | `FASTVIDEO_WAN_VAE` | a1 |
| `wan/pipeline.rs:39` | `load_taehv` | `FASTVIDEO_TAEHV_WEIGHTS` | a1 |
| `wan/pipeline.rs:88` | `default_taehv_dirs` | `FASTVIDEO_TAE_DIR` | a1 |
| `wan/transformer.rs:1410` (`rt/quant.rs:530`) | `load_with_residency` | `FASTVIDEO_WAN_QUANT` | a2 |
| `rt/offload.rs:139` | streamed / parked blocks | `FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD` | a2 |
| `wan/plug.rs:224` | `load_merged_transformer` (fv-serve, before `load_with_dit`) | `FASTVIDEO_PLUG_ROOT` | read by the caller before the load starts, live, unchanged |
| `wan/fuse.rs:47`; `wan/transformer.rs:793`; `wan/pipeline.rs:2131`; `rt/graph.rs:40`; `rt/nn.rs:2033`, `:2041`, `:2045`, `:2202`; `rt/tensor.rs:66` (`wan/fuse.rs:60`, `wan/pipeline.rs:2174`, `:2271`, `wan/stream.rs:706`, `:1042`) | getters | `FASTVIDEO_WAN_FUSE`, `_WAN_COND_CACHE`, `_WAN_CAUSAL_AR`, `_WAN_GRAPH`, `_SP_WORLD`, `_SDPA`, `_VSA`, `_WAN_CAUSAL_FLASH`, `_BF16_ACT` | b |
| the runtime and loader getters listed under H3 | | | b |
| `wan/pipeline.rs:144`, `:1001`, `:1292`, `:1307`, `:1388`, `:1585`, `:1588`, `:1600`, `:1692`, `:2311`; `wan/transformer.rs:325`, `:1215`, `:1340`, `:1347`, `:1565`, `:1572`, `:1685`, `:1692`, `:1731`, `:1775`, `:1780`, `:1965`, `:2147`, `:2148`; `wan/sol_cache.rs:24`, `:53`, `:61`; `wan/vae.rs:1135`; `rt/graph.rs:101` | generate, steps, per-generate configuration of the caches and routes, decode | `FASTVIDEO_DEVICE_STATS`, `_NVFP4`, `_TEACACHE`, `_TEACACHE_THRESH`, `_TEACACHE_RET`, `_WAN_SOL_CACHE`, `_WAN_UNIPC_SIGMAS`, `_SLA_*`, `_WAN_EASYCACHE_*`, `_WAN_TEACACHE_COEFFS`, `_WAN_TAYLOR_LITE`, `_WAN_SOL_ATTN`, `_WAN_PISA`, `_VSA_SPARSITY`, `_VSA_GROUP`, `_VAE_CHUNK`, `_WAN_GRAPH_CUBLAS_WS_MIB` | c |

Cited lines per class (a line in parentheses, the helper a site calls, is
counted too): H3 a1 17, a2 31, b 49, c 18; LTX-2 a1 8, a2 13, b 7 plus the
shared getters listed under H3, c 25; Wan a1 13, a2 3, b 14 plus the shared
getters, c 29, and one read by the caller before the load starts. No read
changed class: a1 and a2 were load-time reads before, and b and c keep their
caches and call sites.

### The order dependence of first-read caching

The review's concern (§4.2: "the caching makes settings order-dependent
process globals (`gpucheck/src/mode.rs` must run at process start)"):

- **Removed for the load-time reads (a1, a2)**: their values are the
  environment and profile at load start, whatever ran before inside the
  process and whatever runs during the load
  (`snapshot::tests::a_scope_serves_the_capture_not_a_later_change`: a change
  after capture is invisible until the scope ends; a nested capture equals
  the outer snapshot; another thread reads live).
- **Not removed for the cached getters (b)**: a getter keeps the value of
  its first read for the life of the process, whether that read happens in a
  load (then the snapshot's value) or later
  (`rt::envflag::tests::a_cache_filled_inside_a_load_keeps_the_snapshot_value`:
  a second load with another snapshot does not refill it). Moving them into
  the snapshot would make them per load, a behaviour change for a process
  that loads more than once (fv-serve `[engine] swap`), which the
  `ProcessPlan` check exists to keep consistent. So `fv-gpucheck`'s
  `Mode::apply_env` (`FASTVIDEO_BF16`, `FASTVIDEO_TF32`: b;
  `FASTVIDEO_TEACACHE`: c, each Wan generate) must still run before the first
  cudarc call; `mode.rs`'s doc says so and points at the test.

### `model_spec`

`registry::model_spec(model)` answers whether a technique profile may name
`[pipeline].model`; it stays `h3` and `ltx2` / `ltx`. `ModelSpec` gained what
the review asked a spec to say, which techniques-layer inputs a model's
runtime reads: `families` (the settings families whose `Layered` rows it may
read), `extra_settings`, `kernels` (the `[kernels]` ops it dispatches) and
`profiles` (whether `model_spec` returns it). `registry::family_spec(name)`
returns a spec for every family `fastvideo-core` registers: `h3_spec`,
`ltx2_spec`, `wan_spec` and one per small family (`SMALL_FAMILIES`, the 18
others; capabilities empty, so every technique fails its capability check
there).

**Decision: the minimal version.** `wan_spec()` lists the Wan runtime's
capabilities (its seams exist and are driven by its own settings: attention
backend, step cache, FFN precision, the TAEHV decoder, streamed blocks), its
settings families (Wan and the runtime) and its kernel ops, with
`profiles: false`: a profile naming `wan` is refused with the same error as
before (`registry::tests::only_h3_and_ltx2_profiles_load`). The full version
(Wan profiles loadable) is not small and not safe here:

- the technique-to-setting mapping is H3's and LTX-2's (`settings_for`):
  `taehv` maps to `FASTVIDEO_LTX2_TAE_WEIGHTS`, while Wan reads
  `FASTVIDEO_TAEHV_WEIGHTS` through the env-only reader, so the entry would
  do nothing; the precision recipes map to `FASTVIDEO_H3_QUANT` or
  `FASTVIDEO_FP8`, not `FASTVIDEO_WAN_QUANT`; Wan's step caches and routes
  (`FASTVIDEO_WAN_SOL_CACHE`, `_WAN_SOL_ATTN`, `_WAN_PISA`, `FASTVIDEO_TEACACHE`)
  are env-only rows, and promoting any of them to the layered reader is the
  reviewed behaviour change above;
- it needs a Wan profile adapter (H3 and LTX-2 each have one,
  `h3::techniques`, `ltx2::techniques`) and a GPU byte-identity cell, and the
  EU volume has no Wan tree beyond `--tiny`.

**Wan's techniques-layer reads go through the spec** in the declarative
sense the list itself uses: the spec is the authority and a test keeps the
code inside it. `workspace_check::family_specs_declare_their_techniques_reads`
scans each family's sources (`crates/fastvideo-{cudarc,models}/src/<family>/`)
and the review's 11 files (now `rt/{attn,attn_dc,attn_fp4,attn_sage,conv,
envflag,offload,ops,quant,tensor,vsa}.rs`, moved out of `wan/` by B1) for
every layered read (the typed getters, `settings::var` / `is_set`, a
`Source`'s `.var` / `.is_set`, with a literal or a same-file `const` name) and
every `KernelOp`, and fails when the family's spec does not declare it. In
the runtime files a runtime row and every kernel op must be declared by the
Wan, H3 and LTX-2 specs, and a family row (`FASTVIDEO_H3_QUANT` in
`rt/quant.rs`) by its own family's spec. At run time the same reads come from
the Wan load's snapshot. The attention files are plan B5's; nothing in them
changed. `fastvideo-core`'s `family::tests::every_family_has_a_model_spec`
ties the specs to `FAMILIES`.

### The scan, extended

Rule 4 of `workspace_check` (a read agrees with its row) now also matches a
`Source`'s methods (`settings.bool_flag(..)`, `.var(..)`, `.env_var(..)`, the
snapshot reads in the loaders), so the literal defaults and readers of the
moved reads stay checked.

### Not in this change

- **Explicit parameters all the way down.** The callees served by the load
  scope (a2) keep reading through the readers; plan B6, which splits
  `H3Pipeline::load` and `Ltx2Pipeline::load` into plan, open, place and
  fuse, is where each stage takes the snapshot as a parameter and the scope
  can go.
- **fv-serve's decoder `EnvGuard`** (`engine-service/src/cuda/wan.rs`) still
  sets `FASTVIDEO_WAN_VAE` / `FASTVIDEO_TAE_DIR` in the process environment
  around a Wan load; with the snapshot it could pass them in a snapshot
  instead (`Snapshot::from_parts`). A behaviour-identical follow-up, not
  needed for B3.
- **Wan profiles** (the full `model_spec`, above).

## Status of the evidence

CPU: the tests above on the build pod (numbers in the PR). GPU: the
coordinator's `techniques` byte-identity cell (H3 and LTX profiles, env and
profile arms); Wan has no trees on the EU volume and relies on the routing
being `std::env::var` at the same call sites. B3b: the same family, run
against the B3b runtime image; every H3 and LTX-2 cell loads through
`load_with_settings`, so its env arm exercises the snapshot's environment
half and its profile arm the table half; Wan's load is covered by the CPU
identity tests above only (`--tiny` builds its pipeline in memory, without
`load`).
