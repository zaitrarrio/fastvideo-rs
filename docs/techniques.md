# Technique composition layer

Acceleration techniques (sparse attention routes, step caches, precision
recipes, tiny decoders, offload, kernel choices) used to be per-model env
flags and if/else chains inside each pipeline. They are now typed items that
compose, are checked for conflicts, and are selected by a profile file, with
every old `FASTVIDEO_*` flag still working and still winning.

The layer is a port of sol-engine's `techniques/` package (NVlabs/Sana,
branch `sol-engine` @ `6c2f582`, the checkout `scripts/gpu/upstream/setup.sh`
pins and the `fastvideo-rs-upstream-*` images carry). Citations below are
`file:line` in that checkout.

Code: `crates/fastvideo-models/src/techniques/` (host-side, no GPU), the
model adapters `crates/fastvideo-models/src/h3/techniques.rs` and
`crates/fastvideo-models/src/ltx2/techniques.rs`, and the readers in
`fastvideo-cudarc` (`wan::envflag`, the kernel dispatch sites, the H3 and
LTX-2 pipelines). Profiles: `profiles/<model>/*.toml`, also compiled into the
binaries.

## Using it

```bash
# A shipped profile, by name (compiled in) or by path:
fv-gpucheck --mode fast --techniques h3/rtx5090_fullopt h3 gen --weights $W/h3-base ...
fastvideo --techniques profiles/h3/fasth3_8step.toml generate --model ...
# The same through the environment:
FASTVIDEO_TECHNIQUES=h3/rtx5090_sol fv-gpucheck ... h3 gen ...
# A sol-engine config loads as it is:
fv-gpucheck --techniques /path/to/sol-engine/config/minimax_h3/rtx5090_fullopt.toml ...
```

Without a profile nothing changes: every read falls through to the env var
or the built-in default, exactly as before (the off-identity invariant, below).

### Precedence

For every setting, highest first:

1. the command line (`--h3-recipe`, `--taeh3-weights`, `--dit-offload`, ...);
2. the environment variable (`FASTVIDEO_H3_SOL_ATTN`, `FASTVIDEO_H3_QUANT`,
   `FASTVIDEO_SOL_KERNEL`, ...): **every existing flag keeps working and
   overrides the profile**;
3. the profile (a technique, a `[kernels]` entry, or a raw `[env]` entry);
4. the recipe's default (e.g. `sol-h3-rtx` runs the Sol RTX route);
5. the built-in default.

The pipeline log prints the resolved plan, each installed setting with the
technique that set it, and `overridden by env` where an env var won. The H3
`benchmark.json` request block records the same under `techniques`.

## Design, and what it takes from sol-engine

| sol-engine | here |
|---|---|
| `Technique` with a `Schedule[bool]` `enabled`, a `phase`, `reads` / `writes` seams and `required_capabilities` (`technique.py:101-139`) | `trait Technique` (`technique.rs`) |
| `Phase`: `WRAP_ATTENTION` 10, `PRE_BLOCKS` 20, `IN_BLOCKS` 30, `POST_BLOCKS` 40, `ON_STEP` 50 (`technique.py:27-38`) | `Phase`, same order |
| `ModelTransform` with `TransformPhase` `LOAD` / `BUILD`, applied once (`transform.py:32-77`) | `Kind::Transform(TransformPhase)`; one trait for both kinds |
| `Seam` and `EXCLUSIVE_SEAMS` = attention backend, token set, step output, FFN precision (`technique.py:41-62`) | `Seam`, same four exclusive, plus `activation_precision`, `video_decoder`, `residency` (one owner each in this runtime) |
| `Capability` / `ModelSpec` (`technique.py:65-82`, `spec.py:20-55`) | `Capability`, `ModelSpec`; `registry::h3_spec()`, `ltx2_spec()` |
| `compose()`: capability check (:163-171), exclusive-seam rule with schedule overlap (:65-80), same-phase write-read rule (:82-93), ordering (:176-178) | `compose::compose`, `check_conflicts`, same rules and messages |
| `Schedule`: `const`, `at_steps`, `before`, `by_stage`, `parse_steps("1-2,5")`, `truthy_steps` (`schedule.py:18-108`) | `schedule::Schedule`, `StepSet` |
| `register_technique` / `build_technique` (`registry.py:22-64`) | `registry::TECHNIQUES`, `registry::build` |
| `set_env` into `TransformContext.env` (`transform.py:39-46, :73-74`) | `Technique::settings()` into the process `Settings` table |
| config manifests `config/<dim>/<name>.toml`, `[efficiency]` / `[requirements]` (`config_manifest.py:59-115`) and the profile configs `config/minimax_h3/rtx5090_*.toml` | `profiles/<model>/*.toml` (below); sol-engine configs import directly |

Where it differs, on purpose:

* **Hooks live in the model adapter.** sol-engine's hooks (`before_blocks`,
  `wrap_attention`, `on_step`, `technique.py:124-139`) are Python callables
  over torch tensors. Here a technique is a validated parameter set, and the
  pipeline asks the composed plan what to do at each step and layer. That is
  the split sol-engine's own transforms use: they "delegate to the EXISTING
  mechanisms (they set the env/config the current load/build code already
  reads)" (`transform.py:18-20`), and sparse attention leaves "translating
  latent/text shapes into backend metadata" to the model runtime
  (`transforms/sparse_attention.py:3-6`).
* **Settings go to a table, not the process environment.** `set_env`
  writes `os.environ`; here `Technique::settings()` fills a table that
  `settings::var(name)` consults *after* the env var. That keeps env flags
  authoritative and never mutates the environment.
* **A malformed step set is an error.** `parse_steps` silently skips a bad
  token (`schedule.py:36-37, :44-45`), which makes a typo mean "never".
  `StepSet` also allows an open tail (`"4-"`), which the Spark ladder needs.
* **Unknown keys are errors** everywhere (techniques, `[kernels]`, `[env]`,
  top level): a typo must not silently mean the default.

### Seams

| seam | exclusive | written by |
|---|---|---|
| `attention_backend` | yes | `dense_attention`, `sol_attn`, `vsa`, `pisa` |
| `step_output` | yes | `teacache` |
| `ffn_precision` | yes | `bf16_linears`, `mxfp8`, `w8a8`, `fp8`, `nvfp4` |
| `activation_precision` | yes | `bf16_activations`, `f32_activations` |
| `video_decoder` | yes | `taeh3`, `taehv` |
| `residency` | yes | `offload` |
| `token_set` | yes | (none yet: token pruning is not ported) |
| `kernel_fusion`, `residual_cache`, `attention`, `hidden_states` | no | `kernel_fusion`, `teacache`, `fp8_attention` (`attention`) |

Two active writers of an exclusive seam are a config error at load:

```
profile x: conflicts:
  - exclusive seam 'attention_backend' has multiple active writers: ["sol_attn", "vsa"]
```

As in sol-engine, two *runtime* techniques on one exclusive seam are allowed
when their `enabled` schedules never overlap (`compose.py:71-75`); anything
involving a load/build transform always clashes. A technique with
`enabled = false` is dropped before the checks and installs nothing.

### Registered techniques

| name | kind | parameters | legacy flag it replaces |
|---|---|---|---|
| `dense_attention` | build | — | `FASTVIDEO_H3_SOL_ATTN=off` on a Sol recipe; `--dense` |
| `sol_attn` | build | `preset` (`rtx` / `engine` / `spark` / `ltx25_stage2`), `tau`, `dense_steps`, `dense_layers`, `sink` (`text` / `prefix` / `suffix`), `thresh_type` (`diag`), `correctness_gate`, `force_dense`, `dense_backend` (`dense` / `vsa`) | `FASTVIDEO_H3_SOL_ATTN=rtx|engine|spark`; LTX `--sol-stage2` |
| `vsa` | build | `sparsity` (a number or a per-step table), `group`, `dense_steps`, `dense_layers`, `dense_sparsity` | `FASTVIDEO_VSA_SPARSITY` (one sparsity everywhere), `FASTVIDEO_VSA_GROUP` |
| `fp8_attention` | build | `ops` (`all` / `dense` / `vsa`, or a list) | `FASTVIDEO_ATTN_FP8` = `0` / `1` / `dense` / `vsa` (opt-in, lossy; H3) |
| `pisa` | build | `sparsity`, `dense_layers` | LTX `--pisa-stage2` |
| `teacache` | on_step | `threshold`, `retain_steps`, `cooldown_steps`, `num_forwards`, `coefficients` | `FASTVIDEO_H3_SOL_CACHE=teacache` |
| `bf16_linears` / `mxfp8` / `w8a8` | load | — | `FASTVIDEO_H3_QUANT=off|mxfp8|w8a8` (H3) |
| `fp8` | load | — | `FASTVIDEO_FP8=1` (W8A8 on every linear; LTX-2, Wan) |
| `nvfp4` | load | `rule` (`static_6`, `static_4`, `mse`) | `FASTVIDEO_NVFP4` |
| `bf16_activations` / `f32_activations` | load | — | `FASTVIDEO_BF16_ACT=1|0` |
| `taeh3` (H3) / `taehv` (LTX-2) | load | `weights` | `FASTVIDEO_TAEH3_WEIGHTS` / `--taeh3-weights`; `FASTVIDEO_LTX2_TAE_WEIGHTS` / `--ltx-tae-weights` |
| `offload` | load | `dit` (`auto` / `resident` / `streamed`), `lookahead`, `placement` (LTX-2: `none` / `cpu`) | `FASTVIDEO_DIT_OFFLOAD`, `_LOOKAHEAD`, `--dit-offload`; `FASTVIDEO_LTX_OFFLOAD`, `--offload` |
| `kernel_fusion` | build | `h3`, `ltx`, `split_rows` (bools) | `FASTVIDEO_H3_FUSE`, `FASTVIDEO_LTX_FUSE`, `FASTVIDEO_SPLIT_ROWS` |

Every technique also takes `enabled`: `true` / `false`, a step set string
(`"0-3"`: on at those steps), `{ before = n }`, `{ from = n }`, or a
per-stage table (`{ stage2 = true }`).

### The step-schedule DSL

A parameter can vary with the step (and stage), and an attention route with
the step and the layer:

```toml
[techniques.sol_attn]
dense_steps = 10          # first 10 forwards dense (an integer is "the first n")
dense_layers = "0-1"      # or a set: points, ranges, an open tail "4-"
tau = 1.0                 # or per step: tau = { 1 = 1.0, 2 = 1.25, 3 = 1.5 }
```

A call is dense when its step is in `dense_steps` **or** its layer is in
`dense_layers`, otherwise Sol-Attn at `tau(step)`. The three hand-written
H3 clocks are three values of it, and a test checks them against the old
code at every step below 64 and every block:

| route | `dense_steps` | `dense_layers` | `tau` | `sink` |
|---|---|---|---|---|
| RTX 5090 cell (`RTX5090/adapter.py:452-461`) | `10` | `2` | `1.0` | `text` |
| Sol-H3 engine (`engine.py` `sparse_attention.install`) | `1` | `2` | `1.0` | `prefix` |
| Spark Ref2VA draft (`stage1_ops/sol.py`) | `"0,4-"` | `1` | `{1 = 1.0, 2 = 1.25, 3 = 1.5}` | `suffix` |
| LTX-2.5 stage 2 (`models/ltx25/RTX5090/attention.py`) | none | `1` | `{0 = 1.0, 1 = 1.25, 2 = 1.5}` | none |

`dense_backend = "vsa"` makes a Sol route's dense calls run the recipe's VSA
(with its trained compression gate) instead of plain dense attention: on the
VSA-distilled FastH3 checkpoints VSA is the trained function, dense attention
is not. It needs a VSA recipe (the DiT must load `to_gate_compress`).

VSA takes the same shape of schedule over its sparsity:

```toml
[techniques.vsa]
sparsity = { default = 0.95, 0 = 0.8 }   # or one number; unlisted steps take default
dense_layers = "0-1,48-49"               # these blocks (and dense_steps) ...
dense_sparsity = 0.8                     # ... run at this sparsity (default 0: every tile)
```

A call on a step in `dense_steps` or a block in `dense_layers` runs at
`dense_sparsity`, otherwise at the step's value, else the default (the
technique's, else the recipe's). The block loop sets the top-k per block
(`H3Vsa::select`); a schedule with no per-step or per-layer entry is the
uniform VSA it replaces (a unit test checks both). `FASTVIDEO_VSA_SPARSITY`
still wins and means one sparsity everywhere.

## Profile format

```toml
[id]                       # required: name; optional: family, description, upstream
name = "h3_rtx5090_fullopt"
family = "minimax_h3"
upstream = "sol-engine config/minimax_h3/rtx5090_fullopt.toml"

[pipeline]                 # required
model = "h3"               # h3 | ltx2
recipe = "sol-h3-rtx"      # optional; --h3-recipe wins (with a log line)

[techniques.sol_attn]      # one table per technique (registry names above)
preset = "rtx"
tau = 1.0
dense_steps = 10
dense_layers = 2
sink = "text"

[techniques.teacache]
threshold = 0.10

[kernels]                  # optional: one implementation per op
dense_attention = "auto"   # auto | nvcc:v1 | nvcc:v2 | cudnn
sol_attention = "x4f"      # auto | v1 | x4 | x4f | ws (sm90+)
vsa_attention = "auto"     # auto | gather | fused | mma | tma | tma2
nvfp4_gemm = "cublas"      # cublas | oxide (sm_100 / sm_120 cubins)
conv3d = "auto"            # auto | cudnn | cudnn-bf16 | nvcc:unfold

[env]                      # optional: raw FASTVIDEO_* settings, lowest level
FASTVIDEO_VSA_TMA = "1"

notes = ["free text"]      # optional
```

Workload values (resolution, duration, seed, prompt) stay on the command
line; a profile carries techniques only, as sol-engine's `[env]` does.

### sol-engine configs

`Profile::parse` recognises a sol-engine config (top-level `id = "..."` or
`model_profile`) and maps the keys whose concepts match
(`profile.rs` `from_sol_engine`):

| sol-engine key | here |
|---|---|
| `H3_RTX5090_PROFILE=dense` | `dense_attention` |
| `H3_RTX5090_PROFILE=sol` / `fullopt` | `sol_attn` (RTX preset); `fullopt` adds `teacache` unless `H3_TEACACHE_ENABLED=0` (`run_minimax_h3_gpu.sh:75-93`) |
| `SOL_ATTN_TAU`, `_THRESH_TYPE`, `_FIRST_DENSE_STEPS`, `_FIRST_DENSE_LAYERS`, `_FORCE_DENSE` | `sol_attn.tau`, `.thresh_type`, `.dense_steps`, `.dense_layers`, `.force_dense` (`adapter.py:452-461, :476-477`) |
| `H3_TEACACHE_THRESHOLD`, `_RETAIN_STEPS`, `_COOLDOWN_STEPS`, `_NUM_FORWARDS`, `_COEFFICIENTS` | `teacache.*` (`teacache.py:23-50, :136`) |
| `official_config.transformer_dtype = "bf16"` | `bf16_linears` |
| `official_config.steps = 50` with an RTX profile | recipe `sol-h3-rtx` (49 forwards) |
| `SOL_ATTN_CORRECTNESS_GATE=1` | recorded in `notes`: the sampled dense-vs-Sol gate is not implemented; Sol runs ungated |
| `H3_FULL_VAE_*` | recorded: the decoders here stay resident unless the DiT streams |
| paths, warmup counts, seeds, prompt files | recorded as harness keys |

A test loads the three sol-engine `rtx5090_*.toml` files (copied verbatim
under `techniques/fixtures/`) and checks that each composes to the same
techniques, parameters and settings as the shipped profile of the same name.

### Shipped profiles

| profile | recipe | techniques |
|---|---|---|
| `h3/rtx5090_dense` | `sol-h3-rtx` | `dense_attention`, `bf16_linears` |
| `h3/rtx5090_sol` | `sol-h3-rtx` | `sol_attn` (RTX), `bf16_linears` |
| `h3/rtx5090_fullopt` | `sol-h3-rtx` | `sol_attn` (RTX), `teacache`, `bf16_linears` |
| `h3/rtx5090_fullopt_taeh3` | `sol-h3-rtx` | + `taeh3` |
| `h3/fasth3_8step` | `8step` | `vsa`, `mxfp8`, `bf16_activations` |
| `h3/fasth3_4step_vsa` | `4step-vsa` | `vsa`, `mxfp8`, `bf16_activations` |
| `h3/fasth3_4step_dense` | `4step-dense` | `dense_attention`, `mxfp8`, `bf16_activations` |
| `h3/sol_h3_4step` | `sol-h3` | `dense_attention`, `mxfp8`, `bf16_activations` |
| `h3/sol_h3_4step_engine` | `sol-h3` | `sol_attn` (engine, prefix sink), `mxfp8`, `bf16_activations` |
| `h3/fasth3_8step_sol` | `8step` | `sol_attn` (tau 1.0, text sink, no dense step or block), `mxfp8`, `bf16_activations` |
| `h3/fasth3_8step_teacache` | `8step` | `vsa`, `teacache` (threshold 1.0, retain 3, cooldown 3: only steps 3-4 can reuse), `mxfp8`, `bf16_activations` |
| `h3/fasth3_8step_sol_teacache` | `8step` | both of the above |
| `ltx2/ltx25_rtx5090_distill_bf16` | (two-stage) | `sol_attn` (`ltx25_stage2`), `bf16_linears`, `offload.placement = "cpu"` |
| `ltx2/ltx25_distill_sol` | (two-stage) | `sol_attn` (`ltx25_stage2`) |
| `ltx2/ltx25_distill_dense` | (two-stage) | `dense_attention` |
| `ltx2/ltx25_distill_sol_taehv` | (two-stage) | `sol_attn`, `taehv` |
| `ltx2/ltx25_distill_sol_fp8` | (two-stage) | `sol_attn`, `fp8` |

`ltx2/ltx25_rtx5090_distill_bf16` is sol-engine's
`models/ltx25/RTX5090/ltx25_rtx5090_distill_bf16.toml` (`LTX25_PIPELINE=bf16`:
Sol stage 2 from `gpu_infer.py`, `--offload cpu` from `run_ltx25_gpu.sh`).
Its `nvfp4` sibling needs the pre-quantized NVFP4 LTX-2.5 checkpoint path,
which is not ported, so it has no profile. LTX workloads (`--workload 4k5s`,
`--two-stage`, the resolution) stay on the command line; `[pipeline] recipe`
is H3-only.

The H3 `rtx5090_*` profiles are sol-engine's configs, which run the transformer
in BF16. The rtx5090 matrix family runs the same recipe with this runtime's
default, MXFP8 on sm_100+; with a profile, `FASTVIDEO_H3_QUANT=mxfp8` gives
that precision back (the env var overrides the profile's `bf16_linears`).

The three `fasth3_8step_*` profiles are the FastH3 8-step arms of the
`h3arms` matrix family (768p and 480p, TAEH3 twins at 480p, each gated with
LPIPS against plain FastH3 8-step on the five-prompt set). Every TeaCache
decision is logged (`h3 teacache step N: compute|REUSE (reason; rel_l1 ..
indicator .. acc ..)`) and recorded in `benchmark.json` under `teacache`.

## The kernel seam

`techniques/kernels.rs`. A `KernelBackend` is one provider of device code
and lists what it implements per `KernelOp`:

| backend | dense attention | Sol attention | VSA attention | NVFP4 GEMM | conv3d |
|---|---|---|---|---|---|
| `nvcc` (our CUDA C++, NVRTC or AOT cubins) | `v1`, `v2` | `v1`, `x4`, `x4f`, `ws` (sm90+) | `gather`, `fused`, `mma`, `tma`, `tma2` (sm90+) | — | `unfold` |
| `oxide` (Tile-IR cubins, `fastvideo-oxide-kernels`) | — | — | — | `oxide` (sm_100 / sm_120) | — |
| `cudnn` | `cudnn` | — | — | — | `cudnn`, `cudnn-bf16` |
| `cublas` | — | — | — | `cublas` (default) | — |

`[kernels] <op> = "<id>"` or `"<provider>:<id>"` is resolved against the
backends (an unknown id, the wrong op, an ambiguous id or a too-old SM is a
config error) and installs the op's setting. The dispatch sites
(`wan::attn::flash_kernel_for`, `wan::ops::sol_kernel_choice`, the VSA
pickers in `wan::ops` / `wan::vsa` / `h3::vsa`, `wan::conv`) read the op
through `kernels::choice(op)`, env first: switching a kernel touches no
pipeline and no model.

## The H3 adapter

`fastvideo_models::h3::techniques::H3Techniques::resolve` turns the recipe,
the profile and the legacy flags into the pipeline's inputs:

| seam | env flag (wins) | profile | recipe default |
|---|---|---|---|
| attention backend | `FASTVIDEO_H3_SOL_ATTN` | `dense_attention` / `sol_attn` / `vsa` | `sol-h3-rtx`: RTX Sol route; VSA recipes: VSA; else dense |
| VSA sparsity / group | `FASTVIDEO_VSA_SPARSITY` / `_GROUP` | `vsa.sparsity` / `.group` | contract / 8 |
| step output | `FASTVIDEO_H3_SOL_CACHE` | `teacache` | off |
| video decoder | `--taeh3-weights`, `FASTVIDEO_TAEH3_WEIGHTS` | `taeh3` (+ `weights`) | official ViT VAE |
| precision, residency, kernels, fusion | their `FASTVIDEO_*` names | the techniques' settings | built-in |

The pipeline (`cudarc::h3::pipeline`) reads the plan instead of the flags:
`H3SolPolicy::from_technique` takes the route and sink, the block loop asks
`route(step, layer)` of the DSL, `H3Transformer::enable_teacache` takes the
technique's controller, and VSA takes the plan's sparsity and group. The
checks the flags had carry over to the profile path: a `suffix` (Spark) sink
outside `sol-h3-spark`, a `prefix` (engine) sink on Ref2VA, `vsa` on a recipe
whose DiT has no compression gate, `taehv` on H3, and `taeh3` without
weights are errors.

## The LTX-2 adapter

`fastvideo_models::ltx2::techniques::Ltx2Techniques::resolve` decides the
stage-2 video self-attention route; precision, the tiny decoder and the
placement reach the pipeline through their settings.

| seam | command line (wins) | profile | default |
|---|---|---|---|
| stage-2 attention | `--sol-stage2` / `--dense-stage2` / `--pisa-stage2` | `sol_attn` (`ltx25_stage2`) / `dense_attention` / `pisa` | Sol on the 2.5 distilled two-stage 3-forward refine (`default_sol_stage2`) |
| ffn precision | `FASTVIDEO_FP8` | `fp8` / `bf16_linears` | bf16 |
| video decoder | `--ltx-tae-weights`, `FASTVIDEO_LTX2_TAE_WEIGHTS` | `taehv` (+ `weights`) | conv VAE |
| placement / residency | `--offload`, `--dit-offload`, `FASTVIDEO_LTX_OFFLOAD`, `FASTVIDEO_DIT_OFFLOAD` | `offload.placement` / `.dit` | `none` / `auto` |

The stage-2 kernels implement exactly the published routes
(`ltx2::sol::route`, `ltx2::pisa::route`), so a profile's `sol_attn` / `pisa`
must describe that route: a different one is refused rather than silently
run as the published one. LTX-2 declares no step-cache capability, so
`teacache` on an `ltx2` profile is a capability error; `taeh3` there, or
`offload.placement` on H3, is an error too.

## Invariants and their tests

| invariant | test |
|---|---|
| no profile = the legacy resolution, for every recipe x `FASTVIDEO_H3_SOL_ATTN` x `FASTVIDEO_H3_SOL_CACHE` x Ref2VA, including the error cases | `h3::techniques::tests::no_profile_is_the_legacy_resolution` |
| LTX-2: no profile = `sol = flag \|\| default_sol_stage2(.., pisa, dense)`, `pisa = flag`, for every flag combination | `ltx2::techniques::tests::no_profile_is_the_legacy_resolution` |
| the `ltx25_stage2` preset is the published stage-2 route at every forward and layer | `ltx2::techniques::tests::the_stage2_preset_is_the_published_route` |
| a technique listed with `enabled = false` resolves exactly like not listing it, and installs no setting | `h3::techniques::tests::disabled_profile_techniques_are_the_baseline`, `techniques::tests::a_disabled_technique_is_not_in_the_plan_and_conflicts_with_nothing` |
| the DSL routes equal the three hand-written clocks at every step and block; the sinks are equal | `h3::techniques::tests::technique_routes_equal_the_policy_routes_everywhere`, `technique_sinks_equal_the_policy_sinks` |
| an inactive Sol route and a never-reusing TeaCache leave every output bit of a 3-step block-stack run unchanged (CPU) | `cudarc h3::transformer::tests::inactive_technique_seams_are_the_dense_forward_bit_for_bit` |
| env beats profile beats default; two techniques cannot set one name differently | `techniques::settings::tests`, `techniques::tests::settings_are_what_the_legacy_flags_were` |
| conflicts, capabilities, unknown names and keys are config errors | `techniques::tests::*` |
| each shipped rtx5090 profile equals its sol-engine config | `techniques::profile::tests::rtx5090_profiles_equal_the_sol_engine_configs` |
| every shipped profile parses, composes and settles | `techniques::profile::tests::every_builtin_profile_parses_composes_and_settles` |
| byte-identical clips before and after, on the GPU | the `techniques` matrix family (below) |

### On the GPU

`FV_FAMILY=techniques` (`scripts/gpu/runpod-matrix.sh`) fetches the
pre-refactor binary from its runtime image (`scripts/gpu/fetch-baseline.sh
<sha>`: the OCI registry API, one 13 MB layer) and runs each cell three times
on one card: the old binary with the matrix cell's command line (`-base`),
this build with the same command line (`-env`), and this build with the
matching profile (`-prof`). `compare-clips --off-identity` must find `-env`
and `-prof` byte-identical to `-base`; `benchmark.json` holds each arm's
timings.

```bash
FV_FAMILY=techniques FV_EXTRA_ENV="FV_BASELINE_SHA=<pre-refactor sha7>" \
  scripts/gpu/runpod-http.sh run <this sha7>
```

Cells: FastH3 8-step 768p (MXFP8 default), FastH3 4-step VSA 768p, the H3
fullopt route (`sol-h3-rtx` + TeaCache, 480p) and LTX-2.5 two-stage 512p with
Sol stage 2 (`-prof`: `ltx2/ltx25_distill_sol`).

### Results (RTX PRO 6000, 2026-09-27)

Baseline: the runtime image of `a73173e` (main just before this layer).
Candidate: `89e1fba` (H3 cells), `5ea271c` (LTX rerun). Single prompt,
seed 1024, warm process; `denoise_s` from `benchmark.json`.

| cell | -env vs -base | -prof vs -base | denoise s (base / env / prof) |
|---|---|---|---|
| FastH3 8-step 768p (MXFP8) | identical, 124/124 frames | identical | 46.21 / 46.36 / 46.30 |
| FastH3 4-step VSA 768p | identical | identical | 19.59 / 19.54 / 19.46 |
| H3 fullopt 480p (`sol-h3-rtx` + TeaCache) | identical | identical (`h3/rtx5090_fullopt` + `FASTVIDEO_H3_QUANT=mxfp8`) | 25.75 / 25.68 / 25.67 |
| LTX-2.5 two-stage 512p, Sol stage 2 | identical, 121/121 | identical (`ltx2/ltx25_distill_sol`) | 4.259 / 4.257 / 4.250 |

"Identical" is `compare-clips --off-identity`: `max_abs_diff_uint8 = 0`
on every frame. Timings agree within 0.4%.

**The LTX-2 conditioning cache is not transparent.** The first LTX run
shared the default text cache between arms: the `-base` process missed and
stored its prompt, the `-env` and `-prof` processes hit that entry, and
every frame of both differed from `-base` (max 238 / 255), with a 14%
slower denoise (5.44 s against 4.78 s). With every arm encoding its own
prompt (`--no-text-cache`, as the family now runs) the three are byte for
byte the same. `ltx2/text_cache.rs` stores the connector outputs as f32 on
the host, and a hit feeds those to the DiT where a miss passes the device
tensors, which is the likely cause (not yet isolated: a process never hits
an entry it stored itself). The technique layer does not touch that code.

### FastH3 8-step technique arms (`h3arms`, RTX PRO 6000, 2026-09-27)

`2c637f7`, five-prompt set (`FV_PROMPTS=5`), medians over the prompts, warm
process, LPIPS(alex) on sol-engine's frame selection, gate
`scripts/gpu/gate-policy.toml` (`lossy`). Quality is against plain FastH3
8-step at the same resolution, from the same run. The 768p baseline's
`total_s` (147.9 s) includes 94.6 s of first-time streamed prompt encoding;
without it, it is 53.6 s, the arms' conditions (every later cell hits the
text cache).

| arm | denoise s | total s | LPIPS | PSNR dB | sharpness | gate |
|---|---|---|---|---|---|---|
| 768p FastH3 8-step (baseline) | 46.52 | 53.6 | — | — | — | — |
| 768p + Sol (dense nowhere) | 40.65 | 48.16 | 0.701 | 10.4 | 0.63 | fail |
| 768p + TeaCache (steps 3-4 reused) | 36.01 | 43.74 | 0.383 | 14.1 | 0.97 | pass |
| 768p + Sol + TeaCache | 30.78 | 38.20 | 0.706 | 10.4 | 0.65 | fail |
| 480p FastH3 8-step (baseline) | 16.48 | 19.95 | — | — | — | — |
| 480p + Sol | 13.30 | 16.71 | 0.670 | 10.7 | 0.61 | fail |
| 480p + TeaCache | 12.50 | 15.89 | 0.434 | 13.3 | 0.98 | fail (one prompt's sharpness 0.944 < 0.95) |
| 480p + Sol + TeaCache | 10.21 | 13.61 | 0.673 | 10.7 | 0.60 | fail |

TAEH3 decode at 480p, against its own twin (the decoder's cost alone) and
against the plain baseline:

| arm | denoise s | total s | vs twin: LPIPS / PSNR / sharpness | vs baseline: LPIPS / PSNR / sharpness |
|---|---|---|---|---|
| Sol + TAEH3 | 13.22 | 14.28 | 0.149 / 27.9 / 0.93 | 0.669 / 10.7 / 0.56 |
| TeaCache + TAEH3 | 12.49 | 13.93 | 0.125 / 25.2 / 0.84 | 0.453 / 13.5 / 0.83 |
| Sol + TeaCache + TAEH3 | 10.03 | 11.15 | 0.145 / 27.6 / 0.93 | 0.672 / 10.7 / 0.55 |
| FastH3 4-step VSA 480p | 7.53 | 10.92 | — | — |
| FastH3 4-step VSA 480p + TAEH3 | 7.50 | 8.72 | 0.123 / 26.3 / 0.87 (twin = baseline) | |

Every TeaCache run reused exactly steps 3 and 4 (rel-L1 of the block-0
modulated input 0.08 per step, accumulator 0.16 against the 1.0 threshold).
Sol with no dense step or block is not viable on the VSA-distilled 8-step
checkpoint (LPIPS 0.67-0.71); TeaCache on the middle two steps passes the
768p gate at a 23% denoise saving but moves the clip visibly (LPIPS 0.38).
TAEH3 fails the gate's sharpness floor everywhere (0.84-0.93 against its
twin) and saves decode time only (`total_s` -2.2 to -2.5 s at 480p).

## Adding a technique

1. A parameter struct in `techniques/methods.rs` implementing `Technique`
   (name, kind, seams, capabilities, `enabled`, and `settings()` if deep code
   reads a process-wide knob).
2. A factory in `registry::TECHNIQUES` that parses its table with `Params`
   (unknown keys are errors).
3. Where the model reads it: either the setting it installs (nothing else to
   do), or the model adapter (`h3/techniques.rs`) takes it from the plan.
4. Tests: its OFF (`enabled = false`) resolves to the baseline, and its
   conflicts are rejected.
