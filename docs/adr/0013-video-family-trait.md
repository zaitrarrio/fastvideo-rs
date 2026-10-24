# ADR-0013: One `VideoFamily` trait, a static family list and a common `Request`

- Status: accepted (plan B2, `docs/plan/2026-10-23-execution-plan.md`)
- Date: 2026-10-24
- Source: `docs/gaps/2026-10-22-architecture-review.md` §4.1, §4.2, §7.3 item 2

## Context

Before B2, `crates/fastvideo-core` defined a family in four disconnected
places: a `ModelFamily` variant, one of 21 static tables in `registry.rs`, an
arm in a 22-branch `resolve()` and a loop in `all_registered_ids()`; then a
`ModelFamily::<Fam> => generate_<fam>(def, opts)` arm in `av_generate.rs`
and one of 20 copy-pasted `generate_<fam>` functions reading a 35-field
`AvGenerateOptions` in which half the fields belonged to LTX or H3. The CLI
branched on `ResolvedModel::Wan` against `ResolvedModel::Family` and kept two
copies of the argument hand-off. A mismatch between any two places was a
runtime error.

## Decision

1. **`VideoFamily`** (`crates/fastvideo-core/src/family/mod.rs`): `name()`,
   `variants()`, `resolve(model_id) -> Option<Variant>` (with a default
   `resolve_exact`), `detects_last()`, `supports_tiny()`, `requires_cuda()`
   and `load(variant, &AvGenerateOptions) -> Box<dyn Pipeline>`.
   **`Pipeline`**: `describe(&Request)` (the line the CLI prints) and
   `generate(&Request, Hooks) -> Output`.
2. **One static list**, `FAMILIES: &[&dyn VideoFamily]` (21 families, the
   old resolution order). `resolve()` walks it in three passes that keep the
   old semantics exactly: the exact Hub ids of the families that detect last
   (Wan), every other family (exact id, then its detectors), then the late
   families' detectors. `registered()` replaces `all_registered_ids()`.
3. **One module per family** (`family/<fam>.rs`) holds its variant table (a
   `HubVariant` per preset; Wan keeps its `WanModelDefinition`, which also
   carries the sampler) and its run. The registry tables, `resolve()`'s arms,
   `all_registered_ids()`, `generate_av` and the 20 `generate_*` functions
   are deleted with `registry.rs` and `av_generate.rs`.
4. **A common `Request`** (`crates/fastvideo-core/src/request.rs`):
   `prompt`, `negative`, `seed`, `canvas { width, height }`,
   `timing { frames, seconds, fps }`, `sampler { steps, guidance,
   guidance_2 }`, `conditioning: Vec<Conditioning>` with
   `Conditioning::{Image, Keyframes, Refs, Control, Audio}` (text only = no
   conditioning). `None` means the variant's default; the family fills it in.
   **`Request::validate(&ModelCaps)`** checks the tasks the conditioning asks
   for, the canvas (multiple, aspect, pixel budget per short edge), the frame
   grid, the duration, the fps, the reference counts by kind and the knobs
   the request sets.
5. **`AvGenerateOptions` keeps only the common fields**: model id, weights,
   output, device, mp4, plus `FamilyOptions { wan: WanOptions, ltx2:
   Ltx2Options, h3: H3Options }` for the flags only one family reads (Wan:
   backend, GPUs, tiny, dtype; LTX: two-stage, DiffVAE, refine steps, the
   stage-2 routes, HQ, DiT root; LTX and H3: text root, DiT residency, TAE;
   H3: recipe, adapter). `--seconds` became `Request.timing.seconds`.
6. **Hooks** reach the pipelines that take them: Wan
   (`VideoGenerator::generate_video_with_hooks`), LTX-2 and H3
   (`generate_with_hooks`). `Hooks::NONE` is exactly the previous call.

## Why these shapes

- **`conditioning` is a list, not one enum value.** The CLI accepts
  `--image`, `--last-image`, `--ref` and `--control` together and each family
  has always taken the inputs it understands and ignored the rest (LTX with
  `--image --ref` runs I2V; H3 refuses FL2VA with Ref2VA; Wan Fun Control
  reads `--control`, else `--image`). A single enum value would have to drop
  inputs before the family sees them and change those behaviours.
  `validate` is where contradictory combinations are refused for a model
  with caps.
- **The CLI does not call `validate`.** Its presets have no caps: only the 30
  served variants (`families/*.toml`, projected to `ModelCaps` by the
  engine) do, and calling it would narrow what the CLI accepts (the tiny CPU
  goldens run at 64x64x5, below every served frame grid). The serve path is
  the user: wiring it into `engine-service`'s `validate.rs` is E2's (that
  file and `fastvideo-protocol` are E2's in this round).
- **The Hub-id tables stay in Rust, one per family module.** The family
  manifests describe served variants (`fasth3-4step-vsa`, `ltx25-distill-sol`,
  ...), not the CLI's 63 Hub presets, and only Wan's variants name a preset.
  Moving the Hub ids into `families/*.toml` is a layer-2 widening for the
  manifest's owner (C1 holds `family.rs` and the family files now); the trait
  does not change when it happens: `VideoFamily::variants` reads the manifest
  instead of a static.
- **`load` plans, `generate` loads.** Every family but Wan loads and
  generates in one call today (`H3Pipeline::load` is one ~830-line function;
  B6 splits it into plan, open, place, fuse). `load` therefore records the
  variant and options (`Planned`), and the family's run opens the weights on
  `generate`. The CLI's observable order is unchanged: the `family=...` line
  prints before any device or weight error.
- **`fastvideo-core` depends on `fastvideo-protocol`** (a workspace crate, no
  new external package) for `ModelCaps` and `Task`; the CLI reaches `Hooks`
  through a `fastvideo-core` re-export.

## Consequences

- A new family touches one module in `crates/fastvideo-core/src/family/` and
  one line in `FAMILIES` (plus its runtime port and oracle);
  `docs/ports/CHECKLIST.md` says what remains.
- CLI behaviour is pinned by goldens taken before the change and unchanged
  after it: `crates/fastvideo-cli/tests/cli_golden.rs` (`--help` of every
  command, `list-models`, `generate` and `schedule` for every Hub id and
  preset on a CPU build, tiny CPU generations hashed per frame) and
  `crates/fastvideo-core/tests/resolve_golden.rs` (459 inputs -> family,
  preset, workloads, line, Wan sampling defaults). 83 Hub ids (82 distinct)
  resolve to their family (`family::tests`).
- Only Wan has a CPU generation path (`--tiny`); the 20 other families need
  CUDA, so their generations are GPU cells, not CPU goldens. Their CUDA code
  was moved verbatim behind the trait and type-checks with
  `--features cuda`.
- `ModelFamily` is gone; a family is named by `VideoFamily::name()`.
