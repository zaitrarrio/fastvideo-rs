# ADR-0008: The family manifest, layer 2

- Status: Accepted
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-family-manifest-layer-2
- Crate: `fastvideo-manifest`, module `family`; schema
  `crates/fastvideo-manifest/schemas/family.v1.json`; files `families/*.toml`;
  engine projection `crates/fastvideo-engine-service/src/cuda/families.rs`

## Context

The architecture review (`docs/gaps/2026-10-22-architecture-review.md` §4.1)
found model identity defined in nine places and proposed one family manifest
as the source everything else is generated from (§7.2). Layer 1, the
registry's public formats, shipped as plan A1 (ADR-0005..0007) and is
pinned. Layer 2 is the runtime's own superset: `families/<family>.toml`,
one file per served family, one `[[variants]]` table per servable
configuration; a published model's `model.toml` is the per-clone projection
of one variant of it.

Plan A2 writes the types and the first three files (`h3`, `ltx2`, `wan`) and
proves, by an equality test, that they carry every row of the hand-written
engine catalog (`cuda/caps.rs` `catalog()`). A3 then makes the files the
source and deletes the catalog; A4 generates the control plane and the
scripts from them. This ADR records the shape and three choices.

## Decision

### Shape

```toml
schema = 1

[family]
id = "h3"                              # layer 1's runtime.pipeline; lowercase, digits, `-`
display = "MiniMax-H3 / FastH3"
licence = "MiniMax-H3 Community License"
port_doc = "docs/ports/h3.md"

[[variants]]
id = "fasth3-4step-vsa"                # the engine catalog id; everything else derives from it
registry_name = "minimax/h3/dmd:4:bf16" # optional: the layer-1 name it publishes as (none yet)
tier = "turbo"                         # draft | turbo | max; absent when untiered
served_names = ["fasth3"]              # names besides the id
resident_default = true
recipe_name = "fasth3-4step-vsa"       # the short recipe name the capabilities report
trees = ["FastH3-4-step-Preview-v1-LoRA", "h3-base"]   # weights-manifest.tsv names, sorted

[variants.gpu]
min_vram_gib = 80
min_sm = 89                            # major * 10 + minor
needs = ["fp8"]

[variants.caps]                        # layer 1's `Caps`, reused unchanged
# frames, canvas, fps, refs, knobs

[variants.engine]                      # opaque to the crate: the runtime's recipe
family = "h3"
recipe = "4step-vsa"
weights = "${weights}/h3-base"
# ...
```

Rules enforced by `FamilyManifest::parse` / `validate` (every error names a
dotted path, `FamilyError::path()`): `schema` is 1; the header is complete;
variant ids and served names are unique across the file (and, in a
`FamilySet`, across every file of a tree); a served name never repeats its
variant's id; `registry_name` carries no `@rev`; `trees` is non-empty,
sorted, unique, each entry a relative `/`-separated path (the same rule as
layer 1's component paths); `gpu.min_vram_gib > 0`, `gpu.min_sm` is a
two- or three-digit compute capability; `caps` passes layer 1's
`Caps::validate`; `engine` is a non-empty table whose every
`${weights}/<rel>` reference is covered by an entry of `trees` (equal to it
or a component prefix of it).

### Choice 1: the pure types and the engine's table are split

`fastvideo-manifest` knows the variant's identity, tier, trees, GPU needs
and caps, and treats `engine` as an opaque TOML table. The engine
(`fastvideo-engine-service`) owns what is inside it: today the serde form of
`CudaRecipe`, deserialised by `cuda::families::recipe_from_engine`. The
crate stays free of engine types and CUDA, so the registry, `fv-boot`,
`fv-lab` and the control-plane generator read a family file without
depending on the inference core; the engine can change its recipe structs
(plan B2) without a format change here. The price is that the crate cannot
validate the inside of `engine`; the engine's tests do (every committed
file deserialises and equals the catalog).

### Choice 2: trees are names, not paths

A family file names what a variant reads as weights trees, exactly as
`scripts/gpu/weights-manifest.tsv` names them (`h3-base`,
`longlive-plug/wan21-t2v-14b-cfg`, `auxiliary/tae/taeh3.safetensors`), and
never where they are. Inside `engine`, every path under the engine's
`WeightLayout` is written with a placeholder: a path under `tae_dir`
becomes `${tae}/<rest>` (`${tae}` alone for the directory), any other path
under `root` becomes `${weights}/<rest>`; the engine substitutes its own
layout when it loads the table. One file therefore serves every machine
(Runpod volume, R2 hydration, a laptop), the control plane can plan a
weights fetch from `trees` alone, and the registry can check that every
referenced tree resolves to a published manifest (review §7.2). The
equality test runs under a synthetic layout with a separate TAE directory
to prove nothing in the files depends on the generating machine.

### Choice 3: the one directory reader is a feature

The crate's contract is "no I/O"; layer 2 needs one directory reader every
runtime consumer shares. `family::load_dir` exists behind the default-off
cargo feature `fs`, which the engine enables and the registry never does.
`FamilySet::parse_all` takes `(label, text)` pairs for callers that read
their own files.

### What A2 leaves to A3

- `registry_name` is `None` in every variant: no model is published yet.
- `caps` is the layer-1 subset of the engine's `ModelCaps`: the H3 1080P
  tier's own pixel budget, the container-only fps flag, the task list and
  the audio caps stay derived by the engine until A3 decides whether layer 1
  widens or layer 2 carries them beside `caps`.
- `gpu.min_vram_gib` is the card class of `control/src/presets.ts`, not a
  measured peak; `fv-lab` gold files (D2) will replace it with measurements.
- The first files are generated from `catalog()` by
  `crates/fastvideo-engine-service/tests/families.rs`
  (`regenerate_family_files`), never hand-typed, and a test keeps them equal
  to the generator until A3 inverts the direction.

## Consequences

- One more schema under drift check: `schemas/family.v1.json`.
- `fastvideo-engine-service` depends on `fastvideo-manifest` (with `fs`)
  and `toml`; the lockfile gains no new package.
- Layer 2 is not a public format: `schema` versions it, and a change is a
  PR with this ADR amended, not a breaking change for published models.
- Enforced by: `crates/fastvideo-manifest/src/family.rs` unit tests,
  `tests/schema.rs`, `crates/fastvideo-engine-service/src/cuda/families.rs`
  unit tests and `tests/families.rs` (projection equality row for row,
  file-generator equality, id-set equality, tier-alias coverage).
