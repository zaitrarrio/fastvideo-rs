# ADR-0006: model.toml v1

- Status: Accepted
- Date: 2026-10-23
- Decision-id: FVID-2026-10-23-model-toml-v1
- Crate: `fastvideo-manifest`, module `model_toml`; schema
  `crates/fastvideo-manifest/schemas/model-toml.v1.json`

## Context

A clone from the registry is three things: verified weights, one executable
and one config file, `model.toml`. The registry's WO-001 pinned v1 of that
file (`schema`, `name`, `parent`, `license`, `[runtime]`, `[defaults]`,
`[components]`, `[change]`). The owner's decision of 2026-10-23 moved the
format here (ADR-0005 "Context") and, because a model served by this
runtime varies on more than distillation, step count and text-encoder
precision (review §6: attention route, DiT quantisation, decoder, LoRA
adapters, stage count, resolution tiers, and the ranges the engine
negotiates against), widened the file **before the first publish**, when a
widening is still free. After the first publish a change is a versioned
schema (`schema = 2`), never an edit.

## Decision

`ModelToml` is the WO-001 shape, verbatim, with every table
`deny_unknown_fields`:

```toml
schema = 1
name = "zay/wan2.2-t2v/dmd:4:fp8"          # the published name, without @rev
parent = "wan-ai/wan2.2-t2v@0123456789ab"  # forks only; MUST carry @rev; same model id
license = { spdx = "Apache-2.0", redistribution = true, url = "https://..." }   # url optional

[runtime]
engine = "fastvideo-rs"
min_version = "0.1.0"                      # semver
pipeline = "wan"                           # family id (layer 2); open set, validated by the runtime

[defaults]                                 # every key optional; floats allowed here
steps = 4
guidance = 1.0
width = 832
height = 480
frames = 81
fps = 16

[components]                               # at least one; paths relative, '/' separators, no '..'
transformer = { path = "weights/transformer.safetensors", dtype = "bf16" }

[change]                                   # forks only (required)
summary = "Distilled to 4 steps"
reason = "Drafts in a fraction of the time"
```

Rules enforced by `ModelToml::parse` (each error carries a dotted field
path, `ModelTomlError::path()`): `schema` must be 1; `name` must parse
(ADR-0005) and must not carry `@rev`; a fork has a `parent` with `@rev`
and the same model id as its own name, and a `[change]`; a vendor ref has
neither; `license.spdx`, `runtime.engine`, `runtime.pipeline` are
non-empty; `min_version` is semver; component paths are valid in-tree
paths; positive defaults. Unknown keys anywhere are
`UnknownKey { path }` (`runtime.engin`, `components.vae.precision`,
`caps.frames.default`); type errors are `Field { path, msg }`. Serialising
and re-parsing gives an equal value.

**The two widenings (owner decision 2026-10-23):**

1. `[variant]`: what the name grammar does not carry. All keys optional:
   `attention_route`, `dit_quant`, `decoder` (strings), `adapters`
   (array of strings), `stages` (positive integer), `tiers` (array of
   strings). `[runtime]` gains two optional pointers, `recipe` (the runtime
   recipe id) and `profile` (the technique profile path), so a `model.toml`
   can name exactly the configuration the engine runs.
2. `[caps]`: the ranges the engine negotiates against, mirroring the
   shapes of `crates/fastvideo-protocol/src/caps.rs` without depending on
   that crate: `frames = { step, offset, min, max }` (counts
   `offset + step*k` within `min..=max`; `min` and `max` on the grid,
   `min <= max`), `canvas = { multiple, max_area, aspect = [min, max],
   short_edges }`, `fps = { allowed, default }` (`default` in `allowed`),
   `refs = { images, videos, audios, total }` (`total` at least each),
   `knobs = { seed, negative, steps, guidance, guidance_2, flow_shift }`
   (booleans, default false). Every sub-table is optional; `[caps]` is
   independent of fork-ness.

Why widen rather than extend the name: the name is the identifier people
type and the registry indexes; the extra dimensions are properties of a
variant that the engine and the control plane read, and they would make
names unreadable. Why `[caps]` in the public file: it is what lets the
registry's model page and a fresh pod know what a clone can do without
loading it, and it is what layer 2 (the family manifest, A2) checks a
`model.toml` against.

Floats are allowed in `model.toml` (`guidance`, `aspect`); they are never
allowed in manifests (ADR-0007), which is why `model.toml` is referenced
from the manifest by the BLAKE3 of its exact bytes rather than embedded.

## Consequences

- One parser, one schema: the registry's `gia`, the runtime's `fv-boot`
  and the engine projection (A3) read the same `ModelToml`.
- `[variant]`, `[runtime].recipe/profile` and `[caps]` are optional, so a
  minimal vendor `model.toml` is still the WO-001 example. A2/A3 decide
  which of them a served variant must fill.
- The `.expect` fixtures make every error path a regression test; the JSON
  Schema (`schemas/model-toml.v1.json`, `additionalProperties: false`
  everywhere) is committed and drift-checked, so editors and the registry
  API validate against the same strictness.
- Open question for the registry owner (not decided here): whether
  `[defaults]` should gain the knobs `[caps].knobs` names (`guidance_2`,
  `flow_shift`, `negative`, `seed`); today it has the six WO-001 keys.

## Tests

- `crates/fastvideo-manifest/tests/model_toml.rs`:
  `valid_fixtures_parse_and_round_trip` (every
  `fixtures/model-toml/valid/*.toml`, through `toml` and through JSON),
  `invalid_fixtures_fail_with_kind_and_path` (every
  `fixtures/model-toml/invalid/*.toml` against its `.expect`, 33 cases),
  `unknown_keys_anywhere_name_their_path`, `widenings_parse`
  (`fork-widened.toml`: `[variant]`, `recipe`/`profile`, every `[caps]`
  table), `caps_min_greater_than_max_is_rejected`,
  `the_wo001_example_is_a_fork_with_a_pinned_parent`.
- `crates/fastvideo-manifest/tests/schema.rs`:
  `model_toml_schema_has_not_drifted`,
  `schemas_are_strict_and_name_the_pinned_fields`.
- `crates/fastvideo-manifest/src/model_toml.rs` unit tests.
