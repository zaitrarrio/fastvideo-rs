# fastvideo-manifest

The registry's public formats, layer 1 of the model family manifest
(`docs/gaps/2026-10-22-architecture-review.md` §7.2; plan
`docs/plan/2026-10-23-execution-plan.md` A1):

| Module | What | ADR |
|---|---|---|
| `name` | `ModelName`: `namespace/model[@rev]`, `namespace/model/variant[@rev]`, `variant = distillation:steps:encoder` | [0005](../../docs/adr/0005-model-naming-grammar.md) |
| `model_toml` | `ModelToml` v1: the clone's config, plus the `[variant]`, `[runtime].recipe/profile` and `[caps]` widenings | [0006](../../docs/adr/0006-model-toml-v1.md) |
| `manifest`, `canonical` | `Manifest` v1 and `canonical_digest = blake3(JCS(manifest without signatures))`, integers only | [0007](../../docs/adr/0007-manifest-digest-lineage-signatures.md) |
| `sig` | `KeyRing`, `sign`, `verify`: Ed25519 over `gia-manifest-v1\0 || digest` | 0007 |
| `lineage` | `verify_chain` over `[vendor, fork1, fork2, ...]`; `verify_lineage` with each clone's `model.toml` | 0007 |
| `schema` | JSON Schema export, committed under `schemas/` and drift-checked | 0006, 0007, 0008 |
| `family` | Layer 2: `FamilyManifest` / `VariantManifest`, `families/<family>.toml`, the runtime's superset the engine catalog and the deployment tooling are generated from; `engine` is an opaque table the runtime interprets | [0008](../../docs/adr/0008-family-manifest-layer-2.md) |

Pure: no I/O (callers read the files and pass bytes), no CUDA, no other
fastvideo crate. Dependencies: serde, serde_json, toml, blake3,
ed25519-dalek, base64, thiserror, semver, schemars; proptest for tests.
The default-off feature `fs` adds `family::load_dir`, the one directory
reader (`families/*.toml`) the runtime consumers share; the registry never
enables it. Layer 2 is this repository's format, not a public one: `schema`
versions it (ADR-0008).

## Who depends on it

`zaitrarrio/model-registry-rs` (the Gia Labs Registry) depends on this crate
as a pinned git dependency and never re-implements a format; its WO-001 runs
the fixtures under `fixtures/` as its acceptance tests. Here, layer 2
(`families/*.toml`, plan A2, module `family`), the engine projection
(`fastvideo-engine-service`'s `cuda::families`, A2/A3), `fv-boot` and
`fv-lab` build on it.

## The pinned-contract rule

These formats are public: once a model is published, a change to the
grammar, to `model.toml` or to the manifest breaks every published model
(review §7.10). So: every widening happens **before the first publish**,
with its ADR and fixtures; after the first publish a change is a versioned
schema (`schema = 2`), never an edit. If the crate and the registry's
specification disagree, the fix is a PR here, with the fixture that shows
the disagreement, not a fork of the format in the registry.

## Fixtures

`fixtures/README.md`. The manifest, key and lineage fixtures are produced by
an independent Python oracle (`fixtures/gen_fixtures.py`), so the tests are a
cross-implementation check. The keys under `fixtures/keys/` are TEST ONLY.

## Build and test

On the shared build pod (`docs/dev/build-pod.md`), never in an agent
container:

```sh
bash scripts/dev/build-pod.sh run fastvideo-manifest -- cargo test -p fastvideo-manifest
bash scripts/dev/build-pod.sh run fastvideo-manifest -- cargo clippy -p fastvideo-manifest --all-targets -- -D warnings
bash scripts/dev/build-pod.sh run fastvideo-manifest -- cargo fmt -p fastvideo-manifest -- --check
```

To regenerate `schemas/*.json` after a type change, run the tests with
`FV_MANIFEST_WRITE=<dir>` and copy `<dir>/schemas/*.json` back (see
`tests/schema.rs`).
