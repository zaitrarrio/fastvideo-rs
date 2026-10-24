# Family manifests (layer 2)

One file per served model family, one `[[variants]]` table per servable
configuration: the superset the engine catalog, the serve configs and the
deployment tooling are generated from (review
`docs/gaps/2026-10-22-architecture-review.md` §7.2; ADR
`docs/adr/0008-family-manifest-layer-2.md`). Types and validation:
`crates/fastvideo-manifest/src/family.rs`; schema
`crates/fastvideo-manifest/schemas/family.v1.json`.

| File | Family | Port doc |
|---|---|---|
| `h3.toml` | MiniMax-H3 / FastH3 | `docs/ports/h3.md` |
| `ltx2.toml` | LTX-2 / 2.3 / 2.5 | `docs/ports/ltx2.md` |
| `wan.toml` | Wan 2.1 / 2.2 / FastWan | `docs/ports/wan.md` |

## Status (plan A2)

The files are **generated** from the hand-written engine catalog
(`crates/fastvideo-engine-service/src/cuda/caps.rs` `catalog()`) and tests
prove they carry every row: `crates/fastvideo-engine-service/tests/families.rs`
rebuilds the catalog from them (`catalog_from_manifests`) and compares row for
row, checks the id sets agree and that every `(family, tier)` alias has a
variant. Until plan A3 deletes `catalog()` and makes these files the source,
do not hand-edit them: change the catalog and regenerate.

```sh
# on the build pod (docs/dev/build-pod.md)
bash scripts/dev/build-pod.sh run <agent> -- FV_FAMILIES_WRITE=/root/fvb/target/<agent>/gen \
  cargo test -p fastvideo-engine-service --test families regenerate_family_files -- --ignored
bash scripts/dev/build-pod.sh fetch <agent> gen/families/h3.toml   # and ltx2, wan
```

## Reading a file

- `id` is the engine catalog id; `served_names` are the other names it
  answers to; `tier` is `draft` / `turbo` / `max` or absent (untiered,
  addressed by id).
- `trees` names the weights the variant reads as
  `scripts/gpu/weights-manifest.tsv` names them (relative to the weights
  root). It includes trees the recipe reads without naming a path (the
  FastH3 Preview adapters beside `h3-base`, the Trim NVFP4 file, the Turbo
  LoRAs, the LongLive-Plug adapters, the TAEHV decoder a `tae_dir` holds).
- `gpu`: `min_vram_gib` is the card class the variant is served on
  (`control/src/presets.ts`), `min_sm` the lowest compute capability as
  `major * 10 + minor`, `needs` the hardware features (`fp8`, `nvfp4`).
- `caps` is layer 1's `[caps]` (`model.toml`), the ranges the engine
  negotiates against.
- `engine` is the engine's recipe (`CudaRecipe`) in its serde form, opaque to
  the manifest crate; its `family` key is the recipe kind (`h3`, `ltx2`,
  `wan`, `sf_wan`), not the family id. Paths are relative to the weights layout:
  `${weights}/<tree>/...` for anything under the weights root and
  `${tae}/<file>` (or `${tae}` for the directory) for the tiny autoencoders;
  the engine substitutes its `WeightLayout` at load
  (`cuda::families::recipe_from_engine`).
- `registry_name` (absent for now) will be the layer-1 name each variant
  publishes as once the registry holds it.
