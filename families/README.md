# Family manifests (layer 2)

One file per served model family, one `[[variants]]` table per servable
configuration: the source the engine catalog, the serve configs and the
deployment tooling are projected from (review
`docs/gaps/2026-10-22-architecture-review.md` §7.2; ADR
`docs/adr/0008-family-manifest-layer-2.md`, amended by plan A3). Types and
validation: `crates/fastvideo-manifest/src/family.rs`; schema
`crates/fastvideo-manifest/schemas/family.v1.json`.

| File | Family | Port doc |
|---|---|---|
| `h3.toml` | MiniMax-H3 / FastH3 | `docs/ports/h3.md` |
| `ltx2.toml` | LTX-2 / 2.3 / 2.5 | `docs/ports/ltx2.md` |
| `wan.toml` | Wan 2.1 / 2.2 / FastWan | `docs/ports/wan.md` |

## Status (plan A3): these files are the source

The engine has no hand-written catalog. `fastvideo-engine-service` embeds
these three files at build time (`cuda::families::EMBEDDED_FAMILIES`) and
projects every variant onto an engine model (`cuda::families::catalog`);
`FV_FAMILIES_DIR=<dir>` makes a running `fv-serve` / `fv-gpucheck` read a
directory of `*.toml` instead. A serve config (`configs/serve/*.toml`) names
a variant and says how the deployment serves it:

```toml
[[models]]
variant = "h3-turbo"          # a variant id, one of its served names, or a tier alias
id = "fasth3"                 # optional: the id this deployment serves it under
resident = true
warmup = true                 # deployment overrides: weights, warmup, text_encoder,
                              # i2v_encoder, h3_1080p, steps, lora_strength,
                              # longlive, longlive_recache, continuity, ...
```

With `[engine] backend = "fake"`, the same `[[models]]` are served by the
fake engine with their real caps, recipes and job checks (CPU tests run a
GPU config unchanged but for the backend). There is no `[aliases]` table,
no `family` and no `recipe` key any more
(the config loader refuses them and names the replacement): the public API
names come from `public_names`, the tier aliases from `tier`, the recipe
from `engine`.

## Changing a family

Edit the file by hand, then on the build pod (`docs/dev/build-pod.md`):

```sh
bash scripts/dev/build-pod.sh run <agent> -- cargo test --locked -p fastvideo-manifest -p fastvideo-engine-service
bash scripts/dev/build-pod.sh run <agent> -- cargo test --locked -p fastvideo-serve --test capabilities_golden
```

- `fastvideo-manifest` validates the file (ids, names, trees, GPU needs,
  caps, serving, placeholder rules) and the schema drift.
- `fastvideo-engine-service` `tests/families.rs` checks that the embedded
  copies are the committed files, that every `engine` table is a recipe
  under any weight layout, that served and public names resolve, and that
  `trees` lists what the recipe reads.
- `capabilities_golden` renders `/fv/v1/capabilities` and `/v1/models` for
  every shipped serve config. A deliberate API change regenerates
  `crates/fastvideo-serve/tests/goldens/capabilities.json`
  (`FV_GOLDEN_WRITE=<file>`, fetched from the pod) and is listed in the PR.

A new variant is a new `[[variants]]` table; give it the trees it reads
(they must exist in the weights store), its `gpu` needs, its layer-1 `caps`
and its `serving` table, and the `engine` table in the serde form of the
engine's `CudaRecipe`. The control plane and the scripts are generated from
these files (plan A4, `control/gen-configs.mjs`).

## Deploying a variant: `deploy/deploy.toml`

These files say what a variant is; how it is deployed lives in
`deploy/deploy.toml` (plan C1, [ADR-0011](../docs/adr/0011-deploy-section.md)),
not here: the engine embeds these files, and the deployment unit is a
preset (one pool: a set of variants on one image variant with one worker
config, queue and timeouts), not a variant. To serve a new variant, add a
`[[presets]]` table there whose `models` name its variant id (and a
`configs/serve` worker config naming it), then run `node
control/gen-configs.mjs`: the presets, image and preset tables, the GCP rows,
the autoscale pools and the Dockerfile stages are generated from that file.
`crates/fastvideo-manifest` (module `deploy`) checks it against these
files: every variant id exists, every recipe name resolves to its variant,
a family preset's card class meets its variants' `gpu.min_vram_gib`, the
only volume is the EU one.

## Reading a file

- `id` is the variant id (the engine model id unless a serve config renames
  it); `served_names` are the other names it answers to; `public_names` are
  the public API names a server maps onto it when it serves it (the MiniMax
  `model` ids). A public name several served variants claim goes to the one
  that serves text-to-video (the tier rather than its task companion), else
  the first by id. `tier` is `draft` / `turbo` / `max` or absent (untiered,
  addressed by id).
- `trees` names everything the variant reads, as the weights store names it
  under the weights root: a `scripts/gpu/weights-manifest.tsv` row, or a tree
  that file documents without a Hub row (`h3-ref2va`, fetched from two
  repositories by `scripts/gpu/fetch-h3-ref2va.sh`;
  `longlive-1.3b-safetensors`, converted on the volume; both are
  `scripts/gpu/weights-r2.tsv` rows). It includes trees the recipe reads
  without naming a path: the FastH3 Preview adapters beside `h3-base`, the
  Trim NVFP4 file, the Turbo LoRAs, the LongLive-Plug adapters, the TAEHV
  file a Wan `tae_dir` holds.
- `gpu`: `min_vram_gib` is the card class the variant is served on
  (`control/src/presets.ts`), `min_sm` the lowest compute capability as
  `major * 10 + minor`, `needs` the hardware features (`fp8`, `nvfp4`); the
  engine's startup device check reads `needs`.
- `caps` is layer 1's `[caps]` (`model.toml`): frame grid, canvas, fps,
  references, knobs.
- `serving` is what the runtime advertises beside it (layer 2 only): the
  tasks, the default frame count, native audio, a container-only fps, the
  pad-and-crop canvas, the H3 1080P tier (`hd`) and the reference-strength
  knobs. A deployment can switch the 1080P tier off (`h3_1080p = false`, or
  the device memory gate); the streaming shape follows the recipe.
- `engine` is the engine's recipe (`CudaRecipe`) in its serde form, opaque to
  the manifest crate. Its `family` key is the recipe kind (`h3`, `ltx2`,
  `wan`, `sf_wan`), not the family id: an SF-Wan causal rollout is a `wan`
  family variant with an `sf_wan` recipe. Paths are relative to the weights
  layout: `${weights}/<tree>/...` for anything under the weights root (a
  `trees` entry must cover it; `${weights}` alone is refused) and
  `${tae}/<file>` for a tiny autoencoder (`trees` must list
  `auxiliary/tae/<file>`); `${tae}` alone is the directory as a search path.
  The engine substitutes its `WeightLayout` at load
  (`cuda::families::recipe_from_engine`).
- `registry_name` (absent for now) will be the layer-1 name each variant
  publishes as once the registry holds it.
