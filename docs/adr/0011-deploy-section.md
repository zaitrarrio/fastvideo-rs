# ADR-0011: The deployment data (`deploy/deploy.toml`)

- Status: Accepted; one open owner decision (one image with boot-time config, below)
- Date: 2026-10-24
- Decision-id: FVID-2026-10-24-deploy-section
- Crate: `fastvideo-manifest`, module `deploy`; schema
  `crates/fastvideo-manifest/schemas/deploy.v1.json`; file `deploy/deploy.toml`;
  generator `control/gen-configs.mjs`
- Plan: C1 (`docs/plan/2026-10-23-execution-plan.md`); review
  `docs/gaps/2026-10-22-architecture-review.md` §7.4

## Context

The family manifests (layer 2, ADR-0008, amended by plan A3) are the source
of model identity: variants, trees, GPU needs, caps. Plan A4 generated the
control plane's identity tables from them and inventoried what stayed
hand-written because the manifests carry no deployment data
(`docs/control/README.md`): the preset ids, titles, image variants, configs,
queues, timeouts, container disks and pre-quantized trees of
`control/src/presets.ts`, the family presets' card class, `enums.ts`
`VARIANTS` / `POOL_PRESET_IDS`, the tables of `scripts/serve/variants.sh`,
the machine / cells / API-name columns of `scripts/gcp/families.tsv`, the
pools of `configs/autoscale.toml`, the image table of `docs/serve/images.md`
and the thin `serve-<variant>` stages of `docker/gpucheck.Dockerfile`. Each
was a separate hand list; three of them listed the same presets in three
different orders.

The review (§7.4) asks for "the deployment half of the family manifest":
worker config, image variant, residency plan, weights source default per
region, disk, GPU class with `needs`, timeouts, pool name, upsampler, and
everything above generated from it.

## Decision

### 1. One file, `deploy/deploy.toml`, beside the family files

Not a `[deploy]` table in `families/*.toml`, and not one file per family:

- **The engine embeds the family files** (`include_str!`, A3 decision 1)
  and a test keeps the embedded copies equal to the committed ones. Queue
  sizes, timeouts or a GCP machine type in them would change the engine's
  embedded data, rebuild it and rerun its capabilities goldens for a change
  that is not about the model. Deployment data changes on a different
  cadence and is reviewed by different people.
- **The deployment unit is a preset, not a variant.** A preset is one pool:
  a set of variants served together by one fv-serve process, on one image
  variant, with one worker config, queue and timeouts. Family presets serve
  the variants of several model presets in one process. A per-variant
  `[deploy]` table cannot say that.
- **Order and non-family entries.** The dashboard's preset order interleaves
  families (the four standard pools first), the autoscaler's pool order is
  asserted by `fastvideo-autoscale`'s own test, and some entries belong to no
  family (the `cpu` image, the legacy `debug` tag set, the fake-engine GCP
  row). Per-family files would need an ordering key and a fifth file.

`families/*.toml` and layer 1 are untouched. `deploy/deploy.toml` lives
outside `families/`, so no family reader (`load_dir`, `FV_FAMILIES_DIR`,
`gen-configs.mjs` `loadFamilies`) sees it.

### 2. Shape

```toml
schema = 1

[dispatch]                     # model family -> edge / dispatch family
h3 = "h3"
ltx2 = "ltx"
wan = "wan"

[[images]]                     # a serve image variant (Dockerfile stage serve-<id>)
id = "wan5b"
config = "runpod-wan5b.toml"   # baked as FV_CONFIG
also = ["runpod-wan14b.toml"]  # further configs it carries
# compute = "cpu" (fake engine, its own build) / legacy = true (documented tag set) / note

[[presets]]                    # a model preset: ONE pool
id = "h3-turbo"
title = "H3 turbo (fasth3)"
description = "..."
standard = true                # one of the default cluster's pools
image = "h3-turbo"
config = "runpod.toml"         # configs/serve; inline when the image does not carry it
prequant = ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"]
models = [{ variant = "fasth3-4step-vsa", id = "fasth3", recipe = "h3-turbo" }]
max_queued = 32
job_timeout_s = 1800
stale_after_s = 120
# licence, container_disk_gb when set

[[family_presets]]             # one pool serving several model presets of one family
id = "h3-family"
title = "..."                  # `{members}` stands for the members
description = "..."
image = "h3-max"
config = "runpod-family-h3.toml"
min_vram_gb = 96               # hand-chosen card class, >= the members' floor
members = ["h3-turbo", "h3-max", ...]
[[family_presets.left_out]]
what = "..."
why = "..."

[[gcp]]                        # a scripts/gcp/families.tsv row
name = "h3-turbo"
config = "gcp-h3-turbo.toml"
runpod_twin = "runpod.toml"    # its [[models]] give weight_trees and served
dispatch_family = "h3"
verify_cells = ["fasth3-4step-vsa", "aux"]
machine = "g4-standard-48"
fal_app = "minimax/h3-turbo"
minimax_model = "MiniMax-H3"
reactor_mode = "av"

[[autoscale.pools]]            # copied verbatim into configs/autoscale.toml
name = "h3-turbo"
family = "h3"
# ... fastvideo-autoscale's PoolConfig, opaque here
```

A pool model names its **variant id** (never a served name) and, when the
pool uses another name for it, that `recipe` (a served name or the tier
alias); its `id` defaults to the variant id; its family is the variant's.

### 3. What is data, what derives, what stays where it is

The review's list, field by field:

| Review item | Where |
|---|---|
| worker config | `config` names a `configs/serve` file; the file stays the worker config (A3: the variants a process serves and their **residency**, `resident` / `[engine] swap`). A second copy of residency here would be a second source |
| image variant | `image`, an `[[images]]` id |
| weights source default per region | derived, not data: the EU volume when it holds every tree (`volumes.sh` `FV_EU_VOLUME_TREES`), else R2 (`control/src/providers.ts` `weightsSource`). The only region with a volume is `eu`; the validator refuses any other volume (below) |
| disk | derived from the trees (fv-control sizes it), except where a preset sets `container_disk_gb` (`ltx-a2v`: 60); a family preset takes its members' largest |
| GPU class with `needs` | a model preset's class is its variants' floor (`gpu.min_vram_gib`, derived as before); a family preset states `min_vram_gb`, checked against the floor. `min_sm` / `needs` are computed (`DeployManifest::preset_floor`) but not yet a filter: fv-control filters GPU types by memory only, which is why the NVFP4 families state 96 GB |
| timeouts, queue | `max_queued`, `job_timeout_s`, `stale_after_s` |
| pool name | the preset id (every pool id equals its preset id) |
| upsampler | not per preset: a cluster / endpoint switch (`upsampler_local`, `control/src/upsampler.ts`); stays there |

### 4. Types in `fastvideo-manifest`, not `fastvideo-deploy`

`fastvideo_manifest::deploy` holds the types, the validation and the schema:

- Validation needs the family manifests' types (`FamilySet`,
  `VariantManifest`, `tree_covers`); the module is pure data like layer 2
  (no I/O, no new dependency: `serde`, `toml`, `schemars` and `thiserror`
  are already the crate's).
- `fastvideo-deploy` is the Runpod queue worker inside fv-serve (`axum`,
  `tokio`, `reqwest`); putting the types there would make every reader of
  deployment data (the generator's Rust twin, `fv-lab`, a future `fv-deploy`)
  depend on a server stack, and it would need `fastvideo-manifest` anyway.
  C2's `fv-boot` (in `fastvideo-deploy`) reads both files through this crate.
- The registry also depends on this crate; the module adds no dependency
  and is not a registry format (the module docs and this ADR say so). Like
  layer 2 it is versioned by `schema` and changes by amending this ADR.

### 5. Validation is split by what each side can see

`DeployManifest::parse` (the file alone): `schema = 1`; unknown keys refused;
ids are DNS labels of at most 31 characters (fv-control's name rule), unique
across model and family presets; config names are `configs/serve` file
names; a preset's image exists, is built (not `legacy`) and runs on a GPU;
positive queues, timeouts and disks; family members are model presets;
GCP rows: unique names, `served` only (and then required) without a
`runpod_twin`, `dispatch_family` a dispatch family; autoscale pools: unique
names, `family` a dispatch family, and **every table naming a `volume_id`
is `bz0ofkfx45` in `EU-RO-1`** (CLAUDE.md, EU only; `EU_VOLUME_ID`).

`DeployManifest::validate_with(&FamilySet)`: `dispatch` names exactly the
families; every model's variant id exists; its recipe is the id, a served
name of it, or (a name the manifests do not know) a tier alias, allowed only
for the first variant of its tier; a family preset serves one model family
and its `min_vram_gb` meets its variants' `gpu.min_vram_gib`; `prequant`
entries are nested strictly inside one of the preset's weight trees.

The Rust recipe rule is deliberately loose for tier aliases: the crate does
not know the engine's alias spellings (`tier_alias`: `h3-turbo`, `ltx-pro`,
…), so `check_recipe` accepts any name the manifests do not know for the
first variant of a tier. The exact spelling is checked in JavaScript only.

`control/gen-configs.mjs` repeats what it needs to generate and adds what
only it can see (`deployChecks`, run by `--check`): the exact tier-alias
spelling (the engine's `tier_alias`, resolved through `MODEL_NAMES`), the named configs exist, the
hand-written `serve-cpu` stage (read from the Dockerfile as the run generates
it, so an image edit regenerates in one run) and
`.github/workflows/serve-image.yml`'s two build lists name the image
variants, GCP `verify_cells` are
`verify-weights.sh` cells, `configs/autoscale.toml`'s prices cover every
pool's GPU types. fv-control's unit tests check its presets against the
family rules (`templates.test.ts`: the listed members equal
`H3_FAMILY_MEMBERS` etc.) and against `deploy-gen.ts`.

### 6. Generated from it (`--check` in `.github/workflows/control.yml`)

| Output | What |
|---|---|
| `control/src/deploy-gen.ts` | images, presets, family presets, dispatch map; `presets.ts` builds `STANDARD_POOLS`, `MODEL_PRESETS`, `FAMILY_PRESETS` from them |
| `control/src/enums.ts` regions | `VARIANTS`, `POOL_PRESET_IDS` |
| `control/src/cluster/worker-configs.ts` | the inline configs: every preset config its image does not carry |
| `scripts/serve/variants.sh` region | the image and preset tables, `FV_VARIANTS`, `FV_POOL_PRESETS`, `fv_variant_config` |
| `docker/gpucheck.Dockerfile` region | the six thin `serve-<variant>` CUDA stages (the image build is unchanged: every non-comment line identical) |
| `configs/autoscale.toml` region | the `[[autoscale.pools]]`, copied as text with their comments (checked to parse to the same pools) |
| `docs/serve/images.md` region | the image table |
| `scripts/gcp/families.tsv` | the whole file |

Byte identity of the move (first commit): fv-control's runtime values
(`STANDARD_POOLS`, `MODEL_PRESETS`, `FAMILY_PRESETS`, `POOL_PRESETS`, the
member lists, `VARIANTS`, `TEMPLATES`, `SLS_PRESETS`, `IMAGE_CONFIGS`, every
worker config's text) identical; `autoscale.toml` identical but two marker
lines; the Dockerfile's build lines identical; every `families.tsv` row
identical. Intended differences: one preset order for `POOL_PRESET_IDS`,
`FV_POOL_PRESETS` and the inline-config list (the dashboard's); the
`variants.sh` / `images.md` tables rendered uniformly from the data (the
stale "(wan-turbo)" label of the `wan` image dropped: `wan-turbo` resolves
to `fastwan22-ti2v-5b`, served by `wan5b`; what an extra config serves is
derived from its `[[models]]`, so `wan5b`'s row reads "serving
`wan14b-plug-4step` / `wan14b-turbo`" where it said "the `wan14b-turbo`
tier"); region markers.

### 7. Still hand-written, and why

- `configs/autoscale.toml` `[autoscale]` settings, lease and prices: the
  autoscaler's own configuration and the account's price table, not per
  pool (prices are cross-checked against the pools' GPU types).
- The `serve-cpu` Dockerfile stage: its own build graph (a CPU compile
  stage, `base-os`, its entrypoint); cross-checked.
- `serve-image.yml`'s build lists: YAML strings of the image build, which
  C1 does not change; cross-checked.
- The component cells of `verify-weights.sh` (`needs()`): components inside
  trees and research trees no family has; they belong to the weights store
  and go when `gia verify` replaces the script (review §7.2).
- `enums.ts` `EDGE_FAMILIES`: the edge's routing set, computed by code
  (`cluster/spec.ts` `modelFamily`: `sfwan` and `fake` are not model
  families), not a deployment table.
- The family membership rules (`presets.ts` `H3_FAMILY_MEMBERS`, …): code
  that states the owner's "every variant of the family" decision; the data
  lists the members and the tests compare the two.
- `cluster/spec.ts` `TEMPLATES` and `WAN_FAMILY_LEFT_OUT`, `presets.ts`
  `H3_DRAFT`, `cluster/regions.ts`, `cluster/catalog.json` descriptions: not
  in A4's inventory; they can move in a follow-up without a format change.

## Open owner decision: one image with boot-time config

Today CI builds seven image variants (six CUDA, one CPU) that differ only in
their last layer, a few KB: the baked `FV_CONFIG` and `FV_VARIANT`
(`docs/serve/images.md` "Layers"). The review (§7.4) proposes one CUDA image
whose config is always supplied at boot. **Not implemented** (plan §1: an
owner decision); the analysis:

**The path is proven.** 22 of the 28 presets already run on an image whose
baked config they do not use: fv-control sends their worker config as
`FV_WORKER_TOML_B64`, which the cluster, serverless and GMI / Brev boot
scripts decode to `/fv-worker.toml` before fv-serve starts
(`control/src/cluster/payloads.ts`, `serverless/payloads.ts`,
`providers.ts`). Only the four standard pools and `fastwan21` / `sfwan` use
a baked config.

**What it removes:** the six thin CUDA stages and their seven-way CI loop
(`ci-images.sh build`, one push and one smoke check per variant); five of the
seven GHCR tag families and their channel tags (`release.sh` promotes per
variant); ten of the thirteen Runpod templates (the six CUDA variants'
`fv-serve-<variant>-sls / -pod` pairs become one pair; `fv-serve-cpu-pod`
stays); the `VARIANTS` enum and the `variant` field of every
pool spec, launch form and serverless spec; `variants.sh`'s tables and
`fv_variant_for_config`; the inline-versus-baked distinction (every preset
sends its config); `image-configs.ts` (`IMAGE_CONFIGS` / `BAKED_CONFIGS`,
from which fv-control derives what an endpoint serves before a worker
boots). In this file, `[[images]]` shrinks to one GPU and one CPU entry and
every preset's `image` becomes that one; no other data moves.

**What it needs:** (1) every launch path sends a config (`FV_WORKER_TOML_B64`
or `FV_WORKER_CONFIG`), and an image started without one refuses to boot
instead of serving a default (`fv-entry.sh` today falls back to
`FV_CONFIG`); (2) serverless endpoints derive what they serve from the
spec's config, not the image's baked file (`serverless/serves.ts`); (3) the
size of a worker config in a Runpod template's env (the largest family
config is a few KB; the template env limit is UNVERIFIED); (4) stored specs
that name a variant migrate (the field becomes ignored, or maps to the one
image); (5) the CPU image stays separate: a different binary (no CUDA), so
"one image" is one CUDA image plus `cpu`; (6) the one CUDA image still carries
`runpod-fake.toml` for CI's no-GPU smoke check (`ci-images.sh smoke`), as
every variant image does today; (7) the autoscaler's pod pools name a
Runpod template (`[autoscale.pools.pod] template_id`, today a placeholder for
`sfwan-live`), and every template id that names a variant template
(`fv-serve-<variant>-pod`) is re-pointed at the one image's template, with
the worker config in the pool's env. C2's `fv-boot` (config
materialization from `FV_WORKER_TOML_B64` / `FV_WORKER_CONFIG` / baked) is
the natural place for (1).

**What it costs:** a rollback or a pin is per image, no longer per variant
(today a bad `h3-turbo` push can be reverted without touching `ltx`); every
launch carries the config, so a template edit no longer changes what a
running endpoint serves until it is relaunched.

## Consequences

- One more schema under drift check: `schemas/deploy.v1.json`.
- `fastvideo-manifest` joins the adapter-tests crate set
  (`scripts/serve/coverage.sh`), so CI runs its tests (the deploy validation
  against the committed families among them) when `deploy/deploy.toml`, the
  families or the crate change.
- A preset, image, GCP row or autoscale pool is edited in one place; a
  variant is still added in `families/*.toml` and then deployed by naming it
  in a preset.
- Enforced by: `crates/fastvideo-manifest/src/deploy.rs` unit tests,
  `tests/deploy.rs` (the committed file against `families/*.toml`, configs
  exist), `tests/schema.rs`; `control/test/unit/deploy-gen.test.ts` and
  `gen-configs.mjs --check` (`deployChecks`); `fastvideo-autoscale`
  `committed_example_parses` and `fastvideo-serve` `tests/e2e.rs`
  `gcp_configs_*` on the generated files.
