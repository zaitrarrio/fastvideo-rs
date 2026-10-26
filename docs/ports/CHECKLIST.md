# Port checklist: what a new model family touches

Date: 2026-10-24 (plan B2; first written 2026-10-23 from
`docs/gaps/2026-10-22-architecture-review.md` §4.1 and §4.2 with nine
places). B2 (`docs/adr/0013-video-family-trait.md`) folded the registry, the
generate dispatch and the CLI hand-off into one family module; A2/A3 made
`families/*.toml` the served catalog. Where a port's write-up exists
(`docs/ports/<fam>.md`), it says which of these it filled and how.

1. **The family module** (`crates/fastvideo-core/src/family/<fam>.rs`): the
   `VideoFamily` impl (its name, its `HubVariant` table of presets, Hub ids,
   workloads and detector tokens, and `load`), and the run that maps the
   common `Request` (canvas, timing, sampler, conditioning) and
   `AvGenerateOptions` onto the runtime port. One line adds it to
   `FAMILIES` in `family/mod.rs`; its position is its place in the
   resolution order. Options that only this family reads get a
   `<Fam>Options` in `FamilyOptions` and, when the CLI should set them, a
   flag in `crates/fastvideo-cli/src/main.rs` (most families need none: the
   common `Request` carries size, length, steps, guidance, seed, negative
   and the conditioning inputs). Re-bless the CLI goldens
   (`FV_GOLDEN_BLESS=1 cargo test -p fastvideo-cli -p fastvideo-core`, then
   copy `$CARGO_TARGET_TMPDIR/b2-goldens/*` over `tests/golden/`) and check
   the diff is only the new family.
2. **Model definition and runtime port**: host configs, schedules, geometry
   and reference math in `crates/fastvideo-models/src/<fam>/` (no CUDA); the
   pipeline, weight loading and device ops in
   `crates/fastvideo-cudarc/src/<fam>/` on `cudarc/src/rt`, with the
   checkpoint key mapping in one of `cudarc/src/hub_keys.rs`,
   `cudarc/src/ltx2/keys.rs` or `cudarc/src/h3/dit_tree.rs`; kernel region
   files appended to `KERNEL_PARTS` in `cudarc/build.rs` (`rt/kernel_src.rs`
   pins the concatenation's sha256), and arms in `wan/fuse.rs` /
   `rt/quant.rs` when the family needs block fusion or quantization plans.
   Settings it reads are rows of
   `crates/fastvideo-models/src/techniques/settings/table.rs` (ADR-0010);
   technique profiles go under `profiles/<fam>/*.toml`, with a `ModelSpec`
   in `techniques/registry.rs::model_spec` where the family has one.
3. **The oracle and GPU checks** (`crates/fastvideo-gpucheck/`): the `Cmd`
   variant, the stage file `src/<fam>_stage.rs`, and the Python reference
   that judges it (`scripts/gpu/<fam>_oracle.py` or
   `scripts/gpu/upstream/oracle_<fam>.py` through
   `scripts/gpu/upstream/pod.sh`).

To serve the family, in addition: its `families/<fam>.toml` (layer 2,
ADR-0008: variants, trees, GPU needs, caps, serving, engine recipe), from
which the engine catalog, the serve configs and the control plane are
projected (A3, A4); the served adapter
`crates/fastvideo-engine-service/src/cuda/<fam>.rs`; the weight trees in
the four `scripts/gpu/weights-{manifest,r2,revisions,sha256}.tsv`, the R2
publish (`docs/ops/r2-weights.md`) and `scripts/gpu/verify-weights.sh`; a
gold file `lab/gold/<variant>.toml` (ADR-0009); and the write-up
`docs/ports/<fam>.md` plus its `docs/scope.md` row.

GPU runs that prove a port (timing, byte identity, quality against another
arm) are fv-lab cells and a tier (`lab/cells/`, `lab/tiers.toml`,
`docs/dev/fv-lab.md`), not a new one-off pod script: D1e
(`docs/adr/0019-fv-lab.md` §14) deleted the one-offs under
`scripts/gpu/experiments/` but the one waiting for its tier's GPU check.
Parity against the upstream implementation is the oracle's
(`scripts/gpu/upstream/`, `docs/oracle.md`).
