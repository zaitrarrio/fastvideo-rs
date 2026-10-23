# Port checklist: what a new model family touches today

Date: 2026-10-23. Source: `docs/gaps/2026-10-22-architecture-review.md` §4.1
and §4.2, written down so that a port is not discovered one missing piece at
a time. These are the nine places a family (`<fam>`) is defined or dispatched
by hand today. Each is edited separately; a mismatch between any two is a
runtime error, not a compile error. Where a port's write-up exists
(`docs/ports/<fam>.md`), it says which of these it filled and how.

1. **Registry** (`crates/fastvideo-core/src/registry.rs`): the `ModelFamily`
   variant, the family's static definition table, the arm in `resolve()`
   that maps a Hub id or preset to it, and the Hub ids in
   `all_registered_ids()`.
2. **Audio-video generate** (`crates/fastvideo-core/src/av_generate.rs`):
   the `ModelFamily::<Fam> => generate_<fam>(def, opts)` arm in
   `generate_av` and the `generate_<fam>` function, plus any family-specific
   fields it reads from `AvGenerateOptions`.
3. **CLI** (`crates/fastvideo-cli/src/main.rs`): the family's arguments
   (`--h3-recipe` is the pattern) and their hand-off into
   `AvGenerateOptions`.
4. **Model definition** (`crates/fastvideo-models/src/<fam>/`): configs,
   schedules, geometry and host reference math, with no CUDA.
5. **Runtime port** (`crates/fastvideo-cudarc/src/<fam>/`): the pipeline,
   its weight loading and its device ops on `cudarc/src/wan` (the runtime
   that is still named after Wan), plus the checkpoint key mapping in one of
   `cudarc/src/hub_keys.rs`, `cudarc/src/ltx2/keys.rs` or
   `cudarc/src/h3/dit_tree.rs` (three conventions today; a new family picks
   one or adds a fourth).
6. **Kernels** (`crates/fastvideo-cudarc/src/wan/`): the family's region of
   `kernels.cu` (the `// ==== region: <name> ====` and `// ---- ` markers),
   and its arms in `fuse.rs` (block fusion) and `quant.rs` (quantization
   plans), when the family needs either.
7. **Techniques** (`crates/fastvideo-models/src/techniques/`): the family's
   `ModelSpec` in `registry.rs::model_spec` (only H3 and LTX-2 have one
   today; Wan reads the techniques layer without a spec) and its profiles
   under `profiles/<fam>/*.toml`.
8. **GPU checks and oracle** (`crates/fastvideo-gpucheck/`): the `Cmd`
   variant in `src/main.rs`, the stage file `src/<fam>_stage.rs`, and the
   Python reference that judges it (`scripts/gpu/<fam>_oracle.py` or a
   `scripts/gpu/upstream/oracle_<fam>.py` run through
   `scripts/gpu/upstream/pod.sh`).
9. **Weights, serving and docs**: rows in the four
   `scripts/gpu/weights-{manifest,r2,revisions,sha256}.tsv`, a verify cell
   in `scripts/gpu/verify-weights.sh`, the R2 publish
   (`docs/ops/r2-weights.md`), the served adapter
   `crates/fastvideo-engine-service/src/cuda/<fam>.rs` with its catalog rows
   in `cuda/caps.rs` and limits in `cuda/validate.rs`, the serve TOML under
   `configs/serve/`, and the write-up `docs/ports/<fam>.md` plus the
   `docs/scope.md` table row.

Plan WP-A2 (the family manifest, `families/<fam>.toml`, replacing the
engine catalog rows) and WP-B2 (the `VideoFamily` trait and a static family
list replacing the registry tables, the `resolve()` arms and the
`generate_<fam>` functions) of `docs/plan/2026-10-23-execution-plan.md`
collapse this list to three things: the manifest, the family directory and
the oracle.
