//! The lab: gold standards per served variant (ADR-0009) and `fv-lab`, the
//! validation driver that runs them (ADR-0019).
//!
//! - [`gold`]: [`Gold`], one `lab/gold/<variant>.toml`: the variant and the
//!   registry name and `@rev` it certifies, the tier and its GPU class, the
//!   reference cell, measured latency, quality and cost, the last `fv-lab`
//!   validation, and a `[[source]]` for every number. Every numeric field is
//!   optional; a present one must be cited and an absent one declared as a
//!   `[[gap]]`.
//! - [`profiles`]: the measurement lint over `profiles/**/*.toml` comments,
//!   which keeps the numbers in gold files and `docs/perf`, not in profile
//!   comments.
//! - [`plan`]: the run plan, `lab/tiers.toml` and `lab/cells/*.toml`
//!   ([`Lab`]): types, validation and the rendering of a tier into command
//!   lines.
//! - [`run`]: `fv-lab run`, the on-pod cell runner (the port of
//!   `scripts/gpu/runpod-matrix.sh`'s frame) and its `--dry-run`.
//! - [`pod`]: `fv-lab pod`, a tier's Runpod pod on fv-deploy's `Provider`
//!   trait, guards and lifecycle.
//! - [`goldcheck`]: `fv-lab gold check`, the CI bridge (plan D4): every
//!   gold file parses and resolves against the family files, a validated
//!   commit exists, and the per-variant validation age.
//! - [`goldcell`]: the `gold` tier's cells, each gold variant's canonical
//!   cell generated from its gold file (`fv-lab run gold --all`).
//! - [`report`]: `fv-lab report tiers`, `docs/perf/tiers.md` from the gold
//!   files and one run set (plan D3).
//! - [`quote`]: bash's `printf %q`, for the dry-run equality.
//! - [`schema`]: the JSON Schema export, committed under `schemas/` and
//!   drift-checked.
//!
//! No CUDA: `fv-lab` calls `fv-gpucheck` and `verify-weights.sh` as
//! processes. The binary needs the `cli` feature (its command line and the
//! HTTP client of `fastvideo-deploy`).

pub mod gold;
pub mod goldcell;
pub mod goldcheck;
pub mod plan;
pub mod pod;
pub mod profiles;
pub mod quote;
pub mod report;
pub mod run;
pub mod schema;

pub use gold::{
    check_date, load_dir, Cost, Gap, Gold, GoldError, LastValidated, Latency, Measures, Model,
    Quality, Source, TierSpec, Upstream, Workload, CITABLE_FIELDS, GOLD_SCHEMA, NUMERIC_FIELDS,
};
pub use plan::{Arm, CellFile, Lab, LabError, Step, Tier, Tiers, Verdict};
pub use profiles::{comment_of, measurement_claims, Claim};
