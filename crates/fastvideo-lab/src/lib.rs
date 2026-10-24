//! The lab: gold standards per served variant (ADR-0009) and, with plan D1,
//! the `fv-lab` validation driver that updates them.
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
//! - [`schema`]: the JSON Schema export, committed under `schemas/` and
//!   drift-checked.
//!
//! CPU only, no CUDA. Reads files (`gold::load_dir`); the registry's pure
//! crate is `fastvideo-manifest`, which this one builds on for the family
//! id, tier and model-name types.

pub mod gold;
pub mod profiles;
pub mod schema;

pub use gold::{
    check_date, load_dir, Cost, Gap, Gold, GoldError, LastValidated, Latency, Measures, Model,
    Quality, Source, TierSpec, Upstream, Workload, CITABLE_FIELDS, GOLD_SCHEMA, NUMERIC_FIELDS,
};
pub use profiles::{comment_of, measurement_claims, Claim};
