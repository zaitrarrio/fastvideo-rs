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
//!   `scripts/gpu/runpod-matrix.sh`'s frame, and a cell's verdict step) and
//!   its `--dry-run`; `fv-lab gate`, a finished run's gates and verdicts
//!   evaluated again offline.
//! - [`weights`]: a `volume+r2` tier's trees: which come from the EU volume
//!   and which from the R2 weights store, fetched per arm (ADR-0019 §16).
//! - [`pod`]: `fv-lab pod`, a tier's Runpod pod on fv-deploy's `Provider`
//!   trait, guards and lifecycle.
//! - [`goldcheck`]: `fv-lab gold check`, the CI bridge (plan D4): every
//!   gold file parses and resolves against the family files, a validated
//!   commit exists, and the per-variant validation age.
//! - [`goldcell`]: the `gold` tier's cells, each gold variant's canonical
//!   cell generated from its gold file (`fv-lab run gold --all`).
//! - [`report`]: `fv-lab report tiers`, `docs/perf/tiers.md` from the gold
//!   files and one run set (plan D3).
//! - [`summary`], [`table`], [`experiments`]: `fv-lab report summary` and
//!   `report table`, what the replaced report scripts printed (the retired
//!   families' `live.log` summary lines, `hd_report.py`, `sol_bench_table.py`,
//!   `timings.sh`, the experiments' `*-summary.py`), from a run directory's
//!   files; [`pyfmt`] spells values as Python and jq did.
//! - [`knobs`]: the script settings no tier reads (the retired families'
//!   mode knobs, the restored modes' settings), refused by `fv-lab run` and
//!   `pod up --extra-env` instead of ignored.
//! - [`quote`]: bash's `printf %q`, for the dry-run equality.
//! - [`schema`]: the JSON Schema export, committed under `schemas/` and
//!   drift-checked.
//!
//! No CUDA: `fv-lab` calls `fv-gpucheck` and `verify-weights.sh` as
//! processes. The binary needs the `cli` feature (its command line and the
//! HTTP client of `fastvideo-deploy`).

pub mod experiments;
pub mod gold;
pub mod goldcell;
pub mod goldcheck;
pub mod knobs;
pub mod plan;
pub mod pod;
pub mod profiles;
pub mod pyfmt;
pub mod quote;
pub mod report;
pub mod run;
pub mod schema;
pub mod summary;
pub mod table;
pub mod weights;

pub use gold::{
    check_date, load_dir, Cost, Gap, Gold, GoldError, LastValidated, Latency, Measures, Model,
    Quality, Source, TierSpec, Upstream, Workload, CITABLE_FIELDS, GOLD_SCHEMA, NUMERIC_FIELDS,
};
pub use plan::{Arm, CellFile, Lab, LabError, Step, Tier, Tiers, Verdict};
pub use profiles::{comment_of, measurement_claims, Claim};
