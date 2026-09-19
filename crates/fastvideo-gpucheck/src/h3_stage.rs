//! MiniMax-H3 / FastH3 stages: each judges one part of the port against the
//! reference dump written by `scripts/gpu/h3_oracle.py` (see docs/ports/h3.md,
//! section j). Owned by the H3 track; `main.rs` only dispatches here.

use crate::report::{Report, StageResult};

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Print the inference contract and geometry the port targets.
    Info,
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = fastvideo_models::h3::config::H3InferenceContract::fasth3_8step();
            report.set("contract", format!("{c:?}"));
            Ok(())
        }
    }
}
