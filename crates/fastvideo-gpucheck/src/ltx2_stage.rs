//! LTX-2 stages: each judges one part of the port against the reference dump
//! written by `scripts/gpu/ltx2_oracle.py` (see docs/ports/ltx2.md, section j).
//! Owned by the LTX-2 track; `main.rs` only dispatches here.

use crate::report::{Report, StageResult};

#[derive(clap::Subcommand, Debug)]
pub enum Stage {
    /// Print the transformer configuration the port targets.
    Info,
}

pub fn run(report: &mut Report, stage: &Stage) -> StageResult<()> {
    match stage {
        Stage::Info => {
            let c = fastvideo_models::ltx2::config::ltx2_19b_distilled();
            report.set("config", format!("{c:?}"));
            Ok(())
        }
    }
}
