//! LTX-2 model family (Lightricks, 19B asymmetric dual-stream audio+video):
//! configuration structs and host-side math the device graph in
//! `fastvideo-cudarc::ltx2` consumes. Reference: diffusers `transformer_ltx2.py`
//! and `pipelines/ltx2`. See docs/ports/ltx2.md.

pub mod config;
pub mod schedule;

pub use config::{ltx2_19b, ltx2_19b_distilled, Ltx2Config};
pub use schedule::Ltx2Schedule;
