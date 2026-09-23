//! LongCat-Video host math for `fastvideo-cudarc::longcat`.
//! See docs/ports/longcat.md.

pub mod config;
pub mod schedule;

pub use config::{LongCatPreset, LongCatTransformerConfig};
pub use schedule::LongCatSchedule;
