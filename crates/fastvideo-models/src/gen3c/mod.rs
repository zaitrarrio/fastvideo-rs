//! GEN3C host math (Cosmos DiT + 3D cache). Spec: docs/ports/gen3c.md.

pub mod config;
pub mod schedule;
pub mod trajectory;

pub use config::{Gen3CPreset, Gen3CTransformerConfig};
pub use schedule::Gen3CSchedule;
pub use trajectory::{CameraRotation, TrajectoryType};
