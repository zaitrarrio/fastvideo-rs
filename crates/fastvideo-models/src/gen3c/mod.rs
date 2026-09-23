//! GEN3C host math (Cosmos DiT + 3D cache). Spec: docs/ports/gen3c.md.

pub mod cache_3d;
pub mod camera;
pub mod config;
pub mod schedule;
pub mod trajectory;

pub use cache_3d::{
    forward_warp_rgb, pack_rgb_buffers, pack_vae_buffers, render_trajectory, synthetic_depth,
    unproject_points,
};
pub use camera::{default_intrinsics, generate_camera_trajectory, identity4, Mat3, Mat4};
pub use config::{Gen3CPreset, Gen3CTransformerConfig};
pub use schedule::Gen3CSchedule;
pub use trajectory::{CameraRotation, TrajectoryType};
