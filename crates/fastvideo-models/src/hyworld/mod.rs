//! HY-WorldPlay host configs. Spec: docs/ports/hyworld.md.

pub mod pose;
pub mod trajectory;

pub use pose::{
    compute_latent_num, parse_pose_string, pose_to_input, HyWorldPoseInput, DEFAULT_FORWARD_SPEED,
};
pub use trajectory::{generate_camera_trajectory_local, Motion};

use crate::hunyuan15::Hunyuan15TransformerConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyWorldPreset {
    Bidirectional,
}

impl HyWorldPreset {
    pub fn as_str(self) -> &'static str {
        "hyworld_bidirectional"
    }

    pub fn default_height(self) -> usize {
        480
    }

    pub fn default_width(self) -> usize {
        832
    }

    pub fn default_num_frames(self) -> usize {
        81
    }

    pub fn default_steps(self) -> usize {
        40
    }

    pub fn flow_shift(self) -> f64 {
        5.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyWorldConfig {
    pub hy: Hunyuan15TransformerConfig,
    pub has_action_in: bool,
    pub has_siglip: bool,
    /// SigLIP token dim (load hook; zeros until vision weights present).
    pub siglip_dim: usize,
    pub siglip_tokens: usize,
}

impl HyWorldConfig {
    pub fn for_preset(_preset: HyWorldPreset) -> Self {
        Self {
            hy: Hunyuan15TransformerConfig::fasthunyuan15(),
            has_action_in: true,
            has_siglip: true,
            siglip_dim: 1152,
            siglip_tokens: 256,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hy: Hunyuan15TransformerConfig::tiny(),
            has_action_in: true,
            has_siglip: true,
            siglip_dim: 32,
            siglip_tokens: 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bidirectional_flags() {
        let c = HyWorldConfig::for_preset(HyWorldPreset::Bidirectional);
        assert!(c.has_action_in && c.has_siglip);
        assert!(c.hy.num_layers > 0);
    }
}
