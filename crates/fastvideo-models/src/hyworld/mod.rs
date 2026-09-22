//! HY-WorldPlay host configs. Spec: docs/ports/hyworld.md.

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
}

impl HyWorldConfig {
    pub fn for_preset(_preset: HyWorldPreset) -> Self {
        Self {
            hy: Hunyuan15TransformerConfig::fasthunyuan15(),
            has_action_in: true,
            has_siglip: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            hy: Hunyuan15TransformerConfig::tiny(),
            has_action_in: true,
            has_siglip: true,
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
