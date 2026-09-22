//! LingBot-World host configs (Wan2.2 I2V + cam). Spec: docs/ports/lingbotworld.md.

use crate::wan::WanVideoArchConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LingBotWorldPreset {
    BaseCam,
    V2CausalFast,
}

impl LingBotWorldPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BaseCam => "lingbotworld_base_cam",
            Self::V2CausalFast => "lingbotworld2_causal_fast",
        }
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
        match self {
            Self::BaseCam => 40,
            Self::V2CausalFast => 4,
        }
    }

    pub fn flow_shift(self) -> f64 {
        10.0
    }

    pub fn boundary_ratio(self) -> f32 {
        0.947
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LingBotWorldConfig {
    pub wan: WanVideoArchConfig,
    pub has_cam_injector: bool,
}

impl LingBotWorldConfig {
    pub fn for_preset(preset: LingBotWorldPreset) -> Self {
        let mut wan = WanVideoArchConfig::wan_2_2_i2v_a14b();
        wan.boundary_ratio = Some(preset.boundary_ratio());
        if matches!(preset, LingBotWorldPreset::V2CausalFast) {
            wan.causal = true;
            wan.local_attn_size = 21;
        }
        Self {
            wan,
            has_cam_injector: true,
        }
    }

    pub fn tiny() -> Self {
        Self {
            wan: WanVideoArchConfig::i2v_block(),
            has_cam_injector: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_cam_moe_boundary() {
        let c = LingBotWorldConfig::for_preset(LingBotWorldPreset::BaseCam);
        assert_eq!(c.wan.boundary_ratio, Some(0.947));
        assert!(c.has_cam_injector);
        assert_eq!(c.wan.in_channels, 36);
    }
}
