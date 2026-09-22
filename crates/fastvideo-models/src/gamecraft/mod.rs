//! HunyuanGameCraft host configs. Spec: docs/ports/gamecraft.md.

use crate::hunyuan15::Hunyuan15TransformerConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameCraftPreset {
    I2v,
}

impl GameCraftPreset {
    pub fn as_str(self) -> &'static str {
        "gamecraft_i2v"
    }

    pub fn default_height(self) -> usize {
        704
    }

    pub fn default_width(self) -> usize {
        1280
    }

    pub fn default_num_frames(self) -> usize {
        33
    }

    pub fn default_steps(self) -> usize {
        50
    }

    pub fn flow_shift(self) -> f64 {
        5.0
    }

    pub fn guidance_scale(self) -> f32 {
        6.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameCraftConfig {
    pub hy: Hunyuan15TransformerConfig,
    pub camera_net: bool,
}

impl GameCraftConfig {
    pub fn for_preset(_preset: GameCraftPreset) -> Self {
        let mut hy = Hunyuan15TransformerConfig::fasthunyuan15();
        // 16 noise + 16 gt + 1 mask
        hy.in_channels = 33;
        Self {
            hy,
            camera_net: true,
        }
    }

    pub fn tiny() -> Self {
        let mut hy = Hunyuan15TransformerConfig::tiny();
        hy.in_channels = 9; // 4 + 4 + 1 for tiny
        Self {
            hy,
            camera_net: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thirty_three_channels() {
        let c = GameCraftConfig::for_preset(GameCraftPreset::I2v);
        assert_eq!(c.hy.in_channels, 33);
        assert!(c.camera_net);
    }
}
