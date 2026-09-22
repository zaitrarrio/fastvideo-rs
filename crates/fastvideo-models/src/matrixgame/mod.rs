//! Matrix-Game host configs (Wan DiT + action dims). Spec: docs/ports/matrixgame.md.

use crate::wan::WanVideoArchConfig;

/// Hub recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixGamePreset {
    Mg2BaseDistilled,
    Mg2GtaDistilled,
    Mg2TempleRunDistilled,
    Mg2Base,
    Mg3BaseDistilled,
}

impl MatrixGamePreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mg2BaseDistilled => "mg2_base_distilled",
            Self::Mg2GtaDistilled => "mg2_gta_distilled",
            Self::Mg2TempleRunDistilled => "mg2_templerun_distilled",
            Self::Mg2Base => "mg2_base",
            Self::Mg3BaseDistilled => "mg3_base_distilled",
        }
    }

    pub fn is_mg3(self) -> bool {
        matches!(self, Self::Mg3BaseDistilled)
    }

    pub fn is_distilled(self) -> bool {
        !matches!(self, Self::Mg2Base)
    }

    pub fn keyboard_dim(self) -> usize {
        match self {
            Self::Mg2BaseDistilled | Self::Mg2Base => 4,
            Self::Mg2GtaDistilled => 2,
            Self::Mg2TempleRunDistilled => 7,
            Self::Mg3BaseDistilled => 6,
        }
    }

    pub fn mouse_dim(self) -> usize {
        2
    }

    pub fn default_height(self) -> usize {
        if self.is_mg3() {
            720
        } else {
            352
        }
    }

    pub fn default_width(self) -> usize {
        if self.is_mg3() {
            1280
        } else {
            640
        }
    }

    pub fn default_num_frames(self) -> usize {
        if self.is_mg3() {
            57
        } else {
            81
        }
    }

    pub fn default_steps(self) -> usize {
        if self.is_mg3() {
            3
        } else if self.is_distilled() {
            3
        } else {
            50
        }
    }

    pub fn flow_shift(self) -> f64 {
        5.0
    }

    pub fn action_blocks(self) -> usize {
        15
    }
}

/// Matrix-Game DiT = Wan arch + action metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixGameConfig {
    pub wan: WanVideoArchConfig,
    pub keyboard_dim: usize,
    pub mouse_dim: usize,
    pub action_blocks: usize,
    pub num_frames_per_block: usize,
}

impl MatrixGameConfig {
    pub fn for_preset(preset: MatrixGamePreset) -> Self {
        let wan = if preset.is_mg3() {
            WanVideoArchConfig::wan_2_2_ti2v_5b()
        } else {
            // FastVideo MatrixGame2WanVideoArchConfig: Wan defaults + image_dim.
            let mut w = WanVideoArchConfig::wan_t2v_14b();
            w.image_dim = Some(1280);
            w.text_dim = 0;
            w
        };
        Self {
            wan,
            keyboard_dim: preset.keyboard_dim(),
            mouse_dim: preset.mouse_dim(),
            action_blocks: preset.action_blocks(),
            num_frames_per_block: 3,
        }
    }

    pub fn tiny() -> Self {
        Self {
            wan: WanVideoArchConfig::tiny(),
            keyboard_dim: 4,
            mouse_dim: 2,
            action_blocks: 1,
            num_frames_per_block: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mg3_is_ti2v_5b() {
        let c = MatrixGameConfig::for_preset(MatrixGamePreset::Mg3BaseDistilled);
        assert_eq!(c.wan.in_channels, 48);
        assert_eq!(c.wan.num_attention_heads, 24);
        assert_eq!(c.keyboard_dim, 6);
    }

    #[test]
    fn mg2_variants_keyboard() {
        assert_eq!(MatrixGamePreset::Mg2GtaDistilled.keyboard_dim(), 2);
        assert_eq!(MatrixGamePreset::Mg2TempleRunDistilled.keyboard_dim(), 7);
    }
}
