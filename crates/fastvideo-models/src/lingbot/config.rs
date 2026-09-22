//! LingBot-Video host configs. Spec: docs/ports/lingbot.md.

/// Diffusers / FastVideo prompt template crop (system+user prefix tokens).
pub const PROMPT_CROP_START: usize = 140;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LingBotPreset {
    Dense13b,
    Moe30b,
}

impl LingBotPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense13b => "lingbot_dense_1_3b",
            Self::Moe30b => "lingbot_moe_30b",
        }
    }

    pub fn flow_shift(self) -> f64 {
        3.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LingBotTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub depth: usize,
    pub intermediate_size: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub patch_size: [usize; 3],
    pub rope_theta: f32,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    pub norm_eps: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl LingBotTransformerConfig {
    pub fn dense_1_3b() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            hidden_size: 2048,
            num_attention_heads: 16,
            depth: 24,
            intermediate_size: 6144,
            text_dim: 2560,
            freq_dim: 256,
            patch_size: [1, 2, 2],
            rope_theta: 256.0,
            axes_dims: [32, 48, 48],
            axes_lens: [8192, 1024, 1024],
            norm_eps: 1e-6,
            num_experts: 0,
            num_experts_per_tok: 8,
        }
    }

    pub fn moe_30b() -> Self {
        let mut c = Self::dense_1_3b();
        // MoE scaffold sizes — refine from Hub config when weights land.
        c.hidden_size = 4096;
        c.num_attention_heads = 32;
        c.depth = 40;
        c.intermediate_size = 11_008;
        c.num_experts = 64;
        c
    }

    pub fn for_preset(preset: LingBotPreset) -> Self {
        match preset {
            LingBotPreset::Dense13b => Self::dense_1_3b(),
            LingBotPreset::Moe30b => Self::moe_30b(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            hidden_size: 32,
            num_attention_heads: 2,
            depth: 2,
            intermediate_size: 64,
            text_dim: 16,
            freq_dim: 16,
            patch_size: [1, 2, 2],
            rope_theta: 256.0,
            axes_dims: [8, 8, 8],
            axes_lens: [64, 32, 32],
            norm_eps: 1e-6,
            num_experts: 0,
            num_experts_per_tok: 2,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_dims() {
        let c = LingBotTransformerConfig::dense_1_3b();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.depth, 24);
        assert_eq!(c.text_dim, 2560);
        assert_eq!(c.head_dim(), 128);
        assert!(!c.is_moe());
    }

    #[test]
    fn crop_constant() {
        assert_eq!(PROMPT_CROP_START, 140);
    }
}
