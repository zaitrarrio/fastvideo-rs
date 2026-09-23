//! LongCat-Video host configs. Spec: docs/ports/longcat.md.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongCatPreset {
    T2v480p,
    T2v720p,
}

impl LongCatPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T2v480p => "longcat_t2v_480p",
            Self::T2v720p => "longcat_t2v_720p",
        }
    }

    pub fn enable_bsa(self) -> bool {
        matches!(self, Self::T2v720p)
    }

    pub fn flow_shift(self) -> f64 {
        1.0
    }

    pub fn default_height(self) -> usize {
        match self {
            Self::T2v480p => 480,
            Self::T2v720p => 720,
        }
    }

    pub fn default_width(self) -> usize {
        match self {
            Self::T2v480p => 832,
            Self::T2v720p => 1280,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LongCatTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub caption_channels: usize,
    pub mlp_ratio: usize,
    pub adaln_tembed_dim: usize,
    pub frequency_embedding_size: usize,
    pub patch_size: [usize; 3],
    pub enable_bsa: bool,
    pub bsa_sparsity: f32,
    pub bsa_chunk: [usize; 3],
    pub text_tokens_zero_pad: bool,
}

impl LongCatTransformerConfig {
    pub fn base() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            hidden_size: 4096,
            depth: 48,
            num_heads: 32,
            caption_channels: 4096,
            mlp_ratio: 4,
            adaln_tembed_dim: 512,
            frequency_embedding_size: 256,
            patch_size: [1, 2, 2],
            enable_bsa: false,
            bsa_sparsity: 0.9375,
            bsa_chunk: [4, 4, 4],
            text_tokens_zero_pad: true,
        }
    }

    pub fn for_preset(preset: LongCatPreset) -> Self {
        let mut c = Self::base();
        c.enable_bsa = preset.enable_bsa();
        c
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            hidden_size: 32,
            depth: 2,
            num_heads: 2,
            caption_channels: 16,
            mlp_ratio: 2,
            adaln_tembed_dim: 16,
            frequency_embedding_size: 16,
            patch_size: [1, 2, 2],
            enable_bsa: false,
            bsa_sparsity: 0.5,
            bsa_chunk: [1, 1, 1],
            text_tokens_zero_pad: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn mlp_hidden(&self) -> usize {
        self.hidden_size * self.mlp_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_dims() {
        let c = LongCatTransformerConfig::base();
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.depth, 48);
        assert_eq!(c.head_dim(), 128);
    }

    #[test]
    fn preset_bsa() {
        assert!(!LongCatPreset::T2v480p.enable_bsa());
        assert!(LongCatPreset::T2v720p.enable_bsa());
    }
}
