//! HunyuanVideo 1.5 architecture sizes (FastVideo `HunyuanVideo15ArchConfig`).

/// Which Hub recipe / canvas this config targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hunyuan15Preset {
    T2v480p,
    I2v480pDistilled,
    T2v720p,
    I2v720pDistilled,
    Sr1080p,
}

impl Hunyuan15Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T2v480p => "hy15_480p_t2v",
            Self::I2v480pDistilled => "hy15_480p_i2v_distilled",
            Self::T2v720p => "hy15_720p_t2v",
            Self::I2v720pDistilled => "hy15_720p_i2v_distilled",
            Self::Sr1080p => "hy15_1080p_sr",
        }
    }

    pub fn flow_shift(self) -> f64 {
        match self {
            Self::T2v480p => 5.0,
            Self::I2v480pDistilled | Self::I2v720pDistilled | Self::Sr1080p => 7.0,
            Self::T2v720p => 9.0,
        }
    }

    /// MeanFlow SR second-stage shift (`flow_shift_sr`).
    pub fn flow_shift_sr(self) -> Option<f64> {
        match self {
            Self::Sr1080p => Some(2.0),
            _ => None,
        }
    }

    pub fn is_i2v(self) -> bool {
        matches!(self, Self::I2v480pDistilled | Self::I2v720pDistilled)
    }

    /// `(height, width, num_frames)` for the preset's published canvas.
    pub fn canvas(self) -> (usize, usize, usize) {
        match self {
            Self::T2v480p | Self::I2v480pDistilled => (480, 854, 121),
            Self::T2v720p | Self::I2v720pDistilled => (720, 1280, 129),
            Self::Sr1080p => (1080, 1920, 129),
        }
    }

    /// Distilled I2V packs ship an 8-step schedule; T2V stays at 50.
    pub fn default_steps(self) -> usize {
        if self.is_i2v() {
            8
        } else {
            50
        }
    }

    pub fn from_cli(s: &str) -> Option<Self> {
        match s.trim() {
            "hy15_480p_t2v" => Some(Self::T2v480p),
            "hy15_480p_i2v_distilled" => Some(Self::I2v480pDistilled),
            "hy15_720p_t2v" => Some(Self::T2v720p),
            "hy15_720p_i2v_distilled" => Some(Self::I2v720pDistilled),
            "hy15_1080p_sr" => Some(Self::Sr1080p),
            _ => None,
        }
    }
}

/// DiT sizes shared across 480p/720p/1080p packs.
#[derive(Debug, Clone, PartialEq)]
pub struct Hunyuan15TransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub num_refiner_layers: usize,
    pub mlp_ratio: f64,
    pub patch_size: usize,
    pub patch_size_t: usize,
    pub text_embed_dim: usize,
    pub text_embed_2_dim: usize,
    pub image_embed_dim: usize,
    pub rope_theta: f64,
    pub rope_axes_dim: [usize; 3],
    pub use_meanflow: bool,
    pub rms_eps: f32,
}

impl Hunyuan15TransformerConfig {
    pub fn fasthunyuan15() -> Self {
        Self {
            in_channels: 65,
            out_channels: 32,
            num_attention_heads: 16,
            attention_head_dim: 128,
            num_layers: 54,
            num_refiner_layers: 2,
            mlp_ratio: 4.0,
            patch_size: 1,
            patch_size_t: 1,
            text_embed_dim: 3584,
            text_embed_2_dim: 1472,
            image_embed_dim: 1152,
            rope_theta: 256.0,
            rope_axes_dim: [16, 56, 56],
            use_meanflow: false,
            rms_eps: 1e-6,
        }
    }

    /// Tiny graph for unit tests (2 heads × 8 dim, 2 double blocks).
    pub fn tiny() -> Self {
        Self {
            in_channels: 8,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8,
            num_layers: 2,
            num_refiner_layers: 1,
            mlp_ratio: 2.0,
            patch_size: 1,
            patch_size_t: 1,
            text_embed_dim: 16,
            text_embed_2_dim: 8,
            image_embed_dim: 8,
            rope_theta: 256.0,
            // All even so rotate-half RoPE pairs cleanly (prod = head_dim).
            rope_axes_dim: [2, 2, 4],
            use_meanflow: false,
            rms_eps: 1e-6,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size() as f64 * self.mlp_ratio) as usize
    }

    pub fn with_meanflow(mut self) -> Self {
        self.use_meanflow = true;
        self
    }
}

/// Causal video VAE (`Hunyuan15VAEArchConfig`).
#[derive(Debug, Clone, PartialEq)]
pub struct Hunyuan15VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: [usize; 5],
    pub layers_per_block: usize,
    pub spatial_compression_ratio: usize,
    pub temporal_compression_ratio: usize,
    pub scaling_factor: f32,
}

impl Hunyuan15VaeConfig {
    pub fn fasthunyuan15() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 32,
            block_out_channels: [128, 256, 512, 1024, 1024],
            layers_per_block: 2,
            spatial_compression_ratio: 16,
            temporal_compression_ratio: 4,
            scaling_factor: 1.03682,
        }
    }
}

/// Pipeline defaults for one preset.
#[derive(Debug, Clone, PartialEq)]
pub struct Hunyuan15PipelineDefaults {
    pub preset: Hunyuan15Preset,
    pub flow_shift: f64,
    pub flow_shift_sr: Option<f64>,
    /// Qwen system-template tokens cropped before DiT (`PROMPT_TEMPLATE_TOKEN_LENGTH`).
    pub text_crop_start: usize,
    pub qwen_max_length: usize,
    pub byt5_max_length: usize,
}

impl Hunyuan15PipelineDefaults {
    pub fn for_preset(preset: Hunyuan15Preset) -> Self {
        Self {
            preset,
            flow_shift: preset.flow_shift(),
            flow_shift_sr: preset.flow_shift_sr(),
            text_crop_start: 108,
            qwen_max_length: 1000 + 108,
            byt5_max_length: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_matches_heads() {
        let c = Hunyuan15TransformerConfig::fasthunyuan15();
        assert_eq!(c.hidden_size(), 2048);
        assert_eq!(c.rope_axes_dim.iter().sum::<usize>(), c.attention_head_dim);
        assert_eq!(c.mlp_hidden(), 8192);
    }

    #[test]
    fn flow_shifts() {
        assert_eq!(Hunyuan15Preset::T2v480p.flow_shift(), 5.0);
        assert_eq!(Hunyuan15Preset::T2v720p.flow_shift(), 9.0);
        assert_eq!(Hunyuan15Preset::Sr1080p.flow_shift_sr(), Some(2.0));
    }

    #[test]
    fn canvas_and_cli() {
        assert_eq!(Hunyuan15Preset::T2v480p.canvas(), (480, 854, 121));
        assert_eq!(Hunyuan15Preset::T2v480p.default_steps(), 50);
        assert_eq!(Hunyuan15Preset::I2v480pDistilled.default_steps(), 8);
        assert_eq!(
            Hunyuan15Preset::from_cli("hy15_480p_t2v"),
            Some(Hunyuan15Preset::T2v480p)
        );
        assert_eq!(Hunyuan15Preset::from_cli("nope"), None);
    }
}
