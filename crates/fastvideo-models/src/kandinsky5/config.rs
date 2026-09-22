//! Kandinsky 5.0 Video host configs. Spec: docs/ports/kandinsky5.md.

/// Hub recipe / canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kandinsky5Preset {
    LiteT2v5s,
    ProT2v5s,
}

impl Kandinsky5Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LiteT2v5s => "k5_lite_t2v_5s",
            Self::ProT2v5s => "k5_pro_t2v_5s",
        }
    }

    pub fn flow_shift(self) -> f64 {
        5.0
    }
}

/// Lite DiT sizes from Hub `transformer/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Kandinsky5TransformerConfig {
    pub in_visual_dim: usize,
    pub out_visual_dim: usize,
    pub time_dim: usize,
    pub patch_size: [usize; 3],
    pub model_dim: usize,
    pub ff_dim: usize,
    pub num_text_blocks: usize,
    pub num_visual_blocks: usize,
    pub axes_dims: [usize; 3],
    pub in_text_dim: usize,
    pub in_text_dim2: usize,
    pub visual_cond: bool,
    pub qwen_crop_start: usize,
}

impl Kandinsky5TransformerConfig {
    pub fn lite() -> Self {
        Self {
            in_visual_dim: 16,
            out_visual_dim: 16,
            time_dim: 512,
            patch_size: [1, 2, 2],
            model_dim: 1792,
            ff_dim: 7168,
            num_text_blocks: 2,
            num_visual_blocks: 32,
            axes_dims: [16, 24, 24],
            in_text_dim: 3584,
            in_text_dim2: 768,
            visual_cond: true,
            qwen_crop_start: 129,
        }
    }

    /// Tiny graph for unit tests (matches Diffusers issue repro sizes).
    pub fn tiny() -> Self {
        Self {
            in_visual_dim: 4,
            out_visual_dim: 4,
            time_dim: 16,
            patch_size: [1, 1, 1],
            model_dim: 32,
            ff_dim: 64,
            num_text_blocks: 1,
            num_visual_blocks: 2,
            axes_dims: [4, 4, 8],
            in_text_dim: 16,
            in_text_dim2: 8,
            visual_cond: false,
            qwen_crop_start: 0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.axes_dims.iter().sum()
    }

    pub fn num_heads(&self) -> usize {
        self.model_dim / self.head_dim()
    }

    /// Channel count fed to `visual_embeddings` after optional I2V pack.
    pub fn visual_embed_in_dim(&self) -> usize {
        if self.visual_cond {
            2 * self.in_visual_dim + 1
        } else {
            self.in_visual_dim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lite_dims() {
        let c = Kandinsky5TransformerConfig::lite();
        assert_eq!(c.model_dim, 1792);
        assert_eq!(c.num_visual_blocks, 32);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.num_heads(), 28);
        assert_eq!(c.visual_embed_in_dim(), 33);
    }
}
