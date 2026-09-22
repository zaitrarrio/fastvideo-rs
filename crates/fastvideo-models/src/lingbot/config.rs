//! LingBot-Video host configs. Spec: docs/ports/lingbot.md.

/// Diffusers / FastVideo prompt template crop (system+user prefix tokens).
pub const PROMPT_CROP_START: usize = 140;

/// FastVideo `PROMPT_TEMPLATE` for LingBot-Video T2V (Qwen chat turns).
pub const PROMPT_TEMPLATE: &str = "<|im_start|>system\nGiven a user input that may include a text prompt alone, \
a text prompt with an image reference, or a text prompt with a video reference \
or a video reference alone, generate an \"Enhanced prompt\" that provides detailed \
visual descriptions suitable for video generation. Evaluate the level of detail \
in the user's input: if it is simple, enrich it by adding specifics about colors, \
shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal \
progression, and spatial relationships to create vivid, concrete, and temporally \
coherent scenes to create vivid and concrete scenes. Please generate only the \
enhanced description for the prompt below and avoid including any additional \
commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
<|im_start|>assistant\n";

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
    /// Per-expert FFN width when MoE (`moe_intermediate_size`).
    pub moe_intermediate_size: usize,
    pub score_func_sigmoid: bool,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
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
            moe_intermediate_size: 512,
            score_func_sigmoid: true,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        }
    }

    pub fn moe_30b() -> Self {
        let mut c = Self::dense_1_3b();
        // MoE 30B-A3B (FastVideo + Hub packaging): 128 experts, 8 active.
        c.hidden_size = 4096;
        c.num_attention_heads = 32;
        c.depth = 40;
        c.intermediate_size = 11_008;
        c.num_experts = 128;
        c.num_experts_per_tok = 8;
        c.moe_intermediate_size = 512;
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
            moe_intermediate_size: 16,
            score_func_sigmoid: true,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        }
    }

    /// Tiny MoE graph for unit tests (4 experts, top-2).
    pub fn tiny_moe() -> Self {
        let mut c = Self::tiny();
        c.num_experts = 4;
        c.num_experts_per_tok = 2;
        c.moe_intermediate_size = 16;
        c
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }
}

/// Apply [`PROMPT_TEMPLATE`] then tokenize with Diffusers `tokenizer/`.
pub fn tokenize_lingbot_prompt(
    root: &std::path::Path,
    prompt: &str,
    max_length: usize,
) -> Result<Vec<u32>, String> {
    let path = root.join("tokenizer").join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let body = PROMPT_TEMPLATE.replace("{}", prompt);
    let encoding = tokenizer
        .encode(body.as_str(), true)
        .map_err(|e| format!("lingbot qwen tokenize: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_length {
        ids.truncate(max_length);
    }
    Ok(ids)
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
    fn moe_preset() {
        let c = LingBotTransformerConfig::moe_30b();
        assert!(c.is_moe());
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 512);
    }

    #[test]
    fn crop_constant() {
        assert_eq!(PROMPT_CROP_START, 140);
        assert!(PROMPT_TEMPLATE.contains("<|im_start|>assistant"));
    }
}
