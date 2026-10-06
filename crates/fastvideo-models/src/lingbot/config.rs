//! LingBot-Video host configs. Spec: docs/ports/lingbot.md.
//!
//! The MoE preset is the Hub `transformer/config.json` of
//! `robbyant/lingbot-video-moe-30b-a3b` (vendored under `hub/`, checked by
//! [`tests::moe_preset_matches_hub_config`]); the refiner shares it.

use serde::Deserialize;

/// Upstream prompt template crop for the published Qwen3-VL processor. The
/// pipeline recomputes it from the tokenizer ([`prompt_crop_start`]); this is
/// the value the reference processor yields and what tests pin.
pub const PROMPT_CROP_START: usize = 140;

/// Upstream `TOKEN_LENGTH` (`pipeline_lingbot_video.py`): processor truncation.
pub const TOKEN_LENGTH: usize = 37_698;

/// `PROMPT_TEMPLATE` of `pipeline_lingbot_video.py` (Qwen chat turns).
pub const PROMPT_TEMPLATE: &str =
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, \
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

/// `DEFAULT_NEGATIVE_PROMPT` of `pipeline_lingbot_video.py` (video modes). The
/// base stage of the official run uses it; the refiner uses a zero negative
/// (`null_cond_clone_zero`).
pub const DEFAULT_NEGATIVE_PROMPT: &str = r#"{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}"#;

/// Hub repo and the revision the vendored config and key list were read at.
pub const MOE_30B_HUB_REPO: &str = "robbyant/lingbot-video-moe-30b-a3b";
pub const MOE_30B_HUB_REVISION: &str = "f2e538f64afe00cc4ae674db2aeb52e2945edfd5";

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

/// Router score function (`score_func`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFunc {
    Sigmoid,
    Softmax,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LingBotTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub depth: usize,
    /// Dense FFN width (dense layers, and every layer of the dense model).
    pub intermediate_size: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub patch_size: [usize; 3],
    pub rope_theta: f32,
    pub axes_dims: [usize; 3],
    pub axes_lens: [usize; 3],
    pub norm_eps: f32,
    pub qkv_bias: bool,
    pub out_bias: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Per-expert FFN width when MoE (`moe_intermediate_size`).
    pub moe_intermediate_size: usize,
    /// Every `decoder_sparse_step`-th layer is MoE (`(i + 1) % step == 0`).
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    /// Shared experts: one SwiGLU of width `moe_intermediate_size * n` on every token.
    pub n_shared_experts: usize,
    pub score_func: ScoreFunc,
    pub norm_topk_prob: bool,
    /// Group-limited routing (`n_group`, `topk_group`); `None` routes over all experts.
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub routed_scaling_factor: f32,
}

/// The Hub `transformer/config.json` fields this port reads.
#[derive(Debug, Clone, Deserialize)]
struct HubConfig {
    patch_size: [usize; 3],
    in_channels: usize,
    out_channels: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    depth: usize,
    intermediate_size: usize,
    text_dim: usize,
    freq_dim: usize,
    norm_eps: f32,
    rope_theta: f32,
    axes_dims: [usize; 3],
    axes_lens: [usize; 3],
    qkv_bias: bool,
    out_bias: bool,
    #[serde(default = "default_true")]
    patch_embed_bias: bool,
    #[serde(default = "default_true")]
    timestep_mlp_bias: bool,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    decoder_sparse_step: usize,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    n_shared_experts: Option<usize>,
    score_func: String,
    norm_topk_prob: bool,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    routed_scaling_factor: f32,
}

fn default_true() -> bool {
    true
}

impl LingBotTransformerConfig {
    /// `robbyant/lingbot-video-dense-1.3b` `transformer/config.json`.
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
            qkv_bias: false,
            out_bias: true,
            num_experts: 0,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            n_shared_experts: 0,
            score_func: ScoreFunc::Sigmoid,
            norm_topk_prob: true,
            n_group: None,
            topk_group: None,
            routed_scaling_factor: 1.0,
        }
    }

    /// `robbyant/lingbot-video-moe-30b-a3b` `transformer/config.json` (and
    /// `refiner/config.json`, identical but for the diffusers version): 48
    /// layers of hidden 2048, 128 routed experts of width 768 (top-8 within
    /// the top-2 of 4 groups), one shared expert, routed scale 2.5.
    pub fn moe_30b() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            hidden_size: 2048,
            num_attention_heads: 16,
            depth: 48,
            intermediate_size: 6144,
            text_dim: 2560,
            freq_dim: 256,
            patch_size: [1, 2, 2],
            rope_theta: 256.0,
            axes_dims: [32, 48, 48],
            axes_lens: [4096, 512, 512],
            norm_eps: 1e-6,
            qkv_bias: false,
            out_bias: true,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 768,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            n_shared_experts: 1,
            score_func: ScoreFunc::Sigmoid,
            norm_topk_prob: true,
            n_group: Some(4),
            topk_group: Some(2),
            routed_scaling_factor: 2.5,
        }
    }

    /// Parse a Hub `transformer/config.json`. Fails on fields this port does
    /// not implement (no patch-embed / timestep-MLP bias, unknown score func).
    pub fn from_hub_json(text: &str) -> Result<Self, String> {
        let h: HubConfig =
            serde_json::from_str(text).map_err(|e| format!("lingbot config.json: {e}"))?;
        if !h.patch_embed_bias || !h.timestep_mlp_bias {
            return Err("lingbot config: bias-free patch / timestep MLP is not ported".into());
        }
        let score_func = match h.score_func.as_str() {
            "sigmoid" => ScoreFunc::Sigmoid,
            "softmax" => ScoreFunc::Softmax,
            other => return Err(format!("lingbot config: score_func {other:?}")),
        };
        let cfg = Self {
            in_channels: h.in_channels,
            out_channels: h.out_channels,
            hidden_size: h.hidden_size,
            num_attention_heads: h.num_attention_heads,
            depth: h.depth,
            intermediate_size: h.intermediate_size,
            text_dim: h.text_dim,
            freq_dim: h.freq_dim,
            patch_size: h.patch_size,
            rope_theta: h.rope_theta,
            axes_dims: h.axes_dims,
            axes_lens: h.axes_lens,
            norm_eps: h.norm_eps,
            qkv_bias: h.qkv_bias,
            out_bias: h.out_bias,
            num_experts: h.num_experts,
            num_experts_per_tok: h.num_experts_per_tok,
            moe_intermediate_size: h.moe_intermediate_size,
            decoder_sparse_step: h.decoder_sparse_step.max(1),
            mlp_only_layers: h.mlp_only_layers,
            n_shared_experts: h.n_shared_experts.unwrap_or(0),
            score_func,
            norm_topk_prob: h.norm_topk_prob,
            n_group: h.n_group.filter(|&g| g > 1),
            topk_group: h.topk_group,
            routed_scaling_factor: h.routed_scaling_factor,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The upstream constructor's assertions plus what routing needs.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err("lingbot config: hidden not divisible by heads".into());
        }
        if self.head_dim() != 2 * self.rope_half() {
            return Err(format!(
                "lingbot config: head_dim {} != sum(axes_dims) {}",
                self.head_dim(),
                self.axes_dims.iter().sum::<usize>()
            ));
        }
        if self.axes_dims.iter().any(|d| d % 2 != 0) {
            return Err("lingbot config: odd rope axis".into());
        }
        if self.is_moe() {
            if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
                return Err("lingbot config: top-k outside the expert count".into());
            }
            if let Some(g) = self.n_group {
                let tg = self.topk_group.unwrap_or(g);
                if self.num_experts % g != 0 || tg == 0 || tg > g {
                    return Err("lingbot config: bad group routing".into());
                }
                if tg * (self.num_experts / g) < self.num_experts_per_tok {
                    return Err("lingbot config: top-k larger than the selected groups".into());
                }
            }
        }
        Ok(())
    }

    pub fn for_preset(preset: LingBotPreset) -> Self {
        match preset {
            LingBotPreset::Dense13b => Self::dense_1_3b(),
            LingBotPreset::Moe30b => Self::moe_30b(),
        }
    }

    /// Tiny dense graph for unit tests (head_dim 16 = 2·(2+3+3)).
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            hidden_size: 32,
            num_attention_heads: 2,
            depth: 2,
            intermediate_size: 64,
            text_dim: 24,
            freq_dim: 16,
            patch_size: [1, 2, 2],
            rope_theta: 256.0,
            axes_dims: [4, 6, 6],
            axes_lens: [64, 32, 32],
            norm_eps: 1e-6,
            qkv_bias: false,
            out_bias: true,
            num_experts: 0,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            decoder_sparse_step: 1,
            mlp_only_layers: Vec::new(),
            n_shared_experts: 0,
            score_func: ScoreFunc::Sigmoid,
            norm_topk_prob: true,
            n_group: None,
            topk_group: None,
            routed_scaling_factor: 1.0,
        }
    }

    /// Tiny MoE graph shaped like the 30B one: 8 experts in 4 groups (top-2
    /// groups), top-2, one shared expert, scale 2.5, layer 0 dense.
    pub fn tiny_moe() -> Self {
        let mut c = Self::tiny();
        c.num_experts = 8;
        c.num_experts_per_tok = 2;
        c.moe_intermediate_size = 16;
        c.n_shared_experts = 1;
        c.n_group = Some(4);
        c.topk_group = Some(2);
        c.routed_scaling_factor = 2.5;
        c.mlp_only_layers = vec![0];
        c
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Complex RoPE pairs per head (`sum(axes_dims) / 2`).
    pub fn rope_half(&self) -> usize {
        self.axes_dims.iter().sum::<usize>() / 2
    }

    pub fn is_moe(&self) -> bool {
        self.num_experts > 0
    }

    /// Upstream `LingBotVideoBlock`: MoE unless listed in `mlp_only_layers` or
    /// off the `decoder_sparse_step` stride.
    pub fn layer_is_moe(&self, layer: usize) -> bool {
        self.is_moe()
            && !self.mlp_only_layers.contains(&layer)
            && (layer + 1) % self.decoder_sparse_step.max(1) == 0
    }

    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_size.iter().product::<usize>()
    }

    pub fn out_patch_dim(&self) -> usize {
        self.out_channels * self.patch_size.iter().product::<usize>()
    }

    /// Every parameter tensor (key, shape) the DiT holds, in the Hub naming.
    pub fn expected_keys(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden_size;
        let hd = self.head_dim();
        let mut out: Vec<(String, Vec<usize>)> = vec![
            ("patch_embedder.weight".into(), vec![h, self.patch_dim()]),
            ("patch_embedder.bias".into(), vec![h]),
            ("time_embedder.linear_1.weight".into(), vec![h, self.freq_dim]),
            ("time_embedder.linear_1.bias".into(), vec![h]),
            ("time_embedder.linear_2.weight".into(), vec![h, h]),
            ("time_embedder.linear_2.bias".into(), vec![h]),
            ("time_modulation.1.weight".into(), vec![6 * h, h]),
            ("time_modulation.1.bias".into(), vec![6 * h]),
            ("text_embedder.norm.weight".into(), vec![self.text_dim]),
            ("text_embedder.linear_1.weight".into(), vec![h, self.text_dim]),
            ("text_embedder.linear_1.bias".into(), vec![h]),
            ("text_embedder.linear_2.weight".into(), vec![h, h]),
            ("text_embedder.linear_2.bias".into(), vec![h]),
            ("norm_out_modulation.1.weight".into(), vec![2 * h, h]),
            ("norm_out_modulation.1.bias".into(), vec![2 * h]),
            ("proj_out.weight".into(), vec![self.out_patch_dim(), h]),
            ("proj_out.bias".into(), vec![self.out_patch_dim()]),
        ];
        for i in 0..self.depth {
            let p = |n: &str| format!("blocks.{i}.{n}");
            out.push((p("scale_shift_table"), vec![1, 6 * h]));
            for n in ["norm1", "norm2", "norm_post_attn", "norm_post_ffn"] {
                out.push((p(&format!("{n}.weight")), vec![h]));
            }
            for n in ["to_q", "to_k", "to_v"] {
                out.push((p(&format!("attn.{n}.weight")), vec![h, h]));
                if self.qkv_bias {
                    out.push((p(&format!("attn.{n}.bias")), vec![h]));
                }
            }
            out.push((p("attn.to_out.weight"), vec![h, h]));
            if self.out_bias {
                out.push((p("attn.to_out.bias"), vec![h]));
            }
            out.push((p("attn.norm_q.weight"), vec![hd]));
            out.push((p("attn.norm_k.weight"), vec![hd]));
            if self.layer_is_moe(i) {
                let (e, m) = (self.num_experts, self.moe_intermediate_size);
                out.push((p("ffn.router.weight"), vec![e, h]));
                out.push((p("ffn.router.e_score_correction_bias"), vec![e]));
                out.push((p("ffn.experts.w1"), vec![e, m, h]));
                out.push((p("ffn.experts.w2"), vec![e, h, m]));
                out.push((p("ffn.experts.w3"), vec![e, m, h]));
                if self.n_shared_experts > 0 {
                    let s = m * self.n_shared_experts;
                    out.push((p("ffn.shared_experts.gate_proj.weight"), vec![s, h]));
                    out.push((p("ffn.shared_experts.up_proj.weight"), vec![s, h]));
                    out.push((p("ffn.shared_experts.down_proj.weight"), vec![h, s]));
                }
            } else {
                let m = self.intermediate_size;
                out.push((p("ffn.gate_proj.weight"), vec![m, h]));
                out.push((p("ffn.up_proj.weight"), vec![m, h]));
                out.push((p("ffn.down_proj.weight"), vec![h, m]));
            }
        }
        out
    }

    /// Parameter count of [`Self::expected_keys`].
    pub fn num_params(&self) -> usize {
        self.expected_keys()
            .iter()
            .map(|(_, s)| s.iter().product::<usize>())
            .sum()
    }

    /// Parameters a token touches per forward (routed top-k + shared + dense parts).
    pub fn active_params(&self) -> usize {
        if !self.is_moe() {
            return self.num_params();
        }
        let per_expert = 3 * self.hidden_size * self.moe_intermediate_size;
        let moe_layers = (0..self.depth).filter(|&i| self.layer_is_moe(i)).count();
        self.num_params() - moe_layers * per_expert * (self.num_experts - self.num_experts_per_tok)
    }
}

/// Token count of the template text before the user prompt (`_compute_crop_start`):
/// what the processor makes of `PROMPT_TEMPLATE` cut at the `{}` marker.
pub fn prompt_crop_start(tokenizer: &tokenizers::Tokenizer) -> Result<usize, String> {
    let marker = PROMPT_TEMPLATE
        .find("{}")
        .ok_or("lingbot template has no {} marker")?;
    let enc = tokenizer
        .encode(&PROMPT_TEMPLATE[..marker], false)
        .map_err(|e| format!("lingbot crop tokenize: {e}"))?;
    Ok(enc.get_ids().len())
}

/// `tokenizer.json` of a Diffusers LingBot tree: `processor/` on the Hub
/// (`text_encoder/` carries an identical copy); `tokenizer/` for older packs.
pub fn tokenizer_path(root: &std::path::Path) -> Option<std::path::PathBuf> {
    ["processor", "text_encoder", "tokenizer"]
        .iter()
        .map(|d| root.join(d).join("tokenizer.json"))
        .find(|p| p.is_file())
}

/// Apply [`PROMPT_TEMPLATE`] and tokenize; returns `(ids, crop_start)`.
pub fn tokenize_lingbot_prompt(
    root: &std::path::Path,
    prompt: &str,
    max_length: usize,
) -> Result<(Vec<u32>, usize), String> {
    let path = tokenizer_path(root)
        .ok_or_else(|| format!("{}: no processor/tokenizer.json", root.display()))?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let crop = prompt_crop_start(&tokenizer)?;
    let body = PROMPT_TEMPLATE.replace("{}", prompt);
    let encoding = tokenizer
        .encode(body.as_str(), false)
        .map_err(|e| format!("lingbot qwen tokenize: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() > max_length {
        ids.truncate(max_length);
    }
    Ok((ids, crop))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_CONFIG: &str = include_str!("hub/moe_30b_transformer_config.json");
    const HUB_KEYS: &str = include_str!("hub/moe_30b_transformer_keys.tsv");

    #[test]
    fn dense_dims() {
        let c = LingBotTransformerConfig::dense_1_3b();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.depth, 24);
        assert_eq!(c.text_dim, 2560);
        assert_eq!(c.head_dim(), 128);
        assert!(!c.is_moe());
        c.validate().unwrap();
    }

    #[test]
    fn moe_preset_matches_hub_config() {
        let hub = LingBotTransformerConfig::from_hub_json(HUB_CONFIG).unwrap();
        assert_eq!(hub, LingBotTransformerConfig::moe_30b());
        assert_eq!(hub.hidden_size, 2048);
        assert_eq!(hub.depth, 48);
        assert_eq!(hub.moe_intermediate_size, 768);
        assert_eq!(hub.n_shared_experts, 1);
        assert_eq!((hub.n_group, hub.topk_group), (Some(4), Some(2)));
        assert_eq!(hub.routed_scaling_factor, 2.5);
        assert!((0..48).all(|i| hub.layer_is_moe(i)));
    }

    #[test]
    fn expected_keys_match_hub_index_and_headers() {
        let cfg = LingBotTransformerConfig::moe_30b();
        let want: std::collections::BTreeMap<String, Vec<usize>> =
            cfg.expected_keys().into_iter().collect();
        let mut hub = std::collections::BTreeMap::new();
        for line in HUB_KEYS.lines().filter(|l| !l.starts_with('#')) {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 3, "{line}");
            hub.insert(cols[0].to_string(), cols[2].to_string());
        }
        assert_eq!(hub.len(), 977);
        let want_keys: Vec<&String> = want.keys().collect();
        let hub_keys: Vec<&String> = hub.keys().collect();
        assert_eq!(want_keys, hub_keys);
        let mut checked = 0;
        for (k, shape) in &hub {
            if shape == "-" {
                continue;
            }
            let dims: Vec<usize> = shape.split('x').map(|d| d.parse().unwrap()).collect();
            assert_eq!(&dims, &want[k], "{k}");
            checked += 1;
        }
        // Shards 1 and 13: the globals plus blocks 0, 46 and 47.
        assert!(checked > 60, "{checked}");
    }

    #[test]
    fn moe_param_counts() {
        let c = LingBotTransformerConfig::moe_30b();
        let total = c.num_params();
        // 60.27 GB of bf16/f32 on the Hub; ~30B parameters.
        assert!((29_500_000_000..30_500_000_000).contains(&total), "{total}");
        let active = c.active_params();
        assert!((2_000_000_000..3_500_000_000).contains(&active), "{active}");
    }

    #[test]
    fn tiny_moe_validates() {
        let c = LingBotTransformerConfig::tiny_moe();
        c.validate().unwrap();
        assert!(!c.layer_is_moe(0));
        assert!(c.layer_is_moe(1));
        assert_eq!(c.head_dim(), 16);
    }

    #[test]
    fn crop_constant() {
        assert_eq!(PROMPT_CROP_START, 140);
        assert!(PROMPT_TEMPLATE.contains("<|im_start|>assistant"));
        assert!(DEFAULT_NEGATIVE_PROMPT.starts_with("{\"universal_negative\""));
    }
}
