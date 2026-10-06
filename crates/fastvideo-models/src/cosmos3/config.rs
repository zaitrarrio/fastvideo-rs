//! Cosmos3-Super host configs. Spec: docs/ports/cosmos3.md.
//!
//! `nvidia/Cosmos3-Super` `transformer/config.json` (vendored under `hub/`)
//! describes a Mixture-of-Transformers: every one of the 64 Qwen3-VL-text
//! shaped layers has two weight sets, the *understanding* (text, causal) one
//! (`to_q`, `mlp`, `input_layernorm`, …) and the *generation* one (`add_q_proj`,
//! `mlp_moe_gen`, `input_layernorm_moe_gen`, …). "use_moe" in the config means
//! this two-tower routing by modality, not routed experts. Reference:
//! diffusers `models/transformers/transformer_cosmos3.py`.

use serde::Deserialize;

use crate::cosmos::sol::{
    OFFICIAL_FLOW_SHIFT, OFFICIAL_FPS, OFFICIAL_FRAMES, OFFICIAL_GUIDANCE, OFFICIAL_HEIGHT,
    OFFICIAL_STEPS, OFFICIAL_WIDTH,
};

/// Hub repo and the revision the vendored configs and key list were read at.
pub const SUPER_HUB_REPO: &str = "nvidia/Cosmos3-Super";
pub const SUPER_HUB_REVISION: &str = "f543c56225b2e04d0ad141e29655be3a45d9c455";

/// 64B text-to-video Super.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cosmos3Preset {
    Super64bT2v,
}

impl Cosmos3Preset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Super64bT2v => "cosmos3_super_64b_t2v",
        }
    }

    pub fn hub_repo(self) -> &'static str {
        SUPER_HUB_REPO
    }

    /// `(height, width, num_frames)` from `models/cosmos3.toml`.
    pub fn canvas(self) -> (usize, usize, usize) {
        (OFFICIAL_HEIGHT, OFFICIAL_WIDTH, OFFICIAL_FRAMES)
    }

    pub fn default_steps(self) -> usize {
        OFFICIAL_STEPS
    }

    pub fn guidance(self) -> f32 {
        OFFICIAL_GUIDANCE
    }

    /// Recorded in `models/cosmos3.toml` and passed by the HF example as
    /// `UniPCMultistepScheduler.from_config(..., flow_shift=10.0)`. The
    /// Hub scheduler sets `use_karras_sigmas`, whose branch of
    /// `set_timesteps` never reads `flow_shift` (see [`super::schedule`]).
    pub fn flow_shift(self) -> f64 {
        OFFICIAL_FLOW_SHIFT
    }

    pub fn fps(self) -> u32 {
        OFFICIAL_FPS
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cosmos3TransformerConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub latent_channel: usize,
    pub latent_patch_size: usize,
    /// `latent_channel * patch²` (192).
    pub patch_latent_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// `rope_scaling.mrope_section` (T, H, W frequency counts).
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub reset_spatial_ids: bool,
    pub temporal_modality_margin: usize,
    pub enable_fps_modulation: bool,
    pub base_fps: f64,
    pub timestep_scale: f32,
    pub qk_norm_for_text: bool,
    pub vocab_size: usize,
    pub sound_gen: bool,
    pub action_gen: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeScaling {
    #[serde(default)]
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
}

#[derive(Debug, Clone, Deserialize)]
struct HubConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    latent_channel: usize,
    latent_patch_size: usize,
    patch_latent_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f64,
    rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_true")]
    unified_3d_mrope_reset_spatial_ids: bool,
    unified_3d_mrope_temporal_modality_margin: usize,
    #[serde(default = "default_true")]
    enable_fps_modulation: bool,
    base_fps: f64,
    timestep_scale: f32,
    #[serde(default = "default_true")]
    qk_norm_for_text: bool,
    #[serde(default)]
    use_und_k_norm_for_gen: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default = "default_silu")]
    hidden_act: String,
    position_embedding_type: Option<String>,
    vocab_size: usize,
    #[serde(default)]
    sound_gen: bool,
    #[serde(default)]
    action_gen: bool,
}

fn default_true() -> bool {
    true
}

fn default_silu() -> String {
    "silu".into()
}

impl Cosmos3TransformerConfig {
    /// `nvidia/Cosmos3-Super` (64 layers × hidden 5120, 64 query / 8 KV heads
    /// of 128, SwiGLU 25600 in both towers, Wan 2.2 48-channel latents in 2×2
    /// patches, interleaved 3D M-RoPE 24/20/20 at theta 5e6).
    pub fn super_64b() -> Self {
        Self {
            hidden_size: 5120,
            num_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 25_600,
            latent_channel: 48,
            latent_patch_size: 2,
            patch_latent_dim: 192,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
            reset_spatial_ids: true,
            temporal_modality_margin: 15_000,
            enable_fps_modulation: true,
            base_fps: 24.0,
            timestep_scale: 0.001,
            qk_norm_for_text: true,
            vocab_size: 151_936,
            sound_gen: true,
            action_gen: true,
        }
    }

    /// Parse a Hub `transformer/config.json`; fails on variants this port does
    /// not implement (attention bias, relu² / Nemotron norms, a separate
    /// und-key norm for the generation path, non-interleaved M-RoPE).
    pub fn from_hub_json(text: &str) -> Result<Self, String> {
        let h: HubConfig =
            serde_json::from_str(text).map_err(|e| format!("cosmos3 config.json: {e}"))?;
        if h.attention_bias {
            return Err("cosmos3: attention_bias is not ported".into());
        }
        if h.hidden_act != "silu" {
            return Err(format!("cosmos3: hidden_act {:?} is not ported", h.hidden_act));
        }
        if h.use_und_k_norm_for_gen {
            return Err("cosmos3: use_und_k_norm_for_gen is not ported".into());
        }
        if h.position_embedding_type.as_deref().is_some_and(|p| p != "unified_3d_mrope") {
            return Err("cosmos3: only unified_3d_mrope is ported".into());
        }
        let rs = h.rope_scaling.ok_or("cosmos3: rope_scaling missing")?;
        let cfg = Self {
            hidden_size: h.hidden_size,
            num_layers: h.num_hidden_layers,
            num_attention_heads: h.num_attention_heads,
            num_key_value_heads: h.num_key_value_heads,
            head_dim: h.head_dim,
            intermediate_size: h.intermediate_size,
            latent_channel: h.latent_channel,
            latent_patch_size: h.latent_patch_size,
            patch_latent_dim: h.patch_latent_dim,
            rms_norm_eps: h.rms_norm_eps,
            rope_theta: h.rope_theta,
            mrope_section: rs.mrope_section,
            mrope_interleaved: rs.mrope_interleaved,
            reset_spatial_ids: h.unified_3d_mrope_reset_spatial_ids,
            temporal_modality_margin: h.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation: h.enable_fps_modulation,
            base_fps: h.base_fps,
            timestep_scale: h.timestep_scale,
            qk_norm_for_text: h.qk_norm_for_text,
            vocab_size: h.vocab_size,
            sound_gen: h.sound_gen,
            action_gen: h.action_gen,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads % self.num_key_value_heads.max(1) != 0 {
            return Err("cosmos3: query heads not a multiple of kv heads".into());
        }
        if self.patch_latent_dim != self.latent_channel * self.latent_patch_size.pow(2) {
            return Err("cosmos3: patch_latent_dim != C · p²".into());
        }
        if self.mrope_section.iter().sum::<usize>() != self.head_dim / 2 {
            return Err("cosmos3: mrope sections must cover head_dim / 2".into());
        }
        if !self.mrope_interleaved {
            return Err("cosmos3: only interleaved M-RoPE is ported".into());
        }
        Ok(())
    }

    /// Tiny MoT graph for unit tests: 2 layers, hidden 32, 4 query / 2 KV
    /// heads of 8, sections 2/1/1, latent 3 channels in 2×2 patches.
    pub fn tiny() -> Self {
        Self {
            hidden_size: 32,
            num_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 48,
            latent_channel: 3,
            latent_patch_size: 2,
            patch_latent_dim: 12,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            mrope_section: [2, 1, 1],
            mrope_interleaved: true,
            reset_spatial_ids: true,
            temporal_modality_margin: 15_000,
            enable_fps_modulation: true,
            base_fps: 24.0,
            timestep_scale: 0.001,
            qk_norm_for_text: true,
            vocab_size: 50,
            sound_gen: false,
            action_gen: false,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Keys of one layer's tower: `(und, gen)` names for each role.
    pub fn layer_keys(&self, layer: usize) -> Vec<(String, Vec<usize>)> {
        let (h, q, kv, m, hd) = (
            self.hidden_size,
            self.q_dim(),
            self.kv_dim(),
            self.intermediate_size,
            self.head_dim,
        );
        let p = |n: &str| format!("layers.{layer}.{n}");
        let mut out = vec![
            (p("input_layernorm.weight"), vec![h]),
            (p("input_layernorm_moe_gen.weight"), vec![h]),
            (p("post_attention_layernorm.weight"), vec![h]),
            (p("post_attention_layernorm_moe_gen.weight"), vec![h]),
            (p("self_attn.to_q.weight"), vec![q, h]),
            (p("self_attn.to_k.weight"), vec![kv, h]),
            (p("self_attn.to_v.weight"), vec![kv, h]),
            (p("self_attn.to_out.weight"), vec![h, q]),
            (p("self_attn.add_q_proj.weight"), vec![q, h]),
            (p("self_attn.add_k_proj.weight"), vec![kv, h]),
            (p("self_attn.add_v_proj.weight"), vec![kv, h]),
            (p("self_attn.to_add_out.weight"), vec![h, q]),
            (p("self_attn.norm_added_q.weight"), vec![hd]),
            (p("self_attn.norm_added_k.weight"), vec![hd]),
        ];
        if self.qk_norm_for_text {
            out.push((p("self_attn.norm_q.weight"), vec![hd]));
            out.push((p("self_attn.norm_k.weight"), vec![hd]));
        }
        for mlp in ["mlp", "mlp_moe_gen"] {
            out.push((p(&format!("{mlp}.gate_proj.weight")), vec![m, h]));
            out.push((p(&format!("{mlp}.up_proj.weight")), vec![m, h]));
            out.push((p(&format!("{mlp}.down_proj.weight")), vec![h, m]));
        }
        out
    }

    /// Every tensor the T2V path reads (`embed_tokens`, both towers, the
    /// vision projections, the timestep MLP and the final gen norm).
    pub fn t2v_keys(&self) -> Vec<(String, Vec<usize>)> {
        let h = self.hidden_size;
        let mut out = vec![
            ("embed_tokens.weight".to_string(), vec![self.vocab_size, h]),
            ("norm_moe_gen.weight".into(), vec![h]),
            ("proj_in.weight".into(), vec![h, self.patch_latent_dim]),
            ("proj_in.bias".into(), vec![h]),
            ("proj_out.weight".into(), vec![self.patch_latent_dim, h]),
            ("proj_out.bias".into(), vec![self.patch_latent_dim]),
            ("time_embedder.linear_1.weight".into(), vec![h, 256]),
            ("time_embedder.linear_1.bias".into(), vec![h]),
            ("time_embedder.linear_2.weight".into(), vec![h, h]),
            ("time_embedder.linear_2.bias".into(), vec![h]),
        ];
        for i in 0..self.num_layers {
            out.extend(self.layer_keys(i));
        }
        out
    }

    /// Keys on the Hub the T2V path does not read: the text head, the und
    /// final norm (only the gen output is decoded) and the sound / action
    /// heads.
    pub fn unused_keys() -> &'static [&'static str] {
        &[
            "lm_head.weight",
            "norm.weight",
            "audio_modality_embed",
            "audio_proj_in.weight",
            "audio_proj_in.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
            "action_modality_embed",
            "action_proj_in.fc.weight",
            "action_proj_in.bias.weight",
            "action_proj_out.fc.weight",
            "action_proj_out.bias.weight",
        ]
    }

    /// Parameters of one tower (all layers), and of the gen-only resident set.
    pub fn tower_params(&self) -> usize {
        let (h, q, kv, m) = (self.hidden_size, self.q_dim(), self.kv_dim(), self.intermediate_size);
        let per = q * h + 2 * kv * h + h * q + 3 * m * h + 2 * h + 2 * self.head_dim;
        per * self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_CONFIG: &str = include_str!("hub/super_transformer_config.json");
    const HUB_KEYS: &str = include_str!("hub/super_transformer_keys.tsv");

    #[test]
    fn official_canvas_matches_sol() {
        let p = Cosmos3Preset::Super64bT2v;
        assert_eq!(p.as_str(), "cosmos3_super_64b_t2v");
        assert_eq!(p.canvas(), (720, 1280, 189));
        assert_eq!(p.default_steps(), 35);
        assert_eq!(p.guidance(), 6.0);
        assert_eq!(p.flow_shift(), 10.0);
        assert_eq!(p.fps(), 24);
        assert_eq!(p.hub_repo(), "nvidia/Cosmos3-Super");
    }

    #[test]
    fn super_preset_matches_hub_config() {
        let hub = Cosmos3TransformerConfig::from_hub_json(HUB_CONFIG).unwrap();
        assert_eq!(hub, Cosmos3TransformerConfig::super_64b());
    }

    #[test]
    fn keys_match_hub_index_and_headers() {
        let cfg = Cosmos3TransformerConfig::super_64b();
        let mut want: std::collections::BTreeMap<String, Vec<usize>> =
            cfg.t2v_keys().into_iter().collect();
        let mut hub = std::collections::BTreeMap::new();
        for line in HUB_KEYS.lines().filter(|l| !l.starts_with('#')) {
            let cols: Vec<&str> = line.split('\t').collect();
            hub.insert(cols[0].to_string(), cols[2].to_string());
        }
        assert_eq!(hub.len(), 1430);
        for k in Cosmos3TransformerConfig::unused_keys() {
            assert!(hub.remove(*k).is_some(), "{k} not on the Hub");
        }
        let a: Vec<&String> = want.keys().collect();
        let b: Vec<&String> = hub.keys().collect();
        assert_eq!(a, b);
        let mut checked = 0;
        for (k, shape) in &hub {
            if shape == "-" {
                continue;
            }
            let dims: Vec<usize> = shape.split('x').map(|d| d.parse().unwrap()).collect();
            assert_eq!(&dims, want.get(k).unwrap(), "{k}");
            checked += 1;
        }
        assert!(checked >= 30, "{checked}");
        want.clear();
    }

    #[test]
    fn tower_sizes() {
        let c = Cosmos3TransformerConfig::super_64b();
        let tower = c.tower_params();
        // ≈31.2B per tower; two towers + embeddings + lm_head ≈ 64B.
        assert!((31_000_000_000..31_500_000_000).contains(&tower), "{tower}");
        c.validate().unwrap();
        Cosmos3TransformerConfig::tiny().validate().unwrap();
    }
}
