//! Wan transformer architecture defaults.
//!
//! Numbers follow FastVideo `fastvideo/models/wan/config.py` and the published
//! Diffusers `transformer/config.json` files for each family.

#[derive(Debug, Clone, PartialEq)]
pub struct WanVideoArchConfig {
    pub patch_size: [usize; 3],
    pub text_len: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub eps: f32,
    pub rope_max_seq_len: usize,
    pub image_dim: Option<usize>,
    pub added_kv_proj_dim: Option<usize>,
    /// Wan2.2 MoE: `t >= boundary_ratio * num_train` uses high-noise expert.
    pub boundary_ratio: Option<f32>,
    /// Causal / Self-Forcing temporal window in latent frames (FastVideo
    /// `local_attn_size`). `-1` is global attention, capped at
    /// `sliding_window_num_frames` frames of KV cache.
    pub local_attn_size: i32,
    /// Frames at the head of the KV cache kept when it rolls (`sink_size`).
    pub sink_size: usize,
    pub causal: bool,
    /// Latent frames generated (and mutually visible) per autoregressive
    /// block (FastVideo `num_frames_per_block`).
    pub num_frames_per_block: usize,
    /// KV cache length in frames when `local_attn_size == -1`
    /// (FastVideo `sliding_window_num_frames`).
    pub sliding_window_num_frames: usize,
}

impl WanVideoArchConfig {
    pub fn hidden_size(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    pub fn is_i2v(&self) -> bool {
        self.image_dim.is_some() || self.in_channels > self.out_channels
    }

    pub fn is_moe(&self) -> bool {
        self.boundary_ratio.is_some()
    }

    fn base() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 512,
            num_attention_heads: 40,
            attention_head_dim: 128,
            in_channels: 16,
            out_channels: 16,
            text_dim: 4096,
            freq_dim: 256,
            ffn_dim: 13824,
            num_layers: 40,
            eps: 1e-6,
            rope_max_seq_len: 1024,
            image_dim: None,
            added_kv_proj_dim: None,
            boundary_ratio: None,
            local_attn_size: -1,
            sink_size: 0,
            causal: false,
            num_frames_per_block: 3,
            sliding_window_num_frames: 21,
        }
    }

    /// Wan 2.1 T2V 1.3B / FastWan 1.3B. HF `transformer/config.json`.
    pub fn wan_t2v_1_3b() -> Self {
        Self {
            num_attention_heads: 12,
            ffn_dim: 8960,
            num_layers: 30,
            ..Self::base()
        }
    }

    /// FastVideo `WanVideoArchConfig` defaults (14B T2V).
    pub fn wan_t2v_14b() -> Self {
        Self::base()
    }

    /// Wan 2.1 I2V 14B (480p/720p share the DiT; 36-channel concat).
    pub fn wan_i2v_14b() -> Self {
        Self {
            in_channels: 36,
            image_dim: Some(1280),
            added_kv_proj_dim: Some(5120),
            ..Self::base()
        }
    }

    /// Wan 2.2 TI2V 5B (48-channel VAE).
    pub fn wan_2_2_ti2v_5b() -> Self {
        Self {
            num_attention_heads: 24,
            ffn_dim: 14336,
            num_layers: 30,
            in_channels: 48,
            out_channels: 48,
            ..Self::base()
        }
    }

    /// Wan 2.2 T2V A14B high/low-noise experts (same 14B DiT, MoE routing).
    pub fn wan_2_2_t2v_a14b() -> Self {
        Self {
            boundary_ratio: Some(0.875),
            ..Self::base()
        }
    }

    /// Wan 2.2 I2V A14B.
    pub fn wan_2_2_i2v_a14b() -> Self {
        Self {
            in_channels: 36,
            image_dim: Some(1280),
            added_kv_proj_dim: Some(5120),
            boundary_ratio: Some(0.875),
            ..Self::base()
        }
    }

    /// Self-Forcing causal Wan 2.1 1.3B (`wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`:
    /// its `transformer/config.json` sets none of the causal fields, so
    /// FastVideo's `WanVideoArchConfig` defaults apply: global attention over
    /// a 21-frame KV cache, no sink, 3 frames per block).
    pub fn sf_wan_t2v_1_3b() -> Self {
        Self {
            causal: true,
            local_attn_size: -1,
            sink_size: 0,
            ..Self::wan_t2v_1_3b()
        }
    }

    pub fn tiny() -> Self {
        Self {
            patch_size: [1, 2, 2],
            text_len: 8,
            num_attention_heads: 2,
            attention_head_dim: 8,
            in_channels: 4,
            out_channels: 4,
            text_dim: 16,
            freq_dim: 16,
            ffn_dim: 32,
            num_layers: 1,
            eps: 1e-6,
            rope_max_seq_len: 64,
            image_dim: None,
            added_kv_proj_dim: None,
            boundary_ratio: None,
            local_attn_size: -1,
            sink_size: 0,
            causal: false,
            num_frames_per_block: 3,
            sliding_window_num_frames: 21,
        }
    }

    /// Small I2V graph for tests (real 36-channel concat + added KV, not 14B).
    pub fn i2v_block() -> Self {
        Self {
            in_channels: 36,
            out_channels: 16,
            image_dim: Some(32),
            added_kv_proj_dim: Some(32),
            num_layers: 1,
            num_attention_heads: 2,
            attention_head_dim: 8,
            ffn_dim: 32,
            text_len: 8,
            text_dim: 16,
            freq_dim: 16,
            rope_max_seq_len: 64,
            ..Self::tiny()
        }
    }

    pub fn from_preset(preset: &str) -> Self {
        match preset {
            "wan_t2v_1_3b"
            | "fast_wan_t2v_480p"
            | "wan_fun_1_3b_inp"
            | "wan_fun_1_3b_control"
            | "turbo_t2v_1_3b" => Self::wan_t2v_1_3b(),
            "wan_t2v_14b" | "turbo_t2v_14b" => Self::wan_t2v_14b(),
            "wan_i2v_14b_480p" | "wan_i2v_14b_720p" => Self::wan_i2v_14b(),
            "wan_2_2_ti2v_5b" | "fast_wan_2_2_ti2v_5b" | "lucy_edit_dev" => Self::wan_2_2_ti2v_5b(),
            "wan_2_2_t2v_a14b" | "sf_wan_2_2_t2v_a14b" => Self::wan_2_2_t2v_a14b(),
            "wan_2_2_i2v_a14b" | "sf_wan_2_2_i2v_a14b" | "turbo_i2v_a14b" => {
                let mut cfg = Self::wan_2_2_i2v_a14b();
                if preset == "turbo_i2v_a14b" {
                    cfg.boundary_ratio = Some(0.9);
                }
                cfg
            }
            "sf_wan_t2v_1_3b" => Self::sf_wan_t2v_1_3b(),
            _ => Self::wan_t2v_1_3b(),
        }
    }
}

/// Diffusers → FastVideo weight-name rewrite rules (regex pairs).
pub const PARAM_NAMES_MAPPING: &[(&str, &str)] = &[
    (r"^patch_embedding\.(.*)$", r"patch_embedding.proj.$1"),
    (
        r"^condition_embedder\.text_embedder\.linear_1\.(.*)$",
        r"condition_embedder.text_embedder.fc_in.$1",
    ),
    (
        r"^condition_embedder\.text_embedder\.linear_2\.(.*)$",
        r"condition_embedder.text_embedder.fc_out.$1",
    ),
    (
        r"^condition_embedder\.time_embedder\.linear_1\.(.*)$",
        r"condition_embedder.time_embedder.mlp.fc_in.$1",
    ),
    (
        r"^condition_embedder\.time_embedder\.linear_2\.(.*)$",
        r"condition_embedder.time_embedder.mlp.fc_out.$1",
    ),
    (
        r"^condition_embedder\.time_proj\.(.*)$",
        r"condition_embedder.time_modulation.linear.$1",
    ),
];

/// Keys that must exist in Wan 2.1 T2V 1.3B Diffusers `weight_map`.
pub const WAN_T2V_1_3B_REQUIRED_KEYS: &[&str] = &[
    "patch_embedding.weight",
    "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_2.weight",
    "condition_embedder.time_proj.weight",
    "condition_embedder.text_embedder.linear_1.weight",
    "condition_embedder.text_embedder.linear_2.weight",
    "blocks.0.attn1.to_q.weight",
    "blocks.0.attn1.to_out.0.weight",
    "blocks.0.attn2.to_q.weight",
    "blocks.0.ffn.net.0.proj.weight",
    "blocks.0.ffn.net.2.weight",
    "blocks.29.attn1.to_q.weight",
    "proj_out.weight",
    "scale_shift_table",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_matches_hf_configs() {
        let b = WanVideoArchConfig::wan_t2v_1_3b();
        assert_eq!(b.hidden_size(), 1536);
        assert_eq!(b.num_layers, 30);
        assert_eq!(b.ffn_dim, 8960);
        let f = WanVideoArchConfig::wan_t2v_14b();
        assert_eq!(f.hidden_size(), 5120);
        assert_eq!(f.num_layers, 40);
        let i = WanVideoArchConfig::wan_i2v_14b();
        assert_eq!(i.in_channels, 36);
        assert_eq!(i.out_channels, 16);
        assert_eq!(i.image_dim, Some(1280));
        assert_eq!(i.added_kv_proj_dim, Some(5120));
        let t = WanVideoArchConfig::wan_2_2_ti2v_5b();
        assert_eq!(t.hidden_size(), 3072);
        assert_eq!(t.in_channels, 48);
        assert_eq!(t.out_channels, 48);
        let moe = WanVideoArchConfig::wan_2_2_t2v_a14b();
        assert_eq!(moe.boundary_ratio, Some(0.875));
        assert!(moe.is_moe());
        let causal = WanVideoArchConfig::sf_wan_t2v_1_3b();
        assert!(causal.causal);
        assert_eq!(causal.local_attn_size, -1);
        assert_eq!(causal.sink_size, 0);
        assert_eq!(causal.num_frames_per_block, 3);
        assert_eq!(causal.sliding_window_num_frames, 21);
    }

    #[test]
    fn fun_inp_preset_is_i2v_ready_1_3b() {
        let cfg = WanVideoArchConfig::from_preset("wan_fun_1_3b_inp");
        // Fun InP shares the 1.3B DiT body; I2V packing uses 36-ch when in > out.
        // Diffusers Fun InP uses the T2V 1.3B channel layout at the DiT; conditioning
        // still goes through the I2V pack helpers when in_channels is raised by weights.
        assert_eq!(cfg.num_layers, 30);
        assert_eq!(cfg.hidden_size(), 1536);
        assert_eq!(cfg.out_channels, 16);
        // Candle/cudarc Fun InP registry maps to wan_t2v_1_3b arch (16-ch); I2V 14B is 36-ch.
        let i2v = WanVideoArchConfig::wan_i2v_14b();
        assert!(i2v.is_i2v());
        assert_eq!(i2v.in_channels, 36);
        assert_eq!(i2v.in_channels, 2 * i2v.out_channels + 4);
    }
}

/// The side multiple a Wan preset generates at exactly: the VAE's spatial
/// compression times the DiT's spatial patch (Wan 2.1: 8 x 2 = 16; Wan 2.2
/// TI2V-5B: 16 x 2 = 32). The pipeline floors other sizes to it silently.
pub fn canvas_multiple(preset: &str) -> usize {
    let dit = WanVideoArchConfig::from_preset(preset);
    let vae = if dit.out_channels == 48 {
        crate::wan::WanVaeConfig::wan_2_2()
    } else {
        crate::wan::WanVaeConfig::wan_2_1()
    };
    vae.spatial_compression() * dit.patch_size[1].max(1)
}

/// The request geometry a Wan preset generates exactly: height and width
/// positive multiples of [`canvas_multiple`], `4k + 1` frames (the VAE's
/// temporal compression), and for a causal preset latent frames in whole
/// blocks (`num_frames_per_block`). The serving engine's job check.
pub fn check_geometry(preset: &str, height: usize, width: usize, num_frames: usize) -> Result<(), String> {
    let dit = WanVideoArchConfig::from_preset(preset);
    let m = canvas_multiple(preset);
    if height == 0 || width == 0 || !height.is_multiple_of(m) || !width.is_multiple_of(m) {
        return Err(format!("wan: {width}x{height} — height and width must be positive multiples of {m}"));
    }
    let tc = if dit.out_channels == 48 {
        crate::wan::WanVaeConfig::wan_2_2()
    } else {
        crate::wan::WanVaeConfig::wan_2_1()
    }
    .temporal_compression();
    if num_frames == 0 || (num_frames - 1) % tc != 0 {
        return Err(format!("wan: {num_frames} frames — the frame count must be {tc}k + 1"));
    }
    let latents = (num_frames - 1) / tc + 1;
    let fpb = dit.num_frames_per_block.max(1);
    if dit.causal && latents % fpb != 0 {
        return Err(format!(
            "wan: {num_frames} frames are {latents} latent frames, not a multiple of the causal block ({fpb})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn wan_geometry() {
        assert_eq!(canvas_multiple("wan_2_2_ti2v_5b"), 32);
        assert_eq!(canvas_multiple("fast_wan_t2v_480p"), 16);
        assert!(check_geometry("wan_2_2_ti2v_5b", 704, 1280, 121).is_ok());
        assert!(check_geometry("wan_2_2_ti2v_5b", 720, 1280, 121).is_err());
        assert!(check_geometry("wan_2_2_ti2v_5b", 704, 1280, 120).is_err());
        assert!(check_geometry("fast_wan_t2v_480p", 480, 832, 81).is_ok());
        assert!(check_geometry("sf_wan_t2v_1_3b", 480, 832, 81).is_ok());
        assert!(check_geometry("sf_wan_t2v_1_3b", 480, 832, 77).is_err());
    }
}
