//! SANA-Video architecture configs, parsed from the Diffusers tree
//! (`Efficient-Large-Model/SANA-Video_2B_480p_diffusers`).
//!
//! Reference: Diffusers `SanaVideoTransformer3DModel`
//! (`models/transformers/transformer_sana_video.py`) and the transformers
//! `Gemma2Model` the pipeline encodes prompts with.

use serde_json::Value;

/// `transformer/config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct SanaVideoTransformerConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub num_layers: usize,
    pub num_cross_attention_heads: usize,
    pub cross_attention_head_dim: usize,
    pub cross_attention_dim: usize,
    /// Text encoder width (Gemma-2-2B: 2304).
    pub caption_channels: usize,
    /// GLUMBTempConv expansion: hidden = `int(mlp_ratio * dim)`.
    pub mlp_ratio: f64,
    /// Bias on the self-attention q/k/v projections (`attention_bias`).
    pub attention_bias: bool,
    /// 30 selects the 480p aspect-ratio bins, 22 the 720p bins.
    pub sample_size: usize,
    /// `(p_t, p_h, p_w)`.
    pub patch_size: [usize; 3],
    pub norm_eps: f32,
    pub guidance_embeds: bool,
    pub rope_max_seq_len: usize,
}

/// The self- and cross-attention q/k norm (`Attention(eps=1e-5)` default).
pub const SANA_QK_NORM_EPS: f32 = 1e-5;
/// `caption_norm` = `RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)`.
pub const SANA_CAPTION_NORM_EPS: f32 = 1e-5;
/// The linear-attention normaliser epsilon (`z = 1 / (k_sum · q + 1e-15)`).
pub const SANA_LINEAR_ATTN_EPS: f32 = 1e-15;

impl SanaVideoTransformerConfig {
    /// `SANA-Video_2B_480p_diffusers` @ db5f398b `transformer/config.json`.
    pub fn sana_video_2b_480p() -> Self {
        Self {
            in_channels: 16,
            out_channels: 16,
            num_attention_heads: 20,
            attention_head_dim: 112,
            num_layers: 20,
            num_cross_attention_heads: 20,
            cross_attention_head_dim: 112,
            cross_attention_dim: 2240,
            caption_channels: 2304,
            mlp_ratio: 3.0,
            attention_bias: false,
            sample_size: 30,
            patch_size: [1, 2, 2],
            norm_eps: 1e-6,
            guidance_embeds: false,
            rope_max_seq_len: 1024,
        }
    }

    /// A few-channel model with the same graph (tests).
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 12,
            num_layers: 2,
            num_cross_attention_heads: 2,
            cross_attention_head_dim: 12,
            cross_attention_dim: 24,
            caption_channels: 16,
            mlp_ratio: 2.0,
            attention_bias: false,
            sample_size: 30,
            patch_size: [1, 2, 2],
            norm_eps: 1e-6,
            guidance_embeds: false,
            rope_max_seq_len: 64,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// `GLUMBTempConv` hidden channels: `int(expand_ratio * in_channels)`.
    pub fn ff_hidden(&self) -> usize {
        (self.mlp_ratio * self.inner_dim() as f64) as usize
    }

    pub fn patch_volume(&self) -> usize {
        self.patch_size.iter().product()
    }

    /// Parse `transformer/config.json`. Missing keys take the Diffusers
    /// `__init__` defaults; a field the port does not implement is an error.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| format!("transformer config.json: {e}"))?;
        let d = Self::sana_video_2b_480p();
        let u = |k: &str, def: usize| -> Result<usize, String> {
            match v.get(k) {
                None | Some(Value::Null) => Ok(def),
                Some(x) => x
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("transformer config.json: {k} is not an integer")),
            }
        };
        let f = |k: &str, def: f64| -> Result<f64, String> {
            match v.get(k) {
                None | Some(Value::Null) => Ok(def),
                Some(x) => x
                    .as_f64()
                    .ok_or_else(|| format!("transformer config.json: {k} is not a number")),
            }
        };
        let b = |k: &str, def: bool| -> Result<bool, String> {
            match v.get(k) {
                None | Some(Value::Null) => Ok(def),
                Some(x) => x
                    .as_bool()
                    .ok_or_else(|| format!("transformer config.json: {k} is not a bool")),
            }
        };
        let patch_size = match v.get("patch_size") {
            Some(Value::Array(a)) if a.len() == 3 => {
                let mut p = [0usize; 3];
                for (slot, x) in p.iter_mut().zip(a) {
                    *slot = x
                        .as_u64()
                        .ok_or("transformer config.json: patch_size holds a non-integer")?
                        as usize;
                }
                p
            }
            None => d.patch_size,
            Some(other) => {
                return Err(format!(
                    "transformer config.json: patch_size {other} is not [t, h, w]"
                ))
            }
        };
        match v.get("qk_norm") {
            None | Some(Value::Null) => {
                return Err("transformer config.json: qk_norm unset; the port implements rms_norm_across_heads only".into())
            }
            Some(Value::String(s)) if s == "rms_norm_across_heads" => {}
            Some(other) => {
                return Err(format!(
                    "transformer config.json: qk_norm {other} (port implements rms_norm_across_heads)"
                ))
            }
        }
        if b("norm_elementwise_affine", false)? {
            return Err("transformer config.json: norm_elementwise_affine=true is not ported".into());
        }
        let in_channels = u("in_channels", d.in_channels)?;
        let cfg = Self {
            in_channels,
            out_channels: u("out_channels", in_channels)?,
            num_attention_heads: u("num_attention_heads", d.num_attention_heads)?,
            attention_head_dim: u("attention_head_dim", d.attention_head_dim)?,
            num_layers: u("num_layers", d.num_layers)?,
            num_cross_attention_heads: u("num_cross_attention_heads", d.num_cross_attention_heads)?,
            cross_attention_head_dim: u("cross_attention_head_dim", d.cross_attention_head_dim)?,
            cross_attention_dim: u("cross_attention_dim", d.cross_attention_dim)?,
            caption_channels: u("caption_channels", d.caption_channels)?,
            mlp_ratio: f("mlp_ratio", 2.5)?,
            attention_bias: b("attention_bias", false)?,
            sample_size: u("sample_size", d.sample_size)?,
            patch_size,
            norm_eps: f("norm_eps", 1e-6)? as f32,
            guidance_embeds: b("guidance_embeds", false)?,
            rope_max_seq_len: u("rope_max_seq_len", d.rope_max_seq_len)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.guidance_embeds {
            return Err("SANA-Video: guidance_embeds=true (SanaCombinedTimestepGuidanceEmbeddings) is not ported".into());
        }
        if self.num_cross_attention_heads * self.cross_attention_head_dim != self.inner_dim()
            || self.cross_attention_dim != self.inner_dim()
        {
            return Err(format!(
                "SANA-Video: cross attention {}x{} (dim {}) must equal inner dim {}",
                self.num_cross_attention_heads,
                self.cross_attention_head_dim,
                self.cross_attention_dim,
                self.inner_dim()
            ));
        }
        if self.patch_size[0] != 1 {
            return Err("SANA-Video: temporal patch size > 1 is not ported".into());
        }
        if self.attention_head_dim % 2 != 0 {
            return Err("SANA-Video: odd head dim has no rotary layout".into());
        }
        Ok(())
    }
}

/// `text_encoder/config.json` (`Gemma2Model`).
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma2TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Softmax scale is `query_pre_attn_scalar ** -0.5`.
    pub query_pre_attn_scalar: f64,
    /// `tanh(scores / cap) * cap` before the mask (Gemma-2 only).
    pub attn_logit_softcapping: Option<f32>,
    pub sliding_window: usize,
    /// `true` for a sliding-window layer (`layer_types`).
    pub sliding_layers: Vec<bool>,
}

impl Gemma2TextConfig {
    /// `SANA-Video_2B_480p_diffusers/text_encoder/config.json` (Gemma-2-2B-it).
    pub fn gemma2_2b() -> Self {
        Self {
            vocab_size: 256_000,
            hidden_size: 2304,
            intermediate_size: 9216,
            num_hidden_layers: 26,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            query_pre_attn_scalar: 256.0,
            attn_logit_softcapping: Some(50.0),
            sliding_window: 4096,
            sliding_layers: (0..26).map(|i| i % 2 == 0).collect(),
        }
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| format!("text_encoder config.json: {e}"))?;
        let u = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("text_encoder config.json: {k} missing"))
        };
        let f = |k: &str| -> Result<f64, String> {
            v.get(k)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("text_encoder config.json: {k} missing"))
        };
        if v.get("model_type").and_then(Value::as_str) != Some("gemma2") {
            return Err("text_encoder config.json: model_type is not gemma2".into());
        }
        let act = v
            .get("hidden_activation")
            .or_else(|| v.get("hidden_act"))
            .and_then(Value::as_str)
            .unwrap_or("gelu_pytorch_tanh");
        if act != "gelu_pytorch_tanh" {
            return Err(format!("text_encoder config.json: activation {act} not ported"));
        }
        let layers = u("num_hidden_layers")?;
        let sliding_layers = match v.get("layer_types") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|t| match t.as_str() {
                    Some("sliding_attention") => Ok(true),
                    Some("full_attention") => Ok(false),
                    other => Err(format!("text_encoder config.json: layer type {other:?}")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            // transformers' Gemma-2 default: even layers slide.
            _ => (0..layers).map(|i| i % 2 == 0).collect(),
        };
        if sliding_layers.len() != layers {
            return Err("text_encoder config.json: layer_types length".into());
        }
        Ok(Self {
            vocab_size: u("vocab_size")?,
            hidden_size: u("hidden_size")?,
            intermediate_size: u("intermediate_size")?,
            num_hidden_layers: layers,
            num_attention_heads: u("num_attention_heads")?,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            rms_norm_eps: f("rms_norm_eps")? as f32,
            rope_theta: f("rope_theta")?,
            query_pre_attn_scalar: f("query_pre_attn_scalar")?,
            attn_logit_softcapping: v
                .get("attn_logit_softcapping")
                .and_then(Value::as_f64)
                .map(|c| c as f32),
            sliding_window: u("sliding_window")?,
            sliding_layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUB_TRANSFORMER: &str = r#"{
      "_class_name": "SanaVideoTransformer3DModel", "attention_bias": false,
      "attention_head_dim": 112, "caption_channels": 2304, "cross_attention_dim": 2240,
      "cross_attention_head_dim": 112, "dropout": 0.0, "guidance_embeds": false,
      "guidance_embeds_scale": 0.1, "in_channels": 16, "interpolation_scale": null,
      "mlp_ratio": 3.0, "norm_elementwise_affine": false, "norm_eps": 1e-06,
      "num_attention_heads": 20, "num_cross_attention_heads": 20, "num_layers": 20,
      "out_channels": 16, "patch_size": [1, 2, 2], "qk_norm": "rms_norm_across_heads",
      "rope_max_seq_len": 1024, "sample_size": 30 }"#;

    #[test]
    fn parses_the_hub_transformer_config() {
        let cfg = SanaVideoTransformerConfig::from_json(HUB_TRANSFORMER).unwrap();
        assert_eq!(cfg, SanaVideoTransformerConfig::sana_video_2b_480p());
        assert_eq!(cfg.inner_dim(), 2240);
        assert_eq!(cfg.ff_hidden(), 6720);
    }

    #[test]
    fn rejects_unported_variants() {
        let g = HUB_TRANSFORMER.replace("\"guidance_embeds\": false", "\"guidance_embeds\": true");
        assert!(SanaVideoTransformerConfig::from_json(&g).is_err());
        let q = HUB_TRANSFORMER.replace("rms_norm_across_heads", "rms_norm");
        assert!(SanaVideoTransformerConfig::from_json(&q).is_err());
    }

    #[test]
    fn parses_the_hub_gemma2_config() {
        let mut types = String::new();
        for i in 0..26 {
            if i > 0 {
                types.push(',');
            }
            types.push_str(if i % 2 == 0 {
                "\"sliding_attention\""
            } else {
                "\"full_attention\""
            });
        }
        let text = format!(
            r#"{{"model_type": "gemma2", "attn_logit_softcapping": 50.0, "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 2304,
            "intermediate_size": 9216, "layer_types": [{types}], "num_attention_heads": 8,
            "num_hidden_layers": 26, "num_key_value_heads": 4, "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 4096,
            "vocab_size": 256000}}"#
        );
        assert_eq!(
            Gemma2TextConfig::from_json(&text).unwrap(),
            Gemma2TextConfig::gemma2_2b()
        );
    }
}
