//! Local Diffusers checkpoint discovery for FLUX.1. Never hits the Hub from tests.

use std::path::{Path, PathBuf};

use super::FluxTransformerConfig;

pub fn local_flux1(repo: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FASTVIDEO_WEIGHTS") {
        let p = PathBuf::from(path);
        if looks_like_flux1(&p) {
            return Some(p);
        }
    }
    crate::wan::weights::hf_snapshot(repo).filter(|root| looks_like_flux1(root))
}

pub fn looks_like_flux1(root: &Path) -> bool {
    root.join("transformer").is_dir()
        && root.join("vae").is_dir()
        && (root.join("text_encoder").is_dir() || root.join("text_encoder_2").is_dir())
}

pub fn transformer_config_json(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join("transformer/config.json")).ok()
}

/// Override arch fields from a Diffusers `transformer/config.json` body.
pub fn arch_from_transformer_config(
    cfg: &FluxTransformerConfig,
    raw: &str,
) -> Result<FluxTransformerConfig, String> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("transformer/config.json: {e}"))?;
    let mut out = cfg.clone();
    if let Some(n) = v.get("num_layers").and_then(|x| x.as_u64()) {
        out.num_layers = n as usize;
    }
    if let Some(n) = v.get("num_single_layers").and_then(|x| x.as_u64()) {
        out.num_single_layers = n as usize;
    }
    if let Some(n) = v.get("in_channels").and_then(|x| x.as_u64()) {
        out.in_channels = n as usize;
        if out.out_channels == cfg.out_channels {
            out.out_channels = out.in_channels;
        }
    }
    if let Some(n) = v.get("out_channels").and_then(|x| x.as_u64()) {
        out.out_channels = n as usize;
    }
    if let Some(n) = v.get("num_attention_heads").and_then(|x| x.as_u64()) {
        out.num_attention_heads = n as usize;
    }
    if let Some(n) = v.get("attention_head_dim").and_then(|x| x.as_u64()) {
        out.attention_head_dim = n as usize;
    }
    if let Some(n) = v.get("joint_attention_dim").and_then(|x| x.as_u64()) {
        out.joint_attention_dim = n as usize;
    }
    if let Some(n) = v.get("pooled_projection_dim").and_then(|x| x.as_u64()) {
        out.pooled_projection_dim = n as usize;
    }
    if let Some(g) = v.get("guidance_embeds").and_then(|x| x.as_bool()) {
        out.guidance_embeds = g;
    }
    if let Some(t) = v.get("rope_theta").and_then(|x| x.as_f64()) {
        out.rope_theta = t as f32;
    }
    if let Some(arr) = v.get("axes_dims_rope").and_then(|x| x.as_array()) {
        if arr.len() == 3 {
            let mut axes = [0usize; 3];
            for (i, item) in arr.iter().enumerate() {
                axes[i] = item.as_u64().unwrap_or(16) as usize;
            }
            out.axes_dims_rope = axes;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flux1_dev_config_json() {
        let raw = r#"{
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": true,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "pooled_projection_dim": 768,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38
        }"#;
        let cfg = arch_from_transformer_config(&FluxTransformerConfig::tiny(), raw).unwrap();
        assert_eq!(cfg.num_layers, 19);
        assert_eq!(cfg.in_channels, 64);
        assert_eq!(cfg.joint_attention_dim, 4096);
        assert_eq!(cfg.axes_dims_rope, [16, 56, 56]);
        assert!(cfg.guidance_embeds);
    }
}
