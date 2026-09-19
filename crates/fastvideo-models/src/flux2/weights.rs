//! Local Diffusers checkpoint discovery for Flux2. Never hits the Hub from tests.

use std::path::{Path, PathBuf};

use super::config::Flux2ArchConfig;

/// `FASTVIDEO_WEIGHTS` or the Hugging Face hub snapshot for `repo`.
pub fn local_flux2(repo: &str) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FASTVIDEO_WEIGHTS") {
        let p = PathBuf::from(path);
        if looks_like_flux2(&p) {
            return Some(p);
        }
    }
    hf_snapshot(repo).filter(|root| looks_like_flux2(root))
}

pub fn looks_like_flux2(root: &Path) -> bool {
    root.join("transformer").is_dir() && root.join("vae").is_dir()
}

pub fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    crate::wan::weights::hf_snapshot(repo)
}

pub fn transformer_config_json(root: &Path) -> Option<String> {
    std::fs::read_to_string(root.join("transformer/config.json")).ok()
}

/// Override arch fields from a Diffusers `transformer/config.json` body.
pub fn arch_from_transformer_config(cfg: &Flux2ArchConfig, raw: &str) -> Result<Flux2ArchConfig, String> {
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
    if let Some(n) = v.get("timestep_guidance_channels").and_then(|x| x.as_u64()) {
        out.timestep_guidance_channels = n as usize;
    }
    if let Some(g) = v.get("guidance_embeds").and_then(|x| x.as_bool()) {
        out.guidance_embeds = g;
    }
    if let Some(r) = v.get("mlp_ratio").and_then(|x| x.as_f64()) {
        out.mlp_ratio = r as f32;
    }
    if let Some(t) = v.get("rope_theta").and_then(|x| x.as_f64()) {
        out.rope_theta = t as f32;
    }
    if let Some(arr) = v.get("axes_dims_rope").and_then(|x| x.as_array()) {
        if arr.len() == 4 {
            let mut axes = [0usize; 4];
            for (i, item) in arr.iter().enumerate() {
                axes[i] = item.as_u64().unwrap_or(32) as usize;
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
    fn parses_klein_4b_config_json() {
        let raw = r#"{
            "attention_head_dim": 128,
            "axes_dims_rope": [32, 32, 32, 32],
            "guidance_embeds": false,
            "in_channels": 128,
            "joint_attention_dim": 7680,
            "mlp_ratio": 3.0,
            "num_attention_heads": 24,
            "num_layers": 5,
            "num_single_layers": 20,
            "rope_theta": 2000
        }"#;
        let cfg = arch_from_transformer_config(&Flux2ArchConfig::flux2_dev(), raw).unwrap();
        assert_eq!(cfg.num_layers, 5);
        assert_eq!(cfg.num_single_layers, 20);
        assert_eq!(cfg.joint_attention_dim, 7680);
        assert!(!cfg.guidance_embeds);
    }
}
