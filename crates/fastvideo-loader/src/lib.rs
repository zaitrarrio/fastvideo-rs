//! Hugging Face Diffusers weight loading for Wan components.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use thiserror::Error;

use fastvideo_models::wan::PARAM_NAMES_MAPPING;

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Apply FastVideo `param_names_mapping` rules. Regex rewrite lands with
/// FastVideo-native checkpoints; Diffusers keys are used as-is for Phase 1.
pub fn map_param_name(source: &str) -> String {
    let _ = PARAM_NAMES_MAPPING;
    source.to_string()
}

pub fn collect_safetensors(dir: &Path) -> Result<Vec<PathBuf>, LoaderError> {
    if !dir.is_dir() {
        return Err(LoaderError::Message(format!(
            "weight directory not found: {}",
            dir.display()
        )));
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(collect_safetensors(&path)?);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "safetensors")
        {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

pub fn var_builder_from_dir(
    dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>, LoaderError> {
    let files = collect_safetensors(dir)?;
    if files.is_empty() {
        return Err(LoaderError::Message(format!(
            "no .safetensors files under {}",
            dir.display()
        )));
    }
    let mut tensors = std::collections::HashMap::new();
    for file in &files {
        // Decode on CPU, cast, then move. Loading f32 shards straight onto a
        // 24GB GPU OOMs Wan 1.3B (UMT5+DiT+VAE are ~27GB on disk).
        let loaded = candle_core::safetensors::load(file, &Device::Cpu)?;
        for (name, tensor) in loaded {
            tensors.insert(name, tensor.to_dtype(dtype)?.to_device(device)?);
        }
    }
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

pub struct DiffusersComponents {
    pub transformer: VarBuilder<'static>,
    /// Wan 2.2 MoE low-noise expert (`transformer_2/`).
    pub transformer_2: Option<VarBuilder<'static>>,
    pub vae: VarBuilder<'static>,
    pub text: VarBuilder<'static>,
    /// Wan I2V CLIP ViT-H (`image_encoder/`).
    pub image_encoder: Option<VarBuilder<'static>>,
}

pub fn load_diffusers_components(
    root: &Path,
    dtype: DType,
    device: &Device,
) -> Result<DiffusersComponents, LoaderError> {
    let transformer = var_builder_from_dir(&root.join("transformer"), dtype, device)?;
    let transformer_2 = {
        let dir = root.join("transformer_2");
        if dir.is_dir() {
            match collect_safetensors(&dir) {
                Ok(files) if !files.is_empty() => {
                    Some(var_builder_from_dir(&dir, dtype, device)?)
                }
                _ => None,
            }
        } else {
            None
        }
    };
    let vae = var_builder_from_dir(&root.join("vae"), dtype, device)?;
    let text = {
        let te = root.join("text_encoder");
        if te.is_dir() {
            var_builder_from_dir(&te, dtype, device)?
        } else {
            var_builder_from_dir(&root.join("text_encoder_2"), dtype, device)?
        }
    };
    let image_encoder = {
        let dir = root.join("image_encoder");
        if dir.is_dir() {
            match collect_safetensors(&dir) {
                Ok(files) if !files.is_empty() => {
                    Some(var_builder_from_dir(&dir, dtype, device)?)
                }
                _ => None,
            }
        } else {
            None
        }
    };
    Ok(DiffusersComponents {
        transformer,
        transformer_2,
        vae,
        text,
        image_encoder,
    })
}

pub fn weight_map_keys(index_json: &Path) -> Result<Vec<String>, LoaderError> {
    let raw = std::fs::read_to_string(index_json)?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| LoaderError::Message(format!("weight index json: {e}")))?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| LoaderError::Message("missing weight_map".into()))?;
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    Ok(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_models::wan::config::WanVideoArchConfig;
    use fastvideo_models::wan::weights::local_wan_t2v_1_3b;
    use fastvideo_models::wan::WAN_T2V_1_3B_REQUIRED_KEYS;

    #[test]
    fn mapping_table_is_nonempty() {
        assert!(!PARAM_NAMES_MAPPING.is_empty());
        assert_eq!(map_param_name("a.b"), "a.b");
    }

    #[test]
    fn missing_dir_errors() {
        let err = collect_safetensors(Path::new("/tmp/fastvideo-rs-no-such-dir")).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn local_1_3b_diffusers_layout_without_loading_weights() {
        let Some(root) = local_wan_t2v_1_3b() else {
            eprintln!("skip: Wan-AI/Wan2.1-T2V-1.3B-Diffusers not in HF cache / FASTVIDEO_WEIGHTS");
            return;
        };
        assert!(root.join("transformer").is_dir());
        assert!(root.join("vae").is_dir());
        assert!(root.join("text_encoder").is_dir() || root.join("text_encoder_2").is_dir());
        assert!(root.join("tokenizer/tokenizer.json").is_file());
        let cfg_raw = std::fs::read_to_string(root.join("transformer/config.json")).unwrap();
        let cfg: serde_json::Value = serde_json::from_str(&cfg_raw).unwrap();
        let expected = WanVideoArchConfig::wan_t2v_1_3b();
        assert_eq!(cfg["num_layers"].as_u64().unwrap() as usize, expected.num_layers);
        assert_eq!(
            cfg["num_attention_heads"].as_u64().unwrap() as usize,
            expected.num_attention_heads
        );
        assert_eq!(cfg["ffn_dim"].as_u64().unwrap() as usize, expected.ffn_dim);
        assert_eq!(
            cfg["in_channels"].as_u64().unwrap() as usize,
            expected.in_channels
        );
        let files = collect_safetensors(&root.join("transformer")).unwrap();
        assert!(
            files.len() >= 2,
            "expected sharded safetensors, got {files:?}"
        );
        let index = root.join("transformer/diffusion_pytorch_model.safetensors.index.json");
        let keys = weight_map_keys(&index).unwrap();
        for required in WAN_T2V_1_3B_REQUIRED_KEYS {
            assert!(
                keys.iter().any(|k| k == required),
                "missing Diffusers key {required}"
            );
        }
        assert!(keys.iter().any(|k| k.starts_with("blocks.29.")));
        assert!(!keys.iter().any(|k| k.contains("blocks.30.")));
        assert!(
            !root.join("transformer_2").is_dir(),
            "1.3B T2V is a single DiT; MoE transformer_2 belongs on A14B"
        );
    }
}
