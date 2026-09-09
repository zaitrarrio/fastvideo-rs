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
        let loaded = candle_core::safetensors::load(file, device)?;
        for (name, tensor) in loaded {
            tensors.insert(name, tensor.to_dtype(dtype)?);
        }
    }
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

pub fn load_diffusers_components(
    root: &Path,
    dtype: DType,
    device: &Device,
) -> Result<(VarBuilder<'static>, VarBuilder<'static>, VarBuilder<'static>), LoaderError> {
    let transformer = var_builder_from_dir(&root.join("transformer"), dtype, device)?;
    let vae = var_builder_from_dir(&root.join("vae"), dtype, device)?;
    let text = {
        let te = root.join("text_encoder");
        if te.is_dir() {
            var_builder_from_dir(&te, dtype, device)?
        } else {
            var_builder_from_dir(&root.join("text_encoder_2"), dtype, device)?
        }
    };
    Ok((transformer, vae, text))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
