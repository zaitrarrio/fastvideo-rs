//! Diffusers safetensors → `CudaTensor` weights.
//!
//! Loads **native** on-disk dtypes (BF16/F16/F32) via
//! [`fastvideo_loader::load_raw_tensors_native`], then materializes f32 host
//! views for the eager graph. BF16 bytes remain available via
//! [`WeightMap::get_bf16_bytes`] for GPU upload when the `cuda` feature is active.

use std::collections::HashMap;
use std::path::Path;

use fastvideo_loader::{load_raw_tensors_native, RawTensor};

use super::tensor::{CudaTensor, Result, TensorError};

/// Produces F32 values for a missing key given the shape the loader expects.
pub type WeightGenerator = dyn Fn(&str, &[usize]) -> Vec<f32> + Send + Sync;

pub struct WeightMap {
    tensors: HashMap<String, RawTensor>,
    /// When set, shape-checked loads of absent keys are generated instead of
    /// failing (seeded random weights for GPU-vs-CPU parity tests).
    generator: Option<Box<WeightGenerator>>,
}

impl WeightMap {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let tensors =
            load_raw_tensors_native(dir).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors,
            generator: None,
        })
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::load_dir(dir)
    }

    /// Every weight comes from `generator(key, expected_shape)`: no files, any
    /// config. Loaders always pass the expected shape, so the generated model
    /// has exactly the architecture the config describes.
    pub fn generated(
        generator: impl Fn(&str, &[usize]) -> Vec<f32> + Send + Sync + 'static,
    ) -> Self {
        Self {
            tensors: HashMap::new(),
            generator: Some(Box::new(generator)),
        }
    }

    pub fn require(&self, key: &str) -> Result<&RawTensor> {
        self.tensors
            .get(key)
            .ok_or_else(|| TensorError::Message(format!("missing weight key: {key}")))
    }

    pub fn get_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = self.require(key)?;
        let values = t
            .to_f32_vec()
            .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok((t.shape.clone(), values))
    }

    /// Native BF16 payload for CUDA upload (converts from F32/F16 if needed).
    pub fn get_bf16_bytes(&self, key: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let t = self.require(key)?;
        let bytes = t
            .to_bf16_bytes()
            .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok((t.shape.clone(), bytes))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key) || self.generator.is_some()
    }
}

fn expect_shape(key: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(TensorError::Message(format!(
            "key {key}: shape {actual:?} != expected {expected:?}"
        )));
    }
    Ok(())
}

pub fn cuda_tensor(map: &WeightMap, key: &str) -> Result<CudaTensor> {
    let (shape, values) = map.get_f32(key)?;
    CudaTensor::from_vec(values, shape)
}

pub fn cuda_tensor_shaped(map: &WeightMap, key: &str, expected: &[usize]) -> Result<CudaTensor> {
    if let (None, Some(generate)) = (map.tensors.get(key), map.generator.as_ref()) {
        let values = generate(key, expected);
        return CudaTensor::from_vec(values, expected.to_vec());
    }
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, expected)?;
    CudaTensor::from_vec(values, shape)
}

pub fn join_key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_map_missing_key_errors() {
        let map = WeightMap {
            tensors: HashMap::new(),
            generator: None,
        };
        let err = map.require("no.such.key").unwrap_err();
        assert!(err.to_string().contains("missing weight key"));
    }

    #[test]
    fn gated_transformer_first_tensor_shapes() {
        let Some(root) = std::env::var_os("FASTVIDEO_WEIGHTS") else {
            eprintln!("skip: set FASTVIDEO_WEIGHTS to exercise Diffusers DiT load");
            return;
        };
        let map = WeightMap::from_dir(Path::new(&root).join("transformer").as_path()).unwrap();
        let (shape, _) = map.get_f32("patch_embedding.weight").unwrap();
        assert!(!shape.is_empty());
        let (_shape, bf16) = map.get_bf16_bytes("patch_embedding.weight").unwrap();
        assert_eq!(bf16.len() % 2, 0);
    }
}
