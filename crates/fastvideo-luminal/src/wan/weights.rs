//! Diffusers safetensors → host `NdTensor` weights.

use std::collections::HashMap;
use std::path::Path;

use fastvideo_loader::{load_raw_tensors, RawTensor};

use super::tensor::{NdTensor, Result, TensorError};

pub struct WeightMap {
    tensors: HashMap<String, RawTensor>,
}

impl WeightMap {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let tensors = load_raw_tensors(dir).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self { tensors })
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::load_dir(dir)
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

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
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

pub fn nd_tensor(map: &WeightMap, key: &str) -> Result<NdTensor> {
    let (shape, values) = map.get_f32(key)?;
    NdTensor::from_vec(values, shape)
}

pub fn nd_tensor_shaped(map: &WeightMap, key: &str, expected: &[usize]) -> Result<NdTensor> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, expected)?;
    NdTensor::from_vec(values, shape)
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
        };
        let err = map.require("no.such.key").unwrap_err();
        assert!(err.to_string().contains("missing weight key"));
        let err = map.get_f32("also.missing").unwrap_err();
        assert!(err.to_string().contains("missing weight key"));
    }

    #[test]
    fn gated_transformer_first_tensor_shapes() {
        let Ok(root) = std::env::var("FASTVIDEO_WEIGHTS") else {
            eprintln!("skip: set FASTVIDEO_WEIGHTS to exercise Diffusers DiT load");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let map = WeightMap::load_dir(&root.join("transformer")).expect("load transformer");
        let (shape, vals) = map.get_f32("patch_embedding.weight").unwrap();
        assert_eq!(shape.len(), 5);
        assert_eq!(vals.len(), shape.iter().product::<usize>());
    }
}
