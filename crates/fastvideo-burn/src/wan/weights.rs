//! Diffusers safetensors → Burn ndarray tensors.

use std::collections::HashMap;
use std::path::Path;

use burn::prelude::*;
use fastvideo_loader::{load_raw_tensors, RawTensor};

use super::nn::{B, Device};
use crate::error::{BurnError, Result};

pub struct WeightMap {
    tensors: HashMap<String, RawTensor>,
}

impl WeightMap {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        Ok(Self {
            tensors: load_raw_tensors(dir)?,
        })
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::load_dir(dir)
    }

    pub fn require(&self, key: &str) -> Result<&RawTensor> {
        self.tensors
            .get(key)
            .ok_or_else(|| BurnError::msg(format!("missing weight key: {key}")))
    }

    pub fn get_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = self.require(key)?;
        Ok((t.shape.clone(), t.to_f32_vec()?))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }
}

fn expect_shape(key: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(BurnError::msg(format!(
            "key {key}: shape {actual:?} != expected {expected:?}"
        )));
    }
    Ok(())
}

fn from_f32_shape<const D: usize>(
    values: &[f32],
    shape: [usize; D],
    device: &Device,
) -> Tensor<B, D> {
    Tensor::<B, 1>::from_floats(values, device).reshape(shape)
}

pub fn tensor1(map: &WeightMap, key: &str, device: &Device) -> Result<Tensor<B, 1>> {
    let (shape, values) = map.get_f32(key)?;
    if shape.len() != 1 {
        return Err(BurnError::msg(format!(
            "key {key}: expected rank 1, got {shape:?}"
        )));
    }
    Ok(from_f32_shape(&values, [shape[0]], device))
}

pub fn tensor2(map: &WeightMap, key: &str, device: &Device) -> Result<Tensor<B, 2>> {
    let (shape, values) = map.get_f32(key)?;
    if shape.len() != 2 {
        return Err(BurnError::msg(format!(
            "key {key}: expected rank 2, got {shape:?}"
        )));
    }
    Ok(from_f32_shape(&values, [shape[0], shape[1]], device))
}

pub fn tensor3(map: &WeightMap, key: &str, device: &Device) -> Result<Tensor<B, 3>> {
    let (shape, values) = map.get_f32(key)?;
    if shape.len() != 3 {
        return Err(BurnError::msg(format!(
            "key {key}: expected rank 3, got {shape:?}"
        )));
    }
    Ok(from_f32_shape(
        &values,
        [shape[0], shape[1], shape[2]],
        device,
    ))
}

pub fn tensor4(map: &WeightMap, key: &str, device: &Device) -> Result<Tensor<B, 4>> {
    let (shape, values) = map.get_f32(key)?;
    if shape.len() != 4 {
        return Err(BurnError::msg(format!(
            "key {key}: expected rank 4, got {shape:?}"
        )));
    }
    Ok(from_f32_shape(
        &values,
        [shape[0], shape[1], shape[2], shape[3]],
        device,
    ))
}

pub fn tensor5(map: &WeightMap, key: &str, device: &Device) -> Result<Tensor<B, 5>> {
    let (shape, values) = map.get_f32(key)?;
    if shape.len() != 5 {
        return Err(BurnError::msg(format!(
            "key {key}: expected rank 5, got {shape:?}"
        )));
    }
    Ok(from_f32_shape(
        &values,
        [shape[0], shape[1], shape[2], shape[3], shape[4]],
        device,
    ))
}

pub fn tensor1_shaped(
    map: &WeightMap,
    key: &str,
    expected: usize,
    device: &Device,
) -> Result<Tensor<B, 1>> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, &[expected])?;
    Ok(from_f32_shape(&values, [expected], device))
}

pub fn tensor2_shaped(
    map: &WeightMap,
    key: &str,
    expected: [usize; 2],
    device: &Device,
) -> Result<Tensor<B, 2>> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, &expected)?;
    Ok(from_f32_shape(&values, expected, device))
}

pub fn tensor3_shaped(
    map: &WeightMap,
    key: &str,
    expected: [usize; 3],
    device: &Device,
) -> Result<Tensor<B, 3>> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, &expected)?;
    Ok(from_f32_shape(&values, expected, device))
}

pub fn tensor4_shaped(
    map: &WeightMap,
    key: &str,
    expected: [usize; 4],
    device: &Device,
) -> Result<Tensor<B, 4>> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, &expected)?;
    Ok(from_f32_shape(&values, expected, device))
}

pub fn tensor5_shaped(
    map: &WeightMap,
    key: &str,
    expected: [usize; 5],
    device: &Device,
) -> Result<Tensor<B, 5>> {
    let (shape, values) = map.get_f32(key)?;
    expect_shape(key, &shape, &expected)?;
    Ok(from_f32_shape(&values, expected, device))
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
        assert_eq!(shape.len(), 5, "patch_embedding.weight rank");
        assert_eq!(vals.len(), shape.iter().product::<usize>());
        let (bshape, _) = map.get_f32("patch_embedding.bias").unwrap();
        assert_eq!(bshape.len(), 1);
    }
}
