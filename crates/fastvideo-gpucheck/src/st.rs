//! Minimal F32 safetensors read/write for goldens and prompt embeddings.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

#[derive(Debug, Clone)]
pub struct F32Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl F32Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            bail!("shape {shape:?} needs {n} values, got {}", data.len());
        }
        Ok(Self { shape, data })
    }
}

pub fn save(path: &Path, tensors: &[(&str, &F32Tensor)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, t)| t.data.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let mut views = Vec::with_capacity(tensors.len());
    for ((name, t), b) in tensors.iter().zip(&bytes) {
        views.push((
            name.to_string(),
            TensorView::new(Dtype::F32, t.shape.clone(), b)
                .with_context(|| format!("tensor view {name}"))?,
        ));
    }
    safetensors::serialize_to_file(views, None, path)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn load(path: &Path) -> Result<HashMap<String, F32Tensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", path.display()))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            bail!(
                "{}: tensor {name} is {:?}, expected F32",
                path.display(),
                view.dtype()
            );
        }
        let data = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(
            name.to_string(),
            F32Tensor::new(view.shape().to_vec(), data)?,
        );
    }
    Ok(out)
}

pub fn take(map: &mut HashMap<String, F32Tensor>, name: &str, path: &Path) -> Result<F32Tensor> {
    map.remove(name)
        .with_context(|| format!("{} has no tensor `{name}`", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join("fv-gpucheck-st-roundtrip");
        let path = dir.join("t.safetensors");
        let t = F32Tensor::new(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, -4.0, 5.5, f32::MIN_POSITIVE],
        )
        .unwrap();
        save(&path, &[("x", &t)]).unwrap();
        let mut back = load(&path).unwrap();
        let x = take(&mut back, "x", &path).unwrap();
        assert_eq!(x.shape, t.shape);
        assert_eq!(x.data, t.data);
    }
}
