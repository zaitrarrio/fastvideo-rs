//! Diffusers safetensors → `CudaTensor` weights.
//!
//! Loads **native** on-disk dtypes (BF16/F16/F32) via
//! [`fastvideo_loader::load_raw_tensors_native`], then materializes f32 host
//! views for the eager graph. BF16 bytes remain available via
//! [`WeightMap::get_bf16_bytes`] for GPU upload when the `cuda` feature is active.

use std::collections::HashMap;
use std::path::Path;

use fastvideo_loader::{load_raw_tensors_native, LazyStore, RawTensor};

use super::tensor::{CudaTensor, Result, TensorError};

/// Produces F32 values for a missing key given the shape the loader expects.
pub type WeightGenerator = dyn Fn(&str, &[usize]) -> Vec<f32> + Send + Sync;

pub struct WeightMap {
    tensors: HashMap<String, RawTensor>,
    /// Set by [`WeightMap::open`]: tensors stay in the mapped shards and are
    /// converted one at a time as they are asked for, so host memory holds a
    /// single tensor rather than the checkpoint. What the 20B+ models load
    /// through; `load_dir` remains the eager path the Wan graph was validated on.
    lazy: Option<LazyStore>,
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
            lazy: None,
            generator: None,
        })
    }

    /// Map every shard under `dir` without reading tensor data.
    pub fn open(dir: &Path) -> Result<Self> {
        let lazy = LazyStore::open(dir).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors: HashMap::new(),
            lazy: Some(lazy),
            generator: None,
        })
    }

    /// Exactly these files: a single-file checkpoint or a chosen subset of shards.
    pub fn open_files(files: &[std::path::PathBuf]) -> Result<Self> {
        let lazy = LazyStore::open_files(files).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors: HashMap::new(),
            lazy: Some(lazy),
            generator: None,
        })
    }

    /// The lazy store, when this map was opened rather than loaded.
    pub fn lazy(&self) -> Option<&LazyStore> {
        self.lazy.as_ref()
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
            lazy: None,
            generator: Some(Box::new(generator)),
        }
    }

    pub fn require(&self, key: &str) -> Result<&RawTensor> {
        self.tensors
            .get(key)
            .ok_or_else(|| TensorError::Message(format!("missing weight key: {key}")))
    }

    pub fn get_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        if let Some(lazy) = &self.lazy {
            return lazy.to_f32(key).map_err(|e| TensorError::Message(e.to_string()));
        }
        let t = self.require(key)?;
        let values = t
            .to_f32_vec()
            .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok((t.shape.clone(), values))
    }

    /// Native BF16 payload for CUDA upload (converts from F32/F16 if needed).
    pub fn get_bf16_bytes(&self, key: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        if let Some(lazy) = &self.lazy {
            let (shape, bytes) = lazy.to_bf16(key).map_err(|e| TensorError::Message(e.to_string()))?;
            return Ok((shape, bytes.into_owned()));
        }
        let t = self.require(key)?;
        let bytes = t
            .to_bf16_bytes()
            .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok((t.shape.clone(), bytes))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.has_tensor(key) || self.generator.is_some()
    }

    /// A real tensor under `key` (as opposed to one a generator would invent).
    pub fn has_tensor(&self, key: &str) -> bool {
        self.tensors.contains_key(key) || self.lazy.as_ref().is_some_and(|l| l.contains(key))
    }

    /// Shape of a real tensor, without touching its data.
    pub fn shape(&self, key: &str) -> Option<Vec<usize>> {
        match &self.lazy {
            Some(l) => l.shape(key).map(<[usize]>::to_vec),
            None => self.tensors.get(key).map(|t| t.shape.clone()),
        }
    }

    /// bf16 values of a lazily mapped tensor, for upload as a device bf16
    /// weight without an f32 detour. `None` for eager and generated maps, whose
    /// callers keep the f32 path they were validated on.
    ///
    /// bf16 on disk is reinterpreted bit for bit. F32/F16 on disk goes through
    /// `half::bf16::from_f32`, the same rounding `Linear::from_tensors` applies,
    /// so a checkpoint loads to identical device bits either way.
    #[cfg(feature = "cuda")]
    pub fn lazy_bf16(&self, key: &str) -> Result<Option<(Vec<usize>, Vec<half::bf16>)>> {
        use fastvideo_loader::LazyDType;
        use rayon::prelude::*;
        let Some(lazy) = &self.lazy else { return Ok(None) };
        let view = lazy.view(key).map_err(|e| TensorError::Message(e.to_string()))?;
        let values: Vec<half::bf16> = if *view.dtype == LazyDType::BF16 {
            view.bytes
                .par_chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        } else {
            let (_, f) = lazy.to_f32(key).map_err(|e| TensorError::Message(e.to_string()))?;
            f.par_iter().map(|&v| half::bf16::from_f32(v)).collect()
        };
        Ok(Some((view.shape.to_vec(), values)))
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
    if let (false, Some(generate)) = (map.has_tensor(key), map.generator.as_ref()) {
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
            lazy: None,
            generator: None,
        };
        let err = map.require("no.such.key").unwrap_err();
        assert!(err.to_string().contains("missing weight key"));
    }

    /// A lazily opened map serves the same f32 values as the eager loader,
    /// shape-checks, and feeds `Linear::load` — what every large model loads through.
    #[test]
    fn lazy_map_feeds_the_layer_loaders() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("fv-lazy-wm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // One shard: a bf16 [2,3] weight and an f32 [2] bias.
        let w: Vec<u8> = [1.0f32, 2.0, 3.0, -1.0, 0.5, 0.25]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let b: Vec<u8> = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"proj.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{}]}},"proj.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{},{}]}}}}"#,
            w.len(),
            w.len(),
            w.len() + b.len()
        );
        let mut f = std::fs::File::create(dir.join("model-00001-of-00001.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&w).unwrap();
        f.write_all(&b).unwrap();
        drop(f);

        let map = WeightMap::open(&dir).unwrap();
        assert!(map.has_tensor("proj.weight") && !map.has_tensor("proj.nope"));
        assert_eq!(map.shape("proj.weight"), Some(vec![2, 3]));
        assert!(cuda_tensor_shaped(&map, "proj.weight", &[3, 2]).is_err(), "shape must be checked");
        let lin = crate::wan::nn::Linear::load(&map, "proj", 3, 2, true).unwrap();
        let x = CudaTensor::from_vec(vec![1.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(&*y.host_cow().unwrap(), &[16.0, 19.75]);
        let _ = std::fs::remove_dir_all(&dir);
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
