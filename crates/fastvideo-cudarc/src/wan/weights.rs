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
    /// Official `mlx_h3_dit.safetensors` uses `blocks.{i}.*` / `refiner.{i}.*`
    /// instead of the diffusers names this crate loads. Lookups try the alias.
    mlx_h3: bool,
}

impl WeightMap {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let tensors =
            load_raw_tensors_native(dir).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors,
            lazy: None,
            generator: None,
            mlx_h3: false,
        })
    }

    /// Map every shard under `dir` without reading tensor data.
    pub fn open(dir: &Path) -> Result<Self> {
        let lazy = LazyStore::open(dir).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors: HashMap::new(),
            lazy: Some(lazy),
            generator: None,
            mlx_h3: false,
        })
    }

    /// Exactly these files: a single-file checkpoint or a chosen subset of shards.
    pub fn open_files(files: &[std::path::PathBuf]) -> Result<Self> {
        let lazy = LazyStore::open_files(files).map_err(|e| TensorError::Message(e.to_string()))?;
        Ok(Self {
            tensors: HashMap::new(),
            lazy: Some(lazy),
            generator: None,
            mlx_h3: false,
        })
    }

    /// Treat this map as a FastVideo `mlx_h3_dit.safetensors` (flattened
    /// `blocks.` / `refiner.` keys). Diffusers names still resolve.
    pub fn with_mlx_h3_aliases(mut self) -> Self {
        self.mlx_h3 = true;
        self
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
            mlx_h3: false,
        }
    }

    fn has_direct(&self, key: &str) -> bool {
        self.tensors.contains_key(key) || self.lazy.as_ref().is_some_and(|l| l.contains(key))
    }

    fn mlx_h3_alias(key: &str) -> Option<String> {
        if let Some(rest) = key.strip_prefix("transformer_blocks.") {
            return Some(format!("blocks.{rest}"));
        }
        if let Some(rest) = key.strip_prefix("token_refiner.refiner_blocks.") {
            return Some(format!("refiner.{rest}"));
        }
        None
    }

    fn resolved(&self, key: &str) -> String {
        if !self.mlx_h3 || self.has_direct(key) {
            return key.to_string();
        }
        if let Some(alias) = Self::mlx_h3_alias(key) {
            if self.has_direct(&alias) {
                return alias;
            }
        }
        key.to_string()
    }

    pub fn require(&self, key: &str) -> Result<&RawTensor> {
        let key = self.resolved(key);
        self.tensors
            .get(&key)
            .ok_or_else(|| TensorError::Message(format!("missing weight key: {key}")))
    }

    pub fn get_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let key = self.resolved(key);
        if let Some(lazy) = &self.lazy {
            return lazy.to_f32(&key).map_err(|e| TensorError::Message(e.to_string()));
        }
        let t = self.require(&key)?;
        let values = t
            .to_f32_vec()
            .map_err(|e| TensorError::Message(e.to_string()))?;
        Ok((t.shape.clone(), values))
    }

    /// Native BF16 payload for CUDA upload (converts from F32/F16 if needed).
    pub fn get_bf16_bytes(&self, key: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let key = self.resolved(key);
        if let Some(lazy) = &self.lazy {
            let (shape, bytes) = lazy.to_bf16(&key).map_err(|e| TensorError::Message(e.to_string()))?;
            return Ok((shape, bytes.into_owned()));
        }
        let t = self.require(&key)?;
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
        let key = self.resolved(key);
        self.has_direct(&key)
    }

    /// Shape of a real tensor, without touching its data.
    pub fn shape(&self, key: &str) -> Option<Vec<usize>> {
        let key = self.resolved(key);
        match &self.lazy {
            Some(l) => l.shape(&key).map(<[usize]>::to_vec),
            None => self.tensors.get(&key).map(|t| t.shape.clone()),
        }
    }

    /// On-disk bytes of `key` (U8/U32 packed codes, or any other dtype).
    pub fn get_raw(&self, key: &str) -> Result<(Vec<usize>, Vec<u8>)> {
        let key = self.resolved(key);
        if let Some(lazy) = &self.lazy {
            let view = lazy.view(&key).map_err(|e| TensorError::Message(e.to_string()))?;
            return Ok((view.shape.to_vec(), view.bytes.to_vec()));
        }
        let t = self.require(&key)?;
        Ok((t.shape.clone(), t.data.clone()))
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
        let Some(lazy) = &self.lazy else { return Ok(None) };
        let key = self.resolved(key);
        let view = lazy.view(&key).map_err(|e| TensorError::Message(e.to_string()))?;
        let mut values = vec![half::bf16::ZERO; view.numel()];
        fill_bf16(lazy, &key, &mut values)?;
        Ok(Some((view.shape.to_vec(), values)))
    }
}

/// `key` as bfloat16 into `out`, which must have exactly its element count.
/// A bf16 tensor is copied bit for bit; anything else goes through f32 and
/// `half::bf16::from_f32`, the rounding `Linear::from_tensors` applies. The one
/// conversion behind both [`WeightMap::lazy_bf16`] and the text-encoder
/// prefetcher's pinned staging buffer, so the two produce the same device bits.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn fill_bf16(lazy: &LazyStore, key: &str, out: &mut [half::bf16]) -> Result<()> {
    use fastvideo_loader::LazyDType;
    use rayon::prelude::*;
    let view = lazy.view(key).map_err(|e| TensorError::Message(e.to_string()))?;
    if view.numel() != out.len() {
        return Err(TensorError::Message(format!("key {key}: {} elements into a buffer of {}", view.numel(), out.len())));
    }
    // Chunked so a 130M-element tensor is a few hundred tasks, not one per element.
    const CHUNK: usize = 1 << 18;
    if *view.dtype == LazyDType::BF16 {
        out.par_chunks_mut(CHUNK).zip(view.bytes.par_chunks(2 * CHUNK)).for_each(|(o, b)| {
            for (o, c) in o.iter_mut().zip(b.chunks_exact(2)) {
                *o = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
            }
        });
    } else {
        let (_, f) = lazy.to_f32(key).map_err(|e| TensorError::Message(e.to_string()))?;
        out.par_chunks_mut(CHUNK).zip(f.par_chunks(CHUNK)).for_each(|(o, f)| {
            for (o, &v) in o.iter_mut().zip(f) {
                *o = half::bf16::from_f32(v);
            }
        });
    }
    Ok(())
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

    /// The prefetcher's pinned-buffer fill: stored bf16 is copied bit for bit
    /// (including patterns a float round trip would not preserve), f32 and f16
    /// are rounded with `half::bf16::from_f32`, across the chunk boundary.
    #[test]
    fn fill_bf16_copies_bf16_and_rounds_the_rest() {
        use fastvideo_loader::{LazyDType, SafetensorsWriter, TensorSpec};
        let n = (1usize << 18) + 5;
        let f: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 10f32.powi((i % 9) as i32 - 4)).collect();
        let mut bits: Vec<u16> = (0..n).map(|i| (i as u32).wrapping_mul(40_503) as u16).collect();
        bits[0] = 0x7fc1; // a NaN payload
        bits[1] = 0x8000; // -0
        let h: Vec<half::f16> = f.iter().map(|&v| half::f16::from_f32(v)).collect();

        let dir = std::env::temp_dir().join(format!("fv-fill-bf16-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        let specs = [
            TensorSpec::new("b", LazyDType::BF16, vec![n]),
            TensorSpec::new("f", LazyDType::F32, vec![n]),
            TensorSpec::new("h", LazyDType::F16, vec![n]),
        ];
        let mut w = SafetensorsWriter::create(&path, &specs, &[]).unwrap();
        w.write("b", &bits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        w.write("f", &f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        w.write("h", &h.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        w.finish().unwrap();
        let lazy = LazyStore::open_files(std::slice::from_ref(&path)).unwrap();

        let mut out = vec![half::bf16::ZERO; n];
        fill_bf16(&lazy, "b", &mut out).unwrap();
        assert!(out.iter().zip(&bits).all(|(o, b)| o.to_bits() == *b));
        fill_bf16(&lazy, "f", &mut out).unwrap();
        assert!(out.iter().zip(&f).all(|(o, v)| o.to_bits() == half::bf16::from_f32(*v).to_bits()));
        fill_bf16(&lazy, "h", &mut out).unwrap();
        assert!(out.iter().zip(&h).all(|(o, v)| o.to_bits() == half::bf16::from_f32(v.to_f32()).to_bits()));

        assert!(fill_bf16(&lazy, "b", &mut out[..n - 1]).is_err(), "a short buffer must be refused");
        assert!(fill_bf16(&lazy, "absent", &mut out).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn weight_map_missing_key_errors() {
        let map = WeightMap {
            tensors: HashMap::new(),
            lazy: None,
            generator: None,
            mlx_h3: false,
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
    fn mlx_h3_aliases_blocks_and_serves_raw_codes() {
        use fastvideo_loader::{LazyDType, SafetensorsWriter, TensorSpec};
        let dir = std::env::temp_dir().join(format!("fv-mlx-alias-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mlx_h3_dit.safetensors");
        let codes = vec![1u8, 2, 3, 4];
        let scales: Vec<u8> = [0.5f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut w = SafetensorsWriter::create(
            &path,
            &[
                TensorSpec::new("blocks.0.attn.to_q.weight", LazyDType::U8, vec![2, 2]),
                TensorSpec::new("blocks.0.attn.to_q.weight.scales", LazyDType::F32, vec![1]),
            ],
            &[],
        )
        .unwrap();
        w.write("blocks.0.attn.to_q.weight", &codes).unwrap();
        w.write("blocks.0.attn.to_q.weight.scales", &scales).unwrap();
        w.finish().unwrap();
        let map = WeightMap::open_files(&[path]).unwrap().with_mlx_h3_aliases();
        assert!(map.has_tensor("transformer_blocks.0.attn.to_q.weight"));
        assert!(map.has_tensor("transformer_blocks.0.attn.to_q.weight.scales"));
        let (shape, raw) = map.get_raw("transformer_blocks.0.attn.to_q.weight").unwrap();
        assert_eq!((shape, raw), (vec![2, 2], codes));
        let (_, s) = map.get_f32("transformer_blocks.0.attn.to_q.weight.scales").unwrap();
        assert_eq!(s, vec![0.5]);
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
