//! Lazy, shard-aware safetensors access.
//!
//! [`crate::raw`] copies every tensor of a component into owned host buffers
//! before anything is built. That is fine for a 1.3B DiT and ruinous for a 33B
//! one: the checkpoint alone would not fit in the host RAM of the boxes that
//! have the GPU for it. A [`LazyStore`] maps every shard, reads only the
//! headers, and hands out borrowed views into the mappings, so a tensor's bytes
//! are first touched when they are uploaded, and a caller that streams a model
//! layer by layer never holds more than the pages of the layer in flight.
//!
//! The header is parsed here rather than through the `safetensors` crate so
//! that dtypes the crate's enum does not know (FP8 variants in quantized
//! single-file checkpoints) stay loadable as raw bytes.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;

use crate::{collect_safetensors, LoaderError};

/// On-disk element type. `Other` keeps the safetensors spelling for types the
/// float paths cannot convert (`I64` position buffers, packed FP4, …).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LazyDType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    U8,
    I8,
    Other(String),
}

impl LazyDType {
    fn parse(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F8_E4M3" | "F8_E4M3FN" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "U8" => Self::U8,
            "I8" => Self::I8,
            other => Self::Other(other.to_string()),
        }
    }

    /// Bytes per element, where the type has a whole number of them.
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::F8E4M3 | Self::F8E5M2 | Self::U8 | Self::I8 => Some(1),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
struct Entry {
    file: usize,
    dtype: LazyDType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A borrowed tensor: bytes live in the store's mapping.
#[derive(Debug, Clone)]
pub struct LazyView<'a> {
    pub dtype: &'a LazyDType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl LazyView<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct LazyStore {
    files: Vec<(PathBuf, Mmap)>,
    index: HashMap<String, Entry>,
}

impl std::fmt::Debug for LazyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyStore")
            .field("files", &self.files.len())
            .field("tensors", &self.index.len())
            .finish()
    }
}

fn err(path: &Path, what: impl std::fmt::Display) -> LoaderError {
    LoaderError::Message(format!("{}: {what}", path.display()))
}

impl LazyStore {
    /// Every `.safetensors` under `dir`, recursively. An `*.index.json` is not
    /// needed and not read: the shard headers are the index.
    pub fn open(dir: &Path) -> Result<Self, LoaderError> {
        let files = collect_safetensors(dir)?;
        if files.is_empty() {
            return Err(LoaderError::Message(format!(
                "no .safetensors files under {}",
                dir.display()
            )));
        }
        Self::open_files(&files)
    }

    /// Exactly these files (a single-file checkpoint, or a chosen subset of
    /// shards). A key present in two files is an error rather than a silent
    /// last-one-wins.
    pub fn open_files(paths: &[PathBuf]) -> Result<Self, LoaderError> {
        let mut files = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (fi, path) in paths.iter().enumerate() {
            let map = crate::raw::mmap_file(path)?;
            if map.len() < 8 {
                return Err(err(path, "shorter than a safetensors header"));
            }
            let hlen = u64::from_le_bytes(map[..8].try_into().expect("8 bytes")) as usize;
            let base = 8usize
                .checked_add(hlen)
                .filter(|&b| b <= map.len())
                .ok_or_else(|| err(path, format!("header length {hlen} exceeds the file")))?;
            let header: serde_json::Value =
                serde_json::from_slice(&map[8..base]).map_err(|e| err(path, format!("header: {e}")))?;
            let obj = header
                .as_object()
                .ok_or_else(|| err(path, "header is not a JSON object"))?;
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = info["dtype"]
                    .as_str()
                    .ok_or_else(|| err(path, format!("{name}: no dtype")))?;
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .ok_or_else(|| err(path, format!("{name}: no shape")))?
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize))
                    .collect::<Option<_>>()
                    .ok_or_else(|| err(path, format!("{name}: bad shape")))?;
                let off = info["data_offsets"]
                    .as_array()
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| err(path, format!("{name}: no data_offsets")))?;
                let (s, e) = (
                    off[0].as_u64().ok_or_else(|| err(path, format!("{name}: bad offset")))? as usize,
                    off[1].as_u64().ok_or_else(|| err(path, format!("{name}: bad offset")))? as usize,
                );
                let (start, end) = (base + s, base + e);
                if s > e || end > map.len() {
                    // A truncated download: say so here, not as a fault on first touch.
                    return Err(err(path, format!("{name}: data [{s}, {e}) lies outside the file (truncated?)")));
                }
                let dtype = LazyDType::parse(dtype);
                if let Some(size) = dtype.size() {
                    let want = shape.iter().product::<usize>() * size;
                    if want != e - s {
                        return Err(err(path, format!("{name}: {} bytes for shape {shape:?} of {dtype:?}", e - s)));
                    }
                }
                let entry = Entry { file: fi, dtype, shape, start, end };
                if index.insert(name.clone(), entry).is_some() {
                    return Err(err(path, format!("{name}: also present in an earlier shard")));
                }
            }
            files.push((path.clone(), map));
        }
        Ok(Self { files, index })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.index.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    /// Keys under `prefix`, sorted: the stable order a layer-by-layer loader
    /// and its error messages want.
    pub fn keys_with_prefix(&self, prefix: &str) -> Vec<&str> {
        let mut keys: Vec<&str> = self.keys().filter(|k| k.starts_with(prefix)).collect();
        keys.sort_unstable();
        keys
    }

    /// On-disk bytes of every tensor under `prefix` (`""` for the whole store).
    pub fn bytes_with_prefix(&self, prefix: &str) -> u64 {
        self.index
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(_, e)| (e.end - e.start) as u64)
            .sum()
    }

    pub fn view(&self, key: &str) -> Result<LazyView<'_>, LoaderError> {
        let e = self
            .index
            .get(key)
            .ok_or_else(|| LoaderError::Message(format!("missing weight key: {key}")))?;
        Ok(LazyView {
            dtype: &e.dtype,
            shape: &e.shape,
            bytes: &self.files[e.file].1[e.start..e.end],
        })
    }

    pub fn shape(&self, key: &str) -> Option<&[usize]> {
        self.index.get(key).map(|e| e.shape.as_slice())
    }

    /// F32 values of a float tensor. Large tensors convert across the rayon
    /// pool: a 5376x14336 bf16 weight is 77M elements.
    pub fn to_f32(&self, key: &str) -> Result<(Vec<usize>, Vec<f32>), LoaderError> {
        let v = self.view(key)?;
        let out: Vec<f32> = match v.dtype {
            LazyDType::F32 => v
                .bytes
                .par_chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            LazyDType::BF16 => v
                .bytes
                .par_chunks_exact(2)
                .map(|c| crate::raw::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            LazyDType::F16 => v
                .bytes
                .par_chunks_exact(2)
                .map(|c| crate::raw::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(LoaderError::Message(format!(
                    "{key}: {other:?} has no F32 conversion (read it as raw bytes)"
                )))
            }
        };
        Ok((v.shape.to_vec(), out))
    }

    /// Little-endian bf16 bytes. Borrowed straight from the mapping when the
    /// tensor is stored as bf16 — the common case for every checkpoint these
    /// stores exist for — so the only copy made is the upload itself.
    pub fn to_bf16(&self, key: &str) -> Result<(Vec<usize>, Cow<'_, [u8]>), LoaderError> {
        let v = self.view(key)?;
        let bytes = match v.dtype {
            LazyDType::BF16 => Cow::Borrowed(v.bytes),
            LazyDType::F32 => Cow::Owned(
                v.bytes
                    .par_chunks_exact(4)
                    .flat_map_iter(|c| {
                        crate::raw::f32_to_bf16_bits(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()
                    })
                    .collect(),
            ),
            LazyDType::F16 => Cow::Owned(
                v.bytes
                    .par_chunks_exact(2)
                    .flat_map_iter(|c| {
                        let f = crate::raw::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
                        crate::raw::f32_to_bf16_bits(f).to_le_bytes()
                    })
                    .collect(),
            ),
            other => {
                return Err(LoaderError::Message(format!(
                    "{key}: {other:?} has no bf16 conversion (read it as raw bytes)"
                )))
            }
        };
        Ok((v.shape.to_vec(), bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A minimal safetensors file: `tensors` is `(name, dtype, shape, bytes)`.
    fn write_st(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let h = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&data).unwrap();
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-lazy-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn bf16(v: f32) -> [u8; 2] {
        crate::raw::f32_to_bf16_bits(v).to_le_bytes()
    }

    #[test]
    fn shards_merge_and_views_borrow() {
        let d = tmp("shards");
        let a: Vec<u8> = [1.0f32, -2.0].iter().flat_map(|v| bf16(*v)).collect();
        let b: Vec<u8> = [0.5f32, 4.0, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        write_st(&d.join("model-00001-of-00002.safetensors"), &[("blocks.0.w", "BF16", vec![2], a.clone())]);
        write_st(&d.join("model-00002-of-00002.safetensors"), &[("blocks.1.w", "F32", vec![3], b)]);
        let s = LazyStore::open(&d).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.keys_with_prefix("blocks."), vec!["blocks.0.w", "blocks.1.w"]);
        assert_eq!(s.bytes_with_prefix("blocks.1"), 12);
        assert_eq!(s.to_f32("blocks.0.w").unwrap().1, vec![1.0, -2.0]);
        assert_eq!(s.to_f32("blocks.1.w").unwrap().1, vec![0.5, 4.0, 8.0]);
        // bf16 on disk is handed out without a copy.
        let (_, bytes) = s.to_bf16("blocks.0.w").unwrap();
        assert!(matches!(bytes, Cow::Borrowed(_)));
        assert_eq!(&*bytes, a.as_slice());
        // f32 on disk converts to the same bits the eager loader produces.
        let (_, conv) = s.to_bf16("blocks.1.w").unwrap();
        assert_eq!(&*conv, [bf16(0.5), bf16(4.0), bf16(8.0)].concat().as_slice());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_truncated_shard_is_refused_at_open() {
        let d = tmp("trunc");
        let p = d.join("m.safetensors");
        write_st(&p, &[("w", "F32", vec![4], vec![0u8; 16])]);
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() - 5]).unwrap();
        let e = LazyStore::open(&d).unwrap_err().to_string();
        assert!(e.contains("truncated"), "{e}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn duplicate_keys_and_bad_sizes_are_errors() {
        let d = tmp("dup");
        write_st(&d.join("a.safetensors"), &[("w", "F32", vec![1], vec![0u8; 4])]);
        write_st(&d.join("b.safetensors"), &[("w", "F32", vec![1], vec![0u8; 4])]);
        assert!(LazyStore::open(&d).unwrap_err().to_string().contains("earlier shard"));
        let d2 = tmp("size");
        write_st(&d2.join("a.safetensors"), &[("w", "BF16", vec![3], vec![0u8; 4])]);
        assert!(LazyStore::open(&d2).unwrap_err().to_string().contains("bytes for shape"));
        let _ = std::fs::remove_dir_all(&d);
        let _ = std::fs::remove_dir_all(&d2);
    }

    #[test]
    fn unknown_dtypes_stay_readable_as_bytes() {
        let d = tmp("fp8");
        write_st(
            &d.join("q.safetensors"),
            &[("w", "F8_E4M3", vec![2, 2], vec![1, 2, 3, 4]), ("pos", "I64", vec![1], vec![0u8; 8])],
        );
        let s = LazyStore::open(&d).unwrap();
        let v = s.view("w").unwrap();
        assert_eq!((v.dtype, v.bytes), (&LazyDType::F8E4M3, &[1u8, 2, 3, 4][..]));
        assert!(s.to_f32("w").is_err());
        assert_eq!(s.view("pos").unwrap().dtype, &LazyDType::Other("I64".into()));
        let _ = std::fs::remove_dir_all(&d);
    }
}
