//! Backend-agnostic safetensors loading.
//!
//! Default [`load_raw_tensors`] converts everything to F32 for Burn / Luminal.
//! [`load_raw_tensors_native`] preserves on-disk F16/BF16/F32 bytes for CUDA
//! backends that upload native dtypes.
//!
//! Files are memory-mapped (not `std::fs::read`) and each tensor's dtype
//! conversion runs on a `rayon` thread pool: for a multi-GB checkpoint (the
//! 14B Wan variants are tens of GB) `std::fs::read` pays a full buffered copy
//! of the whole file before any parsing can start, and converting BF16/F16→F32
//! one tensor at a time on a single thread is pure wall-clock during process
//! startup. Neither matters for steady-state generate throughput, but both
//! directly hit "time to first frame" / iteration speed while developing.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

use crate::{collect_safetensors, LoaderError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawDType {
    F32,
    F16,
    BF16,
}

#[derive(Debug, Clone)]
pub struct RawTensor {
    pub shape: Vec<usize>,
    pub dtype: RawDType,
    /// Little-endian element bytes matching [`Self::dtype`].
    pub data: Vec<u8>,
    /// Parallel f32 view when `dtype == F32` (avoids unsafe byte casts).
    values: Vec<f32>,
}

impl RawTensor {
    fn from_f32(shape: Vec<usize>, values: Vec<f32>) -> Self {
        let data = f32_bytes_le(&values);
        Self {
            shape,
            dtype: RawDType::F32,
            data,
            values,
        }
    }

    fn from_native(shape: Vec<usize>, dtype: RawDType, data: Vec<u8>) -> Self {
        let values = if dtype == RawDType::F32 {
            let mut vals = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            vals
        } else {
            Vec::new()
        };
        Self {
            shape,
            dtype,
            data,
            values,
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], LoaderError> {
        if self.dtype != RawDType::F32 {
            return Err(LoaderError::Message(format!(
                "as_f32_slice requires F32, got {:?}",
                self.dtype
            )));
        }
        Ok(&self.values)
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, LoaderError> {
        match self.dtype {
            RawDType::F32 => Ok(self.values.clone()),
            RawDType::F16 => {
                if self.data.len() % 2 != 0 {
                    return Err(LoaderError::Message(
                        "F16 tensor byte length is not a multiple of 2".into(),
                    ));
                }
                let mut out = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f16_bits_to_f32(bits));
                }
                Ok(out)
            }
            RawDType::BF16 => {
                if self.data.len() % 2 != 0 {
                    return Err(LoaderError::Message(
                        "BF16 tensor byte length is not a multiple of 2".into(),
                    ));
                }
                let mut out = Vec::with_capacity(self.data.len() / 2);
                for chunk in self.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(bf16_bits_to_f32(bits));
                }
                Ok(out)
            }
        }
    }

    /// Little-endian BF16 element bytes (converts from F32/F16 if needed).
    pub fn to_bf16_bytes(&self) -> Result<Vec<u8>, LoaderError> {
        match self.dtype {
            RawDType::BF16 => Ok(self.data.clone()),
            RawDType::F32 | RawDType::F16 => {
                let f32s = self.to_f32_vec()?;
                let mut out = Vec::with_capacity(f32s.len() * 2);
                for v in f32s {
                    let bits = f32_to_bf16_bits(v);
                    out.extend_from_slice(&bits.to_le_bytes());
                }
                Ok(out)
            }
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub(crate) fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7fff + lsb;
    ((bits + rounding) >> 16) as u16
}

pub(crate) fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from((h >> 15) & 1);
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = -14i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub(crate) fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

fn f32_bytes_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn view_to_f32(view: &TensorView<'_>) -> Result<(Vec<usize>, Vec<f32>), LoaderError> {
    let shape = view.shape().to_vec();
    let raw = view.data();
    let f32s = match view.dtype() {
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                return Err(LoaderError::Message("invalid F32 safetensors payload".into()));
            }
            let mut vals = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            vals
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                return Err(LoaderError::Message("invalid F16 safetensors payload".into()));
            }
            let mut vals = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                vals.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            vals
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                return Err(LoaderError::Message("invalid BF16 safetensors payload".into()));
            }
            let mut vals = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                vals.push(bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
            vals
        }
        other => {
            return Err(LoaderError::Message(format!(
                "unsupported safetensors dtype {other:?} (expected F32/F16/BF16)"
            )));
        }
    };
    Ok((shape, f32s))
}

fn view_to_native(view: &TensorView<'_>) -> Result<(Vec<usize>, RawDType, Vec<u8>), LoaderError> {
    let shape = view.shape().to_vec();
    let raw = view.data().to_vec();
    let dtype = match view.dtype() {
        Dtype::F32 => RawDType::F32,
        Dtype::F16 => RawDType::F16,
        Dtype::BF16 => RawDType::BF16,
        other => {
            return Err(LoaderError::Message(format!(
                "unsupported safetensors dtype {other:?} (expected F32/F16/BF16)"
            )));
        }
    };
    Ok((shape, dtype, raw))
}

/// Load every `.safetensors` under `dir` (recursive). Values are converted to F32.
pub fn load_raw_tensors(dir: &Path) -> Result<HashMap<String, RawTensor>, LoaderError> {
    load_raw_tensors_with(dir, |view| {
        let (shape, values) = view_to_f32(view)?;
        Ok(RawTensor::from_f32(shape, values))
    })
}

/// Load safetensors preserving on-disk F16/BF16/F32 element bytes (for CUDA upload).
pub fn load_raw_tensors_native(dir: &Path) -> Result<HashMap<String, RawTensor>, LoaderError> {
    load_raw_tensors_with(dir, |view| {
        let (shape, dtype, data) = view_to_native(view)?;
        Ok(RawTensor::from_native(shape, dtype, data))
    })
}

/// mmap every `.safetensors` file under `dir`, then convert every tensor
/// across every file in parallel via `convert` (`view_to_f32`/`view_to_native`
/// wrapped by the two public loaders above). `convert` must be `Sync`: it
/// runs concurrently across a `rayon` thread pool, one call per tensor, with
/// no shared mutable state — each call reads its own `TensorView` (borrowed
/// from one of the mmaps, kept alive for the whole function) and returns an
/// owned `RawTensor`, so there's nothing to synchronize.
fn load_raw_tensors_with(
    dir: &Path,
    convert: impl Fn(&TensorView<'_>) -> Result<RawTensor, LoaderError> + Sync,
) -> Result<HashMap<String, RawTensor>, LoaderError> {
    let files = collect_safetensors(dir)?;
    if files.is_empty() {
        return Err(LoaderError::Message(format!(
            "no .safetensors files under {}",
            dir.display()
        )));
    }
    // Keep every mmap alive for the whole function: `SafeTensors::deserialize`
    // borrows from it, and every `TensorView` below borrows from that.
    let mmaps: Vec<(PathBuf, Mmap)> = files
        .into_iter()
        .map(|f| {
            let mapped = mmap_file(&f)?;
            Ok((f, mapped))
        })
        .collect::<Result<_, LoaderError>>()?;
    let parsed: Vec<(&PathBuf, SafeTensors<'_>)> = mmaps
        .iter()
        .map(|(f, m)| {
            let st = SafeTensors::deserialize(m)
                .map_err(|e| LoaderError::Message(format!("{}: {e}", f.display())))?;
            Ok((f, st))
        })
        .collect::<Result<_, LoaderError>>()?;

    // Flatten to one (name, view) list across every file, then convert in
    // parallel — this is what actually gets threaded across cores; mmap'ing
    // and parsing the safetensors header above is comparatively cheap
    // (header-only; tensor bytes are only touched on first access, whether
    // that's here or in the sequential fallback).
    let entries: Vec<(String, TensorView<'_>)> = parsed
        .iter()
        .flat_map(|(_, st)| st.tensors())
        .collect();

    entries
        .into_par_iter()
        .map(|(name, view)| convert(&view).map(|t| (name, t)))
        .collect::<Result<HashMap<_, _>, LoaderError>>()
}

/// # Safety (why this is sound in practice, not just permitted by the lint)
/// `Mmap::map` is `unsafe` because the OS gives no guarantee the backing file
/// won't be mutated or truncated after mapping, which could produce a SIGBUS
/// or (for mutation) a torn read on access. In practice this loads
/// `.safetensors` checkpoint files that are:
/// - written once by a model export step and treated as read-only inputs for
///   the rest of their life (never opened for writing by this process or, in
///   the intended deployment, by anything else while a load is in flight);
/// - the same files `std::fs::read` was reading before this change, just
///   without copying them through a buffer first — mmap doesn't add new
///   exposure to concurrent-mutation risk that a plain `read` didn't already
///   have a race-free version of (a `read` racing a concurrent truncate can
///   also observe a short/torn file; mmap's failure mode for that case is a
///   SIGBUS instead of a short read, not a new class of unsoundness).
pub(crate) fn mmap_file(path: &Path) -> Result<Mmap, LoaderError> {
    let file = File::open(path)?;
    // SAFETY: see the doc comment above.
    unsafe { Mmap::map(&file) }
        .map_err(|e| LoaderError::Message(format!("mmap {}: {e}", path.display())))
}

/// Load one Diffusers component subdirectory (`transformer`, `vae`, `text_encoder`, …).
pub fn load_raw_component(
    root: &Path,
    component: &str,
) -> Result<HashMap<String, RawTensor>, LoaderError> {
    let dir = root.join(component);
    load_raw_tensors(&dir)
}

/// Native-dtype load of one Diffusers component subdirectory.
pub fn load_raw_component_native(
    root: &Path,
    component: &str,
) -> Result<HashMap<String, RawTensor>, LoaderError> {
    let dir = root.join(component);
    load_raw_tensors_native(&dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_raw_tensors_missing_dir_errors() {
        let err = load_raw_tensors(Path::new("/tmp/fastvideo-rs-no-such-raw-dir")).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn as_f32_slice_roundtrip() {
        let vals = vec![1.0f32, -2.5, 0.0];
        let t = RawTensor::from_f32(vec![3], vals.clone());
        assert_eq!(t.as_f32_slice().unwrap(), vals.as_slice());
        assert_eq!(t.to_f32_vec().unwrap(), vals);
        assert_eq!(t.data.len(), 12);
    }

    #[test]
    fn bf16_to_f32_vec() {
        let t = RawTensor {
            shape: vec![1],
            dtype: RawDType::BF16,
            data: 0x3f80u16.to_le_bytes().to_vec(),
            values: Vec::new(),
        };
        let v = t.to_f32_vec().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn to_bf16_bytes_from_f32() {
        let t = RawTensor::from_f32(vec![1], vec![1.0]);
        let b = t.to_bf16_bytes().unwrap();
        assert_eq!(u16::from_le_bytes([b[0], b[1]]), 0x3f80);
    }

    #[test]
    fn native_preserves_bf16() {
        let t = RawTensor::from_native(vec![1], RawDType::BF16, 0x3f80u16.to_le_bytes().to_vec());
        assert_eq!(t.dtype, RawDType::BF16);
        assert_eq!(t.to_bf16_bytes().unwrap(), 0x3f80u16.to_le_bytes());
    }

    /// End-to-end through the real mmap + rayon-parallel path: write a
    /// multi-file, multi-tensor, mixed-dtype (F32 + BF16) fixture to disk and
    /// load it back both ways, checking values round-trip. This is the part
    /// that actually changed (`std::fs::read` → `Mmap` + parallel per-tensor
    /// convert); the unit tests above only cover the dtype math in isolation.
    #[test]
    fn load_raw_tensors_mmap_roundtrip_multi_file() {
        let dir = std::env::temp_dir().join(format!(
            "fastvideo-loader-mmap-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // File 1: one F32 tensor.
        let f32_vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.0).collect();
        let f32_bytes = f32_bytes_le(&f32_vals);
        let view_a = TensorView::new(Dtype::F32, vec![2, 4], &f32_bytes).unwrap();
        safetensors::serialize_to_file([("a.weight", &view_a)], None, &dir.join("part-1.safetensors"))
            .unwrap();

        // File 2: one BF16 tensor (1.0, 2.0, -1.5, 0.0 as bf16 bit patterns).
        let bf16_bits: [u16; 4] = [0x3f80, 0x4000, 0xbfc0, 0x0000];
        let bf16_bytes: Vec<u8> = bf16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let view_b = TensorView::new(Dtype::BF16, vec![4], &bf16_bytes).unwrap();
        safetensors::serialize_to_file([("b.weight", &view_b)], None, &dir.join("part-2.safetensors"))
            .unwrap();

        // Native path: dtypes preserved as on disk.
        let native = load_raw_tensors_native(&dir).unwrap();
        assert_eq!(native.len(), 2);
        assert_eq!(native["a.weight"].dtype, RawDType::F32);
        assert_eq!(native["a.weight"].shape, vec![2, 4]);
        assert_eq!(native["a.weight"].to_f32_vec().unwrap(), f32_vals);
        assert_eq!(native["b.weight"].dtype, RawDType::BF16);
        assert_eq!(native["b.weight"].shape, vec![4]);
        let got_bf16 = native["b.weight"].to_f32_vec().unwrap();
        for (a, e) in got_bf16.iter().zip([1.0f32, 2.0, -1.5, 0.0].iter()) {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }

        // F32 path: everything converted, including the BF16 tensor.
        let f32_only = load_raw_tensors(&dir).unwrap();
        assert_eq!(f32_only.len(), 2);
        assert_eq!(f32_only["a.weight"].dtype, RawDType::F32);
        assert_eq!(f32_only["a.weight"].as_f32_slice().unwrap(), f32_vals.as_slice());
        assert_eq!(f32_only["b.weight"].dtype, RawDType::F32);
        for (a, e) in f32_only["b.weight"]
            .as_f32_slice()
            .unwrap()
            .iter()
            .zip([1.0f32, 2.0, -1.5, 0.0].iter())
        {
            assert!((a - e).abs() < 1e-6, "{a} vs {e}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
