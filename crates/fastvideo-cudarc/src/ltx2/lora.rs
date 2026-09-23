//! Installs the distilled LoRA while [`super::transformer::Ltx2Transformer::load`]
//! reads weights. The guard is current-thread only; the DiT loader is single
//! threaded. See [`fastvideo_models::ltx2::lora`].

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fastvideo_models::ltx2::config::Ltx2ModelVersion;
use fastvideo_models::ltx2::lora::{fuse_into, weight_key_aliases, weight_key_for_lora_a};

use crate::wan::tensor::{Result, TensorError};
use crate::wan::weights::WeightMap;

use super::msg;

struct Installed {
    strength: f32,
    map: WeightMap,
    /// Base weight key → `(lora_A key, lora_B key)`.
    pairs: HashMap<String, (String, String)>,
    hits: u32,
}

thread_local! {
    static FUSE: RefCell<Option<Installed>> = const { RefCell::new(None) };
}

/// Clears the fuse table when dropped, including on load failure.
pub struct Guard;

impl Drop for Guard {
    fn drop(&mut self) {
        FUSE.with(|slot| *slot.borrow_mut() = None);
    }
}

pub fn install(path: &Path, strength: f32) -> Result<Guard> {
    let map = if path.is_file() {
        WeightMap::open_files(&[path.to_path_buf()])?
    } else {
        WeightMap::open(path)?
    };
    let keys: Vec<String> = map
        .lazy()
        .ok_or_else(|| msg("ltx2 lora: checkpoint is not a safetensors map"))?
        .keys()
        .map(str::to_string)
        .collect();
    let mut pairs = HashMap::<String, (String, String)>::new();
    for key in &keys {
        let Some(stem) = weight_key_for_lora_a(key) else {
            continue;
        };
        let b_key = format!("{stem}.lora_B.weight");
        if !map.has_tensor(&b_key) {
            return Err(msg(format!("ltx2 lora: {key} has no {b_key}")));
        }
        for alias in weight_key_aliases(stem) {
            pairs.insert(alias, (key.clone(), b_key.clone()));
        }
    }
    if pairs.is_empty() {
        return Err(msg(format!(
            "ltx2 lora: no .lora_A.weight keys in {}",
            path.display()
        )));
    }
    FUSE.with(|slot| {
        *slot.borrow_mut() = Some(Installed {
            strength,
            map,
            pairs,
            hits: 0,
        });
    });
    Ok(Guard)
}

pub fn hits() -> u32 {
    FUSE.with(|slot| slot.borrow().as_ref().map(|g| g.hits).unwrap_or(0))
}

fn wants(key: &str) -> bool {
    FUSE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|g| g.strength != 0.0 && g.pairs.contains_key(key))
    })
}

/// `FASTVIDEO_LTX2_LORA`, or a known filename beside the weights or the DiT.
/// Missing file with the env unset leaves the base DiT unfused.
pub fn resolve(weights: &Path, dit: &Path, version: Ltx2ModelVersion) -> Result<Option<PathBuf>> {
    if let Ok(raw) = std::env::var("FASTVIDEO_LTX2_LORA") {
        let path = PathBuf::from(raw);
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(msg(format!(
            "ltx2 lora: FASTVIDEO_LTX2_LORA {} is not a file",
            path.display()
        )));
    }
    let names = fastvideo_models::ltx2::lora::file_names(version);
    if names.is_empty() {
        return Ok(None);
    }
    let mut roots = vec![weights.to_path_buf()];
    if let Some(parent) = weights.parent() {
        roots.push(parent.to_path_buf());
    }
    if let Some(parent) = dit.parent() {
        roots.push(parent.to_path_buf());
    }
    for root in roots {
        for name in names {
            let direct = root.join(name);
            if direct.is_file() {
                return Ok(Some(direct));
            }
            let nested = root.join("loras").join(name);
            if nested.is_file() {
                return Ok(Some(nested));
            }
        }
    }
    Ok(None)
}

/// Fuse into an f32 weight buffer when `key` is one of the installed pairs.
pub fn apply_f32(key: &str, weight: &mut [f32], shape: &[usize]) -> Result<()> {
    if !wants(key) {
        return Ok(());
    }
    FUSE.with(|slot| {
        let mut guard = slot.borrow_mut();
        let Some(inst) = guard.as_mut() else {
            return Ok(());
        };
        let Some((a_key, b_key)) = inst.pairs.get(key).cloned() else {
            return Ok(());
        };
        let [out, inn] = shape else {
            return Err(msg(format!(
                "ltx2 lora: {key} is rank {}, the fused weight is a matrix",
                shape.len()
            )));
        };
        let (b_shape, b) = inst.map.get_f32(&b_key)?;
        let (a_shape, a) = inst.map.get_f32(&a_key)?;
        if b_shape.len() != 2 || a_shape.len() != 2 || b_shape[0] != *out || a_shape[1] != *inn {
            return Err(msg(format!(
                "ltx2 lora: {key} base {shape:?} A {a_shape:?} B {b_shape:?}"
            )));
        }
        fuse_into(weight, *out, *inn, &b, &a, inst.strength).map_err(TensorError::Message)?;
        inst.hits += 1;
        Ok(())
    })
}

#[cfg(feature = "cuda")]
pub fn apply_bf16(key: &str, weight: &mut [half::bf16], shape: &[usize]) -> Result<()> {
    if !wants(key) {
        return Ok(());
    }
    let mut f32s: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
    apply_f32(key, &mut f32s, shape)?;
    for (dst, src) in weight.iter_mut().zip(f32s) {
        *dst = half::bf16::from_f32(src);
    }
    Ok(())
}
