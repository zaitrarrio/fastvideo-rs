//! Fuse a Sol-H3 adapter into MiniMax-H3 base weights before they upload.
//!
//! The product is the host form of `weight.addmm_(B, A)`: `W += multiplier * B @ A`,
//! then any FastVideo `.diff` correction in float32. See
//! [`fastvideo_models::h3::lora`].

use std::collections::HashMap;
use std::path::Path;

use fastvideo_models::h3::lora::{
    add_diff, add_low_rank, lora_multiplier, plan_from_keys, LoraFormat, LoraPlan,
};

use crate::wan::tensor::{Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

struct Pair {
    b: Vec<f32>,
    a: Vec<f32>,
    out: usize,
    inn: usize,
}

/// One adapter, consumed as the transformer loader touches each parameter.
pub struct H3LoraFuse {
    pairs: HashMap<String, Pair>,
    diffs: HashMap<String, Vec<f32>>,
    multiplier: f32,
    diff_scale: f32,
    pub rank: usize,
    pub alpha: u32,
    pub pairs_total: usize,
    pub diffs_total: usize,
    pub path: String,
}

impl H3LoraFuse {
    /// Validate `adapter` against `base` and hold the low-rank factors on the host.
    pub fn open(base: &WeightMap, adapter: &Path, alpha: u32, scale: f32) -> Result<Self> {
        let adapter_map = WeightMap::open_files(&[adapter.to_path_buf()])?;
        let lazy = adapter_map
            .lazy()
            .ok_or_else(|| msg("adapter map has no store"))?;
        let keys: Vec<String> = lazy.keys().map(str::to_string).collect();
        let plan =
            plan_from_keys(&keys, lazy.metadata(), &adapter.display().to_string()).map_err(msg)?;
        Self::from_plan(base, &adapter_map, adapter, plan, alpha, scale)
    }

    fn from_plan(
        base: &WeightMap,
        adapter_map: &WeightMap,
        adapter: &Path,
        plan: LoraPlan,
        alpha: u32,
        scale: f32,
    ) -> Result<Self> {
        let mut pairs = HashMap::new();
        let mut ranks = Vec::new();
        for (module, a_key, b_key) in &plan.pairs {
            let weight_key = format!("{module}.weight");
            let base_shape = base
                .shape(&weight_key)
                .ok_or_else(|| msg(format!("LoRA target is absent from transformer: {module}")))?;
            if base_shape.len() != 2 {
                return Err(msg(format!(
                    "LoRA target {module} is not a matrix, shape {base_shape:?}"
                )));
            }
            let (a_shape, a) = adapter_map.get_f32(a_key)?;
            let (b_shape, b) = adapter_map.get_f32(b_key)?;
            if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[0] != b_shape[1] {
                return Err(msg(format!(
                    "Invalid LoRA pair for {module}: A{a_shape:?}, B{b_shape:?}"
                )));
            }
            if base_shape[0] != b_shape[0] || base_shape[1] != a_shape[1] {
                return Err(msg(format!(
                    "LoRA/base mismatch for {module}: base{base_shape:?}, A{a_shape:?}, B{b_shape:?}"
                )));
            }
            ranks.push(a_shape[0]);
            pairs.insert(
                module.clone(),
                Pair {
                    b,
                    a,
                    out: base_shape[0],
                    inn: base_shape[1],
                },
            );
        }
        if ranks.iter().any(|r| *r != ranks[0]) {
            return Err(msg(format!("Mixed LoRA ranks are unsupported: {ranks:?}")));
        }
        let rank = ranks[0];
        if let Some(meta) = plan.metadata_rank {
            if meta != rank {
                return Err(msg(format!(
                    "FastVideo metadata rank {meta} does not match tensor rank {rank}"
                )));
            }
        }
        let mut diffs = HashMap::new();
        for (param, key) in &plan.diffs {
            let base_shape = base
                .shape(param)
                .ok_or_else(|| msg(format!("Adapter diff target is absent: {param}")))?;
            let (shape, data) = adapter_map.get_f32(key)?;
            if shape != base_shape {
                return Err(msg(format!(
                    "Adapter diff/base mismatch for {param}: base{base_shape:?}, diff{shape:?}"
                )));
            }
            diffs.insert(param.clone(), data);
        }
        let multiplier = lora_multiplier(plan.format, rank, alpha, scale).map_err(msg)?;
        let pairs_total = pairs.len();
        let diffs_total = diffs.len();
        Ok(Self {
            pairs,
            diffs,
            multiplier,
            diff_scale: scale,
            rank,
            alpha: match plan.format {
                LoraFormat::FastvideoV2 => rank as u32,
                LoraFormat::Peft => alpha,
            },
            pairs_total,
            diffs_total,
            path: adapter.display().to_string(),
        })
    }

    pub fn effective_scale(&self) -> f32 {
        self.multiplier
    }

    /// Apply a pair (if `param` is `{module}.weight`) and then a `.diff`.
    pub fn fuse(&mut self, param: &str, data: &mut [f32], shape: &[usize]) -> Result<()> {
        if let Some(module) = param.strip_suffix(".weight") {
            if let Some(pair) = self.pairs.remove(module) {
                if shape != [pair.out, pair.inn] || data.len() != pair.out * pair.inn {
                    return Err(msg(format!(
                        "lora fuse {module}: host {:?} != {}x{}",
                        shape, pair.out, pair.inn
                    )));
                }
                add_low_rank(data, pair.out, pair.inn, &pair.b, &pair.a, self.multiplier)
                    .map_err(msg)?;
            }
        }
        if let Some(delta) = self.diffs.remove(param) {
            add_diff(data, &delta, self.diff_scale).map_err(msg)?;
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        if self.pairs.is_empty() && self.diffs.is_empty() {
            return Ok(());
        }
        let extra = self.pairs.len() + self.diffs.len();
        let mut left: Vec<_> = self.pairs.keys().cloned().collect();
        left.extend(self.diffs.keys().cloned());
        left.sort();
        left.truncate(3);
        Err(msg(format!(
            "adapter targets were not applied: {left:?} (and {} more)",
            extra - left.len()
        )))
    }
}
