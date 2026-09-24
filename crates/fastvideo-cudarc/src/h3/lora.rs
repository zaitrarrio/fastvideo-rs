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

use crate::wan::nn::Linear;
use crate::wan::tensor::{CudaTensor, Result, TensorError};
use crate::wan::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

#[derive(Clone)]
struct Pair {
    b: Vec<f32>,
    a: Vec<f32>,
    out: usize,
    inn: usize,
}

/// One adapter, consumed as the transformer loader touches each parameter.
pub struct H3LoraFuse {
    pairs: HashMap<String, Pair>,
    /// Copy of [`Self::pairs`] that survives host `fuse` consume, for device re-fuse.
    factors: HashMap<String, Pair>,
    diffs: HashMap<String, Vec<f32>>,
    format: LoraFormat,
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
            factors: pairs.clone(),
            pairs,
            diffs,
            format: plan.format,
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

    /// Recompute the host/device multiplier from a new adapter scale. Does not
    /// touch disk. Call [`Self::set_strength_on`] to write attached linears.
    pub fn set_lora_strength(&mut self, s: f32) -> Result<()> {
        self.multiplier = lora_multiplier(self.format, self.rank, self.alpha, s).map_err(msg)?;
        self.diff_scale = s;
        Ok(())
    }

    /// `A [rank, in]` / `B [out, rank]` for a transformer module, if this
    /// adapter has a pair for it.
    pub fn factors(&self, module: &str) -> Result<Option<(CudaTensor, CudaTensor)>> {
        let Some(pair) = self.factors.get(module) else {
            return Ok(None);
        };
        Ok(Some(pair_tensors(pair)?))
    }

    /// Snapshot `W0` on `linear` and attach this adapter's `(A, B)` for `module`.
    pub fn attach_linear(&self, module: &str, linear: &mut Linear) -> Result<()> {
        let Some((a, b)) = self.factors(module)? else {
            return Err(msg(format!("h3 lora: no factors for {module}")));
        };
        linear.attach_lora(a, b)?;
        linear.set_lora_strength(self.multiplier)
    }

    /// [`Self::set_lora_strength`] then re-fuse each attached linear.
    pub fn set_strength_on<'a>(
        &mut self,
        s: f32,
        linears: impl IntoIterator<Item = &'a mut Linear>,
    ) -> Result<()> {
        self.set_lora_strength(s)?;
        for lin in linears {
            lin.set_lora_strength(self.multiplier)?;
        }
        Ok(())
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

fn pair_tensors(pair: &Pair) -> Result<(CudaTensor, CudaTensor)> {
    let rank = pair.a.len() / pair.inn;
    Ok((
        CudaTensor::from_vec(pair.a.clone(), vec![rank, pair.inn])?,
        CudaTensor::from_vec(pair.b.clone(), vec![pair.out, rank])?,
    ))
}

/// Re-fuse `linears` at scale `s` (WS-B / later callers). Host `fuse` stays
/// the load-time default.
pub fn set_strength<'a>(
    fuse: &mut H3LoraFuse,
    s: f32,
    linears: impl IntoIterator<Item = &'a mut Linear>,
) -> Result<()> {
    fuse.set_strength_on(s, linears)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_lora_strength_updates_multiplier_and_linear() {
        let (out, inn, rank) = (2usize, 3usize, 1usize);
        let w0 = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let a = vec![1.0f32, 0.0, 0.5];
        let b = vec![2.0f32, -1.0];
        let mut want = w0.clone();
        add_low_rank(&mut want, out, inn, &b, &a, 0.8).unwrap();

        let mut lin =
            Linear::from_tensors(CudaTensor::from_vec(w0, vec![out, inn]).unwrap(), None).unwrap();
        lin.attach_lora(
            CudaTensor::from_vec(a, vec![rank, inn]).unwrap(),
            CudaTensor::from_vec(b, vec![out, rank]).unwrap(),
        )
        .unwrap();

        let mut fuse = H3LoraFuse {
            pairs: HashMap::new(),
            factors: HashMap::new(),
            diffs: HashMap::new(),
            format: LoraFormat::FastvideoV2,
            multiplier: 1.0,
            diff_scale: 1.0,
            rank,
            alpha: rank as u32,
            pairs_total: 0,
            diffs_total: 0,
            path: "test".into(),
        };
        set_strength(&mut fuse, 0.8, std::iter::once(&mut lin)).unwrap();
        assert!((fuse.effective_scale() - 0.8).abs() < 1e-6);
        let got = lin.weight.host_cow().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-5, "{g} vs {w}");
        }
    }
}
