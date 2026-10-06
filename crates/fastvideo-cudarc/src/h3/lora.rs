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

/// An adapter file as a weight map plus its safetensors metadata. A `.pt`
/// (`torch.save`) is read whole, its LoRA keys normalized to
/// `<module>.lora_{A,B}.weight`; a safetensors file is mapped.
fn open_adapter_map(path: &Path) -> Result<(WeightMap, HashMap<String, String>)> {
    let is_pt = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "pt" || e == "pth");
    if !is_pt {
        let map = WeightMap::open_files(&[path.to_path_buf()])?;
        let metadata = map.lazy().map(|l| l.metadata().clone()).unwrap_or_default();
        return Ok((map, metadata));
    }
    let tensors = fastvideo_loader::pth::read_pth_nested(
        path,
        &["student_lora", "generator_lora", "state_dict"],
        |_| true,
    )
    .map_err(|e| msg(e.to_string()))?;
    let mut raw = HashMap::new();
    for (key, t) in tensors {
        let name = match fastvideo_models::plug::lora_key(&key) {
            Some((module, side)) => format!(
                "{module}.lora_{}.weight",
                if side == fastvideo_models::plug::LoraSide::A {
                    "A"
                } else {
                    "B"
                }
            ),
            None => key,
        };
        if raw
            .insert(
                name.clone(),
                fastvideo_loader::RawTensor::from_f32(t.shape, t.data),
            )
            .is_some()
        {
            return Err(msg(format!(
                "{}: two tensors normalize to {name}",
                path.display()
            )));
        }
    }
    Ok((WeightMap::from_raw_tensors(raw), HashMap::new()))
}

/// Every tensor name of an adapter map, sorted.
fn adapter_keys(map: &WeightMap) -> Vec<String> {
    let mut keys: Vec<String> = match map.lazy() {
        Some(l) => l.keys().map(str::to_string).collect(),
        None => map.raw_keys(),
    };
    keys.sort();
    keys
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
    /// Hybrid `.set_weight` overwrite / inject, consumed by [`Self::fuse`].
    replacements: HashMap<String, (Vec<usize>, Vec<f32>)>,
    format: LoraFormat,
    multiplier: f32,
    diff_scale: f32,
    pub rank: usize,
    pub alpha: u32,
    pub pairs_total: usize,
    pub diffs_total: usize,
    pub replacements_total: usize,
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

    /// A LongLive-Plug H3 adapter ([`fastvideo_models::plug`]): `explicit`,
    /// else the recipe's file beside `root` (`../longlive-plug/<dir>/`). The
    /// `.pt` release (`generator_lora.pt`, tensors under `student_lora`) is
    /// read without Python; the safetensors one is mapped. Alpha comes from
    /// the directory's `adapter_config.json` (else the file's metadata, else
    /// `alpha = rank`). Every LoRA module must match a base parameter: the
    /// planner's matched / skipped / unknown counts are logged and any skip
    /// is an error.
    pub fn open_plug(
        base: &WeightMap,
        root: &Path,
        recipe: &fastvideo_models::plug::PlugRecipe,
        explicit: Option<&Path>,
    ) -> Result<Self> {
        use fastvideo_models::plug::{plan_adapter, AdapterConfig, PLUG_ROOT_ENV};
        let [spec] = recipe.adapters.as_slice() else {
            return Err(msg(format!(
                "{}: the H3 Plug adapters are used one at a time, got {}",
                recipe.name,
                recipe.adapters.len()
            )));
        };
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => {
                let env = std::env::var_os(PLUG_ROOT_ENV).map(std::path::PathBuf::from);
                spec.resolve(root, env.as_deref()).map_err(msg)?
            }
        };
        let (adapter_map, metadata) = open_adapter_map(&path)?;
        let config_json = path
            .parent()
            .map(|d| d.join("adapter_config.json"))
            .filter(|p| p.is_file())
            .map(|p| std::fs::read_to_string(&p).map_err(|e| msg(format!("{}: {e}", p.display()))))
            .transpose()?;
        let config = AdapterConfig::from_sources(config_json.as_deref(), &metadata).map_err(msg)?;
        let keys: Vec<(String, Vec<usize>)> = adapter_keys(&adapter_map)
            .into_iter()
            .map(|k| {
                let shape = adapter_map.shape(&k).unwrap_or_default();
                (k, shape)
            })
            .collect();
        let report = plan_adapter(
            &format!("{} {}/{}", recipe.name, spec.dir, spec.file),
            &keys,
            config,
            spec.weight,
            &|m| Some(format!("{m}.weight")),
            &|p| base.shape(p),
        )
        .map_err(msg)?;
        crate::wan::log::info(format_args!("plug lora {}", report.summary()));
        report.require_complete().map_err(msg)?;
        let names: Vec<String> = keys.into_iter().map(|(k, _)| k).collect();
        let plan =
            plan_from_keys(&names, &HashMap::new(), &path.display().to_string()).map_err(msg)?;
        let rank = report.ranks().first().copied().unwrap_or(1);
        let alpha = config.alpha.unwrap_or(rank as f64);
        if alpha.fract() != 0.0 || alpha < 1.0 {
            return Err(msg(format!(
                "{}: alpha {alpha} is not a positive integer",
                path.display()
            )));
        }
        Self::from_plan(base, &adapter_map, &path, plan, alpha as u32, spec.weight)
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
        let mut replacements = HashMap::new();
        for (param, key) in &plan.replacements {
            let (shape, data) = adapter_map.get_f32(key)?;
            if let Some(base_shape) = base.shape(param) {
                if shape != base_shape {
                    return Err(msg(format!(
                        "Adapter set_weight/base mismatch for {param}: base{base_shape:?}, set{shape:?}"
                    )));
                }
            }
            replacements.insert(param.clone(), (shape, data));
        }
        let rank = ranks.first().copied().unwrap_or(1);
        if !ranks.is_empty() && ranks.iter().any(|r| *r != rank) {
            return Err(msg(format!("Mixed LoRA ranks are unsupported: {ranks:?}")));
        }
        if let Some(meta) = plan.metadata_rank {
            if !ranks.is_empty() && meta != rank {
                return Err(msg(format!(
                    "FastVideo metadata rank {meta} does not match tensor rank {rank}"
                )));
            }
        }
        let multiplier = lora_multiplier(plan.format, rank, alpha, scale).map_err(msg)?;
        let pairs_total = pairs.len();
        let diffs_total = diffs.len();
        let replacements_total = replacements.len();
        Ok(Self {
            factors: pairs.clone(),
            pairs,
            diffs,
            replacements,
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
            replacements_total,
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

    /// Host copy of a `.set_weight` row when the base checkpoint has no key.
    pub fn replacement(&self, param: &str) -> Option<(Vec<usize>, Vec<f32>)> {
        self.replacements
            .get(param)
            .map(|(shape, data)| (shape.clone(), data.clone()))
    }

    /// Parameter names this adapter supplies whole (`.set_weight`), sorted.
    pub fn replacement_params(&self) -> Vec<String> {
        let mut names: Vec<String> = self.replacements.keys().cloned().collect();
        names.sort();
        names
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

    /// Whether [`Self::fuse`] would change `param` (a pair, a `.diff` or a
    /// replacement targets it).
    pub fn touches(&self, param: &str) -> bool {
        self.replacements.contains_key(param)
            || self.diffs.contains_key(param)
            || param
                .strip_suffix(".weight")
                .is_some_and(|m| self.pairs.contains_key(m))
    }

    /// Apply a pair (if `param` is `{module}.weight`) and then a `.diff`.
    pub fn fuse(&mut self, param: &str, data: &mut [f32], shape: &[usize]) -> Result<()> {
        self.take_parts(param, shape)?.apply_host(param, data, shape)
    }

    /// Remove and return what [`Self::fuse`] would apply to `param` (as
    /// [`Self::fuse`] does, so [`Self::finish`] sees it consumed), checked
    /// against `shape`. [`FuseParts::apply_host`] is the host merge; the
    /// device merge (`fastvideo_cudarc::h3::transformer`) runs the same
    /// arithmetic in a kernel.
    pub fn take_parts(&mut self, param: &str, shape: &[usize]) -> Result<FuseParts> {
        let numel: usize = shape.iter().product();
        let replacement = match self.replacements.remove(param) {
            Some((rshape, rdata)) => {
                if rshape != shape || rdata.len() != numel {
                    return Err(msg(format!(
                        "set_weight fuse {param}: host {shape:?} != {rshape:?}"
                    )));
                }
                Some(rdata)
            }
            None => None,
        };
        let pair = match param.strip_suffix(".weight") {
            Some(module) => match self.pairs.remove(module) {
                Some(pair) => {
                    if shape != [pair.out, pair.inn] || numel != pair.out * pair.inn {
                        return Err(msg(format!(
                            "lora fuse {module}: host {:?} != {}x{}",
                            shape, pair.out, pair.inn
                        )));
                    }
                    Some(pair)
                }
                None => None,
            },
            None => None,
        };
        let diff = self.diffs.remove(param);
        Ok(FuseParts {
            replacement,
            pair,
            diff,
            multiplier: self.multiplier,
            diff_scale: self.diff_scale,
        })
    }

    pub fn finish(&self) -> Result<()> {
        if self.pairs.is_empty() && self.diffs.is_empty() && self.replacements.is_empty() {
            return Ok(());
        }
        let extra = self.pairs.len() + self.diffs.len() + self.replacements.len();
        let mut left: Vec<_> = self.pairs.keys().cloned().collect();
        left.extend(self.diffs.keys().cloned());
        left.extend(self.replacements.keys().cloned());
        left.sort();
        left.truncate(3);
        Err(msg(format!(
            "adapter targets were not applied: {left:?} (and {} more)",
            extra - left.len()
        )))
    }
}

/// What one parameter of an adapter merge applies: a `.set_weight`
/// replacement, then a low-rank pair, then a `.diff`, in that order.
#[derive(Clone)]
pub struct FuseParts {
    pub replacement: Option<Vec<f32>>,
    pair: Option<Pair>,
    pub diff: Option<Vec<f32>>,
    pub multiplier: f32,
    pub diff_scale: f32,
}

impl FuseParts {
    /// Nothing to apply.
    pub fn is_empty(&self) -> bool {
        self.replacement.is_none() && self.pair.is_none() && self.diff.is_none()
    }

    /// `(A [rank, in], B [out, rank], rank)` of the pair, if any.
    pub fn pair(&self) -> Option<(&[f32], &[f32], usize)> {
        self.pair
            .as_ref()
            .map(|p| (&p.a[..], &p.b[..], p.a.len() / p.inn.max(1)))
    }

    /// The host merge, in float32: `data = replacement * diff_scale`, then
    /// `data += multiplier * B @ A` ([`add_low_rank`]), then
    /// `data += diff_scale * diff` ([`add_diff`]).
    pub fn apply_host(self, param: &str, data: &mut [f32], shape: &[usize]) -> Result<()> {
        if let Some(rdata) = self.replacement {
            if rdata.len() != data.len() {
                return Err(msg(format!(
                    "set_weight fuse {param}: host {shape:?} holds {} values",
                    rdata.len()
                )));
            }
            // FastVideo `DenseLoRAPatch.replacement_for`: value * strength.
            for (d, r) in data.iter_mut().zip(&rdata) {
                *d = r * self.diff_scale;
            }
        }
        if let Some(pair) = self.pair {
            if data.len() != pair.out * pair.inn {
                return Err(msg(format!(
                    "lora fuse {param}: host {:?} != {}x{}",
                    shape, pair.out, pair.inn
                )));
            }
            add_low_rank(data, pair.out, pair.inn, &pair.b, &pair.a, self.multiplier)
                .map_err(msg)?;
        }
        if let Some(delta) = self.diff {
            add_diff(data, &delta, self.diff_scale).map_err(msg)?;
        }
        Ok(())
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
    use crate::wan::weights::WeightMap;
    use fastvideo_models::h3::lora::plan_from_keys;
    use std::collections::HashMap;
    use std::path::Path;

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
            replacements: HashMap::new(),
            format: LoraFormat::FastvideoV2,
            multiplier: 1.0,
            diff_scale: 1.0,
            rank,
            alpha: rank as u32,
            pairs_total: 0,
            diffs_total: 0,
            replacements_total: 0,
            path: "test".into(),
        };
        set_strength(&mut fuse, 0.8, std::iter::once(&mut lin)).unwrap();
        assert!((fuse.effective_scale() - 0.8).abs() < 1e-6);
        let got = lin.weight.host_cow().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-5, "{g} vs {w}");
        }
    }

    #[test]
    fn fuse_overwrites_from_set_weight() {
        let mut fuse = H3LoraFuse {
            pairs: HashMap::new(),
            factors: HashMap::new(),
            diffs: HashMap::new(),
            replacements: HashMap::from([(
                "blocks.0.attn.to_q.weight".into(),
                (vec![2, 2], vec![9.0, 8.0, 7.0, 6.0]),
            )]),
            format: LoraFormat::FastvideoV2,
            multiplier: 1.0,
            diff_scale: 1.0,
            rank: 1,
            alpha: 1,
            pairs_total: 0,
            diffs_total: 0,
            replacements_total: 1,
            path: "test".into(),
        };
        let mut data = vec![1.0f32, 1.0, 1.0, 1.0];
        fuse.fuse("blocks.0.attn.to_q.weight", &mut data, &[2, 2])
            .unwrap();
        assert_eq!(data, vec![9.0, 8.0, 7.0, 6.0]);
        fuse.finish().unwrap();
    }

    #[test]
    fn fifty_gates_inject_when_base_has_no_key() {
        let keys: Vec<String> = (0..50)
            .map(|i| format!("transformer_blocks.{i}.attn.to_gate_compress.weight.set_weight"))
            .collect();
        let mut meta = HashMap::new();
        meta.insert("format".into(), "fastvideo-lora-v2".into());
        meta.insert("set_weight_tensors".into(), "50".into());
        meta.insert("low_rank_tensors".into(), "0".into());
        let plan = plan_from_keys(&keys, &meta, "vsa-datafree/adapter_model.safetensors").unwrap();
        let adapter = WeightMap::from_f32_tensors(keys.iter().map(|k| {
            (
                k.clone(),
                vec![2usize, 3],
                vec![0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5],
            )
        }));
        let base = WeightMap::from_f32_tensors([]);
        let mut fuse = H3LoraFuse::from_plan(
            &base,
            &adapter,
            Path::new("vsa-datafree/adapter_model.safetensors"),
            plan,
            1,
            1.0,
        )
        .unwrap();
        assert_eq!(fuse.replacements_total, 50);
        assert!(base
            .shape("transformer_blocks.0.attn.to_gate_compress.weight")
            .is_none());
        for i in 0..50 {
            let key = format!("transformer_blocks.{i}.attn.to_gate_compress.weight");
            let (shape, row) = fuse.replacement(&key).expect("injected gate");
            assert_eq!(shape, vec![2, 3]);
            let mut data = vec![0.0f32; 6];
            fuse.fuse(&key, &mut data, &shape).unwrap();
            assert_eq!(data, row);
        }
        fuse.finish().unwrap();
    }
}
