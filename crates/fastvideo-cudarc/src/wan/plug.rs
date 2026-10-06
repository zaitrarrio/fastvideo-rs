//! LongLive-Plug adapters on the Wan ports ([`fastvideo_models::plug`]):
//! the Diffusers transformer of the base checkpoint with every recipe
//! adapter merged on the host, `W += weight · alpha / rank · B @ A` per
//! adapter, accumulated in float32 and stored back in the base dtype (what
//! `LongLive-Plug/scripts/merge_lora.py` writes). The adapters use original
//! Wan module names (`blocks.N.self_attn.q`, PEFT `base_model.model.` prefix
//! or the lightx2v export's bare names), renamed to Diffusers with
//! [`super::longlive::diffusers_key`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fastvideo_loader::{LazyStore, RawDType, RawTensor};
use fastvideo_models::plug::{
    merge_pair, plan_adapter, AdapterConfig, AdapterPlan, PlugFamily, PlugRecipe, PLUG_ROOT_ENV,
};
use rayon::prelude::*;

use super::tensor::{Result, TensorError};
use super::weights::WeightMap;

fn msg(s: impl Into<String>) -> TensorError {
    TensorError::Message(s.into())
}

/// An adapter's tensors, read as float32 on demand.
pub trait AdapterTensors {
    /// `(name, shape)` of every tensor.
    fn shapes(&self) -> Vec<(String, Vec<usize>)>;
    fn f32(&self, key: &str) -> Result<Vec<f32>>;
}

impl AdapterTensors for LazyStore {
    fn shapes(&self) -> Vec<(String, Vec<usize>)> {
        self.keys()
            .map(|k| {
                (
                    k.to_string(),
                    self.shape(k).map(<[usize]>::to_vec).unwrap_or_default(),
                )
            })
            .collect()
    }
    fn f32(&self, key: &str) -> Result<Vec<f32>> {
        Ok(self.to_f32(key).map_err(|e| msg(e.to_string()))?.1)
    }
}

/// In-memory adapter (`.pt` files, tests).
impl AdapterTensors for HashMap<String, (Vec<usize>, Vec<f32>)> {
    fn shapes(&self) -> Vec<(String, Vec<usize>)> {
        let mut v: Vec<_> = self
            .iter()
            .map(|(k, (s, _))| (k.clone(), s.clone()))
            .collect();
        v.sort();
        v
    }
    fn f32(&self, key: &str) -> Result<Vec<f32>> {
        self.get(key)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| msg(format!("adapter: no tensor {key}")))
    }
}

/// The Diffusers parameter an original-Wan LoRA module updates.
pub fn wan_target(module: &str) -> Option<String> {
    super::longlive::diffusers_key(&format!("{module}.weight"))
        .ok()
        .flatten()
}

/// Plan one adapter against the base tensors (names and shapes only).
pub fn plan_wan_adapter(
    label: &str,
    adapter: &dyn AdapterTensors,
    config: AdapterConfig,
    weight: f32,
    base: &HashMap<String, RawTensor>,
) -> Result<AdapterPlan> {
    plan_adapter(
        label,
        &adapter.shapes(),
        config,
        weight,
        &wan_target,
        &|p| base.get(p).map(|t| t.shape.clone()),
    )
    .map_err(msg)
}

fn store_like(dtype: RawDType, shape: Vec<usize>, w: Vec<f32>) -> RawTensor {
    match dtype {
        RawDType::F32 => RawTensor::from_f32(shape, w),
        RawDType::BF16 => RawTensor::from_le_bytes(
            shape,
            RawDType::BF16,
            w.iter()
                .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
                .collect(),
        ),
        RawDType::F16 => RawTensor::from_le_bytes(
            shape,
            RawDType::F16,
            w.iter()
                .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
                .collect(),
        ),
    }
}

/// Merge every planned pair into `base`: per parameter, the base weight in
/// float32 plus each adapter's delta in the given order, then the base dtype.
/// Returns the number of parameters changed.
pub fn merge_planned(
    base: &mut HashMap<String, RawTensor>,
    adapters: &[(&AdapterPlan, &dyn AdapterTensors)],
) -> Result<usize> {
    let mut by_target: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (ai, (plan, _)) in adapters.iter().enumerate() {
        for (pi, p) in plan.pairs.iter().enumerate() {
            by_target
                .entry(p.target.clone())
                .or_default()
                .push((ai, pi));
        }
    }
    let mut targets: Vec<String> = by_target.keys().cloned().collect();
    targets.sort();
    // One parameter at a time (each merge is row-parallel inside).
    for target in &targets {
        let t = base
            .remove(target)
            .ok_or_else(|| msg(format!("plug: base lost {target}")))?;
        let mut w = t.to_f32_vec().map_err(|e| msg(e.to_string()))?;
        for &(ai, pi) in &by_target[target] {
            let (plan, src) = adapters[ai];
            let p = &plan.pairs[pi];
            let (a, b) = (src.f32(&p.a_key)?, src.f32(&p.b_key)?);
            merge_pair(&mut w, p.out, p.inn, &a, &b, p.scale).map_err(msg)?;
        }
        if w.par_iter().any(|x| !x.is_finite()) {
            return Err(msg(format!("plug: merged {target} is not finite")));
        }
        base.insert(target.clone(), store_like(t.dtype, t.shape.clone(), w));
    }
    Ok(targets.len())
}

/// An adapter file: `(tensors, safetensors metadata)`.
fn open_adapter(path: &Path) -> Result<(Box<dyn AdapterTensors>, HashMap<String, String>)> {
    let is_pt = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e == "pt" || e == "pth");
    if is_pt {
        let t = fastvideo_loader::pth::read_pth_nested(
            path,
            &["generator_lora", "student_lora", "state_dict"],
            |_| true,
        )
        .map_err(|e| msg(e.to_string()))?;
        let map: HashMap<String, (Vec<usize>, Vec<f32>)> =
            t.into_iter().map(|(k, v)| (k, (v.shape, v.data))).collect();
        return Ok((Box::new(map), HashMap::new()));
    }
    let lazy = LazyStore::open_files(&[path.to_path_buf()])
        .map_err(|e| msg(format!("{}: {e}", path.display())))?;
    let metadata = lazy.metadata().clone();
    Ok((Box::new(lazy), metadata))
}

/// `adapter_config.json` beside the adapter file, if any.
fn adapter_config(path: &Path, metadata: &HashMap<String, String>) -> Result<AdapterConfig> {
    let json = path
        .parent()
        .map(|d| d.join("adapter_config.json"))
        .filter(|p| p.is_file())
        .map(|p| std::fs::read_to_string(&p).map_err(|e| msg(format!("{}: {e}", p.display()))))
        .transpose()?;
    AdapterConfig::from_sources(json.as_deref(), metadata).map_err(msg)
}

/// What [`load_merged_transformer`] did.
#[derive(Debug, Clone)]
pub struct PlugLoad {
    pub recipe: String,
    pub adapters: Vec<(PathBuf, AdapterPlan)>,
    /// Base parameters changed (a parameter both adapters touch counts once).
    pub merged_params: usize,
    pub tensors: usize,
    pub seconds: f64,
}

impl PlugLoad {
    pub fn log_lines(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .adapters
            .iter()
            .map(|(p, plan)| format!("plug lora {} ({})", plan.summary(), p.display()))
            .collect();
        v.push(format!(
            "plug {}: {} base parameters merged of {} tensors in {:.1} s",
            self.recipe, self.merged_params, self.tensors, self.seconds
        ));
        v
    }
}

/// The base checkpoint's Diffusers transformer (`<root>/transformer`) with
/// `recipe`'s adapters merged, as an eager [`WeightMap`] for
/// [`super::pipeline::WanPipeline::load_with_dit`]. Adapters resolve under
/// `plug_root` (default `$FASTVIDEO_PLUG_ROOT`, else `<root>/../longlive-plug`).
/// Every LoRA module of every adapter must match a base weight.
pub fn load_merged_transformer(
    root: &Path,
    recipe: &PlugRecipe,
    plug_root: Option<&Path>,
) -> Result<(WeightMap, PlugLoad)> {
    if recipe.family != PlugFamily::Wan {
        return Err(msg(format!("{} is not a Wan recipe", recipe.name)));
    }
    let timer = std::time::Instant::now();
    let env = std::env::var_os(PLUG_ROOT_ENV).map(PathBuf::from);
    let plug_root = plug_root.map(Path::to_path_buf).or(env);
    let mut base = fastvideo_loader::load_raw_tensors_native(&root.join("transformer"))
        .map_err(|e| msg(format!("{}: {e}", root.join("transformer").display())))?;
    let mut opened = Vec::new();
    for spec in &recipe.adapters {
        let path = spec.resolve(root, plug_root.as_deref()).map_err(msg)?;
        let (src, metadata) = open_adapter(&path)?;
        let config = adapter_config(&path, &metadata)?;
        let plan = plan_wan_adapter(
            &format!("{} {}/{}", recipe.name, spec.dir, spec.file),
            src.as_ref(),
            config,
            spec.weight,
            &base,
        )?;
        super::log::info(format_args!("plug lora {}", plan.summary()));
        plan.require_complete().map_err(msg)?;
        opened.push((path, plan, src));
    }
    let refs: Vec<(&AdapterPlan, &dyn AdapterTensors)> = opened
        .iter()
        .map(|(_, plan, src)| (plan, src.as_ref()))
        .collect();
    let merged_params = merge_planned(&mut base, &refs)?;
    let load = PlugLoad {
        recipe: recipe.name.to_string(),
        tensors: base.len(),
        adapters: opened.into_iter().map(|(p, plan, _)| (p, plan)).collect(),
        merged_params,
        seconds: timer.elapsed().as_secs_f64(),
    };
    for line in load.log_lines() {
        super::log::info(format_args!("{line}"));
    }
    Ok((WeightMap::from_raw_tensors(base), load))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
            .collect()
    }

    fn base() -> HashMap<String, RawTensor> {
        let mut m = HashMap::new();
        // blocks.0.attn1.to_q: [2, 3] bf16; blocks.0.ffn.net.0.proj: [3, 2] f32;
        // an untouched norm.
        m.insert(
            "blocks.0.attn1.to_q.weight".into(),
            RawTensor::from_le_bytes(
                vec![2, 3],
                RawDType::BF16,
                bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
        );
        m.insert(
            "blocks.0.ffn.net.0.proj.weight".into(),
            RawTensor::from_f32(vec![3, 2], vec![0.5; 6]),
        );
        m.insert(
            "blocks.0.norm2.weight".into(),
            RawTensor::from_f32(vec![2], vec![1.0, 1.0]),
        );
        m
    }

    fn adapter(prefix: &str, rank: usize) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
        let mut m = HashMap::new();
        // q: A [rank, 3], B [2, rank]; ffn.0: A [rank, 2], B [3, rank]. All ones.
        let mut put = |k: &str, s: Vec<usize>| {
            let n = s.iter().product();
            m.insert(format!("{prefix}{k}"), (s, vec![1.0f32; n]));
        };
        put("blocks.0.self_attn.q.lora_A.weight", vec![rank, 3]);
        put("blocks.0.self_attn.q.lora_B.weight", vec![2, rank]);
        put("blocks.0.ffn.0.lora_A.weight", vec![rank, 2]);
        put("blocks.0.ffn.0.lora_B.weight", vec![3, rank]);
        m
    }

    #[test]
    fn original_wan_names_map_to_diffusers() {
        for (module, want) in [
            ("blocks.3.self_attn.q", "blocks.3.attn1.to_q.weight"),
            ("blocks.3.self_attn.o", "blocks.3.attn1.to_out.0.weight"),
            ("blocks.3.cross_attn.v", "blocks.3.attn2.to_v.weight"),
            ("blocks.3.ffn.0", "blocks.3.ffn.net.0.proj.weight"),
            ("blocks.3.ffn.2", "blocks.3.ffn.net.2.weight"),
        ] {
            assert_eq!(wan_target(module).as_deref(), Some(want), "{module}");
        }
        assert_eq!(wan_target("blocks.3.bogus.q"), None);
    }

    #[test]
    fn few_step_plus_half_cfg_merges_like_merge_lora_py() {
        let mut b = base();
        let few = adapter("base_model.model.", 2); // PEFT names, rank 2, alpha 4
        let cfg = adapter("", 1); // lightx2v-style bare names, rank 1, alpha = rank
        let p1 = plan_wan_adapter(
            "few",
            &few,
            AdapterConfig {
                alpha: Some(4.0),
                rank: Some(2),
            },
            1.0,
            &b,
        )
        .unwrap();
        let p2 = plan_wan_adapter("cfg", &cfg, AdapterConfig::default(), 0.5, &b).unwrap();
        for p in [&p1, &p2] {
            p.require_complete().unwrap();
            assert_eq!(
                (p.matched(), p.skipped.len(), p.unknown_keys.len()),
                (2, 0, 0)
            );
        }
        let untouched = b["blocks.0.norm2.weight"].data.clone();
        let n = merge_planned(&mut b, &[(&p1, &few), (&p2, &cfg)]).unwrap();
        assert_eq!(n, 2);
        // B @ A of all-ones is `rank` everywhere: 1.0 * (4/2) * 2 + 0.5 * (1/1) * 1 = 4.5.
        let q = &b["blocks.0.attn1.to_q.weight"];
        assert_eq!(q.dtype, RawDType::BF16, "stored back in the base dtype");
        assert_eq!(q.to_f32_vec().unwrap(), vec![5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
        let f = &b["blocks.0.ffn.net.0.proj.weight"];
        assert_eq!(f.dtype, RawDType::F32);
        assert_eq!(f.to_f32_vec().unwrap(), vec![5.0; 6]);
        assert_eq!(b["blocks.0.norm2.weight"].data, untouched);
    }

    #[test]
    fn unmatched_and_unknown_keys_are_reported() {
        let b = base();
        let mut a = adapter("base_model.model.", 2);
        // A block the base does not have, a module family with no Diffusers
        // name, and a non-LoRA tensor.
        a.insert(
            "blocks.7.self_attn.q.lora_A.weight".into(),
            (vec![2, 3], vec![0.0; 6]),
        );
        a.insert(
            "blocks.7.self_attn.q.lora_B.weight".into(),
            (vec![2, 2], vec![0.0; 4]),
        );
        a.insert(
            "blocks.0.modulation.lora_A.weight".into(),
            (vec![2, 3], vec![0.0; 6]),
        );
        a.insert(
            "blocks.0.modulation.lora_B.weight".into(),
            (vec![2, 2], vec![0.0; 4]),
        );
        a.insert("lora_unet_meta".into(), (vec![1], vec![0.0]));
        let plan = plan_wan_adapter("bad", &a, AdapterConfig::default(), 1.0, &b).unwrap();
        assert_eq!(plan.matched(), 2);
        assert_eq!(plan.skipped.len(), 2, "{:?}", plan.skipped);
        assert_eq!(plan.unknown_keys, vec!["lora_unet_meta".to_string()]);
        let e = plan.require_complete().unwrap_err();
        assert!(
            e.contains("2 modules matched, 2 skipped, 1 unknown keys"),
            "{e}"
        );
    }

    #[test]
    fn a_wrong_shape_is_skipped_not_merged() {
        let b = base();
        // q's B claims out=4 but the base q is [2, 3].
        let mut a = adapter("", 1);
        a.insert(
            "blocks.0.self_attn.q.lora_B.weight".into(),
            (vec![4, 1], vec![1.0; 4]),
        );
        let plan = plan_wan_adapter("shape", &a, AdapterConfig::default(), 1.0, &b).unwrap();
        assert_eq!(plan.matched(), 1);
        assert!(
            plan.skipped[0].1.contains("is [2, 3]"),
            "{:?}",
            plan.skipped
        );
    }
}
