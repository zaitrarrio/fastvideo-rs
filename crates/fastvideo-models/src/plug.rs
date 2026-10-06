//! LongLive-Plug (NVlabs, arXiv 2609.38154; `NVlabs/LongLive` `LongLive-Plug/`
//! @ `fb16a879`): capability LoRAs distilled once per backbone, merged into the
//! base weights at load. Opt-in recipes only; none is in the default catalog.
//!
//! Merge rule (`LongLive-Plug/scripts/merge_lora.py`, README "How to run
//! inference"): `W = W_base + Σ_i weight_i · (alpha_i / rank_i) · B_i @ A_i`,
//! accumulated in float32 per weight, then stored in the base dtype. The
//! released Wan recipe is few-step at 1.0 plus CFG at 0.5, 4 steps,
//! guidance 1.0 (conditional pass only). MiniMax-H3's two adapters are used
//! **separately** ("Combined use ... is not recommended at present", both H3
//! cards), each at 1.0.
//!
//! What each recipe runs, and where the source says so:
//!
//! | recipe | adapters (weight) | sampler |
//! |---|---|---|
//! | `h3-plug-4step` | `minimax-h3-few-step/generator_lora.pt` (1.0) | H3 `set_timesteps(5)`: 4 forwards, video shift 12, audio shift 3, predict-x0 then *fresh* re-noise (`source_snapshot/minimax_h3/fresh_noise_scheduler_4step.py`), no CFG |
//! | `h3-plug-cfg` | `minimax-h3-cfg/adapter_model.safetensors` (1.0) | the base 50-point grid (49 forwards), shifts 12 / 3, Euler, positive pass only (`training_config.json`: 50 grid points, `student_conditioning: positive_only`) |
//! | `wan5b-plug-4step` | `wan22-ti2v-5b-few-step` (1.0) + `wan22-ti2v-5b-cfg` (0.5) | `FlowUniPCMultistepScheduler`, 4 steps, shift 5, guidance 1.0 (`LongLive-Plug/inference.py`, `model/base.py` rollout, card `inference_config.yaml`) |
//! | `wan14b-plug-4step` | `wan21-t2v-14b-few-step` lightx2v (1.0) + `wan21-t2v-14b-cfg` (0.5) | LightX2V step-distill Euler on `[1000, 750, 500, 250]` warped with shift 5, `enable_cfg: false` (card `inference_config.json`, `WanStepDistillScheduler`) |
//!
//! Licences: the H3 adapters follow the MiniMax H3 Community Licence
//! (byte-identical `LICENSE` to `MiniMaxAI/MiniMax-H3`'s); the four Wan
//! adapters are Apache-2.0 (docs/serve/research-longlive.md §4.3).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The backbone family an adapter was distilled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlugFamily {
    H3,
    Wan,
}

/// One adapter of a recipe: a directory under the weights root's
/// `longlive-plug/` and the file inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct PlugAdapterSpec {
    /// `longlive-plug/<dir>` (the weights-manifest destination).
    pub dir: &'static str,
    pub file: &'static str,
    /// Merge weight (`--few-step-weight` / `--cfg-weight`).
    pub weight: f32,
}

/// The sampler a recipe's adapters were trained against.
#[derive(Debug, Clone, PartialEq)]
pub enum PlugSampler {
    /// H3's published four-forward configuration with the DMD student's
    /// predict-x0 / fresh re-noise transition.
    H3FreshNoise4,
    /// H3 base grid of `points` sigma points (`points - 1` forwards), Euler.
    H3Base { points: usize },
    /// Wan `FlowUniPCMultistepScheduler`, guidance 1.0.
    WanUniPc { steps: usize, shift: f64 },
    /// LightX2V `WanStepDistillScheduler`: deterministic Euler at these
    /// train timesteps, warped onto the shift-`shift` 1000-point table.
    WanEuler {
        timesteps: &'static [i32],
        shift: f64,
    },
}

/// A LongLive-Plug recipe: base checkpoint, adapters, sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct PlugRecipe {
    pub name: &'static str,
    pub family: PlugFamily,
    /// Weights-manifest destination of the base checkpoint.
    pub base: &'static str,
    /// Wan registry preset (`""` for H3).
    pub preset: &'static str,
    pub adapters: Vec<PlugAdapterSpec>,
    pub sampler: PlugSampler,
    pub licence: &'static str,
}

/// LightX2V `denoising_step_list` of the Wan2.1-14B few-step card.
pub const WAN14B_PLUG_TIMESTEPS: [i32; 4] = [1000, 750, 500, 250];

/// Every recipe name, for help text and tests.
pub const PLUG_RECIPES: [&str; 4] = [
    "h3-plug-4step",
    "h3-plug-cfg",
    "wan5b-plug-4step",
    "wan14b-plug-4step",
];

const H3_LICENCE: &str = "MiniMax H3 Community Licence (as MiniMaxAI/MiniMax-H3)";
const WAN_LICENCE: &str = "Apache-2.0";

impl PlugRecipe {
    /// The recipe `name` names, if it is a LongLive-Plug recipe.
    pub fn named(name: &str) -> Option<Self> {
        let h3 = |name, adapter: PlugAdapterSpec, sampler| Self {
            name,
            family: PlugFamily::H3,
            base: "h3-base",
            preset: "",
            adapters: vec![adapter],
            sampler,
            licence: H3_LICENCE,
        };
        Some(match name {
            "h3-plug-4step" => h3(
                "h3-plug-4step",
                PlugAdapterSpec {
                    dir: "minimax-h3-few-step",
                    file: "generator_lora.pt",
                    weight: 1.0,
                },
                PlugSampler::H3FreshNoise4,
            ),
            "h3-plug-cfg" => h3(
                "h3-plug-cfg",
                PlugAdapterSpec {
                    dir: "minimax-h3-cfg",
                    file: "adapter_model.safetensors",
                    weight: 1.0,
                },
                PlugSampler::H3Base { points: 50 },
            ),
            "wan5b-plug-4step" => Self {
                name: "wan5b-plug-4step",
                family: PlugFamily::Wan,
                base: "wan22-ti2v-5b",
                preset: "wan_2_2_ti2v_5b",
                adapters: vec![
                    PlugAdapterSpec {
                        dir: "wan22-ti2v-5b-few-step",
                        file: "adapter_model.safetensors",
                        weight: 1.0,
                    },
                    PlugAdapterSpec {
                        dir: "wan22-ti2v-5b-cfg",
                        file: "adapter_model.safetensors",
                        weight: 0.5,
                    },
                ],
                sampler: PlugSampler::WanUniPc {
                    steps: 4,
                    shift: 5.0,
                },
                licence: WAN_LICENCE,
            },
            "wan14b-plug-4step" => Self {
                name: "wan14b-plug-4step",
                family: PlugFamily::Wan,
                base: "wan21-t2v-14b",
                preset: "wan_t2v_14b",
                adapters: vec![
                    PlugAdapterSpec {
                        dir: "wan21-t2v-14b-few-step",
                        file: "generator_lora_lightx2v.safetensors",
                        weight: 1.0,
                    },
                    PlugAdapterSpec {
                        dir: "wan21-t2v-14b-cfg",
                        file: "adapter_model.safetensors",
                        weight: 0.5,
                    },
                ],
                sampler: PlugSampler::WanEuler {
                    timesteps: &WAN14B_PLUG_TIMESTEPS,
                    shift: 5.0,
                },
                licence: WAN_LICENCE,
            },
            _ => return None,
        })
    }

    /// Whether `name` is a LongLive-Plug recipe.
    pub fn is_plug(name: &str) -> bool {
        Self::named(name).is_some()
    }
}

/// Environment override of the directory holding the six `longlive-plug/*`
/// adapter directories.
pub const PLUG_ROOT_ENV: &str = "FASTVIDEO_PLUG_ROOT";

impl PlugAdapterSpec {
    /// The adapter file: under `$FASTVIDEO_PLUG_ROOT/<dir>/`, else
    /// `<base_root>/../longlive-plug/<dir>/` (the volume layout, beside the
    /// base checkpoint), else `<base_root>/longlive-plug/<dir>/`.
    pub fn resolve(&self, base_root: &Path, plug_root: Option<&Path>) -> Result<PathBuf, String> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Some(r) = plug_root {
            roots.push(r.to_path_buf());
        }
        if let Some(parent) = base_root.parent() {
            roots.push(parent.join("longlive-plug"));
        }
        roots.push(base_root.join("longlive-plug"));
        let mut tried = Vec::new();
        for r in roots {
            let p = r.join(self.dir).join(self.file);
            if p.is_file() {
                return Ok(p);
            }
            tried.push(p.display().to_string());
        }
        Err(format!(
            "LongLive-Plug adapter {}/{} not found (looked for {})",
            self.dir,
            self.file,
            tried.join(", ")
        ))
    }
}

/// Which half of a LoRA pair a key holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraSide {
    /// `lora_A`: `[rank, in]`.
    A,
    /// `lora_B`: `[out, rank]`.
    B,
}

/// Wrappers a PEFT / LongLive / ComfyUI export puts in front of the module
/// name (`merge_lora.py` strips `base_model.model.`; the lightx2v export
/// also accepts `model.` / `module.` in front of it).
const KEY_PREFIXES: &[&str] = &[
    "model.base_model.model.",
    "module.base_model.model.",
    "base_model.model.",
    "diffusion_model.",
];

/// `(module, side)` of a LoRA key: `<prefix><module>.lora_{A,B}[.default].weight`.
/// `None` for anything else (it is reported as an unknown key).
pub fn lora_key(key: &str) -> Option<(String, LoraSide)> {
    let mut k = key;
    for p in KEY_PREFIXES {
        if let Some(rest) = k.strip_prefix(p) {
            k = rest;
            break;
        }
    }
    let k = k.strip_suffix(".weight")?;
    let k = k.strip_suffix(".default").unwrap_or(k);
    if let Some(m) = k.strip_suffix(".lora_A") {
        return (!m.is_empty()).then(|| (m.to_string(), LoraSide::A));
    }
    if let Some(m) = k.strip_suffix(".lora_B") {
        return (!m.is_empty()).then(|| (m.to_string(), LoraSide::B));
    }
    None
}

/// `lora_alpha` and `r` an adapter declares: its `adapter_config.json`
/// (PEFT), else the safetensors metadata (`alpha` / `lora_alpha`, `rank` /
/// `lora_rank`), else nothing (the merge then uses `alpha = rank`, as
/// `merge_lora.py` does without `--*-alpha`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdapterConfig {
    pub alpha: Option<f64>,
    pub rank: Option<usize>,
}

impl AdapterConfig {
    pub fn from_sources(
        config_json: Option<&str>,
        metadata: &HashMap<String, String>,
    ) -> Result<Self, String> {
        if let Some(text) = config_json {
            let v: serde_json::Value =
                serde_json::from_str(text).map_err(|e| format!("adapter_config.json: {e}"))?;
            if v.get("rank_pattern")
                .and_then(|p| p.as_object())
                .is_some_and(|p| !p.is_empty())
                || v.get("alpha_pattern")
                    .and_then(|p| p.as_object())
                    .is_some_and(|p| !p.is_empty())
            {
                return Err(
                    "adapter_config.json: per-module rank/alpha patterns are not supported".into(),
                );
            }
            if v.get("use_dora").and_then(|d| d.as_bool()) == Some(true)
                || v.get("use_rslora").and_then(|d| d.as_bool()) == Some(true)
            {
                return Err("adapter_config.json: DoRA / rsLoRA adapters are not supported".into());
            }
            return Ok(Self {
                alpha: v.get("lora_alpha").and_then(|a| a.as_f64()),
                rank: v.get("r").and_then(|r| r.as_u64()).map(|r| r as usize),
            });
        }
        let num = |keys: &[&str]| -> Result<Option<f64>, String> {
            for k in keys {
                if let Some(s) = metadata.get(*k) {
                    return s
                        .trim()
                        .parse::<f64>()
                        .map(Some)
                        .map_err(|_| format!("adapter metadata {k}={s:?} is not a number"));
                }
            }
            Ok(None)
        };
        Ok(Self {
            alpha: num(&["lora_alpha", "alpha"])?,
            rank: num(&["lora_rank", "rank"])?.map(|r| r as usize),
        })
    }
}

/// `weight · alpha / rank` (`alpha = rank` when the adapter declares none).
pub fn delta_scale(weight: f32, alpha: Option<f64>, rank: usize) -> Result<f32, String> {
    if rank == 0 {
        return Err("LoRA rank must be at least 1".into());
    }
    if !weight.is_finite() {
        return Err("adapter weight must be finite".into());
    }
    let alpha = alpha.unwrap_or(rank as f64);
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(format!("LoRA alpha must be positive, got {alpha}"));
    }
    Ok((f64::from(weight) * alpha / rank as f64) as f32)
}

/// One pair the merge applies.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedPair {
    /// The adapter's module name (prefix stripped).
    pub module: String,
    /// The base parameter it updates (`<target>.weight`, base naming).
    pub target: String,
    pub a_key: String,
    pub b_key: String,
    pub rank: usize,
    pub out: usize,
    pub inn: usize,
    /// `weight · alpha / rank`.
    pub scale: f32,
}

/// One adapter against one base checkpoint: what merges, what does not.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterPlan {
    pub label: String,
    pub weight: f32,
    pub alpha: Option<f64>,
    pub pairs: Vec<PlannedPair>,
    /// `(module, reason)` of LoRA modules that do not merge.
    pub skipped: Vec<(String, String)>,
    /// Keys that are not LoRA A/B tensors.
    pub unknown_keys: Vec<String>,
}

impl AdapterPlan {
    pub fn matched(&self) -> usize {
        self.pairs.len()
    }

    /// Distinct ranks of the merged pairs.
    pub fn ranks(&self) -> Vec<usize> {
        let mut r: Vec<usize> = self.pairs.iter().map(|p| p.rank).collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// `label: N matched, M skipped, K unknown keys (rank R, alpha A, weight W)`.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} modules matched, {} skipped, {} unknown keys (rank {:?}, alpha {}, weight {}, scale {})",
            self.label,
            self.matched(),
            self.skipped.len(),
            self.unknown_keys.len(),
            self.ranks(),
            self.alpha
                .map_or_else(|| "=rank".to_string(), |a| a.to_string()),
            self.weight,
            self.pairs.first().map_or(0.0, |p| p.scale),
        )
    }

    /// Ok when every LoRA module merges and every key is a LoRA tensor
    /// (the released adapters all do); otherwise the first few offenders.
    pub fn require_complete(&self) -> Result<(), String> {
        if self.skipped.is_empty() && self.unknown_keys.is_empty() && !self.pairs.is_empty() {
            return Ok(());
        }
        let skipped: Vec<String> = self
            .skipped
            .iter()
            .take(3)
            .map(|(m, why)| format!("{m} ({why})"))
            .collect();
        Err(format!(
            "{}; skipped {:?}, unknown {:?}",
            self.summary(),
            skipped,
            &self.unknown_keys[..self.unknown_keys.len().min(3)]
        ))
    }
}

/// Pair `keys` (name and shape of every adapter tensor) into LoRA modules and
/// check each against the base: `rename(module)` gives the base parameter
/// name (`None`: no such parameter family), `target_shape(param)` its shape
/// (`None`: absent). Nothing is read but names and shapes.
pub fn plan_adapter(
    label: &str,
    keys: &[(String, Vec<usize>)],
    config: AdapterConfig,
    weight: f32,
    rename: &dyn Fn(&str) -> Option<String>,
    target_shape: &dyn Fn(&str) -> Option<Vec<usize>>,
) -> Result<AdapterPlan, String> {
    // module -> [A, B], each `(key, shape)`.
    type Halves = [Option<(String, Vec<usize>)>; 2];
    let mut modules: BTreeMap<String, Halves> = BTreeMap::new();
    let mut unknown_keys = Vec::new();
    for (key, shape) in keys {
        match lora_key(key) {
            Some((module, side)) => {
                let slot = &mut modules.entry(module.clone()).or_default()
                    [usize::from(side == LoraSide::B)];
                if slot.replace((key.clone(), shape.clone())).is_some() {
                    return Err(format!("{label}: two {side:?} tensors for {module}"));
                }
            }
            None => unknown_keys.push(key.clone()),
        }
    }
    unknown_keys.sort();
    let mut pairs = Vec::new();
    let mut skipped = Vec::new();
    for (module, [a, b]) in modules {
        let (Some((a_key, a_shape)), Some((b_key, b_shape))) = (a, b) else {
            skipped.push((module, "unpaired (only one of lora_A / lora_B)".into()));
            continue;
        };
        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[0] != b_shape[1] || a_shape[0] == 0 {
            skipped.push((module, format!("bad pair A{a_shape:?} B{b_shape:?}")));
            continue;
        }
        let Some(target) = rename(&module) else {
            skipped.push((module, "no base parameter name".into()));
            continue;
        };
        let Some(base) = target_shape(&target) else {
            skipped.push((module, format!("{target} absent from the base")));
            continue;
        };
        if base != [b_shape[0], a_shape[1]] {
            skipped.push((
                module,
                format!(
                    "{target} is {base:?}, the pair makes [{}, {}]",
                    b_shape[0], a_shape[1]
                ),
            ));
            continue;
        }
        if let Some(declared) = config.rank {
            if declared != a_shape[0] {
                skipped.push((
                    module,
                    format!("rank {} but the config declares r={declared}", a_shape[0]),
                ));
                continue;
            }
        }
        let rank = a_shape[0];
        pairs.push(PlannedPair {
            module,
            target,
            a_key,
            b_key,
            rank,
            out: b_shape[0],
            inn: a_shape[1],
            scale: delta_scale(weight, config.alpha, rank)?,
        });
    }
    // Two modules must not update the same base parameter within one adapter.
    let mut seen = std::collections::HashSet::new();
    for p in &pairs {
        if !seen.insert(p.target.as_str()) {
            return Err(format!("{label}: two LoRA modules map to {}", p.target));
        }
    }
    Ok(AdapterPlan {
        label: label.to_string(),
        weight,
        alpha: config.alpha,
        pairs,
        skipped,
        unknown_keys,
    })
}

/// `w[out, in] += scale · b[out, rank] @ a[rank, in]` in float32.
pub fn merge_pair(
    w: &mut [f32],
    out: usize,
    inn: usize,
    a: &[f32],
    b: &[f32],
    scale: f32,
) -> Result<(), String> {
    crate::h3::lora::add_low_rank(w, out, inn, b, a, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wan_like_rename(module: &str) -> Option<String> {
        // Test stand-in for the original-Wan -> Diffusers map.
        let (blk, rest) = module.strip_prefix("blocks.")?.split_once('.')?;
        let to = match rest {
            "self_attn.q" => "attn1.to_q",
            "self_attn.o" => "attn1.to_out.0",
            "cross_attn.k" => "attn2.to_k",
            "ffn.0" => "ffn.net.0.proj",
            _ => return None,
        };
        Some(format!("blocks.{blk}.{to}.weight"))
    }

    #[test]
    fn keys_strip_every_released_prefix() {
        let cases = [
            (
                "base_model.model.blocks.0.self_attn.q.lora_A.weight",
                "blocks.0.self_attn.q",
                LoraSide::A,
            ),
            (
                "blocks.9.ffn.2.lora_B.weight",
                "blocks.9.ffn.2",
                LoraSide::B,
            ),
            (
                "transformer_blocks.3.attn.to_out.0.lora_A.default.weight",
                "transformer_blocks.3.attn.to_out.0",
                LoraSide::A,
            ),
            (
                "diffusion_model.token_refiner.refiner_blocks.1.ff.net.2.lora_B.weight",
                "token_refiner.refiner_blocks.1.ff.net.2",
                LoraSide::B,
            ),
            (
                "module.base_model.model.blocks.1.cross_attn.v.lora_B.weight",
                "blocks.1.cross_attn.v",
                LoraSide::B,
            ),
        ];
        for (key, module, side) in cases {
            assert_eq!(lora_key(key), Some((module.to_string(), side)), "{key}");
        }
        for bad in [
            "blocks.0.self_attn.q.weight",
            "lora_A.weight",
            "blocks.0.alpha",
            "x.lora_A.bias",
        ] {
            assert_eq!(lora_key(bad), None, "{bad}");
        }
    }

    #[test]
    fn plan_counts_matched_skipped_and_unknown() {
        let base: HashMap<&str, Vec<usize>> = [
            ("blocks.0.attn1.to_q.weight", vec![4, 3]),
            ("blocks.0.attn1.to_out.0.weight", vec![4, 4]),
            ("blocks.0.attn2.to_k.weight", vec![4, 3]),
        ]
        .into_iter()
        .collect();
        let keys: Vec<(String, Vec<usize>)> = [
            // matched (PEFT prefix)
            (
                "base_model.model.blocks.0.self_attn.q.lora_A.weight",
                vec![2, 3],
            ),
            (
                "base_model.model.blocks.0.self_attn.q.lora_B.weight",
                vec![4, 2],
            ),
            // shape mismatch: the base is [4, 4]
            ("blocks.0.self_attn.o.lora_A.weight", vec![2, 3]),
            ("blocks.0.self_attn.o.lora_B.weight", vec![4, 2]),
            // absent from the base (block 1)
            ("blocks.1.self_attn.q.lora_A.weight", vec![2, 3]),
            ("blocks.1.self_attn.q.lora_B.weight", vec![4, 2]),
            // no Diffusers name for this family
            ("blocks.0.norm3.lora_A.weight", vec![2, 3]),
            ("blocks.0.norm3.lora_B.weight", vec![3, 2]),
            // unpaired
            ("blocks.0.cross_attn.k.lora_A.weight", vec![2, 3]),
            // not a LoRA tensor at all
            ("blocks.0.self_attn.q.alpha", vec![]),
        ]
        .into_iter()
        .map(|(k, s)| (k.to_string(), s))
        .collect();
        let plan = plan_adapter(
            "test",
            &keys,
            AdapterConfig {
                alpha: Some(4.0),
                rank: Some(2),
            },
            0.5,
            &wan_like_rename,
            &|p| base.get(p).cloned(),
        )
        .unwrap();
        assert_eq!(plan.matched(), 1);
        assert_eq!(plan.pairs[0].target, "blocks.0.attn1.to_q.weight");
        assert_eq!(plan.pairs[0].scale, 0.5 * 4.0 / 2.0);
        assert_eq!(plan.skipped.len(), 4, "{:?}", plan.skipped);
        let why: HashMap<&str, &str> = plan
            .skipped
            .iter()
            .map(|(m, w)| (m.as_str(), w.as_str()))
            .collect();
        assert!(why["blocks.0.self_attn.o"].contains("is [4, 4]"));
        assert!(why["blocks.1.self_attn.q"].contains("absent"));
        assert!(why["blocks.0.norm3"].contains("no base parameter"));
        assert!(why["blocks.0.cross_attn.k"].contains("unpaired"));
        assert_eq!(
            plan.unknown_keys,
            vec!["blocks.0.self_attn.q.alpha".to_string()]
        );
        let e = plan.require_complete().unwrap_err();
        assert!(
            e.contains("1 modules matched, 4 skipped, 1 unknown keys"),
            "{e}"
        );
    }

    #[test]
    fn a_complete_adapter_passes_and_declared_rank_is_checked() {
        let keys: Vec<(String, Vec<usize>)> = vec![
            (
                "transformer_blocks.0.attn.to_q.lora_A.weight".into(),
                vec![2, 3],
            ),
            (
                "transformer_blocks.0.attn.to_q.lora_B.weight".into(),
                vec![5, 2],
            ),
        ];
        let ident = |m: &str| Some(format!("{m}.weight"));
        let shape = |_: &str| Some(vec![5, 3]);
        let ok = plan_adapter("h3", &keys, AdapterConfig::default(), 1.0, &ident, &shape).unwrap();
        ok.require_complete().unwrap();
        assert_eq!(ok.pairs[0].scale, 1.0, "alpha defaults to the rank");
        let bad = plan_adapter(
            "h3",
            &keys,
            AdapterConfig {
                alpha: Some(128.0),
                rank: Some(128),
            },
            1.0,
            &ident,
            &shape,
        )
        .unwrap();
        assert_eq!(bad.matched(), 0);
        assert!(bad.skipped[0].1.contains("declares r=128"));
    }

    #[test]
    fn two_adapters_merge_as_the_plug_rule() {
        // W (2x3) + 1.0 * (a1/r1) B1 A1 + 0.5 * (a2/r2) B2 A2, by hand.
        let w0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (a1, b1) = (vec![1.0f32, 0.0, -1.0], vec![2.0f32, 1.0]); // rank 1
        let (a2, b2) = (
            vec![1.0f32, 1.0, 1.0, 0.0, 2.0, 0.0], // rank 2: [2, 3]
            vec![1.0f32, 0.0, 0.0, 3.0],           // [2, 2]
        );
        let s1 = delta_scale(1.0, Some(2.0), 1).unwrap(); // alpha 2 / rank 1 = 2
        let s2 = delta_scale(0.5, None, 2).unwrap(); // alpha = rank: 0.5
        assert_eq!((s1, s2), (2.0, 0.5));
        let mut w = w0.clone();
        merge_pair(&mut w, 2, 3, &a1, &b1, s1).unwrap();
        merge_pair(&mut w, 2, 3, &a2, &b2, s2).unwrap();
        // B1 A1 = [[2,0,-2],[1,0,-1]]; B2 A2 = [[1,1,1],[0,6,0]].
        let want = [
            1.0 + 2.0 * 2.0 + 0.5 * 1.0,
            2.0 + 0.0 + 0.5 * 1.0,
            3.0 - 2.0 * 2.0 + 0.5 * 1.0,
            4.0 + 2.0 * 1.0 + 0.0,
            5.0 + 0.0 + 0.5 * 6.0,
            6.0 - 2.0 * 1.0 + 0.0,
        ];
        assert_eq!(w, want);
        assert!(merge_pair(&mut w, 2, 3, &a1, &[1.0], 1.0).is_err());
    }

    #[test]
    fn adapter_config_reads_peft_json_then_metadata() {
        let none = HashMap::new();
        let peft = r#"{"r": 64, "lora_alpha": 64, "rank_pattern": {}, "alpha_pattern": {}, "use_dora": false}"#;
        assert_eq!(
            AdapterConfig::from_sources(Some(peft), &none).unwrap(),
            AdapterConfig {
                alpha: Some(64.0),
                rank: Some(64)
            }
        );
        let lx: HashMap<String, String> = [("rank", "128"), ("alpha", "128"), ("format", "pt")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(
            AdapterConfig::from_sources(None, &lx).unwrap(),
            AdapterConfig {
                alpha: Some(128.0),
                rank: Some(128)
            }
        );
        // 5B-cfg card metadata names them lora_rank / lora_alpha.
        let cfg: HashMap<String, String> = [("lora_rank", "64"), ("lora_alpha", "64")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(
            AdapterConfig::from_sources(None, &cfg).unwrap().rank,
            Some(64)
        );
        assert_eq!(
            AdapterConfig::from_sources(None, &none).unwrap(),
            AdapterConfig::default()
        );
        assert!(
            AdapterConfig::from_sources(Some(r#"{"r": 8, "rank_pattern": {"x": 4}}"#), &none)
                .is_err()
        );
        assert!(AdapterConfig::from_sources(Some(r#"{"r": 8, "use_dora": true}"#), &none).is_err());
    }

    #[test]
    fn recipes_follow_the_cards() {
        for name in PLUG_RECIPES {
            let r = PlugRecipe::named(name).unwrap();
            assert_eq!(r.name, name);
            assert!(PlugRecipe::is_plug(name));
        }
        assert!(!PlugRecipe::is_plug("4step-dense"));
        let h3 = PlugRecipe::named("h3-plug-4step").unwrap();
        assert_eq!(h3.adapters.len(), 1, "H3 adapters are used separately");
        assert_eq!(h3.sampler, PlugSampler::H3FreshNoise4);
        assert!(h3.licence.contains("MiniMax"));
        let cfg = PlugRecipe::named("h3-plug-cfg").unwrap();
        assert_eq!(cfg.adapters[0].weight, 1.0);
        assert_eq!(cfg.sampler, PlugSampler::H3Base { points: 50 });
        for name in ["wan5b-plug-4step", "wan14b-plug-4step"] {
            let r = PlugRecipe::named(name).unwrap();
            let w: Vec<f32> = r.adapters.iter().map(|a| a.weight).collect();
            assert_eq!(w, vec![1.0, 0.5], "{name}: few-step 1.0 + CFG 0.5");
            assert_eq!(r.licence, "Apache-2.0");
            assert_eq!(r.family, PlugFamily::Wan);
        }
        assert_eq!(
            PlugRecipe::named("wan5b-plug-4step").unwrap().sampler,
            PlugSampler::WanUniPc {
                steps: 4,
                shift: 5.0
            }
        );
        assert_eq!(
            PlugRecipe::named("wan14b-plug-4step").unwrap().sampler,
            PlugSampler::WanEuler {
                timesteps: &[1000, 750, 500, 250],
                shift: 5.0
            }
        );
    }

    #[test]
    fn adapters_resolve_beside_the_base_checkpoint() {
        let root = std::env::temp_dir().join(format!("fv-plug-{}", std::process::id()));
        let base = root.join("wan22-ti2v-5b");
        let dir = root.join("longlive-plug/wan22-ti2v-5b-cfg");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("adapter_model.safetensors"), b"x").unwrap();
        let spec = &PlugRecipe::named("wan5b-plug-4step").unwrap().adapters[1];
        assert_eq!(
            spec.resolve(&base, None).unwrap(),
            dir.join("adapter_model.safetensors")
        );
        let e = PlugRecipe::named("wan5b-plug-4step").unwrap().adapters[0]
            .resolve(&base, None)
            .unwrap_err();
        assert!(e.contains("wan22-ti2v-5b-few-step"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
