//! Sol-H3 adapter fusion (`h3_runtime/lora.py`).
//!
//! FastVideo `fastvideo-lora-v2` checkpoints are hybrid: `W += B @ A`, plus
//! tiny `.diff` / `.diff_b` corrections that must accumulate in float32.
//! PEFT adapters use `W += scale * alpha / rank * B @ A`. Hybrid adapters
//! may also carry `.set_weight` replacements (Spark VSA-DataFree
//! `to_gate_compress`); PEFT replacements stay rejected.
//!
//! `A` is `[rank, in]`, `B` is `[out, rank]`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

const LORA_SUFFIXES: &[(&str, Side)] = &[
    (".lora_A.default.weight", Side::A),
    (".lora_B.default.weight", Side::B),
    (".lora_A.weight", Side::A),
    (".lora_B.weight", Side::B),
];

const LORA_TARGET_SUFFIXES: &[&str] = &[
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
];

#[derive(Clone, Copy)]
enum Side {
    A,
    B,
}

/// On-disk adapter family, from safetensors `__metadata__.format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraFormat {
    /// `format` absent or anything other than the FastVideo hybrid tag.
    Peft,
    /// `format == "fastvideo-lora-v2"`. Alpha is the rank; `.diff` tensors apply.
    FastvideoV2,
}

/// Keys of one adapter, validated before any base weight is touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraPlan {
    pub format: LoraFormat,
    /// `(module, A key, B key)`, sorted by module name.
    pub pairs: Vec<(String, String, String)>,
    /// `(parameter name, diff key)`, sorted. Parameter names end in `.weight` or `.bias`.
    pub diffs: Vec<(String, String)>,
    /// `(parameter name, adapter key)` for hybrid `.set_weight` overwrites.
    /// Parameter names end in `.weight` or `.bias`.
    pub replacements: Vec<(String, String)>,
    /// `rank` from hybrid metadata, when the file declared one.
    pub metadata_rank: Option<usize>,
}

/// `W += multiplier * B @ A` with `A: [rank, inn]`, `B: [out, rank]`.
pub fn add_low_rank(
    weight: &mut [f32],
    out: usize,
    inn: usize,
    b: &[f32],
    a: &[f32],
    multiplier: f32,
) -> Result<(), String> {
    if out == 0 || inn == 0 || weight.len() != out * inn {
        return Err(format!("lora weight len {} != {out}x{inn}", weight.len()));
    }
    if b.is_empty() || b.len() % out != 0 {
        return Err(format!(
            "lora B len {} is not a multiple of out {out}",
            b.len()
        ));
    }
    let rank = b.len() / out;
    if a.len() != rank * inn {
        return Err(format!("lora A len {} != rank {rank} * in {inn}", a.len()));
    }
    weight.par_chunks_mut(inn).enumerate().for_each(|(o, row)| {
        for r in 0..rank {
            let scale = multiplier * b[o * rank + r];
            if scale == 0.0 {
                continue;
            }
            let a_row = &a[r * inn..(r + 1) * inn];
            for (dst, &av) in row.iter_mut().zip(a_row) {
                *dst += scale * av;
            }
        }
    });
    Ok(())
}

/// Float32 `.diff` correction: `param += scale * delta`.
pub fn add_diff(param: &mut [f32], delta: &[f32], scale: f32) -> Result<(), String> {
    if param.len() != delta.len() {
        return Err(format!(
            "adapter diff len {} != parameter {}",
            delta.len(),
            param.len()
        ));
    }
    for (p, d) in param.iter_mut().zip(delta) {
        *p += scale * *d;
    }
    Ok(())
}

/// `scale * alpha / rank`. Hybrid adapters use `alpha = rank`, so the product is `scale`.
pub fn lora_multiplier(
    format: LoraFormat,
    rank: usize,
    alpha: u32,
    scale: f32,
) -> Result<f32, String> {
    if rank == 0 {
        return Err("lora rank must be at least 1".into());
    }
    if alpha < 1 {
        return Err("LoRA alpha must be at least 1".into());
    }
    if !scale.is_finite() || scale < 0.0 {
        return Err("LoRA scale must be finite and non-negative".into());
    }
    let applied = match format {
        LoraFormat::FastvideoV2 => rank as f32,
        LoraFormat::Peft => alpha as f32,
    };
    Ok(scale * applied / rank as f32)
}

/// Classify adapter keys the way `lora.py:_payload` does.
pub fn plan_from_keys(
    keys: &[String],
    metadata: &HashMap<String, String>,
    path: &str,
) -> Result<LoraPlan, String> {
    let format = match metadata.get("format").map(String::as_str) {
        Some("fastvideo-lora-v2") => LoraFormat::FastvideoV2,
        _ => LoraFormat::Peft,
    };
    let hybrid = format == LoraFormat::FastvideoV2;
    let mut a_keys: HashMap<String, String> = HashMap::new();
    let mut b_keys: HashMap<String, String> = HashMap::new();
    let mut diffs: HashMap<String, String> = HashMap::new();
    let mut replacements: HashMap<String, String> = HashMap::new();
    let mut unsupported = Vec::new();

    for key in keys {
        let mut matched = false;
        for (suffix, side) in LORA_SUFFIXES {
            if let Some(stem) = key.strip_suffix(suffix) {
                let module = stem.strip_prefix("diffusion_model.").unwrap_or(stem);
                match side {
                    Side::A => a_keys.insert(module.to_string(), key.clone()),
                    Side::B => b_keys.insert(module.to_string(), key.clone()),
                };
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        if hybrid {
            if let Some(stem) = key.strip_suffix(".diff_b") {
                let module = stem.strip_prefix("diffusion_model.").unwrap_or(stem);
                diffs.insert(format!("{module}.bias"), key.clone());
                continue;
            }
            if let Some(stem) = key.strip_suffix(".diff") {
                let module = stem.strip_prefix("diffusion_model.").unwrap_or(stem);
                diffs.insert(format!("{module}.weight"), key.clone());
                continue;
            }
        }
        if key.ends_with(".set_weight") {
            if !hybrid {
                return Err(format!(
                    "{path} contains replacement tensors; PEFT adapters cannot use .set_weight"
                ));
            }
            let stem = key
                .strip_suffix(".set_weight")
                .expect("suffix")
                .strip_prefix("diffusion_model.")
                .unwrap_or(key.strip_suffix(".set_weight").expect("suffix"));
            let param = if stem.ends_with(".weight") || stem.ends_with(".bias") {
                stem.to_string()
            } else {
                format!("{stem}.weight")
            };
            replacements.insert(param, key.clone());
        } else {
            unsupported.push(key.clone());
        }
    }
    if !unsupported.is_empty() {
        return Err(format!(
            "{path} contains unsupported adapter keys: {:?}",
            &unsupported[..unsupported.len().min(3)]
        ));
    }
    if a_keys.is_empty() && replacements.is_empty() {
        return Err(format!("No LoRA A tensors found in {path}"));
    }
    let missing_a: Vec<_> = b_keys
        .keys()
        .filter(|k| !a_keys.contains_key(*k))
        .cloned()
        .collect();
    let missing_b: Vec<_> = a_keys
        .keys()
        .filter(|k| !b_keys.contains_key(*k))
        .cloned()
        .collect();
    if !missing_a.is_empty() || !missing_b.is_empty() {
        return Err(format!(
            "Unpaired LoRA tensors: missing A={:?}, missing B={:?}",
            &missing_a[..missing_a.len().min(3)],
            &missing_b[..missing_b.len().min(3)]
        ));
    }
    if !hybrid {
        let bad: Vec<_> = a_keys
            .keys()
            .filter(|name| !LORA_TARGET_SUFFIXES.iter().any(|s| name.ends_with(s)))
            .cloned()
            .collect();
        if !bad.is_empty() {
            return Err(format!("Unsupported LoRA target module: {}", bad[0]));
        }
    }
    if hybrid {
        let pairs = a_keys.len();
        let expect_low = meta_usize(metadata, "low_rank_tensors").unwrap_or(pairs * 2);
        let expect_diffs = meta_usize(metadata, "diff_tensors").unwrap_or(diffs.len());
        let expect_repl = meta_usize(metadata, "set_weight_tensors").unwrap_or(replacements.len());
        if expect_low != pairs * 2
            || expect_diffs != diffs.len()
            || expect_repl != replacements.len()
        {
            return Err(format!(
                "FastVideo adapter metadata/payload mismatch: low_rank={expect_low}/{}, diffs={expect_diffs}/{}, set_weight={expect_repl}/{}",
                pairs * 2,
                diffs.len(),
                replacements.len()
            ));
        }
    }
    let mut modules: Vec<_> = a_keys.keys().cloned().collect();
    modules.sort();
    let pairs = modules
        .into_iter()
        .map(|name| {
            let a = a_keys.remove(&name).expect("a key");
            let b = b_keys.remove(&name).expect("b key");
            (name, a, b)
        })
        .collect();
    let mut diff_names: Vec<_> = diffs.keys().cloned().collect();
    diff_names.sort();
    let diffs = diff_names
        .into_iter()
        .map(|name| {
            let key = diffs.remove(&name).expect("diff key");
            (name, key)
        })
        .collect();
    let mut repl_names: Vec<_> = replacements.keys().cloned().collect();
    repl_names.sort();
    let replacements = repl_names
        .into_iter()
        .map(|name| {
            let key = replacements.remove(&name).expect("replacement key");
            (name, key)
        })
        .collect();
    Ok(LoraPlan {
        format,
        pairs,
        diffs,
        replacements,
        metadata_rank: if hybrid {
            meta_usize(metadata, "rank")
        } else {
            None
        },
    })
}

fn meta_usize(metadata: &HashMap<String, String>, key: &str) -> Option<usize> {
    metadata.get(key)?.parse().ok()
}

/// Where a Sol-H3 task looks for its adapter, and the PEFT alpha that task uses.
/// Hybrid FastVideo files ignore `alpha` (`lora_multiplier`).
#[derive(Debug, Clone, Copy)]
pub struct SolH3AdapterSpec {
    pub alpha: u32,
    pub scale: f32,
    pub relative_paths: &'static [&'static str],
}

impl SolH3AdapterSpec {
    /// T2V and first-frame I2V: FastH3 4-step dense-datafree LoRA, alpha 64.
    pub fn t2v_i2v() -> Self {
        Self {
            alpha: 64,
            scale: 1.0,
            relative_paths: &[
                "adapter/dense-datafree/adapter_model.safetensors",
                "dense-datafree/adapter_model.safetensors",
                "FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors",
            ],
        }
    }

    /// Ref2VA: lightx2v turbo adapter, alpha 8.
    pub fn ref2va() -> Self {
        Self {
            alpha: 8,
            scale: 1.0,
            relative_paths: &[
                "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
                "Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
            ],
        }
    }

    /// `explicit`, then each relative path under `root` and under `root`'s parent
    /// (the Sol-H3 `checkpoints/` layout puts the adapter beside `MiniMax-H3/`).
    pub fn resolve(&self, root: &Path, explicit: Option<&Path>) -> Result<PathBuf, String> {
        if let Some(path) = explicit {
            if path.is_file() {
                return Ok(path.to_path_buf());
            }
            return Err(format!(
                "LoRA checkpoint does not exist: {}",
                path.display()
            ));
        }
        let mut tried = Vec::new();
        let parents = [Some(root), root.parent()];
        for base in parents.into_iter().flatten() {
            for rel in self.relative_paths {
                let path = base.join(rel);
                if path.is_file() {
                    return Ok(path);
                }
                tried.push(path.display().to_string());
            }
        }
        Err(format!(
            "Sol-H3 adapter not found (alpha {}). Looked for: {}",
            self.alpha,
            tried.join(", ")
        ))
    }
}

/// Recipe names that select the Sol-H3 contract.
pub fn is_sol_h3_recipe(name: &str) -> bool {
    matches!(
        name,
        "sol-h3"
            | "sol_h3"
            | "sol-h3-t2v"
            | "sol-h3-i2v"
            | "sol-h3-ref2va"
            | "sol_h3_ref2va"
            | "sol-h3-spark"
            | "sol_h3_spark"
    )
}

/// Names that load `transformer_ref/` and the turbo adapter.
pub fn sol_h3_forces_ref2va(name: &str) -> bool {
    matches!(name, "sol-h3-ref2va" | "sol_h3_ref2va")
}

/// Spark draft geometry (`672×384×124`) on the existing Sol-H3 recipe.
pub fn is_sol_h3_spark_recipe(name: &str) -> bool {
    matches!(name, "sol-h3-spark" | "sol_h3_spark")
}

/// Official 49-forward RTX 5090 cell. Does not fuse the 4-step adapter.
pub fn is_sol_h3_rtx_recipe(name: &str) -> bool {
    matches!(name, "sol-h3-rtx" | "sol_h3_rtx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_is_a_sol_h3_recipe() {
        assert!(is_sol_h3_recipe("sol-h3-spark"));
        assert!(is_sol_h3_spark_recipe("sol-h3-spark"));
        assert!(!is_sol_h3_spark_recipe("sol-h3"));
        assert!(!sol_h3_forces_ref2va("sol-h3-spark"));
        assert!(is_sol_h3_rtx_recipe("sol-h3-rtx"));
        assert!(!is_sol_h3_rtx_recipe("sol-h3"));
        assert!(!is_sol_h3_recipe("sol-h3-rtx"));
    }

    #[test]
    fn peft_multiplier_uses_alpha() {
        assert_eq!(lora_multiplier(LoraFormat::Peft, 16, 64, 1.0).unwrap(), 4.0);
        assert_eq!(lora_multiplier(LoraFormat::Peft, 32, 8, 1.0).unwrap(), 0.25);
        assert_eq!(
            lora_multiplier(LoraFormat::FastvideoV2, 16, 64, 1.0).unwrap(),
            1.0
        );
    }

    #[test]
    fn low_rank_update_is_b_at_a() {
        // out=2, in=3, rank=1. B = [2, 1] = [2, 3], A = [1, 3] = [1, 0, -1]
        // delta row0 = 2 * A, row1 = 3 * A, multiplier 1.
        let mut w = vec![0.0; 6];
        add_low_rank(&mut w, 2, 3, &[2.0, 3.0], &[1.0, 0.0, -1.0], 1.0).unwrap();
        assert_eq!(w, vec![2.0, 0.0, -2.0, 3.0, 0.0, -3.0]);
    }

    #[test]
    fn hybrid_plan_accepts_replacements_and_counts() {
        let keys = vec![
            "blocks.0.attn.to_q.lora_A.weight".into(),
            "blocks.0.attn.to_q.lora_B.weight".into(),
            "blocks.0.attn.to_q.diff".into(),
            "blocks.0.attn.to_q.set_weight".into(),
        ];
        let mut meta = HashMap::new();
        meta.insert("format".into(), "fastvideo-lora-v2".into());
        let plan = plan_from_keys(&keys, &meta, "adapter.safetensors").unwrap();
        assert_eq!(plan.pairs.len(), 1);
        assert_eq!(plan.diffs.len(), 1);
        assert_eq!(plan.replacements.len(), 1);
        assert_eq!(plan.replacements[0].0, "blocks.0.attn.to_q.weight");
    }

    #[test]
    fn hybrid_plan_fifty_gates_without_lora_pairs() {
        let keys: Vec<String> = (0..50)
            .map(|i| format!("transformer_blocks.{i}.attn.to_gate_compress.weight.set_weight"))
            .collect();
        let mut meta = HashMap::new();
        meta.insert("format".into(), "fastvideo-lora-v2".into());
        meta.insert("set_weight_tensors".into(), "50".into());
        meta.insert("low_rank_tensors".into(), "0".into());
        let plan = plan_from_keys(&keys, &meta, "vsa-datafree/adapter_model.safetensors").unwrap();
        assert_eq!(plan.pairs.len(), 0);
        assert_eq!(plan.replacements.len(), 50);
        assert!(plan
            .replacements
            .iter()
            .all(|(p, _)| p.ends_with("to_gate_compress.weight")));
    }

    #[test]
    fn peft_plan_rejects_set_weight() {
        let keys = vec![
            "transformer_blocks.0.attn.to_q.lora_A.default.weight".into(),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight".into(),
            "transformer_blocks.0.attn.to_q.set_weight".into(),
        ];
        let err = plan_from_keys(&keys, &HashMap::new(), "peft.safetensors").unwrap_err();
        assert!(
            err.contains("set_weight") || err.contains("replacement"),
            "{err}"
        );
    }

    #[test]
    fn peft_plan_pairs_and_rejects_unknown_modules() {
        let keys = vec![
            "transformer_blocks.0.attn.to_q.lora_A.default.weight".into(),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight".into(),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight".into(),
            "diffusion_model.transformer_blocks.0.ff.net.2.lora_B.weight".into(),
        ];
        let plan = plan_from_keys(&keys, &HashMap::new(), "a.safetensors").unwrap();
        assert_eq!(plan.format, LoraFormat::Peft);
        assert_eq!(plan.pairs.len(), 2);
        assert_eq!(plan.pairs[0].0, "transformer_blocks.0.attn.to_q");
        assert_eq!(plan.pairs[1].0, "transformer_blocks.0.ff.net.2");

        let bad = vec![
            "proj_in.lora_A.weight".into(),
            "proj_in.lora_B.weight".into(),
        ];
        assert!(plan_from_keys(&bad, &HashMap::new(), "a").is_err());
    }

    #[test]
    fn diff_accumulates_in_place() {
        let mut w = vec![1.0, 2.0];
        add_diff(&mut w, &[0.5, -0.25], 2.0).unwrap();
        assert_eq!(w, vec![2.0, 1.5]);
    }

    #[test]
    fn weights_manifest_lists_fastwan21() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/gpu/weights-manifest.tsv");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("fastwan21-1.3b\tFastVideo/FastWan2.1-T2V-1.3B-Diffusers"));
    }
}
