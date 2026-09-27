//! Technique registry: name → factory from a `[techniques.<name>]` table
//! (sol-engine `techniques/registry.py:17-64`, `register_technique` /
//! `build_technique`), and the model specs (`register_model_spec`, :48-56).

use std::collections::BTreeMap;

use super::methods::*;
use super::schedule::{parse_sparsity, Schedule, SparseRoute, StepSet, VsaSchedule};
use super::technique::{Capability, ModelSpec, Technique};

type Factory = fn(&str, &toml::Table) -> Result<Box<dyn Technique>, String>;

/// `(name, one-line doc, factory)`.
pub static TECHNIQUES: &[(&str, &str, Factory)] = &[
    (
        "dense_attention",
        "dense attention, stated explicitly",
        dense_attention,
    ),
    (
        "sol_attn",
        "Sol-Attn sparse route + exact KV sink",
        sol_attn,
    ),
    ("vsa", "FastVideo VSA", vsa),
    (
        "fp8_attention",
        "SageAttention-style FP8 Q K^T (opt-in, lossy)",
        fp8_attention,
    ),
    ("pisa", "PISA piecewise sparse attention", pisa),
    ("teacache", "TeaCache step-output reuse", teacache),
    ("bf16_linears", "bf16 linears", |n, t| {
        linear(n, t, LinearRecipe::Bf16)
    }),
    ("mxfp8", "Sol-H3 MXFP8 linears (blocks 2..=46)", |n, t| {
        linear(n, t, LinearRecipe::Mxfp8)
    }),
    ("w8a8", "FastVideo tensorwise W8A8 linears", |n, t| {
        linear(n, t, LinearRecipe::W8A8)
    }),
    ("fp8", "FP8 linears (FASTVIDEO_FP8)", |n, t| {
        linear(n, t, LinearRecipe::Fp8)
    }),
    ("nvfp4", "NVFP4 W4A4 linears", |n, t| {
        linear(n, t, LinearRecipe::Nvfp4)
    }),
    ("bf16_activations", "bf16 activations end to end", |n, t| {
        activations(n, t, true)
    }),
    ("f32_activations", "f32 activations", |n, t| {
        activations(n, t, false)
    }),
    ("taeh3", "TAEH3 tiny video decoder", |n, t| {
        tiny(n, t, TinyDecoderKind::Taeh3)
    }),
    ("taehv", "TAEHV tiny video decoder (LTX-2)", |n, t| {
        tiny(n, t, TinyDecoderKind::Taehv)
    }),
    ("offload", "DiT block residency / streaming", offload),
    ("kernel_fusion", "DiT block fusion switches", kernel_fusion),
];

pub fn registered() -> Vec<&'static str> {
    TECHNIQUES.iter().map(|(n, _, _)| *n).collect()
}

/// `build_technique` (:59-64): unknown names list the registered ones.
pub fn build(name: &str, table: &toml::Table) -> Result<Box<dyn Technique>, String> {
    match TECHNIQUES.iter().find(|(n, _, _)| *n == name) {
        Some((_, _, f)) => f(name, table),
        None => Err(format!(
            "unknown technique {name:?}; registered: {:?}",
            registered()
        )),
    }
}

fn dense_attention(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let p = Params::new(name, t);
    let enabled = p.enabled()?;
    p.finish()?;
    Ok(Box::new(DenseAttention { enabled }))
}

fn sol_attn(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let preset = p.string("preset")?;
    let mut sol = match preset.as_deref() {
        Some(n) => SolAttn::preset(n).ok_or_else(|| {
            format!("techniques.{name}.preset = {n:?}: expected rtx|engine|spark|ltx25_stage2")
        })?,
        // No preset: sol-engine's adapter defaults (`adapter.py:457-460`:
        // 10 dense steps; tau 1.0) with the RTX text sink.
        None => SolAttn::rtx(),
    };
    if preset.is_none() {
        sol.preset = None;
    }
    sol.enabled = enabled;
    if let Some(v) = p.tau("tau")? {
        sol.route.tau = v;
    }
    if let Some(v) = p.index_set("dense_steps")? {
        sol.route.dense_steps = v;
    }
    if let Some(v) = p.index_set("dense_layers")? {
        sol.route.dense_layers = v;
    }
    if let Some(v) = p.string("sink")? {
        sol.sink = SinkMode::parse(&v)?;
    }
    if let Some(v) = p.string("thresh_type")? {
        if v != "diag" {
            return Err(format!(
                "techniques.{name}.thresh_type = {v:?}: only \"diag\" is implemented"
            ));
        }
        sol.thresh_type = v;
    }
    if let Some(v) = p.bool("correctness_gate")? {
        sol.correctness_gate = v;
    }
    match p.string("dense_backend")?.as_deref() {
        None | Some("dense") => {}
        Some("vsa") => sol.dense_vsa = true,
        Some(other) => {
            return Err(format!(
                "techniques.{name}.dense_backend = {other:?}: expected \"dense\" or \"vsa\""
            ))
        }
    }
    if p.bool("force_dense")? == Some(true) {
        // SOL_ATTN_FORCE_DENSE: every call dense.
        sol.route = SparseRoute {
            dense_steps: StepSet::from(0),
            ..sol.route
        };
    }
    p.finish()?;
    Ok(Box::new(sol))
}

fn vsa(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let (sparsity, per_step) = match p.get("sparsity") {
        Some(v) => parse_sparsity(v).map_err(|e| format!("techniques.{name}.{e}"))?,
        None => (None, BTreeMap::new()),
    };
    let group = p.usize("group")?.map(|g| g.max(1));
    let dense_steps = p.index_set("dense_steps")?.unwrap_or_default();
    let dense_layers = p.index_set("dense_layers")?.unwrap_or_default();
    let dense_sparsity = p.f64("dense_sparsity")?.unwrap_or(0.0);
    if !(0.0..1.0).contains(&dense_sparsity) {
        return Err(format!(
            "techniques.{name}.dense_sparsity = {dense_sparsity}: need [0, 1)"
        ));
    }
    p.finish()?;
    Ok(Box::new(Vsa {
        enabled,
        sparsity,
        group,
        schedule: VsaSchedule {
            per_step,
            dense_steps,
            dense_layers,
            dense_sparsity,
        },
    }))
}

fn fp8_attention(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let ops = match p.get("ops") {
        None => Fp8AttentionOps::parse("all")?,
        Some(toml::Value::String(s)) => {
            Fp8AttentionOps::parse(s).map_err(|e| format!("techniques.{name}.ops: {e}"))?
        }
        Some(toml::Value::Array(a)) => {
            let list: Vec<&str> = a.iter().filter_map(|v| v.as_str()).collect();
            if list.len() != a.len() {
                return Err(format!("techniques.{name}.ops must be strings"));
            }
            Fp8AttentionOps::parse(&list.join(","))
                .map_err(|e| format!("techniques.{name}.ops: {e}"))?
        }
        Some(other) => {
            return Err(format!(
                "techniques.{name}.ops = {other}: expected \"all\", \"dense\", \"vsa\" or a list"
            ))
        }
    };
    if !ops.any() {
        return Err(format!("techniques.{name}.ops: nothing selected"));
    }
    p.finish()?;
    Ok(Box::new(Fp8Attention { enabled, ops }))
}

fn pisa(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let sparsity = p.f64("sparsity")?;
    let dense_layers = p.index_set("dense_layers")?;
    p.finish()?;
    Ok(Box::new(Pisa {
        enabled,
        sparsity,
        dense_layers,
    }))
}

fn teacache(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let mut tc = TeaCache::rtx();
    tc.enabled = p.enabled()?;
    if let Some(v) = p.f64("threshold")? {
        if v <= 0.0 {
            return Err(format!("techniques.{name}.threshold must be positive"));
        }
        tc.threshold = v;
    }
    if let Some(v) = p.usize("retain_steps")? {
        tc.retain_steps = v;
    }
    if let Some(v) = p.usize("cooldown_steps")? {
        tc.cooldown_steps = v;
    }
    tc.num_forwards = p.usize("num_forwards")?;
    if let Some(v) = p.f64_list("coefficients")? {
        if v.is_empty() {
            return Err(format!("techniques.{name}.coefficients must not be empty"));
        }
        tc.coefficients = v;
    }
    p.finish()?;
    Ok(Box::new(tc))
}

fn linear(name: &str, t: &toml::Table, recipe: LinearRecipe) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let nvfp4_rule = if recipe == LinearRecipe::Nvfp4 {
        p.string("rule")?
    } else {
        None
    };
    p.finish()?;
    Ok(Box::new(LinearPrecision {
        enabled,
        recipe,
        nvfp4_rule,
    }))
}

fn activations(name: &str, t: &toml::Table, bf16: bool) -> Result<Box<dyn Technique>, String> {
    let p = Params::new(name, t);
    let enabled = p.enabled()?;
    p.finish()?;
    Ok(Box::new(ActivationPrecision { enabled, bf16 }))
}

fn tiny(name: &str, t: &toml::Table, kind: TinyDecoderKind) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let weights = p.string("weights")?;
    p.finish()?;
    Ok(Box::new(TinyDecoder {
        enabled,
        kind,
        weights,
    }))
}

/// `dit` is a `FASTVIDEO_DIT_OFFLOAD` spelling; `wan::offload::DitOffload::parse`
/// validates it at load (it owns the list of placements).
fn offload(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let dit = p.string("dit")?;
    let lookahead = p.usize("lookahead")?;
    let placement = p.string("placement")?;
    p.finish()?;
    if dit.is_none() && placement.is_none() {
        return Err(format!("techniques.{name}: set dit and/or placement"));
    }
    Ok(Box::new(Offload {
        enabled,
        dit,
        lookahead,
        placement,
    }))
}

fn kernel_fusion(name: &str, t: &toml::Table) -> Result<Box<dyn Technique>, String> {
    let mut p = Params::new(name, t);
    let enabled = p.enabled()?;
    let mut flags = BTreeMap::new();
    for (key, _) in FUSION_FLAGS {
        if let Some(v) = p.bool(key)? {
            flags.insert(key, v);
        }
    }
    p.finish()?;
    Ok(Box::new(KernelFusion { enabled, flags }))
}

// ---------------------------------------------------------------------------
// model specs
// ---------------------------------------------------------------------------

/// MiniMax-H3 / FastH3 (`cudarc::h3`): 50 blocks, swappable attention
/// (dense / VSA / Sol), whole-stack step cache, tiny decoder, streaming.
pub fn h3_spec() -> ModelSpec {
    ModelSpec {
        name: "h3",
        capabilities: vec![
            Capability::Blocks,
            Capability::SwappableAttention,
            Capability::HasDenoiseSteps,
            Capability::HasTransformerBlocks,
            Capability::HasAttentionLayers,
            Capability::HasAttentionBackendSwitch,
            Capability::HasFfnLinearModules,
            Capability::HasSpatiotemporalTokenLayout,
            Capability::SupportsStepCache,
            Capability::SwappableVideoDecoder,
            Capability::SupportsLayerOffload,
        ],
        layers: crate::h3::sol::LAYERS_PER_FORWARD,
    }
}

/// LTX-2 / LTX-2.5 (`cudarc::ltx2`): 48 blocks per stage.
pub fn ltx2_spec() -> ModelSpec {
    ModelSpec {
        name: "ltx2",
        capabilities: vec![
            Capability::Blocks,
            Capability::SwappableAttention,
            Capability::HasDenoiseSteps,
            Capability::HasTransformerBlocks,
            Capability::HasAttentionLayers,
            Capability::HasAttentionBackendSwitch,
            Capability::HasFfnLinearModules,
            Capability::HasSpatiotemporalTokenLayout,
            Capability::SupportsNvfp4Linear,
            Capability::SwappableVideoDecoder,
            Capability::SupportsLayerOffload,
        ],
        layers: 48,
    }
}

/// `get_model_spec` (:67-78) by pipeline name.
pub fn model_spec(model: &str) -> Option<ModelSpec> {
    match model {
        "h3" => Some(h3_spec()),
        "ltx2" | "ltx" => Some(ltx2_spec()),
        _ => None,
    }
}

/// An always-true schedule, for code that builds techniques directly.
pub fn on() -> Schedule<bool> {
    Schedule::Const(true)
}
