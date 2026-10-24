//! Technique registry: name → factory from a `[techniques.<name>]` table
//! (sol-engine `techniques/registry.py:17-64`, `register_technique` /
//! `build_technique`), and the model specs (`register_model_spec`, :48-56).

use std::collections::BTreeMap;

use super::kernels::KernelOp;
use super::methods::*;
use super::schedule::{parse_sparsity, Schedule, SparseRoute, StepSet, VsaSchedule};
use super::settings::Family;
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

/// `dit` is a `FASTVIDEO_DIT_OFFLOAD` spelling; `rt::offload::DitOffload::parse`
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

/// The kernel ops every CUDA family reaches through the shared runtime's
/// dense attention and 3-D convolution (`rt::attn`, `rt::conv`).
const SHARED_KERNELS: &[KernelOp] = &[KernelOp::DenseAttention, KernelOp::Conv3d];

/// The block-sparse kernels (VSA and Sol-Attn) besides the shared ones.
const SPARSE_KERNELS: &[KernelOp] = &[
    KernelOp::DenseAttention,
    KernelOp::SolAttention,
    KernelOp::VsaAttention,
    KernelOp::Conv3d,
];

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
        profiles: true,
        families: &[Family::H3, Family::Runtime, Family::TextEncoder],
        extra_settings: &[],
        kernels: SPARSE_KERNELS,
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
        profiles: true,
        families: &[Family::Ltx2, Family::Runtime, Family::TextEncoder],
        extra_settings: &[],
        kernels: SPARSE_KERNELS,
    }
}

/// Wan 2.1 / 2.2 (`cudarc::wan`), the minimal spec
/// (`docs/adr/0010-settings-layer.md` "`model_spec`"): what the runtime
/// exposes and which techniques-layer settings and kernels it reads, with no
/// profile adapter, so `[pipeline].model = "wan"` stays refused. Its seams
/// are driven by its own settings today: attention backend (`FASTVIDEO_SDPA`,
/// `FASTVIDEO_VSA`, `FASTVIDEO_WAN_SOL_ATTN`, `FASTVIDEO_WAN_PISA`), step
/// cache (`FASTVIDEO_TEACACHE`, `FASTVIDEO_WAN_SOL_CACHE`), FFN precision
/// (`FASTVIDEO_WAN_QUANT`, `FASTVIDEO_FP8`), the TAEHV decoder
/// (`FASTVIDEO_WAN_VAE`, `FASTVIDEO_TAEHV_WEIGHTS`) and streamed blocks
/// (`FASTVIDEO_DIT_OFFLOAD=streamed`). `layers` is the widest Wan DiT (14B /
/// A14B, 40 blocks; 1.3B and TI2V-5B have 30).
pub fn wan_spec() -> ModelSpec {
    ModelSpec {
        name: "wan",
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
        layers: 40,
        profiles: false,
        families: &[Family::Wan, Family::Runtime],
        extra_settings: &[],
        kernels: SPARSE_KERNELS,
    }
}

/// A family with no technique seam (`capabilities` empty: nothing composes
/// for it) whose runtime reads the shared runtime's settings and its own.
fn small_spec(name: &'static str, families: &'static [Family]) -> ModelSpec {
    ModelSpec {
        name,
        capabilities: Vec::new(),
        layers: 0,
        profiles: false,
        families,
        extra_settings: &[],
        kernels: SHARED_KERNELS,
    }
}

/// The small families: every `fastvideo-core` family but Wan, LTX-2 and H3,
/// by `VideoFamily::name`, in the order of `fastvideo_core::family::FAMILIES`.
pub const SMALL_FAMILIES: &[&str] = &[
    "hunyuan15",
    "kandinsky5",
    "cosmos",
    "longcat",
    "lingbot",
    "gen3c",
    "matrixgame",
    "dreamx_world",
    "lingbotworld",
    "gamecraft",
    "hyworld",
    "zimage",
    "sd35",
    "flux",
    "flux2",
    "glm_image",
    "stable_audio",
    "mmaudio",
];

/// The spec of any family `fastvideo-core` registers (its
/// `VideoFamily::name`): H3, LTX-2 (`ltx2` / `ltx`), Wan and the small
/// families. Describes what the family reads; whether a profile may name it
/// is [`model_spec`]'s answer.
pub fn family_spec(name: &str) -> Option<ModelSpec> {
    use Family as F;
    const RT: F = F::Runtime;
    Some(match name {
        "h3" => h3_spec(),
        "ltx2" | "ltx" => ltx2_spec(),
        "wan" => wan_spec(),
        "hunyuan15" => small_spec("hunyuan15", &[F::Hunyuan15, RT, F::TextEncoder]),
        "cosmos" => small_spec("cosmos", &[F::Cosmos, F::Cosmos3, RT, F::TextEncoder]),
        "lingbot" => small_spec("lingbot", &[F::Lingbot, RT]),
        "lingbotworld" => small_spec("lingbotworld", &[F::Lingbot, RT]),
        "flux" => small_spec("flux", &[F::Flux, RT, F::TextEncoder]),
        "flux2" => small_spec("flux2", &[F::Flux2, RT, F::TextEncoder]),
        "mmaudio" => small_spec("mmaudio", &[F::MmAudio, RT]),
        "kandinsky5" => small_spec("kandinsky5", &[RT, F::TextEncoder]),
        "longcat" => small_spec("longcat", &[RT, F::TextEncoder]),
        "gen3c" => small_spec("gen3c", &[RT, F::TextEncoder]),
        "matrixgame" => small_spec("matrixgame", &[RT]),
        "dreamx_world" => small_spec("dreamx_world", &[RT, F::TextEncoder]),
        "gamecraft" => small_spec("gamecraft", &[RT, F::TextEncoder]),
        "hyworld" => small_spec("hyworld", &[RT, F::TextEncoder]),
        "zimage" => small_spec("zimage", &[RT, F::TextEncoder]),
        "sd35" => small_spec("sd35", &[RT, F::TextEncoder]),
        "glm_image" => small_spec("glm_image", &[RT, F::TextEncoder]),
        "stable_audio" => small_spec("stable_audio", &[RT, F::TextEncoder]),
        _ => return None,
    })
}

/// `get_model_spec` (:67-78) by pipeline name: the models a technique profile
/// may name (`[pipeline].model`), H3 and LTX-2. Wan and the small families
/// have a [`family_spec`] but no profile adapter (the minimal spec,
/// `docs/adr/0010-settings-layer.md`).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_h3_and_ltx2_profiles_load() {
        for name in ["h3", "ltx2", "ltx", "wan"]
            .into_iter()
            .chain(SMALL_FAMILIES.iter().copied())
        {
            let spec = family_spec(name).unwrap_or_else(|| panic!("{name}: no spec"));
            assert_eq!(spec.profiles, model_spec(name).is_some(), "{name}");
        }
        // The minimal Wan spec adds no profile loadability: the error a Wan
        // profile got before the spec existed.
        let e = crate::techniques::Profile::parse("[id]\nname='x'\n[pipeline]\nmodel='wan'\n")
            .unwrap_err();
        assert_eq!(e, "[pipeline].model = \"wan\": expected h3 | ltx2");
    }

    #[test]
    fn spec_names_are_the_family_names() {
        for name in ["h3", "ltx2", "wan"]
            .into_iter()
            .chain(SMALL_FAMILIES.iter().copied())
        {
            assert_eq!(family_spec(name).map(|s| s.name), Some(name));
        }
        assert_eq!(family_spec("ltx").map(|s| s.name), Some("ltx2"));
        assert!(family_spec("sora").is_none());
    }

    /// A small family exposes no technique seam: every technique that needs
    /// one fails its capability check there.
    #[test]
    fn small_families_compose_nothing() {
        use crate::techniques::methods::TeaCache;
        for name in SMALL_FAMILIES {
            let spec = family_spec(name).unwrap_or_else(|| panic!("{name}"));
            assert!(spec.capabilities.is_empty(), "{name}");
            let items: Vec<Box<dyn Technique>> = vec![Box::new(TeaCache::rtx())];
            assert!(
                crate::techniques::compose(items, &spec, crate::techniques::HORIZON).is_err(),
                "{name}"
            );
        }
    }

    /// Wan reads its own layered rows and the runtime's, never H3's or
    /// LTX-2's own (the precision recipes map to those names).
    #[test]
    fn wan_reads_its_own_and_the_runtime_settings() {
        let wan = wan_spec();
        assert!(wan.reads("FASTVIDEO_WAN_FUSE"));
        assert!(wan.reads("FASTVIDEO_BF16_ACT"));
        assert!(wan.reads("FASTVIDEO_DIT_OFFLOAD"));
        assert!(wan.dispatches(KernelOp::VsaAttention));
        // `Env` rows are not the techniques layer, whatever their family.
        assert!(!wan.reads("FASTVIDEO_WAN_VAE"));
        assert!(!wan.reads("FASTVIDEO_H3_QUANT"));
        assert!(!wan.reads("FASTVIDEO_LTX_OFFLOAD"));
        let names = wan.settings();
        assert!(names.windows(2).all(|w| w[0] < w[1]));
        assert!(names.iter().all(|n| wan.reads(n)));
    }
}
