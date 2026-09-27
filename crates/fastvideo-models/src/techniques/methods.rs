//! The registered techniques: one parameter struct each, parsed from a
//! `[techniques.<name>]` table by [`super::registry`].
//!
//! | name | kind | writes | sol-engine counterpart |
//! |---|---|---|---|
//! | `dense_attention` | build | attention_backend | the dense `fa` backend (`sparse_attention.py:118`) |
//! | `sol_attn` | build | attention_backend | `SparseAttention` (`transforms/sparse_attention.py:23-36`) with the RTX adapter's route (`RTX5090/adapter.py:452-461`) |
//! | `vsa` | build | attention_backend | (FastVideo VSA; no sol-engine twin) |
//! | `pisa` | build | attention_backend | `SparseAttention` route_mode `score` (PISA) |
//! | `teacache` | on_step | step_output, residual_cache | `TeaCache` (`methods/teacache.py:60-79`) |
//! | `bf16_linears` / `mxfp8` / `w8a8` / `fp8` / `nvfp4` | load | ffn_precision | `NVFP4FFN` (`transforms/nvfp4_ffn.py:24-36`) |
//! | `bf16_activations` / `f32_activations` | load | activation_precision | — |
//! | `taeh3` / `taehv` | load | video_decoder | super_acceleration stage 1 / ltx2.5-refiner decode |
//! | `offload` | load | residency | SGLang layerwise offload (`gpu_infer.py:153-156`) |
//! | `kernel_fusion` | build | kernel_fusion (shared) | `KWLFusions` (`transforms/kwl_fusions.py:127-134`) |

use std::any::Any;
use std::collections::BTreeMap;

use super::schedule::{parse_index_set, parse_tau, Schedule, SparseRoute, StepSet, VsaSchedule};
use super::technique::{Capability, Kind, ModelSpec, Phase, Seam, Technique, TransformPhase};

/// Where a Sol layer's exact KV sink sits (and whether Q/K/V are permuted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkMode {
    /// No sink.
    None,
    /// Every row before the target video (`sink_mode="prefix"`, the Sol-H3
    /// engine).
    Prefix,
    /// The contiguous text rows, dense text query rows (the RTX cell,
    /// `adapter.py` `_sol_varlen`).
    Text,
    /// Q/K/V permuted to `[visual | text+audio]`, suffix sink (the Spark
    /// draft, `stage1_ops/sol.py`).
    Suffix,
}

impl SinkMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "" => Ok(Self::None),
            "prefix" => Ok(Self::Prefix),
            "text" => Ok(Self::Text),
            "suffix" | "visual_suffix" => Ok(Self::Suffix),
            other => Err(format!(
                "sink = {other:?}: expected none|prefix|text|suffix"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Prefix => "prefix",
            Self::Text => "text",
            Self::Suffix => "suffix",
        }
    }
}

// ---------------------------------------------------------------------------
// attention backends (exclusive seam)
// ---------------------------------------------------------------------------

/// Plain dense attention, stated explicitly (a profile that turns a recipe's
/// default sparse route off, as sol-engine's `rtx5090_dense.toml` does).
#[derive(Clone, Debug, PartialEq)]
pub struct DenseAttention {
    pub enabled: Schedule<bool>,
}

/// Sol-Attn: a sparse route over `(step, layer)` plus an exact KV sink.
#[derive(Clone, Debug, PartialEq)]
pub struct SolAttn {
    pub enabled: Schedule<bool>,
    pub route: SparseRoute,
    pub sink: SinkMode,
    /// `SOL_ATTN_THRESH_TYPE`; only `diag` is implemented.
    pub thresh_type: String,
    /// `SOL_ATTN_CORRECTNESS_GATE`: sol-engine's sampled dense-vs-Sol gate.
    /// Not implemented here (recorded, and a warning when on).
    pub correctness_gate: bool,
    /// What a dense `(step, layer)` of the route runs: dense attention
    /// (`false`, every published route), or the recipe's VSA with its gate
    /// (`dense_backend = "vsa"`, for the VSA-distilled FastH3 checkpoints,
    /// whose trained function is VSA, not dense).
    pub dense_vsa: bool,
    /// The named route this came from, if any (`rtx`, `engine`, `spark`).
    pub preset: Option<String>,
}

impl SolAttn {
    /// RTX 5090 cell: forwards 0-9 dense, blocks 0-1 dense, tau 1.0, text sink
    /// (`rtx5090_sol.toml` [env], `adapter.py:452-461`).
    pub fn rtx() -> Self {
        Self {
            enabled: Schedule::Const(true),
            route: SparseRoute {
                dense_steps: StepSet::first(10),
                dense_layers: StepSet::first(2),
                tau: Schedule::Const(1.0),
            },
            sink: SinkMode::Text,
            thresh_type: "diag".into(),
            correctness_gate: false,
            dense_vsa: false,
            preset: Some("rtx".into()),
        }
    }

    /// Sol-H3 engine: forward 0 dense, blocks 0-1 dense, tau 1.0, prefix sink.
    pub fn engine() -> Self {
        Self {
            route: SparseRoute {
                dense_steps: StepSet::first(1),
                dense_layers: StepSet::first(2),
                tau: Schedule::Const(1.0),
            },
            sink: SinkMode::Prefix,
            preset: Some("engine".into()),
            ..Self::rtx()
        }
    }

    /// Spark Ref2VA draft: update 0 dense, block 0 dense, taus 1 / 1.25 / 1.5
    /// on updates 1-3, every later update dense, suffix sink.
    pub fn spark() -> Self {
        Self {
            route: SparseRoute {
                dense_steps: StepSet::parse("0,4-").expect("static"),
                dense_layers: StepSet::first(1),
                tau: Schedule::PerStep {
                    values: [(1, 1.0), (2, 1.25), (3, 1.5)].into(),
                    default: 1.0,
                },
            },
            sink: SinkMode::Suffix,
            preset: Some("spark".into()),
            ..Self::rtx()
        }
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "rtx" => Some(Self::rtx()),
            "engine" | "sol" => Some(Self::engine()),
            "spark" => Some(Self::spark()),
            "ltx25_stage2" => Some(Self::ltx25_stage2()),
            _ => None,
        }
    }

    /// LTX-2.5 stage 2 (`models/ltx25/RTX5090/attention.py`): the three
    /// refine forwards, video self-attention layer 0 dense, layers 1..=47 at
    /// tau 1.0 / 1.25 / 1.5; no sink.
    pub fn ltx25_stage2() -> Self {
        Self {
            route: SparseRoute {
                dense_steps: StepSet::empty(),
                dense_layers: StepSet::first(1),
                tau: Schedule::PerStep {
                    values: [(0, 1.0), (1, 1.25), (2, 1.5)].into(),
                    default: 1.0,
                },
            },
            sink: SinkMode::None,
            preset: Some("ltx25_stage2".into()),
            ..Self::rtx()
        }
    }
}

/// FastVideo VSA (the FastH3 distilled checkpoints' attention).
#[derive(Clone, Debug, PartialEq)]
pub struct Vsa {
    pub enabled: Schedule<bool>,
    /// Top-k sparsity; `None` keeps the recipe's.
    pub sparsity: Option<f64>,
    /// Query-tile group size; `None` keeps the model default.
    pub group: Option<usize>,
    /// Per-step / per-layer sparsity (`sparsity = { 0 = 0.8 }`,
    /// `dense_steps`, `dense_layers`, `dense_sparsity`); uniform by default.
    pub schedule: VsaSchedule,
}

impl Vsa {
    /// VSA at one sparsity everywhere (`None`: the recipe's).
    pub fn uniform(sparsity: Option<f64>, group: Option<usize>) -> Self {
        Self {
            enabled: Schedule::Const(true),
            sparsity,
            group,
            schedule: VsaSchedule::default(),
        }
    }
}

/// Where FP8 attention replaces the bf16 kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fp8AttentionOps {
    /// Dense attention calls (the dense recipes, dense Sol steps / layers).
    pub dense: bool,
    /// The VSA fine stage.
    pub vsa: bool,
}

impl Fp8AttentionOps {
    pub fn any(self) -> bool {
        self.dense || self.vsa
    }

    /// `FASTVIDEO_ATTN_FP8`: `0` / `off`, `1` / `on` / `all`, `dense`, `vsa`,
    /// or a comma list.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut out = Self::default();
        for tok in s.split(',').map(|t| t.trim().to_ascii_lowercase()) {
            match tok.as_str() {
                "" | "0" | "off" | "false" => {}
                "1" | "on" | "all" | "true" => {
                    out.dense = true;
                    out.vsa = true;
                }
                "dense" => out.dense = true,
                "vsa" => out.vsa = true,
                other => {
                    return Err(format!(
                        "FP8 attention op {other:?}: expected off|all|dense|vsa"
                    ))
                }
            }
        }
        Ok(out)
    }

    pub fn describe(self) -> &'static str {
        match (self.dense, self.vsa) {
            (true, true) => "dense+vsa",
            (true, false) => "dense",
            (false, true) => "vsa",
            (false, false) => "off",
        }
    }
}

/// SageAttention-style FP8 attention (opt-in, lossy): Q and smoothed K
/// quantized to E4M3 per 64-row block, `Q K^T` on FP8 tensor cores with f32
/// accumulation, `P V` in bf16 (`cudarc::wan::attn_fp8`).
#[derive(Clone, Debug, PartialEq)]
pub struct Fp8Attention {
    pub enabled: Schedule<bool>,
    pub ops: Fp8AttentionOps,
}

/// PISA piecewise sparse attention (LTX-2 stage 2).
#[derive(Clone, Debug, PartialEq)]
pub struct Pisa {
    pub enabled: Schedule<bool>,
    pub sparsity: Option<f64>,
    pub dense_layers: Option<StepSet>,
}

// ---------------------------------------------------------------------------
// step output (exclusive)
// ---------------------------------------------------------------------------

/// TeaCache: skip the block stack while the accumulated rescaled rel-L1 of
/// the modulated input stays under `threshold`, reusing the residual.
#[derive(Clone, Debug, PartialEq)]
pub struct TeaCache {
    pub enabled: Schedule<bool>,
    pub threshold: f64,
    pub retain_steps: usize,
    pub cooldown_steps: usize,
    /// `None`: the recipe's forward count.
    pub num_forwards: Option<usize>,
    /// Horner polynomial, highest degree first (`teacache.py:32-42`).
    pub coefficients: Vec<f64>,
}

impl TeaCache {
    /// `run_minimax_h3_gpu.sh` defaults (threshold 0.10, retain 5, cooldown 1,
    /// identity polynomial).
    pub fn rtx() -> Self {
        Self {
            enabled: Schedule::Const(true),
            threshold: 0.10,
            retain_steps: 5,
            cooldown_steps: 1,
            num_forwards: None,
            coefficients: vec![1.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------------
// precision
// ---------------------------------------------------------------------------

/// Linear-layer precision recipe (exclusive `ffn_precision` seam).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearRecipe {
    /// bf16 weights and GEMMs.
    Bf16,
    /// Sol-H3 MXFP8 (blocks 2..=46).
    Mxfp8,
    /// FastVideo tensorwise W8A8 (Sol-H3-Spark).
    W8A8,
    /// FP8 checkpoint / FP8 linears (`FASTVIDEO_FP8`).
    Fp8,
    /// NVFP4 W4A4 (`FASTVIDEO_NVFP4`).
    Nvfp4,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearPrecision {
    pub enabled: Schedule<bool>,
    pub recipe: LinearRecipe,
    /// NVFP4 scale rule (`static_6`, `static_4`, `mse`).
    pub nvfp4_rule: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActivationPrecision {
    pub enabled: Schedule<bool>,
    pub bf16: bool,
}

// ---------------------------------------------------------------------------
// load-time: decoder, residency; build-time: fusion
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyDecoderKind {
    /// madebyollin TAEH3 for MiniMax-H3 latents.
    Taeh3,
    /// madebyollin TAEHV family (`taeltx2_3_wide` for LTX-2).
    Taehv,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyDecoder {
    pub enabled: Schedule<bool>,
    pub kind: TinyDecoderKind,
    /// Weights file or directory; `None` leaves it to the command line / env.
    pub weights: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Offload {
    pub enabled: Schedule<bool>,
    /// `auto` | `resident` | `streamed` (`FASTVIDEO_DIT_OFFLOAD`); `None`
    /// leaves the DiT policy alone.
    pub dit: Option<String>,
    pub lookahead: Option<usize>,
    /// LTX-2's whole-pipeline placement, the reference's `--offload`
    /// (`none` | `cpu`, `FASTVIDEO_LTX_OFFLOAD`).
    pub placement: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelFusion {
    pub enabled: Schedule<bool>,
    pub flags: BTreeMap<&'static str, bool>,
}

/// `[techniques.kernel_fusion]` keys and the settings they drive.
pub const FUSION_FLAGS: [(&str, &str); 3] = [
    ("h3", "FASTVIDEO_H3_FUSE"),
    ("ltx", "FASTVIDEO_LTX_FUSE"),
    ("split_rows", "FASTVIDEO_SPLIT_ROWS"),
];

// ---------------------------------------------------------------------------
// Technique impls
// ---------------------------------------------------------------------------

const SWAPPABLE_ATTENTION: &[Capability] = &[Capability::SwappableAttention];

macro_rules! any {
    () => {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn Technique> {
            Box::new(self.clone())
        }
    };
}

impl Technique for DenseAttention {
    fn name(&self) -> &'static str {
        "dense_attention"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::AttentionBackend]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    any!();
}

impl Technique for SolAttn {
    fn name(&self) -> &'static str {
        "sol_attn"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::AttentionBackend]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        SWAPPABLE_ATTENTION
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn describe(&self) -> String {
        format!(
            "sol_attn{} ({}, sink {}, thresh {}{})",
            self.preset
                .as_deref()
                .map(|p| format!(" [{p}]"))
                .unwrap_or_default(),
            self.route.describe(),
            self.sink.as_str(),
            self.thresh_type,
            if self.dense_vsa { ", dense calls on VSA" } else { "" }
        )
    }
    any!();
}

impl Technique for Vsa {
    fn name(&self) -> &'static str {
        "vsa"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::AttentionBackend]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        SWAPPABLE_ATTENTION
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn describe(&self) -> String {
        let schedule = if self.schedule.is_uniform() {
            String::new()
        } else {
            let mut parts: Vec<String> = self
                .schedule
                .per_step
                .iter()
                .map(|(s, v)| format!("step {s}: {v}"))
                .collect();
            if !self.schedule.dense_steps.is_empty() {
                parts.push(format!(
                    "steps {} at {}",
                    self.schedule.dense_steps, self.schedule.dense_sparsity
                ));
            }
            if !self.schedule.dense_layers.is_empty() {
                parts.push(format!(
                    "layers {} at {}",
                    self.schedule.dense_layers, self.schedule.dense_sparsity
                ));
            }
            format!("; {}", parts.join(", "))
        };
        format!(
            "vsa (sparsity {}, group {}{schedule})",
            self.sparsity.map_or("recipe".into(), |s| s.to_string()),
            self.group.map_or("default".into(), |g| g.to_string())
        )
    }
    any!();
}

impl Technique for Fp8Attention {
    fn name(&self) -> &'static str {
        "fp8_attention"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::Attention]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        SWAPPABLE_ATTENTION
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn describe(&self) -> String {
        format!("fp8_attention ({}; Q K^T e4m3, P V bf16)", self.ops.describe())
    }
    any!();
}

impl Technique for Pisa {
    fn name(&self) -> &'static str {
        "pisa"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::AttentionBackend]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        SWAPPABLE_ATTENTION
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    any!();
}

impl Technique for TeaCache {
    fn name(&self) -> &'static str {
        "teacache"
    }
    fn kind(&self) -> Kind {
        Kind::Runtime(Phase::OnStep)
    }
    fn reads(&self) -> &'static [Seam] {
        &[Seam::StepOutput, Seam::ResidualCache]
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::StepOutput, Seam::ResidualCache]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        &[Capability::SupportsStepCache]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn describe(&self) -> String {
        format!(
            "teacache (threshold {}, retain {}, cooldown {}, coefficients {:?})",
            self.threshold, self.retain_steps, self.cooldown_steps, self.coefficients
        )
    }
    any!();
}

impl LinearPrecision {
    /// The settings for `recipe` on `model` (the H3 recipe tables live
    /// behind `FASTVIDEO_H3_QUANT`; FP8 and NVFP4 are process-wide).
    pub fn settings_for(&self, model: &str) -> Result<Vec<(&'static str, String)>, String> {
        let h3 = model == "h3";
        Ok(match self.recipe {
            LinearRecipe::Bf16 if h3 => vec![("FASTVIDEO_H3_QUANT", "off".into())],
            LinearRecipe::Bf16 => vec![("FASTVIDEO_FP8", "0".into())],
            LinearRecipe::Mxfp8 if h3 => vec![("FASTVIDEO_H3_QUANT", "mxfp8".into())],
            LinearRecipe::W8A8 if h3 => vec![("FASTVIDEO_H3_QUANT", "w8a8".into())],
            LinearRecipe::Mxfp8 | LinearRecipe::W8A8 => {
                return Err(format!(
                    "{}: the MXFP8 / W8A8 recipe tables are H3's (model {model}); use fp8",
                    self.name()
                ))
            }
            LinearRecipe::Fp8 => vec![("FASTVIDEO_FP8", "1".into())],
            LinearRecipe::Nvfp4 => vec![(
                "FASTVIDEO_NVFP4",
                self.nvfp4_rule.clone().unwrap_or_else(|| "static_6".into()),
            )],
        })
    }
}

impl Technique for LinearPrecision {
    fn name(&self) -> &'static str {
        match self.recipe {
            LinearRecipe::Bf16 => "bf16_linears",
            LinearRecipe::Mxfp8 => "mxfp8",
            LinearRecipe::W8A8 => "w8a8",
            LinearRecipe::Fp8 => "fp8",
            LinearRecipe::Nvfp4 => "nvfp4",
        }
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Load)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::FfnPrecision]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        match self.recipe {
            LinearRecipe::Nvfp4 => &[Capability::SupportsNvfp4Linear],
            _ => &[Capability::HasFfnLinearModules],
        }
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    any!();
}

impl Technique for ActivationPrecision {
    fn name(&self) -> &'static str {
        if self.bf16 {
            "bf16_activations"
        } else {
            "f32_activations"
        }
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Load)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::ActivationPrecision]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn settings(&self) -> Vec<(&'static str, String)> {
        vec![(
            "FASTVIDEO_BF16_ACT",
            if self.bf16 { "1" } else { "0" }.into(),
        )]
    }
    any!();
}

impl Technique for TinyDecoder {
    fn name(&self) -> &'static str {
        match self.kind {
            TinyDecoderKind::Taeh3 => "taeh3",
            TinyDecoderKind::Taehv => "taehv",
        }
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Load)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::VideoDecoder]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        &[Capability::SwappableVideoDecoder]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    any!();
}

impl TinyDecoder {
    /// The weights setting each pipeline reads: H3 `FASTVIDEO_TAEH3_WEIGHTS`,
    /// LTX-2 `FASTVIDEO_LTX2_TAE_WEIGHTS`.
    pub fn settings_for(&self, model: &str) -> Result<Vec<(&'static str, String)>, String> {
        let name = match (self.kind, model) {
            (TinyDecoderKind::Taeh3, "h3") => "FASTVIDEO_TAEH3_WEIGHTS",
            (TinyDecoderKind::Taehv, "ltx2") => "FASTVIDEO_LTX2_TAE_WEIGHTS",
            (kind, model) => {
                return Err(format!(
                    "{}: not a decoder for {model} ({kind:?})",
                    self.name()
                ))
            }
        };
        Ok(self.weights.iter().map(|w| (name, w.clone())).collect())
    }
}

impl Technique for Offload {
    fn name(&self) -> &'static str {
        "offload"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Load)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::Residency]
    }
    fn required_capabilities(&self) -> &'static [Capability] {
        &[Capability::SupportsLayerOffload]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    any!();
}

impl Offload {
    pub fn settings_for(&self, model: &str) -> Result<Vec<(&'static str, String)>, String> {
        let mut v = Vec::new();
        if let Some(d) = &self.dit {
            v.push(("FASTVIDEO_DIT_OFFLOAD", d.clone()));
        }
        if let Some(n) = self.lookahead {
            v.push(("FASTVIDEO_DIT_OFFLOAD_LOOKAHEAD", n.to_string()));
        }
        if let Some(p) = &self.placement {
            if model != "ltx2" {
                return Err(format!(
                    "offload.placement is LTX-2's --offload; {model} takes offload.dit"
                ));
            }
            v.push(("FASTVIDEO_LTX_OFFLOAD", p.clone()));
        }
        Ok(v)
    }
}

impl Technique for KernelFusion {
    fn name(&self) -> &'static str {
        "kernel_fusion"
    }
    fn kind(&self) -> Kind {
        Kind::Transform(TransformPhase::Build)
    }
    fn writes(&self) -> &'static [Seam] {
        &[Seam::KernelFusion]
    }
    fn enabled(&self) -> &Schedule<bool> {
        &self.enabled
    }
    fn settings(&self) -> Vec<(&'static str, String)> {
        FUSION_FLAGS
            .iter()
            .filter_map(|(k, env)| {
                self.flags
                    .get(k)
                    .map(|on| (*env, if *on { "1" } else { "0" }.to_string()))
            })
            .collect()
    }
    any!();
}

/// A technique's settings for `spec` (precision recipes are model-specific).
pub fn settings_for(
    t: &dyn Technique,
    spec: &ModelSpec,
) -> Result<Vec<(&'static str, String)>, String> {
    if let Some(p) = t.downcast_ref::<LinearPrecision>() {
        return p.settings_for(spec.name);
    }
    if let Some(d) = t.downcast_ref::<TinyDecoder>() {
        return d.settings_for(spec.name);
    }
    if let Some(o) = t.downcast_ref::<Offload>() {
        return o.settings_for(spec.name);
    }
    Ok(t.settings())
}

// ---------------------------------------------------------------------------
// parsing helpers shared by the registry
// ---------------------------------------------------------------------------

pub(crate) struct Params<'a> {
    pub name: &'a str,
    pub table: &'a toml::Table,
    used: Vec<&'static str>,
}

impl<'a> Params<'a> {
    pub fn new(name: &'a str, table: &'a toml::Table) -> Self {
        Self {
            name,
            table,
            used: vec!["enabled"],
        }
    }

    pub fn enabled(&self) -> Result<Schedule<bool>, String> {
        match self.table.get("enabled") {
            None => Ok(Schedule::Const(true)),
            Some(v) => {
                Schedule::parse_enabled(v).map_err(|e| format!("techniques.{}: {e}", self.name))
            }
        }
    }

    pub fn get(&mut self, key: &'static str) -> Option<&'a toml::Value> {
        self.used.push(key);
        self.table.get(key)
    }

    pub fn f64(&mut self, key: &'static str) -> Result<Option<f64>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| {
                v.as_float()
                    .or_else(|| v.as_integer().map(|i| i as f64))
                    .ok_or_else(|| format!("techniques.{name}.{key} must be a number"))
            })
            .transpose()
    }

    pub fn usize(&mut self, key: &'static str) -> Result<Option<usize>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| {
                v.as_integer()
                    .and_then(|i| usize::try_from(i).ok())
                    .ok_or_else(|| {
                        format!("techniques.{name}.{key} must be a non-negative integer")
                    })
            })
            .transpose()
    }

    pub fn bool(&mut self, key: &'static str) -> Result<Option<bool>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| format!("techniques.{name}.{key} must be true or false"))
            })
            .transpose()
    }

    pub fn string(&mut self, key: &'static str) -> Result<Option<String>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("techniques.{name}.{key} must be a string"))
            })
            .transpose()
    }

    pub fn index_set(&mut self, key: &'static str) -> Result<Option<StepSet>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| parse_index_set(v, &format!("techniques.{name}.{key}")))
            .transpose()
    }

    pub fn tau(&mut self, key: &'static str) -> Result<Option<Schedule<f64>>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| parse_tau(v).map_err(|e| format!("techniques.{name}: {e}")))
            .transpose()
    }

    pub fn f64_list(&mut self, key: &'static str) -> Result<Option<Vec<f64>>, String> {
        let name = self.name;
        self.get(key)
            .map(|v| {
                v.as_array()
                    .and_then(|a| {
                        a.iter()
                            .map(|x| x.as_float().or_else(|| x.as_integer().map(|i| i as f64)))
                            .collect::<Option<Vec<f64>>>()
                    })
                    .ok_or_else(|| format!("techniques.{name}.{key} must be a list of numbers"))
            })
            .transpose()
    }

    /// Unknown keys are errors (a typo must not silently mean the default).
    pub fn finish(self) -> Result<(), String> {
        let unknown: Vec<&String> = self
            .table
            .keys()
            .filter(|k| !self.used.contains(&k.as_str()))
            .collect();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "techniques.{}: unknown key(s) {unknown:?} (known: {:?})",
                self.name, self.used
            ))
        }
    }
}
