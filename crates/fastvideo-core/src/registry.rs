//! Wan/FastWan model registry.
//!
//! Ported from FastVideo `fastvideo/models/wan/definition.py`. First-match
//! ordering is preserved, including the two-group split around DreamX.

use crate::error::{FastVideoError, Result};
use crate::sampling::WorkloadType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingAlgorithm {
    UniPc,
    Dmd,
    CausalDmd,
    /// TurboDiffusion rCM (1–4 step); SLA attention preferred but dense works.
    Rcm,
}

impl SamplingAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UniPc => "unipc",
            Self::Dmd => "dmd",
            Self::CausalDmd => "causal_dmd",
            Self::Rcm => "rcm",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WanModelDefinition {
    pub pipeline_config: &'static str,
    pub preset: &'static str,
    pub sampling: SamplingAlgorithm,
    pub hf_model_paths: &'static [&'static str],
    pub workload_types: &'static [WorkloadType],
    pub match_any: &'static [&'static str],
    pub match_all: &'static [&'static str],
    pub exclude: &'static [&'static str],
}

impl WanModelDefinition {
    pub fn matches(&self, path_or_class: &str) -> bool {
        let value = path_or_class.to_ascii_lowercase();
        let any_ok = if self.match_any.is_empty() {
            self.hf_model_paths
                .iter()
                .any(|p| value.contains(&p.to_ascii_lowercase()) || p.to_ascii_lowercase().contains(&value))
                || value.contains(&self.preset.replace('_', "-"))
                || value.contains(&self.preset.replace('_', ""))
        } else {
            self.match_any.iter().any(|token| value.contains(token))
        };
        let all_ok = self.match_all.iter().all(|token| value.contains(token));
        let excluded = self.exclude.iter().any(|token| value.contains(token));
        any_ok && all_ok && !excluded
    }

    fn matches_hf_id(&self, model_id: &str) -> bool {
        let value = model_id.to_ascii_lowercase();
        if self
            .hf_model_paths
            .iter()
            .any(|p| p.eq_ignore_ascii_case(model_id) || value.contains(&p.to_ascii_lowercase()))
        {
            return true;
        }
        self.matches(model_id)
    }
}

macro_rules! defn {
    (
        $pipe:expr, $preset:expr, $sampling:expr, $paths:expr, $work:expr
        $(, match_any = $any:expr)?
        $(, match_all = $all:expr)?
        $(, exclude = $ex:expr)?
    ) => {
        WanModelDefinition {
            pipeline_config: $pipe,
            preset: $preset,
            sampling: $sampling,
            hf_model_paths: $paths,
            workload_types: $work,
            match_any: {
                #[allow(unused_assignments, unused_mut)]
                let mut v: &[&str] = &[];
                $(v = $any;)?
                v
            },
            match_all: {
                #[allow(unused_assignments, unused_mut)]
                let mut v: &[&str] = &[];
                $(v = $all;)?
                v
            },
            exclude: {
                #[allow(unused_assignments, unused_mut)]
                let mut v: &[&str] = &[];
                $(v = $ex;)?
                v
            },
        }
    };
}

pub static WAN_MODEL_DEFINITIONS: &[WanModelDefinition] = &[
    defn!(
        "WanT2V480PConfig",
        "wan_t2v_1_3b",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["wanpipeline"]
    ),
    defn!(
        "WanT2V720PConfig",
        "wan_t2v_14b",
        SamplingAlgorithm::UniPc,
        &[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
        ],
        &[WorkloadType::T2V]
    ),
    defn!(
        "WanI2V480PConfig",
        "wan_i2v_14b_480p",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["wanimagetovideo"]
    ),
    defn!(
        "WanI2V720PConfig",
        "wan_i2v_14b_720p",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"],
        &[WorkloadType::I2V]
    ),
    defn!(
        "WanI2V480PConfig",
        "wan_fun_1_3b_inp",
        SamplingAlgorithm::UniPc,
        &["weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"],
        &[WorkloadType::I2V]
    ),
    defn!(
        "WANV2VConfig",
        "wan_fun_1_3b_control",
        SamplingAlgorithm::UniPc,
        &["IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers"],
        &[]
    ),
    // TurboDiffusion / TurboWan — before generic Wan fuzzy matches.
    defn!(
        "TurboDiffusionT2V_1_3B_Config",
        "turbo_t2v_1_3b",
        SamplingAlgorithm::Rcm,
        &["loayrashid/TurboWan2.1-T2V-1.3B-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["turbowan2.1-t2v-1.3b", "turbodiffusion-t2v-1.3b"]
    ),
    defn!(
        "TurboDiffusionT2V_14B_Config",
        "turbo_t2v_14b",
        SamplingAlgorithm::Rcm,
        &["loayrashid/TurboWan2.1-T2V-14B-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["turbowan2.1-t2v-14b", "turbodiffusion-t2v-14b"]
    ),
    defn!(
        "TurboDiffusionI2V_A14B_Config",
        "turbo_i2v_a14b",
        SamplingAlgorithm::Rcm,
        &["loayrashid/TurboWan2.2-I2V-A14B-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["turbowan2.2-i2v", "turbodiffusion-i2v"]
    ),
    defn!(
        "FastWan2_1_T2V_480P_Config",
        "fast_wan_t2v_480p",
        SamplingAlgorithm::Dmd,
        &[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers",
            // QAD is a *training* recipe: quantization-aware finetuning then
            // quantization-aware DMD. The checkpoints are architecturally
            // identical to FastWan 1.3B and ship unquantized (F32, no
            // quantization_config), so the only thing that differs at load time
            // is the weights. They must be listed explicitly: their
            // model_index.json says `WanPipeline`, so class matching would
            // resolve them to the UniPC preset and run the wrong sampler on a
            // 3-step DMD model.
            "FastVideo/FastWan-QAD-1.3B",
            "FastVideo/FastWan-QAD-1.3B-SA2",
            "FastVideo/FastWan-QAD-FP8-1.3B",
        ],
        &[WorkloadType::T2V],
        match_any = &["wandmdpipeline"]
    ),
    defn!(
        "Wan2_2_TI2V_5B_Config",
        "wan_2_2_ti2v_5b",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.2-TI2V-5B-Diffusers"],
        &[WorkloadType::T2V, WorkloadType::I2V]
    ),
    defn!(
        "FastWan2_2_TI2V_5B_Config",
        "fast_wan_2_2_ti2v_5b",
        SamplingAlgorithm::Dmd,
        &[
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        ],
        &[WorkloadType::T2V, WorkloadType::I2V]
    ),
    defn!(
        "LucyEditDevConfig",
        "lucy_edit_dev",
        SamplingAlgorithm::UniPc,
        &["decart-ai/Lucy-Edit-Dev", "decart-ai/Lucy-Edit-1.1-Dev"],
        &[],
        match_any = &["lucy-edit"]
    ),
    defn!(
        "Wan2_2_T2V_A14B_Config",
        "wan_2_2_t2v_a14b",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.2-T2V-A14B-Diffusers"],
        &[WorkloadType::T2V]
    ),
    defn!(
        "Wan2_2_I2V_A14B_Config",
        "wan_2_2_i2v_a14b",
        SamplingAlgorithm::UniPc,
        &["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
        &[WorkloadType::I2V]
    ),
    defn!(
        "SelfForcingWanT2V480PConfig",
        "sf_wan_t2v_1_3b",
        SamplingAlgorithm::CausalDmd,
        &["wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["wancausaldmdpipeline"]
    ),
    defn!(
        "SelfForcingWan2_2_T2V480PConfig",
        "sf_wan_2_2_t2v_a14b",
        SamplingAlgorithm::CausalDmd,
        &["rand0nmr/SFWan2.2-T2V-A14B-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["sfwan2.2", "sfwan2_2"],
        exclude = &["i2v"]
    ),
    defn!(
        "SelfForcingWan2_2_T2V480PConfig",
        "sf_wan_2_2_i2v_a14b",
        SamplingAlgorithm::CausalDmd,
        &["FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["sfwan2.2", "sfwan2_2"],
        match_all = &["i2v"]
    ),
];

/// Resolve a Hugging Face repo id or local path to a Wan definition.
/// Exact HF ids win, then first-match detectors.
pub fn resolve_wan(model_id: &str) -> Result<&'static WanModelDefinition> {
    if let Some(def) = WAN_MODEL_DEFINITIONS
        .iter()
        .find(|d| d.hf_model_paths.iter().any(|p| p.eq_ignore_ascii_case(model_id)))
    {
        return Ok(def);
    }
    WAN_MODEL_DEFINITIONS
        .iter()
        .find(|d| d.matches_hf_id(model_id))
        .ok_or_else(|| FastVideoError::UnknownModel(model_id.to_string()))
}

// ---- Cross-family registry (Wan + LTX + H3; more families in later phases) ----

/// Top-level model family. Matches FastVideo's registry groupings for ids we port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Wan,
    Ltx2,
    H3,
}

impl ModelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wan => "wan",
            Self::Ltx2 => "ltx2",
            Self::H3 => "h3",
        }
    }
}

/// Distilled / base LTX checkpoint line (maps to [`fastvideo_models::ltx2::Ltx2ModelVersion`] at generate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ltx2Line {
    /// LTX-2.0 distilled (8-step CFG=1).
    Distilled20,
    /// LTX-2.5 distilled stage-1 (ancestral + Gemma4 + BWE).
    Distilled25,
    /// LTX-2.0 base (40-step CFG).
    Base20,
    /// LTX-2.3 distilled (5+2 / 8+3 Euler + Gemma3 + BWE + spatial upscaler).
    Distilled23,
    /// LTX-2.3 base (30-step CFG; STG deferred).
    Base23,
}

/// Non-Wan family entry: exact Hub ids + preset + workloads.
#[derive(Debug, Clone, Copy)]
pub struct FamilyModelDefinition {
    pub family: ModelFamily,
    pub preset: &'static str,
    pub hf_model_paths: &'static [&'static str],
    pub workload_types: &'static [WorkloadType],
    pub match_any: &'static [&'static str],
    /// Set for [`ModelFamily::Ltx2`].
    pub ltx_line: Option<Ltx2Line>,
}

impl FamilyModelDefinition {
    fn matches_exact(&self, model_id: &str) -> bool {
        self.hf_model_paths
            .iter()
            .any(|p| p.eq_ignore_ascii_case(model_id))
    }

    fn matches_fuzzy(&self, model_id: &str) -> bool {
        let value = model_id.to_ascii_lowercase();
        if self.matches_exact(model_id) {
            return true;
        }
        if !self.match_any.is_empty() {
            return self.match_any.iter().any(|t| value.contains(t));
        }
        self.hf_model_paths
            .iter()
            .any(|p| value.contains(&p.to_ascii_lowercase()))
            || value.contains(&self.preset.replace('_', "-"))
    }
}

/// Result of [`resolve`]: Wan keeps its rich definition; LTX/H3 use [`FamilyModelDefinition`].
#[derive(Debug, Clone, Copy)]
pub enum ResolvedModel {
    Wan(&'static WanModelDefinition),
    Family(&'static FamilyModelDefinition),
}

impl ResolvedModel {
    pub fn family(self) -> ModelFamily {
        match self {
            Self::Wan(_) => ModelFamily::Wan,
            Self::Family(d) => d.family,
        }
    }

    pub fn preset(self) -> &'static str {
        match self {
            Self::Wan(d) => d.preset,
            Self::Family(d) => d.preset,
        }
    }

    pub fn hf_model_paths(self) -> &'static [&'static str] {
        match self {
            Self::Wan(d) => d.hf_model_paths,
            Self::Family(d) => d.hf_model_paths,
        }
    }

    pub fn workload_types(self) -> &'static [WorkloadType] {
        match self {
            Self::Wan(d) => d.workload_types,
            Self::Family(d) => d.workload_types,
        }
    }
}

macro_rules! family_defn {
    (
        $family:expr, $preset:expr, $paths:expr, $work:expr
        $(, match_any = $any:expr)?
        $(, ltx_line = $ltx:expr)?
    ) => {
        FamilyModelDefinition {
            family: $family,
            preset: $preset,
            hf_model_paths: $paths,
            workload_types: $work,
            match_any: {
                #[allow(unused_assignments, unused_mut)]
                let mut v: &[&str] = &[];
                $(v = $any;)?
                v
            },
            ltx_line: {
                #[allow(unused_assignments, unused_mut)]
                let mut v: Option<Ltx2Line> = None;
                $(v = Some($ltx);)?
                v
            },
        }
    };
}

/// LTX-2 / 2.3 / 2.5 Hub ids we recognize.
pub static LTX2_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Ltx2,
        "ltx2_distilled_20",
        &[
            "FastVideo/LTX2-Distilled-Diffusers",
            "rootonchair/LTX-2-19b-distilled",
        ],
        &[WorkloadType::T2AV],
        match_any = &["ltx2-distilled", "ltx-2-19b-distilled"],
        ltx_line = Ltx2Line::Distilled20
    ),
    family_defn!(
        ModelFamily::Ltx2,
        "ltx2_distilled_25",
        &["Lightricks/LTX-2.5-Diffusers"],
        &[WorkloadType::T2AV],
        match_any = &["ltx-2.5", "ltx2.5"],
        ltx_line = Ltx2Line::Distilled25
    ),
    family_defn!(
        ModelFamily::Ltx2,
        "ltx2_base_20",
        &[
            "Lightricks/LTX-2",
            "FastVideo/LTX2-base",
            "FastVideo/LTX2-Diffusers",
        ],
        &[WorkloadType::T2AV],
        match_any = &["ltx2-base", "ltx-2-diffusers"],
        ltx_line = Ltx2Line::Base20
    ),
    family_defn!(
        ModelFamily::Ltx2,
        "ltx2_distilled_23",
        &[
            "FastVideo/LTX-2.3-Distilled-Diffusers",
            "FastVideo/LTX2.3-Distilled-Diffusers",
            "diffusers/LTX-2.3-Distilled-Diffusers",
        ],
        &[WorkloadType::T2AV, WorkloadType::I2V],
        match_any = &["ltx-2.3-distilled", "ltx2.3-distilled"],
        ltx_line = Ltx2Line::Distilled23
    ),
    family_defn!(
        ModelFamily::Ltx2,
        "ltx2_base_23",
        &[
            "Lightricks/LTX-2.3",
            "FastVideo/LTX2.3-base",
            "FastVideo/LTX2.3-Diffusers",
            "diffusers/LTX-2.3-Diffusers",
        ],
        &[WorkloadType::T2AV, WorkloadType::I2V],
        match_any = &["ltx-2.3", "ltx2.3"],
        ltx_line = Ltx2Line::Base23
    ),
];

/// MiniMax-H3 / FastH3 Hub ids.
pub static H3_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::H3,
        "fasth3_8step",
        &[
            "FastVideo/FastVideo-FastH3-8-Step-V2",
            "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2",
        ],
        &[WorkloadType::T2AV, WorkloadType::FL2VA],
        match_any = &["fasth3", "fast-h3"]
    ),
    family_defn!(
        ModelFamily::H3,
        "minimax_h3",
        &["MiniMaxAI/MiniMax-H3"],
        &[WorkloadType::T2AV, WorkloadType::FL2VA, WorkloadType::Ref2VA],
        match_any = &["minimax-h3", "minimax_h3"]
    ),
];

fn resolve_family_table(table: &'static [FamilyModelDefinition], model_id: &str) -> Option<&'static FamilyModelDefinition> {
    table
        .iter()
        .find(|d| d.matches_exact(model_id))
        .or_else(|| table.iter().find(|d| d.matches_fuzzy(model_id)))
}

/// Resolve any registered family. Exact Hub ids win within each table; Wan is
/// tried first so Wan-specific detectors do not steal LTX/H3 ids.
pub fn resolve(model_id: &str) -> Result<ResolvedModel> {
    if let Ok(wan) = resolve_wan(model_id) {
        // Prefer an exact LTX/H3 Hub id when Wan fuzzy-matched something else.
        let wan_exact = wan
            .hf_model_paths
            .iter()
            .any(|p| p.eq_ignore_ascii_case(model_id));
        if wan_exact {
            return Ok(ResolvedModel::Wan(wan));
        }
    }
    if let Some(d) = resolve_family_table(LTX2_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(H3_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Ok(wan) = resolve_wan(model_id) {
        return Ok(ResolvedModel::Wan(wan));
    }
    Err(FastVideoError::UnknownModel(model_id.to_string()))
}

/// Every registered Hub id for `list-models` (Wan, then LTX, then H3).
pub fn all_registered_ids() -> Vec<(ModelFamily, &'static str, &'static str)> {
    let mut out = Vec::new();
    for d in WAN_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((ModelFamily::Wan, *id, d.preset));
        }
    }
    for d in LTX2_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in H3_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::WorkloadType;

    /// QAD ships `"_class_name": "WanPipeline"`, so class matching alone would
    /// hand it the UniPC preset and silently run a 3-step DMD model on a
    /// multi-step sampler. The explicit repo entry is what prevents that.
    #[test]
    fn qad_checkpoints_resolve_to_dmd_not_unipc() {
        for id in [
            "FastVideo/FastWan-QAD-FP8-1.3B",
            "FastVideo/FastWan-QAD-1.3B",
            "FastVideo/FastWan-QAD-1.3B-SA2",
        ] {
            let def = resolve_wan(id).unwrap_or_else(|_| panic!("{id} did not resolve"));
            assert_eq!(def.sampling, SamplingAlgorithm::Dmd, "{id} must use DMD");
            assert_eq!(def.preset, "fast_wan_t2v_480p", "{id} preset");
        }
    }

    #[test]
    fn resolves_wan_1_3b_exact() {
        let def = resolve_wan("Wan-AI/Wan2.1-T2V-1.3B-Diffusers").unwrap();
        assert_eq!(def.preset, "wan_t2v_1_3b");
        assert_eq!(def.sampling, SamplingAlgorithm::UniPc);
    }

    #[test]
    fn resolves_fastwan_dmd() {
        let def = resolve_wan("FastVideo/FastWan2.1-T2V-1.3B-Diffusers").unwrap();
        assert_eq!(def.preset, "fast_wan_t2v_480p");
        assert_eq!(def.sampling, SamplingAlgorithm::Dmd);
    }

    #[test]
    fn resolves_turbowan_rcm() {
        let t2v = resolve_wan("loayrashid/TurboWan2.1-T2V-1.3B-Diffusers").unwrap();
        assert_eq!(t2v.preset, "turbo_t2v_1_3b");
        assert_eq!(t2v.sampling, SamplingAlgorithm::Rcm);
        assert!(t2v.workload_types.contains(&WorkloadType::T2V));

        let t2v14 = resolve_wan("turbowan2.1-t2v-14b").unwrap();
        assert_eq!(t2v14.preset, "turbo_t2v_14b");
        assert_eq!(t2v14.sampling, SamplingAlgorithm::Rcm);

        let i2v = resolve_wan("loayrashid/TurboWan2.2-I2V-A14B-Diffusers").unwrap();
        assert_eq!(i2v.preset, "turbo_i2v_a14b");
        assert_eq!(i2v.sampling, SamplingAlgorithm::Rcm);
        assert!(i2v.workload_types.contains(&WorkloadType::I2V));
    }

    #[test]
    fn unknown_id_errors() {
        assert!(resolve_wan("not-a-real/model").is_err());
    }

    #[test]
    fn family_ids_resolve() {
        assert_eq!(
            resolve_wan("Wan-AI/Wan2.1-T2V-14B-Diffusers")
                .unwrap()
                .preset,
            "wan_t2v_14b"
        );
        assert_eq!(
            resolve_wan("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
                .unwrap()
                .workload_types[0],
            WorkloadType::I2V
        );
        assert_eq!(
            resolve_wan("Wan-AI/Wan2.2-TI2V-5B-Diffusers")
                .unwrap()
                .preset,
            "wan_2_2_ti2v_5b"
        );
        assert_eq!(
            resolve_wan("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
                .unwrap()
                .pipeline_config,
            "Wan2_2_T2V_A14B_Config"
        );
        assert_eq!(
            resolve_wan("wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers")
                .unwrap()
                .sampling,
            SamplingAlgorithm::CausalDmd
        );
    }

    #[test]
    fn resolve_cross_family() {
        let wan = resolve("Wan-AI/Wan2.1-T2V-1.3B-Diffusers").unwrap();
        assert_eq!(wan.family(), ModelFamily::Wan);
        assert_eq!(wan.preset(), "wan_t2v_1_3b");

        let ltx25 = resolve("Lightricks/LTX-2.5-Diffusers").unwrap();
        assert_eq!(ltx25.family(), ModelFamily::Ltx2);
        assert_eq!(ltx25.preset(), "ltx2_distilled_25");
        match ltx25 {
            ResolvedModel::Family(d) => assert_eq!(d.ltx_line, Some(Ltx2Line::Distilled25)),
            _ => panic!("expected family"),
        }

        let ltx20 = resolve("FastVideo/LTX2-Distilled-Diffusers").unwrap();
        assert_eq!(ltx20.preset(), "ltx2_distilled_20");

        let ltx23 = resolve("FastVideo/LTX-2.3-Distilled-Diffusers").unwrap();
        assert_eq!(ltx23.preset(), "ltx2_distilled_23");
        match ltx23 {
            ResolvedModel::Family(d) => assert_eq!(d.ltx_line, Some(Ltx2Line::Distilled23)),
            _ => panic!("expected family"),
        }
        let ltx23_base = resolve("diffusers/LTX-2.3-Diffusers").unwrap();
        assert_eq!(ltx23_base.preset(), "ltx2_base_23");

        let h3 = resolve("MiniMaxAI/MiniMax-H3").unwrap();
        assert_eq!(h3.family(), ModelFamily::H3);
        assert!(h3.workload_types().contains(&WorkloadType::FL2VA));

        let fasth3 = resolve("FastVideo/FastVideo-FastH3-8-Step-V2").unwrap();
        assert_eq!(fasth3.preset(), "fasth3_8step");

        assert!(resolve("not-a-real/model").is_err());
    }

    #[test]
    fn list_models_covers_families() {
        let ids = all_registered_ids();
        assert!(ids.iter().any(|(f, id, _)| *f == ModelFamily::Wan && id.contains("Wan2.1")));
        assert!(ids.iter().any(|(f, id, _)| *f == ModelFamily::Ltx2 && id.contains("LTX-2.5")));
        assert!(ids.iter().any(|(f, id, _)| *f == ModelFamily::H3 && id.contains("MiniMax-H3")));
    }
}
