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
    /// TurboDiffusion rCM (1–4 step); SLA via `FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`.
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
            self.hf_model_paths.iter().any(|p| {
                value.contains(&p.to_ascii_lowercase()) || p.to_ascii_lowercase().contains(&value)
            }) || value.contains(&self.preset.replace('_', "-"))
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
    if let Some(def) = WAN_MODEL_DEFINITIONS.iter().find(|d| {
        d.hf_model_paths
            .iter()
            .any(|p| p.eq_ignore_ascii_case(model_id))
    }) {
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
    Hunyuan15,
    Kandinsky5,
    Cosmos,
    LongCat,
    LingBot,
    Gen3C,
    MatrixGame,
    DreamX,
    LingBotWorld,
    GameCraft,
    HyWorld,
    ZImage,
    Sd35,
    Flux,
    Flux2,
    GlmImage,
    StableAudio,
    MmAudio,
}

impl ModelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wan => "wan",
            Self::Ltx2 => "ltx2",
            Self::H3 => "h3",
            Self::Hunyuan15 => "hunyuan15",
            Self::Kandinsky5 => "kandinsky5",
            Self::Cosmos => "cosmos",
            Self::LongCat => "longcat",
            Self::LingBot => "lingbot",
            Self::Gen3C => "gen3c",
            Self::MatrixGame => "matrixgame",
            Self::DreamX => "dreamx_world",
            Self::LingBotWorld => "lingbotworld",
            Self::GameCraft => "gamecraft",
            Self::HyWorld => "hyworld",
            Self::ZImage => "zimage",
            Self::Sd35 => "sd35",
            Self::Flux => "flux",
            Self::Flux2 => "flux2",
            Self::GlmImage => "glm_image",
            Self::StableAudio => "stable_audio",
            Self::MmAudio => "mmaudio",
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
    // Before `minimax_h3`: `minimax-h3-turbo` also contains `minimax-h3`.
    family_defn!(
        ModelFamily::H3,
        "sol_h3_ref2va",
        &["lightx2v/Minimax-h3-Turbo"],
        &[WorkloadType::Ref2VA],
        match_any = &["sol-h3-ref2va", "sol_h3_ref2va", "minimax-h3-turbo"]
    ),
    family_defn!(
        ModelFamily::H3,
        "sol_h3",
        &["FastVideo/FastH3-4-step-Preview-v1-LoRA"],
        &[
            WorkloadType::T2AV,
            WorkloadType::I2V,
            WorkloadType::FL2VA,
            WorkloadType::Ref2VA,
        ],
        match_any = &["sol-h3", "sol_h3"]
    ),
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
        &[
            WorkloadType::T2AV,
            WorkloadType::FL2VA,
            WorkloadType::Ref2VA
        ],
        match_any = &["minimax-h3", "minimax_h3"]
    ),
];

/// HunyuanVideo 1.5 Hub ids (FastVideo `hunyuan15` presets).
pub static HUNYUAN15_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Hunyuan15,
        "hy15_480p_t2v",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"],
        &[WorkloadType::T2V],
        match_any = &["hunyuanvideo-1.5-diffusers-480p_t2v", "hy15-480p-t2v"]
    ),
    family_defn!(
        ModelFamily::Hunyuan15,
        "hy15_480p_i2v_distilled",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"],
        &[WorkloadType::I2V],
        match_any = &["480p_i2v_step_distilled", "hy15-480p-i2v"]
    ),
    family_defn!(
        ModelFamily::Hunyuan15,
        "hy15_720p_t2v",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"],
        &[WorkloadType::T2V],
        match_any = &["hunyuanvideo-1.5-diffusers-720p_t2v", "hy15-720p-t2v"]
    ),
    family_defn!(
        ModelFamily::Hunyuan15,
        "hy15_720p_i2v_distilled",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"],
        &[WorkloadType::I2V],
        match_any = &["720p_i2v_distilled", "hy15-720p-i2v"]
    ),
    family_defn!(
        ModelFamily::Hunyuan15,
        "hy15_1080p_sr",
        &[
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p",
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR",
        ],
        &[WorkloadType::T2V],
        match_any = &["hunyuanvideo-1.5-diffusers-1080p", "hy15-1080p"]
    ),
];

/// Kandinsky 5.0 Hub ids (FastVideo `kandinsky5` presets).
pub static KANDINSKY5_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Kandinsky5,
        "k5_lite_t2v_5s",
        &["kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["kandinsky-5.0-t2v-lite", "k5-lite-t2v"]
    ),
    family_defn!(
        ModelFamily::Kandinsky5,
        "k5_pro_t2v_5s",
        &["kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["kandinsky-5.0-t2v-pro", "k5-pro-t2v"]
    ),
];

/// Cosmos Predict2 Video2World Hub ids.
pub static COSMOS_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Cosmos,
        "cosmos2_v2w_2b",
        &["nvidia/Cosmos-Predict2-2B-Video2World"],
        &[WorkloadType::I2V],
        match_any = &["cosmos-predict2-2b-video2world", "cosmos2-2b-v2w"]
    ),
    family_defn!(
        ModelFamily::Cosmos,
        "cosmos2_v2w_14b",
        &["nvidia/Cosmos-Predict2-14B-Video2World"],
        &[WorkloadType::I2V],
        match_any = &["cosmos-predict2-14b-video2world", "cosmos2-14b-v2w"]
    ),
];

/// LongCat-Video Hub ids.
pub static LONGCAT_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::LongCat,
        "longcat_t2v_480p",
        &["FastVideo/LongCat-Video-T2V-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["longcat-video-t2v", "longcat-t2v-480p"]
    ),
    family_defn!(
        ModelFamily::LongCat,
        "longcat_t2v_720p",
        &["FastVideo/LongCat-Video-T2V-Diffusers"],
        &[WorkloadType::T2V],
        match_any = &["longcat-t2v-720p", "longcat-720p-bsa"]
    ),
];

/// LingBot-Video Hub ids.
pub static LINGBOT_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::LingBot,
        "lingbot_dense_1_3b",
        &["robbyant/lingbot-video-dense-1.3b"],
        &[WorkloadType::T2V],
        match_any = &["lingbot-video-dense", "lingbot-dense-1.3b"]
    ),
    family_defn!(
        ModelFamily::LingBot,
        "lingbot_moe_30b",
        &["robbyant/lingbot-video-moe-30b-a3b"],
        &[WorkloadType::T2V],
        match_any = &["lingbot-video-moe", "lingbot-moe-30b"]
    ),
];

/// GEN3C (Cosmos + 3D cache) Hub ids.
pub static GEN3C_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::Gen3C,
    "gen3c_cosmos_7b",
    &["FastVideo/GEN3C-Cosmos-7B-Diffusers"],
    &[WorkloadType::I2V],
    match_any = &["gen3c-cosmos", "gen3c_cosmos_7b"]
)];

/// Matrix-Game Hub ids.
pub static MATRIXGAME_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::MatrixGame,
        "mg2_base_distilled",
        &["FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["matrix-game-2.0-base-distilled", "mg2-base-distilled"]
    ),
    family_defn!(
        ModelFamily::MatrixGame,
        "mg2_gta_distilled",
        &["FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["matrix-game-2.0-gta-distilled", "mg2-gta"]
    ),
    family_defn!(
        ModelFamily::MatrixGame,
        "mg2_templerun_distilled",
        &["FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["matrix-game-2.0-templerun", "mg2-templerun"]
    ),
    family_defn!(
        ModelFamily::MatrixGame,
        "mg2_base",
        &[
            "FastVideo/Matrix-Game-2.0-Base-Diffusers",
            "FastVideo/Matrix-Game-2.0-GTA-Diffusers",
            "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers",
        ],
        &[WorkloadType::I2V],
        match_any = &["matrix-game-2.0-base-diffusers", "mg2-base"]
    ),
    family_defn!(
        ModelFamily::MatrixGame,
        "mg3_base_distilled",
        &["FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["matrix-game-3.0", "mg3-base"]
    ),
];

/// DreamX-World Hub ids.
pub static DREAMX_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::DreamX,
        "dreamx_5b_cam",
        &["FastVideo/DreamX-World-5B-Cam-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["dreamx-world-5b-cam", "dreamx_5b_cam"]
    ),
    family_defn!(
        ModelFamily::DreamX,
        "dreamx_5b_ar",
        &["FastVideo/DreamX-World-5B-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["dreamx-world-5b-diffusers", "dreamx_5b_ar"]
    ),
];

/// LingBot-World Hub ids.
pub static LINGBOTWORLD_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::LingBotWorld,
        "lingbotworld_base_cam",
        &["FastVideo/LingBot-World-Base-Cam-Diffusers"],
        &[WorkloadType::I2V],
        match_any = &["lingbot-world-base-cam", "lingbotworld_base_cam"]
    ),
    family_defn!(
        ModelFamily::LingBotWorld,
        "lingbotworld2_causal_fast",
        &["robbyant/lingbot-world-v2-14b-causal-fast"],
        &[WorkloadType::I2V],
        match_any = &["lingbot-world-v2", "lingbotworld2"]
    ),
];

/// HunyuanGameCraft Hub ids.
pub static GAMECRAFT_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::GameCraft,
    "gamecraft_i2v",
    &["FastVideo/HunyuanGameCraft-Diffusers"],
    &[WorkloadType::I2V],
    match_any = &["hunyuangamecraft", "gamecraft"]
)];

/// HY-WorldPlay Hub ids.
pub static HYWORLD_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::HyWorld,
    "hyworld_bidirectional",
    &["FastVideo/HY-WorldPlay-Bidirectional-Diffusers"],
    &[WorkloadType::I2V],
    match_any = &["hy-worldplay", "hyworld"]
)];

/// Z-Image T2I Hub ids.
pub static ZIMAGE_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::ZImage,
    "zimage_turbo",
    &["Tongyi-MAI/Z-Image-Turbo"],
    &[WorkloadType::T2I],
    match_any = &["z-image", "zimage"]
)];

/// SD 3.5 T2I Hub ids.
pub static SD35_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::Sd35,
    "sd35_medium",
    &["stabilityai/stable-diffusion-3.5-medium"],
    &[WorkloadType::T2I],
    match_any = &["stable-diffusion-3.5", "sd35", "sd3.5"]
)];

/// FLUX.1 T2I Hub ids.
pub static FLUX_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Flux,
        "flux1_dev",
        &["black-forest-labs/FLUX.1-dev"],
        &[WorkloadType::T2I],
        match_any = &["flux.1-dev", "flux1-dev", "flux-1-dev"]
    ),
    family_defn!(
        ModelFamily::Flux,
        "flux1_schnell",
        &["black-forest-labs/FLUX.1-schnell"],
        &[WorkloadType::T2I],
        match_any = &["flux.1-schnell", "flux1-schnell", "flux-schnell"]
    ),
];

/// FLUX.2 T2I Hub ids.
pub static FLUX2_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::Flux2,
        "flux2_klein_4b",
        &["black-forest-labs/FLUX.2-klein-4B"],
        &[WorkloadType::T2I],
        match_any = &["flux.2-klein-4b", "flux2-klein-4b"]
    ),
    family_defn!(
        ModelFamily::Flux2,
        "flux2_klein_9b",
        &["black-forest-labs/FLUX.2-klein-9B"],
        &[WorkloadType::T2I],
        match_any = &["flux.2-klein-9b", "flux2-klein-9b"]
    ),
    family_defn!(
        ModelFamily::Flux2,
        "flux2_dev",
        &["black-forest-labs/FLUX.2-dev"],
        &[WorkloadType::T2I],
        match_any = &["flux.2-dev", "flux2-dev"]
    ),
];

/// GLM-Image T2I Hub ids.
pub static GLM_IMAGE_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::GlmImage,
    "glm_image",
    &["zai-org/GLM-Image"],
    &[WorkloadType::T2I],
    match_any = &["glm-image", "glm_image"]
)];

/// Stable Audio Open T2A Hub ids.
pub static STABLE_AUDIO_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[
    family_defn!(
        ModelFamily::StableAudio,
        "stable_audio_open_1_0",
        &["FastVideo/stable-audio-open-1.0-Diffusers"],
        &[WorkloadType::T2A],
        match_any = &["stable-audio-open-1.0", "stable_audio_open_1_0"]
    ),
    family_defn!(
        ModelFamily::StableAudio,
        "stable_audio_open_small",
        &["FastVideo/stable-audio-open-small-Diffusers"],
        &[WorkloadType::T2A],
        match_any = &["stable-audio-open-small", "stable_audio_open_small"]
    ),
];

/// MMAudio T2A/V2A Hub ids.
///
/// Public upstream pack is `hkchengrex/MMAudio` (native `.pth`, not Diffusers).
/// Diffusers conversion remains `FastVideo/MMAudio-large-44k-v2-Diffusers`
/// (reserved). Prefer `MMAUDIO_MODEL_PATH` for a local converted layout.
pub static MMAUDIO_MODEL_DEFINITIONS: &[FamilyModelDefinition] = &[family_defn!(
    ModelFamily::MmAudio,
    "mmaudio_large_44k_v2",
    &[
        "hkchengrex/MMAudio",
        "FastVideo/MMAudio-large-44k-v2-Diffusers",
    ],
    &[WorkloadType::V2A, WorkloadType::T2A],
    match_any = &["mmaudio-large", "mmaudio", "mmaudio_large_44k_v2"]
)];

fn resolve_family_table(
    table: &'static [FamilyModelDefinition],
    model_id: &str,
) -> Option<&'static FamilyModelDefinition> {
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
    if let Some(d) = resolve_family_table(HUNYUAN15_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(KANDINSKY5_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(COSMOS_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(LONGCAT_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(LINGBOT_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(GEN3C_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(MATRIXGAME_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(DREAMX_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(LINGBOTWORLD_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(GAMECRAFT_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(HYWORLD_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(ZIMAGE_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(SD35_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(FLUX_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(FLUX2_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(GLM_IMAGE_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(STABLE_AUDIO_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Some(d) = resolve_family_table(MMAUDIO_MODEL_DEFINITIONS, model_id) {
        return Ok(ResolvedModel::Family(d));
    }
    if let Ok(wan) = resolve_wan(model_id) {
        return Ok(ResolvedModel::Wan(wan));
    }
    Err(FastVideoError::UnknownModel(model_id.to_string()))
}

/// Every registered Hub id for `list-models` (Wan, then LTX, H3, Hunyuan15, Kandinsky5).
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
    for d in HUNYUAN15_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in KANDINSKY5_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in COSMOS_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in LONGCAT_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in LINGBOT_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in GEN3C_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in MATRIXGAME_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in DREAMX_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in LINGBOTWORLD_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in GAMECRAFT_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in HYWORLD_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in ZIMAGE_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in SD35_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in FLUX_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in FLUX2_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in GLM_IMAGE_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in STABLE_AUDIO_MODEL_DEFINITIONS {
        for id in d.hf_model_paths {
            out.push((d.family, *id, d.preset));
        }
    }
    for d in MMAUDIO_MODEL_DEFINITIONS {
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

        let sol = resolve("FastVideo/FastH3-4-step-Preview-v1-LoRA").unwrap();
        assert_eq!(sol.preset(), "sol_h3");
        assert!(sol.workload_types().contains(&WorkloadType::I2V));
        let sol_ref = resolve("lightx2v/Minimax-h3-Turbo").unwrap();
        assert_eq!(sol_ref.preset(), "sol_h3_ref2va");
        assert_eq!(resolve("sol-h3").unwrap().preset(), "sol_h3");
        assert_eq!(resolve("sol-h3-ref2va").unwrap().preset(), "sol_h3_ref2va");

        let hy = resolve("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v").unwrap();
        assert_eq!(hy.family(), ModelFamily::Hunyuan15);
        assert_eq!(hy.preset(), "hy15_480p_t2v");
        assert!(hy.workload_types().contains(&WorkloadType::T2V));

        let k5 = resolve("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers").unwrap();
        assert_eq!(k5.family(), ModelFamily::Kandinsky5);
        assert_eq!(k5.preset(), "k5_lite_t2v_5s");

        let cosmos = resolve("nvidia/Cosmos-Predict2-2B-Video2World").unwrap();
        assert_eq!(cosmos.family(), ModelFamily::Cosmos);
        assert_eq!(cosmos.preset(), "cosmos2_v2w_2b");

        let longcat = resolve("FastVideo/LongCat-Video-T2V-Diffusers").unwrap();
        assert_eq!(longcat.family(), ModelFamily::LongCat);
        assert_eq!(longcat.preset(), "longcat_t2v_480p");
        let longcat_720 = resolve("longcat-t2v-720p").unwrap();
        assert_eq!(longcat_720.preset(), "longcat_t2v_720p");

        let lingbot = resolve("robbyant/lingbot-video-dense-1.3b").unwrap();
        assert_eq!(lingbot.family(), ModelFamily::LingBot);
        assert_eq!(lingbot.preset(), "lingbot_dense_1_3b");

        let gen3c = resolve("FastVideo/GEN3C-Cosmos-7B-Diffusers").unwrap();
        assert_eq!(gen3c.family(), ModelFamily::Gen3C);
        assert_eq!(gen3c.preset(), "gen3c_cosmos_7b");

        let mg3 = resolve("FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers").unwrap();
        assert_eq!(mg3.family(), ModelFamily::MatrixGame);
        assert_eq!(mg3.preset(), "mg3_base_distilled");

        let dreamx = resolve("FastVideo/DreamX-World-5B-Cam-Diffusers").unwrap();
        assert_eq!(dreamx.family(), ModelFamily::DreamX);
        assert_eq!(dreamx.preset(), "dreamx_5b_cam");

        let lbw = resolve("FastVideo/LingBot-World-Base-Cam-Diffusers").unwrap();
        assert_eq!(lbw.family(), ModelFamily::LingBotWorld);

        let gc = resolve("FastVideo/HunyuanGameCraft-Diffusers").unwrap();
        assert_eq!(gc.family(), ModelFamily::GameCraft);

        let hyw = resolve("FastVideo/HY-WorldPlay-Bidirectional-Diffusers").unwrap();
        assert_eq!(hyw.family(), ModelFamily::HyWorld);

        let zimg = resolve("Tongyi-MAI/Z-Image-Turbo").unwrap();
        assert_eq!(zimg.family(), ModelFamily::ZImage);
        assert_eq!(zimg.preset(), "zimage_turbo");

        let sd35 = resolve("stabilityai/stable-diffusion-3.5-medium").unwrap();
        assert_eq!(sd35.family(), ModelFamily::Sd35);
        assert_eq!(sd35.preset(), "sd35_medium");

        let flux = resolve("black-forest-labs/FLUX.1-dev").unwrap();
        assert_eq!(flux.family(), ModelFamily::Flux);
        let schnell = resolve("black-forest-labs/FLUX.1-schnell").unwrap();
        assert_eq!(schnell.preset(), "flux1_schnell");

        let flux2 = resolve("black-forest-labs/FLUX.2-klein-4B").unwrap();
        assert_eq!(flux2.preset(), "flux2_klein_4b");
        let flux2_dev = resolve("black-forest-labs/FLUX.2-dev").unwrap();
        assert_eq!(flux2_dev.preset(), "flux2_dev");

        let glm = resolve("zai-org/GLM-Image").unwrap();
        assert_eq!(glm.family(), ModelFamily::GlmImage);

        let sa = resolve("FastVideo/stable-audio-open-small-Diffusers").unwrap();
        assert_eq!(sa.family(), ModelFamily::StableAudio);
        assert!(sa.workload_types().contains(&WorkloadType::T2A));

        let mma = resolve("FastVideo/MMAudio-large-44k-v2-Diffusers").unwrap();
        assert_eq!(mma.family(), ModelFamily::MmAudio);
        assert!(mma.workload_types().contains(&WorkloadType::V2A));
        let mma_pub = resolve("hkchengrex/MMAudio").unwrap();
        assert_eq!(mma_pub.family(), ModelFamily::MmAudio);
        assert_eq!(mma_pub.preset(), "mmaudio_large_44k_v2");

        assert!(resolve("not-a-real/model").is_err());
    }

    #[test]
    fn list_models_covers_families() {
        let ids = all_registered_ids();
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Wan && id.contains("Wan2.1")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Ltx2 && id.contains("LTX-2.5")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::H3 && id.contains("MiniMax-H3")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Hunyuan15 && id.contains("HunyuanVideo-1.5")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Kandinsky5 && id.contains("Kandinsky-5.0")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::LongCat && id.contains("LongCat")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::LingBot && id.contains("lingbot")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Gen3C && id.contains("GEN3C")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::MatrixGame && id.contains("Matrix-Game")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::DreamX && id.contains("DreamX")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::LingBotWorld && id.contains("LingBot-World")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::GameCraft && id.contains("GameCraft")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::HyWorld && id.contains("WorldPlay")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::ZImage && id.contains("Z-Image")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Sd35 && id.contains("stable-diffusion-3.5")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Flux && id.contains("FLUX.1")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::Flux2 && id.contains("FLUX.2")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::GlmImage && id.contains("GLM-Image")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::StableAudio && id.contains("stable-audio")));
        assert!(ids
            .iter()
            .any(|(f, id, _)| *f == ModelFamily::MmAudio && id.contains("MMAudio")));
    }
}
