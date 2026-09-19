//! Wan/FastWan and Flux2 model registry.
//!
//! Wan entries are ported from FastVideo `fastvideo/models/wan/definition.py`.
//! Flux2 entries follow `fastvideo/registry.py` (`FLUX.2-dev`, Klein 4B/9B).

use crate::error::{FastVideoError, Result};
use crate::sampling::WorkloadType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Wan,
    Flux2,
}

impl ModelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wan => "wan",
            Self::Flux2 => "flux2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingAlgorithm {
    UniPc,
    Dmd,
    CausalDmd,
    FlowMatchEuler,
}

impl SamplingAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UniPc => "unipc",
            Self::Dmd => "dmd",
            Self::CausalDmd => "causal_dmd",
            Self::FlowMatchEuler => "flow_match_euler",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WanModelDefinition {
    pub family: ModelFamily,
    pub pipeline_config: &'static str,
    pub preset: &'static str,
    pub sampling: SamplingAlgorithm,
    pub hf_model_paths: &'static [&'static str],
    pub workload_types: &'static [WorkloadType],
    pub match_any: &'static [&'static str],
    pub match_all: &'static [&'static str],
    pub exclude: &'static [&'static str],
}

pub type ModelDefinition = WanModelDefinition;
pub type Flux2ModelDefinition = WanModelDefinition;

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
            family: ModelFamily::Wan,
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

macro_rules! flux2_defn {
    (
        $pipe:expr, $preset:expr, $paths:expr, $work:expr
        $(, match_any = $any:expr)?
        $(, match_all = $all:expr)?
        $(, exclude = $ex:expr)?
    ) => {
        WanModelDefinition {
            family: ModelFamily::Flux2,
            pipeline_config: $pipe,
            preset: $preset,
            sampling: SamplingAlgorithm::FlowMatchEuler,
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
    defn!(
        "FastWan2_1_T2V_480P_Config",
        "fast_wan_t2v_480p",
        SamplingAlgorithm::Dmd,
        &[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
            "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers",
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

/// Flux2 T2I ids from FastVideo `register_configs` (Klein first so it wins
/// over the broader `flux.2` detector).
pub static FLUX2_MODEL_DEFINITIONS: &[WanModelDefinition] = &[
    flux2_defn!(
        "Flux2KleinPipelineConfig",
        "flux2_klein_4b",
        &["black-forest-labs/FLUX.2-klein-4B"],
        &[WorkloadType::T2I],
        match_any = &["flux.2-klein-4b", "flux2-klein-4b", "flux2klein-4b"]
    ),
    flux2_defn!(
        "Flux2KleinPipelineConfig",
        "flux2_klein_9b",
        &["black-forest-labs/FLUX.2-klein-9B"],
        &[WorkloadType::T2I],
        match_any = &["flux.2-klein-9b", "flux2-klein-9b", "flux2klein-9b"]
    ),
    flux2_defn!(
        "Flux2KleinPipelineConfig",
        "flux2_klein_4b",
        &[],
        &[WorkloadType::T2I],
        match_any = &["flux.2-klein", "flux2-klein", "flux2klein"]
    ),
    flux2_defn!(
        "Flux2PipelineConfig",
        "flux2_dev",
        &["black-forest-labs/FLUX.2-dev"],
        &[WorkloadType::T2I],
        match_any = &["flux.2", "flux2", "flux_2", "flux-2"],
        exclude = &["klein"]
    ),
];

fn resolve_in<'a>(
    table: &'a [WanModelDefinition],
    model_id: &str,
) -> Option<&'a WanModelDefinition> {
    if let Some(def) = table
        .iter()
        .find(|d| d.hf_model_paths.iter().any(|p| p.eq_ignore_ascii_case(model_id)))
    {
        return Some(def);
    }
    table.iter().find(|d| d.matches_hf_id(model_id))
}

/// Resolve a Hugging Face repo id or local path to a Wan definition.
/// Exact HF ids win, then first-match detectors.
pub fn resolve_wan(model_id: &str) -> Result<&'static WanModelDefinition> {
    resolve_in(WAN_MODEL_DEFINITIONS, model_id)
        .ok_or_else(|| FastVideoError::UnknownModel(model_id.to_string()))
}

/// Resolve Wan or Flux2. Flux2 exact ids win over Wan detectors.
pub fn resolve_model(model_id: &str) -> Result<&'static WanModelDefinition> {
    if let Some(def) = FLUX2_MODEL_DEFINITIONS
        .iter()
        .find(|d| d.hf_model_paths.iter().any(|p| p.eq_ignore_ascii_case(model_id)))
    {
        return Ok(def);
    }
    if let Some(def) = WAN_MODEL_DEFINITIONS
        .iter()
        .find(|d| d.hf_model_paths.iter().any(|p| p.eq_ignore_ascii_case(model_id)))
    {
        return Ok(def);
    }
    if let Some(def) = resolve_in(FLUX2_MODEL_DEFINITIONS, model_id) {
        return Ok(def);
    }
    resolve_wan(model_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::WorkloadType;

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
    fn resolves_flux2_dev_and_klein() {
        let dev = resolve_model("black-forest-labs/FLUX.2-dev").unwrap();
        assert_eq!(dev.family, ModelFamily::Flux2);
        assert_eq!(dev.preset, "flux2_dev");
        assert_eq!(dev.sampling, SamplingAlgorithm::FlowMatchEuler);
        assert_eq!(dev.workload_types[0], WorkloadType::T2I);
        let klein = resolve_model("black-forest-labs/FLUX.2-klein-4B").unwrap();
        assert_eq!(klein.preset, "flux2_klein_4b");
        let klein9 = resolve_model("black-forest-labs/FLUX.2-klein-9B").unwrap();
        assert_eq!(klein9.preset, "flux2_klein_9b");
        let fuzzy = resolve_model("local/FLUX.2-klein-preview").unwrap();
        assert!(fuzzy.preset.contains("klein"));
        assert!(resolve_wan("black-forest-labs/FLUX.2-dev").is_err());
    }
}
