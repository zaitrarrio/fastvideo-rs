//! The Wan family: Wan 2.1 / 2.2, FastWan, TurboWan, Self-Forcing Wan, Fun,
//! Lucy (plan B2: one [`VideoFamily`] behind [`super::FAMILIES`]).
//!
//! The variant table is ported from FastVideo
//! `fastvideo/models/wan/definition.py`; first-match ordering is preserved,
//! including the two-group split around DreamX. The CLI's Wan path is
//! [`VideoGenerator`]: [`Wan::load`](VideoFamily::load) plans one and
//! [`Pipeline::generate`] runs it.

use fastvideo_cudarc::hooks::Hooks;

use super::{AvGenerateOptions, Output, Pipeline, Variant, VideoFamily};
use crate::backend_kind::BackendKind;
use crate::error::{FastVideoError, Result};
use crate::generator::{LoadOptions, VideoGenerator};
use crate::request::Request;
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

/// The Wan family.
pub struct Wan;

/// The family name.
pub const NAME: &str = "wan";

impl WanModelDefinition {
    fn variant(&'static self) -> Variant {
        Variant {
            family: NAME,
            preset: self.preset,
            hub_ids: self.hf_model_paths,
            workloads: self.workload_types,
        }
    }
}

/// The Wan definition of a variant's preset.
pub fn definition(preset: &str) -> Option<&'static WanModelDefinition> {
    WAN_MODEL_DEFINITIONS.iter().find(|d| d.preset == preset)
}

impl VideoFamily for Wan {
    fn name(&self) -> &'static str {
        NAME
    }

    fn variants(&self) -> Vec<Variant> {
        WAN_MODEL_DEFINITIONS
            .iter()
            .map(WanModelDefinition::variant)
            .collect()
    }

    fn resolve(&self, model_id: &str) -> Option<Variant> {
        resolve_wan(model_id).ok().map(WanModelDefinition::variant)
    }

    fn detects_last(&self) -> bool {
        true
    }

    fn supports_tiny(&self) -> bool {
        true
    }

    fn requires_cuda(&self) -> bool {
        false
    }

    fn load(&self, variant: Variant, options: &AvGenerateOptions) -> Result<Box<dyn Pipeline>> {
        if definition(variant.preset).is_none() {
            return Err(FastVideoError::UnknownModel(variant.preset.to_string()));
        }
        Ok(Box::new(WanRun {
            options: options.clone(),
        }))
    }
}

/// The options only Wan reads.
#[derive(Debug, Clone)]
pub struct WanOptions {
    pub backend: BackendKind,
    /// Sequence-parallel world (`FASTVIDEO_SP_WORLD`).
    pub num_gpus: u32,
    /// The zero-weight tiny graph (no Hub download).
    pub tiny: bool,
    /// `f32`, `f16` or `bf16` (`None`: bf16 on CUDA, f32 on CPU).
    pub dtype: Option<String>,
}

impl Default for WanOptions {
    fn default() -> Self {
        Self {
            backend: BackendKind::Cudarc,
            num_gpus: 1,
            tiny: false,
            dtype: None,
        }
    }
}

/// A planned Wan run: the [`VideoGenerator`] for the request.
struct WanRun {
    options: AvGenerateOptions,
}

fn path_string(p: &std::path::Path) -> String {
    p.to_string_lossy().into_owned()
}

impl WanRun {
    fn generator(&self, request: &Request) -> Result<VideoGenerator> {
        let o = &self.options;
        let w = &o.family.wan;
        VideoGenerator::from_pretrained(
            &o.model_id,
            LoadOptions {
                backend: w.backend,
                num_gpus: w.num_gpus,
                tiny: w.tiny,
                weights_path: o.weights.as_deref().map(path_string),
                output_path: o.output.as_deref().map(path_string),
                device: o.device.clone(),
                dtype: w.dtype.clone(),
                height: request.canvas.height,
                width: request.canvas.width,
                num_frames: request.timing.frames,
                num_inference_steps: request.sampler.steps,
                guidance_scale: request.sampler.guidance,
                guidance_scale_2: request.sampler.guidance_2,
                seed: request.seed,
                negative_prompt: request.negative.clone(),
                image_path: request.first_image().map(path_string),
                control_path: request.control().map(path_string),
                save_mp4: o.save_mp4,
            },
        )
    }
}

impl Pipeline for WanRun {
    fn describe(&self, request: &Request) -> Result<String> {
        Ok(self.generator(request)?.summary())
    }

    fn generate(&mut self, request: &Request, hooks: Hooks<'_>) -> Result<Output> {
        let generator = self.generator(request)?;
        let out = generator.generate_video_with_hooks(&request.prompt, hooks)?;
        Ok(Output {
            family: NAME,
            preset: generator.definition.preset,
            frame_paths: out.frame_paths,
            mp4: None,
            wav: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
