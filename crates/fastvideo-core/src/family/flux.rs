//! FLUX.1 (T2I): the variant table and the run (a research port behind
//! [`VideoFamily`], plan B2).

use fastvideo_cudarc::hooks::Hooks;

use super::{
    require_cuda, resolve_table, table_variants, AvGenerateOptions, HubVariant, Output, Pipeline,
    Planned, Variant, VideoFamily,
};
use crate::error::{FastVideoError, Result};
use crate::request::Request;
use crate::sampling::WorkloadType;

/// The family name.
pub const NAME: &str = "flux";

/// The FLUX.1 (T2I) family.
pub struct Flux;

/// FLUX.1 T2I Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "flux1_dev",
        &["black-forest-labs/FLUX.1-dev"],
        &[WorkloadType::T2I],
        &["flux.1-dev", "flux1-dev", "flux-1-dev"],
    ),
    HubVariant::new(
        "flux1_schnell",
        &["black-forest-labs/FLUX.1-schnell"],
        &[WorkloadType::T2I],
        &["flux.1-schnell", "flux1-schnell", "flux-schnell"],
    ),
];

impl VideoFamily for Flux {
    fn name(&self) -> &'static str {
        NAME
    }

    fn variants(&self) -> Vec<Variant> {
        table_variants(NAME, VARIANTS)
    }

    fn resolve(&self, model_id: &str) -> Option<Variant> {
        resolve_table(NAME, VARIANTS, model_id)
    }

    fn load(&self, variant: Variant, options: &AvGenerateOptions) -> Result<Box<dyn Pipeline>> {
        Ok(Planned::boxed(variant, options, run))
    }
}

fn run(
    variant: &Variant,
    options: &AvGenerateOptions,
    req: &Request,
    _hooks: Hooks<'_>,
) -> Result<Output> {
    require_cuda(&options.device)?;
    #[cfg(feature = "cuda-cudarc")]
    {
        use super::{output_dir, resolve_device, weights_root, DEFAULT_SEED};
        use fastvideo_cudarc::flux::pipeline::{Flux1Pipeline, GenerateConfig};
        use fastvideo_models::flux::FluxPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "flux1_dev" => FluxPreset::Dev,
            "flux1_schnell" => FluxPreset::Schnell,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown FLUX.1 preset {other}"
                )));
            }
        };
        let mut pipe = if weights.join("transformer").is_dir() {
            Flux1Pipeline::load(&weights, preset.as_str())
                .map_err(|e| FastVideoError::Message(e.to_string()))?
        } else {
            Flux1Pipeline::tiny()
        };
        let mut cfg = GenerateConfig {
            prompt: req.prompt.clone(),
            seed: req.seed.unwrap_or(DEFAULT_SEED),
            output_dir: output.display().to_string(),
            preset: preset.as_str().into(),
            tiny: !weights.join("transformer").is_dir(),
            height: preset.default_height(),
            width: preset.default_width(),
            num_inference_steps: preset.default_steps(),
            guidance_scale: preset.guidance_scale(),
            embedded_cfg_scale: Some(preset.guidance_scale()),
            t5_max_len: if matches!(preset, FluxPreset::Schnell) {
                256
            } else {
                512
            },
            ..GenerateConfig::default()
        };
        if let Some(h) = req.canvas.height {
            cfg.height = h as usize;
        }
        if let Some(w) = req.canvas.width {
            cfg.width = w as usize;
        }
        if let Some(s) = req.sampler.steps {
            cfg.num_inference_steps = s as usize;
        }
        if let Some(g) = req.sampler.guidance {
            cfg.guidance_scale = g;
            cfg.embedded_cfg_scale = Some(g);
        }
        std::fs::create_dir_all(&output).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let paths = pipe
            .generate(&cfg)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: paths,
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate FLUX.1".into(),
        ))
    }
}
