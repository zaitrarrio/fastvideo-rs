//! FLUX.2 (T2I): the variant table and the run (a research port behind
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
pub const NAME: &str = "flux2";

/// The FLUX.2 (T2I) family.
pub struct Flux2;

/// FLUX.2 T2I Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "flux2_klein_4b",
        &["black-forest-labs/FLUX.2-klein-4B"],
        &[WorkloadType::T2I],
        &["flux.2-klein-4b", "flux2-klein-4b"],
    ),
    HubVariant::new(
        "flux2_klein_9b",
        &["black-forest-labs/FLUX.2-klein-9B"],
        &[WorkloadType::T2I],
        &["flux.2-klein-9b", "flux2-klein-9b"],
    ),
    HubVariant::new(
        "flux2_dev",
        &["black-forest-labs/FLUX.2-dev"],
        &[WorkloadType::T2I],
        &["flux.2-dev", "flux2-dev"],
    ),
];

impl VideoFamily for Flux2 {
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
        use fastvideo_cudarc::flux2::pipeline::{Flux2Pipeline, GenerateConfig};
        use fastvideo_models::flux2::Flux2Preset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "flux2_klein_4b" => Flux2Preset::Klein4b,
            "flux2_klein_9b" => Flux2Preset::Klein9b,
            "flux2_dev" => Flux2Preset::Dev,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown FLUX.2 preset {other}"
                )));
            }
        };
        let mut pipe = if weights.join("transformer").is_dir() {
            Flux2Pipeline::load(&weights, preset.as_str())
                .map_err(|e| FastVideoError::Message(e.to_string()))?
        } else {
            Flux2Pipeline::tiny_for_preset(preset.as_str())
        };
        let mut cfg = GenerateConfig {
            prompt: req.prompt.clone(),
            seed: req.seed.unwrap_or(DEFAULT_SEED),
            output_dir: output.display().to_string(),
            preset: preset.as_str().into(),
            tiny: !weights.join("transformer").is_dir(),
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
            "rebuild with --features cuda-cudarc to generate FLUX.2".into(),
        ))
    }
}
