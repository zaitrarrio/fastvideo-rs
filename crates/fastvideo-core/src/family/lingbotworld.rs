//! LingBot-World: the variant table and the run (a research port behind
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
pub const NAME: &str = "lingbotworld";

/// The LingBot-World family.
pub struct LingBotWorld;

/// LingBot-World Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "lingbotworld_base_cam",
        &["FastVideo/LingBot-World-Base-Cam-Diffusers"],
        &[WorkloadType::I2V],
        &["lingbot-world-base-cam", "lingbotworld_base_cam"],
    ),
    HubVariant::new(
        "lingbotworld2_causal_fast",
        &["robbyant/lingbot-world-v2-14b-causal-fast"],
        &[WorkloadType::I2V],
        &["lingbot-world-v2", "lingbotworld2"],
    ),
];

impl VideoFamily for LingBotWorld {
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
        use super::{collect_png_frames, output_dir, resolve_device, weights_root, DEFAULT_SEED};
        use fastvideo_cudarc::lingbotworld::pipeline::{LingBotWorldPipeline, LingBotWorldRequest};
        use fastvideo_models::lingbotworld::LingBotWorldPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "lingbotworld_base_cam" => LingBotWorldPreset::BaseCam,
            "lingbotworld2_causal_fast" => LingBotWorldPreset::V2CausalFast,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown LingBot-World preset {other}"
                )));
            }
        };
        let mut pipe = LingBotWorldPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request = LingBotWorldRequest::for_preset(
            preset,
            req.prompt.clone(),
            req.seed.unwrap_or(DEFAULT_SEED),
        );
        request.image_path = req.first_image().map(std::path::Path::to_path_buf);
        if let Some(h) = req.canvas.height {
            request.height = h as usize;
        }
        if let Some(w) = req.canvas.width {
            request.width = w as usize;
        }
        if let Some(f) = req.timing.frames {
            request.num_frames = f as usize;
        }
        if let Some(s) = req.sampler.steps {
            request.num_steps = s as usize;
        }
        pipe.generate(&request, &output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: collect_png_frames(&output),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate LingBot-World".into(),
        ))
    }
}
