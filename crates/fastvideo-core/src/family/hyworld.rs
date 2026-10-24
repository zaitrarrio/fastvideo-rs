//! HY-WorldPlay: the variant table and the run (a research port behind
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
pub const NAME: &str = "hyworld";

/// The HY-WorldPlay family.
pub struct HyWorld;

/// HY-WorldPlay Hub ids.
pub static VARIANTS: &[HubVariant] = &[HubVariant::new(
    "hyworld_bidirectional",
    &["FastVideo/HY-WorldPlay-Bidirectional-Diffusers"],
    &[WorkloadType::I2V],
    &["hy-worldplay", "hyworld"],
)];

impl VideoFamily for HyWorld {
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
        use fastvideo_cudarc::hyworld::pipeline::{HyWorldPipeline, HyWorldRequest};
        use fastvideo_models::hyworld::HyWorldPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "hyworld_bidirectional" => HyWorldPreset::Bidirectional,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown HY-World preset {other}"
                )));
            }
        };
        let mut pipe = HyWorldPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request =
            HyWorldRequest::bidirectional(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
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
            "rebuild with --features cuda-cudarc to generate HY-World".into(),
        ))
    }
}
