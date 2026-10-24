//! Stable Diffusion 3.5 (T2I): the variant table and the run (a research port behind
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
pub const NAME: &str = "sd35";

/// The Stable Diffusion 3.5 (T2I) family.
pub struct Sd35;

/// SD 3.5 T2I Hub ids.
pub static VARIANTS: &[HubVariant] = &[HubVariant::new(
    "sd35_medium",
    &["stabilityai/stable-diffusion-3.5-medium"],
    &[WorkloadType::T2I],
    &["stable-diffusion-3.5", "sd35", "sd3.5"],
)];

impl VideoFamily for Sd35 {
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
        use fastvideo_cudarc::sd35::pipeline::{Sd35Pipeline, Sd35Request};
        use fastvideo_models::sd35::Sd35Preset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "sd35_medium" => Sd35Preset::Medium,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown SD3.5 preset {other}"
                )));
            }
        };
        let mut pipe = Sd35Pipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        } else {
            pipe.load_dit_zeros_tiny()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        } else {
            pipe.load_vae_stub();
        }
        let mut request = Sd35Request::medium(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        if let Some(h) = req.canvas.height {
            request.height = h as usize;
        }
        if let Some(w) = req.canvas.width {
            request.width = w as usize;
        }
        if let Some(s) = req.sampler.steps {
            request.num_steps = s as usize;
        }
        if let Some(g) = req.sampler.guidance {
            request.guidance_scale = g;
        }
        std::fs::create_dir_all(&output).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out_png = output.join("sd35.png");
        pipe.generate(&request, &out_png)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: vec![out_png.display().to_string()],
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate SD 3.5".into(),
        ))
    }
}
