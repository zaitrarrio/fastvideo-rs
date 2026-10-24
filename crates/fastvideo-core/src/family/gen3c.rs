//! GEN3C (Cosmos + 3D cache): the variant table and the run (a research port behind
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
pub const NAME: &str = "gen3c";

/// The GEN3C (Cosmos + 3D cache) family.
pub struct Gen3C;

/// GEN3C (Cosmos + 3D cache) Hub ids.
pub static VARIANTS: &[HubVariant] = &[HubVariant::new(
    "gen3c_cosmos_7b",
    &["FastVideo/GEN3C-Cosmos-7B-Diffusers"],
    &[WorkloadType::I2V],
    &["gen3c-cosmos", "gen3c_cosmos_7b"],
)];

impl VideoFamily for Gen3C {
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
        use fastvideo_cudarc::gen3c::pipeline::{Gen3CPipeline, Gen3CRequest};
        use fastvideo_models::gen3c::Gen3CPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "gen3c_cosmos_7b" => Gen3CPreset::Cosmos7b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown GEN3C preset {other}"
                )));
            }
        };
        let mut pipe = Gen3CPipeline::open(&weights, preset)
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
            Gen3CRequest::cosmos_7b(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        request.preset = preset;
        request.image_path = req.first_image().map(std::path::Path::to_path_buf);
        request.negative_prompt = req.negative.clone().unwrap_or_default();
        if let Some(g) = req.sampler.guidance {
            request.guidance_scale = g;
        }
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
        let mut frame_paths: Vec<String> = Vec::new();
        if output.is_dir() {
            if let Ok(rd) = std::fs::read_dir(&output) {
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("png") {
                        frame_paths.push(p.display().to_string());
                    }
                }
                frame_paths.sort();
            }
        }
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths,
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate GEN3C".into(),
        ))
    }
}
