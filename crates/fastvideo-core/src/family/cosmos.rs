//! Cosmos Predict2 Video2World: the variant table and the run (a research port behind
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
pub const NAME: &str = "cosmos";

/// The Cosmos Predict2 Video2World family.
pub struct Cosmos;

/// Cosmos Predict2 Video2World Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "cosmos2_v2w_2b",
        &["nvidia/Cosmos-Predict2-2B-Video2World"],
        &[WorkloadType::I2V],
        &["cosmos-predict2-2b-video2world", "cosmos2-2b-v2w"],
    ),
    HubVariant::new(
        "cosmos2_v2w_14b",
        &["nvidia/Cosmos-Predict2-14B-Video2World"],
        &[WorkloadType::I2V],
        &["cosmos-predict2-14b-video2world", "cosmos2-14b-v2w"],
    ),
];

impl VideoFamily for Cosmos {
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
        use fastvideo_cudarc::cosmos::pipeline::{CosmosPipeline, CosmosRequest};
        use fastvideo_models::cosmos::CosmosPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "cosmos2_v2w_2b" => CosmosPreset::V2w2b,
            "cosmos2_v2w_14b" => CosmosPreset::V2w14b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Cosmos preset {other}"
                )));
            }
        };
        let mut pipe = CosmosPipeline::open(&weights, preset)
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
            CosmosRequest::v2w_2b(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        request.preset = preset;
        request.image_path = req.first_image().map(std::path::Path::to_path_buf);
        request.negative_prompt = req.negative.clone().unwrap_or_default();
        let official = fastvideo_models::cosmos::sol::official_requested(
            fastvideo_models::settings::env_var("FASTVIDEO_COSMOS3_OFFICIAL")
                .ok()
                .as_deref(),
        );
        if official {
            if req.canvas.height.is_none() {
                request.height = fastvideo_models::cosmos::sol::OFFICIAL_HEIGHT;
            }
            if req.canvas.width.is_none() {
                request.width = fastvideo_models::cosmos::sol::OFFICIAL_WIDTH;
            }
            if req.timing.frames.is_none() {
                request.num_frames = fastvideo_models::cosmos::sol::OFFICIAL_FRAMES;
            }
            if req.sampler.steps.is_none() {
                request.num_steps = fastvideo_models::cosmos::sol::OFFICIAL_STEPS;
            }
            if req.sampler.guidance.is_none() {
                request.guidance_scale = fastvideo_models::cosmos::sol::OFFICIAL_GUIDANCE;
            }
            request.fps = fastvideo_models::cosmos::sol::OFFICIAL_FPS;
        }
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
            "rebuild with --features cuda-cudarc to generate Cosmos".into(),
        ))
    }
}
