//! HunyuanVideo 1.5: the variant table and the run (a research port behind
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
pub const NAME: &str = "hunyuan15";

/// The HunyuanVideo 1.5 family.
pub struct Hunyuan15;

/// HunyuanVideo 1.5 Hub ids (FastVideo `hunyuan15` presets).
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "hy15_480p_t2v",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"],
        &[WorkloadType::T2V],
        &["hunyuanvideo-1.5-diffusers-480p_t2v", "hy15-480p-t2v"],
    ),
    HubVariant::new(
        "hy15_480p_i2v_distilled",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"],
        &[WorkloadType::I2V],
        &["480p_i2v_step_distilled", "hy15-480p-i2v"],
    ),
    HubVariant::new(
        "hy15_720p_t2v",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"],
        &[WorkloadType::T2V],
        &["hunyuanvideo-1.5-diffusers-720p_t2v", "hy15-720p-t2v"],
    ),
    HubVariant::new(
        "hy15_720p_i2v_distilled",
        &["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled"],
        &[WorkloadType::I2V],
        &["720p_i2v_distilled", "hy15-720p-i2v"],
    ),
    HubVariant::new(
        "hy15_1080p_sr",
        &[
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p",
            "weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR",
        ],
        &[WorkloadType::T2V],
        &["hunyuanvideo-1.5-diffusers-1080p", "hy15-1080p"],
    ),
];

impl VideoFamily for Hunyuan15 {
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
        use fastvideo_cudarc::hunyuan15::pipeline::{Hunyuan15Pipeline, Hunyuan15Request};
        use fastvideo_models::hunyuan15::Hunyuan15Preset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "hy15_480p_t2v" => Hunyuan15Preset::T2v480p,
            "hy15_480p_i2v_distilled" => Hunyuan15Preset::I2v480pDistilled,
            "hy15_720p_t2v" => Hunyuan15Preset::T2v720p,
            "hy15_720p_i2v_distilled" => Hunyuan15Preset::I2v720pDistilled,
            "hy15_1080p_sr" => Hunyuan15Preset::Sr1080p,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Hunyuan15 preset {other}"
                )));
            }
        };
        let mut pipe = Hunyuan15Pipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        // Optional early weight check so missing transformer/ fails loudly.
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        if weights.join("vae").is_dir() {
            pipe.load_vae()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        let mut request =
            Hunyuan15Request::t2v_480p(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        request.preset = preset;
        let official = fastvideo_models::hunyuan15::sol::official_requested(
            fastvideo_models::settings::env_var("FASTVIDEO_HUNYUAN15_OFFICIAL")
                .ok()
                .as_deref(),
        );
        if official {
            if req.canvas.height.is_none() {
                request.height = fastvideo_models::hunyuan15::sol::OFFICIAL_HEIGHT;
            }
            if req.canvas.width.is_none() {
                request.width = fastvideo_models::hunyuan15::sol::OFFICIAL_WIDTH;
            }
            if req.timing.frames.is_none() {
                request.num_frames = fastvideo_models::hunyuan15::sol::OFFICIAL_FRAMES;
            }
            if req.sampler.steps.is_none() {
                request.num_steps = fastvideo_models::hunyuan15::sol::OFFICIAL_STEPS;
            }
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
        request.image_path = req.first_image().map(std::path::Path::to_path_buf);
        pipe.generate(&request, &output)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: Vec::new(),
            mp4: None,
            wav: None,
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate HunyuanVideo 1.5".into(),
        ))
    }
}
