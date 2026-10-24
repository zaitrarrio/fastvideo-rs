//! LingBot-Video: the variant table and the run (a research port behind
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
pub const NAME: &str = "lingbot";

/// The LingBot-Video family.
pub struct LingBot;

/// LingBot-Video Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "lingbot_dense_1_3b",
        &["robbyant/lingbot-video-dense-1.3b"],
        &[WorkloadType::T2V],
        &["lingbot-video-dense", "lingbot-dense-1.3b"],
    ),
    HubVariant::new(
        "lingbot_moe_30b",
        &["robbyant/lingbot-video-moe-30b-a3b"],
        &[WorkloadType::T2V],
        &["lingbot-video-moe", "lingbot-moe-30b"],
    ),
];

impl VideoFamily for LingBot {
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
        use fastvideo_cudarc::lingbot::pipeline::{LingBotPipeline, LingBotRequest};
        use fastvideo_models::lingbot::LingBotPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "lingbot_dense_1_3b" => LingBotPreset::Dense13b,
            "lingbot_moe_30b" => LingBotPreset::Moe30b,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown LingBot preset {other}"
                )));
            }
        };
        let mut pipe = LingBotPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let official = fastvideo_models::lingbot::sol::official_requested(
            fastvideo_models::settings::env_var("FASTVIDEO_LINGBOT_OFFICIAL")
                .ok()
                .as_deref(),
        );
        // Official: the models/lingbot_video.toml two-stage contract (base +
        // 1080p refiner). `FASTVIDEO_LINGBOT_REFINE=0` stops after the base.
        let mut request = if official {
            LingBotRequest::official(req.prompt.clone())
        } else {
            let mut r =
                LingBotRequest::dense_1_3b(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
            r.preset = preset;
            r
        };
        if !official || req.seed.unwrap_or(DEFAULT_SEED) != 0 {
            request.seed = req.seed.unwrap_or(DEFAULT_SEED);
        }
        if matches!(
            fastvideo_models::settings::env_var("FASTVIDEO_LINGBOT_REFINE")
                .ok()
                .as_deref(),
            Some("0") | Some("off")
        ) {
            request.refiner = None;
        }
        request.sol = fastvideo_models::lingbot::sol::SolArm::parse(
            fastvideo_models::settings::env_var("FASTVIDEO_LINGBOT_SOL")
                .ok()
                .as_deref(),
        )
        .map_err(FastVideoError::Message)?;
        if !req.negative.as_deref().unwrap_or_default().is_empty() {
            request.negative_prompt = Some(req.negative.clone().unwrap_or_default());
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
        if let Some(g) = req.sampler.guidance {
            request.guidance_scale = g;
        }
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
            "rebuild with --features cuda-cudarc to generate LingBot".into(),
        ))
    }
}
