//! Stable Audio Open (T2A): the variant table and the run (a research port behind
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
pub const NAME: &str = "stable_audio";

/// The Stable Audio Open (T2A) family.
pub struct StableAudio;

/// Stable Audio Open T2A Hub ids.
pub static VARIANTS: &[HubVariant] = &[
    HubVariant::new(
        "stable_audio_open_1_0",
        &["FastVideo/stable-audio-open-1.0-Diffusers"],
        &[WorkloadType::T2A],
        &["stable-audio-open-1.0", "stable_audio_open_1_0"],
    ),
    HubVariant::new(
        "stable_audio_open_small",
        &["FastVideo/stable-audio-open-small-Diffusers"],
        &[WorkloadType::T2A],
        &["stable-audio-open-small", "stable_audio_open_small"],
    ),
];

impl VideoFamily for StableAudio {
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
        use fastvideo_cudarc::stable_audio::pipeline::{StableAudioPipeline, StableAudioRequest};
        use fastvideo_models::stable_audio::StableAudioPreset;

        resolve_device(&options.device)?;
        let weights = weights_root(options)?;
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "stable_audio_open_1_0" => StableAudioPreset::Open10,
            "stable_audio_open_small" => StableAudioPreset::OpenSmall,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown Stable Audio preset {other}"
                )));
            }
        };
        let mut pipe = StableAudioPipeline::open(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        if weights.join("transformer").is_dir() {
            pipe.load_dit()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        } else {
            pipe.load_dit_zeros_tiny()
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
        }
        pipe.load_vae_stub();
        let mut request = StableAudioRequest::for_preset(
            preset,
            req.prompt.clone(),
            req.seed.unwrap_or(DEFAULT_SEED),
        );
        if let Some(s) = req.sampler.steps {
            request.num_steps = s as usize;
        }
        std::fs::create_dir_all(&output).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out_wav = output.join("stable-audio.wav");
        pipe.generate(&request, &out_wav)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        Ok(Output {
            family: NAME,
            preset: variant.preset,
            frame_paths: vec![],
            mp4: None,
            wav: Some(out_wav.display().to_string()),
        })
    }
    #[cfg(not(feature = "cuda-cudarc"))]
    {
        let _ = (variant, options, req);
        Err(FastVideoError::Message(
            "rebuild with --features cuda-cudarc to generate Stable Audio".into(),
        ))
    }
}
