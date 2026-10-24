//! MMAudio (T2A / V2A): the variant table and the run (a research port behind
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
pub const NAME: &str = "mmaudio";

/// The MMAudio (T2A / V2A) family.
pub struct MmAudio;

/// MMAudio T2A/V2A Hub ids.
///
/// Public upstream pack is `hkchengrex/MMAudio` (native `.pth`, not Diffusers).
/// Diffusers conversion remains `FastVideo/MMAudio-large-44k-v2-Diffusers`
/// (reserved). Prefer `MMAUDIO_MODEL_PATH` for a local converted layout.
pub static VARIANTS: &[HubVariant] = &[HubVariant::new(
    "mmaudio_large_44k_v2",
    &[
        "hkchengrex/MMAudio",
        "FastVideo/MMAudio-large-44k-v2-Diffusers",
    ],
    &[WorkloadType::V2A, WorkloadType::T2A],
    &["mmaudio-large", "mmaudio", "mmaudio_large_44k_v2"],
)];

impl VideoFamily for MmAudio {
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
        use fastvideo_cudarc::mmaudio::pipeline::{MmAudioPipeline, MmAudioRequest};
        use fastvideo_models::mmaudio::MmAudioPreset;

        resolve_device(&options.device)?;
        let weights = MmAudioPipeline::resolve_root(weights_root(options)?);
        let output = output_dir(variant, options);
        let preset = match variant.preset {
            "mmaudio_large_44k_v2" => MmAudioPreset::Large44kV2,
            other => {
                return Err(FastVideoError::Message(format!(
                    "unknown MMAudio preset {other}"
                )));
            }
        };
        let pipe = MmAudioPipeline::load(&weights, preset)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        let mut request = MmAudioRequest::t2a(req.prompt.clone(), req.seed.unwrap_or(DEFAULT_SEED));
        if let Some(v) = req.first_image() {
            let frames = fastvideo_cudarc::mmaudio::VideoFrames::from_file(std::path::Path::new(v))
                .map_err(|e| FastVideoError::Message(e.to_string()))?;
            request.duration_s = frames.times.len() as f64
                / (frames.times.get(1).map_or(1.0, |t| 1.0 / t.max(1e-9)));
            request.video = Some(frames);
        }
        if let Some(s) = req.sampler.steps {
            request.num_steps = s as usize;
        }
        std::fs::create_dir_all(&output).map_err(|e| FastVideoError::Message(e.to_string()))?;
        let out_wav = output.join("mmaudio.wav");
        let out = pipe
            .generate(&request)
            .map_err(|e| FastVideoError::Message(e.to_string()))?;
        fastvideo_cudarc::mmaudio::pipeline::write_wav(&out_wav, &out.waveform, out.sample_rate)
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
            "rebuild with --features cuda-cudarc to generate MMAudio".into(),
        ))
    }
}
