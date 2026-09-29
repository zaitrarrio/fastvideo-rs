//! LTX-2.3 / LTX-2.5 on `Ltx2Pipeline` (WP-11): the distilled DiT and
//! decoders resident, the stage-2 route resolved from the model's own
//! technique profile (`Ltx2Techniques::resolve`), and a `ResolvedJob`
//! mapped to an `Ltx2Request`.
//!
//! `fv-gpucheck --techniques <profile> ltx2 gen --model-version 2.5
//! --two-stage --text <text>` with the same weights, canvas, frames and seed
//! produces the same frames.

use std::path::Path;

use fastvideo_cudarc::ltx2::a2v::DrivingAudio;
use fastvideo_cudarc::ltx2::i2v_encode::ConditioningImage;
use fastvideo_cudarc::ltx2::pipeline::{
    default_sol_stage2, IcReference, Ltx2Paths, Ltx2Pipeline, Ltx2Request, PipelineOptions,
    TextResidency,
};
use fastvideo_models::ltx2::config::Ltx2Config;
use fastvideo_models::ltx2::techniques::{Ltx2Techniques, Stage2Flags};
use fastvideo_protocol::{ApiError, AudioRole, GapId, JobMetrics, ResolvedJob, Task};

use super::caps::{load_profile, ltx_config, Ltx2Recipe, LtxStage2, LtxVersion};
use super::output::{api_err, bytes_mb, stages, wants_audio};
use fastvideo_cudarc::Hooks;

/// The job's keyframes as image conditionings: the first frame at pixel frame
/// 0, the last at `num_frames - 1` (`last_frame_uri`), both at strength 1 and
/// the checkpoint's CRF (`ltx_pipelines` `--image PATH FRAME_IDX 1.0`).
pub fn conditioning_images(job: &ResolvedJob) -> Vec<ConditioningImage> {
    let mut out: Vec<ConditioningImage> = job
        .keyframes
        .iter()
        .map(|(anchor, path)| ConditioningImage {
            path: path.clone(),
            frame_idx: match anchor {
                fastvideo_protocol::Anchor::First => 0,
                fastvideo_protocol::Anchor::Last => (job.num_frames as usize).saturating_sub(1),
            },
            strength: 1.0,
            crf: None,
        })
        .collect();
    // The first frame first, as the reference lists them.
    out.sort_by_key(|c| c.frame_idx);
    out
}

/// The job's reference sheet for the IC-LoRA (`Task::Ref2V` on a model loaded
/// with one): the first reference image at the job's strengths (1 and 1 when
/// unset: the model card's defaults).
pub fn reference(job: &ResolvedJob, ic_lora: bool) -> Result<Option<IcReference>, ApiError> {
    if job.task != Task::Ref2V {
        return Ok(None);
    }
    if !ic_lora {
        return Err(ApiError::invalid_param(
            "task",
            "reference-to-video needs an LTX model loaded with the IC-LoRA",
        ));
    }
    let mut images = job
        .references
        .iter()
        .filter(|(k, _)| *k == fastvideo_protocol::MediaKind::Image);
    let (Some((_, path)), None) = (images.next(), images.next()) else {
        return Err(ApiError::invalid_param(
            "references",
            "LTX reference-to-video takes exactly one reference image (the reference sheet)",
        ));
    };
    if job.references.len() != 1 {
        return Err(ApiError::invalid_param(
            "references",
            "LTX reference-to-video takes one reference image and no video or audio",
        ));
    }
    Ok(Some(IcReference {
        path: path.clone(),
        strength: job.sampling.reference_strength.unwrap_or(1.0),
        lora_strength: job.sampling.reference_lora_strength.unwrap_or(1.0),
    }))
}

/// The job's driving audio (`Task::A2V`, `AudioRole::Drive`): the output
/// carries it, the model's own audio is not generated.
pub fn driving_audio(job: &ResolvedJob) -> Result<Option<DrivingAudio>, ApiError> {
    match (job.task, &job.audio_in) {
        (Task::A2V, Some((AudioRole::Drive, path))) => Ok(Some(DrivingAudio::new(path.clone()))),
        (Task::A2V, _) => Err(ApiError::invalid_param(
            "audio_url",
            "audio-to-video needs the driving audio",
        )),
        (_, Some(_)) => Err(ApiError::invalid_param(
            "audio_url",
            "input audio is only taken by audio-to-video",
        )),
        _ => Ok(None),
    }
}

/// One resident LTX-2 pipeline.
pub struct Ltx2Model {
    pipe: Ltx2Pipeline,
    cfg: Ltx2Config,
    recipe: Ltx2Recipe,
    sol_stage2: bool,
    pisa_stage2: bool,
    use_text_cache: bool,
}

fn text_residency(s: &str) -> Result<TextResidency, ApiError> {
    match s {
        "auto" => Ok(TextResidency::Auto),
        "resident" => Ok(TextResidency::Resident),
        "streamed" => Ok(TextResidency::Streamed),
        other => Err(ApiError::internal(format!(
            "ltx2 text residency `{other}`: expected auto, resident or streamed"
        ))),
    }
}

/// The stage-2 route a recipe runs: its profile's, else the recipe's.
pub fn stage2_route(recipe: &Ltx2Recipe, cfg: &Ltx2Config) -> Result<(bool, bool), ApiError> {
    let recipe_default = default_sol_stage2(cfg, recipe.two_stage, None, false, false);
    let profile = recipe
        .profile
        .as_deref()
        .map(load_profile)
        .transpose()
        .map_err(ApiError::internal)?;
    let flags = match (profile.is_some(), recipe.stage2) {
        (true, _) => Stage2Flags::default(),
        (false, LtxStage2::Sol) => Stage2Flags {
            sol: true,
            ..Stage2Flags::default()
        },
        (false, LtxStage2::Dense) => Stage2Flags {
            dense: true,
            ..Stage2Flags::default()
        },
        (false, LtxStage2::Pisa) => Stage2Flags {
            pisa: true,
            ..Stage2Flags::default()
        },
    };
    let t = Ltx2Techniques::resolve(flags, recipe_default, profile.as_ref())
        .map_err(ApiError::internal)?;
    let got = (t.sol_stage2(), t.pisa_stage2());
    let want = match recipe.stage2 {
        LtxStage2::Sol => (true, false),
        LtxStage2::Pisa => (false, true),
        LtxStage2::Dense => (false, false),
    };
    if recipe.two_stage && got != want {
        return Err(ApiError::internal(format!(
            "ltx2: profile {:?} resolves stage 2 to sol={} pisa={}, the recipe says {:?}",
            recipe.profile, got.0, got.1, recipe.stage2
        )));
    }
    Ok(got)
}

impl Ltx2Model {
    pub fn load(
        recipe: &Ltx2Recipe,
        text_cache: Option<&Path>,
        obs: &mut dyn FnMut(&'static str),
    ) -> Result<Self, ApiError> {
        let cfg = ltx_config(recipe.version);
        let (sol_stage2, pisa_stage2) = stage2_route(recipe, &cfg)?;
        let paths = Ltx2Paths {
            weights: recipe.weights.clone(),
            dit: recipe.dit.clone(),
            text: recipe.text_weights.clone(),
        };
        let options = PipelineOptions {
            text_cache: text_cache.map(Path::to_path_buf),
            text_residency: text_residency(&recipe.text)?,
            dit_offload: None,
            offload: None,
            tae: recipe.tae.clone(),
            ic_lora: recipe.ic_lora.clone(),
        };
        obs("ltx2_pipeline");
        let pipe = Ltx2Pipeline::load(&paths, &cfg, &options)
            .map_err(|e| api_err(&format!("ltx2 load {}", recipe.weights.display()), e))?;
        let _ = fastvideo_cudarc::wan::device::trim_pool();
        Ok(Self {
            pipe,
            cfg,
            recipe: recipe.clone(),
            sol_stage2,
            pisa_stage2,
            use_text_cache: text_cache.is_some(),
        })
    }

    pub fn recipe(&self) -> &Ltx2Recipe {
        &self.recipe
    }

    /// The request `fv-gpucheck ltx2 gen` would build for this job.
    pub fn request(&self, job: &ResolvedJob, dir: &Path) -> Result<Ltx2Request, ApiError> {
        let reference = reference(job, self.recipe.ic_lora.is_some())?;
        let audio = driving_audio(job)?;
        let images = match job.task {
            Task::Ref2V => Vec::new(),
            Task::T2V if job.keyframes.is_empty() => Vec::new(),
            Task::I2V | Task::Keyframes | Task::A2V if self.recipe.version == LtxVersion::V25 => {
                conditioning_images(job)
            }
            Task::A2V => return Err(ApiError::unsupported(GapId::LtxEndpoint)),
            Task::Keyframes => return Err(ApiError::unsupported(GapId::LtxKeyframes)),
            _ => return Err(ApiError::unsupported(GapId::Ltx25I2V)),
        };
        let mut req = Ltx2Request::new(&self.cfg, job.prompt.clone(), dir.to_path_buf());
        req.height = job.height as usize;
        req.width = job.width as usize;
        req.num_frames = job.num_frames as usize;
        req.frame_rate = f64::from(job.fps);
        req.seed = job.seed;
        req.mp4 = false;
        req.two_stage = self.recipe.two_stage;
        req.guidance_scale = 1.0;
        req.audio_guidance_scale = 1.0;
        req.sol_stage2 = self.sol_stage2;
        req.pisa_stage2 = self.pisa_stage2;
        // E4: no audio decode when the output drops it.
        req.skip_audio_decode = !wants_audio(job);
        req.images = images;
        if reference.is_some() {
            // `ICLoraPipeline`'s stage 2 is the plain dense refine.
            req.sol_stage2 = false;
            req.pisa_stage2 = false;
        }
        req.reference = reference;
        req.audio = audio;
        req.validate()
            .map_err(|e| ApiError::invalid(e.to_string()))?;
        Ok(req)
    }

    /// Planned denoise steps (stage 1 + refine).
    pub(crate) fn planned_steps(&self) -> u32 {
        self.recipe.stage1_steps
            + if self.recipe.two_stage {
                self.recipe.refine_steps
            } else {
                0
            }
    }

    pub(crate) fn generate(
        &mut self,
        job: &ResolvedJob,
        dir: &Path,
        hooks: Hooks<'_>,
    ) -> Result<JobMetrics, ApiError> {
        let req = self.request(job, dir)?;
        let out = self
            .pipe
            .generate_with_hooks(&req, self.use_text_cache, None, hooks)
            .map_err(|e| api_err("ltx2 generate", e))?;
        let t = &out.timings;
        let peak = out.memory.iter().map(|p| p.peak_used).max();
        Ok(JobMetrics {
            inference_s: Some(t.denoise_s),
            stage_durations: stages(&[
                ("text", t.text_s),
                ("image_encode", t.image_s),
                ("audio_encode", t.audio_encode_s),
                ("stage1", t.stage1_s),
                ("upsample", t.upsample_s),
                ("stage2", t.stage2_s),
                ("denoise", t.denoise_s),
                ("audio_decode", t.decode_audio_s),
                ("video_decode", t.decode_video_s),
            ]),
            peak_memory_mb: peak.map(bytes_mb),
            build_rtf: None,
        })
    }
}
