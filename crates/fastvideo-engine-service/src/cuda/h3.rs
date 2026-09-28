//! H3 / FastH3 / Sol-H3 on `H3Pipeline` (WP-11): load once with the
//! recipe's options, switch to the recipe's technique profile (`set_arm`)
//! when it is not the process's, and map a `ResolvedJob` to an `H3Request`.
//!
//! `fv-gpucheck h3 gen --h3-recipe <r> [--techniques <profile>] [--dense]`
//! with the same weights, canvas, frames and seed produces the same frames.

use std::path::Path;

use fastvideo_cudarc::h3::pipeline::{H3Pipeline, H3PipelineOptions, H3Request, TextEncoderChoice};
use fastvideo_protocol::{Anchor, ApiError, JobMetrics, ResolvedJob, Task};

use super::caps::{load_profile, H3Recipe};
use super::output::{api_err, bytes_mb, stages};
use fastvideo_cudarc::Hooks;

/// One resident H3 pipeline.
pub struct H3Model {
    pipe: H3Pipeline,
    recipe: H3Recipe,
}

impl H3Model {
    pub fn load(
        recipe: &H3Recipe,
        text_cache: Option<&Path>,
        obs: &mut dyn FnMut(&'static str),
    ) -> Result<Self, ApiError> {
        let options = H3PipelineOptions {
            dense: recipe.dense,
            adaln_cache: recipe.adaln_cache.clone(),
            text_root: recipe.text_weights.clone(),
            text_cache: text_cache.map(Path::to_path_buf),
            text_encoder: TextEncoderChoice::parse(&recipe.text_encoder)
                .map_err(ApiError::internal)?,
            taeh3: recipe.taeh3.clone(),
            recipe: Some(recipe.recipe.clone()),
            ref2va: false,
            adapter: None,
            reference_image_resize: Default::default(),
            dit_offload: recipe
                .dit_offload
                .as_deref()
                .map(fastvideo_cudarc::wan::offload::DitOffload::parse)
                .transpose()
                .map_err(ApiError::internal)?,
            i2v_encoder: fastvideo_cudarc::h3::pipeline::I2vEncoderChoice::parse(
                &recipe.i2v_encoder,
            )
            .map_err(ApiError::internal)?,
        };
        obs("h3_pipeline");
        let mut pipe = H3Pipeline::load(&recipe.weights, options)
            .map_err(|e| api_err(&format!("h3 load {}", recipe.weights.display()), e))?;
        // The process profile is the plan's; switch this model to its own
        // per-request techniques when it differs (same load-time settings,
        // checked by ProcessPlan; set_arm re-checks).
        let active = fastvideo_models::techniques::settings::active();
        let active_name = active.profile.as_ref().map(|p| p.name.clone());
        let want = recipe
            .profile
            .as_deref()
            .map(load_profile)
            .transpose()
            .map_err(ApiError::internal)?;
        if want.as_ref().map(|p| &p.name) != active_name.as_ref() {
            obs("h3_techniques");
            pipe.set_arm(want.as_ref())
                .map_err(|e| api_err("h3 technique profile", e))?;
        }
        Ok(Self {
            pipe,
            recipe: recipe.clone(),
        })
    }

    pub fn recipe(&self) -> &H3Recipe {
        &self.recipe
    }

    /// The request `fv-gpucheck h3 gen` would build for this job.
    pub fn request(job: &ResolvedJob) -> Result<H3Request, ApiError> {
        let mut req = H3Request::sized(
            job.prompt.clone(),
            job.height as usize,
            job.width as usize,
            job.num_frames as usize,
            job.seed,
        )
        .map_err(|e| ApiError::invalid(format!("h3: {e}")))?;
        // NVENC encodes the MP4 afterwards (design §0.1); the pipeline writes PNGs.
        req.mp4 = false;
        for (anchor, path) in &job.keyframes {
            match anchor {
                Anchor::First => req.first_image = Some(path.clone()),
                Anchor::Last => req.last_image = Some(path.clone()),
            }
        }
        if job.task == Task::Ref2V || !job.references.is_empty() {
            return Err(ApiError::unsupported(
                fastvideo_protocol::GapId::H3Ref2vaNotLoaded,
            ));
        }
        Ok(req)
    }

    /// Planned denoise steps (progress totals).
    pub(crate) fn planned_steps(&self) -> u32 {
        self.recipe.steps
    }

    pub(crate) fn generate(
        &self,
        job: &ResolvedJob,
        dir: &Path,
        hooks: Hooks<'_>,
    ) -> Result<JobMetrics, ApiError> {
        let req = Self::request(job)?;
        let out = self
            .pipe
            .generate_with_hooks(&req, dir, hooks)
            .map_err(|e| api_err("h3 generate", e))?;
        let t = &out.timings;
        let peak = out.memory.iter().map(|p| p.peak_used).max();
        Ok(JobMetrics {
            inference_s: Some(t.denoise_s),
            stage_durations: stages(&[
                ("text", t.text_s),
                ("refine", t.refine_s),
                ("denoise", t.denoise_s),
                ("audio_decode", t.audio_decode_s),
                ("video_decode", t.video_decode_s),
            ]),
            peak_memory_mb: peak.map(bytes_mb),
            build_rtf: None,
        })
    }
}
