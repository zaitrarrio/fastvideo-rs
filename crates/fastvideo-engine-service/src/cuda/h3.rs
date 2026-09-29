//! H3 / FastH3 / Sol-H3 on `H3Pipeline` (WP-11): load once with the
//! recipe's options, switch to the recipe's technique profile (`set_arm`)
//! when it is not the process's, and map a `ResolvedJob` to an `H3Request`.
//!
//! `fv-gpucheck h3 gen --h3-recipe <r> [--techniques <profile>] [--dense]`
//! with the same weights, canvas, frames and seed produces the same frames.

use std::path::Path;

use fastvideo_cudarc::h3::pipeline::{H3Pipeline, H3PipelineOptions, H3Request, TextEncoderChoice};
use fastvideo_models::h3::reference::{H3ReferenceSpec, ReferenceKind};
use fastvideo_protocol::{Anchor, ApiError, JobMetrics, MediaKind, ResolvedJob, Task};

use super::caps::{load_profile, H3Recipe};
use super::output::{api_err, bytes_mb, stages};
use fastvideo_cudarc::Hooks;

/// A 1080P-tier job (above the trained 768 x 1344 budget) starts only when
/// the device can hold its working set beside the loaded weights: the
/// measured-rate estimate (`measured_working_bytes`) against the driver's
/// free memory plus what the pool has cached but not in use. Otherwise it is
/// refused before any work, naming the sizes (the long 1080P clips on a card
/// that also keeps the text encoder resident).
fn check_hd_memory(req: &H3Request) -> Result<(), ApiError> {
    use fastvideo_cudarc::wan::device::{free_memory, pool_usage};
    use fastvideo_models::h3::config::H3Geometry;
    let g = H3Geometry::new(req.height, req.width, req.num_frames)
        .map_err(|e| ApiError::invalid(format!("h3: {e}")))?;
    let need = fastvideo_models::h3::memory::measured_working_bytes(&g);
    let Some((free, _)) = free_memory() else {
        return Ok(());
    };
    let cached = pool_usage().map_or(0, |p| p.reserved.saturating_sub(p.used));
    let available = free + cached;
    if need <= available {
        return Ok(());
    }
    let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
    let seconds = req.num_frames as f64 / fastvideo_models::h3::config::H3_FPS as f64;
    Err(ApiError::unsupported_msg(
        fastvideo_protocol::GapId::H3Refine1080P,
        format!(
            "1080P at {}x{} for {seconds:.1} s needs about {:.1} GiB of GPU memory beside the loaded model; {:.1} GiB is free on this server. Use a shorter duration or 768P",
            req.width,
            req.height,
            gib(need),
            gib(available)
        ),
    ))
}

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
            // The AdaLN table is the base DiT's; a Ref2VA DiT builds its own.
            adaln_cache: recipe.adaln_cache.clone().filter(|_| !recipe.ref2va),
            text_root: recipe.text_weights.clone(),
            text_cache: text_cache.map(Path::to_path_buf),
            text_encoder: TextEncoderChoice::parse(&recipe.text_encoder)
                .map_err(ApiError::internal)?,
            taeh3: recipe.taeh3.clone(),
            recipe: Some(recipe.recipe.clone()),
            ref2va: recipe.ref2va,
            ref_root: recipe.ref_weights.clone(),
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

    /// The request `fv-gpucheck h3 gen` would build for this job. `hd_1080p`:
    /// the model serves the native 1080P tier, so canvases up to 1088 x 1920
    /// are admitted (`fv-gpucheck h3 gen --oversize-canvas`).
    pub fn request(job: &ResolvedJob, hd_1080p: bool) -> Result<H3Request, ApiError> {
        let (h, w, n) = (job.height as usize, job.width as usize, job.num_frames as usize);
        let mut req = if hd_1080p {
            H3Request::sized_1080p(job.prompt.clone(), h, w, n, job.seed)
        } else {
            H3Request::sized(job.prompt.clone(), h, w, n, job.seed)
        }
        .map_err(|e| ApiError::invalid(format!("h3: {e}")))?;
        // NVENC encodes the MP4 afterwards (design §0.1); the pipeline writes PNGs.
        req.mp4 = false;
        for (anchor, path) in &job.keyframes {
            match anchor {
                Anchor::First => req.first_image = Some(path.clone()),
                Anchor::Last => req.last_image = Some(path.clone()),
            }
        }
        // Ordered Ref2VA references, in the client's order (prompts say
        // "Image 1", "Video 1"; research-minimax-fastvideo.md §3).
        req.references = job
            .references
            .iter()
            .map(|(kind, path)| H3ReferenceSpec {
                path: path.clone(),
                kind: match kind {
                    MediaKind::Image => ReferenceKind::Image,
                    MediaKind::Video => ReferenceKind::Video,
                    MediaKind::Audio => ReferenceKind::Audio,
                },
            })
            .collect();
        if job.task == Task::Ref2V && req.references.is_empty() {
            return Err(ApiError::invalid_param(
                "references",
                "reference-to-video takes one or more references",
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
        super::validate::h3(&self.recipe, job)?;
        let req = Self::request(job, self.recipe.hd_1080p)?;
        if req.height * req.width > fastvideo_models::h3::config::H3_MAX_PIXELS {
            check_hd_memory(&req)?;
        }
        // A pipeline serves either `transformer/` or `transformer_ref/`; the
        // caps route each task to the right one, so a mismatch here is a
        // routing bug, not a client error.
        if req.is_ref2va() != self.recipe.ref2va {
            return Err(if req.is_ref2va() {
                ApiError::unsupported(fastvideo_protocol::GapId::H3Ref2vaNotLoaded)
            } else {
                ApiError::invalid_param(
                    "task",
                    "this H3 model serves reference-to-video only",
                )
            });
        }
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
