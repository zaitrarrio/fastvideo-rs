//! The engine-side job checks: what each CUDA pipeline refuses before it
//! starts, as plain functions of the recipe and the `ResolvedJob`. The CUDA
//! backend calls them on every job ([`super::h3`], [`super::ltx2`],
//! [`super::wan`]), and the fake backend calls the same ones when it serves
//! the real catalog ([`crate::FakeConfig::validator`]), so a test on the fake
//! engine fails a job exactly where a GPU server would.
//!
//! Built without the `cuda` feature. Device-dependent refusals (the H3
//! 1080P memory check against free GPU memory) are not here.

use fastvideo_models::h3::config::H3Geometry;
use fastvideo_protocol::{ApiError, GapId, MediaKind, ResolvedJob, Task};

use super::caps::{CudaModel, CudaRecipe, H3Recipe, Ltx2Recipe, LtxVersion, WanRecipe};

/// Every check the CUDA backend makes on `job` for a model of `recipe`
/// before generating. `Err` is what the engine fails the job with.
pub fn check_job(recipe: &CudaRecipe, job: &ResolvedJob) -> Result<(), ApiError> {
    match recipe {
        CudaRecipe::H3(r) => h3(r, job).map(|_| ()),
        CudaRecipe::Ltx2(r) => ltx2(r, job),
        CudaRecipe::Wan(r) => wan(r, job),
        CudaRecipe::SfWan(r) => wan(&r.wan, job),
    }
}

impl CudaModel {
    /// [`check_job`] for this model.
    pub fn check_job(&self, job: &ResolvedJob) -> Result<(), ApiError> {
        check_job(&self.recipe, job)
    }
}

/// H3: `H3Request::sized` / `sized_1080p` (the canvas check, 768x1344 or
/// with the 1080P tier 1088x1920, and the 17n+5 grid within 4..15 s), and
/// the transformer the pipeline loaded (`transformer_ref/` serves
/// reference-to-video only).
pub fn h3(r: &H3Recipe, job: &ResolvedJob) -> Result<H3Geometry, ApiError> {
    let (h, w, n) = (job.height as usize, job.width as usize, job.num_frames as usize);
    let g = if r.hd_1080p { H3Geometry::checked_1080p(h, w, n) } else { H3Geometry::checked(h, w, n) }
        .map_err(|e| ApiError::invalid(format!("h3: {e}")))?;
    let refs = job.task == Task::Ref2V || !job.references.is_empty();
    if job.task == Task::Ref2V && job.references.is_empty() {
        return Err(ApiError::invalid_param("references", "reference-to-video takes one or more references"));
    }
    if refs != r.ref2va {
        return Err(if refs {
            ApiError::unsupported(GapId::H3Ref2vaNotLoaded)
        } else {
            ApiError::invalid_param("task", "this H3 model serves reference-to-video only")
        });
    }
    Ok(g)
}

/// LTX-2: the task the checkpoint serves (2.3: text-to-video only; the
/// IC-LoRA model: reference-to-video with one reference sheet), then
/// `Ltx2Request::validate`'s geometry (multiples of 64 two-stage, 8k+1).
pub fn ltx2(r: &Ltx2Recipe, job: &ResolvedJob) -> Result<(), ApiError> {
    ltx2_reference(job, r.ic_lora.is_some())?;
    match job.task {
        Task::Ref2V => {}
        Task::T2V if job.keyframes.is_empty() => {}
        Task::I2V | Task::Keyframes if r.version == LtxVersion::V25 => {}
        Task::Keyframes => return Err(ApiError::unsupported(GapId::LtxKeyframes)),
        _ => return Err(ApiError::unsupported(GapId::Ltx25I2V)),
    }
    if job.fps == 0 || job.prompt.trim().is_empty() {
        return Err(ApiError::invalid("ltx2: needs a positive frame rate and a non-empty prompt"));
    }
    fastvideo_models::ltx2::config::check_geometry(
        job.height as usize,
        job.width as usize,
        job.num_frames as usize,
        r.two_stage,
    )
    .map_err(ApiError::invalid)
}

/// The LTX reference sheet of a `Ref2V` job: exactly one image reference
/// on a model loaded with the IC-LoRA. `None` for other tasks.
pub fn ltx2_reference(job: &ResolvedJob, ic_lora: bool) -> Result<Option<&std::path::Path>, ApiError> {
    if job.task != Task::Ref2V {
        return Ok(None);
    }
    if !ic_lora {
        return Err(ApiError::invalid_param(
            "task",
            "reference-to-video needs an LTX model loaded with the IC-LoRA",
        ));
    }
    let mut images = job.references.iter().filter(|(k, _)| *k == MediaKind::Image);
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
    Ok(Some(path.as_path()))
}

/// Wan: image-to-video only on an I2V model with a first frame, and the
/// preset's exact geometry (`fastvideo_models::wan::config::check_geometry`:
/// the VAE x patch multiple, 4k+1 frames, whole causal blocks).
pub fn wan(r: &WanRecipe, job: &ResolvedJob) -> Result<(), ApiError> {
    let first = job.keyframes.iter().any(|(a, _)| *a == fastvideo_protocol::Anchor::First);
    if job.task == Task::I2V && (!first || !r.i2v) {
        return Err(ApiError::invalid("wan: image-to-video needs a first-frame image on an I2V model"));
    }
    fastvideo_models::wan::config::check_geometry(
        &r.preset,
        job.height as usize,
        job.width as usize,
        job.num_frames as usize,
    )
    .map_err(ApiError::invalid)
}
