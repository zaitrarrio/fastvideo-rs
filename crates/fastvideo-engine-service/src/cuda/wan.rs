//! Wan 2.1 / FastWan / Wan2.2 TI2V-5B (and the bounded SF-Wan clip) on
//! `WanPipeline` (WP-11): text encoder, DiT and decoders resident; a
//! `ResolvedJob` mapped to a `GenerateConfig`, where per-request `fps`,
//! steps, guidance and flow shift come for free (design §2.2).
//!
//! `[FASTVIDEO_WAN_VAE=<decoder>] fv-gpucheck [--vsa] wan gen --preset <p>
//! --steps <n> [--unipc --guidance <g>] --flow-shift <s>` with the same
//! weights, canvas, frames and seed produces the same frames.

use std::path::{Path, PathBuf};

use fastvideo_cudarc::{GenerateConfig, LoadParts, WanPipeline};
use fastvideo_protocol::{Anchor, ApiError, JobMetrics, ResolvedJob};

use super::caps::{WanDecoder, WanRecipe, WanSampler};
use super::output::{api_err, stages};
use fastvideo_cudarc::Hooks;

/// `fv-gpucheck wan gen`'s DMD timesteps for `steps`.
pub fn dmd_steps(steps: usize) -> Vec<i32> {
    let base = fastvideo_models::schedulers::FAST_WAN_1_3B_DMD_STEPS;
    if steps == base.len() {
        return base.to_vec();
    }
    (0..steps)
        .map(|i| 1000 - (i as i32 * 1000 / steps.max(1) as i32))
        .collect()
}

/// Sets `key` for the duration of a load (the Wan decoder choice is read
/// from the environment at `WanPipeline::load_with`), restoring it after.
struct EnvGuard {
    key: &'static str,
    old: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var_os(key);
        std::env::set_var(key, value);
        Self { key, old }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.old.take() {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

/// Loads a Wan pipeline with the recipe's decoder.
pub fn load_pipeline(
    recipe: &WanRecipe,
    obs: &mut dyn FnMut(&'static str),
) -> Result<WanPipeline, ApiError> {
    let _vae = (recipe.decoder != WanDecoder::Auto)
        .then(|| EnvGuard::set("FASTVIDEO_WAN_VAE", recipe.decoder.env_value()));
    let _tae = recipe
        .tae_dir
        .as_ref()
        .filter(|_| std::env::var_os("FASTVIDEO_TAE_DIR").is_none())
        .map(|d| EnvGuard::set("FASTVIDEO_TAE_DIR", &d.to_string_lossy()));
    obs("wan_pipeline");
    let pipe = WanPipeline::load_with(
        &recipe.weights,
        &recipe.preset,
        LoadParts { text_encoder: true },
    )
    .map_err(|e| api_err(&format!("wan load {}", recipe.weights.display()), e))?;
    if recipe.decoder == WanDecoder::Taehv && pipe.taehv().is_none() {
        return Err(ApiError::engine_failed(format!(
            "wan: recipe decodes with TAEHV but no taew2_* was found (tae dir {:?})",
            recipe.tae_dir
        )));
    }
    Ok(pipe)
}

/// One resident Wan pipeline.
pub struct WanModel {
    pipe: WanPipeline,
    recipe: WanRecipe,
    text_cache: Option<PathBuf>,
}

impl WanModel {
    pub fn load(
        recipe: &WanRecipe,
        text_cache: Option<&Path>,
        obs: &mut dyn FnMut(&'static str),
    ) -> Result<Self, ApiError> {
        Ok(Self {
            pipe: load_pipeline(recipe, obs)?,
            recipe: recipe.clone(),
            text_cache: text_cache.map(Path::to_path_buf),
        })
    }

    pub fn recipe(&self) -> &WanRecipe {
        &self.recipe
    }

    /// The config `fv-gpucheck wan gen` would build for this job.
    pub fn config(
        recipe: &WanRecipe,
        text_cache: Option<&Path>,
        job: &ResolvedJob,
    ) -> Result<GenerateConfig, ApiError> {
        let image = job
            .keyframes
            .iter()
            .find(|(a, _)| *a == Anchor::First)
            .map(|(_, p)| p.to_string_lossy().into_owned());
        super::validate::wan(recipe, job)?;
        let (dmd, steps, guidance) = match recipe.sampler {
            WanSampler::Dmd { steps } => (true, steps, 1.0),
            WanSampler::Unipc { steps, guidance } => (false, steps, guidance),
        };
        let steps = job.sampling.steps.unwrap_or(steps) as usize;
        let tokenizer = recipe.weights.join("tokenizer").join("tokenizer.json");
        Ok(GenerateConfig {
            prompt: job.prompt.clone(),
            negative_prompt: if job.negative_prompt.is_empty() {
                recipe.negative.clone()
            } else {
                job.negative_prompt.clone()
            },
            height: job.height as usize,
            width: job.width as usize,
            num_frames: job.num_frames as usize,
            num_inference_steps: steps,
            guidance_scale: job.sampling.guidance.unwrap_or(guidance),
            flow_shift: job.sampling.flow_shift.unwrap_or(recipe.flow_shift),
            is_dmd: dmd,
            dmd_steps: dmd.then(|| dmd_steps(steps)),
            seed: job.seed,
            tokenizer_path: Some(tokenizer.to_string_lossy().into_owned()),
            image_path: image,
            fps: job.fps,
            text_cache: text_cache.map(Path::to_path_buf),
            ..GenerateConfig::default()
        })
    }

    pub(crate) fn generate(
        &self,
        job: &ResolvedJob,
        dir: &Path,
        hooks: Hooks<'_>,
    ) -> Result<JobMetrics, ApiError> {
        let cfg = Self::config(&self.recipe, self.text_cache.as_deref(), job)?;
        run_pipeline(&self.pipe, &cfg, dir, hooks)
    }
}

/// Planned steps of a job on a recipe.
pub(crate) fn planned_steps(recipe: &WanRecipe, job: &ResolvedJob) -> u32 {
    job.sampling.steps.unwrap_or(recipe.sampler.steps())
}

/// One `generate_to_with_hooks` (no mp4: the engine sink encodes).
pub(crate) fn run_pipeline(
    pipe: &WanPipeline,
    cfg: &GenerateConfig,
    dir: &Path,
    hooks: Hooks<'_>,
) -> Result<JobMetrics, ApiError> {
    let out = pipe
        .generate_to_with_hooks(cfg, dir, false, hooks)
        .map_err(|e| api_err("wan generate", e))?;
    let t = &out.timings;
    Ok(JobMetrics {
        inference_s: Some(t.denoise_s),
        stage_durations: stages(&[
            ("text", t.text_s),
            ("denoise", t.denoise_s),
            ("video_decode", t.decode_s),
        ]),
        peak_memory_mb: None,
        build_rtf: None,
    })
}
