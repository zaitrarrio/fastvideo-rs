//! `CudaBackend`: one GPU's resident models behind `EngineBackend`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{ApiError, ModelCaps, ModelId, ResolvedJob};
use serde::{Deserialize, Serialize};

use super::caps::{CudaModel, CudaRecipe, ProcessPlan};
use super::caps::SfWanRecipe;
use super::causal::CausalDriver;
use super::h3::H3Model;
use super::ltx2::Ltx2Model;
use super::output::{deliver, remove_dir, Mp4Options};
use super::wan::WanModel;
use crate::backend::{
    BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, DeviceInfo, EngineBackend, LoadEvent,
    SessionId,
};
use crate::cancel::{OutputMode, StepControl};
use crate::caps::Recipe;

/// The MP4 encoder (`fastvideo_media::video::FfmpegH264`), serializable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mp4Encoder {
    /// `h264_nvenc` (design §0.1; production).
    #[default]
    Nvenc,
    /// `libx264`: only for boxes without NVENC in tests. Never deployed.
    Libx264CpuTest,
}

/// One GPU's CUDA backend.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CudaBackendConfig {
    /// CUDA ordinal.
    pub device: u32,
    /// The models this GPU serves (`resident` ones load before readiness).
    pub models: Vec<CudaModel>,
    /// Frames-mode scratch: `<work_dir>/<uuid>/` holds a job's PNGs while
    /// they are read back.
    pub work_dir: PathBuf,
    /// Prompt-conditioning cache directory (shared by the families; each
    /// keys its own entries). `None` encodes every prompt.
    pub text_cache: Option<PathBuf>,
    /// Keep the pipelines' `frame-NNN.png` + `audio.wav` next to the MP4
    /// (identity checks); removed otherwise.
    pub keep_frames: bool,
    pub encoder: Mp4Encoder,
    /// NVENC constant quality (`-cq`).
    pub quality: u8,
}

impl CudaBackendConfig {
    pub fn new(device: u32, models: Vec<CudaModel>) -> Self {
        Self {
            device,
            models,
            work_dir: std::env::temp_dir().join("fv-cuda-work"),
            text_cache: None,
            keep_frames: false,
            encoder: Mp4Encoder::Nvenc,
            quality: 19,
        }
    }
}

enum Loaded {
    H3(Box<H3Model>),
    Ltx2(Box<Ltx2Model>),
    Wan(Box<WanModel>),
    SfWan(Box<SfWan>),
}

/// A resident SF-Wan pipeline: bounded clips through `WanPipeline`, causal
/// sessions through WP-15's [`CausalDriver`].
struct SfWan {
    pipe: &'static fastvideo_cudarc::WanPipeline,
    recipe: SfWanRecipe,
    driver: CausalDriver,
}

/// The rollout settings of a recipe (everything but prompt, canvas, seed).
fn rollout_base(r: &SfWanRecipe, text_cache: Option<&Path>) -> fastvideo_cudarc::wan::stream::RolloutConfig {
    use fastvideo_cudarc::wan::stream::{PromptSwitch, RolloutConfig, RopePolicy};
    let tok = r.wan.weights.join("tokenizer").join("tokenizer.json");
    RolloutConfig {
        flow_shift: r.wan.flow_shift,
        local_attn_frames: r.local_attn_frames as usize,
        sink_frames: r.sink_frames as usize,
        rope: RopePolicy::Relativistic,
        prompt_switch: PromptSwitch::Keep,
        rgb8: true,
        tokenizer_path: tok.is_file().then(|| tok.to_string_lossy().into_owned()),
        text_cache: text_cache.map(Path::to_path_buf),
        ..RolloutConfig::default()
    }
}

/// `EngineBackend` over the fastvideo-cudarc pipelines.
pub struct CudaBackend {
    cfg: CudaBackendConfig,
    plan: ProcessPlan,
    models: BTreeMap<ModelId, CudaModel>,
    loaded: BTreeMap<ModelId, Loaded>,
    /// SF-Wan pipelines are borrowed by their rollouts for the process
    /// lifetime; kept here across unload so a reload reuses them.
    causal_pipes: BTreeMap<ModelId, &'static fastvideo_cudarc::WanPipeline>,
    /// Open causal sessions and the model serving each.
    causal_owner: BTreeMap<SessionId, ModelId>,
    device_up: bool,
    info: DeviceInfo,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device", &self.cfg.device)
            .field("models", &self.models.keys().collect::<Vec<_>>())
            .field("loaded", &self.loaded.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Installs the plan's technique profile and env, before any `FASTVIDEO_*`
/// setting is read. A process whose profile is already installed (a binary's
/// `--techniques`) must carry the same load-time settings.
pub fn install_process_plan(plan: &ProcessPlan) -> Result<(), ApiError> {
    use fastvideo_models::techniques::settings;
    for (k, v) in &plan.env {
        match std::env::var(k) {
            Ok(have) if have != *v => {
                return Err(ApiError::internal(format!(
                    "{k}={have} in the environment, but the served models need {k}={v}"
                )))
            }
            Ok(_) => {}
            Err(_) => std::env::set_var(k, v),
        }
    }
    let installed = match &plan.profile {
        Some(p) => settings::install_file(Path::new(p)).map(|_| ()),
        None => Err(String::new()),
    };
    if installed.is_ok() {
        tracing::info!(profile = ?plan.profile, "technique profile installed");
        return Ok(());
    }
    let active = settings::active();
    let have: BTreeMap<String, String> = active
        .settings
        .iter()
        .map(|(k, v, _)| (k.to_owned(), v.to_owned()))
        .collect();
    if have != plan.settings {
        return Err(ApiError::internal(format!(
            "technique settings already installed ({:?}: {have:?}) differ from what the served models need ({:?}: {:?}); \
             start the process with that profile or none",
            active.path, plan.profile, plan.settings
        )));
    }
    Ok(())
}

impl CudaBackend {
    /// Checks the model set ([`ProcessPlan`]) and installs its process-wide
    /// settings. Call at process start, before any other CUDA/pipeline use.
    /// Nothing loads here: the executor loads the resident models (warm pool).
    pub fn new(cfg: CudaBackendConfig) -> Result<Self, ApiError> {
        let plan = ProcessPlan::for_models(&cfg.models).map_err(ApiError::internal)?;
        install_process_plan(&plan)?;
        let mut models = BTreeMap::new();
        for m in &cfg.models {
            if models.insert(m.id.clone(), m.clone()).is_some() {
                return Err(ApiError::internal(format!("model `{}` listed twice", m.id)));
            }
        }
        let info = DeviceInfo {
            index: cfg.device,
            name: format!("cuda:{}", cfg.device),
            total_memory_mb: 0,
        };
        Ok(Self {
            cfg,
            plan,
            models,
            loaded: BTreeMap::new(),
            causal_pipes: BTreeMap::new(),
            causal_owner: BTreeMap::new(),
            device_up: false,
            info,
        })
    }

    pub fn plan(&self) -> &ProcessPlan {
        &self.plan
    }

    pub fn config(&self) -> &CudaBackendConfig {
        &self.cfg
    }

    /// Creates the CUDA context on the calling (executor) thread once.
    fn device(&mut self) -> Result<(), ApiError> {
        if self.device_up {
            return Ok(());
        }
        let idx = self.cfg.device as usize;
        if idx == 0 {
            fastvideo_cudarc::resolve_device("cuda:0")
                .map_err(|e| ApiError::engine_failed(format!("cuda:0: {e}")))?;
        } else {
            let dev = fastvideo_cudarc::wan::device::device_for_index(idx)
                .map_err(|e| ApiError::engine_failed(format!("cuda:{idx}: {e}")))?;
            fastvideo_cudarc::wan::device::set_thread_device(Some(dev));
        }
        if let Some(dev) = fastvideo_cudarc::wan::device::global_device() {
            if let Ok(name) = dev.ctx.name() {
                self.info.name = name;
            }
        }
        if let Some((_, total)) = fastvideo_cudarc::wan::device::free_memory() {
            self.info.total_memory_mb = total >> 20;
        }
        self.device_up = true;
        Ok(())
    }

    fn model(&self, id: &ModelId) -> Result<&CudaModel, ApiError> {
        self.models
            .get(id)
            .ok_or_else(|| ApiError::invalid(format!("model `{id}` is not served by this GPU")))
    }

    fn mp4_options(&self) -> Mp4Options {
        Mp4Options {
            encoder: match self.cfg.encoder {
                Mp4Encoder::Nvenc => FfmpegH264::Nvenc,
                Mp4Encoder::Libx264CpuTest => FfmpegH264::Libx264CpuTest,
            },
            quality: self.cfg.quality,
            keep_frames: self.cfg.keep_frames,
        }
    }
}

impl EngineBackend for CudaBackend {
    fn device(&self) -> DeviceInfo {
        self.info.clone()
    }

    fn caps(&self) -> Vec<ModelCaps> {
        self.cfg.models.iter().map(CudaModel::caps).collect()
    }

    fn recipe(&self, model: &ModelId) -> Recipe {
        self.models
            .get(model)
            .map(CudaModel::describe)
            .unwrap_or_default()
    }

    fn load(&mut self, model: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError> {
        if self.loaded.contains_key(model) {
            return Ok(());
        }
        let m = self.model(model)?.clone();
        obs(LoadEvent::Stage("device"));
        self.device()?;
        obs(LoadEvent::Progress { done: 0, total: 1 });
        let t0 = std::time::Instant::now();
        // One conditioning-cache directory per family (each keys its own).
        let fam = match m.family() {
            fastvideo_protocol::Family::H3 => "h3",
            fastvideo_protocol::Family::Ltx2 => "ltx2",
            _ => "wan",
        };
        let cache = self.cfg.text_cache.as_ref().map(|c| c.join(fam));
        let cache = cache.as_deref();
        let mut stage = |s: &'static str| obs(LoadEvent::Stage(s));
        let loaded = match &m.recipe {
            CudaRecipe::H3(r) => Loaded::H3(Box::new(H3Model::load(r, cache, &mut stage)?)),
            CudaRecipe::Ltx2(r) => Loaded::Ltx2(Box::new(Ltx2Model::load(r, cache, &mut stage)?)),
            CudaRecipe::Wan(r) => Loaded::Wan(Box::new(WanModel::load(r, cache, &mut stage)?)),
            CudaRecipe::SfWan(r) => {
                let pipe = match self.causal_pipes.get(model) {
                    Some(p) => *p,
                    None => {
                        let p: &'static _ =
                            Box::leak(Box::new(super::wan::load_pipeline(&r.wan, &mut stage)?));
                        self.causal_pipes.insert(model.clone(), p);
                        p
                    }
                };
                Loaded::SfWan(Box::new(SfWan {
                    pipe,
                    recipe: r.clone(),
                    driver: CausalDriver::new(pipe, rollout_base(r, cache)),
                }))
            }
        };
        obs(LoadEvent::Progress { done: 1, total: 1 });
        tracing::info!(model = %model, seconds = t0.elapsed().as_secs_f64(), "model resident");
        self.loaded.insert(model.clone(), loaded);
        Ok(())
    }

    fn unload(&mut self, model: &ModelId) {
        if self.loaded.remove(model).is_some() {
            if self.causal_pipes.contains_key(model) {
                tracing::warn!(model = %model, "SF-Wan pipeline stays resident (borrowed by rollouts)");
            }
            let _ = fastvideo_cudarc::wan::device::trim_pool();
        }
    }

    fn generate(
        &mut self,
        job: &ResolvedJob,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<ClipOutput, ApiError> {
        ctl.check()?;
        // Where the pipeline's own files go: `audio.wav`, and the PNG frames
        // when `keep_frames` (E2 delivers the frames in memory either way).
        let work = match &ctl.mode {
            OutputMode::File { dir } => dir.join("frames"),
            OutputMode::Frames => self.cfg.work_dir.join(uuid::Uuid::new_v4().to_string()),
        };
        std::fs::create_dir_all(&work)
            .map_err(|e| ApiError::internal(format!("{}: {e}", work.display())))?;
        let opts = self.mp4_options();
        let text_cache = self.cfg.text_cache.clone();
        let loaded = self
            .loaded
            .get_mut(&job.model)
            .ok_or_else(|| ApiError::loading(format!("model `{}` is not resident", job.model)))?;
        let r = match loaded {
            Loaded::H3(m) => {
                let planned = m.planned_steps();
                deliver(job, ctl, out, &opts, Some(planned), |h| m.generate(job, &work, h))
            }
            Loaded::Ltx2(m) => {
                let planned = m.planned_steps();
                deliver(job, ctl, out, &opts, Some(planned), |h| m.generate(job, &work, h))
            }
            Loaded::Wan(m) => {
                let planned = super::wan::planned_steps(m.recipe(), job);
                deliver(job, ctl, out, &opts, Some(planned), |h| m.generate(job, &work, h))
            }
            Loaded::SfWan(m) => {
                // The bounded SF-Wan clip (`wan gen --preset sf_wan_t2v_1_3b`).
                let cfg = WanModel::config(&m.recipe.wan, text_cache.as_ref().map(|c| c.join("wan")).as_deref(), job)?;
                let planned = super::wan::planned_steps(&m.recipe.wan, job);
                let pipe = m.pipe;
                deliver(job, ctl, out, &opts, Some(planned), |h| {
                    super::wan::run_pipeline(pipe, &cfg, &work, h)
                })
            }
        };
        let keep = self.cfg.keep_frames && matches!(ctl.mode, OutputMode::File { .. }) && r.is_ok();
        if !keep {
            remove_dir(&work);
        }
        r
    }

    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError> {
        match self.loaded.get_mut(&spec.model) {
            Some(Loaded::SfWan(m)) => {
                m.driver.open(s, spec)?;
                self.causal_owner.insert(s, spec.model.clone());
                Ok(())
            }
            Some(_) => Err(ApiError::invalid(format!(
                "model `{}` is not a causal model",
                spec.model
            ))),
            None => Err(ApiError::loading(format!(
                "model `{}` is not resident",
                spec.model
            ))),
        }
    }

    fn causal_block(
        &mut self,
        s: SessionId,
        input: &BlockInput,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<BlockStats, ApiError> {
        let model = self
            .causal_owner
            .get(&s)
            .ok_or_else(|| ApiError::invalid(format!("causal session {s} is not open")))?;
        match self.loaded.get_mut(model) {
            Some(Loaded::SfWan(m)) => m.driver.block(s, input, out, ctl),
            _ => Err(ApiError::engine_failed(format!("causal model `{model}` is not resident"))),
        }
    }

    fn causal_close(&mut self, s: SessionId) {
        if let Some(model) = self.causal_owner.remove(&s) {
            if let Some(Loaded::SfWan(m)) = self.loaded.get_mut(&model) {
                m.driver.close(s);
            }
        }
    }
}
