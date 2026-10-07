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
    /// Fragmented, append-only MP4 output (docs/serve/dispatch-do-family.md §7.2).
    #[serde(default)]
    pub mp4_fragmented: bool,
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
            mp4_fragmented: false,
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
    let base = RolloutConfig {
        flow_shift: r.wan.flow_shift,
        local_attn_frames: r.local_attn_frames as usize,
        sink_frames: r.sink_frames as usize,
        // The rebased sink: stable over long rollouts and 0.1 s per block
        // cheaper than relativistic, which flickers and collapses after
        // about a minute (docs/ports/wan.md, long-run stability).
        rope: RopePolicy::RebasedSink,
        prompt_switch: PromptSwitch::Keep,
        rgb8: true,
        tokenizer_path: tok.is_file().then(|| tok.to_string_lossy().into_owned()),
        text_cache: text_cache.map(Path::to_path_buf),
        ..RolloutConfig::default()
    };
    match longlive_config(r) {
        Some(ll) => ll.rollout(base),
        None => base,
    }
}

/// The LongLive settings of a recipe, its window and sink taken from the
/// recipe (12 and 3 for the release).
fn longlive_config(r: &SfWanRecipe) -> Option<fastvideo_cudarc::wan::longlive::LongLiveConfig> {
    let l = r.longlive.as_ref()?;
    Some(fastvideo_cudarc::wan::longlive::LongLiveConfig {
        local_attn_frames: r.local_attn_frames as usize,
        sink_frames: r.sink_frames as usize,
        global_sink: l.global_sink,
        recache: l.recache,
        relative_rope: l.relative_rope,
        ..fastvideo_cudarc::wan::longlive::LongLiveConfig::interactive()
    })
}

/// The LongLive transformer of a recipe (renamed, LoRA merged), if any.
fn longlive_dit(
    r: &SfWanRecipe,
) -> Result<Option<fastvideo_cudarc::wan::weights::WeightMap>, ApiError> {
    let (Some(l), Some(cfg)) = (r.longlive.as_ref(), longlive_config(r)) else {
        return Ok(None);
    };
    let mut w = fastvideo_cudarc::wan::longlive::LongLiveWeights::in_dir(&l.weights, &cfg);
    if !l.lora {
        w.lora = None;
    } else if w.lora.is_none() {
        return Err(ApiError::engine_failed(format!(
            "longlive: {} has no lora.safetensors (set lora: false for the base generator)",
            l.weights.display()
        )));
    }
    let (map, _) = fastvideo_cudarc::wan::longlive::load_transformer_map(&w)
        .map_err(|e| ApiError::engine_failed(format!("longlive weights {}: {e}", l.weights.display())))?;
    Ok(Some(map))
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
    /// Models this GPU cannot run (the startup capability check,
    /// [`crate::device`]) and why: their load fails at once with the reason.
    unsupported: BTreeMap<ModelId, String>,
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
    pub fn new(mut cfg: CudaBackendConfig) -> Result<Self, ApiError> {
        // The caps are published before the executor creates the context:
        // decide the H3 1080P tier on the device's total memory now.
        if cfg.models.iter().any(|m| matches!(&m.recipe, CudaRecipe::H3(r) if r.hd_1080p)) {
            let total =
                fastvideo_cudarc::wan::device::device_total_memory(cfg.device as usize);
            match super::caps::gate_h3_1080p(&mut cfg.models, total) {
                Some(off) => tracing::warn!("{off}"),
                None => tracing::info!(
                    device_gib = total.map(|t| t as f64 / f64::from(1u32 << 30)),
                    "H3 1080P tier offered"
                ),
            }
        }
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
        // The capability check: a model this GPU cannot run (compute
        // capability, FP8/NVFP4, memory) is failed with the reason instead
        // of loading and reporting ready.
        let mut unsupported = BTreeMap::new();
        if let Some((name, (major, minor))) = fastvideo_cudarc::wan::device::device_identity(cfg.device as usize) {
            let total = fastvideo_cudarc::wan::device::device_total_memory(cfg.device as usize);
            let dev = crate::device::DeviceProfile::new(name, (major.max(0) as u32, minor.max(0) as u32), total);
            for m in &cfg.models {
                if let Err(why) = m.requirements().check(m.id.as_str(), &dev) {
                    tracing::error!(model = %m.id, "{why}");
                    unsupported.insert(m.id.clone(), why);
                }
            }
        }
        Ok(Self {
            cfg,
            plan,
            models,
            loaded: BTreeMap::new(),
            causal_pipes: BTreeMap::new(),
            causal_owner: BTreeMap::new(),
            device_up: false,
            info,
            unsupported,
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

    /// Blocking warm-up (`warmup = "blocking"`, the behaviour before fast
    /// boot B): one image-to-video and one text-to-video job at the default
    /// canvas and length, through the same `generate` a request takes (MP4
    /// encode included), before the model is reported ready. The first
    /// request then does not pay kernel compilation, plan search and
    /// allocator growth. A failure is logged, not fatal: the model still
    /// serves. The default (background) runs the same jobs through
    /// [`EngineBackend::warmup_run`] after readiness instead.
    fn warmup(&mut self, m: &CudaModel) {
        let t0 = std::time::Instant::now();
        let mut runs = Vec::new();
        let mut failed = None;
        for name in Self::warmup_names(m) {
            match self.warmup_one(m, &name, &crate::cancel::CancelToken::new()) {
                Ok(what) => runs.push(what),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        match failed {
            None => tracing::info!(
                model = %m.id,
                seconds = t0.elapsed().as_secs_f64(),
                runs = %runs.join(", "),
                "warmup done"
            ),
            Some(e) => tracing::warn!(model = %m.id, error = %e, "warmup failed; serving without it"),
        }
    }

    /// The warm-up runs of `m`, in order: `i2v` then `t2v`, each when the
    /// model takes that task (a Ref2VA-only model takes neither).
    fn warmup_names(m: &CudaModel) -> Vec<String> {
        use fastvideo_protocol::Task;
        let caps = m.caps();
        [("i2v", Task::I2V), ("t2v", Task::T2V)]
            .into_iter()
            .filter(|(_, t)| caps.supports(*t))
            .map(|(n, _)| n.to_owned())
            .collect()
    }

    /// One warm-up generation (`i2v` or `t2v`) under `cancel`, in a scratch
    /// directory removed afterwards. A fixed prompt and seed at the default
    /// canvas and length: the same work as the first real request.
    fn warmup_one(
        &mut self,
        m: &CudaModel,
        name: &str,
        cancel: &crate::cancel::CancelToken,
    ) -> Result<String, ApiError> {
        let dir = self.cfg.work_dir.join(format!("warmup-{}", uuid::Uuid::new_v4()));
        let r = self.warmup_job(m, name, &dir, cancel);
        remove_dir(&dir);
        r
    }

    fn warmup_job(
        &mut self,
        m: &CudaModel,
        name: &str,
        dir: &Path,
        cancel: &crate::cancel::CancelToken,
    ) -> Result<String, ApiError> {
        use fastvideo_protocol::{
            canvas_for_aspect, Anchor, AudioPlan, PostProcess, SamplingOverrides, Task,
        };
        let caps = m.caps();
        let short = caps.canvas.short_edges.first().copied().unwrap_or(768);
        let (width, height) = canvas_for_aspect(&caps.canvas, 16.0 / 9.0, short);
        std::fs::create_dir_all(dir)
            .map_err(|e| ApiError::internal(format!("{}: {e}", dir.display())))?;
        let (task, keyframes) = match name {
            "i2v" => {
                // A smooth gradient: something for the vision tower and the
                // keyframe encoder to look at.
                let image = dir.join("first.png");
                image::RgbImage::from_fn(width, height, |x, y| {
                    image::Rgb([
                        (x * 255 / width.max(1)) as u8,
                        (y * 255 / height.max(1)) as u8,
                        128,
                    ])
                })
                .save(&image)
                .map_err(|e| ApiError::internal(format!("{}: {e}", image.display())))?;
                (Task::I2V, vec![(Anchor::First, image)])
            }
            "t2v" => (Task::T2V, Vec::new()),
            other => return Err(ApiError::internal(format!("unknown warm-up run `{other}`"))),
        };
        let audio = match &caps.audio {
            Some(a) if !a.via_sidecar => AudioPlan::Native {
                rate: a.native_rate,
                channels: a.channels,
            },
            _ => AudioPlan::None,
        };
        let j = ResolvedJob {
            model: m.id.clone(),
            task,
            prompt: "A calm lake at sunrise, mist over the water, birds calling.".into(),
            negative_prompt: String::new(),
            seed: 0,
            width,
            height,
            num_frames: caps.frames.default,
            fps: caps.fps.default,
            keyframes,
            references: Vec::new(),
            audio_in: None,
            audio,
            post: PostProcess {
                crop: None,
                drop_audio: false,
            },
            sampling: SamplingOverrides::default(),
            tier: caps.tier,
            recipe: caps.recipe.clone(),
            edit: None,
        };
        let t = std::time::Instant::now();
        let out = dir.join(name);
        std::fs::create_dir_all(&out)
            .map_err(|e| ApiError::internal(format!("{}: {e}", out.display())))?;
        let ctl = StepControl::detached(cancel.clone(), OutputMode::File { dir: out });
        self.generate(&j, &mut crate::backend::NullSink, &ctl)?;
        Ok(format!(
            "{name} {width}x{height}x{} {:.1}s",
            j.num_frames,
            t.elapsed().as_secs_f64()
        ))
    }

    fn mp4_options(&self) -> Mp4Options {
        Mp4Options {
            encoder: match self.cfg.encoder {
                Mp4Encoder::Nvenc => FfmpegH264::Nvenc,
                Mp4Encoder::Libx264CpuTest => FfmpegH264::Libx264CpuTest,
            },
            quality: self.cfg.quality,
            fragmented: self.cfg.mp4_fragmented,
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
        if let Some(why) = self.unsupported.get(model) {
            return Err(ApiError::engine_failed(why.clone()));
        }
        if !m.weights().is_dir() {
            return Err(ApiError::engine_failed(format!(
                "model `{model}`: weight directory {} does not exist (set `weights` in [[models]] or \
                 FV_WEIGHTS; is the weight volume mounted?)",
                m.weights().display()
            )));
        }
        obs(LoadEvent::Stage("device"));
        self.device()?;
        // Co-residency check: the DiT alone must fit in what is free now
        // (the models already resident hold the rest).
        if let (Some(need), Some((free, total))) = (m.dit_bytes(), fastvideo_cudarc::wan::device::free_memory()) {
            if need > free {
                let gb = |b: u64| b as f64 / 1e9;
                return Err(ApiError::engine_failed(format!(
                    "model `{model}`: its DiT needs ~{:.1} GB but only {:.1} of {:.1} GB are free on cuda:{} \
                     with {:?} resident; serve it from a separate process/GPU, or set `resident = false` \
                     and `[engine] swap = true`",
                    gb(need),
                    gb(free),
                    gb(total),
                    self.cfg.device,
                    self.loaded.keys().map(ModelId::as_str).collect::<Vec<_>>()
                )));
            }
            tracing::info!(model = %model, dit_gb = need as f64 / 1e9, free_gb = free as f64 / 1e9, "loading");
        }
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
                        let dit = longlive_dit(r)?;
                        let p: &'static _ = Box::leak(Box::new(super::wan::load_pipeline_with_dit(
                            &r.wan, &mut stage, dit,
                        )?));
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
        if matches!(&m.recipe, CudaRecipe::H3(r) if super::caps::WarmupMode::of(r) == super::caps::WarmupMode::Blocking) {
            obs(LoadEvent::Stage("warmup"));
            self.warmup(&m);
        }
        Ok(())
    }

    fn warmup_pending(&self, model: &ModelId) -> Vec<String> {
        match self.models.get(model) {
            Some(m @ CudaModel { recipe: CudaRecipe::H3(r), .. })
                if self.loaded.contains_key(model)
                    && super::caps::WarmupMode::of(r) == super::caps::WarmupMode::Background =>
            {
                Self::warmup_names(m)
            }
            _ => Vec::new(),
        }
    }

    fn warmup_run(
        &mut self,
        model: &ModelId,
        name: &str,
        cancel: &crate::cancel::CancelToken,
    ) -> Result<String, ApiError> {
        let m = self.model(model)?.clone();
        self.warmup_one(&m, name, cancel)
    }

    fn unload(&mut self, model: &ModelId) {
        if self.loaded.remove(model).is_some() {
            if self.causal_pipes.contains_key(model) {
                tracing::warn!(model = %model, "SF-Wan pipeline stays resident (borrowed by rollouts)");
            }
            let _ = fastvideo_cudarc::wan::device::trim_pool();
        }
    }

    fn marks(&mut self, cap: usize) -> Option<Box<dyn fastvideo_trace::MarkPool>> {
        let m = fastvideo_cudarc::timing::EventMarks::new(cap)?;
        Some(Box::new(super::output::CudaMarks(m)))
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
                let planned = m.planned_steps(job);
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
