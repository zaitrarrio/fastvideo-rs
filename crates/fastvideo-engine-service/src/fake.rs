//! `FakeBackend`: deterministic synthetic video and audio with configurable
//! timing, failure and cancel injection (design §7.3), so every adapter and
//! CI run without a GPU.
//!
//! - **Frames**: a gradient (red across, green down, blue from seed and
//!   prompt) with the frame index burned in as 32 black/white cells along the
//!   top edge; [`decode_frame_index`] reads it back, so tests can check order
//!   and drops.
//! - **Audio**: a sine at the model's native rate (pitch from the seed) with
//!   a 1 ms full-scale click at every clip start, exactly
//!   `round(frames / fps * rate)` sample frames long.
//! - **Timing**: per-step latency (or a build RTF) on a [`Clock`]; with a
//!   [`ManualClock`](crate::clock::ManualClock) tests step generation by hand.
//! - **Faults**: a prompt containing [`FakeFaults::fail_marker`] fails at
//!   `fail_at_step`; listed models fail to load.
//! - **MP4**: in `OutputMode::File`, written through `ffmpeg` when it is on
//!   `PATH` (skipped otherwise: `ClipOutput::mp4` is `None`).
//!
//! The default model set mirrors the real caps shapes: H3 max/turbo, Sol-H3
//! 4-step, LTX pro/turbo, a video-only Wan clip model, and a causal SF-Wan model.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use fastvideo_protocol::{
    ApiError, AudioCaps, AudioPlan, CanvasCaps, Family, FpsCaps, FrameGrid, JobMetrics, KnobCaps,
    ModelCaps, ModelId, Pcm, RefLimits, ResolvedJob, RgbFrame, StreamCaps, Task, Tier,
};

use crate::backend::{
    BlockInput, BlockStats, CausalSpec, ClipOutput, ClipSink, DeviceInfo, EngineBackend,
    LoadEvent, SessionId,
};
use crate::cancel::{OutputMode, StepControl};
use crate::caps::Recipe;
use crate::clock::{Clock, SystemClock};

/// One model the fake serves.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeModel {
    pub caps: ModelCaps,
    pub recipe: Recipe,
}

fn recipe(name: &str, steps: u32, attention: &str, vae: &str, summary: &str) -> Recipe {
    Recipe {
        name: name.into(),
        profile: None,
        steps: Some(steps),
        attention: attention.into(),
        vae: vae.into(),
        summary: summary.into(),
    }
}

impl FakeModel {
    /// H3 highest-quality tier (t2v, i2v, fl2va, ref2va; stereo 32 kHz).
    pub fn h3_max() -> Self {
        Self {
            caps: ModelCaps::h3("fake-h3-max", true).with_tier(Tier::Max, "full-dense"),
            recipe: recipe(
                "full-dense",
                8,
                "dense",
                "full",
                "fake: full step count, dense attention, full VAE",
            ),
        }
    }

    /// H3 fastest tier (no ref2va).
    pub fn h3_turbo() -> Self {
        Self {
            caps: ModelCaps::h3("fake-h3-turbo", false).with_tier(Tier::Turbo, "4step-vsa"),
            recipe: recipe(
                "4step-vsa",
                4,
                "vsa",
                "full",
                "fake: 4-step distilled, sparse attention",
            ),
        }
    }

    fn ltx(id: &str) -> ModelCaps {
        let grid = FrameGrid::new(8, 1, 145, 481, 145);
        ModelCaps {
            id: ModelId::new(id),
            family: Family::Ltx2,
            served_names: vec![id.to_owned()],
            tasks: [Task::T2V, Task::I2V, Task::Keyframes].into_iter().collect(),
            audio: Some(AudioCaps {
                native_rate: 48_000,
                channels: 2,
                via_sidecar: false,
            }),
            fps: crate::caps::ltx_fps_caps(),
            stream: Some(StreamCaps::Clip {
                min_s: grid.min as f32 / 24.0,
                max_s: grid.max as f32 / 24.0,
            }),
            frames: grid,
            canvas: CanvasCaps {
                multiple: 64,
                max_area: 3840 * 2176,
                aspect: (0.25, 4.0),
                short_edges: vec![1080, 720, 1440, 2160],
                pad_and_crop: true,
                hd: None,
            },
            refs: RefLimits::none(),
            knobs: KnobCaps {
                seed: true,
                negative: true,
                steps: true,
                guidance: true,
                ..KnobCaps::default()
            },
            resident: true,
            tier: None,
            recipe: None,
        }
    }

    /// Sol-H3 4-step (untiered). Its recipe names no profile, so the
    /// capability table selects the tau-ladder profile (design §0.5).
    pub fn sol_h3() -> Self {
        Self {
            caps: ModelCaps::h3("fake-sol-h3", false),
            recipe: Recipe {
                profile: None,
                ..Recipe::sol_h3_4step()
            },
        }
    }

    /// LTX highest-quality tier (stereo 48 kHz, 8k+1 grid, pad-and-crop).
    pub fn ltx_pro() -> Self {
        Self {
            caps: Self::ltx("fake-ltx-pro").with_tier(Tier::Max, "two-stage-dense"),
            recipe: recipe(
                "two-stage-dense",
                8,
                "dense",
                "full",
                "fake: two-stage, dense stage 2",
            ),
        }
    }

    /// LTX fastest tier.
    pub fn ltx_turbo() -> Self {
        Self {
            caps: Self::ltx("fake-ltx-turbo").with_tier(Tier::Turbo, "two-stage-sol"),
            recipe: recipe(
                "two-stage-sol",
                4,
                "sol",
                "full",
                "fake: distilled two-stage, sparse stage 2",
            ),
        }
    }

    fn wan_caps(id: &str) -> ModelCaps {
        let grid = FrameGrid::new(4, 1, 49, 121, 81);
        ModelCaps {
            id: ModelId::new(id),
            family: Family::Wan,
            served_names: vec![id.to_owned()],
            tasks: [Task::T2V].into_iter().collect(),
            audio: None,
            fps: FpsCaps {
                allowed: vec![16],
                default: 16,
                container_only: true,
            },
            stream: Some(StreamCaps::Clip {
                min_s: grid.min as f32 / 16.0,
                max_s: grid.max as f32 / 16.0,
            }),
            frames: grid,
            canvas: CanvasCaps {
                multiple: 16,
                max_area: 832 * 480,
                aspect: (0.25, 4.0),
                short_edges: vec![480],
                pad_and_crop: false,
                hd: None,
            },
            refs: RefLimits::none(),
            knobs: KnobCaps {
                guidance_2: false,
                ..KnobCaps::all()
            },
            resident: true,
            tier: None,
            recipe: None,
        }
    }

    /// Video-only Wan clip model (4k+1 grid, 16 fps container rate).
    pub fn wan() -> Self {
        Self {
            caps: Self::wan_caps("fake-wan"),
            recipe: recipe("dmd-3step", 3, "dense", "full", "fake: video-only clip model"),
        }
    }

    /// Causal SF-Wan (12-frame blocks at 16 fps, video-only).
    pub fn sf_wan() -> Self {
        let mut caps = Self::wan_caps("fake-sfwan");
        caps.fps = FpsCaps::fixed(16);
        caps.stream = Some(StreamCaps::Causal {
            block_frames: 12,
            target_fps: 16,
        });
        caps.knobs = KnobCaps {
            seed: true,
            ..KnobCaps::default()
        };
        Self {
            caps,
            recipe: recipe("causal-4step", 4, "dense", "tiny", "fake: causal block rollout"),
        }
    }

    /// Every default fake model.
    pub fn default_set() -> Vec<FakeModel> {
        vec![
            Self::h3_max(),
            Self::h3_turbo(),
            Self::sol_h3(),
            Self::ltx_pro(),
            Self::ltx_turbo(),
            Self::wan(),
            Self::sf_wan(),
        ]
    }
}

/// Latency model.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeTiming {
    /// Whole-model load time (reported in 4 progress slices).
    pub load: Duration,
    /// Time per denoise step (ignored when `rtf` is set).
    pub step: Duration,
    /// Build time / clip time; spread evenly over the steps.
    pub rtf: Option<f64>,
    /// Denoise steps per causal block.
    pub causal_steps: u32,
}

impl Default for FakeTiming {
    fn default() -> Self {
        Self {
            load: Duration::ZERO,
            step: Duration::from_millis(2),
            rtf: None,
            causal_steps: 4,
        }
    }
}

/// Failure injection.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeFaults {
    /// A prompt containing this fails the job (or causal block).
    pub fail_marker: String,
    /// 1-based step at which a marked job fails.
    pub fail_at_step: u32,
    /// Models whose load fails.
    pub load_fail: BTreeSet<ModelId>,
}

impl Default for FakeFaults {
    fn default() -> Self {
        Self {
            fail_marker: "[fake:fail]".into(),
            fail_at_step: 1,
            load_fail: BTreeSet::new(),
        }
    }
}

/// MP4 writing in `OutputMode::File`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp4Mode {
    /// Write through `ffmpeg` when it runs; otherwise no file.
    Auto,
    /// Never write a file.
    Off,
}

/// Fake backend configuration.
#[derive(Clone, Debug)]
pub struct FakeConfig {
    pub models: Vec<FakeModel>,
    pub device: DeviceInfo,
    pub clock: Arc<dyn Clock>,
    pub timing: FakeTiming,
    pub faults: FakeFaults,
    pub mp4: Mp4Mode,
    pub ffmpeg: PathBuf,
}

impl Default for FakeConfig {
    fn default() -> Self {
        Self {
            models: FakeModel::default_set(),
            device: DeviceInfo {
                index: 0,
                name: "fake".into(),
                total_memory_mb: 80 * 1024,
            },
            clock: Arc::new(SystemClock::default()),
            timing: FakeTiming::default(),
            faults: FakeFaults::default(),
            mp4: Mp4Mode::Auto,
            ffmpeg: PathBuf::from("ffmpeg"),
        }
    }
}

impl FakeConfig {
    /// Only the listed models (by id) from the default set.
    pub fn with_models(mut self, ids: &[&str]) -> Self {
        self.models.retain(|m| ids.contains(&m.caps.id.as_str()));
        self
    }
}

#[derive(Debug)]
struct FakeCausal {
    spec: CausalSpec,
    block_frames: u32,
}

/// The fake engine backend.
#[derive(Debug)]
pub struct FakeBackend {
    cfg: FakeConfig,
    models: BTreeMap<ModelId, FakeModel>,
    loaded: BTreeSet<ModelId>,
    sessions: HashMap<SessionId, FakeCausal>,
    ffmpeg_ok: Option<bool>,
    /// Every load / unload, in order (tests inspect swap behaviour).
    pub history: Vec<String>,
}

impl FakeBackend {
    pub fn new(cfg: FakeConfig) -> Self {
        let models = cfg
            .models
            .iter()
            .map(|m| (m.caps.id.clone(), m.clone()))
            .collect();
        Self {
            cfg,
            models,
            loaded: BTreeSet::new(),
            sessions: HashMap::new(),
            ffmpeg_ok: None,
            history: Vec::new(),
        }
    }

    fn model(&self, id: &ModelId) -> Result<&FakeModel, ApiError> {
        self.models.get(id).ok_or_else(|| {
            ApiError::invalid_param("model", format!("fake backend does not serve `{id}`"))
        })
    }

    fn ffmpeg_available(&mut self) -> bool {
        if self.cfg.mp4 == Mp4Mode::Off {
            return false;
        }
        let bin = self.cfg.ffmpeg.clone();
        *self.ffmpeg_ok.get_or_insert_with(|| {
            Command::new(bin)
                .arg("-version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        })
    }

    fn step_time(&self, job: &ResolvedJob, steps: u32) -> Duration {
        match self.cfg.timing.rtf {
            Some(rtf) if steps > 0 => {
                Duration::from_secs_f64((rtf * job.duration_s()).max(0.0) / steps as f64)
            }
            _ => self.cfg.timing.step,
        }
    }
}

/// FNV-1a over a prompt (stable across runs and platforms).
pub fn prompt_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn cell_size(width: u32, height: u32) -> Option<(u32, u32)> {
    if width < 32 || height == 0 {
        return None;
    }
    let cw = width / 32;
    Some((cw, cw.min(height)))
}

/// The deterministic frame `index` for `(seed, prompt)` at `width x height`.
pub fn render_frame(seed: u64, prompt: &str, width: u32, height: u32, index: u64) -> RgbFrame {
    let blue = ((seed ^ prompt_hash(prompt)) & 0xff) as u8;
    let (w, h) = (width as usize, height as usize);
    let mut v = vec![0u8; w * h * 3];
    let wd = (w.max(2) - 1) as u32;
    let hd = (h.max(2) - 1) as u32;
    for y in 0..h {
        let g = (y as u32 * 255 / hd) as u8;
        let row = &mut v[y * w * 3..(y + 1) * w * 3];
        for (x, px) in row.chunks_exact_mut(3).enumerate() {
            px[0] = (x as u32 * 255 / wd) as u8;
            px[1] = g;
            px[2] = blue;
        }
    }
    if let Some((cw, ch)) = cell_size(width, height) {
        let idx = index as u32;
        for bit in 0..32u32 {
            let on = (idx >> (31 - bit)) & 1 == 1;
            let val = if on { 255 } else { 0 };
            for y in 0..ch as usize {
                for x in (bit * cw) as usize..((bit + 1) * cw) as usize {
                    let i = (y * w + x) * 3;
                    v[i..i + 3].copy_from_slice(&[val, val, val]);
                }
            }
        }
    }
    RgbFrame {
        width,
        height,
        data: v.into(),
        index,
    }
}

/// Reads back the index [`render_frame`] burned in (`None` if the cells are
/// not clean black/white, or the frame is narrower than 32 px).
pub fn decode_frame_index(f: &RgbFrame) -> Option<u32> {
    let (cw, ch) = cell_size(f.width, f.height)?;
    let mut idx = 0u32;
    for bit in 0..32u32 {
        let p = f.pixel(bit * cw + cw / 2, ch / 2)?;
        let b = if p.iter().all(|&c| c >= 200) {
            1
        } else if p.iter().all(|&c| c <= 55) {
            0
        } else {
            return None;
        };
        idx = (idx << 1) | b;
    }
    Some(idx)
}

/// Sample frames in a clip of `frames` at `fps` and `rate`:
/// `round(frames / fps * rate)`.
pub fn audio_len(frames: u32, fps: u32, rate: u32) -> usize {
    if fps == 0 {
        return 0;
    }
    ((frames as u64 * rate as u64 + fps as u64 / 2) / fps as u64) as usize
}

/// The deterministic clip audio: a sine (220 + 55·(seed mod 8) Hz, amplitude
/// 0.25) with a 1 ms click of 1.0 at the start.
pub fn render_audio(seed: u64, frames: u32, fps: u32, rate: u32, channels: u8) -> Pcm {
    let n = audio_len(frames, fps, rate);
    let f = 220.0 + 55.0 * (seed % 8) as f64;
    let click = (rate / 1000).max(1) as usize;
    let c = channels.max(1) as usize;
    let mut v = Vec::with_capacity(n * c);
    for i in 0..n {
        let s = if i < click {
            1.0
        } else {
            (0.25 * (2.0 * std::f64::consts::PI * f * i as f64 / rate as f64).sin()) as f32
        };
        v.extend(std::iter::repeat_n(s, c));
    }
    Pcm::new(rate, channels.max(1), v)
}

fn write_wav(path: &Path, pcm: &Pcm) -> std::io::Result<()> {
    let data_len = (pcm.samples.len() * 2) as u32;
    let ch = pcm.channels as u16;
    let mut b = Vec::with_capacity(44 + data_len as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&ch.to_le_bytes());
    b.extend_from_slice(&pcm.rate.to_le_bytes());
    b.extend_from_slice(&(pcm.rate * ch as u32 * 2).to_le_bytes());
    b.extend_from_slice(&(ch * 2).to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm.samples.iter() {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b)
}

fn io_err(what: &str, e: impl std::fmt::Display) -> ApiError {
    ApiError::engine_failed(format!("fake mp4: {what}: {e}"))
}

/// Pipes RGB frames (and a WAV) through ffmpeg into `dir/output.mp4`.
fn write_mp4(
    ffmpeg: &Path,
    dir: &Path,
    job: &ResolvedJob,
    frames: impl Iterator<Item = RgbFrame>,
    audio: Option<&Pcm>,
) -> Result<PathBuf, ApiError> {
    std::fs::create_dir_all(dir).map_err(|e| io_err("mkdir", e))?;
    let out = dir.join("output.mp4");
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg("-s")
        .arg(format!("{}x{}", job.width, job.height))
        .arg("-r")
        .arg(job.fps.to_string())
        .args(["-i", "-"]);
    if let Some(pcm) = audio {
        let wav = dir.join("audio.wav");
        write_wav(&wav, pcm).map_err(|e| io_err("wav", e))?;
        cmd.arg("-i").arg(&wav).args(["-c:a", "aac"]);
    }
    cmd.args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        .arg(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| io_err("spawn", e))?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| io_err("stdin", "missing"))?;
        for f in frames {
            stdin.write_all(&f.data).map_err(|e| io_err("pipe", e))?;
        }
    }
    let o = child.wait_with_output().map_err(|e| io_err("wait", e))?;
    if !o.status.success() {
        return Err(io_err("ffmpeg", String::from_utf8_lossy(&o.stderr)));
    }
    let _ = std::fs::remove_file(dir.join("audio.wav"));
    Ok(out)
}

impl EngineBackend for FakeBackend {
    fn device(&self) -> DeviceInfo {
        self.cfg.device.clone()
    }

    fn caps(&self) -> Vec<ModelCaps> {
        self.cfg.models.iter().map(|m| m.caps.clone()).collect()
    }

    fn recipe(&self, model: &ModelId) -> Recipe {
        self.models
            .get(model)
            .map(|m| m.recipe.clone())
            .unwrap_or_default()
    }

    fn load(&mut self, model: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError> {
        self.model(model)?;
        obs(LoadEvent::Stage("weights"));
        let slice = self.cfg.timing.load / 4;
        for i in 1..=4u64 {
            self.cfg.clock.sleep(slice);
            obs(LoadEvent::Progress { done: i, total: 4 });
        }
        if self.cfg.faults.load_fail.contains(model) {
            return Err(ApiError::engine_failed(format!(
                "injected load failure for `{model}`"
            )));
        }
        self.loaded.insert(model.clone());
        self.history.push(format!("load {model}"));
        Ok(())
    }

    fn unload(&mut self, model: &ModelId) {
        if self.loaded.remove(model) {
            self.history.push(format!("unload {model}"));
        }
    }

    fn generate(
        &mut self,
        job: &ResolvedJob,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<ClipOutput, ApiError> {
        let m = self.model(&job.model)?.clone();
        if !self.loaded.contains(&job.model) {
            return Err(ApiError::engine_failed(format!("`{}` is not loaded", job.model)));
        }
        let steps = match job.sampling.steps {
            Some(s) if m.caps.knobs.steps && s > 0 => s,
            _ => m.recipe.steps.unwrap_or(4).max(1),
        };
        let step = self.step_time(job, steps);
        let clock = self.cfg.clock.clone();
        let t0 = clock.now();
        let fail = job.prompt.contains(&self.cfg.faults.fail_marker);

        ctl.stage("text_encode");
        ctl.check()?;
        ctl.stage("denoise");
        for s in 1..=steps {
            clock.sleep(step);
            if fail && s >= self.cfg.faults.fail_at_step {
                return Err(ApiError::engine_failed(format!(
                    "injected failure at step {s}/{steps}"
                )));
            }
            ctl.step(s, steps)?;
        }
        let denoise_s = (clock.now() - t0).as_secs_f64();

        ctl.stage("decode");
        // Batch jobs (`OutputMode::File`) drop what goes to the sink, so the
        // frames are rendered only for a stream build or the MP4. Rendering
        // them for nothing cost seconds per 1344x768 clip in debug builds,
        // enough to time tests out on a loaded host.
        let to_sink = matches!(ctl.mode, OutputMode::Frames);
        let mp4_dir = match &ctl.mode {
            OutputMode::File { dir } if self.ffmpeg_available() => Some(dir.clone()),
            _ => None,
        };
        let audio = match job.audio {
            _ if !to_sink && mp4_dir.is_none() => None,
            AudioPlan::Native { rate, channels } => Some(render_audio(
                job.seed,
                job.num_frames,
                job.fps,
                rate,
                channels,
            )),
            _ => None,
        };
        let render = |i: u32| render_frame(job.seed, &job.prompt, job.width, job.height, i as u64);
        const CHUNK: u32 = 8;
        let mut i = 0;
        while to_sink && i < job.num_frames {
            let end = (i + CHUNK).min(job.num_frames);
            let chunk: Vec<RgbFrame> = (i..end).map(render).collect();
            out.frames(&chunk);
            i = end;
        }
        if let (true, Some(a)) = (to_sink, &audio) {
            out.audio(a);
        }
        ctl.check()?;

        let mut metrics = JobMetrics::default();
        metrics.stage_durations.insert("denoise".into(), denoise_s);
        metrics.inference_s = Some(denoise_s);
        if job.duration_s() > 0.0 {
            metrics.build_rtf = Some(denoise_s / job.duration_s());
        }
        let mp4 = match &mp4_dir {
            Some(dir) => {
                ctl.stage("mux");
                Some(write_mp4(
                    &self.cfg.ffmpeg,
                    dir,
                    job,
                    (0..job.num_frames).map(render),
                    audio.as_ref(),
                )?)
            }
            _ => None,
        };
        Ok(ClipOutput {
            mp4,
            frames: None,
            audio: None,
            metrics,
        })
    }

    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError> {
        let m = self.model(&spec.model)?;
        let Some(StreamCaps::Causal { block_frames, .. }) = m.caps.stream else {
            return Err(ApiError::invalid_param(
                "model",
                format!("`{}` is not a causal model", spec.model),
            ));
        };
        if !self.loaded.contains(&spec.model) {
            return Err(ApiError::engine_failed(format!("`{}` is not loaded", spec.model)));
        }
        self.sessions.insert(
            s,
            FakeCausal {
                spec: spec.clone(),
                block_frames,
            },
        );
        Ok(())
    }

    fn causal_block(
        &mut self,
        s: SessionId,
        input: &BlockInput,
        out: &mut dyn ClipSink,
        ctl: &StepControl,
    ) -> Result<BlockStats, ApiError> {
        let sess = self
            .sessions
            .get(&s)
            .ok_or_else(|| ApiError::internal(format!("unknown causal session {s}")))?;
        let (w, h, n) = (sess.spec.width, sess.spec.height, sess.block_frames);
        let steps = self.cfg.timing.causal_steps.max(1);
        let clock = self.cfg.clock.clone();
        let t0 = clock.now();
        for st in 1..=steps {
            clock.sleep(self.cfg.timing.step);
            if input.prompt.contains(&self.cfg.faults.fail_marker) {
                return Err(ApiError::engine_failed("injected causal block failure"));
            }
            ctl.step(st, steps)?;
        }
        let base = input.block_index * n as u64;
        let frames: Vec<RgbFrame> = (0..n as u64)
            .map(|i| render_frame(input.seed, &input.prompt, w, h, base + i))
            .collect();
        out.frames(&frames);
        Ok(BlockStats {
            block_index: input.block_index,
            frames: n,
            block_ms: (clock.now() - t0).as_secs_f64() * 1e3,
        })
    }

    fn causal_close(&mut self, s: SessionId) {
        self.sessions.remove(&s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_index_round_trips() {
        for idx in [0u64, 1, 5, 123, 0xdead_beef] {
            let f = render_frame(7, "p", 64, 36, idx);
            assert_eq!(f.data.len(), 64 * 36 * 3);
            assert_eq!(decode_frame_index(&f), Some(idx as u32));
        }
        assert_eq!(decode_frame_index(&render_frame(1, "p", 16, 16, 3)), None);
    }

    #[test]
    fn frames_deterministic_and_prompt_sensitive() {
        let a = render_frame(1, "cat", 64, 32, 9);
        assert_eq!(a, render_frame(1, "cat", 64, 32, 9));
        assert_ne!(a.data, render_frame(1, "dog", 64, 32, 9).data);
    }

    #[test]
    fn audio_len_and_click() {
        assert_eq!(audio_len(124, 24, 32_000), 165_333);
        assert_eq!(audio_len(24, 24, 48_000), 48_000);
        let p = render_audio(3, 24, 24, 48_000, 2);
        assert_eq!(p.frames(), 48_000);
        assert_eq!(p.channels, 2);
        assert_eq!(p.samples[0], 1.0);
        assert_eq!(p.samples[2 * 47], 1.0);
        assert!(p.samples[2 * 48].abs() <= 0.25);
        assert_eq!(p, render_audio(3, 24, 24, 48_000, 2));
    }

    #[test]
    fn default_models_are_consistent() {
        for m in FakeModel::default_set() {
            let c = &m.caps;
            assert!(c.frames.contains(c.frames.default), "{}", c.id);
            assert!(c.fps.allows(c.fps.default));
            assert_eq!(48_000 % c.fps.default, 0);
        }
    }
}
