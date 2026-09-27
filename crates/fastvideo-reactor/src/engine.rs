//! The thin engine seam the Reactor runtime drives (design §5.1, §5.2).
//!
//! [`StreamEngine`] is all the runtime needs from the engine: readiness, the
//! caps of the streamed model, and opening a clip-build session
//! ([`ClipEngine`]) or a causal block stream ([`CausalEngine`]). Everything
//! it hands back is already in **wire form**: RGB frames at the session
//! canvas and 48 kHz **mono** PCM, exactly `48000/fps` samples per frame
//! (design §5.3, Reactor row).
//!
//! The implementation for `fastvideo_engine_service::EngineService` is here
//! too, over the WP-02 `ClipSession::build` / `CausalSession` seams. The
//! fast-h3 queue-and-playout semantics run in this crate
//! ([`crate::clip`]) on top of [`ClipEngine::build`], so swapping in WP-12's
//! engine-side queue later only changes this file and `clip.rs`.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastvideo_engine_service::{CausalSession, ClipBuild, ClipSession, EngineService, Readiness};
use fastvideo_media::lockstep;
use fastvideo_protocol::{ApiError, ModelCaps, ModelId, Pcm, RgbFrame, SessionSpec, StreamCaps};

/// Model loading as the runtime's state machine sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadState {
    /// `CREATED` (503 + `Retry-After: 1` on `/start_session`).
    Loading,
    /// `READY`.
    Ready,
    /// `TERMINATED`.
    Failed(String),
}

/// One clip to build, in wire terms.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildRequest {
    pub prompt: String,
    pub seed: u64,
    /// Clip length; snapped onto the model grid by the engine.
    pub seconds: f64,
    /// `(width, height)`; a change reopens the engine session.
    pub canvas: (u32, u32),
}

/// A built clip: frames plus 48 kHz mono audio, `frames·48000/fps` samples
/// long (`None` for a video-only session).
#[derive(Clone, Debug, PartialEq)]
pub struct WireClip {
    pub frames: Vec<RgbFrame>,
    pub audio: Option<Vec<f32>>,
}

/// One causal block in wire form.
#[derive(Clone, Debug, PartialEq)]
pub struct WireBlock {
    pub index: u64,
    pub frames: Vec<RgbFrame>,
    pub audio: Option<Vec<f32>>,
    pub block_ms: f64,
}

/// The engine as the Reactor runtime sees it.
#[async_trait]
pub trait StreamEngine: Send + Sync + 'static {
    fn load_state(&self) -> LoadState;
    /// Caps of the model to stream: `want` by id/served name, else the
    /// first model with stream caps. `None`: nothing streamable.
    fn stream_model(&self, want: Option<&str>) -> Option<ModelCaps>;
    /// Opens a clip-build session (`StreamCaps::Clip` models).
    async fn open_clip(&self, spec: SessionSpec) -> Result<Arc<dyn ClipEngine>, ApiError>;
    /// Opens a causal block stream (`StreamCaps::Causal` models).
    async fn open_causal(&self, spec: SessionSpec) -> Result<Box<dyn CausalEngine>, ApiError>;
}

/// Clip builds for one session.
#[async_trait]
pub trait ClipEngine: Send + Sync {
    /// The frame count a clip of `seconds` gets (snapped up onto the grid).
    fn frames_for(&self, seconds: Option<f64>) -> Result<u32, ApiError>;
    /// Builds one clip. Dropping the future cancels the build.
    async fn build(&self, req: BuildRequest) -> Result<WireClip, ApiError>;
    /// Releases the engine session.
    fn close(&self);
}

/// A causal block stream for one session.
#[async_trait]
pub trait CausalEngine: Send {
    fn set_prompt(&mut self, prompt: &str);
    fn set_paused(&mut self, paused: bool);
    fn set_seed(&mut self, seed: u64);
    fn reset(&mut self);
    /// The next block; `None` once the session closed.
    async fn next_block(&mut self) -> Option<Result<WireBlock, ApiError>>;
    fn close(self: Box<Self>);
}

/// Native PCM → wire PCM (48 kHz mono) for `frames` frames at `fps`.
pub fn wire_audio(native: Option<&Pcm>, frames: u32, fps: u32) -> Result<Vec<f32>, ApiError> {
    lockstep::prepare_clip_audio(native, frames, fps, fastvideo_protocol::WIRE_AUDIO_RATE, 1)
        .map(|p| p.samples.to_vec())
        .map_err(|e| ApiError::internal(format!("audio: {e}")))
}

// ---- EngineService -----------------------------------------------------------

#[async_trait]
impl StreamEngine for EngineService {
    fn load_state(&self) -> LoadState {
        match self.readiness() {
            Readiness::Ready => LoadState::Ready,
            Readiness::Loading { .. } => LoadState::Loading,
            Readiness::Failed(e) => LoadState::Failed(e),
        }
    }

    fn stream_model(&self, want: Option<&str>) -> Option<ModelCaps> {
        let caps = self.caps();
        match want {
            Some(w) => caps
                .get(&ModelId::new(w))
                .or_else(|| caps.resolve(w))
                .filter(|c| c.stream.is_some())
                .cloned(),
            None => caps.models().find(|c| c.stream.is_some() && c.resident).cloned(),
        }
    }

    async fn open_clip(&self, spec: SessionSpec) -> Result<Arc<dyn ClipEngine>, ApiError> {
        let s = self.open_clip_session(spec.clone()).await?;
        Ok(Arc::new(ServiceClips {
            engine: self.clone(),
            cur: Mutex::new(Some(Arc::new(s))),
            spec: Mutex::new(spec),
        }))
    }

    async fn open_causal(&self, spec: SessionSpec) -> Result<Box<dyn CausalEngine>, ApiError> {
        let fps = spec.fps;
        let s = self.open_causal_session(spec).await?;
        Ok(Box::new(ServiceCausal { s, fps }))
    }
}

struct ServiceClips {
    engine: EngineService,
    cur: Mutex<Option<Arc<ClipSession>>>,
    spec: Mutex<SessionSpec>,
}

impl ServiceClips {
    fn session(&self) -> Result<Arc<ClipSession>, ApiError> {
        lock(&self.cur)
            .clone()
            .ok_or_else(|| ApiError::internal("the clip session is closed"))
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Cancels the engine job when a build future is dropped.
struct CancelOnDrop(fastvideo_engine_service::CancelToken, bool);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if !self.1 {
            self.0.cancel();
        }
    }
}

#[async_trait]
impl ClipEngine for ServiceClips {
    fn frames_for(&self, seconds: Option<f64>) -> Result<u32, ApiError> {
        self.session()?.frames_for(seconds)
    }

    async fn build(&self, req: BuildRequest) -> Result<WireClip, ApiError> {
        // A canvas change reopens the engine session at the new canvas
        // (`set_canvas` is only valid while nothing is queued or playing).
        let reopen = {
            let spec = lock(&self.spec);
            (spec.canvas != req.canvas).then(|| {
                let mut s = spec.clone();
                s.canvas = req.canvas;
                s.tracks.video.width = req.canvas.0;
                s.tracks.video.height = req.canvas.1;
                s
            })
        };
        if let Some(spec) = reopen {
            if let Some(old) = lock(&self.cur).take() {
                drop(old);
            }
            let s = self.engine.open_clip_session(spec.clone()).await?;
            *lock(&self.cur) = Some(Arc::new(s));
            *lock(&self.spec) = spec;
        }
        let session = self.session()?;
        let fps = session.spec().fps;
        let has_audio = session.spec().tracks.has_audio();
        let handle = session
            .build(ClipBuild {
                prompt: req.prompt,
                seed: Some(req.seed),
                seconds: Some(req.seconds),
                ..ClipBuild::default()
            })
            .await?;
        let mut guard = CancelOnDrop(handle.cancel.clone(), false);
        let out = handle.wait().await;
        guard.1 = true;
        let out = out?;
        let frames = out.frames.unwrap_or_default();
        let audio = if has_audio {
            Some(wire_audio(out.audio.as_ref(), frames.len() as u32, fps)?)
        } else {
            None
        };
        Ok(WireClip { frames, audio })
    }

    fn close(&self) {
        lock(&self.cur).take();
    }
}

struct ServiceCausal {
    s: CausalSession,
    fps: u32,
}

#[async_trait]
impl CausalEngine for ServiceCausal {
    fn set_prompt(&mut self, prompt: &str) {
        self.s.set_prompt(prompt);
    }
    fn set_paused(&mut self, paused: bool) {
        self.s.set_paused(paused);
    }
    fn set_seed(&mut self, seed: u64) {
        self.s.set_seed(seed);
    }
    fn reset(&mut self) {
        self.s.reset();
    }
    async fn next_block(&mut self) -> Option<Result<WireBlock, ApiError>> {
        let b = self.s.next_block().await?;
        Some(b.and_then(|b| {
            let audio = match &b.audio {
                Some(a) => Some(wire_audio(Some(a), b.frames.len() as u32, self.fps)?),
                None => None,
            };
            Ok(WireBlock { index: b.index, frames: b.frames, audio, block_ms: b.stats.block_ms })
        }))
    }
    fn close(self: Box<Self>) {
        self.s.close();
    }
}

/// Which command set a model gets (design §5.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Clip,
    Causal,
}

impl Mode {
    pub fn of(caps: &ModelCaps) -> Option<Mode> {
        match caps.stream.as_ref()? {
            StreamCaps::Clip { .. } => Some(Mode::Clip),
            StreamCaps::Causal { .. } => Some(Mode::Causal),
        }
    }
}
