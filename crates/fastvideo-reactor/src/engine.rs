//! The engine seam the Reactor runtime drives (design §5.1, §5.2).
//!
//! The runtime needs readiness, the caps of the streamed model, and the
//! WP-12 / WP-15 session API of `fastvideo_engine_service::stream`: a
//! [`ClipSession`] (turned into a fast-h3 `ClipPlayer`) or a
//! [`CausalSession`] (driven through `CausalControl`). [`StreamEngine`] is
//! that seam; `EngineService` implements it (with the fake backend in CI,
//! the CUDA backend in production).

use async_trait::async_trait;
use fastvideo_engine_service::{CausalSession, ClipSession, DuplexSession, EngineService, Readiness};
use fastvideo_protocol::{ApiError, DuplexSpec, ModelCaps, ModelId, SessionSpec, StreamCaps};

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

/// The engine as the Reactor runtime sees it.
#[async_trait]
pub trait StreamEngine: Send + Sync + 'static {
    fn load_state(&self) -> LoadState;
    /// Caps of the model to stream: `want` by id/served name, else the
    /// first resident model with stream caps. `None`: nothing streamable.
    fn stream_model(&self, want: Option<&str>) -> Option<ModelCaps>;
    /// Admission of a clip session (`StreamCaps::Clip` models).
    async fn open_clip(&self, spec: SessionSpec) -> Result<ClipSession, ApiError>;
    /// Admission of a causal session (`StreamCaps::Causal` models).
    async fn open_causal(&self, spec: SessionSpec) -> Result<CausalSession, ApiError>;
    /// Admission of a duplex session (`StreamCaps::Duplex` models: client
    /// input tracks, design §5.11).
    async fn open_duplex(&self, spec: DuplexSpec) -> Result<DuplexSession, ApiError>;
}

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

    async fn open_clip(&self, spec: SessionSpec) -> Result<ClipSession, ApiError> {
        self.open_clip_session(spec).await
    }

    async fn open_causal(&self, spec: SessionSpec) -> Result<CausalSession, ApiError> {
        self.open_causal_session(spec).await
    }

    async fn open_duplex(&self, spec: DuplexSpec) -> Result<DuplexSession, ApiError> {
        self.open_duplex_session(spec).await
    }
}

/// Which command set a model gets (design §5.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Clip,
    Causal,
    /// The script avatar (Reactor `ltx`): image-to-video clips with audio,
    /// windowed. Chosen by config (`[reactor] mode = "avatar"`), never
    /// implied by caps.
    Avatar,
    /// Client input tracks into a duplex model (design §5.11): the client
    /// publishes its camera and microphone with `PublishTrack`.
    Duplex,
}

impl Mode {
    /// Whether a model can serve the script avatar: image-to-video with
    /// native audio, clip-streamable.
    pub fn avatar_capable(caps: &ModelCaps) -> bool {
        matches!(caps.stream, Some(StreamCaps::Clip { .. }))
            && caps.supports(fastvideo_protocol::Task::I2V)
            && caps.audio.as_ref().is_some_and(|a| !a.via_sidecar)
    }

    /// Parses `clip` | `causal` | `avatar`.
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "clip" => Some(Mode::Clip),
            "causal" => Some(Mode::Causal),
            "avatar" => Some(Mode::Avatar),
            _ => None,
        }
    }

    pub fn of(caps: &ModelCaps) -> Option<Mode> {
        match caps.stream.as_ref()? {
            StreamCaps::Clip { .. } => Some(Mode::Clip),
            StreamCaps::Causal { .. } => Some(Mode::Causal),
            StreamCaps::Duplex(_) => Some(Mode::Duplex),
        }
    }
}
