//! The engine seam of the director: a thin adapter over the engine's clip
//! sessions (`fastvideo-engine-service::ClipSession`, design §5.5), which
//! adapters may not depend on (design §2.2). `fv-serve` implements it over
//! `EngineService::open_clip_session` + `ClipSession::build`; the tests
//! implement it over the fake engine.

use std::path::PathBuf;
use std::sync::Arc;

use fastvideo_protocol::{ApiError, ModelCaps, Pcm, RgbFrame, SessionSpec};

/// One chunk to build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkBuild {
    pub prompt: String,
    /// `None`: the session seed (else a fresh one).
    pub seed: Option<u64>,
    /// Snapped up onto the model's frame grid by the engine.
    pub seconds: f64,
    /// `(width, height)` of this chunk (`configure` resolution and aspect);
    /// `None`: the session canvas.
    pub canvas: Option<(u32, u32)>,
    /// `Keyframe{First}`: `configure.image_url`, or the previous chunk's
    /// last frame (continuity `AnchorLastFrame`).
    pub first_frame: Option<PathBuf>,
    /// `Keyframe{Last}`: `end_image_url` / end-image script beats.
    pub last_frame: Option<PathBuf>,
}

/// What a build produced: frames in order and native-rate PCM.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkOutput {
    pub frames: Vec<RgbFrame>,
    pub audio: Option<Pcm>,
}

/// An open engine clip session. Dropping the last reference releases the
/// executor.
#[async_trait::async_trait]
pub trait DirectorClips: Send + Sync + 'static {
    fn caps(&self) -> &ModelCaps;
    fn spec(&self) -> &SessionSpec;
    /// Builds one chunk at `Priority::Stream` on the session's executor.
    async fn build(&self, chunk: ChunkBuild) -> Result<ChunkOutput, ApiError>;
    /// Releases the engine session now (idempotent), even while a build's
    /// future still holds a reference; later builds fail.
    async fn close(&self);
}

/// Opens clip sessions: one per executor, the model resident, `48000 % fps
/// == 0` for audio sessions. Busy is `ApiError` kind `Busy`/`Conflict`;
/// not resident is `Loading` with `Retry-After`.
#[async_trait::async_trait]
pub trait DirectorEngine: Send + Sync + 'static {
    async fn open(&self, spec: SessionSpec) -> Result<Arc<dyn DirectorClips>, ApiError>;
}

/// Frames a chunk of `seconds` gets on `caps`' grid at `fps` (snapped up,
/// as `ClipSession::frames_for` does).
pub fn frames_for(caps: &ModelCaps, fps: u32, seconds: f64) -> Option<u32> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    caps.frames.align_up((seconds * f64::from(fps)).round() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_grid() {
        let caps = ModelCaps::h3("h3", false);
        assert_eq!(frames_for(&caps, 24, 10.0), Some(243));
        assert_eq!(frames_for(&caps, 24, 5.0), Some(124));
        assert_eq!(frames_for(&caps, 24, 15.0), Some(362));
        assert_eq!(frames_for(&caps, 24, 16.0), None);
        assert_eq!(frames_for(&caps, 24, 0.0), None);
    }
}
