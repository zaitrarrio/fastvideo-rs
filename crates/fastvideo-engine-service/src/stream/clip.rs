//! `ClipSession`: the engine seam for clip-queue playout (design §5.5).
//!
//! WP-02 provides admission (one session per executor, resident model,
//! `48000 % fps == 0` when the session has audio) and clip **builds**: each
//! build is a `Priority::Stream` job pinned to the session's executor whose
//! output (frames + native-rate PCM) stays in memory.
//!
//! WP-12 adds the fast-h3 queue semantics on top: [`ClipSession::into_player`]
//! starts a [`ClipPlayer`](super::player::ClipPlayer) (generation/playout
//! queues with reservation, autoplay, `valid_commands`, lockstep slicing and
//! `Continuity`), see `stream/{player,queue,rules,pace}.rs`.

use std::path::PathBuf;
use std::sync::Arc;

use fastvideo_protocol::{
    draw_seed, Anchor, ApiError, AudioPlan, JobId, ModelCaps, PostProcess, ResolvedJob,
    SamplingOverrides, SessionSpec, Task,
};

use crate::backend::SessionId;
use crate::cancel::OutputMode;
use crate::scheduler::Priority;
use crate::service::{JobHandle, Shared};

/// One clip to build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipBuild {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    /// `None`: the session seed, else a fresh one.
    pub seed: Option<u64>,
    /// Snapped up onto the model's frame grid; `None`: the model default.
    pub seconds: Option<f64>,
    /// An exact frame count (must be admissible); overrides `seconds`.
    pub frames: Option<u32>,
    /// `(width, height)`; `None`: the session canvas.
    pub canvas: Option<(u32, u32)>,
    /// `Keyframe{First}` (continuity `AnchorLastFrame`, director `image_url`).
    pub first_frame: Option<PathBuf>,
    /// `Keyframe{Last}` (director `end_image_url`).
    pub last_frame: Option<PathBuf>,
}

/// An open clip session. Dropping it releases the executor slot.
pub struct ClipSession {
    engine: Arc<Shared>,
    id: SessionId,
    executor: usize,
    spec: SessionSpec,
    caps: ModelCaps,
    closed: bool,
}

impl std::fmt::Debug for ClipSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClipSession")
            .field("id", &self.id)
            .field("model", &self.spec.model)
            .field("executor", &self.executor)
            .finish()
    }
}

impl ClipSession {
    pub(crate) fn new(
        engine: Arc<Shared>,
        id: SessionId,
        executor: usize,
        spec: SessionSpec,
        caps: ModelCaps,
    ) -> Self {
        Self {
            engine,
            id,
            executor,
            spec,
            caps,
            closed: false,
        }
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn spec(&self) -> &SessionSpec {
        &self.spec
    }

    pub fn caps(&self) -> &ModelCaps {
        &self.caps
    }

    pub fn executor(&self) -> usize {
        self.executor
    }

    /// The frame count a clip of `seconds` gets (snapped up onto the grid).
    pub fn frames_for(&self, seconds: Option<f64>) -> Result<u32, ApiError> {
        let grid = &self.caps.frames;
        let Some(s) = seconds else {
            return Ok(grid.default);
        };
        if !s.is_finite() || s <= 0.0 {
            return Err(ApiError::invalid_param("seconds", "clip length must be positive"));
        }
        let n = (s * self.spec.fps as f64).round() as u32;
        grid.align_up(n).ok_or_else(|| {
            ApiError::invalid_param(
                "seconds",
                format!(
                    "clip length {s} s is outside {:.3}..{:.3} s",
                    grid.min as f64 / self.spec.fps as f64,
                    grid.max as f64 / self.spec.fps as f64
                ),
            )
        })
    }

    /// The engine job for one build (pure).
    pub fn resolve(&self, b: &ClipBuild) -> Result<ResolvedJob, ApiError> {
        let num_frames = match b.frames {
            Some(n) if self.caps.frames.contains(n) => n,
            Some(n) => {
                return Err(ApiError::invalid_param(
                    "frames",
                    format!("{n} frames is not an admissible clip length for `{}`", self.caps.id),
                ))
            }
            None => self.frames_for(b.seconds)?,
        };
        let (width, height) = b.canvas.unwrap_or(self.spec.canvas);
        let mut keyframes = Vec::new();
        if let Some(p) = &b.first_frame {
            keyframes.push((Anchor::First, p.clone()));
        }
        if let Some(p) = &b.last_frame {
            keyframes.push((Anchor::Last, p.clone()));
        }
        let task = match (&b.first_frame, &b.last_frame) {
            (None, None) => Task::T2V,
            (Some(_), None) => Task::I2V,
            _ => Task::Keyframes,
        };
        if !self.caps.supports(task) {
            return Err(ApiError::invalid_param(
                "image_url",
                format!("model `{}` cannot build {task:?} clips", self.caps.id),
            ));
        }
        let audio = match (&self.caps.audio, self.spec.tracks.has_audio()) {
            (Some(a), true) if !a.via_sidecar => AudioPlan::Native {
                rate: a.native_rate,
                channels: a.channels,
            },
            (Some(_), true) => AudioPlan::Sidecar,
            _ => AudioPlan::None,
        };
        Ok(ResolvedJob {
            model: self.caps.id.clone(),
            task,
            prompt: b.prompt.clone(),
            negative_prompt: b.negative_prompt.clone().unwrap_or_default(),
            seed: b.seed.or(self.spec.seed).unwrap_or_else(draw_seed),
            width,
            height,
            num_frames,
            fps: self.spec.fps,
            keyframes,
            references: Vec::new(),
            audio_in: None,
            audio,
            post: PostProcess::default(),
            sampling: SamplingOverrides::default(),
            tier: self.caps.tier,
            recipe: self.caps.recipe.clone(),
            edit: None,
        })
    }

    /// Queues a build at `Priority::Stream` on this session's executor. The
    /// `Finished` output carries `frames` and native-rate `audio`.
    pub async fn build(&self, b: ClipBuild) -> Result<JobHandle, ApiError> {
        let job = self.resolve(&b)?;
        self.engine.submit(
            JobId::new(),
            job,
            Priority::Stream,
            Some(self.executor),
            Some(OutputMode::Frames),
        )
    }

    /// Starts the fast-h3 queue-and-playout player over this session (needs
    /// a tokio runtime). The player owns the session from now on.
    pub fn into_player(
        self,
        cfg: super::player::ClipPlayerConfig,
    ) -> Result<(super::player::ClipPlayer, super::player::ClipOutputs), ApiError> {
        super::player::ClipPlayer::start(self, cfg)
    }

    /// Ends the session.
    pub fn close(mut self) {
        self.do_close();
    }

    /// Ends the session in place (the player's shutdown path).
    pub(crate) fn end(&mut self) {
        self.do_close();
    }

    fn do_close(&mut self) {
        if !self.closed {
            self.closed = true;
            self.engine.close_clip_session(self.id);
        }
    }
}

impl Drop for ClipSession {
    fn drop(&mut self) {
        self.do_close();
    }
}
