//! `TrackSet`, `StreamProtocol`, `SessionSpec`, `Continuity` (design §3.5, §5.2, §5.3).

use serde::{Deserialize, Serialize};

use crate::caps::ModelCaps;
use crate::error::ApiError;
use crate::request::{ModelId, ProtocolId};

/// Audio sample rate on every WebRTC path (design §5.3).
pub const WIRE_AUDIO_RATE: u32 = 48_000;

/// The tracks a streaming session carries, fixed at session creation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackSet {
    pub video: VideoTrack,
    /// `None` for a video-only model: no audio m-line / transceiver.
    pub audio: Option<AudioTrack>,
}

impl TrackSet {
    /// Tracks for `caps` at `canvas = (width, height)` and `fps`: audio exactly
    /// when the model has native audio, or sidecar audio when `sidecar` is
    /// requested and available. Audio is always [`WIRE_AUDIO_RATE`] with
    /// `audio_channels` channels (Reactor mono, WMA/WHIP stereo).
    pub fn for_model(
        caps: &ModelCaps,
        canvas: (u32, u32),
        fps: u32,
        names: (&str, &str),
        audio_channels: u8,
        sidecar: bool,
    ) -> Self {
        let with_audio = match &caps.audio {
            Some(a) if !a.via_sidecar => true,
            Some(_) => sidecar,
            None => false,
        };
        Self {
            video: VideoTrack {
                name: names.0.to_owned(),
                width: canvas.0,
                height: canvas.1,
                fps,
            },
            audio: with_audio.then(|| AudioTrack {
                name: names.1.to_owned(),
                rate: WIRE_AUDIO_RATE,
                channels: audio_channels,
            }),
        }
    }
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
    /// Track names, video first.
    pub fn names(&self) -> Vec<&str> {
        let mut v = vec![self.video.name.as_str()];
        if let Some(a) = &self.audio {
            v.push(a.name.as_str());
        }
        v
    }
    /// Audio samples per channel per video frame (`48000 / fps`), or an error
    /// when it is not an integer: the clip-session admission check (design §5.5).
    pub fn samples_per_frame(&self) -> Result<u32, ApiError> {
        let fps = self.video.fps;
        if fps == 0 || WIRE_AUDIO_RATE % fps != 0 {
            return Err(ApiError::invalid_param(
                "fps",
                format!("fps {fps} does not divide the {WIRE_AUDIO_RATE} Hz audio clock"),
            ));
        }
        Ok(WIRE_AUDIO_RATE / fps)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoTrack {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioTrack {
    pub name: String,
    /// Always 48000 on the wire.
    pub rate: u32,
    pub channels: u8,
}

/// One streaming front-end (Reactor, fal director, native streams).
pub trait StreamProtocol: Send + Sync + 'static {
    fn id(&self) -> ProtocolId;
    /// Track names and channel count this protocol uses for a model
    /// (Reactor: `main_video`/`main_audio`, mono).
    fn tracks(&self, caps: &ModelCaps, canvas: (u32, u32), fps: u32) -> TrackSet;
}

/// Protocol-agnostic streaming session state (design §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
pub enum SessionState {
    Starting,
    Ready,
    Streaming,
    /// All peers gone; generation paused until `orphan_timeout`.
    Orphaned,
    Closing,
    Closed(EndReason),
}

impl SessionState {
    /// A session occupies its executor from `Starting` until `Closed`.
    pub fn is_busy(&self) -> bool {
        !matches!(self, SessionState::Closed(_))
    }
    /// Whether the lifecycle allows `self -> next`: forward along
    /// `Starting -> Ready -> Streaming <-> Orphaned -> Closing -> Closed`, with
    /// `Closing` reachable from any open state and `Closed` from any state
    /// but `Closed`.
    pub fn can_transition_to(&self, next: &SessionState) -> bool {
        use SessionState::*;
        match (self, next) {
            (Closed(_), _) => false,
            (_, Closed(_)) => true,
            (Closing, _) => false,
            (_, Closing) => true,
            (Starting, Ready)
            | (Ready, Streaming)
            | (Streaming, Orphaned)
            | (Orphaned, Streaming) => true,
            (Ready, Orphaned) => true,
            _ => false,
        }
    }
}

/// Why a session ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    Stopped,
    TimedOut,
    SessionLimit,
    Evicted,
    Error(ApiError),
    ClientGone,
}

/// What a front-end asks the engine to open (design §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionSpec {
    pub model: ModelId,
    pub tracks: TrackSet,
    /// `(width, height)`.
    pub canvas: (u32, u32),
    pub fps: u32,
    pub continuity: Continuity,
    /// Enforced in video time from the first emitted frame (causal
    /// sessions: from the first frame or the last `reset`, see
    /// [`CausalLimits`]).
    pub max_seconds: Option<u32>,
    pub seed: Option<u64>,
}

/// Length limits of live causal (SF-Wan) sessions (design §5.2, §5.4).
///
/// Every front-end that opens a causal session resolves its `max_seconds`
/// here: the request's value, else `default_max_s`, clamped to
/// `hard_max_s`. The clock is video time from the first emitted frame. A
/// `reset` restarts it (a reset renews the anchor, so quality recovers), but
/// the whole session never exceeds `hard_max_s` of video. The defaults follow
/// the R12 study (docs/ports/wan.md): 120 s is clean unattended, 300 s is
/// usable with a transient top-edge strip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalLimits {
    /// Per-session limit when the request gives none (seconds of video).
    pub default_max_s: u32,
    /// Ceiling of a requested limit, and of the whole session across resets.
    pub hard_max_s: u32,
}

impl Default for CausalLimits {
    fn default() -> Self {
        Self { default_max_s: 120, hard_max_s: 300 }
    }
}

impl CausalLimits {
    /// The session limit for a request: `requested`, else the default, then
    /// clamped to the hard ceiling. `Some(0)` is refused (param
    /// `max_seconds`).
    pub fn resolve(&self, requested: Option<u32>) -> Result<u32, ApiError> {
        match requested {
            Some(0) => Err(ApiError::invalid_param("max_seconds", "max_seconds must be at least 1")),
            Some(s) => Ok(s.min(self.hard_max_s)),
            None => Ok(self.default_max_s.min(self.hard_max_s)),
        }
    }

    /// Whether the limits are usable: `1 <= default <= hard`.
    pub fn check(&self) -> Result<(), String> {
        if self.default_max_s == 0 || self.hard_max_s < self.default_max_s {
            return Err(format!(
                "causal stream limits need 1 <= default ({}) <= hard ({})",
                self.default_max_s, self.hard_max_s
            ));
        }
        Ok(())
    }

    /// How capabilities advertise the limits.
    pub fn advertised(&self) -> serde_json::Value {
        serde_json::json!({
            "default_max_s": self.default_max_s,
            "hard_max_s": self.hard_max_s,
            "clock": "video_seconds_from_first_frame",
            "reset_restarts_clock": true,
        })
    }
}

/// How consecutive clips join (design §5.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Continuity {
    /// IL parity; the Reactor default.
    HardCut,
    /// Raised-cosine audio fade across the boundary; sample count unchanged.
    Crossfade { ms: u16 },
    /// Clip N+1 starts from clip N's last frame, plus a crossfade. The fal
    /// director default.
    AnchorLastFrame { crossfade_ms: u16 },
}

impl Default for Continuity {
    /// `Crossfade{20}`: the default for every clip session other than Reactor
    /// fast-h3 parity mode.
    fn default() -> Self {
        Continuity::Crossfade { ms: 20 }
    }
}
