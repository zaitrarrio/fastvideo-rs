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
    /// Enforced in video time from the first emitted frame.
    pub max_seconds: Option<u32>,
    pub seed: Option<u64>,
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
