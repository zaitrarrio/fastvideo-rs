//! Duplex streaming: client input tracks into a live session (design §5.11).
//!
//! A duplex model reads the client's camera and microphone while it streams
//! its own output (real-time V2V, a live avatar, a full-duplex agent). It
//! declares [`StreamCaps::Duplex`](crate::StreamCaps::Duplex) with a
//! [`DuplexCaps`]:
//!
//! - the **input tracks** it reads ([`InputCaps`]): video decoded to RGB at
//!   the model's input size, audio decoded to PCM at a fixed rate, and the
//!   limits a client must stay under (resolution, frame rate, bitrate);
//! - its **output**: video at `target_fps`, and audio when `audio_out`;
//! - a **unit length** in milliseconds: the stretch of input the model
//!   consumes and of output it produces per step (1000/fps for a per-frame
//!   model; 160 ms for a Wan-Streamer-like agent);
//! - whether it takes a **context** at session start ([`SessionContext`]:
//!   scene and persona text, set once before streaming).
//!
//! Transports (Reactor `PublishTrack`, native WHIP ingest) decode the tracks
//! into [`InputFrame`]s and [`InputAudio`] chunks with presentation
//! timestamps; the engine reads them from bounded ring buffers
//! (`fastvideo_media::ring`).

use serde::{Deserialize, Serialize};

use crate::av::{Pcm, RgbFrame};
use crate::error::ApiError;
use crate::stream::SessionSpec;

/// A video codec a client may publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputVideoCodec {
    Vp8,
    H264,
}

impl InputVideoCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            InputVideoCodec::Vp8 => "vp8",
            InputVideoCodec::H264 => "h264",
        }
    }
}

/// The input video track: what the model reads and what a client may send.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VideoInputCaps {
    /// The model's input size: every accepted frame is decoded and scaled
    /// (fit, padded black) to exactly this.
    pub width: u32,
    pub height: u32,
    /// The largest picture a client may send (either orientation). Larger
    /// keyframes are refused: their frames are dropped until a keyframe
    /// within the limit arrives.
    pub max_width: u32,
    pub max_height: u32,
    /// Frames per second above this are not buffered.
    pub max_fps: u32,
    /// Accepted codecs, in preference order.
    pub codecs: Vec<InputVideoCodec>,
}

impl VideoInputCaps {
    /// Whether a `w`x`h` picture is within the limit (either orientation).
    pub fn allows_size(&self, w: u32, h: u32) -> bool {
        let (long, short) = (w.max(h), w.min(h));
        let (lmax, smax) = (self.max_width.max(self.max_height), self.max_width.min(self.max_height));
        w > 0 && h > 0 && long <= lmax && short <= smax
    }
}

/// The input audio track (Opus on the wire, decoded to PCM).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioInputCaps {
    /// PCM rate the model reads (48000: Opus's native rate).
    pub rate: u32,
    /// PCM channels the model reads (a stereo track is downmixed to mono).
    pub channels: u8,
}

/// Everything a duplex model accepts from the client.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputCaps {
    pub video: Option<VideoInputCaps>,
    pub audio: Option<AudioInputCaps>,
    /// Inbound media (all tracks) above this, averaged over 2 s, is dropped;
    /// the answer SDP also carries it as `b=AS` so browsers encode under it.
    pub max_bitrate_kbps: u32,
    /// How much decoded input the ring buffers hold before dropping the
    /// oldest (the engine reads the newest when it is behind).
    pub buffer_ms: u32,
}

impl InputCaps {
    pub fn accepts_video(&self) -> bool {
        self.video.is_some()
    }
    pub fn accepts_audio(&self) -> bool {
        self.audio.is_some()
    }
}

/// Streaming shape of a duplex model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DuplexCaps {
    /// Model step in milliseconds of input consumed and output produced.
    pub unit_ms: u32,
    /// Output video rate.
    pub target_fps: u32,
    pub input: InputCaps,
    /// The output carries audio (a talking avatar, an agent's speech).
    pub audio_out: bool,
    /// The model takes a [`SessionContext`] at session start.
    pub context: bool,
}

impl DuplexCaps {
    /// Units per second of output.
    pub fn units_per_second(&self) -> f64 {
        1000.0 / f64::from(self.unit_ms.max(1))
    }
}

/// Longest scene or persona text.
pub const CONTEXT_MAX_CHARS: usize = 2000;

/// Scene and persona context, set once at session start (a Wan-Streamer
/// style "world context" is prefilled before streaming).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionContext {
    /// The scene or world: setting, characters, ambient sound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<String>,
    /// Who the model plays: appearance, voice, behaviour.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
}

impl SessionContext {
    pub fn is_empty(&self) -> bool {
        self.scene.as_deref().is_none_or(str::is_empty) && self.persona.as_deref().is_none_or(str::is_empty)
    }

    /// Length checks (`CONTEXT_MAX_CHARS` each).
    pub fn validate(&self) -> Result<(), ApiError> {
        for (name, v) in [("context.scene", &self.scene), ("context.persona", &self.persona)] {
            if v.as_deref().is_some_and(|s| s.chars().count() > CONTEXT_MAX_CHARS) {
                return Err(ApiError::invalid_param(name, format!("longer than {CONTEXT_MAX_CHARS} characters")));
            }
        }
        Ok(())
    }
}

/// What a front-end asks the engine to open for a duplex model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DuplexSpec {
    pub session: SessionSpec,
    #[serde(default)]
    pub context: SessionContext,
}

/// One decoded input video frame.
#[derive(Clone, Debug, PartialEq)]
pub struct InputFrame {
    /// RGB24 at the model's input size.
    pub frame: RgbFrame,
    /// Presentation time from the track's RTP clock, microseconds since the
    /// track's first frame.
    pub pts_us: u64,
    /// The source picture size before scaling.
    pub source: (u32, u32),
}

/// One decoded input audio chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct InputAudio {
    /// At [`AudioInputCaps::rate`] / `channels`.
    pub pcm: Pcm,
    /// Presentation time of the first sample, microseconds since the
    /// track's first packet.
    pub pts_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> VideoInputCaps {
        VideoInputCaps {
            width: 640,
            height: 360,
            max_width: 1280,
            max_height: 720,
            max_fps: 30,
            codecs: vec![InputVideoCodec::Vp8, InputVideoCodec::H264],
        }
    }

    #[test]
    fn size_limit_accepts_either_orientation() {
        let c = caps();
        assert!(c.allows_size(1280, 720));
        assert!(c.allows_size(720, 1280));
        assert!(c.allows_size(640, 480));
        assert!(!c.allows_size(1920, 1080));
        assert!(!c.allows_size(1281, 10));
        assert!(!c.allows_size(0, 10));
    }

    #[test]
    fn context_limits() {
        assert!(SessionContext::default().is_empty());
        let c = SessionContext { scene: Some("a kitchen".into()), persona: None };
        assert!(!c.is_empty());
        c.validate().unwrap();
        let long = SessionContext { scene: None, persona: Some("x".repeat(CONTEXT_MAX_CHARS + 1)) };
        assert_eq!(long.validate().unwrap_err().param.as_deref(), Some("context.persona"));
        assert!(serde_json::from_str::<SessionContext>(r#"{"scene":"s","mood":"x"}"#).is_err());
    }

    #[test]
    fn duplex_caps_serde_shape() {
        let d = DuplexCaps {
            unit_ms: 160,
            target_fps: 25,
            input: InputCaps { video: Some(caps()), audio: None, max_bitrate_kbps: 4000, buffer_ms: 500 },
            audio_out: true,
            context: true,
        };
        let v = serde_json::to_value(crate::StreamCaps::Duplex(d.clone())).unwrap();
        assert_eq!(v["duplex"]["unit_ms"], 160);
        assert_eq!(v["duplex"]["input"]["video"]["codecs"], serde_json::json!(["vp8", "h264"]));
        let back: crate::StreamCaps = serde_json::from_value(v).unwrap();
        assert_eq!(back, crate::StreamCaps::Duplex(d.clone()));
        assert!((d.units_per_second() - 6.25).abs() < 1e-9);
    }
}
