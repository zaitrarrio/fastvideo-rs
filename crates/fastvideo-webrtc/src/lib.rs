//! WebRTC transport for fv-serve (design §5.8, WP-04).
//!
//! - [`host`] (feature `str0m`): the str0m peer host. One UDP mux socket and
//!   one ICE-TCP passive listener carry every peer; answers are non-trickle
//!   with every candidate plus `a=end-of-candidates`; client trickle is
//!   accepted; data channels; pre-encoded H.264/Opus writers; per-mid pause
//!   gate; PLI events; stats; negotiation/idle/ICE timeouts; receiving client
//!   camera/microphone tracks on a bounded, bitrate-capped queue.
//! - [`ingest`] (feature `str0m`): client media → keyframe gate → ffmpeg
//!   decode (warm spare) / Opus decode (`opus`) → the session's input rings
//!   (design §5.11).
//! - [`whip`]: the WHIP publisher. HTTP behind `whip`; the full publisher
//!   (offer, POST, answer, DELETE) needs `whip` + `str0m`.
//! - Pure, always-built helpers: [`sdp`] (validation and munging), [`ice`]
//!   (ICE servers for clients, public addresses, candidate plans),
//!   [`channel`] (labels, policies), [`writer`] (frame types, PLI limiter),
//!   [`framing`] (RFC 4571), [`stun`] (binding client codec), [`profile`]
//!   (per-sink resolution cap and H.264 level).
//!
//! Owned by WP-04 (docs/serve/design.md §8). Scaffolded by WP-00.

pub mod channel;
pub mod framing;
#[cfg(feature = "str0m")]
pub mod host;
pub mod ice;
#[cfg(feature = "str0m")]
pub mod ingest;
pub mod profile;
pub mod sdp;
pub mod stun;
pub mod whip;
pub mod writer;

/// Re-export so front-ends and tests can name str0m types without a
/// separate dependency.
#[cfg(feature = "str0m")]
pub use str0m;

/// Errors from the WebRTC layer.
#[derive(Debug, thiserror::Error)]
pub enum WebrtcError {
    #[error("invalid sdp: {0}")]
    Sdp(#[from] sdp::SdpError),
    #[error("webrtc: {0}")]
    Rtc(String),
    #[error("config: {0}")]
    Config(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("peer limit ({0}) reached")]
    PeerLimit(usize),
    #[error("webrtc host is not running")]
    HostGone,
    #[error("peer is closed")]
    PeerGone,
    #[error("queue full, message dropped")]
    Backpressure,
    #[error("data channel {0:?} is closed")]
    ChannelClosed(String),
    #[error("stun: {0}")]
    Stun(String),
    #[error("whip: {0}")]
    Whip(String),
    #[error("whip endpoint answered HTTP {status}: {body}")]
    WhipStatus { status: u16, body: String },
}
