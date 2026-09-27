//! Reactor local runtime contract (design §5.7, WP-13): a Rust stand-in for
//! the Python Reactor Runtime that unmodified Reactor SDKs connect to.
//!
//! | Module | What |
//! |---|---|
//! | [`pb`] | `reactor_wire.v1` prost bindings from the vendored `proto/` (Apache-2.0, `proto/LICENSE` + `proto/NOTICE`) |
//! | [`wire`] | v1 protobuf and legacy v0 JSON codecs, one vocabulary, sniffed per connection |
//! | [`session`] | the RT state machine (CREATED … TERMINATED), the fixed-id local session, descriptor, start/stop, orphan timeout |
//! | [`signalling`] | `/sessions/{sid}/transport/webrtc/…`: connections, async answers (202 → 200 once), trickle buffering |
//! | [`gateway`] | per connection: v0/v1 latch, 20 s watchdog, pause gate, routing |
//! | [`commands`] / [`schema`] | fast-h3 (clip) and SF-Wan (causal) command sets; the OpenAPI `/schema` |
//! | [`clip`] / [`causal`] | the session drivers over the [`engine`] seam |
//! | [`media`] | pacing, encode once per codec (H.264 / VP8), 10 ms mono Opus, black frames |
//! | [`http`] | the axum router ([`router`]) |
//!
//! Tracks come from the model: `[main_video, main_audio]` for audio models
//! and `[main_video]` for video-only ones, identical in the descriptor,
//! `x-reactor.tracks` and `track_map` (design §5.3).
//!
//! Owned by WP-13 (docs/serve/design.md §8).

pub mod causal;
pub mod clip;
pub mod commands;
pub mod driver;
pub mod engine;
pub mod gateway;
pub mod http;
pub mod journal;
pub mod media;
pub mod pb;
pub mod schema;
pub mod session;
pub mod signalling;
pub mod wire;

pub use engine::{LoadState, Mode, StreamEngine};
pub use http::router;
pub use media::H264Backend;
pub use session::{Reactor, ReactorConfig, RtState, SESSION_ID};
