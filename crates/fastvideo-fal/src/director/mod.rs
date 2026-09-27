//! WMA director: `minimax/h3-max/director` as a realtime WebRTC session
//! (design §5.6, WP-14; fal §8).
//!
//! The browser opens it with `fal.realtime.open(wma("minimax/h3-max/director"))`
//! from `@fal-ai/client`: `POST /wma/ice` for ICE servers, a complete
//! (non-trickle) offer to `POST /wma/session`, heartbeats every 5 s to
//! `POST /wma/session/heartbeat`, then media and a client-created `control`
//! data channel carrying JSON text straight between browser and runner.
//! Our server is both the bridge and the runner.
//!
//! | Module | What |
//! |---|---|
//! | [`messages`] | strict client schemas (`invalid_message` on anything extra) and the server message shapes |
//! | [`control`] | the pure control state machine: configure once, prompt versions, `replan`, the planned deck, scripts |
//! | [`info`] | `session_info` / `DirectorInfo` with **our** constants |
//! | [`engine`] | the thin adapter trait over the engine's clip sessions (fv-serve implements it over `EngineService`) |
//! | [`vp8`] | intra-only VP8 (libwebp), the fallback when ffmpeg has no `libvpx` (else `fastvideo_media::vp8`), for offers without H.264 (open-source Chromium) |
//! | `service`, `session`, `media`, `routes` (feature `director`) | the WebRTC runtime: admission, heartbeats, the session task, lockstep playout and encoding, the HTTP routes |
//!
//! Chunks run as a clip session with continuity `AnchorLastFrame` (chunk
//! N+1 starts from chunk N's last frame) and autoplay; each built chunk is
//! reported with `chunk` + `chunk_metrics`, late ones with
//! `deadline_missed` (freeze video, silence audio), and `session_metrics`
//! goes out every 10 s and at the end with `final: true`.

pub mod control;
pub mod engine;
pub mod info;
pub mod messages;
pub mod vp8;

#[cfg(feature = "director")]
pub mod media;
#[cfg(feature = "director")]
pub mod routes;
#[cfg(feature = "director")]
pub mod service;
#[cfg(feature = "director")]
pub mod session;

pub use engine::{ChunkBuild, ChunkOutput, DirectorClips, DirectorEngine};

#[cfg(feature = "director")]
pub use routes::routes;
#[cfg(feature = "director")]
pub use service::{DirectorConfig, DirectorService};
