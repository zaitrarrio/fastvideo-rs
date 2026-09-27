//! `fv-serve`: one Rust server for the FastVideo, MiniMax, fal, LTX and
//! Reactor APIs over the fastvideo-rs engines (docs/serve/design.md, WP-10).
//!
//! - [`config`]: TOML + `FV_*` environment, secret redaction.
//! - [`gate`]: serve-kit's `EngineGate` over `EngineService` and the
//!   `EngineEvent` → `JobEvent` pump.
//! - [`storage`]: job store (memory / file / Cloudflare D1) and artifact
//!   store (local / S3-compatible R2).
//! - [`router`]: the §9 route table and its collision check; [`app`]
//!   assembles the router; [`adapters`] holds the feature-gated mount points.
//! - [`health`] serves `/health`, `/healthz`, `/ping`, `/` and `/metrics`
//!   (Prometheus, via [`metrics`]).
//! - [`native`] serves `/fv/v1/*`.
//! - [`shutdown`] waits for signals and [`app::drain`] drains.
//! - [`whip`] fixes the WHIP encoder geometry (Cloudflare gets a padded
//!   1280x720 frame).
//! - `reactor` (feature `reactor`) binds the WebRTC host from `[webrtc]` and
//!   builds the Reactor local runtime from `[reactor]` (WP-13).

pub mod adapters;
pub mod app;
pub mod config;
pub mod gate;
pub mod health;
pub mod metrics;
pub mod native;
#[cfg(feature = "reactor")]
pub mod reactor;
pub mod router;
pub mod shutdown;
pub mod storage;
pub mod whip;

pub use app::{App, Overrides};
pub use config::Config;
