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
//! - [`multiworker`]: the routes a replica serves behind a load balancer
//!   with `server.workers_max > 1` (design §6.2).
//! - [`native`] serves `/fv/v1/*`; serve-kit's `admin_routes` serve
//!   `/fv/v1/admin/keys` (minted API keys, admin token).
//! - [`console`] serves the `/console` pages (docs/serve/console.md).
//! - [`deploy`]: Runpod queue mode, the Vast forwarder route and the
//!   `info` diagnostics (WP-16, over `fastvideo-deploy`).
//! - [`shutdown`] waits for signals and [`app::drain`] drains.
//! - `director` (features `fal` + `webrtc`) mounts the fal WMA director:
//!   the WebRTC host from `[webrtc]` and the engine seam.
//! - [`encoders`] resolves the `auto` H.264 encoder settings at startup
//!   (NVENC probe, else OpenH264).
//! - [`whip`] fixes the WHIP encoder geometry (Cloudflare gets a padded
//!   1280x720 frame).
//! - `reactor` (feature `reactor`) builds the Reactor local runtime from
//!   `[reactor]` (WP-13) on the shared WebRTC host of `rtc` (`[webrtc]`).

pub mod adapters;
pub mod app;
pub mod config;
pub mod console;
pub mod deploy;
#[cfg(all(feature = "fal", feature = "webrtc"))]
pub mod director;
pub mod encoders;
pub mod gate;
/// Gateway mode (`engine.backend = "remote"`, docs/serve/gateway.md).
#[cfg(feature = "http-client")]
pub mod autoscale;
#[cfg(feature = "http-client")]
pub mod gateway;
pub mod health;
pub mod metrics;
pub mod multiworker;
pub mod native;
#[cfg(feature = "reactor")]
pub mod reactor;
#[cfg(any(feature = "reactor", all(feature = "fal", feature = "webrtc")))]
pub mod rtc;
pub mod router;
pub mod shutdown;
pub mod storage;
pub mod streams;
pub mod whip;
/// The worker role behind a gateway (`server.role = "worker"`).
#[cfg(feature = "http-client")]
pub mod worker;

pub use app::{App, Overrides};
pub use config::Config;
