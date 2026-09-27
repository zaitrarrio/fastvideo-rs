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
//! - [`native`] serves `/fv/v1/*`; serve-kit's `admin_routes` serve
//!   `/fv/v1/admin/keys` (minted API keys, admin token).
//! - [`console`] serves the `/console` pages (docs/serve/console.md).
//! - [`deploy`]: Runpod queue mode, the Vast forwarder route and the
//!   `info` diagnostics (WP-16, over `fastvideo-deploy`).
//! - [`shutdown`] waits for signals and [`app::drain`] drains.
//! - [`whip`] fixes the WHIP encoder geometry (Cloudflare gets a padded
//!   1280x720 frame).

pub mod adapters;
pub mod app;
pub mod config;
pub mod console;
pub mod deploy;
pub mod gate;
pub mod health;
pub mod metrics;
pub mod native;
pub mod router;
pub mod shutdown;
pub mod storage;
pub mod whip;

pub use app::{App, Overrides};
pub use config::Config;
