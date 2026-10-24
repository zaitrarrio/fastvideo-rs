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
//!   (Prometheus, via [`metrics`]); [`build_info`] is the build and image
//!   identity they (and `fv-serve --version`) report.
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
//! - [`whip`] fixes the WHIP encoder geometry (native canvas; the offered
//!   H.264 level per target).
//! - [`stream_outputs`]: where live streams publish (Cloudflare Stream by default,
//!   WHIP, RTMP(S), SRT), for native streams and director / Reactor output
//!   copies (docs/serve/stream-outputs.md).
//! - `reactor` (feature `reactor`) builds the Reactor local runtime from
//!   `[reactor]` (WP-13) on the shared WebRTC host of `rtc` (`[webrtc]`).

pub mod adapters;
pub mod admin_token;
pub mod app;
/// A worker's socket to its pool's Durable Object (`[dispatch] do_url`,
/// docs/serve/gateway-cloudflare.md).
#[cfg(feature = "http-client")]
pub mod arbiter;
pub mod build_info;
pub mod config;
pub mod console;
pub mod deploy;
#[cfg(all(feature = "fal", feature = "webrtc"))]
pub mod director;
/// The edge on the host: family objects, the registry and the public front
/// in one process (tests, the compat suites, `fv-edge-local`).
#[cfg(feature = "http-client")]
pub mod edge_host;
#[cfg(feature = "http-client")]
pub mod edge_link;
/// Edge sessions held here and the local sessions they stand for.
pub mod edge_sessions;
pub mod encoders;
pub mod flags;
/// An API front behind the edge Worker (docs/serve/edge-control-plane.md).
#[cfg(feature = "http-client")]
pub mod front;
pub mod gate;
pub mod health;
/// Native WHIP ingest into duplex models (`/fv/v1/streams/ingest`).
#[cfg(feature = "webrtc")]
pub mod ingest;
pub mod log_ship;
pub mod metrics;
pub mod moderation;
pub mod multiworker;
pub mod native;
#[cfg(feature = "otel")]
pub mod otel;
#[cfg(feature = "reactor")]
pub mod reactor;
pub mod router;
#[cfg(feature = "webrtc")]
pub mod rtc;
pub mod segments;
pub mod shutdown;
pub mod status;
pub mod storage;
pub mod stream_outputs;
pub mod streams;
pub mod tokens;
pub mod trace_http;
/// The worker role behind a gateway (`server.role = "worker"`).
#[cfg(feature = "http-client")]
pub mod upload;
pub mod whip;
#[cfg(feature = "http-client")]
pub mod worker;

pub use app::{App, Overrides};
pub use config::Config;
