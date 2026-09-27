//! Mount points for the external-API adapter crates (design §2.1, §9).
//!
//! Each adapter is behind a cargo feature of this crate (`openai-videos`,
//! `minimax`, `ltxapi`, `fal`, `reactor`) and a `[protocols]` switch at run
//! time. **Contract for the adapter packages (WP-06..09, WP-13/14):** export
//!
//! ```ignore
//! pub fn router() -> axum::Router<fastvideo_serve_kit::ServeCtx>;           // openai-videos, minimax, reactor
//! pub fn router(apps: &[String]) -> axum::Router<fastvideo_serve_kit::ServeCtx>; // fal (static app prefixes)
//! ```
//!
//! (adapters cannot depend on this crate, so anything more goes in the
//! adapter's own config type and the one-line call below changes when the
//! adapter merges), then add the crate's feature to `default` in `Cargo.toml`.
//! Mounted today: **ltxapi** (`fastvideo_ltxapi::router(LtxConfig)`, from
//! `[ltx]`). Until then the feature is off and the
//! API is reported as "not built" at startup. The route-collision test in
//! [`crate::router`] already reserves each adapter's paths from §9.

use axum::Router;
use fastvideo_serve_kit::ServeCtx;

use crate::config::{LtxCfg, ProtocolsCfg};

/// What an adapter needs from the server config.
#[derive(Clone, Debug)]
pub struct MountCfg {
    pub protocols: ProtocolsCfg,
    /// Request body cap in bytes (`limits.body_max_mb`).
    pub body_max: usize,
    /// Wait of sync endpoints.
    pub sync_timeout: std::time::Duration,
    /// Signed-URL lifetime (`artifacts.url_ttl_s`).
    pub url_ttl: std::time::Duration,
    pub ltx: LtxCfg,
}

/// `fastvideo_ltxapi::LtxConfig` from `[ltx]`.
#[cfg(feature = "ltxapi")]
pub fn ltx_config(cfg: &MountCfg) -> fastvideo_ltxapi::LtxConfig {
    use std::time::Duration;
    fastvideo_ltxapi::LtxConfig {
        sync_timeout: Some(cfg.ltx.sync_timeout_s.map_or(cfg.sync_timeout, Duration::from_secs)),
        url_ttl: cfg.ltx.url_ttl_s.map_or(cfg.url_ttl, Duration::from_secs),
        v1_concurrency: (cfg.ltx.v1_concurrency > 0).then_some(cfg.ltx.v1_concurrency),
        body_max_bytes: cfg.body_max,
        ..fastvideo_ltxapi::LtxConfig::default()
    }
}

/// One adapter's mount status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mounted {
    pub api: &'static str,
    pub enabled: bool,
    pub built: bool,
}

/// Every adapter and whether this build includes it.
pub fn inventory(p: &ProtocolsCfg) -> Vec<Mounted> {
    vec![
        Mounted { api: "openai_videos", enabled: p.openai_videos || p.fastwan, built: cfg!(feature = "openai-videos") },
        Mounted { api: "minimax", enabled: p.minimax, built: cfg!(feature = "minimax") },
        Mounted { api: "ltx", enabled: p.ltx, built: cfg!(feature = "ltxapi") },
        Mounted { api: "fal", enabled: p.fal || p.fal_director, built: cfg!(feature = "fal") },
        Mounted { api: "reactor", enabled: p.reactor, built: cfg!(feature = "reactor") },
    ]
}

/// Merges every enabled, built adapter router.
pub fn mount(cfg: &MountCfg) -> Router<ServeCtx> {
    #[allow(unused_mut)]
    let mut r: Router<ServeCtx> = Router::new();
    let p = &cfg.protocols;
    #[cfg(feature = "openai-videos")]
    if p.openai_videos || p.fastwan {
        r = r.merge(fastvideo_openai_videos::router());
    }
    #[cfg(feature = "minimax")]
    if p.minimax {
        r = r.merge(fastvideo_minimax::router());
    }
    #[cfg(feature = "ltxapi")]
    if p.ltx {
        r = r.merge(fastvideo_ltxapi::router(ltx_config(cfg)));
    }
    #[cfg(feature = "fal")]
    if p.fal || p.fal_director {
        r = r.merge(fastvideo_fal::router(&p.fal_apps));
    }
    #[cfg(feature = "reactor")]
    if p.reactor {
        r = r.merge(fastvideo_reactor::router());
    }
    for m in inventory(p) {
        if m.enabled && !m.built {
            tracing::warn!(api = m.api, "API enabled in [protocols] but not built into this binary; not mounted");
        }
    }
    r
}
