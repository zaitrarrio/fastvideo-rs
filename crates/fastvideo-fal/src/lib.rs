//! fal queue and sync API (design §4.4, WP-09) and the WMA director
//! streaming session (design §5.6, WP-14).
//!
//! Queue side (this package, WP-09):
//!
//! - [`schema`]: the t2v / i2v / r2v inputs with their defaults and
//!   constraints, normalization to `GenerationRequest`, the output `File`.
//! - [`queue`]: submit, status, result, cancel and the SSE status stream,
//!   under both request path forms, plus the pure [`queue::FalView`].
//! - [`sync`]: `POST /run/{app}/{sub}`.
//! - [`proxy`]: `ANY /fal/proxy`, routed by `x-fal-target-url`.
//! - [`storage`]: `POST /storage/upload/initiate`.
//! - [`webhook`]: webhook bodies (signed by serve-kit with our Ed25519 key)
//!   and `/.well-known/jwks.json`.
//! - [`error`]: the fal error envelopes.
//!
//! Apps are static routes built from [`FalConfig::apps`], never wildcards.
//! Each app resolves to a model by name (engine aliases / served names) and
//! falls back to its tier (design §0.3, §0.6): `minimax/h3-max` → H3 `Max`,
//! `minimax/h3-turbo` → `Turbo`, `minimax/h3-draft` → `Draft`.
//!
//! The binary mounts [`router`] and registers [`webhook::FalWebhook`] as the
//! `ProtocolId::Fal` callback renderer.
//!
//! Owned by WP-09 / WP-14 (docs/serve/design.md §8).

pub mod director;
pub mod error;
pub mod proxy;
pub mod queue;
pub mod schema;
pub mod storage;
pub mod sync;
pub mod webhook;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use fastvideo_protocol::{Family, Tier};
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};

pub use error::FalProtocol;
pub use queue::{FalEndpoint, FalView};
pub use schema::{output_file_name, Endpoint, FalInput};
pub use webhook::FalWebhook;

/// One fal app (`owner/alias`) this server answers for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FalApp {
    /// `minimax/h3-max`.
    pub id: String,
    /// The name resolved through the engine's aliases and served names
    /// (e.g. the tier alias `h3-max`).
    pub model: String,
    /// Fallback when `model` does not resolve: the served model of this
    /// family and tier.
    pub tier: Option<(Family, Tier)>,
}

impl FalApp {
    /// `minimax/h3-{max,turbo,draft}` → the H3 model of that tier.
    pub fn h3(tier: Tier) -> Self {
        let alias = format!("h3-{}", tier.as_str());
        Self { id: format!("minimax/{alias}"), model: alias, tier: Some((Family::H3, tier)) }
    }

    /// `owner/alias` with both parts non-empty and URL-safe.
    pub fn is_valid(&self) -> bool {
        let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        matches!(self.id.split_once('/'), Some((o, a)) if ok(o) && ok(a))
    }
}

/// `[fal]` configuration.
#[derive(Clone, Debug)]
pub struct FalConfig {
    /// Default: `minimax/h3-max`, `minimax/h3-turbo`, `minimax/h3-draft`.
    pub apps: Vec<FalApp>,
    /// Media fetch limits for `*_url` inputs.
    pub ingest: IngestPolicy,
    /// Lifetime of signed output and upload URLs (24 h: fal retention).
    pub url_ttl: Duration,
    /// Largest output inlined as a data URI for `sync_mode: true`.
    pub inline_max_bytes: u64,
    /// Request body cap (data URIs make bodies large).
    pub body_max: usize,
}

impl Default for FalConfig {
    fn default() -> Self {
        Self {
            apps: vec![FalApp::h3(Tier::Max), FalApp::h3(Tier::Turbo), FalApp::h3(Tier::Draft)],
            ingest: IngestPolicy::default(),
            url_ttl: Duration::from_secs(24 * 3600),
            inline_max_bytes: 64 * 1024 * 1024,
            body_max: 64 * 1024 * 1024,
        }
    }
}

/// Every fal route except `/fal/proxy`, still needing the `ServeCtx` state.
/// Invalid app ids are skipped with a warning.
pub fn routes(cfg: FalConfig) -> Router<ServeCtx> {
    let cfg = Arc::new(cfg);
    let mut r = Router::new();
    for app in &cfg.apps {
        if !app.is_valid() {
            tracing::warn!(app = %app.id, "fal app ids must be `owner/alias`; skipped");
            continue;
        }
        r = queue::app_routes(r, &cfg, app);
        r = sync::app_routes(r, &cfg, app);
    }
    r = storage::routes(r, &cfg);
    webhook::routes(r)
}

/// The fal API with its state, plus `/fal/proxy` dispatching into it.
pub fn router(ctx: ServeCtx, cfg: FalConfig) -> Router {
    let inner = routes(cfg).with_state(ctx);
    inner.clone().merge(proxy::proxy_router(inner))
}
