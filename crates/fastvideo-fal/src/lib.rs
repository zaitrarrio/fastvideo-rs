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
//! - [`catalog`]: `GET /fal/schema[/{app}/{sub}]`, the input JSON Schemas
//!   (from the same limits as [`schema`]) for the console.
//!
//! Apps are static routes built from [`FalConfig::apps`], never wildcards.
//! Each app resolves to a model by name (engine aliases / served names) and
//! falls back to its tier (design §0.3, §0.6): `minimax/h3-max` → H3 `Max`,
//! `minimax/h3-turbo` and fal's `minimax/h3-max-turbo` → `Turbo`,
//! `minimax/h3-draft` → `Draft`, fal's base `minimax/h3` → `Max` (480P and
//! 768P; 2K / 4K refused). fal's family apps pick a tier per endpoint
//! (docs/serve/fal-parity.md): `lightricks/ltx-2.5` (`…/fast` → LTX
//! `Turbo`, `…/pro` → `Max`) and `fal-ai/wan` (`v2.2-5b/…` → Wan `Max`,
//! `v2.2-5b/text-to-video/fast-wan` → `Turbo`). Sub-paths may have several
//! segments; the queue's status and result URLs are app-only.
//!
//! The binary mounts [`router`] and registers [`webhook::FalWebhook`] as the
//! `ProtocolId::Fal` callback renderer.
//!
//! Director side (WP-14): [`director`], mounted with [`router_with`] and
//! `director::routes` (feature `director`).
//!
//! Owned by WP-09 / WP-14 (docs/serve/design.md §8).

pub mod catalog;
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
pub use schema::{output_file_name, output_slug, AppKind, Endpoint, FalInput};
pub use webhook::FalWebhook;

/// The engine's canonical tier alias (`fastvideo_engine_service::tier_alias`):
/// `h3-max`, `ltx-pro`, `ltx-turbo`, `wan-max`, `wan-turbo`, ...
pub fn tier_alias(family: Family, tier: Tier) -> String {
    let fam = match family {
        Family::H3 => "h3",
        Family::Ltx2 => "ltx",
        Family::Wan => "wan",
        Family::MmAudio => "mmaudio",
        Family::Loopback => "loopback",
    };
    match (family, tier) {
        (Family::Ltx2, Tier::Max) => "ltx-pro".to_owned(),
        _ => format!("{fam}-{}", tier.as_str()),
    }
}

/// The apps mounted when none are configured.
pub const DEFAULT_APPS: [&str; 5] =
    ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft", "minimax/h3-max-turbo", "minimax/h3"];

/// The endpoint ids (`app/sub`) of a configured app id, for route tables.
pub fn endpoint_ids(app_id: &str) -> Vec<String> {
    let app = FalApp::from_id(app_id);
    app.endpoints().iter().map(|e| format!("{}/{}", app.id, e.sub())).collect()
}

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

    /// fal's `minimax/h3-max-turbo` (the H3 Turbo tier) and `minimax/h3`
    /// (base H3; served by the H3 Max tier, 480P / 768P only).
    pub fn h3_named(id: &str) -> Option<Self> {
        let (alias, tier) = match id {
            "minimax/h3-max" => ("h3-max", Tier::Max),
            "minimax/h3-turbo" => ("h3-turbo", Tier::Turbo),
            "minimax/h3-draft" => ("h3-draft", Tier::Draft),
            "minimax/h3-max-turbo" => ("h3-turbo", Tier::Turbo),
            "minimax/h3" => ("h3-max", Tier::Max),
            _ => return None,
        };
        Some(Self { id: id.to_owned(), model: alias.to_owned(), tier: Some((Family::H3, tier)) })
    }

    /// fal's LTX-2.5 (`lightricks/ltx-2.5`) and Wan (`fal-ai/wan`) apps: each
    /// endpoint picks its tier ([`Endpoint::target`]).
    pub fn family_app(id: &str) -> Option<Self> {
        let (model, tier) = match AppKind::of(id) {
            AppKind::Ltx25 => ("ltx-turbo", (Family::Ltx2, Tier::Turbo)),
            AppKind::Wan => ("wan-max", (Family::Wan, Tier::Max)),
            AppKind::LtxQuality => ("ltx-pro", (Family::Ltx2, Tier::Max)),
            AppKind::H3 | AppKind::H3Base => return None,
        };
        Some(Self { id: id.to_owned(), model: model.to_owned(), tier: Some(tier) })
    }

    /// The app for a configured id: the named H3 and family apps get their
    /// tier fallbacks; any other id resolves its alias part by name only.
    pub fn from_id(id: &str) -> Self {
        let id = id.trim_matches('/');
        Self::h3_named(id).or_else(|| Self::family_app(id)).unwrap_or_else(|| Self {
            id: id.to_owned(),
            model: id.rsplit('/').next().unwrap_or(id).to_owned(),
            tier: None,
        })
    }

    /// The schema family (from the id).
    pub fn kind(&self) -> AppKind {
        AppKind::of(&self.id)
    }

    /// The endpoints this app serves.
    pub fn endpoints(&self) -> &'static [Endpoint] {
        self.kind().endpoints()
    }

    /// `(name, tier fallback)` an endpoint resolves through: the endpoint's
    /// family tier on the LTX and Wan apps, else the app's model.
    pub fn target(&self, e: Endpoint) -> (String, Option<(Family, Tier)>) {
        match e.target() {
            Some((family, tier)) => (tier_alias(family, tier), Some((family, tier))),
            None => (self.model.clone(), self.tier),
        }
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
    /// Default: `minimax/h3-max`, `minimax/h3-turbo`, `minimax/h3-draft`,
    /// `minimax/h3-max-turbo`, `minimax/h3`.
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
            apps: DEFAULT_APPS.iter().map(|id| FalApp::from_id(id)).collect(),
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
    r = catalog::routes(r, &cfg);
    webhook::routes(r)
}

/// The fal API with its state, plus `/fal/proxy` dispatching into it.
pub fn router(ctx: ServeCtx, cfg: FalConfig) -> Router {
    router_with(ctx, cfg, Router::new())
}

/// [`router`] plus `extra` routes (the director's, see
/// `director::routes`) that `/fal/proxy` also reaches: `x-fal-target-url`
/// `https://wma.fal.run/session` maps to `/wma/session`.
pub fn router_with(ctx: ServeCtx, cfg: FalConfig, extra: Router<ServeCtx>) -> Router {
    let inner = routes(cfg).merge(extra).with_state(ctx);
    inner.clone().merge(proxy::proxy_router(inner))
}
