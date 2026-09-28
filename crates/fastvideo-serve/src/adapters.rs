//! Mount points for the external-API adapter crates (design §2.1, §9).
//!
//! Each adapter is a cargo feature of this crate (`openai-videos`,
//! `minimax`, `ltxapi`, `fal`, `reactor`) and a `[protocols]` switch at run
//! time. Mounted today (default features):
//!
//! | API | Call | Also wired here |
//! |---|---|---|
//! | FastVideo `/v1/videos` + FastWan | `videos::routes` + `models::routes` (`openai_videos`), `fastwan::routes` (`fastwan`) | `/` and `/health` bodies (FastWan model) |
//! | MiniMax V2 | `MiniMax::new(cfg).router()` | callback renderer (`callback_url`) |
//! | LTX | `fastvideo_ltxapi::router(LtxConfig)` | — |
//! | fal queue/sync | `fastvideo_fal::router(ctx, FalConfig)` (a stateful `Router`) | `FalWebhook` renderer, output file names |
//!
//! | fal director (WMA) | `fastvideo_fal::director::routes(DirectorService)` merged into the fal router (`router_with`), built in `App::build` by `crate::director` | needs features `fal` + `webrtc` |
//!
//! The Reactor runtime (WP-13) and the fal director (WP-14) answer WebRTC
//! offers, so they are built in `App::build` on one shared WebRTC host
//! (`crate::rtc`, from `[webrtc]`): the Reactor router is mounted next to
//! these routers, the director's routes go into the fal router.
//!
//! Adapters cannot depend on this crate, so their settings are their own
//! config types, built here from the `[protocols]`/`[ltx]` sections.

use std::sync::Arc;

use axum::Router;
use fastvideo_protocol::{Job, ProtocolId};
use fastvideo_serve_kit::{CallbackRender, ServeCtx};

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
        Mounted { api: "fal", enabled: p.fal, built: cfg!(feature = "fal") },
        Mounted { api: "fal_director", enabled: p.fal && p.fal_director, built: cfg!(all(feature = "fal", feature = "webrtc")) },
        Mounted { api: "reactor", enabled: p.reactor, built: cfg!(feature = "reactor") },
    ]
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

/// `fastvideo_minimax::MiniMaxConfig` from the server config.
#[cfg(feature = "minimax")]
pub fn minimax_config(cfg: &MountCfg) -> fastvideo_minimax::MiniMaxConfig {
    fastvideo_minimax::MiniMaxConfig {
        url_ttl: cfg.url_ttl,
        body_max: cfg.body_max,
        ..fastvideo_minimax::MiniMaxConfig::default()
    }
}

/// `owner/alias` app ids → fal apps: `minimax/h3-{max,turbo,draft}` get their
/// H3 tier fallback; any other id resolves its alias part by name only.
#[cfg(feature = "fal")]
pub fn fal_config(cfg: &MountCfg) -> fastvideo_fal::FalConfig {
    use fastvideo_protocol::Tier;
    let apps = cfg
        .protocols
        .fal_apps
        .iter()
        .map(|id| {
            let id = id.trim_matches('/');
            match id {
                "minimax/h3-max" => fastvideo_fal::FalApp::h3(Tier::Max),
                "minimax/h3-turbo" => fastvideo_fal::FalApp::h3(Tier::Turbo),
                "minimax/h3-draft" => fastvideo_fal::FalApp::h3(Tier::Draft),
                other => fastvideo_fal::FalApp {
                    id: other.to_owned(),
                    model: other.rsplit('/').next().unwrap_or(other).to_owned(),
                    tier: None,
                },
            }
        })
        .collect();
    fastvideo_fal::FalConfig {
        apps,
        url_ttl: cfg.url_ttl,
        body_max: cfg.body_max,
        ..fastvideo_fal::FalConfig::default()
    }
}

/// fal app ids → the model name each resolves to (the H3 tier alias for
/// `minimax/h3-{max,turbo,draft}`, else the alias part): the gateway's
/// director routing (docs/serve/gateway.md §5.1).
pub fn fal_app_models(apps: &[String]) -> Vec<(String, String)> {
    apps.iter()
        .map(|id| {
            let id = id.trim_matches('/').to_owned();
            let model = match id.as_str() {
                "minimax/h3-max" => "h3-max".to_owned(),
                "minimax/h3-turbo" => "h3-turbo".to_owned(),
                "minimax/h3-draft" => "h3-draft".to_owned(),
                other => other.rsplit('/').next().unwrap_or(other).to_owned(),
            };
            (id, model)
        })
        .collect()
}

/// Callback renderers to register on the `ServeCtx` builder.
pub fn renderers(cfg: &MountCfg) -> Vec<(ProtocolId, Arc<dyn CallbackRender>)> {
    #[allow(unused_mut)]
    let mut v: Vec<(ProtocolId, Arc<dyn CallbackRender>)> = Vec::new();
    #[cfg(feature = "minimax")]
    if cfg.protocols.minimax {
        let mm = fastvideo_minimax::MiniMax::new(minimax_config(cfg));
        v.push((ProtocolId::MiniMaxV2, fastvideo_minimax::callback_renderer(&mm)));
    }
    #[cfg(feature = "fal")]
    if cfg.protocols.fal {
        v.push((ProtocolId::Fal, Arc::new(fastvideo_fal::FalWebhook { url_ttl: cfg.url_ttl })));
    }
    let _ = cfg;
    v
}

/// The output file name a job's artifact is stored under: fal's own
/// `<nanoid21>_minimax-h3.mp4`, else `<job id>.mp4`.
pub fn artifact_file_name(job: &Job) -> String {
    #[cfg(feature = "fal")]
    if job.protocol == ProtocolId::Fal {
        return fastvideo_fal::output_file_name(job);
    }
    format!("{}.mp4", job.id.0.simple())
}

/// The model `GET /` names: FastWan's model when FastWan is mounted.
pub fn root_model(cfg: &MountCfg, ctx: &ServeCtx) -> Option<String> {
    #[cfg(feature = "openai-videos")]
    if cfg.protocols.fastwan {
        return fastvideo_openai_videos::fastwan::fastwan_model(&ctx.engine().models(), None);
    }
    let _ = (cfg, ctx);
    None
}

/// The enabled, built adapters: routes that still need the `ServeCtx`
/// state, and routers that already carry it (fal). `fal_extra` is merged
/// into the fal router (the director's routes, so `/fal/proxy` reaches
/// them).
pub fn mount(cfg: &MountCfg, ctx: &ServeCtx, fal_extra: Router<ServeCtx>) -> (Router<ServeCtx>, Router) {
    #[allow(unused_mut)]
    let mut r: Router<ServeCtx> = Router::new();
    #[allow(unused_mut)]
    let mut stateful: Router = Router::new();
    let p = &cfg.protocols;
    #[cfg(feature = "openai-videos")]
    {
        use fastvideo_openai_videos as ov;
        let vc = Arc::new(ov::VideosConfig::default());
        if p.openai_videos {
            r = r.merge(ov::videos::routes(vc.clone())).merge(ov::models::routes(vc.clone()));
        }
        if p.fastwan {
            r = r.merge(ov::fastwan::routes(ctx, &vc));
        }
    }
    #[cfg(feature = "minimax")]
    if p.minimax {
        r = r.merge(Arc::new(fastvideo_minimax::MiniMax::new(minimax_config(cfg))).router());
    }
    #[cfg(feature = "ltxapi")]
    if p.ltx {
        r = r.merge(fastvideo_ltxapi::router(ltx_config(cfg)));
    }
    #[cfg(feature = "fal")]
    if p.fal {
        stateful = stateful.merge(fastvideo_fal::router_with(ctx.clone(), fal_config(cfg), fal_extra));
    }
    for m in inventory(p) {
        if m.enabled && !m.built {
            tracing::warn!(api = m.api, "API enabled in [protocols] but not built into this binary; not mounted");
        }
    }
    #[cfg(not(feature = "fal"))]
    let _ = fal_extra;
    let _ = ctx;
    (r, stateful)
}
