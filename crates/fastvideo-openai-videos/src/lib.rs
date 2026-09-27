//! FastVideo `/v1/videos*`, `/v1/models*`, `/v1/model_info`, and the FastWan
//! `/generate`, `/status`, `/video` routes (design §4.1-4.2, WP-06).
//!
//! - [`videos`]: the FastVideo `fastvideo serve` job API: create (JSON,
//!   multipart, form), sync, list, retrieve, content, delete.
//! - [`models`]: model cards and public model-name resolution, including the
//!   tier ids (`h3-draft` / `h3-turbo` / `h3-max`, `ltx-*`, `wan-*`).
//! - [`fastwan`]: the FastWan Video API exactly as `fastwan_link.py` calls
//!   it (`prompt_id`, `processing`, string `error`, `/health.model_loaded`).
//! - [`error`]: the OpenAI and FastAPI error envelopes.
//!
//! Mount with [`router`] (both APIs) and `.with_state(ctx)`. `GET /` and
//! `GET /health` belong to the binary (design §9); [`fastwan::service_routes`]
//! offers them for a standalone FastWan server, and [`fastwan::health_body`] /
//! [`fastwan::root_body`] give the bodies to merge.
//!
//! Owned by WP-06 (docs/serve/design.md §8).

use std::sync::Arc;

use axum::Router;
use fastvideo_serve_kit::{IngestPolicy, ServeCtx};

pub mod error;
pub mod fastwan;
pub mod models;
pub mod videos;

pub use fastwan::{FastWanApi, FastWanGenerate, FastWanSubmit, FastWanView};
pub use videos::{OpenAiVideos, VideoGenerationRequest, VideosCreate, VideosView};

/// Adapter settings (the `[openai_videos]` part of the serve config).
#[derive(Clone, Debug)]
pub struct VideosConfig {
    /// The model a request without `model` gets, and the one `/v1/model_info`
    /// describes. `None`: the only served model, else the first clip model.
    pub default_model: Option<String>,
    /// The model FastWan `/generate` runs (the API has no model field).
    /// `None`: the first served Wan clip model.
    pub fastwan_model: Option<String>,
    /// `created` of every model card (unix seconds): server start by default.
    pub created: i64,
    /// Media limits for references (design §4.1: the default policy).
    pub ingest: IngestPolicy,
}

impl Default for VideosConfig {
    fn default() -> Self {
        Self {
            default_model: None,
            fastwan_model: None,
            created: time::OffsetDateTime::now_utc().unix_timestamp(),
            ingest: IngestPolicy::default(),
        }
    }
}

/// Both APIs: the FastVideo routes and the FastWan routes (without `/` and
/// `/health`, which the binary owns).
pub fn router(ctx: &ServeCtx, cfg: VideosConfig) -> Router<ServeCtx> {
    let cfg = Arc::new(cfg);
    videos::routes(cfg.clone())
        .merge(models::routes(cfg.clone()))
        .merge(fastwan::routes(ctx, &cfg))
}
