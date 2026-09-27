//! `/v1/models*` and `/v1/model_info` (design §4.1), and public model-name
//! resolution for both APIs.
//!
//! Public names are every model's `served_names` plus the canonical tier id of
//! each tiered model (design §0.3, §0.6): `h3-draft` / `h3-turbo` / `h3-max`,
//! `ltx-draft` / `ltx-turbo` / `ltx-pro`, `wan-draft` / `wan-turbo` /
//! `wan-max`. A card's `root` is the engine model id the name runs.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use fastvideo_engine_service::{parse_tier_alias, tier_alias};
use fastvideo_protocol::{
    resolve_model, resolve_tier, ApiError, HttpReply, ModelCaps, ProtocolId, StreamCaps,
};
use fastvideo_serve_kit::{into_response, EngineGate, ServeCtx};
use serde_json::{json, Value};

use crate::error::{openai_error, openai_error_status};
use crate::VideosConfig;

/// The public names of the served models, in model order: each model's
/// served names, then its tier id. `(public name, engine model id)`, no
/// duplicates.
pub fn public_names(models: &[ModelCaps]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |name: &str, root: &str| {
        if !out.iter().any(|(n, _)| n == name) {
            out.push((name.to_owned(), root.to_owned()));
        }
    };
    for m in models {
        for n in &m.served_names {
            push(n, &m.id.0);
        }
    }
    for m in models {
        if let Some(alias) = m.tier.and_then(|t| tier_alias(m.family, t)) {
            push(alias, &m.id.0);
        }
    }
    out
}

/// Resolves a public name: the gate's aliases, a model id or served name,
/// then a tier id (`h3-turbo`, ...) through the models' tier tags. Unknown ->
/// 400 naming `model`.
pub fn resolve_public(engine: &dyn EngineGate, name: &str) -> Result<ModelCaps, ApiError> {
    let models = engine.models();
    if let Ok(c) = resolve_model(name, |n| engine.alias(n), &models) {
        return Ok(c.clone());
    }
    if let Some((family, tier)) = parse_tier_alias(name) {
        if let Ok(c) = resolve_tier(family, tier, &models) {
            return Ok(c.clone());
        }
    }
    Err(ApiError::invalid_param(
        "model",
        format!("model `{name}` is not served here"),
    ))
}

/// The model a request without `model` gets: the configured default, else
/// the only served model, else the first model that is not causal-only.
pub fn default_model(models: &[ModelCaps], configured: Option<&str>) -> Option<String> {
    if let Some(c) = configured {
        return Some(c.to_owned());
    }
    let first_name = |m: &ModelCaps| {
        m.served_names
            .first()
            .cloned()
            .unwrap_or_else(|| m.id.0.clone())
    };
    if models.len() == 1 {
        return models.first().map(first_name);
    }
    models
        .iter()
        .find(|m| !matches!(m.stream, Some(StreamCaps::Causal { .. })))
        .or_else(|| models.first())
        .map(first_name)
}

/// One FastVideo model card.
pub fn card(id: &str, root: &str, created: i64) -> Value {
    json!({ "id": id, "object": "model", "created": created, "owned_by": "fastvideo", "root": root })
}

/// `GET /v1/models` body.
pub fn list_body(models: &[ModelCaps], created: i64) -> Value {
    let data: Vec<Value> = public_names(models)
        .iter()
        .map(|(n, r)| card(n, r, created))
        .collect();
    json!({ "object": "list", "data": data })
}

/// `GET /v1/models/{model}`: the card, or 404.
pub fn get_reply(models: &[ModelCaps], name: &str, created: i64) -> HttpReply {
    match public_names(models).into_iter().find(|(n, _)| n == name) {
        Some((n, r)) => HttpReply::json(200, card(&n, &r, created)),
        None => openai_error_status(
            404,
            &ApiError::not_found(format!("The model `{name}` does not exist.")).with_param("model"),
        ),
    }
}

/// `GET /v1/model_info`: `{model_path, served_model_name, lora}` for the
/// default model. `model_path` is the engine model id (weights paths are
/// not exposed).
pub fn model_info_body(models: &[ModelCaps], configured: Option<&str>) -> Value {
    let name = default_model(models, configured);
    let root = name.as_deref().and_then(|n| {
        public_names(models)
            .into_iter()
            .find(|(p, _)| p == n)
            .map(|(_, r)| r)
    });
    json!({ "model_path": root.or(name.clone()), "served_model_name": name, "lora": null })
}

fn auth(ctx: &ServeCtx, headers: &HeaderMap) -> Result<(), ApiError> {
    ctx.auth()
        .authenticate(ProtocolId::OpenAiVideos, headers)
        .map(|_| ())
}

async fn reply(ctx: &ServeCtx, r: Result<HttpReply, ApiError>) -> Response {
    into_response(r.unwrap_or_else(|e| openai_error(&e)), ctx, None).await
}

/// `/v1/models`, `/v1/models/{model}`, `/v1/model_info`.
pub fn routes(cfg: Arc<VideosConfig>) -> Router<ServeCtx> {
    let (c1, c2, c3) = (cfg.clone(), cfg.clone(), cfg);
    Router::new()
        .route(
            "/v1/models",
            get(
                move |State(ctx): State<ServeCtx>, headers: HeaderMap| async move {
                    let r = auth(&ctx, &headers).map(|_| {
                        HttpReply::json(200, list_body(&ctx.engine().models(), c1.created))
                    });
                    reply(&ctx, r).await
                },
            ),
        )
        .route(
            "/v1/models/{model}",
            get(
                move |State(ctx): State<ServeCtx>,
                      Path(model): Path<String>,
                      headers: HeaderMap| async move {
                    let r = auth(&ctx, &headers)
                        .map(|_| get_reply(&ctx.engine().models(), &model, c2.created));
                    reply(&ctx, r).await
                },
            ),
        )
        .route(
            "/v1/model_info",
            get(
                move |State(ctx): State<ServeCtx>, headers: HeaderMap| async move {
                    let r = auth(&ctx, &headers).map(|_| {
                        HttpReply::json(
                            200,
                            model_info_body(&ctx.engine().models(), c3.default_model.as_deref()),
                        )
                    });
                    reply(&ctx, r).await
                },
            ),
        )
}
