//! `POST /storage/upload/initiate` (design §4.4; fal §11).
//!
//! `@fal-ai/client`'s `transformInput` uploads every `Blob`/`File` in an
//! input before submitting: it POSTs `{content_type, file_name}` to
//! `rest.fal.ai/storage/upload/initiate?storage_type=…` (through the proxy or
//! `requestMiddleware`), `PUT`s the bytes to the returned `upload_url` with
//! plain fetch and no auth, and puts `file_url` into the input. Here
//! `upload_url` is serve-kit's `PUT /uploads/{token}` and `file_url` a signed
//! `/files/{token}/{name}` URL on this server. When an input then references
//! such a `file_url`, [`rewrite_own_uploads`] turns it into the upload id so
//! ingestion reads the file locally instead of fetching our own URL.
//!
//! The multipart flow for files over 90 MB (`…/initiate-multipart`) is not
//! served; the single PUT takes up to the upload store's `max_bytes`.
//! Python's uploads are hard-coded to fal's hosts and cannot be redirected
//! (fal §12.2): Python callers pass URLs or data URIs.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use fastvideo_protocol::{ApiError, ErrorCtx, GenerationRequest, HttpReply, MediaRef, ProtocolId, UploadId};
use fastvideo_serve_kit::handlers::error_reply;
use fastvideo_serve_kit::{into_response, ServeCtx};
use serde::Deserialize;
use serde_json::json;

use crate::error::FalProtocol;
use crate::FalConfig;

/// The initiate body.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct InitiateBody {
    #[serde(default)]
    pub content_type: Option<String>,
    #[serde(default)]
    pub file_name: Option<String>,
}

async fn initiate(ctx: &ServeCtx, ttl: Duration, headers: &HeaderMap, body: &Bytes) -> Result<HttpReply, ApiError> {
    ctx.auth().authenticate(ProtocolId::Fal, headers)?;
    let b: InitiateBody = if body.is_empty() {
        InitiateBody::default()
    } else {
        serde_json::from_slice(body).map_err(|e| ApiError::invalid(format!("JSON decode error: {e}")))?
    };
    let t = ctx
        .uploads()
        .create(b.file_name.as_deref(), b.content_type.as_deref(), ctx.now())
        .map_err(|e| ApiError::internal(format!("creating the upload: {e}")))?;
    let file_url = ctx
        .uploads()
        .file_url(&t.token, ttl)
        .ok_or_else(|| ApiError::internal("the upload vanished"))?;
    Ok(HttpReply::json(200, json!({"upload_url": t.upload_url.as_str(), "file_url": file_url.as_str()})))
}

async fn initiate_handler(ctx: ServeCtx, cfg: Arc<FalConfig>, headers: HeaderMap, body: Bytes) -> Response {
    let reply = initiate(&ctx, cfg.url_ttl, &headers, &body).await.unwrap_or_else(|e| {
        let ecx = ErrorCtx { request_id: None, route: Some("/storage/upload/initiate".into()), external_id: None };
        error_reply(&FalProtocol, &e, &ecx)
    });
    into_response(reply, &ctx, None).await
}

pub(crate) fn routes(router: Router<ServeCtx>, cfg: &Arc<FalConfig>) -> Router<ServeCtx> {
    let c = cfg.clone();
    router.route(
        "/storage/upload/initiate",
        post(move |State(ctx): State<ServeCtx>, headers: HeaderMap, body: Bytes| initiate_handler(ctx, c.clone(), headers, body)),
    )
}

/// The upload token of `url` if it is one of our `/files/{token}/{name}`
/// URLs (under `public_base`).
pub fn own_upload_token(public_base: &url::Url, url: &url::Url) -> Option<String> {
    let base = public_base.as_str().trim_end_matches('/');
    let rest = url.as_str().strip_prefix(base)?.strip_prefix("/files/")?;
    let rest = rest.split(['?', '#']).next()?;
    let (token, name) = rest.split_once('/')?;
    (!token.is_empty() && !name.is_empty() && !name.contains('/')).then(|| token.to_owned())
}

/// Replaces HTTP refs that point at our own completed uploads with the
/// upload id (see the module docs).
pub fn rewrite_own_uploads(ctx: &ServeCtx, req: &mut GenerationRequest) {
    let base = ctx.config().public_base.clone();
    let now = ctx.now();
    let fix = |m: &mut MediaRef| {
        if let MediaRef::Http(u) = m {
            if let Some(t) = own_upload_token(&base, u) {
                let id = UploadId(t);
                // Ours, or another front's behind the same edge (fetched
                // through the edge, not over the public URL).
                if ctx.uploads().resolve(&id, now).is_some() || ctx.uploads().is_remote(&id.0) {
                    *m = MediaRef::Upload(id);
                }
            }
        }
    };
    req.keyframes.iter_mut().for_each(|k| fix(&mut k.image));
    req.references.iter_mut().for_each(|r| fix(&mut r.media));
    if let Some(a) = req.audio_in.as_mut() {
        fix(&mut a.media);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_tokens() {
        let base = url::Url::parse("https://fv.test/api").unwrap();
        let u = |s: &str| url::Url::parse(s).unwrap();
        assert_eq!(own_upload_token(&base, &u("https://fv.test/api/files/abc/x.png?exp=1&sig=2")).as_deref(), Some("abc"));
        assert_eq!(own_upload_token(&base, &u("https://other.test/api/files/abc/x.png")), None);
        assert_eq!(own_upload_token(&base, &u("https://fv.test/api/files/abc")), None);
        assert_eq!(own_upload_token(&base, &u("https://fv.test/api/uploads/abc/x")), None);
    }
}
