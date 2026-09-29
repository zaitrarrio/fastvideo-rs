//! The Reactor HTTP surface (reactor §3.2-3.4, design §5.7, §9): local
//! session routes, `/schema`, `/events`, and the signalling group. No auth;
//! CORS `*` with every method and header (RT `http/server.py:build_app`).

use std::convert::Infallible;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::session::{Reactor, Refusal};
use crate::signalling::{AnswerPoll, Candidates, SdpParams};

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let mut r = (
            StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(json!({"detail": self.detail})),
        )
            .into_response();
        if let Some(s) = self.retry_after {
            r.headers_mut().insert("retry-after", HeaderValue::from(s));
        }
        r
    }
}

/// Every Reactor route, with CORS, over `rt`.
pub fn router(rt: Reactor) -> Router {
    const W: &str = "/sessions/{sid}/transport/webrtc";
    Router::new()
        .route("/start_session", post(start_session))
        .route("/session", get(session))
        .route("/stop_session", post(stop_session))
        .route("/schema", get(schema))
        .route("/events", get(events))
        .route(&format!("{W}/ice_servers"), get(ice_servers))
        .route(&format!("{W}/connections"), post(connections))
        .route(
            &format!("{W}/connections/{{cid}}/sdp_params"),
            post(offer).put(offer).get(poll_answer),
        )
        .route(&format!("{W}/connections/{{cid}}/ice_candidates"), post(candidates))
        .route("/sessions/{sid}/uploads", post(create_upload))
        .route("/sessions/{sid}/uploads/{id}", axum::routing::put(put_upload).layer(axum::extract::DefaultBodyLimit::max(crate::uploads::MAX_UPLOAD_BYTES as usize)))
        .layer(CorsLayer::permissive())
        .with_state(rt)
}

/// A JSON object body that may be empty (RT: an optional body).
fn object_body(b: &Bytes) -> Result<Value, Refusal> {
    if b.iter().all(u8::is_ascii_whitespace) {
        return Ok(json!({}));
    }
    match serde_json::from_slice::<Value>(b) {
        Ok(v @ Value::Object(_)) => Ok(v),
        Ok(Value::Null) => Ok(json!({})),
        Ok(_) => Err(Refusal::new(422, "the body must be a JSON object")),
        Err(e) => Err(Refusal::new(422, format!("invalid JSON: {e}"))),
    }
}

async fn start_session(State(rt): State<Reactor>, body: Bytes) -> Response {
    let params = match object_body(&body) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    match rt.start_session(params).await {
        Ok(d) => Json(d).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn session(State(rt): State<Reactor>) -> Response {
    Json(rt.descriptor()).into_response()
}

async fn stop_session(State(rt): State<Reactor>, body: Bytes) -> Response {
    let p = match object_body(&body) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let moderate = p.get("moderate").and_then(Value::as_bool).unwrap_or(false);
    let reason = p.get("reason").and_then(Value::as_str).unwrap_or_default().to_owned();
    if reason.chars().count() > 64 {
        return Refusal::new(422, "reason: at most 64 characters").into_response();
    }
    match rt.stop_session(moderate, reason).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn schema(State(rt): State<Reactor>) -> Response {
    Json(rt.schema()).into_response()
}

#[derive(Deserialize)]
struct Since {
    since: Option<u64>,
}

async fn events(State(rt): State<Reactor>, headers: HeaderMap, Query(q): Query<Since>) -> Response {
    let since = q
        .since
        .or_else(|| headers.get("last-event-id").and_then(|v| v.to_str().ok()?.parse().ok()))
        .unwrap_or(0);
    Sse::new(journal_stream(&rt, since))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

fn journal_stream(rt: &Reactor, since: u64) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let (replay, rx) = rt.journal().subscribe(since);
    let last = replay.last().map_or(since, |e| e.seq);
    let to_sse = |e: crate::journal::Event| SseEvent::default().id(e.seq.to_string()).data(e.data.to_string());
    let live = stream::unfold((rx, last), |(mut rx, last)| async move {
        loop {
            match rx.recv().await {
                Ok(e) if e.seq <= last => continue,
                Ok(e) => {
                    let seq = e.seq;
                    return Some((e, (rx, seq)));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return None,
            }
        }
    });
    stream::iter(replay).chain(live).map(move |e| Ok(to_sse(e)))
}

async fn ice_servers(State(rt): State<Reactor>, Path(sid): Path<String>) -> Response {
    match rt.ice_servers(&sid) {
        Ok(v) => Json(v).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn connections(State(rt): State<Reactor>, Path(sid): Path<String>) -> Response {
    match rt.register_connection(&sid) {
        Ok(v) => (StatusCode::CREATED, Json(v)).into_response(),
        Err(e) => e.into_response(),
    }
}

fn parse_cid(cid: &str) -> Result<u32, Refusal> {
    cid.parse().map_err(|_| Refusal::new(422, "connection_id must be an integer"))
}

async fn offer(
    State(rt): State<Reactor>,
    Path((sid, cid)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Session checks come before body validation (RT order).
    if let Err(e) = rt.require_running(&sid) {
        return e.into_response();
    }
    let cid = match parse_cid(&cid) {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let params: SdpParams = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => return Refusal::new(422, format!("invalid SdpParamsRequest: {e}")).into_response(),
    };
    let header = headers.get("reactor-webrtc-version").and_then(|v| v.to_str().ok());
    match rt.offer(&sid, cid, params, header) {
        Ok(v) => (StatusCode::ACCEPTED, Json(v)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn poll_answer(State(rt): State<Reactor>, Path((sid, cid)): Path<(String, String)>) -> Response {
    let cid = match rt.require_running(&sid).and_then(|_| parse_cid(&cid)) {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    match rt.poll_answer(&sid, cid) {
        Ok(AnswerPoll::Pending) => StatusCode::ACCEPTED.into_response(),
        Ok(AnswerPoll::Ready(a)) => Json(json!({"sdp_answer": a, "connection_id": cid})).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn candidates(
    State(rt): State<Reactor>,
    Path((sid, cid)): Path<(String, String)>,
    body: Bytes,
) -> Response {
    let cid = match rt.require_running(&sid).and_then(|_| parse_cid(&cid)) {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let c: Candidates = match serde_json::from_slice(&body) {
        Ok(c) => c,
        Err(e) => return Refusal::new(422, format!("invalid candidates: {e}")).into_response(),
    };
    match rt.add_candidates(&sid, cid, c).await {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(e) => e.into_response(),
    }
}

#[derive(Deserialize)]
struct CreateUpload {
    #[serde(default)]
    name: String,
    size: u64,
    #[serde(default)]
    mime_type: String,
    #[serde(default)]
    upload_id: Option<String>,
}

/// The origin a client reached us at (`x-forwarded-proto` + `Host`), for
/// the presigned URL; empty (a relative URL) without a `Host`.
fn origin(headers: &HeaderMap) -> String {
    let host = headers.get("x-forwarded-host").or_else(|| headers.get("host")).and_then(|v| v.to_str().ok());
    let proto = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|p| p.split(',').next())
        .map(str::trim)
        .unwrap_or("http");
    match host {
        Some(h) if !h.is_empty() && !h.contains(['/', ' ']) => format!("{proto}://{h}"),
        _ => String::new(),
    }
}

/// `POST /sessions/{sid}/uploads` (reactor §3.5): 201 with the presigned URL.
async fn create_upload(State(rt): State<Reactor>, Path(sid): Path<String>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(e) = rt.require_running(&sid) {
        return e.into_response();
    }
    let req: CreateUpload = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return Refusal::new(422, format!("invalid upload request: {e}")).into_response(),
    };
    match rt.uploads().create(&req.name, req.size, &req.mime_type, req.upload_id.as_deref()) {
        Ok(id) => {
            let path = format!("/sessions/{sid}/uploads/{id}");
            (
                StatusCode::CREATED,
                Json(json!({"presigned_id": id, "upload_id": id, "presigned_url": format!("{}{path}", origin(&headers)), "path": path})),
            )
                .into_response()
        }
        Err(e) => e.into_response(),
    }
}

/// `PUT /sessions/{sid}/uploads/{id}`: the raw bytes (exactly the declared size).
async fn put_upload(State(rt): State<Reactor>, Path((sid, id)): Path<(String, String)>, body: Bytes) -> Response {
    if let Err(e) = rt.require_running(&sid) {
        return e.into_response();
    }
    match rt.uploads().put(&id, &body) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => e.into_response(),
    }
}
