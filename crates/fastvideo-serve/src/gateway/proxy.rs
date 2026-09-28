//! Streaming through the gateway (docs/serve/gateway.md §5): the gateway
//! authenticates, allocates a worker of the model's pool, proxies the
//! **signalling** HTTP to it (internal token; the caller's `Authorization`
//! is not forwarded) and keeps a lease (`gw_sessions`) so later calls of the
//! same session reach the same worker from any gateway replica. WebRTC
//! media flows client ↔ worker directly (the worker's answer carries its
//! own candidates); WHIP streams go worker → SFU.
//!
//! - fal director: `/wma/ice`, `/{app}/director/ice`, `/wma/session`
//!   (lease on the answer's `session_id`), `/wma/session/heartbeat`,
//!   `/start-session` (SSE), `/info`.
//! - Reactor local runtime: `/start_session` leases a worker to the caller
//!   (API key owner, else `anon`); `/session`, `/stop_session`, `/events`,
//!   `/schema` and `/sessions/{sid}/…` follow that lease.
//! - Native `/fv/v1/streams*`: pods proxied (lease on the stream id);
//!   serverless pools run a `kind:stream` Runpod job.
//!
//! Peer sessions need pod pools (inbound WebRTC); a serverless pool answers
//! 503 for them.

// Handler helpers return a ready `Response` as their error (early return).
#![allow(clippy::result_large_err)]

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use fastvideo_protocol::{ApiError, ProtocolId};
use fastvideo_serve_kit::d1::Stmt;
use fastvideo_serve_kit::ServeCtx;
use futures::StreamExt;
use serde_json::{json, Value};

use super::schema::now_ms;
use super::{Gateway, Lease, Pool};

/// Request headers passed to the worker.
const PASS: &[&str] = &["content-type", "accept", "last-event-id", "user-agent", "x-request-id", "cache-control"];
/// Response headers not copied back (hop-by-hop, recomputed).
const DROP: &[&str] = &["connection", "transfer-encoding", "content-length", "keep-alive", "upgrade"];

impl Gateway {
    /// Forwards one request to `url` and streams the answer back.
    pub(crate) async fn forward(&self, method: Method, url: &str, headers: &HeaderMap, body: Bytes, timeout: Option<Duration>) -> Result<Response, String> {
        let m = reqwest::Method::from_bytes(method.as_str().as_bytes()).map_err(|e| e.to_string())?;
        let mut r = self.worker_req(m, url);
        for (k, v) in headers {
            if PASS.contains(&k.as_str()) {
                r = r.header(k.as_str(), v.as_bytes());
            }
        }
        if let Some(t) = timeout {
            r = r.timeout(t);
        }
        if !body.is_empty() {
            r = r.body(body);
        }
        let resp = r.send().await.map_err(|e| e.without_url().to_string())?;
        let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let mut out = Response::builder().status(status);
        for (k, v) in resp.headers() {
            if DROP.contains(&k.as_str()) {
                continue;
            }
            if let (Ok(n), Ok(v)) = (HeaderName::from_bytes(k.as_str().as_bytes()), HeaderValue::from_bytes(v.as_bytes())) {
                out = out.header(n, v);
            }
        }
        let stream = resp.bytes_stream().map(|c| c.map_err(std::io::Error::other));
        out.body(Body::from_stream(stream)).map_err(|e| e.to_string())
    }

    /// Forwards and reads the whole JSON answer (small signalling replies).
    async fn forward_json(&self, method: Method, url: &str, headers: &HeaderMap, body: Bytes) -> Result<(StatusCode, HeaderMap, Value), String> {
        let resp = self.forward(method, url, headers, body, Some(Duration::from_secs(60))).await?;
        let (parts, body) = resp.into_parts();
        let bytes = axum::body::to_bytes(body, 16 << 20).await.map_err(|e| e.to_string())?;
        Ok((parts.status, parts.headers, serde_json::from_slice(&bytes).unwrap_or(Value::Null)))
    }

    /// A worker of `pool` with no session (one streaming session per GPU).
    pub(crate) fn pick_session_worker(&self, pool: &Pool) -> Option<String> {
        let leased: Vec<String> = {
            let g = self.leases.lock().unwrap_or_else(|p| p.into_inner());
            g.values().filter(|l| l.state == "live" && l.pool == pool.id()).map(|l| l.target.clone()).collect()
        };
        let st = pool.lock();
        st.workers
            .values()
            .filter(|w| w.usable() && w.ready && w.sessions == 0 && !leased.contains(&w.url))
            .min_by_key(|w| w.load())
            .map(|w| w.url.clone())
    }

    /// Any usable worker of `pool` (stateless signalling: ICE servers, info).
    pub(crate) fn any_worker(&self, pool: &Pool) -> Option<String> {
        let st = pool.lock();
        st.workers.values().filter(|w| w.usable()).min_by_key(|w| w.load()).map(|w| w.url.clone())
    }

    pub(crate) async fn lease_put(&self, l: Lease) {
        let now = now_ms();
        let r = self
            .db
            .query(Stmt::new(
                "INSERT INTO gw_sessions (id, pool, kind, target, ref, owner, lease_key, state, created_at, updated_at) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, 'live', ?, ?) ON CONFLICT(id) DO UPDATE SET pool = excluded.pool, \
                 kind = excluded.kind, target = excluded.target, ref = excluded.ref, owner = excluded.owner, \
                 lease_key = excluded.lease_key, state = 'live', updated_at = excluded.updated_at",
                vec![
                    json!(l.id),
                    json!(l.pool),
                    json!(l.kind),
                    json!(l.target),
                    l.r#ref.as_ref().map_or(Value::Null, |v| json!(v)),
                    l.owner.as_ref().map_or(Value::Null, |v| json!(v)),
                    l.lease_key.as_ref().map_or(Value::Null, |v| json!(v)),
                    json!(now),
                    json!(now),
                ],
            ))
            .await;
        if let Err(e) = r {
            tracing::warn!(session = %l.id, error = %e, "gateway: writing a session lease failed");
        }
        self.leases.lock().unwrap_or_else(|p| p.into_inner()).insert(l.id.clone(), l);
    }

    fn lease_from_row(r: &serde_json::Map<String, Value>) -> Lease {
        let s = |k: &str| r.get(k).and_then(Value::as_str).map(str::to_owned);
        Lease {
            id: s("id").unwrap_or_default(),
            pool: s("pool").unwrap_or_default(),
            kind: s("kind").unwrap_or_default(),
            target: s("target").unwrap_or_default(),
            r#ref: s("ref"),
            owner: s("owner"),
            lease_key: s("lease_key"),
            state: s("state").unwrap_or_default(),
            created_at: r.get("created_at").and_then(Value::as_f64).unwrap_or(0.0) as i64,
        }
    }

    /// A live lease by session id (this replica's cache, else D1).
    pub(crate) async fn lease_get(&self, id: &str) -> Option<Lease> {
        if let Some(l) = self.leases.lock().unwrap_or_else(|p| p.into_inner()).get(id).filter(|l| l.state == "live") {
            return Some(l.clone());
        }
        let r = self.db.query(Stmt::new("SELECT * FROM gw_sessions WHERE id = ? AND state = 'live'", vec![json!(id)])).await.ok()?;
        r.rows.first().map(Self::lease_from_row)
    }

    /// The caller's live lease of `kind` (Reactor).
    pub(crate) async fn lease_by_key(&self, key: &str, kind: &str) -> Option<Lease> {
        let r = self
            .db
            .query(Stmt::new(
                "SELECT * FROM gw_sessions WHERE lease_key = ? AND kind = ? AND state = 'live' ORDER BY created_at DESC LIMIT 1",
                vec![json!(key), json!(kind)],
            ))
            .await
            .ok()?;
        r.rows.first().map(Self::lease_from_row)
    }

    pub(crate) async fn lease_end(&self, id: &str) {
        let _ = self
            .db
            .query(Stmt::new("UPDATE gw_sessions SET state = 'ended', updated_at = ? WHERE id = ?", vec![json!(now_ms()), json!(id)]))
            .await;
        if let Some(l) = self.leases.lock().unwrap_or_else(|p| p.into_inner()).get_mut(id) {
            l.state = "ended".into();
        }
    }

    pub(crate) async fn lease_touch(&self, id: &str) {
        let _ = self
            .db
            .query(Stmt::new("UPDATE gw_sessions SET updated_at = ? WHERE id = ? AND state = 'live'", vec![json!(now_ms()), json!(id)]))
            .await;
    }

    /// The first pod pool serving `name`, else the first pool, with the
    /// resolved model id.
    pub(crate) fn stream_pool(&self, name: &str) -> Result<(&Pool, String), ApiError> {
        let (model, pools) = self.pools_for_name(name).ok_or_else(|| ApiError::not_found(format!("model `{name}` is not served by any pool")))?;
        let list: Vec<&Pool> = pools.iter().filter_map(|i| self.pools.get(*i)).collect();
        let p = list
            .iter()
            .find(|p| p.is_pod() && p.available())
            .or_else(|| list.iter().find(|p| p.is_pod()))
            .or_else(|| list.first())
            .copied()
            .ok_or_else(|| ApiError::not_found(format!("model `{name}` is not served by any pool")))?;
        Ok((p, model.0))
    }

    /// The pool of a fal app id (`minimax/h3-max`, with or without
    /// `/director`).
    pub(crate) fn fal_app_pool(&self, app_id: &str) -> Result<&Pool, ApiError> {
        let app = app_id.trim_matches('/').trim_end_matches("/director").trim_end_matches("/realtime");
        let name = self
            .fal_apps
            .iter()
            .find(|(id, _)| id == app)
            .map(|(_, m)| m.clone())
            .ok_or_else(|| ApiError::not_found(format!("unknown app `{app_id}`")))?;
        self.stream_pool(&name).map(|(p, _)| p)
    }
}

/// `{"error": message}` with a status (the director bridge's shape).
fn bridge_error(status: StatusCode, msg: impl Into<String>, retry: Option<u32>) -> Response {
    let mut r = (status, Json(json!({"error": msg.into()}))).into_response();
    if let Some(s) = retry {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

fn api_status(e: &ApiError) -> StatusCode {
    StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn peer_worker_or_503(gw: &Gateway, pool: &Pool, session: bool) -> Result<String, Response> {
    if !pool.is_pod() {
        return Err(bridge_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("pool `{}` is a serverless pool: WebRTC peer sessions need a pod pool", pool.id()),
            None,
        ));
    }
    let w = if session { gw.pick_session_worker(pool) } else { gw.any_worker(pool) };
    w.ok_or_else(|| {
        bridge_error(
            if session { StatusCode::TOO_MANY_REQUESTS } else { StatusCode::SERVICE_UNAVAILABLE },
            format!("pool `{}` has no free worker for a session", pool.id()),
            Some(10),
        )
    })
}

fn proxy_error(e: String) -> Response {
    bridge_error(StatusCode::BAD_GATEWAY, format!("worker unreachable: {e}"), Some(5))
}

#[derive(Clone)]
struct DirState {
    gw: Arc<Gateway>,
}

/// The fal director's routes in gateway mode (merged into the fal router,
/// so `/fal/proxy` reaches them).
pub fn director_routes(gw: Arc<Gateway>) -> Router<ServeCtx> {
    let mut r: Router<ServeCtx> = Router::new()
        .route("/wma/ice", post(wma_ice))
        .route("/wma/session", post(wma_session))
        .route("/wma/session/heartbeat", post(wma_heartbeat))
        .route("/start-session", post(start_session))
        .route("/info", get(info).post(info));
    for (app, _) in &gw.fal_apps {
        r = r
            .route(&format!("/{app}/director/ice"), post(app_ice))
            .route(&format!("/run/{app}/director/ice"), post(app_ice));
    }
    r.layer(axum::Extension(DirState { gw }))
}

type Dir = axum::Extension<DirState>;

fn fal_auth(ctx: &ServeCtx, headers: &HeaderMap) -> Result<Option<String>, Response> {
    ctx.auth()
        .authenticate(ProtocolId::Fal, headers)
        .map(|o| o.map(|k| k.0))
        .map_err(|e| bridge_error(api_status(&e), e.message, None))
}

fn app_of(raw: &Bytes) -> Option<String> {
    serde_json::from_slice::<Value>(raw).ok()?.get("app_id")?.as_str().map(str::to_owned)
}

async fn wma_ice(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, uri: Uri, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(r) = fal_auth(&ctx, &headers) {
        return r;
    }
    let app = app_of(&raw).unwrap_or_default();
    let pool = match d.gw.fal_app_pool(&app) {
        Ok(p) => p,
        Err(e) => return bridge_error(api_status(&e), e.message, None),
    };
    let w = match peer_worker_or_503(&d.gw, pool, false) {
        Ok(w) => w,
        Err(r) => return r,
    };
    d.gw.forward(Method::POST, &format!("{w}{}", uri.path()), &headers, raw, Some(Duration::from_secs(30))).await.unwrap_or_else(proxy_error)
}

async fn app_ice(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, uri: Uri, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(r) = fal_auth(&ctx, &headers) {
        return r;
    }
    let path = uri.path().to_owned();
    let app = path.trim_start_matches("/run/").trim_start_matches('/').trim_end_matches("/director/ice").to_owned();
    let pool = match d.gw.fal_app_pool(&app) {
        Ok(p) => p,
        Err(e) => return bridge_error(api_status(&e), e.message, None),
    };
    let w = match peer_worker_or_503(&d.gw, pool, false) {
        Ok(w) => w,
        Err(r) => return r,
    };
    d.gw.forward(Method::POST, &format!("{w}{path}"), &headers, raw, Some(Duration::from_secs(30))).await.unwrap_or_else(proxy_error)
}

async fn wma_session(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, headers: HeaderMap, raw: Bytes) -> Response {
    let owner = match fal_auth(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let app = app_of(&raw).unwrap_or_default();
    let pool = match d.gw.fal_app_pool(&app) {
        Ok(p) => p,
        Err(e) => return bridge_error(api_status(&e), e.message, None),
    };
    let w = match peer_worker_or_503(&d.gw, pool, true) {
        Ok(w) => w,
        Err(r) => return r,
    };
    let (status, hs, v) = match d.gw.forward_json(Method::POST, &format!("{w}/wma/session"), &headers, raw).await {
        Ok(x) => x,
        Err(e) => return proxy_error(e),
    };
    if status.is_success() {
        if let Some(sid) = v.get("session_id").and_then(Value::as_str) {
            d.gw.lease_put(Lease {
                id: sid.to_owned(),
                pool: pool.id().to_owned(),
                kind: "director".into(),
                target: w.clone(),
                r#ref: None,
                owner,
                lease_key: None,
                state: "live".into(),
                created_at: now_ms(),
            })
            .await;
        }
    }
    let mut r = (status, Json(v)).into_response();
    for (k, val) in hs.iter() {
        if k.as_str().starts_with("x-fv-") || k == "retry-after" {
            r.headers_mut().insert(k.clone(), val.clone());
        }
    }
    r
}

async fn wma_heartbeat(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(r) = fal_auth(&ctx, &headers) {
        return r;
    }
    let sid = serde_json::from_slice::<Value>(&raw).ok().and_then(|v| v.get("session_id").and_then(Value::as_str).map(str::to_owned)).unwrap_or_default();
    let Some(l) = d.gw.lease_get(&sid).await else {
        return Json(json!({"alive": false})).into_response();
    };
    let (status, _, v) = match d.gw.forward_json(Method::POST, &format!("{}/wma/session/heartbeat", l.target), &headers, raw).await {
        Ok(x) => x,
        Err(e) => return proxy_error(e),
    };
    if v.get("alive") == Some(&Value::Bool(false)) {
        d.gw.lease_end(&sid).await;
    } else {
        d.gw.lease_touch(&sid).await;
    }
    (status, Json(v)).into_response()
}

async fn start_session(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(r) = fal_auth(&ctx, &headers) {
        return r;
    }
    let Some((app, _)) = d.gw.fal_apps.first().cloned() else {
        return bridge_error(StatusCode::NOT_FOUND, "no fal app is configured", None);
    };
    let pool = match d.gw.fal_app_pool(&app) {
        Ok(p) => p,
        Err(e) => return bridge_error(api_status(&e), e.message, None),
    };
    let w = match peer_worker_or_503(&d.gw, pool, true) {
        Ok(w) => w,
        Err(r) => return r,
    };
    // SSE for the session's life: no timeout.
    d.gw.forward(Method::POST, &format!("{w}/start-session"), &headers, raw, None).await.unwrap_or_else(proxy_error)
}

async fn info(State(ctx): State<ServeCtx>, axum::Extension(d): Dir, method: Method, headers: HeaderMap, raw: Bytes) -> Response {
    if let Err(r) = fal_auth(&ctx, &headers) {
        return r;
    }
    let Some((app, _)) = d.gw.fal_apps.first().cloned() else {
        return bridge_error(StatusCode::NOT_FOUND, "no fal app is configured", None);
    };
    let pool = match d.gw.fal_app_pool(&app) {
        Ok(p) => p,
        Err(e) => return bridge_error(api_status(&e), e.message, None),
    };
    let w = match peer_worker_or_503(&d.gw, pool, false) {
        Ok(w) => w,
        Err(r) => return r,
    };
    d.gw.forward(method, &format!("{w}/info"), &headers, raw, Some(Duration::from_secs(30))).await.unwrap_or_else(proxy_error)
}

// ------------------------------------------------------------------ Reactor

#[derive(Clone)]
struct RtState {
    gw: Arc<Gateway>,
    ctx: ServeCtx,
}

/// The Reactor local-runtime routes in gateway mode (CORS `*` as the
/// runtime's own router).
pub fn reactor_routes(gw: Arc<Gateway>, ctx: ServeCtx) -> Router {
    Router::new()
        .route("/start_session", post(rt_start))
        .route("/session", get(rt_follow))
        .route("/stop_session", post(rt_stop))
        .route("/schema", get(rt_follow))
        .route("/events", get(rt_follow))
        .route("/sessions/{*rest}", any(rt_follow))
        .with_state(RtState { gw, ctx })
        .layer(tower_http::cors::CorsLayer::very_permissive())
}

fn rt_detail(status: StatusCode, msg: impl Into<String>, retry: Option<u32>) -> Response {
    let mut r = (status, Json(json!({"detail": msg.into()}))).into_response();
    if let Some(s) = retry {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

impl RtState {
    fn key(&self, headers: &HeaderMap) -> String {
        let owner = self.ctx.auth().authenticate(ProtocolId::Reactor, headers).ok().flatten();
        format!("reactor:{}", owner.map(|k| k.0).unwrap_or_else(|| "anon".into()))
    }

    fn pool(&self) -> Result<&Pool, Response> {
        let name = match &self.gw.cfg.reactor_model {
            Some(m) => m.clone(),
            None => {
                // The first stream-capable model of a pod pool.
                let cat = self.gw.catalog();
                let found = cat.table.models().find(|m| {
                    m.stream.is_some() && cat.pools_of.get(&m.id).is_some_and(|ps| ps.iter().any(|i| self.gw.pools.get(*i).is_some_and(Pool::is_pod)))
                });
                match found {
                    Some(m) => m.id.0.clone(),
                    None => return Err(rt_detail(StatusCode::SERVICE_UNAVAILABLE, "no pod pool serves a streaming model", None)),
                }
            }
        };
        self.gw.stream_pool(&name).map(|(p, _)| p).map_err(|e| rt_detail(api_status(&e), e.message, None))
    }
}

async fn rt_start(State(s): State<RtState>, headers: HeaderMap, raw: Bytes) -> Response {
    let key = s.key(&headers);
    if let Some(l) = s.gw.lease_by_key(&key, "reactor").await {
        // One session per caller: the worker answers (busy or idempotent).
        return s.gw.forward(Method::POST, &format!("{}/start_session", l.target), &headers, raw, Some(Duration::from_secs(60))).await.unwrap_or_else(proxy_error);
    }
    let pool = match s.pool() {
        Ok(p) => p,
        Err(r) => return r,
    };
    if !pool.is_pod() {
        return rt_detail(StatusCode::SERVICE_UNAVAILABLE, format!("pool `{}` is serverless: Reactor sessions need a pod pool", pool.id()), None);
    }
    let Some(w) = s.gw.pick_session_worker(pool) else {
        return rt_detail(StatusCode::SERVICE_UNAVAILABLE, format!("pool `{}` has no free worker for a session", pool.id()), Some(10));
    };
    let (status, _, v) = match s.gw.forward_json(Method::POST, &format!("{w}/start_session"), &headers, raw).await {
        Ok(x) => x,
        Err(e) => return proxy_error(e),
    };
    if status.is_success() {
        let sid = v.get("session_id").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let owner = key.strip_prefix("reactor:").filter(|o| *o != "anon").map(str::to_owned);
        // The runtime's session id is fixed per process: the lease id is
        // the caller's key (one Reactor session per caller).
        s.gw.lease_put(Lease {
            id: format!("{key}:{sid}"),
            pool: pool.id().to_owned(),
            kind: "reactor".into(),
            target: w,
            r#ref: Some(sid),
            owner,
            lease_key: Some(key),
            state: "live".into(),
            created_at: now_ms(),
        })
        .await;
    }
    (status, Json(v)).into_response()
}

async fn rt_follow(State(s): State<RtState>, method: Method, uri: Uri, headers: HeaderMap, raw: Bytes) -> Response {
    let key = s.key(&headers);
    let target = match s.gw.lease_by_key(&key, "reactor").await {
        Some(l) => {
            s.gw.lease_touch(&l.id).await;
            l.target
        }
        None => {
            // No session: a ready worker answers the descriptor/schema/journal.
            let pool = match s.pool() {
                Ok(p) => p,
                Err(r) => return r,
            };
            match s.gw.any_worker(pool) {
                Some(w) if pool.is_pod() => w,
                _ => return rt_detail(StatusCode::SERVICE_UNAVAILABLE, format!("pool `{}` has no worker", pool.id()), Some(10)),
            }
        }
    };
    let pq = uri.path_and_query().map(|p| p.as_str().to_owned()).unwrap_or_else(|| uri.path().to_owned());
    // `/events` is a long-lived SSE journal: no timeout.
    let timeout = (uri.path() != "/events").then_some(Duration::from_secs(60));
    s.gw.forward(method, &format!("{target}{pq}"), &headers, raw, timeout).await.unwrap_or_else(proxy_error)
}

async fn rt_stop(State(s): State<RtState>, headers: HeaderMap, raw: Bytes) -> Response {
    let key = s.key(&headers);
    let Some(l) = s.gw.lease_by_key(&key, "reactor").await else {
        return rt_detail(StatusCode::NOT_FOUND, "no active session", None);
    };
    let r = s.gw.forward(Method::POST, &format!("{}/stop_session", l.target), &headers, raw, Some(Duration::from_secs(60))).await;
    if r.as_ref().is_ok_and(|r| r.status().is_success()) {
        s.gw.lease_end(&l.id).await;
    }
    r.unwrap_or_else(proxy_error)
}

// ------------------------------------------------------- native /fv/v1/streams

#[derive(Clone)]
struct StreamState {
    gw: Arc<Gateway>,
}

type Ss = axum::Extension<StreamState>;

/// `/fv/v1/streams*` in gateway mode.
pub fn stream_routes(gw: Arc<Gateway>) -> Router<ServeCtx> {
    Router::new()
        .route("/fv/v1/streams", post(st_create).get(st_list))
        .route("/fv/v1/streams/{id}", get(st_get).delete(st_delete))
        .route("/fv/v1/streams/{id}/commands", post(st_command))
        .layer(axum::Extension(StreamState { gw }))
}

fn native_error(e: &ApiError) -> Response {
    let kind = serde_json::to_value(e.kind).unwrap_or(Value::Null);
    let mut r = (api_status(e), Json(json!({"error": {"kind": kind, "message": e.message, "param": e.param}}))).into_response();
    if let Some(s) = e.retry_after_s {
        r.headers_mut().insert("retry-after", HeaderValue::from(s));
    }
    r
}

fn native_owner(ctx: &ServeCtx, headers: &HeaderMap) -> Result<Option<String>, Response> {
    ctx.auth().authenticate(ProtocolId::Native, headers).map(|o| o.map(|k| k.0)).map_err(|e| native_error(&e))
}

async fn st_lease(gw: &Gateway, id: &str, owner: &Option<String>) -> Result<Lease, Response> {
    match gw.lease_get(id).await {
        Some(l) if l.kind == "stream" && (l.owner.is_none() || l.owner == *owner) => Ok(l),
        _ => Err(native_error(&ApiError::not_found(format!("stream `{id}` was not found")))),
    }
}

async fn st_create(State(ctx): State<ServeCtx>, axum::Extension(s): Ss, headers: HeaderMap, raw: Bytes) -> Response {
    let owner = match native_owner(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    if !s.gw.admitting() {
        return native_error(&ApiError::loading("the gateway is shutting down"));
    }
    let body: Value = match serde_json::from_slice(&raw) {
        Ok(v @ Value::Object(_)) => v,
        _ => return native_error(&ApiError::invalid("the body must be a JSON object")),
    };
    let model = body.get("model").and_then(Value::as_str).unwrap_or_default().to_owned();
    let (pool, _) = match s.gw.stream_pool(&model) {
        Ok(x) => x,
        Err(e) => return native_error(&e),
    };
    if pool.cfg.max_streams > 0 && pool.lock().streams >= pool.cfg.max_streams {
        return native_error(&ApiError::queue_full(format!("pool `{}` is at its stream limit", pool.id())).with_retry_after(10));
    }
    if pool.is_pod() {
        let Some(w) = s.gw.pick_session_worker(pool) else {
            return native_error(&ApiError::queue_full(format!("pool `{}` has no free worker for a stream", pool.id())).with_retry_after(10));
        };
        let (status, hs, v) = match s.gw.forward_json(Method::POST, &format!("{w}/fv/v1/streams"), &headers, raw).await {
            Ok(x) => x,
            Err(e) => return proxy_error(e),
        };
        if status.is_success() {
            if let Some(id) = v.get("id").and_then(Value::as_str) {
                s.gw.lease_put(Lease {
                    id: id.to_owned(),
                    pool: pool.id().to_owned(),
                    kind: "stream".into(),
                    target: w,
                    r#ref: None,
                    owner,
                    lease_key: None,
                    state: "live".into(),
                    created_at: now_ms(),
                })
                .await;
            }
        }
        let mut r = (status, Json(v)).into_response();
        if let Some(ra) = hs.get("retry-after") {
            r.headers_mut().insert("retry-after", ra.clone());
        }
        return r;
    }
    // Serverless: one `kind: stream` Runpod job (the worker publishes over WHIP).
    if body.get("whip_url").and_then(Value::as_str).is_none_or(str::is_empty) {
        return native_error(&ApiError::invalid_param("whip_url", "a stream on a serverless pool needs `whip_url`"));
    }
    let mut input = body.clone();
    input["kind"] = json!("stream");
    let ep = pool.cfg.endpoint_id.clone().unwrap_or_default();
    let rid = match s.gw.runpod.run(&ep, &input, Duration::from_secs(pool.cfg.dispatch_timeout_s.max(1))).await {
        Ok(id) => id,
        Err(e) => return native_error(&ApiError::loading(format!("pool `{}` did not take the stream: {e}", pool.id())).with_retry_after(10)),
    };
    let id = format!("gws_{}", uuid::Uuid::new_v4().simple());
    s.gw.lease_put(Lease {
        id: id.clone(),
        pool: pool.id().to_owned(),
        kind: "stream".into(),
        target: ep,
        r#ref: Some(rid),
        owner,
        lease_key: None,
        state: "live".into(),
        created_at: now_ms(),
    })
    .await;
    (StatusCode::CREATED, Json(json!({"id": id, "object": "fv.stream", "state": "starting", "model": model, "pool": pool.id()}))).into_response()
}

async fn st_list(State(ctx): State<ServeCtx>, axum::Extension(s): Ss, headers: HeaderMap) -> Response {
    let owner = match native_owner(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let rows = s
        .gw
        .db
        .query(Stmt::new(
            "SELECT * FROM gw_sessions WHERE kind = 'stream' AND (owner IS ? OR ? IS NULL) ORDER BY created_at DESC LIMIT 100",
            vec![owner.as_ref().map_or(Value::Null, |o| json!(o)), owner.as_ref().map_or(Value::Null, |o| json!(o))],
        ))
        .await
        .map(|r| r.rows)
        .unwrap_or_default();
    let data: Vec<Value> = rows
        .iter()
        .map(Gateway::lease_from_row)
        .map(|l| json!({"id": l.id, "object": "fv.stream", "state": l.state, "pool": l.pool, "created_at": l.created_at}))
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

/// A serverless stream's view from its Runpod job.
fn runpod_stream_view(l: &Lease, st: Option<super::runpod::RunStatus>) -> Value {
    let Some(st) = st else {
        return json!({"id": l.id, "object": "fv.stream", "state": "ended", "pool": l.pool});
    };
    let state = match st.status.as_str() {
        "IN_QUEUE" => "starting",
        "IN_PROGRESS" => {
            if st.output.as_ref().and_then(|o| o.get("state")).and_then(Value::as_str) == Some("live") {
                "live"
            } else {
                "starting"
            }
        }
        "COMPLETED" => "ended",
        "CANCELLED" => "stopped",
        _ => "failed",
    };
    let mut v = json!({"id": l.id, "object": "fv.stream", "state": state, "pool": l.pool, "runpod_status": st.status});
    if let Some(o) = st.output {
        v["session"] = o;
    }
    if let Some(e) = st.error {
        v["error"] = e;
    }
    v
}

async fn st_get(State(ctx): State<ServeCtx>, axum::Extension(s): Ss, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let owner = match native_owner(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let l = match st_lease(&s.gw, &id, &owner).await {
        Ok(l) => l,
        Err(r) => return r,
    };
    if let Some(rid) = &l.r#ref {
        let st = s.gw.runpod.status(&l.target, rid).await;
        return match st {
            Ok(st) => {
                let v = runpod_stream_view(&l, st);
                if matches!(v["state"].as_str(), Some("ended" | "stopped" | "failed")) {
                    s.gw.lease_end(&l.id).await;
                }
                Json(v).into_response()
            }
            Err(e) => native_error(&ApiError::loading(e).with_retry_after(5)),
        };
    }
    let r = s.gw.forward(Method::GET, &format!("{}/fv/v1/streams/{id}", l.target), &headers, Bytes::new(), Some(Duration::from_secs(30))).await;
    r.unwrap_or_else(proxy_error)
}

async fn st_delete(State(ctx): State<ServeCtx>, axum::Extension(s): Ss, Path(id): Path<String>, headers: HeaderMap) -> Response {
    let owner = match native_owner(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let l = match st_lease(&s.gw, &id, &owner).await {
        Ok(l) => l,
        Err(r) => return r,
    };
    if let Some(rid) = &l.r#ref {
        if let Err(e) = s.gw.runpod.cancel(&l.target, rid).await {
            return native_error(&ApiError::loading(e).with_retry_after(5));
        }
        s.gw.lease_end(&l.id).await;
        return Json(json!({"id": id, "object": "fv.stream", "state": "stopping"})).into_response();
    }
    let r = s.gw.forward(Method::DELETE, &format!("{}/fv/v1/streams/{id}", l.target), &headers, Bytes::new(), Some(Duration::from_secs(60))).await;
    if r.as_ref().is_ok_and(|r| r.status().is_success()) {
        s.gw.lease_end(&l.id).await;
    }
    r.unwrap_or_else(proxy_error)
}

async fn st_command(State(ctx): State<ServeCtx>, axum::Extension(s): Ss, Path(id): Path<String>, headers: HeaderMap, raw: Bytes) -> Response {
    let owner = match native_owner(&ctx, &headers) {
        Ok(o) => o,
        Err(r) => return r,
    };
    let l = match st_lease(&s.gw, &id, &owner).await {
        Ok(l) => l,
        Err(r) => return r,
    };
    if l.r#ref.is_some() {
        return native_error(&ApiError::invalid("commands are not supported for streams on a serverless pool"));
    }
    s.gw.lease_touch(&l.id).await;
    let r = s.gw.forward(Method::POST, &format!("{}/fv/v1/streams/{id}/commands", l.target), &headers, raw, Some(Duration::from_secs(60))).await;
    r.unwrap_or_else(proxy_error)
}
