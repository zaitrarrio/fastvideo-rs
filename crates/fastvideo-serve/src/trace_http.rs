//! Request tracing on the HTTP side (docs/serve/tracing.md).
//!
//! - [`layer`]: for a request that opts in (`x-fv-trace: 1` or
//!   `?fv_trace=1`, under `FV_TRACE`'s mode), runs the handler with the
//!   trace as the task's [`fastvideo_trace::current`], records one `http`
//!   span (receive → response headers) and answers `traceparent` (this
//!   hop's span), `x-fv-trace-t: <recv wall ns>;<send wall ns>` (the
//!   clock-alignment sample) and `server-timing`. Untraced requests pass
//!   straight through: one header lookup.
//! - [`routes`]: `GET /fv/v1/traces/{id}` (everything this process recorded
//!   under the trace, with the recorder's drop counters) and
//!   `POST /fv/v1/traces/{id}/events` (events from other hosts: the edge's
//!   `waitUntil` shipment, the browser's beacon).
//! - [`install`]: names this host in its events and ships each event as a
//!   log line (target `fv_trace`) through the log shipper (`FV_TRACE_LOG=0`
//!   turns that off).

use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, Request};
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use fastvideo_trace::{decide, mode, new_span_id, now_ns, wall_ns, Comp, Event, Mode, OPT_IN_HEADER, TIME_HEADER, TRACEPARENT};

/// Events one POST may carry.
const MAX_INGEST_EVENTS: usize = 2000;
/// Its body cap.
const MAX_INGEST_BYTES: usize = 512 * 1024;

/// Names this host (`pod:<id>`) and wires the log shipper.
pub fn install(worker_id: &str) {
    let pod = std::env::var("RUNPOD_POD_ID").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| worker_id.to_owned());
    fastvideo_trace::set_host(format!("pod:{pod}"));
    if std::env::var("FV_TRACE_LOG").ok().is_none_or(|v| v != "0") {
        fastvideo_trace::set_sink(|e: &Event| {
            tracing::info!(
                target: "fv_trace",
                trace_id = %e.trace,
                comp = %e.comp,
                stage = %e.name,
                clock = %e.clock,
                t_wall_ns = e.t_wall_ns,
                dur_ms = e.dur_ns as f64 / 1e6,
                arg = e.arg.unwrap_or(0),
                host = %e.host,
                "trace {} {}.{}",
                e.trace,
                e.comp,
                e.name
            );
        });
    }
}

/// The span name of a traced request (static: no formatting on the hot path).
fn span_name(method: &Method, path: &str) -> &'static str {
    let get = method == Method::GET || method == Method::HEAD;
    if get && path.starts_with("/files/") {
        "get_file"
    } else if get && path.ends_with("/content") {
        "get_content"
    } else if path.starts_with("/fv/v1/traces") {
        "trace_api"
    } else if get {
        "get"
    } else if method == Method::POST {
        "post"
    } else if method == Method::DELETE {
        "delete"
    } else {
        "other"
    }
}

fn query_opt_in(q: Option<&str>) -> bool {
    q.is_some_and(|q| q.split('&').any(|kv| matches!(kv, "fv_trace=1" | "fv_trace=true")))
}

/// The tracing middleware (see the module docs).
pub async fn layer(req: Request, next: Next) -> Response {
    let m = mode();
    if m == Mode::Off {
        return next.run(req).await;
    }
    let h = req.headers();
    let opt = if query_opt_in(req.uri().query()) { Some("1") } else { h.get(OPT_IN_HEADER).and_then(|v| v.to_str().ok()) };
    if m == Mode::OptIn && opt.is_none() {
        return next.run(req).await;
    }
    let Some(trace) = decide(m, h.get(TRACEPARENT).and_then(|v| v.to_str().ok()), opt) else {
        return next.run(req).await;
    };
    let t_recv = now_ns();
    let name = span_name(req.method(), req.uri().path());
    let span = new_span_id();
    let mut resp = fastvideo_trace::scope(trace.child(span), next.run(req)).await;
    let status = resp.status().as_u16();
    trace.span_since(Comp::Http, name, t_recv, i64::from(status));
    let t_send = now_ns();
    let hs = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&trace.traceparent(span)) {
        hs.insert(TRACEPARENT, v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("{};{}", wall_ns(t_recv), wall_ns(t_send))) {
        hs.insert(TIME_HEADER, v);
    }
    let dur_ms = t_send.saturating_sub(t_recv) as f64 / 1e6;
    if let Ok(v) = HeaderValue::from_str(&format!("fv-pod;dur={dur_ms:.3}")) {
        hs.append("server-timing", v);
    }
    resp
}

/// `GET /fv/v1/traces/{id}`, `POST /fv/v1/traces/{id}/events`.
pub fn routes() -> Router {
    Router::new()
        .route("/fv/v1/traces/{id}", get(get_trace))
        .route("/fv/v1/traces/{id}/events", post(post_events))
}

fn valid_id(id: &str) -> Option<String> {
    fastvideo_trace::TraceId::parse_hex(id).map(|t| t.hex())
}

async fn get_trace(Path(id): Path<String>) -> Response {
    if mode() == Mode::Off {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": {"message": "tracing is off (FV_TRACE=off)"}}))).into_response();
    }
    let Some(id) = valid_id(&id) else {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": {"message": "a trace id is 32 hex digits"}}))).into_response();
    };
    // The snapshot waits (briefly) for the drain: off the async runtime.
    let dump = tokio::task::spawn_blocking(move || fastvideo_trace::snapshot(&id, Duration::from_secs(2))).await.ok().flatten();
    match dump {
        Some(d) => Json(d).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": {"message": "no events for this trace here"}}))).into_response(),
    }
}

/// Body: `{"events": [...]}` or `[...]`, any content type (a beacon sends
/// `text/plain`). Each event's `trace` is set to the path's id.
async fn post_events(Path(id): Path<String>, body: Bytes) -> Response {
    if mode() == Mode::Off {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(id) = valid_id(&id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if body.len() > MAX_INGEST_BYTES {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let v: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let list = match v {
        serde_json::Value::Array(a) => a,
        serde_json::Value::Object(mut o) => match o.remove("events") {
            Some(serde_json::Value::Array(a)) => a,
            _ => return StatusCode::BAD_REQUEST.into_response(),
        },
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    if list.len() > MAX_INGEST_EVENTS {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }
    let mut events = Vec::with_capacity(list.len());
    for e in list {
        let mut e: Event = match serde_json::from_value(e) {
            Ok(e) => e,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        };
        e.trace = id.clone();
        if e.host.is_empty() {
            e.host = "client".into();
        }
        events.push(e);
    }
    let n = events.len();
    if fastvideo_trace::ingest(events) {
        (StatusCode::ACCEPTED, Json(serde_json::json!({"accepted": n}))).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"accepted": 0, "dropped": n}))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_names() {
        assert_eq!(span_name(&Method::GET, "/files/a/b.mp4"), "get_file");
        assert_eq!(span_name(&Method::GET, "/v1/videos/x/content"), "get_content");
        assert_eq!(span_name(&Method::POST, "/fv/v1/jobs"), "post");
        assert_eq!(span_name(&Method::GET, "/fv/v1/jobs/x"), "get");
        assert!(query_opt_in(Some("a=1&fv_trace=1")));
        assert!(!query_opt_in(Some("fv_trace=0")));
    }
}
