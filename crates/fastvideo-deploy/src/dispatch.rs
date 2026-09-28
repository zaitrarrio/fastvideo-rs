//! Runs **native** job envelopes against an axum router in-process
//! (design §6.4): the Runpod queue worker and the Vast PyWorker forwarder
//! both use it, so every API fv-serve mounts is reachable through them.
//!
//! - `kind: http` → one request through `tower::ServiceExt::oneshot`; with
//!   `wait: true` the created job is polled to a terminal state (the
//!   submit reply's `status_url` — fal, then fetching its `response_url`;
//!   MiniMax's `/v2/query/video_generation?task_id=`; else
//!   `<submit path>/{id}`) and its last status body is returned.
//! - `kind: stream` → `POST /fv/v1/streams` (the native WHIP session), progress
//!   `{"state":"live",…}`, then `GET /fv/v1/streams/{id}` until it ends;
//!   a cancel sends `DELETE /fv/v1/streams/{id}`. Output: the final body
//!   (session stats).
//! - `kind: info` → the host's diagnostics callback.
//!
//! Output of an http job (**native**):
//! `{"status":<u16>,"headers":{…},"body":<json|string>,"body_b64"?:…,
//!   "submit"?:<first body>,"poll_path"?:…,"elapsed_s":…}`.
//!
//! [`forward_routes`] mounts the same dispatcher at `POST /fv/v1/forward`
//! (the Vast PyWorker target; enabled by the host).

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use base64::Engine;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::runpod::{HttpJob, InfoJob, JobCtx, JobError, JobHandler, JobInput, StreamJob};

/// The forwarder route (never forwarded to itself).
pub const FORWARD_PATH: &str = "/fv/v1/forward";

/// Diagnostics callback for `kind: info`.
pub type InfoFn = Arc<dyn Fn(InfoJob) -> Pin<Box<dyn Future<Output = Value> + Send>> + Send + Sync>;

/// Limits of the dispatcher.
#[derive(Clone, Debug)]
pub struct DispatchOpts {
    /// Largest response body read back (Runpod caps job output at 20 MB).
    pub response_max: usize,
    pub poll_interval: Duration,
    pub wait_timeout: Duration,
    pub stream_poll: Duration,
}

impl Default for DispatchOpts {
    fn default() -> Self {
        Self {
            response_max: 15 << 20,
            poll_interval: Duration::from_secs(1),
            wait_timeout: Duration::from_secs(3600),
            stream_poll: Duration::from_secs(2),
        }
    }
}

/// A [`JobHandler`] over a router.
#[derive(Clone)]
pub struct RouterHandler {
    router: Router,
    info: Option<InfoFn>,
    opts: DispatchOpts,
    /// Added to every dispatched request unless the envelope sets it.
    extra_headers: Vec<(HeaderName, HeaderValue)>,
}

impl std::fmt::Debug for RouterHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterHandler").field("opts", &self.opts).finish_non_exhaustive()
    }
}

/// One answer read back from the router.
#[derive(Clone, Debug)]
struct Reply {
    status: u16,
    headers: BTreeMap<String, String>,
    body: Value,
    body_b64: Option<String>,
}

impl Reply {
    fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
    fn to_json(&self) -> Value {
        let mut v = json!({"status": self.status, "headers": self.headers, "body": self.body});
        if let Some(b) = &self.body_b64 {
            v["body_b64"] = json!(b);
        }
        v
    }
}

const KEEP_HEADERS: &[&str] = &["content-type", "location", "x-request-id", "retry-after", "content-length"];

/// The status word of a polled body, lower-cased (`status` or `state`,
/// MiniMax's `Success` / `Fail` included).
fn status_word(v: &Value) -> Option<String> {
    ["status", "state"].iter().find_map(|k| v.get(*k).and_then(Value::as_str)).map(str::to_ascii_lowercase)
}

fn is_terminal(word: &str) -> bool {
    matches!(
        word,
        "completed" | "complete" | "succeeded" | "success" | "failed" | "fail" | "error" | "cancelled" | "canceled" | "expired" | "ended" | "stopped"
    )
}

/// The id a submit reply names.
fn created_id(v: &Value) -> Option<String> {
    ["id", "task_id", "request_id", "job_id"].iter().find_map(|k| match v.get(*k) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    })
}

/// Path and query of an absolute or relative URL.
fn path_of(u: &str) -> Option<String> {
    if u.starts_with('/') {
        return Some(u.to_owned());
    }
    let p = url::Url::parse(u).ok()?;
    Some(match p.query() {
        Some(q) => format!("{}?{q}", p.path()),
        None => p.path().to_owned(),
    })
}

impl RouterHandler {
    pub fn new(router: Router) -> Self {
        Self { router, info: None, opts: DispatchOpts::default(), extra_headers: Vec::new() }
    }
    /// Adds `name: value` to every request this handler dispatches (a queue
    /// worker behind the gateway adds its internal token: queue jobs are
    /// already authenticated by the platform). Invalid names or values are
    /// ignored.
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        if let (Ok(n), Ok(v)) = (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(value)) {
            self.extra_headers.push((n, v));
        }
        self
    }
    pub fn with_info(mut self, f: InfoFn) -> Self {
        self.info = Some(f);
        self
    }
    pub fn with_opts(mut self, opts: DispatchOpts) -> Self {
        self.opts = opts;
        self
    }

    async fn call(
        &self,
        method: &str,
        path: &str,
        headers: &BTreeMap<String, String>,
        body: Option<(Vec<u8>, Option<&'static str>)>,
    ) -> Result<Reply, JobError> {
        if !path.starts_with('/') {
            return Err(JobError::new("InvalidInput", format!("path must start with `/`: {path}")));
        }
        if path.split('?').next() == Some(FORWARD_PATH) {
            return Err(JobError::new("InvalidInput", "the forwarder route cannot be forwarded to"));
        }
        let m = axum::http::Method::from_bytes(method.to_ascii_uppercase().as_bytes())
            .map_err(|_| JobError::new("InvalidInput", format!("bad method `{method}`")))?;
        let mut b = Request::builder().method(m).uri(path);
        let mut has_ct = false;
        let mut seen = Vec::new();
        for (k, v) in headers {
            let name = HeaderName::from_bytes(k.as_bytes()).map_err(|_| JobError::new("InvalidInput", format!("bad header name `{k}`")))?;
            let val = HeaderValue::from_str(v).map_err(|_| JobError::new("InvalidInput", format!("bad value for header `{k}`")))?;
            has_ct |= name == axum::http::header::CONTENT_TYPE;
            seen.push(name.clone());
            b = b.header(name, val);
        }
        for (n, v) in &self.extra_headers {
            if !seen.contains(n) {
                b = b.header(n.clone(), v.clone());
            }
        }
        let body = match body {
            Some((bytes, ct)) => {
                if let (false, Some(ct)) = (has_ct, ct) {
                    b = b.header(axum::http::header::CONTENT_TYPE, ct);
                }
                Body::from(bytes)
            }
            None => Body::empty(),
        };
        let req = b.body(body).map_err(|e| JobError::new("InvalidInput", e.to_string()))?;
        let resp = self.router.clone().oneshot(req).await.map_err(|e| JobError::new("DispatchError", e.to_string()))?;
        let status = resp.status().as_u16();
        let mut hs = BTreeMap::new();
        for (k, v) in resp.headers() {
            if KEEP_HEADERS.contains(&k.as_str()) {
                if let Ok(s) = v.to_str() {
                    hs.insert(k.as_str().to_owned(), s.to_owned());
                }
            }
        }
        let bytes = axum::body::to_bytes(resp.into_body(), self.opts.response_max).await.map_err(|e| {
            JobError::new("ResponseTooLarge", format!("response body over {} bytes (fetch it by URL instead): {e}", self.opts.response_max))
        })?;
        let (body, body_b64) = if bytes.is_empty() {
            (Value::Null, None)
        } else if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            (v, None)
        } else if let Ok(s) = std::str::from_utf8(&bytes) {
            (Value::String(s.to_owned()), None)
        } else {
            (Value::Null, Some(base64::engine::general_purpose::STANDARD.encode(&bytes)))
        };
        Ok(Reply { status, headers: hs, body, body_b64 })
    }

    /// Runs an http job.
    pub async fn http(&self, job: HttpJob, cx: &JobCtx) -> Result<Value, JobError> {
        let t0 = Instant::now();
        let body = match (&job.body, &job.body_b64) {
            (_, Some(b64)) => Some((
                base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| JobError::new("InvalidInput", format!("body_b64: {e}")))?,
                Some("application/octet-stream"),
            )),
            (Some(Value::String(s)), None) => Some((s.clone().into_bytes(), Some("text/plain"))),
            (Some(v), None) => Some((v.to_string().into_bytes(), Some("application/json"))),
            (None, None) => None,
        };
        let first = self.call(&job.method, &job.path, &job.headers, body).await?;
        if !job.wait || !first.ok() {
            let mut out = first.to_json();
            out["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
            return Ok(out);
        }
        // Wait for the created job.
        let poll = job
            .poll_path
            .clone()
            .or_else(|| first.body.get("status_url").and_then(Value::as_str).and_then(path_of))
            .or_else(|| {
                let base = job.path.split('?').next().unwrap_or_default();
                if base == "/v2/video_generation" {
                    Some("/v2/query/video_generation?task_id={id}".to_owned())
                } else {
                    Some(format!("{}/{{id}}", base.trim_end_matches('/')))
                }
            })
            .unwrap_or_default();
        let poll = match created_id(&first.body) {
            Some(id) => poll.replace("{id}", &id),
            None if poll.contains("{id}") => {
                return Err(JobError::new("WaitError", "wait: the submit reply names no id to poll (set `poll_path`)")
                    .with_output(first.to_json()))
            }
            None => poll,
        };
        let interval = job.poll_interval_ms.map(Duration::from_millis).unwrap_or(self.opts.poll_interval).max(Duration::from_millis(10));
        let deadline = t0 + job.timeout_s.map(Duration::from_secs).unwrap_or(self.opts.wait_timeout);
        let mut last_word = String::new();
        let mut last = first.clone();
        loop {
            if cx.cancel.is_cancelled() {
                if let Some(cp) = &job.cancel_path {
                    let cp = match created_id(&first.body) {
                        Some(id) => cp.replace("{id}", &id),
                        None => cp.clone(),
                    };
                    if let Err(e) = self.call("DELETE", &cp, &job.headers, None).await {
                        tracing::warn!(error = %e, path = %cp, "cancel_path failed");
                    }
                }
                return Err(JobError::new("Cancelled", "cancelled while waiting").with_output(last.to_json()));
            }
            if Instant::now() > deadline {
                return Err(JobError::new("WaitTimeout", format!("{poll} did not finish in time")).with_output(last.to_json()));
            }
            let r = self.call("GET", &poll, &job.headers, None).await?;
            let word = status_word(&r.body).unwrap_or_default();
            if word != last_word {
                cx.progress(json!({"state": word, "poll_path": poll}));
                last_word.clone_from(&word);
            }
            let done = !r.ok() || is_terminal(&word);
            last = r;
            if done {
                break;
            }
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = cx.cancel.cancelled() => {}
            }
        }
        // fal: the result lives at response_url.
        if last.ok() && matches!(last_word.as_str(), "completed") {
            if let Some(rp) = first.body.get("response_url").and_then(Value::as_str).and_then(path_of) {
                last = self.call("GET", &rp, &job.headers, None).await?;
            }
        }
        let mut out = last.to_json();
        out["submit"] = first.body;
        out["poll_path"] = json!(poll);
        out["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
        Ok(out)
    }

    /// Runs a stream job: one native WHIP session.
    pub async fn stream(&self, job: StreamJob, cx: &JobCtx) -> Result<Value, JobError> {
        let t0 = Instant::now();
        let mut headers = BTreeMap::new();
        headers.insert("content-type".to_owned(), "application/json".to_owned());
        let body = serde_json::to_vec(&job).map_err(|e| JobError::new("InvalidInput", e.to_string()))?;
        let created = self.call("POST", "/fv/v1/streams", &headers, Some((body, None))).await?;
        if !created.ok() {
            let msg = created
                .body
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("POST /fv/v1/streams answered {}", created.status));
            return Err(JobError::new("StreamRefused", msg).with_output(created.to_json()));
        }
        let id = created_id(&created.body).ok_or_else(|| JobError::new("StreamError", "the stream reply names no id"))?;
        let mut live = json!({"state": "live", "stream": created.body});
        live["stream_id"] = json!(id);
        cx.progress(live);
        let path = format!("/fv/v1/streams/{id}");
        let limit = job.duration_s.map(|d| Duration::from_secs(d + 120));
        let no_headers = BTreeMap::new();
        loop {
            tokio::select! {
                _ = tokio::time::sleep(self.opts.stream_poll) => {}
                _ = cx.cancel.cancelled() => {
                    let r = self.call("DELETE", &path, &no_headers, None).await;
                    let stats = r.map(|r| r.body).unwrap_or(Value::Null);
                    return Err(JobError::new("Cancelled", "stream stopped by cancel").with_output(stats));
                }
            }
            let r = self.call("GET", &path, &no_headers, None).await?;
            if !r.ok() {
                return Err(JobError::new("StreamError", format!("GET {path} answered {}", r.status)).with_output(r.to_json()));
            }
            let word = status_word(&r.body).unwrap_or_default();
            if is_terminal(&word) {
                let mut out = r.body;
                if out.is_object() {
                    out["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
                }
                return if matches!(word.as_str(), "failed" | "error") {
                    Err(JobError::new("StreamFailed", format!("stream {id} ended with `{word}`")).with_output(out))
                } else {
                    Ok(out)
                };
            }
            if limit.is_some_and(|l| t0.elapsed() > l) {
                let _ = self.call("DELETE", &path, &no_headers, None).await;
                return Err(JobError::new("StreamTimeout", format!("stream {id} outlived duration_s + 120 s")));
            }
        }
    }
}

#[async_trait::async_trait]
impl JobHandler for RouterHandler {
    async fn handle(&self, input: JobInput, cx: JobCtx) -> Result<Value, JobError> {
        match input {
            JobInput::Http(j) => self.http(j, &cx).await,
            JobInput::Stream(j) => self.stream(j, &cx).await,
            JobInput::Info(j) => match &self.info {
                Some(f) => Ok(f(j).await),
                None => Ok(json!({"version": crate::runpod::version()})),
            },
        }
    }
}

/// `POST /fv/v1/forward`: the body is a job envelope; the reply is the job
/// output (200) or `{"error":{"kind","message"},"output"?}` (502, or 400 for
/// a bad envelope). Streams are refused (batch only, design §6.2).
pub fn forward_routes<S: Clone + Send + Sync + 'static>(handler: RouterHandler) -> Router<S> {
    Router::new().route(FORWARD_PATH, post(forward)).with_state(Arc::new(handler))
}

async fn forward(State(h): State<Arc<RouterHandler>>, Json(v): Json<Value>) -> Response {
    let input = match JobInput::parse(&v) {
        Ok(JobInput::Stream(_)) => {
            return (StatusCode::BAD_REQUEST, Json(json!({"error": {"kind": "InvalidInput", "message": "streams are not forwarded"}})))
                .into_response()
        }
        Ok(i) => i,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": {"kind": "InvalidInput", "message": e}}))).into_response(),
    };
    let (cx, _rx) = JobCtx::detached(format!("fwd-{}", std::process::id()));
    match h.handle(input, cx).await {
        Ok(out) => Json(out).into_response(),
        Err(e) => {
            let code = if e.kind == "InvalidInput" { StatusCode::BAD_REQUEST } else { StatusCode::BAD_GATEWAY };
            (code, Json(json!({"error": {"kind": e.kind, "message": e.message}, "output": e.output}))).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(created_id(&json!({"task_id": "123"})).as_deref(), Some("123"));
        assert_eq!(created_id(&json!({"id": 5})).as_deref(), Some("5"));
        assert_eq!(path_of("https://h.example/minimax/h3-max/requests/abc/status?logs=1").as_deref(), Some("/minimax/h3-max/requests/abc/status?logs=1"));
        assert_eq!(status_word(&json!({"status": "Success"})).as_deref(), Some("success"));
        assert!(is_terminal("fail") && is_terminal("completed") && !is_terminal("processing") && !is_terminal("in_queue"));
    }
}
