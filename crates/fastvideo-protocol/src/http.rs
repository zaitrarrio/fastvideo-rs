//! Framework-free `HttpReply`/`ReplyBody` and the `BatchProtocol`,
//! `SubmitEndpoint`, `JobView` traits (design §3.5).
//!
//! Every method of [`SubmitEndpoint`] and [`JobView`] is **pure**: golden
//! tests call them directly. serve-kit turns an [`HttpReply`] into an axum
//! response and supplies the generic `submit<E>`, `status<V>` and `result<V>`
//! handlers (auth -> ingestion -> `negotiate` -> `engine.submit` -> `JobStore`).

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use serde::Serialize;
use time::OffsetDateTime;

use crate::error::ApiError;
use crate::job::{Artifact, Job, JobId, KeyId};
use crate::request::{GenerationRequest, ProtocolId};

/// A response, independent of any HTTP framework.
#[derive(Clone, Debug, PartialEq)]
pub struct HttpReply {
    pub status: u16,
    /// Header names are lowercase static strings (e.g. `"x-fal-request-id"`).
    pub headers: Vec<(&'static str, String)>,
    pub body: ReplyBody,
}

impl HttpReply {
    /// JSON body with `status`.
    pub fn json(status: u16, value: serde_json::Value) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: ReplyBody::Json(value),
        }
    }
    /// Serializes `value` as the JSON body. A serialization failure (never
    /// for plain data types) becomes a 500 with a plain JSON message.
    pub fn json_of<T: Serialize>(status: u16, value: &T) -> Self {
        match serde_json::to_value(value) {
            Ok(v) => Self::json(status, v),
            Err(e) => Self::json(500, serde_json::json!({ "error": e.to_string() })),
        }
    }
    /// No body.
    pub fn empty(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: ReplyBody::Empty,
        }
    }
    /// In-memory bytes.
    pub fn bytes(status: u16, mime: impl Into<String>, data: impl Into<bytes::Bytes>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: ReplyBody::Bytes {
                mime: mime.into(),
                data: data.into(),
            },
        }
    }
    /// A file streamed from disk.
    pub fn file(status: u16, path: impl Into<PathBuf>, mime: impl Into<String>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: ReplyBody::File {
                path: path.into(),
                mime: mime.into(),
            },
        }
    }
    /// A server-sent-events stream.
    pub fn sse(spec: SseSpec) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body: ReplyBody::Sse(spec),
        }
    }
    /// Appends a header (builder style).
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }
    /// Appends a header.
    pub fn push_header(&mut self, name: &'static str, value: impl Into<String>) {
        self.headers.push((name, value.into()));
    }
    /// The first header named `name` (ASCII case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    /// The JSON body, if any.
    pub fn json_body(&self) -> Option<&serde_json::Value> {
        match &self.body {
            ReplyBody::Json(v) => Some(v),
            _ => None,
        }
    }
}

/// Response body.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplyBody {
    Json(serde_json::Value),
    Bytes { mime: String, data: bytes::Bytes },
    File { path: PathBuf, mime: String },
    Sse(SseSpec),
    Empty,
}

/// What an SSE response streams. The serve-kit handler emits `initial`, then
/// follows `follow` (if any), sending a `: keepalive` comment every
/// `keepalive` while idle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SseSpec {
    pub initial: Vec<SseEvent>,
    pub follow: Option<SseFollow>,
    pub keepalive: Option<Duration>,
}

/// A live source of SSE events.
#[derive(Clone, Debug, PartialEq)]
pub enum SseFollow {
    /// One event per job change: the handler watches the job
    /// (`JobStore::watch`) and renders each change with the endpoint's
    /// `JobView::status_reply` JSON as `data`. With `close_on_terminal`, the
    /// stream ends after the first terminal status (fal `/status/stream`).
    JobStatus { job: JobId, close_on_terminal: bool },
}

/// One SSE event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub id: Option<String>,
    pub data: String,
}

impl SseEvent {
    /// A `data:` only event.
    pub fn data(data: impl Into<String>) -> Self {
        Self {
            event: None,
            id: None,
            data: data.into(),
        }
    }
    /// Renders the event in wire form (`event:`/`id:`/`data:` lines, blank line).
    pub fn to_wire(&self) -> String {
        let mut s = String::new();
        if let Some(e) = &self.event {
            s.push_str("event: ");
            s.push_str(e);
            s.push('\n');
        }
        if let Some(i) = &self.id {
            s.push_str("id: ");
            s.push_str(i);
            s.push('\n');
        }
        for line in self.data.split('\n') {
            s.push_str("data: ");
            s.push_str(line);
            s.push('\n');
        }
        s.push('\n');
        s
    }
}

/// Context for rendering an error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ErrorCtx {
    /// The request id the reply carries (MiniMax `request_id`, LTX
    /// `x-request-id`, fal `x-fal-request-id`).
    pub request_id: Option<String>,
    /// The matched route template, e.g. `"/v1/text-to-video"` (LTX v1 vs v2 429s).
    pub route: Option<String>,
    /// The job's external id when the error concerns one.
    pub external_id: Option<String>,
}

/// Context for `SubmitEndpoint::normalize`.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizeCtx {
    pub now: OffsetDateTime,
    /// Authenticated caller, if any.
    pub owner: Option<KeyId>,
    /// Decoded query parameters in order (fal `fal_webhook`, `fal_max_queue_length`).
    pub query: Vec<(String, String)>,
    /// Request headers, names lowercase.
    pub headers: Vec<(String, String)>,
    pub request_id: String,
}

impl NormalizeCtx {
    pub fn new(now: OffsetDateTime) -> Self {
        Self {
            now,
            owner: None,
            query: Vec::new(),
            headers: Vec::new(),
            request_id: String::new(),
        }
    }
    /// First query parameter named `name`.
    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
    /// First header named `name` (ASCII case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Context for rendering job views.
#[derive(Clone, Copy)]
pub struct ViewCtx<'a> {
    pub now: OffsetDateTime,
    pub urls: &'a dyn UrlSigner,
    pub public_base: &'a url::Url,
    /// fal `?logs=1`.
    pub with_logs: bool,
}

impl fmt::Debug for ViewCtx<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ViewCtx")
            .field("now", &self.now)
            .field("public_base", &self.public_base.as_str())
            .field("with_logs", &self.with_logs)
            .finish_non_exhaustive()
    }
}

impl ViewCtx<'_> {
    /// `public_base` joined with `path` (a leading `/` is relative to the base
    /// path, not the host root). Falls back to the base on a malformed path.
    pub fn public_url(&self, path: &str) -> url::Url {
        let mut base = self.public_base.clone();
        if !base.path().ends_with('/') {
            let p = format!("{}/", base.path());
            base.set_path(&p);
        }
        base.join(path.trim_start_matches('/')).unwrap_or(base)
    }
}

/// Signs artifact download URLs (local HMAC `/files/...` or S3 presign).
pub trait UrlSigner: Send + Sync {
    fn url_for(&self, a: &Artifact, ttl: Duration) -> url::Url;
}

/// One batch API (per crate).
pub trait BatchProtocol: Send + Sync + 'static {
    fn id(&self) -> ProtocolId;
    /// The wire id for a new job (fal uuid, MiniMax 18 digits, `video_gen_<32hex>`, ...).
    fn new_external_id(&self, job: JobId) -> String;
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply;
}

/// One per submit endpoint (e.g. fal text-to-video, LTX v2 image-to-video).
pub trait SubmitEndpoint: Send + Sync + 'static {
    type Body: serde::de::DeserializeOwned + Send;
    fn normalize(&self, body: Self::Body, cx: &NormalizeCtx)
        -> Result<GenerationRequest, ApiError>;
    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
}

/// Status and result rendering for one API's jobs.
pub trait JobView: Send + Sync + 'static {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
}
