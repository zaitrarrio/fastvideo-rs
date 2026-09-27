//! Runpod serverless queue worker (design §6.4; research-deploy §1.1,
//! behaviour pinned to `runpod-python` @ `760aea2`).
//!
//! ```text
//! task take:  loop GET {RUNPOD_WEBHOOK_GET_JOB}&job_in_progress={0|1}
//!             204/400 → continue; 429 → sleep 5 s; 200 {id,input} (or a list) → spawn job
//! task ping:  every RUNPOD_PING_INTERVAL: GET {PING}?job_id=<ids>&runpod_version=fv-rs/<ver>
//! task stop:  long-poll GET {GET_JOB with /job-take/→/job-stop/} → cancel matching jobs
//! per job:    progress → POST {POST_OUTPUT}&isStream=false {"status":"IN_PROGRESS","output":p}
//!             stream   → POST {POST_STREAM}&isStream=false {"output":chunk}
//!             done     → POST {POST_OUTPUT}&isStream=<streamed> {"output":…} | {"error":"<json>"}
//!                        3 attempts, Fibonacci backoff
//! headers:    Authorization: $RUNPOD_AI_API_KEY (raw), X-Request-ID: <job id>,
//!             Content-Type: application/x-www-form-urlencoded (the body is JSON text)
//! ```
//!
//! The URL templates are opaque: `$ID` / `$RUNPOD_POD_ID` are substituted,
//! query parameters are appended with `?` or `&` as needed, and nothing else
//! is assumed about their shape.
//!
//! - [`RunpodEnv`]: the injected variables.
//! - [`JobInput`]: the **native** job envelope (`kind: http | stream | info`).
//! - [`JobHandler`]: what runs a job; [`crate::dispatch::RouterHandler`] runs
//!   it against the fv-serve router in-process.
//! - [`Worker`]: the loop. [`Transport`] abstracts HTTP; `ReqwestTransport`
//!   (feature `runpod`) is the production one, [`RouterTransport`] drives an
//!   in-process router (the [`sim`] simulator in tests).

mod transport;
mod worker;
pub mod sim;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, watch};

#[cfg(feature = "runpod")]
pub use transport::ReqwestTransport;
pub use transport::{InResp, Method, OutReq, RouterTransport, Transport};
pub use worker::{Worker, WorkerOptions, WorkerReport};

/// Version string sent as `runpod_version` and in error reports.
pub fn version() -> String {
    format!("fv-rs/{}", env!("CARGO_PKG_VERSION"))
}

/// The variables Runpod injects into a queue worker (research-deploy §1.1).
#[derive(Clone)]
pub struct RunpodEnv {
    pub get_job: String,
    pub post_output: String,
    pub post_stream: Option<String>,
    pub ping: Option<String>,
    pub ping_interval: Duration,
    pub api_key: String,
    pub worker_id: String,
    pub hostname: Option<String>,
}

impl fmt::Debug for RunpodEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunpodEnv")
            .field("get_job", &self.get_job)
            .field("post_output", &self.post_output)
            .field("post_stream", &self.post_stream)
            .field("ping", &self.ping)
            .field("ping_interval", &self.ping_interval)
            .field("api_key", &if self.api_key.is_empty() { "<unset>" } else { "<redacted>" })
            .field("worker_id", &self.worker_id)
            .finish()
    }
}

/// Appends `k=v` to a URL that may or may not carry a query already.
pub fn append_query(url: &str, kv: &str) -> String {
    if url.contains('?') {
        if url.ends_with('?') || url.ends_with('&') {
            format!("{url}{kv}")
        } else {
            format!("{url}&{kv}")
        }
    } else {
        format!("{url}?{kv}")
    }
}

/// Percent-encodes a query value (job ids and versions are simple, but the
/// ids come from the platform).
fn qenc(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

impl RunpodEnv {
    /// Reads the variables through `var`. `Err` when this is not a queue
    /// worker (`RUNPOD_WEBHOOK_GET_JOB` unset: the SDK's "local" mode).
    pub fn from_lookup(var: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let get = |k: &str| var(k).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty());
        let get_job = get("RUNPOD_WEBHOOK_GET_JOB").ok_or("RUNPOD_WEBHOOK_GET_JOB is not set (not a Runpod queue worker)")?;
        let post_output = get("RUNPOD_WEBHOOK_POST_OUTPUT").ok_or("RUNPOD_WEBHOOK_POST_OUTPUT is not set")?;
        let ping_ms: u64 = get("RUNPOD_PING_INTERVAL").and_then(|v| v.parse().ok()).unwrap_or(10_000);
        Ok(Self {
            get_job,
            post_output,
            post_stream: get("RUNPOD_WEBHOOK_POST_STREAM"),
            ping: get("RUNPOD_WEBHOOK_PING"),
            ping_interval: Duration::from_millis(ping_ms.max(100)),
            api_key: get("RUNPOD_AI_API_KEY").unwrap_or_default(),
            // The SDK falls back to a uuid; any unique string works.
            worker_id: get("RUNPOD_POD_ID").unwrap_or_else(|| format!("fv-{}", std::process::id())),
            hostname: get("RUNPOD_POD_HOSTNAME").or_else(|| get("HOSTNAME")),
        })
    }

    /// From the process environment.
    pub fn from_process() -> Result<Self, String> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    fn worker_subst(&self, t: &str) -> String {
        t.replace("$RUNPOD_POD_ID", &self.worker_id).replace("$ID", &self.worker_id)
    }

    fn job_subst(&self, t: &str, job: &str) -> String {
        t.replace("$RUNPOD_POD_ID", &self.worker_id).replace("$ID", job)
    }

    /// `GET` job-take.
    pub fn take_url(&self, in_progress: bool) -> String {
        append_query(&self.worker_subst(&self.get_job), &format!("job_in_progress={}", u8::from(in_progress)))
    }

    /// `GET` job-take-batch (concurrency > 1).
    pub fn take_batch_url(&self, in_progress: bool, n: usize) -> String {
        let base = self.worker_subst(&self.get_job).replacen("/job-take/", "/job-take-batch/", 1);
        append_query(&append_query(&base, &format!("job_in_progress={}", u8::from(in_progress))), &format!("batch_size={n}"))
    }

    /// `GET` job-stop (long poll), or `None` if the template has no
    /// `/job-take/` segment to rewrite.
    pub fn stop_url(&self) -> Option<String> {
        self.get_job.contains("/job-take/").then(|| self.worker_subst(&self.get_job).replacen("/job-take/", "/job-stop/", 1))
    }

    /// `POST` job-done (also progress, with `is_stream = false`).
    pub fn done_url(&self, job: &str, is_stream: bool) -> String {
        append_query(&self.job_subst(&self.post_output, job), &format!("isStream={is_stream}"))
    }

    /// `POST` stream chunk.
    pub fn stream_url(&self, job: &str) -> Option<String> {
        self.post_stream.as_ref().map(|t| append_query(&self.job_subst(t, job), "isStream=false"))
    }

    /// `GET` heartbeat.
    pub fn ping_url(&self, jobs: &[String], version: &str) -> Option<String> {
        let t = self.ping.as_ref()?;
        let mut u = self.worker_subst(t);
        if !jobs.is_empty() {
            u = append_query(&u, &format!("job_id={}", qenc(&jobs.join(","))));
        }
        Some(append_query(&u, &format!("runpod_version={}", qenc(version))))
    }
}

/// A taken job as the platform sends it (`id` and `input` are required;
/// `webhook`, `batchId`, … are ignored).
#[derive(Clone, Debug, Deserialize)]
pub struct RawJob {
    pub id: String,
    pub input: Value,
}

/// `kind: "http"`: one request into the in-process router (**native**).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpJob {
    #[serde(default = "default_method")]
    pub method: String,
    /// Path and optional query, e.g. `/v2/video_generation`.
    pub path: String,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// JSON body (a string is sent as text).
    #[serde(default)]
    pub body: Option<Value>,
    /// Binary body, base64 (uploads).
    #[serde(default)]
    pub body_b64: Option<String>,
    /// Poll the created job to a terminal state and return its status body.
    #[serde(default)]
    pub wait: bool,
    /// Status path to poll; `{id}` is replaced by the created id. Default:
    /// the submit reply's `status_url` (fal), MiniMax's query route, else
    /// `<path>/{id}`.
    #[serde(default)]
    pub poll_path: Option<String>,
    #[serde(default)]
    pub poll_interval_ms: Option<u64>,
    /// Give up waiting after this long (default 3600 s).
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

fn default_method() -> String {
    "POST".into()
}

/// `kind: "stream"`: one WHIP session (**native**). `extra` passes any other
/// field through to `POST /fv/v1/streams`.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct StreamJob {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    pub whip_url: String,
    #[serde(default)]
    pub whip_token: Option<String>,
    #[serde(default)]
    pub duration_s: Option<u64>,
    #[serde(default)]
    pub image_url: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl fmt::Debug for StreamJob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamJob")
            .field("model", &self.model)
            .field("whip_url", &self.whip_url)
            .field("whip_token", &self.whip_token.as_ref().map(|_| "<redacted>"))
            .field("duration_s", &self.duration_s)
            .finish_non_exhaustive()
    }
}

/// `kind: "info"`: worker diagnostics (platform, weights, NVENC).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InfoJob {
    /// Also run a real NVENC encode.
    #[serde(default)]
    pub nvenc: bool,
}

/// The **native** job envelope (design §6.4).
#[derive(Clone, Debug)]
pub enum JobInput {
    Http(HttpJob),
    Stream(StreamJob),
    Info(InfoJob),
}

impl JobInput {
    /// Parses `input`; a missing `kind` with a `path` means `http`.
    pub fn parse(input: &Value) -> Result<Self, String> {
        let Value::Object(map) = input else { return Err("job input must be a JSON object".into()) };
        let mut map = map.clone();
        let kind = match map.remove("kind") {
            Some(Value::String(k)) => k,
            Some(_) => return Err("`kind` must be a string".into()),
            None if map.contains_key("path") => "http".into(),
            None => return Err("job input needs `kind` (http | stream | info)".into()),
        };
        let v = Value::Object(map);
        match kind.as_str() {
            "http" => serde_json::from_value(v).map(Self::Http).map_err(|e| format!("http job: {e}")),
            "stream" => serde_json::from_value(v).map(Self::Stream).map_err(|e| format!("stream job: {e}")),
            "info" => serde_json::from_value(v).map(Self::Info).map_err(|e| format!("info job: {e}")),
            k => Err(format!("unknown job kind `{k}` (http | stream | info)")),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Http(_) => "http",
            Self::Stream(_) => "stream",
            Self::Info(_) => "info",
        }
    }
}

/// A job failure: `kind` becomes `error_type` in the Runpod error report.
#[derive(Clone, Debug, PartialEq)]
pub struct JobError {
    pub kind: String,
    pub message: String,
    /// Partial output kept next to the error (`{"output":…,"error":…}`).
    pub output: Option<Value>,
}

impl JobError {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self { kind: kind.into(), message: message.into(), output: None }
    }
    pub fn with_output(mut self, v: Value) -> Self {
        self.output = Some(v);
        self
    }
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

/// Cancellation shared by the stop channel, shutdown and the handler.
#[derive(Clone, Debug)]
pub struct Cancel(Arc<watch::Sender<bool>>);

impl Default for Cancel {
    fn default() -> Self {
        Self(Arc::new(watch::channel(false).0))
    }
}

impl Cancel {
    pub fn cancel(&self) {
        self.0.send_replace(true);
    }
    pub fn is_cancelled(&self) -> bool {
        *self.0.borrow()
    }
    /// Resolves once cancelled.
    pub async fn cancelled(&self) {
        let mut rx = self.0.subscribe();
        // The sender lives in `self`, so this only ends by cancellation.
        let _ = rx.wait_for(|c| *c).await;
    }
}

/// What a running job sends back while it runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    /// `progress_update`: visible in `/status` as the output.
    Progress(Value),
    /// A stream chunk: visible at `/stream/{id}`.
    Stream(Value),
}

/// Per-job context handed to the [`JobHandler`].
#[derive(Clone, Debug)]
pub struct JobCtx {
    pub id: String,
    pub cancel: Cancel,
    tx: mpsc::UnboundedSender<Update>,
}

impl JobCtx {
    pub fn new(id: impl Into<String>, cancel: Cancel, tx: mpsc::UnboundedSender<Update>) -> Self {
        Self { id: id.into(), cancel, tx }
    }
    /// A context whose updates go nowhere (local one-shot runs).
    pub fn detached(id: impl Into<String>) -> (Self, mpsc::UnboundedReceiver<Update>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self::new(id, Cancel::default(), tx), rx)
    }
    pub fn progress(&self, v: Value) {
        let _ = self.tx.send(Update::Progress(v));
    }
    pub fn stream(&self, v: Value) {
        let _ = self.tx.send(Update::Stream(v));
    }
}

/// Runs one job.
#[async_trait::async_trait]
pub trait JobHandler: Send + Sync + 'static {
    async fn handle(&self, input: JobInput, cx: JobCtx) -> Result<Value, JobError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env() -> RunpodEnv {
        let m: BTreeMap<&str, &str> = [
            ("RUNPOD_WEBHOOK_GET_JOB", "https://api.runpod.ai/v2/ep1/job-take/$ID?gpu=NVIDIA+H200"),
            ("RUNPOD_WEBHOOK_POST_OUTPUT", "https://api.runpod.ai/v2/ep1/job-done/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200"),
            ("RUNPOD_WEBHOOK_POST_STREAM", "https://api.runpod.ai/v2/ep1/job-stream/$RUNPOD_POD_ID/$ID?gpu=NVIDIA+H200"),
            ("RUNPOD_WEBHOOK_PING", "https://api.runpod.ai/v2/ep1/ping/$RUNPOD_POD_ID"),
            ("RUNPOD_PING_INTERVAL", "4000"),
            ("RUNPOD_AI_API_KEY", "secret-key"),
            ("RUNPOD_POD_ID", "w42"),
        ]
        .into_iter()
        .collect();
        RunpodEnv::from_lookup(|k| m.get(k).map(|s| s.to_string())).unwrap()
    }

    #[test]
    fn url_templates() {
        let e = env();
        assert_eq!(e.take_url(false), "https://api.runpod.ai/v2/ep1/job-take/w42?gpu=NVIDIA+H200&job_in_progress=0");
        assert_eq!(
            e.take_batch_url(true, 3),
            "https://api.runpod.ai/v2/ep1/job-take-batch/w42?gpu=NVIDIA+H200&job_in_progress=1&batch_size=3"
        );
        assert_eq!(e.stop_url().unwrap(), "https://api.runpod.ai/v2/ep1/job-stop/w42?gpu=NVIDIA+H200");
        assert_eq!(e.done_url("j1", true), "https://api.runpod.ai/v2/ep1/job-done/w42/j1?gpu=NVIDIA+H200&isStream=true");
        assert_eq!(e.stream_url("j1").unwrap(), "https://api.runpod.ai/v2/ep1/job-stream/w42/j1?gpu=NVIDIA+H200&isStream=false");
        assert_eq!(
            e.ping_url(&["a".into(), "b".into()], "fv-rs/0.1").unwrap(),
            "https://api.runpod.ai/v2/ep1/ping/w42?job_id=a%2Cb&runpod_version=fv-rs%2F0.1"
        );
        assert_eq!(e.ping_url(&[], "v").unwrap(), "https://api.runpod.ai/v2/ep1/ping/w42?runpod_version=v");
        assert_eq!(e.ping_interval, Duration::from_secs(4));
        assert!(!format!("{e:?}").contains("secret-key"));
    }

    #[test]
    fn local_mode_is_an_error() {
        assert!(RunpodEnv::from_lookup(|_| None).is_err());
    }

    #[test]
    fn envelope_parsing() {
        let j = JobInput::parse(&json!({"kind":"http","method":"GET","path":"/fv/v1/capabilities"})).unwrap();
        assert!(matches!(j, JobInput::Http(HttpJob { ref method, .. }) if method == "GET"));
        let j = JobInput::parse(&json!({"path":"/v2/video_generation","body":{"model":"MiniMax-H3"},"wait":true})).unwrap();
        assert!(matches!(j, JobInput::Http(HttpJob { wait: true, ref method, .. }) if method == "POST"));
        let j = JobInput::parse(&json!({"kind":"stream","model":"wan-sf","prompt":"p","whip_url":"https://w/x","whip_token":"s3cr3t","duration_s":600,"image_url":null,"fps":16})).unwrap();
        let JobInput::Stream(s) = j else { panic!() };
        assert_eq!(s.extra["fps"], 16);
        assert!(!format!("{s:?}").contains("s3cr3t"));
        assert!(JobInput::parse(&json!({"kind":"info"})).is_ok());
        assert!(JobInput::parse(&json!({"kind":"shell"})).unwrap_err().contains("unknown job kind"));
        assert!(JobInput::parse(&json!({"prompt":"x"})).is_err());
        assert!(JobInput::parse(&json!("x")).is_err());
        assert!(JobInput::parse(&json!({"kind":"http","path":"/x","bogus":1})).is_err());
    }

    #[tokio::test]
    async fn cancel_token() {
        let c = Cancel::default();
        let c2 = c.clone();
        let t = tokio::spawn(async move { c2.cancelled().await });
        tokio::task::yield_now().await;
        c.cancel();
        t.await.unwrap();
        assert!(c.is_cancelled());
        // Already cancelled resolves immediately.
        c.cancelled().await;
    }
}
