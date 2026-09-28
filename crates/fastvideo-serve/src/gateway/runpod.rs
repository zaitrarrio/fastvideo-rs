//! The Runpod serverless queue API as the gateway uses it
//! (research-deploy §1.2): `POST /{ep}/run`, `GET /{ep}/status/{id}`,
//! `POST /{ep}/cancel/{id}`, `GET /{ep}/health`. The base is
//! `gateway.runpod_api_base` (`https://api.runpod.ai/v2`; the local
//! simulator `fastvideo_deploy::runpod::sim` in tests). The API key is sent
//! as `Authorization: Bearer` and never logged.

use std::time::Duration;

use serde_json::{json, Value};

/// A Runpod job's client-side view.
#[derive(Clone, Debug, PartialEq)]
pub struct RunStatus {
    /// `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMED_OUT`.
    pub status: String,
    pub output: Option<Value>,
    pub error: Option<Value>,
}

impl RunStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "COMPLETED" | "FAILED" | "CANCELLED" | "TIMED_OUT")
    }
    /// Still waiting in the queue or running.
    pub fn is_alive(&self) -> bool {
        matches!(self.status.as_str(), "IN_QUEUE" | "IN_PROGRESS")
    }
}

/// Client for the queue API.
#[derive(Clone)]
pub struct RunpodApi {
    http: reqwest::Client,
    base: String,
    key: String,
}

impl std::fmt::Debug for RunpodApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunpodApi").field("base", &self.base).finish_non_exhaustive()
    }
}

impl RunpodApi {
    pub fn new(http: reqwest::Client, base: &str, key: &str) -> Self {
        Self { http, base: base.trim_end_matches('/').to_owned(), key: key.to_owned() }
    }

    fn req(&self, m: reqwest::Method, path: &str, timeout: Duration) -> reqwest::RequestBuilder {
        let r = self.http.request(m, format!("{}{path}", self.base)).timeout(timeout);
        if self.key.is_empty() {
            r
        } else {
            r.bearer_auth(&self.key)
        }
    }

    async fn json(r: reqwest::RequestBuilder) -> Result<(u16, Value), String> {
        let resp = r.send().await.map_err(|e| format!("runpod: {}", e.without_url()))?;
        let status = resp.status().as_u16();
        let body = resp.bytes().await.map_err(|e| format!("runpod: {}", e.without_url()))?;
        Ok((status, serde_json::from_slice(&body).unwrap_or(Value::Null)))
    }

    /// Queues `input`; returns the Runpod job id.
    pub async fn run(&self, endpoint: &str, input: &Value, timeout: Duration) -> Result<String, String> {
        let (s, v) = Self::json(self.req(reqwest::Method::POST, &format!("/{endpoint}/run"), timeout).json(&json!({"input": input}))).await?;
        if !(200..300).contains(&s) {
            return Err(format!("runpod /run answered {s}: {}", short(&v)));
        }
        v.get("id").and_then(Value::as_str).map(str::to_owned).ok_or_else(|| format!("runpod /run reply has no id: {}", short(&v)))
    }

    /// The job's status; `None` when Runpod does not know it (404).
    pub async fn status(&self, endpoint: &str, id: &str) -> Result<Option<RunStatus>, String> {
        let (s, v) = Self::json(self.req(reqwest::Method::GET, &format!("/{endpoint}/status/{id}"), Duration::from_secs(15))).await?;
        if s == 404 {
            return Ok(None);
        }
        if !(200..300).contains(&s) {
            return Err(format!("runpod /status answered {s}"));
        }
        let status = v.get("status").and_then(Value::as_str).unwrap_or_default().to_owned();
        Ok(Some(RunStatus { status, output: v.get("output").cloned(), error: v.get("error").cloned() }))
    }

    pub async fn cancel(&self, endpoint: &str, id: &str) -> Result<(), String> {
        let (s, _) = Self::json(self.req(reqwest::Method::POST, &format!("/{endpoint}/cancel/{id}"), Duration::from_secs(15))).await?;
        if (200..300).contains(&s) || s == 404 {
            Ok(())
        } else {
            Err(format!("runpod /cancel answered {s}"))
        }
    }

    /// `GET /{ep}/health`: `{"jobs":{…},"workers":{…}}`.
    pub async fn health(&self, endpoint: &str) -> Result<Value, String> {
        let (s, v) = Self::json(self.req(reqwest::Method::GET, &format!("/{endpoint}/health"), Duration::from_secs(10))).await?;
        if (200..300).contains(&s) {
            Ok(v)
        } else {
            Err(format!("runpod /health answered {s}"))
        }
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 300 {
        format!("{}…", &s[..s.char_indices().nth(300).map_or(s.len(), |(i, _)| i)])
    } else {
        s
    }
}
