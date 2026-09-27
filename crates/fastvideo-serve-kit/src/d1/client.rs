//! Cloudflare D1 HTTP API client (design §0 decision 7).
//!
//! One endpoint: `POST {api_base}/accounts/{account_id}/d1/database/{database_id}/query`
//! with `Authorization: Bearer <token>` and either one statement
//! `{"sql", "params"}` or a batch `{"batch": [{"sql", "params"}, ...]}`. A
//! batch runs as one SQL transaction (a failing statement rolls the whole
//! batch back). The reply is the Cloudflare v4 envelope:
//!
//! ```json
//! {"result":[{"results":[{"col":1}],"success":true,"meta":{"changes":0,...}}],
//!  "errors":[],"messages":[],"success":true}
//! ```
//!
//! and on failure `{"success":false,"errors":[{"code":7500,"message":"..."}]}`
//! (HTTP 400 for SQL errors such as `UNIQUE constraint failed ...
//! SQLITE_CONSTRAINT`). Observed against the live API on 2026-09-27: JSON
//! numbers bind as SQLite REAL (an INTEGER-affinity column stores them as
//! integers when lossless), `RETURNING` works, `sqlite_version()` is refused.
//!
//! [`D1Client`] adds retries with exponential backoff and jitter for
//! transport failures, HTTP 429/5xx and D1's transient errors; SQL errors
//! are never retried.

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value};

/// One SQL statement with positional (`?`) parameters.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Stmt {
    pub sql: String,
    pub params: Vec<Value>,
}

impl Stmt {
    pub fn new(sql: impl Into<String>, params: Vec<Value>) -> Self {
        Self { sql: sql.into(), params }
    }
    pub fn raw(sql: impl Into<String>) -> Self {
        Self::new(sql, Vec::new())
    }
}

/// Rows and write count of one statement.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StmtResult {
    pub rows: Vec<Map<String, Value>>,
    /// `meta.changes`: rows written by an INSERT/UPDATE/DELETE.
    pub changes: u64,
}

/// A D1 failure.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum D1Error {
    /// Network, TLS, timeout, or no response body.
    #[error("D1 transport: {0}")]
    Transport(String),
    /// A non-success HTTP status without a D1 error body.
    #[error("D1 HTTP {status}: {body}")]
    Http {
        status: u16,
        body: String,
        retry_after: Option<Duration>,
    },
    /// D1 reported an error (`errors[0]`), with the HTTP status.
    #[error("D1 error {code} (HTTP {status}): {message}")]
    Api { status: u16, code: i64, message: String },
    /// The reply did not have the expected shape.
    #[error("D1 reply: {0}")]
    Decode(String),
}

impl D1Error {
    /// Whether a retry may succeed.
    pub fn retryable(&self) -> bool {
        match self {
            D1Error::Transport(_) => true,
            D1Error::Http { status, .. } => *status == 429 || *status >= 500,
            D1Error::Api { status, message, .. } => {
                if *status == 429 || *status >= 500 {
                    return !is_sql_error(message);
                }
                is_transient_message(message)
            }
            D1Error::Decode(_) => false,
        }
    }
    /// A `UNIQUE`/`PRIMARY KEY` constraint violation.
    pub fn is_constraint(&self) -> bool {
        matches!(self, D1Error::Api { message, .. } if message.contains("SQLITE_CONSTRAINT") || message.contains("UNIQUE constraint failed"))
    }
    fn retry_after(&self) -> Option<Duration> {
        match self {
            D1Error::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

fn is_sql_error(m: &str) -> bool {
    m.contains("SQLITE_CONSTRAINT") || m.contains("syntax error") || m.contains("no such")
}

/// D1 errors that its docs describe as safe to retry (the storage object was
/// reset, the network to the primary was lost, it is overloaded, ...).
fn is_transient_message(m: &str) -> bool {
    let m = m.to_ascii_lowercase();
    [
        "network connection lost",
        "storage caused object to be reset",
        "object to be reset",
        "reset because its code was updated",
        "overloaded",
        "timed out",
        "timeout",
        "internal error",
        "try again",
    ]
    .iter()
    .any(|p| m.contains(p))
}

/// The raw reply of one POST.
#[derive(Clone, Debug, PartialEq)]
pub struct RawReply {
    pub status: u16,
    pub retry_after: Option<Duration>,
    /// The body parsed as JSON; `Value::Null` when it was not JSON.
    pub body: Value,
    /// The body as text when it was not JSON (for error messages).
    pub text: String,
}

/// Sends one `/query` body. Implemented over HTTP ([`HttpD1Transport`],
/// feature `fetch`) and by the SQLite mock (`d1-mock`).
#[async_trait::async_trait]
pub trait D1Transport: Send + Sync + 'static {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error>;
}

/// Where the database lives. `Debug` never prints the token.
#[derive(Clone, PartialEq, Eq)]
pub struct D1Config {
    /// `fv_cf_account_id`.
    pub account_id: String,
    /// `fv_cf_api_token` (D1 edit permission).
    pub api_token: String,
    /// `fv_d1_database_id` (fv-jobs: `1796e295-a7f0-4402-bbed-ec94ccb27c15`).
    pub database_id: String,
    /// `https://api.cloudflare.com/client/v4` (overridable for the mock).
    pub api_base: String,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl std::fmt::Debug for D1Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D1Config")
            .field("account_id", &self.account_id)
            .field("database_id", &self.database_id)
            .field("api_base", &self.api_base)
            .field("api_token", &"<redacted>")
            .finish()
    }
}

/// The public Cloudflare API base.
pub const CLOUDFLARE_API_BASE: &str = "https://api.cloudflare.com/client/v4";

impl D1Config {
    pub fn new(account_id: impl Into<String>, api_token: impl Into<String>, database_id: impl Into<String>) -> Self {
        Self {
            account_id: account_id.into(),
            api_token: api_token.into(),
            database_id: database_id.into(),
            api_base: CLOUDFLARE_API_BASE.into(),
            timeout: Duration::from_secs(30),
        }
    }
    /// `{api_base}/accounts/{account}/d1/database/{db}/query`.
    pub fn query_url(&self) -> String {
        format!(
            "{}/accounts/{}/d1/database/{}/query",
            self.api_base.trim_end_matches('/'),
            self.account_id,
            self.database_id
        )
    }
}

/// D1 over HTTPS (reqwest).
#[cfg(feature = "fetch")]
#[derive(Clone, Debug)]
pub struct HttpD1Transport {
    cfg: D1Config,
    http: reqwest::Client,
}

#[cfg(feature = "fetch")]
impl HttpD1Transport {
    pub fn new(cfg: D1Config) -> Result<Self, D1Error> {
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| D1Error::Transport(e.to_string()))?;
        Ok(Self { cfg, http })
    }
}

#[cfg(feature = "fetch")]
#[async_trait::async_trait]
impl D1Transport for HttpD1Transport {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        let r = self
            .http
            .post(self.cfg.query_url())
            .bearer_auth(&self.cfg.api_token)
            .json(body)
            .send()
            .await
            .map_err(|e| D1Error::Transport(e.to_string()))?;
        let status = r.status().as_u16();
        let retry_after = r
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        let text = r.text().await.map_err(|e| D1Error::Transport(e.to_string()))?;
        let body = serde_json::from_str(&text).unwrap_or(Value::Null);
        Ok(RawReply { status, retry_after, body, text })
    }
}

/// Retry schedule: `attempts` tries in all, delays `base * 2^n` capped at
/// `max_delay`, with up to 25% jitter; a `Retry-After` wins when longer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub attempts: u32,
    pub base: Duration,
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { attempts: 5, base: Duration::from_millis(100), max_delay: Duration::from_secs(5) }
    }
}

impl RetryPolicy {
    /// No waiting (tests).
    pub fn immediate(attempts: u32) -> Self {
        Self { attempts, base: Duration::ZERO, max_delay: Duration::ZERO }
    }
    fn delay(&self, retry: u32, hint: Option<Duration>) -> Duration {
        let exp = self.base.saturating_mul(1u32 << retry.min(16)).min(self.max_delay);
        let jitter = if exp.is_zero() {
            Duration::ZERO
        } else {
            // Cheap jitter from the uuid RNG; no extra dependency.
            let r = (uuid::Uuid::new_v4().as_u128() % 1000) as u32;
            exp.mul_f64(r as f64 / 4000.0)
        };
        (exp + jitter).max(hint.unwrap_or_default().min(Duration::from_secs(30)))
    }
}

/// D1 with retries; cheap to clone.
#[derive(Clone)]
pub struct D1Client {
    transport: Arc<dyn D1Transport>,
    retry: RetryPolicy,
}

impl std::fmt::Debug for D1Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D1Client").field("retry", &self.retry).finish_non_exhaustive()
    }
}

impl D1Client {
    pub fn new(transport: Arc<dyn D1Transport>) -> Self {
        Self { transport, retry: RetryPolicy::default() }
    }
    /// The HTTPS client for `cfg`.
    #[cfg(feature = "fetch")]
    pub fn http(cfg: D1Config) -> Result<Self, D1Error> {
        Ok(Self::new(Arc::new(HttpD1Transport::new(cfg)?)))
    }
    pub fn with_retry(mut self, r: RetryPolicy) -> Self {
        self.retry = r;
        self
    }

    /// One statement.
    pub async fn query(&self, stmt: Stmt) -> Result<StmtResult, D1Error> {
        let mut v = self.batch(vec![stmt]).await?;
        v.pop().ok_or_else(|| D1Error::Decode("empty result".into()))
    }

    /// Statements as one transaction (a single statement is sent unbatched).
    pub async fn batch(&self, stmts: Vec<Stmt>) -> Result<Vec<StmtResult>, D1Error> {
        if stmts.is_empty() {
            return Ok(Vec::new());
        }
        let n = stmts.len();
        let body = if n == 1 {
            serde_json::to_value(&stmts[0])
        } else {
            serde_json::to_value(serde_json::json!({ "batch": stmts }))
        }
        .map_err(|e| D1Error::Decode(e.to_string()))?;
        let mut tries = 0u32;
        loop {
            let res = match self.transport.post(&body).await {
                Ok(r) => decode(r, n),
                Err(e) => Err(e),
            };
            match res {
                Ok(v) => return Ok(v),
                Err(e) if e.retryable() && tries + 1 < self.retry.attempts => {
                    let d = self.retry.delay(tries, e.retry_after());
                    tracing::debug!(error = %e, attempt = tries + 1, delay_ms = d.as_millis() as u64, "D1 retry");
                    tries += 1;
                    if !d.is_zero() {
                        tokio::time::sleep(d).await;
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }
}

fn decode(r: RawReply, n: usize) -> Result<Vec<StmtResult>, D1Error> {
    let ok = r.body.get("success").and_then(Value::as_bool);
    if ok != Some(true) || !(200..300).contains(&r.status) {
        if let Some(e) = r.body.get("errors").and_then(Value::as_array).and_then(|a| a.first()) {
            return Err(D1Error::Api {
                status: r.status,
                code: e.get("code").and_then(Value::as_i64).unwrap_or(0),
                message: e.get("message").and_then(Value::as_str).unwrap_or_default().to_owned(),
            });
        }
        let mut body = if r.text.is_empty() { r.body.to_string() } else { r.text };
        body.truncate(500);
        return Err(D1Error::Http { status: r.status, body, retry_after: r.retry_after });
    }
    let arr = r
        .body
        .get("result")
        .and_then(Value::as_array)
        .ok_or_else(|| D1Error::Decode("missing `result`".into()))?;
    if arr.len() != n {
        return Err(D1Error::Decode(format!("expected {n} results, got {}", arr.len())));
    }
    arr.iter()
        .map(|x| {
            if x.get("success").and_then(Value::as_bool) == Some(false) {
                return Err(D1Error::Api {
                    status: r.status,
                    code: 0,
                    message: x.get("error").and_then(Value::as_str).unwrap_or("statement failed").to_owned(),
                });
            }
            let rows = x
                .get("results")
                .and_then(Value::as_array)
                .map(|rows| rows.iter().filter_map(|r| r.as_object().cloned()).collect())
                .unwrap_or_default();
            let changes = x
                .get("meta")
                .and_then(|m| m.get("changes"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0) as u64;
            Ok(StmtResult { rows, changes })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Script(Mutex<Vec<Result<RawReply, D1Error>>>, Mutex<u32>);

    #[async_trait::async_trait]
    impl D1Transport for Script {
        async fn post(&self, _b: &Value) -> Result<RawReply, D1Error> {
            *self.1.lock().unwrap() += 1;
            self.0.lock().unwrap().remove(0)
        }
    }

    fn reply(status: u16, body: Value) -> Result<RawReply, D1Error> {
        Ok(RawReply { status, retry_after: None, text: String::new(), body })
    }

    fn ok_body() -> Value {
        serde_json::json!({"result":[{"results":[{"a":1}],"success":true,"meta":{"changes":2}}],"errors":[],"success":true})
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let s = Arc::new(Script(
            Mutex::new(vec![
                Err(D1Error::Transport("reset".into())),
                reply(503, Value::Null),
                reply(500, serde_json::json!({"success":false,"errors":[{"code":7500,"message":"Network connection lost."}]})),
                reply(200, ok_body()),
            ]),
            Mutex::new(0),
        ));
        let c = D1Client::new(s.clone()).with_retry(RetryPolicy::immediate(5));
        let r = c.query(Stmt::raw("SELECT 1 AS a")).await.unwrap();
        assert_eq!(r.changes, 2);
        assert_eq!(r.rows[0]["a"], 1);
        assert_eq!(*s.1.lock().unwrap(), 4);
    }

    #[tokio::test]
    async fn sql_errors_are_not_retried() {
        let s = Arc::new(Script(
            Mutex::new(vec![reply(
                400,
                serde_json::json!({"success":false,"result":[],"errors":[{"code":7500,"message":"UNIQUE constraint failed: jobs.id: SQLITE_CONSTRAINT (extended: SQLITE_CONSTRAINT_PRIMARYKEY)"}]}),
            )]),
            Mutex::new(0),
        ));
        let c = D1Client::new(s.clone()).with_retry(RetryPolicy::immediate(5));
        let e = c.query(Stmt::raw("INSERT")).await.unwrap_err();
        assert!(e.is_constraint() && !e.retryable(), "{e}");
        assert_eq!(*s.1.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn gives_up_after_attempts() {
        let s = Arc::new(Script(
            Mutex::new((0..3).map(|_| reply(429, Value::Null)).collect()),
            Mutex::new(0),
        ));
        let c = D1Client::new(s.clone()).with_retry(RetryPolicy::immediate(3));
        assert!(matches!(c.query(Stmt::raw("x")).await, Err(D1Error::Http { status: 429, .. })));
        assert_eq!(*s.1.lock().unwrap(), 3);
    }

    #[test]
    fn config_redacts_and_builds_url() {
        let c = D1Config::new("acct", "sekrit", "db");
        assert!(!format!("{c:?}").contains("sekrit"));
        assert_eq!(c.query_url(), "https://api.cloudflare.com/client/v4/accounts/acct/d1/database/db/query");
        let p = RetryPolicy::default();
        assert!(p.delay(0, None) >= Duration::from_millis(100));
        assert!(p.delay(10, None) <= Duration::from_millis(6250));
        assert_eq!(p.delay(0, Some(Duration::from_secs(2))), Duration::from_secs(2));
    }
}
