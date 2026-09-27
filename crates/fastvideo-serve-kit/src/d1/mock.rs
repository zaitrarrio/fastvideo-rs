//! A local stand-in for the D1 HTTP API over a real SQLite database
//! (feature `d1-mock`; tests only).
//!
//! It answers the same JSON envelope as `POST .../d1/database/{id}/query`
//! (single statement or `{"batch": [...]}` in one transaction), binds JSON
//! numbers as REAL like D1 does, returns D1's error shape (HTTP 400, code
//! 7500, `... SQLITE_CONSTRAINT`), and can inject transient failures to
//! exercise retries. Use it directly as a [`D1Transport`] or serve
//! [`MockD1::router`] and point `D1Config::api_base` at it.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use rusqlite::types::{Value as SqlValue, ValueRef};
use rusqlite::Connection;
use serde_json::{json, Map, Value};

use super::client::{D1Error, D1Transport, RawReply};

struct Inner {
    conn: Connection,
    /// Replies to send instead of executing (HTTP status), front first.
    faults: Vec<u16>,
    requests: u64,
    statements: Vec<String>,
}

/// SQLite-backed D1 mock. Cheap to clone (shared database).
#[derive(Clone)]
pub struct MockD1 {
    inner: Arc<Mutex<Inner>>,
    token: Option<String>,
}

impl std::fmt::Debug for MockD1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MockD1")
    }
}

impl Default for MockD1 {
    fn default() -> Self {
        Self::new()
    }
}

impl MockD1 {
    /// A fresh in-memory database.
    pub fn new() -> Self {
        let conn = Connection::open_in_memory().expect("sqlite in memory");
        Self {
            inner: Arc::new(Mutex::new(Inner { conn, faults: Vec::new(), requests: 0, statements: Vec::new() })),
            token: None,
        }
    }
    /// Require `Authorization: Bearer <token>` on the HTTP router.
    pub fn with_token(mut self, t: impl Into<String>) -> Self {
        self.token = Some(t.into());
        self
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
    /// The next `n` requests fail with HTTP `status` (500, 503, 429, ...).
    pub fn fail_next(&self, n: usize, status: u16) {
        let mut g = self.lock();
        for _ in 0..n {
            g.faults.push(status);
        }
    }
    /// Requests received (including injected failures).
    pub fn requests(&self) -> u64 {
        self.lock().requests
    }
    /// Every executed SQL statement, in order.
    pub fn statements(&self) -> Vec<String> {
        self.lock().statements.clone()
    }
    /// Runs one statement locally (test inspection); rows as JSON objects.
    pub fn sql(&self, sql: &str, params: &[Value]) -> Result<Vec<Map<String, Value>>, String> {
        let g = self.lock();
        exec(&g.conn, sql, params).map(|(rows, _)| rows)
    }

    /// Executes one D1 request body; returns `(http status, reply)`.
    pub fn handle(&self, body: &Value) -> (u16, Value) {
        let mut g = self.lock();
        g.requests += 1;
        if !g.faults.is_empty() {
            let st = g.faults.remove(0);
            let msg = if st == 429 { "Too many requests" } else { "Network connection lost." };
            return (st, json!({"success": false, "result": [], "messages": [], "errors": [{"code": 7500, "message": msg}]}));
        }
        let stmts: Vec<(String, Vec<Value>)> = match body.get("batch").and_then(Value::as_array) {
            Some(b) => b.iter().map(parse_stmt).collect(),
            None => vec![parse_stmt(body)],
        };
        let batch = stmts.len() > 1;
        for (s, _) in &stmts {
            g.statements.push(s.clone());
        }
        let conn = &g.conn;
        if batch {
            let _ = conn.execute_batch("BEGIN");
        }
        let mut results = Vec::new();
        for (sql, params) in &stmts {
            match exec(conn, sql, params) {
                Ok((rows, changes)) => results.push(json!({
                    "results": rows, "success": true,
                    "meta": {"changes": changes, "served_by": "mock", "duration": 0.0}
                })),
                Err(e) => {
                    if batch {
                        let _ = conn.execute_batch("ROLLBACK");
                    }
                    return (400, json!({"success": false, "result": [], "messages": [], "errors": [{"code": 7500, "message": e}]}));
                }
            }
        }
        if batch {
            let _ = conn.execute_batch("COMMIT");
        }
        (200, json!({"result": results, "errors": [], "messages": [], "success": true}))
    }

    /// `POST /client/v4/accounts/{a}/d1/database/{d}/query`. Point
    /// `D1Config::api_base` at `http://<addr>/client/v4`.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/client/v4/accounts/{account}/d1/database/{db}/query", post(http_query))
            .with_state(self.clone())
    }
}

async fn http_query(State(m): State<MockD1>, headers: HeaderMap, Json(body): Json<Value>) -> impl IntoResponse {
    if let Some(t) = &m.token {
        let ok = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == format!("Bearer {t}"));
        if !ok {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"success": false, "result": null, "messages": [], "errors": [{"code": 10000, "message": "Authentication error"}]})),
            );
        }
    }
    let (st, v) = m.handle(&body);
    (StatusCode::from_u16(st).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(v))
}

#[async_trait::async_trait]
impl D1Transport for MockD1 {
    async fn post(&self, body: &Value) -> Result<RawReply, D1Error> {
        let (status, body) = self.handle(body);
        Ok(RawReply { status, retry_after: None, text: String::new(), body })
    }
}

fn parse_stmt(v: &Value) -> (String, Vec<Value>) {
    (
        v.get("sql").and_then(Value::as_str).unwrap_or_default().to_owned(),
        v.get("params").and_then(Value::as_array).cloned().unwrap_or_default(),
    )
}

fn bind(v: &Value) -> SqlValue {
    match v {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(*b as i64),
        // D1 binds every JSON number as REAL.
        Value::Number(n) => SqlValue::Real(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}

fn exec(conn: &Connection, sql: &str, params: &[Value]) -> Result<(Vec<Map<String, Value>>, usize), String> {
    let fmt = |e: rusqlite::Error| {
        let code = match &e {
            rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::ConstraintViolation => {
                ": SQLITE_CONSTRAINT"
            }
            _ => ": SQLITE_ERROR",
        };
        format!("{e}{code}")
    };
    let mut st = conn.prepare(sql).map_err(fmt)?;
    let names: Vec<String> = st.column_names().iter().map(|s| s.to_string()).collect();
    let readonly = st.readonly();
    let vals: Vec<SqlValue> = params.iter().map(bind).collect();
    let mut rows = st.query(rusqlite::params_from_iter(vals.iter())).map_err(fmt)?;
    let mut out = Vec::new();
    while let Some(r) = rows.next().map_err(fmt)? {
        let mut m = Map::new();
        for (i, n) in names.iter().enumerate() {
            let v = match r.get_ref(i).map_err(fmt)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(x) => json!(x),
                ValueRef::Real(x) => json!(x),
                ValueRef::Text(t) => json!(String::from_utf8_lossy(t)),
                ValueRef::Blob(b) => json!(b),
            };
            m.insert(n.clone(), v);
        }
        out.push(m);
    }
    drop(rows);
    drop(st);
    Ok((out, if readonly { 0 } else { conn.changes() as usize }))
}
