//! Cloudflare D1 over its HTTP API (`scripts/serve/lib/d1.sh`): the release
//! history and the deployments ledger live in the `fv-jobs` database, next
//! to fv-serve's jobs and the gateway's tables (`deploy/d1/registry.sql`).
//!
//! Credentials, first set wins (the names fv-serve and the scripts read):
//! the token `FV_CF_API_TOKEN`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_API_KEY`;
//! the account `FV_CF_ACCOUNT_ID`, else the token's first account; the
//! database `FV_D1_DATABASE_ID`, else the one named `FV_D1_DATABASE_NAME`
//! (default `fv-jobs`); the API base `FV_D1_API_BASE` (tests: a mock). The
//! token is only ever a header; `Debug` never prints it and error texts are
//! scrubbed of it.

use std::cell::{Cell, RefCell};
use std::time::Duration;

use serde_json::{json, Value};

use crate::boot::env::BootEnv;
use crate::boot::sys::{Http, Method, Request};
use crate::provider::scrub;

/// The API base.
pub const API_BASE: &str = "https://api.cloudflare.com/client/v4";
/// The database the scripts and fv-serve use.
pub const DATABASE_NAME: &str = "fv-jobs";
/// One request's timeout (`--max-time 60`).
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// A D1 call that failed (the token never in the text).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("D1: {0}")]
pub struct D1Error(pub String);

/// The D1 client.
pub struct D1<'a> {
    http: &'a dyn Http,
    base: String,
    token: String,
    account: RefCell<Option<String>>,
    database: RefCell<Option<String>>,
    name: String,
    /// `deploy/d1/registry.sql`, applied once before the first use
    /// ([`D1::ensure_schema`]).
    schema_sql: Option<String>,
    schema_ok: Cell<bool>,
}

impl std::fmt::Debug for D1<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D1")
            .field("base", &self.base)
            .field("token", &"<redacted>")
            .field("account", &self.account.borrow())
            .field("database", &self.database.borrow())
            .finish_non_exhaustive()
    }
}

/// The token variables, in order (`fv_d1_token`).
pub const TOKEN_VARS: [&str; 3] = [
    "FV_CF_API_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_API_KEY",
];

impl<'a> D1<'a> {
    /// From the environment; `None` without a token (`fv_d1_available`).
    /// `schema_sql`: the text of `deploy/d1/registry.sql`.
    pub fn from_env(http: &'a dyn Http, env: &BootEnv, schema_sql: Option<String>) -> Option<Self> {
        let token = TOKEN_VARS.iter().find_map(|k| env.get(k))?.to_owned();
        Some(Self {
            http,
            base: env
                .get("FV_D1_API_BASE")
                .unwrap_or(API_BASE)
                .trim_end_matches('/')
                .to_owned(),
            token,
            account: RefCell::new(env.get("FV_CF_ACCOUNT_ID").map(str::to_owned)),
            database: RefCell::new(env.get("FV_D1_DATABASE_ID").map(str::to_owned)),
            name: env
                .get("FV_D1_DATABASE_NAME")
                .unwrap_or(DATABASE_NAME)
                .to_owned(),
            schema_sql,
            schema_ok: Cell::new(false),
        })
    }

    fn call(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Value, D1Error> {
        let mut req =
            Request::new(method, format!("{}{path}", self.base), TIMEOUT).bearer(&self.token);
        if let Some(b) = body {
            req = req.json(b.to_string());
        }
        let r = self
            .http
            .send(&req)
            .map_err(|e| D1Error(scrub(&e.to_string(), &self.token, 400)))?;
        let v: Value = serde_json::from_slice(&r.body).map_err(|_| {
            D1Error(format!(
                "HTTP {}: {}",
                r.status,
                scrub(&String::from_utf8_lossy(&r.body), &self.token, 400)
            ))
        })?;
        Ok(v)
    }

    /// Looks up (once) the account and the database (`fv_d1_resolve`).
    fn resolve(&self) -> Result<(String, String), D1Error> {
        if self.account.borrow().is_none() {
            let r = self.call(Method::Get, "/accounts", None)?;
            let id = r
                .pointer("/result/0/id")
                .and_then(Value::as_str)
                .ok_or_else(|| D1Error("the token sees no account".into()))?;
            *self.account.borrow_mut() = Some(id.to_owned());
        }
        let account = self.account.borrow().clone().unwrap_or_default();
        if self.database.borrow().is_none() {
            let r = self.call(
                Method::Get,
                &format!("/accounts/{account}/d1/database?name={}", self.name),
                None,
            )?;
            let id = r
                .get("result")
                .and_then(Value::as_array)
                .and_then(|a| {
                    a.iter()
                        .find(|d| d.get("name").and_then(Value::as_str) == Some(self.name.as_str()))
                })
                .and_then(|d| d.get("uuid").and_then(Value::as_str))
                .ok_or_else(|| D1Error(format!("no database named {}", self.name)))?;
            *self.database.borrow_mut() = Some(id.to_owned());
        }
        let database = self.database.borrow().clone().unwrap_or_default();
        Ok((account, database))
    }

    /// One statement with `?` parameters; the result rows (`fv_d1_query`).
    pub fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Value>, D1Error> {
        let (account, database) = self.resolve()?;
        let mut body = json!({ "sql": sql });
        if !params.is_empty() {
            body["params"] = Value::Array(params.to_vec());
        }
        let r = self.call(
            Method::Post,
            &format!("/accounts/{account}/d1/database/{database}/query"),
            Some(&body),
        )?;
        if r.get("success") != Some(&Value::Bool(true)) {
            let errs = r.get("errors").cloned().unwrap_or_else(|| r.clone());
            return Err(D1Error(scrub(&errs.to_string(), &self.token, 400)));
        }
        Ok(r.get("result")
            .and_then(Value::as_array)
            .map(|res| {
                res.iter()
                    .filter_map(|x| x.get("results").and_then(Value::as_array))
                    .flatten()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Applies `deploy/d1/registry.sql` once (`fv_registry_schema`: comments
    /// dropped, the statements sent together; every statement is `IF NOT
    /// EXISTS`).
    pub fn ensure_schema(&self) -> Result<(), D1Error> {
        if self.schema_ok.get() {
            return Ok(());
        }
        let Some(sql) = &self.schema_sql else {
            return Err(D1Error("no deploy/d1/registry.sql to apply".into()));
        };
        self.query(&schema_statements(sql), &[])?;
        self.schema_ok.set(true);
        Ok(())
    }
}

/// `fv_d1_exec_file`'s text: `--` comments dropped, newlines as spaces,
/// runs of spaces as one.
pub fn schema_statements(sql: &str) -> String {
    let text: Vec<&str> = sql
        .split('\n')
        .map(|l| l.split_once("--").map_or(l, |(a, _)| a))
        .collect();
    let joined = text.join(" ");
    let mut out = String::with_capacity(joined.len());
    let mut space = false;
    for c in joined.chars() {
        if c == ' ' {
            if !space {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_text_drops_comments_and_joins_lines() {
        let s = schema_statements("-- head\nCREATE TABLE a (x INT, -- note\n  y TEXT);\n");
        assert_eq!(s, " CREATE TABLE a (x INT, y TEXT); ");
    }
}
