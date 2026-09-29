//! Experimental feature flags (docs/serve/console.md §7), toggled by an
//! admin on the console's admin page.
//!
//! | Route (admin token) | Behaviour |
//! |---|---|
//! | `GET /fv/v1/admin/flags` | every known flag: `name`, `enabled`, `default`, `description`, `updated_at`, `updated_by`, plus the `backend` |
//! | `PUT /fv/v1/admin/flags/{name}` `{"enabled": bool}` | sets it (404 for an unknown name); applies on this process at once |
//!
//! **Design.** Flags are durable in D1 (`feature_flags`, created on first
//! use like the release registry tables) when the server has the D1 job
//! store, else in `<state_dir>/feature_flags.json`. Every process caches
//! them and re-reads D1 every 30 s, so gateway replicas and workers
//! converge within that. A flag never reaches a worker through a dispatch:
//! flags act on the **caps** (`fastvideo_protocol::apply_feature_flags`)
//! through [`FlaggedGate`], the engine gate every API negotiates against
//! and the console's forms (`/fal/schema/…`) are built from. So the
//! gateway (or a single server) enforces them at negotiation, before any
//! job exists; workers receive only jobs that already passed. The only
//! worker-side use is the fal director, whose sessions run on a worker:
//! the worker reads the same D1 table.
//!
//! Flags:
//!
//! - `h3_1080p_long` (off by default): H3 native 1080P clips up to 10 s.
//!   Off, 1080P is limited to 5 s and a longer request gets a 4xx naming
//!   the limit and this flag (owner decision 2026-09-29).

// Handler helpers return a ready `Response` as their error.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, put};
use axum::{Json, Router};
use fastvideo_protocol::{ApiError, Job, JobId, ModelCaps};
use fastvideo_serve_kit::d1::{D1Client, Stmt};
use fastvideo_serve_kit::{AdminToken, EngineGate};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// A flag this server knows.
#[derive(Clone, Copy, Debug)]
pub struct FlagDef {
    pub name: &'static str,
    pub default: bool,
    pub description: &'static str,
}

/// Every known flag.
pub const FLAGS: &[FlagDef] = &[FlagDef {
    name: fastvideo_protocol::FLAG_H3_1080P_LONG,
    default: false,
    description: "H3 native 1080P clips up to 10 s. Off: 1080P is limited to 5 s and longer requests are refused. \
                  Experimental: 10 s at 1080P needs about 47 GiB of working memory (80 GB-class GPU with a streamed text encoder).",
}];

/// The D1 table (idempotent).
pub const TABLE_SQL: &str = "CREATE TABLE IF NOT EXISTS feature_flags (name TEXT PRIMARY KEY NOT NULL, enabled INTEGER NOT NULL, \
                             updated_at INTEGER NOT NULL, updated_by TEXT)";

/// One stored flag value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlagValue {
    pub enabled: bool,
    /// Unix milliseconds.
    pub updated_at: i64,
    pub updated_by: Option<String>,
}

enum Backend {
    D1 { db: D1Client, schema: tokio::sync::OnceCell<()> },
    File(PathBuf),
    Memory,
}

/// The flag cache and its durable store.
pub struct FeatureFlags {
    values: RwLock<BTreeMap<String, FlagValue>>,
    backend: Backend,
}

impl std::fmt::Debug for FeatureFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureFlags").field("backend", &self.backend_kind()).field("values", &self.values.read().ok()).finish()
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn known(name: &str) -> Option<&'static FlagDef> {
    FLAGS.iter().find(|f| f.name == name)
}

impl FeatureFlags {
    /// Process-local flags (tests).
    pub fn memory() -> Arc<Self> {
        Arc::new(Self { values: RwLock::new(BTreeMap::new()), backend: Backend::Memory })
    }

    /// Flags in D1, loaded now (a D1 error leaves the defaults and logs;
    /// the refresh retries).
    pub async fn d1(db: D1Client) -> Arc<Self> {
        let f = Arc::new(Self { values: RwLock::new(BTreeMap::new()), backend: Backend::D1 { db, schema: tokio::sync::OnceCell::new() } });
        if let Err(e) = f.reload().await {
            tracing::warn!(error = %e, "feature flags: D1 load failed; using the defaults until the next refresh");
        }
        f
    }

    /// Flags in a JSON file (`<state_dir>/feature_flags.json`).
    pub async fn file(path: PathBuf) -> Arc<Self> {
        let f = Arc::new(Self { values: RwLock::new(BTreeMap::new()), backend: Backend::File(path) });
        if let Err(e) = f.reload().await {
            tracing::warn!(error = %e, "feature flags: reading the file failed; using the defaults");
        }
        f
    }

    pub fn backend_kind(&self) -> &'static str {
        match self.backend {
            Backend::D1 { .. } => "d1",
            Backend::File(_) => "file",
            Backend::Memory => "memory",
        }
    }

    /// Whether `name` is on (its default when never set; `false` for an
    /// unknown name).
    pub fn enabled(&self, name: &str) -> bool {
        let Some(def) = known(name) else { return false };
        self.values.read().expect("flags").get(name).map_or(def.default, |v| v.enabled)
    }

    /// `caps` with the flags applied (see the module docs).
    pub fn apply(&self, caps: &mut ModelCaps) {
        fastvideo_protocol::apply_feature_flags(caps, &|n| self.enabled(n));
    }

    /// Every known flag as the admin API lists it.
    pub fn list(&self) -> Vec<Value> {
        let v = self.values.read().expect("flags");
        FLAGS
            .iter()
            .map(|d| {
                let s = v.get(d.name);
                json!({
                    "name": d.name,
                    "enabled": s.map_or(d.default, |s| s.enabled),
                    "default": d.default,
                    "experimental": true,
                    "description": d.description,
                    "updated_at": s.map(|s| s.updated_at),
                    "updated_by": s.and_then(|s| s.updated_by.clone()),
                })
            })
            .collect()
    }

    async fn ensure(&self) -> Result<(), String> {
        if let Backend::D1 { db, schema } = &self.backend {
            schema.get_or_try_init(|| async { db.query(Stmt::raw(TABLE_SQL)).await.map(|_| ()) }).await.map_err(|e| format!("feature_flags table: {e}"))?;
        }
        Ok(())
    }

    /// Re-reads the store into the cache.
    pub async fn reload(&self) -> Result<(), String> {
        let fresh: BTreeMap<String, FlagValue> = match &self.backend {
            Backend::Memory => return Ok(()),
            Backend::File(p) => match tokio::fs::read(p).await {
                Ok(b) => serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", p.display()))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
                Err(e) => return Err(format!("{}: {e}", p.display())),
            },
            Backend::D1 { db, .. } => {
                self.ensure().await?;
                let r = db
                    .query(Stmt::raw("SELECT name, enabled, updated_at, updated_by FROM feature_flags"))
                    .await
                    .map_err(|e| e.to_string())?;
                r.rows
                    .into_iter()
                    .filter_map(|row| {
                        let name = row.get("name")?.as_str()?.to_owned();
                        let enabled = row.get("enabled").and_then(Value::as_i64).unwrap_or(0) != 0;
                        let updated_at = row.get("updated_at").and_then(Value::as_i64).unwrap_or(0);
                        let updated_by = row.get("updated_by").and_then(Value::as_str).map(str::to_owned);
                        Some((name, FlagValue { enabled, updated_at, updated_by }))
                    })
                    .collect()
            }
        };
        *self.values.write().expect("flags") = fresh;
        Ok(())
    }

    /// Sets a known flag durably, then in this process's cache.
    pub async fn set(&self, name: &str, enabled: bool, by: Option<String>) -> Result<FlagValue, String> {
        if known(name).is_none() {
            return Err(format!("unknown feature flag `{name}`"));
        }
        let v = FlagValue { enabled, updated_at: now_ms(), updated_by: by };
        match &self.backend {
            Backend::Memory => {}
            Backend::File(p) => {
                let mut all = self.values.read().expect("flags").clone();
                all.insert(name.to_owned(), v.clone());
                let body = serde_json::to_vec_pretty(&all).map_err(|e| e.to_string())?;
                let tmp = p.with_extension("json.tmp");
                tokio::fs::write(&tmp, body).await.map_err(|e| format!("{}: {e}", tmp.display()))?;
                tokio::fs::rename(&tmp, p).await.map_err(|e| format!("{}: {e}", p.display()))?;
            }
            Backend::D1 { db, .. } => {
                self.ensure().await?;
                db.query(Stmt::new(
                    "INSERT INTO feature_flags (name, enabled, updated_at, updated_by) VALUES (?, ?, ?, ?) \
                     ON CONFLICT(name) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at, updated_by = excluded.updated_by",
                    vec![json!(name), json!(i64::from(enabled)), json!(v.updated_at), json!(v.updated_by)],
                ))
                .await
                .map_err(|e| e.to_string())?;
            }
        }
        self.values.write().expect("flags").insert(name.to_owned(), v.clone());
        tracing::info!(flag = name, enabled, "feature flag set");
        Ok(v)
    }

    /// D1: re-read every `every` so a flag set on another replica applies
    /// here (no-op task for the other backends).
    pub fn spawn_refresh(self: &Arc<Self>, every: Duration) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        tokio::spawn(async move {
            if !matches!(me.backend, Backend::D1 { .. }) {
                return;
            }
            let mut t = tokio::time::interval(every);
            t.tick().await;
            loop {
                t.tick().await;
                if let Err(e) = me.reload().await {
                    tracing::warn!(error = %e, "feature flags: refresh failed");
                }
            }
        })
    }
}

/// The engine gate with the flags applied to its caps: what every API
/// negotiates against and what the console's forms are built from.
pub struct FlaggedGate {
    pub inner: Arc<dyn EngineGate>,
    pub flags: Arc<FeatureFlags>,
}

#[async_trait::async_trait]
impl EngineGate for FlaggedGate {
    fn models(&self) -> Vec<ModelCaps> {
        let mut m = self.inner.models();
        for c in &mut m {
            self.flags.apply(c);
        }
        m
    }
    fn alias(&self, name: &str) -> Option<String> {
        self.inner.alias(name)
    }
    fn admit(&self) -> Result<(), ApiError> {
        self.inner.admit()
    }
    async fn submit(&self, job: &Job) -> Result<(), ApiError> {
        self.inner.submit(job).await
    }
    async fn cancel(&self, id: JobId) -> bool {
        self.inner.cancel(id).await
    }
}

#[derive(Clone)]
struct St {
    flags: Arc<FeatureFlags>,
    admin: Arc<AdminToken>,
}

fn err(code: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (code, Json(json!({"error": {"kind": kind, "message": msg.into()}}))).into_response()
}

fn body(flags: &FeatureFlags) -> Value {
    json!({"object": "fv.feature_flags", "flags": flags.list(), "backend": flags.backend_kind()})
}

async fn list(State(st): State<St>, headers: HeaderMap) -> Response {
    if !st.admin.check_headers(&headers) {
        return err(StatusCode::UNAUTHORIZED, "unauthorized", "the admin token is required");
    }
    // Fresh from the store, so the page shows what other replicas set.
    if let Err(e) = st.flags.reload().await {
        tracing::warn!(error = %e, "feature flags: reload for the admin list failed");
    }
    Json(body(&st.flags)).into_response()
}

#[derive(Deserialize)]
struct SetBody {
    enabled: bool,
}

async fn set(State(st): State<St>, headers: HeaderMap, Path(name): Path<String>, b: Option<Json<Value>>) -> Response {
    if !st.admin.check_headers(&headers) {
        return err(StatusCode::UNAUTHORIZED, "unauthorized", "the admin token is required");
    }
    if known(&name).is_none() {
        let names: Vec<&str> = FLAGS.iter().map(|f| f.name).collect();
        return err(StatusCode::NOT_FOUND, "not_found", format!("unknown feature flag `{name}`; known: {}", names.join(", ")));
    }
    let Some(Ok(SetBody { enabled })) = b.map(|Json(v)| serde_json::from_value::<SetBody>(v)) else {
        return err(StatusCode::BAD_REQUEST, "invalid_request", "expected a JSON body {\"enabled\": true | false}");
    };
    match st.flags.set(&name, enabled, Some("admin".into())).await {
        Ok(_) => Json(body(&st.flags)).into_response(),
        Err(e) => err(StatusCode::BAD_GATEWAY, "store", e),
    }
}

/// `/fv/v1/admin/flags` routes (admin token).
pub fn routes<S: Clone + Send + Sync + 'static>(flags: Arc<FeatureFlags>, admin: Arc<AdminToken>) -> Router<S> {
    Router::new()
        .route("/fv/v1/admin/flags", get(list))
        .route("/fv/v1/admin/flags/{name}", put(set))
        .with_state(St { flags, admin })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_flags_persist_and_default_off() {
        let dir = std::env::temp_dir().join(format!("fv-flags-{}", fastvideo_serve_kit::random_token()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("feature_flags.json");
        let f = FeatureFlags::file(p.clone()).await;
        assert!(!f.enabled("h3_1080p_long"));
        assert!(!f.enabled("nope"));
        f.set("h3_1080p_long", true, None).await.unwrap();
        assert!(f.set("nope", true, None).await.is_err());
        let g = FeatureFlags::file(p).await;
        assert!(g.enabled("h3_1080p_long"));
        assert_eq!(g.list()[0]["enabled"], true);
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn d1_flags_are_shared_between_processes() {
        let client = D1Client::new(Arc::new(fastvideo_serve_kit::d1::mock::MockD1::new()));
        let a = FeatureFlags::d1(client.clone()).await;
        let b = FeatureFlags::d1(client).await;
        a.set("h3_1080p_long", true, Some("admin".into())).await.unwrap();
        assert!(a.enabled("h3_1080p_long"));
        assert!(!b.enabled("h3_1080p_long"), "cached until the refresh");
        b.reload().await.unwrap();
        assert!(b.enabled("h3_1080p_long"));
        a.set("h3_1080p_long", false, None).await.unwrap();
        b.reload().await.unwrap();
        assert!(!b.enabled("h3_1080p_long"));
    }
}
