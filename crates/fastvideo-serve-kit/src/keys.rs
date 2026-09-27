//! Dynamic API keys and the admin token (console, WP-20).
//!
//! - [`AdminToken`]: the operator secret for `/fv/v1/admin/*`. Set it with
//!   `FV_ADMIN_TOKEN`, or let the server generate one at startup
//!   ([`AdminToken::generate`], `fvadm_` + 32 CSPRNG bytes, base64url) and log
//!   it once. Only its SHA-256 digest is held; checks are constant time.
//! - [`KeyStore`]: keys minted at run time (`fv_` + 32 CSPRNG bytes). The
//!   plaintext is returned once by [`KeyStore::mint`]; the store keeps only
//!   the SHA-256 digest, a display prefix, a name and timestamps. Backends:
//!   [`D1KeyBackend`] (table `api_keys`, D1 migration 2), [`FileKeyBackend`]
//!   (a JSON file, mode 0600) and [`MemoryKeyBackend`].
//! - Lookups are synchronous against an in-memory cache ([`KeyCheck`], used
//!   by [`crate::Auth`] for every API and scheme). `last_used_at` is updated
//!   in memory at most once per `touch_every` per key and written back by
//!   [`KeyStore::flush`]; [`KeyStore::refresh`] reloads the backend so keys
//!   minted or revoked by another worker (D1) take effect here.
//! - [`admin_routes`]: `POST/GET /fv/v1/admin/keys`,
//!   `DELETE /fv/v1/admin/keys/{id}` (native shapes).
//!
//! A key's id is its owner [`KeyId`] (`key_<first 12 hex of the digest>`),
//! the same id `FV_API_KEYS` keys get, so jobs list under it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get};
use axum::{Json, Router};
use base64::Engine as _;
use fastvideo_protocol::KeyId;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::d1::{D1Client, Stmt};

/// Prefix of minted API keys.
pub const KEY_PREFIX: &str = "fv_";
/// Prefix of generated admin tokens.
pub const ADMIN_PREFIX: &str = "fvadm_";
/// Longest key name.
pub const NAME_MAX_CHARS: usize = 64;

/// `prefix` + base64url(32 bytes from the thread CSPRNG, OS-seeded ChaCha).
pub fn random_secret(prefix: &str) -> String {
    let mut b = [0u8; 32];
    rand::rng().fill_bytes(&mut b);
    format!("{prefix}{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b))
}

fn digest(s: &str) -> [u8; 32] {
    Sha256::digest(s.as_bytes()).into()
}

fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The owner id of a key digest (matches `KeyRing::check`).
pub fn key_id_of(d: &[u8; 32]) -> String {
    format!("key_{}", &crate::hex::encode(d)[..12])
}

fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}

// ---------------------------------------------------------------- admin token

/// The admin token's digest. `Debug` never prints it.
#[derive(Clone)]
pub struct AdminToken {
    digest: [u8; 32],
}

impl std::fmt::Debug for AdminToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AdminToken(<redacted>)")
    }
}

impl AdminToken {
    /// From a configured secret (`FV_ADMIN_TOKEN`).
    pub fn from_secret(s: &str) -> Self {
        Self { digest: digest(s.trim()) }
    }

    /// A fresh token: `(token, plaintext)`. The caller shows the plaintext
    /// once and drops it.
    pub fn generate() -> (Self, String) {
        let t = random_secret(ADMIN_PREFIX);
        (Self::from_secret(&t), t)
    }

    /// Constant-time comparison of digests.
    pub fn check(&self, presented: &str) -> bool {
        let p = presented.trim();
        !p.is_empty() && ct_eq(&self.digest, &digest(p))
    }

    /// `Authorization: Bearer <t>` (or `Key <t>`), or `x-fv-admin-token`.
    pub fn check_headers(&self, headers: &HeaderMap) -> bool {
        if let Some(v) = headers.get("x-fv-admin-token").and_then(|v| v.to_str().ok()) {
            return self.check(v);
        }
        headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(crate::auth::parse_authorization)
            .is_some_and(|(_, t)| self.check(&t))
    }
}

// ---------------------------------------------------------------- records

/// One stored key. `digest` is the SHA-256 hex of the key; the key itself is
/// never stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    /// `key_<12 hex>`: the owner id jobs are recorded under.
    pub id: String,
    pub name: String,
    /// Display hint, e.g. `fv_Ab3dE9…`.
    pub prefix: String,
    pub digest: String,
    /// Unix milliseconds.
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub revoked_at: Option<i64>,
}

impl ApiKeyRecord {
    fn digest_bytes(&self) -> Option<[u8; 32]> {
        let b = crate::hex::decode(&self.digest)?;
        b.try_into().ok()
    }

    /// The admin API's view (no digest), timestamps as RFC 3339.
    pub fn view(&self) -> Value {
        let rfc = |ms: i64| {
            time::OffsetDateTime::from_unix_timestamp_nanos(ms as i128 * 1_000_000)
                .ok()
                .and_then(|t| t.format(&time::format_description::well_known::Rfc3339).ok())
        };
        json!({
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "created_at": rfc(self.created_at),
            "last_used_at": self.last_used_at.and_then(rfc),
            "revoked_at": self.revoked_at.and_then(rfc),
            "revoked": self.revoked_at.is_some(),
        })
    }
}

/// Where records persist.
#[async_trait::async_trait]
pub trait KeyBackend: Send + Sync + 'static {
    /// `memory`, `file` or `d1`.
    fn kind(&self) -> &'static str;
    async fn load(&self) -> Result<Vec<ApiKeyRecord>, String>;
    async fn insert(&self, r: &ApiKeyRecord) -> Result<(), String>;
    /// Sets `revoked_at` if unset; `false` when the id is unknown.
    async fn revoke(&self, id: &str, at: i64) -> Result<bool, String>;
    /// Raises `last_used_at` for each `(id, at)`.
    async fn touch(&self, updates: &[(String, i64)]) -> Result<(), String>;
}

/// Records in memory only (lost at exit).
#[derive(Debug, Default)]
pub struct MemoryKeyBackend {
    rows: std::sync::Mutex<Vec<ApiKeyRecord>>,
}

fn apply_revoke(rows: &mut [ApiKeyRecord], id: &str, at: i64) -> bool {
    match rows.iter_mut().find(|r| r.id == id) {
        Some(r) => {
            r.revoked_at.get_or_insert(at);
            true
        }
        None => false,
    }
}

fn apply_touch(rows: &mut [ApiKeyRecord], updates: &[(String, i64)]) {
    for (id, at) in updates {
        if let Some(r) = rows.iter_mut().find(|r| &r.id == id) {
            r.last_used_at = Some(r.last_used_at.map_or(*at, |x| x.max(*at)));
        }
    }
}

#[async_trait::async_trait]
impl KeyBackend for MemoryKeyBackend {
    fn kind(&self) -> &'static str {
        "memory"
    }
    async fn load(&self) -> Result<Vec<ApiKeyRecord>, String> {
        Ok(self.rows.lock().map_err(|e| e.to_string())?.clone())
    }
    async fn insert(&self, r: &ApiKeyRecord) -> Result<(), String> {
        self.rows.lock().map_err(|e| e.to_string())?.push(r.clone());
        Ok(())
    }
    async fn revoke(&self, id: &str, at: i64) -> Result<bool, String> {
        Ok(apply_revoke(&mut self.rows.lock().map_err(|e| e.to_string())?, id, at))
    }
    async fn touch(&self, updates: &[(String, i64)]) -> Result<(), String> {
        apply_touch(&mut self.rows.lock().map_err(|e| e.to_string())?, updates);
        Ok(())
    }
}

/// Records in one JSON file (`{"keys": [...]}`), rewritten atomically
/// (temp file + rename, mode 0600 on Unix).
#[derive(Debug)]
pub struct FileKeyBackend {
    path: PathBuf,
    lock: tokio::sync::Mutex<()>,
}

#[derive(Default, Serialize, Deserialize)]
struct KeyFile {
    #[serde(default)]
    keys: Vec<ApiKeyRecord>,
}

impl FileKeyBackend {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), lock: tokio::sync::Mutex::new(()) }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    async fn read(&self) -> Result<Vec<ApiKeyRecord>, String> {
        match tokio::fs::read(&self.path).await {
            Ok(b) => serde_json::from_slice::<KeyFile>(&b)
                .map(|f| f.keys)
                .map_err(|e| format!("{}: {e}", self.path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("{}: {e}", self.path.display())),
        }
    }

    async fn write(&self, keys: Vec<ApiKeyRecord>) -> Result<(), String> {
        let err = |e: std::io::Error| format!("{}: {e}", self.path.display());
        if let Some(dir) = self.path.parent() {
            tokio::fs::create_dir_all(dir).await.map_err(err)?;
        }
        let tmp = self.path.with_extension(format!("tmp-{}", crate::random_token()));
        let body = serde_json::to_vec_pretty(&KeyFile { keys }).map_err(|e| e.to_string())?;
        tokio::fs::write(&tmp, body).await.map_err(err)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600)).await.map_err(err)?;
        }
        tokio::fs::rename(&tmp, &self.path).await.map_err(err)
    }
}

#[async_trait::async_trait]
impl KeyBackend for FileKeyBackend {
    fn kind(&self) -> &'static str {
        "file"
    }
    async fn load(&self) -> Result<Vec<ApiKeyRecord>, String> {
        let _g = self.lock.lock().await;
        self.read().await
    }
    async fn insert(&self, r: &ApiKeyRecord) -> Result<(), String> {
        let _g = self.lock.lock().await;
        let mut rows = self.read().await?;
        rows.push(r.clone());
        self.write(rows).await
    }
    async fn revoke(&self, id: &str, at: i64) -> Result<bool, String> {
        let _g = self.lock.lock().await;
        let mut rows = self.read().await?;
        let hit = apply_revoke(&mut rows, id, at);
        if hit {
            self.write(rows).await?;
        }
        Ok(hit)
    }
    async fn touch(&self, updates: &[(String, i64)]) -> Result<(), String> {
        let _g = self.lock.lock().await;
        let mut rows = self.read().await?;
        apply_touch(&mut rows, updates);
        self.write(rows).await
    }
}

/// Records in the D1 table `api_keys` (created by the job store's migration
/// 2; [`D1KeyBackend::open`] runs the migrations itself).
#[derive(Debug)]
pub struct D1KeyBackend {
    db: D1Client,
}

impl D1KeyBackend {
    pub async fn open(db: D1Client) -> Result<Self, String> {
        crate::d1::schema::migrate(&db).await.map_err(|e| e.to_string())?;
        Ok(Self { db })
    }
}

fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)))
}

#[async_trait::async_trait]
impl KeyBackend for D1KeyBackend {
    fn kind(&self) -> &'static str {
        "d1"
    }
    async fn load(&self) -> Result<Vec<ApiKeyRecord>, String> {
        let r = self
            .db
            .query(Stmt::raw(
                "SELECT id, name, prefix, digest, created_at, last_used_at, revoked_at FROM api_keys ORDER BY created_at",
            ))
            .await
            .map_err(|e| e.to_string())?;
        Ok(r.rows
            .iter()
            .filter_map(|row| {
                let s = |k: &str| row.get(k).and_then(Value::as_str).map(str::to_owned);
                Some(ApiKeyRecord {
                    id: s("id")?,
                    name: s("name")?,
                    prefix: s("prefix")?,
                    digest: s("digest")?,
                    created_at: int(row.get("created_at"))?,
                    last_used_at: int(row.get("last_used_at")),
                    revoked_at: int(row.get("revoked_at")),
                })
            })
            .collect())
    }
    async fn insert(&self, r: &ApiKeyRecord) -> Result<(), String> {
        self.db
            .query(Stmt::new(
                "INSERT INTO api_keys (id, name, prefix, digest, created_at, last_used_at, revoked_at) VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                vec![json!(r.id), json!(r.name), json!(r.prefix), json!(r.digest), json!(r.created_at)],
            ))
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    async fn revoke(&self, id: &str, at: i64) -> Result<bool, String> {
        self.db
            .query(Stmt::new(
                "UPDATE api_keys SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?",
                vec![json!(at), json!(id)],
            ))
            .await
            .map(|r| r.changes > 0)
            .map_err(|e| e.to_string())
    }
    async fn touch(&self, updates: &[(String, i64)]) -> Result<(), String> {
        if updates.is_empty() {
            return Ok(());
        }
        let stmts = updates
            .iter()
            .map(|(id, at)| {
                Stmt::new(
                    "UPDATE api_keys SET last_used_at = MAX(COALESCE(last_used_at, 0), ?) WHERE id = ?",
                    vec![json!(at), json!(id)],
                )
            })
            .collect();
        self.db.batch(stmts).await.map(|_| ()).map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------- store

/// A synchronous key lookup (the auth hot path).
pub trait KeyCheck: Send + Sync + 'static {
    /// The owner id if `key` is a live key.
    fn check(&self, key: &str) -> Option<KeyId>;
}

struct Entry {
    record: ApiKeyRecord,
    /// `last_used_at` as the backend last saw it.
    persisted: Option<i64>,
}

/// Minted keys: a cache over a [`KeyBackend`].
pub struct KeyStore {
    backend: Arc<dyn KeyBackend>,
    cache: RwLock<HashMap<[u8; 32], Entry>>,
    touch_every: Duration,
}

impl std::fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.cache.read().map(|c| c.len()).unwrap_or(0);
        write!(f, "KeyStore({}, {n} keys)", self.backend.kind())
    }
}

/// Why a mint or revoke failed.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeyError {
    #[error("{0}")]
    Invalid(String),
    #[error("key store: {0}")]
    Backend(String),
}

impl KeyStore {
    /// Loads the backend's records.
    pub async fn open(backend: Arc<dyn KeyBackend>) -> Result<Arc<Self>, String> {
        let s = Arc::new(Self { backend, cache: RwLock::new(HashMap::new()), touch_every: Duration::from_secs(60) });
        s.refresh().await?;
        Ok(s)
    }

    /// An empty in-memory store.
    pub async fn memory() -> Arc<Self> {
        Self::open(Arc::new(MemoryKeyBackend::default())).await.expect("memory backend")
    }

    /// How often `last_used_at` may change per key (default 60 s).
    pub fn with_touch_every(self: Arc<Self>, d: Duration) -> Arc<Self> {
        match Arc::try_unwrap(self) {
            Ok(mut s) => {
                s.touch_every = d;
                Arc::new(s)
            }
            Err(s) => s,
        }
    }

    pub fn backend_kind(&self) -> &'static str {
        self.backend.kind()
    }

    /// Mints a key named `name`; returns `(plaintext, record)`. The
    /// plaintext is not kept anywhere.
    pub async fn mint(&self, name: &str) -> Result<(String, ApiKeyRecord), KeyError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(KeyError::Invalid("`name` must not be empty".into()));
        }
        if name.chars().count() > NAME_MAX_CHARS {
            return Err(KeyError::Invalid(format!("`name` must be at most {NAME_MAX_CHARS} characters")));
        }
        if name.chars().any(char::is_control) {
            return Err(KeyError::Invalid("`name` must not contain control characters".into()));
        }
        let (key, d) = loop {
            let k = random_secret(KEY_PREFIX);
            let d = digest(&k);
            let id = key_id_of(&d);
            let taken = self.cache.read().map(|c| c.values().any(|e| e.record.id == id)).unwrap_or(false);
            if !taken {
                break (k, d);
            }
        };
        let rec = ApiKeyRecord {
            id: key_id_of(&d),
            name: name.to_owned(),
            prefix: format!("{}…", &key[..KEY_PREFIX.len() + 6]),
            digest: crate::hex::encode(&d),
            created_at: now_ms(),
            last_used_at: None,
            revoked_at: None,
        };
        self.backend.insert(&rec).await.map_err(KeyError::Backend)?;
        if let Ok(mut c) = self.cache.write() {
            c.insert(d, Entry { record: rec.clone(), persisted: None });
        }
        Ok((key, rec))
    }

    /// Every key, newest first.
    pub fn list(&self) -> Vec<ApiKeyRecord> {
        let mut v: Vec<ApiKeyRecord> =
            self.cache.read().map(|c| c.values().map(|e| e.record.clone()).collect()).unwrap_or_default();
        v.sort_by(|a, b| b.created_at.cmp(&a.created_at).then_with(|| a.id.cmp(&b.id)));
        v
    }

    /// Revokes `id` (idempotent); `None` when unknown.
    pub async fn revoke(&self, id: &str) -> Result<Option<ApiKeyRecord>, KeyError> {
        let at = now_ms();
        let known = self.cache.read().map(|c| c.values().any(|e| e.record.id == id)).unwrap_or(false);
        let hit = self.backend.revoke(id, at).await.map_err(KeyError::Backend)?;
        if !hit && !known {
            return Ok(None);
        }
        let mut out = None;
        if let Ok(mut c) = self.cache.write() {
            if let Some(e) = c.values_mut().find(|e| e.record.id == id) {
                e.record.revoked_at.get_or_insert(at);
                out = Some(e.record.clone());
            }
        }
        if out.is_none() {
            // Minted by another worker since our last refresh.
            self.refresh().await.map_err(KeyError::Backend)?;
            out = self.list().into_iter().find(|r| r.id == id);
        }
        Ok(out)
    }

    /// Reloads from the backend (keeps newer in-memory `last_used_at`).
    pub async fn refresh(&self) -> Result<(), String> {
        let rows = self.backend.load().await?;
        let mut c = self.cache.write().map_err(|e| e.to_string())?;
        let mut next = HashMap::with_capacity(rows.len());
        for r in rows {
            let Some(d) = r.digest_bytes() else {
                tracing::warn!(id = %r.id, "api key record with a malformed digest; skipped");
                continue;
            };
            let persisted = r.last_used_at;
            let mut record = r;
            if let Some(old) = c.get(&d) {
                if old.record.last_used_at > record.last_used_at {
                    record.last_used_at = old.record.last_used_at;
                }
            }
            next.insert(d, Entry { record, persisted });
        }
        *c = next;
        Ok(())
    }

    /// Writes pending `last_used_at` changes; returns how many.
    pub async fn flush(&self) -> Result<usize, String> {
        let pending: Vec<(String, i64)> = self
            .cache
            .read()
            .map_err(|e| e.to_string())?
            .values()
            .filter_map(|e| match e.record.last_used_at {
                Some(t) if Some(t) != e.persisted => Some((e.record.id.clone(), t)),
                _ => None,
            })
            .collect();
        if pending.is_empty() {
            return Ok(0);
        }
        self.backend.touch(&pending).await?;
        if let Ok(mut c) = self.cache.write() {
            for e in c.values_mut() {
                if let Some((_, t)) = pending.iter().find(|(id, _)| *id == e.record.id) {
                    e.persisted = Some(e.persisted.map_or(*t, |p| p.max(*t)));
                }
            }
        }
        Ok(pending.len())
    }

    /// Refreshes every `refresh` (when set) and flushes `last_used_at`
    /// every `flush`, until the handle is aborted.
    pub fn spawn_maintenance(self: &Arc<Self>, refresh: Option<Duration>, flush: Duration) -> tokio::task::JoinHandle<()> {
        let s = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(flush.min(refresh.unwrap_or(flush)).max(Duration::from_millis(100)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            let (mut last_refresh, mut last_flush) = (tokio::time::Instant::now(), tokio::time::Instant::now());
            loop {
                tick.tick().await;
                if last_flush.elapsed() >= flush {
                    last_flush = tokio::time::Instant::now();
                    if let Err(e) = s.flush().await {
                        tracing::warn!(error = %e, "api keys: writing last_used_at failed");
                    }
                }
                if let Some(r) = refresh {
                    if last_refresh.elapsed() >= r {
                        last_refresh = tokio::time::Instant::now();
                        if let Err(e) = s.refresh().await {
                            tracing::warn!(error = %e, "api keys: refresh failed");
                        }
                    }
                }
            }
        })
    }
}

impl KeyCheck for KeyStore {
    fn check(&self, key: &str) -> Option<KeyId> {
        if !key.starts_with(KEY_PREFIX) {
            return None;
        }
        let d = digest(key);
        let now = now_ms();
        let touch_ms = self.touch_every.as_millis() as i64;
        let needs_touch = {
            let c = self.cache.read().ok()?;
            let e = c.get(&d)?;
            if e.record.revoked_at.is_some() {
                return None;
            }
            e.record.last_used_at.is_none_or(|t| now - t >= touch_ms)
        };
        if needs_touch {
            if let Ok(mut c) = self.cache.write() {
                if let Some(e) = c.get_mut(&d) {
                    e.record.last_used_at = Some(now);
                }
            }
        }
        Some(KeyId(key_id_of(&d)))
    }
}

// ---------------------------------------------------------------- admin API

#[derive(Clone)]
struct AdminState {
    store: Arc<KeyStore>,
    admin: Arc<AdminToken>,
}

fn err(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let mut r = (status, Json(json!({"error": {"kind": kind, "message": message.into()}}))).into_response();
    r.headers_mut().insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-store"));
    r
}

fn ok(status: StatusCode, body: Value) -> Response {
    let mut r = (status, Json(body)).into_response();
    r.headers_mut().insert(header::CACHE_CONTROL, header::HeaderValue::from_static("no-store"));
    r
}

/// The 401 to send when the admin token is missing or wrong.
fn refused(st: &AdminState, headers: &HeaderMap) -> Option<Response> {
    (!st.admin.check_headers(headers))
        .then(|| err(StatusCode::UNAUTHORIZED, "unauthorized", "admin token required (Authorization: Bearer <admin token>)"))
}

#[derive(Deserialize)]
struct MintBody {
    name: String,
}

async fn mint_handler(State(st): State<AdminState>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Some(r) = refused(&st, &headers) {
        return r;
    }
    let b: MintBody = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request", format!("expected {{\"name\": \"...\"}}: {e}")),
    };
    match st.store.mint(&b.name).await {
        Ok((key, rec)) => {
            tracing::info!(id = %rec.id, name = %rec.name, "api key minted");
            ok(StatusCode::CREATED, json!({"api_key": key, "key": rec.view()}))
        }
        Err(KeyError::Invalid(m)) => err(StatusCode::BAD_REQUEST, "invalid_request", m),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

async fn list_handler(State(st): State<AdminState>, headers: HeaderMap) -> Response {
    if let Some(r) = refused(&st, &headers) {
        return r;
    }
    let keys: Vec<Value> = st.store.list().iter().map(ApiKeyRecord::view).collect();
    ok(StatusCode::OK, json!({"keys": keys, "backend": st.store.backend_kind()}))
}

async fn revoke_handler(State(st): State<AdminState>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Some(r) = refused(&st, &headers) {
        return r;
    }
    match st.store.revoke(&id).await {
        Ok(Some(rec)) => {
            tracing::info!(id = %rec.id, "api key revoked");
            ok(StatusCode::OK, json!({"key": rec.view()}))
        }
        Ok(None) => err(StatusCode::NOT_FOUND, "not_found", format!("no key `{id}`")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

/// `POST /fv/v1/admin/keys` `{name}` → 201 `{api_key, key}` (the only time
/// the key is shown); `GET /fv/v1/admin/keys` → `{keys, backend}`;
/// `DELETE /fv/v1/admin/keys/{id}` → `{key}` (revoked) or 404. Every call
/// needs the admin token.
pub fn admin_routes<S: Clone + Send + Sync + 'static>(store: Arc<KeyStore>, admin: Arc<AdminToken>) -> Router<S> {
    Router::new()
        .route("/fv/v1/admin/keys", get(list_handler).post(mint_handler))
        .route("/fv/v1/admin/keys/{id}", delete(revoke_handler))
        .with_state(AdminState { store, admin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[test]
    fn admin_token_generation_and_check() {
        let (t, plain) = AdminToken::generate();
        assert!(plain.starts_with(ADMIN_PREFIX));
        // 32 bytes base64url without padding = 43 chars.
        assert_eq!(plain.len(), ADMIN_PREFIX.len() + 43);
        assert!(t.check(&plain));
        assert!(!t.check("fvadm_wrong"));
        assert!(!t.check(""));
        assert!(!format!("{t:?}").contains(&plain));
        let (_, other) = AdminToken::generate();
        assert_ne!(plain, other);
        let mut h = HeaderMap::new();
        h.insert(header::AUTHORIZATION, format!("Bearer {plain}").parse().unwrap());
        assert!(t.check_headers(&h));
        h.insert(header::AUTHORIZATION, "Bearer nope".parse().unwrap());
        assert!(!t.check_headers(&h));
    }

    #[tokio::test]
    async fn mint_check_revoke_memory() {
        let s = KeyStore::memory().await;
        let (k, rec) = s.mint("  laptop ").await.unwrap();
        assert!(k.starts_with(KEY_PREFIX));
        assert_eq!(rec.name, "laptop");
        assert!(!rec.digest.contains(&k) && !rec.prefix.contains(&k[..]));
        assert_eq!(s.check(&k).unwrap().0, rec.id);
        assert!(s.check("fv_nope").is_none());
        assert!(s.check("not-a-key").is_none());
        assert!(s.list()[0].last_used_at.is_some(), "check touches last_used_at");
        assert!(matches!(s.mint(" ").await, Err(KeyError::Invalid(_))));
        assert!(matches!(s.mint(&"x".repeat(65)).await, Err(KeyError::Invalid(_))));
        let r = s.revoke(&rec.id).await.unwrap().unwrap();
        assert!(r.revoked_at.is_some());
        assert!(s.check(&k).is_none());
        assert!(s.revoke("key_000000000000").await.unwrap().is_none());
        // Idempotent.
        assert!(s.revoke(&rec.id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn file_backend_persists_digests_only() {
        let dir = std::env::temp_dir().join(format!("fv-keys-{}", crate::random_token()));
        let path = dir.join("api_keys.json");
        let s = KeyStore::open(Arc::new(FileKeyBackend::new(&path))).await.unwrap();
        let (k1, r1) = s.mint("a").await.unwrap();
        let (k2, r2) = s.mint("b").await.unwrap();
        assert!(s.check(&k1).is_some());
        assert_eq!(s.flush().await.unwrap(), 1);
        s.revoke(&r2.id).await.unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(&k1) && !text.contains(&k2), "plaintext keys must not be stored");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // A new process sees the same keys.
        let s2 = KeyStore::open(Arc::new(FileKeyBackend::new(&path))).await.unwrap();
        assert_eq!(s2.check(&k1).unwrap().0, r1.id);
        assert!(s2.check(&k2).is_none(), "revocation persisted");
        let l = s2.list();
        assert_eq!(l.len(), 2);
        assert!(l.iter().find(|r| r.id == r1.id).unwrap().last_used_at.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn d1_backend_shares_keys_between_workers() {
        let m = crate::d1::mock::MockD1::new();
        let client = || D1Client::new(Arc::new(m.clone())).with_retry(crate::d1::RetryPolicy::immediate(1));
        let a = KeyStore::open(Arc::new(D1KeyBackend::open(client()).await.unwrap())).await.unwrap();
        let b = KeyStore::open(Arc::new(D1KeyBackend::open(client()).await.unwrap())).await.unwrap();
        let (k, rec) = a.mint("ci").await.unwrap();
        assert!(b.check(&k).is_none(), "not refreshed yet");
        b.refresh().await.unwrap();
        assert_eq!(b.check(&k).unwrap().0, rec.id);
        assert_eq!(b.flush().await.unwrap(), 1);
        // Revoked on worker b, seen by a after a refresh.
        b.revoke(&rec.id).await.unwrap().unwrap();
        a.refresh().await.unwrap();
        assert!(a.check(&k).is_none());
        let row = m.sql("SELECT * FROM api_keys", &[]).unwrap();
        assert_eq!(row.len(), 1);
        assert!(row[0]["last_used_at"].as_f64().is_some() || row[0]["last_used_at"].as_i64().is_some());
        assert!(!serde_json::to_string(&row).unwrap().contains(&k));
    }

    #[tokio::test]
    async fn last_used_is_rate_limited() {
        let s = KeyStore::memory().await.with_touch_every(Duration::from_secs(3600));
        let (k, _) = s.mint("x").await.unwrap();
        s.check(&k);
        let t1 = s.list()[0].last_used_at;
        s.check(&k);
        assert_eq!(s.list()[0].last_used_at, t1);
        assert_eq!(s.flush().await.unwrap(), 1);
        assert_eq!(s.flush().await.unwrap(), 0);
    }

    async fn call(r: &Router, method: &str, uri: &str, auth: Option<&str>, body: Option<Value>) -> (StatusCode, Value) {
        let mut b = Request::builder().method(method).uri(uri);
        if let Some(a) = auth {
            b = b.header("authorization", format!("Bearer {a}"));
        }
        let req = b.body(body.map_or_else(Body::empty, |v| Body::from(v.to_string()))).unwrap();
        let resp = r.clone().oneshot(req).await.unwrap();
        let s = resp.status();
        assert_eq!(resp.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        (s, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn admin_api_requires_the_token() {
        let store = KeyStore::memory().await;
        let (admin, plain) = AdminToken::generate();
        let r: Router = admin_routes(store.clone(), Arc::new(admin));
        for (m, u) in [("GET", "/fv/v1/admin/keys"), ("POST", "/fv/v1/admin/keys"), ("DELETE", "/fv/v1/admin/keys/key_x")] {
            assert_eq!(call(&r, m, u, None, None).await.0, StatusCode::UNAUTHORIZED);
            assert_eq!(call(&r, m, u, Some("fvadm_bad"), None).await.0, StatusCode::UNAUTHORIZED);
        }
        let (s, v) = call(&r, "POST", "/fv/v1/admin/keys", Some(&plain), Some(json!({"name": "demo"}))).await;
        assert_eq!(s, StatusCode::CREATED, "{v}");
        let key = v["api_key"].as_str().unwrap().to_owned();
        let id = v["key"]["id"].as_str().unwrap().to_owned();
        assert!(v["key"].get("digest").is_none());
        assert!(store.check(&key).is_some());
        let (s, v) = call(&r, "GET", "/fv/v1/admin/keys", Some(&plain), None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["keys"][0]["id"], id.as_str());
        assert!(!v.to_string().contains(&key), "list never shows the key");
        let (s, _) = call(&r, "POST", "/fv/v1/admin/keys", Some(&plain), Some(json!({"name": ""}))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = call(&r, "POST", "/fv/v1/admin/keys", Some(&plain), Some(json!({}))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, v) = call(&r, "DELETE", &format!("/fv/v1/admin/keys/{id}"), Some(&plain), None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["key"]["revoked"], true);
        assert!(store.check(&key).is_none());
        let (s, _) = call(&r, "DELETE", "/fv/v1/admin/keys/key_nope", Some(&plain), None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
}
