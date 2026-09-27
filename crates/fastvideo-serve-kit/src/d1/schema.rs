//! D1 schema and migrations for the job store.
//!
//! Every migration is a list of idempotent statements applied as one D1
//! batch (one transaction) together with its `schema_migrations` row, so
//! workers starting at the same time can all run [`migrate`] safely.
//!
//! Table `jobs` (one row per job):
//!
//! | column | type | meaning |
//! |---|---|---|
//! | `id` | TEXT PK | `JobId` (uuid, lowercase hyphenated) |
//! | `protocol` | TEXT | `ProtocolId::as_str` (`minimax_v2`, `openai_videos`, ...) |
//! | `external_id` | TEXT | wire id; `UNIQUE(protocol, external_id)` |
//! | `owner` | TEXT NULL | `KeyId` (hash prefix, never a key) |
//! | `status` | TEXT | `queued\|running\|succeeded\|failed\|cancelled` |
//! | `model` | TEXT | model name as sent (`Job::requested_model`) |
//! | `resolved_model` | TEXT | engine model id |
//! | `task` | TEXT | `t2v`, `i2v`, ... |
//! | `progress` | REAL | `0..=1` |
//! | `created_at`, `updated_at`, `completed_at`, `expires_at` | INTEGER | unix ms |
//! | `worker` | TEXT NULL | worker that owns the in-memory copy |
//! | `version` | INTEGER | bumped on every write (optimistic updates) |
//! | `job` | TEXT | the full `Job` as JSON (the source for reads) |
//!
//! Indexes: `(owner, protocol, created_at)`, `(status, created_at)`,
//! `(protocol, created_at)` for the MiniMax / FastVideo list endpoints;
//! `(expires_at)` for the sweep; `(worker, status)` for restart recovery and
//! the heartbeat.
//!
//! Table `api_keys` (migration 2; `crate::keys::D1KeyBackend`): `id` (the
//! owner `KeyId`, `key_<digest prefix>`), `name`, `prefix` (display hint),
//! `digest` (SHA-256 hex, UNIQUE), `created_at`, `last_used_at`,
//! `revoked_at` (unix ms).

use serde_json::json;

use super::client::{D1Client, D1Error, Stmt};

/// `(version, statements)`, in order. Never edit a shipped migration; append.
pub const MIGRATIONS: &[(u32, &[&str])] = &[(
    1,
    &[
        "CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY NOT NULL,
            protocol TEXT NOT NULL,
            external_id TEXT NOT NULL,
            owner TEXT,
            status TEXT NOT NULL,
            model TEXT NOT NULL,
            resolved_model TEXT NOT NULL,
            task TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            expires_at INTEGER NOT NULL,
            worker TEXT,
            version INTEGER NOT NULL DEFAULT 0,
            job TEXT NOT NULL,
            UNIQUE (protocol, external_id)
        )",
        "CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs (owner, protocol, created_at)",
        "CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at)",
        "CREATE INDEX IF NOT EXISTS jobs_protocol_created ON jobs (protocol, created_at)",
        "CREATE INDEX IF NOT EXISTS jobs_expires ON jobs (expires_at)",
        "CREATE INDEX IF NOT EXISTS jobs_worker_status ON jobs (worker, status)",
    ],
), (
    2,
    &[
        // Dynamic API keys (`crate::keys`): only the SHA-256 digest of a key
        // is stored, never the key.
        "CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            prefix TEXT NOT NULL,
            digest TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL,
            last_used_at INTEGER,
            revoked_at INTEGER
        )",
    ],
)];

/// The newest schema version this build knows.
pub fn latest() -> u32 {
    MIGRATIONS.last().map_or(0, |m| m.0)
}

/// Applies pending migrations; returns the schema version afterwards.
pub async fn migrate(db: &D1Client) -> Result<u32, D1Error> {
    db.query(Stmt::raw(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY NOT NULL, applied_at INTEGER NOT NULL)",
    ))
    .await?;
    let cur = current(db).await?;
    for (v, stmts) in MIGRATIONS.iter().filter(|m| m.0 > cur) {
        let mut batch: Vec<Stmt> = stmts.iter().map(|s| Stmt::raw(*s)).collect();
        batch.push(Stmt::new(
            "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            vec![json!(v), json!(now_ms())],
        ));
        db.batch(batch).await?;
        tracing::info!(version = v, "D1 job store: migration applied");
    }
    current(db).await
}

async fn current(db: &D1Client) -> Result<u32, D1Error> {
    let r = db
        .query(Stmt::raw("SELECT MAX(version) AS v FROM schema_migrations"))
        .await?;
    Ok(r.rows
        .first()
        .and_then(|row| row.get("v"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u32)
}

fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}
