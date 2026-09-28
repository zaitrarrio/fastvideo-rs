//! The gateway's D1 tables (docs/serve/gateway.md §3, §5; next to
//! serve-kit's `jobs` table). All are `IF NOT EXISTS`, so every gateway
//! replica and worker can apply them at start.
//!
//! - `gw_dispatch`: one row per dispatched job: pool, kind, target (pod URL
//!   or endpoint id), ref (Runpod job id or worker id), attempt, state
//!   (`active` → `done` | `lost`), the input URLs (for a re-dispatch), and
//!   the run/queue durations once finished (metrics).
//! - `gw_sessions`: stream and peer-session leases (session id → pool,
//!   worker URL or Runpod job, owner, lease key).
//! - `gw_workers`: pod workers that registered themselves (pool, id, URL,
//!   state, load, heartbeat).

use fastvideo_serve_kit::d1::{D1Client, D1Error, Stmt};

pub const TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS gw_dispatch (job_id TEXT PRIMARY KEY, pool TEXT NOT NULL, kind TEXT NOT NULL, \
     target TEXT, ref TEXT, attempt INTEGER NOT NULL DEFAULT 1, state TEXT NOT NULL, inputs TEXT, \
     created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, finished_at INTEGER, run_s REAL, wait_s REAL)",
    "CREATE INDEX IF NOT EXISTS gw_dispatch_pool_state ON gw_dispatch (pool, state, finished_at)",
    "CREATE TABLE IF NOT EXISTS gw_sessions (id TEXT PRIMARY KEY, pool TEXT NOT NULL, kind TEXT NOT NULL, \
     target TEXT, ref TEXT, owner TEXT, lease_key TEXT, state TEXT NOT NULL, body TEXT, \
     created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS gw_sessions_lease ON gw_sessions (lease_key, state)",
    "CREATE INDEX IF NOT EXISTS gw_sessions_pool ON gw_sessions (pool, state)",
    "CREATE TABLE IF NOT EXISTS gw_workers (pool TEXT NOT NULL, worker_id TEXT NOT NULL, url TEXT NOT NULL, \
     state TEXT NOT NULL, running INTEGER NOT NULL DEFAULT 0, sessions INTEGER NOT NULL DEFAULT 0, \
     updated_at INTEGER NOT NULL, PRIMARY KEY (pool, worker_id))",
];

/// Creates the tables (idempotent).
pub async fn migrate(db: &D1Client) -> Result<(), D1Error> {
    db.batch(TABLES.iter().map(|s| Stmt::raw(*s)).collect()).await.map(|_| ())
}

/// Milliseconds since the epoch (the unit of every time column).
pub fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}
