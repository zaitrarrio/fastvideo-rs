-- fv-control's own D1 database (docs/control/README.md). Times are unix
-- milliseconds, as in fv-jobs. Never a secret in clear: cluster secrets and
-- secret env values are sealed with CONTROL_KEK (AES-256-GCM), API tokens
-- and ingest tokens are stored as SHA-256 only.

-- A cluster: its definition (spec) and what exists of it on Runpod (state).
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY NOT NULL,           -- c_<hex>
  name TEXT NOT NULL UNIQUE,
  spec TEXT NOT NULL,                     -- JSON ClusterSpec (src/cluster/spec.ts)
  state TEXT NOT NULL DEFAULT '{}',       -- JSON ClusterState: gateway, workers per pool, rolling, retired, images
  status TEXT NOT NULL DEFAULT 'defined', -- defined | starting | running | stopping | stopped | failed
  deadline INTEGER,                       -- backstop: every pod is deleted at this time
  secrets TEXT,                           -- sealed JSON: internal_token, url_signing_key, admin key pair, admin_token, ingest_token
  ingest_hash TEXT,                       -- sha256 of the log ingest token
  source TEXT NOT NULL DEFAULT 'controller', -- controller | import
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  created_by TEXT NOT NULL
);

-- Every pod the controller created or imported (history kept).
CREATE TABLE IF NOT EXISTS cluster_pods (
  pod_id TEXT PRIMARY KEY NOT NULL,
  cluster_id TEXT NOT NULL,
  role TEXT NOT NULL,                     -- gateway | worker
  pool TEXT,
  slot TEXT NOT NULL DEFAULT 'workers',   -- workers | rolling | retired
  image TEXT,
  gpu TEXT,
  dc TEXT,
  cost_per_hr REAL,
  url TEXT,
  env_hash TEXT,                          -- hash of the env last applied (create / PATCH)
  env_applied_at INTEGER,
  created_at INTEGER NOT NULL,
  ready_at INTEGER,
  deleted_at INTEGER,
  status TEXT NOT NULL DEFAULT 'creating' -- creating | ready | draining | stopped | deleted | gone
);
CREATE INDEX IF NOT EXISTS cluster_pods_cluster ON cluster_pods (cluster_id, deleted_at);

-- Environment variables at three levels: account (scope_id ''), cluster, pod.
CREATE TABLE IF NOT EXISTS env_vars (
  scope TEXT NOT NULL,                    -- account | cluster | pod
  scope_id TEXT NOT NULL DEFAULT '',
  key TEXT NOT NULL,
  value TEXT NOT NULL,                    -- clear, or sealed (secret = 1)
  secret INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER NOT NULL,
  updated_by TEXT NOT NULL,
  PRIMARY KEY (scope, scope_id, key)
);

-- Every mutating action: who, what, when, before/after (never a secret value).
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  at INTEGER NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT,
  before TEXT,
  after TEXT,
  ok INTEGER NOT NULL DEFAULT 1,
  detail TEXT,
  ip TEXT
);
CREATE INDEX IF NOT EXISTS audit_at ON audit (at);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY NOT NULL,
  actor TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  ip TEXT,
  ua TEXT
);

-- Fixed-window counters (login attempts, ingest, API).
CREATE TABLE IF NOT EXISTS rate_limits (
  key TEXT NOT NULL,
  window INTEGER NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY (key, window)
);

CREATE TABLE IF NOT EXISTS api_tokens (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL,                    -- read | admin
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  expires_at INTEGER,
  last_used_at INTEGER,
  revoked_at INTEGER
);

-- The latest view of every pod in the account (cron, every minute).
CREATE TABLE IF NOT EXISTS pods (
  pod_id TEXT PRIMARY KEY NOT NULL,
  name TEXT,
  owner TEXT NOT NULL,                    -- cluster:<name> | external:<prefix> | external
  cluster_id TEXT,
  desired_status TEXT,
  cost_per_hr REAL,
  gpu TEXT,
  gpu_count INTEGER,
  dc TEXT,
  image TEXT,
  uptime_s INTEGER,
  gpu_util REAL,
  gpu_mem REAL,
  cpu REAL,
  mem REAL,
  jobs_running INTEGER,
  jobs_queued INTEGER,
  health TEXT,                            -- ready | loading | down | unknown
  build_sha TEXT,
  idle_since INTEGER,                     -- GPU < threshold and no running jobs since
  first_seen INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  gone_at INTEGER
);

-- Accrued cost per day and pod (UTC day), from the per-minute snapshots.
CREATE TABLE IF NOT EXISTS cost_daily (
  day TEXT NOT NULL,                      -- YYYY-MM-DD
  pod_id TEXT NOT NULL,
  cluster_id TEXT,
  owner TEXT NOT NULL,
  usd REAL NOT NULL DEFAULT 0,
  minutes INTEGER NOT NULL DEFAULT 0,
  idle_minutes INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, pod_id)
);
CREATE INDEX IF NOT EXISTS cost_daily_cluster ON cost_daily (cluster_id, day);

CREATE TABLE IF NOT EXISTS balance_samples (
  at INTEGER PRIMARY KEY NOT NULL,
  balance REAL NOT NULL,
  spend_per_hr REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT NOT NULL,                      -- <policy>:<target>
  kind TEXT NOT NULL,                     -- pod_idle | cluster_dph | daily_spend | balance_margin | balance_floor | deadline | pod_down
  severity TEXT NOT NULL,                 -- info | warn | critical
  target TEXT,
  message TEXT NOT NULL,
  opened_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  resolved_at INTEGER,
  action TEXT                             -- what an auto-action did, if any
);
CREATE INDEX IF NOT EXISTS alerts_open ON alerts (resolved_at, key);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY NOT NULL,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  updated_by TEXT NOT NULL
);

-- Operations on a cluster (run step by step by its ClusterOps Durable Object).
CREATE TABLE IF NOT EXISTS operations (
  id TEXT PRIMARY KEY NOT NULL,
  cluster_id TEXT NOT NULL,
  kind TEXT NOT NULL,                     -- up | down | extend | scale | roll | restart | gateway-start | gateway-stop
  status TEXT NOT NULL,                   -- queued | running | done | failed | cancelled
  params TEXT,
  log TEXT NOT NULL DEFAULT '[]',         -- JSON [{at, msg}]
  error TEXT,
  actor TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS operations_cluster ON operations (cluster_id, created_at);

-- The recent log tail (24 h; older lines are in R2 only).
CREATE TABLE IF NOT EXISTS log_lines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cluster_id TEXT NOT NULL,
  pod_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  level TEXT NOT NULL,
  target TEXT,
  msg TEXT NOT NULL,
  fields TEXT
);
CREATE INDEX IF NOT EXISTS log_lines_pod ON log_lines (pod_id, ts);
CREATE INDEX IF NOT EXISTS log_lines_ts ON log_lines (ts);

-- Per-minute pod samples when the Analytics Engine binding is absent
-- (local dev, tests). With METRICS bound the samples go to Analytics Engine
-- only (docs/control/README.md "Observability"). 24 h retention.
CREATE TABLE IF NOT EXISTS pod_samples (
  at INTEGER NOT NULL,
  pod_id TEXT NOT NULL,
  owner TEXT,
  cluster_id TEXT,
  cost_per_hr REAL,
  gpu_util REAL,
  gpu_mem REAL,
  cpu REAL,
  mem REAL,
  jobs_running INTEGER,
  jobs_queued INTEGER,
  idle INTEGER,
  PRIMARY KEY (pod_id, at)
);
CREATE INDEX IF NOT EXISTS pod_samples_at ON pod_samples (at);
