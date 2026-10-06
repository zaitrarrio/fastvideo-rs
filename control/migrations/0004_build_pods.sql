-- Build pods managed by fv-control (docs/dev/build-pods-fv-control.md): CPU
-- pods running scripts/dev/build-pod-server.py, created, started, stopped and
-- deleted only by the controller. A row stays after its pod is deleted (the
-- ledger keeps its costs under owner build-pod:<name>).
CREATE TABLE IF NOT EXISTS build_pods (
  id TEXT PRIMARY KEY NOT NULL,           -- bp_<hex>
  name TEXT NOT NULL,                     -- fv-build-<region>-<n> (the Runpod pod name)
  pod_id TEXT UNIQUE,                     -- Runpod id (null until the create call answered)
  provider TEXT NOT NULL DEFAULT 'runpod',
  purpose TEXT NOT NULL DEFAULT 'shared', -- shared (handed out by `up`) | test
  state TEXT NOT NULL,                    -- creating | running | stopping | stopped | deleted | failed
  dc TEXT,
  region TEXT,                            -- eu | us | ca | ap (runner label fv-build-<region>)
  flavor TEXT,
  vcpu INTEGER,
  disk_gb INTEGER,
  cost_per_hr REAL,
  volume_id TEXT,
  image TEXT NOT NULL,
  server_ref TEXT NOT NULL,
  server_sha TEXT NOT NULL,               -- sha256[:12] of the server source (what /v1/status reports)
  token_sealed TEXT NOT NULL,             -- AES-GCM under CONTROL_KEK, AAD build-pod:<id>
  token_sha TEXT NOT NULL,                -- sha256 of the token (FV_BUILD_TOKEN_SHA256)
  idle_min REAL NOT NULL,
  max_h REAL NOT NULL,
  max_grace_min REAL NOT NULL,
  labels TEXT,                            -- runner labels, comma list
  runner_name TEXT,
  runner_id INTEGER,
  runner_state TEXT,                      -- none | registering | running | unsupported | failed | removed
  runner_error TEXT,
  runner_at INTEGER,                      -- last registration attempt
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  started_at INTEGER,
  stopped_at INTEGER,
  deleted_at INTEGER,
  last_error TEXT
);
CREATE INDEX IF NOT EXISTS build_pods_state ON build_pods (state, purpose);

-- Short leases (compare-and-set): one `up` at a time, so parallel agents do
-- not create two pods.
CREATE TABLE IF NOT EXISTS locks (
  key TEXT PRIMARY KEY NOT NULL,
  holder TEXT NOT NULL,
  expires_at INTEGER NOT NULL
);
