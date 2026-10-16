-- One-click deployments: saved presets and their pending requests.
-- A preset names model families, GPU specs, storage, instance counts, a max
-- wait and a max spend. A request polls providers until a match deploys or
-- the wait expires.

CREATE TABLE IF NOT EXISTS deploy_presets (
  id TEXT PRIMARY KEY NOT NULL,           -- dp_<hex>
  name TEXT NOT NULL,
  families TEXT NOT NULL,                 -- JSON: ["h3","wan"]
  gpu_specs TEXT NOT NULL,                -- JSON: [{vram_gb, arch, model}]
  gpu_count INTEGER NOT NULL DEFAULT 1,
  storage_gb INTEGER NOT NULL,            -- validated >= 1.1x family weights
  instance_kind TEXT NOT NULL,            -- standalone | cluster | serverless
  instance_count INTEGER NOT NULL DEFAULT 1,
  max_wait_s INTEGER NOT NULL DEFAULT 3600,
  max_spend_usd REAL,                     -- null: no cap
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  deleted_at INTEGER
);
CREATE INDEX IF NOT EXISTS deploy_presets_live ON deploy_presets (deleted_at);

CREATE TABLE IF NOT EXISTS deploy_requests (
  id TEXT PRIMARY KEY NOT NULL,           -- dr_<hex>
  preset_id TEXT NOT NULL REFERENCES deploy_presets(id),
  status TEXT NOT NULL,                   -- pending | matched | deploying | done | expired | cancelled | failed
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,            -- created_at + max_wait_s
  matched_at INTEGER,
  match_json TEXT,                        -- the provider offer that matched
  result_json TEXT,                       -- what was deployed (pod/cluster/endpoint ids)
  last_error TEXT,
  checks INTEGER NOT NULL DEFAULT 0,      -- provider polls so far
  created_by TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS deploy_requests_status ON deploy_requests (status, expires_at);

-- Runpod network volumes fv-control manages.
CREATE TABLE IF NOT EXISTS volumes (
  id TEXT PRIMARY KEY NOT NULL,           -- Runpod volume id
  name TEXT NOT NULL,
  provider TEXT NOT NULL DEFAULT 'runpod',
  size_gb INTEGER NOT NULL,
  region TEXT NOT NULL,                   -- Runpod dataCenterId
  status TEXT NOT NULL,                   -- active | deleted
  families TEXT,                          -- JSON: model families whose weights live here
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  deleted_at INTEGER
);
CREATE INDEX IF NOT EXISTS volumes_status ON volumes (status);

-- Volume cost samples (unit price per GB-month and the month total).
CREATE TABLE IF NOT EXISTS volume_costs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  volume_id TEXT NOT NULL REFERENCES volumes(id),
  usd_per_gb_mo REAL NOT NULL,
  size_gb INTEGER NOT NULL,
  month_usd REAL NOT NULL,
  at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS volume_costs_vol ON volume_costs (volume_id, at);
