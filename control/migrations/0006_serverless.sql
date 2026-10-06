-- Runpod serverless endpoints fv-control creates (docs/control/serverless.md).
-- fv-control only ever updates, scales or deletes endpoints and templates
-- recorded here; a row stays after its endpoint is deleted (the cost ledger
-- keeps its spend under owner serverless:<name>, pod_id sls:<endpoint id>).
CREATE TABLE IF NOT EXISTS serverless_endpoints (
  id TEXT PRIMARY KEY NOT NULL,           -- se_<hex>
  name TEXT NOT NULL,                     -- the spec's name; the Runpod endpoint is fvc-<name>
  endpoint_id TEXT UNIQUE,                -- Runpod endpoint id (null until the create call answered)
  template_id TEXT,                       -- Runpod template id
  own_template INTEGER NOT NULL DEFAULT 1, -- 1: fv-control created the template (deleted with the endpoint)
  mode TEXT NOT NULL,                     -- queue | lb
  spec TEXT NOT NULL,                     -- JSON EndpointSpec (src/serverless/spec.ts)
  image TEXT,                             -- the resolved image (digest reference)
  status TEXT NOT NULL,                   -- creating | active | scaled-down | deleting | deleted | gone | failed
  deadline INTEGER,                       -- backstop (unix ms): deadline_action at this time
  deadline_action TEXT,                   -- scale0 | delete
  health TEXT,                            -- JSON: Runpod /health (jobs, workers) at health_at
  health_at INTEGER,
  workers INTEGER,                        -- live workers (Runpod's endpoint view) at health_at
  live_dph REAL,                          -- their summed $/hr
  cost_usd REAL NOT NULL DEFAULT 0,       -- billed so far (Runpod /billing/endpoints)
  billed_ms INTEGER NOT NULL DEFAULT 0,   -- worker time billed so far
  billed_at INTEGER,
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  deleted_at INTEGER,
  last_error TEXT
);
-- One live endpoint per name; deleted rows keep theirs.
CREATE UNIQUE INDEX IF NOT EXISTS serverless_endpoints_live_name ON serverless_endpoints (name) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS serverless_endpoints_status ON serverless_endpoints (status, deleted_at);

-- Test invokes and their timings (the recent job stats and cold starts).
CREATE TABLE IF NOT EXISTS serverless_jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint TEXT NOT NULL,                 -- serverless_endpoints.id
  job_id TEXT,                            -- Runpod job id (queue), null for a load-balancer request
  route TEXT NOT NULL,                    -- runsync | run | lb:<METHOD> <path>
  status TEXT,                            -- IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | … | HTTP <code>
  cold INTEGER NOT NULL DEFAULT 0,        -- 1: no worker was up when it was submitted
  submitted_at INTEGER NOT NULL,
  finished_at INTEGER,
  delay_ms INTEGER,                       -- Runpod delayTime (queue wait, cold start included)
  exec_ms INTEGER,                        -- Runpod executionTime
  wall_ms INTEGER,                        -- submit to answer, as fv-control saw it
  worker_id TEXT,
  input TEXT,
  output TEXT,                            -- truncated, scrubbed
  error TEXT,
  actor TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS serverless_jobs_endpoint ON serverless_jobs (endpoint, id);
