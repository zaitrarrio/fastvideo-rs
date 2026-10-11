-- Serverless endpoints with weights_source r2 (docs/control/serverless.md §1b, docs/ops/r2-weights.md §9).
-- Each such endpoint has its own weights token (FV_WEIGHTS_TOKEN in its template env): it asks
-- POST /ingest/v1/weights-plan for the presigned URLs of that endpoint's trees only, and reports its boot
-- (POST /ingest/v1/sls-report). Only the SHA-256 is stored; a new one replaces it at every template update
-- (the previous one stays valid for a grace period while Runpod rolls the workers); a delete revokes both.
ALTER TABLE serverless_endpoints ADD COLUMN weights_token_hash TEXT;
ALTER TABLE serverless_endpoints ADD COLUMN weights_token_prev_hash TEXT;
ALTER TABLE serverless_endpoints ADD COLUMN weights_token_rotated_at INTEGER;
CREATE INDEX IF NOT EXISTS serverless_endpoints_weights_token ON serverless_endpoints (weights_token_hash);
CREATE INDEX IF NOT EXISTS serverless_endpoints_weights_token_prev ON serverless_endpoints (weights_token_prev_hash);

-- One row per worker boot of an r2 endpoint (the cold start's breakdown): when the container started, whether the
-- trees were already on its disk (cache hit), the fetch (bytes, seconds, MB/s), when fv-serve started.
CREATE TABLE IF NOT EXISTS serverless_boots (
  endpoint TEXT NOT NULL,                 -- serverless_endpoints.id
  worker_id TEXT NOT NULL,                -- RUNPOD_POD_ID
  boot TEXT NOT NULL,                     -- <worker>@<container start, unix s>
  boot_at INTEGER NOT NULL,               -- first report (ms)
  cache TEXT,                             -- hit | miss | null (not reported yet)
  missing TEXT,                           -- the trees a miss fetched
  fetch_start INTEGER,
  fetch_end INTEGER,
  fetch_bytes REAL,
  fetch_s REAL,
  fetch_mbps REAL,
  serve_at INTEGER,                       -- fv-serve started (weights in place)
  error TEXT,
  last_detail TEXT,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (endpoint, boot)
);
CREATE INDEX IF NOT EXISTS serverless_boots_recent ON serverless_boots (endpoint, boot_at);
