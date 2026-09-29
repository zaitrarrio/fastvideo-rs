-- Release channels and the deployment registry (docs/serve/releases.md),
-- in the fv-jobs D1 database next to serve-kit's `jobs` and the gateway's
-- gw_* tables. Idempotent (IF NOT EXISTS): scripts/serve/lib/registry.sh
-- applies it before its first write. Times are unix milliseconds, as in
-- every other table. Append columns with a new ALTER in a new file; never
-- edit a shipped statement.

-- One row per promotion of an image set to a channel. The current release
-- of a channel is its row with the highest id.
CREATE TABLE IF NOT EXISTS releases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  channel TEXT NOT NULL,                  -- stable | latest | …
  git_sha TEXT NOT NULL,                  -- full sha when known, else the short one
  digests TEXT NOT NULL,                  -- JSON {"debug": "ghcr.io/…@sha256:…", "h3-turbo": "…", …}
  action TEXT NOT NULL,                   -- build | promote | rollback | bootstrap
  promoted_at INTEGER NOT NULL,
  promoted_by TEXT NOT NULL,              -- ci:<workflow>#<run> | user:<name> | agent:<name>
  notes TEXT,
  run_url TEXT,                           -- the workflow run, when CI did it
  templates_updated INTEGER NOT NULL DEFAULT 0,
  source_release INTEGER,                 -- rollback: the release it re-promoted
  rolled_back_at INTEGER                  -- set when a rollback moved the channel away from this sha
);
CREATE INDEX IF NOT EXISTS releases_channel ON releases (channel, id);
CREATE INDEX IF NOT EXISTS releases_sha ON releases (git_sha);

-- One row per Runpod resource a deploy script created (pods, endpoints and
-- gateway pods). `release.sh reconcile` matches rows against the account.
CREATE TABLE IF NOT EXISTS deployments (
  id TEXT PRIMARY KEY NOT NULL,           -- <kind>:<runpod id>
  kind TEXT NOT NULL,                     -- pod | endpoint | gateway
  runpod_id TEXT NOT NULL,
  name TEXT,
  pool TEXT,                              -- gateway pool (h3-turbo, wan, …) or empty
  variant TEXT,                           -- image variant (h3-turbo, …, gateway, debug)
  image TEXT,                             -- the reference it was created with
  digest TEXT,                            -- sha256:…
  git_sha TEXT,
  channel TEXT,                           -- the channel it follows, when deployed from one
  region TEXT,
  dc TEXT,
  gpu TEXT,
  cost_per_hr REAL,
  created_at INTEGER NOT NULL,
  ready_at INTEGER,
  deleted_at INTEGER,
  updated_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,               -- <who>/<script>: ci:…, user:…, agent:…
  status TEXT NOT NULL,                   -- creating | ready | draining | deleted | gone | failed
  meta TEXT                               -- JSON: template id, URL, cluster, notes
);
CREATE INDEX IF NOT EXISTS deployments_status ON deployments (status, created_at);
CREATE INDEX IF NOT EXISTS deployments_pool ON deployments (pool, status);
