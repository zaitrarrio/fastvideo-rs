-- fv-app key provisioning: consumer key requests (moved from app/migrations so
-- fv-control, which owns this D1's schema and applies its migrations on deploy,
-- creates it; fv-app writes rows, the console reads and fulfills them).
CREATE TABLE IF NOT EXISTS key_requests (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  use_case TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending', -- pending | fulfilled | rejected
  key_hint TEXT,                          -- last 4 of the minted key, never the key
  created_at INTEGER NOT NULL,
  fulfilled_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_key_requests_status ON key_requests(status, created_at);
