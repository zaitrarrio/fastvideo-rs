-- Pod logs from boot (src/podlogs.ts, docs/control/README.md §6): the
-- controller copies each controller pod's Runpod container and system log
-- tail into log_lines. The cursor (the newest timestamp stored per stream,
-- and how many lines carried exactly that timestamp) keeps a line from being
-- stored twice; diag is the latest boot diagnosis (pulling, starting,
-- running, error, crashloop, image_error, stuck) as JSON.
CREATE TABLE IF NOT EXISTS pod_log_cursors (
  pod_id TEXT PRIMARY KEY NOT NULL,
  cluster_id TEXT NOT NULL,
  container_ts INTEGER NOT NULL DEFAULT 0,
  container_n INTEGER NOT NULL DEFAULT 0,
  system_ts INTEGER NOT NULL DEFAULT 0,
  system_n INTEGER NOT NULL DEFAULT 0,
  diag TEXT,
  updated_at INTEGER NOT NULL
);

-- Standalone pods (docs/control/standalone-pods.md) are clusters with
-- source = 'standalone' and one pool: listing them is a lookup by source.
CREATE INDEX IF NOT EXISTS clusters_source ON clusters (source);

-- The boot timeline of a controller pod (src/boottime.ts): JSON {t: {milestone: ms}, components: {name: {wall_s, gb, gbps, at}}, load_s, warmup_s, volume}.
ALTER TABLE cluster_pods ADD COLUMN boot TEXT;
