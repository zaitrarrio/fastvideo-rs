-- The log explorer (src/logquery.ts, docs/control/README.md "Logs"): keyset
-- pages of log_lines by cluster and by level, audit rows by target, and the
-- operations that overlap a time range. log_lines already has (pod_id, ts)
-- and (ts); every index carries the rowid, so ORDER BY ts, id needs no sort.
CREATE INDEX IF NOT EXISTS log_lines_cluster_ts ON log_lines (cluster_id, ts);
CREATE INDEX IF NOT EXISTS log_lines_level_ts ON log_lines (level, ts);
CREATE INDEX IF NOT EXISTS audit_target_at ON audit (target, at);
CREATE INDEX IF NOT EXISTS operations_updated ON operations (updated_at);
