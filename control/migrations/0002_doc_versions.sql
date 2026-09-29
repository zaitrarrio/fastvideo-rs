-- Optimistic concurrency for the editable JSON documents (src/docs.ts):
-- every save of a document bumps its version; a save naming an older
-- version is refused (409).
CREATE TABLE IF NOT EXISTS doc_versions (
  kind TEXT NOT NULL,                     -- cluster-spec | policies | attribution | env
  id TEXT NOT NULL,                       -- cluster id | default | account | cluster:<id> | pod:<id>
  version INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER NOT NULL,
  updated_by TEXT NOT NULL,
  PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS audit_target ON audit (target, id);
