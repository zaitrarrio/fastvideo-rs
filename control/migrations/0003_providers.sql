-- A second GPU provider (CloudRift, docs/ops/cloudrift.md) next to Runpod:
-- every pod row says where it runs, so each provider's collector only marks
-- its own rows gone.
ALTER TABLE pods ADD COLUMN provider TEXT NOT NULL DEFAULT 'runpod';
CREATE INDEX IF NOT EXISTS pods_provider ON pods (provider, gone_at);
