-- Named Cloudflare Tunnels for GMI / Brev pods (src/tunnels.ts,
-- docs/serve/deploy-gmi-brev.md §6.4): one tunnel + one proxied CNAME per pod
-- on the owner's zone (FV_TUNNEL_ZONE, FV_TUNNEL_HOST_TEMPLATE), created by
-- fv-control before the provider's create call and deleted with the pod (a
-- parked Brev instance keeps them: its restart reuses them). The row is
-- written BEFORE the Cloudflare calls, so a crash in between leaves a record
-- the cleanup (the cron) can finish; fv-control only ever deletes a tunnel or
-- a DNS record it recorded here (and checks its name / comment first).
--   state: creating | live | delete_failed | deleted
--   The tunnel token is never stored: the pod gets it in its env (GMI: the
--   create call; Brev: the sealed run script), fetched again on a warm restart.
CREATE TABLE IF NOT EXISTS pod_tunnels (
  name TEXT PRIMARY KEY NOT NULL,          -- the tunnel's name: fv-…-<suffix> (the host's slug)
  pod_id TEXT NOT NULL,                    -- gmi:<name> | brev:<name>
  cluster_id TEXT,
  host TEXT NOT NULL,                      -- from FV_TUNNEL_HOST_TEMPLATE
  url TEXT NOT NULL,                       -- what the pod must report (https://<host>)
  zone_id TEXT,
  tunnel_id TEXT,
  dns_record_id TEXT,
  state TEXT NOT NULL DEFAULT 'creating',
  note TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  deleted_at INTEGER
);
CREATE INDEX IF NOT EXISTS pod_tunnels_pod ON pod_tunnels (pod_id, state);
CREATE INDEX IF NOT EXISTS pod_tunnels_state ON pod_tunnels (state);
CREATE INDEX IF NOT EXISTS pod_tunnels_tunnel ON pod_tunnels (tunnel_id);
