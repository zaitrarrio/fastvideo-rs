-- NVIDIA Brev keep-on-stop (src/brev-park.ts, docs/serve/deploy-gmi-brev.md
-- §3.4, §7): every Brev workspace fv-control creates, so that a stoppable one
-- can be stopped ("parked") with its weights under /home/ubuntu/workspace
-- when its cluster stops, and started again for the next launch of the same
-- instance type and trees instead of downloading them again.
--   state: live (a pod of a cluster) | parked (stopped, may be restarted) |
--          held (stopped after a failed or timed-out restart: kept for the owner,
--          never restarted on its own) | deleted | gone (Brev no longer lists it)
--   trees: JSON {"<tree>": "<40-hex revision>"} known complete on its disk (union of the launches that became ready)
--   boot_hash: sha256 of the per-instance boot token its startup script holds;
--   it fetches the current run script (image + env, sealed in run_sealed) from
--   GET /ingest/v1/brev-boot at every boot, so a restart runs the new launch.
CREATE TABLE IF NOT EXISTS brev_instances (
  workspace_id TEXT PRIMARY KEY NOT NULL,
  pod_id TEXT NOT NULL,                   -- brev:<name> (the instance keeps its first name)
  name TEXT NOT NULL,
  instance_type TEXT NOT NULL,
  location TEXT,
  stoppable INTEGER NOT NULL DEFAULT 0,
  disk_gb INTEGER,
  storage_usd_per_gb_hr REAL,
  trees TEXT NOT NULL DEFAULT '{}',
  launch_trees TEXT NOT NULL DEFAULT '{}', -- the trees the current (or last) launch fetches; into `trees` once it was ready
  cluster_id TEXT,                        -- the cluster it serves (live) or served last
  state TEXT NOT NULL DEFAULT 'live',
  boot_hash TEXT,
  run_sealed TEXT,
  boots INTEGER NOT NULL DEFAULT 0,       -- run-script fetches (one per boot)
  created_at INTEGER NOT NULL,
  parked_at INTEGER,
  restarted_at INTEGER,
  restarts INTEGER NOT NULL DEFAULT 0,
  note TEXT,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS brev_instances_state ON brev_instances (state, instance_type);
CREATE INDEX IF NOT EXISTS brev_instances_pod ON brev_instances (pod_id);
CREATE UNIQUE INDEX IF NOT EXISTS brev_instances_boot ON brev_instances (boot_hash);
