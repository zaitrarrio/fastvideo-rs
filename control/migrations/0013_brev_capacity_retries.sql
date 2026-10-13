-- NVIDIA Brev warm restarts that fail for capacity (src/brev-park.ts restartDiag, docs/serve/deploy-gmi-brev.md §3.4).
-- Brev keeps a start's failure in the workspace's environment.tasks[] (e.g. "AWSClient.StartInstance: zone has
-- insufficient resources to fulfill the request, InsufficientCapacity (retryable: true)"), with statusMessage empty.
-- fv-control sends the start again a few times with a short backoff, then gives the VM back parked (its stop had
-- succeeded: never deleted, owner rule 2026-10-13) and the launch makes a fresh VM.
--   capacity_retries:  starts sent again for this restart after an InsufficientCapacity failure (BREV_CAPACITY_RETRIES)
--   capacity_retry_at: when the last of them was sent (the backoff, BREV_CAPACITY_RETRY_S, counts from it)
-- The stop_tried_at column (0012) now means: the time of a stop Brev accepted but has not yet shown STOPPED (or of a
-- stop requested while DEPLOYING); the states running_unstoppable / stop_failed (0012) are no longer written (owner
-- rule 2026-10-13: a VM whose stop does not succeed is deleted; the cron deletes rows left in them).
ALTER TABLE brev_instances ADD COLUMN capacity_retries INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brev_instances ADD COLUMN capacity_retry_at INTEGER;
