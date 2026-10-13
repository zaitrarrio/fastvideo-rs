-- NVIDIA Brev VMs that cannot be stopped (src/brev-park.ts, docs/serve/deploy-gmi-brev.md §3.4; owner decision
-- 2026-10-13: non-stoppable Brev types are launched like any other). A stop Brev answers with "does not support
-- stop" (HTTP 500, measured 2026-10-13) leaves the VM RUNNING: state 'running_unstoppable', never deleted by
-- fv-control, billed at its full GPU $/hr until the owner deletes it.
--   usd_per_hr:    the VM's GPU $/hr (the price at create; the ledger and budget bill a running VM fv-control
--                  released at it)
--   stop_tried_at: the last stop sent after a refusal (the cron's backoff: BREV_UNSTOPPABLE_RETRY_S, default 1 h)
ALTER TABLE brev_instances ADD COLUMN usd_per_hr REAL;
ALTER TABLE brev_instances ADD COLUMN stop_tried_at INTEGER;
