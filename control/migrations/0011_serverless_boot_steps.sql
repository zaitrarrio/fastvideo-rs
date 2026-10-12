-- The timed steps of an r2 worker's boot (docs/ops/r2-bench-2026-10-12.md findings 3 and 4: 70 s and 151 s that no
-- report explained): a JSON object {step: seconds} folded from the boot's `step` reports (tools, fetcher, fetch, aux).
ALTER TABLE serverless_boots ADD COLUMN steps TEXT;
