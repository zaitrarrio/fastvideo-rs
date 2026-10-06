// The verdict of an `up` that waits for its pools (cluster/do.ts), as pure
// functions. A pool is ready, still starting, failed (its pods crash-looped,
// could not pull, or got stuck: docs/control/README.md §4 "Early failure"),
// or has no stock. The wait ends as soon as nothing is starting, not after
// 30 minutes, and its error names every pool that did not come up and why.
import type { PoolStatus } from "./payloads";

export const READY_WAIT_MS = 30 * 60_000;
/** A pool without stock is retried this often while other pools are still starting. */
export const NO_STOCK_RETRY_MS = 3 * 60_000;
/** The wait's progress line is logged when it changes, or at least this often. */
export const SUMMARY_EVERY_MS = 2 * 60_000;

export interface PoolView {
  id: string;
  status: PoolStatus["status"];
  detail?: string;
}

/** One line: every pool by state, e.g. "ready 1/5: fake | starting: h3-max (pulling) | failed: h3-turbo (crash loop …) | no stock: h3-ref2v". */
export function summarize(pools: PoolView[]): string {
  const by = (s: PoolStatus["status"]) => pools.filter((p) => p.status === s);
  const part = (label: string, xs: PoolView[]) => (xs.length ? `${label}: ${xs.map((p) => (p.detail ? `${p.id} (${p.detail.slice(0, 160)})` : p.id)).join(", ")}` : null);
  const ready = by("ready");
  return [`ready ${ready.length}/${pools.length}${ready.length ? `: ${ready.map((p) => p.id).join(", ")}` : ""}`, part("starting", by("starting")), part("failed", by("failed")), part("no stock", by("no_stock"))].filter(Boolean).join(" | ");
}

/** Whether the wait is over, and its error when not every pool came up. */
export function upOutcome(pools: PoolView[], elapsedMs: number, waitMs = READY_WAIT_MS): { done: false } | { done: true; error?: string } {
  const starting = pools.filter((p) => p.status === "starting");
  const bad = pools.filter((p) => p.status === "failed" || p.status === "no_stock");
  if (starting.length && elapsedMs <= waitMs) return { done: false };
  if (!starting.length && !bad.length) return { done: true };
  const why = [
    ...bad.map((p) => `${p.id}: ${p.status === "no_stock" ? "no stock" : "failed"}${p.detail ? ` (${p.detail.slice(0, 200)})` : ""}`),
    ...starting.map((p) => `${p.id}: not ready after ${Math.round(waitMs / 60000)} min${p.detail ? ` (${p.detail.slice(0, 200)})` : ""}`),
  ];
  const ready = pools.filter((p) => p.status === "ready").map((p) => p.id);
  return {
    done: true,
    error: `not every pool came up: ${why.join("; ")}. ${ready.length ? `Ready: ${ready.join(", ")} (those pods stay up; scale or stop the cluster).` : "No pool is ready: stop the cluster."}`,
  };
}
