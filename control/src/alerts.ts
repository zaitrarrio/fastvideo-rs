// Alert policies (docs/control/README.md "Alerts"). Notify-only by default,
// except the two rules that already exist in the cluster script: the
// deadline backstop and the balance floor, which stop controller clusters.
// Auto-actions only ever touch pods of controller clusters, never
// external pods (CLAUDE.md: only touch pods you created), with one
// owner-requested exception: the shared build pod's backstop (buildpod.ts).
import type { Env } from "./env";
import { getSetting, now } from "./util";

export interface Policies {
  idle_gpu_pct: number; // a GPU pod is idle below this utilisation…
  idle_min: number; // …with no running jobs for this long: alert
  auto_stop_idle: boolean; // auto-action: remove idle workers of controller clusters
  auto_stop_idle_min: number; // …after this long
  cluster_dph_max: number; // alert: a cluster's $/hr above this
  daily_spend_max: number; // alert: today's account spend above this
  balance_margin: number; // alert: balance below floor + margin
  stop_on_floor: boolean; // auto-action (default on): stop every controller cluster below the floor
  pod_down_min: number; // alert: a controller pod unhealthy this long
  attribution: { prefix: string; owner: string }[]; // external pods by name prefix
  build_pod_backstop: boolean; // auto-action (default on): stop the build pod past the limits below (buildpod.ts)
  build_pod_max_h: number; // …up this many hours (its own cap is 8 h + 30 min grace)
  build_pod_idle_grace_min: number; // …or idle this many minutes past its own idle stop (per its /healthz)
  // NVIDIA Brev keep-on-stop (brev-park.ts). Owner rule 2026-10-11: fv-control never deletes a Brev VM on its own, so
  // these are alert thresholds only (alert brev_parked_over_limit); nothing is deleted past them. The former
  // brev_park_delete_failed is removed (ignored when stored).
  brev_park_max: number; // alert: more than this many parked / held (stopped, weights kept) instances
  brev_park_max_days: number; // alert: one parked / held longer than this many days
  // GMI / Brev have no balance API: the most fv-control may spend there per UTC month ($), as entered in the console
  // (Settings → Policies, the launch form's budget field). null: the Worker var (GMI_BUDGET_USD / BREV_BUDGET_USD).
  // providers.ts providerBudget; owner report 2026-10-18 (the budget entered was not the one used).
  brev_budget_usd: number | null;
  gmi_budget_usd: number | null;
}
export const DEFAULT_POLICIES: Policies = {
  idle_gpu_pct: 5,
  idle_min: 30,
  auto_stop_idle: false,
  auto_stop_idle_min: 60,
  cluster_dph_max: 15,
  daily_spend_max: 150,
  balance_margin: 10,
  stop_on_floor: true,
  pod_down_min: 10,
  attribution: [
    { prefix: "fv-build", owner: "external:build-pod" },
    { prefix: "fv-cluster-", owner: "external:runpod-cluster.sh" },
    { prefix: "fv-b200", owner: "external:b200-bench" },
    { prefix: "fv-serve-", owner: "external:fv-serve-deploys" },
    { prefix: "fv-gw-", owner: "external:gateway-retired" },
    { prefix: "fv-edge", owner: "external:edge" },
    { prefix: "loom-", owner: "external:loom" },
  ],
  build_pod_backstop: true,
  build_pod_max_h: 9,
  build_pod_idle_grace_min: 15,
  brev_park_max: 2,
  brev_park_max_days: 7,
  brev_budget_usd: null,
  gmi_budget_usd: null,
};
/** Policy keys that were removed: dropped when read or saved (brev_park_delete_failed: removed 2026-10-11, owner rule
 * "stopping a brev vm should NOT delete it"). */
export const REMOVED_POLICIES = ["brev_park_delete_failed"] as const;
const dropRemoved = <T extends object>(p: T): T => {
  const out = { ...p } as Record<string, unknown>;
  for (const k of REMOVED_POLICIES) delete out[k];
  return out as T;
};
/** The policies; a key added since they were saved takes its default; a removed one is dropped. */
export const policies = async (env: Env): Promise<Policies> => dropRemoved({ ...DEFAULT_POLICIES, ...(await getSetting<Partial<Policies>>(env, "policies", DEFAULT_POLICIES)) });
export { dropRemoved as dropRemovedPolicies };

export function attribute(name: string, rules: Policies["attribution"]): string {
  for (const r of rules) if (name.startsWith(r.prefix)) return r.owner;
  const m = /^([a-z0-9]+-[a-z0-9]+)-/.exec(name || "");
  return m ? `external:${m[1]}` : "external";
}

export interface AlertIn {
  key: string;
  kind: string;
  severity: "info" | "warn" | "critical";
  target?: string;
  message: string;
  action?: string;
}
/** Opens or refreshes the given alerts and resolves every open one of `kinds` not in the list. */
export async function syncAlerts(env: Env, seen: AlertIn[], kinds: string[]): Promise<void> {
  const t = now();
  const open = await env.DB.prepare("SELECT id, key, kind FROM alerts WHERE resolved_at IS NULL").all<{ id: number; key: string; kind: string }>();
  const openByKey = new Map((open.results || []).map((a) => [a.key, a]));
  const stmts: D1PreparedStatement[] = [];
  for (const a of seen) {
    const cur = openByKey.get(a.key);
    if (cur) stmts.push(env.DB.prepare("UPDATE alerts SET last_seen_at = ?, message = ?, severity = ?, action = COALESCE(?, action) WHERE id = ?").bind(t, a.message, a.severity, a.action ?? null, cur.id));
    else stmts.push(env.DB.prepare("INSERT INTO alerts (key, kind, severity, target, message, opened_at, last_seen_at, action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)").bind(a.key, a.kind, a.severity, a.target ?? null, a.message, t, t, a.action ?? null));
  }
  const seenKeys = new Set(seen.map((a) => a.key));
  for (const a of open.results || []) if (kinds.includes(a.kind) && !seenKeys.has(a.key)) stmts.push(env.DB.prepare("UPDATE alerts SET resolved_at = ? WHERE id = ?").bind(t, a.id));
  if (stmts.length) await env.DB.batch(stmts);
}
