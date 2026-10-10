// The GMI Cloud / NVIDIA Brev half of the per-minute collector
// (docs/serve/deploy-gmi-brev.md §6.5): each provider's fv-named instances
// into the pods table (provider 'gmi' / 'brev'), their cost into the ledger
// at the $/hr fv-control recorded when it made them, and the budget guard
// that replaces the balance floor (neither provider has a balance API). It
// only ever acts on pods fv-control recorded (cluster_pods); an fv-named
// instance it did not record is reported, never touched.
import type { AlertIn, Policies } from "./alerts";
import { brevUnlisted, enforceParkLimits, markBrevDeleted, parkedRows, storageDph } from "./brev-park";
import { ownerOf, type Cluster } from "./cluster/store";
import { stopCluster } from "./collector";
import { OTHER_PROVIDERS, type OtherProviderId } from "./enums";
import type { Env } from "./env";
import { providerImpl, providerSpend, type ProviderInstance } from "./providers";
import { utcDay } from "./util";

export interface ProvidersCollect {
  alerts: AlertIn[];
  kinds: string[];
  actions: string[];
  summary: Record<string, { instances: number; running: number; spend_per_hr: number; month_usd: number; budget_usd: number | null }>;
}

/** Budget alerts and whether to stop (pure: unit-tested). */
export function budgetDecision(p: OtherProviderId, title: string, month: number, budget: number | null, stopOnFloor: boolean): { alert: AlertIn | null; stop: boolean } {
  if (budget === null) return { alert: null, stop: false };
  if (month >= budget)
    return {
      alert: { key: `${p}_budget`, kind: `${p}_budget`, severity: "critical", message: `${title}: $${month.toFixed(2)} spent this month, at or over the budget $${budget}${stopOnFloor ? ": stopping its pods" : ""}`, action: stopOnFloor ? "stop" : undefined },
      stop: stopOnFloor,
    };
  if (month >= 0.8 * budget) return { alert: { key: `${p}_budget`, kind: `${p}_budget`, severity: "warn", message: `${title}: $${month.toFixed(2)} spent this month, ${Math.round((month / budget) * 100)}% of the budget $${budget}` }, stop: false };
  return { alert: null, stop: false };
}

export async function collectProviders(env: Env, t: number, dtMs: number, pol: Policies, clusters: Cluster[]): Promise<ProvidersCollect> {
  const res: ProvidersCollect = { alerts: [], kinds: [], actions: [], summary: {} };
  const byId = new Map(clusters.map((c) => [c.id, c]));
  const day = utcDay(t);
  for (const p of OTHER_PROVIDERS) {
    const impl = providerImpl(p);
    if (impl.off(env)) continue;
    let insts: ProviderInstance[];
    try {
      insts = await impl.list(env);
    } catch (e) {
      res.alerts.push({ key: `${p}_api`, kind: `${p}_api`, severity: "warn", message: `${impl.title} API: ${(e as Error).message.slice(0, 240)}` });
      continue;
    }
    res.kinds.push(`${p}_api`, `${p}_budget`, `${p}_orphan`);
    const known = await env.DB.prepare("SELECT pod_id, cluster_id, cost_per_hr, created_at, deleted_at FROM cluster_pods WHERE pod_id LIKE ? AND (deleted_at IS NULL OR deleted_at > ?)")
      .bind(`${p}:%`, t - 6 * 3600_000)
      .all<{ pod_id: string; cluster_id: string; cost_per_hr: number | null; created_at: number; deleted_at: number | null }>();
    const ctl = new Map((known.results || []).map((r) => [r.pod_id, r]));
    const prev = await env.DB.prepare("SELECT pod_id, first_seen FROM pods WHERE gone_at IS NULL AND provider = ?").bind(p).all<{ pod_id: string; first_seen: number }>();
    const prevBy = new Map((prev.results || []).map((r) => [r.pod_id, r]));
    const stmts: D1PreparedStatement[] = [];
    const seen = new Set<string>();
    let running = 0;
    let dphSum = 0;
    // Brev keep-on-stop (brev-park.ts): parked / held instances are ours by their record, billed as storage.
    const parked = new Map(p === "brev" ? (await parkedRows(env)).map((r) => [r.workspace_id, r]) : []);
    const seenIds = new Set(insts.map((i) => i.id));
    for (const i of insts) {
      seen.add(i.key);
      const pk = parked.get(i.id);
      if (pk) {
        const sdph = storageDph(pk);
        stmts.push(
          env.DB.prepare(
            `INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, dc, image, uptime_s, gpu_util, gpu_mem, cpu, mem, jobs_running, jobs_queued, health, build_sha, idle_since, first_seen, last_seen, gone_at, provider)
             VALUES (?, ?, 'brev:parked', NULL, ?, ?, ?, 1, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'parked', NULL, NULL, ?, ?, NULL, ?)
             ON CONFLICT (pod_id) DO UPDATE SET name = excluded.name, owner = excluded.owner, cluster_id = NULL, desired_status = excluded.desired_status, cost_per_hr = excluded.cost_per_hr,
               gpu = excluded.gpu, uptime_s = NULL, health = excluded.health, last_seen = excluded.last_seen, gone_at = NULL, provider = excluded.provider`,
          ).bind(i.key, i.name, pk.state.toUpperCase(), sdph, i.gpu, pk.location, prevBy.get(i.key)?.first_seen ?? t, t, p),
          env.DB.prepare(
            `INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, NULL, 'brev:parked', ?, 1, 0)
             ON CONFLICT (day, pod_id) DO UPDATE SET usd = usd + excluded.usd, minutes = minutes + 1`,
          ).bind(day, i.key, (sdph * dtMs) / 3_600_000),
        );
        if (i.state === "running" || i.state === "starting")
          res.alerts.push({ key: `brev_parked_running:${i.id}`, kind: "brev_parked", severity: "warn", target: i.key, message: `parked Brev ${i.name} is ${i.raw} outside fv-control (GPU billing): stop it in the Brev console or delete it (fv-control.sh brev delete-parked ${i.id})` });
        continue;
      }
      const row = ctl.get(i.key);
      const c = row ? byId.get(row.cluster_id) : undefined;
      const owner = c ? ownerOf(c) : `external:${p}`;
      const dph = Number(row?.cost_per_hr ?? 0);
      const live = i.state === "running";
      if (live) running++, (dphSum += dph);
      const uptime = i.createdAt ? Math.max(0, Math.round((t - i.createdAt) / 1000)) : null;
      stmts.push(
        env.DB.prepare(
          `INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, dc, image, uptime_s, gpu_util, gpu_mem, cpu, mem, jobs_running, jobs_queued, health, build_sha, idle_since, first_seen, last_seen, gone_at, provider)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL, NULL, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, NULL, NULL, ?, ?, NULL, ?)
           ON CONFLICT (pod_id) DO UPDATE SET name = excluded.name, owner = excluded.owner, cluster_id = excluded.cluster_id, desired_status = excluded.desired_status, cost_per_hr = excluded.cost_per_hr,
             gpu = excluded.gpu, uptime_s = excluded.uptime_s, health = excluded.health, last_seen = excluded.last_seen, gone_at = NULL, provider = excluded.provider`,
        ).bind(i.key, i.name, owner, row?.cluster_id ?? null, live ? "RUNNING" : i.state.toUpperCase(), dph, i.gpu, uptime, i.state === "failed" ? "down" : live ? "unknown" : "loading", prevBy.get(i.key)?.first_seen ?? t, t, p),
      );
      // Billing starts at create on both providers (UNVERIFIED for GMI's "creating"): every listed instance but a stopped one costs.
      if (i.state !== "stopped")
        stmts.push(
          env.DB.prepare(
            `INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, ?, ?, ?, 1, 0)
             ON CONFLICT (day, pod_id) DO UPDATE SET usd = usd + excluded.usd, minutes = minutes + 1, owner = excluded.owner`,
          ).bind(day, i.key, row?.cluster_id ?? null, owner, (dph * dtMs) / 3_600_000),
        );
      if (!row) res.alerts.push({ key: `${p}_orphan:${i.key}`, kind: `${p}_orphan`, severity: "warn", target: i.key, message: `${impl.title} ${i.name} (${i.raw}) has an fv- name but fv-control did not record it: not touched (delete it in the ${impl.title} console if it is ours)` });
    }
    for (const r of prev.results || []) if (!seen.has(r.pod_id)) stmts.push(env.DB.prepare("UPDATE pods SET gone_at = ?, desired_status = 'GONE' WHERE pod_id = ?").bind(t, r.pod_id));
    // Recorded pods the provider no longer lists (after 10 min: a fresh create may not be listed yet): gone.
    for (const [pod, r] of ctl) if (!r.deleted_at && !seen.has(pod) && t - r.created_at > 10 * 60_000) stmts.push(env.DB.prepare("UPDATE cluster_pods SET status = 'gone', deleted_at = ? WHERE pod_id = ? AND deleted_at IS NULL").bind(t, pod));
    for (let k = 0; k < stmts.length; k += 50) await env.DB.batch(stmts.slice(k, k + 50));
    if (p === "brev") {
      res.kinds.push("brev_parked");
      // Records whose workspace Brev no longer lists (10 min after their last change): gone.
      for (const id of await brevUnlisted(env, seenIds, t - 10 * 60_000)) await markBrevDeleted(env, id, "gone", "no longer listed by Brev");
      // The park limits (brev_park_max, brev_park_max_days): the oldest of ours go.
      res.actions.push(...(await enforceParkLimits(env, pol, t)));
    }

    const spend = await providerSpend(env, p);
    const budget = impl.budget(env);
    res.summary[p] = { instances: insts.length, running, spend_per_hr: dphSum, month_usd: spend.month, budget_usd: budget };
    const d = budgetDecision(p, impl.title, spend.month, budget, pol.stop_on_floor);
    if (d.alert) res.alerts.push(d.alert);
    if (d.stop)
      for (const c of clusters)
        if (Object.values(c.state.workers).some((l) => l.some((r) => r.pod.startsWith(`${p}:`)))) res.actions.push(await stopCluster(env, c, `${p} budget`));
  }
  return res;
}
