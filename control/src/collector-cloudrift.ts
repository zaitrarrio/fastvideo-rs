// The CloudRift half of the per-minute collector (docs/ops/cloudrift.md
// "fv-control"): rentals into the pods table (provider 'cloudrift'), costs,
// idle tracking, the balance, and two backstops that only ever touch our own
// rentals (tag fv-owner:fastvideo-rs): the fv-deadline:<unix> tag every
// script sets, and the CloudRift balance floor.
import type { AlertIn, Policies } from "./alerts";
import { cloudrift, cloudriftEnabled, type CloudriftInstance } from "./cloudrift";
import { defaults, type Env } from "./env";
import { audit, putSetting, utcDay } from "./util";

export const CLOUDRIFT_ALERT_KINDS = ["cloudrift_deadline", "cloudrift_balance_floor", "cloudrift_balance_margin", "cloudrift_failed"];

/** The owner a rental is attributed to: cloudrift:<fv-kind> for ours, external:cloudrift otherwise. */
export function cloudriftOwner(i: CloudriftInstance): string {
  if (!i.ours) return "external:cloudrift";
  const k = i.tags.find((t) => t.startsWith("fv-kind:"));
  return `cloudrift:${k ? k.slice(8) : "fv"}`;
}
const live = (i: CloudriftInstance) => i.status === "Active" || i.status === "Initializing";

export interface CloudriftDecision {
  terminate: { id: string; why: string }[];
  alerts: AlertIn[];
}
/** What the backstops do this minute (pure: unit-tested). */
export function cloudriftDecisions(insts: CloudriftInstance[], balance: number, floor: number, pol: Policies, t: number): CloudriftDecision {
  const out: CloudriftDecision = { terminate: [], alerts: [] };
  for (const i of insts) {
    if (!i.ours) continue;
    if (i.deadlineMs !== null && t >= i.deadlineMs && (live(i) || i.status === "Failed")) {
      out.terminate.push({ id: i.id, why: "deadline" });
      out.alerts.push({ key: `cloudrift_deadline:${i.id}`, kind: "cloudrift_deadline", severity: "critical", target: i.id, message: `CloudRift ${i.name} passed its deadline (${new Date(i.deadlineMs).toISOString()}): terminating`, action: "terminate" });
    } else if (i.status === "Failed") {
      // A failed rental holds no resources but stays listed until dismissed (terminate).
      out.terminate.push({ id: i.id, why: "failed" });
      out.alerts.push({ key: `cloudrift_failed:${i.id}`, kind: "cloudrift_failed", severity: "warn", target: i.id, message: `CloudRift ${i.name} failed: ${i.failure || "no reason given"} (dismissing)` });
    }
  }
  if (balance < floor) {
    out.alerts.push({ key: "cloudrift_balance_floor", kind: "cloudrift_balance_floor", severity: "critical", message: `CloudRift balance $${balance.toFixed(2)} is below the floor $${floor}`, action: pol.stop_on_floor ? "terminate our rentals" : undefined });
    if (pol.stop_on_floor) for (const i of insts) if (i.ours && live(i) && !out.terminate.some((x) => x.id === i.id)) out.terminate.push({ id: i.id, why: "balance floor" });
  } else if (balance < floor + pol.balance_margin)
    out.alerts.push({ key: "cloudrift_balance_margin", kind: "cloudrift_balance_margin", severity: "warn", message: `CloudRift balance $${balance.toFixed(2)} is within $${pol.balance_margin} of the floor $${floor}` });
  return out;
}

export interface CloudriftCollect {
  enabled: boolean;
  ok: boolean;
  balance: number | null;
  instances: number;
  running: number;
  spend_per_hr: number;
  alerts: AlertIn[];
  /** Alert kinds to resolve when absent (only when the API answered). */
  kinds: string[];
  actions: string[];
  error?: string;
}

export async function collectCloudrift(env: Env, t: number, dtMs: number, pol: Policies): Promise<CloudriftCollect> {
  const res: CloudriftCollect = { enabled: cloudriftEnabled(env), ok: false, balance: null, instances: 0, running: 0, spend_per_hr: 0, alerts: [], kinds: [], actions: [] };
  if (!res.enabled) return res;
  let insts: CloudriftInstance[];
  try {
    res.balance = await cloudrift.balance(env);
    insts = await cloudrift.instances(env);
  } catch (e) {
    res.error = (e as Error).message.slice(0, 300);
    res.alerts.push({ key: "cloudrift_api", kind: "cloudrift_api", severity: "warn", message: `CloudRift API: ${res.error}` });
    return res;
  }
  res.ok = true;
  res.kinds = [...CLOUDRIFT_ALERT_KINDS, "cloudrift_api"];
  res.instances = insts.length;
  const util = await cloudrift.gpuUtil(env, insts.filter((i) => i.status === "Active").map((i) => i.id)).catch(() => new Map<string, number>());
  const prev = await env.DB.prepare("SELECT pod_id, idle_since, first_seen FROM pods WHERE gone_at IS NULL AND provider = 'cloudrift'").all<{ pod_id: string; idle_since: number | null; first_seen: number }>();
  const prevBy = new Map((prev.results || []).map((r) => [r.pod_id, r]));
  const day = utcDay(t);
  const stmts: D1PreparedStatement[] = [];
  const seen = new Set<string>();
  for (const i of insts) {
    seen.add(i.id);
    const owner = cloudriftOwner(i);
    const running = i.status === "Active";
    const u = util.get(i.id) ?? null;
    const idleNow = running && u !== null && u < pol.idle_gpu_pct;
    const pv = prevBy.get(i.id);
    const idleSince = idleNow ? (pv?.idle_since ?? t) : null;
    const uptime = i.createdAt ? Math.max(0, Math.round((t - Date.parse(i.createdAt)) / 1000)) : null;
    if (running) res.running++, (res.spend_per_hr += i.costPerHr);
    stmts.push(
      env.DB.prepare(
        `INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, dc, image, uptime_s, gpu_util, gpu_mem, cpu, mem, jobs_running, jobs_queued, health, build_sha, idle_since, first_seen, last_seen, gone_at, provider)
         VALUES (?, ?, ?, NULL, ?, ?, ?, ?, NULL, NULL, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, NULL, ?, ?, ?, NULL, 'cloudrift')
         ON CONFLICT (pod_id) DO UPDATE SET name = excluded.name, owner = excluded.owner, desired_status = excluded.desired_status, cost_per_hr = excluded.cost_per_hr,
           gpu = excluded.gpu, gpu_count = excluded.gpu_count, uptime_s = excluded.uptime_s, gpu_util = excluded.gpu_util, health = excluded.health,
           idle_since = excluded.idle_since, last_seen = excluded.last_seen, gone_at = NULL, provider = 'cloudrift'`,
      ).bind(i.id, i.name, owner, running ? "RUNNING" : i.status.toUpperCase(), i.costPerHr, i.gpu || i.instanceType, i.gpuCount, uptime, u, i.status === "Failed" ? "down" : running ? "unknown" : "loading", idleSince, pv?.first_seen ?? t, t),
    );
    if (running)
      stmts.push(
        env.DB.prepare(
          `INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, NULL, ?, ?, 1, ?)
           ON CONFLICT (day, pod_id) DO UPDATE SET usd = usd + excluded.usd, minutes = minutes + 1, idle_minutes = idle_minutes + excluded.idle_minutes, owner = excluded.owner`,
        ).bind(day, i.id, owner, (i.costPerHr * dtMs) / 3_600_000, idleNow ? 1 : 0),
      );
    if (running && idleSince !== null && (t - idleSince) / 60_000 >= pol.idle_min)
      res.alerts.push({ key: `pod_idle:${i.id}`, kind: "pod_idle", severity: "warn", target: i.id, message: `CloudRift ${i.name} (${owner}) idle for ${Math.round((t - idleSince) / 60_000)} min at $${i.costPerHr}/hr (GPU ${u?.toFixed(0)}%)` });
  }
  for (const r of prev.results || []) if (!seen.has(r.pod_id)) stmts.push(env.DB.prepare("UPDATE pods SET gone_at = ?, desired_status = 'GONE' WHERE pod_id = ?").bind(t, r.pod_id));
  for (let k = 0; k < stmts.length; k += 50) await env.DB.batch(stmts.slice(k, k + 50));
  await putSetting(env, "cloudrift_account", { at: t, balance: res.balance, spend_per_hr: res.spend_per_hr, running: res.running }, "cron");

  const dec = cloudriftDecisions(insts, res.balance!, defaults.cloudriftFloor(env), pol, t);
  res.alerts.push(...dec.alerts);
  for (const x of dec.terminate) {
    let result: string;
    try {
      result = (await cloudrift.terminate(env, x.id)) ? "terminated" : "not confirmed";
    } catch (e) {
      result = `terminate failed: ${(e as Error).message.slice(0, 160)}`;
    }
    res.actions.push(`cloudrift ${x.id}: ${result} (${x.why})`);
    await audit(env, { actor: `policy:cloudrift_${x.why.replace(/ /g, "_")}`, action: "cloudrift.terminate", target: x.id, detail: `${x.why}: ${result}` });
  }
  return res;
}
