// Remote backstop for the shared CPU build pod (scripts/dev/build-pod.sh,
// docs/dev/build-pod.md "Costs and money guards"). The pod stops itself
// (idle / wall-clock cap) and a second watchdog in its start command repeats
// the cap; this cron check is the one that does not depend on the pod at all.
// It is the single exception to "auto-actions never touch external pods",
// asked for by the owner on 2026-10-02 after a build pod ran 9 h because its
// self-stop calls were all refused. It acts only on pods attributed
// `external:build-pod`, and only past the limits the pod itself should have
// enforced plus a margin.
import { defaults, type Env } from "./env";
import { runpod } from "./runpod";
import { fetchWithTimeout, HttpError } from "./util";

/** The build pod's public /healthz (no auth): timers since 2026-10-02's server; `self_stop` and `jobs` since the server of wip/ui-dashboard. */
export interface BuildPodHealth {
  ready?: boolean;
  phase?: string;
  boot?: number; // unix s
  uptime_s?: number;
  idle_s?: number;
  idle_stop_in_s?: number | null; // null: jobs are active (not counting down)
  max_stop_in_s?: number;
  idle_stop_s?: number;
  max_s?: number;
  max_grace_s?: number;
  jobs_active?: number;
  /** The last self-stop attempt (Stopper.info()). */
  self_stop?: { attempts?: number; next_at?: number | null; reason?: string; at?: number; ok?: string | null; error?: string | null };
  /** The active jobs (no argv). */
  jobs?: { id: string; agent: string; state: string; seconds: number | null }[];
}

export interface BuildPodPolicy {
  build_pod_backstop: boolean;
  build_pod_max_h: number;
  build_pod_idle_grace_min: number;
}

/** Why the build pod must stop now, or null. `uptimeS`: Runpod's runtime uptime. */
export function buildPodVerdict(uptimeS: number | null | undefined, h: BuildPodHealth | null, pol: BuildPodPolicy): string | null {
  if (!pol.build_pod_backstop) return null;
  const up = Math.max(Number(uptimeS ?? 0), Number(h?.uptime_s ?? 0));
  if (pol.build_pod_max_h > 0 && up >= pol.build_pod_max_h * 3600) return `up ${(up / 3600).toFixed(1)} h (backstop at ${pol.build_pod_max_h} h)`;
  if (h && typeof h.idle_s === "number" && typeof h.idle_stop_s === "number" && h.jobs_active === 0 && pol.build_pod_idle_grace_min >= 0) {
    if (h.idle_s >= h.idle_stop_s + pol.build_pod_idle_grace_min * 60)
      return `idle ${Math.round(h.idle_s / 60)} min with no jobs (its own stop is at ${Math.round(h.idle_stop_s / 60)} min)`;
  }
  return null;
}

/** The pod's /healthz, or null (old server without timers, or not answering). */
export async function buildPodHealth(env: Env, pod: string): Promise<BuildPodHealth | null> {
  try {
    const r = await fetchWithTimeout(`${defaults.podUrl(env, pod)}/healthz`, { timeoutMs: 10000 });
    if (!r.ok) return null;
    const j = (await r.json()) as BuildPodHealth;
    return j && typeof j === "object" ? j : null;
  } catch {
    return null;
  }
}

/** Stop (keeps the pod record; `up` restarts or recreates it); terminate when stop is refused. */
export async function stopBuildPod(env: Env, pod: string): Promise<string> {
  try {
    await runpod.stop(env, pod);
    return "stopped";
  } catch (e) {
    const why = e instanceof HttpError ? e.message : String(e);
    await runpod.remove(env, pod);
    return `terminated (stop refused: ${why.slice(0, 120)})`;
  }
}

/** One build pod as the dashboard's card shows it (read only). */
export interface BuildPodView {
  pod_id: string;
  name: string | null;
  status: string | null;
  cost_per_hr: number | null;
  dc: string | null;
  uptime_s: number | null; // Runpod's
  health: BuildPodHealth | null; // null: not running, or no answer / an old server
  backstop: { enabled: boolean; cap_in_s: number | null; idle_in_s: number | null; verdict: string | null };
}

/** Every build pod (external:build-pod) the collector knows, with its /healthz timers and the controller backstop's distance. */
export async function buildPodStatus(env: Env, pol: BuildPodPolicy): Promise<BuildPodView[]> {
  const r = await env.DB.prepare("SELECT pod_id, name, desired_status, cost_per_hr, dc, uptime_s FROM pods WHERE owner = 'external:build-pod' AND gone_at IS NULL ORDER BY last_seen DESC LIMIT 5").all<any>();
  const out: BuildPodView[] = [];
  for (const p of r.results || []) {
    const running = p.desired_status === "RUNNING";
    const h = running ? await buildPodHealth(env, p.pod_id) : null;
    const up = Math.max(Number(p.uptime_s ?? 0), Number(h?.uptime_s ?? 0));
    const cap = pol.build_pod_backstop && pol.build_pod_max_h > 0 && running ? Math.max(0, Math.round(pol.build_pod_max_h * 3600 - up)) : null;
    const idle =
      pol.build_pod_backstop && running && h && h.jobs_active === 0 && typeof h.idle_s === "number" && typeof h.idle_stop_s === "number"
        ? Math.max(0, Math.round(h.idle_stop_s + pol.build_pod_idle_grace_min * 60 - h.idle_s))
        : null;
    out.push({
      pod_id: p.pod_id,
      name: p.name ?? null,
      status: p.desired_status ?? null,
      cost_per_hr: p.cost_per_hr ?? null,
      dc: p.dc ?? null,
      uptime_s: p.uptime_s ?? null,
      health: h,
      backstop: { enabled: pol.build_pod_backstop, cap_in_s: cap, idle_in_s: idle, verdict: running ? buildPodVerdict(p.uptime_s, h, pol) : null },
    });
  }
  return out;
}
