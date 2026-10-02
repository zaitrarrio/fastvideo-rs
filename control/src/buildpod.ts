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

/** The build pod's public /healthz (no auth): timers since 2026-10-02's server. */
export interface BuildPodHealth {
  uptime_s?: number;
  idle_s?: number;
  idle_stop_s?: number;
  jobs_active?: number;
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
