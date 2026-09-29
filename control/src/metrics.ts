// Time series (docs/control/README.md "Observability"): per-minute pod
// samples and gateway pool series go to Workers Analytics Engine
// (dataset fv_control_metrics) when it is bound, else to D1 pod_samples
// (24 h). Charts read AE through its SQL API (CLOUDFLARE_API_KEY).
import { defaults, type Env } from "./env";
import { fetchWithTimeout } from "./util";

/** The only gateway /metrics series the controller keeps. */
export const PROM_WHITELIST = new Set([
  "fv_ready",
  "fv_pool_queued",
  "fv_pool_running",
  "fv_pool_workers",
  "fv_pool_available",
  "fv_pool_streams",
  "fv_pool_oldest_queued_seconds",
  "fv_pool_submitted_total",
  "fv_gateway_dispatched_total",
  "fv_gateway_lost_total",
  "fv_gateway_redispatched_total",
  "fv_jobs_submitted_total",
  "fv_jobs_finished_total",
]);

export interface PromSample {
  name: string;
  labels: Record<string, string>;
  value: number;
}
/** Prometheus text exposition, whitelisted series only (no histograms). */
export function parseProm(text: string, whitelist: Set<string> = PROM_WHITELIST): PromSample[] {
  const out: PromSample[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const m = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([^\s]+)/.exec(line);
    if (!m || !whitelist.has(m[1]!)) continue;
    const labels: Record<string, string> = {};
    if (m[3]) for (const lm of m[3].matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g)) labels[lm[1]!] = lm[2]!.replace(/\\"/g, '"').replace(/\\\\/g, "\\");
    const v = Number(m[4]);
    if (Number.isFinite(v)) out.push({ name: m[1]!, labels, value: v });
  }
  return out;
}

export interface PodSample {
  at: number;
  pod_id: string;
  owner: string;
  cluster_id: string | null;
  name: string;
  gpu: string;
  status: string;
  cost_per_hr: number;
  gpu_util: number | null;
  gpu_mem: number | null;
  cpu: number | null;
  mem: number | null;
  uptime_s: number | null;
  jobs_running: number | null;
  jobs_queued: number | null;
  idle: boolean;
}

export async function writePodSamples(env: Env, samples: PodSample[]): Promise<void> {
  if (env.METRICS) {
    for (const s of samples) {
      env.METRICS.writeDataPoint({
        indexes: [s.pod_id],
        blobs: ["pod", s.pod_id, s.owner, s.cluster_id || "", s.name, s.gpu, s.status],
        doubles: [s.cost_per_hr, s.gpu_util ?? -1, s.gpu_mem ?? -1, s.cpu ?? -1, s.mem ?? -1, s.uptime_s ?? -1, s.jobs_running ?? -1, s.jobs_queued ?? -1, s.idle ? 1 : 0],
      });
    }
    return;
  }
  if (!samples.length) return;
  const stmt = env.DB.prepare(
    "INSERT OR REPLACE INTO pod_samples (at, pod_id, owner, cluster_id, cost_per_hr, gpu_util, gpu_mem, cpu, mem, jobs_running, jobs_queued, idle) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );
  await env.DB.batch(samples.map((s) => stmt.bind(s.at, s.pod_id, s.owner, s.cluster_id, s.cost_per_hr, s.gpu_util, s.gpu_mem, s.cpu, s.mem, s.jobs_running, s.jobs_queued, s.idle ? 1 : 0)));
}
export function writeAccountSample(env: Env, balance: number, spend: number) {
  env.METRICS?.writeDataPoint({ indexes: ["account"], blobs: ["account"], doubles: [balance, spend] });
}
export function writePoolSamples(env: Env, clusterId: string, samples: PromSample[]) {
  if (!env.METRICS) return;
  const byPool = new Map<string, Record<string, number>>();
  for (const s of samples) {
    const pool = s.labels.pool || "_";
    const m = byPool.get(pool) || {};
    m[s.name] = (m[s.name] || 0) + s.value;
    byPool.set(pool, m);
  }
  for (const [pool, m] of byPool) {
    env.METRICS.writeDataPoint({
      indexes: [`${clusterId}:${pool}`],
      blobs: ["pool", clusterId, pool],
      doubles: [m.fv_pool_queued ?? -1, m.fv_pool_running ?? -1, m.fv_pool_workers ?? -1, m.fv_pool_available ?? -1, m.fv_pool_oldest_queued_seconds ?? -1, m.fv_pool_submitted_total ?? -1, m.fv_gateway_lost_total ?? -1],
    });
  }
}

export interface SeriesPoint {
  t: number;
  pod: string;
  gpu: number | null;
  cpu: number | null;
  mem: number | null;
  dph: number | null;
  jobs: number | null;
}
const nul = (v: unknown) => (v === null || v === undefined || Number(v) < 0 ? null : Number(v));

/** AE SQL for the per-pod utilisation series (bucketed averages). */
export function aeSeriesSql(hours: number, bucketMin: number, pod?: string): string {
  const safePod = pod && /^[a-z0-9]+$/i.test(pod) ? ` AND blob2 = '${pod}'` : "";
  return `SELECT toStartOfInterval(timestamp, INTERVAL '${bucketMin}' MINUTE) AS t, blob2 AS pod,
  avg(if(double2 < 0, NULL, double2)) AS gpu, avg(if(double4 < 0, NULL, double4)) AS cpu, avg(if(double5 < 0, NULL, double5)) AS mem,
  avg(double1) AS dph, max(double7) AS jobs
FROM fv_control_metrics
WHERE blob1 = 'pod' AND timestamp > NOW() - INTERVAL '${Math.round(hours)}' HOUR${safePod}
GROUP BY t, pod ORDER BY t FORMAT JSON`;
}

export async function querySeries(env: Env, hours: number, pod?: string): Promise<{ source: string; points: SeriesPoint[] }> {
  const bucketMin = hours <= 3 ? 1 : hours <= 12 ? 5 : 15;
  if (env.METRICS && env.CLOUDFLARE_API_KEY && env.CF_ACCOUNT_ID) {
    try {
      const r = await fetchWithTimeout(`${defaults.cfApi(env)}/accounts/${env.CF_ACCOUNT_ID}/analytics_engine/sql`, {
        method: "POST",
        headers: { authorization: `Bearer ${env.CLOUDFLARE_API_KEY}` },
        body: aeSeriesSql(hours, bucketMin, pod),
        timeoutMs: 20000,
      });
      if (r.ok) {
        const j = (await r.json()) as { data: any[] };
        return {
          source: "analytics-engine",
          points: (j.data || []).map((x) => ({ t: Date.parse(String(x.t).replace(" ", "T") + "Z"), pod: x.pod, gpu: nul(x.gpu), cpu: nul(x.cpu), mem: nul(x.mem), dph: nul(x.dph), jobs: nul(x.jobs) })),
        };
      }
    } catch {
      /* fall back to D1 */
    }
  }
  const since = Date.now() - hours * 3600_000;
  const bucket = bucketMin * 60_000;
  const r = await env.DB.prepare(
    `SELECT (at / ?) * ? AS t, pod_id AS pod, avg(gpu_util) AS gpu, avg(cpu) AS cpu, avg(mem) AS mem, avg(cost_per_hr) AS dph, max(jobs_running) AS jobs
     FROM pod_samples WHERE at > ? ${pod ? "AND pod_id = ?" : ""} GROUP BY t, pod ORDER BY t`,
  )
    .bind(bucket, bucket, since, ...(pod ? [pod] : []))
    .all<any>();
  return { source: "d1", points: (r.results || []).map((x) => ({ t: x.t, pod: x.pod, gpu: nul(x.gpu), cpu: nul(x.cpu), mem: nul(x.mem), dph: nul(x.dph), jobs: nul(x.jobs) })) };
}
