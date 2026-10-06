// Time series (docs/control/README.md "Observability"): per-minute pod
// samples go to Workers Analytics Engine
// (dataset fv_control_metrics) when it is bound, else to D1 pod_samples
// (24 h). Charts read AE through its SQL API (CLOUDFLARE_API_KEY).
import { defaults, type Env } from "./env";
import { fetchWithTimeout } from "./util";

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

export interface SeriesPoint {
  t: number;
  pod: string;
  gpu: number | null;
  cpu: number | null;
  mem: number | null;
  dph: number | null;
  jobs: number | null;
}
const nul = (v: unknown) => (v === null || v === undefined || v === "" || !Number.isFinite(Number(v)) || Number(v) < 0 ? null : Number(v));

/** AE SQL for the per-pod utilisation series (bucketed averages). */
export function aeSeriesSql(hours: number, bucketMin: number, pod?: string): string {
  const safePod = pod && /^[a-z0-9]+$/i.test(pod) ? ` AND blob2 = '${pod}'` : "";
  return `SELECT toStartOfInterval(timestamp, INTERVAL '${bucketMin}' MINUTE) AS t, blob2 AS pod,
  avgIf(double2, double2 >= 0) AS gpu, avgIf(double4, double4 >= 0) AS cpu, avgIf(double5, double5 >= 0) AS mem,
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
