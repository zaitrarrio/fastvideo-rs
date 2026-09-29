// Release channels and the deployment registry from fv-jobs (read only;
// docs/serve/releases.md), and each controller cluster's drift against the
// channel it follows.
import type { Cluster } from "./cluster/store";
import type { Env } from "./env";
import { parseJson } from "./util";

export interface ReleaseRow {
  id: number;
  channel: string;
  git_sha: string;
  digests: Record<string, string>;
  action: string;
  promoted_at: number;
  promoted_by: string;
  notes?: string;
  run_url?: string;
  rolled_back_at?: number | null;
}

export async function releaseHeads(env: Env): Promise<{ available: boolean; heads: ReleaseRow[]; history: ReleaseRow[] }> {
  if (!env.JOBS_DB) return { available: false, heads: [], history: [] };
  try {
    const h = await env.JOBS_DB.prepare("SELECT * FROM releases WHERE id IN (SELECT MAX(id) FROM releases GROUP BY channel) ORDER BY channel").all<any>();
    const hist = await env.JOBS_DB.prepare("SELECT * FROM releases ORDER BY id DESC LIMIT 30").all<any>();
    const fix = (r: any): ReleaseRow => ({ ...r, digests: parseJson<Record<string, string>>(r.digests, {}) });
    return { available: true, heads: (h.results || []).map(fix), history: (hist.results || []).map(fix) };
  } catch {
    return { available: false, heads: [], history: [] };
  }
}
export async function registry(env: Env, limit = 100): Promise<any[]> {
  if (!env.JOBS_DB) return [];
  try {
    const r = await env.JOBS_DB.prepare(
      "SELECT id, kind, runpod_id, name, pool, variant, image, digest, git_sha, channel, region, dc, gpu, cost_per_hr, created_at, ready_at, deleted_at, created_by, status FROM deployments ORDER BY created_at DESC LIMIT ?",
    )
      .bind(limit)
      .all<any>();
    return r.results || [];
  } catch {
    return [];
  }
}

/** The release key of a pod's image: its variant (wan pool: wan5b), or `debug` for the all-in-one image. */
export function releaseKey(c: Cluster, podKey: string): string {
  if (c.spec.image.ref) return "debug";
  if (podKey === "gateway") return "gateway";
  return c.spec.pools.find((p) => p.id === podKey)?.variant || podKey;
}

/** Per pod of a cluster: the digest it runs, the channel head's digest, and whether they differ. */
export function clusterDrift(c: Cluster, heads: ReleaseRow[]) {
  const channel = c.spec.image.channel;
  const head = channel ? heads.find((h) => h.channel === channel) : undefined;
  const out: { pod: string; key: string; running: string | null; head: string | null; drift: boolean; sha?: string }[] = [];
  const recs = [...(c.state.gateway ? [["gateway", c.state.gateway] as const] : []), ...Object.entries(c.state.workers).flatMap(([p, l]) => l.map((r) => [p, r] as const))];
  const digestOf = (img?: string) => (img && img.includes("@") ? img.split("@")[1]! : null);
  for (const [key, r] of recs) {
    const rk = releaseKey(c, key);
    const want = head ? digestOf(head.digests[rk]) : null;
    const running = digestOf(r.image);
    out.push({ pod: r.pod, key: rk, running, head: want, drift: !!(want && running && want !== running), sha: head?.git_sha?.slice(0, 7) });
  }
  let sha: string | undefined;
  for (const h of heads) for (const d of Object.values(h.digests)) if (recs.some(([, r]) => digestOf(r.image) === digestOf(d))) sha = h.git_sha.slice(0, 7);
  return { channel: channel || null, head_sha: head?.git_sha?.slice(0, 7) || null, running_sha: sha || null, pods: out, drift: out.some((x) => x.drift) };
}
