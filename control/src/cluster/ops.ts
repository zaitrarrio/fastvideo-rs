// Cluster primitives (each one short enough for one Durable Object alarm):
// create / patch / delete workers, the admin token, the price and balance
// guards, the edge's views and the worker probes. (The gateway pod is
// retired: docs/serve/edge-control-plane.md §9 stage 4.)
import { randomToken, sha256Hex } from "../crypto";
import { defaults, type Env } from "../env";
import { resolvePlain } from "../envvars";
import { cpuPrice, runpod } from "../runpod";
import { fetchWithTimeout, HttpError, now } from "../util";
import {
  canonical,
  isDirect,
  workerCreatePayload,
  workerPlacements,
  workerSystemEnv,
  type ClusterSecrets,
  type EdgeCfg,
  type EnvCtx,
  type PodRec,
} from "./payloads";
import { isEdge, REGIONS, type ClusterSpec, type PoolSpec } from "./spec";
import { isStandalone, podUpdate, recordPod, saveSecrets, saveState, secretsOf, type Cluster } from "./store";

export type Logf = (msg: string) => void;

const stamp = () => new Date().toISOString().replace(/[-:T]/g, "").slice(4, 14); // MMDDHHMMSS

export function ingestUrl(env: Env): string | undefined {
  return env.PUBLIC_URL ? `${env.PUBLIC_URL.replace(/\/$/, "")}/ingest/v1/logs` : undefined;
}
/** The edge Worker of control_plane = edge clusters (fv-control's EDGE_* settings); undefined when they are not set. */
export function edgeCfg(env: Env): EdgeCfg | undefined {
  if (!env.EDGE_URL || !env.EDGE_INTERNAL_TOKEN || !env.EDGE_ADMIN_TOKEN) return undefined;
  return { url: env.EDGE_URL.replace(/\/$/, ""), internal_token: env.EDGE_INTERNAL_TOKEN, admin_token: env.EDGE_ADMIN_TOKEN, d1_database_id: env.EDGE_D1_DATABASE_ID || undefined, outputs_bucket: env.EDGE_OUTPUTS_BUCKET || undefined };
}
export function requireEdge(env: Env): EdgeCfg {
  const e = edgeCfg(env);
  if (!e) throw new HttpError(409, "control_plane = edge: fv-control has no edge (EDGE_URL, EDGE_INTERNAL_TOKEN and EDGE_ADMIN_TOKEN)");
  return e;
}
export async function envCtx(env: Env, c: Cluster, secrets?: ClusterSecrets): Promise<EnvCtx> {
  return {
    edge: isEdge(c.spec) ? edgeCfg(env) : undefined,
    spec: c.spec,
    state: c.state,
    secrets: secrets ?? (await secretsOf(env, c)),
    deadlineMs: c.deadline ?? now(),
    runpodApiKey: env.RUNPOD_API_KEY,
    ingestUrl: ingestUrl(env),
    backstop: isStandalone(c) && !isEdge(c.spec),
  };
}
/** The Runpod pod name: fv-ctl-<cluster>-<pool>-<stamp>, or fv-pod-<name>-<stamp> for a standalone pod. */
export const runpodName = (c: Cluster, poolId: string) => (isStandalone(c) ? `fv-pod-${c.name}-${stamp()}` : `fv-ctl-${c.name}-${poolId}-${stamp()}`);
const poolOf = (c: Cluster, id: string): PoolSpec => {
  const p = c.spec.pools.find((x) => x.id === id);
  if (!p) throw new HttpError(404, `no pool ${id} in ${c.name}`);
  return p;
};

/** The full env a worker gets (system < account < cluster < pool < pod) and its hash. */
export async function desiredEnv(env: Env, c: Cluster, ctx: EnvCtx, _role: "worker", rec: { pod?: string; pool?: string; image: string }) {
  const system = workerSystemEnv(ctx, poolOf(c, rec.pool!), rec.image);
  const full = await resolvePlain(env, c.id, rec.pod ?? null, system, rec.pool ?? null);
  return { system, full, hash: (await sha256Hex(canonical(full))).slice(0, 16) };
}

// ---------------- price and balance guards
export interface Projection {
  ok: boolean;
  reasons: string[];
  balance: number;
  account_spend_per_hr: number;
  cluster_dph: number;
  hours: number;
  projected_balance: number;
  floor: number;
  pods: { role: string; pool?: string; what: string; dph: number | null }[];
}
/** Price check before a start (or an extend / scale-up): the cluster's $/hr, and the balance at the deadline with the whole account's burn. */
export async function projectSpend(env: Env, spec: ClusterSpec, opts: { hours: number; extraOnly?: { pool: string; count: number }; runningDph?: number }): Promise<Projection> {
  const acct = await runpod.account(env);
  const pods: Projection["pods"] = [];
  const priceCache = new Map<string, number | null>();
  const gpuPrice = async (g: string) => {
    if (!priceCache.has(g)) priceCache.set(g, (await runpod.gpuPrice(env, g).catch(() => ({ price: null }))).price);
    return priceCache.get(g)!;
  };
  for (const p of spec.pools) {
    const n = opts.extraOnly ? (opts.extraOnly.pool === p.id ? opts.extraOnly.count : 0) : p.count;
    for (let i = 0; i < n; i++) {
      if (p.compute === "CPU") {
        const f = p.cpu_flavors?.[0] || "cpu3c";
        pods.push({ role: "worker", pool: p.id, what: `cpu:${f}`, dph: cpuPrice(f, p.vcpu ?? 2) });
      } else {
        // The most expensive GPU the pool may land on (the create order tries cheaper regions first, but plan for the worst).
        const regions = p.regions?.length ? p.regions : spec.regions;
        const gpus = p.gpu_types?.length ? p.gpu_types : regions.flatMap((r) => REGIONS[r]!.gpus);
        let worst: number | null = null;
        for (const g of gpus) {
          const pr = await gpuPrice(g);
          if (pr !== null && pr <= spec.max_gpu_dph) worst = Math.max(worst ?? 0, pr);
        }
        pods.push({ role: "worker", pool: p.id, what: gpus[0] || "gpu", dph: worst ?? spec.max_gpu_dph });
      }
    }
  }
  const cluster_dph = pods.reduce((s, p) => s + (p.dph ?? 0), 0);
  // The account's burn already includes the running pods of this cluster.
  // The account's burn includes this cluster's running pods (runningDph) when it replans a running cluster.
  const burn = Math.max(0, acct.spendPerHr - (opts.runningDph ?? 0)) + cluster_dph;
  const projected = acct.balance - burn * opts.hours;
  const floor = Math.max(spec.balance_floor, defaults.balanceFloor(env));
  const reasons: string[] = [];
  if (!opts.extraOnly && acct.balance < spec.min_start) reasons.push(`balance $${acct.balance.toFixed(2)} is below the start minimum $${spec.min_start}`);
  if (projected < floor)
    reasons.push(`at $${burn.toFixed(2)}/hr (account $${acct.spendPerHr.toFixed(2)} + cluster $${cluster_dph.toFixed(2)}) the balance would be $${projected.toFixed(2)} after ${opts.hours.toFixed(2)} h, below the floor $${floor}`);
  return { ok: reasons.length === 0, reasons, balance: acct.balance, account_spend_per_hr: acct.spendPerHr, cluster_dph, hours: opts.hours, projected_balance: projected, floor, pods };
}

// ---------------- create / patch / delete
/** One worker for a pool into `slot` (workers | rolling); null when no stock anywhere. */
export async function createWorker(env: Env, c: Cluster, poolId: string, image: string, slot: "workers" | "rolling", log: Logf): Promise<PodRec | null> {
  const pool = poolOf(c, poolId);
  const ctx = await envCtx(env, c);
  const { full, hash } = await desiredEnv(env, c, ctx, "worker", { pool: poolId, image });
  for (const pl of workerPlacements(c.spec, pool)) {
    const payload = workerCreatePayload(runpodName(c, poolId), image, pool, pl, full);
    let r: any;
    try {
      r = await runpod.create(env, payload);
    } catch (e) {
      log(`${poolId}: no ${pl.gpu || pl.cpu} in ${pl.dc || "any DC"}: ${(e as Error).message.slice(0, 160)}`);
      continue;
    }
    if (!r?.id) {
      log(`${poolId}: create returned no id`);
      continue;
    }
    const dph = Number(r.costPerHr ?? 0);
    const rec: PodRec = { pod: r.id, pool: poolId, gpu: pl.gpu, cpu: pl.cpu, dc: pl.dc || r.machine?.dataCenterId, dph, created: Math.floor(now() / 1000), image, url: defaults.podUrl(env, r.id) };
    const cap = pool.compute === "CPU" ? 1.0 : c.spec.max_gpu_dph;
    if (dph > cap) {
      log(`${poolId}: ${r.id} costs $${dph}/hr > $${cap}: deleting`);
      await runpod.remove(env, r.id).catch(() => {});
      continue;
    }
    const target = slot === "rolling" ? (c.state.rolling ||= {}) : c.state.workers;
    (target[poolId] ||= []).push(rec);
    await saveState(env, c);
    await recordPod(env, c, rec, "worker", slot, hash);
    log(`${poolId}: pod ${rec.pod} on ${pl.gpu || `cpu:${pl.cpu}`} in ${rec.dc || "?"} at $${dph}/hr`);
    return rec;
  }
  log(`${poolId}: no stock in [${(pool.regions?.length ? pool.regions : c.spec.regions).join(" ")}]`);
  return null;
}

/** PATCH a worker's env (a restart); keeps its image unless one is given. */
export async function patchWorker(env: Env, c: Cluster, rec: PodRec, log: Logf, image?: string): Promise<void> {
  const ctx = await envCtx(env, c);
  const { full, hash } = await desiredEnv(env, c, ctx, "worker", { pod: rec.pod, pool: rec.pool, image: image || rec.image });
  await runpod.patch(env, rec.pod, { env: full, ...(image ? { imageName: image } : {}) });
  if (image) rec.image = image;
  await saveState(env, c);
  await podUpdate(env, rec.pod, { env_hash: hash, image });
  log(`${rec.pool}: ${rec.pod} env applied; container restarts`);
}

export async function deletePod(env: Env, podId: string, log: Logf, why: string): Promise<boolean> {
  try {
    await runpod.remove(env, podId);
    await podUpdate(env, podId, { deleted: true });
    log(`deleted ${podId} (${why})`);
    return true;
  } catch (e) {
    log(`WARNING: delete of ${podId} failed: ${(e as Error).message.slice(0, 160)}`);
    return false;
  }
}

// ---------------- the edge and the admin token
/** A request to the edge: through the EDGE service binding when there is one (a
 * Worker cannot fetch another workers.dev Worker of the same account, error
 * 1042), else over the Internet (tests, a custom domain). */
export async function edgeFetch(env: Env, url: string, init: RequestInit & { timeoutMs?: number } = {}): Promise<Response> {
  if (!env.EDGE) return fetchWithTimeout(url, init);
  const { timeoutMs, ...rest } = init;
  return env.EDGE.fetch(url, { ...rest, signal: AbortSignal.timeout(timeoutMs ?? 20000) } as RequestInit);
}
/** Where clients and admin calls go: the edge (edge clusters; direct workers have no single front). */
export function frontUrl(env: Env, c: Cluster): string | undefined {
  return isEdge(c.spec) ? edgeCfg(env)?.url : undefined;
}
async function edgeGet(env: Env, c: Cluster, path: string, init: RequestInit & { timeoutMs?: number } = {}) {
  if (!isEdge(c.spec)) throw new HttpError(409, `${c.name} is a direct cluster: it has no edge`);
  return edgeFetch(env, `${requireEdge(env).url}${path}`, { timeoutMs: 20000, ...init });
}
/** The start of adminToken's error when a direct cluster had no token yet (made before the controller made one). */
export const DIRECT_ADMIN_NEW = "this gateway-less cluster had no admin token";
export const newAdminToken = () => `fvadm_${randomToken("", 24)}`;
/** The admin token: the edge's (edge clusters), or the one the controller makes and passes to every direct worker as FV_ADMIN_TOKEN (docs/control/gateway-less-auth.md). */
export async function adminToken(env: Env, c: Cluster): Promise<string> {
  if (isEdge(c.spec)) return requireEdge(env).admin_token;
  const s = await secretsOf(env, c);
  if (s.admin_token) return s.admin_token;
  s.admin_token = newAdminToken();
  await saveSecrets(env, c, s);
  throw new HttpError(409, `${DIRECT_ADMIN_NEW}; the controller made one: restart the workers (Env: apply) to use it`);
}
export async function adminGet(env: Env, c: Cluster, path: string): Promise<any> {
  const tok = await adminToken(env, c);
  const r = await edgeGet(env, c, path, { headers: { authorization: `Bearer ${tok}` } });
  if (!r.ok) throw new HttpError(502, `edge ${path}: ${r.status}`);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
// ---------------- admin calls: the edge, or every direct worker
export interface AdminTarget {
  url: string;
  pod: string;
  pool?: string;
}
/** Where the admin routes (/fv/v1/admin/*) are: the edge, or every worker of a direct cluster. */
export function adminTargets(c: Cluster, env?: Env): AdminTarget[] {
  if (isEdge(c.spec)) {
    const e = env ? edgeCfg(env) : undefined;
    if (!e) throw new HttpError(409, "control_plane = edge: fv-control has no edge (EDGE_URL, EDGE_INTERNAL_TOKEN and EDGE_ADMIN_TOKEN)");
    return [{ url: e.url, pod: "edge" }];
  }
  const out: AdminTarget[] = [];
  for (const [pool, recs] of Object.entries(c.state.workers || {})) for (const r of recs) if (r.url) out.push({ url: r.url.replace(/\/$/, ""), pod: r.pod, pool });
  if (!out.length) throw new HttpError(409, `${c.name} has no workers`);
  return out;
}
export interface AdminReply {
  pod: string;
  status: number;
  body: any;
}
async function adminCall(env: Env, tok: string, t: AdminTarget, method: string, path: string, body?: unknown): Promise<AdminReply> {
  try {
    const r = await (t.pod === "edge" ? edgeFetch : (_e: Env, u: string, i: RequestInit & { timeoutMs?: number }) => fetchWithTimeout(u, i))(env, `${t.url}${path}`, {
      method,
      headers: { authorization: `Bearer ${tok}`, ...(body === undefined ? {} : { "content-type": "application/json" }) },
      body: body === undefined ? undefined : JSON.stringify(body),
      timeoutMs: 20000,
    });
    const text = await r.text();
    let j: any = text;
    try {
      j = JSON.parse(text);
    } catch {
      /* not JSON */
    }
    return { pod: t.pod, status: r.status, body: j };
  } catch (e) {
    return { pod: t.pod, status: 0, body: { error: (e as Error).message.slice(0, 160) } };
  }
}
/** One admin call: the edge, or the first direct worker that answers (the keys are in D1, shared by all of them). */
export async function adminOne(env: Env, c: Cluster, method: string, path: string, body?: unknown): Promise<AdminReply> {
  const tok = await adminToken(env, c);
  let last: AdminReply | null = null;
  for (const t of adminTargets(c, env)) {
    last = await adminCall(env, tok, t, method, path, body);
    if (last.status !== 0 && last.status < 500 && last.status !== 401) return last;
  }
  return last!;
}
/** The same admin call on every target (a revocation applies on each worker at once instead of at its next D1 refresh, ≤ 30 s). */
export async function adminAll(env: Env, c: Cluster, method: string, path: string): Promise<AdminReply[]> {
  const tok = await adminToken(env, c);
  return Promise.all(adminTargets(c, env).map((t) => adminCall(env, tok, t, method, path)));
}

/** A public route of the edge (its status). */
export async function edgePublic(env: Env, c: Cluster, path: string): Promise<{ status: number; body: any }> {
  const r = await edgeGet(env, c, path);
  const t = await r.text();
  try {
    return { status: r.status, body: JSON.parse(t) };
  } catch {
    return { status: r.status, body: t };
  }
}

/** A worker's /health (public): state AVAILABLE when ready; build.image.digest. */
export async function workerHealth(env: Env, podId: string): Promise<{ ok: boolean; state?: string; digest?: string; sha?: string; code: number }> {
  try {
    const r = await fetchWithTimeout(`${defaults.podUrl(env, podId)}/health`, { timeoutMs: 15000 });
    const j: any = await r.json().catch(() => ({}));
    return { ok: r.ok && j.state === "AVAILABLE", state: j.state, digest: j.build?.image?.digest, sha: j.build?.git_sha, code: r.status };
  } catch {
    return { ok: false, code: 0 };
  }
}
/** Worker internal routes (drain / status) with the internal token (never logged). */
export async function workerInternal(env: Env, c: Cluster, podId: string, method: "GET" | "POST", path: string): Promise<any> {
  // Edge fronts take the edge's internal token, not the cluster's.
  const token = isEdge(c.spec) ? requireEdge(env).internal_token : (await secretsOf(env, c)).internal_token;
  const r = await fetchWithTimeout(`${defaults.podUrl(env, podId)}${path}`, { method, headers: { "x-fv-internal-token": token }, timeoutMs: 20000 });
  if (!r.ok) throw new HttpError(502, `${podId} ${path}: ${r.status}`);
  return r.json().catch(() => ({}));
}
export async function workerBusy(env: Env, c: Cluster, podId: string): Promise<number> {
  try {
    const j = await workerInternal(env, c, podId, "GET", "/fv/v1/internal/status");
    const st = j.stats || {};
    return (st.running || 0) + (st.queued_batch || 0) + (st.queued_stream || 0) + (st.sessions || 0);
  } catch {
    return 0;
  }
}

/** The edge's families view (admin token): each family's workers (worker_id = the pod id) and their readiness. */
export async function edgeFamilies(env: Env): Promise<any> {
  const e = requireEdge(env);
  const r = await edgeFetch(env, `${e.url}/fv/v1/edge/families`, { headers: { authorization: `Bearer ${e.admin_token}` }, timeoutMs: 20000 });
  if (!r.ok) throw new HttpError(502, `edge /fv/v1/edge/families: ${r.status}`);
  return r.json();
}
/** Pod id → its edge view (ready, jobs held) from the families view. */
export function edgeWorkers(view: any): Map<string, { ready: boolean; held: number; families: string[]; sha?: string }> {
  const out = new Map<string, { ready: boolean; held: number; families: string[]; sha?: string }>();
  for (const [fam, st] of Object.entries<any>(view?.families || {})) {
    for (const w of st?.workers || []) {
      const id = String(w.worker_id || "");
      if (!id) continue;
      const cur = out.get(id) || { ready: false, held: 0, families: [] as string[], sha: w.sha || undefined };
      cur.ready = cur.ready || (!!w.connected && w.ready !== false && !w.draining && !!w.front && w.front.ready !== false);
      cur.held += Number(w.held || 0);
      cur.families.push(fam);
      out.set(id, cur);
    }
  }
  return out;
}
