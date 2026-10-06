// Cluster primitives (each one short enough for one Durable Object alarm):
// a port of runpod-cluster.sh's create_gateway, create_worker,
// patch_gateway, admin_token, the price and balance guards, and the
// gateway / worker probes.
import { openSealedToken, randomToken, sha256Hex, type SealedToken } from "../crypto";
import { defaults, type Env } from "../env";
import { resolvePlain } from "../envvars";
import { cpuPrice, runpod } from "../runpod";
import { fetchWithTimeout, HttpError, now } from "../util";
import {
  canonical,
  gatewayCreatePayload,
  gatewaySystemEnv,
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
import { podUpdate, recordPod, saveSecrets, saveState, secretsOf, type Cluster } from "./store";

export type Logf = (msg: string) => void;

const stamp = () => new Date().toISOString().replace(/[-:T]/g, "").slice(4, 14); // MMDDHHMMSS

export function ingestUrl(env: Env): string | undefined {
  return env.PUBLIC_URL ? `${env.PUBLIC_URL.replace(/\/$/, "")}/ingest/v1/logs` : undefined;
}
/** The edge Worker of control_plane = edge clusters (fv-control's EDGE_* settings); undefined when they are not set. */
export function edgeCfg(env: Env): EdgeCfg | undefined {
  if (!env.EDGE_URL || !env.EDGE_INTERNAL_TOKEN || !env.EDGE_ADMIN_TOKEN) return undefined;
  return { url: env.EDGE_URL.replace(/\/$/, ""), internal_token: env.EDGE_INTERNAL_TOKEN, admin_token: env.EDGE_ADMIN_TOKEN, d1_database_id: env.EDGE_D1_DATABASE_ID || undefined };
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
    githubPat: env.GITHUB_PAT,
    ingestUrl: ingestUrl(env),
  };
}
const poolOf = (c: Cluster, id: string): PoolSpec => {
  const p = c.spec.pools.find((x) => x.id === id);
  if (!p) throw new HttpError(404, `no pool ${id} in ${c.name}`);
  return p;
};

/** The full env a pod gets (system < account < cluster < pool < pod; the gateway has no pool layer) and its hash. */
export async function desiredEnv(env: Env, c: Cluster, ctx: EnvCtx, role: "gateway" | "worker", rec: { pod?: string; pool?: string; image: string }) {
  const system = role === "gateway" ? gatewaySystemEnv(ctx, rec.image) : workerSystemEnv(ctx, poolOf(c, rec.pool!), rec.image);
  const full = await resolvePlain(env, c.id, rec.pod ?? null, system, role === "worker" ? rec.pool : null);
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
  if (!opts.extraOnly && spec.gateway.enabled) pods.push({ role: "gateway", what: `cpu:${spec.gateway.cpu_flavors[0]}`, dph: cpuPrice(spec.gateway.cpu_flavors[0] || "cpu3c", spec.gateway.vcpu) });
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
export async function createGateway(env: Env, c: Cluster, log: Logf): Promise<PodRec> {
  const ctx = await envCtx(env, c);
  const image = c.state.images.gateway || c.state.image!;
  const { full, hash } = await desiredEnv(env, c, ctx, "gateway", { image });
  const dc = REGIONS[c.spec.regions[0]!]?.dc;
  let lastErr = "";
  for (const dcs of [dc ? [dc] : null, null]) {
    for (const flavor of c.spec.gateway.cpu_flavors) {
      const payload = gatewayCreatePayload(`fv-ctl-${c.name}-gw-${stamp()}`, image, flavor, c.spec.gateway.vcpu, c.spec.gateway.container_disk_gb, dcs, full);
      try {
        const r = await runpod.create(env, payload);
        if (!r?.id) continue;
        const rec: PodRec = {
          pod: r.id,
          cpu: flavor,
          dph: Number(r.costPerHr ?? 0),
          created: Math.floor(now() / 1000),
          dc: r.machine?.dataCenterId || r.dataCenterId || dcs?.[0],
          image,
          url: defaults.podUrl(env, r.id),
        };
        c.state.gateway = rec;
        c.state.gateway_url = rec.url;
        await saveState(env, c);
        await recordPod(env, c, rec, "gateway", "gateway", hash);
        log(`gateway pod ${rec.pod} (${flavor}, $${rec.dph}/hr)`);
        return rec;
      } catch (e) {
        lastErr = (e as Error).message;
        log(`no gateway on ${flavor} in ${dcs ? dcs.join(",") : "any DC"}: ${lastErr.slice(0, 160)}`);
      }
    }
  }
  throw new HttpError(503, `could not create the gateway pod: ${lastErr.slice(0, 200)}`);
}

/** One worker for a pool into `slot` (workers | rolling); null when no stock anywhere. */
export async function createWorker(env: Env, c: Cluster, poolId: string, image: string, slot: "workers" | "rolling", log: Logf): Promise<PodRec | null> {
  const pool = poolOf(c, poolId);
  const ctx = await envCtx(env, c);
  const { full, hash } = await desiredEnv(env, c, ctx, "worker", { pool: poolId, image });
  for (const pl of workerPlacements(c.spec, pool)) {
    const payload = workerCreatePayload(`fv-ctl-${c.name}-${poolId}-${stamp()}`, image, pool, pl, full);
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

/** PATCH the gateway's env (worker URLs, pod list, deadline, user env) and optionally its image; its container restarts. */
export async function patchGateway(env: Env, c: Cluster, log: Logf, image?: string): Promise<void> {
  const g = c.state.gateway;
  if (!g) return;
  const img = image || g.image;
  const ctx = await envCtx(env, c);
  const { full, hash } = await desiredEnv(env, c, ctx, "gateway", { pod: g.pod, image: img });
  await runpod.patch(env, g.pod, { env: full, ...(image ? { imageName: image } : {}) });
  if (image) {
    g.image = image;
    c.state.images.gateway = image;
    await saveState(env, c);
  }
  await podUpdate(env, g.pod, { env_hash: hash, image });
  log(`gateway: env applied (pools ${Object.keys(c.state.workers).join(", ") || "none"})${image ? `, image ${image}` : ""}; container restarts`);
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

// ---------------- the gateway's admin token (gateway.md §9)
/** Where clients and admin calls go: the edge (control_plane = edge) or the gateway. */
export function frontUrl(env: Env, c: Cluster): string | undefined {
  return isEdge(c.spec) ? edgeCfg(env)?.url : c.state.gateway_url;
}
async function gatewayFetch(env: Env, c: Cluster, path: string, init: RequestInit & { timeoutMs?: number } = {}) {
  const base = isEdge(c.spec) ? requireEdge(env).url : c.state.gateway_url;
  if (!base) throw new HttpError(409, `${c.name} has no gateway`);
  return fetchWithTimeout(`${base}${path}`, { timeoutMs: 20000, ...init });
}
/** The start of adminToken's error when it has just switched an older gateway image to FV_ADMIN_TOKEN. */
export const LEGACY_ADMIN_SWITCH = "this gateway image has no sealed admin token route";
/** The start of adminToken's error when a gateway-less cluster had no token yet (launched before the controller made one). */
export const DIRECT_ADMIN_NEW = "this gateway-less cluster had no admin token";
export const newAdminToken = () => `fvadm_${randomToken("", 24)}`;
export async function adminToken(env: Env, c: Cluster): Promise<string> {
  // The edge's admin token: keys and the families view are the edge's.
  if (isEdge(c.spec)) return requireEdge(env).admin_token;
  const s = await secretsOf(env, c);
  if (isDirect(c.spec, c.state)) {
    // No gateway: the controller makes the token and passes it to every
    // worker as FV_ADMIN_TOKEN (docs/control/gateway-less-auth.md).
    if (s.admin_token) return s.admin_token;
    s.admin_token = newAdminToken();
    await saveSecrets(env, c, s);
    throw new HttpError(409, `${DIRECT_ADMIN_NEW}; the controller made one: restart the workers (Env: apply) to use it`);
  }
  if (s.admin_token) {
    const r = await gatewayFetch(env, c, "/fv/v1/gateway/pools", { headers: { authorization: `Bearer ${s.admin_token}` } }).catch(() => null);
    if (!(r && r.status === 401 && s.admin_recipient && s.admin_private)) return s.admin_token;
  }
  if (!s.admin_private || !s.admin_recipient) throw new HttpError(409, "no admin key pair for this cluster (imported without its .admin-key.pem)");
  const r = await gatewayFetch(env, c, "/fv/v1/admin/token/sealed");
  if (r.status === 404 && (await gatewayHealthy(env, c))) {
    // A gateway image older than the sealed-token route (gateway.md §9) is up but
    // keeps its own token where nobody can read it: switch the cluster to a token
    // the controller makes and passes as FV_ADMIN_TOKEN (sealed in D1, masked in
    // every view), as runpod-cluster.sh did before. It applies on the gateway's
    // next restart (the env view shows it as needing one).
    if (!s.legacy_admin_token) {
      s.admin_token = newAdminToken();
      s.legacy_admin_token = true;
      await saveSecrets(env, c, s);
    }
    throw new HttpError(409, `${LEGACY_ADMIN_SWITCH}; the cluster now passes FV_ADMIN_TOKEN: restart the gateway (Env: apply) to use it`);
  }
  if (!r.ok) throw new HttpError(503, `the gateway did not publish its sealed admin token (${r.status}; not up yet?)`);
  const tok = await openSealedToken((await r.json()) as SealedToken, s.admin_private, s.admin_recipient);
  if (!tok.startsWith("fvadm_")) throw new HttpError(502, "the sealed admin token does not look like one");
  s.admin_token = tok;
  await saveSecrets(env, c, s);
  return tok;
}
export async function adminGet(env: Env, c: Cluster, path: string): Promise<any> {
  const tok = await adminToken(env, c);
  const r = await gatewayFetch(env, c, path, { headers: { authorization: `Bearer ${tok}` } });
  if (!r.ok) throw new HttpError(502, `gateway ${path}: ${r.status}`);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

// ---------------- admin calls on the gateway, or on the workers when there is none
export interface AdminTarget {
  url: string;
  pod: string;
  pool?: string;
}
/** Where the admin routes (/fv/v1/admin/*) are: the gateway, or every worker of a gateway-less cluster. */
export function adminTargets(c: Cluster, env?: Env): AdminTarget[] {
  if (isEdge(c.spec)) {
    const e = env ? edgeCfg(env) : undefined;
    if (!e) throw new HttpError(409, "control_plane = edge: fv-control has no edge (EDGE_URL, EDGE_INTERNAL_TOKEN and EDGE_ADMIN_TOKEN)");
    return [{ url: e.url, pod: "edge" }];
  }
  if (c.state.gateway_url && c.state.gateway) return [{ url: c.state.gateway_url, pod: c.state.gateway.pod }];
  if (!isDirect(c.spec, c.state)) throw new HttpError(409, `${c.name} has no gateway`);
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
async function adminCall(tok: string, t: AdminTarget, method: string, path: string, body?: unknown): Promise<AdminReply> {
  try {
    const r = await fetchWithTimeout(`${t.url}${path}`, {
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
/** One admin call: the gateway, or the first worker that answers (the keys are in D1, shared by all of them). */
export async function adminOne(env: Env, c: Cluster, method: string, path: string, body?: unknown): Promise<AdminReply> {
  const tok = await adminToken(env, c);
  let last: AdminReply | null = null;
  for (const t of adminTargets(c, env)) {
    last = await adminCall(tok, t, method, path, body);
    if (last.status !== 0 && last.status < 500 && last.status !== 401) return last;
  }
  return last!;
}
/** The same admin call on every target (a revocation applies on each worker at once instead of at its next D1 refresh, ≤ 30 s). */
export async function adminAll(env: Env, c: Cluster, method: string, path: string): Promise<AdminReply[]> {
  const tok = await adminToken(env, c);
  return Promise.all(adminTargets(c, env).map((t) => adminCall(tok, t, method, path)));
}

export async function gatewayPublic(env: Env, c: Cluster, path: string): Promise<{ status: number; body: any }> {
  const r = await gatewayFetch(env, c, path);
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
export async function gatewayHealthy(env: Env, c: Cluster): Promise<boolean> {
  try {
    // The edge answers /healthz 503 until a front is ready: its status route says it is up.
    const r = await gatewayFetch(env, c, isEdge(c.spec) ? "/fv/v1/status" : "/healthz", { timeoutMs: 15000 });
    return r.ok;
  } catch {
    return false;
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
  const r = await fetchWithTimeout(`${e.url}/fv/v1/edge/families`, { headers: { authorization: `Bearer ${e.admin_token}` }, timeoutMs: 20000 });
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
