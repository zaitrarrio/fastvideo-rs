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
  workerCreatePayload,
  workerPlacements,
  workerSystemEnv,
  type ClusterSecrets,
  type EnvCtx,
  type PodRec,
} from "./payloads";
import { REGIONS, type ClusterSpec, type PoolSpec } from "./spec";
import { podUpdate, recordPod, saveSecrets, saveState, secretsOf, type Cluster } from "./store";

export type Logf = (msg: string) => void;

const stamp = () => new Date().toISOString().replace(/[-:T]/g, "").slice(4, 14); // MMDDHHMMSS

export function ingestUrl(env: Env): string | undefined {
  return env.PUBLIC_URL ? `${env.PUBLIC_URL.replace(/\/$/, "")}/ingest/v1/logs` : undefined;
}
export async function envCtx(env: Env, c: Cluster, secrets?: ClusterSecrets): Promise<EnvCtx> {
  return {
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

/** The full env a pod gets (system < account < cluster < pod) and its hash. */
export async function desiredEnv(env: Env, c: Cluster, ctx: EnvCtx, role: "gateway" | "worker", rec: { pod?: string; pool?: string; image: string }) {
  const system = role === "gateway" ? gatewaySystemEnv(ctx, rec.image) : workerSystemEnv(ctx, poolOf(c, rec.pool!), rec.image);
  const full = await resolvePlain(env, c.id, rec.pod ?? null, system);
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
async function gatewayFetch(env: Env, c: Cluster, path: string, init: RequestInit & { timeoutMs?: number } = {}) {
  if (!c.state.gateway_url) throw new HttpError(409, `${c.name} has no gateway`);
  return fetchWithTimeout(`${c.state.gateway_url}${path}`, { timeoutMs: 20000, ...init });
}
export async function adminToken(env: Env, c: Cluster): Promise<string> {
  const s = await secretsOf(env, c);
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
      s.admin_token = `fvadm_${randomToken("", 24)}`;
      s.legacy_admin_token = true;
      await saveSecrets(env, c, s);
    }
    throw new HttpError(409, "this gateway image has no sealed admin token route; the cluster now passes FV_ADMIN_TOKEN: restart the gateway (Env: apply) to use it");
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
    const r = await gatewayFetch(env, c, "/healthz", { timeoutMs: 15000 });
    return r.ok;
  } catch {
    return false;
  }
}
/** Worker internal routes (drain / status) with the internal token (never logged). */
export async function workerInternal(env: Env, c: Cluster, podId: string, method: "GET" | "POST", path: string): Promise<any> {
  const s = await secretsOf(env, c);
  const r = await fetchWithTimeout(`${defaults.podUrl(env, podId)}${path}`, { method, headers: { "x-fv-internal-token": s.internal_token }, timeoutMs: 20000 });
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
