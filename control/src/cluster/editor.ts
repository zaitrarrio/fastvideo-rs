// The cluster configuration editor's checks (docs/control/README.md §4):
// every problem of a draft spec at once, with the path of the field it is
// about (the same zod schema and normalizeSpec every write path uses), and a
// stock hint per pool from Runpod (GPU type × data centre) so a launch that
// would find no stock shows before Start.
import type { Env } from "../env";
import { runpod } from "../runpod";
import { validate, type Issue } from "../schemas";
import { HttpError } from "../util";
import { workerPlacements } from "./payloads";
import { gpuTypes } from "../dynamic";
import { normalizeSpec, POOL_PRESETS, REGIONS, type ClusterSpec } from "./spec";

/**
 * Checks that need live data (the Runpod GPU catalog, 5-minute cache): a
 * GPU type Runpod does not offer in Secure Cloud (fv-control creates SECURE
 * pods), and a $/hr cap below every GPU type a pool may use (each pod would
 * be deleted right after create). None when the catalog is unreachable.
 */
export async function liveSpecIssues(env: Env, spec: ClusterSpec, prefix: (string | number)[] = []): Promise<{ issues: Issue[]; warnings: Issue[] }> {
  const issues: Issue[] = [];
  const warnings: Issue[] = [];
  const cat = await gpuTypes(env).catch(() => []);
  if (!cat.length) return { issues, warnings };
  const price = new Map(cat.map((g) => [g.id, g.secure_price]));
  spec.pools.forEach((p, i) => {
    if (p.compute !== "GPU") return;
    const regions = p.regions?.length ? p.regions : spec.regions;
    const gpus = p.gpu_types?.length ? p.gpu_types : [...new Set(regions.flatMap((r) => REGIONS[r]?.gpus || []))];
    (p.gpu_types || []).forEach((g, j) => {
      if (price.has(g) && price.get(g) === null) issues.push({ path: [...prefix, "pools", i, "gpu_types", j], message: `${g} is not offered in Secure Cloud (fv-control creates Secure pods)` });
    });
    const known = gpus.map((g) => price.get(g)).filter((x): x is number => typeof x === "number");
    if (!known.length || !p.count) return;
    const cheapest = Math.min(...known);
    if (cheapest > spec.max_gpu_dph) issues.push({ path: [...prefix, "max_gpu_dph"], message: `below every GPU type pool ${p.id} may use (cheapest $${cheapest.toFixed(2)}/hr): each pod would be deleted right after create` });
    else if (Math.max(...known) > spec.max_gpu_dph) warnings.push({ path: [...prefix, "pools", i, "gpu_types"], message: `some GPU types of ${p.id} cost more than max_gpu_dph ($${spec.max_gpu_dph}/hr): a pod placed on one is deleted` });
  });
  // Dedupe (one max_gpu_dph message per pool is enough, the first one per path).
  const seen = new Set<string>();
  return { issues: issues.filter((x) => (seen.has(x.path.join(".") + x.message) ? false : (seen.add(x.path.join(".") + x.message), true))), warnings };
}

/** "pools[2].models[0].recipe: …" → ["pools", 2, "models", 0, "recipe"]. */
export function pathOf(msg: string): { path: (string | number)[]; message: string } {
  const m = /^([A-Za-z_][\w]*(?:\[\d+\]|\.[A-Za-z_][\w]*)*): (.*)$/s.exec(msg);
  if (!m) return { path: [], message: msg };
  const path: (string | number)[] = [];
  for (const part of m[1]!.matchAll(/([A-Za-z_]\w*)|\[(\d+)\]/g)) path.push(part[1] !== undefined ? part[1] : Number(part[2]));
  return { path, message: m[2]! };
}

export interface SpecCheck {
  ok: boolean;
  issues: Issue[];
  warnings: Issue[];
  normalized?: ClusterSpec;
}
/**
 * Every problem of a draft: the schema's issues, then normalizeSpec's (a
 * region without a volume, a recipe the catalog cannot serve, …), then the
 * name (fixed for an existing cluster, unique for a new one). Warnings do
 * not block a save.
 */
export async function checkSpec(env: Env, doc: unknown, existing?: { id: string; name: string }): Promise<SpecCheck> {
  const issues: Issue[] = [];
  const seen = new Set<string>();
  const add = (i: Issue) => {
    const k = `${i.path.join(".")}|${i.message}`;
    if (!seen.has(k)) seen.add(k), issues.push(i);
  };
  const v = validate("cluster-spec", doc);
  if (!v.ok) v.issues.forEach(add);
  let normalized: ClusterSpec | undefined;
  try {
    normalized = normalizeSpec(doc);
  } catch (e) {
    const ex = e as HttpError;
    const extra = ex.extra?.issues as Issue[] | undefined;
    if (extra) extra.forEach(add);
    else add(pathOf(ex.message));
  }
  const name = (doc as any)?.name;
  if (existing && name !== existing.name) add({ path: ["name"], message: `the name is fixed (${existing.name}); clone the cluster to use another name` });
  if (!existing && typeof name === "string" && name && !issues.some((i) => i.path[0] === "name")) {
    const taken = await env.DB.prepare("SELECT id FROM clusters WHERE name = ?").bind(name).first<{ id: string }>();
    if (taken) add({ path: ["name"], message: `a cluster or standalone pod named ${name} exists` });
  }
  const warnings: Issue[] = [];
  if (normalized) {
    const live = await liveSpecIssues(env, normalized);
    live.issues.forEach(add);
    warnings.push(...live.warnings);
  }
  if (normalized) {
    const s = normalized;
    if (!s.pools.length) warnings.push({ path: ["pools"], message: "no pools: Start would create nothing" });
    s.pools.forEach((p, i) => {
      if (p.count === 0) warnings.push({ path: ["pools", i, "count"], message: `${p.id}: 0 workers (Start skips it; Scale adds workers later)` });
      const pre = POOL_PRESETS.find((x) => x.id === p.id);
      if (pre?.licence) warnings.push({ path: ["pools", i, "id"], message: pre.licence });
    });
    if (s.cap_s > 6 * 3600) warnings.push({ path: ["cap_s"], message: `the backstop is ${(s.cap_s / 3600).toFixed(1)} h away from Start` });
    if (!s.log_shipping) warnings.push({ path: ["log_shipping"], message: "log shipping off: only Runpod's short tail of each pod will be visible" });
    if (s.min_balance > s.balance_floor + 50) warnings.push({ path: ["min_balance"], message: "workers delete themselves well above the floor" });
  }
  return { ok: issues.length === 0, issues, warnings, ...(normalized && !issues.length ? { normalized } : {}) };
}

// ---------------------------------------------------------------- stock
export interface PlacementStock {
  dc: string | null;
  gpu?: string;
  cpu?: string;
  /** Runpod's stockStatus: High | Medium | Low; null: none reported (usually none). */
  stock: string | null;
  /** GPUs of the type Runpod reports unreserved in that DC (null: not reported). */
  max_available: number | null;
  price: number | null;
}
export interface PoolStock {
  pool: string;
  compute: "GPU" | "CPU";
  count: number;
  /** ok: stock reported for every pod wanted; low: Low, or fewer than wanted; none: no stock anywhere it may land; unknown: not reported. */
  status: "ok" | "low" | "none" | "unknown";
  hint: string;
  placements: PlacementStock[];
}
export interface Availability {
  at: number;
  pools: PoolStock[];
  /** Pools that compete for the same GPU type in the same DC beyond what Runpod reports. */
  warnings: string[];
}
const RANK: Record<string, number> = { High: 3, Medium: 2, Low: 1 };
let stockCache: { at: number; key: string; v: Map<string, Omit<PlacementStock, "dc" | "gpu" | "cpu">> } | null = null;
/** Tests: forget the 60 s stock cache. */
export const clearStockCache = () => void (stockCache = null);

async function gpuStock(env: Env, pairs: { dc: string; gpu: string }[]): Promise<Map<string, Omit<PlacementStock, "dc" | "gpu" | "cpu">>> {
  const key = pairs.map((p) => `${p.dc}|${p.gpu}`).sort().join(",");
  if (stockCache && stockCache.key === key && Date.now() - stockCache.at < 60_000) return stockCache.v;
  const out = new Map<string, Omit<PlacementStock, "dc" | "gpu" | "cpu">>();
  if (!pairs.length) return out;
  const q = (withCount: boolean) =>
    `{ ${pairs
      .map((p, i) => `g${i}: gpuTypes(input: {id: ${JSON.stringify(p.gpu)}}) { id securePrice lowestPrice(input: {gpuCount: 1, dataCenterId: ${JSON.stringify(p.dc)}, secureCloud: true}) { stockStatus${withCount ? " maxUnreservedGpuCount" : ""} } }`)
      .join(" ")} }`;
  let d: any;
  try {
    d = await runpod.gql<any>(env, q(true));
  } catch (e) {
    // An older schema without the count: the status alone.
    if (!/maxUnreservedGpuCount/.test((e as Error).message)) throw e;
    d = await runpod.gql<any>(env, q(false));
  }
  pairs.forEach((p, i) => {
    const g = d?.[`g${i}`]?.[0];
    const lp = g?.lowestPrice;
    const n = lp?.maxUnreservedGpuCount;
    out.set(`${p.dc}|${p.gpu}`, { stock: lp?.stockStatus ?? null, max_available: typeof n === "number" ? n : null, price: g?.securePrice > 0 ? g.securePrice : null });
  });
  stockCache = { at: Date.now(), key, v: out };
  return out;
}

const short = (gpu: string) => gpu.replace(/^NVIDIA\s+/, "").replace(/ Blackwell Server Edition$/, "");
/** Stock per pool for a spec: where each pool's pods may land and what Runpod reports there. */
export async function availability(env: Env, spec: ClusterSpec): Promise<Availability> {
  const plan = spec.pools.map((p) => ({ p, pls: workerPlacements(spec, p) }));
  const pairs = new Map<string, { dc: string; gpu: string }>();
  for (const { pls } of plan) for (const pl of pls) if (pl.gpu && pl.dc) pairs.set(`${pl.dc}|${pl.gpu}`, { dc: pl.dc, gpu: pl.gpu });
  const st = await gpuStock(env, [...pairs.values()]);
  // Demand per (DC, GPU type): the pods of every pool whose first choice it is.
  const demand = new Map<string, { n: number; pools: string[] }>();
  const pools: PoolStock[] = plan.map(({ p, pls }) => {
    if (p.compute === "CPU") {
      return { pool: p.id, compute: "CPU", count: p.count, status: "ok", hint: `CPU (${(p.cpu_flavors?.length ? p.cpu_flavors : ["cpu3c", "cpu5c", "cpu3g"]).join(", ")}): any data centre as a fallback`, placements: pls.map((pl) => ({ dc: pl.dc ?? null, cpu: pl.cpu, stock: null, max_available: null, price: null })) };
    }
    const placements: PlacementStock[] = pls.map((pl) => ({ dc: pl.dc ?? null, gpu: pl.gpu, ...(st.get(`${pl.dc}|${pl.gpu}`) ?? { stock: null, max_available: null, price: null }) }));
    const first = placements.find((x) => x.stock && RANK[x.stock]) ?? placements[0];
    if (first?.dc && first.gpu && p.count) {
      const k = `${first.dc}|${first.gpu}`;
      const d = demand.get(k) || { n: 0, pools: [] };
      d.n += p.count;
      d.pools.push(p.id);
      demand.set(k, d);
    }
    const best = placements.reduce((a, x) => Math.max(a, RANK[x.stock ?? ""] ?? 0), 0);
    const avail = placements.reduce<number | null>((a, x) => (x.max_available === null ? a : (a ?? 0) + x.max_available), null);
    const where = placements.map((x) => `${short(x.gpu || "?")} in ${x.dc}: ${x.stock ?? "none"}${x.max_available !== null ? ` (${x.max_available} free)` : ""}`).join("; ");
    let status: PoolStock["status"];
    if (!placements.length) status = "none";
    else if (best === 0) status = "none";
    else if (best === 1 || (avail !== null && avail < p.count)) status = "low";
    else status = "ok";
    const hint = !placements.length ? "no region with a weights volume" : status === "none" ? `no stock reported: ${where}` : status === "low" ? `low stock: ${where}` : where;
    return { pool: p.id, compute: "GPU", count: p.count, status, hint, placements };
  });
  const warnings: string[] = [];
  for (const [k, d] of demand) {
    const s = st.get(k);
    const [dc, gpu] = k.split("|");
    if (s?.max_available !== null && s?.max_available !== undefined && d.n > s.max_available)
      warnings.push(`${d.n} pod(s) (${d.pools.join(", ")}) want ${short(gpu!)} in ${dc}; Runpod reports ${s.max_available} free`);
    else if (d.pools.length > 1 && s?.stock === "Low") warnings.push(`${d.pools.length} pools (${d.pools.join(", ")}) compete for ${short(gpu!)} in ${dc}, where stock is Low`);
  }
  return { at: Date.now(), pools, warnings };
}
