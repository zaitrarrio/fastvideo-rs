// GPU providers besides Runpod (docs/serve/deploy-gmi-brev.md §6): GMI Cloud
// containers and NVIDIA Brev VMs behind one small interface, so a pool (and a
// standalone pod, which is a one-pool cluster) with `provider: gmi | brev`
// goes through the same `up` / `down`, price check, deadline and ledger as a
// Runpod pool. Runpod keeps its own module (runpod.ts) and stays the default:
// nothing here runs for a pool without `provider` or with `provider: runpod`.
//
// A pod of these providers is keyed `<provider>:<name>` (gmi:fv-pod-x-…): the
// name is fv-control's (unique, fv-pod-* / fv-ctl-*) and known before the
// create call, so the pod's env carries its own id (log shipping, the edge's
// worker id); the provider's own id is looked up by name when needed. Neither
// provider has labels, so "ours" is a name fv-control made AND recorded in
// cluster_pods; nothing else on those accounts is ever touched.
//
// What differs from a Runpod pod (all in the env and the start command):
//  - no network volume: weights come from our R2 weights store at boot
//    (weights_source r2, r2weights.ts; the default once every tree of the pool
//    is published), from the Hub at pinned revisions (hub, owner approval) or
//    not at all (fake engine);
//  - no HTTPS proxy: the pod runs a Cloudflare tunnel, a named one fv-control
//    made for it on the owner's zone (tunnels.ts; URL known up front) or,
//    without those settings, a quick tunnel; it reports its https URL to
//    POST /ingest/v1/endpoint (index.ts) with the cluster's ingest token;
//    edge fronts export it as FV_DISPATCH_ENDPOINT;
//  - no balance API: a monthly budget per provider instead of the balance
//    floor: the policy brev_budget_usd / gmi_budget_usd (D1, set in the
//    console: Settings → Policies or the launch form's budget field), else the
//    Worker var BREV_BUDGET_USD / GMI_BUDGET_USD (providerBudget);
//  - Runpod secret references do not resolve there: FV_PROVIDER_SECRET_ENV.
import { brev, brevInstanceTypes, brevOff, brevPrice, brevPriceLive, brevTaskLine, type BrevType } from "./brev";
import { recordBoot } from "./boottime";
import { adoptAndStop, bootEstimate, brevDiskGb, treesGb, brevRowById, brevRunScript, brevStartup, markBrevDeleted, newBootToken, parkedDph, recordBrevCreate, releaseBrev, restartDiag, warmCandidates, type Release, type TreeRevs } from "./brev-park";
import { BAKED_CONFIGS } from "./cluster/image-configs";
import type { PoolSpec } from "./cluster/spec";
import { WORKER_CONFIGS } from "./cluster/worker-configs";
import { sha256Hex } from "./crypto";
import { expectedTunnelUrl, releaseTunnels, tunnelSettings, tunnelView } from "./tunnels";
import { OTHER_PROVIDERS, type OtherProviderId, type ProviderId } from "./enums";
import type { Env } from "./env";
import { FV_NAME_RE, gmi, gmiDefaultIdc, gmiEnabled, gmiProducts } from "./gmi";
import { POOL_PRESETS } from "./presets";
import { HttpError, now, parseJson, putSetting, rateLimit, utcDay } from "./util";
import { AUX_FILES, HUB_TREES } from "./weights-sources";
import { REGIONS } from "./cluster/regions";
import { missingTrees } from "./volumes";
import { DRIVER_CHECK_SH } from "./driver-check";
import { R2_AUX_TREES, r2AuxTrees, R2_DOWNLOAD_GBPS, R2_FETCH_SH, r2Bytes, r2Cfg, r2DiskGb, r2Manifest, r2PodEnv, r2Trees } from "./r2weights";
import { UPSAMPLER_START_SH, UPSAMPLER_WAIT_SH } from "./upsampler";
import { policies, type Policies } from "./alerts";

// ---------------------------------------------------------------- ids
const KEY_RE = /^(gmi|brev):(.+)$/;
/** A pod id of a GMI / Brev pod (`gmi:<name>`); Runpod ids have no prefix. */
export const isOtherPod = (podId: string) => KEY_RE.test(podId);
export function splitKey(podId: string): { provider: OtherProviderId; name: string } | null {
  const m = KEY_RE.exec(podId);
  return m ? { provider: m[1] as OtherProviderId, name: m[2]! } : null;
}
export const podKey = (p: OtherProviderId, name: string) => `${p}:${name}`;
export const poolProvider = (p: { provider?: ProviderId }): ProviderId => p.provider ?? "runpod";
export const isOtherPool = (p: { provider?: ProviderId }) => poolProvider(p) !== "runpod";

// ---------------------------------------------------------------- the interface
export type InstanceState = "starting" | "running" | "stopped" | "failed";
export interface ProviderInstance {
  key: string;
  provider: OtherProviderId;
  /** The provider's own id (GMI uuid, Brev workspace id). */
  id: string;
  name: string;
  state: InstanceState;
  /** The provider's own status word. */
  raw: string;
  reason: string | null;
  gpu: string | null;
  createdAt: number | null;
}
export interface Offer {
  gpu: string;
  region: string | null;
  usd_per_hr: number | null;
  in_stock: boolean | null;
  /** Brev: the type can be stopped (keep-on-stop, warm restarts); null: unknown. */
  stoppable?: boolean | null;
  /** Brev: the list's estimated deploy time (s). */
  deploy_s?: number | null;
  /** Brev: storage $/GB-hr (what a parked instance costs). */
  storage_usd_per_gb_hr?: number | null;
}
export interface ProviderLaunch {
  name: string;
  image: string;
  gpu: string;
  region?: string;
  env: Record<string, string>;
  /** The pod's weights (Brev: the disk size and the trees recorded for keep-on-stop). */
  weights?: WeightsPlan;
  /** The cluster it serves (Brev's record). */
  clusterId?: string;
  /** Brev: the VM's disk in GB the pool asked for (container_disk_gb); unset: sized for the weights (brevDisk). */
  diskGb?: number;
}
export interface ComputeProvider {
  id: OtherProviderId;
  title: string;
  /** Why it is off (a secret unset), or null. */
  off(env: Env): string | null;
  /** The GPU products / instance types pods may use (owner config). */
  gpus(env: Env): string[];
  /** Price and stock of one GPU product for the planner (null fields: unknown). */
  offer(env: Env, gpu: string, region?: string): Promise<Offer>;
  /** fv-named instances of the account (only those). */
  list(env: Env): Promise<ProviderInstance[]>;
  create(env: Env, req: ProviderLaunch): Promise<{ id: string }>;
  remove(env: Env, inst: ProviderInstance): Promise<boolean>;
  logs?(env: Env, inst: ProviderInstance): Promise<string[]>;
  /** The Worker var's monthly budget in $ (GMI_BUDGET_USD / BREV_BUDGET_USD; null: unset). The policy set in the
   * console overrides it: read the budget with providerBudget. */
  budgetVar(env: Env): number | null;
}

const num = (s: string | undefined): number | null => {
  const v = Number(s);
  return s !== undefined && s !== "" && Number.isFinite(v) && v >= 0 ? v : null;
};
const ms = (s: string | null) => (s && Number.isFinite(Date.parse(s)) ? Date.parse(s) : null);

const GMI: ComputeProvider = {
  id: "gmi",
  title: "GMI Cloud",
  off: (env) => (!gmiEnabled(env) ? "GMI_API_KEY is not set" : !gmiProducts(env).length ? "GMI_PRODUCTS is not set (the product ids GMI gave the account)" : null),
  gpus: gmiProducts,
  async offer(env, gpu, region) {
    const idc = region || gmiDefaultIdc(env);
    const p = (await gmi.products(env, idc)).find((x) => x.name === gpu);
    return { gpu, region: idc, usd_per_hr: p?.usd_per_hr ?? null, in_stock: p ? p.valid : false };
  },
  async list(env) {
    return (await gmi.containers(env))
      .filter((c) => FV_NAME_RE.test(c.name))
      .map((c) => ({
        key: podKey("gmi", c.name),
        provider: "gmi" as const,
        id: c.id,
        name: c.name,
        state: c.status === "running" ? "running" : c.status === "error" || c.status === "zombie" ? "failed" : c.status === "stopped" || c.status === "terminating" ? "stopped" : "starting",
        raw: c.status,
        reason: c.reason,
        gpu: c.product,
        createdAt: ms(c.createdAt),
      }));
  },
  async create(env, req) {
    const templateId = await gmi.ensureTemplate(env, req.image, (await sha256Hex(req.image)).slice(0, 12));
    const id = await gmi.create(env, { name: req.name, product: req.gpu, idc: req.region || gmiDefaultIdc(env), templateId, command: "bash", args: ["-c", PROVIDER_BOOT], envs: req.env, ports: [8000] });
    return { id };
  },
  remove: (env, inst) => gmi.remove(env, inst.id),
  logs: (env, inst) => gmi.logs(env, inst.id),
  budgetVar: (env) => num(env.GMI_BUDGET_USD),
};

const BREV: ComputeProvider = {
  id: "brev",
  title: "NVIDIA Brev",
  off: (env) => brevOff(env) ?? (!brevInstanceTypes(env).length ? "BREV_INSTANCE_TYPES is not set (allowed instance types)" : null),
  gpus: brevInstanceTypes,
  async offer(env, gpu) {
    // The live instance-type list (docs/serve/deploy-gmi-brev.md §3.2; cached 1 h), the owner's BREV_PRICES overriding a price.
    const { usd_per_hr, info } = await brevPriceLive(env, gpu);
    return { gpu, region: info?.location ?? null, usd_per_hr, in_stock: info?.available ?? null, stoppable: info ? info.stoppable : null, deploy_s: info?.deploy_s ?? null, storage_usd_per_gb_hr: info?.storage_usd_per_gb_hr ?? null };
  },
  async list(env) {
    return (await brev.workspaces(env))
      .filter((w) => FV_NAME_RE.test(w.name))
      .map((w) => ({
        key: podKey("brev", w.name),
        provider: "brev" as const,
        id: w.id,
        name: w.name,
        state: w.status === "RUNNING" ? "running" : w.status === "FAILURE" ? "failed" : ["STOPPED", "STOPPING", "DELETING"].includes(w.status) ? "stopped" : "starting",
        raw: w.status,
        // Brev's statusMessage first (why a start did not run, e.g. capacity), then its latest start task (environment.tasks: empty statusMessage, the reason there), then its health.
        reason: [w.statusMessage, brevTaskLine(w.lastStart), w.health].filter(Boolean).join("; ") || null,
        gpu: w.instanceType,
        createdAt: ms(w.createdAt),
      }));
  },
  async create(env, req) {
    // The startup script only installs the per-boot bootstrap; the launch itself (image + env) is the run script it
    // fetches from fv-control (brev-park.ts), so a parked VM restarted later runs that later launch.
    const run = brevRun(req.image, req.env);
    // Owner decision 2026-10-13: a type Brev cannot stop is created like any other (its stop fails, so a stop deletes
    // it: brev-park.ts, owner rule 2026-10-13); its price is recorded for the ledger.
    const { usd_per_hr, info } = await brevPriceLive(env, req.gpu);
    const trees = req.weights?.source === "hub" || req.weights?.source === "r2" ? req.weights.trees : [];
    const disk = brevDisk(info, trees, req.diskGb);
    const token = newBootToken();
    const id = await brev.create(env, { name: req.name, instanceType: req.gpu, startupScript: brevStartup(`${(env.PUBLIC_URL || "").replace(/\/$/, "")}/ingest/v1/brev-boot`, token), diskStorage: `${disk}Gi`, isStoppable: env.BREV_IS_STOPPABLE !== "0" && !!info?.stoppable });
    await recordBrevCreate(env, {
      workspace_id: id,
      pod_id: podKey("brev", req.name),
      name: req.name,
      instance_type: req.gpu,
      info,
      disk_gb: info && !info.elastic_disk && info.fixed_disk_gb ? info.fixed_disk_gb : disk,
      launch_trees: launchTrees(req.weights, env),
      cluster_id: req.clusterId ?? null,
      token,
      run,
      usd_per_hr,
    });
    return { id };
  },
  async remove(env, inst) {
    const ok = await brev.remove(env, inst.id);
    if (ok) await markBrevDeleted(env, inst.id);
    return ok;
  },
  budgetVar: (env) => num(env.BREV_BUDGET_USD),
};

export const PROVIDER_IMPLS: Record<OtherProviderId, ComputeProvider> = { gmi: GMI, brev: BREV };
export const providerImpl = (p: OtherProviderId) => PROVIDER_IMPLS[p];
export const providerEnabled = (env: Env, p: OtherProviderId) => !providerImpl(p).off(env);

// ---------------------------------------------------------------- the monthly budget
/** The policy key of a provider's monthly budget (alerts.ts Policies). */
export const budgetPolicyKey = (p: OtherProviderId) => `${p}_budget_usd` as const;
export type BudgetSource = "policy" | "worker_var";
/**
 * A provider's monthly budget in $ and where it comes from. The one the owner enters in the console (the policy
 * brev_budget_usd / gmi_budget_usd: Settings → Policies, the launch form's budget field, PUT
 * /api/v1/providers/<p>/budget) wins; unset (null), the Worker var (BREV_BUDGET_USD / GMI_BUDGET_USD); neither: null
 * (launches are refused). Owner report 2026-10-18: the budget entered was ignored because only the Worker var was read.
 */
export async function providerBudget(env: Env, p: OtherProviderId, pol?: Policies): Promise<{ usd: number | null; source: BudgetSource | null; worker_var: number | null }> {
  const set = (pol ?? (await policies(env)))[budgetPolicyKey(p)];
  const worker_var = providerImpl(p).budgetVar(env);
  if (typeof set === "number" && Number.isFinite(set) && set >= 0) return { usd: set, source: "policy", worker_var };
  return { usd: worker_var, source: worker_var === null ? null : "worker_var", worker_var };
}
/** Where the owner changes a provider's budget (said in every refusal and alert that names it). */
export const budgetWhere = (p: OtherProviderId) => `the launch form's "Monthly budget" field or Settings → Policies ${budgetPolicyKey(p)}`;

/** One provider's state for /api/v1/providers and the UI (enabled, why not, budget, this month's spend). */
export async function providerView(env: Env, p: OtherProviderId) {
  const impl = providerImpl(p);
  const off = impl.off(env);
  const [spend, budget] = await Promise.all([providerSpend(env, p), providerBudget(env, p)]);
  return {
    id: p,
    title: impl.title,
    enabled: !off,
    reason: off ? `${impl.title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)` : null,
    gpus: impl.gpus(env),
    budget_usd: budget.usd,
    // policy: entered in the console (Settings → Policies or the launch form); worker_var: the Worker's var.
    budget_source: budget.source,
    month_usd: spend.month,
    running_dph: spend.running_dph,
    parked_dph: spend.parked_dph,
    hub_downloads_approved: env.FV_HUB_DOWNLOADS_APPROVED === "1",
    // weights_source r2 (r2weights.ts): configured or why not, and the published trees.
    r2_weights: { enabled: !r2Cfg(env).off, reason: r2Cfg(env).off, trees: Object.keys(r2Trees(env)).sort() },
    // How its pods are reached (tunnels.ts): named tunnels on the owner's zone, or quick tunnels; and what is wrong.
    tunnel: tunnelView(env),
  };
}

// ---------------------------------------------------------------- money
/** This UTC month's ledger for a provider's pods (cost_daily) and the $/hr of its running ones (pods table). */
export async function providerSpend(env: Env, p: OtherProviderId): Promise<{ month: number; running_dph: number; parked_dph: number }> {
  const first = utcDay(now()).slice(0, 8) + "01";
  const m = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd FROM cost_daily WHERE day >= ? AND pod_id LIKE ?").bind(first, `${p}:%`).first<{ usd: number }>();
  const r = await env.DB.prepare("SELECT COALESCE(SUM(cost_per_hr), 0) AS dph FROM pods WHERE gone_at IS NULL AND provider = ? AND desired_status = 'RUNNING'").bind(p).first<{ dph: number }>();
  // Brev keep-on-stop: parked instances bill their disk (brev-park.ts).
  const parked = p === "brev" ? await parkedDph(env) : 0;
  return { month: Number(m?.usd || 0), running_dph: Number(r?.dph || 0), parked_dph: parked };
}

export interface BudgetCheck {
  provider: OtherProviderId;
  ok: boolean;
  reasons: string[];
  budget: number | null;
  month: number;
  running_dph: number;
  /** Brev: the storage $/hr of parked instances. */
  parked_dph: number;
  add_dph: number;
  hours: number;
  projected: number;
}
/** No balance API on GMI / Brev: this month's spend + the running pods, the parked instances' storage and `addDph`
 * until the deadline must stay within the budget. (A Brev type that cannot be stopped ends at the deadline too: its
 * stop fails and deletes it, owner rule 2026-10-13.) */
export async function budgetCheck(env: Env, p: OtherProviderId, addDph: number, hours: number): Promise<BudgetCheck> {
  const impl = providerImpl(p);
  const [{ usd: budget }, s] = await Promise.all([providerBudget(env, p), providerSpend(env, p)]);
  const dph = s.running_dph + s.parked_dph + addDph;
  const projected = s.month + dph * hours;
  const reasons: string[] = [];
  if (budget === null) reasons.push(`${impl.title} has no balance API: set its monthly budget (the most fv-control may spend there per month: ${budgetWhere(p)}, or the Worker var ${p.toUpperCase()}_BUDGET_USD) before launching`);
  else if (projected > budget)
    reasons.push(`${impl.title}: $${s.month.toFixed(2)} spent this month + $${dph.toFixed(2)}/hr${s.parked_dph ? ` (parked storage $${s.parked_dph.toFixed(3)}/hr included)` : ""} for ${hours.toFixed(2)} h = $${projected.toFixed(2)}, over the monthly budget $${budget} (raise it: ${budgetWhere(p)})`);
  return { provider: p, ok: !reasons.length, reasons, budget, month: s.month, running_dph: s.running_dph, parked_dph: s.parked_dph, add_dph: addDph, hours, projected };
}

/** A budget check in one line for the operation log (no Runpod balance or floor: GMI / Brev have none):
 * "NVIDIA Brev: $6.04 spent this month + ~$4.04/hr (this launch $4.04/hr) for 1.00 h → projected $10.08 at the
 * deadline; budget $100 (within)". */
export function budgetLine(b: BudgetCheck): string {
  const dph = b.running_dph + b.parked_dph + b.add_dph;
  const parts = [`this launch $${b.add_dph.toFixed(2)}/hr`, b.running_dph ? `running $${b.running_dph.toFixed(2)}/hr` : null, b.parked_dph ? `parked storage $${b.parked_dph.toFixed(3)}/hr` : null].filter(Boolean);
  const verdict = b.budget === null ? "unset" : `$${b.budget} (${b.projected > b.budget ? "OVER" : "within"})`;
  return `${providerImpl(b.provider).title}: $${b.month.toFixed(2)} spent this month + ~$${dph.toFixed(2)}/hr (${parts.join(", ")}) for ${b.hours.toFixed(2)} h → projected $${b.projected.toFixed(2)} at the deadline; budget ${verdict}`;
}

/** Why a GMI / Brev create failed: `capacity` (no stock: the type / zone has none now; retried like Runpod's no
 * stock), `provider` (the provider answered HTTP 5xx with an error of its own: retrying the launch now will not help),
 * else `other` (handled like no stock, as before). */
export interface CreateFailure {
  kind: "capacity" | "provider" | "other";
  /** The provider's HTTP status, when it answered. */
  status: number | null;
  said: string;
  /** What the error means, when fv-control recognises it. */
  hint: string | null;
}
export const CAPACITY_RE = /InsufficientCapacity|insufficient (?:resources|capacity)|no capacity|out of (?:stock|capacity)|sold out|not enough capacity|capacity (?:is )?(?:not available|unavailable)/i;
/** AWS's answer to a call made with a credential it does not know: from inside the provider's backend (fv-control
 * never talks to AWS; its Brev token is a bearer on Brev's own API). */
const PROVIDER_CRED_RE = /UnrecognizedClientException|security token included in the request is invalid|InvalidClientTokenId|ExpiredTokenException/i;
export function createFailure(e: unknown): CreateFailure {
  const said = String((e as Error)?.message ?? e).replace(/\s+/g, " ").trim();
  const up = e instanceof HttpError ? Number(e.extra?.upstream) : NaN;
  const status = Number.isFinite(up) ? up : null;
  if (CAPACITY_RE.test(said)) return { kind: "capacity", status, said, hint: null };
  if (status !== null && status >= 500)
    return {
      kind: "provider",
      status,
      said,
      hint: PROVIDER_CRED_RE.test(said)
        ? "a cloud credential was refused inside the provider's backend (an AWS error raised on its side, not fv-control's request, not a shortage of stock); on 2026-10-18 every Brev create failed this way, AWS, Nebius and Shadeform types alike): check the provider's status or ask its support"
        : null,
    };
  return { kind: "other", status, said, hint: null };
}

/** The $/hr a pool's pod will cost on its provider: the offer's price, else the cap (the most it may cost). */
export async function poolDph(env: Env, pool: PoolSpec, cap: number): Promise<{ dph: number; known: boolean; offer: Offer | null }> {
  const p = poolProvider(pool) as OtherProviderId;
  const offer = await providerImpl(p)
    .offer(env, pool.provider_gpu!, pool.provider_region)
    .catch(() => null);
  return offer?.usd_per_hr != null ? { dph: offer.usd_per_hr, known: true, offer } : { dph: cap, known: false, offer };
}

// ---------------------------------------------------------------- weights
const configText = (pool: PoolSpec): string => {
  if (pool.config_toml) return pool.config_toml;
  const f = (pool.config || "").split("/").pop() || "";
  return BAKED_CONFIGS[f] || WORKER_CONFIGS[f] || "";
};
export interface WeightsPlan {
  source: "hub" | "r2" | "none";
  trees: string[];
  /** hub: trees no plain Hub fetch can make (several repos, a conversion); r2: trees not published in the R2 weights
   * store. The launch is refused with them. */
  unsupported: string[];
  /** TSV for FV_WEIGHTS_TREES_B64: tree<TAB>dest<TAB>repo<TAB>rev<TAB>globs, aux<TAB>path<TAB>url<TAB>sha256<TAB>size. */
  tsv: string;
}
/** The weight trees a pool loads: `${FV_WEIGHTS}/<tree>` in its worker config, plus its preset's list. */
export function poolTrees(pool: PoolSpec): string[] {
  const out = new Set<string>();
  for (const line of configText(pool).split("\n")) {
    if (/^\s*#/.test(line)) continue;
    for (const m of line.matchAll(/\$\{FV_WEIGHTS\}\/([A-Za-z0-9._-]+)(?:\/([A-Za-z0-9._-]+))?/g)) out.add(m[2] && HUB_TREES[`${m[1]}/${m[2]}`] ? `${m[1]}/${m[2]}` : m[1]!);
  }
  for (const p of POOL_PRESETS) if (p.pool.variant === pool.variant && ((pool.config && p.pool.config === pool.config) || (pool.config_toml && p.pool.config_toml === pool.config_toml))) p.weights.forEach((w) => out.add(w));
  return [...out].sort();
}
/** The pre-quantized trees of the pool's preset (presets.ts PoolPreset.prequant): r2 fetches them, the Hub cannot. */
export function poolPrequant(pool: PoolSpec): string[] {
  const p = POOL_PRESETS.find((x) => x.pool.variant === pool.variant && ((pool.config && x.pool.config === pool.config) || (pool.config_toml && x.pool.config_toml === pool.config_toml)));
  return [...(p?.prequant ?? [])];
}
/** A GMI / Brev pool's weights source: its own, else r2 when its config loads trees and every one is published in
 * the R2 weights store (r2weights.ts), else none. Runpod GPU pools: their own, else the volume when it holds every
 * tree (volumes.ts: the 500 GB EU volume has H3 + LTX only, owner decision 2026-10-13), else r2 when every tree is
 * published, else the volume (its launch check names the missing trees). */
export function weightsSource(pool: PoolSpec, env?: Env): "volume" | "hub" | "r2" | "none" {
  if (pool.weights_source) return pool.weights_source;
  const trees = poolTrees(pool).filter((t) => t !== "auxiliary");
  const pub = env ? r2Trees(env) : R2_DEFAULT_TREES();
  const published = trees.length > 0 && trees.every((t) => pub[t]);
  if (!isOtherPool(pool)) {
    if (pool.compute === "CPU" || pool.volume === true) return "volume";
    return (missingTrees(REGIONS.eu.volume, trees) ?? []).length && published ? "r2" : "volume";
  }
  return published ? "r2" : "none";
}
const R2_DEFAULT_TREES = () => r2Trees({} as Env);
export function weightsPlan(pool: PoolSpec, env?: Env): WeightsPlan {
  const ws = weightsSource(pool, env);
  const source = ws === "hub" || ws === "r2" ? ws : "none";
  if (source === "none") return { source, trees: [], unsupported: [], tsv: "" };
  const trees = poolTrees(pool).filter((t) => t !== "auxiliary");
  const pub = env ? r2Trees(env) : R2_DEFAULT_TREES();
  // r2: the presets' pre-quantized trees too (published ones; recipe-trees.ts): the Hub has no source for them.
  if (source === "r2") for (const t of poolPrequant(pool)) if (pub[t] && !trees.includes(t)) trees.push(t);
  // r2: the aux files come with the weights as their R2 trees (not from GitHub / PyTorch); the boot's aux rows below
  // then only check them by SHA-256 (a file missing or different still comes from its pinned URL).
  if (source === "r2" && trees.length && env) for (const t of r2AuxTrees(env)) if (!trees.includes(t)) trees.push(t);
  const unsupported = trees.filter((t) => (source === "hub" ? !HUB_TREES[t] : !pub[t]));
  // r2: the trees come from the fetcher's plan (fv-control's /ingest/v1/weights-plan), so only the aux rows here.
  const rows = source === "r2" ? [] : trees.filter((t) => HUB_TREES[t]).map((t) => ["tree", t, HUB_TREES[t]!.repo, HUB_TREES[t]!.revision, HUB_TREES[t]!.globs.join(" ")].join("\t"));
  // The TAEs (and LPIPS) every engine may load: small, pinned by SHA-256.
  for (const a of AUX_FILES) rows.push(["aux", a.path, a.url, a.sha256, String(a.size)].join("\t"));
  return { source, trees, unsupported, tsv: rows.join("\n") + "\n" };
}
/** The launch form's Brev instance types: every allowed one, stoppable first (owner decision 2026-10-13: a type Brev
 * cannot stop is offered like any other, its detail saying that a stop deletes it: owner rule 2026-10-13),
 * each with its $/hr, deploy time and parked / held instances. A type the live list does not show (or an unreachable
 * list) stays, last. */
export async function brevGpuOptions(env: Env): Promise<{ id: string; detail: string; provider: "brev"; stoppable: boolean | null; usd_per_hr: number | null; parked: number; disk: BrevDiskInfo | null }[]> {
  const types = await brev.types(env).catch(() => []);
  const parked = (await env.DB.prepare("SELECT instance_type, COUNT(*) AS n FROM brev_instances WHERE state IN ('parked', 'held') GROUP BY instance_type").all<{ instance_type: string; n: number }>()).results || [];
  const out = brevInstanceTypes(env).map((id) => {
    const t = types.find((x) => x.type === id);
    const n = Number(parked.find((p) => p.instance_type === id)?.n || 0);
    const price = brevPrice(env, id) ?? t?.usd_per_hr ?? null;
    const stop = !t ? "not in Brev's list" : t.stoppable ? "stoppable: stopped and kept on stop (reused by the next launch)" : "WARNING: can't be stopped: a stop deletes it (its disk and weights go too)";
    const bits = ["NVIDIA Brev", price !== null ? `$${price.toFixed(2)}/hr` : "price unknown", stop, ...(t?.deploy_s ? [`deploy ~${Math.round(t.deploy_s / 60)} min`] : []), ...(n ? [`${n} parked / held`] : [])];
    const disk: BrevDiskInfo | null = t ? { kind: !t.elastic_disk && t.fixed_disk_gb ? "fixed" : t.elastic_disk ? "elastic" : null, gb: t.fixed_disk_gb, min_gb: t.disk_min_gb, max_gb: t.disk_max_gb, rule: brevDiskRule(t) } : null;
    return { id, detail: bits.join(" · "), provider: "brev" as const, stoppable: t ? t.stoppable : null, usd_per_hr: price, parked: n, disk };
  });
  const rank = (o: { stoppable: boolean | null }) => (o.stoppable === true ? 0 : o.stoppable === false ? 1 : 2);
  return out.sort((a, b) => rank(a) - rank(b));
}
/** A Brev instance type's disk for the forms: fixed (its size) or elastic (its range), and the rule in words. */
export interface BrevDiskInfo {
  kind: "fixed" | "elastic" | null;
  gb: number | null;
  min_gb: number | null;
  max_gb: number | null;
  rule: string;
}
/** Owner decision 2026-10-13 ("On Brev we don't want to filter out non-stoppable VMs; we'll just treat them like any
 * other Brev VM"): the warning a type Brev cannot stop gets (launch form, validator, launch log); null when the type is
 * stoppable or the live list does not know it (or is unreachable). */
export async function brevNoStopNote(env: Env, gpu: string): Promise<string | null> {
  const { usd_per_hr, info } = await brevPriceLive(env, gpu).catch(() => ({ usd_per_hr: null, info: null }));
  if (!info || info.stoppable) return null;
  const price = usd_per_hr !== null ? ` ($${usd_per_hr.toFixed(2)}/hr while it runs)` : "";
  return `${gpu} can't be stopped on NVIDIA Brev: a stop (yours, the deadline, the budget floor, a scale-down or roll) deletes the VM${price}: its disk and weights go too, so its next launch downloads them again (owner rule 2026-10-13: a Brev VM whose stop does not succeed is deleted)`;
}
/** brevNoStopNote for every Brev pool of a spec, path-anchored at provider_gpu (warnings: nothing is refused). */
export async function brevNoStopWarnings(env: Env, spec: { pools: PoolSpec[] }): Promise<{ path: (string | number)[]; message: string }[]> {
  const out: { path: (string | number)[]; message: string }[] = [];
  for (const [i, p] of spec.pools.entries()) {
    if (poolProvider(p) !== "brev" || !p.provider_gpu) continue;
    const m = await brevNoStopNote(env, p.provider_gpu);
    if (m) out.push({ path: ["pools", i, "provider_gpu"], message: `WARNING: ${m}` });
  }
  return out;
}
/** A Brev pool's warm / cold outlook (the launch form, the planner): a parked / held instance of the type restarts
 * (warm when it holds some of the trees, else reuse: they download onto it); else every tree downloads (cold). A
 * type Brev cannot stop is always cold (its stop deletes it), the note saying so (brevNoStopNote). */
export async function brevOutlook(env: Env, pool: PoolSpec): Promise<{ warm: boolean; stoppable: boolean | null; instance: string | null; download_gb: number; boot_s: number; note: string } | null> {
  if (poolProvider(pool) !== "brev" || !pool.provider_gpu) return null;
  const plan = weightsPlan(pool, env);
  const want = launchTrees(plan, env);
  const trees = Object.keys(want);
  const info = await brev.type(env, pool.provider_gpu);
  const r2 = plan.source === "r2" ? { bytes: (t: string[]) => r2Bytes(env, t), gbps: R2_DOWNLOAD_GBPS } : undefined;
  const stoppable = info ? info.stoppable : null;
  const min = (s: number) => `~${Math.max(1, Math.round(s / 60))} min`;
  if (!trees.length) {
    const e = bootEstimate(info, []);
    const ns = stoppable === false ? await brevNoStopNote(env, pool.provider_gpu) : null;
    return { warm: false, stoppable, instance: null, download_gb: 0, boot_s: e.boot_s, note: `no weights: boot ${min(e.boot_s)}${ns ? `; WARNING: ${ns}` : ""}` };
  }
  if (stoppable === false) {
    const e = bootEstimate(info, trees, [], r2);
    const ns = await brevNoStopNote(env, pool.provider_gpu);
    return { warm: false, stoppable, instance: null, download_gb: e.download_gb, boot_s: e.boot_s, note: `cold: downloads ${trees.join(", ")} (~${e.download_gb} GB${plan.source === "r2" ? " from R2" : " from the Hub"}): boot ${min(e.boot_s)}; WARNING: ${ns || `${pool.provider_gpu} can't be stopped: a stop deletes it (its disk and weights go too)`}` };
  }
  // Reuse (brev-park.ts): a parked / held instance of the type, the one holding most trees first; the missing ones download.
  const c = (await warmCandidates(env, pool.provider_gpu, want, undefined, { minDiskGb: pool.container_disk_gb }))[0];
  if (c) {
    const e = bootEstimate(info, trees, c.matched, r2);
    const what = `${c.row.state} ${c.row.name}${c.row.state === "held" && c.row.note ? ` (held: ${c.row.note.slice(0, 120)})` : ""}`;
    const note = c.matched.length
      ? `warm: restarts ${what} (holds ${c.matched.join(", ")}${c.missing.length ? `; downloads ${c.missing.join(", ")} ~${e.download_gb} GB` : ""}): boot ${min(e.boot_s)}`
      : `reuse: restarts ${what} (none of the trees at the pinned revisions on it; downloads ${c.missing.join(", ")} ~${e.download_gb} GB onto it): boot ${min(e.boot_s)}`;
    return { warm: true, stoppable, instance: c.row.name, download_gb: e.download_gb, boot_s: e.boot_s, note };
  }
  const e = bootEstimate(info, trees, [], r2);
  let note = `cold: downloads ${trees.join(", ")} (~${e.download_gb} GB${plan.source === "r2" ? " from R2" : " from the Hub"}): boot ${min(e.boot_s)}`;
  if (stoppable) note += "; kept on stop (the next launch is warm)";
  return { warm: false, stoppable, instance: null, download_gb: e.download_gb, boot_s: e.boot_s, note };
}
/** The trees of a plan at their revisions ({} without weights): what a Brev disk keeps. A Hub tree in R2 has the Hub
 * revision as its R2 revision, so a disk filled from either source serves both. */
export function launchTrees(plan: WeightsPlan | undefined, env?: Env): TreeRevs {
  if (!plan) return {};
  if (plan.source === "hub") return Object.fromEntries(plan.trees.filter((t) => HUB_TREES[t]).map((t) => [t, HUB_TREES[t]!.revision]));
  if (plan.source === "r2") {
    const pub = env ? r2Trees(env) : R2_DEFAULT_TREES();
    // The aux trees are not what makes a VM warm (0.37 GB, checked by SHA-256 at every boot): left out, so a VM parked
    // with Hub trees (aux by URL) matches an r2 launch of the same trees.
    return Object.fromEntries(plan.trees.filter((t) => pub[t] && !R2_AUX_TREES.includes(t)).map((t) => [t, pub[t]!.rev]));
  }
  return {};
}

// ---------------------------------------------------------------- the pod's env and start command
/** Runpod secret references a GMI / Brev pod cannot resolve, replaced by FV_PROVIDER_SECRET_ENV's values or left out. */
export function providerEnv(
  env: Env,
  full: Record<string, string>,
  o: { provider: OtherProviderId; key: string; name: string; deadlineMs: number; reportToken: string; weights: WeightsPlan; scriptsRef: string },
): { env: Record<string, string>; dropped: string[]; notes: string[] } {
  const out: Record<string, string> = {};
  const dropped: string[] = [];
  const notes: string[] = [];
  const subst = parseJson<Record<string, unknown>>(env.FV_PROVIDER_SECRET_ENV, {});
  for (const [k, v] of Object.entries(full)) {
    if (/\{\{\s*RUNPOD_SECRET_/.test(v)) {
      if (typeof subst[k] === "string" && subst[k]) out[k] = subst[k] as string;
      else dropped.push(k);
      continue;
    }
    out[k] = v;
  }
  // The Runpod key and floor of the Runpod watchdog never leave Runpod.
  delete out.FV_BACKSTOP_API_KEY;
  delete out.FV_MIN_BALANCE;
  // fv-serve refuses to start with key_store = d1 (direct workers) when the D1 settings are missing ("auth.key_store
  // = d1 needs the D1 settings"): without FV_PROVIDER_SECRET_ENV's FV_CF_* / FV_D1_*, keys live in memory (minted
  // through the worker's admin API with FV_ADMIN_TOKEN; gone at restart). Live on Brev 2026-10-10: exit right after the tunnel.
  const d1 = ["FV_CF_ACCOUNT_ID", "FV_CF_API_TOKEN", "FV_D1_DATABASE_ID"];
  if (out.FV_KEY_STORE === "d1" && d1.some((k) => !out[k])) {
    out.FV_KEY_STORE = "memory";
    notes.push(`FV_KEY_STORE=memory: no D1 settings (${d1.filter((k) => !out[k]).join(", ")}; FV_PROVIDER_SECRET_ENV), so API keys minted on this pod live in its memory`);
  }
  Object.assign(out, {
    FV_PROVIDER: o.provider,
    FV_POD_ID: o.key,
    FV_POD_NAME: o.name,
    FV_WORKER_ID: o.key,
    FV_LOG_SHIP_POD: o.key,
    FV_WEIGHTS: "/workspace/weights",
    FV_WEIGHTS_SOURCE: o.weights.source,
    FV_CLUSTER_DEADLINE: String(Math.floor(o.deadlineMs / 1000)),
    FV_ENDPOINT_REPORT_URL: `${(env.PUBLIC_URL || "").replace(/\/$/, "")}/ingest/v1/endpoint`,
    FV_ENDPOINT_REPORT_TOKEN: o.reportToken,
  });
  if (o.weights.source === "hub") {
    out.FV_WEIGHTS_TREES_B64 = btoa(o.weights.tsv);
    out.FV_SCRIPTS_URL = `https://raw.githubusercontent.com/${env.GITHUB_REPO || "zaitrarrio/fastvideo-rs"}/${o.scriptsRef}/scripts/gpu`;
  }
  // r2: the trees' URLs come from fv-control at boot (presigned, short-lived); the aux files by their pinned URLs.
  if (o.weights.source === "r2") Object.assign(out, r2PodEnv(env), { FV_WEIGHTS_TREES_B64: btoa(o.weights.tsv) });
  // GMI: the in-container watchdog deletes its own container at the deadline (fv-control's cron does too).
  if (o.provider === "gmi" && env.GMI_API_KEY) Object.assign(out, { FV_BACKSTOP_API_KEY: env.GMI_API_KEY, FV_BACKSTOP_API: (env.GMI_API || "https://console.gmicloud.ai/api").replace(/\/$/, "") });
  return { env: out, dropped, notes };
}

/** cloudflared, pinned (2026.9.0, linux-amd64) and checked by SHA-256 before it runs. */
export const CLOUDFLARED = {
  url: "https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64",
  sha256: "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134",
};

// The weights of the boot below: with weights_source r2 the trees come from the R2 weights store (R2_FETCH_SH,
// r2weights.ts) and the aux rows from their pinned URLs; with hub, the weight trees of FV_WEIGHTS_TREES_B64
// (weightsPlan). A
// tree counts as present only with fetch-hub-tree.py's `.complete` AND our
// `.fv-revision` holding its pinned revision (written after the fetch): a Brev
// VM restarted from a stop keeps /home/ubuntu/workspace/weights, so those
// trees are skipped (warm restart); one at another revision, or a leftover
// without both marks, is moved aside and removed (the VM's own disk, never a
// shared volume), then fetched again. `fetch_tree` is the boot's (tests stub
// it). The Hub fetcher is only set up when a tree is missing.
export const WEIGHTS_LOOP_SH = `if [ "\${FV_WEIGHTS_SOURCE:-none}" = hub ] || [ "\${FV_WEIGHTS_SOURCE:-none}" = r2 ]; then
  [ "$FV_WEIGHTS_SOURCE" = hub ] && say "weights: Hub trees at pinned revisions (owner-approved)"
  rm -rf "$FV_WEIGHTS"/.*.partial-* "$FV_WEIGHTS"/.stale-* 2>/dev/null || true
  report weights "" "checking"
  n=0; kept=0; got=0
  while IFS="$(printf '\\t')" read -r kind a b c d; do
    n=$((n + 1))
    case "$kind" in
      tree)
        if [ -f "$FV_WEIGHTS/$a/.complete" ] && [ "$(cat "$FV_WEIGHTS/$a/.fv-revision" 2>/dev/null)" = "$c" ]; then say "tree $a: present at $c"; kept=$((kept + 1)); continue; fi
        if [ -e "$FV_WEIGHTS/$a" ]; then
          old="$FV_WEIGHTS/.stale-$(printf %s "$a" | tr '/' '_')-$(date +%s)"
          say "tree $a: not complete at $c: replaced"
          mv "$FV_WEIGHTS/$a" "$old" && rm -rf "$old" || fail "tree $a: cannot move the old copy aside"
        fi
        say "tree $a <- $b@$c"
        report weights "" "downloading $a"
        fetch_tree "$a" "$b" "$c" "$d" "$n"
        printf %s "$c" > "$FV_WEIGHTS/$a/.fv-revision" || fail "tree $a: revision mark"
        got=$((got + 1))
        ;;
      aux)
        p="$FV_WEIGHTS/$a"; mkdir -p "$(dirname "$p")"
        if [ -f "$p" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$c" ]; then continue; fi
        curl -fsSL --max-time 600 -o "$p.part" "$b" && [ "$(sha256sum "$p.part" | cut -d' ' -f1)" = "$c" ] && [ "$(stat -c %s "$p.part")" = "$d" ] && mv "$p.part" "$p" || fail "aux $a"
        ;;
    esac
  done <<EOF_TREES
$(printf "%s" "$FV_WEIGHTS_TREES_B64" | base64 -d)
EOF_TREES
  say "weights ready: $kept present, $got fetched"
fi
`;
export const WEIGHTS_SH = `${R2_FETCH_SH}${WEIGHTS_LOOP_SH}`;

// The tunnel of a GMI / Brev pod (outbound only; no inbound port), in the boot below. A named tunnel fv-control made
// for this pod (tunnels.ts): FV_TUNNEL_TOKEN, given to cloudflared in its environment (TUNNEL_TOKEN, not its argv)
// and dropped from fv-serve's, and its URL FV_TUNNEL_URL, known up front; reported once a connection registered.
// Else (named tunnels not configured) a quick tunnel: https://<random>.trycloudflare.com from cloudflared's log.
export const TUNNEL_SH = `# ---- the tunnel -> 127.0.0.1:8000
curl -fsSL --max-time 120 -o /usr/local/bin/cloudflared "${CLOUDFLARED.url}" || fail "cloudflared download"
echo "${CLOUDFLARED.sha256}  /usr/local/bin/cloudflared" | sha256sum -c - >/dev/null 2>&1 || fail "cloudflared sha256"
chmod +x /usr/local/bin/cloudflared
tlog() { tail -3 /fvstate/tunnel.log | sed 's/eyJhIjoi[A-Za-z0-9+/_=-]*/[redacted]/g' | tr '\\n"' '  '; }
U=""
if [ -n "\${FV_TUNNEL_TOKEN:-}" ]; then
  say "named tunnel: $FV_TUNNEL_URL"
  TUNNEL_TOKEN="$FV_TUNNEL_TOKEN" /usr/local/bin/cloudflared tunnel --no-autoupdate run >/fvstate/tunnel.log 2>&1 &
  unset FV_TUNNEL_TOKEN
  for _ in $(seq 1 90); do
    if grep -q 'Registered tunnel connection' /fvstate/tunnel.log; then U="$FV_TUNNEL_URL"; break; fi
    sleep 2
  done
  [ -n "$U" ] || fail "named tunnel did not connect: $(tlog)"
else
  /usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >/fvstate/tunnel.log 2>&1 &
  for _ in $(seq 1 90); do
    U="$(grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' /fvstate/tunnel.log | head -1)"
    [ -n "$U" ] && break
    sleep 2
  done
  [ -n "$U" ] || fail "no tunnel URL: $(tlog)"
fi
say "tunnel $U"
[ "\${FV_DISPATCH_FRONT:-0}" = 1 ] && export FV_DISPATCH_ENDPOINT="$U"
report tunnel "$U" ""
`;

/** The Hub fetcher of a boot with weights_source hub (GMI / Brev PROVIDER_BOOT, the Runpod worker boot, a serverless
 * worker): `fetch_tree` sets up a venv with huggingface_hub and scripts/gpu/fetch-hub-tree.py (at the image's commit) on
 * first use, then fetches one tree at its pinned revision (weights-revisions.tsv) into $FV_WEIGHTS/<dest>, every file
 * checked against the Hub listing (fetch-hub-tree.py renames the tree into place only then), and logs its time as a
 * boot step (`hub_step`, which a boot may redefine after this block to report it too). With FV_HUB_SHA256_B64
 * (dest<TAB>path<TAB>bytes<TAB>sha256[|sha256]: scripts/gpu/weights-sha256.tsv's rows, weights-sources.ts HUB_SHA256)
 * every recorded file of the tree must also match the SHA-256 the fetcher computed and the recorded size, or the tree
 * is removed (the container's own disk) and the boot fails before fv-serve starts. */
export const HUB_FETCH_TREE_SH = `FETCHER=""
hub_step() { say "step $1: $2s"; }
hub_verify() { # dest sha256.txt: the tree's weights-sha256.tsv rows (FV_HUB_SHA256_B64) against the fetcher's hashes
  [ -n "\${FV_HUB_SHA256_B64:-}" ] || return 0
  hv_n=0; hv_bad=""
  while IFS="$(printf '\\t')" read -r hd hp hb hh; do
    [ "$hd" = "$1" ] || continue
    hv_n=$((hv_n + 1))
    hg="$(awk -v p="$hp" '{ h = $1; sub(/^[0-9a-f]+  /, ""); if ($0 == p) print h }' "$2" 2>/dev/null | head -n 1)"
    case "|$hh|" in *"|$hg|"*) ;; *) hv_bad="$hv_bad $hp" ;; esac
    if [ "$hb" != "-" ] && [ "$(stat -c %s "$FV_WEIGHTS/$1/$hp" 2>/dev/null || echo none)" != "$hb" ]; then hv_bad="$hv_bad $hp(size)"; fi
  done <<EOF_SHA
$(printf "%s" "$FV_HUB_SHA256_B64" | base64 -d)
EOF_SHA
  if [ -n "$hv_bad" ]; then
    hv_old="$FV_WEIGHTS/.stale-$(printf %s "$1" | tr '/' '_')-$(date +%s)"
    mv "$FV_WEIGHTS/$1" "$hv_old" && rm -rf "$hv_old"
    fail "tree $1: not the recorded SHA-256 (weights-sha256.tsv):$hv_bad"
  fi
  if [ "$hv_n" = 0 ]; then say "tree $1: no weights-sha256.tsv rows: checked against the Hub listing only"; else say "tree $1: $hv_n files match weights-sha256.tsv"; fi
}
fetch_tree() { # dest repo revision globs n: one tree into $FV_WEIGHTS/<dest> (the fetcher is set up on first use)
  if [ -z "$FETCHER" ]; then
    t1="$(date +%s)"
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-venv python3-pip; } >>/fvstate/apt.log 2>&1 || fail "apt python3-venv"
    python3 -m venv /opt/fv/venv && /opt/fv/venv/bin/pip install -q "huggingface_hub>=0.34" >/fvstate/pip.log 2>&1 || fail "pip huggingface_hub"
    mkdir -p /opt/fv/scripts && curl -fsSL --max-time 60 -o /opt/fv/scripts/fetch-hub-tree.py "$FV_SCRIPTS_URL/fetch-hub-tree.py" || fail "fetch-hub-tree.py"
    FETCHER=1
    hub_step hub_tools "$(( $(date +%s) - t1 ))"
  fi
  mkdir -p "/fvstate/fetch-$5"; t0="$(date +%s)"
  FETCH_REPO="$2" FETCH_REVISION="$3" FETCH_DEST="$1" FETCH_GLOBS="$4" FETCH_WEIGHTS="$FV_WEIGHTS" FETCH_SRV="/fvstate/fetch-$5" FETCH_MIN_FREE_GB=5 \\
    /opt/fv/venv/bin/python /opt/fv/scripts/fetch-hub-tree.py >"/fvstate/fetch-$5.out" 2>&1 || fail "tree $1: $(tail -2 "/fvstate/fetch-$5/log.txt" 2>/dev/null | tr '\\n"' '  ')"
  # fetch-hub-tree.py records its outcome in DONE (0 = the tree was renamed into place after every check).
  [ "$(cat "/fvstate/fetch-$5/DONE" 2>/dev/null)" = 0 ] || fail "tree $1: $(grep -v '^ ' "/fvstate/fetch-$5/log.txt" 2>/dev/null | tail -2 | tr '\\n"' '  ')"
  hub_verify "$1" "/fvstate/fetch-$5/sha256.txt"
  hub_step "hub $1" "$(( $(date +%s) - t0 ))"
}
`;

// The start command of a GMI / Brev worker (inside our image, as root): the
// weights (hub), the tunnel and its report, the deadline watchdog, then
// fv-serve with the same config handling as WORKER_BOOT (payloads.ts).
export const PROVIDER_BOOT = `set -u
say() { echo "[fv-boot] $*" >&2; }
echo "[fv-boot] start ($FV_PROVIDER $FV_POD_ID)" >&2
mkdir -p /fvstate "$FV_WEIGHTS"
report() {
  [ -n "\${FV_ENDPOINT_REPORT_URL:-}" ] || return 0
  for _ in 1 2 3 4 5; do
    curl -sS --max-time 20 -X POST -H "Authorization: Bearer $FV_ENDPOINT_REPORT_TOKEN" -H "content-type: application/json" \\
      -d "{\\"pod\\":\\"$FV_POD_ID\\",\\"phase\\":\\"$1\\",\\"url\\":\\"$2\\",\\"detail\\":\\"$3\\"}" "$FV_ENDPOINT_REPORT_URL" >/dev/null && return 0
    sleep 5
  done
}
fail() { say "FAILED: $1"; report failed "" "$1"; sleep 600; exit 1; }
${DRIVER_CHECK_SH}if ! command -v curl >/dev/null 2>&1; then
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl ca-certificates; } >/fvstate/apt.log 2>&1 || fail "no curl"
fi
# ---- preflight: the binary and the config (as WORKER_BOOT makes it), checked before the weights and the tunnel.
# fv-serve exits at once on a bad config ("fv-serve: config: ...", before any log is shipped): say why instead.
FV_SERVE_BIN=/opt/fastvideo-rs/bin/fv-serve
[ -x "$FV_SERVE_BIN" ] || fail "no $FV_SERVE_BIN in the image"
if [ -n "\${FV_WORKER_TOML_B64:-}" ]; then
  printf "%s" "$FV_WORKER_TOML_B64" | base64 -d > /fv-worker.toml || fail "FV_WORKER_TOML_B64 does not decode"
else
  [ -f "$FV_WORKER_CONFIG" ] || fail "no worker config $FV_WORKER_CONFIG in the image"
  cp "$FV_WORKER_CONFIG" /fv-worker.toml || fail "cannot copy $FV_WORKER_CONFIG"
fi
if grep -q "^\\[gateway\\]" /fv-worker.toml; then
  sed -i "/^\\[gateway\\]/a register = false" /fv-worker.toml
else
  printf "\\n[gateway]\\nregister = false\\n" >> /fv-worker.toml
fi
export FV_WORKER_ID="$FV_POD_ID"
cfgerr="$("$FV_SERVE_BIN" --config /fv-worker.toml --print-config 2>&1 >/dev/null)" || fail "fv-serve config: $(printf '%s' "$cfgerr" | tail -n 3 | tr '\\n"' '  ')"
say "config ok: /fv-worker.toml"
${UPSAMPLER_START_SH}# ---- weights: the Hub at pinned revisions (fetch-hub-tree.py checks every file against the Hub listing), aux files by SHA-256
${HUB_FETCH_TREE_SH}${WEIGHTS_SH}${TUNNEL_SH}# ---- the deadline watchdog (fv-control's cron deletes the pod at the deadline as well)
(
  while :; do
    if [ "$(date +%s)" -ge "$FV_CLUSTER_DEADLINE" ]; then
      say "[watchdog] deadline: $FV_POD_ID"
      if [ "$FV_PROVIDER" = gmi ] && [ -n "\${FV_BACKSTOP_API_KEY:-}" ]; then
        id="$(curl -sS --max-time 30 -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$FV_BACKSTOP_API/v1/containers" | tr '{' '\\n' | grep "\\"name\\":\\"$FV_POD_NAME\\"" | sed -n 's/.*"id":"\\([0-9a-fA-F-]*\\)".*/\\1/p' | head -1)"
        [ -n "$id" ] && curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$FV_BACKSTOP_API/v1/containers/$id" >/dev/null
      fi
      pkill -f fv-serve
      sleep 60
    fi
    sleep 30
  done
) &
${UPSAMPLER_WAIT_SH}# ---- fv-serve (its config was checked before the weights)
say "fv-serve: $FV_SERVE_BIN --config /fv-worker.toml (from \${FV_WORKER_TOML_B64:+inline FV_WORKER_TOML_B64}\${FV_WORKER_TOML_B64:-$FV_WORKER_CONFIG}); $("$FV_SERVE_BIN" --version 2>&1 | head -n 1)"
if [ -n "\${FASTVIDEO_STREAMING_FETCH:-}" ]; then date +%s.%N > "\${FV_R2_STATE:-/fvstate}/serve-start"; say "fv-serve starts while the weights arrive (its loader waits on each file's progress map)"; fi
exec "$FV_SERVE_BIN" --config /fv-worker.toml`;

// A Runpod worker with weights_source hub (no volume; the owner approved the download: hub_download_approved and
// FV_HUB_DOWNLOADS_APPROVED, checked before the launch): the pod's trees from the Hub at their pinned revisions onto the
// container disk ($FV_WEIGHTS; never a network volume), then the aux files by SHA-256, as on GMI / Brev
// (WEIGHTS_LOOP_SH). Progress is logged ("[fv-boot] ..." and the "step hub <tree>" times) and reported when the pod has
// a report URL. A failure exits after 30 s: Runpod restarts the container and the complete trees are kept.
// Inserted into the worker boot by payloads.ts bootFor.
export const HUB_RUNPOD_SH = `export FV_POD_ID="\${FV_POD_ID:-\${RUNPOD_POD_ID:-}}"
if [ "\${FV_WEIGHTS_SOURCE:-none}" = hub ]; then
  say() { echo "[fv-boot] $(date -u +%H:%M:%S) $*" >&2; }
  report() {
    [ -n "\${FV_ENDPOINT_REPORT_URL:-}" ] || return 0
    curl -sS --max-time 20 -X POST -H "Authorization: Bearer $FV_ENDPOINT_REPORT_TOKEN" -H "content-type: application/json" \\
      -d "{\\"pod\\":\\"$FV_POD_ID\\",\\"phase\\":\\"$1\\",\\"url\\":\\"$2\\",\\"detail\\":\\"$3\\"}" "$FV_ENDPOINT_REPORT_URL" >/dev/null || true
  }
  fail() { say "FAILED: $1"; report failed "" "$1"; sleep 30; exit 1; }
  if ! command -v curl >/dev/null 2>&1; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl ca-certificates; } >/fvstate/apt.log 2>&1 || fail "no curl"
  fi
  mkdir -p "$FV_WEIGHTS"
  hub_t0="$(date +%s)"
${HUB_FETCH_TREE_SH}${WEIGHTS_LOOP_SH}  say "step weights: $(( $(date +%s) - hub_t0 ))s (Hub)"
fi
`;

/** A Brev VM's run script for one launch (brev-park.ts): our image running PROVIDER_BOOT with the pod's env. */
export const brevRun = (image: string, env: Record<string, string>) => brevRunScript(image, env, PROVIDER_BOOT);

// ---------------------------------------------------------------- lifecycle (cluster/ops.ts and the DO call these)
async function findInstance(env: Env, podId: string): Promise<ProviderInstance | null> {
  const k = splitKey(podId);
  if (!k) return null;
  // A Brev instance in DELETING has its delete accepted (it takes Brev ~5-10 min to finish): it counts as gone, so a
  // stop's verify does not time out on it and a retry does not delete it again.
  return (await providerImpl(k.provider).list(env)).find((i) => i.name === k.name && i.raw !== "DELETING") || null;
}
/** Deletes a GMI / Brev pod by its key; true when it is gone (or was). Only an fv-named instance can match. */
export async function deleteOtherPod(env: Env, podId: string): Promise<boolean> {
  const inst = await findInstance(env, podId);
  if (!inst) return true;
  return providerImpl(inst.provider).remove(env, inst);
}
/** Releases a GMI / Brev pod. GMI: deleted. Brev: every stop stops the VM, parked (weights complete) or held (with
 * why), never deleted once the stop succeeded (owner rule 2026-10-11); a stop that does not succeed deletes it (owner
 * rule 2026-10-13: "deleted"); one still DEPLOYING is stop_pending (the cron's). One with no keep-on-stop record is
 * stopped and recorded held. `manual` (the owner's `pod delete` of a standalone pod) deletes a Brev VM. */
export async function releaseOtherPod(env: Env, podId: string, o: { manual?: boolean; why?: string } = {}, log?: (m: string) => void): Promise<Release> {
  if (splitKey(podId)?.provider === "brev") {
    const r = await releaseBrev(env, podId, o, log);
    if (r !== "none") return r;
    if (!o.manual) {
      const inst = await findInstance(env, podId);
      if (!inst) return "deleted"; // gone (or in the owner's delete)
      // A workspace fv-control already deleted (a failed stop, the owner) that Brev still lists for a moment: gone.
      const known = await brevRowById(env, inst.id);
      if (known && (known.state === "deleted" || known.state === "deleting" || known.state === "gone")) return "deleted";
      return adoptAndStop(env, { workspace_id: inst.id, pod_id: podId, name: inst.name, instance_type: inst.gpu || "?" }, o.why || "stop", log);
    }
  }
  if (!(await deleteOtherPod(env, podId))) throw new Error("the provider did not confirm the delete");
  // Its named tunnel and DNS record go with it (a parked / held Brev instance keeps them: its restart reuses them).
  for (const m of await releaseTunnels(env, podId, "its pod was deleted")) log?.(`${podId}: ${m}`);
  return "deleted";
}
/** A GMI / Brev pod's state for the boot watch; null when the provider no longer lists it. */
export async function otherPodState(env: Env, podId: string): Promise<{ desiredStatus: string; uptimeS: number | null; state: InstanceState; reason: string | null; raw?: string } | null> {
  const inst = await findInstance(env, podId);
  if (!inst) return null;
  return {
    desiredStatus: inst.state === "running" ? "RUNNING" : inst.state.toUpperCase(),
    uptimeS: inst.createdAt ? Math.max(0, Math.round((now() - inst.createdAt) / 1000)) : null,
    state: inst.state,
    reason: inst.reason,
    raw: inst.raw,
  };
}
export async function otherPodLogs(env: Env, podId: string): Promise<string[]> {
  const inst = await findInstance(env, podId);
  const impl = inst ? providerImpl(inst.provider) : null;
  if (!inst || !impl?.logs) return [];
  return impl.logs(env, inst);
}
/** The https URL a GMI / Brev pod reported (its tunnel); null until it has. */
export async function reportedUrl(env: Env, podId: string): Promise<string | null> {
  const r = await env.DB.prepare("SELECT url FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ url: string | null }>();
  return r?.url || null;
}
/** Where a pod answers: Runpod's proxy, or the tunnel a GMI / Brev pod reported (null until then). */
export async function podBaseUrl(env: Env, podId: string, runpodUrl: (pod: string) => string): Promise<string | null> {
  return isOtherPod(podId) ? reportedUrl(env, podId) : runpodUrl(podId);
}

export interface PodReport {
  phase: string;
  detail: string;
  at: number;
  /** The host's last heartbeat (Brev) and what it said. */
  heartbeat_at?: number;
  heartbeat?: string;
}
/** What a GMI / Brev pod reported last (POST /ingest/v1/endpoint): failed boots show here. Heartbeats do not change the phase. */
export async function lastReport(env: Env, podId: string): Promise<PodReport | null> {
  const r = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(`endpoint:${podId}`).first<{ value: string }>();
  return r ? parseJson(r.value, null) : null;
}
/** Every report of a pod but heartbeats (the last REPORT_HISTORY), oldest first. */
export async function reportHistory(env: Env, podId: string): Promise<{ phase: string; detail: string; at: number }[]> {
  const r = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(`endpoint-log:${podId}`).first<{ value: string }>();
  return r ? parseJson(r.value, []) : [];
}
const REPORT_HISTORY = 30;
/** The reports as log lines (provider-logs: Brev keeps no log of its own). */
export const reportLines = (h: { phase: string; detail: string; at: number }[], last: PodReport | null): string[] => [
  ...h.map((x) => `[report ${new Date(x.at).toISOString()}] ${x.phase}${x.detail ? `: ${x.detail}` : ""}`),
  ...(last?.heartbeat_at ? [`[report ${new Date(last.heartbeat_at).toISOString()}] heartbeat (last): ${last.heartbeat || ""}`] : []),
];

/** A starting GMI / Brev pod's boot diagnosis (the `up` wait): the provider's state and the pod's own last report. */
export async function otherPodDiag(env: Env, podId: string, rt: { state: InstanceState; reason: string | null; raw?: string } | undefined): Promise<{ phase: string; detail: string; fatal: boolean; replace?: boolean } | null> {
  const rep = await lastReport(env, podId);
  if (rep?.phase === "failed") return { phase: "boot failed", detail: rep.detail || "the pod reported a failure", fatal: true };
  if (rep?.phase === "host_failed") return { phase: "host failed", detail: rep.detail || "the VM's host script reported a failure", fatal: true };
  if (rep?.phase === "container_exited") return { phase: "container exited", detail: rep.detail || "the container exited", fatal: true };
  if (rep?.phase === "serve_timeout") return { phase: "serve timeout", detail: rep.detail || "fv-serve never answered /health", fatal: true };
  if (!rt) return null;
  // A parked Brev instance being started again: stopped / starting is expected until BREV_RESTART_TIMEOUT_S.
  if (splitKey(podId)?.provider === "brev") {
    const w = await restartDiag(env, podId, rt.state, [rt.raw, rt.reason].filter(Boolean).join(": ") || null);
    if (w) return w;
  }
  if (rt.state === "failed") return { phase: "provider error", detail: rt.reason || "the provider reports an error", fatal: true };
  if (rt.state === "stopped") return { phase: "stopped", detail: "stopped outside fv-control", fatal: true };
  if (rt.state === "starting") return { phase: "provisioning", detail: rt.reason || "", fatal: false };
  const hb = rep?.heartbeat_at ? `; host heartbeat ${Math.round((now() - rep.heartbeat_at) / 1000)} s ago: ${(rep.heartbeat || "").slice(0, 160)}` : "";
  if (rep?.phase === "weights") return { phase: "weights", detail: /^r2 /.test(rep.detail || "") ? `fetching from the R2 weights store (${rep.detail.slice(3)})` : `downloading from the Hub${rep.detail ? ` (${rep.detail})` : ""}`, fatal: false };
  if (rep?.phase === "tunnel") return { phase: "loading", detail: `tunnel up, fv-serve loading${hb}`, fatal: false };
  // upsampler_local (upsampler.ts): the sidecar's outcome, just before fv-serve starts; a failure is not fatal.
  if (rep?.phase === "upsampler") return { phase: "loading", detail: `upsampler sidecar: ${(rep.detail || "").slice(0, 200)}; fv-serve starting${hb}`, fatal: false };
  if (rep?.phase === "serve_ready") return { phase: "loading", detail: `fv-serve answers /health on the host (${rep.detail}); waiting for it through the tunnel`, fatal: false };
  if (rep?.phase === "container_started") return { phase: "container started", detail: `the boot runs in the container${hb}`, fatal: false };
  if (rep?.phase === "host") return { phase: "host", detail: `host script running (${rep.detail.slice(0, 200)})${hb}`, fatal: false };
  return { phase: "booting", detail: `no report from the pod yet${hb}`, fatal: false };
}

/** weights_source r2 (any provider): every tree's manifest read from the bucket (checked against the repo's pin), and
 * what the pod will fetch; null without r2. An unreadable manifest is an issue: the pod could not fetch it either. */
export async function r2Outlook(env: Env, pool: PoolSpec): Promise<{ ok: boolean; note: string } | null> {
  const plan = weightsPlan(pool, env);
  const { cfg } = r2Cfg(env);
  if (plan.source !== "r2" || !cfg || plan.unsupported.length || !plan.trees.length) return null;
  try {
    const ms = await Promise.all(plan.trees.map((t) => r2Manifest(env, cfg, t)));
    const total = ms.reduce((s, m) => s + m.total_bytes, 0);
    return { ok: true, note: `r2: fetches ${ms.map((m) => `${m.dest} ${(m.total_bytes / 1e9).toFixed(1)} GB`).join(", ")} = ${(total / 1e9).toFixed(1)} GB from the R2 weights store (manifests checked; ~${Math.max(1, Math.round(total / 1e9 / R2_DOWNLOAD_GBPS / 60))} min at ${R2_DOWNLOAD_GBPS} GB/s)` };
  } catch (e) {
    return { ok: false, note: `weights_source r2: ${(e as Error).message.slice(0, 300)}` };
  }
}

/** The URL a pod reports must be a quick-tunnel URL (tests: FV_ENDPOINT_URL_RE). */
export function endpointUrlOk(env: Env & { FV_ENDPOINT_URL_RE?: string }, url: string): boolean {
  const re = env.FV_ENDPOINT_URL_RE ? new RegExp(env.FV_ENDPOINT_URL_RE) : /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/;
  return url.length <= 300 && re.test(url);
}

/** POST /ingest/v1/endpoint: {pod, phase: weights | tunnel | failed, url?, detail?} from a GMI / Brev pod with its
 * cluster's ingest token. `tunnel` sets the pod's URL (a quick-tunnel URL only); `failed` fails its boot (the `up` wait). */
export async function endpointReport(env: Env, req: Request): Promise<{ ok: true; pod: string; phase: string }> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "ingest token required");
  const cl = await env.DB.prepare("SELECT id FROM clusters WHERE ingest_hash = ?").bind(await sha256Hex(auth.slice(7).trim())).first<{ id: string }>();
  if (!cl) throw new HttpError(401, "invalid ingest token");
  if (!(await rateLimit(env, `endpoint:${cl.id}`, 60, 60))) throw new HttpError(429, "endpoint report rate limit (60/min per cluster)");
  const text = await req.text();
  if (text.length > 4096) throw new HttpError(413, "report too large");
  const b = parseJson<any>(text, null);
  const pod = typeof b?.pod === "string" ? b.pod : "";
  const phase = typeof b?.phase === "string" ? b.phase : "";
  if (!isOtherPod(pod)) throw new HttpError(400, "pod: a GMI / Brev pod id");
  if (!REPORT_PHASES.includes(phase)) throw new HttpError(400, `phase: ${REPORT_PHASES.join(" | ")}`);
  const row = await env.DB.prepare("SELECT cluster_id FROM cluster_pods WHERE pod_id = ? AND deleted_at IS NULL").bind(pod).first<{ cluster_id: string }>();
  if (!row || row.cluster_id !== cl.id) throw new HttpError(404, "no live pod of this cluster with that id");
  if (phase === "tunnel") {
    const url = typeof b?.url === "string" ? b.url : "";
    // A pod with a named tunnel (tunnels.ts) may report only its own host's URL; any other pod a quick-tunnel URL.
    const want = await expectedTunnelUrl(env, pod);
    if (want ? url !== want : !endpointUrlOk(env, url)) throw new HttpError(400, want ? "url: this pod's named-tunnel URL (the one fv-control gave it in FV_TUNNEL_URL)" : "url: a Cloudflare quick-tunnel URL (https://<name>.trycloudflare.com)");
    await env.DB.prepare("UPDATE cluster_pods SET url = ? WHERE pod_id = ?").bind(url, pod).run();
  }
  // The host's facts and failures (a log tail) are longer than the container's one-line phases.
  const detail = typeof b?.detail === "string" ? b.detail.slice(0, LONG_REPORTS.includes(phase) ? 2000 : 300) : "";
  const t = now();
  const prev = await lastReport(env, pod);
  if (phase === "heartbeat") {
    // A heartbeat keeps the phase: it only says the host script is alive (and what the container does).
    await putSetting(env, `endpoint:${pod}`, { ...(prev ?? { phase: "heartbeat", detail: "", at: t }), heartbeat_at: t, heartbeat: detail }, "pod");
    return { ok: true, pod, phase };
  }
  await putSetting(env, `endpoint:${pod}`, { phase, detail, at: t, ...(prev?.heartbeat_at ? { heartbeat_at: prev.heartbeat_at, heartbeat: prev.heartbeat } : {}) }, "pod");
  const hist = [...(await reportHistory(env, pod)), { phase, detail, at: t }].slice(-REPORT_HISTORY);
  await putSetting(env, `endpoint-log:${pod}`, hist, "pod");
  // Into the boot timeline (and the pod's log): the host script, the container, the tunnel.
  // A failure marks no milestone (the phase was not reached); it is the note of the phase it ended.
  const m = (
    { host: "host_script", host_failed: "host_script", container_started: "container_start", container_exited: "container_start", tunnel: "tunnel", serve_ready: "serve_ready", serve_timeout: "serve_ready" } as const
  )[phase as "host"];
  const bad = ["host_failed", "container_exited", "serve_timeout"].includes(phase);
  if (m) await recordBoot(env, pod, cl.id, [], bad ? {} : { [m]: t }, { [m]: bad ? `${phase.toUpperCase()}: ${detail.slice(-300)}` : detail.slice(0, 300) });
  return { ok: true, pod, phase };
}
/** What a pod may report: the container's boot (weights, tunnel, failed) and a Brev host's (host, host_failed,
 * container_started, heartbeat, then serve_ready / container_exited / serve_timeout from its watcher). */
export const REPORT_PHASES = ["weights", "tunnel", "failed", "host", "host_failed", "container_started", "heartbeat", "serve_ready", "container_exited", "serve_timeout", "upsampler"];
/** Reports that carry a log tail (up to 2000 chars). */
const LONG_REPORTS = ["host", "host_failed", "container_exited", "serve_timeout"];

/** weights_source hub (any provider): the owner's approval for this launch AND on this fv-control (CLAUDE.md: large
 * downloads need approval), a plain Hub source for every tree, and some tree to fetch. */
function hubIssues(env: Env, p: PoolSpec, i: number): { path: (string | number)[]; message: string }[] {
  const out: { path: (string | number)[]; message: string }[] = [];
  const plan = weightsPlan(p, env);
  if (!p.hub_download_approved) out.push({ path: ["pools", i, "hub_download_approved"], message: `weights_source hub downloads ${plan.trees.join(", ") || "the weights"} from the Hub at every boot: it needs the owner's approval for this launch (weights_download_approved: true)` });
  else if (env.FV_HUB_DOWNLOADS_APPROVED !== "1") out.push({ path: ["pools", i, "hub_download_approved"], message: "Hub downloads at boot are not approved on this fv-control (the owner sets FV_HUB_DOWNLOADS_APPROVED=1; CLAUDE.md: large downloads need approval)" });
  if (plan.unsupported.length) out.push({ path: ["pools", i, "weights_source"], message: `no plain Hub source for ${plan.unsupported.join(", ")} (several repos or a conversion: docs/ops/runpod-volumes.md); serve another preset${isOtherPool(p) ? " or use Runpod" : ", r2 or the volume"}` });
  if (!plan.trees.length) out.push({ path: ["pools", i, "weights_source"], message: "the pool's config loads no weight tree fv-control knows: weights_source none" });
  return out;
}
/** A Runpod pod's container disk for these Hub trees (the size table, brev-park.ts treesGb): × 1.15 + 30 GB. */
export const hubDiskGb = (trees: string[]) => Math.ceil(treesGb(trees) * 1.15 + 30);

// ---------------------------------------------------------------- NVIDIA Brev: the VM's disk
export { BREV_DISK_MAX_GB, RUNPOD_DISK_MAX_GB } from "./enums";
/** The least disk a Brev VM needs for these trees: the trees + 40 GB (image, Docker, scratch), as a reuse checks
 * (brev-park.ts fitsDisk); at least 50 GB. */
export const brevDiskNeedGb = (trees: string[]) => Math.max(50, Math.ceil(treesGb(trees) + 40));
/** The disk a Brev VM is created with (GB): the pool's container_disk_gb when set, else fv-control's size for the trees
 * (brev-park.ts brevDiskGb: x1.3 + 40, at least 200), kept inside the type's elastic range when the list gives one. */
export function brevDisk(info: Pick<BrevType, "disk_min_gb" | "disk_max_gb"> | null, trees: string[], want?: number): number {
  if (want !== undefined) return want;
  let d = brevDiskGb(trees);
  if (info?.disk_max_gb) d = Math.min(d, info.disk_max_gb);
  if (info?.disk_min_gb) d = Math.max(d, info.disk_min_gb);
  return d;
}
/** What a Brev instance type allows for its disk, in words (the forms' help, the validator). */
export function brevDiskRule(info: Pick<BrevType, "type" | "elastic_disk" | "fixed_disk_gb" | "disk_min_gb" | "disk_max_gb"> | null): string {
  if (!info) return "the instance type is not in Brev's list now: its disk range is unknown";
  if (!info.elastic_disk && info.fixed_disk_gb) return `${info.type} has a fixed ${info.fixed_disk_gb} GB disk (Brev does not size it)`;
  if (info.elastic_disk) return `${info.type}: an elastic disk of ${info.disk_min_gb ?? "?"}-${info.disk_max_gb ?? "?"} GB, sized at create`;
  return `${info.type}: Brev's list gives no disk size`;
}
/**
 * A Brev pool's disk (container_disk_gb, the VM's diskStorage) against its instance type (the live list): a fixed-disk
 * type takes no other size; an elastic one its storage classes' range. Path-anchored issues; none when the list is
 * unreachable or the type is not in it (the create then says).
 */
export async function brevDiskIssues(env: Env, spec: { pools: PoolSpec[] }): Promise<{ path: (string | number)[]; message: string }[]> {
  const out: { path: (string | number)[]; message: string }[] = [];
  for (const [i, p] of spec.pools.entries()) {
    if (poolProvider(p) !== "brev" || !p.provider_gpu || p.container_disk_gb === undefined) continue;
    const info = await brev.type(env, p.provider_gpu);
    if (!info) continue;
    const v = p.container_disk_gb;
    const at = ["pools", i, "container_disk_gb"];
    if (!info.elastic_disk && info.fixed_disk_gb) {
      if (v !== info.fixed_disk_gb) out.push({ path: at, message: `${brevDiskRule(info)}: leave the disk unset (or ${info.fixed_disk_gb})` });
    } else if (info.disk_min_gb !== null && v < info.disk_min_gb) out.push({ path: at, message: `${v} GB is below ${info.type}'s smallest disk (${info.disk_min_gb} GB)` });
    else if (info.disk_max_gb !== null && v > info.disk_max_gb) out.push({ path: at, message: `${v} GB is above ${info.type}'s largest disk (${info.disk_max_gb} GB)` });
  }
  return out;
}

/** Every reason a spec's GMI / Brev pools cannot start, at the field it is about (the launch form, `up`'s first step). */
export function providerIssues(env: Env, spec: { pools: PoolSpec[] }): { path: (string | number)[]; message: string }[] {
  const out: { path: (string | number)[]; message: string }[] = [];
  spec.pools.forEach((p, i) => {
    if (!isOtherPool(p)) {
      // A Runpod pool may download from the Hub onto its container disk (the owner's approval, as on GMI / Brev).
      if (weightsSource(p, env) === "hub") {
        out.push(...hubIssues(env, p, i));
        if (p.volume === true) out.push({ path: ["pools", i, "volume"], message: "weights_source hub downloads onto the container disk: leave volume off (the volume is not mounted with hub)" });
        const need = hubDiskGb(weightsPlan(p, env).trees);
        if (p.container_disk_gb !== undefined && p.container_disk_gb < need) out.push({ path: ["pools", i, "container_disk_gb"], message: `${p.container_disk_gb} GB is too small for ${weightsPlan(p, env).trees.join(", ")} from the Hub: at least ${need} GB (unset: fv-control sizes it)` });
        if (need > 500) out.push({ path: ["pools", i, "weights_source"], message: `${weightsPlan(p, env).trees.join(", ")} need ${need} GB of container disk, over Runpod's 500 GB: use the volume or r2` });
        return;
      }
      // A Runpod pool may fetch from the R2 weights store onto its container disk instead of mounting the volume.
      if (weightsSource(p, env) !== "r2") return;
      const plan = weightsPlan(p, env);
      const off = r2Cfg(env).off;
      if (off) out.push({ path: ["pools", i, "weights_source"], message: `weights_source r2: ${off}` });
      if (plan.unsupported.length) out.push({ path: ["pools", i, "weights_source"], message: `not in the R2 weights store yet: ${plan.unsupported.join(", ")} (scripts/gpu/r2-weights.py status; docs/ops/r2-weights.md §6): use the volume` });
      if (!plan.trees.length) out.push({ path: ["pools", i, "weights_source"], message: "the pool's config loads no weight tree fv-control knows: weights_source none" });
      if (p.volume === true) out.push({ path: ["pools", i, "volume"], message: "weights_source r2 fetches onto the container disk: leave volume off (the volume is not mounted with r2)" });
      const need = r2DiskGb(env, plan.trees);
      if (p.container_disk_gb !== undefined && p.container_disk_gb < need) out.push({ path: ["pools", i, "container_disk_gb"], message: `${p.container_disk_gb} GB is too small for ${plan.trees.join(", ")} from R2: at least ${need} GB (unset: fv-control sizes it)` });
      if (need > 500) out.push({ path: ["pools", i, "weights_source"], message: `${plan.trees.join(", ")} need ${need} GB of container disk, over Runpod's 500 GB: use the volume` });
      return;
    }
    const prov = poolProvider(p) as OtherProviderId;
    const impl = providerImpl(prov);
    const off = impl.off(env);
    if (off) return out.push({ path: ["pools", i, "provider"], message: `${impl.title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)` });
    if (!env.PUBLIC_URL) out.push({ path: ["pools", i, "provider"], message: `PUBLIC_URL is not set: a ${impl.title} pod reports its tunnel URL to fv-control there` });
    const tun = tunnelSettings(env);
    if (tun.error) out.push({ path: ["pools", i, "provider"], message: `named pod tunnels: ${tun.error} (docs/serve/deploy-gmi-brev.md §6.4)` });
    const gpus = impl.gpus(env);
    if (p.provider_gpu && !gpus.includes(p.provider_gpu)) out.push({ path: ["pools", i, "provider_gpu"], message: `${p.provider_gpu} is not allowed on ${impl.title}: one of ${gpus.join(", ")} (${prov.toUpperCase()}_${prov === "gmi" ? "PRODUCTS" : "INSTANCE_TYPES"})` });
    const plan = weightsPlan(p, env);
    if (plan.source === "none" && !p.weights_source && (p.models?.length ?? 0) > 0)
      out.push({ path: ["pools", i, "weights_source"], message: `models need weights: r2 (the R2 weights store, once every tree of the pool is published: ${poolTrees(p).filter((t) => t !== "auxiliary").filter((t) => !r2Trees(env)[t]).join(", ") || "-"} missing), hub (owner approval) or serve fake_models` });
    if (plan.source === "r2") {
      const off = r2Cfg(env).off;
      if (off) out.push({ path: ["pools", i, "weights_source"], message: `weights_source r2: ${off}` });
      if (plan.unsupported.length) out.push({ path: ["pools", i, "weights_source"], message: `not in the R2 weights store yet: ${plan.unsupported.join(", ")} (scripts/gpu/r2-weights.py status; docs/ops/r2-weights.md §6): use hub or another preset` });
      if (!plan.trees.length) out.push({ path: ["pools", i, "weights_source"], message: "the pool's config loads no weight tree fv-control knows: weights_source none" });
    }
    if (plan.source === "hub") out.push(...hubIssues(env, p, i));
    // NVIDIA Brev: the VM's disk (container_disk_gb) holds the image and the trees (brevDiskIssues: the type's range).
    if (p.container_disk_gb !== undefined) {
      // (GMI: refused by the schema, schemas.ts providerRules.)
      if (prov === "brev") {
        const trees = plan.source === "none" ? [] : plan.trees;
        const need = brevDiskNeedGb(trees);
        if (p.container_disk_gb < need) out.push({ path: ["pools", i, "container_disk_gb"], message: `${p.container_disk_gb} GB is too small for the VM${trees.length ? ` and ${trees.join(", ")}` : ""}: at least ${need} GB (unset: fv-control sizes it)` });
      }
    }
  });
  return out;
}

export { OTHER_PROVIDERS };
