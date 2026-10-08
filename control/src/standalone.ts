// Standalone pods (docs/control/standalone-pods.md): one pod launched on its
// own, not part of a cluster. A standalone pod is stored as a one-pool,
// direct cluster with clusters.source = 'standalone', so it goes through
// the same code as cluster workers: normalizeSpec, the ClusterOps `up` /
// `down` operations, createWorker (placement, payload, env layers, the
// $/hr cap), the price check and the balance floor, the deadline backstop
// (the cron, and the pod's own watchdog), the cost ledger (owner
// pod:<name>), log shipping and the Runpod log capture, the idle stop.
import { normalizeSpec, presetPool, regionAvailable, REGIONS, type ClusterSpec, type PoolSpec, type RegionId } from "./cluster/spec";
import { livePods, type Cluster } from "./cluster/store";
import type { Env } from "./env";
import { validateVar } from "./envvars";
import { bootRows, getBoot } from "./boottime";
import { getDiagnosis } from "./podlogs";
import { HttpError, utcDay } from "./util";
import type { z } from "zod";
import { parseOr400, type StandaloneLaunchZ } from "./schemas";

/** The single pool's id inside a standalone pod's spec. */
export const STANDALONE_POOL = "pod";

/** A launch request (the schema: src/schemas.ts StandaloneLaunchZ, GET /api/schemas/standalone-launch). */
export type LaunchRequest = z.infer<typeof StandaloneLaunchZ>;

/** The region of a region id or a data centre id. */
export function regionOf(x: { region?: string; dc?: string }): RegionId {
  const want = x.region || x.dc || "eu";
  const byDc = (Object.keys(REGIONS) as RegionId[]).find((r) => REGIONS[r].dc === want);
  const r = (byDc || want) as RegionId;
  if (!regionAvailable(r)) throw new HttpError(400, `region: ${want} has no weights volume fv-control can use (EU only: eu / ${REGIONS.eu.dc}; CLAUDE.md)`);
  return r;
}

/** A launch request → the env it sets (validated) and a normalized one-pool direct cluster spec. */
export function standaloneSpec(input: LaunchRequest): { spec: ClusterSpec; env: { key: string; value: string; secret: boolean }[] } {
  if (!input || typeof input !== "object") throw new HttpError(400, "body: a launch request");
  input = parseOr400("standalone-launch", input);
  const base: Partial<PoolSpec> = input.preset ? presetPool(input.preset) ?? {} : {};
  const region = regionOf(input);
  const compute = input.compute ?? base.compute ?? (input.variant === "cpu" ? "CPU" : "GPU");
  const pool: PoolSpec = {
    ...(base as PoolSpec),
    id: STANDALONE_POOL,
    count: 1,
    compute,
    variant: input.variant || base.variant!,
    regions: [region],
  };
  if (input.config || input.config_toml) {
    delete pool.config;
    delete pool.config_toml;
    if (input.config) pool.config = input.config;
    if (input.config_toml) pool.config_toml = input.config_toml;
  }
  if (input.models) pool.models = input.models;
  if (input.fake_models) pool.fake_models = input.fake_models;
  const gpus = input.gpu_types?.length ? input.gpu_types : input.gpu_type ? [input.gpu_type] : undefined;
  if (gpus) pool.gpu_types = gpus;
  if (input.cpu_flavors) pool.cpu_flavors = input.cpu_flavors;
  if (input.vcpu !== undefined) pool.vcpu = input.vcpu;
  if (input.container_disk_gb !== undefined) pool.container_disk_gb = Number(input.container_disk_gb);
  pool.volume = input.volume ?? (compute === "GPU" ? (base.volume ?? true) : false);
  // GMI Cloud / NVIDIA Brev (docs/serve/deploy-gmi-brev.md): no Runpod volume; the weights from the Hub (approved) or none.
  if (input.provider && input.provider !== "runpod") {
    pool.provider = input.provider;
    pool.provider_gpu = input.provider_gpu;
    if (input.provider_region) pool.provider_region = input.provider_region;
    pool.compute = "GPU";
    pool.volume = false;
    pool.weights_source = input.weights_source ?? "none";
    if (input.weights_download_approved) pool.hub_download_approved = true;
    delete pool.gpu_types, delete pool.cpu_flavors, delete pool.vcpu;
  } else if (input.weights_source) pool.weights_source = input.weights_source;
  // An image reference is the pool's own image (resolved to a digest at start); the channel / sha the spec's source.
  if (input.image) pool.image = input.image;
  const deadlineMin = input.deadline_min ?? 60;
  const idle = input.idle_stop_min ?? null;
  const spec = normalizeSpec({
    name: input.name,
    image: input.sha ? { sha: input.sha } : { channel: input.channel || "stable" },
    regions: [region],
    control_plane: "direct",
    auth: input.auth ?? "keys",
    pools: [pool],
    cap_s: Math.round(deadlineMin * 60),
    ...(input.max_gpu_dph !== undefined ? { max_gpu_dph: Number(input.max_gpu_dph) } : {}),
    ...(input.min_balance !== undefined ? { min_balance: Number(input.min_balance) } : {}),
    ...(input.balance_floor !== undefined ? { balance_floor: Number(input.balance_floor) } : {}),
    min_start: input.min_start ?? Math.max(10, input.balance_floor ?? 8),
    auto_stop_idle_min: idle,
    log_shipping: true,
    ...(input.log_level ? { log_level: input.log_level } : {}),
  });
  const env: { key: string; value: string; secret: boolean }[] = [];
  for (const [key, v] of Object.entries(input.env || {})) {
    const value = typeof v === "string" ? v : String(v?.value ?? "");
    try {
      validateVar(key, value, typeof v === "object" && !!v?.secret);
    } catch (e) {
      throw new HttpError(400, `env: ${(e as Error).message}`);
    }
    env.push({ key, value, secret: typeof v === "object" && !!v?.secret });
  }
  return { spec, env };
}

/** What a standalone pod looks like in the API: its definition, its pod (if any), the operation, costs and boot diagnosis. */
export async function standaloneView(env: Env, c: Cluster, op: unknown) {
  const pool = c.spec.pools[0]!;
  const rec = (c.state.workers[STANDALONE_POOL] || [])[0] || null;
  const live = rec ? await env.DB.prepare("SELECT * FROM pods WHERE pod_id = ?").bind(rec.pod).first<any>() : null;
  const ctl = (await livePods(env, c.id))[0] || null;
  const day = utcDay(Date.now());
  const cost = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS total, COALESCE(SUM(CASE WHEN day = ? THEN usd ELSE 0 END), 0) AS today, COALESCE(SUM(minutes), 0) AS minutes FROM cost_daily WHERE cluster_id = ?")
    .bind(day, c.id)
    .first<{ total: number; today: number; minutes: number }>();
  const lastOp = await env.DB.prepare("SELECT id, kind, status, error, created_at, updated_at FROM operations WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 1").bind(c.id).first<any>();
  return {
    id: c.id,
    name: c.name,
    status: c.status,
    deadline: c.deadline,
    created_at: c.created_at,
    created_by: c.created_by,
    definition: {
      provider: pool.provider ?? "runpod",
      ...(pool.provider && pool.provider !== "runpod" ? { provider_gpu: pool.provider_gpu, provider_region: pool.provider_region ?? null, weights_source: pool.weights_source ?? "none", hub_download_approved: !!pool.hub_download_approved } : {}),
      variant: pool.variant,
      image: pool.image ? { image: pool.image } : c.spec.image,
      compute: pool.compute,
      gpu_types: pool.gpu_types || null,
      cpu_flavors: pool.compute === "CPU" ? pool.cpu_flavors || null : undefined,
      region: (pool.regions || c.spec.regions)[0],
      dc: REGIONS[(pool.regions || c.spec.regions)[0] as RegionId]?.dc,
      volume: pool.volume !== false && pool.compute === "GPU" ? REGIONS[(pool.regions || c.spec.regions)[0] as RegionId]?.volume : pool.volume ? REGIONS[(pool.regions || c.spec.regions)[0] as RegionId]?.volume : null,
      config: pool.config || (pool.config_toml ? "(inline config_toml)" : null),
      models: (pool.models || []).map((m) => m.id).concat(pool.fake_models || []),
      deadline_min: Math.round(c.spec.cap_s / 60),
      idle_stop_min: c.spec.auto_stop_idle_min ?? null,
      max_gpu_dph: c.spec.max_gpu_dph,
    },
    pod: rec
      ? {
          pod_id: rec.pod,
          url: rec.url || null,
          gpu: rec.gpu || (rec.cpu ? `cpu:${rec.cpu}` : null),
          dc: rec.dc || null,
          cost_per_hr: live?.cost_per_hr ?? rec.dph,
          image: rec.image,
          created_at: rec.created * 1000,
          ready_at: ctl?.ready_at ?? null,
          desired_status: live?.desired_status ?? null,
          health: live?.health ?? null,
          uptime_s: live?.uptime_s ?? null,
          gpu_util: live?.gpu_util ?? null,
          boot: await getDiagnosis(env, rec.pod),
          boot_timeline: bootRows(await getBoot(env, rec.pod)),
        }
      : null,
    pool_status: c.state.pools?.[STANDALONE_POOL] || null,
    cost: { today: cost?.today ?? 0, total: cost?.total ?? 0, minutes: cost?.minutes ?? 0 },
    op: op || null,
    last_op: lastOp || null,
  };
}

/** A cluster-spec issue (normalizeSpec, the live checks) at the launch request's own field, for the launch form. */
export function mapLaunchIssues(issues: { path: (string | number)[]; message: string }[]): { path: (string | number)[]; message: string }[] {
  const TOP: Record<string, string> = { cap_s: "deadline_min", auto_stop_idle_min: "idle_stop_min", regions: "region" };
  const IMG: Record<string, string> = { channel: "channel", sha: "sha", ref: "image" };
  return issues.map((i) => {
    const [a, b, c, ...rest] = i.path;
    let path: (string | number)[];
    if (a === "pools" && b === 0 && typeof c === "string") path = [c === "regions" ? "region" : c === "hub_download_approved" ? "weights_download_approved" : c, ...rest];
    else if (a === "pools") path = [];
    else if (a === "image" && typeof b === "string") path = [IMG[b] || "channel"];
    else if (typeof a === "string") path = [TOP[a] || a, ...(TOP[a] ? [] : i.path.slice(1))];
    else path = i.path;
    const message = a === "cap_s" ? i.message.replace(/\b(\d+)\b/g, (m) => (Number(m) >= 60 ? `${Number(m) / 60} min` : m)) : i.message;
    return { path, message };
  });
}
