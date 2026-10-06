// CloudRift API (docs/ops/cloudrift.md): rift-server's public REST API at
// api.cloudrift.ai. Every call is a POST of {version, data} to /api/v1/<path>
// and answers {version, data}; the key goes in X-API-Key and nowhere else.
// Money is in cents everywhere (observed live on 2026-10-06): catalog prices,
// resource_info.cost_per_hour and account/info's balance (2000 for a $20
// top-up, although the spec says "Balance in USD"). Only named fields of an instance are
// picked: a listing can carry credentials when asked (mask.with_credentials,
// never set here) and the rental's env is not part of the answer.
import { type Env } from "./env";
import { fetchWithTimeout, HttpError, scrub } from "./util";

/** v0.62.0 (2026-09-09): instances/rent accepts the v062 protocol only; list and terminate accept it too. */
export const CLOUDRIFT_API_VERSION = "2026-09-08";
/** Owner rule (2026-10-06): on CloudRift only RTX PRO 6000 and RTX 5090, i.e. the
 * instance types rtxpro6000-* and rtx59-* (scripts/gpu/cloudrift-lib.sh has the same list). */
export const CLOUDRIFT_ALLOWED_BRANDS = ["RTX PRO 6000", "RTX 5090"];
const ALLOWED_TYPE_RE = /^(rtxpro6000|rtx59)-/;
export const cloudriftTypeAllowed = (instanceType: string | null | undefined) => !!instanceType && ALLOWED_TYPE_RE.test(instanceType);
/** Every rental fv-control or the repo's scripts make carries this tag (CLAUDE.md: only touch what you created). */
export const CLOUDRIFT_OWNER_TAG = "fv-owner:fastvideo-rs";

export type CloudriftStatus = "Initializing" | "Active" | "Deactivating" | "Inactive" | "Failed";
export interface CloudriftInstance {
  id: string;
  name: string;
  status: CloudriftStatus;
  tags: string[];
  host: string | null;
  instanceType: string | null;
  /** $/hr (resource_info.cost_per_hour is cents: 25.0 for a $0.25/hr rental, live 2026-10-06). */
  costPerHr: number;
  gpuCount: number;
  gpu: string | null;
  createdAt: string | null;
  failure: string | null;
  /** fv-deadline:<unix s> tag, ms; null when absent. */
  deadlineMs: number | null;
  ours: boolean;
}

const base = (env: Env) => (env.CLOUDRIFT_API || "https://api.cloudrift.ai").replace(/\/$/, "");
export const cloudriftEnabled = (env: Env) => !!env.CLOUDRIFT_API_KEY;

async function call(env: Env, path: string, data: unknown, opts: { public?: boolean } = {}): Promise<any> {
  if (!opts.public && !env.CLOUDRIFT_API_KEY) throw new HttpError(400, "cloudrift: no CLOUDRIFT_API_KEY configured");
  const r = await fetchWithTimeout(`${base(env)}/api/v1/${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(opts.public ? {} : { "x-api-key": env.CLOUDRIFT_API_KEY! }) },
    body: JSON.stringify({ version: env.CLOUDRIFT_API_VERSION || CLOUDRIFT_API_VERSION, data }),
    timeoutMs: 30000,
  });
  const text = await r.text();
  if (!r.ok) throw new HttpError(r.status >= 500 ? 502 : r.status, `cloudrift ${path}: ${scrub(env, text.slice(0, 300))}`, { upstream: r.status });
  let j: any;
  try {
    j = text ? JSON.parse(text) : {};
  } catch {
    throw new HttpError(502, `cloudrift ${path}: not JSON`);
  }
  return j?.data ?? j;
}

/** The deadline tag (fv-deadline:<unix s>) in ms. */
export function deadlineOf(tags: string[]): number | null {
  for (const t of tags) {
    const m = /^fv-deadline:(\d{9,11})$/.exec(t);
    if (m) return Number(m[1]) * 1000;
  }
  return null;
}

export function toInstance(i: any, costUnit: "usd" | "cents" = "cents"): CloudriftInstance {
  const tags: string[] = Array.isArray(i?.tags) ? i.tags.map(String) : [];
  const raw = Number(i?.resource_info?.cost_per_hour ?? 0);
  return {
    id: String(i.id),
    name: String(i.instance_name || i.id),
    status: i.status as CloudriftStatus,
    tags,
    host: i.host_address ?? null,
    instanceType: i.resource_info?.instance_type ?? null,
    costPerHr: costUnit === "cents" ? raw / 100 : raw,
    gpuCount: Array.isArray(i.gpus) ? i.gpus.length : Number(i.gpu_limit ?? 0),
    gpu: Array.isArray(i.gpus) && i.gpus[0] ? i.gpus[0].brand_short || i.gpus[0].brand || null : null,
    createdAt: i.created_at ?? null,
    failure: i.failure?.user_message ?? null,
    deadlineMs: deadlineOf(tags),
    ours: tags.includes(CLOUDRIFT_OWNER_TAG),
  };
}

export const cloudrift = {
  /** The balance in $ (account/info answers cents). */
  async balance(env: Env): Promise<number> {
    const d = await call(env, "account/info", {});
    const b = Number(d?.balance);
    if (d?.balance == null || !Number.isFinite(b)) throw new HttpError(502, "cloudrift account/info: no balance");
    return b / 100;
  },
  /** Live rentals of the account (Initializing, Active, Deactivating, Failed). */
  async instances(env: Env): Promise<CloudriftInstance[]> {
    const d = await call(env, "instances/list", {
      selector: { ByStatus: { statuses: ["Initializing", "Active", "Deactivating", "Failed"] } },
      mask: { with_connection_info: true, with_usage_info: true, with_hardware_info: true },
    });
    const unit = env.CLOUDRIFT_COST_UNIT === "usd" ? "usd" : "cents";
    return (Array.isArray(d?.instances) ? d.instances : []).map((i: any) => toInstance(i, unit));
  },
  /** Mean GPU utilisation per instance (instances/metrics), percent. */
  async gpuUtil(env: Env, ids: string[]): Promise<Map<string, number>> {
    const out = new Map<string, number>();
    if (!ids.length) return out;
    const d = await call(env, "instances/metrics", { selector: { ById: ids } });
    for (const m of Array.isArray(d?.metrics) ? d.metrics : []) {
      const us = (m.gpus || []).map((g: any) => g.gpu_utilization_percent).filter((x: any) => typeof x === "number");
      if (us.length) out.set(String(m.instance_id), us.reduce((a: number, b: number) => a + b, 0) / us.length);
    }
    return out;
  },
  /** Terminates one rental; an unknown or finished one counts as done. */
  async terminate(env: Env, id: string): Promise<boolean> {
    try {
      await call(env, "instances/terminate", { selector: { ById: [id] } });
      return true;
    } catch (e) {
      if (e instanceof HttpError && (e.status === 404 || e.status === 400)) return true;
      throw e;
    }
  },
  /** 1-GPU on-demand $/hr of an allowed instance type (catalog, public) and its free nodes. */
  async price(env: Env, brandOrVariant: string): Promise<{ variant: string; usd_per_hr: number; free_nodes: number; datacenters: string[] }[]> {
    if (!CLOUDRIFT_ALLOWED_BRANDS.includes(brandOrVariant) && !cloudriftTypeAllowed(brandOrVariant))
      throw new HttpError(400, `CloudRift GPU '${brandOrVariant}' is not allowed: only ${CLOUDRIFT_ALLOWED_BRANDS.join(", ")} (instance types rtxpro6000-*, rtx59-*)`);
    const d = await call(env, "instance-types/list", { selector: { ByServiceAndLocation: { services: ["docker"] } } }, { public: true });
    const out: { variant: string; usd_per_hr: number; free_nodes: number; datacenters: string[] }[] = [];
    for (const t of d?.instance_types || [])
      for (const v of t.variants || []) {
        if ((v.gpu_count ?? 0) !== 1 || !cloudriftTypeAllowed(v.name)) continue;
        if (t.brand_short !== brandOrVariant && v.name !== brandOrVariant && t.name !== brandOrVariant) continue;
        out.push({
          variant: v.name,
          usd_per_hr: Math.round(Number(v.cost_per_hour) * 100) / 10000,
          free_nodes: Number(v.available_nodes || 0),
          datacenters: Object.entries(v.available_nodes_per_dc || {}).filter(([, n]) => Number(n) > 0).map(([k]) => k),
        });
      }
    return out.sort((a, b) => a.usd_per_hr - b.usd_per_hr);
  },
};
