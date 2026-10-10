// NVIDIA Brev (docs/serve/deploy-gmi-brev.md §3). Brev documents only its CLI;
// fv-control runs in a Worker, so it calls the REST API the open-source CLI
// (brevdev/brev-cli pkg/store/workspace.go) uses: workspaces under an org,
// `Authorization: Bearer <BREV_API_TOKEN>`. That API is not a documented
// contract: the instance-type list and the (empty) workspace list were read
// live once (2026-10-10, GET only, bearer accepted); a live create answered
// 400 "Legacy workspace version unsupported" to the old body, so create sends
// brev-cli main's v1 body; stop / start / delete have not been called live,
// so their shapes are UNVERIFIED and read defensively. fv-control rents a VM
// (vmBuild, no container) whose startup script
// installs a per-boot bootstrap (brev-park.ts) that runs our image with Docker
// and the NVIDIA toolkit (both preinstalled, per the docs). Brev has no
// labels: ours are fv-pod-* / fv-ctl-* names that fv-control recorded in D1
// (providers.ts, brev_instances). Prices, the stoppable flag and storage
// prices come from the instance-type list (GET
// api/instances/alltypesavailable/{org}, cached 1 h); BREV_PRICES (JSON
// {"<instanceType>": usd_per_hr}) overrides a price. Nothing here returns a
// workspace's startup script.
import { type Env } from "./env";
import { fetchWithTimeout, HttpError, parseJson, scrub } from "./util";

/** brev-cli entity.go workspace statuses. */
export type BrevStatus = "RUNNING" | "STARTING" | "STOPPING" | "DEPLOYING" | "STOPPED" | "DELETING" | "FAILURE" | string;
export interface BrevWorkspace {
  id: string;
  name: string;
  status: BrevStatus;
  health: string | null;
  instanceType: string | null;
  dns: string | null;
  createdAt: string | null;
}

const base = (env: Env) => (env.BREV_API_URL || "https://brevapi.us-west-2-prod.control-plane.brev.dev").replace(/\/$/, "");
export const brevEnabled = (env: Env) => !!env.BREV_API_TOKEN && !!env.BREV_ORG_ID;
/** Why brev is off (null: on). */
export const brevOff = (env: Env): string | null => (!env.BREV_API_TOKEN ? "BREV_API_TOKEN is not set" : !env.BREV_ORG_ID ? "BREV_ORG_ID is not set" : null);
/** The instance types pods may use (BREV_INSTANCE_TYPES, comma list): the allow-list. */
export const brevInstanceTypes = (env: Env) =>
  (env.BREV_INSTANCE_TYPES || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
/** $/hr of an instance type from the owner's BREV_PRICES override; null when the table has no entry (then: the live list). */
export function brevPrice(env: Env, type: string): number | null {
  const t = parseJson<Record<string, unknown>>(env.BREV_PRICES, {});
  const v = Number(t[type]);
  return Number.isFinite(v) && v > 0 ? v : null;
}

async function call(env: Env, method: string, path: string, body?: unknown): Promise<any> {
  const off = brevOff(env);
  if (off) throw new HttpError(400, `brev: ${off} (docs/serve/deploy-gmi-brev.md §8)`);
  const r = await fetchWithTimeout(`${base(env)}/${path}`, {
    method,
    headers: { authorization: `Bearer ${env.BREV_API_TOKEN}`, "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    timeoutMs: 30000,
  });
  const text = await r.text();
  const shown = path.replace(/organizations\/[^/]+/, "organizations/<org>").replace(/alltypesavailable\/[^/]+/, "alltypesavailable/<org>");
  if (!r.ok) throw new HttpError(r.status >= 500 ? 502 : r.status, `brev ${method} ${shown}: ${r.status} ${scrub(env, text.slice(0, 300))}`, { upstream: r.status });
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    throw new HttpError(502, `brev ${shown}: not JSON`);
  }
}

export function toWorkspace(w: any): BrevWorkspace {
  return {
    id: String(w.id),
    name: String(w.name || w.id),
    status: String(w.status || "").toUpperCase(),
    health: w.healthStatus ?? null,
    instanceType: w.instanceType ?? null,
    dns: w.dns ?? null,
    createdAt: w.createdAt ?? w.created_at ?? null,
  };
}
/** brev-cli store/workspace.go: UserWorkspaceTemplateID, UserWorkspaceClassID, DefaultDiskStorage (non-admin users). */
export const BREV_USER_TEMPLATE_ID = "4nbb4lg2s";
export const BREV_USER_CLASS_ID = "2x8";
export const BREV_DEFAULT_DISK = "120Gi";
const arr = (j: any): any[] => (Array.isArray(j) ? j : Array.isArray(j?.workspaces) ? j.workspaces : []);

export const brev = {
  async workspaces(env: Env): Promise<BrevWorkspace[]> {
    return arr(await call(env, "GET", `api/organizations/${encodeURIComponent(env.BREV_ORG_ID!)}/workspaces`)).map(toWorkspace);
  },
  async byName(env: Env, name: string): Promise<BrevWorkspace | null> {
    return (await brev.workspaces(env)).find((w) => w.name === name && w.status !== "DELETING") || null;
  },
  /** The cloud credential an instance type is created under (its listing row's `cloud_cred_id`); the CLI refuses to create without it. */
  async cloudCredId(env: Env, instanceType: string): Promise<string> {
    const t = (await brev.types(env)).find((x) => x.type === instanceType);
    if (!t?.cloud_cred_id) throw new HttpError(400, `brev: instance type ${instanceType} is not in the org's instance-type listing (invalid or unavailable): no cloud credential to create it under`);
    return t.cloud_cred_id;
  },
  /**
   * A VM of an instance type running `startupScript` on boot; its id. The body is what brev-cli main sends
   * (pkg/store/workspace.go NewCreateWorkspacesOptions, pkg/cmd/gpucreate createWorkspace / applyBuildMode "vm" /
   * resolveWorkspaceUserOptions): workspaceVersion v1 (the old body got 400 "Legacy workspace version unsupported"
   * live), the user template and class, the type's cloudCredId, and the script in vmBuild.lifeCycleScriptAttr.
   * `diskStorage` ("500Gi"; the CLI's default 120Gi) sizes the disk (UNVERIFIED for fixed-disk types).
   */
  async create(env: Env, req: { name: string; instanceType: string; startupScript: string; diskStorage?: string }): Promise<string> {
    const cloudCredId = await brev.cloudCredId(env, req.instanceType);
    const j = await call(env, "POST", `api/organizations/${encodeURIComponent(env.BREV_ORG_ID!)}/workspaces`, {
      name: req.name,
      workspaceVersion: "v1",
      workspaceTemplateId: BREV_USER_TEMPLATE_ID,
      workspaceClassId: BREV_USER_CLASS_ID,
      cloudCredId,
      instanceType: req.instanceType,
      diskStorage: req.diskStorage || BREV_DEFAULT_DISK,
      isStoppable: false,
      vmBuild: { forceJupyterInstall: false, lifeCycleScriptAttr: { script: req.startupScript } },
      portMappings: {},
      execsV1: {},
      reposV1: {},
      labels: null,
      files: null,
      launchJupyterOnStart: false,
    });
    const id = j?.id ?? j?.workspace?.id;
    if (!id) throw new HttpError(502, "brev create: no id");
    return String(id);
  },
  /** Stops one workspace (`PUT api/workspaces/{id}/stop`): the VM goes, /home/ubuntu/workspace stays (billed as storage). */
  async stop(env: Env, id: string): Promise<void> {
    await call(env, "PUT", `api/workspaces/${encodeURIComponent(id)}/stop`);
  },
  /** Starts a stopped workspace (`PUT …/start`); per the docs it fails when its provider / region has no capacity. */
  async start(env: Env, id: string): Promise<void> {
    await call(env, "PUT", `api/workspaces/${encodeURIComponent(id)}/start`);
  },
  /** Deletes one workspace (instance and disk); an unknown one counts as gone. */
  async remove(env: Env, id: string): Promise<boolean> {
    try {
      await call(env, "DELETE", `api/workspaces/${encodeURIComponent(id)}`);
      return true;
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) return true;
      throw e;
    }
  },
  /** Every instance type the org may rent: price, stoppable flag, storage price, deploy time (cached 1 h per isolate). */
  async types(env: Env): Promise<BrevType[]> {
    const k = `${base(env)}|${env.BREV_ORG_ID}`;
    if (typesCache && typesCache.k === k && Date.now() - typesCache.at < TYPES_TTL_MS) return typesCache.v;
    const j = await call(env, "GET", `api/instances/alltypesavailable/${encodeURIComponent(env.BREV_ORG_ID!)}`);
    const v = (Array.isArray(j?.allInstanceTypes) ? j.allInstanceTypes : Array.isArray(j) ? j : []).map(toType).filter((t: BrevType) => t.type);
    typesCache = { k, at: Date.now(), v };
    return v;
  },
  /** One instance type of the list; null when it is not there or the list is unreachable. */
  async type(env: Env, type: string): Promise<BrevType | null> {
    return (await brev.types(env).catch(() => [] as BrevType[])).find((t) => t.type === type) || null;
  },
};

// ---------------------------------------------------------------- instance types (live prices)
export interface BrevType {
  type: string;
  gpu: string | null;
  gpu_count: number;
  /** Brev's `stoppable` (true); null / false: no stop is offered (e.g. the Shadeform types), the instance is deleted. */
  stoppable: boolean;
  /** `base_price` ($/hr at its default location). */
  usd_per_hr: number | null;
  /** The most any of its storage classes costs per GB-hour (which class Brev uses is not in the list: err high). */
  storage_usd_per_gb_hr: number | null;
  /** The disk is sized at create (`elastic_root_volume`); else it is `fixed_disk_gb`. */
  elastic_disk: boolean;
  fixed_disk_gb: number | null;
  location: string | null;
  provider: string | null;
  /** `estimated_deploy_time` ("7m0s") in seconds. */
  deploy_s: number | null;
  available: boolean | null;
  /** The cloud credential creates of this type go under (`cloud_cred_id`; required by create). */
  cloud_cred_id: string | null;
}
const TYPES_TTL_MS = 3600_000;
let typesCache: { k: string; at: number; v: BrevType[] } | null = null;
/** Tests: forget the cached instance-type list. */
export const resetBrevTypes = () => {
  typesCache = null;
};
const amount = (x: any): number | null => {
  const v = Number(x?.amount ?? x);
  return x != null && Number.isFinite(v) && v > 0 ? v : null;
};
/** "1TiB226GiB" / "850GiB" / "0B" / "128GiB" -> GiB (null for 0 or unparsable). */
export function parseGiB(s: unknown): number | null {
  if (typeof s !== "string") return null;
  let gib = 0;
  for (const m of s.matchAll(/(\d+(?:\.\d+)?)\s*(TiB|GiB|MiB|TB|GB|B)/g)) {
    const v = Number(m[1]);
    gib += m[2] === "TiB" || m[2] === "TB" ? v * 1024 : m[2] === "GiB" || m[2] === "GB" ? v : m[2] === "MiB" ? v / 1024 : 0;
  }
  return gib > 0 ? Math.round(gib) : null;
}
/** "7m0s" / "6m30s" / "45s" / "1h2m" -> seconds. */
export function parseDuration(s: unknown): number | null {
  if (typeof s !== "string" || !s) return null;
  let t = 0;
  let any = false;
  for (const m of s.matchAll(/(\d+(?:\.\d+)?)(h|m|s)/g)) {
    any = true;
    t += Number(m[1]) * (m[2] === "h" ? 3600 : m[2] === "m" ? 60 : 1);
  }
  return any ? Math.round(t) : null;
}
export function toType(x: any): BrevType {
  const st: any[] = Array.isArray(x?.supported_storage) ? x.supported_storage : [];
  const prices = st.map((s) => amount(s?.price_per_gb_hr)).filter((v): v is number => v !== null);
  const g = Array.isArray(x?.supported_gpus) ? x.supported_gpus[0] : null;
  const fixed = st.map((s) => parseGiB(s?.size)).filter((v): v is number => v !== null);
  return {
    type: String(x?.type || ""),
    gpu: g?.name ? String(g.name) : null,
    gpu_count: Number(g?.count) || 0,
    stoppable: x?.stoppable === true,
    usd_per_hr: amount(x?.base_price),
    storage_usd_per_gb_hr: prices.length ? Math.max(...prices) : null,
    elastic_disk: x?.elastic_root_volume === true,
    fixed_disk_gb: fixed.length ? Math.max(...fixed) : null,
    location: x?.location ? String(x.location) : null,
    provider: x?.provider ? String(x.provider) : null,
    deploy_s: parseDuration(x?.estimated_deploy_time),
    available: typeof x?.is_available === "boolean" ? x.is_available : null,
    cloud_cred_id: x?.cloud_cred_id ? String(x.cloud_cred_id) : x?.cloud_cred?.cloud_cred_id ? String(x.cloud_cred.cloud_cred_id) : null,
  };
}
/** $/hr of an instance type: the BREV_PRICES override, else the live list; null when neither knows it. */
export async function brevPriceLive(env: Env, type: string): Promise<{ usd_per_hr: number | null; info: BrevType | null }> {
  const info = await brev.type(env, type);
  return { usd_per_hr: brevPrice(env, type) ?? info?.usd_per_hr ?? null, info };
}
