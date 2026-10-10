// NVIDIA Brev (docs/serve/deploy-gmi-brev.md §3). Brev documents only its CLI;
// fv-control runs in a Worker, so it calls the REST API the open-source CLI
// (brevdev/brev-cli pkg/store/workspace.go) uses: workspaces under an org,
// `Authorization: Bearer <BREV_API_TOKEN>`. That API is not a documented
// contract: the bearer key, the workspace list and the instance-type listing
// were read live (GET only, 2026-10-10), and a live create answered 400
// "Legacy workspace version unsupported" to the old body, so create now sends
// what brev-cli main sends (see `create`); the rest is read defensively. fv-control rents a VM (vmOnlyMode) whose startup script
// runs our image with Docker and the NVIDIA toolkit (both preinstalled, per
// the docs). Brev has no labels: ours are fv-pod-* / fv-ctl-* names that
// fv-control recorded in D1 (providers.ts). No price API is documented
// either: BREV_PRICES (JSON {"<instanceType>": usd_per_hr}) is the planner's
// table. Nothing here returns a workspace's startup script (it holds the env).
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
/** The instance types pods may use (BREV_INSTANCE_TYPES, comma list). */
export const brevInstanceTypes = (env: Env) =>
  (env.BREV_INSTANCE_TYPES || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
/** $/hr per instance type (BREV_PRICES); null when the table has no entry. */
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
const TYPES_TTL_MS = 3600_000;
let typesCache: { k: string; at: number; v: any[] } | null = null;
/** Tests: forget the cached instance-type listing. */
export const resetBrevTypes = () => {
  typesCache = null;
};
const arr = (j: any): any[] => (Array.isArray(j) ? j : Array.isArray(j?.workspaces) ? j.workspaces : []);

export const brev = {
  async workspaces(env: Env): Promise<BrevWorkspace[]> {
    return arr(await call(env, "GET", `api/organizations/${encodeURIComponent(env.BREV_ORG_ID!)}/workspaces`)).map(toWorkspace);
  },
  async byName(env: Env, name: string): Promise<BrevWorkspace | null> {
    return (await brev.workspaces(env)).find((w) => w.name === name && w.status !== "DELETING") || null;
  },
  /** The instance-type listing (GET api/instances/alltypesavailable/{org}; read live 2026-10-10), cached 1 h per isolate. */
  async types(env: Env): Promise<any[]> {
    const k = `${base(env)}|${env.BREV_ORG_ID}`;
    if (typesCache && typesCache.k === k && Date.now() - typesCache.at < TYPES_TTL_MS) return typesCache.v;
    const j = await call(env, "GET", `api/instances/alltypesavailable/${encodeURIComponent(env.BREV_ORG_ID!)}`);
    const v: any[] = Array.isArray(j?.allInstanceTypes) ? j.allInstanceTypes : Array.isArray(j) ? j : [];
    typesCache = { k, at: Date.now(), v };
    return v;
  },
  /** The cloud credential an instance type is created under (its listing row's `cloud_cred_id`); the CLI refuses to create without it. */
  async cloudCredId(env: Env, instanceType: string): Promise<string> {
    const row = (await brev.types(env)).find((t) => t?.type === instanceType);
    const id = row?.cloud_cred_id ?? row?.cloud_cred?.cloud_cred_id;
    if (!id) throw new HttpError(400, `brev: instance type ${instanceType} is not in the org's instance-type listing (invalid or unavailable): no cloud credential to create it under`);
    return String(id);
  },
  /**
   * A VM of an instance type running `startupScript` on boot; its id. The body is what brev-cli main sends
   * (pkg/store/workspace.go NewCreateWorkspacesOptions, pkg/cmd/gpucreate createWorkspace / applyBuildMode "vm" /
   * resolveWorkspaceUserOptions): workspaceVersion v1 (the old body gets 400 "Legacy workspace version unsupported"),
   * the user template and class, the type's cloudCredId, and the script in vmBuild.lifeCycleScriptAttr (no
   * startupScript / vmOnlyMode true).
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
};
