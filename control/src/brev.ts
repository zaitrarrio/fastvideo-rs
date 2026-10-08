// NVIDIA Brev (docs/serve/deploy-gmi-brev.md §3). Brev documents only its CLI;
// fv-control runs in a Worker, so it calls the REST API the open-source CLI
// (brevdev/brev-cli pkg/store/workspace.go) uses: workspaces under an org,
// `Authorization: Bearer <BREV_API_TOKEN>`. That API is not a documented
// contract and has not been called live: every shape here is UNVERIFIED and
// read defensively. fv-control rents a VM (vmOnlyMode) whose startup script
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
  if (!r.ok) throw new HttpError(r.status >= 500 ? 502 : r.status, `brev ${method} ${path.replace(/organizations\/[^/]+/, "organizations/<org>")}: ${r.status} ${scrub(env, text.slice(0, 300))}`, { upstream: r.status });
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    throw new HttpError(502, `brev ${path}: not JSON`);
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
const arr = (j: any): any[] => (Array.isArray(j) ? j : Array.isArray(j?.workspaces) ? j.workspaces : []);

export const brev = {
  async workspaces(env: Env): Promise<BrevWorkspace[]> {
    return arr(await call(env, "GET", `api/organizations/${encodeURIComponent(env.BREV_ORG_ID!)}/workspaces`)).map(toWorkspace);
  },
  async byName(env: Env, name: string): Promise<BrevWorkspace | null> {
    return (await brev.workspaces(env)).find((w) => w.name === name && w.status !== "DELETING") || null;
  },
  /** A VM (vmOnlyMode) of an instance type running `startupScript` on boot; its id. */
  async create(env: Env, req: { name: string; instanceType: string; startupScript: string }): Promise<string> {
    const j = await call(env, "POST", `api/organizations/${encodeURIComponent(env.BREV_ORG_ID!)}/workspaces`, {
      name: req.name,
      instanceType: req.instanceType,
      vmOnlyMode: true,
      launchJupyterOnStart: false,
      startupScript: req.startupScript,
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
