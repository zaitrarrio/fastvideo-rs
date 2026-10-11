// GMI Cloud Cluster Engine API (docs/serve/deploy-gmi-brev.md §2): REST at
// console.gmicloud.ai/api/v1, `Authorization: Bearer <API key>`. fv-control
// rents GPU *containers*: a template (= our image) per image reference, then
// POST /v1/containers with the env, command and port. Every call here is from
// the docs' OpenAPI pages; none has been made live (no account yet), so the
// response shapes are read defensively. GMI containers carry no labels: ours
// are the ones whose name fv-control made (fv-pod-* / fv-ctl-*) AND that
// fv-control recorded in D1 (providers.ts decides; this module only filters
// by name). Nothing here returns a container's envs (GET answers them in clear).
import { type Env } from "./env";
import { fetchWithTimeout, HttpError, scrub } from "./util";

/** Name prefixes of the containers fv-control makes (cluster/ops.ts runpodName). */
export const FV_NAME_RE = /^fv-(pod|ctl)-/;
/** A template name for an image (GMI names: ^([A-Za-z0-9][A-Za-z0-9_\-. ]*)?[A-Za-z0-9]$, ≤ 255). */
export const gmiTemplateName = (hash12: string) => `fv-img-${hash12}`;

export type GmiStatus = "unknown" | "creating" | "running" | "terminating" | "stopped" | "error" | "zombie";
export interface GmiContainer {
  id: string;
  name: string;
  status: GmiStatus;
  reason: string | null;
  product: string | null;
  idc: string | null;
  createdAt: string | null;
  publicIp: string | null;
}
export interface GmiProduct {
  name: string;
  idc: string;
  /** $/hr: `price` ÷ GMI_PRICE_DIVISOR (the unit is not documented: UNVERIFIED). */
  usd_per_hr: number | null;
  /** `valid` (undocumented meaning; read as "can be rented now"). */
  valid: boolean;
  gpu: string | null;
}

const base = (env: Env) => (env.GMI_API || "https://console.gmicloud.ai/api").replace(/\/$/, "");
export const gmiEnabled = (env: Env) => !!env.GMI_API_KEY;
const list = (s: string | undefined) =>
  (s || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
/** The product ids pods may use (GMI_PRODUCTS; the owner gets them from GMI sales). */
export const gmiProducts = (env: Env) => list(env.GMI_PRODUCTS);
export const gmiDefaultIdc = (env: Env) => env.GMI_DEFAULT_IDC || "us-denver-1";

async function call(env: Env, method: string, path: string, body?: unknown, opts: { text?: boolean } = {}): Promise<any> {
  if (!env.GMI_API_KEY) throw new HttpError(400, "gmi: GMI_API_KEY is not set (docs/serve/deploy-gmi-brev.md §8)");
  const r = await fetchWithTimeout(`${base(env)}/v1${path}`, {
    method,
    headers: { authorization: `Bearer ${env.GMI_API_KEY}`, "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    timeoutMs: 30000,
  });
  const text = await r.text();
  if (!r.ok) {
    let msg = text.slice(0, 300);
    try {
      const j = JSON.parse(text);
      msg = j?.reason || j?.message || (j?.group ? `${j.group} ${j.code}` : msg);
    } catch {
      /* not JSON */
    }
    throw new HttpError(r.status >= 500 ? 502 : r.status, `gmi ${method} ${path}: ${r.status} ${scrub(env, String(msg))}`, { upstream: r.status });
  }
  if (opts.text) return text;
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    throw new HttpError(502, `gmi ${path}: not JSON`);
  }
}
/** An array answer, or the first array property of an object answer (the list shapes are not all documented). */
const arr = (j: any, key?: string): any[] => (Array.isArray(j) ? j : key && Array.isArray(j?.[key]) ? j[key] : (Object.values(j || {}).find(Array.isArray) as any[]) || []);

export function toContainer(c: any): GmiContainer {
  return {
    id: String(c.id),
    name: String(c.name || c.id),
    status: (String(c.status || "unknown").toLowerCase() as GmiStatus) || "unknown",
    reason: c.reason ? String(c.reason).slice(0, 300) : null,
    product: c.product ?? null,
    idc: c.idc ?? null,
    createdAt: c.createdAt ?? null,
    publicIp: c.publicIP?.ipAddress || c.eipAddress || null,
  };
}

export const gmi = {
  /** Every container of the org (named fields only). */
  async containers(env: Env): Promise<GmiContainer[]> {
    return arr(await call(env, "GET", "/containers"), "containers").map(toContainer);
  },
  async byName(env: Env, name: string): Promise<GmiContainer | null> {
    return (await gmi.containers(env)).find((c) => c.name === name) || null;
  },
  /** The template that runs `image` (created once per image reference, then reused). */
  async ensureTemplate(env: Env, image: string, hash12: string): Promise<string> {
    const name = gmiTemplateName(hash12);
    const have = arr(await call(env, "GET", "/templates"), "templates").find((t: any) => t?.name === name && (t?.path === image || !t?.path));
    if (have?.id) return String(have.id);
    // The image is public (ghcr.io/zaitrarrio/fastvideo-rs-serve, docs/serve/images.md): no registry credential.
    const j = await call(env, "POST", "/templates", { name, path: image, description: `fv-serve ${image}`.slice(0, 4000), status: "published" });
    if (!j?.id) throw new HttpError(502, "gmi POST /templates: no id");
    return String(j.id);
  },
  /** One container; its id. */
  async create(
    env: Env,
    req: { name: string; product: string; idc: string; templateId: string; command: string; args: string[]; envs: Record<string, string>; ports: number[] },
  ): Promise<string> {
    const j = await call(env, "POST", "/containers", {
      name: req.name,
      templateId: req.templateId,
      count: 1,
      product: req.product,
      idc: req.idc,
      command: req.command,
      args: req.args,
      envs: Object.entries(req.envs).map(([name, value]) => ({ name, value })),
      ports: req.ports.map((p) => ({ containerPort: p, protocol: "TCP" })),
    });
    const id = arr(j)[0]?.id ?? j?.id;
    if (!id) throw new HttpError(502, "gmi POST /containers: no id");
    return String(id);
  },
  /** Deletes one container; an unknown one counts as gone. */
  async remove(env: Env, id: string): Promise<boolean> {
    try {
      await call(env, "DELETE", `/containers/${encodeURIComponent(id)}`);
      return true;
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) return true;
      throw e;
    }
  },
  /** The container's log (text/plain), last lines. */
  async logs(env: Env, id: string, max = 500): Promise<string[]> {
    const t = String(await call(env, "GET", `/containers/${encodeURIComponent(id)}/logs`, undefined, { text: true }));
    return t.split("\n").filter(Boolean).slice(-max);
  },
  /** Container products (price, valid) of an IDC, or all. */
  async products(env: Env, idc?: string): Promise<GmiProduct[]> {
    const div = Number(env.GMI_PRICE_DIVISOR || "100") || 100;
    const j = await call(env, "GET", `/containers/products${idc ? `?idc=${encodeURIComponent(idc)}` : ""}`);
    return arr(j, "products")
      .filter((p: any) => !p.type || p.type === "Container")
      .map((p: any) => ({ name: String(p.name), idc: String(p.idc || idc || ""), usd_per_hr: typeof p.price === "number" ? Math.round((p.price / div) * 10000) / 10000 : null, valid: p.valid !== false, gpu: p.gpuModel ?? null }));
  },
};
