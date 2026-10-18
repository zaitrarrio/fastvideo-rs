// One-click deploy (owner 2026-10-18; the console's Deploy page, #/deploy): several model families (or all of them),
// each with its pools (a preset of the family, or a template of it: h3, fasth3, streaming, …) and its instance types
// (the provider, ordered Runpod GPU types and data centres, or a GMI / Brev product), into a new cluster (one pool per
// family, or the template's pools) or added to an existing one. The plan goes through the same checks as the cluster
// configuration page (checkSpec: the schema, normalizeSpec, the live checks; planSpec for an existing cluster) and
// is shown before anything is saved; apply defines (and starts) a new cluster, or saves the existing one's spec
// (saveDoc, its version checked) and starts it when it is stopped. docs/control/README.md §4b.
import { z } from "zod";
import CATALOG from "./cluster/catalog.json";
import { checkSpec, type SpecCheck } from "./cluster/editor";
import { projectSpend } from "./cluster/ops";
import { defaultSpec, normalizeSpec, POOL_PRESETS, presetPool, STANDARD_POOLS, TEMPLATES, type ClusterSpec, type PoolSpec } from "./cluster/spec";
import { allPods, getCluster, isStandalone, type Cluster } from "./cluster/store";
import { docVersion, planSpec } from "./docs";
import { PROVIDERS, WEIGHTS_SOURCES } from "./enums";
import type { Env } from "./env";
import type { Issue } from "./schemas";
import { HttpError } from "./util";
import { zIssues } from "./zissues";

const FAMILY_IDS = CATALOG.families.map((f) => f.id) as [string, ...string[]];
/** The fields a family entry sets on each of its pools (an issue at one of them is shown at the family). */
const PLACEMENT = ["count", "provider", "gpu_types", "data_centers", "weights_source", "provider_gpu", "provider_region", "container_disk_gb", "hub_download_approved"] as const;

export const DeployFamilyZ = z
  .object({
    family: z.enum(FAMILY_IDS).describe("A model family of the serve catalog (h3, ltx2, wan)."),
    pools: z
      .string()
      .regex(/^(template:)?[a-z0-9][a-z0-9-]{0,40}$/, { message: "a preset id of the family, or template:<id>" })
      .optional()
      .describe("What to deploy for the family: one pool preset (default: the family's standard pool), or template:<id> (every pool of a template of the family: h3, fasth3, streaming, …)."),
    count: z.number().int().min(0).max(8).optional().describe("Workers per pool (default: the preset's, 1)."),
    provider: z.enum(PROVIDERS).optional().describe("runpod (default), gmi or brev."),
    gpu_types: z.array(z.string().min(1).max(100)).min(1).max(12).optional().describe("Runpod: GPU types in fallback order."),
    data_centers: z.array(z.string().min(1).max(40)).min(1).max(12).optional().describe("Runpod without the volume: data centres in order."),
    weights_source: z.enum(WEIGHTS_SOURCES).optional(),
    hub_download_approved: z.boolean().optional(),
    provider_gpu: z.string().min(1).max(100).optional().describe("GMI / Brev: the product / instance type."),
    provider_region: z.string().min(1).max(60).optional(),
    container_disk_gb: z.number().int().min(5).max(16384).optional().describe("Runpod: the container disk; Brev: the VM's disk."),
  })
  .strict();
export const DeployZ = z
  .object({
    cluster_id: z.string().min(1).max(80).optional().describe("Add the pools to this cluster (its id or name)."),
    name: z.string().min(1).max(40).optional().describe("A new cluster's name."),
    control_plane: z.enum(["edge", "direct"]).optional().describe("A new cluster's control plane (default edge)."),
    all_families: z.boolean().optional().describe("Every family of the catalog: the ones not listed get their standard pool."),
    families: z.array(DeployFamilyZ).max(8).default([]),
    start: z.boolean().optional().describe("Apply: start a new (or stopped) cluster (default true)."),
    version: z.number().int().optional().describe("Apply to an existing cluster: the spec version the plan was made from."),
  })
  .strict()
  .superRefine((x, ctx) => {
    if (!x.cluster_id === !x.name) ctx.addIssue({ code: "custom", path: [x.cluster_id ? "name" : "cluster_id"], message: "an existing cluster (cluster_id) or a new cluster's name, one of them" });
    if (!x.all_families && !x.families.length) ctx.addIssue({ code: "custom", path: ["families"], message: "pick a model family (or all families)" });
    const seen = new Set<string>();
    x.families.forEach((f, i) => {
      if (seen.has(f.family)) ctx.addIssue({ code: "custom", path: ["families", i, "family"], message: `${f.family} twice` });
      seen.add(f.family);
    });
  });
export type DeployRequest = z.infer<typeof DeployZ>;

// ---------------------------------------------------------------- the catalog the page offers
const familyOfPreset = (id: string): string | null => {
  const ms = POOL_PRESETS.find((p) => p.id === id)?.pool.models || [];
  const fams = [...new Set(ms.map((m) => m.family))];
  return fams.length === 1 ? fams[0]! : null;
};
/** The family's standard pool (STANDARD_POOLS: h3-turbo, ltx, wan), else its first preset. */
export function defaultPreset(family: string): string {
  const std = STANDARD_POOLS.find((p) => (p.models || []).every((m) => m.family === family) && (p.models || []).length);
  return std?.id ?? POOL_PRESETS.find((p) => familyOfPreset(p.id) === family)!.id;
}
/** Every family with its presets and the templates whose pools are all of it. */
export function deployFamilies() {
  return CATALOG.families.map((f) => ({
    id: f.id,
    detail: f.detail,
    default: defaultPreset(f.id),
    presets: POOL_PRESETS.filter((p) => familyOfPreset(p.id) === f.id).map((p) => ({ id: p.id, title: p.title, min_vram_gb: p.min_vram_gb, licence: p.licence ?? null, models: (p.pool.models || []).map((m) => m.id), ...(p.members ? { members: p.members, left_out: p.left_out ?? [] } : {}) })),
    // A family template is one pool (one pod) serving every variant (owner 2026-10-18): its models and what it leaves out.
    templates: Object.entries(TEMPLATES)
      .filter(([, t]) => t.pools.length > 0 && t.pools.every((p) => familyOfPreset(p) === f.id))
      .map(([id, t]) => ({ id, title: t.title, pools: t.pools, models: t.pools.flatMap((p) => (presetPool(p)?.models || []).map((m) => m.id)), left_out: t.left_out ?? [] })),
  }));
}

// ---------------------------------------------------------------- the plan
export interface DeployPoolRow {
  id: string;
  family: string;
  preset: string;
  title: string;
  count: number;
  provider: string;
  where: string;
  models: string[];
}
export interface DeployPlan extends SpecCheck {
  target: { kind: "new" | "existing"; id: string | null; name: string; status: string | null; running: boolean; version: number | null };
  pools: DeployPoolRow[];
  /** The whole spec after the deploy (an existing cluster: its pools plus the new ones). */
  spec: unknown;
  /** New cluster: the price projection; existing: planSpec's actions and projection. */
  projection: unknown;
  actions: { action: string; target: string; detail: string }[];
}

/** A family's pool ids: its preset, or every pool of its template; refused when not of the family. */
function poolIdsOf(f: DeployRequest["families"][number]): { ids: string[]; issue: string | null } {
  const v = f.pools || defaultPreset(f.family);
  if (v.startsWith("template:")) {
    const t = v.slice("template:".length);
    const tpl = TEMPLATES[t];
    if (!tpl || !tpl.pools.length) return { ids: [], issue: `no template ${t} (${Object.keys(TEMPLATES).filter((k) => TEMPLATES[k]!.pools.length).join(", ")})` };
    const off = tpl.pools.filter((p) => familyOfPreset(p) !== f.family);
    if (off.length) return { ids: [], issue: `template ${t} has pools of another family (${off.join(", ")})` };
    return { ids: [...tpl.pools], issue: null };
  }
  if (!presetPool(v)) return { ids: [], issue: `no preset ${v}` };
  if (familyOfPreset(v) !== f.family) return { ids: [], issue: `${v} is not a ${f.family} preset` };
  return { ids: [v], issue: null };
}
/** A preset's pool placed as the family entry says (Runpod GPU types / data centres / weights, or GMI / Brev). */
function placePool(id: string, f: DeployRequest["families"][number], taken: Set<string>): PoolSpec {
  const pool = presetPool(id)!;
  let pid = pool.id;
  for (let n = 2; taken.has(pid); n++) pid = `${pool.id}-${n}`;
  taken.add(pid);
  pool.id = pid;
  if (f.count !== undefined) pool.count = f.count;
  if (f.provider && f.provider !== "runpod") {
    pool.provider = f.provider;
    if (f.provider_gpu) pool.provider_gpu = f.provider_gpu;
    if (f.provider_region) pool.provider_region = f.provider_region;
    pool.compute = "GPU";
    pool.volume = false;
    delete pool.gpu_types, delete pool.cpu_flavors, delete pool.vcpu, delete pool.data_centers, delete pool.regions;
  } else {
    if (f.gpu_types) pool.gpu_types = [...f.gpu_types] as PoolSpec["gpu_types"];
    if (f.data_centers) pool.data_centers = [...f.data_centers] as PoolSpec["data_centers"];
    // r2 / hub / none: no volume (any data centre), as on the standalone launch form.
    if (f.weights_source && f.weights_source !== "volume") pool.volume = false;
  }
  if (f.weights_source) pool.weights_source = f.weights_source;
  if (f.hub_download_approved) pool.hub_download_approved = true;
  if (f.container_disk_gb !== undefined) pool.container_disk_gb = f.container_disk_gb;
  return pool;
}
const whereOf = (p: PoolSpec) =>
  p.provider && p.provider !== "runpod"
    ? `${p.provider}: ${p.provider_gpu ?? "?"}${p.provider_region ? ` in ${p.provider_region}` : ""}${p.container_disk_gb ? ` · disk ${p.container_disk_gb} GB` : ""}`
    : `runpod: ${p.gpu_types?.join(" → ") ?? "the region's GPU types"}${p.data_centers ? ` in ${p.data_centers.join(" → ")}` : ""}${p.weights_source ? ` · weights ${p.weights_source}` : ""}`;

/** The deploy's plan: the pools it adds, every problem at its family's field, the price or the cluster's plan. */
export async function planDeploy(env: Env, raw: unknown): Promise<DeployPlan & { cluster: Cluster | null; merged: unknown }> {
  const parsed = DeployZ.safeParse(raw);
  if (!parsed.success) throw new HttpError(400, zIssues(parsed.error).map((i) => `${i.path.join(".") || "deploy"}: ${i.message}`).join("; "), { issues: zIssues(parsed.error) });
  const req = parsed.data;
  const cl = req.cluster_id ? await getCluster(env, req.cluster_id) : null;
  if (cl && isStandalone(cl)) throw new HttpError(400, `${cl.name} is a standalone pod: deploy adds pools to a cluster`, { issues: [{ path: ["cluster_id"], message: "a standalone pod, not a cluster" }] });
  const fams = [...req.families];
  if (req.all_families) for (const f of CATALOG.families) if (!fams.some((x) => x.family === f.id)) fams.push({ family: f.id });
  const existing: PoolSpec[] = cl ? cl.spec.pools : [];
  const taken = new Set(existing.map((p) => p.id));
  const added: { pool: PoolSpec; fi: number; family: string; preset: string }[] = [];
  const pre: Issue[] = [];
  fams.forEach((f, fi) => {
    const { ids, issue } = poolIdsOf(f);
    if (issue) pre.push({ path: ["families", fi, "pools"], message: issue });
    for (const id of ids) added.push({ pool: placePool(id, f, taken), fi, family: f.family, preset: id });
  });
  const pools = [...existing, ...added.map((a) => a.pool)];
  if (pools.length > 16) pre.push({ path: ["families"], message: `${pools.length} pools: a cluster holds at most 16${cl ? ` (${cl.name} has ${existing.length})` : ""}` });
  // A new cluster: the defaults of a cluster definition (image stable, eu, the floors and backstop) with these pools.
  const draft = cl ? { ...cl.spec, pools } : { ...defaultSpec(req.name!), ...(req.control_plane ? { control_plane: req.control_plane } : {}), pools };
  const check = await checkSpec(env, draft, cl ? { id: cl.id, name: cl.name } : undefined);
  // A new pool's issue at its family's entry (the field the page shows); the rest as they are.
  const base = existing.length;
  const issues: Issue[] = [...pre];
  const at = (i: Issue): Issue => {
    if (i.path[0] !== "pools" || typeof i.path[1] !== "number") return i;
    const a = added[(i.path[1] as number) - base];
    if (!a) return i;
    const field = i.path[2];
    return { path: ["families", a.fi, ...(typeof field === "string" && (PLACEMENT as readonly string[]).includes(field) ? i.path.slice(2) : [])], message: `${a.pool.id}${field !== undefined ? ` ${i.path.slice(2).join(".")}` : ""}: ${i.message}` };
  };
  for (const i of check.issues) issues.push(at(i));
  const warnings = check.warnings.map(at);
  let projection: unknown = null;
  let actions: DeployPlan["actions"] = [];
  const running = !!cl && allPods(cl.state).length > 0;
  if (!issues.length && check.normalized) {
    if (cl) {
      const p: any = await planSpec(env, cl.id, check.normalized).catch((e) => ({ ok: false, error: (e as Error).message }));
      projection = p.projection ?? null;
      actions = p.actions ?? [];
    } else {
      projection = await projectSpend(env, check.normalized, { hours: check.normalized.cap_s / 3600 }).catch((e) => ({ error: (e as Error).message }));
      actions = check.normalized.pools.map((p) => ({ action: "create", target: p.id, detail: `${p.count} worker(s) at Start` }));
    }
  }
  const norm = check.normalized as ClusterSpec | undefined;
  const rows: DeployPoolRow[] = added.map((a) => {
    const p = norm?.pools.find((x) => x.id === a.pool.id) ?? a.pool;
    return { id: p.id, family: a.family, preset: a.preset, title: POOL_PRESETS.find((x) => x.id === a.preset)?.title ?? a.preset, count: p.count, provider: p.provider ?? "runpod", where: whereOf(p), models: (p.models || []).map((m) => m.id) };
  });
  return {
    ok: issues.length === 0,
    issues,
    warnings,
    ...(norm && !issues.length ? { normalized: norm } : {}),
    target: { kind: cl ? "existing" : "new", id: cl?.id ?? null, name: cl?.name ?? req.name ?? "", status: cl?.status ?? null, running, version: cl ? await docVersion(env, "cluster-spec", cl.id) : null },
    pools: rows,
    spec: norm ?? draft,
    projection,
    actions,
    cluster: cl,
    merged: norm ?? draft,
  };
}
/** The plan as the API answers it (no internal fields). */
export function planView(p: Awaited<ReturnType<typeof planDeploy>>) {
  const { cluster: _c, merged: _m, normalized: _n, ...rest } = p;
  return rest;
}
/** Exactly normalizeSpec's output for a new cluster's spec (apply re-validates it). */
export const deploySpecOf = (p: Awaited<ReturnType<typeof planDeploy>>): ClusterSpec => normalizeSpec(p.normalized);
