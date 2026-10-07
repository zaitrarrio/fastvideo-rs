// A cluster definition: pod pools of fv-serve workers behind the edge
// Worker (control_plane "edge", docs/serve/edge-control-plane.md), or
// workers clients call directly ("direct", docs/control/gateway-less-auth.md).
import type { z } from "zod";
import { validate, type ClusterSpecZ, type ModelRefZ, type PoolSpecZ } from "../schemas";
import { HttpError } from "../util";
import CATALOG from "./catalog.json";
import { POOL_PRESETS, presetPool, STANDARD_POOLS } from "../presets";
import { regionProblem, type RegionId } from "./regions";
export * from "./regions";

export type { CpuFlavor } from "../enums";
// The spec's types are the zod schemas' (src/schemas.ts): one definition.
export type ModelRef = z.infer<typeof ModelRefZ>;
export type PoolSpec = z.infer<typeof PoolSpecZ>;
/** A cluster: image source (a release channel, a commit, or one image ref for every pod), regions, control plane
 * (edge: the edge Worker is the only front; direct: clients call each worker), auth, pools, backstops and money floors. */
export type ClusterSpec = z.infer<typeof ClusterSpecZ>;
export type ControlPlane = ClusterSpec["control_plane"];

// The preset catalog (pool presets and the standard pools) lives in
// ../presets.ts: one source for cluster pools, standalone pods and serverless
// endpoints.
export { POOL_PRESETS, presetPool, STANDARD_POOLS, type PoolPreset } from "../presets";

/** Recipes the fv-serve catalog does not have: a pool model with one cannot be served. */
export const UNSERVABLE_RECIPES = new Set(CATALOG.recipes.filter((r) => !r.serve).map((r) => r.id));

/** A fake-engine worker on a CPU pod (the `cpu` image). */
export const FAKE_CPU_WORKER_TOML = `[server]
bind = "0.0.0.0:8000"
state_dir = "/fvstate"
shutdown_grace_s = 10

[auth]
mode = "keys"

[artifacts]
backend = "auto"
url_ttl_s = 3600

[jobs]
backend = "auto"
progress_interval_ms = 1000
stale_after_s = 900

[engine]
backend = "fake"

[engine.fake]
step_ms = 20
load_ms = 500
all_resident = true
placeholder_output = true

[protocols]
native = true

[limits]
queue_max = 8
body_max_mb = 16
`;

/** Cluster templates (GET /api/templates): standard, tiny-cpu, and preset groups. */
export const TEMPLATES: Record<string, { title: string; pools: string[] }> = {
  standard: { title: "h3-turbo, h3-max, ltx, wan GPU pools", pools: ["h3-turbo", "h3-max", "ltx", "wan"] },
  "tiny-cpu": { title: "1 fake-engine CPU worker (tests)", pools: [] },
  ltx: { title: "LTX: turbo, pro, guided A2V and Ref2V pools", pools: ["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v"] },
  h3: { title: "H3: turbo, max and Ref2V pools", pools: ["h3-turbo", "h3-max", "h3-ref2v"] },
  wan: { title: "Wan: 2.2 5B, FastWan 1.3B and SF-Wan streaming pools", pools: ["wan", "fastwan21", "sfwan"] },
  longlive: { title: "LongLive-1.3B streaming pool (NON-COMMERCIAL weights)", pools: ["longlive"] },
};
export type TemplateId = keyof typeof TEMPLATES;

export function defaultSpec(name: string, template: string = "standard"): ClusterSpec {
  const base: ClusterSpec = {
    name,
    image: { channel: "stable" },
    regions: ["eu"],
    control_plane: "edge",
    auth: "keys",
    pools: structuredClone(STANDARD_POOLS),
    cap_s: 6000,
    min_balance: 8.25,
    balance_floor: 8,
    min_start: 20,
    max_gpu_dph: 3.6,
    auto_stop_idle_min: null,
    log_shipping: true,
  };
  if (template !== "standard" && template !== "tiny-cpu" && TEMPLATES[template]) base.pools = TEMPLATES[template]!.pools.map((p) => presetPool(p)!);
  if (template === "tiny-cpu") {
    base.cap_s = 1800;
    base.min_start = 10;
    base.pools = [
      { id: "fake", variant: "cpu", count: 1, compute: "CPU", config_toml: FAKE_CPU_WORKER_TOML, cpu_flavors: ["cpu3c", "cpu5c", "cpu3g"], vcpu: 2, container_disk_gb: 10, volume: false, fake_models: ["fake-wan"], max_queued: 8, job_timeout_s: 600, stale_after_s: 120 },
    ];
  }
  return base;
}


/** Fills defaults (a template, preset pools, legacy gateway fields) and validates with the zod schema; throws 400 with every problem (`extra.issues`, path-anchored). */
export function normalizeSpec(input: any): ClusterSpec {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new HttpError(400, "spec must be an object", { issues: [{ path: [], message: "spec must be an object" }] });
  const name = typeof input.name === "string" ? input.name : "";
  const d = defaultSpec(name, typeof input.template === "string" && TEMPLATES[input.template] ? input.template : "standard");
  // A spec stored before the gateway was retired: its `gateway` block gives
  // the auth mode and, with the gateway off, direct mode; a gateway
  // cluster becomes an edge cluster.
  const legacy = input.gateway && typeof input.gateway === "object" ? input.gateway : null;
  let plane = input.control_plane;
  if (plane === "gateway" || (plane === undefined && legacy)) plane = legacy && legacy.enabled === false ? "direct" : "edge";
  const s: ClusterSpec = {
    ...d,
    ...input,
    name: input.name,
    control_plane: plane ?? d.control_plane,
    auth: input.auth ?? legacy?.auth ?? d.auth,
    image: input.image && typeof input.image === "object" ? { ...input.image } : d.image,
    pools: Array.isArray(input.pools) ? input.pools : d.pools,
    regions: Array.isArray(input.regions) && input.regions.length ? input.regions : d.regions,
  };
  delete (s as any).template;
  delete (s as any).gateway;
  const issues: { path: (string | number)[]; message: string }[] = [];
  // An unavailable region (us: its volume is gone) is rejected, not dropped:
  // dropping would silently change where a saved cluster places workers (and
  // rewrite the stored spec on the next save); a 400 makes the owner choose.
  s.regions.forEach((r, i) => {
    const why = regionProblem(r);
    if (why) issues.push({ path: ["regions", i], message: why });
  });
  s.pools = s.pools.map((p: any, i: number) => {
    if (!p || typeof p !== "object") return p;
    // A standard or preset pool id fills what the entry leaves out; its own config wins.
    const std = presetPool(String(p?.id ?? ""));
    if (std && (p?.config || p?.config_toml)) delete std.config, delete std.config_toml;
    const q: PoolSpec = { ...(std ?? {}), ...p } as PoolSpec;
    if ((q.variant as string) === "gateway") q.variant = "cpu"; // the CPU image's name before the gateway was retired
    if (q.count === undefined) q.count = 1;
    if (q.compute === undefined) q.compute = "GPU";
    (q.regions || []).forEach((r, j) => {
      const why = regionProblem(r);
      if (why) issues.push({ path: ["pools", i, "regions", j], message: why });
    });
    for (const [j, m] of (q.models || []).entries())
      if (UNSERVABLE_RECIPES.has(m?.recipe)) issues.push({ path: ["pools", i, "models", j, "recipe"], message: `${m.recipe} is not in the fv-serve catalog of this build (LongLive-Plug recipes run in fv-gpucheck / the CLI)` });
    return q;
  });
  if (s.control_plane === ("gateway" as any)) issues.push({ path: ["control_plane"], message: "edge | direct (the gateway is retired)" });
  if (s.log_shipping === undefined) s.log_shipping = true;
  // The same schema the editor validates against (src/schemas.ts).
  const v = validate("cluster-spec", s);
  if (!v.ok) for (const i of v.issues) if (!issues.some((x) => x.path.join(".") === i.path.join("."))) issues.push(i);
  if (issues.length) throw new HttpError(400, issues.map((i) => `${i.path.join(".") || "spec"}: ${i.message}`).join("; "), { issues });
  return v.ok ? (v.value as ClusterSpec) : s;
}

/** A spec stored before the gateway was retired, read as today's: its
 * `gateway` block gives the auth mode, and direct mode when the gateway was
 * off; a gateway cluster reads as an edge cluster (normalizeSpec does the
 * same on the next save). */
export function migrateSpec(spec: ClusterSpec): ClusterSpec {
  const any = spec as any;
  if (!any || typeof any !== "object") return spec;
  const legacy = any.gateway && typeof any.gateway === "object" ? any.gateway : null;
  if (any.control_plane === "gateway" || (any.control_plane === undefined && legacy)) any.control_plane = legacy && legacy.enabled === false ? "direct" : "edge";
  if (any.auth === undefined) any.auth = legacy?.auth ?? "keys";
  for (const p of any.pools || []) if (p?.variant === "gateway") p.variant = "cpu";
  delete any.gateway;
  return spec;
}

/** The edge family DO a pool's model queues on (docs/serve/edge-control-plane.md §5.2). */
export function modelFamily(pool: PoolSpec, m?: ModelRef): string {
  if (pool.family) return pool.family;
  if (!m) return "fake";
  if (m.family === "ltx2") return "ltx";
  if (m.family === "wan" && /^(sfwan|longlive)/.test(m.recipe)) return "sfwan";
  return m.family;
}

/** Model id → family for every model a pool serves (fake models included). */
export function poolModelFamilies(pool: PoolSpec): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of pool.models || []) out[m.id] = modelFamily(pool, m);
  for (const f of pool.fake_models || []) out[f] = modelFamily(pool);
  return out;
}

export const isEdge = (spec: ClusterSpec) => spec.control_plane !== "direct";
