// A cluster definition: pod pools of fv-serve workers behind the edge
// Worker (control_plane "edge", docs/serve/edge-control-plane.md), or
// workers clients call directly ("direct", docs/control/gateway-less-auth.md).
import type { z } from "zod";
import { validate, type ClusterSpecZ, type ModelRefZ, type PoolSpecZ } from "../schemas";
import { HttpError } from "../util";
import CATALOG from "./catalog.json";
import { WORKER_CONFIGS } from "./worker-configs";
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

export const STANDARD_POOLS: PoolSpec[] = [
  { id: "h3-turbo", variant: "h3-turbo", count: 1, compute: "GPU", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  { id: "h3-max", variant: "h3-max", count: 1, compute: "GPU", config: "/etc/fv/runpod-h3-max.toml", models: [{ id: "sol-h3", family: "h3", recipe: "h3-max" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 180 },
  { id: "ltx", variant: "ltx", count: 1, compute: "GPU", config: "/etc/fv/runpod-ltx.toml", models: [{ id: "ltx25-distill-sol", family: "ltx2", recipe: "ltx-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  {
    id: "wan",
    variant: "wan5b",
    count: 1,
    compute: "GPU",
    config: "/etc/fv/runpod-wan5b.toml",
    models: [
      { id: "fastwan22-ti2v-5b", family: "wan", recipe: "wan-turbo" },
      { id: "wan22-ti2v-5b", family: "wan", recipe: "wan-max" },
    ],
    max_queued: 64,
    job_timeout_s: 1800,
    stale_after_s: 120,
  },
];

/** A pool the dashboard can add: an image variant plus the worker config it runs. */
export interface PoolPreset {
  id: string;
  title: string;
  description: string;
  /** Weight trees (under /workspace/weights on both volumes) the workers load. */
  weights: string[];
  /** Set when the weights' licence restricts use. */
  licence?: string;
  pool: PoolSpec;
}
const inline = (file: string) => {
  const t = WORKER_CONFIGS[file];
  if (!t) throw new Error(`no generated worker config ${file} (node gen-configs.mjs)`);
  return t;
};
// The presets reuse the image variants CI builds (docs/serve/images.md); a
// config the variant's image does not carry rides inline (FV_WORKER_TOML_B64,
// generated from configs/serve by gen-configs.mjs).
export const POOL_PRESETS: PoolPreset[] = [
  { id: "h3-turbo", title: "H3 turbo (fasth3)", description: "MiniMax H3 turbo tier; also the Reactor's clip model.", weights: ["h3-base"], pool: STANDARD_POOLS[0]! },
  { id: "h3-max", title: "H3 max (Sol-H3)", description: "MiniMax H3 max tier (Sol-H3 4-step ladder).", weights: ["h3-base"], pool: STANDARD_POOLS[1]! },
  { id: "ltx", title: "LTX turbo", description: "LTX-2.5 turbo tier (Sol stage 2): fal lightricks/ltx-2.5 /fast.", weights: ["ltx25"], pool: STANDARD_POOLS[2]! },
  { id: "wan", title: "Wan 2.2 5B (turbo + max)", description: "Wan 2.2 TI2V-5B: FastWan turbo and the 50-step max tier; fal-ai/wan.", weights: ["fastwan22-ti2v-5b", "wan22-ti2v-5b"], pool: STANDARD_POOLS[3]! },
  {
    id: "ltx-pro",
    title: "LTX pro (dense)",
    description: "LTX-2.5 pro tier (ltx25-distill-dense, two-stage dense): fal lightricks/ltx-2.5 /pro and fal-ai/ltx-2.3 retake / extend. Sage attention is on by default on sm_120 (RTX PRO 6000; FASTVIDEO_ATTN_SAGE=0 turns it off).",
    weights: ["ltx25"],
    pool: { id: "ltx-pro", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-pro.toml"), models: [{ id: "ltx25-distill-dense", family: "ltx2", recipe: "ltx-pro" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-a2v",
    title: "LTX guided audio-to-video",
    description: "The guided A2V companion of ltx-pro (LTX-2.5 dev DiT, multimodal guider): fal lightricks/ltx-2.5 audio-to-video/pro. An 80-96 GB card and ~38 GB of host RAM.",
    weights: ["ltx25", "ltx25-dev"],
    pool: { id: "ltx-a2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-a2v.toml"), container_disk_gb: 60, models: [{ id: "ltx25-a2v-guided", family: "ltx2", recipe: "ltx25-a2v-guided" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-ref2v",
    title: "LTX reference-to-video",
    description: "The Ref2V companion of ltx-pro (Ingredients IC-LoRA): fal fal-ai/ltx-2.3-quality ingredient.",
    weights: ["ltx25", "ltx25-ic-lora-ingredients"],
    pool: { id: "ltx-ref2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-ref2v.toml"), models: [{ id: "ltx25-ref2v", family: "ltx2", recipe: "ltx25-ref2v" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "h3-ref2v",
    title: "H3 reference-to-video",
    description: "H3 Ref2VA, turbo and max (the reference companions of h3-turbo / h3-max).",
    weights: ["h3-base", "h3-ref2va"],
    pool: {
      id: "h3-ref2v",
      variant: "h3-max",
      count: 1,
      compute: "GPU",
      config_toml: inline("runpod-h3-ref2v.toml"),
      models: [
        { id: "h3-ref2v-turbo", family: "h3", recipe: "h3-ref2v-turbo" },
        { id: "h3-ref2v-max", family: "h3", recipe: "h3-ref2v-max" },
      ],
      max_queued: 16,
      job_timeout_s: 3600,
      stale_after_s: 180,
    },
  },
  {
    id: "fastwan21",
    title: "FastWan 2.1 1.3B",
    description: "FastWan 2.1 T2V 1.3B (480p, 3 DMD steps): fal fastvideo/fastwan21-1.3b.",
    weights: ["fastwan21-1.3b"],
    pool: { id: "fastwan21", variant: "wan", count: 1, compute: "GPU", config: "/etc/fv/runpod-wan.toml", models: [{ id: "fastwan21-1.3b", family: "wan", recipe: "fastwan21-1.3b" }], max_queued: 64, job_timeout_s: 900, stale_after_s: 90 },
  },
  {
    id: "sfwan",
    title: "SF-Wan causal streaming",
    description: "SF-Wan 1.3B causal rollout: Reactor and native streams (one session per GPU); the cluster's Reactor model when no pool serves fasth3.",
    weights: ["sfwan21-1.3b"],
    pool: { id: "sfwan", variant: "sfwan", count: 1, compute: "GPU", config: "/etc/fv/runpod-sfwan.toml", models: [{ id: "sfwan21-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
  {
    id: "longlive",
    title: "LongLive-1.3B causal (NON-COMMERCIAL)",
    description: "LongLive-1.3B on the SF-Wan engine (window 12, sink 3, KV re-cache at prompt switches): Reactor and native streams. NON-COMMERCIAL licence: research and evaluation only.",
    weights: ["longlive-1.3b-safetensors", "sfwan21-1.3b"],
    licence: "LongLive-1.3B weights: CC-BY-NC-SA-4.0 (non-commercial; research / evaluation only)",
    pool: { id: "longlive", variant: "sfwan", count: 1, compute: "GPU", config_toml: inline("runpod-longlive.toml"), models: [{ id: "longlive-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
];
export const presetPool = (id: string): PoolSpec | undefined => {
  const p = POOL_PRESETS.find((x) => x.id === id || x.pool.id === id)?.pool;
  return p ? structuredClone(p) : undefined;
};
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
