// A cluster definition (the controller's version of runpod-cluster.sh's
// fixed shape): one CPU gateway pod in front of pod pools. Defaults match
// the script (docs/serve/e2e/cluster.md).
import { validate, type GatewayProtocol } from "../schemas";
import { HttpError } from "../util";
import CATALOG from "./catalog.json";
import { WORKER_CONFIGS } from "./worker-configs";
import { regionProblem, type RegionId } from "./regions";
export * from "./regions";

export type CpuFlavor = "cpu3c" | "cpu3g" | "cpu3m" | "cpu5c" | "cpu5g" | "cpu5m";
export interface ModelRef {
  id: string;
  family: string;
  recipe: string;
}
export interface PoolSpec {
  id: string; // h3-turbo | h3-max | ltx | wan | …
  variant: string; // image variant (docs/serve/images.md): h3-turbo, h3-max, ltx, wan5b, gateway (CPU), …
  count: number; // worker pods
  compute: "GPU" | "CPU";
  config?: string; // the worker config inside the image (/etc/fv/runpod.toml …)
  config_toml?: string; // or an inline worker config (FV_WORKER_TOML_B64)
  gpu_types?: string[]; // default: the region's
  regions?: RegionId[]; // default: the cluster's
  cpu_flavors?: CpuFlavor[];
  vcpu?: number;
  container_disk_gb?: number;
  volume?: boolean; // mount the region's network volume at /workspace (GPU default true)
  image?: string; // override: an image reference (resolved to a digest)
  models?: ModelRef[]; // the gateway's static caps
  fake_models?: string[];
  max_queued?: number;
  job_timeout_s?: number;
  stale_after_s?: number;
}
export interface ClusterSpec {
  name: string;
  /** Image source: a release channel (stable, latest, …), a commit (sha), or one image ref for every pod (all-in-one). */
  image: { channel?: string; sha?: string; ref?: string };
  regions: RegionId[];
  gateway: {
    enabled: boolean;
    cpu_flavors: CpuFlavor[];
    vcpu: number;
    container_disk_gb: number;
    base: "pods" | "minimal"; // the gateway TOML (non-pool part)
    github_token: boolean; // FV_GITHUB_TOKEN from GITHUB_PAT (console promote/rollback)
    auth: "keys" | "none";
    /** Replaces the base's `[protocols] fal_apps` (default: the base's: every worker config's apps). */
    fal_apps?: string[];
    /** Overrides single `[protocols]` switches of the base. */
    protocols?: Partial<Record<GatewayProtocol, boolean>>;
    /** `[gateway] reactor_model`; null: none (the gateway takes the first streaming model of a pod pool). Default: the base's when a pool serves it, else a pool's causal model. */
    reactor_model?: string | null;
    /** Replaces the base's `[aliases]`. */
    aliases?: Record<string, string>;
  };
  pools: PoolSpec[];
  /** Backstop: every pod is deleted at create + cap_s (extend moves it). */
  cap_s: number;
  /** The gateway pod's watchdog deletes the cluster below this balance. */
  min_balance: number;
  /** Refuse to start / extend when the projected balance at the deadline would be below this. */
  balance_floor: number;
  /** Refuse to start below this balance (script: FV_CLUSTER_MIN_START). */
  min_start: number;
  /** A GPU pod over this $/hr is deleted right after create (script: RUNPOD_GPU_MAX_DPH). */
  max_gpu_dph: number;
  auto_stop_idle_min?: number | null; // per-cluster override of the idle auto-stop policy
  log_shipping: boolean;
  log_level?: "trace" | "debug" | "info" | "warn" | "error"; // FV_LOG_SHIP_LEVEL
}

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
/** Recipes the fv-serve catalog does not have: a pool model with one stops the gateway at start. */
export const UNSERVABLE_RECIPES = new Set(CATALOG.recipes.filter((r) => !r.serve).map((r) => r.id));

/** A fake-engine worker on a CPU pod (the gateway image carries the fake engine). */
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
  standard: { title: "CPU gateway + h3-turbo, h3-max, ltx, wan GPU pools", pools: ["h3-turbo", "h3-max", "ltx", "wan"] },
  "tiny-cpu": { title: "CPU gateway + 1 fake-engine CPU worker (tests)", pools: [] },
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
    gateway: { enabled: true, cpu_flavors: ["cpu3c", "cpu5c", "cpu3g"], vcpu: 2, container_disk_gb: 20, base: "pods", github_token: true, auth: "keys" },
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
    base.gateway.base = "minimal";
    base.gateway.github_token = false;
    base.cap_s = 1800;
    base.min_start = 10;
    base.pools = [
      { id: "fake", variant: "gateway", count: 1, compute: "CPU", config_toml: FAKE_CPU_WORKER_TOML, cpu_flavors: ["cpu3c", "cpu5c", "cpu3g"], vcpu: 2, container_disk_gb: 10, volume: false, fake_models: ["fake-wan"], max_queued: 8, job_timeout_s: 600, stale_after_s: 120 },
    ];
  }
  return base;
}

const ID_RE = /^[a-z][a-z0-9-]{0,30}$/;
const VARIANT_RE = /^[a-z0-9][a-z0-9-]{0,30}$/;

/** Validates and fills defaults; throws 400 with the first problem. */
export function normalizeSpec(input: any): ClusterSpec {
  if (!input || typeof input !== "object") throw new HttpError(400, "spec must be an object");
  const name = String(input.name || "");
  if (!ID_RE.test(name)) throw new HttpError(400, "name: lower-case letters, digits and '-', starting with a letter (max 31)");
  const d = defaultSpec(name, typeof input.template === "string" && TEMPLATES[input.template] ? input.template : "standard");
  const s: ClusterSpec = {
    ...d,
    ...input,
    gateway: { ...d.gateway, ...(input.gateway || {}) },
    image: input.image ? { ...input.image } : d.image,
    pools: Array.isArray(input.pools) ? input.pools : d.pools,
    regions: Array.isArray(input.regions) && input.regions.length ? input.regions : d.regions,
  };
  delete (s as any).template;
  const img = s.image;
  const nSrc = [img.channel, img.sha, img.ref].filter(Boolean).length;
  if (nSrc !== 1) throw new HttpError(400, "image: exactly one of channel, sha, ref");
  if (img.channel && !/^[a-z][a-z0-9-]{0,30}$/.test(img.channel)) throw new HttpError(400, "image.channel: a lower-case word");
  if (img.sha && !/^[0-9a-f]{7,40}$/.test(img.sha)) throw new HttpError(400, "image.sha: 7-40 hex characters");
  if (img.ref && !/^[a-z0-9.\-]+(:[0-9]+)?\/[a-z0-9._\-/]+(:[A-Za-z0-9._-]+)?(@sha256:[0-9a-f]{64})?$/.test(img.ref)) throw new HttpError(400, "image.ref: an image reference");
  // An unavailable region (us: its volume is gone) is rejected, not dropped:
  // dropping would silently change where a saved cluster places workers (and
  // rewrite the stored spec on the next save); a 400 makes the owner choose.
  for (const r of s.regions) {
    const why = regionProblem(r);
    if (why) throw new HttpError(400, `regions: ${why}`);
  }
  const ids = new Set<string>();
  s.pools = s.pools.map((p: any, i: number) => {
    // A standard or preset pool id fills what the entry leaves out; its own config wins.
    const std = presetPool(String(p?.id ?? ""));
    if (std && (p?.config || p?.config_toml)) delete std.config, delete std.config_toml;
    const q: PoolSpec = { ...(std ?? {}), ...p } as PoolSpec;
    if (!ID_RE.test(q.id || "")) throw new HttpError(400, `pools[${i}].id: invalid`);
    if (ids.has(q.id)) throw new HttpError(400, `pools: ${q.id} twice`);
    ids.add(q.id);
    if (!VARIANT_RE.test(q.variant || "")) throw new HttpError(400, `pools[${i}].variant: invalid`);
    q.count = Number(q.count ?? 1);
    if (!Number.isInteger(q.count) || q.count < 0 || q.count > 8) throw new HttpError(400, `pools[${i}].count: 0-8`);
    q.compute = q.compute === "CPU" ? "CPU" : "GPU";
    if (!q.config && !q.config_toml) throw new HttpError(400, `pools[${i}]: config or config_toml`);
    if (q.config_toml && q.config_toml.length > 32768) throw new HttpError(400, `pools[${i}].config_toml: too long`);
    for (const r of q.regions || []) {
      const why = regionProblem(r);
      if (why) throw new HttpError(400, `pools[${i}].regions: ${why}`);
    }
    if (!(q.models?.length || q.fake_models?.length)) throw new HttpError(400, `pools[${i}]: models or fake_models (the gateway's static caps)`);
    for (const [j, m] of (q.models || []).entries())
      if (UNSERVABLE_RECIPES.has(m?.recipe)) throw new HttpError(400, `pools[${i}].models[${j}].recipe: ${m.recipe} is not in the fv-serve catalog of this build (LongLive-Plug recipes run in fv-gpucheck / the CLI); the gateway would refuse to start`);
    return q;
  });
  const num = (k: keyof ClusterSpec, lo: number, hi: number) => {
    const v = Number(s[k]);
    if (!Number.isFinite(v) || v < lo || v > hi) throw new HttpError(400, `${String(k)}: ${lo}-${hi}`);
    (s as any)[k] = v;
  };
  num("cap_s", 300, 7 * 86400);
  num("min_balance", 8, 10000);
  num("balance_floor", 8, 10000);
  num("min_start", 8, 10000);
  num("max_gpu_dph", 0.1, 50);
  if (!["keys", "none"].includes(s.gateway.auth)) throw new HttpError(400, "gateway.auth: keys | none");
  if (!["pods", "minimal"].includes(s.gateway.base)) throw new HttpError(400, "gateway.base: pods | minimal");
  s.gateway.vcpu = Number(s.gateway.vcpu) || 2;
  s.log_shipping = s.log_shipping !== false;
  // The same schema the editor validates against (src/schemas.ts).
  const v = validate("cluster-spec", s);
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".") || "spec"}: ${i.message}`).join("; "), { issues: v.issues });
  return s;
}
