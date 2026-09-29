// A cluster definition (the controller's version of runpod-cluster.sh's
// fixed shape): one CPU gateway pod in front of pod pools. Defaults match
// the script (docs/serve/e2e/cluster.md).
import { HttpError } from "../util";

export interface RegionDef {
  volume: string;
  dc: string;
  gpus: string[];
}
/** CLAUDE.md: the two network volumes (weights live on both). */
export const REGIONS: Record<string, RegionDef> = {
  eu: { volume: "jg48s6o1w0", dc: "EUR-IS-1", gpus: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] },
  us: { volume: "s2k01690bi", dc: "US-CA-2", gpus: ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200"] },
};

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
  regions?: string[]; // default: the cluster's
  cpu_flavors?: string[];
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
  regions: string[];
  gateway: {
    enabled: boolean;
    cpu_flavors: string[];
    vcpu: number;
    container_disk_gb: number;
    base: "pods" | "minimal"; // the gateway TOML (non-pool part)
    github_token: boolean; // FV_GITHUB_TOKEN from GITHUB_PAT (console promote/rollback)
    auth: "keys" | "none";
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
  log_level?: string; // FV_LOG_SHIP_LEVEL
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

export function defaultSpec(name: string, template: "standard" | "tiny-cpu" = "standard"): ClusterSpec {
  const base: ClusterSpec = {
    name,
    image: { channel: "stable" },
    regions: ["eu", "us"],
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
  const d = defaultSpec(name, input.template === "tiny-cpu" ? "tiny-cpu" : "standard");
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
  for (const r of s.regions) if (!REGIONS[r]) throw new HttpError(400, `regions: unknown region ${r} (eu, us)`);
  const ids = new Set<string>();
  s.pools = s.pools.map((p: any, i: number) => {
    const std = STANDARD_POOLS.find((x) => x.id === p?.id);
    const q: PoolSpec = { ...(std ? structuredClone(std) : {}), ...p } as PoolSpec;
    if (!ID_RE.test(q.id || "")) throw new HttpError(400, `pools[${i}].id: invalid`);
    if (ids.has(q.id)) throw new HttpError(400, `pools: ${q.id} twice`);
    ids.add(q.id);
    if (!VARIANT_RE.test(q.variant || "")) throw new HttpError(400, `pools[${i}].variant: invalid`);
    q.count = Number(q.count ?? 1);
    if (!Number.isInteger(q.count) || q.count < 0 || q.count > 8) throw new HttpError(400, `pools[${i}].count: 0-8`);
    q.compute = q.compute === "CPU" ? "CPU" : "GPU";
    if (!q.config && !q.config_toml) throw new HttpError(400, `pools[${i}]: config or config_toml`);
    if (q.config_toml && q.config_toml.length > 32768) throw new HttpError(400, `pools[${i}].config_toml: too long`);
    for (const r of q.regions || []) if (!REGIONS[r]) throw new HttpError(400, `pools[${i}].regions: unknown ${r}`);
    if (!(q.models?.length || q.fake_models?.length)) throw new HttpError(400, `pools[${i}]: models or fake_models (the gateway's static caps)`);
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
  return s;
}
