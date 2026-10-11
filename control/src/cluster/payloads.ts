// Pod payloads and env of a cluster's workers: edge fronts
// (docs/serve/edge-control-plane.md §5.2) or direct workers
// (docs/control/gateway-less-auth.md). (The gateway pod is retired.)
import { isEdge, poolModelFamilies, REGIONS, regionAvailable, type ClusterSpec, type PoolSpec, type RegionId } from "./spec";

/** Runpod secret references (values live in Runpod, never here). */
export const SECRET_ENV_REFS: Record<string, string> = {
  FV_CF_ACCOUNT_ID: "{{ RUNPOD_SECRET_fv_cf_account_id }}",
  FV_CF_API_TOKEN: "{{ RUNPOD_SECRET_fv_cf_api_token }}",
  FV_D1_DATABASE_ID: "{{ RUNPOD_SECRET_fv_d1_database_id }}",
  FV_R2_BUCKET: "{{ RUNPOD_SECRET_fv_r2_bucket }}",
  FV_R2_ENDPOINT: "{{ RUNPOD_SECRET_fv_r2_endpoint }}",
  FV_R2_ACCESS_KEY_ID: "{{ RUNPOD_SECRET_fv_r2_access_key_id }}",
  FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}",
  FV_WEBHOOK_ED25519_KEY: "{{ RUNPOD_SECRET_fv_webhook_ed25519_key }}",
};

// A worker's start command (direct workers): the config from the image, or
// inline (FV_WORKER_TOML_B64) when the pool has one.
export const WORKER_BOOT = `set -u
echo "[fv-boot] start" >&2
if [ -d /workspace/weights ]; then echo "[fv-boot] volume: /workspace/weights ($(ls /workspace/weights 2>/dev/null | wc -l) trees)" >&2; else echo "[fv-boot] volume: /workspace/weights missing" >&2; fi
mkdir -p /fvstate
if [ -n "\${FV_WORKER_TOML_B64:-}" ]; then
  printf "%s" "$FV_WORKER_TOML_B64" | base64 -d > /fv-worker.toml
else
  cp "$FV_WORKER_CONFIG" /fv-worker.toml
fi
if grep -q "^\\[gateway\\]" /fv-worker.toml; then
  sed -i "/^\\[gateway\\]/a register = false" /fv-worker.toml
else
  printf "\\n[gateway]\\nregister = false\\n" >> /fv-worker.toml
fi
export FV_WORKER_ID="\${RUNPOD_POD_ID}"
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml`;

// The backstop watchdog (a subshell in the background): at the deadline or
// below the balance floor the pod deletes itself (fv-control's cron enforces
// the deadline too). Edge fronts and standalone pods run it.
const WATCHDOG = `(
  command -v curl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl; } >/fvstate/watchdog-apt.log 2>&1
  bye() {
    echo "[watchdog] $1: deleting \${RUNPOD_POD_ID}" >&2
    for _ in 1 2 3; do
      curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null && break
      sleep 5
    done
  }
  n=0
  while :; do
    if [ "$(date +%s)" -ge "$FV_CLUSTER_DEADLINE" ]; then bye deadline; sleep 60; continue; fi
    if [ $((n % 2)) -eq 0 ]; then
      b=$(curl -sS --max-time 20 -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" -H "content-type: application/json" \\
        https://api.runpod.io/graphql -d "{\\"query\\":\\"{ myself { clientBalance } }\\"}" | sed -n "s/.*\\"clientBalance\\":\\([0-9.]*\\).*/\\1/p")
      if [ -n "$b" ] && awk -v b="$b" -v m="$FV_MIN_BALANCE" "BEGIN{exit !(b+0 < m+0)}"; then bye "balance $b below $FV_MIN_BALANCE"; fi
    fi
    n=$((n + 1))
    sleep 30
  done
) &
`;
const EXEC_SERVE = "exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml";
const WORKER_PREFIX = WORKER_BOOT.slice(0, WORKER_BOOT.indexOf(EXEC_SERVE));

// An edge front's start command (control_plane = edge,
// docs/serve/edge-control-plane.md §5.2): the worker's, plus its own
// endpoint (where the edge forwards) and the backstop watchdog.
export const EDGE_WORKER_BOOT = `${WORKER_PREFIX}export FV_DISPATCH_ENDPOINT="https://\${RUNPOD_POD_ID}-8000.proxy.runpod.net"
${WATCHDOG}${EXEC_SERVE}`;

// A standalone pod's start command (docs/control/standalone-pods.md): a
// direct worker with the backstop watchdog (its deadline and balance floor).
export const WATCHDOG_WORKER_BOOT = `${WORKER_PREFIX}${WATCHDOG}${EXEC_SERVE}`;

/** FV_IMAGE_REF / FV_IMAGE_DIGEST / FV_RELEASE_CHANNEL (fv_image_env_json). */
export function imageIdentEnv(image: string, channel?: string): Record<string, string> {
  const e: Record<string, string> = { FV_IMAGE_REF: image };
  if (image.includes("@sha256:")) e.FV_IMAGE_DIGEST = image.split("@")[1]!;
  if (channel) e.FV_RELEASE_CHANNEL = channel;
  return e;
}

export interface PodRec {
  pod: string;
  pool?: string;
  gpu?: string;
  cpu?: string;
  dc?: string;
  dph: number;
  created: number; // unix s (the script's format)
  image: string;
  url?: string;
}
export interface ClusterState {
  image?: string; // the all-in-one image when there is one
  images: Record<string, string>; // per pool
  /** A gateway pod of a state from before the gateway was retired: it is only ever deleted. */
  gateway?: PodRec;
  gateway_url?: string;
  gateway_stopped?: boolean;
  workers: Record<string, PodRec[]>;
  rolling?: Record<string, PodRec[]>;
  retired?: PodRec[];
  /** Per pool, what the last `up` found (docs/control/README.md §4 "Early failure"): no stock, a failed pod, ready. */
  pools?: Record<string, PoolStatus>;
}
/** A pool's state after `up`: starting, ready, or why it has no serving pod. */
export interface PoolStatus {
  status: "starting" | "ready" | "no_stock" | "failed";
  detail?: string;
  at: number;
}
export interface ClusterSecrets {
  internal_token: string;
  url_signing_key: string;
  admin_recipient?: string; // legacy (the retired gateway's sealed admin token): unused
  admin_private?: string; // legacy: unused
  admin_token?: string; // direct workers' FV_ADMIN_TOKEN (made by the controller)
  legacy_admin_token?: boolean; // legacy (imported runpod-cluster.sh state): unused
  ingest_token?: string;
  smoke_api_key?: string;
}

export { RESERVED_KEYS, isReserved } from "../enums";
/** System env keys whose values are secret (masked in every view). */
export const SECRET_SYSTEM_KEYS = new Set(["FV_INTERNAL_TOKEN", "FV_URL_SIGNING_KEY", "FV_BACKSTOP_API_KEY", "FV_ADMIN_TOKEN", "FV_LOG_SHIP_TOKEN", "FV_ENDPOINT_REPORT_TOKEN"]);

/** The edge Worker a control_plane = edge cluster fronts through (fv-control's EDGE_* settings). */
export interface EdgeCfg {
  url: string;
  internal_token: string;
  admin_token: string;
  d1_database_id?: string;
  /** The edge's outputs bucket: direct uploads land there and the workers presign result URLs for it. */
  outputs_bucket?: string;
}
export interface EnvCtx {
  edge?: EdgeCfg;
  spec: ClusterSpec;
  state: ClusterState;
  secrets: ClusterSecrets;
  deadlineMs: number;
  runpodApiKey: string;
  ingestUrl?: string;
  /** A standalone pod (direct, no edge): its own backstop watchdog (deadline, balance floor) like an edge front's. */
  backstop?: boolean;
}

function b64utf8(s: string): string {
  const u = new TextEncoder().encode(s);
  let bin = "";
  for (const x of u) bin += String.fromCharCode(x);
  return btoa(bin);
}

function logShipEnv(ctx: EnvCtx): Record<string, string> {
  if (!ctx.spec.log_shipping || !ctx.ingestUrl || !ctx.secrets.ingest_token) return {};
  return { FV_LOG_SHIP_URL: ctx.ingestUrl, FV_LOG_SHIP_TOKEN: ctx.secrets.ingest_token, FV_LOG_SHIP_LEVEL: ctx.spec.log_level || "info" };
}

/** Workers serve clients themselves (control_plane "direct", docs/control/gateway-less-auth.md). */
export function isDirect(spec: ClusterSpec, _state?: ClusterState): boolean {
  return !isEdge(spec);
}

/** An edge front's env (docs/serve/edge-control-plane.md §5.2): the edge's
 * internal token, its family objects, results through the edge to R2, the
 * edge's D1 (keys, jobs) instead of the account's job store, no R2
 * credentials, and the backstop of EDGE_WORKER_BOOT. */
function edgeEnv(ctx: EnvCtx, pool: PoolSpec): Record<string, string> {
  if (!isEdge(ctx.spec)) return {};
  const edge = ctx.edge;
  if (!edge) throw new Error("control_plane = edge needs fv-control's EDGE_URL, EDGE_INTERNAL_TOKEN and EDGE_ADMIN_TOKEN");
  const fams = poolModelFamilies(pool);
  const e: Record<string, string> = {
    FV_AUTH_MODE: "trust-edge",
    FV_INTERNAL_TOKEN: edge.internal_token,
    FV_PUBLIC_BASE_URL: edge.url,
    FV_DISPATCH_FRONT: "1",
    FV_DISPATCH_DO_URL: edge.url,
    FV_DISPATCH_FAMILIES: [...new Set(Object.values(fams))].sort().join(","),
    FV_DISPATCH_MODEL_FAMILIES: Object.entries(fams)
      .map(([m, f]) => `${m}=${f}`)
      .join(","),
    FV_DISPATCH_DIRECT_UPLOAD: "1",
    FV_MP4_FRAGMENTED: "1",
    FV_CLUSTER_DEADLINE: String(Math.floor(ctx.deadlineMs / 1000)),
    FV_MIN_BALANCE: String(ctx.spec.min_balance),
    FV_BACKSTOP_API_KEY: ctx.runpodApiKey,
  };
  if (pool.max_queued !== undefined) e.FV_DISPATCH_MAX_QUEUED = String(pool.max_queued);
  if (edge.d1_database_id) e.FV_D1_DATABASE_ID = edge.d1_database_id;
  return e;
}

/** A direct worker's client auth: the spec's auth mode, the cluster's admin
 * token, minted keys in the shared D1 table (every worker, restarts and new
 * pods included, sees the same keys). */
function directEnv(ctx: EnvCtx): Record<string, string> {
  if (!isDirect(ctx.spec, ctx.state)) return {};
  const e: Record<string, string> = { FV_WORKER_DIRECT: "1", FV_AUTH_MODE: ctx.spec.auth, FV_KEY_STORE: "d1" };
  if (ctx.secrets.admin_token) e.FV_ADMIN_TOKEN = ctx.secrets.admin_token;
  // A standalone pod deletes itself at its deadline or below the floor (WATCHDOG_WORKER_BOOT).
  if (ctx.backstop) Object.assign(e, { FV_CLUSTER_DEADLINE: String(Math.floor(ctx.deadlineMs / 1000)), FV_MIN_BALANCE: String(ctx.spec.min_balance), FV_BACKSTOP_API_KEY: ctx.runpodApiKey });
  return e;
}

export function workerSystemEnv(ctx: EnvCtx, pool: PoolSpec, image: string): Record<string, string> {
  const e: Record<string, string> = {
    ...SECRET_ENV_REFS,
    ...imageIdentEnv(image, ctx.spec.image.channel),
    FV_SERVE_MODE: "http",
    FV_SERVE_ROLE: "worker",
    FV_INTERNAL_TOKEN: ctx.secrets.internal_token,
    FV_URL_SIGNING_KEY: ctx.secrets.url_signing_key,
    FV_PUBLIC_BASE_URL: "",
    FV_WORKER_CONFIG: pool.config || "/fv-worker.toml",
    FV_STATE_DIR: "/fvstate",
    FV_WEIGHTS: "/workspace/weights",
    FV_JOBS_HEARTBEAT_S: "10",
    RUST_LOG: "info",
  };
  if (pool.config_toml) e.FV_WORKER_TOML_B64 = b64utf8(pool.config_toml);
  const out = { ...e, ...directEnv(ctx), ...edgeEnv(ctx, pool), ...logShipEnv(ctx) };
  // Edge fronts never use the account's (production) bucket: they upload
  // results through the edge into its outputs bucket and presign their URLs
  // there (the account's R2 credentials, the edge's bucket); without that
  // bucket they get no R2 at all.
  if (isEdge(ctx.spec)) {
    if (ctx.edge?.outputs_bucket) out.FV_R2_BUCKET = ctx.edge.outputs_bucket;
    else for (const k of Object.keys(out)) if (k.startsWith("FV_R2_")) delete out[k];
  }
  return out;
}

/** One placement attempt of a worker (a GPU type in a region, or a CPU flavor). */
export interface Placement {
  region?: RegionId;
  dc?: string;
  gpu?: string;
  cpu?: string;
}
export function workerPlacements(spec: ClusterSpec, pool: PoolSpec): Placement[] {
  // Never place in a region without a weights volume (us since 2026-10), even from an old stored spec.
  const regions = (pool.regions?.length ? pool.regions : spec.regions).filter(regionAvailable);
  const out: Placement[] = [];
  if (pool.compute === "CPU") {
    for (const f of pool.cpu_flavors?.length ? pool.cpu_flavors : ["cpu3c", "cpu5c", "cpu3g"]) {
      for (const r of regions) out.push({ region: r, dc: REGIONS[r]!.dc, cpu: f });
      out.push({ cpu: f }); // any DC
    }
    return out;
  }
  for (const r of regions) for (const g of pool.gpu_types?.length ? pool.gpu_types : REGIONS[r]!.gpus) out.push({ region: r, dc: REGIONS[r]!.dc, gpu: g });
  return out;
}
/** The start command for a worker env: an edge front, a worker with the backstop watchdog (standalone), or the plain worker. */
export function bootFor(env: Record<string, string>): string {
  if (env.FV_DISPATCH_FRONT === "1") return EDGE_WORKER_BOOT;
  return env.FV_CLUSTER_DEADLINE && env.FV_BACKSTOP_API_KEY ? WATCHDOG_WORKER_BOOT : WORKER_BOOT;
}
export function workerCreatePayload(name: string, image: string, pool: PoolSpec, pl: Placement, env: Record<string, string>) {
  const common = { name, imageName: image, dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [bootFor(env)], env };
  if (pool.compute === "CPU") {
    return {
      ...common,
      computeType: "CPU",
      cpuFlavorIds: [pl.cpu!],
      vcpuCount: pool.vcpu ?? 2,
      containerDiskInGb: pool.container_disk_gb ?? 10,
      ports: ["8000/http"],
      ...(pl.dc ? { dataCenterIds: [pl.dc] } : {}),
      ...(pool.volume && pl.region ? { networkVolumeId: REGIONS[pl.region]!.volume, volumeMountPath: "/workspace" } : {}),
    };
  }
  const vol = pool.volume !== false && pl.region ? { volumeInGb: 0, networkVolumeId: REGIONS[pl.region]!.volume, volumeMountPath: "/workspace" } : {};
  return {
    ...common,
    cloudType: "SECURE",
    computeType: "GPU",
    gpuTypeIds: [pl.gpu!],
    gpuCount: 1,
    containerDiskInGb: pool.container_disk_gb ?? 40,
    ...vol,
    dataCenterIds: [pl.dc!],
    ports: ["8000/http", "70000/tcp"],
  };
}

/** Deterministic JSON (sorted keys) for env hashes. */
export function canonical(o: Record<string, string>): string {
  return JSON.stringify(Object.keys(o).sort().map((k) => [k, o[k]]));
}
