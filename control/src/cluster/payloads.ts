// Pod payloads and env of a cluster's workers: edge fronts
// (docs/serve/edge-control-plane.md §5.2) or direct workers
// (docs/control/gateway-less-auth.md). (The gateway pod is retired.)
import { isEdge, poolModelFamilies, REGIONS, regionAvailable, type ClusterSpec, type PoolSpec, type RegionId } from "./spec";
import type { Env } from "../env";
import { HUB_RUNPOD_SH, hubDiskGb, isOtherPool, weightsPlan, weightsSource } from "../providers";
import { DC_ANY, DEFAULT_ALLOWED_CUDA, noVolumePod } from "../enums";
import { R2_RUNPOD_SH, r2DiskGb, r2PodEnv } from "../r2weights";
import { DRIVER_CHECK_SH } from "../driver-check";
import { UPSAMPLER_START_SH, UPSAMPLER_WAIT_SH, upsamplerDiskGb, upsamplerPodEnv } from "../upsampler";
import { PREFER_FV_BOOT_SH } from "../fv-boot";

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
// inline (FV_WORKER_TOML_B64) when the pool has one. An image with fv-boot runs
// that instead (PREFER_FV_BOOT_SH, fv-boot.ts): the same steps from the same env.
export const WORKER_BOOT = `set -u
${PREFER_FV_BOOT_SH}echo "[fv-boot] start" >&2
FV_DRV_POD="\${RUNPOD_POD_ID:-}"
${DRIVER_CHECK_SH}if [ -d /workspace/weights ]; then echo "[fv-boot] volume: /workspace/weights ($(ls /workspace/weights 2>/dev/null | wc -l) trees)" >&2; else echo "[fv-boot] volume: /workspace/weights missing" >&2; fi
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
// FV_BACKSTOP_API_KEY is fv-control's own account key (envCtx runpodApiKey =
// RUNPOD_API_KEY, the key it creates and deletes pods with over REST), not
// the pod-scoped RUNPOD_API_KEY Runpod injects, which REST refuses with a 403
// (docs/adr/0023-lab-report-and-ci-bridge.md §4). Only a 2xx or a 404 (the pod
// is already gone) ends the attempts; any other answer is retried.
const WATCHDOG = `(
  command -v curl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl; } >/fvstate/watchdog-apt.log 2>&1
  bye() {
    echo "[watchdog] $1: deleting \${RUNPOD_POD_ID}" >&2
    for i in 1 2 3; do
      c=$(curl -sS --max-time 30 -o /dev/null -w "%{http_code}" -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID")
      case "$c" in 2??|404) break ;; esac
      echo "[watchdog] delete: HTTP \${c:-none}" >&2
      [ "$i" = 3 ] || sleep 5
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
  /** fv-control's settings (weights_source r2 on Runpod: the plan URL, the published trees). */
  env?: Env;
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
  const out = { ...e, ...directEnv(ctx), ...edgeEnv(ctx, pool), ...logShipEnv(ctx), ...upsamplerEnv(ctx, pool), ...runpodR2Env(ctx, pool), ...runpodHubEnv(ctx, pool) };
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

/** upsampler_local (upsampler.ts, docs/serve/upsampler-local.md): the sidecar's switch and what its fetch needs (the
 * plan URL, the fetcher, the cluster's ingest token), whatever the pool's weights source. GPU pools only. */
function upsamplerEnv(ctx: EnvCtx, pool: PoolSpec): Record<string, string> {
  if (!pool.upsampler_local || pool.compute === "CPU" || !ctx.env) return {};
  return { ...upsamplerPodEnv(ctx.env), ...(ctx.secrets.ingest_token ? { FV_ENDPOINT_REPORT_TOKEN: ctx.secrets.ingest_token } : {}) };
}

/** A Runpod worker with weights_source r2 (docs/ops/r2-weights.md): no volume; the boot fetches its pool's trees from
 * the R2 weights store onto the container disk (FV_R2_DISK_GB sizes it) with the cluster's ingest token, and the aux
 * files by URL. GMI / Brev pods get theirs in providers.ts providerEnv. */
function runpodR2Env(ctx: EnvCtx, pool: PoolSpec): Record<string, string> {
  if (isOtherPool(pool) || !ctx.env || weightsSource(pool, ctx.env) !== "r2") return {};
  const plan = weightsPlan(pool, ctx.env);
  return {
    FV_WEIGHTS_SOURCE: "r2",
    ...r2PodEnv(ctx.env),
    FV_ENDPOINT_REPORT_TOKEN: ctx.secrets.ingest_token || "",
    FV_WEIGHTS_TREES_B64: btoa(plan.tsv),
    FV_R2_DISK_GB: String(Math.min(500, Math.max(40, r2DiskGb(ctx.env, plan.trees) + (pool.upsampler_local ? upsamplerDiskGb(ctx.env) : 0)))),
  };
}

/** A Runpod worker with weights_source hub (owner-approved; providers.ts providerIssues checks it before any pod): no
 * volume; the boot (HUB_RUNPOD_SH) downloads the pool's trees from the Hub at their pinned revisions (weightsPlan:
 * weights-revisions.tsv) onto the container disk (FV_HUB_DISK_GB sizes it), with fetch-hub-tree.py at the image's commit. */
function runpodHubEnv(ctx: EnvCtx, pool: PoolSpec): Record<string, string> {
  if (isOtherPool(pool) || pool.compute === "CPU" || weightsSource(pool, ctx.env) !== "hub") return {};
  const plan = weightsPlan(pool, ctx.env);
  return {
    FV_WEIGHTS_SOURCE: "hub",
    FV_WEIGHTS_TREES_B64: btoa(plan.tsv),
    FV_SCRIPTS_URL: `https://raw.githubusercontent.com/${ctx.env?.GITHUB_REPO || "zaitrarrio/fastvideo-rs"}/${ctx.spec.image.sha || "main"}/scripts/gpu`,
    FV_HUB_DISK_GB: String(Math.min(500, Math.max(40, hubDiskGb(plan.trees) + (pool.upsampler_local && ctx.env ? upsamplerDiskGb(ctx.env) : 0)))),
  };
}

/** One placement attempt of a worker (a GPU type in a region, or a CPU flavor). */
export interface Placement {
  region?: RegionId;
  dc?: string;
  gpu?: string;
  cpu?: string;
}
/** A Runpod pool that mounts no weights volume, so it may run in any data centre: enums.ts noVolumePod (weights_source
 * r2 / hub / none, a CPU pool with volume off), or a GPU pool with weights_source unset whose default is r2
 * (providers.ts weightsSource: a preset whose trees are not on the EU volume). GMI / Brev pools: false (no Runpod DC). */
export function poolWithoutVolume(pool: PoolSpec, env?: Env): boolean {
  if (isOtherPool(pool)) return false;
  if (noVolumePod(pool)) return true;
  return pool.compute !== "CPU" && !pool.weights_source && pool.volume !== true && weightsSource(pool, env) !== "volume";
}
/** The data centres a Runpod pool without the volume tries, in order: its own data_centers, else the cluster's
 * (spec.data_centers); null: the regions' (the weights volume's data centre). */
export function poolDataCenters(spec: Pick<ClusterSpec, "data_centers">, pool: PoolSpec, env?: Env): string[] | null {
  if (isOtherPool(pool)) return null;
  if (pool.data_centers?.length && noVolumePod(pool)) return pool.data_centers;
  if (spec.data_centers?.length && poolWithoutVolume(pool, env)) return spec.data_centers;
  return null;
}
/** Where a pool's pods run, in words (the editor's outlook): its provider, its data centres, or its volume regions. */
export function placementWhere(spec: ClusterSpec, pool: PoolSpec, env?: Env): string {
  if (isOtherPool(pool)) return `${pool.provider}${pool.provider_region ? ` ${pool.provider_region}` : ""}`;
  const dcs = poolDataCenters(spec, pool, env);
  if (dcs) return dcs.map((d) => (d === DC_ANY ? "any data centre" : d)).join(", ");
  const regions = (pool.regions?.length ? pool.regions : spec.regions).filter(regionAvailable);
  const vol = pool.compute === "CPU" ? pool.volume === true : true;
  return regions.length ? regions.map((r) => `${REGIONS[r]!.dc}${vol ? ` (the ${r} weights volume)` : ""}`).join(", ") + (pool.compute === "CPU" && !vol ? ", then any data centre" : "") : "no region with a weights volume";
}
export function workerPlacements(spec: ClusterSpec, pool: PoolSpec, env?: Env): Placement[] {
  // Never place in a region without a weights volume (us since 2026-10), even from an old stored spec.
  const regions = (pool.regions?.length ? pool.regions : spec.regions).filter(regionAvailable);
  const out: Placement[] = [];
  // A pod without the weights volume (GPU: weights_source r2, hub or none; CPU: volume off) with data_centers (its own,
  // else the cluster's): those data centres, in order (each with every GPU type / flavor in order), instead of the
  // regions'; "any" = no dataCenterIds. A pool that mounts the volume always runs in its regions.
  const dcs = poolDataCenters(spec, pool, env);
  const at = (dc: string) => (dc === DC_ANY ? {} : { dc });
  if (pool.compute === "CPU" && dcs) {
    for (const f of pool.cpu_flavors?.length ? pool.cpu_flavors : ["cpu3c", "cpu5c", "cpu3g"]) for (const dc of dcs) out.push({ ...at(dc), cpu: f });
    return out;
  }
  if (pool.compute === "CPU") {
    for (const f of pool.cpu_flavors?.length ? pool.cpu_flavors : ["cpu3c", "cpu5c", "cpu3g"]) {
      for (const r of regions) out.push({ region: r, dc: REGIONS[r]!.dc, cpu: f });
      out.push({ cpu: f }); // any DC
    }
    return out;
  }
  if (dcs) {
    for (const dc of dcs) for (const g of pool.gpu_types?.length ? pool.gpu_types : REGIONS.eu.gpus) out.push({ ...at(dc), gpu: g });
    return out;
  }
  for (const r of regions) for (const g of pool.gpu_types?.length ? pool.gpu_types : REGIONS[r]!.gpus) out.push({ region: r, dc: REGIONS[r]!.dc, gpu: g });
  return out;
}
/** A GPU pod's host filter (allowedCudaVersions): the pool's, else enums.ts DEFAULT_ALLOWED_CUDA in every data centre
 * (none while Runpod's REST enum cannot name the 13.1 / 13.2 hosts: a "13.0"-only list hid every Secure RTX PRO 6000).
 * The boot's driver check (DRIVER_CHECK_SH) refuses a host older than CUDA 13 and fv-control places a new pod. */
export function podAllowedCuda(pool: PoolSpec): string[] {
  return [...(pool.allowed_cuda ?? DEFAULT_ALLOWED_CUDA)];
}
/** The start command for a worker env: an edge front, a worker with the backstop watchdog (standalone), or the plain worker. */
export function bootFor(env: Record<string, string>): string {
  const boot = env.FV_DISPATCH_FRONT === "1" ? EDGE_WORKER_BOOT : env.FV_CLUSTER_DEADLINE && env.FV_BACKSTOP_API_KEY ? WATCHDOG_WORKER_BOOT : WORKER_BOOT;
  // upsampler_local: the sidecar starts before the weights (its fetch and load overlap theirs) and is awaited just
  // before fv-serve, which then gets FV_UPSAMPLER_BASE_URL (upsampler.ts).
  const ups = env.FV_UPSAMPLER_LOCAL === "1";
  const b = ups ? boot.replace("mkdir -p /fvstate\n", `mkdir -p /fvstate\n${UPSAMPLER_START_SH}`).replace(EXEC_SERVE, `${UPSAMPLER_WAIT_SH}${EXEC_SERVE}`) : boot;
  // weights_source r2: the trees from the R2 weights store before anything else (the volume is not mounted).
  // weights_source hub: the trees from the Hub (owner-approved) before anything else, the same way.
  const fetch = env.FV_WEIGHTS_SOURCE === "r2" ? R2_RUNPOD_SH() : env.FV_WEIGHTS_SOURCE === "hub" ? HUB_RUNPOD_SH : "";
  return fetch ? b.replace(`mkdir -p /fvstate\n${ups ? UPSAMPLER_START_SH : ""}`, `mkdir -p /fvstate\n${ups ? UPSAMPLER_START_SH : ""}${fetch}`) : b;
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
  const r2 = env.FV_WEIGHTS_SOURCE === "r2";
  const hub = env.FV_WEIGHTS_SOURCE === "hub";
  // No network volume: say volumeInGb 0, or Runpod adds its default pod volume at /workspace and the r2 / hub fetch
  // into /workspace/weights lands on that small quota (EDQUOT) instead of the container disk sized for the trees.
  const vol = pool.volume !== false && !r2 && !hub && pl.region ? { volumeInGb: 0, networkVolumeId: REGIONS[pl.region]!.volume, volumeMountPath: "/workspace" } : { volumeInGb: 0 };
  const cuda = podAllowedCuda(pool);
  return {
    ...common,
    cloudType: "SECURE",
    computeType: "GPU",
    gpuTypeIds: [pl.gpu!],
    gpuCount: 1,
    containerDiskInGb: pool.container_disk_gb ?? (r2 ? Number(env.FV_R2_DISK_GB || 40) : hub ? Number(env.FV_HUB_DISK_GB || 40) : 40),
    ...vol,
    ...(pl.dc ? { dataCenterIds: [pl.dc] } : {}),
    ...(cuda.length ? { allowedCudaVersions: cuda } : {}),
    ports: ["8000/http", "70000/tcp"],
  };
}

/** Deterministic JSON (sorted keys) for env hashes. */
export function canonical(o: Record<string, string>): string {
  return JSON.stringify(Object.keys(o).sort().map((k) => [k, o[k]]));
}
