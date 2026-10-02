// Pod payloads and env of a cluster: a port of scripts/serve/runpod-cluster.sh
// (create_gateway, create_worker, patch_gateway, GATEWAY_BOOT, WORKER_BOOT).
import { GATEWAY_BASE_PODS } from "./gateway-base";
import { REGIONS, type ClusterSpec, type PoolSpec, type RegionId } from "./spec";

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

// The gateway pod's start command, byte for byte the script's: the config
// from the env, the watchdog (deadline + balance floor, deletes the workers
// and itself with FV_BACKSTOP_API_KEY) in the background, fv-serve in front.
export const GATEWAY_BOOT = `set -u
mkdir -p /fvstate
printf "%s" "$FV_GATEWAY_TOML_B64" | base64 -d > /fv-gateway.toml
export FV_PUBLIC_BASE_URL="https://\${RUNPOD_POD_ID}-8000.proxy.runpod.net"
(
  command -v curl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl; } >/fvstate/watchdog-apt.log 2>&1
  api=https://rest.runpod.io/v1
  kill_all() {
    echo "[watchdog] $1: deleting \${FV_CLUSTER_PODS:-} and \${RUNPOD_POD_ID}" >&2
    for p in \${FV_CLUSTER_PODS:-} "$RUNPOD_POD_ID"; do
      for _ in 1 2 3; do
        curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$api/pods/$p" >/dev/null && break
        sleep 5
      done
    done
  }
  n=0
  while :; do
    if [ "$(date +%s)" -ge "$FV_CLUSTER_DEADLINE" ]; then kill_all deadline; sleep 60; continue; fi
    if [ $((n % 2)) -eq 0 ]; then
      b=$(curl -sS --max-time 20 -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" -H "content-type: application/json" \\
        https://api.runpod.io/graphql -d "{\\"query\\":\\"{ myself { clientBalance } }\\"}" | sed -n "s/.*\\"clientBalance\\":\\([0-9.]*\\).*/\\1/p")
      if [ -n "$b" ] && awk -v b="$b" -v m="$FV_MIN_BALANCE" "BEGIN{exit !(b+0 < m+0)}"; then kill_all "balance $b below $FV_MIN_BALANCE"; fi
    fi
    n=$((n + 1))
    sleep 30
  done
) &
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-gateway.toml`;

// A worker's start command: the script's, plus an inline config
// (FV_WORKER_TOML_B64) instead of a file in the image when the pool has one.
export const WORKER_BOOT = `set -u
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

/** The gateway config without the Reactor model and fal apps (gateway images older than them). */
export const GATEWAY_BASE_MINIMAL = GATEWAY_BASE_PODS.replace(/^reactor_model = .*\n/m, "")
  .replace(/^fal_director = true$/m, "fal_director = false")
  .replace(/^reactor = true$/m, "reactor = false")
  .replace(/^# Every worker config's fal apps[\s\S]*?(?=^fal_apps = )/m, "")
  .replace(/^fal_apps = .*\n/m, "");

/** Keys newer than released gateway images (release 1 = 2cd1ba0; serde
 * denies unknown fields). Their values in the base equal the gateway's
 * defaults, so leaving them out is behaviour-neutral on newer images. */
const stripNewGatewayKeys = (t: string) =>
  t.replace(/^(inline_inputs_max_bytes|input_passthrough|stage_inputs_for_retry) = .*\n/gm, "");

const tomlStr = (s: string) => JSON.stringify(s);

/** The body lines of `[section]` in a flat TOML text: [first, end) line indices, or null. */
function sectionRange(lines: string[], section: string): [number, number] | null {
  const head = lines.findIndex((l) => l.trim() === `[${section}]`);
  if (head < 0) return null;
  let end = head + 1;
  while (end < lines.length && !/^\s*\[/.test(lines[end]!)) end++;
  while (end > head + 1 && lines[end - 1]!.trim() === "") end--; // keep the blank line before the next section
  return [head + 1, end];
}
/** Sets (or, with null, removes) `key = value` in `[section]`; adds the section when it is missing. */
export function tomlSet(text: string, section: string, key: string, value: string | null): string {
  const lines = text.split("\n");
  const r = sectionRange(lines, section);
  if (!r) return value === null ? text : `${text.replace(/\n*$/, "\n")}\n[${section}]\n${key} = ${value}\n`;
  const re = new RegExp(`^\\s*${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*=`);
  const i = lines.slice(r[0], r[1]).findIndex((l) => re.test(l));
  if (i >= 0) {
    if (value === null) lines.splice(r[0] + i, 1);
    else lines[r[0] + i] = `${key} = ${value}`;
  } else if (value !== null) lines.splice(r[1], 0, `${key} = ${value}`);
  return lines.join("\n");
}
/** Replaces the body of `[section]` (adds the section when it is missing). */
function tomlReplaceSection(text: string, section: string, body: string[]): string {
  const lines = text.split("\n");
  const r = sectionRange(lines, section);
  if (!r) return `${text.replace(/\n*$/, "\n")}\n[${section}]\n${body.join("\n")}\n`;
  lines.splice(r[0], r[1] - r[0], ...body);
  return lines.join("\n");
}

/** The gateway's Reactor model: the spec's, else the base's (fasth3) when a pool serves it, else a pool's causal (SF-Wan / LongLive) model; undefined: leave the base as it is. */
export function reactorModel(spec: ClusterSpec): string | null | undefined {
  if (spec.gateway.reactor_model !== undefined) return spec.gateway.reactor_model;
  if (spec.gateway.base === "minimal") return undefined;
  const models = spec.pools.flatMap((p) => p.models || []);
  if (models.some((m) => m.id === "fasth3")) return undefined;
  return models.find((m) => m.family === "wan" && m.recipe === "sfwan21-1.3b")?.id;
}

/** The base with the spec's gateway overrides (fal apps, protocols, Reactor model, aliases). */
export function gatewayBase(spec: ClusterSpec): string {
  let t = stripNewGatewayKeys(spec.gateway.base === "minimal" ? GATEWAY_BASE_MINIMAL : GATEWAY_BASE_PODS);
  const g = spec.gateway;
  if (g.fal_apps) t = tomlSet(t, "protocols", "fal_apps", `[${g.fal_apps.map(tomlStr).join(", ")}]`);
  for (const [k, v] of Object.entries(g.protocols || {})) if (typeof v === "boolean") t = tomlSet(t, "protocols", k, String(v));
  const rm = reactorModel(spec);
  if (rm !== undefined) t = tomlSet(t, "gateway", "reactor_model", rm === null ? null : tomlStr(rm));
  if (g.aliases) t = tomlReplaceSection(t, "aliases", Object.entries(g.aliases).map(([a, m]) => `${tomlStr(a)} = ${tomlStr(m)}`));
  return t;
}

export function gatewayToml(spec: ClusterSpec): string {
  let t = gatewayBase(spec);
  if (!t.endsWith("\n")) t += "\n";
  for (const p of spec.pools) {
    t += `\n[[pools]]\nid = ${tomlStr(p.id)}\nkind = "pod"\nurls = []\nmax_queued = ${p.max_queued ?? 32}\ndispatch_timeout_s = 30\njob_timeout_s = ${p.job_timeout_s ?? 1800}\nstale_after_s = ${p.stale_after_s ?? 120}\nretries = 1\n`;
    if (p.fake_models?.length) t += `fake_models = [${p.fake_models.map(tomlStr).join(", ")}]\n`;
    for (const m of p.models || []) t += `[[pools.models]]\nid = ${tomlStr(m.id)}\nfamily = ${tomlStr(m.family)}\nrecipe = ${tomlStr(m.recipe)}\n`;
  }
  t += "\n[autoscale]\nenabled = false\n";
  return t;
}

export const poolUrlsKey = (pool: string) => `FV_POOL_${pool.toUpperCase().replace(/-/g, "_")}_URLS`;

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
  image?: string; // the all-in-one image (script) when there is one
  images: Record<string, string>; // gateway + per pool
  gateway?: PodRec;
  gateway_url?: string;
  gateway_stopped?: boolean;
  workers: Record<string, PodRec[]>;
  rolling?: Record<string, PodRec[]>;
  retired?: PodRec[];
}
export interface ClusterSecrets {
  internal_token: string;
  url_signing_key: string;
  admin_recipient?: string; // X25519 public (raw, b64)
  admin_private?: string; // X25519 private (pkcs8, b64)
  admin_token?: string; // opened from the gateway (or a legacy FV_ADMIN_TOKEN)
  legacy_admin_token?: boolean; // imported state that passes FV_ADMIN_TOKEN itself
  ingest_token?: string;
  smoke_api_key?: string;
}

/** Keys the controller owns: user env layers may not set them. */
export const RESERVED_KEYS = new Set([
  "FV_INTERNAL_TOKEN",
  "FV_URL_SIGNING_KEY",
  "FV_BACKSTOP_API_KEY",
  "FV_CLUSTER_DEADLINE",
  "FV_CLUSTER_PODS",
  "FV_MIN_BALANCE",
  "FV_ADMIN_TOKEN",
  "FV_ADMIN_TOKEN_RECIPIENT",
  "FV_GATEWAY_TOML_B64",
  "FV_WORKER_TOML_B64",
  "FV_WORKER_CONFIG",
  "FV_SERVE_ROLE",
  "FV_PUBLIC_BASE_URL",
  "FV_LOG_SHIP_URL",
  "FV_LOG_SHIP_TOKEN",
  "FV_GITHUB_TOKEN",
  "FV_IMAGE_REF",
  "FV_IMAGE_DIGEST",
]);
export const isReserved = (k: string) => RESERVED_KEYS.has(k) || /^FV_POOL_[A-Z0-9_]+_URLS$/.test(k);
/** System env keys whose values are secret (masked in every view). */
export const SECRET_SYSTEM_KEYS = new Set(["FV_INTERNAL_TOKEN", "FV_URL_SIGNING_KEY", "FV_BACKSTOP_API_KEY", "FV_ADMIN_TOKEN", "FV_LOG_SHIP_TOKEN", "FV_GITHUB_TOKEN"]);

export interface EnvCtx {
  spec: ClusterSpec;
  state: ClusterState;
  secrets: ClusterSecrets;
  deadlineMs: number;
  runpodApiKey: string;
  githubPat?: string;
  ingestUrl?: string;
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

/** Every pod the gateway watchdog deletes besides itself (FV_CLUSTER_PODS). */
export function workerPodIds(state: ClusterState): string[] {
  const out: string[] = [];
  for (const l of Object.values(state.workers || {})) for (const r of l) out.push(r.pod);
  for (const l of Object.values(state.rolling || {})) for (const r of l) out.push(r.pod);
  for (const r of state.retired || []) out.push(r.pod);
  return out;
}

/** The gateway's system env (create_gateway + patch_gateway). */
export function gatewaySystemEnv(ctx: EnvCtx, image: string): Record<string, string> {
  const { spec, state, secrets } = ctx;
  const e: Record<string, string> = {
    ...SECRET_ENV_REFS,
    ...imageIdentEnv(image, spec.image.channel),
    FV_SERVE_MODE: "http",
    FV_STATE_DIR: "/fvstate",
    FV_AUTH_MODE: spec.gateway.auth,
    FV_INTERNAL_TOKEN: secrets.internal_token,
    FV_URL_SIGNING_KEY: secrets.url_signing_key,
    FV_GATEWAY_TOML_B64: b64utf8(gatewayToml(spec)),
    FV_CLUSTER_DEADLINE: String(Math.floor(ctx.deadlineMs / 1000)),
    FV_MIN_BALANCE: String(spec.min_balance),
    FV_BACKSTOP_API_KEY: ctx.runpodApiKey,
    FV_CLUSTER_PODS: workerPodIds(state).join(" "),
    RUST_LOG: "info",
  };
  if (secrets.admin_recipient && !secrets.legacy_admin_token) e.FV_ADMIN_TOKEN_RECIPIENT = secrets.admin_recipient;
  else if (secrets.admin_token) e.FV_ADMIN_TOKEN = secrets.admin_token;
  if (spec.gateway.github_token && ctx.githubPat) e.FV_GITHUB_TOKEN = ctx.githubPat;
  const pools = new Set([...Object.keys(state.workers || {}), ...Object.keys(state.rolling || {})]);
  for (const p of [...pools].sort()) {
    const urls = [...(state.workers?.[p] || []), ...(state.rolling?.[p] || [])].map((r) => r.url).filter(Boolean);
    if (urls.length) e[poolUrlsKey(p)] = urls.join(",");
  }
  return { ...e, ...logShipEnv(ctx) };
}

export function workerSystemEnv(ctx: EnvCtx, pool: PoolSpec, image: string): Record<string, string> {
  const e: Record<string, string> = {
    ...SECRET_ENV_REFS,
    ...imageIdentEnv(image, ctx.spec.image.channel),
    FV_SERVE_MODE: "http",
    FV_SERVE_ROLE: "worker",
    FV_INTERNAL_TOKEN: ctx.secrets.internal_token,
    FV_URL_SIGNING_KEY: ctx.secrets.url_signing_key,
    FV_PUBLIC_BASE_URL: ctx.state.gateway_url || "",
    FV_WORKER_CONFIG: pool.config || "/fv-worker.toml",
    FV_STATE_DIR: "/fvstate",
    FV_WEIGHTS: "/workspace/weights",
    FV_JOBS_HEARTBEAT_S: "10",
    RUST_LOG: "info",
  };
  if (pool.config_toml) e.FV_WORKER_TOML_B64 = b64utf8(pool.config_toml);
  return { ...e, ...logShipEnv(ctx) };
}

export function gatewayCreatePayload(name: string, image: string, flavor: string, vcpu: number, diskGb: number, dcs: string[] | null, env: Record<string, string>) {
  return {
    name,
    imageName: image,
    computeType: "CPU",
    cpuFlavorIds: [flavor],
    vcpuCount: vcpu,
    containerDiskInGb: diskGb,
    ports: ["8000/http"],
    dockerEntrypoint: ["bash", "-c"],
    dockerStartCmd: [GATEWAY_BOOT],
    env,
    ...(dcs ? { dataCenterIds: dcs } : {}),
  };
}

/** One placement attempt of a worker (a GPU type in a region, or a CPU flavor). */
export interface Placement {
  region?: RegionId;
  dc?: string;
  gpu?: string;
  cpu?: string;
}
export function workerPlacements(spec: ClusterSpec, pool: PoolSpec): Placement[] {
  const regions = pool.regions?.length ? pool.regions : spec.regions;
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
export function workerCreatePayload(name: string, image: string, pool: PoolSpec, pl: Placement, env: Record<string, string>) {
  const common = { name, imageName: image, dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [WORKER_BOOT], env };
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
