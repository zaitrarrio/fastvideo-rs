import { CUDA_VERSIONS } from "../enums";
// Runpod payloads of a serverless endpoint (docs/control/serverless.md),
// the same shapes scripts/serve/runpod-endpoint.sh and runpod-templates.sh
// proved: a serverless template (REST v1 POST /templates) plus a queue
// endpoint (REST v1 POST /endpoints), or a load-balancer endpoint (REST v2
// POST /serverless, the only API with `type`). Secrets are Runpod secret
// references, never values.
import { imageIdentEnv, SECRET_ENV_REFS } from "../cluster/payloads";
import type { EndpointSpec } from "./spec";

/** Every Runpod endpoint and template fv-control creates is named fvc-<name>…; nothing else is ever touched. */
export const SLS_PREFIX = "fvc-";
export const endpointName = (spec: Pick<EndpointSpec, "name">) => `${SLS_PREFIX}${spec.name}`;
/** Template names are unique per account: one per create (and per recreate). */
export const templateName = (spec: Pick<EndpointSpec, "name">, at: number) => `${SLS_PREFIX}${spec.name}-${new Date(at).toISOString().replace(/[-:T]/g, "").slice(2, 14)}`;

/** The worker's start when the spec names a config: the weights link (a serverless worker mounts the
 * volume at /runpod-volume; its HF-cache trees link into /workspace/weights) and an inline config, then
 * fv-serve. Without a config the image's own entrypoint (fv-entry) and baked FV_CONFIG run. */
export const SLS_BOOT = `set -eu
if [ -d /runpod-volume/weights ] && [ ! -e /workspace/weights ]; then mkdir -p /workspace && ln -s /runpod-volume/weights /workspace/weights; fi
mkdir -p /fvstate
cfg="\${FV_CONFIG:-}"
if [ -n "\${FV_WORKER_TOML_B64:-}" ]; then printf "%s" "$FV_WORKER_TOML_B64" | base64 -d > /fv-worker.toml; cfg=/fv-worker.toml; fi
if [ -n "$cfg" ]; then exec /opt/fastvideo-rs/bin/fv-serve --config "$cfg"; fi
exec /opt/fastvideo-rs/bin/fv-serve`;

function b64utf8(s: string): string {
  const u = new TextEncoder().encode(s);
  let bin = "";
  for (const x of u) bin += String.fromCharCode(x);
  return btoa(bin);
}

/** The workers' env: secret references, the image's identity, the serve mode, the spec's env last. */
export function workerEnv(spec: EndpointSpec, image: string): Record<string, string> {
  const e: Record<string, string> = {
    ...SECRET_ENV_REFS,
    ...imageIdentEnv(image, spec.image.channel),
    FV_SERVE_MODE: spec.mode === "queue" ? "runpod-queue" : "http",
    // Runpod's API key guards /run and the LB URL; fv-serve trusts what reaches it.
    FV_AUTH_MODE: "trust-gateway",
    FV_STATE_DIR: "/fvstate",
    FV_CACHE_DIR: "/fvstate/cache",
    FV_CONTROL_ENDPOINT: spec.name,
    RUST_LOG: "info",
  };
  if (spec.network_volume) e.FV_WEIGHTS = "/runpod-volume/weights";
  if (spec.config) e.FV_CONFIG = spec.config;
  if (spec.config_toml) e.FV_WORKER_TOML_B64 = b64utf8(spec.config_toml);
  if (spec.mode === "lb") Object.assign(e, { PORT: "8000", PORT_HEALTH: "8000", FV_WORKERS_MAX: String(Math.max(1, spec.workers_max)) });
  return { ...e, ...(spec.env || {}) };
}

const bootFields = (spec: EndpointSpec) => (spec.config || spec.config_toml ? { dockerEntrypoint: ["/bin/sh", "-c", SLS_BOOT], dockerStartCmd: [] as string[] } : {});

/** REST v1 POST /templates (queue endpoints). */
export function templateCreatePayload(spec: EndpointSpec, image: string, name: string) {
  return {
    name,
    imageName: image,
    isServerless: true,
    category: spec.compute === "CPU" ? "CPU" : "NVIDIA",
    containerDiskInGb: spec.container_disk_gb,
    volumeInGb: 0,
    env: workerEnv(spec, image),
    ...bootFields(spec),
    readme: `fv-control serverless endpoint ${spec.name} (${spec.variant}); docs/control/serverless.md`,
  };
}
/** REST v1 PATCH /templates/<id>: image, env, start and disk (a rolling release of the endpoint's workers). */
export function templateUpdatePayload(spec: EndpointSpec, image: string) {
  const { name: _n, isServerless: _s, category: _c, readme: _r, ...rest } = templateCreatePayload(spec, image, "");
  // Without a config the template goes back to the image's own entrypoint.
  return { ...rest, ...(spec.config || spec.config_toml ? {} : { dockerEntrypoint: [], dockerStartCmd: [] }) };
}

/** The scaling and placement fields REST v1 takes on create and PATCH. */
export function endpointFields(spec: EndpointSpec) {
  const common = {
    workersMin: spec.workers_min,
    workersMax: spec.workers_max,
    idleTimeout: spec.idle_timeout_s,
    flashboot: spec.flashboot,
    executionTimeoutMs: spec.execution_timeout_s * 1000,
    scalerType: spec.scaler_type,
    scalerValue: spec.scaler_value,
    ...(spec.data_centers?.length ? { dataCenterIds: spec.data_centers } : {}),
  };
  if (spec.compute === "CPU") return { ...common, computeType: "CPU" as const, cpuFlavorIds: spec.cpu_flavors || ["cpu3c", "cpu5c"], vcpuCount: spec.vcpu ?? 2 };
  return {
    ...common,
    computeType: "GPU" as const,
    gpuTypeIds: spec.gpu_types || [],
    gpuCount: spec.gpu_count ?? 1,
    ...((spec.allowed_cuda ?? []).length ? { allowedCudaVersions: spec.allowed_cuda } : {}),
  };
}

/** REST v1 POST /endpoints (a queue endpoint on a template). */
export function endpointCreatePayload(spec: EndpointSpec, templateId: string) {
  return { name: endpointName(spec), templateId, ...endpointFields(spec), ...(spec.network_volume ? { networkVolumeId: spec.network_volume } : {}) };
}

/** REST v1 PATCH /endpoints/<id> (EndpointUpdateInput; also scales a v2-made endpoint): scaling, timeouts,
 * placement. Never computeType (not an update field: Runpod refuses "extra input keys"); a CPU endpoint's
 * flavors and vCPUs are v2 fields, changed only by a new endpoint. */
export function endpointUpdatePayload(spec: EndpointSpec) {
  const { computeType: _c, ...f } = endpointFields(spec) as Record<string, unknown>;
  // An empty allowed_cuda must also clear a filter set earlier: PATCH with every version Runpod knows.
  if (spec.compute === "GPU" && !(spec.allowed_cuda ?? []).length) f.allowedCudaVersions = [...CUDA_VERSIONS];
  if (spec.compute === "CPU") {
    delete f.cpuFlavorIds;
    delete f.vcpuCount;
  }
  return f;
}

/** REST v2 POST /serverless: a load-balancer endpoint (GPU pools from Runpod's catalog), or a CPU queue
 * endpoint (REST v1 creates every endpoint with GPU workers: it ignores computeType CPU, live 2026-10-06).
 * v2 makes the endpoint's template itself (Runpod deletes it with the endpoint). */
export function v2CreatePayload(spec: EndpointSpec, image: string, pools: string[] = []) {
  const lb = spec.mode === "lb";
  return {
    name: endpointName(spec),
    type: lb ? "LOAD_BALANCER" : "QUEUE",
    image,
    // The image's entrypoint (fv-entry: the weights link) passes these to fv-serve (runpod-endpoint.sh up-lb).
    ...(lb && spec.config ? { args: `--config ${spec.config}` } : {}),
    ...(lb ? { ports: ["8000/http"] } : {}),
    disk: spec.container_disk_gb,
    env: workerEnv(spec, image),
    ...(spec.compute === "CPU"
      ? { cpu: (spec.cpu_flavors || ["cpu3c", "cpu5c"]).map((id) => ({ id, vcpuCount: spec.vcpu ?? 2 })) }
      : { gpu: { pools: pools.length ? pools : ["BLACKWELL_96"], count: spec.gpu_count ?? 1, ...((spec.allowed_cuda ?? []).length ? { allowedCudaVersions: spec.allowed_cuda } : {}) } }),
    workers: { min: spec.workers_min, max: spec.workers_max, idleTimeout: spec.idle_timeout_s },
    scaling: spec.scaler_type === "REQUEST_COUNT" ? { type: "REQUEST_COUNT", requestCount: spec.scaler_value } : { type: "QUEUE_DELAY", queueDelay: spec.scaler_value },
    ...(spec.network_volume ? { networkVolumes: [spec.network_volume] } : {}),
    ...(spec.data_centers?.length ? { dataCenterIds: spec.data_centers } : {}),
    flashboot: spec.flashboot ? "FLASHBOOT" : "OFF",
    timeout: spec.execution_timeout_s * 1000,
  };
}
/** What a v2-made queue endpoint's template gets after create (REST v1 PATCH /templates): the boot of a named config. */
export function v2TemplateBoot(spec: EndpointSpec) {
  return spec.mode === "queue" && (spec.config || spec.config_toml) ? { dockerEntrypoint: ["/bin/sh", "-c", SLS_BOOT], dockerStartCmd: [] as string[] } : null;
}
/** Runpod catalog pools for the spec's GPU types, in the spec's order (unknown types are dropped). */
export function lbPools(gpuTypes: string[], catalog: { id: string; pool: string | null }[]): string[] {
  const out: string[] = [];
  for (const g of gpuTypes) {
    const p = catalog.find((c) => c.id === g)?.pool;
    if (p && !out.includes(p)) out.push(p);
  }
  return out;
}

/** The queue job body of a test invoke (the native envelope; docs/serve/e2e/serverless.md). */
export function invokeBody(input: unknown, executionTimeoutS: number) {
  return { input: input ?? { kind: "info" }, policy: { executionTimeout: executionTimeoutS * 1000 } };
}

/** A Runpod endpoint (REST v1 GET) reduced to what fv-control shows: never the template's env (secrets in clear). */
export function endpointView(e: any) {
  if (!e || typeof e !== "object") return null;
  return {
    id: e.id,
    name: e.name,
    templateId: e.templateId ?? e.template?.id ?? null,
    computeType: e.computeType ?? null,
    gpuTypeIds: e.gpuTypeIds ?? null,
    instanceIds: e.instanceIds ?? null,
    dataCenterIds: e.dataCenterIds ?? null,
    networkVolumeId: e.networkVolumeId ?? null,
    workersMin: e.workersMin,
    workersMax: e.workersMax,
    idleTimeout: e.idleTimeout,
    flashboot: e.flashboot,
    executionTimeoutMs: e.executionTimeoutMs,
    scalerType: e.scalerType,
    scalerValue: e.scalerValue,
    version: e.version,
    image: e.template?.imageName ?? null,
    workers: Array.isArray(e.workers) ? e.workers.map((w: any) => ({ id: w.id, desiredStatus: w.desiredStatus, costPerHr: w.costPerHr ?? w.adjustedCostPerHr ?? null, machineId: w.machineId ?? null, gpu: w.machine?.gpuTypeId ?? w.gpu?.id ?? null, dc: w.machine?.dataCenterId ?? null, lastStartedAt: w.lastStartedAt ?? null })) : undefined,
  };
}
