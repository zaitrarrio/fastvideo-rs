import { imageOf, MIN_CUDA } from "../enums";
// Runpod payloads of a serverless endpoint (docs/control/serverless.md),
// the same shapes scripts/serve/runpod-endpoint.sh and runpod-templates.sh
// proved: a serverless template (REST v1 POST /templates) plus a queue
// endpoint (REST v1 POST /endpoints), or a load-balancer endpoint (REST v2
// POST /serverless, the only API with `type`). Secrets are Runpod secret
// references, never values.
import { imageIdentEnv, SECRET_ENV_REFS } from "../cluster/payloads";
import type { EndpointSpec } from "./spec";
import { SLS_HUB_SH, SLS_R2_SH } from "./weights";
import { DRIVER_CHECK_SH } from "../driver-check";
import { UPSAMPLER_START_SH, UPSAMPLER_WAIT_SH } from "../upsampler";
import { PREFER_FV_BOOT_SH } from "../fv-boot";
import { workerConfigEnv } from "../worker-config";
import { checkLaunch, LAUNCH_SCHEMA, runpodEndpointBody, runpodTemplateBody, SLS_VOLUME_MOUNT, type Guards, type Launch } from "../launch";

/** Every Runpod endpoint and template fv-control creates is named fvc-<name>…; nothing else is ever touched. */
export const SLS_PREFIX = "fvc-";
export const endpointName = (spec: Pick<EndpointSpec, "name">) => `${SLS_PREFIX}${spec.name}`;
/** Template names are unique per account: one per create (and per recreate). */
export const templateName = (spec: Pick<EndpointSpec, "name">, at: number) => `${SLS_PREFIX}${spec.name}-${new Date(at).toISOString().replace(/[-:T]/g, "").slice(2, 14)}`;

/** The worker's start: the host's driver (driver-check.ts), the weights link (a serverless worker mounts the volume at
 * /runpod-volume; its HF-cache trees link into /workspace/weights) and the config (FV_WORKER_TOML_B64, which every
 * endpoint sends: the images bake none, docs/adr/0011-deploy-section.md; or FV_WORKER_CONFIG, a file in the image),
 * then fv-serve. Without a config it stops, as fv-boot does. An image with fv-boot runs that instead of this script
 * (PREFER_FV_BOOT_SH, fv-boot.ts). */
export const SLS_BOOT = `set -eu
${PREFER_FV_BOOT_SH}FV_DRV_POD="\${RUNPOD_POD_ID:-}"
${DRIVER_CHECK_SH}if [ -d /runpod-volume/weights ] && [ ! -e /workspace/weights ]; then mkdir -p /workspace && ln -s /runpod-volume/weights /workspace/weights; fi
mkdir -p /fvstate
${UPSAMPLER_START_SH}cfg="\${FV_WORKER_CONFIG:-}"
if [ -n "\${FV_WORKER_TOML_B64:-}" ]; then printf "%s" "$FV_WORKER_TOML_B64" | base64 -d > /fv-worker.toml; cfg=/fv-worker.toml; fi
if [ -z "$cfg" ]; then echo "[fv-boot] FAILED: no worker config: the image serves no default; send FV_WORKER_TOML_B64 or FV_WORKER_CONFIG" >&2; exit 1; fi
${UPSAMPLER_WAIT_SH}exec /opt/fastvideo-rs/bin/fv-serve --config "$cfg"`;

/** The start of a weights_source r2 worker (bash: the fetch, then SLS_BOOT's config handling and fv-serve). */
export const SLS_R2_BOOT = SLS_BOOT.replace(`mkdir -p /fvstate\n${UPSAMPLER_START_SH}`, `mkdir -p /fvstate\n${UPSAMPLER_START_SH}${SLS_R2_SH}`);
const isR2 = (spec: EndpointSpec) => spec.weights_source === "r2" && spec.compute === "GPU";
/** The start of a weights_source hub worker (bash: the Hub download and its checks, then SLS_BOOT's config handling and fv-serve). */
export const SLS_HUB_BOOT = SLS_BOOT.replace(`mkdir -p /fvstate\n${UPSAMPLER_START_SH}`, `mkdir -p /fvstate\n${UPSAMPLER_START_SH}${SLS_HUB_SH}`);
const isHub = (spec: EndpointSpec) => spec.weights_source === "hub" && spec.compute === "GPU";
/** A worker that downloads its weights at boot (r2 or hub): its own bash start, whatever the config. */
const bashBoot = (spec: EndpointSpec) => (isR2(spec) ? SLS_R2_BOOT : isHub(spec) ? SLS_HUB_BOOT : null);

/** The workers' env: secret references, the image's identity, the serve mode, the worker config (FV_WORKER_TOML_B64:
 * the spec's inline config or its named one's text; the images bake none), weights_source r2's or hub's (`weights`:
 * serverless/weights.ts slsWeightsEnv / slsHubEnv), the spec's env last. */
export function workerEnv(spec: EndpointSpec, image: string, weights: Record<string, string> = {}): Record<string, string> {
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
  Object.assign(e, workerConfigEnv(spec));
  if (spec.mode === "lb") Object.assign(e, { PORT: "8000", PORT_HEALTH: "8000", FV_WORKERS_MAX: String(Math.max(1, spec.workers_max)) });
  // weights_source r2's env, and upsampler_local's (the sidecar's switch and its fetch, any weights source).
  if (isR2(spec) || isHub(spec) || (spec.upsampler_local && spec.compute === "GPU")) Object.assign(e, weights);
  return { ...e, ...(spec.env || {}) };
}

// Every endpoint sends its config (the images bake none): r2 / hub start with their fetch (bash), the rest with SLS_BOOT.
const bootFields = (spec: EndpointSpec) => {
  const bash = bashBoot(spec);
  return bash ? { dockerEntrypoint: ["bash", "-c", bash], dockerStartCmd: [] as string[] } : { dockerEntrypoint: ["/bin/sh", "-c", SLS_BOOT], dockerStartCmd: [] as string[] };
};

/** The rules fv-control holds a serverless endpoint to, as a Launch's guards (ADR-0018 §7.2). Runpod reports no
 * serverless price and fv-control checks none (fv-deploy's known gap, ADR-0018 §5.3), nor a worker's readiness
 * (Runpod replaces workers): 0 there means "not checked by fv-control". The floor: create, scale-up and invoke need
 * the account floor plus the serverless margin (ops.ts assertFloor; the $8 floor of CLAUDE.md is its least). The cap:
 * deadline_min (the cron's backstop: scale to 0 or delete), 0 without one. */
export function endpointGuards(spec: Pick<EndpointSpec, "deadline_min">): Guards {
  return { max_usd_per_hr: 0, min_balance_usd: 8, cap_s: (spec.deadline_min ?? 0) * 60, ready_timeout_s: 0, ready_cycles: 0 };
}

/** A GPU endpoint as fv-deploy's Launch (plan C3c, docs/adr/0018-fv-deploy.md §7), without its template's image and
 * env: what the endpoint itself carries (its GPU types and count, the CUDA minimum, the volume and data centres, the
 * scaling). `endpointLaunch` adds the template's part. */
function endpointLaunchBase(spec: EndpointSpec, guards: Guards): Launch {
  if (spec.compute === "CPU") throw new Error(`endpoint ${spec.name}: a CPU endpoint is not a Launch (fv-deploy's schema has no CPU compute)`);
  const ws = spec.weights_source ?? (spec.network_volume ? "volume" : "none");
  return {
    schema: LAUNCH_SCHEMA,
    name: endpointName(spec),
    provider: "runpod-serverless",
    image: "",
    env: {},
    gpu: {
      types: spec.gpu_types || [],
      count: spec.gpu_count ?? 1,
      // "At least CUDA 13.0": Runpod applies minCudaVersion as a minimum, so it keeps the 13.1 / 13.2 hosts (which REST
      // cannot name in allowedCudaVersions) and drops the pre-13 ones (enums.ts MIN_CUDA, DEFAULT_ALLOWED_CUDA).
      min_cuda: MIN_CUDA,
      allowed_cuda: [...(spec.allowed_cuda ?? [])],
    },
    placement: { data_centers: [...(spec.data_centers ?? [])], volume: spec.network_volume ? { id: spec.network_volume, mount_path: SLS_VOLUME_MOUNT } : null, cloud: "SECURE" },
    disk: { container_gb: spec.container_disk_gb, volume_gb: 0 },
    weights: ws,
    guards,
    serverless: {
      workers_min: spec.workers_min,
      workers_max: spec.workers_max,
      idle_timeout_s: spec.idle_timeout_s,
      flashboot: spec.flashboot,
      execution_timeout_ms: spec.execution_timeout_s * 1000,
      scaler_type: spec.scaler_type,
      scaler_value: spec.scaler_value,
    },
  };
}

/** A GPU endpoint's Launch: its template's image, start (the boot as the entrypoint, no start command) and env, and
 * the endpoint's scaling and placement; checked against the schema. Queue and load-balancer endpoints alike: an LB
 * endpoint's template and REST v1 fields are the same, only its create is the REST v2 envelope (v2CreatePayload). */
export function endpointLaunch(spec: EndpointSpec, image: string, weights: Record<string, string> = {}, guards: Guards = endpointGuards(spec)): Launch {
  return checkLaunch({ ...endpointLaunchBase(spec, guards), image, entrypoint: bootFields(spec).dockerEntrypoint, args: [], env: workerEnv(spec, image, weights) });
}

/** REST v1 POST /templates (queue endpoints): a GPU endpoint's from its Launch (src/launch.ts runpodTemplateBody, the
 * Rust provider's rules) under the template's own name, plus what only fv-control sends: the category, a readme and
 * an explicit empty start command (so a PATCH clears an older template's). A CPU endpoint's from its spec. */
export function templateCreatePayload(spec: EndpointSpec, image: string, name: string, weights: Record<string, string> = {}) {
  const readme = `fv-control serverless endpoint ${spec.name} (${imageOf(spec)}); docs/control/serverless.md`;
  if (spec.compute !== "CPU") return { ...runpodTemplateBody({ ...endpointLaunch(spec, image, weights), name }), category: "NVIDIA", dockerStartCmd: [] as string[], readme };
  return {
    name,
    imageName: image,
    isServerless: true,
    category: "CPU",
    containerDiskInGb: spec.container_disk_gb,
    volumeInGb: 0,
    env: workerEnv(spec, image, weights),
    ...bootFields(spec),
    readme,
  };
}
/** REST v1 PATCH /templates/<id>: image, env, start and disk (a rolling release of the endpoint's workers). */
export function templateUpdatePayload(spec: EndpointSpec, image: string, weights: Record<string, string> = {}) {
  const { name: _n, isServerless: _s, category: _c, readme: _r, ...rest } = templateCreatePayload(spec, image, "", weights);
  return rest;
}

/** The scaling and placement fields REST v1 takes on create and PATCH: a GPU endpoint's from its Launch
 * (runpodEndpointBody without the create-only name, template and volume), a CPU endpoint's from its spec. */
export function endpointFields(spec: EndpointSpec) {
  if (spec.compute === "CPU")
    return {
      workersMin: spec.workers_min,
      workersMax: spec.workers_max,
      idleTimeout: spec.idle_timeout_s,
      flashboot: spec.flashboot,
      executionTimeoutMs: spec.execution_timeout_s * 1000,
      scalerType: spec.scaler_type,
      scalerValue: spec.scaler_value,
      ...(spec.data_centers?.length ? { dataCenterIds: spec.data_centers } : {}),
      computeType: "CPU" as const,
      cpuFlavorIds: spec.cpu_flavors || ["cpu3c", "cpu5c"],
      vcpuCount: spec.vcpu ?? 2,
    };
  const { name: _n, templateId: _t, networkVolumeId: _v, ...f } = runpodEndpointBody(checkLaunch(endpointLaunchBase(spec, endpointGuards(spec))), "");
  return f;
}

/** REST v1 POST /endpoints (a queue endpoint on a template): a GPU endpoint's from its Launch (runpodEndpointBody). */
export function endpointCreatePayload(spec: EndpointSpec, templateId: string) {
  if (spec.compute !== "CPU") return runpodEndpointBody(checkLaunch(endpointLaunchBase(spec, endpointGuards(spec))), templateId);
  return { name: endpointName(spec), templateId, ...endpointFields(spec), ...(spec.network_volume ? { networkVolumeId: spec.network_volume } : {}) };
}

/** REST v1 PATCH /endpoints/<id> (EndpointUpdateInput; also scales a v2-made endpoint): scaling, timeouts,
 * placement. Never computeType (not an update field: Runpod refuses "extra input keys"); a CPU endpoint's
 * flavors and vCPUs are v2 fields, changed only by a new endpoint. */
export function endpointUpdatePayload(spec: EndpointSpec, prev?: EndpointSpec) {
  const { computeType: _c, ...f } = endpointFields(spec) as Record<string, unknown>;
  // An empty allowed_cuda after a list clears the endpoint's filter: [] ("not set: any CUDA version", REST v1). Never
  // "every version Runpod knows": REST's enum stops at 13.0, so that list hid the 13.1 / 13.2 hosts (all Secure RTX
  // PRO 6000s, 2026-10-14) from every endpoint fv-control made. minCudaVersion keeps the CUDA 13 minimum.
  if (spec.compute === "GPU" && !(spec.allowed_cuda ?? []).length && (prev?.allowed_cuda ?? []).length) f.allowedCudaVersions = [];
  if (spec.compute === "CPU") {
    delete f.cpuFlavorIds;
    delete f.vcpuCount;
  }
  return f;
}

/** REST v2 POST /serverless: a load-balancer endpoint (GPU pools from Runpod's catalog), or a CPU queue
 * endpoint (REST v1 creates every endpoint with GPU workers: it ignores computeType CPU, live 2026-10-06).
 * v2 makes the endpoint's template itself (Runpod deletes it with the endpoint). The envelope is REST v2's own (not a
 * Launch payload: LB endpoints are the owner's open decision, ADR-0018); a GPU endpoint's image, env, disk, GPU
 * count, CUDA filter, scaling and placement come from its Launch. */
export function v2CreatePayload(spec: EndpointSpec, image: string, pools: string[] = [], weights: Record<string, string> = {}) {
  const lb = spec.mode === "lb";
  const scaling = spec.scaler_type === "REQUEST_COUNT" ? { type: "REQUEST_COUNT", requestCount: spec.scaler_value } : { type: "QUEUE_DELAY", queueDelay: spec.scaler_value };
  if (spec.compute !== "CPU") {
    const l = endpointLaunch(spec, image, weights);
    const s = l.serverless!;
    const vol = l.placement.volume;
    const dcs = l.placement.data_centers ?? [];
    const cuda = l.gpu.allowed_cuda ?? [];
    return {
      name: l.name,
      type: lb ? "LOAD_BALANCER" : "QUEUE",
      image: l.image,
      ...(lb ? { ports: ["8000/http"] } : {}),
      disk: l.disk.container_gb,
      env: l.env ?? {},
      gpu: { pools: pools.length ? pools : ["BLACKWELL_96"], count: l.gpu.count, ...(cuda.length ? { allowedCudaVersions: cuda } : {}) },
      workers: { min: s.workers_min, max: s.workers_max, idleTimeout: s.idle_timeout_s },
      scaling,
      ...(vol ? { networkVolumes: [vol.id] } : {}),
      ...(dcs.length ? { dataCenterIds: dcs } : {}),
      flashboot: s.flashboot ? "FLASHBOOT" : "OFF",
      timeout: s.execution_timeout_ms,
    };
  }
  return {
    name: endpointName(spec),
    type: lb ? "LOAD_BALANCER" : "QUEUE",
    image,
    ...(lb ? { ports: ["8000/http"] } : {}),
    disk: spec.container_disk_gb,
    env: workerEnv(spec, image, weights),
    cpu: (spec.cpu_flavors || ["cpu3c", "cpu5c"]).map((id) => ({ id, vcpuCount: spec.vcpu ?? 2 })),
    workers: { min: spec.workers_min, max: spec.workers_max, idleTimeout: spec.idle_timeout_s },
    scaling,
    ...(spec.network_volume ? { networkVolumes: [spec.network_volume] } : {}),
    ...(spec.data_centers?.length ? { dataCenterIds: spec.data_centers } : {}),
    flashboot: spec.flashboot ? "FLASHBOOT" : "OFF",
    timeout: spec.execution_timeout_s * 1000,
  };
}
/** What a v2-made template gets after create (REST v1 PATCH /templates): the boot of its config (v2 create has no
 * entrypoint field, and an image from before fv-boot does not read FV_WORKER_TOML_B64 on its own). Every endpoint
 * sends its config, so every v2-made template gets it. */
export function v2TemplateBoot(spec: EndpointSpec) {
  return bootFields(spec);
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
