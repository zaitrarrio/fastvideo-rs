import { imageOf, MIN_CUDA } from "../enums";
// Runpod payloads of a serverless endpoint (docs/control/serverless.md), on
// REST v2 (api.runpod.io/v2): a serverless template (POST /templates) plus
// the endpoint on it (POST /serverless with `templateId`; `type` QUEUE or
// LOAD_BALANCER, a `gpu` block of catalog pools or a `cpu` list). The shapes
// scripts/serve/runpod-endpoint.sh and runpod-templates.sh proved on REST v1
// map field by field (src/launch.ts). Secrets are Runpod secret references,
// never values.
import { imageIdentEnv, SECRET_ENV_REFS } from "../cluster/payloads";
import type { EndpointSpec } from "./spec";
import { SLS_HUB_SH, SLS_R2_SH } from "./weights";
import { DRIVER_CHECK_SH } from "../driver-check";
import { UPSAMPLER_START_SH, UPSAMPLER_WAIT_SH } from "../upsampler";
import { PREFER_FV_BOOT_SH } from "../fv-boot";
import { workerConfigEnv } from "../worker-config";
import { checkLaunch, type GpuPools, LAUNCH_SCHEMA, runpodEndpointBody, runpodTemplateBody, serverlessScaling, serverlessWorkers, SLS_VOLUME_MOUNT, type Guards, type Launch } from "../launch";

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
  return bash ? { entrypoint: ["bash", "-c", bash], cmd: [] as string[] } : { entrypoint: ["/bin/sh", "-c", SLS_BOOT], cmd: [] as string[] };
};

/**
 * The rules fv-control holds a serverless endpoint to, as a Launch's guards (ADR-0018 §7.2, §10.4).
 *
 * fv-control checks no serverless price (Runpod reports none at create: fv-deploy's own known gap, ADR-0018 §5.3)
 * and no worker readiness (Runpod starts and replaces the workers), so those guards are `null`: "not checked by this
 * launcher" (Launch v1 §10.4; before C3d they carried fv-deploy's fail-closed 0). fv-deploy's own `up` refuses a
 * launch with a `null` guard, and a ledger shows it as "not checked". `SLS_GUARDS_NOT_CHECKED` names them. What is
 * checked: `ready_cycles` 1 (one endpoint per create; fv-control never replaces it), `min_balance_usd` 8 (the least
 * of what create, scale-up and invoke need: the account floor plus the serverless margin, ops.ts assertFloor),
 * `cap_s` = deadline_min (the cron's backstop: scale to 0 or delete; `null` without a deadline: not checked).
 */
/** Always `null` on a serverless endpoint. `cap_s` is also `null` (not checked) when the endpoint has no deadline:
 * `SLS_GUARDS_NOT_CHECKED_WITHOUT_DEADLINE`. */
export const SLS_GUARDS_NOT_CHECKED = ["max_usd_per_hr", "ready_timeout_s"] as const;
export const SLS_GUARDS_NOT_CHECKED_WITHOUT_DEADLINE = ["cap_s"] as const;
export function endpointGuards(spec: Pick<EndpointSpec, "deadline_min">): Guards {
  return { max_usd_per_hr: null, min_balance_usd: 8, cap_s: spec.deadline_min ? spec.deadline_min * 60 : null, ready_timeout_s: null, ready_cycles: 1 };
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
 * endpoint differs by its `type` and the template's port (endpointCreatePayload, templateCreatePayload). */
export function endpointLaunch(spec: EndpointSpec, image: string, weights: Record<string, string> = {}, guards: Guards = endpointGuards(spec)): Launch {
  return checkLaunch({ ...endpointLaunchBase(spec, guards), image, entrypoint: bootFields(spec).entrypoint, args: [], env: workerEnv(spec, image, weights) });
}

/** The load balancer's port: fv-serve's HTTP on 8000 (`ports` is container config: it lives on the template, which the
 * endpoint create resolves). */
const LB_PORTS = ["8000/http"];

/** REST v2 POST /templates: a GPU endpoint's from its Launch (src/launch.ts runpodTemplateBody, the Rust provider's
 * rules) under the template's own name, plus what only fv-control sends: the category and an explicit empty start
 * command (so a PATCH clears an older template's); a load balancer's port. A CPU endpoint's from its spec. (v1's
 * `readme` has no v2 field: v2 refuses unknown keys.) */
export function templateCreatePayload(spec: EndpointSpec, image: string, name: string, weights: Record<string, string> = {}) {
  const ports = spec.mode === "lb" ? { ports: LB_PORTS } : {};
  if (spec.compute !== "CPU") return { ...runpodTemplateBody({ ...endpointLaunch(spec, image, weights), name }), ...ports, category: "NVIDIA", cmd: [] as string[] };
  return {
    name,
    image,
    serverless: true,
    category: "CPU",
    disk: spec.container_disk_gb,
    env: workerEnv(spec, image, weights),
    ...bootFields(spec),
    ...ports,
  };
}
/** REST v2 PATCH /templates/<id>: image, env, start and disk (a rolling release of the endpoint's workers, once the
 * endpoint re-applies the template: ops.ts applyUpdate PATCHes it with `templateId`, v2 keeping no link). */
export function templateUpdatePayload(spec: EndpointSpec, image: string, weights: Record<string, string> = {}) {
  const { name: _n, serverless: _s, category: _c, ...rest } = templateCreatePayload(spec, image, "", weights);
  return rest;
}

/** The scaling and placement fields REST v2 takes on create and PATCH: a GPU endpoint's from its Launch
 * (runpodEndpointBody without the create-only name, template, type and volume), a CPU endpoint's from its spec. `gpu`:
 * the catalog pools of the spec's GPU types (launch.ts gpuPools); without them (`pools` omitted) a PATCH leaves the
 * endpoint's pools as they are. */
export function endpointFields(spec: EndpointSpec, gpu: GpuPools = { pools: [] }) {
  if (spec.compute === "CPU")
    return {
      cpu: (spec.cpu_flavors || ["cpu3c", "cpu5c"]).map((id) => ({ id, vcpuCount: spec.vcpu ?? 2 })),
      ...(spec.data_centers?.length ? { dataCenterIds: spec.data_centers } : {}),
      workers: serverlessWorkers({ workers_min: spec.workers_min, workers_max: spec.workers_max, idle_timeout_s: spec.idle_timeout_s, scaler_type: spec.scaler_type }),
      scaling: serverlessScaling({ scaler_type: spec.scaler_type, scaler_value: spec.scaler_value }),
      flashboot: spec.flashboot ? ("FLASHBOOT" as const) : ("OFF" as const),
      timeout: spec.execution_timeout_s * 1000,
    };
  const { name: _n, templateId: _t, type: _y, networkVolumes: _v, gpu: g, ...f } = runpodEndpointBody(checkLaunch(endpointLaunchBase(spec, endpointGuards(spec))), "", gpu);
  const { pools, excludedTypes, ...rest } = g;
  return { ...f, gpu: { ...(pools.length ? { pools, ...(excludedTypes ? { excludedTypes } : {}) } : {}), ...rest } };
}

/** REST v2 POST /serverless (an endpoint on a template): a GPU endpoint's from its Launch (runpodEndpointBody) with its
 * mode's `type`; a CPU endpoint's from its spec. `gpu`: the catalog pools of a GPU endpoint's types (required on create). */
export function endpointCreatePayload(spec: EndpointSpec, templateId: string, gpu: GpuPools = { pools: [] }) {
  const type = spec.mode === "lb" ? ("LOAD_BALANCER" as const) : ("QUEUE" as const);
  if (spec.compute !== "CPU") return { ...runpodEndpointBody(checkLaunch(endpointLaunchBase(spec, endpointGuards(spec))), templateId, gpu), type };
  return { name: endpointName(spec), templateId, type, ...endpointFields(spec), ...(spec.network_volume ? { networkVolumes: [spec.network_volume] } : {}) };
}

/** REST v2 PATCH /serverless/<id> (an UpdateEndpointRequest): scaling, timeouts, placement, the GPU block. Never the
 * `cpu` list: fv-control recreates a CPU endpoint for new flavors or vCPUs (spec.ts specDiff; v2 could replace them in
 * place). `gpu`: the catalog pools when the caller has them (ops.ts applyUpdate), else only the count and the CUDA
 * filter (pools unchanged). */
export function endpointUpdatePayload(spec: EndpointSpec, prev?: EndpointSpec, gpu: GpuPools = { pools: [] }) {
  const { cpu: _c, ...f } = endpointFields(spec, gpu) as Record<string, any>;
  // An empty allowed_cuda after a list clears the endpoint's filter: [] ("no constraint", which may accompany the
  // minCudaVersion floor). Never "every version Runpod knows": REST v1's enum stopped at 13.0, so that list hid the
  // 13.1 / 13.2 hosts (all Secure RTX PRO 6000s, 2026-10-14) from every endpoint fv-control made. minCudaVersion keeps
  // the CUDA 13 minimum.
  if (spec.compute === "GPU" && !(spec.allowed_cuda ?? []).length && (prev?.allowed_cuda ?? []).length) f.gpu = { ...f.gpu, allowedCudaVersions: [] };
  return f;
}

/** The queue job body of a test invoke (the native envelope; docs/serve/e2e/serverless.md). */
export function invokeBody(input: unknown, executionTimeoutS: number) {
  return { input: input ?? { kind: "info" }, policy: { executionTimeout: executionTimeoutS * 1000 } };
}

/** A Runpod endpoint (REST v2 GET /serverless/<id> with its GET …/workers list: runpod-sls.ts getEndpoint) reduced to
 * what fv-control shows: never its env (secrets in clear). `templateId`: fv-control's own record (v2 keeps no link from
 * an endpoint to its template). `cost`: a worker's $/hr from GraphQL (`costs`, by worker id) when the caller has it; v2's
 * worker list carries none. */
export function endpointView(e: any, o: { templateId?: string | null; costs?: Map<string, number> } = {}) {
  if (!e || typeof e !== "object") return null;
  return {
    id: e.id,
    name: e.name,
    type: e.type ?? null,
    templateId: o.templateId ?? null,
    compute: Array.isArray(e.cpu) ? "CPU" : e.gpu ? "GPU" : null,
    gpu: e.gpu ? { pools: e.gpu.pools ?? null, excludedTypes: e.gpu.excludedTypes ?? null, count: e.gpu.count ?? null, allowedCudaVersions: e.gpu.allowedCudaVersions ?? null, minCudaVersion: e.gpu.minCudaVersion ?? null } : null,
    cpu: Array.isArray(e.cpu) ? e.cpu.map((c: any) => ({ id: c.id, vcpuCount: c.vcpuCount, memory: c.memory ?? null })) : null,
    dataCenterIds: e.dataCenterIds ?? null,
    networkVolumes: e.networkVolumes ?? null,
    workersMin: e.workers?.min,
    workersMax: e.workers?.max,
    idleTimeout: e.workers?.idleTimeout,
    flashboot: e.flashboot,
    timeout: e.timeout,
    scaling: e.scaling ?? null,
    version: e.version,
    image: e.image ?? null,
    workers: Array.isArray(e.workerList)
      ? e.workerList.map((w: any) => ({ id: w.id, status: w.status ?? null, cost: o.costs?.get(String(w.id)) ?? null, gpu: w.gpuTypeId ?? null, dc: w.dataCenterId ?? null, startedAt: w.startedAt ?? null, uptimeS: w.uptimeSeconds ?? null, isStale: w.isStale ?? null, image: w.image ?? null }))
      : undefined,
  };
}
