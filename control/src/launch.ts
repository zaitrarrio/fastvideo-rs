// fv-control's launches on fv-deploy's `Launch` (plan C3c, docs/adr/0018-fv-deploy.md §7): every Runpod GPU pod
// (cluster workers, edge fronts, standalone pods) and every serverless queue endpoint (its template and its endpoint)
// is built as a `Launch` value (the type generated from crates/fastvideo-deploy/schemas/launch.v1.json,
// src/launch-gen.ts), checked against the schema, and its Runpod request bodies are derived from it here by the same
// rules as the Rust provider (crates/fastvideo-deploy/src/provider/runpod/{pod,serverless}.rs `create_body`,
// `template_body`, `endpoint_body`). test/unit/launch-goldens.test.ts holds both to the payload goldens C3a committed
// (crates/fastvideo-deploy/fixtures/deploy/*.json, what runpod-pod.sh / runpod-endpoint.sh send).
//
// What a Launch v1 cannot say stays outside it (ADR-0018 §7.3): Runpod CPU pods and CPU endpoints (no CPU compute in
// the schema), GMI and Brev pods (no such provider kind), load-balancer endpoints (REST v2; the owner's open Runpod LB
// decision) and the build pods.
import { LaunchZ, type Guards, type Launch, type ServerlessSpec } from "./launch-gen";

export type { CloudType, Disk, GpuSpec, Guards, Launch, LaunchSource, Placement, ProviderKind, ServerlessSpec, VolumeMount, WeightsSource } from "./launch-gen";

/** The only Launch schema fv-deploy reads and writes (types.rs LAUNCH_SCHEMA). */
export const LAUNCH_SCHEMA = 1;
/** A pod's network-volume mount (launch.rs POD_VOLUME_MOUNT). */
export const POD_VOLUME_MOUNT = "/workspace";
/** Where a serverless worker sees its network volume. */
export const SLS_VOLUME_MOUNT = "/runpod-volume";

/** `runpod-endpoint.sh`'s endpoint scaling (launch.rs default_serverless): what `endpointBody` sends for a serverless
 * launch without its own. */
export const DEFAULT_SERVERLESS: ServerlessSpec = { workers_min: 0, workers_max: 1, idle_timeout_s: 5, flashboot: false, execution_timeout_ms: 1_800_000, scaler_type: "QUEUE_DELAY", scaler_value: 1 };

/** fv-deploy's default guards (guard.rs default_guards): $2.50/hr, the $8 floor, 1800 s cap, 600 s readiness, 3 cycles. */
export const FV_DEPLOY_DEFAULT_GUARDS: Guards = { max_usd_per_hr: 2.5, min_balance_usd: 8, cap_s: 1800, ready_timeout_s: 600, ready_cycles: 3 };

/** A Launch that does not match the schema (a bug in the caller: fv-control builds every field itself). */
export class LaunchSchemaError extends Error {
  constructor(
    readonly launch: string,
    readonly issues: { path: (string | number)[]; message: string }[],
  ) {
    super(`launch ${launch} does not match fv-deploy's Launch schema: ${issues.map((i) => `${i.path.join(".") || "launch"}: ${i.message}`).join("; ")}`);
  }
}

/** The launch, checked against the schema (launch.v1.json through LaunchZ: unknown fields refused, as serde's
 * deny_unknown_fields); throws LaunchSchemaError. */
export function checkLaunch(l: Launch): Launch {
  const r = LaunchZ.safeParse(l);
  if (!r.success) throw new LaunchSchemaError(l?.name ?? "?", r.error.issues.map((i) => ({ path: i.path.map((p) => (typeof p === "symbol" ? String(p) : p)), message: i.message })));
  return l;
}

const nonEmpty = <T>(xs: T[] | undefined): xs is T[] => !!xs && xs.length > 0;

/** A pod's `mounts` (REST v2; pod.rs mounts_of): the network volume at its mount path, else the pod's own disk
 * (`volume_gb`, at /workspace) when it has one; nothing for neither (v1 sent `volumeInGb: 0`: no default pod volume).
 * v2 refuses `persistent` together with `network` (400). */
function podMounts(l: Launch) {
  const v = l.placement.volume;
  if (v) return { mounts: { network: [{ volumeId: v.id, path: v.mount_path }] } };
  return l.disk.volume_gb > 0 ? { mounts: { persistent: { size: l.disk.volume_gb, path: POD_VOLUME_MOUNT } } } : {};
}

/** REST v2 `POST /pods` (a CreatePodRequest; pod.rs create_body: v1's `imageName` → `image`, `cloudType` → `cloud`,
 * `gpuTypeIds` + `gpuCount` → `gpu.{id,count}`, `containerDiskInGb` → `disk`, `volumeInGb` + `networkVolumeId` +
 * `volumeMountPath` → `mounts`, `dockerEntrypoint` → `entrypoint`, `dockerStartCmd` → `cmd`, `allowedCudaVersions` and
 * `minRAMPerGPU` → `gpu.allowedCudaVersions` / `gpu.minRamPerGpu`; no `computeType`: a `gpu` object is a GPU pod). v2
 * takes one GPU type per pod where v1 took a list: every fv-control launch names one type per placement attempt
 * (cluster/payloads.ts workerPlacements, as fv-deploy's `Launch::for_gpu`), so the first is the one; a launch with
 * none sends no `gpu` (refused by Runpod, as v1's empty list was). */
export function runpodPodBody(l: Launch) {
  if (l.provider !== "runpod-pod") throw new Error(`launch ${l.name}: a ${l.provider} launch has no pod body`);
  const gpu = l.gpu.types[0];
  return {
    name: l.name,
    image: l.image,
    cloud: l.placement.cloud ?? "SECURE",
    ...(gpu !== undefined
      ? {
          gpu: {
            id: gpu,
            count: l.gpu.count,
            ...(nonEmpty(l.gpu.allowed_cuda) ? { allowedCudaVersions: l.gpu.allowed_cuda } : {}),
            // The host-RAM floor, only when set (pod.rs create_body's minRAMPerGPU; ADR-0019 §16.10).
            ...(l.gpu.min_ram_gb != null ? { minRamPerGpu: l.gpu.min_ram_gb } : {}),
          },
        }
      : {}),
    disk: l.disk.container_gb,
    ...(nonEmpty(l.ports) ? { ports: l.ports } : {}),
    ...(nonEmpty(l.entrypoint) ? { entrypoint: l.entrypoint } : {}),
    ...(nonEmpty(l.args) ? { cmd: l.args } : {}),
    env: l.env ?? {},
    ...podMounts(l),
    ...(nonEmpty(l.placement.data_centers) ? { dataCenterIds: l.placement.data_centers } : {}),
  };
}

/** REST v2 `POST /templates` of a serverless launch (serverless.rs template_body: v1's `imageName` → `image`,
 * `isServerless` → `serverless`, `containerDiskInGb` → `disk`, `volumeInGb` → `mounts.persistent` at /workspace (none
 * at 0), `dockerEntrypoint` → `entrypoint`, `dockerStartCmd` → `cmd`). */
export function runpodTemplateBody(l: Launch) {
  if (l.provider !== "runpod-serverless") throw new Error(`launch ${l.name}: a ${l.provider} launch has no serverless template`);
  return {
    name: l.name,
    image: l.image,
    serverless: true as const,
    disk: l.disk.container_gb,
    ...(l.disk.volume_gb > 0 ? { mounts: { persistent: { size: l.disk.volume_gb, path: POD_VOLUME_MOUNT } } } : {}),
    ...(nonEmpty(l.entrypoint) ? { entrypoint: l.entrypoint } : {}),
    ...(nonEmpty(l.args) ? { cmd: l.args } : {}),
    env: l.env ?? {},
  };
}

/** The GPUs of a serverless endpoint as REST v2 names them: serverless **pools** (`GET /catalog/gpus` `pool`, e.g.
 * `BLACKWELL_96`), narrowed to exact GPU types with `excludedTypes` (no inclusive allowlist). v1 took the GPU type
 * ids themselves (`gpuTypeIds`). */
export interface GpuPools {
  pools: string[];
  excludedTypes?: string[];
}
/** A launch's GPU names over the catalog (serverless.rs `EndpointGpu::resolve`): a name equal to a pool id selects
 * the pool whole; a name equal to a type id selects its pool and excludes the pool's other types (unless they are
 * named too), so the endpoint runs on exactly the types asked for (v1's `gpuTypeIds`); a name that is neither, or a
 * type outside every serverless pool, is refused (an Error naming it), never silently widened. Pools in the names'
 * order, the exclusions sorted. */
export function gpuPools(names: string[], catalog: { id: string; pool: string | null }[]): GpuPools {
  const pools: string[] = [];
  const whole: string[] = [];
  for (const n of names) {
    if (catalog.some((c) => c.pool === n)) {
      if (!pools.includes(n)) pools.push(n);
      whole.push(n);
      continue;
    }
    const g = catalog.find((c) => c.id === n);
    if (!g) throw new Error(`serverless GPU ${n}: neither a GPU type nor a pool of GET /catalog/gpus`);
    if (!g.pool) throw new Error(`serverless GPU ${n}: not in any serverless pool (GET /catalog/gpus pool is null)`);
    if (!pools.includes(g.pool)) pools.push(g.pool);
  }
  const excludedTypes = [...new Set(catalog.filter((c) => c.pool && pools.includes(c.pool) && !whole.includes(c.pool) && !names.includes(c.id)).map((c) => c.id))].sort();
  return { pools, ...(excludedTypes.length ? { excludedTypes } : {}) };
}

/** REST v2 `POST /serverless` of a serverless launch on template `templateId`: a queue endpoint (serverless.rs
 * endpoint_body's v1 fields mapped: `computeType: GPU` + `gpuTypeIds` + `gpuCount` + the CUDA filter → `gpu`
 * (`pools` / `excludedTypes` from the catalog, `gpu`; `count`, `allowedCudaVersions`, `minCudaVersion`), `workersMin` /
 * `workersMax` / `idleTimeout` → `workers`, `scalerType` + `scalerValue` → `scaling`, `flashboot` true / false →
 * `"FLASHBOOT"` / `"OFF"`, `executionTimeoutMs` → `timeout`, `networkVolumeId` → `networkVolumes`; `type: "QUEUE"` is
 * required). `workers.idleTimeout` is left out under the REQUEST_COUNT scaler (v2 rejects it there); a CUDA list and
 * the CUDA floor are mutually exclusive in v2 (400), so the floor goes only without a list (serverless.rs
 * endpoint_body). */
export function runpodEndpointBody(l: Launch, templateId: string, gpu: GpuPools) {
  if (l.provider !== "runpod-serverless") throw new Error(`launch ${l.name}: a ${l.provider} launch has no serverless endpoint`);
  const s = l.serverless ?? DEFAULT_SERVERLESS;
  const v = l.placement.volume;
  return {
    name: l.name,
    templateId,
    type: "QUEUE" as const,
    gpu: {
      pools: gpu.pools,
      ...(nonEmpty(gpu.excludedTypes) ? { excludedTypes: gpu.excludedTypes } : {}),
      count: l.gpu.count,
      ...(nonEmpty(l.gpu.allowed_cuda) ? { allowedCudaVersions: l.gpu.allowed_cuda } : l.gpu.min_cuda != null ? { minCudaVersion: l.gpu.min_cuda } : {}),
    },
    ...(v ? { networkVolumes: [v.id] } : {}),
    ...(nonEmpty(l.placement.data_centers) ? { dataCenterIds: l.placement.data_centers } : {}),
    workers: serverlessWorkers(s),
    scaling: serverlessScaling(s),
    flashboot: s.flashboot ? ("FLASHBOOT" as const) : ("OFF" as const),
    timeout: s.execution_timeout_ms,
  };
}
/** `workers` of a v2 endpoint body (no `idleTimeout` under REQUEST_COUNT: v2 rejects it on create and update). */
export function serverlessWorkers(s: Pick<ServerlessSpec, "workers_min" | "workers_max" | "idle_timeout_s" | "scaler_type">) {
  return { min: s.workers_min, max: s.workers_max, ...(s.scaler_type === "REQUEST_COUNT" ? {} : { idleTimeout: s.idle_timeout_s }) };
}
/** `scaling` of a v2 endpoint body: the discriminated union on `type`. */
export function serverlessScaling(s: Pick<ServerlessSpec, "scaler_type" | "scaler_value">) {
  return s.scaler_type === "REQUEST_COUNT" ? { type: "REQUEST_COUNT" as const, requestCount: s.scaler_value } : { type: "QUEUE_DELAY" as const, queueDelay: s.scaler_value };
}
