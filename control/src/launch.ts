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

/** REST v1 `POST /pods` (pod.rs create_body; fixtures/deploy/runpod-pod.json). */
export function runpodPodBody(l: Launch) {
  if (l.provider !== "runpod-pod") throw new Error(`launch ${l.name}: a ${l.provider} launch has no pod body`);
  const v = l.placement.volume;
  return {
    name: l.name,
    imageName: l.image,
    cloudType: l.placement.cloud ?? "SECURE",
    computeType: "GPU" as const,
    gpuTypeIds: l.gpu.types,
    gpuCount: l.gpu.count,
    containerDiskInGb: l.disk.container_gb,
    volumeInGb: l.disk.volume_gb,
    ...(nonEmpty(l.ports) ? { ports: l.ports } : {}),
    ...(nonEmpty(l.entrypoint) ? { dockerEntrypoint: l.entrypoint } : {}),
    ...(nonEmpty(l.args) ? { dockerStartCmd: l.args } : {}),
    env: l.env ?? {},
    ...(v ? { networkVolumeId: v.id, volumeMountPath: v.mount_path } : {}),
    ...(nonEmpty(l.placement.data_centers) ? { dataCenterIds: l.placement.data_centers } : {}),
    ...(nonEmpty(l.gpu.allowed_cuda) ? { allowedCudaVersions: l.gpu.allowed_cuda } : {}),
  };
}

/** REST v1 `POST /templates` of a serverless launch (serverless.rs template_body; fixtures/deploy/runpod-template.json). */
export function runpodTemplateBody(l: Launch) {
  if (l.provider !== "runpod-serverless") throw new Error(`launch ${l.name}: a ${l.provider} launch has no serverless template`);
  return {
    name: l.name,
    imageName: l.image,
    isServerless: true as const,
    containerDiskInGb: l.disk.container_gb,
    volumeInGb: l.disk.volume_gb,
    ...(nonEmpty(l.entrypoint) ? { dockerEntrypoint: l.entrypoint } : {}),
    ...(nonEmpty(l.args) ? { dockerStartCmd: l.args } : {}),
    env: l.env ?? {},
  };
}

/** REST v1 `POST /endpoints` of a serverless launch on template `templateId` (serverless.rs endpoint_body;
 * fixtures/deploy/runpod-endpoint.json). */
export function runpodEndpointBody(l: Launch, templateId: string) {
  if (l.provider !== "runpod-serverless") throw new Error(`launch ${l.name}: a ${l.provider} launch has no serverless endpoint`);
  const s = l.serverless ?? DEFAULT_SERVERLESS;
  const v = l.placement.volume;
  return {
    name: l.name,
    templateId,
    computeType: "GPU" as const,
    gpuTypeIds: l.gpu.types,
    gpuCount: l.gpu.count,
    ...(v ? { networkVolumeId: v.id } : {}),
    ...(nonEmpty(l.placement.data_centers) ? { dataCenterIds: l.placement.data_centers } : {}),
    workersMin: s.workers_min,
    workersMax: s.workers_max,
    idleTimeout: s.idle_timeout_s,
    flashboot: s.flashboot,
    executionTimeoutMs: s.execution_timeout_ms,
    scalerType: s.scaler_type,
    scalerValue: s.scaler_value,
    ...(l.gpu.min_cuda != null ? { minCudaVersion: l.gpu.min_cuda } : {}),
    ...(nonEmpty(l.gpu.allowed_cuda) ? { allowedCudaVersions: l.gpu.allowed_cuda } : {}),
  };
}
