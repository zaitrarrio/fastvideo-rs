// A Runpod serverless endpoint fv-control manages (docs/control/serverless.md):
// one fv-serve image variant behind Runpod's queue (`/run`, `/runsync`; the
// worker runs FV_SERVE_MODE=runpod-queue) or its load balancer (fv-serve's
// HTTP on port 8000, `/ping`). The zod schema is the one validator (create,
// update, the editor's JSON Schema at /api/schemas/serverless-endpoint), as
// for cluster specs (src/schemas.ts).
import { z } from "zod";
import { REGIONS, regionAvailable, type RegionId } from "../cluster/regions";
import { HttpError } from "../util";

/** The CUDA versions the serve images run on (docs/serve/images.md): the driver must offer 13.0. */
export const DEFAULT_CUDA = ["13.0"];
export const CUDA_VERSIONS = ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"] as const;
/** Runpod serverless CPU flavors (REST v1 EndpointCreateInput.cpuFlavorIds). */
export const SLS_CPU_FLAVORS = ["cpu3c", "cpu3g", "cpu5c", "cpu5g"] as const;

/** The weights volumes fv-control may mount (CLAUDE.md: EU only; a region without a volume is unavailable). */
export function knownVolumes(): { id: string; dc: string; region: RegionId }[] {
  return (Object.keys(REGIONS) as RegionId[]).filter(regionAvailable).map((r) => ({ id: REGIONS[r].volume, dc: REGIONS[r].dc, region: r }));
}

/** Env keys fv-control sets itself: a spec's `env` may not. */
export const SLS_RESERVED = new Set([
  "FV_SERVE_MODE",
  "FV_CONFIG",
  "FV_WORKER_TOML_B64",
  "FV_IMAGE_REF",
  "FV_IMAGE_DIGEST",
  "FV_RELEASE_CHANNEL",
  "FV_STATE_DIR",
  "FV_CACHE_DIR",
  "FV_WORKERS_MAX",
  "FV_CONTROL_ENDPOINT",
  "PORT",
  "PORT_HEALTH",
  "RUNPOD_API_KEY",
  "FV_BACKSTOP_API_KEY",
  // Runpod secret references (payloads.ts SECRET_ENV_REFS): values live in Runpod only.
  "FV_CF_ACCOUNT_ID",
  "FV_CF_API_TOKEN",
  "FV_D1_DATABASE_ID",
  "FV_R2_BUCKET",
  "FV_R2_ENDPOINT",
  "FV_R2_ACCESS_KEY_ID",
  "FV_R2_SECRET_ACCESS_KEY",
  "FV_WEBHOOK_ED25519_KEY",
]);

const ID = /^[a-z][a-z0-9-]{0,30}$/;
const ENV_KEY = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;

export const EndpointSpecZ = z
  .object({
    name: z.string().regex(ID).describe("Endpoint name: lower-case letters, digits and '-', starting with a letter (max 31). The Runpod endpoint is fvc-<name>."),
    mode: z.enum(["queue", "lb"]).describe("queue: Runpod's job queue (/run, /runsync; fv-serve runs FV_SERVE_MODE=runpod-queue and takes the native job envelope). lb: Runpod's load balancer in front of fv-serve's HTTP server (port 8000, /ping); GPU only."),
    image: z
      .object({
        channel: z.string().regex(/^[a-z][a-z0-9-]{0,30}$/).meta({ "x-dynamic": "channels" }).optional().describe("Release channel (stable, latest, …): the image <variant>-<channel>."),
        sha: z.string().regex(/^[0-9a-f]{7,40}$/).meta({ "x-dynamic": "shas" }).optional().describe("A git commit: the image <variant>-sha-<sha7>."),
        ref: z.string().regex(/^[a-z0-9.\-]+(:[0-9]+)?\/[a-z0-9._\-/]+(:[A-Za-z0-9._-]+)?(@sha256:[0-9a-f]{64})?$/).optional().describe("An image reference."),
      })
      .strict()
      .refine((i) => [i.channel, i.sha, i.ref].filter(Boolean).length === 1, { message: "exactly one of channel, sha, ref" })
      .describe("Which build the workers run; resolved to a digest at create and update, as clusters resolve theirs."),
    variant: z
      .string()
      .regex(/^[a-z0-9][a-z0-9-]{0,30}$/)
      .meta({ "x-dynamic": "variants" })
      .describe("Image variant (docs/serve/images.md): h3-turbo, h3-max, ltx, wan, wan5b, sfwan (GPU) or cpu (the fake engine, CPU workers)."),
    compute: z.enum(["GPU", "CPU"]).describe("GPU or CPU workers (CPU: the cpu variant, fake engine)."),
    config: z.string().regex(/^\/[A-Za-z0-9._/-]+\.toml$/).optional().describe("A worker config inside the image (FV_CONFIG). Default: the variant's baked config."),
    config_toml: z.string().max(32768).optional().describe("An inline worker config (FV_WORKER_TOML_B64) instead of a file in the image."),
    env: z.record(z.string().regex(ENV_KEY), z.string().max(4096)).optional().describe("Extra env for the workers (plain values; never a secret: use Runpod secrets)."),
    gpu_types: z.array(z.string().min(3).max(80).meta({ "x-dynamic": "gpu_types" })).min(1).max(12).optional().describe("GPU workers: Runpod GPU type ids, in priority order. Default: the EU volume's (RTX PRO 6000 Server)."),
    gpu_count: z.number().int().min(1).max(8).optional().describe("GPUs per worker (default 1)."),
    cpu_flavors: z.array(z.enum(SLS_CPU_FLAVORS)).min(1).optional().describe("CPU workers: flavors in priority order (default cpu3c, cpu5c)."),
    vcpu: z.number().int().min(1).max(32).optional().describe("CPU workers: vCPUs per worker (default 2)."),
    data_centers: z.array(z.string().regex(/^[A-Z]{2,3}-[A-Z]{2,3}-\d{1,2}$/)).max(30).optional().describe("Runpod data centers the workers may run in. With a network volume: its data center only. Default: the volume's, or any."),
    network_volume: z.string().regex(/^[a-z0-9]{6,16}$/).nullable().describe("A weights network volume mounted at /runpod-volume (EU jg48s6o1w0; CLAUDE.md), or null. Default: the EU volume for GPU workers, none for CPU."),
    workers_min: z.number().int().min(0).max(4).describe("Always-on workers (billed while idle). 0 scales to zero."),
    workers_max: z.number().int().min(0).max(8).describe("Most workers at once."),
    idle_timeout_s: z.number().int().min(1).max(3600).describe("A worker without a job this long is stopped."),
    flashboot: z.boolean().describe("Runpod FlashBoot (faster warm starts; docs/serve/images.md §FlashBoot)."),
    execution_timeout_s: z.number().int().min(10).max(86400).describe("A job running longer fails (executionTimeoutMs)."),
    scaler_type: z.enum(["QUEUE_DELAY", "REQUEST_COUNT"]).describe("QUEUE_DELAY: add a worker when a job waited scaler_value seconds; REQUEST_COUNT: one worker per scaler_value queued jobs."),
    scaler_value: z.number().min(0.5).max(3600).describe("The scaler's value (seconds or jobs per worker)."),
    allowed_cuda: z.array(z.enum(CUDA_VERSIONS)).optional().describe("GPU workers: CUDA versions a host may offer (default 13.0, what the images need); [] drops the filter."),
    container_disk_gb: z.number().int().min(5).max(200).describe("Container disk per worker (GB)."),
    deadline_min: z.number().int().min(5).max(7 * 1440).nullable().describe("Backstop: minutes after create (or the last extend) at which deadline_action runs; null: none."),
    deadline_action: z.enum(["scale0", "delete"]).describe("At the deadline: scale0 (workers 0/0, the endpoint stays) or delete (endpoint and template)."),
  })
  .strict()
  .superRefine((s, ctx) => {
    const add = (path: string, message: string) => ctx.addIssue({ code: "custom", message, path: [path] });
    if (s.workers_min > s.workers_max) add("workers_min", "workers_min is above workers_max");
    if (s.config && s.config_toml) add("config", "one of config / config_toml");
    if (s.mode === "lb" && s.config_toml) add("config_toml", "load-balancer endpoints take a config file in the image (config), not an inline one");
    if (s.mode === "lb" && s.scaler_type !== "REQUEST_COUNT") add("scaler_type", "load-balancer endpoints scale by REQUEST_COUNT");
    if (s.compute === "CPU") {
      if (s.mode === "lb") add("mode", "load-balancer endpoints are GPU only");
      if (s.gpu_types || s.gpu_count || s.allowed_cuda) add("gpu_types", "CPU workers take no gpu_types / gpu_count / allowed_cuda");
    } else if (s.cpu_flavors || s.vcpu) add("cpu_flavors", "GPU workers take no cpu_flavors / vcpu");
    if (s.variant === "cpu" && s.compute === "GPU") add("variant", "the cpu variant runs on CPU workers (compute: CPU)");
    if (s.compute === "CPU" && s.variant !== "cpu" && !s.image.ref) add("variant", "CPU workers run the cpu variant (the CUDA variants need a GPU)");
    if (s.network_volume !== null) {
      const v = knownVolumes().find((x) => x.id === s.network_volume);
      if (!v) add("network_volume", `unknown or unavailable volume (${knownVolumes().map((x) => x.id).join(", ") || "none"}; the US volume was deleted 2026-10, CLAUDE.md)`);
      else if (s.data_centers && s.data_centers.some((d) => d !== v.dc)) add("data_centers", `a worker with volume ${v.id} must run in ${v.dc}`);
    }
    for (const k of Object.keys(s.env || {})) if (SLS_RESERVED.has(k)) add("env", `${k} is set by fv-control`);
  })
  .describe("A Runpod serverless endpoint of fv-serve workers (docs/control/serverless.md).");

export type EndpointSpec = z.infer<typeof EndpointSpecZ>;
export interface SpecIssue {
  path: (string | number)[];
  message: string;
}

/** The defaults for a variant: CPU fake-engine workers for `cpu`, one GPU worker on the EU volume otherwise. */
export function defaultEndpointSpec(name: string, variant = "cpu"): EndpointSpec {
  const cpu = variant === "cpu";
  const eu = knownVolumes()[0];
  return {
    name,
    mode: "queue",
    image: { channel: "stable" },
    variant,
    compute: cpu ? "CPU" : "GPU",
    ...(cpu ? { cpu_flavors: ["cpu3c", "cpu5c"] as EndpointSpec["cpu_flavors"], vcpu: 2 } : { gpu_types: [...(REGIONS.eu.gpus || [])], gpu_count: 1, allowed_cuda: [...DEFAULT_CUDA] as EndpointSpec["allowed_cuda"] }),
    network_volume: cpu || !eu ? null : eu.id,
    ...(cpu || !eu ? {} : { data_centers: [eu.dc] }),
    workers_min: 0,
    workers_max: 1,
    idle_timeout_s: 5,
    flashboot: false,
    execution_timeout_s: 1800,
    scaler_type: "QUEUE_DELAY",
    scaler_value: 4,
    container_disk_gb: 20,
    deadline_min: 120,
    deadline_action: "delete",
  };
}

/** Validates a spec without filling defaults (the editor's check). */
export function checkEndpointSpec(doc: unknown): { ok: true; value: EndpointSpec } | { ok: false; issues: SpecIssue[] } {
  const r = EndpointSpecZ.safeParse(doc);
  if (r.success) return { ok: true, value: r.data };
  return { ok: false, issues: r.error.issues.map((i) => ({ path: i.path.map((p) => (typeof p === "symbol" ? String(p) : p)) as (string | number)[], message: i.message })) };
}

/** Fills the variant's defaults under what the input gives, then validates; throws 400 with every issue. */
export function normalizeEndpointSpec(input: any): EndpointSpec {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new HttpError(400, "spec must be an object");
  const variant = typeof input.variant === "string" ? input.variant : "cpu";
  const d = defaultEndpointSpec(String(input.name ?? ""), variant);
  const compute = input.compute === "CPU" || input.compute === "GPU" ? input.compute : d.compute;
  // A compute other than the variant's default drops the other kind's defaults.
  const base: any = { ...d, compute };
  if (compute !== d.compute) {
    for (const k of ["gpu_types", "gpu_count", "allowed_cuda", "cpu_flavors", "vcpu", "data_centers"]) delete base[k];
    if (compute === "CPU") Object.assign(base, { cpu_flavors: ["cpu3c", "cpu5c"], vcpu: 2, network_volume: null });
    else Object.assign(base, { gpu_types: [...REGIONS.eu.gpus], gpu_count: 1, allowed_cuda: [...DEFAULT_CUDA], network_volume: knownVolumes()[0]?.id ?? null, data_centers: knownVolumes()[0] ? [knownVolumes()[0]!.dc] : undefined });
  }
  const s: any = { ...base, ...input, image: input.image && typeof input.image === "object" ? { ...input.image } : base.image };
  // A volume given without data centers runs in the volume's data center.
  if (input.network_volume !== undefined && input.data_centers === undefined) {
    const v = knownVolumes().find((x) => x.id === s.network_volume);
    if (v) s.data_centers = [v.dc];
    else if (s.network_volume === null && compute === "GPU") delete s.data_centers;
  }
  for (const k of Object.keys(s)) if (s[k] === undefined) delete s[k];
  const v = checkEndpointSpec(s);
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".") || "spec"}: ${i.message}`).join("; "), { issues: v.issues });
  return v.value;
}

/** What a change from `a` to `b` touches: the template (image, env, config, disk), the endpoint, or needs a new endpoint. */
export function specDiff(a: EndpointSpec, b: EndpointSpec): { template: boolean; endpoint: boolean; recreate: string[]; scaleUp: boolean } {
  const j = (x: unknown) => JSON.stringify(x ?? null);
  const recreate: string[] = (["name", "mode", "compute", "network_volume"] as const).filter((k) => j(a[k]) !== j(b[k]));
  // A CPU endpoint's flavors and vCPUs are fixed at create (REST v2 fields).
  if (b.compute === "CPU") recreate.push(...(["cpu_flavors", "vcpu"] as const).filter((k) => j(a[k]) !== j(b[k])));
  const template = (["image", "variant", "config", "config_toml", "env", "container_disk_gb"] as const).some((k) => j(a[k]) !== j(b[k]));
  const endpoint = (
    ["gpu_types", "gpu_count", "data_centers", "workers_min", "workers_max", "idle_timeout_s", "flashboot", "execution_timeout_s", "scaler_type", "scaler_value", "allowed_cuda"] as const
  ).some((k) => j(a[k]) !== j(b[k]));
  return { template, endpoint, recreate, scaleUp: b.workers_max > a.workers_max || b.workers_min > a.workers_min };
}
