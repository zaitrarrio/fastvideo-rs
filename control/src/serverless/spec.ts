// A Runpod serverless endpoint fv-control manages (docs/control/serverless.md):
// one fv-serve image variant behind Runpod's queue (`/run`, `/runsync`; the
// worker runs FV_SERVE_MODE=runpod-queue) or its load balancer (fv-serve's
// HTTP on port 8000, `/ping`). The zod schema is the one validator (create,
// update, the editor's JSON Schema at /api/schemas/serverless-endpoint), as
// for cluster specs (src/schemas.ts).
import { z } from "zod";
import { REGIONS, regionAvailable, type RegionId } from "../cluster/regions";
import { HttpError } from "../util";
import { zIssues } from "../zissues";
import {
  CHANNEL_RE,
  CONFIG_PATH_RE,
  CPU_VCPUS,
  CUDA_VERSIONS,
  ENV_KEY_MESSAGE,
  envValueProblem,
  enumMessage,
  IMAGE_REF_RE,
  NAME_MAX,
  NAME_RE,
  NAME_RULE,
  RESERVED_NAMES,
  RUNPOD_DATA_CENTERS,
  RUNPOD_GPU_TYPES,
  SHA_RE,
  SLS_CPU_FLAVORS,
  SLS_PRESET_IDS,
  VARIANTS,
} from "../enums";
import { PRESET_OWNED, presetOwned, slsPreset } from "./presets";

/** The CUDA versions the serve images run on (docs/serve/images.md): the driver must offer 13.0. */
// No CUDA filter by default: Runpod's allowedCudaVersions is a host filter, and "13.0" alone hid the
// EUR-IS-1 RTX PRO 6000 hosts (driver 595.x) so workers never started (docs/serve/e2e/ltx.md, h3-max.md;
// staging h3-max2, 2026-10-07). The images check the driver (>= 580) themselves.
export const DEFAULT_CUDA: string[] = [];
export { CUDA_VERSIONS, SLS_CPU_FLAVORS };

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

const ENV_KEY = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
const uniq = <T extends z.ZodType>(item: T) => z.array(item).refine((a) => new Set(a.map((x) => JSON.stringify(x))).size === a.length, { message: "each value once" });

export const EndpointSpecZ = z
  .object({
    name: z
      .string()
      .min(1)
      .max(NAME_MAX)
      .regex(NAME_RE, { message: NAME_RULE })
      .refine((v) => !(RESERVED_NAMES.endpoint as readonly string[]).includes(v), { message: `reserved (a route uses it): not ${RESERVED_NAMES.endpoint.join(", ")}` })
      .meta({ "x-rule": `${NAME_RULE}; unique among the live endpoints; not ${RESERVED_NAMES.endpoint.join(", ")}`, "x-name": "endpoint" })
      .describe("Endpoint name. The Runpod endpoint is fvc-<name>."),
    mode: z.enum(["queue", "lb"]).describe("queue: Runpod's job queue (/run, /runsync; fv-serve runs FV_SERVE_MODE=runpod-queue and takes the native job envelope). lb: Runpod's load balancer in front of fv-serve's HTTP server (port 8000, /ping); GPU only."),
    preset: z
      .enum(SLS_PRESET_IDS, { error: (i) => enumMessage("a preset", i.input, SLS_PRESET_IDS) })
      .optional()
      .meta({ "x-dynamic": "sls_presets" })
      .describe("What the endpoint serves (GET /api/serverless/presets): the preset sets the image variant, the worker config (in the image or inline), the weights it needs and sensible GPU, disk and timeouts. Without one: a custom endpoint (variant + config)."),
    image: z
      .object({
        channel: z.string().regex(CHANNEL_RE, { message: "a channel: a lower-case word" }).meta({ "x-dynamic": "channels" }).optional().describe("Release channel (stable, latest, …): the image <variant>-<channel>."),
        sha: z.string().regex(SHA_RE, { message: "7-40 lower-case hex characters" }).meta({ "x-dynamic": "shas" }).optional().describe("A git commit: the image <variant>-sha-<sha7>."),
        ref: z.string().max(300).regex(IMAGE_REF_RE, { message: "an image reference: registry/repo[:tag][@sha256:<64 hex>]" }).optional().describe("An image reference."),
      })
      .strict()
      .refine((i) => [i.channel, i.sha, i.ref].filter(Boolean).length === 1, { message: "exactly one of channel, sha, ref" })
      .describe("Which build the workers run; resolved to a digest at create and update, as clusters resolve theirs."),
    variant: z
      .enum(VARIANTS)
      .meta({ "x-dynamic": "variants" })
      .describe("Image variant (docs/serve/images.md): h3-turbo, h3-max, ltx, wan, wan5b, sfwan (GPU) or cpu (the fake engine, CPU workers). Set by the preset; a custom endpoint names it."),
    compute: z.enum(["GPU", "CPU"]).describe("GPU or CPU workers (CPU: the cpu variant, fake engine)."),
    config: z.string().max(200).regex(CONFIG_PATH_RE, { message: "an absolute .toml path in the image" }).optional().meta({ "x-dynamic": "config_paths", "x-rule": "an absolute path ending in .toml" }).describe("Custom: a worker config inside the image (FV_CONFIG; it must be one the variant's image carries). Default: the variant's baked config. Set by the preset."),
    config_toml: z.string().min(1).max(32768).optional().meta({ "x-ui": "textarea" }).describe("Custom: an inline worker config (FV_WORKER_TOML_B64) instead of a file in the image; queue and load-balancer endpoints alike. Set by the preset."),
    env: z.record(z.string().regex(ENV_KEY, { message: ENV_KEY_MESSAGE }).refine((k) => !SLS_RESERVED.has(k), { message: "is set by fv-control" }).meta({ "x-dynamic": "env_keys" }), z.string().max(4096)).optional().describe("Extra env for the workers (plain values; never a secret: use Runpod secrets)."),
    gpu_types: uniq(z.enum(RUNPOD_GPU_TYPES, { error: (i) => enumMessage("a Runpod GPU type id", i.input, RUNPOD_GPU_TYPES) }).meta({ "x-dynamic": "gpu_types" })).min(1).max(12).optional().describe("GPU workers: Runpod GPU type ids, in priority order. Default: the EU volume's (RTX PRO 6000 Server)."),
    gpu_count: z.number().int().min(1).max(8).optional().meta({ "x-unit": "GPUs" }).describe("GPUs per worker (default 1)."),
    cpu_flavors: uniq(z.enum(SLS_CPU_FLAVORS, { error: (i) => enumMessage("a Runpod serverless CPU flavor", i.input, SLS_CPU_FLAVORS) }).meta({ "x-dynamic": "cpu_flavors" })).min(1).optional().describe("CPU workers: flavors in priority order (default cpu3c, cpu5c)."),
    vcpu: z.literal(CPU_VCPUS).optional().meta({ "x-unit": "vCPU" }).describe("CPU workers: vCPUs per worker (default 2)."),
    data_centers: uniq(z.enum(RUNPOD_DATA_CENTERS, { error: (i) => enumMessage("a Runpod data centre", i.input, RUNPOD_DATA_CENTERS) }).meta({ "x-dynamic": "data_centers" })).min(1).max(30).optional().describe("Runpod data centers the workers may run in. With a network volume: its data center only. Default: the volume's, or any."),
    network_volume: z.string().regex(/^[a-z0-9]{6,16}$/, { message: "a Runpod network volume id" }).nullable().meta({ "x-dynamic": "volumes" }).describe("A weights network volume mounted at /runpod-volume (EU jg48s6o1w0; CLAUDE.md), or null. Default: the EU volume for GPU workers, none for CPU."),
    workers_min: z.number().int().min(0).max(4).meta({ "x-unit": "workers" }).describe("Always-on workers (billed while idle). 0 scales to zero."),
    workers_max: z.number().int().min(0).max(8).meta({ "x-unit": "workers" }).describe("Most workers at once."),
    idle_timeout_s: z.number().int().min(5).max(3600).meta({ "x-unit": "s" }).describe("A worker without a job this long is stopped (Runpod: 5-3600 s)."),
    flashboot: z.boolean().describe("Runpod FlashBoot (faster warm starts; docs/serve/images.md §FlashBoot)."),
    execution_timeout_s: z.number().int().min(10).max(86400).meta({ "x-unit": "s" }).describe("A job running longer fails (executionTimeoutMs)."),
    scaler_type: z.enum(["QUEUE_DELAY", "REQUEST_COUNT"], { error: (i) => enumMessage("a Runpod scaler type", i.input, ["QUEUE_DELAY", "REQUEST_COUNT"]) }).describe("QUEUE_DELAY: add a worker when a job waited scaler_value seconds; REQUEST_COUNT: one worker per scaler_value queued jobs."),
    scaler_value: z.number().int().min(1).max(500).describe("The scaler's value: seconds of queue delay (QUEUE_DELAY) or jobs per worker (REQUEST_COUNT); Runpod takes an integer 1-500."),
    allowed_cuda: uniq(z.enum(CUDA_VERSIONS, { error: (i) => enumMessage("a CUDA version Runpod knows", i.input, CUDA_VERSIONS) })).optional().describe("GPU workers: CUDA versions a host may offer (default: no filter, any CUDA; the images check the driver themselves). Setting one can leave workers unplaced."),
    container_disk_gb: z.number().int().min(5).max(200).meta({ "x-unit": "GB" }).describe("Container disk per worker (GB)."),
    deadline_min: z.number().int().min(5).max(7 * 1440).nullable().meta({ "x-unit": "min" }).describe("Backstop: minutes after create (or the last extend) at which deadline_action runs; null: none."),
    deadline_action: z.enum(["scale0", "delete"]).describe("At the deadline: scale0 (workers 0/0, the endpoint stays) or delete (endpoint and template)."),
  })
  .strict()
  .superRefine((s, ctx) => {
    const add = (path: string, message: string) => ctx.addIssue({ code: "custom", message, path: [path] });
    if (s.workers_min > s.workers_max) add("workers_min", "workers_min is above workers_max");
    if (s.config && s.config_toml) add("config", "one of config / config_toml");
    // An inline config works in both modes: the template's start (payloads.ts SLS_BOOT) decodes FV_WORKER_TOML_B64.
    const p = s.preset ? slsPreset(s.preset) : undefined;
    if (p) {
      const own = presetOwned(p);
      for (const k of PRESET_OWNED) if (s[k] !== undefined && s[k] !== own[k]) add(k, `set by preset ${p.id} (${k === "config_toml" ? "its inline config" : JSON.stringify(own[k] ?? "the image's default")}): leave it out, or drop preset for a custom endpoint`);
      if (p.id !== "cpu" && s.compute === "GPU" && s.network_volume === null) add("network_volume", `preset ${p.id} loads weights from the volume: network_volume ${knownVolumes()[0]?.id ?? "(none available)"}`);
    }
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
    for (const [k, v] of Object.entries(s.env || {})) {
      const why = envValueProblem(k, v);
      if (why) ctx.addIssue({ code: "custom", message: why, path: ["env", k] });
    }
  })
  .describe("A Runpod serverless endpoint of fv-serve workers (docs/control/serverless.md).");

export type EndpointSpec = z.infer<typeof EndpointSpecZ>;
export interface SpecIssue {
  path: (string | number)[];
  message: string;
}

/** The defaults for a preset (its variant, config, disk and timeout), or for a bare variant: CPU fake-engine workers for `cpu`, one GPU worker on the EU volume otherwise. */
export function defaultEndpointSpec(name: string, variant = "cpu", preset?: string): EndpointSpec {
  const p = slsPreset(preset);
  if (p) {
    const d = defaultEndpointSpec(name, p.variant);
    delete d.config, delete d.config_toml;
    return { ...d, preset: p.id, compute: p.compute, ...(p.config ? { config: p.config } : {}), ...(p.config_toml ? { config_toml: p.config_toml } : {}), container_disk_gb: p.container_disk_gb, execution_timeout_s: p.execution_timeout_s };
  }
  const cpu = variant === "cpu";
  const eu = knownVolumes()[0];
  return {
    name,
    mode: "queue",
    image: { channel: "stable" },
    variant: variant as EndpointSpec["variant"],
    compute: cpu ? "CPU" : "GPU",
    ...(cpu ? { cpu_flavors: ["cpu3c", "cpu5c"] as EndpointSpec["cpu_flavors"], vcpu: 2 } : { gpu_types: [...(REGIONS.eu.gpus || [])] as EndpointSpec["gpu_types"], gpu_count: 1, allowed_cuda: [...DEFAULT_CUDA] as EndpointSpec["allowed_cuda"] }),
    network_volume: cpu || !eu ? null : eu.id,
    ...(cpu || !eu ? {} : { data_centers: [eu.dc] as EndpointSpec["data_centers"] }),
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
  return { ok: false, issues: zIssues(r.error) };
}

/** Fills the variant's defaults under what the input gives, then validates; throws 400 with every issue. */
export function normalizeEndpointSpec(input: any): EndpointSpec {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new HttpError(400, "spec must be an object");
  // A preset sets the variant, compute and config (the schema refuses an input that changes them).
  const p = slsPreset(input.preset);
  const variant = p ? p.variant : typeof input.variant === "string" ? input.variant : "cpu";
  const d = defaultEndpointSpec(String(input.name ?? ""), variant, p?.id);
  const compute = !p && (input.compute === "CPU" || input.compute === "GPU") ? input.compute : d.compute;
  // A compute other than the variant's default drops the other kind's defaults.
  const base: any = { ...d, compute };
  // A load balancer scales by request count: its default scaler.
  if (input.mode === "lb" && input.scaler_type === undefined) Object.assign(base, { scaler_type: "REQUEST_COUNT", scaler_value: input.scaler_value ?? 1 });
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

/**
 * A GPU endpoint's GPU types against the data centres it may run in (with a
 * volume: the volume's, EUR-IS-1 for the EU weights volume), from Runpod's
 * stock per (type, data centre). A type with no stock anywhere it may run is
 * a warning; when none of the types has stock there it is an error (the
 * workers could never start: e.g. H100 only, on the EU volume). Nothing when
 * Runpod does not answer.
 */
export async function placementIssues(
  spec: EndpointSpec,
  stockOf: (pairs: { dc: string; gpu: string }[]) => Promise<Map<string, { stock: string | null; max_available: number | null }>>,
): Promise<{ issues: SpecIssue[]; warnings: SpecIssue[] }> {
  const issues: SpecIssue[] = [];
  const warnings: SpecIssue[] = [];
  if (spec.compute !== "GPU") return { issues, warnings };
  const vol = knownVolumes().find((v) => v.id === spec.network_volume);
  const dcs = spec.data_centers?.length ? spec.data_centers : vol ? [vol.dc] : [];
  const gpus = spec.gpu_types?.length ? spec.gpu_types : [...REGIONS.eu.gpus];
  if (!dcs.length) return { issues, warnings };
  let st: Map<string, { stock: string | null; max_available: number | null }>;
  try {
    st = await stockOf(gpus.flatMap((gpu) => dcs.map((dc) => ({ dc, gpu }))));
  } catch {
    return { issues, warnings };
  }
  const offered = (g: string) => dcs.some((dc) => { const x = st.get(`${dc}|${g}`); return !!x?.stock || (x?.max_available ?? 0) > 0; });
  const none = gpus.filter((g) => !offered(g));
  const where = vol ? `${dcs.join(", ")} (the ${vol.region} volume ${vol.id}'s data centre)` : dcs.join(", ");
  if (none.length === gpus.length) issues.push({ path: ["gpu_types"], message: `none of ${gpus.join(", ")} is offered in ${where}: the workers could never start (pick a GPU type Runpod has there, e.g. ${REGIONS.eu.gpus[0]})` });
  else for (const g of none) warnings.push({ path: ["gpu_types", gpus.indexOf(g)], message: `${g}: Runpod reports no stock in ${where}; the other types are tried first only in order` });
  return { issues, warnings };
}
