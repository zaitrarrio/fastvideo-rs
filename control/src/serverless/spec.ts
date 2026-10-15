// A Runpod serverless endpoint fv-control manages (docs/control/serverless.md):
// one fv-serve image variant behind Runpod's queue (`/run`, `/runsync`; the
// worker runs FV_SERVE_MODE=runpod-queue) or its load balancer (fv-serve's
// HTTP on port 8000, `/ping`). The zod schema is the one validator (create,
// update, the editor's JSON Schema at /api/v1/schemas/serverless-endpoint), as
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
  DEFAULT_ALLOWED_CUDA,
  MIN_CUDA,
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
import { SLS_MAX_DISK_GB, slsHubDiskGb, slsHubPlan, slsR2DiskGb, slsTrees } from "./weights";
import { missingTrees } from "../volumes";
import { upsamplerDiskGb } from "../upsampler";
import type { Env } from "../env";

/**
 * Where a GPU endpoint's workers get their weights (docs/control/serverless.md §1b, docs/ops/r2-weights.md §9):
 * volume: the EU network volume mounted at /runpod-volume (one data centre); r2: the R2 weights store, fetched onto
 * the container disk at boot (no volume, so any data centre); hub: the Hub at the pinned revisions, downloaded onto the
 * container disk at boot (owner-approved: weights_download_approved and FV_HUB_DOWNLOADS_APPROVED=1; FlashBoot on).
 * The default for new endpoints: flipped to "r2" by the coordinator after the benchmarks (one line).
 */
export const DEFAULT_SLS_WEIGHTS_SOURCE: "volume" | "r2" = "volume";
export const SLS_WEIGHTS_SOURCES = ["volume", "r2", "hub"] as const;
export type SlsWeightsSource = (typeof SLS_WEIGHTS_SOURCES)[number];

/** A GPU endpoint's default allowed_cuda, whatever its weights source (enums.ts DEFAULT_ALLOWED_CUDA): no filter
 * while Runpod's REST enum cannot name the 13.1 / 13.2 hosts ("13.0" alone hid every Secure RTX PRO 6000, so workers
 * never started: staging h3-max2 2026-10-07, 2026-10-14). The endpoint still gets minCudaVersion MIN_CUDA
 * (payloads.ts), and the boot's driver check refuses a host older than CUDA 13 (docs/ops/r2-bench-2026-10-12.md
 * finding 2). */
export const DEFAULT_CUDA: string[] = [...DEFAULT_ALLOWED_CUDA];
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
  // weights_source r2 (serverless/weights.ts slsWeightsEnv): the plan URL, the endpoint's weights token, the trees.
  "FV_WEIGHTS_SOURCE",
  "FV_WEIGHTS_TOKEN",
  "FV_WEIGHTS_PLAN_URL",
  "FV_R2_FETCH_URL",
  "FV_R2_FETCH_SHA256",
  "FV_WEIGHTS_TREES_B64",
  "FV_R2_TREE_REVS",
  "FV_R2_CACHE_ROOT",
  "FV_SLS_REPORT_URL",
  // weights_source hub (serverless/weights.ts slsHubEnv): the recorded hashes, fetch-hub-tree.py's URL.
  "FV_HUB_SHA256_B64",
  "FV_SCRIPTS_URL",
  // upsampler_local (upsampler.ts): set from the spec's switch.
  "FV_UPSAMPLER_LOCAL",
  "FV_UPSAMPLER_REV",
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
      .describe("What the endpoint serves (GET /api/v1/serverless/presets): the preset sets the image variant, the worker config (in the image or inline), the weights it needs and sensible GPU, disk and timeouts. Without one: a custom endpoint (variant + config)."),
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
    weights_source: z
      .enum(SLS_WEIGHTS_SOURCES, { error: (i) => enumMessage("a weights source", i.input, SLS_WEIGHTS_SOURCES) })
      .optional()
      .describe(
        "GPU workers: where the weights come from. volume: the EU weights network volume at /runpod-volume (its data centre only). r2: the R2 weights store, fetched onto the container disk at boot with presigned URLs (no volume, any data centre; skipped when the disk already holds the trees at their revisions). hub: the Hub at the pinned revisions (weights-revisions.tsv), downloaded onto the container disk at boot and checked (the Hub listing, weights-sha256.tsv) before fv-serve starts; no volume, any data centre, FlashBoot on (a revival downloads nothing); needs the owner's approval: weights_download_approved and fv-control's FV_HUB_DOWNLOADS_APPROVED=1. Default: " +
          DEFAULT_SLS_WEIGHTS_SOURCE +
          ".",
      ),
    weights_download_approved: z.boolean().optional().describe("weights_source hub: the owner approved the Hub download at every cold start of this endpoint's workers (also needs fv-control's FV_HUB_DOWNLOADS_APPROVED=1; CLAUDE.md: large downloads need approval)."),
    network_volume: z
      .string()
      .regex(/^[a-z0-9]{6,16}$/, { message: "a Runpod network volume id" })
      .nullable()
      .meta({ "x-dynamic": "volumes" })
      .describe("weights_source volume: the weights network volume mounted at /runpod-volume (EU bz0ofkfx45 in EU-RO-1: H3 + LTX trees only; CLAUDE.md), or null. weights_source r2: null, or an R2 cache volume the owner listed in FV_SLS_R2_CACHE_VOLUMES (off by default). Default: the EU volume for GPU workers on volume, none otherwise."),
    workers_min: z.number().int().min(0).max(4).meta({ "x-unit": "workers" }).describe("Always-on workers (billed while idle). 0 scales to zero."),
    workers_max: z.number().int().min(0).max(8).meta({ "x-unit": "workers" }).describe("Most workers at once."),
    idle_timeout_s: z.number().int().min(5).max(3600).meta({ "x-unit": "s" }).describe("A worker without a job this long is stopped (Runpod: 5-3600 s)."),
    flashboot: z.boolean().describe("Runpod FlashBoot (faster warm starts; docs/serve/images.md §FlashBoot)."),
    execution_timeout_s: z.number().int().min(10).max(86400).meta({ "x-unit": "s" }).describe("A job running longer fails (executionTimeoutMs)."),
    scaler_type: z.enum(["QUEUE_DELAY", "REQUEST_COUNT"], { error: (i) => enumMessage("a Runpod scaler type", i.input, ["QUEUE_DELAY", "REQUEST_COUNT"]) }).describe("QUEUE_DELAY: add a worker when a job waited scaler_value seconds; REQUEST_COUNT: one worker per scaler_value queued jobs."),
    scaler_value: z.number().int().min(1).max(500).describe("The scaler's value: seconds of queue delay (QUEUE_DELAY) or jobs per worker (REQUEST_COUNT); Runpod takes an integer 1-500."),
    allowed_cuda: uniq(z.enum(CUDA_VERSIONS, { error: (i) => enumMessage("a CUDA version Runpod knows", i.input, CUDA_VERSIONS) })).optional().describe(`GPU workers: an exact list of CUDA versions a host may offer (allowedCudaVersions). Default: none, as [] (every endpoint gets minCudaVersion ${MIN_CUDA}, and the boot refuses a driver older than CUDA 13). Runpod's REST enum stops at 13.0 while hosts report 13.1 / 13.2, so a list leaves those hosts out (all Secure RTX PRO 6000s, 2026-10-14).`),
    container_disk_gb: z.number().int().min(5).max(500).meta({ "x-unit": "GB" }).describe("Container disk per worker (GB). weights_source r2: at least the trees' size × 1.15 + 30 GB (the default is sized from the manifests); otherwise up to 200."),
    deadline_min: z.number().int().min(5).max(7 * 1440).nullable().meta({ "x-unit": "min" }).describe("Backstop: minutes after create (or the last extend) at which deadline_action runs; null: none."),
    deadline_action: z.enum(["scale0", "delete"]).describe("At the deadline: scale0 (workers 0/0, the endpoint stays) or delete (endpoint and template)."),
    upsampler_local: z.boolean().optional().describe("The prompt upsampler's local sidecar (docs/serve/upsampler-local.md): llama-server with Qwen3-VL-4B-Instruct (GGUF Q8_0 + its F16 vision projector, 5.1 GB from the R2 weights store; GPU memory: see the doc) next to fv-serve on the same GPU; fv-serve gets FV_UPSAMPLER_BASE_URL. If it fails, fv-serve starts anyway and upsampling falls back. GPU only; default off."),
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
      if (p.id !== "cpu" && s.compute === "GPU" && s.network_volume === null && s.weights_source !== "r2" && s.weights_source !== "hub") add("network_volume", `preset ${p.id} loads weights from the volume: network_volume ${knownVolumes()[0]?.id ?? "(none available)"}, or weights_source r2 (or hub, approved)`);
    }
    if (s.mode === "lb" && s.scaler_type !== "REQUEST_COUNT") add("scaler_type", "load-balancer endpoints scale by REQUEST_COUNT");
    if (s.upsampler_local && s.compute === "CPU") add("upsampler_local", "the upsampler sidecar runs on the GPU: GPU workers only");
    if (s.compute === "CPU") {
      if (s.mode === "lb") add("mode", "load-balancer endpoints are GPU only");
      if (s.gpu_types || s.gpu_count || s.allowed_cuda) add("gpu_types", "CPU workers take no gpu_types / gpu_count / allowed_cuda");
    } else if (s.cpu_flavors || s.vcpu) add("cpu_flavors", "GPU workers take no cpu_flavors / vcpu");
    if (s.variant === "cpu" && s.compute === "GPU") add("variant", "the cpu variant runs on CPU workers (compute: CPU)");
    if (s.compute === "CPU" && s.variant !== "cpu" && !s.image.ref) add("variant", "CPU workers run the cpu variant (the CUDA variants need a GPU)");
    const r2 = s.weights_source === "r2";
    if (r2 && s.compute === "CPU") add("weights_source", "weights_source r2 is for GPU workers (CPU workers run the fake engine: no weights)");
    if (r2 && s.network_volume && knownVolumes().some((x) => x.id === s.network_volume)) add("network_volume", `weights_source r2 replaces the weights volume ${s.network_volume}: network_volume null (or an R2 cache volume, FV_SLS_R2_CACHE_VOLUMES)`);
    const hub = s.weights_source === "hub";
    if (!r2 && !hub && s.container_disk_gb > 200) add("container_disk_gb", "over 200 GB only with weights_source r2 or hub (the trees on the container disk)");
    if (r2) {
      const need = slsR2DiskGb(slsTrees(s));
      if (s.container_disk_gb < need) add("container_disk_gb", `weights_source r2 fetches ${slsTrees(s).join(", ") || "the trees"} onto the container disk: at least ${need} GB`);
    }
    // weights_source hub (owner-approved, as for pods): the spec's approval here, fv-control's FV_HUB_DOWNLOADS_APPROVED
    // and the trees' Hub sources in serverless/weights.ts slsWeightsIssues; no volume, a disk for the trees, FlashBoot.
    if (s.weights_download_approved !== undefined && !hub) add("weights_download_approved", "only with weights_source hub");
    if (hub) {
      const trees = slsHubPlan(s).trees;
      if (s.compute === "CPU") add("weights_source", "weights_source hub is for GPU workers (CPU workers run the fake engine: no weights)");
      if (!s.weights_download_approved) add("weights_download_approved", `weights_source hub downloads ${trees.join(", ") || "the weights"} from the Hub at every cold start: it needs the owner's approval for this endpoint (weights_download_approved: true)`);
      if (s.network_volume !== null) add("network_volume", "weights_source hub downloads onto the container disk: network_volume null (no volume is mounted with hub)");
      if (!s.flashboot) add("flashboot", "weights_source hub downloads the weights at every cold start: FlashBoot on (a revival keeps the worker and downloads nothing)");
      const need = slsHubDiskGb(trees);
      if (s.compute === "GPU" && need > SLS_MAX_DISK_GB) add("weights_source", `${trees.join(", ")} need ${need} GB of container disk, over Runpod's ${SLS_MAX_DISK_GB} GB: weights_source r2 or volume`);
      else if (s.compute === "GPU" && s.container_disk_gb < need) add("container_disk_gb", `weights_source hub downloads ${trees.join(", ") || "the trees"} onto the container disk: at least ${need} GB`);
    }
    // weights_source r2: a cache volume is checked against the owner's list (serverless/weights.ts slsWeightsIssues).
    if (s.network_volume !== null && !r2 && !hub) {
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
    const out: EndpointSpec = { ...d, preset: p.id, compute: p.compute, ...(p.config ? { config: p.config } : {}), ...(p.config_toml ? { config_toml: p.config_toml } : {}), container_disk_gb: p.container_disk_gb, execution_timeout_s: p.execution_timeout_s, ...(p.upsampler_local ? { upsampler_local: true } : {}) };
    // The EU volume holds H3 + LTX only (volumes.ts, owner decision 2026-10-13): a preset with a tree elsewhere
    // defaults to the R2 weights store.
    const eu = knownVolumes()[0];
    const offVolume = p.compute === "GPU" && eu && (missingTrees(eu.id, p.weights) ?? []).length > 0;
    return out.weights_source === "r2" || offVolume ? withWeightsSource(out, "r2") : out;
  }
  const cpu = variant === "cpu";
  const eu = knownVolumes()[0];
  const spec: EndpointSpec = {
    name,
    mode: "queue",
    image: { channel: "stable" },
    variant: variant as EndpointSpec["variant"],
    compute: cpu ? "CPU" : "GPU",
    ...(cpu ? { cpu_flavors: ["cpu3c", "cpu5c"] as EndpointSpec["cpu_flavors"], vcpu: 2 } : { gpu_types: [...(REGIONS.eu.gpus || [])] as EndpointSpec["gpu_types"], gpu_count: 1, allowed_cuda: [...DEFAULT_CUDA] as EndpointSpec["allowed_cuda"] }),
    ...(cpu ? {} : { weights_source: DEFAULT_SLS_WEIGHTS_SOURCE }),
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
  return spec.weights_source === "r2" ? withWeightsSource(spec, "r2") : spec;
}

/** A GPU spec's placement defaults for a weights source: r2 drops the volume and its data centre (any DC) and sizes the
 * container disk from the trees' manifests; hub the same (the disk from the trees' sizes) with FlashBoot on; volume
 * mounts the EU volume in its data centre. */
export function withWeightsSource(spec: EndpointSpec, ws: SlsWeightsSource): EndpointSpec {
  const out: EndpointSpec = { ...spec, weights_source: ws };
  if (ws !== "hub") delete out.weights_download_approved;
  if (ws === "hub") {
    out.network_volume = null;
    delete out.data_centers;
    out.flashboot = true;
    if (!(spec.allowed_cuda ?? []).length) out.allowed_cuda = [...DEFAULT_CUDA] as EndpointSpec["allowed_cuda"];
    out.container_disk_gb = Math.min(SLS_MAX_DISK_GB, Math.max(spec.container_disk_gb, slsHubDiskGb(slsHubPlan(out).trees) + (spec.upsampler_local ? upsamplerDiskGb({} as Env) : 0)));
  } else if (ws === "r2") {
    out.network_volume = null;
    delete out.data_centers;
    if (!(spec.allowed_cuda ?? []).length) out.allowed_cuda = [...DEFAULT_CUDA] as EndpointSpec["allowed_cuda"];
    // upsampler_local: its GGUF tree (upsampler.ts) lands on the same disk (the +30 GB headroom would take it; sized anyway).
    out.container_disk_gb = Math.max(spec.container_disk_gb, slsR2DiskGb(slsTrees(out)) + (spec.upsampler_local ? upsamplerDiskGb({} as Env) : 0));
  } else {
    const eu = knownVolumes()[0];
    out.network_volume = eu ? eu.id : null;
    if (eu) out.data_centers = [eu.dc] as EndpointSpec["data_centers"];
    out.container_disk_gb = Math.min(spec.container_disk_gb, 200);
  }
  return out;
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
    else Object.assign(base, { gpu_types: [...REGIONS.eu.gpus], gpu_count: 1, allowed_cuda: [...DEFAULT_CUDA], weights_source: "volume", network_volume: knownVolumes()[0]?.id ?? null, data_centers: knownVolumes()[0] ? [knownVolumes()[0]!.dc] : undefined });
  }
  // A weights source other than the default's brings its placement (r2: no volume, any DC, a disk for the trees).
  // A spec stored before weights_source existed (or one naming the weights volume) is a volume spec.
  const given = input.weights_source === "r2" || input.weights_source === "volume" || input.weights_source === "hub" ? input.weights_source : typeof input.network_volume === "string" && knownVolumes().some((v) => v.id === input.network_volume) ? "volume" : undefined;
  const ws = given ?? (compute !== d.compute ? DEFAULT_SLS_WEIGHTS_SOURCE : base.weights_source);
  if (compute === "GPU" && ws && ws !== base.weights_source) {
    Object.assign(base, withWeightsSource({ ...base, ...(input.config !== undefined ? { config: input.config } : {}), ...(input.config_toml !== undefined ? { config_toml: input.config_toml } : {}) } as EndpointSpec, ws));
    if (ws === "r2" || ws === "hub") delete base.data_centers;
  }
  if (compute === "CPU") delete base.weights_source;
  if (given && input.weights_source === undefined && compute === "GPU") base.weights_source = given;
  const s: any = { ...base, ...input, image: input.image && typeof input.image === "object" ? { ...input.image } : base.image };
  // A volume given without data centers runs in the volume's data center.
  if (input.network_volume !== undefined && input.data_centers === undefined && s.weights_source !== "r2" && s.weights_source !== "hub") {
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
  const template = (["image", "variant", "config", "config_toml", "env", "container_disk_gb", "weights_source", "upsampler_local"] as const).some((k) => j(a[k]) !== j(b[k]));
  const endpoint = (
    ["gpu_types", "gpu_count", "data_centers", "workers_min", "workers_max", "idle_timeout_s", "flashboot", "execution_timeout_s", "scaler_type", "scaler_value", "allowed_cuda"] as const
  ).some((k) => j(a[k]) !== j(b[k]));
  return { template, endpoint, recreate, scaleUp: b.workers_max > a.workers_max || b.workers_min > a.workers_min };
}

/**
 * A GPU endpoint's GPU types against the data centres it may run in (with a
 * volume: the volume's, EU-RO-1 for the EU weights volume), from Runpod's
 * stock per (type, data centre). A type with no stock anywhere it may run is
 * a warning; when none of the types has stock there it is an error (the
 * workers could never start: e.g. H100 only, on the EU volume). Nothing when
 * Runpod does not answer.
 */
export async function placementIssues(
  spec: EndpointSpec,
  stockOf: (pairs: { dc: string; gpu: string }[]) => Promise<Map<string, { stock: string | null; max_available: number | null }>>,
): Promise<{ issues: SpecIssue[]; warnings: SpecIssue[]; stock?: { gpu: string; dcs: string[] }[] }> {
  const issues: SpecIssue[] = [];
  const warnings: SpecIssue[] = [];
  if (spec.compute !== "GPU") return { issues, warnings };
  const noVol = spec.weights_source === "r2" || spec.weights_source === "hub";
  const vol = noVol ? undefined : knownVolumes().find((v) => v.id === spec.network_volume);
  // weights_source r2 / hub without data_centers: no volume, so every data centre Runpod has (the stock shows where).
  const anyDc = noVol && !spec.data_centers?.length;
  const dcs: string[] = spec.data_centers?.length ? spec.data_centers : vol ? [vol.dc] : anyDc ? [...RUNPOD_DATA_CENTERS] : [];
  const gpus = spec.gpu_types?.length ? spec.gpu_types : [...REGIONS.eu.gpus];
  if (!dcs.length) return { issues, warnings };
  let st: Map<string, { stock: string | null; max_available: number | null }>;
  try {
    st = await stockOf(gpus.flatMap((gpu) => dcs.map((dc) => ({ dc, gpu }))));
  } catch {
    return { issues, warnings };
  }
  const has = (dc: string, g: string) => {
    const x = st.get(`${dc}|${g}`);
    return !!x?.stock || (x?.max_available ?? 0) > 0;
  };
  const stock = gpus.map((gpu) => ({ gpu, dcs: dcs.filter((dc) => has(dc, gpu)) }));
  const offered = (g: string) => dcs.some((dc) => has(dc, g));
  const none = gpus.filter((g) => !offered(g));
  const where = vol ? `${dcs.join(", ")} (the ${vol.region} volume ${vol.id}'s data centre)` : anyDc ? "any data centre" : dcs.join(", ");
  if (none.length === gpus.length) issues.push({ path: ["gpu_types"], message: `none of ${gpus.join(", ")} is offered in ${where}: the workers could never start (pick a GPU type Runpod has there, e.g. ${REGIONS.eu.gpus[0]})` });
  else for (const g of none) warnings.push({ path: ["gpu_types", gpus.indexOf(g)], message: `${g}: Runpod reports no stock in ${where}; the other types are tried first only in order` });
  if (anyDc) warnings.push({ path: ["data_centers"], message: `weights_source ${spec.weights_source}, any data centre: in stock ${stock.map((x) => `${x.gpu.replace(/^NVIDIA /, "")} in ${x.dcs.length ? x.dcs.join(", ") : "none"}`).join("; ")}` });
  return { issues, warnings, stock };
}
