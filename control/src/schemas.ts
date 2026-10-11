// Every configuration fv-control accepts, as zod schemas: the one source for
// server-side validation (every write path), the JSON Schemas served at
// /api/schemas (the dashboard binds its controls to them: enums become
// selects, patterns and ranges inline checks, `x-rule` the rule shown before
// typing, `x-unit` the unit, `x-dynamic` the live list a control offers), and
// the TypeScript types of the cluster spec (cluster/spec.ts infers them from
// here). docs/control/config-validation.md has the field-by-field table.
import { z } from "zod";
import type { Policies } from "./alerts";
import CATALOG from "./cluster/catalog.json";
import { AVAILABLE_REGIONS, REGIONS, type RegionId } from "./cluster/regions";
import {
  BUILD_VCPUS,
  CHANNEL_RE,
  CONFIG_PATH_RE,
  CPU_FLAVORS,
  CPU_VCPUS,
  EDGE_FAMILIES,
  ENV_KEY_RE,
  ENV_KEY_MESSAGE,
  ENV_KEY_RULE,
  envValueProblem,
  enumMessage,
  FAKE_MODELS,
  IMAGE_REF_RE,
  isReserved,
  KEY_NAME_RE,
  KEY_NAME_RULE,
  LOG_LEVELS,
  LOG_SOURCES,
  NAME_MAX,
  NAME_RE,
  NAME_RULE,
  POOL_PRESET_IDS,
  PROVIDER_GPU_RE,
  PROVIDER_REGION_RE,
  PROVIDERS,
  RESERVED_NAMES,
  WEIGHTS_SOURCES,
  RUNPOD_DATA_CENTERS,
  RUNPOD_GPU_TYPES,
  SHA_RE,
  TOKEN_NAME_RE,
  TOKEN_NAME_RULE,
  TOKEN_SCOPES,
  VARIANTS,
  type NameKind,
} from "./enums";
import { EndpointSpecZ } from "./serverless/spec";
import { HttpError } from "./util";
import { zIssues, type Issue } from "./zissues";

export { ENV_KEY_RE };

/** An fv-control name (enums.ts NAME_RE) that is not reserved for this kind. */
export const nameZ = (kind: NameKind, what: string) =>
  z
    .string()
    .min(1)
    .max(NAME_MAX)
    .regex(NAME_RE, { message: NAME_RULE })
    .refine((v) => !(RESERVED_NAMES[kind] as readonly string[]).includes(v), { message: `reserved (a route uses it): not ${RESERVED_NAMES[kind].join(", ")}` })
    .meta({ "x-rule": `${NAME_RULE}; not ${RESERVED_NAMES[kind].join(", ")}`, "x-name": kind })
    .describe(`${what}: ${NAME_RULE}.`);
const uniq = <T extends z.ZodType>(item: T) => z.array(item).refine((a) => new Set(a.map((x) => JSON.stringify(x))).size === a.length, { message: "each value once" });
const EU_ONLY = `EU only: ${AVAILABLE_REGIONS.map((r) => `${r} (${REGIONS[r].dc})`).join(", ")}; the us weights volume was deleted 2026-10 (CLAUDE.md)`;
const region = z
  .enum(AVAILABLE_REGIONS as [RegionId, ...RegionId[]], { error: () => EU_ONLY })
  .meta({ "x-dynamic": "regions" })
  .describe("eu = volume jg48s6o1w0 in EUR-IS-1 (RTX PRO 6000). us (US-CA-2) is unavailable: its weights volume was deleted 2026-10; EU only, see docs/ops/runpod-volumes.md.");
const cpuFlavor = z.enum(CPU_FLAVORS, { error: (i) => enumMessage("a Runpod CPU flavor", i.input, CPU_FLAVORS) }).meta({ "x-dynamic": "cpu_flavors" }).describe("Runpod CPU flavor: 3/5 = generation; c compute, g general, m memory optimised.");
const gpuType = z.enum(RUNPOD_GPU_TYPES, { error: (i) => enumMessage("a Runpod GPU type id", i.input, RUNPOD_GPU_TYPES) }).meta({ "x-dynamic": "gpu_types" }).describe("A Runpod GPU type id.");
// GMI Cloud / NVIDIA Brev (docs/serve/deploy-gmi-brev.md): where a pod runs, the provider's GPU product, its region, the weights.
const provider = z
  .enum(PROVIDERS, { error: (i) => enumMessage("a provider", i.input, PROVIDERS) })
  .meta({ "x-dynamic": "providers" })
  .describe("Where the pods run: runpod (default), gmi (GMI Cloud containers) or brev (NVIDIA Brev VMs); docs/serve/deploy-gmi-brev.md.");
const providerGpu = z
  .string()
  .regex(PROVIDER_GPU_RE, { message: "a provider product / instance type id: letters, digits, . _ : -" })
  .meta({ "x-dynamic": "provider_gpus", "x-rule": "the provider's own id: a GMI product (container.h200.x1) or a Brev instanceType" })
  .describe("gmi / brev: the provider's GPU product (GMI) or instance type (Brev); one of GMI_PRODUCTS / BREV_INSTANCE_TYPES.");
const providerRegion = z.string().regex(PROVIDER_REGION_RE, { message: "an IDC id: letters, digits, . _ - (≤ 50)" }).meta({ "x-dynamic": "provider_regions" }).describe("gmi: the IDC (GET /v1/idcs; default GMI_DEFAULT_IDC). brev: unused (the instance type implies it).");
const weightsSource = z
  .enum(WEIGHTS_SOURCES)
  .describe("Weights: volume (the region's Runpod network volume; Runpod only), hub (downloaded from the Hub at pinned revisions at boot, verified; needs the owner's approval) or none (fake engine). Default: volume on Runpod, none elsewhere.");
/** The provider fields' rules shared by a pool and a standalone launch. */
function providerRules(x: { provider?: string; provider_gpu?: string; provider_region?: string; weights_source?: string; gpu_types?: unknown; cpu_flavors?: unknown; vcpu?: unknown; volume?: boolean; compute?: string; models?: unknown[] }, add: (path: string, message: string) => void, gpuTypesKey = "gpu_types") {
  const other = !!x.provider && x.provider !== "runpod";
  if (!other) {
    if (x.provider_gpu) add("provider_gpu", "provider_gpu is for provider gmi / brev (Runpod: gpu_types)");
    if (x.provider_region) add("provider_region", "provider_region is for provider gmi");
    if (x.weights_source === "hub") add("weights_source", "Runpod pods read the weights volume (volume) or none");
    return;
  }
  if (!x.provider_gpu) add("provider_gpu", `provider ${x.provider}: the GPU product / instance type is required`);
  if (x.gpu_types) add(gpuTypesKey, `gpu_types are Runpod GPU type ids: provider ${x.provider} takes provider_gpu`);
  if (x.cpu_flavors || x.vcpu !== undefined) add("cpu_flavors", `provider ${x.provider} has GPU machines only (the cpu variant runs on one)`);
  if (x.compute === "CPU") add("compute", `provider ${x.provider} has GPU machines only: compute GPU (the cpu variant runs there too)`);
  if (x.volume) add("volume", `provider ${x.provider} cannot mount the Runpod weights volume: weights_source hub or none`);
  if (x.weights_source === "volume") add("weights_source", `provider ${x.provider} has no Runpod volume: hub or none`);
  if (x.provider === "brev" && x.provider_region) add("provider_region", "brev: the instance type implies the region");
  if ((x.weights_source ?? "none") === "none" && x.models?.length) add("weights_source", "models need weights: hub (owner approval) or serve fake_models");
}
const imageRef = z.string().max(300).regex(IMAGE_REF_RE, { message: "an image reference: registry/repo[:tag][@sha256:<64 hex>]" }).meta({ "x-rule": "registry/repo[:tag][@sha256:<64 hex>], lower-case" });
const configPath = z.string().max(200).regex(CONFIG_PATH_RE, { message: "an absolute .toml path in the image" }).meta({ "x-rule": "an absolute path ending in .toml, e.g. /etc/fv/runpod.toml", "x-dynamic": "config_paths" });
const channel = z.string().regex(CHANNEL_RE, { message: "a channel: a lower-case word" }).meta({ "x-dynamic": "channels" });
const sha = z.string().regex(SHA_RE, { message: "7-40 lower-case hex characters" }).meta({ "x-dynamic": "shas", "x-rule": "a git commit: 7-40 hex characters" });

// The fv-serve catalog (cluster/catalog.json): static per build, so they are enums.
const MODEL_IDS = CATALOG.models.map((m) => m.id) as [string, ...string[]];
const FAMILY_IDS = CATALOG.families.map((f) => f.id) as [string, ...string[]];
const SERVABLE_RECIPES = CATALOG.recipes.filter((r) => r.serve).map((r) => r.id) as [string, ...string[]];
const familyOfModel = new Map(CATALOG.models.map((m) => [m.id, m.family]));
const familyOfRecipe = new Map(CATALOG.recipes.map((r) => [r.id, r.family]));

export const ModelRefZ = z
  .object({
    id: z.enum(MODEL_IDS).meta({ "x-dynamic": "model_ids" }).describe("Model id the pool serves (e.g. fasth3, ltx25-distill-sol); the worker config's [[models]] id."),
    family: z.enum(FAMILY_IDS).meta({ "x-dynamic": "families" }).describe("Model family (h3, ltx2, wan): the model's family in the catalog."),
    recipe: z
      .enum(SERVABLE_RECIPES)
      .meta({ "x-dynamic": "recipes" })
      .describe("A tier alias (h3-max, h3-turbo, ltx-pro, wan-turbo, …) or a catalog recipe id, of the model's family. Only recipes the fv-serve CUDA catalog serves."),
  })
  .strict()
  .superRefine((m, ctx) => {
    if (familyOfModel.get(m.id) !== m.family) ctx.addIssue({ code: "custom", path: ["family"], message: `${m.id} is a ${familyOfModel.get(m.id)} model` });
    if (familyOfRecipe.get(m.recipe) !== m.family) ctx.addIssue({ code: "custom", path: ["recipe"], message: `${m.recipe} is a ${familyOfRecipe.get(m.recipe)} recipe, not ${m.family}` });
  })
  .describe("A model the pool serves: its family (the edge's family object) and recipe.");

export const PoolSpecZ = z
  .object({
    id: nameZ("pool", "Pool id"),
    variant: z.enum(VARIANTS).meta({ "x-dynamic": "variants" }).describe("Image variant (docs/serve/images.md): h3-turbo, h3-max, ltx, wan, wan5b, sfwan (GPU), cpu (CPU, the fake engine). The pool presets reuse these images with an inline config_toml."),
    count: z.number().int().min(0).max(8).meta({ "x-unit": "pods" }).describe("Worker pods in this pool. Change a running cluster's count with Scale."),
    compute: z.enum(["GPU", "CPU"]).describe("GPU pod, or CPU pod (the cpu variant: fake engine, tests)."),
    config: configPath.optional().describe("Worker config inside the image (e.g. /etc/fv/runpod.toml). Exactly one of config / config_toml."),
    config_toml: z.string().min(1).max(32768).optional().meta({ "x-ui": "textarea" }).describe("An inline worker config (sent as FV_WORKER_TOML_B64) instead of a file in the image."),
    gpu_types: uniq(gpuType).min(1).max(12).optional().describe("Runpod GPU type ids to try, in order. Default: the region's (RTX PRO 6000 in eu)."),
    regions: uniq(region).min(1).optional().describe("Regions to try, in order. Default: the cluster's."),
    cpu_flavors: uniq(cpuFlavor).min(1).optional().describe("CPU pods: flavors to try, in order."),
    vcpu: z.literal(CPU_VCPUS).optional().meta({ "x-unit": "vCPU" }).describe("CPU pods: vCPUs (a Runpod CPU instance size: 2, 4, 8, 16, 32)."),
    container_disk_gb: z.number().int().min(5).max(500).optional().meta({ "x-unit": "GB" }).describe("Container disk in GB (default 40 GPU, 10 CPU)."),
    volume: z.boolean().optional().describe("Mount the region's network volume at /workspace (weights). Default: on for GPU pods."),
    image: imageRef.optional().describe("Override: an image reference for this pool (resolved to a digest at start)."),
    models: z.array(ModelRefZ).max(16).optional().describe("The models this pool serves (their families and recipes)."),
    fake_models: uniq(z.enum(FAKE_MODELS).meta({ "x-dynamic": "fake_models" })).min(1).optional().describe("Fake-engine model ids (fake-wan, fake-h3-turbo, …) the pool serves."),
    max_queued: z.number().int().min(0).max(10000).optional().meta({ "x-unit": "jobs" }).describe("Admission: jobs of a model of this pool waiting in its family object (FV_DISPATCH_MAX_QUEUED)."),
    job_timeout_s: z.number().int().min(10).max(86400).optional().meta({ "x-unit": "s" }).describe("Job timeout in seconds."),
    stale_after_s: z.number().int().min(10).max(86400).optional().meta({ "x-unit": "s" }).describe("A running job without a worker heartbeat for this long is lost (at most the job timeout)."),
    family: z
      .enum(EDGE_FAMILIES)
      .optional()
      .describe("control_plane = edge: the family Durable Object every model of this pool queues on. Default: from each model's family (h3 → h3, ltx2 → ltx, causal wan → sfwan, wan → wan; fake models: fake)."),
    provider: provider.optional(),
    provider_gpu: providerGpu.optional(),
    provider_region: providerRegion.optional(),
    weights_source: weightsSource.optional(),
    hub_download_approved: z.boolean().optional().describe("weights_source hub: the owner approved the Hub download at boot (also needs the Worker's FV_HUB_DOWNLOADS_APPROVED=1)."),
  })
  .strict()
  .superRefine((p, ctx) => {
    const add = (path: (string | number)[], message: string) => ctx.addIssue({ code: "custom", path, message });
    if (!!p.config === !!p.config_toml) add(["config"], p.config ? "one of config / config_toml, not both" : "config or config_toml is required");
    if (!(p.models?.length || p.fake_models?.length)) add(["models"], "models or fake_models (what the pool serves) is required");
    providerRules(p, (path, message) => add([path], message));
    // GMI / Brev machines are GPU machines; the cpu variant (fake engine) runs on one as a smoke test.
    const other = !!p.provider && p.provider !== "runpod";
    if (other) {
      /* providerRules covers it */
    } else if (p.compute === "CPU") {
      if (p.gpu_types) add(["gpu_types"], "a CPU pool takes no gpu_types");
      if (p.variant !== "cpu" && !p.image) add(["variant"], "CPU pods run the cpu variant (the CUDA variants need a GPU)");
    } else {
      if (p.cpu_flavors) add(["cpu_flavors"], "a GPU pool takes no cpu_flavors");
      if (p.vcpu !== undefined) add(["vcpu"], "a GPU pool takes no vcpu");
      if (p.variant === "cpu") add(["variant"], "the cpu variant runs on CPU pods (compute: CPU)");
    }
    if (p.variant === "cpu" && p.models?.length) add(["models"], "the cpu variant runs the fake engine: fake_models, not models");
    if (p.stale_after_s !== undefined && p.job_timeout_s !== undefined && p.stale_after_s > p.job_timeout_s) add(["stale_after_s"], `at most job_timeout_s (${p.job_timeout_s})`);
    const ids = new Set<string>();
    (p.models || []).forEach((m, i) => {
      if (ids.has(m.id)) add(["models", i, "id"], `${m.id} twice`);
      ids.add(m.id);
    });
  })
  .describe("A pool of worker pods.");

export const ClusterSpecZ = z
  .object({
    name: nameZ("cluster", "Cluster name"),
    image: z
      .object({
        channel: channel.optional().describe("Release channel (stable, latest, …): per-variant images <variant>-<channel>."),
        sha: sha.optional().describe("A git commit: per-variant images <variant>-sha-<sha7>."),
        ref: imageRef.optional().describe("One (all-in-one) image reference for every pod."),
      })
      .strict()
      .refine((i) => [i.channel, i.sha, i.ref].filter(Boolean).length === 1, { message: "exactly one of channel, sha, ref" })
      .describe("Which build the pods run. Resolved to digests at start."),
    regions: uniq(region).min(1).describe("Regions to place GPU workers in, in order."),
    control_plane: z
      .enum(["edge", "direct"])
      .describe("edge (default): the edge Worker (EDGE_URL) is the only entry point and every worker is an API front behind it (docs/serve/edge-control-plane.md); one edge cluster runs at a time. direct: clients call each worker with an API key (docs/control/gateway-less-auth.md). The gateway pod is retired."),
    auth: z.enum(["keys", "none"]).describe("direct: the workers' client auth (FV_AUTH_MODE). An edge cluster uses the edge's own setting."),
    pools: z.array(PoolSpecZ).max(16).describe("Worker pools."),
    cap_s: z.number().int().min(300).max(7 * 86400).meta({ "x-unit": "s" }).describe("Backstop: every pod is deleted at start + cap_s seconds (Extend moves it). 5 min to 7 days."),
    min_balance: z.number().min(8).max(10000).meta({ "x-unit": "$" }).describe("Each edge worker's watchdog deletes its pod below this account balance ($, at least the $8 floor of CLAUDE.md)."),
    balance_floor: z.number().min(8).max(10000).meta({ "x-unit": "$" }).describe("Refuse start / extend / scale-up when the projected balance at the deadline would be below this ($); the cron stops the cluster below it."),
    min_start: z.number().min(8).max(10000).meta({ "x-unit": "$" }).describe("Refuse to start below this balance ($; at least the balance floor)."),
    max_gpu_dph: z.number().min(0.1).max(50).meta({ "x-unit": "$/hr" }).describe("A GPU pod costing more than this $/hr is deleted right after create (so it must be above the price of a GPU type the pools may use)."),
    auto_stop_idle_min: z.number().int().min(5).max(1440).nullable().optional().meta({ "x-unit": "min" }).describe("Remove a worker idle (GPU < threshold, no jobs) this many minutes. null: the account policy."),
    log_shipping: z.boolean().describe("Pods ship their logs to the controller (FV_LOG_SHIP_*)."),
    log_level: z.enum(LOG_LEVELS).optional().describe("Most verbose level shipped (default info)."),
  })
  .strict()
  .superRefine((s, ctx) => {
    const seen = new Set<string>();
    s.pools.forEach((p, i) => {
      if (seen.has(p.id)) ctx.addIssue({ code: "custom", message: `pool ${p.id} twice`, path: ["pools", i, "id"] });
      seen.add(p.id);
    });
    if (s.min_start < s.balance_floor) ctx.addIssue({ code: "custom", path: ["min_start"], message: `at least the balance floor ($${s.balance_floor})` });
    if (s.control_plane === "direct") s.pools.forEach((p, i) => p.family && ctx.addIssue({ code: "custom", path: ["pools", i, "family"], message: "family is for edge clusters (control_plane: edge)" }));
  })
  .describe("A cluster on Runpod (docs/control/README.md §4).");

export const AttributionZ = z
  .array(
    z
      .object({
        prefix: z.string().min(1).max(60).regex(/^\S+$/, { message: "no spaces" }).describe("Pod name prefix."),
        owner: z.string().regex(/^[a-z][a-z0-9:._-]{0,60}$/, { message: "lower-case letters, digits and : . _ -" }).meta({ "x-rule": "e.g. external:build-pod (lower-case, : . _ -)" }).describe("Owner label for pods whose name starts with the prefix (e.g. external:build-pod)."),
      })
      .strict(),
  )
  .max(50)
  .superRefine((a, ctx) => {
    const seen = new Set<string>();
    a.forEach((r, i) => {
      if (seen.has(r.prefix)) ctx.addIssue({ code: "custom", path: [i, "prefix"], message: `prefix ${r.prefix} twice` });
      seen.add(r.prefix);
    });
  })
  .describe("External pods by name prefix; the first match wins.");

export const PoliciesZ = z
  .object({
    idle_gpu_pct: z.number().min(0).max(100).meta({ "x-unit": "%" }).describe("A GPU pod is idle below this utilisation (%) with no running jobs."),
    idle_min: z.number().min(1).max(1440).meta({ "x-unit": "min" }).describe("Alert when a GPU pod has been idle this many minutes."),
    auto_stop_idle: z.boolean().describe("Auto-action: remove idle workers of controller clusters (never external pods)."),
    auto_stop_idle_min: z.number().min(5).max(1440).meta({ "x-unit": "min" }).describe("…after this many idle minutes (at least idle_min: the alert comes first)."),
    cluster_dph_max: z.number().min(0).max(1000).meta({ "x-unit": "$/hr" }).describe("Alert when a cluster costs more than this $/hr."),
    daily_spend_max: z.number().min(0).max(100000).meta({ "x-unit": "$" }).describe("Alert when today's account spend exceeds this ($)."),
    balance_margin: z.number().min(0).max(10000).meta({ "x-unit": "$" }).describe("Alert when the balance is within this many $ of the floor."),
    stop_on_floor: z.boolean().describe("Auto-action (default on): stop controller clusters below the balance floor."),
    pod_down_min: z.number().min(1).max(1440).meta({ "x-unit": "min" }).describe("Alert when a controller pod has not answered this many minutes."),
    attribution: AttributionZ,
    build_pod_backstop: z.boolean().describe("Auto-action (default on): stop the shared build pod (external:build-pod) when its own self-stop did not happen."),
    build_pod_max_h: z.number().min(0).max(72).meta({ "x-unit": "h" }).describe("…when it has been up this many hours (0: off; its own cap is 8 h + 30 min grace)."),
    build_pod_idle_grace_min: z.number().min(0).max(1440).meta({ "x-unit": "min" }).describe("…or idle (no jobs, per its /healthz) this many minutes past its own idle stop."),
    brev_park_max: z.number().int().min(0).max(20).describe("NVIDIA Brev keep-on-stop: at most this many parked instances (stopped, weights kept, storage billed); beyond it the oldest is deleted. 0: delete on stop."),
    brev_park_max_days: z.number().min(0).max(90).meta({ "x-unit": "days" }).describe("…a parked instance is deleted after this many days (only ours: fv- named and recorded)."),
    brev_park_delete_failed: z.boolean().describe("Auto-action (default off): delete a parked instance whose restart failed or timed out, instead of holding it for the owner."),
  })
  .strict()
  .superRefine((p, ctx) => {
    if (p.auto_stop_idle_min < p.idle_min) ctx.addIssue({ code: "custom", path: ["auto_stop_idle_min"], message: `at least idle_min (${p.idle_min}): the idle alert comes before the stop` });
  })
  .describe("Alert thresholds and auto-actions (docs/control/README.md §8).");

/** An env set as the editor sees it: KEY -> {value, secret}. A secret's value is never sent back: `value` is null
 *  and a new value goes in `set` (write-only). */
export const EnvVarZ = z
  .object({
    value: z.string().max(32768).nullable().describe("The value (null for a secret: it is never displayed)."),
    secret: z.boolean().describe("Sealed in D1, masked everywhere, written only through `set`."),
    set: z.string().max(32768).optional().meta({ "x-secret": true }).describe("Write-only: a new value for a secret."),
  })
  .strict();
export const envKeyZ = z
  .string()
  .regex(ENV_KEY_RE, { message: ENV_KEY_MESSAGE })
  .refine((k) => !isReserved(k), { message: "set by the controller: it cannot be overridden" })
  .meta({ "x-dynamic": "env_keys", "x-rule": `${ENV_KEY_RULE}; not a controller key` });
export const EnvSetZ = z
  .record(envKeyZ, EnvVarZ)
  .superRefine((d, ctx) => {
    for (const [k, v] of Object.entries(d)) {
      const why = !v.secret && typeof v.value === "string" ? envValueProblem(k, v.value) : null;
      if (why) ctx.addIssue({ code: "custom", path: [k, "value"], message: why });
    }
  })
  .describe("Environment variables at one level (account, cluster, pool or pod).");

export const TokenCreateZ = z
  .object({
    name: z.string().regex(TOKEN_NAME_RE, { message: TOKEN_NAME_RULE }).meta({ "x-rule": TOKEN_NAME_RULE, "x-name": "token" }).describe("A name for the token, unique among the active tokens."),
    scope: z.enum(TOKEN_SCOPES).describe("read: GET only; admin: everything but minting tokens; ci: only /api/ci/* (a GitHub workflow secret)."),
    ttl_days: z.number().int().min(1).max(365).optional().meta({ "x-unit": "days" }).describe("Expiry in days (default 90)."),
  })
  .strict();

export const ReleaseDispatchZ = z
  .object({
    action: z.enum(["promote", "rollback"]).describe("promote: point the channel at a build; rollback: back to an earlier release."),
    target: z
      .string()
      .regex(/^([0-9a-f]{7,40}|sha256:[0-9a-f]{64}|[a-z0-9][a-z0-9._-]{0,60})$/, { message: "a git sha (7-40 hex), a digest (sha256:<64 hex>) or a tag" })
      .optional()
      .meta({ "x-dynamic": "shas", "x-rule": "a git sha, sha256:<digest> or a tag" })
      .describe("promote: a git sha, a digest or a tag."),
    channel: channel.optional().describe("Channel (stable: what deploys and the templates follow)."),
    to: z.string().regex(/^\d+$/, { message: "a release id" }).optional().meta({ "x-dynamic": "releases" }).describe("rollback: a release id (default: the previous release of the channel)."),
    notes: z.string().max(200).optional().describe("Why (kept with the release record)."),
    templates: z.boolean().optional().describe("Also update the Runpod templates that follow the channel (default on)."),
    allow_partial: z.boolean().optional().describe("Promote even when some variants have no image at the target."),
    dry_run: z.boolean().optional().describe("Show what would change; change nothing."),
  })
  .strict()
  .superRefine((d, ctx) => {
    if (d.action === "promote" && !d.target) ctx.addIssue({ code: "custom", path: ["target"], message: "promote needs a target" });
    if (d.action === "promote" && d.to) ctx.addIssue({ code: "custom", path: ["to"], message: "`to` is for rollback" });
  });

/** A standalone pod launch (POST /api/standalone; standalone.ts). */
export const StandaloneLaunchZ = z
  .object({
    name: nameZ("cluster", "Pod name (unique among clusters and standalone pods)"),
    preset: z
      .enum(POOL_PRESET_IDS, { error: (i) => `no pool preset ${JSON.stringify(i.input)} (GET /api/templates: ${POOL_PRESET_IDS.join(", ")})` })
      .optional()
      .meta({ "x-dynamic": "pool_presets" }).describe("A pool preset (variant, config and models); none: variant + config."),
    variant: z.enum(VARIANTS).optional().meta({ "x-dynamic": "variants" }).describe("Without a preset: the image variant."),
    config: configPath.optional().describe("Without a preset: the worker config in the image."),
    config_toml: z.string().min(1).max(32768).optional().meta({ "x-ui": "textarea" }).describe("Without a preset: an inline worker config."),
    models: z.array(ModelRefZ).max(16).optional().describe("Without a preset: the models it serves."),
    fake_models: uniq(z.enum(FAKE_MODELS).meta({ "x-dynamic": "fake_models" })).min(1).optional().describe("Without a preset: fake-engine models (the cpu variant)."),
    channel: channel.optional().describe("Image: a release channel (default stable)."),
    sha: sha.optional().describe("Image: a git commit."),
    image: imageRef.optional().describe("Image: an image reference."),
    compute: z.enum(["GPU", "CPU"]).optional().describe("GPU or CPU pod (default: the preset's, else GPU)."),
    gpu_types: uniq(gpuType).min(1).max(12).optional().describe("GPU types in placement order (default: the region's)."),
    gpu_type: gpuType.optional().describe("One GPU type (shorthand for gpu_types)."),
    cpu_flavors: uniq(cpuFlavor).min(1).optional().describe("CPU pods: flavors in order."),
    vcpu: z.literal(CPU_VCPUS).optional().meta({ "x-unit": "vCPU" }).describe("CPU pods: vCPUs."),
    region: region.optional().describe("Region (eu)."),
    dc: z.enum(AVAILABLE_REGIONS.map((r) => REGIONS[r].dc) as [string, ...string[]], { error: () => EU_ONLY }).optional().meta({ "x-dynamic": "data_centers" }).describe("Or its data centre (EUR-IS-1)."),
    volume: z.boolean().optional().describe("Mount the region's weights volume at /workspace (GPU default on, CPU off)."),
    container_disk_gb: z.number().int().min(5).max(500).optional().meta({ "x-unit": "GB" }).describe("Container disk."),
    env: z
      .record(envKeyZ, z.union([z.string().max(32768), z.object({ value: z.string().max(32768), secret: z.boolean().optional() }).strict()]))
      .optional()
      .describe("Env of the pod: {KEY: \"value\"} or {KEY: {value, secret}}. Reserved controller keys are refused."),
    deadline_min: z.number().int().min(5).max(10080).optional().meta({ "x-unit": "min" }).describe("Backstop: the pod is deleted this many minutes after the start (default 60)."),
    idle_stop_min: z.number().int().min(5).max(1440).nullable().optional().meta({ "x-unit": "min" }).describe("Stop after this many idle minutes; null: the account policy."),
    max_gpu_dph: z.number().min(0.1).max(50).optional().meta({ "x-unit": "$/hr" }).describe("Delete the pod right after create if its GPU costs more than this (default 3.6)."),
    min_balance: z.number().min(8).max(10000).optional().meta({ "x-unit": "$" }),
    balance_floor: z.number().min(8).max(10000).optional().meta({ "x-unit": "$" }),
    min_start: z.number().min(8).max(10000).optional().meta({ "x-unit": "$" }),
    auth: z.enum(["keys", "none"]).optional().describe("Client auth (default keys)."),
    log_level: z.enum(LOG_LEVELS).optional().describe("Most verbose level shipped."),
    start: z.boolean().optional().describe("Start it now (default); false: only define it."),
    skip_image_check: z.boolean().optional().describe("Skip the image preflight (not recommended)."),
    provider: provider.optional(),
    provider_gpu: providerGpu.optional(),
    provider_region: providerRegion.optional(),
    weights_source: weightsSource.optional(),
    weights_download_approved: z.boolean().optional().describe("weights_source hub: the owner approves this pod's Hub download at boot (the Worker's FV_HUB_DOWNLOADS_APPROVED=1 is needed too)."),
  })
  .strict()
  .superRefine((x, ctx) => {
    const add = (path: string, message: string) => ctx.addIssue({ code: "custom", path: [path], message });
    providerRules(x, add, x.gpu_type ? "gpu_type" : "gpu_types");
    if (x.provider && x.provider !== "runpod" && x.gpu_type) add("gpu_type", `gpu_type is a Runpod GPU type id: provider ${x.provider} takes provider_gpu`);
    if (x.provider && x.provider !== "runpod" && (x.region || x.dc)) add(x.region ? "region" : "dc", `region / dc are Runpod's: provider ${x.provider} takes provider_region (gmi)`);
    if (x.weights_download_approved && x.weights_source !== "hub") add("weights_download_approved", "only with weights_source hub");
    if ([x.channel, x.sha, x.image].filter(Boolean).length > 1) add("channel", "at most one of channel, sha, image");
    if (!x.preset && !x.variant) add("variant", "a preset or variant is required");
    if (x.preset && (x.variant || x.config || x.config_toml || x.models || x.fake_models)) add("preset", "a preset brings its variant, config and models: leave those out");
    if (!x.preset && x.variant && !x.config && !x.config_toml) add("config", "config or config_toml (the worker config)");
    if (x.config && x.config_toml) add("config", "one of config / config_toml");
    if (x.gpu_types && x.gpu_type) add("gpu_type", "gpu_types or gpu_type, not both");
    if (x.compute === "CPU" && (x.gpu_types || x.gpu_type)) add("gpu_types", "a CPU pod takes no GPU types");
    if (x.compute === "GPU" && (x.cpu_flavors || x.vcpu !== undefined)) add("cpu_flavors", "a GPU pod takes no cpu_flavors / vcpu");
    if (x.region && x.dc && REGIONS[x.region].dc !== x.dc) add("dc", `region ${x.region} is ${REGIONS[x.region].dc}`);
    if (x.min_start !== undefined && x.balance_floor !== undefined && x.min_start < x.balance_floor) add("min_start", "at least balance_floor");
    for (const [k, v] of Object.entries(x.env || {})) {
      const s = typeof v === "string" ? v : v.secret ? null : v.value;
      const why = s !== null ? envValueProblem(k, s) : null;
      if (why) ctx.addIssue({ code: "custom", path: ["env", k], message: why });
    }
  })
  .describe("A standalone pod (docs/control/standalone-pods.md).");

/** The managed build pods' policy (PUT /api/build-pods/policy; buildpods.ts normalizePolicy fills the defaults). */
export const BuildPodsPolicyZ = z
  .object({
    enabled: z.boolean().describe("fv-control creates and manages build pods."),
    max_pods: z.number().int().min(0).max(8).meta({ "x-unit": "pods" }).describe("Running (or being created) at once."),
    max_dph_per_pod: z.number().min(0.05).max(5).meta({ "x-unit": "$/hr" }).describe("Refuse a size costing more than this."),
    daily_usd_max: z.number().min(0).max(500).meta({ "x-unit": "$" }).describe("Build pods' spend today above this: `up` refuses and alerts."),
    balance_margin: z.number().min(0).max(1000).meta({ "x-unit": "$" }).describe("`up` needs balance ≥ the account floor + this."),
    flavors: uniq(cpuFlavor).min(1).describe("Runpod CPU flavors, preferred first."),
    vcpus: uniq(z.literal(BUILD_VCPUS)).min(1).describe("Sizes, preferred first (vCPUs)."),
    disk_gb: z.number().int().min(20).max(500).meta({ "x-unit": "GB" }).describe("Container disk of the first size."),
    disk_gb_fallback: z.number().int().min(20).max(500).meta({ "x-unit": "GB" }).describe("Container disk for smaller sizes (at most disk_gb)."),
    regions: uniq(z.string().regex(/^[A-Z]{2,4}(-[A-Z0-9]+)*$/, { message: "a data centre or its prefix: EU, EUR-IS, US-CA-2" }).meta({ "x-dynamic": "dc_prefixes" })).describe("Preferred data centres or prefixes (EU, EUR-IS, US-CA-2 …)."),
    regions_only: z.boolean().describe("Only the preferred regions (else they go first)."),
    volumes: z
      .partialRecord(z.enum(RUNPOD_DATA_CENTERS, { error: (i) => enumMessage("a Runpod data centre", i.input, RUNPOD_DATA_CENTERS) }).meta({ "x-dynamic": "data_centers" }), z.string().regex(/^[a-z0-9]{6,40}$/, { message: "a Runpod network volume id" }))
      .describe("Data centre → a build-cache network volume id there (optional)."),
    idle_min: z.number().min(5).max(240).meta({ "x-unit": "min" }).describe("The pod stops itself after this many idle minutes."),
    max_h: z.number().min(0.5).max(24).meta({ "x-unit": "h" }).describe("Its cap: it stops after this many hours up."),
    max_grace_min: z.number().min(0).max(120).meta({ "x-unit": "min" }).describe("…plus this much for running jobs."),
    evict_hours: z.number().min(0).max(72).meta({ "x-unit": "h" }).describe("Delete stopped pods after this many hours (0: never)."),
    backstop_margin_min: z.number().min(5).max(180).meta({ "x-unit": "min" }).describe("The controller stops a pod this long after its own limits."),
    runner: z.boolean().describe("Register each shared pod as a GitHub runner."),
    labels: uniq(z.string().regex(/^[A-Za-z0-9_.-]{1,40}$/, { message: "1-40 of letters, digits . _ -" })).min(1).max(10).describe("Runner labels (+ fv-build-<region>)."),
    wake_on_queue: z.boolean().describe("Queued GitHub jobs that need labels[0] wake a pod."),
    wake_workflows: uniq(z.string().regex(/^[A-Za-z0-9_.\/-]{1,100}$/, { message: "a workflow file or name" })).max(20).describe("A queued or running run of these workflows wakes a pod too."),
    server_ref: z.string().regex(/^[A-Za-z0-9._\/-]{1,200}$/, { message: "a git ref" }).describe("The git ref of the build pod server."),
    image: z
      .union([z.literal(""), z.string().regex(/^([a-z0-9.-]+\/)?[a-z0-9._\/-]+(:[A-Za-z0-9._-]+|@sha256:[0-9a-f]{64})$/, { message: "an image with a tag or digest" })])
      .describe("\"\": the BASE_IMAGE_TAG pin of build-pod.sh at server_ref; else an image with a tag or digest."),
    cache: z
      .object({
        r2_bucket: z.string().regex(/^[a-z0-9][a-z0-9-]{1,62}$/, { message: "an R2 bucket name (3-63 lower-case letters, digits, -)" }).describe("The sccache bucket."),
        r2_endpoint: z.union([z.literal(""), z.string().regex(/^https:\/\/[A-Za-z0-9.-]+$/, { message: "https://host" })]).describe("\"\": https://<CF_ACCOUNT_ID>.r2.cloudflarestorage.com."),
      })
      .strict(),
  })
  .strict()
  .superRefine((p, ctx) => {
    if (p.disk_gb_fallback > p.disk_gb) ctx.addIssue({ code: "custom", path: ["disk_gb_fallback"], message: `at most disk_gb (${p.disk_gb})` });
  })
  .describe("Build pods managed by fv-control (docs/dev/build-pods-fv-control.md).");

/** The serverless policy (PUT /api/serverless/policy). */
export const SlsPolicyZ = z
  .object({
    balance_margin: z.number().min(0).max(1000).meta({ "x-unit": "$" }).describe("Create / scale-up / invoke need balance ≥ the account floor + this ($)."),
    max_endpoints: z.number().int().min(0).max(50).meta({ "x-unit": "endpoints" }).describe("Live endpoints at once."),
    max_workers: z.number().int().min(0).max(200).meta({ "x-unit": "workers" }).describe("Sum of workers_max over the live endpoints."),
    scale0_on_floor: z.boolean().describe("Below the floor the tick scales every endpoint to 0/0."),
  })
  .strict()
  .describe("Limits of the serverless endpoints fv-control manages (docs/control/serverless.md).");

// ---- the small operation forms (dialogs on the cluster, standalone and serverless pages)
export const ExtendZ = z.object({ minutes: z.number().int().min(1).max(10080).meta({ "x-unit": "min" }).describe("Move the deadline this many minutes later (1 min to 7 days).") }).strict();
export const ScaleZ = z
  .object({
    pool: nameZ("pool", "Pool").meta({ "x-dynamic": "pools" }),
    count: z.number().int().min(0).max(8).meta({ "x-unit": "pods" }).describe("Worker pods (0-8)."),
  })
  .strict();
export const RollZ = z
  .object({
    target: z
      .string()
      .min(1)
      .refine((t) => CHANNEL_RE.test(t) || SHA_RE.test(t) || IMAGE_REF_RE.test(t), { message: "a channel, a git sha (7-40 hex) or an image reference" })
      .meta({ "x-dynamic": "channels", "x-rule": "a channel (stable, latest), a git sha or an image reference" })
      .describe("What the pools roll to."),
    pools: z.array(nameZ("pool", "Pool")).optional().describe("Which pools (default: every pool)."),
  })
  .strict();
export const MintKeyZ = z.object({ name: z.string().regex(KEY_NAME_RE, { message: KEY_NAME_RULE }).meta({ "x-rule": KEY_NAME_RULE }).describe("A name for the user API key.") }).strict();
export const SlsScaleZ = z
  .object({
    workers_min: z.number().int().min(0).max(4).optional().meta({ "x-unit": "workers" }).describe("Always-on workers."),
    workers_max: z.number().int().min(0).max(8).optional().meta({ "x-unit": "workers" }).describe("Most workers at once."),
  })
  .strict()
  .superRefine((s, ctx) => {
    if (s.workers_min === undefined && s.workers_max === undefined) ctx.addIssue({ code: "custom", path: ["workers_max"], message: "workers_min and/or workers_max" });
    if (s.workers_min !== undefined && s.workers_max !== undefined && s.workers_min > s.workers_max) ctx.addIssue({ code: "custom", path: ["workers_min"], message: "at most workers_max" });
  });

const csvOf = (xs: readonly string[]) => new RegExp(`^(${xs.join("|")})(,(${xs.join("|")}))*$`);
// ---- cancelling jobs (docs/control/serverless.md "Cancel and purge", docs/control/README.md "Jobs")
/** A job id as the APIs mint them: a Runpod job id, fv-serve's internal uuid, fvjob_…, a fal uuid, MiniMax digits, video_gen_…. */
export const JOB_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$/; // `\-`: valid in an HTML pattern (the v flag) too
const JOB_ID_RULE = "letters, digits, _ . : -; at most 128 characters";
const jobId = (what: string) => z.string().regex(JOB_ID_RE, { message: JOB_ID_RULE }).meta({ "x-rule": JOB_ID_RULE }).describe(what);
/** The fv-serve APIs whose job ids fv-control can cancel through a serverless queue job (fal needs the app path, LTX has no cancel). */
export const SLS_FV_APIS = ["native", "openai_videos", "fastwan", "minimax_v2"] as const;
/** fv-serve job statuses (the jobs D1 `status`). */
export const JOB_STATUSES = ["queued", "running", "succeeded", "failed", "cancelled"] as const;
export const SlsCancelZ = z
  .object({
    job: jobId("The Runpod job id (any job of this endpoint, also one fv-control did not submit), or the number of one of fv-control's test invokes."),
    fv_job: jobId("The fv-serve job the queue job created, when fv-control cannot find it in the job's output (a job submitted elsewhere).").optional(),
    fv_api: z.enum(SLS_FV_APIS).optional().describe("The API that owns fv_job: native DELETE /fv/v1/jobs/{id} (default), openai_videos DELETE /v1/videos/{id}, fastwan DELETE /video/{id}, minimax_v2 DELETE /v2/video_generation/{id}."),
    stop_fv_job: z.boolean().optional().describe("Also send the fv-serve cancel as a queue job (kind http) when the job already reached a worker and a worker is up (default on)."),
  })
  .strict()
  .describe("Cancel one job of a serverless endpoint (POST /api/serverless/:id/jobs/:job/cancel).");
export const SlsPurgeZ = z
  .object({
    confirm: z.string().min(1).max(NAME_MAX).describe("The endpoint's name, typed to confirm: every queued job is dropped (running ones are not)."),
    expected: z.number().int().min(0).max(100000).optional().meta({ "x-unit": "jobs" }).describe("The queued count you saw; refused when the queue grew past it since (default: no check)."),
  })
  .strict()
  .describe("Purge a serverless endpoint's queue (POST /api/serverless/:id/purge).");
export const JobCancelZ = z
  .object({
    job: jobId("A job id from any API (fvjob_…, a fal request id, a MiniMax task id, video_gen_…, or fv-serve's internal uuid)."),
    cluster: z.string().regex(/^[A-Za-z0-9_.\-]{1,80}$/, { message: "a cluster or standalone pod name or id" }).optional().meta({ "x-dynamic": "clusters" }).describe("Only when the id is ambiguous: the cluster or standalone pod that ran it."),
  })
  .strict()
  .describe("Cancel one fv-serve job of a cluster or standalone pod (POST /api/jobs/:job/cancel).");
export const JobsQueryZ = z
  .object({
    status: z.string().regex(csvOf(JOB_STATUSES), { message: `statuses: ${JOB_STATUSES.join(", ")}` }).optional().meta({ "x-enum-list": JOB_STATUSES }).describe("Statuses, comma-separated (default: every status)."),
    pool: z.string().regex(/^[A-Za-z0-9_.\-]{1,80}$/, { message: "a pool id" }).optional().meta({ "x-dynamic": "pools" }).describe("Only this pool's pods."),
    pod: z.string().regex(/^[A-Za-z0-9]{1,40}$/, { message: "a pod id" }).optional().describe("Only this pod."),
    limit: z.string().regex(/^\d{1,3}$/, { message: "1-500 jobs" }).optional().describe("At most this many jobs (default 100, at most 500)."),
  })
  .strict()
  .describe("The Jobs view's filters (GET /api/clusters/:id/jobs).");
export const JobsCancelQueuedZ = z
  .object({
    pool: z.string().regex(/^[A-Za-z0-9_.\-]{1,80}$/, { message: "a pool id" }).optional().meta({ "x-dynamic": "pools" }).describe("Only this pool's queued jobs (default: every pool)."),
    max: z.number().int().min(1).max(200).optional().meta({ "x-unit": "jobs" }).describe("Cancel at most this many (default 100)."),
  })
  .strict()
  .describe("Cancel every queued job of a cluster or standalone pod (POST /api/clusters/:id/jobs/cancel-queued).");

/** A time in the log explorer: now, unix s / ms, an ISO date (UTC unless it says), HH:MM (today) or relative (15m, 2h, 7d). */
export const TIME_RE = /^(now|-?\d+(\.\d+)?\s*(s|m|min|h|d|w)|\d{10,13}|\d{4}-\d{2}-\d{2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?|\d{1,2}:\d{2}(:\d{2})?)$/;
const TIME_RULE = "UTC 2026-10-06 12:00, 12:00 (today), unix ms, or 15m / 2h / 7d (ago)";
/** The log explorer's filters (GET /api/logs/query; logquery.ts parseLogQuery checks them with this schema). */
export const LogQueryZ = z
  .object({
    q: z.string().max(500).optional().describe("Search text (a regular expression with re=1, at most 300 characters)."),
    re: z.enum(["0", "1", "true", "false"]).optional().describe("1: q is a regular expression."),
    cs: z.enum(["0", "1", "true", "false"]).optional().describe("1: case sensitive."),
    src: z.string().regex(csvOf(LOG_SOURCES), { message: `sources: ${LOG_SOURCES.join(", ")}` }).optional().meta({ "x-enum-list": LOG_SOURCES }).describe("Sources, comma-separated."),
    sources: z.string().regex(csvOf(LOG_SOURCES), { message: `sources: ${LOG_SOURCES.join(", ")}` }).optional(),
    lv: z.string().regex(csvOf(LOG_LEVELS), { message: `levels: ${LOG_LEVELS.join(", ")}` }).optional().meta({ "x-enum-list": LOG_LEVELS }).describe("Levels, comma-separated."),
    levels: z.string().regex(csvOf(LOG_LEVELS), { message: `levels: ${LOG_LEVELS.join(", ")}` }).optional(),
    level: z.enum(LOG_LEVELS).optional().describe("This level and above."),
    cluster: z.string().regex(/^[A-Za-z0-9_:.-]{1,80}$/, { message: "invalid" }).optional().meta({ "x-dynamic": "clusters" }),
    pool: z.string().regex(/^[A-Za-z0-9_:.-]{1,80}$/, { message: "invalid" }).optional().meta({ "x-dynamic": "pools" }),
    pod: z.string().regex(/^[A-Za-z0-9_:.-]{1,80}$/, { message: "invalid" }).optional(),
    op: z.string().regex(/^[A-Za-z0-9_:.-]{1,80}$/, { message: "invalid" }).optional(),
    from: z.string().regex(TIME_RE, { message: TIME_RULE }).optional().meta({ "x-rule": TIME_RULE }),
    since: z.string().regex(TIME_RE, { message: TIME_RULE }).optional(),
    to: z.string().regex(TIME_RE, { message: TIME_RULE }).optional().meta({ "x-rule": TIME_RULE }),
    until: z.string().regex(TIME_RE, { message: TIME_RULE }).optional(),
    order: z.enum(["asc", "desc"]).optional(),
    limit: z.string().regex(/^\d{1,5}$/, { message: "a number of lines" }).optional().describe("1-1000 lines a page."),
    ctx: z.number().int().min(0).max(200).optional().meta({ "x-unit": "lines" }).describe("Context lines before and after (the explorer's view)."),
  })
  .describe("Log explorer filters (docs/control/logs.md).");

export const SCHEMAS = {
  "cluster-spec": ClusterSpecZ,
  pool: PoolSpecZ,
  policies: PoliciesZ,
  attribution: AttributionZ,
  env: EnvSetZ,
  "token-create": TokenCreateZ,
  "release-dispatch": ReleaseDispatchZ,
  "serverless-endpoint": EndpointSpecZ,
  "serverless-policy": SlsPolicyZ,
  "serverless-scale": SlsScaleZ,
  "standalone-launch": StandaloneLaunchZ,
  "build-pods-policy": BuildPodsPolicyZ,
  extend: ExtendZ,
  scale: ScaleZ,
  roll: RollZ,
  "mint-key": MintKeyZ,
  "log-query": LogQueryZ,
  "serverless-cancel": SlsCancelZ,
  "serverless-purge": SlsPurgeZ,
  "job-cancel": JobCancelZ,
  "jobs-query": JobsQueryZ,
  "jobs-cancel-queued": JobsCancelQueuedZ,
} as const;
export type SchemaName = keyof typeof SCHEMAS;

let cache: Record<string, unknown> | null = null;
/** Every schema as JSON Schema (draft 2020-12). */
export function jsonSchemas(): Record<string, unknown> {
  if (!cache) cache = Object.fromEntries(Object.entries(SCHEMAS).map(([k, s]) => [k, z.toJSONSchema(s as z.ZodType, { unrepresentable: "any" })]));
  return cache;
}

export type { Issue };
export { zIssues };
export function validate(name: SchemaName, doc: unknown): { ok: true; value: any } | { ok: false; issues: Issue[] } {
  const r = (SCHEMAS[name] as z.ZodType).safeParse(doc);
  if (r.success) return { ok: true, value: r.data };
  return { ok: false, issues: zIssues(r.error) };
}
export const issuesText = (issues: Issue[], what = "body") => issues.map((i) => `${i.path.join(".") || what}: ${i.message}`).join("; ");
/** Validates or throws 400 with every issue (path-anchored, in `issues`). */
export function parseOr400<N extends SchemaName>(name: N, doc: unknown): z.infer<(typeof SCHEMAS)[N]> {
  const v = validate(name, doc);
  if (!v.ok) throw new HttpError(400, issuesText(v.issues, name), { issues: v.issues });
  return v.value;
}

// ---- compile-time check: the policies type and its schema agree both ways.
type Eq<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;
const _checkPolicies: Eq<z.infer<typeof PoliciesZ>, Policies> = true;
void _checkPolicies;
