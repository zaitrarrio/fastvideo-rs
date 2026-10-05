// JSON documents the controller edits, as zod schemas: the one source for
// server-side validation (every write path), the JSON Schemas served at
// /api/schemas (the editor's validation, hover docs, completion and form),
// and compile-time checks against the TypeScript types used everywhere
// else (the `_check*` assignments at the bottom fail `tsc` when they drift).
import { z } from "zod";
import type { Policies } from "./alerts";
import type { ClusterSpec, PoolSpec } from "./cluster/spec";

/** The gateway's `[protocols]` switches (fv-serve ProtocolsCfg; serde denies unknown keys). */
export const GATEWAY_PROTOCOLS = ["openai_videos", "fastwan", "minimax", "fal", "fal_director", "ltx", "reactor", "native"] as const;
export type GatewayProtocol = (typeof GATEWAY_PROTOCOLS)[number];

const id = (what: string) =>
  z
    .string()
    .regex(/^[a-z][a-z0-9-]{0,30}$/)
    .describe(`${what}: lower-case letters, digits and '-', starting with a letter (max 31).`);
const region = z.enum(["eu", "us"]).describe("eu = volume jg48s6o1w0 in EUR-IS-1 (RTX PRO 6000); us = volume s2k01690bi in US-CA-2 (H100 / H200).");
const cpuFlavor = z.enum(["cpu3c", "cpu3g", "cpu3m", "cpu5c", "cpu5g", "cpu5m"]).describe("Runpod CPU flavor: 3/5 = generation; c compute, g general, m memory optimised.");

export const ModelRefZ = z
  .object({
    id: z.string().min(1).meta({ "x-dynamic": "model_ids" }).describe("Model id the gateway advertises (e.g. fasth3, ltx25-distill-sol); the worker config's [[models]] id."),
    family: z.string().min(1).meta({ "x-dynamic": "families" }).describe("Model family (h3, ltx2, wan)."),
    recipe: z
      .string()
      .min(1)
      .meta({ "x-dynamic": "recipes" })
      .describe("A tier alias (h3-max, h3-turbo, h3-draft, ltx-pro, ltx-turbo, ltx-draft, wan-max, wan-turbo, wan-draft) or a catalog model id (sfwan21-1.3b, ltx25-a2v-guided, h3-ref2v-turbo, …). The gateway resolves it against the fv-serve CUDA catalog at start and does not start on an unknown one."),
  })
  .strict()
  .describe("A static capability of the pool: what the gateway advertises while the pool has no ready worker.");

export const PoolSpecZ = z
  .object({
    id: id("Pool id (the gateway's pool, FV_POOL_<ID>_URLS)"),
    variant: z
      .string()
      .regex(/^[a-z0-9][a-z0-9-]{0,30}$/)
      .meta({ "x-dynamic": "variants" })
      .describe("Image variant (docs/serve/images.md): h3-turbo, h3-max, ltx, wan, wan5b, sfwan, gateway (CPU, also carries the fake engine). The pool presets (ltx-pro, ltx-a2v, ltx-ref2v, h3-ref2v, longlive) reuse these images with an inline config_toml."),
    count: z.number().int().min(0).max(8).describe("Worker pods in this pool. Change a running cluster's count with Scale."),
    compute: z.enum(["GPU", "CPU"]).describe("GPU pod, or CPU pod (fake engine, tests)."),
    config: z.string().optional().describe("Worker config inside the image (e.g. /etc/fv/runpod.toml). One of config / config_toml."),
    config_toml: z.string().max(32768).optional().describe("An inline worker config (sent as FV_WORKER_TOML_B64) instead of a file in the image."),
    gpu_types: z.array(z.string().meta({ "x-dynamic": "gpu_types" })).optional().describe("Runpod GPU type ids to try, in order. Default: the region's (RTX PRO 6000 in eu; H100 80GB, H100 NVL, H200 in us)."),
    regions: z.array(region).optional().describe("Regions to try, in order. Default: the cluster's."),
    cpu_flavors: z.array(cpuFlavor).optional().describe("CPU pods: flavors to try, in order."),
    vcpu: z.number().int().min(1).max(32).optional().describe("CPU pods: vCPUs."),
    container_disk_gb: z.number().int().min(5).max(500).optional().describe("Container disk in GB (default 40 GPU, 10 CPU)."),
    volume: z.boolean().optional().describe("Mount the region's network volume at /workspace (weights). Default: on for GPU pods."),
    image: z.string().optional().describe("Override: an image reference for this pool (resolved to a digest at start)."),
    models: z.array(ModelRefZ).optional().describe("The gateway's static caps for this pool."),
    fake_models: z.array(z.string().meta({ "x-dynamic": "fake_models" })).optional().describe("Fake-engine model ids (fake-wan, fake-h3-turbo, …) as static caps."),
    max_queued: z.number().int().min(0).max(10000).optional().describe("Gateway admission: queued jobs for this pool."),
    job_timeout_s: z.number().int().min(10).max(86400).optional().describe("Gateway job timeout in seconds."),
    stale_after_s: z.number().int().min(10).max(86400).optional().describe("A running job without a worker heartbeat for this long is lost."),
  })
  .strict()
  .refine((p) => !!(p.config || p.config_toml), { message: "config or config_toml is required", path: ["config"] })
  .refine((p) => !!(p.models?.length || p.fake_models?.length), { message: "models or fake_models (the gateway's static caps) is required", path: ["models"] })
  .describe("A pool of worker pods behind the gateway.");

export const ClusterSpecZ = z
  .object({
    name: id("Cluster name"),
    image: z
      .object({
        channel: z.string().regex(/^[a-z][a-z0-9-]{0,30}$/).meta({ "x-dynamic": "channels" }).optional().describe("Release channel (stable, latest, …): per-variant images <variant>-<channel>."),
        sha: z.string().regex(/^[0-9a-f]{7,40}$/).meta({ "x-dynamic": "shas" }).optional().describe("A git commit: per-variant images <variant>-sha-<sha7>."),
        ref: z.string().optional().describe("One (all-in-one) image reference for every pod."),
      })
      .strict()
      .refine((i) => [i.channel, i.sha, i.ref].filter(Boolean).length === 1, { message: "exactly one of channel, sha, ref" })
      .describe("Which build the pods run. Resolved to digests at start."),
    regions: z.array(region).min(1).describe("Regions to place GPU workers in, in order (the gateway prefers the first)."),
    gateway: z
      .object({
        enabled: z.boolean().describe("Run a gateway pod in front of the pools."),
        cpu_flavors: z.array(cpuFlavor).min(1).describe("Gateway CPU flavors to try, in order."),
        vcpu: z.number().int().min(1).max(32).describe("Gateway vCPUs."),
        container_disk_gb: z.number().int().min(5).max(200).describe("Gateway container disk (GB)."),
        base: z.enum(["pods", "minimal"]).describe("Gateway config base: pods = configs/serve/gateway-pods.toml; minimal = without the reactor, fal apps and keys newer than older images."),
        github_token: z.boolean().describe("Pass GITHUB_PAT as FV_GITHUB_TOKEN (console promote / rollback)."),
        auth: z.enum(["keys", "none"]).describe("The gateway's user auth mode (FV_AUTH_MODE)."),
        fal_apps: z
          .array(z.string().regex(/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/).meta({ "x-dynamic": "fal_apps" }))
          .max(64)
          .optional()
          .describe("Replaces the base's [protocols] fal_apps (default: every worker config's apps). An app no pool serves answers 404; the gateway still starts."),
        protocols: z
          .object(Object.fromEntries(GATEWAY_PROTOCOLS.map((k) => [k, z.boolean().optional()])) as Record<GatewayProtocol, z.ZodOptional<z.ZodBoolean>>)
          .strict()
          .optional()
          .describe("Overrides single [protocols] switches of the base (openai_videos, fastwan, minimax, fal, fal_director, ltx, reactor, native)."),
        reactor_model: z
          .string()
          .min(1)
          .nullable()
          .optional()
          .meta({ "x-dynamic": "model_ids" })
          .describe("[gateway] reactor_model. Default: the base's (fasth3) when a pool serves it, else a pool's causal model (sfwan21-1.3b, longlive-1.3b). null: none (the gateway takes the first streaming model of a pod pool)."),
        aliases: z
          .record(z.string().min(1).max(80), z.string().min(1).meta({ "x-dynamic": "model_ids" }))
          .optional()
          .describe("Replaces the base's [aliases] (MiniMax-H3 → fasth3, …): public model names → model ids."),
      })
      .strict(),
    pools: z.array(PoolSpecZ).describe("Worker pools."),
    cap_s: z.number().int().min(300).max(7 * 86400).describe("Backstop: every pod is deleted at start + cap_s seconds (Extend moves it)."),
    min_balance: z.number().min(8).max(10000).describe("The gateway pod's watchdog deletes the cluster below this account balance ($)."),
    balance_floor: z.number().min(8).max(10000).describe("Refuse start / extend / scale-up when the projected balance at the deadline would be below this ($); the cron stops the cluster below it."),
    min_start: z.number().min(8).max(10000).describe("Refuse to start below this balance ($)."),
    max_gpu_dph: z.number().min(0.1).max(50).describe("A GPU pod costing more than this $/hr is deleted right after create."),
    auto_stop_idle_min: z.number().int().min(5).max(1440).nullable().optional().describe("Remove a worker idle (GPU < threshold, no jobs) this many minutes. null: the account policy."),
    log_shipping: z.boolean().describe("Pods ship their logs to the controller (FV_LOG_SHIP_*)."),
    log_level: z.enum(["trace", "debug", "info", "warn", "error"]).optional().describe("Most verbose level shipped."),
  })
  .strict()
  .superRefine((s, ctx) => {
    const seen = new Set<string>();
    s.pools.forEach((p, i) => {
      if (seen.has(p.id)) ctx.addIssue({ code: "custom", message: `pool ${p.id} twice`, path: ["pools", i, "id"] });
      if (p.id === "gateway") ctx.addIssue({ code: "custom", message: "'gateway' is reserved", path: ["pools", i, "id"] });
      seen.add(p.id);
    });
  })
  .describe("A gateway cluster on Runpod (docs/control/README.md §4).");

export const AttributionZ = z
  .array(
    z
      .object({
        prefix: z.string().min(1).max(60).describe("Pod name prefix."),
        owner: z.string().regex(/^[a-z][a-z0-9:._-]{0,60}$/).describe("Owner label for pods whose name starts with the prefix (e.g. external:build-pod)."),
      })
      .strict(),
  )
  .max(50)
  .describe("External pods by name prefix; the first match wins.");

export const PoliciesZ = z
  .object({
    idle_gpu_pct: z.number().min(0).max(100).describe("A GPU pod is idle below this utilisation (%) with no running jobs."),
    idle_min: z.number().min(1).max(1440).describe("Alert when a GPU pod has been idle this many minutes."),
    auto_stop_idle: z.boolean().describe("Auto-action: remove idle workers of controller clusters (never external pods)."),
    auto_stop_idle_min: z.number().min(5).max(1440).describe("…after this many idle minutes."),
    cluster_dph_max: z.number().min(0).describe("Alert when a cluster costs more than this $/hr."),
    daily_spend_max: z.number().min(0).describe("Alert when today's account spend exceeds this ($)."),
    balance_margin: z.number().min(0).describe("Alert when the balance is within this many $ of the floor."),
    stop_on_floor: z.boolean().describe("Auto-action (default on): stop controller clusters below the balance floor."),
    pod_down_min: z.number().min(1).max(1440).describe("Alert when a controller pod has not answered this many minutes."),
    attribution: AttributionZ,
    build_pod_backstop: z.boolean().describe("Auto-action (default on): stop the shared build pod (external:build-pod) when its own self-stop did not happen."),
    build_pod_max_h: z.number().min(0).max(72).describe("…when it has been up this many hours (0: off; its own cap is 8 h + 30 min grace)."),
    build_pod_idle_grace_min: z.number().min(0).max(1440).describe("…or idle (no jobs, per its /healthz) this many minutes past its own idle stop."),
  })
  .strict()
  .describe("Alert thresholds and auto-actions (docs/control/README.md §8).");

export const ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
/** An env set as the editor sees it: KEY -> {value, secret}. A secret's value is never sent back: `value` is null
 *  and a new value goes in `set` (write-only). */
export const EnvVarZ = z
  .object({
    value: z.string().max(32768).nullable().describe("The value (null for a secret: it is never displayed)."),
    secret: z.boolean().describe("Sealed in D1, masked everywhere, written only through `set`."),
    set: z.string().max(32768).optional().meta({ "x-secret": true }).describe("Write-only: a new value for a secret."),
  })
  .strict();
export const EnvSetZ = z.record(z.string().regex(ENV_KEY_RE).meta({ "x-dynamic": "env_keys" }), EnvVarZ).describe("Environment variables at one level (account, cluster, pool or pod).");

export const TokenCreateZ = z
  .object({
    name: z.string().regex(/^[A-Za-z0-9._-]{1,40}$/).describe("A name for the token (1-40 of A-Z a-z 0-9 . _ -)."),
    scope: z.enum(["read", "admin"]).describe("read: GET only; admin: everything but minting tokens."),
    ttl_days: z.number().int().min(1).max(365).optional().describe("Expiry in days (default 90)."),
  })
  .strict();

export const ReleaseDispatchZ = z
  .object({
    action: z.enum(["promote", "rollback"]),
    target: z.string().optional().describe("promote: a git sha, a digest or a tag."),
    channel: z.string().regex(/^[a-z][a-z0-9-]{0,30}$/).optional().describe("Channel (stable: what deploys and the templates follow)."),
    to: z.string().regex(/^\d+$/).optional().describe("rollback: a release id."),
    notes: z.string().max(200).optional(),
    templates: z.boolean().optional(),
    allow_partial: z.boolean().optional(),
    dry_run: z.boolean().optional(),
  })
  .strict();

export const SCHEMAS = {
  "cluster-spec": ClusterSpecZ,
  pool: PoolSpecZ,
  policies: PoliciesZ,
  attribution: AttributionZ,
  env: EnvSetZ,
  "token-create": TokenCreateZ,
  "release-dispatch": ReleaseDispatchZ,
} as const;
export type SchemaName = keyof typeof SCHEMAS;

let cache: Record<string, unknown> | null = null;
/** Every schema as JSON Schema (draft 2020-12). */
export function jsonSchemas(): Record<string, unknown> {
  if (!cache) cache = Object.fromEntries(Object.entries(SCHEMAS).map(([k, s]) => [k, z.toJSONSchema(s as z.ZodType, { unrepresentable: "any" })]));
  return cache;
}

export interface Issue {
  path: (string | number)[];
  message: string;
}
export function validate(name: SchemaName, doc: unknown): { ok: true; value: any } | { ok: false; issues: Issue[] } {
  const r = (SCHEMAS[name] as z.ZodType).safeParse(doc);
  if (r.success) return { ok: true, value: r.data };
  return { ok: false, issues: r.error.issues.map((i) => ({ path: i.path.map((p) => (typeof p === "symbol" ? String(p) : p)) as (string | number)[], message: i.message })) };
}

// ---- compile-time checks: the zod types and the TypeScript types agree both ways.
type Eq<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;
const _checkSpec: Eq<z.infer<typeof ClusterSpecZ>, ClusterSpec> = true;
const _checkPool: Eq<z.infer<typeof PoolSpecZ>, PoolSpec> = true;
const _checkPolicies: Eq<z.infer<typeof PoliciesZ>, Policies> = true;
void _checkSpec, _checkPool, _checkPolicies;
