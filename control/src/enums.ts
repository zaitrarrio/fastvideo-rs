// The closed sets and naming rules every fv-control configuration uses
// (docs/control/config-validation.md): the zod schemas (src/schemas.ts),
// the serverless spec, the dynamic enums (/api/schemas/dynamic) and the
// dashboard's controls all read them from here, so a value the UI offers is
// a value the server accepts. No imports: every module may import this one.

/** Image variants CI builds (scripts/serve/variants.sh, docs/serve/images.md). `cpu`: the fake engine on CPU pods. */
export const VARIANTS = ["h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan", "cpu"] as const;
export type Variant = (typeof VARIANTS)[number];
/** Fake-engine model ids the `cpu` image serves (crates/fastvideo-serve fake engine). */
export const FAKE_MODELS = ["fake-h3-max", "fake-h3-turbo", "fake-sol-h3", "fake-ltx-pro", "fake-ltx-turbo", "fake-wan", "fake-sfwan"] as const;
/** Runpod CPU flavors (REST v1 PodCreateInput.cpuFlavorIds). */
export const CPU_FLAVORS = ["cpu3c", "cpu3g", "cpu3m", "cpu5c", "cpu5g", "cpu5m"] as const;
export type CpuFlavor = (typeof CPU_FLAVORS)[number];
/** Runpod serverless CPU flavors (REST v1 EndpointCreateInput.cpuFlavorIds). */
export const SLS_CPU_FLAVORS = ["cpu3c", "cpu3g", "cpu5c", "cpu5g"] as const;
/** vCPU sizes of a Runpod CPU instance (instance ids cpuXy-<vcpu>-<ram>): powers of two. */
export const CPU_VCPUS = [2, 4, 8, 16, 32] as const;
export type CpuVcpu = (typeof CPU_VCPUS)[number];
/** Build pod sizes (the build pod policy also allows 64-vCPU instances). */
export const BUILD_VCPUS = [2, 4, 8, 16, 32, 64] as const;
export const LOG_LEVELS = ["trace", "debug", "info", "warn", "error"] as const;
export type LogLevel = (typeof LOG_LEVELS)[number];
export const LOG_SOURCES = ["pod", "op", "audit", "runpod", "archive"] as const;
/** The edge family Durable Objects (docs/serve/edge-control-plane.md §5.2; cluster/spec.ts modelFamily). */
export const EDGE_FAMILIES = ["h3", "ltx", "wan", "sfwan", "fake"] as const;
export const TOKEN_SCOPES = ["read", "admin", "ci"] as const;
export const CUDA_VERSIONS = ["13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"] as const;

/** Runpod data centre ids (REST v1 openapi, PodCreateInput.dataCenterIds, 2026-10). */
export const RUNPOD_DATA_CENTERS = [
  "EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1", "US-TX-3", "EUR-IS-2", "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1", "CA-MTL-3", "EU-NL-1", "US-TX-4",
  "US-CA-2", "US-NC-1", "OC-AU-1", "US-DE-1", "EUR-IS-3", "CA-MTL-2", "AP-JP-1", "EUR-NO-1", "EU-FR-1", "US-KS-3", "US-GA-1", "AP-IN-1", "US-MD-1", "US-MO-1", "US-MO-2",
] as const;
/** Runpod GPU type ids (REST v1 openapi, PodCreateInput.gpuTypeIds, 2026-10). The live catalog (prices, stock) is /api/schemas/dynamic. */
export const RUNPOD_GPU_TYPES = [
  "AMD Instinct MI300X OAM", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA B200", "NVIDIA B300 SXM6 AC", "NVIDIA B300 SXM6 AC MIG 1g.34gb",
  "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4070 Ti", "NVIDIA GeForce RTX 4080",
  "NVIDIA GeForce RTX 4080 SUPER", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 5090", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200",
  "NVIDIA H200 NVL", "NVIDIA L4", "NVIDIA L40", "NVIDIA L40S", "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 4000 SFF Ada Generation", "NVIDIA RTX 5000 Ada Generation",
  "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A2000", "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX A6000", "NVIDIA RTX PRO 4000 Blackwell", "NVIDIA RTX PRO 4500 Blackwell",
  "NVIDIA RTX PRO 5000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
  "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB",
] as const;
/** Where a pod runs (docs/serve/deploy-gmi-brev.md): Runpod (the default and
 * primary), GMI Cloud containers, NVIDIA Brev VMs. CloudRift is observed by the
 * collector but never launched from fv-control (on hold, docs/ops/cloudrift.md). */
export const PROVIDERS = ["runpod", "gmi", "brev"] as const;
export type ProviderId = (typeof PROVIDERS)[number];
/** The providers fv-control launches pods on besides Runpod. */
export const OTHER_PROVIDERS = ["gmi", "brev"] as const;
export type OtherProviderId = (typeof OTHER_PROVIDERS)[number];
/** Where a worker's weights come from: the region's Runpod network volume, the Hub at pinned revisions at boot (owner approval), or none (fake engine). */
export const WEIGHTS_SOURCES = ["volume", "hub", "none"] as const;
export type WeightsSource = (typeof WEIGHTS_SOURCES)[number];
/** A provider's own GPU product / instance type id (GMI product, Brev instanceType). */
export const PROVIDER_GPU_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
/** A GMI IDC id (GET /v1/idcs: ^[a-zA-Z0-9._-]+$, max 50). */
export const PROVIDER_REGION_RE = /^[A-Za-z0-9._-]{1,50}$/;

/** Runpod's name limit (REST v1: PodCreateInput / EndpointCreateInput / TemplateCreateInput name maxLength 191). */
export const RUNPOD_NAME_MAX = 191;

/** Keys the controller owns: user env layers may not set them. */
export const RESERVED_KEYS = new Set([
  "FV_INTERNAL_TOKEN",
  "FV_URL_SIGNING_KEY",
  "FV_BACKSTOP_API_KEY",
  "FV_CLUSTER_DEADLINE",
  "FV_MIN_BALANCE",
  "FV_ADMIN_TOKEN",
  "FV_ADMIN_TOKEN_RECIPIENT",
  "FV_WORKER_TOML_B64",
  "FV_WORKER_CONFIG",
  "FV_WORKER_DIRECT",
  "FV_SERVE_ROLE",
  "FV_PUBLIC_BASE_URL",
  "FV_LOG_SHIP_URL",
  "FV_LOG_SHIP_TOKEN",
  "FV_IMAGE_REF",
  "FV_IMAGE_DIGEST",
  "FV_DISPATCH_FRONT",
  "FV_DISPATCH_DO_URL",
  "FV_DISPATCH_FAMILIES",
  "FV_DISPATCH_MODEL_FAMILIES",
  "FV_DISPATCH_ENDPOINT",
  "FV_DISPATCH_DIRECT_UPLOAD",
  // GMI / Brev pods (providers.ts): their id, the weights plan, the tunnel report.
  "FV_POD_ID",
  "FV_POD_NAME",
  "FV_PROVIDER",
  "FV_WEIGHTS_SOURCE",
  "FV_WEIGHTS_TREES_B64",
  "FV_SCRIPTS_URL",
  "FV_ENDPOINT_REPORT_URL",
  "FV_ENDPOINT_REPORT_TOKEN",
  "FV_BACKSTOP_API",
]);
export const isReserved = (k: string) => RESERVED_KEYS.has(k);

/** The pool presets' ids (cluster/spec.ts POOL_PRESETS; test/unit/config-validation.test.ts checks they agree). */
export const POOL_PRESET_IDS = ["h3-turbo", "h3-max", "ltx", "wan", "ltx-pro", "ltx-a2v", "ltx-ref2v", "h3-ref2v", "fastwan21", "sfwan", "longlive"] as const;
/** A serverless endpoint's presets: the pool presets plus `cpu` (the fake engine on CPU workers, for tests); serverless/presets.ts. */
export const SLS_PRESET_IDS = [...POOL_PRESET_IDS, "cpu"] as const;

// ---------------------------------------------------------------- names
/**
 * An fv-control name (cluster, standalone pod, pool, serverless endpoint):
 * a DNS label (RFC 1123: lower-case letters, digits and '-', starting with a
 * letter, not ending with '-'), at most 31 characters. It goes into Runpod
 * pod names (fv-ctl-<cluster>-<pool>-<MMDDHHMMSS>: at most 81 of Runpod's
 * 191), endpoint and template names (fvc-<name>-<stamp>), Durable Object
 * names and URLs (/api/clusters/<name>).
 */
export const NAME_RE = /^[a-z](?:[a-z0-9-]{0,29}[a-z0-9])?$/;
export const NAME_MAX = 31;
export const NAME_RULE = "1-31 lower-case letters, digits and '-'; starts with a letter, does not end with '-'";
/** Names a route already uses (/api/clusters/validate, /api/clusters/new/price, #/clusters/new; /api/serverless/{defaults,policy,validate,tick}). */
export const RESERVED_NAMES = {
  cluster: ["new", "validate"],
  pool: ["gateway"],
  endpoint: ["defaults", "policy", "validate", "tick"],
} as const;
export type NameKind = keyof typeof RESERVED_NAMES;
/** API token names (D1 api_tokens.name). */
export const TOKEN_NAME_RE = /^[A-Za-z0-9._-]{1,40}$/;
export const TOKEN_NAME_RULE = "1-40 of A-Z a-z 0-9 . _ -; unique among the active tokens";
/** User API key names (fv-serve /fv/v1/admin/keys). */
export const KEY_NAME_RE = /^[A-Za-z0-9 ._-]{1,60}$/;
export const KEY_NAME_RULE = "1-60 of letters, digits, space . _ -";
/** Env variable names (POSIX-ish; Runpod passes them to the container as is). */
export const ENV_KEY_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
export const ENV_KEY_RULE = "a letter or '_', then letters, digits or '_' (max 128)";
export const ENV_KEY_MESSAGE = `invalid variable name: ${ENV_KEY_RULE}`;

export const SHA_RE = /^[0-9a-f]{7,40}$/;
export const CHANNEL_RE = /^[a-z][a-z0-9-]{0,30}$/;
/** An OCI image reference: registry/repo[:tag][@sha256:digest] (lower-case registry and path). */
export const IMAGE_REF_RE = /^[a-z0-9.\-]+(:[0-9]+)?\/[a-z0-9._\-/]+(:[A-Za-z0-9._-]+)?(@sha256:[0-9a-f]{64})?$/;
/** A worker config file inside the image. */
export const CONFIG_PATH_RE = /^\/[A-Za-z0-9._/-]+\.toml$/;

/**
 * Values of the engine env keys fv-serve and the engine read
 * (cluster/catalog.json engine_env; the parsers in crates/): what the env
 * editors offer as a select or a toggle, and what the server accepts.
 */
export const ENV_VALUE_TYPES: Record<string, { kind: "enum" | "bool" | "path"; values?: readonly string[] }> = {
  FASTVIDEO_ATTN_SAGE: { kind: "enum", values: ["0", "2", "off", "on"] }, // crates/fastvideo-cudarc/src/wan/attn_sage.rs forced()
  FASTVIDEO_FLASH_KERNEL: { kind: "enum", values: ["auto", "v1", "v2", "cudnn", "dc"] },
  FASTVIDEO_NVFP4: { kind: "enum", values: ["1", "static_6", "static_4", "mse"] },
  FASTVIDEO_FP8: { kind: "bool" }, // envflag::bool_flag
  FASTVIDEO_H3_QUANT: { kind: "enum", values: ["w8a8", "mxfp8", "off"] },
  FASTVIDEO_LTX2_TEXT_FP8: { kind: "bool" },
  FASTVIDEO_TAE_DIR: { kind: "path" },
  FASTVIDEO_LTX_OFFLOAD: { kind: "enum", values: ["none", "cpu"] }, // ltx2/memory.rs LtxOffload::parse
  FASTVIDEO_DIT_OFFLOAD: { kind: "enum", values: ["auto", "resident", "streamed"] }, // wan/offload.rs DitOffload::parse
  FASTVIDEO_WAN_AUDIO: { kind: "enum", values: ["mmaudio"] },
  FV_LONGLIVE_WEIGHTS: { kind: "path" },
  FV_LONGLIVE_RECACHE: { kind: "bool" },
  FV_LONGLIVE_INFINITY: { kind: "bool" },
};
export const BOOL_VALUES = ["0", "1", "true", "false", "on", "off"] as const;

/** Why a value of a known env key is wrong, or null (unknown keys take any string). */
export function envValueProblem(key: string, value: string): string | null {
  const t = ENV_VALUE_TYPES[key];
  if (!t) return null;
  const v = value.trim();
  if (t.kind === "bool") return (BOOL_VALUES as readonly string[]).includes(v.toLowerCase()) ? null : `${key} is a switch: 0 or 1`;
  if (t.kind === "enum") return t.values!.includes(v.toLowerCase()) ? null : `${key}: one of ${t.values!.join(", ")}`;
  return /^\/\S*$/.test(v) ? null : `${key}: an absolute path`;
}

/** Why a name of this kind is not allowed (pattern, reserved), or null. Uniqueness is the caller's (it needs D1). */
export function nameProblem(kind: NameKind, name: unknown): string | null {
  if (typeof name !== "string" || !name) return "required";
  if (name.length > NAME_MAX) return `at most ${NAME_MAX} characters`;
  if (!NAME_RE.test(name)) return NAME_RULE;
  if ((RESERVED_NAMES[kind] as readonly string[]).includes(name)) return `"${name}" is reserved (a route uses it)`;
  return null;
}

// ---------------------------------------------------------------- close matches
const tokens = (s: string) => s.toLowerCase().replace(/^nvidia\s+|geforce\s+/g, "").split(/[^a-z0-9.]+/).filter(Boolean);
/** The allowed values closest to a wrong one (shared tokens, then the shortest): "RTX 6000 PRO" → the RTX PRO 6000 Server Edition first. */
export function closeMatches(input: string, allowed: readonly string[], n = 3): string[] {
  const want = new Set(tokens(String(input)));
  if (!want.size) return [];
  const scored = allowed
    .map((a) => {
      const t = tokens(a);
      const hit = t.filter((x) => want.has(x)).length;
      const partial = [...want].filter((w) => t.some((x) => x !== w && (x.includes(w) || w.includes(x)))).length;
      return { a, score: hit * 2 + partial, len: a.length };
    })
    .filter((x) => x.score > 0)
    .sort((x, y) => y.score - x.score || x.len - y.len);
  if (scored.length) return scored.slice(0, n).map((x) => x.a);
  // No shared word: the longest common prefix (cpu3x → cpu3c, cpu3g, cpu3m).
  const lc = String(input).toLowerCase();
  const pre = (a: string) => {
    let i = 0;
    while (i < a.length && i < lc.length && a[i]!.toLowerCase() === lc[i]) i++;
    return i;
  };
  const best = Math.max(0, ...allowed.map(pre));
  return best >= 3 ? allowed.filter((a) => pre(a) === best).slice(0, n) : [];
}
/** "not a Runpod GPU type id: did you mean …" for an enum refusal. */
export function enumMessage(what: string, input: unknown, allowed: readonly string[]): string {
  const near = typeof input === "string" ? closeMatches(input, allowed) : [];
  return `${JSON.stringify(input)} is not ${what}${near.length ? `; did you mean ${near.map((x) => JSON.stringify(x)).join(" or ")}?` : ` (${allowed.length} allowed: ${allowed.slice(0, 6).join(", ")}…)`}`;
}
