// What a serverless endpoint serves, derived from its worker config before
// any worker boots (docs/control/serverless.md §1a): the models, their
// tasks and tiers, the API names that reach them (config aliases and the
// MiniMax tier names), the mounted APIs and fal apps, and the weight trees
// they load. The config is the preset's, the inline config_toml, a config
// file of the image (src/cluster/image-configs.ts, generated from the
// Dockerfile) or the variant's baked default. The tasks and tiers per recipe
// mirror crates/fastvideo-engine-service/src/cuda/caps.rs (and fake.rs for
// the fake engine); a worker's /fv/v1/capabilities is the truth, and
// compareCapabilities flags a difference once one is up.
import { BAKED_CONFIGS, IMAGE_CONFIGS } from "../cluster/image-configs";

// ---------------------------------------------------------------- a small TOML reader
/** Reads the subset of TOML the serve configs use: tables, [[arrays of tables]], quoted keys, strings,
 * booleans, numbers and one-line arrays. Enough to list models, aliases and protocols; not a validator. */
export function readToml(text: string): Record<string, any> {
  const root: Record<string, any> = {};
  let cur: Record<string, any> = root;
  const table = (path: string[], array: boolean) => {
    let t: any = root;
    path.forEach((k, i) => {
      const last = i === path.length - 1;
      if (last && array) {
        if (!Array.isArray(t[k])) t[k] = [];
        const o = {};
        t[k].push(o);
        t = o;
      } else {
        if (Array.isArray(t[k])) t = t[k][t[k].length - 1];
        else t = t[k] ??= {};
      }
    });
    return t;
  };
  const splitKey = (k: string) => (k.match(/"(?:[^"\\]|\\.)*"|[^.\s]+/g) || []).map((x) => (x.startsWith('"') ? JSON.parse(x) : x));
  for (const raw of text.split(/\r?\n/)) {
    const line = stripComment(raw).trim();
    if (!line) continue;
    let m = /^\[\[\s*(.+?)\s*\]\]$/.exec(line);
    if (m) {
      cur = table(splitKey(m[1]!), true);
      continue;
    }
    m = /^\[\s*(.+?)\s*\]$/.exec(line);
    if (m) {
      cur = table(splitKey(m[1]!), false);
      continue;
    }
    m = /^("(?:[^"\\]|\\.)*"|[A-Za-z0-9_.-]+)\s*=\s*(.+)$/.exec(line);
    if (!m) continue;
    const key = m[1]!.startsWith('"') ? JSON.parse(m[1]!) : m[1]!;
    cur[key] = value(m[2]!.trim());
  }
  return root;
}
function stripComment(s: string): string {
  let q: string | null = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i]!;
    if (q) {
      if (c === "\\" && q === '"') i++;
      else if (c === q) q = null;
    } else if (c === '"' || c === "'") q = c;
    else if (c === "#") return s.slice(0, i);
  }
  return s;
}
function value(v: string): unknown {
  if (v === "true") return true;
  if (v === "false") return false;
  if (/^[+-]?\d[\d_]*(\.\d+)?$/.test(v)) return Number(v.replace(/_/g, ""));
  if (v.startsWith('"')) {
    try {
      return JSON.parse(v);
    } catch {
      return v.slice(1, -1);
    }
  }
  if (v.startsWith("'")) return v.slice(1, v.lastIndexOf("'"));
  if (v.startsWith("[") && v.endsWith("]")) {
    const items = v.slice(1, -1).match(/"(?:[^"\\]|\\.)*"|'[^']*'|[^,\s][^,]*/g) || [];
    return items.map((x) => value(x.trim()));
  }
  return v;
}

// ---------------------------------------------------------------- recipes
export type Tier = "max" | "turbo" | "draft";
const H3 = ["t2v", "i2v", "keyframes"];
const LTX_TURBO = ["t2v", "i2v", "keyframes", "a2v", "retake", "extend"];
const LTX_PRO = ["t2v", "i2v", "keyframes", "retake", "extend"];
/** recipe → tasks and tier (crates/fastvideo-engine-service/src/cuda/caps.rs: h3_caps, ltx2_caps, wan_caps, sfwan_caps). */
export const RECIPES: Record<string, { tasks: string[]; tier: Tier | null }> = {
  "h3-max": { tasks: H3, tier: "max" },
  "sol-h3": { tasks: H3, tier: "max" },
  "h3-turbo": { tasks: H3, tier: "turbo" },
  "fasth3-4step-vsa": { tasks: H3, tier: "turbo" },
  "fasth3-8step-dense": { tasks: H3, tier: null },
  "h3-draft": { tasks: H3, tier: "draft" },
  "fasth3-4step-vsa-480p-taeh3": { tasks: H3, tier: "draft" },
  "h3-ref2v-max": { tasks: ["ref2v"], tier: "max" },
  "h3-ref2v-turbo": { tasks: ["ref2v"], tier: "turbo" },
  "ltx-turbo": { tasks: LTX_TURBO, tier: "turbo" },
  "ltx25-distill-sol": { tasks: LTX_TURBO, tier: "turbo" },
  "ltx-draft": { tasks: LTX_TURBO, tier: "draft" },
  "ltx25-distill-sol-nvfp4-taehv": { tasks: LTX_TURBO, tier: "draft" },
  "ltx-pro": { tasks: LTX_PRO, tier: "max" },
  "ltx25-distill-dense": { tasks: LTX_PRO, tier: "max" },
  "ltx25-ref2v": { tasks: ["ref2v"], tier: "max" },
  "ltx25-a2v-guided": { tasks: ["a2v"], tier: "max" },
  "wan-max": { tasks: ["t2v", "i2v"], tier: "max" },
  "wan22-ti2v-5b": { tasks: ["t2v", "i2v"], tier: "max" },
  "wan-turbo": { tasks: ["t2v", "i2v"], tier: "turbo" },
  "fastwan22-ti2v-5b": { tasks: ["t2v", "i2v"], tier: "turbo" },
  "wan-draft": { tasks: ["t2v", "i2v"], tier: "draft" },
  "fastwan22-ti2v-5b-taehv": { tasks: ["t2v", "i2v"], tier: "draft" },
  "fastwan21-1.3b": { tasks: ["t2v"], tier: null },
  "fastwan21-1.3b-taehv": { tasks: ["t2v"], tier: null },
  "wan14b-turbo": { tasks: ["t2v"], tier: null },
  "sfwan21-1.3b": { tasks: ["stream"], tier: null },
};
/** The fake engine's models (crates/fastvideo-engine-service/src/fake.rs). */
const FAKE: Record<string, { family: string; recipe: string }> = {
  "fake-h3-max": { family: "h3", recipe: "h3-max" },
  "fake-sol-h3": { family: "h3", recipe: "h3-max" },
  "fake-h3-turbo": { family: "h3", recipe: "h3-turbo" },
  "fake-ltx-pro": { family: "ltx2", recipe: "ltx-pro" },
  "fake-ltx-turbo": { family: "ltx2", recipe: "ltx-turbo" },
  "fake-wan": { family: "wan", recipe: "fastwan21-1.3b" },
  "fake-sfwan": { family: "wan", recipe: "sfwan21-1.3b" },
};

/** fv-serve's [protocols] defaults (crates/fastvideo-serve/src/config.rs ProtocolsCfg::default). */
const PROTOCOL_DEFAULTS: Record<string, boolean> = { openai_videos: true, fastwan: false, minimax: true, fal: true, fal_director: true, ltx: true, reactor: true, native: true };
const DEFAULT_FAL_APPS = ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft", "minimax/h3-max-turbo", "minimax/h3"];
/** What each API is, for the endpoint page. */
export const PROTOCOL_INFO: Record<string, string> = {
  native: "native (/fv/v1/jobs)",
  minimax: "MiniMax V2 (/v2/video_generation)",
  fal: "fal queue (/<app>/<endpoint>)",
  openai_videos: "OpenAI videos (/v1/videos)",
  ltx: "LTX API (/v1, /v2 text-to-video …)",
  fastwan: "FastWan API",
  fal_director: "fal director",
  reactor: "Reactor (streams)",
};

export interface ServedModel {
  id: string;
  family: string;
  recipe: string;
  tier: Tier | null;
  tasks: string[];
  served_names: string[];
  resident: boolean;
  weights: string[];
}
export interface Serves {
  /** Where the config came from. */
  source: { kind: "inline" | "image" | "image-default" | "unknown"; config: string | null; note?: string };
  engine: string;
  swap: boolean;
  models: ServedModel[];
  /** API names that reach a model: config [aliases] and the MiniMax names (tier resolution, as fv-serve's MiniMax adapter does). */
  aliases: { name: string; model: string; via: "config" | "tier" }[];
  protocols: string[];
  fal_apps: string[];
  tasks: string[];
  weights: string[];
}

const treeOf = (p: unknown): string | null => {
  if (typeof p !== "string") return null;
  const m = /^(?:\$\{FV_WEIGHTS\}|\/workspace\/weights|\/runpod-volume\/weights)\/([^/"]+)/.exec(p);
  return m ? m[1]! : null;
};

/** The serving view of one worker config (TOML text). */
export function servesOfToml(text: string, source: Serves["source"]): Serves {
  const t = readToml(text);
  const engine = String(t.engine?.backend ?? "cuda");
  const models: ServedModel[] = [];
  if (engine === "fake") {
    for (const [id, f] of Object.entries(FAKE)) {
      const r = RECIPES[f.recipe];
      models.push({ id, family: f.family, recipe: f.recipe, tier: r?.tier ?? null, tasks: r?.tasks ?? [], served_names: [id], resident: true, weights: [] });
    }
  }
  for (const m of Array.isArray(t.models) ? t.models : []) {
    const recipe = String(m.recipe ?? m.id ?? "");
    const r = RECIPES[recipe];
    const weights = [treeOf(m.weights), treeOf(m.longlive)].filter((x): x is string => !!x);
    models.push({
      id: String(m.id),
      family: String(m.family ?? ""),
      recipe,
      tier: r?.tier ?? null,
      tasks: r?.tasks ?? [],
      served_names: Array.isArray(m.served_names) ? m.served_names.map(String) : [String(m.id)],
      resident: m.resident === true,
      weights: [...new Set(weights)],
    });
  }
  const p = t.protocols && typeof t.protocols === "object" ? t.protocols : {};
  const protocols = Object.keys(PROTOCOL_DEFAULTS).filter((k) => (typeof p[k] === "boolean" ? p[k] : PROTOCOL_DEFAULTS[k]));
  const fal_apps = protocols.includes("fal") ? (Array.isArray(p.fal_apps) ? p.fal_apps.map(String) : DEFAULT_FAL_APPS) : [];
  const aliases: Serves["aliases"] = [];
  const cfgAliases: Record<string, string> = t.aliases && typeof t.aliases === "object" ? t.aliases : {};
  for (const [name, model] of Object.entries(cfgAliases)) aliases.push({ name, model: String(model), via: "config" });
  if (protocols.includes("minimax")) {
    // crates/fastvideo-minimax resolve_caps: a config alias, else the canonical tier alias, else the H3 model of that tier.
    const h3 = models.filter((m) => m.family === "h3");
    const ofTier = (tier: Tier) => {
      const at = h3.filter((m) => m.tier === tier);
      return at.find((m) => m.tasks.includes("t2v")) ?? at[0];
    };
    for (const [name, tier] of [["MiniMax-H3-Max", "max"], ["MiniMax-H3-Turbo", "turbo"], ["MiniMax-H3-Draft", "draft"], ["MiniMax-H3", "max"]] as const) {
      if (cfgAliases[name]) continue;
      const canon = cfgAliases[`h3-${tier}`];
      const m = canon ? h3.find((x) => x.id === canon) : ofTier(tier) ?? (name === "MiniMax-H3" ? h3[0] : undefined);
      if (m) aliases.push({ name, model: m.id, via: "tier" });
    }
  }
  return {
    source,
    engine,
    swap: t.engine?.swap === true,
    models,
    aliases,
    protocols,
    fal_apps,
    tasks: [...new Set(models.flatMap((m) => m.tasks))],
    weights: [...new Set(models.flatMap((m) => m.weights))],
  };
}

/** The config TOML a spec runs and where it comes from. */
export function configOf(spec: { variant: string; config?: string; config_toml?: string; image?: { ref?: string } }): { text: string | null; source: Serves["source"] } {
  if (spec.config_toml) return { text: spec.config_toml, source: { kind: "inline", config: null } };
  const img = IMAGE_CONFIGS[spec.variant];
  const path = spec.config || img?.default || null;
  const file = path && path.startsWith("/etc/fv/") ? path.slice("/etc/fv/".length) : null;
  const note = spec.image?.ref ? "an image ref: derived from this build's config of the variant (the image may differ)" : undefined;
  const text = file && BAKED_CONFIGS[file] && (!img || img.files.includes(file) || spec.image?.ref) ? BAKED_CONFIGS[file]! : null;
  return { text, source: { kind: text ? (spec.config ? "image" : "image-default") : "unknown", config: path, ...(note ? { note } : {}) } };
}

/** What a spec serves (null fields when its config is unknown, e.g. a path not in the image). */
export function servesOf(spec: Parameters<typeof configOf>[0]): Serves {
  const c = configOf(spec);
  if (!c.text) return { source: { ...c.source, note: c.source.note ?? "the config is not one fv-control knows: nothing derived" }, engine: "?", swap: false, models: [], aliases: [], protocols: [], fal_apps: [], tasks: [], weights: [] };
  return servesOfToml(c.text, c.source);
}

/** A worker's /fv/v1/capabilities (the body, or a queue http job's output wrapping it) against the derived view. */
export function compareCapabilities(serves: Serves, caps: unknown): { ok: boolean; served: string[]; missing: string[]; extra: string[]; note?: string } {
  const body = findCaps(caps);
  if (!body) return { ok: false, served: [], missing: [], extra: [], note: "no capabilities body (models[]) in the reply" };
  const ids: string[] = body.models.map((m: any) => String(m?.caps?.id ?? m?.id ?? "")).filter(Boolean);
  const want = serves.models.map((m) => m.id);
  const missing = want.filter((x) => !ids.includes(x));
  const extra = ids.filter((x) => !want.includes(x));
  return { ok: !missing.length && !extra.length, served: ids, missing, extra };
}
function findCaps(v: unknown, depth = 0): { models: unknown[] } | null {
  if (depth > 4 || v === null || v === undefined) return null;
  if (typeof v === "string") {
    try {
      return findCaps(JSON.parse(v), depth + 1);
    } catch {
      return null;
    }
  }
  if (typeof v !== "object") return null;
  const o = v as any;
  if (Array.isArray(o.models)) return o;
  for (const k of ["body", "output", "result"]) {
    const r = findCaps(o[k], depth + 1);
    if (r) return r;
  }
  return null;
}
