// The preset catalog: an image variant plus the worker config it runs, the
// models it serves, the weight trees it loads and the GPU memory it needs.
// One source for cluster pools (cluster/spec.ts re-exports it), standalone
// pods (standalone.ts) and serverless endpoints (serverless/presets.ts). The
// deployment data (ids, titles, image variants, configs, models, queues,
// timeouts, disks, family members) is deploy/deploy.toml, generated into
// ./deploy-gen.ts (plan C1); the weights and GPU memory come from the family
// manifests (./families-gen.ts). What stays here is how they combine.
// Kept free of runtime imports from ./schemas and ./cluster/spec (both import
// this module's users), so there is no module cycle.
import type { ModelRef, PoolSpec } from "./cluster/spec";
import CATALOG from "./cluster/catalog.json";
import { WORKER_CONFIGS } from "./cluster/worker-configs";
import { MANIFEST_VARIANTS } from "./families-gen";
import { mustVariant } from "./families";
import { recipeTreesOf } from "./recipe-trees";
import { readToml } from "./serverless/serves";
import { DEPLOY_FAMILY_PRESETS, DEPLOY_PRESETS, type DeployFamilyPreset, type DeployPreset } from "./deploy-gen";
import type { Variant } from "./enums";

const inline = (file: string) => {
  const t = WORKER_CONFIGS[file];
  if (!t) throw new Error(`no generated worker config ${file} (node gen-configs.mjs)`);
  return t;
};
/** A deploy/deploy.toml preset's pool: its image variant, its worker config (the path in the image, or inline as
 * FV_WORKER_TOML_B64 when the image does not carry it), its models, queue and timeouts. */
function poolOf(p: DeployPreset): PoolSpec {
  return {
    id: p.id,
    variant: p.image as Variant,
    count: 1,
    compute: "GPU",
    ...(p.inline ? { config_toml: inline(p.config) } : { config: `/etc/fv/${p.config}` }),
    ...(p.container_disk_gb !== undefined ? { container_disk_gb: p.container_disk_gb } : {}),
    models: p.models.map(({ id, family, recipe }) => ({ id, family, recipe })),
    max_queued: p.max_queued,
    job_timeout_s: p.job_timeout_s,
    stale_after_s: p.stale_after_s,
  };
}
/** The default cluster's pools (deploy/deploy.toml presets with `standard = true`). */
export const STANDARD_POOLS: PoolSpec[] = DEPLOY_PRESETS.filter((p) => p.standard).map(poolOf);

/** A pool the dashboard can add: an image variant plus the worker config it runs. */
export interface PoolPreset {
  id: string;
  title: string;
  description: string;
  /** Weight trees (under /workspace/weights on the volume) the workers load: the config's `weights` plus every tree
   * the engine finds beside it for the recipe (recipe-trees.ts: adapters, Ref2VA, dev DiT, IC-LoRA). The load fails
   * without any of them. test/unit/preset-trees.test.ts checks the list against the config. */
  weights: string[];
  /** Derived pre-quantized trees (nested in a `weights` tree, as on the volume) that only speed up the load: fetched
   * with weights_source r2 (the boot skips the MXFP8 ones below sm_100), never from the Hub (recipe-trees.ts). */
  prequant?: string[];
  /** Set when the weights' licence restricts use. */
  licence?: string;
  /** The least GPU memory a worker needs (GB, Runpod's memoryInGb): a serverless endpoint whose GPU types offer
   * less is refused (gpus.ts GPU_MEMORY_GB). Conservative: the H3 and LTX-2.5 DiTs run on 80-96 GB cards only. */
  min_vram_gb: number;
  pool: PoolSpec;
  /** A family preset (FAMILY_PRESETS): the presets whose models its one pool serves. */
  members?: string[];
  /** A family preset: what of the family it does not serve, and why (shown with the preset; never silent). */
  left_out?: { what: string; why: string }[];
}
const uniq = <T,>(xs: T[]) => [...new Set(xs)];
const recipeOf = (r: string) => {
  const t = recipeTreesOf(r);
  if (!t) throw new Error(`recipe-trees.ts has no entry for recipe ${r}`);
  return t;
};
/** The weight trees the models' recipes read (recipe-trees.ts, from the family manifests), in the recipes' order. */
export const recipeTrees = (models: ModelRef[]): string[] => uniq(models.flatMap((m) => recipeOf(m.recipe).required));
/** The GPU memory the largest of the models' variants needs (families/*.toml gpu.min_vram_gib). */
export const recipeVramGb = (models: ModelRef[]): number => Math.max(...models.map((m) => MANIFEST_VARIANTS[recipeOf(m.recipe).catalog]!.gpu.min_vram_gib));
/** A model preset as written: weights and GPU memory come from the family manifests through its models' recipes,
 * unless given (a tree the manifests do not know, `weights_extra`; a hand-chosen card class, `min_vram_gb`). */
type PresetInput = Omit<PoolPreset, "weights" | "min_vram_gb"> & { min_vram_gb?: number; weights_extra?: string[] };
function fromManifest({ weights_extra, min_vram_gb, ...p }: PresetInput): PoolPreset {
  const models = p.pool.models ?? [];
  return { ...p, weights: uniq([...recipeTrees(models), ...(weights_extra ?? [])]), min_vram_gb: min_vram_gb ?? recipeVramGb(models) };
}
/** The presets of one model (or one tier and its companions) each (deploy/deploy.toml [[presets]]). Their weights
 * and min_vram_gb derive from the family manifests (fromManifest); the standard ones share STANDARD_POOLS' objects. */
const MODEL_PRESET_INPUTS: PresetInput[] = DEPLOY_PRESETS.map((p) => ({
  id: p.id,
  title: p.title,
  description: p.description,
  ...(p.prequant ? { prequant: [...p.prequant] } : {}),
  ...(p.licence !== undefined ? { licence: p.licence } : {}),
  pool: p.standard ? STANDARD_POOLS.find((s) => s.id === p.id)! : poolOf(p),
}));
export const MODEL_PRESETS: PoolPreset[] = MODEL_PRESET_INPUTS.map(fromManifest);

// ---------------------------------------------------------------- family presets
// Owner decision 2026-10-18: "for the presets means all of the weights for a model family and it's variants are
// installed onto 1 [pod]". A family preset is ONE pool (one pod by default, one GPU, one fv-serve process) whose
// worker config lists every variant of the family (configs/serve/runpod-family-*.toml): the first resident, the rest
// `resident = false` with `[engine] swap = true` (one DiT at a time; a request for another variant unloads the
// resident one and loads it from the pod's disk, each with its own technique profile and load-time settings: the
// engine's ProcessPlan and settings::scope_model). Its weights are the union of the members' trees, so its container
// disk (r2) is sized for the union; its GPU fits the largest member. Its members are listed in deploy/deploy.toml; the
// rules below say which model presets a family must hold, and test/unit/templates.test.ts fails when the listed
// members differ from them or a member's models are not in the family's config, so a new preset joins its family.
const recipeStream = new Map(CATALOG.recipes.map((r) => [r.id, (r as { stream?: string }).stream ?? null]));
/** The model presets every one of whose models matches (in the presets' order). */
export function presetsWhere(ok: (m: ModelRef) => boolean): string[] {
  return MODEL_PRESETS.filter((p) => (p.pool.models || []).length > 0 && (p.pool.models || []).every(ok)).map((p) => p.id);
}
/** Every H3 variant a preset serves (turbo, max, Ref2V, the NVFP4 tier, previews, Trim, VDN-H3, the turbo LoRAs). */
export const H3_FAMILY_MEMBERS = presetsWhere((m) => m.family === "h3");
/** Every FastH3 variant: the H3 presets whose models are FastH3 releases (model ids fasth3*). */
export const FASTH3_FAMILY_MEMBERS = presetsWhere((m) => m.family === "h3" && /^fasth3/.test(m.id));
/** Every streaming model: presets whose recipes the catalog marks causal (caps.rs StreamCaps::Causal). */
export const STREAMING_FAMILY_MEMBERS = presetsWhere((m) => recipeStream.get(m.recipe) === "causal");
/** Every LTX-2.5 preset (turbo, pro, guided A2V, Ref2V). */
export const LTX_FAMILY_MEMBERS = presetsWhere((m) => m.family === "ltx2");
/** A serve config's [[models]] as pool models: the id (default: the variant's id), the variant's family, and the
 * variant as written for the recipe (plan A3: a config names a variant of families/*.toml). */
export function configModelRefs(toml: string): ModelRef[] {
  const doc = readToml(toml);
  return (Array.isArray(doc.models) ? doc.models : []).map((m: Record<string, unknown>) => {
    const v = mustVariant(String(m.variant ?? ""));
    return { id: typeof m.id === "string" && m.id ? m.id : v.id, family: v.family, recipe: String(m.variant) };
  });
}
/** The catalog's 480p TAEH3 draft tier: no model preset of its own; the H3 and FastH3 family pools serve it. */
export const H3_DRAFT: ModelRef = { id: "fasth3-4step-vsa-480p-taeh3", family: "h3", recipe: "h3-draft" };

/** A deploy/deploy.toml family preset as a pool preset: `{members}` in its title and description is its members
 * (comma separated); its weights the union of the members' trees; its queue, timeouts and disk the members' largest;
 * its min_vram_gb hand-chosen (deploy/deploy.toml: the NVFP4 members need sm_100, which only 96 GB cards offer here;
 * fv-control filters by memory only). */
function familyPreset(o: DeployFamilyPreset): PoolPreset {
  const ms = o.members.map((id) => MODEL_PRESETS.find((p) => p.id === id)!);
  const names = (s: string) => s.split("{members}").join(o.members.join(", "));
  const toml = inline(o.config);
  // The pool's models are its config's [[models]], in its order (the edge routes each id to this pool); the members'
  // models (and the draft tier, H3_DRAFT, for the H3 families) must be exactly those (test/unit/templates.test.ts).
  // A config names a variant (plan A3: `variant`, `id` defaulting to the variant's id); a model keeps the recipe its
  // member preset uses, which must resolve to the same variant.
  const known = new Map([...ms.flatMap((p) => p.pool.models ?? []), H3_DRAFT].map((m) => [m.id, m]));
  const models: ModelRef[] = configModelRefs(toml).map((m) => {
    const k = known.get(m.id);
    if (k && mustVariant(k.recipe).id !== mustVariant(m.recipe).id) throw new Error(`${o.config}: model ${m.id} is variant ${m.recipe}, its preset says ${k.recipe}`);
    return k ?? m;
  });
  const max = (k: "max_queued" | "job_timeout_s" | "stale_after_s" | "container_disk_gb") => {
    const v = ms.map((p) => p.pool[k]).filter((x): x is number => typeof x === "number");
    return v.length ? Math.max(...v) : undefined;
  };
  const disk = max("container_disk_gb");
  const licence = uniq(ms.map((p) => p.licence).filter((x): x is string => !!x)).join("; ");
  return {
    id: o.id,
    title: names(o.title),
    description: names(o.description),
    min_vram_gb: o.min_vram_gb,
    weights: uniq(ms.flatMap((p) => p.weights)),
    prequant: [...o.prequant],
    ...(licence ? { licence } : {}),
    members: [...o.members],
    ...(o.left_out.length ? { left_out: o.left_out.map((l) => ({ ...l })) } : {}),
    pool: {
      id: o.id,
      variant: o.image as Variant,
      count: 1,
      compute: "GPU",
      config_toml: toml,
      ...(disk !== undefined ? { container_disk_gb: disk } : {}),
      models,
      max_queued: max("max_queued") ?? 32,
      job_timeout_s: max("job_timeout_s") ?? 1800,
      stale_after_s: max("stale_after_s") ?? 120,
    },
  };
}
/** One pool per model family, every variant of it on one pod (owner 2026-10-18; deploy/deploy.toml [[family_presets]]). */
export const FAMILY_PRESETS: PoolPreset[] = DEPLOY_FAMILY_PRESETS.map(familyPreset);
/** The model presets, then the family presets. */
export const POOL_PRESETS: PoolPreset[] = [...MODEL_PRESETS, ...FAMILY_PRESETS];
/** Wan's family pool is the wan preset itself (2.2 5B turbo + max, one process with swap); what it leaves out, and why. */
export const WAN_FAMILY_LEFT_OUT = [
  { what: "fastwan21 (FastWan 2.1 1.3B)", why: "its VSA is a process-wide setting (FASTVIDEO_VSA=1) the 5B tiers run without: its own pool" },
  { what: "sfwan / longlive (causal streaming)", why: "an SF-Wan pipeline stays resident once loaded, and the two 5B tiers with their f32 UMT5 encoders and a 720p decode already need the card: the streaming family's pool" },
];
export const presetPool = (id: string): PoolSpec | undefined => {
  const p = POOL_PRESETS.find((x) => x.id === id || x.pool.id === id)?.pool;
  return p ? structuredClone(p) : undefined;
};
export const presetById = (id: string): PoolPreset | undefined => POOL_PRESETS.find((x) => x.id === id);
