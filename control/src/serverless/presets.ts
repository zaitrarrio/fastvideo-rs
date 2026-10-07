// Model-first serverless endpoints (docs/control/serverless.md §1a): a
// preset from the one catalog (../presets.ts POOL_PRESETS, plus `cpu`, the
// fake engine) fixes the image variant, the worker config (a file in the
// image, or inline), the weight trees, the GPU memory and sensible disk and
// timeouts. servingIssues checks what Runpod cannot tell us before a worker
// boots: the config is in the image, the weights are on the volume, the GPU
// types have the memory, a GPU preset has a volume.
import { IMAGE_CONFIGS } from "../cluster/image-configs";
import { REGIONS } from "../cluster/regions";
import { WORKER_CONFIGS } from "../cluster/worker-configs";
import { SLS_PRESET_IDS } from "../enums";
import { gpuMemoryGb, gpusWithAtLeast } from "../gpus";
import { POOL_PRESETS } from "../presets";
import { missingTrees, VOLUME_TREES_SOURCE } from "../volumes";
import { examplesFor, type InvokeExample } from "./examples";
import { servesOf, type Serves } from "./serves";

export type SlsPresetId = (typeof SLS_PRESET_IDS)[number];
export interface SlsPreset {
  id: SlsPresetId;
  title: string;
  description: string;
  variant: string;
  compute: "GPU" | "CPU";
  /** A config file of the image other than its baked default (FV_CONFIG). */
  config?: string;
  /** An inline config (FV_WORKER_TOML_B64): one the image does not carry. */
  config_toml?: string;
  weights: string[];
  min_vram_gb: number;
  container_disk_gb: number;
  execution_timeout_s: number;
  licence?: string;
}

/** Every serverless preset: the pool presets (their image-default config left unset, so the image's own entrypoint runs it) and `cpu`. */
export const SLS_PRESETS: SlsPreset[] = [
  ...POOL_PRESETS.map((p): SlsPreset => {
    const pool = p.pool;
    const img = IMAGE_CONFIGS[pool.variant];
    return {
      id: p.id as SlsPresetId,
      title: p.title,
      description: p.description,
      variant: pool.variant,
      compute: pool.compute === "CPU" ? "CPU" : "GPU",
      ...(pool.config && pool.config !== img?.default ? { config: pool.config } : {}),
      ...(pool.config_toml ? { config_toml: pool.config_toml } : {}),
      weights: p.weights,
      min_vram_gb: p.min_vram_gb,
      container_disk_gb: pool.container_disk_gb ?? 20,
      execution_timeout_s: pool.job_timeout_s ?? 1800,
      ...(p.licence ? { licence: p.licence } : {}),
    };
  }),
  { id: "cpu", title: "Fake engine (CPU, tests)", description: "The cpu image's fake engine on CPU workers: every API, placeholder outputs, no GPU and no weights. For tests.", variant: "cpu", compute: "CPU", weights: [], min_vram_gb: 0, container_disk_gb: 20, execution_timeout_s: 600 },
];
export const slsPreset = (id: unknown): SlsPreset | undefined => SLS_PRESETS.find((p) => p.id === id);

/** The fields a preset owns (an input may repeat them, not change them). */
export const PRESET_OWNED = ["variant", "compute", "config", "config_toml"] as const;
export function presetOwned(p: SlsPreset): Record<(typeof PRESET_OWNED)[number], unknown> {
  return { variant: p.variant, compute: p.compute, config: p.config, config_toml: p.config_toml };
}

/** The preset a spec without one runs anyway (same variant, compute and config), or null (a custom spec). */
export function inferPreset(spec: { variant: string; compute: string; config?: string; config_toml?: string }): SlsPresetId | null {
  const def = IMAGE_CONFIGS[spec.variant]?.default;
  const path = (c?: string) => c || def;
  for (const p of SLS_PRESETS) {
    if (p.variant !== spec.variant || p.compute !== spec.compute) continue;
    if (p.config_toml ? spec.config_toml === p.config_toml : !spec.config_toml && path(spec.config) === path(p.config)) return p.id;
  }
  return null;
}

export interface SpecLike {
  preset?: string;
  mode: "queue" | "lb";
  variant: string;
  compute: "GPU" | "CPU";
  config?: string;
  config_toml?: string;
  image: { channel?: string; sha?: string; ref?: string };
  gpu_types?: string[];
  network_volume: string | null;
}
export interface Issue {
  path: (string | number)[];
  message: string;
}

/** The least GPU memory a spec needs: its preset's, else the most any served model's family needs. */
export function minVramGb(spec: SpecLike, s: Serves): number {
  const p = slsPreset(spec.preset ?? inferPreset(spec));
  if (p) return p.min_vram_gb;
  if (s.engine === "fake" || spec.compute === "CPU") return 0;
  const of = (fam: string, recipe: string) => (fam === "h3" || fam === "ltx2" ? 80 : /5b|14b/.test(recipe) ? 32 : 24);
  return Math.max(0, ...s.models.map((m) => of(m.family, m.recipe)));
}

/**
 * The checks only fv-control can make before a worker boots, field by field:
 * a config path in the image, the weight trees on the volume (a static list,
 * volumes.ts), a volume for a config that loads weights, the GPU types'
 * memory. Warnings: a licence, an image ref (its configs cannot be checked).
 */
export function servingIssues(spec: SpecLike, s: Serves = servesOf(spec)): { issues: Issue[]; warnings: Issue[] } {
  const issues: Issue[] = [];
  const warnings: Issue[] = [];
  const preset = slsPreset(spec.preset);
  if (spec.config) {
    const img = IMAGE_CONFIGS[spec.variant];
    const file = spec.config.startsWith("/etc/fv/") ? spec.config.slice("/etc/fv/".length) : null;
    if (spec.image.ref) warnings.push({ path: ["config"], message: `an image ref: fv-control cannot check that ${spec.config} is in it` });
    else if (img && (!file || !img.files.includes(file))) {
      const text = file ? WORKER_CONFIGS[file] : undefined;
      const inline = text ? SLS_PRESETS.find((p) => p.config_toml === text) : undefined;
      issues.push({ path: ["config"], message: `the ${spec.variant} image carries ${img.files.map((f) => `/etc/fv/${f}`).join(", ")}; ${spec.config} is not in it${inline ? ` (preset ${inline.id} sends that config inline)` : " (a config the image does not carry goes inline: config_toml, or pick a preset)"}` });
    }
  }
  if (!s.models.length && s.source.kind === "unknown" && !spec.config) warnings.push({ path: ["variant"], message: "fv-control does not know this variant's config: nothing derived" });
  const trees = [...new Set([...(preset?.weights ?? []), ...s.weights])];
  if (spec.compute === "GPU" && s.engine !== "fake") {
    if (!spec.network_volume) {
      // A preset's missing volume is the schema's issue (spec.ts superRefine).
      if (trees.length && !preset) issues.push({ path: ["network_volume"], message: `the config loads weight trees (${trees.join(", ")}): mount the weights volume ${REGIONS.eu.volume} (${REGIONS.eu.dc})` });
    } else {
      const miss = missingTrees(spec.network_volume, trees);
      if (miss === null) warnings.push({ path: ["network_volume"], message: `fv-control has no tree list for volume ${spec.network_volume}: the weights were not checked` });
      else if (miss.length) issues.push({ path: ["network_volume"], message: `volume ${spec.network_volume} has no weights tree ${miss.join(", ")} (needed by ${preset ? `preset ${preset.id}` : "the config"}; source: ${VOLUME_TREES_SOURCE})` });
    }
    const need = minVramGb(spec, s);
    const gpus = spec.gpu_types?.length ? spec.gpu_types : REGIONS.eu.gpus;
    if (need > 0)
      gpus.forEach((g, i) => {
        const gb = gpuMemoryGb(g);
        const who = preset ? `preset ${preset.id}` : "the models served";
        if (gb !== null && gb < need) issues.push({ path: spec.gpu_types?.length ? ["gpu_types", i] : ["gpu_types"], message: `${g} has ${gb} GB; ${who} needs a GPU with at least ${need} GB (e.g. ${gpusWithAtLeast(need).filter((x) => x.includes("RTX PRO 6000") || x.includes("H100")).slice(0, 2).join(", ")})` });
      });
  }
  if (preset?.licence) warnings.push({ path: ["preset"], message: preset.licence });
  return { issues, warnings };
}

/** What the validate route, the endpoint page and the CLI show: the derived serving view, the preset (given or inferred), the examples. */
export function servingView(spec: SpecLike): { preset: string | null; preset_inferred: boolean; serves: Serves; examples: InvokeExample[] } {
  const serves = servesOf(spec);
  const inferred = spec.preset ? null : inferPreset(spec);
  return { preset: spec.preset ?? inferred, preset_inferred: !spec.preset && !!inferred, serves, examples: examplesFor(serves, spec.mode) };
}

/**
 * An update's document merged over the stored spec (PUT /api/serverless/:id takes a partial spec). A
 * different preset brings its own variant, compute and config, so the stored ones are dropped; a
 * document that names a variant or config but no preset is a custom endpoint, so the stored preset is
 * dropped (`preset: null` drops it too).
 */
export function mergeSpec(prev: Record<string, any>, patch: Record<string, any>): Record<string, any> {
  const base = { ...prev };
  if ("preset" in patch && patch.preset !== prev.preset) for (const k of PRESET_OWNED) delete base[k];
  else if (!("preset" in patch) && PRESET_OWNED.some((k) => k in patch && k !== "compute" && patch[k] !== prev[k])) delete base.preset;
  const out = { ...base, ...patch };
  if (out.preset === null) delete out.preset;
  return out;
}

/** Fields whose change makes an update re-check what the endpoint serves (servingIssues). */
export const SERVING_FIELDS = ["preset", "variant", "compute", "config", "config_toml", "image", "gpu_types", "network_volume"] as const;
