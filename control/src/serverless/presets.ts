// Model-first serverless endpoints (docs/control/serverless.md §1a): a
// preset from the one catalog (../presets.ts POOL_PRESETS, plus `cpu`, the
// fake engine) fixes the compute (the CUDA image, or the cpu image), the
// worker config (sent at boot: the images bake none), the weight trees, the
// GPU memory and sensible disk and timeouts. servingIssues checks what Runpod
// cannot tell us before a worker boots: a config fv-control can send, the
// weights are on the volume, the GPU types have the memory, a GPU preset has
// a volume.
import { REGIONS } from "../cluster/regions";
import { legacyVariant, SLS_PRESET_IDS } from "../enums";
import { workerConfigProblem } from "../worker-config";
import { gpuMemoryGb, gpusWithAtLeast } from "../gpus";
import { POOL_PRESETS } from "../presets";
import { missingTrees, VOLUME_TREES_SOURCE } from "../volumes";
import { examplesFor, type InvokeExample } from "./examples";
import { configOf, servesOf, type Serves } from "./serves";

export type SlsPresetId = (typeof SLS_PRESET_IDS)[number];
export interface SlsPreset {
  id: SlsPresetId;
  title: string;
  description: string;
  /** GPU: the CUDA image; CPU: the cpu image (the fake engine). */
  compute: "GPU" | "CPU";
  /** A named config (a configs/serve file fv-control sends at boot). */
  config?: string;
  /** An inline config: both are sent as FV_WORKER_TOML_B64 (the images bake none). */
  config_toml?: string;
  weights: string[];
  /** Pre-quantized trees fetched with weights_source r2 (../presets.ts PoolPreset.prequant). */
  prequant: string[];
  min_vram_gb: number;
  container_disk_gb: number;
  execution_timeout_s: number;
  licence?: string;
  /** The prompt upsampler's local sidecar on by default (upsampler.ts; the pool preset's `upsampler_local`). */
  upsampler_local?: boolean;
}

/** Every serverless preset: the pool presets (their config sent at boot) and `cpu` (the fake engine's config). */
export const SLS_PRESETS: SlsPreset[] = [
  ...POOL_PRESETS.map((p): SlsPreset => {
    const pool = p.pool;
    return {
      id: p.id as SlsPresetId,
      title: p.title,
      description: p.description,
      compute: pool.compute === "CPU" ? "CPU" : "GPU",
      ...(pool.config ? { config: pool.config } : {}),
      ...(pool.config_toml ? { config_toml: pool.config_toml } : {}),
      weights: p.weights,
      prequant: p.prequant ?? [],
      min_vram_gb: p.min_vram_gb,
      container_disk_gb: pool.container_disk_gb ?? 20,
      execution_timeout_s: pool.job_timeout_s ?? 1800,
      ...(p.licence ? { licence: p.licence } : {}),
      ...(pool.upsampler_local ? { upsampler_local: true } : {}),
    };
  }),
  { id: "cpu", title: "Fake engine (CPU, tests)", description: "The cpu image's fake engine on CPU workers: every API, placeholder outputs, no GPU and no weights. For tests.", compute: "CPU", config: "runpod-fake.toml", weights: [], prequant: [], min_vram_gb: 0, container_disk_gb: 20, execution_timeout_s: 600 },
];
export const slsPreset = (id: unknown): SlsPreset | undefined => SLS_PRESETS.find((p) => p.id === id);

/** The fields a preset owns (an input may repeat them, not change them). */
export const PRESET_OWNED = ["compute", "config", "config_toml"] as const;
export function presetOwned(p: SlsPreset): Record<(typeof PRESET_OWNED)[number], unknown> {
  return { compute: p.compute, config: p.config, config_toml: p.config_toml };
}

/** The preset a spec without one runs anyway (same compute and the same config text), or null (a custom spec). */
export function inferPreset(spec: { compute: string; config?: string; config_toml?: string }): SlsPresetId | null {
  const mine = configOf(spec).text;
  if (!mine) return null;
  for (const p of SLS_PRESETS) if (p.compute === spec.compute && configOf(p).text === mine) return p.id;
  return null;
}

export interface SpecLike {
  preset?: string;
  mode: "queue" | "lb";
  compute: "GPU" | "CPU";
  config?: string;
  config_toml?: string;
  image: { channel?: string; sha?: string; ref?: string };
  gpu_types?: string[];
  network_volume: string | null;
  weights_source?: "volume" | "r2" | "hub";
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
 * a config fv-control can send (named or inline, within the env size guard),
 * the weight trees on the volume (a static list, volumes.ts), a volume for a
 * config that loads weights, the GPU types' memory. Warnings: a licence.
 */
export function servingIssues(spec: SpecLike, s: Serves = servesOf(spec)): { issues: Issue[]; warnings: Issue[] } {
  const issues: Issue[] = [];
  const warnings: Issue[] = [];
  const preset = slsPreset(spec.preset);
  // Every endpoint sends its config at boot (the images bake none): one fv-control can send.
  const cfg = workerConfigProblem(spec);
  if (cfg) issues.push({ path: [spec.config_toml ? "config_toml" : "config"], message: cfg });
  const trees = [...new Set([...(preset?.weights ?? []), ...s.weights])];
  if (spec.compute === "GPU" && s.engine !== "fake") {
    if (spec.weights_source === "r2" || spec.weights_source === "hub") {
      // The R2 weights store or the Hub: serverless/weights.ts slsWeightsIssues checks the trees are published / have a Hub source.
    } else if (!spec.network_volume) {
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
 * An update's document merged over the stored spec (PUT /api/v1/serverless/:id takes a partial spec). A
 * different preset brings its own compute and config, so the stored ones are dropped; a document that
 * names a config but no preset is a custom endpoint, so the stored preset is dropped (`preset: null`
 * drops it too). A stored spec or a patch naming an image `variant` (before the one-image decision)
 * reads as its compute (enums.ts legacyVariant).
 */
export function mergeSpec(prevIn: Record<string, any>, patchIn: Record<string, any>): Record<string, any> {
  const prev = legacyVariant({ ...prevIn });
  const patch = legacyVariant({ ...patchIn });
  const base = { ...prev };
  if ("preset" in patch && patch.preset !== prev.preset) for (const k of PRESET_OWNED) delete base[k];
  else if (!("preset" in patch) && PRESET_OWNED.some((k) => k in patch && k !== "compute" && patch[k] !== prev[k])) delete base.preset;
  const out = { ...base, ...patch };
  if (out.preset === null) delete out.preset;
  return out;
}

/** Fields whose change makes an update re-check what the endpoint serves (servingIssues). */
export const SERVING_FIELDS = ["preset", "compute", "config", "config_toml", "image", "gpu_types", "network_volume", "weights_source", "container_disk_gb"] as const;
