// The preset catalog: an image variant plus the worker config it runs, the
// models it serves, the weight trees it loads and the GPU memory it needs.
// One source for cluster pools (cluster/spec.ts re-exports it), standalone
// pods (standalone.ts) and serverless endpoints (serverless/presets.ts).
// Kept free of runtime imports from ./schemas and ./cluster/spec (both import
// this module's users), so there is no module cycle.
import type { ModelRef, PoolSpec } from "./cluster/spec";
import CATALOG from "./cluster/catalog.json";
import { WORKER_CONFIGS } from "./cluster/worker-configs";
import { MANIFEST_VARIANTS } from "./families-gen";
import { RECIPE_TREES } from "./recipe-trees";

export const STANDARD_POOLS: PoolSpec[] = [
  { id: "h3-turbo", variant: "h3-turbo", count: 1, compute: "GPU", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  { id: "h3-max", variant: "h3-max", count: 1, compute: "GPU", config: "/etc/fv/runpod-h3-max.toml", models: [{ id: "sol-h3", family: "h3", recipe: "h3-max" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 180 },
  { id: "ltx", variant: "ltx", count: 1, compute: "GPU", config: "/etc/fv/runpod-ltx.toml", models: [{ id: "ltx25-distill-sol", family: "ltx2", recipe: "ltx-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  {
    id: "wan",
    variant: "wan5b",
    count: 1,
    compute: "GPU",
    config: "/etc/fv/runpod-wan5b.toml",
    models: [
      { id: "fastwan22-ti2v-5b", family: "wan", recipe: "wan-turbo" },
      { id: "wan22-ti2v-5b", family: "wan", recipe: "wan-max" },
    ],
    max_queued: 64,
    job_timeout_s: 1800,
    stale_after_s: 120,
  },
];

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
  const t = RECIPE_TREES[r];
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
const inline = (file: string) => {
  const t = WORKER_CONFIGS[file];
  if (!t) throw new Error(`no generated worker config ${file} (node gen-configs.mjs)`);
  return t;
};
// The presets reuse the image variants CI builds (docs/serve/images.md); a
// config the variant's image does not carry rides inline (FV_WORKER_TOML_B64,
// generated from configs/serve by gen-configs.mjs).
/** The presets of one model (or one tier and its companions) each. Their weights and min_vram_gb derive from the
 * family manifests (fromManifest); the rest (id, image variant, config, queue, timeouts) is deployment data the
 * manifests do not carry yet (plan C1). */
const MODEL_PRESET_INPUTS: PresetInput[] = [
  { id: "h3-turbo", title: "H3 turbo (fasth3)", description: "MiniMax H3 turbo tier; also the Reactor's clip model.", prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"], pool: STANDARD_POOLS[0]! },
  { id: "h3-max", title: "H3 max (Sol-H3)", description: "MiniMax H3 max tier (Sol-H3 4-step ladder).", prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_sol-h3_mxfp8"], pool: STANDARD_POOLS[1]! },
  { id: "ltx", title: "LTX turbo", description: "LTX-2.5 turbo tier (Sol stage 2): fal lightricks/ltx-2.5 /fast.", pool: STANDARD_POOLS[2]! },
  { id: "wan", title: "Wan 2.2 5B (turbo + max)", description: "Wan 2.2 TI2V-5B: FastWan turbo and the 50-step max tier; fal-ai/wan.", pool: STANDARD_POOLS[3]! },
  {
    id: "ltx-pro",
    title: "LTX pro (dense)",
    description: "LTX-2.5 pro tier (ltx25-distill-dense, two-stage dense): fal lightricks/ltx-2.5 /pro and fal-ai/ltx-2.3 retake / extend. Sage attention is on by default on sm_120 (RTX PRO 6000; FASTVIDEO_ATTN_SAGE=0 turns it off).",
    pool: { id: "ltx-pro", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-pro.toml"), models: [{ id: "ltx25-distill-dense", family: "ltx2", recipe: "ltx-pro" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-a2v",
    title: "LTX guided audio-to-video",
    description: "The guided A2V companion of ltx-pro (LTX-2.5 dev DiT, multimodal guider): fal lightricks/ltx-2.5 audio-to-video/pro. An 80-96 GB card and ~38 GB of host RAM.",
    pool: { id: "ltx-a2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-a2v.toml"), container_disk_gb: 60, models: [{ id: "ltx25-a2v-guided", family: "ltx2", recipe: "ltx25-a2v-guided" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-ref2v",
    title: "LTX reference-to-video",
    description: "The Ref2V companion of ltx-pro (Ingredients IC-LoRA): fal fal-ai/ltx-2.3-quality ingredient.",
    pool: { id: "ltx-ref2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-ref2v.toml"), models: [{ id: "ltx25-ref2v", family: "ltx2", recipe: "ltx25-ref2v" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "h3-ref2v",
    title: "H3 reference-to-video",
    description: "H3 Ref2VA, turbo and max (the reference companions of h3-turbo / h3-max).",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: {
      id: "h3-ref2v",
      variant: "h3-max",
      count: 1,
      compute: "GPU",
      config_toml: inline("runpod-h3-ref2v.toml"),
      models: [
        { id: "h3-ref2v-turbo", family: "h3", recipe: "h3-ref2v-turbo" },
        { id: "h3-ref2v-max", family: "h3", recipe: "h3-ref2v-max" },
      ],
      max_queued: 16,
      job_timeout_s: 3600,
      stale_after_s: 180,
    },
  },
  {
    id: "fasth3-trim",
    title: "FastH3 Trim 8-step",
    description: "FastH3 Trim (FastVideo/FastVideo-FastH3-Trim-8-Step): 42 of the 50 H3 blocks, 8 DMD steps, VSA; text-to-video with audio only. Faster and smaller than FastH3 V2, with some quality cost. Served as h3-trim.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-trim", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-trim.toml"), models: [{ id: "fasth3-trim-8step", family: "h3", recipe: "h3-trim" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // FastH3 Trim with the release's NVFP4 DiT (docs/ports/h3-trim.md §5): opt-in, the fasth3-trim preset stays the
  // default. NVFP4 tensor cores only (RTX PRO 6000 / B200): the model fails to load on any other GPU.
  {
    id: "fasth3-trim-nvfp4",
    title: "FastH3 Trim 8-step NVFP4",
    description: "FastH3 Trim with the release's NVFP4 DiT (FastVideo/FastVideo-FastH3-Trim-Comfy): every block's attention and FFN linears W4A4 on NVFP4 tensor cores (RTX PRO 6000 / B200 only, no fallback); text-to-video with audio only. Opt-in; served as h3-trim-nvfp4.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-trim-nvfp4", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-trim-nvfp4.toml"), models: [{ id: "fasth3-trim-nvfp4", family: "h3", recipe: "h3-trim-nvfp4" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // The H3 NVFP4 tier (owner decision 2026-10-14; docs/perf/h3-nvfp4-tier-options.md): the FastH3 V2 / 4-step NVFP4
  // releases, each decoding with its own published NVFP4 VAE by default (FASTVIDEO_H3_VAE_NVFP4=0: h3-base's, MXFP8).
  // Opt-in, h3-turbo stays the default. NVFP4 tensor cores only (RTX PRO 6000 / B200). The release trees are in R2 only.
  {
    id: "fasth3-v2-nvfp4",
    title: "H3 NVFP4 tier: FastH3 V2 NVFP4",
    description: "FastH3 V2 NVFP4 (FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4): the release's ModelOpt NVFP4 DiT, 8 DMD steps, sparse FP4 VSA, and its own published NVFP4 VAE decoder (gate pass, 5 GiB less peak; FASTVIDEO_H3_VAE_NVFP4=0 decodes with h3-base's VAE). NVFP4 tensor cores only (RTX PRO 6000 / B200, no fallback); text-to-video with audio. Opt-in; served as h3-v2-nvfp4.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-v2-nvfp4", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-v2-nvfp4.toml"), models: [{ id: "fasth3-v2-nvfp4", family: "h3", recipe: "h3-v2-nvfp4" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "fasth3-4step-nvfp4",
    title: "H3 NVFP4 tier: FastH3 4-step NVFP4",
    description: "FastH3 4-step VSA DataFree NVFP4 (FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4): the release's ModelOpt NVFP4 DiT, 4 DMD steps (VSA 0.9), sparse FP4 VSA, and its own published NVFP4 VAE decoder (FASTVIDEO_H3_VAE_NVFP4=0: h3-base's). NVFP4 tensor cores only (RTX PRO 6000 / B200, no fallback); text-to-video with audio. Opt-in; served as h3-turbo-nvfp4.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-4step-nvfp4", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-turbo-nvfp4.toml"), models: [{ id: "fasth3-4step-vsa-nvfp4", family: "h3", recipe: "h3-turbo-nvfp4" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // "Fast preview" (owner decision 2026-10-14): the NVFP4 tier's recipes and h3-turbo decoded by TAEH3, each its own
  // model id and pool. An explicit choice, never a default: fails the quality gate's sharpness floor (0.82-0.91),
  // saves about 10-12% end to end (docs/perf/h3-nvfp4-tier-options.md option 5).
  {
    id: "fasth3-v2-nvfp4-preview",
    title: "H3 NVFP4 tier, fast preview (TAEH3)",
    description: "FastH3 V2 NVFP4 decoded by TAEH3 instead of the official VAE: a fast preview for drafts and scrubbing, never a default. Fails the quality gate's sharpness floor (0.82-0.91, softer frames of the same picture); about 10-12% faster end to end. NVFP4 tensor cores only; text-to-video with audio. Served as h3-v2-nvfp4-preview.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-v2-nvfp4-preview", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-v2-nvfp4-preview.toml"), models: [{ id: "fasth3-v2-nvfp4-taeh3", family: "h3", recipe: "h3-v2-nvfp4-preview" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "fasth3-4step-nvfp4-preview",
    title: "H3 NVFP4 4-step, fast preview (TAEH3)",
    description: "FastH3 4-step VSA DataFree NVFP4 decoded by TAEH3 instead of the official VAE: a fast preview, never a default. Fails the quality gate's sharpness floor (0.82-0.91); about 10-12% faster end to end. NVFP4 tensor cores only; text-to-video with audio. Served as h3-turbo-nvfp4-preview.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-4step-nvfp4-preview", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-turbo-nvfp4-preview.toml"), models: [{ id: "fasth3-4step-vsa-nvfp4-taeh3", family: "h3", recipe: "h3-turbo-nvfp4-preview" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "h3-turbo-preview",
    title: "H3 turbo, fast preview (TAEH3)",
    description: "The h3-turbo recipe (FastH3 Preview 4-step VSA, MXFP8) decoded by TAEH3 instead of the official VAE, at 768p and 480p: a fast preview, never a default (the 480p draft tier is h3-draft). Fails the quality gate's sharpness floor; about 10-12% faster end to end. Served as h3-turbo-preview.",
    prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"],
    pool: { id: "h3-turbo-preview", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-turbo-preview.toml"), models: [{ id: "fasth3-4step-vsa-taeh3", family: "h3", recipe: "h3-turbo-preview" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // The opt-in "fastest" H3 tier (owner decision 2026-10-14): the Trim NVFP4 with sparse FP4 VSA turned on by its
  // recipe. A different model from FastH3 V2 and not gate-equivalent. RTX PRO 6000 (sm_120 kernel) only.
  {
    id: "fasth3-trim-nvfp4-fast",
    title: "FastH3 Trim NVFP4 + FP4 attention (fastest, opt-in)",
    description: "The fastest H3 configuration measured: FastH3 Trim 8-step with the release's NVFP4 DiT and the VSA fine stage on sparse FP4 tensor cores (9.2 s denoise at 480p, 24.6 s at 768p on an RTX PRO 6000, about 20 GiB peak). A DIFFERENT MODEL from FastH3 V2 and NOT gate-equivalent to any H3 tier: judge it by eye. RTX PRO 6000 (sm_120) only, no fallback; text-to-video with audio only. Opt-in; served as h3-trim-nvfp4-fast.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-trim-nvfp4-fast", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-trim-nvfp4-fast.toml"), models: [{ id: "fasth3-trim-nvfp4-fast", family: "h3", recipe: "h3-trim-nvfp4-fast" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // MiniMax-H3 Turbo LoRAs (docs/ports/h3-turbo-loras.md): base H3 + one adapter folded at load. The adapters are in R2
  // only (not on the EU volume), so these pools fetch from R2 by default.
  {
    id: "h3-larry-turbo",
    title: "MiniMax-H3 Turbo LoRA (larry)",
    description: "MiniMax-H3 with larryvrh/MiniMax-H3-Turbo-Lora (ckpt850 EMA, Apache-2.0) folded in at load, as the MiniMaxAI/MiniMax-H3-Turbo-Lora Space runs it: 6 forwards (4 allowed), dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-larry-turbo", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-larry-turbo.toml"), models: [{ id: "h3-larry-turbo", family: "h3", recipe: "h3-larry-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "h3-lightx2v-turbo",
    title: "MiniMax-H3 Turbo LoRA (lightx2v)",
    description: "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v1.2 768p (Apache-2.0) folded in at load: 4 forwards at shifts 6 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-lightx2v-turbo", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-lightx2v-turbo.toml"), models: [{ id: "h3-lightx2v-turbo", family: "h3", recipe: "h3-lightx2v-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "vdn-h3",
    title: "VDN-H3 (Video DeltaNet)",
    description: "VDN-H3 (OpenVDN/vdn-minimax-h3, stage B): MiniMax-H3 with hybrid attention, a chunk-aligned window softmax plus a frame-wise delta-rule linear branch in every block; 50 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "vdn-h3", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-vdn-h3.toml"), models: [{ id: "vdn-h3", family: "h3", recipe: "vdn-h3-50step" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 300 },
  },
  {
    id: "vdn-h3-8step",
    title: "VDN-H3 8-step (Video DeltaNet, DMD)",
    description: "VDN-H3 8-step (OpenVDN/vdn-minimax-h3 stage-dmd-step-250): the stage-B hybrid attention plus the DMD turbo LoRA, 8 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3-8step.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "vdn-h3-8step", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-vdn-h3-8step.toml"), models: [{ id: "vdn-h3-8step", family: "h3", recipe: "vdn-h3-8step" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "h3-lightx2v-turbo-v01",
    title: "MiniMax-H3 Turbo LoRA (lightx2v v0.1, the Space's)",
    description: "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v0.1 (Apache-2.0), the file the MiniMax-H3-Turbo-Lora Space folds: 4 forwards at shifts 12 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-lightx2v-turbo-v01", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-lightx2v-turbo-v01.toml"), models: [{ id: "h3-lightx2v-turbo-v01", family: "h3", recipe: "lightx2v-turbo-v0.1" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "fastwan21",
    title: "FastWan 2.1 1.3B",
    description: "FastWan 2.1 T2V 1.3B (480p, 3 DMD steps): fal fastvideo/fastwan21-1.3b.",
    pool: { id: "fastwan21", variant: "wan", count: 1, compute: "GPU", config: "/etc/fv/runpod-wan.toml", models: [{ id: "fastwan21-1.3b", family: "wan", recipe: "fastwan21-1.3b" }], max_queued: 64, job_timeout_s: 900, stale_after_s: 90 },
  },
  {
    id: "sfwan",
    title: "SF-Wan causal streaming",
    description: "SF-Wan 1.3B causal rollout: Reactor and native streams (one session per GPU); the cluster's Reactor model when no pool serves fasth3.",
    pool: { id: "sfwan", variant: "sfwan", count: 1, compute: "GPU", config: "/etc/fv/runpod-sfwan.toml", models: [{ id: "sfwan21-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
  {
    id: "longlive",
    title: "LongLive-1.3B causal (NON-COMMERCIAL)",
    description: "LongLive-1.3B on the SF-Wan engine (window 12, sink 3, KV re-cache at prompt switches): Reactor and native streams. NON-COMMERCIAL licence: research and evaluation only.",
    // The converted LongLive checkpoint the config's `longlive` path names: wan.toml has no LongLive variant.
    weights_extra: ["longlive-1.3b-safetensors"],
    licence: "LongLive-1.3B weights: CC-BY-NC-SA-4.0 (non-commercial; research / evaluation only)",
    pool: { id: "longlive", variant: "sfwan", count: 1, compute: "GPU", config_toml: inline("runpod-longlive.toml"), models: [{ id: "longlive-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
];
export const MODEL_PRESETS: PoolPreset[] = MODEL_PRESET_INPUTS.map(fromManifest);

// ---------------------------------------------------------------- family presets
// Owner decision 2026-10-18: "for the presets means all of the weights for a model family and it's variants are
// installed onto 1 [pod]". A family preset is ONE pool (one pod by default, one GPU, one fv-serve process) whose
// worker config lists every variant of the family (configs/serve/runpod-family-*.toml): the first resident, the rest
// `resident = false` with `[engine] swap = true` (one DiT at a time; a request for another variant unloads the
// resident one and loads it from the pod's disk, each with its own technique profile and load-time settings: the
// engine's ProcessPlan and settings::scope_model). Its weights are the union of the members' trees, so its container
// disk (r2) is sized for the union; its GPU fits the largest member. The members are derived from the model presets
// and the serve catalog: test/unit/templates.test.ts fails when a new preset's models are not in its family's config.
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
/** The catalog's 480p TAEH3 draft tier: no model preset of its own; the H3 and FastH3 family pools serve it. */
export const H3_DRAFT: ModelRef = { id: "fasth3-4step-vsa-480p-taeh3", family: "h3", recipe: "h3-draft" };

interface FamilyOpts {
  id: string;
  title: string;
  description: string;
  members: string[];
  config: string;
  variant: PoolSpec["variant"];
  /** Hand-chosen (not the members' largest gpu.min_vram_gib): the NVFP4 members need sm_100, which only 96 GB
   * cards offer here (RTX PRO 6000, B200), and two resident SF-Wan pipelines want an 80 GB-class card. The
   * manifests carry gpu.min_sm / needs for this; fv-control filters by memory only (plan C1). */
  min_vram_gb: number;
  prequant?: string[];
  left_out?: { what: string; why: string }[];
}
function familyPreset(o: FamilyOpts): PoolPreset {
  const ms = o.members.map((id) => MODEL_PRESETS.find((p) => p.id === id)!);
  const toml = inline(o.config);
  // The pool's models are its config's [[models]], in its order (the edge routes each id to this pool); the members'
  // models (and the draft tier, H3_DRAFT, for the H3 families) must be exactly those (test/unit/templates.test.ts).
  const models: ModelRef[] = [...toml.matchAll(/^\[\[models\]\]\nid = "([^"]+)"\nfamily = "([^"]+)"\nrecipe = "([^"]+)"/gm)].map((m) => ({ id: m[1]!, family: m[2]!, recipe: m[3]! }));
  const max = (k: "max_queued" | "job_timeout_s" | "stale_after_s" | "container_disk_gb") => {
    const v = ms.map((p) => p.pool[k]).filter((x): x is number => typeof x === "number");
    return v.length ? Math.max(...v) : undefined;
  };
  const disk = max("container_disk_gb");
  const licence = uniq(ms.map((p) => p.licence).filter((x): x is string => !!x)).join("; ");
  return {
    id: o.id,
    title: o.title,
    description: o.description,
    min_vram_gb: o.min_vram_gb,
    weights: uniq(ms.flatMap((p) => p.weights)),
    prequant: o.prequant ?? [],
    ...(licence ? { licence } : {}),
    members: [...o.members],
    ...(o.left_out?.length ? { left_out: o.left_out } : {}),
    pool: {
      id: o.id,
      variant: o.variant,
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
const NOT_8STEP = { what: "fasth3-8step-dense (catalog, no preset)", why: "its 148 GB h3-8step tree would take the union past Runpod's 500 GB container disk" };
/** One pool per model family, every variant of it on one pod (owner 2026-10-18). */
export const FAMILY_PRESETS: PoolPreset[] = [
  familyPreset({
    id: "h3-family",
    title: "H3 family: every H3 variant on one pod",
    description: `Every H3 variant a preset serves on ONE GPU, one fv-serve process (${H3_FAMILY_MEMBERS.join(", ")}, and the 480p draft tier): turbo resident, the others loaded on their first request (one DiT at a time: a switch unloads the resident one and loads the next from the pod's disk, about a minute). RTX PRO 6000 (96 GB, sm_120) or B200: the NVFP4 releases need NVFP4 tensor cores and VDN-H3 a 96 GB card. Weights from R2 by default (the EU volume lacks the NVFP4, Trim, LoRA and VDN trees): about 396 GB, a 485 GB container disk.`,
    members: H3_FAMILY_MEMBERS,
    config: "runpod-family-h3.toml",
    variant: "h3-max",
    min_vram_gb: 96,
    left_out: [
      NOT_8STEP,
      { what: "the MXFP8 pre-quantized trees (77 GB)", why: "they would take the union (about 396 GB, 485 GB of disk) to 573 GB of disk, past Runpod's 500 GB: turbo and max quantize at load instead" },
    ],
  }),
  familyPreset({
    id: "fasth3-family",
    title: "FastH3 family: every FastH3 release on one pod",
    description: `Every FastH3 release a preset serves on ONE GPU, one fv-serve process (${FASTH3_FAMILY_MEMBERS.join(", ")}, and the 480p draft tier): turbo resident, the previews, Trim and the NVFP4 tier loaded on their first request (one DiT at a time). RTX PRO 6000 (96 GB, sm_120) or B200 (NVFP4 tensor cores). Weights from R2 by default: about 371 GB with turbo's pre-quantized trees, a 457 GB container disk.`,
    members: FASTH3_FAMILY_MEMBERS,
    config: "runpod-family-fasth3.toml",
    variant: "h3-max",
    min_vram_gb: 96,
    prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"],
    left_out: [NOT_8STEP],
  }),
  familyPreset({
    id: "ltx-family",
    title: "LTX family: turbo, pro, guided A2V and Ref2V on one pod",
    description: `Every LTX-2.5 preset on ONE GPU, one fv-serve process (${LTX_FAMILY_MEMBERS.join(", ")}): turbo resident, the others loaded on their first request (one 22B DiT at a time); the fal apps route by tier within the process. An 80-96 GB card and ~38 GB of host RAM (the guided A2V model). Its trees (about 165 GB) are on the EU volume.`,
    members: LTX_FAMILY_MEMBERS,
    config: "runpod-family-ltx.toml",
    variant: "ltx",
    min_vram_gb: 80,
    left_out: [{ what: "ltx25-distill-sol-nvfp4-taehv (catalog ltx-draft, no preset)", why: "its NVFP4 linears are a process-wide load setting the other LTX recipes do not share (the engine's ProcessPlan): its own process" }],
  }),
  familyPreset({
    id: "streaming-family",
    title: `Streaming: every causal streaming model on one pod (${STREAMING_FAMILY_MEMBERS.join(", ")}; LongLive weights NON-COMMERCIAL)`,
    description: `Every causal streaming model on ONE GPU, one fv-serve process (${STREAMING_FAMILY_MEMBERS.join(", ")}): SF-Wan resident (the Reactor's model), LongLive loaded on its first request; an SF-Wan pipeline stays resident once loaded, so both end up resident (26-33 GB peak each): an 80 GB-class card. Weights from R2 by default (about 36 GB).`,
    members: STREAMING_FAMILY_MEMBERS,
    config: "runpod-family-streaming.toml",
    variant: "sfwan",
    min_vram_gb: 80,
  }),
];
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
