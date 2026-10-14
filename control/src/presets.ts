// The preset catalog: an image variant plus the worker config it runs, the
// models it serves, the weight trees it loads and the GPU memory it needs.
// One source for cluster pools (cluster/spec.ts re-exports it), standalone
// pods (standalone.ts) and serverless endpoints (serverless/presets.ts).
// Kept free of runtime imports from ./schemas and ./cluster/spec (both import
// this module's users), so there is no module cycle.
import type { PoolSpec } from "./cluster/spec";
import { WORKER_CONFIGS } from "./cluster/worker-configs";

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
}
const inline = (file: string) => {
  const t = WORKER_CONFIGS[file];
  if (!t) throw new Error(`no generated worker config ${file} (node gen-configs.mjs)`);
  return t;
};
// The presets reuse the image variants CI builds (docs/serve/images.md); a
// config the variant's image does not carry rides inline (FV_WORKER_TOML_B64,
// generated from configs/serve by gen-configs.mjs).
export const POOL_PRESETS: PoolPreset[] = [
  { id: "h3-turbo", min_vram_gb: 80, title: "H3 turbo (fasth3)", description: "MiniMax H3 turbo tier; also the Reactor's clip model.", weights: ["h3-base", "FastH3-4-step-Preview-v1-LoRA"], prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"], pool: STANDARD_POOLS[0]! },
  { id: "h3-max", min_vram_gb: 80, title: "H3 max (Sol-H3)", description: "MiniMax H3 max tier (Sol-H3 4-step ladder).", weights: ["h3-base", "FastH3-4-step-Preview-v1-LoRA"], prequant: ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_sol-h3_mxfp8"], pool: STANDARD_POOLS[1]! },
  { id: "ltx", min_vram_gb: 80, title: "LTX turbo", description: "LTX-2.5 turbo tier (Sol stage 2): fal lightricks/ltx-2.5 /fast.", weights: ["ltx25"], pool: STANDARD_POOLS[2]! },
  { id: "wan", min_vram_gb: 32, title: "Wan 2.2 5B (turbo + max)", description: "Wan 2.2 TI2V-5B: FastWan turbo and the 50-step max tier; fal-ai/wan.", weights: ["fastwan22-ti2v-5b", "wan22-ti2v-5b"], pool: STANDARD_POOLS[3]! },
  {
    id: "ltx-pro",
    min_vram_gb: 80,
    title: "LTX pro (dense)",
    description: "LTX-2.5 pro tier (ltx25-distill-dense, two-stage dense): fal lightricks/ltx-2.5 /pro and fal-ai/ltx-2.3 retake / extend. Sage attention is on by default on sm_120 (RTX PRO 6000; FASTVIDEO_ATTN_SAGE=0 turns it off).",
    weights: ["ltx25"],
    pool: { id: "ltx-pro", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-pro.toml"), models: [{ id: "ltx25-distill-dense", family: "ltx2", recipe: "ltx-pro" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-a2v",
    min_vram_gb: 80,
    title: "LTX guided audio-to-video",
    description: "The guided A2V companion of ltx-pro (LTX-2.5 dev DiT, multimodal guider): fal lightricks/ltx-2.5 audio-to-video/pro. An 80-96 GB card and ~38 GB of host RAM.",
    weights: ["ltx25", "ltx25-dev"],
    pool: { id: "ltx-a2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-a2v.toml"), container_disk_gb: 60, models: [{ id: "ltx25-a2v-guided", family: "ltx2", recipe: "ltx25-a2v-guided" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "ltx-ref2v",
    min_vram_gb: 80,
    title: "LTX reference-to-video",
    description: "The Ref2V companion of ltx-pro (Ingredients IC-LoRA): fal fal-ai/ltx-2.3-quality ingredient.",
    weights: ["ltx25", "ltx25-ic-lora-ingredients"],
    pool: { id: "ltx-ref2v", variant: "ltx", count: 1, compute: "GPU", config_toml: inline("runpod-ltx-ref2v.toml"), models: [{ id: "ltx25-ref2v", family: "ltx2", recipe: "ltx25-ref2v" }], max_queued: 8, job_timeout_s: 3600, stale_after_s: 180 },
  },
  {
    id: "h3-ref2v",
    min_vram_gb: 80,
    title: "H3 reference-to-video",
    description: "H3 Ref2VA, turbo and max (the reference companions of h3-turbo / h3-max).",
    weights: ["h3-base", "h3-ref2va"],
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
    min_vram_gb: 80,
    title: "FastH3 Trim 8-step",
    description: "FastH3 Trim (FastVideo/FastVideo-FastH3-Trim-8-Step): 42 of the 50 H3 blocks, 8 DMD steps, VSA; text-to-video with audio only. Faster and smaller than FastH3 V2, with some quality cost. Served as h3-trim.",
    weights: ["fasth3-trim-8step", "h3-base"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-trim", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-trim.toml"), models: [{ id: "fasth3-trim-8step", family: "h3", recipe: "h3-trim" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // FastH3 Trim with the release's NVFP4 DiT (docs/ports/h3-trim.md §5): opt-in, the fasth3-trim preset stays the
  // default. NVFP4 tensor cores only (RTX PRO 6000 / B200): the model fails to load on any other GPU.
  {
    id: "fasth3-trim-nvfp4",
    min_vram_gb: 80,
    title: "FastH3 Trim 8-step NVFP4",
    description: "FastH3 Trim with the release's NVFP4 DiT (FastVideo/FastVideo-FastH3-Trim-Comfy): every block's attention and FFN linears W4A4 on NVFP4 tensor cores (RTX PRO 6000 / B200 only, no fallback); text-to-video with audio only. Opt-in; served as h3-trim-nvfp4.",
    weights: ["fasth3-trim-8step", "fasth3-trim-8step-nvfp4", "h3-base"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "fasth3-trim-nvfp4", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-trim-nvfp4.toml"), models: [{ id: "fasth3-trim-nvfp4", family: "h3", recipe: "h3-trim-nvfp4" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  // MiniMax-H3 Turbo LoRAs (docs/ports/h3-turbo-loras.md): base H3 + one adapter folded at load. The adapters are in R2
  // only (not on the EU volume), so these pools fetch from R2 by default.
  {
    id: "h3-larry-turbo",
    min_vram_gb: 80,
    title: "MiniMax-H3 Turbo LoRA (larry)",
    description: "MiniMax-H3 with larryvrh/MiniMax-H3-Turbo-Lora (ckpt850 EMA, Apache-2.0) folded in at load, as the MiniMaxAI/MiniMax-H3-Turbo-Lora Space runs it: 6 forwards (4 allowed), dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    weights: ["h3-base", "h3-turbo-loras/larry"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-larry-turbo", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-larry-turbo.toml"), models: [{ id: "h3-larry-turbo", family: "h3", recipe: "h3-larry-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "h3-lightx2v-turbo",
    min_vram_gb: 80,
    title: "MiniMax-H3 Turbo LoRA (lightx2v)",
    description: "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v1.2 768p (Apache-2.0) folded in at load: 4 forwards at shifts 6 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    weights: ["h3-base", "h3-turbo-loras/lightx2v"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-lightx2v-turbo", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-lightx2v-turbo.toml"), models: [{ id: "h3-lightx2v-turbo", family: "h3", recipe: "h3-lightx2v-turbo" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "vdn-h3",
    min_vram_gb: 96,
    title: "VDN-H3 (Video DeltaNet)",
    description: "VDN-H3 (OpenVDN/vdn-minimax-h3, stage B): MiniMax-H3 with hybrid attention, a chunk-aligned window softmax plus a frame-wise delta-rule linear branch in every block; 50 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3.",
    weights: ["vdn-h3-stage-b-2000", "h3-base"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "vdn-h3", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-vdn-h3.toml"), models: [{ id: "vdn-h3", family: "h3", recipe: "vdn-h3-50step" }], max_queued: 16, job_timeout_s: 3600, stale_after_s: 300 },
  },
  {
    id: "vdn-h3-8step",
    min_vram_gb: 96,
    title: "VDN-H3 8-step (Video DeltaNet, DMD)",
    description: "VDN-H3 8-step (OpenVDN/vdn-minimax-h3 stage-dmd-step-250): the stage-B hybrid attention plus the DMD turbo LoRA, 8 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3-8step.",
    weights: ["vdn-h3-dmd-250-turbo", "vdn-h3-stage-b-2000", "h3-base"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "vdn-h3-8step", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-vdn-h3-8step.toml"), models: [{ id: "vdn-h3-8step", family: "h3", recipe: "vdn-h3-8step" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "h3-lightx2v-turbo-v01",
    min_vram_gb: 80,
    title: "MiniMax-H3 Turbo LoRA (lightx2v v0.1, the Space's)",
    description: "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v0.1 (Apache-2.0), the file the MiniMax-H3-Turbo-Lora Space folds: 4 forwards at shifts 12 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    weights: ["h3-base", "h3-turbo-loras/lightx2v-v0.1"],
    prequant: ["h3-base/text_encoder_fp8"],
    pool: { id: "h3-lightx2v-turbo-v01", variant: "h3-max", count: 1, compute: "GPU", config_toml: inline("runpod-h3-lightx2v-turbo-v01.toml"), models: [{ id: "h3-lightx2v-turbo-v01", family: "h3", recipe: "lightx2v-turbo-v0.1" }], max_queued: 32, job_timeout_s: 1800, stale_after_s: 120 },
  },
  {
    id: "fastwan21",
    min_vram_gb: 24,
    title: "FastWan 2.1 1.3B",
    description: "FastWan 2.1 T2V 1.3B (480p, 3 DMD steps): fal fastvideo/fastwan21-1.3b.",
    weights: ["fastwan21-1.3b"],
    pool: { id: "fastwan21", variant: "wan", count: 1, compute: "GPU", config: "/etc/fv/runpod-wan.toml", models: [{ id: "fastwan21-1.3b", family: "wan", recipe: "fastwan21-1.3b" }], max_queued: 64, job_timeout_s: 900, stale_after_s: 90 },
  },
  {
    id: "sfwan",
    min_vram_gb: 24,
    title: "SF-Wan causal streaming",
    description: "SF-Wan 1.3B causal rollout: Reactor and native streams (one session per GPU); the cluster's Reactor model when no pool serves fasth3.",
    weights: ["sfwan21-1.3b"],
    pool: { id: "sfwan", variant: "sfwan", count: 1, compute: "GPU", config: "/etc/fv/runpod-sfwan.toml", models: [{ id: "sfwan21-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
  {
    id: "longlive",
    min_vram_gb: 24,
    title: "LongLive-1.3B causal (NON-COMMERCIAL)",
    description: "LongLive-1.3B on the SF-Wan engine (window 12, sink 3, KV re-cache at prompt switches): Reactor and native streams. NON-COMMERCIAL licence: research and evaluation only.",
    weights: ["longlive-1.3b-safetensors", "sfwan21-1.3b"],
    licence: "LongLive-1.3B weights: CC-BY-NC-SA-4.0 (non-commercial; research / evaluation only)",
    pool: { id: "longlive", variant: "sfwan", count: 1, compute: "GPU", config_toml: inline("runpod-longlive.toml"), models: [{ id: "longlive-1.3b", family: "wan", recipe: "sfwan21-1.3b" }], max_queued: 8, job_timeout_s: 1800, stale_after_s: 60 },
  },
];
export const presetPool = (id: string): PoolSpec | undefined => {
  const p = POOL_PRESETS.find((x) => x.id === id || x.pool.id === id)?.pool;
  return p ? structuredClone(p) : undefined;
};
export const presetById = (id: string): PoolPreset | undefined => POOL_PRESETS.find((x) => x.id === id);
