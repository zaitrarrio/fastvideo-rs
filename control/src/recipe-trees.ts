// The weight trees fv-serve reads for a worker-config recipe beyond the `weights = "${FV_WEIGHTS}/<tree>"` its
// config names (docs/ops/r2-bench-2026-10-12.md finding 1: an r2 worker gets only the trees its preset lists, so a
// tree the engine finds by itself beside `weights` must be listed too). Kept free of imports (presets.ts and the
// tests read it). test/unit/preset-trees.test.ts checks every preset's lists against its config through this table,
// and this table against the Rust sources named below, so a new adapter or tree layout cannot be missed silently.
//
// required: the load fails without it. prequant: a derived pre-quantized form that only speeds up the load (the
// loader quantizes at load without it, "quantizing at load (no pre-quantized form)"): fetched with weights_source r2,
// on the EU volume anyway, never fetched from the Hub (no Hub source).

export interface RecipeTrees {
  /** The catalog entry the recipe (a tier alias or catalog id) resolves to (crates/fastvideo-engine-service/src/cuda/caps.rs). */
  catalog: string;
  required: string[];
  prequant: string[];
  /** Why, with the code that reads each tree. */
  note: string;
}

/** H3's resident FP8 text encoder (the default on 80-96 GB cards) reads `<weights>/text_encoder_fp8/` when valid
 * (crates/fastvideo-cudarc/src/h3/text.rs open_resident_encoder, llm/prequant.rs DIR_NAME). */
const H3_TEXT_FP8 = "h3-base/text_encoder_fp8";
/** The FastH3 4-step Preview v1 adapters (vsa-datafree, dense-datafree), looked for beside the base
 * (crates/fastvideo-models/src/h3/lora.rs FastH3PreviewVariant::adapter_spec, SolH3AdapterSpec::t2v_i2v). */
export const FASTH3_LORA = "FastH3-4-step-Preview-v1-LoRA";

export const RECIPE_TREES: Record<string, RecipeTrees> = {
  "h3-turbo": {
    catalog: "fasth3-4step-vsa",
    required: [FASTH3_LORA],
    // MXFP8 on sm_100+ only: below that QuantMode::for_device runs W8A8, whose tree does not exist, and the boot skips
    // the *_mxfp8 trees (FV_R2_SKIP_TREES).
    prequant: [H3_TEXT_FP8, "h3-base/transformer_prequant_4step-vsa_mxfp8"],
    note: "recipe 4step-vsa: the vsa-datafree adapter (lora.rs); the DiT tree transformer_prequant_<recipe>_<quant> (cudarc h3/dit_tree.rs dir_name)",
  },
  "h3-max": {
    catalog: "sol-h3",
    required: [FASTH3_LORA],
    prequant: [H3_TEXT_FP8, "h3-base/transformer_prequant_sol-h3_mxfp8"],
    note: "recipe sol-h3: the dense-datafree adapter (SolH3AdapterSpec::t2v_i2v, under the base or its parent)",
  },
  "h3-ref2v-turbo": {
    catalog: "sol-h3-ref2va",
    required: ["h3-ref2va"],
    prequant: [H3_TEXT_FP8],
    note: "Ref2VA: transformer_ref and the lightx2v turbo adapter in h3-ref2va (caps.rs ref_weights; SolH3AdapterSpec::ref2va); no DiT tree (sol_h3_forces_ref2va)",
  },
  "h3-ref2v-max": {
    catalog: "h3-ref2va-base-49step",
    required: ["h3-ref2va"],
    prequant: [H3_TEXT_FP8],
    note: "Ref2VA base: transformer_ref in h3-ref2va (caps.rs ref_weights)",
  },
  "vdn-h3-8step": {
    catalog: "vdn-h3-8step",
    required: ["vdn-h3-dmd-250-turbo", "vdn-h3-stage-b-2000"],
    prequant: [H3_TEXT_FP8],
    note: "VDN-H3 8-step: h3-base is the config's weights; the turbo LoRA in vdn-h3-dmd-250-turbo/stage-dmd-step-250, the branch and default LoRA from vdn-h3-stage-b-2000/stage-b-step-2000 (caps.rs VDN_H3_8STEP_TREE, H3Recipe vdn_weights + vdn_shared)",
  },
  "vdn-h3-50step": {
    catalog: "vdn-h3",
    required: ["vdn-h3-stage-b-2000"],
    prequant: [H3_TEXT_FP8],
    note: "VDN-H3: h3-base is the config's weights; the linear branch and LoRA in vdn-h3-stage-b-2000/stage-b-step-2000 (caps.rs VDN_H3_TREE / VDN_H3_STAGE, H3Recipe vdn_weights; cudarc h3/pipeline.rs vdn_checkpoint)",
  },
  "h3-larry-turbo": {
    catalog: "h3-larry-turbo",
    required: ["h3-turbo-loras/larry"],
    prequant: [H3_TEXT_FP8],
    note: "Turbo LoRA larry: the adapter in h3-turbo-loras/larry beside h3-base (crates/fastvideo-models/src/h3/turbo_lora.rs TurboLora::resolve); no DiT tree (merged at load)",
  },
  "h3-lightx2v-turbo": {
    catalog: "h3-lightx2v-turbo",
    required: ["h3-turbo-loras/lightx2v"],
    prequant: [H3_TEXT_FP8],
    note: "Turbo LoRA lightx2v v1.2 768p: the adapter in h3-turbo-loras/lightx2v beside h3-base (turbo_lora.rs TurboLora::resolve); no DiT tree (merged at load)",
  },
  "lightx2v-turbo-v0.1": {
    catalog: "h3-lightx2v-turbo",
    required: ["h3-turbo-loras/lightx2v-v0.1"],
    prequant: [H3_TEXT_FP8],
    note: "Turbo LoRA lightx2v v0.1 (the Space's file): the catalog's lightx2v model with recipe lightx2v-turbo-v0.1 (caps.rs model_from_config); the adapter in h3-turbo-loras/lightx2v-v0.1 beside h3-base (turbo_lora.rs TurboLora::resolve)",
  },
  "h3-trim": {
    catalog: "fasth3-trim-8step",
    required: ["h3-base"],
    prequant: [H3_TEXT_FP8],
    note: "FastH3 Trim: the DiT and VAE in fasth3-trim-8step (the config's weights); the text encoder, tokenizer and audio VAE from h3-base (caps.rs h3() text_weights; cudarc h3/pipeline.rs audio_vae_dir)",
  },
  "h3-trim-nvfp4": {
    catalog: "fasth3-trim-nvfp4",
    required: ["h3-base", "fasth3-trim-8step-nvfp4"],
    prequant: [H3_TEXT_FP8],
    note: "FastH3 Trim NVFP4: as h3-trim, plus the published NVFP4 DiT in fasth3-trim-8step-nvfp4 beside the config's weights (cudarc h3/nvfp4.rs sibling_tree; h3/pipeline.rs nvfp4_dit_file)",
  },
  "h3-trim-nvfp4-fast": {
    catalog: "fasth3-trim-nvfp4-fast",
    required: ["h3-base", "fasth3-trim-8step-nvfp4"],
    prequant: [H3_TEXT_FP8],
    note: "FastH3 Trim NVFP4 + sparse FP4 VSA (the opt-in fastest tier, recipe trim-8step-nvfp4-fp4attn): the trees of h3-trim-nvfp4 (cudarc h3/nvfp4.rs sibling_tree; h3/pipeline.rs vsa_fp4_choice turns the FP4 attention on for the recipe)",
  },
  // The H3 NVFP4 tier (owner decision 2026-10-14): each release's own tree (ModelOpt transformer/, the published NVFP4
  // vae/ shards 1-2) is the config's weights; h3-base gives the text encoder, tokenizer, audio VAE and the vae/'s
  // byte-identical third shard (cudarc h3/pipeline.rs resolve_video_vae / completed_vae_files, caps.rs h3() text_weights).
  "h3-v2-nvfp4": {
    catalog: "fasth3-v2-nvfp4",
    required: ["h3-base"],
    prequant: [H3_TEXT_FP8],
    note: "FastH3 V2 NVFP4: the DiT and the NVFP4 VAE in fasth3-v2-nvfp4 (the config's weights); text, tokenizer, audio VAE and vae/ shard 3 from h3-base (h3/pipeline.rs resolve_video_vae)",
  },
  "h3-turbo-nvfp4": {
    catalog: "fasth3-4step-vsa-nvfp4",
    required: ["h3-base"],
    prequant: [H3_TEXT_FP8],
    note: "FastH3 4-step VSA DataFree NVFP4: as h3-v2-nvfp4, tree fasth3-4step-vsa-nvfp4",
  },
  // "Fast preview" (owner decision 2026-10-14): the recipe decoded by TAEH3 (auxiliary/tae, an aux file every boot
  // fetches), its own model id; never a default.
  "h3-v2-nvfp4-preview": {
    catalog: "fasth3-v2-nvfp4-taeh3",
    required: ["h3-base"],
    prequant: [H3_TEXT_FP8],
    note: "h3-v2-nvfp4 with the TAEH3 decoder (aux files); the load still resolves the tree's vae/ (shard 3 from h3-base), so h3-base is required as for h3-v2-nvfp4",
  },
  "h3-turbo-nvfp4-preview": {
    catalog: "fasth3-4step-vsa-nvfp4-taeh3",
    required: ["h3-base"],
    prequant: [H3_TEXT_FP8],
    note: "h3-turbo-nvfp4 with the TAEH3 decoder (aux files)",
  },
  "h3-turbo-preview": {
    catalog: "fasth3-4step-vsa-taeh3",
    required: [FASTH3_LORA],
    prequant: [H3_TEXT_FP8, "h3-base/transformer_prequant_4step-vsa_mxfp8"],
    note: "h3-turbo with the TAEH3 decoder (aux files): the same adapter and DiT tree as h3-turbo (dit_tree.rs keys the tree by recipe, not decoder)",
  },
  // LTX-2.5: its FP8 Gemma tree (ltx25/text_encoder_fp8) is read only with FASTVIDEO_LTX2_TEXT_FP8=1 (opt-in,
  // cudarc ltx2/pipeline.rs), which no preset sets.
  "ltx-turbo": { catalog: "ltx25-distill-sol", required: [], prequant: [], note: "everything under ltx25" },
  "ltx-pro": { catalog: "ltx25-distill-dense", required: [], prequant: [], note: "everything under ltx25" },
  "ltx25-a2v-guided": { catalog: "ltx25-a2v-guided", required: ["ltx25-dev"], prequant: [], note: "the dev DiT ltx25-dev/transformer_full (caps.rs LTX25_DEV_DIT)" },
  "ltx25-ref2v": { catalog: "ltx25-ref2v", required: ["ltx25-ic-lora-ingredients"], prequant: [], note: "the Ingredients IC-LoRA (fastvideo-models ltx2/lora.rs LTX25_INGREDIENTS_DIR)" },
  "wan-turbo": { catalog: "fastwan22-ti2v-5b", required: [], prequant: [], note: "fastwan22-ti2v-5b; the TAE from auxiliary/tae (aux files)" },
  "wan-max": { catalog: "wan22-ti2v-5b", required: [], prequant: [], note: "wan22-ti2v-5b" },
  "fastwan21-1.3b": { catalog: "fastwan21-1.3b", required: [], prequant: [], note: "fastwan21-1.3b" },
  "sfwan21-1.3b": { catalog: "sfwan21-1.3b", required: [], prequant: [], note: "sfwan21-1.3b (LongLive: the config's longlive path)" },
};

/** The MXFP8 pre-quantized trees: of use on sm_100+ only (the boot skips them below that). */
export const MXFP8_TREE_GLOB = "*_mxfp8";
