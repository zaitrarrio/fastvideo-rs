// GENERATED from deploy/deploy.toml by control/gen-configs.mjs; do not edit.
// The deployment data (plan C1, docs/adr/0011-deploy-section.md): the image variants CI builds, the pool presets
// (each one pool: one image, one worker config, its models, queue and timeouts) and the family presets. Consumers:
// src/presets.ts (POOL_PRESETS, STANDARD_POOLS), the regions of src/enums.ts. No imports: every module may import it.
// Regenerate with `node gen-configs.mjs`; CI runs `--check`.
export interface DeployImage {
  id: string;
  /** Baked as FV_CONFIG (configs/serve/<config>). */
  config: string;
  /** Further worker configs the image carries under /etc/fv. */
  also: string[];
  compute: "gpu" | "cpu";
  note: string | null;
}
export interface DeployModel {
  /** The model id the pool serves (the edge routes it). */
  id: string;
  /** Its variant (families/*.toml) and that variant's family. */
  variant: string;
  family: string;
  /** The name the pool model uses for it (a tier alias, the variant id or a served name). */
  recipe: string;
}
export interface DeployPreset {
  id: string;
  title: string;
  description: string;
  /** One of the default cluster's pools (STANDARD_POOLS). */
  standard: boolean;
  /** The image variant (DeployImage.id). */
  image: string;
  /** The worker config (configs/serve/<config>); `inline`: the image does not carry it, it rides as FV_WORKER_TOML_B64. */
  config: string;
  inline: boolean;
  prequant?: string[];
  licence?: string;
  container_disk_gb?: number;
  models: DeployModel[];
  max_queued: number;
  job_timeout_s: number;
  stale_after_s: number;
}
export interface DeployFamilyPreset {
  id: string;
  /** `{members}` stands for the members, comma separated. */
  title: string;
  description: string;
  image: string;
  config: string;
  inline: boolean;
  /** Hand-chosen card class (GB), at least the members' variants' gpu.min_vram_gib. */
  min_vram_gb: number;
  /** The model presets whose models the one pool serves, in the presets' order. */
  members: string[];
  prequant: string[];
  left_out: { what: string; why: string }[];
}
export const DEPLOY_IMAGES: DeployImage[] = [
  {
    "id": "h3-turbo",
    "config": "runpod.toml",
    "also": [],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "h3-max",
    "config": "runpod-h3-max.toml",
    "also": [],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "ltx",
    "config": "runpod-ltx.toml",
    "also": [],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "wan",
    "config": "runpod-wan.toml",
    "also": [],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "wan5b",
    "config": "runpod-wan5b.toml",
    "also": [
      "runpod-wan14b.toml"
    ],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "sfwan",
    "config": "runpod-sfwan.toml",
    "also": [],
    "compute": "gpu",
    "note": null
  },
  {
    "id": "cpu",
    "config": "runpod-fake.toml",
    "also": [],
    "compute": "cpu",
    "note": "CPU only, fake engine; `gateway` until 2026-10-06"
  }
];
export const DEPLOY_PRESETS: DeployPreset[] = [
  {
    "id": "h3-turbo",
    "title": "H3 turbo (fasth3)",
    "description": "MiniMax H3 turbo tier; also the Reactor's clip model.",
    "standard": true,
    "image": "h3-turbo",
    "config": "runpod.toml",
    "inline": false,
    "prequant": [
      "h3-base/text_encoder_fp8",
      "h3-base/transformer_prequant_4step-vsa_mxfp8"
    ],
    "models": [
      {
        "id": "fasth3",
        "variant": "fasth3-4step-vsa",
        "family": "h3",
        "recipe": "h3-turbo"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "h3-max",
    "title": "H3 max (Sol-H3)",
    "description": "MiniMax H3 max tier (Sol-H3 4-step ladder).",
    "standard": true,
    "image": "h3-max",
    "config": "runpod-h3-max.toml",
    "inline": false,
    "prequant": [
      "h3-base/text_encoder_fp8",
      "h3-base/transformer_prequant_sol-h3_mxfp8"
    ],
    "models": [
      {
        "id": "sol-h3",
        "variant": "sol-h3",
        "family": "h3",
        "recipe": "h3-max"
      }
    ],
    "max_queued": 16,
    "job_timeout_s": 3600,
    "stale_after_s": 180
  },
  {
    "id": "ltx",
    "title": "LTX turbo",
    "description": "LTX-2.5 turbo tier (Sol stage 2): fal lightricks/ltx-2.5 /fast.",
    "standard": true,
    "image": "ltx",
    "config": "runpod-ltx.toml",
    "inline": false,
    "models": [
      {
        "id": "ltx25-distill-sol",
        "variant": "ltx25-distill-sol",
        "family": "ltx2",
        "recipe": "ltx-turbo"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "wan",
    "title": "Wan 2.2 5B (turbo + max)",
    "description": "Wan 2.2 TI2V-5B: FastWan turbo and the 50-step max tier; fal-ai/wan.",
    "standard": true,
    "image": "wan5b",
    "config": "runpod-wan5b.toml",
    "inline": false,
    "models": [
      {
        "id": "fastwan22-ti2v-5b",
        "variant": "fastwan22-ti2v-5b",
        "family": "wan",
        "recipe": "wan-turbo"
      },
      {
        "id": "wan22-ti2v-5b",
        "variant": "wan22-ti2v-5b",
        "family": "wan",
        "recipe": "wan-max"
      }
    ],
    "max_queued": 64,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "ltx-pro",
    "title": "LTX pro (dense)",
    "description": "LTX-2.5 pro tier (ltx25-distill-dense, two-stage dense): fal lightricks/ltx-2.5 /pro and fal-ai/ltx-2.3 retake / extend. Sage attention is on by default on sm_120 (RTX PRO 6000; FASTVIDEO_ATTN_SAGE=0 turns it off).",
    "standard": false,
    "image": "ltx",
    "config": "runpod-ltx-pro.toml",
    "inline": true,
    "models": [
      {
        "id": "ltx25-distill-dense",
        "variant": "ltx25-distill-dense",
        "family": "ltx2",
        "recipe": "ltx-pro"
      }
    ],
    "max_queued": 16,
    "job_timeout_s": 3600,
    "stale_after_s": 180
  },
  {
    "id": "ltx-a2v",
    "title": "LTX guided audio-to-video",
    "description": "The guided A2V companion of ltx-pro (LTX-2.5 dev DiT, multimodal guider): fal lightricks/ltx-2.5 audio-to-video/pro. An 80-96 GB card and ~38 GB of host RAM.",
    "standard": false,
    "image": "ltx",
    "config": "runpod-ltx-a2v.toml",
    "inline": true,
    "container_disk_gb": 60,
    "models": [
      {
        "id": "ltx25-a2v-guided",
        "variant": "ltx25-a2v-guided",
        "family": "ltx2",
        "recipe": "ltx25-a2v-guided"
      }
    ],
    "max_queued": 8,
    "job_timeout_s": 3600,
    "stale_after_s": 180
  },
  {
    "id": "ltx-ref2v",
    "title": "LTX reference-to-video",
    "description": "The Ref2V companion of ltx-pro (Ingredients IC-LoRA): fal fal-ai/ltx-2.3-quality ingredient.",
    "standard": false,
    "image": "ltx",
    "config": "runpod-ltx-ref2v.toml",
    "inline": true,
    "models": [
      {
        "id": "ltx25-ref2v",
        "variant": "ltx25-ref2v",
        "family": "ltx2",
        "recipe": "ltx25-ref2v"
      }
    ],
    "max_queued": 8,
    "job_timeout_s": 3600,
    "stale_after_s": 180
  },
  {
    "id": "h3-ref2v",
    "title": "H3 reference-to-video",
    "description": "H3 Ref2VA, turbo and max (the reference companions of h3-turbo / h3-max).",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-ref2v.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "h3-ref2v-turbo",
        "variant": "h3-ref2v-turbo",
        "family": "h3",
        "recipe": "h3-ref2v-turbo"
      },
      {
        "id": "h3-ref2v-max",
        "variant": "h3-ref2v-max",
        "family": "h3",
        "recipe": "h3-ref2v-max"
      }
    ],
    "max_queued": 16,
    "job_timeout_s": 3600,
    "stale_after_s": 180
  },
  {
    "id": "fasth3-trim",
    "title": "FastH3 Trim 8-step",
    "description": "FastH3 Trim (FastVideo/FastVideo-FastH3-Trim-8-Step): 42 of the 50 H3 blocks, 8 DMD steps, VSA; text-to-video with audio only. Faster and smaller than FastH3 V2, with some quality cost. Served as h3-trim.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-trim.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-trim-8step",
        "variant": "fasth3-trim-8step",
        "family": "h3",
        "recipe": "h3-trim"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-trim-nvfp4",
    "title": "FastH3 Trim 8-step NVFP4",
    "description": "FastH3 Trim with the release's NVFP4 DiT (FastVideo/FastVideo-FastH3-Trim-Comfy): every block's attention and FFN linears W4A4 on NVFP4 tensor cores (RTX PRO 6000 / B200 only, no fallback); text-to-video with audio only. Opt-in; served as h3-trim-nvfp4.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-trim-nvfp4.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-trim-nvfp4",
        "variant": "fasth3-trim-nvfp4",
        "family": "h3",
        "recipe": "h3-trim-nvfp4"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-v2-nvfp4",
    "title": "H3 NVFP4 tier: FastH3 V2 NVFP4",
    "description": "FastH3 V2 NVFP4 (FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4): the release's ModelOpt NVFP4 DiT, 8 DMD steps, sparse FP4 VSA, and its own published NVFP4 VAE decoder (gate pass, 5 GiB less peak; FASTVIDEO_H3_VAE_NVFP4=0 decodes with h3-base's VAE). NVFP4 tensor cores only (RTX PRO 6000 / B200, no fallback); text-to-video with audio. Opt-in; served as h3-v2-nvfp4.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-v2-nvfp4.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-v2-nvfp4",
        "variant": "fasth3-v2-nvfp4",
        "family": "h3",
        "recipe": "h3-v2-nvfp4"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-4step-nvfp4",
    "title": "H3 NVFP4 tier: FastH3 4-step NVFP4",
    "description": "FastH3 4-step VSA DataFree NVFP4 (FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4): the release's ModelOpt NVFP4 DiT, 4 DMD steps (VSA 0.9), sparse FP4 VSA, and its own published NVFP4 VAE decoder (FASTVIDEO_H3_VAE_NVFP4=0: h3-base's). NVFP4 tensor cores only (RTX PRO 6000 / B200, no fallback); text-to-video with audio. Opt-in; served as h3-turbo-nvfp4.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-turbo-nvfp4.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-4step-vsa-nvfp4",
        "variant": "fasth3-4step-vsa-nvfp4",
        "family": "h3",
        "recipe": "h3-turbo-nvfp4"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-v2-nvfp4-preview",
    "title": "H3 NVFP4 tier, fast preview (TAEH3)",
    "description": "FastH3 V2 NVFP4 decoded by TAEH3 instead of the official VAE: a fast preview for drafts and scrubbing, never a default. Fails the quality gate's sharpness floor (0.82-0.91, softer frames of the same picture); about 10-12% faster end to end. NVFP4 tensor cores only; text-to-video with audio. Served as h3-v2-nvfp4-preview.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-v2-nvfp4-preview.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-v2-nvfp4-taeh3",
        "variant": "fasth3-v2-nvfp4-taeh3",
        "family": "h3",
        "recipe": "h3-v2-nvfp4-preview"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-4step-nvfp4-preview",
    "title": "H3 NVFP4 4-step, fast preview (TAEH3)",
    "description": "FastH3 4-step VSA DataFree NVFP4 decoded by TAEH3 instead of the official VAE: a fast preview, never a default. Fails the quality gate's sharpness floor (0.82-0.91); about 10-12% faster end to end. NVFP4 tensor cores only; text-to-video with audio. Served as h3-turbo-nvfp4-preview.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-turbo-nvfp4-preview.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-4step-vsa-nvfp4-taeh3",
        "variant": "fasth3-4step-vsa-nvfp4-taeh3",
        "family": "h3",
        "recipe": "h3-turbo-nvfp4-preview"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "h3-turbo-preview",
    "title": "H3 turbo, fast preview (TAEH3)",
    "description": "The h3-turbo recipe (FastH3 Preview 4-step VSA, MXFP8) decoded by TAEH3 instead of the official VAE, at 768p and 480p: a fast preview, never a default (the 480p draft tier is h3-draft). Fails the quality gate's sharpness floor; about 10-12% faster end to end. Served as h3-turbo-preview.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-turbo-preview.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8",
      "h3-base/transformer_prequant_4step-vsa_mxfp8"
    ],
    "models": [
      {
        "id": "fasth3-4step-vsa-taeh3",
        "variant": "fasth3-4step-vsa-taeh3",
        "family": "h3",
        "recipe": "h3-turbo-preview"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fasth3-trim-nvfp4-fast",
    "title": "FastH3 Trim NVFP4 + FP4 attention (fastest, opt-in)",
    "description": "The fastest H3 configuration measured: FastH3 Trim 8-step with the release's NVFP4 DiT and the VSA fine stage on sparse FP4 tensor cores (9.2 s denoise at 480p, 24.6 s at 768p on an RTX PRO 6000, about 20 GiB peak). A DIFFERENT MODEL from FastH3 V2 and NOT gate-equivalent to any H3 tier: judge it by eye. RTX PRO 6000 (sm_120) only, no fallback; text-to-video with audio only. Opt-in; served as h3-trim-nvfp4-fast.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-trim-nvfp4-fast.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "fasth3-trim-nvfp4-fast",
        "variant": "fasth3-trim-nvfp4-fast",
        "family": "h3",
        "recipe": "h3-trim-nvfp4-fast"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "h3-larry-turbo",
    "title": "MiniMax-H3 Turbo LoRA (larry)",
    "description": "MiniMax-H3 with larryvrh/MiniMax-H3-Turbo-Lora (ckpt850 EMA, Apache-2.0) folded in at load, as the MiniMaxAI/MiniMax-H3-Turbo-Lora Space runs it: 6 forwards (4 allowed), dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-larry-turbo.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "h3-larry-turbo",
        "variant": "h3-larry-turbo",
        "family": "h3",
        "recipe": "h3-larry-turbo"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "h3-lightx2v-turbo",
    "title": "MiniMax-H3 Turbo LoRA (lightx2v)",
    "description": "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v1.2 768p (Apache-2.0) folded in at load: 4 forwards at shifts 6 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-lightx2v-turbo.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "h3-lightx2v-turbo",
        "variant": "h3-lightx2v-turbo",
        "family": "h3",
        "recipe": "h3-lightx2v-turbo"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "vdn-h3",
    "title": "VDN-H3 (Video DeltaNet)",
    "description": "VDN-H3 (OpenVDN/vdn-minimax-h3, stage B): MiniMax-H3 with hybrid attention, a chunk-aligned window softmax plus a frame-wise delta-rule linear branch in every block; 50 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-vdn-h3.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "vdn-h3",
        "variant": "vdn-h3",
        "family": "h3",
        "recipe": "vdn-h3-50step"
      }
    ],
    "max_queued": 16,
    "job_timeout_s": 3600,
    "stale_after_s": 300
  },
  {
    "id": "vdn-h3-8step",
    "title": "VDN-H3 8-step (Video DeltaNet, DMD)",
    "description": "VDN-H3 8-step (OpenVDN/vdn-minimax-h3 stage-dmd-step-250): the stage-B hybrid attention plus the DMD turbo LoRA, 8 forwards, no CFG, bf16 linears; text-to-video with audio (docs/ports/vdn-h3.md). Served as vdn-h3-8step.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-vdn-h3-8step.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "vdn-h3-8step",
        "variant": "vdn-h3-8step",
        "family": "h3",
        "recipe": "vdn-h3-8step"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "h3-lightx2v-turbo-v01",
    "title": "MiniMax-H3 Turbo LoRA (lightx2v v0.1, the Space's)",
    "description": "MiniMax-H3 with lightx2v/Minimax-h3-Turbo FL2VA/T2VA 4-step v0.1 (Apache-2.0), the file the MiniMax-H3-Turbo-Lora Space folds: 4 forwards at shifts 12 / 3, dense attention, MXFP8; text-, image- and first/last-frame-to-video with audio.",
    "standard": false,
    "image": "h3-max",
    "config": "runpod-h3-lightx2v-turbo-v01.toml",
    "inline": true,
    "prequant": [
      "h3-base/text_encoder_fp8"
    ],
    "models": [
      {
        "id": "h3-lightx2v-turbo-v01",
        "variant": "h3-lightx2v-turbo-v01",
        "family": "h3",
        "recipe": "h3-lightx2v-turbo-v01"
      }
    ],
    "max_queued": 32,
    "job_timeout_s": 1800,
    "stale_after_s": 120
  },
  {
    "id": "fastwan21",
    "title": "FastWan 2.1 1.3B",
    "description": "FastWan 2.1 T2V 1.3B (480p, 3 DMD steps): fal fastvideo/fastwan21-1.3b.",
    "standard": false,
    "image": "wan",
    "config": "runpod-wan.toml",
    "inline": false,
    "models": [
      {
        "id": "fastwan21-1.3b",
        "variant": "fastwan21-1.3b",
        "family": "wan",
        "recipe": "fastwan21-1.3b"
      }
    ],
    "max_queued": 64,
    "job_timeout_s": 900,
    "stale_after_s": 90
  },
  {
    "id": "sfwan",
    "title": "SF-Wan causal streaming",
    "description": "SF-Wan 1.3B causal rollout: Reactor and native streams (one session per GPU); the cluster's Reactor model when no pool serves fasth3.",
    "standard": false,
    "image": "sfwan",
    "config": "runpod-sfwan.toml",
    "inline": false,
    "models": [
      {
        "id": "sfwan21-1.3b",
        "variant": "sfwan21-1.3b",
        "family": "wan",
        "recipe": "sfwan21-1.3b"
      }
    ],
    "max_queued": 8,
    "job_timeout_s": 1800,
    "stale_after_s": 60
  },
  {
    "id": "longlive",
    "title": "LongLive-1.3B causal (NON-COMMERCIAL)",
    "description": "LongLive-1.3B on the SF-Wan engine (window 12, sink 3, KV re-cache at prompt switches): Reactor and native streams. NON-COMMERCIAL licence: research and evaluation only.",
    "standard": false,
    "image": "sfwan",
    "config": "runpod-longlive.toml",
    "inline": true,
    "licence": "LongLive-1.3B weights: CC-BY-NC-SA-4.0 (non-commercial; research / evaluation only)",
    "models": [
      {
        "id": "longlive-1.3b",
        "variant": "longlive-1.3b",
        "family": "wan",
        "recipe": "longlive-1.3b"
      }
    ],
    "max_queued": 8,
    "job_timeout_s": 1800,
    "stale_after_s": 60
  }
];
export const DEPLOY_FAMILY_PRESETS: DeployFamilyPreset[] = [
  {
    "id": "h3-family",
    "title": "H3 family: every H3 variant on one pod",
    "description": "Every H3 variant a preset serves on ONE GPU, one fv-serve process ({members}, and the 480p draft tier): turbo resident, the others loaded on their first request (one DiT at a time: a switch unloads the resident one and loads the next from the pod's disk, about a minute). RTX PRO 6000 (96 GB, sm_120) or B200: the NVFP4 releases need NVFP4 tensor cores and VDN-H3 a 96 GB card. Weights from R2 by default (the EU volume lacks the NVFP4, Trim, LoRA and VDN trees): about 396 GB, a 485 GB container disk.",
    "image": "h3-max",
    "config": "runpod-family-h3.toml",
    "inline": true,
    "min_vram_gb": 96,
    "members": [
      "h3-turbo",
      "h3-max",
      "h3-ref2v",
      "fasth3-trim",
      "fasth3-trim-nvfp4",
      "fasth3-v2-nvfp4",
      "fasth3-4step-nvfp4",
      "fasth3-v2-nvfp4-preview",
      "fasth3-4step-nvfp4-preview",
      "h3-turbo-preview",
      "fasth3-trim-nvfp4-fast",
      "h3-larry-turbo",
      "h3-lightx2v-turbo",
      "vdn-h3",
      "vdn-h3-8step",
      "h3-lightx2v-turbo-v01"
    ],
    "prequant": [],
    "left_out": [
      {
        "what": "fasth3-8step-dense (catalog, no preset)",
        "why": "its 148 GB h3-8step tree would take the union past Runpod's 500 GB container disk"
      },
      {
        "what": "the MXFP8 pre-quantized trees (77 GB)",
        "why": "they would take the union (about 396 GB, 485 GB of disk) to 573 GB of disk, past Runpod's 500 GB: turbo and max quantize at load instead"
      }
    ]
  },
  {
    "id": "fasth3-family",
    "title": "FastH3 family: every FastH3 release on one pod",
    "description": "Every FastH3 release a preset serves on ONE GPU, one fv-serve process ({members}, and the 480p draft tier): turbo resident, the previews, Trim and the NVFP4 tier loaded on their first request (one DiT at a time). RTX PRO 6000 (96 GB, sm_120) or B200 (NVFP4 tensor cores). Weights from R2 by default: about 371 GB with turbo's pre-quantized trees, a 457 GB container disk.",
    "image": "h3-max",
    "config": "runpod-family-fasth3.toml",
    "inline": true,
    "min_vram_gb": 96,
    "members": [
      "h3-turbo",
      "fasth3-trim",
      "fasth3-trim-nvfp4",
      "fasth3-v2-nvfp4",
      "fasth3-4step-nvfp4",
      "fasth3-v2-nvfp4-preview",
      "fasth3-4step-nvfp4-preview",
      "h3-turbo-preview",
      "fasth3-trim-nvfp4-fast"
    ],
    "prequant": [
      "h3-base/text_encoder_fp8",
      "h3-base/transformer_prequant_4step-vsa_mxfp8"
    ],
    "left_out": [
      {
        "what": "fasth3-8step-dense (catalog, no preset)",
        "why": "its 148 GB h3-8step tree would take the union past Runpod's 500 GB container disk"
      }
    ]
  },
  {
    "id": "ltx-family",
    "title": "LTX family: turbo, pro, guided A2V and Ref2V on one pod",
    "description": "Every LTX-2.5 preset on ONE GPU, one fv-serve process ({members}): turbo resident, the others loaded on their first request (one 22B DiT at a time); the fal apps route by tier within the process. An 80-96 GB card and ~38 GB of host RAM (the guided A2V model). Its trees (about 165 GB) are on the EU volume.",
    "image": "ltx",
    "config": "runpod-family-ltx.toml",
    "inline": true,
    "min_vram_gb": 80,
    "members": [
      "ltx",
      "ltx-pro",
      "ltx-a2v",
      "ltx-ref2v"
    ],
    "prequant": [],
    "left_out": [
      {
        "what": "ltx25-distill-sol-nvfp4-taehv (catalog ltx-draft, no preset)",
        "why": "its NVFP4 linears are a process-wide load setting the other LTX recipes do not share (the engine's ProcessPlan): its own process"
      }
    ]
  },
  {
    "id": "streaming-family",
    "title": "Streaming: every causal streaming model on one pod ({members}; LongLive weights NON-COMMERCIAL)",
    "description": "Every causal streaming model on ONE GPU, one fv-serve process ({members}): SF-Wan resident (the Reactor's model), LongLive loaded on its first request; an SF-Wan pipeline stays resident once loaded, so both end up resident (26-33 GB peak each): an 80 GB-class card. Weights from R2 by default (about 36 GB).",
    "image": "sfwan",
    "config": "runpod-family-streaming.toml",
    "inline": true,
    "min_vram_gb": 80,
    "members": [
      "sfwan",
      "longlive"
    ],
    "prequant": [],
    "left_out": []
  }
];
/** The edge / dispatch family of each model family. */
export const DISPATCH_FAMILY: Record<string, string> = {
  "h3": "h3",
  "ltx2": "ltx",
  "wan": "wan"
};
