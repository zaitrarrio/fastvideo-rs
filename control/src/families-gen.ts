// GENERATED from families/*.toml by control/gen-configs.mjs; do not edit.
// The runtime family manifests (layer 2, docs/adr/0008-family-manifest-layer-2.md, families/README.md): every served
// model family and variant, the names a serve config may use for a variant, the weight trees it reads and its GPU
// needs. Helpers: src/families.ts. Regenerate with `node gen-configs.mjs`; CI runs `--check`.
export type Tier = "max" | "turbo" | "draft";
export interface ManifestFamily {
  id: string;
  display: string;
  licence: string;
  port_doc: string;
  /** Variant ids in file order (the engine catalog's order: a tier alias resolves to the first of its tier). */
  variants: string[];
}
export interface ManifestVariant {
  family: string;
  tier: Tier | null;
  /** Names besides the id the variant answers to. */
  served_names: string[];
  resident_default: boolean;
  /** The short recipe name the capabilities report. */
  recipe_name: string;
  /** Every weight tree the variant reads, as scripts/gpu/weights-manifest.tsv names them (sorted). */
  trees: string[];
  /** `trees` without the auxiliary/ files (TAE decoders, LPIPS), which every boot fetches as the aux trees:
   * what a preset lists as its weights and a weights plan fetches. */
  weight_trees: string[];
  gpu: { min_vram_gib: number; min_sm: number; needs: string[] };
}
export const MANIFEST_FAMILIES: ManifestFamily[] = [
  {
    "id": "h3",
    "display": "MiniMax-H3 / FastH3",
    "licence": "MiniMax-H3 Community License",
    "port_doc": "docs/ports/h3.md",
    "variants": [
      "sol-h3",
      "fasth3-4step-vsa",
      "fasth3-4step-vsa-480p-taeh3",
      "fasth3-8step-dense",
      "fasth3-trim-8step",
      "fasth3-trim-nvfp4",
      "fasth3-trim-nvfp4-fast",
      "fasth3-v2-nvfp4",
      "fasth3-4step-vsa-nvfp4",
      "h3-larry-turbo",
      "h3-lightx2v-turbo",
      "vdn-h3",
      "vdn-h3-8step",
      "h3-ref2v-max",
      "h3-ref2v-turbo",
      "fasth3-v2-nvfp4-taeh3",
      "fasth3-4step-vsa-nvfp4-taeh3",
      "fasth3-4step-vsa-taeh3"
    ]
  },
  {
    "id": "ltx2",
    "display": "LTX-2 / LTX-2.3 / LTX-2.5",
    "licence": "LTX-2 Community License",
    "port_doc": "docs/ports/ltx2.md",
    "variants": [
      "ltx25-distill-dense",
      "ltx25-distill-sol",
      "ltx25-distill-sol-nvfp4-taehv",
      "ltx25-ref2v",
      "ltx25-a2v-guided"
    ]
  },
  {
    "id": "wan",
    "display": "Wan 2.1 / Wan 2.2 / FastWan",
    "licence": "Apache-2.0",
    "port_doc": "docs/ports/wan.md",
    "variants": [
      "wan22-ti2v-5b",
      "fastwan22-ti2v-5b",
      "fastwan22-ti2v-5b-taehv",
      "wan14b-plug-4step",
      "fastwan21-1.3b",
      "fastwan21-1.3b-taehv",
      "sfwan21-1.3b"
    ]
  }
];
export const MANIFEST_VARIANTS: Record<string, ManifestVariant> = {
  "sol-h3": {
    "family": "h3",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "sol-h3-4step-engine-ladder",
    "trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "weight_trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "fasth3-4step-vsa": {
    "family": "h3",
    "tier": "turbo",
    "served_names": [
      "fasth3"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-4step-vsa",
    "trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "weight_trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "fasth3-4step-vsa-480p-taeh3": {
    "family": "h3",
    "tier": "draft",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "fasth3-4step-vsa-480p-taeh3",
    "trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "auxiliary/tae/taeh3.safetensors",
      "h3-base"
    ],
    "weight_trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "fasth3-8step-dense": {
    "family": "h3",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "fasth3-8step-dense",
    "trees": [
      "h3-8step",
      "h3-base"
    ],
    "weight_trees": [
      "h3-8step",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "fasth3-trim-8step": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-trim"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-trim-8step-vsa",
    "trees": [
      "fasth3-trim-8step",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-trim-8step",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "fasth3-trim-nvfp4": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-trim-nvfp4"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-trim-8step-vsa-nvfp4",
    "trees": [
      "fasth3-trim-8step",
      "fasth3-trim-8step-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-trim-8step",
      "fasth3-trim-8step-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "fasth3-trim-nvfp4-fast": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-trim-nvfp4-fast"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-trim-8step-vsa-fp4-nvfp4",
    "trees": [
      "fasth3-trim-8step",
      "fasth3-trim-8step-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-trim-8step",
      "fasth3-trim-8step-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "fasth3-v2-nvfp4": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-v2-nvfp4"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-v2-8step-vsa-nvfp4",
    "trees": [
      "fasth3-v2-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-v2-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "fasth3-4step-vsa-nvfp4": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-turbo-nvfp4"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-4step-vsa-nvfp4",
    "trees": [
      "fasth3-4step-vsa-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-4step-vsa-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "h3-larry-turbo": {
    "family": "h3",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "h3-larry-turbo-ckpt850-dense",
    "trees": [
      "h3-base",
      "h3-turbo-loras/larry"
    ],
    "weight_trees": [
      "h3-base",
      "h3-turbo-loras/larry"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "h3-lightx2v-turbo": {
    "family": "h3",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "h3-lightx2v-turbo-v1.2-768p-dense",
    "trees": [
      "h3-base",
      "h3-turbo-loras/lightx2v"
    ],
    "weight_trees": [
      "h3-base",
      "h3-turbo-loras/lightx2v"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "vdn-h3": {
    "family": "h3",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "vdn-h3-50step",
    "trees": [
      "h3-base",
      "vdn-h3-stage-b-2000"
    ],
    "weight_trees": [
      "h3-base",
      "vdn-h3-stage-b-2000"
    ],
    "gpu": {
      "min_vram_gib": 96,
      "min_sm": 80,
      "needs": []
    }
  },
  "vdn-h3-8step": {
    "family": "h3",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "vdn-h3-8step",
    "trees": [
      "h3-base",
      "vdn-h3-dmd-250-turbo",
      "vdn-h3-stage-b-2000"
    ],
    "weight_trees": [
      "h3-base",
      "vdn-h3-dmd-250-turbo",
      "vdn-h3-stage-b-2000"
    ],
    "gpu": {
      "min_vram_gib": 96,
      "min_sm": 80,
      "needs": []
    }
  },
  "h3-ref2v-max": {
    "family": "h3",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "h3-ref2va-base-49step",
    "trees": [
      "h3-base",
      "h3-ref2va"
    ],
    "weight_trees": [
      "h3-base",
      "h3-ref2va"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "h3-ref2v-turbo": {
    "family": "h3",
    "tier": "turbo",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "sol-h3-ref2va",
    "trees": [
      "h3-base",
      "h3-ref2va"
    ],
    "weight_trees": [
      "h3-base",
      "h3-ref2va"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "fasth3-v2-nvfp4-taeh3": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-v2-nvfp4-preview"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-v2-nvfp4-taeh3",
    "trees": [
      "auxiliary/tae/taeh3.safetensors",
      "fasth3-v2-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-v2-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "fasth3-4step-vsa-nvfp4-taeh3": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-turbo-nvfp4-preview"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-4step-vsa-nvfp4-taeh3",
    "trees": [
      "auxiliary/tae/taeh3.safetensors",
      "fasth3-4step-vsa-nvfp4",
      "h3-base"
    ],
    "weight_trees": [
      "fasth3-4step-vsa-nvfp4",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "fasth3-4step-vsa-taeh3": {
    "family": "h3",
    "tier": null,
    "served_names": [
      "h3-turbo-preview"
    ],
    "resident_default": true,
    "recipe_name": "fasth3-4step-vsa-taeh3",
    "trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "auxiliary/tae/taeh3.safetensors",
      "h3-base"
    ],
    "weight_trees": [
      "FastH3-4-step-Preview-v1-LoRA",
      "h3-base"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 89,
      "needs": [
        "fp8"
      ]
    }
  },
  "ltx25-distill-dense": {
    "family": "ltx2",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "ltx25-distill-two-stage-dense",
    "trees": [
      "ltx25"
    ],
    "weight_trees": [
      "ltx25"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "ltx25-distill-sol": {
    "family": "ltx2",
    "tier": "turbo",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "ltx25-distill-two-stage-sol",
    "trees": [
      "ltx25"
    ],
    "weight_trees": [
      "ltx25"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "ltx25-distill-sol-nvfp4-taehv": {
    "family": "ltx2",
    "tier": "draft",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "ltx25-distill-two-stage-sol-nvfp4-taehv",
    "trees": [
      "auxiliary/tae/taeltx2_3_wide.safetensors",
      "ltx25"
    ],
    "weight_trees": [
      "ltx25"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 100,
      "needs": [
        "fp8",
        "nvfp4"
      ]
    }
  },
  "ltx25-ref2v": {
    "family": "ltx2",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "ltx25-ic-lora-ingredients-dense",
    "trees": [
      "ltx25",
      "ltx25-ic-lora-ingredients"
    ],
    "weight_trees": [
      "ltx25",
      "ltx25-ic-lora-ingredients"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "ltx25-a2v-guided": {
    "family": "ltx2",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "ltx25-dev-a2v-guided",
    "trees": [
      "ltx25",
      "ltx25-dev"
    ],
    "weight_trees": [
      "ltx25",
      "ltx25-dev"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "wan22-ti2v-5b": {
    "family": "wan",
    "tier": "max",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "wan22-ti2v-5b-unipc50",
    "trees": [
      "wan22-ti2v-5b"
    ],
    "weight_trees": [
      "wan22-ti2v-5b"
    ],
    "gpu": {
      "min_vram_gib": 32,
      "min_sm": 80,
      "needs": []
    }
  },
  "fastwan22-ti2v-5b": {
    "family": "wan",
    "tier": "turbo",
    "served_names": [
      "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    ],
    "resident_default": true,
    "recipe_name": "fastwan22-ti2v-5b-dmd3",
    "trees": [
      "fastwan22-ti2v-5b"
    ],
    "weight_trees": [
      "fastwan22-ti2v-5b"
    ],
    "gpu": {
      "min_vram_gib": 32,
      "min_sm": 80,
      "needs": []
    }
  },
  "fastwan22-ti2v-5b-taehv": {
    "family": "wan",
    "tier": "draft",
    "served_names": [],
    "resident_default": true,
    "recipe_name": "fastwan22-ti2v-5b-dmd3-taehv",
    "trees": [
      "auxiliary/tae/taew2_2.safetensors",
      "fastwan22-ti2v-5b"
    ],
    "weight_trees": [
      "fastwan22-ti2v-5b"
    ],
    "gpu": {
      "min_vram_gib": 32,
      "min_sm": 80,
      "needs": []
    }
  },
  "wan14b-plug-4step": {
    "family": "wan",
    "tier": null,
    "served_names": [
      "wan14b-turbo"
    ],
    "resident_default": true,
    "recipe_name": "wan14b-plug-4step",
    "trees": [
      "longlive-plug/wan21-t2v-14b-cfg",
      "longlive-plug/wan21-t2v-14b-few-step",
      "wan21-t2v-14b"
    ],
    "weight_trees": [
      "longlive-plug/wan21-t2v-14b-cfg",
      "longlive-plug/wan21-t2v-14b-few-step",
      "wan21-t2v-14b"
    ],
    "gpu": {
      "min_vram_gib": 80,
      "min_sm": 80,
      "needs": []
    }
  },
  "fastwan21-1.3b": {
    "family": "wan",
    "tier": null,
    "served_names": [
      "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    ],
    "resident_default": true,
    "recipe_name": "fastwan21-1.3b-dmd3-vsa",
    "trees": [
      "fastwan21-1.3b"
    ],
    "weight_trees": [
      "fastwan21-1.3b"
    ],
    "gpu": {
      "min_vram_gib": 24,
      "min_sm": 80,
      "needs": []
    }
  },
  "fastwan21-1.3b-taehv": {
    "family": "wan",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "fastwan21-1.3b-dmd3-vsa-taehv",
    "trees": [
      "auxiliary/tae/taew2_1.safetensors",
      "fastwan21-1.3b"
    ],
    "weight_trees": [
      "fastwan21-1.3b"
    ],
    "gpu": {
      "min_vram_gib": 24,
      "min_sm": 80,
      "needs": []
    }
  },
  "sfwan21-1.3b": {
    "family": "wan",
    "tier": null,
    "served_names": [],
    "resident_default": true,
    "recipe_name": "sfwan21-1.3b-causal",
    "trees": [
      "auxiliary/tae/taew2_1.safetensors",
      "sfwan21-1.3b"
    ],
    "weight_trees": [
      "sfwan21-1.3b"
    ],
    "gpu": {
      "min_vram_gib": 24,
      "min_sm": 80,
      "needs": []
    }
  }
};
/** Every name a serve config's `recipe` (or `id`) may use for a variant, as the engine resolves it (caps.rs `find`,
 * `parse_tier_alias`): a tier alias (`h3-turbo`: the first variant of its tier), the variant id or a served name. A
 * variant's `recipe_name` is what the capabilities report, not a config name, and is not here. */
export const MODEL_NAMES: Record<string, { variant: string; kind: "tier" | "id" | "served" }> = {
  "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": {
    "variant": "fastwan21-1.3b",
    "kind": "served"
  },
  "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers": {
    "variant": "fastwan22-ti2v-5b",
    "kind": "served"
  },
  "fasth3": {
    "variant": "fasth3-4step-vsa",
    "kind": "served"
  },
  "fasth3-4step-vsa": {
    "variant": "fasth3-4step-vsa",
    "kind": "id"
  },
  "fasth3-4step-vsa-480p-taeh3": {
    "variant": "fasth3-4step-vsa-480p-taeh3",
    "kind": "id"
  },
  "fasth3-4step-vsa-nvfp4": {
    "variant": "fasth3-4step-vsa-nvfp4",
    "kind": "id"
  },
  "fasth3-4step-vsa-nvfp4-taeh3": {
    "variant": "fasth3-4step-vsa-nvfp4-taeh3",
    "kind": "id"
  },
  "fasth3-4step-vsa-taeh3": {
    "variant": "fasth3-4step-vsa-taeh3",
    "kind": "id"
  },
  "fasth3-8step-dense": {
    "variant": "fasth3-8step-dense",
    "kind": "id"
  },
  "fasth3-trim-8step": {
    "variant": "fasth3-trim-8step",
    "kind": "id"
  },
  "fasth3-trim-nvfp4": {
    "variant": "fasth3-trim-nvfp4",
    "kind": "id"
  },
  "fasth3-trim-nvfp4-fast": {
    "variant": "fasth3-trim-nvfp4-fast",
    "kind": "id"
  },
  "fasth3-v2-nvfp4": {
    "variant": "fasth3-v2-nvfp4",
    "kind": "id"
  },
  "fasth3-v2-nvfp4-taeh3": {
    "variant": "fasth3-v2-nvfp4-taeh3",
    "kind": "id"
  },
  "fastwan21-1.3b": {
    "variant": "fastwan21-1.3b",
    "kind": "id"
  },
  "fastwan21-1.3b-taehv": {
    "variant": "fastwan21-1.3b-taehv",
    "kind": "id"
  },
  "fastwan22-ti2v-5b": {
    "variant": "fastwan22-ti2v-5b",
    "kind": "id"
  },
  "fastwan22-ti2v-5b-taehv": {
    "variant": "fastwan22-ti2v-5b-taehv",
    "kind": "id"
  },
  "h3-draft": {
    "variant": "fasth3-4step-vsa-480p-taeh3",
    "kind": "tier"
  },
  "h3-larry-turbo": {
    "variant": "h3-larry-turbo",
    "kind": "id"
  },
  "h3-lightx2v-turbo": {
    "variant": "h3-lightx2v-turbo",
    "kind": "id"
  },
  "h3-max": {
    "variant": "sol-h3",
    "kind": "tier"
  },
  "h3-ref2v-max": {
    "variant": "h3-ref2v-max",
    "kind": "id"
  },
  "h3-ref2v-turbo": {
    "variant": "h3-ref2v-turbo",
    "kind": "id"
  },
  "h3-trim": {
    "variant": "fasth3-trim-8step",
    "kind": "served"
  },
  "h3-trim-nvfp4": {
    "variant": "fasth3-trim-nvfp4",
    "kind": "served"
  },
  "h3-trim-nvfp4-fast": {
    "variant": "fasth3-trim-nvfp4-fast",
    "kind": "served"
  },
  "h3-turbo": {
    "variant": "fasth3-4step-vsa",
    "kind": "tier"
  },
  "h3-turbo-nvfp4": {
    "variant": "fasth3-4step-vsa-nvfp4",
    "kind": "served"
  },
  "h3-turbo-nvfp4-preview": {
    "variant": "fasth3-4step-vsa-nvfp4-taeh3",
    "kind": "served"
  },
  "h3-turbo-preview": {
    "variant": "fasth3-4step-vsa-taeh3",
    "kind": "served"
  },
  "h3-v2-nvfp4": {
    "variant": "fasth3-v2-nvfp4",
    "kind": "served"
  },
  "h3-v2-nvfp4-preview": {
    "variant": "fasth3-v2-nvfp4-taeh3",
    "kind": "served"
  },
  "ltx-draft": {
    "variant": "ltx25-distill-sol-nvfp4-taehv",
    "kind": "tier"
  },
  "ltx-pro": {
    "variant": "ltx25-distill-dense",
    "kind": "tier"
  },
  "ltx-turbo": {
    "variant": "ltx25-distill-sol",
    "kind": "tier"
  },
  "ltx25-a2v-guided": {
    "variant": "ltx25-a2v-guided",
    "kind": "id"
  },
  "ltx25-distill-dense": {
    "variant": "ltx25-distill-dense",
    "kind": "id"
  },
  "ltx25-distill-sol": {
    "variant": "ltx25-distill-sol",
    "kind": "id"
  },
  "ltx25-distill-sol-nvfp4-taehv": {
    "variant": "ltx25-distill-sol-nvfp4-taehv",
    "kind": "id"
  },
  "ltx25-ref2v": {
    "variant": "ltx25-ref2v",
    "kind": "id"
  },
  "sfwan21-1.3b": {
    "variant": "sfwan21-1.3b",
    "kind": "id"
  },
  "sol-h3": {
    "variant": "sol-h3",
    "kind": "id"
  },
  "vdn-h3": {
    "variant": "vdn-h3",
    "kind": "id"
  },
  "vdn-h3-8step": {
    "variant": "vdn-h3-8step",
    "kind": "id"
  },
  "wan-draft": {
    "variant": "fastwan22-ti2v-5b-taehv",
    "kind": "tier"
  },
  "wan-max": {
    "variant": "wan22-ti2v-5b",
    "kind": "tier"
  },
  "wan-turbo": {
    "variant": "fastwan22-ti2v-5b",
    "kind": "tier"
  },
  "wan14b-plug-4step": {
    "variant": "wan14b-plug-4step",
    "kind": "id"
  },
  "wan14b-turbo": {
    "variant": "wan14b-plug-4step",
    "kind": "served"
  },
  "wan22-ti2v-5b": {
    "variant": "wan22-ti2v-5b",
    "kind": "id"
  }
};
/** Recipe aliases the engine resolves besides MODEL_NAMES (caps.rs `model_from_config` arms a config here uses);
 * `variantOf` does not consult them, `recipeVariant` does. The engine also accepts contract names neither table
 * lists (`4step-vsa`, `8step`, `sol-h3` / `sol_h3`, `trim-8step`, an empty recipe, family `ltx`): plan A3 settles
 * which the manifests carry. */
export const ENGINE_RECIPE_ALIASES: Record<string, string> = {
  "vdn-h3-50step": "vdn-h3"
};
