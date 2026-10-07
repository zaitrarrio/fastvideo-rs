// The weight trees on each weights network volume fv-control may mount
// (under <volume>/weights; a serverless worker sees them at
// /runpod-volume/weights). A static list, not a live listing: fv-control
// cannot read a volume without a pod. The source is the per-volume table of
// docs/ops/runpod-volumes.md (§ "Volume path | US | EU", every row "yes" on
// EU, verified 2026-10-06) and scripts/gpu/weights-manifest.tsv;
// test/unit/serverless-presets.test.ts checks every manifest row is here and
// every preset's trees are. A new tree on the volume (CLAUDE.md: add-only,
// recorded in the manifest) is added here in the same change.
import { REGIONS } from "./cluster/regions";

const EU_TREES = [
  "h3-8step",
  "h3-base",
  "FastH3-4-step-Preview-v1-LoRA",
  "upscaler",
  "h3-to-ltx",
  "h3-ref2va",
  "ltx2",
  "ltx25",
  "ltx25-dev",
  "ltx25-ic-lora-ingredients",
  "ltx23",
  "ltx23-dev",
  "hy15-480-t2v",
  "hy15-480-i2v",
  "hy15-720-t2v",
  "hy15-720-i2v",
  "fastwan21-1.3b",
  "wan21-t2v-1.3b",
  "wan22-ti2v-5b",
  "fastwan22-ti2v-5b",
  "wan21-t2v-14b",
  "wan22-t2v-a14b",
  "sfwan21-1.3b",
  "longlive-1.3b",
  "longlive-1.3b-safetensors",
  "longlive2-5b",
  "longlive2-5b-nvfp4-s4",
  "longlive2-5b-nvfp4-s2",
  "longlive-plug",
  "mmaudio-44k-v2",
  "auxiliary",
  "sana-video-2b-480p",
  "lingbot-video-moe-30b-a3b",
  "cosmos3-super",
  "taeh3",
];

/** Volume id → the weight trees on it (top-level names under weights/). */
export const VOLUME_TREES: Record<string, readonly string[]> = {
  [REGIONS.eu.volume]: EU_TREES,
};
/** Where VOLUME_TREES comes from (shown with a missing-tree error). */
export const VOLUME_TREES_SOURCE = "docs/ops/runpod-volumes.md (per-volume table) and scripts/gpu/weights-manifest.tsv";

/** The trees of `trees` a volume does not have (a tree path's first segment is matched); null when fv-control has no list for the volume. */
export function missingTrees(volume: string, trees: string[]): string[] | null {
  const have = VOLUME_TREES[volume];
  if (!have) return null;
  return trees.filter((t) => !have.includes(t.split("/")[0]!));
}
