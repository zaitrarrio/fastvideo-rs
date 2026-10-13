// The weight trees on each weights network volume fv-control may mount
// (under <volume>/weights; a serverless worker sees them at
// /runpod-volume/weights). A static list, not a live listing: fv-control
// cannot read a volume without a pod. The source is scripts/gpu/volumes.sh
// FV_EU_VOLUME_TREES (what r2-fill-volume.sh wrote and compared on
// 2026-10-13) and docs/ops/runpod-volumes.md §1;
// test/unit/serverless-presets.test.ts checks this list against volumes.sh.
// A new tree on the volume (CLAUDE.md: add-only) is added here in the same change.
import { REGIONS } from "./cluster/regions";

/** fv-weights-h3-ltx-500 (owner decision 2026-10-13: 500 GB, H3 + LTX only; everything else from R2). Full tree
 * names: a nested tree (h3-base/text_encoder_fp8) is on the volume only when it is listed itself. */
export const EU_TREES = [
  "h3-base",
  "h3-base/text_encoder_fp8",
  "h3-base/transformer_prequant_4step-vsa_mxfp8",
  "h3-base/transformer_prequant_sol-h3_mxfp8",
  "FastH3-4-step-Preview-v1-LoRA",
  "h3-ref2va",
  "ltx25",
  "ltx25-dev",
  "ltx25-ic-lora-ingredients",
  "upscaler",
  "h3-to-ltx",
  "auxiliary/tae",
  "auxiliary/lpips",
] as const;

/** Volume id → the weight trees on it (paths under weights/). */
export const VOLUME_TREES: Record<string, readonly string[]> = {
  [REGIONS.eu.volume]: EU_TREES,
};
/** Where VOLUME_TREES comes from (shown with a missing-tree error). */
export const VOLUME_TREES_SOURCE = "docs/ops/runpod-volumes.md §1 and scripts/gpu/volumes.sh FV_EU_VOLUME_TREES";

/** Whether a tree path is on a volume holding `have`: listed itself, or a parent directory of listed trees
 * (`auxiliary` for auxiliary/tae and auxiliary/lpips). A nested tree is never implied by its parent. */
export function treeOnVolume(have: readonly string[], tree: string): boolean {
  return have.includes(tree) || have.some((h) => h.startsWith(tree + "/"));
}

/** The trees of `trees` a volume does not have (full-path match, treeOnVolume); null when fv-control has no list for the volume. */
export function missingTrees(volume: string, trees: string[]): string[] | null {
  const have = VOLUME_TREES[volume];
  if (!have) return null;
  return trees.filter((t) => !treeOnVolume(have, t));
}
