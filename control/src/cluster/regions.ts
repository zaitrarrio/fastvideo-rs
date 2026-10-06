// Regions: a data centre plus the weights network volume mounted there.
// Kept free of imports from ../schemas (which imports AVAILABLE_REGIONS) so
// the module graph has no cycle.
import { HttpError } from "../util";
import type { ClusterSpec } from "./spec";

export type RegionId = "eu" | "us";
export interface RegionDef {
  volume: string;
  dc: string;
  gpus: string[];
}
/**
 * The US weights volume (fv-weights-b200-us, US-CA-2). Runpod deleted it
 * (s2k01690bi) on about 2026-10-05 while the balance was negative; the owner
 * chose EU only for now (2026-10-06, CLAUDE.md). Empty = `us` is unavailable.
 * When the owner rebuilds US (docs/ops/runpod-volumes.md §5,
 * scripts/gpu/rebuild-volume.sh us), put the new volume id here: that one
 * change brings the region back.
 */
export const US_VOLUME_ID = "";
/** CLAUDE.md: the weights network volumes. A region with an empty volume id is unavailable. */
export const REGIONS: Record<RegionId, RegionDef> = {
  eu: { volume: "jg48s6o1w0", dc: "EUR-IS-1", gpus: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] },
  us: { volume: US_VOLUME_ID, dc: "US-CA-2", gpus: ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200"] },
};
export const REGION_UNAVAILABLE: Partial<Record<RegionId, string>> = {
  us: "US weights volume deleted 2026-10; EU only, see docs/ops/runpod-volumes.md",
};
/** A region is usable when it is known and has a weights volume. */
export function regionAvailable(r: string): r is RegionId {
  return Object.hasOwn(REGIONS, r) && !!REGIONS[r as RegionId].volume;
}
export const AVAILABLE_REGIONS = (Object.keys(REGIONS) as RegionId[]).filter(regionAvailable);
/** Why a region cannot be used (unknown, or no weights volume), or null when it can. */
export function regionProblem(r: string): string | null {
  if (regionAvailable(r)) return null;
  if (Object.hasOwn(REGIONS, r)) return `region ${r} is unavailable: ${REGION_UNAVAILABLE[r as RegionId] || "no weights volume"}`;
  return `unknown region ${r} (${AVAILABLE_REGIONS.join(", ")})`;
}
/** Throws 409 when a stored spec still names an unavailable region (a start must not mount a missing volume). */
export function assertRegionsAvailable(spec: ClusterSpec): void {
  const all = [...spec.regions, ...spec.pools.flatMap((p) => p.regions || [])];
  for (const r of new Set(all)) {
    const why = regionProblem(r);
    if (why) throw new HttpError(409, `regions: ${why}. Edit the cluster spec (regions: ["eu"]) first.`);
  }
}

