// Model families: the deploy-time unit the console speaks in. A family is a
// set of weight trees; deploying a family deploys ALL of its weights (the
// user never picks individual trees). Virtual families compose real ones:
// STREAMING covers every streaming model's weights, ALL covers everything.
// Sizes come from weights-sources.ts HUB_SHA256 (exact byte sums); storage
// is always validated at 1.1x the family's total (the one-click deploy rule).
import { POOL_PRESETS } from "./presets";
import { HUB_SHA256 } from "./weights-sources";

export interface ModelFamily {
  id: string;
  /** H3, LTX, WAN… */
  title: string;
  description: string;
  /** Weight tree names (HUB_TREES keys). Deploying the family fetches all. */
  weights: string[];
  /** True when the family's models are causal streaming (Reactor/native). */
  streaming: boolean;
  /** Total download size of weights, bytes (null: unknown). */
  bytes: number | null;
}

/** Exact byte sum of a Hub tree from the sha256 manifest (null: unrecorded). */
export function treeBytes(tree: string): number | null {
  const rows = HUB_SHA256[tree];
  if (!rows) return null;
  let sum = 0;
  for (const [, b] of rows) {
    if (b == null) return null;
    sum += b;
  }
  return sum;
}

/** Sum of known tree sizes (null if any tree is unrecorded). */
export function treesBytes(trees: string[]): number | null {
  let sum = 0;
  for (const t of trees) {
    const b = treeBytes(t);
    if (b == null) return null;
    sum += b;
  }
  return sum;
}

const norm = (f: string) => f.toLowerCase().replace(/[^a-z0-9]/g, "");

function buildFamilies(): ModelFamily[] {
  // Group presets by their model family tag.
  const byFam = new Map<string, { weights: Set<string>; streaming: boolean; titles: string[] }>();
  for (const p of POOL_PRESETS) {
    for (const m of p.pool.models || []) {
      const fid = norm(m.family || "other");
      let e = byFam.get(fid);
      if (!e) { e = { weights: new Set(), streaming: false, titles: [] }; byFam.set(fid, e); }
      for (const w of p.weights || []) e.weights.add(w);
      for (const w of p.prequant || []) e.weights.add(w);
      e.titles.push(p.title);
      // Causal streaming recipes ride the sfwan engine.
      if (/sfwan|stream/i.test(p.id) || /stream/i.test(p.description)) e.streaming = true;
    }
  }
  const FAMILY_TITLES: Record<string, [string, string]> = {
    h3: ["H3", "MiniMax H3 video generation (turbo, max, trim, ref2v)"],
    ltx2: ["LTX", "LTX-2.5 video generation (turbo, pro, a2v, ref2v)"],
    wan: ["WAN", "Wan 2.2 5B plus the causal streaming companions (SF-Wan, LongLive)"],
  };
  const out: ModelFamily[] = [];
  for (const [fid, e] of byFam) {
    const [title, description] = FAMILY_TITLES[fid] || [fid.toUpperCase(), e.titles[0] || fid];
    const weights = [...e.weights].sort();
    out.push({ id: fid, title, description, weights, streaming: e.streaming, bytes: treesBytes(weights) });
  }
  out.sort((a, b) => a.title.localeCompare(b.title));

  // Virtual families.
  const streamWeights = [...new Set(out.filter((f) => f.streaming).flatMap((f) => f.weights))].sort();
  out.push({
    id: "streaming",
    title: "STREAMING",
    description: "Every streaming model's weights (SF-Wan, LongLive): the Reactor and native streams.",
    weights: streamWeights,
    streaming: true,
    bytes: treesBytes(streamWeights),
  });
  const allWeights = [...new Set(out.filter((f) => f.id !== "streaming").flatMap((f) => f.weights))].sort();
  out.push({
    id: "all",
    title: "ALL",
    description: "Every model family's weights: the full catalog on one volume.",
    weights: allWeights,
    streaming: false,
    bytes: treesBytes(allWeights),
  });
  return out;
}

export const MODEL_FAMILIES: ModelFamily[] = buildFamilies();
export const familyById = (id: string): ModelFamily | undefined => MODEL_FAMILIES.find((f) => f.id === norm(id));

/** Union of weights for several families (deduped, sorted). */
export function familiesWeights(ids: string[]): string[] {
  const s = new Set<string>();
  for (const id of ids) for (const w of familyById(id)?.weights || []) s.add(w);
  return [...s].sort();
}

/** Total bytes for several families (null if any tree size is unknown). */
export function familiesBytes(ids: string[]): number | null {
  return treesBytes(familiesWeights(ids));
}

/** Minimum volume size GB: 10% headroom over the weights (the one-click rule). */
export function minStorageGb(ids: string[]): number | null {
  const b = familiesBytes(ids);
  return b == null ? null : Math.ceil((b * 1.1) / 1e9);
}

export const fmtGb = (gb: number): string => (gb >= 1000 ? `${(gb / 1000).toFixed(1)} TB` : `${gb} GB`);
export const fmtBytes = (b: number | null): string => (b == null ? "unknown" : b >= 1e12 ? `${(b / 1e12).toFixed(2)} TB` : `${(b / 1e9).toFixed(1)} GB`);
