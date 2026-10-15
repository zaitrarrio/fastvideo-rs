// fv-app pricing: the source of truth for the portal's displayed prices.
//
// PLACEHOLDER — these numbers are representative ($0.05–$0.13/s, anchored
// on the known $0.13/s 2K figure), NOT a real price list. The real per-model
// per-second prices must be supplied from the serving stack before this
// portal goes anywhere public. When they are, replace the `usd_per_sec`
// values below; the shape stays the same.
export interface PriceTier {
  id: string;
  label: string;
  usd_per_sec: number;
  note?: string;
}

export interface ModelPrice {
  model: string;
  tiers: PriceTier[];
}

export const PRICES: ModelPrice[] = [
  {
    model: "fasth3",
    tiers: [
      { id: "720p", label: "720p", usd_per_sec: 0.05 },
      { id: "1080p", label: "1080p", usd_per_sec: 0.08 },
    ],
  },
  {
    model: "h3-max",
    tiers: [
      { id: "720p", label: "720p", usd_per_sec: 0.07 },
      { id: "1080p", label: "1080p", usd_per_sec: 0.11 },
      { id: "2k", label: "2K", usd_per_sec: 0.13, note: "anchor price" },
    ],
  },
  {
    model: "wan22-ti2v-5b",
    tiers: [
      { id: "720p", label: "720p", usd_per_sec: 0.05 },
      { id: "1080p", label: "1080p", usd_per_sec: 0.09 },
    ],
  },
  {
    model: "ltx-2.5",
    tiers: [
      { id: "720p", label: "720p", usd_per_sec: 0.06 },
      { id: "1080p", label: "1080p", usd_per_sec: 0.1 },
    ],
  },
];

export function priceFor(model: string, tierId: string): PriceTier | null {
  return PRICES.find((p) => p.model === model)?.tiers.find((t) => t.id === tierId) || null;
}
