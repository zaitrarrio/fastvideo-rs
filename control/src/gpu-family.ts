// GPU model -> architecture ("model family") and market segment ("GPU
// family") for the launch screen's instance-type tables (instance-types.ts,
// docs/control/README.md "Instance types"). Providers spell a GPU many ways
// (Runpod "NVIDIA H100 80GB HBM3" / "H100 SXM", Brev "H100", "RTX6000Ada",
// "RTX PRO Server 6000", GMI "H200"), so the table matches the first rule
// whose pattern finds the name; unknown names get null (shown "-").
//
// model family: the GPU architecture: Blackwell, Hopper, Ada Lovelace, Ampere,
//   Turing, Volta, Pascal, Kepler (NVIDIA); CDNA 3 / CDNA 4 / RDNA (AMD).
// GPU family: the product segment the model is sold in:
//   datacenter  - server parts (H100, A100, L40S, T4, B200, RTX PRO ... Server Edition, MI300X);
//   workstation - professional cards (RTX A6000, RTX 6000 Ada, RTX PRO 6000 Blackwell Workstation / Max-Q);
//   consumer    - GeForce cards (RTX 3090, 4090, 5090).

export type GpuSegment = "datacenter" | "workstation" | "consumer";
export interface GpuClass {
  vendor: "NVIDIA" | "AMD";
  arch: string;
  segment: GpuSegment;
}

/** Ordered: the first match wins (specific before general: "RTX 6000 Ada" before "A6000", "RTX PRO … Server" before "RTX PRO"). */
export const GPU_FAMILY_RULES: { re: RegExp; cls: GpuClass }[] = [
  { re: /\bMI35\d/i, cls: { vendor: "AMD", arch: "CDNA 4", segment: "datacenter" } },
  { re: /\bMI3[02]\d/i, cls: { vendor: "AMD", arch: "CDNA 3", segment: "datacenter" } },
  { re: /\bMI2\d\d/i, cls: { vendor: "AMD", arch: "CDNA 2", segment: "datacenter" } },
  { re: /Radeon Pro V\d+/i, cls: { vendor: "AMD", arch: "RDNA", segment: "datacenter" } },
  { re: /\bGB?[23]00\b|\bB[12]00\b|\bB300\b/i, cls: { vendor: "NVIDIA", arch: "Blackwell", segment: "datacenter" } },
  { re: /RTX\s*PRO\s*\d+\s*Blackwell\s*Server|RTX\s*PRO\s*Server|RTX\s*PRO\s*\d+.*\bServer\b/i, cls: { vendor: "NVIDIA", arch: "Blackwell", segment: "datacenter" } },
  { re: /RTX\s*PRO\s*\d+/i, cls: { vendor: "NVIDIA", arch: "Blackwell", segment: "workstation" } },
  { re: /RTX\s*50\d0\b(?!.*Ada)/i, cls: { vendor: "NVIDIA", arch: "Blackwell", segment: "consumer" } },
  { re: /\bG?H(100|200|800|20)\b/i, cls: { vendor: "NVIDIA", arch: "Hopper", segment: "datacenter" } },
  { re: /\bL40S?\b|\bL4\b|\bL20\b/i, cls: { vendor: "NVIDIA", arch: "Ada Lovelace", segment: "datacenter" } },
  { re: /RTX\s*\d000\s*(SFF\s*)?Ada|RTX\d000Ada|\bAda\b/i, cls: { vendor: "NVIDIA", arch: "Ada Lovelace", segment: "workstation" } },
  { re: /RTX\s*40\d0/i, cls: { vendor: "NVIDIA", arch: "Ada Lovelace", segment: "consumer" } },
  { re: /\bA100\b|\bA800\b|\bA40\b|\bA30\b|\bA10G?\b|\bA16\b|\bA2\b|A100-SXM/i, cls: { vendor: "NVIDIA", arch: "Ampere", segment: "datacenter" } },
  { re: /\bA(2|4|45|5|6)000\b|\bA4500\b|RTX\s*A\d{4}/i, cls: { vendor: "NVIDIA", arch: "Ampere", segment: "workstation" } },
  { re: /RTX\s*30\d0/i, cls: { vendor: "NVIDIA", arch: "Ampere", segment: "consumer" } },
  { re: /\bT4G?\b/i, cls: { vendor: "NVIDIA", arch: "Turing", segment: "datacenter" } },
  { re: /Quadro\s*RTX|RTX\s*[468]000\b(?!\s*Ada)/i, cls: { vendor: "NVIDIA", arch: "Turing", segment: "workstation" } },
  { re: /RTX\s*20\d0|GTX\s*16\d0/i, cls: { vendor: "NVIDIA", arch: "Turing", segment: "consumer" } },
  { re: /\bV100\b/i, cls: { vendor: "NVIDIA", arch: "Volta", segment: "datacenter" } },
  { re: /\bP(4|40|100)\b/i, cls: { vendor: "NVIDIA", arch: "Pascal", segment: "datacenter" } },
  { re: /\bK80\b/i, cls: { vendor: "NVIDIA", arch: "Kepler", segment: "datacenter" } },
];

/** The class of a GPU from any of its names (first match over each name in turn); null when no rule knows it. */
export function gpuClass(...names: (string | null | undefined)[]): GpuClass | null {
  for (const n of names) {
    if (!n) continue;
    for (const r of GPU_FAMILY_RULES) if (r.re.test(n)) return r.cls;
  }
  return null;
}

/** A GPU name without the vendor's prefixes ("NVIDIA GeForce RTX 4090" -> "RTX 4090", "AMD Instinct MI300X OAM" -> "MI300X OAM"). */
export const shortGpuName = (s: string) =>
  s
    .replace(/^(NVIDIA|AMD)\s+/i, "")
    .replace(/^(GeForce|Tesla|Instinct)\s+/i, "")
    .trim();
