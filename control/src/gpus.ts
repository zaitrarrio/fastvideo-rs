// GPU memory per Runpod GPU type id (GB, as Runpod's gpuTypes.memoryInGb
// reports it), for checks that must not wait on Runpod: a serverless preset's
// least GPU memory (presets.ts min_vram_gb) against the GPU types an endpoint
// may run on. test/unit/serverless-presets.test.ts checks every id in
// RUNPOD_GPU_TYPES has a value here.
import type { RUNPOD_GPU_TYPES } from "./enums";

export const GPU_MEMORY_GB: Record<(typeof RUNPOD_GPU_TYPES)[number], number> = {
  "AMD Instinct MI300X OAM": 192,
  "NVIDIA A100 80GB PCIe": 80,
  "NVIDIA A100-SXM4-40GB": 40,
  "NVIDIA A100-SXM4-80GB": 80,
  "NVIDIA A40": 48,
  "NVIDIA B200": 180,
  "NVIDIA B300 SXM6 AC": 288,
  "NVIDIA B300 SXM6 AC MIG 1g.34gb": 34,
  "NVIDIA GeForce RTX 3070": 8,
  "NVIDIA GeForce RTX 3080": 10,
  "NVIDIA GeForce RTX 3080 Ti": 12,
  "NVIDIA GeForce RTX 3090": 24,
  "NVIDIA GeForce RTX 3090 Ti": 24,
  "NVIDIA GeForce RTX 4070 Ti": 12,
  "NVIDIA GeForce RTX 4080": 16,
  "NVIDIA GeForce RTX 4080 SUPER": 16,
  "NVIDIA GeForce RTX 4090": 24,
  "NVIDIA GeForce RTX 5080": 16,
  "NVIDIA GeForce RTX 5090": 32,
  "NVIDIA H100 80GB HBM3": 80,
  "NVIDIA H100 NVL": 94,
  "NVIDIA H100 PCIe": 80,
  "NVIDIA H200": 141,
  "NVIDIA H200 NVL": 143,
  "NVIDIA L4": 24,
  "NVIDIA L40": 48,
  "NVIDIA L40S": 48,
  "NVIDIA RTX 2000 Ada Generation": 16,
  "NVIDIA RTX 4000 Ada Generation": 20,
  "NVIDIA RTX 4000 SFF Ada Generation": 20,
  "NVIDIA RTX 5000 Ada Generation": 32,
  "NVIDIA RTX 6000 Ada Generation": 48,
  "NVIDIA RTX A2000": 6,
  "NVIDIA RTX A4000": 16,
  "NVIDIA RTX A4500": 20,
  "NVIDIA RTX A5000": 24,
  "NVIDIA RTX A6000": 48,
  "NVIDIA RTX PRO 4000 Blackwell": 24,
  "NVIDIA RTX PRO 4500 Blackwell": 32,
  "NVIDIA RTX PRO 5000 Blackwell": 48,
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": 96,
  "NVIDIA RTX PRO 6000 Blackwell Server Edition": 96,
  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 96,
  "Tesla V100-PCIE-16GB": 16,
  "Tesla V100-SXM2-16GB": 16,
};

/** A GPU type's memory (GB), or null when fv-control does not know the type. */
export const gpuMemoryGb = (id: string): number | null => (GPU_MEMORY_GB as Record<string, number>)[id] ?? null;
/** The GPU types fv-control knows with at least `gb` GB, largest-first order kept from GPU_MEMORY_GB. */
export const gpusWithAtLeast = (gb: number): string[] => Object.entries(GPU_MEMORY_GB).filter(([id, m]) => m >= gb && !id.startsWith("AMD")).map(([id]) => id);
