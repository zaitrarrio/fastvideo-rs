// The launch screen's instance-type tables (owner 2026-10-13): GET /api/providers/:id/instance-types normalizes
// every type each provider lists (Runpod GPU types × data centres, every Brev instance type, every GMI product),
// marks the ones fv-control cannot launch with the reason, caches the upstream list and lets ?refresh=1 bypass it
// (at most once per 30 s per provider). Fixtures are trimmed from the live answers (2026-10-13, GET / read-only).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mintApiToken } from "../../src/auth";
import { resetBrevTypes } from "../../src/brev";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { gpuClass, shortGpuName } from "../../src/gpu-family";
import { _app } from "../../src/index";
import { normalizeBrev, normalizeGmi, normalizeRunpod, REFRESH_MIN_MS, resetRefreshLimit } from "../../src/instance-types";
import { clearTtlCache, ttlCached, ttlCachedAt } from "../../src/ttlcache";
import { launchable, type ITRow } from "../../ui/forms/itable";
import { d1 } from "./d1shim";

// ---------------------------------------------------------------- fixtures (live shapes)
const RP_TYPES = [
  { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", displayName: "RTX PRO 6000", manufacturer: "Nvidia", memoryInGb: 96, securePrice: 2.49, communityPrice: 1.69, lowestPrice: { stockStatus: "Medium", minVcpu: 16, minMemory: 90 } },
  { id: "NVIDIA H100 80GB HBM3", displayName: "H100 SXM", manufacturer: "Nvidia", memoryInGb: 80, securePrice: 3.99, communityPrice: 2.69, lowestPrice: { stockStatus: "Medium", minVcpu: 8, minMemory: 125 } },
  { id: "AMD Instinct MI300X OAM", displayName: "MI300X", manufacturer: "AMD", memoryInGb: 192, securePrice: 2.39, communityPrice: 0.5, lowestPrice: { stockStatus: null, minVcpu: null, minMemory: null } },
  { id: "NVIDIA GeForce RTX 3070", displayName: "RTX 3070", manufacturer: "Nvidia", memoryInGb: 8, securePrice: 0, communityPrice: 0.13, lowestPrice: { stockStatus: "Low", minVcpu: 18, minMemory: 17 } },
  { id: "NVIDIA RTX PRO 4500 Blackwell Server Edition", displayName: "RTX PRO 4500", manufacturer: "Nvidia", memoryInGb: 32, securePrice: 0.72, communityPrice: 0.5, lowestPrice: { stockStatus: "Low", minVcpu: 16, minMemory: 94 } },
  { id: "NVIDIA B200", displayName: "B200", manufacturer: "Nvidia", memoryInGb: 180, securePrice: 7.99, communityPrice: 5.98, lowestPrice: { stockStatus: null } },
  { id: "unknown", displayName: "unknown", memoryInGb: 0, securePrice: 0 },
];
const RP_DCS = [
  { id: "EU-RO-1", name: "EU-RO-1", location: "Europe", storageSupport: false, listed: true, gpuAvailability: [
    { available: true, stockStatus: "Low", gpuTypeId: "NVIDIA RTX PRO 6000 Blackwell Server Edition" },
    { available: false, stockStatus: null, gpuTypeId: "NVIDIA H100 80GB HBM3" },
    { available: true, stockStatus: "Low", gpuTypeId: "AMD Instinct MI300X OAM" },
    { available: true, stockStatus: "Low", gpuTypeId: "NVIDIA RTX PRO 4500 Blackwell Server Edition" },
  ] },
  { id: "US-CA-2", name: "US-CA-2", location: "United States", storageSupport: true, listed: true, gpuAvailability: [
    { available: true, stockStatus: "High", gpuTypeId: "NVIDIA H100 80GB HBM3" },
    { available: true, stockStatus: "Low", gpuTypeId: "NVIDIA GeForce RTX 3070" },
  ] },
  { id: "AP-IN-2", name: "AP-IN-2", location: "India", storageSupport: false, listed: false, gpuAvailability: [{ available: true, stockStatus: "Low", gpuTypeId: "NVIDIA H100 80GB HBM3" }] },
];
const gp = (n: number) => ({ size: "0B", type: n === 1 ? "gp3" : n === 2 ? "gp2" : n === 3 ? "io1" : "io2", min_size: "10GiB", max_size: "16TiB", price_per_gb_hr: { currency: "USD", amount: "0.000132" } });
const BREV_RAW = [
  { type: "g4dn.8xlarge", supported_gpus: [{ count: 1, memory: "16GiB", manufacturer: "NVIDIA", name: "T4", memory_bytes: { value: 16, unit: "GB" } }], memory: "128GiB", memory_bytes: { value: 131072, unit: "MiB" }, vcpu: 32, supported_architectures: ["x86_64"], elastic_root_volume: true, location: "us-east-1", is_available: true, base_price: { currency: "USD", amount: "2.611200" }, estimated_deploy_time: "7m0s", provider: "aws", stoppable: true, supported_storage: [gp(1), gp(1), gp(2), gp(3), gp(4)] },
  { type: "massedcompute_L40Sx4", supported_gpus: [{ count: 4, memory: "48GiB", manufacturer: "NVIDIA", name: "L40S", memory_bytes: { value: 48, unit: "GB" } }], memory: "288GiB", memory_bytes: { value: 288, unit: "GB" }, vcpu: 88, supported_architectures: ["x86_64"], location: "kansascity-usa-1", is_available: true, base_price: { currency: "USD", amount: "4.224000" }, estimated_deploy_time: "5m30s", provider: "shadeform", cloud: "massedcompute", supported_storage: [{ count: 1, size: "2TiB452GiB", type: "ssd", size_bytes: { value: 2500, unit: "GB" } }] },
  { type: "m8idb.4xlarge", memory: "64GiB", vcpu: 16, supported_architectures: ["x86_64"], location: "us-east-1", is_available: true, base_price: { currency: "USD", amount: "1.1" }, provider: "aws", stoppable: true, elastic_root_volume: true, supported_storage: [gp(1)] },
  { type: "g4ad.xlarge", supported_gpus: [{ count: 1, memory: "8GiB", manufacturer: "AMD", name: "Radeon Pro V520" }], memory: "16GiB", vcpu: 4, supported_architectures: ["x86_64"], location: "us-east-1", is_available: true, base_price: { currency: "USD", amount: "0.38" }, provider: "aws", stoppable: true },
  { type: "g5g.xlarge", supported_gpus: [{ count: 1, memory: "16GiB", manufacturer: "NVIDIA", name: "T4g" }], memory: "8GiB", vcpu: 4, supported_architectures: ["arm64"], location: "us-east-1", is_available: false, base_price: { currency: "USD", amount: "0.42" }, provider: "aws", stoppable: true },
  { type: "gpu-l40s-d.1gpu-48vcpu-288gb", supported_gpus: [{ count: 1, memory: "48GiB", manufacturer: "NVIDIA", name: "L40S" }], memory: "288GiB", memory_bytes: { value: 288, unit: "GiB" }, vcpu: 48, supported_architectures: ["x86_64"], elastic_root_volume: true, location: "eu-north1", is_available: true, base_price: { currency: "USD", amount: "3.301920" }, provider: "nebius", stoppable: true, supported_storage: [{ count: 1, size: "0B", type: "network-ssd", min_size: "50GiB", max_size: "2TiB512GiB", min_size_bytes: { value: 50, unit: "GiB" }, max_size_bytes: { value: 2560, unit: "GiB" } }] },
];
const GMI_RAW = [
  { gpuModel: "CPU-only", idc: "us-central-iowa1", name: "gmi.container.intel.x4660.large", price: 47500, productLine: "Container", type: "Container", valid: true, spec: { basic: [{ brand: "Intel", model: "Xeon", name: "CPU", qty: "2", unit: "Core" }, { model: "DDR", name: "Memory", qty: "4", unit: "GiB" }, { model: "SATA SSD", name: "OS Storage (Ephemeral)", qty: "10", unit: "GiB" }, { model: "SATA SSD", name: "Data Storage", qty: "30", unit: "GiB" }], resource: { cpu: { qty: 2, unit: "" }, gpu: { qty: 0, unit: "" }, memory: { qty: 4, unit: "Gi" }, localStorage: { qty: 10, unit: "Gi" } } } },
  { gpuModel: "H200", idc: "us-denver-1", name: "container.h200.x1", price: 320, type: "Container", valid: true, spec: { resource: { cpu: { qty: 16 }, gpu: { qty: 1 }, memory: { qty: 200, unit: "Gi" } } } },
  { gpuModel: "B200", idc: "us-denver-1", name: "container.b200.x8", price: 7200, type: "Container", valid: false, spec: {} },
];

describe("gpu-family: model family (architecture) and GPU family (segment)", () => {
  const cases: [string[], string | null, string | null][] = [
    [["NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000"], "Blackwell", "datacenter"],
    [["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], "Blackwell", "workstation"],
    [["NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"], "Blackwell", "workstation"],
    [["RTX PRO Server 6000"], "Blackwell", "datacenter"],
    [["NVIDIA B200"], "Blackwell", "datacenter"],
    [["NVIDIA B300 SXM6 AC MIG 1g.34gb"], "Blackwell", "datacenter"],
    [["NVIDIA GeForce RTX 5090"], "Blackwell", "consumer"],
    [["NVIDIA RTX 5000 Ada Generation"], "Ada Lovelace", "workstation"],
    [["RTX6000Ada"], "Ada Lovelace", "workstation"],
    [["NVIDIA RTX 4000 SFF Ada Generation"], "Ada Lovelace", "workstation"],
    [["NVIDIA GeForce RTX 4090"], "Ada Lovelace", "consumer"],
    [["NVIDIA L40S"], "Ada Lovelace", "datacenter"],
    [["NVIDIA L4"], "Ada Lovelace", "datacenter"],
    [["NVIDIA H100 80GB HBM3"], "Hopper", "datacenter"],
    [["NVIDIA H200 NVL"], "Hopper", "datacenter"],
    [["GH200"], "Hopper", "datacenter"],
    [["NVIDIA A100-SXM4-80GB"], "Ampere", "datacenter"],
    [["A10G"], "Ampere", "datacenter"],
    [["NVIDIA RTX A6000"], "Ampere", "workstation"],
    [["A4000"], "Ampere", "workstation"],
    [["NVIDIA GeForce RTX 3090 Ti"], "Ampere", "consumer"],
    [["T4g"], "Turing", "datacenter"],
    [["Tesla V100-SXM2-16GB"], "Volta", "datacenter"],
    [["P4"], "Pascal", "datacenter"],
    [["AMD Instinct MI300X OAM"], "CDNA 3", "datacenter"],
    [["AMD Instinct MI350 OAM"], "CDNA 4", "datacenter"],
    [["Radeon Pro V520"], "RDNA", "datacenter"],
    [["container.h200.x1"], "Hopper", "datacenter"],
    [["Some Future GPU"], null, null],
  ];
  it.each(cases)("%j -> %s / %s", (names, arch, seg) => {
    const c = gpuClass(...names);
    expect(c?.arch ?? null).toBe(arch);
    expect(c?.segment ?? null).toBe(seg);
  });
  it("shortGpuName drops the vendor prefixes", () => {
    expect(shortGpuName("NVIDIA GeForce RTX 4090")).toBe("RTX 4090");
    expect(shortGpuName("AMD Instinct MI300X OAM")).toBe("MI300X OAM");
    expect(shortGpuName("Tesla V100-PCIE-16GB")).toBe("V100-PCIE-16GB");
  });
});

describe("normalizeRunpod: GPU types × data centres", () => {
  const rows = normalizeRunpod({ gpuTypes: RP_TYPES, dataCenters: RP_DCS });
  const by = (id: string) => rows.find((r) => r.id === id)!;
  it("one row per (type, data centre) Runpod lists, one per type no data centre lists; never 'unknown'", () => {
    expect(rows.map((r) => r.id).sort()).toEqual(
      [
        "NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1",
        "NVIDIA H100 80GB HBM3@EU-RO-1",
        "AMD Instinct MI300X OAM@EU-RO-1",
        "NVIDIA RTX PRO 4500 Blackwell Server Edition@EU-RO-1",
        "NVIDIA H100 80GB HBM3@US-CA-2",
        "NVIDIA GeForce RTX 3070@US-CA-2",
        "NVIDIA H100 80GB HBM3@AP-IN-2",
        "NVIDIA B200@-",
      ].sort(),
    );
  });
  it("the columns of a launchable row: stock, $/hr, GPU, families, RAM / vCPU (the least per GPU), storage, region; launch fields", () => {
    const r = by("NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1");
    expect(r).toMatchObject({
      name: "NVIDIA RTX PRO 6000 Blackwell Server Edition",
      available: true,
      stock: "Low",
      usd_per_hr: 2.49,
      usd_per_gpu_hr: 2.49,
      gpu_model: "RTX PRO 6000",
      model_family: "Blackwell",
      gpu_family: "datacenter",
      gpu: "1 × RTX PRO 6000",
      gpu_count: 1,
      vram_gb: 96,
      vram_total_gb: 96,
      ram_gb: 90,
      vcpu: 16,
      storage_type: "container disk + network volume (bz0ofkfx45)",
      storage_kind: "elastic",
      storage_min_gb: 5,
      storage_max_gb: 500,
      region: "EU-RO-1",
      cloud: "Runpod Secure Cloud · Europe",
      supported: true,
      reason: null,
      launch: { gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], region: "eu" },
    });
    expect(r.spec).toBe("1× RTX PRO 6000 96 GB · ≥ 16 vCPU · ≥ 90 GB RAM");
    // Listed but out of stock there: launchable, not available.
    expect(by("NVIDIA H100 80GB HBM3@EU-RO-1")).toMatchObject({ available: false, stock: null, supported: true });
  });
  it("not launchable, with the reason: no weights volume, unlisted DC, AMD, not in Secure Cloud, unknown type, no DC", () => {
    expect(by("NVIDIA H100 80GB HBM3@US-CA-2")).toMatchObject({ supported: false, available: true, stock: "High", launch: null, storage_type: "container disk (network volumes offered)" });
    expect(by("NVIDIA H100 80GB HBM3@US-CA-2").reason).toMatch(/no fv weights volume in US-CA-2.*EU-RO-1.*weights_source r2/);
    expect(by("NVIDIA H100 80GB HBM3@AP-IN-2").reason).toMatch(/AP-IN-2 is not a listed data centre/);
    expect(by("AMD Instinct MI300X OAM@EU-RO-1")).toMatchObject({ supported: false, gpu_vendor: "AMD", model_family: "CDNA 3" });
    expect(by("AMD Instinct MI300X OAM@EU-RO-1").reason).toMatch(/AMD GPU: the serve images need NVIDIA CUDA/);
    expect(by("NVIDIA GeForce RTX 3070@US-CA-2").reason).toMatch(/not offered in Secure Cloud/);
    expect(by("NVIDIA GeForce RTX 3070@US-CA-2").usd_per_hr).toBeNull();
    expect(by("NVIDIA RTX PRO 4500 Blackwell Server Edition@EU-RO-1").reason).toMatch(/not in fv-control's Runpod GPU type list/);
    expect(by("NVIDIA B200@-")).toMatchObject({ region: null, available: false, supported: false });
    expect(by("NVIDIA B200@-").reason).toMatch(/no data centre lists it now/);
  });
  it("without the data-centre stock: one row per type with the global stock", () => {
    const r = normalizeRunpod({ gpuTypes: RP_TYPES, dataCenters: null });
    expect(r).toHaveLength(6);
    expect(r.find((x) => x.name === "NVIDIA H100 80GB HBM3")).toMatchObject({ region: null, stock: "Medium", available: true, supported: false });
  });
  // Owner 2026-10-13: "I should be able to launch Runpod in any region if I am using R2": each row says why it cannot
  // be launched per weights source, so the launch form recomputes the table when its Weights change (no refetch).
  it("launchable per weights source: r2 / none in any data centre; volume in EU-RO-1 only; AMD, unknown, not Secure Cloud never", () => {
    const rows = normalizeRunpod({ gpuTypes: RP_TYPES, dataCenters: RP_DCS });
    const by = (id: string) => rows.find((r) => r.id === id)!;
    const ok = (id: string, w: "volume" | "r2" | "hub" | "none") => launchable(by(id) as ITRow, w).supported;
    // A non-volume data centre: launchable with r2 / none, not with the volume (the API's own flags stay the volume's).
    const us = by("NVIDIA H100 80GB HBM3@US-CA-2");
    expect([us.volume_dc, us.supported, us.reasons]).toEqual([false, false, { volume: expect.stringMatching(/no fv weights volume in US-CA-2/), r2: null, hub: null, none: null }]);
    expect([ok(us.id, "volume"), ok(us.id, "r2"), ok(us.id, "none")]).toEqual([false, true, true]);
    // The volume's data centre: launchable with each.
    const eu = by("NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1");
    expect([eu.volume_dc, eu.reasons]).toEqual([true, { volume: null, r2: null, hub: null, none: null }]);
    // The other reasons hold whatever the weights.
    for (const id of ["AMD Instinct MI300X OAM@EU-RO-1", "NVIDIA GeForce RTX 3070@US-CA-2", "NVIDIA RTX PRO 4500 Blackwell Server Edition@EU-RO-1", "NVIDIA H100 80GB HBM3@AP-IN-2", "NVIDIA B200@-"])
      for (const w of ["volume", "r2", "hub", "none"] as const) expect(ok(id, w), `${id} ${w}`).toBe(false);
    expect(by("NVIDIA GeForce RTX 3070@US-CA-2").reasons!.r2).toMatch(/not offered in Secure Cloud/);
    expect(by("NVIDIA GeForce RTX 3070@US-CA-2").reasons!.volume).toMatch(/not offered in Secure Cloud.*no fv weights volume in US-CA-2/);
    expect(by("NVIDIA B200@-").volume_dc).toBeNull();
    // Without the map (an older server): the row's own flags.
    expect(launchable({ ...us, reasons: undefined } as ITRow, "r2")).toEqual({ supported: false, reason: us.reason });
  });
});

describe("normalizeBrev: every instance type of the listing", () => {
  const rows = normalizeBrev(BREV_RAW, { allowed: ["g4dn.8xlarge", "massedcompute_L40Sx4", "m8idb.4xlarge", "g4ad.xlarge", "g5g.xlarge"], off: null, price: (t) => (t === "massedcompute_L40Sx4" ? 4 : null) });
  const by = (id: string) => rows.find((r) => r.id === id)!;
  it("an elastic (EBS) type: spec, $/GPU-hr, RAM from memory_bytes, storage classes and range, cloud, stoppable, deploy time", () => {
    expect(by("g4dn.8xlarge")).toMatchObject({
      spec: "1× T4 16 GB · 32 vCPU · 128 GB RAM",
      available: true,
      usd_per_hr: 2.6112,
      usd_per_gpu_hr: 2.61,
      gpu_model: "T4",
      model_family: "Turing",
      gpu_family: "datacenter",
      vram_gb: 16,
      ram_gb: 128,
      vcpu: 32,
      storage_type: "gp3, gp2, io1 +1",
      storage_kind: "elastic",
      storage_gb: null,
      storage_min_gb: 10,
      storage_max_gb: 16384,
      region: "us-east-1",
      cloud: "aws",
      stoppable: true,
      deploy_s: 420,
      supported: true,
      launch: { provider_gpu: "g4dn.8xlarge" },
    });
  });
  it("a fixed-disk multi-GPU type behind a marketplace: VRAM per GPU and in all, the price override, provider/cloud, not stoppable", () => {
    expect(by("massedcompute_L40Sx4")).toMatchObject({ gpu: "4 × L40S", gpu_count: 4, vram_gb: 48, vram_total_gb: 192, usd_per_hr: 4, usd_per_gpu_hr: 1, storage_type: "ssd", storage_kind: "fixed", storage_gb: 2500, cloud: "shadeform/massedcompute", stoppable: false, deploy_s: 330, model_family: "Ada Lovelace" });
    expect(by("gpu-l40s-d.1gpu-48vcpu-288gb")).toMatchObject({ storage_type: "network-ssd", storage_min_gb: 50, storage_max_gb: 2560, ram_gb: 288 });
  });
  it("not launchable: CPU-only, AMD, arm64-only, outside BREV_INSTANCE_TYPES; and every row when Brev is off", () => {
    expect(by("m8idb.4xlarge")).toMatchObject({ supported: false, gpu: "-", gpu_count: 0, vram_gb: null, spec: "CPU only · 16 vCPU · 64 GB RAM" });
    expect(by("m8idb.4xlarge").reason).toMatch(/no GPU/);
    expect(by("g4ad.xlarge").reason).toMatch(/AMD GPU/);
    expect(by("g5g.xlarge").reason).toMatch(/arm64 only: the serve images are x86_64/);
    expect(by("g5g.xlarge").available).toBe(false);
    expect(by("gpu-l40s-d.1gpu-48vcpu-288gb").reason).toBe("not in BREV_INSTANCE_TYPES (the account's allowed types)");
    const off = normalizeBrev(BREV_RAW, { allowed: ["g4dn.8xlarge"], off: "BREV_API_TOKEN is not set" });
    expect(off.every((r) => !r.supported && /NVIDIA Brev is off/.test(r.reason!))).toBe(true);
  });
});

describe("normalizeGmi: every container product", () => {
  const rows = normalizeGmi(GMI_RAW, { allowed: ["container.h200.x1", "gmi.container.intel.x4660.large"], off: null, divisor: 100 });
  const by = (id: string) => rows.find((r) => r.id === id)!;
  it("the spec's resources, storage, the price divisor, VRAM from the model table, the IDC as region and in the launch", () => {
    expect(by("container.h200.x1@us-denver-1")).toMatchObject({ gpu: "1 × H200", vram_gb: 141, ram_gb: 200, vcpu: 16, usd_per_hr: 3.2, region: "us-denver-1", cloud: "GMI Cloud", model_family: "Hopper", supported: true, launch: { provider_gpu: "container.h200.x1", provider_region: "us-denver-1" } });
    expect(by("gmi.container.intel.x4660.large@us-central-iowa1")).toMatchObject({ gpu_count: 0, ram_gb: 4, vcpu: 2, storage_type: "SATA SSD OS (ephemeral) + SATA SSD data", storage_kind: "fixed", storage_gb: 30, supported: false });
    expect(by("gmi.container.intel.x4660.large@us-central-iowa1").reason).toMatch(/no GPU/);
  });
  it("the GPU count from the product name when the spec has none; not valid = not available; outside GMI_PRODUCTS", () => {
    expect(by("container.b200.x8@us-denver-1")).toMatchObject({ gpu_count: 8, vram_total_gb: 1440, usd_per_hr: 72, usd_per_gpu_hr: 9, available: false, supported: false, reason: "not in GMI_PRODUCTS (the account's allowed products)" });
  });
});

// ---------------------------------------------------------------- the endpoint
const realFetch = globalThis.fetch;
let calls: string[] = [];
let failRunpod = false;
let failDcs = false;
function upstream() {
  calls = [];
  globalThis.fetch = (async (input: any, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.url;
    const host = new URL(url).host;
    if (host === "api.runpod.io") {
      const q = JSON.parse(String(init?.body || "{}")).query || "";
      calls.push(q.includes("gpuAvailability") ? "runpod:dcs" : "runpod:types");
      if (failRunpod) return new Response("down", { status: 503 });
      if (q.includes("gpuAvailability")) return failDcs ? Response.json({ errors: [{ message: "Cannot query field gpuAvailability" }] }) : Response.json({ data: { dataCenters: RP_DCS } });
      return Response.json({ data: { gpuTypes: RP_TYPES } });
    }
    if (host === "brev.test" && url.includes("/alltypesavailable/")) return calls.push("brev"), Response.json({ allInstanceTypes: BREV_RAW });
    if (host === "gmi.test" && url.includes("/containers/products")) return calls.push(`gmi${new URL(url).search}`), Response.json(GMI_RAW);
    return new Response(`unexpected upstream ${url}`, { status: 599 });
  }) as typeof fetch;
}
const mkEnv = (over: Partial<Env> = {}) =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_SECRETKEY123",
    GITHUB_PAT: "ghp_x",
    PUBLIC_URL: "https://ctl.test",
    GMI_API_KEY: "gmi_SECRET_0123456789",
    GMI_API: "https://gmi.test/api",
    GMI_PRODUCTS: "container.h200.x1",
    GMI_BUDGET_USD: "20",
    BREV_API_TOKEN: "brev_SECRET_0123456789",
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "g4dn.8xlarge",
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;
const ctx = { waitUntil() {}, passThroughOnException() {} } as any;
async function client(env: Env) {
  const auth = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
  return async (path: string) => {
    const r = await _app.fetch(new Request(`https://ctl.test${path}`, { headers: auth }), env, ctx);
    return { status: r.status, j: (await r.json()) as any, text: "" };
  };
}
beforeEach(() => {
  clearTtlCache();
  resetBrevTypes();
  resetRefreshLimit();
  failRunpod = false;
  failDcs = false;
});
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("GET /api/providers/:id/instance-types", () => {
  it("Runpod: every row, launchable first, counts; the second ask is served from the cache", async () => {
    upstream();
    const api = await client(mkEnv());
    const r = await api("/api/providers/runpod/instance-types");
    expect(r.status).toBe(200);
    expect(r.j).toMatchObject({ provider: "runpod", enabled: true, live_ok: true, error: null, cached: false, refresh_limited: null, counts: { total: 8, supported: 2 } });
    expect(r.j.rows[0].supported).toBe(true);
    expect(r.j.rows[0].id).toBe("NVIDIA RTX PRO 6000 Blackwell Server Edition@EU-RO-1"); // available before not available
    expect(typeof r.j.at).toBe("number");
    expect(calls.sort()).toEqual(["runpod:dcs", "runpod:types"]);
    const again = await api("/api/providers/runpod/instance-types");
    expect(again.j.at).toBe(r.j.at);
    expect(calls).toHaveLength(2);
    // No secret in the answer.
    expect(JSON.stringify(r.j)).not.toContain("SECRET");
  });
  it("?refresh=1 reads the provider again; a second refresh within 30 s is served cached (refresh_limited); after it, allowed", async () => {
    upstream();
    const api = await client(mkEnv());
    let t = 1_800_000_000_000;
    vi.spyOn(Date, "now").mockImplementation(() => t);
    await api("/api/providers/gmi/instance-types");
    expect(calls).toEqual(["gmi"]);
    t += 1000;
    const r1 = await api("/api/providers/gmi/instance-types?refresh=1");
    expect(calls).toEqual(["gmi", "gmi"]);
    expect(r1.j).toMatchObject({ cached: false, refresh_limited: null, at: t });
    t += 5000;
    const r2 = await api("/api/providers/gmi/instance-types?refresh=1");
    expect(calls).toHaveLength(2);
    expect(r2.j).toMatchObject({ cached: true, refresh_limited: Math.ceil((REFRESH_MIN_MS - 5000) / 1000), at: t - 5000 });
    t += REFRESH_MIN_MS;
    const r3 = await api("/api/providers/gmi/instance-types?refresh=1");
    expect(calls).toHaveLength(3);
    expect(r3.j.refresh_limited).toBeNull();
    // The limit is per provider: Brev's first refresh is allowed now.
    await api("/api/providers/brev/instance-types?refresh=1");
    expect(calls.filter((c) => c === "brev")).toHaveLength(1);
    const b2 = await api("/api/providers/brev/instance-types?refresh=1");
    expect(b2.j.refresh_limited).toBeGreaterThan(0);
    expect(calls.filter((c) => c === "brev")).toHaveLength(1);
  });
  it("GMI lists every product (all IDCs: no idc filter); Brev every type of the org", async () => {
    upstream();
    const api = await client(mkEnv());
    const g = await api("/api/providers/gmi/instance-types");
    expect(calls).toEqual(["gmi"]);
    expect(g.j.counts).toEqual({ total: 3, supported: 1, available: 2 });
    const b = await api("/api/providers/brev/instance-types");
    expect(b.j.counts.total).toBe(6);
    expect(b.j.rows.filter((x: any) => x.supported).map((x: any) => x.name)).toEqual(["g4dn.8xlarge"]);
  });
  it("a provider that is off: its configured types, not launchable, no upstream call", async () => {
    upstream();
    const api = await client(mkEnv({ BREV_API_TOKEN: "" }));
    const r = await api("/api/providers/brev/instance-types");
    expect(r.j).toMatchObject({ enabled: false, live_ok: false, counts: { total: 1, supported: 0 } });
    expect(r.j.reason).toMatch(/NVIDIA Brev is off: BREV_API_TOKEN is not set/);
    expect(r.j.rows[0]).toMatchObject({ name: "g4dn.8xlarge", supported: false });
    expect(calls).toEqual([]);
  });
  it("an unreachable upstream: the error, the configured types (not launchable); a failed refresh serves the cached list", async () => {
    upstream();
    failRunpod = true;
    const api = await client(mkEnv());
    const r = await api("/api/providers/runpod/instance-types");
    expect(r.status).toBe(200);
    expect(r.j.live_ok).toBe(false);
    expect(r.j.error).toMatch(/runpod/);
    expect(r.j.rows.every((x: any) => !x.supported)).toBe(true);
    failRunpod = false;
    const ok = await api("/api/providers/runpod/instance-types");
    expect(ok.j.live_ok).toBe(true);
    failRunpod = true;
    const rf = await api("/api/providers/runpod/instance-types?refresh=1");
    expect(rf.j.live_ok).toBe(true);
    expect(rf.j.error).toMatch(/the refresh failed: .*showing the cached list/);
    expect(rf.j.counts.total).toBe(8);
  });
  it("Runpod without the data-centre stock (a schema change): one row per GPU type, saying so", async () => {
    upstream();
    failDcs = true;
    const r = await (await client(mkEnv()))("/api/providers/runpod/instance-types");
    expect(r.j.counts.total).toBe(6);
    expect(r.j.error).toMatch(/data-centre stock could not be read/);
  });
  it("an unknown provider: 404; the CLI's endpoints are unchanged", async () => {
    upstream();
    const api = await client(mkEnv());
    expect((await api("/api/providers/cloudrift/instance-types")).status).toBe(404);
    expect((await api("/api/providers/nope/instance-types")).status).toBe(404);
    expect((await api("/api/providers/gmi/offers")).status).toBe(200);
  });
});

describe("ttlCached refresh", () => {
  it("refresh skips a stored value; a failed refresh keeps it", async () => {
    let n = 0;
    expect(await ttlCached("r", 60_000, async () => ++n)).toBe(1);
    expect(await ttlCached("r", 60_000, async () => ++n)).toBe(1);
    expect(await ttlCached("r", 60_000, async () => ++n, { refresh: true })).toBe(2);
    const at = ttlCachedAt("r");
    await expect(ttlCached("r", 60_000, async () => Promise.reject(new Error("down")), { refresh: true })).rejects.toThrow("down");
    expect(await ttlCached("r", 60_000, async () => ++n)).toBe(2);
    expect(ttlCachedAt("r")).toBe(at);
    expect(ttlCachedAt("none")).toBeNull();
  });
});
