// Runpod standalone pods in any data centre (owner 2026-10-13: "why are we only show Runpod with EU weights as a
// launch option? I should be able to launch Runpod in any region if I am using R2"): the launch request's shape for
// r2 in a non-EU data centre, the volume in EU, the fallbacks' order; the placements and the create payload it
// becomes; the rules the validator, the launch and the image preflight share (they all go through standaloneSpec).
import { describe, expect, it } from "vitest";
import { workerCreatePayload, workerPlacements } from "../../src/cluster/payloads";
import { presetWeightsDefault, standaloneSpec } from "../../src/standalone";
import { validate } from "../../src/schemas";

const H100 = "NVIDIA H100 80GB HBM3";
const H200 = "NVIDIA H200";
const launch = (x: Record<string, unknown>) => standaloneSpec({ name: "anydc", ...x } as any);
const placements = (x: Record<string, unknown>) => {
  const { spec } = launch(x);
  return workerPlacements(spec, spec.pools[0]!);
};
/** The launch check's answer at a path: what the validator (POST /api/standalone/validate) and the preflight say. */
const refused = (x: Record<string, unknown>) => {
  try {
    launch(x);
    return null;
  } catch (e) {
    const issues = ((e as any).extra?.issues as { path: (string | number)[]; message: string }[]) || [{ path: [], message: (e as Error).message }];
    return issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ");
  }
};

describe("the request shape: r2 in any data centre, the volume in EU, the fallbacks in order", () => {
  it("r2 in a non-EU data centre: the pool's data_centers, no volume, the nominal region eu; dataCenterIds = that DC", () => {
    const { spec } = launch({ preset: "wan", weights_source: "r2", gpu_types: [H100], data_centers: ["US-TX-3"] });
    const pool = spec.pools[0]!;
    expect([pool.weights_source, pool.volume, pool.data_centers, spec.regions]).toEqual(["r2", false, ["US-TX-3"], ["eu"]]);
    const pl = workerPlacements(spec, pool);
    expect(pl).toEqual([{ dc: "US-TX-3", gpu: H100 }]);
    const body = workerCreatePayload("n", "img", pool, pl[0]!, { FV_WEIGHTS_SOURCE: "r2" });
    expect(body.dataCenterIds).toEqual(["US-TX-3"]);
    expect(body).not.toHaveProperty("networkVolumeId");
  });
  it("US-CA-2 (the deleted us volume's data centre) is any other data centre with r2", () => {
    expect(placements({ preset: "wan", weights_source: "r2", gpu_types: [H100], dc: "US-CA-2" })).toEqual([{ dc: "US-CA-2", gpu: H100 }]);
  });
  it("the volume in EU: region eu, the volume mounted in EU-RO-1", () => {
    const { spec } = launch({ preset: "h3-turbo", weights_source: "volume", gpu_types: [H100], region: "eu" });
    const pool = spec.pools[0]!;
    expect([pool.volume, pool.data_centers]).toEqual([true, undefined]);
    const pl = workerPlacements(spec, pool);
    expect(pl).toEqual([{ region: "eu", dc: "EU-RO-1", gpu: H100 }]);
    const body = workerCreatePayload("n", "img", pool, pl[0]!, {});
    expect([body.dataCenterIds, (body as any).networkVolumeId]).toEqual([["EU-RO-1"], "bz0ofkfx45"]);
  });
  it("fallbacks: gpu_types in the picked order, each data centre in order with every type; a last any = no dataCenterIds", () => {
    const { spec } = launch({ preset: "wan", weights_source: "r2", gpu_types: [H200, H100], data_centers: ["US-CA-2", "US-TX-3", "any"] });
    const pool = spec.pools[0]!;
    expect(pool.gpu_types).toEqual([H200, H100]);
    const pl = workerPlacements(spec, pool);
    expect(pl).toEqual([
      { dc: "US-CA-2", gpu: H200 },
      { dc: "US-CA-2", gpu: H100 },
      { dc: "US-TX-3", gpu: H200 },
      { dc: "US-TX-3", gpu: H100 },
      { gpu: H200 },
      { gpu: H100 },
    ]);
    const any = workerCreatePayload("n", "img", pool, pl[4]!, { FV_WEIGHTS_SOURCE: "r2" });
    expect(any).not.toHaveProperty("dataCenterIds");
    // No exact CUDA list: REST cannot name the 13.1 / 13.2 hosts; the boot's driver check keeps the CUDA 13 minimum.
    expect(any).not.toHaveProperty("allowedCudaVersions");
    expect(workerCreatePayload("n", "img", { ...pool, allowed_cuda: ["13.0"] }, pl[4]!, { FV_WEIGHTS_SOURCE: "r2" }).allowedCudaVersions).toEqual(["13.0"]);
  });
  it("none (fake engine) and a CPU pod without the volume: any data centre too", () => {
    const g = launch({ variant: "wan", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], weights_source: "none", gpu_types: [H100], data_centers: ["US-TX-3"] }).spec.pools[0]!;
    expect([g.volume, g.data_centers]).toEqual([false, ["US-TX-3"]]);
    expect(placements({ variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], compute: "CPU", cpu_flavors: ["cpu3c"], data_centers: ["US-TX-3", "any"] })).toEqual([{ dc: "US-TX-3", cpu: "cpu3c" }, { cpu: "cpu3c" }]);
  });
  it("the preset's default weights match the backend's: H3 / LTX the volume, wan r2 (no volume)", () => {
    expect(presetWeightsDefault("h3-turbo")).toBe("volume");
    expect(presetWeightsDefault("ltx")).toBe("volume");
    expect(presetWeightsDefault("wan")).toBe("r2");
    const wan = launch({ preset: "wan" }).spec.pools[0]!;
    expect([wan.volume, wan.weights_source]).toEqual([false, undefined]);
    expect(launch({ preset: "h3-turbo" }).spec.pools[0]!.volume).toBe(true);
  });
});

describe("the validator, the launch and the preflight agree (one rule: schemas.ts StandaloneLaunchZ)", () => {
  it("an r2 pod in any data centre passes", () => {
    for (const dcs of [["US-TX-3"], ["AP-JP-1", "EU-RO-1"], ["any"]]) {
      expect(refused({ preset: "wan", weights_source: "r2", data_centers: dcs }), dcs.join()).toBeNull();
      expect(validate("standalone-launch", { name: "anydc", preset: "wan", weights_source: "r2", data_centers: dcs }).ok).toBe(true);
    }
  });
  it("a volume pod outside the volume's data centre fails with a clear message", () => {
    expect(refused({ preset: "h3-turbo", weights_source: "volume", data_centers: ["US-TX-3"] })).toMatch(/^data_centers: US-TX-3 has no weights volume: weights_source r2 \(the R2 weights store\), hub \(approved\) or none runs in any data centre \(now: volume\); the volume needs EU only: eu \(EU-RO-1\)/);
    expect(refused({ preset: "h3-turbo", dc: "US-TX-3" })).toMatch(/^dc: US-TX-3 has no weights volume: .*\(now: unset, the volume\)/);
    expect(refused({ preset: "wan", weights_source: "r2", data_centers: ["US-TX-3"], volume: true })).toMatch(/^volume: US-TX-3 has no weights volume: volume off with weights_source r2/);
    expect(refused({ variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], compute: "CPU", volume: true, data_centers: ["US-TX-3"] })).toMatch(/^volume: US-TX-3 has no weights volume: a CPU pod there runs with volume off/);
    expect(validate("standalone-launch", { name: "anydc", preset: "h3-turbo", data_centers: ["US-TX-3"] }).ok).toBe(false);
  });
  it("any only last; dc or data_centers; region only with the volume's data centre", () => {
    expect(refused({ preset: "wan", weights_source: "r2", data_centers: ["any", "US-TX-3"] })).toMatch(/data_centers: "any" only last/);
    expect(refused({ preset: "wan", weights_source: "r2", dc: "US-TX-3", data_centers: ["US-TX-3"] })).toMatch(/dc or data_centers, not both/);
    expect(refused({ preset: "wan", weights_source: "r2", region: "eu", data_centers: ["US-TX-3"] })).toMatch(/region: region is the weights volume's/);
    expect(refused({ preset: "wan", weights_source: "r2", region: "eu", data_centers: ["EU-RO-1"] })).toBeNull();
    expect(refused({ provider: "gmi", provider_gpu: "container.h200.x1", preset: "wan", data_centers: ["US-TX-3"] })).toMatch(/data_centers: region \/ dc \/ data_centers are Runpod's/);
  });
  it("a cluster pool takes data_centers without the volume too (r2, none, CPU volume off), never with it", () => {
    const pool = (p: Record<string, unknown>) => ({ id: "p", count: 1, compute: "GPU", variant: "wan", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], ...p });
    const v1 = validate("pool", pool({ weights_source: "none", data_centers: ["US-TX-3", "any"] }));
    expect(v1.ok ? "" : JSON.stringify(v1.issues)).toBe("");
    expect(validate("pool", pool({ compute: "CPU", variant: "cpu", data_centers: ["US-TX-3"] })).ok).toBe(true);
    const bad = validate("pool", pool({ data_centers: ["US-TX-3"] }));
    expect(bad.ok ? "" : bad.issues.map((i) => i.message).join()).toMatch(/data centres outside the weights volume's need a pod without the volume/);
  });
});
