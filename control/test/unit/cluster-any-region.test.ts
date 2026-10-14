// A cluster in any Runpod data centre, and any provider per pool (owner 2026-10-14: "the cluster page is showing me
// only eu region. just like standalone pods I should be able to create a cluster in any Runpod region if using R2 or
// hub. I should also be able to create a cluster with any provider."): the cluster-level data_centers, which the pools
// without the weights volume inherit; the volume pools stay in the regions; old specs read and place as before; GMI /
// Brev pools next to Runpod ones; the editor's outlook.
import { afterEach, describe, expect, it, vi } from "vitest";
import { availability, checkSpec, clearStockCache } from "../../src/cluster/editor";
import { placementWhere, poolDataCenters, poolWithoutVolume, workerCreatePayload, workerPlacements } from "../../src/cluster/payloads";
import { defaultSpec, normalizeSpec, type ClusterSpec } from "../../src/cluster/spec";
import type { Env } from "../../src/env";
import { providerIssues } from "../../src/providers";
import { presetWeightsDefault } from "../../src/standalone";
import { d1 } from "./d1shim";

const RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
const H100 = "NVIDIA H100 80GB HBM3";
const spec = (x: Record<string, unknown>) => normalizeSpec({ name: "anyrg", ...x });
const refused = (x: Record<string, unknown>) => {
  try {
    spec(x);
    return [];
  } catch (e) {
    return ((e as any).extra?.issues as { path: (string | number)[]; message: string }[]).map((i) => `${i.path.join(".")}: ${i.message}`);
  }
};
const pool = (s: ClusterSpec, id: string) => s.pools.find((p) => p.id === id)!;
const gmiPool = { id: "g", variant: "cpu", count: 1, compute: "GPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], provider: "gmi", provider_gpu: "container.h200.x1" };
const brevPool = { ...gmiPool, id: "b", provider: "brev", provider_gpu: "g5.xlarge" };

describe("the spec: a cluster-level data_centers, old specs unchanged", () => {
  it("an old stored spec (no data_centers): regions [eu], every pool placed exactly as before", () => {
    const s = spec({ template: "standard" });
    expect(s.regions).toEqual(["eu"]);
    expect(s.data_centers).toBeUndefined();
    // The volume pools and wan (r2 by default, no volume) alike: EU-RO-1 in region eu.
    for (const p of s.pools) expect(workerPlacements(s, p)).toEqual([{ region: "eu", dc: "EU-RO-1", gpu: RTX }]);
    // regions left out or empty: the default [eu], as before.
    expect(spec({ template: "standard", regions: [] }).regions).toEqual(["eu"]);
  });
  it("data_centers: any Runpod data centre, in order, a last \"any\"; refused: any not last, an unknown one", () => {
    expect(spec({ template: "wan", data_centers: ["US-TX-3", "EU-SE-1", "any"] }).data_centers).toEqual(["US-TX-3", "EU-SE-1", "any"]);
    expect(refused({ template: "wan", data_centers: ["any", "US-TX-3"] }).join("; ")).toMatch(/^data_centers: "any" only last/);
    expect(refused({ template: "wan", data_centers: ["MARS-1"] }).some((m) => m.startsWith("data_centers.0:"))).toBe(true);
    expect(refused({ template: "wan", data_centers: ["US-TX-3", "US-TX-3"] }).some((m) => m.startsWith("data_centers:"))).toBe(true);
  });
  it("a volume pool next to the cluster's data_centers is valid: it stays in its region (only a warning, below)", () => {
    expect(refused({ template: "standard", data_centers: ["US-TX-3"] })).toEqual([]);
    // A region without a volume is still refused (us), whatever data_centers says.
    expect(refused({ template: "wan", regions: ["us"], data_centers: ["US-TX-3"] }).some((m) => m.startsWith("regions.0:"))).toBe(true);
  });
});

describe("placement: inheritance, order, any", () => {
  it("pools without the volume inherit the cluster's data centres; the volume pools keep EU-RO-1 with the volume", () => {
    const s = spec({ template: "standard", data_centers: ["US-TX-3", "any"] });
    // wan: weights_source unset, r2 by default (its trees are not on the 500 GB EU volume).
    expect(presetWeightsDefault("wan")).toBe("r2");
    const wan = pool(s, "wan");
    expect([wan.weights_source, poolWithoutVolume(wan)]).toEqual([undefined, true]);
    expect(workerPlacements(s, wan)).toEqual([{ dc: "US-TX-3", gpu: RTX }, { gpu: RTX }]);
    const body = workerCreatePayload("n", "img", wan, workerPlacements(s, wan)[1]!, { FV_WEIGHTS_SOURCE: "r2" });
    expect(body).not.toHaveProperty("dataCenterIds");
    expect(body).not.toHaveProperty("networkVolumeId");
    for (const id of ["h3-turbo", "h3-max", "ltx"]) {
      const p = pool(s, id);
      expect(poolWithoutVolume(p)).toBe(false);
      const pl = workerPlacements(s, p);
      expect(pl).toEqual([{ region: "eu", dc: "EU-RO-1", gpu: RTX }]);
      expect((workerCreatePayload("n", "img", p, pl[0]!, {}) as any).networkVolumeId).toBe("bz0ofkfx45");
    }
  });
  it("order: each data centre in order with every GPU type in order; the pool's own data centres win", () => {
    const s = spec({ data_centers: ["US-CA-2", "US-TX-3"], pools: [{ id: "wan", gpu_types: ["NVIDIA H200", H100] }, { id: "sfwan", weights_source: "r2", gpu_types: [H100], data_centers: ["EU-SE-1"] }] });
    expect(workerPlacements(s, pool(s, "wan"))).toEqual([
      { dc: "US-CA-2", gpu: "NVIDIA H200" },
      { dc: "US-CA-2", gpu: H100 },
      { dc: "US-TX-3", gpu: "NVIDIA H200" },
      { dc: "US-TX-3", gpu: H100 },
    ]);
    expect(workerPlacements(s, pool(s, "sfwan"))).toEqual([{ dc: "EU-SE-1", gpu: H100 }]);
    expect(poolDataCenters(s, pool(s, "sfwan"))).toEqual(["EU-SE-1"]);
  });
  it("hub and none, and a CPU pool with volume off, inherit too; a volume:true pool does not", () => {
    const s = spec({
      data_centers: ["US-TX-3"],
      pools: [
        { id: "ltx", weights_source: "hub", hub_download_approved: true, volume: false },
        { id: "fakegpu", variant: "wan", count: 1, compute: "GPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], weights_source: "none" },
        { ...defaultSpec("t", "tiny-cpu").pools[0]! },
        { id: "wan", volume: true },
      ],
    });
    expect(workerPlacements(s, pool(s, "ltx"))).toEqual([{ dc: "US-TX-3", gpu: RTX }]);
    expect(workerPlacements(s, pool(s, "fakegpu"))).toEqual([{ dc: "US-TX-3", gpu: RTX }]);
    expect(workerPlacements(s, pool(s, "fake"))).toEqual(["cpu3c", "cpu5c", "cpu3g"].map((cpu) => ({ dc: "US-TX-3", cpu })));
    expect(workerPlacements(s, pool(s, "wan"))).toEqual([{ region: "eu", dc: "EU-RO-1", gpu: RTX }]);
  });
  it("in words (the editor's outlook)", () => {
    const s = spec({ data_centers: ["US-TX-3", "any"], pools: [{ id: "wan" }, { id: "h3-turbo" }, gmiPool] });
    expect(s.pools.map((p) => placementWhere(s, p))).toEqual(["US-TX-3, any data centre", "EU-RO-1 (the eu weights volume)", "gmi"]);
  });
});

describe("any provider: GMI / Brev pools next to Runpod ones, or alone", () => {
  const env = (over: Record<string, unknown> = {}) =>
    ({
      DB: d1(),
      RUNPOD_API_KEY: "rpa_x",
      RUNPOD_GRAPHQL: "https://rp.test/graphql",
      PUBLIC_URL: "https://fvc.test",
      GMI_API_KEY: "gmi_SECRET_0123456789",
      GMI_API: "https://gmi.test/api",
      GMI_PRODUCTS: "container.h200.x1",
      BREV_API_TOKEN: "brev_SECRET_0123456789",
      BREV_API_URL: "https://brev.test",
      BREV_ORG_ID: "org1",
      BREV_INSTANCE_TYPES: "g5.xlarge",
      BREV_PRICES: '{"g5.xlarge": 1.25}',
      ...over,
    }) as unknown as Env;
  afterEach(() => (vi.unstubAllGlobals(), clearStockCache()));
  it("a mixed cluster and an all-GMI / all-Brev one are valid; the cluster's data centres never apply to them", () => {
    const mixed = spec({ data_centers: ["US-TX-3"], pools: [{ id: "wan" }, gmiPool, brevPool] });
    expect(poolDataCenters(mixed, pool(mixed, "g"))).toBeNull();
    expect(placementWhere(mixed, pool(mixed, "b"))).toBe("brev");
    expect(refused({ pools: [gmiPool] })).toEqual([]);
    expect(refused({ pools: [brevPool], data_centers: ["any"] })).toEqual([]);
    // The nominal regions default stays (no Runpod pool uses it); nothing blocks a cluster without a Runpod pool.
    expect(spec({ pools: [gmiPool] }).regions).toEqual(["eu"]);
  });
  it("the owner gates hold: a provider without keys is off, a Hub download needs the approval", () => {
    const s = spec({ data_centers: ["US-TX-3"], pools: [gmiPool, { id: "ltx", weights_source: "hub", volume: false }] });
    const is = providerIssues(env({ GMI_API_KEY: undefined }), s).map((i) => `${i.path.join(".")}: ${i.message}`);
    expect(is.some((m) => /^pools\.0\.provider: GMI Cloud is off/.test(m))).toBe(true);
    expect(is.some((m) => /^pools\.1\.hub_download_approved: .*owner's approval/.test(m))).toBe(true);
    const approved = spec({ data_centers: ["US-TX-3"], pools: [{ id: "ltx", weights_source: "hub", volume: false, hub_download_approved: true }] });
    expect(providerIssues(env(), approved).map((i) => i.message).join("; ")).toMatch(/not approved on this fv-control.*FV_HUB_DOWNLOADS_APPROVED/);
  });
  it("the editor's outlook: a volume pool is told it stays in EU-RO-1; stock asked where the pools may land", async () => {
    let query = "";
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => {
      const q = JSON.parse(String(init.body)).query as string;
      if (q.includes("lowestPrice")) query = q;
      const data: Record<string, unknown> = {};
      for (const m of q.matchAll(/(g\d+): gpuTypes/g)) data[m[1]!] = [{ id: "x", securePrice: 1.89, lowestPrice: { stockStatus: null, maxUnreservedGpuCount: 0 } }];
      return new Response(JSON.stringify({ data }), { status: 200 });
    });
    const e = env();
    const doc = { ...defaultSpec("anyrg"), data_centers: ["US-TX-3", "any"], pools: [{ id: "wan", count: 1 }, { id: "h3-turbo", count: 1 }, gmiPool] };
    const r = await checkSpec(e, doc);
    const warn = r.warnings.map((w) => `${w.path.join(".")}: ${w.message}`);
    expect(warn).toContain("pools.1.weights_source: h3-turbo mounts the weights volume, so it runs in EU-RO-1 (the eu weights volume), not the cluster's data_centers (weights_source r2, hub or none runs there)");
    expect(warn.some((m) => m.startsWith("pools.0.weights_source: h3") || m.startsWith("pools.2.weights_source: g "))).toBe(false);
    // The wan pool ends with "any": no "no stock" warning for it (any data centre may have stock).
    expect(warn.some((m) => m.startsWith("pools.0.gpu_types:"))).toBe(false);
    const a = await availability(e, normalizeSpec(doc));
    expect(query).toContain('dataCenterId: "US-TX-3"');
    expect(query).toContain('dataCenterId: "EU-RO-1"');
    expect(a.pools.map((p) => [p.pool, p.status])).toEqual([["wan", "unknown"], ["h3-turbo", "none"], ["g", "ok"]]);
    expect(a.pools[0]!.hint).toBe("no stock reported in the named data centres: RTX PRO 6000 in US-TX-3: none (0 free); then any data centre (RTX PRO 6000): Runpod places it");
  });
});
