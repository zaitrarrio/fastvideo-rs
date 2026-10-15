// The launch forms' lazy provider details (owner 2026-10-12: "we should not load provider details / prices until a
// provider is selected"): GET /api/v1/schemas/dynamic?live=0 calls no upstream API, ?provider=<p> calls only that
// provider's (cached briefly, one call in flight), /api/v1/providers?provider=<p> reads only that one, and the launch
// validator's ?live=0 checks the request without any provider call.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mintApiToken } from "../../src/auth";
import { resetBrevTypes } from "../../src/brev";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { clearTtlCache, ttlCached } from "../../src/ttlcache";
import { d1 } from "./d1shim";

const realFetch = globalThis.fetch;
type Call = { host: string; url: string };
let calls: Call[] = [];
/** Upstream stand-ins by host: Runpod GraphQL, Brev's instance types, GMI's products; anything else fails the test. */
function upstream(o: { runpodDelayMs?: number } = {}) {
  calls = [];
  globalThis.fetch = (async (input: any, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.url;
    const host = new URL(url).host;
    calls.push({ host, url });
    if (host === "api.runpod.io") {
      if (o.runpodDelayMs) await new Promise((r) => setTimeout(r, o.runpodDelayMs));
      const q = JSON.parse(String(init?.body || "{}")).query || "";
      if (q.includes("displayName"))
        return Response.json({ data: { gpuTypes: [{ id: "NVIDIA H100 80GB HBM3", displayName: "H100 SXM", memoryInGb: 80, securePrice: 2.99, communityPrice: 0, lowestPrice: { stockStatus: "High" } }] } });
      return Response.json({ data: {} });
    }
    if (host === "brev.test" && url.includes("/alltypesavailable/"))
      return Response.json({ allInstanceTypes: [{ type: "g5.xlarge", cloud_cred_id: "cred-aws-1", stoppable: true, base_price: { currency: "USD", amount: "1.006" }, location: "us-east-1", is_available: true }] });
    if (host === "gmi.test" && url.includes("/containers/products")) return Response.json({ products: [{ name: "container.h200.x1", idc: "us-denver-1", type: "Container", price: 320, valid: true, gpuModel: "H200" }] });
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
    GHCR: "https://ghcr.test",
    GMI_API_KEY: "gmi_SECRET_0123456789",
    GMI_API: "https://gmi.test/api",
    GMI_PRODUCTS: "container.h200.x1",
    GMI_DEFAULT_IDC: "us-denver-1",
    GMI_BUDGET_USD: "20",
    BREV_API_TOKEN: "brev_SECRET_0123456789",
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "g5.xlarge",
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;
const ctx = { waitUntil() {}, passThroughOnException() {} } as any;
async function client(env: Env) {
  const auth = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
  return async (path: string, init: { method?: string; body?: unknown } = {}) => {
    const r = await _app.fetch(new Request(`https://ctl.test${path}`, { method: init.method || "GET", headers: { ...auth, "content-type": "application/json" }, body: init.body === undefined ? undefined : JSON.stringify(init.body) }), env, ctx);
    return { status: r.status, j: (await r.json()) as any };
  };
}
const hosts = () => [...new Set(calls.map((c) => c.host))].sort();

beforeEach(() => (clearTtlCache(), resetBrevTypes()));
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.useRealTimers();
});

describe("GET /api/v1/schemas/dynamic: the launch form's open and per-provider details", () => {
  it("?live=0 calls no upstream API: presets, regions and the provider list (on / off from the config) only", async () => {
    upstream();
    const env = mkEnv({ GMI_PRODUCTS: "" });
    const api = await client(env);
    const r = await api("/api/v1/schemas/dynamic?live=0");
    expect(r.status).toBe(200);
    expect(calls).toEqual([]);
    expect(r.j.live).toBe(false);
    expect(r.j.gpu_types).toEqual([]);
    expect(r.j.pool_presets.length).toBeGreaterThan(0);
    expect(r.j.regions.length).toBeGreaterThan(0);
    expect(r.j.providers.map((p: any) => p.id)).toEqual(["runpod", "gmi", "brev"]);
    expect(r.j.providers.find((p: any) => p.id === "gmi").detail).toMatch(/^off: GMI_PRODUCTS is not set/);
    expect(r.j.providers.find((p: any) => p.id === "brev").detail).toBe("NVIDIA Brev");
    // The config's Brev types, without a live price.
    expect(r.j.provider_gpus).toEqual([{ id: "g5.xlarge", detail: "NVIDIA Brev", provider: "brev" }]);
  });

  it("the default (no ?live=0) is still the full view (the JSON editor, the CLI): Runpod and Brev are asked", async () => {
    upstream();
    const api = await client(mkEnv());
    const r = await api("/api/v1/schemas/dynamic");
    expect(r.status).toBe(200);
    expect(r.j.live).toBe(true);
    expect(r.j.gpu_types[0].id).toBe("NVIDIA H100 80GB HBM3");
    expect(hosts()).toEqual(expect.arrayContaining(["api.runpod.io", "brev.test"]));
  });

  it("?provider=brev asks Brev only: instance types with the live price, parked VMs, the budget", async () => {
    upstream();
    const env = mkEnv();
    const api = await client(env);
    const r = await api("/api/v1/schemas/dynamic?provider=brev");
    expect(r.status).toBe(200);
    expect(hosts()).toEqual(["brev.test"]);
    expect(r.j).toMatchObject({ provider: "brev", enabled: true, live_ok: true, parked: [], budget: { budget_usd: 10, month_usd: 0 } });
    expect(r.j.provider_gpus).toHaveLength(1);
    expect(r.j.provider_gpus[0]).toMatchObject({ id: "g5.xlarge", provider: "brev", stoppable: true });
    expect(r.j.provider_gpus[0].detail).toMatch(/\$1\.01\/hr · stoppable/);
    // The type list is cached (1 h): a second pick asks Brev nothing.
    calls = [];
    expect((await api("/api/v1/schemas/dynamic?provider=brev")).status).toBe(200);
    expect(calls).toEqual([]);
  });

  it("?provider=gmi asks GMI only (products cached 60 s); ?provider=runpod asks Runpod only (cached 5 min)", async () => {
    upstream();
    const api = await client(mkEnv());
    const g = await api("/api/v1/schemas/dynamic?provider=gmi");
    expect(hosts()).toEqual(["gmi.test"]);
    expect(g.j).toMatchObject({ provider: "gmi", enabled: true, live_ok: true, provider_regions: [{ id: "us-denver-1" }], budget: { budget_usd: 20 } });
    expect(g.j.provider_gpus).toEqual([{ id: "container.h200.x1", detail: "GMI Cloud · $3.20/hr · available", provider: "gmi", usd_per_hr: 3.2, in_stock: true }]);
    calls = [];
    await api("/api/v1/schemas/dynamic?provider=gmi");
    expect(calls).toEqual([]);
    const rp = await api("/api/v1/schemas/dynamic?provider=runpod");
    expect(hosts()).toEqual(["api.runpod.io"]);
    expect(rp.j).toMatchObject({ provider: "runpod", enabled: true, live_ok: true });
    expect(rp.j.gpu_types[0]).toMatchObject({ id: "NVIDIA H100 80GB HBM3", secure_price: 2.99, stock: "High" });
    calls = [];
    await api("/api/v1/schemas/dynamic?provider=runpod");
    expect(calls).toEqual([]);
  });

  it("an off provider is answered from the config without a call; an unknown one is a 404", async () => {
    upstream();
    const api = await client(mkEnv({ BREV_API_TOKEN: undefined }));
    const r = await api("/api/v1/schemas/dynamic?provider=brev");
    expect(r.status).toBe(200);
    expect(calls).toEqual([]);
    expect(r.j).toMatchObject({ provider: "brev", enabled: false, live_ok: false, provider_gpus: [{ id: "g5.xlarge", provider: "brev" }] });
    expect(r.j.reason).toMatch(/NVIDIA Brev is off/);
    expect((await api("/api/v1/schemas/dynamic?provider=cloudrift")).status).toBe(404);
  });

  it("an upstream failure is reported (live_ok false), not cached: the next pick asks again", async () => {
    upstream();
    const env = mkEnv({ GMI_API: "https://gmi-down.test/api" });
    const api = await client(env);
    const r = await api("/api/v1/schemas/dynamic?provider=gmi");
    expect(r.status).toBe(200);
    expect(r.j.live_ok).toBe(false);
    expect(r.j.provider_gpus[0].detail).toBe("GMI Cloud · price unknown");
    const n = calls.length;
    await api("/api/v1/schemas/dynamic?provider=gmi");
    expect(calls.length).toBeGreaterThan(n);
  });

  it("concurrent asks share one Runpod call (the details and the validator at once)", async () => {
    upstream({ runpodDelayMs: 30 });
    const api = await client(mkEnv());
    const rs = await Promise.all([api("/api/v1/schemas/dynamic?provider=runpod"), api("/api/v1/schemas/dynamic?provider=runpod"), api("/api/v1/schemas/dynamic?provider=runpod")]);
    expect(rs.every((r) => r.j.gpu_types.length === 1)).toBe(true);
    expect(calls.filter((c) => c.host === "api.runpod.io")).toHaveLength(1);
  });

  it("?images=1: the commits with images (GHCR) on their own", async () => {
    upstream();
    const api = await client(mkEnv());
    const r = await api("/api/v1/schemas/dynamic?images=1");
    expect(r.status).toBe(200);
    expect(Array.isArray(r.j.shas)).toBe(true);
    expect(hosts().every((h) => h.startsWith("ghcr"))).toBe(true);
  });
});

describe("GET /api/v1/providers?provider= and the launch validator's ?live=0", () => {
  it("?provider=brev returns only Brev's view; an unknown id is a 404; no filter: all four", async () => {
    upstream();
    const api = await client(mkEnv());
    const one = await api("/api/v1/providers?provider=brev");
    expect(one.j.providers.map((p: any) => p.id)).toEqual(["brev"]);
    expect(one.j.providers[0]).toMatchObject({ enabled: true, budget_usd: 10, launch: true });
    expect((await api("/api/v1/providers?provider=nope")).status).toBe(404);
    expect((await api("/api/v1/providers")).j.providers.map((p: any) => p.id)).toEqual(["runpod", "cloudrift", "gmi", "brev"]);
    expect(calls).toEqual([]);
  });

  it("POST /api/v1/standalone/validate?live=0 checks the request (rules, name) without a provider call; without it, Runpod is asked", async () => {
    upstream();
    const env = mkEnv();
    const api = await client(env);
    const body = { name: "lazy-a", preset: "h3-turbo", channel: "stable", region: "eu", deadline_min: 60, env: {} };
    const r = await api("/api/v1/standalone/validate?live=0", { method: "POST", body });
    expect(r.status).toBe(200);
    expect(r.j.ok).toBe(true);
    expect(calls).toEqual([]);
    const bad = await api("/api/v1/standalone/validate?live=0", { method: "POST", body: { ...body, deadline_min: 1 } });
    expect(bad.j.issues.map((i: any) => i.path[0])).toContain("deadline_min");
    expect(calls).toEqual([]);
    await api("/api/v1/standalone/validate", { method: "POST", body: { ...body, provider: "runpod" } });
    expect(hosts()).toContain("api.runpod.io");
    expect(hosts()).not.toContain("brev.test");
    expect(hosts()).not.toContain("gmi.test");
  });

  it("a Brev launch's live check asks no Runpod API (its catalog is only for Runpod GPU pools)", async () => {
    upstream();
    const api = await client(mkEnv());
    const r = await api("/api/v1/standalone/validate", { method: "POST", body: { name: "lazy-b", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], channel: "stable", deadline_min: 60, env: {}, provider: "brev", provider_gpu: "g5.xlarge", weights_source: "none" } });
    expect(r.status).toBe(200);
    expect(r.j.issues).toEqual([]);
    expect(hosts()).toContain("brev.test");
    expect(hosts()).not.toContain("api.runpod.io");
    expect(hosts()).not.toContain("gmi.test");
  });
});

describe("ttlCached", () => {
  it("keeps a value for its TTL, shares one call in flight, never caches a failure", async () => {
    let n = 0;
    const f = async () => ++n;
    expect(await Promise.all([ttlCached("k", 1000, f), ttlCached("k", 1000, f)])).toEqual([1, 1]);
    expect(await ttlCached("k", 1000, f)).toBe(1);
    expect(await ttlCached("k", 0, f)).toBe(2);
    await expect(ttlCached("bad", 1000, async () => Promise.reject(new Error("down")))).rejects.toThrow("down");
    expect(await ttlCached("bad", 1000, async () => "up")).toBe("up");
    clearTtlCache("k");
    expect(await ttlCached("k", 1000, f)).toBe(3);
  });
});
