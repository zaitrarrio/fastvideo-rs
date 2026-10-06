import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_POLICIES } from "../../src/alerts";
import { cloudrift, CLOUDRIFT_API_VERSION, cloudriftTypeAllowed, deadlineOf, toInstance } from "../../src/cloudrift";
import { cloudriftDecisions, cloudriftOwner, collectCloudrift } from "../../src/collector-cloudrift";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { d1 } from "./d1shim";

const KEY = "crk_SECRET_0123456789";
const mkEnv = (over: Partial<Env> = {}): Env =>
  ({ DB: d1(), CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_x", CLOUDRIFT_API_KEY: KEY, CLOUDRIFT_API: "https://cr.test", ...over }) as unknown as Env;

const inst = (over: any = {}) => ({
  id: "i1",
  instance_name: "fv-gpucheck-1006",
  status: "Active",
  tags: ["fv", "fv-owner:fastvideo-rs", "fv-kind:gpucheck", "fv-deadline:1800000000"],
  host_address: "203.0.113.7",
  created_at: "2026-10-06T00:00:00Z",
  resource_info: { cost_per_hour: 139.36, instance_type: "rtxpro6000-11-50-500-1l.1", provider_name: "p" },
  gpus: [{ brand_short: "RTX PRO 6000", vram: 1, pci_device_id: 1, pci_vendor_id: 1 }],
  ...over,
});

/** A fetch mock: records calls, answers per path. */
function mockFetch(answers: Record<string, (data: any) => [number, any]>) {
  const calls: { path: string; body: any; headers: Record<string, string> }[] = [];
  const f = vi.fn(async (url: string, init: any) => {
    const path = new URL(url).pathname.replace(/^\/api\/v1\//, "");
    const body = JSON.parse(init.body);
    calls.push({ path, body, headers: init.headers });
    const [code, data] = (answers[path] || (() => [404, "no route"]))(body.data);
    return new Response(typeof data === "string" ? data : JSON.stringify({ version: body.version, data }), { status: code });
  });
  vi.stubGlobal("fetch", f);
  return calls;
}
afterEach(() => vi.unstubAllGlobals());

describe("CloudRift client", () => {
  it("posts {version, data} with X-API-Key and no bearer", async () => {
    // Live shape: cents, plus fields the spec does not list.
    const calls = mockFetch({ "account/info": () => [200, { balance: 1250, pending: 0.0, disputed: 0, dispute_fees: 0, current_cost_per_hour: null }] });
    expect(await cloudrift.balance(mkEnv())).toBe(12.5);
    expect(calls[0]!.body.version).toBe(CLOUDRIFT_API_VERSION);
    expect(calls[0]!.headers["x-api-key"]).toBe(KEY);
    expect(JSON.stringify(calls[0]!.headers)).not.toMatch(/authorization/i);
  });
  it("the catalog is public, prices are cents, only 1-GPU variants of the brand", async () => {
    const calls = mockFetch({
      "instance-types/list": () => [200, { instance_types: [
        { name: "rtxpro6000-a", brand_short: "RTX PRO 6000", variants: [
          { name: "rtxpro6000-a.1", gpu_count: 1, cost_per_hour: 139.36, available_nodes: 2, available_nodes_per_dc: { "us-x": 2, "eu-y": 0 } },
          { name: "rtxpro6000-a.2", gpu_count: 2, cost_per_hour: 278.72, available_nodes: 1 },
        ] },
        { name: "rtxpro6000-b", brand_short: "RTX PRO 6000", variants: [{ name: "rtxpro6000-b.1", gpu_count: 1, cost_per_hour: 134.16, available_nodes: 0 }] },
        { name: "rtx49", brand_short: "RTX 4090", variants: [{ name: "rtx49.1", gpu_count: 1, cost_per_hour: 39 }] },
      ] }],
    });
    const offers = await cloudrift.price(mkEnv({ CLOUDRIFT_API_KEY: undefined }), "RTX PRO 6000");
    expect(offers).toEqual([
      { variant: "rtxpro6000-b.1", usd_per_hr: 1.3416, free_nodes: 0, datacenters: [] },
      { variant: "rtxpro6000-a.1", usd_per_hr: 1.3936, free_nodes: 2, datacenters: ["us-x"] },
    ]);
    expect(calls[0]!.headers["x-api-key"]).toBeUndefined();
  });
  it("refuses GPUs outside the allow-list (RTX PRO 6000, RTX 5090) before any call", async () => {
    const calls = mockFetch({});
    for (const g of ["V100 SXM2", "RTX 4090", "v100-6-52-400-generic.1", "rtx49-7c-kn.1"])
      await expect(cloudrift.price(mkEnv(), g)).rejects.toThrow(/not allowed: only RTX PRO 6000, RTX 5090/);
    expect(calls).toHaveLength(0);
    expect(cloudriftTypeAllowed("rtx59-16c-nr.1")).toBe(true);
    expect(cloudriftTypeAllowed("rtxpro6000-12-100-1500-nr.1")).toBe(true);
    expect(cloudriftTypeAllowed("rtx49-7c-kn.1")).toBe(false);
    expect(cloudriftTypeAllowed(null)).toBe(false);
  });
  it("lists live rentals without asking for credentials; parses tags", async () => {
    const calls = mockFetch({ "instances/list": () => [200, { instances: [inst(), inst({ id: "x", tags: ["other"], instance_name: null })] }] });
    const l = await cloudrift.instances(mkEnv());
    expect(calls[0]!.body.data.mask.with_credentials).toBeUndefined();
    expect(calls[0]!.body.data.selector.ByStatus.statuses).toContain("Failed");
    expect(l[0]!.costPerHr).toBeCloseTo(1.3936);
    expect(l[0]).toMatchObject({ id: "i1", ours: true, gpu: "RTX PRO 6000", deadlineMs: 1_800_000_000_000, host: "203.0.113.7" });
    expect(l[1]).toMatchObject({ id: "x", ours: false, name: "x", deadlineMs: null });
    expect(toInstance(inst({ resource_info: { cost_per_hour: 25.0 } })).costPerHr).toBeCloseTo(0.25);
    expect(toInstance(inst({ resource_info: { cost_per_hour: 1.3936 } }), "usd").costPerHr).toBeCloseTo(1.3936);
  });
  it("errors are scrubbed of the key; a 404 terminate counts as done", async () => {
    mockFetch({ "account/info": () => [401, `bad key ${KEY}`], "instances/terminate": () => [404, "gone"] });
    await expect(cloudrift.balance(mkEnv())).rejects.toThrow(/\[redacted\]/);
    await expect(cloudrift.balance(mkEnv())).rejects.not.toThrow(new RegExp(KEY));
    expect(await cloudrift.terminate(mkEnv(), "i1")).toBe(true);
  });
  it("deadline tags", () => {
    expect(deadlineOf(["fv-deadline:1800000000"])).toBe(1_800_000_000_000);
    expect(deadlineOf(["fv-deadline:soon", "x"])).toBeNull();
  });
});

describe("CloudRift backstops (pure)", () => {
  const pol = { ...DEFAULT_POLICIES };
  const t = 1_800_000_100_000;
  it("terminates our rentals past their deadline, never someone else's", () => {
    const d = cloudriftDecisions([toInstance(inst()), toInstance(inst({ id: "f", tags: ["fv-deadline:1"] }))], 50, 8, pol, t);
    expect(d.terminate).toEqual([{ id: "i1", why: "deadline" }]);
    expect(d.alerts.map((a) => a.kind)).toEqual(["cloudrift_deadline"]);
  });
  it("dismisses our failed rentals", () => {
    const d = cloudriftDecisions([toInstance(inst({ status: "Failed", tags: ["fv-owner:fastvideo-rs"], failure: { user_message: "bad image" } }))], 50, 8, pol, t);
    expect(d.terminate).toEqual([{ id: "i1", why: "failed" }]);
    expect(d.alerts[0]!.message).toContain("bad image");
  });
  it("below the floor: alert and terminate ours (stop_on_floor); margin warns", () => {
    const ours = toInstance(inst({ tags: ["fv-owner:fastvideo-rs"] }));
    const theirs = toInstance(inst({ id: "f", tags: [] }));
    const d = cloudriftDecisions([ours, theirs], 7.5, 8, pol, t);
    expect(d.terminate).toEqual([{ id: "i1", why: "balance floor" }]);
    expect(d.alerts[0]!.severity).toBe("critical");
    expect(cloudriftDecisions([ours], 7.5, 8, { ...pol, stop_on_floor: false }, t).terminate).toEqual([]);
    expect(cloudriftDecisions([ours], 12, 8, pol, t).alerts[0]!.kind).toBe("cloudrift_balance_margin");
  });
  it("terminates our live rental on a type outside the allow-list; never someone else's", () => {
    const v100 = { resource_info: { cost_per_hour: 25, instance_type: "v100-6-52-400-generic.1" } };
    const d = cloudriftDecisions([toInstance(inst(v100)), toInstance(inst({ ...v100, id: "f", tags: [] }))], 50, 8, pol, t - 200_000_000);
    expect(d.terminate).toEqual([{ id: "i1", why: "type not allowed" }]);
    expect(d.alerts.map((a) => a.kind)).toEqual(["cloudrift_type"]);
    expect(cloudriftDecisions([toInstance(inst({ resource_info: { cost_per_hour: 62.4, instance_type: "rtx59-16c-nr.1" } }))], 50, 8, pol, t - 200_000_000).terminate).toEqual([]);
  });
  it("owners", () => {
    expect(cloudriftOwner(toInstance(inst()))).toBe("cloudrift:gpucheck");
    expect(cloudriftOwner(toInstance(inst({ tags: [] })))).toBe("external:cloudrift");
  });
});

describe("collectCloudrift", () => {
  it("off without a key", async () => {
    const r = await collectCloudrift(mkEnv({ CLOUDRIFT_API_KEY: undefined }), 1, 60_000, DEFAULT_POLICIES);
    expect(r.enabled).toBe(false);
  });
  it("rows (provider cloudrift), costs, idle, deadline termination, gone", async () => {
    const env = mkEnv();
    const terminated: string[] = [];
    let list = [inst(), inst({ id: "i2", tags: ["fv-owner:fastvideo-rs", "fv-deadline:1700000000"], instance_name: "fv-serve-smoke-1" }), inst({ id: "ext", tags: [] })];
    mockFetch({
      "account/info": () => [200, { balance: 4000 }],
      "instances/list": () => [200, { instances: list }],
      "instances/metrics": (d) => [200, { metrics: d.selector.ById.map((id: string) => ({ instance_id: id, gpus: [{ gpu_utilization_percent: id === "ext" ? 1 : 60 }] })) }],
      "instances/terminate": (d) => (terminated.push(...d.selector.ById), [201, { terminated: [] }]),
    });
    const t = 1_750_000_000_000;
    const r = await collectCloudrift(env, t, 60_000, { ...DEFAULT_POLICIES, idle_min: 0 });
    expect(r).toMatchObject({ ok: true, balance: 40, instances: 3, running: 3 });
    expect(terminated).toEqual(["i2"]);
    expect(r.alerts.map((a) => a.key).sort()).toEqual(["cloudrift_deadline:i2", "pod_idle:ext"]);
    const rows = (await env.DB.prepare("SELECT pod_id, owner, provider, desired_status, cost_per_hr, gpu_util FROM pods ORDER BY pod_id").all<any>()).results;
    expect(rows.map((x: any) => [x.pod_id, x.owner, x.provider, x.desired_status])).toEqual([
      ["ext", "external:cloudrift", "cloudrift", "RUNNING"],
      ["i1", "cloudrift:gpucheck", "cloudrift", "RUNNING"],
      ["i2", "cloudrift:fv", "cloudrift", "RUNNING"],
    ]);
    const cost = await env.DB.prepare("SELECT SUM(usd) AS usd FROM cost_daily").first<{ usd: number }>();
    expect(cost!.usd).toBeCloseTo((3 * 1.3936) / 60);
    // Next minute: i2 is gone from the listing.
    list = list.filter((x) => x.id !== "i2");
    await collectCloudrift(env, t + 60_000, 60_000, DEFAULT_POLICIES);
    const gone = await env.DB.prepare("SELECT gone_at FROM pods WHERE pod_id = 'i2'").first<{ gone_at: number }>();
    expect(gone!.gone_at).toBe(t + 60_000);
  });
  it("an API failure is an alert and resolves nothing", async () => {
    mockFetch({ "account/info": () => [503, "down"] });
    const r = await collectCloudrift(mkEnv(), 1, 60_000, DEFAULT_POLICIES);
    expect(r.ok).toBe(false);
    expect(r.kinds).toEqual([]);
    expect(r.alerts[0]!.kind).toBe("cloudrift_api");
  });
});
