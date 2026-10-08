// GMI Cloud / NVIDIA Brev providers (src/providers.ts, gmi.ts, brev.ts,
// collector-providers.ts; docs/serve/deploy-gmi-brev.md): the schema rules,
// the pod env and start command, the weights plan, the off / approval
// refusals, the budget, the clients against a fetch mock.
import { afterEach, describe, expect, it, vi } from "vitest";
import { brev, brevPrice, toWorkspace } from "../../src/brev";
import { budgetDecision } from "../../src/collector-providers";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { gmi, toContainer } from "../../src/gmi";
import { brevStartup, budgetCheck, CLOUDFLARED, endpointUrlOk, isOtherPod, podKey, poolTrees, PROVIDER_BOOT, providerEnv, providerImpl, providerIssues, splitKey, weightsPlan } from "../../src/providers";
import { validate } from "../../src/schemas";
import { standaloneSpec } from "../../src/standalone";
import { presetPool } from "../../src/cluster/spec";
import { HUB_TREES } from "../../src/weights-sources";
import { d1 } from "./d1shim";

const GMI_KEY = "gmi_SECRET_0123456789";
const BREV_TOKEN = "brev_SECRET_0123456789";
const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_x",
    PUBLIC_URL: "https://fvc.test",
    GMI_API_KEY: GMI_KEY,
    GMI_API: "https://gmi.test/api",
    GMI_PRODUCTS: "container.h200.x1",
    GMI_BUDGET_USD: "20",
    BREV_API_TOKEN: BREV_TOKEN,
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "g5.xlarge",
    BREV_PRICES: '{"g5.xlarge": 1.25}',
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;
const fakePool = (over: any = {}) => ({ id: "pod", variant: "cpu", count: 1, compute: "GPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], provider: "gmi", provider_gpu: "container.h200.x1", ...over });

afterEach(() => vi.unstubAllGlobals());

describe("pod keys", () => {
  it("gmi:/brev: prefixes; Runpod ids have none", () => {
    expect(podKey("gmi", "fv-pod-a-1")).toBe("gmi:fv-pod-a-1");
    expect(isOtherPod("brev:fv-pod-a-1")).toBe(true);
    expect(isOtherPod("mp0abc123")).toBe(false);
    expect(splitKey("gmi:fv-ctl-c-p-1")).toEqual({ provider: "gmi", name: "fv-ctl-c-p-1" });
  });
});

describe("schema rules", () => {
  const pool = (over: any) => validate("pool", fakePool(over));
  it("a GMI / Brev pool takes provider_gpu, not Runpod's fields", () => {
    expect(pool({}).ok).toBe(true);
    const bad = pool({ provider_gpu: undefined, gpu_types: ["NVIDIA H200"], volume: true, weights_source: "volume" });
    expect(bad.ok).toBe(false);
    const paths = (bad as any).issues.map((i: any) => i.path.join("."));
    expect(paths).toEqual(expect.arrayContaining(["provider_gpu", "gpu_types", "volume", "weights_source"]));
  });
  it("the cpu variant runs on a GMI GPU (smoke test); a real model needs weights", () => {
    expect(pool({ variant: "cpu" }).ok).toBe(true);
    const r = pool({ variant: "h3-turbo", config: "/etc/fv/runpod.toml", fake_models: undefined, models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] });
    expect((r as any).issues.some((i: any) => i.path[0] === "weights_source")).toBe(true);
  });
  it("Runpod pools refuse the provider fields", () => {
    const r = validate("pool", { ...fakePool({ provider: undefined }), compute: "CPU" });
    expect((r as any).issues.some((i: any) => i.path[0] === "provider_gpu")).toBe(true);
  });
  it("a standalone launch on GMI becomes a one-pool spec without Runpod fields", () => {
    const { spec } = standaloneSpec({ name: "g1", provider: "gmi", provider_gpu: "container.h200.x1", provider_region: "us-denver-1", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"] } as any);
    const p = spec.pools[0]!;
    expect(p).toMatchObject({ provider: "gmi", provider_gpu: "container.h200.x1", provider_region: "us-denver-1", compute: "GPU", volume: false, weights_source: "none" });
    expect(p.gpu_types).toBeUndefined();
    expect(() => standaloneSpec({ name: "g1", provider: "brev", provider_gpu: "g5.xlarge", provider_region: "x", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"] } as any)).toThrow(/instance type implies the region/);
  });
});

describe("refusals: off, not allowed, Hub downloads", () => {
  it("off with a readable reason when the secrets are unset", () => {
    expect(providerImpl("gmi").off(mkEnv({ GMI_API_KEY: undefined }))).toMatch(/GMI_API_KEY is not set/);
    expect(providerImpl("brev").off(mkEnv({ BREV_ORG_ID: undefined }))).toMatch(/BREV_ORG_ID is not set/);
    const is = providerIssues(mkEnv({ GMI_API_KEY: undefined }), { pools: [fakePool() as any] });
    expect(is).toEqual([{ path: ["pools", 0, "provider"], message: expect.stringMatching(/GMI Cloud is off: GMI_API_KEY is not set/) }]);
  });
  it("a product the owner did not allow; no PUBLIC_URL", () => {
    const is = providerIssues(mkEnv({ PUBLIC_URL: undefined }), { pools: [fakePool({ provider_gpu: "container.b300.x8" }) as any] });
    expect(is.map((i) => i.path[3] ?? i.path[2])).toEqual(["provider", "provider_gpu"]);
  });
  it("hub needs the launch's approval AND the Worker's", () => {
    const h3 = { ...presetPool("h3-turbo")!, provider: "gmi", provider_gpu: "container.h200.x1", weights_source: "hub", volume: false } as any;
    expect(providerIssues(mkEnv(), { pools: [h3] })[0]!.message).toMatch(/owner's approval/);
    expect(providerIssues(mkEnv(), { pools: [{ ...h3, hub_download_approved: true }] })[0]!.message).toMatch(/FV_HUB_DOWNLOADS_APPROVED/);
    expect(providerIssues(mkEnv({ FV_HUB_DOWNLOADS_APPROVED: "1" }), { pools: [{ ...h3, hub_download_approved: true }] })).toEqual([]);
  });
});

describe("weights plan", () => {
  it("the trees from the worker config and the preset, at pinned revisions", () => {
    const h3 = { ...presetPool("h3-turbo")!, weights_source: "hub" } as any;
    expect(poolTrees(h3)).toContain("h3-base");
    const plan = weightsPlan(h3);
    expect(plan.unsupported).toEqual([]);
    const rows = plan.tsv.trim().split("\n").map((l) => l.split("\t"));
    const base = rows.find((r) => r[0] === "tree" && r[1] === "h3-base")!;
    expect(base[2]).toBe("MiniMaxAI/MiniMax-H3");
    expect(base[3]).toMatch(/^[0-9a-f]{40}$/);
    expect(rows.some((r) => r[0] === "aux" && /^[0-9a-f]{64}$/.test(r[3]!))).toBe(true);
  });
  it("every generated tree has a 40-hex revision; multi-repo trees are not plain Hub trees", () => {
    for (const t of Object.values(HUB_TREES)) expect(t.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(HUB_TREES["h3-ref2va"]).toBeUndefined();
    expect(HUB_TREES["mmaudio-44k-v2"]).toBeUndefined();
  });
  it("none: nothing downloaded", () => expect(weightsPlan(fakePool() as any)).toMatchObject({ source: "none", trees: [], tsv: "" }));
});

describe("the pod env and start command", () => {
  const full = { FV_R2_BUCKET: "{{ RUNPOD_SECRET_fv_r2_bucket }}", FV_CF_API_TOKEN: "{{ RUNPOD_SECRET_fv_cf_api_token }}", FV_BACKSTOP_API_KEY: "rpa_x", FV_MIN_BALANCE: "8.25", FV_WORKER_DIRECT: "1", RUST_LOG: "info" };
  const o = { key: "gmi:fv-pod-a-1", name: "fv-pod-a-1", deadlineMs: 1_800_000_000_000, reportToken: "ingest_tok", weights: weightsPlan(fakePool() as any), scriptsRef: "main" };
  it("Runpod secret refs resolved from FV_PROVIDER_SECRET_ENV or dropped; the Runpod key never leaves Runpod", () => {
    const r = providerEnv(mkEnv({ FV_PROVIDER_SECRET_ENV: '{"FV_R2_BUCKET": "fv-out"}' }), full, { provider: "gmi", ...o });
    expect(r.env.FV_R2_BUCKET).toBe("fv-out");
    expect(r.dropped).toEqual(["FV_CF_API_TOKEN"]);
    expect(r.env.FV_MIN_BALANCE).toBeUndefined();
    expect(r.env.FV_BACKSTOP_API_KEY).toBe(GMI_KEY);
    expect(r.env).toMatchObject({ FV_PROVIDER: "gmi", FV_POD_ID: o.key, FV_WORKER_ID: o.key, FV_LOG_SHIP_POD: o.key, FV_CLUSTER_DEADLINE: "1800000000", FV_ENDPOINT_REPORT_URL: "https://fvc.test/ingest/v1/endpoint", FV_ENDPOINT_REPORT_TOKEN: "ingest_tok", FV_WEIGHTS_SOURCE: "none" });
    const b = providerEnv(mkEnv(), full, { provider: "brev", ...o, key: "brev:fv-pod-a-1" });
    expect(b.env.FV_BACKSTOP_API_KEY).toBeUndefined();
    expect(Object.values(b.env)).not.toContain("rpa_x");
  });
  it("hub: the trees and the scripts at the image's commit", () => {
    const h3 = { ...presetPool("h3-turbo")!, weights_source: "hub" } as any;
    const r = providerEnv(mkEnv(), {}, { provider: "gmi", ...o, weights: weightsPlan(h3), scriptsRef: "abc1234" });
    expect(atob(r.env.FV_WEIGHTS_TREES_B64!)).toContain("MiniMaxAI/MiniMax-H3");
    expect(r.env.FV_SCRIPTS_URL).toBe("https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/abc1234/scripts/gpu");
  });
  it("the boot: pinned cloudflared checked by SHA-256, the report, fv-serve last", () => {
    expect(PROVIDER_BOOT).toContain(CLOUDFLARED.url);
    expect(PROVIDER_BOOT).toContain(`${CLOUDFLARED.sha256}  /usr/local/bin/cloudflared" | sha256sum -c -`);
    expect(PROVIDER_BOOT).toContain('report tunnel "$U"');
    expect(PROVIDER_BOOT).toContain("fetch-hub-tree.py");
    expect(PROVIDER_BOOT.trim().endsWith("exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml")).toBe(true);
    expect(PROVIDER_BOOT).not.toMatch(/RUNPOD_POD_ID/);
  });
  it("Brev startup: env file 0600, the GPU, the weights dir, the host deadline; no line breaks in env", () => {
    const s = brevStartup("ghcr.io/x/y@sha256:" + "a".repeat(64), { A: "1", FV_CLUSTER_DEADLINE: "1800000000" });
    expect(s).toContain("umask 077");
    expect(s).toContain("--gpus all --network host --env-file /home/ubuntu/workspace/fv/env");
    expect(s).toContain("-v /home/ubuntu/workspace/weights:/workspace/weights");
    expect(s).toContain("-lt 1800000000");
    const envB64 = /printf %s '([^']+)' \| base64 -d > \/home\/ubuntu\/workspace\/fv\/env/.exec(s)![1]!;
    expect(atob(envB64)).toBe("A=1\nFV_CLUSTER_DEADLINE=1800000000");
    expect(() => brevStartup("img", { X: "a\nb" })).toThrow(/line break/);
  });
  it("only quick-tunnel URLs are accepted as a pod's endpoint", () => {
    expect(endpointUrlOk(mkEnv(), "https://abc-def.trycloudflare.com")).toBe(true);
    expect(endpointUrlOk(mkEnv(), "https://evil.example.com")).toBe(false);
    expect(endpointUrlOk(mkEnv(), "http://abc.trycloudflare.com")).toBe(false);
  });
});

describe("budget (no balance API)", () => {
  it("unset refuses; over refuses; within passes", async () => {
    expect((await budgetCheck(mkEnv({ GMI_BUDGET_USD: undefined }), "gmi", 3, 1)).reasons[0]).toMatch(/set GMI_BUDGET_USD/);
    expect((await budgetCheck(mkEnv(), "gmi", 3.2, 10)).ok).toBe(false);
    expect((await budgetCheck(mkEnv(), "gmi", 3.2, 1)).ok).toBe(true);
  });
  it("the collector's decision: warn at 80 %, stop at the budget", () => {
    expect(budgetDecision("gmi", "GMI Cloud", 5, 20, true)).toEqual({ alert: null, stop: false });
    expect(budgetDecision("gmi", "GMI Cloud", 17, 20, true).alert?.severity).toBe("warn");
    expect(budgetDecision("gmi", "GMI Cloud", 20, 20, true)).toMatchObject({ stop: true, alert: { severity: "critical", kind: "gmi_budget" } });
    expect(budgetDecision("gmi", "GMI Cloud", 25, 20, false).stop).toBe(false);
  });
});

describe("clients (fetch mock)", () => {
  it("GMI: bearer key, named fields only, readable errors without the key", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({ url, init });
        if (url.endsWith("/v1/containers") && init.method === "GET") return new Response(JSON.stringify([{ id: "u1", name: "fv-pod-a-1", status: "running", envs: [{ name: "S", value: "secret" }], publicIP: { ipAddress: "1.2.3.4" } }]));
        if (url.includes("/containers/products")) return new Response(JSON.stringify([{ name: "container.h200.x1", idc: "us-denver-1", type: "Container", price: 320, valid: true, gpuModel: "H200" }]));
        return new Response(JSON.stringify({ reason: `bad request ${GMI_KEY}` }), { status: 400 });
      }),
    );
    const env = mkEnv();
    const cs = await gmi.containers(env);
    expect(cs[0]).toEqual({ id: "u1", name: "fv-pod-a-1", status: "running", reason: null, product: null, idc: null, createdAt: null, publicIp: "1.2.3.4" });
    expect(JSON.stringify(cs)).not.toContain("secret");
    expect(calls[0].init.headers.authorization).toBe(`Bearer ${GMI_KEY}`);
    expect((await gmi.products(env, "us-denver-1"))[0]!.usd_per_hr).toBe(3.2);
    const err = await gmi.remove(env, "u1").catch((e) => e);
    expect(err.message).toMatch(/gmi DELETE \/containers\/u1: 400 bad request \[redacted\]/);
    expect(err.message).not.toContain(GMI_KEY);
    expect(toContainer({ id: 1, status: "ERROR" }).status).toBe("error");
  });
  it("GMI: off without the key, before any request", async () => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    await expect(gmi.containers(mkEnv({ GMI_API_KEY: undefined }))).rejects.toThrow(/GMI_API_KEY is not set/);
    expect(f).not.toHaveBeenCalled();
  });
  it("Brev: org path, vmOnlyMode create, delete of an unknown id is done; prices from BREV_PRICES", async () => {
    const calls: any[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: any) => {
        calls.push({ url, method: init.method, body: init.body ? JSON.parse(init.body) : null, auth: init.headers.authorization });
        if (init.method === "POST") return new Response(JSON.stringify({ id: "ws1" }));
        if (init.method === "DELETE") return new Response("{}", { status: 404 });
        return new Response(JSON.stringify([{ id: "ws1", name: "fv-pod-b-1", status: "running", instanceType: "g5.xlarge" }]));
      }),
    );
    const env = mkEnv();
    expect(await brev.create(env, { name: "fv-pod-b-1", instanceType: "g5.xlarge", startupScript: "#!/bin/bash" })).toBe("ws1");
    expect(calls[0]).toMatchObject({ url: "https://brev.test/api/organizations/org1/workspaces", method: "POST", auth: `Bearer ${BREV_TOKEN}`, body: { name: "fv-pod-b-1", instanceType: "g5.xlarge", vmOnlyMode: true } });
    expect((await brev.workspaces(env))[0]!.status).toBe("RUNNING");
    expect(await brev.remove(env, "gone")).toBe(true);
    expect(brevPrice(env, "g5.xlarge")).toBe(1.25);
    expect(brevPrice(env, "other")).toBeNull();
    expect(toWorkspace({ id: "x" }).name).toBe("x");
  });
  it("list keeps only fv- names (never anyone else's)", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify([{ id: "1", name: "someone-else", status: "running" }, { id: "2", name: "fv-ctl-c-p-1", status: "creating" }]))));
    const l = await providerImpl("gmi").list(mkEnv());
    expect(l.map((i) => [i.key, i.state])).toEqual([["gmi:fv-ctl-c-p-1", "starting"]]);
  });
});
