// /api/v1/benchmarks and /api/v1/compare: observed performance by model.
import { afterEach, describe, expect, it } from "vitest";
import { mintApiToken } from "../../src/auth";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { d1 } from "./d1shim";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const mkEnv = () =>
  ({
    DB: d1(),
    CONTROL_KEK: "k".repeat(32),
    SESSION_SECRET: "s".repeat(40),
  }) as unknown as Env;

const CAPS = {
  models: [{ caps: { id: "fasth3", family: "h3", served_names: [], tier: "Turbo", resident: true }, recipe: "r", executors: [] }],
};
const JOBS = {
  jobs: [
    { id: "j1", state: "succeeded", resolved: { model: "fasth3" }, metrics: { inference_s: 10, peak_memory_mb: 35000, stage_durations: { denoise: 8 } } },
    { id: "j2", state: "succeeded", resolved: { model: "fasth3" }, metrics: { inference_s: 20, peak_memory_mb: 36000, stage_durations: { denoise: 16 } } },
    { id: "j3", state: "running", resolved: { model: "fasth3" }, metrics: {} },
    { id: "j4", state: "failed", resolved: { model: "fasth3" }, metrics: { inference_s: 99 } },
  ],
};

const call = (env: Env, path: string, headers: any) =>
  _app.fetch(new Request(`https://ctl.test${path}`, { headers }), env, { waitUntil() {}, passThroughOnException() {} } as any);

describe("/api/v1/benchmarks", () => {
  it("aggregates succeeded jobs by model with percentiles", async () => {
    const env = mkEnv();
    await env.DB.prepare("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'a', 'o', 'c', 0, 0)").run();
    globalThis.fetch = (async (url: any) => {
      const u = String(url);
      if (u.includes("pod-a") && u.includes("/fv/v1/jobs")) return json(JOBS);
      throw new Error("unreachable");
    }) as any;
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await call(env, "/api/v1/benchmarks", read);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.models).toHaveLength(1);
    const m = j.models[0];
    expect(m.model).toBe("fasth3");
    expect(m.jobs).toBe(2); // running + failed excluded
    expect(m.inference_s.p50).toBe(20); // pct: floor(0.5*2)=idx 1
    expect(m.inference_s.p90).toBe(20);
    expect(m.inference_s.min).toBe(10);
    expect(m.peak_memory_mb.max).toBe(36000);
    expect(m.stages.denoise).toBe(16); // p50 of [8,16]
  });
});

describe("/api/v1/compare", () => {
  it("returns side-by-side capabilities and benchmarks", async () => {
    const env = mkEnv();
    await env.DB.prepare("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'a', 'o', 'c', 0, 0)").run();
    globalThis.fetch = (async (url: any) => {
      const u = String(url);
      if (u.includes("pod-a") && u.endsWith("/fv/v1/capabilities")) return json(CAPS);
      if (u.includes("pod-a") && u.includes("/fv/v1/jobs")) return json(JOBS);
      throw new Error("unreachable");
    }) as any;
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await call(env, "/api/v1/compare?models=fasth3,missing", read);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.models).toHaveLength(2);
    expect(j.models[0].id).toBe("fasth3");
    expect(j.models[0].capabilities.family).toBe("h3");
    expect(j.models[0].benchmarks.jobs).toBe(2);
    expect(j.models[1].id).toBe("missing");
    expect(j.models[1].capabilities).toBeNull();
    expect(j.models[1].benchmarks).toBeNull();
  });

  it("requires the models param", async () => {
    const env = mkEnv();
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await call(env, "/api/v1/compare", read);
    expect(r.status).toBe(400);
  });
});
