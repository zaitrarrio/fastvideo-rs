// /api/v1/deploy/*: guided recipe deploys.
import { afterEach, describe, expect, it } from "vitest";
import { mintApiToken } from "../../src/auth";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { d1 } from "./d1shim";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

function opsNs() {
  const stub = {
    fetch: async (url: string) => {
      const path = new URL(url).pathname;
      if (path === "/op/start") return json({ id: "op_1", status: "running" }, 202);
      return json({ ok: true });
    },
  };
  return { idFromName: (n: string) => n, get: () => stub } as unknown as DurableObjectNamespace;
}

const mkEnv = () =>
  ({
    DB: d1(),
    CLUSTER_OPS: opsNs(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_SECRETKEY123",
  }) as unknown as Env;

const call = (env: Env, path: string, init: RequestInit = {}) =>
  _app.fetch(new Request(`https://ctl.test${path}`, init), env, { waitUntil() {}, passThroughOnException() {} } as any);

describe("/api/v1/deploy", () => {
  it("GET /api/v1/deploy/recipes lists presets", async () => {
    const env = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const r = await call(env, "/api/v1/deploy/recipes", { headers: admin });
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.recipes.length).toBeGreaterThan(0);
    const turbo = j.recipes.find((x: any) => x.id === "h3-turbo");
    expect(turbo.min_vram_gb).toBe(80);
    expect(turbo.models.map((m: any) => m.id)).toContain("fasth3");
  });

  it("POST /api/v1/deploy/validate checks a recipe spec", async () => {
    const env = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const good = await call(env, "/api/v1/deploy/validate", {
      method: "POST",
      headers: { ...admin, "content-type": "application/json" },
      body: JSON.stringify({ recipe: "h3-turbo", name: "fv-test-1", count: 1 }),
    });
    expect(good.status).toBe(200);

    const bad = await call(env, "/api/v1/deploy/validate", {
      method: "POST",
      headers: { ...admin, "content-type": "application/json" },
      body: JSON.stringify({ recipe: "nope", name: "fv-test-1" }),
    });
    expect(bad.status).toBe(404);
  });

  it("POST /api/v1/deploy creates and starts a cluster", async () => {
    const env = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const r = await call(env, "/api/v1/deploy", {
      method: "POST",
      headers: { ...admin, "content-type": "application/json" },
      body: JSON.stringify({ recipe: "h3-turbo", name: "fv-deploy-test-1", count: 1, start: true }),
    });
    expect(r.status).toBe(201);
    const j: any = await r.json();
    expect(j.cluster.name).toBe("fv-deploy-test-1");
    expect(j.operation).toMatch(/^op_/);
  });

  it("requires auth", async () => {
    const env = mkEnv();
    const r = await call(env, "/api/v1/deploy/recipes");
    expect(r.status).toBe(401);
  });
});
