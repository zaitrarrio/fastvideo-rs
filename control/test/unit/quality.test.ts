// /api/v1/quality: gate definitions from the engine.
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

const call = (env: Env, path: string, headers: any) =>
  _app.fetch(new Request(`https://ctl.test${path}`, { headers }), env, { waitUntil() {}, passThroughOnException() {} } as any);

const GATES = {
  object: "quality.gates",
  gates: [{ id: "non_finite", title: "No NaNs / infinities", threshold: "0", unit: "pixels", failure_mode: "dead weights" }],
  tiers: { Max: "must pass all gates" },
};

describe("/api/v1/quality", () => {
  it("returns gates from the first reachable engine", async () => {
    const env = mkEnv();
    await env.DB.prepare("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'a', 'o', 'c', 0, 0)").run();
    globalThis.fetch = (async (url: any) => {
      if (String(url).includes("pod-a")) return json(GATES);
      throw new Error("unreachable");
    }) as any;
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await call(env, "/api/v1/quality", read);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.quality.gates).toHaveLength(1);
    expect(j.quality.gates[0].id).toBe("non_finite");
    expect(j.engines_reached).toBe(1);
  });

  it("returns null quality when no engine is reachable", async () => {
    const env = mkEnv();
    await env.DB.prepare("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'a', 'o', 'c', 0, 0)").run();
    globalThis.fetch = (async () => {
      throw new Error("unreachable");
    }) as any;
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await call(env, "/api/v1/quality", read);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.quality).toBeNull();
    expect(j.engines_reached).toBe(0);
  });
});
