// GET /api/v1/models: aggregates /fv/v1/capabilities across the fleet.
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
  object: "fv.capabilities",
  models: [
    {
      caps: {
        id: "fasth3",
        family: "h3",
        served_names: ["h3-turbo"],
        tasks: ["t2v"],
        tier: "Turbo",
        recipe: "fasth3-4step-vsa",
        resident: true,
      },
      recipe: "fasth3-4step-vsa",
      executors: ["cuda"],
    },
    {
      caps: { id: "wan22-ti2v-5b", family: "wan", served_names: [], tasks: ["t2v", "i2v"], resident: false },
      recipe: "wan-max",
      executors: ["cuda"],
    },
  ],
  tiers: [],
  aliases: {},
};

describe("GET /api/v1/models", () => {
  it("aggregates capabilities by model id with per-engine availability", async () => {
    const env = mkEnv();
    const run = (sql: string, ...a: unknown[]) => env.DB.prepare(sql).bind(...a).run();
    await run("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'a', 'o', 'c', 0, 0), ('pod-b', 'b', 'o', 'c', 0, 0)");
    globalThis.fetch = (async (url: any) => {
      if (String(url).includes("pod-a") && String(url).endsWith("/fv/v1/capabilities")) return json(CAPS);
      throw new Error("unreachable");
    }) as any;

    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/models", { headers: read }), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.engines_total).toBe(2);
    expect(j.engines_reached).toBe(1);
    expect(j.models).toHaveLength(2);
    const fh = j.models.find((m: any) => m.id === "fasth3");
    expect(fh.family).toBe("h3");
    expect(fh.tier).toBe("Turbo");
    expect(fh.recipe).toBe("fasth3-4step-vsa");
    expect(fh.resident).toBe(true);
    expect(fh.available_on).toEqual(["pod-a"]);
  });

  it("requires auth", async () => {
    const env = mkEnv();
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/models"), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(401);
  });
});
