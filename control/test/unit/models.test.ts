// GET /api/v1/models: the deployable catalog from POOL_PRESETS.
import { describe, expect, it } from "vitest";
import { mintApiToken } from "../../src/auth";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { d1 } from "./d1shim";

const mkEnv = () =>
  ({
    DB: d1(),
    CONTROL_KEK: "k".repeat(32),
    SESSION_SECRET: "s".repeat(40),
  }) as unknown as Env;

describe("GET /api/v1/models", () => {
  it("returns the preset catalog with pool models", async () => {
    const env = mkEnv();
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/models", { headers: read }), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.models.length).toBeGreaterThan(0);
    const turbo = j.models.find((m: any) => m.id === "h3-turbo");
    expect(turbo).toBeDefined();
    expect(turbo.min_vram_gb).toBe(80);
    expect(turbo.pool.models.map((m: any) => m.id)).toContain("fasth3");
    for (const m of j.models) {
      expect(m.id).toBeTruthy();
      expect(m.title).toBeTruthy();
      expect(typeof m.min_vram_gb).toBe("number");
      expect(Array.isArray(m.weights)).toBe(true);
    }
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
