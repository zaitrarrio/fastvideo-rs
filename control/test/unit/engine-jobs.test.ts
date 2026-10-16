// GET /api/v1/jobs: engine jobs aggregated across the fleet.
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

const JOBS_A = {
  jobs: [
    { id: "job-1", state: "running", progress: 0.5, created_at: "2026-10-15T10:00:00Z", resolved: { model: "fasth3" }, metrics: { inference_time_s: 12.5 } },
    { id: "job-2", state: "queued", progress: 0, created_at: "2026-10-15T10:01:00Z", resolved: { model: "fasth3" } },
  ],
};

describe("GET /api/v1/jobs", () => {
  it("aggregates engine jobs, tagged by pod, newest first", async () => {
    const env = mkEnv();
    const run = (sql: string, ...a: unknown[]) => env.DB.prepare(sql).bind(...a).run();
    await run("INSERT INTO pods (pod_id, name, owner, cluster_id, first_seen, last_seen) VALUES ('pod-a', 'fv-a', 'o', 'c', 0, 0), ('pod-b', 'fv-b', 'o', 'c', 0, 0)");
    globalThis.fetch = (async (url: any) => {
      if (String(url).includes("pod-a") && String(url).includes("/fv/v1/jobs")) return json(JOBS_A);
      throw new Error("unreachable");
    }) as any;

    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/jobs?limit=50", { headers: read }), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.engines_total).toBe(2);
    expect(j.engines_reached).toBe(1);
    expect(j.jobs).toHaveLength(2);
    // Newest first.
    expect(j.jobs[0].id).toBe("job-2");
    expect(j.jobs[0].pod_id).toBe("pod-a");
    expect(j.jobs[0].pod_name).toBe("fv-a");
    expect(j.jobs[1].metrics.inference_time_s).toBe(12.5);
  });

  it("requires auth", async () => {
    const env = mkEnv();
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/jobs"), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(401);
  });
});
