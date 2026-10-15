// GET /api/v1/fleet: pods joined with their engines' /fv/v1/status (best-effort).
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

const ENGINE_STATUS = {
  pools: [
    {
      id: "h3-turbo",
      kind: "pod",
      state: "Ready",
      models: ["fasth3"],
      queued: 2,
      running: 1,
      versions: [{ sha: "abc123", channel: "stable", workers: 1 }],
      mixed_versions: false,
    },
  ],
};

describe("GET /api/v1/fleet", () => {
  it("joins pods with engine status; engine:null when unreachable", async () => {
    const env = mkEnv();
    const run = (sql: string, ...a: unknown[]) => env.DB.prepare(sql).bind(...a).run();
    await run(
      "INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, gpu_util, gpu_mem, jobs_running, jobs_queued, health, build_sha, uptime_s, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      "pod-up-1",
      "fv-h3-turbo-1",
      "cluster:alpha",
      "c_a",
      "RUNNING",
      1.5,
      "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
      1,
      80,
      50,
      1,
      2,
      "ready",
      "abc123",
      3600,
      0,
      0
    );
    await run(
      "INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      "pod-down-1",
      "fv-h3-turbo-2",
      "cluster:alpha",
      "c_a",
      "RUNNING",
      1.5,
      "NVIDIA RTX A6000",
      1,
      0,
      0
    );
    // Engine reachable for pod-up-1 only.
    globalThis.fetch = (async (url: any) => {
      if (String(url).includes("pod-up-1") && String(url).endsWith("/fv/v1/status")) return json(ENGINE_STATUS);
      throw new Error("unreachable");
    }) as any;

    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const r = await _app.fetch(
      new Request("https://ctl.test/api/v1/fleet", { headers: read }),
      env,
      { waitUntil() {}, passThroughOnException() {} } as any
    );
    expect(r.status).toBe(200);
    const j: any = await r.json();

    expect(j.summary.workers).toBe(2);
    expect(j.summary.workers_with_engine).toBe(1);
    expect(j.summary.gpus).toBe(2);
    expect(j.summary.cost_per_hr).toBe(3);
    expect(j.summary.queued).toBe(2);
    expect(j.summary.running).toBe(1);

    const up = j.workers.find((w: any) => w.pod_id === "pod-up-1");
    expect(up.vram_total_gb).toBe(96);
    expect(up.engine.pools[0].id).toBe("h3-turbo");
    expect(up.engine.queued).toBe(2);

    const down = j.workers.find((w: any) => w.pod_id === "pod-down-1");
    expect(down.engine).toBeNull();
    expect(down.vram_total_gb).toBe(48);

    expect(j.pools).toHaveLength(1);
    expect(j.pools[0].id).toBe("h3-turbo");
    expect(j.pools[0].workers).toEqual(["pod-up-1"]);
  });

  it("requires auth", async () => {
    const env = mkEnv();
    const r = await _app.fetch(new Request("https://ctl.test/api/v1/fleet"), env, {
      waitUntil() {},
      passThroughOnException() {},
    } as any);
    expect(r.status).toBe(401);
  });
});
