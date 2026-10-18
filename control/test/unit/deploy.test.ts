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

// One-click deploy (deploy.ts): families × instance types into a new cluster or an existing one.
describe("/api/v1/deploy/plan and /apply", () => {
  const post = async (env: Env, path: string, b: unknown) => {
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}`, "content-type": "application/json" };
    const r = await call(env, path, { method: "POST", headers: admin, body: JSON.stringify(b) });
    return { status: r.status, j: (await r.json()) as any };
  };
  // No upstream is reached: Runpod's catalog and stock answer nothing (the live checks then skip).
  const quiet = () => (globalThis.fetch = (async () => json({ errors: [{ message: "offline" }] }, 500)) as any);

  it("GET /api/v1/deploy/catalog: the families with their presets and templates, the clusters", async () => {
    const env = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const j: any = await (await call(env, "/api/v1/deploy/catalog", { headers: admin })).json();
    const fam = (id: string) => j.families.find((f: any) => f.id === id);
    expect(j.families.map((f: any) => f.id)).toEqual(["h3", "ltx2", "wan"]);
    expect([fam("h3").default, fam("ltx2").default, fam("wan").default]).toEqual(["h3-turbo", "ltx", "wan"]);
    expect(fam("h3").templates.map((t: any) => t.id)).toEqual(expect.arrayContaining(["h3", "fasth3"]));
    expect(fam("wan").templates.map((t: any) => t.id)).toEqual(expect.arrayContaining(["streaming", "wan"]));
    expect(fam("h3").presets.map((p: any) => p.id)).toContain("vdn-h3");
    expect(j.templates.map((t: any) => t.id)).toEqual(expect.arrayContaining(["h3", "fasth3", "streaming"]));
  });

  it("plan for a new cluster: one pool per family (a template's pools for one), with the instance types; nothing saved", async () => {
    quiet();
    const env = mkEnv();
    const { status, j } = await post(env, "/api/v1/deploy/plan", {
      name: "one-click-1",
      families: [
        { family: "h3", pools: "template:streaming" },
        { family: "ltx2", gpu_types: ["NVIDIA H100 80GB HBM3", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], count: 2 },
      ],
    });
    expect(status).toBe(200);
    // h3 with a wan template: refused at its entry.
    expect(j.issues).toContainEqual({ path: ["families", 0, "pools"], message: expect.stringMatching(/another family/) });
    const { j: ok } = await post(env, "/api/v1/deploy/plan", {
      name: "one-click-1",
      families: [
        { family: "h3", pools: "h3-max", gpu_types: ["NVIDIA H100 80GB HBM3", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], count: 2 },
        { family: "ltx2", pools: "template:ltx" },
      ],
    });
    expect(ok.issues, JSON.stringify(ok.issues)).toEqual([]);
    expect(ok.target).toMatchObject({ kind: "new", name: "one-click-1", running: false });
    expect(ok.pools.map((p: any) => [p.id, p.family, p.count])).toEqual([["h3-max", "h3", 2], ["ltx-family", "ltx2", 1]]);
    // The ltx template is ONE pool serving every LTX variant (owner 2026-10-18).
    expect(ok.pools[1].models).toEqual(["ltx25-distill-sol", "ltx25-distill-dense", "ltx25-a2v-guided", "ltx25-ref2v"]);
    expect(ok.pools[0].where).toMatch(/H100 80GB HBM3 → NVIDIA RTX PRO 6000/);
    expect(ok.spec.pools[0]).toMatchObject({ id: "h3-max", gpu_types: ["NVIDIA H100 80GB HBM3", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], count: 2 });
    expect(ok.actions.map((a: any) => a.target)).toEqual(["h3-max", "ltx-family"]);
    // A Runpod family without the volume: the data centres in order (r2 here; its store is checked live).
    const dc = await post(env, "/api/v1/deploy/plan", { name: "one-click-1", families: [{ family: "wan", pools: "template:streaming", weights_source: "r2", data_centers: ["US-KS-2", "any"] }] });
    expect(dc.j.spec.pools.find((p: any) => p.id === "streaming-family")).toMatchObject({ weights_source: "r2", volume: false, data_centers: ["US-KS-2", "any"] });
    expect(dc.j.pools.map((p: any) => [p.id, p.models])).toEqual([["streaming-family", ["sfwan21-1.3b", "longlive-1.3b"]]]);
    // The h3 template: ONE pod with every H3 variant (one worker), its weights from R2 by default.
    const h3 = await post(env, "/api/v1/deploy/plan", { name: "one-click-1", families: [{ family: "h3", pools: "template:h3" }] });
    expect(h3.j.pools.map((p: any) => [p.id, p.count])).toEqual([["h3-family", 1]]);
    expect(h3.j.pools[0].models).toEqual(expect.arrayContaining(["fasth3", "sol-h3", "fasth3-v2-nvfp4", "fasth3-trim-8step", "vdn-h3", "h3-ref2v-turbo", "h3-larry-turbo"]));
    expect(h3.j.spec.pools[0].config_toml).toMatch(/^swap = true$/m);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM clusters").first<any>()).toMatchObject({ n: 0 });
  });

  it("all families: every family's standard pool; a Brev family's disk reaches the pool", async () => {
    quiet();
    const env = mkEnv();
    const { j } = await post(env, "/api/v1/deploy/plan", { name: "all-fam", all_families: true, families: [{ family: "wan", provider: "brev", provider_gpu: "g5.xlarge", weights_source: "r2", container_disk_gb: 900 }] });
    expect(j.pools.map((p: any) => p.id)).toEqual(["wan", "h3-turbo", "ltx"]);
    expect(j.spec.pools[0]).toMatchObject({ provider: "brev", provider_gpu: "g5.xlarge", container_disk_gb: 900, volume: false });
    // Brev is off on this fv-control: said at the wan entry.
    expect(j.issues.some((i: any) => i.path.join(".") === "families.0.provider" && /NVIDIA Brev is off/.test(i.message)), JSON.stringify(j.issues)).toBe(true);
  });

  it("apply: a new cluster is defined and started; an existing one gets the pools added (ids made unique, version checked)", async () => {
    quiet();
    const env = mkEnv();
    const a = await post(env, "/api/v1/deploy/apply", { name: "one-click-2", control_plane: "direct", families: [{ family: "h3" }] });
    expect(a.status).toBe(201);
    expect(a.j.cluster.spec.pools.map((p: any) => p.id)).toEqual(["h3-turbo"]);
    expect(a.j.operation).toMatch(/^op_/);
    const id = a.j.cluster.id;
    // Existing (stopped): h3-turbo again becomes h3-turbo-2; the streaming template's pools join; the up starts.
    const plan = await post(env, "/api/v1/deploy/plan", { cluster_id: id, families: [{ family: "h3", pools: "h3-turbo" }, { family: "ltx2", pools: "template:ltx" }] });
    expect(plan.j.issues, JSON.stringify(plan.j.issues)).toEqual([]);
    expect(plan.j.target).toMatchObject({ kind: "existing", id, name: "one-click-2", version: expect.any(Number) });
    expect(plan.j.pools.map((p: any) => p.id)).toEqual(["h3-turbo-2", "ltx-family"]);
    const stale = await post(env, "/api/v1/deploy/apply", { cluster_id: id, families: [{ family: "ltx2" }], version: plan.j.target.version + 5 });
    expect(stale.status).toBe(409);
    const ap = await post(env, "/api/v1/deploy/apply", { cluster_id: id, families: [{ family: "h3", pools: "h3-turbo" }, { family: "ltx2", pools: "template:ltx" }], version: plan.j.target.version });
    expect(ap.status).toBe(200);
    expect(ap.j.cluster.spec.pools.map((p: any) => p.id)).toEqual(["h3-turbo", "h3-turbo-2", "ltx-family"]);
    expect(ap.j.operation).toMatch(/^op_/);
    // A name and a cluster_id together are refused; no family neither.
    expect((await post(env, "/api/v1/deploy/plan", { cluster_id: id, name: "x", families: [{ family: "h3" }] })).status).toBe(400);
    expect((await post(env, "/api/v1/deploy/plan", { name: "x", families: [] })).j.issues).toContainEqual({ path: ["families"], message: "pick a model family (or all families)" });
  });
});
