// control_plane = "edge" (docs/serve/edge-control-plane.md §5): no gateway
// pod; every worker is an API front behind the edge Worker.
import { execFileSync, spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { EDGE_WORKER_BOOT, isDirect, isReserved, WORKER_BOOT, workerCreatePayload, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { edgeWorkers } from "../../src/cluster/ops";
import { defaultSpec, isEdge, modelFamily, normalizeSpec, poolModelFamilies, presetPool } from "../../src/cluster/spec";
void defaultSpec;

const EDGE = { url: "https://fv-edge-staging.example.workers.dev", internal_token: "edge-it", admin_token: "fvadm_edge", d1_database_id: "d1-edge" };
const edgeSpec = (template = "standard") => normalizeSpec({ name: "e1", template, control_plane: "edge" });
const ctx = (over: Partial<EnvCtx> = {}): EnvCtx => ({
  edge: EDGE,
  spec: edgeSpec(),
  state: { images: {}, workers: {} },
  secrets: { internal_token: "cluster-it", url_signing_key: "us", ingest_token: "fvi_x" },
  deadlineMs: 1_700_000_000_000,
  runpodApiKey: "rpa_KEY",
  ingestUrl: "https://ctl/ingest/v1/logs",
  ...over,
});

describe("spec", () => {
  it("control_plane defaults to edge; direct is the other mode", () => {
    expect(normalizeSpec({ name: "g1" }).control_plane).toBe("edge");
    expect(isEdge(edgeSpec())).toBe(true);
    expect(isEdge(normalizeSpec({ name: "d1", control_plane: "direct" }))).toBe(false);
    expect(() => normalizeSpec({ name: "e3", control_plane: "both" })).toThrow(/control_plane/);
    expect(() => normalizeSpec({ name: "e4", control_plane: "edge", pools: [{ ...presetPool("ltx"), family: "Bad" }] })).toThrow();
  });
  it("families from the pools' models (h3, ltx, causal wan → sfwan, wan, fake), or the pool's own", () => {
    const fam = (id: string) => poolModelFamilies(presetPool(id)!);
    expect(fam("h3-turbo")).toEqual({ fasth3: "h3" });
    expect(fam("ltx")).toEqual({ "ltx25-distill-sol": "ltx" });
    expect(fam("wan")).toEqual({ "fastwan22-ti2v-5b": "wan", "wan22-ti2v-5b": "wan" });
    expect(fam("sfwan")).toEqual({ "sfwan21-1.3b": "sfwan" });
    expect(fam("longlive")).toEqual({ "longlive-1.3b": "sfwan" });
    const tiny = defaultSpec("t", "tiny-cpu").pools[0]!;
    expect(poolModelFamilies(tiny)).toEqual({ "fake-wan": "fake" });
    expect(modelFamily({ ...tiny, family: "wan" })).toBe("wan");
  });
});

describe("worker env", () => {
  it("an edge front: the edge's token, URL, families and D1; direct uploads; its own backstop; no R2 and no direct mode", () => {
    const c = ctx();
    expect(isDirect(c.spec, c.state)).toBe(false);
    const h3 = c.spec.pools.find((p) => p.id === "h3-turbo")!;
    const e = workerSystemEnv(c, h3, "img@sha256:abc");
    expect(e).toMatchObject({
      FV_SERVE_ROLE: "worker",
      FV_AUTH_MODE: "trust-edge",
      FV_INTERNAL_TOKEN: "edge-it",
      FV_PUBLIC_BASE_URL: EDGE.url,
      FV_DISPATCH_FRONT: "1",
      FV_DISPATCH_DO_URL: EDGE.url,
      FV_DISPATCH_FAMILIES: "h3",
      FV_DISPATCH_MODEL_FAMILIES: "fasth3=h3",
      FV_DISPATCH_DIRECT_UPLOAD: "1",
      FV_DISPATCH_MAX_QUEUED: "32",
      FV_MP4_FRAGMENTED: "1",
      FV_D1_DATABASE_ID: "d1-edge",
      FV_CLUSTER_DEADLINE: "1700000000",
      FV_MIN_BALANCE: String(c.spec.min_balance),
      FV_BACKSTOP_API_KEY: "rpa_KEY",
    });
    expect(Object.keys(e).filter((k) => k.startsWith("FV_R2_"))).toEqual([]);
    // With the edge's outputs bucket: the account's R2 credentials, the edge's bucket.
    const e2 = workerSystemEnv(ctx({ edge: { ...EDGE, outputs_bucket: "fv-edge-staging-outputs" } }), h3, "img");
    expect(e2.FV_R2_BUCKET).toBe("fv-edge-staging-outputs");
    expect(e2.FV_R2_ACCESS_KEY_ID).toBe("{{ RUNPOD_SECRET_fv_r2_access_key_id }}");
    expect(e.FV_WORKER_DIRECT).toBeUndefined();
    expect(e.FV_ADMIN_TOKEN).toBeUndefined();
    expect(e.FV_CF_API_TOKEN).toBe("{{ RUNPOD_SECRET_fv_cf_api_token }}");
    // Without the edge settings the env cannot be made.
    expect(() => workerSystemEnv(ctx({ edge: undefined }), h3, "img")).toThrow(/EDGE_URL/);
    for (const k of ["FV_DISPATCH_FRONT", "FV_DISPATCH_DO_URL", "FV_DISPATCH_FAMILIES", "FV_DISPATCH_MODEL_FAMILIES", "FV_DISPATCH_ENDPOINT"]) expect(isReserved(k)).toBe(true);
  });
  it("a direct cluster's workers run the plain boot and their own client auth", () => {
    const spec = normalizeSpec({ name: "d", control_plane: "direct" });
    const e = workerSystemEnv({ ...ctx(), spec, secrets: { internal_token: "cluster-it", url_signing_key: "us", admin_token: "fvadm_c" } }, spec.pools[0]!, "img");
    expect(e.FV_DISPATCH_FRONT).toBeUndefined();
    expect(e.FV_INTERNAL_TOKEN).toBe("cluster-it");
    expect(e.FV_WORKER_DIRECT).toBe("1");
    expect(e.FV_R2_BUCKET).toBeDefined();
    const pl = workerPlacements(spec, spec.pools[0]!)[0]!;
    expect(workerCreatePayload("n", "img", spec.pools[0]!, pl, e).dockerStartCmd).toEqual([WORKER_BOOT]);
  });
  it("the create payload runs the edge boot", () => {
    const c = ctx();
    const p = c.spec.pools[0]!;
    const e = workerSystemEnv(c, p, "img");
    expect(workerCreatePayload("n", "img", p, workerPlacements(c.spec, p)[0]!, e).dockerStartCmd).toEqual([EDGE_WORKER_BOOT]);
  });
});

describe("the edge boot", () => {
  const watchdog = /\n\(\n([\s\S]*?)\n\) &\n/.exec(EDGE_WORKER_BOOT)![1]!;
  it("is the worker boot plus the endpoint and the watchdog", () => {
    expect(EDGE_WORKER_BOOT.replace(/export FV_DISPATCH_ENDPOINT=.*\n/, "").replace(/\n\(\n[\s\S]*?\n\) &\n/, "\n")).toBe(WORKER_BOOT);
  });
  it("exports the pod's own proxy URL as its endpoint", () => {
    const dir = mkdtempSync(join(tmpdir(), "fve-"));
    const boot = EDGE_WORKER_BOOT.replace(/\n\(\n[\s\S]*?\n\) &\n/, "\n").replaceAll("/fvstate", `${dir}/state`).replaceAll("/fv-worker.toml", `${dir}/w.toml`).replace(/exec .*$/, 'echo "$FV_DISPATCH_ENDPOINT $FV_WORKER_ID"');
    const out = execFileSync("bash", ["-c", boot], { env: { PATH: process.env.PATH!, RUNPOD_POD_ID: "pod42", FV_WORKER_TOML_B64: Buffer.from("[server]\n").toString("base64") } }).toString();
    expect(out.trim()).toBe("https://pod42-8000.proxy.runpod.net pod42");
  });
  it("deletes its own pod at the deadline and below the balance floor", () => {
    const dir = mkdtempSync(join(tmpdir(), "fvwd-"));
    const log = join(dir, "curl.log");
    // A curl stand-in: records its arguments, answers the balance query.
    writeFileSync(join(dir, "curl"), `#!/bin/sh\necho "$*" >> ${log}\ncase "$*" in *graphql*) echo '{"data":{"myself":{"clientBalance":5.5}}}';; esac\n`);
    chmodSync(join(dir, "curl"), 0o755);
    const run = (deadline: string) =>
      spawnSync("bash", ["-c", watchdog.replaceAll("/fvstate", dir).replace(/sleep (30|60|5)/g, "sleep 0.2")], {
        env: { PATH: `${dir}:${process.env.PATH}`, RUNPOD_POD_ID: "pod42", FV_BACKSTOP_API_KEY: "rpa_K", FV_CLUSTER_DEADLINE: deadline, FV_MIN_BALANCE: "8.25" },
        timeout: 1500,
      });
    run("0");
    expect(readFileSync(log, "utf8")).toContain("-X DELETE -H Authorization: Bearer rpa_K https://rest.runpod.io/v1/pods/pod42");
    writeFileSync(log, "");
    run(String(Math.floor(Date.now() / 1000) + 3600));
    const l = readFileSync(log, "utf8");
    expect(l).toContain("graphql");
    expect(l).toContain("-X DELETE -H Authorization: Bearer rpa_K https://rest.runpod.io/v1/pods/pod42");
  });
});

describe("the families view", () => {
  it("a pod is ready when it is a connected, ready front in any of its families", () => {
    const view = {
      families: {
        h3: { workers: [{ worker_id: "p1", connected: true, ready: true, draining: false, held: 2, front: { url: "u", ready: true }, sha: "abc" }] },
        ltx: {
          workers: [
            { worker_id: "p2", connected: true, ready: false, held: 0, front: { url: "u" } },
            { worker_id: "p3", connected: true, ready: true, held: 0 },
            { worker_id: "p4", connected: true, ready: true, draining: true, held: 1, front: { url: "u" } },
          ],
        },
      },
    };
    const m = edgeWorkers(view);
    expect(m.get("p1")).toEqual({ ready: true, held: 2, families: ["h3"], sha: "abc" });
    expect(m.get("p2")?.ready).toBe(false);
    expect(m.get("p3")?.ready).toBe(false); // not a front
    expect(m.get("p4")?.ready).toBe(false); // draining
    expect(edgeWorkers({}).size).toBe(0);
  });
});
