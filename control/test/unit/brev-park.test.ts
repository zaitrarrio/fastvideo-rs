// NVIDIA Brev keep-on-stop (src/brev-park.ts, docs/serve/deploy-gmi-brev.md §3.4, §7): the weights loop of the
// boot (run in bash with a stub fetcher), the instance-type list, park on stop / delete, warm candidates, limits,
// the budget with parked storage, the boot endpoint; Brev's API is a fetch mock over a D1 shim.
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_POLICIES } from "../../src/alerts";
import { parseDuration, parseGiB, resetBrevTypes, toType } from "../../src/brev";
import {
  brevBoot,
  brevDiskGb,
  brevRow,
  claimWarm,
  deleteParked,
  enforceParkLimits,
  holdFailed,
  parkedView,
  recordBrevCreate,
  releaseBrev,
  restartDiag,
  storageDph,
  unpark,
  warmCandidates,
} from "../../src/brev-park";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { brevOutlook, budgetCheck, WEIGHTS_SH } from "../../src/providers";
import { putSetting } from "../../src/util";
import { d1 } from "./d1shim";

const REV_A = "a".repeat(40);
const REV_B = "b".repeat(40);
const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    PUBLIC_URL: "https://fvc.test",
    BREV_API_TOKEN: "brev_SECRET",
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "stop.x1,nostop.x1",
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;

/** A Brev API stand-in: workspaces with a status, stop / start / delete, the instance-type list. */
function brevMock(o: { startFail?: boolean } = {}) {
  const ws = new Map<string, { status: string }>();
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      const path = url.replace("https://brev.test/api/", "");
      calls.push(`${init.method} ${path}`);
      if (path.startsWith("instances/alltypesavailable/"))
        return new Response(
          JSON.stringify({
            allInstanceTypes: [
              { type: "stop.x1", stoppable: true, base_price: { amount: "2.000000" }, supported_storage: [{ size: "0B", price_per_gb_hr: { amount: "0.000132" } }, { size: "0B", price_per_gb_hr: { amount: "0.000205" } }], elastic_root_volume: true, estimated_deploy_time: "7m0s", cloud_cred_id: "c1", supported_gpus: [{ name: "H100", count: 1 }] },
              { type: "nostop.x1", stoppable: null, base_price: { amount: "1.500000" }, supported_storage: [{ size: "1TiB226GiB", type: "ssd" }], estimated_deploy_time: "6m30s", cloud_cred_id: "c2" },
            ],
          }),
        );
      const m = /^workspaces\/([^/]+)(?:\/(stop|start))?$/.exec(path);
      if (m) {
        const w = ws.get(m[1]!) || { status: "RUNNING" };
        ws.set(m[1]!, w);
        if (m[2] === "stop") w.status = "STOPPED";
        if (m[2] === "start") {
          if (o.startFail) return new Response(JSON.stringify({ message: "no capacity" }), { status: 409 });
          w.status = "RUNNING";
        }
        if (init.method === "DELETE") ws.delete(m[1]!);
        return new Response("{}");
      }
      return new Response("[]");
    }),
  );
  return { ws, calls };
}

async function live(env: Env, id: string, o: { type?: string; stoppable?: boolean; trees?: Record<string, string>; ready?: boolean; disk?: number } = {}) {
  const name = `fv-pod-${id}-1010000000`;
  await recordBrevCreate(env, {
    workspace_id: id,
    pod_id: `brev:${name}`,
    name,
    instance_type: o.type ?? "stop.x1",
    info: toType({ type: o.type ?? "stop.x1", stoppable: o.stoppable ?? true, supported_storage: [{ price_per_gb_hr: { amount: "0.000132" } }], location: "us-east-1" }),
    disk_gb: o.disk ?? 300,
    launch_trees: o.trees ?? { "wan22-ti2v-5b": REV_A },
    cluster_id: "c1",
    token: `fvb_${id}`,
    run: `#!/bin/bash\necho run ${id}`,
  });
  await env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, created_at, ready_at) VALUES (?, 'c1', 'worker', 1, ?)").bind(`brev:${name}`, o.ready === false ? null : 2).run();
  return `brev:${name}`;
}

beforeEach(() => resetBrevTypes());
afterEach(() => vi.unstubAllGlobals());

describe("the boot's weights loop (bash, stub fetcher)", () => {
  const run = (dir: string, rows: string[][], log: string) => {
    const script = `set -u
say() { echo "say: $*" >&2; }
report() { :; }
fail() { echo "FAIL $1"; exit 1; }
fetch_tree() { mkdir -p "$FV_WEIGHTS/$1"; echo "$3" > "$FV_WEIGHTS/$1/model.bin"; touch "$FV_WEIGHTS/$1/.complete"; echo "$1" >> ${log}; }
${WEIGHTS_SH}`;
    return execFileSync("bash", ["-c", script], { env: { PATH: process.env.PATH!, FV_WEIGHTS: dir, FV_WEIGHTS_SOURCE: "hub", FV_WEIGHTS_TREES_B64: Buffer.from(rows.map((r) => r.join("\t")).join("\n") + "\n").toString("base64") }, stdio: "pipe" }).toString();
  };
  it("fetches missing trees, skips ones complete at the pinned revision, replaces another revision or a leftover", () => {
    const dir = mkdtempSync(join(tmpdir(), "fvw-"));
    const log = join(dir, "..", `fetched-${Date.now()}.log`);
    run(dir, [["tree", "t1", "r/t1", REV_A, "*"], ["tree", "t2", "r/t2", REV_A, "*"]], log);
    expect(readFileSync(log, "utf8").trim().split("\n")).toEqual(["t1", "t2"]);
    expect(readFileSync(join(dir, "t1/.fv-revision"), "utf8")).toBe(REV_A);
    // Warm: both present at the pinned revision -> nothing fetched.
    writeFileSync(log, "");
    run(dir, [["tree", "t1", "r/t1", REV_A, "*"], ["tree", "t2", "r/t2", REV_A, "*"]], log);
    expect(readFileSync(log, "utf8")).toBe("");
    // t2 pinned to a new revision; t3 a leftover with no marks; an interrupted download's partial folder.
    mkdirSync(join(dir, "t3"));
    mkdirSync(join(dir, ".t4.partial-123"));
    run(dir, [["tree", "t1", "r/t1", REV_A, "*"], ["tree", "t2", "r/t2", REV_B, "*"], ["tree", "t3", "r/t3", REV_A, "*"]], log);
    expect(readFileSync(log, "utf8").trim().split("\n")).toEqual(["t2", "t3"]);
    expect(readFileSync(join(dir, "t2/.fv-revision"), "utf8")).toBe(REV_B);
    expect(readFileSync(join(dir, "t2/model.bin"), "utf8").trim()).toBe(REV_B);
    expect(readdirSync(dir).filter((f) => f.startsWith("."))).toEqual([]);
    expect(existsSync(join(dir, "t1/.complete"))).toBe(true);
  });
  it("a tree with .complete but no revision mark (or none at all) is fetched again", () => {
    const dir = mkdtempSync(join(tmpdir(), "fvw-"));
    const log = join(dir, "..", `fetched-b-${Date.now()}.log`);
    mkdirSync(join(dir, "t1"));
    writeFileSync(join(dir, "t1/.complete"), "1\n");
    run(dir, [["tree", "t1", "r/t1", REV_A, "*"]], log);
    expect(readFileSync(log, "utf8").trim()).toBe("t1");
  });
});

describe("instance types (the live list)", () => {
  it("parses sizes, durations, prices, stoppable; storage price errs high", () => {
    expect(parseGiB("1TiB226GiB")).toBe(1250);
    expect(parseGiB("850GiB")).toBe(850);
    expect(parseGiB("0B")).toBeNull();
    expect(parseDuration("6m30s")).toBe(390);
    const t = toType({ type: "x", stoppable: true, base_price: { amount: "2.5" }, supported_storage: [{ price_per_gb_hr: { amount: "0.000132" } }, { price_per_gb_hr: { amount: "0.000205" } }], elastic_root_volume: true, cloud_cred_id: "c" });
    expect([t.stoppable, t.usd_per_hr, t.storage_usd_per_gb_hr, t.elastic_disk, t.cloud_cred_id]).toEqual([true, 2.5, 0.000205, true, "c"]);
    expect(toType({ type: "y", stoppable: null }).stoppable).toBe(false);
  });
  it("disk for the trees: × 1.3 + 40 GB, at least 200, in 50s", () => {
    expect(brevDiskGb([])).toBe(200);
    expect(brevDiskGb(["h3-base"])).toBe(250);
    expect(brevDiskGb(["h3-base", "ltx25"])).toBe(400);
  });
});

describe("park on stop, warm restart, fallbacks, limits", () => {
  it("a stoppable pod whose weights completed is stopped and parked with its trees; storage is booked", async () => {
    const env = mkEnv();
    const m = brevMock();
    const pod = await live(env, "w1");
    expect(await releaseBrev(env, pod, true)).toBe("parked");
    expect(m.calls).toContain("PUT workspaces/w1/stop");
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    const r = (await brevRow(env, pod))!;
    expect(r.state).toBe("parked");
    expect(JSON.parse(r.trees)).toEqual({ "wan22-ti2v-5b": REV_A });
    expect(storageDph(r)).toBeCloseTo(300 * 0.000132);
    const v = await parkedView(env);
    expect(v.parked[0]).toMatchObject({ name: "fv-pod-w1-1010000000", instance_type: "stop.x1", disk_gb: 300, trees: { "wan22-ti2v-5b": REV_A } });
    expect(v.storage_usd_per_day).toBeCloseTo(300 * 0.000132 * 24);
  });
  it("non-stoppable, no weights, weights never completed, or a non-stop release: deleted", async () => {
    const env = mkEnv();
    const m = brevMock();
    expect(await releaseBrev(env, await live(env, "n1", { type: "nostop.x1", stoppable: false }), true)).toBe("deleted");
    expect(await releaseBrev(env, await live(env, "n2", { trees: {} }), true)).toBe("deleted");
    expect(await releaseBrev(env, await live(env, "n3", { ready: false }), true)).toBe("deleted");
    expect(await releaseBrev(env, await live(env, "n4"), false)).toBe("deleted");
    expect(m.calls.filter((c) => c.startsWith("DELETE")).sort()).toEqual(["DELETE workspaces/n1", "DELETE workspaces/n2", "DELETE workspaces/n3", "DELETE workspaces/n4"]);
    expect(m.calls.some((c) => c.includes("/stop"))).toBe(false);
    // The ready mark can also be the pod's tunnel report (the boot reports it after the weights).
    const pod = await live(env, "n5", { ready: false });
    await putSetting(env, `endpoint:${pod}`, { phase: "tunnel", detail: "", at: 1 }, "pod");
    expect(await releaseBrev(env, pod, true)).toBe("parked");
  });
  it("keep-on-stop off by policy (brev_park_max 0): deleted", async () => {
    const env = mkEnv();
    brevMock();
    await putSetting(env, "policies", { ...DEFAULT_POLICIES, brev_park_max: 0 }, "t");
    expect(await releaseBrev(env, await live(env, "p0"), true)).toBe("deleted");
  });
  it("warm candidates: same type, a tree at the pinned revision, the disk fits; most trees first; claimed once", async () => {
    const env = mkEnv();
    brevMock();
    await putSetting(env, "policies", { ...DEFAULT_POLICIES, brev_park_max: 10 }, "t");
    await releaseBrev(env, await live(env, "a", { trees: { "wan22-ti2v-5b": REV_A } }), true);
    await releaseBrev(env, await live(env, "b", { trees: { "wan22-ti2v-5b": REV_A, "fastwan22-ti2v-5b": REV_B } }), true);
    await releaseBrev(env, await live(env, "c", { trees: { "wan22-ti2v-5b": REV_B } }), true);
    await releaseBrev(env, await live(env, "d", { trees: { "wan22-ti2v-5b": REV_A }, disk: 60 }), true);
    const want = { "wan22-ti2v-5b": REV_A, "fastwan22-ti2v-5b": REV_B };
    expect((await warmCandidates(env, "stop.x1", want)).map((c) => c.row.workspace_id)).toEqual(["b", "a"]);
    expect(await warmCandidates(env, "nostop.x1", want)).toEqual([]);
    const c1 = await claimWarm(env, "stop.x1", want);
    expect(c1?.row.workspace_id).toBe("b");
    expect(c1?.missing).toEqual([]);
    const c2 = await claimWarm(env, "stop.x1", want);
    expect([c2?.row.workspace_id, c2?.missing]).toEqual(["a", ["fastwan22-ti2v-5b"]]);
    expect(await claimWarm(env, "stop.x1", want)).toBeNull();
    expect(await claimWarm(env, "stop.x1", {})).toBeNull();
  });
  it("a failed restart is held (stopped, never restarted on its own) or deleted by policy; unpark returns it", async () => {
    const env = mkEnv();
    const m = brevMock({ startFail: true });
    await releaseBrev(env, await live(env, "h1"), true);
    const row = (await brevRow(env, "brev:fv-pod-h1-1010000000"))!;
    expect(await holdFailed(env, row, "no capacity")).toBe("held");
    expect((await brevRow(env, row.pod_id))!.state).toBe("held");
    expect(await warmCandidates(env, "stop.x1", { "wan22-ti2v-5b": REV_A })).toEqual([]);
    expect((await unpark(env, "h1")).state).toBe("parked");
    await putSetting(env, "policies", { ...DEFAULT_POLICIES, brev_park_delete_failed: true }, "t");
    expect(await holdFailed(env, (await brevRow(env, row.pod_id))!, "x")).toBe("deleted");
    expect(m.calls).toContain("DELETE workspaces/h1");
  });
  it("a restart that does not run within BREV_RESTART_TIMEOUT_S is fatal and replaced; a never-ready restart is held at release", async () => {
    const env = mkEnv({ BREV_RESTART_TIMEOUT_S: "60" });
    brevMock();
    const pod = await live(env, "r1", { ready: false });
    await env.DB.prepare("UPDATE brev_instances SET restarted_at = ? WHERE workspace_id = 'r1'").bind(Date.now() - 10_000).run();
    expect(await restartDiag(env, pod, "stopped")).toMatchObject({ phase: "restarting", fatal: false });
    await env.DB.prepare("UPDATE brev_instances SET restarted_at = ? WHERE workspace_id = 'r1'").bind(Date.now() - 120_000).run();
    expect(await restartDiag(env, pod, "starting")).toMatchObject({ phase: "restart timed out", fatal: true, replace: true });
    expect(await restartDiag(env, pod, "running")).toBeNull();
    expect(await releaseBrev(env, pod, false)).toBe("held");
  });
  it("limits: more than brev_park_max, or older than brev_park_max_days: the oldest of ours are deleted", async () => {
    const env = mkEnv();
    const m = brevMock();
    for (const id of ["l1", "l2", "l3"]) await releaseBrev(env, await live(env, id), true);
    // Parking the third went over brev_park_max 2: the oldest went at once.
    const states = async () => Object.fromEntries(((await env.DB.prepare("SELECT workspace_id, state FROM brev_instances").all<any>()).results || []).map((r: any) => [r.workspace_id, r.state]));
    await env.DB.prepare("UPDATE brev_instances SET parked_at = ? WHERE workspace_id = 'l1'").bind(1).run();
    await env.DB.prepare("UPDATE brev_instances SET parked_at = ? WHERE workspace_id = 'l2'").bind(2).run();
    expect(await enforceParkLimits(env, DEFAULT_POLICIES)).toEqual(expect.arrayContaining([expect.stringMatching(/parked longer than brev_park_max_days 7/)]));
    expect(await states()).toMatchObject({ l1: "deleted", l2: "deleted", l3: "parked" });
    expect(m.calls.filter((c) => c.startsWith("DELETE")).sort()).toEqual(["DELETE workspaces/l1", "DELETE workspaces/l2"]);
    await expect(deleteParked(env, "nope", "x")).rejects.toMatchObject({ status: 404 });
    expect(await deleteParked(env, "l3", "owner")).toMatch(/parked instance deleted/);
  });
  it("the budget counts parked storage", async () => {
    const env = mkEnv({ BREV_BUDGET_USD: "1" });
    brevMock();
    for (const id of ["s1", "s2"]) await releaseBrev(env, await live(env, id, { disk: 1000 }), true);
    const b = await budgetCheck(env, "brev", 0, 24 * 3);
    expect(b.parked_dph).toBeCloseTo(2 * 1000 * 0.000132);
    expect(b.ok).toBe(false);
    expect(b.reasons[0]).toMatch(/parked storage \$0\.264\/hr included/);
  });
});

describe("the boot endpoint", () => {
  const req = (tok: string) => new Request("https://fvc.test/ingest/v1/brev-boot", { headers: { authorization: `Bearer ${tok}` } });
  it("serves the live launch's run script to its token; refuses a parked one, a wrong token", async () => {
    const env = mkEnv();
    brevMock();
    const pod = await live(env, "b1");
    const r = await brevBoot(env, req("fvb_b1"));
    expect(await r.text()).toBe("#!/bin/bash\necho run b1");
    expect((await brevRow(env, pod))!.boots).toBe(1);
    await expect(brevBoot(env, req("fvb_nope"))).rejects.toMatchObject({ status: 401 });
    await releaseBrev(env, pod, true);
    await expect(brevBoot(env, req("fvb_b1"))).rejects.toMatchObject({ status: 409 });
    // The run script is sealed at rest.
    const row = await env.DB.prepare("SELECT run_sealed FROM brev_instances WHERE workspace_id = 'b1'").first<{ run_sealed: string }>();
    expect(row!.run_sealed).toMatch(/^v1\./);
    expect(row!.run_sealed).not.toContain("echo run");
  });
});

describe("warm / cold outlook (launch form, planner)", () => {
  it("cold on a stoppable type, warm once parked, the non-stoppable type flagged", async () => {
    const env = mkEnv({ FV_HUB_DOWNLOADS_APPROVED: "1" });
    brevMock();
    const preset = (await import("../../src/presets")).POOL_PRESETS.find((p) => p.id === "wan")!;
    const pool: any = { ...preset.pool, id: "pod", count: 1, provider: "brev", provider_gpu: "stop.x1", weights_source: "hub", hub_download_approved: true };
    const presetTrees = (await import("../../src/providers")).weightsPlan(pool).trees;
    expect(presetTrees).toEqual(["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    const cold = (await brevOutlook(env, pool))!;
    expect(cold.warm).toBe(false);
    expect(cold.note).toMatch(/^cold: downloads .* kept on stop/);
    const ns = (await brevOutlook(env, { ...pool, provider_gpu: "nostop.x1" }))!;
    expect(ns.note).toMatch(/nostop\.x1 is not stoppable: the weights download at every launch \(stoppable allowed types keep them: stop\.x1\)/);
    const { launchTrees, weightsPlan } = await import("../../src/providers");
    await releaseBrev(env, await live(env, "o1", { trees: launchTrees(weightsPlan(pool)), disk: 2000 }), true);
    const warm = (await brevOutlook(env, pool))!;
    expect(warm.warm).toBe(true);
    expect(warm.boot_s).toBeLessThan(cold.boot_s);
    expect(warm.note).toMatch(/^warm: restarts parked fv-pod-o1-1010000000/);
  });
});
