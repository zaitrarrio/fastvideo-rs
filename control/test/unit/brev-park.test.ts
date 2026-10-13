// NVIDIA Brev keep-on-stop (src/brev-park.ts, docs/serve/deploy-gmi-brev.md §3.4, §7): the weights loop of the
// boot (run in bash with a stub fetcher), the instance-type list, park / hold on stop (never delete), reuse, limits,
// the budget with parked storage, the boot endpoint; Brev's API is a fetch mock over a D1 shim.
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_POLICIES, policies } from "../../src/alerts";
import { brev, parseDuration, parseGiB, resetBrevTypes, toType } from "../../src/brev";
import {
  brevBoot,
  brevDiskGb,
  brevRow,
  brevRowById,
  claimWarm,
  deleteParked,
  fitsDisk,
  holdFailed,
  parkedView,
  parkLimitAlert,
  recordBrevCreate,
  releaseBrev,
  restartDiag,
  restartStuckAlerts,
  restartWarm,
  retryRefusedStops,
  storageDph,
  unclaimWarm,
  unpark,
  warmCandidates,
} from "../../src/brev-park";
import { deletePod } from "../../src/cluster/ops";
import { stopCluster } from "../../src/collector";
import { collectProviders } from "../../src/collector-providers";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { brevGpuOptions, brevNoStopNote, brevNoStopWarnings, brevOutlook, budgetCheck, WEIGHTS_SH } from "../../src/providers";
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

/** A Brev API stand-in: workspaces with a status, stop / start / delete, the list, the instance-type list. `o` is live
 * (a test flips stopFail: Brev refusing a stop, as for a type that is not stoppable). */
type MockWs = { status: string; name?: string; type?: string; statusMessage?: string };
type MockOpts = { startFail?: boolean; stopFail?: boolean; startNoop?: string };
let M: { ws: Map<string, MockWs>; calls: string[]; o: MockOpts } | null = null;
function brevMock(o: MockOpts = {}) {
  const ws = new Map<string, MockWs>();
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
      if (path === "organizations/org1/workspaces") return new Response(JSON.stringify([...ws].map(([id, w]) => ({ id, name: w.name ?? id, status: w.status, statusMessage: w.statusMessage ?? "", instanceType: w.type ?? "stop.x1" }))));
      const m = /^workspaces\/([^/]+)(?:\/(stop|start))?$/.exec(path);
      if (m) {
        const w = ws.get(m[1]!) || { status: "RUNNING" };
        ws.set(m[1]!, w);
        if (m[2] === "stop") {
          if (o.stopFail) return new Response(JSON.stringify({ message: "brev API error" }), { status: 400 });
          // Brev's answers measured live 2026-10-13: a workspace still DEPLOYING (never provisioned), and a RUNNING one
          // of a type Brev cannot stop (it stays RUNNING).
          if (w.status === "DEPLOYING") return new Response(JSON.stringify({ errors: [{ type: "InternalServerError", message: "rpc error: code = NotFound desc = instance not found" }] }), { status: 500 });
          if (w.type === "nostop.x1") return new Response(JSON.stringify({ errors: [{ type: "InternalServerError", message: `rpc error: code = Internal desc = instance "inst-${m[1]}" does not support stop` }] }), { status: 500 });
          w.status = "STOPPED";
        }
        if (m[2] === "start") {
          if (o.startFail) return new Response(JSON.stringify({ message: "no capacity" }), { status: 409 });
          // startNoop: Brev accepts the start but the VM stays STOPPED, with its reason in statusMessage
          if (o.startNoop) w.statusMessage = o.startNoop;
          else w.status = "RUNNING";
          return new Response(JSON.stringify({ id: m[1], name: w.name ?? m[1], status: w.status, statusMessage: w.statusMessage ?? "", instanceTypeInfo: { stoppable: true } }));
        }
        if (init.method === "GET" && !m[2]) return new Response(JSON.stringify({ id: m[1], name: w.name ?? m[1], status: w.status, statusMessage: w.statusMessage ?? "", instanceTypeInfo: { stoppable: true } }));
        if (init.method === "DELETE") ws.delete(m[1]!);
        return new Response("{}");
      }
      return new Response("[]");
    }),
  );
  M = { ws, calls, o };
  return M;
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
  M?.ws.set(id, { status: "RUNNING", name, type: o.type ?? "stop.x1" });
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

describe("every stop parks or holds (owner rule 2026-10-11: never deleted), reuse, limits as alerts", () => {
  const states = async (env: Env) => Object.fromEntries(((await env.DB.prepare("SELECT workspace_id, state FROM brev_instances").all<any>()).results || []).map((r: any) => [r.workspace_id, r.state]));
  it("a pod whose weights completed is stopped and parked with its trees; storage is booked", async () => {
    const env = mkEnv();
    const m = brevMock();
    const pod = await live(env, "w1");
    expect(await releaseBrev(env, pod)).toBe("parked");
    expect(m.calls).toContain("PUT workspaces/w1/stop");
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    const r = (await brevRow(env, pod))!;
    expect(r.state).toBe("parked");
    expect(JSON.parse(r.trees)).toEqual({ "wan22-ti2v-5b": REV_A });
    expect(storageDph(r)).toBeCloseTo(300 * 0.000132);
    const v = await parkedView(env);
    expect(v.parked[0]).toMatchObject({ name: "fv-pod-w1-1010000000", instance_type: "stop.x1", disk_gb: 300, trees: { "wan22-ti2v-5b": REV_A }, reuse: "reusable: the next launch of stop.x1" });
    expect(v.storage_usd_per_day).toBeCloseTo(300 * 0.000132 * 24);
    expect(v.policy).toMatchObject({ auto_delete: false, line: expect.stringMatching(/never deleted by fv-control \(owner rule 2026-10-11\).*alert past 2 parked or 7 days each/) });
  });
  it("no weights: parked; weights never completed: stopped and held with why (partial trees not recorded); a ready tunnel report counts", async () => {
    const env = mkEnv();
    const m = brevMock();
    expect(await releaseBrev(env, await live(env, "n2", { trees: {} }))).toBe("parked");
    expect(await releaseBrev(env, await live(env, "n3", { ready: false }), { why: "pod FAILED (host failed)" })).toBe("held");
    const h = (await brevRowById(env, "n3"))!;
    expect(h.note).toMatch(/stopped before its boot became ready \(pod FAILED \(host failed\)\): weights may be incomplete \(the next boot resumes them\)/);
    expect(JSON.parse(h.trees)).toEqual({});
    expect(m.calls.filter((c) => c.startsWith("DELETE"))).toEqual([]);
    expect(m.calls.filter((c) => c.endsWith("/stop")).sort()).toEqual(["PUT workspaces/n2/stop", "PUT workspaces/n3/stop"]);
    const pod = await live(env, "n5", { ready: false });
    await putSetting(env, `endpoint:${pod}`, { phase: "tunnel", detail: "", at: 1 }, "pod");
    expect(await releaseBrev(env, pod)).toBe("parked");
  });
  it("brev_park_max 0 no longer means delete on stop: parked, and an alert", async () => {
    const env = mkEnv();
    const m = brevMock();
    await putSetting(env, "policies", { ...DEFAULT_POLICIES, brev_park_max: 0 }, "t");
    const logs: string[] = [];
    expect(await releaseBrev(env, await live(env, "p0"), {}, (x) => logs.push(x))).toBe("parked");
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    expect(logs.some((l) => /WARNING: NVIDIA Brev: 1 parked \/ held, more than brev_park_max 0: fv-pod-p0-1010000000 \(parked, stop\.x1, 0\.0 d, \$0\.95\/day\).*Not deleted/.test(l))).toBe(true);
  });
  it("a stop Brev refuses (an API error) leaves it running, alerted with its $/hr; the cron retries; never deleted", async () => {
    const env = mkEnv();
    const m = brevMock({ stopFail: true });
    const pod = await live(env, "f1");
    await env.DB.prepare("UPDATE cluster_pods SET cost_per_hr = 1.5 WHERE pod_id = ?").bind(pod).run();
    const logs: string[] = [];
    expect(await releaseBrev(env, pod, {}, (x) => logs.push(x))).toBe("stop_failed");
    expect(logs.join("\n")).toMatch(/stop refused by Brev .*left RUNNING at \$1\.50\/hr.*never deletes a Brev VM on its own/);
    const r1 = await retryRefusedStops(env, new Map([["f1", "running"]]));
    expect(r1.alerts).toEqual([expect.objectContaining({ kind: "brev_stop_failed", severity: "critical", message: expect.stringMatching(/still RUNNING at \$1\.50\/hr \(\$36\.00\/day\)/) })]);
    expect((await parkedView(env)).parked.map((p) => [p.workspace_id, p.state, p.storage_usd_per_day])).toEqual([["f1", "stop_failed", 0]]);
    m.o.stopFail = false;
    const r2 = await retryRefusedStops(env, new Map([["f1", "running"]]));
    expect([r2.alerts, r2.actions]).toEqual([[], ["brev fv-pod-f1-1010000000: stopped (an earlier stop was refused): held"]]);
    expect((await states(env)).f1).toBe("held");
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
  });
  it("a type Brev cannot stop (500 \"does not support stop\"): running_unstoppable, released, one alert, stop retried hourly, billed in the ledger and budget, never deleted", async () => {
    const env = mkEnv();
    const m = brevMock();
    const remove = vi.spyOn(brev, "remove");
    const pod = await live(env, "u1", { type: "nostop.x1", stoppable: false });
    await env.DB.prepare("UPDATE cluster_pods SET cost_per_hr = 1.5 WHERE pod_id = ?").bind(pod).run();
    const logs: string[] = [];
    const stops = () => m.calls.filter((c) => c === "PUT workspaces/u1/stop").length;
    // The stop: released from its cluster (deletePod true: no blind retries by the down's verify), the VM still running.
    expect(await deletePod(env, pod, (x) => logs.push(x), "stop")).toBe(true);
    expect(stops()).toBe(1);
    const row = (await brevRowById(env, "u1"))!;
    expect([row.state, row.stoppable, row.usd_per_hr]).toEqual(["running_unstoppable", 0, 1.5]);
    expect(row.note).toMatch(/^not_stoppable: Brev cannot stop nostop\.x1 \(.*does not support stop.*\): left RUNNING at \$1\.50\/hr \(\$36\.00\/day\)/);
    expect(logs.join("\n")).toMatch(/STILL BILLING: fv-control never deletes a Brev VM on its own; delete it yourself: fv-control\.sh brev delete-parked u1/);
    expect(m.ws.get("u1")!.status).toBe("RUNNING");
    // A second stop (the down's verify, a retry): no stop call, still released.
    expect(await releaseBrev(env, pod, { why: "retry" })).toBe("running_unstoppable");
    expect(stops()).toBe(1);
    // The cron: one alert (the same key every pass), no stop until the backoff (BREV_UNSTOPPABLE_RETRY_S, 1 h) passed.
    const t0 = Date.now();
    const r1 = await retryRefusedStops(env, new Map([["u1", "RUNNING"]]), t0 + 60_000);
    const r2 = await retryRefusedStops(env, new Map([["u1", "RUNNING"]]), t0 + 120_000);
    expect(r1.alerts).toEqual([expect.objectContaining({ key: "brev_unstoppable:u1", kind: "brev_unstoppable", severity: "critical" })]);
    expect(r1.alerts[0]!.message).toMatch(/cannot be stopped .*still RUNNING at \$1\.50\/hr \(\$36\.00\/day\).*never deletes a Brev VM on its own: delete it yourself: fv-control\.sh brev delete-parked u1, or Delete in the Pods page's parked list/);
    expect(r2.alerts.map((a) => a.key)).toEqual(["brev_unstoppable:u1"]);
    expect(stops()).toBe(1);
    const r3 = await retryRefusedStops(env, new Map([["u1", "RUNNING"]]), t0 + 3700_000);
    expect(stops()).toBe(2);
    expect(r3.alerts.map((a) => a.key)).toEqual(["brev_unstoppable:u1"]);
    expect((await brevRowById(env, "u1"))!.stop_tried_at).toBe(t0 + 3700_000);
    await retryRefusedStops(env, new Map([["u1", "RUNNING"]]), t0 + 3800_000);
    expect(stops()).toBe(2);
    // The Pods page: running at its GPU $/hr, no storage.
    const pv = await parkedView(env);
    expect(pv.parked.map((p) => [p.workspace_id, p.state, p.storage_usd_per_day, p.running_usd_per_day])).toEqual([["u1", "running_unstoppable", 0, 36]]);
    expect(pv.running_usd_per_day).toBe(36);
    expect(pv.parked[0]!.reuse).toMatch(/can't|cannot stop nostop\.x1/);
    // The budget: its $/hr counted BREV_NOSTOP_MARGIN_H (24 h) past the deadline.
    const b = await budgetCheck(env, "brev", 0, 1);
    expect([b.nostop_dph, b.margin_h]).toEqual([1.5, 24]);
    expect(b.reasons[0]).toMatch(/\+ \$1\.50\/hr that a stop cannot end .* for 24 h past the deadline/);
    // The collector: billed at its full $/hr (owner brev:unstoppable, no orphan alert), its alert, the budget warned early.
    const res = await collectProviders(env, Date.now(), 60_000, DEFAULT_POLICIES, []);
    expect(res.alerts.map((a) => a.kind)).toEqual(expect.arrayContaining(["brev_unstoppable", "brev_budget"]));
    expect(res.alerts.some((a) => a.kind === "brev_orphan")).toBe(false);
    expect(res.alerts.find((a) => a.kind === "brev_budget")).toMatchObject({ severity: "warn", message: expect.stringMatching(/\$1\.50\/hr that the budget floor cannot stop for 24 h .*warned early.*1 Brev VM at \$1\.50\/hr \(\$36\.00\/day\) can't be stopped/) });
    expect(res.kinds).toContain("brev_unstoppable");
    const led = await env.DB.prepare("SELECT owner, usd FROM cost_daily WHERE pod_id = ?").bind(pod).first<{ owner: string; usd: number }>();
    expect(led!.owner).toBe("brev:unstoppable");
    expect(led!.usd).toBeCloseTo(1.5 / 60, 6);
    const pr = await env.DB.prepare("SELECT desired_status, cost_per_hr FROM pods WHERE pod_id = ?").bind(pod).first<{ desired_status: string; cost_per_hr: number }>();
    expect(pr).toEqual({ desired_status: "RUNNING", cost_per_hr: 1.5 });
    expect((await budgetCheck(env, "brev", 0, 1)).running_dph).toBe(1.5);
    // Nothing deleted it; the owner's delete does.
    expect(remove).not.toHaveBeenCalled();
    expect(await deleteParked(env, "u1", "owner")).toMatch(/running instance deleted/);
    expect((await states(env)).u1).toBe("deleted");
    remove.mockRestore();
  });
  it("a stop while DEPLOYING (500 \"instance not found\"): stop_failed, sent again once it left DEPLOYING", async () => {
    const env = mkEnv();
    const m = brevMock();
    const pod = await live(env, "dp1", { ready: false });
    m.ws.get("dp1")!.status = "DEPLOYING";
    const logs: string[] = [];
    expect(await deletePod(env, pod, (x) => logs.push(x), "stop")).toBe(false);
    expect((await brevRowById(env, "dp1"))!.state).toBe("stop_failed");
    expect(logs.join("\n")).toMatch(/instance not found.*still DEPLOYING: the stop is sent again once it leaves DEPLOYING/);
    const stops = () => m.calls.filter((c) => c === "PUT workspaces/dp1/stop").length;
    // Still DEPLOYING: no stop sent, alerted.
    const r1 = await retryRefusedStops(env, new Map([["dp1", "DEPLOYING"]]));
    expect(stops()).toBe(1);
    expect(r1.alerts).toEqual([expect.objectContaining({ kind: "brev_stop_failed", message: expect.stringMatching(/still DEPLOYING .*sent once it leaves DEPLOYING/) })]);
    // Provisioned: the stop goes and takes.
    m.ws.get("dp1")!.status = "RUNNING";
    const r2 = await retryRefusedStops(env, new Map([["dp1", "RUNNING"]]));
    expect(stops()).toBe(2);
    expect([r2.alerts, (await states(env)).dp1]).toEqual([[], "held"]);
    // A non-stoppable type caught DEPLOYING: once provisioned its stop says "does not support stop": running_unstoppable.
    const p2 = await live(env, "dp2", { type: "nostop.x1", stoppable: false, ready: false });
    m.ws.get("dp2")!.status = "DEPLOYING";
    expect(await deletePod(env, p2, () => {}, "stop")).toBe(false);
    m.ws.get("dp2")!.status = "RUNNING";
    const r3 = await retryRefusedStops(env, new Map([["dp2", "RUNNING"]]));
    expect((await states(env)).dp2).toBe("running_unstoppable");
    expect(r3.alerts.map((a) => a.kind)).toEqual(["brev_unstoppable"]);
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
  });
  it("reuse: any parked / held instance of the type with the disk for the trees; most trees first, parked before held; a held one after the cooldown", async () => {
    const env = mkEnv();
    brevMock();
    await releaseBrev(env, await live(env, "a", { trees: { "wan22-ti2v-5b": REV_A } }));
    await releaseBrev(env, await live(env, "b", { trees: { "wan22-ti2v-5b": REV_A, "fastwan22-ti2v-5b": REV_B } }));
    await releaseBrev(env, await live(env, "c", { trees: { "wan22-ti2v-5b": REV_B } }));
    await releaseBrev(env, await live(env, "d", { trees: { "wan22-ti2v-5b": REV_A }, disk: 60 }));
    await releaseBrev(env, await live(env, "e", { ready: false, trees: { "h3-base": REV_A }, disk: 220 }), { why: "failed boot" });
    const want = { "wan22-ti2v-5b": REV_A, "fastwan22-ti2v-5b": REV_B };
    // d's disk is too small; c holds no tree at the pinned revision but is reusable (the trees download onto it); e is
    // held too recently (BREV_HELD_REUSE_AFTER_S 900).
    expect((await warmCandidates(env, "stop.x1", want)).map((c) => c.row.workspace_id)).toEqual(["b", "a", "c"]);
    expect(await warmCandidates(env, "nostop.x1", want)).toEqual([]);
    const later = Date.now() + 901_000;
    // After the cooldown the held one qualifies too (after the parked ones with as many trees), if its disk fits
    // the trees it holds (h3-base, from its unfinished launch) plus the wanted ones: 144 + 35 + 25 + 40 > 220, no.
    expect((await warmCandidates(env, "stop.x1", want, later)).map((c) => c.row.workspace_id)).toEqual(["b", "a", "c"]);
    // e's h3-base never completed (not in its trees): it matches nothing and ranks after the parked ones as held.
    const forH3 = (await warmCandidates(env, "stop.x1", { "h3-base": REV_A }, later)).map((c) => c.row.workspace_id);
    expect([forH3.slice(0, 3).sort(), forH3[3]]).toEqual([["a", "b", "c"], "e"]);
    expect(fitsDisk({ disk_gb: 300, trees: "{}", launch_trees: JSON.stringify({ "h3-base": REV_A }) }, { "wan22-ti2v-5b": REV_A })).toBe(true);
    expect(fitsDisk({ disk_gb: 200, trees: JSON.stringify({ "h3-base": REV_A }), launch_trees: "{}" }, { "wan22-ti2v-5b": REV_A })).toBe(false);
    const c1 = await claimWarm(env, "stop.x1", want);
    expect([c1?.row.workspace_id, c1?.missing]).toEqual(["b", []]);
    const c2 = await claimWarm(env, "stop.x1", want);
    expect([c2?.row.workspace_id, c2?.missing]).toEqual(["a", ["fastwan22-ti2v-5b"]]);
    const c3 = await claimWarm(env, "stop.x1", want);
    expect([c3?.row.workspace_id, c3?.matched, c3?.missing]).toEqual(["c", [], ["wan22-ti2v-5b", "fastwan22-ti2v-5b"]]);
    expect(await claimWarm(env, "stop.x1", want)).toBeNull();
    // A launch without weights reuses one too; a claim given back is parked / held as before.
    await unclaimWarm(env, c3!.row);
    expect((await brevRowById(env, "c"))!.state).toBe("parked");
    expect((await claimWarm(env, "stop.x1", {}))?.row.workspace_id).toBe("c");
    // The held one, once its cooldown passed, is claimed (and given back as held).
    await env.DB.prepare("UPDATE brev_instances SET updated_at = ? WHERE workspace_id = 'e'").bind(Date.now() - 901_000).run();
    const ce = await claimWarm(env, "stop.x1", { "h3-base": REV_A });
    expect([ce?.row.workspace_id, ce?.row.state, ce?.row.note]).toEqual(["e", "held", expect.stringMatching(/failed boot/)]);
    await unclaimWarm(env, ce!.row);
    expect((await brevRowById(env, "e"))!.state).toBe("held");
  });
  it("a failed restart is stopped and held (never deleted); unpark makes it reusable at once; brev_park_delete_failed is gone", async () => {
    const env = mkEnv();
    const m = brevMock({ startFail: true });
    await releaseBrev(env, await live(env, "h1"));
    const row = (await brevRow(env, "brev:fv-pod-h1-1010000000"))!;
    expect(await holdFailed(env, row, "no capacity")).toBe("held");
    expect((await brevRow(env, row.pod_id))!.state).toBe("held");
    expect(await warmCandidates(env, "stop.x1", { "wan22-ti2v-5b": REV_A })).toEqual([]);
    expect((await unpark(env, "h1")).state).toBe("parked");
    // A stored brev_park_delete_failed (removed) is ignored: still held, not deleted.
    await putSetting(env, "policies", { ...DEFAULT_POLICIES, brev_park_delete_failed: true }, "t");
    expect("brev_park_delete_failed" in (await policies(env))).toBe(false);
    expect(await holdFailed(env, (await brevRow(env, row.pod_id))!, "x")).toBe("held");
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
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
    expect(await releaseBrev(env, pod, { why: "restart timed out" })).toBe("held");
    expect((await brevRowById(env, "r1"))!.note).toMatch(/^restart failed: it did not come up after the restart \(restart timed out\)/);
  });
  it("a warm restart starts only a STOPPED workspace (waits out STOPPING, skips a running one) and keeps Brev's answer", async () => {
    const env = mkEnv({ BREV_STOPPING_WAIT_S: "0" });
    const m = brevMock();
    const pod = await live(env, "w1", { ready: false });
    m.ws.get("w1")!.status = "STOPPED";
    const x = { run: "#!/bin/bash\necho x", launch_trees: {}, cluster_id: "c1" };
    const said = await restartWarm(env, (await brevRow(env, pod))!, x);
    expect(said).toMatch(/^start accepted \(was STOPPED; Brev answered RUNNING\)/);
    expect(m.calls.filter((c) => /workspaces\/w1/.test(c))).toEqual(["GET workspaces/w1", "PUT workspaces/w1/start"]);
    expect((await brevRow(env, pod))!).toMatchObject({ state: "live", note: said });
    // already running (a start that took a while ago): not started twice
    expect(await restartWarm(env, (await brevRow(env, pod))!, x)).toMatch(/^already RUNNING: not started again/);
    expect(m.calls.filter((c) => c === "PUT workspaces/w1/start")).toHaveLength(1);
    // still STOPPING after the wait, or another status: refused with Brev's status (the launch holds it and moves on)
    m.ws.get("w1")!.status = "STOPPING";
    await expect(restartWarm(env, (await brevRow(env, pod))!, x)).rejects.toThrow(/brev fv-pod-w1-1010000000 is STOPPING: only a STOPPED workspace is started/);
  });
  it("a start Brev accepts but leaves STOPPED: every poll shows Brev's status and message, the start goes once more, then an alert", async () => {
    const env = mkEnv({ BREV_RESTART_RETRY_S: "60", BREV_RESTART_ALERT_S: "120", BREV_RESTART_TIMEOUT_S: "900" });
    const m = brevMock({ startNoop: "insufficient capacity for stop.x1 in us-east-1a" });
    const pod = await live(env, "s9", { ready: false });
    m.ws.get("s9")!.status = "STOPPED";
    const said = await restartWarm(env, (await brevRow(env, pod))!, { run: "#!/bin/bash\necho x", launch_trees: {}, cluster_id: "c1" });
    expect(said).toBe("start accepted (was STOPPED; Brev answered STOPPED: insufficient capacity for stop.x1 in us-east-1a)");
    const says = "STOPPED: insufficient capacity for stop.x1 in us-east-1a";
    const d1 = await restartDiag(env, pod, "stopped", says);
    expect(d1).toMatchObject({ phase: "restarting", fatal: false });
    expect(d1!.detail).toMatch(/starting \(\d+ s\); Brev: STOPPED: insufficient capacity/);
    expect(m.calls.filter((c) => c === "PUT workspaces/s9/start")).toHaveLength(1);
    await env.DB.prepare("UPDATE brev_instances SET restarted_at = ? WHERE workspace_id = 's9'").bind(Date.now() - 90_000).run();
    expect((await restartDiag(env, pod, "stopped", says))!.detail).toMatch(/still STOPPED 90 s after its start: start sent again \(Brev answered STOPPED: insufficient capacity/);
    await restartDiag(env, pod, "stopped", says);
    expect(m.calls.filter((c) => c === "PUT workspaces/s9/start"), "the start is sent once more, not every poll").toHaveLength(2);
    expect(await restartStuckAlerts(env, new Map([["s9", says]]))).toEqual([]); // 90 s < BREV_RESTART_ALERT_S
    await env.DB.prepare("UPDATE brev_instances SET restarted_at = ? WHERE workspace_id = 's9'").bind(Date.now() - 400_000).run();
    const a = await restartStuckAlerts(env, new Map([["s9", says]]));
    expect(a).toHaveLength(1);
    expect(a[0]).toMatchObject({ kind: "brev_restart_stuck", key: "brev_restart_stuck:s9", target: pod });
    expect(a[0]!.message).toMatch(/is still STOPPED: insufficient capacity .* 7 min after fv-control started it/);
    expect(await restartStuckAlerts(env, new Map([["s9", "RUNNING"]]))).toEqual([]);
    expect(m.calls.some((c) => c.startsWith("DELETE")), "never deleted").toBe(false);
  });
  it("limits: an alert listing the instances, their age and $/day; nothing deleted; the owner's delete works", async () => {
    const env = mkEnv();
    const m = brevMock();
    for (const id of ["l1", "l2", "l3"]) await releaseBrev(env, await live(env, id));
    await env.DB.prepare("UPDATE brev_instances SET parked_at = ? WHERE workspace_id = 'l1'").bind(Date.now() - 8 * 86400_000).run();
    const a = (await parkLimitAlert(env, DEFAULT_POLICIES))!;
    expect(a).toMatchObject({ kind: "brev_parked_over_limit", severity: "warn" });
    expect(a.message).toMatch(/3 parked \/ held, more than brev_park_max 2; 1 parked longer than brev_park_max_days 7: .*fv-pod-l1-1010000000 \(parked, stop\.x1, 8\.0 d, \$0\.95\/day\)/);
    expect(a.message).toMatch(/storage \$2\.85\/day in all\. Not deleted/);
    expect(await parkLimitAlert(env, { ...DEFAULT_POLICIES, brev_park_max: 5, brev_park_max_days: 30 })).toBeNull();
    expect(await states(env)).toEqual({ l1: "parked", l2: "parked", l3: "parked" });
    expect((await parkedView(env)).over_limit).toBe(true);
    expect(m.calls.some((c) => c.startsWith("DELETE"))).toBe(false);
    await expect(deleteParked(env, "nope", "x")).rejects.toMatchObject({ status: 404 });
    expect(await deleteParked(env, "l3", "owner")).toMatch(/parked instance deleted/);
    expect(m.calls.filter((c) => c.startsWith("DELETE"))).toEqual(["DELETE workspaces/l3"]);
  });
  it("the budget counts parked storage", async () => {
    const env = mkEnv({ BREV_BUDGET_USD: "1" });
    brevMock();
    for (const id of ["s1", "s2"]) await releaseBrev(env, await live(env, id, { disk: 1000 }));
    const b = await budgetCheck(env, "brev", 0, 24 * 3);
    expect(b.parked_dph).toBeCloseTo(2 * 1000 * 0.000132);
    expect(b.ok).toBe(false);
    expect(b.reasons[0]).toMatch(/parked storage \$0\.264\/hr included/);
  });
});

describe("owner rule 2026-10-11: no automatic path deletes a Brev VM", () => {
  it("stop, scale-down, roll, orphan, failed boot, failed restart, no record, refused stop, deadline / budget backstop, limits cron: brev.remove never called; only the owner's deletes call it", async () => {
    const env = mkEnv({ BREV_BUDGET_USD: "1" });
    const m = brevMock();
    const remove = vi.spyOn(brev, "remove");
    const logs: string[] = [];
    const L = (x: string) => logs.push(x);
    const cluster = (pod: string) => ({ id: "c1", name: "cl1", source: "standalone", spec: {}, state: { workers: { pod: [{ pod }] } } }) as any;
    // The `down` / scale / roll / up paths all release through deletePod (cluster/do.ts).
    for (const [id, why, o] of [
      ["d1", "stop", {}],
      ["d2", "deadline", {}],
      ["d3", "scale-down", {}],
      ["d4", "roll", {}],
      ["d5", "roll aborted", { ready: false }],
      ["d6", "orphan", { ready: false }],
      ["d7", "pod FAILED (host failed)", { ready: false }],
      ["d8", "retry", { trees: {} }],
    ] as const)
      expect(await deletePod(env, await live(env, id, o), L, why)).toBe(true);
    // A warm restart that never came up (restart timed out), and a failed start.
    const r9 = await live(env, "d9", { ready: false });
    await env.DB.prepare("UPDATE brev_instances SET restarted_at = ? WHERE workspace_id = 'd9'").bind(Date.now() - 3600_000).run();
    expect(await deletePod(env, r9, L, "restart timed out")).toBe(true);
    expect(await holdFailed(env, (await brevRowById(env, "d1"))!, "no capacity")).toBe("held");
    // A Brev pod with no keep-on-stop record (made before it): stopped and recorded held.
    M!.ws.set("old1", { status: "RUNNING", name: "fv-pod-old-0101000000", type: "stop.x1" });
    expect(await deletePod(env, "brev:fv-pod-old-0101000000", L, "stop")).toBe(true);
    expect((await brevRowById(env, "old1"))!.state).toBe("held");
    // Brev refuses the stop (an API error): left running, alerted, not deleted.
    m.o.stopFail = true;
    expect(await deletePod(env, await live(env, "d10"), L, "stop")).toBe(false);
    m.o.stopFail = false;
    // A type Brev cannot stop (owner decision 2026-10-13: launched like any other): every stop path leaves it running
    // (running_unstoppable), released, never deleted.
    for (const [id, why, o] of [
      ["n1", "stop", {}],
      ["n2", "deadline", {}],
      ["n3", "scale-down", {}],
      ["n4", "roll", {}],
      ["n5", "pod FAILED (host failed)", { ready: false }],
    ] as const)
      expect(await deletePod(env, await live(env, id, { ...o, type: "nostop.x1", stoppable: false }), L, why)).toBe(true);
    const p15 = await live(env, "n6", { type: "nostop.x1", stoppable: false });
    expect(await stopCluster(env, cluster(p15), "brev budget")).toMatch(/directly/);
    // The deadline backstop / budget floor with the operations DO unreachable: stopCluster's direct release.
    const p11 = await live(env, "d11");
    expect(await stopCluster(env, cluster(p11), "deadline")).toMatch(/directly/);
    // The cron: limits far over (0 / 0 days), the refused stop (still refused), the budget floor stopping a live pod.
    const p12 = await live(env, "d12");
    await env.DB.prepare("INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, 'c1', 'pod:cl1', 5, 1, 0)").bind(new Date().toISOString().slice(0, 10), p12).run();
    m.o.stopFail = true;
    const res = await collectProviders(env, Date.now(), 60_000, { ...DEFAULT_POLICIES, brev_park_max: 0, brev_park_max_days: 0, stop_on_floor: true }, [cluster(p12)]);
    m.o.stopFail = false;
    expect(res.alerts.map((a) => a.kind)).toEqual(expect.arrayContaining(["brev_parked_over_limit", "brev_stop_failed", "brev_unstoppable", "brev_budget"]));
    expect(res.actions.some((a) => /directly \(brev budget\)/.test(a))).toBe(true);
    // Nothing above deleted anything.
    expect(remove).not.toHaveBeenCalled();
    expect(m.calls.filter((c) => c.startsWith("DELETE"))).toEqual([]);
    const st = Object.fromEntries(((await env.DB.prepare("SELECT workspace_id, state FROM brev_instances").all<any>()).results || []).map((r: any) => [r.workspace_id, r.state]));
    expect(Object.values(st).every((s) => ["parked", "held", "stop_failed", "running_unstoppable", "live"].includes(s as string))).toBe(true);
    expect(st).toMatchObject({ d1: "held", d2: "parked", d3: "parked", d4: "parked", d5: "held", d6: "held", d7: "held", d8: "parked", d9: "held", d10: "stop_failed", d11: "parked", old1: "held" });
    expect(["n1", "n2", "n3", "n4", "n5", "n6"].map((k) => st[k])).toEqual(Array(6).fill("running_unstoppable"));
    // The owner's explicit deletes are the only ones: delete-parked (API / CLI / UI), and `pod delete` of a live VM.
    await deleteParked(env, "d2", "deleted by owner");
    expect(await deletePod(env, await live(env, "d13"), L, "delete", { manual: true })).toBe(true);
    expect(remove.mock.calls.map((c) => c[1])).toEqual(["d2", "d13"]);
    remove.mockRestore();
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

describe("warm / cold outlook (launch form, planner), non-stoppable types allowed with a warning", () => {
  it("cold on a stoppable type, warm once parked, reuse of one holding other trees; the non-stoppable type offered with what it bills after a stop", async () => {
    const env = mkEnv({ FV_HUB_DOWNLOADS_APPROVED: "1" });
    brevMock();
    const preset = (await import("../../src/presets")).POOL_PRESETS.find((p) => p.id === "wan")!;
    const pool: any = { ...preset.pool, id: "pod", count: 1, provider: "brev", provider_gpu: "stop.x1", weights_source: "hub", hub_download_approved: true };
    const presetTrees = (await import("../../src/providers")).weightsPlan(pool).trees;
    expect(presetTrees).toEqual(["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    const cold = (await brevOutlook(env, pool))!;
    expect(cold.warm).toBe(false);
    expect(cold.note).toMatch(/^cold: downloads .* kept on stop/);
    // Owner decision 2026-10-13: a type Brev cannot stop is offered and allowed, with a warning (no refusal).
    const ns = (await brevOutlook(env, { ...pool, provider_gpu: "nostop.x1" }))!;
    expect([ns.warm, ns.stoppable]).toEqual([false, false]);
    expect(ns.note).toMatch(/^cold: downloads .*; WARNING: nostop\.x1 can't be stopped on NVIDIA Brev: a stop .* leaves the VM running at \$1\.50\/hr \(\$36\.00\/day\) until you delete it .*24 h past the deadline/);
    expect(await brevNoStopNote(env, "stop.x1")).toBeNull();
    expect(await brevNoStopWarnings(env, { pools: [pool, { ...pool, id: "b", provider_gpu: "nostop.x1" }] })).toEqual([{ path: ["pools", 1, "provider_gpu"], message: expect.stringMatching(/^WARNING: nostop\.x1 can't be stopped/) }]);
    // The launch form lists every allowed type, stoppable first; the other says what it bills after a stop.
    const opts = await brevGpuOptions(env);
    expect(opts.map((o) => [o.id, o.stoppable])).toEqual([["stop.x1", true], ["nostop.x1", false]]);
    expect(opts[1]!.detail).toMatch(/\$1\.50\/hr · WARNING: can't be stopped: a stop leaves it running at \$1\.50\/hr \(\$36\.00\/day\): delete it yourself/);
    // The launch's budget check: a non-stoppable pod counts BREV_NOSTOP_MARGIN_H past the deadline (2 h + 24 h × $1.50).
    const bc = await budgetCheck(env, "brev", 1.5, 2, { nostopDph: 1.5 });
    expect([bc.ok, bc.projected]).toEqual([false, 39]);
    expect(bc.reasons[0]).toMatch(/for 2\.00 h \+ \$1\.50\/hr that a stop cannot end .* for 24 h past the deadline \(BREV_NOSTOP_MARGIN_H\) = \$39\.00, over the budget \$10/);
    expect((await budgetCheck(mkEnv({ BREV_NOSTOP_MARGIN_H: "2" }), "brev", 1.5, 2, { nostopDph: 1.5 })).projected).toBe(6);
    // A parked instance holding other trees: reused (the trees download onto it).
    await releaseBrev(env, await live(env, "o0", { trees: { ltx2: REV_A }, disk: 2000 }));
    const reuse = (await brevOutlook(env, pool))!;
    expect([reuse.warm, reuse.instance]).toEqual([true, "fv-pod-o0-1010000000"]);
    expect(reuse.note).toMatch(/^reuse: restarts parked fv-pod-o0-1010000000 \(none of the trees at the pinned revisions on it; downloads /);
    const { launchTrees, weightsPlan } = await import("../../src/providers");
    await releaseBrev(env, await live(env, "o1", { trees: launchTrees(weightsPlan(pool)), disk: 2000 }));
    const warm = (await brevOutlook(env, pool))!;
    expect(warm.warm).toBe(true);
    expect(warm.boot_s).toBeLessThan(cold.boot_s);
    expect(warm.note).toMatch(/^warm: restarts parked fv-pod-o1-1010000000/);
  });
});

describe("reuse across weights sources (r2 and hub)", () => {
  it("a VM parked with Hub-fetched trees is warm for an r2 launch of the same trees (same revisions)", async () => {
    const { HUB_TREES } = await import("../../src/weights-sources");
    const { launchTrees, weightsPlan, weightsSource } = await import("../../src/providers");
    const preset = (await import("../../src/presets")).POOL_PRESETS.find((p) => p.id === "wan")!;
    const trees = ["fastwan22-ti2v-5b", "wan22-ti2v-5b"];
    // Both trees published in R2 at their Hub revisions (as publish-hub writes them).
    const pub = Object.fromEntries(trees.map((t) => [t, { rev: HUB_TREES[t]!.revision, files: 1, bytes: 30e9, manifest_sha256: "c".repeat(64), source: "hub" }]));
    const env = mkEnv({ FV_HUB_DOWNLOADS_APPROVED: "1", WEIGHTS_R2_EXTRA_TREES: JSON.stringify(pub), WEIGHTS_R2_ACCESS_KEY_ID: "k", WEIGHTS_R2_SECRET_ACCESS_KEY: "s" });
    brevMock();
    const r2pool: any = { ...preset.pool, id: "pod", count: 1, provider: "brev", provider_gpu: "stop.x1" };
    const hubPool: any = { ...r2pool, weights_source: "hub", hub_download_approved: true };
    expect(weightsSource(r2pool, env)).toBe("r2");
    const r2want = launchTrees(weightsPlan(r2pool, env), env);
    const hubWant = launchTrees(weightsPlan(hubPool, env), env);
    expect(r2want).toEqual(hubWant);
    await releaseBrev(env, await live(env, "h1", { trees: hubWant, disk: 2000 }));
    const [c] = await warmCandidates(env, "stop.x1", r2want);
    expect([c!.row.workspace_id, [...c!.matched].sort(), c!.missing]).toEqual(["h1", trees, []]);
    const o = (await brevOutlook(env, r2pool))!;
    expect([o.warm, o.instance, o.download_gb]).toEqual([true, "fv-pod-h1-1010000000", 0]);
    expect(o.note).toMatch(/^warm: restarts parked fv-pod-h1-1010000000 \(holds /);
  });
});
