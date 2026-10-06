// The log explorer's API (src/logquery.ts) and the cluster editor's checks
// (src/cluster/editor.ts) over node:sqlite D1 and an in-memory R2.
import { afterEach, describe, expect, it, vi } from "vitest";
import { availability, checkSpec, clearStockCache, pathOf } from "../../src/cluster/editor";
import { defaultSpec, normalizeSpec } from "../../src/cluster/spec";
import type { Env } from "../../src/env";
import {
  archiveHours,
  beyond,
  decodeCursor,
  decodeTail,
  encodeCursor,
  encodeTail,
  inferLevel,
  lineContext,
  logFacets,
  parseLogQuery,
  parseTime,
  queryLogs,
  regexLiteral,
  runpodLines,
  tailLogs,
  type XLine,
} from "../../src/logquery";
import { d1 } from "./d1shim";

const T0 = Date.UTC(2026, 9, 6, 12, 0, 0);
const NOW = T0 + 3600_000;

class R2 {
  objs = new Map<string, string>();
  async put(k: string, v: string) {
    this.objs.set(k, v);
  }
  async get(k: string) {
    const v = this.objs.get(k);
    return v === undefined ? null : { text: async () => v, arrayBuffer: async () => new TextEncoder().encode(v).buffer };
  }
  async list(o: { prefix: string }) {
    return { objects: [...this.objs.keys()].filter((k) => k.startsWith(o.prefix)).sort().map((key) => ({ key })), truncated: false };
  }
}

async function seed() {
  const DB = d1();
  const env = { DB, LOGS: new R2(), RUNPOD_API_KEY: "rpa_TESTKEY_0123456789", RUNPOD_GRAPHQL: "https://rp.test/graphql" } as unknown as Env & { LOGS: R2 };
  const run = (sql: string, ...a: unknown[]) => DB.prepare(sql).bind(...a).run();
  await run("INSERT INTO clusters (id, name, spec, created_at, updated_at, created_by) VALUES ('c_a', 'alpha', '{}', 0, 0, 't'), ('c_b', 'beta', '{}', 0, 0, 't')");
  for (const [pod, cl, pool] of [
    ["poda000000001x", "c_a", "ltx"],
    ["podb000000002x", "c_a", "h3-turbo"],
    ["podc000000003x", "c_b", "fake"],
  ])
    await run("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, created_at) VALUES (?, ?, 'worker', ?, 0)", pod, cl, pool);
  await run("INSERT INTO pods (pod_id, name, owner, first_seen, last_seen) VALUES ('poda000000001x', 'fv-ctl-alpha-ltx-1', 'cluster:alpha', 0, 0)");
  // 60 pod lines, one a second, round-robin over the three pods; every 7th a warn, every 10th an error.
  const levels = (i: number) => (i % 10 === 9 ? "error" : i % 7 === 6 ? "warn" : i % 5 === 4 ? "debug" : "info");
  const pods = ["poda000000001x", "podb000000002x", "podc000000003x"];
  const cls = ["c_a", "c_a", "c_b"];
  for (let i = 0; i < 60; i++)
    await run(
      "INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES (?, ?, ?, ?, ?, ?, ?)",
      cls[i % 3],
      pods[i % 3],
      T0 + i * 1000,
      levels(i),
      "fastvideo_serve::app",
      i % 4 === 0 ? `job_${i} done in ${i * 10} ms` : `step ${i} 100%_ok`,
      JSON.stringify({ job_id: `j${i}`, n: i }),
    );
  // Two lines at the same ts as an op line and an audit row (ties across sources).
  await run("INSERT INTO operations (id, cluster_id, kind, status, log, actor, created_at, updated_at) VALUES (?, 'c_a', 'up', 'failed', ?, 'owner', ?, ?)", "op_1", JSON.stringify([
    { at: T0 + 5000, msg: "balance $30.63, projected $26.08 at the deadline (floor $8)" },
    { at: T0 + 6000, msg: "ltx: pod poda000000001x on NVIDIA RTX PRO 6000 in EUR-IS-1" },
    { at: T0 + 7000, msg: "h3-turbo: no stock in [eu]" },
    { at: T0 + 8000, msg: "WARNING: no pod for: h3-turbo" },
  ]), T0 + 4000, T0 + 9000);
  await DB.prepare("UPDATE operations SET error = 'no pod for h3-turbo' WHERE id = 'op_1'").run();
  await run("INSERT INTO audit (at, actor, action, target, after, ok) VALUES (?, 'owner', 'cluster.up', 'alpha', '{\"x\":1}', 1)", T0 + 6000);
  await run("INSERT INTO audit (at, actor, action, target, detail, ok) VALUES (?, 'owner', 'login', NULL, 'wrong passphrase', 0)", T0 + 20000);
  return env;
}
const Q = (p: Record<string, string>) => parseLogQuery(p, NOW);

describe("parsing", () => {
  it("times: unix ms / s, ISO, relative", () => {
    expect(parseTime("1791320156463", NOW)).toBe(1791320156463);
    expect(parseTime("1791320156", NOW)).toBe(1791320156000);
    expect(parseTime("2026-10-06T12:00:00Z", NOW)).toBe(T0);
    expect(parseTime("15m", NOW)).toBe(NOW - 900_000);
    expect(parseTime("-2h", NOW)).toBe(NOW - 7_200_000);
    expect(parseTime("7d", NOW)).toBe(NOW - 7 * 86_400_000);
    expect(parseTime("", NOW)).toBeUndefined();
    expect(() => parseTime("yesterday-ish", NOW)).toThrow(/time/);
  });
  it("query parameters: sources, levels (set or minimum), validation", () => {
    expect(Q({}).sources).toEqual(["pod", "op", "audit"]);
    expect(Q({ pod: "poda000000001x" }).sources).toEqual(["pod"]);
    expect(Q({ src: "audit,op" }).sources).toEqual(["audit", "op"]);
    expect(Q({ level: "warn" }).levels).toEqual(["warn", "error"]);
    expect(Q({ lv: "debug,error" }).levels).toEqual(["debug", "error"]);
    expect(() => Q({ src: "nope" })).toThrow(/src/);
    expect(() => Q({ q: "(", re: "1" })).toThrow(/regex/);
    expect(() => Q({ pod: "a b" })).toThrow(/pod/);
    expect(() => Q({ since: "1h", until: "2h" })).toThrow(/after/);
    expect(Q({ limit: "99999" }).limit).toBe(1000);
  });
  it("cursors round-trip; a foreign one is refused", () => {
    const c = { ts: 123, uid: "p:000000000042" };
    expect(decodeCursor(encodeCursor(c))).toEqual(c);
    expect(() => decodeCursor("garbage!")).toThrow(/cursor/);
    expect(decodeTail(encodeTail({ p: 1, a: 2, o: 3 }))).toEqual({ p: 1, a: 2, o: 3 });
  });
  it("regex pre-filter literal: only what every match must contain", () => {
    expect(regexLiteral("job_\\d+ done")).toBe(" done");
    expect(regexLiteral("connected to the dispatcher")).toBe("connected to the dispatcher");
    expect(regexLiteral("a|b")).toBeNull();
    expect(regexLiteral("[0-9]+")).toBeNull();
    expect(regexLiteral("colou?r")).toBe("colo");
    expect(regexLiteral("timeout(s)? after")).toBe("timeout");
    expect(regexLiteral("a\\.b\\.c")).toBe("a.b.c");
    expect(regexLiteral("^ready$")).toBe("ready");
  });
  it("levels of op / Runpod lines are inferred", () => {
    expect(inferLevel("h3-ref2v: no stock in [eu]")).toBe("warn");
    expect(inferLevel("create failed: 500")).toBe("error");
    expect(inferLevel("deleted 4506k60farotxy (stop)")).toBe("info");
  });
});

describe("queryLogs", () => {
  const uids = (ls: XLine[]) => ls.map((l) => l.uid);
  it("pages pod lines newest first with a cursor: no gaps, no repeats", async () => {
    const env = await seed();
    const seen: XLine[] = [];
    let cursor: string | undefined;
    for (let i = 0; i < 10; i++) {
      const r = await queryLogs(env, Q({ src: "pod", limit: "25", ...(cursor ? { cursor } : {}) }), NOW);
      seen.push(...r.lines);
      if (!r.next) break;
      cursor = r.next;
    }
    expect(seen).toHaveLength(60);
    expect(new Set(uids(seen)).size).toBe(60);
    for (let i = 1; i < seen.length; i++) expect(seen[i - 1]!.ts).toBeGreaterThanOrEqual(seen[i]!.ts);
    expect(seen[0]!.msg).toContain("59");
    expect(seen[0]!.cluster).toBe("beta");
  });
  it("ascending, with pool and pod names joined in", async () => {
    const env = await seed();
    const r = await queryLogs(env, Q({ src: "pod", order: "asc", limit: "3" }), NOW);
    expect(r.lines.map((l) => [l.pod_id, l.pool, l.pod_name])).toEqual([
      ["poda000000001x", "ltx", "fv-ctl-alpha-ltx-1"],
      ["podb000000002x", "h3-turbo", null],
      ["podc000000003x", "fake", null],
    ]);
    expect(r.next).not.toBeNull();
  });
  it("filters combine: cluster, pool, pod, levels, time range", async () => {
    const env = await seed();
    const all = (p: Record<string, string>) => queryLogs(env, Q({ src: "pod", limit: "1000", ...p }), NOW).then((r) => r.lines);
    expect((await all({ cluster: "c_b" })).every((l) => l.pod_id === "podc000000003x")).toBe(true);
    expect((await all({ pool: "h3-turbo" })).every((l) => l.pod_id === "podb000000002x")).toBe(true);
    expect(await all({ pool: "h3-turbo", cluster: "c_b" })).toHaveLength(0);
    const errs = await all({ lv: "error" });
    expect(errs.map((l) => l.level)).toEqual(Array(6).fill("error"));
    const warnPlus = await all({ level: "warn", pod: "poda000000001x" });
    expect(warnPlus.every((l) => ["warn", "error"].includes(l.level) && l.pod_id === "poda000000001x")).toBe(true);
    const range = await all({ since: String(T0 + 10_000), until: String(T0 + 19_000) });
    expect(range).toHaveLength(10);
  });
  it("text: LIKE with % and _ taken literally; case-sensitive and regex checked in the Worker", async () => {
    const env = await seed();
    const all = (p: Record<string, string>) => queryLogs(env, Q({ src: "pod", limit: "1000", ...p }), NOW).then((r) => r.lines);
    expect(await all({ q: "100%_ok" })).toHaveLength(45);
    expect(await all({ q: "0%" })).toHaveLength(45);
    expect(await all({ q: "JOB_4 DONE" })).toHaveLength(1);
    expect(await all({ q: "JOB_4 DONE", cs: "1" })).toHaveLength(0);
    const re = await all({ q: "job_\\d*[02468] done in \\d{3} ms", re: "1" });
    expect(re.map((l) => l.msg).every((m) => /job_\d*[02468] done in \d{3} ms/.test(m))).toBe(true);
    expect(re).toHaveLength(12); // job_12 … job_56 (every 4th line), 3-digit ms
    // A field value is searchable (job ids live in fields).
    expect((await all({ q: '"j17"' })).map((l) => l.msg)).toEqual(["step 17 100%_ok"]);
  });
  it("a regex scan that cannot fill the page says how far it searched", async () => {
    const env = await seed();
    const r = await queryLogs(env, Q({ src: "pod", q: "nomatch.*", re: "1", limit: "5" }), NOW);
    expect(r.lines).toHaveLength(0);
    expect(r.next).toBeNull(); // 60 rows fit one scan batch: done
    expect(r.partial).toBe(false);
  });
  it("merges op, audit and pod lines in one order; ties broken by uid across pages", async () => {
    const env = await seed();
    const seen: XLine[] = [];
    let cursor: string | undefined;
    for (let i = 0; i < 40; i++) {
      const r = await queryLogs(env, Q({ order: "asc", limit: "4", ...(cursor ? { cursor } : {}) }), NOW);
      seen.push(...r.lines);
      if (!r.next) break;
      cursor = r.next;
    }
    // 60 pod + 5 op (4 entries + the failure) + 2 audit
    expect(seen).toHaveLength(67);
    expect(new Set(uids(seen)).size).toBe(67);
    for (let i = 1; i < seen.length; i++) expect(beyond(seen[i]!, seen[i - 1]!, "asc")).toBe(true);
    const at6 = seen.filter((l) => l.ts === T0 + 6000).map((l) => l.source);
    expect(at6.sort()).toEqual(["audit", "op", "pod"]);
    const op = seen.filter((l) => l.source === "op");
    expect(op.map((l) => l.level)).toEqual(["info", "info", "warn", "warn", "error"]);
    expect(op[1]!.pod_id).toBe("poda000000001x");
    expect(op[2]!.pool).toBe("h3-turbo");
    expect(seen.find((l) => l.source === "audit" && l.level === "error")!.msg).toContain("wrong passphrase");
  });
  it("op and audit lines follow the cluster, pod and op filters", async () => {
    const env = await seed();
    const r = await queryLogs(env, Q({ src: "op,audit", cluster: "c_a", limit: "100" }), NOW);
    expect(r.lines.every((l) => l.cluster === "alpha")).toBe(true);
    expect(r.lines.filter((l) => l.source === "audit")).toHaveLength(1);
    const byPod = await queryLogs(env, Q({ src: "op", pod: "poda000000001x" }), NOW);
    expect(byPod.lines).toHaveLength(1);
    const byOp = await queryLogs(env, Q({ src: "op", op: "op_1", lv: "warn,error" }), NOW);
    expect(byOp.lines).toHaveLength(3);
  });
  it("context: N lines before and after a pod line, from the same pod", async () => {
    const env = await seed();
    const anchor = (await queryLogs(env, Q({ src: "pod", pod: "podb000000002x", order: "asc", limit: "10" }), NOW)).lines[5]!;
    const c = await lineContext(env, anchor.uid, 2, 3);
    expect(c.lines).toHaveLength(6);
    expect(c.lines[2]!.uid).toBe(anchor.uid);
    expect(c.lines.every((l) => l.pod_id === "podb000000002x")).toBe(true);
    for (let i = 1; i < c.lines.length; i++) expect(c.lines[i]!.ts).toBeGreaterThan(c.lines[i - 1]!.ts);
    const oc = await lineContext(env, "o:op_1:00002", 1, 1);
    expect(oc.lines.map((l) => l.uid)).toEqual(["o:op_1:00001", "o:op_1:00002", "o:op_1:00003"]);
  });
  it("live tail: lines ingested after the state, whatever their timestamps", async () => {
    const env = await seed();
    const q = Q({ src: "pod,audit", lv: "info,warn,error" });
    const first = await tailLogs(env, q, null, NOW);
    expect(first.lines).toHaveLength(0);
    await env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, msg) VALUES ('c_a', 'poda000000001x', ?, 'info', 'late batch')").bind(T0 - 5000).run();
    await env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, msg) VALUES ('c_a', 'poda000000001x', ?, 'debug', 'filtered out')").bind(NOW).run();
    await env.DB.prepare("INSERT INTO audit (at, actor, action, target, ok) VALUES (?, 'owner', 'cluster.down', 'alpha', 1)").bind(NOW).run();
    const next = await tailLogs(env, q, first.state, NOW);
    expect(next.lines.map((l) => l.msg)).toEqual(["late batch", "cluster.down alpha"]);
    const again = await tailLogs(env, q, next.state, NOW);
    expect(again.lines).toHaveLength(0);
  });
  it("facets: a level histogram and per-pod counts", async () => {
    const env = await seed();
    const f = await logFacets(env, Q({ since: String(T0), until: String(T0 + 60_000) }), 6, NOW);
    expect(f.step_ms).toBe(10_000);
    expect(f.histogram).toHaveLength(6);
    const total = f.histogram.reduce((s, b) => s + Object.values(b.counts).reduce((a, n) => a + n, 0), 0);
    expect(total).toBe(60);
    expect(f.pods.map((p: any) => p.pod_id).sort()).toEqual(["poda000000001x", "podb000000002x", "podc000000003x"]);
    expect(f.pods.reduce((s: number, p: any) => s + p.errors, 0)).toBe(6);
  });
  it("archive: R2 hours older than the D1 tail, filtered and paged", async () => {
    const env = await seed();
    const old = NOW - 30 * 3600_000;
    const hour = new Date(old).toISOString();
    const key = (n: number) => `logs/c_a/poda000000001x/${hour.slice(0, 10)}/${hour.slice(11, 13)}/${old + n}-x.ndjson`;
    await env.LOGS.put(key(1), [0, 1, 2].map((i) => JSON.stringify({ ts: old + i, level: i === 2 ? "error" : "info", msg: `archived ${i}` })).join("\n") + "\n");
    await env.LOGS.put(key(2), JSON.stringify({ ts: old + 10, level: "info", msg: "archived 3" }) + "\n");
    const r = await queryLogs(env, Q({ src: "archive", pod: "poda000000001x", since: "2d", limit: "3" }), NOW);
    expect(r.lines.map((l) => l.msg)).toEqual(["archived 3", "archived 2", "archived 1"]);
    expect(r.lines[0]!.pool).toBe("ltx");
    const r2 = await queryLogs(env, Q({ src: "archive", pod: "poda000000001x", since: "2d", limit: "3", cursor: r.next! }), NOW);
    expect(r2.lines.map((l) => l.msg)).toEqual(["archived 0"]);
    expect(r2.next).toBeNull();
    const errs = await queryLogs(env, Q({ src: "archive", pod: "poda000000001x", since: "2d", lv: "error" }), NOW);
    expect(errs.lines.map((l) => l.msg)).toEqual(["archived 2"]);
    expect(archiveHours(T0, T0 + 2 * 3600_000, "desc")).toEqual(["2026-10-06/14/", "2026-10-06/13/", "2026-10-06/12/"]);
  });
  it("Runpod tail lines: a leading timestamp is the line's time", () => {
    const ls = runpodLines("podx", ["2026-10-06T12:00:01Z fv-serve starting", "no timestamp here", "ERROR: boom"], ["pulling image"], NOW);
    expect(ls[0]!.ts).toBe(T0 + 1000);
    expect(ls[0]!.msg).toBe("fv-serve starting");
    expect(ls[1]!.ts).toBe(T0 + 1001);
    expect(ls[2]!.level).toBe("error");
    expect(ls[3]!.fields).toEqual({ stream: "system" });
  });
});

describe("cluster editor checks", () => {
  it("issue paths from normalizeSpec messages", () => {
    expect(pathOf("pools[2].models[0].recipe: x is not servable")).toEqual({ path: ["pools", 2, "models", 0, "recipe"], message: "x is not servable" });
    expect(pathOf("cap_s: 300-604800")).toEqual({ path: ["cap_s"], message: "300-604800" });
    expect(pathOf("something odd")).toEqual({ path: [], message: "something odd" });
  });
  it("every problem at once, each at its field; name rules for new and existing clusters", async () => {
    const env = await seed();
    const bad = { ...defaultSpec("alpha"), cap_s: 10, regions: ["us"], pools: [{ ...defaultSpec("x").pools[0]!, count: 12 }] };
    const r = await checkSpec(env, bad);
    expect(r.ok).toBe(false);
    const paths = r.issues.map((i) => i.path.join("."));
    expect(paths).toContain("cap_s");
    expect(paths).toContain("pools.0.count");
    expect(paths.some((p) => p.startsWith("regions"))).toBe(true);
    expect(r.issues.find((i) => i.path[0] === "name")!.message).toMatch(/exists/);
    const ok = await checkSpec(env, defaultSpec("gamma", "tiny-cpu"));
    expect(ok.ok).toBe(true);
    expect(ok.normalized!.name).toBe("gamma");
    const renamed = await checkSpec(env, defaultSpec("gamma"), { id: "c_a", name: "alpha" });
    expect(renamed.issues[0]!.message).toMatch(/fixed/);
    const zero = await checkSpec(env, { ...defaultSpec("delta", "tiny-cpu"), pools: [{ ...defaultSpec("d", "tiny-cpu").pools[0]!, count: 0 }] });
    expect(zero.ok).toBe(true);
    expect(zero.warnings.map((w) => w.path.join("."))).toContain("pools.0.count");
  });
});

describe("stock hint", () => {
  afterEach(() => (vi.unstubAllGlobals(), clearStockCache()));
  it("per pool, and a warning when pools want more GPUs of a DC than Runpod reports free", async () => {
    const env = await seed();
    let query = "";
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => {
      query = JSON.parse(String(init.body)).query;
      const data: Record<string, unknown> = {};
      for (const m of query.matchAll(/(g\d+): gpuTypes/g)) data[m[1]!] = [{ id: "x", securePrice: 1.89, lowestPrice: { stockStatus: "Low", maxUnreservedGpuCount: 2 } }];
      return new Response(JSON.stringify({ data }), { status: 200 });
    });
    const spec = normalizeSpec({ name: "std", template: "standard" });
    const a = await availability(env, spec);
    expect(query).toContain('dataCenterId: "EUR-IS-1"');
    expect(a.pools.map((p) => p.status)).toEqual(["low", "low", "low", "low"]);
    expect(a.pools[0]!.hint).toContain("RTX PRO 6000 in EUR-IS-1: Low (2 free)");
    expect(a.warnings[0]).toMatch(/4 pod\(s\) .* want RTX PRO 6000 in EUR-IS-1; Runpod reports 2 free/);
  });
  it("no stock reported: none; CPU pools are not GPU-bound", async () => {
    const env = await seed();
    vi.stubGlobal("fetch", async (_u: string, init: RequestInit) => {
      const q = JSON.parse(String(init.body)).query as string;
      const data: Record<string, unknown> = {};
      for (const m of q.matchAll(/(g\d+): gpuTypes/g)) data[m[1]!] = [{ id: "x", securePrice: 1.89, lowestPrice: { stockStatus: null, maxUnreservedGpuCount: 0 } }];
      return new Response(JSON.stringify({ data }), { status: 200 });
    });
    const spec = normalizeSpec({ name: "mix", pools: [{ id: "ltx" }, defaultSpec("t", "tiny-cpu").pools[0]] });
    const a = await availability(env, spec);
    expect(a.pools.map((p) => [p.pool, p.status])).toEqual([["ltx", "none"], ["fake", "ok"]]);
    expect(a.pools[0]!.hint).toMatch(/no stock reported/);
  });
});
