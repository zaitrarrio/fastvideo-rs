// weights_source r2 for serverless endpoints (src/serverless/weights.ts, ingest.ts, weights-plan.ts; docs/control/
// serverless.md §1b, docs/ops/r2-weights.md §9): the spec (default, placement, disk, validation), the payloads
// (no volume, the r2 boot, the weights token in the env, never an R2 key), the per-endpoint weights token (wrong,
// another endpoint's, revoked, rotated), the boot reports and the outlook; and the worker's boot (SLS_R2_SH) run in
// bash with the real scripts/gpu/r2-fetch.py against a fake R2: a cold boot fetches, a second boot is a cache hit that
// asks fv-control for nothing, a stale revision is fetched again, a cache volume is add-only.
import { execFile } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { b64, randomBytes, sha256Hex } from "../../src/crypto";
import type { Env } from "../../src/env";
import { R2_FETCH_PY, R2_FETCH_SHA256 } from "../../src/r2-fetch-script";
import { manifestKey, objectKey, R2_DOWNLOAD_GBPS } from "../../src/r2weights";
import { slsReport, slsTokenRow } from "../../src/serverless/ingest";
import { deleteEndpoint, getRow, updateEndpoint } from "../../src/serverless/ops";
import { endpointCreatePayload, templateCreatePayload, templateUpdatePayload, v2CreatePayload, v2TemplateBoot } from "../../src/serverless/payloads";
import { DEFAULT_SLS_WEIGHTS_SOURCE, defaultEndpointSpec, normalizeEndpointSpec, placementIssues, specDiff, type EndpointSpec } from "../../src/serverless/spec";
import { applyReport, newWeightsToken, SLS_R2_SH, SLS_TOKEN_GRACE_MS, slsOutlook, slsR2DiskGb, slsTrees, slsWeightsEnv, slsWeightsIssues, type BootRow } from "../../src/serverless/weights";
import { weightsPlanRequest } from "../../src/weights-plan";
import { d1 } from "./d1shim";
import { ACCESS, fakeR2, makeTree, rnd, SECRET } from "./fake-r2";

const EU = "jg48s6o1w0";
const toml = (tree: string) => `# a test config\n[r2probe]\npath = "\${FV_WEIGHTS}/${tree}/config.json"\n`;
const r2Spec = (name: string, tree: string, over: Record<string, unknown> = {}) => normalizeEndpointSpec({ name, variant: "wan", compute: "GPU", config_toml: toml(tree), weights_source: "r2", ...over });
const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_test_key_0123456789",
    PUBLIC_URL: "https://fvc.test",
    CF_ACCOUNT_ID: "acct",
    WEIGHTS_R2_ACCESS_KEY_ID: ACCESS,
    WEIGHTS_R2_SECRET_ACCESS_KEY: SECRET,
    ...over,
  }) as unknown as Env;
const pin = (rev: string, bytes = 1000) => ({ rev, files: 1, bytes, manifest_sha256: "a".repeat(64), source: "test" });

describe("the spec: weights_source", () => {
  it("defaults to volume (the coordinator flips DEFAULT_SLS_WEIGHTS_SOURCE after the benchmarks)", () => {
    expect(DEFAULT_SLS_WEIGHTS_SOURCE).toBe("volume");
    const d = defaultEndpointSpec("x", "wan", "wan");
    expect(d.weights_source).toBe("volume");
    expect(d.network_volume).toBe(EU);
    expect(d.data_centers).toEqual(["EUR-IS-1"]);
    expect(defaultEndpointSpec("x", "cpu").weights_source).toBeUndefined();
  });
  it("r2: no volume, no data-centre restriction, a container disk sized from the trees", () => {
    const s = normalizeEndpointSpec({ name: "h", preset: "h3-turbo", weights_source: "r2" });
    expect(s.weights_source).toBe("r2");
    expect(s.network_volume).toBeNull();
    expect(s.data_centers).toBeUndefined();
    expect(slsTrees(s)).toContain("h3-base");
    expect(s.container_disk_gb).toBe(slsR2DiskGb(slsTrees(s)));
    expect(s.container_disk_gb).toBeGreaterThanOrEqual(Math.ceil(144 * 1.15 + 30));
    // GPU types and data centres from anywhere.
    const any = normalizeEndpointSpec({ name: "h", preset: "h3-turbo", weights_source: "r2", gpu_types: ["NVIDIA H100 80GB HBM3", "NVIDIA H200"], data_centers: ["US-TX-3", "EU-RO-1"] });
    expect(any.data_centers).toEqual(["US-TX-3", "EU-RO-1"]);
  });
  it("refuses the weights volume, a disk too small for the trees, r2 on CPU; > 200 GB only with r2", () => {
    expect(() => normalizeEndpointSpec({ name: "h", preset: "h3-turbo", weights_source: "r2", network_volume: EU })).toThrow(/replaces the weights volume/);
    expect(() => normalizeEndpointSpec({ name: "h", preset: "h3-turbo", weights_source: "r2", container_disk_gb: 60 })).toThrow(/at least \d+ GB/);
    expect(() => normalizeEndpointSpec({ name: "c", variant: "cpu", weights_source: "r2" })).toThrow(/GPU workers/);
    expect(() => normalizeEndpointSpec({ name: "w", preset: "wan", container_disk_gb: 300 })).toThrow(/only with weights_source r2/);
  });
  it("a spec stored before weights_source (the EU volume, no field) stays a volume spec; back to volume restores the placement", () => {
    const old: any = { ...defaultEndpointSpec("w", "wan", "wan") };
    delete old.weights_source;
    expect(normalizeEndpointSpec(old).weights_source).toBe("volume");
    const r2 = normalizeEndpointSpec({ name: "w", preset: "wan", weights_source: "r2" });
    const { network_volume: _v, data_centers: _d, container_disk_gb: _c, ...rest } = r2;
    const back = normalizeEndpointSpec({ ...rest, weights_source: "volume" });
    expect([back.network_volume, back.data_centers]).toEqual([EU, ["EUR-IS-1"]]);
  });
  it("a switch changes the template (boot, env) and needs a new endpoint (the volume)", () => {
    const a = normalizeEndpointSpec({ name: "w", preset: "wan" });
    const b = normalizeEndpointSpec({ name: "w", preset: "wan", weights_source: "r2" });
    const d = specDiff(a, b);
    expect(d.template).toBe(true);
    expect(d.recreate).toContain("network_volume");
  });
  it("placement with r2 and no data_centers: every data centre's stock, reported per GPU type", async () => {
    const asked: string[] = [];
    const s = normalizeEndpointSpec({ name: "w", preset: "wan", weights_source: "r2", gpu_types: ["NVIDIA H100 80GB HBM3"] });
    const r = await placementIssues(s, async (pairs) => {
      asked.push(...pairs.map((p) => p.dc));
      return new Map(pairs.map((p) => [`${p.dc}|${p.gpu}`, { stock: p.dc === "US-TX-3" ? "High" : null, max_available: null }]));
    });
    expect(asked.length).toBeGreaterThan(20);
    expect(r.issues).toEqual([]);
    expect(r.stock).toEqual([{ gpu: "NVIDIA H100 80GB HBM3", dcs: ["US-TX-3"] }]);
    expect(r.warnings.some((w) => /any data centre: in stock H100 80GB HBM3 in US-TX-3/.test(w.message))).toBe(true);
    // The volume spec still asks its data centre only.
    const asked2: string[] = [];
    await placementIssues(normalizeEndpointSpec({ name: "w", preset: "wan" }), async (pairs) => (asked2.push(...pairs.map((p) => p.dc)), new Map()));
    expect([...new Set(asked2)]).toEqual(["EUR-IS-1"]);
  });
});

describe("payloads and the checks", () => {
  const env = mkEnv({ WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: pin("rev1") }) });
  const s = r2Spec("a", "demo");
  it("the template: the bash r2 boot, the weights token and the trees in the env, no volume, never an R2 key", () => {
    const w = slsWeightsEnv(env, s, "fvw_" + "1".repeat(64));
    const t = templateCreatePayload(s, "img@sha256:" + "0".repeat(64), "fvc-a-1", w);
    expect(t.dockerEntrypoint).toEqual(["bash", "-c", expect.stringContaining("r2-fetch.py")]);
    expect(t.dockerEntrypoint![2]).toMatch(/exec \/opt\/fastvideo-rs\/bin\/fv-serve/);
    expect(t.env.FV_WEIGHTS_TOKEN).toBe("fvw_" + "1".repeat(64));
    expect(t.env.FV_WEIGHTS).toBe("/workspace/weights");
    expect(t.env.FV_R2_TREE_REVS).toBe("demo@rev1");
    expect(t.env.FV_WEIGHTS_PLAN_URL).toBe("https://fvc.test/ingest/v1/weights-plan");
    expect(t.env.FV_SLS_REPORT_URL).toBe("https://fvc.test/ingest/v1/sls-report");
    expect(t.env.FV_R2_FETCH_SHA256).toBe(R2_FETCH_SHA256);
    expect(JSON.stringify(t)).not.toContain(SECRET);
    expect(JSON.stringify(t)).not.toContain(ACCESS);
    expect(t.containerDiskInGb).toBe(s.container_disk_gb);
    const e = endpointCreatePayload(s, "tpl");
    expect(e).not.toHaveProperty("networkVolumeId");
    expect(e).not.toHaveProperty("dataCenterIds");
    expect(v2CreatePayload({ ...s, mode: "lb", scaler_type: "REQUEST_COUNT" }, "img", [], w)).not.toHaveProperty("networkVolumes");
    expect(v2TemplateBoot({ ...s, mode: "lb" })!.dockerEntrypoint[0]).toBe("bash");
    expect(templateUpdatePayload(s, "img", w).dockerEntrypoint![0]).toBe("bash");
  });
  it("a volume endpoint is unchanged: no token, the volume's weights", () => {
    const v = normalizeEndpointSpec({ name: "w", preset: "wan" });
    const t = templateCreatePayload(v, "img", "n", slsWeightsEnv(env, s, "fvw_x"));
    expect(t.env.FV_WEIGHTS_TOKEN).toBeUndefined();
    expect(t.env.FV_WEIGHTS).toBe("/runpod-volume/weights");
    expect(endpointCreatePayload(v, "tpl")).toMatchObject({ networkVolumeId: EU, dataCenterIds: ["EUR-IS-1"] });
  });
  it("refuses unpublished trees, a store not configured, a volume that is not a listed cache; a listed cache is pinned to its DC", () => {
    expect(slsWeightsIssues(env, s).issues).toEqual([]);
    expect(slsWeightsIssues(mkEnv(), s).issues.map((i) => i.message).join()).toMatch(/not in the R2 weights store yet: demo/);
    expect(slsWeightsIssues(mkEnv({ WEIGHTS_R2_ACCESS_KEY_ID: undefined, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: pin("rev1") }) }), s).issues[0]!.message).toMatch(/not configured/);
    const cached = { ...s, network_volume: "cachevol01" };
    expect(slsWeightsIssues(env, cached).issues[0]!.message).toMatch(/not an R2 cache volume/);
    const env2 = mkEnv({ WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: pin("rev1") }), FV_SLS_R2_CACHE_VOLUMES: JSON.stringify({ cachevol01: "US-TX-3" }) });
    expect(slsWeightsIssues(env2, cached).issues).toEqual([]);
    expect(slsWeightsIssues(env2, cached).warnings[0]!.message).toMatch(/R2 cache volume cachevol01 \(US-TX-3\)/);
    expect(slsWeightsIssues(env2, { ...cached, data_centers: ["EU-RO-1"] as any }).issues[0]!.message).toMatch(/must run in US-TX-3/);
    expect(slsWeightsEnv(env2, cached, "fvw_x").FV_R2_CACHE_ROOT).toBe("/runpod-volume/fv-r2-cache");
  });
});

describe("boot reports and the outlook", () => {
  it("folds the boot's phases and the fetcher's lines into the breakdown", () => {
    let b: Partial<BootRow> = { worker_id: "w1", boot: "w1@1700000000", boot_at: 1000 };
    b = applyReport(b, "cache", "miss: h3-base", 2000);
    expect([b.cache, b.missing, b.fetch_start]).toEqual(["miss", "h3-base", 2000]);
    b = applyReport(b, "weights", "r2 40.0/144.0 GB, 1500 MB/s, eta 70s, trees 0/1 (now 1600 MB/s)", 30000);
    expect([b.fetch_bytes, b.fetch_mbps]).toEqual([40e9, 1500]);
    b = applyReport(b, "weights", "r2 weights ready: 1 tree(s), 144.00 GB fetched in 96.0s (1500 MB/s); retries 0, bad parts 0, resumed parts 0", 100000);
    expect([b.fetch_bytes, b.fetch_s, b.fetch_mbps, b.fetch_end]).toEqual([144e9, 96, 1500, 100000]);
    b = applyReport(b, "serve", "weights ready", 101000);
    expect(b.serve_at).toBe(101000);
    expect(applyReport({}, "cache", "hit", 5).cache).toBe("hit");
  });
  it("the estimate: start + fetch at the planning rate (or the measured one) + load; the cache state", () => {
    const env = mkEnv();
    const s = normalizeEndpointSpec({ name: "h", preset: "h3-turbo", weights_source: "r2" });
    const o = slsOutlook(env, s);
    expect(o.fetch_gb).toBeGreaterThanOrEqual(144);
    expect(o.fetch_s).toBe(Math.round(o.fetch_gb / R2_DOWNLOAD_GBPS));
    expect(o.cold_start_s).toBe(o.image_start_s + o.fetch_s + o.load_s);
    expect(o.cached_cold_start_s).toBe(o.image_start_s + o.load_s);
    expect(o.cache.state).toBe("unknown");
    const boots: BootRow[] = [{ worker_id: "w", boot: "w@1", boot_at: 1, cache: "hit", missing: null, fetch_start: null, fetch_end: null, fetch_bytes: null, fetch_s: null, fetch_mbps: null, serve_at: 3001, error: null, last_detail: null, updated_at: 1 }];
    expect(slsOutlook(env, s, boots).cache.state).toBe("hit");
    const measured = slsOutlook(env, s, [{ ...boots[0]!, cache: "miss", fetch_bytes: 144e9, fetch_mbps: 2000 }]);
    expect([measured.rate_source, measured.rate_gbps]).toEqual(["measured", 2]);
    expect(slsOutlook(env, normalizeEndpointSpec({ name: "h", preset: "h3-turbo" })).fetch_s).toBe(0);
  });
});

// ---------------------------------------------------------------- the weights token, against D1 and a fake R2
describe("the weights token: the plan and the reports", () => {
  const r2 = fakeR2();
  let base = "";
  const ta = makeTree("demo", "rev1", { "config.json": Buffer.from('{"a": 1}'), "transformer/model.safetensors": rnd(2_300_000, 7) });
  const tb = makeTree("other", "revB", { "config.json": Buffer.from('{"b": 1}') });
  const tpl: any[] = [];
  let env: Env;
  let tokA = "";
  let tokB = "";
  const insert = async (id: string, name: string, spec: EndpointSpec, hash: string) => {
    const t = Date.now();
    await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, own_template, mode, spec, image, status, created_at, created_by, updated_at, weights_token_hash) VALUES (?, ?, ?, ?, 1, 'queue', ?, 'img@sha256:0', 'active', ?, 'owner', ?, ?)")
      .bind(id, name, `rp${name}`, `tpl${name}`, JSON.stringify(spec), t, t, hash)
      .run();
  };
  const plan = (tok: string | null, body: unknown = { pod: "w1@1700000000" }) =>
    weightsPlanRequest(env, new Request("https://fvc.test/ingest/v1/weights-plan", { method: "POST", headers: tok ? { authorization: `Bearer ${tok}` } : {}, body: JSON.stringify(body) })).then(
      async (r) => ({ status: r.status, j: (await r.json()) as any }),
      (e) => ({ status: e.status as number, j: { error: e.message } }),
    );
  beforeAll(async () => {
    base = await r2.start();
    env = mkEnv({ WEIGHTS_R2_ENDPOINT: base, RUNPOD_REST: `${base}/rest`, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: ta.pin, other: tb.pin }) });
    r2.objects.set(`fv-weights/${manifestKey("demo", "rev1")}`, ta.raw);
    r2.objects.set(`fv-weights/${manifestKey("other", "revB")}`, tb.raw);
    const a = await newWeightsToken();
    const b = await newWeightsToken();
    tokA = a.token;
    tokB = b.token;
    await insert("se_a", "a", r2Spec("a", "demo"), a.hash);
    await insert("se_b", "b", r2Spec("b", "other"), b.hash);
    await insert("se_v", "v", normalizeEndpointSpec({ name: "v", preset: "wan" }), (await newWeightsToken()).hash);
    // The Runpod REST calls of update and delete (fv-control only touches its own fvc-… endpoints).
    const json = (res: any, code: number, body: unknown) => res.writeHead(code, { "content-type": "application/json" }).end(JSON.stringify(body));
    r2.routes["/rest/endpoints/rpa"] = (req, res) => (req.method === "GET" ? json(res, 200, { id: "rpa", name: "fvc-a", workers: [] }) : json(res, 200, {}));
    r2.routes["/rest/endpoints/rpb"] = (req, res) => (req.method === "GET" ? json(res, 404, { error: "not found" }) : json(res, 200, {}));
    r2.routes["/rest/templates/tpla"] = (req, res) => {
      let body = "";
      req.on("data", (d: Buffer) => (body += d));
      req.on("end", () => (tpl.push({ method: req.method, body: body ? JSON.parse(body) : null }), json(res, 200, {})));
    };
    r2.routes["/rest/templates/tplb"] = (_req, res) => json(res, 200, {});
  });
  afterAll(() => r2.stop());

  it("no token, a wrong one, a cluster's kind: 401; its own: its trees only, presigned", async () => {
    expect((await plan(null)).status).toBe(401);
    expect((await plan("fvw_" + "0".repeat(64))).status).toBe(401);
    expect((await plan("fvi_" + "0".repeat(64))).status).toBe(401);
    const a = await plan(tokA);
    expect(a.status).toBe(200);
    expect(a.j.trees.map((t: any) => [t.dest, t.rev, t.files.length])).toEqual([["demo", "rev1", 2]]);
    expect(a.j.trees[0].files.every((f: any) => /X-Amz-Signature=[0-9a-f]{64}/.test(f.url))).toBe(true);
    expect(JSON.stringify(a.j)).not.toContain(SECRET);
    const b = await plan(tokB);
    expect(b.j.trees.map((t: any) => t.dest)).toEqual(["other"]);
  });
  it("another endpoint's trees: never (the token names its endpoint; a request naming another is 403)", async () => {
    expect((await plan(tokA, { pod: "w1@1700000000", endpoint: "b" })).status).toBe(403);
    expect((await plan(tokA, { pod: "w1@1700000000", endpoint: "se_b" })).status).toBe(403);
    expect((await plan(tokA, { pod: "w1@1700000000", endpoint: "a" })).status).toBe(200);
    // The request cannot widen the set: whatever it names, A gets demo.
    expect((await plan(tokA, { pod: "x", trees: ["other"] })).j.trees.map((t: any) => t.dest)).toEqual(["demo"]);
  });
  it("a volume endpoint's token gets nothing (409); the reports take the same token", async () => {
    const v = await getRow(env, "v");
    const t = await newWeightsToken();
    await env.DB.prepare("UPDATE serverless_endpoints SET weights_token_hash = ? WHERE id = ?").bind(t.hash, v.id).run();
    expect((await plan(t.token)).status).toBe(409);
    const rep = (tok: string, body: unknown) =>
      slsReport(env, new Request("https://fvc.test/ingest/v1/sls-report", { method: "POST", headers: { authorization: `Bearer ${tok}` }, body: JSON.stringify(body) })).then(
        () => 200,
        (e) => e.status,
      );
    expect(await rep("fvw_" + "0".repeat(64), { pod: "w1@1700000000", phase: "boot", detail: "x" })).toBe(401);
    expect(await rep(tokA, { pod: "no-at-sign", phase: "boot" })).toBe(400);
    expect(await rep(tokA, { pod: "w1@1700000000", phase: "nope" })).toBe(400);
    for (const [phase, detail] of [["boot", "container started"], ["cache", "miss: demo"], ["weights", "r2 weights ready: 1 tree(s), 2.30 GB fetched in 2.0s (1150 MB/s); retries 0"], ["serve", "weights ready"]])
      expect(await rep(tokA, { pod: "w1@1700000000", phase, url: "", detail })).toBe(200);
    const row = await env.DB.prepare("SELECT * FROM serverless_boots WHERE endpoint = 'se_a'").first<any>();
    expect([row.worker_id, row.cache, row.missing, row.fetch_mbps, row.fetch_s, !!row.serve_at]).toEqual(["w1", "miss", "demo", 1150, 2, true]);
    const lines = await env.DB.prepare("SELECT msg, target FROM log_lines WHERE cluster_id = 'serverless:se_a' AND pod_id = 'w1' ORDER BY id").all<any>();
    expect(lines.results.map((l: any) => l.msg)).toEqual(["[boot] container started", "[cache] miss: demo", expect.stringMatching(/^\[weights\] r2 weights ready/), "[serve] weights ready"]);
  });
  it("rotation: a template update gives a new token; the old one works for the grace period only", async () => {
    const row = await getRow(env, "a");
    const spec = JSON.parse(row.spec);
    await updateEndpoint(env as any, { actor: "owner" }, row, { ...spec, env: { FOO: "1" } }, { image: "img@sha256:" + "1".repeat(64) });
    const patch = tpl.find((x) => x.method === "PATCH")!.body;
    const fresh = patch.env.FV_WEIGHTS_TOKEN as string;
    expect(fresh).toMatch(/^fvw_[0-9a-f]{64}$/);
    expect(fresh).not.toBe(tokA);
    expect(patch.dockerEntrypoint[0]).toBe("bash");
    const after = await getRow(env, "a");
    expect(after.weights_token_hash).toBe(await sha256Hex(fresh));
    expect(after.weights_token_prev_hash).toBe(await sha256Hex(tokA));
    expect(JSON.stringify(after)).not.toContain(fresh);
    expect((await plan(fresh)).status).toBe(200);
    expect((await plan(tokA)).status).toBe(200);
    expect(await slsTokenRow(env, tokA, Date.now() + SLS_TOKEN_GRACE_MS + 1000)).toBeNull();
    expect((await slsTokenRow(env, fresh, Date.now() + SLS_TOKEN_GRACE_MS + 1000))?.id).toBe("se_a");
    tokA = fresh;
  });
  it("revoked: a deleted endpoint's token gets nothing", async () => {
    expect((await plan(tokB)).status).toBe(200);
    const row = await getRow(env, "b");
    const out = await deleteEndpoint(env, { actor: "owner" }, row, "test");
    expect(out.status).toBe("deleted");
    expect((await plan(tokB)).status).toBe(401);
  });
});

// ---------------------------------------------------------------- the worker's boot, in bash, with the real fetcher
const py = (() => existsSync("/usr/bin/python3") || existsSync("/usr/local/bin/python3"))();
describe.skipIf(!py)("the worker's boot (SLS_R2_SH) with the real r2-fetch.py against a fake R2 and fv-control's handlers", () => {
  const r2 = fakeR2();
  let base = "";
  const files = { "config.json": Buffer.from('{"x": 1}'), "text_encoder/model.safetensors": rnd(1_500_000, 11), "transformer/model.safetensors": rnd(2_700_000, 12) };
  const t = makeTree("demo", "rev1", files);
  let env: Env;
  let token = "";
  let plans = 0;
  const handler = (fn: (req: Request) => Promise<Response | unknown>) => (req: any, res: any) => {
    let body = "";
    req.on("data", (d: Buffer) => (body += d));
    req.on("end", async () => {
      try {
        const out = await fn(new Request(`${base}${req.url}`, { method: "POST", headers: { authorization: req.headers.authorization || "", "content-type": "application/json" }, body }));
        if (out instanceof Response) res.writeHead(out.status, { "content-type": "application/json" }).end(await out.text());
        else res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(out));
      } catch (e: any) {
        res.writeHead(e.status || 500).end(JSON.stringify({ error: e.message }));
      }
    });
  };
  beforeAll(async () => {
    base = await r2.start();
    env = mkEnv({ WEIGHTS_R2_ENDPOINT: base, PUBLIC_URL: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: t.pin }) });
    r2.objects.set(`fv-weights/${manifestKey("demo", "rev1")}`, t.raw);
    for (const [p, b] of Object.entries(files)) r2.objects.set(`fv-weights/${objectKey("demo", "rev1", p)}`, b);
    const tk = await newWeightsToken();
    token = tk.token;
    const now = Date.now();
    await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, endpoint_id, mode, spec, status, created_at, created_by, updated_at, weights_token_hash) VALUES ('se_w', 'w', 'rpw', 'queue', ?, 'active', ?, 'owner', ?, ?)")
      .bind(JSON.stringify(r2Spec("w", "demo")), now, now, tk.hash)
      .run();
    r2.routes["/ingest/v1/weights-plan"] = handler((req) => (plans++, weightsPlanRequest(env, req)));
    r2.routes["/ingest/v1/sls-report"] = handler((req) => slsReport(env, req));
    r2.routes["/ingest/v1/r2-fetch.py"] = (_req, res) => res.writeHead(200).end(R2_FETCH_PY);
  });
  afterAll(() => r2.stop());
  const boot = (root: string, over: Record<string, string> = {}) => {
    const w = slsWeightsEnv(env, r2Spec("w", "demo"), token);
    const e = { PATH: process.env.PATH!, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1", ...w, FV_WEIGHTS_TREES_B64: "", FV_WEIGHTS: join(root, "weights"), FV_R2_STATE: join(root, "state"), FV_R2_TOOLS: join(root, "tools"), FV_R2_MIN_FREE_GB: "0", FV_FAIL_SLEEP_S: "0", RUNPOD_POD_ID: "wk1", ...over };
    return promisify(execFile)("bash", ["-c", `set -eu\n${SLS_R2_SH}echo BOOT-DONE`], { env: e, timeout: 60000 });
  };
  const boots = async () => (await env.DB.prepare("SELECT * FROM serverless_boots WHERE endpoint = 'se_w' ORDER BY boot_at, updated_at").all<any>()).results;

  it("a cold boot fetches the trees (presigned, ranged, verified) and reports its breakdown; a second boot is a cache hit that asks for nothing", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvsls-"));
    const r = await boot(root);
    expect(r.stdout).toContain("BOOT-DONE");
    for (const [p, b] of Object.entries(files)) expect(readFileSync(join(root, "weights/demo", p)).equals(b)).toBe(true);
    expect(readFileSync(join(root, "weights/demo/.fv-revision"), "utf8")).toBe("rev1");
    expect(r2.bad()).toBe(0);
    expect(plans).toBe(1);
    const b1 = (await boots()).at(-1);
    expect([b1.worker_id, b1.cache, b1.missing, b1.fetch_s !== null, b1.fetch_mbps !== null, !!b1.serve_at]).toEqual(["wk1", "miss", "demo", true, true, true]);
    // The same disk again (a FlashBoot revival or a restarted container): no plan request, no GET.
    const gets = r2.gets.length;
    const r2nd = await boot(root, { RUNPOD_POD_ID: "wk1b" });
    expect(r2nd.stderr).toMatch(/cache hit/);
    expect(plans).toBe(1);
    expect(r2.gets.length).toBe(gets);
    expect((await boots()).find((b: any) => b.worker_id === "wk1b").cache).toBe("hit");
  }, 90000);
  it("a tree at another revision is fetched again; a wrong token fails the boot loudly", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvsls-"));
    await boot(root);
    writeFileSync(join(root, "weights/demo/.fv-revision"), "rev0");
    const n = plans;
    const r = await boot(root);
    expect(r.stderr).toMatch(/cache miss: demo/);
    expect(plans).toBe(n + 1);
    expect(readFileSync(join(root, "weights/demo/.fv-revision"), "utf8")).toBe("rev1");
    const bad = mkdtempSync(join(tmpdir(), "fvsls-"));
    await expect(boot(bad, { FV_WEIGHTS_TOKEN: "fvw_" + "0".repeat(64) })).rejects.toMatchObject({ stderr: expect.stringMatching(/FAILED: r2 weights: .*plan: HTTP 401/) });
    expect(existsSync(join(bad, "weights/demo"))).toBe(false);
  }, 90000);
  it("an R2 cache volume is add-only: per-worker staging, a second worker hits, another revision there is never replaced", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvsls-"));
    const vol = join(root, "runpod-volume");
    mkdirSync(vol);
    const cache = { FV_R2_CACHE_ROOT: join(vol, "fv-r2-cache") };
    await boot(join(root, "w1"), cache);
    expect(readFileSync(join(vol, "fv-r2-cache/demo/.fv-revision"), "utf8")).toBe("rev1");
    // $FV_WEIGHTS links to the cache.
    expect(readFileSync(join(root, "w1/weights/demo/config.json"), "utf8")).toBe('{"x": 1}');
    const n = plans;
    const r = await boot(join(root, "w2"), { ...cache, RUNPOD_POD_ID: "wk2" });
    expect(r.stderr).toMatch(/cache hit/);
    expect(plans).toBe(n);
    // Another revision on the volume: refused, left as it is.
    const vol2 = join(root, "vol2");
    mkdirSync(join(vol2, "fv-r2-cache/demo"), { recursive: true });
    writeFileSync(join(vol2, "fv-r2-cache/demo/.fv-revision"), "rev0");
    writeFileSync(join(vol2, "fv-r2-cache/demo/keep.txt"), "old");
    await expect(boot(join(root, "w3"), { FV_R2_CACHE_ROOT: join(vol2, "fv-r2-cache"), RUNPOD_POD_ID: "wk3" })).rejects.toMatchObject({ stderr: expect.stringMatching(/add-only: .* not replaced/) });
    expect(readFileSync(join(vol2, "fv-r2-cache/demo/keep.txt"), "utf8")).toBe("old");
    expect(readdirSync(join(vol2, "fv-r2-cache/demo"))).toEqual([".fv-revision", "keep.txt"]);
  }, 90000);
});
