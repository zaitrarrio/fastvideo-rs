// Standalone pods (docs/control/standalone-pods.md), pod logs from boot
// (src/podlogs.ts), the image preflight (src/cluster/preflight.ts) and the
// early verdict of `up` (src/cluster/upwait.ts): the fixes for the
// 2026-10-06 h3-and-ltx start that never came up.
import { afterEach, describe, expect, it, vi } from "vitest";
import { mintApiToken } from "../../src/auth";
import { bootMatch, bootPhase, bootRows, foldBoot, getBoot, markReady, recordBoot } from "../../src/boottime";
import { checkImages, imageRevision, IMAGE_FEATURES, neededFeatures, revisionContains } from "../../src/cluster/preflight";
import { bootFor, EDGE_WORKER_BOOT, WATCHDOG_WORKER_BOOT, WORKER_BOOT, workerCreatePayload, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { normalizeSpec } from "../../src/cluster/spec";
import { getCluster, ownerOf } from "../../src/cluster/store";
import { summarize, upOutcome, type PoolView } from "../../src/cluster/upwait";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { searchLogs } from "../../src/logs";
import { parseLogQuery, queryLogs } from "../../src/logquery";
import { captureRunpodLogs, checkPod, diagnose, getDiagnosis, lineLevel, newSince, parseRunpodLine, PULL_DEADLINE_MS, saveDiagnosis, STUCK_MS } from "../../src/podlogs";
import { standaloneSpec, STANDALONE_POOL } from "../../src/standalone";
import { d1 } from "./d1shim";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});
const json = (body: unknown, status = 200, headers: Record<string, string> = {}) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

/** An R2 stand-in that keeps what was put. */
function bucket() {
  const objs = new Map<string, string>();
  return { objs, put: async (k: string, v: string) => void objs.set(k, v), get: async () => null, list: async () => ({ objects: [], truncated: false }) } as unknown as R2Bucket & { objs: Map<string, string> };
}
/** The ClusterOps namespace stand-in: records operations, no operation running. */
function opsNs() {
  const calls: { path: string; body: any }[] = [];
  const stub = {
    fetch: async (url: string, init?: RequestInit) => {
      const path = new URL(url).pathname;
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      calls.push({ path, body });
      if (path === "/op") return json({ op: null });
      if (path === "/op/start") return json({ id: body.id, status: "running" }, 202);
      if (path === "/op/cancel") return json({ cancelled: false });
      return json({ ok: true });
    },
  };
  return { calls, ns: { idFromName: (n: string) => n, get: () => stub } as unknown as DurableObjectNamespace };
}
const mkEnv = (over: Partial<Env> = {}) => {
  const ops = opsNs();
  const env = { DB: d1(), LOGS: bucket(), CLUSTER_OPS: ops.ns, CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_SECRETKEY123", ...over } as unknown as Env;
  return { env, ops };
};

// The line the 2cd1ba0 image's fv-serve prints (reproduced from the image on 2026-10-06), then exits 2.
const CRASH = "fv-serve: config: FV_AUTH_MODE=trust-edge: unknown variant `trust-edge`, expected one of `none`, `keys`, `trust-gateway`";

describe("image preflight", () => {
  const index = { manifests: [{ digest: "sha256:arm", platform: { architecture: "arm64", os: "linux" } }, { digest: "sha256:amd", platform: { architecture: "amd64", os: "linux" } }] };
  const ghcr = (rev: string | null) =>
    vi.fn(async (u: any) => {
      const url = String(u);
      if (url.includes("/token")) return json({ token: "anon" });
      if (url.endsWith("/manifests/sha256:amd")) return json({ config: { digest: "sha256:cfg" }, layers: [] });
      if (url.includes("/manifests/")) return json(index);
      if (url.endsWith("/blobs/sha256:cfg")) return json({ config: { Labels: rev ? { "org.opencontainers.image.revision": rev } : {} } });
      if (url.includes("/compare/")) {
        const [, base, head] = /compare\/([0-9a-f]+)\.\.\.([0-9a-f]+)/.exec(url)!;
        // 2cd1ba0 is older than every feature commit; b51ddcd is newer.
        return json({ status: head.startsWith("2cd1ba0") ? "behind" : head.startsWith("b51ddcd") ? "ahead" : base === head ? "identical" : "diverged" });
      }
      return json({}, 404);
    });
  const img = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "c".repeat(64);
  const env = { GITHUB_PAT: "ghp_x", GHCR: "https://ghcr.test", GITHUB_API: "https://gh.test" } as unknown as Env;

  it("reads the revision label through the index's amd64 manifest", async () => {
    globalThis.fetch = ghcr("2cd1ba0e5531f3bec378d745668d040c63e36ba0") as any;
    expect(await imageRevision(env, img)).toBe("2cd1ba0e5531f3bec378d745668d040c63e36ba0");
    globalThis.fetch = ghcr(null) as any;
    expect(await imageRevision(env, img)).toBeNull();
    expect(await imageRevision(env, "not a ref")).toBeNull();
  });
  it("asks GitHub whether a revision contains a commit; unknown without a PAT", async () => {
    globalThis.fetch = ghcr(null) as any;
    expect(await revisionContains(env, "b51ddcdcf64562b325342fa3a0a6a51664e53552", IMAGE_FEATURES.edge_front.sha)).toBe(true);
    expect(await revisionContains(env, "2cd1ba0e5531f3bec378d745668d040c63e36ba0", IMAGE_FEATURES.edge_front.sha)).toBe(false);
    expect(await revisionContains({} as Env, "1111111", IMAGE_FEATURES.edge_front.sha)).toBeNull();
    expect(await revisionContains({} as Env, IMAGE_FEATURES.edge_front.sha, IMAGE_FEATURES.edge_front.sha)).toBe(true);
  });
  it("refuses an edge cluster on a pre-edge image (the stable = 2cd1ba0 case); a direct one only warns", async () => {
    globalThis.fetch = ghcr("2cd1ba0e5531f3bec378d745668d040c63e36ba0") as any;
    const edge = normalizeSpec({ name: "h3-and-ltx", template: "h3", image: { channel: "stable" } });
    expect(neededFeatures(edge).find((f) => f.feature === "edge_front")?.fatal).toBe(true);
    const pf = await checkImages(env, edge, Object.fromEntries(edge.pools.map((p) => [p.id, img])));
    expect(pf.errors[0]).toMatch(/h3-turbo, h3-max, h3-ref2v: the image \(channel stable, built from 2cd1ba0\) predates edge fronts \(FV_AUTH_MODE=trust-edge/);
    expect(pf.errors.at(-1)).toMatch(/channel "latest"/);
    const direct = normalizeSpec({ name: "d", template: "h3", image: { channel: "stable" }, control_plane: "direct" });
    const pd = await checkImages(env, direct, Object.fromEntries(direct.pools.map((p) => [p.id, img])));
    expect(pd.errors).toEqual([]);
    expect(pd.warnings.join(" ")).toMatch(/direct workers/);
    expect(pd.warnings.join(" ")).toMatch(/log shipping/);
  });
  it("passes a newer image, and never refuses when the revision cannot be read", async () => {
    globalThis.fetch = ghcr("b51ddcdcf64562b325342fa3a0a6a51664e53552") as any;
    const edge = normalizeSpec({ name: "e", template: "h3", image: { channel: "latest" } });
    const ok = await checkImages(env, edge, { "h3-turbo": img });
    expect(ok).toMatchObject({ errors: [], warnings: [] });
    globalThis.fetch = ghcr(null) as any;
    const unk = await checkImages(env, edge, { "h3-turbo": img });
    expect(unk.errors).toEqual([]);
    expect(unk.warnings[0]).toMatch(/could not read the image's revision label/);
  });
});

describe("pod logs from boot", () => {
  it("parses Runpod's timestamped lines (nanoseconds, no timestamp, ANSI)", () => {
    expect(parseRunpodLine("2026-10-06T20:41:02.230004105Z [20:41:02] fv-build service on :8000")).toEqual({ ts: Date.parse("2026-10-06T20:41:02.230Z"), text: "[20:41:02] fv-build service on :8000" });
    expect(parseRunpodLine("2026-10-06T20:39:46Z create container x")).toEqual({ ts: Date.parse("2026-10-06T20:39:46Z"), text: "create container x" });
    expect(parseRunpodLine("\x1b[32mplain\x1b[0m")).toEqual({ ts: 0, text: "plain" });
  });
  it("a cursor keeps every line once, also several lines with one timestamp", () => {
    const L = (ts: number, text: string) => ({ ts, text });
    const a = [L(1, "a"), L(2, "b"), L(2, "c")];
    const r1 = newSince(a, { ts: 0, n: 0 });
    expect(r1.fresh.map((l) => l.text)).toEqual(["a", "b", "c"]);
    expect(r1.cursor).toEqual({ ts: 2, n: 2 });
    const r2 = newSince([...a, L(2, "d"), L(3, "e")], r1.cursor);
    expect(r2.fresh.map((l) => l.text)).toEqual(["d", "e"]);
    expect(newSince([L(2, "c"), L(2, "d"), L(3, "e")], r2.cursor).fresh).toEqual([]);
    // A line without a timestamp rides with the one before it.
    expect(newSince([L(5, "x"), L(0, "y")], { ts: 3, n: 1 }).fresh).toEqual([L(5, "x"), L(5, "y")]);
  });
  it("levels: fv-serve JSON lines keep theirs; a fatal line is an error", () => {
    expect(lineLevel('{"level":"WARN","fields":{"message":"slow","job_id":"j1"}}', "container")).toEqual({ level: "warn", msg: "slow", fields: { job_id: "j1" } });
    expect(lineLevel(CRASH, "container").level).toBe("error");
    expect(lineLevel("Pulling fs layer", "system").level).toBe("info");
    expect(lineLevel("failed to pull image: manifest unknown", "system").level).toBe("error");
  });
  it("diagnoses a boot: crash loop, a single error, image error, pulling, starting, stuck, running", () => {
    const sys = ["create container img", "img Pulling from zaitrarrio/fastvideo-rs-serve", "abc Downloading"];
    expect(diagnose({ ageMs: 60_000, container: [CRASH, CRASH], system: sys })).toMatchObject({ phase: "crashloop", fatal: true });
    expect(diagnose({ ageMs: 60_000, container: [CRASH, CRASH], system: sys }).detail).toContain("unknown variant `trust-edge`");
    expect(diagnose({ ageMs: 60_000, container: [CRASH], system: sys })).toMatchObject({ phase: "error", fatal: false });
    expect(diagnose({ ageMs: 60_000, container: [], system: [...sys, "failed to pull image: manifest unknown"] })).toMatchObject({ phase: "image_error", fatal: true });
    expect(diagnose({ ageMs: 60_000, container: [], system: sys })).toMatchObject({ phase: "pulling", fatal: false });
    expect(diagnose({ ageMs: 60_000, container: [], system: [...sys, "start container for img: begin"] })).toMatchObject({ phase: "starting", fatal: false });
    expect(diagnose({ ageMs: STUCK_MS + 1, uptimeS: null, container: [], system: sys })).toMatchObject({ phase: "stuck", fatal: true });
    expect(diagnose({ ageMs: STUCK_MS + 1, uptimeS: 30, container: [], system: sys }).fatal).toBe(false);
    expect(diagnose({ ageMs: 60_000, container: ["loading weights /workspace/weights/h3-base"], system: sys })).toMatchObject({ phase: "running", fatal: false });
    expect(diagnose({ ageMs: 60_000, container: ["bash: line 17: /opt/fastvideo-rs/bin/fv-serve: No such file or directory", "bash: line 17: /opt/fastvideo-rs/bin/fv-serve: No such file or directory"], system: [] }).phase).toBe("crashloop");
  });
  it("copies the tail into the log store once, scrubbed, with Runpod targets; search by source, follow with after_id", async () => {
    const { env } = mkEnv({ RUNPOD_HAPI: "https://hapi.test" } as Partial<Env>);
    let tail = { container: ["2026-10-06T20:56:10.1Z " + CRASH], system: ["2026-10-06T20:55:42Z create container img", "2026-10-06T20:55:43Z leaked rpa_SECRETKEY123"] };
    globalThis.fetch = vi.fn(async (u: any, init: any) => {
      expect(String(u)).toBe("https://hapi.test/pod/pod1/logs");
      expect(init.headers.authorization).toBe("Bearer rpa_SECRETKEY123");
      return json(tail);
    }) as any;
    const c1 = await captureRunpodLogs(env, "pod1", "c_1");
    expect(c1?.added).toBe(3);
    expect(c1?.container).toEqual([CRASH]);
    expect((await captureRunpodLogs(env, "pod1", "c_1"))?.added).toBe(0);
    tail = { container: [...tail.container, "2026-10-06T20:56:12.5Z " + CRASH], system: tail.system };
    expect((await captureRunpodLogs(env, "pod1", "c_1"))?.added).toBe(1);
    const all = await searchLogs(env, { pod: "pod1" });
    expect(all.map((l: any) => [l.target, l.level])).toEqual([
      ["runpod.system", "info"],
      ["runpod.system", "info"],
      ["runpod.container", "error"],
      ["runpod.container", "error"],
    ]);
    expect(JSON.stringify(all)).not.toContain("rpa_SECRETKEY123");
    expect(JSON.stringify(all)).toContain("[redacted]");
    expect((await searchLogs(env, { pod: "pod1", source: "runpod" })).length).toBe(4);
    expect((await searchLogs(env, { pod: "pod1", source: "serve" })).length).toBe(0);
    const page = await searchLogs(env, { pod: "pod1", after_id: all[0].id, limit: 2 });
    expect(page.map((l: any) => l.id)).toEqual([all[1].id, all[2].id]);
    expect((env.LOGS as any).objs.size).toBe(2);
    // A gone pod: no log, no error.
    globalThis.fetch = vi.fn(async () => json({ error: "" }, 403)) as any;
    expect(await captureRunpodLogs(env, "pod1", "c_1")).toBeNull();
    await saveDiagnosis(env, "pod1", "c_1", { phase: "crashloop", detail: CRASH, fatal: true });
    expect(await getDiagnosis(env, "pod1")).toMatchObject({ phase: "crashloop", fatal: true });
  });
});

describe("up: an early, clear verdict", () => {
  const v = (id: string, status: PoolView["status"], detail?: string): PoolView => ({ id, status, detail });
  it("the h3-and-ltx start: names the crash loop and the pools without stock, without waiting 30 min", () => {
    const pools = [v("fake", "ready"), v("h3-turbo", "failed", `crash loop: ${CRASH}`), v("h3-max", "starting", "pulling"), v("h3-ref2v", "no_stock", "no NVIDIA RTX PRO 6000 in EUR-IS-1"), v("ltx-ref2v", "no_stock")];
    expect(summarize(pools)).toBe(
      `ready 1/5: fake | starting: h3-max (pulling) | failed: h3-turbo (crash loop: ${CRASH}) | no stock: h3-ref2v (no NVIDIA RTX PRO 6000 in EUR-IS-1), ltx-ref2v`,
    );
    expect(upOutcome(pools, 60_000)).toEqual({ done: false });
    pools[2] = v("h3-max", "failed", "crash loop");
    const out = upOutcome(pools, 90_000);
    expect(out.done).toBe(true);
    const err = (out as { error: string }).error;
    expect(err).toContain("h3-turbo: failed (crash loop: fv-serve: config: FV_AUTH_MODE=trust-edge");
    expect(err).toContain("h3-ref2v: no stock");
    expect(err).toContain("Ready: fake");
  });
  it("done when all are ready; a timeout names what is still starting; nothing ready says so", () => {
    expect(upOutcome([v("a", "ready"), v("b", "ready")], 1)).toEqual({ done: true });
    expect((upOutcome([v("a", "ready"), v("b", "starting", "running: loading weights")], 31 * 60_000) as any).error).toMatch(/b: not ready after 30 min \(running: loading weights\)/);
    expect((upOutcome([v("a", "no_stock")], 1) as any).error).toMatch(/No pool is ready: stop the cluster/);
  });
});

describe("standalone pods: the spec", () => {
  it("a preset → a one-pool direct cluster in EU with the volume, a deadline and the pod's env", () => {
    const { spec, env } = standaloneSpec({ name: "h3-solo", preset: "h3-turbo", channel: "latest", deadline_min: 45, idle_stop_min: 20, env: { FASTVIDEO_ATTN_SAGE: "0", HF_TOKEN: { value: "hf_x", secret: true } } });
    expect(spec).toMatchObject({ name: "h3-solo", control_plane: "direct", regions: ["eu"], image: { channel: "latest" }, cap_s: 2700, auto_stop_idle_min: 20, log_shipping: true });
    expect(spec.pools).toHaveLength(1);
    expect(spec.pools[0]).toMatchObject({ id: STANDALONE_POOL, variant: "h3-turbo", count: 1, compute: "GPU", volume: true, config: "/etc/fv/runpod.toml", regions: ["eu"] });
    expect(env).toEqual([
      { key: "FASTVIDEO_ATTN_SAGE", value: "0", secret: false },
      { key: "HF_TOKEN", value: "hf_x", secret: true },
    ]);
  });
  it("custom variant on CPU, a DC alias, a GPU type, an image ref or a sha", () => {
    const cpu = standaloneSpec({ name: "fake1", variant: "cpu", compute: "CPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], dc: "EUR-IS-1" }).spec;
    expect(cpu.pools[0]).toMatchObject({ compute: "CPU", volume: false, fake_models: ["fake-wan"] });
    expect(cpu.image).toEqual({ channel: "stable" });
    const g = standaloneSpec({ name: "g", preset: "ltx-pro", gpu_type: "NVIDIA RTX PRO 6000 Blackwell Server Edition", sha: "b51ddcd" }).spec;
    expect(g.image).toEqual({ sha: "b51ddcd" });
    expect(g.pools[0]!.gpu_types).toEqual(["NVIDIA RTX PRO 6000 Blackwell Server Edition"]);
    expect(g.pools[0]!.config_toml).toContain("ltx");
    const r = standaloneSpec({ name: "r", preset: "h3-max", image: "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "a".repeat(64) }).spec;
    expect(r.pools[0]!.image).toMatch(/@sha256:a{64}$/);
  });
  it("refuses what a cluster would refuse, and more", () => {
    const bad = (x: any) => () => standaloneSpec({ name: "x1", preset: "h3-turbo", ...x });
    expect(bad({ region: "us" })).toThrow(/EU only/);
    expect(bad({ dc: "US-CA-2" })).toThrow(/EU only/);
    expect(bad({ env: { FV_ADMIN_TOKEN: "x" } })).toThrow(/set by the controller/);
    expect(bad({ env: { "bad key": "x" } })).toThrow(/invalid variable name/);
    expect(bad({ deadline_min: 2 })).toThrow(/deadline_min/);
    expect(bad({ idle_stop_min: 1 })).toThrow(/idle_stop_min/);
    expect(bad({ channel: "latest", sha: "abcdef1" })).toThrow(/at most one/);
    expect(bad({ preset: "nope" })).toThrow(/no pool preset/);
    expect(bad({ name: "Bad Name" })).toThrow(/name/);
    expect(() => standaloneSpec({ name: "x" } as any)).toThrow(/preset or variant/);
  });
  it("its worker env carries the backstop and its boot runs the watchdog; a direct cluster's does not", () => {
    const { spec } = standaloneSpec({ name: "h3-solo", preset: "h3-turbo" });
    const base: EnvCtx = { spec, state: { images: {}, workers: {} }, secrets: { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_x", ingest_token: "fvi_x" }, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", ingestUrl: "https://ctl/ingest/v1/logs" };
    const e = workerSystemEnv({ ...base, backstop: true }, spec.pools[0]!, "img");
    expect(e).toMatchObject({ FV_WORKER_DIRECT: "1", FV_ADMIN_TOKEN: "fvadm_x", FV_CLUSTER_DEADLINE: "1700000000", FV_MIN_BALANCE: String(spec.min_balance), FV_BACKSTOP_API_KEY: "rpa_K", FV_LOG_SHIP_TOKEN: "fvi_x" });
    expect(e.FV_DISPATCH_FRONT).toBeUndefined();
    const pl = workerPlacements(spec, spec.pools[0]!)[0]!;
    const payload = workerCreatePayload("fv-pod-h3-solo-1006", "img", spec.pools[0]!, pl, e);
    expect(payload.dockerStartCmd).toEqual([WATCHDOG_WORKER_BOOT]);
    expect(payload).toMatchObject({ networkVolumeId: "jg48s6o1w0", dataCenterIds: ["EUR-IS-1"], gpuTypeIds: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] });
    const plain = workerSystemEnv(base, spec.pools[0]!, "img");
    expect(plain.FV_CLUSTER_DEADLINE).toBeUndefined();
    expect(bootFor(plain)).toBe(WORKER_BOOT);
    expect(bootFor({ FV_DISPATCH_FRONT: "1" })).toBe(EDGE_WORKER_BOOT);
    // The watchdog boot is the worker boot plus the edge boot's watchdog.
    const wd = /\n\(\n[\s\S]*?\n\) &\n/.exec(EDGE_WORKER_BOOT)![0];
    expect(WATCHDOG_WORKER_BOOT.replace(wd.slice(1), "")).toBe(WORKER_BOOT);
    expect(ownerOf({ name: "h3-solo", source: "standalone" })).toBe("pod:h3-solo");
    expect(ownerOf({ name: "c", source: "controller" })).toBe("cluster:c");
  });
});

describe("standalone pods: the API", () => {
  const call = (env: Env, path: string, init: { method?: string; body?: unknown; headers?: Record<string, string> } = {}) =>
    _app.fetch(new Request(`https://ctl.test${path}`, { method: init.method || "GET", headers: { "content-type": "application/json", ...(init.headers || {}) }, body: init.body === undefined ? undefined : JSON.stringify(init.body) }), env, { waitUntil() {}, passThroughOnException() {} } as any);
  it("launch (admin), list, view, start, stop, delete; read tokens and sessions without CSRF are refused", async () => {
    const { env, ops } = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const launch = { name: "h3-solo", preset: "h3-turbo", channel: "latest", deadline_min: 30, env: { FOO: "bar", TOK: { value: "s3cret-value", secret: true } } };
    expect((await call(env, "/api/standalone", { method: "POST", body: launch, headers: read })).status).toBe(403);
    // A cookie session needs the CSRF token (requireAuth): a forged cookie is refused before any of this.
    expect((await call(env, "/api/standalone", { method: "POST", body: launch, headers: { cookie: "__Host-fvc_session=x.y" } })).status).toBe(401);
    expect((await call(env, "/api/standalone", { method: "POST", body: { ...launch, env: { FV_INTERNAL_TOKEN: "x" } }, headers: admin })).status).toBe(400);

    const r = await call(env, "/api/standalone", { method: "POST", body: launch, headers: admin });
    expect(r.status).toBe(201);
    const j: any = await r.json();
    expect(j.pod).toMatchObject({ name: "h3-solo", status: "defined", definition: { variant: "h3-turbo", image: { channel: "latest" }, region: "eu", dc: "EUR-IS-1", volume: "jg48s6o1w0", deadline_min: 30 }, pod: null });
    expect(j.operation).toMatch(/^op_/);
    expect(ops.calls.find((x) => x.path === "/op/start")?.body).toMatchObject({ kind: "up", params: { skip_price_check: false } });
    const cl = await getCluster(env, "h3-solo");
    expect(cl.source).toBe("standalone");
    // The env went to the pod's definition (cluster scope); the secret sealed.
    const vars = await env.DB.prepare("SELECT key, value, secret FROM env_vars WHERE scope = 'cluster' AND scope_id = ? ORDER BY key").bind(cl.id).all<any>();
    expect(vars.results.map((v: any) => [v.key, v.secret])).toEqual([["FOO", 0], ["TOK", 1]]);
    expect(vars.results.find((v: any) => v.key === "TOK").value).not.toContain("s3cret-value");
    const audit = await env.DB.prepare("SELECT after FROM audit WHERE action = 'standalone.launch'").first<any>();
    expect(audit.after).not.toContain("s3cret-value");
    // Listed as standalone, not among clusters (unless all=1); readable with a read token.
    expect(((await (await call(env, "/api/standalone", { headers: read })).json()) as any).pods.map((p: any) => p.name)).toEqual(["h3-solo"]);
    expect(((await (await call(env, "/api/clusters", { headers: read })).json()) as any).clusters).toEqual([]);
    expect(((await (await call(env, "/api/clusters?all=1", { headers: read })).json()) as any).clusters[0]).toMatchObject({ name: "h3-solo", kind: "standalone" });
    expect((await call(env, "/api/standalone/h3-solo", { headers: read })).status).toBe(200);
    // Start / stop / extend go through the same operations; a read token cannot.
    expect((await call(env, "/api/standalone/h3-solo/stop", { method: "POST", body: {}, headers: read })).status).toBe(403);
    expect((await call(env, "/api/standalone/h3-solo/stop", { method: "POST", body: {}, headers: admin })).status).toBe(202);
    expect(ops.calls.filter((x) => x.path === "/op/start").at(-1)?.body).toMatchObject({ kind: "down", params: { reason: "stop" } });
    expect((await call(env, "/api/standalone/h3-solo/start", { method: "POST", body: {}, headers: admin })).status).toBe(202);
    expect((await call(env, "/api/standalone/h3-solo/extend", { method: "POST", body: { minutes: 15 }, headers: admin })).status).toBe(202);
    // A cluster is not a standalone pod, and a standalone pod does not scale past one.
    await env.DB.prepare("UPDATE clusters SET source = 'controller' WHERE id = ?").bind(cl.id).run();
    expect((await call(env, "/api/standalone/h3-solo", { headers: read })).status).toBe(404);
    await env.DB.prepare("UPDATE clusters SET source = 'standalone' WHERE id = ?").bind(cl.id).run();
    expect((await call(env, `/api/clusters/${cl.id}/scale`, { method: "POST", body: { pool: "pod", count: 2 }, headers: admin })).status).toBe(400);
    // Delete without a pod: the definition and its env go now.
    expect((await call(env, "/api/standalone/h3-solo", { method: "DELETE", headers: admin })).status).toBe(200);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM env_vars WHERE scope_id = ?").bind(cl.id).first<any>()).toEqual({ n: 0 });
    expect((await call(env, "/api/standalone/h3-solo", { headers: read })).status).toBe(404);
  });
  it("delete with a pod stops it first and deletes the definition after (down with delete_definition)", async () => {
    const { env, ops } = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    await call(env, "/api/standalone", { method: "POST", body: { name: "solo", preset: "h3-max", start: false }, headers: admin });
    const cl = await getCluster(env, "solo");
    cl.state.workers[STANDALONE_POOL] = [{ pod: "podx", pool: STANDALONE_POOL, dph: 2.09, created: 1, image: "img" }];
    await env.DB.prepare("UPDATE clusters SET state = ?, status = 'running' WHERE id = ?").bind(JSON.stringify(cl.state), cl.id).run();
    const r = await call(env, "/api/standalone/solo", { method: "DELETE", headers: admin });
    expect(r.status).toBe(202);
    expect(ops.calls.filter((x) => x.path === "/op/start").at(-1)?.body).toMatchObject({ kind: "down", params: { delete_definition: true } });
    expect((await call(env, "/api/standalone/solo", { headers: admin })).status).toBe(200);
  });
  it("pod status and logs as JSON for any controller pod", async () => {
    const { env } = mkEnv();
    const read = { authorization: `Bearer ${(await mintApiToken(env, "r", "read", "test")).token}` };
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    await call(env, "/api/standalone", { method: "POST", body: { name: "solo", preset: "h3-turbo", start: false }, headers: admin });
    const cl = await getCluster(env, "solo");
    await env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, created_at, status) VALUES ('podz', ?, 'worker', 'pod', 'workers', 'img', 1, 'creating')").bind(cl.id).run();
    await env.DB.prepare("INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg) VALUES (?, 'podz', 1, 'error', 'runpod.container', ?), (?, 'podz', 2, 'info', 'fastvideo_serve', 'shipped')").bind(cl.id, CRASH, cl.id).run();
    await saveDiagnosis(env, "podz", cl.id, { phase: "crashloop", detail: CRASH, fatal: true });
    const st: any = await (await call(env, "/api/pods/podz/status", { headers: read })).json();
    expect(st).toMatchObject({ pod_id: "podz", kind: "standalone", controller: { cluster: "solo", pool: "pod" }, boot: { phase: "crashloop" }, logs: { lines: 2 } });
    const lg: any = await (await call(env, "/api/pods/podz/logs?source=runpod", { headers: read })).json();
    expect(lg.lines.map((l: any) => l.msg)).toEqual([CRASH]);
    const all: any = await (await call(env, "/api/pods/podz/logs", { headers: read })).json();
    expect(all.lines).toHaveLength(2);
    const next: any = await (await call(env, `/api/pods/podz/logs?after_id=${all.next_after_id}`, { headers: read })).json();
    expect(next).toMatchObject({ lines: [], next_after_id: all.next_after_id });
    expect((await call(env, "/api/pods/nope/status", { headers: read })).status).toBe(404);
  });
});

describe("boot timeline", () => {
  // Lines from the 2026-10-06 measurement (h3-turbo, latest = b51ddcd, RTX PRO 6000, EUR-IS-1; pod created 21:50:38).
  const T = (s: string) => Date.parse(s);
  const lines = [
    { stream: "system" as const, ts: T("2026-10-06T21:50:38Z"), text: "create container ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:ff54" },
    { stream: "system" as const, ts: T("2026-10-06T21:50:39Z"), text: "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:ff54 Pulling from zaitrarrio/fastvideo-rs-serve" },
    { stream: "system" as const, ts: T("2026-10-06T21:50:46Z"), text: "create container: still fetching image ghcr.io/x" },
    { stream: "system" as const, ts: T("2026-10-06T21:51:00Z"), text: "Digest: sha256:ff54" },
    { stream: "system" as const, ts: T("2026-10-06T21:51:02Z"), text: "Status: Image is up to date for ghcr.io/x" },
    { stream: "system" as const, ts: T("2026-10-06T21:51:02Z"), text: "start container for ghcr.io/x: begin" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:02.100Z"), text: "[fv-boot] start" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:02.200Z"), text: "[fv-boot] volume: /workspace/weights (41 trees)" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:02.804Z"), text: "2026-10-06T21:51:02.803592Z  INFO fv_serve: fv-serve 0.1.2 (b51ddcd 2026-10-06T20:50:35Z) h3-turbo latest sha256:ff54a4b2886e config={}" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:04.914Z"), text: "2026-10-06T21:51:04.913905Z  INFO fastvideo_engine_service::cuda::backend: loading model=fasth3 dit_gb=1.106e-6 free_gb=101.2" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:06.750Z"), text: "2026-10-06T21:51:06.749504Z  INFO fastvideo_serve::edge_link: worker: connected to the dispatcher scope=family:h3 held=0 connect_ms=713" },
    { stream: "container" as const, ts: T("2026-10-06T21:51:46.177Z"), text: '[fastvideo] load/io h3 text_encoder {"wall_s":41.26,"viewed_gb":25.95,"prefetch_read_gb":26.91,"viewed_gbps":0.63}' },
    { stream: "container" as const, ts: T("2026-10-06T21:52:34.535Z"), text: "[fastvideo] h3 i2v encoder: resident (71.5 GiB free covers the vision tower (2.0 GiB) and the resident plan (64.6 GiB)): vision tower 1.09 GiB loaded in 21.1 s; the language model is the resident text encoder" },
    { stream: "container" as const, ts: T("2026-10-06T21:53:54.776Z"), text: '[fastvideo] load/io h3 dit {"wall_s":169.86,"viewed_gb":98.76,"viewed_gbps":0.58}' },
    { stream: "container" as const, ts: T("2026-10-06T21:54:06.317Z"), text: '[fastvideo] load/io h3 total {"wall_s":181.40,"viewed_gb":108.71,"viewed_gbps":0.60}' },
    { stream: "container" as const, ts: T("2026-10-06T21:54:06.366Z"), text: "2026-10-06T21:54:06.366098Z  INFO fastvideo_engine_service::cuda::backend: model resident model=fasth3 seconds=181.452138581" },
    { stream: "container" as const, ts: T("2026-10-06T21:55:06.676Z"), text: "2026-10-06T21:55:06.676082Z  INFO fastvideo_engine_service::cuda::backend: warmup done model=fasth3 seconds=60.309928506 runs=i2v 1344x768x124 34.1s, t2v 1344x768x124 26.2s" },
    { stream: "container" as const, ts: T("2026-10-06T21:55:06.677Z"), text: "FV-SERVE READY models=fasth3" },
  ];
  it("every phase from the Runpod and fv-serve lines; components with GB and GB/s; first seen wins", () => {
    const b = { t: { create: T("2026-10-06T21:50:38Z") } };
    expect(foldBoot(b, lines)).toEqual(["machine", "pull_start", "pull_end", "container_start", "boot_script", "volume", "serve_start", "load_start", "edge_link", "model_resident", "warmup_done", "serve_ready"]);
    expect(b).toMatchObject({ load_s: 181.452138581, warmup_s: 60.309928506, volume: "/workspace/weights (41 trees)", components: { text_encoder: { wall_s: 41.26, gb: 25.95, gbps: 0.63 }, dit: { wall_s: 169.86, gb: 98.76, gbps: 0.58 }, vision_tower: { wall_s: 21.1 } } });
    expect((b as any).components.total).toBeUndefined();
    expect(foldBoot(b, lines)).toEqual([]);
    const rows = bootRows(b as any);
    const at = (phase: string) => rows.find((r) => r.phase === phase)!;
    expect(at("image pull end")).toMatchObject({ t_s: 22, took_s: 21, detail: "pull 21 s" });
    expect(at("container start").t_s).toBe(24);
    expect(at("weights: dit")).toMatchObject({ took_s: 169.86, detail: "98.8 GB at 0.58 GB/s" });
    expect(at("weights: every component resident")).toMatchObject({ t_s: 208.4, detail: "load 181.5 s" });
    expect(at("warm-up done")).toMatchObject({ t_s: 268.7, took_s: 60.3 });
    expect(at("ready (at the edge / health)").at).toBeNull();
    expect(bootPhase(b as any, T("2026-10-06T21:55:10Z"))).toEqual({ phase: "serve_ready", since_s: 3 });
  });
  it("shipped tracing events (as the ingest renders them) mark the same phases", () => {
    const b = { t: {} };
    foldBoot(b, [
      { stream: "shipped", ts: 1, text: "fv_serve: fv-serve 0.1.2 (b51ddcd) build=x" },
      { stream: "shipped", ts: 2, text: "fastvideo_engine_service::cuda::backend: model resident model=fasth3 seconds=129.7" },
      { stream: "shipped", ts: 3, text: "fastvideo_serve::app: ready models=1" },
    ]);
    expect(b).toMatchObject({ t: { serve_start: 1, model_resident: 2, serve_ready: 3 }, load_s: 129.7 });
  });
  it("on the pod's record and in its log; ready from the DO; a pull past the deadline is re-placed", async () => {
    const { env } = mkEnv();
    await env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, created_at, status) VALUES ('podb', 'c_1', 'worker', 'h3', 'workers', 'img', ?, 'creating')").bind(T("2026-10-06T21:50:38Z")).run();
    expect(await recordBoot(env, "podb", "c_1", lines.slice(0, 4))).toEqual(["machine", "pull_start", "pull_end"]);
    expect(await recordBoot(env, "podb", "c_1", lines.slice(0, 4))).toEqual([]);
    await markReady(env, "podb", "c_1", T("2026-10-06T21:55:12Z"));
    const b = await getBoot(env, "podb");
    expect(b?.t).toMatchObject({ create: T("2026-10-06T21:50:38Z"), pull_end: T("2026-10-06T21:51:00Z"), ready: T("2026-10-06T21:55:12Z") });
    const log = await searchLogs(env, { pod: "podb" });
    expect(log.map((l: any) => l.msg)).toEqual(["boot: machine at +0.0 s", "boot: pull_start at +1.0 s", "boot: pull_end at +22.0 s (pull 21 s)", "boot: ready at +274.0 s"]);
    expect(log.every((l: any) => l.target === "fv-control.boot")).toBe(true);
    expect(await recordBoot(env, "nope", "c_1", lines)).toEqual([]);
    // Pulling for longer than the deadline on this host: replace, not fail the pool.
    const slow = { t: { create: 0, machine: 0, pull_start: 1000 } };
    expect(diagnose({ ageMs: PULL_DEADLINE_MS + 2000, container: [], system: [], boot: slow, now: PULL_DEADLINE_MS + 2000 })).toMatchObject({ phase: "slow_pull", fatal: true, replace: true });
    expect(diagnose({ ageMs: 60_000, container: [], system: ["x Pulling from y"], boot: slow, now: 60_000 }).phase).toBe("pulling");
  });
  it("the worker boot prints its markers (start, volume) on stderr", () => {
    expect(WORKER_BOOT).toMatch(/echo "\[fv-boot\] start" >&2/);
    expect(WORKER_BOOT).toMatch(/\[fv-boot\] volume: \/workspace\/weights/);
    expect(bootMatch("container", "[fv-boot] volume: /workspace/weights missing")).toEqual({ m: "volume", volume: "/workspace/weights missing" });
  });
});

describe("no secret in a diagnosis", () => {
  it("the diagnosis quotes the scrubbed line", async () => {
    const { env } = mkEnv({ RUNPOD_HAPI: "https://hapi.test" } as Partial<Env>);
    await env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, created_at, status) VALUES ('pods', 'c_1', 'worker', 'p', 'workers', 'img', 1, 'creating')").run();
    globalThis.fetch = vi.fn(async () => json({ container: ["2026-10-06T21:51:02Z token rpa_SECRETKEY123 printed"], system: [] })) as any;
    const d = await checkPod(env, "pods", "c_1", Date.now());
    expect(JSON.stringify(d)).not.toContain("rpa_SECRETKEY123");
    expect(JSON.stringify(await getDiagnosis(env, "pods"))).not.toContain("rpa_SECRETKEY123");
  });
});

describe("the log explorer (logquery.ts) sees captured Runpod lines and boot milestones", () => {
  it("as pod lines of the standalone pod, its pool and name resolved; /api/logs source + after_id and /api/logs/query agree", async () => {
    const { env } = mkEnv({ RUNPOD_HAPI: "https://hapi.test" } as Partial<Env>);
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "t", "admin", "test")).token}` };
    const call = (path: string) => _app.fetch(new Request(`https://ctl.test${path}`, { headers: admin }), env, { waitUntil() {}, passThroughOnException() {} } as any);
    await _app.fetch(new Request("https://ctl.test/api/standalone", { method: "POST", headers: { ...admin, "content-type": "application/json" }, body: JSON.stringify({ name: "solo-x", preset: "h3-turbo", start: false }) }), env, {} as any);
    const cl = await getCluster(env, "solo-x");
    const t0 = Date.now() - 60_000;
    await env.DB.prepare("INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, slot, image, created_at, status) VALUES ('podq00000000ab', ?, 'worker', 'pod', 'workers', 'img', ?, 'creating')").bind(cl.id, t0).run();
    const iso = (ms: number) => new Date(ms).toISOString();
    globalThis.fetch = vi.fn(async () => json({ container: [`${iso(t0 + 20_000)} [fv-boot] start`, `${iso(t0 + 21_000)} ${CRASH}`], system: [`${iso(t0 + 1000)} create container img`, `${iso(t0 + 2000)} img Pulling from x`, `${iso(t0 + 15_000)} Digest: sha256:abc`] })) as any;
    expect((await captureRunpodLogs(env, "podq00000000ab", cl.id))?.added).toBe(5);
    const q = await queryLogs(env, parseLogQuery({ src: "pod", pod: "podq00000000ab", since: "1h" }));
    const rows = q.lines.map((l) => [l.source, l.target, l.pool, l.cluster]);
    expect(rows).toContainEqual(["pod", "runpod.system", "pod", "solo-x"]);
    expect(rows).toContainEqual(["pod", "runpod.container", "pod", "solo-x"]);
    expect(rows).toContainEqual(["pod", "fv-control.boot", "pod", "solo-x"]);
    expect(q.lines.find((l) => l.target === "fv-control.boot" && /pull_end/.test(l.msg))?.msg).toBe("boot: pull_end at +15.0 s (pull 13 s)");
    expect(q.lines.find((l) => l.msg === CRASH)?.level).toBe("error");
    // By pool (the explorer resolves it through cluster_pods) and cluster.
    expect((await queryLogs(env, parseLogQuery({ src: "pod", pool: "pod", cluster: cl.id, since: "1h" }))).lines.length).toBe(q.lines.length);
    // The routes: the explorer's query, and /api/logs with source / after_id.
    const qr: any = await (await call(`/api/logs/query?src=pod&pod=podq00000000ab&since=1h`)).json();
    expect(qr.lines.length).toBe(q.lines.length);
    const rp: any = await (await call(`/api/logs?pod=podq00000000ab&source=runpod&level=trace`)).json();
    expect(rp.lines.map((l: any) => l.target)).toEqual(["runpod.system", "runpod.system", "runpod.system", "runpod.container", "runpod.container"]);
    const ctlLines: any = await (await call(`/api/logs?pod=podq00000000ab&source=control`)).json();
    expect(ctlLines.lines.every((l: any) => l.target === "fv-control.boot")).toBe(true);
    const after: any = await (await call(`/api/logs?pod=podq00000000ab&level=trace&after_id=${rp.lines[2].id}&limit=1`)).json();
    expect(after.lines).toHaveLength(1);
    expect(after.lines[0].id).toBeGreaterThan(rp.lines[2].id);
  });
});
