// The R2 weights store (src/r2weights.ts, weights-plan.ts; docs/ops/r2-weights.md): SigV4 presigning against AWS's
// published example, manifests pinned by SHA-256, the plan, weights_source r2 as the GMI / Brev default once every
// tree is published, the launch checks (GMI / Brev and Runpod), the pod env and boot; and the boot's weights step
// (R2_FETCH_SH) run in bash with the real scripts/gpu/r2-fetch.py against a fake R2 that checks every presigned URL.
import { createHash, createHmac } from "node:crypto";
import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { bootFor, workerCreatePayload, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { presetPool } from "../../src/cluster/spec";
import { launchTrees, poolTrees, PROVIDER_BOOT, providerEnv, providerIssues, weightsPlan, weightsSource } from "../../src/providers";
import { R2_FETCH_PY, R2_FETCH_SHA256 } from "../../src/r2-fetch-script";
import { encodeKey, manifestKey, objectKey, presignGet, presignPath, R2_FETCH_SH, r2Cfg, r2Manifest, r2Plan, r2Trees, type R2Manifest } from "../../src/r2weights";
import { validate } from "../../src/schemas";
import { d1 } from "./d1shim";

const ACCESS = "r2-access-key-id";
const SECRET = "r2-SECRET-access-key-0123456789";
const sha = (b: Buffer | string) => createHash("sha256").update(b).digest("hex");

/** A tree, its manifest (part size 1 MiB) and the R2_TREES pin. */
function makeTree(dest: string, rev: string, files: Record<string, Buffer>, part = 1 << 20) {
  const m: R2Manifest = {
    format: "fv-weights-manifest/1",
    dest,
    rev,
    part_size: part,
    total_bytes: Object.values(files).reduce((s, b) => s + b.length, 0),
    files: Object.entries(files).map(([path, b]) => ({ path, size: b.length, sha256: sha(b), parts: Array.from({ length: Math.ceil(b.length / part) }, (_, i) => sha(b.subarray(i * part, (i + 1) * part))) })),
  };
  const raw = Buffer.from(JSON.stringify(m));
  return { m, raw, pin: { rev, files: m.files.length, bytes: m.total_bytes, manifest_sha256: sha(raw), source: "test" } };
}
const rnd = (n: number, seed: number) => {
  const out = Buffer.alloc(n);
  let h = createHash("sha256").update(String(seed)).digest();
  for (let o = 0; o < n; o += 32) {
    h = createHash("sha256").update(h).digest();
    h.copy(out, o);
  }
  return out;
};

/** A fake R2: path-style objects, presigned GETs checked with node:crypto (an implementation independent of src's). */
function fakeR2() {
  const objects = new Map<string, Buffer>(); // "<bucket>/<key>"
  const gets: { key: string; range: string | null }[] = [];
  let bad = 0;
  const routes: Record<string, (req: any, res: any) => void> = {};
  const verify = (u: URL, host: string) => {
    const q = u.searchParams;
    const amz = q.get("X-Amz-Date") || "";
    const when = Date.UTC(+amz.slice(0, 4), +amz.slice(4, 6) - 1, +amz.slice(6, 8), +amz.slice(9, 11), +amz.slice(11, 13), +amz.slice(13, 15));
    if (Date.now() > when + Number(q.get("X-Amz-Expires")) * 1000) return false;
    const [ak, date, region] = (q.get("X-Amz-Credential") || "").split("/");
    const pairs = [...q.entries()].filter(([k]) => k !== "X-Amz-Signature").map(([k, v]) => [encodeURIComponent(k), encodeURIComponent(v)]).sort((a, b) => (a[0]! < b[0]! ? -1 : 1));
    const canonical = ["GET", u.pathname, pairs.map(([k, v]) => `${k}=${v}`).join("&"), `host:${host}\n`, "host", "UNSIGNED-PAYLOAD"].join("\n");
    const toSign = ["AWS4-HMAC-SHA256", amz, `${date}/${region}/s3/aws4_request`, sha(canonical)].join("\n");
    let k: Buffer = createHmac("sha256", "AWS4" + SECRET).update(date!).digest();
    for (const p of [region!, "s3", "aws4_request"]) k = createHmac("sha256", k).update(p).digest();
    return ak === ACCESS && createHmac("sha256", k).update(toSign).digest("hex") === q.get("X-Amz-Signature");
  };
  const srv: Server = createServer((req, res) => {
    const u = new URL(req.url!, "http://x");
    if (routes[u.pathname]) return routes[u.pathname]!(req, res);
    if (!verify(u, req.headers.host!)) {
      bad++;
      return res.writeHead(403).end("<Error><Code>AccessDenied</Code></Error>");
    }
    const key = decodeURIComponent(u.pathname.slice(1));
    const obj = objects.get(key);
    if (!obj) return res.writeHead(404).end("<Error><Code>NoSuchKey</Code></Error>");
    const m = /bytes=(\d+)-(\d+)/.exec(req.headers.range || "");
    gets.push({ key, range: req.headers.range || null });
    if (!m) return res.writeHead(200, { "content-length": obj.length }).end(obj);
    const a = +m[1]!, b = Math.min(+m[2]!, obj.length - 1);
    res.writeHead(206, { "content-length": b - a + 1, "content-range": `bytes ${a}-${b}/${obj.length}` }).end(obj.subarray(a, b + 1));
  });
  return {
    objects,
    gets,
    routes,
    bad: () => bad,
    async start() {
      await new Promise<void>((r) => srv.listen(0, "127.0.0.1", () => r()));
      return `http://127.0.0.1:${(srv.address() as any).port}`;
    },
    stop: () => new Promise<void>((r) => srv.close(() => r())),
  };
}

const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_x",
    PUBLIC_URL: "https://fvc.test",
    CF_ACCOUNT_ID: "acct",
    GMI_API_KEY: "gmi_x",
    GMI_PRODUCTS: "container.h200.x1",
    GMI_BUDGET_USD: "20",
    WEIGHTS_R2_ACCESS_KEY_ID: ACCESS,
    WEIGHTS_R2_SECRET_ACCESS_KEY: SECRET,
    ...over,
  }) as unknown as Env;

afterEach(() => vi.unstubAllGlobals());

describe("SigV4 presigning", () => {
  it("matches AWS's published presigned-URL example (sigv4-query-string-auth)", async () => {
    const u = await presignPath("https", "examplebucket.s3.amazonaws.com", "/test.txt", "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", 86400, new Date(Date.UTC(2013, 4, 24)), "us-east-1");
    expect(u).toBe(
      "https://examplebucket.s3.amazonaws.com/test.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20130524T000000Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404",
    );
  });
  it("path-style on the account endpoint, keys encoded per segment (RFC 3986), the secret never in the URL", async () => {
    const { cfg } = r2Cfg(mkEnv());
    expect(cfg).toMatchObject({ endpoint: "https://acct.r2.cloudflarestorage.com", bucket: "fv-weights", ttlS: 3600 });
    const u = await presignGet(cfg!, "weights/a b/r1/x(1)!.json");
    expect(u.startsWith("https://acct.r2.cloudflarestorage.com/fv-weights/weights/a%20b/r1/x%281%29%21.json?")).toBe(true);
    expect(u).not.toContain(SECRET);
    expect(encodeKey("a/b c/*")).toBe("a/b%20c/%2A");
  });
  it("off with a reason without the key; the TTL is clamped", () => {
    expect(r2Cfg(mkEnv({ WEIGHTS_R2_SECRET_ACCESS_KEY: undefined })).off).toMatch(/WEIGHTS_R2_ACCESS_KEY_ID/);
    expect(r2Cfg(mkEnv({ WEIGHTS_R2_URL_TTL_S: "5" })).cfg!.ttlS).toBe(300);
    expect(r2Cfg(mkEnv({ WEIGHTS_R2_URL_TTL_S: "999999" })).cfg!.ttlS).toBe(21600);
  });
});

describe("manifests and the plan (against the fake R2)", () => {
  const r2 = fakeR2();
  let base = "";
  const t = makeTree("demo", "rev1", { "config.json": Buffer.from("{}"), "transformer/model.safetensors": rnd(2_500_000, 1) });
  const env = () => mkEnv({ WEIGHTS_R2_ENDPOINT: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: t.pin }) });
  beforeAll(async () => {
    base = await r2.start();
    r2.objects.set(`fv-weights/${manifestKey("demo", "rev1")}`, t.raw);
  });
  afterAll(() => r2.stop());
  it("reads a published manifest; refuses one whose bytes are not the pinned ones", async () => {
    const e = env();
    const { cfg } = r2Cfg(e);
    expect((await r2Manifest(e, cfg!, "demo")).total_bytes).toBe(t.m.total_bytes);
    const tampered = mkEnv({ WEIGHTS_R2_ENDPOINT: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: { ...t.pin, manifest_sha256: "0".repeat(64) } }) });
    await expect(r2Manifest(tampered, r2Cfg(tampered).cfg!, "demo")).rejects.toThrow(/not the one scripts\/gpu\/weights-r2.tsv pins/);
    await expect(r2Manifest(e, cfg!, "nope")).rejects.toThrow(/not in the R2 weights store/);
  });
  it("a manifest with an unsafe path is refused", async () => {
    const bad = makeTree("evil", "r", { "../x": Buffer.from("x") });
    r2.objects.set(`fv-weights/${manifestKey("evil", "r")}`, bad.raw);
    const e = mkEnv({ WEIGHTS_R2_ENDPOINT: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ evil: bad.pin }) });
    await expect(r2Manifest(e, r2Cfg(e).cfg!, "evil")).rejects.toThrow(/bad entry/);
  });
  it("the plan: every file with a presigned URL the bucket accepts, expiring with the TTL", async () => {
    const e = env();
    const at = new Date();
    const plan = await r2Plan(e, r2Cfg(e).cfg!, ["demo"], at);
    expect(plan.format).toBe("fv-weights-plan/1");
    expect(plan.expires_at).toBe(Math.floor(at.getTime() / 1000) + 3600);
    expect(plan.trees[0]!.files.map((f) => f.path)).toEqual(["config.json", "transformer/model.safetensors"]);
    r2.objects.set(`fv-weights/${objectKey("demo", "rev1", "config.json")}`, Buffer.from("{}"));
    const before = r2.bad();
    const res = await fetch(plan.trees[0]!.files[0]!.url);
    expect([res.status, await res.text(), r2.bad()]).toEqual([200, "{}", before]);
    const forged = plan.trees[0]!.files[0]!.url.replace("config.json", "transformer/model.safetensors");
    expect((await fetch(forged)).status).toBe(403);
  });
});

describe("weights_source r2: the default, the checks, the env", () => {
  const h3 = presetPool("h3-turbo")!;
  const pin = (trees: string[]) => JSON.stringify(Object.fromEntries(trees.map((t) => [t, { rev: `rev-${t}`, files: 1, bytes: 100e9, manifest_sha256: "a".repeat(64), source: "test" }])));
  const gmiPool = (over: any = {}) => ({ ...h3, provider: "gmi", provider_gpu: "container.h200.x1", volume: false, ...over }) as any;
  const trees = poolTrees(gmiPool()).filter((t) => t !== "auxiliary");
  it("GMI / Brev: r2 once every tree of the pool is published, else none; a source given stays", () => {
    expect(trees).toContain("h3-base");
    expect(weightsSource(gmiPool(), mkEnv())).toBe("none");
    expect(weightsSource(gmiPool(), mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees.slice(1)) }))).toBe("none");
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    expect(weightsSource(gmiPool(), all)).toBe("r2");
    expect(weightsSource(gmiPool({ weights_source: "hub" }), all)).toBe("hub");
    expect(weightsSource({ ...h3 } as any, all)).toBe("volume");
    const plan = weightsPlan(gmiPool(), all);
    expect([plan.source, plan.unsupported]).toEqual(["r2", []]);
    // Only the aux rows travel in the env: the trees come from the plan endpoint.
    expect(plan.tsv.split("\n").filter(Boolean).every((l) => l.startsWith("aux\t"))).toBe(true);
    expect(launchTrees(plan, all)).toEqual(Object.fromEntries(trees.map((t) => [t, `rev-${t}`])));
  });
  it("needs no download approval; refuses what is not published, or no store configured", () => {
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    expect(providerIssues(all, { pools: [gmiPool({ weights_source: "r2" })] })).toEqual([]);
    const some = providerIssues(mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees.slice(1)) }), { pools: [gmiPool({ weights_source: "r2" })] });
    expect(some.map((i) => i.message)).toEqual([expect.stringMatching(new RegExp(`not in the R2 weights store yet: ${trees[0]}`))]);
    const off = providerIssues(mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees), WEIGHTS_R2_ACCESS_KEY_ID: undefined }), { pools: [gmiPool({ weights_source: "r2" })] });
    expect(off.map((i) => i.message)).toEqual([expect.stringMatching(/weights_source r2: the R2 weights store is not configured/)]);
    expect(validate("pool", gmiPool({ weights_source: "r2" })).ok).toBe(true);
  });
  it("the GMI / Brev pod env: the plan and fetcher URLs and the fetcher's hash, never an R2 key", () => {
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    const r = providerEnv(all, {}, { provider: "gmi", key: "gmi:fv-pod-a-1", name: "fv-pod-a-1", deadlineMs: 1.8e12, reportToken: "ingest_tok", weights: weightsPlan(gmiPool(), all), scriptsRef: "main" });
    expect(r.env).toMatchObject({ FV_WEIGHTS_SOURCE: "r2", FV_WEIGHTS_PLAN_URL: "https://fvc.test/ingest/v1/weights-plan", FV_R2_FETCH_URL: "https://fvc.test/ingest/v1/r2-fetch.py", FV_R2_FETCH_SHA256: R2_FETCH_SHA256, FV_ENDPOINT_REPORT_TOKEN: "ingest_tok" });
    expect(r.env.FV_SCRIPTS_URL).toBeUndefined();
    expect(JSON.stringify(r.env)).not.toContain(SECRET);
    expect(JSON.stringify(r.env)).not.toContain(ACCESS);
    expect(PROVIDER_BOOT).toContain(`python3 "$R2T/r2-fetch.py"`);
    expect(PROVIDER_BOOT.indexOf("r2-fetch.py")).toBeLessThan(PROVIDER_BOOT.indexOf("cloudflared tunnel"));
    expect(R2_FETCH_PY).toContain("fv-weights-plan/1");
    expect(createHash("sha256").update(R2_FETCH_PY).digest("hex")).toBe(R2_FETCH_SHA256);
  });
  it("Runpod: r2 instead of the volume: no volume, a container disk sized for the trees, the r2 boot; checks", () => {
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    const pool = { ...h3, weights_source: "r2" } as any;
    expect(validate("pool", pool).ok).toBe(true);
    expect(validate("pool", { ...pool, weights_source: "hub" }).ok).toBe(false);
    expect(providerIssues(all, { pools: [pool] })).toEqual([]);
    expect(providerIssues(all, { pools: [{ ...pool, volume: true }] }).map((i) => i.path.join("."))).toEqual(["pools.0.volume"]);
    expect(providerIssues(all, { pools: [{ ...pool, container_disk_gb: 50 }] })[0]!.message).toMatch(/too small/);
    const ctx = { spec: { control_plane: "direct", auth: "keys", log_level: "info", image: {} }, state: {}, secrets: { internal_token: "i", url_signing_key: "u", ingest_token: "ingest_tok" }, deadlineMs: 0, runpodApiKey: "rpa_x", ingestUrl: "https://fvc.test/ingest/v1/logs", env: all } as unknown as EnvCtx;
    const e = workerSystemEnv(ctx, pool, "img@sha256:" + "0".repeat(64));
    expect(e).toMatchObject({ FV_WEIGHTS_SOURCE: "r2", FV_WEIGHTS_PLAN_URL: "https://fvc.test/ingest/v1/weights-plan", FV_ENDPOINT_REPORT_TOKEN: "ingest_tok" });
    const gb = Number(e.FV_R2_DISK_GB);
    expect(gb).toBe(Math.min(500, Math.ceil(trees.length * 100 * 1.15 + 30)));
    const p = workerCreatePayload("fv-ctl-x", "img", pool, { region: "eu", dc: "EUR-IS-1", gpu: "g" }, e) as any;
    expect(p.networkVolumeId).toBeUndefined();
    expect(p.containerDiskInGb).toBe(gb);
    expect(p.dockerStartCmd[0]).toContain('export FV_POD_ID="${FV_POD_ID:-${RUNPOD_POD_ID:-}}"');
    expect(p.dockerStartCmd[0].indexOf("r2-fetch.py")).toBeLessThan(p.dockerStartCmd[0].indexOf("exec /opt/fastvideo-rs/bin/fv-serve"));
    // The volume path is unchanged without r2.
    const v = workerSystemEnv({ ...ctx, env: all } as any, { ...h3 } as any, "img");
    expect(v.FV_WEIGHTS_SOURCE).toBeUndefined();
    expect((workerCreatePayload("fv-ctl-x", "img", { ...h3 } as any, { region: "eu", dc: "EUR-IS-1", gpu: "g" }, v) as any).networkVolumeId).toBeTruthy();
    expect(bootFor(v)).not.toContain("r2-fetch.py");
  });
  it("r2Trees: the repo's record plus valid WEIGHTS_R2_EXTRA_TREES rows only", () => {
    expect(Object.keys(r2Trees(mkEnv({ WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ ok: { rev: "r", files: 1, bytes: 1, manifest_sha256: "b".repeat(64) }, bad: { rev: "r", bytes: 1, manifest_sha256: "x" } }) })))).toContain("ok");
    expect(Object.keys(r2Trees(mkEnv({ WEIGHTS_R2_EXTRA_TREES: "{nope" })))).not.toContain("bad");
  });
});

const py = (() => {
  try {
    return existsSync("/usr/bin/python3") || existsSync("/usr/local/bin/python3");
  } catch {
    return false;
  }
})();

describe.skipIf(!py)("the boot's weights step (R2_FETCH_SH) with the real r2-fetch.py against a fake R2", () => {
  const r2 = fakeR2();
  let base = "";
  const files = { "config.json": Buffer.from('{"x": 1}'), "text_encoder/model.safetensors": rnd(1_700_000, 2), "transformer/model.safetensors": rnd(3_300_000, 3) };
  const t = makeTree("demo", "rev1", files);
  const reports: any[] = [];
  let env: Env;
  beforeAll(async () => {
    base = await r2.start();
    env = mkEnv({ WEIGHTS_R2_ENDPOINT: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ demo: t.pin }), PUBLIC_URL: base });
    r2.objects.set(`fv-weights/${manifestKey("demo", "rev1")}`, t.raw);
    for (const [p, b] of Object.entries(files)) r2.objects.set(`fv-weights/${objectKey("demo", "rev1", p)}`, b);
    // fv-control's side, as weights-plan.ts and index.ts answer it.
    r2.routes["/ingest/v1/weights-plan"] = (req, res) => {
      let body = "";
      req.on("data", (d: Buffer) => (body += d));
      req.on("end", async () => {
        if (req.headers.authorization !== "Bearer ingest_tok" || JSON.parse(body).pod !== "gmi:fv-pod-a-1") return res.writeHead(401).end();
        res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(await r2Plan(env, r2Cfg(env).cfg!, ["demo"])));
      });
    };
    r2.routes["/ingest/v1/r2-fetch.py"] = (_req, res) => res.writeHead(200).end(R2_FETCH_PY);
    r2.routes["/ingest/v1/endpoint"] = (req, res) => {
      let body = "";
      req.on("data", (d: Buffer) => (body += d));
      req.on("end", () => (reports.push(JSON.parse(body)), res.writeHead(200).end("{}")));
    };
  });
  afterAll(() => r2.stop());
  const run = async (dir: string) => {
    const script = `set -u
say() { echo "[fv-boot] $*" >&2; }
fail() { echo "FAIL: $1" > ${dir}/../failed-$$; exit 1; }
report() { :; }
${R2_FETCH_SH}`;
    const e = { PATH: process.env.PATH!, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1", FV_R2_STATE: join(dir, "..", "state"), FV_R2_TOOLS: join(dir, "..", "tools"), FV_R2_MIN_FREE_GB: "0", FV_WEIGHTS: dir, FV_WEIGHTS_SOURCE: "r2", FV_POD_ID: "gmi:fv-pod-a-1", FV_ENDPOINT_REPORT_TOKEN: "ingest_tok", FV_ENDPOINT_REPORT_URL: `${base}/ingest/v1/endpoint`, ...providerEnvR2(base) };
    return promisify(execFile)("bash", ["-c", script], { env: e, timeout: 60000 });
  };
  const providerEnvR2 = (b: string) => ({ FV_WEIGHTS_PLAN_URL: `${b}/ingest/v1/weights-plan`, FV_R2_FETCH_URL: `${b}/ingest/v1/r2-fetch.py`, FV_R2_FETCH_SHA256: R2_FETCH_SHA256 });
  it("fetches the tree with ranged GETs on presigned URLs, verified, into place; a second boot keeps it", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvr2-"));
    const dir = join(root, "w");
    await run(dir);
    for (const [p, b] of Object.entries(files)) expect(readFileSync(join(dir, "demo", p)).equals(b)).toBe(true);
    expect(readFileSync(join(dir, "demo/.fv-revision"), "utf8")).toBe("rev1");
    expect(existsSync(join(dir, "demo/.complete"))).toBe(true);
    expect(r2.bad()).toBe(0);
    const ranged = r2.gets.filter((g) => g.key.includes("/weights/") && g.range);
    expect(ranged.length).toBe(t.m.files.reduce((s, f) => s + f.parts.length, 0));
    expect(reports.some((r) => r.phase === "weights" && /^r2 weights ready/.test(r.detail))).toBe(true);
    const n = r2.gets.length;
    await run(dir);
    expect(r2.gets.length).toBe(n);
  }, 90000);
  it("a fetcher whose bytes are not the pinned ones is refused before it runs", async () => {
    r2.routes["/ingest/v1/r2-fetch.py"] = (_req, res) => res.writeHead(200).end(R2_FETCH_PY + "\n# tampered\n");
    const root = mkdtempSync(join(tmpdir(), "fvr2-"));
    await expect(run(join(root, "w"))).rejects.toThrow();
    expect(readFileSync(join(root, (await import("node:fs")).readdirSync(root).find((f) => f.startsWith("failed-"))!), "utf8")).toMatch(/r2-fetch.py sha256/);
    expect(existsSync(join(root, "w", "demo"))).toBe(false);
    r2.routes["/ingest/v1/r2-fetch.py"] = (_req, res) => res.writeHead(200).end(R2_FETCH_PY);
    writeFileSync(join(root, "done"), "");
  }, 30000);
});
