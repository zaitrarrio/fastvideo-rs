// The R2 weights store (src/r2weights.ts, weights-plan.ts; docs/ops/r2-weights.md): SigV4 presigning against AWS's
// published example, manifests pinned by SHA-256, the plan, weights_source r2 as the GMI / Brev default once every
// tree is published, the launch checks (GMI / Brev and Runpod), the pod env and boot; and the boot's weights step
// (R2_FETCH_SH) run in bash with the real scripts/gpu/r2-fetch.py against a fake R2 that checks every presigned URL.
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { bootFor, workerCreatePayload, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { presetPool } from "../../src/cluster/spec";
import { launchTrees, poolTrees, PROVIDER_BOOT, providerEnv, providerIssues, weightsPlan, weightsSource } from "../../src/providers";
import { R2_FETCH_PY, R2_FETCH_SHA256 } from "../../src/r2-fetch-script";
import { R2_TREES } from "../../src/r2-trees";
import { encodeKey, manifestKey, objectKey, presignGet, presignPath, R2_FETCH_SH, r2Cfg, r2DiskGb, r2Manifest, r2Plan, r2Trees } from "../../src/r2weights";
import { validate } from "../../src/schemas";
import { d1 } from "./d1shim";
import { ACCESS, fakeR2, makeTree, rnd, SECRET, sha } from "./fake-r2";

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
    WEIGHTS_R2_AUX: "0", // the fake store has only the trees a test publishes (aux from R2: preset-trees.test.ts)
    ...over,
  }) as unknown as Env;

afterEach(() => vi.unstubAllGlobals());

describe("the repo's record of the R2 weights store (scripts/gpu/weights-r2.tsv)", () => {
  it("holds the benchmark round's trees at their pinned Hub revisions, and every volume-only tree", () => {
    expect(R2_TREES["h3-base"]?.rev).toBe("42ed227ee7df40d41602854ae760620d6eb651fe");
    expect(R2_TREES["fastwan22-ti2v-5b"]?.rev).toBe("3e187042a324f6f5fb68fd22110a78725253de8f");
    expect(R2_TREES["wan22-ti2v-5b"]?.rev).toBe("b8fff7315c768468a5333511427288870b2e9635");
    for (const d of ["h3-base/text_encoder_fp8", "ltx25/text_encoder_fp8", "mmaudio-44k-v2", "h3-ref2va", "longlive-1.3b-safetensors"]) expect(R2_TREES[d]?.rev).toBe("eu-20261011");
    for (const t of Object.values(R2_TREES)) expect(t.manifest_sha256).toMatch(/^[0-9a-f]{64}$/);
  });
  it("a GMI / Brev pool whose trees are all recorded defaults to r2 (h3-turbo: h3-base)", () => {
    const pool = { ...presetPool("h3-turbo")!, provider: "gmi", provider_gpu: "container.h200.x1", volume: false } as any;
    expect(weightsSource(pool, mkEnv())).toBe("r2");
  });
});

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
  // These cases need h3-base unpublished: the repo's record (R2_TREES) holds it since the 2026-10-12 copy.
  let saved: any;
  beforeAll(() => {
    saved = R2_TREES["h3-base"];
    delete R2_TREES["h3-base"];
  });
  afterAll(() => {
    if (saved) R2_TREES["h3-base"] = saved;
  });
  const pin = (trees: string[]) => JSON.stringify(Object.fromEntries(trees.map((t) => [t, { rev: `rev-${t}`, files: 1, bytes: 100e9, manifest_sha256: "a".repeat(64), source: "test" }])));
  const gmiPool = (over: any = {}) => ({ ...h3, provider: "gmi", provider_gpu: "container.h200.x1", volume: false, ...over }) as any;
  const trees = poolTrees(gmiPool()).filter((t) => t !== "auxiliary");
  it("GMI / Brev: r2 once every tree of the pool is published, else none; a source given stays", () => {
    // h3-turbo loads h3-base and the FastH3 LoRA beside it (recipe-trees.ts; docs/ops/r2-bench-2026-10-12.md finding 1)
    expect(trees).toEqual(["FastH3-4-step-Preview-v1-LoRA", "h3-base"]);
    expect(weightsSource(gmiPool(), mkEnv())).toBe("none");
    expect(weightsSource(gmiPool(), mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees.filter((t) => t !== "h3-base")) }))).toBe("none");
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    expect(weightsSource(gmiPool(), all)).toBe("r2");
    expect(weightsSource(gmiPool({ weights_source: "hub" }), all)).toBe("hub");
    expect(weightsSource({ ...h3 } as any, all)).toBe("volume");
    const plan = weightsPlan(gmiPool(), all);
    expect([plan.source, plan.unsupported]).toEqual(["r2", []]);
    // Only the aux rows travel in the env: the trees come from the plan endpoint.
    expect(plan.tsv.split("\n").filter(Boolean).every((l) => l.startsWith("aux\t"))).toBe(true);
    // r2 also fetches the preset's pre-quantized trees (published), never the Hub
    expect(plan.trees).toEqual([...trees, "h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"]);
    expect(launchTrees(plan, all)).toEqual(Object.fromEntries(plan.trees.map((t) => [t, r2Trees(all)[t]!.rev])));
  });
  it("needs no download approval; refuses what is not published, or no store configured", () => {
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees) });
    expect(providerIssues(all, { pools: [gmiPool({ weights_source: "r2" })] })).toEqual([]);
    const some = providerIssues(mkEnv({ WEIGHTS_R2_EXTRA_TREES: pin(trees.filter((t) => t !== "h3-base")) }), { pools: [gmiPool({ weights_source: "r2" })] });
    expect(some.map((i) => i.message)).toEqual([expect.stringMatching(/not in the R2 weights store yet: h3-base/)]);
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
    expect(PROVIDER_BOOT).toContain(`python3 "$R2F" "$@"`);
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
    // the trees (2 x 100 GB pinned here) plus the preset's pre-quantized trees (text_encoder_fp8, the MXFP8 DiT)
    const planned = weightsPlan(pool, all).trees;
    expect(planned).toEqual([...trees, "h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"]);
    expect(gb).toBe(Math.min(500, r2DiskGb(all, planned)));
    const p = workerCreatePayload("fv-ctl-x", "img", pool, { region: "eu", dc: "EU-RO-1", gpu: "g" }, e) as any;
    expect(p.networkVolumeId).toBeUndefined();
    // EU-RO-1 (the volume's data centre, hosts known): no CUDA filter
    expect(p.allowedCudaVersions).toBeUndefined();
    // r2 anywhere else (docs/ops/r2-bench-2026-10-12.md findings 2 and 5): the data centres given, CUDA 13 hosts only
    const us = { ...pool, data_centers: ["US-TX-3", "US-KS-2"] };
    expect(validate("pool", us).ok).toBe(true);
    expect(validate("pool", { ...h3, data_centers: ["US-TX-3"] }).ok, "data_centers needs r2 (the volume pins its DC)").toBe(false);
    const pls = workerPlacements({ regions: ["eu"] } as any, us);
    expect(pls.map((x) => x.dc)).toEqual(["US-TX-3", "US-KS-2"]);
    expect(pls.every((x) => x.region === undefined)).toBe(true);
    const pu = workerCreatePayload("fv-ctl-x", "img", us, pls[0]!, e) as any;
    expect([pu.dataCenterIds, pu.allowedCudaVersions, pu.networkVolumeId]).toEqual([["US-TX-3"], ["13.0"], undefined]);
    expect((workerCreatePayload("fv-ctl-x", "img", { ...us, allowed_cuda: [] }, pls[0]!, e) as any).allowedCudaVersions, "allowed_cuda [] lifts the filter").toBeUndefined();
    expect(p.containerDiskInGb).toBe(gb);
    expect(p.dockerStartCmd[0]).toContain('export FV_POD_ID="${FV_POD_ID:-${RUNPOD_POD_ID:-}}"');
    expect(p.dockerStartCmd[0].indexOf("r2-fetch.py")).toBeLessThan(p.dockerStartCmd[0].indexOf("exec /opt/fastvideo-rs/bin/fv-serve"));
    // The volume path is unchanged without r2.
    const v = workerSystemEnv({ ...ctx, env: all } as any, { ...h3 } as any, "img");
    expect(v.FV_WEIGHTS_SOURCE).toBeUndefined();
    expect((workerCreatePayload("fv-ctl-x", "img", { ...h3 } as any, { region: "eu", dc: "EU-RO-1", gpu: "g" }, v) as any).networkVolumeId).toBeTruthy();
    expect(bootFor(v)).not.toContain("r2-fetch.py");
    // A Runpod pool whose trees are not on the EU volume (H3 + LTX only since 2026-10-13) defaults to r2: the
    // fetch boot, no volume, the region's data centre; an explicit volume stays the volume.
    const wan = { ...presetPool("wan")! };
    expect(wan.weights_source).toBeUndefined();
    expect(weightsSource(wan, all)).toBe("r2");
    const we = workerSystemEnv(ctx, wan, "img");
    expect(we.FV_WEIGHTS_SOURCE).toBe("r2");
    const wp = workerCreatePayload("fv-ctl-x", "img", wan, workerPlacements({ regions: ["eu"] } as any, wan)[0]!, we) as any;
    expect([wp.networkVolumeId, wp.dataCenterIds]).toEqual([undefined, ["EU-RO-1"]]);
    expect(providerIssues(all, { pools: [wan] })).toEqual([]);
    expect(weightsSource({ ...wan, weights_source: "volume" }, all)).toBe("volume");
    expect(weightsSource({ ...presetPool("ltx")! }, all)).toBe("volume");
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
    // one ranged GET per two manifest parts (r2-fetch.py DEFAULT_GET_PARTS), each part checked on its own
    expect(ranged.length).toBe(t.m.files.reduce((s, f) => s + Math.ceil(f.parts.length / 2), 0));
    expect(reports.some((r) => r.phase === "weights" && /^r2 weights ready/.test(r.detail))).toBe(true);
    const n = r2.gets.length;
    await run(dir);
    expect(r2.gets.length).toBe(n);
  }, 90000);
  it("streaming (a fv-serve with the marker): the boot goes on once the unguarded files are in place, fv-serve loads while the rest arrives, the fetch and the overlap are timed", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvr2-"));
    const dir = join(root, "w");
    const serve = join(root, "fv-serve");
    writeFileSync(serve, "\x7fELF... fv-streaming-fetch/1 ...");
    r2.hooks.delayMs = (k) => (k.includes("/transformer/") ? 1500 : 0);
    try {
      const script = `set -u
say() { echo "[fv-boot] $*" >&2; }
fail() { echo "FAIL: $1" > ${root}/failed-$$; exit 1; }
report() { :; }
FV_SERVE_BIN=${serve}
${R2_FETCH_SH}echo "SERVE stream=\${FASTVIDEO_STREAMING_FETCH:-}"; [ -f "${dir}/demo/transformer/model.safetensors.fvparts" ] && echo MAP-PRESENT
date +%s.%N > "${join(root, "state")}/serve-start"
wait; echo FETCH-DONE`;
      const e = { PATH: process.env.PATH!, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1", FV_R2_STATE: join(root, "state"), FV_R2_TOOLS: join(root, "tools"), FV_R2_MIN_FREE_GB: "0", FV_WEIGHTS: dir, FV_WEIGHTS_SOURCE: "r2", FV_POD_ID: "gmi:fv-pod-a-1", FV_ENDPOINT_REPORT_TOKEN: "ingest_tok", FV_ENDPOINT_REPORT_URL: `${base}/ingest/v1/endpoint`, ...providerEnvR2(base) };
      const r = await promisify(execFile)("bash", ["-c", script], { env: e, timeout: 60000 });
      expect(r.stdout).toContain("SERVE stream=1");
      expect(r.stdout).toContain("MAP-PRESENT");
      expect(r.stdout).toContain("FETCH-DONE");
      expect(r.stderr).toMatch(/step stream_ready: [\d.]+s/);
      expect(r.stderr).toMatch(/step fetch: [\d.]+s/);
      expect(r.stderr).toMatch(/step load_overlap: [\d.]+s/);
      for (const [p, b] of Object.entries(files)) {
        expect(readFileSync(join(dir, "demo", p)).equals(b)).toBe(true);
        expect(existsSync(join(dir, "demo", p + ".fvparts"))).toBe(false);
      }
      expect(existsSync(join(dir, "demo/.complete"))).toBe(true);
    } finally {
      delete r2.hooks.delayMs;
    }
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
