// The prompt upsampler's local sidecar (src/upsampler.ts, deploy/runpod/fv-upsampler.sh, docs/serve/upsampler-local.md):
// the switch in the specs (default off, GPU only), the pods' env and boots (Runpod cluster pods, GMI / Brev, serverless),
// the plan request ?set=upsampler (any weights source, only for a pool / endpoint that turned it on), and the launcher
// itself run in bash: a stub llama-server, the real r2-fetch.py against a fake R2 and fv-control's plan handler.
import { execFile } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { bootFor, workerCreatePayload, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { normalizeSpec } from "../../src/cluster/spec";
import { PROVIDER_BOOT, REPORT_PHASES } from "../../src/providers";
import { R2_FETCH_PY, R2_FETCH_SHA256 } from "../../src/r2-fetch-script";
import { R2_TREES } from "../../src/r2-trees";
import { manifestKey, objectKey } from "../../src/r2weights";
import { validate } from "../../src/schemas";
import { SLS_BOOT, SLS_R2_BOOT, templateCreatePayload, workerEnv } from "../../src/serverless/payloads";
import { normalizeEndpointSpec } from "../../src/serverless/spec";
import { newWeightsToken, slsWeightsIssues } from "../../src/serverless/weights";
import { standaloneSpec } from "../../src/standalone";
import { UPSAMPLER_MODEL, UPSAMPLER_START_SH, UPSAMPLER_TREE, UPSAMPLER_WAIT_SH, upsamplerIssue, upsamplerPodEnv } from "../../src/upsampler";
import { weightsPlanRequest } from "../../src/weights-plan";
import { d1 } from "./d1shim";
import { ACCESS, fakeR2, makeTree, rnd, SECRET } from "./fake-r2";

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
    WEIGHTS_R2_AUX: "0",
    ...over,
  }) as unknown as Env;
const h3 = { id: "h3", variant: "h3-turbo", count: 1, compute: "GPU", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] };
const ctxOf = (env: Env) =>
  ({ spec: { control_plane: "direct", auth: "keys", log_level: "info", image: {} }, state: {}, secrets: { internal_token: "i", url_signing_key: "u", ingest_token: "ingest_tok" }, deadlineMs: 0, runpodApiKey: "rpa_x", env }) as unknown as EnvCtx;

describe("the tree and the switch", () => {
  it("the published tree is the Qwen3-VL GGUF (model + vision projector); the text-only one stays published, unused", () => {
    expect(UPSAMPLER_TREE).toBe("qwen3-vl-4b-instruct-gguf");
    expect(R2_TREES[UPSAMPLER_TREE]).toMatchObject({ rev: "1cd86afb9a95c410a6038ab3b40d8b578c892266", files: 3, source: "hub" });
    expect(R2_TREES["qwen3-4b-instruct-gguf"]).toMatchObject({ rev: "gguf-20261013" });
    expect(upsamplerIssue(mkEnv())).toBeNull();
    expect(upsamplerIssue(mkEnv({ WEIGHTS_R2_ACCESS_KEY_ID: undefined }))).toMatch(/R2 weights store/);
  });
  it("default off; GPU only, in pools, standalone launches and serverless endpoints", () => {
    expect(validate("pool", { ...h3, upsampler_local: true }).ok).toBe(true);
    expect(validate("pool", { id: "c", variant: "cpu", count: 1, compute: "CPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], upsampler_local: true }).ok).toBe(false);
    expect(validate("standalone-launch", { name: "a", preset: "h3-turbo", upsampler_local: true }).ok).toBe(true);
    expect(validate("standalone-launch", { name: "a", variant: "cpu", config: "/etc/fv/runpod-fake.toml", compute: "CPU", upsampler_local: true }).ok).toBe(false);
    expect(normalizeEndpointSpec({ name: "e", preset: "h3-turbo" }).upsampler_local).toBeUndefined();
    expect(normalizeEndpointSpec({ name: "e", preset: "h3-turbo", upsampler_local: true }).upsampler_local).toBe(true);
    expect(() => normalizeEndpointSpec({ name: "c", preset: "cpu", upsampler_local: true })).toThrow(/GPU/);
    // a standalone launch carries it onto its pool; without it the pool has none
    expect(standaloneSpec({ name: "a", preset: "h3-turbo", upsampler_local: true } as any).spec.pools[0]!.upsampler_local).toBe(true);
    expect(standaloneSpec({ name: "a", preset: "h3-turbo" } as any).spec.pools[0]!.upsampler_local).toBeUndefined();
    // FV_UPSAMPLER_LOCAL is fv-control's: a spec's env may not set it
    expect(() => normalizeSpec({ name: "c", image: { channel: "stable" }, regions: ["eu"], control_plane: "direct", auth: "keys", pools: [{ ...h3, upsampler_local: true }], cap_s: 3600, min_balance: 8, balance_floor: 8, min_start: 8, max_gpu_dph: 5, log_shipping: false })).not.toThrow();
  });
});

describe("the pods: env and boot", () => {
  const env = mkEnv();
  it("Runpod cluster pods (volume): the switch, the revision, the plan URL and the ingest token; the boot starts it early and waits before fv-serve", () => {
    const e = workerSystemEnv(ctxOf(env), { ...h3, upsampler_local: true } as any, "img");
    expect(e).toMatchObject({ FV_UPSAMPLER_LOCAL: "1", FV_UPSAMPLER_REV: R2_TREES[UPSAMPLER_TREE]!.rev, FV_UPSAMPLER_MODEL: UPSAMPLER_MODEL, FV_WEIGHTS_PLAN_URL: "https://fvc.test/ingest/v1/weights-plan", FV_R2_FETCH_SHA256: R2_FETCH_SHA256, FV_ENDPOINT_REPORT_TOKEN: "ingest_tok" });
    expect(e.FV_WEIGHTS_SOURCE).toBeUndefined();
    expect(JSON.stringify(e)).not.toContain(SECRET);
    const b = bootFor(e);
    expect(b.indexOf(UPSAMPLER_START_SH)).toBeGreaterThan(b.indexOf("mkdir -p /fvstate"));
    expect(b.indexOf(UPSAMPLER_WAIT_SH)).toBeLessThan(b.indexOf("exec /opt/fastvideo-rs/bin/fv-serve"));
    // the volume is still mounted (the GGUF goes to the container disk, never the volume)
    expect((workerCreatePayload("n", "img", { ...h3, upsampler_local: true } as any, { region: "eu", dc: "EU-RO-1", gpu: "g" }, e) as any).networkVolumeId).toBeTruthy();
    // off: no env, the boot unchanged
    const off = workerSystemEnv(ctxOf(env), h3 as any, "img");
    expect(Object.keys(off).filter((k) => k.startsWith("FV_UPSAMPLER"))).toEqual([]);
    expect(bootFor(off)).not.toContain("fv-upsampler");
  });
  it("r2 pods: the sidecar starts before the weights fetch; the disk grows by its tree", () => {
    const all = mkEnv({ WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ "h3-base": { rev: "r", files: 1, bytes: 100e9, manifest_sha256: "a".repeat(64), source: "t" }, "FastH3-4-step-Preview-v1-LoRA": { rev: "r", files: 1, bytes: 1e9, manifest_sha256: "b".repeat(64), source: "t" } }) });
    const on = workerSystemEnv(ctxOf(all), { ...h3, weights_source: "r2", upsampler_local: true } as any, "img");
    const off = workerSystemEnv(ctxOf(all), { ...h3, weights_source: "r2" } as any, "img");
    expect(Number(on.FV_R2_DISK_GB) - Number(off.FV_R2_DISK_GB)).toBe(Math.ceil((R2_TREES[UPSAMPLER_TREE]!.bytes / 1e9) * 1.15));
    const b = bootFor(on);
    expect(b.indexOf("fv-upsampler start")).toBeGreaterThan(0);
    expect(b.indexOf("fv-upsampler start")).toBeLessThan(b.indexOf("r2-fetch.py"));
  });
  it("GMI / Brev and serverless boots carry it (no-ops unless FV_UPSAMPLER_LOCAL=1); the report phase exists", () => {
    for (const boot of [PROVIDER_BOOT, SLS_BOOT, SLS_R2_BOOT]) {
      expect(boot.indexOf(UPSAMPLER_START_SH)).toBeGreaterThan(0);
      expect(boot.indexOf(UPSAMPLER_WAIT_SH)).toBeLessThan(boot.lastIndexOf("exec "));
      expect(boot.indexOf(UPSAMPLER_START_SH)).toBeLessThan(boot.indexOf(UPSAMPLER_WAIT_SH));
    }
    expect(SLS_R2_BOOT.indexOf("fv-upsampler start")).toBeLessThan(SLS_R2_BOOT.indexOf("r2-fetch.py"));
    expect(REPORT_PHASES).toContain("upsampler");
  });
  it("serverless: a volume endpoint with the sidecar gets the token and the plan URL, not weights_source r2's env", () => {
    const s = normalizeEndpointSpec({ name: "e", preset: "h3-turbo", upsampler_local: true });
    expect(s.weights_source).toBe("volume");
    const w = { FV_WEIGHTS_TOKEN: "fvw_x", ...upsamplerPodEnv(env) };
    const e = workerEnv(s, "img", w);
    expect(e).toMatchObject({ FV_UPSAMPLER_LOCAL: "1", FV_WEIGHTS_TOKEN: "fvw_x", FV_WEIGHTS_PLAN_URL: "https://fvc.test/ingest/v1/weights-plan" });
    expect(e.FV_WEIGHTS_SOURCE).toBeUndefined();
    expect(workerEnv(normalizeEndpointSpec({ name: "e", preset: "h3-turbo" }), "img", w).FV_UPSAMPLER_LOCAL).toBeUndefined();
    expect(templateCreatePayload(s, "img", "t", w).env.FV_UPSAMPLER_LOCAL).toBe("1");
    expect(slsWeightsIssues(env, s).issues).toEqual([]);
    expect(slsWeightsIssues(mkEnv({ WEIGHTS_R2_ACCESS_KEY_ID: undefined }), s).issues.map((i) => i.path.join("."))).toEqual(["upsampler_local"]);
  });
});

// ---------------------------------------------------------------- the launcher in bash
const py = existsSync("/usr/bin/python3") || existsSync("/usr/local/bin/python3");
const LAUNCHER = join(__dirname, "../../../deploy/runpod/fv-upsampler.sh");
// A stub llama-server: serves /health and /v1/chat/completions; records its argv.
const STUB = `#!/usr/bin/env python3
import sys, http.server, json, os
a = sys.argv[1:]
open(os.environ["STUB_ARGS"], "w").write(json.dumps(a))
if "--crash" in a: sys.exit(3)
port = int(a[a.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
`;
describe.skipIf(!py)("fv-upsampler.sh with a stub llama-server, the real r2-fetch.py, a fake R2 and fv-control's plan handler", () => {
  const r2 = fakeR2();
  let base = "";
  const files = { "Qwen3VL-4B-Instruct-Q8_0.gguf": rnd(1_300_000, 21), "mmproj-Qwen3VL-4B-Instruct-F16.gguf": rnd(300_000, 22), "README.md": Buffer.from("# test\n") };
  const t = makeTree(UPSAMPLER_TREE, "revU", files);
  let env: Env;
  let tokOn = "";
  let tokOff = "";
  let port = 18890;
  const pids: string[] = [];
  const handler = (fn: (req: Request) => Promise<Response>) => (req: any, res: any) => {
    let body = "";
    req.on("data", (d: Buffer) => (body += d));
    req.on("end", async () => {
      try {
        const out = await fn(new Request(`${base}${req.url}`, { method: "POST", headers: { authorization: req.headers.authorization || "" }, body }));
        res.writeHead(out.status, { "content-type": "application/json" }).end(await out.text());
      } catch (e: any) {
        res.writeHead(e.status || 500).end(JSON.stringify({ error: e.message }));
      }
    });
  };
  beforeAll(async () => {
    base = await r2.start();
    env = mkEnv({ WEIGHTS_R2_ENDPOINT: base, PUBLIC_URL: base, WEIGHTS_R2_EXTRA_TREES: JSON.stringify({ [UPSAMPLER_TREE]: t.pin }) });
    r2.objects.set(`fv-weights/${manifestKey(UPSAMPLER_TREE, "revU")}`, t.raw);
    for (const [p, b] of Object.entries(files)) r2.objects.set(`fv-weights/${objectKey(UPSAMPLER_TREE, "revU", p)}`, b);
    const now = Date.now();
    const ins = async (id: string, spec: unknown) => {
      const tk = await newWeightsToken();
      await env.DB.prepare("INSERT INTO serverless_endpoints (id, name, endpoint_id, mode, spec, status, created_at, created_by, updated_at, weights_token_hash) VALUES (?, ?, ?, 'queue', ?, 'active', ?, 'owner', ?, ?)")
        .bind(id, id, `rp${id}`, JSON.stringify(spec), now, now, tk.hash)
        .run();
      return tk.token;
    };
    tokOn = await ins("uon", normalizeEndpointSpec({ name: "uon", preset: "h3-turbo", upsampler_local: true }));
    tokOff = await ins("uoff", normalizeEndpointSpec({ name: "uoff", preset: "h3-turbo" }));
    r2.routes["/ingest/v1/weights-plan"] = handler((req) => weightsPlanRequest(env, req));
    r2.routes["/ingest/v1/r2-fetch.py"] = (_req, res) => res.writeHead(200).end(R2_FETCH_PY);
  });
  afterAll(async () => {
    for (const p of pids) {
      try {
        process.kill(Number(p));
      } catch {}
    }
    await r2.stop();
  });
  const run = async (root: string, over: Record<string, string> = {}) => {
    mkdirSync(root, { recursive: true });
    const stub = join(root, "llama-server");
    writeFileSync(stub, STUB);
    chmodSync(stub, 0o755);
    port += 1;
    const e = { PATH: process.env.PATH!, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1", ...upsamplerPodEnv(env), FV_UPSAMPLER_REV: "revU", FV_WEIGHTS_TOKEN: tokOn, FV_STATE_DIR: join(root, "state"), FV_WEIGHTS: join(root, "volume"), FV_LLAMA_SERVER: stub, FV_UPSAMPLER_PORT: String(port), FV_R2_TOOLS: join(root, "tools"), FV_UPSAMPLER_RESTARTS: "0", STUB_ARGS: join(root, "args.json"), FV_UPSAMPLER_LOG: join(root, "launcher.log"), RUNPOD_POD_ID: "wk1", ...over };
    const r = await promisify(execFile)("bash", [LAUNCHER, "run"], { env: e, timeout: 60000 });
    for (const f of ["upsampler.pid", "llama-server.pid"]) if (existsSync(join(root, "state", f))) pids.push(readFileSync(join(root, "state", f), "utf8").trim());
    return r;
  };
  it("a volume pod: fetched from R2 onto the container disk (never the volume), started with --mmproj, exports for fv-serve", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvups-"));
    const r = await run(root);
    expect(r.stdout).toContain(`export FV_UPSAMPLER_BASE_URL='http://127.0.0.1:${port}/v1'`);
    expect(r.stdout).toContain(`export FV_UPSAMPLER_MODEL='${UPSAMPLER_MODEL}'`);
    expect(r.stdout).toContain("export FV_UPSAMPLER_VISION='1'");
    // a local server: no hedge, a long timeout, unless the operator set them (the boot evals these lines)
    const sh = (pre: string) => promisify(execFile)("bash", ["-c", `${pre}\n${r.stdout}\necho "$FV_UPSAMPLER_HEDGE $FV_UPSAMPLER_TIMEOUT_MS"`]).then((x) => x.stdout.trim());
    expect(await sh("")).toBe("0 30000");
    expect(await sh("FV_UPSAMPLER_HEDGE=1; FV_UPSAMPLER_TIMEOUT_MS=5000")).toBe("1 5000");
    expect(r.stderr).toMatch(/\[fv-boot\] \d\d:\d\d:\d\d step upsampler: [\d.]+s \(llama-server 127\.0\.0\.1:\d+, qwen3-vl-4b-instruct; fetch [\d.]+ load [\d.]+ vision on total [\d.]+\)/);
    const d = join(root, "state/upsampler-weights", UPSAMPLER_TREE);
    for (const [p, b] of Object.entries(files)) expect(readFileSync(join(d, p)).equals(b)).toBe(true);
    expect(readFileSync(join(d, ".fv-revision"), "utf8")).toBe("revU");
    expect(existsSync(join(root, "volume"))).toBe(false);
    const args = JSON.parse(readFileSync(join(root, "args.json"), "utf8"));
    expect(args.slice(0, 4)).toEqual(["-m", join(d, "Qwen3VL-4B-Instruct-Q8_0.gguf"), "--mmproj", join(d, "mmproj-Qwen3VL-4B-Instruct-F16.gguf")]);
    for (const f of ["-ngl", "99", "-c", "65536", "-np", "2", "--no-kv-unified", "-fa", "on", "-ctk", "q8_0", "--cache-reuse", "--no-webui"]) expect(args).toContain(f);
    expect(r2.bad()).toBe(0);
  }, 90000);
  it("text only on request (FV_UPSAMPLER_VISION=0); a tree already on the disk is used as it is", async () => {
    const root = mkdtempSync(join(tmpdir(), "fvups-"));
    await run(root);
    const gets = r2.gets.length;
    const root2 = join(root, "again");
    const r = await run(root2, { FV_UPSAMPLER_ROOT: join(root, "state/upsampler-weights"), FV_UPSAMPLER_VISION: "0" });
    expect(r2.gets.length).toBe(gets);
    expect(r.stdout).toContain("export FV_UPSAMPLER_VISION='0'");
    expect(JSON.parse(readFileSync(join(root2, "args.json"), "utf8"))).not.toContain("--mmproj");
  }, 90000);
  it("failures never fail the boot: an endpoint without the switch is refused its plan, a crash at load, no binary", async () => {
    const a = mkdtempSync(join(tmpdir(), "fvups-"));
    const r = await run(a, { FV_WEIGHTS_TOKEN: tokOff });
    expect(r.stdout).toBe("");
    expect(r.stderr).toMatch(/upsampler: FAILED: fetch: .*HTTP 409.*does not run the upsampler sidecar.*upsampling falls back/);
    const b = mkdtempSync(join(tmpdir(), "fvups-"));
    await run(b);
    const c = await run(join(b, "c"), { FV_UPSAMPLER_ROOT: join(b, "state/upsampler-weights"), FV_UPSAMPLER_ARGS: "--crash" });
    expect(c.stdout).toBe("");
    expect(c.stderr).toMatch(/upsampler: FAILED: llama-server exited while loading/);
    const d = await run(mkdtempSync(join(tmpdir(), "fvups-")), { FV_LLAMA_SERVER: "/nonexistent/llama-server" });
    expect(d.stderr).toMatch(/upsampler: FAILED: \/nonexistent\/llama-server is not in this image/);
  }, 90000);
});
