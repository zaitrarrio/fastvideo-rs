import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  imageIdentEnv,
  isDirect,
  isReserved,
  WORKER_BOOT,
  workerCreatePayload,
  workerPlacements,
  workerSystemEnv,
  type EnvCtx,
} from "../../src/cluster/payloads";
import { assertRegionsAvailable, AVAILABLE_REGIONS, defaultSpec, migrateSpec, normalizeSpec, POOL_PRESETS, REGIONS, TEMPLATES, type RegionId } from "../../src/cluster/spec";
import CATALOG from "../../src/cluster/catalog.json";
import { OUTPUTS } from "../../gen-configs.mjs";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";

const repo = new URL("../../../", import.meta.url).pathname;

describe("boot and generated configs", () => {
  it("the generated files are current (node gen-configs.mjs)", () => {
    for (const [rel, gen] of Object.entries(OUTPUTS)) expect(readFileSync(join(repo, "control", rel), "utf8"), rel).toBe((gen as () => string)());
  });
  it("the worker boot writes an inline config and turns registration off", () => {
    const dir = mkdtempSync(join(tmpdir(), "fvw-"));
    const boot = WORKER_BOOT.replaceAll("/fvstate", `${dir}/state`).replaceAll("/fv-worker.toml", `${dir}/w.toml`).replace(/exec .*$/, "cat " + `${dir}/w.toml`);
    const out = execFileSync("bash", ["-c", boot], { env: { PATH: process.env.PATH!, RUNPOD_POD_ID: "abc", FV_WORKER_TOML_B64: Buffer.from("[server]\nbind = \"x\"\n").toString("base64") } }).toString();
    expect(out).toContain('bind = "x"');
    expect(out).toContain("[gateway]\nregister = false");
    writeFileSync(join(dir, "img.toml"), "[gateway]\npool = \"a\"\n");
    const out2 = execFileSync("bash", ["-c", boot], { env: { PATH: process.env.PATH!, RUNPOD_POD_ID: "abc", FV_WORKER_CONFIG: join(dir, "img.toml") } }).toString();
    expect(out2).toBe('[gateway]\nregister = false\npool = "a"\n');
  });
});

const ctx = (over: Partial<EnvCtx> = {}): EnvCtx => ({
  spec: normalizeSpec({ name: "t1", control_plane: "direct" }),
  state: { images: {}, workers: {} },
  secrets: { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_ctl", ingest_token: "fvi_x" },
  deadlineMs: 1_700_000_000_000,
  runpodApiKey: "rpa_KEY",
  ingestUrl: "https://ctl/ingest/v1/logs",
  ...over,
});

describe("env", () => {
  it("direct worker env: its own client auth, the controller's admin token and the D1 key store", () => {
    const c = ctx();
    expect(isDirect(c.spec, c.state)).toBe(true);
    const e = workerSystemEnv(c, c.spec.pools[0]!, "img@sha256:abc");
    expect(e).toMatchObject({ FV_SERVE_ROLE: "worker", FV_WORKER_DIRECT: "1", FV_AUTH_MODE: "keys", FV_KEY_STORE: "d1", FV_ADMIN_TOKEN: "fvadm_ctl", FV_INTERNAL_TOKEN: "it" });
    expect(e.FV_WORKER_CONFIG).toBe("/etc/fv/runpod.toml");
    expect(e.FV_PUBLIC_BASE_URL).toBe("");
    expect(e.FV_IMAGE_DIGEST).toBe("sha256:abc");
    expect(e.FV_LOG_SHIP_URL).toBe("https://ctl/ingest/v1/logs");
    expect(e.FV_CF_API_TOKEN).toBe("{{ RUNPOD_SECRET_fv_cf_api_token }}");
    expect(e.FV_DISPATCH_FRONT).toBeUndefined();
    const none = normalizeSpec({ name: "t2", control_plane: "direct", auth: "none" });
    expect(workerSystemEnv({ ...c, spec: none }, none.pools[0]!, "img").FV_AUTH_MODE).toBe("none");
    const tiny = normalizeSpec({ name: "t3", template: "tiny-cpu", control_plane: "direct" });
    const e2 = workerSystemEnv({ ...c, spec: tiny }, tiny.pools[0]!, "img");
    expect(Buffer.from(e2.FV_WORKER_TOML_B64!, "base64").toString()).toContain('backend = "fake"');
    expect(isReserved("FV_WORKER_DIRECT")).toBe(true);
  });
  it("reserved keys", () => {
    expect(isReserved("FV_INTERNAL_TOKEN")).toBe(true);
    expect(isReserved("FV_DISPATCH_FRONT")).toBe(true);
    expect(isReserved("RUST_LOG")).toBe(false);
    expect(imageIdentEnv("a:b")).toEqual({ FV_IMAGE_REF: "a:b" });
  });
});

describe("payloads", () => {
  it("worker create payloads", () => {
    const spec = defaultSpec("s");
    const pl = workerPlacements(spec, spec.pools[0]!);
    expect(pl[0]).toEqual({ region: "eu", dc: "EUR-IS-1", gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition" });
    expect(spec.regions).toEqual(["eu"]);
    expect(pl.map((p) => p.dc)).not.toContain("US-CA-2");
    // An old stored spec that still lists us never places there (its volume is gone).
    const old = { ...spec, regions: ["eu", "us"] as RegionId[] };
    expect(workerPlacements(old, old.pools[0]!).every((p) => p.region === "eu" && p.dc === "EUR-IS-1")).toBe(true);
    expect(workerPlacements({ ...spec, regions: ["us"] }, spec.pools[0]!)).toEqual([]);
    const w = workerCreatePayload("n", "img", spec.pools[0]!, pl[0]!, {});
    expect(w).toMatchObject({ computeType: "GPU", cloudType: "SECURE", gpuCount: 1, networkVolumeId: "jg48s6o1w0", volumeMountPath: "/workspace", ports: ["8000/http", "70000/tcp"] });
    const tiny = defaultSpec("t", "tiny-cpu");
    const cp = workerCreatePayload("n", "img", tiny.pools[0]!, workerPlacements(tiny, tiny.pools[0]!)[0]!, {});
    expect(cp).toMatchObject({ computeType: "CPU", cpuFlavorIds: ["cpu3c"], ports: ["8000/http"] });
    expect((cp as any).networkVolumeId).toBeUndefined();
  });
});

describe("spec", () => {
  it("defaults and validation", () => {
    const s = normalizeSpec({ name: "demo" });
    expect(s.pools.length).toBe(4);
    expect(s.balance_floor).toBe(8);
    expect(() => normalizeSpec({ name: "Bad Name" })).toThrow(/name/);
    expect(() => normalizeSpec({ name: "a", image: { channel: "stable", sha: "abcdef1" } })).toThrow(/exactly one/);
    expect(() => normalizeSpec({ name: "a", balance_floor: 5 })).toThrow(/balance_floor/);
    expect(() => normalizeSpec({ name: "a", regions: ["mars"] })).toThrow(/region/);
    expect(() => normalizeSpec({ name: "a", regions: ["eu", "us"] })).toThrow(/US weights volume deleted 2026-10; EU only, see docs\/ops\/runpod-volumes.md/);
    expect(() => normalizeSpec({ name: "a", regions: ["us"] })).toThrow(/region us is unavailable/);
    expect(() => normalizeSpec({ name: "a", pools: [{ id: "wan", regions: ["us"] }] })).toThrow(/pools\[0\]\.regions: region us is unavailable/);
    expect(normalizeSpec({ name: "a", regions: ["eu"] }).regions).toEqual(["eu"]);
    expect(normalizeSpec({ name: "a" }).regions).toEqual(["eu"]);
    const t = normalizeSpec({ name: "tiny", template: "tiny-cpu" });
    expect(t.pools[0]!.compute).toBe("CPU");
    expect(normalizeSpec({ name: "a", pools: [{ id: "wan", count: 2 }] }).pools[0]).toMatchObject({ id: "wan", variant: "wan5b", count: 2 });
    expect(s.control_plane).toBe("edge");
    expect(s.auth).toBe("keys");
    expect(t.pools[0]!.variant).toBe("cpu");
    expect(() => normalizeSpec({ name: "a", control_plane: "both" })).toThrow(/control_plane/);
  });
  it("specs stored before the gateway was retired: a gateway cluster reads as edge, a gateway-less one as direct", () => {
    const old = { name: "o", gateway: { enabled: true, cpu_flavors: ["cpu3c"], vcpu: 2, container_disk_gb: 20, base: "pods", github_token: true, auth: "none" }, control_plane: "gateway" };
    const n = normalizeSpec(old);
    expect([n.control_plane, n.auth, (n as any).gateway]).toEqual(["edge", "none", undefined]);
    expect(normalizeSpec({ name: "o", gateway: { enabled: false, auth: "keys" } }).control_plane).toBe("direct");
    const stored: any = { ...defaultSpec("o", "tiny-cpu"), gateway: { enabled: false, auth: "none" } };
    delete stored.control_plane;
    delete stored.auth;
    stored.pools[0].variant = "gateway";
    const m = migrateSpec(stored);
    expect([m.control_plane, m.auth, m.pools[0]!.variant, (m as any).gateway]).toEqual(["direct", "none", "cpu", undefined]);
  });
});

// The inline config file of each preset (by its content's [gateway] pool line).
const WORKER_CONFIG_OF: Record<string, string> = Object.fromEntries(
  POOL_PRESETS.filter((p) => p.pool.config_toml).map((p) => [p.id, Object.entries(WORKER_CONFIGS).find(([, v]) => v === p.pool.config_toml)![0]]),
);
describe("templates and pool presets", () => {
  it("every template is a valid spec; the presets fill a bare pool id", () => {
    for (const k of Object.keys(TEMPLATES)) {
      const s = normalizeSpec({ name: "t", template: k });
      expect(s.pools.map((p) => p.id), k).toEqual(k === "tiny-cpu" ? ["fake"] : TEMPLATES[k]!.pools.map((id) => POOL_PRESETS.find((p) => p.id === id)!.pool.id));
    }
    const s = normalizeSpec({ name: "a", pools: [{ id: "ltx-pro", count: 2 }, { id: "longlive" }] });
    expect(s.pools[0]).toMatchObject({ id: "ltx-pro", variant: "ltx", count: 2, models: [{ id: "ltx25-distill-dense", family: "ltx2", recipe: "ltx-pro" }] });
    expect(s.pools[0]!.config_toml).toMatch(/^recipe = "ltx-pro"$/m);
    expect(s.pools[1]).toMatchObject({ id: "longlive", variant: "sfwan", count: 1 });
    // A pool's own config wins over the preset's inline one.
    const own = normalizeSpec({ name: "b", pools: [{ id: "ltx-pro", config: "/etc/fv/runpod-ltx-pro.toml" }] }).pools[0]!;
    expect(own.config).toBe("/etc/fv/runpod-ltx-pro.toml");
    expect(own.config_toml).toBeUndefined();
  });
  it("each preset: its variant is an image CI builds, its config is in that image or inline, its models repeat the worker's", () => {
    const variants = /^FV_VARIANTS="([^"]+)"$/m.exec(readFileSync(join(repo, "scripts/serve/variants.sh"), "utf8"))![1]!.split(" ");
    const docker = readFileSync(join(repo, "docker/gpucheck.Dockerfile"), "utf8");
    for (const pr of POOL_PRESETS) {
      const p = pr.pool;
      expect(variants, pr.id).toContain(p.variant);
      expect(p.id).toMatch(/^[a-z][a-z0-9-]{0,30}$/);
      let toml = p.config_toml;
      if (!toml) {
        const file = p.config!.replace("/etc/fv/", "");
        const stage = new RegExp(`FROM serve-cuda-bin AS serve-${p.variant}\\n(COPY [^\\n]*)`).exec(docker)![1]!;
        expect(stage, `${pr.id}: the ${p.variant} image carries ${file}`).toContain(`configs/serve/${file}`);
        toml = readFileSync(join(repo, "configs/serve", file), "utf8");
      }
      const ids = [...toml.matchAll(/^\[\[models\]\]\nid = "([^"]+)"\nfamily = "([^"]+)"\nrecipe = "([^"]+)"/gm)].map((m) => ({ id: m[1], family: m[2], recipe: m[3] }));
      expect(p.models, pr.id).toEqual(ids);
      for (const m of p.models!) expect(CATALOG.recipes.find((r) => r.id === m.recipe)?.serve, `${pr.id}: ${m.recipe}`).toBe(true);
      expect(pr.description.length).toBeGreaterThan(10);
    }
    expect(POOL_PRESETS.find((p) => p.id === "longlive")!.licence).toMatch(/non-commercial/i);
    expect(POOL_PRESETS.find((p) => p.id === "longlive")!.title).toMatch(/NON-COMMERCIAL/);
  });
  it("scripts/serve/variants.sh lists the same presets (variant and config)", () => {
    const sh = readFileSync(join(repo, "scripts/serve/variants.sh"), "utf8");
    const listed = /^FV_POOL_PRESETS="([^"]+)"$/m.exec(sh)![1]!.split(" ").sort();
    const std = new Set(["h3-turbo", "h3-max", "ltx", "wan"]);
    const ours = POOL_PRESETS.filter((p) => !std.has(p.id)).map((p) => `${p.id}:${p.pool.variant}:${p.pool.config ? p.pool.config.replace("/etc/fv/", "") : WORKER_CONFIG_OF[p.id]}`).sort();
    expect(listed).toEqual(ours);
    for (const x of listed) {
      const out = execFileSync("bash", ["-c", `source "${join(repo, "scripts/serve/variants.sh")}"; fv_preset ${x.split(":")[0]}`]).toString().trim();
      expect(out).toBe(x.split(":").slice(1).join(" "));
    }
  });
  it("refuses recipes the fv-serve catalog does not have", () => {
    expect(() => normalizeSpec({ name: "p", pools: [{ id: "plug", variant: "h3-turbo", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3-plug", family: "h3", recipe: "h3-plug-4step" }] }] })).toThrow(/not in the fv-serve catalog/);
  });
});

describe("regions: EU only (US weights volume deleted 2026-10)", () => {
  it("us is known but unavailable; eu is the only available region", () => {
    expect(AVAILABLE_REGIONS).toEqual(["eu"]);
    expect(REGIONS.us.volume).toBe("");
    expect(REGIONS.us.dc).toBe("US-CA-2");
    expect(REGIONS.eu).toMatchObject({ volume: "jg48s6o1w0", dc: "EUR-IS-1" });
    expect(defaultSpec("x").regions).toEqual(["eu"]);
    expect(defaultSpec("x", "tiny-cpu").regions).toEqual(["eu"]);
  });
  it("a stored spec with us cannot start (409, clear message); an eu spec can", () => {
    const s = { ...defaultSpec("x"), regions: ["eu", "us"] as RegionId[] };
    expect(() => assertRegionsAvailable(s)).toThrow(/US weights volume deleted 2026-10/);
    const p = defaultSpec("x");
    p.pools[0]!.regions = ["us"];
    expect(() => assertRegionsAvailable(p)).toThrow(/region us is unavailable/);
    expect(() => assertRegionsAvailable(defaultSpec("x"))).not.toThrow();
  });
});
