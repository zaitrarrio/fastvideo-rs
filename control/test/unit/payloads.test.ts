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
import { mustVariant } from "../../src/families";
import { configModelRefs } from "../../src/presets";
import { OUTPUTS } from "../../gen-configs.mjs";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { DEPLOY_FAMILY_PRESETS, DEPLOY_PRESETS } from "../../src/deploy-gen";

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
    // `[cluster]` (the table's name since plan F3b): edited in place, never a second `[gateway]` beside it.
    const out3 = execFileSync("bash", ["-c", boot], { env: { PATH: process.env.PATH!, RUNPOD_POD_ID: "abc", FV_WORKER_TOML_B64: Buffer.from('[cluster]\npool = "a"\n').toString("base64") } }).toString();
    expect(out3).toBe('[cluster]\nregister = false\npool = "a"\n');
    // Neither: the boot stops (`set -u`), never serving a default (the images bake none).
    let failed = false;
    try {
      execFileSync("bash", ["-c", boot], { env: { PATH: process.env.PATH!, RUNPOD_POD_ID: "abc" }, stdio: "pipe" });
    } catch {
      failed = true;
    }
    expect(failed).toBe(true);
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
    // The pool's config rides inline (the one image bakes none): the standard h3-turbo pool's runpod.toml.
    expect(e.FV_WORKER_CONFIG).toBeUndefined();
    expect(Buffer.from(e.FV_WORKER_TOML_B64!, "base64").toString()).toBe(WORKER_CONFIGS["runpod.toml"]);
    // A pool naming a config (a stored spec's /etc/fv/<file>, or the file name): its text, inline, resolved at launch.
    const named = workerSystemEnv(c, { ...c.spec.pools[0]!, config: "/etc/fv/runpod-ltx.toml", config_toml: undefined }, "img");
    expect(Buffer.from(named.FV_WORKER_TOML_B64!, "base64").toString()).toBe(WORKER_CONFIGS["runpod-ltx.toml"]);
    expect(() => workerSystemEnv(c, { ...c.spec.pools[0]!, config: undefined, config_toml: undefined }, "img")).toThrow(/no worker config/);
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
    expect(pl[0]).toEqual({ region: "eu", dc: "EU-RO-1", gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition" });
    expect(spec.regions).toEqual(["eu"]);
    expect(pl.map((p) => p.dc)).not.toContain("US-CA-2");
    // An old stored spec that still lists us never places there (its volume is gone).
    const old = { ...spec, regions: ["eu", "us"] as RegionId[] };
    expect(workerPlacements(old, old.pools[0]!).every((p) => p.region === "eu" && p.dc === "EU-RO-1")).toBe(true);
    expect(workerPlacements({ ...spec, regions: ["us"] }, spec.pools[0]!)).toEqual([]);
    const w = workerCreatePayload("n", "img", spec.pools[0]!, pl[0]!, {});
    expect(w).toMatchObject({ computeType: "GPU", cloudType: "SECURE", gpuCount: 1, networkVolumeId: "bz0ofkfx45", volumeMountPath: "/workspace", ports: ["8000/http", "70000/tcp"] });
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
    expect(() => normalizeSpec({ name: "a", pools: [{ id: "wan", regions: ["us"] }] })).toThrow(/pools\.0\.regions\.0: region us is unavailable/);
    expect(normalizeSpec({ name: "a", regions: ["eu"] }).regions).toEqual(["eu"]);
    expect(normalizeSpec({ name: "a" }).regions).toEqual(["eu"]);
    const t = normalizeSpec({ name: "tiny", template: "tiny-cpu" });
    expect(t.pools[0]!.compute).toBe("CPU");
    expect(normalizeSpec({ name: "a", pools: [{ id: "wan", count: 2 }] }).pools[0]).toMatchObject({ id: "wan", compute: "GPU", count: 2, config: "runpod-wan5b.toml" });
    expect(s.control_plane).toBe("edge");
    expect(s.auth).toBe("keys");
    expect("variant" in t.pools[0]!).toBe(false);
    // Every standard pool sends its config (the one image bakes none).
    for (const p of s.pools) expect(WORKER_CONFIGS[p.config!], p.id).toBeTruthy();
    // A config the images used to carry but fv-control does not know, or none at all: refused before any pod.
    expect(() => normalizeSpec({ name: "a", pools: [{ id: "x", compute: "GPU", count: 1, config: "/etc/fv/nope.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] }] })).toThrow(/pools\.0\.config: not a worker config fv-control carries/);
    expect(() => normalizeSpec({ name: "a", pools: [{ id: "x", compute: "GPU", count: 1, models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] }] })).toThrow(/config or config_toml is required/);
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
    delete stored.pools[0].compute;
    const m = migrateSpec(stored);
    expect([m.control_plane, m.auth, (m.pools[0] as any).variant, m.pools[0]!.compute, (m as any).gateway]).toEqual(["direct", "none", undefined, "CPU", undefined]);
  });
  it("specs stored before the one-image decision (2026-10-25): a pool's image variant reads as its compute, its config is sent", () => {
    // A cluster spec as fv-control stored it with per-variant images: variants, configs in the image.
    const stored: any = {
      ...defaultSpec("old"),
      control_plane: "direct",
      pools: [
        { id: "h3-turbo", variant: "h3-turbo", count: 1, compute: "GPU", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] },
        { id: "wan", variant: "wan5b", count: 1, compute: "GPU", config: "/etc/fv/runpod-wan5b.toml", models: [{ id: "fastwan22-ti2v-5b", family: "wan", recipe: "wan-turbo" }] },
        { id: "fake", variant: "cpu", count: 1, config_toml: "[engine]\nbackend = \"fake\"\n", fake_models: ["fake-wan"], volume: false },
      ],
    };
    const m = migrateSpec(JSON.parse(JSON.stringify(stored)));
    expect(m.pools.map((p) => [(p as any).variant, p.compute])).toEqual([[undefined, "GPU"], [undefined, "GPU"], [undefined, "CPU"]]);
    // It validates as stored (the editor and every save) and as posted (the API, a CLI from before).
    const n = normalizeSpec(JSON.parse(JSON.stringify(stored)));
    expect(n.pools.map((p) => [(p as any).variant, p.compute, p.config])).toEqual([[undefined, "GPU", "/etc/fv/runpod.toml"], [undefined, "GPU", "/etc/fv/runpod-wan5b.toml"], [undefined, "CPU", undefined]]);
    // Its pods get the config the variant image baked, inline, on the one image of their compute.
    const c = ctx({ spec: n });
    expect(Buffer.from(workerSystemEnv(c, n.pools[1]!, "img").FV_WORKER_TOML_B64!, "base64").toString()).toBe(WORKER_CONFIGS["runpod-wan5b.toml"]);
    // A pool that relied on its variant's baked config (no config of its own) gets that config by name.
    const bare = normalizeSpec({ name: "b", pools: [{ id: "x", variant: "ltx", count: 1, models: [{ id: "ltx25-distill-sol", family: "ltx2", recipe: "ltx-turbo" }] }] });
    expect([bare.pools[0]!.compute, bare.pools[0]!.config]).toEqual(["GPU", "runpod-ltx.toml"]);
  });
  it("specs stored before plan A3: the v0.1 Turbo LoRA and LongLive recipes read as their variants and validate", () => {
    const stored: any = normalizeSpec({ name: "o", pools: [{ id: "h3-lightx2v-turbo-v01" }, { id: "longlive" }, { id: "sfwan" }] });
    stored.pools[0].models = [{ id: "h3-lightx2v-turbo-v01", family: "h3", recipe: "lightx2v-turbo-v0.1" }];
    stored.pools[1].models = [{ id: "longlive-1.3b", family: "wan", recipe: "sfwan21-1.3b" }];
    expect(() => normalizeSpec(JSON.parse(JSON.stringify(stored)))).toThrow(/pools\.0\.models\.0\.recipe/);
    const m = migrateSpec(JSON.parse(JSON.stringify(stored)));
    expect(m.pools.map((p) => p.models)).toEqual([
      [{ id: "h3-lightx2v-turbo-v01", family: "h3", recipe: "h3-lightx2v-turbo-v01" }],
      [{ id: "longlive-1.3b", family: "wan", recipe: "longlive-1.3b" }],
      // The SF-Wan model keeps its recipe (only the LongLive id was renamed).
      [{ id: "sfwan21-1.3b", family: "wan", recipe: "sfwan21-1.3b" }],
    ]);
    expect(normalizeSpec(m).pools.map((p) => p.id)).toEqual(["h3-lightx2v-turbo-v01", "longlive", "sfwan"]);
  });
});

// The config file of each preset, from deploy/deploy.toml (the generated tables).
const WORKER_CONFIG_OF: Record<string, string> = Object.fromEntries([...DEPLOY_PRESETS, ...DEPLOY_FAMILY_PRESETS].map((p) => [p.id, p.config]));
describe("templates and pool presets", () => {
  it("every template is a valid spec; the presets fill a bare pool id", () => {
    for (const k of Object.keys(TEMPLATES)) {
      const s = normalizeSpec({ name: "t", template: k });
      expect(s.pools.map((p) => p.id), k).toEqual(k === "tiny-cpu" ? ["fake"] : TEMPLATES[k]!.pools.map((id) => POOL_PRESETS.find((p) => p.id === id)!.pool.id));
    }
    const s = normalizeSpec({ name: "a", pools: [{ id: "ltx-pro", count: 2 }, { id: "longlive" }] });
    expect(s.pools[0]).toMatchObject({ id: "ltx-pro", compute: "GPU", count: 2, models: [{ id: "ltx25-distill-dense", family: "ltx2", recipe: "ltx-pro" }] });
    expect(WORKER_CONFIGS[s.pools[0]!.config!]).toMatch(/^variant = "ltx25-distill-dense"$/m);
    expect(s.pools[1]).toMatchObject({ id: "longlive", compute: "GPU", count: 1 });
    // A pool's own config wins over the preset's.
    const own = normalizeSpec({ name: "b", pools: [{ id: "ltx-pro", config: "/etc/fv/runpod-ltx-pro.toml" }] }).pools[0]!;
    expect(own.config).toBe("/etc/fv/runpod-ltx-pro.toml");
    expect(own.config_toml).toBeUndefined();
  });
  it("each preset: a GPU pool on the CUDA image, its config sent inline (a committed configs/serve file), its models repeat the worker's", () => {
    const images = /^FV_IMAGES="([^"]+)"$/m.exec(readFileSync(join(repo, "scripts/serve/variants.sh"), "utf8"))![1]!.split(" ");
    expect(images).toEqual(["cuda", "cpu"]);
    for (const pr of POOL_PRESETS) {
      const p = pr.pool;
      expect("variant" in p, pr.id).toBe(false);
      expect(p.compute, pr.id).toBe("GPU");
      expect(p.id).toMatch(/^[a-z][a-z0-9-]{0,30}$/);
      expect(p.config_toml, pr.id).toBeUndefined();
      const toml = WORKER_CONFIGS[p.config!]!;
      expect(WORKER_CONFIG_OF[pr.id], pr.id).toBe(p.config);
      expect(toml, pr.id).toBe(readFileSync(join(repo, "configs/serve", WORKER_CONFIG_OF[pr.id]!), "utf8"));
      // The worker config names variants (plan A3); the preset's models are the same ids, families and variants.
      const cfg = configModelRefs(toml);
      expect(p.models!.map((m) => [m.id, m.family]), pr.id).toEqual(cfg.map((m) => [m.id, m.family]));
      for (const [i, m] of p.models!.entries()) expect(mustVariant(m.recipe).id, `${pr.id}: ${m.id}`).toBe(mustVariant(cfg[i]!.recipe).id);
      for (const m of p.models!) expect(CATALOG.recipes.find((r) => r.id === m.recipe)?.serve, `${pr.id}: ${m.recipe}`).toBe(true);
      expect(pr.description.length).toBeGreaterThan(10);
    }
    expect(POOL_PRESETS.find((p) => p.id === "longlive")!.licence).toMatch(/non-commercial/i);
    expect(POOL_PRESETS.find((p) => p.id === "longlive")!.title).toMatch(/NON-COMMERCIAL/);
  });
  it("scripts/serve/variants.sh: no per-variant tables; the env that sends a config (fv_config_env_json)", () => {
    const sh = readFileSync(join(repo, "scripts/serve/variants.sh"), "utf8");
    for (const gone of ["FV_VARIANTS=", "FV_POOL_PRESETS=", "fv_variant_config", "fv_variant_for_config", "fv_preset"]) expect(sh, gone).not.toContain(gone);
    const env = (cfg: string) => JSON.parse(execFileSync("bash", ["-c", `source "${join(repo, "scripts/gpu/lib.sh")}"; source "${join(repo, "scripts/serve/variants.sh")}"; fv_config_env_json "$1"`, "x", cfg]).toString());
    // The one config the images carry: named in the image. Any other: its configs/serve text, inline.
    expect(env("/etc/fv/runpod-fake.toml")).toEqual({ FV_WORKER_CONFIG: "/etc/fv/runpod-fake.toml" });
    expect(Buffer.from(env("/etc/fv/runpod-ltx.toml").FV_WORKER_TOML_B64, "base64").toString()).toBe(readFileSync(join(repo, "configs/serve/runpod-ltx.toml"), "utf8"));
    expect(Buffer.from(env("runpod-family-h3.toml").FV_WORKER_TOML_B64, "base64").toString()).toBe(WORKER_CONFIGS["runpod-family-h3.toml"]);
    expect(() => execFileSync("bash", ["-c", `source "${join(repo, "scripts/gpu/lib.sh")}"; source "${join(repo, "scripts/serve/variants.sh")}"; fv_config_env_json /etc/fv/nope.toml`], { stdio: "pipe" })).toThrow();
  });
  it("refuses recipes the fv-serve catalog does not have", () => {
    expect(() => normalizeSpec({ name: "p", pools: [{ id: "plug", compute: "GPU", config: "runpod.toml", models: [{ id: "fasth3-plug", family: "h3", recipe: "h3-plug-4step" }] }] })).toThrow(/not in the fv-serve catalog/);
  });
});

describe("regions: EU only (US weights volume deleted 2026-10)", () => {
  it("us is known but unavailable; eu is the only available region", () => {
    expect(AVAILABLE_REGIONS).toEqual(["eu"]);
    expect(REGIONS.us.volume).toBe("");
    expect(REGIONS.us.dc).toBe("US-CA-2");
    expect(REGIONS.eu).toMatchObject({ volume: "bz0ofkfx45", dc: "EU-RO-1" });
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

describe("the worker config in the env (one image: every launch sends it)", () => {
  it("every config fv-control sends fits the FV_WORKER_TOML_B64 guard, the same one fv-deploy keeps", async () => {
    const { MAX_WORKER_TOML_B64, b64utf8, workerConfigProblem } = await import("../../src/worker-config");
    // Runpod documents no per-value limit (50 variables per pod; 1 MiB per v2 request): a conservative guard.
    const rust = readFileSync(join(repo, "crates/fastvideo-deploy/src/launch.rs"), "utf8");
    expect(/pub const MAX_WORKER_TOML_B64: usize = 16 \* 1024;/.test(rust)).toBe(true);
    expect(MAX_WORKER_TOML_B64).toBe(16 * 1024);
    const sizes = Object.entries(WORKER_CONFIGS).map(([k, v]) => [k, b64utf8(v).length] as const);
    for (const [k, n] of sizes) expect(n, k).toBeLessThan(MAX_WORKER_TOML_B64 / 2);
    // The largest is the H3 family's (5600 bytes encoded on 2026-10-25).
    expect([...sizes].sort((a, b) => b[1] - a[1])[0]![0]).toBe("runpod-family-h3.toml");
    expect(workerConfigProblem({ config_toml: "#".repeat(MAX_WORKER_TOML_B64) })).toMatch(/over the 16384-byte guard/);
  });
});
