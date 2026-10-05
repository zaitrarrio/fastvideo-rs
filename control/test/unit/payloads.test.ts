import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GATEWAY_BASE_PODS } from "../../src/cluster/gateway-base";
import {
  GATEWAY_BOOT,
  gatewayCreatePayload,
  gatewaySystemEnv,
  gatewayToml,
  GATEWAY_BASE_MINIMAL,
  reactorModel,
  tomlSet,
  imageIdentEnv,
  isDirect,
  isReserved,
  WORKER_BOOT,
  workerCreatePayload,
  workerPlacements,
  workerSystemEnv,
  type EnvCtx,
} from "../../src/cluster/payloads";
import { defaultSpec, normalizeSpec, POOL_PRESETS, TEMPLATES } from "../../src/cluster/spec";
import CATALOG from "../../src/cluster/catalog.json";
import { OUTPUTS } from "../../gen-configs.mjs";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";

const repo = new URL("../../../", import.meta.url).pathname;
const script = readFileSync(join(repo, "scripts/serve/runpod-cluster.sh"), "utf8");

describe("parity with scripts/serve/runpod-cluster.sh", () => {
  it("the gateway boot command is the script's, byte for byte", () => {
    const m = /\nGATEWAY_BOOT='([\s\S]*?)'\n/.exec(script)!;
    expect(GATEWAY_BOOT).toBe(m[1]);
  });
  it("the worker boot command is the script's plus the inline config", () => {
    const m = /\nWORKER_BOOT='([\s\S]*?)'\n/.exec(script)!;
    const ours = WORKER_BOOT.replace(/if \[ -n "\$\{FV_WORKER_TOML_B64:-\}" \]; then\n.*\nelse\n  (cp .*)\nfi\n/, "$1\n");
    expect(ours).toBe(m[1]);
  });
  it("the embedded gateway base config is gateway-pods.toml up to its pools", () => {
    const f = readFileSync(join(repo, "configs/serve/gateway-pods.toml"), "utf8");
    const base = f.slice(f.search(/^\[server\]$/m), f.search(/^\[\[pools\]\]$/m));
    expect(GATEWAY_BASE_PODS).toBe(base);
  });
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
  spec: defaultSpec("t1"),
  state: {
    images: {},
    gateway_url: "https://gw-8000.proxy.runpod.net",
    workers: { "h3-turbo": [{ pod: "w1", dph: 2, created: 1, image: "i", url: "https://w1-8000.proxy.runpod.net" }], wan: [{ pod: "w2", dph: 2, created: 1, image: "i", url: "https://w2-8000.proxy.runpod.net" }] },
    rolling: { wan: [{ pod: "w3", dph: 2, created: 1, image: "j", url: "https://w3-8000.proxy.runpod.net" }] },
  },
  secrets: { internal_token: "it", url_signing_key: "us", admin_recipient: "PUB", ingest_token: "fvi_x" },
  deadlineMs: 1_700_000_000_000,
  runpodApiKey: "rpa_KEY",
  githubPat: "ghp_PAT",
  ingestUrl: "https://ctl/ingest/v1/logs",
  ...over,
});

describe("env", () => {
  it("gateway env: the script's keys, pool URLs (both while rolling), pods, recipient, GitHub token, log shipping", () => {
    const e = gatewaySystemEnv(ctx(), "ghcr.io/x/y@sha256:abc");
    expect(e.FV_POOL_H3_TURBO_URLS).toBe("https://w1-8000.proxy.runpod.net");
    expect(e.FV_POOL_WAN_URLS).toBe("https://w2-8000.proxy.runpod.net,https://w3-8000.proxy.runpod.net");
    expect(e.FV_CLUSTER_PODS).toBe("w1 w2 w3");
    expect(e.FV_CLUSTER_DEADLINE).toBe("1700000000");
    expect(e.FV_ADMIN_TOKEN_RECIPIENT).toBe("PUB");
    expect(e.FV_ADMIN_TOKEN).toBeUndefined();
    expect(e.FV_BACKSTOP_API_KEY).toBe("rpa_KEY");
    expect(e.FV_GITHUB_TOKEN).toBe("ghp_PAT");
    expect(e.FV_IMAGE_DIGEST).toBe("sha256:abc");
    expect(e.FV_RELEASE_CHANNEL).toBe("stable");
    expect(e.FV_LOG_SHIP_URL).toBe("https://ctl/ingest/v1/logs");
    expect(e.FV_CF_API_TOKEN).toBe("{{ RUNPOD_SECRET_fv_cf_api_token }}");
    expect(Buffer.from(e.FV_GATEWAY_TOML_B64!, "base64").toString()).toContain('id = "wan"');
  });
  it("a legacy imported state passes its own FV_ADMIN_TOKEN", () => {
    const e = gatewaySystemEnv(ctx({ secrets: { internal_token: "a", url_signing_key: "b", admin_token: "fvadm_old", legacy_admin_token: true } }), "img");
    expect(e.FV_ADMIN_TOKEN).toBe("fvadm_old");
    expect(e.FV_ADMIN_TOKEN_RECIPIENT).toBeUndefined();
  });
  it("worker env", () => {
    const c = ctx();
    const e = workerSystemEnv(c, c.spec.pools[0]!, "img");
    expect(e.FV_SERVE_ROLE).toBe("worker");
    expect(e.FV_PUBLIC_BASE_URL).toBe("https://gw-8000.proxy.runpod.net");
    expect(e.FV_WORKER_CONFIG).toBe("/etc/fv/runpod.toml");
    expect(e.FV_BACKSTOP_API_KEY).toBeUndefined();
    const tiny = defaultSpec("t", "tiny-cpu");
    const e2 = workerSystemEnv({ ...c, spec: tiny }, tiny.pools[0]!, "img");
    expect(Buffer.from(e2.FV_WORKER_TOML_B64!, "base64").toString()).toContain('backend = "fake"');
  });
  it("gateway-less worker env: direct client auth with the controller's admin token and the D1 key store", () => {
    const spec = defaultSpec("t");
    spec.gateway.enabled = false;
    const c = ctx({ spec, state: { images: {}, workers: {} }, secrets: { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_ctl" } });
    expect(isDirect(c.spec, c.state)).toBe(true);
    const e = workerSystemEnv(c, spec.pools[0]!, "img");
    expect(e).toMatchObject({ FV_SERVE_ROLE: "worker", FV_WORKER_DIRECT: "1", FV_AUTH_MODE: "keys", FV_KEY_STORE: "d1", FV_ADMIN_TOKEN: "fvadm_ctl", FV_INTERNAL_TOKEN: "it" });
    expect(e.FV_PUBLIC_BASE_URL).toBe("");
    spec.gateway.auth = "none";
    expect(workerSystemEnv(c, spec.pools[0]!, "img").FV_AUTH_MODE).toBe("none");
    // A gateway started later (gateway/start) takes over: the workers become gateway workers again.
    const withGw = { ...c, state: { images: {}, workers: {}, gateway: { pod: "g", dph: 0, created: 1, image: "i" }, gateway_url: "https://g-8000.proxy.runpod.net" } };
    expect(isDirect(withGw.spec, withGw.state)).toBe(false);
    const g = workerSystemEnv(withGw, spec.pools[0]!, "img");
    expect(g.FV_WORKER_DIRECT).toBeUndefined();
    expect(g.FV_ADMIN_TOKEN).toBeUndefined();
    // A gateway cluster's workers never get the admin token.
    expect(workerSystemEnv(ctx(), ctx().spec.pools[0]!, "img").FV_ADMIN_TOKEN).toBeUndefined();
    expect(isReserved("FV_WORKER_DIRECT")).toBe(true);
  });
  it("reserved keys", () => {
    expect(isReserved("FV_INTERNAL_TOKEN")).toBe(true);
    expect(isReserved("FV_POOL_WAN_URLS")).toBe(true);
    expect(isReserved("RUST_LOG")).toBe(false);
    expect(imageIdentEnv("a:b")).toEqual({ FV_IMAGE_REF: "a:b" });
  });
});

describe("payloads and TOML", () => {
  it("gateway and worker create payloads", () => {
    const g = gatewayCreatePayload("n", "img", "cpu3c", 2, 20, ["EUR-IS-1"], { A: "1" });
    expect(g).toMatchObject({ computeType: "CPU", cpuFlavorIds: ["cpu3c"], ports: ["8000/http"], dataCenterIds: ["EUR-IS-1"], dockerEntrypoint: ["bash", "-c"] });
    const spec = defaultSpec("s");
    const pl = workerPlacements(spec, spec.pools[0]!);
    expect(pl[0]).toEqual({ region: "eu", dc: "EUR-IS-1", gpu: "NVIDIA RTX PRO 6000 Blackwell Server Edition" });
    expect(pl.map((p) => p.dc)).toContain("US-CA-2");
    const w = workerCreatePayload("n", "img", spec.pools[0]!, pl[0]!, {});
    expect(w).toMatchObject({ computeType: "GPU", cloudType: "SECURE", gpuCount: 1, networkVolumeId: "jg48s6o1w0", volumeMountPath: "/workspace", ports: ["8000/http", "70000/tcp"] });
    const tiny = defaultSpec("t", "tiny-cpu");
    const cp = workerCreatePayload("n", "img", tiny.pools[0]!, workerPlacements(tiny, tiny.pools[0]!)[0]!, {});
    expect(cp).toMatchObject({ computeType: "CPU", cpuFlavorIds: ["cpu3c"], ports: ["8000/http"] });
    expect((cp as any).networkVolumeId).toBeUndefined();
  });
  it("gateway TOML: pools with static caps, fake models, minimal base without the reactor", () => {
    const t = gatewayToml(defaultSpec("s"));
    expect(t).toContain('reactor_model = "fasth3"');
    // keys newer than the stable (release 1) gateway image stay out of every base
    expect(t).not.toMatch(/^(inline_inputs_max_bytes|input_passthrough|stage_inputs_for_retry) =/m);
    expect(t.match(/\[\[pools\]\]/g)!.length).toBe(4);
    expect(t).toContain('id = "fastwan22-ti2v-5b"');
    const m = gatewayToml(defaultSpec("t", "tiny-cpu"));
    expect(m).not.toContain("reactor_model");
    expect(m).not.toContain("fal_apps");
    expect(m).not.toContain("inline_inputs_max_bytes");
    expect(m).toContain('fake_models = ["fake-wan"]');
    expect(m).toContain("[autoscale]\nenabled = false");
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
    const t = normalizeSpec({ name: "tiny", template: "tiny-cpu" });
    expect(t.pools[0]!.compute).toBe("CPU");
    expect(normalizeSpec({ name: "a", pools: [{ id: "wan", count: 2 }] }).pools[0]).toMatchObject({ id: "wan", variant: "wan5b", count: 2 });
  });
});

describe("gateway apps and overrides", () => {
  const protos = (t: string) => t.slice(t.indexOf("[protocols]"), t.indexOf("[limits]"));
  const apps = (t: string) => JSON.parse(/^fal_apps = (.*)$/m.exec(t)![1]!) as string[];
  it("the pods base mounts every worker config's fal apps; the catalog lists the same ids", () => {
    const want = new Set<string>();
    for (const f of readdirSync(join(repo, "configs/serve")).filter((n) => /^runpod.*\.toml$/.test(n) && n !== "runpod-fake.toml")) {
      const w = readFileSync(join(repo, "configs/serve", f), "utf8");
      if (/^backend = "cuda"$/m.test(w) === false) continue;
      const m = /^fal_apps = (.*)$/m.exec(w);
      const fal = !/^fal = false$/m.test(w);
      if (!fal) continue;
      // No fal_apps: fv-serve's default (the H3 apps).
      for (const a of m ? (JSON.parse(m[1]!) as string[]) : ["minimax/h3-max", "minimax/h3-turbo", "minimax/h3-draft", "minimax/h3-max-turbo", "minimax/h3"]) want.add(a);
    }
    const base = apps(GATEWAY_BASE_PODS);
    for (const a of want) expect(base, a).toContain(a);
    for (const a of ["fal-ai/ltx-2.3", "fal-ai/ltx-2.3-quality", "fastvideo/fastwan21-1.3b", "minimax/h3-draft", "minimax/h3"]) expect(base).toContain(a);
    expect(CATALOG.fal_apps.map((a) => a.id).sort()).toEqual([...base].sort());
    // gateway.toml (serverless pools) mounts the same apps.
    expect(apps(readFileSync(join(repo, "configs/serve/gateway.toml"), "utf8"))).toEqual(base);
  });
  it("defaults to the base; fal_apps, protocols, reactor_model and aliases override it", () => {
    const s = defaultSpec("s");
    expect(gatewayToml(s)).toContain(GATEWAY_BASE_PODS.slice(GATEWAY_BASE_PODS.indexOf("[protocols]"), GATEWAY_BASE_PODS.indexOf("[limits]")));
    s.gateway.fal_apps = ["minimax/h3-turbo", "fal-ai/ltx-2.3"];
    s.gateway.protocols = { fastwan: true, reactor: false };
    s.gateway.reactor_model = null;
    s.gateway.aliases = { "MiniMax-H3": "fasth3", "my-ltx": "ltx25-distill-sol" };
    const t = gatewayToml(normalizeSpec(s));
    expect(apps(t)).toEqual(["minimax/h3-turbo", "fal-ai/ltx-2.3"]);
    expect(protos(t)).toMatch(/^fastwan = true$/m);
    expect(protos(t)).toMatch(/^reactor = false$/m);
    expect(protos(t)).toMatch(/^minimax = true$/m);
    expect(t).not.toContain("reactor_model");
    const al = t.slice(t.indexOf("[aliases]"), t.indexOf("[protocols]"));
    expect(al).toBe('[aliases]\n"MiniMax-H3" = "fasth3"\n"my-ltx" = "ltx25-distill-sol"\n\n');
    // The minimal base (older gateway images) has no fal apps until a spec names them.
    const m = defaultSpec("m", "tiny-cpu");
    expect(protos(gatewayToml(m))).not.toMatch(/fal apps|fal_apps/);
    m.gateway.fal_apps = ["minimax/h3-max"];
    expect(apps(gatewayToml(m))).toEqual(["minimax/h3-max"]);
    expect(GATEWAY_BASE_MINIMAL).not.toContain("# Every worker config");
  });
  it("the Reactor model follows the pools: fasth3, else a causal model, else the spec's", () => {
    expect(reactorModel(defaultSpec("a"))).toBeUndefined();
    const w = defaultSpec("w", "wan");
    expect(reactorModel(w)).toBe("sfwan21-1.3b");
    expect(gatewayToml(w)).toMatch(/^reactor_model = "sfwan21-1\.3b"$/m);
    expect(reactorModel(defaultSpec("l", "longlive"))).toBe("longlive-1.3b");
    w.gateway.reactor_model = "fasth3";
    expect(gatewayToml(w)).toMatch(/^reactor_model = "fasth3"$/m);
    expect(reactorModel(defaultSpec("t", "tiny-cpu"))).toBeUndefined();
  });
  it("tomlSet edits, adds and removes keys inside their section only", () => {
    const t = "[a]\nx = 1\n\n[b]\nx = 2\n";
    expect(tomlSet(t, "b", "x", "3")).toBe("[a]\nx = 1\n\n[b]\nx = 3\n");
    expect(tomlSet(t, "a", "y", "4")).toBe("[a]\nx = 1\ny = 4\n\n[b]\nx = 2\n");
    expect(tomlSet(t, "a", "x", null)).toBe("[a]\n\n[b]\nx = 2\n");
    expect(tomlSet(t, "c", "z", "5")).toBe("[a]\nx = 1\n\n[b]\nx = 2\n\n[c]\nz = 5\n");
  });
  it("the schema takes the overrides and refuses unknown protocols and bad app ids", () => {
    const s = defaultSpec("v");
    expect(() => normalizeSpec({ ...s, gateway: { ...s.gateway, protocols: { teleport: true } } })).toThrow(/protocols/);
    expect(() => normalizeSpec({ ...s, gateway: { ...s.gateway, fal_apps: ["no-slash"] } })).toThrow(/fal_apps/);
    expect(normalizeSpec({ ...s, gateway: { ...s.gateway, fal_apps: [], reactor_model: null, aliases: {} } }).gateway).toMatchObject({ fal_apps: [], reactor_model: null, aliases: {} });
  });
  // The gateway configs every template makes, for crates/fastvideo-serve/tests/gateway_bases.rs
  // (which starts a gateway on each): FV_UPDATE_GOLDEN=1 rewrites them.
  it("fixtures: the gateway TOML of every template", () => {
    const dir = join(repo, "control/test/fixtures");
    const over = defaultSpec("over", "wan");
    over.gateway.fal_apps = ["minimax/h3-max", "fal-ai/ltx-2.3", "fal-ai/wan", "nobody/serves-this"];
    over.gateway.protocols = { fastwan: true };
    over.gateway.reactor_model = null;
    over.gateway.aliases = { "MiniMax-H3": "fasth3" };
    const all = defaultSpec("all");
    all.pools = POOL_PRESETS.map((p) => structuredClone(p.pool));
    const specs: Record<string, ReturnType<typeof defaultSpec>> = { ...Object.fromEntries(Object.keys(TEMPLATES).map((k) => [k, defaultSpec("x", k)])), overrides: over, "all-presets": all };
    for (const [k, s] of Object.entries(specs)) {
      const t = gatewayToml(normalizeSpec(s));
      const p = join(dir, `gateway-${k}.toml`);
      if (process.env.FV_UPDATE_GOLDEN === "1") {
        mkdirSync(dir, { recursive: true });
        writeFileSync(p, t);
      }
      expect(readFileSync(p, "utf8"), `${p} (FV_UPDATE_GOLDEN=1 rewrites it)`).toBe(t);
    }
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
  it("refuses recipes the fv-serve catalog does not have (the gateway would not start)", () => {
    expect(() => normalizeSpec({ name: "p", pools: [{ id: "plug", variant: "h3-turbo", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3-plug", family: "h3", recipe: "h3-plug-4step" }] }] })).toThrow(/not in the fv-serve catalog/);
  });
});
