// Every configuration schema (src/schemas.ts): valid and invalid cases for
// each rule, the naming rules, the server paths that enforce them, and the
// UI coverage check: each form field has a schema entry, every schema
// property is offered (or listed with the reason it is not), and no enum is
// ever rendered as free text (docs/control/config-validation.md).
import { describe, expect, it } from "vitest";
import { defaultSpec, normalizeSpec, POOL_PRESETS } from "../../src/cluster/spec";
import { envValueProblem, nameProblem, POOL_PRESET_IDS, RUNPOD_GPU_TYPES } from "../../src/enums";
import { DEFAULT_BUILD_PODS_POLICY } from "../../src/buildpods";
import { DEFAULT_POLICIES } from "../../src/alerts";
import { DEFAULT_SLS_POLICY } from "../../src/serverless/ops";
import { defaultEndpointSpec, normalizeEndpointSpec } from "../../src/serverless/spec";
import { jsonSchemas, validate, type SchemaName } from "../../src/schemas";
import { standaloneSpec } from "../../src/standalone";
import { controlKind, enumValues } from "../../ui/fields";
import { FORM_FIELDS } from "../../ui/forms/common";
import { schemaAt, unwrap, type Schema } from "../../ui/schema";

const S = jsonSchemas() as Record<string, Schema>;
const ok = (n: SchemaName, d: unknown) => {
  const r = validate(n, d);
  if (!r.ok) throw new Error(`${n}: ${JSON.stringify(r.issues)}`);
};
/** The issue paths a document gets (joined with "."). */
const bad = (n: SchemaName, d: unknown): string[] => {
  const r = validate(n, d);
  expect(r.ok, `${n} should refuse ${JSON.stringify(d).slice(0, 200)}`).toBe(false);
  return (r as any).issues.map((i: any) => i.path.join("."));
};

describe("names", () => {
  it("DNS-label names, reserved route names", () => {
    for (const n of ["a", "h3-and-ltx", "x1", "a".repeat(31)]) expect(nameProblem("cluster", n)).toBeNull();
    expect(nameProblem("cluster", "")).toMatch(/required/);
    expect(nameProblem("cluster", "a".repeat(32))).toMatch(/31/);
    for (const n of ["Bad", "1abc", "a_b", "ab-", "a b", "é"]) expect(nameProblem("cluster", n), n).not.toBeNull();
    expect(nameProblem("cluster", "new")).toMatch(/reserved/);
    expect(nameProblem("cluster", "validate")).toMatch(/reserved/);
    expect(nameProblem("endpoint", "policy")).toMatch(/reserved/);
    expect(nameProblem("pool", "gateway")).toMatch(/reserved/);
    expect(nameProblem("endpoint", "new")).toBeNull();
  });
  it("the preset ids agree with POOL_PRESETS", () => {
    expect([...POOL_PRESET_IDS].sort()).toEqual(POOL_PRESETS.map((p) => p.id).sort());
  });
});

describe("cluster-spec", () => {
  const base = () => normalizeSpec({ name: "c1", template: "tiny-cpu" });
  const gpu = () => normalizeSpec({ name: "c2" });
  it("templates are valid", () => {
    for (const t of ["standard", "tiny-cpu", "ltx", "h3", "wan", "longlive"]) ok("cluster-spec", normalizeSpec({ name: "t", template: t }));
  });
  it("name, image, regions, enums", () => {
    expect(bad("cluster-spec", { ...base(), name: "new" })).toContain("name");
    expect(bad("cluster-spec", { ...base(), name: "x-" })).toContain("name");
    expect(bad("cluster-spec", { ...base(), image: { channel: "Stable" } })).toContain("image.channel");
    expect(bad("cluster-spec", { ...base(), image: { sha: "xyz" } })).toContain("image.sha");
    expect(bad("cluster-spec", { ...base(), image: { ref: "not an image" } })).toContain("image.ref");
    expect(bad("cluster-spec", { ...base(), image: {} })).toContain("image");
    expect(bad("cluster-spec", { ...base(), regions: ["eu", "eu"] })).toContain("regions");
    expect(bad("cluster-spec", { ...base(), regions: ["us"] })).toContain("regions.0");
    expect(bad("cluster-spec", { ...base(), log_level: "verbose" })).toContain("log_level");
    expect(bad("cluster-spec", { ...base(), control_plane: "gateway" })).toContain("control_plane");
  });
  it("ranges and money cross-checks", () => {
    expect(bad("cluster-spec", { ...base(), cap_s: 299 })).toContain("cap_s");
    expect(bad("cluster-spec", { ...base(), cap_s: 7 * 86400 + 1 })).toContain("cap_s");
    expect(bad("cluster-spec", { ...base(), cap_s: 600.5 })).toContain("cap_s");
    expect(bad("cluster-spec", { ...base(), min_balance: 7.99 })).toContain("min_balance");
    expect(bad("cluster-spec", { ...base(), max_gpu_dph: 0 })).toContain("max_gpu_dph");
    expect(bad("cluster-spec", { ...base(), min_start: 9, balance_floor: 10 })).toContain("min_start");
    expect(bad("cluster-spec", { ...base(), auto_stop_idle_min: 4 })).toContain("auto_stop_idle_min");
    ok("cluster-spec", { ...base(), auto_stop_idle_min: null });
  });
  it("pools: variant, compute, config, models, ranges", () => {
    const P = (p: any) => ({ ...base(), pools: [{ ...base().pools[0], ...p }] });
    const G = (p: any) => ({ ...gpu(), pools: [{ ...gpu().pools[0], ...p }] });
    expect(bad("cluster-spec", P({ variant: "nope" }))).toContain("pools.0.variant");
    expect(bad("cluster-spec", P({ count: 9 }))).toContain("pools.0.count");
    expect(bad("cluster-spec", P({ vcpu: 3 }))).toContain("pools.0.vcpu");
    expect(bad("cluster-spec", P({ cpu_flavors: ["cpu9z"] }))).toContain("pools.0.cpu_flavors.0");
    expect(bad("cluster-spec", P({ gpu_types: ["NVIDIA H200"] }))).toContain("pools.0.gpu_types");
    expect(bad("cluster-spec", P({ fake_models: ["fake-nope"] }))).toContain("pools.0.fake_models.0");
    expect(bad("cluster-spec", P({ config: "/etc/fv/x.toml" }))).toContain("pools.0.config"); // both config and config_toml
    expect(bad("cluster-spec", P({ config_toml: undefined, config: "etc/x" }))).toContain("pools.0.config");
    expect(bad("cluster-spec", P({ models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] }))).toContain("pools.0.models");
    expect(bad("cluster-spec", P({ job_timeout_s: 100, stale_after_s: 200 }))).toContain("pools.0.stale_after_s");
    expect(bad("cluster-spec", P({ id: "gateway" }))).toContain("pools.0.id");
    expect(bad("cluster-spec", G({ variant: "cpu" }))).toContain("pools.0.variant");
    expect(bad("cluster-spec", G({ cpu_flavors: ["cpu3c"] }))).toContain("pools.0.cpu_flavors");
    expect(bad("cluster-spec", G({ gpu_types: ["NVIDIA RTX 9999"] }))).toContain("pools.0.gpu_types.0");
    expect(bad("cluster-spec", G({ gpu_types: ["NVIDIA H200", "NVIDIA H200"] }))).toContain("pools.0.gpu_types");
    expect(bad("cluster-spec", G({ family: "mystery" }))).toContain("pools.0.family");
    expect(bad("cluster-spec", G({ models: [{ id: "fasth3", family: "ltx2", recipe: "h3-turbo" }] }))).toContain("pools.0.models.0.family");
    expect(bad("cluster-spec", G({ models: [{ id: "fasth3", family: "h3", recipe: "ltx-pro" }] }))).toContain("pools.0.models.0.recipe");
    expect(bad("cluster-spec", G({ models: [{ id: "fasth3", family: "h3", recipe: "h3-plug-4step" }] }))).toContain("pools.0.models.0.recipe");
    expect(bad("cluster-spec", G({ models: [{ id: "nope", family: "h3", recipe: "h3-turbo" }] }))).toContain("pools.0.models.0.id");
    expect(bad("cluster-spec", { ...gpu(), control_plane: "direct", pools: [{ ...gpu().pools[0], family: "h3" }] })).toContain("pools.0.family");
    ok("cluster-spec", G({ gpu_types: ["NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], family: "h3" }));
    const two = gpu();
    expect(bad("cluster-spec", { ...two, pools: [two.pools[0], two.pools[0]] })).toContain("pools.1.id");
  });
  it("normalizeSpec reports every problem with its path", () => {
    try {
      normalizeSpec({ name: "Bad Name", cap_s: 1, pools: [{ id: "x", variant: "nope", config: "/etc/fv/runpod.toml", models: [{ id: "fasth3", family: "h3", recipe: "h3-turbo" }] }] });
      throw new Error("accepted");
    } catch (e: any) {
      const paths = e.extra.issues.map((i: any) => i.path.join("."));
      expect(paths).toEqual(expect.arrayContaining(["name", "cap_s", "pools.0.variant"]));
    }
  });
});

describe("standalone-launch", () => {
  it("valid launches", () => {
    ok("standalone-launch", { name: "s1", preset: "h3-turbo" });
    ok("standalone-launch", { name: "s2", variant: "cpu", compute: "CPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], vcpu: 4, cpu_flavors: ["cpu3c"] });
    ok("standalone-launch", { name: "s3", preset: "ltx", sha: "abcdef1", gpu_types: ["NVIDIA H200"], deadline_min: 30, idle_stop_min: null, env: { FASTVIDEO_ATTN_SAGE: "2", TOK: { value: "x", secret: true } } });
  });
  it("every rule", () => {
    const b = { name: "s", preset: "h3-turbo" };
    expect(bad("standalone-launch", { ...b, name: "New" })).toContain("name");
    expect(bad("standalone-launch", { ...b, preset: "nope" })).toContain("preset");
    expect(bad("standalone-launch", { ...b, variant: "ltx" })).toContain("preset");
    expect(bad("standalone-launch", { name: "s" })).toContain("variant");
    expect(bad("standalone-launch", { name: "s", variant: "ltx" })).toContain("config");
    expect(bad("standalone-launch", { ...b, channel: "stable", sha: "abcdef1" })).toContain("channel");
    expect(bad("standalone-launch", { ...b, compute: "CPU", gpu_types: ["NVIDIA H200"] })).toContain("gpu_types");
    expect(bad("standalone-launch", { ...b, compute: "GPU", vcpu: 4 })).toContain("cpu_flavors");
    expect(bad("standalone-launch", { ...b, vcpu: 3 })).toContain("vcpu");
    expect(bad("standalone-launch", { ...b, region: "us" })).toContain("region");
    expect(bad("standalone-launch", { ...b, dc: "US-CA-2" })).toContain("dc");
    expect(bad("standalone-launch", { ...b, deadline_min: 4 })).toContain("deadline_min");
    expect(bad("standalone-launch", { ...b, deadline_min: 10081 })).toContain("deadline_min");
    expect(bad("standalone-launch", { ...b, idle_stop_min: 1441 })).toContain("idle_stop_min");
    expect(bad("standalone-launch", { ...b, max_gpu_dph: 51 })).toContain("max_gpu_dph");
    expect(bad("standalone-launch", { ...b, env: { FV_ADMIN_TOKEN: "x" } })).toContain("env.FV_ADMIN_TOKEN");
    expect(bad("standalone-launch", { ...b, env: { "1BAD": "x" } })).toContain("env.1BAD");
    expect(bad("standalone-launch", { ...b, env: { FASTVIDEO_DIT_OFFLOAD: "sometimes" } })).toContain("env.FASTVIDEO_DIT_OFFLOAD");
    expect(bad("standalone-launch", { ...b, min_start: 9, balance_floor: 10 })).toContain("min_start");
    expect(bad("standalone-launch", { ...b, bogus: 1 })).toContain("");
  });
  it("the cpu variant defaults to a CPU pod", () => {
    expect(standaloneSpec({ name: "c", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"] }).spec.pools[0]!.compute).toBe("CPU");
  });
});

describe("serverless-endpoint", () => {
  it("defaults are valid; every rule", () => {
    ok("serverless-endpoint", defaultEndpointSpec("x", "cpu"));
    ok("serverless-endpoint", defaultEndpointSpec("x", "h3-turbo"));
    const c = defaultEndpointSpec("x", "cpu");
    const g = defaultEndpointSpec("x", "h3-turbo");
    expect(bad("serverless-endpoint", { ...c, name: "tick" })).toContain("name");
    expect(bad("serverless-endpoint", { ...c, variant: "nope" })).toContain("variant");
    expect(bad("serverless-endpoint", { ...c, idle_timeout_s: 4 })).toContain("idle_timeout_s");
    expect(bad("serverless-endpoint", { ...c, scaler_value: 0.5 })).toContain("scaler_value");
    expect(bad("serverless-endpoint", { ...c, scaler_value: 501 })).toContain("scaler_value");
    expect(bad("serverless-endpoint", { ...c, workers_min: 2, workers_max: 1 })).toContain("workers_min");
    expect(bad("serverless-endpoint", { ...c, vcpu: 3 })).toContain("vcpu");
    expect(bad("serverless-endpoint", { ...c, mode: "lb" })).toContain("mode");
    expect(bad("serverless-endpoint", { ...c, env: { PORT: "1" } })).toContain("env.PORT");
    expect(bad("serverless-endpoint", { ...g, data_centers: ["XX-YY-1"] })).toContain("data_centers.0");
    expect(bad("serverless-endpoint", { ...g, data_centers: ["EU-RO-1"] })).toContain("data_centers");
    expect(bad("serverless-endpoint", { ...g, network_volume: "abcdef12" })).toContain("network_volume");
    expect(bad("serverless-endpoint", { ...g, gpu_types: ["NVIDIA RTX 9999"] })).toContain("gpu_types.0");
    expect(bad("serverless-endpoint", { ...g, cpu_flavors: ["cpu3c"] })).toContain("cpu_flavors");
    expect(bad("serverless-endpoint", { ...g, mode: "lb" })).toContain("scaler_type");
    expect(bad("serverless-endpoint", { ...g, config: "x.toml" })).toContain("config");
    expect(() => normalizeEndpointSpec({ name: "a", scaler_value: 1.5 })).toThrow(/scaler_value/);
  });
});

describe("policies and small forms", () => {
  it("policies", () => {
    ok("policies", DEFAULT_POLICIES);
    expect(bad("policies", { ...DEFAULT_POLICIES, auto_stop_idle_min: 10, idle_min: 30 })).toContain("auto_stop_idle_min");
    expect(bad("policies", { ...DEFAULT_POLICIES, idle_gpu_pct: 101 })).toContain("idle_gpu_pct");
    expect(bad("attribution", [{ prefix: "a", owner: "x" }, { prefix: "a", owner: "y" }])).toContain("1.prefix");
    expect(bad("attribution", [{ prefix: "a b", owner: "x" }])).toContain("0.prefix");
  });
  it("build pods and serverless policies", () => {
    ok("build-pods-policy", DEFAULT_BUILD_PODS_POLICY);
    const b = DEFAULT_BUILD_PODS_POLICY;
    expect(bad("build-pods-policy", { ...b, vcpus: [3] })).toContain("vcpus.0");
    expect(bad("build-pods-policy", { ...b, flavors: ["rm -rf"] })).toContain("flavors.0");
    expect(bad("build-pods-policy", { ...b, max_pods: 9 })).toContain("max_pods");
    expect(bad("build-pods-policy", { ...b, disk_gb: 80, disk_gb_fallback: 100 })).toContain("disk_gb_fallback");
    expect(bad("build-pods-policy", { ...b, volumes: { "XX-1": "abcdef" } })).toContain("volumes");
    expect(bad("build-pods-policy", { ...b, image: "not an image" })).toContain("image");
    expect(bad("build-pods-policy", { ...b, labels: [] })).toContain("labels");
    ok("serverless-policy", DEFAULT_SLS_POLICY);
    expect(bad("serverless-policy", { ...DEFAULT_SLS_POLICY, max_workers: 201 })).toContain("max_workers");
  });
  it("tokens, releases, operations, env, log filters", () => {
    ok("token-create", { name: "ci.bot-1", scope: "ci", ttl_days: 30 });
    expect(bad("token-create", { name: "a b", scope: "read" })).toContain("name");
    expect(bad("token-create", { name: "a", scope: "root" })).toContain("scope");
    expect(bad("token-create", { name: "a", scope: "read", ttl_days: 366 })).toContain("ttl_days");
    ok("release-dispatch", { action: "promote", target: "abcdef1", channel: "stable" });
    ok("release-dispatch", { action: "rollback", channel: "stable", to: "12" });
    expect(bad("release-dispatch", { action: "promote", channel: "stable" })).toContain("target");
    expect(bad("release-dispatch", { action: "promote", target: "a b" })).toContain("target");
    expect(bad("release-dispatch", { action: "rollback", to: "x" })).toContain("to");
    ok("extend", { minutes: 30 });
    expect(bad("extend", { minutes: 0 })).toContain("minutes");
    expect(bad("extend", { minutes: 1.5 })).toContain("minutes");
    expect(bad("scale", { pool: "fake", count: 9 })).toContain("count");
    ok("roll", { target: "stable" });
    ok("roll", { target: "abcdef1", pools: ["fake"] });
    expect(bad("roll", { target: "Not A Target!" })).toContain("target");
    expect(bad("mint-key", { name: "a/b" })).toContain("name");
    expect(bad("serverless-scale", { workers_min: 3, workers_max: 1 })).toContain("workers_min");
    expect(bad("serverless-scale", {})).toContain("workers_max");
    ok("env", { FOO: { value: "x", secret: false }, FASTVIDEO_FP8: { value: "1", secret: false } });
    expect(bad("env", { FV_INTERNAL_TOKEN: { value: "x", secret: false } })).toContain("FV_INTERNAL_TOKEN");
    expect(bad("env", { FASTVIDEO_FP8: { value: "yes", secret: false } })).toContain("FASTVIDEO_FP8.value");
    expect(envValueProblem("FASTVIDEO_TAE_DIR", "rel/path")).toMatch(/absolute/);
    ok("log-query", { from: "2h", to: "2026-10-06 12:00", lv: "info,warn", src: "pod,op", order: "desc" });
    expect(bad("log-query", { from: "yesterday" })).toContain("from");
    expect(bad("log-query", { lv: "info,loud" })).toContain("lv");
    expect(bad("log-query", { src: "kafka" })).toContain("src");
  });
  it("the GPU enum covers the regions' GPU types", () => {
    for (const g of ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3", "NVIDIA H200"]) expect(RUNPOD_GPU_TYPES as readonly string[]).toContain(g);
  });
});

// ---------------------------------------------------------------- UI coverage
/** Every node of a schema with its path (array items as index 0). */
function* walk(s: Schema | undefined, root: Schema, path: (string | number)[] = []): Generator<[(string | number)[], Schema]> {
  const u = unwrap(s, root);
  if (!u) return;
  yield [path, u];
  for (const [k, ps] of Object.entries<Schema>(u.properties || {})) yield* walk(ps, root, [...path, k]);
  if (u.items) yield* walk(u.items, root, [...path, 0]);
  if (u.additionalProperties && typeof u.additionalProperties === "object") yield* walk(u.additionalProperties, root, [...path, "*"]);
}
describe("UI coverage", () => {
  it("no schema enum is rendered as free text", () => {
    let enums = 0;
    for (const [name, root] of Object.entries(S))
      for (const [path, u] of walk(root, root)) {
        const kind = controlKind(u, root);
        if (enumValues(u)) {
          enums++;
          expect(["select", "segmented"], `${name}:${path.join(".")}`).toContain(kind);
        }
        const it = unwrap(u.items, root);
        if (it && enumValues(it)) expect(kind, `${name}:${path.join(".")}[]`).toBe("chips");
        if (u["x-dynamic"] && ["channels", "variants", "regions", "pool_presets", "data_centers", "volumes", "releases", "pools", "clusters"].includes(u["x-dynamic"])) expect(kind, `${name}:${path.join(".")}`).toBe("select");
      }
    expect(enums).toBeGreaterThan(30);
  });
  it("every form field has a schema entry; every schema property is offered or listed", () => {
    for (const [form, m] of Object.entries(FORM_FIELDS)) {
      const root = S[m.schema];
      expect(root, form).toBeTruthy();
      for (const p of m.fields) expect(schemaAt(root!, p), `${form}: ${p.join(".")}`).toBeTruthy();
      const offered = new Set(m.fields.map((p) => String(p[0])));
      for (const k of Object.keys(unwrap(root, root!)?.properties || {})) expect(offered.has(k) || !!m.notOffered?.[k], `${form}: ${k} is neither offered nor listed in notOffered`).toBe(true);
    }
  });
  it("the cluster-spec page offers every spec field (ui/cluster/config.ts)", async () => {
    const { readFileSync } = await import("node:fs");
    const src = readFileSync(new URL("../../ui/cluster/config.ts", import.meta.url), "utf8");
    for (const k of Object.keys(S["cluster-spec"]!.properties)) expect(src.includes(`["${k}"]`) || src.includes(`"${k}"`), `cluster-spec.${k}`).toBe(true);
    for (const k of Object.keys(S.pool!.properties)) expect(src.includes(`P("${k}")`) || src.includes(`"${k}"`) || new RegExp(`\\.${k}\\b`).test(src), `pool.${k}`).toBe(true);
    // No free-text input for an enum field there.
    for (const k of ["family", "vcpu", "fake_models", "log_level", "variant"]) expect(src).not.toMatch(new RegExp(`text\\(P\\("${k}"\\)`));
  });
});

// ---------------------------------------------------------------- the staging failures of 2026-10-07
import { closeMatches } from "../../src/enums";
import { readableProblem, runpodErrorText } from "../../src/runpoderr";
import { placementIssues } from "../../src/serverless/spec";

describe("serverless GPU types (staging rows scale2, scale, testing)", () => {
  // What the owner typed: free-text GPU names Runpod refused at POST /endpoints.
  const OWNER = [
    { name: "scale2", variant: "h3-turbo", gpu_types: ["RTX 6000 PRO", "H100"] },
    { name: "scale", variant: "h3-turbo", gpu_types: ["RTX 6000 PRO", "H100"] },
    { name: "testing", variant: "ltx", gpu_types: ["H100", "RTX PRO 6000"] },
  ];
  it("each is refused before Runpod, at gpu_types, with the id to use", () => {
    for (const spec of OWNER) {
      let err: any;
      try {
        normalizeEndpointSpec(spec);
      } catch (e) {
        err = e;
      }
      expect(err, spec.name).toBeTruthy();
      const paths = err.extra.issues.map((i: any) => i.path.join("."));
      expect(paths).toEqual(expect.arrayContaining(["gpu_types.0", "gpu_types.1"]));
      expect(err.message).toMatch(/did you mean "NVIDIA RTX PRO 6000 Blackwell Server Edition"/);
      expect(err.message).toMatch(/"H100" is not a Runpod GPU type id; did you mean "NVIDIA H100/);
    }
  });
  it("close matches put the server card first", () => {
    expect(closeMatches("RTX 6000 PRO", RUNPOD_GPU_TYPES)[0]).toBe("NVIDIA RTX PRO 6000 Blackwell Server Edition");
    expect(closeMatches("RTX PRO 6000", RUNPOD_GPU_TYPES)[0]).toBe("NVIDIA RTX PRO 6000 Blackwell Server Edition");
    expect(closeMatches("H100", RUNPOD_GPU_TYPES)).toEqual(expect.arrayContaining(["NVIDIA H100 NVL"]));
    expect(closeMatches("rtx 5090", RUNPOD_GPU_TYPES)[0]).toBe("NVIDIA GeForce RTX 5090");
  });
  it("the corrected spec is valid", () => {
    expect(normalizeEndpointSpec({ name: "scale2", variant: "h3-turbo", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] }).gpu_types).toEqual(["NVIDIA RTX PRO 6000 Blackwell Server Edition"]);
  });
  it("GPU types against the volume's data centre: H100 only on the EU volume is refused, a mix warns", async () => {
    const stock = async (pairs: { dc: string; gpu: string }[]) => new Map(pairs.map((p) => [`${p.dc}|${p.gpu}`, p.gpu.includes("RTX PRO 6000") ? { stock: "Low", max_available: 2 } : { stock: null, max_available: 0 }]));
    const h100 = normalizeEndpointSpec({ name: "h", variant: "h3-turbo", gpu_types: ["NVIDIA H100 80GB HBM3"] });
    const r1 = await placementIssues(h100, stock);
    expect(r1.issues[0]!.message).toMatch(/none of NVIDIA H100 80GB HBM3 is offered in EUR-IS-1/);
    const mix = normalizeEndpointSpec({ name: "m", variant: "h3-turbo", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3"] });
    const r2 = await placementIssues(mix, stock);
    expect(r2.issues).toEqual([]);
    expect(r2.warnings.map((w) => w.path.join("."))).toEqual(["gpu_types.1"]);
    // CPU endpoints and Runpod not answering: nothing.
    expect((await placementIssues(defaultEndpointSpec("c", "cpu"), stock)).issues).toEqual([]);
    expect((await placementIssues(h100, async () => Promise.reject(new Error("down")))).issues).toEqual([]);
  });
});

describe("Runpod errors made readable", () => {
  const ALLOWED = RUNPOD_GPU_TYPES.map((g) => `'${g}'`).join(", ");
  it("the enum problem names the field, our value, the close match, then the list (nothing cut)", () => {
    const body = { error: "request body validation failed", problems: [`At /endpoints/properties/gpuTypeIds/items/enum: value must be one of ${ALLOWED}`] };
    const t = runpodErrorText(body, JSON.stringify({ name: "fvc-scale2", gpuTypeIds: ["RTX 6000 PRO", "H100"] }));
    expect(t.startsWith('gpuTypeIds: Runpod refused "RTX 6000 PRO" (did you mean "NVIDIA RTX PRO 6000 Blackwell Server Edition"')).toBe(true);
    expect(t).toContain('"H100" (did you mean "NVIDIA H100');
    expect(t).toContain("Tesla V100-SXM2-16GB");
    expect(t).toContain("(request body validation failed)");
  });
  it("other enums Runpod checks: dataCenterIds, cpuFlavorIds, allowedCudaVersions, scalerType", () => {
    expect(readableProblem("At /endpoints/properties/dataCenterIds/items/enum: value must be one of 'EU-RO-1', 'EUR-IS-1'", { dataCenterIds: ["EU-IS-1"] })).toMatch(/^dataCenterIds: Runpod refused "EU-IS-1" \(did you mean "EUR-IS-1"/);
    expect(readableProblem("At /endpoints/properties/scalerType/enum: value must be one of 'QUEUE_DELAY', 'REQUEST_COUNT'", { scalerType: "QUEUE" })).toMatch(/scalerType: Runpod refused "QUEUE" \(did you mean "QUEUE_DELAY"/);
    expect(readableProblem("something else")).toBe("something else");
    expect(runpodErrorText({ error: "plain" })).toBe("plain");
    // Our own schema refuses the same values first, with the same hint.
    for (const [spec, re] of [
      [{ name: "a", variant: "h3-turbo", data_centers: ["EU-IS-1"] }, /data_centers\.0: "EU-IS-1" is not a Runpod data centre/],
      [{ name: "a", cpu_flavors: ["cpu3x"] }, /cpu_flavors\.0: "cpu3x" is not a Runpod serverless CPU flavor; did you mean/],
      [{ name: "a", variant: "h3-turbo", allowed_cuda: ["13"] }, /allowed_cuda\.0: "13" is not a CUDA version/],
      [{ name: "a", scaler_type: "QUEUE" }, /scaler_type: "QUEUE" is not a Runpod scaler type; did you mean "QUEUE_DELAY"/],
    ] as const)
      expect(() => normalizeEndpointSpec(spec)).toThrow(re);
  });
});
