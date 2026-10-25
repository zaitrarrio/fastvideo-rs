// fv-control's launches on fv-deploy's `Launch` (plan C3c, docs/adr/0018-fv-deploy.md §7):
//   1. src/launch-gen.ts is generated from crates/fastvideo-deploy/schemas/*.v1.json, and a schema change that is not
//      regenerated fails `gen-configs.mjs --check`;
//   2. src/launch.ts builds, from a Launch, exactly the Runpod bodies the Rust provider builds: the C3a payload goldens
//      (crates/fastvideo-deploy/fixtures/deploy/*.json, what runpod-pod.sh / runpod-endpoint.sh send) from the same
//      launches crates/fastvideo-deploy/tests/payload_goldens.rs builds them from;
//   3. for the presets fv-deploy launches, fv-control's pod, template and endpoint equal fv-deploy's field by field,
//      but for the differences listed here (each explained in ADR-0018 §7.4); a new difference, or a listed one that no
//      longer occurs, fails;
//   4. every fv-control launch path builds a Launch that passes the schema, stored specs included (read-compat).
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { DEPLOY_SCHEMAS, launchGenTs, OUTPUTS } from "../../gen-configs.mjs";
import { LaunchZ } from "../../src/launch-gen";
import { checkLaunch, DEFAULT_SERVERLESS, FV_DEPLOY_DEFAULT_GUARDS, LAUNCH_SCHEMA, LaunchSchemaError, runpodEndpointBody, runpodPodBody, runpodTemplateBody, type Launch } from "../../src/launch";
import { clusterGuards, EDGE_WORKER_BOOT, MAX_REPLACE, PULL_DEADLINE_S, WATCHDOG_WORKER_BOOT, WORKER_BOOT, workerCreatePayload, workerLaunch, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { defaultSpec, migrateSpec, normalizeSpec, POOL_PRESETS, presetPool, TEMPLATES, type ClusterSpec } from "../../src/cluster/spec";
import { standaloneSpec } from "../../src/standalone";
import { endpointCreatePayload, endpointGuards, endpointLaunch, endpointUpdatePayload, SLS_BOOT, SLS_GUARDS_NOT_CHECKED, templateCreatePayload } from "../../src/serverless/payloads";
import { normalizeEndpointSpec, type EndpointSpec } from "../../src/serverless/spec";
import { SLS_PRESETS } from "../../src/serverless/presets";
import { specOf, type SlsRow } from "../../src/serverless/ops";
import { b64utf8 } from "../../src/worker-config";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";

const repo = join(__dirname, "../../..");
const fixture = (name: string) => JSON.parse(readFileSync(join(repo, "crates/fastvideo-deploy/fixtures/deploy", name), "utf8"));
/** A body as it goes over the wire (undefined fields dropped). */
const wire = (x: unknown) => JSON.parse(JSON.stringify(x));

// The inputs of scripts/serve/tests/fv-deploy-goldens.sh and payload_goldens.rs.
const IMAGE = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:1111111111111111111111111111111111111111111111111111111111111111";
const GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition";
const KEY_HASH_PLACEHOLDER = "<sha256 of the run key>";
const SECRET_REFS = Object.fromEntries(
  ["cf_account_id", "cf_api_token", "d1_database_id", "r2_bucket", "r2_endpoint", "r2_access_key_id", "r2_secret_access_key", "webhook_ed25519_key"].map((s) => [`FV_${s.toUpperCase()}`, `{{ RUNPOD_SECRET_fv_${s} }}`]),
);
const imageIdent = (image: string) => ({ FV_IMAGE_REF: image, FV_IMAGE_DIGEST: image.split("@")[1]! });

/** launch.rs script_pod for the goldens' inputs (the config inline: runpod.toml). */
function scriptPod(): Launch {
  return {
    schema: LAUNCH_SCHEMA,
    name: "fv-serve-plan",
    provider: "runpod-pod",
    image: IMAGE,
    entrypoint: ["/opt/fastvideo-rs/bin/fv-boot"],
    env: { ...SECRET_REFS, ...imageIdent(IMAGE), FV_SERVE_MODE: "http", FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/workspace/weights", FV_API_KEYS: KEY_HASH_PLACEHOLDER, FV_SERVE_FORWARD: "1", RUST_LOG: "info", FV_WORKER_TOML_B64: b64utf8(WORKER_CONFIGS["runpod.toml"]!) },
    ports: ["8000/http", "70000/tcp"],
    gpu: { types: [GPU], count: 1 },
    placement: { data_centers: ["EU-RO-1"], volume: { id: "bz0ofkfx45", mount_path: "/workspace" }, cloud: "SECURE" },
    disk: { container_gb: 30, volume_gb: 0 },
    weights: "volume",
    guards: FV_DEPLOY_DEFAULT_GUARDS,
  };
}
/** launch.rs script_endpoint for the goldens' inputs. */
function scriptEndpoint(): Launch {
  return {
    schema: LAUNCH_SCHEMA,
    name: "fv-serve-plan",
    provider: "runpod-serverless",
    image: IMAGE,
    entrypoint: ["/opt/fastvideo-rs/bin/fv-boot"],
    env: { ...SECRET_REFS, ...imageIdent(IMAGE), FV_SERVE_MODE: "runpod-queue", FV_AUTH_MODE: "trust-gateway", FV_STATE_DIR: "/fvstate", FV_WEIGHTS: "/runpod-volume/weights", FV_CACHE_DIR: "/fvstate/cache", RUST_LOG: "info", FV_WORKER_TOML_B64: b64utf8(WORKER_CONFIGS["runpod.toml"]!) },
    gpu: { types: [GPU], count: 1, min_cuda: "13.0" },
    placement: { data_centers: ["EU-RO-1"], volume: { id: "bz0ofkfx45", mount_path: "/runpod-volume" }, cloud: "SECURE" },
    disk: { container_gb: 20, volume_gb: 0 },
    weights: "volume",
    guards: FV_DEPLOY_DEFAULT_GUARDS,
    serverless: DEFAULT_SERVERLESS,
  };
}

describe("fv-deploy's types in TypeScript (src/launch-gen.ts, gen-configs.mjs)", () => {
  it("is generated from the committed schemas and current", () => {
    expect(DEPLOY_SCHEMAS).toEqual(["launch", "instance", "offer", "account"]);
    expect(readFileSync(join(repo, "control/src/launch-gen.ts"), "utf8")).toBe((OUTPUTS as Record<string, () => string>)["src/launch-gen.ts"]!());
    const ts = launchGenTs();
    for (const t of ["interface Launch ", "interface Instance ", "interface Offer ", "interface Account ", 'type ProviderKind = "runpod-pod" | "runpod-serverless"', "LaunchZ: z.ZodType<Launch>"]) expect(ts).toContain(t);
  });
  it("validates as serde reads: unknown fields, a wrong kind, a negative disk or a missing guard are refused", () => {
    for (const l of [scriptPod(), scriptEndpoint()]) expect(LaunchZ.safeParse(wire(l)).success, l.provider).toBe(true);
    const bad = (f: (l: any) => void) => {
      const l = wire(scriptPod());
      f(l);
      return LaunchZ.safeParse(l).success;
    };
    expect(bad(() => {})).toBe(true);
    expect(bad((l) => (l.extra = 1))).toBe(false);
    expect(bad((l) => (l.gpu.vram = 96))).toBe(false);
    expect(bad((l) => (l.provider = "gmi"))).toBe(false);
    expect(bad((l) => (l.disk.container_gb = -1))).toBe(false);
    expect(bad((l) => (l.disk.container_gb = 1.5))).toBe(false);
    expect(bad((l) => delete l.guards)).toBe(false);
    expect(bad((l) => (l.weights = "s3"))).toBe(false);
    expect(bad((l) => (l.env.X = 1))).toBe(false);
    expect(bad((l) => (l.placement.volume = null))).toBe(true);
    expect(() => checkLaunch({ ...scriptPod(), provider: "nope" as any })).toThrow(LaunchSchemaError);
  });
  it("a schema construct the generator does not translate stops it", () => {
    const dir = mkdtempSync(join(tmpdir(), "c3c-schemas-"));
    try {
      for (const s of DEPLOY_SCHEMAS) cpSync(join(repo, "crates/fastvideo-deploy/schemas", `${s}.v1.json`), join(dir, `${s}.v1.json`));
      expect(launchGenTs(dir)).toBe(launchGenTs());
      const p = join(dir, "launch.v1.json");
      const j = JSON.parse(readFileSync(p, "utf8"));
      const schemaProp = j.properties.schema;
      j.properties.schema = { type: "integer", format: "int8" };
      writeFileSync(p, JSON.stringify(j));
      expect(() => launchGenTs(dir)).toThrow(/^Launch\.schema: integer format int8 \(gen-configs\.mjs schemaNode/);
      j.properties.schema = schemaProp;
      // A keyword the validator would not enforce stops the generator instead of being dropped.
      for (const [prop, node, why] of [
        ["name", { type: "string", pattern: "^fv" }, "unsupported keyword pattern on a string node"],
        ["name", { type: "string", minLength: 1 }, "unsupported keyword minLength on a string node"],
        ["name", { type: "string", format: "uri" }, "unsupported keyword format on a string node"],
        ["name", { type: "string", const: "x" }, "unsupported keyword const on a string node"],
        ["ports", { type: "array", items: { type: "string" }, maxItems: 4 }, "unsupported keyword maxItems on a array node"],
        ["ports", { type: "array", items: { type: "string" }, uniqueItems: true }, "unsupported keyword uniqueItems on a array node"],
        ["schema", { type: "integer", format: "uint32", minimum: 0, maximum: 1 }, "unsupported keyword maximum on a integer node"],
        ["schema", { type: "integer", format: "uint32", minimum: 0, exclusiveMinimum: 0 }, "unsupported keyword exclusiveMinimum on a integer node"],
      ] as [string, unknown, string][]) {
        const keep = j.properties[prop];
        j.properties[prop] = node;
        writeFileSync(p, JSON.stringify(j));
        expect(() => launchGenTs(dir), why).toThrow(new RegExp(`^Launch\\.${prop}: ${why} \\(`));
        j.properties[prop] = keep;
      }
      writeFileSync(p, JSON.stringify(j));
      expect(launchGenTs(dir)).toBe(launchGenTs());
      j.$defs.ProviderKind.oneOf[0] = { type: "string", const: "runpod-pod" };
      j.$defs.ProviderKind.oneOf.push({ type: "string", const: "vast" });
      writeFileSync(p, JSON.stringify(j));
      // A $defs entry two schemas share must agree (ProviderKind is in all four).
      expect(() => launchGenTs(dir)).toThrow(/^instance\.v1\.json: \$defs ProviderKind differs from launch\.v1\.json's$/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
  it("a Rust type change whose schema is committed but not regenerated fails --check (on a copy of the tree)", () => {
    const PATHS = ["control/gen-configs.mjs", "control/package.json", "control/src", "families", "deploy", "configs", "docker", "docs/serve/images.md", "scripts/gpu", "scripts/serve/variants.sh", "scripts/gcp", ".github/workflows/serve-image.yml", "crates/fastvideo-serve/console", "crates/fastvideo-deploy/schemas"];
    const root = mkdtempSync(join(tmpdir(), "c3c-gen-"));
    try {
      for (const p of PATHS) {
        mkdirSync(dirname(join(root, p)), { recursive: true });
        cpSync(join(repo, p), join(root, p), { recursive: true });
      }
      symlinkSync(join(repo, "control/node_modules"), join(root, "control/node_modules"), "dir");
      const gen = (...args: string[]) => execFileSync(process.execPath, ["gen-configs.mjs", ...args], { cwd: join(root, "control"), encoding: "utf8", stdio: "pipe" });
      gen("--check");
      // As C3b adds a provider kind: every schema that names ProviderKind changes.
      for (const s of DEPLOY_SCHEMAS) {
        const p = join(root, "crates/fastvideo-deploy/schemas", `${s}.v1.json`);
        const j = JSON.parse(readFileSync(p, "utf8"));
        j.$defs.ProviderKind.oneOf.push({ description: "A Vast.ai instance.", type: "string", const: "vast" });
        writeFileSync(p, JSON.stringify(j, null, 2) + "\n");
      }
      let err = "";
      try {
        gen("--check");
      } catch (e) {
        err = String((e as { stderr?: string }).stderr ?? e);
      }
      expect(err).toContain("stale: src/launch-gen.ts");
      gen();
      expect(readFileSync(join(root, "control/src/launch-gen.ts"), "utf8")).toContain('export type ProviderKind = "runpod-pod" | "runpod-serverless" | "vast";');
      gen("--check");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});

describe("a Launch's Runpod bodies are the Rust provider's (the C3a payload goldens)", () => {
  it("pod: runpod-pod.json", () => {
    expect(wire(runpodPodBody(checkLaunch(scriptPod())))).toStrictEqual(fixture("runpod-pod.json"));
  });
  it("serverless: runpod-template.json and runpod-endpoint.json", () => {
    const l = checkLaunch(scriptEndpoint());
    expect(wire(runpodTemplateBody(l))).toStrictEqual(fixture("runpod-template.json"));
    expect(wire(runpodEndpointBody(l, "<template id>"))).toStrictEqual(fixture("runpod-endpoint.json"));
    // No serverless section: runpod-endpoint.sh's scaling (launch.rs default_serverless).
    expect(wire(runpodEndpointBody({ ...l, serverless: null }, "<template id>"))).toStrictEqual(fixture("runpod-endpoint.json"));
  });
  it("optional fields as the Rust builders: empty lists and absent options are left out, a filter and a start command are sent", () => {
    const l: Launch = { ...scriptPod(), ports: [], entrypoint: [], args: ["--config", "/etc/fv/x.toml"], placement: { cloud: "COMMUNITY" }, gpu: { types: [GPU], count: 2, allowed_cuda: ["13.0"] } };
    const b = wire(runpodPodBody(l));
    expect(b).not.toHaveProperty("ports");
    expect(b).not.toHaveProperty("dockerEntrypoint");
    expect(b).not.toHaveProperty("networkVolumeId");
    expect(b).not.toHaveProperty("dataCenterIds");
    expect(b).toMatchObject({ dockerStartCmd: ["--config", "/etc/fv/x.toml"], cloudType: "COMMUNITY", gpuCount: 2, allowedCudaVersions: ["13.0"] });
    const e = wire(runpodEndpointBody({ ...scriptEndpoint(), gpu: { types: [GPU], count: 1 }, placement: {} }, "t"));
    expect(e).not.toHaveProperty("minCudaVersion");
    expect(e).not.toHaveProperty("networkVolumeId");
    expect(() => runpodPodBody(scriptEndpoint())).toThrow(/no pod body/);
    expect(() => runpodTemplateBody(scriptPod())).toThrow(/no serverless template/);
  });
});

// ---------------------------------------------------------------- fv-control = fv-deploy for the same preset
// The presets fv-deploy launches (payload_goldens.rs every_preset_builds_or_is_refused_for_its_trees: their trees
// are on the EU volume) and their configs (deploy/deploy.toml).
const PRESETS: Record<string, string> = { "h3-turbo": "runpod.toml", "h3-max": "runpod-h3-max.toml", ltx: "runpod-ltx.toml", "ltx-pro": "runpod-ltx-pro.toml" };

/** fv-deploy's preset pod (`fv-deploy plan --kind pod --preset <p>`, pinned to IMAGE): the script pod plus what
 * payload_goldens.rs preset_pod_is_the_script_pod_plus_fv_boot allows (name, fv-control's 40 GB disk, the weights
 * source and the channel), with the preset's config inline (every_preset_builds_…: the committed file). */
function fvDeployPod(preset: string) {
  const want = fixture("runpod-pod.json");
  want.name = `fv-deploy-${preset}-1024120000`;
  want.containerDiskInGb = 40;
  Object.assign(want.env, { FV_WEIGHTS_SOURCE: "volume", FV_RELEASE_CHANNEL: "stable", FV_WORKER_TOML_B64: b64utf8(WORKER_CONFIGS[PRESETS[preset]!]!) });
  return want;
}
/** fv-deploy's preset template and endpoint (preset_endpoint_is_the_script_endpoint_plus_fv_boot). */
function fvDeployServerless(preset: string) {
  const t = fixture("runpod-template.json");
  t.name = `fv-deploy-${preset}-1024120000`;
  Object.assign(t.env, { FV_WEIGHTS_SOURCE: "volume", FV_RELEASE_CHANNEL: "stable", FV_WORKER_TOML_B64: b64utf8(WORKER_CONFIGS[PRESETS[preset]!]!) });
  const e = fixture("runpod-endpoint.json");
  e.name = `fv-deploy-${preset}-1024120000`;
  return { t, e };
}

const SECRETS = { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_x", ingest_token: "fvi_x" };
const EDGE = { url: "https://edge.test", internal_token: "eit", admin_token: "eat", d1_database_id: "d1x", outputs_bucket: "outb" };
const INGEST = "https://ctl/ingest/v1/logs";
type PodContext = "standalone" | "direct" | "edge";
/** The edge family each preset's models queue on (cluster/spec.ts modelFamily). */
const EDGE_FAMILY: Record<string, string> = { "h3-turbo": "h3", "h3-max": "h3", ltx: "ltx", "ltx-pro": "ltx" };

/** What fv-control sends instead of fv-deploy's pod, by context: the intended differences of ADR-0018 §7.4, every
 * one pinned to its value. `set`: fields and env keys fv-control sets to these values (absent from fv-deploy's, or
 * there with another value); `drop`: env keys only fv-deploy sends. */
function podDifferences(ctx: PodContext, preset: string, spec: ClusterSpec): { set: Record<string, unknown>; env: Record<string, string>; drop: string[] } {
  const pool = spec.pools[0]!;
  const backstop = { FV_CLUSTER_DEADLINE: "1700000000", FV_MIN_BALANCE: String(spec.min_balance), FV_BACKSTOP_API_KEY: "rpa_K" };
  const cluster = { FV_SERVE_ROLE: "worker", FV_URL_SIGNING_KEY: "us", FV_JOBS_HEARTBEAT_S: "10", FV_LOG_SHIP_URL: INGEST, FV_LOG_SHIP_TOKEN: "fvi_x", FV_LOG_SHIP_LEVEL: "info" };
  const direct = { FV_INTERNAL_TOKEN: "it", FV_PUBLIC_BASE_URL: "", FV_WORKER_DIRECT: "1", FV_AUTH_MODE: "keys", FV_KEY_STORE: "d1", FV_ADMIN_TOKEN: "fvadm_x" };
  const fam = EDGE_FAMILY[preset]!;
  const edge = {
    FV_INTERNAL_TOKEN: "eit",
    FV_PUBLIC_BASE_URL: EDGE.url,
    FV_AUTH_MODE: "trust-edge",
    FV_DISPATCH_FRONT: "1",
    FV_DISPATCH_DO_URL: EDGE.url,
    FV_DISPATCH_FAMILIES: fam,
    FV_DISPATCH_MODEL_FAMILIES: (pool.models || []).map((m) => `${m.id}=${fam}`).join(","),
    FV_DISPATCH_DIRECT_UPLOAD: "1",
    FV_MP4_FRAGMENTED: "1",
    FV_DISPATCH_MAX_QUEUED: String(pool.max_queued),
    // An edge front writes through the edge: its outputs bucket and its D1 instead of the account's secret references.
    FV_R2_BUCKET: EDGE.outputs_bucket,
    FV_D1_DATABASE_ID: EDGE.d1_database_id,
  };
  const boot = { standalone: WATCHDOG_WORKER_BOOT, direct: WORKER_BOOT, edge: EDGE_WORKER_BOOT }[ctx];
  return {
    set: { name: "fvc-x", dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [boot] },
    env: { ...cluster, ...(ctx === "edge" ? { ...edge, ...backstop } : ctx === "standalone" ? { ...direct, ...backstop } : direct) },
    drop: ["FV_API_KEYS", "FV_SERVE_FORWARD", "FV_WEIGHTS_SOURCE"],
  };
}
/** fv-deploy's body with fv-control's intended differences applied; and which of them are real differences. */
function applyDifferences(deploy: Record<string, any>, d: { set: Record<string, unknown>; env: Record<string, string>; drop: string[] }) {
  const want = structuredClone(deploy);
  const real: string[] = [];
  for (const [k, v] of Object.entries(d.set)) {
    if (JSON.stringify(deploy[k]) !== JSON.stringify(v)) real.push(k);
    want[k] = v;
  }
  for (const [k, v] of Object.entries(d.env)) {
    if (deploy.env[k] !== v) real.push(`env.${k}`);
    want.env[k] = v;
  }
  for (const k of d.drop) {
    if (k in deploy.env) real.push(`env.-${k}`);
    delete want.env[k];
  }
  return { want, real };
}

function controlPod(spec: ClusterSpec, over: Partial<EnvCtx>) {
  const pool = spec.pools[0]!;
  const ctx: EnvCtx = { spec, state: { images: {}, workers: {} }, secrets: SECRETS, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", ingestUrl: INGEST, ...over };
  const env = workerSystemEnv(ctx, pool, IMAGE);
  const pl = workerPlacements(spec, pool)[0]!;
  return { launch: workerLaunch("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec)), body: wire(workerCreatePayload("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec))) };
}
const podSpec = (ctx: PodContext, preset: string): [ClusterSpec, Partial<EnvCtx>] =>
  ctx === "standalone"
    ? [standaloneSpec({ name: "solo", preset } as never).spec, { backstop: true }]
    : [normalizeSpec({ ...defaultSpec("c"), control_plane: ctx, pools: [presetPool(preset)] }), ctx === "edge" ? { edge: EDGE } : {}];

describe("for a preset fv-deploy launches, fv-control's payloads are fv-deploy's plus the intended differences of its context", () => {
  for (const ctx of ["standalone", "direct", "edge"] as PodContext[])
    it(`${ctx} pods (every preset): exactly fv-deploy's pod with the ${ctx} differences, each of which occurs`, () => {
      const seen = new Set<string>();
      let listed: string[] = [];
      for (const preset of Object.keys(PRESETS)) {
        const [spec, over] = podSpec(ctx, preset);
        const { launch, body } = controlPod(spec, over);
        expect(launch).toMatchObject({ provider: "runpod-pod", weights: "volume", placement: { region: "eu", volume: { id: "bz0ofkfx45", mount_path: "/workspace" } } });
        const diff = podDifferences(ctx, preset, spec);
        const { want, real } = applyDifferences(fvDeployPod(preset), diff);
        expect(body, `${preset} ${ctx}`).toStrictEqual(want);
        real.forEach((r) => seen.add(r));
        listed = [...Object.keys(diff.set), ...Object.keys(diff.env).map((k) => `env.${k}`), ...diff.drop.map((k) => `env.-${k}`)];
      }
      // No listed difference is stale: each differs from fv-deploy's for at least one preset (this test alone).
      expect(listed.filter((d) => !seen.has(d))).toEqual([]);
    });
  it("serverless (every preset): exactly fv-deploy's template and endpoint with the serverless differences, each of which occurs", () => {
    const TIMEOUT_MS: Record<string, number> = { "h3-turbo": 1_800_000, "h3-max": 3_600_000, ltx: 1_800_000, "ltx-pro": 3_600_000 };
    const seen = new Set<string>();
    let listed: string[] = [];
    for (const preset of Object.keys(PRESETS)) {
      const { t, e } = fvDeployServerless(preset);
      const spec = normalizeEndpointSpec({ name: "ep", preset });
      expect(endpointLaunch(spec, IMAGE)).toMatchObject({ provider: "runpod-serverless", weights: "volume", gpu: { min_cuda: "13.0" }, placement: { volume: { id: "bz0ofkfx45", mount_path: "/runpod-volume" } } });
      const td = {
        set: { name: "fvc-ep-261025120000", dockerEntrypoint: ["/bin/sh", "-c", SLS_BOOT], dockerStartCmd: [], category: "NVIDIA", readme: "fv-control serverless endpoint ep (cuda); docs/control/serverless.md" },
        env: { FV_CONTROL_ENDPOINT: "ep" },
        drop: ["FV_WEIGHTS_SOURCE"],
      };
      const tw = applyDifferences(t, td);
      expect(wire(templateCreatePayload(spec, IMAGE, "fvc-ep-261025120000")), `${preset} template`).toStrictEqual(tw.want);
      const ed = { set: { name: "fvc-ep", scalerValue: 4, executionTimeoutMs: TIMEOUT_MS[preset] }, env: {}, drop: [] };
      const ew = applyDifferences({ ...e, env: {} }, ed);
      delete ew.want.env;
      expect(wire(endpointCreatePayload(spec, "<template id>")), `${preset} endpoint`).toStrictEqual(ew.want);
      [...tw.real.map((r) => `template.${r}`), ...ew.real.map((r) => `endpoint.${r}`)].forEach((r) => seen.add(r));
      listed = [
        ...Object.keys(td.set).map((k) => `template.${k}`),
        ...Object.keys(td.env).map((k) => `template.env.${k}`),
        ...td.drop.map((k) => `template.env.-${k}`),
        ...Object.keys(ed.set).map((k) => `endpoint.${k}`),
      ];
    }
    expect(listed.filter((d) => !seen.has(d))).toEqual([]);
  });
});

// ---------------------------------------------------------------- every path builds a Launch that passes the schema
// The bodies themselves are held to main's builders by payload-corpus.test.ts (a snapshot taken before C3c); here the
// Launch each path builds is checked against the spec it came from, independently of the body code.
describe("every fv-control Runpod GPU launch is a schema-valid Launch that says what the spec asks, stored specs included", () => {
  it("every cluster template and every pool preset (as a standalone pod, on each weights source it allows)", () => {
    let n = 0;
    const specs: ClusterSpec[] = Object.keys(TEMPLATES).map((t) => ({ ...defaultSpec("c", t), control_plane: "direct" as const }));
    for (const p of POOL_PRESETS)
      for (const x of [{}, { weights_source: "r2" }, { weights_source: "r2", data_centers: ["US-TX-3", "any"] }, { weights_source: "hub", weights_download_approved: true }, { weights_source: "none" }, { upsampler_local: true }]) {
        try {
          specs.push(standaloneSpec({ name: "solo", preset: p.id, ...x } as never).spec);
        } catch {
          /* a combination the launch form refuses */
        }
      }
    for (const spec of specs)
      for (const pool of spec.pools) {
        const env = workerSystemEnv({ spec, state: { images: {}, workers: {} }, secrets: SECRETS, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", backstop: true }, pool, IMAGE);
        for (const pl of workerPlacements(spec, pool)) {
          if (pool.compute === "CPU") {
            expect(() => workerLaunch("fvc-x", IMAGE, pool, pl, env)).toThrow(/CPU pod is not a Launch/);
            expect(workerCreatePayload("fvc-x", IMAGE, pool, pl, env)).toMatchObject({ computeType: "CPU", cpuFlavorIds: [pl.cpu] });
            continue;
          }
          const l = workerLaunch("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec));
          expect(LaunchZ.safeParse(wire(l)).success).toBe(true);
          const ws = env.FV_WEIGHTS_SOURCE;
          const mounts = pool.volume !== false && ws !== "r2" && ws !== "hub" && !!pl.region;
          expect(l).toMatchObject({ provider: "runpod-pod", image: IMAGE, env, entrypoint: ["bash", "-c"], ports: ["8000/http", "70000/tcp"], gpu: { types: [pl.gpu], count: 1, allowed_cuda: pool.allowed_cuda ?? [] }, disk: { volume_gb: 0 } });
          expect(l.args).toHaveLength(1);
          expect(l.args![0]).toContain("exec /opt/fastvideo-rs/bin/fv-serve");
          expect(l.placement.data_centers).toEqual(pl.dc ? [pl.dc] : []);
          expect(l.placement.volume ?? null).toEqual(mounts ? { id: "bz0ofkfx45", mount_path: "/workspace" } : null);
          if (pool.container_disk_gb !== undefined) expect(l.disk.container_gb).toBe(pool.container_disk_gb);
          expect(l.guards).toEqual({ max_usd_per_hr: spec.max_gpu_dph, min_balance_usd: spec.balance_floor, cap_s: spec.cap_s, ready_timeout_s: PULL_DEADLINE_S, ready_cycles: 1 + MAX_REPLACE });
          expect(l.weights).toBe(ws === "r2" ? "r2" : ws === "hub" ? "hub" : mounts ? "volume" : "none");
          n++;
        }
      }
    expect(n).toBeGreaterThan(100);
  });
  it("every serverless preset, both modes and each weights source; CPU endpoints stay outside the schema", () => {
    let n = 0;
    for (const p of SLS_PRESETS)
      for (const x of [{}, { weights_source: "r2" }, { weights_source: "hub", weights_download_approved: true, flashboot: true }, { mode: "lb" }, { allowed_cuda: ["12.8"], gpu_count: 2, env: { FOO: "bar" } }]) {
        let spec: EndpointSpec;
        try {
          spec = normalizeEndpointSpec({ name: "ep", preset: p.id, ...x });
        } catch {
          continue;
        }
        if (spec.compute === "CPU") {
          expect(() => endpointLaunch(spec, IMAGE)).toThrow(/CPU endpoint is not a Launch/);
          continue;
        }
        const l = endpointLaunch(spec, IMAGE);
        expect(LaunchZ.safeParse(wire(l)).success).toBe(true);
        expect(l).toMatchObject({
          provider: "runpod-serverless",
          name: `fvc-${spec.name}`,
          image: IMAGE,
          args: [],
          gpu: { types: spec.gpu_types, count: spec.gpu_count ?? 1, min_cuda: "13.0", allowed_cuda: spec.allowed_cuda ?? [] },
          placement: { data_centers: spec.data_centers ?? [], volume: spec.network_volume ? { id: spec.network_volume, mount_path: "/runpod-volume" } : null },
          disk: { container_gb: spec.container_disk_gb, volume_gb: 0 },
          serverless: { workers_min: spec.workers_min, workers_max: spec.workers_max, idle_timeout_s: spec.idle_timeout_s, flashboot: spec.flashboot, execution_timeout_ms: spec.execution_timeout_s * 1000, scaler_type: spec.scaler_type, scaler_value: spec.scaler_value },
        });
        expect(l.entrypoint![0]).toMatch(/^(\/bin\/sh|bash)$/);
        expect(l.env).toMatchObject({ FV_CONTROL_ENDPOINT: "ep", FV_IMAGE_REF: IMAGE, ...(spec.env || {}) });
        expect(l.weights).toBe(spec.weights_source ?? "volume");
        expect(l.guards).toEqual(endpointGuards(spec));
        n++;
      }
    expect(n).toBeGreaterThan(20);
  });
  it("stored specs load and launch: a cluster stored with the gateway block and image variants, an endpoint row with a variant, a spec without compute", () => {
    // A cluster spec as an older fv-control stored it (D1 clusters.spec): the gateway block, pools by image variant.
    const stored = migrateSpec(
      JSON.parse(
        JSON.stringify({
          name: "old",
          image: { channel: "stable" },
          regions: ["eu"],
          gateway: { enabled: false, auth: "keys" },
          pools: [
            { id: "h3-turbo", count: 1, variant: "h3-turbo", models: presetPool("h3-turbo")!.models },
            { id: "fake", count: 1, variant: "cpu", fake_models: ["fake-wan"], cpu_flavors: ["cpu3c"], vcpu: 2, container_disk_gb: 10, volume: false },
          ],
          cap_s: 3600,
          min_balance: 8.25,
          balance_floor: 8,
          min_start: 20,
          max_gpu_dph: 3.6,
          log_shipping: true,
        }),
      ),
    );
    expect(stored.control_plane).toBe("direct");
    const [gpu, cpu] = stored.pools;
    expect([gpu!.compute, gpu!.config, cpu!.compute]).toEqual(["GPU", "runpod.toml", "CPU"]);
    const { launch, body } = controlPod({ ...stored, pools: [gpu!] }, {});
    expect(LaunchZ.safeParse(wire(launch)).success).toBe(true);
    expect(launch.guards).toMatchObject({ cap_s: 3600, max_usd_per_hr: 3.6 });
    expect(body.env.FV_WORKER_TOML_B64).toBe(b64utf8(WORKER_CONFIGS["runpod.toml"]!));
    expect(body).toMatchObject({ networkVolumeId: "bz0ofkfx45", dataCenterIds: ["EU-RO-1"], dockerEntrypoint: ["bash", "-c"], dockerStartCmd: [WORKER_BOOT] });
    // An endpoint row stored before the one image (serverless_endpoints.spec): a variant and no config.
    const row = { spec: JSON.stringify({ ...normalizeEndpointSpec({ name: "old", compute: "GPU", config: "runpod-ltx.toml" }), config: undefined, compute: undefined, variant: "ltx" }) } as unknown as SlsRow;
    const s = specOf(row);
    expect([s.compute, s.config]).toEqual(["GPU", "runpod-ltx.toml"]);
    expect(LaunchZ.safeParse(wire(endpointLaunch(s, IMAGE))).success).toBe(true);
    expect(wire(templateCreatePayload(s, IMAGE, "t")).env.FV_WORKER_TOML_B64).toBe(b64utf8(WORKER_CONFIGS["runpod-ltx.toml"]!));
    const want = { name: "fvc-old", templateId: "tpl", computeType: "GPU", gpuTypeIds: s.gpu_types, gpuCount: 1, networkVolumeId: "bz0ofkfx45", dataCenterIds: ["EU-RO-1"], workersMin: 0, workersMax: 1, idleTimeout: 5, flashboot: false, executionTimeoutMs: 1_800_000, scalerType: "QUEUE_DELAY", scalerValue: 4, minCudaVersion: "13.0" };
    expect(wire(endpointCreatePayload(s, "tpl"))).toStrictEqual(want);
    // A spec without `compute` reads as GPU, as before C3c: the GPU Launch, template category and endpoint fields.
    const { compute: _c, ...bare } = s;
    const nb = bare as EndpointSpec;
    expect(wire(templateCreatePayload(nb, IMAGE, "t")).category).toBe("NVIDIA");
    expect(wire(endpointCreatePayload(nb, "tpl"))).toStrictEqual(want);
    const { name: _n, templateId: _t, networkVolumeId: _v, computeType: _ct, ...fields } = want;
    expect(wire(endpointUpdatePayload(nb))).toStrictEqual(fields);
  });
});

// ---------------------------------------------------------------- failures: typed, clean, and never with a value
describe("a launch that fails the schema", () => {
  it("never puts an env (or any) value in the error: zod reports paths and expectations only", () => {
    const SECRET = "s3cret-VALUE-do-not-print";
    const cases: ((l: any) => void)[] = [
      (l) => (l.env.FV_TOKEN = { value: SECRET }),
      (l) => (l.env.FV_OTHER = 7),
      (l) => (l.weights = SECRET),
      (l) => (l.provider = SECRET),
      (l) => (l.disk.container_gb = SECRET),
      (l) => (l.placement.cloud = SECRET),
      (l) => (l.extra = SECRET),
      (l) => (l.gpu.types = [SECRET, 3]),
    ];
    for (const f of cases) {
      const l = wire(scriptPod());
      l.env.FV_ADMIN_TOKEN = SECRET;
      l.env.FV_BACKSTOP_API_KEY = SECRET;
      f(l);
      let msg = "";
      try {
        checkLaunch(l);
      } catch (e) {
        expect(e).toBeInstanceOf(LaunchSchemaError);
        msg = (e as Error).message + JSON.stringify((e as LaunchSchemaError).issues);
      }
      expect(msg, f.toString()).toMatch(/does not match fv-deploy's Launch schema/);
      expect(msg, f.toString()).not.toContain("s3cret");
    }
  });
  it("is refused before the op (409, typed issues) and, should one reach a create, ends that pool's create cleanly", async () => {
    const { d1 } = await import("./d1shim");
    const { insertCluster, newSecrets } = await import("../../src/cluster/store");
    const { createWorker } = await import("../../src/cluster/ops");
    const { assertPoolLaunches, poolLaunchIssues } = await import("../../src/cluster/payloads");
    const { HttpError } = await import("../../src/util");
    const { b64, randomBytes } = await import("../../src/crypto");
    const env = { DB: d1(), CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_x" } as never;
    const spec = normalizeSpec({ ...defaultSpec("bad"), control_plane: "direct", pools: [presetPool("h3-turbo")] });
    expect(poolLaunchIssues(spec)).toEqual([]);
    // A value the Launch schema refuses (a fractional disk), as a spec stored by an older fv-control could carry.
    const stored = structuredClone(spec);
    stored.pools[0]!.container_disk_gb = 40.5;
    const issues = poolLaunchIssues(stored);
    expect(issues.map((i) => i.path)).toEqual([["pools", 0]]);
    expect(issues[0]!.message).toMatch(/disk\.container_gb/);
    let err: unknown;
    try {
      assertPoolLaunches(stored);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(HttpError);
    expect((err as InstanceType<typeof HttpError>).status).toBe(409);
    const { s, ingestHash } = await newSecrets();
    const c = await insertCluster(env, spec, { images: {}, workers: {} }, s, ingestHash, "owner", "controller", "running", null);
    c.spec = stored;
    const calls: string[] = [];
    const prev = globalThis.fetch;
    globalThis.fetch = (async (u: unknown) => {
      calls.push(String(u));
      throw new Error("no network in this test");
    }) as typeof fetch;
    const log: string[] = [];
    try {
      expect(await createWorker(env, c, "h3-turbo", IMAGE, "workers", (m: string) => log.push(m))).toBeNull();
    } finally {
      globalThis.fetch = prev;
    }
    expect(calls.filter((u) => u.includes("runpod"))).toEqual([]);
    expect(log.join("\n")).toMatch(/h3-turbo: no pod: launch fv-ctl-bad-.* does not match fv-deploy's Launch schema: disk\.container_gb/);
  });
  it("serverless guards: the fields fv-control does not check carry fv-deploy's fail-closed values, and are named", () => {
    expect(endpointGuards({ deadline_min: 120 })).toEqual({ max_usd_per_hr: 0, min_balance_usd: 8, cap_s: 7200, ready_timeout_s: 0, ready_cycles: 1 });
    expect(endpointGuards({ deadline_min: null }).cap_s).toBe(0);
    expect([...SLS_GUARDS_NOT_CHECKED]).toEqual(["max_usd_per_hr", "ready_timeout_s"]);
  });
});
