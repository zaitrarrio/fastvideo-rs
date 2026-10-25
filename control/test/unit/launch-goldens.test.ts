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
import { clusterGuards, workerCreatePayload, workerLaunch, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { defaultSpec, migrateSpec, normalizeSpec, POOL_PRESETS, presetPool, TEMPLATES, type ClusterSpec } from "../../src/cluster/spec";
import { standaloneSpec } from "../../src/standalone";
import { endpointCreatePayload, endpointGuards, endpointLaunch, endpointUpdatePayload, templateCreatePayload, templateUpdatePayload, v2CreatePayload } from "../../src/serverless/payloads";
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
      j.properties.schema = { type: "integer", format: "int8" };
      writeFileSync(p, JSON.stringify(j));
      expect(() => launchGenTs(dir)).toThrow(/launch.v1.json|Launch\.schema: integer format int8/);
      j.properties.schema = { type: "integer", format: "uint32", minimum: 0 };
      j.$defs.ProviderKind.oneOf[0] = { type: "string", const: "runpod-pod" };
      j.$defs.ProviderKind.oneOf.push({ type: "string", const: "vast" });
      writeFileSync(p, JSON.stringify(j));
      // A $defs entry two schemas share must agree (ProviderKind is in all four).
      expect(() => launchGenTs(dir)).toThrow(/\$defs ProviderKind differs/);
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

/** The intended differences (ADR-0018 §7.4): the field, and why. Anything else must be equal. */
const POD_DIFFERS: Record<string, string> = {
  name: "fv-control names its pods fvc-… / its cluster's names, fv-deploy fv-deploy-<preset>-<stamp>",
  dockerEntrypoint: "fv-control starts `bash -c <boot>` (its shell boot runs fv-boot first when the image has it, else the shell steps; the watchdog, r2 / hub fetch and upsampler sidecar for images without fv-boot); fv-deploy starts fv-boot",
  dockerStartCmd: "the shell boot (see dockerEntrypoint); fv-deploy sends none",
};
const TEMPLATE_DIFFERS: Record<string, string> = {
  name: "fv-control's template name is fvc-<endpoint>-<UTC stamp> (one per create), fv-deploy's the launch name",
  dockerEntrypoint: "fv-control starts `/bin/sh -c <SLS_BOOT>` (fv-boot first when the image has it), fv-deploy fv-boot",
  dockerStartCmd: "fv-control sends [] so a PATCH clears an older template's start command",
  category: "fv-control names the template's category (NVIDIA)",
  readme: "fv-control's template readme",
};
const ENDPOINT_DIFFERS: Record<string, string> = {
  name: "fvc-<endpoint> vs fv-deploy-<preset>-<stamp>",
  scalerValue: "fv-control's serverless default is QUEUE_DELAY 4 s (a queued job waits 4 s before another worker), runpod-endpoint.sh's 1 s",
  executionTimeoutMs: "fv-control's per-preset job timeout (serverless/presets.ts), runpod-endpoint.sh's 30 min",
};
/** Env keys only fv-control sets (its cluster, direct and edge workers, the in-pod backstop, log shipping), only
 * fv-deploy sets, or both with another value. */
const ENV_CONTROL_ONLY: Record<string, string> = {
  FV_SERVE_ROLE: "fv-control's workers register as workers",
  FV_INTERNAL_TOKEN: "the cluster's internal token (the edge's for an edge front)",
  FV_URL_SIGNING_KEY: "the cluster's URL signing key",
  FV_PUBLIC_BASE_URL: "empty (direct) or the edge's URL",
  FV_JOBS_HEARTBEAT_S: "the job heartbeat the cluster's lost-job check reads",
  FV_WORKER_DIRECT: "a direct worker serves clients itself",
  FV_AUTH_MODE: "direct: the spec's auth; edge: trust-edge (fv-deploy's pod uses the image's keys mode with FV_API_KEYS)",
  FV_KEY_STORE: "direct workers share minted keys in D1",
  FV_ADMIN_TOKEN: "the cluster's admin token (minted keys) instead of fv-deploy's one run key",
  FV_CLUSTER_DEADLINE: "the in-pod watchdog's deadline (standalone pods, edge fronts)",
  FV_MIN_BALANCE: "the in-pod watchdog's balance floor",
  FV_BACKSTOP_API_KEY: "the in-pod watchdog's Runpod key (fv-control's own; ADR-0018 open decision 'In-pod backstop': fv-deploy sends none)",
  FV_LOG_SHIP_URL: "log shipping to fv-control",
  FV_LOG_SHIP_TOKEN: "log shipping to fv-control",
  FV_LOG_SHIP_LEVEL: "log shipping to fv-control",
  FV_DISPATCH_FRONT: "edge front",
  FV_DISPATCH_DO_URL: "edge front",
  FV_DISPATCH_FAMILIES: "edge front",
  FV_DISPATCH_MODEL_FAMILIES: "edge front",
  FV_DISPATCH_DIRECT_UPLOAD: "edge front",
  FV_MP4_FRAGMENTED: "edge front",
  FV_DISPATCH_MAX_QUEUED: "edge front: the pool's admission limit",
  FV_CONTROL_ENDPOINT: "a serverless worker names its fv-control endpoint",
};
const ENV_DEPLOY_ONLY: Record<string, string> = {
  FV_API_KEYS: "fv-deploy's run key (its SHA-256); fv-control's workers take minted keys (FV_KEY_STORE d1, FV_ADMIN_TOKEN)",
  FV_SERVE_FORWARD: "runpod-pod.sh's forward switch, which fv-control has never sent",
  FV_WEIGHTS_SOURCE: "fv-deploy names `volume`; fv-control sends it only for r2 / hub (fv-boot's default is the volume)",
};
const ENV_VALUE_DIFFERS: Record<string, string> = {
  FV_R2_BUCKET: "an edge front writes to the edge's outputs bucket",
  FV_D1_DATABASE_ID: "an edge front uses the edge's D1",
};

/** Field-by-field differences of two bodies (env key by key), as `field` / `env.KEY` / `env.KEY=`. */
function differences(control: Record<string, any>, deploy: Record<string, any>): string[] {
  const out: string[] = [];
  for (const k of new Set([...Object.keys(control), ...Object.keys(deploy)])) {
    if (k === "env") {
      const a = control.env ?? {};
      const b = deploy.env ?? {};
      for (const e of new Set([...Object.keys(a), ...Object.keys(b)])) if (a[e] !== b[e]) out.push(e in a && e in b ? `env.${e}=` : `env.${e}`);
    } else if (JSON.stringify(control[k]) !== JSON.stringify(deploy[k])) out.push(k);
  }
  return out.sort();
}
const used = new Set<string>();
function expectOnlyIntended(diffs: string[], fields: Record<string, string>, what: string, control: Record<string, any>) {
  for (const d of diffs) {
    used.add(d);
    if (d.startsWith("env.")) {
      const k = d.replace(/^env\./, "").replace(/=$/, "");
      const listed = d.endsWith("=") ? ENV_VALUE_DIFFERS[k] : k in (control.env ?? {}) ? ENV_CONTROL_ONLY[k] : ENV_DEPLOY_ONLY[k];
      expect(listed, `${what}: ${d} is not an intended difference (ADR-0018 §7.4)`).toBeTruthy();
    } else expect(fields[d], `${what}: ${d} is not an intended difference (ADR-0018 §7.4)`).toBeTruthy();
  }
}

const SECRETS = { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_x", ingest_token: "fvi_x" };
const EDGE = { url: "https://edge.test", internal_token: "eit", admin_token: "eat", d1_database_id: "d1x", outputs_bucket: "outb" };
function controlPod(spec: ClusterSpec, over: Partial<EnvCtx>) {
  const pool = spec.pools[0]!;
  const ctx: EnvCtx = { spec, state: { images: {}, workers: {} }, secrets: SECRETS, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", ingestUrl: "https://ctl/ingest/v1/logs", ...over };
  const env = workerSystemEnv(ctx, pool, IMAGE);
  const pl = workerPlacements(spec, pool)[0]!;
  const launch = workerLaunch("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec));
  return { launch, body: wire(workerCreatePayload("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec))) };
}

describe("for a preset fv-deploy launches, fv-control's payloads equal fv-deploy's but for the intended differences", () => {
  for (const preset of Object.keys(PRESETS)) {
    it(`${preset}: a standalone pod, a direct cluster pool and an edge front against fv-deploy's pod`, () => {
      const deploy = fvDeployPod(preset);
      const solo = standaloneSpec({ name: "solo", preset } as any).spec;
      const direct = normalizeSpec({ ...defaultSpec("c"), control_plane: "direct", pools: [presetPool(preset)] });
      const edge = normalizeSpec({ ...defaultSpec("c"), control_plane: "edge", pools: [presetPool(preset)] });
      for (const [what, spec, over] of [
        ["standalone", solo, { backstop: true }],
        ["direct", direct, {}],
        ["edge", edge, { edge: EDGE }],
      ] as [string, ClusterSpec, Partial<EnvCtx>][]) {
        const { launch, body } = controlPod(spec, over);
        // The pod is built from its Launch.
        expect(body).toStrictEqual(wire(runpodPodBody(launch)));
        expect(launch).toMatchObject({ provider: "runpod-pod", weights: "volume", placement: { region: "eu", volume: { id: "bz0ofkfx45", mount_path: "/workspace" } } });
        const diffs = differences(body, deploy);
        expectOnlyIntended(diffs, POD_DIFFERS, `${preset} ${what}`, body);
        // The same config, image, GPU, volume, data centre, disk, ports and secret references.
        for (const k of ["imageName", "gpuTypeIds", "gpuCount", "networkVolumeId", "volumeMountPath", "dataCenterIds", "containerDiskInGb", "volumeInGb", "ports", "cloudType", "computeType"]) expect(body[k], `${preset} ${what} ${k}`).toEqual(deploy[k]);
        expect(body.env.FV_WORKER_TOML_B64).toBe(deploy.env.FV_WORKER_TOML_B64);
      }
    });
    it(`${preset}: a serverless endpoint against fv-deploy's template and endpoint`, () => {
      const { t, e } = fvDeployServerless(preset);
      const spec = normalizeEndpointSpec({ name: "ep", preset });
      const launch = endpointLaunch(spec, IMAGE);
      expect(launch).toMatchObject({ provider: "runpod-serverless", weights: "volume", gpu: { min_cuda: "13.0" }, placement: { volume: { id: "bz0ofkfx45", mount_path: "/runpod-volume" } } });
      const tpl = wire(templateCreatePayload(spec, IMAGE, "fvc-ep-261025120000"));
      expect(tpl).toStrictEqual({ ...wire(runpodTemplateBody({ ...launch, name: "fvc-ep-261025120000" })), category: "NVIDIA", dockerStartCmd: [], readme: tpl.readme });
      expectOnlyIntended(differences(tpl, t), TEMPLATE_DIFFERS, `${preset} template`, tpl);
      expect(tpl.env.FV_WORKER_TOML_B64).toBe(t.env.FV_WORKER_TOML_B64);
      const ep = wire(endpointCreatePayload(spec, "<template id>"));
      expect(ep).toStrictEqual(wire(runpodEndpointBody(launch, "<template id>")));
      expectOnlyIntended(differences(ep, e), ENDPOINT_DIFFERS, `${preset} endpoint`, ep);
    });
  }
  it("every listed difference occurs (none is stale)", () => {
    const listed = [
      ...Object.keys({ ...POD_DIFFERS, ...TEMPLATE_DIFFERS, ...ENDPOINT_DIFFERS }),
      ...Object.keys(ENV_CONTROL_ONLY).map((k) => `env.${k}`),
      ...Object.keys(ENV_DEPLOY_ONLY).map((k) => `env.${k}`),
      ...Object.keys(ENV_VALUE_DIFFERS).map((k) => `env.${k}=`),
    ];
    expect(listed.filter((d) => !used.has(d))).toEqual([]);
  });
});

// ---------------------------------------------------------------- every path builds a Launch that passes the schema
describe("every fv-control Runpod GPU launch is a schema-valid Launch, stored specs included", () => {
  it("every cluster template and every pool preset (as a standalone pod, on each weights source it allows)", () => {
    let n = 0;
    const specs: ClusterSpec[] = Object.keys(TEMPLATES).map((t) => ({ ...defaultSpec("c", t), control_plane: "direct" as const }));
    for (const p of POOL_PRESETS)
      for (const x of [{}, { weights_source: "r2" }, { weights_source: "r2", data_centers: ["US-TX-3", "any"] }, { weights_source: "hub", weights_download_approved: true }, { weights_source: "none" }, { upsampler_local: true }]) {
        try {
          specs.push(standaloneSpec({ name: "solo", preset: p.id, ...x } as any).spec);
        } catch {
          /* a combination the launch form refuses */
        }
      }
    for (const spec of specs)
      for (const pool of spec.pools) {
        const env = workerSystemEnv({ spec, state: { images: {}, workers: {} }, secrets: SECRETS, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", backstop: true }, pool, IMAGE);
        for (const pl of workerPlacements(spec, pool)) {
          const body = workerCreatePayload("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec));
          if (pool.compute === "CPU") {
            expect(() => workerLaunch("fvc-x", IMAGE, pool, pl, env)).toThrow(/CPU pod is not a Launch/);
            expect(body).toMatchObject({ computeType: "CPU" });
            continue;
          }
          const l = workerLaunch("fvc-x", IMAGE, pool, pl, env, clusterGuards(spec));
          expect(LaunchZ.safeParse(wire(l)).success).toBe(true);
          expect(wire(body)).toStrictEqual(wire(runpodPodBody(l)));
          expect(l.guards).toEqual({ max_usd_per_hr: spec.max_gpu_dph, min_balance_usd: spec.balance_floor, cap_s: spec.cap_s, ready_timeout_s: 360, ready_cycles: 3 });
          expect(l.weights).toBe(env.FV_WEIGHTS_SOURCE === "r2" ? "r2" : env.FV_WEIGHTS_SOURCE === "hub" ? "hub" : l.placement.volume ? "volume" : "none");
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
        const l = endpointLaunch(spec, IMAGE, { FV_WEIGHTS_SOURCE: spec.weights_source ?? "volume" });
        expect(LaunchZ.safeParse(wire(l)).success).toBe(true);
        expect(l.guards).toEqual(endpointGuards(spec));
        const tu = wire(templateUpdatePayload(spec, IMAGE, { FV_WEIGHTS_SOURCE: spec.weights_source ?? "volume" }));
        expect(tu.env).toEqual(l.env);
        expect(tu.dockerEntrypoint).toEqual(l.entrypoint);
        const { name: _n, templateId: _t, networkVolumeId: _v, computeType: _c, ...fields } = wire(runpodEndpointBody(l, "t"));
        expect(wire(endpointUpdatePayload(spec))).toStrictEqual(fields);
        const v2 = wire(v2CreatePayload(spec, IMAGE, ["BLACKWELL_96"], {}));
        expect([v2.image, v2.disk, v2.timeout, v2.gpu.count]).toEqual([l.image, l.disk.container_gb, l.serverless!.execution_timeout_ms, l.gpu.count]);
        n++;
      }
    expect(n).toBeGreaterThan(20);
  });
  it("stored specs load and launch: a cluster stored with the gateway block and image variants, an endpoint row with a variant", () => {
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
    expect(body.env.FV_WORKER_TOML_B64).toBe(b64utf8(WORKER_CONFIGS["runpod.toml"]!));
    expect(body).toMatchObject({ networkVolumeId: "bz0ofkfx45", dataCenterIds: ["EU-RO-1"], dockerEntrypoint: ["bash", "-c"] });
    // An endpoint row stored before the one image (serverless_endpoints.spec): a variant and no config.
    const row = { spec: JSON.stringify({ ...normalizeEndpointSpec({ name: "old", compute: "GPU", config: "runpod-ltx.toml" }), config: undefined, compute: undefined, variant: "ltx" }) } as unknown as SlsRow;
    const s = specOf(row);
    expect([s.compute, s.config]).toEqual(["GPU", "runpod-ltx.toml"]);
    const l = endpointLaunch(s, IMAGE);
    expect(LaunchZ.safeParse(wire(l)).success).toBe(true);
    expect(wire(templateCreatePayload(s, IMAGE, "t")).env.FV_WORKER_TOML_B64).toBe(b64utf8(WORKER_CONFIGS["runpod-ltx.toml"]!));
    expect(wire(endpointCreatePayload(s, "tpl"))).toStrictEqual(wire(runpodEndpointBody(l, "tpl")));
    // A spec without `compute` reads as GPU, as before C3c: the GPU Launch, template category and endpoint fields.
    const { compute: _c, ...bare } = s;
    const nb = bare as EndpointSpec;
    expect(wire(templateCreatePayload(nb, IMAGE, "t")).category).toBe("NVIDIA");
    expect(wire(endpointCreatePayload(nb, "tpl"))).toStrictEqual(wire(runpodEndpointBody(endpointLaunch(nb, IMAGE), "tpl")));
    expect(wire(endpointUpdatePayload(nb))).toMatchObject({ gpuTypeIds: s.gpu_types, minCudaVersion: "13.0" });
  });
});
