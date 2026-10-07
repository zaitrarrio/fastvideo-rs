import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { validate } from "../../src/schemas";
import {
  assertFloor,
  checkLimits,
  createEndpoint,
  DEFAULT_SLS_POLICY,
  deleteEndpoint,
  getRow,
  invoke,
  jobStats,
  parseRunpodLine,
  recordBilling,
  resolveImage,
  scaleEndpoint,
  serverlessTick,
  updateEndpoint,
  type SlsRow,
} from "../../src/serverless/ops";
import {
  endpointCreatePayload,
  endpointView,
  invokeBody,
  lbPools,
  SLS_BOOT,
  templateCreatePayload,
  templateName,
  templateUpdatePayload,
  v2CreatePayload,
  v2TemplateBoot,
  workerEnv,
  endpointUpdatePayload,
} from "../../src/serverless/payloads";
import { checkEndpointSpec, defaultEndpointSpec, normalizeEndpointSpec, specDiff } from "../../src/serverless/spec";
import { CUDA_VERSIONS } from "../../src/enums";
import { d1 } from "./d1shim";

const IMG = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "a".repeat(64);
const KEY = "rpa_TESTKEY_0123456789";

describe("serverless spec", () => {
  it("cpu defaults: CPU queue workers, no volume, 0..1 workers, a 2 h delete backstop", () => {
    const s = normalizeEndpointSpec({ name: "fake" });
    expect(s).toMatchObject({ mode: "queue", variant: "cpu", compute: "CPU", cpu_flavors: ["cpu3c", "cpu5c"], vcpu: 2, network_volume: null, workers_min: 0, workers_max: 1, deadline_min: 120, deadline_action: "delete", image: { channel: "stable" } });
    expect(s.gpu_types).toBeUndefined();
    expect(s.data_centers).toBeUndefined();
  });
  it("GPU defaults: the EU volume, its data center, RTX PRO 6000 Server, no CUDA filter", () => {
    const s = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo" });
    expect(s).toMatchObject({ compute: "GPU", network_volume: "jg48s6o1w0", data_centers: ["EUR-IS-1"], gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], gpu_count: 1, allowed_cuda: [] });
  });
  it("GPU types keep their priority order; a 5090 fallback is allowed", () => {
    const s = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090"] });
    expect(s.gpu_types).toEqual(["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090"]);
  });
  it("refuses the deleted US volume, a volume outside its data center, unknown fields and bad names", () => {
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", network_volume: "s2k01690bi" })).toThrow(/unknown or unavailable volume/);
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", data_centers: ["US-CA-2"] })).toThrow(/must run in EUR-IS-1/);
    expect(() => normalizeEndpointSpec({ name: "x", bogus: 1 })).toThrow();
    expect(() => normalizeEndpointSpec({ name: "Bad Name" })).toThrow(/name/);
  });
  it("workers, modes, compute and env", () => {
    expect(() => normalizeEndpointSpec({ name: "x", workers_min: 2, workers_max: 1 })).toThrow(/workers_min is above workers_max/);
    expect(() => normalizeEndpointSpec({ name: "x", mode: "lb" })).toThrow(/GPU only/);
    expect(() => normalizeEndpointSpec({ name: "x", variant: "cpu", compute: "GPU" })).toThrow(/cpu variant runs on CPU/);
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", compute: "CPU" })).toThrow(/CPU workers run the cpu variant/);
    expect(() => normalizeEndpointSpec({ name: "x", env: { FV_SERVE_MODE: "http" } })).toThrow(/set by fv-control/);
    expect(() => normalizeEndpointSpec({ name: "x", env: { FV_R2_SECRET_ACCESS_KEY: "s" } })).toThrow(/set by fv-control/);
    expect(() => normalizeEndpointSpec({ name: "x", image: { channel: "stable", sha: "abcdef0" } })).toThrow(/exactly one/);
    // A load balancer's default scaler is REQUEST_COUNT; asking for QUEUE_DELAY is refused.
    expect(normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb" })).toMatchObject({ scaler_type: "REQUEST_COUNT", scaler_value: 1 });
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb", scaler_type: "QUEUE_DELAY" })).toThrow(/REQUEST_COUNT/);
    expect(normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb", scaler_type: "REQUEST_COUNT", scaler_value: 1 }).mode).toBe("lb");
  });
  it("is the same schema /api/schemas serves (the cluster specs' framework)", () => {
    const s = normalizeEndpointSpec({ name: "fake" });
    expect(validate("serverless-endpoint", s).ok).toBe(true);
    const bad = validate("serverless-endpoint", { ...s, workers_max: 99 });
    expect(bad.ok).toBe(false);
    const c = checkEndpointSpec({ ...s, workers_max: 99 });
    expect(c.ok ? [] : c.issues.map((i) => i.path.join("."))).toEqual(["workers_max"]);
  });
  it("diff: template vs endpoint fields, scale-up, what needs a new endpoint", () => {
    const a = normalizeEndpointSpec({ name: "x" });
    expect(specDiff(a, { ...a, workers_max: 2 })).toEqual({ template: false, endpoint: true, recreate: [], scaleUp: true });
    expect(specDiff(a, { ...a, workers_max: 0 }).scaleUp).toBe(false);
    expect(specDiff(a, { ...a, image: { sha: "abcdef0" } }).template).toBe(true);
    expect(specDiff(a, { ...a, env: { A: "1" } }).template).toBe(true);
    expect(specDiff(a, { ...a, compute: "GPU" }).recreate).toEqual(["compute"]);
  });
});

describe("serverless payloads", () => {
  const cpu = normalizeEndpointSpec({ name: "fake", env: { FV_LOG: "x" } });
  const gpu = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo", workers_max: 2, flashboot: true, config: "/etc/fv/runpod.toml" });
  it("template: serverless, CPU category, secret references (never values), runpod-queue, the image's identity", () => {
    const t = templateCreatePayload(cpu, IMG, templateName(cpu, Date.UTC(2026, 9, 6, 21, 5, 7)));
    expect(t).toMatchObject({ name: "fvc-fake-261006210507", imageName: IMG, isServerless: true, category: "CPU", containerDiskInGb: 20, volumeInGb: 0 });
    expect(t.env).toMatchObject({ FV_SERVE_MODE: "runpod-queue", FV_AUTH_MODE: "trust-gateway", FV_IMAGE_REF: IMG, FV_IMAGE_DIGEST: "sha256:" + "a".repeat(64), FV_RELEASE_CHANNEL: "stable", FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}", FV_LOG: "x" });
    expect(t.env.FV_WEIGHTS).toBeUndefined();
    // No config: the image's own entrypoint (fv-entry) and baked FV_CONFIG.
    expect("dockerEntrypoint" in t).toBe(false);
  });
  it("GPU template with a config: the boot links the weights and runs fv-serve with it", () => {
    const t = templateCreatePayload(gpu, IMG, "n");
    expect(t.category).toBe("NVIDIA");
    expect(t.env).toMatchObject({ FV_WEIGHTS: "/runpod-volume/weights", FV_CONFIG: "/etc/fv/runpod.toml" });
    expect(t.dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(SLS_BOOT).toContain("ln -s /runpod-volume/weights /workspace/weights");
    const inline = workerEnv(normalizeEndpointSpec({ name: "i", config_toml: "[engine]\nbackend = \"fake\"\n" }), IMG);
    expect(atob(inline.FV_WORKER_TOML_B64!)).toContain('backend = "fake"');
    // The update drops create-only fields and resets the entrypoint when no config is named.
    const u = templateUpdatePayload(cpu, IMG) as any;
    expect(u.name).toBeUndefined();
    expect(u.isServerless).toBeUndefined();
    expect(u.dockerEntrypoint).toEqual([]);
  });
  it("queue endpoint: CPU flavors / GPU types in order, volume, data centers, scaler, timeouts", () => {
    expect(endpointCreatePayload(cpu, "tpl1")).toEqual({
      name: "fvc-fake", templateId: "tpl1", computeType: "CPU", cpuFlavorIds: ["cpu3c", "cpu5c"], vcpuCount: 2,
      workersMin: 0, workersMax: 1, idleTimeout: 5, flashboot: false, executionTimeoutMs: 1_800_000, scalerType: "QUEUE_DELAY", scalerValue: 4,
    });
    expect(endpointCreatePayload(gpu, "tpl2")).toEqual({
      name: "fvc-h3", templateId: "tpl2", computeType: "GPU", gpuTypeIds: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], gpuCount: 1,
      networkVolumeId: "jg48s6o1w0", dataCenterIds: ["EUR-IS-1"], workersMin: 0, workersMax: 2, idleTimeout: 5, flashboot: true, executionTimeoutMs: 1_800_000, scalerType: "QUEUE_DELAY", scalerValue: 4,
    });
  });
  it("CUDA: no filter on create by default; an update clears an old filter; an explicit filter is kept", () => {
    // "13.0" alone hid the EUR-IS-1 RTX PRO 6000 hosts, so workers never started (staging h3-max2, 2026-10-07).
    expect((endpointCreatePayload(gpu, "t") as any).allowedCudaVersions).toBeUndefined();
    expect((endpointUpdatePayload(gpu) as any).allowedCudaVersions).toEqual([...CUDA_VERSIONS]);
    const pinned = normalizeEndpointSpec({ name: "pin", variant: "h3-max", allowed_cuda: ["12.8"] } as any);
    expect((endpointCreatePayload(pinned, "t") as any).allowedCudaVersions).toEqual(["12.8"]);
    expect((endpointUpdatePayload(pinned) as any).allowedCudaVersions).toEqual(["12.8"]);
  });
  it("load balancer: REST v2 shape, catalog pools in the spec's order, HTTP mode", () => {
    const lb = normalizeEndpointSpec({ name: "lb", variant: "wan", mode: "lb", scaler_type: "REQUEST_COUNT", scaler_value: 1, gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090"], config: "/etc/fv/runpod-wan.toml" });
    const pools = lbPools(lb.gpu_types!, [{ id: "NVIDIA GeForce RTX 5090", pool: "ADA_32_PRO" }, { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", pool: "BLACKWELL_96" }]);
    expect(pools).toEqual(["BLACKWELL_96", "ADA_32_PRO"]);
    const p = v2CreatePayload(lb, IMG, pools) as any;
    expect(p).toMatchObject({ name: "fvc-lb", type: "LOAD_BALANCER", image: IMG, args: "--config /etc/fv/runpod-wan.toml", ports: ["8000/http"], networkVolumes: ["jg48s6o1w0"], dataCenterIds: ["EUR-IS-1"], flashboot: "OFF", workers: { min: 0, max: 1, idleTimeout: 5 }, scaling: { type: "REQUEST_COUNT", requestCount: 1 } });
    expect(p.env).toMatchObject({ FV_SERVE_MODE: "http", PORT: "8000", PORT_HEALTH: "8000", FV_WORKERS_MAX: "1" });
  });
  it("CPU queue: REST v2 with the cpu flavors (REST v1 ignores computeType CPU); a config goes into its template", () => {
    const p = v2CreatePayload(cpu, IMG) as any;
    expect(p).toMatchObject({ name: "fvc-fake", type: "QUEUE", image: IMG, disk: 20, cpu: [{ id: "cpu3c", vcpuCount: 2 }, { id: "cpu5c", vcpuCount: 2 }], workers: { min: 0, max: 1, idleTimeout: 5 }, scaling: { type: "QUEUE_DELAY", queueDelay: 4 }, flashboot: "OFF", timeout: 1_800_000 });
    expect(p.gpu).toBeUndefined();
    expect(p.ports).toBeUndefined();
    expect(p.env.FV_SERVE_MODE).toBe("runpod-queue");
    expect(v2TemplateBoot(cpu)).toBeNull();
    expect(v2TemplateBoot(normalizeEndpointSpec({ name: "c", config: "/etc/fv/runpod-fake.toml" }))!.dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
  });
  it("update payload: only EndpointUpdateInput keys (Runpod refuses extra keys); CPU flavors are not updatable", () => {
    const UPDATE_KEYS = ["allowedCudaVersions", "cpuFlavorIds", "dataCenterIds", "executionTimeoutMs", "flashboot", "gpuCount", "gpuTypeIds", "idleTimeout", "minCudaVersion", "name", "networkVolumeId", "networkVolumeIds", "scalerType", "scalerValue", "templateId", "vcpuCount", "workersMax", "workersMin"];
    for (const sp of [cpu, gpu]) for (const k of Object.keys(endpointUpdatePayload(sp))) expect(UPDATE_KEYS).toContain(k);
    expect(endpointUpdatePayload(cpu)).toEqual({ workersMin: 0, workersMax: 1, idleTimeout: 5, flashboot: false, executionTimeoutMs: 1_800_000, scalerType: "QUEUE_DELAY", scalerValue: 4 });
    expect(specDiff(cpu, { ...cpu, vcpu: 4 }).recreate).toEqual(["vcpu"]);
  });
  it("invoke body and the endpoint view (never the template's env)", () => {
    expect(invokeBody(undefined, 60)).toEqual({ input: { kind: "info" }, policy: { executionTimeout: 60000 } });
    const v = endpointView({ id: "e", name: "fvc-x", template: { id: "t", imageName: IMG, env: { SECRET: "v" } }, workers: [{ id: "w1", desiredStatus: "RUNNING", costPerHr: 0.1, env: { S: "x" } }] }) as any;
    expect(JSON.stringify(v)).not.toContain("SECRET");
    expect(JSON.stringify(v)).not.toContain('"S"');
    expect(v.image).toBe(IMG);
    expect(v.workers[0]).toMatchObject({ id: "w1", costPerHr: 0.1 });
  });
  it("Runpod log lines", () => {
    expect(parseRunpodLine("2026-10-06T21:50:01.123456Z FV-SERVE READY models=fake-wan", 0)).toEqual({ ts: Date.parse("2026-10-06T21:50:01.123Z"), msg: "FV-SERVE READY models=fake-wan", level: "info" });
    expect(parseRunpodLine("no time ERROR here", 5)).toEqual({ ts: 5, msg: "no time ERROR here", level: "error" });
    expect(parseRunpodLine("2026-10-06T22:07:24.733Z \x1b[2m2026-10-06T22:07:24Z\x1b[0m \x1b[33m WARN\x1b[0m x", 0).msg).toBe("2026-10-06T22:07:24Z  WARN x");
  });
  it("job stats: cold start from jobs submitted with no worker up", () => {
    const s = jobStats([
      { cold: 1, delay_ms: 30000, exec_ms: 700, wall_ms: 31000, status: "COMPLETED" },
      { cold: 0, delay_ms: 100, exec_ms: 600, wall_ms: 900, status: "COMPLETED" },
      { cold: 0, delay_ms: null, exec_ms: null, wall_ms: null, status: "FAILED" },
    ]);
    expect(s).toEqual({ jobs: 3, completed: 2, failed: 1, cold_start_ms: 30000, warm_delay_ms: 100, exec_ms: 600 });
  });
});

// ---------------------------------------------------------------- operations against a Runpod mock
type Call = { method: string; url: string; body: any };
function mockRunpod(o: { balance?: number } = {}) {
  const m = {
    balance: o.balance ?? 30,
    calls: [] as Call[],
    endpoints: new Map<string, any>(),
    templates: new Map<string, any>(),
    foreign: { id: "strobe0000001", name: "strobe", type: "QB", pods: [] as any[] },
    deleteFails: 0,
    jobs: 0,
    billing: [] as any[],
    n: 0,
  };
  const res = (code: number, body: unknown, headers: Record<string, string> = {}) => new Response(body === undefined ? null : JSON.stringify(body), { status: code, headers: { "content-type": "application/json", ...headers } });
  const fetchMock = async (input: any, init: any = {}) => {
    const url = String(input);
    const method = (init.method || "GET").toUpperCase();
    const body = init.body ? JSON.parse(init.body) : undefined;
    m.calls.push({ method, url, body });
    const u = new URL(url);
    if (u.host === "ghcr.io") {
      if (u.pathname === "/token") return res(200, { token: "t" });
      if (u.pathname.includes("/manifests/cpu-stable")) return res(200, undefined, { "docker-content-digest": "sha256:" + "b".repeat(64) });
      return res(404, {});
    }
    if (u.host === "api.runpod.io" && u.pathname === "/graphql") {
      return res(200, { data: { myself: { clientBalance: m.balance, endpoints: [m.foreign, ...[...m.endpoints.values()].map((e) => ({ id: e.id, name: e.name, type: "QB", pods: e.pods || [] }))] } } });
    }
    if (u.host === "rest.runpod.io") {
      const p = u.pathname.replace("/v1", "");
      if (p === "/templates" && method === "POST") {
        const id = `tpl${++m.n}`;
        m.templates.set(id, { id, ...body });
        return res(200, { id, ...body });
      }
      if (p === "/endpoints" && method === "POST") {
        const id = `ep${++m.n}aaaaaaaaaaa`.slice(0, 14);
        m.endpoints.set(id, { id, ...body, pods: [] });
        return res(200, { id, name: body.name });
      }
      let mm = /^\/endpoints\/([^/]+)$/.exec(p);
      if (mm) {
        const e = m.endpoints.get(mm[1]!) || (mm[1] === m.foreign.id ? m.foreign : null);
        if (!e) return res(404, { error: "not found" });
        if (method === "GET") return res(200, { ...e, workers: (e.pods || []).map((x: any) => ({ ...x, env: { SECRET: "x" } })), template: { id: e.templateId, env: { SECRET: "x" } } });
        if (method === "PATCH") {
          Object.assign(e, body);
          return res(200, e);
        }
        if (method === "DELETE") {
          if (m.deleteFails > 0) {
            m.deleteFails--;
            return res(400, { error: "endpoint has running workers" });
          }
          if (e.v2) m.templates.delete(e.templateId);
          m.endpoints.delete(mm[1]!);
          return res(204, undefined);
        }
      }
      mm = /^\/templates\/([^/]+)$/.exec(p);
      if (mm) {
        if (method === "PATCH") {
          Object.assign(m.templates.get(mm[1]!) || {}, body);
          return res(200, {});
        }
        if (method === "DELETE") {
          if (!m.templates.has(mm[1]!)) return res(400, { error: "delete template: get template: template not found" });
          m.templates.delete(mm[1]!);
          return res(204, undefined);
        }
      }
      if (p === "/billing/endpoints") return res(200, m.billing);
    }
    if (u.host === "api.runpod.io" && u.pathname === "/v2/serverless" && method === "POST") {
      const id = `ep${++m.n}vvvvvvvvvvv`.slice(0, 14);
      const tid = `tpl${++m.n}`;
      m.templates.set(tid, { id: tid, v2: true, name: `${body.name}-template`, imageName: body.image, env: body.env });
      m.endpoints.set(id, { id, name: body.name, templateId: tid, v2: body, workersMax: body.workers.max, flashboot: true, pods: [] });
      return res(200, { id, name: body.name });
    }
    if (u.host === "api.runpod.ai") {
      const mm = /^\/v2\/([^/]+)\/(health|runsync|run|status\/.+)$/.exec(u.pathname);
      if (mm) {
        if (mm[2] === "health") return res(200, { jobs: { completed: m.jobs, failed: 0, inProgress: 0, inQueue: 0, retried: 0 }, workers: { idle: 0, initializing: 0, ready: 0, running: 0, throttled: 0, unhealthy: 0 } });
        if (mm[2] === "runsync") {
          m.jobs++;
          return res(200, { id: "job-1", status: "COMPLETED", delayTime: 31000, executionTime: 700, workerId: "w1", output: { engine: "fake", token: KEY } });
        }
      }
    }
    return res(500, { error: `mock: no route ${method} ${url}` });
  };
  return { m, fetchMock };
}

describe("serverless operations", () => {
  let env: any;
  let mock: ReturnType<typeof mockRunpod>;
  const who = { actor: "token:test" };
  beforeEach(() => {
    mock = mockRunpod();
    vi.stubGlobal("fetch", mock.fetchMock);
    env = { DB: d1(), RUNPOD_API_KEY: KEY, BALANCE_FLOOR: "8" };
  });
  afterEach(() => vi.unstubAllGlobals());

  it("resolves the image like clusters do (<variant>-<channel> to a digest)", async () => {
    expect(await resolveImage(env, normalizeEndpointSpec({ name: "fake" }))).toBe("ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "b".repeat(64));
  });

  it("create (CPU): REST v2, then the reconcile PATCH; recorded in D1 and audited; a second with the same name is refused", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    expect(row).toMatchObject({ name: "fake", status: "active", own_template: 1, mode: "queue", image: IMG });
    expect(row.endpoint_id).toMatch(/^ep1/);
    expect(row.template_id).toBe("tpl2");
    expect(row.deadline! - row.created_at).toBe(120 * 60_000);
    const writes = mock.m.calls.filter((c) => c.method !== "GET" && !c.url.includes("graphql")).map((c) => `${c.method} ${new URL(c.url).host}${new URL(c.url).pathname}`);
    expect(writes).toEqual(["POST api.runpod.io/v2/serverless", `PATCH rest.runpod.io/v1/endpoints/${row.endpoint_id}`]);
    // The reconcile PATCH set flashboot back to false (Runpod's create had dropped it).
    expect(mock.m.endpoints.get(row.endpoint_id!)!.flashboot).toBe(false);
    const a = await env.DB.prepare("SELECT action, ok FROM audit WHERE action = 'serverless.create'").all();
    expect(a.results).toEqual([{ action: "serverless.create", ok: 1 }]);
    await expect(createEndpoint(env, who, { name: "fake" }, { image: IMG })).rejects.toThrow(/exists/);
  });

  it("create (GPU queue): a template, then the endpoint on it (REST v1), then the reconcile PATCH", async () => {
    const row = await createEndpoint(env, who, { name: "h3", variant: "h3-turbo" }, { image: IMG });
    expect(row).toMatchObject({ status: "active", template_id: "tpl1", own_template: 1 });
    const writes = mock.m.calls.filter((c) => c.method !== "GET" && !c.url.includes("graphql"));
    expect(writes.map((c) => `${c.method} ${new URL(c.url).pathname}`)).toEqual(["POST /v1/templates", "POST /v1/endpoints", `PATCH /v1/endpoints/${row.endpoint_id}`]);
    expect(writes[1]!.body).toMatchObject({ templateId: "tpl1", networkVolumeId: "jg48s6o1w0", dataCenterIds: ["EUR-IS-1"] });
    expect(writes[2]!.body.computeType).toBeUndefined();
  });

  it("create refuses below the floor + margin, and over the account limits", async () => {
    mock.m.balance = 9.5; // floor 8 + margin 2 = 10
    await expect(createEndpoint(env, who, { name: "fake" }, { image: IMG })).rejects.toMatchObject({ status: 402 });
    expect(mock.m.calls.some((c) => c.method === "POST" && c.url.includes("rest.runpod.io"))).toBe(false);
    mock.m.balance = 30;
    await expect(assertFloor(env, "x")).resolves.toBe(30);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [{ id: "a", workers_max: 4 }, { id: "b", workers_max: 4 }], { workers_max: 1 })).toThrow(/max_workers/);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [1, 2, 3, 4].map((i) => ({ id: `e${i}`, workers_max: 1 })), { workers_max: 1 })).toThrow(/max_endpoints/);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [{ id: "a", workers_max: 4 }], { id: "a", workers_max: 8 })).not.toThrow();
  });

  it("a failed endpoint create deletes the template it made and marks the row failed", async () => {
    const orig = mock.fetchMock;
    vi.stubGlobal("fetch", async (u: any, i: any = {}) => (String(u).endsWith("/v1/endpoints") && i.method === "POST" ? new Response(JSON.stringify({ error: "no capacity" }), { status: 400 }) : orig(u, i)));
    await expect(createEndpoint(env, who, { name: "h3", variant: "h3-turbo" }, { image: IMG })).rejects.toThrow(/no capacity/);
    expect(mock.m.templates.size).toBe(0);
    const r = await env.DB.prepare("SELECT status, deleted_at FROM serverless_endpoints").first();
    expect(r.status).toBe("failed");
    expect(r.deleted_at).toBeTruthy();
  });

  it("only endpoints fv-control created: unknown ids and names are 404, a foreign name on Runpod is refused", async () => {
    await expect(getRow(env, "strobe0000001")).rejects.toMatchObject({ status: 404 });
    await expect(getRow(env, "strobe")).rejects.toMatchObject({ status: 404 });
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    // Someone renamed the Runpod endpoint: fv-control no longer touches it.
    mock.m.endpoints.get(row.endpoint_id!)!.name = "someone-else";
    await expect(scaleEndpoint(env, who, row, { workers_max: 0 })).rejects.toMatchObject({ status: 403 });
  });

  it("scale: down without a floor check, up with one; update patches the template for an image change", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    mock.m.balance = 5;
    const down = await scaleEndpoint(env, who, row, { workers_max: 0 });
    expect(down.status).toBe("scaled-down");
    expect(mock.m.endpoints.get(row.endpoint_id!)!.workersMax).toBe(0);
    await expect(scaleEndpoint(env, who, down, { workers_max: 1 })).rejects.toMatchObject({ status: 402 });
    mock.m.balance = 30;
    const up = await scaleEndpoint(env, who, down, { workers_max: 2 });
    expect(up.status).toBe("active");
    const IMG2 = IMG.replace(/a{64}/, "c".repeat(64));
    const u = await updateEndpoint(env, who, up, { ...JSON.parse(up.spec), image: { ref: IMG2 } }, { image: IMG2 });
    expect(u.image).toBe(IMG2);
    expect(mock.m.templates.get(up.template_id!)!.imageName).toBe(IMG2);
    await expect(updateEndpoint(env, who, u, { ...JSON.parse(u.spec), network_volume: "jg48s6o1w0" })).rejects.toThrow(/network_volume: cannot change in place/);
  });

  it("invoke: /runsync, timings, cold start flag, output scrubbed of secrets", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    const r = await invoke(env, who, row, {});
    expect(r).toMatchObject({ status: "COMPLETED", done: true, cold: true, delay_ms: 31000, exec_ms: 700, worker_id: "w1" });
    const call = mock.m.calls.find((c) => c.url.endsWith("/runsync"))!;
    expect(call.body).toEqual({ input: { kind: "info" }, policy: { executionTimeout: 1_800_000 } });
    const j = await env.DB.prepare("SELECT * FROM serverless_jobs").first();
    expect(j).toMatchObject({ status: "COMPLETED", cold: 1, delay_ms: 31000, exec_ms: 700, route: "runsync" });
    expect(j.output).not.toContain(KEY);
    expect(j.output).toContain("[redacted]");
  });

  it("delete: scale to 0, delete endpoint then template; a refused delete is retried by the tick", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    mock.m.deleteFails = 1;
    const r1 = await deleteEndpoint(env, who, row);
    expect(r1.status).toBe("deleting");
    expect(r1.last_error).toMatch(/running workers/);
    expect(mock.m.endpoints.get(row.endpoint_id!)!.workersMax).toBe(0);
    const t = await serverlessTick(env);
    expect(t.actions).toContain("fake: delete done");
    const r2 = await getRow(env, row.id);
    expect(r2.status).toBe("deleted");
    expect(mock.m.endpoints.size).toBe(0);
    expect(mock.m.templates.size).toBe(0);
    // The foreign endpoint was never touched.
    expect(mock.m.calls.some((c) => c.url.includes("strobe0000001") && c.method !== "GET")).toBe(false);
  });

  it("tick: the deadline backstop deletes (or scales to 0), the floor scales to 0, a vanished endpoint is gone", async () => {
    const a = await createEndpoint(env, who, { name: "a", deadline_action: "delete" }, { image: IMG });
    const b = await createEndpoint(env, who, { name: "b", deadline_action: "scale0" }, { image: IMG });
    const c = await createEndpoint(env, who, { name: "c" }, { image: IMG });
    await env.DB.prepare("UPDATE serverless_endpoints SET deadline = ? WHERE id IN (?, ?)").bind(Date.now() - 1000, a.id, b.id).run();
    mock.m.endpoints.delete(c.endpoint_id!);
    const t = await serverlessTick(env);
    expect(t.actions.sort()).toEqual(["a: deleted (deadline)", "b: scaled to 0 (deadline)", "c: gone"]);
    expect((await getRow(env, a.id)).status).toBe("deleted");
    expect((await getRow(env, b.id)).status).toBe("scaled-down");
    expect(mock.m.endpoints.get(b.endpoint_id!)!.workersMax).toBe(0);
    expect((await getRow(env, c.id)).status).toBe("gone");
    const al = await env.DB.prepare("SELECT kind, key FROM alerts WHERE resolved_at IS NULL ORDER BY key").all();
    expect(al.results.map((x: any) => x.key).sort()).toEqual([`serverless_deadline:${a.id}`, `serverless_deadline:${b.id}`].sort());
    // Below the floor: scaled to 0.
    const d = await createEndpoint(env, who, { name: "d" }, { image: IMG });
    mock.m.balance = 7;
    const t2 = await serverlessTick(env);
    expect(t2.actions).toContain("d: scaled to 0 (balance floor)");
    expect((await getRow(env, d.id)).status).toBe("scaled-down");
    const au = await env.DB.prepare("SELECT actor FROM audit WHERE action = 'serverless.scale' AND target = 'd'").first();
    expect(au.actor).toBe("policy:balance_floor");
  });

  it("billing: Runpod's per-endpoint spend into the cost ledger (idempotent), only for fv-control's endpoints", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    const day = new Date().toISOString().slice(0, 10);
    mock.m.billing = [
      { endpointId: row.endpoint_id!, time: `${day} 00:00:00`, amount: 0.0123, timeBilledMs: 90_000 },
      { endpointId: "strobe0000001", time: `${day} 00:00:00`, amount: 5, timeBilledMs: 1 },
    ];
    const rows = (await env.DB.prepare("SELECT * FROM serverless_endpoints").all()).results as SlsRow[];
    expect(await recordBilling(env, rows)).toBe(1);
    expect(await recordBilling(env, rows)).toBe(1);
    const c = await env.DB.prepare("SELECT * FROM cost_daily").all();
    expect(c.results).toEqual([{ day, pod_id: `sls:${row.endpoint_id}`, cluster_id: null, owner: "serverless:fake", usd: 0.0123, minutes: 2, idle_minutes: 0 }]);
    expect((await getRow(env, row.id)).cost_usd).toBeCloseTo(0.0123);
  });

  it("the tick makes no Runpod call when fv-control has no endpoint", async () => {
    const t = await serverlessTick(env);
    expect(t).toEqual({ endpoints: 0, actions: [] });
    expect(mock.m.calls).toEqual([]);
  });
});

describe("defaults", () => {
  it("defaultEndpointSpec validates for every variant", () => {
    for (const v of ["cpu", "h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan"]) expect(checkEndpointSpec(defaultEndpointSpec("x", v)).ok).toBe(true);
  });
});
