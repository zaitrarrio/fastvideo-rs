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
  specOf,
  updateEndpoint,
  type SlsRow,
} from "../../src/serverless/ops";
import { endpointCreatePayload, endpointView, invokeBody, SLS_BOOT, templateCreatePayload, templateName, templateUpdatePayload, workerEnv, endpointUpdatePayload } from "../../src/serverless/payloads";
import { gpuPools } from "../../src/launch";
import { checkEndpointSpec, defaultEndpointSpec, normalizeEndpointSpec, specDiff } from "../../src/serverless/spec";
import { CUDA_VERSIONS } from "../../src/enums";
import { d1 } from "./d1shim";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { mergeSpec, SLS_PRESETS } from "../../src/serverless/presets";

const IMG = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "a".repeat(64);
const KEY = "rpa_TESTKEY_0123456789";

describe("serverless spec", () => {
  it("cpu defaults: CPU queue workers, no volume, 0..1 workers, a 2 h delete backstop", () => {
    const s = normalizeEndpointSpec({ name: "fake" });
    expect(s).toMatchObject({ mode: "queue", compute: "CPU", config: "runpod-fake.toml", cpu_flavors: ["cpu3c", "cpu5c"], vcpu: 2, network_volume: null, workers_min: 0, workers_max: 1, deadline_min: 120, deadline_action: "delete", image: { channel: "stable" } });
    expect(s.gpu_types).toBeUndefined();
    expect(s.data_centers).toBeUndefined();
  });
  it("GPU defaults: the EU volume, its data center, RTX PRO 6000 Server, no CUDA filter", () => {
    const s = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo" });
    expect(s).toMatchObject({ compute: "GPU", network_volume: "bz0ofkfx45", data_centers: ["EU-RO-1"], gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], gpu_count: 1, allowed_cuda: [] });
  });
  it("GPU types keep their priority order; a 5090 fallback is allowed", () => {
    const s = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090"] });
    expect(s.gpu_types).toEqual(["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090"]);
  });
  it("refuses the deleted US volume, a volume outside its data center, unknown fields and bad names", () => {
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", network_volume: "s2k01690bi" })).toThrow(/unknown or unavailable volume/);
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", data_centers: ["US-CA-2"] })).toThrow(/must run in EU-RO-1/);
    expect(() => normalizeEndpointSpec({ name: "x", bogus: 1 })).toThrow();
    expect(() => normalizeEndpointSpec({ name: "Bad Name" })).toThrow(/name/);
  });
  it("workers, modes, compute and env", () => {
    expect(() => normalizeEndpointSpec({ name: "x", workers_min: 2, workers_max: 1 })).toThrow(/workers_min is above workers_max/);
    expect(() => normalizeEndpointSpec({ name: "x", mode: "lb" })).toThrow(/GPU only/);
    // One image (2026-10-25): the compute picks it; a custom GPU endpoint names its config (the images bake none).
    expect(() => normalizeEndpointSpec({ name: "x", compute: "GPU" })).toThrow(/a custom endpoint names its worker config/);
    expect(normalizeEndpointSpec({ name: "x", compute: "GPU", config: "runpod-ltx.toml" })).toMatchObject({ compute: "GPU", config: "runpod-ltx.toml", network_volume: "bz0ofkfx45" });
    expect(() => normalizeEndpointSpec({ name: "x", compute: "GPU", config: "/etc/fv/nope.toml" })).toThrow(/not a worker config fv-control carries/);
    expect(() => normalizeEndpointSpec({ name: "x", env: { FV_SERVE_MODE: "http" } })).toThrow(/set by fv-control/);
    expect(() => normalizeEndpointSpec({ name: "x", env: { FV_R2_SECRET_ACCESS_KEY: "s" } })).toThrow(/set by fv-control/);
    expect(() => normalizeEndpointSpec({ name: "x", image: { channel: "stable", sha: "abcdef0" } })).toThrow(/exactly one/);
    // A load balancer's default scaler is REQUEST_COUNT; asking for QUEUE_DELAY is refused.
    expect(normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb" })).toMatchObject({ scaler_type: "REQUEST_COUNT", scaler_value: 1 });
    expect(() => normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb", scaler_type: "QUEUE_DELAY" })).toThrow(/REQUEST_COUNT/);
    expect(normalizeEndpointSpec({ name: "x", variant: "h3-turbo", mode: "lb", scaler_type: "REQUEST_COUNT", scaler_value: 1 }).mode).toBe("lb");
  });
  it("is the same schema /api/v1/schemas serves (the cluster specs' framework)", () => {
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

/** Runpod's GPU catalog as `GET /catalog/gpus` lists it (type id → serverless pool): what the v2 endpoint's `gpu` block is
 * built from (launch.ts gpuPools). */
const CATALOG = [
  { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", pool: "BLACKWELL_96" },
  { id: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", pool: "BLACKWELL_96" },
  { id: "NVIDIA GeForce RTX 5090", pool: "ADA_32_PRO" },
  { id: "NVIDIA H100 80GB HBM3", pool: "HOPPER_80" },
  { id: "NVIDIA A40", pool: null },
];
const RTX6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition";

describe("serverless payloads", () => {
  const cpu = normalizeEndpointSpec({ name: "fake", env: { FV_LOG: "x" } });
  const gpu = normalizeEndpointSpec({ name: "h3", variant: "h3-turbo", workers_max: 2, flashboot: true, config: "/etc/fv/runpod.toml" });
  it("template: serverless, CPU category, secret references (never values), runpod-queue, the image's identity", () => {
    const t = templateCreatePayload(cpu, IMG, templateName(cpu, Date.UTC(2026, 9, 6, 21, 5, 7)));
    // REST v2 names: image, serverless, disk; no pod volume means no `mounts`; v1's readme has no v2 field.
    expect(t).toMatchObject({ name: "fvc-fake-261006210507", image: IMG, serverless: true, category: "CPU", disk: 20 });
    expect(t).not.toHaveProperty("mounts");
    expect(t).not.toHaveProperty("readme");
    expect(t).not.toHaveProperty("ports");
    expect(t.env).toMatchObject({ FV_SERVE_MODE: "runpod-queue", FV_AUTH_MODE: "trust-gateway", FV_IMAGE_REF: IMG, FV_IMAGE_DIGEST: "sha256:" + "a".repeat(64), FV_RELEASE_CHANNEL: "stable", FV_R2_SECRET_ACCESS_KEY: "{{ RUNPOD_SECRET_fv_r2_secret_access_key }}", FV_LOG: "x" });
    expect(t.env.FV_WEIGHTS).toBeUndefined();
    // Every endpoint sends its config (the images bake none): the CPU default's fake config, inline, and its boot.
    expect(atob(t.env.FV_WORKER_TOML_B64!)).toContain('backend = "fake"');
    expect(t.env.FV_CONFIG).toBeUndefined();
    expect(t.entrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(t.cmd).toEqual([]);
  });
  it("GPU template with a config: the boot links the weights and runs fv-serve with it", () => {
    const t = templateCreatePayload(gpu, IMG, "n");
    expect(t.category).toBe("NVIDIA");
    expect(t).toMatchObject({ image: IMG, serverless: true, disk: gpu.container_disk_gb });
    expect(t.env).toMatchObject({ FV_WEIGHTS: "/runpod-volume/weights" });
    // The named config's text rides inline (the image bakes none); no FV_CONFIG.
    expect(Buffer.from(t.env.FV_WORKER_TOML_B64!, "base64").toString("utf8")).toBe(WORKER_CONFIGS["runpod.toml"]);
    expect(t.env.FV_CONFIG).toBeUndefined();
    expect(t.entrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(SLS_BOOT).toContain("ln -s /runpod-volume/weights /workspace/weights");
    expect(SLS_BOOT).toContain("FAILED: no worker config");
    expect(SLS_BOOT).not.toContain("FV_CONFIG");
    const inline = workerEnv(normalizeEndpointSpec({ name: "i", config_toml: "[engine]\nbackend = \"fake\"\n" }), IMG);
    expect(atob(inline.FV_WORKER_TOML_B64!)).toContain('backend = "fake"');
    // The update drops create-only fields and keeps the boot of its config.
    const u = templateUpdatePayload(cpu, IMG) as any;
    expect(u.name).toBeUndefined();
    expect(u.serverless).toBeUndefined();
    expect(u.category).toBeUndefined();
    expect(u.entrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(u.cmd).toEqual([]);
  });
  it("the GPU block: catalog pools in the spec's order, the pools' other types excluded (exactly the types asked for)", () => {
    expect(gpuPools([RTX6000, "NVIDIA GeForce RTX 5090"], CATALOG)).toEqual({ pools: ["BLACKWELL_96", "ADA_32_PRO"], excludedTypes: ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"] });
    expect(gpuPools(["NVIDIA GeForce RTX 5090", RTX6000, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], CATALOG)).toEqual({ pools: ["ADA_32_PRO", "BLACKWELL_96"] });
    // A name that is itself a pool selects the pool whole (v1's gpuIds such as ADA_24 were pool names).
    expect(gpuPools(["BLACKWELL_96"], CATALOG)).toEqual({ pools: ["BLACKWELL_96"] });
    // A type in no pool, or not in the catalog, is refused by name, never widened or dropped (ops.ts endpointGpuPools: 409).
    expect(() => gpuPools(["NVIDIA A40"], CATALOG)).toThrow(/NVIDIA A40: not in any serverless pool/);
    expect(() => gpuPools([RTX6000, "nope"], CATALOG)).toThrow(/nope: neither a GPU type nor a pool/);
  });
  it("queue endpoint (REST v2 POST /serverless on a template): CPU flavors / GPU pools in order, volume, data centers, scaler, timeouts", () => {
    expect(endpointCreatePayload(cpu, "tpl1")).toEqual({
      name: "fvc-fake", templateId: "tpl1", type: "QUEUE", cpu: [{ id: "cpu3c", vcpuCount: 2 }, { id: "cpu5c", vcpuCount: 2 }],
      workers: { min: 0, max: 1, idleTimeout: 5 }, scaling: { type: "QUEUE_DELAY", queueDelay: 4 }, flashboot: "OFF", timeout: 1_800_000,
    });
    expect(endpointCreatePayload(gpu, "tpl2", gpuPools(gpu.gpu_types!, CATALOG))).toEqual({
      name: "fvc-h3", templateId: "tpl2", type: "QUEUE",
      gpu: { pools: ["BLACKWELL_96"], excludedTypes: ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], count: 1, minCudaVersion: "13.0" },
      networkVolumes: ["bz0ofkfx45"], dataCenterIds: ["EU-RO-1"], workers: { min: 0, max: 2, idleTimeout: 5 }, scaling: { type: "QUEUE_DELAY", queueDelay: 4 }, flashboot: "FLASHBOOT", timeout: 1_800_000,
    });
    // Without the catalog (a PATCH that leaves the pools alone): the GPU block carries only the count and the CUDA filter.
    expect((endpointUpdatePayload(gpu) as any).gpu).toEqual({ count: 1, minCudaVersion: "13.0" });
    // REQUEST_COUNT: v2 rejects workers.idleTimeout there.
    const rc = normalizeEndpointSpec({ name: "rc", scaler_type: "REQUEST_COUNT", scaler_value: 2 });
    expect(endpointCreatePayload(rc, "t")).toMatchObject({ workers: { min: 0, max: 1 }, scaling: { type: "REQUEST_COUNT", requestCount: 2 } });
    expect((endpointCreatePayload(rc, "t") as any).workers).not.toHaveProperty("idleTimeout");
  });
  it("CUDA: at least 13.0 (gpu.minCudaVersion), no exact list by default; an update clears an old list; an explicit list is kept", () => {
    // "13.0" alone hid the RTX PRO 6000 hosts (labelled 13.2), so workers never started (staging h3-max2, 2026-10-07;
    // 2026-10-14). Nor may the update send every REST version: that enum stops at 13.0 too.
    expect((endpointCreatePayload(gpu, "t") as any).gpu.allowedCudaVersions).toBeUndefined();
    expect((endpointCreatePayload(gpu, "t") as any).gpu.minCudaVersion).toBe("13.0");
    expect((endpointUpdatePayload(gpu) as any).gpu.allowedCudaVersions).toBeUndefined();
    expect((endpointUpdatePayload(gpu) as any).gpu.minCudaVersion).toBe("13.0");
    expect((endpointUpdatePayload(gpu, { ...gpu, allowed_cuda: [...CUDA_VERSIONS] } as any) as any).gpu.allowedCudaVersions).toEqual([]);
    const pinned = normalizeEndpointSpec({ name: "pin", variant: "h3-max", allowed_cuda: ["12.8"] } as any);
    expect((endpointCreatePayload(pinned, "t") as any).gpu.allowedCudaVersions).toEqual(["12.8"]);
    expect((endpointUpdatePayload(pinned) as any).gpu.allowedCudaVersions).toEqual(["12.8"]);
  });
  it("load balancer: type LOAD_BALANCER on the same template shape, the port on the template, HTTP mode", () => {
    const lb = normalizeEndpointSpec({ name: "lb", variant: "wan", mode: "lb", scaler_type: "REQUEST_COUNT", scaler_value: 1, gpu_types: [RTX6000, "NVIDIA GeForce RTX 5090"], config: "/etc/fv/runpod-wan.toml" });
    const t = templateCreatePayload(lb, IMG, "fvc-lb-x") as any;
    expect(t).toMatchObject({ image: IMG, serverless: true, ports: ["8000/http"], entrypoint: ["/bin/sh", "-c", SLS_BOOT], cmd: [] });
    expect(Buffer.from(t.env.FV_WORKER_TOML_B64, "base64").toString("utf8")).toBe(WORKER_CONFIGS["runpod-wan.toml"]);
    expect(t.env).toMatchObject({ FV_SERVE_MODE: "http", PORT: "8000", PORT_HEALTH: "8000", FV_WORKERS_MAX: "1" });
    const p = endpointCreatePayload(lb, "tplx", gpuPools(lb.gpu_types!, CATALOG)) as any;
    expect(p).toEqual({
      name: "fvc-lb", templateId: "tplx", type: "LOAD_BALANCER",
      gpu: { pools: ["BLACKWELL_96", "ADA_32_PRO"], excludedTypes: ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], count: 1, minCudaVersion: "13.0" },
      networkVolumes: ["bz0ofkfx45"], dataCenterIds: ["EU-RO-1"], workers: { min: 0, max: 1 }, scaling: { type: "REQUEST_COUNT", requestCount: 1 }, flashboot: "OFF", timeout: 1_800_000,
    });
    // The update (templateUpdatePayload) keeps the port.
    expect((templateUpdatePayload(lb, IMG) as any).ports).toEqual(["8000/http"]);
  });
  it("CPU queue: the cpu list on the endpoint (v2 places CPU workers; v1 ignored computeType CPU); its config in the template", () => {
    const p = endpointCreatePayload(cpu, "tplc") as any;
    expect(p).toMatchObject({ name: "fvc-fake", type: "QUEUE", templateId: "tplc", cpu: [{ id: "cpu3c", vcpuCount: 2 }, { id: "cpu5c", vcpuCount: 2 }], workers: { min: 0, max: 1, idleTimeout: 5 }, scaling: { type: "QUEUE_DELAY", queueDelay: 4 }, flashboot: "OFF", timeout: 1_800_000 });
    expect(p.gpu).toBeUndefined();
    expect(p.ports).toBeUndefined();
    expect(p.image).toBeUndefined();
    const t = templateCreatePayload(cpu, IMG, "tplc") as any;
    expect(t.env.FV_SERVE_MODE).toBe("runpod-queue");
    expect(t.entrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(templateCreatePayload(normalizeEndpointSpec({ name: "c", config: "/etc/fv/runpod-fake.toml" }), IMG, "t").entrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
  });
  it("update payload: only UpdateEndpointRequest keys (Runpod refuses extra keys); never the cpu list, type or templateId", () => {
    const UPDATE_KEYS = ["dataCenterIds", "flashboot", "gpu", "name", "networkVolumes", "scaling", "templateId", "timeout", "workers"];
    for (const sp of [cpu, gpu]) for (const k of Object.keys(endpointUpdatePayload(sp))) expect(UPDATE_KEYS).toContain(k);
    expect(endpointUpdatePayload(cpu)).toEqual({ workers: { min: 0, max: 1, idleTimeout: 5 }, scaling: { type: "QUEUE_DELAY", queueDelay: 4 }, flashboot: "OFF", timeout: 1_800_000 });
    expect(specDiff(cpu, { ...cpu, vcpu: 4 }).recreate).toEqual(["vcpu"]);
    // With the catalog, the GPU block carries the pools too.
    expect((endpointUpdatePayload(gpu, undefined, gpuPools(gpu.gpu_types!, CATALOG)) as any).gpu).toEqual({ pools: ["BLACKWELL_96"], excludedTypes: ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], count: 1, minCudaVersion: "13.0" });
  });
  it("invoke body and the endpoint view (never the endpoint's env)", () => {
    expect(invokeBody(undefined, 60)).toEqual({ input: { kind: "info" }, policy: { executionTimeout: 60000 } });
    const live = { id: "e", name: "fvc-x", type: "QUEUE", image: IMG, env: { SECRET: "v" }, gpu: { pools: ["BLACKWELL_96"], count: 1 }, workers: { min: 0, max: 1, idleTimeout: 5 }, flashboot: "OFF", timeout: 1_800_000, workerList: [{ id: "w1", status: "RUNNING", gpuTypeId: RTX6000, dataCenterId: "EU-RO-1", image: IMG, uptimeSeconds: 12 }] };
    const v = endpointView(live, { templateId: "t", costs: new Map([["w1", 0.1]]) }) as any;
    expect(JSON.stringify(v)).not.toContain("SECRET");
    expect(v).toMatchObject({ id: "e", templateId: "t", type: "QUEUE", compute: "GPU", image: IMG, workersMin: 0, workersMax: 1, flashboot: "OFF", timeout: 1_800_000 });
    expect(v.workers[0]).toMatchObject({ id: "w1", status: "RUNNING", cost: 0.1, gpu: RTX6000, dc: "EU-RO-1" });
    // Without GraphQL's $/hr (v2's worker list carries none): null, not missing.
    expect((endpointView(live) as any).workers[0].cost).toBeNull();
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

// ---------------------------------------------------------------- operations against a Runpod mock (REST v2)
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
  /** REST v2's ErrorResponse. */
  const err = (status: number, detail: string, errors?: string[]) => res(status, { title: status === 404 ? "Not Found" : "Bad Request", status, detail, ...(errors ? { errors } : {}) });
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
    // REST v1 is retired: its host does not resolve.
    if (u.host === "rest.runpod.io") throw new TypeError("fetch failed: getaddrinfo ENOTFOUND rest.runpod.io");
    if (u.host === "api.runpod.io" && u.pathname === "/graphql") {
      return res(200, { data: { myself: { clientBalance: m.balance, endpoints: [m.foreign, ...[...m.endpoints.values()].map((e) => ({ id: e.id, name: e.name, type: "QB", pods: e.pods || [] }))] } } });
    }
    if (u.host === "api.runpod.io" && u.pathname.startsWith("/v2/")) {
      const p = u.pathname.slice("/v2".length);
      if (p === "/catalog/gpus") return res(200, { gpus: CATALOG });
      if (p === "/templates" && method === "POST") {
        const id = `tpl${++m.n}`;
        m.templates.set(id, { id, ...body });
        return res(201, { id, ...body });
      }
      let mm = /^\/templates\/([^/]+)$/.exec(p);
      if (mm) {
        const t = m.templates.get(mm[1]!);
        if (method === "GET") return t ? res(200, t) : err(404, "template not found");
        if (method === "PATCH") {
          if (!t) return err(404, "template not found");
          Object.assign(t, body);
          return res(200, t);
        }
        if (method === "DELETE") {
          if (!t) return err(404, "delete template: get template: template not found");
          m.templates.delete(mm[1]!);
          return res(204, undefined);
        }
      }
      if (p === "/serverless" && method === "POST") {
        const id = `ep${++m.n}aaaaaaaaaaa`.slice(0, 14);
        const tpl = m.templates.get(body.templateId);
        if (!tpl) return err(404, "template not found");
        // The template resolved into the endpoint's container settings (no link kept); FlashBoot on as Runpod's create
        // left it on v1 (the reconcile PATCH sets the spec's), the env in clear.
        const { templateId: _t, ...rest } = body;
        m.endpoints.set(id, { id, ...rest, image: tpl.image, env: tpl.env, entrypoint: tpl.entrypoint, disk: tpl.disk, flashboot: "FLASHBOOT", pods: [] });
        return res(201, { id, name: body.name });
      }
      mm = /^\/serverless\/([^/]+)(\/workers)?$/.exec(p);
      if (mm) {
        const e = m.endpoints.get(mm[1]!) || (mm[1] === m.foreign.id ? m.foreign : null);
        if (!e) return err(404, "endpoint not found");
        if (mm[2] && method === "GET") return res(200, { workers: (e.pods || []).map((x: any) => ({ id: x.id, status: "RUNNING", isStale: false, gpuCount: 1, gpuTypeId: RTX6000, dataCenterId: "EU-RO-1" })), summary: { running: (e.pods || []).length, idle: 0, initializing: 0, throttled: 0, unhealthy: 0, total: (e.pods || []).length } });
        if (mm[2]) return err(405, "method not allowed");
        if (method === "GET") {
          const { pods: _p, ...view } = e;
          return res(200, { ...view, env: { ...(view.env || {}), SECRET: "x" } });
        }
        if (method === "PATCH") {
          const { templateId, ...rest } = body;
          if (templateId) {
            const tpl = m.templates.get(templateId);
            if (!tpl) return err(404, "template not found");
            Object.assign(e, { image: tpl.image, env: tpl.env, entrypoint: tpl.entrypoint, disk: tpl.disk, reapplied: (e.reapplied || 0) + 1 });
          }
          for (const [k, v] of Object.entries(rest)) e[k] = v && typeof v === "object" && !Array.isArray(v) && e[k] && typeof e[k] === "object" && !Array.isArray(e[k]) ? { ...e[k], ...(v as object) } : v;
          return res(200, e);
        }
        if (method === "DELETE") {
          if (m.deleteFails > 0) {
            m.deleteFails--;
            return err(400, "endpoint has running workers");
          }
          m.endpoints.delete(mm[1]!);
          return res(204, undefined);
        }
      }
      if (p === "/billing/serverless") return res(200, { records: m.billing, metadata: { query: { startTime: u.searchParams.get("startTime"), endTime: new Date().toISOString(), bucketSize: u.searchParams.get("bucketSize") }, recordCount: m.billing.length, uniqueServerlessCount: new Set(m.billing.map((b) => b.serverlessId)).size, totals: { totalAmount: m.billing.reduce((a, b) => a + b.totalAmount, 0) } } });
      return err(404, `mock: no route ${method} ${p}`);
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
  afterEach(() => {
    // Nothing ever goes to REST v1 (its host is gone): every Runpod REST call is api.runpod.io/v2.
    expect(mock.m.calls.filter((c) => c.url.includes("rest.runpod.io"))).toEqual([]);
    for (const c of mock.m.calls) if (c.url.includes("api.runpod.io") && !c.url.includes("/graphql")) expect(c.url).toMatch(/^https:\/\/api\.runpod\.io\/v2\//);
    vi.unstubAllGlobals();
  });

  it("resolves the image like clusters do (<image>-<channel> to a digest: cpu for CPU workers)", async () => {
    expect(await resolveImage(env, normalizeEndpointSpec({ name: "fake" }))).toBe("ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "b".repeat(64));
  });

  it("create (CPU): a template, then the endpoint on it with the cpu list (REST v2), then the reconcile PATCH; recorded in D1 and audited; a second with the same name is refused", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    expect(row).toMatchObject({ name: "fake", status: "active", own_template: 1, mode: "queue", image: IMG });
    expect(row.template_id).toBe("tpl1");
    expect(row.endpoint_id).toMatch(/^ep2/);
    expect(row.deadline! - row.created_at).toBe(120 * 60_000);
    const writes = mock.m.calls.filter((c) => c.method !== "GET" && !c.url.includes("graphql"));
    expect(writes.map((c) => `${c.method} ${new URL(c.url).host}${new URL(c.url).pathname}`)).toEqual(["POST api.runpod.io/v2/templates", "POST api.runpod.io/v2/serverless", `PATCH api.runpod.io/v2/serverless/${row.endpoint_id}`]);
    // The template carries the boot of its config (every endpoint sends one); the endpoint is a CPU queue on it.
    expect(writes[0]!.body).toMatchObject({ serverless: true, category: "CPU", image: IMG, entrypoint: ["/bin/sh", "-c", SLS_BOOT] });
    expect(writes[1]!.body).toMatchObject({ templateId: "tpl1", type: "QUEUE", cpu: [{ id: "cpu3c", vcpuCount: 2 }, { id: "cpu5c", vcpuCount: 2 }] });
    expect(writes[1]!.body).not.toHaveProperty("gpu");
    // The reconcile PATCH set flashboot back to OFF (Runpod's v1 create had dropped it; kept until a v2 run says otherwise).
    expect(mock.m.endpoints.get(row.endpoint_id!)!.flashboot).toBe("OFF");
    expect(mock.m.endpoints.get(row.endpoint_id!)!.env.FV_SERVE_MODE).toBe("runpod-queue");
    const a = await env.DB.prepare("SELECT action, ok FROM audit WHERE action = 'serverless.create'").all();
    expect(a.results).toEqual([{ action: "serverless.create", ok: 1 }]);
    await expect(createEndpoint(env, who, { name: "fake" }, { image: IMG })).rejects.toThrow(/exists/);
  });

  it("create (GPU queue): the catalog, a template, then the endpoint on it with the GPU pools (REST v2), then the reconcile PATCH", async () => {
    const row = await createEndpoint(env, who, { name: "h3", variant: "h3-turbo" }, { image: IMG });
    expect(row).toMatchObject({ status: "active", template_id: "tpl1", own_template: 1 });
    expect(mock.m.calls.filter((c) => c.url.endsWith("/v2/catalog/gpus"))).toHaveLength(1);
    const writes = mock.m.calls.filter((c) => c.method !== "GET" && !c.url.includes("graphql"));
    expect(writes.map((c) => `${c.method} ${new URL(c.url).pathname}`)).toEqual(["POST /v2/templates", "POST /v2/serverless", `PATCH /v2/serverless/${row.endpoint_id}`]);
    expect(writes[1]!.body).toMatchObject({ templateId: "tpl1", type: "QUEUE", networkVolumes: ["bz0ofkfx45"], dataCenterIds: ["EU-RO-1"], gpu: { pools: ["BLACKWELL_96"], excludedTypes: ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"], count: 1, minCudaVersion: "13.0" } });
    // The reconcile carries the pools again (the catalog answered), never the create-only fields.
    expect(writes[2]!.body).toMatchObject({ gpu: { pools: ["BLACKWELL_96"] }, workers: { min: 0, max: 1, idleTimeout: 5 }, flashboot: "OFF" });
    for (const k of ["type", "templateId", "name", "cpu"]) expect(writes[2]!.body).not.toHaveProperty(k);
  });

  it("create (GPU): a type Runpod's catalog puts in no serverless pool is refused before anything is made", async () => {
    await expect(createEndpoint(env, who, { name: "a40", variant: "h3-turbo", gpu_types: ["NVIDIA A40"] }, { image: IMG })).rejects.toMatchObject({ status: 409 });
    expect(mock.m.calls.filter((c) => c.method === "POST" && c.url.includes("api.runpod.io/v2"))).toEqual([]);
    expect(mock.m.templates.size).toBe(0);
  });

  it("create refuses below the floor + margin, and over the account limits", async () => {
    mock.m.balance = 9.5; // floor 8 + margin 2 = 10
    await expect(createEndpoint(env, who, { name: "fake" }, { image: IMG })).rejects.toMatchObject({ status: 402 });
    expect(mock.m.calls.some((c) => c.method === "POST" && c.url.includes("api.runpod.io/v2"))).toBe(false);
    mock.m.balance = 30;
    await expect(assertFloor(env, "x")).resolves.toBe(30);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [{ id: "a", workers_max: 4 }, { id: "b", workers_max: 4 }], { workers_max: 1 })).toThrow(/max_workers/);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [1, 2, 3, 4].map((i) => ({ id: `e${i}`, workers_max: 1 })), { workers_max: 1 })).toThrow(/max_endpoints/);
    expect(() => checkLimits(DEFAULT_SLS_POLICY, [{ id: "a", workers_max: 4 }], { id: "a", workers_max: 8 })).not.toThrow();
  });

  it("a failed endpoint create deletes the template it made and marks the row failed", async () => {
    const orig = mock.fetchMock;
    vi.stubGlobal("fetch", async (u: any, i: any = {}) => (String(u).endsWith("/v2/serverless") && i.method === "POST" ? new Response(JSON.stringify({ title: "Bad Request", status: 400, detail: "no capacity" }), { status: 400 }) : orig(u, i)));
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
    expect(mock.m.endpoints.get(row.endpoint_id!)!.workers).toEqual({ min: 0, max: 0, idleTimeout: 5 });
    await expect(scaleEndpoint(env, who, down, { workers_max: 1 })).rejects.toMatchObject({ status: 402 });
    mock.m.balance = 30;
    const up = await scaleEndpoint(env, who, down, { workers_max: 2 });
    expect(up.status).toBe("active");
    const IMG2 = IMG.replace(/a{64}/, "c".repeat(64));
    const u = await updateEndpoint(env, who, up, { ...JSON.parse(up.spec), image: { ref: IMG2 } }, { image: IMG2 });
    expect(u.image).toBe(IMG2);
    expect(mock.m.templates.get(up.template_id!)!.image).toBe(IMG2);
    // v2 keeps no link from the endpoint to its template: the template PATCH is followed by an endpoint PATCH with
    // `templateId`, which re-applies the image (and the config env) to the endpoint's workers.
    const reapply = mock.m.calls.filter((c) => c.method === "PATCH" && c.url.endsWith(`/v2/serverless/${row.endpoint_id}`) && c.body.templateId);
    expect(reapply.map((c) => c.body.templateId)).toEqual([up.template_id]);
    expect(mock.m.endpoints.get(row.endpoint_id!)!.image).toBe(IMG2);
    await expect(updateEndpoint(env, who, u, { ...JSON.parse(u.spec), network_volume: "bz0ofkfx45" })).rejects.toThrow(/network_volume: cannot change in place/);
  });

  it("a stored preset endpoint from before the one image: specOf carries the preset's config; a scale or a workers-only update never touches the template", async () => {
    // The three shapes an older fv-control stored for a preset endpoint: the variant image's baked config (no config at
    // all), a path in the variant image, or an inline snapshot of the config text.
    const legacyShapes = (p: (typeof SLS_PRESETS)[number], n: any) => {
      const base: any = { ...n };
      delete base.config;
      delete base.config_toml;
      const variant = p.compute === "CPU" ? "cpu" : "h3-turbo";
      return [
        { ...base, variant },
        { ...base, variant, config: `/etc/fv/${p.config ?? "runpod.toml"}` },
        { ...base, variant, config_toml: "# an older snapshot of the preset's config\n[engine]\nbackend = \"cuda\"\n" },
      ];
    };
    for (const p of SLS_PRESETS) {
      mock = mockRunpod();
      vi.stubGlobal("fetch", mock.fetchMock);
      env = { DB: d1(), RUNPOD_API_KEY: KEY, BALANCE_FLOOR: "8" };
      const name = `l-${p.id}`.slice(0, 31);
      const row0 = await createEndpoint(env, who, { name, preset: p.id }, { image: IMG });
      const n = specOf(row0);
      for (const [i, stored] of legacyShapes(p, n).entries()) {
        await env.DB.prepare("UPDATE serverless_endpoints SET spec = ?, status = 'active' WHERE id = ?").bind(JSON.stringify(stored), row0.id).run();
        const row = await getRow(env, row0.id);
        // What GET /:id, /serves and /outlook read: the preset's compute and config by name, never "no config".
        const so = specOf(row);
        expect([so.preset, so.compute, so.config, so.config_toml, (so as any).variant], `${p.id} #${i}`).toEqual([p.id, p.compute, p.config, undefined, undefined]);
        const before = mock.m.calls.length;
        const down = await scaleEndpoint(env, who, row, { workers_max: 0 });
        expect(down.status, `${p.id} #${i}`).toBe("scaled-down");
        expect(down.image, `${p.id} #${i}: the image never rolls on a scale`).toBe(row0.image);
        const writes = mock.m.calls.slice(before).filter((c) => c.method !== "GET" && !c.url.includes("graphql")).map((c) => `${c.method} ${new URL(c.url).pathname}`);
        expect(writes, `${p.id} #${i}`).toEqual([`PATCH /v2/serverless/${row0.endpoint_id}`]);
        // A whole-document update that changes only the worker counts (the PUT route merges over the stored spec) is a scale too.
        await env.DB.prepare("UPDATE serverless_endpoints SET spec = ?, status = 'active' WHERE id = ?").bind(JSON.stringify(stored), row0.id).run();
        const before2 = mock.m.calls.length;
        const u = await updateEndpoint(env, who, await getRow(env, row0.id), mergeSpec(stored, { workers_max: 0 }));
        expect(u.image, `${p.id} #${i}`).toBe(row0.image);
        expect(mock.m.calls.slice(before2).some((c) => c.method !== "GET" && c.url.includes("/templates")), `${p.id} #${i}: no template write`).toBe(false);
      }
    }
  });

  it("presets own their config by name: after a configs/serve file changes, preset endpoints still scale and update", async () => {
    const row = await createEndpoint(env, who, { name: "ref", preset: "h3-ref2v" }, { image: IMG });
    const prev = specOf(row);
    expect([prev.config, prev.config_toml]).toEqual(["runpod-h3-ref2v.toml", undefined]);
    const file = "runpod-h3-ref2v.toml";
    const orig = WORKER_CONFIGS[file]!;
    try {
      // The file was edited and fv-control redeployed (the generated table carries the new text).
      WORKER_CONFIGS[file] = orig + "\n# edited after the endpoint was created\n";
      expect(() => normalizeEndpointSpec(mergeSpec(prev, { workers_max: 0 }))).not.toThrow();
      expect(() => normalizeEndpointSpec(mergeSpec(prev, { execution_timeout_s: 900 }))).not.toThrow();
      const down = await scaleEndpoint(env, who, row, { workers_max: 0 });
      expect(down.status).toBe("scaled-down");
      // The next template update sends the edited text.
      mock.m.balance = 30;
      const IMG2 = IMG.replace(/a{64}/, "c".repeat(64));
      const u = await updateEndpoint(env, who, down, { ...specOf(down), image: { ref: IMG2 } }, { image: IMG2 });
      const tpl = mock.m.templates.get(u.template_id!)!;
      expect(Buffer.from(tpl.env.FV_WORKER_TOML_B64, "base64").toString("utf8")).toBe(WORKER_CONFIGS[file]);
    } finally {
      WORKER_CONFIGS[file] = orig;
    }
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
    expect(mock.m.endpoints.get(row.endpoint_id!)!.workers).toMatchObject({ min: 0, max: 0 });
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
    expect(mock.m.endpoints.get(b.endpoint_id!)!.workers).toMatchObject({ min: 0, max: 0 });
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

  it("billing: Runpod's per-endpoint spend (GET /billing/serverless) into the cost ledger (idempotent), only for fv-control's endpoints", async () => {
    const row = await createEndpoint(env, who, { name: "fake" }, { image: IMG });
    const day = new Date().toISOString().slice(0, 10);
    const next = new Date(Date.parse(`${day}T00:00:00Z`) + 86400_000).toISOString();
    mock.m.billing = [
      { serverlessId: row.endpoint_id!, startTime: `${day}T00:00:00Z`, endTime: next, totalAmount: 0.0123, gpuAmount: 0, cpuAmount: 0.0123, diskAmount: 0, feeAmount: 0 },
      { serverlessId: "strobe0000001", startTime: `${day}T00:00:00Z`, endTime: next, totalAmount: 5, gpuAmount: 5, cpuAmount: 0, diskAmount: 0, feeAmount: 0 },
    ];
    const rows = (await env.DB.prepare("SELECT * FROM serverless_endpoints").all()).results as SlsRow[];
    expect(await recordBilling(env, rows)).toBe(1);
    expect(await recordBilling(env, rows)).toBe(1);
    const q = mock.m.calls.filter((c) => c.url.includes("/v2/billing/serverless"));
    expect(q).toHaveLength(2);
    expect(new URL(q[0]!.url).searchParams.get("bucketSize")).toBe("day");
    expect(new URL(q[0]!.url).searchParams.get("startTime")).toMatch(/^\d{4}-\d{2}-\d{2}T00:00:00Z$/);
    // v2 reports no billed time (v1's timeBilledMs): minutes 0.
    const c = await env.DB.prepare("SELECT * FROM cost_daily").all();
    expect(c.results).toEqual([{ day, pod_id: `sls:${row.endpoint_id}`, cluster_id: null, owner: "serverless:fake", usd: 0.0123, minutes: 0, idle_minutes: 0 }]);
    expect((await getRow(env, row.id)).cost_usd).toBeCloseTo(0.0123);
  });

  it("the tick makes no Runpod call when fv-control has no endpoint", async () => {
    const t = await serverlessTick(env);
    expect(t).toEqual({ endpoints: 0, actions: [] });
    expect(mock.m.calls).toEqual([]);
  });
});

describe("defaults", () => {
  it("defaultEndpointSpec validates for CPU and every preset; a stored spec from before the one image reads as its compute", () => {
    expect(checkEndpointSpec(defaultEndpointSpec("x", "CPU")).ok).toBe(true);
    for (const p of SLS_PRESETS) expect(checkEndpointSpec(defaultEndpointSpec("x", p.compute, p.id)).ok, p.id).toBe(true);
    // As stored with per-variant images: a variant and no config (the image's baked one). It normalizes to its compute
    // and that config by name, which fv-control now sends at boot; for every old variant.
    for (const [v, cfg] of [["h3-turbo", "runpod.toml"], ["h3-max", "runpod-h3-max.toml"], ["ltx", "runpod-ltx.toml"], ["wan", "runpod-wan.toml"], ["wan5b", "runpod-wan5b.toml"], ["sfwan", "runpod-sfwan.toml"]] as const) {
      const n = normalizeEndpointSpec({ name: "old", variant: v });
      expect([(n as any).variant, n.compute, n.config], v).toEqual([undefined, "GPU", cfg]);
      expect(Buffer.from(workerEnv(n, IMG).FV_WORKER_TOML_B64!, "base64").toString("utf8"), v).toBe(WORKER_CONFIGS[cfg]);
    }
    const cpuOld = normalizeEndpointSpec({ name: "old", variant: "cpu", compute: "CPU" });
    expect([(cpuOld as any).variant, cpuOld.compute, cpuOld.config]).toEqual([undefined, "CPU", "runpod-fake.toml"]);
  });
});
