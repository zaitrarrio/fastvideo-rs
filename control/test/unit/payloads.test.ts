import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { GATEWAY_BASE_PODS } from "../../src/cluster/gateway-base";
import {
  GATEWAY_BOOT,
  gatewayCreatePayload,
  gatewaySystemEnv,
  gatewayToml,
  imageIdentEnv,
  isReserved,
  WORKER_BOOT,
  workerCreatePayload,
  workerPlacements,
  workerSystemEnv,
  type EnvCtx,
} from "../../src/cluster/payloads";
import { defaultSpec, normalizeSpec } from "../../src/cluster/spec";

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
    const base = f.slice(f.indexOf("[server]"), f.indexOf("[[pools]]"));
    expect(GATEWAY_BASE_PODS).toBe(base);
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
