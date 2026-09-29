import { describe, expect, it } from "vitest";
import { attribute, DEFAULT_POLICIES, syncAlerts } from "../../src/alerts";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { deleteVar, listVars, MASK, resolvePlain, resolveView, setVar } from "../../src/envvars";
import { validateDispatch } from "../../src/github";
import { aeSeriesSql, parseProm } from "../../src/metrics";
import { clusterDrift } from "../../src/releases";
import { defaultSpec } from "../../src/cluster/spec";
import { rateLimit, scrub } from "../../src/util";
import { d1 } from "./d1shim";

const mkEnv = (): Env => ({ DB: d1(), CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_SECRETKEY123" }) as unknown as Env;

describe("env resolution: pod > cluster > account > system", () => {
  it("resolves, masks secrets and refuses reserved keys", async () => {
    const env = mkEnv();
    await setVar(env, "account", "", "RUST_LOG", "warn", false, "t");
    await setVar(env, "account", "", "HF_TOKEN", "hf_secret", true, "t");
    await setVar(env, "cluster", "c1", "RUST_LOG", "info,fv=debug", false, "t");
    await setVar(env, "pod", "p1", "RUST_LOG", "trace", false, "t");
    await setVar(env, "cluster", "c1", "FV_FEATURE_X", "1", false, "t");
    const system = { RUST_LOG: "info", FV_INTERNAL_TOKEN: "tok", FV_CF_API_TOKEN: "{{ RUNPOD_SECRET_fv_cf_api_token }}" };
    const p1 = await resolvePlain(env, "c1", "p1", system);
    expect(p1.RUST_LOG).toBe("trace");
    expect(p1.HF_TOKEN).toBe("hf_secret");
    expect(p1.FV_INTERNAL_TOKEN).toBe("tok");
    const p2 = await resolvePlain(env, "c1", "p2", system);
    expect(p2.RUST_LOG).toBe("info,fv=debug");
    const other = await resolvePlain(env, "c2", null, system);
    expect(other.RUST_LOG).toBe("warn");
    expect(other.FV_FEATURE_X).toBeUndefined();
    const view = await resolveView(env, "c1", "p1", system);
    const by = Object.fromEntries(view.map((v) => [v.key, v]));
    expect(by.RUST_LOG).toMatchObject({ value: "trace", source: "pod", overrides: ["system", "account", "cluster"] });
    expect(by.HF_TOKEN).toMatchObject({ value: MASK, secret: true, source: "account" });
    expect(by.FV_INTERNAL_TOKEN).toMatchObject({ value: MASK, secret: true, source: "system" });
    expect(by.FV_CF_API_TOKEN!.runpod_secret_ref).toBe(true);
    expect(JSON.stringify(view)).not.toContain("hf_secret");
    expect(JSON.stringify(view)).not.toContain('"tok"');
    // Stored sealed.
    const rows = await listVars(env, "account", "");
    expect(rows.find((r) => r.key === "HF_TOKEN")!.value).not.toContain("hf_secret");
    await expect(setVar(env, "cluster", "c1", "FV_INTERNAL_TOKEN", "x", false, "t")).rejects.toThrow(/controller/);
    await expect(setVar(env, "cluster", "c1", "FV_POOL_WAN_URLS", "x", false, "t")).rejects.toThrow(/controller/);
    await expect(setVar(env, "cluster", "c1", "bad-key", "x", false, "t")).rejects.toThrow(/invalid/);
    await deleteVar(env, "pod", "p1", "RUST_LOG");
    expect((await resolvePlain(env, "c1", "p1", system)).RUST_LOG).toBe("info,fv=debug");
  });
});

describe("metrics", () => {
  it("parses whitelisted Prometheus series only", () => {
    const t = `# HELP fv_pool_queued x\n# TYPE fv_pool_queued gauge\nfv_pool_queued{pool="wan"} 3\nfv_pool_queued{pool="ltx",x="a\\"b"} 1\nfv_http_request_duration_seconds_bucket{le="1"} 5\nfv_ready 1\nfv_pool_running{pool="wan"} NaN\n`;
    const s = parseProm(t);
    expect(s.map((x) => x.name)).toEqual(["fv_pool_queued", "fv_pool_queued", "fv_ready"]);
    expect(s[1]!.labels).toEqual({ pool: "ltx", x: 'a"b' });
  });
  it("AE SQL never interpolates an unsafe pod id", () => {
    expect(aeSeriesSql(6, 5, "abc123")).toContain("blob2 = 'abc123'");
    expect(aeSeriesSql(6, 5, "x' OR 1=1 --")).not.toContain("OR 1=1");
  });
});

describe("attribution, alerts, rate limits", () => {
  it("attributes external pods by prefix", () => {
    expect(attribute("fv-build", DEFAULT_POLICIES.attribution)).toBe("external:build-pod");
    expect(attribute("fv-b200-bench-0929", DEFAULT_POLICIES.attribution)).toBe("external:b200-bench");
    expect(attribute("little_azure_rook", DEFAULT_POLICIES.attribution)).toBe("external");
    expect(attribute("foo-bar-123", DEFAULT_POLICIES.attribution)).toBe("external:foo-bar");
  });
  it("opens, refreshes and resolves alerts", async () => {
    const env = mkEnv();
    await syncAlerts(env, [{ key: "pod_idle:a", kind: "pod_idle", severity: "warn", message: "m1" }], ["pod_idle"]);
    await syncAlerts(env, [{ key: "pod_idle:a", kind: "pod_idle", severity: "warn", message: "m2" }], ["pod_idle"]);
    let r = await env.DB.prepare("SELECT * FROM alerts").all<any>();
    expect(r.results.length).toBe(1);
    expect(r.results[0].message).toBe("m2");
    await syncAlerts(env, [], ["pod_idle"]);
    r = await env.DB.prepare("SELECT * FROM alerts WHERE resolved_at IS NULL").all<any>();
    expect(r.results.length).toBe(0);
  });
  it("rate limits per window", async () => {
    const env = mkEnv();
    const got = [];
    for (let i = 0; i < 7; i++) got.push(await rateLimit(env, "login:1.2.3.4", 5, 900));
    expect(got).toEqual([true, true, true, true, true, false, false]);
  });
  it("scrubs the controller's secrets from text", () => {
    expect(scrub(mkEnv(), "error with rpa_SECRETKEY123 inside")).toBe("error with [redacted] inside");
  });
});

describe("github and releases", () => {
  it("validates release dispatch inputs", () => {
    expect(validateDispatch({ action: "promote", target: "2cd1ba0", channel: "stable" })).toMatchObject({ action: "promote", target: "2cd1ba0", templates: "true", dry_run: "false" });
    expect(() => validateDispatch({ action: "promote", target: "rm -rf /" })).toThrow(/target/);
    expect(() => validateDispatch({ action: "delete" as any })).toThrow(/action/);
    expect(validateDispatch({ action: "rollback", channel: "stable", to: "12" }).to).toBe("12");
  });
  it("drift against the channel head", () => {
    const spec = defaultSpec("x");
    const c: any = { spec, state: { images: {}, gateway: { pod: "g", image: "r@sha256:g1" }, workers: { wan: [{ pod: "w", image: "r@sha256:w0" }] } } };
    const heads: any = [{ channel: "stable", git_sha: "abcdef1234", digests: { gateway: "r@sha256:g1", wan5b: "r@sha256:w1" } }];
    const d = clusterDrift(c, heads);
    expect(d.drift).toBe(true);
    expect(d.pods.find((p) => p.pod === "w")).toMatchObject({ key: "wan5b", running: "sha256:w0", head: "sha256:w1", drift: true });
    expect(d.pods.find((p) => p.pod === "g")!.drift).toBe(false);
  });
});
