import { json } from "@codemirror/lang-json";
import { ensureSyntaxTree } from "@codemirror/language";
import { EditorState } from "@codemirror/state";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defaultSpec, normalizeSpec } from "../../src/cluster/spec";
import { b64, randomBytes } from "../../src/crypto";
import { docHistory, docVersion, planSpec, readDoc, restoreDoc, saveDoc, validateDoc } from "../../src/docs";
import type { Env } from "../../src/env";
import { jsonSchemas, validate } from "../../src/schemas";
import { pathOfNode, rangeOfPath } from "../../ui/editor";
import { validate as miniValidate, schemaAt, setAt, unwrap } from "../../ui/schema";
import { diffLines } from "../../ui/view";
import { d1 } from "./d1shim";

const S = jsonSchemas() as Record<string, any>;

describe("JSON Schemas from the zod schemas", () => {
  it("covers every editable document, with descriptions and live-value hints", () => {
    expect(Object.keys(S).sort()).toEqual(["attribution", "build-pods-policy", "cluster-spec", "env", "extend", "log-query", "mint-key", "policies", "pool", "release-dispatch", "roll", "scale", "serverless-endpoint", "serverless-policy", "serverless-scale", "standalone-launch", "token-create"]);
    const spec = S["cluster-spec"];
    expect(spec.additionalProperties).toBe(false);
    for (const k of ["name", "image", "regions", "control_plane", "auth", "pools", "cap_s", "balance_floor"]) expect(spec.required).toContain(k);
    expect(spec.properties.gateway).toBeUndefined();
    const pool = spec.properties.pools.items;
    expect(pool.properties.count.description).toMatch(/Worker pods/);
    expect(pool.properties.gpu_types.items["x-dynamic"]).toBe("gpu_types");
    expect(spec.properties.image.properties.channel["x-dynamic"]).toBe("channels");
    expect(S.env.propertyNames["x-dynamic"]).toBe("env_keys");
    expect(S.env.additionalProperties.properties.set["x-secret"]).toBe(true);
    expect(S.policies.properties.attribution.maxItems).toBe(50);
  });
  it("server validation: the templates pass, mistakes fail with paths", () => {
    expect(validate("cluster-spec", normalizeSpec({ name: "a" })).ok).toBe(true);
    expect(validate("cluster-spec", normalizeSpec({ name: "b", template: "tiny-cpu" })).ok).toBe(true);
    const bad = { ...defaultSpec("c"), cap_s: 10, extra: 1, pools: [{ ...defaultSpec("c").pools[0], count: 99 }] };
    const r = validate("cluster-spec", bad);
    expect(r.ok).toBe(false);
    const paths = (r as any).issues.map((i: any) => i.path.join("."));
    expect(paths).toContain("cap_s");
    expect(paths).toContain("pools.0.count");
    expect((r as any).issues.some((i: any) => /extra/.test(i.message) || i.path.includes("extra"))).toBe(true);
    const two = validate("cluster-spec", { ...defaultSpec("d"), image: { channel: "stable", sha: "abcdef1" } });
    expect((two as any).issues[0].message).toMatch(/exactly one/);
    expect(() => normalizeSpec({ name: "e", control_plane: "gateway", gateway: { enabled: true }, auth: "open" })).toThrow(/auth/);
  });
  it("the editor's in-browser validator agrees with zod on the schema-expressible cases", () => {
    const spec = S["cluster-spec"];
    const cases: [unknown, boolean][] = [
      [defaultSpec("ok1"), true],
      [defaultSpec("ok2", "tiny-cpu"), true],
      [{ ...defaultSpec("x"), cap_s: 1 }, false],
      [{ ...defaultSpec("x"), regions: ["mars"] }, false],
      [{ ...defaultSpec("x"), regions: [] }, false],
      [{ ...defaultSpec("x"), regions: ["us"] }, false],
      [{ ...defaultSpec("x"), regions: ["eu", "us"] }, false],
      [{ ...defaultSpec("x"), bogus: true }, false],
      [{ ...defaultSpec("x"), name: "Bad Name" }, false],
      [{ ...defaultSpec("x"), auth: "open" }, false],
      [{ ...defaultSpec("x"), control_plane: "gateway" }, false],
      [{ ...defaultSpec("x"), auto_stop_idle_min: null }, true],
      [{ ...defaultSpec("x"), auto_stop_idle_min: 2 }, false],
      [setAt(defaultSpec("x"), ["pools", 0, "count"], 2.5), false],
      [setAt(defaultSpec("x"), ["pools", 0, "compute"], "TPU"), false],
      [(() => { const s: any = defaultSpec("x"); delete s.cap_s; return s; })(), false],
    ];
    for (const [doc, ok] of cases) {
      expect(validate("cluster-spec", doc).ok, JSON.stringify(doc).slice(0, 80)).toBe(ok);
      expect(miniValidate(spec, doc).length === 0, JSON.stringify(doc).slice(0, 80)).toBe(ok);
    }
    const env = S.env;
    expect(miniValidate(env, { RUST_LOG: { value: "info", secret: false } })).toEqual([]);
    expect(miniValidate(env, { "bad-key": { value: "x", secret: false } }).length).toBe(1);
    expect(miniValidate(S.policies, { idle_gpu_pct: 5 }).map((i) => i.message).join()).toMatch(/missing required/);
  });
  it("schemaAt / unwrap find the governing subschema", () => {
    const spec = S["cluster-spec"];
    expect(schemaAt(spec, ["pools", 3, "count"])?.type).toBe("integer");
    expect(unwrap(schemaAt(spec, ["auto_stop_idle_min"]), spec)?.nullable).toBe(true);
    expect(schemaAt(S.env, ["ANY_KEY", "secret"])?.type).toBe("boolean");
  });
});

describe("editor: JSON paths and positions (CodeMirror state, no DOM)", () => {
  it("maps positions to paths and paths to ranges", () => {
    const doc = JSON.stringify({ name: "t", pools: [{ id: "a", count: 1 }, { id: "b", count: 2 }] }, null, 2);
    const state = EditorState.create({ doc, extensions: [json()] });
    const tree = ensureSyntaxTree(state, doc.length)!;
    const pos = doc.indexOf("2", doc.indexOf('"b"'));
    expect(pathOfNode(state, tree.resolveInner(pos + 1, -1)).path).toEqual(["pools", 1, "count"]);
    const key = pathOfNode(state, tree.resolveInner(doc.indexOf('"name"') + 2, 1));
    expect(key).toEqual({ path: ["name"], key: true });
    const r = rangeOfPath(state, ["pools", 1, "count"]);
    expect(doc.slice(r.from, r.to)).toBe("2");
    const k = rangeOfPath(state, ["pools", 0, "id"], true);
    expect(doc.slice(k.from, k.to)).toBe('"id"');
    const missing = rangeOfPath(state, ["pools", 0, "nope"]);
    expect(doc.slice(missing.from, missing.to)).toBe("{");
  });
  it("diffs lines", () => {
    const d = diffLines("a\nb\nc\nd", "a\nB\nc\nd\ne");
    expect(d.filter((l) => l.op !== " ").map((l) => l.op + l.text)).toEqual(["-b", "+B", "+e"]);
  });
});

describe("documents: versions, validation, history, restore, secrets", () => {
  let env: Env;
  const realFetch = globalThis.fetch;
  beforeEach(async () => {
    env = { DB: d1(), CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_x" } as unknown as Env;
    const spec = normalizeSpec({ name: "t1", template: "tiny-cpu" });
    await env.DB.prepare("INSERT INTO clusters (id, name, spec, state, status, source, created_at, updated_at, created_by) VALUES ('c_1', 't1', ?, '{}', 'defined', 'controller', 0, 0, 't')").bind(JSON.stringify(spec)).run();
    globalThis.fetch = vi.fn(async (_u: any, init: any) => {
      const q = JSON.parse(init.body).query as string;
      const data = q.includes("gpuTypes") ? { gpuTypes: [{ id: "x", securePrice: 2 }] } : { myself: { clientBalance: 40, currentSpendPerHr: 1, spendLimit: 80 } };
      return new Response(JSON.stringify({ data }), { headers: { "content-type": "application/json" } });
    }) as any;
  });
  afterEach(() => {
    globalThis.fetch = realFetch;
  });

  it("saves a spec with the version check, keeps history, restores", async () => {
    const d0 = (await readDoc(env, "cluster-spec", "c_1")) as any;
    expect(await docVersion(env, "cluster-spec", "c_1")).toBe(0);
    const d1v = { ...d0, cap_s: 3600 };
    const r1 = await saveDoc(env, "cluster-spec", "c_1", d1v, { version: 0, actor: "t" });
    expect(r1.version).toBe(1);
    await expect(saveDoc(env, "cluster-spec", "c_1", { ...d0, cap_s: 4000 }, { version: 0, actor: "t2" })).rejects.toMatchObject({ status: 409 });
    await expect(saveDoc(env, "cluster-spec", "c_1", { ...d0, cap_s: 5 }, { version: 1, actor: "t" })).rejects.toMatchObject({ status: 400 });
    await expect(saveDoc(env, "cluster-spec", "c_1", { ...d0, name: "other" }, { version: 1, actor: "t" })).rejects.toThrow(/name is fixed/);
    const hist = await docHistory(env, "cluster-spec", "c_1");
    expect(hist.length).toBe(1);
    expect((hist[0]!.before as any).cap_s).toBe(1800);
    const rr = await restoreDoc(env, "cluster-spec", "t1", hist[0]!.audit_id, "before", { version: 1, actor: "t" });
    expect((rr.doc as any).cap_s).toBe(1800);
    expect(rr.version).toBe(2);
    expect((await docHistory(env, "cluster-spec", "c_1"))[0]!.action).toBe("doc.restore");
  });

  it("plans a stopped cluster's change with the projection", async () => {
    const d0 = (await readDoc(env, "cluster-spec", "c_1")) as any;
    const p: any = await planSpec(env, "c_1", { ...d0, pools: [{ ...d0.pools[0], count: 3 }] });
    expect(p.ok).toBe(true);
    expect(p.actions[0].detail).toMatch(/Start would create 3 pod/);
    expect(p.projection.cluster_dph).toBeGreaterThan(0.1);
    const bad: any = await planSpec(env, "c_1", { ...d0, cap_s: "x" });
    expect(bad.ok).toBe(false);
  });

  it("env documents: secrets are write-only and never read back", async () => {
    const r = await saveDoc(env, "env", "cluster:t1", { RUST_LOG: { value: "debug", secret: false }, HF_TOKEN: { value: null, secret: true, set: "hf_secret_value" } }, { version: 0, actor: "t" });
    expect(r.doc).toEqual({ HF_TOKEN: { value: null, secret: true }, RUST_LOG: { value: "debug", secret: false } });
    expect(JSON.stringify(await docHistory(env, "env", "cluster:c_1"))).not.toContain("hf_secret_value");
    // Keeping a secret: value null and no `set`.
    const r2 = await saveDoc(env, "env", "cluster:c_1", { HF_TOKEN: { value: null, secret: true } }, { version: 1, actor: "t" });
    expect(r2.doc).toEqual({ HF_TOKEN: { value: null, secret: true } });
    const rows = await env.DB.prepare("SELECT value FROM env_vars WHERE key = 'HF_TOKEN'").all<any>();
    expect(rows.results[0].value).not.toContain("hf_secret");
    const v1 = await validateDoc(env, "env", "cluster:c_1", { NEW_SECRET: { value: null, secret: true } });
    expect(v1.issues[0]!.message).toMatch(/needs its value in `set`/);
    const v2 = await validateDoc(env, "env", "cluster:c_1", { HF_TOKEN: { value: "shown", secret: true } });
    expect(v2.issues[0]!.message).toMatch(/write-only/);
    const v3 = await validateDoc(env, "env", "cluster:c_1", { FV_INTERNAL_TOKEN: { value: "x", secret: false } });
    expect(v3.issues[0]!.message).toMatch(/set by the controller/);
    // Restoring a snapshot from before the secret existed deletes it; restoring one with it cannot bring its value back.
    const h = await docHistory(env, "env", "cluster:c_1");
    const back = await restoreDoc(env, "env", "cluster:c_1", h[h.length - 1]!.audit_id, "before", { version: 2, actor: "t" });
    expect(back.doc).toEqual({});
    const again = await restoreDoc(env, "env", "cluster:c_1", h[h.length - 1]!.audit_id, "after", { version: 3, actor: "t" });
    expect(again.skipped).toEqual(["HF_TOKEN"]);
    expect(again.doc).toEqual({ RUST_LOG: { value: "debug", secret: false } });
  });

  it("policies and attribution share one setting", async () => {
    const p = (await readDoc(env, "policies", "default")) as any;
    await saveDoc(env, "attribution", "default", [{ prefix: "zz-", owner: "external:zz" }], { version: 0, actor: "t" });
    expect(((await readDoc(env, "policies", "default")) as any).attribution).toEqual([{ prefix: "zz-", owner: "external:zz" }]);
    await expect(saveDoc(env, "policies", "default", { ...p, idle_gpu_pct: 500 }, { version: 0, actor: "t" })).rejects.toMatchObject({ status: 400 });
    await expect(saveDoc(env, "attribution", "default", [{ prefix: "", owner: "x" }], { version: 1, actor: "t" })).rejects.toMatchObject({ status: 400 });
  });
});
