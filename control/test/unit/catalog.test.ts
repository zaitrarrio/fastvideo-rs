import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";
import CATALOG from "../../src/cluster/catalog.json";
import { defaultSpec } from "../../src/cluster/spec";
import { b64, randomBytes } from "../../src/crypto";
import { readDoc, saveDoc } from "../../src/docs";
import { envKeyOptions } from "../../src/dynamic";
import type { Env } from "../../src/env";
import { poolScopeId, resolvePlain, resolveView, setVar } from "../../src/envvars";
import { d1 } from "./d1shim";

const repo = new URL("../../../", import.meta.url).pathname;
function rustSources(dir: string, out: string[] = []): string[] {
  for (const n of readdirSync(dir)) {
    const p = join(dir, n);
    if (n === "target" || n.startsWith(".")) continue;
    if (statSync(p).isDirectory()) rustSources(p, out);
    else if (n.endsWith(".rs")) out.push(readFileSync(p, "utf8"));
  }
  return out;
}

describe("catalog.json", () => {
  it("every engine env key is read by the Rust code under that exact name", () => {
    const src = rustSources(join(repo, "crates")).join("\n");
    for (const e of CATALOG.engine_env) expect(src.includes(`"${e.id}"`) || new RegExp(`var\\("${e.id}"|\\b${e.id}\\b`).test(src), e.id).toBe(true);
    // The values the descriptions promise.
    expect(src).toMatch(/"0" \| "off" => 0,\n\s*"2" \| "on" => 2,/); // FASTVIDEO_ATTN_SAGE
    expect(src).toMatch(/FASTVIDEO_FLASH_KERNEL=v1\|v2\|cudnn\|dc\|auto/);
    expect(src).toMatch(/FASTVIDEO_H3_QUANT=w8a8\|mxfp8\|off/);
    expect(src).toMatch(/FASTVIDEO_DIT_OFFLOAD=auto\|resident\|streamed/);
    expect(src).toMatch(/FASTVIDEO_WAN_AUDIO=mmaudio/);
    expect(src).toMatch(/flag\("FV_LONGLIVE_RECACHE", true\)/);
    expect(src).toMatch(/flag\("FV_LONGLIVE_INFINITY", false\)/);
  });
  it("recipes: unique, a known family, the Plug recipes marked as not servable", () => {
    const ids = CATALOG.recipes.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
    const fams = CATALOG.families.map((f) => f.id);
    for (const r of CATALOG.recipes) expect(fams).toContain(r.family);
    for (const p of ["h3-plug-4step", "wan5b-plug-4step", "wan14b-plug-4step"]) expect(CATALOG.recipes.find((r) => r.id === p)?.serve).toBe(false);
    for (const p of ["ltx-pro", "ltx-draft", "ltx25-ref2v", "ltx25-a2v-guided", "h3-ref2v-turbo", "sfwan21-1.3b"]) expect(CATALOG.recipes.find((r) => r.id === p)?.serve, p).toBe(true);
    for (const m of CATALOG.models) expect(CATALOG.recipes.find((r) => r.id === m.recipe)?.serve, m.id).toBe(true);
  });
  it("env key suggestions: the engine's with what they do, plus the keys in use", () => {
    const o = envKeyOptions([
      { key: "FASTVIDEO_ATTN_SAGE", scope: "pool", n: 1 },
      { key: "MY_KEY", scope: "cluster", n: 2 },
    ]);
    expect(o.find((x) => x.id === "FASTVIDEO_ATTN_SAGE")!.detail).toMatch(/recipe decides.*in use: pool×1/);
    expect(o.find((x) => x.id === "MY_KEY")!.detail).toBe("in use: cluster×2");
    expect(o.find((x) => x.id === "FV_LONGLIVE_WEIGHTS")!.detail).toMatch(/NON-COMMERCIAL/);
  });
});

describe("pool env scope", () => {
  let env: Env;
  beforeEach(async () => {
    env = { DB: d1(), CONTROL_KEK: b64(randomBytes(32)), SESSION_SECRET: "s".repeat(40), RUNPOD_API_KEY: "rpa_x" } as unknown as Env;
    const spec = defaultSpec("t1", "ltx");
    await env.DB.prepare("INSERT INTO clusters (id, name, spec, state, status, source, created_at, updated_at, created_by) VALUES ('c_1', 't1', ?, '{}', 'defined', 'controller', 0, 0, 't')").bind(JSON.stringify(spec)).run();
  });
  it("applies to the pool's workers only, between cluster and pod", async () => {
    await setVar(env, "cluster", "c_1", "FASTVIDEO_ATTN_SAGE", "2", false, "t");
    await setVar(env, "pool", poolScopeId("c_1", "ltx-pro"), "FASTVIDEO_ATTN_SAGE", "0", false, "t");
    await setVar(env, "pool", poolScopeId("c_1", "ltx-pro"), "SECRET_X", "s3cret", true, "t");
    await setVar(env, "pod", "pod1", "FASTVIDEO_ATTN_SAGE", "2", false, "t");
    const sys = { RUST_LOG: "info" };
    // A new worker of ltx-pro (no pod id yet: a scale-up or a roll) gets the pool's value.
    expect(await resolvePlain(env, "c_1", null, sys, "ltx-pro")).toMatchObject({ FASTVIDEO_ATTN_SAGE: "0", SECRET_X: "s3cret" });
    // Another pool and the gateway (no pool) do not.
    expect((await resolvePlain(env, "c_1", null, sys, "ltx")).FASTVIDEO_ATTN_SAGE).toBe("2");
    expect((await resolvePlain(env, "c_1", null, sys, null)).SECRET_X).toBeUndefined();
    // The pod level still wins.
    expect((await resolvePlain(env, "c_1", "pod1", sys, "ltx-pro")).FASTVIDEO_ATTN_SAGE).toBe("2");
    const view = await resolveView(env, "c_1", null, sys, "ltx-pro");
    expect(view.find((v) => v.key === "FASTVIDEO_ATTN_SAGE")).toMatchObject({ source: "pool", overrides: ["cluster"] });
    expect(view.find((v) => v.key === "SECRET_X")).toMatchObject({ source: "pool", secret: true, value: "••••••••" });
  });
  it("is an env document pool:<cluster>:<pool> (by cluster id or name); unknown pools are refused", async () => {
    const r = await saveDoc(env, "env", "pool:t1:ltx-pro", { FASTVIDEO_ATTN_SAGE: { value: "0", secret: false } }, { version: 0, actor: "t" });
    expect(r.doc).toEqual({ FASTVIDEO_ATTN_SAGE: { value: "0", secret: false } });
    expect(await readDoc(env, "env", "pool:c_1:ltx-pro")).toEqual({ FASTVIDEO_ATTN_SAGE: { value: "0", secret: false } });
    await expect(saveDoc(env, "env", "pool:c_1:nope", {}, { version: 0, actor: "t" })).rejects.toMatchObject({ status: 404 });
    await expect(saveDoc(env, "env", "pool:c_1:ltx-pro", { FV_INTERNAL_TOKEN: { value: "x", secret: false } }, { version: 1, actor: "t" })).rejects.toThrow(/set by the controller/);
  });
});
