// The family templates (owner 2026-10-18: "for the presets means all of the weights for a model family and it's
// variants are installed onto 1 [pod]"): h3, fasth3, ltx, wan and streaming are ONE pool each, a family preset whose
// worker config serves every variant from one fv-serve process (the first model resident, the rest swapped in), whose
// weights are the union of its members' trees and whose disk and GPU fit that union and its largest member.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import CATALOG from "../../src/cluster/catalog.json";
import { defaultSpec, FAMILY_PRESETS, FASTH3_FAMILY_MEMBERS, H3_FAMILY_MEMBERS, LTX_FAMILY_MEMBERS, MODEL_PRESETS, normalizeSpec, POOL_PRESETS, poolModelFamilies, STREAMING_FAMILY_MEMBERS, TEMPLATES, type ModelRef } from "../../src/cluster/spec";
import type { Env } from "../../src/env";
import { mustVariant } from "../../src/families";
import { H3_DRAFT } from "../../src/presets";
import { weightsPlan, weightsSource } from "../../src/providers";
import { r2Bytes, r2DiskGb } from "../../src/r2weights";

const repo = new URL("../../../", import.meta.url).pathname;
/** Catalog models no model preset serves (no worker config in configs/serve runs them as a pool of their own). */
const NO_PRESET = new Set(["fasth3-4step-vsa-480p-taeh3", "fasth3-8step-dense", "ltx25-distill-sol-nvfp4-taehv", "fastwan22-ti2v-5b-taehv", "wan14b-plug-4step"]);
const preset = (id: string) => POOL_PRESETS.find((p) => p.id === id)!;
const modelsOf = (ids: string[]) => ids.flatMap((id) => preset(id).pool.models || []);
const byId = (ms: ModelRef[]) => [...ms].sort((a, b) => a.id.localeCompare(b.id));
const env = {} as Env;
/** A family config's [[models]] blocks: id (default: the variant's id), resident. */
const blocks = (toml: string) =>
  toml
    .split(/^\[\[models\]\]$/m)
    .slice(1)
    .map((b) => ({ id: /^id = "([^"]+)"/m.exec(b)?.[1] ?? mustVariant(/^variant = "([^"]+)"/m.exec(b)![1]!).id, resident: /^resident = (true|false)/m.exec(b)?.[1] }));

describe("family templates: one pod per family", () => {
  it("every family template is ONE pool, one worker, a family preset (wan: the wan preset)", () => {
    const want: Record<string, string> = { h3: "h3-family", fasth3: "fasth3-family", ltx: "ltx-family", streaming: "streaming-family", wan: "wan" };
    for (const [t, pool] of Object.entries(want)) {
      expect(TEMPLATES[t]!.pools, t).toEqual([pool]);
      const s = normalizeSpec(defaultSpec("tpl-x", t));
      expect(s.pools.map((p) => [p.id, p.count]), t).toEqual([[pool, 1]]);
    }
    expect(TEMPLATES.h3!.title).toMatch(/ONE pod/);
    // What a family pool leaves out is in its title, never silent.
    expect(TEMPLATES.h3!.title).toMatch(/fasth3-8step-dense/);
    expect(TEMPLATES.wan!.title).toMatch(/fastwan21/);
  });
  it("h3: every H3 model the presets serve, plus the catalog's draft tier, in one process", () => {
    const h3 = preset("h3-family");
    expect(h3.members).toEqual(H3_FAMILY_MEMBERS);
    expect(H3_FAMILY_MEMBERS.length).toBeGreaterThanOrEqual(16);
    expect(byId(h3.pool.models!)).toEqual(byId([...modelsOf(H3_FAMILY_MEMBERS), H3_DRAFT]));
    // Every H3 model of the catalog but the one that does not fit the disk (named in left_out).
    const want = CATALOG.models.filter((m) => m.family === "h3" && m.id !== "fasth3-8step-dense").map((m) => m.id).sort();
    expect(h3.pool.models!.map((m) => m.id).sort()).toEqual(want);
    expect(JSON.stringify(h3.left_out)).toMatch(/fasth3-8step-dense/);
    for (const id of ["fasth3", "sol-h3", "fasth3-v2-nvfp4", "fasth3-4step-vsa-nvfp4", "fasth3-trim-8step", "vdn-h3", "h3-larry-turbo", "h3-ref2v-max"]) expect(h3.pool.models!.map((m) => m.id)).toContain(id);
  });
  it("fasth3: every FastH3 release (model ids fasth3*), none of the others", () => {
    const f = preset("fasth3-family");
    expect(byId(f.pool.models!)).toEqual(byId([...modelsOf(FASTH3_FAMILY_MEMBERS), H3_DRAFT]));
    const want = CATALOG.models.filter((m) => m.family === "h3" && m.id.startsWith("fasth3") && m.id !== "fasth3-8step-dense").map((m) => m.id).sort();
    expect(f.pool.models!.map((m) => m.id).sort()).toEqual(want);
    expect(FASTH3_FAMILY_MEMBERS).toContain("h3-turbo");
    expect(FASTH3_FAMILY_MEMBERS).not.toContain("h3-max");
  });
  it("ltx: turbo, pro, guided A2V and Ref2V", () => {
    expect(LTX_FAMILY_MEMBERS).toEqual(["ltx", "ltx-pro", "ltx-a2v", "ltx-ref2v"]);
    expect(byId(preset("ltx-family").pool.models!)).toEqual(byId(modelsOf(LTX_FAMILY_MEMBERS)));
    const want = CATALOG.models.filter((m) => m.family === "ltx2" && !NO_PRESET.has(m.id)).map((m) => m.id).sort();
    expect(preset("ltx-family").pool.models!.map((m) => m.id).sort()).toEqual(want);
  });
  it("streaming: the presets whose variants are causal (SF-Wan, LongLive), on one pod", () => {
    // The variants whose engine recipe is the SF-Wan causal rollout (families/wan.toml `engine.family = "sf_wan"`).
    const wan = readFileSync(join(repo, "families/wan.toml"), "utf8");
    const causal = wan
      .split(/^\[\[variants\]\]$/m)
      .slice(1)
      .filter((chunk) => /^family = "sf_wan"$/m.test(chunk))
      .map((chunk) => /^id = "([^"]+)"/m.exec(chunk)?.[1]);
    expect(causal.length).toBeGreaterThan(0);
    expect(CATALOG.recipes.filter((r) => (r as { stream?: string }).stream === "causal").map((r) => r.id).sort()).toEqual([...new Set(causal)].sort());
    expect(STREAMING_FAMILY_MEMBERS).toEqual(["sfwan", "longlive"]);
    expect(byId(preset("streaming-family").pool.models!)).toEqual(byId(modelsOf(STREAMING_FAMILY_MEMBERS)));
    expect(preset("streaming-family").licence).toMatch(/non-commercial/i);
  });
  it("each family config: one process, the first model resident, every other one swapped in", () => {
    for (const p of FAMILY_PRESETS) {
      const toml = p.pool.config_toml!;
      expect(toml, p.id).toMatch(/^swap = true$/m);
      const bs = blocks(toml);
      expect(bs.map((b) => b.id), p.id).toEqual(p.pool.models!.map((m) => m.id));
      expect(bs[0]!.resident, p.id).toBe("true");
      for (const b of bs.slice(1)) expect(b.resident, `${p.id}: ${b.id}`).toBe("false");
      expect(toml, p.id).toMatch(new RegExp(`^pool = "${p.id}"$`, "m"));
      // The same file as configs/serve (gen-configs.mjs inlines it).
      const file = { "h3-family": "runpod-family-h3.toml", "fasth3-family": "runpod-family-fasth3.toml", "ltx-family": "runpod-family-ltx.toml", "streaming-family": "runpod-family-streaming.toml" }[p.id]!;
      expect(readFileSync(join(repo, "configs/serve", file), "utf8")).toBe(toml);
    }
  });
  it("weights: the union of the members' trees; the disk fits it (Runpod's 500 GB cap); the GPU fits the largest member", () => {
    for (const p of FAMILY_PRESETS) {
      const ms = p.members!.map(preset);
      for (const m of ms) for (const t of m.weights) expect(p.weights, `${p.id}: ${m.id}'s ${t}`).toContain(t);
      expect(new Set(p.weights).size).toBe(p.weights.length);
      expect(p.min_vram_gb, p.id).toBeGreaterThanOrEqual(Math.max(...ms.map((m) => m.min_vram_gb)));
      const plan = weightsPlan({ ...p.pool }, env);
      if (plan.source === "r2") {
        expect(plan.unsupported, p.id).toEqual([]);
        expect(r2DiskGb(env, plan.trees), `${p.id}: ${plan.trees.join(", ")}`).toBeLessThanOrEqual(500);
      }
    }
    // H3: the union without the MXFP8 pre-quantized trees fits (about 396 GB, 485 GB of disk); with them it would not.
    const h3 = preset("h3-family");
    const plan = weightsPlan({ ...h3.pool }, env);
    expect(weightsSource({ ...h3.pool }, env)).toBe("r2");
    expect(r2Bytes(env, plan.trees) / 1e9).toBeGreaterThan(390);
    expect(r2DiskGb(env, plan.trees)).toBeLessThanOrEqual(500);
    const pq = ["h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8", "h3-base/transformer_prequant_sol-h3_mxfp8"];
    expect(r2DiskGb(env, [...plan.trees, ...pq])).toBeGreaterThan(500);
    expect(h3.prequant).toEqual([]);
    expect(h3.min_vram_gb).toBe(96); // VDN-H3
    // FastH3 keeps turbo's pre-quantized trees; LTX is on the EU volume; streaming from R2.
    expect(weightsPlan({ ...preset("fasth3-family").pool }, env).trees).toEqual(expect.arrayContaining(["h3-base/transformer_prequant_4step-vsa_mxfp8"]));
    expect(weightsSource({ ...preset("ltx-family").pool }, env)).toBe("volume");
    expect(weightsSource({ ...preset("streaming-family").pool }, env)).toBe("r2");
  });
  it("routing: every variant's model id goes to the family pool's dispatch family (FV_DISPATCH_MODEL_FAMILIES)", () => {
    const h3 = poolModelFamilies(preset("h3-family").pool);
    expect(Object.keys(h3).sort()).toEqual(preset("h3-family").pool.models!.map((m) => m.id).sort());
    expect(new Set(Object.values(h3))).toEqual(new Set(["h3"]));
    expect(new Set(Object.values(poolModelFamilies(preset("ltx-family").pool)))).toEqual(new Set(["ltx"]));
    expect(poolModelFamilies(preset("streaming-family").pool)).toEqual({ "sfwan21-1.3b": "sfwan", "longlive-1.3b": "sfwan" });
  });
  it("stored specs: the per-variant pools of the old templates still read and validate; every model preset stays addable", () => {
    const old = normalizeSpec({ name: "old-h3", pools: H3_FAMILY_MEMBERS.map((id) => ({ id })) });
    expect(old.pools.map((p) => p.id)).toEqual(H3_FAMILY_MEMBERS);
    for (const p of MODEL_PRESETS) expect(normalizeSpec({ name: "one", pools: [{ id: p.id }] }).pools[0]!.models).toEqual(p.pool.models);
    // A family preset by its id alone.
    expect(normalizeSpec({ name: "fam", pools: [{ id: "h3-family" }] }).pools[0]!.models!.length).toBe(preset("h3-family").pool.models!.length);
  });
  it("every template is a valid spec (at most 16 pools)", () => {
    for (const k of Object.keys(TEMPLATES)) {
      const s = normalizeSpec(defaultSpec("tpl-x", k));
      expect(s.pools.length, k).toBeLessThanOrEqual(16);
      if (TEMPLATES[k]!.pools.length) expect(s.pools.map((p) => p.id)).toEqual(TEMPLATES[k]!.pools);
    }
  });
});
