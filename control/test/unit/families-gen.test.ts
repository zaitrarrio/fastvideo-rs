// The family-manifest generation (plan A4, gen-configs.mjs): families/*.toml parse and derive as the engine resolves
// names, the generated tables are byte-stable against a fixture, the hand-written copies the manifests do not
// replace yet agree with them (crossChecks), and the presets' weights and GPU memory come from them.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import CATALOG from "../../src/cluster/catalog.json";
import { MANIFEST_FAMILIES, MANIFEST_VARIANTS, MODEL_NAMES } from "../../src/families-gen";
import { minVramGib, MODEL_FAMILY_IDS, mustVariant, recipeVariant, variantOf } from "../../src/families";
import { MODEL_PRESETS, POOL_PRESETS, recipeTrees, recipeVramGb } from "../../src/presets";
import { RECIPE_TREES } from "../../src/recipe-trees";
import { EU_TREES } from "../../src/volumes";
import * as GEN from "../../gen-configs.mjs";
import { configModels, crossChecks, DISPATCH_FAMILY, euVolumeTrees, familiesGenTs, familySet, isAuxTree, loadFamilies, parseFamily, TIER_ALIAS } from "../../gen-configs.mjs";

const repo = new URL("../../../", import.meta.url).pathname;
const fixture = readFileSync(join(repo, "control/test/unit/fixtures/family-fixture.toml"), "utf8");

describe("families/*.toml: parse and derive", () => {
  const set = loadFamilies();
  it("three families, every catalog variant, ids and names unique", () => {
    expect(set.families.map((f) => f.id)).toEqual(["h3", "ltx2", "wan"]);
    expect(set.families.map((f) => f.variants.length)).toEqual([19, 5, 8]);
    expect(set.variants.size).toBe(32);
    for (const [id, v] of set.variants) {
      expect(v.trees, id).toEqual([...v.trees].sort());
      expect(v.weight_trees, id).toEqual(v.trees.filter((t) => !isAuxTree(t)));
      expect(v.weight_trees.length, `${id} reads a weight tree`).toBeGreaterThan(0);
      expect(set.names.get(id)).toMatchObject({ variant: id, kind: "id", family: v.family });
    }
  });
  it("the generated tables are what the files say", () => {
    expect(readFileSync(join(repo, "control/src/families-gen.ts"), "utf8")).toBe(familiesGenTs(set));
    expect(MANIFEST_FAMILIES.map((f) => f.id)).toEqual(MODEL_FAMILY_IDS);
    expect(Object.keys(MANIFEST_VARIANTS)).toEqual([...set.variants.keys()]);
    expect(Object.keys(MODEL_NAMES).length).toBe(set.names.size);
    // Ids, served names and the nine tier aliases; never a recipe_name (what the capabilities report, not a config name).
    expect(Object.values(MODEL_NAMES).map((n) => n.kind).sort()).toEqual([...Array(32).fill("id"), ...Array(15).fill("served"), ...Array(9).fill("tier")]);
    for (const v of set.variants.values()) if (v.recipe_name !== v.id && !v.served_names.includes(v.recipe_name)) expect(MODEL_NAMES[v.recipe_name], v.recipe_name).toBeUndefined();
    // Plan A3: the engine resolves exactly the manifest names; no recipe-alias table and no exception list remain.
    expect("ENGINE_RECIPE_ALIASES" in GEN).toBe(false);
    expect("CATALOG_EXCEPTIONS" in GEN).toBe(false);
    expect(MANIFEST_VARIANTS["fasth3-4step-vsa"]!.public_names).toEqual(["MiniMax-H3", "MiniMax-H3-Turbo"]);
    expect(MANIFEST_VARIANTS["h3-ref2v-turbo"]!.tasks).toEqual(["ref2v"]);
  });
  it("a name resolves as the engine does (caps.rs find): tier alias first of its tier, then id, then served name", () => {
    // catalog.json's recipe details state each alias's target; the manifests must agree.
    for (const r of CATALOG.recipes) {
      const tier = /^tier alias: (\S+)/.exec(r.detail)?.[1];
      const alias = /^alias: (\S+)/.exec(r.detail)?.[1];
      if (tier) expect(MODEL_NAMES[r.id], r.id).toEqual({ variant: tier, kind: "tier" });
      else if (alias) expect(recipeVariant(r.id)?.id, r.id).toBe(alias);
      else if (/^catalog model/.test(r.detail)) expect(MODEL_NAMES[r.id], r.id).toEqual({ variant: r.id, kind: "id" });
    }
    for (const f of set.families) for (const [tier, alias] of Object.entries(TIER_ALIAS[f.id as keyof typeof TIER_ALIAS])) {
      const first = f.variants.find((v) => v.tier === tier);
      if (first) expect(variantOf(alias)?.id, alias).toBe(first.id);
      else expect(variantOf(alias), alias).toBeUndefined();
    }
    expect(variantOf("fasth3")).toMatchObject({ id: "fasth3-4step-vsa", family: "h3", tier: "turbo" });
    // A served name since plan A3 (it was a model_from_config special case before).
    expect(MODEL_NAMES["vdn-h3-50step"]).toEqual({ variant: "vdn-h3", kind: "served" });
    expect(recipeVariant("vdn-h3-50step")?.id).toBe("vdn-h3");
    expect(variantOf("sol-h3-4step-engine-ladder"), "a recipe_name is not a config name").toBeUndefined();
    expect(variantOf("wan14b-turbo")?.id).toBe("wan14b-plug-4step");
    // An H3 contract name, not a config name; the v0.1 adapter is its own variant.
    expect(variantOf("lightx2v-turbo-v0.1")).toBeUndefined();
    expect(variantOf("h3-lightx2v-turbo-v01")).toMatchObject({ family: "h3", trees: ["h3-base", "h3-turbo-loras/lightx2v-v0.1"] });
    expect(variantOf("longlive")).toMatchObject({ id: "longlive-1.3b", family: "wan", weight_trees: ["longlive-1.3b-safetensors", "sfwan21-1.3b"] });
    expect(() => mustVariant("nope")).toThrow(/family manifests/);
    expect(minVramGib([mustVariant("h3-turbo"), mustVariant("vdn-h3")])).toBe(96);
  });
  it("the parser refuses what family.rs refuses (for what the generator reads)", () => {
    const base = parseFamily("f", fixture);
    expect(base.variants.map((v) => v.id)).toEqual(["fix-turbo", "fix-draft-tae"]);
    expect(() => parseFamily("f", fixture.replace("schema = 1", "schema = 2"))).toThrow(/schema/);
    expect(() => parseFamily("f", fixture.replace('trees = ["fix-adapter", "fix-base"]', 'trees = ["fix-base", "fix-adapter"]'))).toThrow(/sorted/);
    expect(() => parseFamily("f", fixture.replace('trees = ["fix-adapter", "fix-base"]', 'trees = ["/abs"]'))).toThrow(/relative/);
    expect(() => parseFamily("f", fixture.replace("min_vram_gib = 80", "min_vram_gib = 0"))).toThrow(/min_vram_gib/);
    expect(() => parseFamily("f", fixture.replace('tier = "turbo"', 'tier = "pro"'))).toThrow(/tier/);
    expect(base.variants[0]).toMatchObject({ public_names: ["Fixture-Turbo"], tasks: ["t2v", "i2v"] });
    expect(() => parseFamily("f", fixture.replace('tasks = ["t2v", "i2v"]', "tasks = []"))).toThrow(/serving\.tasks/);
    expect(() => parseFamily("f", fixture.replace('public_names = ["Fixture-Turbo"]', 'public_names = ["X", "X"]'))).toThrow(/public_names/);
    // Across a set: a served name that is another variant's id.
    const other = parseFamily("g", fixture.replace('id = "h3"', 'id = "wan"').replaceAll("fix-", "wix-").replace('served_names = ["fix", "fixture-turbo"]', 'served_names = ["fix-turbo"]'));
    expect(() => familySet([base, other])).toThrow(/fix-turbo is the id of fix-turbo and the served of wix-turbo/);
    expect(() => familySet([base, { ...base, id: "nope" }])).toThrow(/no tier alias/);
  });
  it("byte-stable: the fixture family renders to the committed golden", () => {
    const want = readFileSync(join(repo, "control/test/unit/fixtures/family-fixture.gen.ts.txt"), "utf8");
    expect(familiesGenTs(familySet([parseFamily("fixture", fixture)]))).toBe(want);
  });
});

describe("the hand-written copies agree with the manifests (gen-configs.mjs crossChecks)", () => {
  it("no problem today", () => {
    expect(crossChecks()).toEqual([]);
  });
  it("the LongLive model and the lightx2v v0.1 model are variants (no exception left)", () => {
    for (const k of ["longlive-1.3b", "h3-lightx2v-turbo-v01"]) expect(MODEL_NAMES[k], k).toEqual({ variant: k, kind: "id" });
  });
  it("volumes.ts EU_TREES is volumes.sh FV_EU_VOLUME_TREES", () => {
    expect([...EU_TREES]).toEqual(euVolumeTrees());
  });
  it("scripts/gcp/families.tsv: weight_trees and served are the twin config's models' variants", () => {
    const rows = readFileSync(join(repo, "scripts/gcp/families.tsv"), "utf8")
      .split("\n")
      .filter((l) => l.trim() && !l.startsWith("#"))
      .map((l) => l.split("\t"));
    expect(rows.length).toBe(5);
    for (const r of rows.filter((r) => r[2] !== "-")) {
      const models = configModels(readFileSync(join(repo, "configs/serve", r[2]!), "utf8"));
      const vs = models.map((m) => mustVariant(m.variant));
      for (const [i, m] of models.entries()) expect(m.catalog, r[0]).toBe(vs[i]!.id);
      expect(r[5]!.split(" "), r[0]).toEqual([...new Set(vs.flatMap((v) => v.weight_trees))]);
      expect(r[9]!.split(" "), r[0]).toEqual(models.map((m) => m.id));
      expect(r[3], r[0]).toBe(DISPATCH_FAMILY[vs[0]!.family as keyof typeof DISPATCH_FAMILY]);
    }
  });
  it("scripts/gpu/verify-weights.sh: a variant:<id> cell per manifest variant, listing its trees", () => {
    const list = execFileSync("bash", [join(repo, "scripts/gpu/verify-weights.sh"), "--list"]).toString();
    for (const [id, v] of Object.entries(MANIFEST_VARIANTS)) expect(list, id).toMatch(new RegExp(`^variant:${id.replace(/\./g, "\\.")}\\s+${v.trees.join(" ").replace(/[.]/g, "\\.")}$`, "m"));
    expect(execFileSync("bash", ["-c", `FV_WEIGHTS=/nonexistent bash ${join(repo, "scripts/gpu/verify-weights.sh")} variant:ltx25-distill-sol 2>&1; echo rc=$?`]).toString()).toMatch(/MISSING dir \/nonexistent\/ltx25[\s\S]*weights INCOMPLETE: variant:ltx25-distill-sol[\s\S]*rc=1/);
    expect(execFileSync("bash", ["-c", `bash ${join(repo, "scripts/gpu/verify-weights.sh")} variant:nope 2>&1; echo rc=$?`]).toString()).toMatch(/unknown variant nope[\s\S]*rc=2/);
  });
});

describe("presets from the manifests", () => {
  it("every pool model's recipe is a manifest name of the model's family (or a documented exception), and its id too", () => {
    for (const p of POOL_PRESETS) {
      for (const m of p.pool.models ?? []) {
        const v = recipeVariant(m.recipe);
        expect(v?.family, `${p.id}: ${m.recipe}`).toBe(m.family);
        expect(variantOf(m.id)?.family, `${p.id}: model id ${m.id}`).toBe(m.family);
      }
    }
  });
  it("a model preset's weights are its recipes' trees and its GPU memory the largest variant's", () => {
    for (const p of MODEL_PRESETS) {
      const models = p.pool.models ?? [];
      expect(p.weights, p.id).toEqual(recipeTrees(models));
      expect(p.min_vram_gb, p.id).toBe(recipeVramGb(models));
    }
    expect(MODEL_PRESETS.find((p) => p.id === "vdn-h3")!.min_vram_gb).toBe(96);
    expect(MODEL_PRESETS.find((p) => p.id === "wan")!.weights).toEqual(["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    expect(MODEL_PRESETS.find((p) => p.id === "h3-turbo")!.weights).toEqual(["FastH3-4-step-Preview-v1-LoRA", "h3-base"]);
    expect(MODEL_PRESETS.find((p) => p.id === "longlive")!.weights).toEqual(["longlive-1.3b-safetensors", "sfwan21-1.3b"]);
  });
});
