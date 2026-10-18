// The pool templates derived from the presets and the serve catalog (owner 2026-10-18): h3 (every H3 variant),
// fasth3 (every FastH3 variant), streaming (every causal streaming model).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import CATALOG from "../../src/cluster/catalog.json";
import { defaultSpec, FASTH3_TEMPLATE_POOLS, H3_TEMPLATE_POOLS, normalizeSpec, POOL_PRESETS, STREAMING_TEMPLATE_POOLS, TEMPLATES } from "../../src/cluster/spec";

const repo = new URL("../../../", import.meta.url).pathname;
/** Catalog models no preset serves (no worker config in configs/serve runs them as a pool yet). */
const NO_PRESET = new Set(["fasth3-4step-vsa-480p-taeh3", "fasth3-8step-dense", "ltx25-distill-sol-nvfp4-taehv", "fastwan22-ti2v-5b-taehv", "wan14b-plug-4step"]);
const presetModels = (ids: string[]) => ids.flatMap((id) => (POOL_PRESETS.find((p) => p.id === id)!.pool.models || []).map((m) => m.id));

describe("templates", () => {
  it("h3: every H3 model of the catalog that a preset serves, one pool per preset", () => {
    const want = CATALOG.models.filter((m) => m.family === "h3" && !NO_PRESET.has(m.id)).map((m) => m.id).sort();
    expect(presetModels(H3_TEMPLATE_POOLS).sort()).toEqual(want);
    for (const id of ["h3-turbo", "h3-max", "fasth3-v2-nvfp4", "fasth3-4step-nvfp4", "h3-turbo-preview", "fasth3-trim", "vdn-h3", "h3-larry-turbo", "h3-ref2v"]) expect(H3_TEMPLATE_POOLS).toContain(id);
    expect(TEMPLATES.h3!.pools).toBe(H3_TEMPLATE_POOLS);
  });
  it("fasth3: every FastH3 release (model ids fasth3*), none of the others", () => {
    const want = CATALOG.models.filter((m) => m.family === "h3" && m.id.startsWith("fasth3") && !NO_PRESET.has(m.id)).map((m) => m.id).sort();
    expect(presetModels(FASTH3_TEMPLATE_POOLS).sort()).toEqual(want);
    expect(FASTH3_TEMPLATE_POOLS).toContain("h3-turbo");
    expect(FASTH3_TEMPLATE_POOLS).not.toContain("h3-max");
    expect(FASTH3_TEMPLATE_POOLS).not.toContain("vdn-h3");
  });
  it("streaming: the presets whose recipes caps.rs builds as causal (SF-Wan, LongLive)", () => {
    // The catalog's stream marks against caps.rs: the recipes of the CudaRecipe::SfWan models.
    const caps = readFileSync(join(repo, "crates/fastvideo-engine-service/src/cuda/caps.rs"), "utf8");
    const causal = caps
      .split("CudaModel::new(")
      .slice(1)
      .filter((chunk) => chunk.includes("CudaRecipe::SfWan("))
      .map((chunk) => /^\s*"([^"]+)"/.exec(chunk)?.[1]);
    expect(causal.length).toBeGreaterThan(0);
    expect(CATALOG.recipes.filter((r) => (r as { stream?: string }).stream === "causal").map((r) => r.id).sort()).toEqual([...new Set(causal)].sort());
    expect(STREAMING_TEMPLATE_POOLS).toEqual(["sfwan", "longlive"]);
  });
  it("every template is a valid spec (at most 16 pools)", () => {
    for (const k of Object.keys(TEMPLATES)) {
      const s = normalizeSpec(defaultSpec("tpl-x", k));
      expect(s.pools.length, k).toBeLessThanOrEqual(16);
      if (TEMPLATES[k]!.pools.length) expect(s.pools.map((p) => p.id)).toEqual(TEMPLATES[k]!.pools);
    }
  });
});
