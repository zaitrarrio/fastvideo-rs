// Every preset's weight trees against what its fv-serve config reads (docs/ops/r2-bench-2026-10-12.md finding 1: the
// h3 presets listed h3-base only, so an r2 worker never got the FastH3 LoRA and the load failed). The config names
// `weights = "${FV_WEIGHTS}/<tree>"` per model; the engine finds more beside it per recipe (recipe-trees.ts), and that
// table is tied to the Rust sources here, so a new adapter or tree layout fails this test instead of a worker.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { BAKED_CONFIGS, IMAGE_CONFIGS } from "../../src/cluster/image-configs";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { POOL_PRESETS } from "../../src/presets";
import { weightsPlan } from "../../src/providers";
import { R2_AUX_TREES } from "../../src/r2weights";
import { R2_TREES } from "../../src/r2-trees";
import { FASTH3_LORA, MXFP8_TREE_GLOB, RECIPE_TREES } from "../../src/recipe-trees";
import { readToml } from "../../src/serverless/serves";
import { SLS_PRESETS } from "../../src/serverless/presets";
import { slsFetchTrees, slsTrees } from "../../src/serverless/weights";
import { missingTrees } from "../../src/volumes";
import { AUX_FILES, HUB_TREES } from "../../src/weights-sources";
import type { Env } from "../../src/env";

const ROOT = new URL("../../../", import.meta.url).pathname;
const rust = (p: string) => readFileSync(`${ROOT}${p}`, "utf8");
const configOf = (p: (typeof POOL_PRESETS)[number]) => {
  if (p.pool.config_toml) return p.pool.config_toml;
  const f = (p.pool.config || IMAGE_CONFIGS[p.pool.variant]!.default).split("/").pop()!;
  return BAKED_CONFIGS[f] || WORKER_CONFIGS[f] || readFileSync(`${ROOT}configs/serve/${f}`, "utf8");
};
/** A config path's tree: the first segment under the weights root, or a published nested tree by its full name. */
const treeOf = (path: string) => {
  const m = /(?:\$\{FV_WEIGHTS\}|\/workspace\/weights|\/runpod-volume\/weights)\/([A-Za-z0-9._-]+)(?:\/([A-Za-z0-9._-]+))?/.exec(path);
  if (!m) return null;
  const nested = m[2] ? `${m[1]}/${m[2]}` : "";
  return nested && (HUB_TREES[nested] || R2_TREES[nested]) ? nested : m[1]!;
};
/** What the preset's config reads: its models' weights (and LongLive paths) plus each recipe's own trees. */
function expected(p: (typeof POOL_PRESETS)[number]) {
  const cfg = readToml(configOf(p));
  const required = new Set<string>();
  const prequant = new Set<string>();
  for (const m of cfg.models as Record<string, unknown>[]) {
    for (const k of ["weights", "longlive"]) if (typeof m[k] === "string") required.add(treeOf(m[k] as string)!);
    const r = RECIPE_TREES[String(m.recipe)];
    expect(r, `${p.id}: recipe ${String(m.recipe)} is not in recipe-trees.ts: add what the engine reads for it`).toBeTruthy();
    r!.required.forEach((t) => required.add(t));
    r!.prequant.forEach((t) => prequant.add(t));
  }
  return { required: [...required].sort(), prequant: [...prequant].sort() };
}

describe("preset trees: each preset lists every tree its config reads", () => {
  for (const p of POOL_PRESETS) {
    it(p.id, () => {
      const e = expected(p);
      expect([...p.weights].sort(), `${p.id}: weights`).toEqual(e.required);
      expect([...(p.prequant ?? [])].sort(), `${p.id}: prequant`).toEqual(e.prequant);
      for (const t of [...p.weights, ...(p.prequant ?? [])]) {
        expect(R2_TREES[t], `${p.id}: ${t} is published in the R2 weights store`).toBeTruthy();
        expect(missingTrees("jg48s6o1w0", [t]), `${p.id}: ${t} is on the EU volume`).toEqual([]);
      }
      // a nested tree sits inside a tree of the same preset (the fetcher places it there, as on the volume)
      for (const t of p.prequant ?? []) expect(p.weights).toContain(t.split("/")[0]);
    });
  }
  it("the h3 turbo and max presets carry the FastH3 LoRA (vsa-datafree / dense-datafree)", () => {
    for (const id of ["h3-turbo", "h3-max"]) expect(POOL_PRESETS.find((p) => p.id === id)!.weights).toContain(FASTH3_LORA);
  });
});

describe("recipe-trees.ts against the Rust sources", () => {
  const lora = rust("crates/fastvideo-models/src/h3/lora.rs");
  const caps = rust("crates/fastvideo-engine-service/src/cuda/caps.rs");
  it("the FastH3 adapters are looked for beside the base, both variants", () => {
    expect(lora).toContain(`"${FASTH3_LORA}/vsa-datafree/adapter_model.safetensors"`);
    expect(lora).toContain(`"${FASTH3_LORA}/dense-datafree/adapter_model.safetensors"`);
    expect(lora).toMatch(/"4step-vsa" \| "preview-vsa" \| "fasth3-4step-vsa" => Some\(Self::Vsa\)/);
  });
  it("the tier aliases resolve to the catalog entries the table names, on h3-base", () => {
    expect(caps).toContain('h3(layout, "h3-base", "4step-vsa", Some("h3/fasth3_4step_vsa"), 4)');
    expect(caps).toContain('h3(layout, "h3-base", "sol-h3", Some(SOL_H3_4STEP_PROFILE), 4)');
    expect(caps).toContain('ref_weights: Some(layout.at("h3-ref2va"))');
    for (const [recipe, r] of Object.entries(RECIPE_TREES)) expect(caps, `${recipe} -> ${r.catalog}`).toContain(`"${r.catalog}"`);
    expect(caps).toContain('pub const LTX25_DEV_DIT: &str = "ltx25-dev/transformer_full"');
    expect(rust("crates/fastvideo-models/src/ltx2/lora.rs")).toContain('pub const LTX25_INGREDIENTS_DIR: &str = "ltx25-ic-lora-ingredients"');
  });
  it("the pre-quantized trees are named as the loaders look for them", () => {
    expect(rust("crates/fastvideo-cudarc/src/llm/prequant.rs")).toContain('pub const DIR_NAME: &str = "text_encoder_fp8"');
    expect(rust("crates/fastvideo-cudarc/src/h3/dit_tree.rs")).toContain('format!("transformer_prequant_{recipe}_{}", quant.as_str())');
    for (const r of Object.values(RECIPE_TREES))
      for (const t of r.prequant) expect(t, "a prequant tree is text_encoder_fp8 or transformer_prequant_<recipe>_<quant>").toMatch(/^[a-z0-9-]+\/(text_encoder_fp8|transformer_prequant_[a-z0-9-]+_(mxfp8|w8a8))$/);
    // the MXFP8 DiT trees only help on sm_100+: QuantMode::for_device runs W8A8 below that (the boot skips them)
    expect(rust("crates/fastvideo-cudarc/src/wan/quant.rs")).toMatch(/if d\.sm_major < 10 \{/);
    expect(MXFP8_TREE_GLOB).toBe("*_mxfp8");
  });
});

describe("the trees a boot fetches", () => {
  const env = {} as Env;
  it("serverless r2: the preset's weights, its pre-quantized trees and the aux trees", () => {
    const p = SLS_PRESETS.find((x) => x.id === "h3-turbo")!;
    const spec = { variant: p.variant, compute: "GPU" as const, preset: p.id, image: { channel: "stable" } };
    expect(slsTrees(spec)).toEqual(["FastH3-4-step-Preview-v1-LoRA", "h3-base", "h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"]);
    expect(slsFetchTrees(env, spec)).toEqual(["FastH3-4-step-Preview-v1-LoRA", "auxiliary/lpips", "auxiliary/tae", "h3-base", "h3-base/text_encoder_fp8", "h3-base/transformer_prequant_4step-vsa_mxfp8"]);
    expect(slsFetchTrees({ WEIGHTS_R2_AUX: "0" } as Env, spec)).toEqual(slsTrees(spec));
    expect(slsFetchTrees(env, { variant: "cpu", compute: "CPU", preset: "cpu", image: { channel: "stable" } })).toEqual([]);
  });
  it("pods: r2 adds the pre-quantized and aux trees; the Hub gets only what it has (no prequant)", () => {
    const pool = { ...POOL_PRESETS.find((x) => x.id === "h3-max")!.pool, provider: "brev" as const, provider_gpu: "g7e.2xlarge" };
    const r2 = weightsPlan({ ...pool, weights_source: "r2" }, env);
    expect(r2.unsupported).toEqual([]);
    expect([...r2.trees].sort()).toEqual(["FastH3-4-step-Preview-v1-LoRA", "auxiliary/lpips", "auxiliary/tae", "h3-base", "h3-base/text_encoder_fp8", "h3-base/transformer_prequant_sol-h3_mxfp8"]);
    const hub = weightsPlan({ ...pool, weights_source: "hub" }, env);
    expect(hub.unsupported).toEqual([]);
    expect([...hub.trees].sort()).toEqual(["FastH3-4-step-Preview-v1-LoRA", "h3-base"]);
  });
  it("the aux trees in R2 hold exactly the aux files (same paths, same bytes)", () => {
    for (const t of R2_AUX_TREES) {
      const files = AUX_FILES.filter((a) => a.path.startsWith(`${t}/`));
      expect(R2_TREES[t]!.files, t).toBe(files.length);
      expect(R2_TREES[t]!.bytes, t).toBe(files.reduce((s, a) => s + a.size, 0));
    }
    expect(AUX_FILES.every((a) => R2_AUX_TREES.some((t) => a.path.startsWith(`${t}/`)))).toBe(true);
  });
});
