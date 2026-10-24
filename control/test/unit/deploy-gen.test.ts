// The deployment data (plan C1, deploy/deploy.toml, docs/adr/0011-deploy-section.md): it parses against the family
// manifests, refuses what would deploy wrongly (an unknown variant, a recipe naming another variant, a card below its
// variants' floor, a volume other than the EU one), and fv-control's presets and enums are exactly what it says.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DEPLOY_FAMILY_PRESETS, DEPLOY_IMAGES, DEPLOY_PRESETS } from "../../src/deploy-gen";
import { POOL_PRESET_IDS, VARIANTS } from "../../src/enums";
import { FAMILY_PRESETS, MODEL_PRESETS, STANDARD_POOLS } from "../../src/presets";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { autoscalePoolsText, deployChecks, DEPLOY_FILE, EU_VOLUME, inlineWorkerConfigs, loadDeploy, loadFamilies, parseDeploy } from "../../gen-configs.mjs";
import { parse as parseToml } from "smol-toml";

const repo = new URL("../../../", import.meta.url).pathname;
const text = readFileSync(DEPLOY_FILE, "utf8");
const set = loadFamilies();
/** deploy/deploy.toml with one substitution (which must apply). */
const edited = (from: string, to: string) => {
  expect(text.includes(from), from).toBe(true);
  return () => parseDeploy(text.replace(from, to), set);
};

describe("deploy/deploy.toml", () => {
  const dep = loadDeploy(set);
  it("parses: 8 images (one legacy), 24 model presets, 4 family presets, 5 GCP rows, 4 autoscale pools", () => {
    expect(dep.images.map((i) => i.id)).toEqual(["h3-turbo", "h3-max", "ltx", "wan", "wan5b", "sfwan", "cpu", "debug"]);
    expect(dep.images.filter((i) => i.legacy).map((i) => i.id)).toEqual(["debug"]);
    expect(dep.presets.length).toBe(24);
    expect(dep.presets.filter((p) => p.standard).map((p) => p.id)).toEqual(["h3-turbo", "h3-max", "ltx", "wan"]);
    expect(dep.family_presets.map((p) => p.id)).toEqual(["h3-family", "fasth3-family", "ltx-family", "streaming-family"]);
    expect(dep.gcp.map((r) => r.name)).toEqual(["h3-turbo", "h3-max", "ltx", "wan", "fake"]);
    expect(dep.autoscale_pools.map((p) => p.name)).toEqual(["h3-turbo", "wan", "ltx", "sfwan-live"]);
  });
  it("agrees with what it is kept beside (deployChecks)", () => {
    expect(deployChecks(set)).toEqual([]);
  });
  it("fv-control reads it unchanged: images, enums, presets, inline configs", () => {
    expect(DEPLOY_IMAGES.map((i) => i.id)).toEqual([...VARIANTS]);
    expect([...POOL_PRESET_IDS]).toEqual([...MODEL_PRESETS, ...FAMILY_PRESETS].map((p) => p.id));
    expect(DEPLOY_PRESETS.map((p) => p.id)).toEqual(MODEL_PRESETS.map((p) => p.id));
    expect(DEPLOY_FAMILY_PRESETS.map((p) => p.members)).toEqual(FAMILY_PRESETS.map((p) => p.members));
    expect(STANDARD_POOLS.map((p) => p.id)).toEqual(["h3-turbo", "h3-max", "ltx", "wan"]);
    for (const p of MODEL_PRESETS) {
      const d = DEPLOY_PRESETS.find((x) => x.id === p.id)!;
      expect(p.pool.variant, p.id).toBe(d.image);
      expect(p.pool.models, p.id).toEqual(d.models.map(({ id, family, recipe }) => ({ id, family, recipe })));
      expect(p.pool.config_toml ?? p.pool.config, p.id).toBe(d.inline ? WORKER_CONFIGS[d.config] : `/etc/fv/${d.config}`);
    }
    expect(Object.keys(WORKER_CONFIGS)).toEqual(inlineWorkerConfigs(dep));
  });
  it("the autoscale pools are copied verbatim into configs/autoscale.toml", () => {
    const auto = readFileSync(join(repo, "configs/autoscale.toml"), "utf8");
    expect(auto).toContain(autoscalePoolsText(text));
    expect(parseToml(auto).autoscale).toMatchObject({ pools: parseToml(text).autoscale!.pools });
  });
});

describe("deploy/deploy.toml refusals", () => {
  it("unknown keys, images and variants", () => {
    expect(edited('id = "h3-turbo"\ntitle', 'id = "h3-turbo"\ntitel = "x"\ntitle')).toThrow(/titel: unknown key/);
    expect(edited('image = "ltx"\nconfig = "runpod-ltx-pro.toml"', 'image = "ltx-2"\nconfig = "runpod-ltx-pro.toml"')).toThrow(/not a GPU image variant/);
    expect(edited('image = "ltx"\nconfig = "runpod-ltx-pro.toml"', 'image = "cpu"\nconfig = "runpod-ltx-pro.toml"')).toThrow(/not a GPU image variant/);
    expect(edited('{ variant = "ltx25-ref2v" }', '{ variant = "ltx25-ref2v-x" }')).toThrow(/not a variant id/);
    // A served name is not a variant id: the canonical id is required.
    expect(edited('{ variant = "fasth3-trim-8step", recipe = "h3-trim" }', '{ variant = "h3-trim" }')).toThrow(/not a variant id/);
  });
  it("a recipe must name the model's variant", () => {
    expect(edited('{ variant = "sol-h3", recipe = "h3-max" }', '{ variant = "sol-h3", recipe = "h3-turbo" }')).toThrow(/does not resolve to sol-h3/);
  });
  it("ids are unique across model and family presets", () => {
    expect(edited('id = "streaming-family"', 'id = "sfwan"')).toThrow(/preset sfwan twice/);
  });
  it("a family preset's card meets its variants' floor and serves one family", () => {
    expect(edited("min_vram_gb = 96\nmembers = [\"h3-turbo\", \"h3-max\"", "min_vram_gb = 48\nmembers = [\"h3-turbo\", \"h3-max\"")).toThrow(/below its variants' gpu.min_vram_gib/);
    expect(edited('members = ["sfwan", "longlive"]', 'members = ["sfwan", "ltx"]')).toThrow(/one model family per process/);
    expect(edited('members = ["sfwan", "longlive"]', 'members = ["sfwan", "nope"]')).toThrow(/nope is not a model preset/);
  });
  it("pre-quantized trees are nested in the preset's weight trees", () => {
    expect(edited('prequant = ["h3-base/text_encoder_fp8"]\nmodels = [{ variant = "fasth3-trim-8step"', 'prequant = ["ltx25/text_encoder_fp8"]\nmodels = [{ variant = "fasth3-trim-8step"')).toThrow(/not nested/);
  });
  it("only the EU weights volume (CLAUDE.md)", () => {
    expect(EU_VOLUME).toEqual({ region: "eu", data_center: "EU-RO-1", volume_id: "bz0ofkfx45" });
    expect(edited('{ data_center = "EU-RO-1", volume_id = "bz0ofkfx45" }', '{ data_center = "US-CA-2", volume_id = "s2k01690bi" }')).toThrow(/only weights volume is bz0ofkfx45/);
    expect(edited('{ data_center = "EU-RO-1", volume_id = "bz0ofkfx45" }', '{ data_center = "US-CA-2", volume_id = "bz0ofkfx45" }')).toThrow(/only weights volume/);
  });
  it("dispatch names every family; GCP and autoscale families are dispatch families", () => {
    expect(edited('wan = "wan"\n', "")).toThrow(/dispatch names/);
    expect(edited('dispatch_family = "ltx"', 'dispatch_family = "ltx2"')).toThrow(/dispatch_family ltx2/);
    expect(edited('name = "sfwan-live"\nfamily = "wan"', 'name = "sfwan-live"\nfamily = "sfwan"')).toThrow(/a dispatch family/);
  });
  it("a GCP row names served ids only without a Runpod twin", () => {
    expect(edited('served = ["fake-h3-turbo"]\n', "")).toThrow(/served/);
  });
});
