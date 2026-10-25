// The deployment data (plan C1, deploy/deploy.toml, docs/adr/0011-deploy-section.md): it parses against the family
// manifests, refuses what would deploy wrongly (an unknown variant, a recipe naming another variant, a card below its
// variants' floor, a volume other than the EU one), and fv-control's presets and enums are exactly what it says.
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { DEPLOY_FAMILY_PRESETS, DEPLOY_IMAGES, DEPLOY_PRESETS } from "../../src/deploy-gen";
import { POOL_PRESET_IDS, SERVE_IMAGES } from "../../src/enums";
import { FAMILY_PRESETS, MODEL_PRESETS, STANDARD_POOLS } from "../../src/presets";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { autoscalePoolsText, deployChecks, DEPLOY_FILE, EU_VOLUME, loadDeploy, loadFamilies, parseDeploy, serveStages, workerConfigNames } from "../../gen-configs.mjs";
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
  it("parses: 2 images (one CUDA image, the cpu image), 24 model presets, 4 family presets, 5 GCP rows, 4 autoscale pools", () => {
    // One image with boot-time config (owner decision 2026-10-25): no per-variant images, no baked config.
    expect(dep.images).toEqual([
      { id: "cuda", carries: ["runpod-fake.toml"], compute: "gpu", note: null },
      { id: "cpu", carries: ["runpod-fake.toml"], compute: "cpu", note: "CPU only, fake engine; `gateway` until 2026-10-06" },
    ]);
    expect(new Set([...dep.presets, ...dep.family_presets].map((p) => p.image))).toEqual(new Set(["cuda"]));
    expect(dep.presets.length).toBe(24);
    expect(dep.presets.filter((p) => p.standard).map((p) => p.id)).toEqual(["h3-turbo", "h3-max", "ltx", "wan"]);
    expect(dep.family_presets.map((p) => p.id)).toEqual(["h3-family", "fasth3-family", "ltx-family", "streaming-family"]);
    expect(dep.gcp.map((r) => r.name)).toEqual(["h3-turbo", "h3-max", "ltx", "wan", "fake"]);
    expect(dep.autoscale_pools.map((p) => p.name)).toEqual(["h3-turbo", "wan", "ltx", "sfwan-live"]);
  });
  it("agrees with what it is kept beside (deployChecks)", () => {
    expect(deployChecks(set)).toEqual([]);
  });
  it("fv-control reads it unchanged: images, enums, presets, configs (every preset's sent inline)", () => {
    expect(DEPLOY_IMAGES.map((i) => i.id)).toEqual(["cuda", "cpu"]);
    expect(SERVE_IMAGES).toEqual({ GPU: "cuda", CPU: "cpu" });
    expect([...POOL_PRESET_IDS]).toEqual([...MODEL_PRESETS, ...FAMILY_PRESETS].map((p) => p.id));
    expect(DEPLOY_PRESETS.map((p) => p.id)).toEqual(MODEL_PRESETS.map((p) => p.id));
    expect(DEPLOY_FAMILY_PRESETS.map((p) => p.members)).toEqual(FAMILY_PRESETS.map((p) => p.members));
    expect(STANDARD_POOLS.map((p) => p.id)).toEqual(["h3-turbo", "h3-max", "ltx", "wan"]);
    for (const p of MODEL_PRESETS) {
      const d = DEPLOY_PRESETS.find((x) => x.id === p.id)!;
      expect("variant" in p.pool, p.id).toBe(false);
      expect(p.pool.compute, p.id).toBe("GPU");
      expect(p.pool.models, p.id).toEqual(d.models.map(({ id, family, recipe }) => ({ id, family, recipe })));
      // Owned by name (sent as its text at launch, so an edited config reaches the next launch).
      expect(p.pool.config, p.id).toBe(d.config);
      expect(p.pool.config_toml, p.id).toBeUndefined();
      expect(WORKER_CONFIGS[d.config], p.id).toBeTruthy();
    }
    // Every Runpod worker config, the presets' first: fv-control sends any of them at boot.
    expect(Object.keys(WORKER_CONFIGS)).toEqual(workerConfigNames(dep));
    expect(Object.keys(WORKER_CONFIGS)).toEqual(expect.arrayContaining(["runpod-fake.toml", "runpod-wan14b.toml", "runpod.toml"]));
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
    expect(edited('image = "cuda"\nconfig = "runpod-ltx-pro.toml"', 'image = "ltx"\nconfig = "runpod-ltx-pro.toml"')).toThrow(/not the GPU image/);
    expect(edited('image = "cuda"\nconfig = "runpod-ltx-pro.toml"', 'image = "cpu"\nconfig = "runpod-ltx-pro.toml"')).toThrow(/not the GPU image/);
    expect(edited('{ variant = "ltx25-ref2v" }', '{ variant = "ltx25-ref2v-x" }')).toThrow(/not a variant id/);
    // A served name is not a variant id: the canonical id is required.
    expect(edited('{ variant = "fasth3-trim-8step", recipe = "h3-trim" }', '{ variant = "h3-trim" }')).toThrow(/not a variant id/);
  });
  it("one CUDA image and at most one CPU image, each carrying the smoke config, none baking one", () => {
    expect(edited('[[images]]\nid = "cpu"', '[[images]]\nid = "h3-max"\ncarries = ["runpod-fake.toml"]\n\n[[images]]\nid = "cpu"')).toThrow(/2 GPU images/);
    expect(edited('id = "cuda"\ncarries = ["runpod-fake.toml"]', 'id = "cuda"\ncarries = ["runpod.toml"]')).toThrow(/carries runpod-fake.toml/);
    expect(edited('id = "cuda"\ncarries', 'id = "cuda"\nconfig = "runpod.toml"\ncarries')).toThrow(/config: unknown key/);
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

describe("the hand-written serve stages are held to the data (gen-configs.mjs on a copy of the tree)", () => {
  // Everything the generator reads or writes, copied; control/node_modules linked.
  const PATHS = ["control/gen-configs.mjs", "control/package.json", "control/src", "families", "deploy", "configs", "docker", "docs/serve/images.md", "scripts/gpu", "scripts/serve/variants.sh", "scripts/gcp", ".github/workflows/serve-image.yml", "crates/fastvideo-serve/console"];
  it("the committed stages: serve-cuda and serve-cpu carry the smoke config, bake no FV_CONFIG, name their image", () => {
    const st = serveStages();
    expect(Object.keys(st).sort()).toEqual(["cpu", "cuda"]);
    for (const id of ["cuda", "cpu"]) {
      expect(st[id].files, id).toEqual(["runpod-fake.toml"]);
      expect(st[id].env, id).toEqual({ FV_VARIANT: id });
    }
  });
  it("a baked config, a config the data does not list, or a per-variant stage fails --check", () => {
    const root = mkdtempSync(join(tmpdir(), "c4-gen-"));
    try {
      for (const p of PATHS) {
        mkdirSync(dirname(join(root, p)), { recursive: true });
        cpSync(join(repo, p), join(root, p), { recursive: true });
      }
      symlinkSync(join(repo, "control/node_modules"), join(root, "control/node_modules"), "dir");
      const gen = (...args: string[]) => execFileSync(process.execPath, ["gen-configs.mjs", ...args], { cwd: join(root, "control"), encoding: "utf8", stdio: "pipe" });
      gen("--check");
      const dfPath = join(root, "docker/gpucheck.Dockerfile");
      const df = readFileSync(dfPath, "utf8");
      const fails = (edit: string, why: RegExp) => {
        writeFileSync(dfPath, edit);
        let err = "";
        try {
          gen("--check");
        } catch (e) {
          err = String((e as { stderr?: string }).stderr ?? e);
        }
        expect(err, why.source).toMatch(why);
        writeFileSync(dfPath, df);
      };
      expect(df.includes("ENV FV_VARIANT=cuda\n")).toBe(true);
      fails(df.replace("ENV FV_VARIANT=cuda\n", "ENV FV_VARIANT=cuda FV_CONFIG=/etc/fv/runpod.toml\n"), /serve-cuda: bakes FV_CONFIG=\/etc\/fv\/runpod.toml/);
      fails(df.replace("COPY configs/serve/runpod-fake.toml /etc/fv/\nENV FV_VARIANT=cuda", "COPY configs/serve/runpod-fake.toml configs/serve/runpod.toml /etc/fv/\nENV FV_VARIANT=cuda"), /serve-cuda: \/etc\/fv runpod-fake.toml runpod.toml/);
      fails(`${df}\nFROM serve-cuda AS serve-ltx\nENV FV_VARIANT=ltx\n`, /serve-ltx is not an image of deploy\/deploy.toml/);
      gen("--check");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }, 60000);
});
