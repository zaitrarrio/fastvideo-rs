// Model-first serverless endpoints (docs/control/serverless.md §1a): a preset
// → the spec and Runpod payloads (queue and load balancer, the inline config
// delivered in both), the checks before create (GPU memory, weights on the
// volume, the volume's data centre, a config fv-control can send: the images
// bake none), what an endpoint serves, and the test invokes generated from it.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { POOL_PRESETS } from "../../src/cluster/spec";
import { LEGACY_VARIANT_CONFIGS, RUNPOD_GPU_TYPES, SLS_PRESET_IDS } from "../../src/enums";
import { GPU_MEMORY_GB } from "../../src/gpus";
import { examplesFor, fillExample } from "../../src/serverless/examples";
import { SLS_BOOT, templateCreatePayload, templateUpdatePayload, v2CreatePayload, v2TemplateBoot, workerEnv } from "../../src/serverless/payloads";
import { inferPreset, mergeSpec, servingIssues, servingView, SLS_PRESETS } from "../../src/serverless/presets";
import { compareCapabilities, readToml, servesOf, servesOfToml } from "../../src/serverless/serves";
import { defaultEndpointSpec, normalizeEndpointSpec } from "../../src/serverless/spec";
import { missingTrees, VOLUME_TREES } from "../../src/volumes";

const IMG = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "a".repeat(64);
const EU = "bz0ofkfx45";
const ROOT = new URL("../../../", import.meta.url).pathname;
const REF2V = readFileSync(`${ROOT}configs/serve/runpod-h3-ref2v.toml`, "utf8");
const b64 = (s: string) => Buffer.from(s, "base64").toString();
const paths = (r: { issues: { path: (string | number)[] }[] }) => r.issues.map((i) => i.path.join("."));

describe("presets: one catalog", () => {
  it("the serverless presets are the pool presets plus cpu, in the schema's enum", () => {
    expect(SLS_PRESETS.map((p) => p.id).sort()).toEqual([...POOL_PRESETS.map((p) => p.id), "cpu"].sort());
    expect([...SLS_PRESET_IDS].sort()).toEqual(SLS_PRESETS.map((p) => p.id).sort());
  });
  it("every preset makes a valid spec in both modes (GPU presets: lb too) with no serving issue on the EU volume", () => {
    for (const p of SLS_PRESETS) {
      for (const mode of p.compute === "CPU" ? (["queue"] as const) : (["queue", "lb"] as const)) {
        const s = normalizeEndpointSpec({ name: "t", preset: p.id, mode });
        expect(s, `${p.id} ${mode}`).toMatchObject({ preset: p.id, compute: p.compute, mode });
        expect("variant" in s, p.id).toBe(false);
        expect(servingIssues(s).issues, `${p.id} ${mode}`).toEqual([]);
        // The 500 GB EU volume holds H3 + LTX only (2026-10-13): other presets default to the R2 weights store.
        if (p.compute === "GPU" && missingTrees(EU, p.weights)!.length) expect(s, p.id).toMatchObject({ weights_source: "r2", network_volume: null });
        else if (p.compute === "GPU") expect(s, p.id).toMatchObject({ weights_source: "volume", network_volume: EU, data_centers: ["EU-RO-1"] });
      }
    }
  });
  it("every preset sends its config inline: the standard ones too (the one image bakes none)", () => {
    const max = normalizeEndpointSpec({ name: "m", preset: "h3-max" });
    expect(max.config).toBeUndefined();
    expect(max.config_toml).toBe(WORKER_CONFIGS["runpod-h3-max.toml"]);
    expect(v2TemplateBoot(max).dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(templateCreatePayload(max, IMG, "t").dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(b64(templateCreatePayload(max, IMG, "t").env.FV_WORKER_TOML_B64!)).toBe(WORKER_CONFIGS["runpod-h3-max.toml"]);
    const r = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" });
    expect(r).toMatchObject({ compute: "GPU", config_toml: WORKER_CONFIGS["runpod-h3-ref2v.toml"], execution_timeout_s: 3600 });
    expect(normalizeEndpointSpec({ name: "a", preset: "ltx-a2v" }).container_disk_gb).toBe(60);
  });
  it("the preset owns compute and config: repeating them is fine, changing them is refused; an old variant is ignored", () => {
    expect(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", variant: "h3-max", compute: "GPU" }).preset).toBe("h3-ref2v");
    expect(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", variant: "ltx" }).preset).toBe("h3-ref2v");
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", compute: "CPU" })).toThrow(/compute: set by preset h3-ref2v/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-max", config: "/etc/fv/runpod-fake.toml" })).toThrow(/config: set by preset h3-max/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", config_toml: "[engine]\n" })).toThrow(/config_toml: set by preset/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "nope" })).toThrow(/preset/);
  });
  it("defaults for a preset (GET /api/v1/serverless/defaults?preset=)", () => {
    expect(defaultEndpointSpec("x", "CPU", "cpu")).toMatchObject({ preset: "cpu", compute: "CPU", config: "runpod-fake.toml", network_volume: null });
    expect(defaultEndpointSpec("x", "CPU", "h3-ref2v")).toMatchObject({ preset: "h3-ref2v", compute: "GPU", network_volume: EU });
  });
});

describe("payloads: the inline config in both modes", () => {
  it("queue (REST v1 template): FV_WORKER_TOML_B64 is the ref2v config, the boot decodes it", () => {
    const s = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" });
    const t = templateCreatePayload(s, IMG, "fvc-r-x");
    expect(b64(t.env.FV_WORKER_TOML_B64!)).toBe(REF2V);
    expect(t.env.FV_SERVE_MODE).toBe("runpod-queue");
    expect(t.dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    expect(templateUpdatePayload(s, IMG).dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
  });
  it("load balancer (REST v2): the env carries it and the template gets the boot after create (no longer refused)", () => {
    const s = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", mode: "lb" });
    expect(s.scaler_type).toBe("REQUEST_COUNT");
    const p = v2CreatePayload(s, IMG, ["BLACKWELL_96"]) as any;
    expect(p.type).toBe("LOAD_BALANCER");
    expect(b64(p.env.FV_WORKER_TOML_B64)).toBe(REF2V);
    expect(p.env.FV_SERVE_MODE).toBe("http");
    expect(p.args).toBeUndefined();
    expect(v2TemplateBoot(s)!.dockerEntrypoint).toEqual(["/bin/sh", "-c", SLS_BOOT]);
    // A custom inline config on a load balancer validates too.
    const c = normalizeEndpointSpec({ name: "c", variant: "ltx", mode: "lb", config_toml: WORKER_CONFIGS["runpod-ltx-pro.toml"] });
    expect(workerEnv(c, IMG).FV_WORKER_TOML_B64).toBeTruthy();
    expect(inferPreset(c)).toBe("ltx-pro");
  });
  it("the boot runs fv-serve with the decoded config (the same start in both modes)", async () => {
    const { execFileSync } = await import("node:child_process");
    const boot = SLS_BOOT.replaceAll("/opt/fastvideo-rs/bin/fv-serve", "echo fv-serve").replaceAll("/fv-worker.toml", "/tmp/fvc-sls-test.toml").replace("mkdir -p /fvstate", ":");
    const out = execFileSync("sh", ["-c", boot], { env: { PATH: process.env.PATH!, FV_WORKER_TOML_B64: Buffer.from(REF2V).toString("base64") } }).toString();
    expect(out.trim()).toBe("fv-serve --config /tmp/fvc-sls-test.toml");
    expect(readFileSync("/tmp/fvc-sls-test.toml", "utf8")).toBe(REF2V);
    // Without a config the boot stops (the images bake none): never fv-serve's default.
    let failed = "";
    try {
      execFileSync("sh", ["-c", boot], { env: { PATH: process.env.PATH! }, stdio: "pipe" });
    } catch (e) {
      failed = String((e as { stderr?: Buffer }).stderr ?? "");
    }
    expect(failed).toContain("FAILED: no worker config");
  });
});

describe("validation before create", () => {
  it("GPU memory: a 5090 or a 48 GB card cannot run h3-ref2v (field-level, per GPU type)", () => {
    const s = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA GeForce RTX 5090", "NVIDIA L40S"] });
    const r = servingIssues(s);
    expect(paths(r)).toEqual(["gpu_types.1", "gpu_types.2"]);
    expect(r.issues[0]!.message).toMatch(/RTX 5090 has 32 GB; preset h3-ref2v needs a GPU with at least 80 GB/);
    expect(paths(servingIssues(normalizeEndpointSpec({ name: "w", preset: "fastwan21", gpu_types: ["NVIDIA GeForce RTX 5090"] })))).toEqual([]);
    // A custom spec: the families' needs (an H3 model: 80 GB).
    expect(paths(servingIssues(normalizeEndpointSpec({ name: "c", variant: "h3-turbo", gpu_types: ["NVIDIA A40"] })))).toEqual(["gpu_types.0"]);
  });
  it("weights: a tree the volume lacks is refused, with the source of the list", () => {
    const toml = `[engine]\nbackend = "cuda"\n[[models]]\nvariant = "wan-max"\nid = "x"\nweights = "\${FV_WEIGHTS}/not-on-the-volume"\n`;
    const r = servingIssues(normalizeEndpointSpec({ name: "c", variant: "wan5b", config_toml: toml }));
    expect(paths(r)).toEqual(["network_volume"]);
    expect(r.issues[0]!.message).toMatch(/no weights tree not-on-the-volume .*volumes\.sh FV_EU_VOLUME_TREES/);
    // No volume at all: a preset is refused by the schema, a custom config that loads weights by the serving check.
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", network_volume: null })).toThrow(/network_volume: preset h3-ref2v loads weights/);
    const nv = servingIssues(normalizeEndpointSpec({ name: "c", variant: "h3-turbo", network_volume: null }));
    expect(nv.issues[0]).toMatchObject({ path: ["network_volume"], message: expect.stringMatching(/loads weight trees \(FastH3-4-step-Preview-v1-LoRA, h3-base\)/) });
  });
  it("data centre: a worker with the EU volume runs in EU-RO-1 only", () => {
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", data_centers: ["US-CA-2"] })).toThrow(/must run in EU-RO-1/);
  });
  it("a named config is any configs/serve file fv-control carries (the incident's companion config works on the one image)", () => {
    const companion = normalizeEndpointSpec({ name: "c", compute: "GPU", config: "/etc/fv/runpod-h3-ref2v.toml" });
    expect(paths(servingIssues(companion))).toEqual([]);
    expect(inferPreset(companion)).toBe("h3-ref2v");
    expect(b64(workerEnv(companion, IMG).FV_WORKER_TOML_B64!)).toBe(REF2V);
    expect(paths(servingIssues(normalizeEndpointSpec({ name: "c", compute: "GPU", config: "runpod-wan14b.toml", weights_source: "r2" })))).toEqual([]);
    // A name fv-control does not carry: refused before create, with an image ref too (no image carries configs now).
    expect(() => normalizeEndpointSpec({ name: "c", compute: "GPU", config: "/etc/fv/x.toml", image: { ref: IMG } })).toThrow(/config: not a worker config fv-control carries/);
    expect(paths(servingIssues({ mode: "queue", compute: "GPU", config: "x.toml", image: { channel: "stable" }, network_volume: EU }))).toContain("config");
    expect(paths(servingIssues({ mode: "queue", compute: "GPU", image: { channel: "stable" }, network_volume: EU }))).toContain("config");
  });
  it("licence: the LongLive preset warns", () => {
    expect(servingIssues(normalizeEndpointSpec({ name: "l", preset: "longlive" })).warnings[0]!.message).toMatch(/non-commercial/i);
  });
  it("an update merges: a preset switch drops the old preset's config; a variant or config without a preset is custom", () => {
    const prev = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" });
    const sw = normalizeEndpointSpec(mergeSpec(prev, { preset: "h3-max" }));
    expect(sw).toMatchObject({ preset: "h3-max", compute: "GPU", config_toml: WORKER_CONFIGS["runpod-h3-max.toml"] });
    const custom = mergeSpec(prev, { config_toml: "[engine]\nbackend = \"fake\"\n" });
    expect(custom.preset).toBeUndefined();
    expect(mergeSpec(prev, { workers_max: 2 })).toMatchObject({ preset: "h3-ref2v", config_toml: prev.config_toml, workers_max: 2 });
    expect(mergeSpec(prev, { preset: null }).preset).toBeUndefined();
    // A stored spec from before the one image (a variant, no config): the merge reads it as its compute and config.
    const old = mergeSpec({ name: "o", mode: "queue", variant: "ltx", compute: "GPU", image: { channel: "stable" } }, { workers_max: 2 });
    expect(old).toMatchObject({ compute: "GPU", config: "runpod-ltx.toml", workers_max: 2 });
    expect("variant" in old).toBe(false);
  });
});

describe("what an endpoint serves", () => {
  it("reads the serve configs' TOML", () => {
    const t = readToml(`# c\n[engine]\nswap = true # x\n[[models]]\nid = "a"\nweights = "\${FV_WEIGHTS}/t"\n[[models]]\nid = "b"\n[aliases]\n"MiniMax-H3" = "a"\n[protocols]\nfal_apps = ["x/y", "z"]\nnative = false\n`);
    expect(t).toEqual({ engine: { swap: true }, models: [{ id: "a", weights: "${FV_WEIGHTS}/t" }, { id: "b" }], aliases: { "MiniMax-H3": "a" }, protocols: { fal_apps: ["x/y", "z"], native: false } });
  });
  it("h3-ref2v: the two Ref2VA models, ref2v only, MiniMax names to them, swap, weights the variants' trees", () => {
    const v = servingView(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" }));
    expect(v.preset).toBe("h3-ref2v");
    expect(v.serves.models.map((m) => [m.id, m.tier, m.tasks.join(","), m.resident])).toEqual([
      ["h3-ref2v-turbo", "turbo", "ref2v", true],
      ["h3-ref2v-max", "max", "ref2v", false],
    ]);
    // The variants' public names (families/h3.toml), as fv-serve maps them (CapabilityTable::public_aliases).
    expect(v.serves.aliases).toEqual(expect.arrayContaining([{ name: "MiniMax-H3-Turbo", model: "h3-ref2v-turbo", via: "manifest" }, { name: "MiniMax-H3-Max", model: "h3-ref2v-max", via: "manifest" }]));
    expect(v.serves).toMatchObject({ swap: true, engine: "cuda", source: { kind: "inline" }, weights: ["h3-base", "h3-ref2va"] });
    expect(v.serves.protocols).toEqual(expect.arrayContaining(["native", "minimax", "fal", "openai_videos"]));
    expect(v.serves.protocols).not.toContain("ltx");
    expect(v.serves.fal_apps).toContain("minimax/h3-turbo");
  });
  it("a [[models]] block names a variant: id, family, tier, tasks, served and public names from the manifests", () => {
    const toml = `[engine]\nbackend = "cuda"\nswap = true\n[[models]]\nvariant = "h3-turbo"\nid = "fasth3"\nresident = true\n[[models]]\nvariant = "h3-ref2v-turbo"\nresident = false\n[[models]]\nvariant = "longlive"\n`;
    const s = servesOfToml(toml, { kind: "inline", config: null });
    expect(s.models.map((m) => [m.id, m.family, m.recipe, m.tier, m.tasks.join(","), m.resident])).toEqual([
      ["fasth3", "h3", "h3-turbo", "turbo", "t2v,i2v,keyframes", true],
      ["h3-ref2v-turbo", "h3", "h3-ref2v-turbo", "turbo", "ref2v", false],
      ["longlive-1.3b", "wan", "longlive", null, "stream", false],
    ]);
    expect(s.models[0]!.served_names).toEqual(["fasth3", "fasth3-4step-vsa"]);
    expect(s.models[2]!.served_names).toEqual(["longlive-1.3b", "longlive"]);
    expect(s.models[2]!.weights).toEqual(["longlive-1.3b-safetensors", "sfwan21-1.3b"]);
    // MiniMax-H3-Turbo is claimed by the turbo model and its Ref2V companion: the one serving t2v gets it.
    const manifest = s.aliases.filter((a) => a.via === "manifest");
    expect(manifest).toEqual([
      { name: "MiniMax-H3", model: "fasth3", via: "manifest" },
      { name: "MiniMax-H3-Turbo", model: "fasth3", via: "manifest" },
    ]);
    // A config without [[models]] on the fake engine: its built-in models and their public names.
    const fake = servesOfToml(`[engine]\nbackend = "fake"\n`, { kind: "inline", config: null });
    expect(fake.aliases.filter((a) => a.via === "manifest").map((a) => [a.name, a.model])).toEqual([
      ["MiniMax-H3", "fake-h3-turbo"],
      ["MiniMax-H3-Max", "fake-h3-max"],
      ["MiniMax-H3-Turbo", "fake-h3-turbo"],
      ["fastwan-1.3b", "fake-wan"],
    ]);
  });
  it("the incident: a stored h3-max endpoint with no config is preset h3-max (inferred), serving sol-h3 only; MiniMax-H3-Turbo is not a name it answers", () => {
    // As stored before the one image: the variant, and the config its image baked; it now names that config.
    const spec = normalizeEndpointSpec({ name: "ref2va", variant: "h3-max", mode: "lb" });
    const v = servingView(spec);
    expect(v).toMatchObject({ preset: "h3-max", preset_inferred: true });
    expect(v.serves.source).toMatchObject({ kind: "named", config: "runpod-h3-max.toml" });
    expect(v.serves.models.map((m) => m.id)).toEqual(["sol-h3"]);
    expect(v.serves.tasks).not.toContain("ref2v");
    expect(v.serves.aliases.map((a) => a.name).sort()).toEqual(["MiniMax-H3", "MiniMax-H3-Max"]);
  });
  it("every preset derives models (a custom spec with an unknown config derives none, and says so)", () => {
    for (const p of SLS_PRESETS) expect(servesOf(normalizeEndpointSpec({ name: "t", preset: p.id })).models.length, p.id).toBeGreaterThan(0);
    expect(servesOf({ config: "/etc/fv/nope.toml" }).source).toMatchObject({ kind: "unknown" });
    expect(servesOf({}).source).toMatchObject({ kind: "unknown", note: expect.stringMatching(/no worker config/) });
    expect(inferPreset({ compute: "GPU", config: "/etc/fv/nope.toml" })).toBeNull();
    expect(inferPreset({ compute: "GPU", config: "runpod-wan.toml" })).toBe("fastwan21");
    expect(inferPreset({ compute: "CPU", config: "runpod-fake.toml" })).toBe("cpu");
  });
  it("a worker's capabilities against the derived view: a match, and the incident's mismatch", () => {
    const s = servesOf(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" }));
    const caps = (ids: string[]) => ({ status: 200, body: { object: "fv.capabilities", models: ids.map((id) => ({ caps: { id } })) } });
    expect(compareCapabilities(s, caps(["h3-ref2v-turbo", "h3-ref2v-max"]))).toMatchObject({ ok: true });
    expect(compareCapabilities(s, JSON.stringify(caps(["sol-h3"])))).toMatchObject({ ok: false, missing: ["h3-ref2v-turbo", "h3-ref2v-max"], extra: ["sol-h3"] });
    expect(compareCapabilities(s, { nothing: 1 }).ok).toBe(false);
  });
  it("the configs the old variant images baked (stored specs' read-compat) are configs fv-control carries", () => {
    for (const [v, cfg] of Object.entries(LEGACY_VARIANT_CONFIGS)) expect(WORKER_CONFIGS[cfg], `${v}: ${cfg}`).toBeTruthy();
  });
});

describe("test invokes generated from what it serves", () => {
  const ROUTE = /^\/[A-Za-z0-9._~\/?=&%-]{0,300}$/; // the invoke route's lb path check (ops.ts invoke)
  it("every preset, both modes: info / ping first, capabilities, then per model × task × API, correctly shaped", () => {
    for (const p of SLS_PRESETS)
      for (const mode of ["queue", "lb"] as const) {
        const spec = { ...normalizeEndpointSpec({ name: "t", preset: p.id }), mode };
        const ex = examplesFor(servesOf(spec), mode);
        expect(ex[0]!.id, p.id).toBe(mode === "queue" ? "info" : "ping");
        expect(ex[1]!.id).toBe("capabilities");
        expect(new Set(ex.map((e) => e.id)).size).toBe(ex.length);
        for (const e of ex) {
          if (mode === "lb") expect(String(e.invoke.path), e.label).toMatch(ROUTE);
          else if (e.id !== "info") expect(e.invoke.input, e.label).toMatchObject({ kind: "http", path: expect.stringMatching(/^\//) });
        }
        if (!["sfwan", "longlive", "streaming-family"].includes(p.id)) expect(ex.length, `${p.id} has job examples`).toBeGreaterThan(2);
      }
  });
  it("h3-ref2v: MiniMax V2 ref2v Turbo 768P 5 s (ref2v.py t_minimax), fal reference-to-video, an image URL to fill", () => {
    const s = servesOf(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" }));
    const q = examplesFor(s, "queue");
    const mm = q.find((e) => e.api === "minimax" && e.model === "h3-ref2v-turbo")!;
    expect(mm.label).toMatch(/MiniMax V2 · ref2v · Turbo \(h3-ref2v-turbo\) · MiniMax-H3-Turbo, 768P 5 s/);
    expect(mm.needs).toEqual(["image_url"]);
    expect(mm.invoke).toEqual({
      input: {
        kind: "http",
        method: "POST",
        path: "/v2/video_generation",
        wait: true,
        body: { model: "MiniMax-H3-Turbo", resolution: "768P", duration: 5, content: [{ type: "text", text: expect.stringContaining("Picture 1") }, { type: "image_url", image_url: { url: "{{image_url}}" }, role: "reference_image" }] },
      },
    });
    const filled = fillExample(mm.invoke, { image_url: "https://example.com/a.jpg" }) as any;
    expect(filled.input.body.content[1].image_url.url).toBe("https://example.com/a.jpg");
    const fal = q.find((e) => e.api === "fal" && e.model === "h3-ref2v-turbo")!;
    expect(fal.invoke.input).toMatchObject({ path: "/minimax/h3-turbo/reference-to-video", body: { reference_image_urls: ["{{image_url}}"], duration: 5, resolution: "768P" } });
    // Ref2V models serve no text-to-video: no native t2v example for them.
    expect(q.some((e) => e.api === "native" && e.task === "t2v")).toBe(false);
    const lb = examplesFor(s, "lb").find((e) => e.api === "minimax" && e.model === "h3-ref2v-turbo")!;
    expect(lb.invoke).toMatchObject({ method: "POST", path: "/v2/video_generation", body: { model: "MiniMax-H3-Turbo" } });
    expect(lb.poll).toBe("/v2/query/video_generation/{id}");
  });
  it("h3-max offers no MiniMax-H3-Turbo request; ltx-ref2v the fal ingredient; ltx-a2v an audio URL; cpu a fake native job", () => {
    const ex = (id: string) => examplesFor(servesOf(normalizeEndpointSpec({ name: "t", preset: id })), "queue");
    expect(JSON.stringify(ex("h3-max"))).not.toContain("MiniMax-H3-Turbo");
    expect(ex("h3-max").find((e) => e.api === "minimax" && e.task === "t2v")!.invoke).toMatchObject({ input: { body: { model: "MiniMax-H3-Max" } } });
    expect(ex("ltx-ref2v").find((e) => e.task === "ref2v")!.invoke).toMatchObject({ input: { path: "/fal-ai/ltx-2.3-quality/ingredient" } });
    expect(ex("ltx-a2v").find((e) => e.task === "a2v")!.needs).toEqual(["audio_url"]);
    expect(ex("cpu").find((e) => e.api === "native" && e.model === "fake-wan")!.invoke).toMatchObject({ input: { kind: "http", path: "/fv/v1/jobs", body: { model: "fake-wan" }, wait: true } });
    expect(ex("wan").find((e) => e.api === "fal" && e.model === "fastwan22-ti2v-5b" && e.task === "t2v")!.invoke).toMatchObject({ input: { path: "/fal-ai/wan/v2.2-5b/text-to-video/fast-wan" } });
  });
});

describe("static data", () => {
  it("every Runpod GPU type has its memory", () => {
    expect(Object.keys(GPU_MEMORY_GB).sort()).toEqual([...RUNPOD_GPU_TYPES].sort());
  });
  it("the EU volume's tree list is volumes.sh FV_EU_VOLUME_TREES; every H3 / LTX preset's trees are on it", () => {
    const eu = VOLUME_TREES[EU]!;
    const sh = /FV_EU_VOLUME_TREES="([^"]*)"/.exec(readFileSync(`${ROOT}scripts/gpu/volumes.sh`, "utf8"))![1]!.replace(/\\\n/g, " ").split(/\s+/).filter(Boolean);
    expect([...eu].sort()).toEqual(sh.sort());
    const rows = new Set(readFileSync(`${ROOT}scripts/gpu/weights-r2.tsv`, "utf8").split("\n").filter((l) => l && !l.startsWith("#")).map((l) => l.split("\t")[0]!));
    for (const t of eu) expect(rows.has(t), `${t} is published in R2 (r2-fill-volume.sh filled it from there)`).toBe(true);
    // The Turbo LoRA presets' adapters are in R2 only (docs/ports/h3-turbo-loras.md); their base trees are on the volume.
    // h3-family carries every H3 tree, the R2-only ones (NVFP4, Trim, VDN) included: it defaults to r2.
    const r2Only = (t: string) => t.startsWith("h3-turbo-loras/");
    for (const p of SLS_PRESETS) if (/^(h3|ltx)/.test(p.id) && p.id !== "h3-family") for (const t of [...p.weights, ...p.prequant].filter((t) => !r2Only(t))) expect(missingTrees(EU, [t]), `${p.id}: ${t}`).toEqual([]);
  });
});
