// Model-first serverless endpoints (docs/control/serverless.md §1a): a preset
// → the spec and Runpod payloads (queue and load balancer, the inline config
// delivered in both), the checks before create (GPU memory, weights on the
// volume, the volume's data centre, a config the image carries), what an
// endpoint serves, and the test invokes generated from it.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { IMAGE_CONFIGS } from "../../src/cluster/image-configs";
import { WORKER_CONFIGS } from "../../src/cluster/worker-configs";
import { POOL_PRESETS } from "../../src/cluster/spec";
import { RUNPOD_GPU_TYPES, SLS_PRESET_IDS } from "../../src/enums";
import { GPU_MEMORY_GB } from "../../src/gpus";
import { examplesFor, fillExample } from "../../src/serverless/examples";
import { SLS_BOOT, templateCreatePayload, templateUpdatePayload, v2CreatePayload, v2TemplateBoot, workerEnv } from "../../src/serverless/payloads";
import { inferPreset, mergeSpec, servingIssues, servingView, SLS_PRESETS } from "../../src/serverless/presets";
import { compareCapabilities, readToml, servesOf } from "../../src/serverless/serves";
import { defaultEndpointSpec, normalizeEndpointSpec } from "../../src/serverless/spec";
import { VOLUME_TREES } from "../../src/volumes";

const IMG = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "a".repeat(64);
const EU = "jg48s6o1w0";
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
        expect(s, `${p.id} ${mode}`).toMatchObject({ preset: p.id, variant: p.variant, compute: p.compute, mode });
        expect(servingIssues(s).issues, `${p.id} ${mode}`).toEqual([]);
        if (p.compute === "GPU") expect(s).toMatchObject({ network_volume: EU, data_centers: ["EUR-IS-1"] });
      }
    }
  });
  it("an image-default config stays unset (the image's entrypoint runs it); a config the image lacks rides inline", () => {
    const max = normalizeEndpointSpec({ name: "m", preset: "h3-max" });
    expect(max.config).toBeUndefined();
    expect(max.config_toml).toBeUndefined();
    expect(v2TemplateBoot(max)).toBeNull();
    expect(templateCreatePayload(max, IMG, "t").dockerEntrypoint).toBeUndefined();
    const r = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" });
    expect(r).toMatchObject({ variant: "h3-max", config_toml: WORKER_CONFIGS["runpod-h3-ref2v.toml"], execution_timeout_s: 3600 });
    expect(normalizeEndpointSpec({ name: "a", preset: "ltx-a2v" }).container_disk_gb).toBe(60);
  });
  it("the preset owns variant, compute and config: repeating them is fine, changing them is refused", () => {
    expect(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", variant: "h3-max", compute: "GPU" }).preset).toBe("h3-ref2v");
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", variant: "ltx" })).toThrow(/variant: set by preset h3-ref2v/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-max", config: "/etc/fv/runpod-fake.toml" })).toThrow(/config: set by preset h3-max/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", config_toml: "[engine]\n" })).toThrow(/config_toml: set by preset/);
    expect(() => normalizeEndpointSpec({ name: "r", preset: "nope" })).toThrow(/preset/);
  });
  it("defaults for a preset (GET /api/serverless/defaults?preset=)", () => {
    expect(defaultEndpointSpec("x", "cpu", "cpu")).toMatchObject({ preset: "cpu", variant: "cpu", compute: "CPU", network_volume: null });
    expect(defaultEndpointSpec("x", "cpu", "h3-ref2v")).toMatchObject({ preset: "h3-ref2v", variant: "h3-max", compute: "GPU", network_volume: EU });
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
    const toml = `[engine]\nbackend = "cuda"\n[[models]]\nid = "x"\nfamily = "wan"\nrecipe = "wan-max"\nweights = "\${FV_WEIGHTS}/not-on-the-volume"\n`;
    const r = servingIssues(normalizeEndpointSpec({ name: "c", variant: "wan5b", config_toml: toml }));
    expect(paths(r)).toEqual(["network_volume"]);
    expect(r.issues[0]!.message).toMatch(/no weights tree not-on-the-volume .*weights-manifest\.tsv/);
    // No volume at all: a preset is refused by the schema, a custom config that loads weights by the serving check.
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", network_volume: null })).toThrow(/network_volume: preset h3-ref2v loads weights/);
    const nv = servingIssues(normalizeEndpointSpec({ name: "c", variant: "h3-turbo", network_volume: null }));
    expect(nv.issues[0]).toMatchObject({ path: ["network_volume"], message: expect.stringMatching(/loads weight trees \(h3-base\)/) });
  });
  it("data centre: a worker with the EU volume runs in EUR-IS-1 only", () => {
    expect(() => normalizeEndpointSpec({ name: "r", preset: "h3-ref2v", data_centers: ["US-CA-2"] })).toThrow(/must run in EUR-IS-1/);
  });
  it("a config path must be in the variant's image (the incident's companion config is not in h3-max)", () => {
    const r = servingIssues(normalizeEndpointSpec({ name: "c", variant: "h3-max", config: "/etc/fv/runpod-h3-ref2v.toml" }));
    expect(paths(r)).toContain("config");
    expect(r.issues.find((i) => i.path[0] === "config")!.message).toMatch(/carries \/etc\/fv\/runpod-h3-max\.toml, \/etc\/fv\/runpod-fake\.toml; .* not in it \(preset h3-ref2v sends that config inline\)/);
    expect(paths(servingIssues(normalizeEndpointSpec({ name: "c", variant: "wan5b", config: "/etc/fv/runpod-wan14b.toml" })))).toEqual([]);
    // An image ref: not checkable, a warning.
    const ref = servingIssues(normalizeEndpointSpec({ name: "c", variant: "h3-max", config: "/etc/fv/x.toml", image: { ref: IMG } }));
    expect(ref.warnings.map((w) => w.path.join("."))).toContain("config");
  });
  it("licence: the LongLive preset warns", () => {
    expect(servingIssues(normalizeEndpointSpec({ name: "l", preset: "longlive" })).warnings[0]!.message).toMatch(/non-commercial/i);
  });
  it("an update merges: a preset switch drops the old preset's config; a variant or config without a preset is custom", () => {
    const prev = normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" });
    const sw = normalizeEndpointSpec(mergeSpec(prev, { preset: "h3-max" }));
    expect(sw).toMatchObject({ preset: "h3-max", variant: "h3-max" });
    expect(sw.config_toml).toBeUndefined();
    const custom = mergeSpec(prev, { config_toml: "[engine]\nbackend = \"fake\"\n" });
    expect(custom.preset).toBeUndefined();
    expect(mergeSpec(prev, { workers_max: 2 })).toMatchObject({ preset: "h3-ref2v", config_toml: prev.config_toml, workers_max: 2 });
    expect(mergeSpec(prev, { preset: null }).preset).toBeUndefined();
  });
});

describe("what an endpoint serves", () => {
  it("reads the serve configs' TOML", () => {
    const t = readToml(`# c\n[engine]\nswap = true # x\n[[models]]\nid = "a"\nweights = "\${FV_WEIGHTS}/t"\n[[models]]\nid = "b"\n[aliases]\n"MiniMax-H3" = "a"\n[protocols]\nfal_apps = ["x/y", "z"]\nnative = false\n`);
    expect(t).toEqual({ engine: { swap: true }, models: [{ id: "a", weights: "${FV_WEIGHTS}/t" }, { id: "b" }], aliases: { "MiniMax-H3": "a" }, protocols: { fal_apps: ["x/y", "z"], native: false } });
  });
  it("h3-ref2v: the two Ref2VA models, ref2v only, MiniMax names to them, swap, weights h3-base (+ h3-ref2va from the preset)", () => {
    const v = servingView(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" }));
    expect(v.preset).toBe("h3-ref2v");
    expect(v.serves.models.map((m) => [m.id, m.tier, m.tasks.join(","), m.resident])).toEqual([
      ["h3-ref2v-turbo", "turbo", "ref2v", true],
      ["h3-ref2v-max", "max", "ref2v", false],
    ]);
    expect(v.serves.aliases).toEqual(expect.arrayContaining([{ name: "MiniMax-H3-Turbo", model: "h3-ref2v-turbo", via: "config" }, { name: "MiniMax-H3-Max", model: "h3-ref2v-max", via: "config" }]));
    expect(v.serves).toMatchObject({ swap: true, engine: "cuda", source: { kind: "inline" }, weights: ["h3-base"] });
    expect(v.serves.protocols).toEqual(expect.arrayContaining(["native", "minimax", "fal", "openai_videos"]));
    expect(v.serves.protocols).not.toContain("ltx");
    expect(v.serves.fal_apps).toContain("minimax/h3-turbo");
  });
  it("the incident: h3-max with no config is preset h3-max (inferred), serving sol-h3 only; MiniMax-H3-Turbo is not a name it answers", () => {
    const spec = normalizeEndpointSpec({ name: "ref2va", variant: "h3-max", mode: "lb" });
    const v = servingView(spec);
    expect(v).toMatchObject({ preset: "h3-max", preset_inferred: true });
    expect(v.serves.source).toMatchObject({ kind: "image-default", config: "/etc/fv/runpod-h3-max.toml" });
    expect(v.serves.models.map((m) => m.id)).toEqual(["sol-h3"]);
    expect(v.serves.tasks).not.toContain("ref2v");
    expect(v.serves.aliases.map((a) => a.name).sort()).toEqual(["MiniMax-H3", "MiniMax-H3-Max"]);
  });
  it("every preset derives models (a custom spec with an unknown config derives none, and says so)", () => {
    for (const p of SLS_PRESETS) expect(servesOf(normalizeEndpointSpec({ name: "t", preset: p.id })).models.length, p.id).toBeGreaterThan(0);
    expect(servesOf({ variant: "wan", config: "/etc/fv/nope.toml" }).source).toMatchObject({ kind: "unknown" });
    expect(inferPreset({ variant: "wan", compute: "GPU", config: "/etc/fv/nope.toml" })).toBeNull();
    expect(inferPreset({ variant: "wan", compute: "GPU" })).toBe("fastwan21");
    expect(inferPreset({ variant: "cpu", compute: "CPU" })).toBe("cpu");
  });
  it("a worker's capabilities against the derived view: a match, and the incident's mismatch", () => {
    const s = servesOf(normalizeEndpointSpec({ name: "r", preset: "h3-ref2v" }));
    const caps = (ids: string[]) => ({ status: 200, body: { object: "fv.capabilities", models: ids.map((id) => ({ caps: { id } })) } });
    expect(compareCapabilities(s, caps(["h3-ref2v-turbo", "h3-ref2v-max"]))).toMatchObject({ ok: true });
    expect(compareCapabilities(s, JSON.stringify(caps(["sol-h3"])))).toMatchObject({ ok: false, missing: ["h3-ref2v-turbo", "h3-ref2v-max"], extra: ["sol-h3"] });
    expect(compareCapabilities(s, { nothing: 1 }).ok).toBe(false);
  });
  it("the generated image map agrees with scripts/serve/variants.sh", () => {
    const sh = readFileSync(`${ROOT}scripts/serve/variants.sh`, "utf8");
    for (const [v, img] of Object.entries(IMAGE_CONFIGS)) {
      const m = new RegExp(`^    ${v.replace("-", "\\-")}\\) echo (\\S+) ;;$`, "m").exec(sh);
      expect(m?.[1], v).toBe(img.default.replace("/etc/fv/", ""));
    }
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
        if (!["sfwan", "longlive"].includes(p.id)) expect(ex.length, `${p.id} has job examples`).toBeGreaterThan(2);
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
  it("the EU volume's tree list has every manifest row and every preset's trees", () => {
    const eu = VOLUME_TREES[EU]!;
    const rows = readFileSync(`${ROOT}scripts/gpu/weights-manifest.tsv`, "utf8").split("\n").filter((l) => l && !l.startsWith("#")).map((l) => l.split("\t")[0]!.split("/")[0]!);
    for (const r of rows) expect(eu, r).toContain(r);
    for (const p of SLS_PRESETS) for (const t of p.weights) expect(eu, `${p.id}: ${t}`).toContain(t);
  });
});
