// weights_source hub for serverless endpoints (owner 2026-10-14: "get the weights and use flashboot"; src/serverless/
// weights.ts slsHubPlan / slsHubEnv / SLS_HUB_SH, spec.ts, payloads.ts): the same two-key approval as pods (the spec's
// weights_download_approved and fv-control's FV_HUB_DOWNLOADS_APPROVED=1), no volume, any data centre, a container disk
// sized from the trees, FlashBoot on; the template (the bash Hub boot, the pinned trees, the recorded hashes); the
// outlook; and the worker's boot run in bash with a stand-in for fetch-hub-tree.py: a cold boot downloads, checks the
// recorded SHA-256s and reports its breakdown; a second boot on the same disk is a cache hit; a file that is not the
// recorded one fails the boot before fv-serve, and its tree is removed.
import { createHash } from "node:crypto";
import { execFile, execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { Env } from "../../src/env";
import { HUB_FETCH_TREE_SH } from "../../src/providers";
import { endpointCreatePayload, SLS_HUB_BOOT, templateCreatePayload, templateUpdatePayload, v2CreatePayload, v2TemplateBoot } from "../../src/serverless/payloads";
import { defaultEndpointSpec, normalizeEndpointSpec, placementIssues, withWeightsSource } from "../../src/serverless/spec";
import { applyReport, HUB_DOWNLOAD_GBPS, SLS_HUB_SH, slsHubDiskGb, slsHubEnv, slsHubPlan, slsOutlook, slsWeightsIssues, type BootRow } from "../../src/serverless/weights";
import { HUB_SHA256, HUB_TREES } from "../../src/weights-sources";

const H100 = "NVIDIA H100 80GB HBM3";
const hub = (over: Record<string, unknown> = {}) => normalizeEndpointSpec({ name: "hw", preset: "wan", weights_source: "hub", weights_download_approved: true, gpu_types: [H100], ...over });
const issuesOf = (f: () => unknown) => {
  try {
    f();
    return [];
  } catch (e: any) {
    return (e.extra?.issues ?? []).map((i: any) => `${i.path.join(".")}: ${i.message}`);
  }
};
const APPROVED = { FV_HUB_DOWNLOADS_APPROVED: "1", PUBLIC_URL: "https://fvc.test", GITHUB_REPO: "zaitrarrio/fastvideo-rs" } as unknown as Env;

describe("the spec: weights_source hub", () => {
  it("no volume, any data centre, FlashBoot on, no exact CUDA list, a container disk sized from the trees", () => {
    const s = hub();
    expect([s.weights_source, s.network_volume, s.data_centers, s.flashboot, s.weights_download_approved, s.allowed_cuda]).toEqual(["hub", null, undefined, true, true, []]);
    const { trees } = slsHubPlan(s);
    expect(trees).toEqual(["fastwan22-ti2v-5b", "wan22-ti2v-5b"]);
    expect(s.container_disk_gb).toBe(slsHubDiskGb(trees));
    expect(s.container_disk_gb).toBeGreaterThan(60);
    // The defaults route's placement for hub (the launch form asks for it when Weights changes).
    const d = withWeightsSource(defaultEndpointSpec("x", "ltx", "ltx"), "hub");
    expect([d.network_volume, d.data_centers, d.flashboot, d.weights_download_approved]).toEqual([null, undefined, true, undefined]);
  });
  it("without the spec's approval: refused at weights_download_approved", () => {
    expect(issuesOf(() => hub({ weights_download_approved: undefined }))).toEqual([expect.stringMatching(/^weights_download_approved: weights_source hub downloads fastwan22-ti2v-5b, wan22-ti2v-5b from the Hub at every cold start: it needs the owner's approval for this endpoint \(weights_download_approved: true\)$/)]);
  });
  it("refuses the weights volume, FlashBoot off, a disk too small, CPU workers; the approval only with hub", () => {
    expect(issuesOf(() => hub({ network_volume: "bz0ofkfx45" }))).toContainEqual(expect.stringMatching(/^network_volume: weights_source hub downloads onto the container disk: network_volume null/));
    expect(issuesOf(() => hub({ flashboot: false }))).toEqual([expect.stringMatching(/^flashboot: weights_source hub downloads the weights at every cold start: FlashBoot on/)]);
    expect(issuesOf(() => hub({ container_disk_gb: 40 }))).toEqual([expect.stringMatching(/^container_disk_gb: weights_source hub downloads fastwan22-ti2v-5b, wan22-ti2v-5b onto the container disk: at least \d+ GB$/)]);
    expect(issuesOf(() => normalizeEndpointSpec({ name: "c", variant: "cpu", weights_source: "hub", weights_download_approved: true }))).toContainEqual(expect.stringMatching(/^weights_source: weights_source hub is for GPU workers/));
    expect(issuesOf(() => normalizeEndpointSpec({ name: "w", preset: "wan", weights_source: "r2", weights_download_approved: true }))).toEqual(["weights_download_approved: only with weights_source hub"]);
    // Over 200 GB of disk is fine with hub (the trees are on it).
    expect(hub({ container_disk_gb: 300 }).container_disk_gb).toBe(300);
  });
  it("placement: any data centre, the stock per GPU type", async () => {
    const r = await placementIssues(hub(), async (pairs) => new Map(pairs.map((p) => [`${p.dc}|${p.gpu}`, { stock: p.dc === "US-TX-3" ? "High" : null, max_available: null }])));
    expect(r.issues).toEqual([]);
    expect(r.warnings.map((w) => w.message)).toEqual([expect.stringMatching(/^weights_source hub, any data centre: in stock H100 80GB HBM3 in US-TX-3$/)]);
  });
});

describe("fv-control's checks: the owner's second key, the trees' Hub sources", () => {
  it("without FV_HUB_DOWNLOADS_APPROVED: refused; with it: a warning with the download", () => {
    expect(slsWeightsIssues({} as Env, hub()).issues.map((i) => `${i.path.join(".")}: ${i.message}`)).toEqual([expect.stringMatching(/^weights_download_approved: Hub downloads at boot are not approved on this fv-control \(the owner sets FV_HUB_DOWNLOADS_APPROVED=1/)]);
    const ok = slsWeightsIssues(APPROVED, hub());
    expect(ok.issues).toEqual([]);
    expect(ok.warnings.map((w) => w.message)).toEqual([expect.stringMatching(/^weights_source hub: every new worker downloads fastwan22-ti2v-5b, wan22-ti2v-5b \([\d.]+ GB\) from the Hub before it serves .*a FlashBoot revival keeps the worker and downloads nothing$/)]);
  });
  it("a tree without a plain Hub source is refused (a conversion, several repos)", () => {
    const s = normalizeEndpointSpec({ name: "ll", preset: "longlive", weights_source: "hub", weights_download_approved: true });
    expect(slsWeightsIssues(APPROVED, s).issues.map((i) => i.message)).toEqual([expect.stringMatching(/^no plain Hub source for longlive-1\.3b-safetensors/)]);
  });
  it("h3-turbo: the pre-quantized trees stay R2-only (the engine quantizes at load); every tree it downloads is pinned", () => {
    const s = hub({ preset: "h3-turbo", gpu_types: [H100] });
    expect(slsHubPlan(s)).toEqual({ trees: ["FastH3-4-step-Preview-v1-LoRA", "h3-base"], unsupported: [] });
    expect(slsWeightsIssues(APPROVED, s).issues).toEqual([]);
  });
});

describe("the template and the endpoint", () => {
  const spec = hub({ image: { sha: "abcdef1" } });
  const env = slsHubEnv(APPROVED, spec, "fvw_" + "a".repeat(64));
  it("the env: the trees at their pinned revisions, the aux files, the recorded hashes, the fetcher at the image's commit, the report URL", () => {
    expect(env.FV_WEIGHTS_SOURCE).toBe("hub");
    expect(env.FV_WEIGHTS).toBe("/workspace/weights");
    const rows = atob(env.FV_WEIGHTS_TREES_B64!).split("\n").filter(Boolean).map((l) => l.split("\t"));
    for (const t of ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]) expect(rows.find((r) => r[0] === "tree" && r[1] === t)?.slice(2, 4)).toEqual([HUB_TREES[t]!.repo, HUB_TREES[t]!.revision]);
    expect(rows.filter((r) => r[0] === "aux").length).toBeGreaterThan(0);
    // weights-sha256.tsv records fastwan22-ti2v-5b (15 files), not wan22-ti2v-5b (the Hub listing check only).
    const sums = atob(env.FV_HUB_SHA256_B64!).split("\n").filter(Boolean).map((l) => l.split("\t"));
    expect(sums.length).toBe(HUB_SHA256["fastwan22-ti2v-5b"]!.length);
    expect(sums.every((r) => r[0] === "fastwan22-ti2v-5b" && /^[0-9a-f]{64}$/.test(r[3]!))).toBe(true);
    expect(sums.find((r) => r[1] === "transformer/diffusion_pytorch_model.safetensors")?.[2]).toBe("9999660080");
    expect(env.FV_SCRIPTS_URL).toBe("https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/abcdef1/scripts/gpu");
    expect(env.FV_SLS_REPORT_URL).toBe("https://fvc.test/ingest/v1/sls-report");
    expect(env.FV_WEIGHTS_PLAN_URL).toBeUndefined();
    // A channel: the fetcher from main.
    expect(slsHubEnv(APPROVED, hub(), "t").FV_SCRIPTS_URL).toMatch(/\/main\/scripts\/gpu$/);
  });
  it("the payloads: the bash Hub boot before fv-serve, the disk, FlashBoot, no volume and no data centre", () => {
    const t = templateCreatePayload(spec, "img", "fvc-hw-x", env);
    expect(t.dockerEntrypoint).toEqual(["bash", "-c", SLS_HUB_BOOT]);
    expect(t.containerDiskInGb).toBe(spec.container_disk_gb);
    expect(t.env.FV_WEIGHTS_SOURCE).toBe("hub");
    expect(t.env.FV_WEIGHTS_TOKEN).toBe("fvw_" + "a".repeat(64));
    expect(SLS_HUB_BOOT.indexOf("fetch-hub-tree.py")).toBeLessThan(SLS_HUB_BOOT.indexOf("exec /opt/fastvideo-rs/bin/fv-serve"));
    expect(SLS_HUB_BOOT).not.toContain("r2-fetch.py");
    execFileSync("bash", ["-n", "-c", SLS_HUB_BOOT]);
    expect(templateUpdatePayload(spec, "img", env).dockerEntrypoint).toEqual(["bash", "-c", SLS_HUB_BOOT]);
    expect(v2TemplateBoot(spec)?.dockerEntrypoint).toEqual(["bash", "-c", SLS_HUB_BOOT]);
    const e = endpointCreatePayload(spec, "tpl");
    expect([e.flashboot, (e as any).networkVolumeId, (e as any).dataCenterIds, e.allowedCudaVersions, e.minCudaVersion]).toEqual([true, undefined, undefined, undefined, "13.0"]);
    expect(v2CreatePayload(hub({ mode: "lb", scaler_type: "REQUEST_COUNT", scaler_value: 1 }), "img", [], env).flashboot).toBe("FLASHBOOT");
  });
});

describe("the outlook", () => {
  it("start + Hub download (planning rate, tool setup) + load; the measured rate once a boot reported one", () => {
    const o = slsOutlook(APPROVED, hub());
    expect(o.weights_source).toBe("hub");
    expect(o.trees.map((t) => [t.dest, t.published])).toEqual([["fastwan22-ti2v-5b", true], ["wan22-ti2v-5b", true]]);
    expect(o.rate_gbps).toBe(HUB_DOWNLOAD_GBPS);
    expect(o.note).toMatch(/Hub download [\d.]+ GB at 0\.20 GB\/s \(planning rate: the Hub on Runpod.*FlashBoot revival keeps the worker/);
    expect(o.cache.note).toMatch(/a new worker downloads every tree from the Hub/);
    const row = applyReport(applyReport({ worker_id: "w1", boot: "w1@1700000000", boot_at: 1 }, "cache", "miss: fastwan22-ti2v-5b wan22-ti2v-5b", 2), "weights", "weights ready: hub 58.40 GB fetched in 292.0s (200 MB/s)", 3) as BootRow;
    expect([row.fetch_bytes, row.fetch_s, row.fetch_mbps]).toEqual([58.4e9, 292, 200]);
    const m = slsOutlook(APPROVED, hub(), [row]);
    expect([m.rate_source, m.rate_gbps, m.cache.state]).toEqual(["measured", 0.2, "miss"]);
  });
});

// ---------------------------------------------------------------- the worker's boot, in bash
const py = existsSync("/usr/bin/python3") || existsSync("/usr/local/bin/python3");
describe.skipIf(!py)("the worker's boot (SLS_HUB_SH) with a stand-in fetch-hub-tree.py", () => {
  const files: Record<string, Buffer> = { "config.json": Buffer.from('{"hub": 1}'), "transformer/model.safetensors": Buffer.from(Array.from({ length: 300_000 }, (_, i) => (i * 31) % 251)) };
  const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");
  const reports: { phase: string; detail: string; pod: string }[] = [];
  let url = "";
  const srv = createServer((req, res) => {
    let b = "";
    req.on("data", (d) => (b += d));
    req.on("end", () => {
      reports.push(JSON.parse(b));
      res.writeHead(200).end("{}");
    });
  });
  beforeAll(async () => {
    await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
    url = `http://127.0.0.1:${(srv.address() as any).port}/ingest/v1/sls-report`;
  });
  afterAll(() => srv.close());
  // fetch-hub-tree.py's contract (scripts/gpu/fetch-hub-tree.py): the tree renamed into $FETCH_WEIGHTS/$FETCH_DEST with
  // .complete, $FETCH_SRV/sha256.txt ("<sha256>  <path>"), $FETCH_SRV/DONE = 0. The stand-in copies a source dir.
  const fake = (src: string) => `import hashlib, os, shutil, pathlib
dest = pathlib.Path(os.environ["FETCH_WEIGHTS"]) / os.environ["FETCH_DEST"]
srv = pathlib.Path(os.environ["FETCH_SRV"]); srv.mkdir(parents=True, exist_ok=True)
open(os.path.join(${JSON.stringify(src)}, "calls"), "a").write(os.environ["FETCH_DEST"] + " " + os.environ["FETCH_REPO"] + "@" + os.environ["FETCH_REVISION"] + "\\n")
shutil.copytree(os.path.join(${JSON.stringify(src)}, "tree"), dest)
(dest / ".complete").write_text("1\\n")
lines = [hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(dest)) for p in sorted(dest.rglob("*")) if p.is_file() and p.name != ".complete"]
(srv / "sha256.txt").write_text("\\n".join(lines) + "\\n")
(srv / "DONE").write_text("0")
`;
  const setup = (recorded: Record<string, Buffer>) => {
    const root = mkdtempSync(join(tmpdir(), "fvhub-"));
    for (const [p, b] of Object.entries(files)) {
      execFileSync("mkdir", ["-p", join(root, "src/tree", p, "..")]);
      writeFileSync(join(root, "src/tree", p), b);
    }
    writeFileSync(join(root, "fake.py"), fake(join(root, "src")));
    writeFileSync(join(root, "aux.bin"), "tae");
    const tsv = [["tree", "demo", "org/demo", "a".repeat(40), "*"].join("\t"), ["aux", "auxiliary/tae/x.safetensors", `file://${join(root, "aux.bin")}`, sha(Buffer.from("tae")), "3"].join("\t")].join("\n") + "\n";
    const sums = Object.entries(recorded).map(([p, b]) => ["demo", p, String(b.length), `${"0".repeat(64)}|${sha(b)}`].join("\t")).join("\n") + "\n";
    // The tool setup (apt, venv, the download of fetch-hub-tree.py) is the real boot's; here the stand-in runs instead.
    const script = SLS_HUB_SH.replace(HUB_FETCH_TREE_SH, HUB_FETCH_TREE_SH.replace(/ {4}\{ apt-get[^\n]*\n {4}python3 -m venv[^\n]*\n {4}mkdir -p \/opt\/fv\/scripts[^\n]*\n/, "").replace("/opt/fv/venv/bin/python /opt/fv/scripts/fetch-hub-tree.py", `python3 ${join(root, "fake.py")}`)).replaceAll("/fvstate", `${root}/fvstate`);
    expect(script).not.toContain("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-venv");
    // Async: the report server runs in this process.
    const run = (worker: string) =>
      new Promise<{ status: number; stdout: string; stderr: string }>((resolve) =>
        execFile(
          "bash",
          ["-c", `set -eu\nmkdir -p ${root}/fvstate\n${script}echo FV-SERVE-START`],
          {
            env: { PATH: process.env.PATH!, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1", FV_WEIGHTS_SOURCE: "hub", FV_WEIGHTS: join(root, "weights"), FV_WEIGHTS_TREES_B64: Buffer.from(tsv).toString("base64"), FV_HUB_SHA256_B64: Buffer.from(sums).toString("base64"), FV_SLS_REPORT_URL: url, FV_WEIGHTS_TOKEN: "fvw_x", FV_FAIL_SLEEP_S: "0", RUNPOD_POD_ID: worker },
            timeout: 60000,
          },
          (e, stdout, stderr) => resolve({ status: e ? Number((e as any).code ?? -1) : 0, stdout: String(stdout), stderr: String(stderr) }),
        ),
      );
    return { root, run };
  };

  it("a cold boot downloads, checks the recorded SHA-256s, fetches the aux files and reports its breakdown; a second boot on the same disk is a cache hit", async () => {
    const { root, run } = setup(files);
    const r = await run("wh1");
    const err = r.stderr;
    expect(r.stdout.trim(), err).toBe("FV-SERVE-START");
    for (const [p, b] of Object.entries(files)) expect(readFileSync(join(root, "weights/demo", p)).equals(b)).toBe(true);
    expect(readFileSync(join(root, "weights/demo/.fv-revision"), "utf8")).toBe("a".repeat(40));
    expect(readFileSync(join(root, "weights/auxiliary/tae/x.safetensors"), "utf8")).toBe("tae");
    expect(err).toMatch(/weights: cache miss: demo: downloading from the Hub at the pinned revisions/);
    expect(err).toMatch(/tree demo: 2 files match weights-sha256\.tsv/);
    expect(err).toMatch(/step hub demo: \d+s/);
    expect(err).toMatch(/step weights: [\d.]+s \(Hub\)/);
    expect(readFileSync(join(root, "src/calls"), "utf8")).toBe(`demo org/demo@${"a".repeat(40)}\n`);
    const mine = reports.filter((x) => x.pod.startsWith("wh1@"));
    expect(mine.map((x) => x.phase)).toEqual(["boot", "cache", "weights", "weights", "step", "step", "weights", "step", "serve"]);
    expect(mine.find((x) => x.phase === "cache")?.detail).toBe("miss: demo");
    expect(mine.filter((x) => x.phase === "step").map((x) => x.detail.split(" ")[0])).toEqual(["hub_tools", "hub_demo", "weights"]);
    const done = mine.filter((x) => x.phase === "weights").at(-1)!.detail;
    expect(done).toMatch(/^weights ready: hub [\d.]+ GB fetched in [\d.]+s \(\d+ MB\/s\)$/);
    // the reports fold into the boot's row as the r2 ones do
    let row: any = { worker_id: "wh1", boot: mine[0]!.pod, boot_at: 1 };
    for (const x of mine) row = applyReport(row, x.phase, x.detail, 2);
    expect([row.cache, row.missing, row.fetch_s !== null, !!row.serve_at, Object.keys(JSON.parse(row.steps)).sort()]).toEqual(["miss", "demo", true, true, ["hub_demo", "hub_tools", "weights"]]);
    // The same disk again (a restarted container): no download.
    const r2 = await run("wh2");
    expect(r2.stdout.trim(), r2.stderr).toBe("FV-SERVE-START");
    expect(r2.stderr).toMatch(/cache hit .*no download/);
    expect(r2.stderr).toMatch(/tree demo: present at a{40}/);
    expect(readFileSync(join(root, "src/calls"), "utf8").trim().split("\n").length).toBe(1);
    expect(reports.find((x) => x.pod.startsWith("wh2@") && x.phase === "cache")?.detail).toBe("hit");
  }, 60000);

  it("a file that is not the recorded one: the tree is removed and the boot fails before fv-serve", async () => {
    const { root, run } = setup({ ...files, "transformer/model.safetensors": Buffer.from("something else") });
    const r = await run("wh3");
    expect(r.status).toBe(1);
    expect(r.stdout).not.toContain("FV-SERVE-START");
    expect(r.stderr).toMatch(/FAILED: tree demo: not the recorded SHA-256 \(weights-sha256\.tsv\): transformer\/model\.safetensors transformer\/model\.safetensors\(size\)/);
    expect(existsSync(join(root, "weights/demo"))).toBe(false);
    expect(reports.find((x) => x.pod.startsWith("wh3@") && x.phase === "failed")?.detail).toMatch(/not the recorded SHA-256/);
  }, 60000);

  it("fetch-hub-tree.py reporting a failure (DONE 1): the boot fails with its log, nothing served", async () => {
    const { root, run } = setup(files);
    writeFileSync(join(root, "fake.py"), `import os, pathlib\nsrv = pathlib.Path(os.environ["FETCH_SRV"]); srv.mkdir(parents=True, exist_ok=True)\n(srv / "log.txt").write_text("12:00:00 not enough space: nothing written\\n")\n(srv / "DONE").write_text("1")\n`);
    const r = await run("wh4");
    expect(r.status).toBe(1);
    expect(r.stdout).not.toContain("FV-SERVE-START");
    expect(r.stderr).toMatch(/FAILED: tree demo: 12:00:00 not enough space: nothing written/);
  }, 60000);
});
