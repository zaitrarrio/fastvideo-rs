// weights_source hub on Runpod (owner 2026-10-14: "Runpod should have a hub option"): the same Hub path as GMI / Brev
// (the trees at their pinned revisions, fetch-hub-tree.py, the owner's approval for each launch and on fv-control),
// onto the pod's container disk (never a network volume), in any data centre. Checks: the validator with and without
// the approval; the pod's env, disk and boot; the boot's weights step run in bash with a stub fetcher.
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { bootFor, workerCreatePayload, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import type { Env } from "../../src/env";
import { HUB_FETCH_TREE_SH, HUB_RUNPOD_SH, hubDiskGb, providerIssues, weightsPlan } from "../../src/providers";
import { mapLaunchIssues, standaloneSpec } from "../../src/standalone";
import { HUB_TREES } from "../../src/weights-sources";

const H100 = "NVIDIA H100 80GB HBM3";
const hubPod = (x: Record<string, unknown> = {}) => standaloneSpec({ name: "hubus", preset: "wan", weights_source: "hub", gpu_types: [H100], data_centers: ["US-TX-3"], ...x } as any).spec;
const issues = (env: Partial<Env>, x: Record<string, unknown> = {}) => mapLaunchIssues(providerIssues(env as Env, hubPod(x))).map((i) => `${i.path.join(".")}: ${i.message}`);

describe("the validator: hub on Runpod needs the owner's approval, as on GMI / Brev", () => {
  it("legal in any data centre with the volume off", () => {
    const pool = hubPod({ weights_download_approved: true }).pools[0]!;
    expect([pool.weights_source, pool.volume, pool.data_centers, pool.hub_download_approved]).toEqual(["hub", false, ["US-TX-3"], true]);
    expect(() => hubPod({ volume: true })).toThrow(/volume: US-TX-3 has no weights volume: volume off with weights_source hub/);
  });
  it("without weights_download_approved: refused at its field", () => {
    expect(issues({ FV_HUB_DOWNLOADS_APPROVED: "1" })).toEqual([expect.stringMatching(/^weights_download_approved: weights_source hub downloads .*wan22-ti2v-5b.* from the Hub at every boot: it needs the owner's approval for this launch \(weights_download_approved: true\)$/)]);
  });
  it("approved for the launch but not on this fv-control (FV_HUB_DOWNLOADS_APPROVED): refused", () => {
    expect(issues({}, { weights_download_approved: true })).toEqual([expect.stringMatching(/^weights_download_approved: Hub downloads at boot are not approved on this fv-control \(the owner sets FV_HUB_DOWNLOADS_APPROVED=1/)]);
  });
  it("both approvals: no issue", () => {
    expect(issues({ FV_HUB_DOWNLOADS_APPROVED: "1" }, { weights_download_approved: true })).toEqual([]);
  });
  it("h3-turbo: every tree it downloads has a pinned Hub source (the pre-quantized ones stay r2-only)", () => {
    const plan = weightsPlan(hubPod({ preset: "h3-turbo", weights_download_approved: true }).pools[0]!);
    expect(plan.source).toBe("hub");
    expect(plan.trees.every((t) => HUB_TREES[t] || t.startsWith("auxiliary"))).toBe(true);
  });
});

describe("the pod: env, disk, payload and boot", () => {
  const env = { FV_HUB_DOWNLOADS_APPROVED: "1", GITHUB_REPO: "zaitrarrio/fastvideo-rs", PUBLIC_URL: "https://fvc.test" } as unknown as Env;
  const spec = hubPod({ weights_download_approved: true, sha: "abcdef1" });
  const pool = spec.pools[0]!;
  const ctx = { spec, state: {}, secrets: { internal_token: "i", url_signing_key: "u", ingest_token: "ingest_tok" }, deadlineMs: 0, runpodApiKey: "rpa_x", ingestUrl: "https://fvc.test/ingest/v1/logs", env } as unknown as EnvCtx;
  const e = workerSystemEnv(ctx, pool, "img");
  const plan = weightsPlan(pool, env);
  it("the pinned trees (weights-revisions.tsv), the fetcher at the image's commit, a disk sized from the trees", () => {
    expect(e.FV_WEIGHTS_SOURCE).toBe("hub");
    const rows = atob(e.FV_WEIGHTS_TREES_B64!).split("\n").filter(Boolean).map((l) => l.split("\t"));
    for (const t of ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]) expect(rows.find((r) => r[0] === "tree" && r[1] === t)?.slice(2, 4)).toEqual([HUB_TREES[t]!.repo, HUB_TREES[t]!.revision]);
    expect(e.FV_SCRIPTS_URL).toBe("https://raw.githubusercontent.com/zaitrarrio/fastvideo-rs/abcdef1/scripts/gpu");
    expect(Number(e.FV_HUB_DISK_GB)).toBe(Math.min(500, Math.max(40, hubDiskGb(plan.trees))));
    expect(e.FV_R2_DISK_GB).toBeUndefined();
  });
  it("the payload: the container disk, no network volume, the data centre; the boot downloads before fv-serve", () => {
    const pl = workerPlacements(spec, pool);
    expect(pl).toEqual([{ dc: "US-TX-3", gpu: H100 }]);
    const body = workerCreatePayload("n", "img", pool, pl[0]!, e);
    expect([body.containerDiskInGb, body.dataCenterIds, (body as any).networkVolumeId]).toEqual([Number(e.FV_HUB_DISK_GB), ["US-TX-3"], undefined]);
    const boot = bootFor(e);
    expect(boot).toContain(HUB_RUNPOD_SH);
    expect(boot.indexOf("fetch-hub-tree.py")).toBeLessThan(boot.indexOf("exec /opt/fastvideo-rs/bin/fv-serve"));
    expect(boot).not.toContain("r2-fetch.py");
    execFileSync("bash", ["-n", "-c", boot]);
    // Without hub (the volume): no Hub step.
    expect(bootFor({})).not.toContain("fetch-hub-tree.py");
  });
  it("the weights step in bash (a stub fetcher): every tree into $FV_WEIGHTS at its revision, the aux files by SHA-256, step lines", () => {
    const dir = mkdtempSync(join(tmpdir(), "hubrp-"));
    const log = join(dir, "fetched.log");
    const aux = join(dir, "aux.bin");
    execFileSync("bash", ["-c", `printf 'tae' > ${aux}`]);
    const sha = execFileSync("sha256sum", [aux]).toString().split(" ")[0];
    const tsv = [...plan.tsv.split("\n").filter((l) => l.startsWith("tree\t")), `aux\tauxiliary/tae/x.safetensors\tfile://${aux}\t${sha}\t3`].join("\n") + "\n";
    const stub = `FETCHER=""\nfetch_tree() { mkdir -p "$FV_WEIGHTS/$1"; touch "$FV_WEIGHTS/$1/.complete"; echo "$1 $2 $3" >> ${log}; say "step hub $1: 0s"; }\n`;
    const sh = `set -u\nmkdir -p ${dir}/fvstate\n${HUB_RUNPOD_SH.replace(HUB_FETCH_TREE_SH, stub).replaceAll("/fvstate", `${dir}/fvstate`)}echo done`;
    const r = spawnSync("bash", ["-c", sh], { env: { ...process.env, FV_WEIGHTS_SOURCE: "hub", FV_WEIGHTS: join(dir, "weights"), FV_WEIGHTS_TREES_B64: Buffer.from(tsv).toString("base64"), RUNPOD_POD_ID: "pod1" } });
    expect(r.stdout.toString().trim(), r.stderr.toString()).toBe("done");
    expect(r.stderr.toString()).toMatch(/weights: Hub trees at pinned revisions \(owner-approved\)/);
    expect(r.stderr.toString()).toMatch(/weights ready: 0 present, 2 fetched/);
    expect(r.stderr.toString()).toMatch(/step weights: \d+s \(Hub\)/);
    const fetched = readFileSync(log, "utf8").trim().split("\n");
    expect(fetched.sort()).toEqual(["fastwan22-ti2v-5b", "wan22-ti2v-5b"].map((t) => `${t} ${HUB_TREES[t]!.repo} ${HUB_TREES[t]!.revision}`).sort());
    for (const t of ["fastwan22-ti2v-5b", "wan22-ti2v-5b"]) expect(readFileSync(join(dir, "weights", t, ".fv-revision"), "utf8")).toBe(HUB_TREES[t]!.revision);
    expect(existsSync(join(dir, "weights", "auxiliary/tae/x.safetensors"))).toBe(true);
  });
});
