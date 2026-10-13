// The fixes of the R2 benchmark's round-1 findings (docs/ops/r2-bench-2026-10-12.md): the boot's driver check (run in
// sh and bash with a stub nvidia-smi and libcuda), the CUDA host filter for r2 placements, standalone Runpod pods in
// any data centre with r2, the boot diagnosis of an old driver, and the timed boot steps.
import { execFile } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { WORKER_BOOT } from "../../src/cluster/payloads";
import { DRIVER_CHECK_SH, MIN_DRIVER_MAJOR, R2_DEFAULT_CUDA } from "../../src/driver-check";
import { diagnose } from "../../src/podlogs";
import { PROVIDER_BOOT } from "../../src/providers";
import { SLS_BOOT, SLS_R2_BOOT, endpointCreatePayload, endpointUpdatePayload } from "../../src/serverless/payloads";
import { normalizeEndpointSpec, withWeightsSource } from "../../src/serverless/spec";
import { applyReport, slsOutlook, type BootRow } from "../../src/serverless/weights";
import { standaloneSpec } from "../../src/standalone";
import type { Env } from "../../src/env";

const run = promisify(execFile);

describe("the boot's driver check (driver-check.ts)", () => {
  const reports: any[] = [];
  let url = "";
  const srv = createServer((req, res) => {
    let b = "";
    req.on("data", (d) => (b += d));
    req.on("end", () => (reports.push({ auth: req.headers.authorization, body: JSON.parse(b) }), res.writeHead(200).end("{}")));
  });
  beforeAll(async () => {
    await new Promise<void>((r) => srv.listen(0, "127.0.0.1", r));
    url = `http://127.0.0.1:${(srv.address() as any).port}/ingest/v1/sls-report`;
  });
  afterAll(() => srv.close());
  /** A host: nvidia-smi answering `driver` (none: no nvidia-smi) and a libcuda with or without the CUDA 13 symbol. */
  const host = (driver: string | null, symbol: boolean | null) => {
    const d = mkdtempSync(join(tmpdir(), "fvdrv-"));
    mkdirSync(join(d, "bin"));
    if (driver) {
      writeFileSync(join(d, "bin/nvidia-smi"), `#!/bin/sh\necho "${driver}"\n`);
      chmodSync(join(d, "bin/nvidia-smi"), 0o755);
    }
    const lib = join(d, "libcuda.so.1");
    if (symbol !== null) writeFileSync(lib, Buffer.concat([Buffer.alloc(64, 1), Buffer.from(symbol ? "cuInit\0cuCtxGetDevice\0cuCtxGetDevice_v2\0" : "cuInit\0cuCtxGetDevice\0"), Buffer.alloc(64, 2)]));
    return { PATH: `${join(d, "bin")}:/usr/bin:/bin`, FV_LIBCUDA: symbol === null ? "" : lib, FV_DRIVER_FAIL_SLEEP_S: "0" };
  };
  const check = (shell: "sh" | "bash", env: Record<string, string>) => run(shell, ["-c", `set -eu\nFV_DRV_POD="\${RUNPOD_POD_ID:-}"\n${DRIVER_CHECK_SH}echo PAST-CHECK`], { env: { ...env, NO_PROXY: "127.0.0.1", no_proxy: "127.0.0.1" } });

  for (const shell of ["sh", "bash"] as const)
    it(`${shell}: a CUDA 13 host passes; an old driver or a libcuda without cuCtxGetDevice_v2 stops the boot before anything else`, async () => {
      const ok = await check(shell, host("595.91.07", true));
      expect(ok.stdout).toContain("PAST-CHECK");
      expect(ok.stderr).toMatch(/\[fv-boot\] \d\d:\d\d:\d\d driver 595\.91\.07 ok/);
      // the 2026-10-12 host: a driver whose libcuda lacks the CUDA 13.0 symbol cudarc binds
      await expect(check(shell, host("575.57.08", false))).rejects.toMatchObject({ code: 1, stderr: expect.stringMatching(new RegExp(`FAILED: driver_too_old: host driver 575\\.57\\.08 < ${MIN_DRIVER_MAJOR}`)) });
      await expect(check(shell, host("580.65.06", false))).rejects.toMatchObject({ stderr: expect.stringMatching(/driver_too_old: .*libcuda\.so\.1 \(driver 580\.65\.06\) has no cuCtxGetDevice_v2/) });
      // no GPU tooling at all (a CPU worker): nothing to check
      expect((await check(shell, host(null, null))).stdout).toContain("PAST-CHECK");
      expect((await check(shell, { ...host("550.1", false), FV_SKIP_DRIVER_CHECK: "1" })).stdout).toContain("PAST-CHECK");
    });
  it("reports driver_too_old to fv-control with the boot's token (serverless: the weights token, a boot id)", async () => {
    reports.length = 0;
    await expect(check("bash", { ...host("570.1", false), FV_SLS_REPORT_URL: url, FV_WEIGHTS_TOKEN: "fvw_tok", RUNPOD_POD_ID: "wk9" })).rejects.toMatchObject({ code: 1 });
    expect(reports).toHaveLength(1);
    expect(reports[0].auth).toBe("Bearer fvw_tok");
    expect(reports[0].body).toMatchObject({ phase: "failed", detail: expect.stringMatching(/^driver_too_old: host driver 570\.1 < 580/) });
    expect(reports[0].body.pod).toMatch(/^wk9@\d{10}$/);
  });
  it("every boot runs it first: serverless (sh and the r2 bash boot), Runpod pods, GMI / Brev", () => {
    for (const b of [SLS_BOOT, SLS_R2_BOOT, WORKER_BOOT, PROVIDER_BOOT]) {
      expect(b).toContain("fv_driver_why()");
      expect(b.indexOf("fv_driver_why()")).toBeLessThan(b.lastIndexOf("bin/fv-serve"));
    }
    expect(SLS_R2_BOOT.indexOf("fv_driver_why")).toBeLessThan(SLS_R2_BOOT.indexOf("r2-fetch.py"));
    expect(PROVIDER_BOOT).not.toMatch(/RUNPOD_POD_ID/);
  });
  it("the boot diagnosis: an old driver is fatal for the pod and places a new one (no crash loop on that host)", () => {
    for (const line of ["[fv-boot] 05:01:02 FAILED: driver_too_old: host driver 570.1 < 580 (the image needs CUDA 13)", "thread 'main' panicked at cudarc: libcuda.so: undefined symbol: cuCtxGetDevice_v2"])
      expect(diagnose({ ageMs: 60_000, container: ["[fv-boot] start", line], system: [] })).toMatchObject({ phase: "driver_too_old", fatal: true, replace: true });
  });
});

describe("the CUDA host filter (allowedCudaVersions) for placements outside the EU volume", () => {
  it("serverless r2 defaults to CUDA 13.0 hosts; volume endpoints keep no filter; [] lifts it; a switch back resets it", () => {
    expect([...R2_DEFAULT_CUDA]).toEqual(["13.0"]);
    const r2 = normalizeEndpointSpec({ name: "a", preset: "wan", weights_source: "r2" });
    expect(r2.allowed_cuda).toEqual(["13.0"]);
    expect(endpointCreatePayload(r2, "tpl")).toMatchObject({ allowedCudaVersions: ["13.0"] });
    expect(endpointCreatePayload(r2, "tpl")).not.toHaveProperty("dataCenterIds");
    const vol = normalizeEndpointSpec({ name: "b", preset: "ltx" });
    expect(vol.allowed_cuda).toEqual([]);
    expect(endpointCreatePayload(vol, "tpl")).not.toHaveProperty("allowedCudaVersions");
    const open = normalizeEndpointSpec({ name: "c", preset: "wan", weights_source: "r2", allowed_cuda: [] });
    expect(open.allowed_cuda).toEqual([]);
    expect(endpointUpdatePayload(open).allowedCudaVersions!.length).toBeGreaterThan(1); // clears an earlier filter
    expect(withWeightsSource(r2, "volume").allowed_cuda).toEqual([]);
    expect(withWeightsSource({ ...r2, allowed_cuda: ["12.8", "13.0"] } as any, "volume").allowed_cuda).toEqual(["12.8", "13.0"]);
  });
});

describe("standalone Runpod pods: any data centre with weights_source r2 (finding 5)", () => {
  it("dc outside the EU with r2: the pod's data_centers, no volume; without r2 it is refused; EU-RO-1 (the volume) unchanged", () => {
    const { spec } = standaloneSpec({ name: "us1", preset: "wan", weights_source: "r2", dc: "US-TX-3" } as any);
    const pool = spec.pools[0]!;
    expect([pool.weights_source, pool.volume, pool.data_centers, spec.regions]).toEqual(["r2", false, ["US-TX-3"], ["eu"]]);
    expect(() => standaloneSpec({ name: "us2", preset: "wan", dc: "US-TX-3" } as any)).toThrow(/US-TX-3 has no weights volume: weights_source r2/);
    expect(() => standaloneSpec({ name: "us3", preset: "wan", weights_source: "r2", dc: "US-TX-3", volume: true } as any)).toThrow(/volume/);
    const eu = standaloneSpec({ name: "eu1", preset: "ltx", dc: "EU-RO-1" } as any).spec.pools[0]!;
    expect([eu.volume, eu.data_centers]).toEqual([true, undefined]);
  });
});

describe("the timed boot steps (findings 3 and 4)", () => {
  it("each `step` report folds into the boot's steps; the breakdown shows them; a refused host is named in the outlook", () => {
    let r: Partial<BootRow> = { worker_id: "w", boot: "w@1700000000", boot_at: 1 };
    r = applyReport(r, "step", "tools 0.1s (python3 and curl in the image)", 2);
    r = applyReport(r, "step", "fetcher 12.5s (downloaded from fv-control)", 3);
    r = applyReport(r, "step", "fetch 288.3s (weights ready: 1 tree(s), 144.04 GB fetched in 288.3s (500 MB/s))", 4);
    r = applyReport(r, "step", "aux 0.9s (6 files checked, 0 from their pinned URLs)", 5);
    r = applyReport(r, "step", "no number here", 6);
    expect(JSON.parse(r.steps!)).toEqual({ tools: 0.1, fetcher: 12.5, fetch: 288.3, aux: 0.9 });
    const row = { ...r, cache: "miss", missing: null, fetch_start: null, fetch_end: null, fetch_bytes: null, fetch_s: null, fetch_mbps: null, serve_at: null, error: null, last_detail: null, updated_at: 6 } as BootRow;
    const o = slsOutlook({} as Env, { variant: "wan5b", compute: "GPU", preset: "wan", weights_source: "r2", image: { channel: "stable" } } as any, [row]);
    expect(o.last_boot!.breakdown).toMatchObject({ tools_s: 0.1, fetcher_s: 12.5, fetch_s: 288.3, aux_s: 0.9 });
    const bad = slsOutlook({} as Env, { variant: "wan5b", compute: "GPU", preset: "wan", weights_source: "r2", image: { channel: "stable" } } as any, [{ ...row, error: "driver_too_old: host driver 570.1 < 580" }]);
    expect(bad.cache.note).toMatch(/the last boot's host was refused: driver_too_old/);
  });
});
