// fv-boot (docs/adr/0014-fv-boot.md) against what fv-control parses: the shared fixture
// crates/fastvideo-deploy/fixtures/boot-protocol.json (its Rust test proves fv-boot sends these bytes; this one proves
// fv-control accepts and reads them), the constants both sides pin, and the staging preamble every shell boot starts with.
import { execFileSync, spawnSync } from "node:child_process";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { bootMatch } from "../../src/boottime";
import { EDGE_WORKER_BOOT, WATCHDOG_WORKER_BOOT, WORKER_BOOT, bootFor } from "../../src/cluster/payloads";
import { CUDA13_SYMBOL, DRIVER_TOO_OLD, MIN_DRIVER_MAJOR } from "../../src/driver-check";
import { FV_BOOT_BIN, PREFER_FV_BOOT_SH } from "../../src/fv-boot";
import { diagnose } from "../../src/podlogs";
import { CLOUDFLARED, isOtherPod, PROVIDER_BOOT, REPORT_PHASES } from "../../src/providers";
import { R2_FETCH_SH } from "../../src/r2weights";
import { MXFP8_TREE_GLOB } from "../../src/recipe-trees";
import { BOOT_RE, SLS_PHASES } from "../../src/serverless/ingest";
import { SLS_BOOT, SLS_HUB_BOOT, SLS_R2_BOOT } from "../../src/serverless/payloads";
import { applyReport } from "../../src/serverless/weights";
import { UPSAMPLER_START_SH } from "../../src/upsampler";

const repo = new URL("../../../", import.meta.url).pathname;
const fx = JSON.parse(readFileSync(join(repo, "crates/fastvideo-deploy/fixtures/boot-protocol.json"), "utf8"));

describe("fv-boot's protocol (the shared fixture)", () => {
  it("the pinned constants are fv-control's", () => {
    const k = fx.constants;
    expect(k.min_driver_major).toBe(MIN_DRIVER_MAJOR);
    expect(k.cuda13_symbol).toBe(CUDA13_SYMBOL);
    expect(k.driver_too_old).toBe(DRIVER_TOO_OLD);
    expect(k.cloudflared_url).toBe(CLOUDFLARED.url);
    expect(k.cloudflared_sha256).toBe(CLOUDFLARED.sha256);
    expect(k.mxfp8_tree_glob).toBe(MXFP8_TREE_GLOB);
    expect(R2_FETCH_SH).toContain(`'${k.streaming_fetch_marker}'`);
    expect(k.fv_boot_bin).toBe(FV_BOOT_BIN);
    expect(UPSAMPLER_START_SH).toContain(k.upsampler_bin);
    expect(R2_FETCH_SH).toContain(k.r2_fetch_image);
    expect(WORKER_BOOT).toContain(`exec ${k.serve_bin} --config ${k.worker_toml}`);
  });

  it("pod reports: endpointReport's pod ids and phases, valid JSON with the four keys", () => {
    for (const r of fx.pod_reports) {
      const b = JSON.parse(r.body);
      expect(Object.keys(b)).toEqual(["pod", "phase", "url", "detail"]);
      expect(isOtherPod(b.pod), b.pod).toBe(true);
      expect(REPORT_PHASES).toContain(b.phase);
      expect(b.detail).not.toMatch(/["\\\n]/);
      expect(b.detail.length).toBeLessThanOrEqual(280);
    }
  });

  it("serverless reports: slsReport's boot ids and phases, and applyReport reads the steps and the summary", () => {
    for (const r of fx.sls_reports) {
      const b = JSON.parse(r.body);
      expect(Object.keys(b)).toEqual(["pod", "phase", "url", "detail"]);
      expect(BOOT_RE.test(b.pod), b.pod).toBe(true);
      expect(SLS_PHASES.has(b.phase)).toBe(true);
      expect(b.url).toBe("");
      const row = applyReport({}, b.phase, b.detail, 1000);
      if (r.step) expect(JSON.parse(row.steps!)).toEqual({ [r.step[0]]: r.step[1] });
      if (r.cache) expect(row.cache).toBe(r.cache);
      if (r.missing) expect(row.missing).toBe(r.missing);
      if (r.fetch) expect([row.fetch_bytes, row.fetch_s, row.fetch_mbps]).toEqual([r.fetch.bytes, r.fetch.s, r.fetch.mbps]);
      if (b.phase === "serve") expect(row.serve_at).toBe(1000);
      if (b.phase === "failed") expect(row.error).toBe(b.detail);
    }
  });

  it("log markers: the boot timeline, the Brev host watcher's grep, the driver diagnosis", () => {
    for (const m of fx.log_markers) {
      const got = bootMatch("container", m.line);
      if (m.milestone) expect(got?.m).toBe(m.milestone);
      if (m.volume) expect(got?.volume).toBe(m.volume);
      // brev-host.ts: grep -q '\[fv-boot\] tunnel '
      if (m.brev_tunnel) expect(/\[fv-boot\] tunnel /.test(m.line)).toBe(true);
      if (m.fv_serve_fatal) {
        // fv-boot logs fv-serve's refusal verbatim; Runpod restarts the container, so a bad config shows twice: a crash loop.
        const boot = ["[fv-boot] start (cluster worker; fv-boot 0.1.1)", "[fv-boot] 10:00:00 +0.0s driver 595.91.07 ok", m.line, "[fv-boot] 10:00:00 +0.1s FAILED: fv-serve config: x"];
        expect(diagnose({ ageMs: 60_000, container: boot, system: [] }).phase).toBe("error");
        const d = diagnose({ ageMs: 120_000, container: [...boot, ...boot], system: [] });
        expect(d.phase).toBe("crashloop");
        expect(d.fatal).toBe(true);
      }
      if (m.driver_too_old) {
        const d = diagnose({ ageMs: 60_000, container: [m.line], system: [] });
        expect(d.phase).toBe("driver_too_old");
        expect(d.fatal).toBe(true);
      }
    }
  });
});

describe("the staging preamble (fv-boot when the image has it, the shell boot otherwise)", () => {
  const boots: Record<string, string> = {
    WORKER_BOOT,
    EDGE_WORKER_BOOT,
    WATCHDOG_WORKER_BOOT,
    PROVIDER_BOOT,
    SLS_BOOT,
    SLS_R2_BOOT,
    SLS_HUB_BOOT,
    "bootFor r2 + upsampler": bootFor({ FV_WEIGHTS_SOURCE: "r2", FV_UPSAMPLER_LOCAL: "1" }),
    "bootFor hub": bootFor({ FV_WEIGHTS_SOURCE: "hub" }),
  };
  it("every shell boot fv-control sends starts with it", () => {
    for (const [name, b] of Object.entries(boots)) {
      const first = b.split("\n").slice(0, 2).join("\n") + "\n";
      expect(first, name).toContain(PREFER_FV_BOOT_SH);
      execFileSync("bash", ["-n", "-c", b]);
    }
    execFileSync("sh", ["-n", "-c", SLS_BOOT]);
  });
  it("runs fv-boot when it is there, the rest of the script when it is not or FV_BOOT_LEGACY=1", () => {
    const dir = mkdtempSync(join(tmpdir(), "fvboot-"));
    const fake = join(dir, "fv-boot");
    writeFileSync(fake, `#!/bin/sh\necho "fv-boot ran: $FV_WORKER_TOML_B64"\n`);
    chmodSync(fake, 0o755);
    const tail = `echo "shell boot ran"\n`;
    const run = (bin: string, env: Record<string, string> = {}) =>
      spawnSync("sh", ["-c", `set -eu\n${PREFER_FV_BOOT_SH.split(FV_BOOT_BIN).join(bin)}${tail}`], { env: { PATH: process.env.PATH!, FV_WORKER_TOML_B64: "eA==", ...env } }).stdout.toString();
    expect(run(fake)).toBe("fv-boot ran: eA==\n");
    expect(run(fake, { FV_BOOT_LEGACY: "1" })).toBe("shell boot ran\n");
    expect(run(join(dir, "missing"))).toBe("shell boot ran\n");
  });
});
