// The Runpod CUDA host filter means "the host's driver offers at least CUDA 13.0", never "exactly 13.0" (2026-10-14:
// a "13.0"-only allowedCudaVersions hid every Secure RTX PRO 6000, whose hosts Runpod labels "13.2"). One set of
// constants (enums.ts MIN_CUDA, IMAGE_CUDA, CUDA_VERSIONS) drives fv-control; scripts/gpu/runpod-cuda.sh carries the
// same for the shell launchers, and this test keeps the two equal.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { podAllowedCuda, workerCreatePayload } from "../../src/cluster/payloads";
import { CUDA_VERSIONS, cudaAtLeast, DEFAULT_ALLOWED_CUDA, IMAGE_CUDA, MIN_CUDA } from "../../src/enums";
import { endpointCreatePayload, endpointUpdatePayload, v2CreatePayload } from "../../src/serverless/payloads";
import { normalizeEndpointSpec } from "../../src/serverless/spec";

const SH = new URL("../../../scripts/gpu/runpod-cuda.sh", import.meta.url);
const shVar = (name: string) => new RegExp(`^${name}="([^"]*)"`, "m").exec(readFileSync(SH, "utf8"))![1]!.split(/\s+/).filter(Boolean);

describe("the CUDA filter constants", () => {
  it("compare versions numerically, and every image label is at least the minimum", () => {
    expect([cudaAtLeast("13.2"), cudaAtLeast("13.0"), cudaAtLeast("12.9"), cudaAtLeast("13.10", "13.2"), cudaAtLeast("14.0")]).toEqual([true, true, false, true, true]);
    expect(IMAGE_CUDA.every((v) => cudaAtLeast(v))).toBe(true);
    expect(IMAGE_CUDA).toContain(MIN_CUDA);
  });
  it("a partial list is never the default: none while REST v1 cannot name 13.1 / 13.2", () => {
    expect(CUDA_VERSIONS).not.toContain("13.2");
    expect(DEFAULT_ALLOWED_CUDA).toEqual([]);
  });
  it("scripts/gpu/runpod-cuda.sh carries the same constants, and its default agrees", () => {
    expect(shVar("FV_MIN_CUDA")).toEqual([MIN_CUDA]);
    expect(shVar("FV_IMAGE_CUDA")).toEqual([...IMAGE_CUDA]);
    expect(shVar("FV_RUNPOD_REST_CUDA")).toEqual([...CUDA_VERSIONS]);
    const out = (rest?: string) =>
      execFileSync("bash", ["-c", `source "$1"; ${rest ? `FV_RUNPOD_REST_CUDA="${rest}"; ` : ""}fv_runpod_cuda`, "x", SH.pathname], { encoding: "utf8" }).trim();
    expect(out()).toBe(DEFAULT_ALLOWED_CUDA.join(" "));
    // Once REST names every label, the default is the full list (never "13.0" alone).
    expect(out("13.2 13.1 13.0 12.9")).toBe("13.0 13.1 13.2");
  });
});

describe("placements: at least CUDA 13.0", () => {
  const pool = { name: "p", compute: "GPU", gpu_types: ["NVIDIA RTX PRO 6000 Blackwell Server Edition"] } as any;
  it("pods: no exact list by default in any data centre; a pool's own list is kept as is", () => {
    expect(podAllowedCuda(pool)).toEqual([]);
    for (const dc of ["EU-RO-1", "US-TX-3", undefined]) {
      expect(workerCreatePayload("n", "img", pool, { dc, gpu: pool.gpu_types[0] } as any, { FV_WEIGHTS_SOURCE: "r2" })).not.toHaveProperty("allowedCudaVersions");
    }
    expect(podAllowedCuda({ ...pool, allowed_cuda: ["13.0", "12.8"] })).toEqual(["13.0", "12.8"]);
  });
  it("serverless GPU endpoints: minCudaVersion 13.0 on create and update; CPU endpoints none; v2 gets it from the follow-up PATCH", () => {
    for (const ws of ["volume", "r2"] as const) {
      const s = normalizeEndpointSpec({ name: `c-${ws}`, preset: "wan", weights_source: ws } as any);
      expect(endpointCreatePayload(s, "t")).toMatchObject({ minCudaVersion: MIN_CUDA });
      expect(endpointCreatePayload(s, "t")).not.toHaveProperty("allowedCudaVersions");
      expect(endpointUpdatePayload(s)).toMatchObject({ minCudaVersion: MIN_CUDA });
      expect((v2CreatePayload({ ...s, mode: "lb" } as any, "img") as any).gpu).not.toHaveProperty("allowedCudaVersions");
    }
    const cpu = normalizeEndpointSpec({ name: "cpu", variant: "cpu" } as any);
    expect(endpointCreatePayload(cpu, "t")).not.toHaveProperty("minCudaVersion");
  });
});
