import { describe, expect, it } from "vitest";
import {
  acquireLock,
  buildPodPayload,
  DEFAULT_BUILD_PODS_POLICY,
  imagePin,
  isOutdated,
  matchesRegion,
  normalizePolicy,
  rankCandidates,
  regionOf,
  releaseLock,
  type Candidate,
} from "../../src/buildpods";
import { BUILD_POD_START_CMD } from "../../src/buildpod-start";
import { d1 } from "./d1shim";

const pol = normalizePolicy({ enabled: true });

describe("build pods policy", () => {
  it("defaults: off, 2 pods, $1.5/hr, cpu5c then cpu3c, 32 then 16 vCPU, EU preferred, main", () => {
    const d = normalizePolicy(undefined);
    expect(d).toEqual(DEFAULT_BUILD_PODS_POLICY);
    expect([d.enabled, d.max_pods, d.max_dph_per_pod, d.flavors, d.vcpus, d.regions, d.server_ref]).toEqual([false, 2, 1.5, ["cpu5c", "cpu3c"], [32, 16], ["EU"], "main"]);
  });
  it("clamps and drops what does not validate", () => {
    const p = normalizePolicy({ max_pods: 99, max_dph_per_pod: -1, flavors: ["cpu3c", "rm -rf"], vcpus: [1, 16, 1000], volumes: { "EU-RO-1": "pxy4hlsnwq", bad: "x" }, server_ref: "a b", image: "not an image", labels: ["fv-build", "bad label"] } as any);
    expect(p.max_pods).toBe(8);
    expect(p.max_dph_per_pod).toBe(0.05);
    expect(p.flavors).toEqual(["cpu3c"]);
    expect(p.vcpus).toEqual([16]);
    expect(p.volumes).toEqual({ "EU-RO-1": "pxy4hlsnwq" });
    expect(p.server_ref).toBe("main");
    expect(p.image).toBe("");
    expect(p.labels).toEqual(["fv-build"]);
  });
});

describe("placement", () => {
  it("regions", () => {
    expect(["EU-RO-1", "EUR-IS-1", "US-CA-2", "CA-MTL-1", "AP-JP-1", "OC-AU-1", "SEA-SG-1"].map(regionOf)).toEqual(["eu", "eu", "us", "ca", "ap", "ap", "ap"]);
    expect(matchesRegion("EUR-IS-1", "eu")).toBe(true);
    expect(matchesRegion("EUR-IS-1", "EU-RO-1")).toBe(false);
    expect(matchesRegion("US-CA-2", null)).toBe(true);
  });

  const c = (dc: string, flavor: string, vcpu: number, stock: string | null, price = vcpu * 0.03, extra: Partial<Candidate> = {}): Candidate => ({ dc, flavor, vcpu, ram: vcpu * 2, stock, price, ...extra });
  it("drops candidates without stock or over the $/hr cap; ranks volume, preferred region, size, flavor, stock", () => {
    // The live picture of 2026-10-06: no 32-vCPU cpu5c anywhere, cpu3c-32 High in EU-RO-1 / EUR-IS-1.
    const cands = [
      c("US-CA-2", "cpu3c", 16, "High", 0.48),
      c("EU-RO-1", "cpu5c", 32, null, 1.12),
      c("EUR-IS-1", "cpu3c", 32, "Low", 0.96),
      c("EU-RO-1", "cpu3c", 32, "High", 0.96),
      c("EU-NL-1", "cpu3c", 16, "High", 0.48),
      c("EUR-IS-1", "cpu5c", 16, "High", 0.56),
      c("US-IL-1", "cpu3g", 32, "High", 1.6),
    ];
    const r = rankCandidates(cands, pol);
    expect(r.map((x) => `${x.flavor}-${x.vcpu}@${x.dc}`)).toEqual(["cpu3c-32@EU-RO-1", "cpu3c-32@EUR-IS-1", "cpu5c-16@EUR-IS-1", "cpu3c-16@EU-NL-1", "cpu3c-16@US-CA-2"]);
    // A DC with a cache volume goes first.
    const v = rankCandidates([...cands, c("EUR-IS-1", "cpu3c", 16, "Medium", 0.48, { volume_id: "jg48s6o1w0" })], pol);
    expect(v[0]).toMatchObject({ dc: "EUR-IS-1", vcpu: 16, volume_id: "jg48s6o1w0" });
    // A wanted region, and regions_only.
    expect(rankCandidates(cands, pol, "us").map((x) => x.dc)).toEqual(["US-CA-2"]);
    expect(rankCandidates(cands, { ...pol, regions_only: true }).every((x) => x.dc.startsWith("EU"))).toBe(true);
  });
});

describe("what a pod runs", () => {
  it("the image pin of build-pod.sh", () => {
    expect(imagePin('X=1\nBASE_IMAGE_TAG="bb-d872f7724765429b"\nIMAGE="${FV_BUILD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-build-base:$BASE_IMAGE_TAG}"')).toBe("ghcr.io/zaitrarrio/fastvideo-rs-build-base:bb-d872f7724765429b");
    expect(imagePin("nothing here")).toBeNull();
  });
  it("outdated: another server sha or image at the same ref; other refs (test pods) never", () => {
    const cur = { ref: "main", sha: "aaa", image: "img:1", at: 1 };
    expect(isOutdated({ server_ref: "main", server_sha: "aaa", image: "img:1" }, cur)).toBe(false);
    expect(isOutdated({ server_ref: "main", server_sha: "bbb", image: "img:1" }, cur)).toBe(true);
    expect(isOutdated({ server_ref: "main", server_sha: "aaa", image: "img:2" }, cur)).toBe(true);
    expect(isOutdated({ server_ref: "wip/x", server_sha: "bbb", image: "img:2" }, cur)).toBe(false);
    expect(isOutdated({ server_ref: "main", server_sha: "bbb", image: "img:2" }, { at: 0 })).toBe(false);
  });
  it("payload: CPU secure cloud, one DC and flavor, the server and limits in env; no volume: caches on the container disk", () => {
    const p = buildPodPayload({ name: "fv-build-eu-abc123", image: "img:1", flavor: "cpu3c", vcpu: 32, disk_gb: 200, dc: "EU-RO-1", token_sha: "f".repeat(64), server_b64: "SRV", pol }) as any;
    expect(p).toMatchObject({ computeType: "CPU", cloudType: "SECURE", cpuFlavorIds: ["cpu3c"], vcpuCount: 32, containerDiskInGb: 200, volumeInGb: 0, dataCenterIds: ["EU-RO-1"], ports: ["8000/http"] });
    expect(p.networkVolumeId).toBeUndefined();
    expect(p.dockerStartCmd).toEqual(["/bin/bash", "-c", BUILD_POD_START_CMD]);
    expect(p.env).toMatchObject({ FV_BUILD_TOKEN_SHA256: "f".repeat(64), FV_BUILD_SERVER_B64: "SRV", FV_BUILD_IDLE_MIN: "20", FV_BUILD_MAX_HOURS: "8", FV_BUILD_MAX_GRACE_MIN: "30", FV_BUILD_EVICT_FREE_GB: "40", FV_BUILD_ROOT: "/root/fvb-cache", FV_BUILD_MANAGED: "fv-control" });
    expect(Object.keys(p.env).some((k) => k.startsWith("FV_BUILD_R2_"))).toBe(false);
    const v = buildPodPayload({ name: "n", image: "i", flavor: "cpu3c", vcpu: 16, disk_gb: 80, dc: "EU-RO-1", volume_id: "pxy4hlsnwq", token_sha: "x", server_b64: "y", pol, r2: { endpoint: "https://a.r2", bucket: "fv-build-cache", key_id: "K", secret: "S" } }) as any;
    expect(v).toMatchObject({ networkVolumeId: "pxy4hlsnwq", volumeMountPath: "/workspace" });
    expect(v.env.FV_BUILD_ROOT).toBeUndefined();
    expect(v.env.FV_BUILD_EVICT_FREE_GB).toBe("16");
    expect(v.env).toMatchObject({ FV_BUILD_R2_ENDPOINT: "https://a.r2", FV_BUILD_R2_BUCKET: "fv-build-cache", FV_BUILD_R2_ACCESS_KEY_ID: "K", FV_BUILD_R2_SECRET_ACCESS_KEY: "S" });
  });
  it("the start command unpacks the server, runs the curl watchdog and the server loop", () => {
    expect(BUILD_POD_START_CMD).toContain('base64 -d | gunzip >/opt/fvb/server.py');
    expect(BUILD_POD_START_CMD).toContain("podTerminate");
    expect(BUILD_POD_START_CMD).toContain("while true; do python3 /opt/fvb/server.py");
  });
});

describe("locks", () => {
  it("one holder at a time; free again after release or expiry", async () => {
    const env = { DB: d1() } as any;
    expect(await acquireLock(env, "k", "a", 60_000)).toBe(true);
    expect(await acquireLock(env, "k", "b", 60_000)).toBe(false);
    await releaseLock(env, "k", "b"); // not the holder: no effect
    expect(await acquireLock(env, "k", "b", 60_000)).toBe(false);
    await releaseLock(env, "k", "a");
    expect(await acquireLock(env, "k", "b", 1)).toBe(true);
    await new Promise((r) => setTimeout(r, 5));
    expect(await acquireLock(env, "k", "c", 60_000)).toBe(true);
  });
});
