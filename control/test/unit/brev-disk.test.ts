// NVIDIA Brev's VM disk (owner 2026-10-18: "I am still not able to configure the disk size for Brev instances that
// are standalone or a pool"): container_disk_gb on a Brev pool or launch is the VM's diskStorage at create, validated
// against the instance type (fixed disk, or the elastic range) and the weights, kept by reuse, shown back.
import { afterEach, describe, expect, it, vi } from "vitest";
import { resetBrevTypes, toType } from "../../src/brev";
import { brevRow, claimWarm, recordBrevCreate } from "../../src/brev-park";
import { checkSpec } from "../../src/cluster/editor";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { brevDisk, brevDiskIssues, brevDiskRule, brevGpuOptions, PROVIDER_IMPLS, providerIssues } from "../../src/providers";
import { validate } from "../../src/schemas";
import { standaloneSpec } from "../../src/standalone";
import { d1 } from "./d1shim";

const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    PUBLIC_URL: "https://fvc.test",
    BREV_API_TOKEN: "brev_SECRET",
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "elastic.x1,fixed.x1",
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;

const TYPES = [
  { type: "elastic.x1", stoppable: true, base_price: { amount: "2.0" }, elastic_root_volume: true, supported_storage: [{ size: "0B", type: "gp3", min_size: "100GiB", max_size: "4TiB", price_per_gb_hr: { amount: "0.000132" } }], cloud_cred_id: "c1", supported_gpus: [{ name: "H100", count: 1 }] },
  { type: "fixed.x1", stoppable: null, base_price: { amount: "1.5" }, supported_storage: [{ size: "850GiB", type: "ssd" }], cloud_cred_id: "c2", supported_gpus: [{ name: "A100", count: 1 }] },
];
/** Brev's API: the instance-type list and the create; every create body recorded. */
function brevMock() {
  const creates: any[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      if (url.includes("/alltypesavailable/")) return new Response(JSON.stringify({ allInstanceTypes: TYPES }));
      if (init.method === "POST" && url.endsWith("/organizations/org1/workspaces")) {
        creates.push(JSON.parse(init.body));
        return new Response(JSON.stringify({ id: `ws${creates.length}` }), { status: 201 });
      }
      return new Response("[]");
    }),
  );
  return creates;
}
afterEach(() => {
  vi.unstubAllGlobals();
  resetBrevTypes();
});

const brevPool = (o: Record<string, unknown> = {}) => ({ id: "b", variant: "cpu", count: 1, compute: "GPU", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], provider: "brev", provider_gpu: "elastic.x1", weights_source: "none", ...o }) as any;

describe("Brev disk: the instance type's storage", () => {
  it("toType reads an elastic disk's range and a fixed disk's size", () => {
    const e = toType(TYPES[0]);
    expect([e.elastic_disk, e.fixed_disk_gb, e.disk_min_gb, e.disk_max_gb]).toEqual([true, null, 100, 4096]);
    const f = toType(TYPES[1]);
    expect([f.elastic_disk, f.fixed_disk_gb, f.disk_min_gb, f.disk_max_gb]).toEqual([false, 850, null, null]);
    expect(brevDiskRule(e)).toMatch(/elastic disk of 100-4096 GB/);
    expect(brevDiskRule(f)).toMatch(/fixed 850 GB disk/);
  });
  it("brevDisk: the pool's size wins; unset, the weights' size inside the type's range", () => {
    expect(brevDisk(toType(TYPES[0]), [], 750)).toBe(750);
    expect(brevDisk(toType(TYPES[0]), [])).toBe(200);
    expect(brevDisk({ disk_min_gb: 300, disk_max_gb: 400 }, [])).toBe(300);
    expect(brevDisk({ disk_min_gb: 10, disk_max_gb: 150 }, [])).toBe(150);
  });
});

describe("Brev disk: the create request", () => {
  it("the chosen size is the create's diskStorage and the record's disk_gb", async () => {
    const creates = brevMock();
    const env = mkEnv();
    await PROVIDER_IMPLS.brev.create(env, { name: "fv-pod-d-1", image: "ghcr.io/x/y:1", gpu: "elastic.x1", env: {}, diskGb: 750 });
    expect(creates[0].diskStorage).toBe("750Gi");
    expect((await brevRow(env, "brev:fv-pod-d-1"))!.disk_gb).toBe(750);
    // Unset: fv-control's size for the weights (none here: the 200 GB floor).
    await PROVIDER_IMPLS.brev.create(env, { name: "fv-pod-d-2", image: "ghcr.io/x/y:1", gpu: "elastic.x1", env: {} });
    expect(creates[1].diskStorage).toBe("200Gi");
  });
  it("a standalone launch on Brev keeps container_disk_gb in its pool", () => {
    const { spec } = standaloneSpec({ name: "bd", preset: "h3-turbo", provider: "brev", provider_gpu: "elastic.x1", container_disk_gb: 900 } as any);
    expect(spec.pools[0]!.container_disk_gb).toBe(900);
    expect(spec.pools[0]!.provider).toBe("brev");
  });
  it("a reuse (warm restart) needs at least the disk asked for", async () => {
    brevMock();
    const env = mkEnv();
    for (const [id, disk] of [["w1", 300], ["w2", 900]] as const) {
      const name = `fv-pod-${id}-1010000000`;
      await recordBrevCreate(env, { workspace_id: id, pod_id: `brev:${name}`, name, instance_type: "elastic.x1", info: toType(TYPES[0]), disk_gb: disk, launch_trees: {}, cluster_id: null, token: `t-${id}`, run: "#!/bin/bash" });
      await env.DB.prepare("UPDATE brev_instances SET state = 'parked', parked_at = ? WHERE workspace_id = ?").bind(Date.now() - (disk === 300 ? 0 : 1000), id).run();
    }
    expect((await claimWarm(env, "elastic.x1", {}, { minDiskGb: 500 }))!.row.workspace_id).toBe("w2");
    expect(await claimWarm(env, "elastic.x1", {}, { minDiskGb: 1000 })).toBeNull();
  });
});

describe("Brev disk: validation", () => {
  it("schema: Runpod at most 500 GB, Brev up to 16 TiB, GMI none", () => {
    const pool = (o: any) => ({ name: "c", image: { channel: "stable" }, regions: ["eu"], control_plane: "direct", auth: "keys", pools: [{ ...brevPool(), ...o }], cap_s: 3600, min_balance: 8, balance_floor: 8, min_start: 10, max_gpu_dph: 3, log_shipping: true });
    expect(validate("cluster-spec", pool({ container_disk_gb: 2000 })).ok).toBe(true);
    const rp = validate("cluster-spec", pool({ provider: undefined, provider_gpu: undefined, weights_source: undefined, compute: "CPU", volume: false, container_disk_gb: 800 }));
    expect(rp.ok).toBe(false);
    expect(JSON.stringify(rp)).toMatch(/at most 500 GB on Runpod/);
    const gm = validate("cluster-spec", pool({ provider: "gmi", provider_gpu: "container.h100.x1", container_disk_gb: 100 }));
    expect(JSON.stringify(gm)).toMatch(/GMI Cloud: the product sets the container's disk/);
    expect(validate("standalone-launch", { name: "x", preset: "h3-turbo", provider: "brev", provider_gpu: "elastic.x1", container_disk_gb: 3000 }).ok).toBe(true);
    expect(validate("standalone-launch", { name: "x", preset: "h3-turbo", container_disk_gb: 600 }).ok).toBe(false);
  });
  it("providerIssues: at least the trees + 40 GB (50 without weights)", () => {
    const env = mkEnv();
    const out = providerIssues(env, { pools: [brevPool({ container_disk_gb: 20 })] });
    expect(out).toContainEqual({ path: ["pools", 0, "container_disk_gb"], message: expect.stringMatching(/20 GB is too small for the VM: at least 50 GB/) });
    expect(providerIssues(env, { pools: [brevPool({ container_disk_gb: 60 })] }).filter((x) => x.path[2] === "container_disk_gb")).toEqual([]);
  });
  it("brevDiskIssues: the elastic range, a fixed disk takes no other size; checkSpec reports them at the field", async () => {
    brevMock();
    const env = mkEnv();
    expect(await brevDiskIssues(env, { pools: [brevPool({ container_disk_gb: 600 })] })).toEqual([]);
    expect((await brevDiskIssues(env, { pools: [brevPool({ container_disk_gb: 5000 })] }))[0]!.message).toMatch(/above elastic.x1's largest disk \(4096 GB\)/);
    expect((await brevDiskIssues(env, { pools: [brevPool({ container_disk_gb: 60 })] }))[0]!.message).toMatch(/below elastic.x1's smallest disk \(100 GB\)/);
    expect((await brevDiskIssues(env, { pools: [brevPool({ provider_gpu: "fixed.x1", container_disk_gb: 500 })] }))[0]!.message).toMatch(/fixed 850 GB disk .*leave the disk unset/);
    expect(await brevDiskIssues(env, { pools: [brevPool({ provider_gpu: "fixed.x1", container_disk_gb: 850 })] })).toEqual([]);
    const spec = { name: "bdisk", image: { channel: "stable" }, regions: ["eu"], control_plane: "direct", auth: "keys", pools: [brevPool({ container_disk_gb: 5000 })], cap_s: 3600, min_balance: 8, balance_floor: 8, min_start: 10, max_gpu_dph: 3, log_shipping: true };
    const r = await checkSpec(env, spec);
    expect(r.issues).toContainEqual({ path: ["pools", 0, "container_disk_gb"], message: expect.stringMatching(/largest disk/) });
  });
  it("the forms' instance-type options carry each type's disk", async () => {
    brevMock();
    const o = await brevGpuOptions(mkEnv());
    expect(o.find((x) => x.id === "elastic.x1")!.disk).toMatchObject({ kind: "elastic", min_gb: 100, max_gb: 4096 });
    expect(o.find((x) => x.id === "fixed.x1")!.disk).toMatchObject({ kind: "fixed", gb: 850 });
  });
});
