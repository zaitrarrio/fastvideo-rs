// The payload corpus (plan C3c, docs/adr/0018-fv-deploy.md §7.2): every Runpod request body fv-control builds, over a
// representative set of inputs, through the public payload builders only (workerSystemEnv / workerPlacements /
// workerCreatePayload, templateCreatePayload / templateUpdatePayload / endpointCreatePayload / endpointUpdatePayload /
// v2CreatePayload, migrateSpec / specOf for stored specs), so the same file runs against any revision that has them.
// fixtures/payload-corpus.json is the snapshot; payload-corpus.test.ts compares. The snapshot was first written from
// main at 2258aec2 (the builders before C3c); see the test for the regenerate command.
import { createHash } from "node:crypto";
import type { Env } from "../../src/env";
import { workerCreatePayload, workerPlacements, workerSystemEnv, type EnvCtx } from "../../src/cluster/payloads";
import { defaultSpec, migrateSpec, normalizeSpec, POOL_PRESETS, presetPool, TEMPLATES, type ClusterSpec } from "../../src/cluster/spec";
import { standaloneSpec } from "../../src/standalone";
import { endpointCreatePayload, endpointUpdatePayload, templateCreatePayload, templateUpdatePayload, v2CreatePayload } from "../../src/serverless/payloads";
import { normalizeEndpointSpec, type EndpointSpec } from "../../src/serverless/spec";
import { SLS_PRESETS } from "../../src/serverless/presets";
import { specOf, type SlsRow } from "../../src/serverless/ops";
import { slsHubEnv, slsWeightsEnv } from "../../src/serverless/weights";

/** fv-control's settings for the corpus (fixed: the bodies must not depend on anything random). */
export const CORPUS_ENV = {
  RUNPOD_API_KEY: "rpa_x",
  PUBLIC_URL: "https://fvc.test",
  CF_ACCOUNT_ID: "acct",
  WEIGHTS_R2_ACCESS_KEY_ID: "AK",
  WEIGHTS_R2_SECRET_ACCESS_KEY: "SK",
  FV_HUB_DOWNLOADS_APPROVED: "1",
} as unknown as Env;
const IMAGE = "ghcr.io/zaitrarrio/fastvideo-rs-serve@sha256:" + "2".repeat(64);
const SECRETS = { internal_token: "it", url_signing_key: "us", admin_token: "fvadm_x", ingest_token: "fvi_x" };
const EDGE = { url: "https://edge.test", internal_token: "eit", admin_token: "eat", d1_database_id: "d1x", outputs_bucket: "outb" };
const CONTEXTS: [string, Partial<EnvCtx>][] = [
  ["edge", { edge: EDGE }],
  ["edge-nobucket", { edge: { ...EDGE, outputs_bucket: undefined, d1_database_id: undefined } }],
  ["direct", {}],
  ["backstop", { backstop: true }],
];

/** Sorted keys, recursively: the body as JSON whatever the builder's key order (a JSON body's order is not sent meaning). */
export function canonical(x: unknown): unknown {
  if (Array.isArray(x)) return x.map(canonical);
  if (x && typeof x === "object") {
    const o: Record<string, unknown> = {};
    for (const k of Object.keys(x).sort()) if ((x as Record<string, unknown>)[k] !== undefined) o[k] = canonical((x as Record<string, unknown>)[k]);
    return o;
  }
  return x;
}
export const digest = (x: unknown): string => createHash("sha256").update(JSON.stringify(canonical(x))).digest("hex").slice(0, 16);
/** A body as the snapshot shows it: canonical, every string over 160 characters (boot scripts, inline configs, tree
 * lists) replaced by its length and SHA-256, so the file stays reviewable and the comparison stays exact. */
export function shown(x: unknown): unknown {
  const c = canonical(x);
  const walk = (v: unknown): unknown =>
    typeof v === "string" && v.length > 160 ? `<${v.length} chars sha256:${createHash("sha256").update(v).digest("hex").slice(0, 16)}>` : Array.isArray(v) ? v.map(walk) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, w]) => [k, walk(w)])) : v;
  return walk(c);
}

/** The cases: name → body (or the error a builder threw). */
export function corpus(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const put = (k: string, f: () => unknown) => {
    try {
      out[k] = f();
    } catch (e) {
      out[k] = { THROWS: (e as Error).message };
    }
  };
  const pods = (name: string, spec: ClusterSpec, ctxs = CONTEXTS) => {
    for (const [cv, over] of ctxs)
      for (const pool of spec.pools) {
        const ctx: EnvCtx = { spec, state: { images: {}, workers: {} }, secrets: SECRETS, deadlineMs: 1_700_000_000_000, runpodApiKey: "rpa_K", ingestUrl: "https://ctl/ingest/v1/logs", env: CORPUS_ENV, ...over };
        let e: Record<string, string>;
        try {
          e = workerSystemEnv(ctx, pool, IMAGE);
        } catch (err) {
          out[`pod/${name}/${cv}/${pool.id}`] = { THROWS: (err as Error).message };
          continue;
        }
        const pls = workerPlacements(spec, pool, CORPUS_ENV);
        // The first and the last placement attempt (the others vary only the GPU type / flavor / data centre).
        for (const i of [...new Set([0, pls.length - 1])]) if (pls[i]) put(`pod/${name}/${cv}/${pool.id}/${i}`, () => workerCreatePayload(`fvc-${pool.id}`, IMAGE, pool, pls[i]!, e));
      }
  };
  // Cluster templates, as defined (edge) and as direct clusters.
  for (const t of Object.keys(TEMPLATES)) {
    pods(`tpl-${t}`, defaultSpec("c", t));
    pods(`tpl-${t}-direct`, { ...defaultSpec("c", t), control_plane: "direct" }, CONTEXTS.filter(([c]) => c !== "edge" && c !== "edge-nobucket"));
  }
  // Every pool preset as a standalone pod, on each weights source it allows, and as an edge pool.
  for (const p of POOL_PRESETS) {
    for (const [v, x] of [
      ["default", {}],
      ["r2", { weights_source: "r2" }],
      ["r2-dcs", { weights_source: "r2", data_centers: ["US-TX-3", "any"] }],
      ["hub", { weights_source: "hub", weights_download_approved: true }],
      ["none", { weights_source: "none" }],
    ] as [string, Record<string, unknown>][]) {
      let spec: ClusterSpec;
      try {
        spec = standaloneSpec({ name: "solo", preset: p.id, ...x } as never).spec;
      } catch (e) {
        out[`pod/sa-${p.id}-${v}`] = { THROWS: (e as Error).message };
        continue;
      }
      pods(`sa-${p.id}-${v}`, spec, CONTEXTS.filter(([c]) => c === "backstop"));
    }
    try {
      pods(`edge-${p.id}`, normalizeSpec({ ...defaultSpec("c"), pools: [presetPool(p.id)] }), CONTEXTS.filter(([c]) => c === "edge"));
    } catch (e) {
      out[`pod/edge-${p.id}`] = { THROWS: (e as Error).message };
    }
  }
  // A few pool settings: the upsampler sidecar, a CUDA filter, a disk, GPU types in order, a CPU pool on the volume.
  const solo = (x: Record<string, unknown>) => standaloneSpec({ name: "solo", preset: "h3-turbo", ...x } as never).spec;
  pods("sa-h3-turbo-upsampler", solo({ upsampler_local: true }), CONTEXTS.filter(([c]) => c === "backstop"));
  pods("sa-h3-turbo-disk-gpus", solo({ container_disk_gb: 77, gpu_types: ["NVIDIA H100 80GB HBM3", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], channel: "latest" }), CONTEXTS.filter(([c]) => c === "backstop"));
  const cuda = solo({});
  cuda.pools[0]!.allowed_cuda = ["13.0"];
  pods("sa-h3-turbo-cuda", cuda, CONTEXTS.filter(([c]) => c === "backstop"));
  const tiny = defaultSpec("c", "tiny-cpu");
  pods("cpu-volume", normalizeSpec({ ...tiny, control_plane: "direct", pools: [{ ...tiny.pools[0], volume: true }] }), CONTEXTS.filter(([c]) => c === "direct"));
  // A cluster spec as an older fv-control stored it: the retired gateway block, pools by image variant.
  pods(
    "stored-legacy",
    migrateSpec(
      JSON.parse(
        JSON.stringify({
          name: "old",
          image: { channel: "stable" },
          regions: ["eu"],
          gateway: { enabled: false, auth: "keys" },
          pools: [
            { id: "h3-turbo", count: 1, variant: "h3-turbo", models: presetPool("h3-turbo")!.models },
            { id: "fake", count: 1, variant: "cpu", fake_models: ["fake-wan"], cpu_flavors: ["cpu3c"], vcpu: 2, container_disk_gb: 10, volume: false },
          ],
          cap_s: 3600,
          min_balance: 8.25,
          balance_floor: 8,
          min_start: 20,
          max_gpu_dph: 3.6,
          log_shipping: true,
        }),
      ),
    ),
    CONTEXTS.filter(([c]) => c === "direct"),
  );

  // Serverless: every preset on each weights source and mode, custom endpoints, stored rows.
  const sls: [string, EndpointSpec][] = [];
  for (const p of SLS_PRESETS)
    for (const [v, x] of [
      ["default", {}],
      ["r2", { weights_source: "r2" }],
      ["hub", { weights_source: "hub", weights_download_approved: true, flashboot: true }],
      ["lb", { mode: "lb" }],
      ["set", { env: { FOO: "bar" }, allowed_cuda: ["12.8"], gpu_count: 2, workers_max: 3, upsampler_local: true }],
    ] as [string, Record<string, unknown>][]) {
      try {
        sls.push([`${p.id}-${v}`, normalizeEndpointSpec({ name: "ep", preset: p.id, ...x })]);
      } catch (e) {
        out[`sls/${p.id}-${v}`] = { THROWS: (e as Error).message };
      }
    }
  sls.push(["custom-cpu", normalizeEndpointSpec({ name: "c", compute: "CPU", config: "runpod-fake.toml" })]);
  sls.push(["custom-cpu-dcs", normalizeEndpointSpec({ name: "c", compute: "CPU", config: "runpod-fake.toml", data_centers: ["EU-RO-1"], vcpu: 4 })]);
  sls.push(["custom-gpu", normalizeEndpointSpec({ name: "g", compute: "GPU", config: "runpod-ltx.toml" })]);
  sls.push(["custom-gpu-inline", normalizeEndpointSpec({ name: "g", compute: "GPU", config_toml: '[engine]\nbackend = "cuda"\n' })]);
  // Rows as an older fv-control stored them: an image variant and no config; no compute at all.
  const row = (spec: unknown) => specOf({ spec: JSON.stringify(spec) } as unknown as SlsRow);
  const ltx = normalizeEndpointSpec({ name: "old", compute: "GPU", config: "runpod-ltx.toml" });
  sls.push(["stored-variant", row({ ...ltx, config: undefined, compute: undefined, variant: "ltx" })]);
  sls.push(["stored-cpu-variant", row({ ...normalizeEndpointSpec({ name: "oldc", compute: "CPU", config: "runpod-fake.toml" }), config: undefined, compute: undefined, variant: "cpu" })]);
  for (const id of ["h3-turbo", "ltx"]) {
    const { compute: _c, ...bare } = normalizeEndpointSpec({ name: "ep", preset: id });
    sls.push([`stored-nocompute-${id}`, bare as EndpointSpec]);
  }
  for (const [name, s] of sls) {
    const w = s.weights_source === "hub" ? slsHubEnv(CORPUS_ENV, s, "tok") : slsWeightsEnv(CORPUS_ENV, s, "tok");
    put(`sls/${name}/template`, () => templateCreatePayload(s, IMAGE, "fvc-ep-261025000000", w));
    put(`sls/${name}/template-update`, () => templateUpdatePayload(s, IMAGE, w));
    put(`sls/${name}/endpoint`, () => endpointCreatePayload(s, "tpl1"));
    put(`sls/${name}/endpoint-update`, () => endpointUpdatePayload(s, { ...s, allowed_cuda: ["13.0"] }));
    put(`sls/${name}/v2`, () => v2CreatePayload(s, IMAGE, ["BLACKWELL_96"], w));
  }
  return out;
}

/** The cases whose whole body (long strings by hash) the snapshot shows, for review: one of each kind. */
export const SHOWN = [
  "pod/tpl-standard/edge/h3-turbo/0",
  "pod/tpl-standard/edge-nobucket/h3-turbo/0",
  "pod/tpl-standard-direct/direct/h3-turbo/0",
  "pod/tpl-tiny-cpu-direct/direct/fake/0",
  "pod/sa-h3-turbo-default/backstop/pod/0",
  "pod/sa-h3-turbo-r2/backstop/pod/0",
  "pod/sa-h3-turbo-r2-dcs/backstop/pod/1",
  "pod/sa-h3-turbo-hub/backstop/pod/0",
  "pod/sa-h3-turbo-none/backstop/pod/0",
  "pod/sa-wan-default/backstop/pod/0",
  "pod/sa-h3-turbo-upsampler/backstop/pod/0",
  "pod/sa-h3-turbo-cuda/backstop/pod/0",
  "pod/cpu-volume/direct/fake/0",
  "pod/stored-legacy/direct/h3-turbo/0",
  "pod/stored-legacy/direct/fake/0",
  "sls/h3-turbo-default/template",
  "sls/h3-turbo-default/template-update",
  "sls/h3-turbo-default/endpoint",
  "sls/h3-turbo-default/endpoint-update",
  "sls/h3-turbo-default/v2",
  "sls/h3-turbo-lb/v2",
  "sls/wan-default/template",
  "sls/wan-default/endpoint",
  "sls/h3-turbo-hub/template",
  "sls/h3-turbo-set/endpoint",
  "sls/custom-cpu/template",
  "sls/custom-cpu/endpoint",
  "sls/custom-cpu/endpoint-update",
  "sls/custom-cpu/v2",
  "sls/stored-variant/template",
  "sls/stored-nocompute-h3-turbo/endpoint",
];
