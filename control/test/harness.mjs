// Test harness: a mock of every upstream fv-control talks to (Runpod REST,
// GraphQL and the log endpoint, CloudRift, GHCR, GitHub, the Cloudflare API) and of the
// cluster's own pods (fv-serve workers, keyed by pod id) and the edge Worker, plus
// the Worker itself under `wrangler dev` (workerd, local D1/R2/DO).
import { spawn, execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { webcrypto as wc } from "node:crypto";

const te = new TextEncoder();
const b64 = (u) => Buffer.from(u).toString("base64");
export const HERE = new URL("..", import.meta.url).pathname;

export async function hashPassphrase(pass, pepper, iter = 100000) {
  const hk = await wc.subtle.importKey("raw", te.encode(pepper), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const peppered = new Uint8Array(await wc.subtle.sign("HMAC", hk, te.encode(pass)));
  const key = await wc.subtle.importKey("raw", peppered, "PBKDF2", false, ["deriveBits"]);
  const salt = wc.getRandomValues(new Uint8Array(16));
  const bits = await wc.subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations: iter }, key, 256);
  return `pbkdf2-sha256$${iter}$${b64(salt)}$${b64(new Uint8Array(bits))}`;
}

export function startMock() {
  const m = {
    balance: 50,
    spend: 1.5,
    pods: new Map(), // id -> {payload, env, desiredStatus, costPerHr, name, image, created}
    log: [], // every request: {method, path, body}
    dispatches: [],
    drained: [],
    directKeys: [], // direct workers' minted keys (the shared D1 table)
    directCalls: [], // {pod, method, route} of their admin calls
    external: [
      { id: "extbuild0001", name: "fv-build", desiredStatus: "RUNNING", costPerHr: 1.12, imageName: "rust:1-bookworm", gpuCount: 0, machine: { gpuDisplayName: "unknown", dataCenterId: "EU-RO-1" }, runtime: { uptimeInSeconds: 3600, gpus: [], container: { cpuPercent: 80, memoryPercent: 10 } } },
      { id: "extgpu00001", name: "fv-b200-bench-x", desiredStatus: "RUNNING", costPerHr: 6.79, imageName: "ghcr.io/x@sha256:1", gpuCount: 1, machine: { gpuDisplayName: "B200", dataCenterId: "US-CA-2" }, runtime: { uptimeInSeconds: 600, gpus: [{ id: "g", gpuUtilPercent: 2, memoryUtilPercent: 5 }], container: { cpuPercent: 1, memoryPercent: 4 } } },
      { id: "extold00001", name: "little_azure_rook", desiredStatus: "EXITED", costPerHr: 4.59, gpuCount: 1, machine: { gpuDisplayName: "H200 SXM", dataCenterId: "EUR-IS-4" }, runtime: null },
    ],
    failCreate: 0,
    // Build pods (src/buildpods.ts): CPU stock per "<dc>|<instanceId>", repo files, GitHub runners and queued jobs.
    cpuStock: { "EU-RO-1|cpu3c-32-64": ["High", 0.96], "EUR-IS-1|cpu3c-32-64": ["Low", 0.96], "EUR-IS-1|cpu5c-16-32": ["High", 0.56], "US-CA-2|cpu3c-16-32": ["High", 0.48] },
    noStockDcs: new Set(), // create answers "no longer any instances available" there
    repoFiles: {
      "scripts/dev/build-pod-server.py": "#!/usr/bin/env python3\n# mock build pod server v1\n",
      "scripts/dev/build-pod.sh": '#!/usr/bin/env bash\nBASE_IMAGE_TAG="bb-0123456789abcdef"\nIMAGE="${FV_BUILD_IMAGE:-ghcr.io/zaitrarrio/fastvideo-rs-build-base:$BASE_IMAGE_TAG}"\n',
    },
    runners: [], // {id, name, status, busy, labels: [{name}]}
    runnerRegs: [], // POST /v1/runner bodies the pods got (the token checked, then dropped)
    regTokens: 0,
    ghQueued: [], // {run_id, path, jobs: [{id, status, labels}]}
    bp: {}, // pod id -> {jobs_active, idle_s}
    edgeAdmin: "fvadm_edge_mock_admin",
    edgeInternal: "edge-internal-mock-token",
    edgeKeys: [],
    edgeCalls: [],
    edgeEpoch: 1,
    // CloudRift (docs/ops/cloudrift.md): {version, data} over POST, X-API-Key.
    cloudrift: {
      balance: 3000, // cents, as live
      calls: [],
      instances: [
        { id: "cr-ours-1", instance_name: "fv-gpucheck-1006", status: "Active", tags: ["fv", "fv-owner:fastvideo-rs", "fv-kind:gpucheck", `fv-deadline:${Math.floor(Date.now() / 1000) - 60}`], host_address: "203.0.113.7", created_at: new Date(Date.now() - 600_000).toISOString(), resource_info: { cost_per_hour: 139.36, instance_type: "rtxpro6000-11-50-500-1l.1", provider_name: "p" }, gpus: [{ brand_short: "RTX PRO 6000" }] },
        { id: "cr-ours-2", instance_name: "fv-serve-h3-turbo-1006", status: "Active", tags: ["fv", "fv-owner:fastvideo-rs", "fv-kind:serve-h3-turbo", `fv-deadline:${Math.floor(Date.now() / 1000) + 3600}`], host_address: "203.0.113.8", created_at: new Date().toISOString(), resource_info: { cost_per_hour: 62.4, instance_type: "rtx59-16c-nr.1", provider_name: "p" }, gpus: [{ brand_short: "RTX 5090" }] },
        { id: "cr-foreign", instance_name: "someone-else", status: "Active", tags: [], host_address: "203.0.113.9", created_at: new Date().toISOString(), resource_info: { cost_per_hour: 25, instance_type: "v100-6-52-400-generic.1", provider_name: "p" }, gpus: [{ brand_short: "V100 SXM2" }] },
      ],
    },
  };
  let n = 0;
  const newId = () => `mp${Date.now().toString(36)}${(n++).toString(36)}`.slice(0, 14).padEnd(14, "0");
  const digestOf = (tag) => "sha256:" + Buffer.from(tag).toString("hex").padEnd(64, "0").slice(0, 64);
  const json = (res, code, body) => {
    res.writeHead(code, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  };
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString();
    let body = null;
    try { body = raw ? JSON.parse(raw) : null; } catch { body = raw; }
    const p = url.pathname;
    m.log.push({ method: req.method, path: p, body, auth: req.headers.authorization || "" });
    const bearer = (req.headers.authorization || "").replace(/^Bearer /, "");

    // ---- Runpod REST
    if (p.startsWith("/rp/rest/")) {
      if (bearer !== m.runpodKey) return json(res, 401, { error: "bad key" });
      const rest = p.slice("/rp/rest".length);
      if (rest === "/pods" && req.method === "POST") {
        if (m.failCreate > 0) { m.failCreate--; return json(res, 500, { error: "There are no instances currently available" }); }
        if (body.computeType === "CPU" && m.noStockDcs.has((body.dataCenterIds || [])[0])) return json(res, 500, { error: "This machine does not have the resources to deploy your pod. There are no longer any instances available with the requested specifications." });
        const id = newId();
        const cpu = body.computeType === "CPU";
        const pod = { id, name: body.name, payload: body, env: body.env, image: body.imageName, desiredStatus: "RUNNING", costPerHr: cpu ? 0.06 * (body.vcpuCount || 2) / 2 : 2.09, created: Date.now() };
        m.pods.set(id, pod);
        return json(res, 200, { id, name: body.name, costPerHr: pod.costPerHr, desiredStatus: "RUNNING", machine: { dataCenterId: (body.dataCenterIds || ["EUR-IS-1"])[0] }, env: body.env });
      }
      const mm = /^\/pods\/([^/]+)(\/(stop|start))?$/.exec(rest);
      if (mm) {
        const pod = m.pods.get(mm[1]);
        if (!pod) return json(res, 404, { error: "pod not found" });
        if (req.method === "GET") return json(res, 200, { id: pod.id, name: pod.name, desiredStatus: pod.desiredStatus, costPerHr: pod.costPerHr, imageName: pod.image, env: pod.env });
        if (req.method === "PATCH") {
          if (body.env) pod.env = body.env;
          if (body.imageName) pod.image = body.imageName;
          pod.patches = (pod.patches || 0) + 1;
          return json(res, 200, { id: pod.id });
        }
        if (req.method === "DELETE") { m.pods.delete(pod.id); return json(res, 200, {}); }
        if (req.method === "POST" && mm[3] === "stop") { pod.desiredStatus = "EXITED"; return json(res, 200, {}); }
        if (req.method === "POST" && mm[3] === "start") { if (pod.refuseStart) return json(res, 500, { error: "not enough free CPU on the host" }); pod.desiredStatus = "RUNNING"; pod.startedAt = Date.now(); return json(res, 200, {}); }
      }
      return json(res, 404, { error: "no route" });
    }
    if (p === "/rp/graphql") {
      if (bearer !== m.runpodKey) return json(res, 401, { errors: [{ message: "bad key" }] });
      const q = body.query || "";
      if (q.includes("dataCenters") && q.includes("ramMultiplier"))
        return json(res, 200, { data: { dataCenters: ["EU-RO-1", "EUR-IS-1", "EU-NL-1", "US-CA-2", "AP-IN-2"].map((id) => ({ id, listed: id !== "AP-IN-2" })), cpuFlavors: [{ id: "cpu3c", ramMultiplier: 2 }, { id: "cpu5c", ramMultiplier: 2 }, { id: "cpu3g", ramMultiplier: 4 }] } });
      if (q.includes("specifics(")) {
        const data = {};
        for (const mm of q.matchAll(/(a\d+): cpuFlavors \{ id specifics\(input: \{dataCenterId: "([^"]+)", instanceId: "([^"]+)"\}\)/g)) {
          const st = m.cpuStock[`${mm[2]}|${mm[3]}`];
          const fl = mm[3].split("-")[0];
          data[mm[1]] = ["cpu3c", "cpu5c", "cpu3g"].map((id) => ({ id, specifics: { stockStatus: id === fl && st ? st[0] : null, securePrice: id === fl && st ? st[1] : 9 } }));
        }
        return json(res, 200, { data });
      }
      if (q.includes("gpuTypes") && !body.variables?.id)
        return json(res, 200, { data: { gpuTypes: [
          { id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", displayName: "RTX PRO 6000", memoryInGb: 96, securePrice: 2.09, communityPrice: 1.69, lowestPrice: { stockStatus: "Low" } },
          { id: "NVIDIA H100 80GB HBM3", displayName: "H100 SXM", memoryInGb: 80, securePrice: 3.29, communityPrice: 2.99, lowestPrice: { stockStatus: "High" } },
          { id: "NVIDIA H200", displayName: "H200 SXM", memoryInGb: 141, securePrice: 3.59, communityPrice: null, lowestPrice: { stockStatus: "Medium" } },
        ] } });
      if (q.includes("gpuTypes")) return json(res, 200, { data: { gpuTypes: [{ id: body.variables?.id, securePrice: body.variables?.id?.includes("H200") ? 3.59 : 2.09, lowestPrice: { stockStatus: "High" } }] } });
      const pods = [...m.pods.values()].map((x) => ({
        id: x.id, name: x.name, desiredStatus: x.desiredStatus, costPerHr: x.costPerHr, imageName: x.image, gpuCount: x.payload.computeType === "CPU" ? 0 : 1, vcpuCount: 2,
        machine: { gpuDisplayName: x.payload.computeType === "CPU" ? "unknown" : "RTX PRO 6000", dataCenterId: (x.payload.dataCenterIds || ["EUR-IS-1"])[0] },
        runtime: x.desiredStatus === "RUNNING" ? { uptimeInSeconds: 120, gpus: x.payload.computeType === "CPU" ? [] : [{ id: "g", gpuUtilPercent: 3, memoryUtilPercent: 40 }], container: { cpuPercent: 12, memoryPercent: 20 } } : null,
      }));
      return json(res, 200, { data: { myself: { clientBalance: m.balance, currentSpendPerHr: m.spend, spendLimit: 80, pods: [...pods, ...m.external] } } });
    }
    if (p.startsWith("/rp/hapi/pod/")) {
      if (bearer !== m.runpodKey) return json(res, 401, {});
      return json(res, 200, { container: ["2026-09-29T00:00:00Z fv-serve starting", `leak? ${m.runpodKey}`], system: ["pulling image"] });
    }
    // ---- CloudRift
    if (p.startsWith("/cr/api/v1/")) {
      const path = p.slice("/cr/api/v1/".length);
      const cr = m.cloudrift;
      cr.calls.push({ path, version: body?.version, key: req.headers["x-api-key"] === m.cloudriftKey, bearer: !!req.headers.authorization });
      if (!body || !body.version || !("data" in body)) return json(res, 400, "request must be {version, data}");
      const d = body.data;
      const ok = (data) => json(res, path === "instances/terminate" ? 201 : 200, { version: body.version, data });
      if (path === "instance-types/list") return ok({ instance_types: [{ name: "rtxpro6000-11-50-500-1l", brand_short: "RTX PRO 6000", variants: [{ name: "rtxpro6000-11-50-500-1l.1", gpu_count: 1, cost_per_hour: 139.36, available_nodes: 1, available_nodes_per_dc: { "us-x": 1 } }] }] });
      if (req.headers["x-api-key"] !== m.cloudriftKey) { res.writeHead(401); return res.end("User cannot be authenticated from the request"); }
      const sel = (s) => cr.instances.filter((i) => (s?.ById ? s.ById.includes(i.id) : s?.ByStatus ? s.ByStatus.statuses.includes(i.status) : true));
      if (path === "account/info") return ok({ balance: cr.balance, pending: 0.0, disputed: 0, dispute_fees: 0, current_cost_per_hour: null });
      if (path === "instances/list") return ok({ instances: sel(d.selector) });
      if (path === "instances/metrics") return ok({ metrics: (d.selector.ById || []).map((id) => ({ instance_id: id, node_id: "n", gpus: [{ gpu_index: "0", gpu_utilization_percent: 50 }] })) });
      if (path === "instances/terminate") { const t = sel(d.selector); for (const i of t) i.status = "Inactive"; return ok({ terminated: t }); }
      return json(res, 404, `no route ${path}`);
    }
    // ---- GHCR
    if (p === "/ghcr/token") return json(res, 200, { token: "anon" });
    let g = /^\/ghcr\/v2\/(.+)\/manifests\/(.+)$/.exec(p);
    if (g) {
      if (g[2].includes("missing") || g[2].endsWith("-stable") && m.noStable) return json(res, 404, {});
      res.writeHead(200, { "docker-content-digest": digestOf(g[2]) });
      return res.end();
    }
    g = /^\/ghcr\/v2\/(.+)\/tags\/list$/.exec(p);
    if (g) return json(res, 200, { tags: ["latest", "stable", "cpu-stable", "sha-abcdef1", "cpu-sha-abcdef1"] });
    // ---- GitHub
    if (p.startsWith("/gh/")) {
      if (bearer !== m.githubPat) return json(res, 401, { message: "Bad credentials" });
      if (p.endsWith("/dispatches")) { m.dispatches.push(body); res.writeHead(204); return res.end(); }
      let gm = /^\/gh\/repos\/[^/]+\/[^/]+\/contents\/(.+)$/.exec(p);
      if (gm) {
        const f = m.repoFiles[gm[1]];
        if (f === undefined || url.searchParams.get("ref") !== "main") return json(res, 404, { message: "Not Found" });
        res.writeHead(200, { "content-type": "text/plain" });
        return res.end(f);
      }
      if (p.endsWith("/actions/runners/registration-token") && req.method === "POST") { m.regTokens++; return json(res, 201, { token: `REGTOKEN${m.regTokens}xyz`, expires_at: new Date(Date.now() + 3600e3).toISOString() }); }
      if (p.endsWith("/actions/runners") && req.method === "GET") return json(res, 200, { total_count: m.runners.length, runners: m.runners });
      gm = /\/actions\/runners\/(\d+)$/.exec(p);
      if (gm && req.method === "DELETE") {
        const r = m.runners.find((x) => x.id === Number(gm[1]));
        if (!r) return json(res, 404, {});
        if (r.busy) return json(res, 422, { message: "busy" });
        m.runners = m.runners.filter((x) => x !== r);
        res.writeHead(204);
        return res.end();
      }
      gm = /\/actions\/runs\/(\d+)\/jobs$/.exec(p);
      if (gm) return json(res, 200, { jobs: (m.ghQueued.find((r) => r.run_id === Number(gm[1]))?.jobs || []).map((j) => ({ ...j, created_at: new Date().toISOString() })) });
      if (p.endsWith("/actions/runs") && url.searchParams.get("status")) {
        const st = url.searchParams.get("status");
        return json(res, 200, { workflow_runs: m.ghQueued.filter((r) => (r.status || "in_progress") === st).map((r) => ({ id: r.run_id, path: r.path, name: r.path, status: st, created_at: new Date().toISOString() })) });
      }
      if (p.includes("/actions/runs") || p.includes("/runs")) return json(res, 200, { workflow_runs: [{ id: 1, name: "serve-image", event: "push", status: "completed", conclusion: "success", head_sha: "abcdef1234", created_at: new Date().toISOString(), html_url: "https://github.com/x", display_title: "t" }] });
      return json(res, 404, {});
    }
    // ---- Cloudflare API (Analytics Engine SQL)
    if (p.startsWith("/cf/")) return json(res, 200, { data: [] });
    // ---- the edge Worker (control_plane = edge): its fronts are the pods
    // whose env makes them fronts with the edge's internal token.
    if (p.startsWith("/edge/")) {
      const route = p.slice("/edge".length);
      m.edgeCalls.push({ method: req.method, route });
      if (route === "/fv/v1/status") return json(res, 200, { object: "fv.status", edge: true, pools: [] });
      if (bearer !== m.edgeAdmin) return json(res, 401, { error: { kind: "unauthorized" } });
      if (route === "/fv/v1/edge/families") {
        const families = {};
        for (const pod of m.pods.values()) {
          const env = pod.env || {};
          if (env.FV_DISPATCH_FRONT !== "1" || env.FV_INTERNAL_TOKEN !== m.edgeInternal || pod.desiredStatus !== "RUNNING") continue;
          for (const f of String(env.FV_DISPATCH_FAMILIES || "").split(",").filter(Boolean))
            (families[f] ||= { pool: `family:${f}`, workers: [] }).workers.push({ worker_id: pod.id, connected: true, ready: true, draining: false, held: m.edgeHeld?.[pod.id] || 0, sha: "abcdef1234", front: { url: `http://127.0.0.1:${m.port}/pod/${pod.id}`, ready: true } });
        }
        return json(res, 200, { object: "fv.edge.families", families, metrics: {}, key_epoch: m.edgeEpoch });
      }
      if (route === "/fv/v1/admin/keys" && req.method === "POST") {
        const k = { id: `key_${String(m.edgeKeys.length + 1).padStart(12, "e")}`, name: body.name, revoked: false };
        m.edgeKeys.push(k);
        return json(res, 201, { api_key: `fv_edge_${k.id}`, key: k });
      }
      if (route === "/fv/v1/admin/keys" && req.method === "GET") return json(res, 200, { keys: m.edgeKeys, backend: "d1" });
      const ek = m.edgeKeys.find((x) => route === `/fv/v1/admin/keys/${x.id}`);
      if (route.startsWith("/fv/v1/admin/keys/") && req.method === "DELETE") return ek ? ((ek.revoked = true), m.edgeEpoch++, json(res, 200, { key: ek })) : json(res, 404, {});
      return json(res, 404, {});
    }
    // ---- the cluster's pods
    const pm = /^\/pod\/([^/]+)(\/.*)$/.exec(p);
    // The shared build pod's public /healthz (an external pod: not in m.pods).
    if (pm && m.buildHealth?.[pm[1]] && pm[2] === "/healthz") return json(res, 200, m.buildHealth[pm[1]]);
    if (pm) {
      const pod = m.pods.get(pm[1]);
      if (!pod || pod.desiredStatus !== "RUNNING") { res.writeHead(502); return res.end("no pod"); }
      const env = pod.env || {};
      const route = pm[2];
      if (env.FV_BUILD_TOKEN_SHA256) {
        // A build pod (scripts/dev/build-pod-server.py): public /healthz, the rest with its token.
        const st = m.bp[pod.id] || {};
        if (route === "/healthz") return json(res, 200, { ok: true, ready: true, phase: "ready", boot: Math.floor((pod.startedAt || pod.created) / 1000), uptime_s: 60, idle_s: st.idle_s ?? 30, idle_stop_s: Number(env.FV_BUILD_IDLE_MIN) * 60, max_s: 8 * 3600, jobs_active: st.jobs_active ?? 0, jobs: [] });
        const sha = wc.subtle ? Buffer.from(await wc.subtle.digest("SHA-256", te.encode(bearer))).toString("hex") : "";
        if (sha !== env.FV_BUILD_TOKEN_SHA256) return json(res, 401, { error: "unauthorized" });
        if (route === "/v1/runner" && req.method === "POST") {
          if (st.noRunner) return json(res, 404, { error: "no such endpoint" });
          m.runnerRegs.push({ pod: pod.id, ...body, token: body.token?.startsWith("REGTOKEN") ? "<reg>" : "<OTHER>" });
          m.runners = m.runners.filter((x) => x.name !== body.name);
          m.runners.push({ id: 100 + m.runnerRegs.length, name: body.name, status: "online", busy: false, labels: ["self-hosted", "Linux", "X64", ...String(body.labels).split(",")].map((name) => ({ name })) });
          st.runner = body.name;
          m.bp[pod.id] = st;
          return json(res, 202, { registering: body.name, labels: body.labels });
        }
        if (route === "/v1/runner") return json(res, 200, { phase: st.runner ? "running" : "absent", name: st.runner || null, busy: false });
        return json(res, 404, {});
      }
      if (env.FV_WORKER_DIRECT === "1" && route.startsWith("/fv/v1/admin/keys")) {
        // A direct worker: its admin routes take the cluster's FV_ADMIN_TOKEN.
        if (!env.FV_ADMIN_TOKEN || bearer !== env.FV_ADMIN_TOKEN) return json(res, 401, { error: { kind: "unauthorized" } });
        m.directCalls.push({ pod: pod.id, method: req.method, route });
        if (route === "/fv/v1/admin/keys" && req.method === "POST") {
          const k = { id: `key_${String(m.directKeys.length + 1).padStart(12, "0")}`, name: body.name, revoked: false };
          m.directKeys.push(k);
          return json(res, 201, { api_key: `fv_direct_${k.id}`, key: k });
        }
        if (route === "/fv/v1/admin/keys" && req.method === "GET") return json(res, 200, { keys: m.directKeys, backend: "d1" });
        const k = m.directKeys.find((x) => route === `/fv/v1/admin/keys/${x.id}`);
        if (req.method === "DELETE") return k ? ((k.revoked = true), json(res, 200, { key: k })) : json(res, 404, {});
        return json(res, 404, {});
      }
      if (route === "/health") return json(res, 200, { state: "AVAILABLE", build: { git_sha: "abcdef1234", image: { digest: env.FV_IMAGE_DIGEST } } });
      if (route === "/ping") return json(res, 200, {});
      if (req.headers["x-fv-internal-token"] !== env.FV_INTERNAL_TOKEN) return json(res, 401, {});
      if (route === "/fv/v1/internal/drain") { m.drained.push(pod.id); return json(res, 200, { draining: true }); }
      if (route === "/fv/v1/internal/status") return json(res, 200, { stats: { running: 0, queued_batch: 0, queued_stream: 0, sessions: 0 } });
      return json(res, 404, {});
    }
    json(res, 404, { error: `mock: no route ${p}` });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => { m.port = server.address().port; m.close = () => server.close(); resolve(m); }));
}

/** `wrangler dev` of the Worker against the mock, with a fresh local D1/R2/DO state. */
export async function startWorker(mock, secrets) {
  const dir = mkdtempSync(join(tmpdir(), "fvc-state-"));
  const wr = join(HERE, "node_modules/.bin/wrangler");
  execFileSync(wr, ["d1", "migrations", "apply", "fv-control", "--local", "--persist-to", dir], { cwd: HERE, stdio: "pipe", env: { ...process.env, CI: "1" } });
  const port = 18000 + Math.floor(Math.random() * 2000);
  const base = `http://127.0.0.1:${mock.port}`;
  const vars = {
    RUNPOD_REST: `${base}/rp/rest`, RUNPOD_GRAPHQL: `${base}/rp/graphql`, RUNPOD_HAPI: `${base}/rp/hapi`, CLOUDRIFT_API: `${base}/cr`, GITHUB_API: `${base}/gh`, GHCR: `${base}/ghcr`, CF_API: `${base}/cf`,
    POD_URL_TEMPLATE: `${base}/pod/{pod}`, PUBLIC_URL: `http://127.0.0.1:${port}`, CRON_DISABLED: "1", ENVIRONMENT: "test", CF_ACCOUNT_ID: "acct",
    ...secrets,
  };
  const args = ["dev", "--port", String(port), "--ip", "127.0.0.1", "--persist-to", dir, "--show-interactive-dev-session=false", "--log-level", "warn"];
  for (const [k, v] of Object.entries(vars)) args.push("--var", `${k}:${v}`);
  const child = spawn(wr, args, { cwd: HERE, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, CI: "1", WRANGLER_SEND_METRICS: "false", NO_PROXY: "127.0.0.1,localhost", no_proxy: "127.0.0.1,localhost" } });
  let out = "";
  child.stdout.on("data", (d) => (out += d));
  child.stderr.on("data", (d) => (out += d));
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 120; i++) {
    try {
      const r = await fetch(`${url}/healthz`);
      if (r.ok) return { url, dir, child, output: () => out, stop: () => { child.kill("SIGTERM"); rmSync(dir, { recursive: true, force: true }); } };
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  child.kill("SIGTERM");
  throw new Error(`wrangler dev did not start:\n${out.slice(-3000)}`);
}

export function d1Exec(dir, sql) {
  return execFileSync(join(HERE, "node_modules/.bin/wrangler"), ["d1", "execute", "fv-control", "--local", "--persist-to", dir, "--json", "--command", sql], { cwd: HERE, stdio: "pipe", env: { ...process.env, CI: "1" } }).toString();
}

export const SECRETS = {
  RUNPOD_API_KEY: "rpa_TESTKEY_0123456789abcdef",
  GITHUB_PAT: "github_pat_TEST_0123456789",
  CLOUDRIFT_API_KEY: "crk_TEST_0123456789abcdef",
  CLOUDFLARE_API_KEY: "cf_TEST_0123456789abcdef",
  CONTROL_KEK: b64(wc.getRandomValues(new Uint8Array(32))),
  SESSION_SECRET: "sess_" + Buffer.from(wc.getRandomValues(new Uint8Array(24))).toString("hex"),
};
export const PASSPHRASE = "test passphrase for the owner 42";
