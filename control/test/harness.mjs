// Test harness: a mock of every upstream fv-control talks to (Runpod REST,
// GraphQL and the log endpoint, CloudRift, GHCR, GitHub, the Cloudflare API) and of the
// cluster's own pods (fv-serve workers, keyed by pod id) and the edge Worker, plus
// the Worker itself under `wrangler dev` (workerd, local D1/R2/DO).
import { spawn, execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash, createHmac, webcrypto as wc } from "node:crypto";

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
    // GPU stock per "<dc>|<gpu type>": [stockStatus, maxUnreservedGpuCount]; unlisted: High, 8.
    gpuStock: { "EU-RO-1|NVIDIA RTX PRO 6000 Blackwell Server Edition": ["Low", 2] },
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
    // Runpod's serverless queue API (api.runpod.ai/v2/<endpoint>/…): jobs by id, the queue, the live workers, /run bodies.
    queue: { jobs: new Map(), queued: 0, running: 0, workers: 0, ran: [], purges: 0 },
    // Runpod serverless endpoints and templates (REST v1 /templates, /endpoints; REST v2 /serverless).
    sls: { templates: new Map(), endpoints: new Map() },
    // Workers' internal job route (DELETE /fv/v1/internal/jobs/{id}): the answer per job id, and the calls.
    internalJobs: {},
    internalCancels: [],
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
  // The R2 weights store (src/r2weights.ts, docs/ops/r2-weights.md): path-style objects of bucket `bucket`, served only
  // through presigned GETs (checked here with node:crypto against `secret`), with Range. `fetched`: what pods pulled.
  // Runpod's data centres and their GPU stock (dataCenters { gpuAvailability }: the launch screen's table).
  m.runpodDcs = [
    { id: "EUR-IS-1", name: "EUR-IS-1", location: "Europe", storageSupport: false, listed: true, gpuAvailability: [
      { available: true, stockStatus: "Low", gpuTypeId: "NVIDIA RTX PRO 6000 Blackwell Server Edition" },
      { available: false, stockStatus: null, gpuTypeId: "NVIDIA H100 80GB HBM3" },
    ] },
    { id: "US-CA-2", name: "US-CA-2", location: "United States", storageSupport: true, listed: true, gpuAvailability: [
      { available: true, stockStatus: "High", gpuTypeId: "NVIDIA H100 80GB HBM3" },
      { available: true, stockStatus: "Medium", gpuTypeId: "NVIDIA H200" },
    ] },
  ];
  m.r2 = { bucket: "fv-weights-test", access: "r2-test-access-id", secret: "r2-TEST-secret-0123456789abcdef", objects: new Map(), gets: [], bad: 0, fetched: [] };
  // GMI Cloud (docs/serve/deploy-gmi-brev.md §2): REST under /gmi/v1, Bearer key. Containers boot like a pod:
  // `/pod/gmi:<name>` serves fv-serve, and the boot's tunnel report goes to fv-control (FV_ENDPOINT_REPORT_URL).
  m.gmi = {
    calls: [],
    templates: [],
    containers: [
      { id: "00000000-0000-4000-8000-0000000f0e1a", name: "someone-else", status: "running", reason: "", product: "container.h100.x1", idc: "us-denver-1", templateId: "t", createdAt: new Date().toISOString(), envs: [{ name: "SECRET", value: "foreign-secret" }] },
      { id: "00000000-0000-4000-8000-00000000057a", name: "fv-pod-stray-0101000000", status: "running", reason: "", product: "container.h200.x1", idc: "us-denver-1", templateId: "t", createdAt: new Date().toISOString(), envs: [] },
    ],
    products: [
      { name: "container.h200.x1", idc: "us-denver-1", type: "Container", price: 320, valid: true, spec: {}, gpuModel: "H200" },
      { name: "container.b200.x1", idc: "us-denver-1", type: "Container", price: 900, valid: true, spec: {}, gpuModel: "B200" },
    ],
    boot: "ok", // ok | fail (the boot reports failed) | silent (no report)
  };
  // NVIDIA Brev (§3): the CLI's REST paths under /brev/api, Bearer token; the env rides in the startup script.
  const brevType = (type, o) => ({
    type,
    supported_gpus: [{ count: 1, name: o.gpu, memory: "80GiB" }],
    supported_storage: o.stoppable ? [{ size: "0B", type: "gp3", min_size: "10GiB", max_size: "16TiB", price_per_gb_hr: { currency: "USD", amount: "0.000132" } }] : [{ size: "850GiB", type: "ssd" }],
    base_price: { currency: "USD", amount: o.price },
    location: o.location,
    provider: o.provider,
    stoppable: o.stoppable ? true : null,
    elastic_root_volume: o.stoppable ? true : null,
    estimated_deploy_time: "7m0s",
    is_available: true,
    cloud_cred_id: o.cred,
  });
  m.brev = {
    calls: [],
    workspaces: [],
    boots: [], // every simulated VM boot: {ws, name, status, kept, fetched, env, run, error}
    startFail: false, // PUT …/start answers 409 (no capacity)
    startHang: false, // PUT …/start leaves it STARTING (the restart times out)
    stopFail: false, // PUT …/stop answers 400 (an API error)
    boot: "ok",
    types: [
      brevType("g5.xlarge-test", { gpu: "A10G", price: "1.006000", location: "us-east-1", provider: "aws", stoppable: true, cred: "devplane-brev-1-credential" }),
      brevType("a100-test", { gpu: "A100", price: "1.800000", location: "houston-usa-1", provider: "shadeform", stoppable: false, cred: "shadeform-brev-1" }),
    ],
  };
  // The Cloudflare API's tunnels and DNS (src/tunnels.ts): named tunnels of one account, records of the zones, the
  // tunnels' connections (a simulated pod's cloudflared), and the edge routing <host> through them (/tunnel/<host>/…).
  m.cf = {
    token: null, // CF_TUNNEL_API_TOKEN (Bearer); unset: every tunnel / DNS call is refused
    account: "acct",
    zones: [{ id: "zone0maximal0000000000000000000", name: "maximal.llc" }],
    tunnels: [], // {id, name, config_src, created_at, deleted_at, secret, token, config, connections: [{pod}]}
    dns: [], // {id, zone, type, name, content, proxied, comment}
    calls: [], // {method, path, auth}
    fail: {}, // "<METHOD> <path regex source>" -> [status, code, message], once
  };
  const cfOk = (res, result, extra = {}) => json(res, 200, { success: true, errors: [], messages: [], result, ...extra });
  const cfErr = (res, status, code, message) => json(res, status, { success: false, errors: [{ code, message }], messages: [], result: null });
  const tunnelView = ({ secret, token, config, connections, ...t }) => ({ ...t, connections: connections.map((c) => ({ id: c.id, colo_name: "mock" })), status: connections.length ? "healthy" : "inactive" });
  /** Where a host answers through the simulated edge: its proxied CNAME -> a live tunnel -> its ingress -> the pod connected to it. */
  m.routeHost = (host) => {
    const rec = m.cf.dns.find((r) => r.name === host && r.type === "CNAME" && r.proxied);
    const tid = /^([0-9a-f-]{36})\.cfargotunnel\.com$/.exec(rec?.content || "")?.[1];
    const t = m.cf.tunnels.find((x) => x.id === tid && !x.deleted_at);
    const rule = t?.config?.ingress?.find((r) => r.hostname === host);
    const conn = t?.connections.find((c) => m.pods.get(c.pod)?.desiredStatus === "RUNNING");
    if (!rec) return { error: "no DNS record (1016)" };
    if (!t) return { error: "no tunnel (1033)" };
    if (rule?.service !== "http://127.0.0.1:8000") return { error: "no ingress rule for the host" };
    if (!conn) return { error: "tunnel has no connection (1033)" };
    return { pod: conn.pod };
  };
  let n = 0;
  const newId = () => `mp${Date.now().toString(36)}${(n++).toString(36)}`.slice(0, 14).padEnd(14, "0");
  const uuid = () => wc.randomUUID();
  /**
   * A GMI / Brev pod's boot (providers.ts PROVIDER_BOOT): it serves at /pod/<key> and reports its "tunnel" URL to fv-control:
   * with FV_TUNNEL_TOKEN (a named tunnel) its cloudflared connects with the token (checked against the simulated
   * account) and it reports FV_TUNNEL_URL; else a quick tunnel (its /pod/<key> URL).
   */
  const simBoot = (key, env, mode, extra) => {
    m.pods.set(key, { id: key, name: key.split(":")[1], payload: extra, env, image: extra.image, desiredStatus: "RUNNING", costPerHr: 0, created: Date.now(), provider: key.split(":")[0] });
    if (mode === "silent" || !env.FV_ENDPOINT_REPORT_URL) return;
    const report = (b) => fetch(env.FV_ENDPOINT_REPORT_URL, { method: "POST", headers: { authorization: `Bearer ${env.FV_ENDPOINT_REPORT_TOKEN}`, "content-type": "application/json" }, body: JSON.stringify({ pod: key, ...b }) }).then((r) => (m.reports ||= []).push({ key, status: r.status, ...b })).catch((e) => (m.reports ||= []).push({ key, error: String(e) }));
    const tunnel = () => {
      if (!env.FV_TUNNEL_TOKEN) return report({ phase: "tunnel", url: `http://127.0.0.1:${m.port}/pod/${key}`, detail: "" });
      // cloudflared tunnel run: the token is {a, t, s} (base64 JSON); the tunnel must exist, live, with that secret.
      let tok = null;
      try { tok = JSON.parse(Buffer.from(env.FV_TUNNEL_TOKEN, "base64").toString()); } catch {}
      const t = m.cf.tunnels.find((x) => x.id === tok?.t && x.secret === tok?.s && tok?.a === m.cf.account && !x.deleted_at);
      if (!t) return report({ phase: "failed", url: "", detail: "named tunnel did not connect: Unauthorized: Invalid tunnel secret" });
      t.connections.push({ id: uuid(), pod: key });
      (m.tunnelConnects ||= []).push({ key, tunnel: t.id });
      return report({ phase: "tunnel", url: env.FV_TUNNEL_URL, detail: "" });
    };
    // weights_source r2: the boot's R2_FETCH_SH and r2-fetch.py in short: the fetcher from fv-control (its SHA-256
    // checked), the plan with the ingest token, then every part by a ranged GET on its presigned URL, each part and file
    // checked against the plan's SHA-256s.
    const r2 = async () => {
      const py = Buffer.from(await (await fetch(env.FV_R2_FETCH_URL)).arrayBuffer());
      if (createHash("sha256").update(py).digest("hex") !== env.FV_R2_FETCH_SHA256) throw new Error("r2-fetch.py sha256");
      const pr = await fetch(env.FV_WEIGHTS_PLAN_URL, { method: "POST", headers: { authorization: `Bearer ${env.FV_ENDPOINT_REPORT_TOKEN}`, "content-type": "application/json" }, body: JSON.stringify({ pod: key }) });
      if (!pr.ok) throw new Error(`plan: HTTP ${pr.status} ${(await pr.text()).slice(0, 200)}`);
      const plan = await pr.json();
      let bytes = 0;
      for (const t of plan.trees) {
        for (const f of t.files) {
          const whole = createHash("sha256");
          for (let i = 0; i < f.parts.length; i++) {
            const a = i * t.part_size, b = Math.min(f.size, a + t.part_size) - 1;
            const r = await fetch(f.url, { headers: { range: `bytes=${a}-${b}` } });
            const buf = Buffer.from(await r.arrayBuffer());
            if (r.status !== 206 || createHash("sha256").update(buf).digest("hex") !== f.parts[i]) throw new Error(`${t.dest}/${f.path} part ${i}: HTTP ${r.status}`);
            whole.update(buf);
            bytes += buf.length;
          }
          if (whole.digest("hex") !== f.sha256) throw new Error(`${t.dest}/${f.path}: sha256`);
        }
        m.r2.fetched.push({ key, dest: t.dest, rev: t.rev, files: t.files.length, bytes: t.total_bytes });
      }
      return `r2 weights ready: ${plan.trees.length} tree(s), ${bytes} bytes`;
    };
    setTimeout(async () => {
      if (mode === "fail") return report({ phase: "failed", url: "", detail: "tree h3-base: hub 404" });
      if (env.FV_WEIGHTS_SOURCE === "r2") {
        try {
          await report({ phase: "weights", url: "", detail: await r2() });
        } catch (e) {
          return report({ phase: "failed", url: "", detail: `r2 weights: ${String(e.message || e).slice(0, 200)}` });
        }
      }
      tunnel();
    }, 300);
  };
  /**
   * A Brev VM's boot (brev-park.ts): the startup script's bootstrap fetches the run script from fv-control with the
   * boot token, the run script's env file is the pod's env, and its weights loop keeps trees on the VM's disk at the
   * pinned revision (WEIGHTS_SH) and "downloads" the rest. Every boot is recorded in m.brev.boots.
   */
  const brevVmBoot = (w) => {
    const url = /printf %s '([^']+)' > "\$D\/boot-url"/.exec(w.startupScript)?.[1];
    const token = /printf 'Authorization: Bearer %s\\n' '([^']+)' > "\$D\/boot-header"/.exec(w.startupScript)?.[1];
    const rec = { ws: w.id, name: w.name, status: 0, kept: [], fetched: [] };
    m.brev.boots.push(rec);
    if (!url || !token) return (rec.error = "no boot url / token in the startup script");
    setTimeout(async () => {
      try {
        const r = await fetch(url, { headers: { authorization: `Bearer ${token}` } });
        rec.status = r.status;
        const run = await r.text();
        if (!r.ok) return (rec.error = run.slice(0, 200));
        rec.run = run;
        // The env file the host script writes (base64 in its printf to $W/fv/env).
        const envB64 = /printf %s '([A-Za-z0-9+/=]+)' \| base64 -d > '\$W\/fv\/env'/.exec(run)?.[1] || "";
        const env = Object.fromEntries(Buffer.from(envB64, "base64").toString().split("\n").filter(Boolean).map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
        rec.env = env;
        // The host script's own reports (brev-host.ts) come first: its facts, then container_started (or host_failed).
        const hostReport = (phase, detail) =>
          fetch(env.FV_ENDPOINT_REPORT_URL, { method: "POST", headers: { authorization: `Bearer ${env.FV_ENDPOINT_REPORT_TOKEN}`, "content-type": "application/json" }, body: JSON.stringify({ pod: `brev:${w.name}`, phase, url: "", detail }) })
            .then((r) => (m.reports ||= []).push({ key: `brev:${w.name}`, status: r.status, phase }))
            .catch(() => {});
        if (env.FV_ENDPOINT_REPORT_URL) {
          await hostReport("host", "user=ubuntu uid=1000 sudo=yes docker=/usr/bin/docker info=[27.1.1 runtimes=nvidia,runc,] gpu=[GPU 0: NVIDIA A10G] toolkit=[NVIDIA Container Toolkit CLI version 1.17.8] disk=[180G free of 197G] kernel=6.8.0 os=[ubuntu 22.04] ws=/home/ubuntu/workspace");
          if (m.brev.boot === "hostfail") return void (await hostReport("host_failed", 'docker run (exit 125): docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]] | '));
          await hostReport("heartbeat", "container: no container; last: ");
          await hostReport("container_started", "c0ffee000000 image");
          // The host watcher: the container exits (as live 2026-10-10: fv-serve's config refused) or /health answers.
          if (m.brev.boot === "exit") return void (await hostReport("container_exited", "container exited exit=2 oom=false: [fv-boot] start | fv-serve: config: auth.key_store = d1 needs the D1 settings | "));
          setTimeout(() => hostReport("serve_ready", "fv-serve /health 200, 1 s after the tunnel"), 800);
        }
        if (env.FV_WEIGHTS_SOURCE === "hub")
          for (const line of Buffer.from(env.FV_WEIGHTS_TREES_B64 || "", "base64").toString().split("\n")) {
            const [kind, tree, , rev] = line.split("\t");
            if (kind !== "tree") continue;
            if (w.disk[tree] === rev) rec.kept.push(tree);
            else rec.fetched.push(tree), m.brev.boot === "ok" && (w.disk[tree] = rev);
          }
        simBoot(`brev:${w.name}`, env, m.brev.boot, { startupScript: w.startupScript, image: /--entrypoint bash '([^']+)'/.exec(run)?.[1] });
      } catch (e) {
        rec.error = String(e);
      }
    }, 50);
  };
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
    let p = url.pathname;
    m.log.push({ method: req.method, path: p, body, auth: req.headers.authorization || "" });
    // ---- the R2 weights store's S3 endpoint (WEIGHTS_R2_ENDPOINT): presigned GETs only
    if (p.startsWith(`/${m.r2.bucket}/`)) {
      const q = url.searchParams;
      const amz = q.get("X-Amz-Date") || "";
      const when = Date.UTC(+amz.slice(0, 4), +amz.slice(4, 6) - 1, +amz.slice(6, 8), +amz.slice(9, 11), +amz.slice(11, 13), +amz.slice(13, 15));
      const [ak, date, region] = (q.get("X-Amz-Credential") || "").split("/");
      const pairs = [...q.entries()].filter(([k]) => k !== "X-Amz-Signature").map(([k, v]) => [encodeURIComponent(k), encodeURIComponent(v)]).sort((a, b) => (a[0] < b[0] ? -1 : 1));
      const canonical = ["GET", p, pairs.map(([k, v]) => `${k}=${v}`).join("&"), `host:${req.headers.host}\n`, "host", "UNSIGNED-PAYLOAD"].join("\n");
      const toSign = ["AWS4-HMAC-SHA256", amz, `${date}/${region}/s3/aws4_request`, createHash("sha256").update(canonical).digest("hex")].join("\n");
      let k = createHmac("sha256", "AWS4" + m.r2.secret).update(date || "").digest();
      for (const s of [region || "", "s3", "aws4_request"]) k = createHmac("sha256", k).update(s).digest();
      const ok = req.method === "GET" && ak === m.r2.access && Date.now() <= when + Number(q.get("X-Amz-Expires")) * 1000 && createHmac("sha256", k).update(toSign).digest("hex") === q.get("X-Amz-Signature");
      if (!ok) { m.r2.bad++; res.writeHead(403); return res.end("<Error><Code>AccessDenied</Code></Error>"); }
      const objKey = decodeURIComponent(p.slice(m.r2.bucket.length + 2));
      const obj = m.r2.objects.get(objKey);
      m.r2.gets.push({ key: objKey, range: req.headers.range || null });
      if (!obj) { res.writeHead(404); return res.end("<Error><Code>NoSuchKey</Code></Error>"); }
      const rg = /bytes=(\d+)-(\d+)/.exec(req.headers.range || "");
      if (!rg) { res.writeHead(200, { "content-length": obj.length }); return res.end(obj); }
      const a = +rg[1], b = Math.min(+rg[2], obj.length - 1);
      res.writeHead(206, { "content-length": b - a + 1, "content-range": `bytes ${a}-${b}/${obj.length}` });
      return res.end(obj.subarray(a, b + 1));
    }
    const bearer = (req.headers.authorization || "").replace(/^Bearer /, "");
    // ---- the Cloudflare edge for a named tunnel's host (FV_TUNNEL_URL_TEMPLATE): DNS -> tunnel -> ingress -> the connected pod
    const tm0 = /^\/tunnel\/([a-z0-9.-]+)(\/.*)$/.exec(p);
    if (tm0) {
      const r = m.routeHost(tm0[1]);
      (m.tunnelHits ||= []).push({ host: tm0[1], route: tm0[2], ...r });
      if (r.error) { res.writeHead(530); return res.end(r.error); }
      p = `/pod/${r.pod}${tm0[2]}`;
    }
    // ---- the Cloudflare API: tunnels and DNS (src/tunnels.ts)
    if (/^\/cf\/(accounts\/[^/]+\/cfd_tunnel|zones)/.test(p)) {
      const c = m.cf;
      const rest = p.slice("/cf".length);
      c.calls.push({ method: req.method, path: rest + (url.search || ""), auth: !!c.token && bearer === c.token });
      if (!c.token || bearer !== c.token) return cfErr(res, 403, 10000, "Authentication error");
      for (const [k, v] of Object.entries(c.fail)) {
        const [meth, re] = [k.slice(0, k.indexOf(" ")), k.slice(k.indexOf(" ") + 1)];
        if (meth === req.method && new RegExp(re).test(rest)) { delete c.fail[k]; return cfErr(res, v[0], v[1], v[2]); }
      }
      let mm = /^\/accounts\/([^/]+)\/cfd_tunnel(?:\/([^/]+))?(?:\/(configurations|token|connections))?$/.exec(rest);
      if (mm) {
        if (mm[1] !== c.account) return cfErr(res, 403, 10000, "Authentication error");
        if (!mm[2] && req.method === "POST") {
          if (!/^[\w.-]{1,64}$/.test(body?.name || "") || body.config_src !== "cloudflare") return cfErr(res, 400, 1001, "name and config_src cloudflare");
          if (c.tunnels.some((t) => t.name === body.name && !t.deleted_at)) return cfErr(res, 409, 1013, "You already have a tunnel with this name");
          const id = uuid();
          const secret = Buffer.from(wc.getRandomValues(new Uint8Array(32))).toString("base64");
          const t = { id, name: body.name, config_src: "cloudflare", created_at: new Date().toISOString(), deleted_at: null, secret, token: Buffer.from(JSON.stringify({ a: c.account, t: id, s: secret })).toString("base64"), config: null, connections: [] };
          c.tunnels.push(t);
          return cfOk(res, tunnelView(t));
        }
        if (!mm[2] && req.method === "GET") {
          let l = c.tunnels;
          if (url.searchParams.get("is_deleted") === "false") l = l.filter((t) => !t.deleted_at);
          if (url.searchParams.get("name")) l = l.filter((t) => t.name === url.searchParams.get("name"));
          const per = Number(url.searchParams.get("per_page") || 20), page = Number(url.searchParams.get("page") || 1);
          return cfOk(res, l.slice((page - 1) * per, page * per).map(tunnelView), { result_info: { page, per_page: per, count: Math.min(per, l.length), total_count: l.length } });
        }
        const t = c.tunnels.find((x) => x.id === mm[2]);
        if (!t) return cfErr(res, 404, 1003, "Tunnel not found");
        if (!mm[3] && req.method === "GET") return cfOk(res, tunnelView(t));
        if (!mm[3] && req.method === "DELETE") {
          if (t.deleted_at) return cfErr(res, 400, 1022, "Tunnel is already deleted");
          if (t.connections.length) return cfErr(res, 400, 1022, "Cannot delete a tunnel with active connections");
          t.deleted_at = new Date().toISOString();
          return cfOk(res, tunnelView(t));
        }
        if (mm[3] === "configurations" && req.method === "PUT") {
          const ing = body?.config?.ingress;
          if (!Array.isArray(ing) || !ing.length || ing.at(-1).hostname || !ing.at(-1).service) return cfErr(res, 400, 1056, "the last ingress rule must be a catch-all");
          t.config = body.config;
          return cfOk(res, { tunnel_id: t.id, version: 1, config: t.config });
        }
        if (mm[3] === "configurations" && req.method === "GET") return cfOk(res, { tunnel_id: t.id, config: t.config });
        if (mm[3] === "token" && req.method === "GET") return t.deleted_at ? cfErr(res, 404, 1003, "Tunnel not found") : cfOk(res, t.token);
        if (mm[3] === "connections" && req.method === "DELETE") { t.connections = []; return cfOk(res, null); }
        return cfErr(res, 405, 10405, "method not allowed");
      }
      if (rest === "/zones" && req.method === "GET") return cfOk(res, c.zones.filter((z) => !url.searchParams.get("name") || z.name === url.searchParams.get("name")));
      mm = /^\/zones\/([^/]+)\/dns_records(?:\/([^/]+))?$/.exec(rest);
      if (mm) {
        const z = c.zones.find((x) => x.id === mm[1]);
        if (!z) return cfErr(res, 404, 7003, "Could not route to /zones/x, perhaps your object identifier is invalid?");
        if (!mm[2] && req.method === "POST") {
          if (body?.type !== "CNAME" || !body.name?.endsWith(`.${z.name}`) || !body.content) return cfErr(res, 400, 9000, "invalid record");
          if (c.dns.some((r) => r.zone === z.id && r.name === body.name)) return cfErr(res, 400, 81053, "An A, AAAA, or CNAME record with that host already exists.");
          const r = { id: uuid().replace(/-/g, ""), zone: z.id, type: body.type, name: body.name, content: body.content, proxied: !!body.proxied, ttl: body.ttl, comment: body.comment ?? null };
          c.dns.push(r);
          return cfOk(res, r);
        }
        if (!mm[2] && req.method === "GET") {
          const pre = url.searchParams.get("comment.startswith");
          return cfOk(res, c.dns.filter((r) => r.zone === z.id && (!pre || (r.comment || "").startsWith(pre))));
        }
        const r = c.dns.find((x) => x.zone === z.id && x.id === mm[2]);
        if (!r) return cfErr(res, 404, 81044, "Record does not exist.");
        if (req.method === "GET") return cfOk(res, r);
        if (req.method === "DELETE") { c.dns = c.dns.filter((x) => x !== r); return cfOk(res, { id: r.id }); }
      }
      return cfErr(res, 404, 7000, `mock: no Cloudflare route ${req.method} ${rest}`);
    }

    // ---- Runpod REST
    if (p.startsWith("/rp/rest/")) {
      if (bearer !== m.runpodKey) return json(res, 401, { error: "bad key" });
      const rest = p.slice("/rp/rest".length);
      if (rest === "/pods" && req.method === "POST") {
        if (m.failCreate > 0) { m.failCreate--; return json(res, 500, { error: "There are no instances currently available" }); }
        if (body.computeType === "CPU" && m.cpuDiskCap && body.containerDiskInGb > m.cpuDiskCap) return json(res, 400, { error: `create pod: Container Disk must be less than or equal to ${m.cpuDiskCap} GB for this instance` });
        if (body.computeType === "CPU" && m.noStockDcs.has((body.dataCenterIds || [])[0])) return json(res, 500, { error: "This machine does not have the resources to deploy your pod. There are no longer any instances available with the requested specifications." });
        const id = newId();
        const cpu = body.computeType === "CPU";
        const pod = { id, name: body.name, payload: body, env: body.env, image: body.imageName, desiredStatus: "RUNNING", costPerHr: cpu ? 0.06 * (body.vcpuCount || 2) / 2 : 2.09, created: Date.now() };
        m.pods.set(id, pod);
        return json(res, 200, { id, name: body.name, costPerHr: pod.costPerHr, desiredStatus: "RUNNING", machine: { dataCenterId: (body.dataCenterIds || ["EU-RO-1"])[0] }, env: body.env });
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
      // Serverless templates and endpoints (src/serverless/runpod-sls.ts).
      const sls = m.sls;
      if (rest === "/templates" && req.method === "POST") {
        const id = `tpl${newId()}`;
        sls.templates.set(id, { id, ...body });
        return json(res, 200, { id, ...body });
      }
      let st = /^\/templates\/([^/]+)$/.exec(rest);
      if (st) {
        const t = sls.templates.get(st[1]);
        if (!t) return json(res, 400, { error: "template not found" });
        if (req.method === "PATCH") { Object.assign(t, body); t.patches = (t.patches || 0) + 1; return json(res, 200, t); }
        if (req.method === "DELETE") { sls.templates.delete(st[1]); return json(res, 200, {}); }
      }
      if (rest === "/endpoints" && req.method === "POST") {
        const id = newId();
        sls.endpoints.set(id, { id, ...body, workers: [] });
        return json(res, 200, { id, name: body.name });
      }
      st = /^\/endpoints\/([^/]+)$/.exec(rest);
      if (st) {
        const e = sls.endpoints.get(st[1]);
        if (!e) return json(res, 404, { error: "endpoint not found" });
        if (req.method === "GET") return json(res, 200, e);
        if (req.method === "PATCH") { Object.assign(e, body); return json(res, 200, e); }
        if (req.method === "DELETE") { sls.endpoints.delete(st[1]); return json(res, 200, {}); }
      }
      return json(res, 404, { error: "no route" });
    }
    if (p.startsWith("/rp/rest2/")) {
      if (bearer !== m.runpodKey) return json(res, 401, { error: "bad key" });
      const rest = p.slice("/rp/rest2".length);
      if (rest === "/catalog/gpus") return json(res, 200, { gpus: [{ id: "NVIDIA RTX PRO 6000 Blackwell Server Edition", pool: "BLACKWELL_96" }, { id: "NVIDIA H100 80GB HBM3", pool: "HOPPER_141" }] });
      if (rest === "/serverless" && req.method === "POST") {
        // v2 makes the endpoint's template itself.
        const id = newId();
        const tid = `tpl${newId()}`;
        m.sls.templates.set(tid, { id: tid, v2: true, name: `${body.name}-template`, imageName: body.image, env: body.env, args: body.args });
        m.sls.endpoints.set(id, { id, name: body.name, templateId: tid, v2: body, workers: [] });
        return json(res, 200, { id, name: body.name, templateId: tid });
      }
      return json(res, 404, { error: "no route" });
    }
    if (p === "/rp/graphql") {
      if (bearer !== m.runpodKey) return json(res, 401, { errors: [{ message: "bad key" }] });
      const q = body.query || "";
      if (q.includes("dataCenters") && q.includes("ramMultiplier"))
        return json(res, 200, { data: { dataCenters: ["EU-RO-1", "EUR-IS-1", "EU-NL-1", "US-CA-2", "AP-IN-2"].map((id) => ({ id, listed: id !== "AP-IN-2" })), cpuFlavors: [{ id: "cpu3c", ramMultiplier: 2 }, { id: "cpu5c", ramMultiplier: 2 }, { id: "cpu3g", ramMultiplier: 4 }] } });
      // GPU stock per data centre (src/instance-types.ts, the launch table): every data centre's gpuAvailability.
      if (q.includes("gpuAvailability"))
        return m.runpodDcsFail ? json(res, 200, { errors: [{ message: "Cannot query field gpuAvailability" }] }) : json(res, 200, { data: { dataCenters: m.runpodDcs } });
      if (q.includes("specifics(")) {
        const data = {};
        for (const mm of q.matchAll(/(a\d+): cpuFlavors \{ id specifics\(input: \{dataCenterId: "([^"]+)", instanceId: "([^"]+)"\}\)/g)) {
          const st = m.cpuStock[`${mm[2]}|${mm[3]}`];
          const fl = mm[3].split("-")[0];
          data[mm[1]] = ["cpu3c", "cpu5c", "cpu3g"].map((id) => ({ id, specifics: { stockStatus: id === fl && st ? st[0] : null, securePrice: id === fl && st ? st[1] : 9 } }));
        }
        return json(res, 200, { data });
      }
      // Stock per GPU type and data centre (src/cluster/editor.ts): aliased gpuTypes with lowestPrice(dataCenterId).
      if (q.includes("dataCenterId") && /g\d+: gpuTypes/.test(q)) {
        const data = {};
        for (const part of q.split(/(?=g\d+: gpuTypes)/).slice(1)) {
          const mm = /^(g\d+): gpuTypes\(input: \{id: "([^"]+)"\}\).*dataCenterId: "([^"]+)"/s.exec(part);
          if (!mm) continue;
          const st = m.gpuStock[`${mm[3]}|${mm[2]}`] ?? ["High", 8];
          data[mm[1]] = [{ id: mm[2], securePrice: mm[2].includes("H200") ? 3.59 : 2.09, lowestPrice: { stockStatus: st[0], ...(part.includes("maxUnreservedGpuCount") ? { maxUnreservedGpuCount: st[1] } : {}) } }];
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
        machine: { gpuDisplayName: x.payload.computeType === "CPU" ? "unknown" : "RTX PRO 6000", dataCenterId: (x.payload.dataCenterIds || ["EU-RO-1"])[0] },
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
    // ---- GMI Cloud
    if (p.startsWith("/gmi/v1/")) {
      const g = m.gmi;
      const rest = p.slice("/gmi/v1".length);
      g.calls.push({ method: req.method, path: rest, auth: bearer === m.gmiKey });
      if (bearer !== m.gmiKey) return json(res, 401, { code: 0, group: "auth_verify", message: "invalid token" });
      if (rest === "/containers/products") return json(res, 200, g.products.filter((x) => !url.searchParams.get("idc") || x.idc === url.searchParams.get("idc")));
      if (rest === "/templates" && req.method === "GET") return json(res, 200, g.templates);
      if (rest === "/templates" && req.method === "POST") {
        if (!/^([A-Za-z0-9][A-Za-z0-9_\-. ]*)?[A-Za-z0-9]$/.test(body?.name || "") || !body?.path) return json(res, 400, { reason: "bad template" });
        const t = { id: `tpl-${g.templates.length + 1}`, ...body };
        g.templates.push(t);
        return json(res, 200, { id: t.id });
      }
      if (rest === "/containers" && req.method === "GET") return json(res, 200, g.containers);
      if (rest === "/containers" && req.method === "POST") {
        if (g.failCreate > 0) { g.failCreate--; return json(res, 503, { reason: "no capacity for the product" }); }
        for (const k of ["name", "templateId", "product", "idc"]) if (!body?.[k]) return json(res, 400, { reason: `${k} is required` });
        if (!/^([A-Za-z0-9][A-Za-z0-9_\-. ]*)?[A-Za-z0-9]$/.test(body.name)) return json(res, 400, { reason: "bad name" });
        if (!g.templates.some((t) => t.id === body.templateId)) return json(res, 404, { reason: "template not found" });
        if (!g.products.some((x) => x.name === body.product)) return json(res, 400, { reason: `no product ${body.product}` });
        const id = `00000000-0000-4000-8000-${String(g.containers.length + 1).padStart(12, "0")}`;
        const env = Object.fromEntries((body.envs || []).map((e) => [e.name, e.value]));
        g.containers.push({ id, name: body.name, status: "running", reason: "", product: body.product, idc: body.idc, templateId: body.templateId, createdAt: new Date().toISOString(), envs: body.envs, command: body.command, args: body.args, ports: body.ports });
        simBoot(`gmi:${body.name}`, env, g.boot, { command: body.command, args: body.args, image: g.templates.find((t) => t.id === body.templateId)?.path });
        return json(res, 200, [{ id }]);
      }
      let gm = /^\/containers\/([^/]+)(\/logs)?$/.exec(rest);
      if (gm) {
        const c = g.containers.find((x) => x.id === gm[1]);
        if (!c) return json(res, 404, { reason: "container not found" });
        if (gm[2]) {
          // A container log that leaks the key and (a named tunnel's pod) its tunnel token: fv-control must scrub both.
          const tt = (c.envs || []).find((e) => e.name === "FV_TUNNEL_TOKEN")?.value;
          res.writeHead(200, { "content-type": "text/plain" });
          return res.end(`[fv-boot] start (gmi gmi:${c.name})\n[fv-boot] tunnel ok\nleak? ${m.gmiKey}\n${tt ? `leak? TUNNEL_TOKEN=${tt}\n` : ""}`);
        }
        if (req.method === "GET") return json(res, 200, c);
        if (req.method === "DELETE") {
          g.containers = g.containers.filter((x) => x.id !== c.id);
          m.pods.delete(`gmi:${c.name}`);
          return json(res, 200, { result: "deleted" });
        }
      }
      return json(res, 404, { reason: `mock: no GMI route ${req.method} ${rest}` });
    }
    // ---- NVIDIA Brev (the CLI's API)
    if (p.startsWith("/brev/api/")) {
      const b = m.brev;
      const rest = p.slice("/brev/api".length);
      b.calls.push({ method: req.method, path: rest, auth: bearer === m.brevToken });
      if (bearer !== m.brevToken) return json(res, 401, { message: "unauthorized" });
      const om = /^\/organizations\/([^/]+)\/workspaces$/.exec(rest);
      if (om && om[1] !== m.brevOrg) return json(res, 403, { message: "not a member" });
      if (om && req.method === "GET") return json(res, 200, b.workspaces.map(({ startupScript, body: _b, disk: _d, ...w }) => w));
      // The instance-type listing (shape read live 2026-10-10): each type's cloud_cred_id, stoppable, prices.
      const tm = /^\/instances\/alltypesavailable\/([^/]+)$/.exec(rest);
      if (tm && req.method === "GET") return tm[1] !== m.brevOrg ? json(res, 403, { message: "not a member" }) : json(res, 200, { allInstanceTypes: b.types });
      if (om && req.method === "POST") {
        // brev-cli main's body: the old one (vmOnlyMode + startupScript, no workspaceVersion) is refused as live.
        // Live (2026-10-10): a body missing any of Go's non-omitempty fields got this 400 even with workspaceVersion v1.
        const full = ["description", "primaryApplicationId", "applications", "startupScript", "gitRepo", "initBranch", "startupScriptPath", "dotBrevPath", "baseImage", "vmOnlyMode", "portMappings", "execsV1", "reposV1", "labels", "files", "launchJupyterOnStart", "diskStorage", "isStoppable"];
        if (body?.workspaceVersion !== "v1" || full.some((k) => !(k in body)) || body.vmOnlyMode !== false || body.startupScript !== "")
          return json(res, 400, { errors: [{ type: "BadRequestError", message: "Legacy workspace version unsupported" }] });
        const type = b.types.find((t) => t.type === body.instanceType);
        const script = body?.vmBuild?.lifeCycleScriptAttr?.script;
        if (!body?.name || !type || body.cloudCredId !== type.cloud_cred_id || !body.workspaceTemplateId || !body.workspaceClassId || !script)
          return json(res, 400, { errors: [{ type: "BadRequestError", message: "name, instanceType (listed), its cloudCredId, workspaceTemplateId, workspaceClassId, vmBuild.lifeCycleScriptAttr.script" }] });
        const id = `ws${String((b.seq = (b.seq || 0) + 1)).padStart(6, "0")}`;
        const w = { id, name: body.name, status: "RUNNING", healthStatus: "HEALTHY", instanceType: body.instanceType, dns: `${id}.brev.example`, createdAt: new Date().toISOString(), startupScript: script, body, disk: {}, diskStorage: body.diskStorage };
        b.workspaces.push(w);
        brevVmBoot(w);
        return json(res, 201, { id, name: body.name, status: "DEPLOYING", instanceType: body.instanceType, workspaceVersion: "v1" });
      }
      const wm = /^\/workspaces\/([^/]+)(\/(stop|start))?$/.exec(rest);
      if (wm) {
        const w = b.workspaces.find((x) => x.id === wm[1]);
        if (!w) return json(res, 404, { message: "workspace not found" });
        if (req.method === "GET" && !wm[3]) { const { startupScript, body: _b, disk: _d, ...o } = w; return json(res, 200, o); }
        if (req.method === "DELETE" && !wm[3]) {
          b.workspaces = b.workspaces.filter((x) => x.id !== w.id);
          m.pods.delete(`brev:${w.name}`);
          return json(res, 200, {});
        }
        // Stop keeps the disk (w.disk: the trees on it); start fails without capacity, or hangs, or boots again.
        if (req.method === "PUT" && wm[3] === "stop") {
          const t = b.types.find((x) => x.type === w.instanceType);
          if (b.stopFail) return json(res, 400, { errors: [{ type: "BadRequestError", message: "stop refused (simulated API error)" }] });
          // Measured live 2026-10-13: a workspace still DEPLOYING (never provisioned), and a RUNNING one of a type Brev
          // cannot stop (isStoppable false): HTTP 500, the VM stays RUNNING / HEALTHY.
          if (w.status === "DEPLOYING") return json(res, 500, { errors: [{ type: "InternalServerError", message: "rpc error: code = NotFound desc = instance not found" }] });
          if (!t?.stoppable) return json(res, 500, { errors: [{ type: "InternalServerError", message: `rpc error: code = Internal desc = instance "inst-${w.id}" does not support stop` }] });
          w.status = "STOPPED";
          m.pods.delete(`brev:${w.name}`);
          return json(res, 200, { id: w.id, status: "STOPPING" });
        }
        if (req.method === "PUT" && wm[3] === "start") {
          if (b.startFail) return json(res, 409, { errors: [{ type: "ConflictError", message: "no capacity in the same provider/region" }] });
          if (w.status !== "STOPPED") return json(res, 409, { errors: [{ type: "ConflictError", message: `workspace is ${w.status}` }] });
          w.status = b.startHang ? "STARTING" : "RUNNING";
          if (!b.startHang) brevVmBoot(w);
          return json(res, 200, { id: w.id, status: "STARTING" });
        }
      }
      return json(res, 404, { message: `mock: no Brev route ${req.method} ${rest}` });
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
    // ---- Runpod's serverless queue API
    const qm = /^\/rpq\/v2\/([^/]+)\/(health|run|purge-queue|status\/(.+)|cancel\/(.+))$/.exec(p);
    if (qm) {
      if (bearer !== m.runpodKey) return json(res, 401, { error: "unauthorized" });
      const q = m.queue;
      if (qm[2] === "health") return json(res, 200, { jobs: { inQueue: q.queued, inProgress: q.running, completed: 3, failed: 0, retried: 0 }, workers: { idle: q.workers, running: 0, ready: 0, initializing: 0, throttled: 0, unhealthy: 0 } });
      if (qm[2] === "purge-queue" && req.method === "POST") {
        const removed = q.queued;
        q.queued = 0;
        q.purges++;
        for (const j of q.jobs.values()) if (j.status === "IN_QUEUE") j.status = "CANCELLED";
        return json(res, 200, { removed, status: "completed" });
      }
      if (qm[2] === "run" && req.method === "POST") {
        const id = `mock-run-${q.ran.length + 1}`;
        q.ran.push(body);
        // q.onRun(input): the states the job goes through, one per status poll (a simulated worker, fakeServe()); none: queued for good.
        q.jobs.set(id, { id, status: "IN_QUEUE", plan: q.onRun ? q.onRun(body.input, id) : undefined });
        return json(res, 200, { id, status: "IN_QUEUE" });
      }
      const j = q.jobs.get(decodeURIComponent(qm[3] || qm[4] || ""));
      if (!j) return json(res, 404, { error: "request does not exist" });
      if (qm[4] && req.method === "POST") {
        if (j.status === "IN_QUEUE") q.queued = Math.max(0, q.queued - 1);
        j.status = "CANCELLED";
        j.plan = undefined;
      } else if (j.plan?.length) Object.assign(j, j.plan.shift());
      const { plan: _plan, ...view } = j;
      return json(res, 200, view);
    }
    // ---- media a simulated worker's results point at (R2 presigned URLs stand-in)
    if (p.startsWith("/media/")) {
      m.media = (m.media || 0) + 1;
      res.writeHead(200, { "content-type": "video/mp4" });
      return res.end(Buffer.from("00000018667479706d703432", "hex"));
    }
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
      const ij = /^\/fv\/v1\/internal\/jobs\/([^/]+)$/.exec(route);
      if (ij && req.method === "DELETE") {
        const id = decodeURIComponent(ij[1]);
        m.internalCancels.push({ pod: pod.id, id });
        const a = m.internalJobs[id];
        return a ? json(res, a[0], a[1]) : json(res, 404, { error: { kind: "not_found", message: `job \`${id}\` was not found` } });
      }
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
    RUNPOD_REST: `${base}/rp/rest`, RUNPOD_REST2: `${base}/rp/rest2`, RUNPOD_QUEUE: `${base}/rpq/v2`, RUNPOD_GRAPHQL: `${base}/rp/graphql`, RUNPOD_HAPI: `${base}/rp/hapi`, CLOUDRIFT_API: `${base}/cr`, GMI_API: `${base}/gmi`, BREV_API_URL: `${base}/brev`, FV_ENDPOINT_URL_RE: "^http://127\\.0\\.0\\.1:\\d+/pod/(gmi|brev):fv-[a-z0-9-]+$", GITHUB_API: `${base}/gh`, GHCR: `${base}/ghcr`, CF_API: `${base}/cf`,
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

/** SQL on a local D1 of the Worker: fv-control (default), fv-jobs or fv-edge (the jobs D1s, wrangler.toml). */
export function d1Exec(dir, sql, db = "fv-control") {
  return execFileSync(join(HERE, "node_modules/.bin/wrangler"), ["d1", "execute", db, "--local", "--persist-to", dir, "--json", "--command", sql], { cwd: HERE, stdio: "pipe", env: { ...process.env, CI: "1" } }).toString();
}

export const SECRETS = {
  RUNPOD_API_KEY: "rpa_TESTKEY_0123456789abcdef",
  GITHUB_PAT: "github_pat_TEST_0123456789",
  CLOUDRIFT_API_KEY: "crk_TEST_0123456789abcdef",
  CLOUDFLARE_API_KEY: "cf_TEST_0123456789abcdef",
  GMI_API_KEY: "gmi_TEST_0123456789abcdef",
  BREV_API_TOKEN: "brev_TEST_0123456789abcdef",
  CONTROL_KEK: b64(wc.getRandomValues(new Uint8Array(32))),
  SESSION_SECRET: "sess_" + Buffer.from(wc.getRandomValues(new Uint8Array(24))).toString("hex"),
};
export const PASSPHRASE = "test passphrase for the owner 42";

/** The jobs table fv-serve workers write (crates/fastvideo-serve-kit/src/d1/schema.rs), for the local fv-jobs / fv-edge. */
export const JOBS_TABLE = "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY NOT NULL, protocol TEXT NOT NULL, external_id TEXT NOT NULL, owner TEXT, status TEXT NOT NULL, model TEXT NOT NULL, resolved_model TEXT NOT NULL, task TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, completed_at INTEGER, expires_at INTEGER NOT NULL, worker TEXT, version INTEGER NOT NULL DEFAULT 0, job TEXT NOT NULL, UNIQUE (protocol, external_id))";
/** One jobs row (a worker's D1 write) as SQL. */
export function jobInsert(j) {
  const q = (v) => (v === null || v === undefined ? "NULL" : typeof v === "number" ? String(v) : `'${String(v).replace(/'/g, "''")}'`);
  const job = JSON.stringify({ id: j.id, started_at: j.started ?? null, cancel_requested: false, request_echo: { model: j.model, image_url: "https://pod/files/x?sig=SIGNED_URL_SECRET" } });
  return `INSERT INTO jobs (id, protocol, external_id, owner, status, model, resolved_model, task, progress, created_at, updated_at, expires_at, worker, job) VALUES (${[j.id, j.api || "native", j.ext, "key_000000000001", j.status, j.model || "fake-wan", "fake-wan", "t2v", j.progress ?? 0, j.at, j.at, j.at + 86400_000, j.worker ?? null, job].map(q).join(", ")})`;
}

/**
 * A simulated fv-serve queue worker for the serverless console (src/serverless/console.ts): what each
 * `kind: http` job's status polls show (mock.queue.onRun = fakeServe(mock)). GETs answer at once; a
 * waiting submit is queued, then running (progress `{state, poll_path}`, as dispatch.rs reports it),
 * then done with the worker's final reply (`steps` polls each). Results point at the mock's /media/.
 */
export function fakeServe(mock, { steps = 2 } = {}) {
  const media = (id) => `http://127.0.0.1:${mock.port}/media/${id}.mp4`;
  const caps = {
    object: "fv.capabilities",
    auth: { mode: "trust-gateway" },
    protocols: { native: true, fal: true, openai_videos: true, minimax: false, reactor: false },
    models: [{ caps: { id: "fake-wan", served_names: ["fake-wan"], tier: "turbo", tasks: ["t2v", "i2v"], knobs: { seed: true, steps: true }, frames: { min: 9, max: 81, step: 4, offset: 1 }, fps: { default: 16, allowed: [16] }, canvas: { short_edges: [480] } }, recipe: { steps: 4 } }],
    tiers: [],
    aliases: { "fake-wan-alias": "fake-wan" },
  };
  const catalog = { apps: [{ id: "fastvideo/fake-wan", model: "fake-wan", endpoints: [{ sub: "text-to-video", title: "Text to video" }, { sub: "image-to-video", title: "Image to video" }] }] };
  const schema = (sub) => ({
    type: "object",
    title: sub,
    required: sub === "image-to-video" ? ["prompt", "image_url"] : ["prompt"],
    properties: { prompt: { type: "string", title: "Prompt" }, ...(sub === "image-to-video" ? { image_url: { type: "string", title: "Image", "x-fv-media": "image" } } : {}), seed: { type: "integer", title: "Seed" } },
  });
  const reply = (status, body) => ({ status: "COMPLETED", output: { status, headers: { "content-type": "application/json" }, body, elapsed_s: 0.01 } });
  let n = 0;
  return (input) => {
    if (!input || input.kind !== "http") return [{ status: "COMPLETED", output: { engine: "fake" } }];
    const p = String(input.path || "");
    const method = String(input.method || "POST").toUpperCase();
    mock.served = [...(mock.served || []), `${method} ${p}`];
    if (method === "GET") {
      if (p === "/fv/v1/capabilities") return [reply(200, caps)];
      if (p === "/fal/schema") return [reply(200, catalog)];
      const s = /^\/fal\/schema\/fastvideo\/fake-wan\/(.+)$/.exec(p);
      if (s) return [reply(200, schema(s[1]))];
      return [reply(404, { error: { kind: "not_found", message: `no route ${p}` } })];
    }
    if (method !== "POST") return [reply(202, { status: "CANCELLATION_REQUESTED" })];
    const b = input.body || {};
    const k = ++n;
    const hold = (x) => Array.from({ length: steps }, () => x);
    if (!b.prompt) return [reply(422, { error: { kind: "invalid_request", message: "prompt is required" } })];
    if (p === "/fv/v1/jobs" || p === "/v1/videos") {
      const id = p === "/fv/v1/jobs" ? `fvjob_sim${k}` : `video_sim${k}`;
      const pp = `${p}/${id}`;
      const final = p === "/fv/v1/jobs"
        ? { id, object: "fv.job", status: "succeeded", model: b.model, task: b.task || "t2v", output: { url: media(id), mime: "video/mp4", width: 832, height: 480, frames: 33, fps: 16 }, metrics: { inference_s: 1.25 } }
        : { id, object: "video", status: "completed", model: b.model, progress: 100, url: media(id) };
      return [...hold({ status: "IN_PROGRESS", output: { state: "queued", poll_path: pp } }), ...hold({ status: "IN_PROGRESS", output: { state: "running", poll_path: pp } }), { status: "COMPLETED", output: { status: 200, headers: {}, body: final, submit: { id, status: "queued" }, poll_path: pp, elapsed_s: 2 } }];
    }
    // fal: POST /{app}/{sub}
    const id = `f-sim-${k}`;
    const app = p.split("/").slice(1, 3).join("/");
    const sp = `/${app}/requests/${id}/status`;
    return [
      ...hold({ status: "IN_PROGRESS", output: { state: "in_queue", poll_path: sp } }),
      ...hold({ status: "IN_PROGRESS", output: { state: "in_progress", poll_path: sp } }),
      { status: "COMPLETED", output: { status: 200, headers: { "content-type": "application/json" }, body: { video: { url: media(id), content_type: "video/mp4", file_name: `${id}.mp4`, file_size: 12 }, seed: 42, timings: { inference: 1.25 } }, submit: { request_id: id }, poll_path: sp, elapsed_s: 2 } },
    ];
  };
}
