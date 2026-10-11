// Named pod tunnels (src/tunnels.ts, docs/serve/deploy-gmi-brev.md §6.4): the settings and the two-level host
// warning, the host slug, the create / rollback / only-ours delete against a fetch mock of the Cloudflare API, the
// boot's tunnel section in bash with a stub cloudflared, and the token scrub.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { CLOUDFLARED, providerIssues, TUNNEL_SH } from "../../src/providers";
import { destroyTunnel, ensureTunnel, resetZoneCache, tunnelHost, tunnelSettings, type PodTunnel } from "../../src/tunnels";
import { scrub } from "../../src/util";
import { d1 } from "./d1shim";

const API = "cftun_SECRET_0123456789";
const named = { CF_TUNNEL_API_TOKEN: API, CF_ACCOUNT_ID: "acct", FV_TUNNEL_ZONE: "maximal.llc", FV_TUNNEL_HOST_TEMPLATE: "{pod}-fv-pods.maximal.llc", CF_API: "https://cf.test/client/v4" };
const mkEnv = (over: Partial<Env> = {}): Env =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_x",
    PUBLIC_URL: "https://fvc.test",
    GMI_API_KEY: "gmi_x_0123456789",
    GMI_PRODUCTS: "container.h200.x1",
    GMI_BUDGET_USD: "20",
    ...over,
  }) as Env;

describe("settings", () => {
  it("unset: the quick-tunnel fallback, nothing to say", () => {
    expect(tunnelSettings(mkEnv())).toMatchObject({ mode: "quick", warning: null, error: null });
  });
  it("half set: quick, and it says what is missing", () => {
    const s = tunnelSettings(mkEnv({ FV_TUNNEL_HOST_TEMPLATE: "{pod}-fv-pods.maximal.llc", FV_TUNNEL_ZONE: "maximal.llc" }));
    expect(s.mode).toBe("quick");
    expect(s.missing).toEqual(["CF_ACCOUNT_ID", "CF_TUNNEL_API_TOKEN"]);
    expect(s.warning).toMatch(/named pod tunnels need CF_ACCOUNT_ID, CF_TUNNEL_API_TOKEN: GMI \/ Brev pods use quick tunnels/);
  });
  it("one level below the zone: named, no warning", () => {
    expect(tunnelSettings(mkEnv(named))).toMatchObject({ mode: "named", levels: 1, warning: null, error: null });
  });
  it("two levels below the zone: a loud warning naming Advanced Certificate Manager (Universal SSL covers *.zone only)", () => {
    const s = tunnelSettings(mkEnv({ ...named, FV_TUNNEL_HOST_TEMPLATE: "{pod}.fv-pods.maximal.llc" }));
    expect(s).toMatchObject({ mode: "named", levels: 2, error: null });
    expect(s.warning).toMatch(/2 levels below maximal\.llc: Cloudflare's Universal SSL covers only \*\.maximal\.llc/);
    expect(s.warning).toMatch(/Advanced Certificate Manager certificate for \*\.fv-pods\.maximal\.llc/);
    expect(s.warning).toMatch(/Use one level, e\.g\. \{pod\}-fv-pods\.maximal\.llc/);
  });
  it("an unusable template is an error that refuses GMI / Brev launches", () => {
    for (const [tpl, re] of [
      ["fv-pods.maximal.llc", /\{pod\} exactly once/],
      ["{pod}-{pod}.maximal.llc", /\{pod\} exactly once/],
      ["{pod}.example.com", /must end with \.maximal\.llc/],
      ["{pod}_x.maximal.llc", /not a DNS label/],
      [`{pod}${"x".repeat(56)}.maximal.llc`, /too little room/],
    ] as const) {
      const env = mkEnv({ ...named, FV_TUNNEL_HOST_TEMPLATE: tpl });
      expect(tunnelSettings(env).error, tpl).toMatch(re);
      const iss = providerIssues(env, { pools: [{ id: "a", provider: "gmi", provider_gpu: "container.h200.x1", count: 1 } as any] });
      expect(iss.some((i) => /named pod tunnels: FV_TUNNEL_HOST_TEMPLATE/.test(i.message)), tpl).toBe(true);
    }
  });
});

describe("host", () => {
  it("a DNS-safe slug of the pod name + 6 hex; the tunnel is named fv-…", () => {
    const h = tunnelHost("{pod}-fv-pods.maximal.llc", "fv-pod-My_Pod.1-1011120000", "abc123");
    expect(h).toEqual({ name: "fv-pod-my-pod-1-1011120000-abc123", host: "fv-pod-my-pod-1-1011120000-abc123-fv-pods.maximal.llc" });
    expect(tunnelHost("{pod}.maximal.llc", "x", "abc123").name).toBe("fv-x-abc123");
  });
  it("a label stays ≤ 63 characters, the random suffix kept", () => {
    const h = tunnelHost("{pod}-fv-pods.maximal.llc", `fv-ctl-${"a".repeat(80)}-pool-1011120000`);
    const label = h.host.split(".")[0]!;
    expect(label.length).toBeLessThanOrEqual(63);
    expect(label).toMatch(/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/);
    expect(label).toMatch(/-[0-9a-f]{6}-fv-pods$/);
    expect(tunnelHost("{pod}-fv-pods.maximal.llc", "fv-pod-a-1").host).not.toBe(tunnelHost("{pod}-fv-pods.maximal.llc", "fv-pod-a-1").host);
  });
});

// ---------------------------------------------------------------- the Cloudflare API (fetch mock)
interface Call {
  method: string;
  path: string;
  body: any;
}
function cfMock(o: { failDns?: boolean; renamed?: boolean } = {}) {
  const calls: Call[] = [];
  const tunnels = new Map<string, { id: string; name: string; deleted_at: string | null }>();
  const recs = new Map<string, any>();
  const ok = (result: unknown) => new Response(JSON.stringify({ success: true, errors: [], result }), { status: 200 });
  const err = (status: number, code: number, message: string) => new Response(JSON.stringify({ success: false, errors: [{ code, message }], result: null }), { status });
  vi.stubGlobal("fetch", async (input: string, init: RequestInit = {}) => {
    const u = new URL(input);
    const path = u.pathname.replace("/client/v4", "") + u.search;
    const method = init.method || "GET";
    const body = init.body ? JSON.parse(String(init.body)) : null;
    calls.push({ method, path, body });
    if ((init.headers as Record<string, string>).authorization !== `Bearer ${API}`) return err(403, 10000, "Authentication error");
    if (path === "/zones?name=maximal.llc") return ok([{ id: "z1", name: "maximal.llc" }]);
    if (path === "/accounts/acct/cfd_tunnel" && method === "POST") {
      const t = { id: `t${tunnels.size + 1}`, name: body.name, deleted_at: null };
      tunnels.set(t.id, t);
      return ok(t);
    }
    let m = /^\/accounts\/acct\/cfd_tunnel\/([^/]+)(\/\w+)?$/.exec(path);
    if (m) {
      const t = tunnels.get(m[1]!);
      if (!t) return err(404, 1003, "not found");
      if (!m[2] && method === "GET") return ok(o.renamed ? { ...t, name: "someone-else" } : t);
      if (!m[2] && method === "DELETE") return (t.deleted_at = "now"), ok(t);
      if (m[2] === "/configurations") return ok({});
      if (m[2] === "/token") return ok(`eyJhIjoiYWNjdCIsInQiOiJ${t.id}SECRETTUNNELTOKEN`);
      if (m[2] === "/connections") return ok(null);
    }
    if (path === "/zones/z1/dns_records" && method === "POST") {
      if (o.failDns) return err(400, 81053, "An A, AAAA, or CNAME record with that host already exists.");
      const r = { id: `r${recs.size + 1}`, ...body };
      recs.set(r.id, r);
      return ok(r);
    }
    m = /^\/zones\/z1\/dns_records\/([^/]+)$/.exec(path);
    if (m) {
      const r = recs.get(m[1]!);
      if (!r) return err(404, 81044, "Record does not exist.");
      if (method === "GET") return ok(r);
      if (method === "DELETE") return recs.delete(r.id), ok({ id: r.id });
    }
    return err(404, 7000, `no route ${method} ${path}`);
  });
  return { calls, tunnels, recs };
}
const row = async (env: Env, name: string) => env.DB.prepare("SELECT * FROM pod_tunnels WHERE name = ?").bind(name).first<PodTunnel>();

describe("create and delete (fetch mock)", () => {
  beforeEach(() => resetZoneCache());
  afterEach(() => vi.unstubAllGlobals());
  it("tunnel, ingress, proxied CNAME with fv-control's comment, token; recorded live; reused for the same pod", async () => {
    const cf = cfMock();
    const env = mkEnv(named);
    const g = await ensureTunnel(env, { podId: "gmi:fv-pod-a-1", podName: "fv-pod-a-1", clusterId: "c1" });
    expect(g.url).toBe(`https://${g.host}`);
    expect(g.token).toMatch(/^eyJhIjoi/);
    expect(cf.calls.map((c) => `${c.method} ${c.path.replace(/\/t\d+/, "/<t>")}`)).toEqual([
      "GET /zones?name=maximal.llc",
      "POST /accounts/acct/cfd_tunnel",
      "PUT /accounts/acct/cfd_tunnel/<t>/configurations",
      "POST /zones/z1/dns_records",
      "GET /accounts/acct/cfd_tunnel/<t>/token",
    ]);
    expect(cf.calls[1]!.body).toEqual({ name: g.name, config_src: "cloudflare" });
    expect(cf.calls[2]!.body).toEqual({ config: { ingress: [{ hostname: g.host, service: "http://127.0.0.1:8000" }, { service: "http_status:404" }] } });
    expect(cf.calls[3]!.body).toEqual({ type: "CNAME", name: g.host, content: "t1.cfargotunnel.com", proxied: true, ttl: 1, comment: "fv-control gmi:fv-pod-a-1" });
    const r = await row(env, g.name);
    expect(r).toMatchObject({ pod_id: "gmi:fv-pod-a-1", state: "live", tunnel_id: "t1", dns_record_id: "r1", zone_id: "z1", host: g.host });
    expect(JSON.stringify(r)).not.toContain(g.token);
    // The same pod again (a parked Brev instance restarted): the same tunnel, a fresh token read, nothing made.
    const n = cf.calls.length;
    const g2 = await ensureTunnel(env, { podId: "gmi:fv-pod-a-1", podName: "fv-pod-a-1", clusterId: "c2" });
    expect([g2.reused, g2.name, g2.host]).toEqual([true, g.name, g.host]);
    expect(cf.calls.slice(n).map((c) => c.method)).toEqual(["GET", "GET"]);
    // Delete: record, connections, tunnel.
    await destroyTunnel(env, (await row(env, g.name))!, "test");
    expect(cf.calls.filter((c) => c.method === "DELETE").map((c) => c.path)).toEqual(["/zones/z1/dns_records/r1", "/accounts/acct/cfd_tunnel/t1/connections", "/accounts/acct/cfd_tunnel/t1"]);
    expect((await row(env, g.name))!.state).toBe("deleted");
  });
  it("a refused DNS record: the tunnel just made is deleted, the row too, the error says so", async () => {
    const cf = cfMock({ failDns: true });
    const env = mkEnv(named);
    await expect(ensureTunnel(env, { podId: "gmi:fv-pod-b-1", podName: "fv-pod-b-1", clusterId: null })).rejects.toThrow(/named tunnel fv-pod-b-1-[0-9a-f]{6}: cloudflare POST \/zones\/z1\/dns_records: 400 81053/);
    expect([...cf.tunnels.values()].map((t) => t.deleted_at)).toEqual(["now"]);
    const r = await env.DB.prepare("SELECT * FROM pod_tunnels").first<PodTunnel>();
    expect(r!.state).toBe("deleted");
    expect(r!.note).toMatch(/^create failed: /);
  });
  it("only ours: a tunnel whose name is not the recorded one is not deleted (delete_failed, the cron retries)", async () => {
    const env = mkEnv(named);
    cfMock();
    const g = await ensureTunnel(env, { podId: "gmi:fv-pod-c-1", podName: "fv-pod-c-1", clusterId: null });
    const cf = cfMock({ renamed: true });
    // (a new mock: the tunnel it lists is "someone-else"; seed it so the GET finds it)
    cf.tunnels.set("t1", { id: "t1", name: g.name, deleted_at: null });
    await expect(destroyTunnel(env, { ...(await row(env, g.name))!, dns_record_id: null }, "test")).rejects.toThrow(/named someone-else, not fv-pod-c-1-/);
    expect(cf.calls.some((c) => c.method === "DELETE" && c.path.includes("cfd_tunnel"))).toBe(false);
    expect((await row(env, g.name))!.state).toBe("delete_failed");
  });
});

describe("the boot's tunnel section, in bash with a stub cloudflared", () => {
  const run = (env: Record<string, string>) => {
    const dir = mkdtempSync(join(tmpdir(), "fvtun-"));
    const cfd = join(dir, "cloudflared");
    // The stub records its argv and TUNNEL_TOKEN, then logs what the real one does (a registered connection or a quick URL).
    writeFileSync(
      cfd,
      `#!/bin/bash\nprintf '%s\\n' "$*" > ${dir}/argv\nprintf '%s' "\${TUNNEL_TOKEN:-}" > ${dir}/token_env\ncase "$*" in *run*) echo "2026-10-11T00:00:00Z INF Registered tunnel connection connIndex=0 protocol=quic" ;; *) echo "INF |  https://abc-def.trycloudflare.com  |" ;; esac\n`,
      { mode: 0o755 },
    );
    const body = TUNNEL_SH.split("\n")
      .filter((l) => !l.includes(CLOUDFLARED.url) && !l.includes(CLOUDFLARED.sha256) && !l.startsWith("chmod "))
      .join("\n")
      .replaceAll("/usr/local/bin/cloudflared", cfd)
      .replaceAll("/fvstate/", `${dir}/`);
    const script = `set -u
say() { echo "$*" >&2; }
report() { printf '%s|%s|%s\\n' "$1" "$2" "$3" >> ${dir}/reports; }
fail() { echo "FAIL|$1" >> ${dir}/reports; exit 1; }
${body}
echo "after|\${FV_TUNNEL_TOKEN:-unset}|\${FV_DISPATCH_ENDPOINT:-}" >> ${dir}/reports
`;
    writeFileSync(join(dir, "t.sh"), script);
    execFileSync("bash", [join(dir, "t.sh")], { env: { PATH: process.env.PATH!, ...env }, stdio: "pipe", timeout: 20000 });
    const rd = (f: string) => readFileSync(join(dir, f), "utf8");
    return { reports: rd("reports").trim().split("\n"), argv: rd("argv").trim(), tokenEnv: rd("token_env") };
  };
  it("named: the token in cloudflared's environment (not argv), dropped before fv-serve; reports FV_TUNNEL_URL once connected", () => {
    const tok = "eyJhIjoiYWNjdCIsInQiOiJ0MSIsInMiOiJzZWNyZXQifQ==";
    const r = run({ FV_TUNNEL_TOKEN: tok, FV_TUNNEL_URL: "https://fv-pod-a-1-abc123-fv-pods.maximal.llc", FV_DISPATCH_FRONT: "1" });
    expect(r.argv).toBe("tunnel --no-autoupdate run");
    expect(r.tokenEnv).toBe(tok);
    expect(r.reports).toEqual(["tunnel|https://fv-pod-a-1-abc123-fv-pods.maximal.llc|", "after|unset|https://fv-pod-a-1-abc123-fv-pods.maximal.llc"]);
  });
  it("quick (no token): the trycloudflare URL from the log", () => {
    const r = run({});
    expect(r.argv).toBe("tunnel --no-autoupdate --url http://127.0.0.1:8000");
    expect(r.tokenEnv).toBe("");
    expect(r.reports[0]).toBe("tunnel|https://abc-def.trycloudflare.com|");
  });
});

it("scrub removes any tunnel token and CF_TUNNEL_API_TOKEN", () => {
  const env = mkEnv(named);
  expect(scrub(env, `x TUNNEL_TOKEN=eyJhIjoiYWNjdCIsInQiOiJ0MSIsInMiOiJzZWNyZXQifQ== y ${API}`)).toBe("x TUNNEL_TOKEN=[redacted] y [redacted]");
});
