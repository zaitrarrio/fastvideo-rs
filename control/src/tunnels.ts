// Named Cloudflare Tunnels for GMI / Brev pods (docs/serve/deploy-gmi-brev.md §6.4).
//
// Neither provider has an HTTPS proxy, so a pod reaches the Internet through a
// Cloudflare tunnel. Without the settings below it opens a quick tunnel
// (random https://<x>.trycloudflare.com, reported to fv-control). With them,
// fv-control makes one NAMED tunnel per pod on the owner's zone before the
// provider's create call:
//   POST   /accounts/{acct}/cfd_tunnel                  {name: fv-…, config_src: cloudflare}
//   PUT    /accounts/{acct}/cfd_tunnel/{id}/configurations  ingress: <host> -> http://127.0.0.1:8000, then 404
//   POST   /zones/{zone}/dns_records                    CNAME <host> -> <id>.cfargotunnel.com, proxied, comment "fv-control <pod>"
//   GET    /accounts/{acct}/cfd_tunnel/{id}/token       into the pod's env (FV_TUNNEL_TOKEN; never stored, logged or returned)
// The pod runs `cloudflared tunnel run` with it (providers.ts PROVIDER_BOOT) and
// still reports `tunnel` once connected; only its own URL is accepted.
// fv-control knows the URL up front (https://<host>).
//
// Delete (the pod deleted, a failed create, a parked Brev instance deleted):
// the DNS record, then the tunnel's connections, then the tunnel. A parked
// Brev instance keeps both (its restart runs the same tunnel).
//
// Only ours: a row of pod_tunnels (migration 0009) is written before the
// Cloudflare calls; fv-control deletes only a tunnel or a record it recorded
// there, after checking the tunnel's name (fv-…, the recorded one) and the
// record's comment (fv-control …) and name. The cron (tunnelSweep) deletes
// recorded tunnels whose pod is gone (leaks after a crash) and raises an
// alert for an fv- tunnel / fv-control record it did not record, untouched.
//
// Settings: FV_TUNNEL_HOST_TEMPLATE ({pod}-fv-pods.example.com), FV_TUNNEL_ZONE
// (example.com), CF_ACCOUNT_ID and the secret CF_TUNNEL_API_TOKEN (Tunnel edit
// + DNS edit on the zone). Universal SSL covers one level below the zone
// (*.example.com): a host two levels down (x.fv-pods.example.com) fails the
// edge's TLS handshake unless the zone has an Advanced Certificate Manager
// certificate for *.fv-pods.example.com, so such a template is warned about.
import type { AlertIn } from "./alerts";
import { randomToken } from "./crypto";
import { defaults, type Env } from "./env";
import { fetchWithTimeout, getSetting, HttpError, now, parseJson, putSetting } from "./util";

// ---------------------------------------------------------------- settings
export interface TunnelSettings {
  /** named: fv-control makes a tunnel per pod; quick: the pod opens a quick tunnel (the fallback). */
  mode: "named" | "quick";
  zone: string | null;
  host_template: string | null;
  /** Settings a named tunnel still needs (empty when named). */
  missing: string[];
  /** Labels between {pod}'s host and the zone (1: <pod>-fv-pods.zone; 2+: needs an advanced certificate). */
  levels: number | null;
  /** A template fv-control cannot use: GMI / Brev launches are refused until it is fixed. */
  error: string | null;
  /** Loud but not fatal: a host Universal SSL does not cover, or a half-configured setup. */
  warning: string | null;
}
const NAMED_SETTINGS = ["FV_TUNNEL_HOST_TEMPLATE", "FV_TUNNEL_ZONE", "CF_ACCOUNT_ID", "CF_TUNNEL_API_TOKEN"] as const;
const LABEL_RE = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;
/** Characters a host label's {pod} part gets: <slug>-<6 hex>. */
const SUFFIX_LEN = 7;

export function tunnelSettings(env: Env): TunnelSettings {
  const tpl = (env.FV_TUNNEL_HOST_TEMPLATE || "").trim().toLowerCase();
  const zone = (env.FV_TUNNEL_ZONE || "").trim().toLowerCase().replace(/\.$/, "");
  const missing = NAMED_SETTINGS.filter((k) => !env[k]);
  const base: TunnelSettings = { mode: "quick", zone: zone || null, host_template: tpl || null, missing: [...missing], levels: null, error: null, warning: null };
  if (missing.length) {
    // Nothing of it set: the quick-tunnel fallback, as before. Some of it: say what is missing (not silently quick).
    if (env.FV_TUNNEL_HOST_TEMPLATE || env.FV_TUNNEL_ZONE || env.CF_TUNNEL_API_TOKEN)
      base.warning = `named pod tunnels need ${missing.join(", ")}: GMI / Brev pods use quick tunnels (*.trycloudflare.com) until they are set`;
    return base;
  }
  const s: TunnelSettings = { ...base, mode: "named" };
  const err = templateError(tpl, zone);
  if (err) return { ...s, error: `FV_TUNNEL_HOST_TEMPLATE ${JSON.stringify(tpl)}: ${err}` };
  s.levels = tpl.slice(0, -(zone.length + 1)).split(".").length;
  if (s.levels > 1)
    s.warning = `FV_TUNNEL_HOST_TEMPLATE ${tpl} puts pod hosts ${s.levels} levels below ${zone}: Cloudflare's Universal SSL covers only *.${zone}, so the edge's TLS handshake fails for them unless the zone has an Advanced Certificate Manager certificate for *.${tpl.split(".").slice(1).join(".")}. Use one level, e.g. {pod}-${tpl.split(".").slice(1, -zone.split(".").length).join("-") || "fv-pods"}.${zone}`;
  return s;
}
/** Why a host template cannot be used (null: it can). */
export function templateError(tpl: string, zone: string): string | null {
  if (!zone || !/^[a-z0-9-]+(\.[a-z0-9-]+)+$/.test(zone)) return `FV_TUNNEL_ZONE ${JSON.stringify(zone)} is not a zone name`;
  if (tpl.split("{pod}").length !== 2) return "needs {pod} exactly once";
  if (!tpl.endsWith(`.${zone}`)) return `must end with .${zone} (FV_TUNNEL_ZONE)`;
  const labels = tpl.slice(0, -(zone.length + 1)).split(".");
  const podLabel = labels.find((l) => l.includes("{pod}"))!;
  for (const l of labels) if (!LABEL_RE.test(l.replace("{pod}", "x"))) return `${JSON.stringify(l)} is not a DNS label (a-z, 0-9, -)`;
  if (63 - (podLabel.length - 5) - SUFFIX_LEN < 8) return `the label ${podLabel} leaves too little room for the pod's name (63 characters per label)`;
  if (tpl.length - 5 + 8 + SUFFIX_LEN > 253) return "too long for a host name";
  return null;
}
/** The tunnel's name and host for a pod: {pod} = a DNS-safe slug of its name + a short random suffix (≤ 63 per label). */
export function tunnelHost(tpl: string, podName: string, suffix = randomToken("", 3)): { name: string; host: string } {
  const podLabel = tpl.split(".").find((l) => l.includes("{pod}"))!;
  const room = Math.min(63 - (podLabel.length - 5), 253 - (tpl.length - 5)) - SUFFIX_LEN;
  const slug =
    podName
      .toLowerCase()
      .replace(/[^a-z0-9-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-+/, "")
      .slice(0, room)
      .replace(/-+$/, "") || "pod";
  const label = `${slug}-${suffix}`;
  return { name: label.startsWith("fv-") ? label : `fv-${label}`, host: tpl.replace("{pod}", label) };
}
/** What a pod must report as its URL (tests: FV_TUNNEL_URL_TEMPLATE routes it to the simulated edge). */
export const tunnelUrl = (env: Env, host: string) => (env.FV_TUNNEL_URL_TEMPLATE || "https://{host}").replace("{host}", host);

/** /api/providers and /api/tunnels: the mode and what is wrong with it (no secret). */
export function tunnelView(env: Env) {
  const s = tunnelSettings(env);
  return {
    ...s,
    note:
      s.mode === "named"
        ? s.error
          ? `named pod tunnels are configured but unusable: ${s.error}`
          : `named pod tunnels: one Cloudflare Tunnel + proxied CNAME per GMI / Brev pod at ${s.host_template} (made and deleted by fv-control)`
        : "quick pod tunnels: each GMI / Brev pod opens https://<random>.trycloudflare.com (no uptime guarantee, 200 concurrent requests, no SSE)",
  };
}

// ---------------------------------------------------------------- the Cloudflare API
export class CfError extends Error {
  constructor(
    public status: number,
    message: string,
    public codes: number[] = [],
  ) {
    super(message);
  }
}
async function cf<T>(env: Env, method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetchWithTimeout(`${defaults.cfApi(env)}${path}`, {
    method,
    headers: { authorization: `Bearer ${env.CF_TUNNEL_API_TOKEN}`, ...(body !== undefined ? { "content-type": "application/json" } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
    timeoutMs: 20000,
  });
  const text = await r.text();
  const j = parseJson<{ success?: boolean; result?: T; errors?: { code: number; message: string }[] } | null>(text, null);
  if (!r.ok || !j || j.success === false) {
    const errs = (j?.errors || []).map((e) => `${e.code} ${e.message}`).join("; ") || text.slice(0, 200);
    // The path holds no secret (account, zone, tunnel ids); the reply never carries the API token.
    throw new CfError(r.status, `cloudflare ${method} ${path.split("?")[0]}: ${r.status} ${errs}`.slice(0, 400), (j?.errors || []).map((e) => e.code));
  }
  return j.result as T;
}
const gone = (e: unknown) => e instanceof CfError && e.status === 404;
interface CfTunnel {
  id: string;
  name: string;
  deleted_at?: string | null;
  connections?: unknown[];
}
interface CfRecord {
  id: string;
  name: string;
  type: string;
  content: string;
  comment?: string | null;
}

const zoneCache = new Map<string, string>();
/** The zone's id by its name (cached: memory, then D1). */
export async function zoneId(env: Env, zone: string): Promise<string> {
  const hit = zoneCache.get(zone);
  if (hit) return hit;
  const saved = await getSetting<{ id: string | null }>(env, `tunnel_zone:${zone}`, { id: null });
  if (saved.id) return zoneCache.set(zone, saved.id), saved.id;
  const zs = await cf<{ id: string; name: string }[]>(env, "GET", `/zones?name=${encodeURIComponent(zone)}`);
  const z = (zs || []).find((x) => x.name === zone);
  if (!z) throw new Error(`zone ${zone} not found with CF_TUNNEL_API_TOKEN (FV_TUNNEL_ZONE)`);
  zoneCache.set(zone, z.id);
  await putSetting(env, `tunnel_zone:${zone}`, { id: z.id }, "tunnels");
  return z.id;
}
/** Tests: forget the cached zone ids. */
export const resetZoneCache = () => zoneCache.clear();

// ---------------------------------------------------------------- the record
export type TunnelState = "creating" | "live" | "delete_failed" | "deleted";
export interface PodTunnel {
  name: string;
  pod_id: string;
  cluster_id: string | null;
  host: string;
  url: string;
  zone_id: string | null;
  tunnel_id: string | null;
  dns_record_id: string | null;
  state: TunnelState;
  note: string | null;
  created_at: number;
  updated_at: number;
  deleted_at: number | null;
}
const OPEN = "('creating', 'live', 'delete_failed')";
async function setRow(env: Env, name: string, f: Partial<Omit<PodTunnel, "name" | "created_at">>) {
  const keys = Object.keys(f) as (keyof typeof f)[];
  await env.DB.prepare(`UPDATE pod_tunnels SET ${[...keys.map((k) => `${k} = ?`), "updated_at = ?"].join(", ")} WHERE name = ?`)
    .bind(...keys.map((k) => f[k] ?? null), now(), name)
    .run();
}
/** A pod's live tunnel (null: none, e.g. a quick-tunnel pod). */
export async function podTunnel(env: Env, podId: string): Promise<PodTunnel | null> {
  return env.DB.prepare("SELECT * FROM pod_tunnels WHERE pod_id = ? AND state = 'live' ORDER BY created_at DESC LIMIT 1").bind(podId).first<PodTunnel>();
}
/** The URL a pod with a named tunnel must report (null: the pod has none; quick-tunnel rules apply). */
export async function expectedTunnelUrl(env: Env, podId: string): Promise<string | null> {
  return (await podTunnel(env, podId))?.url ?? null;
}
export async function tunnelRows(env: Env, all = false): Promise<PodTunnel[]> {
  return (await env.DB.prepare(`SELECT * FROM pod_tunnels ${all ? "" : `WHERE state IN ${OPEN}`} ORDER BY created_at DESC LIMIT 500`).all<PodTunnel>()).results || [];
}

// ---------------------------------------------------------------- create / reuse
export interface PodTunnelGrant {
  name: string;
  host: string;
  url: string;
  tunnel_id: string;
  /** For the pod's env only: never stored, logged or returned. */
  token: string;
  /** A parked Brev instance's tunnel, restarted with it. */
  reused: boolean;
}
/**
 * The named tunnel of a pod about to be created (or restarted): its existing live one (a parked Brev instance's),
 * else a new tunnel, its ingress and its DNS record. Throws (after removing what it made) when Cloudflare refuses.
 */
export async function ensureTunnel(env: Env, x: { podId: string; podName: string; clusterId: string | null }): Promise<PodTunnelGrant> {
  const s = tunnelSettings(env);
  if (s.mode !== "named") throw new Error(`named tunnels are off (${s.missing.join(", ")} unset)`);
  if (s.error) throw new Error(s.error);
  const acct = env.CF_ACCOUNT_ID!;
  const have = await podTunnel(env, x.podId);
  if (have?.tunnel_id) {
    try {
      const t = await cf<CfTunnel>(env, "GET", `/accounts/${acct}/cfd_tunnel/${have.tunnel_id}`);
      if (t.deleted_at || t.name !== have.name) throw new Error(`tunnel ${have.name} is gone`);
      const token = await cf<string>(env, "GET", `/accounts/${acct}/cfd_tunnel/${have.tunnel_id}/token`);
      await setRow(env, have.name, { cluster_id: x.clusterId });
      return { name: have.name, host: have.host, url: have.url, tunnel_id: have.tunnel_id, token, reused: true };
    } catch (e) {
      // Deleted outside fv-control (or unreadable): clean up what is left of it and make a new one.
      await destroyTunnel(env, have, `replaced: ${(e as Error).message}`).catch(() => {});
    }
  }
  const { name, host } = tunnelHost(s.host_template!, x.podName);
  const url = tunnelUrl(env, host);
  const t0 = now();
  await env.DB.prepare("INSERT INTO pod_tunnels (name, pod_id, cluster_id, host, url, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'creating', ?, ?)").bind(name, x.podId, x.clusterId, host, url, t0, t0).run();
  try {
    const zone = await zoneId(env, s.zone!);
    await setRow(env, name, { zone_id: zone });
    const t = await cf<CfTunnel>(env, "POST", `/accounts/${acct}/cfd_tunnel`, { name, config_src: "cloudflare" });
    await setRow(env, name, { tunnel_id: t.id });
    // The ingress: the pod's fv-serve, then 404. No originRequest: cloudflared's defaults keep websockets, SSE
    // (streamed, unbuffered) and the WebRTC signalling (plain HTTP / websocket) working.
    await cf(env, "PUT", `/accounts/${acct}/cfd_tunnel/${t.id}/configurations`, { config: { ingress: [{ hostname: host, service: "http://127.0.0.1:8000" }, { service: "http_status:404" }] } });
    const rec = await cf<CfRecord>(env, "POST", `/zones/${zone}/dns_records`, { type: "CNAME", name: host, content: `${t.id}.cfargotunnel.com`, proxied: true, ttl: 1, comment: `fv-control ${x.podId}` });
    await setRow(env, name, { dns_record_id: rec.id });
    const token = await cf<string>(env, "GET", `/accounts/${acct}/cfd_tunnel/${t.id}/token`);
    await setRow(env, name, { state: "live" });
    return { name, host, url, tunnel_id: t.id, token, reused: false };
  } catch (e) {
    const row = await env.DB.prepare("SELECT * FROM pod_tunnels WHERE name = ?").bind(name).first<PodTunnel>();
    if (row) await destroyTunnel(env, row, `create failed: ${(e as Error).message}`).catch(() => {});
    throw new Error(`named tunnel ${name}: ${(e as Error).message}`);
  }
}

// ---------------------------------------------------------------- delete
/**
 * Deletes a recorded tunnel: its DNS record (if its comment is fv-control's and its name the recorded host), the
 * tunnel's connections, the tunnel (if its name is the recorded fv- name). Marks the row deleted, or delete_failed
 * with why (the cron retries). Never touches anything the row does not name.
 */
export async function destroyTunnel(env: Env, row: PodTunnel, why: string): Promise<string> {
  const acct = env.CF_ACCOUNT_ID;
  try {
    if (!env.CF_TUNNEL_API_TOKEN || !acct) throw new Error("CF_TUNNEL_API_TOKEN / CF_ACCOUNT_ID unset: cannot delete it");
    if (row.dns_record_id && row.zone_id) {
      const rec = await cf<CfRecord>(env, "GET", `/zones/${row.zone_id}/dns_records/${row.dns_record_id}`).catch((e) => (gone(e) ? null : Promise.reject(e)));
      if (rec) {
        if (!(rec.comment || "").startsWith("fv-control") || rec.name !== row.host) throw new Error(`DNS record ${row.dns_record_id} is not ours any more (${rec.name}, comment ${JSON.stringify(rec.comment || "")}): not deleted`);
        await cf(env, "DELETE", `/zones/${row.zone_id}/dns_records/${row.dns_record_id}`).catch((e) => (gone(e) ? null : Promise.reject(e)));
      }
    }
    if (row.tunnel_id) {
      const t = await cf<CfTunnel>(env, "GET", `/accounts/${acct}/cfd_tunnel/${row.tunnel_id}`).catch((e) => (gone(e) ? null : Promise.reject(e)));
      if (t && !t.deleted_at) {
        if (t.name !== row.name || !t.name.startsWith("fv-")) throw new Error(`tunnel ${row.tunnel_id} is named ${t.name}, not ${row.name}: not deleted`);
        // A deleted pod's connections linger until Cloudflare times them out; a tunnel with connections cannot be deleted.
        await cf(env, "DELETE", `/accounts/${acct}/cfd_tunnel/${row.tunnel_id}/connections`).catch((e) => (gone(e) ? null : Promise.reject(e)));
        await cf(env, "DELETE", `/accounts/${acct}/cfd_tunnel/${row.tunnel_id}`).catch((e) => (gone(e) ? null : Promise.reject(e)));
      }
    }
    const t = now();
    await env.DB.prepare("UPDATE pod_tunnels SET state = 'deleted', deleted_at = ?, updated_at = ?, note = ? WHERE name = ?").bind(t, t, why.slice(0, 300), row.name).run();
    return `tunnel ${row.name} (${row.host}) deleted: ${why}`;
  } catch (e) {
    await setRow(env, row.name, { state: "delete_failed", note: `${why}; delete failed: ${(e as Error).message}`.slice(0, 400) });
    throw e;
  }
}
/** Deletes every open tunnel of a pod (it was deleted). Returns what happened, one line each; never throws. */
export async function releaseTunnels(env: Env, podId: string, why: string): Promise<string[]> {
  const rows = (await env.DB.prepare(`SELECT * FROM pod_tunnels WHERE pod_id = ? AND state IN ${OPEN}`).bind(podId).all<PodTunnel>()).results || [];
  const out: string[] = [];
  for (const r of rows) out.push(await destroyTunnel(env, r, why).catch((e) => `WARNING: tunnel ${r.name}: ${(e as Error).message.slice(0, 200)} (the cron retries)`));
  return out;
}

// ---------------------------------------------------------------- the cron: leaks and strays
/** A pod that still needs its tunnel: a live cluster pod, or a Brev instance that is live, restarting, parked or held. */
async function podAlive(env: Env, podId: string): Promise<boolean> {
  const c = await env.DB.prepare("SELECT 1 AS x FROM cluster_pods WHERE pod_id = ? AND deleted_at IS NULL").bind(podId).first();
  if (c) return true;
  const b = await env.DB.prepare("SELECT 1 AS x FROM brev_instances WHERE pod_id = ? AND state IN ('live', 'restarting', 'parked', 'held')").bind(podId).first();
  return !!b;
}
export interface TunnelSweep {
  alerts: AlertIn[];
  kinds: string[];
  actions: string[];
  scanned: boolean;
}
/** A create in flight may not have its pod recorded yet. */
export const TUNNEL_GRACE_MS = 10 * 60_000;
/** How often the cron lists the account's tunnels and the zone's records for strays. */
export const TUNNEL_SCAN_MS = 10 * 60_000;
/**
 * The cron's pass: recorded tunnels whose pod is gone (a crash, a pod deleted outside fv-control) are deleted, failed
 * deletes retried; every TUNNEL_SCAN_MS (or `force`) the account's fv- tunnels and the zone's fv-control records are
 * listed, and one fv-control did not record raises an alert and is left alone.
 */
export async function tunnelSweep(env: Env, t = now(), force = false): Promise<TunnelSweep> {
  const res: TunnelSweep = { alerts: [], kinds: ["tunnel_cleanup"], actions: [], scanned: false };
  const s = tunnelSettings(env);
  const rows = await tunnelRows(env);
  if (s.mode !== "named" && !rows.length) return { ...res, kinds: [] };
  if (!env.CF_TUNNEL_API_TOKEN || !env.CF_ACCOUNT_ID) {
    if (rows.length) res.alerts.push({ key: "tunnel_cleanup:config", kind: "tunnel_cleanup", severity: "warn", message: `${rows.length} recorded pod tunnel(s) but CF_TUNNEL_API_TOKEN / CF_ACCOUNT_ID unset: they cannot be deleted` });
    return res;
  }
  for (const r of rows) {
    if (r.state !== "delete_failed" && t - r.created_at < TUNNEL_GRACE_MS) continue;
    if (r.state !== "delete_failed" && (await podAlive(env, r.pod_id))) continue;
    try {
      res.actions.push(await destroyTunnel(env, r, r.state === "delete_failed" ? "retry" : `its pod ${r.pod_id} is gone (leak cleanup)`));
    } catch (e) {
      res.alerts.push({ key: `tunnel_cleanup:${r.name}`, kind: "tunnel_cleanup", severity: "warn", target: r.pod_id, message: `pod tunnel ${r.name} (${r.host}): delete failed: ${(e as Error).message.slice(0, 200)}` });
    }
  }
  const last = await getSetting<{ at: number }>(env, "tunnel_scan", { at: 0 });
  if (!force && t - last.at < TUNNEL_SCAN_MS) return res;
  await putSetting(env, "tunnel_scan", { at: t }, "cron");
  res.scanned = true;
  res.kinds.push("tunnel_orphan");
  const all = await tunnelRows(env, true);
  const byId = new Map(all.filter((r) => r.tunnel_id).map((r) => [r.tunnel_id!, r]));
  const byName = new Map(all.map((r) => [r.name, r]));
  try {
    const listed: CfTunnel[] = [];
    for (let page = 1; page <= 10; page++) {
      const got = await cf<CfTunnel[]>(env, "GET", `/accounts/${env.CF_ACCOUNT_ID}/cfd_tunnel?is_deleted=false&per_page=100&page=${page}`);
      listed.push(...(got || []));
      if (!got || got.length < 100) break;
    }
    for (const tun of listed) {
      if (!tun.name.startsWith("fv-") || tun.deleted_at) continue;
      const row = byId.get(tun.id) ?? byName.get(tun.name);
      if (row && !row.tunnel_id) {
        // Made, then fv-control crashed before recording its id: ours by its recorded name.
        await setRow(env, row.name, { tunnel_id: tun.id, ...(row.state === "deleted" ? { state: "delete_failed" as const } : {}) });
        res.actions.push(`tunnel ${tun.name}: id ${tun.id} recorded (made before a crash)`);
      } else if (row?.state === "deleted") {
        // Ours, deleted before, still listed: delete again.
        res.actions.push(await destroyTunnel(env, row, "still listed after its delete").catch((e) => `WARNING: tunnel ${row.name}: ${(e as Error).message.slice(0, 200)}`));
      } else if (!row)
        res.alerts.push({ key: `tunnel_orphan:${tun.id}`, kind: "tunnel_orphan", severity: "warn", target: tun.name, message: `Cloudflare tunnel ${tun.name} (${tun.id}) has an fv- name but fv-control did not record it: not touched (delete it in the Cloudflare dashboard if it is ours)` });
    }
    if (s.zone) {
      const zone = await zoneId(env, s.zone);
      const recs = await cf<CfRecord[]>(env, "GET", `/zones/${zone}/dns_records?comment.startswith=fv-control&per_page=500`);
      const byRec = new Map(all.filter((r) => r.dns_record_id).map((r) => [r.dns_record_id!, r]));
      const byHost = new Map(all.map((r) => [r.host, r]));
      for (const rec of recs || []) {
        if (!(rec.comment || "").startsWith("fv-control")) continue;
        const row = byRec.get(rec.id) ?? byHost.get(rec.name);
        if (row && !row.dns_record_id) {
          await setRow(env, row.name, { dns_record_id: rec.id, zone_id: zone, ...(row.state === "deleted" ? { state: "delete_failed" as const } : {}) });
          res.actions.push(`DNS ${rec.name}: record ${rec.id} recorded (made before a crash)`);
        } else if (!row)
          res.alerts.push({ key: `tunnel_orphan:dns:${rec.id}`, kind: "tunnel_orphan", severity: "warn", target: rec.name, message: `DNS record ${rec.name} (${rec.type}, comment ${JSON.stringify(rec.comment)}) says fv-control but fv-control did not record it: not touched` });
      }
    }
  } catch (e) {
    res.kinds = res.kinds.filter((k) => k !== "tunnel_orphan");
    res.alerts.push({ key: "tunnel_cleanup:api", kind: "tunnel_cleanup", severity: "warn", message: `pod tunnels: the Cloudflare listing failed: ${(e as Error).message.slice(0, 200)}` });
  }
  return res;
}

/** The owner's view: the settings and the recorded tunnels (no token). */
export async function tunnelsView(env: Env, all = false) {
  return { settings: tunnelView(env), tunnels: await tunnelRows(env, all) };
}
/** Refuses a tunnel action while named tunnels are not configured. */
export function requireNamed(env: Env) {
  const s = tunnelSettings(env);
  if (s.mode !== "named") throw new HttpError(409, `named pod tunnels are off: ${s.missing.join(", ")} unset`);
}
