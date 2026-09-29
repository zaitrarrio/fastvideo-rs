// Editable JSON documents (docs/control/README.md "Editors"): one API for
// the smart editors: read with a version, validate, plan (cluster specs),
// save with an optimistic-concurrency check, history from the audit log,
// restore. Kinds:
//   cluster-spec/<cluster id>   the cluster definition
//   policies/default            alert thresholds and auto-actions (with attribution)
//   attribution/default         external pod attribution rules (a slice of policies)
//   env/account | env/cluster:<id> | env/pod:<pod id>   env vars at one level
// Secrets never leave the Worker: an env document shows a secret as
// {value: null, secret: true}; a new value goes in the write-only `set`.
import { DEFAULT_POLICIES, policies, type Policies } from "./alerts";
import { desiredEnv, envCtx, projectSpend } from "./cluster/ops";
import { isReserved } from "./cluster/payloads";
import { normalizeSpec, type ClusterSpec } from "./cluster/spec";
import { allPods, getCluster, livePods, saveSpec, type Cluster } from "./cluster/store";
import type { Env } from "./env";
import { deleteVar, listVars, setVar, type Scope } from "./envvars";
import { validate, type Issue, type SchemaName } from "./schemas";
import { audit, HttpError, now, parseJson, putSetting } from "./util";

export type DocKind = "cluster-spec" | "policies" | "attribution" | "env";
export const DOC_KINDS: DocKind[] = ["cluster-spec", "policies", "attribution", "env"];
export const SCHEMA_OF: Record<DocKind, SchemaName> = { "cluster-spec": "cluster-spec", policies: "policies", attribution: "attribution", env: "env" };

export async function docVersion(env: Env, kind: DocKind, id: string): Promise<number> {
  const r = await env.DB.prepare("SELECT version FROM doc_versions WHERE kind = ? AND id = ?").bind(kind, id).first<{ version: number }>();
  return r?.version ?? 0;
}
/** Bumps a document's version; with `expect`, only if it is still that version (false: stale). */
export async function bumpDoc(env: Env, kind: DocKind, id: string, by: string, expect?: number): Promise<boolean> {
  await env.DB.prepare("INSERT OR IGNORE INTO doc_versions (kind, id, version, updated_at, updated_by) VALUES (?, ?, 0, ?, ?)").bind(kind, id, now(), by).run();
  const r = await env.DB.prepare(
    `UPDATE doc_versions SET version = version + 1, updated_at = ?, updated_by = ? WHERE kind = ? AND id = ?${expect === undefined ? "" : " AND version = ?"}`,
  )
    .bind(now(), by, kind, id, ...(expect === undefined ? [] : [expect]))
    .run();
  return (r.meta?.changes ?? 0) > 0;
}

function envScope(id: string): { scope: Scope; sid: string } {
  if (id === "account") return { scope: "account", sid: "" };
  const m = /^(cluster|pod):([A-Za-z0-9_-]{1,64})$/.exec(id);
  if (!m) throw new HttpError(400, "env id: account | cluster:<id> | pod:<id>");
  return { scope: m[1] as Scope, sid: m[2]! };
}
export type EnvDoc = Record<string, { value: string | null; secret: boolean; set?: string }>;
async function readEnvDoc(env: Env, id: string): Promise<EnvDoc> {
  const { scope, sid } = envScope(id);
  const out: EnvDoc = {};
  for (const r of await listVars(env, scope, sid)) out[r.key] = { value: r.secret ? null : r.value, secret: !!r.secret };
  return out;
}

/** The document as the editor sees it. */
export async function readDoc(env: Env, kind: DocKind, id: string): Promise<unknown> {
  if (kind === "cluster-spec") return (await getCluster(env, id)).spec;
  if (kind === "policies") {
    if (id !== "default") throw new HttpError(404, "policies/default only");
    return await policies(env);
  }
  if (kind === "attribution") {
    if (id !== "default") throw new HttpError(404, "attribution/default only");
    return (await policies(env)).attribution;
  }
  if (kind === "env") return readEnvDoc(env, id);
  throw new HttpError(404, `no document kind ${kind}`);
}
/** Canonical document id (a cluster by name resolves to its id). */
export async function canonicalId(env: Env, kind: DocKind, id: string): Promise<string> {
  if (kind === "cluster-spec") return (await getCluster(env, id)).id;
  if (kind === "env" && id.startsWith("cluster:")) return `cluster:${(await getCluster(env, id.slice(8))).id}`;
  return id;
}

export interface ValidateResult {
  ok: boolean;
  issues: Issue[];
  normalized?: unknown;
}
/** Schema + semantic checks, without saving. */
export async function validateDoc(env: Env, kind: DocKind, id: string, doc: unknown, restoring = false): Promise<ValidateResult> {
  const v = validate(SCHEMA_OF[kind], doc);
  if (!v.ok) return { ok: false, issues: v.issues };
  try {
    if (kind === "cluster-spec") {
      const c = await getCluster(env, id);
      if ((doc as ClusterSpec).name !== c.name) return { ok: false, issues: [{ path: ["name"], message: `the name is fixed (${c.name}); define a new cluster to rename` }] };
      return { ok: true, issues: [], normalized: normalizeSpec(doc) };
    }
    if (kind === "env") {
      const cur = await readEnvDoc(env, id);
      const issues: Issue[] = [];
      for (const [k, v2] of Object.entries(doc as EnvDoc)) {
        if (v2.secret && v2.value !== null && v2.value !== undefined) issues.push({ path: [k, "value"], message: "a secret's value is write-only: put the new value in `set` and leave `value` null" });
        if (v2.secret && v2.set === undefined && !cur[k]?.secret && !restoring) issues.push({ path: [k, "set"], message: "a new secret needs its value in `set`" });
        if (!v2.secret && v2.set !== undefined) issues.push({ path: [k, "set"], message: "`set` is for secrets; use `value`" });
        if (!v2.secret && v2.value === null) issues.push({ path: [k, "value"], message: "value is required" });
        if (isReserved(k)) issues.push({ path: [k], message: `${k} is set by the controller and cannot be overridden` });
      }
      return { ok: issues.length === 0, issues, normalized: doc };
    }
    return { ok: true, issues: [], normalized: v.value };
  } catch (e) {
    const ex = e as HttpError;
    return { ok: false, issues: (ex.extra?.issues as Issue[]) || [{ path: [], message: ex.message }] };
  }
}

/** What an env document looks like in the audit log: secrets masked, `set` dropped. */
export function maskEnvDoc(d: EnvDoc): EnvDoc {
  return Object.fromEntries(Object.entries(d).map(([k, v]) => [k, { value: v.secret ? null : v.value, secret: v.secret }]));
}

async function writeDoc(env: Env, kind: DocKind, id: string, doc: any, by: string): Promise<{ skipped?: string[] }> {
  if (kind === "cluster-spec") {
    const c = await getCluster(env, id);
    c.spec = doc as ClusterSpec;
    await saveSpec(env, c);
    return {};
  }
  if (kind === "policies" || kind === "attribution") {
    const cur = await policies(env);
    const next: Policies = kind === "policies" ? { ...DEFAULT_POLICIES, ...doc } : { ...cur, attribution: doc };
    await putSetting(env, "policies", next, by);
    return {};
  }
  // env: apply the difference key by key.
  const { scope, sid } = envScope(id);
  const cur = await readEnvDoc(env, id);
  const skipped: string[] = [];
  for (const k of Object.keys(cur)) if (!(k in doc)) await deleteVar(env, scope, sid, k);
  for (const [k, v] of Object.entries(doc as EnvDoc)) {
    if (v.secret) {
      if (v.set !== undefined) await setVar(env, scope, sid, k, v.set, true, by);
      else if (!cur[k]?.secret) skipped.push(k); // a restored snapshot cannot bring back a secret value
    } else if (!cur[k] || cur[k].secret || cur[k].value !== v.value) await setVar(env, scope, sid, k, String(v.value), false, by);
  }
  return { skipped };
}

/** Saves a document: validation, the version check (when `version` is given), the write, the audit entry. */
export async function saveDoc(
  env: Env,
  kind: DocKind,
  rawId: string,
  doc: unknown,
  o: { version?: number; actor: string; ip?: string; action?: string; detail?: string },
): Promise<{ version: number; doc: unknown; skipped?: string[] }> {
  const id = await canonicalId(env, kind, rawId);
  const v = await validateDoc(env, kind, id, doc, o.action === "doc.restore");
  if (!v.ok) throw new HttpError(400, v.issues.map((i) => `${i.path.join(".") || kind}: ${i.message}`).join("; "), { issues: v.issues });
  const before = await readDoc(env, kind, id);
  if (!(await bumpDoc(env, kind, id, o.actor, o.version))) {
    throw new HttpError(409, "the document changed since you loaded it: reload, re-apply your edit and save again", { current_version: await docVersion(env, kind, id) });
  }
  const r = await writeDoc(env, kind, id, v.normalized, o.actor);
  const after = await readDoc(env, kind, id);
  const mask = (d: unknown) => (kind === "env" ? maskEnvDoc(d as EnvDoc) : d);
  await audit(env, { actor: o.actor, action: o.action || "doc.save", target: `${kind}:${id}`, before: mask(before), after: mask(after), ip: o.ip, detail: r.skipped?.length ? `secrets not restored: ${r.skipped.join(", ")}` : o.detail });
  return { version: await docVersion(env, kind, id), doc: after, ...(r.skipped?.length ? { skipped: r.skipped } : {}) };
}

/** The document's saves from the audit log, newest first. */
export async function docHistory(env: Env, kind: DocKind, rawId: string, limit = 50) {
  const id = await canonicalId(env, kind, rawId);
  const r = await env.DB.prepare("SELECT id, at, actor, action, before, after, detail FROM audit WHERE target = ? AND action IN ('doc.save', 'doc.restore') AND ok = 1 ORDER BY id DESC LIMIT ?")
    .bind(`${kind}:${id}`, limit)
    .all<any>();
  return (r.results || []).map((x) => ({ audit_id: x.id, at: x.at, actor: x.actor, action: x.action, detail: x.detail, before: parseJson(x.before, null), after: parseJson(x.after, null) }));
}
export async function restoreDoc(env: Env, kind: DocKind, rawId: string, auditId: number, which: "after" | "before", o: { version?: number; actor: string; ip?: string }) {
  const id = await canonicalId(env, kind, rawId);
  const row = await env.DB.prepare("SELECT target, before, after FROM audit WHERE id = ?").bind(auditId).first<{ target: string; before: string; after: string }>();
  if (!row || row.target !== `${kind}:${id}`) throw new HttpError(404, "no such history entry for this document");
  const snap = parseJson<unknown>(which === "before" ? row.before : row.after, null);
  if (snap === null) throw new HttpError(409, "that entry has no snapshot");
  return saveDoc(env, kind, id, snap, { ...o, action: "doc.restore", detail: `restored audit #${auditId} (${which})` });
}

// ---------------- the plan of a cluster spec change
export interface PlanAction {
  action: "create" | "delete" | "restart" | "roll" | "none";
  target: string;
  detail: string;
}
export async function planSpec(env: Env, rawId: string, doc: unknown) {
  const c = await getCluster(env, rawId);
  const v = await validateDoc(env, "cluster-spec", c.id, doc);
  if (!v.ok) return { ok: false, issues: v.issues };
  const next = v.normalized as ClusterSpec;
  const running = allPods(c.state).length > 0;
  const actions: PlanAction[] = [];
  const warnings: string[] = [];
  const cur = c.spec;
  const imgChanged = JSON.stringify(cur.image) !== JSON.stringify(next.image);
  if (running) {
    for (const p of next.pools) {
      const have = c.state.workers[p.id]?.length || 0;
      const was = cur.pools.find((x) => x.id === p.id);
      if (!was) actions.push({ action: "create", target: p.id, detail: `new pool: ${p.count} worker(s) on Scale; the gateway restarts to learn it` });
      else if (p.count > have) actions.push({ action: "create", target: p.id, detail: `${p.count - have} more worker(s) (Scale ${p.id} to ${p.count})` });
      else if (p.count < have) actions.push({ action: "delete", target: p.id, detail: `${have - p.count} worker(s) drained and deleted (Scale ${p.id} to ${p.count})` });
      if (was && (was.variant !== p.variant || was.image !== p.image)) actions.push({ action: "roll", target: p.id, detail: `image of ${p.id} changes: Roll` });
    }
    for (const p of cur.pools) if (!next.pools.some((x) => x.id === p.id)) actions.push({ action: "delete", target: p.id, detail: `pool removed: its ${c.state.workers[p.id]?.length || 0} worker(s) are stopped (Scale to 0 first)` });
    if (imgChanged) actions.push({ action: "roll", target: "all", detail: `image source ${JSON.stringify(cur.image)} → ${JSON.stringify(next.image)}: Roll every pool (and the gateway)` });
    // Env-level effect: pods whose env would differ under the new spec need a restart.
    const shadow: Cluster = { ...c, spec: next };
    const ctx = await envCtx(env, shadow);
    for (const r of (await livePods(env, c.id)).filter((x: any) => x.slot !== "retired")) {
      if (r.role === "worker" && !next.pools.some((p) => p.id === r.pool)) continue;
      const d = await desiredEnv(env, shadow, ctx, r.role, { pod: r.pod_id, pool: r.pool, image: r.image });
      if (d.hash !== r.env_hash) actions.push({ action: "restart", target: r.pod_id, detail: `${r.role}${r.pool ? ` ${r.pool}` : ""}: its env changes (Env: apply with a rolling restart)` });
    }
    if (next.cap_s !== cur.cap_s) warnings.push("cap_s applies at the next start; move a running cluster's deadline with Extend");
  } else {
    const n = next.pools.reduce((s, p) => s + p.count, 0) + (next.gateway.enabled ? 1 : 0);
    actions.push({ action: "none", target: c.name, detail: `the cluster is stopped: nothing changes now; Start would create ${n} pod(s)` });
  }
  const hours = running && c.deadline ? Math.max(0.1, (c.deadline - now()) / 3_600_000) : next.cap_s / 3600;
  const proj = await projectSpend(env, next, { hours, runningDph: running ? allPods(c.state).reduce((s, x) => s + (x.rec.dph || 0), 0) : 0 }).catch((e) => ({ error: (e as Error).message }) as any);
  const dphNow = allPods(c.state).reduce((s, x) => s + (x.rec.dph || 0), 0);
  return { ok: true, running, actions, warnings, dph_now: dphNow, dph_after: proj.cluster_dph ?? null, projection: proj };
}
