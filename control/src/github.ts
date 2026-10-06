// GitHub with GITHUB_PAT (a fine-grained token: actions:write, contents
// and metadata read on the repository): dispatch release.yml (promote /
// rollback), CI status of main. The token is never returned or logged.
import { defaults, type Env } from "./env";
import { fetchWithTimeout, HttpError, scrub } from "./util";

async function gh(env: Env, path: string, init: RequestInit = {}, token = env.GITHUB_PAT): Promise<Response> {
  if (!token) throw new HttpError(503, "GITHUB_PAT is not set on the controller");
  return fetchWithTimeout(`${defaults.githubApi(env)}${path}`, {
    ...init,
    headers: {
      authorization: `Bearer ${token}`,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
      "user-agent": "fv-control",
      ...(init.body ? { "content-type": "application/json" } : {}),
      ...((init.headers as Record<string, string>) || {}),
    },
    timeoutMs: 20000,
  });
}
async function ghJson(env: Env, path: string): Promise<any> {
  const r = await gh(env, path);
  if (!r.ok) throw new HttpError(502, `github ${path.split("?")[0]}: ${r.status} ${scrub(env, (await r.text()).slice(0, 200))}`);
  return r.json();
}

export interface ReleaseDispatch {
  action: "promote" | "rollback";
  target?: string;
  channel?: string;
  to?: string;
  notes?: string;
  templates?: boolean;
  allow_partial?: boolean;
  dry_run?: boolean;
}
export function validateDispatch(d: ReleaseDispatch): Record<string, string> {
  if (d.action !== "promote" && d.action !== "rollback") throw new HttpError(400, "action: promote | rollback");
  const channel = d.channel || "stable";
  if (!/^[a-z][a-z0-9-]{0,30}$/.test(channel)) throw new HttpError(400, "channel: a lower-case word");
  const target = (d.target || "").trim();
  if (d.action === "promote" && !/^([0-9a-f]{7,40}|sha256:[0-9a-f]{64}|[a-z0-9][a-z0-9._-]{0,60})$/.test(target)) throw new HttpError(400, "target: a git sha, a digest or a tag");
  if (d.to && !/^\d+$/.test(String(d.to))) throw new HttpError(400, "to: a release id");
  return {
    action: d.action,
    target,
    channel,
    to: d.to ? String(d.to) : "",
    notes: String(d.notes || "").slice(0, 200),
    templates: d.templates === false ? "false" : "true",
    allow_partial: d.allow_partial ? "true" : "false",
    dry_run: d.dry_run ? "true" : "false",
  };
}
export async function dispatchRelease(env: Env, d: ReleaseDispatch, ref = "main"): Promise<{ dispatched: true; inputs: Record<string, string>; runs_url: string }> {
  const inputs = validateDispatch(d);
  const repo = defaults.githubRepo(env);
  const wf = defaults.releaseWorkflow(env);
  const r = await gh(env, `/repos/${repo}/actions/workflows/${wf}/dispatches`, { method: "POST", body: JSON.stringify({ ref, inputs }) });
  if (r.status !== 204 && r.status !== 200) throw new HttpError(502, `github dispatch: ${r.status} ${scrub(env, (await r.text()).slice(0, 300))}`);
  return { dispatched: true, inputs, runs_url: `https://github.com/${repo}/actions/workflows/${wf}` };
}

/** Recent workflow runs on main (the CI view), and release.yml's runs. */
export async function ciStatus(env: Env): Promise<any> {
  const repo = defaults.githubRepo(env);
  const [main, rel] = await Promise.all([
    ghJson(env, `/repos/${repo}/actions/runs?branch=main&per_page=30`),
    ghJson(env, `/repos/${repo}/actions/workflows/${defaults.releaseWorkflow(env)}/runs?per_page=10`).catch(() => ({ workflow_runs: [] })),
  ]);
  const pick = (w: any) => ({ id: w.id, name: w.name, event: w.event, status: w.status, conclusion: w.conclusion, head_sha: String(w.head_sha || "").slice(0, 7), created_at: w.created_at, html_url: w.html_url, title: w.display_title });
  const runs = (main.workflow_runs || []).map(pick);
  // The newest run of each workflow on main: is main green?
  const latest = new Map<string, any>();
  for (const w of runs) if (!latest.has(w.name)) latest.set(w.name, w);
  return { main: [...latest.values()], recent: runs.slice(0, 15), releases: (rel.workflow_runs || []).map(pick) };
}

// ---- self-hosted runners of the build pods (docs/dev/build-pods-fv-control.md §5, §6)
// GITHUB_RUNNER_PAT: fine-grained, Administration read/write + Actions read
// on the repository; falls back to GITHUB_PAT when that one has them.
const runnerPat = (env: Env) => env.GITHUB_RUNNER_PAT || env.GITHUB_PAT;
export const runnerPatSet = (env: Env) => !!runnerPat(env);

async function ghRunner(env: Env, path: string, init: RequestInit = {}): Promise<any> {
  if (!runnerPat(env)) throw new HttpError(503, "GITHUB_RUNNER_PAT is not set on the controller");
  const r = await gh(env, path, init, runnerPat(env));
  if (r.status === 204) return null;
  const text = await r.text();
  if (!r.ok) throw new HttpError(r.status === 404 ? 404 : 502, `github ${init.method || "GET"} ${path.split("?")[0]}: ${r.status} ${scrub(env, text.slice(0, 200))}`);
  return text ? JSON.parse(text) : null;
}

export interface GhRunner {
  id: number;
  name: string;
  status: string; // online | offline
  busy: boolean;
  labels: string[];
}

/** A one-hour, single-use registration token (never stored or logged). */
export async function runnerRegistrationToken(env: Env): Promise<string> {
  const j = await ghRunner(env, `/repos/${defaults.githubRepo(env)}/actions/runners/registration-token`, { method: "POST" });
  if (!j?.token) throw new HttpError(502, "github: no registration token in the answer");
  return String(j.token);
}

export async function listRunners(env: Env): Promise<GhRunner[]> {
  const out: GhRunner[] = [];
  for (let page = 1; page <= 5; page++) {
    const j = await ghRunner(env, `/repos/${defaults.githubRepo(env)}/actions/runners?per_page=100&page=${page}`);
    const rs = (j?.runners || []) as any[];
    for (const r of rs) out.push({ id: Number(r.id), name: String(r.name), status: String(r.status), busy: !!r.busy, labels: (r.labels || []).map((l: any) => String(l.name ?? l)) });
    if (rs.length < 100) break;
  }
  return out;
}

export async function deleteRunner(env: Env, id: number): Promise<void> {
  try {
    await ghRunner(env, `/repos/${defaults.githubRepo(env)}/actions/runners/${id}`, { method: "DELETE" });
  } catch (e) {
    if (e instanceof HttpError && e.status === 404) return;
    throw e;
  }
}

export interface QueuedJob {
  run_id: number;
  job_id: number;
  workflow: string;
  labels: string[];
  created_at: string;
}

/** Jobs waiting for a runner whose labels include `label`, in runs younger than maxAgeH. */
export async function queuedJobs(env: Env, label: string, maxAgeH = 24): Promise<{ jobs: QueuedJob[]; active_runs: { id: number; path: string; status: string }[] }> {
  const repo = defaults.githubRepo(env);
  const cutoff = Date.now() - maxAgeH * 3600_000;
  const runs: any[] = [];
  for (const st of ["queued", "in_progress"]) {
    const j = await ghRunner(env, `/repos/${repo}/actions/runs?status=${st}&per_page=30`);
    for (const r of j?.workflow_runs || []) if (Date.parse(r.created_at) >= cutoff) runs.push(r);
  }
  const jobs: QueuedJob[] = [];
  for (const r of runs.slice(0, 20)) {
    const j = await ghRunner(env, `/repos/${repo}/actions/runs/${r.id}/jobs?filter=latest&per_page=100`);
    for (const x of j?.jobs || []) {
      const labels = (x.labels || []).map(String);
      if (x.status === "queued" && labels.includes(label)) jobs.push({ run_id: Number(r.id), job_id: Number(x.id), workflow: String(r.path || r.name || ""), labels, created_at: String(x.created_at || r.created_at) });
    }
  }
  return { jobs, active_runs: runs.map((r) => ({ id: Number(r.id), path: String(r.path || r.name || ""), status: String(r.status) })) };
}

/** A file of the repository at a ref (raw), e.g. the build pod server from main. */
export async function repoFile(env: Env, path: string, ref: string): Promise<string> {
  if (!/^[A-Za-z0-9._\/-]{1,200}$/.test(ref)) throw new HttpError(400, "server_ref: a branch, tag or sha");
  const r = await gh(env, `/repos/${defaults.githubRepo(env)}/contents/${path}?ref=${encodeURIComponent(ref)}`, { headers: { accept: "application/vnd.github.raw+json" } } as RequestInit, runnerPat(env) || env.GITHUB_PAT);
  if (!r.ok) throw new HttpError(r.status === 404 ? 404 : 502, `github ${path}@${ref}: ${r.status} ${scrub(env, (await r.text()).slice(0, 200))}`);
  return r.text();
}
