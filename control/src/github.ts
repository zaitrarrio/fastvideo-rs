// GitHub with GITHUB_PAT (a fine-grained token: actions:write, contents
// and metadata read on the repository): dispatch release.yml (promote /
// rollback), CI status of main. The token is never returned or logged.
import { defaults, type Env } from "./env";
import { fetchWithTimeout, HttpError, scrub } from "./util";

async function gh(env: Env, path: string, init: RequestInit = {}): Promise<Response> {
  if (!env.GITHUB_PAT) throw new HttpError(503, "GITHUB_PAT is not set on the controller");
  return fetchWithTimeout(`${defaults.githubApi(env)}${path}`, {
    ...init,
    headers: {
      authorization: `Bearer ${env.GITHUB_PAT}`,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
      "user-agent": "fv-control",
      ...(init.body ? { "content-type": "application/json" } : {}),
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
