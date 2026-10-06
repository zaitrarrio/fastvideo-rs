// GHCR (public package, anonymous pull token): tag -> digest, tag lists.
import type { ClusterSpec } from "./cluster/spec";
import { defaults, type Env } from "./env";
import { fetchWithTimeout, HttpError } from "./util";

const ACCEPT =
  "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json";

function split(ref: string): { host: string; repo: string; tag: string } {
  const m = /^([^/]+)\/(.+?)(?::([A-Za-z0-9._-]+))?$/.exec(ref);
  if (!m) throw new HttpError(400, `not an image reference: ${ref}`);
  return { host: m[1]!, repo: m[2]!, tag: m[3] || "latest" };
}
async function pullToken(env: Env, repo: string): Promise<string> {
  const r = await fetchWithTimeout(`${defaults.ghcr(env)}/token?scope=repository:${repo}:pull`, { timeoutMs: 15000 });
  const j = (await r.json().catch(() => ({}))) as { token?: string };
  return j.token || "";
}
/** ghcr.io/<repo>:<tag> -> ghcr.io/<repo>@sha256:… (a digest ref is returned as is). */
export async function resolveDigest(env: Env, ref: string): Promise<string> {
  if (ref.includes("@sha256:")) return ref;
  const { host, repo, tag } = split(ref);
  const tok = await pullToken(env, repo);
  const r = await fetchWithTimeout(`${defaults.ghcr(env)}/v2/${repo}/manifests/${tag}`, {
    method: "HEAD",
    headers: { accept: ACCEPT, authorization: `Bearer ${tok}` },
    timeoutMs: 15000,
  });
  const digest = r.headers.get("docker-content-digest");
  if (!r.ok || !digest?.startsWith("sha256:")) throw new HttpError(404, `could not resolve ${ref} to a digest (${r.status})`);
  return `${host}/${repo}@${digest}`;
}
export async function listTags(env: Env, repoRef = defaults.serveRepo(env)): Promise<string[]> {
  const { repo } = split(repoRef);
  const tok = await pullToken(env, repo);
  const out: string[] = [];
  let last = "";
  for (let i = 0; i < 20; i++) {
    const r = await fetchWithTimeout(`${defaults.ghcr(env)}/v2/${repo}/tags/list?n=1000${last ? `&last=${encodeURIComponent(last)}` : ""}`, {
      headers: { authorization: `Bearer ${tok}` },
      timeoutMs: 20000,
    });
    if (!r.ok) throw new HttpError(502, `ghcr tags: ${r.status}`);
    const j = (await r.json()) as { tags?: string[] };
    const tags = j.tags || [];
    out.push(...tags);
    if (tags.length < 1000) break;
    last = tags[tags.length - 1]!;
  }
  return out;
}

/** The image of each pod of a cluster: per-variant digests for a channel or sha, one all-in-one image for a ref. */
export async function resolveClusterImages(env: Env, spec: ClusterSpec): Promise<Record<string, string>> {
  const repo = defaults.serveRepo(env);
  const out: Record<string, string> = {};
  const suffix = spec.image.channel ? spec.image.channel : spec.image.sha ? `sha-${spec.image.sha.slice(0, 7)}` : null;
  const imageFor = async (variant: string, override?: string) => {
    if (override) return resolveDigest(env, override);
    if (spec.image.ref) return resolveDigest(env, spec.image.ref);
    try {
      return await resolveDigest(env, `${repo}:${variant}-${suffix}`);
    } catch (e) {
      // Before the first promotion there is no :stable (releases.md): :latest, as the scripts do.
      if (spec.image.channel === "stable") return resolveDigest(env, `${repo}:${variant}-latest`);
      throw e;
    }
  };
  for (const p of spec.pools) out[p.id] = await imageFor(p.variant, p.image);
  return out;
}
