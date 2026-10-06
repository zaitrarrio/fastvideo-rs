// Image preflight: before any pod is paid for, check that each image can run
// what the controller is about to ask of it.
//
// Why (docs/control/README.md §4 "Image preflight"): on 2026-10-06 the edge
// cluster h3-and-ltx started on channel `stable` = 2cd1ba0 (2026-09-29), a
// build from before the edge fronts. Its fv-serve rejects
// FV_AUTH_MODE=trust-edge at config load ("unknown variant `trust-edge`")
// and exits 2 before tracing or log shipping start, so the two RTX PRO 6000
// pods crash-looped for 14 minutes with no log and no ready front.
//
// How: every image carries the label org.opencontainers.image.revision (the
// git sha CI built it from). GitHub's compare API says whether that revision
// contains the commit that introduced a feature. When either answer is
// unavailable (no label, no GITHUB_PAT, GitHub down) the check is "unknown":
// a warning, never a refusal.
import { defaults, type Env } from "../env";
import { fetchWithTimeout } from "../util";
import type { ClusterSpec } from "./spec";
import { isEdge } from "./spec";

/** Features a worker image must have, by the commit that introduced each one. */
export const IMAGE_FEATURES = {
  edge_front: {
    sha: "73d770d7cb349fe999c3cb1660ca4df02378baa9",
    date: "2026-10-06",
    what: "edge fronts (FV_AUTH_MODE=trust-edge, FV_DISPATCH_FRONT): an older fv-serve exits at config load",
  },
  direct_worker: {
    sha: "6764df8a05c587610b7a0843989def8d646243b0",
    date: "2026-10-02",
    what: "direct workers (FV_WORKER_DIRECT, the controller's admin token and D1 keys)",
  },
  log_ship: {
    sha: "7ca2e4a0ad33aa3a04baa413a9df6b83ceb8177e",
    date: "2026-09-29",
    what: "log shipping (FV_LOG_SHIP_*): an older image's logs come only from the Runpod container log",
  },
} as const;
export type ImageFeature = keyof typeof IMAGE_FEATURES;

const ACCEPT = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json";

/** The git revision an image was built from (its org.opencontainers.image.revision label), or null when it cannot be read. */
export async function imageRevision(env: Env, ref: string): Promise<string | null> {
  const m = /^([^/]+)\/([^@:]+(?:\/[^@:]+)*)(?::([A-Za-z0-9._-]+))?(?:@(sha256:[0-9a-f]{64}))?$/.exec(ref);
  if (!m) return null;
  const repo = m[2]!;
  const want = m[4] || m[3] || "latest";
  const base = defaults.ghcr(env);
  try {
    const tok = ((await (await fetchWithTimeout(`${base}/token?scope=repository:${repo}:pull`, { timeoutMs: 15000 })).json().catch(() => ({}))) as { token?: string }).token || "";
    const get = async (path: string, accept = ACCEPT) => {
      const r = await fetchWithTimeout(`${base}/v2/${repo}/${path}`, { headers: { accept, authorization: `Bearer ${tok}` }, timeoutMs: 15000 });
      if (!r.ok) return null;
      return (await r.json().catch(() => null)) as any;
    };
    let man = await get(`manifests/${want}`);
    if (man?.manifests) {
      const sub = man.manifests.find((x: any) => x.platform?.architecture === "amd64" && x.platform?.os === "linux") || man.manifests[0];
      man = sub?.digest ? await get(`manifests/${sub.digest}`) : null;
    }
    const cfg = man?.config?.digest;
    if (!cfg) return null;
    const conf = await get(`blobs/${cfg}`, "application/json");
    const rev = conf?.config?.Labels?.["org.opencontainers.image.revision"];
    return typeof rev === "string" && /^[0-9a-f]{7,40}$/.test(rev) ? rev : null;
  } catch {
    return null;
  }
}

const containsCache = new Map<string, boolean>();
/** Whether `rev` contains commit `base` (GitHub compare: ahead or identical); null when GitHub cannot say. */
export async function revisionContains(env: Env, rev: string, base: string): Promise<boolean | null> {
  if (rev === base || base.startsWith(rev) || rev.startsWith(base)) return true;
  const key = `${rev}>${base}`;
  if (containsCache.has(key)) return containsCache.get(key)!;
  if (!env.GITHUB_PAT) return null;
  try {
    const r = await fetchWithTimeout(`${defaults.githubApi(env)}/repos/${defaults.githubRepo(env)}/compare/${base}...${rev}?per_page=1`, {
      headers: { authorization: `Bearer ${env.GITHUB_PAT}`, accept: "application/vnd.github+json", "x-github-api-version": "2022-11-28", "user-agent": "fv-control" },
      timeoutMs: 20000,
    });
    if (!r.ok) return null;
    const st = ((await r.json().catch(() => ({}))) as { status?: string }).status;
    if (!st) return null;
    const yes = st === "ahead" || st === "identical";
    containsCache.set(key, yes);
    return yes;
  } catch {
    return null;
  }
}

/** The features a cluster's workers need: errors refuse the start, warnings are logged. */
export function neededFeatures(spec: ClusterSpec): { feature: ImageFeature; fatal: boolean }[] {
  const out: { feature: ImageFeature; fatal: boolean }[] = [];
  if (isEdge(spec)) out.push({ feature: "edge_front", fatal: true });
  else out.push({ feature: "direct_worker", fatal: false });
  if (spec.log_shipping) out.push({ feature: "log_ship", fatal: false });
  return out;
}

export interface Preflight {
  errors: string[];
  warnings: string[];
  /** Image → revision (null: unreadable). */
  revisions: Record<string, string | null>;
}
/** Checks each pool's image against what the spec asks of it (one lookup per distinct image). */
export async function checkImages(env: Env, spec: ClusterSpec, images: Record<string, string>): Promise<Preflight> {
  const out: Preflight = { errors: [], warnings: [], revisions: {} };
  const byImage = new Map<string, string[]>();
  for (const [pool, img] of Object.entries(images)) {
    if (!spec.pools.some((p) => p.id === pool && p.count > 0)) continue;
    byImage.set(img, [...(byImage.get(img) || []), pool]);
  }
  const src = spec.image.channel ? `channel ${spec.image.channel}` : spec.image.sha ? `sha ${spec.image.sha}` : `image ${spec.image.ref}`;
  for (const [img, pools] of byImage) {
    const rev = await imageRevision(env, img);
    out.revisions[img] = rev;
    if (!rev) {
      out.warnings.push(`${pools.join(", ")}: could not read the image's revision label (${img.split("@")[1]?.slice(0, 19) || img}); feature check skipped`);
      continue;
    }
    for (const { feature, fatal } of neededFeatures(spec)) {
      const f = IMAGE_FEATURES[feature];
      const has = await revisionContains(env, rev, f.sha);
      if (has === true) continue;
      if (has === null) {
        out.warnings.push(`${pools.join(", ")}: could not check whether ${rev.slice(0, 7)} has ${feature} (GitHub compare unavailable)`);
        continue;
      }
      const msg = `${pools.join(", ")}: the image (${src}, built from ${rev.slice(0, 7)}) predates ${f.what} (${f.sha.slice(0, 7)}, ${f.date})`;
      (fatal ? out.errors : out.warnings).push(msg);
    }
  }
  if (out.errors.length) out.errors.push(`use a newer image: image.channel "latest", image.sha of a build after these commits, or promote ${spec.image.channel || "the channel"} (fv-control.sh promote <sha> ${spec.image.channel || "stable"})`);
  return out;
}
