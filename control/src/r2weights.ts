// The R2 weights store (docs/ops/r2-weights.md): weights_source r2. Our own
// bucket (fv-weights) holds each weight tree under weights/<dest>/<rev>/ with a
// manifest (file list, sizes, SHA-256 of every file and of every 64 MiB part)
// at manifests/<dest>/<rev>.json, written last by scripts/gpu/r2-weights.py.
//
// fv-control holds a read-only R2 key (WEIGHTS_R2_ACCESS_KEY_ID / _SECRET_ACCESS_KEY)
// and never hands it out: a pod asks POST /ingest/v1/weights-plan (weights-plan.ts)
// with its cluster's ingest token and gets a *plan*, the manifests of its pool's
// trees with a presigned GET URL per file (WEIGHTS_R2_URL_TTL_S, default 1 h;
// the fetcher asks again on a 403). The pod runs scripts/gpu/r2-fetch.py
// (served at /ingest/v1/r2-fetch.py, checked against R2_FETCH_SHA256): many
// ranged GETs at once, every part checked against the manifest.
//
// Which trees exist is pinned in the repo (scripts/gpu/weights-r2.tsv ->
// R2_TREES, with the manifest's SHA-256): a manifest whose bytes differ is
// refused, so what a pod loads is what the repo recorded.
import type { Env } from "./env";
import { hex } from "./crypto";
import { R2_FETCH_SHA256 } from "./r2-fetch-script";
import { R2_TREES, type R2Tree } from "./r2-trees";
import { MXFP8_TREE_GLOB } from "./recipe-trees";
import { fetchWithTimeout, parseJson } from "./util";

export const DEFAULT_WEIGHTS_BUCKET = "fv-weights";
export const PLAN_FORMAT = "fv-weights-plan/1";
export const MANIFEST_FORMAT = "fv-weights-manifest/1";
/** A boot's R2 download rate for estimates (GB/s): the fixed fetcher (64 × 128 MiB, watchdog) on two Runpod serverless
 * H100 workers, 2026-10-12 round 2: 0.69 (h3, 177 GB) and 0.41 GB/s (wan, 59 GB, one part's tail cost ~45 s); mean
 * 0.55 (docs/ops/r2-weights.md §5.4, docs/ops/r2-bench-2026-10-12-round2.md). Round 1 (old fetcher): 0.50 / 0.77 / 0.33. */
export const R2_DOWNLOAD_GBPS = 0.55;
/** The aux files (AUX_FILES: TAE decoders, LPIPS) as published trees: a boot with weights_source r2 fetches them with the
 * weights instead of from GitHub / download.pytorch.org (2026-10-12: up to 151 s on one host), then checks each file
 * against AUX_FILES' SHA-256 (test/unit/preset-trees.test.ts: the same files and bytes). */
export const R2_AUX_TREES = ["auxiliary/lpips", "auxiliary/tae"];
/** The aux trees a boot fetches with its weights: the published ones, unless WEIGHTS_R2_AUX=0 (then the pinned URLs). */
export const r2AuxTrees = (env: Env) => (env.WEIGHTS_R2_AUX === "0" ? [] : R2_AUX_TREES.filter((t) => r2Trees(env)[t]));

export interface R2Cfg {
  endpoint: string;
  bucket: string;
  accessKeyId: string;
  secretAccessKey: string;
  ttlS: number;
}

/** The weights store's settings, or why weights_source r2 is off. */
export function r2Cfg(env: Env): { cfg: R2Cfg | null; off: string | null } {
  if (!env.WEIGHTS_R2_ACCESS_KEY_ID || !env.WEIGHTS_R2_SECRET_ACCESS_KEY) return { cfg: null, off: "the R2 weights store is not configured (secrets WEIGHTS_R2_ACCESS_KEY_ID / WEIGHTS_R2_SECRET_ACCESS_KEY; docs/ops/r2-weights.md §4)" };
  const endpoint = (env.WEIGHTS_R2_ENDPOINT || (env.CF_ACCOUNT_ID ? `https://${env.CF_ACCOUNT_ID}.r2.cloudflarestorage.com` : "")).replace(/\/$/, "");
  if (!endpoint) return { cfg: null, off: "the R2 weights store has no endpoint (WEIGHTS_R2_ENDPOINT or CF_ACCOUNT_ID)" };
  if (!env.PUBLIC_URL) return { cfg: null, off: "PUBLIC_URL is not set: pods ask fv-control for their weights' URLs there" };
  const ttl = Number(env.WEIGHTS_R2_URL_TTL_S || 3600);
  return { cfg: { endpoint, bucket: env.WEIGHTS_R2_BUCKET || DEFAULT_WEIGHTS_BUCKET, accessKeyId: env.WEIGHTS_R2_ACCESS_KEY_ID, secretAccessKey: env.WEIGHTS_R2_SECRET_ACCESS_KEY, ttlS: Number.isFinite(ttl) ? Math.min(Math.max(ttl, 300), 6 * 3600) : 3600 }, off: null };
}

/** The published trees: R2_TREES (the repo's record) plus WEIGHTS_R2_EXTRA_TREES (JSON, the same shape; tests and an
 * owner's override until the next deploy). */
export function r2Trees(env: Env): Record<string, R2Tree> {
  const extra = parseJson<Record<string, R2Tree>>(env.WEIGHTS_R2_EXTRA_TREES, {});
  const ok = Object.fromEntries(Object.entries(extra && typeof extra === "object" ? extra : {}).filter(([, t]) => t && typeof t.rev === "string" && /^[0-9a-f]{64}$/.test(String(t.manifest_sha256)) && Number.isFinite(t.bytes)));
  return { ...R2_TREES, ...ok };
}
export const r2Published = (env: Env, tree: string) => !!r2Trees(env)[tree];
/** The bytes of these trees per the record (unknown trees: 0). */
export const r2Bytes = (env: Env, trees: string[]) => trees.reduce((s, t) => s + (r2Trees(env)[t]?.bytes ?? 0), 0);

// ---------------------------------------------------------------- SigV4 presigned GET (WebCrypto)
const te = new TextEncoder();
/** RFC 3986 encoding as SigV4 wants it (encodeURIComponent leaves !'()* alone). */
export const uriEncode = (s: string) => encodeURIComponent(s).replace(/[!'()*]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase());
export const encodeKey = (k: string) => k.split("/").map(uriEncode).join("/");
async function hmac(key: ArrayBuffer | Uint8Array, msg: string): Promise<ArrayBuffer> {
  const k = await crypto.subtle.importKey("raw", key as BufferSource, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return crypto.subtle.sign("HMAC", k, te.encode(msg));
}
const sha256 = async (s: string | Uint8Array) => hex(await crypto.subtle.digest("SHA-256", (typeof s === "string" ? te.encode(s) : s) as BufferSource));

/** A presigned GET URL for host + path (path already encoded), query auth with UNSIGNED-PAYLOAD (docs.aws.amazon.com
 * AmazonS3/latest/API/sigv4-query-string-auth.html; R2: developers.cloudflare.com/r2/api/s3/presigned-urls). */
export async function presignPath(scheme: string, host: string, path: string, accessKeyId: string, secret: string, ttlS: number, at: Date = new Date(), region = "auto"): Promise<string> {
  const amzDate = at.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  const date = amzDate.slice(0, 8);
  const scope = `${date}/${region}/s3/aws4_request`;
  const q: Record<string, string> = { "X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": `${accessKeyId}/${scope}`, "X-Amz-Date": amzDate, "X-Amz-Expires": String(Math.floor(ttlS)), "X-Amz-SignedHeaders": "host" };
  const cq = Object.keys(q)
    .sort()
    .map((k) => `${uriEncode(k)}=${uriEncode(q[k]!)}`)
    .join("&");
  const canonical = ["GET", path, cq, `host:${host}\n`, "host", "UNSIGNED-PAYLOAD"].join("\n");
  const toSign = ["AWS4-HMAC-SHA256", amzDate, scope, await sha256(canonical)].join("\n");
  let k = await hmac(te.encode("AWS4" + secret), date);
  for (const p of [region, "s3", "aws4_request"]) k = await hmac(k, p);
  const sig = hex(await hmac(k, toSign));
  return `${scheme}://${host}${path}?${cq}&X-Amz-Signature=${sig}`;
}
/** A presigned GET of one object of the weights bucket (path style). */
export function presignGet(cfg: R2Cfg, key: string, ttlS = cfg.ttlS, at: Date = new Date()): Promise<string> {
  const u = new URL(cfg.endpoint);
  return presignPath(u.protocol.replace(":", ""), u.host, `/${uriEncode(cfg.bucket)}/${encodeKey(key)}`, cfg.accessKeyId, cfg.secretAccessKey, ttlS, at);
}

export const manifestKey = (dest: string, rev: string) => `manifests/${dest}/${rev}.json`;
export const objectKey = (dest: string, rev: string, path: string) => `weights/${dest}/${rev}/${path}`;

// ---------------------------------------------------------------- manifests
export interface ManifestFile {
  path: string;
  size: number;
  sha256: string;
  parts: string[];
}
export interface R2Manifest {
  format: string;
  dest: string;
  rev: string;
  part_size: number;
  total_bytes: number;
  files: ManifestFile[];
  source?: unknown;
}
const SAFE_PATH = (p: string) => !!p && !p.startsWith("/") && !p.includes("\\") && p.split("/").every((x) => x && x !== "." && x !== "..");
const CACHE = new Map<string, R2Manifest>();

/** A published tree's manifest from the bucket, checked against the SHA-256 the repo pinned; cached per isolate. */
export async function r2Manifest(env: Env, cfg: R2Cfg, dest: string): Promise<R2Manifest> {
  const t = r2Trees(env)[dest];
  if (!t) throw new Error(`${dest} is not in the R2 weights store (scripts/gpu/weights-r2.tsv)`);
  const hit = CACHE.get(t.manifest_sha256);
  if (hit && hit.dest === dest) return hit;
  const res = await fetchWithTimeout(await presignGet(cfg, manifestKey(dest, t.rev), 300), { timeoutMs: 20000 });
  if (!res.ok) throw new Error(`R2 manifest of ${dest}@${t.rev}: HTTP ${res.status}`);
  const raw = new Uint8Array(await res.arrayBuffer());
  const got = await sha256(raw);
  if (got !== t.manifest_sha256) throw new Error(`R2 manifest of ${dest}@${t.rev}: sha256 ${got.slice(0, 16)}… is not the one scripts/gpu/weights-r2.tsv pins (${t.manifest_sha256.slice(0, 16)}…): refused`);
  const m = parseJson<R2Manifest | null>(new TextDecoder().decode(raw), null);
  if (!m || m.format !== MANIFEST_FORMAT || m.dest !== dest || m.rev !== t.rev || !Array.isArray(m.files)) throw new Error(`R2 manifest of ${dest}@${t.rev}: not a ${MANIFEST_FORMAT} manifest of that tree`);
  for (const f of m.files) {
    const parts = f.size ? Math.ceil(f.size / m.part_size) : 0;
    if (!SAFE_PATH(f.path) || !/^[0-9a-f]{64}$/.test(f.sha256) || !Array.isArray(f.parts) || f.parts.length !== parts) throw new Error(`R2 manifest of ${dest}@${t.rev}: bad entry ${JSON.stringify(f.path)}`);
  }
  CACHE.set(t.manifest_sha256, m);
  return m;
}

export interface PlanTree {
  dest: string;
  rev: string;
  part_size: number;
  total_bytes: number;
  files: (ManifestFile & { url: string })[];
}
/** The plan r2-fetch.py runs: these trees' manifests with a presigned URL per file. */
export async function r2Plan(env: Env, cfg: R2Cfg, trees: string[], at: Date = new Date()): Promise<{ format: string; expires_at: number; trees: PlanTree[] }> {
  const out: PlanTree[] = [];
  for (const dest of trees) {
    const m = await r2Manifest(env, cfg, dest);
    const files = await Promise.all(m.files.map(async (f) => ({ path: f.path, size: f.size, sha256: f.sha256, parts: f.parts, url: await presignGet(cfg, objectKey(dest, m.rev, f.path), cfg.ttlS, at) })));
    out.push({ dest, rev: m.rev, part_size: m.part_size, total_bytes: m.total_bytes, files });
  }
  return { format: PLAN_FORMAT, expires_at: Math.floor(at.getTime() / 1000) + cfg.ttlS, trees: out };
}

/** The env a pod with weights_source r2 gets (beside FV_WEIGHTS_SOURCE, FV_ENDPOINT_REPORT_TOKEN, FV_POD_ID). */
export function r2PodEnv(env: Env): Record<string, string> {
  const base = (env.PUBLIC_URL || "").replace(/\/$/, "");
  return { FV_WEIGHTS_PLAN_URL: `${base}/ingest/v1/weights-plan`, FV_R2_FETCH_URL: `${base}/ingest/v1/r2-fetch.py`, FV_R2_FETCH_SHA256: R2_FETCH_SHA256 };
}

/** Container / VM disk for these trees: the bytes × 1.15 + 30 GB (image, scratch, the whole-file checks' headroom). */
export const r2DiskGb = (env: Env, trees: string[]) => Math.ceil((r2Bytes(env, trees) / 1e9) * 1.15 + 30);

// A Runpod worker with weights_source r2 (no volume): its pod id for the plan request, the trees (R2_FETCH_SH), then
// the aux files (TAE decoders, LPIPS) by their pinned URLs and SHA-256 (FV_WEIGHTS_TREES_B64 aux rows), as the
// GMI / Brev boot's WEIGHTS_SH does. Inserted into the worker boot by payloads.ts bootFor.
export const R2_RUNPOD_SH = () => `export FV_POD_ID="\${FV_POD_ID:-\${RUNPOD_POD_ID:-}}"
${R2_FETCH_SH}if [ "\${FV_WEIGHTS_SOURCE:-none}" = r2 ] && [ -n "\${FV_WEIGHTS_TREES_B64:-}" ]; then
  printf "%s" "$FV_WEIGHTS_TREES_B64" | base64 -d | while IFS="$(printf '\\t')" read -r kind a b c d; do
    [ "$kind" = aux ] || continue
    p="$FV_WEIGHTS/$a"; mkdir -p "$(dirname "$p")"
    if [ -f "$p" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$c" ]; then continue; fi
    echo "[fv-boot] $(date -u +%H:%M:%S) aux $a <- $b" >&2
    curl -fsSL --connect-timeout 10 --retry 3 --max-time 600 -o "$p.part" "$b" && [ "$(sha256sum "$p.part" | cut -d' ' -f1)" = "$c" ] && [ "$(stat -c %s "$p.part")" = "$d" ] && mv "$p.part" "$p" || { echo "[fv-boot] $(date -u +%H:%M:%S) FAILED: aux $a" >&2; sleep 30; exit 1; }
  done || exit 1
fi
`;

// The weights step of a pod's boot with weights_source r2 (PROVIDER_BOOT on GMI / Brev, the Runpod worker boot):
// python3 (installed if missing; stdlib only), the fetcher from fv-control checked by SHA-256, then the fetch of every
// tree of the pod's plan into $FV_WEIGHTS (kept: a tree complete at its revision; resumed: a partial one). A failure
// goes through the boot's `fail` (GMI / Brev: reported to fv-control) or exits (Runpod restarts the container, and the
// fetch resumes). Progress lines reach the pod's log; GMI / Brev also report them (FV_ENDPOINT_REPORT_URL).
export const R2_FETCH_SH = `# ---- weights: the R2 weights store (docs/ops/r2-weights.md): presigned URLs from fv-control, no R2 key on the pod
if [ "\${FV_WEIGHTS_SOURCE:-none}" = r2 ]; then
  r2fail() { if command -v fail >/dev/null 2>&1; then fail "$1"; else echo "[fv-boot] $(date -u +%H:%M:%S) FAILED: $1" >&2; sleep 30; exit 1; fi; }
  r2say() { echo "[fv-boot] $(date -u +%H:%M:%S) $*" >&2; }
  r2el() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }
  r2say "weights: R2 weights store (plan: $FV_WEIGHTS_PLAN_URL)"
  R2S="\${FV_R2_STATE:-/fvstate}"; R2T="\${FV_R2_TOOLS:-/opt/fv}"
  mkdir -p "$R2S" "$R2T" "$FV_WEIGHTS"
  t="$(date +%s.%N)"
  if ! command -v python3 >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    { apt-get -o Acquire::http::Timeout=15 -o Acquire::Retries=3 update -qq && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::http::Timeout=15 -o Acquire::Retries=3 install -y -qq --no-install-recommends python3 curl ca-certificates; } >"$R2S/apt-r2.log" 2>&1 || r2fail "apt python3"
    r2say "step tools: $(r2el "$t")s (apt-get python3 curl)"
  fi
  # The fetcher: the image's copy when it is the one fv-control pins, else fv-control's (short timeouts, retried).
  t="$(date +%s.%N)"; R2F="\${FV_R2_FETCH_IMAGE:-/opt/fastvideo-rs/bin/r2-fetch.py}"
  if [ -f "$R2F" ] && echo "$FV_R2_FETCH_SHA256  $R2F" | sha256sum -c - >/dev/null 2>&1; then r2say "step fetcher: from the image"
  else
    R2F="$R2T/r2-fetch.py"
    curl -fsS --connect-timeout 5 --max-time 30 --retry 4 --retry-delay 1 --retry-all-errors -o "$R2F" "$FV_R2_FETCH_URL" || r2fail "r2-fetch.py download"
    echo "$FV_R2_FETCH_SHA256  $R2F" | sha256sum -c - >/dev/null 2>&1 || r2fail "r2-fetch.py sha256"
    r2say "step fetcher: $(r2el "$t")s (downloaded from fv-control)"
  fi
  # The MXFP8 pre-quantized DiTs are of no use below sm_100 (the engine runs W8A8 there): not fetched.
  if [ -z "\${FV_R2_SKIP_TREES+x}" ]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
    case "$cc" in [1-9].*) export FV_R2_SKIP_TREES="${MXFP8_TREE_GLOB}" ;; esac
  fi
  t="$(date +%s.%N)"
  set -- --root "$FV_WEIGHTS" --plan-url "$FV_WEIGHTS_PLAN_URL" --token-env FV_ENDPOINT_REPORT_TOKEN --pod "\${FV_POD_ID:-}" --status "$R2S/r2-fetch.json"
  if [ -n "\${FV_ENDPOINT_REPORT_URL:-}" ]; then set -- "$@" --report-url "$FV_ENDPOINT_REPORT_URL"; fi
  python3 "$R2F" "$@" 2>&1 | tee "$R2S/r2-fetch.log" >&2
  [ "\${PIPESTATUS[0]}" = 0 ] || r2fail "r2 weights: $(grep FAILED "$R2S/r2-fetch.log" | tail -1 | cut -c1-240 | tr '"' "'")"
  r2say "step fetch: $(r2el "$t")s"
  set --
fi
`;
