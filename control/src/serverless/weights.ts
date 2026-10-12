// weights_source r2 for Runpod serverless endpoints (docs/control/serverless.md §1b, docs/ops/r2-weights.md §9).
//
// A worker of such an endpoint mounts no network volume (so no data-centre restriction): at boot, before fv-serve
// starts, it fetches its endpoint's weight trees from the R2 weights store onto its container disk with the same
// fetcher as GMI / Brev / Runpod pods (scripts/gpu/r2-fetch.py, served at /ingest/v1/r2-fetch.py and checked against
// R2_FETCH_SHA256), through presigned URLs fv-control mints (weights-plan.ts). It never holds an R2 key.
//
// Auth: every r2 endpoint has its own weights token (fvw_<64 hex>) in its template env (FV_WEIGHTS_TOKEN); D1 keeps
// its SHA-256 (serverless_endpoints.weights_token_hash). It is good for one thing: the plan of *that* endpoint's
// trees (derived from its stored spec, never from the request) and its boot reports. A template update rotates it
// (the previous one stays valid for SLS_TOKEN_GRACE_MS while Runpod rolls the workers); a delete revokes it.
//
// Caching (the worker's disk): the boot first checks every tree at $FV_WEIGHTS/<dest> for `.complete` and
// `.fv-revision` = the pinned revision (FV_R2_TREE_REVS); when all are there (a FlashBoot revival that kept its
// disk, or a restarted container) it skips the plan request and the fetch entirely. Otherwise the fetcher keeps the
// complete trees and resumes partial files part by part. Optionally (off by default) a small per-DC network volume
// the owner made serves as an add-only cache filled from R2 (FV_SLS_R2_CACHE_VOLUMES).
import { sha256Hex, randomToken } from "../crypto";
import type { Env } from "../env";
import { DEFAULT_TREE_GB, TREE_GB } from "../brev-park";
import { r2AuxTrees, R2_DOWNLOAD_GBPS, r2Cfg, r2PodEnv, r2Trees } from "../r2weights";
import { R2_TREES } from "../r2-trees";
import { AUX_FILES, HUB_TREES } from "../weights-sources";
import { MXFP8_TREE_GLOB } from "../recipe-trees";
import { parseJson } from "../util";
import { slsPreset } from "./presets";
import { configOf } from "./serves";
import type { EndpointSpec } from "./spec";

type SpecLike = Pick<EndpointSpec, "variant" | "compute"> & Partial<Pick<EndpointSpec, "preset" | "config" | "config_toml" | "image" | "weights_source" | "network_volume" | "data_centers" | "container_disk_gb">>;

/** The previous weights token stays valid this long after a rotation (Runpod's rolling release of the workers). */
export const SLS_TOKEN_GRACE_MS = 30 * 60_000;
/** Where the trees go on a worker (the configs' ${FV_WEIGHTS}; runpod-longlive.toml names /workspace/weights). */
export const SLS_R2_WEIGHTS = "/workspace/weights";
/** On an R2 cache volume: the trees under this directory of /runpod-volume (never the volume's weights/). */
export const SLS_R2_CACHE_ROOT = "/runpod-volume/fv-r2-cache";

/** The weight trees an endpoint's workers load: `${FV_WEIGHTS}/<tree>` (or /workspace/weights/<tree>) in its worker
 * config, plus its preset's list and (published) pre-quantized trees; a nested published tree (longlive-plug/x,
 * h3-base/text_encoder_fp8) by its full name: the fetcher places it inside its parent, as on the volume. */
export function slsTrees(spec: SpecLike): string[] {
  if (spec.compute !== "GPU") return [];
  const p = slsPreset(spec.preset);
  const out = new Set<string>([...(p?.weights ?? []), ...(p?.prequant ?? []).filter((t) => R2_TREES[t])]);
  const text = configOf({ variant: spec.variant, config: spec.config, config_toml: spec.config_toml, image: spec.image }).text || "";
  for (const line of text.split("\n")) {
    if (/^\s*#/.test(line)) continue;
    for (const m of line.matchAll(/(?:\$\{FV_WEIGHTS\}|\/workspace\/weights|\/runpod-volume\/weights)\/([A-Za-z0-9._-]+)(?:\/([A-Za-z0-9._-]+))?/g)) {
      const nested = m[2] ? `${m[1]}/${m[2]}` : "";
      out.add(nested && (HUB_TREES[nested] || R2_TREES[nested]) ? nested : m[1]!);
    }
  }
  out.delete("auxiliary");
  return [...out].sort();
}
/** What an r2 worker fetches: its trees plus the aux trees (TAE, LPIPS) from R2 (r2AuxTrees), when it loads any tree. */
export function slsFetchTrees(env: Env, spec: SpecLike): string[] {
  const t = slsTrees(spec);
  return t.length ? [...new Set([...t, ...r2AuxTrees(env)])].sort() : t;
}

/** A tree's bytes: the published manifest's, else the size table's (brev-park.ts TREE_GB), else DEFAULT_TREE_GB. */
export function slsTreeBytes(trees: string[], pub: Record<string, { bytes: number }> = R2_TREES): number {
  return trees.reduce((s, t) => s + (pub[t]?.bytes ?? (TREE_GB[t] ?? DEFAULT_TREE_GB) * 1e9), 0);
}
/** Container disk for the trees on a worker: bytes × 1.15 + 30 GB (image, scratch, whole-file check headroom), as r2DiskGb. */
export const slsR2DiskGb = (trees: string[], pub?: Record<string, { bytes: number }>) => Math.ceil((slsTreeBytes(trees, pub) / 1e9) * 1.15 + 30);

/** The owner's R2 cache volumes ({id: dc}; FV_SLS_R2_CACHE_VOLUMES). Empty by default. */
export function cacheVolumes(env: Env): Record<string, string> {
  const v = parseJson<Record<string, unknown>>(env.FV_SLS_R2_CACHE_VOLUMES, {});
  return Object.fromEntries(Object.entries(v && typeof v === "object" ? v : {}).filter(([id, dc]) => /^[a-z0-9]{6,16}$/.test(id) && typeof dc === "string")) as Record<string, string>;
}

export interface WeightsIssue {
  path: (string | number)[];
  message: string;
}
/** What only fv-control can check for weights_source r2 (env: the store's settings, the published trees, the owner's
 * cache volumes): refused on create / update. */
export function slsWeightsIssues(env: Env, spec: SpecLike): { issues: WeightsIssue[]; warnings: WeightsIssue[] } {
  const issues: WeightsIssue[] = [];
  const warnings: WeightsIssue[] = [];
  if (spec.weights_source !== "r2" || spec.compute !== "GPU") return { issues, warnings };
  const { off } = r2Cfg(env);
  if (off) issues.push({ path: ["weights_source"], message: `weights_source r2: ${off}` });
  const trees = slsTrees(spec);
  const pub = r2Trees(env);
  const missing = trees.filter((t) => !pub[t]);
  if (!trees.length) issues.push({ path: ["weights_source"], message: "the config loads no weight tree fv-control knows: weights_source r2 has nothing to fetch (weights_source volume, or a preset)" });
  if (missing.length) issues.push({ path: ["weights_source"], message: `not in the R2 weights store yet: ${missing.join(", ")} (scripts/gpu/weights-r2.tsv; docs/ops/r2-weights.md §6): weights_source volume` });
  const need = slsR2DiskGb(trees, pub);
  if ((spec.container_disk_gb ?? 0) < need) issues.push({ path: ["container_disk_gb"], message: `weights_source r2 fetches ${(slsTreeBytes(trees, pub) / 1e9).toFixed(1)} GB onto the container disk: at least ${need} GB` });
  if (spec.network_volume) {
    const dc = cacheVolumes(env)[spec.network_volume];
    if (!dc) issues.push({ path: ["network_volume"], message: `weights_source r2 takes no weights volume; ${spec.network_volume} is not an R2 cache volume (FV_SLS_R2_CACHE_VOLUMES: ${Object.keys(cacheVolumes(env)).join(", ") || "none"})` });
    else if (spec.data_centers?.some((d) => d !== dc)) issues.push({ path: ["data_centers"], message: `a worker with cache volume ${spec.network_volume} must run in ${dc}` });
    else warnings.push({ path: ["network_volume"], message: `R2 cache volume ${spec.network_volume} (${dc}): workers there load the trees from the volume (filled from R2 once, add-only); the workers run in ${dc} only` });
  }
  return { issues, warnings };
}

/** A new weights token and its SHA-256 (what D1 keeps). */
export async function newWeightsToken(): Promise<{ token: string; hash: string }> {
  const token = randomToken("fvw_", 32);
  return { token, hash: await sha256Hex(token) };
}

/** The env an r2 endpoint's workers get beside workerEnv's (payloads.ts): the plan URL, the fetcher and its SHA-256,
 * the weights token, the trees at their revisions (the cache check), the aux files (pinned URL + SHA-256). */
export function slsWeightsEnv(env: Env, spec: SpecLike, token: string): Record<string, string> {
  const pub = r2Trees(env);
  const trees = slsFetchTrees(env, spec).filter((t) => pub[t]);
  const aux = AUX_FILES.map((a) => ["aux", a.path, a.url, a.sha256, String(a.size)].join("\t")).join("\n") + "\n";
  const base = (env.PUBLIC_URL || "").replace(/\/$/, "");
  return {
    FV_WEIGHTS_SOURCE: "r2",
    FV_WEIGHTS: SLS_R2_WEIGHTS,
    ...r2PodEnv(env),
    FV_WEIGHTS_TOKEN: token,
    FV_WEIGHTS_TREES_B64: btoa(aux),
    FV_R2_TREE_REVS: trees.map((t) => `${t}@${pub[t]!.rev}`).join(" "),
    FV_SLS_REPORT_URL: `${base}/ingest/v1/sls-report`,
    ...(spec.network_volume && cacheVolumes(env)[spec.network_volume] ? { FV_R2_CACHE_ROOT: SLS_R2_CACHE_ROOT } : {}),
  };
}

// ---------------------------------------------------------------- the worker's boot
/**
 * The weights step of an r2 worker's start (bash; payloads.ts puts it before fv-serve). It reports each phase to
 * fv-control (POST $FV_SLS_REPORT_URL with the weights token; best effort): boot, cache hit / miss, the fetcher's
 * progress and summary (its --report-url), serve. A cache hit (every tree complete at its pinned revision) asks
 * fv-control for nothing. With an R2 cache volume (FV_R2_CACHE_ROOT) the trees go there add-only (r2-fetch.py
 * --add-only: per-worker staging, never a removal) and $FV_WEIGHTS links to it; else onto the container disk.
 */
export const SLS_R2_SH = `# ---- weights: the R2 weights store (weights_source r2, docs/control/serverless.md §1b)
if [ "\${FV_WEIGHTS_SOURCE:-}" = r2 ]; then
  FV_BOOT_T0="$(date +%s)"; FV_BOOT_T0F="$(date +%s.%N)"; FV_BOOT="\${RUNPOD_POD_ID:-worker}@$FV_BOOT_T0"
  el() { awk -v a="$1" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }'; }
  say() { echo "[fv-boot] $(date -u +%H:%M:%S) +$(el "$FV_BOOT_T0F")s $*" >&2; }
  rep() { [ -n "\${FV_SLS_REPORT_URL:-}" ] || return 0; local d; d="$(printf "%s" "$2" | tr -d '"\\\\' | tr '\\n' ' ' | cut -c1-280)"
    curl -fsS --connect-timeout 5 --max-time 10 -o /dev/null -X POST -H "authorization: Bearer \${FV_WEIGHTS_TOKEN:-}" -H "content-type: application/json" --data "{\\"pod\\":\\"$FV_BOOT\\",\\"phase\\":\\"$1\\",\\"url\\":\\"\\",\\"detail\\":\\"$d\\"}" "$FV_SLS_REPORT_URL" >/dev/null 2>&1 || true; }
  # A timed boot step (name, its start from date +%s.%N, a note): logged with the time of day and the time since the
  # container's start, and reported (phase step) so the endpoint's boots show where a cold start went.
  step() { local d; d="$(el "$2")"; say "step $1: \${d}s\${3:+ ($3)}"; rep step "$1 \${d}s\${3:+ ($3)}"; }
  slsfail() { say "FAILED: $1"; rep failed "$1"; sleep "\${FV_FAIL_SLEEP_S:-30}"; exit 1; }
  S="\${FV_R2_STATE:-/fvstate}"
  rep boot "container started; \${fv_drv:-driver not checked}"
  WR="$FV_WEIGHTS"; ADD=""
  if [ -n "\${FV_R2_CACHE_ROOT:-}" ] && [ -d "$(dirname "$FV_R2_CACHE_ROOT")" ]; then WR="$FV_R2_CACHE_ROOT"; ADD="--add-only"; fi
  mkdir -p "$WR" "$S"
  if [ "$WR" != "$FV_WEIGHTS" ] && [ ! -e "$FV_WEIGHTS" ]; then mkdir -p "$(dirname "$FV_WEIGHTS")"; ln -s "$WR" "$FV_WEIGHTS"; fi
  # Trees this host does not fetch: the MXFP8 pre-quantized DiTs below sm_100 (the engine runs W8A8 there: unused),
  # and on an add-only cache volume the aux trees (their files may predate the trees; the pinned URLs fill them).
  if [ -z "\${FV_R2_SKIP_TREES+x}" ]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')"
    FV_R2_SKIP_TREES=""; case "$cc" in [1-9].*) FV_R2_SKIP_TREES="${MXFP8_TREE_GLOB}" ;; esac
    [ -n "$ADD" ] && FV_R2_SKIP_TREES="$FV_R2_SKIP_TREES auxiliary/*"
  fi
  export FV_R2_SKIP_TREES
  skipped() { local p r=1; set -f; for p in $FV_R2_SKIP_TREES; do case "$1" in $p) r=0 ;; esac; done; set +f; return $r; }
  t="$(date +%s.%N)"; miss=""
  for tr in \${FV_R2_TREE_REVS:-}; do
    d="\${tr%@*}"; r="\${tr##*@}"
    skipped "$d" && continue
    if [ -f "$WR/$d/.complete" ] && [ "$(cat "$WR/$d/.fv-revision" 2>/dev/null || true)" = "$r" ]; then :; else miss="$miss $d"; fi
  done
  if [ -z "$miss" ]; then
    say "weights: cache hit (every tree at its revision under $WR): no fetch"; rep cache "hit"
  else
    say "weights: cache miss:$miss: fetching from the R2 weights store\${FV_R2_SKIP_TREES:+ (skipping $FV_R2_SKIP_TREES)}"; rep cache "miss:$miss"
    T="\${FV_R2_TOOLS:-/opt/fv}"; mkdir -p "$T"
    t="$(date +%s.%N)"
    if command -v python3 >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then step tools "$t" "python3 and curl in the image"
    else
      { apt-get -o Acquire::http::Timeout=15 -o Acquire::Retries=3 update -qq && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::http::Timeout=15 -o Acquire::Retries=3 install -y -qq --no-install-recommends python3 curl ca-certificates; } >"$S/apt-r2.log" 2>&1 || slsfail "apt python3: $(tail -n 2 "$S/apt-r2.log" | tr '\\n' ' ')"
      step tools "$t" "apt-get python3 curl: the image has no python3 or curl"
    fi
    # The fetcher: the image's copy when it is the one fv-control pins (no download), else fv-control's.
    t="$(date +%s.%N)"; F="\${FV_R2_FETCH_IMAGE:-/opt/fastvideo-rs/bin/r2-fetch.py}"
    if [ -f "$F" ] && echo "$FV_R2_FETCH_SHA256  $F" | sha256sum -c - >/dev/null 2>&1; then step fetcher "$t" "from the image"
    else
      F="$T/r2-fetch.py"
      curl -fsS --connect-timeout 5 --max-time 30 --retry 4 --retry-delay 1 --retry-all-errors -o "$F" "$FV_R2_FETCH_URL" || slsfail "r2-fetch.py download from fv-control"
      echo "$FV_R2_FETCH_SHA256  $F" | sha256sum -c - >/dev/null 2>&1 || slsfail "r2-fetch.py sha256"
      step fetcher "$t" "downloaded from fv-control"
    fi
    t="$(date +%s.%N)"
    set +e
    python3 "$F" --root "$WR" --plan-url "$FV_WEIGHTS_PLAN_URL" --token-env FV_WEIGHTS_TOKEN --pod "$FV_BOOT" --status "$S/r2-fetch.json" --report-url "\${FV_SLS_REPORT_URL:-}" $ADD 2>&1 | tee "$S/r2-fetch.log" >&2
    rc="\${PIPESTATUS[0]}"; set -e
    [ "$rc" = 0 ] || slsfail "r2 weights: $(grep FAILED "$S/r2-fetch.log" | tail -1 | cut -c1-200)"
    step fetch "$t" "$(grep -o 'weights ready: .*' "$S/r2-fetch.log" | tail -1 | cut -c1-200)"
  fi
  # Aux files (TAE decoders, LPIPS): from R2 as the auxiliary/* trees when the plan has them (above), checked here by
  # SHA-256; a file missing or different comes from its pinned URL (an add-only cache volume, an older plan).
  if [ -n "\${FV_WEIGHTS_TREES_B64:-}" ]; then
    t="$(date +%s.%N)"; n=0; got=0
    printf "%s" "$FV_WEIGHTS_TREES_B64" | base64 -d > "$S/aux.tsv"
    while IFS="$(printf '\\t')" read -r kind a b c d; do
      [ "$kind" = aux ] || continue
      n=$((n + 1)); p="$WR/$a"; mkdir -p "$(dirname "$p")"
      if [ -f "$p" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$c" ]; then continue; fi
      if [ -n "$ADD" ] && [ -e "$p" ]; then slsfail "aux $a differs on the cache volume (add-only: not replaced)"; fi
      q="$p.part-\${RUNPOD_POD_ID:-w}"; got=$((got + 1))
      say "aux $a <- $b"
      curl -fsSL --connect-timeout 10 --retry 3 --max-time 600 -o "$q" "$b" && [ "$(sha256sum "$q" | cut -d' ' -f1)" = "$c" ] && [ "$(stat -c %s "$q")" = "$d" ] || slsfail "aux $a"
      if [ -n "$ADD" ] && [ -e "$p" ]; then rm -f "$q"; else mv "$q" "$p"; fi
    done < "$S/aux.tsv"
    step aux "$t" "$n files checked, $got from their pinned URLs"
  fi
  rep serve "weights ready under $WR after $(el "$FV_BOOT_T0F")s: starting fv-serve"
  say "starting fv-serve"
fi
`;

// ---------------------------------------------------------------- cold-start outlook
/** Container start of a scale-from-zero worker with the image on the host (measured +49..+62 s from submit, WP-18/19,
 * docs/serve/images.md §FlashBoot); a host without the image: up to 458 s (about 10 % of cold starts). */
export const SLS_IMAGE_START_S = 60;
export const SLS_IMAGE_PULL_TAIL_S = 458;
/** Model load after start, by variant (measured from the volume: wan 70, h3 52, ltx 33 s; docs/serve/gateway.md §8.6).
 * From a local NVMe container disk: UNVERIFIED (expected no slower). */
export function slsLoadS(variant: string): number {
  if (variant.startsWith("h3")) return 52;
  if (variant.startsWith("ltx")) return 33;
  if (variant.startsWith("wan") || variant === "sfwan") return 70;
  return variant === "cpu" ? 2 : 60;
}

export interface BootRow {
  worker_id: string;
  boot: string;
  boot_at: number;
  cache: string | null;
  missing: string | null;
  fetch_start: number | null;
  fetch_end: number | null;
  fetch_bytes: number | null;
  fetch_s: number | null;
  fetch_mbps: number | null;
  serve_at: number | null;
  error: string | null;
  last_detail: string | null;
  /** The boot's timed steps, JSON {step: seconds} (phase `step` reports: tools, fetcher, fetch, aux). */
  steps?: string | null;
  updated_at: number;
}
export interface Outlook {
  weights_source: "volume" | "r2";
  trees: { dest: string; gb: number; published: boolean }[];
  fetch_gb: number;
  rate_gbps: number;
  rate_source: "planning" | "measured";
  image_start_s: number;
  image_pull_tail_s: number;
  fetch_s: number;
  load_s: number;
  cold_start_s: number;
  cached_cold_start_s: number | null;
  cache: { state: "unknown" | "hit" | "miss"; at: number | null; worker: string | null; note: string };
  last_boot: (BootRow & { breakdown: Record<string, number | null> }) | null;
  note: string;
}

/** The estimated cold start of an endpoint: container start (image pull) + fetch (the manifests' bytes at
 * R2_DOWNLOAD_GBPS, or the last measured rate) + load; with the cache state the last boots reported. */
export function slsOutlook(env: Env, spec: SpecLike, boots: BootRow[] = []): Outlook {
  const r2 = spec.weights_source === "r2" && spec.compute === "GPU";
  const pub = r2Trees(env);
  const trees = slsTrees(spec).map((t) => ({ dest: t, gb: Math.round((slsTreeBytes([t], pub) / 1e9) * 10) / 10, published: !!pub[t] }));
  const fetchGb = r2 ? Math.round(trees.reduce((s, t) => s + t.gb, 0) * 10) / 10 : 0;
  const measured = boots.find((b) => b.fetch_mbps && b.fetch_bytes && b.fetch_bytes > 5e9);
  const rate = measured ? measured.fetch_mbps! / 1000 : R2_DOWNLOAD_GBPS;
  const fetchS = r2 ? Math.round(fetchGb / rate) : 0;
  const loadS = slsLoadS(spec.variant);
  const last = boots[0] ?? null;
  const cacheState: Outlook["cache"] = !r2
    ? { state: "unknown", at: null, worker: null, note: "weights_source volume: no fetch (the volume is mounted)" }
    : last?.cache === "hit"
      ? { state: "hit", at: last.boot_at, worker: last.worker_id, note: `the last boot (${last.worker_id}) found every tree on its disk: no fetch` }
      : last?.cache === "miss"
        ? { state: "miss", at: last.boot_at, worker: last.worker_id, note: `the last boot (${last.worker_id}) fetched${last.missing ? ` ${last.missing}` : ""}` }
        : { state: "unknown", at: null, worker: null, note: "no boot reported yet: a new worker fetches every tree" };
  // A host whose driver is too old for the image (driver-check.ts): the boot stopped before the fetch.
  if (last?.error?.startsWith("driver_too_old")) cacheState.note += `; the last boot's host was refused: ${last.error} (allowed_cuda filters such hosts out)`;
  const breakdown = (b: BootRow) => ({
    fetch_s: b.fetch_s ?? (b.fetch_start && b.fetch_end ? Math.round((b.fetch_end - b.fetch_start) / 1000) : null),
    boot_to_serve_s: b.serve_at ? Math.round((b.serve_at - b.boot_at) / 1000) : null,
    mbps: b.fetch_mbps,
    ...Object.fromEntries(Object.entries(parseJson<Record<string, number>>(b.steps ?? null, {}) || {}).map(([k, v]) => [`${k}_s`, v])),
  });
  const cold = SLS_IMAGE_START_S + fetchS + loadS;
  const m = (s: number) => (s >= 90 ? `${Math.round(s / 60)} min` : `${s} s`);
  const note = r2
    ? `cold start ~${m(cold)}: start ~${SLS_IMAGE_START_S} s (image on the host; up to ${SLS_IMAGE_PULL_TAIL_S} s without) + fetch ${fetchGb} GB at ${rate.toFixed(2)} GB/s${measured ? " (measured)" : " (planning rate: the 2026-10-12 median)"} ~${m(fetchS)} + load ~${loadS} s; with the trees on the worker's disk ~${m(SLS_IMAGE_START_S + loadS)}`
    : `cold start ~${m(SLS_IMAGE_START_S + loadS)}: start ~${SLS_IMAGE_START_S} s (up to ${SLS_IMAGE_PULL_TAIL_S} s on a host without the image) + load from the volume ~${loadS} s`;
  return {
    weights_source: r2 ? "r2" : "volume",
    trees,
    fetch_gb: fetchGb,
    rate_gbps: rate,
    rate_source: measured ? "measured" : "planning",
    image_start_s: SLS_IMAGE_START_S,
    image_pull_tail_s: SLS_IMAGE_PULL_TAIL_S,
    fetch_s: fetchS,
    load_s: loadS,
    cold_start_s: r2 ? cold : SLS_IMAGE_START_S + loadS,
    cached_cold_start_s: r2 ? SLS_IMAGE_START_S + loadS : null,
    cache: cacheState,
    last_boot: last ? { ...last, breakdown: breakdown(last) } : null,
    note,
  };
}

// ---------------------------------------------------------------- boot reports (POST /ingest/v1/sls-report)
const num = (s: string | undefined) => (s === undefined ? null : Number(s));
/** Folds one report into the boot's row: the phase and the fetcher's progress / summary lines (r2-fetch.py). */
export function applyReport(prev: Partial<BootRow>, phase: string, detail: string, at: number): Partial<BootRow> {
  const r: Partial<BootRow> = { ...prev, last_detail: detail.slice(0, 300), updated_at: at };
  if (phase === "cache") {
    r.cache = /^hit/.test(detail) ? "hit" : "miss";
    if (r.cache === "miss") {
      r.missing = detail.replace(/^miss:\s*/, "").trim().slice(0, 300) || null;
      r.fetch_start = r.fetch_start ?? at;
    }
  } else if (phase === "weights") {
    const done = /weights ready: .*?([\d.]+) GB fetched in ([\d.]+)s \((\d+) MB\/s\)/.exec(detail);
    const prog = /^r2 ([\d.]+)\/([\d.]+) GB, (\d+) MB\/s/.exec(detail);
    if (done) Object.assign(r, { fetch_end: at, fetch_bytes: num(done[1])! * 1e9, fetch_s: num(done[2]), fetch_mbps: num(done[3]) });
    else if (prog) Object.assign(r, { fetch_start: r.fetch_start ?? at, fetch_bytes: num(prog[1])! * 1e9, fetch_mbps: num(prog[3]) });
    else if (/failed/.test(detail)) r.error = detail.slice(0, 300);
  } else if (phase === "step") {
    // "<step> <seconds>s (<note>)": the boot's own timing of one step (SLS_R2_SH step()).
    const m = /^([a-z][a-z0-9_-]{0,31}) ([\d.]+)s\b/.exec(detail);
    if (m) {
      const st = parseJson<Record<string, number>>(prev.steps ?? null, {}) || {};
      st[m[1]!] = Number(m[2]);
      r.steps = JSON.stringify(st).slice(0, 1000);
    }
  } else if (phase === "serve") r.serve_at = at;
  else if (phase === "failed") r.error = detail.slice(0, 300);
  return r;
}
