// GPU providers besides Runpod (docs/serve/deploy-gmi-brev.md §6): GMI Cloud
// containers and NVIDIA Brev VMs behind one small interface, so a pool (and a
// standalone pod, which is a one-pool cluster) with `provider: gmi | brev`
// goes through the same `up` / `down`, price check, deadline and ledger as a
// Runpod pool. Runpod keeps its own module (runpod.ts) and stays the default:
// nothing here runs for a pool without `provider` or with `provider: runpod`.
//
// A pod of these providers is keyed `<provider>:<name>` (gmi:fv-pod-x-…): the
// name is fv-control's (unique, fv-pod-* / fv-ctl-*) and known before the
// create call, so the pod's env carries its own id (log shipping, the edge's
// worker id); the provider's own id is looked up by name when needed. Neither
// provider has labels, so "ours" is a name fv-control made AND recorded in
// cluster_pods; nothing else on those accounts is ever touched.
//
// What differs from a Runpod pod (all in the env and the start command):
//  - no network volume: weights come from the Hub at pinned revisions at
//    boot (weights_source hub, owner approval) or not at all (fake engine);
//  - no HTTPS proxy: the pod opens a Cloudflare quick tunnel and reports its
//    https URL to POST /ingest/v1/endpoint (index.ts) with the cluster's
//    ingest token; edge fronts export it as FV_DISPATCH_ENDPOINT;
//  - no balance API: a monthly budget per provider (GMI_BUDGET_USD,
//    BREV_BUDGET_USD) instead of the balance floor;
//  - Runpod secret references do not resolve there: FV_PROVIDER_SECRET_ENV.
import { brev, brevInstanceTypes, brevOff, brevPriceLive } from "./brev";
import { brevDiskGb, brevRunScript, brevStartup, markBrevDeleted, newBootToken, parkedDph, recordBrevCreate, releaseBrev, restartDiag, type Release, type TreeRevs } from "./brev-park";
import { BAKED_CONFIGS } from "./cluster/image-configs";
import type { PoolSpec } from "./cluster/spec";
import { WORKER_CONFIGS } from "./cluster/worker-configs";
import { sha256Hex } from "./crypto";
import { OTHER_PROVIDERS, type OtherProviderId, type ProviderId } from "./enums";
import type { Env } from "./env";
import { FV_NAME_RE, gmi, gmiDefaultIdc, gmiEnabled, gmiProducts } from "./gmi";
import { POOL_PRESETS } from "./presets";
import { HttpError, now, parseJson, putSetting, rateLimit, utcDay } from "./util";
import { AUX_FILES, HUB_TREES } from "./weights-sources";

// ---------------------------------------------------------------- ids
const KEY_RE = /^(gmi|brev):(.+)$/;
/** A pod id of a GMI / Brev pod (`gmi:<name>`); Runpod ids have no prefix. */
export const isOtherPod = (podId: string) => KEY_RE.test(podId);
export function splitKey(podId: string): { provider: OtherProviderId; name: string } | null {
  const m = KEY_RE.exec(podId);
  return m ? { provider: m[1] as OtherProviderId, name: m[2]! } : null;
}
export const podKey = (p: OtherProviderId, name: string) => `${p}:${name}`;
export const poolProvider = (p: { provider?: ProviderId }): ProviderId => p.provider ?? "runpod";
export const isOtherPool = (p: { provider?: ProviderId }) => poolProvider(p) !== "runpod";

// ---------------------------------------------------------------- the interface
export type InstanceState = "starting" | "running" | "stopped" | "failed";
export interface ProviderInstance {
  key: string;
  provider: OtherProviderId;
  /** The provider's own id (GMI uuid, Brev workspace id). */
  id: string;
  name: string;
  state: InstanceState;
  /** The provider's own status word. */
  raw: string;
  reason: string | null;
  gpu: string | null;
  createdAt: number | null;
}
export interface Offer {
  gpu: string;
  region: string | null;
  usd_per_hr: number | null;
  in_stock: boolean | null;
  /** Brev: the type can be stopped (keep-on-stop, warm restarts); null: unknown. */
  stoppable?: boolean | null;
  /** Brev: the list's estimated deploy time (s). */
  deploy_s?: number | null;
  /** Brev: storage $/GB-hr (what a parked instance costs). */
  storage_usd_per_gb_hr?: number | null;
}
export interface ProviderLaunch {
  name: string;
  image: string;
  gpu: string;
  region?: string;
  env: Record<string, string>;
  /** The pod's weights (Brev: the disk size and the trees recorded for keep-on-stop). */
  weights?: WeightsPlan;
  /** The cluster it serves (Brev's record). */
  clusterId?: string;
}
export interface ComputeProvider {
  id: OtherProviderId;
  title: string;
  /** Why it is off (a secret unset), or null. */
  off(env: Env): string | null;
  /** The GPU products / instance types pods may use (owner config). */
  gpus(env: Env): string[];
  /** Price and stock of one GPU product for the planner (null fields: unknown). */
  offer(env: Env, gpu: string, region?: string): Promise<Offer>;
  /** fv-named instances of the account (only those). */
  list(env: Env): Promise<ProviderInstance[]>;
  create(env: Env, req: ProviderLaunch): Promise<{ id: string }>;
  remove(env: Env, inst: ProviderInstance): Promise<boolean>;
  logs?(env: Env, inst: ProviderInstance): Promise<string[]>;
  /** The monthly budget in $ (null: unset, launches are refused). */
  budget(env: Env): number | null;
}

const num = (s: string | undefined): number | null => {
  const v = Number(s);
  return s !== undefined && s !== "" && Number.isFinite(v) && v >= 0 ? v : null;
};
const ms = (s: string | null) => (s && Number.isFinite(Date.parse(s)) ? Date.parse(s) : null);

const GMI: ComputeProvider = {
  id: "gmi",
  title: "GMI Cloud",
  off: (env) => (!gmiEnabled(env) ? "GMI_API_KEY is not set" : !gmiProducts(env).length ? "GMI_PRODUCTS is not set (the product ids GMI gave the account)" : null),
  gpus: gmiProducts,
  async offer(env, gpu, region) {
    const idc = region || gmiDefaultIdc(env);
    const p = (await gmi.products(env, idc)).find((x) => x.name === gpu);
    return { gpu, region: idc, usd_per_hr: p?.usd_per_hr ?? null, in_stock: p ? p.valid : false };
  },
  async list(env) {
    return (await gmi.containers(env))
      .filter((c) => FV_NAME_RE.test(c.name))
      .map((c) => ({
        key: podKey("gmi", c.name),
        provider: "gmi" as const,
        id: c.id,
        name: c.name,
        state: c.status === "running" ? "running" : c.status === "error" || c.status === "zombie" ? "failed" : c.status === "stopped" || c.status === "terminating" ? "stopped" : "starting",
        raw: c.status,
        reason: c.reason,
        gpu: c.product,
        createdAt: ms(c.createdAt),
      }));
  },
  async create(env, req) {
    const templateId = await gmi.ensureTemplate(env, req.image, (await sha256Hex(req.image)).slice(0, 12));
    const id = await gmi.create(env, { name: req.name, product: req.gpu, idc: req.region || gmiDefaultIdc(env), templateId, command: "bash", args: ["-c", PROVIDER_BOOT], envs: req.env, ports: [8000] });
    return { id };
  },
  remove: (env, inst) => gmi.remove(env, inst.id),
  logs: (env, inst) => gmi.logs(env, inst.id),
  budget: (env) => num(env.GMI_BUDGET_USD),
};

const BREV: ComputeProvider = {
  id: "brev",
  title: "NVIDIA Brev",
  off: (env) => brevOff(env) ?? (!brevInstanceTypes(env).length ? "BREV_INSTANCE_TYPES is not set (allowed instance types)" : null),
  gpus: brevInstanceTypes,
  async offer(env, gpu) {
    // The live instance-type list (docs/serve/deploy-gmi-brev.md §3.2; cached 1 h), the owner's BREV_PRICES overriding a price.
    const { usd_per_hr, info } = await brevPriceLive(env, gpu);
    return { gpu, region: info?.location ?? null, usd_per_hr, in_stock: info?.available ?? null, stoppable: info ? info.stoppable : null, deploy_s: info?.deploy_s ?? null, storage_usd_per_gb_hr: info?.storage_usd_per_gb_hr ?? null };
  },
  async list(env) {
    return (await brev.workspaces(env))
      .filter((w) => FV_NAME_RE.test(w.name))
      .map((w) => ({
        key: podKey("brev", w.name),
        provider: "brev" as const,
        id: w.id,
        name: w.name,
        state: w.status === "RUNNING" ? "running" : w.status === "FAILURE" ? "failed" : ["STOPPED", "STOPPING", "DELETING"].includes(w.status) ? "stopped" : "starting",
        raw: w.status,
        reason: w.health,
        gpu: w.instanceType,
        createdAt: ms(w.createdAt),
      }));
  },
  async create(env, req) {
    // The startup script only installs the per-boot bootstrap; the launch itself (image + env) is the run script it
    // fetches from fv-control (brev-park.ts), so a parked VM restarted later runs that later launch.
    const run = brevRun(req.image, req.env);
    const info = await brev.type(env, req.gpu);
    const trees = req.weights?.source === "hub" ? req.weights.trees : [];
    const disk = brevDiskGb(trees);
    const token = newBootToken();
    const id = await brev.create(env, { name: req.name, instanceType: req.gpu, startupScript: brevStartup(`${(env.PUBLIC_URL || "").replace(/\/$/, "")}/ingest/v1/brev-boot`, token), diskStorage: `${disk}Gi` });
    await recordBrevCreate(env, {
      workspace_id: id,
      pod_id: podKey("brev", req.name),
      name: req.name,
      instance_type: req.gpu,
      info,
      disk_gb: info && !info.elastic_disk && info.fixed_disk_gb ? info.fixed_disk_gb : disk,
      launch_trees: launchTrees(req.weights),
      cluster_id: req.clusterId ?? null,
      token,
      run,
    });
    return { id };
  },
  async remove(env, inst) {
    const ok = await brev.remove(env, inst.id);
    if (ok) await markBrevDeleted(env, inst.id);
    return ok;
  },
  budget: (env) => num(env.BREV_BUDGET_USD),
};

export const PROVIDER_IMPLS: Record<OtherProviderId, ComputeProvider> = { gmi: GMI, brev: BREV };
export const providerImpl = (p: OtherProviderId) => PROVIDER_IMPLS[p];
export const providerEnabled = (env: Env, p: OtherProviderId) => !providerImpl(p).off(env);

/** One provider's state for /api/providers and the UI (enabled, why not, budget, this month's spend). */
export async function providerView(env: Env, p: OtherProviderId) {
  const impl = providerImpl(p);
  const off = impl.off(env);
  const spend = await providerSpend(env, p);
  return {
    id: p,
    title: impl.title,
    enabled: !off,
    reason: off ? `${impl.title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)` : null,
    gpus: impl.gpus(env),
    budget_usd: impl.budget(env),
    month_usd: spend.month,
    running_dph: spend.running_dph,
    parked_dph: spend.parked_dph,
    hub_downloads_approved: env.FV_HUB_DOWNLOADS_APPROVED === "1",
  };
}

// ---------------------------------------------------------------- money
/** This UTC month's ledger for a provider's pods (cost_daily) and the $/hr of its running ones (pods table). */
export async function providerSpend(env: Env, p: OtherProviderId): Promise<{ month: number; running_dph: number; parked_dph: number }> {
  const first = utcDay(now()).slice(0, 8) + "01";
  const m = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd FROM cost_daily WHERE day >= ? AND pod_id LIKE ?").bind(first, `${p}:%`).first<{ usd: number }>();
  const r = await env.DB.prepare("SELECT COALESCE(SUM(cost_per_hr), 0) AS dph FROM pods WHERE gone_at IS NULL AND provider = ? AND desired_status = 'RUNNING'").bind(p).first<{ dph: number }>();
  // Brev keep-on-stop: parked instances bill their disk (brev-park.ts).
  const parked = p === "brev" ? await parkedDph(env) : 0;
  return { month: Number(m?.usd || 0), running_dph: Number(r?.dph || 0), parked_dph: parked };
}

export interface BudgetCheck {
  provider: OtherProviderId;
  ok: boolean;
  reasons: string[];
  budget: number | null;
  month: number;
  running_dph: number;
  /** Brev: the storage $/hr of parked instances. */
  parked_dph: number;
  add_dph: number;
  hours: number;
  projected: number;
}
/** No balance API on GMI / Brev: this month's spend + the running pods, the parked instances' storage and `addDph`
 * until the deadline must stay within the budget. */
export async function budgetCheck(env: Env, p: OtherProviderId, addDph: number, hours: number): Promise<BudgetCheck> {
  const impl = providerImpl(p);
  const budget = impl.budget(env);
  const s = await providerSpend(env, p);
  const dph = s.running_dph + s.parked_dph + addDph;
  const projected = s.month + dph * hours;
  const reasons: string[] = [];
  if (budget === null) reasons.push(`${impl.title} has no balance API: set ${p.toUpperCase()}_BUDGET_USD (the most fv-control may spend there per month) before launching`);
  else if (projected > budget)
    reasons.push(`${impl.title}: $${s.month.toFixed(2)} spent this month + $${dph.toFixed(2)}/hr${s.parked_dph ? ` (parked storage $${s.parked_dph.toFixed(3)}/hr included)` : ""} for ${hours.toFixed(2)} h = $${projected.toFixed(2)}, over the budget $${budget}`);
  return { provider: p, ok: !reasons.length, reasons, budget, month: s.month, running_dph: s.running_dph, parked_dph: s.parked_dph, add_dph: addDph, hours, projected };
}

/** The $/hr a pool's pod will cost on its provider: the offer's price, else the cap (the most it may cost). */
export async function poolDph(env: Env, pool: PoolSpec, cap: number): Promise<{ dph: number; known: boolean; offer: Offer | null }> {
  const p = poolProvider(pool) as OtherProviderId;
  const offer = await providerImpl(p)
    .offer(env, pool.provider_gpu!, pool.provider_region)
    .catch(() => null);
  return offer?.usd_per_hr != null ? { dph: offer.usd_per_hr, known: true, offer } : { dph: cap, known: false, offer };
}

// ---------------------------------------------------------------- weights
const configText = (pool: PoolSpec): string => {
  if (pool.config_toml) return pool.config_toml;
  const f = (pool.config || "").split("/").pop() || "";
  return BAKED_CONFIGS[f] || WORKER_CONFIGS[f] || "";
};
export interface WeightsPlan {
  source: "hub" | "none";
  trees: string[];
  /** Trees no plain Hub fetch can make (several repos, a conversion): the launch is refused with them. */
  unsupported: string[];
  /** TSV for FV_WEIGHTS_TREES_B64: tree<TAB>dest<TAB>repo<TAB>rev<TAB>globs, aux<TAB>path<TAB>url<TAB>sha256<TAB>size. */
  tsv: string;
}
/** The weight trees a pool loads: `${FV_WEIGHTS}/<tree>` in its worker config, plus its preset's list. */
export function poolTrees(pool: PoolSpec): string[] {
  const out = new Set<string>();
  for (const line of configText(pool).split("\n")) {
    if (/^\s*#/.test(line)) continue;
    for (const m of line.matchAll(/\$\{FV_WEIGHTS\}\/([A-Za-z0-9._-]+)(?:\/([A-Za-z0-9._-]+))?/g)) out.add(m[2] && HUB_TREES[`${m[1]}/${m[2]}`] ? `${m[1]}/${m[2]}` : m[1]!);
  }
  for (const p of POOL_PRESETS) if (p.pool.variant === pool.variant && ((pool.config && p.pool.config === pool.config) || (pool.config_toml && p.pool.config_toml === pool.config_toml))) p.weights.forEach((w) => out.add(w));
  return [...out].sort();
}
export function weightsPlan(pool: PoolSpec): WeightsPlan {
  const source = pool.weights_source === "hub" ? "hub" : "none";
  if (source === "none") return { source, trees: [], unsupported: [], tsv: "" };
  const trees = poolTrees(pool).filter((t) => t !== "auxiliary");
  const unsupported = trees.filter((t) => !HUB_TREES[t]);
  const rows = trees.filter((t) => HUB_TREES[t]).map((t) => ["tree", t, HUB_TREES[t]!.repo, HUB_TREES[t]!.revision, HUB_TREES[t]!.globs.join(" ")].join("\t"));
  // The TAEs (and LPIPS) every engine may load: small, pinned by SHA-256.
  for (const a of AUX_FILES) rows.push(["aux", a.path, a.url, a.sha256, String(a.size)].join("\t"));
  return { source, trees, unsupported, tsv: rows.join("\n") + "\n" };
}
/** The Hub trees of a plan at their pinned revisions ({} without hub): what a Brev disk keeps. */
export function launchTrees(plan: WeightsPlan | undefined): TreeRevs {
  if (!plan || plan.source !== "hub") return {};
  return Object.fromEntries(plan.trees.filter((t) => HUB_TREES[t]).map((t) => [t, HUB_TREES[t]!.revision]));
}

// ---------------------------------------------------------------- the pod's env and start command
/** Runpod secret references a GMI / Brev pod cannot resolve, replaced by FV_PROVIDER_SECRET_ENV's values or left out. */
export function providerEnv(
  env: Env,
  full: Record<string, string>,
  o: { provider: OtherProviderId; key: string; name: string; deadlineMs: number; reportToken: string; weights: WeightsPlan; scriptsRef: string },
): { env: Record<string, string>; dropped: string[] } {
  const out: Record<string, string> = {};
  const dropped: string[] = [];
  const subst = parseJson<Record<string, unknown>>(env.FV_PROVIDER_SECRET_ENV, {});
  for (const [k, v] of Object.entries(full)) {
    if (/\{\{\s*RUNPOD_SECRET_/.test(v)) {
      if (typeof subst[k] === "string" && subst[k]) out[k] = subst[k] as string;
      else dropped.push(k);
      continue;
    }
    out[k] = v;
  }
  // The Runpod key and floor of the Runpod watchdog never leave Runpod.
  delete out.FV_BACKSTOP_API_KEY;
  delete out.FV_MIN_BALANCE;
  Object.assign(out, {
    FV_PROVIDER: o.provider,
    FV_POD_ID: o.key,
    FV_POD_NAME: o.name,
    FV_WORKER_ID: o.key,
    FV_LOG_SHIP_POD: o.key,
    FV_WEIGHTS: "/workspace/weights",
    FV_WEIGHTS_SOURCE: o.weights.source,
    FV_CLUSTER_DEADLINE: String(Math.floor(o.deadlineMs / 1000)),
    FV_ENDPOINT_REPORT_URL: `${(env.PUBLIC_URL || "").replace(/\/$/, "")}/ingest/v1/endpoint`,
    FV_ENDPOINT_REPORT_TOKEN: o.reportToken,
  });
  if (o.weights.source === "hub") {
    out.FV_WEIGHTS_TREES_B64 = btoa(o.weights.tsv);
    out.FV_SCRIPTS_URL = `https://raw.githubusercontent.com/${env.GITHUB_REPO || "zaitrarrio/fastvideo-rs"}/${o.scriptsRef}/scripts/gpu`;
  }
  // GMI: the in-container watchdog deletes its own container at the deadline (fv-control's cron does too).
  if (o.provider === "gmi" && env.GMI_API_KEY) Object.assign(out, { FV_BACKSTOP_API_KEY: env.GMI_API_KEY, FV_BACKSTOP_API: (env.GMI_API || "https://console.gmicloud.ai/api").replace(/\/$/, "") });
  return { env: out, dropped };
}

/** cloudflared, pinned (2026.9.0, linux-amd64) and checked by SHA-256 before it runs. */
export const CLOUDFLARED = {
  url: "https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64",
  sha256: "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134",
};

// The weight trees of FV_WEIGHTS_TREES_B64 (weightsPlan), in the boot below. A
// tree counts as present only with fetch-hub-tree.py's `.complete` AND our
// `.fv-revision` holding its pinned revision (written after the fetch): a Brev
// VM restarted from a stop keeps /home/ubuntu/workspace/weights, so those
// trees are skipped (warm restart); one at another revision, or a leftover
// without both marks, is moved aside and removed (the VM's own disk, never a
// shared volume), then fetched again. `fetch_tree` is the boot's (tests stub
// it). The Hub fetcher is only set up when a tree is missing.
export const WEIGHTS_SH = `if [ "\${FV_WEIGHTS_SOURCE:-none}" = hub ]; then
  say "weights: Hub trees at pinned revisions (owner-approved)"
  rm -rf "$FV_WEIGHTS"/.*.partial-* "$FV_WEIGHTS"/.stale-* 2>/dev/null || true
  report weights "" "checking"
  n=0; kept=0; got=0
  while IFS="$(printf '\\t')" read -r kind a b c d; do
    n=$((n + 1))
    case "$kind" in
      tree)
        if [ -f "$FV_WEIGHTS/$a/.complete" ] && [ "$(cat "$FV_WEIGHTS/$a/.fv-revision" 2>/dev/null)" = "$c" ]; then say "tree $a: present at $c"; kept=$((kept + 1)); continue; fi
        if [ -e "$FV_WEIGHTS/$a" ]; then
          old="$FV_WEIGHTS/.stale-$(printf %s "$a" | tr '/' '_')-$(date +%s)"
          say "tree $a: not complete at $c: replaced"
          mv "$FV_WEIGHTS/$a" "$old" && rm -rf "$old" || fail "tree $a: cannot move the old copy aside"
        fi
        say "tree $a <- $b@$c"
        report weights "" "downloading $a"
        fetch_tree "$a" "$b" "$c" "$d" "$n"
        printf %s "$c" > "$FV_WEIGHTS/$a/.fv-revision" || fail "tree $a: revision mark"
        got=$((got + 1))
        ;;
      aux)
        p="$FV_WEIGHTS/$a"; mkdir -p "$(dirname "$p")"
        if [ -f "$p" ] && [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$c" ]; then continue; fi
        curl -fsSL --max-time 600 -o "$p.part" "$b" && [ "$(sha256sum "$p.part" | cut -d' ' -f1)" = "$c" ] && [ "$(stat -c %s "$p.part")" = "$d" ] && mv "$p.part" "$p" || fail "aux $a"
        ;;
    esac
  done <<EOF_TREES
$(printf "%s" "$FV_WEIGHTS_TREES_B64" | base64 -d)
EOF_TREES
  say "weights ready: $kept present, $got fetched"
fi
`;

// The start command of a GMI / Brev worker (inside our image, as root): the
// weights (hub), the tunnel and its report, the deadline watchdog, then
// fv-serve with the same config handling as WORKER_BOOT (payloads.ts).
export const PROVIDER_BOOT = `set -u
say() { echo "[fv-boot] $*" >&2; }
echo "[fv-boot] start ($FV_PROVIDER $FV_POD_ID)" >&2
mkdir -p /fvstate "$FV_WEIGHTS"
report() {
  [ -n "\${FV_ENDPOINT_REPORT_URL:-}" ] || return 0
  for _ in 1 2 3 4 5; do
    curl -sS --max-time 20 -X POST -H "Authorization: Bearer $FV_ENDPOINT_REPORT_TOKEN" -H "content-type: application/json" \\
      -d "{\\"pod\\":\\"$FV_POD_ID\\",\\"phase\\":\\"$1\\",\\"url\\":\\"$2\\",\\"detail\\":\\"$3\\"}" "$FV_ENDPOINT_REPORT_URL" >/dev/null && return 0
    sleep 5
  done
}
fail() { say "FAILED: $1"; report failed "" "$1"; sleep 600; exit 1; }
if ! command -v curl >/dev/null 2>&1; then
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl ca-certificates; } >/fvstate/apt.log 2>&1 || fail "no curl"
fi
# ---- weights: the Hub at pinned revisions (fetch-hub-tree.py checks every file against the Hub listing), aux files by SHA-256
FETCHER=""
fetch_tree() { # dest repo revision globs n: one tree into $FV_WEIGHTS/<dest> (the fetcher is set up on first use)
  if [ -z "$FETCHER" ]; then
    { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-venv python3-pip; } >>/fvstate/apt.log 2>&1 || fail "apt python3-venv"
    python3 -m venv /opt/fv/venv && /opt/fv/venv/bin/pip install -q "huggingface_hub>=0.34" >/fvstate/pip.log 2>&1 || fail "pip huggingface_hub"
    mkdir -p /opt/fv/scripts && curl -fsSL --max-time 60 -o /opt/fv/scripts/fetch-hub-tree.py "$FV_SCRIPTS_URL/fetch-hub-tree.py" || fail "fetch-hub-tree.py"
    FETCHER=1
  fi
  mkdir -p "/fvstate/fetch-$5"
  FETCH_REPO="$2" FETCH_REVISION="$3" FETCH_DEST="$1" FETCH_GLOBS="$4" FETCH_WEIGHTS="$FV_WEIGHTS" FETCH_SRV="/fvstate/fetch-$5" FETCH_MIN_FREE_GB=5 \\
    /opt/fv/venv/bin/python /opt/fv/scripts/fetch-hub-tree.py >"/fvstate/fetch-$5.out" 2>&1 || fail "tree $1: $(tail -2 "/fvstate/fetch-$5/log.txt" 2>/dev/null | tr '\\n"' '  ')"
}
${WEIGHTS_SH}# ---- the tunnel: https://<random>.trycloudflare.com -> 127.0.0.1:8000 (outbound only; no inbound port needed)
curl -fsSL --max-time 120 -o /usr/local/bin/cloudflared "${CLOUDFLARED.url}" || fail "cloudflared download"
echo "${CLOUDFLARED.sha256}  /usr/local/bin/cloudflared" | sha256sum -c - >/dev/null 2>&1 || fail "cloudflared sha256"
chmod +x /usr/local/bin/cloudflared
/usr/local/bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >/fvstate/tunnel.log 2>&1 &
U=""
for _ in $(seq 1 90); do
  U="$(grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' /fvstate/tunnel.log | head -1)"
  [ -n "$U" ] && break
  sleep 2
done
[ -n "$U" ] || fail "no tunnel URL: $(tail -3 /fvstate/tunnel.log | tr '\\n"' '  ')"
say "tunnel $U"
[ "\${FV_DISPATCH_FRONT:-0}" = 1 ] && export FV_DISPATCH_ENDPOINT="$U"
report tunnel "$U" ""
# ---- the deadline watchdog (fv-control's cron deletes the pod at the deadline as well)
(
  while :; do
    if [ "$(date +%s)" -ge "$FV_CLUSTER_DEADLINE" ]; then
      say "[watchdog] deadline: $FV_POD_ID"
      if [ "$FV_PROVIDER" = gmi ] && [ -n "\${FV_BACKSTOP_API_KEY:-}" ]; then
        id="$(curl -sS --max-time 30 -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$FV_BACKSTOP_API/v1/containers" | tr '{' '\\n' | grep "\\"name\\":\\"$FV_POD_NAME\\"" | sed -n 's/.*"id":"\\([0-9a-fA-F-]*\\)".*/\\1/p' | head -1)"
        [ -n "$id" ] && curl -sS --max-time 30 -X DELETE -H "Authorization: Bearer $FV_BACKSTOP_API_KEY" "$FV_BACKSTOP_API/v1/containers/$id" >/dev/null
      fi
      pkill -f fv-serve
      sleep 60
    fi
    sleep 30
  done
) &
# ---- fv-serve (the config as WORKER_BOOT does it)
if [ -n "\${FV_WORKER_TOML_B64:-}" ]; then
  printf "%s" "$FV_WORKER_TOML_B64" | base64 -d > /fv-worker.toml
else
  cp "$FV_WORKER_CONFIG" /fv-worker.toml
fi
if grep -q "^\\[gateway\\]" /fv-worker.toml; then
  sed -i "/^\\[gateway\\]/a register = false" /fv-worker.toml
else
  printf "\\n[gateway]\\nregister = false\\n" >> /fv-worker.toml
fi
export FV_WORKER_ID="$FV_POD_ID"
exec /opt/fastvideo-rs/bin/fv-serve --config /fv-worker.toml`;

/** A Brev VM's run script for one launch (brev-park.ts): our image running PROVIDER_BOOT with the pod's env. */
export const brevRun = (image: string, env: Record<string, string>) => brevRunScript(image, env, PROVIDER_BOOT);

// ---------------------------------------------------------------- lifecycle (cluster/ops.ts and the DO call these)
async function findInstance(env: Env, podId: string): Promise<ProviderInstance | null> {
  const k = splitKey(podId);
  if (!k) return null;
  return (await providerImpl(k.provider).list(env)).find((i) => i.name === k.name) || null;
}
/** Deletes a GMI / Brev pod by its key; true when it is gone (or was). Only an fv-named instance can match. */
export async function deleteOtherPod(env: Env, podId: string): Promise<boolean> {
  const inst = await findInstance(env, podId);
  if (!inst) return true;
  return providerImpl(inst.provider).remove(env, inst);
}
/** Releases a GMI / Brev pod: with `park`, a Brev pod of a stoppable type whose weights completed is stopped and
 * parked (brev-park.ts); a warm restart that never came up is held; everything else is deleted. */
export async function releaseOtherPod(env: Env, podId: string, park: boolean, log?: (m: string) => void): Promise<Release | "deleted"> {
  if (splitKey(podId)?.provider === "brev") {
    const r = await releaseBrev(env, podId, park, log);
    if (r !== "none") return r;
  }
  if (!(await deleteOtherPod(env, podId))) throw new Error("the provider did not confirm the delete");
  return "deleted";
}
/** A GMI / Brev pod's state for the boot watch; null when the provider no longer lists it. */
export async function otherPodState(env: Env, podId: string): Promise<{ desiredStatus: string; uptimeS: number | null; state: InstanceState; reason: string | null } | null> {
  const inst = await findInstance(env, podId);
  if (!inst) return null;
  return {
    desiredStatus: inst.state === "running" ? "RUNNING" : inst.state.toUpperCase(),
    uptimeS: inst.createdAt ? Math.max(0, Math.round((now() - inst.createdAt) / 1000)) : null,
    state: inst.state,
    reason: inst.reason,
  };
}
export async function otherPodLogs(env: Env, podId: string): Promise<string[]> {
  const inst = await findInstance(env, podId);
  const impl = inst ? providerImpl(inst.provider) : null;
  if (!inst || !impl?.logs) return [];
  return impl.logs(env, inst);
}
/** The https URL a GMI / Brev pod reported (its tunnel); null until it has. */
export async function reportedUrl(env: Env, podId: string): Promise<string | null> {
  const r = await env.DB.prepare("SELECT url FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ url: string | null }>();
  return r?.url || null;
}
/** Where a pod answers: Runpod's proxy, or the tunnel a GMI / Brev pod reported (null until then). */
export async function podBaseUrl(env: Env, podId: string, runpodUrl: (pod: string) => string): Promise<string | null> {
  return isOtherPod(podId) ? reportedUrl(env, podId) : runpodUrl(podId);
}

/** What a GMI / Brev pod reported last (POST /ingest/v1/endpoint): failed boots show here. */
export async function lastReport(env: Env, podId: string): Promise<{ phase: string; detail: string; at: number } | null> {
  const r = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(`endpoint:${podId}`).first<{ value: string }>();
  return r ? parseJson(r.value, null) : null;
}

/** A starting GMI / Brev pod's boot diagnosis (the `up` wait): the provider's state and the pod's own last report. */
export async function otherPodDiag(env: Env, podId: string, rt: { state: InstanceState; reason: string | null } | undefined): Promise<{ phase: string; detail: string; fatal: boolean; replace?: boolean } | null> {
  const rep = await lastReport(env, podId);
  if (rep?.phase === "failed") return { phase: "boot failed", detail: rep.detail || "the pod reported a failure", fatal: true };
  if (!rt) return null;
  // A parked Brev instance being started again: stopped / starting is expected until BREV_RESTART_TIMEOUT_S.
  if (splitKey(podId)?.provider === "brev") {
    const w = await restartDiag(env, podId, rt.state);
    if (w) return w;
  }
  if (rt.state === "failed") return { phase: "provider error", detail: rt.reason || "the provider reports an error", fatal: true };
  if (rt.state === "stopped") return { phase: "stopped", detail: "stopped outside fv-control", fatal: true };
  if (rt.state === "starting") return { phase: "provisioning", detail: rt.reason || "", fatal: false };
  if (rep?.phase === "weights") return { phase: "weights", detail: "downloading from the Hub", fatal: false };
  if (rep?.phase === "tunnel") return { phase: "loading", detail: "tunnel up, fv-serve loading", fatal: false };
  return { phase: "booting", detail: "no report from the pod yet", fatal: false };
}

/** The URL a pod reports must be a quick-tunnel URL (tests: FV_ENDPOINT_URL_RE). */
export function endpointUrlOk(env: Env & { FV_ENDPOINT_URL_RE?: string }, url: string): boolean {
  const re = env.FV_ENDPOINT_URL_RE ? new RegExp(env.FV_ENDPOINT_URL_RE) : /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/;
  return url.length <= 300 && re.test(url);
}

/** POST /ingest/v1/endpoint: {pod, phase: weights | tunnel | failed, url?, detail?} from a GMI / Brev pod with its
 * cluster's ingest token. `tunnel` sets the pod's URL (a quick-tunnel URL only); `failed` fails its boot (the `up` wait). */
export async function endpointReport(env: Env, req: Request): Promise<{ ok: true; pod: string; phase: string }> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "ingest token required");
  const cl = await env.DB.prepare("SELECT id FROM clusters WHERE ingest_hash = ?").bind(await sha256Hex(auth.slice(7).trim())).first<{ id: string }>();
  if (!cl) throw new HttpError(401, "invalid ingest token");
  if (!(await rateLimit(env, `endpoint:${cl.id}`, 60, 60))) throw new HttpError(429, "endpoint report rate limit (60/min per cluster)");
  const text = await req.text();
  if (text.length > 4096) throw new HttpError(413, "report too large");
  const b = parseJson<any>(text, null);
  const pod = typeof b?.pod === "string" ? b.pod : "";
  const phase = typeof b?.phase === "string" ? b.phase : "";
  if (!isOtherPod(pod)) throw new HttpError(400, "pod: a GMI / Brev pod id");
  if (!["weights", "tunnel", "failed"].includes(phase)) throw new HttpError(400, "phase: weights | tunnel | failed");
  const row = await env.DB.prepare("SELECT cluster_id FROM cluster_pods WHERE pod_id = ? AND deleted_at IS NULL").bind(pod).first<{ cluster_id: string }>();
  if (!row || row.cluster_id !== cl.id) throw new HttpError(404, "no live pod of this cluster with that id");
  if (phase === "tunnel") {
    const url = typeof b?.url === "string" ? b.url : "";
    if (!endpointUrlOk(env, url)) throw new HttpError(400, "url: a Cloudflare quick-tunnel URL (https://<name>.trycloudflare.com)");
    await env.DB.prepare("UPDATE cluster_pods SET url = ? WHERE pod_id = ?").bind(url, pod).run();
  }
  const detail = typeof b?.detail === "string" ? b.detail.slice(0, 300) : "";
  await putSetting(env, `endpoint:${pod}`, { phase, detail, at: now() }, "pod");
  return { ok: true, pod, phase };
}

/** Every reason a spec's GMI / Brev pools cannot start, at the field it is about (the launch form, `up`'s first step). */
export function providerIssues(env: Env, spec: { pools: PoolSpec[] }): { path: (string | number)[]; message: string }[] {
  const out: { path: (string | number)[]; message: string }[] = [];
  spec.pools.forEach((p, i) => {
    if (!isOtherPool(p)) return;
    const prov = poolProvider(p) as OtherProviderId;
    const impl = providerImpl(prov);
    const off = impl.off(env);
    if (off) return out.push({ path: ["pools", i, "provider"], message: `${impl.title} is off: ${off} (docs/serve/deploy-gmi-brev.md §8)` });
    if (!env.PUBLIC_URL) out.push({ path: ["pools", i, "provider"], message: `PUBLIC_URL is not set: a ${impl.title} pod reports its tunnel URL to fv-control there` });
    const gpus = impl.gpus(env);
    if (p.provider_gpu && !gpus.includes(p.provider_gpu)) out.push({ path: ["pools", i, "provider_gpu"], message: `${p.provider_gpu} is not allowed on ${impl.title}: one of ${gpus.join(", ")} (${prov.toUpperCase()}_${prov === "gmi" ? "PRODUCTS" : "INSTANCE_TYPES"})` });
    const plan = weightsPlan(p);
    if (plan.source === "hub") {
      if (!p.hub_download_approved) out.push({ path: ["pools", i, "hub_download_approved"], message: `weights_source hub downloads ${plan.trees.join(", ") || "the weights"} from the Hub at every boot: it needs the owner's approval for this launch (weights_download_approved: true)` });
      else if (env.FV_HUB_DOWNLOADS_APPROVED !== "1") out.push({ path: ["pools", i, "hub_download_approved"], message: "Hub downloads at boot are not approved on this fv-control (the owner sets FV_HUB_DOWNLOADS_APPROVED=1; CLAUDE.md: large downloads need approval)" });
      if (plan.unsupported.length) out.push({ path: ["pools", i, "weights_source"], message: `no plain Hub source for ${plan.unsupported.join(", ")} (several repos or a conversion: docs/ops/runpod-volumes.md); serve another preset or use Runpod` });
      if (!plan.trees.length) out.push({ path: ["pools", i, "weights_source"], message: "the pool's config loads no weight tree fv-control knows: weights_source none" });
    }
  });
  return out;
}

export { OTHER_PROVIDERS };
