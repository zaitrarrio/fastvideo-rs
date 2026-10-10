// NVIDIA Brev keep-on-stop (docs/serve/deploy-gmi-brev.md §3.4, §7).
//
// Brev keeps /home/ubuntu/workspace across a stop (not a delete); the weights
// a pod fetched into ~/workspace/weights are 35-120 GB. So when a Brev pod of a
// stoppable instance type stops with its cluster (deadline, user stop, budget
// stop) and its weights were complete (the pod became ready), fv-control
// stops the workspace instead of deleting it: it is "parked" (brev_instances,
// migration 0008). The next launch of the same instance type whose trees it
// holds at the pinned revisions restarts it instead of creating one; its boot
// skips those trees (providers.ts WEIGHTS_SH).
//
// The new launch's image and env reach a restarted VM without changing the
// workspace (the CLI's API has no startup-script update): the startup script
// given at create installs a systemd unit that, at every boot, fetches the
// VM's current run script (docker run of our image with its env) from
// GET /ingest/v1/brev-boot with a per-instance token, and runs it. fv-control
// serves it only while the instance is live (not parked or held).
//
// Fallbacks: a start that fails (Brev: no capacity in the same provider /
// region) or does not run within BREV_RESTART_TIMEOUT_S leaves the instance
// "held" for the owner (or deleted, policy brev_park_delete_failed) and a
// fresh instance is created. Non-stoppable types, pods without weights or
// whose weights never completed are deleted as before.
//
// Money and guards: a parked instance bills storage (disk GB × the type's
// $/GB-hr); it is booked in the ledger (owner brev:parked), counted in the
// Brev budget, and the oldest go past brev_park_max instances or
// brev_park_max_days days. Only instances fv-control created and recorded
// here (fv- names) are ever stopped, started or deleted.
import { policies, type Policies } from "./alerts";
import { brev, type BrevType } from "./brev";
import { randomToken, seal, sha256Hex, unseal } from "./crypto";
import type { Env } from "./env";
import { FV_NAME_RE } from "./gmi";
import { getSetting, HttpError, now, parseJson, rateLimit } from "./util";

export const BREV_HOME = "/home/ubuntu/workspace";

// ---------------------------------------------------------------- sizes
/** Approximate tree sizes in GB (scripts/gcp/weights.sh SIZES, du on the volumes; docs/ops/runpod-volumes.md §2). */
export const TREE_GB: Record<string, number> = {
  "h3-base": 144,
  "h3-8step": 144,
  "FastH3-4-step-Preview-v1-LoRA": 7,
  ltx25: 125,
  "ltx25-dev": 45,
  "ltx25-ic-lora-ingredients": 2,
  ltx23: 125,
  ltx2: 90,
  "fastwan21-1.3b": 30,
  "fastwan22-ti2v-5b": 25,
  "sfwan21-1.3b": 29,
  "wan21-t2v-14b": 81,
  "wan22-ti2v-5b": 35,
  upscaler: 1,
  "h3-to-ltx": 1,
};
/** A tree not in the table plans as this many GB. */
export const DEFAULT_TREE_GB = 60;
export const treesGb = (trees: string[]) => trees.reduce((s, t) => s + (TREE_GB[t] ?? DEFAULT_TREE_GB), 0);
/** The disk a Brev VM gets for these trees (`diskStorage`): trees × 1.3 + 40 GB (image, Docker, scratch), at least 200, in 50s. */
export function brevDiskGb(trees: string[]): number {
  const want = treesGb(trees) * 1.3 + 40;
  return Math.max(200, Math.ceil(want / 50) * 50);
}
/** $/GB-hr when the type's storage price is unknown: the highest the list shows (2026-10-10), so the budget errs high. */
export const DEFAULT_STORAGE_USD_PER_GB_HR = 0.000263;
/** A boot's download rate for estimates (GB/s; docs §7: 35-120 GB in 5-15 min). */
export const DOWNLOAD_GBPS = 0.15;
/** Deploy time when the list has none (s). */
export const DEFAULT_DEPLOY_S = 420;

// ---------------------------------------------------------------- the record
export type TreeRevs = Record<string, string>;
export type BrevState = "live" | "restarting" | "parked" | "held" | "deleting" | "deleted" | "gone";
export interface BrevInstance {
  workspace_id: string;
  pod_id: string;
  name: string;
  instance_type: string;
  location: string | null;
  stoppable: number;
  disk_gb: number | null;
  storage_usd_per_gb_hr: number | null;
  trees: string;
  launch_trees: string;
  cluster_id: string | null;
  state: BrevState;
  boot_hash: string | null;
  run_sealed: string | null;
  boots: number;
  created_at: number;
  parked_at: number | null;
  restarted_at: number | null;
  restarts: number;
  note: string | null;
  updated_at: number;
}
/** Storage $/hr of a parked instance. */
export const storageDph = (r: Pick<BrevInstance, "disk_gb" | "storage_usd_per_gb_hr">) => (r.disk_gb || 0) * (r.storage_usd_per_gb_hr ?? DEFAULT_STORAGE_USD_PER_GB_HR);
const COLS = "*";
const PARKED_STATES = "('parked', 'held')";
/** The current record of a Brev pod (by its key brev:<name>); null when fv-control has none (made before keep-on-stop). */
export async function brevRow(env: Env, podId: string): Promise<BrevInstance | null> {
  return env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE pod_id = ? AND state NOT IN ('deleted', 'gone') ORDER BY updated_at DESC LIMIT 1`).bind(podId).first<BrevInstance>();
}
export async function brevRowById(env: Env, workspaceId: string): Promise<BrevInstance | null> {
  return env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE workspace_id = ?`).bind(workspaceId).first<BrevInstance>();
}
async function setState(env: Env, workspaceId: string, state: BrevState, f: { note?: string | null; parked_at?: number | null; trees?: TreeRevs } = {}) {
  const sets = ["state = ?", "updated_at = ?"];
  const vals: unknown[] = [state, now()];
  if (f.note !== undefined) sets.push("note = ?"), vals.push(f.note);
  if (f.parked_at !== undefined) sets.push("parked_at = ?"), vals.push(f.parked_at);
  if (f.trees) sets.push("trees = ?"), vals.push(JSON.stringify(f.trees));
  await env.DB.prepare(`UPDATE brev_instances SET ${sets.join(", ")} WHERE workspace_id = ?`)
    .bind(...vals, workspaceId)
    .run();
}
/** A workspace Brev deleted (DELETE, or no longer listed). */
export async function markBrevDeleted(env: Env, workspaceId: string, state: "deleted" | "gone" = "deleted", note?: string) {
  await env.DB.prepare("UPDATE brev_instances SET state = ?, updated_at = ?, note = COALESCE(?, note) WHERE workspace_id = ? AND state NOT IN ('deleted', 'gone')")
    .bind(state, now(), note ?? null, workspaceId)
    .run();
}
const runAad = (name: string) => `brev-run:${name}`;
/** The run script a VM fetches at its next boot (sealed: it holds the pod's env). */
export async function storeRunScript(env: Env, workspaceId: string, name: string, script: string) {
  await env.DB.prepare("UPDATE brev_instances SET run_sealed = ?, updated_at = ? WHERE workspace_id = ?")
    .bind(await seal(env.CONTROL_KEK, script, runAad(name)), now(), workspaceId)
    .run();
}

// ---------------------------------------------------------------- scripts
const shq = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
/** The per-boot bootstrap: fetch this VM's run script from fv-control (token in a header file, not argv) and run it. */
export const BREV_BOOTSTRAP = `#!/usr/bin/env bash
# fv-control (docs/serve/deploy-gmi-brev.md §3.4): this VM's current launch, fetched at every boot.
set -u
umask 077
D=${BREV_HOME}/fv
for _ in $(seq 1 40); do
  if curl -fsS --max-time 30 -H @"$D/boot-header" -o "$D/run.sh.part" "$(cat "$D/boot-url")"; then
    mv "$D/run.sh.part" "$D/run.sh"
    exec bash "$D/run.sh"
  fi
  sleep 15
done
echo "fv-boot: no run script from fv-control (parked, deleted or unreachable)" >&2
exit 1
`;
export const BREV_UNIT = `[Unit]
Description=fv-control boot: fetch and run this VM's fv-serve launch
After=network-online.target docker.service
Wants=network-online.target docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
KillMode=process
TimeoutStartSec=0
ExecStart=/bin/bash ${BREV_HOME}/fv/bootstrap.sh
[Install]
WantedBy=multi-user.target
`;
/** A Brev VM's startup script (given once at create; root or passwordless sudo, UNVERIFIED): the boot URL and token,
 * the bootstrap and a systemd unit that runs it at every boot (so a stop / start runs the restarting launch even if
 * Brev does not re-run this script; if it does, `start` of the active unit is a no-op). */
export function brevStartup(bootUrl: string, token: string): string {
  return `#!/usr/bin/env bash
set -u
S=""; [ "$(id -u)" = 0 ] || S="sudo -n"
umask 077
D=${BREV_HOME}/fv
mkdir -p ${BREV_HOME}/weights "$D"
printf %s ${shq(bootUrl)} > "$D/boot-url"
printf 'Authorization: Bearer %s\\n' ${shq(token)} > "$D/boot-header"
printf %s ${shq(btoa(BREV_BOOTSTRAP))} | base64 -d > "$D/bootstrap.sh"
printf %s ${shq(btoa(BREV_UNIT))} | base64 -d > "$D/fv-boot.service"
$S install -m 0644 "$D/fv-boot.service" /etc/systemd/system/fv-boot.service
$S systemctl daemon-reload
$S systemctl enable fv-boot.service
$S systemctl start --no-block fv-boot.service
`;
}
/** The run script of one launch (root, from the unit): the env file (0600), our image with the GPU and the weights
 * dir mounted, and a host watchdog that removes the container at the deadline and powers the VM off 5 min later
 * (fv-control parks or deletes it at the deadline itself; this is the backstop). */
export function brevRunScript(image: string, env: Record<string, string>, boot: string): string {
  for (const [k, v] of Object.entries(env)) if (/[\r\n]/.test(v)) throw new HttpError(400, `brev: env ${k} has a line break (docker --env-file takes one line per variable)`);
  const envFile = Object.entries(env)
    .map(([k, v]) => `${k}=${v}`)
    .join("\n");
  const deadline = Number(env.FV_CLUSTER_DEADLINE || "0");
  return `#!/usr/bin/env bash
set -u
S=""; [ "$(id -u)" = 0 ] || S="sudo -n"
umask 077
mkdir -p ${BREV_HOME}/weights ${BREV_HOME}/fv
printf %s ${shq(btoa(envFile))} | base64 -d > ${BREV_HOME}/fv/env
printf %s ${shq(btoa(boot))} | base64 -d > ${BREV_HOME}/fv/boot.sh
$S docker rm -f fv-serve >/dev/null 2>&1 || true
$S docker run -d --name fv-serve --restart no --gpus all --network host --env-file ${BREV_HOME}/fv/env \\
  -v ${BREV_HOME}/weights:/workspace/weights -v ${BREV_HOME}/fv/boot.sh:/fv-boot.sh:ro \\
  --entrypoint bash ${shq(image)} /fv-boot.sh
nohup setsid bash -c 'while [ "$(date +%s)" -lt ${deadline} ]; do sleep 30; done; '"$S"' docker rm -f fv-serve; sleep 300; '"$S"' shutdown -h now' >/dev/null 2>&1 &
`;
}

// ---------------------------------------------------------------- create / restart
/** Records a workspace fv-control just created (live) with its run script; the token's hash finds it at boot. */
export async function recordBrevCreate(
  env: Env,
  x: { workspace_id: string; pod_id: string; name: string; instance_type: string; info: BrevType | null; disk_gb: number | null; launch_trees: TreeRevs; cluster_id: string | null; token: string; run: string },
) {
  const t = now();
  await env.DB.prepare(
    `INSERT INTO brev_instances (workspace_id, pod_id, name, instance_type, location, stoppable, disk_gb, storage_usd_per_gb_hr, trees, launch_trees, cluster_id, state, boot_hash, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, 'live', ?, ?, ?)`,
  )
    .bind(x.workspace_id, x.pod_id, x.name, x.instance_type, x.info?.location ?? null, x.info?.stoppable ? 1 : 0, x.disk_gb, x.info?.storage_usd_per_gb_hr ?? null, JSON.stringify(x.launch_trees), x.cluster_id, await sha256Hex(x.token), t, t)
    .run();
  await storeRunScript(env, x.workspace_id, x.name, x.run);
}
/** A new boot token (the startup script holds it). */
export const newBootToken = () => randomToken("fvb_", 24);

/** Disk GB that trees `present` ∪ `want` need on an instance (the same headroom as at create, without the 1.3). */
const fits = (r: BrevInstance, want: TreeRevs) => {
  if (!r.disk_gb) return true;
  const all = new Set([...Object.keys(parseJson<TreeRevs>(r.trees, {})), ...Object.keys(want)]);
  return treesGb([...all]) + 40 <= r.disk_gb;
};
/** Parked instances of an instance type ranked for a launch needing `want`: most trees at the pinned revision, then newest. */
export async function warmCandidates(env: Env, instanceType: string, want: TreeRevs): Promise<{ row: BrevInstance; matched: string[]; missing: string[] }[]> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state = 'parked' AND instance_type = ? AND stoppable = 1`).bind(instanceType).all<BrevInstance>()).results || [];
  const out: { row: BrevInstance; matched: string[]; missing: string[] }[] = [];
  for (const row of rows) {
    if (!FV_NAME_RE.test(row.name)) continue;
    const have = parseJson<TreeRevs>(row.trees, {});
    const matched = Object.keys(want).filter((t) => have[t] === want[t]);
    if (!matched.length || !fits(row, want)) continue;
    out.push({ row, matched, missing: Object.keys(want).filter((t) => have[t] !== want[t]) });
  }
  return out.sort((a, b) => b.matched.length - a.matched.length || (b.row.parked_at ?? 0) - (a.row.parked_at ?? 0));
}
/** Claims the best parked instance for a launch (parked -> restarting, atomically); null: none fits. */
export async function claimWarm(env: Env, instanceType: string, want: TreeRevs): Promise<{ row: BrevInstance; matched: string[]; missing: string[] } | null> {
  if (!Object.keys(want).length) return null;
  for (const c of await warmCandidates(env, instanceType, want)) {
    const r = await env.DB.prepare("UPDATE brev_instances SET state = 'restarting', updated_at = ? WHERE workspace_id = ? AND state = 'parked'").bind(now(), c.row.workspace_id).run();
    if (Number(r.meta?.changes ?? 0) === 1) return c;
  }
  return null;
}
/** Restarts a claimed instance for a launch: its new run script, then Brev's start; live on success. Throws on a failed start. */
export async function restartWarm(env: Env, row: BrevInstance, x: { run: string; launch_trees: TreeRevs; cluster_id: string }) {
  await storeRunScript(env, row.workspace_id, row.name, x.run);
  await brev.start(env, row.workspace_id);
  const t = now();
  await env.DB.prepare("UPDATE brev_instances SET state = 'live', launch_trees = ?, cluster_id = ?, restarted_at = ?, restarts = restarts + 1, note = NULL, updated_at = ? WHERE workspace_id = ?")
    .bind(JSON.stringify(x.launch_trees), x.cluster_id, t, t, row.workspace_id)
    .run();
}
/** A restart that failed or timed out: stop it again and hold it for the owner (never restarted on its own), or
 * delete it (policy brev_park_delete_failed). The weights it holds stay unless deleted. */
export async function holdFailed(env: Env, row: BrevInstance, why: string, pol?: Policies): Promise<"held" | "deleted"> {
  const p = pol ?? (await policies(env));
  if (p.brev_park_delete_failed) {
    await brev.remove(env, row.workspace_id).catch(() => false);
    await markBrevDeleted(env, row.workspace_id, "deleted", `restart failed: ${why}`.slice(0, 300));
    return "deleted";
  }
  await brev.stop(env, row.workspace_id).catch(() => {});
  await setState(env, row.workspace_id, "held", { note: `restart failed: ${why}`.slice(0, 300), parked_at: row.parked_at ?? now() });
  return "held";
}

// ---------------------------------------------------------------- stop: park or delete
/** The pod ran its weights to completion (it became ready, or reported its tunnel, which the boot does after the weights). */
async function weightsDone(env: Env, podId: string): Promise<boolean> {
  const r = await env.DB.prepare("SELECT ready_at FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ ready_at: number | null }>();
  if (r?.ready_at) return true;
  const rep = await getSetting<{ phase?: string } | null>(env, `endpoint:${podId}`, null);
  return rep?.phase === "tunnel";
}
/** Why a live Brev pod is not parked on stop (null: it is parkable). */
export async function parkRefusal(env: Env, row: BrevInstance, pol: Policies): Promise<string | null> {
  if (!row.stoppable) return `${row.instance_type} is not stoppable`;
  if (pol.brev_park_max <= 0 || pol.brev_park_max_days <= 0) return "keep-on-stop is off (policy brev_park_max / brev_park_max_days)";
  if (!FV_NAME_RE.test(row.name)) return "not an fv- name";
  const trees = { ...parseJson<TreeRevs>(row.trees, {}), ...parseJson<TreeRevs>(row.launch_trees, {}) };
  if (!Object.keys(trees).length) return "no weights to keep";
  if (!(await weightsDone(env, row.pod_id))) return "its weights never completed";
  return null;
}
export type Release = "parked" | "held" | "deleted" | "none";
/**
 * Releases a Brev pod at a stop. `park`: a stoppable one whose weights completed is stopped and parked (its trees
 * recorded), else deleted. A warm restart that never became ready is held (or deleted by policy), whatever `park`.
 * An already parked / held one gets its stop again (the down's verify). Returns what happened, "none" when
 * fv-control has no record (the caller deletes as before).
 */
export async function releaseBrev(env: Env, podId: string, park: boolean, log: (m: string) => void = () => {}): Promise<Release> {
  const row = await brevRow(env, podId);
  if (!row) return "none";
  const pol = await policies(env);
  if (row.state === "parked" || row.state === "held") {
    await brev.stop(env, row.workspace_id).catch(() => {});
    return row.state;
  }
  if (row.restarted_at && !(await weightsDone(env, podId))) {
    const r = await holdFailed(env, row, "it did not come up after the restart", pol);
    log(`${podId}: the restarted instance never came up: ${r === "held" ? "stopped and held for the owner (fv-control.sh brev parked)" : "deleted (policy brev_park_delete_failed)"}`);
    return r;
  }
  const why = park ? await parkRefusal(env, row, pol) : "not a stop";
  if (!why) {
    try {
      await brev.stop(env, row.workspace_id);
      const trees = { ...parseJson<TreeRevs>(row.trees, {}), ...parseJson<TreeRevs>(row.launch_trees, {}) };
      await setState(env, row.workspace_id, "parked", { parked_at: now(), trees, note: null });
      log(`${podId}: parked (stopped; ${Object.keys(trees).join(", ")} kept on its ${row.disk_gb ?? "?"} GB disk, $${(storageDph(row) * 24).toFixed(2)}/day storage)`);
      const lim = await enforceParkLimits(env, pol);
      for (const a of lim) log(a);
      return "parked";
    } catch (e) {
      log(`${podId}: stop failed (${(e as Error).message.slice(0, 160)}): deleting instead`);
    }
  } else if (park) log(`${podId}: not parked (${why}): deleting`);
  await brev.remove(env, row.workspace_id);
  await markBrevDeleted(env, row.workspace_id);
  return "deleted";
}
/** Whether a pod is parked or held (a stopped VM is then the expected end state of a stop). */
export async function isParked(env: Env, podId: string): Promise<boolean> {
  const r = await brevRow(env, podId);
  return r?.state === "parked" || r?.state === "held";
}

/** The `up` wait's view of a warm restart: still starting (fine until the timeout), or timed out (fatal, replace). */
export async function restartDiag(env: Env, podId: string, state: string): Promise<{ phase: string; detail: string; fatal: boolean; replace?: boolean } | null> {
  const row = await brevRow(env, podId);
  if (!row?.restarted_at || row.state !== "live" || (await weightsDone(env, podId))) return null;
  const limit = Math.max(30, Number(env.BREV_RESTART_TIMEOUT_S || "900")) * 1000;
  const waited = now() - row.restarted_at;
  if (state === "failed") return { phase: "restart failed", detail: `Brev reports an error on the restarted ${row.name}`, fatal: true, replace: true };
  if (state === "stopped" || state === "starting") {
    if (waited > limit) return { phase: "restart timed out", detail: `${row.name} not running after ${Math.round(waited / 1000)} s (BREV_RESTART_TIMEOUT_S ${limit / 1000})`, fatal: true, replace: true };
    return { phase: "restarting", detail: `parked ${row.name} starting (${Math.round(waited / 1000)} s)`, fatal: false };
  }
  return null;
}

// ---------------------------------------------------------------- limits, views, owner actions
/** Deletes parked / held instances past brev_park_max (oldest first) or older than brev_park_max_days. Only ours. */
export async function enforceParkLimits(env: Env, pol: Policies, t = now()): Promise<string[]> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PARKED_STATES} ORDER BY parked_at DESC`).all<BrevInstance>()).results || [];
  const out: string[] = [];
  const maxAge = pol.brev_park_max_days * 86400_000;
  for (const [i, r] of rows.entries()) {
    const old = t - (r.parked_at ?? r.updated_at) > maxAge;
    const over = i >= pol.brev_park_max;
    if (!old && !over) continue;
    if (!FV_NAME_RE.test(r.name)) continue;
    const why = over ? `more than brev_park_max ${pol.brev_park_max} parked` : `parked longer than brev_park_max_days ${pol.brev_park_max_days}`;
    out.push(await deleteParked(env, r.workspace_id, why).catch((e) => `brev ${r.name}: delete failed: ${(e as Error).message.slice(0, 160)}`));
  }
  return out;
}
/** Deletes one parked / held instance (owner, or a limit); refused for anything else. */
export async function deleteParked(env: Env, workspaceId: string, why: string): Promise<string> {
  const r = await env.DB.prepare(`UPDATE brev_instances SET state = 'deleting', updated_at = ? WHERE workspace_id = ? AND state IN ${PARKED_STATES}`).bind(now(), workspaceId).run();
  const row = await brevRowById(env, workspaceId);
  if (Number(r.meta?.changes ?? 0) !== 1 || !row) throw new HttpError(404, `no parked Brev instance ${workspaceId} (only parked / held ones fv-control made)`);
  try {
    await brev.remove(env, workspaceId);
  } catch (e) {
    await setState(env, workspaceId, "held", { note: `delete failed: ${(e as Error).message.slice(0, 200)}` });
    throw e;
  }
  await markBrevDeleted(env, workspaceId, "deleted", why.slice(0, 300));
  return `brev ${row.name}: parked instance deleted (${why})`;
}
/** A held instance (restart failed) back into the warm pool: the next launch of its type may restart it. */
export async function unpark(env: Env, workspaceId: string): Promise<BrevInstance> {
  const r = await env.DB.prepare("UPDATE brev_instances SET state = 'parked', note = NULL, updated_at = ? WHERE workspace_id = ? AND state IN ('held', 'parked')").bind(now(), workspaceId).run();
  if (Number(r.meta?.changes ?? 0) !== 1) throw new HttpError(404, `no held Brev instance ${workspaceId}`);
  return (await brevRowById(env, workspaceId))!;
}
/** Parked / held instances with their weights, disk and storage cost (UI, CLI, budget). */
export async function parkedRows(env: Env): Promise<BrevInstance[]> {
  return (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PARKED_STATES} ORDER BY parked_at DESC`).all<BrevInstance>()).results || [];
}
export async function parkedDph(env: Env): Promise<number> {
  return (await parkedRows(env)).reduce((s, r) => s + storageDph(r), 0);
}
export async function parkedView(env: Env) {
  const pol = await policies(env);
  const rows = await parkedRows(env);
  const t = now();
  const parked = rows.map((r) => ({
    workspace_id: r.workspace_id,
    pod_id: r.pod_id,
    name: r.name,
    state: r.state,
    instance_type: r.instance_type,
    location: r.location,
    trees: parseJson<TreeRevs>(r.trees, {}),
    disk_gb: r.disk_gb,
    storage_usd_per_hr: storageDph(r),
    storage_usd_per_day: storageDph(r) * 24,
    parked_at: r.parked_at,
    parked_days: r.parked_at ? (t - r.parked_at) / 86400_000 : null,
    deleted_after: r.parked_at ? r.parked_at + pol.brev_park_max_days * 86400_000 : null,
    restarts: r.restarts,
    note: r.note,
  }));
  return { policy: { brev_park_max: pol.brev_park_max, brev_park_max_days: pol.brev_park_max_days, brev_park_delete_failed: pol.brev_park_delete_failed }, parked, storage_usd_per_day: parked.reduce((s, p) => s + p.storage_usd_per_day, 0) };
}

// ---------------------------------------------------------------- the boot endpoint
/** GET /ingest/v1/brev-boot (Bearer <the VM's boot token>): the run script of its current launch; refused while parked / held. */
export async function brevBoot(env: Env, req: Request): Promise<Response> {
  const auth = req.headers.get("authorization") || "";
  if (!auth.toLowerCase().startsWith("bearer ")) throw new HttpError(401, "boot token required");
  const row = await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE boot_hash = ?`).bind(await sha256Hex(auth.slice(7).trim())).first<BrevInstance>();
  if (!row) throw new HttpError(401, "invalid boot token");
  if (!(await rateLimit(env, `brev-boot:${row.workspace_id}`, 20, 60))) throw new HttpError(429, "boot rate limit (20/min per instance)");
  if (row.state !== "live" && row.state !== "restarting") throw new HttpError(409, `${row.name} is ${row.state}: no launch to run`);
  if (!row.run_sealed) throw new HttpError(404, "no run script yet");
  const script = await unseal(env.CONTROL_KEK, row.run_sealed, runAad(row.name));
  await env.DB.prepare("UPDATE brev_instances SET boots = boots + 1 WHERE workspace_id = ?").bind(row.workspace_id).run();
  return new Response(script, { headers: { "content-type": "text/x-shellscript; charset=utf-8", "cache-control": "no-store" } });
}

// ---------------------------------------------------------------- estimates (launch form, planner)
/** Boot estimate for a launch on a type: warm (a parked instance holds `matched`) or cold (every tree downloads). */
export function bootEstimate(info: Pick<BrevType, "deploy_s"> | null, trees: string[], have: string[] = []): { download_gb: number; boot_s: number } {
  const missing = trees.filter((t) => !have.includes(t));
  const gb = treesGb(missing);
  return { download_gb: gb, boot_s: Math.round((info?.deploy_s ?? DEFAULT_DEPLOY_S) + gb / DOWNLOAD_GBPS) };
}
