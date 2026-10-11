// NVIDIA Brev keep-on-stop (docs/serve/deploy-gmi-brev.md §3.4, §7).
//
// Owner rule (2026-10-11): "stopping a brev vm should NOT delete it. brev VM's
// should be re-used or manually deleted". fv-control never deletes a Brev VM
// on its own. Every stop (user stop, deadline backstop, budget floor, idle
// stop, scale-down, roll, a failed or never-ready boot, a failed warm restart,
// the down's orphan sweep) STOPS the workspace and records it:
//   parked: its weights completed (the pod became ready): kept with its trees;
//   held:   stopped with a note why (incomplete weights, failed boot, failed
//           restart, no record); partial trees stay on its disk and the next
//           boot's weights loop resumes / cleans them (providers.ts WEIGHTS_SH).
// The only deletes are the owner's: `fv-control.sh pod delete <name>` (a
// standalone pod with a live VM), `fv-control.sh brev delete-parked <id>`, the
// UI's Delete on the Pods page's parked list, and
// POST /api/providers/brev/parked/<id>/delete (deleteParked).
//
// Brev keeps /home/ubuntu/workspace across a stop; the weights a pod fetched
// into ~/workspace/weights are 35-120 GB. The next launch of the same instance
// type reuses a parked or held instance (state 'restarting', then live) before
// creating a VM: the one holding most of the requested trees at the pinned
// revisions first, else any of the type with the disk for them (the missing
// trees download onto it). A held instance is reused once BREV_HELD_REUSE_AFTER_S
// (default 900) passed since it was held, so a re-placement after a failed boot
// does not land on the VM that just failed.
//
// The new launch's image and env reach a restarted VM without changing the
// workspace (the CLI's API has no startup-script update): the startup script
// given at create installs a systemd unit that, at every boot, fetches the
// VM's current run script (docker run of our image with its env) from
// GET /ingest/v1/brev-boot with a per-instance token, and runs it. fv-control
// serves it only while the instance is live (not parked or held).
//
// A stop Brev refuses (a type that turned out not stoppable, an API error)
// leaves the VM running in state 'stop_failed': alert brev_stop_failed with its
// $/hr, the cron retries the stop, nothing is deleted. Non-stoppable types are
// refused at launch (providers.ts brevStoppableIssue).
//
// Money and guards: a parked / held instance bills storage (disk GB × the type's
// $/GB-hr); it is booked in the ledger (owner brev:parked) and counted in the
// Brev budget. brev_park_max / brev_park_max_days are alert thresholds only
// (alert brev_parked_over_limit); brev_park_delete_failed is removed. Only
// instances fv-control created and recorded here (fv- names) are ever stopped,
// started or deleted.
import { policies, type AlertIn, type Policies } from "./alerts";
import { brev, type BrevType } from "./brev";
import { brevHostScript } from "./brev-host";
import { randomToken, seal, sha256Hex, unseal } from "./crypto";
import type { Env } from "./env";
import { FV_NAME_RE } from "./gmi";
import { releaseTunnels } from "./tunnels";
import { getSetting, HttpError, now, parseJson, putSetting, rateLimit } from "./util";

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
/** live | restarting (claimed for a launch) | parked | held | stop_failed (Brev refused the stop: still running) |
 * deleting (the owner's delete) | deleted | gone (Brev no longer lists it). */
export type BrevState = "live" | "restarting" | "parked" | "held" | "stop_failed" | "deleting" | "deleted" | "gone";
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
  // Its named tunnel and DNS record (tunnels.ts), kept while it was parked: gone with it (failures: the cron retries).
  const row = await brevRowById(env, workspaceId);
  if (row && !(await env.DB.prepare("SELECT 1 AS x FROM brev_instances WHERE pod_id = ? AND state IN ('live', 'restarting', 'parked', 'held')").bind(row.pod_id).first()))
    await releaseTunnels(env, row.pod_id, `its Brev instance was ${state}`);
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
/** The run script of one launch (root, from the unit): brev-host.ts's host script (the host's facts and failures
 * reported, the env file 0600, our image with the GPU and the weights dir, a heartbeat until the tunnel, the deadline
 * watchdog; fv-control parks or deletes the VM at the deadline itself). */
export function brevRunScript(image: string, env: Record<string, string>, boot: string): string {
  return brevHostScript(image, env, boot);
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

/** Whether the trees on an instance plus `want` fit its disk (the same 40 GB headroom as at create, without the 1.3).
 * On the disk: its complete trees (`trees`) and whatever its last launch fetched (`launch_trees`, complete or partial:
 * the weights loop replaces a partial tree in place, so a name counts once whatever its revision). Trees it holds
 * that the launch does not want stay on the disk and count. */
export function fitsDisk(r: Pick<BrevInstance, "disk_gb" | "trees" | "launch_trees">, want: TreeRevs): boolean {
  if (!r.disk_gb) return true;
  const all = new Set([...Object.keys(parseJson<TreeRevs>(r.trees, {})), ...Object.keys(parseJson<TreeRevs>(r.launch_trees, {})), ...Object.keys(want)]);
  return treesGb([...all]) + 40 <= r.disk_gb;
}
/** A held instance is reused for a launch once this long passed since it was held (BREV_HELD_REUSE_AFTER_S, default
 * 900 s): a re-placement right after a failed boot / restart does not land on the same VM. */
export const heldReuseAfterMs = (env: Env) => Math.max(0, Number(env.BREV_HELD_REUSE_AFTER_S ?? "900") || 0) * 1000;
export interface WarmCandidate {
  row: BrevInstance;
  matched: string[];
  missing: string[];
}
/** Parked / held instances of an instance type a launch needing `want` may reuse, best first: most trees at the pinned
 * revision, then parked before held, then the newest. Any of the type with the disk for the trees qualifies (the
 * missing ones download onto it); a held one once heldReuseAfterMs passed. Only ours (fv- names). */
export async function warmCandidates(env: Env, instanceType: string, want: TreeRevs, t = now()): Promise<WarmCandidate[]> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PARKED_STATES} AND instance_type = ?`).bind(instanceType).all<BrevInstance>()).results || [];
  const out: WarmCandidate[] = [];
  const cool = heldReuseAfterMs(env);
  for (const row of rows) {
    if (!FV_NAME_RE.test(row.name)) continue;
    if (row.state === "held" && t - row.updated_at < cool) continue;
    if (!fitsDisk(row, want)) continue;
    const have = parseJson<TreeRevs>(row.trees, {});
    out.push({ row, matched: Object.keys(want).filter((k) => have[k] === want[k]), missing: Object.keys(want).filter((k) => have[k] !== want[k]) });
  }
  const held = (c: WarmCandidate) => Number(c.row.state === "held");
  return out.sort((a, b) => b.matched.length - a.matched.length || held(a) - held(b) || (b.row.parked_at ?? 0) - (a.row.parked_at ?? 0));
}
/** Claims the best parked / held instance for a launch (-> restarting, atomically); null: none of the type fits. */
export async function claimWarm(env: Env, instanceType: string, want: TreeRevs): Promise<WarmCandidate | null> {
  for (const c of await warmCandidates(env, instanceType, want)) {
    const r = await env.DB.prepare("UPDATE brev_instances SET state = 'restarting', updated_at = ? WHERE workspace_id = ? AND state = ?").bind(now(), c.row.workspace_id, c.row.state).run();
    if (Number(r.meta?.changes ?? 0) === 1) return c;
  }
  return null;
}
/** Gives a claimed instance back (the launch did not get to start it): parked or held as it was. */
export async function unclaimWarm(env: Env, row: BrevInstance) {
  await env.DB.prepare("UPDATE brev_instances SET state = ?, updated_at = ? WHERE workspace_id = ? AND state = 'restarting'").bind(row.state, row.updated_at, row.workspace_id).run();
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

// ---------------------------------------------------------------- stop: park or hold (never delete)
/** The pod ran its weights to completion (it became ready, or reported its tunnel, which the boot does after the weights). */
async function weightsDone(env: Env, podId: string): Promise<boolean> {
  const r = await env.DB.prepare("SELECT ready_at FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ ready_at: number | null }>();
  if (r?.ready_at) return true;
  const rep = await getSetting<{ phase?: string } | null>(env, `endpoint:${podId}`, null);
  return rep?.phase === "tunnel";
}
/** The $/hr a running Brev pod costs (the price fv-control recorded at create); null when unknown. */
async function podDph(env: Env, podId: string): Promise<number | null> {
  const r = await env.DB.prepare("SELECT cost_per_hr FROM cluster_pods WHERE pod_id = ?").bind(podId).first<{ cost_per_hr: number | null }>();
  return r?.cost_per_hr ?? null;
}
const money = (x: number | null) => (x === null ? "an unknown $/hr" : `$${x.toFixed(2)}/hr`);
export type StopOutcome = "parked" | "held" | "stop_failed";
/** Stops a live / restarting instance and records it: parked with `trees` (its weights completed), else held with
 * `note`. Brev refusing the stop leaves it running as stop_failed (alert brev_stop_failed; the cron retries).
 * Never deletes. */
async function stopAndRecord(env: Env, row: BrevInstance, rec: { parked: TreeRevs } | { note: string }, log: (m: string) => void): Promise<StopOutcome> {
  try {
    await brev.stop(env, row.workspace_id);
  } catch (e) {
    const dph = await podDph(env, row.pod_id);
    const why = `stop refused by Brev (${(e as Error).message.slice(0, 160)}): left RUNNING at ${money(dph)}${"note" in rec ? `; ${rec.note}` : ""}`;
    await setState(env, row.workspace_id, "stop_failed", { note: why.slice(0, 300) });
    log(`${row.pod_id}: WARNING: ${why}. fv-control never deletes a Brev VM on its own: stop it in the Brev console or delete it (fv-control.sh brev delete-parked ${row.workspace_id}); alert brev_stop_failed`);
    return "stop_failed";
  }
  if ("parked" in rec) {
    await setState(env, row.workspace_id, "parked", { parked_at: now(), trees: rec.parked, note: null });
    return "parked";
  }
  await setState(env, row.workspace_id, "held", { note: rec.note.slice(0, 300), parked_at: now() });
  return "held";
}
/** A restart that failed or timed out: stop it again and hold it (reused by a later launch of its type after
 * heldReuseAfterMs, or deleted by the owner). Its weights stay. */
export async function holdFailed(env: Env, row: BrevInstance, why: string, log: (m: string) => void = () => {}): Promise<StopOutcome> {
  return stopAndRecord(env, row, { note: `restart failed: ${why}` }, log);
}
export type Release = StopOutcome | "deleted" | "none";
/**
 * Releases a Brev pod. Every stop (`manual` unset) stops the VM: parked when its weights completed (its trees
 * recorded), else held with a note (a never-ready boot: `why`; a warm restart that never came up). An already
 * parked / held one gets its stop again (the down's verify); one in the owner's delete counts as gone. A stop Brev
 * refuses leaves it running (stop_failed, alerted). `manual` (the owner's `pod delete` of a standalone pod) deletes
 * the VM: the only delete here. Returns "none" when fv-control has no record of it (the caller decides).
 */
export async function releaseBrev(env: Env, podId: string, o: { manual?: boolean; why?: string } = {}, log: (m: string) => void = () => {}): Promise<Release> {
  const row = await brevRow(env, podId);
  if (!row) return "none";
  if (row.state === "deleting") return "deleted";
  if (o.manual) {
    if (!FV_NAME_RE.test(row.name)) throw new Error(`${row.name}: not an fv- name: not deleted`);
    await brev.remove(env, row.workspace_id);
    await markBrevDeleted(env, row.workspace_id, "deleted", `deleted by the owner (${o.why || "pod delete"})`.slice(0, 300));
    log(`${podId}: Brev VM deleted (the owner's delete: ${o.why || "pod delete"})`);
    return "deleted";
  }
  if (row.state === "parked" || row.state === "held") {
    await brev.stop(env, row.workspace_id).catch(() => {});
    return row.state;
  }
  const why = o.why ? ` (${o.why})` : "";
  const keep = `reused by a later launch of ${row.instance_type}, or delete it: fv-control.sh brev parked`;
  if (row.restarted_at && !(await weightsDone(env, podId))) {
    const r = await stopAndRecord(env, row, { note: `restart failed: it did not come up after the restart${why}` }, log);
    if (r === "held") log(`${podId}: the restarted instance never came up: stopped and held (${keep})`);
    return r;
  }
  if (!(await weightsDone(env, podId))) {
    const r = await stopAndRecord(env, row, { note: `stopped before its boot became ready${why}: weights may be incomplete (the next boot resumes them)` }, log);
    if (r === "held") log(`${podId}: stopped and held, not deleted: its boot never became ready${why}; partial weights stay on its disk (${keep})`);
    return r;
  }
  const trees = { ...parseJson<TreeRevs>(row.trees, {}), ...parseJson<TreeRevs>(row.launch_trees, {}) };
  const r = await stopAndRecord(env, row, { parked: trees }, log);
  if (r !== "parked") return r;
  log(`${podId}: parked (stopped, not deleted; ${Object.keys(trees).join(", ") || "no weights"} kept on its ${row.disk_gb ?? "?"} GB disk, $${(storageDph(row) * 24).toFixed(2)}/day storage)`);
  const lim = await parkLimitAlert(env, await policies(env));
  if (lim) log(`WARNING: ${lim.message}`);
  return "parked";
}
/** A Brev pod fv-control has no keep-on-stop record of (made before it): stopped and recorded held. Never deleted. */
export async function adoptAndStop(env: Env, x: { workspace_id: string; pod_id: string; name: string; instance_type: string }, why: string, log: (m: string) => void = () => {}): Promise<StopOutcome> {
  const t = now();
  await env.DB.prepare(
    `INSERT INTO brev_instances (workspace_id, pod_id, name, instance_type, stoppable, trees, launch_trees, state, created_at, updated_at, note)
     VALUES (?, ?, ?, ?, 0, '{}', '{}', 'live', ?, ?, ?) ON CONFLICT (workspace_id) DO NOTHING`,
  )
    .bind(x.workspace_id, x.pod_id, x.name, x.instance_type, t, t, "no keep-on-stop record (made before it)")
    .run();
  const row = await brevRowById(env, x.workspace_id);
  if (!row) throw new Error(`${x.name}: no record`);
  const r = await stopAndRecord(env, row, { note: `no keep-on-stop record: its weights are unknown (${why})` }, log);
  if (r === "held") log(`${x.pod_id}: stopped and held, not deleted (no keep-on-stop record; reused by a later launch of ${x.instance_type}, or delete it: fv-control.sh brev parked)`);
  return r;
}
/** Workspace ids of records Brev should still list (live / parked / held / stop_failed) that it did not, changed before `before`. */
export async function brevUnlisted(env: Env, listed: Set<string>, before: number): Promise<string[]> {
  const rows = (await env.DB.prepare("SELECT workspace_id FROM brev_instances WHERE state IN ('live', 'parked', 'held', 'stop_failed') AND updated_at < ?").bind(before).all<{ workspace_id: string }>()).results || [];
  return rows.map((r) => r.workspace_id).filter((id) => !listed.has(id));
}
/** Whether a pod is parked or held (a stopped VM is then the expected end state of a stop). */
export async function isParked(env: Env, podId: string): Promise<boolean> {
  const r = await brevRow(env, podId);
  return r?.state === "parked" || r?.state === "held";
}
/** The cron's pass over stops Brev refused: the stop again (held once it took, or once Brev shows it stopped); still
 * running: alert brev_stop_failed naming its $/hr. Never deletes. `listed`: workspace id -> its state in Brev's list. */
export async function retryRefusedStops(env: Env, listed: Map<string, string>): Promise<{ alerts: AlertIn[]; actions: string[] }> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state = 'stop_failed'`).all<BrevInstance>()).results || [];
  const alerts: AlertIn[] = [];
  const actions: string[] = [];
  for (const r of rows) {
    const st = listed.get(r.workspace_id);
    if (st === undefined) continue; // brevUnlisted marks it gone
    let stopped = st === "stopped";
    let err = "";
    if (!stopped)
      try {
        await brev.stop(env, r.workspace_id);
        stopped = true;
      } catch (e) {
        err = (e as Error).message.slice(0, 160);
      }
    if (stopped) {
      await setState(env, r.workspace_id, "held", { note: `stopped after Brev first refused the stop (${(r.note || "").slice(0, 200)})`, parked_at: now() });
      actions.push(`brev ${r.name}: stopped (an earlier stop was refused): held`);
      continue;
    }
    const dph = await podDph(env, r.pod_id);
    alerts.push({
      key: `brev_stop_failed:${r.workspace_id}`,
      kind: "brev_stop_failed",
      severity: "critical",
      target: r.pod_id,
      message: `Brev ${r.name} (${r.instance_type}) is still RUNNING at ${money(dph)}${dph !== null ? ` ($${(dph * 24).toFixed(2)}/day)` : ""}: Brev refused its stop (${err || r.note || "?"}). fv-control never deletes a Brev VM on its own: stop it in the Brev console or delete it (fv-control.sh brev delete-parked ${r.workspace_id})`,
    });
  }
  return { alerts, actions };
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

// ---------------------------------------------------------------- limits (alerts only), views, owner actions
/** The park thresholds as an alert (brev_parked_over_limit): more than brev_park_max parked / held instances, or one
 * parked longer than brev_park_max_days. Lists each (age, $/day). Nothing is deleted (owner rule 2026-10-11). */
export async function parkLimitAlert(env: Env, pol: Policies, t = now()): Promise<AlertIn | null> {
  const rows = await parkedRows(env);
  const maxAge = pol.brev_park_max_days * 86400_000;
  const old = rows.filter((r) => t - (r.parked_at ?? r.updated_at) > maxAge);
  const over = rows.length > pol.brev_park_max;
  if (!over && !old.length) return null;
  const age = (r: BrevInstance) => ((t - (r.parked_at ?? r.updated_at)) / 86400_000).toFixed(1);
  const list = rows.map((r) => `${r.name} (${r.state}, ${r.instance_type}, ${age(r)} d, $${(storageDph(r) * 24).toFixed(2)}/day)`).join("; ");
  const why = [over ? `${rows.length} parked / held, more than brev_park_max ${pol.brev_park_max}` : "", old.length ? `${old.length} parked longer than brev_park_max_days ${pol.brev_park_max_days}` : ""].filter(Boolean).join("; ");
  const total = rows.reduce((s, r) => s + storageDph(r) * 24, 0);
  return {
    key: "brev_parked_over_limit",
    kind: "brev_parked_over_limit",
    severity: "warn",
    message: `NVIDIA Brev: ${why}: ${list}; storage $${total.toFixed(2)}/day in all. Not deleted (fv-control never deletes a Brev VM on its own): the next launch of a type reuses them, or delete one (Pods page, fv-control.sh brev delete-parked <id>)`,
  };
}
/** Deletes one parked / held (or stop_failed) instance: the owner's action only (API, CLI, UI); refused for anything else. */
export async function deleteParked(env: Env, workspaceId: string, why: string): Promise<string> {
  const r = await env.DB.prepare("UPDATE brev_instances SET state = 'deleting', updated_at = ? WHERE workspace_id = ? AND state IN ('parked', 'held', 'stop_failed')").bind(now(), workspaceId).run();
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
/** A held instance back into the warm pool now (without waiting for BREV_HELD_REUSE_AFTER_S), its note cleared. */
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
/** The policy line the UI and `fv-control.sh brev parked` show. */
export const parkPolicyLine = (pol: Pick<Policies, "brev_park_max" | "brev_park_max_days">) =>
  `stopped Brev VMs are kept, never deleted by fv-control (owner rule 2026-10-11): reused by the next launch of their type, or deleted by you; alert past ${pol.brev_park_max} parked or ${pol.brev_park_max_days} days each`;
export async function parkedView(env: Env) {
  const pol = await policies(env);
  const rows = [...(await parkedRows(env)), ...((await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state = 'stop_failed'`).all<BrevInstance>()).results || [])];
  const t = now();
  const cool = heldReuseAfterMs(env);
  const reuse = (r: BrevInstance) =>
    r.state === "stop_failed"
      ? "running: Brev refused its stop"
      : r.state === "held" && t - r.updated_at < cool
        ? `reusable from ${new Date(r.updated_at + cool).toISOString().slice(11, 16)}Z (Unpark: now)`
        : `reusable: the next launch of ${r.instance_type}`;
  const parked = rows.map((r) => ({
    workspace_id: r.workspace_id,
    pod_id: r.pod_id,
    name: r.name,
    state: r.state,
    instance_type: r.instance_type,
    location: r.location,
    trees: parseJson<TreeRevs>(r.trees, {}),
    disk_gb: r.disk_gb,
    storage_usd_per_hr: r.state === "stop_failed" ? 0 : storageDph(r),
    storage_usd_per_day: r.state === "stop_failed" ? 0 : storageDph(r) * 24,
    parked_at: r.parked_at,
    parked_days: r.parked_at ? (t - r.parked_at) / 86400_000 : null,
    over_age: r.state !== "stop_failed" && !!r.parked_at && t - r.parked_at > pol.brev_park_max_days * 86400_000,
    reuse: reuse(r),
    restarts: r.restarts,
    note: r.note,
  }));
  return {
    policy: { brev_park_max: pol.brev_park_max, brev_park_max_days: pol.brev_park_max_days, auto_delete: false, line: parkPolicyLine(pol) },
    parked,
    over_limit: parked.filter((p) => p.state !== "stop_failed").length > pol.brev_park_max || parked.some((p) => p.over_age),
    storage_usd_per_day: parked.reduce((s, p) => s + p.storage_usd_per_day, 0),
  };
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
  // Into the pod's report history (provider-logs): the VM booted and fetched its launch (before the host script reports).
  // (Read raw: getSetting merges objects, which turns an array into one.)
  const raw = await env.DB.prepare("SELECT value FROM settings WHERE key = ?").bind(`endpoint-log:${row.pod_id}`).first<{ value: string }>();
  const h0 = parseJson<unknown>(raw?.value, []);
  const hist = Array.isArray(h0) ? (h0 as { phase: string; detail: string; at: number }[]) : [];
  await putSetting(env, `endpoint-log:${row.pod_id}`, [...hist, { phase: "run_fetched", detail: `boot ${row.boots + 1}: the VM fetched its run script`, at: now() }].slice(-30), "pod");
  return new Response(script, { headers: { "content-type": "text/x-shellscript; charset=utf-8", "cache-control": "no-store" } });
}

// ---------------------------------------------------------------- estimates (launch form, planner)
/** Boot estimate for a launch on a type: warm (a parked instance holds `matched`) or cold (every tree downloads). */
export function bootEstimate(info: Pick<BrevType, "deploy_s"> | null, trees: string[], have: string[] = [], r2?: { bytes: (trees: string[]) => number; gbps: number }): { download_gb: number; boot_s: number } {
  const missing = trees.filter((t) => !have.includes(t));
  // weights_source r2: the manifests' exact bytes and the R2 rate; else the table and the Hub rate.
  const exact = r2 ? r2.bytes(missing) / 1e9 : 0;
  const gb = r2 && exact ? Math.round(exact) : treesGb(missing);
  return { download_gb: gb, boot_s: Math.round((info?.deploy_s ?? DEFAULT_DEPLOY_S) + gb / (r2 ? r2.gbps : DOWNLOAD_GBPS)) };
}
