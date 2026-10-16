// NVIDIA Brev keep-on-stop (docs/serve/deploy-gmi-brev.md §3.4, §7).
//
// Owner rule (2026-10-11): "stopping a brev vm should NOT delete it. brev VM's
// should be re-used or manually deleted". Every stop (user stop, deadline
// backstop, budget floor, idle stop, scale-down, roll, a failed or never-ready
// boot, a failed warm restart, the down's orphan sweep) STOPS the workspace and,
// once the stop succeeded, keeps it:
//   parked: its weights completed (the pod became ready): kept with its trees;
//   held:   stopped with a note why (incomplete weights, failed boot, failed
//           restart, no record); partial trees stay on its disk and the next
//           boot's weights loop resumes / cleans them (providers.ts WEIGHTS_SH).
// A VM whose stop succeeded is never deleted by fv-control.
//
// Owner rule (2026-10-13): "We should go ahead and delete VMs that need to be
// stopped if a stop command does not succeed. If the stop command succeeds then
// we should NOT delete the VM." A stop does not succeed when Brev refuses it
// ("does not support stop", HTTP 500, for a type it cannot stop: measured
// 2026-10-13), when any other stop error persists over STOP_TRIES tries
// (BREV_STOP_RETRY_S apart), or when Brev accepted it but the VM is not STOPPED
// BREV_STOP_TIMEOUT_S (default 600 s = 10 min) after it (the cron checks). Then
// the VM is deleted (fv- names and recorded rows only), recorded 'deleted' with
// the reason ("stop failed (<reason>): deleted"), its tunnel released, logged,
// and an info alert brev_stop_deleted is raised. A VM still DEPLOYING when its
// stop comes (Brev answers "instance not found") is 'stop_pending': released
// from its cluster; the cron sends the stop once it leaves DEPLOYING, or deletes
// it when it is still DEPLOYING BREV_DEPLOYING_WAIT_S (default 900 s) after the
// stop was requested. This replaces the 2026-10-13 morning states
// 'running_unstoppable' (released, left running) and 'stop_failed' (left
// running, retried): rows still in them are deleted by the cron (their stop
// already failed), unless Brev shows them STOPPED (held).
//
// The owner's deletes stay as they were: `fv-control.sh pod delete <name>` (a
// standalone pod with a live VM), `fv-control.sh brev delete-parked <id>`, the
// UI's Delete on the Pods page's parked list, and
// POST /api/v1/providers/brev/parked/<id>/delete (deleteParked).
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
// Money and guards: a parked / held instance bills storage (disk GB × the type's
// $/GB-hr); it is booked in the ledger (owner brev:parked) and counted in the
// Brev budget. brev_park_max / brev_park_max_days are alert thresholds only
// (alert brev_parked_over_limit); brev_park_delete_failed is removed. A VM whose
// stop is not yet confirmed (or stop_pending) bills its GPU $/hr (owner
// brev:stopping) until it is STOPPED or deleted. Only instances fv-control
// created and recorded here (fv- names) are ever stopped, started or deleted.
import { policies, type AlertIn, type Policies } from "./alerts";
import { brev, brevStatusLine, brevTaskLine, type BrevType, type BrevWorkspace } from "./brev";
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
  "fasth3-trim-8step": 44,
  "fasth3-trim-8step-nvfp4": 12,
  "fasth3-v2-nvfp4": 56,
  "fasth3-4step-vsa-nvfp4": 56,
  "h3-turbo-loras/larry": 1,
  "h3-turbo-loras/lightx2v": 2,
  "vdn-h3-stage-b-2000": 5,
  "vdn-h3-dmd-250-turbo": 1,
  "h3-turbo-loras/lightx2v-v0.1": 2,
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
/** live | restarting (claimed for a launch) | parked | held | stop_pending (its stop came while Brev still showed it
 * DEPLOYING: released; the cron stops it once it leaves DEPLOYING, or deletes it) | deleting (the owner's delete) |
 * deleted | gone (Brev no longer lists it). Legacy (before the owner rule of 2026-10-13, migration 0012): stop_failed
 * and running_unstoppable, rows whose stop failed; the cron deletes them. */
export type BrevState = "live" | "restarting" | "parked" | "held" | "stop_pending" | "stop_failed" | "running_unstoppable" | "deleting" | "deleted" | "gone";
/** States whose VM still runs after a stop that did not (yet) succeed: the cron settles them (stop or delete). */
const PENDING_STATES = "('stop_pending', 'stop_failed', 'running_unstoppable')";
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
  /** The VM's GPU $/hr (recorded at create); null when unknown. */
  usd_per_hr: number | null;
  /** parked / held: when the stop Brev accepted was sent, until Brev shows the VM STOPPED (null: confirmed); the cron
   * deletes it when it is not STOPPED BREV_STOP_TIMEOUT_S after. stop_pending: when the stop was requested (the
   * DEPLOYING wait, BREV_DEPLOYING_WAIT_S, counts from it). */
  stop_tried_at: number | null;
  /** Starts sent again for the current restart after Brev failed it for capacity (migration 0013), and the last one's time. */
  capacity_retries: number;
  capacity_retry_at: number | null;
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
async function setState(env: Env, workspaceId: string, state: BrevState, f: { note?: string | null; parked_at?: number | null; trees?: TreeRevs; usd_per_hr?: number | null; stop_tried_at?: number | null } = {}) {
  const sets = ["state = ?", "updated_at = ?"];
  const vals: unknown[] = [state, now()];
  if (f.note !== undefined) sets.push("note = ?"), vals.push(f.note);
  if (f.parked_at !== undefined) sets.push("parked_at = ?"), vals.push(f.parked_at);
  if (f.trees) sets.push("trees = ?"), vals.push(JSON.stringify(f.trees));
  if (f.usd_per_hr !== undefined && f.usd_per_hr !== null) sets.push("usd_per_hr = ?"), vals.push(f.usd_per_hr);
  if (f.stop_tried_at !== undefined) sets.push("stop_tried_at = ?"), vals.push(f.stop_tried_at);
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
  x: { workspace_id: string; pod_id: string; name: string; instance_type: string; info: BrevType | null; disk_gb: number | null; launch_trees: TreeRevs; cluster_id: string | null; token: string; run: string; usd_per_hr?: number | null },
) {
  const t = now();
  await env.DB.prepare(
    `INSERT INTO brev_instances (workspace_id, pod_id, name, instance_type, location, stoppable, disk_gb, storage_usd_per_gb_hr, trees, launch_trees, cluster_id, state, boot_hash, created_at, updated_at, usd_per_hr)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, 'live', ?, ?, ?, ?)`,
  )
    .bind(x.workspace_id, x.pod_id, x.name, x.instance_type, x.info?.location ?? null, x.info?.stoppable ? 1 : 0, x.disk_gb, x.info?.storage_usd_per_gb_hr ?? null, JSON.stringify(x.launch_trees), x.cluster_id, await sha256Hex(x.token), t, t, x.usd_per_hr ?? x.info?.usd_per_hr ?? null)
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
    // Given back after its restart found no capacity: not the launch's next pick for the same cooldown.
    if ((row.note || "").startsWith(NO_CAPACITY_PREFIX) && t - row.updated_at < cool) continue;
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
/** How long a warm restart waits for a workspace still STOPPING (from its park) before its start (BREV_STOPPING_WAIT_S). */
const stoppingWaitMs = (env: Env) => Math.max(0, Number(env.BREV_STOPPING_WAIT_S ?? "60") || 0) * 1000;
/** A start that left the workspace STOPPED this long is sent once more (BREV_RESTART_RETRY_S, 0: never). */
const restartRetryMs = (env: Env) => Math.max(0, Number(env.BREV_RESTART_RETRY_S ?? "180") || 0) * 1000;
/** A restarted workspace still STOPPED this long after its start raises brev_restart_stuck (BREV_RESTART_ALERT_S). */
const restartAlertMs = (env: Env) => Math.max(60, Number(env.BREV_RESTART_ALERT_S ?? "300") || 300) * 1000;
const sleepMs = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Restarts a claimed instance for a launch: its new run script, then Brev's start; live on success. Throws on a failed
 * start. As the CLI does (brev-cli pkg/cmd/start startStopppedWorkspace), it starts only a STOPPED workspace whose type
 * can be stopped: one still STOPPING from its park is waited for (BREV_STOPPING_WAIT_S), one already RUNNING or
 * STARTING is not started again, anything else is refused with Brev's status and message. Brev's answer to the start
 * (its status and statusMessage) is kept in the record's note and returned for the operation's log
 * (docs/ops/r2-bench-2026-10-12.md finding 7: a start left the VM STOPPED for 700 s and nothing said why).
 */
export async function restartWarm(env: Env, row: BrevInstance, x: { run: string; launch_trees: TreeRevs; cluster_id: string }): Promise<string> {
  await storeRunScript(env, row.workspace_id, row.name, x.run);
  let w: BrevWorkspace | null = await brev.get(env, row.workspace_id).catch(() => null);
  const until = now() + stoppingWaitMs(env);
  while (w?.status === "STOPPING" && now() < until) {
    await sleepMs(5000);
    w = await brev.get(env, row.workspace_id).catch(() => w);
  }
  const before = w ? brevStatusLine(w) : "unknown (GET failed)";
  if (w && w.stoppable === false) throw new HttpError(409, `brev ${row.name}: its instance type cannot be started again (instanceTypeInfo.stoppable false)`);
  let said: string;
  if (w && (w.status === "RUNNING" || w.status === "STARTING" || w.status === "DEPLOYING")) said = `already ${before}: not started again`;
  else if (w && w.status !== "STOPPED") throw new HttpError(409, `brev ${row.name} is ${before}: only a STOPPED workspace is started`);
  else {
    const ans = await brev.start(env, row.workspace_id);
    said = `start accepted (was ${before}; Brev answered ${ans ? brevStatusLine(ans) : "no workspace"})`;
  }
  const t = now();
  await env.DB.prepare("UPDATE brev_instances SET state = 'live', launch_trees = ?, cluster_id = ?, restarted_at = ?, restarts = restarts + 1, note = ?, stop_tried_at = NULL, capacity_retries = 0, capacity_retry_at = NULL, updated_at = ? WHERE workspace_id = ?")
    .bind(JSON.stringify(x.launch_trees), x.cluster_id, t, said.slice(0, 300), t, row.workspace_id)
    .run();
  return said;
}

// ---------------------------------------------------------------- stop: park or hold; delete when the stop fails
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
/** The GPU $/hr a Brev VM costs while it runs: its pod's recorded price, else its record's (create), else the type's
 * live price; null when none is known. */
export async function vmDph(env: Env, row: Pick<BrevInstance, "pod_id" | "usd_per_hr" | "instance_type">): Promise<number | null> {
  const p = await podDph(env, row.pod_id);
  if (p !== null && p > 0) return p;
  if (row.usd_per_hr !== null && row.usd_per_hr !== undefined) return row.usd_per_hr;
  return (await brev.type(env, row.instance_type).catch(() => null))?.usd_per_hr ?? null;
}
const money = (x: number | null) => (x === null ? "an unknown $/hr" : `$${x.toFixed(2)}/hr`);
const hhmm = (ms: number) => `${new Date(ms).toISOString().slice(11, 16)}Z`;

/** Brev's answer to the stop of a type it cannot stop (measured 2026-10-13 on a RUNNING massedcompute_A6000,
 * isStoppable false: HTTP 500 `rpc error: code = Internal desc = instance "inst-…" does not support stop`; the VM
 * stays RUNNING / HEALTHY and keeps billing). Permanent: the VM is deleted at once. */
export const NOT_STOPPABLE_RE = /does not support stop/i;
/** Brev's answer to the stop of a workspace still DEPLOYING, never provisioned (measured 2026-10-13: HTTP 500
 * `rpc error: code = NotFound desc = instance not found`). The stop is sent again once it leaves DEPLOYING. */
export const NOT_PROVISIONED_RE = /instance not found/i;
export type StopRefusal = "not_stoppable" | "not_provisioned" | "other";
export const stopRefusal = (msg: string): StopRefusal => (NOT_STOPPABLE_RE.test(msg) ? "not_stoppable" : NOT_PROVISIONED_RE.test(msg) ? "not_provisioned" : "other");
/** A Brev API error in a few words: the HTTP status and Brev's own message (`{"errors":[{"message": …}]}`) when the
 * body has one, else the error's first 160 characters. */
export function brevErrText(msg: string): string {
  const st = /: (\d{3}) /.exec(msg)?.[1];
  const m = /"message"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(msg)?.[1];
  if (!m) return msg.slice(0, 160);
  return `${st ? `HTTP ${st} ` : ""}${m.replace(/\\"/g, '"').replace(/\\\\/g, "\\")}`.slice(0, 200);
}
const envSeconds = (v: string | undefined, dflt: number, min: number) => {
  const n = Number(v ?? dflt);
  return Math.max(min, Number.isFinite(n) ? n : dflt) * 1000;
};
/** Tries of a stop Brev answers with an error other than "does not support stop" / "instance not found". */
export const STOP_TRIES = 3;
/** Seconds between those tries (BREV_STOP_RETRY_S, default 3). */
const stopRetryMs = (env: Env) => envSeconds(env.BREV_STOP_RETRY_S, 3, 0);
/** A stop Brev accepted must leave the VM STOPPED within this (BREV_STOP_TIMEOUT_S, default 600 = 10 min), else the VM
 * is deleted (the cron checks). */
export const stopTimeoutMs = (env: Env) => envSeconds(env.BREV_STOP_TIMEOUT_S, 600, 60);
/** A VM still DEPLOYING when its stop came is waited for this long (BREV_DEPLOYING_WAIT_S, default 900) before it is
 * deleted (the cron sends the stop once it leaves DEPLOYING). */
export const deployingWaitMs = (env: Env) => envSeconds(env.BREV_DEPLOYING_WAIT_S, 900, 60);

type StopSent = { ok: true; confirmed: boolean } | { ok: false; kind: StopRefusal; err: string };
/** Sends the stop: up to STOP_TRIES tries for a transient error, none more for "does not support stop" or "instance not
 * found". A stop Brev refused for a VM it shows STOPPED / STOPPING (e.g. already stopping) counts as accepted; one it
 * shows DEPLOYING as not provisioned. `confirmed`: Brev shows it STOPPED already. */
async function sendStop(env: Env, workspaceId: string): Promise<StopSent> {
  let err = "";
  for (let i = 0; i < STOP_TRIES; i++) {
    if (i) await sleepMs(stopRetryMs(env));
    try {
      await brev.stop(env, workspaceId);
      const w = await brev.get(env, workspaceId).catch(() => null);
      return { ok: true, confirmed: w?.status === "STOPPED" };
    } catch (e) {
      err = (e as Error).message;
      const kind = stopRefusal(err);
      if (kind !== "other") return { ok: false, kind, err };
    }
  }
  const w = await brev.get(env, workspaceId).catch(() => null);
  if (w?.status === "STOPPED" || w?.status === "STOPPING") return { ok: true, confirmed: w.status === "STOPPED" };
  if (w?.status === "DEPLOYING") return { ok: false, kind: "not_provisioned", err };
  return { ok: false, kind: "other", err };
}

/** stop_pending: still running; deleted: the stop failed and the VM was deleted. */
export type StopOutcome = "parked" | "held" | "stop_pending" | "deleted";
/** The note of a "stop failed: deleted" record (the cron's info alert brev_stop_deleted finds it by it). */
const STOP_DELETED_PREFIX = "stop failed (";
/**
 * Owner rule 2026-10-13: a Brev VM whose stop does not succeed is deleted (fv- names and recorded rows only): marked
 * deleted with the reason, its tunnel released (markBrevDeleted), its cluster row closed, logged; the cron raises the
 * info alert brev_stop_deleted from the record. A delete Brev refuses leaves it stop_pending (the cron tries the stop,
 * then the delete, again).
 */
async function deleteFailedStop(env: Env, row: BrevInstance, reason: string, log: (m: string) => void): Promise<"deleted" | "stop_pending"> {
  if (!FV_NAME_RE.test(row.name)) throw new Error(`${row.name}: its stop failed (${reason}) but it is not an fv- name: not deleted`);
  try {
    await brev.remove(env, row.workspace_id);
  } catch (e) {
    const why = `stop failed (${reason}); its delete failed too (${brevErrText((e as Error).message)}): the cron tries again`;
    await setState(env, row.workspace_id, "stop_pending", { note: why.slice(0, 300), stop_tried_at: row.stop_tried_at ?? now() });
    log(`${row.pod_id}: WARNING: ${why}`);
    return "stop_pending";
  }
  await markBrevDeleted(env, row.workspace_id, "deleted", `${STOP_DELETED_PREFIX}${reason}): deleted`.slice(0, 300));
  await env.DB.prepare("UPDATE cluster_pods SET deleted_at = ?, status = 'deleted' WHERE pod_id = ? AND deleted_at IS NULL").bind(now(), row.pod_id).run();
  log(`${row.pod_id}: stop failed (${reason}): deleted (owner rule 2026-10-13: a Brev VM whose stop does not succeed is deleted; its disk and weights went with it)`);
  return "deleted";
}
/**
 * Stops a live / restarting instance and records it. Brev accepting the stop: parked with `trees` (its weights
 * completed), else held with `note`, never deleted; until Brev shows it STOPPED the record keeps the stop's time
 * (stop_tried_at) and the cron deletes it if it is not STOPPED BREV_STOP_TIMEOUT_S later. Brev refusing it ("does not
 * support stop", or another error STOP_TRIES times): deleted (deleteFailedStop). Still DEPLOYING ("instance not
 * found"): stop_pending (the cron stops it once it leaves DEPLOYING, or deletes it after BREV_DEPLOYING_WAIT_S).
 */
async function stopAndRecord(env: Env, row: BrevInstance, rec: { parked: TreeRevs } | { note: string }, log: (m: string) => void): Promise<StopOutcome> {
  const s = await sendStop(env, row.workspace_id);
  if (!s.ok) {
    if (s.kind === "not_provisioned") {
      // The DEPLOYING wait counts from the first stop request (a stop_pending row keeps its time).
      const since = row.state === "stop_pending" && row.stop_tried_at ? row.stop_tried_at : now();
      const why = `stop_pending: still DEPLOYING when its stop came (Brev: ${brevErrText(s.err)}): stopped once it leaves DEPLOYING, deleted if still DEPLOYING at ${hhmm(since + deployingWaitMs(env))} (BREV_DEPLOYING_WAIT_S) or if its stop fails${"note" in rec ? `; ${rec.note}` : ""}`;
      await setState(env, row.workspace_id, "stop_pending", { note: why.slice(0, 300), stop_tried_at: since, ...("parked" in rec ? { trees: rec.parked } : {}) });
      log(`${row.pod_id}: ${why}`);
      return "stop_pending";
    }
    const reason = s.kind === "not_stoppable" ? `Brev cannot stop ${row.instance_type}: ${brevErrText(s.err)}` : `Brev refused the stop ${STOP_TRIES} times: ${brevErrText(s.err)}`;
    return deleteFailedStop(env, row, reason, log);
  }
  const pending = s.confirmed ? null : now();
  if ("parked" in rec) {
    await setState(env, row.workspace_id, "parked", { parked_at: now(), trees: rec.parked, note: null, stop_tried_at: pending });
    return "parked";
  }
  await setState(env, row.workspace_id, "held", { note: rec.note.slice(0, 300), parked_at: now(), stop_tried_at: pending });
  return "held";
}
/** A restart that failed or timed out: stop it again and hold it (reused by a later launch of its type after
 * heldReuseAfterMs, or deleted by the owner). Its weights stay. A stop that fails deletes it. */
export async function holdFailed(env: Env, row: BrevInstance, why: string, log: (m: string) => void = () => {}): Promise<StopOutcome> {
  return stopAndRecord(env, row, { note: `restart failed: ${why}` }, log);
}
export type Release = StopOutcome | "none";
/**
 * Releases a Brev pod. Every stop (`manual` unset) stops the VM: parked when its weights completed (its trees
 * recorded), else held with a note (a never-ready boot: `why`; a warm restart that never came up). An already
 * parked / held one gets its stop again (the down's verify); one in the owner's delete counts as gone; a stop_pending
 * one is the cron's. A stop that does not succeed deletes the VM (owner rule 2026-10-13: "deleted"); one still
 * DEPLOYING is stop_pending. `manual` (the owner's `pod delete` of a standalone pod) deletes the VM. Returns "none"
 * when fv-control has no record of it (the caller decides).
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
  // Its stop came while it was DEPLOYING: the cron stops it once it leaves DEPLOYING (or deletes it).
  if (row.state === "stop_pending") return "stop_pending";
  // live / restarting, and the legacy stop_failed / running_unstoppable (their stop is sent again: a failure deletes).
  const why = o.why ? ` (${o.why})` : "";
  const keep = `reused by a later launch of ${row.instance_type}, or delete it: fv-control.sh brev parked`;
  if (row.restarted_at && !(await weightsDone(env, podId))) {
    const r = await stopAndRecord(env, row, { note: `restart failed: it did not come up after the restart${why}` }, log);
    if (r === "held") log(`${podId}: the restarted instance never came up: stopped and held, not deleted (${keep})`);
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
/** A Brev pod fv-control has no keep-on-stop record of (made before it): stopped and recorded held (deleted if its stop
 * fails). A workspace fv-control already recorded deleted / gone is not adopted again: "deleted". */
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
  if (row.state === "deleted" || row.state === "deleting" || row.state === "gone") return "deleted";
  const r = await stopAndRecord(env, row, { note: `no keep-on-stop record: its weights are unknown (${why})` }, log);
  if (r === "held") log(`${x.pod_id}: stopped and held, not deleted (no keep-on-stop record; reused by a later launch of ${x.instance_type}, or delete it: fv-control.sh brev parked)`);
  return r;
}
/** Workspace ids of records Brev should still list (live / parked / held / stop_pending and the legacy states) that it
 * did not, changed before `before`. */
export async function brevUnlisted(env: Env, listed: Set<string>, before: number): Promise<string[]> {
  const rows = (await env.DB.prepare(`SELECT workspace_id FROM brev_instances WHERE (state IN ('live', 'parked', 'held') OR state IN ${PENDING_STATES}) AND updated_at < ?`).bind(before).all<{ workspace_id: string }>()).results || [];
  return rows.map((r) => r.workspace_id).filter((id) => !listed.has(id));
}
/** Whether a pod is parked or held (a stopped VM is then the expected end state of a stop). */
export async function isParked(env: Env, podId: string): Promise<boolean> {
  const r = await brevRow(env, podId);
  return r?.state === "parked" || r?.state === "held";
}
/** Whether a stop of a Brev pod is done for its cluster: parked / held and Brev shows it stopped, or its stop is not yet
 * confirmed (the cron confirms it or deletes the VM); stop_pending (the cron's); or deleted by a failed stop.
 * `providerState`: the instance's state. */
export async function brevReleased(env: Env, podId: string, providerState: string | undefined): Promise<boolean> {
  const r = await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE pod_id = ? ORDER BY updated_at DESC LIMIT 1`).bind(podId).first<BrevInstance>();
  if (!r) return false;
  if (r.state === "stop_pending" || r.state === "deleted" || r.state === "deleting") return true;
  return (r.state === "parked" || r.state === "held") && (providerState === "stopped" || !!r.stop_tried_at);
}
/** Brev VMs whose stop has not succeeded yet and that still bill their GPU $/hr: stop_pending (and the legacy states),
 * and parked / held ones whose stop Brev has not shown STOPPED. `dph`: vmDph. The collector bills them (owner
 * brev:stopping) while Brev lists them not stopped. */
export async function brevStopping(env: Env): Promise<{ row: BrevInstance; dph: number | null }[]> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PENDING_STATES} OR (state IN ${PARKED_STATES} AND stop_tried_at IS NOT NULL)`).all<BrevInstance>()).results || [];
  const out: { row: BrevInstance; dph: number | null }[] = [];
  for (const row of rows) out.push({ row, dph: await vmDph(env, row) });
  return out;
}
const STATUS_OF: Record<string, string> = { stopped: "STOPPED", running: "RUNNING", starting: "STARTING", failed: "FAILURE" };
/**
 * The cron's pass over stops that have not succeeded yet (owner rule 2026-10-13). `listed`: workspace id -> its status
 * in Brev's list (Brev's raw status, e.g. DEPLOYING; the provider states stopped / running / starting / failed are read
 * too); an unlisted one is left to brevUnlisted.
 *   parked / held with a stop not yet confirmed: STOPPED confirms it (kept); not STOPPED BREV_STOP_TIMEOUT_S after the
 *     stop: deleted ("still <status> N min after its stop").
 *   stop_pending: STOPPED: held; still DEPLOYING: waited for (alert brev_stop_pending) until BREV_DEPLOYING_WAIT_S after
 *     the stop request, then deleted; STOPPING: held (its stop is confirmed or timed out like any other); anything
 *     else: the stop now (parked / held, or deleted when it fails).
 *   stop_failed / running_unstoppable (legacy): their stop already failed: deleted (held when Brev shows them STOPPED).
 * Every "stop failed: deleted" record of the last 24 h raises the info alert brev_stop_deleted (from any path).
 */
export async function settleBrevStops(env: Env, listed: Map<string, string>, t = now()): Promise<{ alerts: AlertIn[]; actions: string[] }> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PENDING_STATES} OR (state IN ${PARKED_STATES} AND stop_tried_at IS NOT NULL)`).all<BrevInstance>()).results || [];
  const alerts: AlertIn[] = [];
  const actions: string[] = [];
  const log = (m: string) => actions.push(m);
  for (const r of rows) {
    const raw = listed.get(r.workspace_id);
    if (raw === undefined) continue; // brevUnlisted marks it gone
    const st = (STATUS_OF[raw] ?? raw).toUpperCase();
    if (r.state === "parked" || r.state === "held") {
      if (st === "STOPPED") {
        await env.DB.prepare("UPDATE brev_instances SET stop_tried_at = NULL WHERE workspace_id = ? AND state = ?").bind(r.workspace_id, r.state).run();
        continue;
      }
      const waited = t - (r.stop_tried_at ?? t);
      if (waited >= stopTimeoutMs(env)) await deleteFailedStop(env, r, `still ${st} ${Math.round(waited / 60000)} min after its stop (BREV_STOP_TIMEOUT_S ${stopTimeoutMs(env) / 1000})`, log);
      continue;
    }
    if (st === "STOPPED") {
      await setState(env, r.workspace_id, "held", { note: `stopped after its stop was pending (${(r.note || "").slice(0, 220)})`.slice(0, 300), parked_at: t, stop_tried_at: null });
      log(`brev ${r.name}: Brev shows it STOPPED: held`);
      continue;
    }
    if (r.state !== "stop_pending") {
      await deleteFailedStop(env, r, `its stop had already failed (${r.state}, before the owner rule of 2026-10-13: ${(r.note || "").slice(0, 120)})`, log);
      continue;
    }
    if (st === "DEPLOYING") {
      const since = r.stop_tried_at ?? r.updated_at;
      if (t - since >= deployingWaitMs(env)) {
        await deleteFailedStop(env, r, `still DEPLOYING ${Math.round((t - since) / 60000)} min after its stop was requested (BREV_DEPLOYING_WAIT_S ${deployingWaitMs(env) / 1000})`, log);
        continue;
      }
      const dph = await vmDph(env, r);
      alerts.push({
        key: `brev_stop_pending:${r.workspace_id}`,
        kind: "brev_stop_pending",
        severity: "warn",
        target: r.pod_id,
        message: `Brev ${r.name} (${r.instance_type}) is still DEPLOYING at ${money(dph)}: its stop is sent once it leaves DEPLOYING; deleted if it is still DEPLOYING at ${hhmm(since + deployingWaitMs(env))} (BREV_DEPLOYING_WAIT_S) or its stop fails (owner rule 2026-10-13)`,
      });
      continue;
    }
    if (st === "STOPPING") {
      await setState(env, r.workspace_id, "held", { note: `stopping after its stop was pending (${(r.note || "").slice(0, 220)})`.slice(0, 300), parked_at: t, stop_tried_at: t });
      continue;
    }
    // It left DEPLOYING: the stop now (held, or deleted when the stop fails).
    const out = await stopAndRecord(env, { ...r, state: "stop_pending" }, { note: `stopped once it left DEPLOYING (${(r.note || "").replace(/^stop_pending: /, "").slice(0, 200)})` }, log);
    if (out === "held") log(`brev ${r.name}: stopped once it left DEPLOYING: held`);
  }
  // The info alert of every VM a failed stop deleted (inline or above) in the last 24 h.
  const gone = (await env.DB.prepare("SELECT workspace_id, pod_id, name, instance_type, note FROM brev_instances WHERE state = 'deleted' AND note LIKE ? AND updated_at > ?").bind(`${STOP_DELETED_PREFIX}%`, t - 86400_000).all<BrevInstance>()).results || [];
  for (const g of gone)
    alerts.push({ key: `brev_stop_deleted:${g.workspace_id}`, kind: "brev_stop_deleted", severity: "info", target: g.pod_id, message: `Brev ${g.name} (${g.instance_type}): ${g.note} (owner rule 2026-10-13: a Brev VM whose stop does not succeed is deleted)` });
  return { alerts, actions };
}

/** Starts sent again after a start Brev failed for capacity (BREV_CAPACITY_RETRIES, default 3). */
export const capacityRetries = (env: Env) => {
  const n = Number(env.BREV_CAPACITY_RETRIES ?? 3);
  return Math.max(0, Number.isFinite(n) ? Math.floor(n) : 3);
};
/** Seconds between them, and after the last one before the launch gives the VM up (BREV_CAPACITY_RETRY_S, default 75:
 * 3 tries and the final check within ~5 min). */
export const capacityRetryMs = (env: Env) => envSeconds(env.BREV_CAPACITY_RETRY_S, 75, 0);
/** The note of a parked VM given back after its restart failed for capacity (warmCandidates skips it for
 * heldReuseAfterMs, so the launch's fresh VM is not this one again). */
export const NO_CAPACITY_PREFIX = "no capacity to restart";
/**
 * The `up` wait's view of a warm restart: still starting (fine until the timeout), or timed out (fatal, replace).
 * `brevSays` is Brev's status, statusMessage and health this poll: every poll's detail carries it, with the latest start
 * task of the workspace (environment.tasks[]: why a start did not run; its statusMessage is empty then; truncated).
 * A start that failed for capacity (InsufficientCapacity, "zone has insufficient resources") is sent again up to
 * BREV_CAPACITY_RETRIES times BREV_CAPACITY_RETRY_S apart; still failing, the VM goes back to parked (its stop had
 * succeeded: never deleted, owner rule 2026-10-13) and the restart is fatal with replace (the launch makes a fresh VM).
 * Otherwise a workspace still STOPPED BREV_RESTART_RETRY_S after its start (its start did not take) gets the start once
 * more.
 */
export async function restartDiag(env: Env, podId: string, state: string, brevSays: string | null = null): Promise<{ phase: string; detail: string; fatal: boolean; replace?: boolean } | null> {
  const row = await brevRow(env, podId);
  if (!row?.restarted_at || row.state !== "live" || (await weightsDone(env, podId))) return null;
  const limit = Math.max(30, Number(env.BREV_RESTART_TIMEOUT_S || "900")) * 1000;
  const t = now();
  const waited = t - row.restarted_at;
  // The latest start task (one GET; the list may not carry environment.tasks).
  const task = state === "stopped" || state === "failed" || state === "starting" ? ((await brev.get(env, row.workspace_id).catch(() => null))?.lastStart ?? null) : null;
  const tl = brevTaskLine(task);
  const saysAll = [brevSays, tl && !(brevSays || "").includes(tl) ? tl : null].filter(Boolean).join("; ");
  const says = saysAll ? `; Brev: ${saysAll.slice(0, 400)}` : "";
  // A start Brev could not run for lack of capacity in the zone: a few quick starts, then a fresh VM.
  if (task?.capacity && (state === "stopped" || state === "failed")) {
    const n = row.capacity_retries || 0;
    const since = row.capacity_retry_at ?? row.restarted_at;
    if (t - since < capacityRetryMs(env)) return { phase: "restarting", detail: `parked ${row.name}: no capacity for its start (${n}/${capacityRetries(env)} starts sent again)${says}`, fatal: false };
    if (n < capacityRetries(env)) {
      let ans: string;
      try {
        const w = await brev.start(env, row.workspace_id);
        ans = `start sent again ${n + 1}/${capacityRetries(env)} (Brev answered ${w ? brevStatusLine(w) : "no workspace"})`;
      } catch (e) {
        ans = `start sent again ${n + 1}/${capacityRetries(env)}, refused: ${brevErrText((e as Error).message)}`;
      }
      await env.DB.prepare("UPDATE brev_instances SET capacity_retries = ?, capacity_retry_at = ?, note = ?, updated_at = ? WHERE workspace_id = ?")
        .bind(n + 1, t, `no capacity for its start: ${ans}`.slice(0, 300), t, row.workspace_id)
        .run();
      return { phase: "restarting", detail: `parked ${row.name}: no capacity for its start: ${ans.slice(0, 200)}${says}`, fatal: false };
    }
    // Still no capacity after the retries: back to parked (stopped, kept), and the launch makes a fresh VM.
    const why = `${NO_CAPACITY_PREFIX} ${row.name}: ${task.message || "InsufficientCapacity"} (${n} starts sent again over ${Math.round(waited / 60000)} min)`;
    await env.DB.prepare("UPDATE brev_instances SET state = 'parked', parked_at = ?, note = ?, stop_tried_at = NULL, updated_at = ? WHERE workspace_id = ? AND state = 'live'")
      .bind(t, `${why}: kept parked`.slice(0, 300), t, row.workspace_id)
      .run();
    return { phase: "restart no capacity", detail: `${why}: left parked (its stop had succeeded: not deleted); a fresh VM is made for the launch`, fatal: true, replace: true };
  }
  if (state === "failed") return { phase: "restart failed", detail: `Brev reports an error on the restarted ${row.name}${says}`, fatal: true, replace: true };
  if (state === "stopped" || state === "starting") {
    if (waited > limit) return { phase: "restart timed out", detail: `${row.name} not running after ${Math.round(waited / 1000)} s (BREV_RESTART_TIMEOUT_S ${limit / 1000})${says}`, fatal: true, replace: true };
    let retry = "";
    const rm = restartRetryMs(env);
    if (state === "stopped" && /^STOPPED\b/.test(brevSays || "") && rm && waited > rm && !/start sent again/.test(row.note || "")) {
      try {
        const ans = await brev.start(env, row.workspace_id);
        retry = `; still STOPPED ${Math.round(waited / 1000)} s after its start: start sent again (Brev answered ${ans ? brevStatusLine(ans) : "no workspace"})`;
      } catch (e) {
        retry = `; still STOPPED ${Math.round(waited / 1000)} s after its start: start sent again, refused: ${(e as Error).message.slice(0, 160)}`;
      }
      await env.DB.prepare("UPDATE brev_instances SET note = ?, updated_at = ? WHERE workspace_id = ?").bind(`${row.note || ""}${retry}`.slice(-300), now(), row.workspace_id).run();
    }
    return { phase: "restarting", detail: `parked ${row.name} starting (${Math.round(waited / 1000)} s)${says}${retry}`, fatal: false };
  }
  return null;
}

/** The cron's view of warm restarts that do not come up: a restarted instance (live, not ready) that Brev still lists as
 * STOPPED BREV_RESTART_ALERT_S after its start raises brev_restart_stuck with Brev's status and message and its latest
 * start task (environment.tasks[], truncated). `listed`: workspace id -> Brev's status line (status: statusMessage).
 * Changes nothing. */
export async function restartStuckAlerts(env: Env, listed: Map<string, string>, t = now()): Promise<AlertIn[]> {
  const rows = (await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state = 'live' AND restarted_at IS NOT NULL`).all<BrevInstance>()).results || [];
  const out: AlertIn[] = [];
  for (const r of rows) {
    const st = listed.get(r.workspace_id);
    if (!st || !/^STOPP/.test(st) || t - r.restarted_at! < restartAlertMs(env) || (await weightsDone(env, r.pod_id))) continue;
    const tl = brevTaskLine((await brev.get(env, r.workspace_id).catch(() => null))?.lastStart ?? null);
    const says = [st.slice(0, 200), tl && !st.includes(tl) ? tl : null].filter(Boolean).join("; ");
    out.push({
      key: `brev_restart_stuck:${r.workspace_id}`,
      kind: "brev_restart_stuck",
      severity: "warn",
      target: r.pod_id,
      message: `Brev ${r.name} (${r.instance_type}) is still ${says} ${Math.round((t - r.restarted_at!) / 60000)} min after fv-control started it for a launch (${(r.note || "no answer recorded").slice(0, 200)}). The launch replaces it after BREV_RESTART_TIMEOUT_S (sooner when the start task says InsufficientCapacity); check the instance in the Brev console (capacity for its type in its region is the usual cause).`,
    });
  }
  return out;
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
    message: `NVIDIA Brev: ${why}: ${list}; storage $${total.toFixed(2)}/day in all. Not deleted (fv-control never deletes a stopped Brev VM on its own): the next launch of a type reuses them, or delete one (Pods page, fv-control.sh brev delete-parked <id>)`,
  };
}
/** Deletes one parked / held (or stop_pending, or a legacy stop_failed / running_unstoppable: still running) instance:
 * the owner's action only (API, CLI, UI); refused for anything else. A failed delete leaves a running one in the state
 * it had. */
export async function deleteParked(env: Env, workspaceId: string, why: string): Promise<string> {
  const before = await brevRowById(env, workspaceId);
  const r = await env.DB.prepare(`UPDATE brev_instances SET state = 'deleting', updated_at = ? WHERE workspace_id = ? AND (state IN ${PARKED_STATES} OR state IN ${PENDING_STATES})`).bind(now(), workspaceId).run();
  const row = await brevRowById(env, workspaceId);
  if (Number(r.meta?.changes ?? 0) !== 1 || !row || !before) throw new HttpError(404, `no parked Brev instance ${workspaceId} (only parked / held / stop-pending ones fv-control made)`);
  const running = !isKept(before);
  try {
    await brev.remove(env, workspaceId);
  } catch (e) {
    await setState(env, workspaceId, running ? before.state : "held", { note: `delete failed: ${(e as Error).message.slice(0, 200)}${running && before.note ? `; ${before.note}` : ""}`.slice(0, 300) });
    throw e;
  }
  await markBrevDeleted(env, workspaceId, "deleted", why.slice(0, 300));
  return `brev ${row.name}: ${running ? "running" : "parked"} instance deleted (${why})`;
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
/** Parked or held (stopped and kept), not a stop still pending. */
const isKept = (r: Pick<BrevInstance, "state">) => r.state === "parked" || r.state === "held";
/** The policy line the UI and `fv-control.sh brev parked` show. */
export const parkPolicyLine = (pol: Pick<Policies, "brev_park_max" | "brev_park_max_days">) =>
  `stopped Brev VMs are kept, never deleted by fv-control (owner rule 2026-10-11): reused by the next launch of their type, or deleted by you; a VM whose stop does not succeed is deleted instead (owner rule 2026-10-13); alert past ${pol.brev_park_max} parked or ${pol.brev_park_max_days} days each`;
export async function parkedView(env: Env) {
  const pol = await policies(env);
  const rows = [...(await parkedRows(env)), ...((await env.DB.prepare(`SELECT ${COLS} FROM brev_instances WHERE state IN ${PENDING_STATES} ORDER BY parked_at DESC`).all<BrevInstance>()).results || [])];
  const t = now();
  const cool = heldReuseAfterMs(env);
  const running = (r: BrevInstance) => !isKept(r);
  const reuse = (r: BrevInstance) =>
    r.state === "stop_pending"
      ? `stop pending: stopped once it leaves DEPLOYING, deleted if still DEPLOYING at ${hhmm((r.stop_tried_at ?? r.updated_at) + deployingWaitMs(env))} or if its stop fails`
      : running(r)
        ? "its stop failed: the cron deletes it (owner rule 2026-10-13)"
        : r.stop_tried_at
          ? `stopping: kept once Brev shows it STOPPED, deleted if not by ${hhmm(r.stop_tried_at + stopTimeoutMs(env))}`
          : r.state === "held" && t - r.updated_at < cool
          ? `reusable from ${new Date(r.updated_at + cool).toISOString().slice(11, 16)}Z (Unpark: now)`
          : `reusable: the next launch of ${r.instance_type}`;
  const parked: {
    workspace_id: string;
    pod_id: string;
    name: string;
    state: BrevState;
    instance_type: string;
    location: string | null;
    trees: TreeRevs;
    disk_gb: number | null;
    storage_usd_per_hr: number;
    storage_usd_per_day: number;
    running_usd_per_hr: number | null;
    running_usd_per_day: number;
    parked_at: number | null;
    parked_days: number | null;
    over_age: boolean;
    reuse: string;
    restarts: number;
    note: string | null;
  }[] = [];
  for (const r of rows) {
    const dph = running(r) ? await vmDph(env, r) : null;
    parked.push({
      workspace_id: r.workspace_id,
      pod_id: r.pod_id,
      name: r.name,
      state: r.state,
      instance_type: r.instance_type,
      location: r.location,
      trees: parseJson<TreeRevs>(r.trees, {}),
      disk_gb: r.disk_gb,
      storage_usd_per_hr: running(r) ? 0 : storageDph(r),
      storage_usd_per_day: running(r) ? 0 : storageDph(r) * 24,
      // Still running (its stop pending, or a legacy failed stop the cron deletes): its full GPU $/hr.
      running_usd_per_hr: dph,
      running_usd_per_day: (dph ?? 0) * 24,
      parked_at: r.parked_at,
      parked_days: r.parked_at ? (t - r.parked_at) / 86400_000 : null,
      over_age: !running(r) && !!r.parked_at && t - r.parked_at > pol.brev_park_max_days * 86400_000,
      reuse: reuse(r),
      restarts: r.restarts,
      note: r.note,
    });
  }
  return {
    policy: { brev_park_max: pol.brev_park_max, brev_park_max_days: pol.brev_park_max_days, auto_delete: false, line: parkPolicyLine(pol) },
    parked,
    over_limit: parked.filter((p) => p.state === "parked" || p.state === "held").length > pol.brev_park_max || parked.some((p) => p.over_age),
    storage_usd_per_day: parked.reduce((s, p) => s + p.storage_usd_per_day, 0),
    running_usd_per_day: parked.reduce((s, p) => s + p.running_usd_per_day, 0),
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
