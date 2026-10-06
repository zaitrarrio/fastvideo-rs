// The per-minute cron: one GraphQL call for every pod of the account with
// its runtime metrics, the edge's families view (jobs and readiness of every
// edge front), then costs, idle tracking, alerts, the backstops
// and retention (docs/control/README.md "Cost model", "Alerts").
import { attribute, policies, syncAlerts, type AlertIn } from "./alerts";
import { buildPodHealth, buildPodVerdict, stopBuildPod } from "./buildpod";
import { buildPodsTick, managedPodNames } from "./buildpods";
import { collectCloudrift, type CloudriftCollect } from "./collector-cloudrift";
import { edgeCfg, edgeFamilies, edgeWorkers } from "./cluster/ops";
import { isEdge } from "./cluster/spec";
import { cancelOp, currentOp, startOp } from "./cluster/control";
import { isStandalone, listClusters, ownerOf, type Cluster } from "./cluster/store";
import { checkPod } from "./podlogs";
import { defaults, type Env } from "./env";
import { writeAccountSample, writePodSamples, type PodSample } from "./metrics";
import { runpod, type RunpodPod } from "./runpod";
import { audit, getSetting, now, putSetting, utcDay } from "./util";

/** Pods whose Runpod log is copied per cron pass (one hapi call each). */
const LOG_CAPTURE_MAX = 25;
const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

export interface CollectResult {
  at: number;
  balance: number;
  spend_per_hr: number;
  pods: number;
  running: number;
  alerts: number;
  actions: string[];
  cloudrift?: Omit<CloudriftCollect, "alerts" | "kinds" | "actions">;
}

export async function collect(env: Env): Promise<CollectResult> {
  const t = now();
  const last = await getSetting<{ at: number }>(env, "collector", { at: 0 });
  const dtMs = last.at ? Math.min(Math.max(t - last.at, 0), 5 * 60_000) : 60_000;
  await putSetting(env, "collector", { at: t }, "cron");
  const pol = await policies(env);
  const floor = defaults.balanceFloor(env);
  const { balance, spendPerHr, pods } = await runpod.pods(env);
  await env.DB.prepare("INSERT OR REPLACE INTO balance_samples (at, balance, spend_per_hr) VALUES (?, ?, ?)").bind(t, balance, spendPerHr).run();
  writeAccountSample(env, balance, spendPerHr);

  // Which pods are the controller's.
  const clusters = await listClusters(env);
  const byId = new Map(clusters.map((c) => [c.id, c]));
  // Recently deleted controller pods too: Runpod lists a deleted pod for a minute or more, and its
  // cost must stay with its cluster (not move to external:<prefix> with its whole cost_daily row).
  const rows = await env.DB.prepare("SELECT pod_id, cluster_id, role, pool, url, created_at, deleted_at FROM cluster_pods WHERE deleted_at IS NULL OR deleted_at > ?")
    .bind(t - 6 * 3600_000)
    .all<{ pod_id: string; cluster_id: string; role: string; pool: string | null; url: string | null; created_at: number; deleted_at: number | null }>();
  const ctl = new Map((rows.results || []).map((r) => [r.pod_id, r]));
  // Build pods fv-control manages (buildpods.ts): owner build-pod:<name>.
  const managed = await managedPodNames(env);

  // Jobs per worker and health.
  const jobs = new Map<string, { running: number; queued: number; ready: boolean; healthy: boolean; sha?: string }>();
  const health = new Map<string, string>();
  // Edge clusters: each front's jobs from the edge's families view (one read for every edge cluster).
  if (edgeCfg(env) && clusters.some((c) => isEdge(c.spec) && ["running", "starting"].includes(c.status))) {
    try {
      for (const [pod, w] of edgeWorkers(await edgeFamilies(env))) {
        jobs.set(pod, { running: w.held, queued: 0, ready: w.ready, healthy: true, sha: w.sha });
        health.set(pod, w.ready ? "ready" : "loading");
      }
    } catch {
      /* the edge is down: pods keep their Runpod-side health */
    }
  }

  const prev = await env.DB.prepare("SELECT pod_id, idle_since, health, first_seen FROM pods WHERE gone_at IS NULL AND provider = 'runpod'").all<{ pod_id: string; idle_since: number | null; health: string | null; first_seen: number }>();
  const prevBy = new Map((prev.results || []).map((r) => [r.pod_id, r]));
  const samples: PodSample[] = [];
  const stmts: D1PreparedStatement[] = [];
  const day = utcDay(t);
  const seen = new Set<string>();
  const podInfo: { p: RunpodPod; owner: string; clusterId: string | null; idleSince: number | null; running: boolean; gpuUtil: number | null }[] = [];
  for (const p of pods) {
    seen.add(p.id);
    const row = ctl.get(p.id);
    const c = row ? byId.get(row.cluster_id) : undefined;
    const bp = managed.get(p.id);
    const owner = c ? ownerOf(c) : bp ? `build-pod:${bp}` : attribute(p.name, pol.attribution);
    const running = p.desiredStatus === "RUNNING";
    const gpus = p.runtime?.gpus || [];
    const gpuUtil = avg(gpus.map((g) => Number(g.gpuUtilPercent ?? 0)));
    const gpuMem = avg(gpus.map((g) => Number(g.memoryUtilPercent ?? 0)));
    const cpu = p.runtime?.container?.cpuPercent ?? null;
    const mem = p.runtime?.container?.memoryPercent ?? null;
    const j = jobs.get(p.id);
    const isGpu = (p.gpuCount ?? gpus.length) > 0;
    const idleNow = running && isGpu && gpuUtil !== null && gpuUtil < pol.idle_gpu_pct && (j ? j.running === 0 : true);
    const pv = prevBy.get(p.id);
    const idleSince = idleNow ? (pv?.idle_since ?? t) : null;
    const h = health.get(p.id) ?? (running ? (p.runtime ? "unknown" : "loading") : "stopped");
    podInfo.push({ p, owner, clusterId: row?.cluster_id ?? null, idleSince, running, gpuUtil });
    stmts.push(
      env.DB.prepare(
        `INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, cost_per_hr, gpu, gpu_count, dc, image, uptime_s, gpu_util, gpu_mem, cpu, mem, jobs_running, jobs_queued, health, build_sha, idle_since, first_seen, last_seen, gone_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
         ON CONFLICT (pod_id) DO UPDATE SET name = excluded.name, owner = excluded.owner, cluster_id = excluded.cluster_id, desired_status = excluded.desired_status,
           cost_per_hr = excluded.cost_per_hr, gpu = excluded.gpu, gpu_count = excluded.gpu_count, dc = excluded.dc, image = excluded.image, uptime_s = excluded.uptime_s,
           gpu_util = excluded.gpu_util, gpu_mem = excluded.gpu_mem, cpu = excluded.cpu, mem = excluded.mem, jobs_running = excluded.jobs_running, jobs_queued = excluded.jobs_queued,
           health = excluded.health, build_sha = COALESCE(excluded.build_sha, pods.build_sha), idle_since = excluded.idle_since, last_seen = excluded.last_seen, gone_at = NULL`,
      ).bind(
        p.id,
        p.name,
        owner,
        row?.cluster_id ?? null,
        p.desiredStatus,
        Number(p.costPerHr ?? 0),
        p.machine?.gpuDisplayName ?? null,
        p.gpuCount ?? gpus.length,
        p.machine?.dataCenterId ?? null,
        (p.imageName || "").slice(0, 200),
        p.runtime?.uptimeInSeconds ?? null,
        gpuUtil,
        gpuMem,
        cpu,
        mem,
        j?.running ?? null,
        j?.queued ?? null,
        h,
        j?.sha ?? null,
        idleSince,
        pv?.first_seen ?? t,
        t,
      ),
    );
    if (running) {
      // Cost accrues for running pods at their $/hr over the time since the last sample.
      const usd = (Number(p.costPerHr ?? 0) * dtMs) / 3_600_000;
      stmts.push(
        env.DB.prepare(
          `INSERT INTO cost_daily (day, pod_id, cluster_id, owner, usd, minutes, idle_minutes) VALUES (?, ?, ?, ?, ?, 1, ?)
           ON CONFLICT (day, pod_id) DO UPDATE SET usd = usd + excluded.usd, minutes = minutes + 1, idle_minutes = idle_minutes + excluded.idle_minutes, owner = excluded.owner, cluster_id = excluded.cluster_id`,
        ).bind(day, p.id, row?.cluster_id ?? null, owner, usd, idleNow ? 1 : 0),
      );
      samples.push({
        at: t,
        pod_id: p.id,
        owner,
        cluster_id: row?.cluster_id ?? null,
        name: p.name,
        gpu: p.machine?.gpuDisplayName || (isGpu ? "gpu" : "cpu"),
        status: p.desiredStatus,
        cost_per_hr: Number(p.costPerHr ?? 0),
        gpu_util: gpuUtil,
        gpu_mem: gpuMem,
        cpu,
        mem,
        uptime_s: p.runtime?.uptimeInSeconds ?? null,
        jobs_running: j?.running ?? null,
        jobs_queued: j?.queued ?? null,
        idle: idleNow,
      });
    }
  }
  for (const r of prev.results || []) if (!seen.has(r.pod_id)) stmts.push(env.DB.prepare("UPDATE pods SET gone_at = ?, desired_status = 'GONE' WHERE pod_id = ?").bind(t, r.pod_id));
  // Controller pods Runpod no longer has: gone (a backstop deleted them, or someone else).
  for (const [pod, r] of ctl) if (!r.deleted_at && !seen.has(pod)) stmts.push(env.DB.prepare("UPDATE cluster_pods SET status = 'gone', deleted_at = ? WHERE pod_id = ? AND deleted_at IS NULL").bind(t, pod));
  for (let i = 0; i < stmts.length; i += 50) await env.DB.batch(stmts.slice(i, i + 50));
  await writePodSamples(env, samples);

  // Pod logs from boot (podlogs.ts): every live controller pod's Runpod log tail, whatever its image ships.
  let captured = 0;
  for (const x of podInfo) {
    if (!x.clusterId || ctl.get(x.p.id)?.deleted_at || x.p.desiredStatus === "TERMINATED" || captured >= LOG_CAPTURE_MAX) continue;
    captured++;
    await checkPod(env, x.p.id, x.clusterId, ctl.get(x.p.id)?.created_at ?? prevBy.get(x.p.id)?.first_seen ?? t, { broadcast: true, uptimeS: x.p.runtime?.uptimeInSeconds ?? null }).catch(() => null);
  }

  // ---- alerts and actions
  const alerts: AlertIn[] = [];
  const actions: string[] = [];
  for (const x of podInfo) {
    if (!x.running || x.idleSince === null) continue;
    const idleMin = (t - x.idleSince) / 60_000;
    if (idleMin >= pol.idle_min)
      alerts.push({ key: `pod_idle:${x.p.id}`, kind: "pod_idle", severity: "warn", target: x.p.id, message: `${x.p.name} (${x.owner}) idle for ${Math.round(idleMin)} min at $${x.p.costPerHr}/hr (GPU ${x.gpuUtil?.toFixed(0)}%, no running jobs)` });
    const c = x.clusterId ? byId.get(x.clusterId) : undefined;
    const limit = c?.spec.auto_stop_idle_min ?? (pol.auto_stop_idle ? pol.auto_stop_idle_min : null);
    if (c && limit && idleMin >= limit && isStandalone(c)) {
      // A standalone pod has nothing to drain to: it stops (its definition stays; start makes a new pod).
      actions.push(await stopCluster(env, c, "idle"));
      continue;
    }
    if (c && limit && idleMin >= limit) {
      const rec = Object.values(c.state.workers).flat().find((r) => r.pod === x.p.id);
      if (rec?.pool) {
        const cur = c.state.workers[rec.pool]!.length;
        try {
          await startOp(env, c.id, "scale", { pool: rec.pool, count: cur - 1, victims: [x.p.id], reason: "idle" }, "policy:auto_stop_idle");
          actions.push(`scale ${c.name}/${rec.pool} to ${cur - 1} (idle ${x.p.id})`);
          await audit(env, { actor: "policy:auto_stop_idle", action: "cluster.scale", target: c.name, after: { pool: rec.pool, count: cur - 1, pod: x.p.id } });
        } catch {
          /* an operation is running; next minute */
        }
      }
    }
  }
  for (const c of clusters) {
    // A pool the last `up` could not bring up (no stock, crash loop, image error, stuck): docs/control/README.md §4.
    if (["starting", "running", "failed"].includes(c.status))
      for (const [pool, ps] of Object.entries(c.state.pools || {}))
        if (ps.status === "no_stock" || ps.status === "failed")
          alerts.push({ key: `pool_failed:${c.id}:${pool}`, kind: "pool_failed", severity: "warn", target: c.name, message: `${c.name}: pool ${pool} ${ps.status === "no_stock" ? "has no stock" : "failed"}${ps.detail ? `: ${ps.detail.slice(0, 240)}` : ""}` });
    const dph = podInfo.filter((x) => x.clusterId === c.id && x.running).reduce((s, x) => s + Number(x.p.costPerHr || 0), 0);
    if (dph > pol.cluster_dph_max) alerts.push({ key: `cluster_dph:${c.id}`, kind: "cluster_dph", severity: "warn", target: c.name, message: `${c.name} costs $${dph.toFixed(2)}/hr (> $${pol.cluster_dph_max})` });
    for (const r of Object.values(c.state.workers).flat().concat(c.state.gateway ? [c.state.gateway] : [])) {
      const h = health.get(r.pod);
      const born = r.created * 1000;
      if (h === "down" && t - born > pol.pod_down_min * 60_000) alerts.push({ key: `pod_down:${r.pod}`, kind: "pod_down", severity: "warn", target: r.pod, message: `${c.name}: ${r.pool || "legacy gateway"} pod ${r.pod} is not answering` });
    }
    // Backstop: the deadline (each edge worker's watchdog and this cron both enforce it).
    const hasPods = !!c.state.gateway || Object.values(c.state.workers).some((l) => l.length);
    if (hasPods && c.deadline) {
      const left = c.deadline - t;
      if (left <= 0) {
        alerts.push({ key: `deadline:${c.id}`, kind: "deadline", severity: "critical", target: c.name, message: `${c.name} passed its deadline: stopping`, action: "stop" });
        actions.push(await stopCluster(env, c, "deadline backstop"));
      } else if (left < 15 * 60_000) alerts.push({ key: `deadline:${c.id}`, kind: "deadline", severity: "info", target: c.name, message: `${c.name} stops at its deadline in ${Math.round(left / 60000)} min (extend to keep it)` });
    }
  }
  // Backstop for the shared build pod (buildpod.ts): its own self-stop failed before.
  if (pol.build_pod_backstop)
    for (const x of podInfo) {
      if (!x.running || x.owner !== "external:build-pod") continue;
      const why = buildPodVerdict(x.p.runtime?.uptimeInSeconds, await buildPodHealth(env, x.p.id), pol);
      if (!why) continue;
      let result: string;
      try {
        result = await stopBuildPod(env, x.p.id);
      } catch (e) {
        result = `stop failed: ${(e as Error).message.slice(0, 160)}`;
      }
      alerts.push({ key: `build_pod:${x.p.id}`, kind: "build_pod", severity: "critical", target: x.p.id, message: `${x.p.name} ${x.p.id}: ${why}: ${result}`, action: "stop" });
      actions.push(`build pod ${x.p.id}: ${result} (${why})`);
      await audit(env, { actor: "policy:build_pod_backstop", action: "pod.stop", target: x.p.id, detail: `${why}: ${result}` });
    }
  // Managed build pods (buildpods.ts): states, per-pod backstop, floor, runners, wake on queued CI jobs.
  try {
    const bt = await buildPodsTick(env, pods, balance, { stopOnFloor: pol.stop_on_floor, floor });
    alerts.push(...bt.alerts);
    actions.push(...bt.actions);
  } catch (e) {
    alerts.push({ key: "build_pods_tick", kind: "build_pod_runner", severity: "warn", message: `build pods: ${(e as Error).message.slice(0, 200)}` });
  }
  const today = await env.DB.prepare("SELECT COALESCE(SUM(usd), 0) AS usd FROM cost_daily WHERE day = ?").bind(day).first<{ usd: number }>();
  if ((today?.usd || 0) > pol.daily_spend_max) alerts.push({ key: "daily_spend", kind: "daily_spend", severity: "warn", message: `spend today $${today!.usd.toFixed(2)} (> $${pol.daily_spend_max})` });
  if (balance < floor) {
    alerts.push({ key: "balance_floor", kind: "balance_floor", severity: "critical", message: `balance $${balance.toFixed(2)} is below the floor $${floor}`, action: pol.stop_on_floor ? "stop controller clusters" : undefined });
    if (pol.stop_on_floor) for (const c of clusters) if (c.state.gateway || Object.values(c.state.workers).some((l) => l.length)) actions.push(await stopCluster(env, c, "balance floor"));
  } else if (balance < floor + pol.balance_margin) alerts.push({ key: "balance_margin", kind: "balance_margin", severity: "warn", message: `balance $${balance.toFixed(2)} is within $${pol.balance_margin} of the floor $${floor}` });
  // Clusters with their own higher floor.
  for (const c of clusters) {
    if (!(c.state.gateway || Object.values(c.state.workers).some((l) => l.length))) continue;
    if (balance < c.spec.balance_floor && balance >= floor && pol.stop_on_floor) {
      alerts.push({ key: `cluster_floor:${c.id}`, kind: "balance_floor", severity: "critical", target: c.name, message: `balance $${balance.toFixed(2)} below ${c.name}'s floor $${c.spec.balance_floor}: stopping`, action: "stop" });
      actions.push(await stopCluster(env, c, "cluster balance floor"));
    }
  }
  // The second provider (docs/ops/cloudrift.md); its failure never stops the Runpod half.
  const cr = await collectCloudrift(env, t, dtMs, pol).catch((e) => ({ enabled: true, ok: false, error: (e as Error).message, alerts: [], kinds: [], actions: [] }) as unknown as CloudriftCollect);
  alerts.push(...cr.alerts);
  actions.push(...cr.actions);
  await syncAlerts(env, alerts, ["pod_idle", "pool_failed", "cluster_dph", "pod_down", "deadline", "daily_spend", "balance_floor", "balance_margin", "build_pod", "build_pod_spend", "build_pod_runner", ...cr.kinds]);

  // ---- retention
  await env.DB.batch([
    env.DB.prepare("DELETE FROM log_lines WHERE ts < ?").bind(t - 24 * 3600_000),
    env.DB.prepare("DELETE FROM pod_samples WHERE at < ?").bind(t - 24 * 3600_000),
    env.DB.prepare("DELETE FROM balance_samples WHERE at < ?").bind(t - 30 * 86400_000),
    env.DB.prepare("DELETE FROM rate_limits WHERE window < ?").bind(Math.floor(t / 1000 / 3600) - 48),
    env.DB.prepare("DELETE FROM alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?").bind(t - 30 * 86400_000),
    env.DB.prepare("DELETE FROM pods WHERE gone_at IS NOT NULL AND gone_at < ?").bind(t - 7 * 86400_000),
    env.DB.prepare("DELETE FROM pod_log_cursors WHERE updated_at < ?").bind(t - 7 * 86400_000),
  ]);
  const { alerts: _a, kinds: _k, actions: _x, ...crSummary } = cr;
  return { at: t, balance, spend_per_hr: spendPerHr, pods: pods.length, running: podInfo.filter((x) => x.running).length, alerts: alerts.length, actions, ...(cr.enabled ? { cloudrift: crSummary } : {}) };
}

/** Stops a cluster now (backstop / floor): a `down` operation, cancelling whatever runs; deletes the pods directly if the DO fails. */
export async function stopCluster(env: Env, c: Cluster, reason: string): Promise<string> {
  try {
    const cur = (await currentOp(env, c.id)) as { kind?: string } | null;
    if (cur?.kind === "down") return `${c.name}: already stopping`;
    if (cur) await cancelOp(env, c.id, `policy:${reason}`);
    await startOp(env, c.id, "down", { reason }, `policy:${reason}`);
    await audit(env, { actor: `policy:${reason}`, action: "cluster.stop", target: c.name });
    return `${c.name}: stop (${reason})`;
  } catch (e) {
    const ids = [...Object.values(c.state.workers).flat(), ...(c.state.gateway ? [c.state.gateway] : [])].map((r) => r.pod);
    for (const p of ids) await runpod.remove(env, p).catch(() => {});
    await audit(env, { actor: `policy:${reason}`, action: "cluster.stop.direct", target: c.name, detail: (e as Error).message, after: { deleted: ids } });
    return `${c.name}: deleted pods directly (${reason})`;
  }
}
