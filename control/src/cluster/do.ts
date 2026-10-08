// ClusterOps: one Durable Object per cluster. It runs one operation at a
// time (up, down, extend, scale, roll, restart) as a
// step machine driven by alarms, so a 30-minute rolling redeploy never
// depends on one request staying open. It also fans ingested log lines
// out to live-tail WebSockets (hibernation API).
import { bumpDoc } from "../docs";
import type { Env } from "../env";
import { resolveDigest, resolveClusterImages } from "../ghcr";
import { runpod } from "../runpod";
import { HttpError, now, scrub } from "../util";
import {
  createWorker,
  deletePod,
  patchWorker,
  projectSpend,
  workerBusy,
  workerHealth,
  workerInternal,
  desiredEnv,
  envCtx,
  newAdminToken,
  edgeFamilies,
  edgeWorkers,
  requireEdge,
} from "./ops";
import { markReady } from "../boottime";
import { checkPod } from "../podlogs";
import type { OtherProviderId } from "../enums";
import { budgetCheck, isOtherPod, isOtherPool, otherPodDiag, otherPodState, poolProvider, providerIssues, reportedUrl } from "../providers";
import type { PodRec, PoolStatus } from "./payloads";
import { checkImages } from "./preflight";
import { isEdge } from "./spec";
import { deleteClusterRow, getCluster, listClusters, podUpdate, saveSpec, saveState, secretsOf, saveSecrets, livePods, type Cluster } from "./store";
import { NO_STOCK_RETRY_MS, READY_WAIT_MS, SUMMARY_EVERY_MS, summarize, upOutcome, type PoolView } from "./upwait";

export type OpKind = "up" | "down" | "extend" | "scale" | "roll" | "restart";
export interface Op {
  id: string;
  cluster: string;
  kind: OpKind;
  params: any;
  actor: string;
  phase: string;
  data: any;
  started: number;
  errors: number;
}
type Step = { done: true; error?: string } | { delayMs: number };

/** Re-placements of a pod on a slow host (pull deadline) per pool and `up`. */
const MAX_REPLACE = 2;
const DRAIN_WAIT_MS = 15 * 60_000;
const RESTART_WAIT_MS = 15 * 60_000;

export class ClusterOps implements DurableObject {
  private logBuf: { at: number; msg: string }[] = [];
  constructor(
    private ctx: DurableObjectState,
    private env: Env,
  ) {}

  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    try {
      if (url.pathname === "/op/start" && req.method === "POST") {
        const op = (await req.json()) as Op;
        const cur = await this.ctx.storage.get<Op>("op");
        if (cur) return Response.json({ error: `operation ${cur.kind} (${cur.id}) is running` }, { status: 409 });
        op.phase = "init";
        op.data = {};
        op.started = now();
        op.errors = 0;
        await this.ctx.storage.put("op", op);
        await this.ctx.storage.setAlarm(now() + 10);
        return Response.json({ id: op.id, status: "running" }, { status: 202 });
      }
      if (url.pathname === "/op" && req.method === "GET") {
        const cur = await this.ctx.storage.get<Op>("op");
        return Response.json({ op: cur ? { id: cur.id, kind: cur.kind, phase: cur.phase, started: cur.started } : null });
      }
      if (url.pathname === "/op/cancel" && req.method === "POST") {
        const cur = await this.ctx.storage.get<Op>("op");
        if (!cur) return Response.json({ cancelled: false });
        await this.ctx.storage.delete("op");
        await this.ctx.storage.deleteAlarm();
        await this.finish(cur, "cancelled", "cancelled by " + ((await req.json().catch(() => ({}))) as any).actor);
        return Response.json({ cancelled: true, id: cur.id });
      }
      if (url.pathname === "/op/kick" && req.method === "POST") {
        if (await this.ctx.storage.get("op")) await this.ctx.storage.setAlarm(now() + 10);
        return Response.json({ ok: true });
      }
      if (url.pathname === "/tail") {
        if (req.headers.get("upgrade")?.toLowerCase() !== "websocket") return new Response("websocket expected", { status: 426 });
        const pair = new WebSocketPair();
        const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
        this.ctx.acceptWebSocket(server, [url.searchParams.get("pod") || "*"]);
        return new Response(null, { status: 101, webSocket: client });
      }
      if (url.pathname === "/broadcast" && req.method === "POST") {
        const { pod, lines } = (await req.json()) as { pod: string; lines: unknown[] };
        const msg = JSON.stringify({ pod, lines });
        let n = 0;
        for (const ws of [...this.ctx.getWebSockets(pod), ...this.ctx.getWebSockets("*")]) {
          try {
            ws.send(msg);
            n++;
          } catch {
            /* closed */
          }
        }
        return Response.json({ sent: n });
      }
      return new Response("not found", { status: 404 });
    } catch (e) {
      return Response.json({ error: scrub(this.env, (e as Error).message) }, { status: e instanceof HttpError ? e.status : 500 });
    }
  }

  webSocketMessage(ws: WebSocket, msg: string | ArrayBuffer) {
    if (msg === "ping") ws.send("pong");
  }
  webSocketClose(ws: WebSocket, code: number) {
    try {
      ws.close(code, "bye");
    } catch {
      /* already closed */
    }
  }

  private log = (msg: string) => {
    this.logBuf.push({ at: now(), msg: scrub(this.env, msg) });
  };
  private async flush(op: Op, status = "running", error?: string) {
    const buf = this.logBuf;
    this.logBuf = [];
    const row = await this.env.DB.prepare("SELECT log FROM operations WHERE id = ?").bind(op.id).first<{ log: string }>();
    const all = [...JSON.parse(row?.log || "[]"), ...buf].slice(-400);
    await this.env.DB.prepare("UPDATE operations SET log = ?, status = ?, error = ?, updated_at = ? WHERE id = ?")
      .bind(JSON.stringify(all), status, error ? scrub(this.env, error) : null, now(), op.id)
      .run();
  }
  private async finish(op: Op, status: "done" | "failed" | "cancelled", error?: string) {
    if (error) this.log(`${status}: ${error}`);
    else this.log(status);
    await this.flush(op, status, error);
  }

  async alarm(): Promise<void> {
    const op = await this.ctx.storage.get<Op>("op");
    if (!op) return;
    let res: Step;
    try {
      const c = await getCluster(this.env, op.cluster);
      res = await this.step(c, op);
      op.errors = 0;
    } catch (e) {
      op.errors++;
      const msg = (e as Error).message;
      this.log(`step ${op.phase} failed (${op.errors}/3): ${msg}`);
      if (op.errors >= 3) {
        res = { done: true, error: `${op.phase}: ${msg}` };
        if (op.kind === "roll" && op.phase !== "abort") {
          try {
            const c = await getCluster(this.env, op.cluster);
            await this.abortRoll(c, op);
          } catch (e2) {
            this.log(`abort failed: ${(e2 as Error).message}`);
          }
        }
      } else res = { delayMs: 15_000 };
    }
    if ("done" in res) {
      await this.ctx.storage.delete("op");
      await this.finish(op, res.error ? "failed" : "done", res.error);
      return;
    }
    await this.ctx.storage.put("op", op);
    await this.flush(op);
    await this.ctx.storage.setAlarm(now() + Math.max(10, res.delayMs));
  }

  private async step(c: Cluster, op: Op): Promise<Step> {
    switch (op.kind) {
      case "up":
        return this.up(c, op);
      case "down":
        return this.down(c, op);
      case "extend":
        return this.extend(c, op);
      case "scale":
        return this.scale(c, op);
      case "roll":
        return this.roll(c, op);
      case "restart":
        return this.restart(c, op);
      default:
        return { done: true, error: `unknown operation ${op.kind} (gateway-start / gateway-stop went with the gateway)` };
    }
  }

  // ---------------- up
  private async up(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    if (op.phase === "init") {
      if (c.state.gateway || Object.values(c.state.workers).some((l) => l.length)) return { done: true, error: "the cluster has pods already: stop it first" };
      if (isEdge(c.spec)) {
        requireEdge(env);
        // v1: one edge, one cluster behind it (docs/serve/edge-control-plane.md §10 Q2).
        const other = (await listClusters(env)).find((x) => x.id !== c.id && isEdge(x.spec) && (["starting", "running", "stopping"].includes(x.status) || Object.values(x.state.workers || {}).some((l) => l.length)));
        if (other) return { done: true, error: `the edge already fronts cluster ${other.name} (${other.status}): one edge cluster at a time` };
      }
      // GMI / Brev pools (providers.ts): the provider is configured, the GPU allowed, a Hub download approved.
      const pv = providerIssues(env, c.spec);
      if (pv.length) return { done: true, error: pv.map((i) => `${c.spec.pools[i.path[1] as number]?.id}: ${i.message}`).join("; ") };
      const proj = await projectSpend(env, c.spec, { hours: c.spec.cap_s / 3600 });
      this.log(`balance $${proj.balance.toFixed(2)}, account $${proj.account_spend_per_hr.toFixed(2)}/hr, cluster ~$${proj.cluster_dph.toFixed(2)}/hr; projected $${proj.projected_balance.toFixed(2)} at the deadline (floor $${proj.floor})`);
      if (!proj.ok && !op.params?.skip_price_check) return { done: true, error: proj.reasons.join("; ") };
      c.state.images = await resolveClusterImages(env, c.spec);
      if (c.spec.image.ref) c.state.image = c.state.images.gateway || Object.values(c.state.images)[0];
      this.log(`images: ${JSON.stringify(c.state.images)}`);
      // Image preflight (preflight.ts): an image that cannot run this spec is refused before any pod is paid for.
      const pf = await checkImages(env, c.spec, c.state.images);
      const revs = Object.entries(pf.revisions).filter(([, r]) => r).map(([img, r]) => `${img.split("@")[1]?.slice(7, 19) || img}=${r!.slice(0, 7)}`);
      if (revs.length) this.log(`image revisions: ${revs.join(", ")}`);
      for (const w of pf.warnings) this.log(`WARNING: ${w}`);
      if (pf.errors.length) {
        if (!op.params?.skip_image_check) return { done: true, error: `image preflight: ${pf.errors.join("; ")}` };
        this.log(`WARNING: image preflight overridden (skip_image_check): ${pf.errors.join("; ")}`);
      }
      c.state.pools = {};
      const s = await secretsOf(env, c);
      // Edge clusters use the edge's admin token (EDGE_ADMIN_TOKEN); direct
      // workers get a fresh one from us as FV_ADMIN_TOKEN
      // (docs/control/gateway-less-auth.md).
      if (isEdge(c.spec)) delete s.admin_token;
      else s.admin_token = newAdminToken();
      await saveSecrets(env, c, s);
      await saveState(env, c, { status: "starting", deadline: now() + c.spec.cap_s * 1000 });
      this.log(`deadline ${new Date(c.deadline!).toISOString()} (${c.spec.cap_s}s)`);
      op.phase = isEdge(c.spec) ? "register" : "workers";
      d.failed = [];
      return { delayMs: 10 };
    }
    if (op.phase === "register") {
      // The edge answers and takes our admin token before any GPU is paid for.
      const e = requireEdge(env);
      const view = await edgeFamilies(env);
      this.log(`edge ${e.url}: up (key epoch ${view?.key_epoch ?? "?"}, families ${Object.keys(view?.families || {}).join(", ") || "none yet"})`);
      op.phase = "workers";
      return { delayMs: 10 };
    }
    if (op.phase === "workers") {
      for (const p of c.spec.pools) {
        if (d.failed.includes(p.id)) continue;
        if ((c.state.workers[p.id]?.length || 0) < p.count) {
          const rec = await this.createFor(c, p.id);
          if (!rec) d.failed.push(p.id);
          await saveState(env, c);
          return { delayMs: 10 };
        }
      }
      op.phase = "patch";
      return { delayMs: 10 };
    }
    if (op.phase === "patch") {
      if (d.failed.length)
        this.log(`NO STOCK: ${d.failed.join(", ")} got no pod (${d.failed.map((p: string) => c.state.pools?.[p]?.detail || "no stock").join("; ")}); retried every ${NO_STOCK_RETRY_MS / 60000} min while the other pools start, else the operation fails naming them`);
      // Nothing to wait for: not one worker (no stock anywhere).
      if (!Object.values(c.state.workers).some((l) => l.length) && c.spec.pools.some((p) => p.count > 0))
        return { done: true, error: `no worker pod could be made (${d.failed.join(", ")}): stop the cluster or start it again later` };
      await saveState(env, c, { status: "running" });
      op.phase = "wait";
      d.t0 = now();
      return { delayMs: 20_000 };
    }
    if (op.phase === "wait") return this.waitReady(c, op, d.t0);
    return { done: true, error: `unknown phase ${op.phase}` };
  }

  /** createWorker for `up`, recording the pool's status (starting, or no stock with the last reason). */
  private async createFor(c: Cluster, poolId: string): Promise<PodRec | null> {
    let last = "";
    const rec = await createWorker(this.env, c, poolId, c.state.images[poolId] || c.state.image!, "workers", (m) => {
      if (/: no .* in /.test(m)) last = m.slice(poolId.length + 2);
      this.log(m);
    });
    (c.state.pools ||= {})[poolId] = rec ? { status: "starting", at: now() } : { status: "no_stock", detail: last || "no stock", at: now() };
    return rec;
  }

  /**
   * Until no pool is starting: a pool is ready when one of its pods is (a
   * ready front in the edge's families view, or a direct worker's /health
   * AVAILABLE). Every 20 s each starting pod's Runpod log is captured and
   * diagnosed (podlogs.ts): a crash loop, an image error or a pod stuck
   * without output fails it at once (deleted: the log is kept), instead of
   * a silent 30-minute wait. Pools without stock are retried while others
   * start. The operation then fails naming each pool that did not come up.
   */
  private async waitReady(c: Cluster, op: Op, t0: number): Promise<Step> {
    const env = this.env;
    const d = op.data;
    const pools = c.spec.pools.filter((p) => p.count > 0 || (c.state.workers[p.id] || []).length);
    const st = (c.state.pools ||= {});
    let fronts: ReturnType<typeof edgeWorkers> | null = null;
    if (isEdge(c.spec)) {
      try {
        fronts = edgeWorkers(await edgeFamilies(env));
      } catch (e) {
        this.log(`edge not answering (${Math.round((now() - t0) / 1000)}s): ${(e as Error).message.slice(0, 100)}`);
      }
    }
    for (const p of pools) {
      const recs = c.state.workers[p.id] || [];
      const cur: PoolStatus = st[p.id] || (st[p.id] = { status: recs.length ? "starting" : "no_stock", at: now() });
      if (cur.status === "ready" || cur.status === "failed") continue;
      if (!recs.length) continue; // no stock: retried below
      let ready = false;
      // A GMI / Brev pod's URL is the tunnel it reported (POST /ingest/v1/endpoint): into the state for the admin calls.
      for (const r of recs) if (isOtherPod(r.pod) && !r.url) r.url = (await reportedUrl(env, r.pod)) || undefined;
      for (const r of recs) {
        const ok = isEdge(c.spec) ? !!fronts?.get(r.pod)?.ready : (await workerHealth(env, r.pod)).ok;
        if (ok) {
          ready = true;
          await podUpdate(env, r.pod, { ready: true, status: "ready" });
          await markReady(env, r.pod, c.id).catch(() => {});
        }
      }
      if (ready) {
        st[p.id] = { status: "ready", at: now() };
        continue;
      }
      // Not ready yet: what does each pod's boot look like?
      const notes: string[] = [];
      for (const r of [...recs]) {
        const other = isOtherPod(r.pod);
        const rt = other ? await otherPodState(env, r.pod).catch(() => undefined) : await runpod.podRuntime(env, r.pod).catch(() => undefined);
        const diag: { phase: string; detail: string; fatal: boolean; replace?: boolean } | null =
          rt === null ? { phase: "gone", detail: "the pod is gone (deleted outside this operation)", fatal: true } : other ? await otherPodDiag(env, r.pod, rt as Awaited<ReturnType<typeof otherPodState>> | undefined ?? undefined) : await checkPod(env, r.pod, c.id, r.created * 1000, { uptimeS: rt?.uptimeS }).catch(() => null);
        if (!diag) continue;
        if (diag.fatal) {
          this.log(`${p.id}: pod ${r.pod} FAILED (${diag.phase}): ${diag.detail}; deleting it (its log and boot timeline are kept: /api/pods/${r.pod}/status)`);
          await podUpdate(env, r.pod, { status: "failed" });
          await deletePod(env, r.pod, this.log, `${p.id} ${diag.phase}`);
          c.state.workers[p.id] = (c.state.workers[p.id] || []).filter((x) => x.pod !== r.pod);
          // A slow host (pull deadline): place a new pod, at most MAX_REPLACE times per pool.
          const n = ((d.replaced ||= {})[p.id] || 0) as number;
          if (diag.replace && n < MAX_REPLACE) {
            d.replaced[p.id] = n + 1;
            const rec = await this.createFor(c, p.id);
            this.log(rec ? `${p.id}: re-placed as ${rec.pod} (${n + 1}/${MAX_REPLACE})` : `${p.id}: no stock to re-place`);
            if (rec) continue;
          }
          notes.push(`${diag.phase}: ${diag.detail}`);
        } else notes.push(`${diag.phase}${diag.detail ? `: ${diag.detail}` : ""}`);
      }
      st[p.id] = (c.state.workers[p.id] || []).length ? { status: "starting", detail: notes.join("; ") || undefined, at: now() } : { status: "failed", detail: notes.join("; ") || "every pod failed", at: now() };
    }
    // Pools without stock: another try while the others are still starting.
    const starting = pools.some((p) => st[p.id]?.status === "starting");
    if (starting && now() - (d.lastRetry || t0) >= NO_STOCK_RETRY_MS) {
      d.lastRetry = now();
      for (const p of pools.filter((x) => st[x.id]?.status === "no_stock")) {
        const rec = await this.createFor(c, p.id);
        this.log(rec ? `${p.id}: stock found on retry: pod ${rec.pod}` : `${p.id}: still no stock`);
      }
    }
    await saveState(env, c);
    const views: PoolView[] = pools.map((p) => ({ id: p.id, status: st[p.id]?.status || "no_stock", detail: st[p.id]?.detail }));
    const line = summarize(views);
    if (line !== d.lastSummary || now() - (d.lastSummaryAt || 0) >= SUMMARY_EVERY_MS) {
      this.log(`${isEdge(c.spec) ? "pools (ready = a ready front at the edge)" : "pools"} ${line} (${Math.round((now() - t0) / 1000)}s)`);
      d.lastSummary = line;
      d.lastSummaryAt = now();
    }
    const out = upOutcome(views, now() - t0, READY_WAIT_MS);
    if (!out.done) return { delayMs: 20_000 };
    if (out.error && !views.some((v) => v.status === "ready" || v.status === "starting")) await saveState(env, c, { status: "failed" });
    return out.error ? { done: true, error: out.error } : { done: true };
  }

  // ---------------- down
  private async down(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const ids = () => {
      const s = c.state;
      return [
        ...Object.values(s.workers).flat(),
        ...Object.values(s.rolling || {}).flat(),
        ...(s.retired || []),
        ...(s.gateway ? [s.gateway] : []),
      ].map((r) => r.pod);
    };
    if (op.phase === "init") {
      await saveState(env, c, { status: "stopping" });
      for (const p of ids()) await deletePod(env, p, this.log, op.params?.reason || "cluster stop");
      op.phase = "verify";
      op.data.tries = 0;
      return { delayMs: 5000 };
    }
    // verify
    const left: string[] = [];
    for (const p of ids()) {
      const pod = isOtherPod(p) ? await otherPodState(env, p).catch(() => ({ state: "unknown" })) : await runpod.pod(env, p);
      if (pod) left.push(p);
    }
    if (left.length && op.data.tries++ < 5) {
      for (const p of left) await deletePod(env, p, this.log, "retry");
      return { delayMs: 10_000 };
    }
    // Remaining live rows of this cluster (e.g. a pod the state lost) go too.
    for (const row of await livePods(env, c.id)) if (!ids().includes(row.pod_id)) await deletePod(env, row.pod_id, this.log, "orphan");
    const images = c.state.images;
    c.state = { images, workers: {} };
    await saveState(env, c, { status: left.length ? "failed" : "stopped", deadline: null });
    if (left.length) return { done: true, error: `still present: ${left.join(" ")}` };
    // DELETE of a standalone pod that still had one: the definition goes once the pod is gone.
    if (op.params?.delete_definition) {
      await deleteClusterRow(env, c.id);
      this.log(`definition ${c.name} deleted`);
    }
    return { done: true };
  }

  // ---------------- extend
  private async extend(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const minutes = Number(op.params?.minutes);
    if (!(minutes > 0 && minutes <= 24 * 60)) return { done: true, error: "minutes: 1-1440" };
    const base = Math.max(c.deadline || now(), now());
    const next = base + minutes * 60_000;
    const hours = (next - now()) / 3_600_000;
    // GMI / Brev pools: their budgets (no balance API); a cluster only there does not ask Runpod.
    for (const prov of [...new Set(c.spec.pools.filter(isOtherPool).map((p) => poolProvider(p) as OtherProviderId))]) {
      const b = await budgetCheck(env, prov, 0, hours);
      this.log(`${prov}: $${b.month.toFixed(2)} this month, $${b.running_dph.toFixed(2)}/hr running; projected $${b.projected.toFixed(2)} (budget ${b.budget === null ? "unset" : `$${b.budget}`})`);
      if (!b.ok) return { done: true, error: b.reasons.join("; ") };
    }
    if (!c.spec.pools.some((p) => !isOtherPool(p))) {
      await saveState(env, c, { deadline: next });
      return { done: true };
    }
    const acct = await runpod.account(env);
    const floor = Math.max(c.spec.balance_floor, c.spec.min_balance);
    const projected = acct.balance - acct.spendPerHr * hours;
    this.log(`balance $${acct.balance.toFixed(2)}, account $${acct.spendPerHr.toFixed(2)}/hr; new deadline ${new Date(next).toISOString()}; projected $${projected.toFixed(2)}`);
    if (projected < floor) return { done: true, error: `at $${acct.spendPerHr}/hr the balance would fall below $${floor} before ${new Date(next).toISOString()}` };
    await saveState(env, c, { deadline: next });
    if (isEdge(c.spec)) this.log("fv-control holds the new deadline; each worker's own backstop keeps its launch deadline until it is restarted (Env: apply)");
    return { done: true };
  }

  // ---------------- scale
  private async scale(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    const poolId = String(op.params?.pool);
    const pool = c.spec.pools.find((p) => p.id === poolId);
    if (!pool) return { done: true, error: `no pool ${poolId}` };
    const cur = c.state.workers[poolId] || [];
    if (op.phase === "init") {
      const count = Number(op.params?.count);
      if (!Number.isInteger(count) || count < 0 || count > 8) return { done: true, error: "count: 0-8" };
      pool.count = count;
      await saveSpec(env, c);
      await bumpDoc(env, "cluster-spec", c.id, op.actor);
      if (count > cur.length) {
        const hours = Math.max(0.1, ((c.deadline || now()) - now()) / 3_600_000);
        const proj = await projectSpend(env, c.spec, { hours, extraOnly: { pool: poolId, count: count - cur.length } });
        this.log(`scale-up projection: +$${proj.cluster_dph.toFixed(2)}/hr, $${proj.projected_balance.toFixed(2)} at the deadline`);
        if (!proj.ok) return { done: true, error: proj.reasons.join("; ") };
        op.phase = "grow";
      } else if (count < cur.length) {
        const asked: string[] = (op.params?.victims || []).filter((v: string) => cur.some((r) => r.pod === v));
        d.victims = asked.length === cur.length - count ? asked : cur.slice(count).map((r) => r.pod);
        op.phase = "drain";
      } else return { done: true };
      return { delayMs: 10 };
    }
    if (op.phase === "grow") {
      if (cur.length < pool.count) {
        if (!c.state.images[poolId] && !c.state.image) {
          // A pool added to the spec after the launch: resolve its image the way `up` does.
          const img = (await resolveClusterImages(env, c.spec))[poolId];
          if (!img) return { done: true, error: `no image for pool ${poolId}` };
          c.state.images[poolId] = img;
          await saveState(env, c);
        }
        const rec = await createWorker(env, c, poolId, c.state.images[poolId] || c.state.image!, "workers", this.log);
        if (!rec) {
          op.phase = "patch";
          this.log(`${poolId}: no stock; stopping at ${cur.length}`);
        }
        return { delayMs: 10 };
      }
      op.phase = "patch";
      return { delayMs: 10 };
    }
    if (op.phase === "drain") {
      for (const p of d.victims) {
        await workerInternal(env, c, p, "POST", "/fv/v1/internal/drain").catch((e) => this.log(`WARNING: drain of ${p}: ${(e as Error).message}`));
        await podUpdate(env, p, { status: "draining" });
      }
      d.t0 = now();
      op.phase = "idle";
      return { delayMs: 10_000 };
    }
    if (op.phase === "idle") {
      let busy = 0;
      for (const p of d.victims) busy += await workerBusy(env, c, p);
      if (busy > 0 && now() - d.t0 < DRAIN_WAIT_MS) return { delayMs: 10_000 };
      if (busy > 0) this.log(`WARNING: ${busy} jobs/sessions still running after the drain wait; removing anyway`);
      c.state.retired = [...(c.state.retired || []), ...cur.filter((r) => d.victims.includes(r.pod))];
      c.state.workers[poolId] = cur.filter((r) => !d.victims.includes(r.pod));
      await saveState(env, c);
      op.phase = "patch";
      return { delayMs: 10 };
    }
    if (op.phase === "patch") {
      if (d.victims?.length) {
        for (const p of d.victims) await deletePod(env, p, this.log, "scale-down");
        c.state.retired = (c.state.retired || []).filter((r) => !d.victims.includes(r.pod));
        await saveState(env, c);
      }
      return { done: true };
    }
    return { done: true, error: `unknown phase ${op.phase}` };
  }

  // ---------------- rolling redeploy
  /** A roll target (channel, commit or image) as the digest of a pool's variant; an all-in-one cluster (image.ref) rolls onto the all-in-one image. */
  private async resolveTarget(c: Cluster, variant: string, target: string): Promise<string> {
    if (target.includes("/")) return resolveDigest(this.env, target);
    const repo = this.env.SERVE_REPO || "ghcr.io/zaitrarrio/fastvideo-rs-serve";
    const tag = /^[0-9a-f]{7,40}$/.test(target) ? `sha-${target.slice(0, 7)}` : target;
    return resolveDigest(this.env, `${repo}:${c.spec.image.ref ? "" : `${variant}-`}${tag}`);
  }
  private async roll(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    if (op.phase === "init") {
      // params: {target: "stable" | "<sha>" | "<image>", pools?: string[] (default all)}
      const target = String(op.params?.target || "stable");
      const pools: string[] = op.params?.pools?.length ? op.params.pools : Object.keys(c.state.workers).filter((p) => c.state.workers[p]!.length);
      if (c.state.rolling && Object.keys(c.state.rolling).length) return { done: true, error: "a roll is in progress" };
      if ((c.deadline || 0) - now() < 40 * 60_000) return { done: true, error: "the cluster deadline is less than 40 min away: extend it first" };
      const acct = await runpod.account(env);
      if (acct.balance < c.spec.min_start) return { done: true, error: `balance $${acct.balance} is below $${c.spec.min_start}` };
      d.images = {} as Record<string, string>;
      for (const p of pools) {
        const spec = c.spec.pools.find((x) => x.id === p);
        if (!spec) return { done: true, error: `no pool ${p}` };
        d.images[p] = await this.resolveTarget(c, spec.variant, target);
      }
      d.pools = pools.filter((p) => (c.state.workers[p] || []).some((r) => r.image !== d.images[p]));
      this.log(`roll to ${target}: pools ${d.pools.join(", ") || "(none change)"}`);
      d.queue = d.pools.flatMap((p: string) => (c.state.workers[p] || []).map(() => p));
      op.phase = d.queue.length ? "create" : "done";
      if (op.phase === "done") return { done: true };
      return { delayMs: 10 };
    }
    if (op.phase === "create") {
      const p = d.queue.shift();
      if (p) {
        const rec = await createWorker(env, c, p, d.images[p], "rolling", this.log);
        if (!rec) return this.abortRoll(c, op, `no pod for ${p}`);
        return { delayMs: 10 };
      }
      d.t0 = now();
      op.phase = "ready";
      return { delayMs: 15_000 };
    }
    if (op.phase === "ready") {
      let all = true;
      for (const [p, recs] of Object.entries(c.state.rolling || {})) {
        for (const r of recs) {
          const h = await workerHealth(env, r.pod);
          if (h.ok) {
            const want = d.images[p].split("@")[1];
            if (h.digest && want && h.digest !== want) return this.abortRoll(c, op, `${p}: the new worker runs ${h.digest}, not ${want}`);
            await podUpdate(env, r.pod, { ready: true, status: "ready" });
          } else all = false;
        }
      }
      if (all) {
        op.phase = "drain";
        return { delayMs: 10 };
      }
      if (now() - d.t0 > READY_WAIT_MS) return this.abortRoll(c, op, "the new workers were not ready in time");
      return { delayMs: 15_000 };
    }
    if (op.phase === "drain") {
      for (const p of d.pools) for (const r of c.state.workers[p] || []) {
        await workerInternal(env, c, r.pod, "POST", "/fv/v1/internal/drain").catch((e) => this.log(`WARNING: drain of ${r.pod}: ${(e as Error).message}`));
        await podUpdate(env, r.pod, { status: "draining" });
      }
      d.t0 = now();
      op.phase = "idle";
      return { delayMs: 10_000 };
    }
    if (op.phase === "idle") {
      let busy = 0;
      for (const p of d.pools) for (const r of c.state.workers[p] || []) busy += await workerBusy(env, c, r.pod);
      if (busy > 0 && now() - d.t0 < DRAIN_WAIT_MS) return { delayMs: 10_000 };
      op.phase = "swap";
      return { delayMs: 10 };
    }
    if (op.phase === "swap") {
      const retired: PodRec[] = [];
      for (const p of d.pools || []) {
        retired.push(...(c.state.workers[p] || []));
        c.state.workers[p] = c.state.rolling?.[p] || [];
        c.state.images[p] = d.images[p];
        for (const r of c.state.workers[p]) await podUpdate(env, r.pod, { slot: "workers" });
      }
      for (const r of retired) await podUpdate(env, r.pod, { slot: "retired" });
      c.state.retired = [...(c.state.retired || []), ...retired];
      delete c.state.rolling;
      await saveState(env, c);
      op.phase = "delete";
      return { delayMs: 10 };
    }
    if (op.phase === "delete") {
      const left: PodRec[] = [];
      for (const r of c.state.retired || []) if (!(await deletePod(env, r.pod, this.log, "roll"))) left.push(r);
      c.state.retired = left;
      await saveState(env, c);
      return { done: true };
    }
    return { done: true, error: `unknown phase ${op.phase}` };
  }
  private async abortRoll(c: Cluster, op: Op, why?: string): Promise<Step> {
    op.phase = "abort";
    for (const r of Object.values(c.state.rolling || {}).flat()) await deletePod(this.env, r.pod, this.log, "roll aborted");
    delete c.state.rolling;
    await saveState(this.env, c);
    return { done: true, error: `roll aborted: ${why || "error"} (the old workers keep serving)` };
  }

  // ---------------- rolling restart (apply env changes)
  private async restart(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    if (op.phase === "init") {
      // params: {pods?: string[], pools?: string[]}: those pods (forced); neither: every pod whose env changed.
      const pods: string[] | undefined = op.params?.pods;
      const pools: string[] | undefined = op.params?.pools;
      const chosen = (r: any) => !!pods?.includes(r.pod_id) || !!pools?.includes(r.pool);
      const rows = await livePods(env, c.id);
      const ctx = await envCtx(env, c);
      const queue: { pod: string; role: string }[] = [];
      for (const r of rows.filter((x: any) => x.slot !== "retired" && x.role === "worker")) {
        const forced = !!(pods || pools);
        if (forced && !chosen(r)) continue;
        const des = await desiredEnv(env, c, ctx, "worker", { pod: r.pod_id, pool: r.pool, image: r.image });
        if (forced || des.hash !== r.env_hash) queue.push({ pod: r.pod_id, role: r.role });
      }
      d.queue = queue;
      this.log(`restart: ${queue.map((q) => q.pod).join(", ") || "nothing needs a restart"}`);
      op.phase = "next";
      return { delayMs: 10 };
    }
    if (op.phase === "next") {
      const q = d.queue.shift();
      if (!q) return { done: true };
      const rec = Object.values(c.state.workers).flat().find((r) => r.pod === q.pod);
      if (!rec) return { delayMs: 10 };
      await patchWorker(env, c, rec, this.log);
      d.current = q;
      d.t0 = now();
      op.phase = "wait";
      return { delayMs: 20_000 };
    }
    if (op.phase === "wait") {
      const q = d.current;
      const ok = (await workerHealth(env, q.pod)).ok;
      if (ok) {
        this.log(`${q.pod} is back (${Math.round((now() - d.t0) / 1000)}s)`);
        await podUpdate(env, q.pod, { ready: true, status: "ready" });
        op.phase = "next";
        return { delayMs: 10 };
      }
      if (now() - d.t0 > RESTART_WAIT_MS) return { done: true, error: `${q.pod} did not come back in ${RESTART_WAIT_MS / 60000} min; the rest were not restarted` };
      return { delayMs: 15_000 };
    }
    return { done: true, error: `unknown phase ${op.phase}` };
  }
}
