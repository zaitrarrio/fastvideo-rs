// ClusterOps: one Durable Object per cluster. It runs one operation at a
// time (up, down, extend, scale, roll, restart, gateway-start/stop) as a
// step machine driven by alarms, so a 30-minute rolling redeploy never
// depends on one request staying open. It also fans ingested log lines
// out to live-tail WebSockets (hibernation API).
import { bumpDoc } from "../docs";
import type { Env } from "../env";
import { resolveDigest, resolveClusterImages } from "../ghcr";
import { runpod } from "../runpod";
import { HttpError, now, scrub } from "../util";
import {
  adminGet,
  createGateway,
  createWorker,
  deletePod,
  gatewayHealthy,
  patchGateway,
  patchWorker,
  projectSpend,
  workerBusy,
  workerHealth,
  workerInternal,
  desiredEnv,
  envCtx,
} from "./ops";
import type { PodRec } from "./payloads";
import { getCluster, podUpdate, saveSpec, saveState, secretsOf, saveSecrets, livePods, type Cluster } from "./store";

export type OpKind = "up" | "down" | "extend" | "scale" | "roll" | "restart" | "gateway-start" | "gateway-stop";
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

const READY_WAIT_MS = 30 * 60_000;
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
      case "gateway-stop":
        return this.gatewayStop(c, op);
      case "gateway-start":
        return this.gatewayStart(c, op);
    }
  }

  // ---------------- up
  private async up(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    if (op.phase === "init") {
      if (c.state.gateway || Object.values(c.state.workers).some((l) => l.length)) return { done: true, error: "the cluster has pods already: stop it first" };
      const proj = await projectSpend(env, c.spec, { hours: c.spec.cap_s / 3600 });
      this.log(`balance $${proj.balance.toFixed(2)}, account $${proj.account_spend_per_hr.toFixed(2)}/hr, cluster ~$${proj.cluster_dph.toFixed(2)}/hr; projected $${proj.projected_balance.toFixed(2)} at the deadline (floor $${proj.floor})`);
      if (!proj.ok && !op.params?.skip_price_check) return { done: true, error: proj.reasons.join("; ") };
      c.state.images = await resolveClusterImages(env, c.spec);
      if (c.spec.image.ref) c.state.image = c.state.images.gateway || Object.values(c.state.images)[0];
      this.log(`images: ${JSON.stringify(c.state.images)}`);
      const s = await secretsOf(env, c);
      delete s.admin_token; // a new gateway makes a new one
      await saveSecrets(env, c, s);
      await saveState(env, c, { status: "starting", deadline: now() + c.spec.cap_s * 1000 });
      this.log(`deadline ${new Date(c.deadline!).toISOString()} (${c.spec.cap_s}s)`);
      op.phase = c.spec.gateway.enabled ? "gateway" : "workers";
      d.failed = [];
      return { delayMs: 10 };
    }
    if (op.phase === "gateway") {
      await createGateway(env, c, this.log);
      op.phase = "workers";
      return { delayMs: 10 };
    }
    if (op.phase === "workers") {
      for (const p of c.spec.pools) {
        if (d.failed.includes(p.id)) continue;
        if ((c.state.workers[p.id]?.length || 0) < p.count) {
          const rec = await createWorker(env, c, p.id, c.state.images[p.id] || c.state.image!, "workers", this.log);
          if (!rec) d.failed.push(p.id);
          return { delayMs: 10 };
        }
      }
      op.phase = "patch";
      return { delayMs: 10 };
    }
    if (op.phase === "patch") {
      await patchGateway(env, c, this.log);
      if (d.failed.length) this.log(`WARNING: no pod for: ${d.failed.join(", ")}`);
      await saveState(env, c, { status: "running" });
      op.phase = "wait";
      d.t0 = now();
      return { delayMs: 20_000 };
    }
    if (op.phase === "wait") return this.waitReady(c, op, d.t0);
    return { done: true, error: `unknown phase ${op.phase}` };
  }

  /** Until every pool with workers has a ready one (the admin pools view), or each worker's /health says AVAILABLE without a gateway. */
  private async waitReady(c: Cluster, op: Op, t0: number): Promise<Step> {
    const env = this.env;
    const pools = Object.entries(c.state.workers).filter(([, l]) => l.length);
    let ready = 0;
    if (c.state.gateway) {
      let view: any = null;
      try {
        view = await adminGet(env, c, "/fv/v1/gateway/pools");
      } catch (e) {
        this.log(`gateway not answering yet (${Math.round((now() - t0) / 1000)}s): ${(e as Error).message.slice(0, 100)}`);
      }
      if (view) {
        await podUpdate(env, c.state.gateway.pod, { ready: true, status: "ready" });
        for (const [pool, recs] of pools) {
          const st = (view.state || []).find((x: any) => x.id === pool);
          const readyUrls = new Set<string>((st?.workers || []).filter((w: any) => w.ready).map((w: any) => String(w.url).replace(/\/$/, "")));
          let any = false;
          for (const r of recs) if (r.url && readyUrls.has(r.url.replace(/\/$/, ""))) (any = true), await podUpdate(env, r.pod, { ready: true, status: "ready" });
          if (any) ready++;
        }
        this.log(`pools with a ready worker: ${ready}/${pools.length} (${Math.round((now() - t0) / 1000)}s)`);
      }
    } else {
      for (const [, recs] of pools) {
        let any = false;
        for (const r of recs) if ((await workerHealth(env, r.pod)).ok) (any = true), await podUpdate(env, r.pod, { ready: true, status: "ready" });
        if (any) ready++;
      }
      this.log(`pools with a ready worker: ${ready}/${pools.length}`);
    }
    if (ready >= pools.length && (pools.length > 0 || c.state.gateway)) return { done: true };
    if (now() - t0 > READY_WAIT_MS) return { done: true, error: "not every pool became ready (the pods stay up; see the pool view)" };
    return { delayMs: 20_000 };
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
      const pod = await runpod.pod(env, p);
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
    return left.length ? { done: true, error: `still present: ${left.join(" ")}` } : { done: true };
  }

  // ---------------- extend
  private async extend(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const minutes = Number(op.params?.minutes);
    if (!(minutes > 0 && minutes <= 24 * 60)) return { done: true, error: "minutes: 1-1440" };
    const base = Math.max(c.deadline || now(), now());
    const next = base + minutes * 60_000;
    const acct = await runpod.account(env);
    const hours = (next - now()) / 3_600_000;
    const floor = Math.max(c.spec.balance_floor, c.spec.min_balance);
    const projected = acct.balance - acct.spendPerHr * hours;
    this.log(`balance $${acct.balance.toFixed(2)}, account $${acct.spendPerHr.toFixed(2)}/hr; new deadline ${new Date(next).toISOString()}; projected $${projected.toFixed(2)}`);
    if (projected < floor) return { done: true, error: `at $${acct.spendPerHr}/hr the balance would fall below $${floor} before ${new Date(next).toISOString()}` };
    await saveState(env, c, { deadline: next });
    await patchGateway(env, c, this.log);
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
      await patchGateway(env, c, this.log);
      if (d.victims?.length) {
        for (const p of d.victims) await deletePod(env, p, this.log, "scale-down");
        c.state.retired = (c.state.retired || []).filter((r) => !d.victims.includes(r.pod));
        await saveState(env, c);
      }
      return { done: true };
    }
    return { done: true, error: `unknown phase ${op.phase}` };
  }

  // ---------------- rolling redeploy (release.sh redeploy / runpod-cluster.sh roll)
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
      // params: {target: "stable" | "<sha>" | "<image>", pools?: string[] (default all), gateway?: boolean}
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
      if (op.params?.gateway && c.state.gateway) d.gateway = await this.resolveTarget(c, "gateway", target);
      d.pools = pools.filter((p) => (c.state.workers[p] || []).some((r) => r.image !== d.images[p]));
      this.log(`roll to ${target}: pools ${d.pools.join(", ") || "(none change)"}${d.gateway ? ", gateway" : ""}`);
      d.queue = d.pools.flatMap((p: string) => (c.state.workers[p] || []).map(() => p));
      op.phase = d.queue.length ? "create" : d.gateway ? "swap" : "done";
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
      await patchGateway(env, c, this.log);
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
      await patchGateway(env, c, this.log, d.gateway);
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
    await patchGateway(this.env, c, this.log).catch((e) => this.log(`gateway patch after abort failed: ${(e as Error).message}`));
    return { done: true, error: `roll aborted: ${why || "error"} (the old workers keep serving)` };
  }

  // ---------------- rolling restart (apply env changes)
  private async restart(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    const d = op.data;
    if (op.phase === "init") {
      const want: string[] | undefined = op.params?.pods;
      const rows = await livePods(env, c.id);
      const ctx = await envCtx(env, c);
      const queue: { pod: string; role: string }[] = [];
      for (const r of rows.filter((x: any) => x.slot !== "retired")) {
        if (want && !want.includes(r.pod_id)) continue;
        const des = await desiredEnv(env, c, ctx, r.role, { pod: r.pod_id, pool: r.pool, image: r.image });
        if (want || des.hash !== r.env_hash) queue.push({ pod: r.pod_id, role: r.role });
      }
      // Workers first (the gateway keeps routing to the others), the gateway last.
      queue.sort((a, b) => (a.role === "gateway" ? 1 : 0) - (b.role === "gateway" ? 1 : 0));
      d.queue = queue;
      this.log(`restart: ${queue.map((q) => q.pod).join(", ") || "nothing needs a restart"}`);
      op.phase = "next";
      return { delayMs: 10 };
    }
    if (op.phase === "next") {
      const q = d.queue.shift();
      if (!q) return { done: true };
      if (q.role === "gateway") await patchGateway(env, c, this.log);
      else {
        const rec = Object.values(c.state.workers).flat().find((r) => r.pod === q.pod);
        if (!rec) return { delayMs: 10 };
        await patchWorker(env, c, rec, this.log);
      }
      d.current = q;
      d.t0 = now();
      op.phase = "wait";
      return { delayMs: 20_000 };
    }
    if (op.phase === "wait") {
      const q = d.current;
      const ok = q.role === "gateway" ? await gatewayHealthy(env, c) : (await workerHealth(env, q.pod)).ok;
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

  // ---------------- the gateway alone
  private async gatewayStop(c: Cluster, _op: Op): Promise<Step> {
    if (!c.state.gateway) return { done: true, error: "no gateway" };
    await runpod.stop(this.env, c.state.gateway.pod);
    c.state.gateway_stopped = true;
    await saveState(this.env, c);
    await podUpdate(this.env, c.state.gateway.pod, { status: "stopped" });
    this.log(`gateway ${c.state.gateway.pod} stopped (its watchdog is off until it starts; the controller's backstop still holds the deadline)`);
    return { done: true };
  }
  private async gatewayStart(c: Cluster, op: Op): Promise<Step> {
    const env = this.env;
    if (op.phase === "init") {
      if (c.state.gateway && c.state.gateway_stopped) {
        await runpod.start(env, c.state.gateway.pod);
        c.state.gateway_stopped = false;
        await saveState(env, c);
        this.log(`gateway ${c.state.gateway.pod} starting`);
      } else if (!c.state.gateway) {
        if (!c.state.images.gateway && !c.state.image) c.state.images.gateway = (await resolveClusterImages(env, { ...c.spec, gateway: { ...c.spec.gateway, enabled: true } })).gateway!;
        await createGateway(env, c, this.log);
        // The workers render URLs with the gateway's address: they need the new one.
        for (const r of Object.values(c.state.workers).flat()) await patchWorker(env, c, r, this.log);
        await patchGateway(env, c, this.log);
      } else return { done: true, error: "the gateway is running" };
      op.phase = "wait";
      op.data.t0 = now();
      return { delayMs: 15_000 };
    }
    if (await gatewayHealthy(env, c)) {
      await podUpdate(env, c.state.gateway!.pod, { status: "ready", ready: true });
      return { done: true };
    }
    if (now() - op.data.t0 > READY_WAIT_MS) return { done: true, error: "the gateway did not come up" };
    return { delayMs: 15_000 };
  }
}
