// Live check of src/serverless against the real Runpod API (docs/control/serverless.md
// "Live test"): one CPU queue endpoint of the `cpu` variant (fake engine) made by
// the same code fv-control runs, an info job (cold), a warm info job, a fake
// generation job, the tick (health, logs into the log store, billing), scale to
// 0, delete, verify absent. D1 is an in-memory SQLite with the migrations
// (staging's D1 is not touched). A detached wall-clock backstop deletes the
// endpoint and template after FV_LIVE_CAP_S (default 2700 s) whatever happens.
//
//   RUNPOD_API_KEY=… npx vite-node test/live/serverless-live.ts
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { d1 } from "../unit/d1shim";
import { createEndpoint, deleteEndpoint, getRow, invoke, pollJob, recordBilling, scaleEndpoint, serverlessTick, type SlsRow } from "../../src/serverless/ops";
import { sls } from "../../src/serverless/runpod-sls";
import { runpod } from "../../src/runpod";

const KEY = process.env.RUNPOD_API_KEY || "";
if (!KEY) throw new Error("RUNPOD_API_KEY");
const OUT = process.env.FV_LIVE_OUT || "/tmp/serverless-live.json";
const CAP_S = Number(process.env.FV_LIVE_CAP_S || 2700);
const env: any = { DB: d1(), RUNPOD_API_KEY: KEY, BALANCE_FLOOR: "8" };
const who = { actor: "live-test" };
const t0 = Date.now();
const log = (...a: unknown[]) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);
const report: any = { started: new Date(t0).toISOString(), steps: [] };
const step = (name: string, x: unknown) => {
  report.steps.push({ at_s: (Date.now() - t0) / 1000, name, ...(x as object) });
  writeFileSync(OUT, JSON.stringify(report, null, 2));
};

function backstop(ep: string, tpl: string | null) {
  const api = "https://rest.runpod.io/v1";
  const script = `sleep ${CAP_S}
h=(-H "Authorization: Bearer $RUNPOD_API_KEY" -H "content-type: application/json")
curl -sS -X PATCH "\${h[@]}" -d '{"workersMin":0,"workersMax":0}' "${api}/endpoints/${ep}" >/dev/null
sleep 20
curl -sS -X DELETE "\${h[@]}" "${api}/endpoints/${ep}" >/dev/null
${tpl ? `sleep 5; curl -sS -X DELETE "\${h[@]}" "${api}/templates/${tpl}" >/dev/null` : ""}`;
  const p = spawn("bash", ["-c", script], { detached: true, stdio: "ignore", env: process.env });
  p.unref();
  log(`backstop pid ${p.pid}: deletes ${ep} / ${tpl} after ${CAP_S}s`);
  report.backstop_pid = p.pid;
}

async function waitJob(row: SlsRow, jobRow: number, capMs: number) {
  const end = Date.now() + capMs;
  for (;;) {
    const j = await pollJob(env, row, jobRow);
    if (j.finished_at) return j;
    if (Date.now() > end) return j;
    await new Promise((r) => setTimeout(r, 3000));
  }
}

async function main() {
  const acct = await runpod.account(env);
  log("balance", acct.balance.toFixed(2), "burn/hr", acct.spendPerHr);
  report.balance_before = acct.balance;
  if (acct.balance < 15) throw new Error(`balance $${acct.balance} below $15: not starting`);

  const tc = Date.now();
  const row = await createEndpoint(env, who, { name: "sls-live", variant: "cpu", deadline_min: 40, idle_timeout_s: 5, execution_timeout_s: 600 });
  backstop(row.endpoint_id!, row.template_id);
  step("create", { ms: Date.now() - tc, endpoint: row.endpoint_id, template: row.template_id, image: row.image, spec: JSON.parse(row.spec) });
  log("created", row.endpoint_id, row.template_id, row.image);
  try {
    // Cold: info job (Runpod holds /runsync ~90 s, then we poll).
    const tj = Date.now();
    const cold = await invoke(env, who, row, { input: { kind: "info" } });
    log("cold invoke answered", cold.status, `${cold.wall_ms} ms`);
    const coldDone = cold.done ? cold : await waitJob(row, cold.job!, 15 * 60_000);
    step("cold_info", { client_wall_ms: Date.now() - tj, status: coldDone.status, delay_ms: coldDone.delay_ms, exec_ms: coldDone.exec_ms, worker: coldDone.worker_id, cold: cold.cold, output: typeof coldDone.output === "string" ? JSON.parse(coldDone.output) : coldDone.output });
    log("cold info", coldDone.status, "delay", coldDone.delay_ms, "exec", coldDone.exec_ms);

    // What Runpod shows while the worker is up: GraphQL endpoint pods, myself.pods, the worker's logs.
    const live = await sls.live(env);
    const mine = live.endpoints.find((e) => e.id === row.endpoint_id);
    const pods = await runpod.pods(env);
    const worker = coldDone.worker_id || mine?.pods?.[0]?.id;
    step("live_view", { endpoint_pods: mine?.pods, worker_in_myself_pods: worker ? pods.pods.some((p) => p.id === worker) : null });
    const health = await sls.health(env, row.endpoint_id!);
    step("health_after_cold", { health });

    // Warm: a second info job, then a fake generation job.
    const tw = Date.now();
    const warm = await invoke(env, who, await getRow(env, row.id), { input: { kind: "info" } });
    const warmDone = warm.done ? warm : await waitJob(row, warm.job!, 5 * 60_000);
    step("warm_info", { client_wall_ms: Date.now() - tw, status: warmDone.status, delay_ms: warmDone.delay_ms, exec_ms: warmDone.exec_ms, cold: warm.cold });
    log("warm info", warmDone.status, "delay", warmDone.delay_ms, "exec", warmDone.exec_ms);
    const tg = Date.now();
    const gen = await invoke(env, who, await getRow(env, row.id), { input: { kind: "http", method: "POST", path: "/fv/v1/jobs", body: { model: "fake-wan", prompt: "a red fox trotting through fresh snow", seed: 1 }, wait: true } });
    const genDone = gen.done ? gen : await waitJob(row, gen.job!, 5 * 60_000);
    const go = typeof genDone.output === "string" ? JSON.parse(genDone.output) : genDone.output;
    step("fake_job", { client_wall_ms: Date.now() - tg, status: genDone.status, delay_ms: genDone.delay_ms, exec_ms: genDone.exec_ms, http_status: go?.status, job_status: go?.body?.status, model: go?.body?.model, error: genDone.error });
    log("fake job", genDone.status, go?.status, go?.body?.status);

    // The tick: health, worker logs into log_lines, billing.
    const tick = await serverlessTick(env, { force_billing: true });
    const logs = await env.DB.prepare("SELECT COUNT(*) AS n, MIN(ts) AS first, MAX(ts) AS last FROM log_lines WHERE cluster_id LIKE 'serverless:%'").first();
    const sample = await env.DB.prepare("SELECT ts, level, target, msg FROM log_lines WHERE cluster_id LIKE 'serverless:%' AND msg LIKE '%READY%' LIMIT 3").all();
    const r1 = await getRow(env, row.id);
    step("tick", { tick, workers: r1.workers, live_dph: r1.live_dph, health: JSON.parse(r1.health || "null"), log_lines: logs, ready_lines: sample.results });
    log("tick", JSON.stringify(tick), "workers", r1.workers, "dph", r1.live_dph, "log lines", (logs as any).n);
  } finally {
    // Scale to 0, then delete (the tick retries a refused delete).
    const ts = Date.now();
    let r = await getRow(env, row.id);
    if (r.status === "active") r = await scaleEndpoint(env, who, r, { workers_min: 0, workers_max: 0 }).catch((e) => (log("scale failed", e.message), r));
    step("scale0", { ms: Date.now() - ts, status: r.status });
    const td = Date.now();
    r = await deleteEndpoint(env, who, await getRow(env, row.id));
    for (let i = 0; i < 12 && r.status !== "deleted"; i++) {
      log("delete pending:", r.last_error);
      await new Promise((res) => setTimeout(res, 10_000));
      await serverlessTick(env);
      r = await getRow(env, row.id);
    }
    const gone = (await sls.getEndpoint(env, row.endpoint_id!)) === null;
    const tpls = (await runpod.rest(env, "GET", "/templates")) as any[];
    const tplGone = !tpls.some((t) => t.id === row.template_id);
    step("delete", { ms: Date.now() - td, status: r.status, endpoint_absent: gone, template_absent: tplGone });
    log("deleted", r.status, "endpoint absent", gone, "template absent", tplGone);
    const audits = await env.DB.prepare("SELECT action, actor, ok FROM audit ORDER BY id").all();
    report.audit = audits.results;
    report.jobs = (await env.DB.prepare("SELECT route, status, cold, delay_ms, exec_ms, wall_ms FROM serverless_jobs ORDER BY id").all()).results;
    const acct2 = await runpod.account(env);
    report.balance_after = acct2.balance;
    report.wall_s = (Date.now() - t0) / 1000;
    writeFileSync(OUT, JSON.stringify(report, null, 2));
  }
}
main().catch((e) => {
  console.error("live test failed:", e.message);
  report.error = e.message;
  writeFileSync(OUT, JSON.stringify(report, null, 2));
  process.exit(1);
});
