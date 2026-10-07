// UI test of the log explorer and the cluster configuration page in headless
// Chromium (playwright-core), over the Worker under `wrangler dev` and the
// mocked APIs (test/harness.mjs), with seeded log lines.
//   node test/ui/explorer.mjs            (npm run test:ui:explorer)
// CHROMIUM=<path> picks the browser; SHOTS=<dir> where screenshots go (default test-results/).
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright-core";
import { d1Exec, hashPassphrase, HERE, PASSPHRASE, SECRETS, startMock, startWorker } from "../harness.mjs";

const out = process.env.SHOTS || `${HERE}test-results`;
mkdirSync(out, { recursive: true });
const mock = await startMock();
mock.runpodKey = SECRETS.RUNPOD_API_KEY;
mock.githubPat = SECRETS.GITHUB_PAT;
const w = await startWorker(mock, {
  ...SECRETS,
  OWNER_PASSPHRASE_HASH: await hashPassphrase(PASSPHRASE, SECRETS.SESSION_SECRET),
  EDGE_URL: `http://127.0.0.1:${mock.port}/edge`,
  EDGE_INTERNAL_TOKEN: mock.edgeInternal,
  EDGE_ADMIN_TOKEN: mock.edgeAdmin,
  EDGE_D1_DATABASE_ID: "d1-edge-staging",
});
const B = w.url;
let browser;
const fail = (e) => {
  console.log(`FAIL ${e.stack || e.message}`);
  console.log(w.output().slice(-3000));
  browser?.close();
  w.stop();
  mock.close();
  process.exit(1);
};

/** Realistic fv-serve lines (shapes from staging) for four pods over the last two hours. */
function seedSql(clusterId, now) {
  const pods = [
    ["4506k60farotxy", "fake", "fv-ctl-h3-and-ltx-fake-1006211539"],
    ["6um7ikll681o6o", "h3-turbo", "fv-ctl-h3-and-ltx-h3-turbo-1006211540"],
    ["cq1la8p41t4snl", "h3-max", "fv-ctl-h3-and-ltx-h3-max-1006211542"],
    ["9xq2ltxref2v01", "ltx-ref2v", "fv-ctl-h3-and-ltx-ltx-ref2v-1006211544"],
  ];
  const sql = [];
  for (const [id, pool, name] of pods) {
    sql.push(`INSERT INTO cluster_pods (pod_id, cluster_id, role, pool, created_at, status) VALUES ('${id}', '${clusterId}', 'worker', '${pool}', ${now - 7200_000}, 'ready')`);
    sql.push(`INSERT INTO pods (pod_id, name, owner, cluster_id, desired_status, first_seen, last_seen) VALUES ('${id}', '${name}', 'cluster:h3-and-ltx', '${clusterId}', 'RUNNING', ${now - 7200_000}, ${now})`);
  }
  const msgs = [
    ["info", "fastvideo_serve::edge_link", "worker: connected to the dispatcher", (i) => ({ connect_ms: 500 + (i % 300), held: i % 3, scope: "family:h3" })],
    ["info", "fastvideo_serve::jobs", "job started", (i) => ({ job_id: `job_${(i * 7919) % 100000}`, model: "fasth3", steps: 4 })],
    ["debug", "fastvideo_serve::engine", "denoise step", (i) => ({ job_id: `job_${(i * 7919) % 100000}`, step: i % 4, ms: 180 + (i % 40) })],
    ["info", "fastvideo_serve::jobs", "job done", (i) => ({ job_id: `job_${(i * 7919) % 100000}`, total_ms: 2300 + (i % 900), bytes: 1200000 + i })],
    ["warn", "fastvideo_serve::edge_link", "dispatcher heartbeat late", (i) => ({ late_ms: 3000 + (i % 2000) })],
    ["info", "fastvideo_serve::app", "ready", () => ({ models: 7 })],
    ["error", "fastvideo_serve::jobs", "job failed: CUDA out of memory (tried to allocate 2.00 GiB)", (i) => ({ job_id: `job_${(i * 7919) % 100000}`, gpu_mem_gb: 94.1 })],
    ["trace", "fastvideo_serve::director", "tick", () => ({})],
  ];
  const rows = [];
  const N = 3000;
  for (let i = 0; i < N; i++) {
    const [pod] = pods[i % pods.length];
    let k = i % 13 === 0 ? 4 : i % 97 === 0 ? 6 : i % 5 === 0 ? 2 : i % 7 === 0 ? 7 : i % 3 === 0 ? 3 : i % 11 === 0 ? 5 : i % 2 === 0 ? 1 : 0;
    const [level, target, msg, f] = msgs[k];
    const ts = now - 7000_000 + Math.floor((i / N) * 6990_000);
    rows.push(`('${clusterId}', '${pod}', ${ts}, '${level}', '${target}', '${msg.replace(/'/g, "''")}', '${JSON.stringify(f(i)).replace(/'/g, "''")}')`);
  }
  for (let i = 0; i < rows.length; i += 500) sql.push(`INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg, fields) VALUES ${rows.slice(i, i + 500).join(",")}`);
  return sql;
}

try {
  const login = await fetch(`${B}/api/auth/login`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ passphrase: PASSPHRASE }) });
  const cookie = login.headers.get("set-cookie").split(";")[0];
  const { csrf } = await login.json();
  const tok = (await (await fetch(`${B}/api/tokens`, { method: "POST", headers: { cookie, "x-csrf-token": csrf, "content-type": "application/json" }, body: JSON.stringify({ name: "seed" }) })).json()).token;
  const raw = (p, body, method) => fetch(B + p, { method: method || (body ? "POST" : "GET"), headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const api = (p, body, method) => fetch(B + p, { method: method || (body ? "POST" : "GET"), headers: { authorization: `Bearer ${tok}`, "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined }).then((r) => r.json());
  // A tiny cluster that ran (its operations and audit rows), and a defined h3/ltx cluster with seeded pod lines.
  const tiny = await api("/api/clusters", { spec: { name: "tiny", template: "tiny-cpu", cap_s: 3600 } });
  await api(`/api/clusters/${tiny.cluster.id}/start`, {});
  for (let i = 0; i < 60; i++) {
    const ops = await api(`/api/clusters/${tiny.cluster.id}/ops`);
    if (ops.operations[0]?.status !== "running") break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  const big = await api("/api/clusters", { spec: { name: "h3-and-ltx", template: "h3" } });
  const now = Date.now();
  const sqlFile = `${w.dir}/seed.sql`;
  writeFileSync(sqlFile, seedSql(big.cluster.id, now).join(";\n") + ";\n");
  const { execFileSync } = await import("node:child_process");
  execFileSync(`${HERE}node_modules/.bin/wrangler`, ["d1", "execute", "fv-control", "--local", "--persist-to", w.dir, "--file", sqlFile], { cwd: HERE, stdio: "pipe", env: { ...process.env, CI: "1" } });
  void d1Exec;

  // ---- the API: filters, cursors, export
  const q1 = await api(`/api/logs/query?src=pod&cluster=${big.cluster.id}&lv=error&since=3h&limit=5`);
  assert.equal(q1.lines.length, 5);
  assert.ok(q1.lines.every((l) => l.level === "error" && l.cluster === "h3-and-ltx"));
  const q2 = await api(`/api/logs/query?src=pod&cluster=${big.cluster.id}&lv=error&since=3h&limit=5&cursor=${q1.next}`);
  assert.ok(q2.lines.every((l) => !q1.lines.some((x) => x.uid === l.uid)), "next page has new lines");
  const rx = await api(`/api/logs/query?src=pod&q=${encodeURIComponent("job_\\d+3\\b")}&re=1&lv=trace,debug,info,warn,error&since=3h&limit=50`);
  assert.ok(rx.lines.length > 0 && rx.lines.every((l) => /job_\d+3\b/.test(JSON.stringify(l.fields))), "regex filter");
  const ex = await fetch(`${B}/api/logs/export?src=pod&pool=h3-max&lv=error&since=3h&format=txt`, { headers: { authorization: `Bearer ${tok}` } });
  const exText = await ex.text();
  assert.match(ex.headers.get("content-disposition"), /attachment/);
  assert.ok(exText.trim().split("\n").every((l) => l.includes("ERROR") && l.includes("h3-max")), "export honours filters");

  // ---- the page
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined, headless: true });
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 920 } });
  await ctx.grantPermissions(["clipboard-read", "clipboard-write"]);
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(B + "/");
  await page.fill("#pass", PASSPHRASE);
  await page.click("button[type=submit]");
  await page.waitForSelector("h1:text('Dashboard')");

  // Navigate within the app and wait for a fresh explorer (not the previous page's rows).
  const logs = async (qs) => {
    await page.evaluate(() => document.querySelector("#main").replaceChildren());
    await page.goto(`${B}/#/logs?${qs}`);
    await page.waitForFunction(() => /lines? loaded/.test(document.querySelector(".lx-status")?.textContent || ""));
  };
  await logs("from=3h");
  await page.waitForSelector(".lx-row");
  const rowsVisible = await page.$$eval(".lx-row", (els) => els.length);
  assert.ok(rowsVisible < 200, `virtualized: ${rowsVisible} rows in the DOM`);
  await page.screenshot({ path: `${out}/logs-01-explorer.png` });
  // Live tail is off; the newest lines are at the bottom (oldest first) and the list is scrolled there.
  const atBottom = await page.$eval(".lx-list", (e) => e.scrollHeight - e.scrollTop - e.clientHeight < 30);
  assert.ok(atBottom, "scrolled to the newest lines");
  // Filters in the URL: errors only of one pool.
  await logs("from=3h&lv=warn,error&pool=h3-max");
  await page.waitForFunction(() => document.querySelectorAll(".lx-row").length > 0 && [...document.querySelectorAll(".lx-row .lx-lv")].every((e) => ["WARN", "ERROR"].includes(e.textContent)));
  assert.equal(await page.$eval(".lx-bar select[aria-label=Pool]", (e) => e.value), "h3-max");
  // Click a line: details; shift-click: a range; copy.
  const rowsSel = ".lx-row:not(.lx-group)";
  // Rows by their index in the list (the DOM holds only a window of them).
  const f0 = Number(await page.$eval(rowsSel, (e) => e.dataset.i));
  await page.click(`.lx-row[data-i="${f0 + 32}"]`);
  await page.waitForSelector(".lx-detail.open .lx-ftable");
  await page.click(`.lx-row[data-i="${f0 + 36}"]`, { modifiers: ["Shift"] });
  assert.equal(await page.$$eval(".lx-row.sel", (e) => e.length), 5);
  await page.screenshot({ path: `${out}/logs-02-select-detail.png` });
  await page.keyboard.press("Control+c");
  const clip = await page.evaluate(() => navigator.clipboard.readText());
  assert.equal(clip.split("\n").length, 5, "copied five lines");
  assert.match(clip, /\[pod h3-and-ltx h3-max cq1la8p41t4snl\]/);
  // Context around the focused line.
  await page.click(".lx-detail button:has-text('Context')");
  await page.waitForSelector(".lx-ctx-line.anchor");
  const ctxN = await page.$$eval(".lx-ctx-line", (e) => e.length);
  assert.equal(ctxN, 11, "5 before, the line, 5 after");
  await page.screenshot({ path: `${out}/logs-03-context.png` });
  // Keyboard: j/k, pin, bookmark, n (next warn/error), g/G.
  await page.keyboard.press("Escape");
  await page.focus(".lx-list");
  await page.keyboard.press("g");
  const firstUid = await page.evaluate(() => new URLSearchParams(location.hash.split("?")[1]).get("sel"));
  await page.keyboard.press("j");
  await page.keyboard.press("j");
  const thirdUid = await page.evaluate(() => new URLSearchParams(location.hash.split("?")[1]).get("sel"));
  assert.notEqual(firstUid, thirdUid, "j moves the selection");
  await page.keyboard.press("p");
  await page.waitForSelector(".lx-pin");
  await page.keyboard.press("b");
  await page.keyboard.press("G");
  // Regex search in find mode highlights and n jumps.
  await logs("from=3h&lv=trace,debug,info,warn,error&mode=find&q=CUDA%20out%20of%20memory&group=pod");
  await page.waitForSelector(".lx-row.lx-group");
  await page.focus(".lx-list");
  await page.keyboard.press("n");
  await page.waitForSelector(".lx-row.focus mark");
  await page.screenshot({ path: `${out}/logs-04-find-group.png` });
  // Server-side regex filter, sorted by level, dark theme.
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await logs(`from=3h&q=${encodeURIComponent("late|failed")}&re=1&sort=level&lv=debug,info,warn,error`);
  await page.waitForSelector(".lx-row");
  const lvls = await page.$$eval(".lx-row .lx-lv", (e) => e.map((x) => x.textContent));
  assert.equal(lvls[0], "ERROR", "level sort: errors first");
  assert.ok(lvls.every((l) => l === "ERROR" || l === "WARN"), lvls.join(","));
  await page.screenshot({ path: `${out}/logs-05-regex-level-dark.png` });
  // Operations and audit as sources (a cluster's start), light theme.
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  await logs("from=3h&src=op,audit&lv=trace,debug,info,warn,error");
  await page.waitForSelector(".lx-who.src-op");
  await page.click(".lx-row:has(.lx-who.src-op) >> nth=0");
  await page.waitForSelector(".lx-detail.open");
  await page.screenshot({ path: `${out}/logs-06-ops-audit.png` });
  // Follow one pod: live tail; a line ingested now shows up.
  await logs("from=3h&pod=6um7ikll681o6o&live=1");
  await page.waitForSelector(".lx-status .badge.good");
  await new Promise((r) => setTimeout(r, 2500));
  execFileSync(`${HERE}node_modules/.bin/wrangler`, ["d1", "execute", "fv-control", "--local", "--persist-to", w.dir, "--command", `INSERT INTO log_lines (cluster_id, pod_id, ts, level, target, msg) VALUES ('${big.cluster.id}', '6um7ikll681o6o', ${Date.now()}, 'info', 'fastvideo_serve::jobs', 'a brand new live line')`], { cwd: HERE, stdio: "pipe", env: { ...process.env, CI: "1" } });
  await page.waitForSelector(".lx-row:has-text('a brand new live line')", { timeout: 15000 });
  await page.screenshot({ path: `${out}/logs-07-follow-live.png` });
  // Jump to a time keeps the position there; the histogram is drawn.
  assert.ok(await page.$(".lx-hist svg rect"), "histogram");
  // Phone width.
  await page.setViewportSize({ width: 390, height: 844 });
  await logs("from=3h");
  await page.waitForSelector(".lx-row");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 1, `no horizontal page scroll at phone width (${overflow}px)`);
  await page.screenshot({ path: `${out}/logs-08-phone.png` });
  await page.setViewportSize({ width: 1440, height: 920 });

  if (process.env.CLUSTER_UI !== "0") {
    const { clusterEditorChecks } = await import("./cluster-config.mjs");
    await clusterEditorChecks({ page, B, api, out, tiny, big });
  }
  if (process.env.CONFIG_UI !== "0") {
    const { configValidationChecks } = await import("./config-validation.mjs");
    await configValidationChecks({ page, B, api, raw, out, tiny });
  }
  if (process.env.CANCEL_UI !== "0") {
    const { cancelUiChecks } = await import("./cancel.mjs");
    await cancelUiChecks({ page, B, out, tiny, w, mock });
    console.log("ok   UI cancel: Jobs card (cancel one, all queued, by id), Jobs page, serverless cancel and purge");
  }
  assert.deepEqual(errors.filter((e) => !/favicon|Failed to load resource/.test(e)), [], "no console errors");
  console.log(`ok   UI explorer: filters in the URL, virtualized list, selection + copy, context, pins, find, regex, level sort, ops/audit sources, follow + live tail, phone width (screenshots in ${out})`);
  await browser.close();
  w.stop();
  mock.close();
} catch (e) {
  fail(e);
}
