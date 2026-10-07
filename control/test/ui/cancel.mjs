// Cancelling jobs in Chromium (run from test/ui/explorer.mjs, which starts
// the Worker, the mocks and a running `tiny` edge cluster): the Jobs card on
// the cluster page (cancel one, cancel all queued, cancel by id), the Jobs
// page, and the serverless endpoint's Queue card (cancel a pasted job id,
// purge with the queued count and the name typed). Every dialog is a
// schema-bound form: the coverage audit finds no unbound control.
import assert from "node:assert/strict";
import { d1Exec, jobInsert, JOBS_TABLE } from "../harness.mjs";

export async function cancelUiChecks({ page, B, out, tiny, w, mock }) {
  const shot = (n) => page.screenshot({ path: `${out}/cancel-${n}.png`, fullPage: true });
  const audit = async () => {
    const probs = await page.evaluate(async () => {
      const s = (await (await fetch("/api/schemas", { credentials: "same-origin" })).json()).schemas;
      // The dialog this test opened (the endpoint page's spec editor is marked serverless-endpoint-json, a name with no schema: not this change's).
      return window.FVEditor.auditForms(document.querySelector("dialog.ff-dialog") || document, s);
    });
    assert.deepEqual(probs, [], `coverage audit: ${probs.join("\n")}`);
  };
  const settle = async () => {
    for (let ok = 0, i = 0; i < 40 && ok < 2; i++) {
      ok = (await fetch(`${B}/healthz`).then((r) => r.ok).catch(() => false)) ? ok + 1 : 0;
      await new Promise((r) => setTimeout(r, 250));
    }
  };
  const go = async (hash) => {
    await page.evaluate(() => document.querySelector("#main").replaceChildren());
    if (await page.evaluate((h) => location.hash === h, hash)) await page.reload();
    else await page.goto(`${B}/${hash}`);
  };
  // One handler for the confirm() prompts (the earlier checks each left one: two accepts of one dialog throw).
  page.removeAllListeners("dialog");
  page.on("dialog", (d) => d.accept().catch(() => {}));

  // ---- seed: the tiny cluster's jobs in the edge's D1, a serverless endpoint with a queue
  const pid = [...mock.pods.values()].find((p) => p.env?.FV_DISPATCH_FRONT === "1" && p.desiredStatus === "RUNNING").id;
  const now = Date.now();
  const U = (n) => `0000000${n}-aaaa-4bbb-8ccc-eeeeeeeeeeee`;
  for (const db of ["fv-edge", "fv-jobs"]) d1Exec(w.dir, JOBS_TABLE, db);
  d1Exec(
    w.dir,
    [
      jobInsert({ id: U(1), ext: "fvjob_9f2c41d0a7b34e1d", model: "fake-wan", status: "running", worker: pid, at: now - 50_000, started: new Date(now - 41_000).toISOString(), progress: 0.62 }),
      jobInsert({ id: U(2), ext: "video_gen_68e2b1", api: "openai_videos", model: "fake-wan", status: "queued", worker: null, at: now - 4_000 }),
      jobInsert({ id: U(3), ext: "fvjob_1b77aa0e94c0", model: "fake-wan", status: "queued", worker: pid, at: now - 3_000 }),
      jobInsert({ id: U(4), ext: "4c0f7e2a-5d3b-4f8e-9a61-0c2d8b7e1f33", api: "fal", model: "fal-ai/wan/v2.2-5b/text-to-video/fast-wan", status: "succeeded", worker: pid, at: now - 300_000, started: new Date(now - 290_000).toISOString() }),
      jobInsert({ id: U(5), ext: "fvjob_0d1e2f3a4b5c", model: "fake-wan", status: "cancelled", worker: pid, at: now - 200_000 }),
    ].join(";\n"),
    "fv-edge",
  );
  const spec = JSON.stringify({ name: "h3-max2", mode: "queue", variant: "h3-max", compute: "GPU", workers_min: 0, workers_max: 1, idle_timeout_s: 5, flashboot: false, deadline_min: 120, deadline_action: "scale0" });
  d1Exec(w.dir, `INSERT INTO serverless_endpoints (id, name, endpoint_id, template_id, mode, spec, status, created_at, created_by, updated_at) VALUES ('se_uiq', 'h3-max2', 'lp85qdnkl6mock', 'tpl', 'queue', '${spec}', 'scaled-down', ${now - 3600_000}, 'owner', ${now})`);
  await settle();
  for (const [id, st] of [["4f6a1c2e-8b0d-4e9f-a3c7-5d2b1e0f9a86-e1", "IN_QUEUE"], ["9b3e7d1a-2c4f-4a6b-8e0d-7f5c3a1b9d24-e1", "IN_QUEUE"]]) mock.queue.jobs.set(id, { id, status: st });
  mock.queue.queued = 4;
  mock.queue.workers = 0;
  mock.internalJobs[U(1)] = [200, { id: U(1), status: "running", cancel_requested: true }];
  mock.internalJobs[U(2)] = [200, { id: U(2), status: "cancelled" }];
  mock.internalJobs[U(3)] = [200, { id: U(3), status: "cancelled" }];

  // ---- the cluster page's Jobs card
  await go(`#/cluster/${tiny.cluster.id}`);
  const card = `[data-jobs-card="${tiny.cluster.id}"]`;
  await page.waitForSelector(`${card} td:has-text("fvjob_9f2c41d0a7b34e1d")`);
  assert.equal(await page.locator(`${card} tbody tr`).count(), 3, "queued and running by default");
  assert.match(await page.textContent(card), /2 queued · 1 running/);
  assert.match(await page.textContent(`${card} tbody`), /edge queue/);
  await shot("01-cluster-jobs");
  await page.click(`${card} button:text("All recent")`);
  await page.waitForSelector(`${card} td:has-text("4c0f7e2a-5d3b-4f8e-9a61-0c2d8b7e1f33")`);
  assert.equal(await page.locator(`${card} tbody tr`).count(), 5);
  await shot("02-cluster-jobs-all");
  await page.click(`${card} button:text("Queued and running")`);
  await page.waitForSelector(`${card} tbody tr >> nth=2`);
  // Cancel the running one: straight to its worker.
  await page.click(`${card} tr:has-text("fvjob_9f2c41d0a7b34e1d") button:text("Cancel")`);
  await page.waitForSelector(`${card} :text("via internal on ${pid}")`);
  assert.deepEqual(mock.internalCancels.at(-1), { pod: pid, id: U(1) });
  await shot("03-cancel-running");
  // Cancel all queued: a schema-bound dialog with the count.
  await page.click(`${card} button:has-text("Cancel all queued (2)")`);
  await page.waitForSelector('dialog [data-schema-form="jobs-cancel-queued"]');
  assert.match(await page.textContent("dialog.ff-dialog"), /2 jobs queued/);
  assert.equal(await page.$eval('dialog [data-ctl][data-path="pool"]', (e) => e.tagName), "SELECT");
  await audit();
  await shot("04-cancel-queued-dialog");
  await page.click('dialog.ff-dialog button[type=submit]');
  await page.waitForSelector(`${card} :text("cancelled 2 of 2 queued")`);
  assert.ok(mock.internalCancels.some((x) => x.id === U(2)), "the job queued at the edge went through a front");
  await shot("05-cancel-queued-done");
  // What the workers write back to D1 after those cancels (the mock pods do not).
  d1Exec(w.dir, `UPDATE jobs SET status = 'cancelled' WHERE id IN ('${U(2)}', '${U(3)}')`, "fv-edge");
  await settle();
  // Cancel by id: the job-cancel form; a wrong id is refused inline before anything is sent.
  await page.click(`${card} button:text("Cancel by id…")`);
  await page.waitForSelector('dialog [data-schema-form="job-cancel"]');
  await page.fill('dialog [data-ctl][data-path="job"]', "not an id!");
  await page.waitForSelector('dialog .ff-field[data-path="job"].has-err');
  assert.ok(await page.$eval("dialog.ff-dialog button[type=submit]", (b) => b.disabled), "Cancel job is off while the id is invalid");
  await audit();
  await shot("06-cancel-by-id-invalid");
  await page.fill('dialog [data-ctl][data-path="job"]', "fvjob_9f2c41d0a7b34e1d");
  await page.waitForFunction(() => !document.querySelector("dialog.ff-dialog button[type=submit]").disabled);
  await page.click("dialog.ff-dialog button[type=submit]");
  await page.waitForSelector(`${card} :text("fvjob_9f2c41d0a7b34e1d (native) via internal")`);

  // ---- the Jobs page
  await go(`#/jobs?cluster=${tiny.cluster.id}`);
  await page.waitForSelector(`[data-jobs-card="${tiny.cluster.id}"] h2:text("Jobs: tiny")`);
  await shot("07-jobs-page");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(200);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 1, `no horizontal page scroll at phone width (${overflow}px)`);
  await shot("08-jobs-phone");
  await page.setViewportSize({ width: 1440, height: 920 });

  // ---- serverless: the Queue card
  await go("#/serverless?ep=se_uiq");
  await page.waitForSelector('h2:text("Queue: cancel and purge")');
  // What it serves (the incident's spec: h3-max, no config): the inferred preset, sol-h3, and test invokes built from it.
  await page.waitForSelector('[data-serves] tr[data-model="sol-h3"]');
  assert.match(await page.textContent("[data-serves]"), /preset h3-max \(inferred/);
  const opts = await page.$$eval('select[aria-label="Example request"] option', (os) => os.map((o) => o.textContent));
  assert.ok(opts.some((o) => /^MiniMax V2 · t2v · Max \(sol-h3\)/.test(o)), opts.join("\n"));
  assert.ok(!opts.some((o) => /MiniMax-H3-Turbo/.test(o)), "h3-max serves no MiniMax-H3-Turbo");
  const mm = opts.findIndex((o) => /^MiniMax V2 · i2v/.test(o));
  await page.selectOption('select[aria-label="Example request"]', String(mm));
  await page.fill('input[aria-label="Reference image URL"]', "https://example.com/first.jpg");
  assert.ok(!(await page.isVisible('input[aria-label="Audio URL"]')), "an image example asks for no audio");
  const inv = JSON.parse(await page.inputValue('textarea[aria-label="Invoke input"]'));
  assert.equal(inv.kind, "http");
  assert.equal(inv.body.content[1].image_url.url, "https://example.com/first.jpg");
  await shot("09a-sls-serves-and-examples");
  await page.click('button:has-text("Cancel a job…")');
  await page.waitForSelector('dialog [data-schema-form="serverless-cancel"]');
  assert.ok(["SELECT", "radiogroup"].includes(await page.$eval('dialog [data-ctl][data-path="fv_api"]', (e) => (e.tagName === "SELECT" ? "SELECT" : e.getAttribute("role")))), "the API is a select or segmented control, never free text");
  await audit();
  await page.fill('dialog [data-ctl][data-path="job"]', "4f6a1c2e-8b0d-4e9f-a3c7-5d2b1e0f9a86-e1");
  await shot("09-sls-cancel-dialog");
  await page.waitForFunction(() => !document.querySelector("dialog.ff-dialog button[type=submit]").disabled);
  await page.click("dialog.ff-dialog button[type=submit]");
  await page.waitForSelector('#slsCancelOut .badge:text("CANCELLED")');
  assert.match(await page.textContent("#slsCancelOut"), /IN_QUEUE →/);
  await shot("10-sls-cancelled");
  // Purge: the dialog shows the queued count; a wrong name is refused by the server, readably.
  await page.click('button:has-text("Purge queue")');
  await page.waitForSelector('dialog [data-schema-form="serverless-purge"]');
  assert.match(await page.textContent("dialog.ff-dialog"), /3 jobs queued, 0 running/);
  await audit();
  await page.fill('dialog [data-ctl][data-path="confirm"]', "h3-max");
  await page.waitForFunction(() => !document.querySelector("dialog.ff-dialog button[type=submit]").disabled);
  await shot("11-sls-purge-dialog");
  await page.click("dialog.ff-dialog button[type=submit]");
  await page.waitForSelector("#slsCancelOut :text(\"type the endpoint's name (h3-max2)\")");
  assert.equal(mock.queue.purges, 0);
  await page.click('button:has-text("Purge queue")');
  await page.waitForSelector('dialog [data-schema-form="serverless-purge"]');
  await page.fill('dialog [data-ctl][data-path="confirm"]', "h3-max2");
  await page.waitForFunction(() => !document.querySelector("dialog.ff-dialog button[type=submit]").disabled);
  await page.click("dialog.ff-dialog button[type=submit]");
  await page.waitForSelector('#slsCancelOut .badge:text("purged")');
  assert.match(await page.textContent("#slsCancelOut"), /removed 3 queued job\(s\); queue 3 → 0/);
  assert.equal(mock.queue.purges, 1);
  await shot("12-sls-purged");
}
