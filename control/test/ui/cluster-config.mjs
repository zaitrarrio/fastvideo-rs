// The cluster configuration page in Chromium (run from test/ui/explorer.mjs,
// which starts the Worker and seeds a running `tiny` and a defined
// `h3-and-ltx` cluster): inline errors from the API's validator, price and
// the Runpod stock warning, define, the diff before saving, clone, scale a
// running pool with the operation's live log, the effective env per pod.
import assert from "node:assert/strict";

export async function clusterEditorChecks({ page, B, api, out, tiny, big }) {
  page.on("dialog", (d) => d.accept());
  const fresh = async (hash) => {
    await page.evaluate(() => document.querySelector("#main").replaceChildren());
    if (await page.evaluate((h) => location.hash === h, hash)) await page.reload();
    else await page.goto(`${B}/${hash}`);
  };
  // ---- the clusters list: new, configure, clone, delete
  await fresh("#/clusters");
  await page.waitForSelector("h1:text('Clusters')");
  assert.ok(await page.isVisible("a.btn:text('Configure')"));
  await page.selectOption("select[aria-label=Template]", "standard");
  await page.click("button:text('New cluster')");
  // ---- a new standard cluster: no name yet → an inline error at the name
  await page.waitForSelector(".cf-form .cf-pool");
  await page.waitForSelector('.cf-field[data-path="name"].has-err .cf-err:not(:empty)');
  await page.fill('.cf-field[data-path="name"] input', "eu-main");
  // Too many workers in a pool: the API's limit shows at that field.
  await page.fill('.cf-pool[data-path="pools.0"] input.cf-count', "12");
  await page.waitForSelector('.cf-stepper.has-err, .cf-pool[data-path="pools.0"].has-err');
  await page.waitForSelector(".cf-issue.err:has-text('pools.0.count')");
  await page.screenshot({ path: `${out}/config-01-new-invalid.png`, fullPage: true });
  await page.fill('.cf-pool[data-path="pools.0"] input.cf-count', "1");
  await page.waitForSelector(".cf-issues .badge.good");
  // Price and the stock warning: four pools want RTX PRO 6000 in EUR-IS-1, Runpod reports 2 free (the 2026-10-06 failure).
  await page.waitForSelector(".cf-stockwarn:has-text('RTX PRO 6000 in EUR-IS-1')");
  const warn = await page.textContent(".cf-stockwarn");
  assert.match(warn, /4 pod\(s\).*2 free/);
  assert.ok((await page.$$(".cf-pool-head .badge.warn")).length >= 4, "a stock badge on every GPU pool");
  await page.screenshot({ path: `${out}/config-02-new-price-stock.png`, fullPage: true });
  // Open a pool: every field is there.
  await page.click('.cf-pool[data-path="pools.1"] .cf-caret');
  for (const k of ["id", "variant", "compute", "gpu_types", "regions", "container_disk_gb", "volume", "image", "family", "config", "models", "fake_models", "max_queued", "job_timeout_s", "stale_after_s"])
    assert.ok(await page.$(`.cf-field[data-path="pools.1.${k}"]`), `pool field ${k}`);
  for (const k of ["name", "control_plane", "auth", "image", "regions", "cap_s", "max_gpu_dph", "min_start", "balance_floor", "min_balance", "auto_stop_idle_min", "log_shipping", "log_level"])
    assert.ok(await page.$(`.cf-field[data-path="${k}"]`), `field ${k}`);
  // Define: the review shows the spec, then the page becomes the cluster's.
  await page.click("button:text('Review & define')");
  await page.waitForSelector("dialog.cf-review");
  await page.click("dialog.cf-review button:text('Define')");
  await page.waitForFunction(() => /#\/cluster\/c_[0-9a-f]+\/config/.test(location.hash));
  await page.waitForSelector(".cf-side .badge:text('defined')");
  // ---- edit a saved cluster: the diff before saving
  await page.click(".cf-field[data-path=cap_s] button:text('2h')");
  await page.waitForSelector(".cf-changes .fv-diff .add");
  await page.click("button:text('Review & save')");
  await page.waitForSelector("dialog.cf-review .fv-diff .del");
  await page.waitForSelector("dialog.cf-review .fv-plan li");
  await page.screenshot({ path: `${out}/config-03-review-diff.png` });
  await page.click("dialog.cf-review button:has-text('Save (v0')");
  await page.waitForSelector(".cf-side .muted.small:text('saved')");
  const id = await page.evaluate(() => location.hash.split("/")[2]);
  const doc = await api(`/api/docs/cluster-spec/${id}`);
  assert.equal(doc.doc.cap_s, 7200);
  assert.equal(doc.version, 1);
  // The JSON tab is the same document.
  await page.click(".cf-main .tabs button:text('JSON')");
  await page.waitForSelector(".cf-json .cm-content");
  assert.match(await page.evaluate(() => window.FVEditor.viewOf(document.querySelector(".cf-json .cm-editor")).state.doc.toString()), /"cap_s": 7200/);
  await page.click(".cf-main .tabs button:text('Form')");
  // ---- clone
  await fresh(`#/clusters/new?clone=${big.cluster.id}`);
  await page.waitForSelector('.cf-field[data-path="name"] input');
  assert.equal(await page.inputValue('.cf-field[data-path="name"] input'), "h3-and-ltx-copy");
  // ---- a running cluster: scale a pool from the page and watch the operation's log
  await fresh(`#/cluster/${tiny.cluster.id}/config`);
  await page.waitForSelector(".cf-side .badge:text('running')");
  await page.click('.cf-pool[data-path="pools.0"] .cf-stepper button[aria-label=more]');
  await page.waitForSelector("button:text('Scale to 2')");
  await page.click("button:text('Scale to 2')");
  await page.waitForSelector(".cf-oplog h2:has-text('scale')");
  await page.waitForSelector(".cf-oplog .badge:text('done')", { timeout: 90_000 });
  assert.ok((await page.$$(".cf-oplines div")).length > 0, "the operation's log");
  await page.waitForSelector("#cf-env details summary");
  await page.click("#cf-env details summary >> nth=0");
  await page.waitForSelector(".cf-envtable code:text('FV_LOG_SHIP_URL'), .cf-envtable code");
  await page.screenshot({ path: `${out}/config-04-running-scale-oplog-env.png`, fullPage: true });
  // Dark theme.
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await page.screenshot({ path: `${out}/config-05-dark.png`, fullPage: false });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  // Phone width: no horizontal page scroll.
  await page.setViewportSize({ width: 390, height: 844 });
  await fresh(`#/cluster/${tiny.cluster.id}/config`);
  await page.waitForSelector(".cf-pool");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 1, `config page at phone width scrolls sideways (${overflow}px)`);
  await page.screenshot({ path: `${out}/config-06-phone.png`, fullPage: false });
  await page.setViewportSize({ width: 1440, height: 920 });
}
