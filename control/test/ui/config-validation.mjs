// Typed, validated configuration in Chromium (run from test/ui/explorer.mjs,
// which starts the Worker and seeds a running `tiny` and a defined
// `h3-and-ltx` cluster; docs/control/config-validation.md): for every form,
// enums are selects / segmented controls, the rule shows before typing,
// invalid input gets an inline error and keeps Save / Launch / Create off,
// the coverage audit finds no control without a schema entry, and the API
// refuses the same input without the UI.
import assert from "node:assert/strict";

export async function configValidationChecks({ page, B, api, raw, out }) {
  const shot = (n) => page.screenshot({ path: `${out}/cv-${n}.png`, fullPage: true });
  const go = async (hash) => {
    await page.evaluate(() => document.querySelector("#main").replaceChildren());
    if (await page.evaluate((h) => location.hash === h, hash)) await page.reload();
    else await page.goto(`${B}/${hash}`);
  };
  const audit = async () => {
    const probs = await page.evaluate(async () => {
      const s = (await (await fetch("/api/schemas", { credentials: "same-origin" })).json()).schemas;
      return window.FVEditor.auditForms(document, s);
    });
    assert.deepEqual(probs, [], `coverage audit: ${probs.join("\n")}`);
  };
  const disabled = (sel) => page.$eval(sel, (b) => b.disabled);
  const errAt = (form, path, re) => page.waitForSelector(`[data-schema-form="${form}"] .ff-field[data-path="${path}"].has-err .cf-err:not([hidden])${re ? `:text-matches("${re}", "i")` : ""}`);
  const clearAt = (form, path) => page.waitForSelector(`[data-schema-form="${form}"] .ff-field[data-path="${path}"]:not(.has-err)`);
  page.on("dialog", (d) => d.accept());

  // ---------------------------------------------------------------- standalone launch
  await go("#/standalone");
  const SL = '[data-schema-form="standalone-launch"]';
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  // Enums are selects / segmented controls; the name's rule shows before typing.
  assert.equal(await page.$eval(`${SL} [data-ctl][data-path="preset"]`, (e) => e.tagName), "SELECT");
  assert.equal(await page.$eval(`${SL} [data-ctl][data-path="compute"]`, (e) => e.getAttribute("role")), "radiogroup");
  assert.match(await page.textContent(`${SL} .ff-field[data-path="name"] .ff-rule`), /lower-case letters/);
  const launch = `${SL} button[type=submit]`;
  assert.ok(await disabled(launch), "Launch is off with no name");
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "Bad Name");
  await errAt("standalone-launch", "name");
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "tiny");
  await errAt("standalone-launch", "name", "exists");
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "new");
  await errAt("standalone-launch", "name", "reserved");
  await page.fill(`${SL} [data-ctl][data-path="name"]`, "cv-solo");
  await clearAt("standalone-launch", "name");
  // The live price check (needs an otherwise valid launch): every GPU type of the pod costs more than $0.50/hr.
  await page.fill(`${SL} [data-ctl][data-path="max_gpu_dph"]`, "0.5");
  await errAt("standalone-launch", "max_gpu_dph", "deleted right after create");
  await page.fill(`${SL} [data-ctl][data-path="deadline_min"]`, "2");
  await errAt("standalone-launch", "deadline_min");
  await page.click(`${SL} button:text('+ variable')`);
  await page.fill(`${SL} .ff-env input[aria-label="variable name"]`, "FV_ADMIN_TOKEN");
  await page.waitForSelector(`${SL} .ff-env tr.has-err .cf-err:has-text('set by the controller')`);
  assert.ok(await disabled(launch), "Launch stays off while invalid");
  await page.waitForSelector(`${SL} .ff-summary .ff-issues a[data-path="deadline_min"]`);
  await shot("01-standalone-invalid");
  // Fix everything: Launch turns on.
  await page.fill(`${SL} [data-ctl][data-path="deadline_min"]`, "30");
  await page.fill(`${SL} [data-ctl][data-path="max_gpu_dph"]`, "");
  await page.click(`${SL} .ff-env button[aria-label^="remove"]`);
  await page.waitForFunction((s) => !document.querySelector(s).disabled, launch, { timeout: 15000 });
  // A custom pod: the variant is a select; the cpu variant brings CPU fields (vCPU a select).
  await page.selectOption(`${SL} [data-ctl][data-path="preset"]`, "");
  await page.waitForSelector(`${SL} select[data-ctl][data-path="variant"]`);
  await page.waitForSelector(`${SL} select[data-ctl][data-path="vcpu"]`);
  await audit();
  await shot("02-standalone-valid-custom");
  // GMI Cloud (docs/serve/deploy-gmi-brev.md): a select; picking it hides the Runpod fields and shows the provider's;
  // here GMI has a key but no GMI_PRODUCTS, so the option and the form say why it is off, and the server refuses it.
  assert.equal(await page.$eval(`${SL} [data-ctl][data-path="provider"]`, (e) => e.tagName), "SELECT");
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, "gmi");
  await page.waitForSelector(`${SL} .ff-field[data-path="provider_gpu"]`);
  await page.waitForSelector(`${SL} .ff-field[data-path="weights_source"]`);
  assert.equal(await page.$(`${SL} .ff-field[data-path="gpu_types"]`), null, "Runpod GPU types hidden");
  assert.equal(await page.$(`${SL} .ff-field[data-path="region"]`), null, "Runpod region hidden");
  await page.waitForSelector(`${SL} p:has-text("GMI_PRODUCTS is not set")`);
  await errAt("standalone-launch", "provider", "GMI Cloud is off");
  assert.ok(await disabled(launch), "Launch is off while the provider is off");
  await audit();
  await shot("02b-standalone-gmi-off");
  await page.selectOption(`${SL} [data-ctl][data-path="provider"]`, "");
  await page.waitForSelector(`${SL} .ff-field[data-path="region"]`);

  // ---------------------------------------------------------------- serverless
  await go("#/serverless");
  const SV = '[data-schema-form="serverless-endpoint"]';
  await page.waitForSelector(`${SV} .ff-field[data-path="name"]`);
  const create = `${SV} button.primary`;
  assert.ok(await disabled(create));
  await page.fill(`${SV} [data-ctl][data-path="name"]`, "tick");
  await errAt("serverless-endpoint", "name", "reserved");
  await page.fill(`${SV} [data-ctl][data-path="name"]`, "cv-ep");
  await page.fill(`${SV} [data-ctl][data-path="workers_min"]`, "2");
  await page.fill(`${SV} [data-ctl][data-path="workers_max"]`, "1");
  await errAt("serverless-endpoint", "workers_min", "above workers_max");
  await page.fill(`${SV} [data-ctl][data-path="scaler_value"]`, "0.5");
  await errAt("serverless-endpoint", "scaler_value");
  assert.equal(await page.$eval(`${SV} [data-ctl][data-path="mode"]`, (e) => e.getAttribute("role")), "radiogroup");
  assert.ok(await disabled(create));
  await shot("03-serverless-invalid");
  await page.fill(`${SV} [data-ctl][data-path="workers_min"]`, "0");
  await page.fill(`${SV} [data-ctl][data-path="scaler_value"]`, "4");
  await page.waitForFunction((s) => !document.querySelector(s).disabled, create, { timeout: 15000 });
  // The JSON tab: invalid JSON keeps Create off.
  await page.click(`${SV} .tabs button:text('JSON')`);
  await page.waitForSelector(`${SV} .sv-json .cm-content`);
  await page.evaluate(() => {
    const v = window.FVEditor.viewOf(document.querySelector(".sv-json .cm-editor"));
    v.dispatch({ changes: { from: 0, to: 1, insert: "{{" } });
  });
  await page.waitForSelector(`${SV} .sv-json :text("not valid JSON")`);
  assert.ok(await disabled(create), "Create is off while the JSON does not parse");
  await shot("04-serverless-json-invalid");
  await page.evaluate(() => {
    const v = window.FVEditor.viewOf(document.querySelector(".sv-json .cm-editor"));
    v.dispatch({ changes: { from: 0, to: 2, insert: "{" } });
  });
  await page.click(`${SV} .tabs button:text('Form')`);
  await audit();
  // Model-first: the preset is a select, and the Serves panel follows it live.
  assert.equal(await page.$eval(`${SV} [data-ctl][data-path="preset"]`, (e) => e.tagName), "SELECT");
  await page.waitForSelector(`${SV} [data-serves] tr[data-model="fake-wan"]`);
  await page.selectOption(`${SV} [data-ctl][data-path="preset"]`, JSON.stringify("h3-ref2v"));
  await page.waitForSelector(`${SV} [data-serves] tr[data-model="h3-ref2v-turbo"]`);
  assert.match(await page.textContent(`${SV} [data-serves]`), /MiniMax-H3-Turbo → h3-ref2v-turbo/);
  assert.ok(await page.$(`${SV} details summary:text-matches("Advanced / custom", "i")`), "variant and config sit under Advanced with a preset");
  await page.waitForFunction((s) => !document.querySelector(s).disabled, create, { timeout: 15000 });
  await shot("04a-serverless-preset-h3-ref2v");
  await audit();
  await page.selectOption(`${SV} [data-ctl][data-path="preset"]`, JSON.stringify("cpu"));
  await page.waitForSelector(`${SV} [data-serves] tr[data-model="fake-wan"]`);

  // ---------------------------------------------------------------- settings: tokens, policies
  await go("#/settings");
  const TK = '[data-schema-form="token-create"]';
  await page.waitForSelector(`${TK} .ff-field[data-path="name"]`);
  assert.equal(await page.$$eval(`${TK} [data-ctl][data-path="scope"] button`, (b) => b.map((x) => x.textContent).join(",")), "read,admin,ci");
  await page.fill(`${TK} [data-ctl][data-path="name"]`, "a b");
  await errAt("token-create", "name");
  await page.fill(`${TK} [data-ctl][data-path="name"]`, "seed");
  await errAt("token-create", "name", "exists");
  await page.fill(`${TK} [data-ctl][data-path="ttl_days"]`, "400");
  await errAt("token-create", "ttl_days");
  assert.ok(await disabled(`${TK} button:text('Mint token')`));
  await page.waitForSelector(".fv-panel[data-kind=build-pods] .fv-form");
  await audit();
  await shot("05-settings-token-invalid");
  // The build pods policy: a vCPU size Runpod does not have is not even offered (chips), and the server refuses it.
  assert.ok(!(await page.$(".fv-panel[data-kind=build-pods] [data-path=vcpus] input[type=text]")), "vcpus are chips");

  // ---------------------------------------------------------------- releases
  await go("#/releases");
  const RL = '[data-schema-form="release-dispatch"]';
  await page.waitForSelector(`${RL} .ff-field[data-path="target"]`);
  assert.equal(await page.$eval(`${RL} [data-ctl][data-path="channel"]`, (e) => e.tagName), "SELECT");
  assert.ok(await disabled(`${RL} button:text('Dispatch release.yml')`), "promote needs a target");
  await page.fill(`${RL} [data-ctl][data-path="target"]`, "not a target");
  await errAt("release-dispatch", "target");
  await shot("06-release-invalid");
  await audit();

  // ---------------------------------------------------------------- env editor
  const cl = (await api("/api/clusters")).clusters.find((c) => c.name === "tiny");
  await go(`#/env?cluster=${cl.id}`);
  const envp = ".fv-panel[data-kind=env] >> nth=1";
  await page.waitForSelector(`${envp} >> button:text('+ variable')`);
  await page.click(`${envp} >> button:text('+ variable')`);
  await page.fill(`${envp} >> input[aria-label="variable name"]`, "FV_INTERNAL_TOKEN");
  await page.waitForSelector(`${envp} >> .ff-env tr.has-err .cf-err:has-text('set by the controller')`);
  await page.waitForFunction(() => document.querySelectorAll(".fv-panel[data-kind=env]")[1].querySelector(".fv-bar button.primary").disabled, null, { timeout: 15000 });
  await page.fill(`${envp} >> input[aria-label="variable name"]`, "FASTVIDEO_DIT_OFFLOAD");
  await page.press(`${envp} >> input[aria-label="variable name"]`, "Tab");
  // A known engine key: its value is a select of what the engine accepts.
  await page.waitForSelector(`${envp} >> tr[data-key=FASTVIDEO_DIT_OFFLOAD] select`);
  await shot("07-env-typed");
  await page.click(`${envp} >> button[aria-label="remove FASTVIDEO_DIT_OFFLOAD"]`);

  // ---------------------------------------------------------------- cluster config + typed dialogs
  await go("#/clusters/new?template=tiny-cpu");
  await page.waitForSelector('.cf-field[data-path="name"] input');
  await page.fill('.cf-field[data-path="name"] input', "validate");
  await page.waitForSelector('.cf-field[data-path="name"].has-err .cf-err:has-text("reserved")');
  assert.ok(await disabled("button:text('Review & define')"));
  await page.click('.cf-pool[data-path="pools.0"] .cf-caret');
  assert.equal(await page.$eval('.cf-field[data-path="pools.0.vcpu"] select', (e) => e.tagName), "SELECT");
  assert.ok(!(await page.$('.cf-field[data-path="pools.0.fake_models"] input[type=text]')), "fake models are chips");
  await audit();
  await shot("08-cluster-config-invalid");
  await page.click(".cf-main .tabs button:text('JSON')");
  await page.waitForSelector(".cf-json .cm-content");
  await page.evaluate(() => {
    const v = window.FVEditor.viewOf(document.querySelector(".cf-json .cm-editor"));
    v.dispatch({ changes: { from: 0, to: 1, insert: "[" } });
  });
  await page.waitForSelector(".cf-issue:has-text('not valid JSON')");
  assert.ok(await disabled("button:text('Review & define')"));
  await shot("09-cluster-json-invalid");
  await go(`#/cluster/${cl.id}`);
  await page.waitForSelector("button:text('Extend…'):not([disabled])");
  await page.click("button:text('Extend…')");
  await page.waitForSelector("dialog.ff-dialog [data-ctl][data-path=minutes]");
  await page.fill("dialog.ff-dialog [data-ctl][data-path=minutes]", "0");
  await page.waitForSelector("dialog.ff-dialog .ff-field.has-err");
  assert.ok(await disabled("dialog.ff-dialog button[type=submit]"));
  await shot("10-extend-dialog-invalid");
  await page.click("dialog.ff-dialog button:text('Cancel')");

  // ---------------------------------------------------------------- log explorer filters
  await go("#/logs?from=3h");
  await page.waitForSelector(".lx-time[aria-label=From]");
  await page.fill(".lx-time[aria-label=From]", "yesterday");
  await page.waitForSelector('.lx-time[aria-label=From][aria-invalid="true"]');
  await page.click("button.lx-tog[title='Regular expression']");
  await page.fill(".lx-search", "job (");
  await page.waitForSelector(".lx-err:has-text('regex')");
  await shot("11-logs-invalid-filters");
  await page.fill(".lx-search", "");
  await page.fill(".lx-time[aria-label=From]", "");

  // ---------------------------------------------------------------- dark mode, phone width
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "dark"));
  await go("#/standalone");
  await page.waitForSelector(`${SL} .ff-field[data-path="name"]`);
  await shot("12-standalone-dark");
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  await page.setViewportSize({ width: 390, height: 844 });
  for (const h of ["#/standalone", "#/serverless", "#/settings"]) {
    await go(h);
    await page.waitForSelector("[data-schema-form] .ff-field");
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert.ok(over <= 1, `${h} at phone width scrolls sideways (${over}px)`);
  }
  await shot("13-serverless-phone");
  await page.setViewportSize({ width: 1440, height: 920 });

  // ---------------------------------------------------------------- the server refuses the same input
  const st = async (p, body, method = "POST") => (await raw(p, body, method)).status;
  assert.equal(await st("/api/standalone", { name: "cv-x", preset: "h3-turbo", deadline_min: 2, start: false }), 400);
  assert.equal(await st("/api/standalone", { name: "tiny", preset: "h3-turbo", start: false }), 409);
  assert.equal(await st("/api/serverless", { spec: { name: "cv-y", scaler_value: 0.5 } }), 400);
  assert.equal(await st("/api/build-pods/policy", { policy: { vcpus: [3] } }, "PUT"), 400);
  assert.equal(await st("/api/serverless/policy", { policy: { max_workers: 999 } }, "PUT"), 400);
  assert.equal(await st("/api/github/release", { action: "promote", channel: "stable" }), 400);
  assert.equal(await st("/api/github/release", { action: "promote", channel: "nightly-x", target: "abcdef1" }), 400);
  assert.equal(await st(`/api/clusters/${cl.id}/extend`, { minutes: 0 }), 400);
  assert.equal(await st(`/api/clusters/${cl.id}/scale`, { pool: "fake", count: 9 }), 400);
  assert.equal(await st(`/api/clusters/${cl.id}/scale`, { pool: "nope", count: 1 }), 400);
  assert.equal(await st(`/api/clusters/${cl.id}/mint-key`, { name: "a/b" }), 400);
  assert.equal(await st("/api/clusters", { spec: { name: "new", template: "tiny-cpu" } }), 400);
  assert.equal(await st("/api/logs/query?from=yesterday", undefined, "GET"), 400);
  const n = await api("/api/names/cluster?name=tiny");
  assert.equal(n.taken, true);
  assert.equal((await api("/api/names/endpoint?name=policy")).ok, false);
}
