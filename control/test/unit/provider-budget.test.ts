// The GMI / Brev monthly budget as the owner enters it (owner report 2026-10-18: "when I try to launch a Brev instance
// it is not recognizing the budget I enter it is always sets the budget to $10"). Before: every check read only the
// Worker var BREV_BUDGET_USD (staging: 10) and the console had no place to set it. Now: PUT
// /api/v1/providers/<p>/budget (the launch form's "Monthly budget" field, the CLI) saves it in the policies
// (brev_budget_usd), and the provider view, the launch form's details, the launch validator and the start's
// budgetCheck all read that value back; null goes back to the Worker var.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_POLICIES } from "../../src/alerts";
import { mintApiToken } from "../../src/auth";
import { resetBrevTypes } from "../../src/brev";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { budgetCheck, providerBudget } from "../../src/providers";
import { clearTtlCache } from "../../src/ttlcache";
import { d1 } from "./d1shim";

const realFetch = globalThis.fetch;
function upstream() {
  globalThis.fetch = (async (input: any) => {
    const url = typeof input === "string" ? input : input.url;
    if (new URL(url).host === "brev.test" && url.includes("/alltypesavailable/"))
      return Response.json({ allInstanceTypes: [{ type: "g5.xlarge", cloud_cred_id: "cred-aws-1", stoppable: true, base_price: { currency: "USD", amount: "1.25" }, location: "us-east-1", is_available: true }] });
    return new Response(`unexpected upstream ${url}`, { status: 599 });
  }) as typeof fetch;
}
const mkEnv = (over: Partial<Env> = {}) =>
  ({
    DB: d1(),
    CONTROL_KEK: b64(randomBytes(32)),
    SESSION_SECRET: "s".repeat(40),
    RUNPOD_API_KEY: "rpa_SECRETKEY123",
    PUBLIC_URL: "https://ctl.test",
    BREV_API_TOKEN: "brev_SECRET_0123456789",
    BREV_API_URL: "https://brev.test",
    BREV_ORG_ID: "org1",
    BREV_INSTANCE_TYPES: "g5.xlarge",
    // The staging value the owner kept hitting.
    BREV_BUDGET_USD: "10",
    ...over,
  }) as unknown as Env;
const ctx = { waitUntil() {}, passThroughOnException() {} } as any;
async function client(env: Env, scope: "admin" | "read" = "admin") {
  const auth = { authorization: `Bearer ${(await mintApiToken(env, `t-${scope}`, scope, "test")).token}` };
  return async (path: string, init: { method?: string; body?: unknown } = {}) => {
    const r = await _app.fetch(new Request(`https://ctl.test${path}`, { method: init.method || "GET", headers: { ...auth, "content-type": "application/json" }, body: init.body === undefined ? undefined : JSON.stringify(init.body) }), env, ctx);
    return { status: r.status, j: (await r.json()) as any };
  };
}
/** A Brev launch of the fake engine on g5.xlarge ($1.25/hr): 12 h until the deadline = $15 (over $10, within $75). */
const launch = { name: "budget-a", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], channel: "stable", deadline_min: 720, env: {}, provider: "brev", provider_gpu: "g5.xlarge", weights_source: "none" };

beforeEach(() => (clearTtlCache(), resetBrevTypes(), upstream()));
afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("the monthly budget the owner enters is the one used", () => {
  it("policy unset: the Worker var; the policy, once set, wins; null goes back to the Worker var", async () => {
    const env = mkEnv();
    expect(DEFAULT_POLICIES.brev_budget_usd).toBeNull();
    expect(await providerBudget(env, "brev")).toEqual({ usd: 10, source: "worker_var", worker_var: 10 });
    expect(await providerBudget(env, "brev", { ...DEFAULT_POLICIES, brev_budget_usd: 75 })).toEqual({ usd: 75, source: "policy", worker_var: 10 });
    // 0 is a budget (nothing may be spent), not "unset".
    expect((await providerBudget(env, "brev", { ...DEFAULT_POLICIES, brev_budget_usd: 0 })).usd).toBe(0);
    expect(await providerBudget(mkEnv({ BREV_BUDGET_USD: undefined }), "brev")).toEqual({ usd: null, source: null, worker_var: null });
  });

  it("PUT /api/v1/providers/brev/budget saves 75: the provider view, the form's details, the policies and the checks read 75", async () => {
    const env = mkEnv();
    const api = await client(env);
    // Before: the Worker var's $10, and a $15 launch is refused against it.
    expect((await api("/api/v1/providers?provider=brev")).j.providers[0]).toMatchObject({ budget_usd: 10, budget_source: "worker_var" });
    const before = await budgetCheck(env, "brev", 1.25, 12);
    expect(before.ok).toBe(false);
    expect(before.reasons[0]).toMatch(/over the monthly budget \$10 \(raise it: the launch form's "Monthly budget" field or Settings → Policies brev_budget_usd\)/);

    const put = await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: 75 } });
    expect(put.status).toBe(200);
    expect(put.j.provider).toMatchObject({ id: "brev", budget_usd: 75, budget_source: "policy", month_usd: 0 });

    // Read back from the same store by every reader.
    expect((await api("/api/v1/providers?provider=brev")).j.providers[0]).toMatchObject({ budget_usd: 75, budget_source: "policy" });
    expect((await api("/api/v1/schemas/dynamic?provider=brev")).j.budget).toMatchObject({ budget_usd: 75, budget_source: "policy" });
    expect((await api("/api/v1/policies")).j.policies.brev_budget_usd).toBe(75);
    const after = await budgetCheck(env, "brev", 1.25, 12);
    expect(after).toMatchObject({ ok: true, budget: 75, projected: 15 });
    // The save is audited with the value.
    const audit = await env.DB.prepare("SELECT action, target, detail FROM audit WHERE target = 'policies:default' ORDER BY id DESC LIMIT 1").first<any>();
    expect(audit).toMatchObject({ action: "doc.save", detail: "brev_budget_usd = $75" });

    // null: back to the Worker var.
    expect((await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: null } })).j.provider).toMatchObject({ budget_usd: 10, budget_source: "worker_var" });
  });

  it("the launch validator says the budget refusal at the provider before Launch, and clears once a larger budget is saved", async () => {
    const env = mkEnv();
    const api = await client(env);
    const r = await api("/api/v1/standalone/validate", { method: "POST", body: launch });
    expect(r.status).toBe(200);
    const issue = r.j.issues.find((i: any) => i.path[0] === "provider");
    expect(issue?.message).toMatch(/NVIDIA Brev: \$0\.00 spent this month \+ \$1\.25\/hr for 12\.00 h = \$15\.00, over the monthly budget \$10/);
    await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: 75 } });
    const ok = await api("/api/v1/standalone/validate", { method: "POST", body: launch });
    expect(ok.j.issues).toEqual([]);
    // A 60-minute launch fits even the Worker var's $10.
    await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: null } });
    expect((await api("/api/v1/standalone/validate", { method: "POST", body: { ...launch, deadline_min: 60 } })).j.issues).toEqual([]);
  });

  it("refuses a bad value, a provider without a budget, and a read-only token; the policies document validates it too", async () => {
    const env = mkEnv();
    const api = await client(env);
    expect((await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: "abc" } })).status).toBe(400);
    expect((await api("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: -5 } })).status).toBe(400);
    expect((await api("/api/v1/providers/runpod/budget", { method: "PUT", body: { budget_usd: 5 } })).status).toBe(404);
    const ro = await client(env, "read");
    expect((await ro("/api/v1/providers/brev/budget", { method: "PUT", body: { budget_usd: 500 } })).status).toBe(403);
    expect((await api("/api/v1/providers?provider=brev")).j.providers[0].budget_usd).toBe(10);
    // Settings → Policies saves through the same document: the value there is the one used.
    expect((await api("/api/v1/policies", { method: "PUT", body: { brev_budget_usd: 42 } })).status).toBe(200);
    expect(await providerBudget(env, "brev")).toMatchObject({ usd: 42, source: "policy" });
  });
});
