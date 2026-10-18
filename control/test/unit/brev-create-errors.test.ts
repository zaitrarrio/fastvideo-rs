// Owner report 2026-10-18: a Brev launch on gpu-h200-sxm.1gpu-16vcpu-200gb failed with Brev's HTTP 500
// `{"errors":[{"type":"InternalServerError","message":"UnrecognizedClientException: The security token included in
// the request is invalid.\n\tstatus code: 400, …"}]}`, and the operation log called it "NO STOCK" (cut at 300 raw
// JSON characters). Also, the log of that Brev-only launch printed Runpod's balance line ("balance $0.00, account
// $0.00/hr, cluster ~$6.48/hr; projected $0.00 at the deadline (floor $8)"). Now: Brev's own words, a provider
// error that fails the pool (not retried as no stock), and the Brev budget line instead of Runpod's floor.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { brev, brevErrorText, BREV_ERR_MAX, resetBrevTypes } from "../../src/brev";
import { b64, randomBytes } from "../../src/crypto";
import type { Env } from "../../src/env";
import { projectionLines, projectSpend } from "../../src/cluster/ops";
import { budgetLine, createFailure } from "../../src/providers";
import { standaloneSpec } from "../../src/standalone";
import { clearTtlCache } from "../../src/ttlcache";
import { HttpError } from "../../src/util";
import { d1 } from "./d1shim";

/** Brev's body as it answered live (2026-10-18), the request id made up. */
const LIVE_500 = JSON.stringify({
  errors: [{ type: "InternalServerError", message: "UnrecognizedClientException: The security token included in the request is invalid.\n\tstatus code: 400, request id: 0f3c9a1e-1111-2222-3333-444455556666" }],
});
const realFetch = globalThis.fetch;
let createAnswer: { status: number; body: string } = { status: 500, body: LIVE_500 };
const posts: any[] = [];
function upstream() {
  globalThis.fetch = (async (input: any, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.url;
    const u = new URL(url);
    if (u.host === "brev.test" && url.includes("/alltypesavailable/"))
      return Response.json({
        allInstanceTypes: [
          { type: "gpu-h200-sxm.1gpu-16vcpu-200gb", cloud_cred_id: "brev-nebius-prod", provider: "nebius", stoppable: true, elastic_root_volume: true, base_price: { currency: "USD", amount: "6.480000" }, location: "eu-north1", is_available: true, supported_gpus: [{ count: 1, name: "H200" }] },
          { type: "g7e.2xlarge", cloud_cred_id: "devplane-brev-1-credential", provider: "aws", stoppable: true, elastic_root_volume: true, base_price: { currency: "USD", amount: "4.035744" }, location: "us-east-1", is_available: true, supported_gpus: [{ count: 1, name: "RTX PRO Server 6000" }] },
        ],
      });
    if (u.host === "brev.test" && init?.method === "POST" && u.pathname.endsWith("/workspaces")) {
      posts.push(JSON.parse(String(init.body)));
      return new Response(createAnswer.body, { status: createAnswer.status, headers: { "content-type": "application/json" } });
    }
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
    BREV_INSTANCE_TYPES: "gpu-h200-sxm.1gpu-16vcpu-200gb,g7e.2xlarge",
    BREV_BUDGET_USD: "100",
    ...over,
  }) as unknown as Env;

beforeEach(() => {
  clearTtlCache();
  resetBrevTypes();
  posts.length = 0;
  createAnswer = { status: 500, body: LIVE_500 };
  upstream();
});
afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("Brev's error in its own words", () => {
  it("brevErrorText: type and message, whitespace collapsed, capped; plain text and other JSON shapes kept", () => {
    expect(brevErrorText(LIVE_500)).toBe("InternalServerError: UnrecognizedClientException: The security token included in the request is invalid. status code: 400, request id: 0f3c9a1e-1111-2222-3333-444455556666");
    expect(brevErrorText('{"message":"not a member"}')).toBe("not a member");
    expect(brevErrorText('{"error":"bad"}')).toBe("bad");
    expect(brevErrorText("upstream connect error")).toBe("upstream connect error");
    const long = brevErrorText(JSON.stringify({ errors: [{ type: "X", message: "y".repeat(2000) }] }));
    expect(long.length).toBe(BREV_ERR_MAX);
    expect(long.endsWith("…")).toBe(true);
  });

  it("the create's 500 carries Brev's whole message (not the raw JSON cut at 300 characters) and the org id stays out", async () => {
    const env = mkEnv();
    const e = await brev.create(env, { name: "fv-pod-x-1", instanceType: "gpu-h200-sxm.1gpu-16vcpu-200gb", startupScript: "#!/bin/bash\n", diskStorage: "300Gi", isStoppable: true }).catch((x) => x);
    expect(e).toBeInstanceOf(HttpError);
    expect(e.message).toBe("brev POST api/organizations/<org>/workspaces: 500 InternalServerError: UnrecognizedClientException: The security token included in the request is invalid. status code: 400, request id: 0f3c9a1e-1111-2222-3333-444455556666");
    expect(e.extra.upstream).toBe(500);
    // The body fv-control sent is brev-cli main's, with the type's own cloud credential (nebius here).
    expect(posts[0]).toMatchObject({ workspaceVersion: "v1", cloudCredId: "brev-nebius-prod", instanceType: "gpu-h200-sxm.1gpu-16vcpu-200gb", diskStorage: "300Gi", isStoppable: true });
  });
});

describe("createFailure: a provider error is not no stock", () => {
  it("Brev's 500 UnrecognizedClientException: provider, with the hint; not capacity", async () => {
    const env = mkEnv();
    const e = await brev.create(env, { name: "fv-pod-x-1", instanceType: "g7e.2xlarge", startupScript: "#!/bin/bash\n" }).catch((x) => x);
    const f = createFailure(e);
    expect(f.kind).toBe("provider");
    expect(f.status).toBe(500);
    expect(f.said).toMatch(/UnrecognizedClientException: The security token included in the request is invalid/);
    expect(f.hint).toMatch(/refused inside the provider's backend .*not fv-control's request, not a shortage of stock/);
  });

  it("capacity words: no stock (retried), whatever the status; other 4xx: as before; a plain error: other", () => {
    expect(createFailure(new HttpError(502, "brev POST …: 500 InternalServerError: InsufficientCapacity: no g7e.2xlarge in us-east-1a", { upstream: 500 })).kind).toBe("capacity");
    expect(createFailure(new HttpError(409, "gmi POST /containers: 409 out of stock", { upstream: 409 })).kind).toBe("capacity");
    expect(createFailure(new HttpError(400, "brev POST …: 400 BadRequestError: Legacy workspace version unsupported", { upstream: 400 })).kind).toBe("other");
    expect(createFailure(new Error("network down")).kind).toBe("other");
    // A 5xx without a recognised meaning: a provider error with no hint.
    const f = createFailure(new HttpError(502, "gmi POST /containers: 503 service unavailable", { upstream: 503 }));
    expect([f.kind, f.hint]).toEqual(["provider", null]);
  });
});

describe("the launch's money line: the Brev budget, not Runpod's floor", () => {
  const brevOnly = () => standaloneSpec({ name: "h3b", provider: "brev", provider_gpu: "gpu-h200-sxm.1gpu-16vcpu-200gb", variant: "cpu", config: "/etc/fv/runpod-fake.toml", fake_models: ["fake-wan"], weights_source: "none", deadline_min: 60, max_gpu_dph: 9 } as any).spec;

  it("a Brev-only standalone: runpod false, no balance / floor line; one Brev budget line (month, $/hr, projected at the deadline vs the budget)", async () => {
    const env = mkEnv();
    // Runpod is never asked (its balance does not apply): any Runpod call would answer 599 here.
    const p = await projectSpend(env, brevOnly(), { hours: 1 });
    expect(p.runpod).toBe(false);
    expect(p.ok).toBe(true);
    expect(p.budgets).toHaveLength(1);
    expect(p.budgets[0]).toMatchObject({ provider: "brev", budget: 100, month: 0, add_dph: 6.48, hours: 1, ok: true });
    const lines = projectionLines(p);
    expect(lines).toEqual(["NVIDIA Brev: $0.00 spent this month + ~$6.48/hr (this launch $6.48/hr) for 1.00 h → projected $6.48 at the deadline; budget $100 (within)"]);
    expect(lines.join("\n")).not.toMatch(/floor|balance \$/);
  });

  it("over the budget: the line says OVER and the reason refuses it", async () => {
    const env = mkEnv({ BREV_BUDGET_USD: "5" });
    const p = await projectSpend(env, brevOnly(), { hours: 1 });
    expect(p.ok).toBe(false);
    expect(projectionLines(p)[0]).toMatch(/projected \$6\.48 at the deadline; budget \$5 \(OVER\)$/);
    expect(p.reasons[0]).toMatch(/over the monthly budget \$5/);
  });

  it("budgetLine: unset budget, running and parked $/hr shown", () => {
    expect(budgetLine({ provider: "brev", ok: false, reasons: [], budget: null, month: 1.5, running_dph: 2, parked_dph: 0.05, add_dph: 4, hours: 2, projected: 13.6 })).toBe(
      "NVIDIA Brev: $1.50 spent this month + ~$6.05/hr (this launch $4.00/hr, running $2.00/hr, parked storage $0.050/hr) for 2.00 h → projected $13.60 at the deadline; budget unset",
    );
  });
});
