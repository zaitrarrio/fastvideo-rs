// /api/v1/key-requests (operator): list and fulfill portal key requests.
import { describe, expect, it } from "vitest";
import { mintApiToken } from "../../src/auth";
import type { Env } from "../../src/env";
import { _app } from "../../src/index";
import { d1 } from "./d1shim";

const mkEnv = () =>
  ({
    DB: d1(),
    CONTROL_KEK: "k".repeat(32),
    SESSION_SECRET: "s".repeat(40),
  }) as unknown as Env;

const call = (env: Env, path: string, init: RequestInit = {}) =>
  _app.fetch(new Request(`https://ctl.test${path}`, init), env, { waitUntil() {}, passThroughOnException() {} } as any);

// key_requests comes from control/migrations/0014_key_requests.sql (d1shim applies every migration).

describe("/api/v1/key-requests", () => {
  it("lists requests and fulfills one", async () => {
    const env = mkEnv();
    await env.DB.prepare(
      "INSERT INTO key_requests (id, name, email, use_case, status, created_at) VALUES ('kr_1', 'Ada', 'ada@x.com', 'ads', 'pending', 1)"
    ).run();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "a", "admin", "test")).token}` };

    const l = await call(env, "/api/v1/key-requests", { headers: admin });
    expect(l.status).toBe(200);
    const lj: any = await l.json();
    expect(lj.requests).toHaveLength(1);
    expect(lj.requests[0].email).toBe("ada@x.com");

    const f = await call(env, "/api/v1/key-requests/kr_1/fulfill", {
      method: "POST",
      headers: { ...admin, "Content-Type": "application/json" },
      body: JSON.stringify({ status: "fulfilled", key_hint: "abcd1234" }),
    });
    expect(f.status).toBe(200);
    const row: any = await env.DB.prepare("SELECT status, key_hint FROM key_requests WHERE id = 'kr_1'").first();
    expect(row.status).toBe("fulfilled");
    expect(row.key_hint).toBe("1234"); // only last 4 stored
  });

  it("rejects invalid status", async () => {
    const env = mkEnv();
    const admin = { authorization: `Bearer ${(await mintApiToken(env, "a", "admin", "test")).token}` };
    const r = await call(env, "/api/v1/key-requests/kr_1/fulfill", {
      method: "POST",
      headers: { ...admin, "Content-Type": "application/json" },
      body: JSON.stringify({ status: "bogus" }),
    });
    expect(r.status).toBe(400);
  });
});
