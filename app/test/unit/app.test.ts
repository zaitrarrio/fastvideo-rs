// fv-app worker: public catalog + pricing + key requests.
import { describe, expect, it } from "vitest";
import app from "../../src/index";

const call = (path: string, init?: RequestInit, env: any = {}) =>
  app.fetch(new Request(`https://app.test${path}`, init), env);

// Minimal D1 stand-in for key_requests.
function mockDb() {
  const rows = new Map<string, any>();
  return {
    prepare(sql: string) {
      return {
        bind(...args: any[]) {
          return {
            async run() {
              if (sql.startsWith("INSERT")) {
                rows.set(args[0], { id: args[0], name: args[1], email: args[2], use_case: args[3], status: "pending", created_at: args[4] });
              } else if (sql.startsWith("UPDATE")) {
                const r = rows.get(args[3]);
                if (r) { r.status = args[0]; r.key_hint = args[1]; r.fulfilled_at = args[2]; }
              }
              return { success: true };
            },
            async first() {
              return rows.get(args[0]) || null;
            },
          };
        },
      };
    },
  };
}

describe("fv-app", () => {
  it("GET /api/v1/models returns the catalog", async () => {
    const r = await call("/api/v1/models");
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.models.length).toBeGreaterThan(0);
    for (const m of j.models) {
      expect(m.id).toBeTruthy();
      expect(m.api_model).toBeTruthy();
    }
  });

  it("GET /api/v1/pricing returns placeholder prices", async () => {
    const r = await call("/api/v1/pricing");
    expect(r.status).toBe(200);
    const j: any = await r.json();
    expect(j.placeholder).toBe(true);
    expect(j.prices.length).toBeGreaterThan(0);
    for (const p of j.prices) {
      expect(p.tiers.length).toBeGreaterThan(0);
      for (const t of p.tiers) expect(t.usd_per_sec).toBeGreaterThan(0);
    }
  });

  it("GET /healthz is ok", async () => {
    const r = await call("/healthz");
    expect(r.status).toBe(200);
  });

  it("key requests: create and check status", async () => {
    const env = { DB: mockDb() };
    const r = await call("/api/v1/key-requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Ada", email: "ada@example.com", use_case: "ads" }),
    }, env);
    expect(r.status).toBe(201);
    const created: any = await r.json();
    expect(created.id).toMatch(/^kr_/);
    expect(created.status).toBe("pending");

    const s = await call(`/api/v1/key-requests/${created.id}`, {}, env);
    expect(s.status).toBe(200);
    const st: any = await s.json();
    expect(st.status).toBe("pending");
  });

  it("key requests: rejects bad email", async () => {
    const r = await call("/api/v1/key-requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Ada", email: "not-an-email" }),
    }, { DB: mockDb() });
    expect(r.status).toBe(400);
  });

  it("key requests: 503 without DB", async () => {
    const r = await call("/api/v1/key-requests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "Ada", email: "ada@example.com" }),
    });
    expect(r.status).toBe(503);
  });
});
