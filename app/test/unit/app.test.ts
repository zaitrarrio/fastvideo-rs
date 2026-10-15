// fv-app worker: public catalog + pricing.
import { describe, expect, it } from "vitest";
import app from "../../src/index";

const call = (path: string) => app.fetch(new Request(`https://app.test${path}`), {} as any);

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
});
