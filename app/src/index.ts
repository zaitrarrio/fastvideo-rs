// fv-app: the public consumer portal worker.
// Serves the static gallery/playground and the public JSON API:
//   GET /api/v1/models  — the curated model catalog
//   GET /api/v1/pricing — per-model per-second prices (PLACEHOLDER values)
// No auth, no privileged access: the browser talks to engine public APIs
// directly with the consumer's own key.
import { Hono } from "hono";
import { CATALOG } from "./catalog";
import { PRICES } from "./pricing";

type Env = {
  ASSETS?: { fetch: (req: Request) => Promise<Response> };
  DB?: D1Database;
};

const app = new Hono<{ Bindings: Env }>();

app.get("/api/v1/models", (c) => {
  return c.json({ at: Date.now(), models: CATALOG });
});

app.get("/api/v1/pricing", (c) => {
  return c.json({ at: Date.now(), placeholder: true, prices: PRICES });
});

// Key provisioning: request a key, check its status. The key itself is
// minted operator-side (engine /fv/v1/admin/keys) and delivered out of
// band; the portal only tracks the request. key_hint is the last 4 chars,
// never the key.
function reqId() {
  const b = new Uint8Array(12);
  crypto.getRandomValues(b);
  return "kr_" + [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
}
app.post("/api/v1/key-requests", async (c) => {
  if (!c.env.DB) return c.json({ error: "unavailable" }, 503);
  const body = await c.req.json().catch(() => ({}));
  const name = String(body.name || "").trim().slice(0, 120);
  const email = String(body.email || "").trim().slice(0, 200);
  const use_case = String(body.use_case || "").trim().slice(0, 2000);
  if (!name || !email || !email.includes("@")) {
    return c.json({ error: "name and a valid email are required" }, 400);
  }
  const id = reqId();
  await c.env.DB.prepare(
    "INSERT INTO key_requests (id, name, email, use_case, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)"
  )
    .bind(id, name, email, use_case, Date.now())
    .run();
  return c.json({ id, status: "pending" }, 201);
});
app.get("/api/v1/key-requests/:id", async (c) => {
  if (!c.env.DB) return c.json({ error: "unavailable" }, 503);
  const row = await c.env.DB.prepare(
    "SELECT id, status, key_hint, created_at, fulfilled_at FROM key_requests WHERE id = ?"
  )
    .bind(c.req.param("id"))
    .first<any>();
  if (!row) return c.json({ error: "not found" }, 404);
  return c.json(row);
});

app.get("/healthz", (c) => c.text("ok"));

// Static assets; SPA fallback to index.html.
app.get("*", async (c) => {
  if (c.env.ASSETS) {
    const r = await c.env.ASSETS.fetch(c.req.raw);
    if (r.status !== 404) return r;
    const url = new URL(c.req.url);
    url.pathname = "/index.html";
    return c.env.ASSETS.fetch(new Request(url, c.req.raw));
  }
  return c.text("not found", 404);
});

export default app;
