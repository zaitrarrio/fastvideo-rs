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
};

const app = new Hono<{ Bindings: Env }>();

app.get("/api/v1/models", (c) => {
  return c.json({ at: Date.now(), models: CATALOG });
});

app.get("/api/v1/pricing", (c) => {
  return c.json({ at: Date.now(), placeholder: true, prices: PRICES });
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
