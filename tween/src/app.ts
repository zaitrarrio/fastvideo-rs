// Tween HTTP API (Hono). See tween/README.md for the route table.
import { Hono, type Context } from "hono";
import { pickAnimatic, type AnimaticSource } from "./animatic";
import { COOKIE, clearedCookie, isAuthorised, sessionCookie, sessionValue, tokenMatches } from "./auth";
import { EngineClient } from "./engine-client";
import { engineConfigs, type EngineVars } from "./engines";
import { panelKey, serveObject } from "./media";
import { realtimeFactors } from "./metrics";
import { RenderError, refreshRender, submitShot, type RenderDeps } from "./renders";
import { deriveShots, shotAt } from "./shots";
import { Store, StoreError, newId } from "./store";
import { ASPECT_RATIOS, ENGINE_IDS, type Panel, type Render } from "./types";
import { LIMITS, parseBoardInput, parseBoardPatch, parseCaption, parseImageUpload, parseRenderRequest, parseShotInput } from "./validate";

export interface Env extends EngineVars {
  DB: D1Database;
  MEDIA: R2Bucket;
  ASSETS?: Fetcher;
  /** fv-serve base URL (the edge gateway, or a worker in direct mode). */
  TWEEN_ENGINE_URL?: string;
  /** fv-serve API key (secret). */
  TWEEN_ENGINE_KEY?: string;
  /** Shared access token (secret). Unset: open, for local development only. */
  TWEEN_ACCESS_TOKEN?: string;
}

export interface AppOptions {
  /** fetch used for engine calls (tests inject a fake engine). */
  fetch?: typeof fetch;
  now?: () => number;
  randomSeed?: () => number;
}

type C = Context<{ Bindings: Env }>;

const fail = (c: C, status: number, error: string) => c.json({ error }, status as 400);

export function renderView(r: Render) {
  return {
    ...r,
    factors: realtimeFactors(r),
    video_url: r.video_key ? `/api/renders/${r.id}/video` : null,
  };
}

const panelView = (p: Panel) => ({ ...p, image_url: `/api/boards/${p.board_id}/panels/${p.id}/image` });

async function readJson(c: C): Promise<unknown> {
  try {
    return await c.req.json();
  } catch {
    return undefined;
  }
}

export function createApp(opts: AppOptions = {}) {
  const now = opts.now ?? Date.now;
  const app = new Hono<{ Bindings: Env }>();

  const store = (c: C) => new Store(c.env.DB, now);
  const deps = (c: C): RenderDeps | null => {
    if (!c.env.TWEEN_ENGINE_URL) return null;
    return {
      store: store(c),
      media: c.env.MEDIA,
      client: new EngineClient(c.env.TWEEN_ENGINE_URL, c.env.TWEEN_ENGINE_KEY ?? "", opts.fetch ?? fetch),
      engines: engineConfigs(c.env),
      now,
      randomSeed: opts.randomSeed,
    };
  };

  app.onError((err, c) => {
    if (err instanceof StoreError || err instanceof RenderError) return fail(c, err.status, err.message);
    console.error("tween: unhandled", err);
    return fail(c, 500, "internal error");
  });

  app.get("/healthz", (c) => c.text("ok"));

  // ---- session --------------------------------------------------------------

  app.get("/api/session", async (c) =>
    c.json({ required: Boolean(c.env.TWEEN_ACCESS_TOKEN), authenticated: await isAuthorised(c.env.TWEEN_ACCESS_TOKEN, c.req.raw) }),
  );

  app.post("/api/session", async (c) => {
    const token = c.env.TWEEN_ACCESS_TOKEN;
    if (!token) return c.json({ required: false, authenticated: true });
    const body = (await readJson(c)) as { token?: unknown } | undefined;
    if (!(await tokenMatches(token, body?.token))) return fail(c, 401, "that token is not valid");
    const secure = new URL(c.req.url).protocol === "https:";
    c.header("set-cookie", sessionCookie(await sessionValue(token), secure));
    return c.json({ required: true, authenticated: true });
  });

  app.delete("/api/session", (c) => {
    c.header("set-cookie", clearedCookie(new URL(c.req.url).protocol === "https:"));
    return c.json({ authenticated: false });
  });

  // Everything else under /api needs a session when a token is configured.
  app.use("/api/*", async (c, next) => {
    if (await isAuthorised(c.env.TWEEN_ACCESS_TOKEN, c.req.raw)) return next();
    return c.json({ error: `sign in first (cookie ${COOKIE})` }, 401);
  });

  // ---- config ---------------------------------------------------------------

  app.get("/api/config", (c) =>
    c.json({
      engine_configured: Boolean(c.env.TWEEN_ENGINE_URL),
      engines: engineConfigs(c.env),
      engine_ids: ENGINE_IDS,
      aspect_ratios: ASPECT_RATIOS,
      limits: LIMITS,
    }),
  );

  // ---- boards ---------------------------------------------------------------

  app.get("/api/boards", async (c) => c.json({ boards: await store(c).listBoards() }));

  app.post("/api/boards", async (c) => {
    const p = parseBoardInput(await readJson(c));
    if (!p.ok) return fail(c, 400, p.error);
    return c.json({ board: await store(c).createBoard(p.value) }, 201);
  });

  app.get("/api/boards/:board", async (c) => {
    const s = store(c);
    const board = await s.getBoard(c.req.param("board"));
    if (!board) return fail(c, 404, "board not found");
    const [panels, settings, renders] = await Promise.all([s.listPanels(board.id), s.listShots(board.id), s.listRenders(board.id)]);
    return c.json({
      board,
      panels: panels.map(panelView),
      shots: deriveShots(panels, settings),
      renders: renders.map(renderView),
    });
  });

  app.patch("/api/boards/:board", async (c) => {
    const p = parseBoardPatch(await readJson(c));
    if (!p.ok) return fail(c, 400, p.error);
    const board = await store(c).updateBoard(c.req.param("board"), p.value);
    return board ? c.json({ board }) : fail(c, 404, "board not found");
  });

  // ---- panels ---------------------------------------------------------------

  app.post("/api/boards/:board/panels", async (c) => {
    const s = store(c);
    const board = await s.getBoard(c.req.param("board"));
    if (!board) return fail(c, 404, "board not found");
    const declared = Number(c.req.header("content-length") ?? "0");
    if (declared > LIMITS.image_max_bytes + 64 * 1024) return fail(c, 413, "the upload is too large");

    let bytes: Uint8Array;
    let caption = "";
    const type = c.req.header("content-type") ?? "";
    if (type.startsWith("multipart/form-data")) {
      const form = await c.req.formData();
      const file = form.get("image");
      if (!file || typeof file === "string") return fail(c, 400, "send the image in a form field named image");
      bytes = new Uint8Array(await file.arrayBuffer());
      const cap = form.get("caption");
      caption = typeof cap === "string" ? cap.trim().slice(0, LIMITS.caption_max) : "";
    } else {
      bytes = new Uint8Array(await c.req.arrayBuffer());
    }
    const img = parseImageUpload(bytes);
    if (!img.ok) return fail(c, 400, img.error);

    const key = panelKey(board.id, newId("im"));
    await c.env.MEDIA.put(key, bytes, { httpMetadata: { contentType: img.value.mime } });
    const panel = await s.addPanel(board.id, { image_key: key, mime: img.value.mime, bytes: img.value.bytes, caption });
    return c.json({ panel: panelView(panel) }, 201);
  });

  app.get("/api/boards/:board/panels/:panel/image", async (c) => {
    const p = await store(c).getPanel(c.req.param("board"), c.req.param("panel"));
    if (!p) return fail(c, 404, "panel not found");
    return serveObject(c.env.MEDIA, p.image_key, c.req.raw, p.mime);
  });

  app.patch("/api/boards/:board/panels/:panel", async (c) => {
    const cap = parseCaption(await readJson(c));
    if (!cap.ok) return fail(c, 400, cap.error);
    const p = await store(c).updatePanel(c.req.param("board"), c.req.param("panel"), { caption: cap.value });
    return p ? c.json({ panel: panelView(p) }) : fail(c, 404, "panel not found");
  });

  app.delete("/api/boards/:board/panels/:panel", async (c) => {
    const p = await store(c).deletePanel(c.req.param("board"), c.req.param("panel"));
    if (!p) return fail(c, 404, "panel not found");
    // The image stays in R2: renders of shots that used it remain history.
    return c.json({ deleted: p.id });
  });

  app.put("/api/boards/:board/panel-order", async (c) => {
    const s = store(c);
    const board = await s.getBoard(c.req.param("board"));
    if (!board) return fail(c, 404, "board not found");
    const body = (await readJson(c)) as { panel_ids?: unknown } | undefined;
    const ids = body?.panel_ids;
    if (!Array.isArray(ids) || !ids.every((x) => typeof x === "string")) return fail(c, 400, "panel_ids must be a list of panel ids");
    const panels = await s.reorderPanels(board.id, ids);
    return c.json({ panels: panels.map(panelView), shots: deriveShots(panels, await s.listShots(board.id)) });
  });

  // ---- shots ----------------------------------------------------------------

  app.put("/api/boards/:board/shots/:start", async (c) => {
    const s = store(c);
    const boardId = c.req.param("board");
    const start = c.req.param("start");
    const p = parseShotInput(await readJson(c));
    if (!p.ok) return fail(c, 400, p.error);
    const panels = await s.listPanels(boardId);
    if (!shotAt(panels, [], start)) return fail(c, 404, "no shot starts at that panel (it is missing or the last panel)");
    await s.upsertShot(boardId, start, p.value);
    return c.json({ shot: shotAt(panels, await s.listShots(boardId), start) });
  });

  app.post("/api/boards/:board/shots/:start/renders", async (c) => {
    const d = deps(c);
    if (!d) return fail(c, 503, "no engine is configured (set TWEEN_ENGINE_URL)");
    const req = parseRenderRequest(await readJson(c));
    if (!req.ok) return fail(c, 400, req.error);
    const board = await d.store.getBoard(c.req.param("board"));
    if (!board) return fail(c, 404, "board not found");
    const panels = await d.store.listPanels(board.id);
    const shot = shotAt(panels, await d.store.listShots(board.id), c.req.param("start"));
    if (!shot) return fail(c, 404, "no shot starts at that panel (it is missing or the last panel)");
    const out = await submitShot(d, { board, shot, panels, ...req.value });
    return c.json({ bakeoff_id: out.bakeoff_id, renders: out.renders.map(renderView) }, 202);
  });

  // ---- renders --------------------------------------------------------------

  app.get("/api/renders/:render", async (c) => {
    const s = store(c);
    const r = await s.getRender(c.req.param("render"));
    if (!r) return fail(c, 404, "render not found");
    const d = deps(c);
    return c.json({ render: renderView(d ? await refreshRender(d, r) : r) });
  });

  app.get("/api/renders/:render/video", async (c) => {
    const r = await store(c).getRender(c.req.param("render"));
    if (!r || !r.video_key) return fail(c, 404, "this render has no video yet");
    return serveObject(c.env.MEDIA, r.video_key, c.req.raw, "video/mp4");
  });

  app.post("/api/renders/:render/select", async (c) => {
    const s = store(c);
    const r = await s.getRender(c.req.param("render"));
    if (!r) return fail(c, 404, "render not found");
    if (r.status !== "completed") return fail(c, 409, "only a finished render can be selected");
    return c.json({ render: renderView((await s.selectRender(r.id))!) });
  });

  /** Re-renders at the final tier with the draft's seed, prompt and length. */
  app.post("/api/renders/:render/finalize", async (c) => {
    const d = deps(c);
    if (!d) return fail(c, 503, "no engine is configured (set TWEEN_ENGINE_URL)");
    const r = await d.store.getRender(c.req.param("render"));
    if (!r) return fail(c, 404, "render not found");
    const board = await d.store.getBoard(r.board_id);
    if (!board) return fail(c, 404, "board not found");
    const panels = await d.store.listPanels(board.id);
    const current = shotAt(panels, await d.store.listShots(board.id), r.start_panel_id);
    if (!current || current.end_panel_id !== r.end_panel_id) {
      return fail(c, 409, "the panels of this shot have changed since the draft; render the shot again");
    }
    const shot = { ...current, prompt: r.prompt, seconds: r.seconds, mode: r.mode };
    const out = await submitShot(d, { board, shot, panels, engines: [r.engine], tier: "final", seed: r.seed });
    return c.json({ renders: out.renders.map(renderView) }, 202);
  });

  // ---- animatic -------------------------------------------------------------

  app.get("/api/boards/:board/animatic", async (c) => {
    const source = (c.req.query("source") ?? "cut") as AnimaticSource;
    if (source !== "cut" && !(ENGINE_IDS as readonly string[]).includes(source)) {
      return fail(c, 400, `source must be cut or one of ${ENGINE_IDS.join(", ")}`);
    }
    const s = store(c);
    const board = await s.getBoard(c.req.param("board"));
    if (!board) return fail(c, 404, "board not found");
    const [panels, settings, renders] = await Promise.all([s.listPanels(board.id), s.listShots(board.id), s.listRenders(board.id)]);
    const entries = pickAnimatic(deriveShots(panels, settings), renders, source);
    const playable = entries.filter((e) => e.render);
    return c.json({
      source,
      complete: playable.length === entries.length && entries.length > 0,
      total_seconds: playable.reduce((t, e) => t + (e.render!.clip_seconds ?? e.render!.seconds), 0),
      entries: entries.map((e) => ({ shot: e.shot, render: e.render ? renderView(e.render) : null })),
    });
  });

  app.all("/api/*", (c) => fail(c, 404, "no such route"));

  // ---- static UI ------------------------------------------------------------

  app.get("*", async (c) => {
    if (!c.env.ASSETS) return c.text("not found", 404);
    const r = await c.env.ASSETS.fetch(c.req.raw);
    if (r.status !== 404) return r;
    // "/" rather than "/index.html": the assets layer redirects the latter
    // to "/", which would drop the board id from a deep link.
    const url = new URL(c.req.url);
    url.pathname = "/";
    return c.env.ASSETS.fetch(new Request(url, c.req.raw));
  });

  return app;
}
