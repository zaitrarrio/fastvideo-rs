// Route tests: real SQL (node:sqlite D1), in-memory R2, fake fv-serve.
import { beforeEach, describe, expect, it } from "vitest";
import { createApp, type Env } from "../../src/app";
import { testD1 } from "../support/d1";
import { DEFAULT_MODELS, MP4, fakeEngine } from "../support/fake-engine";
import { memoryR2 } from "../support/r2";

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
const pngWith = (tag: number) => new Uint8Array([...PNG, tag]);

let engine: ReturnType<typeof fakeEngine>;
let env: Env;
let app: ReturnType<typeof createApp>;

function setup(over: Partial<Env> = {}, models = DEFAULT_MODELS) {
  engine = fakeEngine("sk-test", models);
  env = {
    DB: testD1(),
    MEDIA: memoryR2().bucket,
    TWEEN_ENGINE_URL: "http://engine.test",
    TWEEN_ENGINE_KEY: "sk-test",
    ...over,
  };
  let t = 10_000;
  app = createApp({ fetch: engine.fetch, now: () => (t += 1000), randomSeed: () => 4242 });
}

const call = (path: string, init: RequestInit = {}) => app.fetch(new Request(`https://tween.test${path}`, init), env);
const json = async (path: string, method = "GET", body?: unknown, headers: Record<string, string> = {}) => {
  const r = await call(path, {
    method,
    headers: { "content-type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return { status: r.status, body: (await r.json()) as any, headers: r.headers };
};

async function upload(boardId: string, bytes: Uint8Array<ArrayBuffer>, caption = "") {
  const form = new FormData();
  form.set("image", new File([bytes], "panel.png", { type: "image/png" }));
  form.set("caption", caption);
  const r = await call(`/api/boards/${boardId}/panels`, { method: "POST", body: form });
  return { status: r.status, body: (await r.json()) as any };
}

async function board(panelCount: number, aspect = "16:9") {
  const b = (await json("/api/boards", "POST", { title: "Heist", aspect_ratio: aspect })).body.board;
  const panels = [];
  for (let i = 0; i < panelCount; i++) panels.push((await upload(b.id, pngWith(i), `p${i}`)).body.panel);
  // fv-serve refuses an empty prompt, so every shot gets one unless a test says otherwise.
  for (let i = 0; i + 1 < panels.length; i++) {
    await json(`/api/boards/${b.id}/shots/${panels[i].id}`, "PUT", { prompt: `motion ${i}`, seconds: 6 });
  }
  return { b, panels };
}

async function pollDone(renderId: string) {
  for (let i = 0; i < 10; i++) {
    const r = (await json(`/api/renders/${renderId}`)).body.render;
    if (r.status === "completed" || r.status === "failed") return r;
  }
  throw new Error("render never finished");
}

beforeEach(() => setup());

describe("config and health", () => {
  it("answers healthz and publishes engines without secrets", async () => {
    expect((await call("/healthz")).status).toBe(200);
    const { body } = await json("/api/config");
    expect(body.engine_configured).toBe(true);
    expect(body.engines.map((e: any) => e.id)).toEqual(["ltx", "h3"]);
    expect(JSON.stringify(body)).not.toContain("sk-test");
  });
});

describe("boards and panels", () => {
  it("creates a board, uploads panels and derives the shots between them", async () => {
    const { b, panels } = await board(3);
    const { body } = await json(`/api/boards/${b.id}`);
    expect(body.panels.map((p: any) => p.caption)).toEqual(["p0", "p1", "p2"]);
    expect(body.shots.map((s: any) => [s.start_panel_id, s.end_panel_id])).toEqual([
      [panels[0].id, panels[1].id],
      [panels[1].id, panels[2].id],
    ]);
    const img = await call(body.panels[0].image_url);
    expect(img.headers.get("content-type")).toBe("image/png");
    expect(new Uint8Array(await img.arrayBuffer())).toEqual(pngWith(0));
  });

  it("accepts a raw image body and refuses non-images", async () => {
    const { b } = await board(0);
    const raw = await call(`/api/boards/${b.id}/panels`, { method: "POST", body: PNG, headers: { "content-type": "image/png" } });
    expect(raw.status).toBe(201);
    const svg = await call(`/api/boards/${b.id}/panels`, { method: "POST", body: "<svg/>", headers: { "content-type": "image/svg+xml" } });
    expect(svg.status).toBe(400);
  });

  it("reorders panels and re-derives shots", async () => {
    const { b, panels } = await board(3);
    const ids = [panels[2].id, panels[0].id, panels[1].id];
    const { status, body } = await json(`/api/boards/${b.id}/panel-order`, "PUT", { panel_ids: ids });
    expect(status).toBe(200);
    expect(body.shots.map((s: any) => s.start_panel_id)).toEqual([panels[2].id, panels[0].id]);
    expect((await json(`/api/boards/${b.id}/panel-order`, "PUT", { panel_ids: ids.slice(1) })).status).toBe(400);
  });

  it("404s on unknown boards and refuses bad input", async () => {
    expect((await json("/api/boards/bd_nope")).status).toBe(404);
    expect((await json("/api/boards", "POST", { title: "" })).status).toBe(400);
  });
});

describe("shots", () => {
  it("stores settings for a shot and refuses the last panel", async () => {
    const { b, panels } = await board(2);
    const put = await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "the vault door swings", seconds: 6, seed: 7 });
    expect(put.status).toBe(200);
    expect(put.body.shot).toMatchObject({ prompt: "the vault door swings", seconds: 6, seed: 7, end_panel_id: panels[1].id });
    expect((await json(`/api/boards/${b.id}/shots/${panels[1].id}`, "PUT", { prompt: "x" })).status).toBe(404);
    expect((await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "motion", seconds: 2.5 })).status).toBe(400);
  });
});

describe("bake-off", () => {
  it("submits the same keyframe shot to LTX and H3 with one seed and both panels", async () => {
    const { b, panels } = await board(2);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "a slow push in", seconds: 6 });
    const { status, body } = await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {});
    expect(status).toBe(202);
    expect(body.bakeoff_id).toMatch(/^bk_/);
    expect(body.renders.map((r: any) => [r.engine, r.model, r.status])).toEqual([
      ["ltx", "fake-ltx-turbo", "queued"],
      ["h3", "fake-h3-turbo", "queued"],
    ]);
    expect(engine.submitted).toHaveLength(2);
    for (const s of engine.submitted) {
      expect(s).toMatchObject({ prompt: "a slow push in", seconds: 6, seed: 4242, aspect_ratio: "16:9" });
      const refs = s.image_reference as { image_url: string }[];
      expect(refs).toHaveLength(2);
      expect(refs[0]!.image_url).toMatch(/^data:image\/png;base64,/);
      expect(refs[0]!.image_url).not.toBe(refs[1]!.image_url);
      expect(s).not.toHaveProperty("task");
    }
  });

  it("polls to completion, copies the MP4 into R2 and reports the real-time factor", async () => {
    const { b, panels } = await board(2);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "motion", seconds: 6 });
    const started = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["ltx"] })).body.renders[0];
    expect(started.bakeoff_id).toBeNull();
    const done = await pollDone(started.id);
    expect(done).toMatchObject({ status: "completed", progress: 100, clip_seconds: 6, inference_time_s: 1.5, size: "1280x768" });
    expect(done.factors).toMatchObject({ inference: 4, faster_than_realtime: true });
    expect(done.wall_time_s).toBe(3);
    const video = await call(done.video_url);
    expect(video.headers.get("content-type")).toBe("video/mp4");
    expect(new Uint8Array(await video.arrayBuffer())).toEqual(MP4);
    const ranged = await call(done.video_url, { headers: { range: "bytes=4-7" } });
    expect(ranged.status).toBe(206);
    expect(ranged.headers.get("content-range")).toBe(`bytes 4-7/${MP4.length}`);
  });

  it("records an engine refusal on that engine only and keeps the other render running", async () => {
    const { b, panels } = await board(2);
    // H3 refuses 20 s; LTX takes it.
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "motion", seconds: 20 });
    const { body } = await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {});
    const byEngine = Object.fromEntries(body.renders.map((r: any) => [r.engine, r]));
    expect(byEngine.ltx.status).toBe("queued");
    expect(byEngine.h3.status).toBe("failed");
    expect(byEngine.h3.error).toContain("5..15");
  });

  it("refuses a reference bake-off up front because LTX cannot take one", async () => {
    const { b, panels } = await board(2);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "motion", mode: "reference", seconds: 6 });
    const r = await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {});
    expect(r.status).toBe(400);
    expect(r.body.error).toMatch(/H3/);
    expect(engine.submitted).toHaveLength(0);
  });

  it("sends ref2va with the reference panels for a reference shot on H3", async () => {
    const { b, panels } = await board(3);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", {
      prompt: "motion",
      mode: "reference",
      seconds: 5,
      reference_panel_ids: [panels[2].id],
    });
    const r = await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["h3"] });
    expect(r.status).toBe(202);
    expect(r.body.renders[0].model).toBe("fake-h3-max");
    expect(engine.submitted[0]).toMatchObject({ task: "ref2va" });
    expect(engine.submitted[0]!.image_reference).toHaveLength(3);
  });

  it("marks a render failed when the engine fails the job or forgets it", async () => {
    setup({}, { ...DEFAULT_MODELS, "fake-ltx-turbo": { family: "ltx2", inference_s: 1, fail: "CUDA out of memory" } });
    const { b, panels } = await board(2);
    const [ltx, h3] = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {})).body.renders;
    expect((await pollDone(ltx.id)).error).toBe("CUDA out of memory");
    expect(h3.job_id).toBeTruthy();
    engine.forget(h3.job_id);
    const lost = await pollDone(h3.id);
    expect(lost.status).toBe("failed");
    expect(lost.error).toMatch(/no longer has this job/);
  });

  it("treats a failed MP4 download as transient and finishes on the next poll", async () => {
    const { b, panels } = await board(2);
    const r = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["ltx"] })).body.renders[0];
    // The fake engine advances one state per GET: after these two polls the
    // next GET reports `completed`, which triggers the MP4 download.
    await json(`/api/renders/${r.id}`);
    expect((await json(`/api/renders/${r.id}`)).body.render.status).toBe("in_progress");
    const realFetch = engine.fetch;
    let contentCalls = 0;
    let broken = true;
    const flaky = (async (input: RequestInfo | URL, init?: RequestInit) =>
      String(input).endsWith("/content") && broken && ++contentCalls ? new Response("upstream reset", { status: 502 }) : realFetch(input, init)) as typeof fetch;
    let t = 50_000;
    app = createApp({ fetch: flaky, now: () => (t += 1000), randomSeed: () => 1 });
    const first = await json(`/api/renders/${r.id}`);
    expect(first.status).toBe(200);
    expect(contentCalls).toBe(1);
    expect(first.body.render.status).toBe("in_progress");
    expect(first.body.render.video_url).toBeNull();
    broken = false;
    const second = (await json(`/api/renders/${r.id}`)).body.render;
    expect(second.status).toBe("completed");
    expect(second.video_url).toBeTruthy();
  });

  it("refuses to render a shot with no prompt, before anything reaches the engine", async () => {
    const { b, panels } = await board(2);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "   ", seconds: 6 });
    const r = await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {});
    expect(r.status).toBe(400);
    expect(r.body.error).toMatch(/prompt/);
    expect(engine.submitted).toHaveLength(0);
  });

  it("503s without an engine and reads renders without touching one", async () => {
    setup({ TWEEN_ENGINE_URL: undefined });
    const { b, panels } = await board(2);
    expect((await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {})).status).toBe(503);
  });
});

describe("finalize, select and the animatic", () => {
  it("finalizes a draft with the same seed on the final model", async () => {
    const { b, panels } = await board(2);
    await json(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "motion", seconds: 6, seed: 99 });
    const draft = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["h3"] })).body.renders[0];
    const fin = await json(`/api/renders/${draft.id}/finalize`, "POST");
    expect(fin.status).toBe(202);
    expect(fin.body.renders[0]).toMatchObject({ tier: "final", model: "fake-h3-max", seed: 99, engine: "h3" });
    expect(engine.submitted[1]).toMatchObject({ model: "fake-h3-max", seed: 99 });
  });

  it("refuses to finalize after the shot's end panel changed", async () => {
    const { b, panels } = await board(3);
    const draft = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["ltx"] })).body.renders[0];
    await json(`/api/boards/${b.id}/panel-order`, "PUT", { panel_ids: [panels[0].id, panels[2].id, panels[1].id] });
    expect((await json(`/api/renders/${draft.id}/finalize`, "POST")).status).toBe(409);
  });

  it("builds the cut from selected renders and per-engine reels from each engine", async () => {
    const { b, panels } = await board(3);
    const s0 = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {})).body.renders;
    for (const r of s0) await pollDone(r.id);
    const h3First = s0.find((r: any) => r.engine === "h3");
    expect((await json(`/api/renders/${h3First.id}/select`, "POST")).status).toBe(200);

    let cut = (await json(`/api/boards/${b.id}/animatic`)).body;
    expect(cut.complete).toBe(false);
    expect(cut.entries[0].render.id).toBe(h3First.id);
    expect(cut.entries[1].render).toBeNull();

    const s1 = (await json(`/api/boards/${b.id}/shots/${panels[1].id}/renders`, "POST", { engines: ["ltx"] })).body.renders[0];
    await pollDone(s1.id);
    cut = (await json(`/api/boards/${b.id}/animatic`)).body;
    expect(cut.complete).toBe(true);
    expect(cut.total_seconds).toBe(12);

    const ltxReel = (await json(`/api/boards/${b.id}/animatic?source=ltx`)).body;
    expect(ltxReel.entries.map((e: any) => e.render?.engine)).toEqual(["ltx", "ltx"]);
    expect((await json(`/api/boards/${b.id}/animatic?source=wan`)).status).toBe(400);
  });

  it("only selects finished renders", async () => {
    const { b, panels } = await board(2);
    const r = (await json(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["ltx"] })).body.renders[0];
    expect((await json(`/api/renders/${r.id}/select`, "POST")).status).toBe(409);
  });
});

describe("access token", () => {
  beforeEach(() => setup({ TWEEN_ACCESS_TOKEN: "open-sesame" }));

  it("gates the API until a session cookie is set, and keeps health and session open", async () => {
    expect((await call("/healthz")).status).toBe(200);
    expect((await json("/api/session")).body).toEqual({ required: true, authenticated: false });
    expect((await json("/api/boards")).status).toBe(401);
    expect((await json("/api/session", "POST", { token: "wrong" })).status).toBe(401);

    const ok = await json("/api/session", "POST", { token: "open-sesame" });
    expect(ok.status).toBe(200);
    const cookie = ok.headers.get("set-cookie")!;
    expect(cookie).toMatch(/HttpOnly/);
    expect(cookie).toMatch(/Secure/);
    expect(cookie).not.toContain("open-sesame");
    const session = cookie.split(";")[0]!;
    expect((await json("/api/boards", "GET", undefined, { cookie: session })).status).toBe(200);
  });
});
