// Tween against a live fv-serve (task tween:e2e). Skipped unless
// TWEEN_E2E_ENGINE_URL is set. D1 and R2 are the test stand-ins; the engine
// is real, so this checks the wire contract (ADR-0004) rather than a fake of it:
// keyframe requests on both families, H3 ref2va, the job view, /content.
//
//   TWEEN_E2E_ENGINE_URL=http://127.0.0.1:8000 TWEEN_E2E_ENGINE_KEY=sk-local npm run test:e2e
// Model ids default to the fake engine's; override with TWEEN_* model vars.
import { describe, expect, it } from "vitest";
import { createApp, type Env } from "../../src/app";
import { EngineClient } from "../../src/engine-client";
import { testD1 } from "../support/d1";
import { memoryR2 } from "../support/r2";

const URL_ = process.env.TWEEN_E2E_ENGINE_URL;
const KEY = process.env.TWEEN_E2E_ENGINE_KEY ?? "sk-local";
const TIMEOUT_MS = Number(process.env.TWEEN_E2E_TIMEOUT_MS ?? 120_000);

// A real 64x64 PNG (solid colour), so engines that decode images accept it.
function png(r: number, g: number, b: number): Uint8Array<ArrayBuffer> {
  const w = 64;
  const h = 64;
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf: Uint8Array) => {
    let c = 0xffffffff;
    for (const x of buf) c = crcTable[(c ^ x) & 0xff]! ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const u32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
  const chunk = (type: string, data: number[]) => {
    const td = new Uint8Array([...type].map((ch) => ch.charCodeAt(0)).concat(data));
    return [...u32(data.length), ...td, ...u32(crc(td))];
  };
  // Stored (uncompressed) deflate blocks over the raw scanlines.
  const raw: number[] = [];
  for (let y = 0; y < h; y++) {
    raw.push(0);
    for (let x = 0; x < w; x++) raw.push(r, g, b);
  }
  const z: number[] = [0x78, 0x01];
  for (let i = 0; i < raw.length; i += 65535) {
    const block = raw.slice(i, i + 65535);
    const last = i + 65535 >= raw.length ? 1 : 0;
    z.push(last, block.length & 255, block.length >>> 8, ~block.length & 255, (~block.length >>> 8) & 255, ...block);
  }
  let a = 1;
  let b2 = 0;
  for (const x of raw) {
    a = (a + x) % 65521;
    b2 = (b2 + a) % 65521;
  }
  z.push(...u32(((b2 << 16) | a) >>> 0));
  return new Uint8Array([
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ...chunk("IHDR", [...u32(w), ...u32(h), 8, 2, 0, 0, 0]),
    ...chunk("IDAT", z),
    ...chunk("IEND", []),
  ]);
}

describe.skipIf(!URL_)("Tween against a live fv-serve", () => {
  const env: Env = {
    DB: testD1(),
    MEDIA: memoryR2().bucket,
    TWEEN_ENGINE_URL: URL_,
    TWEEN_ENGINE_KEY: KEY,
    TWEEN_LTX_DRAFT_MODEL: process.env.TWEEN_LTX_DRAFT_MODEL,
    TWEEN_LTX_FINAL_MODEL: process.env.TWEEN_LTX_FINAL_MODEL,
    TWEEN_H3_DRAFT_MODEL: process.env.TWEEN_H3_DRAFT_MODEL,
    TWEEN_H3_FINAL_MODEL: process.env.TWEEN_H3_FINAL_MODEL,
    TWEEN_H3_REFERENCE_MODEL: process.env.TWEEN_H3_REFERENCE_MODEL,
  };
  const app = createApp();
  const call = async (path: string, method = "GET", body?: unknown) => {
    const r = await app.fetch(
      new Request(`http://tween.test${path}`, {
        method,
        headers: body instanceof FormData ? {} : { "content-type": "application/json" },
        body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
      }),
      env,
    );
    return { status: r.status, body: (await r.json()) as any };
  };

  async function boardWithShot(panelCount = 2) {
    const b = (await call("/api/boards", "POST", { title: "e2e", aspect_ratio: "16:9" })).body.board;
    const colours: [number, number, number][] = [
      [232, 163, 61],
      [194, 65, 45],
      [40, 30, 25],
    ];
    const panels = [];
    for (let i = 0; i < panelCount; i++) {
      const form = new FormData();
      form.set("image", new File([png(...colours[i % 3]!)], `p${i}.png`, { type: "image/png" }));
      panels.push((await call(`/api/boards/${b.id}/panels`, "POST", form)).body.panel);
    }
    return { b, panels };
  }

  async function untilDone(id: string) {
    const deadline = Date.now() + TIMEOUT_MS;
    while (Date.now() < deadline) {
      const r = (await call(`/api/renders/${id}`)).body.render;
      if (r.status === "completed" || r.status === "failed") return r;
      await new Promise((res) => setTimeout(res, 500));
    }
    throw new Error(`render ${id} did not finish in ${TIMEOUT_MS} ms`);
  }

  it("the engine serves every configured model", async () => {
    const { engines } = (await call("/api/config")).body;
    const client = new EngineClient(URL_!, KEY);
    const r = await fetch(`${URL_!.replace(/\/+$/, "")}/v1/models`, { headers: { authorization: `Bearer ${KEY}` } });
    expect(r.status).toBe(200);
    const served = new Set(((await r.json()) as { data: { id: string }[] }).data.map((m) => m.id));
    for (const e of engines) {
      for (const m of [e.draft_model, e.final_model, e.reference_model].filter(Boolean)) {
        expect(served, `model ${m} is not served`).toContain(m);
      }
    }
    expect(client).toBeTruthy();
  });

  it("a keyframe bake-off completes on both engines with timing", async () => {
    const { b, panels } = await boardWithShot();
    await call(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", { prompt: "a slow push in", seconds: 6, seed: 7 });
    const out = await call(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", {});
    expect(out.status).toBe(202);
    for (const started of out.body.renders) {
      expect(started.error, `${started.engine} refused: ${started.error}`).toBeNull();
      const done = await untilDone(started.id);
      expect(done.status, `${done.engine}: ${done.error}`).toBe("completed");
      expect(done.video_key).toBeTruthy();
      expect(done.clip_seconds).toBeGreaterThan(0);
      expect(done.wall_time_s).not.toBeNull();
    }
  }, TIMEOUT_MS * 2);

  it("a reference shot runs ref2va on H3", async () => {
    const { b, panels } = await boardWithShot(3);
    await call(`/api/boards/${b.id}/shots/${panels[0].id}`, "PUT", {
      mode: "reference",
      seconds: 5,
      reference_panel_ids: [panels[2].id],
    });
    const out = await call(`/api/boards/${b.id}/shots/${panels[0].id}/renders`, "POST", { engines: ["h3"] });
    expect(out.status).toBe(202);
    const done = await untilDone(out.body.renders[0].id);
    expect(done.status, done.error).toBe("completed");
  }, TIMEOUT_MS);
});
