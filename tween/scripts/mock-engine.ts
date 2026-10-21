// Mock fv-serve for UI work without Docker or a GPU: the same /v1/videos
// rules as the test fake (test/support/fake-engine.ts) on a timed clock.
// With ffmpeg on PATH, a finished job is a real MP4 that crossfades from the
// first panel to the last, so players and the animatic have something to
// play. It is not a model: use the Docker fake engine (task tween:engine:up)
// to test against real fv-serve, and a GPU pool for real motion.
//
//   node --experimental-strip-types scripts/mock-engine.ts
//   MOCK_ENGINE_PORT=8010 MOCK_ENGINE_KEY=sk-local MOCK_ENGINE_SCALE=1
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { DEFAULT_MODELS, MP4, fakeEngine } from "../test/support/fake-engine.ts";

const run = promisify(execFile);
const PORT = Number(process.env.MOCK_ENGINE_PORT ?? 8010);
const KEY = process.env.MOCK_ENGINE_KEY ?? "sk-local";
const SCALE = Number(process.env.MOCK_ENGINE_SCALE ?? 1);
const SHORT_EDGE = 480;

async function hasFfmpeg(): Promise<boolean> {
  try {
    await run("ffmpeg", ["-hide_banner", "-version"]);
    return true;
  } catch {
    return false;
  }
}

function canvas(aspect: unknown): [number, number] {
  const m = /^(\d+):(\d+)$/.exec(String(aspect ?? "16:9"));
  const [a, b] = m ? [Number(m[1]), Number(m[2])] : [16, 9];
  const even = (x: number) => Math.round(x / 2) * 2;
  return a >= b ? [even((SHORT_EDGE * a) / b), SHORT_EDGE] : [SHORT_EDGE, even((SHORT_EDGE * b) / a)];
}

function decodeDataUri(uri: string): Buffer {
  const comma = uri.indexOf(",");
  if (!uri.startsWith("data:") || comma < 0) throw new Error("expected a data URI");
  return Buffer.from(uri.slice(comma + 1), "base64");
}

/** A crossfade from the first image reference to the last, `seconds` long. */
async function crossfade(body: Record<string, unknown>): Promise<Uint8Array<ArrayBuffer>> {
  const refs = (body.image_reference as { image_url: string }[] | undefined) ?? [];
  if (refs.length === 0) return MP4;
  const seconds = Math.max(1, Number(body.seconds) || 5);
  const [w, h] = canvas(body.aspect_ratio);
  const dir = await mkdtemp(join(tmpdir(), "tween-mock-"));
  try {
    const first = join(dir, "first");
    const last = join(dir, "last");
    await writeFile(first, decodeDataUri(refs[0]!.image_url));
    await writeFile(last, decodeDataUri(refs[refs.length - 1]!.image_url));
    const fit = `scale=${w}:${h}:force_original_aspect_ratio=increase,crop=${w}:${h},setsar=1,fps=24,format=yuv420p`;
    const out = join(dir, "out.mp4");
    await run("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-loop", "1", "-t", String(seconds), "-i", first,
      "-loop", "1", "-t", String(seconds), "-i", last,
      "-filter_complex",
      `[0]${fit}[a];[1]${fit}[b];[a][b]xfade=transition=fade:duration=${(seconds * 0.8).toFixed(2)}:offset=${(seconds * 0.1).toFixed(2)},format=yuv420p[v]`,
      "-map", "[v]", "-t", String(seconds),
      "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
      out,
    ]);
    const bytes = await readFile(out);
    return new Uint8Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const ffmpeg = await hasFfmpeg();
const engine = fakeEngine(KEY, DEFAULT_MODELS, {
  timed: { queued_ms: 400, scale: SCALE },
  content: ffmpeg ? crossfade : undefined,
});

createServer(async (req, res) => {
  try {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = chunks.length ? Buffer.concat(chunks) : undefined;
    const headers = new Headers();
    for (const [k, v] of Object.entries(req.headers)) if (typeof v === "string") headers.set(k, v);
    const r = await engine.fetch(`http://${req.headers.host ?? "localhost"}${req.url ?? "/"}`, {
      method: req.method,
      headers,
      body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
    });
    res.writeHead(r.status, Object.fromEntries(r.headers));
    res.end(Buffer.from(await r.arrayBuffer()));
    if (req.method === "POST") console.log(`mock-engine: ${r.status} POST ${req.url}`);
  } catch (e) {
    res.writeHead(500, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: e instanceof Error ? e.message : String(e) } }));
  }
}).listen(PORT, () => {
  console.log(`mock-engine: http://127.0.0.1:${PORT} (key ${KEY}, ${ffmpeg ? "ffmpeg crossfades" : "placeholder MP4s: install ffmpeg for playable clips"})`);
});
