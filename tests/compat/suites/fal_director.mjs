// fal realtime director through the real `@fal-ai/client` alpha
// (`fal.realtime.open(wma(...))`) in headless Chromium, against fv-serve
// on the fake engine (design §5.6, §7.5). Run by tests/compat/run.sh.
//
// Adapted from crates/fastvideo-fal/tests/director_browser/wma_compat.mjs
// (which drives an in-process fixture with a 480p tier and a silent H3):
// here the app from FV_WMA_APP (default minimax/h3-max/director), the
// resolution from FV_WMA_RESOLUTION (fv-serve's fake H3 is
// 768p only), the alpha client is the `fal-client-alpha` npm alias from
// tests/compat/package.json, and the video-only scenario runs only when an
// app is given.
//
// Scenarios, each a separate WMA session (one session per machine):
//   1. requestMiddleware (wma.fal.run -> <base>/wma/, fal.run -> <base>/run/),
//      receive ["video","audio"]: session_info, strict-schema error,
//      configure/configured, chunks, prompt versions, ping/pong, network
//      info, A/V rates from getStats, heartbeats across > 15 s, stop.
//   2. proxyUrl {url: <base>/fal/proxy, when: "always"}, legacy receive
//      (one recvonly video transceiver): video at 24 fps.
//   3. (optional) a video-only app, receive ["video","audio"]: audio
//      answered inactive, video at 24 fps, no audio samples.
//
// usage: node fal_director.mjs <modules dir> <base url> <key> [<video-only app>]
// Prints one JSON summary line last.

import { createRequire } from "node:module";
import path from "node:path";

const [, , modulesDir, base, key, silentApp = ""] = process.argv;
const RES = process.env.FV_WMA_RESOLUTION || "768p";
const APP = process.env.FV_WMA_APP || "minimax/h3-max/director";
const ALPHA = process.env.FV_FAL_ALPHA_PKG || "fal-client-alpha";
const require = createRequire(path.join(modulesDir, "package.json"));
const esbuild = require("esbuild");
const { chromium } = process.env.FV_PLAYWRIGHT
  ? createRequire(path.join(process.env.FV_PLAYWRIGHT, "package.json"))(process.env.FV_PLAYWRIGHT)
  : require("playwright");

const bundle = await esbuild.build({
  stdin: {
    contents: `
      import { createFalClient } from "${ALPHA}";
      import { wma } from "${ALPHA}/realtime/wma";
      window.FalCompat = { createFalClient, wma };
    `,
    resolveDir: modulesDir,
    loader: "js",
  },
  bundle: true,
  format: "iife",
  platform: "browser",
  write: false,
  logLevel: "silent",
});
const clientJs = bundle.outputFiles[0].text;

const browser = await chromium.launch({
  args: ["--autoplay-policy=no-user-gesture-required", "--use-fake-ui-for-media-stream"],
});
const checks = [];
const results = {};
try {
  const page = await browser.newPage();
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") console.error(`[page ${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => console.error(`[page error] ${e.message}`));
  // Record every peer connection for getStats (test instrumentation only).
  await page.addInitScript(() => {
    const Native = window.RTCPeerConnection;
    window.__pcs = [];
    window.RTCPeerConnection = function (...args) {
      const pc = new Native(...args);
      window.__pcs.push(pc);
      return pc;
    };
    window.RTCPeerConnection.prototype = Native.prototype;
    Object.setPrototypeOf(window.RTCPeerConnection, Native);
  });
  await page.route(`${base}/__compat/`, (route) =>
    route.fulfill({ contentType: "text/html", body: "<!doctype html><title>wma compat</title><video id=v muted autoplay playsinline></video>" }),
  );
  await page.goto(`${base}/__compat/`);
  await page.addScriptTag({ content: clientJs });

  const scenario = async ({ base, key, mode, app, receive, long, expectAudio, res }) => {
    const { createFalClient, wma } = window.FalCompat;
    const log = [];
    const check = (cond, what) => {
      if (!cond) throw new Error(`check failed: ${what} :: ${JSON.stringify(log.slice(-6))}`);
    };
    const rewrite = (url) =>
      url.replace(/^https:\/\/wma\.fal\.run\//, `${base}/wma/`).replace(/^https:\/\/fal\.run\//, `${base}/run/`);
    const fal =
      mode === "middleware"
        ? createFalClient({ credentials: key, requestMiddleware: async (req) => ({ ...req, url: rewrite(req.url) }) })
        : createFalClient({ credentials: key, proxyUrl: { url: `${base}/fal/proxy`, when: "always" } });
    const msgs = [];
    const waiters = [];
    const onData = (raw) => {
      const m = JSON.parse(raw);
      log.push(m.type === "chunk" ? { type: "chunk", chunk_index: m.chunk_index } : m);
      msgs.push(m);
      for (const w of [...waiters]) w();
    };
    const next = (pred, what, ms = 60000) =>
      new Promise((resolve, reject) => {
        const t0 = Date.now();
        const look = () => {
          const i = msgs.findIndex(pred);
          if (i >= 0) {
            const [m] = msgs.splice(i, 1);
            waiters.splice(waiters.indexOf(look), 1);
            clearInterval(timer);
            resolve(m);
          } else if (Date.now() - t0 > ms) {
            waiters.splice(waiters.indexOf(look), 1);
            clearInterval(timer);
            reject(new Error(`timed out waiting for ${what} :: ${JSON.stringify(log.slice(-8))}`));
          }
        };
        const timer = setInterval(look, 100);
        waiters.push(look);
        look();
      });
    const type = (t) => (m) => m.type === t;
    const video = document.getElementById("v");
    let presented = 0;
    const countFrames = () => {
      presented++;
      video.requestVideoFrameCallback(countFrames);
    };
    const states = [];
    // One session per machine: retry while the previous one releases.
    let handle;
    for (let attempt = 0; ; attempt++) {
      const pcsBefore = window.__pcs.length;
      handle = fal.realtime.open(wma(app), {
        ...(receive ? { receive } : {}),
        onData,
        onState: (s) => states.push(s),
        onMedia: (stream) => {
          video.srcObject = stream;
          video.play().catch(() => {});
          video.requestVideoFrameCallback(countFrames);
        },
      });
      try {
        await handle.ready;
        break;
      } catch (e) {
        if (attempt >= 20 || !/already running|busy|already open/.test(String(e))) throw e;
        await new Promise((r) => setTimeout(r, 1000));
      }
      window.__pcs.length = pcsBefore;
    }
    const opened = Date.now();
    const pc = window.__pcs[window.__pcs.length - 1];
    const timeline = [];
    const mark = (what) => timeline.push([what, Date.now() - opened]);
    pc.addEventListener("connectionstatechange", () => mark(`pc:${pc.connectionState}`));
    let firstA = false;
    let firstV = false;
    const poll = setInterval(async () => {
      const r = await pc.getStats();
      r.forEach((x) => {
        if (x.type === "inbound-rtp" && x.kind === "audio" && !firstA && x.packetsReceived > 0) {
          firstA = true;
          mark("first audio rtp");
        }
        if (x.type === "inbound-rtp" && x.kind === "video" && !firstV && x.packetsReceived > 0) {
          firstV = true;
          mark("first video rtp");
        }
      });
    }, 100);
    mark(`ready (pc ${pc.connectionState})`);
    const out = { mode, app, sessionId: handle.session.sessionId };
    check(/^[0-9a-f-]{36}$/.test(out.sessionId), "uuid session id");

    const info = await next(type("session_info"), "session_info");
    mark("session_info");
    check(info.fps === 24 && info.audio_sample_rate === 48000, "session_info constants");
    check(info.default_chunk_duration === 5 && JSON.stringify(info.chunk_duration_options) === "[5,10]", `chunk_duration options ${JSON.stringify(info.chunk_duration_options)}`);
    out.info = { fps: info.fps, resolutions: info.resolutions, app: info.app };

    // Strict schema: an extra property is refused as a diagnostic.
    handle.send({ type: "ping", ts: 1, extra: true });
    const inv = await next(type("error"), "invalid_message");
    check(inv.code === "invalid_message", `invalid_message: ${JSON.stringify(inv)}`);

    handle.send({ type: "configure", prompt_version: 1, prompt: "A lighthouse keeper climbs the stairs at dusk", resolution: res, aspect_ratio: "16:9", memory: 3, protocol_version: 1 });
    const cfgd = await next(type("configured"), "configured");
    check(cfgd.prompt_version === 1 && cfgd.resolution === res && cfgd.chunk_duration === 5, `configured ${JSON.stringify(cfgd)}`);
    const ch0 = await next((m) => m.type === "chunk" && m.chunk_index === 0, "chunk 0", 90000);
    mark("chunk 0");
    check(ch0.generated_frame_count > 0 && ch0.route === "unknown", "chunk 0 fields");

    handle.send({ type: "prompt", prompt_version: 2, prompt: "They follow a narrow path down to the harbor." });
    await next((m) => m.type === "prompt_pending" && m.prompt_version === 2, "prompt_pending v2");
    handle.send({ type: "prompt", prompt_version: 2, prompt: "reused version" });
    const stale = await next(type("prompt_rejected"), "stale rejection");
    check(stale.reason === "stale_prompt_version", "stale_prompt_version");
    await next((m) => m.type === "prompt_applied" && m.prompt_version === 2, "prompt_applied v2", 90000);

    handle.send({ type: "ping", ts: 42.5 });
    const pong = await next(type("pong"), "pong");
    check(pong.client_ts === 42.5, "pong.client_ts");

    const conn = await handle.session.getConnectionInfo();
    check(conn.runner.side === "runner" && conn.browser.side === "browser", "getConnectionInfo");

    // A/V rates over a window, from the browser's own stats.
    const snap = async () => {
      const r = await pc.getStats();
      const o = { t: performance.now() };
      r.forEach((s) => {
        if (s.type === "inbound-rtp" && s.kind === "video") {
          o.framesDecoded = s.framesDecoded;
          o.framesReceived = s.framesReceived;
          o.width = s.frameWidth;
          o.height = s.frameHeight;
          o.videoCodec = s.codecId && r.get(s.codecId)?.mimeType;
        }
        if (s.type === "inbound-rtp" && s.kind === "audio") {
          o.samples = s.totalSamplesReceived;
          o.audioCodec = s.codecId && r.get(s.codecId)?.mimeType;
          o.audioChannels = s.codecId && r.get(s.codecId)?.channels;
        }
      });
      o.presented = presented;
      return o;
    };
    // Measure in steady state: from one second after the first frame.
    for (let i = 0; i < 300; i++) {
      const s0 = await snap();
      if ((s0.framesDecoded || 0) >= 24) break;
      await new Promise((r) => setTimeout(r, 100));
    }
    mark("snap a");
    const a = await snap();
    await new Promise((r) => setTimeout(r, 4000));
    const b = await snap();
    const dt = (b.t - a.t) / 1000;
    out.video = {
      codec: b.videoCodec,
      width: b.width,
      height: b.height,
      decodedFps: (b.framesDecoded - a.framesDecoded) / dt,
      receivedFps: (b.framesReceived - a.framesReceived) / dt,
      presentedFps: (b.presented - a.presented) / dt,
    };
    out.audio = expectAudio
      ? { codec: b.audioCodec, channels: b.audioChannels, samplesPerSecond: (b.samples - a.samples) / dt }
      : { samples: b.samples ?? 0 };
    if (!(Math.abs(out.video.decodedFps - 24) < 2.5)) {
      const all = [];
      (await pc.getStats()).forEach((s) => {
        if (/rtp|codec|transport|candidate-pair/.test(s.type)) all.push(s);
      });
      out.debugStats = all;
      out.pcs = window.__pcs.length;
      out.pcState = pc.connectionState;
    }
    out.timeline = timeline;
    clearInterval(poll);
    check(Math.abs(out.video.decodedFps - 24) < 2.5, `video decoded at ${out.video.decodedFps} fps; ${JSON.stringify({ timeline, a, b })}`);
    if (expectAudio) {
      check(Math.abs(out.audio.samplesPerSecond - 48000) < 4800, `audio at ${out.audio.samplesPerSecond} samples/s`);
      check(/opus/i.test(out.audio.codec || ""), `audio codec ${out.audio.codec}`);
    } else {
      check(!out.audio.samples, "no audio on a video-only session");
    }

    // Heartbeats keep the lease beyond 15 s (three missed beats).
    if (long) {
      while (Date.now() - opened < 17000) await new Promise((r) => setTimeout(r, 500));
      handle.send({ type: "ping", ts: 7 });
      await next((m) => m.type === "pong" && m.client_ts === 7, "pong after 17 s");
      check(handle.state === "live", `still live after 17 s (${handle.state})`);
      out.aliveAfterMs = Date.now() - opened;
    }

    handle.send({ type: "stop" });
    const ex = await next(type("stream_exhausted"), "stream_exhausted");
    check(ex.reason === "stopped" && ex.chunks >= 1, `stream_exhausted ${JSON.stringify(ex)}`);
    const fin = await next((m) => m.type === "session_metrics" && m.final === true, "final session_metrics");
    check(fin.units === "ms", "session_metrics units");
    await handle.close();
    out.states = states;
    out.messageTypes = [...new Set(log.map((m) => m.type))];
    return out;
  };

  results.middleware = await page.evaluate(scenario, {
    base, key, mode: "middleware", app: APP, receive: ["video", "audio"], long: true, expectAudio: true, res: RES,
  });
  checks.push("requestMiddleware: A/V session, strict schemas, versions, heartbeats, stop");
  results.proxy = await page.evaluate(scenario, {
    base, key, mode: "proxy", app: APP, receive: null, long: false, expectAudio: false, res: RES,
  });
  checks.push("proxyUrl: legacy receive (video only transceiver)");
  if (silentApp) {
    results.videoOnly = await page.evaluate(scenario, {
      base, key, mode: "middleware", app: silentApp, receive: ["video", "audio"], long: false, expectAudio: false, res: RES,
    });
    checks.push("video-only model: audio inactive, video at 24 fps");
  }
} finally {
  await browser.close();
}
console.log(JSON.stringify({ checks, results }));
