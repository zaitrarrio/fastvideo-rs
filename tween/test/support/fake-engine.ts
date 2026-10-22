// An in-process stand-in for fv-serve's /v1/videos with the rules Tween
// depends on (crates/fastvideo-openai-videos/src/videos.rs):
//   - bearer auth; blank prompts 400; unknown models 404
//   - `task` only on H3 models; `ref2va` needs a reference-loaded model
//   - one or two images without `task` are i2v / keyframes
//   - jobs go queued -> in_progress -> completed, one step per GET
//   - `inference_time_s` and `seconds` on the job view; MP4 at /content
export interface FakeModel {
  family: "ltx2" | "h3";
  ref2va?: boolean;
  /** Engine-side generation time reported for a finished job. */
  inference_s: number;
  /** H3 refuses outside 5..15 s. */
  seconds?: { min: number; max: number };
  /** Fail the job at the in_progress step with this message. */
  fail?: string;
}

export const DEFAULT_MODELS: Record<string, FakeModel> = {
  "fake-ltx-turbo": { family: "ltx2", inference_s: 1.5 },
  "fake-ltx-pro": { family: "ltx2", inference_s: 4 },
  "fake-h3-turbo": { family: "h3", inference_s: 2, seconds: { min: 5, max: 15 } },
  "fake-h3-max": { family: "h3", ref2va: true, inference_s: 6, seconds: { min: 5, max: 15 } },
};

interface Job {
  id: string;
  model: string;
  body: Record<string, unknown>;
  step: number;
  created_at: number;
  /** Wall-clock submit time (ms), for the timed clock. */
  started_ms: number;
  video?: Promise<Uint8Array<ArrayBuffer>>;
}

export interface FakeEngineOptions {
  /**
   * Timed clock for interactive use (scripts/mock-engine.ts): a job is
   * queued for `queued_ms`, then runs for its model's `inference_s` scaled
   * by `scale`. Without it a job advances one state per GET (tests).
   */
  timed?: { queued_ms: number; scale: number };
  /** The MP4 a finished job serves; defaults to a 12-byte placeholder. */
  content?: (body: Record<string, unknown>, model: FakeModel) => Promise<Uint8Array<ArrayBuffer>>;
}

export const MP4 = new Uint8Array([0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d]);

export function fakeEngine(key = "sk-test", models: Record<string, FakeModel> = DEFAULT_MODELS, opts: FakeEngineOptions = {}) {
  const jobs = new Map<string, Job>();
  const submitted: Record<string, unknown>[] = [];
  let n = 0;
  let clock = 1_700_000_000;

  const err = (status: number, message: string, param?: string) =>
    new Response(JSON.stringify({ error: { message, type: "invalid_request_error", param: param ?? null, code: null } }), {
      status,
      headers: { "content-type": "application/json" },
    });

  /** [step, progress] under the timed clock. */
  const timedStep = (j: Job, m: FakeModel): [number, number] => {
    const t = opts.timed!;
    const ran = Date.now() - j.started_ms - t.queued_ms;
    if (ran < 0) return [0, 0];
    const run = m.inference_s * 1000 * t.scale;
    return ran >= run ? [2, 100] : [1, Math.min(99, Math.round((ran / run) * 100))];
  };

  const view = (j: Job) => {
    const m = models[j.model]!;
    const [step, timedProgress] = opts.timed ? timedStep(j, m) : [j.step, -1];
    const status = step === 0 ? "queued" : step === 1 ? "in_progress" : m.fail ? "failed" : "completed";
    const done = status === "completed";
    const progress = done ? 100 : step === 1 ? (timedProgress >= 0 ? timedProgress : 50) : 0;
    return {
      id: j.id,
      object: "video",
      model: j.model,
      status,
      progress,
      created_at: j.created_at,
      completed_at: done
        ? opts.timed
          ? Math.round((j.started_ms + opts.timed.queued_ms + m.inference_s * 1000 * opts.timed.scale) / 1000)
          : j.created_at + Math.ceil(m.inference_s) + 1
        : null,
      seconds: String(j.body.seconds),
      size: "1280x768",
      inference_time_s: done ? m.inference_s * (opts.timed?.scale ?? 1) : null,
      error: status === "failed" ? { code: "generation_failed", message: m.fail } : null,
    };
  };

  const fetchImpl = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const req = new Request(input, init);
    const url = new URL(req.url);
    if (req.headers.get("authorization") !== `Bearer ${key}`) return err(401, "invalid api key");

    if (req.method === "POST" && url.pathname === "/v1/videos") {
      const body = (await req.json()) as Record<string, unknown>;
      submitted.push(body);
      // videos.rs normalize: `prompt.trim().is_empty()` is refused on every model.
      if (typeof body.prompt !== "string" || !body.prompt.trim()) return err(400, "prompt must not be empty", "prompt");
      const m = models[String(body.model)];
      if (!m) return err(404, `model \`${String(body.model)}\` is not served`, "model");
      const images = (body.image_reference as unknown[] | undefined) ?? [];
      if (body.task !== undefined) {
        if (m.family !== "h3") return err(400, "`task` is only defined for H3 models", "task");
        if (body.task === "ref2va" && !m.ref2va) return err(400, "ref2va is not loaded on this model", "task");
      } else if (images.length > 2) {
        if (!m.ref2va) return err(400, "this model does not support reference-to-video", "image_reference");
      }
      const secs = Number(body.seconds);
      if (m.seconds && (secs < m.seconds.min || secs > m.seconds.max)) {
        return err(400, `seconds must be ${m.seconds.min}..${m.seconds.max} for this model`, "seconds");
      }
      const job: Job = {
        id: `video_${++n}`,
        model: String(body.model),
        body,
        step: 0,
        created_at: opts.timed ? Math.floor(Date.now() / 1000) : clock++,
        started_ms: Date.now(),
      };
      if (opts.content) {
        job.video = opts.content(body, m);
        job.video.catch(() => {});
      }
      jobs.set(job.id, job);
      return Response.json(view(job));
    }

    if (req.method === "GET" && url.pathname === "/v1/models") {
      return Response.json({
        object: "list",
        data: Object.keys(models).map((id) => ({ id, object: "model", created: 0, owned_by: "fastvideo", root: id })),
      });
    }

    const m = /^\/v1\/videos\/([^/]+)(\/content)?$/.exec(url.pathname);
    if (req.method === "GET" && m) {
      const job = jobs.get(decodeURIComponent(m[1]!));
      if (!job) return err(404, "no such video");
      if (m[2]) {
        if (view(job).status !== "completed") return err(404, "the video is not ready");
        const bytes = job.video ? await job.video : MP4;
        return new Response(bytes, { headers: { "content-type": "video/mp4" } });
      }
      const v = view(job);
      if (job.step < 2) job.step++;
      return Response.json(v);
    }
    return err(404, "no such route");
  };

  return { fetch: fetchImpl as typeof fetch, jobs, submitted, forget: (id: string) => jobs.delete(id) };
}
