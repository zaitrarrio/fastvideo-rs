// fv-serve `/v1/videos` client (crates/fastvideo-openai-videos/src/videos.rs).
//
// - Two `image_reference` entries and no `task` normalise to
//   `Task::Keyframes` (first + last) on every family.
// - `task: "ref2va"` is accepted on H3 models only (`apply_model`).
// - The job view carries `inference_time_s`, the engine-side generation time
//   that the real-time factor is computed from.
import type { RenderStatus, ShotMode } from "./types";

export interface BuildInput {
  family: "ltx2" | "h3";
  model: string;
  mode: ShotMode;
  prompt: string;
  seconds: number;
  seed: number;
  aspect_ratio: string;
  /** Data URIs: [first, last] for keyframes, the reference set otherwise. */
  images: string[];
}

export type VideoRequestBody = Record<string, unknown> & {
  model: string;
  prompt: string;
  image_reference: { image_url: string }[];
  task?: "ref2va";
};

export function buildRequest(i: BuildInput): VideoRequestBody {
  if (i.mode === "keyframes" && i.images.length !== 2) {
    throw new Error("a keyframe shot needs exactly two images (first and last panel)");
  }
  // fv-serve refuses `prompt.trim().is_empty()` on every model (videos.rs).
  if (!i.prompt.trim()) throw new Error("a shot needs a prompt before it can render");
  if (i.mode === "reference") {
    if (i.family !== "h3") throw new Error("reference shots run on H3 models only");
    if (i.images.length === 0) throw new Error("a reference shot needs at least one image");
  }
  const body: VideoRequestBody = {
    model: i.model,
    prompt: i.prompt.trim(),
    seconds: i.seconds,
    seed: i.seed,
    aspect_ratio: i.aspect_ratio,
    image_reference: i.images.map((image_url) => ({ image_url })),
  };
  if (i.mode === "reference") body.task = "ref2va";
  return body;
}

export interface EngineJob {
  id: string;
  status: RenderStatus;
  progress: number;
  seconds: number | null;
  size: string | null;
  created_at_s: number | null;
  completed_at_s: number | null;
  inference_time_s: number | null;
  error: string | null;
}

const STATUSES: ReadonlySet<string> = new Set(["queued", "in_progress", "completed", "failed"]);

const num = (v: unknown): number | null => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
  return null;
};

export function parseJob(raw: unknown): EngineJob {
  const o = (raw ?? {}) as Record<string, unknown>;
  if (typeof o.id !== "string" || !o.id) throw new Error("engine reply has no job id");
  if (typeof o.status !== "string" || !STATUSES.has(o.status)) {
    throw new Error(`engine reply has an unknown status: ${String(o.status)}`);
  }
  const err = o.error as { message?: unknown } | null | undefined;
  return {
    id: o.id,
    status: o.status as RenderStatus,
    progress: Math.max(0, Math.min(100, Math.round(num(o.progress) ?? 0))),
    seconds: num(o.seconds),
    size: typeof o.size === "string" ? o.size : null,
    created_at_s: num(o.created_at),
    completed_at_s: num(o.completed_at),
    inference_time_s: num(o.inference_time_s),
    error: err && typeof err.message === "string" ? err.message : null,
  };
}

export class EngineError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "EngineError";
  }
}

export interface EngineContent {
  stream: ReadableStream<Uint8Array>;
  contentType: string;
}

export class EngineClient {
  private readonly base: string;

  constructor(
    baseUrl: string,
    private readonly key: string,
    private readonly fetcher: typeof fetch = fetch,
  ) {
    this.base = baseUrl.replace(/\/+$/, "");
  }

  submit(body: Record<string, unknown>): Promise<EngineJob> {
    return this.json("/v1/videos", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(parseJob);
  }

  get(jobId: string): Promise<EngineJob> {
    return this.json(`/v1/videos/${encodeURIComponent(jobId)}`, { method: "GET" }).then(parseJob);
  }

  /** The finished MP4 (a local file, or a redirect to a presigned URL). */
  async content(jobId: string): Promise<EngineContent> {
    const r = await this.send(`/v1/videos/${encodeURIComponent(jobId)}/content`, { method: "GET" });
    if (!r.ok || !r.body) throw await this.error(r);
    return { stream: r.body, contentType: r.headers.get("content-type") || "video/mp4" };
  }

  private async json(path: string, init: RequestInit): Promise<unknown> {
    const r = await this.send(path, init);
    if (!r.ok) throw await this.error(r);
    return r.json();
  }

  private async send(path: string, init: RequestInit): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set("authorization", `Bearer ${this.key}`);
    // Called unbound: workerd throws "Illegal invocation" when the global
    // fetch runs as a method of another object.
    const f = this.fetcher;
    try {
      return await f(this.base + path, { ...init, headers });
    } catch (e) {
      throw new EngineError(502, `engine unreachable: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  private async error(r: Response): Promise<EngineError> {
    const text = await r.text().catch(() => "");
    let message = text.slice(0, 500) || `engine answered ${r.status}`;
    try {
      const j = JSON.parse(text) as { error?: { message?: unknown } | string; detail?: unknown };
      if (typeof j.error === "string") message = j.error;
      else if (j.error && typeof j.error.message === "string") message = j.error.message;
      else if (typeof j.detail === "string") message = j.detail;
    } catch {
      // Not JSON: keep the text.
    }
    return new EngineError(r.status, message);
  }
}
