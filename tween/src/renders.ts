// Submitting shots to fv-serve and pulling their results (ADR-0004
// decisions 4, 6, 7, 9).
import { EngineClient, EngineError, buildRequest } from "./engine-client";
import { engineById, modelFor, type EngineConfig } from "./engines";
import { clipKey, dataUri } from "./media";
import { newId, type Store } from "./store";
import { TERMINAL, type Board, type EngineId, type Panel, type Render, type Shot, type Tier } from "./types";
import { LIMITS } from "./validate";

export class RenderError extends Error {
  constructor(
    readonly status: 400 | 404 | 409 | 502,
    message: string,
  ) {
    super(message);
    this.name = "RenderError";
  }
}

export interface RenderDeps {
  store: Store;
  media: R2Bucket;
  client: EngineClient;
  engines: EngineConfig[];
  now: () => number;
  randomSeed?: () => number;
}

export interface SubmitInput {
  board: Board;
  shot: Shot;
  panels: Panel[];
  engines: EngineId[];
  tier: Tier;
  /** Explicit seed (a finalize, or the user's); else the shot's, else random. */
  seed: number | null;
}

const randomSeed = () => {
  const b = new Uint32Array(1);
  crypto.getRandomValues(b);
  return b[0]! % (LIMITS.seed_max + 1);
};

/** The images the engine sees for this shot: [first, last] or the reference set. */
function shotImageIds(shot: Shot): string[] {
  if (shot.mode === "keyframes") return [shot.start_panel_id, shot.end_panel_id];
  return [...new Set([shot.start_panel_id, shot.end_panel_id, ...shot.reference_panel_ids])];
}

async function loadDataUris(deps: RenderDeps, panels: Panel[], ids: string[]): Promise<string[]> {
  const byId = new Map(panels.map((p) => [p.id, p]));
  return Promise.all(
    ids.map(async (id) => {
      const p = byId.get(id);
      if (!p) throw new RenderError(409, `panel ${id} is no longer on the board`);
      const obj = await deps.media.get(p.image_key);
      if (!obj) throw new RenderError(409, `the image of panel ${id} is missing`);
      return dataUri(new Uint8Array(await obj.arrayBuffer()), p.mime);
    }),
  );
}

/**
 * Submits one shot to each requested engine with the same prompt, seconds,
 * seed and panels. Every engine is checked before any job is sent, so a
 * bake-off either starts on all engines or on none.
 */
export async function submitShot(deps: RenderDeps, input: SubmitInput): Promise<{ bakeoff_id: string | null; renders: Render[] }> {
  const plan = input.engines.map((id) => {
    const engine = engineById(deps.engines, id);
    if (!engine) throw new RenderError(400, `unknown engine ${id}`);
    const choice = modelFor(engine, input.tier, input.shot.mode);
    if (!choice.ok) throw new RenderError(400, choice.reason);
    return { engine, model: choice.model };
  });
  const images = await loadDataUris(deps, input.panels, shotImageIds(input.shot));
  const seed = input.seed ?? input.shot.seed ?? (deps.randomSeed ?? randomSeed)();
  const bakeoff_id = plan.length > 1 ? newId("bk") : null;

  const renders = await Promise.all(
    plan.map(async ({ engine, model }) => {
      const render = await deps.store.insertRender({
        board_id: input.board.id,
        start_panel_id: input.shot.start_panel_id,
        end_panel_id: input.shot.end_panel_id,
        bakeoff_id,
        engine: engine.id,
        tier: input.tier,
        model,
        mode: input.shot.mode,
        prompt: input.shot.prompt,
        seconds: input.shot.seconds,
        seed,
      });
      const body = buildRequest({
        family: engine.family,
        model,
        mode: input.shot.mode,
        prompt: input.shot.prompt,
        seconds: input.shot.seconds,
        seed,
        aspect_ratio: input.board.aspect_ratio,
        images,
      });
      try {
        const job = await deps.client.submit(body);
        const patch = { job_id: job.id, status: job.status, progress: job.progress, error: job.error };
        await deps.store.updateRender(render.id, patch);
        return { ...render, ...patch };
      } catch (e) {
        // A refusal (unsupported task, clip length, canvas) is the engine's
        // answer for this engine only: record it, keep the other renders.
        const error = e instanceof EngineError ? e.message : String(e);
        const patch = { status: "failed" as const, error, completed_at: deps.now() };
        await deps.store.updateRender(render.id, patch);
        return { ...render, ...patch };
      }
    }),
  );
  return { bakeoff_id, renders };
}

/**
 * Pulls a running render's engine job. On success the MP4 is copied into R2
 * once and the metrics are frozen; terminal renders are returned as stored.
 */
export async function refreshRender(deps: RenderDeps, render: Render): Promise<Render> {
  if (TERMINAL.has(render.status) || !render.job_id) return render;
  let job;
  try {
    job = await deps.client.get(render.job_id);
  } catch (e) {
    if (e instanceof EngineError && e.status === 404) {
      const patch = { status: "failed" as const, error: "the engine no longer has this job", completed_at: deps.now() };
      await deps.store.updateRender(render.id, patch);
      return { ...render, ...patch };
    }
    // Transient: report the stored state; the next poll retries.
    return render;
  }

  if (job.status === "completed") {
    const key = clipKey(render.board_id, render.id);
    try {
      const content = await deps.client.content(job.id);
      const bytes = await new Response(content.stream).arrayBuffer();
      await deps.media.put(key, bytes, { httpMetadata: { contentType: content.contentType } });
    } catch (e) {
      // The job is done but the copy failed (engine or R2 hiccup): keep the
      // render running so the next poll retries the download.
      console.warn(`tween: copying ${job.id} failed, will retry`, e instanceof Error ? e.message : e);
      return render;
    }
    const completed_at = deps.now();
    const wall =
      job.created_at_s !== null && job.completed_at_s !== null && job.completed_at_s >= job.created_at_s
        ? job.completed_at_s - job.created_at_s
        : (completed_at - render.created_at) / 1000;
    const patch = {
      status: "completed" as const,
      progress: 100,
      error: null,
      video_key: key,
      clip_seconds: job.seconds ?? render.seconds,
      inference_time_s: job.inference_time_s,
      wall_time_s: wall,
      size: job.size,
      completed_at,
    };
    await deps.store.updateRender(render.id, patch);
    return { ...render, ...patch };
  }

  if (job.status === "failed") {
    const patch = { status: "failed" as const, progress: job.progress, error: job.error ?? "the engine failed this job", completed_at: deps.now() };
    await deps.store.updateRender(render.id, patch);
    return { ...render, ...patch };
  }

  if (job.status !== render.status || job.progress !== render.progress) {
    const patch = { status: job.status, progress: job.progress };
    await deps.store.updateRender(render.id, patch);
    return { ...render, ...patch };
  }
  return render;
}
