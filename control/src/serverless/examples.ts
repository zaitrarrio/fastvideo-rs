// Ready-made test invokes for a serverless endpoint, generated from what it
// serves (serves.ts): one per model × task × API, shaped for the endpoint's
// mode. Queue: the native job envelope, an http job that waits on the job it
// creates (crates/fastvideo-deploy/src/runpod/mod.rs HttpJob: `wait` polls
// fal's status_url, MiniMax's query route, or <path>/{id}). Load balancer:
// {method, path, body}; a submit answers with the job's id at once (the
// load balancer holds a request ~150 s at most), polled with `poll`.
// The request bodies are the ones the e2e scripts proved
// (scripts/serve/e2e/ref2v.py t_minimax / t_fal_turbo, ltx_e2e.py, batch.py).
import type { Serves, ServedModel } from "./serves";

export const IMAGE_PLACEHOLDER = "{{image_url}}";
export const AUDIO_PLACEHOLDER = "{{audio_url}}";

export interface InvokeExample {
  id: string;
  label: string;
  api: string;
  model: string | null;
  task: string | null;
  /** Inputs the request needs filled ({{image_url}}, {{audio_url}} in the body). */
  needs: ("image_url" | "audio_url")[];
  /** What the invoke route takes: queue → {input}, lb → {method, path, body}. */
  invoke: Record<string, unknown>;
  /** lb: where the created job is polled ({id}: the id in the reply). */
  poll?: string;
}

const PROMPT = "A red fox trots through fresh snow at dawn, low sun, gentle camera push in";
const REF_PROMPT = "Picture 1 walks along a beach at golden hour, waves rolling in, cinematic tracking shot";
const I2V_PROMPT = "The scene comes alive: soft wind, slow camera push in";
const A2V_PROMPT = "A narrator speaks to the camera in a quiet studio";
const TIER_WORD: Record<string, string> = { max: "Max", turbo: "Turbo", draft: "Draft" };

interface Req {
  api: string;
  apiLabel: string;
  method: "GET" | "POST";
  path: string;
  body?: unknown;
  poll?: string;
  detail: string;
}

/** The requests for one model and task, per API the endpoint mounts. */
function requestsFor(s: Serves, m: ServedModel, task: string): Req[] {
  const out: Req[] = [];
  const on = (p: string) => s.protocols.includes(p);
  const app = (a: string) => s.fal_apps.includes(a);
  const tier = m.tier;
  if (m.family === "h3") {
    const res = tier === "draft" ? "480P" : "768P";
    // MiniMax V2: a name that resolves to this model (a config alias or the tier's MiniMax name).
    // A ref2v companion is also reached through the base model of its tier (fastvideo-protocol route_task).
    const reaches = (target: string) => {
      if (target === m.id) return true;
      const t = s.models.find((x) => x.id === target);
      return task === "ref2v" && !!tier && t?.family === "h3" && t.tier === tier && !t.tasks.includes("ref2v");
    };
    const names = s.aliases.filter((a) => a.name.startsWith("MiniMax-") && reaches(a.model)).map((a) => a.name);
    const mm = names.find((n) => n !== "MiniMax-H3") ?? names[0];
    if (on("minimax") && mm && ["t2v", "i2v", "ref2v"].includes(task)) {
      const content: unknown[] = [{ type: "text", text: task === "ref2v" ? REF_PROMPT : task === "i2v" ? I2V_PROMPT : PROMPT }];
      if (task === "i2v") content.push({ type: "image_url", image_url: { url: IMAGE_PLACEHOLDER }, role: "first_frame" });
      if (task === "ref2v") content.push({ type: "image_url", image_url: { url: IMAGE_PLACEHOLDER }, role: "reference_image" });
      out.push({ api: "minimax", apiLabel: "MiniMax V2", method: "POST", path: "/v2/video_generation", body: { model: mm, resolution: res, duration: 5, content }, poll: "/v2/query/video_generation/{id}", detail: `${mm}, ${res} 5 s` });
    }
    const falApp = tier ? `minimax/h3-${tier}` : null;
    if (on("fal") && falApp && app(falApp) && ["t2v", "i2v", "ref2v"].includes(task)) {
      const sub = task === "t2v" ? "text-to-video" : task === "i2v" ? "image-to-video" : "reference-to-video";
      const body: Record<string, unknown> = { prompt: task === "ref2v" ? REF_PROMPT.replace("Picture 1", "Image 1") : task === "i2v" ? I2V_PROMPT : PROMPT, duration: 5, resolution: res, seed: 7 };
      if (task === "i2v") body.image_url = IMAGE_PLACEHOLDER;
      if (task === "ref2v") body.reference_image_urls = [IMAGE_PLACEHOLDER];
      out.push({ api: "fal", apiLabel: `fal ${falApp}`, method: "POST", path: `/${falApp}/${sub}`, body, poll: `/${falApp}/requests/{id}/status`, detail: `${res} 5 s` });
    }
  } else if (m.family === "ltx2") {
    if (on("fal")) {
      const speed = tier === "max" ? "pro" : "fast";
      if (task === "t2v" && app("lightricks/ltx-2.5")) out.push({ api: "fal", apiLabel: "fal lightricks/ltx-2.5", method: "POST", path: `/lightricks/ltx-2.5/text-to-video/${speed}`, body: { prompt: PROMPT, seed: 3 }, poll: "/lightricks/ltx-2.5/requests/{id}/status", detail: speed });
      else if (task === "t2v" && tier === "turbo" && app("fastvideo/ltx-turbo")) out.push({ api: "fal", apiLabel: "fal fastvideo/ltx-turbo", method: "POST", path: "/fastvideo/ltx-turbo/text-to-video", body: { prompt: PROMPT, seed: 3 }, poll: "/fastvideo/ltx-turbo/requests/{id}/status", detail: "turbo" });
      if (task === "i2v" && app("lightricks/ltx-2.5")) out.push({ api: "fal", apiLabel: "fal lightricks/ltx-2.5", method: "POST", path: `/lightricks/ltx-2.5/image-to-video/${speed}`, body: { prompt: I2V_PROMPT, image_url: IMAGE_PLACEHOLDER, seed: 3 }, poll: "/lightricks/ltx-2.5/requests/{id}/status", detail: speed });
      if (task === "a2v" && app("lightricks/ltx-2.5")) out.push({ api: "fal", apiLabel: "fal lightricks/ltx-2.5", method: "POST", path: `/lightricks/ltx-2.5/audio-to-video/${speed}`, body: { prompt: A2V_PROMPT, audio_url: AUDIO_PLACEHOLDER }, poll: "/lightricks/ltx-2.5/requests/{id}/status", detail: speed });
      if (task === "ref2v" && app("fal-ai/ltx-2.3-quality")) out.push({ api: "fal", apiLabel: "fal fal-ai/ltx-2.3-quality", method: "POST", path: "/fal-ai/ltx-2.3-quality/ingredient", body: { prompt: REF_PROMPT, image_url: IMAGE_PLACEHOLDER, seed: 1024 }, poll: "/fal-ai/ltx-2.3-quality/requests/{id}/status", detail: "ingredient (reference sheet)" });
    }
  } else if (m.family === "wan" && on("fal") && app("fal-ai/wan") && (m.recipe.startsWith("wan-") || /ti2v-5b/.test(m.recipe))) {
    const sub = task === "i2v" ? "v2.2-5b/image-to-video" : tier === "turbo" ? "v2.2-5b/text-to-video/fast-wan" : "v2.2-5b/text-to-video";
    if (task === "t2v" || task === "i2v") out.push({ api: "fal", apiLabel: "fal fal-ai/wan", method: "POST", path: `/fal-ai/wan/${sub}`, body: { prompt: task === "i2v" ? I2V_PROMPT : PROMPT, ...(task === "i2v" ? { image_url: IMAGE_PLACEHOLDER } : {}), seed: 3 }, poll: "/fal-ai/wan/requests/{id}/status", detail: sub });
  }
  if (task === "t2v" && on("native")) out.push({ api: "native", apiLabel: "native", method: "POST", path: "/fv/v1/jobs", body: { model: m.id, prompt: PROMPT, seed: 1 }, poll: "/fv/v1/jobs/{id}", detail: "defaults" });
  if (task === "t2v" && on("openai_videos") && m.family !== "h3") out.push({ api: "openai_videos", apiLabel: "OpenAI videos", method: "POST", path: "/v1/videos", body: { model: m.id, prompt: PROMPT, seconds: 5 }, poll: "/v1/videos/{id}", detail: "5 s" });
  return out;
}

/** Every example for an endpoint of `mode` serving `s`, the plain ones (info / ping, capabilities) first. */
export function examplesFor(s: Serves, mode: "queue" | "lb"): InvokeExample[] {
  const ex: InvokeExample[] = [];
  const wrap = (r: Pick<Req, "method" | "path" | "body">): Record<string, unknown> =>
    mode === "lb" ? { method: r.method, path: r.path, ...(r.body !== undefined ? { body: r.body } : {}) } : { input: { kind: "http", method: r.method, path: r.path, ...(r.body !== undefined ? { body: r.body } : {}), ...(r.method === "POST" ? { wait: true } : {}) } };
  if (mode === "queue") ex.push({ id: "info", label: "Worker info (kind info)", api: "runpod", model: null, task: null, needs: [], invoke: { input: { kind: "info" } } });
  else ex.push({ id: "ping", label: "Health (GET /ping)", api: "http", model: null, task: null, needs: [], invoke: { method: "GET", path: "/ping" } });
  ex.push({ id: "capabilities", label: "Capabilities (GET /fv/v1/capabilities)", api: "native", model: null, task: null, needs: [], invoke: wrap({ method: "GET", path: "/fv/v1/capabilities" }) });
  const seen = new Set<string>();
  for (const m of s.models) {
    for (const task of m.tasks) {
      for (const r of requestsFor(s, m, task)) {
        const id = `${r.api}:${m.id}:${task}`;
        if (seen.has(id)) continue;
        seen.add(id);
        const body = JSON.stringify(r.body ?? "");
        const needs = [...(body.includes(IMAGE_PLACEHOLDER) ? ["image_url" as const] : []), ...(body.includes(AUDIO_PLACEHOLDER) ? ["audio_url" as const] : [])];
        const tierWord = m.tier ? `${TIER_WORD[m.tier]} ` : "";
        ex.push({ id, label: `${r.apiLabel} · ${task} · ${tierWord}(${m.id}) · ${r.detail}`, api: r.api, model: m.id, task, needs, invoke: wrap(r), ...(mode === "lb" && r.poll ? { poll: r.poll } : {}) });
      }
    }
  }
  return ex;
}

/** Fills an example's placeholders (a reference image or audio URL the worker can fetch). */
export function fillExample(invoke: unknown, values: { image_url?: string; audio_url?: string }): unknown {
  const s = JSON.stringify(invoke)
    .split(JSON.stringify(IMAGE_PLACEHOLDER).slice(1, -1))
    .join(JSON.stringify(values.image_url ?? "").slice(1, -1))
    .split(JSON.stringify(AUDIO_PLACEHOLDER).slice(1, -1))
    .join(JSON.stringify(values.audio_url ?? "").slice(1, -1));
  return JSON.parse(s);
}
