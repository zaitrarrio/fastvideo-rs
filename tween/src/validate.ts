// Request validation. Everything from the browser is untrusted; the engine
// stays the authority on what a model can render (clip lengths, canvases).
import type { ShotInput } from "./store";
import { ASPECT_RATIOS, ENGINE_IDS, SHOT_DEFAULTS, type AspectRatio, type EngineId, type Tier } from "./types";

export type Parsed<T> = { ok: true; value: T } | { ok: false; error: string };

const ok = <T>(value: T): Parsed<T> => ({ ok: true, value });
const bad = <T>(error: string): Parsed<T> => ({ ok: false, error });
const obj = (v: unknown): Record<string, unknown> | null =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

export const LIMITS = {
  title_max: 120,
  prompt_max: 2000,
  caption_max: 280,
  seconds_min: 1,
  seconds_max: 30,
  seed_max: 2 ** 31 - 1,
  references_max: 8,
  image_max_bytes: 10 * 1024 * 1024,
} as const;

function title(v: unknown): Parsed<string> {
  const t = typeof v === "string" ? v.trim() : "";
  if (!t) return bad("title is required");
  if (t.length > LIMITS.title_max) return bad(`title is longer than ${LIMITS.title_max} characters`);
  return ok(t);
}

function aspect(v: unknown): Parsed<AspectRatio> {
  return (ASPECT_RATIOS as readonly unknown[]).includes(v)
    ? ok(v as AspectRatio)
    : bad(`aspect_ratio must be one of ${ASPECT_RATIOS.join(", ")}`);
}

export function parseBoardInput(v: unknown): Parsed<{ title: string; aspect_ratio: AspectRatio }> {
  const o = obj(v);
  if (!o) return bad("expected a JSON object");
  const t = title(o.title);
  if (!t.ok) return t;
  const a = aspect(o.aspect_ratio ?? "16:9");
  if (!a.ok) return a;
  return ok({ title: t.value, aspect_ratio: a.value });
}

export function parseBoardPatch(v: unknown): Parsed<{ title?: string; aspect_ratio?: AspectRatio }> {
  const o = obj(v);
  if (!o) return bad("expected a JSON object");
  const out: { title?: string; aspect_ratio?: AspectRatio } = {};
  if (o.title !== undefined) {
    const t = title(o.title);
    if (!t.ok) return t;
    out.title = t.value;
  }
  if (o.aspect_ratio !== undefined) {
    const a = aspect(o.aspect_ratio);
    if (!a.ok) return a;
    out.aspect_ratio = a.value;
  }
  if (Object.keys(out).length === 0) return bad("nothing to update: send title and/or aspect_ratio");
  return ok(out);
}

export function parseCaption(v: unknown): Parsed<string> {
  const o = obj(v);
  const c = o && typeof o.caption === "string" ? o.caption.trim() : null;
  if (c === null) return bad("caption is required");
  if (c.length > LIMITS.caption_max) return bad(`caption is longer than ${LIMITS.caption_max} characters`);
  return ok(c);
}

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

export function parseShotInput(v: unknown): Parsed<ShotInput> {
  const o = obj(v);
  if (!o) return bad("expected a JSON object");
  const prompt = o.prompt === undefined ? SHOT_DEFAULTS.prompt : o.prompt;
  if (typeof prompt !== "string") return bad("prompt must be a string");
  if (prompt.length > LIMITS.prompt_max) return bad(`prompt is longer than ${LIMITS.prompt_max} characters`);
  const seconds = o.seconds === undefined ? SHOT_DEFAULTS.seconds : o.seconds;
  if (!isInt(seconds) || seconds < LIMITS.seconds_min || seconds > LIMITS.seconds_max) {
    return bad(`seconds must be a whole number from ${LIMITS.seconds_min} to ${LIMITS.seconds_max}`);
  }
  const mode = o.mode === undefined ? SHOT_DEFAULTS.mode : o.mode;
  if (mode !== "keyframes" && mode !== "reference") return bad("mode must be keyframes or reference");
  const seed = o.seed === undefined ? SHOT_DEFAULTS.seed : o.seed;
  if (seed !== null && (!isInt(seed) || seed < 0 || seed > LIMITS.seed_max)) {
    return bad(`seed must be null or a whole number from 0 to ${LIMITS.seed_max}`);
  }
  const refs = o.reference_panel_ids === undefined ? [] : o.reference_panel_ids;
  if (!Array.isArray(refs) || !refs.every((r) => typeof r === "string" && r.length > 0 && r.length <= 64)) {
    return bad("reference_panel_ids must be a list of panel ids");
  }
  if (refs.length > LIMITS.references_max) return bad(`at most ${LIMITS.references_max} reference panels`);
  return ok({ prompt: prompt.trim(), seconds, mode, seed, reference_panel_ids: [...new Set(refs as string[])] });
}

export function parseRenderRequest(v: unknown): Parsed<{ engines: EngineId[]; tier: Tier; seed: number | null }> {
  const o = obj(v ?? {});
  if (!o) return bad("expected a JSON object");
  const engines = o.engines === undefined ? [...ENGINE_IDS] : o.engines;
  if (!Array.isArray(engines) || engines.length === 0) return bad("engines must be a non-empty list");
  if (!engines.every((e) => (ENGINE_IDS as readonly unknown[]).includes(e))) {
    return bad(`engines must be drawn from ${ENGINE_IDS.join(", ")}`);
  }
  const tier = o.tier === undefined ? "draft" : o.tier;
  if (tier !== "draft" && tier !== "final") return bad("tier must be draft or final");
  const seed = o.seed === undefined ? null : o.seed;
  if (seed !== null && (!isInt(seed) || seed < 0 || seed > LIMITS.seed_max)) {
    return bad(`seed must be null or a whole number from 0 to ${LIMITS.seed_max}`);
  }
  return ok({ engines: [...new Set(engines as EngineId[])], tier, seed });
}

/** Sniffs the image type from its bytes; the declared content type is not trusted. */
export function parseImageUpload(bytes: Uint8Array): Parsed<{ mime: string; bytes: number }> {
  if (bytes.length === 0) return bad("the image is empty");
  if (bytes.length > LIMITS.image_max_bytes) return bad(`images are limited to ${LIMITS.image_max_bytes / 1024 / 1024} MiB`);
  const at = (i: number) => bytes[i] ?? -1;
  const ascii = (from: number, s: string) => [...s].every((ch, i) => at(from + i) === ch.charCodeAt(0));
  let mime: string | null = null;
  if (at(0) === 0x89 && ascii(1, "PNG")) mime = "image/png";
  else if (at(0) === 0xff && at(1) === 0xd8 && at(2) === 0xff) mime = "image/jpeg";
  else if (ascii(0, "RIFF") && ascii(8, "WEBP")) mime = "image/webp";
  if (!mime) return bad("panels must be PNG, JPEG or WebP images");
  return ok({ mime, bytes: bytes.length });
}
