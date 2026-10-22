// Real-time factor (ADR-0004 decision 8): clip seconds generated per second
// of compute. `inference` uses the engine's own `inference_time_s`; `wall`
// spans submit to completion, so queueing and weight swaps show up there.

export interface RenderTiming {
  clip_seconds: number | null;
  inference_time_s: number | null;
  wall_time_s: number | null;
}

export interface RealtimeFactors {
  inference: number | null;
  wall: number | null;
  /** By engine time when known, else by wall time; null when unknown. */
  faster_than_realtime: boolean | null;
}

const ratio = (clip: number | null, t: number | null) =>
  clip !== null && clip > 0 && t !== null && t > 0 ? Math.round((clip / t) * 100) / 100 : null;

export function realtimeFactors(t: RenderTiming): RealtimeFactors {
  const inference = ratio(t.clip_seconds, t.inference_time_s);
  const wall = ratio(t.clip_seconds, t.wall_time_s);
  const basis = inference ?? wall;
  return { inference, wall, faster_than_realtime: basis === null ? null : basis > 1 };
}
