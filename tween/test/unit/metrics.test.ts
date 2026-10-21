import { describe, expect, it } from "vitest";
import { realtimeFactors } from "../../src/metrics";

describe("realtimeFactors", () => {
  it("divides clip seconds by engine inference time and by wall time", () => {
    expect(realtimeFactors({ clip_seconds: 6, inference_time_s: 2, wall_time_s: 3 })).toEqual({
      inference: 3,
      wall: 2,
      faster_than_realtime: true,
    });
  });

  it("is slower than real time below 1x", () => {
    const f = realtimeFactors({ clip_seconds: 5, inference_time_s: 10, wall_time_s: 12 });
    expect(f.inference).toBe(0.5);
    expect(f.faster_than_realtime).toBe(false);
  });

  it("falls back to wall time when the engine reports no inference time", () => {
    const f = realtimeFactors({ clip_seconds: 6, inference_time_s: null, wall_time_s: 4 });
    expect(f.inference).toBeNull();
    expect(f.wall).toBe(1.5);
    expect(f.faster_than_realtime).toBe(true);
  });

  it("is unknown without a clip length or with zero time", () => {
    expect(realtimeFactors({ clip_seconds: null, inference_time_s: 1, wall_time_s: 1 })).toEqual({
      inference: null,
      wall: null,
      faster_than_realtime: null,
    });
    expect(realtimeFactors({ clip_seconds: 6, inference_time_s: 0, wall_time_s: 0 }).inference).toBeNull();
  });
});
