import { describe, expect, it } from "vitest";
import { pickAnimatic } from "../../src/animatic";
import type { Render, Shot } from "../../src/types";

const shot = (i: number, start: string, end: string): Shot => ({
  index: i,
  start_panel_id: start,
  end_panel_id: end,
  prompt: "",
  seconds: 6,
  mode: "keyframes",
  seed: null,
  reference_panel_ids: [],
});

let t = 0;
const render = (over: Partial<Render>): Render => ({
  id: `rn_${t}`,
  board_id: "b",
  start_panel_id: "a",
  end_panel_id: "b",
  bakeoff_id: null,
  engine: "ltx",
  tier: "draft",
  model: "m",
  mode: "keyframes",
  prompt: "",
  seconds: 6,
  seed: 1,
  job_id: "j",
  status: "completed",
  progress: 100,
  error: null,
  video_key: "clips/x.mp4",
  clip_seconds: 6,
  inference_time_s: 2,
  wall_time_s: 3,
  size: "1280x768",
  selected: false,
  created_at: t++,
  completed_at: t,
  ...over,
});

const shots = [shot(0, "a", "b"), shot(1, "b", "c")];

describe("pickAnimatic", () => {
  it("leaves a gap for a shot with no finished render", () => {
    const out = pickAnimatic(shots, [render({ id: "r1" })], "cut");
    expect(out.map((e) => e.render?.id ?? null)).toEqual(["r1", null]);
  });

  it("prefers the selected render for the cut, whatever its engine", () => {
    const out = pickAnimatic(
      shots,
      [render({ id: "final", tier: "final" }), render({ id: "pick", engine: "h3", selected: true })],
      "cut",
    );
    expect(out[0]!.render!.id).toBe("pick");
  });

  it("prefers final over draft, then the newest", () => {
    const out = pickAnimatic(
      shots,
      [render({ id: "old-final", tier: "final" }), render({ id: "new-draft" }), render({ id: "new-final", tier: "final" })],
      "cut",
    );
    expect(out[0]!.render!.id).toBe("new-final");
  });

  it("filters by engine for a per-engine reel", () => {
    const renders = [render({ id: "l", engine: "ltx" }), render({ id: "h", engine: "h3" }), render({ id: "sel", engine: "ltx", selected: true })];
    expect(pickAnimatic(shots, renders, "h3")[0]!.render!.id).toBe("h");
    expect(pickAnimatic(shots, renders, "ltx")[0]!.render!.id).toBe("sel");
  });

  it("ignores failed renders, renders with no video and renders of a different panel pair", () => {
    const out = pickAnimatic(
      shots,
      [
        render({ id: "failed", status: "failed", video_key: null }),
        render({ id: "novideo", video_key: null }),
        render({ id: "stale", end_panel_id: "c" }),
      ],
      "cut",
    );
    expect(out[0]!.render).toBeNull();
  });
});
