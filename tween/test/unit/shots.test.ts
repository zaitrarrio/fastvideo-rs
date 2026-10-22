import { describe, expect, it } from "vitest";
import { deriveShots, shotAt } from "../../src/shots";
import type { Panel, ShotSettings } from "../../src/types";

const panel = (id: string, position: number): Panel => ({
  id,
  board_id: "b",
  position,
  image_key: `panels/${id}`,
  mime: "image/png",
  bytes: 1,
  caption: "",
  created_at: 0,
});

const settings = (start: string, over: Partial<ShotSettings> = {}): ShotSettings => ({
  board_id: "b",
  start_panel_id: start,
  prompt: `from ${start}`,
  seconds: 8,
  mode: "keyframes",
  seed: 7,
  reference_panel_ids: [],
  updated_at: 0,
  ...over,
});

describe("deriveShots", () => {
  it("has no shots for fewer than two panels", () => {
    expect(deriveShots([], [])).toEqual([]);
    expect(deriveShots([panel("a", 0)], [])).toEqual([]);
  });

  it("pairs adjacent panels so consecutive shots share a boundary panel", () => {
    const shots = deriveShots([panel("a", 0), panel("b", 1), panel("c", 2)], []);
    expect(shots.map((s) => [s.start_panel_id, s.end_panel_id])).toEqual([
      ["a", "b"],
      ["b", "c"],
    ]);
    expect(shots[0]!.end_panel_id).toBe(shots[1]!.start_panel_id);
    expect(shots.map((s) => s.index)).toEqual([0, 1]);
  });

  it("orders by position, not by input order", () => {
    const shots = deriveShots([panel("c", 2), panel("a", 0), panel("b", 1)], []);
    expect(shots.map((s) => s.start_panel_id)).toEqual(["a", "b"]);
  });

  it("applies stored settings keyed by the start panel and defaults the rest", () => {
    const shots = deriveShots([panel("a", 0), panel("b", 1), panel("c", 2)], [settings("b")]);
    expect(shots[0]).toMatchObject({ prompt: "", seconds: 6, mode: "keyframes", seed: null });
    expect(shots[1]).toMatchObject({ prompt: "from b", seconds: 8, seed: 7 });
  });

  it("re-derives end panels after a reorder while keeping settings with their start panel", () => {
    const reordered = [panel("b", 0), panel("a", 1), panel("c", 2)];
    const shots = deriveShots(reordered, [settings("a")]);
    expect(shots.map((s) => [s.start_panel_id, s.end_panel_id])).toEqual([
      ["b", "a"],
      ["a", "c"],
    ]);
    expect(shots[1]!.prompt).toBe("from a");
  });

  it("ignores settings whose start panel is now the last panel", () => {
    const shots = deriveShots([panel("a", 0), panel("b", 1)], [settings("b")]);
    expect(shots).toHaveLength(1);
    expect(shots[0]!.prompt).toBe("");
  });

  it("drops reference ids that are no longer on the board", () => {
    const shots = deriveShots(
      [panel("a", 0), panel("b", 1)],
      [settings("a", { mode: "reference", reference_panel_ids: ["b", "gone"] })],
    );
    expect(shots[0]!.reference_panel_ids).toEqual(["b"]);
  });
});

describe("shotAt", () => {
  it("finds the shot by its start panel", () => {
    const panels = [panel("a", 0), panel("b", 1)];
    expect(shotAt(panels, [], "a")?.end_panel_id).toBe("b");
    expect(shotAt(panels, [], "b")).toBeNull();
    expect(shotAt(panels, [], "zzz")).toBeNull();
  });
});
