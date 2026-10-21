import { beforeEach, describe, expect, it } from "vitest";
import { Store } from "../../src/store";
import { testD1 } from "../support/d1";

let store: Store;
let now = 1_000;
const clock = () => now++;

beforeEach(() => {
  now = 1_000;
  store = new Store(testD1(), clock);
});

async function boardWith(n: number) {
  const b = await store.createBoard({ title: "Heist", aspect_ratio: "16:9" });
  const panels = [];
  for (let i = 0; i < n; i++) {
    panels.push(await store.addPanel(b.id, { image_key: `panels/${i}`, mime: "image/png", bytes: 10 + i, caption: `p${i}` }));
  }
  return { b, panels };
}

describe("boards", () => {
  it("creates, lists newest first, reads and renames", async () => {
    const a = await store.createBoard({ title: "A", aspect_ratio: "16:9" });
    const b = await store.createBoard({ title: "B", aspect_ratio: "9:16" });
    expect((await store.listBoards()).map((x) => x.id)).toEqual([b.id, a.id]);
    expect(a.id).toMatch(/^bd_[0-9a-f]{24}$/);
    const renamed = await store.updateBoard(a.id, { title: "A2" });
    expect(renamed).toMatchObject({ title: "A2", aspect_ratio: "16:9" });
    expect(renamed!.updated_at).toBeGreaterThan(a.updated_at);
    expect(await store.getBoard("bd_missing")).toBeNull();
  });
});

describe("panels", () => {
  it("appends at the end", async () => {
    const { panels } = await boardWith(3);
    expect(panels.map((p) => p.position)).toEqual([0, 1, 2]);
  });

  it("reorders by an exact permutation and refuses anything else", async () => {
    const { b, panels } = await boardWith(3);
    const ids = panels.map((p) => p.id);
    await store.reorderPanels(b.id, [ids[2]!, ids[0]!, ids[1]!]);
    expect((await store.listPanels(b.id)).map((p) => p.id)).toEqual([ids[2], ids[0], ids[1]]);
    await expect(store.reorderPanels(b.id, [ids[0]!, ids[1]!])).rejects.toThrow(/every panel/);
    await expect(store.reorderPanels(b.id, [ids[0]!, ids[0]!, ids[1]!])).rejects.toThrow(/every panel/);
  });

  it("deletes a panel, closes the gap and drops the shot settings that started on it", async () => {
    const { b, panels } = await boardWith(3);
    await store.upsertShot(b.id, panels[1]!.id, { prompt: "middle", seconds: 6, mode: "keyframes", seed: 3, reference_panel_ids: [] });
    await store.deletePanel(b.id, panels[1]!.id);
    const left = await store.listPanels(b.id);
    expect(left.map((p) => [p.id, p.position])).toEqual([
      [panels[0]!.id, 0],
      [panels[2]!.id, 1],
    ]);
    expect(await store.listShots(b.id)).toEqual([]);
  });

  it("updates a caption", async () => {
    const { b, panels } = await boardWith(1);
    await store.updatePanel(b.id, panels[0]!.id, { caption: "wide on the vault" });
    expect((await store.getPanel(b.id, panels[0]!.id))!.caption).toBe("wide on the vault");
  });
});

describe("shots", () => {
  it("upserts settings keyed by start panel with references round-tripped", async () => {
    const { b, panels } = await boardWith(2);
    await store.upsertShot(b.id, panels[0]!.id, { prompt: "one", seconds: 6, mode: "keyframes", seed: null, reference_panel_ids: [] });
    await store.upsertShot(b.id, panels[0]!.id, {
      prompt: "two",
      seconds: 8,
      mode: "reference",
      seed: 9,
      reference_panel_ids: [panels[1]!.id],
    });
    const shots = await store.listShots(b.id);
    expect(shots).toHaveLength(1);
    expect(shots[0]).toMatchObject({ prompt: "two", seconds: 8, mode: "reference", seed: 9, reference_panel_ids: [panels[1]!.id] });
  });
});

describe("renders", () => {
  const base = {
    end_panel_id: "",
    bakeoff_id: null,
    engine: "ltx" as const,
    tier: "draft" as const,
    model: "fake-ltx-turbo",
    mode: "keyframes" as const,
    prompt: "go",
    seconds: 6,
    seed: 1,
  };

  it("inserts queued, updates progress and terminal metrics", async () => {
    const { b, panels } = await boardWith(2);
    const r = await store.insertRender({ ...base, board_id: b.id, start_panel_id: panels[0]!.id, end_panel_id: panels[1]!.id });
    expect(r).toMatchObject({ status: "queued", progress: 0, selected: false, job_id: null });
    await store.updateRender(r.id, { job_id: "video_1", status: "in_progress", progress: 50 });
    await store.updateRender(r.id, {
      status: "completed",
      progress: 100,
      video_key: "clips/x.mp4",
      clip_seconds: 6,
      inference_time_s: 2,
      wall_time_s: 3,
      completed_at: 2_000,
    });
    const got = await store.getRender(r.id);
    expect(got).toMatchObject({ job_id: "video_1", status: "completed", clip_seconds: 6, inference_time_s: 2, completed_at: 2_000 });
  });

  it("selects one render per shot", async () => {
    const { b, panels } = await boardWith(2);
    const shot = { ...base, board_id: b.id, start_panel_id: panels[0]!.id, end_panel_id: panels[1]!.id };
    const one = await store.insertRender(shot);
    const two = await store.insertRender({ ...shot, engine: "h3", model: "fake-h3-turbo" });
    await store.selectRender(one.id);
    await store.selectRender(two.id);
    const all = await store.listRenders(b.id);
    expect(all.filter((r) => r.selected).map((r) => r.id)).toEqual([two.id]);
  });

  it("lists a board's renders oldest first and skips other boards", async () => {
    const a = await boardWith(2);
    const other = await boardWith(2);
    const r1 = await store.insertRender({ ...base, board_id: a.b.id, start_panel_id: a.panels[0]!.id, end_panel_id: a.panels[1]!.id });
    await store.insertRender({ ...base, board_id: other.b.id, start_panel_id: other.panels[0]!.id, end_panel_id: other.panels[1]!.id });
    const r2 = await store.insertRender({ ...base, board_id: a.b.id, start_panel_id: a.panels[0]!.id, end_panel_id: a.panels[1]!.id });
    expect((await store.listRenders(a.b.id)).map((r) => r.id)).toEqual([r1.id, r2.id]);
  });
});
