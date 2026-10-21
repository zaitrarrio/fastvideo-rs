// Which render plays for each shot. `cut` is the edit: the shot's selected
// render, else the best finished one from either engine. `ltx` / `h3` are
// per-engine reels for comparing whole boards side by side.
import type { EngineId, Render, Shot } from "./types";

export type AnimaticSource = "cut" | EngineId;

export interface AnimaticEntry {
  shot: Shot;
  render: Render | null;
}

const playable = (r: Render, s: Shot) =>
  r.status === "completed" &&
  r.video_key !== null &&
  r.start_panel_id === s.start_panel_id &&
  r.end_panel_id === s.end_panel_id;

const tierRank = (r: Render) => (r.tier === "final" ? 1 : 0);

export function pickAnimatic(shots: readonly Shot[], renders: readonly Render[], source: AnimaticSource): AnimaticEntry[] {
  return shots.map((shot) => {
    const candidates = renders.filter((r) => playable(r, shot) && (source === "cut" || r.engine === source));
    const selected = candidates.find((r) => r.selected);
    if (selected) return { shot, render: selected };
    const best = [...candidates].sort((a, b) => tierRank(b) - tierRank(a) || b.created_at - a.created_at)[0];
    return { shot, render: best ?? null };
  });
}
