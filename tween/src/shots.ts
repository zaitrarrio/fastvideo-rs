// Shots are derived, never stored: shot i runs from panel i to panel i + 1,
// so consecutive shots share their boundary panel by construction
// (ADR-0004 decision 1). Stored settings follow their start panel.
import { SHOT_DEFAULTS, type Panel, type Shot, type ShotSettings } from "./types";

export function deriveShots(panels: readonly Panel[], settings: readonly ShotSettings[]): Shot[] {
  const ordered = [...panels].sort((a, b) => a.position - b.position);
  const onBoard = new Set(ordered.map((p) => p.id));
  const byStart = new Map(settings.map((s) => [s.start_panel_id, s]));
  const shots: Shot[] = [];
  for (let i = 0; i + 1 < ordered.length; i++) {
    const start = ordered[i]!;
    const end = ordered[i + 1]!;
    const s = byStart.get(start.id);
    shots.push({
      index: i,
      start_panel_id: start.id,
      end_panel_id: end.id,
      prompt: s?.prompt ?? SHOT_DEFAULTS.prompt,
      seconds: s?.seconds ?? SHOT_DEFAULTS.seconds,
      mode: s?.mode ?? SHOT_DEFAULTS.mode,
      seed: s?.seed ?? SHOT_DEFAULTS.seed,
      reference_panel_ids: (s?.reference_panel_ids ?? SHOT_DEFAULTS.reference_panel_ids).filter((id) =>
        onBoard.has(id),
      ),
    });
  }
  return shots;
}

/** The shot starting at `startPanelId`, or null when that panel is last or absent. */
export function shotAt(
  panels: readonly Panel[],
  settings: readonly ShotSettings[],
  startPanelId: string,
): Shot | null {
  return deriveShots(panels, settings).find((s) => s.start_panel_id === startPanelId) ?? null;
}
