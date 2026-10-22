// Tween domain types (docs/adr/0004-tween-storyboard-app.md).

export type EngineId = "ltx" | "h3";
export const ENGINE_IDS: readonly EngineId[] = ["ltx", "h3"];

export type ShotMode = "keyframes" | "reference";
export type Tier = "draft" | "final";

/** The engine job status vocabulary of `/v1/videos` (`status_name`). */
export type RenderStatus = "queued" | "in_progress" | "completed" | "failed";
export const TERMINAL: ReadonlySet<RenderStatus> = new Set(["completed", "failed"]);

export const ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] as const;
export type AspectRatio = (typeof ASPECT_RATIOS)[number];

export interface Board {
  id: string;
  title: string;
  aspect_ratio: AspectRatio;
  created_at: number;
  updated_at: number;
}

export interface Panel {
  id: string;
  board_id: string;
  position: number;
  image_key: string;
  mime: string;
  bytes: number;
  caption: string;
  created_at: number;
}

/** Stored settings of the shot starting at `start_panel_id`. */
export interface ShotSettings {
  board_id: string;
  start_panel_id: string;
  prompt: string;
  seconds: number;
  mode: ShotMode;
  seed: number | null;
  reference_panel_ids: string[];
  updated_at: number;
}

/** A derived shot: two adjacent panels plus their settings. */
export interface Shot {
  index: number;
  start_panel_id: string;
  end_panel_id: string;
  prompt: string;
  seconds: number;
  mode: ShotMode;
  seed: number | null;
  reference_panel_ids: string[];
}

export interface Render {
  id: string;
  board_id: string;
  start_panel_id: string;
  end_panel_id: string;
  bakeoff_id: string | null;
  engine: EngineId;
  tier: Tier;
  model: string;
  mode: ShotMode;
  prompt: string;
  seconds: number;
  seed: number;
  job_id: string | null;
  status: RenderStatus;
  progress: number;
  error: string | null;
  video_key: string | null;
  clip_seconds: number | null;
  inference_time_s: number | null;
  wall_time_s: number | null;
  size: string | null;
  selected: boolean;
  created_at: number;
  completed_at: number | null;
}

/** Defaults for a shot with no stored settings. */
export const SHOT_DEFAULTS = {
  prompt: "",
  seconds: 6,
  mode: "keyframes" as ShotMode,
  seed: null as number | null,
  reference_panel_ids: [] as string[],
};
