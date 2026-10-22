// D1 persistence for boards, panels, shot settings and renders
// (migrations/0001_init.sql).
import type { AspectRatio, Board, EngineId, Panel, Render, ShotMode, ShotSettings, Tier } from "./types";

export function newId(prefix: string): string {
  const b = new Uint8Array(12);
  crypto.getRandomValues(b);
  return `${prefix}_${[...b].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

export class StoreError extends Error {
  constructor(
    readonly status: 400 | 404 | 409,
    message: string,
  ) {
    super(message);
    this.name = "StoreError";
  }
}

type Row = Record<string, unknown>;

const toShot = (r: Row): ShotSettings => ({
  board_id: String(r.board_id),
  start_panel_id: String(r.start_panel_id),
  prompt: String(r.prompt),
  seconds: Number(r.seconds),
  mode: r.mode as ShotMode,
  seed: r.seed === null ? null : Number(r.seed),
  reference_panel_ids: parseIds(r.reference_panel_ids),
  updated_at: Number(r.updated_at),
});

function parseIds(v: unknown): string[] {
  try {
    const a = JSON.parse(String(v ?? "[]"));
    return Array.isArray(a) ? a.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

const nullableNum = (v: unknown) => (v === null || v === undefined ? null : Number(v));
const nullableStr = (v: unknown) => (v === null || v === undefined ? null : String(v));

const toRender = (r: Row): Render => ({
  id: String(r.id),
  board_id: String(r.board_id),
  start_panel_id: String(r.start_panel_id),
  end_panel_id: String(r.end_panel_id),
  bakeoff_id: nullableStr(r.bakeoff_id),
  engine: r.engine as EngineId,
  tier: r.tier as Tier,
  model: String(r.model),
  mode: r.mode as ShotMode,
  prompt: String(r.prompt),
  seconds: Number(r.seconds),
  seed: Number(r.seed),
  job_id: nullableStr(r.job_id),
  status: r.status as Render["status"],
  progress: Number(r.progress),
  error: nullableStr(r.error),
  video_key: nullableStr(r.video_key),
  clip_seconds: nullableNum(r.clip_seconds),
  inference_time_s: nullableNum(r.inference_time_s),
  wall_time_s: nullableNum(r.wall_time_s),
  size: nullableStr(r.size),
  selected: Number(r.selected) === 1,
  created_at: Number(r.created_at),
  completed_at: nullableNum(r.completed_at),
});

export interface NewPanel {
  image_key: string;
  mime: string;
  bytes: number;
  caption: string;
}

export interface ShotInput {
  prompt: string;
  seconds: number;
  mode: ShotMode;
  seed: number | null;
  reference_panel_ids: string[];
}

export type NewRender = Pick<
  Render,
  "board_id" | "start_panel_id" | "end_panel_id" | "bakeoff_id" | "engine" | "tier" | "model" | "mode" | "prompt" | "seconds" | "seed"
>;

export type RenderPatch = Partial<
  Pick<
    Render,
    | "job_id"
    | "status"
    | "progress"
    | "error"
    | "video_key"
    | "clip_seconds"
    | "inference_time_s"
    | "wall_time_s"
    | "size"
    | "completed_at"
  >
>;

const RENDER_PATCH_COLUMNS = [
  "job_id",
  "status",
  "progress",
  "error",
  "video_key",
  "clip_seconds",
  "inference_time_s",
  "wall_time_s",
  "size",
  "completed_at",
] as const;

export class Store {
  constructor(
    private readonly db: D1Database,
    private readonly now: () => number = Date.now,
  ) {}

  // ---- boards ---------------------------------------------------------------

  async createBoard(input: { title: string; aspect_ratio: AspectRatio }): Promise<Board> {
    const t = this.now();
    const b: Board = { id: newId("bd"), title: input.title, aspect_ratio: input.aspect_ratio, created_at: t, updated_at: t };
    await this.db
      .prepare("INSERT INTO boards (id, title, aspect_ratio, created_at, updated_at) VALUES (?, ?, ?, ?, ?)")
      .bind(b.id, b.title, b.aspect_ratio, b.created_at, b.updated_at)
      .run();
    return b;
  }

  async listBoards(): Promise<Board[]> {
    const { results } = await this.db.prepare("SELECT * FROM boards ORDER BY created_at DESC, id DESC").all<Board>();
    return results;
  }

  async getBoard(id: string): Promise<Board | null> {
    return this.db.prepare("SELECT * FROM boards WHERE id = ?").bind(id).first<Board>();
  }

  async updateBoard(id: string, patch: { title?: string; aspect_ratio?: AspectRatio }): Promise<Board | null> {
    const cur = await this.getBoard(id);
    if (!cur) return null;
    const next = { ...cur, ...patch, updated_at: this.now() };
    await this.db
      .prepare("UPDATE boards SET title = ?, aspect_ratio = ?, updated_at = ? WHERE id = ?")
      .bind(next.title, next.aspect_ratio, next.updated_at, id)
      .run();
    return next;
  }

  private touch(boardId: string) {
    return this.db.prepare("UPDATE boards SET updated_at = ? WHERE id = ?").bind(this.now(), boardId);
  }

  // ---- panels ---------------------------------------------------------------

  async listPanels(boardId: string): Promise<Panel[]> {
    const { results } = await this.db
      .prepare("SELECT * FROM panels WHERE board_id = ? ORDER BY position ASC")
      .bind(boardId)
      .all<Panel>();
    return results;
  }

  async getPanel(boardId: string, panelId: string): Promise<Panel | null> {
    return this.db.prepare("SELECT * FROM panels WHERE board_id = ? AND id = ?").bind(boardId, panelId).first<Panel>();
  }

  async addPanel(boardId: string, p: NewPanel): Promise<Panel> {
    const next = await this.db
      .prepare("SELECT COALESCE(MAX(position) + 1, 0) AS n FROM panels WHERE board_id = ?")
      .bind(boardId)
      .first<number>("n");
    const panel: Panel = {
      id: newId("pn"),
      board_id: boardId,
      position: Number(next ?? 0),
      image_key: p.image_key,
      mime: p.mime,
      bytes: p.bytes,
      caption: p.caption,
      created_at: this.now(),
    };
    await this.db.batch([
      this.db
        .prepare(
          "INSERT INTO panels (id, board_id, position, image_key, mime, bytes, caption, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(panel.id, boardId, panel.position, panel.image_key, panel.mime, panel.bytes, panel.caption, panel.created_at),
      this.touch(boardId),
    ]);
    return panel;
  }

  async updatePanel(boardId: string, panelId: string, patch: { caption: string }): Promise<Panel | null> {
    const cur = await this.getPanel(boardId, panelId);
    if (!cur) return null;
    await this.db.batch([
      this.db.prepare("UPDATE panels SET caption = ? WHERE id = ?").bind(patch.caption, panelId),
      this.touch(boardId),
    ]);
    return { ...cur, caption: patch.caption };
  }

  /** Removes the panel and the shot settings that started on it; renders stay as history. */
  async deletePanel(boardId: string, panelId: string): Promise<Panel | null> {
    const cur = await this.getPanel(boardId, panelId);
    if (!cur) return null;
    const rest = (await this.listPanels(boardId)).filter((p) => p.id !== panelId);
    await this.db.batch([
      this.db.prepare("DELETE FROM shots WHERE board_id = ? AND start_panel_id = ?").bind(boardId, panelId),
      this.db.prepare("DELETE FROM panels WHERE id = ?").bind(panelId),
      ...rest.map((p, i) => this.db.prepare("UPDATE panels SET position = ? WHERE id = ?").bind(i, p.id)),
      this.touch(boardId),
    ]);
    return cur;
  }

  async reorderPanels(boardId: string, ids: readonly string[]): Promise<Panel[]> {
    const cur = await this.listPanels(boardId);
    const have = new Set(cur.map((p) => p.id));
    const given = new Set(ids);
    if (ids.length !== cur.length || given.size !== ids.length || ![...given].every((id) => have.has(id))) {
      throw new StoreError(400, "panel_ids must list every panel of the board exactly once");
    }
    await this.db.batch([
      ...ids.map((id, i) => this.db.prepare("UPDATE panels SET position = ? WHERE id = ?").bind(i, id)),
      this.touch(boardId),
    ]);
    return this.listPanels(boardId);
  }

  // ---- shot settings --------------------------------------------------------

  async listShots(boardId: string): Promise<ShotSettings[]> {
    const { results } = await this.db.prepare("SELECT * FROM shots WHERE board_id = ?").bind(boardId).all<Row>();
    return results.map(toShot);
  }

  async upsertShot(boardId: string, startPanelId: string, s: ShotInput): Promise<ShotSettings> {
    const t = this.now();
    await this.db.batch([
      this.db
        .prepare(
          `INSERT INTO shots (board_id, start_panel_id, prompt, seconds, mode, seed, reference_panel_ids, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (board_id, start_panel_id) DO UPDATE SET
             prompt = excluded.prompt, seconds = excluded.seconds, mode = excluded.mode,
             seed = excluded.seed, reference_panel_ids = excluded.reference_panel_ids, updated_at = excluded.updated_at`,
        )
        .bind(boardId, startPanelId, s.prompt, s.seconds, s.mode, s.seed, JSON.stringify(s.reference_panel_ids), t),
      this.touch(boardId),
    ]);
    return { board_id: boardId, start_panel_id: startPanelId, ...s, updated_at: t };
  }

  // ---- renders --------------------------------------------------------------

  async insertRender(r: NewRender): Promise<Render> {
    const render: Render = {
      ...r,
      id: newId("rn"),
      job_id: null,
      status: "queued",
      progress: 0,
      error: null,
      video_key: null,
      clip_seconds: null,
      inference_time_s: null,
      wall_time_s: null,
      size: null,
      selected: false,
      created_at: this.now(),
      completed_at: null,
    };
    await this.db
      .prepare(
        `INSERT INTO renders (id, board_id, start_panel_id, end_panel_id, bakeoff_id, engine, tier, model, mode,
           prompt, seconds, seed, status, progress, selected, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0, ?)`,
      )
      .bind(
        render.id,
        render.board_id,
        render.start_panel_id,
        render.end_panel_id,
        render.bakeoff_id,
        render.engine,
        render.tier,
        render.model,
        render.mode,
        render.prompt,
        render.seconds,
        render.seed,
        render.created_at,
      )
      .run();
    return render;
  }

  async getRender(id: string): Promise<Render | null> {
    const row = await this.db.prepare("SELECT * FROM renders WHERE id = ?").bind(id).first<Row>();
    return row ? toRender(row) : null;
  }

  async listRenders(boardId: string): Promise<Render[]> {
    const { results } = await this.db
      .prepare("SELECT * FROM renders WHERE board_id = ? ORDER BY created_at ASC, id ASC")
      .bind(boardId)
      .all<Row>();
    return results.map(toRender);
  }

  async updateRender(id: string, patch: RenderPatch): Promise<void> {
    const cols = RENDER_PATCH_COLUMNS.filter((c) => c in patch);
    if (cols.length === 0) return;
    await this.db
      .prepare(`UPDATE renders SET ${cols.map((c) => `${c} = ?`).join(", ")} WHERE id = ?`)
      .bind(...cols.map((c) => patch[c] ?? null), id)
      .run();
  }

  /** Makes `id` the shot's one selected render (the animatic's cut). */
  async selectRender(id: string): Promise<Render | null> {
    const r = await this.getRender(id);
    if (!r) return null;
    await this.db.batch([
      this.db
        .prepare("UPDATE renders SET selected = 0 WHERE board_id = ? AND start_panel_id = ?")
        .bind(r.board_id, r.start_panel_id),
      this.db.prepare("UPDATE renders SET selected = 1 WHERE id = ?").bind(id),
    ]);
    return { ...r, selected: true };
  }
}
