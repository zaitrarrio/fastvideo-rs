-- Tween: boards of ordered panels; shots are adjacent panel pairs
-- (docs/adr/0004-tween-storyboard-app.md). Times are unix milliseconds.

CREATE TABLE boards (
  id           TEXT PRIMARY KEY,
  title        TEXT NOT NULL,
  aspect_ratio TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  updated_at   INTEGER NOT NULL
);

CREATE TABLE panels (
  id         TEXT PRIMARY KEY,
  board_id   TEXT NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
  position   INTEGER NOT NULL,
  image_key  TEXT NOT NULL,
  mime       TEXT NOT NULL,
  bytes      INTEGER NOT NULL,
  caption    TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL
);
CREATE INDEX panels_board ON panels(board_id, position);

-- Settings of the shot that starts at start_panel_id. The end panel is
-- whichever panel follows it in the board's current order.
CREATE TABLE shots (
  board_id            TEXT NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
  start_panel_id      TEXT NOT NULL REFERENCES panels(id) ON DELETE CASCADE,
  prompt              TEXT NOT NULL DEFAULT '',
  seconds             INTEGER NOT NULL DEFAULT 6,
  mode                TEXT NOT NULL DEFAULT 'keyframes' CHECK (mode IN ('keyframes', 'reference')),
  seed                INTEGER,
  reference_panel_ids TEXT NOT NULL DEFAULT '[]',
  updated_at          INTEGER NOT NULL,
  PRIMARY KEY (board_id, start_panel_id)
);

-- One engine job. The panel ids, prompt, seconds and seed are a snapshot of
-- the shot at submit time, so history survives reordering and edits.
CREATE TABLE renders (
  id               TEXT PRIMARY KEY,
  board_id         TEXT NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
  start_panel_id   TEXT NOT NULL,
  end_panel_id     TEXT NOT NULL,
  bakeoff_id       TEXT,
  engine           TEXT NOT NULL,
  tier             TEXT NOT NULL CHECK (tier IN ('draft', 'final')),
  model            TEXT NOT NULL,
  mode             TEXT NOT NULL,
  prompt           TEXT NOT NULL,
  seconds          INTEGER NOT NULL,
  seed             INTEGER NOT NULL,
  job_id           TEXT,
  status           TEXT NOT NULL CHECK (status IN ('queued', 'in_progress', 'completed', 'failed')),
  progress         INTEGER NOT NULL DEFAULT 0,
  error            TEXT,
  video_key        TEXT,
  clip_seconds     REAL,
  inference_time_s REAL,
  wall_time_s      REAL,
  size             TEXT,
  selected         INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL,
  completed_at     INTEGER
);
CREATE INDEX renders_shot ON renders(board_id, start_panel_id, created_at);
CREATE INDEX renders_open ON renders(status) WHERE status IN ('queued', 'in_progress');
