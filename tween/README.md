# Tween

Keyframe storyboards rendered by fv-serve. Upload panels in order; every pair
of adjacent panels is a **shot**, and Tween asks a first-last-frame model to
fill in the motion between them. Each shot can render on **LTX-2.5** and
**MiniMax H3** side by side with the same seed, prompt and length, and each
take shows its **real-time factor** (clip seconds / engine generation time).
Pick the take you want, finalize it on the higher-quality tier with the same
seed, and play the board as an **animatic**.

Decision record: [ADR-0004](../docs/adr/0004-tween-storyboard-app.md),
decision log `FVID-2026-10-20-tween-storyboard`.

## Run it

All tasks run from the repository root (`task tween:<name>`; `task --list`).

| Goal | Command | Needs |
|---|---|---|
| First-time setup | `task tween:setup` | Node 22 |
| Typecheck, unit tests, Worker build | `task tween:check` | Node 22 |
| UI with playable stand-in clips | `task tween:up:mock` → http://127.0.0.1:8787 | Node 22, ffmpeg (optional) |
| UI against real fv-serve (fake engine) | `task tween:up` → http://127.0.0.1:8787 | Docker |
| e2e suite, mock engine | `task tween:e2e:mock` | Node 22 |
| e2e suite, real fv-serve | `task tween:e2e` | Docker |
| Stop the engine, drop its state | `task tween:down` | Docker |
| Reset local boards | `task tween:db:reset` | |

**Engines, from cheapest to real:**

- **Mock** (`scripts/mock-engine.ts`): the `/v1/videos` rules Tween relies on,
  on a timed clock, no Docker. With ffmpeg installed a finished clip is a
  crossfade from the first panel to the last, so the players and the animatic
  have something to play. It is not a model.
- **fv-serve, fake engine** (`docker-compose.yml`, the public
  `ghcr.io/zaitrarrio/fastvideo-rs-serve:cpu` image with `engine/fake.toml`):
  the real server and request validation (tasks, references, clip lengths,
  canvases); output is a placeholder file.
- **fv-serve on GPUs**: set `TWEEN_ENGINE_URL` to the edge or a worker
  (`configs/serve/runpod-family-ltx.toml`, `runpod-family-h3.toml`,
  `runpod-h3-ref2v.toml`) and the model ids below.

## Configuration

`.env` (from `.env.example`) feeds the Taskfile; `task tween:env` turns it
into `.dev.vars` for `wrangler dev`. Deployed values live in
`wrangler.toml` (vars) and `wrangler secret put` (secrets).

| Variable | Meaning | Default |
|---|---|---|
| `TWEEN_ENGINE_URL` | fv-serve base URL | none (renders off) |
| `TWEEN_ENGINE_KEY` (secret) | fv-serve API key, sent only server-side | empty |
| `TWEEN_ACCESS_TOKEN` (secret) | studio sign-in token; unset means open (local only) | unset |
| `TWEEN_LTX_DRAFT_MODEL` / `_FINAL_MODEL` | LTX ids per tier | `fake-ltx-turbo` / `fake-ltx-pro` |
| `TWEEN_H3_DRAFT_MODEL` / `_FINAL_MODEL` | H3 ids per tier | `fake-h3-turbo` / `fake-h3-max` |
| `TWEEN_H3_REFERENCE_MODEL` | H3 id for reference shots; empty disables them | `fake-h3-max` |

Staging (`wrangler.toml [env.staging]`) uses `ltx25-distill-sol` /
`ltx25-distill-dense`, `fasth3` / `sol-h3` and `h3-ref2v-turbo`.

## Shots

- **First + last** (`keyframes`): the two panels go to `/v1/videos` as two
  `image_reference` entries, which fv-serve resolves to `Task::Keyframes` on
  both families. LTX-2.5 and every H3 tier support it; LTX-2.3 does not
  (`GapId::LtxKeyframes`).
- **Reference**: `task: "ref2va"` with both panels plus any extra reference
  panels. **H3 only**: `/v1/videos` accepts `task` on H3 models only, so a
  bake-off in reference mode is refused before anything is sent.
- **A prompt is required to render.** fv-serve refuses an empty or
  whitespace-only prompt on every model, so Tween refuses the render first
  (400) rather than sending one.
- **Lengths** are whole seconds. The UI shows each engine's range (LTX 6–20 s,
  H3 5–15 s) but the engine decides: a refusal is recorded on that engine's
  take and the other engine keeps going.
- Mid-shot keyframes are out of scope (ADR-0004 consequences).

## API

All `/api/*` routes need the session cookie when `TWEEN_ACCESS_TOKEN` is set.
Errors are `{"error": "..."}` with a 4xx/5xx status.

| Route | Does |
|---|---|
| `GET /healthz` | `ok` |
| `GET/POST/DELETE /api/session` | sign-in state / sign in with `{token}` / sign out |
| `GET /api/config` | engines, model ids, limits, whether an engine is configured |
| `GET /api/boards`, `POST /api/boards` | list / create `{title, aspect_ratio}` |
| `GET /api/boards/:board` | board, panels, derived shots, renders |
| `PATCH /api/boards/:board` | `{title?, aspect_ratio?}` |
| `POST /api/boards/:board/panels` | multipart `image` (+ `caption`) or a raw PNG/JPEG/WebP body (≤ 10 MiB, sniffed) |
| `GET /api/boards/:board/panels/:panel/image` | the panel image |
| `PATCH`, `DELETE /api/boards/:board/panels/:panel` | caption / remove |
| `PUT /api/boards/:board/panel-order` | `{panel_ids}`, every panel exactly once |
| `PUT /api/boards/:board/shots/:start` | shot settings `{prompt, seconds, mode, seed, reference_panel_ids}` |
| `POST /api/boards/:board/shots/:start/renders` | `{engines?, tier?, seed?}`; default: draft bake-off on both; 400 if the shot has no prompt |
| `GET /api/renders/:render` | the render, pulling the engine job while it runs |
| `GET /api/renders/:render/video` | the MP4 (byte ranges supported) |
| `POST /api/renders/:render/select` | make it the shot's take in the cut |
| `POST /api/renders/:render/finalize` | re-render at the final tier with the same seed |
| `GET /api/boards/:board/animatic?source=cut\|ltx\|h3` | the playlist: one entry per shot, `render: null` for gaps |

## Layout

```
src/          Worker: app.ts (routes), renders.ts (submit/poll), engine-client.ts
              (/v1/videos), store.ts (D1), shots.ts, animatic.ts, metrics.ts,
              validate.ts, auth.ts, media.ts (R2)
public/       UI: index.html, app.css, app.js (no build step)
migrations/   D1 schema
engine/       fv-serve config for the Docker fake engine
scripts/      mock-engine.ts, with-mock.mjs, wait-http.mjs, dev-vars.mjs, env-get.mjs
test/unit     routes, store (real SQL on node:sqlite), client, shots, ...
test/e2e      Tween against a live engine (TWEEN_E2E_ENGINE_URL)
```

## Deploying to staging

1. `task tween:provision:staging`, then paste the D1 id into
   `wrangler.toml` `[[env.staging.d1_databases]]`.
2. Set `TWEEN_ENGINE_URL` in `[env.staging.vars]`.
3. `wrangler secret put TWEEN_ENGINE_KEY --env staging` and
   `wrangler secret put TWEEN_ACCESS_TOKEN --env staging`.
4. `task tween:deploy:staging` (check, migrate, deploy).
