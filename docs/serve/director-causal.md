# The director on causal streaming models (LongLive-1.3B, SF-Wan)

Status: implemented on `wip/director-causal`, **default off**. Before this,
`director/service.rs::caps_for` refused every model whose stream is not a
clip queue (`StreamCaps::Clip`), so the director could not use the SF-Wan
causal engine. LongLive's prompt switch with KV re-cache was reachable only
from the Reactor (`set_prompt`). See docs/serve/research-longlive.md §5–§11.

## 1. Shape

A causal director session is **one continuous causal rollout**. It is not a
chain of clips anchored on their last frames.

| | Clip models (H3, LTX) | Causal models (LongLive, SF-Wan) |
|---|---|---|
| Engine session | `ClipSession`, one `build` per chunk | `CausalSession` (exclusive executor lease), one rollout for the whole session |
| Continuity | chunk N+1 starts from chunk N's last frame (`AnchorLastFrame`) | the KV cache (window + frame sink) |
| Unit of generation | a chunk of 5–15 s | a block of 12 frames = 0.75 s at 16 fps |
| `chunk` message | one per built clip | one per **director chunk** of `causal_chunk_blocks` blocks (default 4 = 48 frames = 3 s): frames since the last `chunk`, generation time of those blocks, pace, buffer depth |
| Playout | a chunk is queued when built | each block is queued **as it arrives** (freeze-and-silence on underrun as before; `deadline_missed` names the director chunk of the late block) |
| Prompt update | the next undispatched chunk | `set_prompt` on the engine, applied at the **next block boundary**; LongLive re-caches the KV window once per switch (several updates before a block re-cache once, with the latest prompt) |

Seam: `director::engine::DirectorStream` (`set_prompt`, `set_seed`,
`set_paused`, `next_block`, `close`), opened by
`DirectorEngine::open_stream`. fv-serve implements it over
`EngineService::open_causal_session` + `CausalControl`. The tests implement it
over the fake engine. The clip path (`DirectorClips`) is unchanged.

**Pacing.** The engine runs ahead of real time (LongLive about 20 fps on an
RTX PRO 6000 at 16 fps playout). The session keeps about
`causal_lead_seconds` (2 s) of video queued for playout. Above that it pauses
the rollout at the next block boundary (`set_paused(true)`), and below it
resumes. A prompt therefore reaches the picture within about one block of
generation plus the lead, and not behind a deep queue of pre-generated blocks.

## 2. Control mapping

- `configure` opens the rollout. With `seed`: `set_seed` + reset before the
  first block. Then `set_prompt(premise)`. Generation starts with the first
  prompt.
- `prompt` (`replan: true`, the default): `prompt_pending`, the deck is
  replaced, and the new direction goes to the engine **now**, so the next
  block the engine starts carries it. `prompt_applied` is sent when the first
  block generated with it arrives. Its version is the engine's prompt
  version, mapped back to the client's `prompt_version`.
- `prompt` with `replan: false` appends: each queued direction holds one
  director chunk (`causal_chunk_blocks` blocks) before the next one takes
  over. The deck bound (`prompt_deck_size`) is unchanged.
- Scripts: text beats only. A beat at offset `t` s switches at block
  `round(t / 0.75)` after the script's start, with a tolerance of about one
  block.
- `stop`, heartbeats, `configure_timeout`, `max_session_seconds` and
  `session_metrics` are unchanged.

## 3. What a causal model refuses (precheck, clear errors)

| Input | Answer |
|---|---|
| `configure.image_url` | `error{code:"invalid_initial_image"}` (session failure): text-to-video only, no image conditioning |
| `configure.end_image_url`, end-image script beats | `invalid_initial_image` / `invalid_initial_script` |
| `configure.audio_url`, audio beats | `invalid_initial_audio` / `invalid_initial_script` (as today) |
| `resolution` other than the model's canvas tier | `invalid_input` (LongLive: `480p` only, 832×480) |
| `aspect_ratio` other than 16:9 | `invalid_input` |
| `prompt.end_image_url` | `prompt_rejected{reason:"invalid_image"}` |
| `prompt.audio_url` | `prompt_rejected{reason:"invalid_audio"}` (as today) |

## 4. `session_info` on a causal model

- `fps: 16`, `resolutions: ["480p"]`, `aspect_ratios: ["16:9"]`.
- `chunk_seconds`, `default_chunk_duration`, `min_chunk_duration` and
  `max_chunk_duration` are the director chunk, 3 s.
- `continuation_context_frames` is the KV window in pixel frames (latent
  frames × 4: 48 for LongLive's 12, 84 for SF-Wan's 21).
- `script_max_end_images: 0`.
- Extension object `causal`: `block_frames`, `block_seconds`,
  `chunk_blocks`, `kv_window_latent_frames`, `sink_latent_frames`,
  `prompt_switch` (`recache` | `keep`).
- No audio track. The offered audio m-line is answered `inactive`, as for
  any video-only model. The console page already handles that
  (`fv/h3-silent`).

## 5. Exposure (default off)

- A causal model appears only when the engine serves it.
  - LongLive needs its weights: `FV_SFWAN_WEIGHTS` + `FV_LONGLIVE_WEIGHTS`
    (standalone causal backend), or a `[[models]]` SF-Wan entry with
    `longlive = "<dir of longlive_base/lora.safetensors>"`.
  - A LongLive model is also served as `longlive`.
- The fal app is opt-in through `[protocols] fal_apps = [...,
  "fastvideo/longlive"]` (an SF-Wan model: `fastvideo/<its id>`). `GET
  /fal/schema` marks an app `director: true` only when its model resolves
  here and supports a director (clip or causal).
- **Licence.** The LongLive-1.3B weights are **non-commercial**
  (CC-BY-NC(-SA) 4.0 on the HF card; research-longlive.md §4.3). The director
  form of a LongLive app says so in its description. Serving it commercially
  is an owner/legal decision.

## 6. Reactor consistency

The Reactor's `set_prompt` and the director's prompt updates both go through
`CausalControl::set_prompt` → `CausalDriver::block` →
`CausalRollout::set_prompt`. The re-cache is logged once per switch
(`prompt switch: KV re-cache`, with `recache_ms`) and reported in
`BlockStats.recache_ms`. The director forwards it in `chunk.causal.recache_ms`.

## 7. Tests

- `fastvideo-fal` unit tests:
  - `control::causal_refusals`, `control::causal_blocks_follow_prompts_and_scripts`
    (replan vs append, block-clock scripts), `info::causal_constants`.
- `tests/director_e2e.rs` (fake engine, real str0m peers):
  - `causal_session_streams_and_recaches_once_per_switch`: 48-frame chunks
    at 16 fps, video only, pacing, one `set_prompt` and one re-cache per
    switch, `invalid_image` for end images, 0 underruns, and the lease is
    released.
  - `causal_refusals_and_keep_policy`: `image_url` → `invalid_initial_image`;
    SF-Wan without re-cache.
  - `causal_catalog_and_form`: director mode, licence, 480p / 16:9 form.
- `fastvideo-engine-service`: `cuda::caps::models_entry_opts_into_longlive`.
  The fake `longlive` model re-caches at switches.
- `FV_SERVE_HEAVY=1 scripts/serve/check.sh` passes. The one exception,
  `gateway_burst::two_gateway_replicas_share_the_pool`, is a timing test
  unrelated to this change, and it passes when re-run alone.
- `tests/console/director_playback.cjs` is unchanged. It covers the clip
  director, and the console renders causal sessions through the same
  `chunk` messages.

## 8. GPU smoke (2026-10-02, one RTX PRO 6000, EUR-IS-1)

**Setup.**
- Pod `b4g43foab0yp4v`, runtime image `fastvideo-rs-runtime@sha256:dccdf956…`,
  EU volume read-only, 707 s, about $0.41.
- `fv-serve` built from this branch (`--features cuda,http-client`).
- `[[models]] longlive-1.3b` (`recipe = "sfwan21-1.3b"`,
  `longlive = ".../longlive-1.3b-safetensors"`),
  `fal_apps = ["fastvideo/longlive"]`.
- HTTP and WebRTC on loopback only. NVENC H.264.
- Driven on the pod by `examples/director_client`, with LongLive's
  `interactive_example.jsonl` prompts (line 1 opens, lines 2–4 switch).
- Script: `scripts/serve/e2e/director-causal-pod.sh`. Summaries, server log
  extracts and contact sheets: `artifacts/serve/director-causal/`.
- Load: 135 s (UMT5 + DiT + LoRA merge).

| Run | Frames received | Rate (arrival / RTP) | Chunks | Underruns | `deadline_missed` | Arrival gaps > 250 ms | Generation pace |
|---|---:|---|---:|---:|---:|---:|---|
| LongLive 60 s, switches at 15 / 30 / 45 s | 964 | 16.06 / 16.00 fps | 20 | 0 | 0 | 0 (max 65 ms) | 20.4 fps mean, 17.8 fps on switch chunks; generation / playback 0.79 |
| LongLive 60 s, no prompt update | 965 | 16.06 / 16.00 fps | 21 | 0 | 0 | 1 (266 ms at frame 9, start-up; RTP continuous) | 20.85 fps; 0.77 |

**Re-cache.** There were exactly 3 re-caches, one per switch. Each is logged
once by the engine (`prompt switch: KV re-cache … switch_recaches=1/2/3`) and
once by the director. They took 390 / 391 / 392 ms, and each switch block
took 965 ms against 573 ms for a normal block.
- `prompt_applied` arrived 1.36 / 1.58 / 1.58 s after the update was sent.
  That is the 2 s playout lead plus one block, and the switched block plays
  within about 2 s.
- TTFF after `configure` was 3.5 s on the first session and 0.76 s on the
  second.

**Picture.** `artifacts/serve/director-causal/llsw-sheet.jpg` (1 fps) and
`llsw-switches.jpg` (frames just before and after each switch) show the same
player, table and lighting throughout. There are no cuts, and each new prompt
is visible within one or two seconds.

**Clip director without prompt updates (LTX turbo, 480p, 10 s chunks, one
`configure`, 50 s).** The coordinator asked for this check: the owner had
reported that the clip director "stops after the first chunk".
- Chunks 1–4 were each dispatched with no client message, the moment the
  previous chunk was built and started playing (`director: chunk dispatched`
  logs). The engine was never idle.
- Generation took 40.9 s for chunk 0 (warm-up), then 14.4 / 12.9 / 13.1 s for
  each 10 s chunk.
- Each chunk therefore arrived 4.4 / 3.0 / 3.3 s after the previous one
  ended, giving `deadline_missed` and an underrun each time. The last frame
  holds, and the audio is silence.
- So the director does not stall without prompts. It holds whenever
  generation is slower than real time. On H3 (25–75 s per 10 s chunk) the
  holds last 15–65 s, which looks like a stop.
- A causal model has no such holds (table above).

## 9. Trying it from the console (owner)

1. A GPU worker with the EU or US weights volume at `/workspace` and this
   build of `fv-serve` (`--features cuda`). In its config:

   ```toml
   [[models]]
   id = "longlive-1.3b"
   family = "wan"
   recipe = "sfwan21-1.3b"
   weights = "${FV_WEIGHTS}/sfwan21-1.3b"
   resident = true
   longlive = "${FV_WEIGHTS}/longlive-1.3b-safetensors"

   [protocols]
   fal = true
   fal_director = true
   fal_apps = ["fastvideo/longlive"]
   ```

   Also set `FV_TAE_DIR` (or the default `${FV_WEIGHTS}/auxiliary/tae`) for
   TAEHV. Without `[[models]]`, the env path also works:
   `FV_SFWAN_WEIGHTS=$W/sfwan21-1.3b FV_LONGLIVE_WEIGHTS=$W/longlive-1.3b-safetensors
   FV_SFWAN_MODEL=longlive-1.3b`.
2. Open `/console/models/fastvideo/longlive/director`.
   - The form offers 480p and 16:9 only and shows the licence note.
   - Start with a full scene prompt. Send the next directions from "Direct".
     Each one shows "applied" about 1.5 s later and is on screen about 2 s
     after it is sent.
   - Leave "Replan" on. Off, each queued direction holds one 3 s chunk.
3. Over the gateway, the worker pool must route `fastvideo/longlive/director`
   to that worker (`gateway.md` §5.1: the app's alias part, `longlive`, must be
   a model of the pool).
4. Licence: research and evaluation only (§5).
