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
