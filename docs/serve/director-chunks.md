# Director chunk length: 5 s or 10 s

A clip director session (H3, LTX, Wan) builds its stream from chunks, and
each chunk continues from the previous one's last frame. The session picks
the chunk length. The default is **5 s**; 10 s is the alternative.
Causal models (LongLive, SF-Wan) have no chunks to size (one rollout, see
`director-causal.md`), so they ignore the setting.

## Protocol

`configure.chunk_duration` is an integer number of seconds, `5` or `10`, and
is optional (`null` or absent means the server's default). fal's published
schema has no such field, but it echoes the session's chunk length as
`configured.chunk_duration`, and we use that name. Any other value or type
gets `error{code:"invalid_message"}` naming the field, as for every
out-of-range field.

A model or tier whose clip range cannot hold the requested length runs the
nearest length it serves. `configured.chunk_duration` always reports what
the session runs.

`session_info` / `POST /info` on a clip model:

| key | example (LTX) | meaning |
|---|---|---|
| `default_chunk_duration`, `chunk_seconds` | `5` | chunk length without `chunk_duration` |
| `min_chunk_duration`, `max_chunk_duration` | `5`, `15` | the model's clip range (script cuts) |
| `chunk_duration_options` | `[5, 10]` | lengths this model serves |
| `chunk_duration_options_by_resolution` | `{"480p":[5,10], ...}` | per tier (H3 1080P: `[5]`) |
| `chunk_duration_frames` | `{"5":121,"10":241}` | generated frames per option |
| `chunk_duration_note` | `null` | why an option is missing |

The director form (`GET /fal/schema/{app}/director`) carries a
`chunk_duration` property on clip models (`enum`, `default`, `x-fv-labels`,
`x-fv-options-by-resolution`).

The server default is `[director] chunk_seconds` (5). It is snapped to the
nearest served option.

## Per model

Frames are snapped up onto each model's grid at its rate (`frames_for`,
as the engine does). Continuations drop the duplicated anchor frame, so a
chunk plays `frames - 1`.

| model | grid, fps | 5 s | 10 s |
|---|---|---|---|
| H3 (turbo, max) | 17n+5, 24 | 124 (5.13 s played) | 243 (10.08 s) |
| H3 1080P tier | 17n+5, 24 | 124 | not served (5 s cap) unless `h3_1080p_long`; a 10 s request runs 5 s |
| LTX-2.5 two-stage | 8k+1, 24 | 121 (5.00 s) | 241 (10.00 s) |
| Wan 2.2 TI2V-5B / FastWan 2.2 | 4k+1 ≤ 161, 24 | 121 | not served (6.7 s max): runs 5 s |
| FastWan 2.1 1.3B | 4k+1 ≤ 129, 16 | 81 | not served (8.1 s max): runs 5 s |
| LongLive, SF-Wan (causal) | 12-frame blocks, 16 | n/a | n/a (3 s chunk of 4 blocks) |

## Behaviour at 5 s

- **Scripts.** Long gaps between beats are cut into chunks of the session's
  length. The last chunk takes the remainder, up to one and a half chunks,
  and never less than `min_chunk_duration` (5 s). At 5 s: 16 s → 5+5+6 and
  10 s → 5+5. At 10 s nothing changed: 32 s → 10+10+12. A gap that
  cannot be cut within `min..max` (a 7 s gap under H3's 5 s 1080p cap) is
  `infeasible_timing` / `invalid_initial_script` when an end image ends it.
  A text beat there directs from the next chunk instead.
- **End images** (`end_image_url`, end-image beats) end a chunk of the
  session's length, or end exactly on their beat. The ≥ 3 s spacing rule
  and the 5 s minimum chunk are unchanged.
- **Buffering.** `[director] buffer_chunks` counts chunks, so at 5 s it
  queues half the video (and half the host RAM). The build loop is
  unchanged: one build at a time. A chunk avoids an underrun when it builds
  faster than the chunk before it plays. Shorter chunks shorten the time
  for a prompt update to reach the picture (it lands at the next
  undispatched chunk).
- **Cost.** The LTX tier labels (`LTX_COST_*`) were measured on 121-frame
  (5 s) chunks (docs/serve/e2e/ltx.md "Director tiers"). A 10 s chunk costs
  a little more than twice that, because attention grows with length.
- **Anchor.** Each continuation starts from the previous chunk's
  uncropped last frame, whatever the length.
