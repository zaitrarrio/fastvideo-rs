# fal.ai parity matrix: modes and resolutions per model family

Status: research note, 2026-09-28. Goal (owner): serve the same modes and
resolutions that fal.ai offers for each model family we serve (MiniMax H3,
LTX, Wan). This note lists every relevant fal endpoint, compares it with what
this repo serves today, and ranks the work. §1 to §4 are the research
snapshot (repo at `fca5810`); the P0 items are now implemented, see §5 (the
status cells below that changed carry **P0 (§5)**).

Status words used below:

- **served**: a client of that fal endpoint gets the same mode, resolutions,
  durations and fps from us, allowing for the field gaps listed.
- **partial**: we generate the mode, but on another wire (LTX API, native,
  OpenAI) or with fewer resolutions, durations or fields.
- **missing**: no engine path.

Kinds of work: **config** (TOML, aliases, catalog constants), **schema** (a fal
request schema, routing or output shape in `crates/fastvideo-fal`), **engine
port** (new conditioning or pipeline code for weights we already load),
**new weights** (a checkpoint we do not ship yet, with its repo and license),
**not possible** (no open weights exist for it).

## 0. Sources

### fal (fetched 2026-09-28)

- **Catalog search**: `https://api.fal.ai/v1/models?q=<term>&limit=100`, run
  for `h3`, `minimax`, `ltx` (two pages), `LTX-2`, `ltx-2.5`, `lightricks`,
  `wan`, `fast-wan`, `self-forcing`, `realtime video` and `longlive`. The
  lists in §1 to §3 come from those results.
- **Model pages**: each endpoint's `https://fal.ai/models/<endpoint_id>/llms.txt`.
  This is the agent-readable copy of the model page and its `/api` tab. It
  includes the input schema (types, defaults, enums, ranges), the output
  schema and the pricing. For each id listed below, this page was fetched and
  read:
  - H3: `minimax/h3-max/{text-to-video, image-to-video, reference-to-video, director, extend-video, 3d-to-video, lip-sync/image-to-video, camera-controls, styles/vhs}`, `minimax/h3-max-turbo/{text-to-video, image-to-video}`, `minimax/h3/{text-to-video, image-to-video, reference-to-video, text-to-video/lora, image-to-video/lora, reference-to-video/lora}`.
  - LTX: `lightricks/ltx-2.5/{text-to-video, image-to-video, audio-to-video}/{pro, fast}`, `fal-ai/ltx-2.3/{text-to-video, text-to-video/fast, image-to-video, image-to-video/fast, audio-to-video, extend-video, retake-video, reframe}`, `fal-ai/ltx-2.3-22b/{text-to-video, image-to-video, audio-to-video, video-to-video, extend-video, reference-video-to-video}`, `fal-ai/ltx-2.3-22b/distilled/{text-to-video, image-to-video, text-to-video/lora}`, `fal-ai/ltx-2.3-quality/{text-to-video, image-to-video, ingredient}`, `fal-ai/ltx-2/text-to-video`, `fal-ai/ltx-2-19b/text-to-video`.
  - Wan: `fal-ai/wan-t2v`, `fal-ai/wan-i2v`, `fal-ai/wan-flf2v`, `fal-ai/wan-pro/text-to-video`, `fal-ai/wan-t2v-lora`, `fal-ai/wan/v2.2-5b/{text-to-video, image-to-video, text-to-video/fast-wan, text-to-video/distill}`, `fal-ai/wan/v2.2-a14b/{text-to-video, image-to-video, text-to-video/turbo, image-to-video/turbo, video-to-video, text-to-video/lora}`, `fal-ai/wan/v2.2-14b/{speech-to-video, animate/move}`, `fal-ai/wan-vace-14b`, `fal-ai/wan-25-preview/{text-to-video, image-to-video}`, `wan/v2.6/text-to-video`, `fal-ai/krea-wan-14b/text-to-video`.
- **OpenAPI**: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/ltx-2.3-22b/text-to-video`.
  This gave the `video_size` enum: `square_hd, square, portrait_4_3,
  portrait_16_9, landscape_4_3, landscape_16_9`, or `{width, height}`.
- **Earlier work**: the H3 Max wire details (queue, director WMA and output
  shapes) are in [research-fal.md](research-fal.md). This note only covers
  modes, fields and pricing.
- **Not fetched individually**. These appear in the catalog search, and their
  names and categories are enough for this note:
  - H3: `minimax/h3-max/styles/{retro-toon-70s, low-poly, hand-drawn, 16bit-pixel}` and the four `minimax/h3/*/trainer` endpoints.
  - LTX: the `fal-ai/ltx-2.3-quality/*` IC-LoRA effect endpoints (clean-plate, hdr, deblur, inpaint, outpaint and others), the `ltx23-trainer-v2/*` trainers, and the `fal-ai/ltx-video*` / `ltxv-13b*` (LTX-Video 0.9.x) endpoints.
  - Wan: `fal-ai/wan-vace-14b/*` sub-apps, `fal-ai/wan-22-vace-fun-a14b/*`, `fal-ai/wan/v2.2-14b/animate/replace`, `wan/v2.6/*`, `fal-ai/wan/v2.7/*`, `alibaba/wan-3.0*`, `fal-ai/wan-effects`, `fal-ai/wan-motion`.
- **Fetch problems**: on the first try, some `llms.txt` fetches failed with
  TLS resets from the proxy. Every one of them succeeded on retry, so every
  page listed above was read.

### Hugging Face (`https://huggingface.co/api/models/<repo>`, fetched 2026-09-28)

These calls gave the license fields and file lists quoted in §2 and §3:

| Repo | License field |
|---|---|
| `Lightricks/LTX-2.5` | `ltx-2.x-community-license-agreement` (gated: auto) |
| `Lightricks/LTX-2.3` | `ltx-2-community-license-agreement` |
| `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | `ltx-2-community-license` |
| `MiniMaxAI/MiniMax-H3` | `minimax-h3-community-license-agreement` |
| `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` | `minimax-h3-community` |
| `Wan-AI/Wan2.2-TI2V-5B` | apache-2.0 |
| `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | apache-2.0 |
| `Wan-AI/Wan2.2-T2V-A14B` | apache-2.0 |
| `Wan-AI/Wan2.2-I2V-A14B` | apache-2.0 |
| `Wan-AI/Wan2.2-S2V-14B` | apache-2.0 |
| `Wan-AI/Wan2.1-VACE-14B` | apache-2.0 |
| `Wan-AI/Wan2.1-FLF2V-14B-720P` | apache-2.0 |
| `krea/krea-realtime-video` | apache-2.0 |
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | apache-2.0 |

The Lightricks repo list comes from
`https://huggingface.co/api/models?author=Lightricks&limit=100`.

### Ours (repo at `fca5810`)

- `crates/fastvideo-engine-service/src/cuda/caps.rs`: the `catalog()` tiers, `h3_caps`, `ltx2_caps` and `wan_caps`.
- `crates/fastvideo-protocol/src/caps.rs`: `ModelCaps::h3`, `CanvasCaps::h3`, `FrameGrid::h3`.
- `crates/fastvideo-protocol/src/error.rs`: `GapId`.
- `crates/fastvideo-fal/src/{schema.rs, catalog.rs, lib.rs, director/control.rs}`.
- `crates/fastvideo-serve/src/adapters.rs`: `fal_config`.
- `crates/fastvideo-ltxapi/src/{models.rs, request.rs, stubs.rs}`.
- `configs/serve/runpod*.toml`.
- `docs/serve/e2e/{h3-max, h3-turbo, ltx, wan}.md`.

### What we serve today, in one table

| Family | Tier alias: model | Tasks | Canvas | Frames / fps | Audio | Knobs |
|---|---|---|---|---|---|---|
| H3 | `h3-max`: `sol-h3`; `h3-turbo`: `fasth3-4step-vsa`; `h3-draft`: 480p TAEH3 | T2V, I2V, keyframes (first, last, first+last). Ref2V only with the ref2va DiT loaded (`H3Ref2vaNotLoaded` otherwise) | short edge 768 or 480 (draft: 480), max 768x1344, multiple of the H3 canvas, aspect 0.25 to 4 | 24 fps fixed; 4 to 15 s on the H3 chunk grid (fal wire: 5 to 15) | yes (32 kHz) | seed |
| LTX | `ltx-pro`: `ltx25-distill-dense`; `ltx-turbo`: `ltx25-distill-sol`; `ltx-draft`: NVFP4 + TAEHV | T2V only (`Ltx25I2V`, `LtxKeyframes`) | short edge 1080, 720, 1440 or 2160, pad and crop, max 3840x2176 | 9 to 481 frames (8k+1); 24, 25, 48, 50 fps | yes (vocoder) | seed |
| Wan | `wan-max`: Wan2.2 TI2V-5B, UniPC 50; `wan-turbo`: FastWan2.1 1.3B DMD-3; `wan-draft`: + TAEHV; untiered `sfwan21-1.3b` (causal) | 5B: T2V + I2V. 1.3B: T2V | 5B: short edge 704 or 480, max 1280x704. 1.3B: 480, max 832x480 | 5B: ≤ 121 frames at 24 fps. 1.3B: ≤ 129 frames at 16 fps (24 accepted as the container rate) | no | 5B: seed, steps, guidance, shift, negative prompt. 1.3B: seed, steps, shift |

The wires:

- **fal wire.** Default apps are `minimax/h3-{max,turbo,draft}`, each with the
  three subs `text-to-video`, `image-to-video` and `reference-to-video`.
  `[protocols] fal_apps` can mount any `owner/alias`, but every app uses the
  **H3 request schema** (`schema.rs`). The director (WMA) is served when
  `fal_director = true`.
- **LTX API.** It mirrors fal's LTX-2.5 and LTX-2.3 partner endpoints
  (`ltx-2-5-{fast,pro}` and `ltx-2-3-{fast,pro}` map to our LTX-2.5 tiers).
- **Other wires.** Native, OpenAI `/v1/videos`, MiniMax and Reactor.

## 1. MiniMax H3

### 1.1 fal endpoints (fal's own fields)

The five `minimax/h3-max[-turbo]` HTTP endpoints share these fields:

- **Core fields**: `prompt` (1 to 50 000 characters), `duration` (integer, 5
  to 15, default 5), `resolution` (`480P`, `768P` (default) or `1080P`, where
  1080P is "latent refinement from a native 768P source"), `seed`,
  `enable_safety_checker`, `sync_mode` and `prompt_expansion_mode`
  (`disabled`, `balanced` (default) or `quality`).
- **t2v**: adds `target_audio_url` and `aspect_ratio` (`21:9`, `16:9`
  (default), `4:3`, `1:1`, `3:4` or `9:16`).
- **i2v**: adds `target_audio_url`, `image_url` and `end_image_url`. Both
  images are optional, and end-only is allowed.
- **r2v**: see §1.3.
- **Output**: 24 fps with audio.

| fal endpoint | Mode / extra fields | Resolutions | Price (per output second) |
|---|---|---|---|
| `minimax/h3-max/text-to-video` | T2V (+ `target_audio_url`, `aspect_ratio`) | 480P, 768P, 1080P | $0.025 / 0.04 / 0.08 promo until Sep 30; list $0.05 / 0.08 / 0.16 |
| `minimax/h3-max/image-to-video` | first frame, last frame, or both (+ `target_audio_url`) | same | same |
| `minimax/h3-max/reference-to-video` | Ref2V (§1.3) | same | $0.05 / 0.08 / 0.16, plus reference tokens |
| `minimax/h3-max/director` | Realtime WMA/WebRTC. `configure` takes `resolution` (`480p`, `768p`, `1080p`), `aspect_ratio` (`16:9`, `9:16`, `1:1`), `image_url`, `end_image_url`, `audio_url` (pinned soundtrack, FL2VA target audio), `memory` (1 to 50), `script` and `seed`. `prompt` messages take `audio_behavior`, `replan` and `script_mode` | 480p, 768p, 1080p | $0.04 promo, $0.08 list, 60 s minimum per session |
| `minimax/h3-max-turbo/text-to-video` | as h3-max t2v | 480P, 768P, 1080P | $0.0125 / 0.02 / 0.04 promo; list $0.025 / 0.04 / 0.08 |
| `minimax/h3-max-turbo/image-to-video` | as h3-max i2v | same | same |
| `minimax/h3/text-to-video` | Base H3. `prompt_expansion_mode` adds `fast` | 480P, 768P native; 2K and 4K upscaled from 768P (default **2K**) | $0.05 / 0.06 / 0.13 / 0.16 |
| `minimax/h3/image-to-video` | base H3 i2v (`image_url`, `end_image_url`) | same | same |
| `minimax/h3/reference-to-video` | base H3 r2v; the first 5 reference images are free, then $0.08 each | same | same |
| `minimax/h3/{text,image,reference}-to-video/lora` | as above, plus `loras: list<LoRAInput>` (required) | same | $0.0625 / 0.075 / 0.1625 / 0.20 |
| `minimax/h3-max/extend-video` | `video_url` (1.625 to 60 s, ≤ 50 MB, aspect 0.4 to 2.5), `prompt`, `enable_prompt_expansion`, `duration` 5 to 15, `aspect_ratio` (`auto` or the 6 ratios), `output` (`extended` or `continuation`), `seed` | 480P, 768P, 1080P, **2K** | $0.05 / 0.08 / 0.16 / 0.32, plus reference tokens |
| `minimax/h3-max/lip-sync/image-to-video` | `image_url` (aspect 0.4 to 2.5) + `audio_url` (≥ 5 s, clipped at 14.8 s), `enable_transcription`, `seed` | 480P, 768P, 1080P, **2K** | $0.05 / 0.08 / 0.16 / 0.32 |
| `minimax/h3-max/camera-controls` | `image_url` (required), `camera_trajectory` (keyframes of azimuth, elevation, distance and time; ≤ 32 turns), `duration` **3** to 15, default prompt "only the camera moves" | 480P (default), 768P, 1080P | promo $0.025 / 0.04 / 0.08 |
| `minimax/h3-max/styles/{vhs, retro-toon-70s, low-poly, hand-drawn, 16bit-pixel}` | style presets (vhs: `damage_level`), optional `image_url`, `duration` 5 to 15, 6 aspect ratios | 768p only | $0.08 |
| `minimax/h3-max/3d-to-video` | Blender `video_url` (≤ 15 s, ≤ 32 shots) + optional `reference_image_urls`. `max_generated_reference_images` (1 to 8) are made automatically when none are given | 480P, 768P, 1080P | $0.05 / 0.08 / 0.16, plus reference tokens and generated images |
| `minimax/h3/{t2v,i2v,flf2v,ref2va}/trainer` | LoRA trainers | n/a | n/a |

### 1.2 Status against ours

| fal endpoint | Ours | Gap | What it takes |
|---|---|---|---|
| `minimax/h3-max/text-to-video` | **served** (`sol-h3`, E2E h3-max.md). `1080P` **served natively** (2026-09-28): generated at 1920x1088 and cropped to 1920x1080 (other aspects at the 1088 short edge within 1088x1920), about 2.5x the GPU time of 768P, on 80 GB-class GPUs; smaller GPUs keep answering `H3Refine1080P` (docs/serve/h3-1080p-and-upscaler.md) | `target_audio_url` is refused (`H3TargetAudio`, E10). `prompt_expansion_mode` is accepted and ignored. No safety checker. 1080P is native generation, not fal's latent refinement from 768P, so the pixels differ from fal's | target audio: **engine port** (FL2VA target-audio conditioning, E10). Prompt expansion: an LLM service integration |
| `minimax/h3-max/image-to-video` | **served** (first frame, last frame, both) | same as t2v | same |
| `minimax/h3-max/reference-to-video` | **served when configured**: the Ref2VA companions `h3-ref2v-max` (base Ref2VA DiT, 49 forwards) and `h3-ref2v-turbo` (Sol-H3 Ref2VA, 4 forwards) take Ref2V requests on `h3-max` / `h3-turbo` (`route_task`); limits 9 / 3 / 3 / 12, clips 2 to 15 s and 15 s per kind, `adaptive` from the first image (else video) reference (docs/ports/h3-ref2v.md) | `1080P` (`H3Refine1080P`); reference-token billing is not modelled | Deploy `configs/serve/runpod-h3-ref2v.toml` (own process on 80-96 GB cards) |
| `minimax/h3-max/director` | **served** (`fal_director`). `1080p` served with the native tier: chunks stream at the 1920x1088 generation canvas, and each chunk takes about 2.5x as long to build as at 768p, so a session falls further behind real time (a 10 s chunk takes roughly 2 min on h3-turbo) | `1080p` without the tier gives `invalid_input`. `audio_url` gives `invalid_initial_audio`. No prompt expander (a chunk's prompt is premise + direction) | `audio_url`: engine port (E10) |
| `minimax/h3-max-turbo/{text,image}-to-video` | **served (P0, §5)**: mounted by default on the H3 Turbo tier. Was **partial**: we serve the turbo tier as `minimax/h3-turbo` (plus `h3-draft`), not under fal's id | app id | **config**: add `"minimax/h3-max-turbo"` to `fal_apps` and `[aliases] "h3-max-turbo" = "fasth3-4step-vsa"`. Better: a one-line arm in `adapters::fal_config` giving it the H3 Turbo tier fallback. fal has no `h3-max-turbo/reference-to-video`; ours adds it on every app, which is harmless |
| `minimax/h3/{t2v,i2v,r2v}` (base H3) | **served (P0, §5)** at 480P and 768P on the H3 Max tier (mounted by default; `2K` / `4K` are listed and answer a clean 422 `H3Resolution2K` on `resolution`). Was **partial**: 480P and 768P are what we generate; the base-model app id is not mounted | `2K` and `4K` (the default!) are refused (`H3Resolution2K`). App id `minimax/h3` | App id: **config**. 2K/4K: **not possible** with fal's pipeline. It would take **new weights**: an open video super-resolution model (none is vetted for this repo yet) |
| `minimax/h3/*/lora` | **missing** | `loras` (per-request LoRA) | **engine port** (`GapId::Lora`: runtime LoRA apply/unapply) plus a LoRA file format matching fal's trainers |
| `minimax/h3-max/extend-video` | **missing** | continuation of a client video, `output` extended or continuation | **engine port**: a video-tail anchor (the director already continues from a last-frame anchor; this needs the source clip's tail as context) plus a stitching output. 2K: not possible |
| `minimax/h3-max/lip-sync/image-to-video` | **missing** | image + driving audio | **engine port**: I2V + E10 target audio. 2K: not possible |
| `minimax/h3-max/camera-controls` | **missing** | `camera_trajectory`, 3 s | **not possible**: no open H3 camera-control weights. 3 s is below the H3 grid (4 s minimum, `H3FourSeconds`) |
| `minimax/h3-max/styles/*` | **missing** | style presets | **not possible** as fal does it (fal's style adapters are not published). A prompt-only approximation is possible but not parity |
| `minimax/h3-max/3d-to-video` | **missing** | Blender proxy video, auto-generated references | After ref2va: **engine port** (proxy video as a Ref2V video reference) plus an image-generation and planning service. Low value |
| trainers | out of scope | | |

H3 fields all fal H3 endpoints share: fps 24 (same), durations 5 to 15
(same), aspect enums (same), seed (same). `sync_mode` is the same, and
`enable_safety_checker` is accepted without a checker.

### 1.3 Alignment targets for the ref2va agent (fal `minimax/h3-max/reference-to-video`)

Match these exactly (the queue OpenAPI maxItems are in research-fal.md §5.2;
this page confirms the texts):

- `prompt`: required, 1 to 50 000 characters. References are named by
  modality and order: "Image 1, Image 2, Video 1, Audio 1".
- `duration`: integer 5 to 15, default 5. `resolution`: `480P`, `768P`
  (default) or `1080P` (1080P stays `H3Refine1080P` on the Ref2VA models:
  the native 1080P tier is validated for T2V and I2V only).
- `aspect_ratio`: **`adaptive` (default)**, `21:9`, `16:9`, `4:3`, `1:1`,
  `3:4` or `9:16`.
- `reference_image_urls`: maxItems 9.
- `reference_video_urls`: maxItems 3; **2 to 15 s each, combined ≤ 15 s**.
- `reference_audio_urls`: maxItems 3; **2 to 15 s each, combined ≤ 15 s**.
- Images, videos and audio can be given alone or together, **≤ 12 files in
  total**. The schema has no `target_audio_url` on r2v.
- `seed`, `enable_safety_checker`, `sync_mode` and `prompt_expansion_mode` as
  in t2v.
- Billing (for any cost report): 4 096 reference tokens are included; after
  that, $0.02 per 1 000 tokens. A square image counts 1 024 tokens, so the
  first four square images with no video or audio are free.
- The base-model variant `minimax/h3/reference-to-video` has the same
  fields, resolutions `480P`, `768P`, `2K` and `4K`, and "first 5 reference
  images free".
- Our `schema.rs` already has these limits (9 / 3 / 3, 12 in total). The
  engine must enforce the per-clip and combined 15 s limits and `adaptive`
  (canvas from the first image reference).

## 2. LTX

### 2.1 fal endpoints

**Partner API endpoints.** These have the same contract as api.ltx.io, which
`crates/fastvideo-ltxapi` already mirrors. Output has audio unless
`generate_audio: false`.

| fal endpoint | Mode / fields | Resolutions | Duration (s) | fps | Price |
|---|---|---|---|---|---|
| `lightricks/ltx-2.5/text-to-video/fast` | `prompt`, `duration`, `resolution`, `aspect_ratio` (`16:9`, `9:16`), `fps`, `generate_audio`, `camera_motion` (dolly in/out/left/right, jib up/down, static, focus_shift) | 720p, 1080p (default), 1440p, 2160p | 6 to 20 (even) or `auto` (default). ≤ 10 at 48/50 fps or at 1440p/2160p | 24, 25 (default), 48, 50 | $0.09 / 0.13 / 0.19 / 0.30 per s |
| `lightricks/ltx-2.5/text-to-video/pro` | same | 720p, 1080p | 6, 8, 10, `auto` | 24, 25, 50 | $0.12 / 0.17 per s |
| `lightricks/ltx-2.5/image-to-video/fast` | + `image_url` (required), `end_image_url`; `aspect_ratio` `auto` (default), `16:9` or `9:16` | as t2v fast | as t2v fast | as t2v fast | as t2v fast |
| `lightricks/ltx-2.5/image-to-video/pro` | same | 720p, 1080p | 6, 8, 10, `auto` | 24, 25, 50 | $0.12 / 0.17 per s |
| `lightricks/ltx-2.5/audio-to-video/{fast,pro}` | `audio_url` (2 to 20 s; pro ≤ 10), optional `image_url`, `prompt` (required without an image), `guidance_scale` 1 to 50 (default 5, or 9 with an image), `aspect_ratio` `auto`, `16:9` or `9:16` | 1080p | follows the audio | n/a | $0.13 fast / $0.17 pro per input second |
| `fal-ai/ltx-2.3/text-to-video[/fast]` | `prompt`, `duration`, `resolution`, `aspect_ratio` (`16:9`, `9:16`), `fps`, `generate_audio` | 1080p, 1440p, 2160p | pro 6, 8, 10. fast 6 to 20 (>10 only at 25 fps and 1080p) | 24, 25 (default), 48, 50 | pro $0.08 / 0.16 / 0.32; fast $0.06 / 0.12 / 0.24 per s |
| `fal-ai/ltx-2.3/image-to-video[/fast]` | + `image_url`, `end_image_url`, `aspect_ratio` `auto` | same | same | same | same |
| `fal-ai/ltx-2.3/audio-to-video` | as 2.5 a2v | | | | $0.10 per s |
| `fal-ai/ltx-2.3/extend-video` | `video_url`, `prompt`, `duration` 2 to 20 (float), `mode` `start` or `end`, `context` 1 to 20 s | source | 2 to 20 | source | $0.10 per s |
| `fal-ai/ltx-2.3/retake-video` | `video_url`, `prompt`, `start_time` 0 to 20, `duration` 2 to 20, `retake_mode` (`replace_audio`, `replace_video`, `replace_audio_and_video`) | source | 2 to 20 | source | $0.10 per s |
| `fal-ai/ltx-2.3/reframe` | `video_url`, `aspect_ratio` (`1:1`, `4:5`, `5:4`, `9:16`, `16:9`); outpaints the exposed area | 720p, 1080p | source | source | $0.10 / 0.20 per input second |
| `fal-ai/ltx-2/text-to-video` | LTX-2 partner API | 1080p, 1440p, 2160p | 6, 8, 10 | 25, 50 | $0.06 / 0.12 / 0.24 per s |

**Open-weight endpoints fal runs itself**, billed per megapixel of generated
`width × height × frames`. The common fields:

- `num_frames`: 9 to 481, default 121.
- `video_size`: `square_hd`, `square`, `portrait_4_3`, `portrait_16_9`,
  `landscape_4_3`, `landscape_16_9`, `auto`, or `{width, height}`.
- `fps`: float, 1 to 60, default 24.
- `generate_audio`, `seed` and `enable_prompt_expansion`.
- `negative_prompt`, and `camera_lora` (the 7 camera LoRAs + `none`).
- `acceleration`: `none`, `regular`, `high` or `full`.
- `video_output_type`: mp4, webm, ProRes or gif.
- `video_quality`, `video_write_mode` and `sync_mode`.

| fal endpoint | Extra fields | Price |
|---|---|---|
| `fal-ai/ltx-2.3-22b/text-to-video` (dev) | `num_inference_steps` 8 to 50 (40); video/audio CFG, STG, rescale and modality scales; `scheduler`; `use_multiscale`; restart sampling; distill-LoRA pass scales | $0.001605 / MP |
| `fal-ai/ltx-2.3-22b/image-to-video` | + `image_url`, `end_image_url`, `image_strength`, `end_image_strength`, `interpolation_direction` | same |
| `fal-ai/ltx-2.3-22b/audio-to-video` | + `audio_url`, optional image and end image, `match_audio_length`, `audio_strength`, `preprocess_audio` | same |
| `fal-ai/ltx-2.3-22b/video-to-video` | `video_url`, `video_strength` (0.01), `audio_strength`, `match_video_length`, `match_input_fps` | same |
| `fal-ai/ltx-2.3-22b/extend-video` | `video_url`, `end_image_url`, `extend_direction` (`forward` or `backward`), `num_context_frames` 0 to 121 (25) | same |
| `fal-ai/ltx-2.3-22b/reference-video-to-video` | IC-LoRA. `video_url`, optional `audio_url`, `image_url` and `end_image_url`, `preprocessor` (`depth`, `canny`, `pose`, `none`), `ic_lora_type` (`match_preprocessor`, `union`, `detailer`, `none`), `video_strength`, `audio_strength` | same |
| `fal-ai/ltx-2.3-22b/distilled/{text,image,audio,video}-to-video` | as dev without steps and guidance; `acceleration` default `none` | $0.001205 / MP |
| `…/lora` variants | + `loras` (required) | $0.001405 / MP (distilled) |
| `fal-ai/ltx-2.3-quality/{text,image}-to-video` | `num_inference_steps` 8 to 30 (15), `frames_per_second`, `resolution` (ImageSize or enum), `image_strength` (0.7) | $0.0024075 / MP |
| `fal-ai/ltx-2.3-quality/ingredient` | Reference-sheet IC-LoRA. `image_url` (the sheet), `ingredient_strength` 0 to 2, `reference_strength` 0 to 2, default size 1536x896 (768x448 first stage + 2x refine) | same |
| `fal-ai/ltx-2.3-quality/{clean-plate, hdr, deblur, colorization, day-to-night, decompression, water-simulation, instant-shave, cross-eyed, render-to-real, inpaint, outpaint, …}` | IC-LoRA effects (video-to-video) | per MP |

### 2.2 Status against ours

Ours today:

- **Engine**: LTX-2.5 22B distilled, T2V + audio, 720p, 1080p, 1440p and
  2160p, 24, 25, 48 and 50 fps, 9 to 481 frames.
- **LTX API**: the fast/pro matrix exactly as fal's LTX-2.5 fast (models.rs),
  with pro allowing up to 4K.
- **fal wire** (`fastvideo/ltx-turbo`): H3 schema. Only `1080P` reaches the
  engine (`480P` is refused; `768P` falls back to 1080, E2E ltx.md bug 11).
  There is no `fps`, `generate_audio` or `camera_motion` field, and
  durations are 5 to 15 only.

| fal endpoint | Ours | Gap | What it takes |
|---|---|---|---|
| `lightricks/ltx-2.5/text-to-video/fast` | **served (P0, §5)** on the fal wire (`lightricks/ltx-2.5` + `text-to-video/fast`, `ltx-turbo`): 720p to 2160p, 24/25/48/50 fps, 6 to 20 s, `generate_audio`; left: `auto` (`LtxAutoDuration`), non-static `camera_motion`, and combinations over 481 frames (20 s at 25 fps, 10 s at 50 fps). Was **partial**: same matrix on the LTX API wire (`ltx-2-5-fast` goes to `ltx-turbo`). On the fal wire only 1080p at 24 fps, 5 to 15 s | fal app id and sub-path, 720p/1440p/2160p, fps, 16 to 20 s, `generate_audio`, `duration: auto`, `camera_motion` | App, schema and routing: **schema** (an LTX fal schema and multi-segment subs `text-to-video/fast`; see P0-2). `auto`: **new weights + engine port** (`model_patches/ltx-2.5-duration-head-bf16.safetensors` in `Lightricks/LTX-2.5`, `LtxAutoDuration`). `camera_motion`: **not possible** on 2.5 (camera LoRAs exist only as `Lightricks/LTX-2-19b-LoRA-Camera-Control-*`). Accept `static` and refuse the rest, or approximate with the prompt |
| `lightricks/ltx-2.5/text-to-video/pro` | **served (P0, §5)** as fast on `ltx-pro` (720p/1080p, 24/25/50 fps, 6 to 10 s; 10 s at 50 fps is over 481 frames). Was **partial**: as fast (`ltx-2-5-pro` goes to `ltx-pro`) | as fast | as fast |
| `lightricks/ltx-2.5/image-to-video/{fast,pro}` | **schema served (P0, §5)**: `image_url` (required), `end_image_url`, `aspect_ratio` `auto`; normalizes to `Task::I2V` / `Task::Keyframes`, so it runs wherever the LTX engine has the I2V port (§2.3). Was **missing** (`Ltx25I2V`; `LtxKeyframes` for `end_image_url`) | whole mode | **engine port** (weights loaded). **In progress (other agent); targets in §2.3** |
| `lightricks/ltx-2.5/audio-to-video/{fast,pro}` | **served** (2026-09-29): `audio_url` (required; 2 to 20 s, pro ≤ 10 s), `image_url` (first frame), `prompt` (required without an image), `aspect_ratio` (`auto`: 9:16 for a portrait image, else 16:9), 1080p at 24 fps; fast on `ltx-turbo`, pro on `ltx-pro`. Engine: `Task::A2V`, the driving audio VAE-encoded and pinned as clean audio latents on both distilled stages, the output carrying the input audio (docs/oracle.md "LTX-2.5 audio-to-video"). Also on the LTX API (`/v1\|v2/audio-to-video`) and the native API (`audio_url`). Was **missing** (403 stub) | `guidance_scale`: validated, then a no-op (the distilled model is unguided; fal's hosted model is guided, `A2VidPipelineTwoStage` with the dev DiT). The frame count is the longest 8k+1 clip within the audio; fal's rule is not published. This server adds `seed` and `sync_mode` | Faithful guided A2V: **new weights** (`transformer_full/` of `Lightricks/LTX-2.5-Diffusers`, 38.0 GB, LTX-2 community license, gated auto) **+ engine port** (CFG/STG/modality guidance sampler) |
| `fal-ai/ltx-2.3/{text,image}-to-video[/fast]`, `audio-to-video` | as the 2.5 rows (`ltx-2-3-*` ids map to our 2.5 tiers on the LTX API) | Real 2.3 weights, if exact 2.3 output is wanted | **config + new weights** (`LtxVersion::V23` exists; `FastVideo/LTX-2.3-Distilled-Diffusers` or `Lightricks/LTX-2.3`, LTX-2 Community License). Low value: 2.5 is a superset |
| `fal-ai/ltx-2.3/extend-video`, `fal-ai/ltx-2.3-22b/extend-video` | **missing** (stub 403) | extend forward or backward with context frames | **engine port**: video-latent prefix or suffix conditioning, same weights |
| `fal-ai/ltx-2.3/retake-video` | **missing** (stub 403) | temporal-window regeneration of audio, video or both | **engine port**: masked temporal denoise, same weights |
| `fal-ai/ltx-2.3/reframe`, `…-quality/outpaint`, `…/inpaint` | **missing** (stub 403) | outpainting and inpainting | **new weights + engine port**: `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting` (2.3 only; no 2.5 build listed) plus IC-LoRA conditioning |
| `fal-ai/ltx-2.3-22b/reference-video-to-video` | **missing** | IC-LoRA control from a reference video plus depth, canny and pose preprocessors | **new weights + engine port**: `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` (2.3 only), `…-Motion-Track-Control`, plus depth and pose estimators |
| `fal-ai/ltx-2.3-quality/ingredient` | **served (engine oracle-checked; serve E2E pending)**: app `fal-ai/ltx-2.3-quality`, endpoint `ingredient` (`AppKind::LtxQuality`, `schema/ingredient.rs`) on `ltx-pro`, which routes Ref2V to the IC-LoRA companion `ltx25-ref2v` (the LTX-2.5 build of the same LoRA). Fields `prompt`, `image_url` (the sheet), `ingredient_strength` 0-2 (the stage-1 LoRA strength), `reference_strength` (fal 0-2; ours 0-1, above 1 answers 422 on `reference_strength`), `num_frames` 9-481 (ours up to 241), `frames_per_second` 1-60 (ours 24/25/48/50), `generate_audio`, `negative_prompt` (no-op), `seed`, `sync_mode`; always 1536x896 | 2.5 weights instead of 2.3; `reference_strength` above 1; clips over 241 frames; fps outside the LTX set; no size field (fal's default only) | done: docs/ports/ltx-ref2v.md, docs/oracle.md "LTX-2.5 reference-to-video" |
| `fal-ai/ltx-2.3-quality/*` effects (clean-plate, colorization, day-to-night, deblur, decompression, water-simulation, HDR…) | **missing** | per-effect IC-LoRA | **new weights + engine port** (the same IC-LoRA path). 2.5 builds exist for clean-plate, colorization, day-to-night, deblur, decompression, water-simulation and pixel-spatial-upscaler; HDR, relight and in-outpainting are 2.3 only |
| `fal-ai/ltx-2.3-22b/video-to-video` | **missing** | SDEdit-style restyle (`video_strength`) | **engine port**, same weights |
| `fal-ai/ltx-2.3-22b/distilled/text-to-video` | **partial**: our model is the 2.5 counterpart | `num_frames`, `video_size` presets and custom sizes, `fps` 1 to 60, `camera_lora`, `negative_prompt` | **schema**: frames and size map onto our caps; fps other than 24/25/48/50 is `LtxFps` (E4 validated only those); negative prompt is meaningless for the unguided distilled model |
| `fal-ai/ltx-2.3-22b/*` (dev), `ltx-2.3-quality/{t2v,i2v}` | **missing** | guided 15 to 50 step sampling, CFG/STG knobs | **new weights** (`ltx-2.5-22b-dev-transformer-bf16` in `Lightricks/LTX-2.5`, or `ltx-2.3-22b-dev` in `Lightricks/LTX-2.3`) + **engine port** (guided sampler). Low priority: slower than our distilled tiers |
| `…/lora` variants | **missing** (`GapId::Lora`; `ltx2/lora.rs` fuses only at startup) | per-request `loras` | **engine port** (runtime LoRA swap) |
| `fal-ai/ltx-2/text-to-video` | **partial**: LTX-2 partner API. Our LTX API answers `ltx-2-{fast,pro}` with 400 (removed upstream) | n/a | nothing |

### 2.3 Alignment targets for the LTX image/reference agent

**`lightricks/ltx-2.5/image-to-video/fast`** (and `/pro`). This is the same
contract as LTX API `/v2/image-to-video` (`image_uri` and `last_frame_uri`):

- `image_url`: required. `end_image_url`: optional. A given end image means a
  first-to-last transition (our `Task::Keyframes`).
- `prompt`: required.
- `duration`:
  - fast: 6, 8, 10, 12, 14, 16, 18 or 20, or `auto` (default). At 720p and
    1080p, 24/25 fps allow up to 20 s and 48/50 fps up to 10 s. At 1440p and
    2160p every fps is limited to 10 s.
  - pro: 6, 8, 10 or `auto`.
- `resolution`: fast `720p`, `1080p` (default), `1440p` or `2160p`; pro
  `720p` or `1080p`.
- `aspect_ratio`: **`auto` (default, follows the image)**, `16:9` or `9:16`.
- `fps`: fast 24, 25 (default), 48 or 50; pro 24, 25 or 50.
- `generate_audio`: default true. `camera_motion`: the 8-value enum (§2.2
  row 1).
- 2.3 variant (`fal-ai/ltx-2.3/image-to-video[/fast]`): resolutions 1080p,
  1440p and 2160p only; fast durations > 10 s only at 25 fps and 1080p.
- Open-weight variant (`fal-ai/ltx-2.3-22b/distilled/image-to-video`): adds
  `image_strength` and `end_image_strength` (0 to 1, default 1),
  `interpolation_direction` (`forward` or `backward`), `num_frames` 9 to 481,
  `video_size` `auto`, and `fps` 1 to 60.

**Reference modes**, in order of fit to our LTX-2.5 base:

1. `fal-ai/ltx-2.3-quality/ingredient` (**implemented 2026-09-28**, see the §2.2 row):
   - Inputs: `image_url` (a reference sheet with character, prop and
     location panels), `prompt` ("Reference sheet: … Generated video: …"),
     `ingredient_strength` 0 to 2 (1), `reference_strength` 0 to 2 (1),
     `num_frames` 9 to 481 (121), `frames_per_second` 1 to 60 (24),
     `generate_audio`, `negative_prompt` and `seed`.
   - Output size: default 1536x896 (stage 1 at the LoRA's 768x448 bucket,
     then 2x refine).
   - Weights: `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`.
2. `fal-ai/ltx-2.3-22b/reference-video-to-video`:
   - Inputs: `video_url` (required), optional `audio_url`, `image_url` and
     `end_image_url`, `preprocessor` (`depth`, `canny`, `pose`, `none`),
     `ic_lora_type` (`match_preprocessor`, `union`, `detailer`, `none`;
     default `union`), `video_strength` and `audio_strength` (default 1),
     `match_video_length` and `match_input_fps` (default true).
   - Weights: Union-Control exists only for 2.3 and 2-19b, so this needs 2.3
     weights or waits for a 2.5 build.

## 3. Wan

### 3.1 fal endpoints (open weights unless noted)

| fal endpoint | Model | Mode / fields | Resolutions / aspect | Frames / fps | Price |
|---|---|---|---|---|---|
| `fal-ai/wan/v2.2-5b/text-to-video` | Wan2.2 TI2V-5B | `negative_prompt`, `num_inference_steps` 2 to 50 (40), `guidance_scale` (3.5), `shift` (5), `interpolator_model` (`none`, `film`, `rife`) + `num_interpolated_frames` 0 to 4, `enable_prompt_expansion` | 580p, 720p (default); 16:9, 9:16, 1:1 | 17 to 161 (81); 4 to 60 fps (24) | $0.15 per video |
| `fal-ai/wan/v2.2-5b/image-to-video` | same | + `image_url`; aspect `auto` (default) | 580p, 720p | same | $0.15 per video |
| `fal-ai/wan/v2.2-5b/text-to-video/fast-wan` | **FastVideo FastWan2.2 5B** ("Wan 2.2's 5B FastVideo model") | no steps field; `guidance_scale`, interpolation | 480p, 580p, 720p; 16:9, 9:16, 1:1 | 17 to 161; 4 to 60 fps (24) | $0.0125 / 0.01875 / 0.025 per video |
| `fal-ai/wan/v2.2-5b/text-to-video/distill` | 5B distilled | `guidance_scale` default 1 | 580p, 720p | 17 to 161 | $0.08 per video |
| `fal-ai/wan/v2.2-a14b/text-to-video` | Wan2.2 T2V-A14B | steps (27), `guidance_scale` + `guidance_scale_2`, `shift`, `acceleration` | 480p, 580p, 720p; 16:9, 9:16, 1:1 | 17 to 161 | $0.04 / 0.06 / 0.08 per s (16 fps seconds) |
| `fal-ai/wan/v2.2-a14b/image-to-video` | Wan2.2 I2V-A14B | + `image_url`, **`end_image_url`** | same, aspect `auto` | same | same |
| `fal-ai/wan/v2.2-a14b/{text,image}-to-video/turbo` | A14B turbo | fixed-length | 480p, 580p, 720p | fixed | $0.05 / 0.075 / 0.10 per video |
| `fal-ai/wan/v2.2-a14b/video-to-video` | A14B | `video_url`, `strength` (0.9), `resample_fps` | same | 17 to 161 | as a14b |
| `fal-ai/wan/v2.2-a14b/text-to-video/lora`, `image-to-video/lora` | A14B | + `loras`, `reverse_video` | same | same | $0.10 per s |
| `fal-ai/wan/v2.2-14b/speech-to-video` | Wan2.2 S2V-14B | `image_url` + `audio_url` | 480p (default), 580p, 720p | 40 to 120 (multiple of 4) | $0.10 / 0.15 / 0.20 per s |
| `fal-ai/wan/v2.2-14b/animate/{move,replace}` | Wan2.2 Animate-14B | `video_url` + `image_url`, `use_turbo` | 480p, 580p, 720p | from the video | $0.04 to 0.08 per s |
| `fal-ai/wan-vace-14b` (+ `/depth`, `/pose`, `/inpainting`, `/outpainting`, `/reframe`) | Wan2.1 VACE-14B | `task`, `video_url`, masks, `ref_image_urls`, `first_frame_url`, `last_frame_url`, `sampler` | auto, 240p to 720p | 17 to 241; 5 to 30 fps | $0.04 / 0.06 / 0.08 per s |
| `fal-ai/wan-t2v` | Wan2.1 T2V-14B | `negative_prompt`, steps (30), `turbo_mode` | 480p, 580p, 720p; 16:9, 9:16 | 81 to 100; 5 to 24 fps (16) | $0.20 / 0.40 per video |
| `fal-ai/wan-i2v` | Wan2.1 I2V-14B | + `image_url`, `guide_scale`, `shift`, `acceleration` | 480p, 720p; auto, 16:9, 9:16, 1:1 | 81 to 100; 5 to 24 fps | same |
| `fal-ai/wan-flf2v` | Wan2.1 FLF2V-14B | `start_image_url` + `end_image_url` (both required), `loras` | 480p, 720p | 81 to 100 | same |
| `fal-ai/wan-t2v-lora` | Wan2.1 T2V | + `loras`, `reverse_video` | 480p, 580p, 720p | 81 to 100 | $0.75 per video |
| `fal-ai/krea-wan-14b/text-to-video` (+ `video-to-video`) | Krea realtime (self-forcing Wan 14B) | `prompt`, `seed`, `enable_prompt_expansion` | fixed | 18 to 162 frames (12k+6) | $0.025 per s |
| `fal-ai/wan-pro/text-to-video` | Wan-2.1 **Pro** (closed) | `prompt`, `seed` | 1080p, 30 fps, ≤ 6 s | | $0.80 per 5 s |
| `fal-ai/wan-25-preview/*`, `wan/v2.6/*`, `fal-ai/wan/v2.7/*`, `alibaba/wan-3.0*` | closed Alibaba models | `audio_url` (BGM), durations 5, 10 (15 on 2.6) | 480p to 1080p | | $0.05 to 0.15 per s |

fal has **no** Wan 2.1 1.3B, FastWan 2.1 1.3B or SF-Wan 1.3B endpoint. Our
`wan-turbo` tier has no direct fal counterpart; fal's fast Wan is FastWan2.2
5B.

### 3.2 Status against ours

Ours today:

- `wan-max` = Wan2.2 TI2V-5B: T2V + I2V, 1280x704 or 480p, ≤ 121 frames at
  24 fps, UniPC 50, all knobs.
- `wan-turbo` = FastWan2.1 1.3B: 832x480, ≤ 129 frames at 16 fps.
- `sfwan21-1.3b`: causal live.
- fal wire (`fastvideo/<model>`): H3 schema only (480P tier, `duration`
  5 to 15, 848x480 bug noted in E2E wan.md).

| fal endpoint | Ours | Gap | What it takes |
|---|---|---|---|
| `fal-ai/wan/v2.2-5b/text-to-video` | **served (P0, §5)**: `fal-ai/wan` + `v2.2-5b/text-to-video` on `wan-max` with fal's fields and defaults (40 steps, CFG 3.5, shift 5), 580p/720p, 17 to 161 frames, 4 to 60 fps; interpolation refused. Was **partial**: same model as `wan-max`, on native, OpenAI and fal (H3 schema) | fal app `fal-ai/wan` + sub `v2.2-5b/text-to-video`, fal Wan fields (`num_frames`, `frames_per_second`, `negative_prompt`, `num_inference_steps`, `guidance_scale`, `shift`). **580p** tier. Frames 122 to 161. fps other than 24/16. Interpolation. Our default is 50 steps with CFG 5; fal's is 40 steps with CFG 3.5 | Fields and ids: **schema** (a Wan fal schema; our knobs already exist in `KnobCaps`). 580p: **config** (add a 576 short edge to `wan_max.short_edges`; check the 5B canvas multiple of 32). 161 frames: **config** (`frames_max`), quality unverified past 121. fps: **config** (`FpsCaps.allowed` is container-only, so any integer 4 to 60 is a muxing choice). Interpolation: **new weights** (RIFE or FILM), low priority |
| `fal-ai/wan/v2.2-5b/image-to-video` | **served (P0, §5)** as t2v plus `image_url` and aspect `auto`. Was **partial**: `wan-max` I2V is served (fal wire uses H3 field names) | as above, + aspect `auto` | as above |
| `fal-ai/wan/v2.2-5b/text-to-video/fast-wan` | **served (P0, §5)**: FastWan2.2 TI2V-5B FullAttn is the `wan-turbo` tier (weights on both volumes), 480p/580p/720p, 17 to 161 frames. Was **missing** (weights) | the whole tier: 480p, 580p, 720p at 24 fps, 17 to 161 frames | **new weights + config**: `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` (Apache-2.0). The preset `fast_wan_2_2_ti2v_5b` already exists in `fastvideo-models` (`wan/config.rs`), and `WanRecipe` takes a preset. This is the fal-comparable **wan-turbo** |
| `fal-ai/wan/v2.2-5b/text-to-video/distill` | **missing** | | Covered by fast-wan above; skip |
| `fal-ai/wan/v2.2-a14b/{text,image}-to-video` (+ `/turbo`, `/video-to-video`) | **missing** | Wan2.2 MoE 14B T2V and I2V (+ end image), 480p, 580p, 720p, 17 to 161 frames, two guidance scales | **new weights** (`Wan-AI/Wan2.2-T2V-A14B`, `Wan-AI/Wan2.2-I2V-A14B`, Apache-2.0) + **engine port**. The presets and `boundary_ratio` exist in `fastvideo-models`, but the engine-service Wan loader (`cuda/wan.rs`) has no two-expert path, and 2x14B bf16 needs fp8 or offload on one 96 GB GPU. v2v (`strength`) is a second port |
| `fal-ai/wan-t2v`, `fal-ai/wan-i2v` (2.1 14B) | **missing** | 14B at 480p, 580p, 720p, 81 to 100 frames | **new weights + config**: `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan2.1-I2V-14B-{480P,720P}-Diffusers` (Apache-2.0). Presets `wan_t2v_14b` and `wan_i2v_14b_*` exist. Lower value than A14B |
| `fal-ai/wan-flf2v` | **missing** | first + last frame | **new weights + engine port**: `Wan-AI/Wan2.1-FLF2V-14B-720P` (Apache-2.0) + last-frame conditioning |
| `…/lora` endpoints | **missing** | per-request LoRA | **engine port** (`GapId::Lora`) |
| `fal-ai/wan/v2.2-14b/speech-to-video` | **missing** | image + speech to talking video | **new weights + engine port**: `Wan-AI/Wan2.2-S2V-14B` (Apache-2.0), a new pipeline (audio encoder) |
| `fal-ai/wan-vace-14b` (+ apps) | **missing** | depth, pose, inpaint, outpaint, reframe, reference images | **new weights + engine port**: `Wan-AI/Wan2.1-VACE-14B` (Apache-2.0) + preprocessors |
| `fal-ai/wan/v2.2-14b/animate/*` | **missing** | character animation or replacement | **new weights + engine port** (Wan2.2 Animate-14B, Apache-2.0; not individually checked) |
| `fal-ai/krea-wan-14b/{text,video}-to-video` | **missing**. Our analogue is SF-Wan 1.3B (causal live), which fal does not host | 14B self-forcing | **new weights + engine port**: `krea/krea-realtime-video` (Apache-2.0) on the causal path (`sfwan` recipe at 14B) |
| `fal-ai/wan-pro`, `wan-25-preview`, `wan/v2.6`, `wan/v2.7`, `alibaba/wan-3.0*` | **missing** | | **not possible** (closed weights) |

## 4. Prioritized work list

**P0: config and schema only (no GPU work beyond a smoke run)**

1. **H3 app ids.** Mount `minimax/h3-max-turbo` (turbo tier) and
   `minimax/h3` (base: 480P and 768P, with 2K/4K refused as
   `H3Resolution2K`). This is a config change, or a one-line arm each in
   `adapters::fal_config`. Also consider making `minimax/h3-max-turbo` a
   default app.
2. **Per-family fal schemas and multi-segment subs.**
   - `Endpoint` today is H3's three single-segment subs. fal's LTX and Wan
     ids are `lightricks/ltx-2.5` + `text-to-video/fast` and `fal-ai/wan` +
     `v2.2-5b/text-to-video/fast-wan`.
   - Add an LTX schema (fields of §2.3 / `lightricks/ltx-2.5/*`, mapped
     through the existing LTX API validation, `models.rs` matrix) and a Wan
     schema (§3.1 5B fields).
   - Queue status and result URLs drop the sub-path (research-fal.md §2), so
     multi-segment subs only affect submit routes.
   - Fix the output file slug per app and tier (`…_minimax-h3.mp4` on every app,
     E2E wan.md).
   - This unlocks, on the fal wire: 720p, 1440p and 2160p; fps
     24/25/48/50; LTX `generate_audio`; LTX 6 to 20 s; and the Wan knobs.
3. **Wan 5B config.** Add a 576 (580p) short edge, `frames_max` 161 (gate on
   a quality check) and container fps 4 to 60.
4. **FastWan2.2 5B tier** (`fal-ai/wan/v2.2-5b/text-to-video/fast-wan`).
   Download `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` and add a
   catalog entry on the existing `fast_wan_2_2_ti2v_5b` preset (480p, 580p,
   720p; 17 to 161 frames; 24 fps). This makes our Wan turbo tier match
   fal's fast Wan.

**P1: engine ports on weights we already load**

5. **H3 ref2va**, in progress: targets in §1.3.
6. **LTX-2.5 I2V + keyframes**, in progress: targets in §2.3.
7. **H3 target audio (E10).** Unlocks `target_audio_url` (t2v/i2v), director
   `audio_url`, and most of `lip-sync/image-to-video` (except 2K).
8. **LTX `duration: "auto"`.** Load the LTX-2.5 duration head
   (`model_patches/ltx-2.5-duration-head-bf16.safetensors`, same repo and
   license); this clears `LtxAutoDuration`.
9. **LTX audio-to-video** (`lightricks/ltx-2.5/audio-to-video/{fast,pro}`):
   **done** 2026-09-29 on the distilled weights (row above).
10. **LTX extend + retake** (`fal-ai/ltx-2.3/{extend,retake}-video`), then
    **H3 extend-video** on the same video-context idea.

**P2: new weights + engine**

11. **Per-request LoRA and IC-LoRA** (`GapId::Lora`). First LTX
    `IC-LoRA-Ingredients` (2.5 build exists; this is the LTX reference-image
    mode), then the 2.5 effect IC-LoRAs, then `minimax/h3/*/lora`.
12. **Wan2.2 A14B T2V/I2V** (MoE expert switch in the engine-service loader,
    fp8/offload for 96 GB).
13. **Krea realtime 14B** on the causal path. This is the only fal
    self-forcing Wan.
14. **Wan2.1 FLF2V-14B, S2V-14B, VACE-14B, Animate-14B.** Each is a new
    pipeline; take them on demand.
15. **LTX dev checkpoint** (guided sampling, CFG/STG knobs) for the
    `ltx-2.3-22b/*` and `ltx-2.3-quality/*` parity. Low value.

**Not possible with open weights (refuse with the existing `GapId`, and document it)**

- H3 `1080P` as fal does it (latent refinement from 768P; we generate
  1080P natively instead, see §1) and H3 `2K`/`4K`
  (`H3Resolution2K`). A third-party open video upscaler could be added as
  new weights, but it would not match fal's output.
- H3 `camera-controls` and `styles/*`: fal's adapters are not published.
- LTX-2.5 `camera_motion` (the camera LoRAs exist only for LTX-2 19B;
  `static` could be accepted as a no-op).
- Wan 2.1 Pro, Wan 2.5, 2.6, 2.7 and Wan 3.0 (closed).
- Prompt expansion (`prompt_expansion_mode`, `enable_prompt_expansion`)
  needs an LLM service, not weights. It stays accepted and ignored until one
  is wired in.

## 5. P0 implementation (2026-09-28)

All four P0 items are in. Code: `crates/fastvideo-fal/src/{schema.rs,
schema/ltx.rs, schema/wan.rs, catalog.rs, queue.rs, sync.rs, lib.rs}`,
`crates/fastvideo-engine-service/src/cuda/caps.rs`, the route table
(`fastvideo-serve/src/router.rs`), the console (`console/{common,home,model,
form}.js`) and `configs/serve/runpod-{wan5b,wan,ltx}.toml`.

### 5.1 What the fal wire serves now

| App (`[protocols] fal_apps`) | Sub-paths | Runs on | Schema |
|---|---|---|---|
| `minimax/h3-max`, `minimax/h3-turbo`, `minimax/h3-draft` | `text-to-video`, `image-to-video`, `reference-to-video` (+ `director`) | their H3 tier | H3 Max (unchanged) |
| **`minimax/h3-max-turbo`** (default app) | same | H3 Turbo | H3 Max |
| **`minimax/h3`** (default app) | same | H3 Max | base H3: `resolution` `480P`, `768P`, `2K`, `4K`; 2K / 4K normalize to the 1440 / 2160 short edge and answer 422 `H3Resolution2K` on `resolution`; an omitted `resolution` is `768P` (fal's default `2K` is not servable) |
| **`lightricks/ltx-2.5`** | `text-to-video/fast`, `image-to-video/fast` → `ltx-turbo`; `text-to-video/pro`, `image-to-video/pro` → `ltx-pro` | per endpoint | §2.3 fields: `duration` (6 to 20 even, or `auto`), `resolution` (fast 720p/1080p/1440p/2160p, pro 720p/1080p), `aspect_ratio` (`16:9`, `9:16`; i2v `auto`), `fps` (fast 24/25/48/50, pro 24/25/50; default 25), `generate_audio`, `camera_motion`, `image_url` / `end_image_url` on i2v, `seed`, `sync_mode` |
| **`fal-ai/wan`** | `v2.2-5b/text-to-video`, `v2.2-5b/image-to-video` → `wan-max`; `v2.2-5b/text-to-video/fast-wan` → `wan-turbo` | per endpoint | §3.1 fields: `num_frames` 17 to 161 (81), `frames_per_second` 4 to 60 (24), `resolution` (580p/720p; fast-wan also 480p), `aspect_ratio` (`16:9`, `9:16`, `1:1`; i2v `auto`), `negative_prompt`, `num_inference_steps` 2 to 50 (40), `guidance_scale` 1 to 10 (3.5), `shift` 1 to 10 (5), `interpolator_model` / `num_interpolated_frames`, `enable_prompt_expansion`, `image_url` on i2v, `seed`, `enable_safety_checker`, `sync_mode` |
| any other `owner/alias` | the three H3 subs | its model by name | H3 Max (as before) |

- Sub-paths may have several segments. Submit and `/run/{app}/{sub}` are
  static routes per endpoint; status, result, stream and cancel answer under
  `/{app}/requests/{id}` (the URLs we return) and under every
  `/{app}/{sub}/requests/{id}`. A job belongs to the app its endpoint id
  names (`schema::app_id` drops the longest known sub).
- Output file names: `<nanoid21>_<slug>.mp4` from the app id
  (`schema::output_slug`: `minimax-<alias>` on `minimax/*`, else the alias,
  e.g. `_ltx-2.5.mp4`, `_wan.mp4`), plus the tier when the slug does not
  name it (the API-fixes work refined the slug on top of this).
- The director is mounted for the H3 apps only.
- `GET /fal/schema` lists each app's own endpoints (`sub`, `endpoint_id`,
  `title`, and the endpoint's `model` / `tier`), plus `kind` and `director`;
  `GET /fal/schema/{owner}/{alias}/{*sub}` serves the endpoint's JSON Schema.
  The console pages (`/console/models/{owner}/{alias}/{*sub}`) build their
  endpoint tabs from it; enum selects keep the JSON type (integer `fps`,
  `duration: 6 | "auto"`), number sliders step by 0.1.
- Default apps (`fastvideo_fal::DEFAULT_APPS`, `ProtocolsCfg`): the three
  H3 tiers plus `minimax/h3-max-turbo` and `minimax/h3`.
  `configs/serve/runpod-ltx.toml` adds `lightricks/ltx-2.5`; the new
  `configs/serve/runpod-wan5b.toml` serves `fal-ai/wan` (turbo resident,
  max on demand with `swap`).

Validation agreement (`crates/fastvideo-fal/src/catalog.rs` tests and
`crates/fastvideo-fal/tests/family_schemas.rs`): every schema enum value,
bound and default is what the parser accepts; every LTX (class ×
resolution × fps × duration) combination the parser accepts negotiates on
the CUDA catalog's LTX caps with the expected frames and delivered size;
every Wan (resolution × aspect × frames × fps) combination negotiates on
`wan22-ti2v-5b` and `fastwan22-ti2v-5b`; `minimax/h3` 2K / 4K give
`H3Resolution2K` on the `sol-h3` caps. The LTX schema's frame ceiling
(`LTX_FRAMES_MAX` = 481) is asserted equal to the engine's.

Deviations kept (each a server limit, named in the schema descriptions):

- LTX: fal allows 20 s at 25 fps and 10 s at 50 fps; both exceed our LTX
  grid (481 frames; 505 would be needed). They run at 481 frames (19.24 s
  and 9.62 s; `Snap::Nearest`, noted in the job log) rather than being
  refused, since the console offers duration and fps independently; a
  duration past fal's matrix (12 s at 1440p) runs at the matrix's longest.
  Raising the grid needs a GPU check of the LTX engine at 505 frames (not
  done here).
  **Owner decision (2026-09-29): over-length LTX clips run shorter.** A
  request past the LTX grid or fal's matrix is served at the longest clip
  that fits (the behaviour since `401a089`, `Snap::Nearest`, with the note
  in the job log), not refused. Revisit only if the engine is validated at
  505 frames.
- H3 1080P: fal lists `duration` 5 to 15 at every resolution; we serve
  1080P up to 5 s, and up to 10 s when the `h3_1080p_long` experimental
  feature is on (owner decision 2026-09-29; docs/serve/console.md §7). A
  longer 1080P request is a 422 on `duration` naming the limit and the
  flag; the served schema carries the cap as
  `x-fv-max-by-resolution: {"1080P": 5}` and the console narrows the
  duration slider when 1080P is chosen.
- Served schemas (`GET /fal/schema/...`, the console forms) list only what
  the endpoint's model serves: `h3-draft` 480P only (and no
  reference-to-video endpoint: no draft Ref2VA model), `minimax/h3` without
  2K / 4K, no 1080P where the GPU lacks the tier, LTX without `auto`,
  `ingredient` at 24/25/48/50 fps and up to 241 frames. The parsers still
  take fal's lists and answer a 422 naming the served values
  (`crates/fastvideo-serve/tests/dimension_sweep.rs` sweeps both).
- LTX `duration: "auto"` answers `LtxAutoDuration`; an omitted duration is
  6 s (fal's default is `auto`). `camera_motion` other than `static` answers
  `LtxCameraMotion`.
- LTX image-to-video: schema and normalization are in (`Task::I2V`, and
  `Task::Keyframes` with `end_image_url`); it runs where the LTX engine has
  image conditioning (the LTX image-conditioning work).
- Wan: `interpolator_model` `film` / `rife` with `num_interpolated_frames`
  > 0 is refused (no interpolator). `enable_prompt_expansion` and
  `enable_safety_checker` are accepted no-ops. On fast-wan,
  `guidance_scale` and `negative_prompt` are no-ops (DMD is unguided).
- Wan sizes: 16:9 `720p` is 1280x704 (the 5B's trained size, multiple of
  32), `580p` 1024x576, `480p` 832x480; `1:1` and `auto` keep the short
  edge 704 / 576 / 480.

### 5.2 Engine catalog (Wan 5B)

- `wan22-ti2v-5b` (`wan-max`): short edges 704, 576, 480; frames up to 161
  (4k+1); container fps any integer 4 to 60 (the frames do not depend on
  it). Every non-causal Wan recipe now accepts 4 to 60 fps.
- **`fastwan22-ti2v-5b` is `wan-turbo`**:
  `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` on the
  `fast_wan_2_2_ti2v_5b` preset, DMD 3 steps (1000/757/522), shift 5, no
  VSA, full Wan 2.2 VAE, same canvas / frames / fps as `wan-max`, T2V and
  I2V. `fastwan22-ti2v-5b-taehv` (TAEHV `taew2_2`) is `wan-draft`. All three
  5B tiers run without VSA, so they share one process.
- `fastwan21-1.3b` and `fastwan21-1.3b-taehv` are untiered ids now (they
  were `wan-turbo` / `wan-draft`); `configs/serve/runpod-wan.toml` and
  `configs/serve/gateway.toml` name `recipe = "fastwan21-1.3b"`. A
  `[[models]] family = "wan"` entry without `recipe` resolves to the 5B
  turbo.

### 5.3 Weights

`fastwan22-ti2v-5b` (Apache-2.0), repo revision
`3e187042a324f6f5fb68fd22110a78725253de8f`, 15 files, 24 201 770 562 bytes
(`scheduler/ tokenizer/ text_encoder/ transformer/ vae/` +
`model_index.json`), on **both** weight volumes:

- `scripts/gpu/weights-manifest.tsv` row and `verify-weights.sh` cell
  `fastwan22-ti2v-5b`.
- US (`s2k01690bi`): CPU pod `k756dt796jz3sx` pulled from the Hub into
  `weights/.fastwan22-ti2v-5b.partial-<stamp>` (`local_dir`, plain files),
  checked every file against the Hub listing at that revision (LFS SHA-256,
  git blob SHA-1 for the small files, exact file list), wrote `.complete`,
  then renamed to `weights/fastwan22-ti2v-5b` (94 s).
- EU (`jg48s6o1w0`): the same (CPU pod `ianwq90j5m11eo`, 166 s), also
  checked against the US copy's SHA-256 list: 15/15 identical. The EU tree
  is a Hub pull verified against the US hashes rather than a pod-to-pod
  byte copy: same bytes, one pod fewer.
- Script: `scripts/gpu/fetch-hub-tree.sh <dest> <revision>
  [expect-sha256.txt]` (add-only: refuses an existing `weights/<dest>`;
  detached backstop). Logs and SHA-256 lists:
  `artifacts/runpod/fetch-fastwan22-ti2v-5b-*`.
- Every GPU cell below passed the matrix's `verify-weights.sh` gate first
  (EU).

### 5.4 GPU results (RTX PRO 6000 Blackwell Server, EU volume, image `sha-9b2c963`)

Runs `artifacts/runpod/wan/9b2c963-09282019` (cells) and
`…/9b2c963-09282112` (control); upstream
`artifacts/runpod/upstream/9b2c963-09282013` (FastVideo image
`fastvideo-rs-upstream-fastvideo:sha-5c10f56`). No H100 or H200 was in
stock in US-CA-2.

Seconds; 5p = medians over the five `prompts-eval.json` prompts, warm;
else one prompt.

| Cell | Size × frames | Steps | Text | Denoise | Decode | Total | Peak MiB |
|---|---|---|---|---|---|---|---|
| `fw22` (wan-turbo, full VAE, 5p) | 1280x704 × 121 | DMD 3 | 0.06 | 4.53 | 13.87 | **18.50** | 61 196 |
| `fw22-taehv` (wan-draft, 5p) | 1280x704 × 121 | DMD 3 | 0.06 | 4.49 | 0.88 | **5.49** | 29 420 |
| Upstream FastVideo `fv-fastwan22-5b` (5p, dense) | 1280x704 × 121 | DMD 3 | 0.09 | 6.85 | 11.77 (+0.6 save) | **19.65** | 44 123 (torch) |
| `fw22-i2v` | 832x480 × 121 | DMD 3 | 0.02 | 1.57 | 5.67 | 7.31 | 39 396 |
| `fw22-580p-161f` | 1024x576 × 161 | DMD 3 | 0.00 | 3.68 | 11.29 | 15.03 | 47 660 |
| `fw22-720p-161f` | 1280x704 × 161 | DMD 3 | 0.00 | 6.85 | 18.29 | 25.20 | 60 044 |
| `wan5b-580p-161f` (wan-max, fal's defaults) | 1024x576 × 161 | UniPC 40, CFG 3.5 | 2.26 | 101.09 | 11.50 | 114.88 | 48 588 |
| `wan5b-720p-161f` (wan-max, fal's defaults) | 1280x704 × 161 | UniPC 40, CFG 3.5 | 1.60 | 191.23 | 18.53 | 211.41 | 58 892 |

- **580p and 161 frames run** on both 5B checkpoints (every cell exit 0,
  161 frames). Quality past the trained 121 frames was not scored.
- **wan-turbo against FastVideo**: 18.50 s vs 19.65 s total (denoise 4.53
  vs 6.85 s, 1.51x; our full-VAE decode, 13.9 vs 11.8 s, is the slower
  stage, as on the base 5B). wan-draft (TAEHV) 5.49 s; TAEHV against our
  full VAE on the same latents: LPIPS 0.030 to 0.066 (mean 0.046), PSNR
  30.4 to 32.9 dB.
- **Engine path**: `engine-fw22` loads the catalog's `wan-turbo` through
  the engine service (`fv-gpucheck engine --model wan-turbo`): Ready in
  110 s, frames **byte-identical** to the CLI (`cli-fw22`, SHA-256
  `e46b7b2c…`), MP4 1280x704, 121 frames at 24 fps; a cancel after step 1
  ends the job.
- **Module parity against Diffusers** (`fw22-oracle`, the upstream
  `oracle:fastwan22-ti2v` dump: `oracle_wan22.py` with the FastWan weights,
  DiT at t = 757): VAE encode rel-L2 5.4e-3, decode PSNR 65.7 dB (same VAE
  as the base); **DiT out rel-L2 6.75e-2 (t2v) / 6.86e-2 (i2v), cosine
  0.9977: over the harness's 5e-2 limit (FAIL)**. Control on the same GPU
  and image (`wan5b-oracle`, base TI2V-5B weights, t = 781): 2.62e-2 /
  1.45e-2, cosine 0.99966 (PASS, as on H100). Same network code and inputs,
  so the larger error comes with the distilled weights (larger activations
  through the bf16 path, or the timestep); not bisected. Frame-level parity
  with FastVideo's own clips was not measured (different noise).
- Spend: about $2.2 (GPU pods at $2.09/hr: 34 + 10 + 5 + 8 + 4 min; two
  CPU fetch pods at $0.24/hr, 2 to 3 min each). Every pod was deleted and
  checked gone (404).

### 5.5 Left from P0

- The distilled 5B's DiT rel-L2 (6.8e-2 vs 2.6e-2 for the base) needs a
  per-block look (`FASTVIDEO_DUMP_OPS`) before `wan-turbo` claims module
  parity; the tier runs and is as fast as FastVideo.
- LTX 20 s at 25 fps and 10 s at 50 fps (a 505-frame grid) and LTX `auto`.
