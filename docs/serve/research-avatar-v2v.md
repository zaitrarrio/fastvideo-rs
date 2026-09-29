# Research: avatars, video-to-video, and the FastVideo mode inventory

Status: research note, 2026-09-28. No code, no GPU pods, no downloads.

The owner asked three questions:

1. How do we add avatar support, real-time and batch, "like MagiHuman"?
2. How do we add V2V, real-time and batch?
3. Which other FastVideo modes should we consider?

Conventions:

- Every fact cites a URL or a repo path.
- **INFERRED** marks a conclusion that no source states directly.
- **UNVERIFIED** marks a claim taken from a secondary source (a blog, a search
  snippet, an aggregator), or a claim that could not be checked here.
- "Quality" columns only repeat what a model's authors claim, or which hosts
  serve the model. No clip was generated for this note.
- Hub metadata (license, gating, total repo bytes, last modified) comes from
  `https://huggingface.co/api/models/<repo>?blobs=true`, fetched 2026-09-28.
  Repo sizes are the whole repo, which is often more than one variant needs.
- fal input schemas come from
  `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<id>`, the same
  source that [research-fal.md](research-fal.md) uses. Prices come from the
  fal model pages, fetched 2026-09-28.
- `FV@442e2d2:` is `hao-ai-lab/FastVideo` at
  `442e2d2e18ed63b9a5d9f5bb6233ed5a884b3b9b` (main, 2026-09-28). It was read
  from a shallow code-only clone in the scratchpad.
- `MH@git:` is `GAIR-NLP/daVinci-MagiHuman` at its 2026-04-11 HEAD (merge of
  PR #41).
- Ours: fastvideo-rs at `cb87b2b`.

---

## 0. Answer first

### What MagiHuman is

daVinci-MagiHuman is published by SII-GAIR and Sand.ai under Apache-2.0. It
is a **15B joint audio+video generator**: one single-stream transformer
takes text (with the dialogue written inside the prompt), an optional
portrait, and noisy video and audio tokens, and it **generates the speech
and the lip-synced video together**. It is not, first of all, an
audio-driven lip-sync model. §1.1 gives the details.

- The code also has an **undocumented audio-conditioned path**
  (`--audio_path`). fal exposes it as an optional `audio_url`.
- MagiHuman is **clip-based, not streaming**. It is fast enough (a 5 s 256p
  clip in 2 s on one H100) to serve as a clip queue, the way FastH3 is served
  on Reactor.
- FastVideo already has a bit-exact port of it. That port can be our
  reference.

### The recommendation in one line

Build **one missing piece of infrastructure**, WebRTC ingest (client video
and microphone tracks into the engine). Then take the **cheap wins on
weights and paths we already serve**:

- LTX-2.5 audio-to-video;
- a script-driven LTX "talking photo" matching Reactor's `ltx` avatar model;
- LTX retake/extend;
- causal real-time V2V on the SF-Wan 1.3B path.

After that, add **MagiHuman** as the first new avatar family, and a
**Wan2.1-1.3B causal talking-head model** as the real-time avatar.

### Top 3 picks per area

| Area | #1 | #2 | #3 |
|---|---|---|---|
| **Avatar, batch** | **LTX-2.5 audio-to-video** (image + speech → video; same weights we serve; fal `lightricks/ltx-2.5/audio-to-video`, LTX API `/audio-to-video`) | **daVinci-MagiHuman** (T2AV/TI2AV + audio-conditioned; Apache; FastVideo reference port; fal `fal-ai/davinci-magihuman`) | **InfiniteTalk** (image or video + audio, unlimited length, dubbing; Apache; Wan2.1-I2V-14B; fal `fal-ai/infinitalk*`) |
| **Avatar, real-time** | **LTX "script avatar"** (photo + script → streamed speech+video in overlapping windows; the Reactor `ltx` model contract; reuses LTX I2V + audio) | **LiveTalk-1.3B** or **SoulX-FlashHead-1.3B** (Apache, Wan2.1-1.3B, block-causal with KV cache, audio-driven; claimed real-time on one consumer GPU; runs on our SF-Wan causal rollout) | **MagiHuman distill 256p/540p as a clip queue** (faster than real time per clip; Reactor FastH3-style playout) |
| **V2V, batch** | **LTX retake + extend** (same LTX weights; fal `fal-ai/ltx-2.3/{retake,extend}-video`, LTX API) | **LTX IC-LoRAs**: Ingredients (in progress), then 2.3 Union-Control on our `ltx23` weights and the 2.5 effect LoRAs (fal `ltx-2.3-22b/reference-video-to-video`, `ltx-2.3-quality/*`) | **Wan2.1 VACE** (1.3B, then 14B; Apache; fal `fal-ai/wan-vace-14b*`) |
| **V2V, real-time** | **StreamDiffusionV2 1.3B** on the SF-Wan 1.3B causal path (Apache; same Wan2.1-1.3B causal DiT) | **Krea realtime 14B** (Apache; self-forcing Wan2.1-14B; T2V + V2V; fal `fal-ai/krea-wan-14b/*`) | **SANA-Streaming 2B** (Apache; 1280×704 at 24 fps end to end on one RTX 5090 per the paper; Reactor already hosts it; new family) |
| **FastVideo modes** | **MagiHuman** (T2AV, TI2AV) | **LTX-2 streaming continuation** (Dreamverse "vibe directing"; maps to fal director and Reactor) | **Wan V2V / Wan2.2-S2V / Wan2.2-Animate** (V2V is on main; S2V and Animate are open PRs) |

The roadmap with effort, GPU cost, licences and API surfaces is in §5.

---

## 1. What we start from (ours, `cb87b2b`)

| Area | State | Source |
|---|---|---|
| Served clip models | H3 (T2V, I2V, keyframes, ref2v), LTX-2.5 (T2V, I2V, keyframes), Wan2.2 TI2V-5B, FastWan 5B/1.3B | [fal-parity.md](fal-parity.md) §0 "What we serve today", `crates/fastvideo-engine-service/src/cuda/caps.rs` |
| Causal real-time | SF-Wan 1.3B (`SfWanRecipe`: `local_attn_frames`, `sink_frames`, `block_frames`) over `wan::stream::CausalRollout` | `caps.rs`, `crates/fastvideo-engine-service/src/cuda/causal.rs` |
| WebRTC **output** | H.264 (NVENC) and VP8 encode, Opus encode; Reactor, fal director (WMA) and native WHIP transports | `crates/fastvideo-media/src/{video,vp8,opus}.rs`, `crates/fastvideo-webrtc` |
| WebRTC **input** | **None.** `fastvideo-webrtc/src/host.rs` says "our server peers only send". Reactor `PublishTrack` answers `publish_refused` ("the model declares no input track") | `crates/fastvideo-reactor/src/gateway.rs:241-244` |
| Decoders we could reuse for ingest | An Opus **decoder** exists (`OpusDecoder`, used in tests). There is no H.264/VP8 decoder | `crates/fastvideo-media/src/opus.rs:239-248` |
| Tasks in the protocol | `Task::{A2V, Extend, Retake, V2V}` exist and are always refused as unsupported (since 2026-09-29 `A2V`, `Retake` and `Extend` are served on LTX-2.5; `V2V` is still refused) | `crates/fastvideo-protocol/src/{request.rs:147-153, negotiate.rs:360}` |
| LTX A2V / retake / extend | Refused (`403 permission_error` on the LTX API, `GapId::LtxEndpoint`); all three served since 2026-09-29 (P0-2, P0-4) | [design.md](design.md) §"LTX endpoints", fal-parity §4 P1 #9-#10 |
| H3 target audio (E10) | Refused (`H3TargetAudio`); would unlock fal `minimax/h3-max/lip-sync/image-to-video` | fal-parity §1.2, design.md E10 |
| LTX IC-LoRA | The Ingredients IC-LoRA is on both volumes (manifest row `ltx25-ic-lora-ingredients`). Serving is in progress | [../ports/ltx-ref2v.md](../ports/ltx-ref2v.md), `scripts/gpu/weights-manifest.tsv` |
| Wan control and edit presets | `wan_fun_1_3b_control` and `lucy_edit_dev` exist, but only as a **one-frame approximation**: the control or edit image is VAE-encoded into latent frame 0. Upstream Lucy-Edit concatenates a whole **video** latent on the channel axis (`FV@442e2d2:fastvideo/pipelines/basic/wan/lucy_edit_pipeline.py` docstring). **Not faithful** | `crates/fastvideo-cudarc/src/wan/pipeline.rs:810-890` |
| Wan VAE encoder | Present (I2V, TI2V). V2V needs it per frame chunk | same file, `encode_i2v_condition`, `encode_first_frame` |
| Stable Audio (Oobleck VAE) | Port in tree (T2A) | [../ports/stable-audio.md](../ports/stable-audio.md) |
| Weights on both volumes that matter here | `wan21-t2v-14b`, `wan22-ti2v-5b` (includes the Wan2.2 VAE), `sfwan21-1.3b`, `fastwan21-1.3b`, `ltx25`, `ltx23` (LTX-2.3 distilled), `ltx2` | `scripts/gpu/weights-manifest.tsv` |

**INFERRED:** every real-time V2V item, and any avatar driven from a live
microphone, is blocked on **ingest**. That means:

- decoding client video tracks (H.264 or VP8) and audio tracks (Opus) into
  per-track ring buffers;
- Reactor `PublishTrack`, WMA client tracks (fal documents webcam input
  tracks: "Register `track` callbacks … (for example, the browser webcam)",
  https://fal.ai/docs/documentation/development/wma), and a native WHIP
  ingest.

That is why ingest is P0-1 in §5.

---

## 2. A. Avatars

### 2.1 daVinci-MagiHuman: exact facts

| Item | Value | Source |
|---|---|---|
| Publisher | SII-GAIR (GAIR-NLP) and Sand.ai | https://github.com/GAIR-NLP/daVinci-MagiHuman, README header |
| Paper | "Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model", arXiv 2603.21986, submitted 2026-03-23 | https://arxiv.org/abs/2603.21986 |
| Repos | code https://github.com/GAIR-NLP/daVinci-MagiHuman; weights https://huggingface.co/GAIR/daVinci-MagiHuman; demo https://huggingface.co/spaces/SII-GAIR/daVinci-MagiHuman; Docker `sandai/magi-human:latest` | README |
| License | **Apache-2.0** (code and weights). **Its dependencies carry other licences**: T5Gemma is under the Gemma terms, with a manual gate on the Hub; Stable Audio Open 1.0 is under the Stability AI Community License, with an auto gate. See §6 | Hub API for `GAIR/daVinci-MagiHuman`, `google/t5gemma-9b-9b-ul2`, `stabilityai/stable-audio-open-1.0` |
| Architecture | 15B, 40 layers, single-stream self-attention only (no cross-attention). "Sandwich" layout: the first and last 4 layers are modality-specific and the middle 32 are shared. Timestep-free denoising. Per-head sigmoid gating | README "Architecture" |
| Components | Text encoder `google/t5gemma-9b-9b-ul2`. Audio VAE **Stable Audio Open 1.0**. Video VAE **Wan2.2-TI2V-5B VAE**. Optional **Turbo VAE** decoder (`TurboV3-Wan22-TinyShallow`). Latent-space SR DiTs for 540p and 1080p | README "Download Model Checkpoints"; `MH@git:example/distill/config.json` |
| Inputs | Prompt (the recommended "enhanced prompt" holds a `Dialogue:` block with the spoken lines and a `Background Sound:` block). Optional `--image_path` for TI2V. **Optional `--audio_path` ("reference audio")**: when set, the audio is encoded through the SA VAE, the frame count follows the audio length, and `is_a2v=True` is passed to the denoiser (`MH@git:inference/pipeline/video_generate.py:271-280`). The README does not document this mode | README "Input Modes", `MH@git:inference/pipeline/entry.py:42-48` |
| Outputs | mp4 with muxed audio. 25 fps by default (`config.py:119`). Duration `--seconds` (4 in the examples). Frames = seconds × 25 + 1 | `MH@git:inference/common/config.py`, `pipeline.py` |
| Resolutions | Base at 256p (448×256 in the examples, 480×256 in the FastVideo preset), SR to 540p or 1080p | README; `FV@442e2d2:fastvideo/pipelines/basic/magi_human/presets.py` |
| Steps | Base 32 steps (FlowUniPC, CFG 2 per FastVideo's notes). Distill (DMD-2) 8 steps, no CFG. SR 5 steps | `config.py`; `FV@442e2d2:…/magi_human/AGENTS.md` |
| Speed (5 s clip, one H100) | 256p **2.0 s** (base 1.6 + decode 0.4); 540p **8.0 s**; 1080p **38.4 s** | README "Inference Speed" |
| Real-time | **No streaming or autoregressive mode.** The 256p clip is 2.5× faster than playback on an H100, so a **clip queue** can keep up. **INFERRED** from the speed table | README; arXiv abstract |
| Quality (authors' claims) | Win rate 80.0% vs Ovi 1.1 and 60.9% vs LTX 2.3 over 2,000 pairwise comparisons; WER 14.60% (LTX 2.3: 19.23%) | README "Performance" |
| Languages | Mandarin, Cantonese, English, Japanese, Korean, German, French | README |
| GPU | Hopper (FlashAttention Hopper build, MagiCompiler; MagiAttention only for SR-1080p). Consumer GPUs need CPU offload (`--offload_config.*`, e.g. RTX 5090). FastVideo: "both fit on a single 80 GB GPU" | `MH@git:example/distill/run_TI2V.sh` comments; `FV@442e2d2:…/magi_human/JOURNAL.md` |
| Weights | Upstream repo **216.1 GB**: `base` 31.6 GB, `distill` ~61 GB, `540p_sr` ~62 GB, `1080p_sr` ~62 GB, `turbo_vae` 1.87 GB. The FastVideo conversion `FastVideo/MagiHuman-Diffusers` (214.4 GB, Apache) stores each transformer at **30.64 GB** (bf16). `sr_540p/sr_transformer` is 61.2 GB. Plus T5Gemma 9B-9B (repo 40.7 GB; FastVideo notes ~18 GB are needed) | Hub API |
| fal endpoint | `fal-ai/davinci-magihuman` (category image-to-video). Required: `prompt`, `image_url`. Optional: **`audio_url`**, `resolution` `256p`/`540p`/`720p`/`1080p` (default 256p), `duration` 1 to 30 (default 5), `num_inference_steps`, `guidance_scale` 0 to 20 (default 5), `seed`, `enable_safety_checker`. **$0.05 per second** | fal OpenAPI; https://fal.ai/models/fal-ai/davinci-magihuman |
| FastVideo | Full port of all 4 variants (base, distill, sr_540p, sr_1080p) × 2 modes (T2V, TI2V), with a 14-test parity battery and "bit-exact parity" claims. The A2V branch is **out of scope** there (`JOURNAL.md:403`). The registry activation (#1302, #1751) is **still open**, so the model is not in `registry.py` or the support matrix | `FV@442e2d2:fastvideo/pipelines/basic/magi_human/`; https://github.com/hao-ai-lab/FastVideo/pulls (#1280, #1295-#1302, #1751) |

Related Sand.ai model: **MAGI-2 Preview** (`sand-ai/MAGI-2-preview`,
Apache-2.0, 306.7 GB). It is a 114B MoE with 6B active parameters that does
T2V and I2V with audio, 10 s clips only, plus a 1080p refiner, and it
"requires 8 Hopper GPUs" (HF README). The FastVideo port PR #1686 is open.
Not a fit for our one-GPU pods.

### 2.2 Open avatar models: batch (audio-driven, clip generation)

Reuse key for the "Reuse" column:

- **W1.3** = our Wan2.1-1.3B DiT (FastWan / SF-Wan);
- **W14** = the Wan2.1-14B DiT (`wan_t2v_14b` preset, weights on both
  volumes);
- **W14-I2V** = the Wan2.1-I2V-14B preset (CLIP image encoder, 36 channels;
  weights not on the volumes);
- **W5** = Wan2.2 TI2V-5B;
- **LTX** = the LTX-2.x port;
- **A** = a new audio encoder (wav2vec2 in almost every case).

| Model | Publisher / license | Base | Inputs → output | Real-time | Claimed quality / adoption | Weights | Reuse |
|---|---|---|---|---|---|---|---|
| **daVinci-MagiHuman** | SII-GAIR + Sand.ai; **Apache-2.0** (deps: Gemma terms, SAO community) | own 15B single-stream; Wan2.2 VAE; SAO VAE | text (+ image) → video **+ speech**; optional audio | clip queue only | §2.1; hosted on fal | 214-216 GB | Wan2.2 VAE ✓, SAO VAE ✓ (stable-audio port), T5Gemma ✗, DiT ✗ (new family) |
| **LTX-2.x audio-to-video** | Lightricks; **LTX-2 community license** (free below USD 10M annual revenue, ltx-ref2v.md §2) | LTX-2.3 / 2.5 | audio (+ image, + prompt) → video (a talking photo when the image is a portrait) | no | fal `lightricks/ltx-2.5/audio-to-video/{fast,pro}`, `fal-ai/ltx-2.3/audio-to-video` ($0.10/s) | **none new** | **served since 2026-09-29** (P0-2) |
| **LTX-2.3 IC-LoRA DubIt ("LipDub")** | Lightricks; LTX-2 community; **gated** | LTX-2.3 22B | source video + new dialogue text → re-dubbed video+audio (V2V) | no | validated on 2.3, "not validated on LTX-2.5 yet" (https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta) | small LoRA (size not read: gated) | LTX-2.3 ✓ (`ltx23` on volumes) + IC-LoRA pipeline (shared with Ingredients) |
| **InfiniteTalk** | MeiGen-AI; **Apache-2.0** | Wan2.1-I2V-14B (480P per upstream docs, **UNVERIFIED**) | image **or video** + audio → talking video of unlimited length (sparse-frame dubbing) | no | fal `fal-ai/infinitalk` $0.20/s at 480p, doubled at 720p; `/video-to-video`; `/single-text` (TTS voices) | 168.6 GB repo (many variants) | W14-I2V + A |
| **MultiTalk** | MeiGen-AI; Apache-2.0 | Wan2.1-I2V-14B | image + 1-2 audio tracks → multi-person talk | no | fal `fal-ai/ai-avatar{,/multi}` is **deprecated** (page notice) | 80.7 GB | W14-I2V + A; superseded by InfiniteTalk (same team) — **INFERRED** |
| **OmniAvatar** | OmniAvatar; Apache-2.0 (14B card) | LoRA + audio projection on **Wan2.1-T2V-14B** (1.3B variant too) | image + audio + prompt → full-body talk | no; 16 s/it at 14B on A800 (card) | not on fal (404) | **1.2 GB** (LoRA) + base | **W14 ✓ (weights on volumes)** + LoRA + A: the cheapest 14B avatar in download terms |
| **FantasyTalking** | acvlab; Apache-2.0 | Wan2.1-I2V-14B-720P | image + audio | no | not on fal (404) | 3.4 GB + base | W14-I2V + A |
| **StableAvatar** | FrancisRing; **MIT** | Wan2.1-Fun-V1.1-1.3B-InP (1.3B basic) | image + audio → infinite length; 512², 480×832, 832×480 | no | fal `fal-ai/stable-avatar` $0.10/s, **deprecated** | 28.1 GB | W1.3 (Fun-InP 1.3B has a preset: `wan_fun_1_3b_inp`) + A |
| **EchoMimicV3** | Ant Group; Apache-2.0 | Wan2.1-Fun-1.3B-InP | image + audio (+ text) multi-task | no | fal `fal-ai/echomimic-v3` $0.20/s | 7.1 GB | W1.3 + A |
| **Wan2.2-S2V-14B** | Wan-AI; Apache-2.0 | own 14B (Wan2.2 family) | image + speech (+ pose video) → video, 480P/720P | no | fal `fal-ai/wan/v2.2-14b/speech-to-video` $0.10/0.15/0.20 per s (fal-parity §3.1); FastVideo PR #1683 open; `FastVideo/Wan2.2-S2V-14B-Diffusers` (58.4 GB) exists | 49.1 GB | Wan DiT blocks partly; audio encoder + injection new |
| **Wan2.2-Animate-14B** | Wan-AI; Apache-2.0 | Wan2.2-I2V-A14B arch | driving video + character image → animate or replace (V2V avatar) | no | fal `…/animate/{move,replace}` $0.04-0.08/s; FastVideo PR #1765 open | 72.4 GB | A14B MoE path (not in the engine-service loader yet, fal-parity §3.2) |
| **HunyuanVideo-Avatar** | Tencent; **Tencent Hunyuan Community License: "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"** | HunyuanVideo 13B | image + audio | no | fal `fal-ai/hunyuan-avatar` $1.40 per 5 s | 80.8 GB | none (HunyuanVideo 1 is not served). **Excluded on licence** |
| **Hallo3** | Fudan; MIT card, but "a fine-tuned derivative … based on the CogVideo-5B I2V model" (CogVideoX licence applies) | CogVideoX-5B-I2V | image + audio | no | — | 52.2 GB | none. **Licence flag** |
| **LongCat-Video-Avatar** | Meituan; **MIT** | LongCat-Video 13.6B | audio+text→video, audio+text+image→video, audio-driven continuation; single and multi person | no | — | 128.6 GB | LongCat spec/preset in tree ([../ports/longcat.md](../ports/longcat.md)); avatar conditioning new |
| **SkyReels-V3-A2V-19B** | Skywork; `skywork-license` (custom; commercial terms not read) | "19B-720P"; base not stated on the card (**UNVERIFIED**) | talking avatar | no | Skywork's own API platform | 56.0 GB | unknown |
| **SkyReels-A3** | Skywork | DiT | portrait + voice (+ text) → talking video; re-dub | no | press release and project page only; **open weights not found** (Hub `Skywork/SkyReels-A3` returns no repo) | — | — |
| **Ovi** | Character.AI (repo `chetwinlow1/Ovi`); Apache-2.0 | Wan2.2-TI2V-5B + audio tower | text (+ image) → video + speech (T2AV, like MagiHuman) | no | MagiHuman's claimed 80% win rate is against Ovi 1.1 | 70.0 GB | W5 ✓ + a new audio tower |
| Closed, for API parity only | ByteDance OmniHuman v1.5 (fal $0.16/s), VEED Fabric 1.0 ($0.08-0.15/s), Kling AI Avatar v2, sync.so lipsync-2 (V2V) | — | image + audio (OmniHuman adds `mask_url`, `turbo_mode`, 720p/1080p) | — | — | closed | not possible |

### 2.3 Open avatar models: real-time and streaming

"RT" is the authors' throughput claim. It is not a measurement by us.

| Model | License | Base | Mode | RT claim (authors) | Weights | Reuse |
|---|---|---|---|---|---|---|
| **LiveTalk-1.3B** (GAIR-NLP, same lab as MagiHuman; arXiv 2512.23576) | Apache-2.0 | Wan2.1-T2V-1.3B + OmniAvatar; wav2vec2-base-960h | image + audio + text; block-AR, 3 latent frames per block, KV cache; an optional audio-LM "thinker/talker" for conversation | **24.82 FPS, 0.33 s first-frame latency**; ≥ 24 GB GPU (RTX 4090, A800, H800 tested); 16 fps output | 5.7 GB | **W1.3 causal ✓✓**: our SF-Wan `CausalRollout` (local attention, sink, block frames) is the same pattern; wav2vec2 + audio cross-attention new |
| **SoulX-FlashHead-1.3B** (Soul AI Lab; arXiv 2602.07449) | Apache-2.0 | Wan2.1 1.3B; wav2vec2-base-960h; the Lite variant uses the **LTX-Video VAE** | image + audio, infinite streaming talking head | Lite: **96 FPS on one RTX 4090**, or 3 concurrent streams at 25+ FPS. Pro: 10.8 FPS on a 4090, 25+ FPS on 2× RTX 5090 | 14.3 GB | W1.3 ✓; the Lite VAE is a different (LTX-Video 0.9) VAE, which is **new** |
| **LiveAvatar** (Alibaba Quark; arXiv 2512.04677, ECCV 2026) | Apache-2.0 | **LoRA on Wan2.2-S2V-14B**; 4-step, block-AR | image + audio, 10,000+ s | **45 FPS on 5× H800**; one GPU needs ≥ 80 GB, or 48 GB with FP8 (v1.1) | 1.4 GB + S2V base | needs S2V first (§2.2) |
| **SoulX-FlashTalk-14B** (arXiv 2512.23379) | Apache-2.0 | InfiniteTalk / Wan 14B; self-correcting bidirectional distillation | image + audio | **32 FPS, 0.87 s start on 8× H800**; one GPU > 64 GB | 54.5 GB | W14-I2V; the multi-GPU target does not fit our one-GPU pods |
| **Hallo-Live** (Fudan; arXiv 2604.23632) | MIT (Ovi base: Apache) | Ovi (Wan2.2-5B dual-stream) | **text → video + speech, streaming** (joint AV, like a streaming MagiHuman) | **20.38 FPS, 0.94 s latency on 2× H200** | 46.7 GB | W5 ✓ + Ovi audio tower (new) |
| **StreamAvatar AROD** | `license: other` | DyStream teacher → one-step blockwise student (audio-to-motion + renderer) | audio → motion → rendered frames | not read | 0.4 GB | not a video DiT; no reuse |
| OmniMate (arXiv 2607.23023, CC-BY-4.0 paper), Vorch-Streamer (arXiv 2608.05663) | — | — | real-time AV avatars | not read | **weights not found** | — |

Hosted real-time avatars (for API matching):

- **Reactor `ltx`**: "Turns a photo and a script into a video-and-audio take
  of that person speaking, lip-synced in one generation pass."
  - Commands: `set_avatar_image`, `set_script`, `set_wpm`, `start`, plus an
    optional scene/delivery prompt, duration and seed.
  - Output: 640×352 at 24 fps with 48 kHz stereo on one clock. At most 300 s
    per take. Generated in overlapping 20 s windows at a 10 s offset.
  - "No separate text-to-speech step."
  - Price is served dynamically and was not read.
  - Sources: https://docs.reactor.inc/model-api-reference/ltx and
    https://www.reactor.inc/models.
  - **INFERRED:** this is LTX I2V with joint audio. The dialogue goes in the
    prompt, and the take is extended window by window. Our LTX-2.5 I2V + audio
    + the continuation idea covers it.
- **Reactor `h3-reference-turbo-realtime`** (MiniMax): listed on
  https://www.reactor.inc/models. Details not read. H3 is licence-excluded
  for us in the US, EU, UK and KR (§6).
- **fal `fal-ai/live-avatar`**:
  - Inputs: `image_url`, `audio_url`, `prompt`, `frames_per_clip` 16 to 80,
    `num_clips` 1 to 100, `acceleration`. $0.01 per video second.
  - The page says "This endpoint is deprecated".
  - Its underlying model is not named. **UNVERIFIED** that it is Quark
    LiveAvatar.
- **fal realtime surface**: WMA WebRTC. The only WMA app listed is H3 Max
  Director (https://fal.ai/wma). There is no WMA avatar app.
- **Decart**: its pricing page lists no avatar or lip-sync model
  (https://docs.platform.decart.ai/getting-started/pricing).

### 2.4 fal avatar endpoints (schemas to match)

`*` marks a required field. Prices are from the model pages on 2026-09-28.

| fal id | Inputs | Price | Our route |
|---|---|---|---|
| `fal-ai/davinci-magihuman` | *prompt, *image_url, audio_url, resolution 256p/540p/720p/1080p, duration 1-30, steps, guidance_scale, seed | $0.05/s | MagiHuman (P1-1) |
| `lightricks/ltx-2.5/audio-to-video/{fast,pro}` | *audio_url (2-20 s; pro ≤ 10), image_url, prompt (required without image), guidance_scale (fal-parity §2.1) | see fal-parity | LTX A2V (P0-2) |
| `fal-ai/ltx-2.3/audio-to-video` | *audio_url, image_url, prompt, guidance_scale, aspect_ratio auto/16:9/9:16 | $0.10/s | LTX A2V (P0-2) |
| `fal-ai/infinitalk` | *image_url, *audio_url, *prompt, resolution 480p/720p, num_frames 41-721, acceleration, seed | $0.20/s 480p, ×2 720p | InfiniteTalk (P1-5) |
| `fal-ai/infinitalk/video-to-video` | *video_url, *audio_url, *prompt, num_frames 41-241 | same | InfiniteTalk (P1-5) |
| `fal-ai/infinitalk/single-text` | *image_url, *text_input, *voice (fixed list), *prompt | same | needs TTS; out of scope |
| `fal-ai/wan/v2.2-14b/speech-to-video` | *image_url, *audio_url, *prompt, frames 40-120, fps 16, 480p/580p/720p, steps, guidance, shift | $0.10-0.20/s | Wan2.2-S2V (P2) |
| `fal-ai/wan/v2.2-14b/animate/{move,replace}` | *video_url, *image_url, resolution, steps, use_turbo | $0.04-0.08/s | Animate (P2) |
| `fal-ai/echomimic-v3` | *image_url, *audio_url, *prompt, frames per generation 49-161, audio_guidance_scale | $0.20/s | optional 1.3B batch |
| `fal-ai/hunyuan-avatar` | *image_url, *audio_url, text, frames 129-401, turbo_mode | $1.40 per 5 s | licence-excluded |
| `fal-ai/bytedance/omnihuman/v1.5`, `veed/fabric-1.0`, `fal-ai/kling-video/ai-avatar/v2/standard` | *image_url, *audio_url (+ prompt, mask, resolution) | $0.16/s, $0.08-0.15/s, not read | closed |
| `fal-ai/sync-lipsync/v2`, `fal-ai/latentsync` | *video_url, *audio_url (V2V lip-sync) | not read | closed / LatentSync 1.6 is OpenRAIL++ |
| `minimax/h3-max/lip-sync/image-to-video` | image_url + audio_url (≥ 5 s, clipped at 14.8 s) | $0.05-0.32 per clip | H3 E10; **licence-excluded in US/EU/UK/KR** |

**INFERRED common denominator:** `image_url` + `audio_url` + `prompt` (+
`resolution`, `seed`, a frame count or duration). One native `A2V` task
(`AudioRole::Drive`, already in design.md §"AudioRole") can back every one of
these schemas.

### 2.5 Avatar reuse analysis

- **Zero new weights:** LTX-2.5 A2V, and the LTX "script avatar" (I2V +
  generated speech). Both run on `weights/ltx25`.
- **W14 already on volumes:** OmniAvatar-14B is a 1.2 GB LoRA plus audio
  projection on Wan2.1-T2V-14B. We still need the 14B engine path to be
  served and a wav2vec2 encoder.
- **W1.3 causal path:** LiveTalk and SoulX-FlashHead are both Wan2.1-1.3B
  distilled to block-causal few-step generation. That is exactly what our
  `CausalRollout` runs for SF-Wan.
  - The new work is the audio encoder, the audio cross-attention (or
    injection) blocks, and the reference-image conditioning.
  - **INFERRED:** it is the cheapest real-time avatar on a single GPU.
- **New family but a clean reference:** MagiHuman. FastVideo's bit-exact
  port gives us an oracle. The same method served H3 and LTX:
  `scripts/gpu/upstream`.

---

## 3. B. Video-to-video

### 3.1 Batch V2V

| Model / mode | License | Base | What it does | Hosted | Weights | Reuse |
|---|---|---|---|---|---|---|
| **LTX retake** | LTX-2 community | LTX-2.x | regenerate a time window of an input video: `replace_audio`, `replace_video` or both; `start_time`, `duration` 2-20 | fal `fal-ai/ltx-2.3/retake-video`; LTX API `/retake` | none new | **LTX ✓✓** (`retake.py` upstream, ltx-ref2v.md §1). **Served 2026-09-29** (P0-4) |
| **LTX extend** | same | LTX-2.x | extend at the start or end, with `context` and `duration` 2-20 | fal `fal-ai/ltx-2.3/extend-video`; LTX API `/extend` | none new | **LTX ✓✓**; shares video-context conditioning with keyframes (served). **Served 2026-09-29** (P0-4; upstream reference LTX-Desktop's retake pipeline) |
| **LTX IC-LoRA control** | LTX-2 community | **2.3** (Union-Control 0.7 GB, Motion-Track 0.3 GB) and **2.0** (Pose, Depth, Canny, Detailer). **No 2.5 control build is listed** in the `Lightricks` org (API listing 2026-09-28) | reference video + preprocessor (depth, canny, pose) drives structure | fal `fal-ai/ltx-2.3-22b/reference-video-to-video` (`ic_lora_type` union/match_preprocessor/detailer, `num_frames` 9-481, 40 steps) | 0.3-0.7 GB each | `ltx23` weights **on volumes** + the IC-LoRA pipeline (Ingredients work) + per-request LoRA (`GapId::Lora`) + preprocessors |
| **LTX IC-LoRA effects** | LTX-2 community | **2.5 builds** (2026-09-10/14): Clean-Plate, Colorization, Day-To-Night, Deblur, Decompression, Water-Simulation, Pixel-Spatial-Upscaler, Ingredients. **2.3 only**: HDR, Relight, In-Outpainting, Instant-Shave, Cross-Eyed, DubIt | per-effect V2V | fal `fal-ai/ltx-2.3-quality/*` | ~1.3 GB each (Ingredients: 1.31 GB) | LTX-2.5 ✓ + the same IC-LoRA pipeline |
| **Wan2.1 VACE** | Apache-2.0 | own 1.3B / 14B (Wan2.1 + VACE context blocks) | one model: pose, depth, inpaint, outpaint, reframe, reference images, first/last frame | fal `fal-ai/wan-vace-14b` (+ `/depth`, `/pose`, `/inpainting`, `/outpainting`, `/reframe`), `fal-ai/wan-vace-apps/video-edit`; $0.04-0.08/s | 19.0 GB (1.3B), 75.1 GB (14B) | W1.3 / W14 + VACE blocks. **FastVideo has no VACE** (support matrix: "not currently supported", issue #1435), so no FastVideo oracle; use Wan's own repo |
| **Wan Fun Control** | Apache-2.0 | Wan2.1-Fun 1.3B/14B, Wan2.2-Fun 5B/A14B (+ Camera) | control video (canny, depth, pose, MLSD, trajectory) + optional start image | — | 19.8 GB (2.1 1.3B), 24.2 GB (2.2 5B), 69.0 GB (A14B) | W1.3 / W5. Our `wan_fun_1_3b_control` is a one-frame stub (§1). FastVideo supports `IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers` (and shows Wan2.2-Fun-A14B-Control in its example) |
| **Wan SDEdit V2V** (`strength`) | Apache-2.0 | any Wan | noise an input video to `strength`, then denoise with a new prompt | fal `fal-ai/wan/v2.2-a14b/video-to-video` (`strength` 0.9) | none new on 5B | W5 ✓ + VAE encode of the whole video. FastVideo `WanVideoToVideoPipeline` (`FV@442e2d2:…/wan/wan_v2v_pipeline.py`) |
| **Lucy-Edit Dev / 1.1-Dev** | **Decart non-commercial licence** | Wan2.2-TI2V-5B, video latent concatenated on channels | instruction editing (clothes, objects, style) | fal `decart/lucy-edit/{pro,fast}` ($0.10/0.15 per s, 480p/720p; hosted Pro is closed) | 34.2 GB | W5 ✓ (preset stub exists), but **non-commercial** |
| **Ditto (Editto)** | **CC-BY-NC-SA-4.0** | Wan2.1-VACE-14B | instruction V2V editing | — | 25.4 GB | non-commercial |
| **Wan2.2-Animate** | Apache-2.0 | A14B arch | move or replace a character | fal §2.4 | 72.4 GB | A14B path |
| **InfiniteTalk V2V dubbing**, **LTX-2.3 DubIt** | Apache / LTX community | §2.2 | lip re-dub of an existing video | fal `infinitalk/video-to-video` | — | §2.2 |
| **LongCat-Video continuation** | MIT | LongCat 13.6B | video continuation (KV cache on the conditioning frames) | — | 83.3 GB (`meituan-longcat/LongCat-Video`); `FastVideo/LongCat-Video-VC-Diffusers` 86.3 GB | LongCat spec in tree |
| **SkyReels-V3-V2V-14B** | `skywork-license` | 14B-720P | video extension | Skywork API | 69.0 GB | unknown base (**UNVERIFIED**) |

### 3.2 Real-time V2V (camera or stream in, edited stream out)

| Model | License | Base | Claimed speed (authors) | Weights | Reuse |
|---|---|---|---|---|---|
| **StreamDiffusionV2** (Oct 2025; https://streamdiffusionv2.github.io/) | Apache-2.0 (Hub `jerryfeng/StreamDiffusionV2`, `daydreamlive/StreamDiffusionV2`) | **Wan2.1-T2V-1.3B** (and 14B), CausVid-style causal DMD (`wan_causal_dmd_v2v` checkpoint); motion-aware noise from inter-frame MSE; rolling KV cache with **sink tokens**; Stream-VAE (4 frames → 1 latent, cached features) | 1.3B **64.52 FPS**, 14B 58.28 FPS on **4× H100**; first frame < 0.5 s; single GPU for 1.3B (RTX 4090/5090, A100). The README example is 832×480, 16 fps, `--step 2` | 39.9 GB repo (1.3B + 14B; the 1.3B size alone was not read) | **W1.3 causal ✓✓**: sink + rolling KV cache is our SF-Wan design (`sink_frames`, `local_attn_frames`). New: per-block input-latent noising, the Wan VAE encoder per chunk (exists), motion-aware noise, ingest |
| **Krea realtime 14B** | Apache-2.0 | Wan2.1-T2V-14B, Self-Forcing; KV-cache recomputation and attention bias | **11 fps at 4 steps on one B200**; ~1 s to first frame; V2V from "real videos, webcam inputs, or canvas primitives" | 114.4 GB repo | W14 ✓ (base on volumes) on the causal path at 14B; fal `fal-ai/krea-wan-14b/{text,video}-to-video` (V2V: *prompt, *video_url, `strength` 0.85) $0.025/s |
| **SANA-Streaming** (NVIDIA; arXiv 2605.30409) | Apache-2.0 (`Efficient-Large-Model/SANA-Streaming`, 9.5 GB) | SANA-Video hybrid DiT (linear Gated-DeltaNet + some softmax blocks), Cycle-Reverse regularisation, FP4/FP8 mixed precision | **1280×704, 24 FPS end to end (DiT 58 FPS) on one RTX 5090** | 9.5 GB | **new family**. Reactor serves it as `sana-streaming` (webcam or file in; `setPrompt`, `setMode live|file`; ~1 s prompt-to-edit latency) |
| **Helios** (PKU-Yuan) | Apache-2.0 | Wan2.1-T2V-14B fine-tune | **19.5 FPS on one H100** (T2V/I2V); minute-scale | 137.8 GB (Distilled) | W14 ✓. Reactor serves `helios` with "image/video" input. Helios's own V2V support is **UNVERIFIED** |
| **LongLive** (NVIDIA) | 1.3B: **CC-BY-NC-SA-4.0** (`daydreamlive/LongLive-1.3B`) | Wan 1.3B causal | interactive prompt switching (T2V) | — | W1.3; non-commercial |
| **Self-Forcing / SF-Wan V2V** | Apache-2.0 (`gdhe17/Self-Forcing`) | Wan2.1-1.3B causal | the paper is T2V. V2V is an **INFERRED** SDEdit-style extension (StreamDiffusionV2 is the published version of it) | — | ours already |
| Closed: **Decart Lucy 2.5** (realtime $0.02/s, fast $0.04/s), **Lucy Restyle 2** ($0.01/s), **MirageLSD**; Reactor **X2** (XMAX: `source` track, `set_prompt`, `set_reference_image`, `set_pointer`, 832p 24 fps, "private preview") | — | — | Decart pricing page; https://docs.reactor.inc/model-api-reference/x2 | — | API shapes only |

### 3.3 Real-time V2V wire contracts

| Surface | Input | Controls | Our gap |
|---|---|---|---|
| **Reactor** | a client **input track** (`PublishTrack` on `source`, or a webcam track), decoded to RGB `InputFrame`s in a per-track ring buffer ([research-reactor.md](research-reactor.md) §4.4) | model commands: `set_prompt`, `set_reference_image`, … (X2, SANA-Streaming) | we answer `publish_refused`; no decoder |
| **fal WMA** | client tracks via `track` callbacks ("the browser webcam") | JSON on the `control` data channel updates `session_params` in place | our director is output-only, clip models only |
| **Native** | WHIP ingest (the inverse of our WHIP egress) | our `/fv/v1/streams` control messages | not designed |

**INFERRED design:**

- Decode with an ffmpeg subprocess (the mirror of `vp8.rs` and the NVENC
  path), or with NVDEC later.
- Put the decoded frames in a ring buffer shared by the causal driver.
- Each causal block pulls the newest `block_frames` × 4 source frames,
  VAE-encodes them, and noises them to the session's `strength`.
- Microphone audio goes through the existing `OpusDecoder` at 48 kHz, then is
  resampled to 16 kHz for wav2vec2 (avatars).

---

## 4. C. FastVideo inventory (`FV@442e2d2`)

Sources:

- `FV@442e2d2:docs/inference/support_matrix.md`, the registered IDs table;
- `FV@442e2d2:fastvideo/models/wan/definition.py`;
- `FV@442e2d2:fastvideo/pipelines/basic/*`;
- the `FastVideo` Hub org listing;
- the FastVideo PR list.

Legend for the "Ours" column:

- **S**: served by `fv-serve`.
- **R**: preset registered in our tree (docs/scope.md). Weight parity and
  serving are **not checked by this note**; several families are scaffolds
  per [../ports/registry-status.md](../ports/registry-status.md).
- **—**: absent.

Legend for the "Cost" column:

- **have**;
- **cheap**: on an engine path or weights we already serve;
- **medium**: a known architecture plus new conditioning;
- **new**: a new family.

| FastVideo family / mode | Checkpoints | Modes | Ours | Cost |
|---|---|---|---|---|
| Wan2.1 T2V 1.3B / 14B | `Wan-AI/Wan2.1-T2V-{1.3B,14B}-Diffusers`, `FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers` | T2V (VSA) | R (14B weights on volumes) | cheap (14B serve) |
| Wan2.1 I2V 14B | `Wan-AI/Wan2.1-I2V-14B-{480P,720P}-Diffusers` | I2V | R | medium (weights + CLIP encoder) |
| Wan2.1-Fun 1.3B InP / Control | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers`, `IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers` | I2V, **control V2V** | R (control: one-frame stub) | medium |
| FastWan (sparse distillation, DMD) | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`, `…-14B-480P…`, `FastVideo/FastWan2.2-TI2V-5B{,-FullAttn}-Diffusers` | T2V, TI2V, VSA | **S** (1.3B, 5B) | have |
| FastWan-QAD (quantization-aware distillation) | `FastVideo/FastWan-QAD-1.3B`, `-SA2`, `-FP8-1.3B` ("5 s of video in 1.8 s E2E", README 2026/06/23) | T2V | R (same arch as FastWan 1.3B; ships unquantized, scope.md) | cheap |
| Wan2.2 TI2V-5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | T2V, I2V | **S** | have |
| Wan2.2 A14B (MoE) | `Wan-AI/Wan2.2-{T2V,I2V}-A14B-Diffusers` | T2V, I2V | R | medium (MoE switch, fp8/offload, fal-parity §3.2) |
| Lucy-Edit | `decart-ai/Lucy-Edit-Dev`, `-1.1-Dev` | **V2V edit** | R (stub) | medium; **non-commercial** |
| Wan V2V | `WanVideoToVideoPipeline` (strength) | **V2V** | — | cheap on 5B |
| Self-Forcing causal Wan | `wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers`, `rand0nmr/SFWan2.2-T2V-A14B-Diffusers`, `FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers`; also `FastVideo/CausalWan2.2-I2V-A14B-{Preview,OdeInit-Preview}`, `FastVideo/CausalForcingWan2.1-T2V-1.3B` | **causal / streaming** T2V, I2V | **S** (SF-Wan 1.3B); R (A14B) | cheap-medium (A14B causal = real-time I2V) |
| TurboDiffusion (rCM + SLA) | `loayrashid/TurboWan2.1-T2V-{1.3B,14B}-Diffusers`, `TurboWan2.2-I2V-A14B-Diffusers` | T2V, I2V | R (`is_rcm`, `wan/sla.rs`) | cheap |
| Wan2.2-S2V-14B | `FastVideo/Wan2.2-S2V-14B-Diffusers` (58.4 GB) | **S2V (avatar)** | — | new-ish; PR #1683 **open** |
| Wan2.2-Animate-14B | (PR) | **animate / replace** | — | medium; PR #1765 **open** |
| MiniMax H3 / FastH3 | `MiniMaxAI/MiniMax-H3`, `FastVideo/FastVideo-FastH3-8-Step-V2`, `…-4-step-Preview-v1-*` (Dense/VSA, DataFree/Synthetic, LoRA, MLX INT4/6/8), `…-Minimax-FastH3-Preview-v0.{1,2}` | T2VA, FL2VA, Ref2VA | **S** | have (**licence-excluded in US/EU/UK/KR**) |
| LTX-2 / 2.3 | `FastVideo/LTX2{,.3}-Distilled-Diffusers`, `Lightricks/LTX-2{,.3}`, `FastVideo/LTX-2.3-OmniNFT-LoRA`, `FastVideo/LTX2-OmniNFT-LoRA` | T2AV, I2V, **streaming continuation** (`ltx2/continuation.py`, used by Dreamverse) | S (2.5); R (2.0/2.3) | cheap (continuation) |
| **daVinci-MagiHuman** | `FastVideo/MagiHuman-Diffusers` (base, distill, sr_540p, sr_1080p) | **T2AV, TI2AV (avatar)** | — | **new** (§2.1); registry PR #1302/#1751 open |
| MAGI-2 Preview | `sand-ai/MAGI-2-preview` | T2AV, I2AV, 1080p refine | — | new; 8 GPUs; PR #1686 open |
| MMAudio | `FastVideo/MMAudio-large-44k-v2-Diffusers` | V2A, T2A | R (Wan soundtrack stage) | have (**non-commercial**) |
| Stable Audio Open 1.0 / small | `FastVideo/stable-audio-open-{1.0,small}-Diffusers` | T2A, A2A, inpaint (`basic_stable_audio_{a2a,inpaint}.py`) | R | cheap (the MagiHuman audio VAE) |
| LongCat-Video | `FastVideo/LongCat-Video-{T2V,I2V,VC}-Diffusers` (+ distilled and refinement LoRAs) | T2V, I2V, **video continuation** | R (T2V) | medium |
| HunyuanVideo / FastHunyuan | `hunyuanvideo-community/HunyuanVideo`, `FastVideo/FastHunyuan-diffusers` | T2V | — | new; Tencent licence |
| HunyuanVideo 1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-*` (480p/720p T2V, I2V distilled, 1080p SR) | T2V, I2V, SR | R | medium; Tencent licence (**check territory**) |
| Kandinsky 5.0 | `kandinskylab/Kandinsky-5.0-{T2V,I2V}-{Lite,Pro}-*` | T2V, I2V | R (T2V) | medium |
| Cosmos Predict2 / 2.5, GEN3C | `nvidia/Cosmos-Predict2-2B-Video2World`, `KyleShao/Cosmos-Predict2.5-2B-Diffusers`, `nvidia/Cosmos-Predict2.5-14B`, `FastVideo/GEN3C-Cosmos-7B-Diffusers` | T2W, I2W, **V2W** (video → future), camera-controlled novel view | R | medium |
| World / action models | Matrix-Game 2.0 (causal, self-forcing variants) and 3.0, HunyuanGameCraft, HY-WorldPlay (bidirectional, AR), LingBot-World (Base-Cam, Base-Act, Fast), LingBot-World-2 14B causal-fast, DreamX-World 5B (Cam, AR), `FastVideo/Waypoint-1-Small-Diffusers` | action/camera-driven streaming I2V | R (most) | medium each; these are Reactor-style "world model" products |
| LingBot-Video | `FastVideo/LingBot-Video-{Dense-1.3B,MoE-30B-A3B}-Diffusers` | T2V | R | medium |
| Image models | FLUX.1-dev, FLUX.2 (dev, klein 4B/9B), SD3.5-medium, Z-Image-Turbo, GLM-Image | T2I (GLM also edit) | R | — |
| FastMetal-QAD (MLX) | `FastVideo/FastMetal-{1.3B,5B,14B}-QAD` | Apple Silicon T2V | scaffold (`fastvideo-mlx`) | out of scope for serving |
| Training-side techniques | VSA, sparse distillation, DMD2, Self-Forcing causal distillation, Attn-QAT, LoRA fine-tuning | — | VSA ✓, DMD ✓, causal DMD ✓ | — |
| **Not in FastVideo** | **VACE** (issue #1435), InfiniteTalk/MultiTalk/OmniAvatar, StreamDiffusionV2, Krea realtime, SANA-Streaming | — | — | no FastVideo oracle; use each upstream repo |

**Top FastVideo modes to add** (beyond the avatar and V2V picks):

1. **MagiHuman**. It is the only FastVideo avatar mode on main, and it has a
   parity-grade reference.
2. **LTX-2 streaming continuation.** This is the Dreamverse mode: segment N+1
   is conditioned on segment N's trailing frames and audio latents
   (`FV@442e2d2:fastvideo/pipelines/basic/ltx2/continuation.py`). It is the
   engine behind (a) the Reactor `ltx` avatar and (b) an LTX director on fal
   WMA. It reuses served LTX-2.5.
3. **Wan V2V and the causal A14B I2V.**
   - `WanVideoToVideoPipeline` gives the fal `…/video-to-video` shape.
   - `SFWan2.2-I2V-A14B` / `CausalWan2.2-I2V-A14B` give real-time I2V
     (start the stream from a user photo).
   - Cheaper alternatives: FastWan-QAD 1.3B and TurboDiffusion. Both are
     already registered, so they are a speed/quality comparison, not new
     modes.

---

## 5. Roadmap

Estimates are **INFERRED** from this repo's own history:

- a config-plus-schema item such as fal-parity P0 cost about $2.2 of GPU;
- an engine port on loaded weights such as H3 ref2v took several sessions
  and cost about $6.5 of GPU plus fetch pods (h3-ref2v.md §8.5);
- GPU rates: RTX PRO 6000 at $2.09/h, H200 at $4.59/h.

"Session" means one agent session (a working day of an agent with build-pod
access). Every download over a few GB needs the owner's approval (CLAUDE.md).
Every new tree goes to both volumes and into `weights-manifest.tsv` and
`verify-weights.sh`.

### P0: infrastructure plus wins on weights we already serve

| # | Item | Effort | GPU $ | New weights | License | API surfaces |
|---|---|---|---|---|---|---|
| P0-1 | **WebRTC ingest.** Client video (H.264/VP8 → RGB via an ffmpeg subprocess) and audio (Opus → PCM, decoder exists) tracks into per-session ring buffers. Reactor `PublishTrack`/`UnpublishTrack` and input-track capabilities; WMA client tracks; native WHIP ingest | 2-3 sessions | ~$2-5 (mostly CPU/loopback tests; one GPU E2E) | none | — | Reactor, fal WMA, native |
| P0-2 | **DONE 2026-09-29.** **LTX-2.5 audio-to-video** (`Task::A2V`, `AudioRole::Drive`): audio → audio latents held clean as conditioning, optional first-frame image. This is the batch talking photo. Served on the distilled weights (the upstream audio mechanism of `a2vid_two_stage.py` on `DistilledPipeline`); GPU oracle at the bf16 floor, serve E2E pass on LTX v2, native and fal, lip-sync proxy in sync (docs/oracle.md "LTX-2.5 audio-to-video", docs/serve/e2e/ltx.md "Audio-to-video"). Spend $1.9. Upstream's guided variant (dev DiT + CFG/STG) would need `transformer_full/` (38.0 GB, not downloaded) | 1 session | $1.9 | none | LTX-2 community (below USD 10M revenue) | fal `lightricks/ltx-2.5/audio-to-video/{fast,pro}`; LTX API `/v1\|v2/audio-to-video`; native `audio_url`. `fal-ai/ltx-2.3/audio-to-video` not mounted (same contract, a 2.3 app id) |
| P0-3 | **LTX "script avatar", real-time.** Portrait + script → joint speech+video, streamed as overlapping windows via LTX continuation (the Reactor `ltx` contract: `set_avatar_image`, `set_script`, `set_wpm`, `start`; 640×352 at 24 fps, 48 kHz stereo, ≤ 300 s). Batch variant: I2V with the dialogue prompt | 2-3 sessions | ~$10-20 | none | LTX-2 community | Reactor (new model schema), native; fal WMA later |
| P0-4 | **LTX retake + extend** | 2-3 sessions | ~$10-20 | none | LTX-2 community | fal `fal-ai/ltx-2.3/{retake,extend}-video`; LTX API `/retake`, `/extend` |
| P0-5 | **Real-time V2V on SF-Wan 1.3B**, StreamDiffusionV2-style. First a training-free SDEdit on the SF-Wan checkpoint, then the `wan_causal_dmd_v2v` checkpoint (download to be approved). Adds per-block input noising, motion-aware noise and a prompt command. Needs P0-1 | 3-4 sessions | ~$15-30 | StreamDiffusionV2 1.3B checkpoint (size of the 1.3B file not read; the repo is 39.9 GB with 14B) | Apache-2.0 | Reactor (`source` input track + `set_prompt`, like SANA-Streaming/X2), fal WMA, native WHIP |

### P1: first new avatar family, real-time avatar, and control V2V

| # | Item | Effort | GPU $ | New weights | License | API surfaces |
|---|---|---|---|---|---|---|
| P1-1 | **daVinci-MagiHuman** (distill 256p + SR 540p first; 1080p later). New DiT (single-stream, sandwich, per-head gating), T5Gemma-9B encoder, Turbo VAE decoder, SR DiT. Reuses the Wan2.2 VAE (on volumes) and the SA Oobleck VAE (stable-audio port). Oracle: the FastVideo port. Also wire the A2V branch (`--audio_path`) to match fal's `audio_url`. Serve real-time as a FastH3-style clip queue | 5-7 sessions | ~$40-80 (H100/H200 or RTX PRO 6000 96 GB; Hopper is the upstream target) | distill 30.6 + sr_540p ~92 (sr + base transformer) **or** sr_1080p 61 + turbo 1.9 + T5Gemma ~18-41 + SAO VAE (small): **~110-170 GB, needs approval** | Apache-2.0 + **Gemma terms** (T5Gemma, manual gate) + **Stability Community License** (SAO VAE; free below USD 1M revenue per Stability's terms, **UNVERIFIED** here) | fal `fal-ai/davinci-magihuman`; native; Reactor clip queue; fal director (clip model) |
| P1-2 | **Real-time audio-driven avatar on W1.3 causal**: LiveTalk-1.3B or SoulX-FlashHead-1.3B. First a 1-session bake-off in upstream Python (~$5), then port the winner. New: wav2vec2-base-960h encoder, audio cross-attention, reference image. Mic via P0-1 | 4-5 sessions (+1 bake-off) | ~$25-45 | 5.7 GB (LiveTalk) or 14.3 GB (FlashHead) + wav2vec2 (~0.4 GB, **UNVERIFIED** size) | Apache-2.0 (both) | Reactor (image command + mic input track), fal WMA, native |
| P1-3 | **LTX IC-LoRA V2V**: finish Ingredients, then the 2.3 **Union-Control** on our `ltx23` weights (depth, canny, pose + preprocessors), then the 2.5 effect LoRAs. Needs per-request LoRA (`GapId::Lora`) | 3-5 sessions | ~$20-40 | 0.3-1.3 GB per LoRA; preprocessor models (licences to check: pose, depth; canny needs none) | LTX-2 community | fal `ltx-2.3-22b/reference-video-to-video`, `ltx-2.3-quality/*`; native |
| P1-4 | **Wan2.1 VACE** (1.3B first on W1.3, then 14B on W14) | 4-6 sessions | ~$30-60 | 19 GB (1.3B), 75 GB (14B): **approval** | Apache-2.0 | fal `fal-ai/wan-vace-14b*`, `wan-vace-apps/video-edit`; native |
| P1-5 | **InfiniteTalk** (image or video + audio, unlimited length). Needs the W14-I2V path (CLIP encoder, 36 channels) | 4-6 sessions | ~$40-80 | Wan2.1-I2V-14B (~70 GB, **UNVERIFIED**) + InfiniteTalk (selected files of 168.6 GB): **approval** | Apache-2.0 | fal `fal-ai/infinitalk`, `/video-to-video`; native |
| P1-6 | **Krea realtime 14B** (T2V + V2V) on the causal path at 14B. One GPU gives about 11 fps on a B200 (authors), so it is **not 24 fps real time on one GPU** | 3-5 sessions | ~$30-60 (B200/H200) | selected files of 114.4 GB: **approval** | Apache-2.0 | fal `fal-ai/krea-wan-14b/{text,video}-to-video`; Reactor/WMA streaming |

### P2: on demand

| Item | Why P2 | License | Surface |
|---|---|---|---|
| Wan2.2-S2V-14B, then the LiveAvatar LoRA | 14B single-GPU memory (≥ 80 GB, 48 GB with FP8); real-time needs 5× H800; FastVideo PR still open | Apache-2.0 | fal `wan/v2.2-14b/speech-to-video`; Reactor |
| Wan2.2-Animate-14B | needs the A14B MoE path first | Apache-2.0 | fal `…/animate/{move,replace}` |
| SANA-Streaming 2B | best claimed real-time edit quality per GPU (24 fps at 1280×704 on a 5090), but a new family (linear Gated-DeltaNet attention, FP4 kernels) | Apache-2.0 | Reactor (`sana-streaming` schema exists), fal WMA |
| Hallo-Live (streaming T2AV avatar) | needs the Ovi audio tower; 2× H200 for the claimed speed | MIT / Apache | Reactor |
| LongCat-Video-Avatar, LongCat continuation | 13.6B new family; our LongCat is a spec/scaffold | MIT | native, fal-style A2V |
| Wan Fun Control (faithful, video latents) | overlaps VACE and IC-LoRA control | Apache-2.0 | native |
| Wan SDEdit V2V on 5B | small, but low value next to LTX retake | Apache-2.0 | fal `wan/v2.2-a14b/video-to-video` shape |
| Helios 14B | real-time T2V/I2V at 19.5 FPS on an H100; V2V unverified | Apache-2.0 | Reactor |
| OmniAvatar-14B | cheap download (1.2 GB LoRA) on W14, but no host sells it | Apache-2.0 | native |
| **Skip** | HunyuanVideo-Avatar (**territory: not EU/UK/KR**); Hallo3 (CogVideoX licence); Lucy-Edit, Ditto, LongLive-1.3B (**non-commercial**); SoulX-FlashTalk, MAGI-2 (multi-GPU only); SkyReels-A3 (no open weights found); closed OmniHuman, Fabric, Lucy 2.5, MirageLSD, X2 | — | — |

Totals (**INFERRED**):

- P0: ~11-16 sessions and ~$45-95 of GPU, with no new weights except the
  StreamDiffusionV2 1.3B checkpoint.
- P1: ~24-35 sessions and ~$190-370 of GPU, plus roughly 0.3-0.5 TB of
  approved downloads per volume.

Keep the $8 balance floor in mind: each GPU cell should stay under the
existing 5-minute and per-pod backstops.

---

## 6. Licence flags

| Model / component | Licence | Flag |
|---|---|---|
| MiniMax H3 (all our H3 tiers, H3 lip-sync, H3 reference realtime) | MiniMax licence | **Excluded in the US, EU, UK and KR (owner's note).** H3-based avatar modes are not a US/EU product |
| MMAudio | non-commercial | **Non-commercial.** Do not use it for avatar soundtracks or dubbing in a paid product |
| LTX-2.x and all IC-LoRAs (DubIt, Ingredients and control are gated) | LTX-2 community licence | free below USD 10M annual revenue; a commercial licence above (ltx-ref2v.md §2) |
| daVinci-MagiHuman | Apache-2.0 | its **T5Gemma** encoder is under the Gemma terms (manual Hub gate); its **Stable Audio Open 1.0** VAE is under the Stability AI Community License (auto gate) |
| Wan2.1/2.2 (T2V, I2V, TI2V, S2V, Animate, VACE, Fun), InfiniteTalk, MultiTalk, OmniAvatar, FantasyTalking, EchoMimicV3, LiveTalk, SoulX-FlashHead/FlashTalk, LiveAvatar, StreamDiffusionV2, Krea realtime, Helios, SANA-Streaming, Self-Forcing | Apache-2.0 | OK. Keep NOTICE/attribution |
| StableAvatar, Hallo-Live, LongCat-Video(-Avatar) | MIT | OK |
| Hallo3 | MIT card on a CogVideoX-5B derivative | CogVideoX licence terms apply (**not read**) |
| HunyuanVideo-Avatar (and HunyuanVideo / 1.5) | Tencent Hunyuan Community License | "DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA" (HunyuanVideo-Avatar LICENSE); the 1.5 licence was **not read** |
| Lucy-Edit Dev, Ditto, LongLive-1.3B | non-commercial / CC-BY-NC-SA | **Non-commercial** |
| SkyReels V3 | `skywork-license` | **not read**; review before use |
| LatentSync 1.6 | OpenRAIL++ | use-based restrictions |

---

## 7. Open questions and unverified points

1. **MagiHuman A2V quality.** The `--audio_path` branch exists in the
   upstream code and fal exposes `audio_url`, but the README does not
   document it, and FastVideo's port left it out of scope. Compare it against
   T2AV in the upstream Python before porting it.
2. **The fal MagiHuman 720p option** has no upstream SR checkpoint at 720p.
   **INFERRED:** fal probably resizes from the 1080p or 540p path.
3. **Real-time claims** (LiveTalk 24.82 FPS, FlashHead 96 FPS Lite,
   StreamDiffusionV2, SANA-Streaming 24 FPS on a 5090, Helios 19.5 FPS,
   Krea 11 fps on a B200) are the authors'. None was measured on our RTX PRO
   6000. The P1-2 bake-off should record FPS and first-frame latency on it.
4. The **StreamDiffusionV2 1.3B checkpoint size** was not read (repo total
   39.9 GB). Check that the architecture is identical to `sfwan21-1.3b`
   before claiming "zero-new-code" reuse.
5. **Reactor `ltx` model version** (2.3 vs 2.5) is not stated in its docs.
6. **Decart avatar products:** none appear on the pricing page. There may be
   non-public ones (**UNVERIFIED**).
7. **fal `live-avatar` and `ai-avatar`** are marked deprecated. fal's avatar
   market is concentrated in InfiniteTalk, OmniHuman, Fabric, EchoMimic,
   Kling and now MagiHuman.

---

## Sources

MagiHuman:

- https://github.com/GAIR-NLP/daVinci-MagiHuman
- https://huggingface.co/GAIR/daVinci-MagiHuman
- https://arxiv.org/abs/2603.21986
- https://fal.ai/models/fal-ai/davinci-magihuman
- https://huggingface.co/FastVideo/MagiHuman-Diffusers
- https://huggingface.co/sand-ai/MAGI-2-preview
- https://github.com/hao-ai-lab/FastVideo/pulls (#1280, #1302, #1683, #1686, #1751, #1765)

Avatar models:

- https://huggingface.co/MeiGen-AI/InfiniteTalk
- https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk
- https://huggingface.co/OmniAvatar/OmniAvatar-14B
- https://huggingface.co/acvlab/FantasyTalking
- https://huggingface.co/FrancisRing/StableAvatar
- https://huggingface.co/BadToBest/EchoMimicV3
- https://github.com/antgroup/echomimic_v3
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- https://huggingface.co/Wan-AI/Wan2.2-Animate-14B
- https://huggingface.co/tencent/HunyuanVideo-Avatar (LICENSE)
- https://huggingface.co/fudan-generative-ai/hallo3
- https://huggingface.co/meituan-longcat/LongCat-Video-Avatar
- https://huggingface.co/Skywork/SkyReels-V3-A2V-19B
- https://www.prnewswire.com/news-releases/day15-skyreels-a3-the-art-of-natural-speech-for-digital-humans-302526394.html
- https://huggingface.co/chetwinlow1/Ovi

Real-time avatars:

- https://github.com/GAIR-NLP/LiveTalk
- https://huggingface.co/GAIR/LiveTalk-1.3B-V0.1
- https://github.com/Soul-AILab/SoulX-FlashHead
- https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B
- https://github.com/Soul-AILab/SoulX-FlashTalk
- https://github.com/Alibaba-Quark/LiveAvatar
- https://huggingface.co/Quark-Vision/Live-Avatar
- https://arxiv.org/abs/2604.23632
- https://huggingface.co/fudan-generative-ai/Hallo-Live
- https://huggingface.co/pancx/StreamAvatar-AROD
- https://arxiv.org/abs/2607.23023

V2V:

- https://streamdiffusionv2.github.io/
- https://huggingface.co/jerryfeng/StreamDiffusionV2
- https://huggingface.co/krea/krea-realtime-video
- https://arxiv.org/abs/2605.30409
- https://huggingface.co/Efficient-Large-Model/SANA-Streaming
- https://huggingface.co/BestWishYsh/Helios-Distilled
- https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control
- https://huggingface.co/decart-ai/Lucy-Edit-Dev
- https://huggingface.co/QingyanBai/Ditto_models
- https://huggingface.co/api/models?author=Lightricks (IC-LoRA list)
- https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta

Hosts:

- fal queue OpenAPI per endpoint (§2.4, §3)
- https://fal.ai/models/fal-ai/infinitalk
- https://fal.ai/models/fal-ai/ai-avatar
- https://fal.ai/models/fal-ai/stable-avatar
- https://fal.ai/models/fal-ai/echomimic-v3
- https://fal.ai/models/fal-ai/hunyuan-avatar
- https://fal.ai/models/fal-ai/live-avatar
- https://fal.ai/models/decart/lucy-edit/pro
- https://fal.ai/models/fal-ai/ltx-2.3/audio-to-video
- https://fal.ai/wma
- https://fal.ai/docs/documentation/development/wma
- https://www.reactor.inc/models
- https://docs.reactor.inc/model-api-reference/overview
- https://docs.reactor.inc/model-api-reference/ltx
- https://docs.reactor.inc/model-api-reference/sana-streaming
- https://docs.reactor.inc/model-api-reference/x2
- https://docs.platform.decart.ai/getting-started/pricing

FastVideo:

- https://github.com/hao-ai-lab/FastVideo at `442e2d2` (`docs/inference/support_matrix.md`, `fastvideo/models/wan/definition.py`, `fastvideo/pipelines/basic/{magi_human,wan,ltx2,longcat}/`, `apps/dreamverse/`)
- https://huggingface.co/api/models?author=FastVideo
