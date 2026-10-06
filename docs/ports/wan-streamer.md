# Wan-Streamer (Tongyi Lab): port assessment

Status: research note, 2026-09-29. No code, no GPU pods, no downloads.

The owner asked whether and how to implement Wan-Streamer, announced in
https://tongyilab.substack.com/p/wan-streamer-a-native-streaming-model.

Conventions (as in [../serve/research-avatar-v2v.md](../serve/research-avatar-v2v.md)):

- Every fact cites a URL or a repo path.
- **INFERRED** marks a conclusion that no source states directly.
- **UNVERIFIED** marks a claim from a secondary source (a blog, an
  aggregator, a search snippet) or one that could not be checked here.
- **NOT DISCLOSED** means the three papers, the post and the project site
  were read in full and do not state it.
- Sources read on 2026-09-29: the Substack post; arXiv 2606.25041v2 (v0.1),
  2607.04443 (v0.2), 2607.15038 (v0.3), read as full text from the arXiv
  HTML; https://wan-streamer.com/ and its v0.1, v0.2 and v0.3 pages; the
  Hugging Face model list of the `Wan-AI` org
  (`https://huggingface.co/api/models?author=Wan-AI`); the GitHub
  `Wan-Video` org page.

---

## 0. Answer first

**Wan-Streamer cannot be ported today: there are no weights and no code.**
It is a research preview of a closed model. Three technical reports
(v0.1 June 2026, v0.2 July 5, v0.3 July 16) describe the design at a high
level. None gives a parameter count, a GPU type, a VAE design or a step
count, and none says anything about a release.

**It is also not a Wan video model in the sense of our Wan port.** It is
not Wan2.1/2.2 with a causal schedule, like Self-Forcing, CausVid or our
SF-Wan. It is a **full-duplex conversational agent**: one Transformer,
initialised from a Qwen language model, that listens to the user's
microphone and camera and answers with text, speech and video in 160 ms
units. The nearest thing we plan is the real-time avatar (P1-2 in
research-avatar-v2v §5), not SF-Wan.

**Recommendation:**

1. Do not schedule a port. Add Wan-Streamer to the watch list (§7) and
   re-check the `Wan-AI` Hub org and `github.com/Wan-Video` monthly.
2. Its prerequisites are on the roadmap anyway and are worth doing for
   other reasons: **P0-1 WebRTC ingest** (mic and camera into the engine),
   duplex audio+video stream caps, and a multi-GPU split of one session. If
   weights appear, those pieces are ready.
3. For a conversational avatar that can ship now, follow P1-2
   (LiveTalk-1.3B or SoulX-FlashHead-1.3B on the SF-Wan causal rollout).
   LiveTalk also has an optional audio-LM "thinker/talker" mode
   (research-avatar-v2v §2.3). That is the open, cascaded approximation of
   what Wan-Streamer does end to end.
4. It **complements** SF-Wan and does not replace it. SF-Wan is
   prompt-driven open-ended world video. Wan-Streamer is a talking agent
   driven by the user's audio and video.

---

## 1. What was published

| Item | Fact | Source |
|---|---|---|
| Name, lab | Wan-Streamer, Tongyi Lab (Alibaba) Wan team. Lead author Lianghua Huang, 24-26 authors | arXiv 2607.04443 author list |
| Papers | v0.1 "End-to-end Real-time Interactive Foundation Models", arXiv **2606.25041** (June 24, 2026). v0.2 "Higher Resolution, Same Latency", arXiv **2607.04443** (July 5). v0.3 "Video = World + Event Stream", arXiv **2607.15038** (July 16) | arXiv abstract pages; wan-streamer.com |
| Paper licence | CC BY 4.0 (the arXiv licence of the **report text**; it grants nothing for a model) | arXiv abstract pages |
| Website | https://wan-streamer.com/ (v0.1, v0.2, v0.3 pages: demo videos and the paper link only; contact e-mail) | site |
| Code | **None.** No link in the post, the papers or the site. The `Wan-Video` GitHub org lists Wan2.1, Wan2.2, Wan-Dancer, Wan-Animate-2, Wan-skills and a diffusers fork | https://github.com/Wan-Video |
| Weights | **None.** The `Wan-AI` Hub org (27 models) has no Streamer repo; the latest are Wan-Dancer-14B and Wan2.2-Animate-2-14B. ModelScope was not reachable from here (**UNVERIFIED** there) | Hub API, 2026-09-29 |
| Model size | **NOT DISCLOSED.** A Hugging Face community blog states "Wan-Streamer's size is undisclosed" and "No model weights released" | https://huggingface.co/blog/ResterChed/wan-3-0 |
| Release plans | The post says only "we'll continue releasing further iterations of Wan-Streamer", which refers to reports. No statement about open weights | Substack post |

The same HF community blog says "the code is CC BY 4.0". That is wrong or a
misreading of the arXiv licence: there is no code to license.

Some tool summaries of the v0.1 PDF during this research returned "8×4×4
VAE compression", "7.5 video tokens per second", "16 kHz audio" and "H100
GPUs". **None of these is in the paper text**: the full text was searched,
and the only GPU mentioned is "two A100 GPUs" in a table row about another
system (X-Streamer). A summary of the v0.3 site also said "single GPU
performer". The v0.3 paper says the v0.2 multi-GPU topology is unchanged.
Do not reuse these numbers.

---

## 2. Architecture

### 2.1 What the papers state

| Aspect | Wan-Streamer | Source |
|---|---|---|
| Model | "a single autoregressive Transformer that jointly models omni-modal understanding and generation". Language, audio and video, as input and output, are "an interleaved causal sequence processed by a single Transformer" | post; v0.1 §1 |
| Initialisation | "We initialize the unified Transformer from a language model [43, 42]": refs 42-43 are the Qwen2.5 and Qwen3 technical reports. **Which model and size: NOT DISCLOSED** | v0.1 §2.3 |
| Video side | **Not** initialised from a Wan video DiT (no Wan2.x checkpoint is mentioned in any of the three reports). "Strictly causal audio and video variational autoencoders" of their own; compression, channels and patching **NOT DISCLOSED** | v0.1 §1, §2.3 |
| Generation | Text: next-token prediction. Audio and video latents: **conditional flow matching**, denoised jointly from the same clean context, then "appended directly to the history as clean context" | v0.1 §2.1 |
| Causality | "block-causal multimodal attention", "full-history autoregressive streaming". Every unit is committed back into the history | v0.1 §1-2 |
| Streaming unit | **160 ms** (4 frames at 25 fps). Real time requires the performer step plus KV and latent transfer to fit in one unit | v0.1 §2.4 |
| KV cache | **Full history**. The thinker builds a KV slice per unit and sends it to the performer, which "appends the received KV slice into its own full-history cache". **No sink, window or eviction is described**; the maximum session length is **NOT DISCLOSED** | v0.1 §2.4; v0.2 §3 |
| Distillation | Teacher with CFG and more solver steps; student via "rolling distillation … self-forcing strategy [Self-Forcing] with distribution matching [DMD, DMD2]". Student step count **NOT DISCLOSED** | v0.1 §2.3 |
| Training data | Understanding (image/audio/video QA, ASR, TTS, dialogue), generation (image, audio, video, joint AV) and duplex interaction data; amounts **NOT DISCLOSED** | v0.1 §2.2 |
| Inputs | User text, microphone audio, camera video, all streaming. v0.3 adds a **world context** (scene, characters, appearance, persona, voice, ambient sound) "tokenized and prefilled once before streaming". How it is given (text only, or also a reference image) is **NOT DISCLOSED** | v0.1 §2.1; v0.3 §1-3 |
| Prompt switching | No mid-session world or prompt switch is described. The user steers by talking, typing or showing things. In v0.3 the model emits free-form behaviour directives such as "(picks up the mug …)" in its own text stream | v0.3 §3 |
| Outputs | Text, **speech audio** and **video**, synchronised, full duplex (it shows listening behaviour while the user talks, and handles interruptions) | v0.1 §1 |
| Resolution / fps | v0.1: 192×336 (called "192p"). v0.2 and v0.3: **640×368 at 25 fps** | v0.2 abstract |
| Latency | **About 200 ms model-side** signal to signal (from a complete 160 ms user unit at the thinker to the decoded response unit). **About 550 ms** total with an assumed 350 ms round-trip network budget | all three |

### 2.2 Serving topology: thinker and performer

- **Thinker (one GPU).** Holds the causal audio/video encoders, the short
  token-causal Transformer pass (language and state update, builds the KV
  slice) and the causal audio/video decoders. Each unit it encodes the
  user's input, sends the KV slice to the performer, receives the previous
  unit's clean latents, decodes them and emits them.
- **Performer.** Holds only the flow-matching latent generation. It
  receives the KV slice, appends it to its own copy of the history and
  denoises the next unit's latents.
- **v0.1:** "the two-GPU thinker-performer serving path".
- **v0.2 and v0.3:** one thinker GPU plus a **multi-GPU Ulysses-style
  context-parallel performer group**. Each rank writes the incoming K/V
  into a pre-sharded local cache. The video latent sequence is split
  across ranks with all-to-all and gather. Audio latents are not sharded.
  **The number of performer GPUs and the GPU type are NOT DISCLOSED.**
  remio.ai: "the paper does not specify a universal deployment
  configuration"
  (https://www.remio.ai/post/tongyi-lab-releases-wan-streamer-v0-2-with-550ms-end-to-end-latency-but-more-gpus-carry-the-load).
- Engineering named in the reports: CUDA graph capture, compilation,
  optimised kernels, KV-cache exchange.
- Throughput claim: real time at 25 fps, one 160 ms unit per step. There
  are no quality metrics (no FVD, lip-sync or LPIPS scores); v0.2 and v0.3
  report "qualitative observations" only.

**INFERRED:** at least 3 GPUs per live session for 640×368 (1 thinker plus
at least 2 performers, since Ulysses needs a group). With a 14B-class
performer and a full-history cache it is probably more. Cost per session
is therefore several times SF-Wan's one GPU.

---

## 3. How it differs from Self-Forcing, CausVid and our SF-Wan

| | Self-Forcing / CausVid | Our SF-Wan ([wan.md](wan.md)) | Wan-Streamer |
|---|---|---|---|
| Base | Wan2.1-T2V-1.3B DiT (bidirectional, made causal) | Self-Forcing's Wan2.1-1.3B causal DMD checkpoint | A Qwen LLM grown into an omni-modal Transformer; own causal audio and video VAEs |
| Conditioning | T5 text prompt (cross-attention) | UMT5 text, prompt switch (keep or reset) at block boundaries | The whole interaction history in one sequence: user text, audio, video; the agent's own past output; a world context |
| Output | Silent video | Silent video, 832×480 at 16 fps | Text + speech + video, 640×368 at 25 fps |
| Block | 3 latent frames (12 video frames) | 3 latent frames = 0.75 s at 16 fps | 160 ms (4 video frames) plus audio |
| Context | Local attention window; Self-Forcing trains on 21 latent frames | Rolling window 21, **rebased sink 15** (R12 study) | **Full history** (eviction not described) |
| Distillation | DMD / self-forcing rollout | inherited | Self-forcing with distribution matching ("rolling distillation") over consecutive units |
| GPUs | 1 | 1 (H100: 23.8 fps with sink 3 and graphs; RTX PRO 6000: 14.8 fps with sink 15) | 2 (v0.1), 1 + an N-GPU performer group (v0.2+) |
| Weights | Apache-2.0, public | on both volumes (`sfwan21-1.3b`) | none |

Takeaways:

- The **training recipe** is the same family (self-forcing plus DMD over
  its own rollouts). The **model** is not: there is no Wan DiT to reuse
  weights or layer code from.
- The long-horizon problem our R12 study found (drift past the training
  horizon, fixed with a deep sink) is solved differently there: full
  history plus rolling distillation. They give no numbers for how long a
  session stays clean.
- Our E7 work (CUDA graphs per cache state, a static KV cache, persistent
  inputs) is the same kind of engineering their reports name ("CUDA graph
  capture"). Wan-Streamer's cache **grows every unit** and is not a fixed
  window, so a per-state graph key like `causal::KvBlockKey` would need
  bucketing by length. **INFERRED.**

---

## 4. Mapping onto our stack (if weights appeared)

### 4.1 What would be reused

| Our piece | Reuse | Where |
|---|---|---|
| Wan DiT kernels (bf16 GEMMs, fused residual+norm, FP8, `attn_dc.cu` dense attention) | **Generic parts only**: GEMM, RMSNorm, RoPE, dense and block-causal flash attention. The Wan-specific blocks (modulation, cross-attention to UMT5) are not used | `crates/fastvideo-cudarc/src/wan/{bf16_gemm,fuse,attn_dc,fp8}.rs` |
| Block-causal flash attention with a KV cache | **Yes, the core primitive.** Queries of the current unit over the full history. It needs an appending cache, not our rolling one | `wan/causal.rs`, `wan/ar_cache.rs`, `wan/attn.rs` |
| Static KV cache and CUDA graphs (E7) | Pattern reusable; cache shape and graph keys new (growing cache) | `wan/graph.rs`, wan.md "CUDA graphs and the static KV cache" |
| Sequence parallel | `wan/sp.rs` shards attention across GPUs, but the gather goes through the host ("NCCL P2P … remains a follow-up"). A Ulysses performer inside 160 ms needs device-to-device all-to-all (NCCL). **New** | `crates/fastvideo-cudarc/src/wan/sp.rs` |
| Qwen3 text stack | We have a Qwen3 *encoder* forward (Z-Image). The unified Transformer is a Qwen-initialised **decoder with generation**, a different use | [registry-status.md](registry-status.md); `crates/fastvideo-cudarc/src/zimage/` |
| Wan VAE / TAEHV | **No.** Wan-Streamer has its own strictly causal audio and video VAEs. TAEHV's carried-state per-block decode is the right *pattern* for a causal decoder | `wan/taehv.rs` |
| Pacer | `FramePacer` (causal jitter buffer, freeze on underrun) and `AvPacer` (one video frame plus `rate/fps` audio samples per tick) fit a 25 fps A+V stream as they are | `crates/fastvideo-media/src/pacer.rs` |
| WebRTC out, WHIP | H.264 (NVENC), VP8 and Opus encode; native WHIP publishing; Reactor and fal WMA transports: **reused as is** | `crates/fastvideo-webrtc`, `crates/fastvideo-media` |
| WebRTC **in** | **Missing.** `host.rs`: "our server peers only send". Reactor `PublishTrack` is refused. Wan-Streamer needs the mic **and** the camera at 160 ms granularity: this is exactly P0-1 | `crates/fastvideo-webrtc/src/host.rs:257`; research-avatar-v2v §1, §5 P0-1 |
| Opus decode | `OpusDecoder` exists (tests only) | `crates/fastvideo-media/src/opus.rs` |
| Reactor causal mode | Session lifecycle, pause gate, `state_update`, session limits and `reset` carry over. The commands differ: no `set_prompt` loop; instead a world-context setter at start, user text messages, and input tracks | `crates/fastvideo-reactor/src/causal.rs`; design.md §5.7 |
| Stream caps | `StreamCaps::Causal { block_frames, target_fps }` has no notion of audio output or input tracks. Needs a duplex variant (input tracks, audio out, unit length in ms) | `crates/fastvideo-protocol/src/caps.rs:447` |
| Session caps (R12) | `causal_default_max_s` / `causal_hard_max_s` apply unchanged; with a growing cache the limit is set by memory as well as quality | design.md §5.2 |

### 4.2 What is new

- The omni-modal Transformer: a Qwen-style decoder with multimodal token
  interleaving, block-causal masks across modalities, a text LM head and
  flow-matching velocity heads for audio and video.
- Causal audio and video encoders (perception) and causal audio and video
  VAEs (generation), none of which we have.
- Tokenizer and chat format (v0.3 behaviour directives).
- The thinker/performer split: two engine processes on different GPUs,
  exchanging a KV slice and latents every 160 ms. Our engine runs one
  session on one GPU today.
- Ulysses context parallel with a device all-to-all (NCCL) and pre-sharded
  caches.
- A duplex session model: the server reads input tracks continuously
  while it writes output; response timing and turn-taking come from the
  model.

### 4.3 Reference implementation for parity

**None exists.** There is no upstream repo, no FastVideo port and no
Diffusers pipeline, and OpenTrain AI lists no maintained implementation
(https://www.opentrain.ai/papers/wan-streamer-v0-1-end-to-end-real-time-interactive-foundation-models--arxiv-2606.25041/,
**UNVERIFIED**). A port would have to wait for an official release. The
papers are far too vague to reimplement from, and training one is out of
scope.

---

## 5. Estimates (all INFERRED; conditional on an open release)

The session and GPU-rate conventions are those of research-avatar-v2v §5:
one agent-session is a working day with build-pod access. Rates: RTX PRO
6000 $2.09/h, H200 $4.59/h, H100 $4.18/h ([gateway.md](../serve/gateway.md)).

| Item | Effort | GPU cost to validate | Notes |
|---|---|---|---|
| Prerequisite: P0-1 WebRTC ingest | 2-3 sessions | ~$2-5 | Already on the roadmap. Needed by V2V and live avatars anyway |
| Port: omni Transformer, causal audio and video VAEs, encoders, tokenizer, flow heads | 8-12 sessions | ~$20-40 (single-GPU bring-up) | Depends heavily on what is released, and on whether it reuses Qwen3 layers exactly |
| Parity against the upstream reference (per unit latents, KV, decoded AV) | 3-5 sessions | ~$30-60 | Needs the upstream Python running on the same multi-GPU topology |
| Thinker/performer split, NCCL Ulysses performer, growing-cache graphs | 4-6 sessions | ~$40-80 | 3+ GPUs per test; H100/H200 nodes with NVLink preferred |
| Serving: duplex caps, Reactor schema, native WHIP in/out, pacing, E2E | 3-4 sessions | ~$20-40 | |
| **Total** | **~20-30 sessions** (plus P0-1) | **~$110-220** | Each live session then occupies 3+ GPUs, **INFERRED** |

**Weights download:** unknown. An **INFERRED** range from the Qwen
initialisation: a 7-8B backbone is ~15-17 GB in bf16, a 14B one ~28-30 GB,
a 32B one ~64 GB, plus VAEs and encoders (a few GB), on the EU volume
`jg48s6o1w0` (EU only since 2026-10: the US volume `s2k01690bi` was
deleted; double it if US is rebuilt). **Needs the owner's approval**
(CLAUDE.md) once a real size is known. Every tree goes into
`scripts/gpu/weights-manifest.tsv` and `scripts/gpu/verify-weights.sh`.

**Commercial licence:** nothing is licensed. The CC BY 4.0 on arXiv covers
the report text only. For context: Wan2.x and Wan-Dancer-14B are
Apache-2.0 (Hub cards; the Wan-Dancer licence per the HF community blog,
**UNVERIFIED** here). An open release under Apache-2.0 is plausible but
not announced. A Qwen-initialised model could also inherit Qwen licence
terms: most Qwen2.5/Qwen3 checkpoints are Apache-2.0, some Qwen2.5 sizes
use the Qwen licence. **Not assessable until a release**; re-check then.

---

## 6. Fit with the roadmap

- **P0-1 WebRTC ingest** is the shared blocker for real-time V2V, a
  mic-driven avatar and Wan-Streamer. Keep it first.
- **P1-2 real-time avatar (LiveTalk / SoulX-FlashHead on the SF-Wan
  rollout)** is the buildable version of the same product today:
  image + audio → streamed talking video on one GPU (claimed 24.82 fps for
  LiveTalk, 16 fps output). Add ASR → LLM → TTS (or LiveTalk's
  thinker/talker) for conversation. Wan-Streamer is the end-to-end version
  of that cascade, with lower claimed latency (about 200 ms against the
  cascade's module boundaries) and joint speech, gaze and gesture.
- **Duplex protocol work** done for P1-2 should be designed so a later
  Wan-Streamer driver fits:
  - input tracks in `StreamCaps`;
  - audio output on causal streams;
  - a unit length in ms rather than latent frames;
  - a world or persona context set once at session start.
  That is cheap now and avoids a second protocol change. **INFERRED.**
- **Multi-GPU sessions** (thinker/performer, NCCL Ulysses) are new for
  this repo. They would also serve LiveAvatar (5× H800) and
  SoulX-FlashTalk (8× H800), both P2 or skipped for the same reason. Do not
  build this speculatively.
- **SF-Wan stays.** It is the only open-ended, prompt-steered world stream
  we serve, and it runs on one GPU. Wan-Streamer does not do prompt-driven
  scene generation. v0.3's "world + event stream" pretraining hints at
  roaming and world-model uses, but the reports evaluate only the
  conversational agent.

---

## 7. Watch list and open questions

Re-check monthly (or when Tongyi posts again):

- `https://huggingface.co/api/models?author=Wan-AI`, the `Wan-Video`
  GitHub org and ModelScope `Wan-AI`, for a Streamer repo;
- wan-streamer.com, for a release, an API or a licence.

Open (all **NOT DISCLOSED**):

1. Parameter counts of the backbone and of the performer, and which Qwen
   model it starts from.
2. Video and audio VAE design (compression, channels, sample rate) and
   tokens per 160 ms unit.
3. Student flow-matching step count.
4. GPU type and the number of performer GPUs at 640×368.
5. How the world context and persona are supplied (text only, or a
   reference image or voice sample).
6. Maximum session length and memory growth under full history.
7. Quality metrics of any kind.

## Sources

- Tongyi Lab, "Wan-Streamer: A Native-Streaming Model …":
  https://tongyilab.substack.com/p/wan-streamer-a-native-streaming-model
- v0.1: https://arxiv.org/abs/2606.25041 (HTML v2 read in full)
- v0.2: https://arxiv.org/abs/2607.04443
- v0.3: https://arxiv.org/abs/2607.15038
- https://wan-streamer.com/ (v0.1, v0.2, v0.3 pages)
- https://github.com/Wan-Video
- https://huggingface.co/api/models?author=Wan-AI
- https://huggingface.co/blog/ResterChed/wan-3-0 (secondary)
- https://www.remio.ai/post/tongyi-lab-releases-wan-streamer-v0-2-with-550ms-end-to-end-latency-but-more-gpus-carry-the-load (secondary)
- Ours: [wan.md](wan.md) (SF-Wan, E7, R12),
  [../serve/research-avatar-v2v.md](../serve/research-avatar-v2v.md),
  [../serve/design.md](../serve/design.md) §5.2, §5.7
