# fastvideo-rs serve: design

Status: design, 2026-09-27. Nothing here is implemented yet.

Inputs, which are the source of truth for every external wire detail:

- [`research-fal.md`](research-fal.md): fal queue API, H3-max t2v/i2v/ref2v, and the WMA director.
- [`research-reactor.md`](research-reactor.md) and [`reactor-proto/`](reactor-proto/): the Reactor runtime.
- [`research-minimax-fastvideo.md`](research-minimax-fastvideo.md): MiniMax V2, FastVideo `/v1/videos`, and the FastWan Video API.
- [`research-ltx-api.md`](research-ltx-api.md): the LTX API, v1 sync and v2 async.
- [`research-deploy.md`](research-deploy.md): Runpod pods, Runpod serverless, Vast.
- [`research-streaming-refs.md`](research-streaming-refs.md): strobe and infinite-livestream.

Citations such as `fal §9.3` point to a section of the matching research doc.
This document **adds no external protocol fields**. Where it defines a shape
of its own (the `/fv/v1/*` native API, the Runpod job envelope, config), the
text says **native**.

---

## 0. Owner decisions (2026-09-27) — these override anything below

1. **Video encoder: NVENC.** The runtime image gains the NVIDIA `video` driver
   capability (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`) and
   `fastvideo-media` encodes H.264 on the GPU's NVENC. OpenH264 stays only as a
   CPU-only test/CI backend behind a feature and is never used in a deployed
   image (no Cisco patent licence when built from source). No x264.
2. **H.264 level / 1344x768 H3 streams.** WHIP to Cloudflare: scale H3 streams to
   1280x720 (level 3.1) before encoding. Full resolution (1344x768, level 4.0) is
   supported when publishing to MediaMTX (self-hosted relay) and for peer WebRTC.
   The WHIP target config selects the profile: `cloudflare` → 720p cap,
   `mediamtx` / peer → native resolution.
3. **Model tiers exposed by the APIs.**
   - `h3-max` (fal `minimax/h3-max/*`, MiniMax `MiniMax-H3-Max`) and LTX
     `ltx-2-5-pro` / `ltx-2-3-pro` map to **our highest-quality configuration**
     of that family (no quality-reducing shortcuts: full step count / non-lossy
     attention route / full VAE; exact recipe chosen per model in the engine
     capability table and documented there).
   - New variants **`h3-turbo`** and **`ltx-turbo`**: our FastH3 and LTX
     configurations with the **fastest generation times** that still pass the
     quality gate (e.g. FastH3 4-step VSA; LTX-2.5 distilled two-stage Sol),
     exposed on every API that accepts a model/endpoint id (fal endpoint ids
     `minimax/h3-turbo/{text,image,reference}-to-video`, MiniMax model
     `MiniMax-H3-Turbo`, LTX model `ltx-turbo`, openai-videos model ids).
   - Responses carry the resolved internal recipe in metadata where the wire
     format allows it.

4. **Cold start (serverless).** Serverless workers run the Rust runtime image
   (`serve` target) with the weight volume mounted (`/runpod-volume`); nothing is
   downloaded at start. Weight loading is the cold start, so it gets its own
   engine packages:
   - **E12 fast weight loading:** parallel large sequential reads from the
     volume, pinned-host staging, and overlapping load with the first stages
     (text encode before the DiT is resident). Target: H3 cold load from ~5.4–8.3
     min to < 2 min; LTX-2.5 from ~2 min to < 1 min.
   - **E13 pre-quantized FP8 text encoders (owner decision):** store the H3
     text encoder's resident FP8 form (the weight-only `Fp8Rows` layout: E4M3
     codes + per-row scales, as `llm::ResidentDecoder` builds today at load,
     ~118 s) as safetensors next to the bf16 shards (e.g.
     `h3-base/text_encoder_fp8/` with a manifest recording the source shard
     hashes and the quantization rule) and load it directly; same treatment for
     LTX's Gemma encoder so the first LTX request does not stream it (~61 s).
     An offline `fv-gpucheck quantize-text-encoder` (or equivalent) tool writes
     it; the loaded tensors must be byte-identical to the load-time
     quantization (hash test). Written to both weight volumes (EU needs the
     volume-sync decision first — it is full).
   - **WP-19 cold-start measurement:** fresh Runpod serverless worker → submit →
     first output, per model family, before and after E12/E13.
   - Status (2026-09-27): E12/E13 implemented, WP-19 measured — results and
     open items in [`docs/gaps/2026-09-27-cold-start.md`](../gaps/2026-09-27-cold-start.md).

5. **Sol-H3 serves the tau-ladder route (owner decision).** Whenever the server
   runs Sol-H3 4-step, the engine capability table selects the profile
   `h3/sol_h3_4step_engine_ladder` (Sol engine route, tau 1.0 / 1.25 / 1.5 on
   forwards 1-3; RTX PRO 6000 768p: denoise 21.8 s vs 33.5 s dense, 1.53x,
   gate PASS, LPIPS 0.375). The `sol-h3` recipe itself stays dense as
   sol-engine publishes it for one GPU, so parity/oracle runs and the
   upstream comparison keep their reference; the dense route remains
   selectable as an explicit profile (`h3/sol_h3_4step`).

6. **Draft tier (owner decision).** A third tier, **`draft`**, exposes our
   faster configurations that do NOT pass the quality gate, for previews and
   iteration: public ids `h3-draft` (fal `minimax/h3-draft/*`, MiniMax
   `MiniMax-H3-Draft`) and `ltx-draft` (LTX API model id), plus openai-videos
   model ids. Candidates (measured, gate FAIL, fastest first): H3 — FastH3
   4-step VSA 480p + TAEH3 (8.1 s, RTX PRO 6000), FastH3 8-step Sol+TeaCache+
   TAEH3; LTX — LTX-2.5 distilled + NVFP4 FFN (4K denoise 1.14x, -8.6 GiB,
   fails sharpness at 4K) with TAEHV decode. Responses must mark the result as
   draft quality (metadata where the wire allows). Tier order: draft < turbo <
   max; `Tier` gains a `Draft` variant.

7. **Cloudflare storage (owner decision).** Job records live in **Cloudflare D1**
   (SQLite: jobs table + indexes on owner/status/created for the MiniMax and
   FastVideo list endpoints; state-change writes, throttled progress). Media
   (outputs, uploads, fetched inputs) lives in **Cloudflare R2** via the
   existing S3-compatible artifact store with presigned URLs (KV was rejected
   for media: 25 MiB value limit; and for jobs: 1 write/s/key, eventual
   consistency, no queries). serve-kit gains a `D1JobStore` (D1 HTTP API)
   next to the in-memory/file stores; running jobs stay authoritative in the
   worker's memory, D1 is the durable/shared copy so any worker can answer
   status/result after restarts or scale-to-zero.
   Provisioned 2026-09-27: D1 `fv-jobs` (id 1796e295-a7f0-4402-bbed-ec94ccb27c15,
   WNAM), R2 bucket `fv-media`. Runtime credentials are Runpod secrets
   (reference as `{{ RUNPOD_SECRET_<name> }}` in templates): `fv_cf_account_id`,
   `fv_cf_api_token` (D1 HTTP API), `fv_d1_database_id`, `fv_r2_bucket`,
   `fv_r2_endpoint`, `fv_r2_access_key_id` / `fv_r2_secret_access_key` (R2 S3
   keys derived from the API token: id / SHA-256 of the value). Vast: same
   names as account env vars once a Vast API key is available.
   **Runtime env names (as implemented, WP-10):** `fv-serve` reads the
   UPPERCASE variables `FV_CF_ACCOUNT_ID`, `FV_CF_API_TOKEN`,
   `FV_D1_DATABASE_ID`, `FV_R2_BUCKET`, `FV_R2_ENDPOINT`,
   `FV_R2_ACCESS_KEY_ID`, `FV_R2_SECRET_ACCESS_KEY` (the lower-case spelling
   is accepted as a fallback). Vast: account env vars under exactly these
   uppercase names (Vast injects them). Runpod: the secrets keep their
   lower-case names and the template maps them, e.g.
   `FV_CF_API_TOKEN={{ RUNPOD_SECRET_fv_cf_api_token }}`, one line per
   variable (`configs/serve/runpod.toml` lists all seven). With all D1
   values set, `jobs.backend = "auto"` selects D1; with all R2 values set,
   `artifacts.backend = "auto"` selects R2 (region `auto`, path-style).

## 1. Goals and non-goals

### 1.1 Goals

1. **One Rust server, `fv-serve`**, drives the fastvideo-rs CUDA pipelines
   directly: H3, FastH3 and Sol-H3; LTX-2, 2.3 and 2.5 (video+audio); and Wan,
   FastWan, SF-Wan causal and TI2V-5B (video only). It keeps models resident
   and warm, and runs one GPU executor thread per device.
2. **Batch APIs**, each wire-compatible with unmodified clients:

| API | Surface | Crate |
|---|---|---|
| fastvideo-api | FastVideo `fastvideo serve` `/v1/videos` family (minimax-fastvideo §2.1) **and** the FastWan `/generate` shape (§2.2) | `fastvideo-openai-videos` |
| MiniMax | V2 `/v2/video_generation` for `MiniMax-H3` / `MiniMax-H3-Max` (minimax-fastvideo §1.2-1.7). V1 is out of scope (§1.2) | `fastvideo-minimax` |
| fal | `minimax/h3-max/{text,image,reference}-to-video` over the queue API and sync (fal §2-§5, §9, §10.1) | `fastvideo-fal` |
| LTX | `/v2/*` async jobs plus `/v1/*` sync and `/v1/upload` (ltx §2) | `fastvideo-ltxapi` |

3. **Streaming APIs**:
   - the fal `minimax/h3-max/director` WMA WebRTC session (fal §8);
   - the Reactor local runtime (reactor §3-§5), using the fast-h3
     queue-and-playout command set for clip models (reactor §4bis) and setter
     commands for causal SF-Wan.

4. **Video+audio and video-only are both first-class.** The track set comes
   from model capability. There is never a silent audio m-line on WebRTC, and
   RTMP/HLS always carry audio (silence if needed).
5. **Four deploy targets from one image** (`serve` Docker target): Runpod
   pod; Runpod serverless queue; Runpod serverless load balancer; Vast
   instance. Vast serverless is reached through a thin PyWorker forwarder.
6. **CPU-only CI** covers every protocol, using a fake engine and golden
   JSON. GPU E2E runs on Runpod.

### 1.2 Non-goals (decided)

| Item | Why |
|---|---|
| MiniMax **V1** (`/v1/video_generation`, `/v1/files/*`) | V1 lists only Hailuo models and never H3 (minimax-fastvideo §0.1, §1.8). No client targets H3 over V1. Listed as stretch package S1 |
| MiniMax Context-IR, `/v2/video_regeneration`, `resolution: 2K` | These need MiniMax platform components and a 2K upscaler we don't have (§1.6b, §3.5 #2). The endpoints return 400 |
| LTX `audio-to-video`, `retake`, `extend`, `video-to-video-hdr`, `video-to-video-reframe` | No engine path (ltx §5 #10-13). They answer `403 permission_error` ("endpoint not available for the account"), a documented LTX type (ltx §1.5) |
| FastVideo `WS /v1/stream`, image routes, playground | These are not part of the requested batch contract |
| fal msgpack realtime WS, `ws.fal.run`, `/stream` SSE on H3 | fal does not expose them for H3 (fal §10.2-10.4) |
| Reactor cloud coordinator (`api.reactor.inc`, `/tokens`), recording and HLS clips | The coordinator is closed source (reactor §9). We serve the **local runtime** contract. `RequestClip` answers `clip_failed` |
| More than one streaming session per GPU | Matches RT (one session per process) and fal `one_session_per_machine: true` |
| TURN client inside the server; NVENC | See §5.8 and §5.9 |
| Prompt expansion, safety checker, billing | Fields are accepted and do nothing where the spec permits (§4). Billing numbers are not emitted |

---

## 2. Crate map

### 2.1 Crates and responsibilities

| Crate | Kind | Responsibility |
|---|---|---|
| `fastvideo-protocol` | lib, **no tokio runtime and no axum** (only `tokio/sync`) | Normalized `GenerationRequest`, `Task`, media refs, `ModelCaps` and `negotiate()`, the `Job` model and `JobStore` trait, `Artifact`, the `ApiError`/`GapId` error model, the `BatchProtocol`/`JobView`/`StreamProtocol` traits over a framework-free `HttpReply`, raw A/V buffer types (`RgbFrame`, `Pcm`), `TrackSet` |
| `fastvideo-engine-service` | lib | Warm model pool; one executor thread per GPU; scheduler; cancellation; progress events; the capability table; `EngineBackend` trait with a `FakeBackend` (always) and a `CudaBackend` (feature `cuda`); the streaming cores `ClipSession` (clip-queue playout) and `CausalSession` (SF-Wan block rollout) |
| `fastvideo-media` | lib | `AvPacer` (a port of strobe-core pacing and clock, plus an audio lane); the resampler (rubato); the Opus framer and encoder (libopus); the H.264 encoder (OpenH264, with an optional ffmpeg-x264 backend); the MP4 post-processor (ffmpeg remux, crop, audio drop); input probing and decoding (image crate, ffprobe/ffmpeg); crossfade; RTMP/HLS sinks (ffmpeg subprocess) |
| `fastvideo-webrtc` | lib | **str0m**-based peer host: one UDP mux port, an ICE-TCP passive listener, non-trickle answers with embedded candidates, data channels, and pre-encoded H.264/Opus writers. Also the WHIP publisher client |
| `fastvideo-serve-kit` | lib (axum) | Shared HTTP glue: `ServeCtx`, auth extractor (Bearer, `Key`, none, trust-gateway), `MemJobStore`, `ArtifactStore` (local signed URLs or S3 presign), `UploadStore` (`PUT /uploads/{token}`), media ingestion (fetch, data URI, SSRF guard, limits), callback sender, SSE helper, generic axum adapters for `JobView` |
| `fastvideo-openai-videos` | lib | FastVideo `/v1/videos*`, `/v1/models*`, `/v1/model_info`; FastWan `/generate`, `/status`, `/video` |
| `fastvideo-minimax` | lib | MiniMax V2 create, query, list and delete; the callback challenge; `OaiError` rendering |
| `fastvideo-ltxapi` | lib | LTX v2 submit and status, v1 sync, `/v1/upload`, 403 stubs |
| `fastvideo-fal` | lib | fal queue (all path forms), sync, `x-fal-target-url` proxy mode, storage initiate, webhooks; **director** (WMA bridge endpoints plus control-channel protocol) |
| `fastvideo-reactor` | lib | `reactor_wire.v1` (prost from the vendored protos) plus the v0 JSON codec; local session routes; signalling routes; the per-connection gateway; command/model-message schemas for fast-h3 and SF-Wan modes; the OpenAPI `/schema` |
| `fastvideo-deploy` | lib | Rust Runpod queue worker (job-take, job-done, stream, ping, job-stop); environment discovery (`RUNPOD_*`, `VAST_*` to public IP and ports); Vast PyWorker forwarder assets |
| `fastvideo-serve` | bin `fv-serve` | Config (TOML plus `FV_*` env), mode selection (`http`, `runpod-queue`), router assembly, native `/fv/v1/*`, health, `/ping`, metrics, graceful shutdown |

**Decision: one crate per external API, not modules inside
`fastvideo-serve`.** Reasons:

- Each API has its own golden fixtures, error envelope and id format.
- Separate crates give each agent a disjoint file tree (§8), so the packages
  can be built in parallel.
- A CI job can compile and test one adapter against the fake engine without
  building the others.

`fastvideo-serve` only mounts routers. The cost is five small `Cargo.toml`
files.

### 2.2 Dependency graph

```
fastvideo-protocol ◄──────────────────────────────────────────────┐
   ▲        ▲                ▲                                     │
   │   fastvideo-media   fastvideo-models (geometry helpers)       │
   │        ▲   ▲            ▲                                     │
   │        │   │   fastvideo-engine-service ──(feature cuda)──► fastvideo-cudarc
   │        │   │            ▲
   │   fastvideo-webrtc      │
   │        ▲                │
   └── fastvideo-serve-kit ──┘   (axum, tower-http, reqwest)
            ▲
  ┌─────────┼──────────┬─────────────┬───────────┬──────────────┐
openai-videos  minimax   ltxapi      fal (+webrtc,media)  reactor (+webrtc,media)
  └─────────┴──────────┴──────┬──────┴───────────┴──────────────┘
                  fastvideo-deploy (axum Router oneshot, reqwest)
                              ▲
                        fastvideo-serve (bin fv-serve)
```

Rules:

- Adapters never depend on each other, or on `fastvideo-cudarc`.
- Only `fastvideo-engine-service` touches CUDA, behind `cuda`, so the default
  workspace build stays CPU-only.
- The engine service drives `fastvideo-cudarc` pipelines directly
  (`H3Pipeline::load/generate`, `Ltx2Pipeline::load/generate`,
  `WanPipeline::load/generate_to`). It bypasses `fastvideo-core`'s per-call
  `VideoGenerator`/`generate_av`, because those reload per call
  (minimax-fastvideo §3.5 #7). One side effect: Wan per-request `fps` and
  `flow_shift` come for free, since `GenerateConfig` has both.
- Workspace lints still apply (`unsafe_code = "forbid"`). FFI stays inside
  third-party crates: `openh264`, `audiopus`, `str0m` (pure Rust).

---

## 3. The protocol abstraction (`fastvideo-protocol`)

The signatures are normative. Derives are elided: every type is
`Clone + Debug`, and the wire-facing ones are `Serialize + Deserialize`.

### 3.1 Normalized request

```rust
pub struct ModelId(pub String);                 // engine model id after alias resolution
pub enum ProtocolId { OpenAiVideos, FastWan, MiniMaxV2, Fal, FalDirector, LtxV1, LtxV2, Reactor, Native }
pub enum Family { H3, Ltx2, Wan, MmAudio }

/// What the client asked for, before capability checks.
pub enum Task {
    T2V,        // text only
    I2V,        // exactly one first-frame image
    Keyframes,  // last-only or first+last (H3 fl2va, LTX last_frame_uri)
    Ref2V,      // H3 ref2va: ordered image/video/audio references
    A2V,        // audio drives output (LTX audio-to-video)  -> Unsupported today
    Extend, Retake, V2V,                          // LTX edit endpoints -> Unsupported today
}

pub struct GenerationRequest {
    pub protocol: ProtocolId,
    pub model: String,              // as sent (alias); resolved via ServeConfig.aliases
    pub task: Task,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: Option<u64>,          // None -> server draws one, recorded on the Job
    pub canvas: CanvasSpec,
    pub timing: TimingSpec,
    pub keyframes: Vec<Keyframe>,
    pub references: Vec<Reference>, // order preserved exactly as sent
    pub audio_in: Option<AudioInput>,
    pub audio_out: AudioOut,
    pub sampling: SamplingOverrides,
    pub output: OutputOptions,
    pub accepted_noop: Vec<&'static str>, // e.g. "prompt_expansion_mode", logged + metrics only
}

pub enum CanvasSpec {
    Exact { width: u32, height: u32 },        // FastVideo size/width/height, FastWan, LTX "WxH"
    Aspect { ratio: Ratio, short_edge: u32 }, // fal/MiniMax ratio + 480P/768P
    FollowImage { short_edge: u32 },          // fal i2v, MiniMax "adaptive"
    ModelDefault,
}
pub struct Ratio { pub w: u32, pub h: u32 }

pub struct TimingSpec { pub length: Length, pub fps: Option<u32> }
pub enum Length {
    Seconds { value: f64, snap: Snap },   // MiniMax/fal/LTX duration, FastVideo seconds
    Frames { value: u32, snap: Snap },    // FastVideo num_frames (Exact), FastWan num_frames
    ModelDefault,
    Auto,                                 // LTX duration:null -> Unsupported(LtxAutoDuration)
}
pub enum Snap { AlignUp, Exact }         // FastVideo: explicit num_frames must be on grid

pub enum MediaRef { Http(url::Url), DataUri(String), Upload(UploadId), ProviderFile(String) }
pub enum MediaKind { Image, Video, Audio }
pub enum Anchor { First, Last }
pub struct Keyframe { pub at: Anchor, pub image: MediaRef }
pub struct Reference { pub kind: MediaKind, pub media: MediaRef }
pub struct AudioInput { pub media: MediaRef, pub role: AudioRole }
pub enum AudioRole { TargetSoundtrack /* fal target_audio_url, director audio_url */, Drive /* LTX A2V */ }
pub enum AudioOut { ModelDefault, Silent /* LTX generate_audio=false */, Sidecar /* MMAudio V2A */ }

pub struct SamplingOverrides {
    pub steps: Option<u32>, pub guidance: Option<f32>, pub guidance_2: Option<f32>,
    pub flow_shift: Option<f64>, pub boundary_ratio: Option<f32>,
}
pub struct OutputOptions { pub inline_data_uri: bool /* fal sync_mode */ }
```

### 3.2 Capabilities and negotiation

```rust
pub struct ModelCaps {
    pub id: ModelId,
    pub family: Family,
    pub served_names: Vec<String>,          // e.g. ["fasth3"], FastVideo /v1/models ids
    pub tasks: BTreeSet<Task>,
    pub audio: Option<AudioCaps>,           // None => video-only
    pub fps: FpsCaps,
    pub frames: FrameGrid,
    pub canvas: CanvasCaps,
    pub refs: RefLimits,                    // H3: 9 img / 3 vid / 3 aud / 12 total
    pub stream: Option<StreamCaps>,
    pub knobs: KnobCaps,                    // which SamplingOverrides are honoured per request
    pub resident: bool,
}
pub struct AudioCaps { pub native_rate: u32, pub channels: u8, pub via_sidecar: bool }
pub struct FpsCaps { pub allowed: Vec<u32>, pub default: u32, pub container_only: bool }
pub struct FrameGrid { pub step: u32, pub offset: u32, pub min: u32, pub max: u32 } // H3 17n+5, LTX 8k+1, Wan 4k+1
pub struct CanvasCaps {
    pub multiple: u32,                      // 32 (LTX two-stage: 64)
    pub max_area: u64,                      // H3: 768*1344
    pub aspect: (f32, f32),                 // H3: 1:4 .. 4:1
    pub short_edges: Vec<u32>,              // tiers the model is validated at
    pub pad_and_crop: bool,                 // LTX: 1080 -> 1088 then crop (ltx §3.1)
}
pub struct KnobCaps { pub seed: bool, pub negative: bool, pub steps: bool, pub guidance: bool,
                      pub guidance_2: bool, pub flow_shift: bool }
pub enum StreamCaps { Causal { block_frames: u32, target_fps: u32 }, Clip { min_s: f32, max_s: f32 } }

impl FrameGrid {
    pub fn align_up(&self, n: u32) -> Option<u32>;
    pub fn contains(&self, n: u32) -> bool;
}

/// Pure and deterministic. Called after media ingestion has staged inputs.
pub fn negotiate(req: &GenerationRequest, caps: &ModelCaps, staged: &StagedInputs)
    -> Result<ResolvedJob, ApiError>;

pub struct StagedInputs { pub keyframes: Vec<(Anchor, StagedMedia)>, pub references: Vec<(MediaKind, StagedMedia)>,
                          pub audio_in: Option<StagedMedia> }
pub struct StagedMedia { pub path: PathBuf, pub mime: String, pub bytes: u64,
                         pub probe: MediaProbe /* w,h,duration_s,fps,audio_rate */ }

pub struct ResolvedJob {
    pub model: ModelId, pub task: Task, pub prompt: String, pub negative_prompt: String,
    pub seed: u64, pub width: u32, pub height: u32, pub num_frames: u32, pub fps: u32,
    pub keyframes: Vec<(Anchor, PathBuf)>, pub references: Vec<(MediaKind, PathBuf)>,
    pub audio_in: Option<(AudioRole, PathBuf)>,
    pub audio: AudioPlan,                  // Native{rate,channels} | Drop | Sidecar | None
    pub post: PostProcess,                 // crop: Option<(w,h)>, drop_audio: bool
    pub sampling: SamplingOverrides,       // only knobs the caps honour; others already rejected
}
```

Negotiation rules, applied in this order:

1. The model resolves through aliases.
2. `task ∈ caps.tasks`.
3. The canvas resolves, reusing `fastvideo_models::h3::config::resolve_canvas_size`
   for H3 and a `short_edge` generalization of it (package E3).
4. The frame count resolves on the grid.
5. The fps is allowed.
6. Reference limits hold.
7. Knobs: any knob the caps do not honour → `InvalidRequest` naming the field.
   This is FastVideo's "refuse, do not drop" rule (minimax-fastvideo §2.1)
   and strobe's (streaming-refs §6.3).
8. Audio: `Sidecar` requires `via_sidecar`. Otherwise asking for audio on a
   video-only model is refused.

### 3.3 Errors

```rust
pub enum ErrorKind {
    InvalidRequest, Unsupported(GapId), Unauthorized, Forbidden, NotFound, AlreadyCompleted,
    Conflict, PayloadTooLarge, UnsupportedMedia, ContentFiltered, RateLimited, QueueFull,
    Loading, Timeout, Cancelled, EngineFailed, Internal,
}
pub struct ApiError { pub kind: ErrorKind, pub message: String, pub param: Option<String>,
                      pub retry_after_s: Option<u32> }

/// Engine gaps that currently answer 4xx; each maps to a work package in §8.
pub enum GapId {
    H3FourSeconds,        // E3: duration 4 / 107 frames
    H3Resolution2K,       // permanent (no upscaler)
    H3Refine1080P,        // fal 1080P latent refinement; permanent
    H3TargetAudio,        // E10: target_audio_url / director audio_url
    H3Ref2vaNotLoaded,    // config: ref2va DiT not resident (E11)
    LtxKeyframes,         // E9: last_frame_uri
    Ltx25I2V,             // E5
    LtxAutoDuration,      // duration:null (needs duration head)
    LtxCameraMotion,      // camera_motion
    LtxFps,               // E4: 25/48/50 until validated
    LtxEndpoint,          // A2V/retake/extend/HDR/reframe
    ProviderFiles,        // mm_file://, OpenAI file_id
    PerRequestSteps,      // H3 steps fixed by recipe
    Lora,                 // any lora other than the startup adapter
}
```

Every adapter renders `ApiError` through `BatchProtocol::render_error` (§4.6).

### 3.4 Jobs and artifacts

```rust
pub struct JobId(pub uuid::Uuid);
pub enum JobState { Queued, Running, Succeeded, Failed(ApiError), Cancelled }
pub struct Job {
    pub id: JobId,
    pub protocol: ProtocolId,
    pub external_id: String,       // fal uuid | MiniMax 18-digit numeric | LTX uuid | video_gen_<32hex> | FastWan uuid
    pub owner: Option<KeyId>,
    pub request_echo: serde_json::Value, // original fields some views echo (prompt, size, seconds, ratio)
    pub resolved: ResolvedJob,
    pub state: JobState,
    pub progress: f32,             // 0..=1 from engine step events
    pub queue_position: Option<u32>,
    pub created_at: OffsetDateTime, pub started_at: Option<OffsetDateTime>,
    pub completed_at: Option<OffsetDateTime>, pub expires_at: OffsetDateTime,
    pub logs: Vec<LogLine>,        // {message, level, timestamp}; fal `logs`
    pub metrics: JobMetrics,       // inference_s, stage_durations, peak_memory_mb, build_rtf
    pub artifacts: Vec<Artifact>,
    pub callback: Option<CallbackSpec>, // fal_webhook | MiniMax callback_url
    pub cancel_requested: bool,
}
pub struct Artifact { pub id: ArtifactId, pub mime: String, pub file_name: String, pub bytes: u64,
                      pub location: ArtifactLocation /* Local(PathBuf) | Object{bucket,key} */,
                      pub width: u32, pub height: u32, pub frames: u32, pub fps: u32,
                      pub audio: Option<(u32 /*rate*/, u8 /*channels*/)> }

#[async_trait::async_trait]
pub trait JobStore: Send + Sync + 'static {
    async fn insert(&self, job: Job) -> Result<(), StoreError>;
    async fn get(&self, id: JobId) -> Option<Job>;
    async fn by_external(&self, p: ProtocolId, external_id: &str) -> Option<Job>;
    async fn update(&self, id: JobId, f: Box<dyn FnOnce(&mut Job) + Send>) -> Result<Job, StoreError>;
    async fn list(&self, q: ListQuery) -> Page<Job>;          // owner/protocol/status/model filters, cursor
    async fn remove(&self, id: JobId) -> Option<Job>;
    fn watch(&self, id: JobId) -> Option<tokio::sync::watch::Receiver<JobSnapshot>>;
    async fn sweep_expired(&self, now: OffsetDateTime) -> usize;
}
```

**Decision:** `MemJobStore` in serve-kit keeps an in-memory map plus one JSON
manifest per job under `<state_dir>/jobs/`. This follows the h3fast pattern
(minimax-fastvideo §2.5).

- On restart, `Running` and `Queued` jobs become
  `Failed(Internal, "interrupted by restart")`. Finished jobs keep their
  artifacts until they expire.
- Retention comes from each protocol's default. It is configurable, and
  capped by disk:

| API | Retention |
|---|---|
| MiniMax | 7 d |
| LTX | 24 h |
| fal | 24 h |
| FastVideo | until DELETE, capped at 24 h |

- We add no Redis. Consequences for the load balancer are in §6.3.

### 3.5 Protocol traits

```rust
pub struct HttpReply { pub status: u16, pub headers: Vec<(&'static str, String)>, pub body: ReplyBody }
pub enum ReplyBody { Json(serde_json::Value), Bytes { mime: String, data: bytes::Bytes },
                     File { path: PathBuf, mime: String }, Sse(SseSpec), Empty }

pub trait BatchProtocol: Send + Sync + 'static {
    fn id(&self) -> ProtocolId;
    fn new_external_id(&self, job: JobId) -> String;
    fn render_error(&self, err: &ApiError, cx: &ErrorCtx) -> HttpReply;
}
/// One per submit endpoint (e.g. fal text-to-video, LTX v2 image-to-video).
pub trait SubmitEndpoint: Send + Sync + 'static {
    type Body: serde::de::DeserializeOwned + Send;
    fn normalize(&self, body: Self::Body, cx: &NormalizeCtx) -> Result<GenerationRequest, ApiError>;
    fn submit_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
}
pub trait JobView: Send + Sync + 'static {
    fn status_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
    fn result_reply(&self, job: &Job, cx: &ViewCtx) -> HttpReply;
}
pub struct ViewCtx<'a> { pub now: OffsetDateTime, pub urls: &'a dyn UrlSigner,
                         pub public_base: &'a url::Url, pub with_logs: bool }
pub trait UrlSigner: Send + Sync { fn url_for(&self, a: &Artifact, ttl: Duration) -> url::Url; }

pub struct TrackSet { pub video: VideoTrack, pub audio: Option<AudioTrack> }
pub struct VideoTrack { pub name: String, pub width: u32, pub height: u32, pub fps: u32 }
pub struct AudioTrack { pub name: String, pub rate: u32 /* always 48000 on the wire */, pub channels: u8 }

pub trait StreamProtocol: Send + Sync + 'static {
    fn id(&self) -> ProtocolId;
    /// Track names and channel count this protocol uses for a model (Reactor: main_video/main_audio, mono).
    fn tracks(&self, caps: &ModelCaps, canvas: (u32, u32), fps: u32) -> TrackSet;
}

pub struct RgbFrame { pub width: u32, pub height: u32, pub data: bytes::Bytes /* RGB24 */, pub index: u64 }
pub struct Pcm { pub rate: u32, pub channels: u8, pub samples: std::sync::Arc<[f32]> /* interleaved */ }
```

Every method of `SubmitEndpoint` and `JobView` is **pure**. Golden tests call
them directly (§7.1). serve-kit supplies one generic axum handler per shape:
`submit<E: SubmitEndpoint>`, `status<V: JobView>` and `result<V>`. That
handler does auth, then ingestion, then `negotiate`, then
`engine.submit`, then `JobStore`.

### 3.6 Engine service API

```rust
pub struct EngineService { /* Arc inner; cloneable */ }
impl EngineService {
    pub fn start(cfg: EngineConfig, backends: Vec<Box<dyn EngineBackend>>) -> Self; // one per GPU
    pub fn caps(&self) -> &CapabilityTable;                 // HashMap<ModelId, ModelCaps>
    pub fn readiness(&self) -> Readiness;                   // Loading{done,total} | Ready | Failed(String)
    pub async fn submit(&self, job: JobId, r: ResolvedJob, prio: Priority)
        -> Result<JobHandle, ApiError>;                     // QueueFull / Loading / Unsupported
    pub async fn open_clip_session(&self, spec: SessionSpec) -> Result<ClipSession, ApiError>;
    pub async fn open_causal_session(&self, spec: SessionSpec) -> Result<CausalSession, ApiError>;
    pub async fn drain(&self, grace: Duration);             // shutdown
}
pub enum Priority { Stream, Batch }
pub struct JobHandle { pub events: tokio::sync::mpsc::Receiver<EngineEvent>, pub cancel: CancelToken }
pub enum EngineEvent {
    Queued { position: u32 }, Started, Stage { name: &'static str },
    Progress { step: u32, total: u32 }, Log(LogLine),
    Finished(ClipOutput), Failed(ApiError), Cancelled,
}
pub struct ClipOutput { pub mp4: Option<PathBuf>, pub frames: Option<Vec<RgbFrame>>,
                        pub audio: Option<Pcm>, pub metrics: JobMetrics }

/// Runs on the executor thread. Never async; never touches tokio.
pub trait EngineBackend: Send + 'static {
    fn device(&self) -> DeviceInfo;
    fn caps(&self) -> Vec<ModelCaps>;
    fn load(&mut self, model: &ModelId, obs: &mut dyn FnMut(LoadEvent)) -> Result<(), ApiError>;
    fn generate(&mut self, job: &ResolvedJob, out: &mut dyn ClipSink, ctl: &StepControl)
        -> Result<ClipOutput, ApiError>;
    fn causal_open(&mut self, s: SessionId, spec: &CausalSpec) -> Result<(), ApiError>;
    fn causal_block(&mut self, s: SessionId, input: &BlockInput, out: &mut dyn ClipSink,
                    ctl: &StepControl) -> Result<BlockStats, ApiError>;
    fn causal_close(&mut self, s: SessionId);
}
pub trait ClipSink { fn frames(&mut self, f: &[RgbFrame]); fn audio(&mut self, pcm: &Pcm); }
pub struct StepControl { pub cancel: CancelToken, pub progress: Box<dyn FnMut(u32, u32) + Send> }
```

`StepControl` plugs into the existing observers:

- Wan `StepObserver = dyn FnMut(&DenoiseStep) -> Result<()>`
  (`wan/pipeline.rs:261`);
- LTX `StepObserver = &mut dyn FnMut(usize, &CudaTensor, &CudaTensor, f64) -> Result<()>`
  (`ltx2/pipeline.rs:716`).

A cancelled token makes the observer return `Err(Cancelled)`, which unwinds
at the next step. H3 has no observer yet; package E1 adds one.

---

## 4. Per-API mappings

Common conventions for every API:

- Output files are served at native `GET /files/{artifact_id}/{file_name}?exp=&sig=`,
  with HMAC-SHA256 over `artifact_id|file_name|exp`.
  - The route is unauthenticated and sets `Access-Control-Allow-Origin: *`,
    because fal, LTX and MiniMax clients download without a key (fal §11,
    ltx §2.1).
  - On `runpod-queue`, the `ArtifactStore` is S3 instead, and URLs are
    presigned (§6.2).
- MP4s come from the engine's existing `VideoWriter` mux
  (`wan/writer.rs`: libx264 crf 19 veryfast yuv420p, plus AAC at the native
  audio rate).
- `fastvideo-media::mp4::finalize` then does two things:
  - it remuxes with `-c copy -movflags +faststart`, because fal clients see
    faststart files (fal §6);
  - it applies `PostProcess`: `-an` for `Silent`, and a crop, which
    re-encodes with the same x264 settings, for pad-and-crop canvases.
- H3 output is therefore H.264 plus AAC stereo at 32 kHz, the same as hosted
  (fal §6).
- A fresh seed is drawn when none was sent and is stored on the job.

### 4.1 FastVideo `/v1/videos` (`fastvideo-openai-videos`)

Source: minimax-fastvideo §2.1 and §3.3.

| Route | Behaviour |
|---|---|
| `POST /v1/videos`, `POST /v1/videos/generations` | JSON, multipart or form (fields `image_reference`… as JSON strings). `extra_body`/`extra_json` are merged. `extra="forbid"`: unknown field → 400. Reply `VideoResponse` with `status:"queued"` |
| `POST /v1/videos/sync` | Blocks. Returns `video/mp4` with `X-Request-Id`, `X-Model`, `X-Inference-Time-S`, `X-Stage-Durations`, `X-Peak-Memory-MB` |
| `GET /v1/videos?after&limit&order` | `{object:"list",data,first_id,last_id,has_more}`, default `desc` |
| `GET /v1/videos/{id}` | `VideoResponse`. A failed job is **200** with `status:"failed"`, `error:{code:"generation_failed",message}`. A completed job's `url` is a signed download URL of the MP4 (1 h, re-signed on every retrieve; FastVideo always sends `null`); other statuses keep `url: null` |
| `GET /v1/videos/{id}/content?variant=video` | File. Other variant → 400; failed → 422; not done → 404 `"Generation is still in-progress"` |
| `DELETE /v1/videos/{id}` | `{id,deleted:true,object:"video.deleted"}`. A running job is **cancelled** through `CancelToken`. This is a deliberate improvement: FastVideo can't interrupt, and our observer can |
| `GET /v1/models`, `/v1/models/{model}`, `GET /v1/model_info` | Cards `{id,object:"model",created,owned_by:"fastvideo",root}` for every `served_names` entry |

Field → normalized → engine (only the differences from minimax-fastvideo §3.3):

| Field | Normalized | Behaviour |
|---|---|---|
| `model` | alias lookup | Unknown → 400 |
| `size`, `width`/`height`, `video_params`, `aspect_ratio`+`short_edge` | `CanvasSpec::Exact` / `Aspect` | Precedence as FastVideo. H3 short edge must be 768 (or 480 after E3) |
| `seconds` / `num_frames` / `fps` | `Length::Seconds{AlignUp}` / `Frames{Exact}` | H3 fps ≠ 24 → 400. LTX fps ∉ caps → 400 `Unsupported(LtxFps)` until E4. Wan fps is container-only (`GenerateConfig.fps`) |
| `task` + `image_reference`/`video_reference`/`audio_reference`/`input_reference` | `T2V`/`Keyframes`/`I2V`/`Ref2V` | FastVideo reorders references to images, videos, audio; **we do too on this API only** (parity), while MiniMax keeps content order |
| `n`/`num_outputs_per_prompt` ≠ 1, `file_id` refs, `lora` ≠ startup adapter, `enable_frame_interpolation*`, `enable_teacache`, `true_cfg_scale`, `max_sequence_length`, `start_time_seconds`, `sound_duration` | — | 400 |
| `num_inference_steps` on H3 | — | 400 `Unsupported(PerRequestSteps)`. LTX and Wan honour it |
| `generate_sound` | — | Accepted no-op, as in FastVideo (H3 and LTX always have audio; Wan never) |

Status mapping: `Queued` → `queued`, `Running` → `in_progress`,
`Succeeded` → `completed`, and `Failed`/`Cancelled` → `failed`.
`progress = round(progress*100)`, where FastVideo only ever sends 0 or 100
(a superset, and OpenAI-valid).

Errors use `{"error":{"message","type":"invalid_request_error"|"server_error","param","code":<http int>}}`.
Validation errors are 400, never 422.

### 4.2 FastWan Video API (same crate)

Source: minimax-fastvideo §2.2 and §2.4.

| Route | Behaviour |
|---|---|
| `POST /generate` `{prompt,width,height,num_frames,fps,seed}` | 200 `{prompt_id,status:"queued"}` |
| `GET /status/{prompt_id}` | 200 `{status, error?}`. `status ∈ queued\|processing\|completed\|failed`, and `error` is a **string** |
| `GET /video/{prompt_id}` | Raw MP4 bytes |
| `DELETE /video/{prompt_id}` | Cancel or delete; 200 `{}` |
| `GET /health` (shared) | Must carry truthy `model_loaded` |
| `GET /` (shared) | `{"model": <served name>}` |

- `num_frames` must be on the Wan `4k+1` grid.
- Errors use FastAPI style `{"detail": "<string>"}`.
- 400, 413, 415 and 422 mean rejected. **Loading returns 503**, because the
  client treats any non-4xx as unreachable and retries (§2.2).

### 4.3 MiniMax V2 (`fastvideo-minimax`)

Source: minimax-fastvideo §1.2-1.7 and §3.2.

| Route | Behaviour |
|---|---|
| `POST /v2/video_generation` | 200 `{"task_id":"<18-digit>"}`: **no `base_resp`**, unlike strobe's adapter (§1.9) |
| `GET /v2/query/video_generation/{task_id}` | `{"task":VideoTask}`. `content.url` is re-signed on every query (TTL `minimax.url_ttl`, default 24 h). `error` only when failed; `usage` only on success |
| `GET /v2/query/video_generation` | `page_num`, `page_size`, `filter.status`, `filter.task_ids`, `filter.model`, `filter.task_type` → `{items,total}` |
| `DELETE /v2/video_generation/{task_id}` | `queued` → cancel (`action:"cancelled"`). `succeeded`/`failed` → delete (`action:"deleted"`). `running`/`cancelled` → 400 |
| `POST /v2/h3_context_ir`, `POST /v2/video_regeneration` | 400 `bad_request_error` "not supported by this server (2013)" |

| MiniMax field | Normalized | Engine / result |
|---|---|---|
| `model: MiniMax-H3` | alias → `minimax.models."MiniMax-H3"` (default: the resident H3 model) | OK |
| `model: MiniMax-H3-Max` | alias → `minimax.models."MiniMax-H3-Max"` (default: same resident FastH3 recipe) | Documented **quality substitution**: no H3-Max weights exist (§3.5 #3). Durations of 4 → 400 per spec |
| `content[text]` (exactly one, ≤7000) | `prompt` | 0 or 2+ texts → 400 `(2013)` |
| `image_url` roles `first_frame` (or roleless single) / `last_frame` | `Keyframe` | `I2V` or `Keyframes` → H3 fl2va |
| `reference_*` | `Reference`, **content order kept** | `Ref2V`. The ref2va DiT must be resident, else 400 `Unsupported(H3Ref2vaNotLoaded)`. Mixed with frame roles → 400 |
| `url` = `mm_file://…` | — | 400 `Unsupported(ProviderFiles)` |
| `resolution: 768P` + `ratio` | `Aspect{ratio, 768}` | `resolve_canvas_size`. `t2va` + `adaptive` → 400 (spec). i2va forces `adaptive` → `FollowImage` |
| `resolution: 480P` | `Aspect{ratio, 480}` | Needs E3 (short-edge 480 helper); until then 400 |
| `resolution: 2K` | — | 400 `Unsupported(H3Resolution2K)` |
| `duration` 5–15 | `Seconds{AlignUp}` → 17n+5 | OK |
| `duration` 4 (`MiniMax-H3` only) | — | 400 `Unsupported(H3FourSeconds)` until E3 |
| `extra.prompt_expansion_mode` (H3-Max only; enum checked) | `accepted_noop` | We have no prompt expander. Rejected for `MiniMax-H3` as the schema says. The query has no expanded-prompt field, so nothing changes on the wire |
| `callback_url` | `CallbackSpec::MiniMax` | `POST {"challenge":<random>}`, which must be echoed within 3 s, then `POST {"task":…}` on every status change (INFERRED V1 form, §1.6). An SSRF guard applies |

Status: `Queued` → `queued`, `Running` → `running`, `Succeeded` → `succeeded`,
`Failed` → `failed`, `Cancelled` → `cancelled`.

`usage` on success: `total_seconds`, `output_seconds`, `input_seconds`,
`input_image_count`, and `input_audio_seconds` when audio references exist.
Token fields are omitted, because we have no tokenizer billing (open question
Q7).

Errors use the `OaiError` envelope
`{"type":"error","error":{"type","message":"… (<code>)","http_code":"<n>"},"request_id"}`:

| Kind | HTTP | `error.type` | Code |
|---|---|---|---|
| InvalidRequest / Unsupported / PayloadTooLarge (64 MB body) | 400 | `bad_request_error` | 2013 |
| Unauthorized | 401 | `authorized_error` | 1004 |
| ContentFiltered | 422 | `unprocessable_entity_error` | 1026 |
| RateLimited / QueueFull | 429 | `rate_limit_error` | 1002 |
| Loading | 529 | `overloaded_error` | — |
| Internal / EngineFailed at submit | 500 | `server_error` | 1000 |
| Unknown `task_id` | 404 | `bad_request_error` "invalid task_id (2013)" | — |

The unknown-`task_id` status code is INFERRED from strobe.

A failed job shows `task.error = {code:"1000", message}`.

### 4.4 fal queue and sync (`fastvideo-fal`)

Source: fal §3-§5, §9-§12.

Apps are static routes built from `fal.apps` (default `["minimax/h3-max"]`),
**never wildcards**. That keeps them from shadowing `/v1/videos/sync` and
similar paths.

| Route | Behaviour |
|---|---|
| `POST /{app}/{sub}` (`sub ∈ text-to-video, image-to-video, reference-to-video`) | Queue submit, **HTTP 200** `{request_id,response_url,status_url,cancel_url,queue_position}`, with **app-only** URLs (no sub-path, no `/response`). `?fal_webhook=` and `?fal_max_queue_length=` (queue deeper than that → 429). Header `x-fal-request-id` |
| `GET /{app}[/{sub}]/requests/{id}[/response]` | Bare output JSON plus `x-fal-request-id`. Not finished → 400 `{"detail":"Request is still in progress"}`. Failed → the error's HTTP status with `detail` (fal §13 recommendation) |
| `GET …/requests/{id}/status?logs=` | `logs` accepts `1\|0\|true\|false`. `IN_QUEUE` always has `queue_position`; `IN_PROGRESS`/`COMPLETED` always have `logs` (possibly `[]`). Every body carries `request_id`, `response_url`, `status_url` and `cancel_url`. Unknown id → 404 `{"status":"NOT_FOUND"}` |
| `GET …/requests/{id}/status/stream` | SSE: one status object per change, closed after `COMPLETED` |
| `PUT …/requests/{id}/cancel` | 202 `{"status":"CANCELLATION_REQUESTED"}` / 400 `{"status":"ALREADY_COMPLETED"}` / 404 `{"status":"NOT_FOUND"}` |
| `POST /run/{app}/{sub}` | Sync: output JSON on the same connection, plus `x-fal-request-id` |
| `ANY /fal/proxy` | Proxy mode for JS `proxyUrl`: route by the `x-fal-target-url` header. Host `queue.fal.run` → queue; `fal.run` → sync or director `…/ice`; `rest.fal.ai/storage/upload/initiate` → storage; `wma.fal.run` → director |
| `POST /storage/upload/initiate?storage_type=` `{content_type,file_name}` | `{upload_url: <our PUT /uploads/{token}>, file_url: <signed /files URL>}` for JS `transformInput` |

Configuring each client (fal §12):

- **Python**: `FAL_RUN_HOST=<host>/run`, which is a URL paste (INFERRED), and
  `FAL_QUEUE_RUN_HOST=<host>`. Python needs https. Python uploads cannot be
  redirected, so pass URLs or data URIs.
- **JS**: `requestMiddleware` rewrites `https://queue.fal.run/` → `<base>/`,
  `https://fal.run/` → `<base>/run/` and `https://wma.fal.run/` →
  `<base>/wma/`. Alternatively `proxyUrl: {url: "<base>/fal/proxy", when: "always"}`.

| fal field | Normalized | Engine / result |
|---|---|---|
| `prompt` (1..50000) | `prompt` | OK |
| `duration` 5..15 | `Seconds{AlignUp}` | 5 → 124 frames, as hosted (5.167 s) |
| `resolution` `768P` / `480P` / `1080P` | `short_edge` 768 / 480 / — | 480P needs E3 (832×480 at 16:9, fal §6), else 422. 1080P → 422 `Unsupported(H3Refine1080P)` |
| `aspect_ratio` (t2v; r2v adds `adaptive`) | `Aspect` / `FollowImage` | `resolve_canvas_size` |
| `image_url` / `end_image_url` (i2v) | `Keyframe{First/Last}`. Neither → `T2V` (spec) | fl2va; the canvas follows the image |
| `reference_{image,video,audio}_urls` (r2v) | `Reference` in list order: images, then videos, then audio, following the "Image 1… Video 1…" numbering | Ref2V. Output includes required `seed` |
| `target_audio_url` | `AudioInput{TargetSoundtrack}` | 422 `Unsupported(H3TargetAudio)` until E10 |
| `seed` | `seed` | Drawn if omitted |
| `enable_safety_checker` | `accepted_noop` | No checker (risk R8) |
| `prompt_expansion_mode` (default `balanced`, all accepted) | `accepted_noop` | Output `expanded_prompt: null`. The schema allows null "when prompt expansion was disabled, left the prompt unchanged" (fal §3.4) |
| `sync_mode: true` | `inline_data_uri` | `video.url = data:video/mp4;base64,…` (INFERRED, fal §14 #1) |

Output: `{video:{url,content_type:"video/mp4",file_name:"<nanoid21>_<app-slug>.mp4",file_size}, expanded_prompt:null, seed, timings:{inference:<denoise s>}}`.
`<app-slug>` is `minimax-<alias>` for the `minimax/*` apps, else the app
alias, plus `-<tier>` when the resolved tier is not already a word of it
(`minimax-h3-max`, `minimax-h3-turbo`, `ltx-2.5-turbo`, `wan-turbo`;
`fastvideo_fal::output_file_name`). Hosted fal writes `minimax-h3` for
every H3 app, which hides the tier.
`seed` is the effective seed (the request's, or the one the server drew)
on every task: fal's r2v schema requires it, and on t2v/i2v it is an extra
key the clients ignore.

`timings` is fal's `object<string, number>` (seconds). `inference` keeps
fal's meaning, the denoise only (also the status `metrics.inference_time`).
The other keys are our breakdown, which the schema allows and the clients
pass through untouched:

| Key | Carries |
|---|---|
| `inference` | Denoise wall time (fal: "the DiT denoising time") |
| one key per engine stage (`text`, `refine`, `denoise`, `audio_decode`, `video_decode`, `encode`, as in `X-Stage-Durations`) | That stage. `text` includes the I2V multimodal text encoder, which `inference` does not show |
| `queue` | Submit to engine start |
| `total` | Engine start to completion; `total - inference` is the time outside the denoise |

Status: `Queued` → `IN_QUEUE`, `Running` → `IN_PROGRESS`, and
`Succeeded`/`Failed`/`Cancelled` → `COMPLETED`. Failed adds `error` and
`error_type`; cancelled adds `error_type:"client_cancelled"`. **Never** a
non-standard status string, because Python raises on it (fal §13).

Errors:

| Kind | fal response |
|---|---|
| InvalidRequest / Unsupported | 422 `{"detail":[{loc,msg,type:"value_error"}]}` |
| ContentFiltered | 422 `type:"content_policy_violation"` |
| Unauthorized | 401 `{"detail":"invalid key credentials"}` |
| QueueFull | 429 `{"detail","error_type":"concurrent_requests_limit"}` + `X-Fal-Needs-Retry: 1` |
| Loading | 503 `runner_scheduling_failure` |
| Timeout | 504 `request_timeout` |
| Internal | 500 `internal_error` |

Every error also carries the `X-Fal-Error-Type` header.

Webhooks: POST `{request_id,gateway_request_id,status:"OK"|"ERROR",payload}`
with the documented retry schedule. They are signed with **our** Ed25519 key
using fal's header scheme. We publish our JWKS at
`/.well-known/jwks.json`, since fal-signed webhooks are impossible (fal §9.7,
risk R9).

### 4.5 LTX API (`fastvideo-ltxapi`)

Source: ltx §1-§3 and §5.

| Route | Behaviour |
|---|---|
| `POST /v2/text-to-video`, `POST /v2/image-to-video` | **202** `{id,created_at}` |
| `GET /v2/{endpoint}/{id}` | `oneOf`: `pending`/`processing` `{status,id,created_at}`; `completed` adds `completed_at` and `result:{video_url}`; `failed` adds `completed_at` and `error:{type,message}`. Wrong endpoint segment → 404 |
| `POST /v1/text-to-video`, `POST /v1/image-to-video` | Sync. 200 with `Content-Type: video/mp4` bytes; generation over `ltx.sync_timeout` → 504 |
| `POST /v1/upload` | 200 `{upload_url, storage_uri:"ltx://uploads/<token>", expires_at, required_headers:{}}`. `PUT /uploads/{token}` accepts and ignores the `x-goog-*` headers clients copy |
| `/v1\|v2/{audio-to-video,retake,extend,video-to-video-hdr,video-to-video-reframe}` | 403 `permission_error` |

Every reply carries `x-request-id` (32 hex characters).

| LTX field | Normalized | Engine |
|---|---|---|
| `model` | `ltx.models` map. Defaults: `ltx-2-3-fast` → ltx2_distilled_23 two-stage; `ltx-2-5-fast` → ltx2_distilled_25 two-stage Sol stage 2; `ltx-2-5-pro` → ltx2_distilled_25 two-stage **dense** stage 2 (a documented downgrade, ltx §3.1); `ltx-2-3-pro` → `ltx2_base_23` if weights are configured, else 403. `ltx-2-fast`/`ltx-2-pro` → 400 (removed upstream) | |
| `duration` (6..20 per matrix) | `Seconds{AlignUp}` → 8k+1 | `null` → 400 `Unsupported(LtxAutoDuration)`. The model matrix (ltx §3) is enforced |
| `fps` 24 (default) / 25 / 48 / 50 | `fps` | Only 24 until E4 validates the rest → 400 `Unsupported(LtxFps)` |
| `resolution` `WxH` | `Exact` with `pad_and_crop` | 1920×1080 → 1920×1088 → crop; 1280×720 → 1280×768 → crop; 3840×2160 → 3840×2176 → crop; portrait by transpose |
| `generate_audio: false` | `AudioOut::Silent` | Post `-an` at launch. E4 also skips the audio decode |
| `image_uri` | `Keyframe{First}`: `I2V` | 2.3 OK. 2.5 → 400 `Unsupported(Ltx25I2V)` until E5 |
| `last_frame_uri` | `Keyframe{Last}` | 400 `Unsupported(LtxKeyframes)` until E9 |
| `camera_motion` | — | 400 `Unsupported(LtxCameraMotion)` |
| `prompt` ≤5000 | `prompt` | |

Media refs: `ltx://uploads/<token>` goes through `UploadStore`. HTTPS URLs
follow ltx §2.0: https only, no IPs, **no redirects**, 10 s timeout for
images and 30 s for video/audio, 15 MB for images and 32 MB for video/audio.
Data URIs are capped at 7 MB and 15 MB encoded.

Status: `Queued` → `pending`, `Running` → `processing`, `Succeeded` →
`completed`, and `Failed`/`Cancelled` → `failed` (`api_error` for cancelled).

Errors are `{"type":"error","error":{"type","message"}}`:

| HTTP | Kind | `error.type` |
|---|---|---|
| 400 | Invalid / Unsupported | `invalid_request_error` |
| 401 | Unauthorized | `authentication_error` |
| 403 | Forbidden | `permission_error` |
| 404 | NotFound | `not_found_error` |
| 422 | ContentFiltered | `content_filtered_error` |
| 429 | QueueFull (v2) | `rate_limit_error` + `Retry-After` |
| 429 | concurrency (v1) | `concurrency_limit_error` + `Retry-After` |
| 500 | Internal | `api_error` |
| 503 | Loading | `service_unavailable_error` |
| 529 | overloaded | `overloaded_error` |

### 4.6 Error-to-HTTP summary across APIs

| Kind | FastVideo | FastWan | MiniMax | fal | LTX |
|---|---|---|---|---|---|
| InvalidRequest / Unsupported | 400 | 400 | 400 | 422 | 400 |
| Unauthorized | 401 | 401 | 401 | 401 | 401 |
| NotFound | 404 | 404 | 404 | 404 | 404 |
| QueueFull | 429 | 429 | 429 | 429 | 429 |
| Loading | 503 | 503 | 529 | 503 | 503 |
| EngineFailed (async) | job `failed` | `failed`+error | `failed`+error | `COMPLETED`+error | `failed`+error |

### 4.7 Engine gaps: 4xx now versus later work

| Gap | Hit by | Now | Package |
|---|---|---|---|
| H3 4 s (107 frames) | MiniMax `duration:4` | 400 | E3 |
| H3 480P canvas helper | MiniMax/fal `480P` | 400/422 | E3 |
| H3 2K, fal 1080P refine, Context-IR, regeneration | MiniMax, fal | 400/422 | none (permanent) |
| H3-Max weights | MiniMax, fal | alias with a documented substitution | none |
| H3 target/conditioning audio | fal `target_audio_url`, director `audio_url` | 422 / `prompt_rejected{invalid_audio}` | E10 |
| H3 ref2va co-resident with fl2va | MiniMax/fal/FastVideo ref2v | 400 unless configured | E11 |
| LTX fps 25/48/50 | LTX, FastVideo | served (validated at 1080p, `artifacts/serve/e4-ltx-fps/benchmark.json`; engines without them in caps still 400) | E4 done |
| LTX silent output | LTX `generate_audio:false` | supported (post `-an`); the engine can skip the audio decode (`Ltx2Request::skip_audio_decode`) | E4 done |
| LTX-2.5 I2V | LTX `image_uri` on 2.5 | 400 | E5 |
| LTX last frame | LTX `last_frame_uri` | 400 | E9 |
| LTX auto duration, camera motion, A2V/retake/extend/HDR/reframe | LTX | 400 / 403 | none planned |
| Cancellation mid-generation | all DELETE/cancel | cancels only while queued | E1 |
| In-memory frames (no PNG) | streaming | `fastvideo_cudarc::sink::FrameSink` via `Hooks::with_sink` | E2 done |
| SF-Wan open-ended block stream | Reactor causal, WHIP | blocked | E6 |
| SF-Wan real-time fps | causal streaming quality | ~18 fps (streaming-refs §3.3) | E7 |
| MMAudio V2A sidecar | video-only plus `AudioOut::Sidecar` | 400 | E8 |

---

## 5. Streaming design

### 5.1 Shape

```
front-end (fal director | Reactor | native /fv/v1/streams)
   │ SessionSpec{model, mode, tracks, canvas, fps, continuity}
   ▼
EngineService ── ClipSession  (H3, LTX, FastWan, TI2V-5B; generation queue ─► build on executor)
             └─ CausalSession (SF-Wan; one block per executor turn)
   │ tokio::mpsc bounded (clips: playout capacity; blocks: depth 4)
   ▼
AvPacer (media)  — one tick = 1 video frame + 48000/fps audio samples (or silence)
   ▼
Encode once per session: OpenH264 (CB, IDR 2 s) + Opus 48 kHz (20 ms frames)
   ▼
Fan-out: WebRTC peers (str0m) | WHIP publisher | RTMP/HLS (ffmpeg) | recorder
```

- The executor thread owns the CUDA context and never awaits.
- Media conversion work never runs on the executor: RGB→I420, resampling and
  encoding.
- Encoding happens **once per session**, and the bitstream is fanned out.
  - A PLI or FIR from any peer is answered by a keyframe within 1 s: a
    keyframe sent less than 1 s before it, or the periodic IDR when it is
    due within 1 s, covers it; otherwise it forces an IDR, rate-limited to
    one per second (`fastvideo_media::video::KeyframePolicy` for WHIP
    streams; the Reactor runtime rate-limits PLI keyframes per codec).
    MediaMTX asks its WebRTC publishers for a keyframe every 2 s; with the
    2 s GOP none of those forces an IDR (a forced NVENC IDR restarts the
    ffmpeg process).
  - This loses per-peer bitrate adaptation, which RT gets from libwebrtc. In
    return we pay for one encoder per session, not one per viewer.
  - The target bitrate is fixed by config: 6 Mb/s at 768p and 2.5 Mb/s at
    480p by default.

### 5.2 Session lifecycle (protocol-agnostic)

```rust
pub enum SessionState { Starting, Ready, Streaming, Orphaned, Closing, Closed(EndReason) }
pub enum EndReason { Stopped, TimedOut, SessionLimit, Evicted, Error(ApiError), ClientGone }
pub struct SessionSpec {
    pub model: ModelId, pub tracks: TrackSet, pub canvas: (u32, u32), pub fps: u32,
    pub continuity: Continuity, pub max_seconds: Option<u32>, pub seed: Option<u64>,
}
pub enum Continuity { HardCut, Crossfade { ms: u16 }, AnchorLastFrame { crossfade_ms: u16 } }
```

1. **Admission.**
   - There is one streaming session per executor. A second session gets the
     protocol's "busy" answer: Reactor 409, WMA `/session` 429,
     `/fv/v1/streams` 429.
   - A session that is still `Starting` counts as busy (strobe's OOM lesson,
     streaming-refs §1.3).
   - The model must be resident. Otherwise the answer is 503 with
     `Retry-After`.
2. **Starting.**
   - Build `TrackSet`, the pacer and the encoders.
   - Transports that we initiate wait for the first frame before the
     handshake: WHIP (streaming-refs §1.2, "don't offer an empty track").
   - Answer-side transports (Reactor, WMA) answer immediately and emit the
     first frame when it is ready.
   - Before the first frame, the video track is silent. There is no fake
     black encode, except for Reactor's start-of-connection black frame
     (reactor §4.2).
3. **Streaming.** The duration clock starts at the first emitted frame
   (streaming-refs §1.2). `max_seconds` is enforced in video time.
4. **Orphaned**: all peers are gone.
   - Generation pauses: clip builds stop and causal blocks stop.
   - After `orphan_timeout` (60 s, as RT) the session enters `Closing`.
5. **Closing.** The protocol farewell goes out first:
   - Reactor `session_ended{reason}`;
   - WMA `stream_exhausted{chunks,reason}`;
   - WHIP `DELETE`.

   Then the engine session is released, which frees the executor for batch
   work.

**Scheduling decision.** While a `CausalSession` is streaming, it holds an
**exclusive** executor lease, and batch jobs wait in queue: they report
`IN_QUEUE` or `queued` and obey the queue limits.

A `ClipSession` submits each build as a `Priority::Stream` job. The next
batch job runs only when the session's playout queue is full or idle. Clip
builds and batch jobs therefore interleave at clip granularity, and
streaming wins ties.

### 5.3 Track declaration

`TrackSet` is fixed at session creation from `ModelCaps.audio` and the
request (`AudioOut::Sidecar` turns on MMAudio for video-only models):

| Transport | Video+audio model | Video-only model |
|---|---|---|
| Reactor | `capabilities.tracks = [main_video, main_audio]` in all three places (descriptor, `/schema` `x-reactor.tracks`, `POST /connections` `track_map`), `direction:"recvonly"` client-side, `out` model-side | `[main_video]` only; the client builds no audio transceiver (reactor §4.5) |
| fal WMA | Answer the offered video and audio m-lines `sendonly` | The offer includes audio (media contract "optional"). Answer that m-line `a=inactive` and send nothing |
| WHIP (we offer) | video + audio m-lines | **video m-line only** |
| RTMP / HLS | AAC | **Always an AAC track**, silence-filled (platforms reject video-only FLV, streaming-refs §4.2) |
| Batch MP4 | audio stream | no audio stream |

Audio wire format:

- 48 kHz on every WebRTC path.
- Reactor: **mono** (mean downmix), for parity with RT's `push_pcm(…,48000,1)`
  and the working fast-h3 client (reactor §4.5).
- fal WMA and WHIP: **stereo** (Opus stereo; `audio_bitrate` 96/128/192k
  when configured, else 96k).
- RTMP/HLS: AAC 128k at 48 kHz.
- Native rates are resampled in `fastvideo-media`: H3 32 kHz, LTX from the
  vocoder config (24 kHz on 2.0; 48 kHz BWE on 2.5), MMAudio at its model
  rate. Director conditioning audio at 32 kHz is the reverse resample,
  reserved for E10.

### 5.4 Real-time causal SF-Wan (`CausalSession`)

- **Engine.** E6 adds `wan::stream::CausalRollout`: an open-ended block loop
  over the existing `CausalKvCache` (`wan/causal.rs`) with rolling
  `local_attn_size` and `sink_size`.
  - Each executor turn denoises one block (3 latent frames → 12 pixel
    frames), writes the clean-context KV, and decodes that block with
    **TAEHV with carried temporal state**.
  - If the E6 parity test shows a seam, the fallback is strobe's latent
    overlap with the `4·n` keep rule (streaming-refs §1.5).
  - The frames go to `ClipSink`.
  - **Status (E6 landed, measured on H100):** no seam, so no overlap: the
    per-block decode is bitwise the whole-clip decode at chunk 3 (81.4 dB
    against the default chunk 4). Default RoPE is `RebasedSink` (FastVideo's
    relativistic offsets at the absolute cost), sink 3 frames: 19.2 frames/s
    steady, TTFF 0.30 s warm, flat device memory over 10 minutes. Details
    and the long-run drift numbers: `docs/ports/wan.md` "SF-Wan open-ended
    streaming".
- **Prompt changes** apply at the next block boundary. The text encoder stays
  resident while a causal session is open (memory is recorded in caps).
  `reset` clears the KV and restarts at block 0.
- **Pacer** (ported strobe-core `FramePacer`):
  - drop-oldest when over `buffer_frames` (default 48);
  - freeze on underrun;
  - adaptive fps: EMA 0.3/0.7 clamped to `[min_fps=4, fps=16]`;
  - stats `pushed/served/dropped/underruns/unique_fps`.

  The RTP video timestamp advances by `90000/effective_fps`.
- **Backpressure**: an mpsc of depth 4 blocks from executor to pacer. The
  executor blocks on a full channel. That means generation is ahead of
  playout, which is desirable, and nothing is dropped upstream of the pacer.
- **Metrics**: TTFF phases (`load`, `first_block`, `transport`), block_ms,
  unique_fps. E7 (CUDA Graphs per block position) closes the gap from our
  ~547 ms/block to strobe's 316 ms (streaming-refs §3.3). Until then the
  default stream canvas is 832×480 at 16 fps.

### 5.5 Clip-queue playout (`ClipSession`) for H3, LTX and FastWan

This is the fast-h3 contract (reactor §4bis; streaming-refs §4.1),
re-implemented in Rust:

```rust
pub struct ClipInfo { pub clip_id: Uuid, pub prompt: String, pub metadata: String,
                      pub frames: u32, pub seconds: f64, pub seed: u64, pub ready: bool }
pub enum ClipCommand {
    Enqueue { prompt: String, metadata: String, seed: Option<u64>, seconds: Option<f64>, position: Option<u32> },
    Play { clip_id: Option<Uuid> }, Pop { clip_id: Uuid }, Move { clip_id: Uuid, position: u32 },
    Stop, Reset, SetClipSeconds(f64), SetSeed(u64), SetAutoplay(bool), SetCanvas(Aspect),
    GetQueue, GetState,
    Chunk { prompt_version: u64, prompt: String, end_image: Option<PathBuf>, seconds: f64 }, // director
}
```

- **Queues.** There are two bounded queues: generation (cap 20) and playout
  (cap 10, which is also the host-RAM budget, about 1 GB per 14 s at 768p).
  - One build is in flight at a time.
  - A build is submitted only when the playout queue has room (submit-time
    reservation).
  - A build whose entry was popped is discarded.
  - Builds run only while a peer is connected.
  - Autoplay is a standing `play`.
  - `valid_commands` is derived from `(queued, playing, autoplay, empty)`,
    as in `fasth3_session_rules`.
- **A/V lockstep.**
  - Each built clip is resampled to 48 kHz, then trimmed or padded to exactly
    `round(frames/fps·48000)` samples.
  - It is then emitted in 3-frame slices, each carrying
    `3·48000/fps` samples. That is 6000 at 24 fps.
  - Emission runs on a re-anchoring metronome that never bursts to catch up.
  - The pacer's video and audio FIFOs share one shallow cap (2 s). Every tick
    sends one frame plus exactly `48000/fps` samples, or silence
    (streaming-refs §4.2). **`48000 % fps == 0` is an admission check.**
- **Idle and underrun**: video holds the last frame, and audio is silent Opus
  frames sent continuously, so the audio clock never stops (RT feeder
  semantics).
  - After a clip ends with nothing armed, Reactor gets `flush` to black (one
    black IDR), as in fast-h3.
  - The fal director gets `deadline_missed{behavior:"freeze_video_and_silence_audio_until_ready"}`.
- **Continuity** (`Continuity`, per model in config):
  - `HardCut`: IL parity, and the Reactor default.
  - `Crossfade{ms: 20}`: a raised-cosine fade-out over the last 20 ms of clip
    N and a fade-in over the first 20 ms of clip N+1. Sample count is
    unchanged. This is the default for every clip session **other than**
    Reactor fast-h3 parity mode.
  - `AnchorLastFrame`: clip N+1 is built with `Keyframe{First}` = clip N's
    last frame. H3 uses fl2va; LTX uses I2V (2.3; 2.5 after E5). It also
    applies `Crossfade`.
    - It is the **fal director default**.
    - It forces sequential builds: clip N+1 can't start until clip N's last
      frame exists, so pipelining is limited to playout overlap.
    - It is not valid for FastWan FullAttn (T2V only).
- **Throughput honesty**: every clip reports `build_s/clip_s`.

### 5.6 fal director (WMA) mapping (`fastvideo-fal::director`)

Signalling, served under `/wma` and via `x-fal-target-url` host `wma.fal.run`
(fal §8.2):

| Route | Behaviour |
|---|---|
| `POST /wma/ice` `{app_id}` | `{"ice_servers":[…]}` from config |
| `POST /minimax/h3-max/director/ice` | Same body; the JS fallback path |
| `POST /wma/session` `{app_id,sdp,type:"offer"}` | Validate `app_id`, then non-trickle answer: `{session_id,sdp,type:"answer"}`. Busy → 429 |
| `POST /wma/session/heartbeat` `{session_id}` | `{alive}`. Three missed 5 s beats (15 s) → Closing(ClientGone) |
| `POST /start-session` | Runner side: SSE with the first event `data:{sdp,type:"answer",session_id}` and `: keepalive` every 15 s |
| `POST /info` | Runner side: `DirectorInfo` |

Control channel `control`:

- The client creates it and we accept it via `ChannelOpen`.
- It carries JSON text.
- Messages are validated **strictly**: extra properties → `error{code:"invalid_message"}`
  (diagnostic).

| Client message | Server behaviour |
|---|---|
| (channel open) | Send `session_info` with **our** constants: `fps:24`, `chunk_seconds`/`default_chunk_duration` = configured (10), `min_chunk_duration:5`, `max_chunk_duration:15`, `continuation_context_frames:1` (last-frame anchor), `audio_sample_rate:48000`, `conditioning_audio_sample_rate:32000`, `resolutions:["480p","768p"]` (after E3; `["768p"]` before), `aspect_ratios:["16:9","9:16","1:1"]`, `one_session_per_machine:true`, `audio_conditioning:false`, `scripts:true`, … |
| `configure` (once, `prompt_version:1`) | Validate. `resolution:"1080p"` → `error{code:"invalid_input"}`. `audio_url` → `error{code:"invalid_initial_audio"}` until E10. `image_url` → first chunk `Keyframe{First}`; `end_image_url` → first chunk `Keyframe{Last}`. `script` beats with only `prompt`/`end_image_url` are accepted; audio beats → `invalid_initial_script`. Reply `configured{prompt_version, enable_safety_checker:false, aspect_ratio, memory, chunk_duration, audio_bitrate, resolution, has_initial_image, has_initial_audio:false, acceleration:null}` and start chunk 0 |
| `prompt` v≥2 | Version ≤ last seen → `prompt_rejected{reason:"stale_prompt_version"}`. Otherwise `prompt_pending`. With `replan:true` it replaces the planned prompt for the next **undispatched** chunk; an older pending version may get no final event (`replace-pending`). With `replan:false` it appends. `prompt_applied` is sent when that chunk is dispatched. `audio_url` → `prompt_rejected{reason:"invalid_audio"}`. Planned deck full → `queue_full` |
| `ping{ts}` | `pong{client_ts}` |
| `stop` | Finish nothing new, then `stream_exhausted{chunks,reason:"stopped"}` and close |

- Chunks run as a `ClipSession` with `AnchorLastFrame` and autoplay.
  - `chunk` messages carry the required fields from our measurements:
    `generation_seconds`, `playback_seconds`, `generated_frame_count`,
    `buffer_depth_*`, `scheduling_*`, `dispatch{overhead_ms,wall_ms,phases_ms,classified_ms}`,
    and `route:"unknown"` (an allowed enum value).
  - `chunk_metrics` is sent per chunk, and `session_metrics` every 10 s and
    at the end with `final:true`.
- Late chunks produce `deadline_missed`.
- `max_session_seconds` → `stream_exhausted{reason:"session_limit"}`.
- The default chunk duration is 10 s → 17n+5 → 243 frames (10.125 s).

**WP-14 notes (as implemented).** `fastvideo-fal::director` (feature
`director`; fv-serve mounts it with `fal` + `webrtc`, merged into the fal
router so `/fal/proxy` reaches `/wma/*`).

- Routes: the table above plus `POST /wma/ice`, `POST /{app}/director/ice`
  and `POST /run/{app}/director/ice` (the JS `context.run` fallback, direct
  or through the proxy), and `GET` + `POST /info`. Every configured fal app
  gets a director (`minimax/h3-{max,turbo,draft}/director`); the first one
  answers the runner routes. Bridge errors are `{"error": ...}`, an
  unparsable body is 422 `text/plain` ("missing field `app_id`"), busy is
  429, not resident 503 + `Retry-After`. `/wma/session` carries `x-fv-model`,
  `x-fv-tier`, `x-fv-recipe` headers (the WMA bodies have closed schemas).
- Admission: the engine clip session is opened at `/session` (Starting is
  busy) at the default canvas and reopened on the configured canvas at
  `configure`. The engine seam is a thin trait (`DirectorEngine` /
  `DirectorClips`: `open`, `build`, `close`) because adapters may not depend
  on the engine crate; fv-serve implements it over
  `EngineService::open_clip_session` + `ClipSession::build`.
- Control (pure state machine, unit-tested): strict schemas (`invalid_message`
  on extra/mistyped/out-of-range fields or an unknown `type`, with the
  message's `prompt_version`); `configure` once, `prompt_version:1`;
  `immutable_settings` for a second one; `not_configured` before it;
  unserved resolution → `invalid_input`, `audio_url` → `invalid_initial_audio`,
  audio script beats / script + `end_image_url` → `invalid_initial_script`
  (all session failures: the session ends). Versions: ≤ last seen →
  `stale_prompt_version`, a version is spent whatever its outcome.
  `replan:true` clears the planned deck (replace-pending), `replan:false`
  appends (`prompt_deck_size` 6 → `queue_full`); `prompt_applied` at the
  chunk's dispatch. Scripts become chunks cut at beat offsets (end-image
  beats end a chunk exactly there, 5–15 s, ≥ 3 s apart, else
  `infeasible_timing`). No prompt expander: a chunk's prompt is the premise
  plus the current direction; `memory` is accepted and echoed.
- `session_info` / `DirectorInfo` report our constants (fps from the model,
  `continuation_context_frames:1`, `resolutions` from the canvas tiers,
  `prompt_expander:"none"`, `audio_conditioning:false`, …). `session_info`
  is sent when the client's `control` channel opens. The reserved
  `wma.network-info.request` is answered (`available:false`: str0m does not
  expose the selected pair here).
- Playout: one thread per session runs the `AvPacer` on a `Metronome` and
  encodes Opus; video is encoded on a second thread behind a 10-tick
  drop-oldest queue (§5.10), so a slow encoder never stops the audio clock.
  Continuations trim the duplicated anchor frame (`trimmed_context_frames:1`)
  and crossfade 20 ms; underruns hold the last frame with silence and the
  late chunk reports `deadline_missed`. At most `buffer_chunks` (1) built
  chunks wait behind the playing one. The answer puts video and audio in
  one `msid` stream (one `MediaStream` in the browser).
- Codecs: H.264 for every offer that has it (NVENC in production, OpenH264
  in CPU tests). Offers **without** H.264 (open-source Chromium, including
  Playwright's) get VP8 through `AnswerOptions::video_codecs =
  [H264, Vp8]` (`[director] vp8_fallback`): inter-frame through ffmpeg
  `libvpx` (`fastvideo_media::vp8`, like the Reactor runtime), intra-only
  libwebp when ffmpeg has no libvpx. (An earlier measurement of ~200 ms per
  832x480 libvpx frame was ffmpeg's default thread count: libvpx's VP8
  worker threads spin-wait and collapse on a contended CPU; one thread
  encodes 1344x768 at ~60-80 fps.)
- Tests: `crates/fastvideo-fal/tests/director_e2e.rs` (a str0m client:
  signalling, strict schemas, versions, heartbeat expiry, `/start-session`
  SSE, session limit, `deadline_missed`, A/V at 24 fps / 48 kHz stereo, a
  video-only model with the audio m-line `inactive`) and
  `director_browser.rs` (`@fal-ai/client@1.11.0-alpha.4`
  `fal.realtime.open(wma(...))` in Playwright Chromium, `requestMiddleware`
  and `proxyUrl` modes, A/V and video-only; decoded 24.0 fps VP8 832x480,
  48.0 k samples/s stereo Opus, alive past 17 s on heartbeats).

### 5.7 Reactor mapping (`fastvideo-reactor`)

Local-runtime routes exactly as reactor §3.2-3.3:

- fixed session id `00000000-0000-0000-0000-000000000000`;
- CORS `*`;
- state machine CREATED/READY/WAITING/STREAMING/ORPHANED/CLOSING/TERMINATED;
- 503 + `Retry-After: 1` while loading, 409 when not READY.

Signalling:

| Item | Behaviour |
|---|---|
| `POST /connections` | 201 `{connection_id:1002..9999, track_map}` |
| Offer | `POST`/`PUT …/sdp_params` → 202 |
| Answer | `GET …/sdp_params` → 202 until ready, then 200 **once** (taken) |
| `POST …/ice_candidates` | Buffered before the offer; limits 128×256 |
| Answers | Embed every candidate plus `a=end-of-candidates` |
| Deadlines | 30 s negotiation deadline; 64-connection limit; re-offers always admitted |
| Metadata trailer | We do **not** mirror `a=x-reactor-frame-metadata:1`, so there is no RXMT trailer (reactor §4.2) |

Wire handling:

- The client creates the `data` and `control` channels.
- Encoding is sniffed from the first inbound frame and latched: v1 protobuf
  (prost, generated from `crates/fastvideo-reactor/proto/`, which is copied
  from `docs/serve/reactor-proto/` with LICENSE and NOTICE) or v0 JSON.
- Any inbound message resets the 20 s watchdog.
- Outbound tracks start **paused** and are sent only after `ResumeTrack`.
- `RequestSchema` → `model_schema` (OpenAPI 3.1 from the command table).
- `RequestClip`/`RequestRecording` → `clip_failed{reason:"recording disabled"}`.
- `PublishTrack` → error `publish_refused` (we declare no IN tracks).
- A command reply is a correlated `ModelMessage`, or a bodyless ack (v1 only).
- Refusals are broadcast `command_error{command,reason}` plus a bodyless ack,
  as fast-h3 does.

Command sets, chosen by `ModelCaps.stream`:

| Mode | Commands (`Command.type`) | Messages |
|---|---|---|
| Clip (H3, LTX, FastWan) — **fast-h3 verbatim** | `enqueue{prompt≤800,metadata≤2000,seed?,seconds?,position?}`, `play{clip_id}`, `pop`, `move`, `stop`, `get_queue`, `get_state`, `set_clip_seconds`, `set_seed`, `set_autoplay`, `set_canvas{aspect∈16:9,1:1,9:16,4:3}`, `reset` | `clip_queued`, `clip_generated`, `clip_moved`, `clip_started`, `clip_finished`, `clip_stopped`, `clip_popped`, `clip_failed`, `clip_length_accepted`, `seed_accepted`, `autoplay_accepted`, `canvas_accepted`, `session_reset`, `state_update`, `queue_update`, `command_error` |
| Causal (SF-Wan), Waypoint-style `InputState` setters | `set_prompt{prompt}`, `set_paused{paused}`, `set_seed{seed}`, `reset` | `state_update{prompt,paused,seed,block_index,unique_fps}`, `command_error` |

- Clip-length bounds and snapping come from caps: H3 5.167–14.375 s on
  17n+5; LTX on 8k+1 at the model fps; FastWan on 4k+1.
- The fps is pinned to the model fps (24 for H3/LTX). Pacing is never
  measured for audio models (reactor §4.5).
- `set_canvas` is valid only when both queues are empty and nothing is
  playing.

### 5.8 Transport per deploy target (WHIP versus peer)

| Target | Peer WebRTC (Reactor, fal WMA) | WHIP publish (native streams) |
|---|---|---|
| Vast instance | **Primary.** UDP mux on the internal port with `-p 70010:70010/udp`; host candidate `PUBLIC_IPADDR:$VAST_UDP_PORT_70010`, plus ICE-TCP | Supported |
| Runpod pod | ICE-TCP passive only, on symmetrical port 70000 (`RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_70000`). Browsers support ICE-TCP; there is no UDP (deploy §2) | **Primary for broadcast.** Outbound UDP is expected to work (deploy §5.1 evidence) |
| Runpod serverless queue | Not supported (no inbound; the job owns the worker) | **Primary.** One job = one session (§6.2) |
| Runpod serverless LB | Not supported (HTTP/WS only; scale-down can kill sessions) | Not supported |

**WebRTC stack decision: str0m.**

- It is sans-IO. We own one UDP socket and one TCP listener and demultiplex
  all peers through `Rtc::accepts`. That is exactly the single-port NAT
  mapping Vast and Runpod need (deploy §3.2).
- Its frame-level writer takes **pre-encoded** H.264 access units and Opus
  packets and does RTP packetization. That is required, because we encode
  once per session.
- It has data channels (SCTP) and exposes ICE candidates as values we embed
  in non-trickle SDP.
- It is pure Rust, with no libwebrtc build in our Docker image.

Why not the alternatives:

- **reactor-webrtc**: libwebrtc parity, but a huge build, software encoding
  on push, and an unverified encoded-frame path (reactor §8.2, §9).
- **webrtc-rs**: async and heavier, and its pre-encoded sample track works,
  but it gives us no mux advantage.

Cost: **str0m has no TURN client**. Mitigations: ICE-TCP on the server, TURN
on the client side for peer mode, and relying on UDP egress for WHIP. This is
risk R2. The strobe workspace already pinned `str0m 0.21.0`
(streaming-refs §2.1).

**WHIP client** (`fastvideo-webrtc::whip`), per streaming-refs §1.9:

- H.264 is offered first, with no VP8.
- The offer is complete (non-trickle). Host candidates plus a STUN-derived
  srflx candidate are gathered before the POST.
- `Content-Type: application/sdp`; 200 or 201 is accepted. The `Location`
  header is resolved per RFC 3986 and kept for `DELETE`.
- Bearer or Basic auth; 30 s timeout.

### 5.9 Encoder decision

**As implemented (supersedes the text below, see §0 decision 1):** every
H.264 encoder setting — `[director] encoder`, `[reactor] h264`, `[webrtc]
encoder` (`/fv/v1/streams` WHIP, `FV_STREAM_ENCODER`) — defaults to `auto`.
At startup `fv-serve` runs one NVENC encode probe (ffmpeg `h264_nvenc`, a
few black frames) when any setting is `auto`, logs the choice, and resolves
`auto` to `nvenc` when the probe encoded, else `openh264`
(`fastvideo-serve::encoders`). Explicit `nvenc` / `openh264` are kept.
On serverless workers NVENC can fail right after start (driver or NVENC
sessions not ready): a failure that may be transient (ffmpeg has
`h264_nvenc` but could not open it) is retried once after 2 s, and if it
still fails the fallback is logged at WARN. The `info` job reports the
resolved encoders and the startup probe (`encoders`,
`encoder_startup_probe`).
Without OpenH264 in the build the fallback is Reactor `off` (VP8 only) and
the streams CPU-test encoder.

**Video: OpenH264 in-process** (the `openh264` crate, built from source).

- It gives Constrained Baseline with no B-frames, `IDR every 2 s`, no
  scene-cut IDRs, and forced IDRs on PLI. That is exactly what WHIP,
  Cloudflare and browsers need (streaming-refs §1.8-1.9).
- In-process encoding keeps A/V timestamps derived from one tick counter.
- The `openh264` crate also provides RGB→I420.

Alternatives considered:

- **ffmpeg libx264 subprocess** is kept behind the same `VideoEncoder` trait,
  as `encoder = "x264-ffmpeg"`. It is the encoder for RTMP/HLS sinks, which
  are ffmpeg anyway.
- **NVENC is out.**
  - The runtime image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
    without `video`.
  - strobe measured software encode at 2.64 ms/frame at 480p and rejected
    NVENC.
  - Revisit only if 768p openh264 exceeds about 15 ms/frame (gate in WP-04).
- Level: we signal `profile-level-id=42e01f` with
  `level-asymmetry-allowed=1`. 1344×768 exceeds level 3.1's frame size (risk
  R3). WHIP sinks to Cloudflare scale to fit 1280×720.

**Audio: libopus** (the `audiopus` crate), 48 kHz, 20 ms frames (960
samples), and an RTP timestamp derived from the global sample counter.

### 5.10 Backpressure summary

| Stage | Bound | Policy |
|---|---|---|
| Executor → causal pacer | 4 blocks | Executor blocks |
| Executor → clip playout | playout cap (10 clips) | Build not submitted |
| Pacer | causal: 48 frames; clip: 2 s A/V | causal: drop-oldest; clip: bounded by reservation |
| Encoder input | 10 ticks | Drop-oldest, then force IDR |
| Per-peer str0m send | str0m internal | A peer whose RTCP shows no progress for 20 s is dropped |
| Control channels | 64 queued messages per peer | Excess → close with `invalid_message` |

---

## 6. Deployment design

### 6.1 Image and binary

- `docker/gpucheck.Dockerfile` gains a `serve` stage `FROM runtime`:
  - it copies `/opt/fastvideo-rs/bin/fv-serve`, built with
    `--features cuda` in the `build` stage;
  - `EXPOSE 8000 70000/tcp 70010/udp`;
  - `ENTRYPOINT ["/opt/fastvideo-rs/bin/fv-serve"]`.
- No Python. ffmpeg is already present. libopus and OpenH264 are statically
  linked.
- CI publishes `ghcr.io/zaitrarrio/fastvideo-rs-serve:sha-<7>`. Deploy tooling
  pins the **digest**.

`fv-serve` flags and environment:

- `--config /etc/fv/serve.toml`.
- `FV_SERVE_MODE=http|runpod-queue`.
- `FV_WEIGHTS=<root>`.
- `PORT` (Runpod LB) overrides `server.port`.
- Readiness line on stdout: `FV-SERVE READY models=<ids>`, for the Vast
  PyWorker log tail.

Config sketch (native):

```toml
[server]    bind = "0.0.0.0:8000"; public_base_url = "https://…"; state_dir = "/workspace/fv-state"
[auth]      mode = "keys"            # none | keys | trust-gateway ; keys from FV_API_KEYS (sha256 hashes)
[artifacts] backend = "local"        # local | s3 ; url_ttl_s = 86400 ; signing key FV_URL_SIGNING_KEY
[[models]]  id = "fasth3"; family = "h3"; weights = "${FV_WEIGHTS}/FastH3"; recipe = "4step-vsa"; resident = true
            served_names = ["fasth3"]; continuity = "anchor-last-frame"
[aliases]   "MiniMax-H3" = "fasth3"; "MiniMax-H3-Max" = "fasth3"; "minimax/h3-max" = "fasth3"
[protocols] openai_videos = true; fastwan = false; minimax = true; fal = true; fal_director = true; ltx = false; reactor = true
[webrtc]    udp_port = 70010; tcp_port = 70000; public_ip = "auto"; ice_servers = [{urls=["stun:stun.l.google.com:19302"]}]
[limits]    queue_max = 32; body_max_mb = 64
```

Auth modes:

- `keys`: accepts `Authorization: Bearer <k>` (MiniMax, LTX, OpenAI,
  FastWan) and `Authorization: Key <k>` or `Key <id>:<secret>` (fal).
- `none`: Reactor-local parity.
- `trust-gateway`: the Runpod gateway authenticates.

`/files`, `/uploads/{token}` (token-authorized), `/health`, `/ping` and `/`
are always open.

### 6.2 Per target

| Target | Mode / entry | Ports | Weights | Health | Notes |
|---|---|---|---|---|---|
| **Runpod pod** | `http`; `scripts/serve/runpod-pod.sh` reuses `runpod-http.sh` `create_pod` (REST v1) with `dockerStartCmd` → `fv-serve` | `8000/http` (proxy, **100 s Cloudflare cap**), `70000/tcp` symmetrical (ICE-TCP) | Network volume at `/workspace` (read-only use) | `/healthz` through `https://<pod>-8000.proxy.runpod.net` | Long sync calls (`/v1/videos/sync`, LTX `/v1/*`, fal `/run`) above 100 s get 524. Document async use. Streaming: peer via ICE-TCP, WHIP for broadcast |
| **Runpod serverless (queue)** | `runpod-queue`: the Rust worker loop (§6.4) plus the in-process router | none | Network volume at `/runpod-volume` (one per DC) | Heartbeat to `RUNPOD_WEBHOOK_PING`. CUDA init failure → fail one job with the reason, then exit 1 | `executionTimeout` ≥ max stream + cold start (1 800 000 ms); `workersMin ≥ 1`; FlashBoot; digest pin; artifacts **S3** (Runpod S3 API or R2) with presigned URLs, because output is capped at 20 MB and pod-local URLs die |
| **Runpod serverless (LB)** | `http` on `$PORT`; `/ping` 204 while loading, 200 ready | `$PORT/http` | `/runpod-volume` | `/ping` | 5.5 min and 30 MB per request. `auth.mode = trust-gateway`. **With `workers.max > 1` set `FV_WORKERS_MAX` to match**: clients can't pin `X-Runpod-Worker-Id`, so fv-serve then serves only routes any worker can answer (§6.5): async job APIs need D1 jobs + R2 artifacts, sync routes need R2; cancel, uploads and streaming are never served. No streaming |
| **Vast instance** | `http`; `scripts/serve/vast.sh` follows strobe's proven REST form: offers with `direct_port_count>=1`, `runtype:"ssh_direct"`, onstart launching `fv-serve`, `env` as a **JSON object** `{"-p 8000:8000":"1","-p 70010:70010/udp":"1",…}` | 8000/tcp, 70010/udp, 70000/tcp | `hf-fm` to container disk at boot (`.part` then rename), or a local volume | `/healthz` at `http://PUBLIC_IPADDR:<ports["8000/tcp"][0].HostPort>` | Best peer-WebRTC target. Reuses `scripts/gpu/lib.sh` (`vast_check_auth`, `vast_destroy`, ledger and destroy-on-exit trap) |
| **Vast serverless** | Template onstart runs `start_server.sh` (pinned `PYWORKER_REF`, `SDK_VERSION`) with our `deploy/vast/worker.py`, plus `fv-serve` started by onstart | `WORKER_PORT` (PyWorker, TLS) → localhost:8000 | as the Vast instance | PyWorker tails the log for `FV-SERVE READY`; its benchmark handler calls `/fv/v1/capabilities` plus one tiny job | The forwarder proxies `payload` `{method,path,headers,body}` to the local router. It is batch only |

### 6.3 Warm start, shutdown, secrets

- **Warm start.**
  - `fv-serve` loads every `resident` model before reporting ready:
    `/ping` is 204 until then, `/healthz` shows `{state:"loading",loaded:[…]}`,
    and Reactor `/start_session` answers 503 + `Retry-After: 1`.
  - In `http` mode the port is bound **before** the app is built
    (`app::serve_while_building`): while the encoder probe, the stores
    (D1, R2) and the engine start, `/ping` answers 204, `/health` and
    `/healthz` 503 `loading`, and every other route 503 + `Retry-After`;
    no probe waits on startup or on the weights. (The WP-18 serverless
    LB run saw `/ping` hang: part of that is the Runpod gateway holding
    requests until a worker container runs, which the server cannot
    change.)
  - Optional `warmup = true` runs one short generation per model and
    geometry, so first-request JIT and allocation costs are paid before
    ready.
  - Text caches and kernel caches live on the volume
    (`FV_CACHE_DIR=/runpod-volume/fv-cache`).
- **Shutdown** on SIGTERM/SIGINT:
  1. Stop admission (503, or stop job-take).
  2. Cancel queued jobs (`Cancelled`; callbacks fire).
  3. Let the running generation finish within `shutdown_grace_s` (25 s), then
     trip its `CancelToken`.
  4. Streaming sessions send their farewell (`session_ended` "the server is
     shutting down", `stream_exhausted{stopped}`, WHIP DELETE).
  5. The Runpod worker posts job-done `{error}` for anything unfinished.
- **Secrets** come only from env: `FV_API_KEYS`, `FV_URL_SIGNING_KEY`,
  `FV_WEBHOOK_ED25519_KEY`, `FV_S3_*`, `FV_WHIP_TOKEN` and `HF_TOKEN`.
  - Runpod: `{{ RUNPOD_SECRET_x }}`; the Cloudflare set is mapped as
    `FV_CF_ACCOUNT_ID={{ RUNPOD_SECRET_fv_cf_account_id }}`,
    `FV_CF_API_TOKEN={{ RUNPOD_SECRET_fv_cf_api_token }}`,
    `FV_D1_DATABASE_ID={{ RUNPOD_SECRET_fv_d1_database_id }}`,
    `FV_R2_BUCKET={{ RUNPOD_SECRET_fv_r2_bucket }}`,
    `FV_R2_ENDPOINT={{ RUNPOD_SECRET_fv_r2_endpoint }}`,
    `FV_R2_ACCESS_KEY_ID={{ RUNPOD_SECRET_fv_r2_access_key_id }}`,
    `FV_R2_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_fv_r2_secret_access_key }}`
    (§0 decision 7).
  - Vast: account env vars (the same uppercase names).
  - Never passed in onstart text, and never logged. The config loader
    redacts them.

### 6.4 Runpod queue worker (`fastvideo-deploy::runpod`)

Implements deploy §1.1 against the environment templates, which are treated
as opaque:

```
task take:  loop GET {RUNPOD_WEBHOOK_GET_JOB}&job_in_progress={0|1}
            204/400 → continue; 429 → sleep 5 s; 200 {id,input} → spawn job (concurrency 1)
task ping:  every RUNPOD_PING_INTERVAL: GET {PING}?job_id=<ids>&runpod_version=fv-rs/<ver>
task stop:  long-poll GET {GET_JOB with /job-take/→/job-stop/} → cancel matching CancelToken
per job:    progress → POST {POST_OUTPUT}&isStream=false {"status":"IN_PROGRESS","output":p}
            stream   → POST {POST_STREAM}&isStream=false {"output":chunk}
            done     → POST {POST_OUTPUT}&isStream=<bool> {"output":…} | {"error":"<json string>"}, 3× Fibonacci retry
headers:    Authorization: $RUNPOD_AI_API_KEY (raw), X-Request-ID: <job id>,
            Content-Type: application/x-www-form-urlencoded (body is JSON text)
```

Job input (**native** envelope):

```jsonc
{"kind":"http","method":"POST","path":"/v2/video_generation","headers":{…},"body":{…},
 "wait":true}                // dispatched via tower::ServiceExt::oneshot into the same Router;
                             // wait=true polls the created job to terminal and returns its status body
{"kind":"stream","model":"wan-sf","prompt":"…","whip_url":"…","whip_token":"…","duration_s":600,
 "image_url":null}           // one job = one WHIP session; progress {"state":"live"} ; output = stats
```

### 6.5 Several workers behind a load balancer (`crate::multiworker`)

A Runpod load-balancer endpoint with `workers.max > 1` sends every request
to any worker, and clients cannot pin one. `server.workers_max`
(`FV_WORKERS_MAX`; `scripts/serve/runpod-endpoint.sh` sets it from the
endpoint's `workers.max`) tells fv-serve so; above 1 it serves a route only
when every worker can answer it. Each §9 route has a scope:

| Scope | Served with `workers_max > 1` when | Routes |
|---|---|---|
| local | always | health, `/metrics`, `/console` pages, catalogs (`/v1/models`, `/fv/v1/capabilities`, `/fal/schema`, JWKS), LTX v1 sync (the reply is the MP4), LTX/MiniMax 4xx stubs |
| sync | artifacts are S3/R2 | `POST /v1/videos/sync`, fal `/run/{app}/…` (the reply links the output) |
| jobs | jobs are D1 **and** artifacts are S3/R2 | async submit, status, result and list of every API: a worker reads another's job from D1 (running jobs heartbeat; progress is throttled to 1 write/s) |
| keys | `auth.key_store = d1` | `/fv/v1/admin/keys` (other workers reload within 30 s) |
| pinned | never | cancel and `DELETE` (the running job is authoritative on its worker; a cancel written to D1 elsewhere would be overwritten), `/uploads`, `/v1/upload`, fal storage initiate, `/files` (worker-local disk), fal `status/stream` (in-memory watch), `/fal/proxy` (re-enters the router unfiltered), `/fv/v1/streams*`, the fal director and Reactor (WebRTC sessions) |

A request that is not served, or that matches no route of the table (fail
closed), gets `404` with `x-fv-multi-worker: not-served` and a JSON reason;
`OPTIONS` passes. `configs/serve/runpod.toml` (D1 + R2 via `auto`) serves
local, sync and jobs routes. `server.workers_max` also applies to the in-process
router of a queue worker; queue `http` jobs should use `"wait": true` so the
worker that took the job sees it through. Tests:
`multiworker::tests` (every route classified), `tests/e2e.rs`
`multi_worker_*` (filtering; a job submitted on one worker polled to
completion on another over the D1 mock).

### 6.6 One gateway in front of per-family pools

`[engine] backend = "remote"` turns fv-serve into the gateway: one URL and
one API key for every API and model, in front of one worker pool per model
family (Runpod serverless queue endpoints or pods), with all state in D1 and
R2 so gateway replicas are interchangeable. Workers run
`server.role = "worker"` (internal token, `/fv/v1/internal/*`). Design,
configuration and the autoscaler interface: [`gateway.md`](gateway.md);
configs `configs/serve/gateway.toml` and the `[gateway] pool` of each worker
config; tests `crates/fastvideo-serve/tests/gateway.rs` and
`FV_COMPAT_GATEWAY=1 tests/compat/run.sh`.

---

## 7. Testing strategy

1. **Golden conformance (CPU, per adapter crate, `tests/golden/`).**
   - Every example body in the research docs is committed verbatim as a
     fixture:
     - fal: submit response, status bodies, cancel codes, File, the r2v
       output with `seed`, director `configure`/`prompt` examples;
     - MiniMax: the three create examples, the succeeded query, the error
       envelope;
     - LTX: the job-created, completed and failed examples, the error body,
       upload;
     - FastVideo: the `VideoResponse` fields; FastWan: the route bodies.
   - Tests call `SubmitEndpoint::normalize`, `JobView::*` and `render_error`
     and compare JSON exactly, with timestamps and ids normalized. A snapshot
     of `GenerationRequest` is pinned per example.
   - Reactor: the v1 protobuf round-trip for every oneof arm; v0 JSON
     envelopes (reactor §2.2); descriptor, signalling and `x-reactor.tracks`
     fixtures for the video-only and A/V variants.
2. **Negotiation tables.**
   - Table-driven tests over `negotiate()` for each gap in §4.7: the exact
     status and error body per API.
   - Plus grid alignment: H3 5 s → 124; LTX 6 s@24 → 145; FastWan 49..121 on
     4k+1.
3. **Fake engine.** `FakeBackend` sits behind `EngineService` with the real
   caps table shapes.
   - It emits deterministic frames: a frame counter burned into a gradient,
     so tests can check order and drops.
   - Audio is a sine at the native rate with a click every clip start.
   - Build latency is configurable (RTF), with failure and cancel injection.
   - It writes a real MP4 through ffmpeg (skipped when ffmpeg is absent).
   - The whole `fv-serve` runs in CI with `--features fake`.
4. **Media and pacer.**
   - The strobe-core pacer tests are ported.
   - A/V lockstep: after N ticks, sent audio samples equal
     `N·48000/fps` exactly, including across underruns and clip boundaries.
   - Crossfade preserves sample counts.
   - Resampler golden tests: 32k→48k, 24k→48k.
   - **Loopback bench** (strobe `loopback.py` equivalent): synthetic source →
     pacer → OpenH264 → str0m → str0m receiver. It reports
     `unique_fps`/underruns; CI asserts `underruns ≤ 10%` at 1.2× RTF
     headroom.
5. **Client-compat CI** (`tests/compat/`, a separate workflow using fake
   engine, TLS via a self-signed CA exported as `SSL_CERT_FILE`/`NODE_EXTRA_CA_CERTS`):
   - Python `fal-client` 1.0.3 (`FAL_RUN_HOST`/`FAL_QUEUE_RUN_HOST`):
     `submit`, `status(with_logs)`, `result`, `cancel` and `subscribe`.
   - JS `@fal-ai/client` 1.10.1 with `requestMiddleware`, plus the
     `proxyUrl` mode.
   - `@fal-ai/client@alpha` `fal.realtime.open` for the director, in
     headless Chromium (Playwright), because the client needs a browser
     RTCPeerConnection.
   - Python `reactor_sdk` 1.6.0 local mode: connect, `get_state`,
     `set_autoplay`, `enqueue`, receive `main_video` and `main_audio` frames
     and assert 48 kHz. A video-only variant asserts no audio track.
   - `openai==3.6.0` `client.videos.create/retrieve/download_content`.
   - Raw-HTTP scripts for LTX (the documented Python `requests` snippets
     with the host swapped), MiniMax (the documented create and query loop)
     and FastWan (the `/generate` flow from minimax-fastvideo §2.2).
6. **GPU E2E on Runpod** (`scripts/serve/e2e-runpod.sh`).
   - Bring up the serve image on a pod with the H3 / LTX-2.5 / SF-Wan configs.
   - Run the compat suites against the proxy URL.
   - For streaming, a headless Chromium Reactor/WMA client over ICE-TCP, and
     a WHIP publish to a MediaMTX on a Vast box.
   - Record `artifacts/serve/<sha>/results.json` with per-model build RTF,
     TTFF phases, unique_fps and underruns, and MP4 ffprobe facts (codec,
     profile, audio rate, faststart). Assert H3 output = 1344×768@24 with
     AAC stereo 32 kHz.
   - A queue-endpoint smoke run uses a `kind:"http"` job and a
     `kind:"stream"` job.

---

## 8. Work breakdown

Merge rules that let packages run in parallel:

1. **WP-00 is the only package that edits the root `Cargo.toml`.** It
   declares every workspace member and shared dependency version. Later
   packages add crate-local dependencies with explicit versions in *their
   own* `Cargo.toml`.
2. `Cargo.lock` conflicts are resolved by regenerating the lockfile. This is
   the only file shared across packages.
3. A package owns **only** the paths listed for it. It may read anything.
4. Engine packages that touch the same pipeline file are serialized as the
   arrows show.

### Phase 0: scaffold (1 agent, blocking)

| WP | Owns | Acceptance |
|---|---|---|
| **WP-00 scaffold** | `Cargo.toml` (members + `[workspace.dependencies]`: axum 0.8, tokio, tower-http, reqwest (rustls), str0m, openh264, audiopus, rubato, prost, prost-build, protox, uuid, time, url, bytes, async-trait, hmac, sha2, ed25519-dalek, tracing, metrics, metrics-exporter-prometheus); `crates/fastvideo-{protocol,engine-service,media,webrtc,serve-kit,openai-videos,minimax,ltxapi,fal,reactor,deploy,serve}/{Cargo.toml,src/lib.rs or main.rs}` stubs | `cargo check --workspace` (no cuda) passes; `cargo check -p fastvideo-engine-service --features cuda` compiles against stubs; clippy clean |

### Phase 1: foundations (parallel)

| WP | Owns | Depends | Acceptance |
|---|---|---|---|
| **WP-01 protocol** | `crates/fastvideo-protocol/**` | 00 | Every §3 type and trait; `negotiate()` with the §4.7 gap table; `FrameGrid`/canvas helpers call `fastvideo-models` (`resolve_canvas_size`, `H3Geometry`); unit tests for grids and negotiation; no tokio runtime dependency |
| **WP-02 engine-service core** | `crates/fastvideo-engine-service/src/{lib,service,executor,scheduler,caps,pool,cancel,fake}.rs`, `tests/` | 01 | Executor thread per backend; `Priority` scheduling; exclusive causal lease; queue positions; `CancelToken` cancels a queued job and (fake) a running one; `EngineEvent` stream; readiness; `FakeBackend` per §7.3; tests with `ManualClock`-style determinism |
| **WP-03 media** | `crates/fastvideo-media/**` | 01 | `AvPacer` port (MIT attribution header from strobe-core) plus audio lane plus first-frame notify; resampler; Opus framer; `VideoEncoder` with openh264 (CB, IDR 2 s, forced IDR) and x264-ffmpeg; `mp4::finalize` (faststart, `-an`, crop); ffprobe `MediaProbe`; crossfade; RTMP/HLS ffmpeg sink with two pipes; lockstep tests (§7.4); 768p openh264 encode ≤15 ms/frame on the CI runner, recorded |
| **WP-04 webrtc** | `crates/fastvideo-webrtc/**` | 01, 03 (types) | str0m host: UDP mux, ICE-TCP passive (RFC 4571 framing), public-address candidates, non-trickle answers with `a=end-of-candidates`, remote trickle add, data channels (accept client-created), per-mid direction (pause gate, `inactive`), pre-encoded H.264 and Opus writers, PLI events; WHIP publisher; loopback bench passes; answering a Chrome offer captured as a fixture |
| **WP-05 serve-kit** | `crates/fastvideo-serve-kit/**` | 01, 02 | `ServeCtx`; auth modes; `MemJobStore` with manifests, restart recovery and expiry sweep; `ArtifactStore` local plus S3 presign; `UrlSigner`; `UploadStore` + `PUT /uploads/{token}`; ingestion (https fetch without redirects when configured, data URI, per-protocol limits, SSRF guard, `image` decode + ffprobe); callback sender (MiniMax challenge, fal webhook Ed25519); generic `submit/status/result` handlers; SSE helper |

**WP-01 notes (as implemented).** The §3 signatures hold, with these
additions and readings; everything is re-exported from the crate root.

- Deviations: `GenerationRequest` gains `callback: Option<CallbackSpec>`
  (fal `?fal_webhook=` / MiniMax `callback_url`, copied onto `Job::callback`),
  since `normalize` is the only place that sees them. `FrameGrid` gains
  `default: u32` (the frame count for `Length::ModelDefault`; caps had no
  default length). `accepted_noop: Vec<&'static str>` is serialized but not
  deserialized.
- Readings: `CanvasCaps::short_edges[0]` is the default tier
  (`CanvasSpec::ModelDefault` = 16:9 at that tier); `max_area` applies at the
  largest tier and scales by `(short/largest)^2` below it (`area_at`), which
  gives 832×480 at 480/16:9. `boundary_ratio` is honoured exactly when
  `KnobCaps::guidance_2` is. A missing seed is drawn inside `negotiate` by
  `draw_seed()` (u32 range, JSON-safe); a sent seed is refused only if
  `knobs.seed` is false. Short edge 1080 on H3 → `Unsupported(H3Refine1080P)`,
  above 1080 → `Unsupported(H3Resolution2K)`; a length snapping to 107 frames
  on H3 → `Unsupported(H3FourSeconds)`.
- Types §3 left open: `NormalizeCtx` and `ErrorCtx` are owned (no lifetime);
  `SseSpec{initial, follow: Option<SseFollow::JobStatus{job, close_on_terminal}>, keepalive}`;
  `JobSnapshot{id, seq, state, progress, queue_position, log_count}`;
  `ListQuery` (owner/protocol/statuses/model/task/external_ids, `order`,
  cursor `after` = external id, `offset`, `limit`) with a pure
  `ListQuery::apply` any store can use; `Page{items, total, has_more}`;
  `StoreError`; `RefLimits{images, videos, audio, total}`;
  `AudioPlan::{Native{rate,channels}, Drop, Sidecar, None}`;
  `PostProcess{crop, drop_audio}`; `MediaProbe` (all fields optional).
- Extras: `precheck()` (every rule not needing staged media, to refuse before
  ingestion), `resolve_model()` (rule 1), `JobStatus` plus checked `Job::mark_*`
  transitions (`Queued→Running|Failed|Cancelled`,
  `Running→Succeeded|Failed|Cancelled`; terminal is final) and
  `Job::recover_after_restart`, `ProtocolId::default_retention`,
  `ErrorKind::http_status` (canonical/native status only; adapters keep their
  own tables), `GapId::{code, work_package, default_message}`,
  `TrackSet::for_model`, `SessionState::can_transition_to`.
- Not serde: `HttpReply`, `ViewCtx` (manual `Debug`), `RgbFrame`, `Pcm`.

**Engine packages that can start in Phase 1** (independent of the server):

| WP | Owns | Acceptance |
|---|---|---|
| **E1 cancel + progress hooks** | Observer plumbing in `crates/fastvideo-cudarc/src/{h3/pipeline.rs, ltx2/pipeline.rs, wan/pipeline.rs}` | H3 gains a step observer; all three stop within one denoise step when the observer errs, with no leaked device memory (allocator stats before and after); GPU test on one family |
| **E3 H3 geometry** | `crates/fastvideo-models/src/h3/config.rs` (+ tests) | `resolve_canvas_size_short(aw, ah, short_edge)` (480 → 832×480 at 16:9); 4 s / 107 frames admitted **only after** checking upstream FastVideo parity (`MINIMAX_H3_MIN_DURATION`); if upstream forbids it, record the decision and keep 400 |
| **E6 SF-Wan streaming rollout** | new `crates/fastvideo-cudarc/src/wan/stream.rs`, `wan/taehv.rs` (carried state) | Open-ended block API (`open/next_block/set_prompt/reset`) over `CausalKvCache` with rolling window and sink; parity: per-block decode equals whole-clip decode (TAEHV) within tolerance on an 81-frame clip; ≥10 min run without growth in device memory |
| **E8 MMAudio sidecar** | `crates/fastvideo-cudarc/src/mmaudio/**` | V2A on a frame buffer returns PCM at the model rate; audio_rtf reported; parity fixture against upstream per `docs/ports/mmaudio.md` |

### Phase 2: batch adapters, binary and CUDA backend (parallel)

| WP | Owns | Depends | Acceptance |
|---|---|---|---|
| **WP-06 openai-videos + FastWan** | `crates/fastvideo-openai-videos/**` | 05 | §4.1-4.2 routes; golden fixtures; `extra="forbid"`; multipart; `openai==3.6.0` compat script green (fake engine) |
| **WP-07 minimax** | `crates/fastvideo-minimax/**` | 05 | §4.3; 18-digit ids; list/delete semantics; callback challenge test with a local receiver; golden create/query/error fixtures |
| **WP-08 ltxapi** | `crates/fastvideo-ltxapi/**` | 05 | §4.5; the `oneOf` status shapes; 202; v1 sync bytes; `/v1/upload` + `ltx://`; 403 stubs; pad-and-crop plan; model matrix table test |
| **WP-09 fal queue** | `crates/fastvideo-fal/src/{lib,queue,schema,sync,proxy,storage,webhook}.rs`, `crates/fastvideo-fal/tests/queue*` | 05 | §4.4; both path forms plus `/response`; status invariants (`queue_position`, `logs`, `response_url`); SSE status stream; cancel codes; `x-fal-target-url` routing; storage initiate; Python and JS client compat green |
| **WP-10 fv-serve binary** | `crates/fastvideo-serve/**`, `configs/serve/*.toml` | 02, 05 | Config and env; router assembly with a route-collision test (§9 table); `/healthz`, `/ping`, `/health` (`{"status":"ok","model_loaded":bool}`), `/`, `/metrics`, `/files`; native `/fv/v1/{capabilities,streams}`; graceful shutdown; `--features fake` e2e smoke in CI |
| **WP-11 CUDA backend (batch)** | `crates/fastvideo-engine-service/src/cuda/{mod,h3,ltx2,wan,caps}.rs` | 02, E1 | `CudaBackend` loads H3 (recipes), LTX 2.3/2.5, Wan/FastWan/TI2V presets resident; caps derived from the loaded configs (audio rate from vocoder); `generate` maps `ResolvedJob` to `H3Request`/`Ltx2Request`/`GenerateConfig`; GPU smoke per family on a Runpod pod via `fv-gpucheck`-style script |

**WP-10 notes (as implemented).**

- `fastvideo-serve` is a lib plus the `fv-serve` bin. Modules: `config`
  (TOML < env; secrets only printed redacted), `gate` (serve-kit
  `EngineGate` over `EngineService`; one pump per job maps `EngineEvent` to
  `JobEvent` and calls `apply_event`; on `Finished` it runs
  `mp4::finalize` for crop/`-an`, and the artifact reports
  `ResolvedJob::output_size`), `storage`, `router` (the §9 table +
  `check_route_table`), `adapters` (feature-gated mount points), `health`,
  `metrics`, `native`, `shutdown`, `whip`.
- Job store `auto`: D1 when `FV_CF_ACCOUNT_ID`/`FV_CF_API_TOKEN`/
  `FV_D1_DATABASE_ID` are set, else `file` (MemJobStore manifests).
  Artifacts `auto`: R2/S3 when `FV_R2_*` (or `FV_S3_*`) are complete, else
  local. D1 and S3 need the `http-client` feature.
- **`D1JobStore`** lives in serve-kit (`fastvideo_serve_kit::d1`, as §0.7
  says): D1 `/query` client with retry/backoff (transport, 429, 5xx and D1's
  transient errors; SQL errors never), migrations (`schema_migrations`;
  table `jobs` with the full `Job` JSON plus `id, protocol, external_id
  UNIQUE(protocol, external_id), owner, status, model, resolved_model,
  task, progress, created_at/updated_at/completed_at/expires_at (unix ms),
  worker, version`; indexes `(owner, protocol, created_at)`, `(status,
  created_at)`, `(protocol, created_at)`, `(expires_at)`, `(worker,
  status)`), write-through inserts, immediate state-change writes,
  progress/log writes coalesced to ≤ 1/s/job, an authoritative in-memory
  cache (and `watch`) for this worker's jobs, a 60 s heartbeat, restart
  recovery of this worker's jobs, and `sweep_expired` failing other
  workers' jobs with no heartbeat for 15 min. Jobs owned by another worker
  are updated with a `version` check; the owning worker's next write wins
  (a cross-worker DELETE cannot trip another worker's cancel token).
  Tested against a SQLite mock of the D1 HTTP API (`d1-mock` feature) and
  once live against `fv-jobs`.
- Native `/fv/v1/jobs` (submit/list/get/content/delete) is the batch path
  the binary's own e2e tests drive. The job object carries `protocol` and
  `metrics` (`inference_s`, `stage_durations`, `peak_memory_mb`,
  `build_rtf` from the engine's `Finished` event; `queue_s`, `run_s` from
  the job timestamps). The list shows native jobs by default (ids resolve
  per API, and `/fv/v1/jobs/{id}` takes native ids);
  `?protocol=all` or `?protocol=<api>` lists the caller's jobs of every
  API or one API, in the native shape. `/fv/v1/streams` answers 501 until the
  streaming packages land. `runpod-queue` mode and `engine.backend = cuda`
  are mount points that fail at startup until WP-16 / WP-11 land.
- Adapters are features of `fastvideo-serve` (`openai-videos`, `minimax`,
  `ltxapi`, `fal`, `reactor`), all but `reactor` on by default:
  FastVideo `/v1/videos` + models (`[protocols] openai_videos`) and FastWan
  (`fastwan`; `/` then names the FastWan model), MiniMax
  (`MiniMax::router`, callback renderer registered), LTX
  (`router(LtxConfig)` from `[ltx]`), fal queue/sync
  (`router(ctx, FalConfig)` from `fal_apps`; `FalWebhook` renderer; a
  `WebhookSigner` from `FV_WEBHOOK_ED25519_KEY`, else per process; fal
  artifacts named by `fastvideo_fal::output_file_name`). The fal director
  (WP-14) is mounted with `fal` + `webrtc` (`src/director.rs`, `[director]`
  config); it and the Reactor runtime (WP-13) answer offers on one shared
  WebRTC host (`src/rtc.rs`, from `[webrtc]`).
  One engine glue (`gate`) serves every adapter.
- `ArtifactStore::open` (serve-kit) reads an artifact back (local path or
  S3 object bytes) so LTX `/v1` sync works on R2; the `PUT /uploads` route
  uses the `ServeCtx` clock.
- WHIP geometry: `whip::whip_h264` takes the encoder frame from
  `fastvideo-media` (`H264Config::for_publish`: Cloudflare = padded
  1280x720) and only the level/box from `fastvideo-webrtc`'s
  `EncodeProfile`, whose `output_size` (1260x720 for H3) is the picture
  inside that frame; `check_profile` asserts they agree. No crate change is
  required; renaming `output_size` to `picture_size` in fastvideo-webrtc
  would make the distinction explicit.
- `[webrtc] udp_port = 70010 / tcp_port = 70000` (§6.1) exceed the 65535
  port range; the config keeps them as integers for the streaming packages
  to settle.

**WP-11 notes (as implemented).**

- Tier table (`cuda/caps.rs`, built without `cuda` so it is CPU-tested):
  `h3-max` = Sol-H3 4-step on `h3/sol_h3_4step_engine_ladder` (`sol-h3`,
  §0.5; owner decision); `h3-turbo` = FastH3 4-step VSA + profile
  `h3/fasth3_4step_vsa` (`fasth3-4step-vsa`, also `fasth3`); `h3-draft` = the
  same at 480p with TAEH3; untiered `fasth3-8step-dense` = FastH3 8-step DMD,
  dense attention, official VAE. `ltx-pro` = LTX-2.5 distilled
  two-stage, dense stage 2 (`ltx2/ltx25_distill_dense`); `ltx-turbo` = the Sol
  stage 2 (`ltx2/ltx25_distill_sol`); `ltx-draft` = + NVFP4 video FFN
  (`ltx2/ltx25_distill_sol_nvfp4`) + TAEHV. `wan-max` = Wan2.2 TI2V-5B, 50
  UniPC steps, CFG 5 (T2V + I2V); `wan-turbo` = FastWan2.1 1.3B DMD 3-step
  VSA, full Wan VAE; `wan-draft` = the same with TAEHV; untiered
  `sfwan21-1.3b` = causal SF-Wan (E6 `CausalRollout`).
- Technique profiles and `FASTVIDEO_VSA` are process-wide and read once:
  `ProcessPlan` refuses a model set whose load-time settings differ (e.g.
  `fasth3-8step-dense` + `h3-turbo`, `ltx-turbo` + `ltx-draft`, `wan-turbo` +
  `wan-max`); such models run in separate processes (they do not co-reside
  anyway, R18).
- Deployment (`[[models]]`, one GPU per process): each entry resolves against
  the catalog (`recipe` = tier alias, catalog id or H3 recipe name; `weights`
  replaces the catalog directory; `resident` models load before readiness).
  `fv-serve` refuses at startup a set with no resident model, an unknown
  recipe or a process-settings conflict; a load fails with a named error when
  the weight directory is missing or the DiT (on-disk bytes) exceeds the free
  device memory. One H3 DiT is ~41 GB, so `h3-max` and `h3-turbo` do not
  co-reside on a 96 GB card: one per process, or `resident = false` +
  `[engine] swap = true`. Shipped configs: `configs/serve/runpod.toml`
  (h3-turbo), `runpod-h3-max.toml`, `runpod-ltx.toml`, `runpod-wan.toml`.
- Output uses E2's `FrameSink`: frames and PCM arrive in memory and go to
  an NVENC MP4 (`fastvideo-media`) or to `ClipOutput::{frames, audio}`;
  LTX skips the audio decode when the output drops audio (E4) and serves
  `ltx_fps_caps()`; Wan accepts 16 and 24 fps as container rates. SF-Wan
  causal sessions delegate to WP-15's `CausalDriver`; the SF-Wan pipeline
  stays resident for the process (the rollout borrows it). `H3Pipeline`'s
  boxed text encoder gained a `Send` bound (the pipeline lives on the
  executor thread).
- GPU check: `fv-gpucheck engine` (one model through `EngineService`, frames
  compared with the CLI clip, a second job cancelled mid-run) and the
  `serve-engine` family of `runpod-matrix.sh`.
- Pod results through the fal queue (`scripts/serve/fal-queue-smoke.sh`,
  `configs/serve/runpod.toml`, RTX PRO 6000, NVENC post encoder):
  h3-turbo T2V 1344x768 124 frames + 32 kHz AAC in 32.8 s wall (19.1 s
  denoise); I2V 110-154 s wall (21.8 s denoise). The I2V gap is the FL2VA
  text stage: `encode_request_multimodal` streams the Qwen-VL encoder (vision
  tower + LM) from the weight volume on every request and ignores the
  resident text encoder; T2V uses the resident one. A resident multimodal
  path (mRoPE + deepstack on the resident encoder) is the follow-up.
- The H3 profiles ask for MXFP8, which cuBLASLt runs only on sm_100+. On
  Hopper every H3 job failed (`h3-max`: "no MXFP8 ... algorithm on sm90";
  `h3-turbo`: the same, masked by the zero-padded retry refusing layers with a
  bf16 section). `FASTVIDEO_H3_QUANT=mxfp8` now runs W8A8 below sm_100 (logged
  once), and the padded retry pads the bf16 input too. Model load on the
  Runpod network volume: h3-turbo ~6 min, h3-max (Sol-H3) ~19 min.

### Phase 3: streaming (parallel after Phase 2 core)

| WP | Owns | Depends | Acceptance |
|---|---|---|---|
| **E2 in-memory sinks** | `wan/writer.rs`, `h3/drain.rs`, and the sink seam in the three pipeline files | E1 → E2 | Pipelines deliver RGB frames and PCM through a `FrameSink` without PNG writes; batch mp4 unchanged (byte-identical frames) |
| **WP-12 ClipSession** | `crates/fastvideo-engine-service/src/stream/{mod,clip,queue,rules}.rs` | 02, 03, E2 | fast-h3 queue semantics (caps, reservation, discard on pop, autoplay, `valid_commands`); lockstep slicing; `Continuity` modes (anchor writes last-frame PNG to the session dir); tests on the fake engine with a scripted command log that matches the fast-h3 message sequence |
| **WP-13 reactor** | `crates/fastvideo-reactor/**` (incl. `proto/` + LICENSE/NOTICE) | 04, 05, 12 | §5.7 complete; v0/v1 sniff; watchdog; pause gate; schema; `reactor_sdk` 1.6.0 compat green for A/V and video-only fake models |
| **WP-14 fal director** | `crates/fastvideo-fal/src/director/**`, `crates/fastvideo-fal/tests/director*` | 04, 09, 12 | §5.6 complete; strict schemas; version rules; heartbeat expiry; headless-browser `fal.realtime.open` compat green; A/V frames received at 24 fps / 48 kHz |
| **WP-15 CausalSession** | `crates/fastvideo-engine-service/src/stream/causal.rs`, `src/cuda/causal.rs` | 12 (shared `stream/mod.rs` owned by WP-12), E6 | Block loop under an exclusive lease; prompt switch at the block boundary; adaptive pacer; Reactor causal command set works; native WHIP stream to MediaMTX plays in a WHEP viewer; TTFF phases reported |
| **E4 LTX fps + silent** | `ltx2/pipeline.rs` (after E2) | E2 → E4 | `skip_audio_decode` flag; 25/48/50 fps validated at 1080p (frame count and MP4 rate), recorded in `benchmark.json`-style results; caps updated |
| **E7 CUDA Graphs for the SF-Wan block loop** | `wan/stream.rs`, new `wan/graph.rs` | E6 → E7 | Graph captured per block position; bitwise-equal output to eager; block time reduced (target ≤ 350 ms on H100 at 832×480; record the actual) |

**WP-12 / WP-15 notes (as implemented): the streaming session API.** All in
`fastvideo_engine_service::stream` (re-exported at the crate root); Reactor
(WP-13), the fal director (WP-14) and native `/fv/v1/streams` build on it.

- Admission is `EngineService::open_clip_session` / `open_causal_session`
  (§3.6, unchanged): one stream session per executor, a session counts busy
  from open (`Starting`) to close → `Conflict` + `Retry-After: 5`; model not
  resident / loading → `Loading` + `Retry-After: 1` (503); fps must divide
  48000 when the tracks carry audio.
- **Clip sessions.** `ClipSession::into_player(ClipPlayerConfig)` →
  `(ClipPlayer, ClipOutputs{events, media})`, a tokio task:
  - `player.command(ClipCommand) -> Result<Option<ClipEvent>, ApiError>`:
    `Some` is the correlated reply, `None` a bodyless ack. Refusals are
    broadcast as `ClipEvent::CommandError{command,reason}` (fast-h3 reasons
    verbatim) and reply `None`. Broadcasts caused by a command are on
    `events` before the reply returns, in fast-h3's order.
  - `ClipCommand` deviates from the §5.5 sketch: clip ids are the wire
    strings (blank/malformed refused inside), `SetCanvas(String)` takes the
    aspect label, and `Chunk{prompt_version, prompt, first_image, end_image,
    seconds}` gains `first_image`. Serde form `{"type","data"}` in
    snake_case.
  - `ClipEvent` serializes as the fast-h3 message (`type_name()`,
    `data()`); `ClipGenerated` also carries a `BuildReport{build_s, clip_s,
    rtf()}` (not serialized). Two native extras (`is_fasth3() == false`):
    `BuildStarted{clip, prompt_version}` (director `prompt_applied`) and
    `Starved{after, pending}` (autoplay ran dry with work pending: director
    `deadline_missed`).
  - `player.set_audience(bool)`: builds run only with an audience; losing it
    cuts a playing clip quietly (`Gone`). `state()` / `watch_state()` give
    the `state_update` snapshot (`ClipState`, fast-h3 fields and
    `valid_commands`). `close()` cancels the build, drops queues, frees the
    executor.
  - Media: `MediaItem::{Slice{clip_id, first_frame, frames, audio}, ClipEnd{
    outcome, armed}, Clear}`; slices are 3 frames plus exactly their 48 kHz
    samples at the track's channel count, emitted on a re-anchoring
    metronome. `Continuity`: `HardCut`; `Crossfade{ms}` fades clip edges at
    play time (never the first clip's head); `AnchorLastFrame` writes the
    last frame of each build to `<session_dir>/anchor-<clip>.png` and uses
    it as `Keyframe{First}` for the next build without its own first frame
    (refused at `into_player` for models without I2V).
- **Causal sessions.** `CausalSession::control()` → `CausalControl`
  (cloneable): `set_prompt` (next block boundary), `set_paused`, `set_seed`
  (applies at the next reset), `reset`, `apply(CausalCommand) ->
  CausalReply::{StateUpdate(CausalState), CommandError}` (the §5.7 causal
  set plus `get_state`), `stats()`, `ttff() -> Ttff{load_ms,
  first_block_ms, transport_ms, total_ms}` (`transport` ends at
  `mark_first_frame_sent()`), `close()`.
- **Pacers** (`stream::pace`). `spawn_clip_pacer(media, ClipPacerConfig)`
  (AvPacer, 2 s shallow cap, `IdlePolicy::{Hold, Black}` for Reactor's
  flush-to-black) and `spawn_causal_pacer(session, CausalPacerConfig)`
  (FramePacer 48, adaptive 4..fps, reports `unique_fps` into the control)
  both yield `PacedStream{ticks: TickReceiver, first_frame, stats}`: one
  `Tick{video: VideoOut, audio (exactly 48000/fps samples or None),
  video_rtp, fps}` per frame into a 10-deep drop-oldest queue
  (`take_dropped()` → force an IDR). Ticks start at the first frame by
  default (`TickStart`). `max_seconds` ends the pacer in video time.
- **CUDA causal path** (`cuda::causal`, feature `cuda`): `CausalDriver`
  runs one `wan::stream::CausalRollout` per session over a resident SF-Wan
  pipeline (reset with the block's seed, prompt change encoded at the block
  boundary with the KV cache kept, engine cancel bridged to the pipeline
  hooks, RGB8 frames to the sink); WP-11's `CudaBackend` delegates its
  `causal_*` calls to it. `CausalCudaBackend` serves SF-Wan alone and is what
  `engine.backend = cuda` builds when `FV_SFWAN_WEIGHTS` is set (until
  WP-11 lands; also `FV_SFWAN_MODEL`, `FV_SFWAN_PRESET`, `FV_CUDA_DEVICE`;
  TAEHV via `FASTVIDEO_TAE_DIR`).
- **Native streams** (`fastvideo-serve::streams`, features `webrtc` +
  `http-client`, `encoders` for OpenH264/Opus): `POST /fv/v1/streams
  {model, whip_url, whip_token?, whip_user?, whip_target?, prompt?,
  clips?[{prompt,seconds?,seed?}], width?, height?, fps?, seed?,
  max_seconds?, audio?, continuity?, autoplay?}` → 201; busy 429 +
  `Retry-After`, not resident 503 + `Retry-After`. `GET /fv/v1/streams`,
  `GET|DELETE /fv/v1/streams/{id}` (status, WHIP resource, pacer stats,
  TTFF, recent session events), `POST /fv/v1/streams/{id}/commands` (a
  `ClipCommand` or `CausalCommand` as `{type,data}`). The publisher waits
  for the first frame, offers H.264 first (no audio m-line for video-only
  models), encodes once (`[webrtc] encoder`, default `auto`: NVENC when
  the startup probe encodes, else OpenH264, else the CPU-test x264;
  `FV_STREAM_ENCODER` overrides), Opus stereo, answers PLI/FIR, tick drops
  and send errors with a keyframe within 1 s (`KeyframePolicy`: the
  periodic IDR when due, else a forced one, 1/s), and sends the WHIP `DELETE` on stop or `max_seconds`.
  `FV_STREAM_STUN` sets the srflx probe (`none` for loopback).
  `tests/streams_whip.rs` decodes what an in-process WHIP endpoint receives;
  `scripts/serve/whip-e2e.sh` adds MediaMTX and a WHEP viewer (CPU run
  2026-09-27, MediaMTX v1.15.1: fake causal stream published over WHIP,
  read back through WHEP, 48 H.264 access units received, 33 decoded with
  their burned-in frame index).
- **GPU run (2026-09-27, `scripts/serve/runpod-sfwan-whip.sh`, L40S,
  driver 580.159, serve image built with `webrtc`):** fv-serve with
  `CausalCudaBackend` → `POST /fv/v1/streams` → WHIP (NVENC, Constrained
  Baseline 832×480, level 4.0) → MediaMTX on the same pod → RTSP reader.
  TTFF load 673 ms (prompt encode + cache), first block 939 ms, transport
  2063 ms (WHIP offer + ICE/DTLS + first NVENC AU), total 3.68 s from
  `POST`. Steady generation on L40S ~6.2 unique frames/s (block ~1.9 s;
  H100 does 19.2), so the adaptive pacer settled at 5.7-6.3 fps with 2
  underruns in 37 s of video; the 30 s RTSP recording holds 188 decodable
  frames. A `set_prompt` at block 10 switched the scene at the next block
  (autumn river → snowy dawn), with the KV cache kept. Device memory 26.0
  GiB. Artifacts: `artifacts/serve/sfwan-whip/09272059/`. That run forced
  19 IDRs in 37 s (`forced_idrs`; `pacer_dropped` 0 and no send errors):
  1 at start plus MediaMTX's PLI every 2 s, each an ffmpeg restart. Fixed
  by `KeyframePolicy` (above): on CPU (`whip-e2e.sh`, OpenH264, 16 fps)
  the same MediaMTX PLIs went from 5 forced IDRs in 9 s to 0 (5
  `keyframe_requests`, all `keyframe_requests_covered`). Open: the encoder
  GOP and CBR budget use the nominal fps (16), so at L40S's ~6 fps the
  periodic IDR comes every ~5.3 s (some PLIs still force one) and the
  stream runs below its target bitrate (the RTSP recording averaged
  ~1.4 Mb/s against 2.5 Mb/s).
- Owned files: `stream/{mod,clip,queue,rules,causal,player,pace}.rs`,
  `src/cuda/causal.rs`, `tests/stream_{clip,causal}.rs`,
  `fastvideo-serve/src/streams.rs`, `fastvideo-serve/tests/streams_whip.rs`,
  `scripts/serve/whip-e2e.sh`.

**WP-13 notes (as implemented): the Reactor local runtime.**
`fastvideo-reactor` (feature `reactor` of fv-serve, on by default; built in
`App::build` by `fastvideo_serve::reactor` from `[webrtc]` and `[reactor]`).

- **Wire.** `proto/` is the vendored `reactor_wire.v1` (with RT's LICENSE
  and NOTICE); the prost bindings are committed (`src/pb/`), and the
  `proto-codegen` feature regenerates them with protox + prost-build and
  checks they match. `wire` maps v1 protobuf and v0 JSON onto one
  vocabulary; the first inbound frame latches the version, and outbound
  messages wait for the latch (at most `latch_grace`, 2 s, then the
  `Reactor-WebRTC-Version` seed, always v0) so a v1 SDK never sees a v0
  greeting.
- **Lifecycle** exactly as reactor §3.2-3.4 (fixed session id, CORS `*`,
  503 + `Retry-After: 1` while loading, 409 when not READY, orphan timeout,
  `session_ended{reason}` / `moderation` before CLOSING, RT's drain reason on
  shutdown, `/events` journal). Admission refusals from the engine map to
  409 (busy) or 503 (not resident).
- **Signalling** as §5.7: ids 1002..9999, 202 → 200-once answers,
  candidates buffered before the offer (≤256 per connection, ≤128
  connections), 64 peers, re-offers (PUT) always admitted, 30 s
  negotiation deadline (host), non-trickle answers, no RXMT. A failed
  negotiation answers `GET sdp_params` with 400 and the reason (RT would
  poll 202 until the deadline). `port_range` is accepted and ignored (one
  shared mux socket).
- **Gateway**: 20 s watchdog on any inbound message (polled every 2 s);
  pause gate (every send m-line starts paused; `ResumeTrack` opens it and
  forces a keyframe of the current picture, black before the first frame:
  the start-of-connection black frame); `RequestClip`/`RequestRecording` →
  `clip_failed{"recording disabled"}`; `PublishTrack` → `publish_refused`;
  commands are validated against the mode's table (`invalid_command`, v1)
  and run **in arrival order per connection**; acks and command errors are
  v1 only.
- **Modes** on the WP-12/WP-15 API: clip mode is `ClipSession::into_player`
  (fast-h3 events broadcast verbatim, audience = at least one connected
  peer) with `spawn_clip_pacer` (idle policy `Black`, ticks from the start);
  causal mode is `CausalControl` with `spawn_causal_pacer` (setters ack
  bodyless and broadcast `state_update`; `get_state` replies with it;
  generation pauses without an audience).
- **Media**: pacer ticks are encoded once per negotiated codec and fanned
  out; repeated ticks are not re-sent unless the held picture changed (the
  flush to black). Audio is RT's feeder: a ≤200 ms buffer drained by exactly
  one 10 ms, 480-sample, 48 kHz **mono** Opus frame every 10 ms (silence
  when short). **Deviation (VP8):** the Python `reactor_sdk` 1.6.0's
  libwebrtc offers VP8/VP9/AV1 and no H.264, so `fastvideo-webrtc` gained VP8
  answers (`AnswerOptions::video_codecs`, `PeerHandle::video_codec`) and the
  runtime sends **inter-frame VP8** to such peers: ffmpeg `libvpx`
  (`fastvideo_media::vp8`: rgb24 in, IVF out, real-time CBR at the canvas
  bitrate, no lag, one thread, a keyframe every 2 s; a forced keyframe
  restarts the process). PLI/FIR is rate-limited to one keyframe per
  second per codec (a new peer or a resumed track is served at once).
  When ffmpeg has no libvpx the runtime falls back to intra-only VP8 by
  libwebp (feature `vp8`, on by default). H.264 peers (browsers) get NVENC
  (`[reactor] h264`, default `auto`: NVENC when the startup probe encodes,
  else OpenH264). Measured with `examples/vp8_bitrate` at 1344x768, 24 fps
  (luma PSNR): moving `testsrc2`, libwebp q70 6100 kb/s at 45.1 dB vs
  libvpx 5835 kb/s at 47.2 dB (2923 kb/s at 42.6 dB); the fake engine's
  near-static frames, libwebp 1964 kb/s at 45.3 dB vs libvpx 798 kb/s at
  45.4 dB. AV1 (`av1_nvenc`) is not used: the serve image's Ubuntu 22.04
  ffmpeg (4.4) has no `av1_nvenc`, and the SDK decodes VP8 everywhere.
- **Compat** (`crates/fastvideo-reactor/tests/compat/run.sh`): Python
  `reactor_sdk` 1.6.0 local mode against `examples/fake_runtime` — A/V clip
  model (tracks `main_video` + `main_audio`, 48 kHz mono audio frames,
  get_state / set_autoplay / enqueue → clip_finished, invalid command
  raises), video-only clip model (no audio track), causal model — green.

### Phase 4: deploy, compat and E2E

| WP | Owns | Depends | Acceptance |
|---|---|---|---|
| **WP-16 deploy** | `crates/fastvideo-deploy/**`, `scripts/serve/{runpod-pod,runpod-endpoint,vast,vast-serverless}.sh`, `deploy/vast/worker.py`, `docker/gpucheck.Dockerfile` (`serve` stage only), `.github/workflows/serve-image.yml` | 10 | Rust Runpod worker against a local simulator (URL shapes from deploy §1.1): take, ping, stop, progress, done and retries; `kind:http` and `kind:stream`; env discovery for ICE candidates; scripts reuse `scripts/gpu/lib.sh`, keep the price and wall-clock caps and destroy-on-exit traps, pin digests; the Vast env dict form |
| **WP-17 client compat CI** | `tests/compat/**`, `.github/workflows/serve-compat.yml` | 06-09, 13, 14 | All §7.5 suites green against `fv-serve --features fake` |
| **WP-18 GPU E2E** | `scripts/serve/e2e/**`, `artifacts/serve/**` | 11, 13-16 | §7.6 run recorded once per release on Runpod (H3, LTX-2.5, SF-Wan); Vast peer-WebRTC run; results committed |

### Later engine packages (serialize on the files shown)

| WP | Owns | Order | Acceptance |
|---|---|---|---|
| **E5 LTX-2.5 I2V** | `ltx2/{i2v_encode.rs,pipeline.rs}`, the preset entry in `crates/fastvideo-core/src/registry.rs` | E4 → E5 | First-frame I2V on the 2.5 distilled two-stage; CLIP frame-0 similarity ≥ the 2.3 baseline; caps add `I2V` |
| **E9 LTX last-frame keyframes** | `ltx2/pipeline.rs` | E5 → E9 | `last_frame_uri` interpolation; `Keyframes` in caps |
| **E10 H3 target-audio conditioning** | `h3/{pipeline.rs,audio_vae.rs}` | after E2 | Research first: upstream FL2VA target-audio semantics. Then `target_audio_url` and director `audio_url` support |
| **E11 H3 fl2va + ref2va co-residency** | `h3/pipeline.rs` (load options) | after E10 | Both DiTs served by one process within the memory budget, or a documented swap cost; caps advertise `Ref2V` |
| **S1 (stretch) MiniMax V1 shape** | `crates/fastvideo-minimax/src/v1.rs` | after WP-07 | Only if a real client needs H3 over V1 |

Critical path: `WP-00 → WP-01 → WP-02 → WP-05 → WP-09 → (E1 → E2) → WP-12 → WP-14 → WP-18`.

---

## 9. Route table (collision check, owned by WP-10)

| Path | Owner | Notes |
|---|---|---|
| `GET /health` | serve | Merged body `{"status":"ok","model_loaded":true,"state":"AVAILABLE"}`; 503 when not ready. Satisfies FastVideo, FastWan and Reactor |
| `GET /`, `/healthz`, `/ping`, `/metrics` | serve | |
| `GET /files/{artifact}/{name}`, `PUT /uploads/{token}` | serve-kit | Shared by LTX, fal storage and Reactor uploads |
| `/v1/videos*`, `/v1/models*`, `/v1/model_info`, `/generate`, `/status/{id}`, `/video/{id}` | openai-videos | |
| `/v2/video_generation*`, `/v2/query/*`, `/v2/h3_context_ir`, `/v2/video_regeneration` | minimax | |
| `/v1\|v2/{text-to-video,image-to-video,…}`, `/v1/upload` | ltxapi | No overlap with `/v1/videos` |
| `/{app}/…` for configured apps, `/run/{app}/…`, `/fal/proxy`, `/storage/upload/initiate`, `/.well-known/jwks.json` | fal | Apps are static prefixes |
| `POST /wma/ice`, `POST /wma/session`, `POST /wma/session/heartbeat`, `POST /{app}/director/ice`, `POST /run/{app}/director/ice`, `POST /start-session`, `GET`/`POST /info` | fal director (WP-14) | Merged into the fal router, so `/fal/proxy` maps `wma.fal.run` → `/wma/*` and `fal.run/{app}/director/ice` → `/run/{app}/director/ice`. Features `fal` + `webrtc` (on by default with `reactor`); the WebRTC host is shared with Reactor |
| `POST /start_session`, `GET /session`, `POST /stop_session`, `GET /schema`, `GET /events` (SSE) | reactor | `GET /session` versus WMA `POST /wma/session` do not collide. CORS `*` |
| `GET /sessions/{sid}/transport/webrtc/ice_servers`, `POST …/connections`, `POST\|PUT\|GET …/connections/{cid}/sdp_params`, `POST …/connections/{cid}/ice_candidates` | reactor | Mounted by `App::build` (the runtime owns the WebRTC host); feature `reactor`, on by default |
| `/fv/v1/*` | serve (native) | Includes `POST/GET /fv/v1/admin/keys`, `DELETE /fv/v1/admin/keys/{id}` (serve-kit `keys::admin_routes`, admin token; WP-20) |
| `GET /fal/schema`, `GET /fal/schema/{owner}/{alias}/{sub}` | fal | Catalog of configured apps and each endpoint's input JSON Schema (native, for the console; WP-20) |
| `GET /console`, `/console/admin`, `/console/models/{owner}/{alias}/{task}`, `/console/assets/{file}` | serve (console) | Embedded static pages, [`console.md`](console.md); off with `FV_CONSOLE=0` |
| `GET /fv/v1/gateway/pools` | serve (gateway) | Per-pool metrics for the autoscaler (admin token); gateway mode only ([`gateway.md`](gateway.md) §7) |
| `POST /fv/v1/internal/jobs`, `GET`/`DELETE /fv/v1/internal/jobs/{id}`, `GET /fv/v1/internal/status` | serve (worker role) | Gateway → worker dispatch, cancel and probes; internal token only ([`gateway.md`](gateway.md) §3) |

**CORS.** One layer outside every route (`app::cors_layer`) answers
preflights (`OPTIONS` with `Access-Control-Request-Method`) for any path,
mirrors the requested method and headers (so `Authorization`, which a `*`
allow-list never covers, and `Content-Type` pass), and exposes the `X-*`
metric headers. Origins come from `server.cors_origins` /
`FV_CORS_ORIGINS`: `*` by default, as fal's own endpoints (a page on
another origin can `POST /storage/upload/initiate` and `PUT
/uploads/{token}`), a list of exact origins, or `none`. Credentials are
never allowed; the APIs authenticate with headers.

---

## 10. Risks and open questions

| # | Risk / question | Mitigation / owner |
|---|---|---|
| R1 | Outbound UDP egress on Runpod **pods** is unconfirmed; serverless has strong evidence (deploy §7 #1) | WP-18 STUN probe on a pod before relying on WHIP there. Fallback: ICE-TCP to MediaMTX |
| R2 | str0m has no TURN client. Clients behind UDP-hostile networks need client-side TURN; the server offers ICE-TCP | Document. Revisit webrtc-rs only if E2E shows real failures |
| R3 | H.264 CB level 3.1 (`42e01f`) cannot carry 1344×768 by the book. Cloudflare Stream requires L3.1 | `level-asymmetry-allowed=1` and E2E in Chrome/Safari; WHIP sinks to Cloudflare scale to 1280×720 |
| R4 | fal director WebRTC codec, frame size, `session_info` order and `wma.network-info` reply are undocumented (fal §8.6, §14 #5) | Our choices (H.264, native canvas, session_info first). Validate once against the JS alpha client in WP-14 |
| R5 | fal `sync_mode` output shape; result GET before completion; 1080P canvases (fal §14 #1-4) | INFERRED choices in §4.4, covered by golden tests and changeable in one place |
| R6 | FastWan Video API server unidentified; `.env.example` unread (minimax-fastvideo "Open items") | We implement exactly the client's observed contract |
| R7 | MiniMax: callback verification method, retries and signature; 480P/21:9 canvases; `usage` token fields | V1-style challenge (INFERRED); canvas helper E3; tokens omitted |
| R8 | No safety checker or moderation: `enable_safety_checker`, `x-reactor-moderate`, and MiniMax 1026 never trigger | `SafetyFilter` hook in serve-kit (no-op default) so a moderation service can be added |
| R9 | fal webhooks cannot carry fal signatures | Our own Ed25519 plus JWKS; receivers must be configured |
| R10 | H3-Max, LTX "pro" and `ltx-2-3-pro` are served by substitutes (quality differs from hosted) | Config aliases are explicit; `/fv/v1/capabilities` states the mapping |
| R11 | H3 streaming needs build RTF ≤ 1.0 for gapless play. Our FastH3 is ~16 s per 5 s clip on B200 (streaming-refs §3.3) | Honest `deadline_missed` and holds; director queues ahead; default 480p after E3; sequence parallelism is out of scope |
| R12 | SF-Wan quality over long rolling-KV horizons is unmeasured; LTX-style collapse possible (streaming-refs §7 #2) | `max_seconds` default 600 s; measure in WP-18 before raising |
| R13 | TAEHV per-block state parity (streaming-refs §7 #1) | E6 acceptance test; latent-overlap fallback |
| R14 | Runpod LB: whether the gateway forwards `Authorization`; WS scaling behaviour (deploy §7 #9) | `trust-gateway` mode; no streaming on LB |
| R15 | Runpod serverless SIGTERM grace is undocumented | 25 s grace, then cancel and report |
| R16 | Reactor v0 clients (older JS) versus v1 SDKs | Both codecs behind the sniff; golden tests for each |
| R17 | OpenH264 source build has no Cisco patent coverage (the binary license applies only to Cisco's prebuilt) | Legal check before a public deployment; the x264-ffmpeg backend is the switch |
| R18 | One resident model family per GPU: H3 (35B) and LTX (22B) do not co-reside | Deploy per family; the engine returns 400 "model not served" rather than swapping, unless `swap = true` is configured |
| Q1 | Should `AnchorLastFrame` be the Reactor default too (continuity versus fast-h3 parity)? | Product decision; config-only either way |
| Q2 | Should MiniMax 7-day retention hold on serverless, where the store is per-worker? | Needs a shared store (S3 manifests); deferred |
| Q3 | Should LTX v1 sync survive after 2026-10-26, when upstream retires it? | Keep; LTX-Desktop still calls it |
| Q4 | Vast `/route` host, and UDP identity mapping above 70000 (deploy §7 #6-7) | Verify in WP-16 |
| Q5 | MMAudio sliding-window V2A for causal streams (latency and quality) | After E8; not scheduled |
| Q6 | Does anyone need the FastVideo `WS /v1/stream` contract? | Non-goal until asked |
| Q7 | MiniMax `usage` token fields: omit or zero? | Omit (fields are billing-only) |
