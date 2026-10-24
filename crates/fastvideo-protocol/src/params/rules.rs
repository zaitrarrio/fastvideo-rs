//! The surface rules: every request limit and allowed set an upstream API
//! contract fixes, one constant each (plan E2, ADR-0012).
//!
//! Each constant names the vendored contract it comes from (paths under
//! `docs/serve/specs/`; `tests/param_rules.rs` checks the machine-readable
//! ones against the files). Adapters validate with these constants and
//! build their [`super::ParamSpec`]s from them; what the *model* can do
//! comes from its [`crate::ModelCaps`] instead. A limit that is a fact of
//! the model (the H3 clip range, the H3 reference counts) is derived from
//! `fastvideo_models` or the caps here, not restated.
//!
//! Some modules hold rules of our own (`ours`) where the upstream contract
//! is silent; each says so.

use fastvideo_models::h3::config as h3cfg;

use super::{IntField, IntRange, NumField, NumRange};
use crate::caps::RefLimits;

/// Facts of the H3 model every H3 surface states in its own contract.
pub mod h3 {
    use super::*;

    /// Whole-second clip lengths of FastVideo `/v1/videos` (`packing.py`,
    /// 5-15 s), fal `minimax/h3/*` (`duration` integer 5..=15,
    /// `fal/minimax/h3/*.openapi.json`) and MiniMax `MiniMax-H3-Max` (4 is
    /// rejected): the model's trained range from `fastvideo_models`.
    pub const SECONDS: IntRange = IntRange::new(
        h3cfg::H3_FASTVIDEO_MIN_DURATION_S as i64,
        h3cfg::H3_MAX_DURATION_S as i64,
    );
    /// MiniMax `duration` enum 4..=15 (`minimax/v2-video-generation.json`
    /// `VideoGenerationV2Req.duration`): the model's 4 s floor.
    pub const SECONDS_FROM_FOUR: IntRange = IntRange::new(
        h3cfg::H3_MIN_DURATION_S as i64,
        h3cfg::H3_MAX_DURATION_S as i64,
    );
    /// Reference inputs of reference-to-video: 9 images, 3 videos, 3 audio
    /// clips, 12 in total (MiniMax README; fal
    /// `minimax/h3*/reference-to-video` `maxItems` 9 / 3 / 3 and
    /// `reference_conditions` 12). The caps' [`RefLimits::h3`] is this.
    pub const REFERENCES: RefLimits = RefLimits {
        images: 9,
        videos: 3,
        audio: 3,
        total: 12,
    };
    /// The fixed aspect ratios (MiniMax `ratio` without `adaptive`; fal
    /// `aspect_ratio` of `text-to-video`), in schema order.
    pub const ASPECT_RATIOS: [&str; 6] = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
    /// The follow-the-image ratio (MiniMax `ratio`, fal r2v `aspect_ratio`).
    pub const ADAPTIVE: &str = "adaptive";
}

/// FastVideo `POST /v1/videos` (`fastvideo/videos.schema.json`, derived from
/// `protocol.py` `VideoGenerationRequest`).
pub mod openai_videos {
    use super::*;

    /// `num_inference_steps` `ge=1, le=200`.
    pub const NUM_INFERENCE_STEPS: IntRange = IntRange::new(1, 200);
    /// `guidance_scale` and `guidance_scale_2` `ge=0, le=20`.
    pub const GUIDANCE_SCALE: NumRange = NumRange::new(0.0, 20.0);
    /// `boundary_ratio` `ge=0, le=1`.
    pub const BOUNDARY_RATIO: NumRange = NumRange::new(0.0, 1.0);
    /// `n` and `num_outputs_per_prompt` `ge=1, le=10`.
    pub const OUTPUTS: IntRange = IntRange::new(1, 10);
    /// Outputs per request this server makes (ours: one).
    pub const OUTPUTS_SERVED: i64 = 1;
    /// `quality` enum.
    pub const QUALITIES: [&str; 4] = ["auto", "default", "standard", "hd"];
    /// `task` values (only defined for H3).
    pub const TASKS: [&str; 3] = ["t2va", "fl2va", "ref2va"];
    /// `fl2va` takes a first and an optional last frame.
    pub const KEYFRAME_IMAGES: IntRange = IntRange::new(1, 2);
    /// H3 `seconds` (FastVideo's 5-15 s, [`h3::SECONDS`]).
    pub const H3_SECONDS: IntRange = h3::SECONDS;
    /// `GET /v1/videos?limit`: OpenAI `limit` `maximum: 100`
    /// (`openai/videos.openapi.yaml`); this server takes 1..=100 (upstream
    /// lists `minimum: 0`), default 20.
    pub const LIST_LIMIT: IntField = IntField::new(1, 100, 20);
}

/// MiniMax `POST /v2/video_generation` (`minimax/v2-video-generation.json`).
pub mod minimax {
    use super::*;

    /// Longest `text` content item (MiniMax README, research §1.3.1).
    pub const PROMPT_MAX_CHARS: usize = 7000;
    /// `ratio` enum.
    pub const RATIOS: [&str; 7] = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
    /// `extra.prompt_expansion_mode` values.
    pub const PROMPT_EXPANSION_MODES: [&str; 3] = ["disabled", "balanced", "quality"];
    /// `duration` of `MiniMax-H3` (and our Turbo / Draft): 4..=15.
    pub const DURATION: IntRange = h3::SECONDS_FROM_FOUR;
    /// `duration` of `MiniMax-H3-Max`: 5..=15 ("4 is rejected").
    pub const DURATION_MAX_MODEL: IntRange = h3::SECONDS;
    /// Reference items per kind and in total.
    pub const REFERENCES: RefLimits = h3::REFERENCES;
    /// `GET /v2/query/video_generation?page_size`: default 20, at most 100
    /// (ours; the spec gives only the example 20).
    pub const PAGE_SIZE: IntField = IntField::new(1, 100, 20);
}

/// The LTX product's contract, shared by the LTX API (`ltx/openapi.json`
/// and its "Supported Models" matrix) and fal `lightricks/ltx-2.5/*`
/// (`fal/lightricks/ltx-2.5/*.openapi.json`).
pub mod ltx {
    use super::*;

    /// `prompt` `maxLength`.
    pub const PROMPT_MAX_CHARS: usize = 5000;
    /// `camera_motion` enum.
    pub const CAMERA_MOTIONS: [&str; 8] = [
        "dolly_in",
        "dolly_out",
        "dolly_left",
        "dolly_right",
        "jib_up",
        "jib_down",
        "static",
        "focus_shift",
    ];
    /// Frame rates of the fast models (LTX API, fal `*/fast` `fps` enum).
    pub const FPS: [u32; 4] = [24, 25, 48, 50];
    /// Frame rates of fal `*/pro` (`fps` enum).
    pub const FPS_FAL_PRO: [u32; 3] = [24, 25, 50];
    /// The rates that allow the long clips (24 and 25).
    pub const FPS_LONG: [u32; 2] = [24, 25];
    /// The duration grid: 6, 8, ... seconds.
    pub const DURATION_MIN: u32 = 6;
    pub const DURATION_STEP: u32 = 2;
    /// The longest clip of a fast model at 720p / 1080p and 24 / 25 fps.
    pub const DURATION_MAX_LONG: u32 = 20;
    /// Every other cell of the matrix.
    pub const DURATION_MAX_SHORT: u32 = 10;
    /// The short edge up to which a fast model has the long clips (1080p).
    pub const LONG_MAX_SHORT_EDGE: u32 = 1080;
    /// The LTX-2 frame grid's ceiling (8k+1 up to 481 frames: 20 s at 24
    /// fps), which fal's ingredient `num_frames` maximum also states. The
    /// engine's caps carry the grid (families/ltx2.toml); the fal tests pin
    /// this to the served caps.
    pub const FRAMES_MAX: u32 = 481;

    /// Whether the matrix cell (`fast` model, short edge, rate) allows the
    /// 20 s clips: fast models at 720p / 1080p and 24 / 25 fps.
    pub fn long_clips(fast: bool, short_edge: u32, fps: u32) -> bool {
        fast && short_edge <= LONG_MAX_SHORT_EDGE && FPS_LONG.contains(&fps)
    }
    /// The longest duration of a matrix cell.
    pub fn max_duration(fast: bool, short_edge: u32, fps: u32) -> u32 {
        if long_clips(fast, short_edge, fps) {
            DURATION_MAX_LONG
        } else {
            DURATION_MAX_SHORT
        }
    }
    /// The durations `DURATION_MIN..=max` on the 2 s grid.
    pub fn durations(max: u32) -> Vec<u32> {
        (DURATION_MIN..=max)
            .step_by(DURATION_STEP as usize)
            .collect()
    }

    /// LTX API `fps` default (`TextToVideoRequest.fps.default`).
    pub const API_DEFAULT_FPS: u32 = 24;
    /// fal `fps` default.
    pub const FAL_DEFAULT_FPS: u32 = 25;
    /// fal `duration` when omitted: `"auto"` upstream; this server runs 6 s
    /// (no duration head).
    pub const FAL_DEFAULT_DURATION: u32 = 6;

    /// LTX API audio-to-video `audio_uri` ceilings: 20 s at 720p / 1080p on
    /// the fast and 2.3 models, 10 s on `ltx-2-5-pro` and at 1440p / 4K.
    pub const A2V_MAX_S_LONG: u32 = 20;
    pub const A2V_MAX_S_SHORT: u32 = 10;
    /// LTX API retake `resolution` enum (`EditVideoRequestResolution`).
    pub const RETAKE_RESOLUTIONS: [&str; 2] = ["1920x1080", "1080x1920"];

    /// fal audio-to-video `guidance_scale` bounds (ours: 1..=50; fal's
    /// schema has no such field).
    pub const FAL_A2V_GUIDANCE: NumRange = NumRange::new(1.0, 50.0);
    /// fal audio-to-video on `pro`: "a maximum of 10 seconds".
    pub const FAL_A2V_PRO_MAX_S: u32 = A2V_MAX_S_SHORT;
    /// fal audio-to-video output: 1080p at 24 fps.
    pub const FAL_A2V_SHORT_EDGE: u32 = 1080;
    pub const FAL_A2V_FPS: u32 = 24;

    /// fal `fal-ai/ltx-2.3/retake-video` `start_time` 0..=20.
    pub const FAL_RETAKE_START: NumRange = NumRange::new(0.0, 20.0);
    /// fal retake / extend `duration` default (5 s).
    pub const FAL_EDIT_DURATION_DEFAULT: f64 = 5.0;

    /// fal `fal-ai/ltx-2.3-quality/ingredient`: `num_frames` 9..=481
    /// (default 121), `frames_per_second` 1..=60 (default 24),
    /// `ingredient_strength` / `reference_strength` 0..=2 (default 1), and
    /// the default canvas 1536x896.
    pub const INGREDIENT_FRAMES: IntField = IntField::new(9, 481, 121);
    pub const INGREDIENT_FPS: IntField = IntField::new(1, 60, 24);
    pub const INGREDIENT_STRENGTH: NumField = NumField::new(0.0, 2.0, 1.0);
    pub const INGREDIENT_CANVAS: (u32, u32) = (1536, 896);
}

/// fal `minimax/h3*` apps (`fal/minimax/**.openapi.json`) and the director
/// (`fal/fal-ai/minimax-h3-max-director/asyncapi.json`).
pub mod fal_h3 {
    use super::*;

    /// `prompt` `maxLength` (queue endpoints and director `prompt`).
    pub const PROMPT_MAX_CHARS: usize = 50_000;
    /// `duration` (integer seconds, [`h3::SECONDS`]).
    pub const DURATION: IntRange = h3::SECONDS;
    /// Reference list limits.
    pub const REFERENCES: RefLimits = h3::REFERENCES;
}

/// fal `fal-ai/wan/v2.2-5b/*` (`fal/fal-ai/wan/v2.2-5b/*.openapi.json`).
pub mod fal_wan {
    use super::*;

    pub const NUM_FRAMES: IntField = IntField::new(17, 161, 81);
    pub const FRAMES_PER_SECOND: IntField = IntField::new(4, 60, 24);
    pub const NUM_INFERENCE_STEPS: IntField = IntField::new(2, 50, 40);
    pub const GUIDANCE_SCALE: NumField = NumField::new(1.0, 10.0, 3.5);
    pub const SHIFT: NumField = NumField::new(1.0, 10.0, 5.0);
    /// `num_interpolated_frames` 0..=4 (default 0).
    pub const INTERPOLATED_FRAMES: IntField = IntField::new(0, 4, 0);
    /// `interpolator_model` enum.
    pub const INTERPOLATORS: [&str; 3] = ["none", "film", "rife"];
}

/// The director's control channel `configure` / `prompt` messages
/// (`fal/fal-ai/minimax-h3-max-director/asyncapi.json`).
pub mod fal_director {
    use super::*;

    /// `prompt` / beat prompts: 1..=50000 characters.
    pub const PROMPT_MAX_CHARS: usize = fal_h3::PROMPT_MAX_CHARS;
    /// `prompt_version` `minimum: 1`, at most JS `Number.MAX_SAFE_INTEGER`.
    pub const PROMPT_VERSION_MAX: u64 = (1 << 53) - 1;
    /// `configure.memory` 1..=50.
    pub const MEMORY: IntRange = IntRange::new(1, 50);
    /// `configure.audio_bitrate` values.
    pub const AUDIO_BITRATES: [u32; 3] = [96_000, 128_000, 192_000];
    /// `configure.protocol_version` `const: 1`.
    pub const PROTOCOL_VERSION: u64 = 1;
    /// `script` beats (ours): 1..=64.
    pub const SCRIPT_MAX_BEATS: usize = 64;
    /// `configure.style` (ours, with `upsample`).
    pub const STYLE_MAX_CHARS: usize = 2_000;
    /// `configure.entities` (ours): entries, id, name, descriptor lengths.
    pub const ENTITIES_MAX: usize = 32;
    pub const ENTITY_ID_MAX_CHARS: usize = 64;
    pub const ENTITY_NAME_MAX_CHARS: usize = 200;
    pub const ENTITY_DESCRIPTOR_MAX_CHARS: usize = 1_000;
    /// `configure.chunk_duration` (ours): 5 or 10 whole seconds, default 5.
    pub const CHUNK_DURATIONS: [u32; 2] = [5, 10];
    pub const DEFAULT_CHUNK_DURATION: u32 = 5;
}

/// The fast-h3 clip-queue contract (Reactor `fast-h3` commands, the
/// engine's stream player `enqueue`, `/v1/segments`): `prompt` at most 800
/// characters, `metadata` at most 2000, the `set_canvas` aspects
/// (`docs/serve/research-reactor.md`; no machine-readable upstream).
pub mod fast_h3 {
    pub const PROMPT_MAX_CHARS: usize = 800;
    pub const METADATA_MAX_CHARS: usize = 2000;
    pub const ASPECTS: [&str; 4] = ["16:9", "1:1", "9:16", "4:3"];
}

/// The Reactor local runtime's HTTP contract (`docs/serve/reactor-proto/`,
/// `docs/serve/research-reactor.md`; the Python `reactor_sdk` the runtime
/// mirrors): `stop_session.reason` length and the upload store.
pub mod reactor {
    /// `POST /stop_session` `reason`: at most 64 characters.
    pub const STOP_REASON_MAX_CHARS: usize = 64;
    /// Largest accepted upload (64 MiB).
    pub const UPLOAD_MAX_BYTES: u64 = 64 * 1024 * 1024;
    /// Uploads a session may hold.
    pub const UPLOADS_MAX: usize = 64;
    /// `upload_id`: 1 to 64 of `[A-Za-z0-9_-]`.
    pub const UPLOAD_ID_MAX_CHARS: usize = 64;
}
