//! `session_info` / `DirectorInfo` (fal §8.5): **our** constants, not the
//! hosted engine's. The hosted values describe fal's own backend (39
//! context frames, a prompt expander, target-audio conditioning); ours
//! describe what this server does (design §5.6):
//!
//! - `fps` is the model's rate (24 for H3), `continuation_context_frames: 1`
//!   (the last-frame anchor, trimmed from the continuation), and
//!   `continuation_playback_seconds` is a default chunk minus that frame;
//! - `audio_sample_rate: 48000` (Opus on the wire), conditioning audio would
//!   be 32 kHz but `audio_conditioning: false` until E10;
//! - `resolutions` lists what the model's canvas tiers serve among `480p`,
//!   `720p`, `768p` and `1080p` (H3: `480p`/`768p`, `1080p` with the opt-in
//!   H3 1080P tier; LTX: all four). `720p` is ours, not in fal's schema;
//!   older clients never ask for it;
//! - `prompt_expander: "none"`, `accelerations: ["none"]`;
//! - the chunk length is a session choice (`configure.chunk_duration`, ours):
//!   `default_chunk_duration` (5) and `chunk_seconds` are the default,
//!   `chunk_duration_options` the lengths this model serves (`[5, 10]`, or
//!   `[5]` on a model whose clips stop short of 10 s), per resolution in
//!   `chunk_duration_options_by_resolution` (the H3 1080P tier: `[5]`),
//!   with the frames each generates in `chunk_duration_frames` and a
//!   `chunk_duration_note` when an option is missing. Causal models have no
//!   chunk to size: those keys are absent.

use fastvideo_protocol::{Family, ModelCaps, StreamCaps};
use serde_json::{json, Value};

use super::control::{CausalLimits, Limits};
use super::engine::frames_for;
use super::messages::{Resolution, CLIENT_MESSAGE_TYPES, SCRIPT_MAX_BEATS, SERVER_MESSAGE_TYPES};

/// Whether the director can run a model: clip streaming (H3, LTX) or a
/// causal rollout (LongLive, SF-Wan).
pub fn director_capable(caps: &ModelCaps) -> bool {
    matches!(caps.stream, Some(StreamCaps::Clip { .. } | StreamCaps::Causal { .. }))
}

/// Whether a causal model is LongLive (served as `longlive`): its weights are
/// non-commercial (docs/serve/research-longlive.md §4.3).
pub fn is_longlive(caps: &ModelCaps) -> bool {
    caps.id.0.contains("longlive") || caps.served_names.iter().any(|n| n.contains("longlive"))
}

/// The licence note of a LongLive model's director.
pub const LONGLIVE_LICENCE: &str = "LongLive-1.3B weights (NVIDIA, HF card: CC-BY-NC-SA 4.0): non-commercial use only (research and evaluation) unless NVIDIA confirms otherwise; see docs/serve/research-longlive.md §4.3.";

/// Resolutions a model serves: its canvas tiers among 480p / 720p / 768p /
/// 1080p (H3: 480p and 768p, 1080p with the opt-in 1080P tier; LTX: all
/// four).
pub fn served_resolutions(caps: &ModelCaps) -> Vec<Resolution> {
    Resolution::ALL.into_iter().filter(|r| caps.canvas.short_edges.contains(&r.short_edge())).collect()
}

/// The control limits for a model: a causal model's block-built director
/// chunk, else the clip range within 5..15 s, the H3 1080P tier's cap and
/// the default chunk (`default_seconds`, the nearest served
/// `chunk_duration` option).
pub fn model_limits(caps: &ModelCaps, default_seconds: f64, causal_chunk_blocks: u32) -> Limits {
    let fps = caps.fps.default;
    if let Some(StreamCaps::Causal { block_frames, context, .. }) = caps.stream {
        let block_seconds = f64::from(block_frames) / f64::from(fps.max(1));
        let chunk_blocks = causal_chunk_blocks.max(1);
        let chunk = block_seconds * f64::from(chunk_blocks);
        return Limits {
            fps,
            chunk_seconds: chunk,
            min_chunk_seconds: chunk,
            max_chunk_seconds: chunk,
            resolutions: served_resolutions(caps),
            script_max_end_images: 0,
            causal: Some(CausalLimits { block_frames, block_seconds, chunk_blocks, context }),
            ..Limits::default()
        };
    }
    let (min_s, max_s) = match caps.stream {
        Some(StreamCaps::Clip { min_s, max_s }) => (f64::from(min_s), f64::from(max_s)),
        _ => (5.0, 15.0),
    };
    let max = max_s.min(15.0);
    let min = min_s.max(5.0).min(max);
    let l = Limits {
        fps,
        chunk_seconds: default_seconds.clamp(min, max),
        min_chunk_seconds: min,
        max_chunk_seconds: max,
        resolutions: served_resolutions(caps),
        // The H3 1080P tier's clip cap (5 s; 10 s with `h3_1080p_long`).
        hd_max_chunk_seconds: caps
            .canvas
            .hd
            .filter(|t| t.short_edge == Resolution::R1080.short_edge())
            .and_then(|t| t.max_frames)
            .map(|n| (f64::from(n) / f64::from(fps.max(1))).floor()),
        ..Limits::default()
    };
    // The default is one of the served `chunk_duration` options.
    Limits { chunk_seconds: l.chunk_for(default_seconds), ..l }
}

/// `(chunk_duration, generated frames)` for each option `limits` serve,
/// on the model's frame grid at its rate (snapped up, as the engine
/// does): H3 `17n+5` 5 s → 124, 10 s → 243; LTX `8k+1` → 121, 241; Wan
/// `4k+1` at 24 fps → 121 (5 s only), FastWan 1.3B at 16 fps → 81.
pub fn chunk_frames(caps: &ModelCaps, limits: &Limits) -> Vec<(u32, u32)> {
    limits
        .chunk_options()
        .into_iter()
        .filter_map(|d| frames_for(caps, limits.fps, f64::from(d)).map(|n| (d, n)))
        .collect()
}

/// What a chunk costs at `res` relative to 768p on this model's family,
/// for the director form's labels (`None`: no note, e.g. at 768p).
/// Measured per chunk on one GPU of the class the family is served on:
/// H3 1080P about 2.5x 768p (docs/serve/h3-1080p-and-upscaler.md); LTX
/// two-stage at the director's canvases, RTX PRO 6000 (docs/serve/e2e/ltx.md
/// "Director tiers").
pub fn relative_chunk_cost(caps: &ModelCaps, res: Resolution) -> Option<f64> {
    match (caps.family, res) {
        (Family::H3, Resolution::R1080) => Some(2.5),
        (Family::Ltx2, Resolution::R480) => Some(LTX_COST_480),
        (Family::Ltx2, Resolution::R720) => Some(LTX_COST_720),
        (Family::Ltx2, Resolution::R1080) => Some(LTX_COST_1080),
        _ => None,
    }
}

/// LTX chunk time relative to 768p (see [`relative_chunk_cost`]): ltx-turbo
/// two-stage, 121 frames, denoise + decode, measured 2026-10-01 on RTX PRO
/// 6000: 480p 5.35 s, 720p 10.16 s, 768p 10.91 s, 1080p 22.72 s.
const LTX_COST_480: f64 = 0.5;
const LTX_COST_720: f64 = 0.9;
const LTX_COST_1080: f64 = 2.1;

/// Per-session facts that `session_info` reports.
#[derive(Clone, Debug, PartialEq)]
pub struct InfoFacts {
    /// `minimax/h3-max` → `minimax-h3-max-director`.
    pub app: String,
    pub limits: Limits,
    /// Frames per default chunk on the model grid (124 for 5 s H3).
    pub default_chunk_frames: u32,
    /// `(chunk_duration, generated frames)` for each served option on the
    /// model grid (LTX: 5 s → 121, 10 s → 241).
    pub chunk_frames: Vec<(u32, u32)>,
    pub max_session_seconds: Option<u64>,
    /// The session carries audio (a video-only model answers `inactive`).
    pub audio: bool,
}

/// `app` name of the director of fal app `owner/alias`.
pub fn app_name(app_id: &str) -> String {
    format!("{}-director", app_id.replace('/', "-"))
}

/// The `DirectorInfo` object (`POST /info`); `session_info` adds `type`.
pub fn director_info(f: &InfoFacts) -> Value {
    let l = &f.limits;
    let resolutions: Vec<&str> = l.resolutions.iter().map(|r| r.as_str()).collect();
    if let Some(c) = &l.causal {
        return causal_info(f, c, &resolutions);
    }
    let playback = f64::from(f.default_chunk_frames.saturating_sub(1)) / f64::from(l.fps.max(1));
    let by_res: serde_json::Map<String, Value> = l.resolutions.iter().map(|r| (r.as_str().to_owned(), json!(l.at(*r).chunk_options()))).collect();
    let frames: serde_json::Map<String, Value> = f.chunk_frames.iter().map(|(s, n)| (s.to_string(), json!(n))).collect();
    let mut v = json!({
        "app": f.app,
        "protocol_version": 1,
        "fps": l.fps,
        "chunk_seconds": l.chunk_seconds.round() as u64,
        "default_chunk_duration": l.chunk_seconds.round() as u64,
        "min_chunk_duration": l.min_chunk_seconds.round() as u64,
        "max_chunk_duration": l.max_chunk_seconds.round() as u64,
        "continuation_context_frames": 1,
        "continuation_playback_seconds": (playback * 1000.0).round() / 1000.0,
        "audio_sample_rate": 48_000,
        "conditioning_audio_sample_rate": 32_000,
        "audio_bitrates": [96_000, 128_000, 192_000],
        "default_audio_bitrate": null,
        "aspect_ratios": ["16:9", "9:16", "1:1"],
        "resolutions": resolutions,
        "max_session_seconds": f.max_session_seconds,
        "session_limit_scope": "configured",
        "prompt_expander": "none",
        "one_session_per_machine": true,
        "prompt_deck_size": l.deck_size,
        "audio_conditioning": false,
        "audio_behaviors": ["replace", "queue"],
        "max_audio_source_seconds": 0,
        "prompt_context_segments": l.default_memory,
        "default_memory": l.default_memory,
        "min_memory": 1,
        "max_memory": 50,
        "default_acceleration": "none",
        "accelerations": ["none"],
        "scripts": true,
        "script_modes": ["replace", "append"],
        "script_max_beats": SCRIPT_MAX_BEATS,
        "script_max_end_images": l.script_max_end_images,
        "script_max_audio_beats": 0,
        "script_max_queued": l.script_max_queued,
        "script_max_pending": l.script_max_queued,
        "script_min_end_image_spacing_seconds": l.end_image_spacing_seconds.round() as u64,
        "script_min_chunk_seconds": l.min_chunk_seconds.round() as u64,
        "script_min_opening_chunk_seconds": l.min_chunk_seconds.round() as u64,
        "client_message_types": CLIENT_MESSAGE_TYPES,
        "server_message_types": SERVER_MESSAGE_TYPES,
    });
    // The session's chunk length (`configure.chunk_duration`), set apart
    // from the macro above (its recursion limit).
    v["chunk_duration_options"] = json!(l.chunk_options());
    v["chunk_duration_options_by_resolution"] = Value::Object(by_res);
    v["chunk_duration_frames"] = Value::Object(frames);
    v["chunk_duration_note"] = json!(chunk_note(l));
    v
}

/// Why a `chunk_duration` option is missing (`None`: 5 and 10 s served
/// at every resolution).
fn chunk_note(l: &Limits) -> Option<String> {
    let all = super::messages::CHUNK_DURATIONS.to_vec();
    let options = l.chunk_options();
    let mut notes = Vec::new();
    if options != all {
        notes.push(match options.as_slice() {
            [] => format!(
                "this model's clips are {:.2}..{:.2} s: chunks are {} s whatever `chunk_duration` asks",
                l.min_chunk_seconds,
                l.max_chunk_seconds,
                (l.chunk_seconds * 100.0).round() / 100.0
            ),
            o => format!(
                "this model's clips are at most {:.2} s: `chunk_duration` {} only (other values run at the nearest)",
                l.max_chunk_seconds,
                o.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join(" or ")
            ),
        });
    }
    for r in &l.resolutions {
        let o = l.at(*r).chunk_options();
        if o != options && !o.is_empty() {
            let list = o.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join(" or ");
            notes.push(format!("at {} chunks are {list} s (the tier's clip cap)", r.as_str()));
        }
    }
    (!notes.is_empty()).then(|| notes.join("; "))
}

/// `DirectorInfo` of a causal model (docs/serve/director-causal.md §4): the
/// chunk is a group of blocks played as they arrive; the continuation
/// context is the KV window, not an anchor frame; text only, 16:9 only.
fn causal_info(f: &InfoFacts, c: &super::control::CausalLimits, resolutions: &[&str]) -> Value {
    let l = &f.limits;
    let mut v = {
        let clip = InfoFacts { limits: super::control::Limits { causal: None, ..l.clone() }, ..f.clone() };
        director_info(&clip)
    };
    let chunk_frames = c.block_frames * c.chunk_blocks.max(1);
    let chunk_s = f64::from(chunk_frames) / f64::from(l.fps.max(1));
    let whole = (chunk_s.round() as u64).max(1);
    v["chunk_seconds"] = whole.into();
    v["default_chunk_duration"] = whole.into();
    v["min_chunk_duration"] = whole.into();
    v["max_chunk_duration"] = whole.into();
    v["continuation_context_frames"] = c.context.map_or(c.block_frames, |x| x.window_pixel_frames()).into();
    v["continuation_playback_seconds"] = ((chunk_s * 1000.0).round() / 1000.0).into();
    v["resolutions"] = json!(resolutions);
    v["aspect_ratios"] = json!(["16:9"]);
    v["audio_behaviors"] = json!([]);
    v["script_max_end_images"] = 0.into();
    v["script_min_end_image_spacing_seconds"] = 0.into();
    v["script_min_chunk_seconds"] = 1.into();
    v["script_min_opening_chunk_seconds"] = 1.into();
    // One continuous rollout: no chunk length to choose.
    if let Some(o) = v.as_object_mut() {
        for k in ["chunk_duration_options", "chunk_duration_options_by_resolution", "chunk_duration_frames", "chunk_duration_note"] {
            o.remove(k);
        }
    }
    v["causal"] = json!({
        "block_frames": c.block_frames,
        "block_seconds": c.block_seconds,
        "chunk_blocks": c.chunk_blocks,
        "kv_window_latent_frames": c.context.map(|x| x.window_latent_frames),
        "sink_latent_frames": c.context.map(|x| x.sink_latent_frames),
        "prompt_switch": match c.context {
            Some(x) if x.prompt_recache => "recache",
            Some(_) => "keep",
            None => "unknown",
        },
        "image_conditioning": false,
    });
    v
}

/// The `session_info` message: `DirectorInfo` plus `type`.
pub fn session_info(f: &InfoFacts) -> Value {
    let mut v = director_info(f);
    v["type"] = json!("session_info");
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_constants() {
        let f = InfoFacts {
            app: app_name("minimax/h3-max"),
            limits: Limits::default(),
            default_chunk_frames: 124,
            chunk_frames: vec![(5, 124), (10, 243)],
            max_session_seconds: None,
            audio: true,
        };
        let v = session_info(&f);
        assert_eq!(v["type"], "session_info");
        assert_eq!(v["app"], "minimax-h3-max-director");
        assert_eq!(v["fps"], 24);
        assert_eq!(v["chunk_seconds"], 5);
        assert_eq!(v["default_chunk_duration"], 5);
        assert_eq!((v["min_chunk_duration"].as_u64(), v["max_chunk_duration"].as_u64()), (Some(5), Some(15)));
        assert_eq!(v["chunk_duration_options"], json!([5, 10]));
        assert_eq!(v["chunk_duration_options_by_resolution"], json!({"768p": [5, 10]}));
        assert_eq!(v["chunk_duration_frames"], json!({"5": 124, "10": 243}));
        assert_eq!(v["chunk_duration_note"], Value::Null);
        assert_eq!(v["continuation_context_frames"], 1);
        assert_eq!(v["continuation_playback_seconds"], 5.125);
        assert_eq!(v["audio_sample_rate"], 48_000);
        assert_eq!(v["conditioning_audio_sample_rate"], 32_000);
        assert_eq!(v["resolutions"], json!(["768p"]));
        assert_eq!(v["aspect_ratios"], json!(["16:9", "9:16", "1:1"]));
        assert_eq!(v["one_session_per_machine"], true);
        assert_eq!(v["audio_conditioning"], false);
        assert_eq!(v["scripts"], true);
        assert_eq!(v["client_message_types"], json!(["configure", "ping", "prompt", "stop"]));
        assert_eq!(v["server_message_types"].as_array().unwrap().len(), 16);
        let l = Limits { resolutions: vec![Resolution::R480, Resolution::R768], ..Limits::default() };
        let v = director_info(&InfoFacts { limits: l, ..f.clone() });
        assert_eq!(v["resolutions"], json!(["480p", "768p"]));
        let l = Limits { resolutions: Resolution::ALL.to_vec(), ..Limits::default() };
        let v = director_info(&InfoFacts { limits: l, ..f.clone() });
        assert_eq!(v["resolutions"], json!(["480p", "720p", "768p", "1080p"]));
        assert!(v.get("type").is_none());
        // The H3 1080P tier caps chunks at 5 s.
        let l = Limits { resolutions: vec![Resolution::R768, Resolution::R1080], hd_max_chunk_seconds: Some(5.0), ..Limits::default() };
        let v = director_info(&InfoFacts { limits: l, ..f.clone() });
        assert_eq!(v["chunk_duration_options"], json!([5, 10]));
        assert_eq!(v["chunk_duration_options_by_resolution"], json!({"768p": [5, 10], "1080p": [5]}));
        assert!(v["chunk_duration_note"].as_str().unwrap().contains("at 1080p chunks are 5 s"), "{v}");
        // A model whose clips stop short of 10 s (Wan 2.2 5B: 161 frames).
        let l = Limits { max_chunk_seconds: 161.0 / 24.0, resolutions: vec![Resolution::R480], ..Limits::default() };
        let v = director_info(&InfoFacts { limits: l, chunk_frames: vec![(5, 121)], default_chunk_frames: 121, ..f });
        assert_eq!((v["chunk_duration_options"].clone(), v["default_chunk_duration"].as_u64()), (json!([5]), Some(5)));
        assert_eq!(v["chunk_duration_frames"], json!({"5": 121}));
        assert!(v["chunk_duration_note"].as_str().unwrap().contains("at most 6.71 s"), "{v}");
        assert_eq!(v["continuation_playback_seconds"], 5.0);
    }

    #[test]
    fn causal_constants() {
        use super::super::control::CausalLimits;
        let ctx = fastvideo_protocol::CausalContext { window_latent_frames: 12, sink_latent_frames: 3, prompt_recache: true };
        let l = Limits {
            fps: 16,
            chunk_seconds: 3.0,
            min_chunk_seconds: 3.0,
            max_chunk_seconds: 3.0,
            resolutions: vec![Resolution::R480],
            causal: Some(CausalLimits { block_frames: 12, block_seconds: 0.75, chunk_blocks: 4, context: Some(ctx) }),
            ..Limits::default()
        };
        let f = InfoFacts { app: app_name("fastvideo/longlive"), limits: l, default_chunk_frames: 48, chunk_frames: Vec::new(), max_session_seconds: Some(600), audio: false };
        let v = session_info(&f);
        assert_eq!(v["type"], "session_info");
        assert_eq!(v["app"], "fastvideo-longlive-director");
        assert_eq!(v["fps"], 16);
        assert_eq!((v["chunk_seconds"].as_u64(), v["min_chunk_duration"].as_u64(), v["max_chunk_duration"].as_u64()), (Some(3), Some(3), Some(3)));
        assert_eq!(v["continuation_context_frames"], 48);
        assert_eq!(v["continuation_playback_seconds"], 3.0);
        assert_eq!(v["resolutions"], json!(["480p"]));
        assert_eq!(v["aspect_ratios"], json!(["16:9"]));
        assert_eq!(v["script_max_end_images"], 0);
        assert_eq!(v["causal"]["prompt_switch"], "recache");
        assert_eq!(v["causal"]["kv_window_latent_frames"], 12);
        assert_eq!(v["causal"]["block_seconds"], 0.75);
        assert_eq!(v["max_session_seconds"], 600);
        // No chunk length to choose on a causal model.
        for k in ["chunk_duration_options", "chunk_duration_options_by_resolution", "chunk_duration_frames", "chunk_duration_note"] {
            assert!(v.get(k).is_none(), "{k}");
        }
        assert_eq!(v["default_chunk_duration"], 3);
    }
}
