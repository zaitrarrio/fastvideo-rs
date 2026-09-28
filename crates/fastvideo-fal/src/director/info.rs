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
//! - `resolutions` lists what the model's canvas tiers serve (`768p`, plus
//!   `480p` once E3 adds the tier, and `1080p` when the model serves the
//!   opt-in H3 1080P tier);
//! - `prompt_expander: "none"`, `accelerations: ["none"]`.

use serde_json::{json, Value};

use super::control::Limits;
use super::messages::{CLIENT_MESSAGE_TYPES, SCRIPT_MAX_BEATS, SERVER_MESSAGE_TYPES};

/// Per-session facts that `session_info` reports.
#[derive(Clone, Debug, PartialEq)]
pub struct InfoFacts {
    /// `minimax/h3-max` → `minimax-h3-max-director`.
    pub app: String,
    pub limits: Limits,
    /// Frames per default chunk on the model grid (243 for 10 s H3).
    pub default_chunk_frames: u32,
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
    let playback = f64::from(f.default_chunk_frames.saturating_sub(1)) / f64::from(l.fps.max(1));
    json!({
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
    })
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
    use crate::director::messages::Resolution;

    #[test]
    fn our_constants() {
        let f = InfoFacts {
            app: app_name("minimax/h3-max"),
            limits: Limits::default(),
            default_chunk_frames: 243,
            max_session_seconds: None,
            audio: true,
        };
        let v = session_info(&f);
        assert_eq!(v["type"], "session_info");
        assert_eq!(v["app"], "minimax-h3-max-director");
        assert_eq!(v["fps"], 24);
        assert_eq!(v["chunk_seconds"], 10);
        assert_eq!(v["default_chunk_duration"], 10);
        assert_eq!((v["min_chunk_duration"].as_u64(), v["max_chunk_duration"].as_u64()), (Some(5), Some(15)));
        assert_eq!(v["continuation_context_frames"], 1);
        assert_eq!(v["continuation_playback_seconds"], 10.083);
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
        let v = director_info(&InfoFacts { limits: l, ..f });
        assert_eq!(v["resolutions"], json!(["480p", "768p"]));
        assert!(v.get("type").is_none());
    }
}
