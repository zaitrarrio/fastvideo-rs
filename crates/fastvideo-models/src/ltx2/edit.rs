//! LTX-2 video editing: retake (regenerate a time window of a video) and
//! extend (continue a video at its end or its start). Host arithmetic shared
//! by negotiation (`fastvideo-protocol`) and the engine
//! (`fastvideo-cudarc::ltx2::v2v`).
//!
//! References:
//!
//! * **Retake**: `ltx_pipelines/retake.py` (Lightricks/LTX-2 `fd4ded7`,
//!   `RetakePipeline`): the whole source clip is VAE-encoded (video and
//!   audio), one distilled stage runs at the source size, and a
//!   `TemporalRegionMask(start_time, end_time, fps)` sets the denoise mask to
//!   1 on the tokens whose time span overlaps the window and 0 elsewhere. A
//!   modality that is not regenerated is `frozen` (mask 0 and its
//!   `Modality.sigma` 0).
//! * **Extend**: LTX-Desktop `backend/services/retake_pipeline/ltx_retake_pipeline.py`
//!   (Lightricks/LTX-Desktop `68cd86c`, the same `RetakePipeline` flow): the
//!   source latents are zero-padded by the new latent frames at the front
//!   (`start`) or the back (`end`), and the regenerated region is the new
//!   part plus a 0.5 s feather into the source (`_EXTEND_MASK_DELTA_SECONDS`,
//!   "mirrors the cloud gateway's MASK_DELTA_SECONDS"). The extension is a
//!   multiple of 8 frames, so the output stays `8k + 1`.
//!
//! Token time spans (`noise_mask_cond.py`, `patchifiers.py`):
//!
//! * video latent frame `i` covers pixel frames `[max(0, 8i − 7), 8i + 1)`
//!   (`get_pixel_coords(…, causal_fix=True)`), in seconds over the fps;
//! * audio latent frame `k` covers mel frames `[max(0, 4k − 3), max(0, 4k + 1))`
//!   at 160 samples of 16 kHz each (`AudioPatchifier._get_audio_latent_time_in_sec`).
//!
//! A token is in the window when `t_end > start && t_start < end`. Both are
//! computed in float32 as the reference's tensors are.

use super::config::round_half_even;

/// The VAE's temporal factor: valid frame counts are `8k + 1`.
pub const TIME_FACTOR: usize = 8;
/// The shortest clip the editor takes (two latent frames; LTX-Desktop
/// `_MIN_FRAMES`).
pub const MIN_FRAMES: usize = 9;
/// The extend seam feather into the kept source, in seconds
/// (`_EXTEND_MASK_DELTA_SECONDS`).
pub const EXTEND_MASK_DELTA_S: f64 = 0.5;
/// Audio latents per second (16 kHz, hop 160, downsample 4).
pub const AUDIO_LATENTS_PER_S: f64 = 25.0;
const AUDIO_DOWNSAMPLE: f32 = 4.0;
const AUDIO_HOP: f32 = 160.0;
const AUDIO_RATE: f32 = 16_000.0;
/// The spatial multiple of a one-stage canvas.
pub const SPATIAL_MULTIPLE: u32 = 32;

/// Which end of the source an extension grows from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendAt {
    /// New frames before the source (LTX `mode: "start"`).
    Start,
    /// New frames after the source (LTX `mode: "end"`).
    End,
}

/// `correct_frame_count`: the longest `8k + 1` count within `frames`
/// (never inventing frames); `None` below [`MIN_FRAMES`].
pub fn frames_8k1_floor(frames: usize) -> Option<usize> {
    if frames < MIN_FRAMES {
        return None;
    }
    Some((frames - 1) / TIME_FACTOR * TIME_FACTOR + 1)
}

/// `ExtendHandler._duration_to_extend_frames`: `round(duration · fps)`
/// (Python rounding) rounded up to a multiple of 8.
pub fn extend_frames(duration_s: f64, fps: f64) -> usize {
    let frames = round_half_even(duration_s * fps).max(0.0) as usize;
    frames.div_ceil(TIME_FACTOR) * TIME_FACTOR
}

/// `VideoLatentShape.from_pixel_shape(...).frames`: `(frames − 1) / 8 + 1`.
pub fn video_latent_frames(frames: usize) -> usize {
    frames.saturating_sub(1) / TIME_FACTOR + 1
}

/// `AudioLatentShape.from_video_pixel_shape(...).frames`:
/// `round(frames / fps · 25)`.
pub fn audio_latent_frames(frames: usize, fps: f64) -> usize {
    round_half_even(frames as f64 / fps * AUDIO_LATENTS_PER_S).max(0.0) as usize
}

/// `[start, end)` in seconds of every video latent frame (float32, as
/// `TemporalRegionMask` computes them).
pub fn video_latent_spans(latent_frames: usize, fps: f64) -> Vec<(f32, f32)> {
    let fps = fps as f32;
    let t = TIME_FACTOR as i64;
    (0..latent_frames as i64)
        .map(|i| {
            let start = (i * t + 1 - t).max(0);
            let end = ((i + 1) * t + 1 - t).max(0);
            (start as f32 / fps, end as f32 / fps)
        })
        .collect()
}

/// `[start, end)` in seconds of every audio latent frame (float32).
pub fn audio_latent_spans(tokens: usize) -> Vec<(f32, f32)> {
    let at = |k: usize| -> f32 {
        let mel = k as f32 * AUDIO_DOWNSAMPLE;
        let mel = (mel + 1.0 - AUDIO_DOWNSAMPLE).max(0.0);
        mel * AUDIO_HOP / AUDIO_RATE
    };
    (0..tokens).map(|k| (at(k), at(k + 1))).collect()
}

/// `TemporalRegionMask.apply_to`: whether each span overlaps
/// `[start_s, end_s)`.
pub fn in_region(spans: &[(f32, f32)], start_s: f64, end_s: f64) -> Vec<bool> {
    let (s, e) = (start_s as f32, end_s as f32);
    spans.iter().map(|&(t0, t1)| t1 > s && t0 < e).collect()
}

/// The runs of `false` (kept, mask 0) in `mask`, as `(start, len)`.
pub fn kept_runs(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < mask.len() {
        if mask[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < mask.len() && !mask[i] {
            i += 1;
        }
        out.push((start, i - start));
    }
    out
}

/// The regenerated window of an extension, in seconds of the extended clip
/// (`LTXRetakePipeline._run` with `extend_frames > 0`).
pub fn extend_window(source_frames: usize, extend: usize, at: ExtendAt, fps: f64) -> (f64, f64) {
    let target = source_frames + extend;
    let delta = round_half_even(EXTEND_MASK_DELTA_S * fps).max(0.0) as usize;
    match at {
        ExtendAt::Start => (0.0, target.min(extend + delta) as f64 / fps),
        ExtendAt::End => (source_frames.saturating_sub(delta) as f64 / fps, target as f64 / fps),
    }
}

/// The generation canvas of a source video: each side snapped down to a
/// multiple of 32 (never above the source, at least 32; LTX-Desktop
/// `correct_resolution`), then, when the area is over `max_pixels`, scaled
/// down with the aspect kept and snapped again.
pub fn edit_canvas(src_w: u32, src_h: u32, max_pixels: u64) -> (u32, u32) {
    let snap = |x: f64| -> u32 {
        let v = (x.floor() as u32) / SPATIAL_MULTIPLE * SPATIAL_MULTIPLE;
        v.max(SPATIAL_MULTIPLE)
    };
    let (mut w, mut h) = (snap(f64::from(src_w)), snap(f64::from(src_h)));
    if u64::from(w) * u64::from(h) > max_pixels {
        let s = (max_pixels as f64 / (f64::from(src_w) * f64::from(src_h))).sqrt();
        w = snap(f64::from(src_w) * s);
        h = snap(f64::from(src_h) * s);
        while u64::from(w) * u64::from(h) > max_pixels && (w > SPATIAL_MULTIPLE || h > SPATIAL_MULTIPLE) {
            if w >= h {
                w -= SPATIAL_MULTIPLE;
            } else {
                h -= SPATIAL_MULTIPLE;
            }
        }
    }
    (w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_follow_the_8k1_grid() {
        assert_eq!(frames_8k1_floor(8), None);
        assert_eq!(frames_8k1_floor(9), Some(9));
        assert_eq!(frames_8k1_floor(120), Some(113));
        assert_eq!(frames_8k1_floor(121), Some(121));
        assert_eq!(video_latent_frames(121), 16);
        assert_eq!(audio_latent_frames(121, 24.0), 126);
        // 5 s at 24 fps: 120 frames, already a multiple of 8.
        assert_eq!(extend_frames(5.0, 24.0), 120);
        // 2 s at 25 fps: 50 → 56.
        assert_eq!(extend_frames(2.0, 25.0), 56);
        // 20 s at 24 fps: 480.
        assert_eq!(extend_frames(20.0, 24.0), 480);
    }

    #[test]
    fn video_spans_are_causal_pixel_frames_over_fps() {
        let s = video_latent_spans(3, 24.0);
        assert_eq!(s[0], (0.0, 1.0 / 24.0));
        assert_eq!(s[1], (1.0 / 24.0, 9.0 / 24.0));
        assert_eq!(s[2], (9.0 / 24.0, 17.0 / 24.0));
    }

    #[test]
    fn audio_spans_are_causal_mel_frames() {
        let s = audio_latent_spans(3);
        assert_eq!(s[0], (0.0, 0.01));
        assert_eq!(s[1], (0.01, 0.05));
        assert_eq!(s[2], (0.05, 0.09));
    }

    #[test]
    fn a_window_marks_every_overlapping_token() {
        // 121 frames at 24 fps, retake [1, 3) s: latent frames whose spans
        // overlap. Frame i covers [(8i-7)/24, (8i+1)/24).
        let m = in_region(&video_latent_spans(16, 24.0), 1.0, 3.0);
        let idx: Vec<usize> = m.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect();
        // i=3: [17/24, 25/24) overlaps 1.0; i=9: [65/24, 73/24) = [2.708, 3.04) overlaps 3.0.
        assert_eq!(idx, (3..=9).collect::<Vec<_>>());
        assert_eq!(kept_runs(&m), vec![(0, 3), (10, 6)]);
        // Audio [1, 3): k with (4k+1)/100 > 1 and (4k-3)/100 < 3 → k 25..=75.
        let a = in_region(&audio_latent_spans(126), 1.0, 3.0);
        let first = a.iter().position(|b| *b).unwrap();
        let last = a.iter().rposition(|b| *b).unwrap();
        assert_eq!((first, last), (25, 75));
    }

    #[test]
    fn extend_windows_feather_half_a_second_into_the_source() {
        // 97 source frames at 24 fps, 48 new frames.
        assert_eq!(extend_window(97, 48, ExtendAt::End, 24.0), (85.0 / 24.0, 145.0 / 24.0));
        assert_eq!(extend_window(97, 48, ExtendAt::Start, 24.0), (0.0, 60.0 / 24.0));
        // The feather never passes the clip.
        assert_eq!(extend_window(9, 8, ExtendAt::Start, 24.0), (0.0, 17.0 / 24.0));
    }

    #[test]
    fn canvases_snap_down_and_fit_the_budget() {
        let budget = 1920 * 1088;
        assert_eq!(edit_canvas(768, 512, budget), (768, 512));
        assert_eq!(edit_canvas(1920, 1080, budget), (1920, 1056));
        assert_eq!(edit_canvas(1080, 1920, budget), (1056, 1920));
        assert_eq!(edit_canvas(854, 480, budget), (832, 480));
        let (w, h) = edit_canvas(3840, 2160, budget);
        assert!(u64::from(w * h) <= budget && w % 32 == 0 && h % 32 == 0, "{w}x{h}");
        assert_eq!((w, h), (1920, 1056));
        assert_eq!(edit_canvas(20, 20, budget), (32, 32));
    }
}
