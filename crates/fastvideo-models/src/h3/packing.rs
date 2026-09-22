//! The packed sequence the H3 DiT runs over, as pure host math (FastVideo
//! `pipelines/basic/minimax_h3/packing.py:151-318`, identical to diffusers
//! `before_denoise.py`).
//!
//! H3 has one transformer stack for every modality. What tells a row which
//! modality it is lives entirely in the layout built here:
//!
//! ```text
//! [ text | optional keyframe / Ref2VA image refs (video tag) | audio L/R | video ]
//! ```
//!
//! T2AV has no keyframes; FL2VA inserts one latent-frame of condition rows per
//! `first` / `last` anchor between text and audio (`packing.py:210-281`).
//! Image-only Ref2VA inserts one latent frame per reference image on the same
//! segment (`build_ref2va_packed_sequence` with image refs only — contiguous).
//! Video/audio references interleave and need index gathers (not yet on this
//! layout).
//!
//! * `token_tags` pick the AdaLN modality (0 video, 1 text, 2 audio);
//! * `position_ids` are **float** `(t, h, w)` rotary coordinates on one shared
//!   clock: a unit of `t` is 1/40 s, which is one audio latent, so audio latent
//!   `a` and the video frame shown at `a / 40` s rotate alike. Text token `i`
//!   sits at `t = i`, and target audio/video start at `t = N` even when
//!   keyframe rows sit in the sequence between text and audio;
//! * text rows take the **video** timestep; keyframe rows take
//!   `max(video_t, KEYFRAME_NOISE_AUG)`.
//!
//! Positions are built in float64 exactly as numpy / torch do and cast to
//! float32 where the reference casts them (first line of its rope).

use super::config::{H3Geometry, H3TransformerConfig, H3_AUDIO_CHANNELS, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use super::reference::{PreparedImageRef, PreparedReference, RefSegment};
use super::schedule::H3RowTimesteps;

/// `MINIMAX_H3_ROPE_FRAME_RESCALE`: rotary time units per pixel frame
/// (24 fps on a 40 Hz clock).
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Pixel frames covered by each latent of a 5-latent chunk: the first latent
/// of a 17-frame clip holds one frame, the rest four.
const FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
/// `_ROPE_SPATIAL_SCALE`: a square canvas spans `[0, 32)` on both axes.
const SPATIAL_SCALE: f64 = 32.0;

/// FastVideo `MINIMAX_H3_KEYFRAME_NOISE_AUG`: keyframe latents are noised to
/// this floor and read that AdaLN timestep during denoise.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;

/// FL2VA keyframe placement on the target rotary clock (`packing.py:241-248`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeAnchor {
    /// First latent frame of the target clip (`t = N`).
    First,
    /// Last pixel-frame tick of the target clip.
    Last,
}

impl KeyframeAnchor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Last => "last",
        }
    }
}

/// A contiguous run of packed rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRange {
    pub start: usize,
    pub len: usize,
}

impl RowRange {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct H3PackedLayout {
    pub text: RowRange,
    /// Contiguous condition video rows (FL2VA / image-only Ref2VA); empty when
    /// [`Self::ref_segments`] is non-empty (interleaved Ref2VA).
    pub cond: RowRange,
    pub audio: RowRange,
    pub video: RowRange,
    /// Audio latents per stereo channel for the **target** (`audio.len / 2`).
    pub audio_latents: usize,
    /// Video token grid `(t, h, w)`; rows are frame-major, then row-major.
    pub token_grid: (usize, usize, usize),
    /// Ordered FL2VA anchors that produced [`Self::cond`] (empty = T2AV / Ref2VA).
    pub keyframe_anchors: Vec<KeyframeAnchor>,
    /// Interleaved Ref2VA condition segments between text and target audio.
    pub ref_segments: Vec<RefSegment>,
    pub num_condition_video_rows: usize,
    pub num_condition_audio_rows: usize,
    /// `(t, h, w)` per row, float64 as the reference builds them.
    pub position_ids: Vec<[f64; 3]>,
    /// 0 video, 1 text, 2 audio.
    pub token_tags: Vec<u8>,
}

/// `np.linspace(left, left + ratio, n, endpoint=False) * 32` with
/// `ratio = dim / sqrt(area)` and `left = (1 - ratio) / 2`, in numpy's own
/// evaluation order (`k * step + start`).
pub fn spatial_position_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = dim / patch;
    let step = ((left + ratio) - left) / n as f64;
    (0..n).map(|k| (k as f64 * step + left) * SPATIAL_SCALE).collect()
}

/// `origin + [0, cumsum(spans)[:-1]]` with spans `5/3 * (1, 4, 4, 4, 4)` repeating.
pub fn temporal_position_grid(latent_frames: usize, origin: f64) -> Vec<f64> {
    let mut acc = 0.0f64;
    (0..latent_frames)
        .map(|f| {
            let at = origin + acc;
            acc += FRAME_RESCALE * FRAMES_PER_LATENT[f % FRAMES_PER_LATENT.len()];
            at
        })
        .collect()
}

/// Sum of frame spans over `latent_frames` (FastVideo `_temporal_position_span`).
pub fn temporal_position_span(latent_frames: usize) -> f64 {
    (0..latent_frames)
        .map(|f| FRAME_RESCALE * FRAMES_PER_LATENT[f % FRAMES_PER_LATENT.len()])
        .sum()
}

impl H3PackedLayout {
    /// Override the text-span AdaLN tags (FL2VA / Ref2VA Qwen presentation may
    /// mark vision pads as [`TAG_VIDEO`]). `tags.len()` must equal `self.text.len`.
    pub fn set_text_token_tags(&mut self, tags: &[u8]) -> Result<(), String> {
        if tags.len() != self.text.len {
            return Err(format!(
                "text token tags: {} values for {} text rows",
                tags.len(),
                self.text.len
            ));
        }
        for (i, &t) in tags.iter().enumerate() {
            if t != TAG_TEXT && t != TAG_VIDEO {
                return Err(format!("text token tags may only be text or video, got {t} at {i}"));
            }
            self.token_tags[self.text.start + i] = t;
        }
        Ok(())
    }

    /// T2AV: no keyframe anchors.
    pub fn new(
        text_tokens: usize,
        latent: (usize, usize, usize),
        audio_latents: usize,
        patch: [usize; 3],
    ) -> Result<Self, String> {
        Self::with_keyframes(text_tokens, latent, audio_latents, patch, &[])
    }

    /// `build_packed_sequence` with optional FL2VA `first` / `last` anchors.
    pub fn with_keyframes(
        text_tokens: usize,
        latent: (usize, usize, usize),
        audio_latents: usize,
        patch: [usize; 3],
        anchors: &[KeyframeAnchor],
    ) -> Result<Self, String> {
        let (lt, lh, lw) = latent;
        let [pt, ph, pw] = patch;
        if text_tokens == 0 {
            return Err("an H3 request needs at least one text row".into());
        }
        if pt == 0 || ph == 0 || pw == 0 || lt == 0 || lt % pt != 0 || lh % ph != 0 || lw % pw != 0 {
            return Err(format!("latents {lt}x{lh}x{lw} are not divisible by the patch {patch:?}"));
        }
        for (i, a) in anchors.iter().enumerate() {
            if anchors[..i].contains(a) {
                return Err(format!("duplicate keyframe anchor {}", a.as_str()));
            }
        }
        let grid = (lt / pt, lh / ph, lw / pw);
        let rows_per_frame = grid.1 * grid.2;
        let text = RowRange {
            start: 0,
            len: text_tokens,
        };
        let cond = RowRange {
            start: text.end(),
            len: anchors.len() * rows_per_frame,
        };
        let audio = RowRange {
            start: cond.end(),
            len: audio_latents * H3_AUDIO_CHANNELS,
        };
        let video = RowRange {
            start: audio.end(),
            len: grid.0 * rows_per_frame,
        };
        let origin = text_tokens as f64;

        let sqrt_area = ((lh * lw) as f64).sqrt();
        let hgrid = spatial_position_grid(lh, ph, sqrt_area);
        let wgrid = spatial_position_grid(lw, pw, sqrt_area);
        let tgrid = temporal_position_grid(grid.0, origin);
        let last_t = origin + temporal_position_span(grid.0) - FRAME_RESCALE;

        let mut position_ids = Vec::with_capacity(video.end());
        position_ids.extend((0..text_tokens).map(|i| [i as f64, 0.0, 0.0]));
        for &anchor in anchors {
            let t = match anchor {
                KeyframeAnchor::First => origin,
                KeyframeAnchor::Last => last_t,
            };
            for h in &hgrid {
                for w in &wgrid {
                    position_ids.push([t, *h, *w]);
                }
            }
        }
        // Left channel at the first width coordinate, right at the last: the
        // stereo image is placed on the picture's horizontal axis.
        let edges = [wgrid[0], wgrid[wgrid.len() - 1]];
        for edge in edges.iter().take(H3_AUDIO_CHANNELS) {
            position_ids.extend((0..audio_latents).map(|a| [origin + a as f64, 0.0, *edge]));
        }
        for t in &tgrid {
            for h in &hgrid {
                position_ids.extend(wgrid.iter().map(|w| [*t, *h, *w]));
            }
        }
        let mut token_tags = vec![TAG_TEXT; text.len];
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, cond.len));
        token_tags.extend(std::iter::repeat_n(TAG_AUDIO, audio.len));
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, video.len));
        Ok(Self {
            text,
            cond,
            audio,
            video,
            audio_latents,
            token_grid: grid,
            keyframe_anchors: anchors.to_vec(),
            ref_segments: Vec::new(),
            num_condition_video_rows: cond.len,
            num_condition_audio_rows: 0,
            position_ids,
            token_tags,
        })
    }

    /// Image-only Ref2VA: `[text | ref images… | target audio | target video]`.
    /// Each image is one latent frame on its own spatial canvas; rotary `t`
    /// advances by 1 per image (`build_ref2va_packed_sequence`).
    pub fn with_image_references(
        text_tokens: usize,
        latent: (usize, usize, usize),
        audio_latents: usize,
        patch: [usize; 3],
        images: &[PreparedImageRef],
    ) -> Result<Self, String> {
        let (lt, lh, lw) = latent;
        let [pt, ph, pw] = patch;
        if text_tokens == 0 {
            return Err("an H3 request needs at least one text row".into());
        }
        if images.is_empty() {
            return Err("Ref2VA image packing needs at least one prepared image".into());
        }
        if pt != 1 || ph == 0 || pw == 0 || lt == 0 || lt % pt != 0 || lh % ph != 0 || lw % pw != 0 {
            return Err(format!("latents {lt}x{lh}x{lw} are not divisible by the patch {patch:?}"));
        }
        let mut cond_len = 0usize;
        let mut image_rows = Vec::with_capacity(images.len());
        for img in images {
            let rows = img.rows_per_frame(patch)?;
            image_rows.push(rows);
            cond_len += rows;
        }
        let grid = (lt / pt, lh / ph, lw / pw);
        let rows_per_frame = grid.1 * grid.2;
        let text = RowRange {
            start: 0,
            len: text_tokens,
        };
        let cond = RowRange {
            start: text.end(),
            len: cond_len,
        };
        let audio = RowRange {
            start: cond.end(),
            len: audio_latents * H3_AUDIO_CHANNELS,
        };
        let video = RowRange {
            start: audio.end(),
            len: grid.0 * rows_per_frame,
        };

        let mut position_ids = Vec::with_capacity(video.end());
        position_ids.extend((0..text_tokens).map(|i| [i as f64, 0.0, 0.0]));
        let mut rotary_time = text_tokens as f64;
        for (img, &n_rows) in images.iter().zip(&image_rows) {
            let sqrt_area = ((img.latent_height * img.latent_width) as f64).sqrt();
            let hgrid = spatial_position_grid(img.latent_height, ph, sqrt_area);
            let wgrid = spatial_position_grid(img.latent_width, pw, sqrt_area);
            debug_assert_eq!(hgrid.len() * wgrid.len(), n_rows);
            for h in &hgrid {
                for w in &wgrid {
                    position_ids.push([rotary_time, *h, *w]);
                }
            }
            rotary_time += 1.0;
        }

        let sqrt_area = ((lh * lw) as f64).sqrt();
        let hgrid = spatial_position_grid(lh, ph, sqrt_area);
        let wgrid = spatial_position_grid(lw, pw, sqrt_area);
        let tgrid = temporal_position_grid(grid.0, rotary_time);
        let edges = [wgrid[0], wgrid[wgrid.len() - 1]];
        for edge in edges.iter().take(H3_AUDIO_CHANNELS) {
            position_ids.extend((0..audio_latents).map(|a| [rotary_time + a as f64, 0.0, *edge]));
        }
        for t in &tgrid {
            for h in &hgrid {
                position_ids.extend(wgrid.iter().map(|w| [*t, *h, *w]));
            }
        }
        let mut token_tags = vec![TAG_TEXT; text.len];
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, cond.len));
        token_tags.extend(std::iter::repeat_n(TAG_AUDIO, audio.len));
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, video.len));
        Ok(Self {
            text,
            cond,
            audio,
            video,
            audio_latents,
            token_grid: grid,
            keyframe_anchors: Vec::new(),
            ref_segments: images
                .iter()
                .zip(&image_rows)
                .map(|(_, &n)| RefSegment::Video { rows: n })
                .collect(),
            num_condition_video_rows: cond.len,
            num_condition_audio_rows: 0,
            position_ids,
            token_tags,
        })
    }

    /// Full Ref2VA: `[text | ordered refs (possibly interleaved audio/video) | target audio | target video]`.
    pub fn with_references(
        text_tokens: usize,
        latent: (usize, usize, usize),
        audio_latents: usize,
        patch: [usize; 3],
        references: &[PreparedReference],
    ) -> Result<Self, String> {
        let (lt, lh, lw) = latent;
        let [pt, ph, pw] = patch;
        if text_tokens == 0 {
            return Err("an H3 request needs at least one text row".into());
        }
        if references.is_empty() {
            return Err("Ref2VA requires at least one prepared reference".into());
        }
        if pt != 1 || ph == 0 || pw == 0 || lt == 0 || lt % pt != 0 || lh % ph != 0 || lw % pw != 0 {
            return Err(format!("latents {lt}x{lh}x{lw} are not divisible by the patch {patch:?}"));
        }
        if audio_latents == 0 {
            return Err("Ref2VA target audio latents must be positive".into());
        }

        let mut ref_segments = Vec::new();
        let mut num_condition_video_rows = 0usize;
        let mut num_condition_audio_rows = 0usize;
        for r in references {
            if matches!(r, PreparedReference::Audio { .. }) && !r.has_audio() {
                return Err("an audio reference must carry audio latents".into());
            }
            if r.has_audio() && r.num_audio_latents() == 0 {
                return Err("an audio-bearing reference has no audio latents".into());
            }
            let vrows = r.video_rows(patch)?;
            let arows = r.audio_rows();
            match r {
                PreparedReference::Image(_) => {
                    ref_segments.push(RefSegment::Video { rows: vrows });
                    num_condition_video_rows += vrows;
                }
                PreparedReference::Audio { .. } => {
                    ref_segments.push(RefSegment::Audio { rows: arows });
                    num_condition_audio_rows += arows;
                }
                PreparedReference::Video { .. } => {
                    if arows > 0 {
                        ref_segments.push(RefSegment::Audio { rows: arows });
                        num_condition_audio_rows += arows;
                    }
                    ref_segments.push(RefSegment::Video { rows: vrows });
                    num_condition_video_rows += vrows;
                }
            }
        }

        let grid = (lt / pt, lh / ph, lw / pw);
        let rows_per_frame = grid.1 * grid.2;
        let text = RowRange {
            start: 0,
            len: text_tokens,
        };
        let cond_span = num_condition_video_rows + num_condition_audio_rows;
        // Contiguous `cond` only when every condition row is video (image-only).
        let image_only = num_condition_audio_rows == 0;
        let cond = RowRange {
            start: text.end(),
            len: if image_only { num_condition_video_rows } else { 0 },
        };
        let audio = RowRange {
            start: text.end() + cond_span,
            len: audio_latents * H3_AUDIO_CHANNELS,
        };
        let video = RowRange {
            start: audio.end(),
            len: grid.0 * rows_per_frame,
        };

        let mut position_ids = Vec::with_capacity(video.end());
        position_ids.extend((0..text_tokens).map(|i| [i as f64, 0.0, 0.0]));
        let mut rotary_time = text_tokens as f64;
        let mut token_tags = vec![TAG_TEXT; text_tokens];

        let target_sqrt = ((lh * lw) as f64).sqrt();
        let target_wgrid = spatial_position_grid(lw, pw, target_sqrt);

        for r in references {
            match r {
                PreparedReference::Image(img) => {
                    let sqrt_area = ((img.latent_height * img.latent_width) as f64).sqrt();
                    let hgrid = spatial_position_grid(img.latent_height, ph, sqrt_area);
                    let wgrid = spatial_position_grid(img.latent_width, pw, sqrt_area);
                    for h in &hgrid {
                        for w in &wgrid {
                            position_ids.push([rotary_time, *h, *w]);
                        }
                    }
                    token_tags.extend(std::iter::repeat_n(TAG_VIDEO, hgrid.len() * wgrid.len()));
                    rotary_time += 1.0;
                }
                PreparedReference::Audio { num_audio_latents: na } => {
                    let edges = [target_wgrid[0], target_wgrid[target_wgrid.len() - 1]];
                    for edge in edges.iter().take(H3_AUDIO_CHANNELS) {
                        position_ids.extend((0..*na).map(|a| [rotary_time + a as f64, 0.0, *edge]));
                    }
                    token_tags.extend(std::iter::repeat_n(TAG_AUDIO, *na * H3_AUDIO_CHANNELS));
                    rotary_time += *na as f64;
                }
                PreparedReference::Video {
                    num_latent_frames,
                    latent_height,
                    latent_width,
                    num_audio_latents: na,
                } => {
                    let sqrt_area = ((*latent_height * *latent_width) as f64).sqrt();
                    let hgrid = spatial_position_grid(*latent_height, ph, sqrt_area);
                    let wgrid = spatial_position_grid(*latent_width, pw, sqrt_area);
                    if *na > 0 {
                        let edges = [wgrid[0], wgrid[wgrid.len() - 1]];
                        for edge in edges.iter().take(H3_AUDIO_CHANNELS) {
                            position_ids.extend((0..*na).map(|a| [rotary_time + a as f64, 0.0, *edge]));
                        }
                        token_tags.extend(std::iter::repeat_n(TAG_AUDIO, *na * H3_AUDIO_CHANNELS));
                    }
                    let tgrid = temporal_position_grid(*num_latent_frames, rotary_time);
                    for t in &tgrid {
                        for h in &hgrid {
                            position_ids.extend(wgrid.iter().map(|w| [*t, *h, *w]));
                        }
                    }
                    let vrows = tgrid.len() * hgrid.len() * wgrid.len();
                    token_tags.extend(std::iter::repeat_n(TAG_VIDEO, vrows));
                    let span = temporal_position_span(*num_latent_frames);
                    rotary_time += (*na as f64).max(span);
                }
            }
        }

        let hgrid = spatial_position_grid(lh, ph, target_sqrt);
        let wgrid = target_wgrid;
        let tgrid = temporal_position_grid(grid.0, rotary_time);
        let edges = [wgrid[0], wgrid[wgrid.len() - 1]];
        for edge in edges.iter().take(H3_AUDIO_CHANNELS) {
            position_ids.extend((0..audio_latents).map(|a| [rotary_time + a as f64, 0.0, *edge]));
        }
        token_tags.extend(std::iter::repeat_n(TAG_AUDIO, audio.len));
        for t in &tgrid {
            for h in &hgrid {
                position_ids.extend(wgrid.iter().map(|w| [*t, *h, *w]));
            }
        }
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, video.len));

        Ok(Self {
            text,
            cond,
            audio,
            video,
            audio_latents,
            token_grid: grid,
            keyframe_anchors: Vec::new(),
            ref_segments,
            num_condition_video_rows,
            num_condition_audio_rows,
            position_ids,
            token_tags,
        })
    }

    pub fn from_geometry(geometry: &H3Geometry, text_tokens: usize) -> Result<Self, String> {
        let patch = H3TransformerConfig::fasth3_8step().patch_size;
        Self::new(
            text_tokens,
            (
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ),
            geometry.audio_latents,
            patch,
        )
    }

    pub fn from_geometry_with_keyframes(
        geometry: &H3Geometry,
        text_tokens: usize,
        anchors: &[KeyframeAnchor],
    ) -> Result<Self, String> {
        let patch = H3TransformerConfig::fasth3_8step().patch_size;
        Self::with_keyframes(
            text_tokens,
            (
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ),
            geometry.audio_latents,
            patch,
            anchors,
        )
    }

    pub fn from_geometry_with_image_refs(
        geometry: &H3Geometry,
        text_tokens: usize,
        images: &[PreparedImageRef],
    ) -> Result<Self, String> {
        let patch = H3TransformerConfig::fasth3_8step().patch_size;
        Self::with_image_references(
            text_tokens,
            (
                geometry.latent_frames,
                geometry.latent_height,
                geometry.latent_width,
            ),
            geometry.audio_latents,
            patch,
            images,
        )
    }

    pub fn sequence_length(&self) -> usize {
        self.video.end()
    }

    pub fn has_keyframes(&self) -> bool {
        self.cond.len > 0
    }

    /// Per-row index into the forward's sorted-unique timestep list
    /// (`build_row_timesteps`). Audio rows read the audio timestep; keyframe
    /// rows the condition timestep; text and target video the video one.
    pub fn timestep_indices(&self, timesteps: &H3RowTimesteps) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.token_tags.len());
        out.extend(std::iter::repeat_n(timesteps.video_index, self.text.len));
        out.extend(std::iter::repeat_n(timesteps.condition_index, self.cond.len));
        out.extend(std::iter::repeat_n(timesteps.audio_index, self.audio.len));
        out.extend(std::iter::repeat_n(timesteps.video_index, self.video.len));
        out
    }

    /// `[S, 6 * F]` cos and sin tables for the MM-RoPE, `cat(A, A)` with
    /// `A = [t x F | h x F | w x F]` (`MiniMaxH3RotaryPosEmbed.forward`):
    /// positions cast to float32 first, angles formed in float32.
    pub fn rope_tables(&self, inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
        rope_tables(&self.position_ids, inv_freq)
    }
}

/// See [`H3PackedLayout::rope_tables`]; standalone so an oracle's own
/// `position_ids` can be pushed through the same arithmetic.
pub fn rope_tables(position_ids: &[[f64; 3]], inv_freq: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let f = inv_freq.len();
    let width = 6 * f;
    let mut cos = vec![0f32; position_ids.len() * width];
    let mut sin = vec![0f32; position_ids.len() * width];
    for (row, pos) in position_ids.iter().enumerate() {
        for (axis, &p) in pos.iter().enumerate() {
            let p = p as f32;
            for (k, &inv) in inv_freq.iter().enumerate() {
                let angle = p * inv;
                for j in [axis * f + k, 3 * f + axis * f + k] {
                    cos[row * width + j] = angle.cos();
                    sin[row * width + j] = angle.sin();
                }
            }
        }
    }
    (cos, sin)
}

/// `patchify_video_latents`: `[C, T, H, W]` (one request, batch dropped) to
/// rows `[T/pt * H/ph * W/pw, C * pt * ph * pw]`. Row features are
/// **channel-major**: `feature[((c * pt + dt) * ph + dy) * pw + dx]`.
pub fn patchify(latents: &[f32], shape: [usize; 4], patch: [usize; 3]) -> Result<Vec<f32>, String> {
    let [c, t, h, w] = shape;
    let [pt, ph, pw] = patch;
    if latents.len() != c * t * h * w || t % pt != 0 || h % ph != 0 || w % pw != 0 {
        return Err(format!("patchify: {} values for {shape:?} with patch {patch:?}", latents.len()));
    }
    let (gt, gh, gw) = (t / pt, h / ph, w / pw);
    let width = c * pt * ph * pw;
    let mut rows = vec![0f32; gt * gh * gw * width];
    for_each_patch_element(shape, patch, |row, feature, source| rows[row * width + feature] = latents[source]);
    Ok(rows)
}

/// Inverse of [`patchify`].
pub fn unpatchify(rows: &[f32], shape: [usize; 4], patch: [usize; 3]) -> Result<Vec<f32>, String> {
    let [c, t, h, w] = shape;
    let [pt, ph, pw] = patch;
    if rows.len() != c * t * h * w || t % pt != 0 || h % ph != 0 || w % pw != 0 {
        return Err(format!("unpatchify: {} values for {shape:?} with patch {patch:?}", rows.len()));
    }
    let width = c * pt * ph * pw;
    let mut latents = vec![0f32; rows.len()];
    for_each_patch_element(shape, patch, |row, feature, source| latents[source] = rows[row * width + feature]);
    Ok(latents)
}

/// Calls `visit(row, feature, latent_index)` for every latent element.
fn for_each_patch_element(shape: [usize; 4], patch: [usize; 3], mut visit: impl FnMut(usize, usize, usize)) {
    let [c, t, h, w] = shape;
    let [pt, ph, pw] = patch;
    let (gh, gw) = (h / ph, w / pw);
    for ci in 0..c {
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    let row = ((ti / pt) * gh + hi / ph) * gw + wi / pw;
                    let feature = ((ci * pt + ti % pt) * ph + hi % ph) * pw + wi % pw;
                    visit(row, feature, ((ci * t + ti) * h + hi) * w + wi);
                }
            }
        }
    }
}

/// FastVideo `keyframe_condition_noise`: Gaussian noise shaped like one
/// keyframe latent frame, already patchified into DiT rows.
pub fn keyframe_condition_noise_rows(
    latent_height: usize,
    latent_width: usize,
    latent_channels: usize,
    patch: [usize; 3],
    seed: u64,
) -> Result<Vec<f32>, String> {
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;
    let [pt, ph, pw] = patch;
    if latent_height % ph != 0 || latent_width % pw != 0 || pt == 0 {
        return Err(format!(
            "keyframe noise: {latent_height}x{latent_width} not divisible by patch {patch:?}"
        ));
    }
    let shape = [latent_channels, 1, latent_height, latent_width];
    let n = shape.iter().product::<usize>();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise: Vec<f32> = (0..n).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
    patchify(&noise, shape, patch)
}

/// FastVideo `prepare_keyframe_image` canvas fit: cover-crop (or stretch) to
/// `(width, height)`. Returns RGB floats in `[-1, 1]`, channel-major `CHW`.
pub fn prepare_keyframe_rgb(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    width: usize,
    height: usize,
    stretch: bool,
) -> Result<Vec<f32>, String> {
    if rgb.len() != src_w * src_h * 3 || src_w == 0 || src_h == 0 || width == 0 || height == 0 {
        return Err(format!(
            "keyframe rgb: {} bytes for {src_w}x{src_h} → {width}x{height}",
            rgb.len()
        ));
    }
    if src_w == width && src_h == height {
        return Ok(rgb
            .chunks(3)
            .flat_map(|p| p.iter().map(|&c| f32::from(c) / 127.5 - 1.0))
            .collect());
    }
    // Nearest-neighbor cover-crop (stretch = force resize). Full Lanczos lives
    // with the encode path once `image` is wired in cudarc.
    let (rw, rh, left, top) = if stretch {
        (width, height, 0usize, 0usize)
    } else {
        let scale = (width as f64 / src_w as f64).max(height as f64 / src_h as f64);
        let rw = (src_w as f64 * scale).round().max(width as f64) as usize;
        let rh = (src_h as f64 * scale).round().max(height as f64) as usize;
        let left = rw.saturating_sub(width) / 2;
        let top = rh.saturating_sub(height) / 2;
        (rw, rh, left, top)
    };
    let mut out = vec![0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let sx = ((x + left) * src_w / rw).min(src_w - 1);
            let sy = ((y + top) * src_h / rh).min(src_h - 1);
            let si = (sy * src_w + sx) * 3;
            for c in 0..3 {
                out[c * height * width + y * width + x] = f32::from(rgb[si + c]) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

/// `unpack_audio_tokens`: rows `[2 Na, C]` (left channel's latents, then the
/// right's) to `[2, C, Na]`.
pub fn unpack_audio_rows(rows: &[f32], audio_latents: usize, channels: usize) -> Result<Vec<f32>, String> {
    if audio_latents == 0 || rows.len() != H3_AUDIO_CHANNELS * audio_latents * channels {
        return Err(format!("audio rows: {} values for {audio_latents} latents of {channels} channels", rows.len()));
    }
    let mut out = vec![0f32; rows.len()];
    for ch in 0..H3_AUDIO_CHANNELS {
        for a in 0..audio_latents {
            for c in 0..channels {
                out[(ch * channels + c) * audio_latents + a] = rows[(ch * audio_latents + a) * channels + c];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_canvas_grids_match_the_values_worked_out_in_the_spec() {
        let a = (48.0f64 * 84.0).sqrt();
        assert_eq!(a, 63.49803146555018);
        let h = spatial_position_grid(48, 2, a);
        let w = spatial_position_grid(84, 2, a);
        assert_eq!((h.len(), w.len()), (24, 42));
        assert!((h[0] - 3.9051368637047297).abs() < 1e-12 && (h[23] - 27.08695787493733).abs() < 1e-12);
        assert!((w[0] + 5.166010488516722).abs() < 1e-12 && (w[41] - 36.15810522715877).abs() < 1e-12);
        assert!(((h[1] - h[0]) - (w[1] - w[0])).abs() < 1e-12, "one step on both axes");
        // A square canvas spans [0, 32).
        let sq = spatial_position_grid(48, 2, 48.0);
        assert_eq!((sq[0], sq[12]), (0.0, 16.0));
    }

    #[test]
    fn latent_frames_sit_on_a_40_hz_clock_with_a_one_frame_chunk_head() {
        let t = temporal_position_grid(37, 0.0);
        let want = [0.0, 5.0 / 3.0, 25.0 / 3.0, 15.0, 65.0 / 3.0, 85.0 / 3.0, 30.0];
        for (got, want) in t.iter().zip(want) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert!((t[36] - 200.0).abs() < 1e-9, "latent 36 starts at frame 120 = 5 s = 200 ticks");
        assert_eq!(temporal_position_grid(2, 7.0)[0], 7.0);
    }

    /// A hand-worked case: 3 text tokens, 2 audio latents, a 2-frame 4x4 latent.
    #[test]
    fn rows_are_text_then_left_then_right_audio_then_video() {
        let l = H3PackedLayout::new(3, (2, 4, 4), 2, [1, 2, 2]).unwrap();
        assert_eq!(l.cond.len, 0);
        assert_eq!(
            (l.text, l.audio, l.video),
            (
                RowRange { start: 0, len: 3 },
                RowRange { start: 3, len: 4 },
                RowRange { start: 7, len: 8 }
            )
        );
        assert_eq!(l.sequence_length(), 15);
        assert_eq!(l.token_tags, vec![1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        // Square 4x4 latent: grid(4, 2, 4) = 32 * (0, 0.5) = (0, 16) on both axes.
        assert_eq!(l.position_ids[1], [1.0, 0.0, 0.0]);
        assert_eq!(l.position_ids[3], [3.0, 0.0, 0.0], "left channel, latent 0: t = N, w = wgrid[0]");
        assert_eq!(l.position_ids[4], [4.0, 0.0, 0.0]);
        assert_eq!(l.position_ids[5], [3.0, 0.0, 16.0], "right channel restarts the clock at w = wgrid[last]");
        assert_eq!(l.position_ids[6], [4.0, 0.0, 16.0]);
        assert_eq!(l.position_ids[7], [3.0, 0.0, 0.0], "video frame 0 starts at t = N");
        assert_eq!(l.position_ids[8], [3.0, 0.0, 16.0], "x varies fastest");
        assert_eq!(l.position_ids[9], [3.0, 16.0, 0.0]);
        assert_eq!(l.position_ids[11], [3.0 + 5.0 / 3.0, 0.0, 0.0], "frame 1 is one pixel frame later");
    }

    #[test]
    fn fl2va_first_inserts_cond_between_text_and_audio() {
        let l = H3PackedLayout::with_keyframes(
            3,
            (2, 4, 4),
            2,
            [1, 2, 2],
            &[KeyframeAnchor::First],
        )
        .unwrap();
        assert_eq!(l.cond, RowRange { start: 3, len: 4 });
        assert_eq!(l.audio.start, 7);
        assert_eq!(l.video.start, 11);
        assert_eq!(l.sequence_length(), 19);
        assert_eq!(&l.token_tags[3..7], &[0, 0, 0, 0]);
        // First keyframe shares t = N with video frame 0; audio/video origins
        // stay at N (not shifted by the cond segment length).
        assert_eq!(l.position_ids[3], [3.0, 0.0, 0.0]);
        assert_eq!(l.position_ids[7], [3.0, 0.0, 0.0], "audio still at t = N");
        assert_eq!(l.position_ids[11], [3.0, 0.0, 0.0], "video still at t = N");
    }

    #[test]
    fn fl2va_last_uses_end_of_clip_tick() {
        let l = H3PackedLayout::with_keyframes(
            3,
            (2, 4, 4),
            1,
            [1, 2, 2],
            &[KeyframeAnchor::Last],
        )
        .unwrap();
        let want = 3.0 + temporal_position_span(2) - FRAME_RESCALE;
        assert!((l.position_ids[3][0] - want).abs() < 1e-12);
        assert_eq!(l.position_ids[3][1], 0.0);
    }

    #[test]
    fn ref2va_images_advance_rotary_clock_then_target() {
        use super::super::reference::PreparedImageRef;
        // Two 4x4 latent images → 4 rows each; target 2x4x4 + 2 audio latents.
        let imgs = [
            PreparedImageRef {
                height: 64,
                width: 64,
                latent_height: 4,
                latent_width: 4,
            },
            PreparedImageRef {
                height: 64,
                width: 64,
                latent_height: 4,
                latent_width: 4,
            },
        ];
        let l = H3PackedLayout::with_image_references(3, (2, 4, 4), 2, [1, 2, 2], &imgs).unwrap();
        assert_eq!(l.cond, RowRange { start: 3, len: 8 });
        assert_eq!(l.audio.start, 11);
        assert_eq!(l.video.start, 15);
        // First image at t = N, second at t = N+1; target audio/video start at N+2.
        assert_eq!(l.position_ids[3][0], 3.0);
        assert_eq!(l.position_ids[7][0], 4.0);
        assert_eq!(l.position_ids[11][0], 5.0, "target audio at rotary_time after images");
        assert_eq!(l.position_ids[15][0], 5.0, "target video shares that origin");
    }

    #[test]
    fn ref2va_mixed_interleaves_audio_and_video_refs() {
        use super::super::config::{TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
        use super::super::reference::PreparedReference;
        let refs = [
            PreparedReference::Image(PreparedImageRef {
                height: 64,
                width: 32,
                latent_height: 4,
                latent_width: 2,
            }),
            PreparedReference::Video {
                num_latent_frames: 2,
                latent_height: 2,
                latent_width: 4,
                num_audio_latents: 2,
            },
            PreparedReference::Audio {
                num_audio_latents: 1,
            },
        ];
        let l = H3PackedLayout::with_references(3, (2, 4, 4), 2, [1, 2, 2], &refs).unwrap();
        // image 2 rows + video visual 4 + video audio 4 + audio ref 2
        assert_eq!(l.num_condition_video_rows, 6);
        assert_eq!(l.num_condition_audio_rows, 6);
        assert_eq!(l.cond.len, 0, "interleaved: cond range unused");
        assert_eq!(l.audio.start, 3 + 12);
        assert_eq!(l.ref_segments.len(), 4); // image, vid-audio, vid-video, audio
        assert_eq!(l.token_tags[..3], [TAG_TEXT; 3]);
        assert_eq!(&l.token_tags[3..5], &[TAG_VIDEO; 2]);
        assert_eq!(&l.token_tags[5..9], &[TAG_AUDIO; 4]);
        assert_eq!(&l.token_tags[9..13], &[TAG_VIDEO; 4]);
        assert_eq!(&l.token_tags[13..15], &[TAG_AUDIO; 2]);
    }

    #[test]
    fn the_5_second_request_has_the_sequence_length_of_the_spec() {
        let g = H3Geometry::default_16x9(5).unwrap();
        let l = H3PackedLayout::from_geometry(&g, 256).unwrap();
        assert_eq!(l.sequence_length(), 37_966);
        assert_eq!((l.audio.len, l.video.len, l.token_grid), (414, 37_296, (37, 24, 42)));
        assert_eq!(l.position_ids.len(), l.token_tags.len());
        let last = l.position_ids[l.sequence_length() - 1];
        assert!((last[0] - 456.0).abs() < 1e-9 && (last[2] - 36.15810522715877).abs() < 1e-12);
    }

    #[test]
    fn audio_rows_read_the_audio_timestep_and_everything_else_the_video_one() {
        let l = H3PackedLayout::new(2, (1, 2, 2), 1, [1, 2, 2]).unwrap();
        let ts = H3RowTimesteps::new(0.25, 0.5);
        assert_eq!(l.timestep_indices(&ts), vec![0, 0, 1, 1, 0]);
        let adaln: Vec<usize> = l.token_tags.iter().map(|&t| ts.adaln_row(t)).collect();
        assert_eq!(adaln, vec![1, 1, 5, 5, 0], "timestep_index * 3 + tag");
    }

    #[test]
    fn rope_tables_are_cat_a_a_over_t_h_w_blocks() {
        let inv = [1.0f32, 0.5];
        let (cos, sin) = rope_tables(&[[2.0, 3.0, -1.0]], &inv);
        assert_eq!(cos.len(), 12);
        let angles = [2.0f32, 1.0, 3.0, 1.5, -1.0, -0.5];
        for (j, a) in angles.iter().enumerate() {
            assert_eq!((cos[j], sin[j]), (a.cos(), a.sin()));
            assert_eq!((cos[6 + j], sin[6 + j]), (a.cos(), a.sin()));
        }
    }

    #[test]
    fn patch_features_are_channel_major() {
        // [C=2, T=1, H=2, W=4], patch (1, 2, 2): two rows of 8 features.
        let lat: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let rows = patchify(&lat, [2, 1, 2, 4], [1, 2, 2]).unwrap();
        // Row 0 = x in 0..2: channel 0 (0, 1, 4, 5) then channel 1 (8, 9, 12, 13).
        assert_eq!(&rows[..8], &[0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0]);
        assert_eq!(&rows[8..], &[2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0]);
        assert_eq!(unpatchify(&rows, [2, 1, 2, 4], [1, 2, 2]).unwrap(), lat);
        assert!(patchify(&lat, [2, 1, 2, 4], [1, 2, 3]).is_err());
    }

    #[test]
    fn audio_rows_unpack_to_channel_then_feature_then_time() {
        // 2 latents per channel, 3 features: rows L0, L1, R0, R1.
        let rows: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let out = unpack_audio_rows(&rows, 2, 3).unwrap();
        assert_eq!(out, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 6.0, 9.0, 7.0, 10.0, 8.0, 11.0]);
    }

    #[test]
    fn prepare_keyframe_identity_and_cover() {
        let rgb: Vec<u8> = (0..12).map(|v| v as u8).collect(); // 2x2
        let id = prepare_keyframe_rgb(&rgb, 2, 2, 2, 2, false).unwrap();
        assert_eq!(id.len(), 12);
        assert!((id[0] - (0.0 / 127.5 - 1.0)).abs() < 1e-5);
        let stretched = prepare_keyframe_rgb(&rgb, 2, 2, 4, 4, true).unwrap();
        assert_eq!(stretched.len(), 3 * 4 * 4);
    }

    #[test]
    fn keyframe_noise_patchifies_one_frame() {
        let rows = keyframe_condition_noise_rows(4, 4, 2, [1, 2, 2], 7).unwrap();
        // 2×2 patches × (2 channels × 1×2×2) features = 32
        assert_eq!(rows.len(), 32);
    }
}
