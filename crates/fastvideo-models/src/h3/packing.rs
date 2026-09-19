//! The packed sequence the H3 DiT runs over, as pure host math (FastVideo
//! `pipelines/basic/minimax_h3/packing.py:151-318`, identical to diffusers
//! `before_denoise.py`).
//!
//! H3 has one transformer stack for every modality. What tells a row which
//! modality it is lives entirely in the layout built here:
//!
//! ```text
//! [ text rows (N) | audio rows, left then right (2 Na) | video rows (T * h * w) ]
//! ```
//!
//! * `token_tags` pick the AdaLN modality (0 video, 1 text, 2 audio);
//! * `position_ids` are **float** `(t, h, w)` rotary coordinates on one shared
//!   clock: a unit of `t` is 1/40 s, which is one audio latent, so audio latent
//!   `a` and the video frame shown at `a / 40` s rotate alike. Text token `i`
//!   sits at `t = i`, and everything else starts at `t = N`;
//! * text rows take the **video** timestep.
//!
//! With no keyframes (this checkpoint is T2AV only) the three segments are
//! contiguous ranges, which is what lets the device graph use `narrow` / `cat`
//! instead of index scatter.
//!
//! Positions are built in float64 exactly as numpy / torch do and cast to
//! float32 where the reference casts them (first line of its rope).

use super::config::{H3Geometry, H3TransformerConfig, H3_AUDIO_CHANNELS, TAG_AUDIO, TAG_TEXT, TAG_VIDEO};
use super::schedule::H3RowTimesteps;

/// `MINIMAX_H3_ROPE_FRAME_RESCALE`: rotary time units per pixel frame
/// (24 fps on a 40 Hz clock).
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Pixel frames covered by each latent of a 5-latent chunk: the first latent
/// of a 17-frame clip holds one frame, the rest four.
const FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
/// `_ROPE_SPATIAL_SCALE`: a square canvas spans `[0, 32)` on both axes.
const SPATIAL_SCALE: f64 = 32.0;

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
    pub audio: RowRange,
    pub video: RowRange,
    /// Audio latents per stereo channel (`audio.len / 2`).
    pub audio_latents: usize,
    /// Video token grid `(t, h, w)`; rows are frame-major, then row-major.
    pub token_grid: (usize, usize, usize),
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

impl H3PackedLayout {
    /// `build_packed_sequence` for a request with no keyframes or references.
    /// `latent` is the VAE latent `(T, H, W)`; `patch` the DiT patch.
    pub fn new(text_tokens: usize, latent: (usize, usize, usize), audio_latents: usize, patch: [usize; 3]) -> Result<Self, String> {
        let (lt, lh, lw) = latent;
        let [pt, ph, pw] = patch;
        if text_tokens == 0 {
            return Err("an H3 request needs at least one text row".into());
        }
        if pt == 0 || ph == 0 || pw == 0 || lt == 0 || lt % pt != 0 || lh % ph != 0 || lw % pw != 0 {
            return Err(format!("latents {lt}x{lh}x{lw} are not divisible by the patch {patch:?}"));
        }
        let grid = (lt / pt, lh / ph, lw / pw);
        let rows_per_frame = grid.1 * grid.2;
        let text = RowRange { start: 0, len: text_tokens };
        let audio = RowRange { start: text.end(), len: audio_latents * H3_AUDIO_CHANNELS };
        let video = RowRange { start: audio.end(), len: grid.0 * rows_per_frame };
        let origin = text_tokens as f64;

        let sqrt_area = ((lh * lw) as f64).sqrt();
        let hgrid = spatial_position_grid(lh, ph, sqrt_area);
        let wgrid = spatial_position_grid(lw, pw, sqrt_area);
        let tgrid = temporal_position_grid(grid.0, origin);

        let mut position_ids = Vec::with_capacity(video.end());
        position_ids.extend((0..text_tokens).map(|i| [i as f64, 0.0, 0.0]));
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
        token_tags.extend(std::iter::repeat_n(TAG_AUDIO, audio.len));
        token_tags.extend(std::iter::repeat_n(TAG_VIDEO, video.len));
        Ok(Self { text, audio, video, audio_latents, token_grid: grid, position_ids, token_tags })
    }

    pub fn from_geometry(geometry: &H3Geometry, text_tokens: usize) -> Result<Self, String> {
        let patch = H3TransformerConfig::fasth3_8step().patch_size;
        Self::new(text_tokens, (geometry.latent_frames, geometry.latent_height, geometry.latent_width), geometry.audio_latents, patch)
    }

    pub fn sequence_length(&self) -> usize {
        self.video.end()
    }

    /// Per-row index into the forward's sorted-unique timestep list
    /// (`build_row_timesteps`): audio rows read the audio timestep, text and
    /// video rows the video one.
    pub fn timestep_indices(&self, timesteps: &H3RowTimesteps) -> Vec<usize> {
        self.token_tags
            .iter()
            .map(|&tag| if tag == TAG_AUDIO { timesteps.audio_index } else { timesteps.video_index })
            .collect()
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
        assert_eq!((l.text, l.audio, l.video), (RowRange { start: 0, len: 3 }, RowRange { start: 3, len: 4 }, RowRange { start: 7, len: 8 }));
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
}
