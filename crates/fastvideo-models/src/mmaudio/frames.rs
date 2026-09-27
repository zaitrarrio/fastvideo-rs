//! Video frames to MMAudio's two visual inputs, on the host.
//!
//! `eval_utils.load_video` reads the clip once and resamples it to two rates
//! by timestamp (`av_utils.read_frames`): every decoded frame at time `t` is
//! appended to a rate's list while `t >= next`, `next += 1 / fps`. So a 16 fps
//! clip gives 8 fps by dropping every other frame and 25 fps by repeating
//! some. Then
//!
//! * CLIP: `Resize((384, 384), bicubic)` (a squash, antialiased), `/ 255`;
//!   `FeaturesUtils` normalizes with the OpenAI CLIP mean/std later.
//! * Synchformer: `Resize(224, bicubic)` (short side), `CenterCrop(224)`,
//!   `/ 255`, then `(x - 0.5) / 0.5`.
//!
//! `Resize` on a uint8 tensor is torch's antialiased bicubic
//! (`_upsample_bicubic2d_aa`, a = -0.5, PIL-style support), run as two
//! separable passes that each round back to uint8 (horizontal first).

use rayon::prelude::*;

pub const CLIP_SIZE: usize = 384;
pub const CLIP_FPS: f64 = 8.0;
pub const SYNC_SIZE: usize = 224;
pub const SYNC_FPS: f64 = 25.0;
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

/// Which source frames `read_frames` keeps for each rate in `fps_list`, given
/// the decoded frames' timestamps (seconds). `end_sec`: frames later than
/// it stop the read.
pub fn resample_indices(times: &[f64], fps_list: &[f64], end_sec: f64) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); fps_list.len()];
    let mut next = vec![0.0f64; fps_list.len()];
    for (i, &t) in times.iter().enumerate() {
        if t > end_sec {
            break;
        }
        for (k, fps) in fps_list.iter().enumerate() {
            let delta = 1.0 / fps;
            while t >= next[k] {
                out[k].push(i);
                next[k] += delta;
            }
        }
    }
    out
}

/// `load_video`'s selection: the clip-rate and sync-rate source indices and
/// the duration after its truncation to what both rates cover.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoSelection {
    pub duration_sec: f64,
    pub clip_indices: Vec<usize>,
    pub sync_indices: Vec<usize>,
}

pub fn select_frames(times: &[f64], duration_sec: f64) -> VideoSelection {
    let lists = resample_indices(times, &[CLIP_FPS, SYNC_FPS], duration_sec);
    let (mut clip, mut sync) = (lists[0].clone(), lists[1].clone());
    let mut duration = duration_sec;
    let clip_len = clip.len() as f64 / CLIP_FPS;
    if clip_len < duration {
        duration = clip_len;
    }
    let sync_len = sync.len() as f64 / SYNC_FPS;
    if sync_len < duration {
        duration = sync_len;
    }
    clip.truncate((CLIP_FPS * duration) as usize);
    sync.truncate((SYNC_FPS * duration) as usize);
    VideoSelection {
        duration_sec: duration,
        clip_indices: clip,
        sync_indices: sync,
    }
}

fn cubic(x: f64) -> f64 {
    // PyTorch aa bicubic filter, a = -0.5.
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Per output index: first source index and normalized weights
/// (`_compute_indices_min_size_weights_aa`, bicubic: interp_size 4).
fn aa_weights(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f32 / output as f32;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..output)
        .map(|i| {
            let center = scale * (i as f32 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut w: Vec<f64> = (xmin..xmax)
                .map(|j| cubic(f64::from((j as f32 - center + 0.5) * invscale)))
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|v| *v /= total);
            }
            (xmin, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

fn to_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Antialiased bicubic resize of one HWC rgb24 frame to `(oh, ow)`.
pub fn resize_bicubic_aa(src: &[u8], h: usize, w: usize, oh: usize, ow: usize) -> Vec<u8> {
    // Horizontal pass: [h, ow].
    let tmp: Vec<u8> = if ow == w {
        src.to_vec()
    } else {
        let wx = aa_weights(w, ow);
        let mut t = vec![0u8; h * ow * 3];
        for y in 0..h {
            for (x, (x0, ws)) in wx.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (k, wt) in ws.iter().enumerate() {
                        acc += wt * f32::from(src[(y * w + x0 + k) * 3 + c]);
                    }
                    t[(y * ow + x) * 3 + c] = to_u8(acc);
                }
            }
        }
        t
    };
    if oh == h {
        return tmp;
    }
    let wy = aa_weights(h, oh);
    let mut out = vec![0u8; oh * ow * 3];
    for (y, (y0, ws)) in wy.iter().enumerate() {
        for x in 0..ow {
            for c in 0..3 {
                let mut acc = 0.0f32;
                for (k, wt) in ws.iter().enumerate() {
                    acc += wt * f32::from(tmp[((y0 + k) * ow + x) * 3 + c]);
                }
                out[(y * ow + x) * 3 + c] = to_u8(acc);
            }
        }
    }
    out
}

/// CLIP input `[T, 3, 384, 384]` (values in `[0, 1]`, CLIP-normalized when
/// `normalize`; MMAudio normalizes inside `encode_video_with_clip`).
pub fn clip_pixels(frames: &[&[u8]], h: usize, w: usize, normalize: bool) -> Vec<f32> {
    let s = CLIP_SIZE;
    let per: Vec<Vec<f32>> = frames
        .par_iter()
        .map(|f| {
            let r = resize_bicubic_aa(f, h, w, s, s);
            let mut out = vec![0.0f32; 3 * s * s];
            for c in 0..3 {
                for p in 0..s * s {
                    let v = f32::from(r[p * 3 + c]) / 255.0;
                    out[c * s * s + p] = if normalize {
                        (v - CLIP_MEAN[c]) / CLIP_STD[c]
                    } else {
                        v
                    };
                }
            }
            out
        })
        .collect();
    per.concat()
}

/// torchvision `Resize(size)` on the short side: `(new_h, new_w)`.
pub fn short_side_size(h: usize, w: usize, size: usize) -> (usize, usize) {
    if h <= w {
        (size, (size as f64 * w as f64 / h as f64) as usize)
    } else {
        ((size as f64 * h as f64 / w as f64) as usize, size)
    }
}

/// Synchformer input `[T, 3, 224, 224]` in `[-1, 1]`.
pub fn sync_pixels(frames: &[&[u8]], h: usize, w: usize) -> Vec<f32> {
    let s = SYNC_SIZE;
    let (rh, rw) = short_side_size(h, w, s);
    // CenterCrop: top = int(round((rh - s) / 2)), left likewise.
    let top = (((rh - s) as f64) / 2.0).round() as usize;
    let left = (((rw - s) as f64) / 2.0).round() as usize;
    let per: Vec<Vec<f32>> = frames
        .par_iter()
        .map(|f| {
            let r = resize_bicubic_aa(f, h, w, rh, rw);
            let mut out = vec![0.0f32; 3 * s * s];
            for c in 0..3 {
                for y in 0..s {
                    for x in 0..s {
                        let v = f32::from(r[((top + y) * rw + left + x) * 3 + c]) / 255.0;
                        out[c * s * s + y * s + x] = (v - 0.5) / 0.5;
                    }
                }
            }
            out
        })
        .collect();
    per.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_fps_clip_rates() {
        // 81 frames at 16 fps, duration 81/16 as the sidecar passes it.
        let times: Vec<f64> = (0..81).map(|i| i as f64 / 16.0).collect();
        let sel = select_frames(&times, 81.0 / 16.0);
        // 8 fps keeps every other frame (41 of them); 25 fps covers t <= 5.0
        // with 125 frames (f64 accumulation of 1/25), so the duration
        // truncates to 125 / 25 = 5.0.
        assert!((sel.duration_sec - 5.0).abs() < 1e-12);
        assert_eq!(sel.clip_indices.len(), 40);
        assert_eq!(sel.sync_indices.len(), 125);
        assert_eq!(&sel.clip_indices[..4], &[0, 2, 4, 6]);
        // 25 fps from 16: t = 0 -> 0; .0625 -> 1; .125 covers .08 and .12 -> 2, 2.
        assert_eq!(&sel.sync_indices[..6], &[0, 1, 2, 2, 3, 4]);
    }

    #[test]
    fn resize_identity_and_constant() {
        let src: Vec<u8> = (0..4 * 6 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_bicubic_aa(&src, 4, 6, 4, 6), src);
        let flat = vec![200u8; 10 * 12 * 3];
        assert!(resize_bicubic_aa(&flat, 10, 12, 3, 5).iter().all(|&v| v == 200));
    }

    #[test]
    fn short_side() {
        assert_eq!(short_side_size(480, 832, 224), (224, 388));
    }
}
