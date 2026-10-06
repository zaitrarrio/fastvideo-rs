//! LingBot 1080p refiner stage, host side (`runner.py` refiner block and
//! `utils.py`).
//!
//! The reference refines the *saved* base video: it writes the base frames to
//! an mp4, reads them back (`decord`), keeps `compute_training_frame_budget`
//! frames, resizes each with `F.interpolate(mode="bicubic",
//! align_corners=False)` to the refiner canvas, clamps to `[0, 1]`, encodes
//! with the Wan VAE (`latent_dist.sample`, then `(z - mean) / std`), and
//! starts the refiner from `(1 - t) * x_up + t * noise` at `t = t_thresh`.
//!
//! This port keeps the frames in memory instead of the mp4 round trip: the
//! decoded base frames are quantized to u8 (what the encoder would receive)
//! and skip the codec's loss. That is the one intended difference.

/// `compute_training_frame_budget(num_source_frames, source_fps, sample_fps, vae_tc)`
/// → `(sample_frame, vae_fps, t_vae)`.
pub fn training_frame_budget(
    num_source_frames: usize,
    source_fps: f64,
    sample_fps: u32,
    vae_tc: usize,
) -> (usize, f64, usize) {
    if num_source_frames == 0 {
        return (1, 0.0, 1);
    }
    let raw = if source_fps > f64::from(sample_fps) {
        (num_source_frames as f64 / source_fps * f64::from(sample_fps)) as i64
    } else {
        num_source_frames as i64
    };
    let tc = vae_tc as i64;
    let sample = ((((raw - 1).div_euclid(tc)) * tc) + 1).max(1) as usize;
    let vae_fps = sample as f64 / num_source_frames as f64 * source_fps;
    let t_vae = (sample - 1) / vae_tc + 1;
    (sample, vae_fps, t_vae)
}

/// `compute_training_aligned_indices`.
pub fn training_aligned_indices(num_source_frames: usize, sample_frame: usize) -> Vec<usize> {
    if sample_frame == 0 {
        return Vec::new();
    }
    if num_source_frames == 0 {
        return vec![0; sample_frame];
    }
    if num_source_frames >= sample_frame {
        // np.linspace(0, n - 1, sample, dtype=int): truncation toward zero.
        if sample_frame == 1 {
            return vec![0];
        }
        return (0..sample_frame)
            .map(|i| ((num_source_frames - 1) as f64 * i as f64 / (sample_frame - 1) as f64) as usize)
            .collect();
    }
    let mut out: Vec<usize> = (0..num_source_frames).collect();
    out.resize(sample_frame, num_source_frames - 1);
    out
}

fn cubic_weights(t: f32) -> [f32; 4] {
    // torch upsample_bicubic2d, A = -0.75.
    const A: f32 = -0.75;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
}

/// Source taps (clamped indices) and weights per output coordinate for
/// `align_corners=False` with the default scale `in / out`.
fn bicubic_plan(in_len: usize, out_len: usize) -> Vec<([usize; 4], [f32; 4])> {
    let scale = in_len as f32 / out_len as f32;
    (0..out_len)
        .map(|o| {
            let real = scale * (o as f32 + 0.5) - 0.5;
            let i0 = real.floor();
            let t = real - i0;
            let i0 = i0 as i64;
            let clamp = |i: i64| i.clamp(0, in_len as i64 - 1) as usize;
            (
                [clamp(i0 - 1), clamp(i0), clamp(i0 + 1), clamp(i0 + 2)],
                cubic_weights(t),
            )
        })
        .collect()
}

/// `F.interpolate(x, (oh, ow), mode="bicubic", align_corners=False)` on one
/// `[C, H, W]` plane set, then `clamp(0, 1)` (`resize_video_tensor`).
pub fn resize_bicubic_chw(src: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize) -> Vec<f32> {
    assert_eq!(src.len(), c * h * w);
    let py = bicubic_plan(h, oh);
    let px = bicubic_plan(w, ow);
    let mut out = vec![0f32; c * oh * ow];
    for ci in 0..c {
        let plane = &src[ci * h * w..(ci + 1) * h * w];
        for (oy, (ys, wy)) in py.iter().enumerate() {
            for (ox, (xs, wx)) in px.iter().enumerate() {
                let mut acc = 0f32;
                for (j, &y) in ys.iter().enumerate() {
                    let row = &plane[y * w..(y + 1) * w];
                    let mut r = 0f32;
                    for (i, &x) in xs.iter().enumerate() {
                        r += row[x] * wx[i];
                    }
                    acc += r * wy[j];
                }
                out[(ci * oh + oy) * ow + ox] = acc.clamp(0.0, 1.0);
            }
        }
    }
    out
}

/// What the mp4 writer stores for a decoded frame in `[0, 1]`: `round(x * 255)`
/// back as `/ 255` (`decord` read, `.float() / 255.0`).
pub fn quantize_u8(x: f32) -> f32 {
    (x.clamp(0.0, 1.0) * 255.0).round() / 255.0
}

/// `prepare_refiner_latent`: `(1 - t) * x_up + t * noise`.
pub fn noised_start(x_up: &[f32], noise: &[f32], t_thresh: f32) -> Vec<f32> {
    x_up.iter()
        .zip(noise)
        .map(|(&x, &n)| (1.0 - t_thresh) * x + t_thresh * n)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_budget_keeps_all_121_frames() {
        let (sample, fps, t_vae) = training_frame_budget(121, 24.0, 24, 4);
        assert_eq!((sample, t_vae), (121, 31));
        assert!((fps - 24.0).abs() < 1e-12);
        let idx = training_aligned_indices(121, sample);
        assert_eq!(idx, (0..121).collect::<Vec<_>>());
    }

    #[test]
    fn budget_downsamples_faster_sources() {
        let (sample, _, _) = training_frame_budget(240, 48.0, 24, 4);
        assert_eq!(sample, 117); // raw 120 → ((119 // 4) * 4) + 1
        let idx = training_aligned_indices(10, 4);
        assert_eq!(idx, vec![0, 3, 6, 9]);
    }

    #[test]
    fn bicubic_is_identity_at_equal_size_and_keeps_constants() {
        let src: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) / 24.0).collect();
        let same = resize_bicubic_chw(&src, 2, 3, 4, 3, 4);
        for (a, b) in src.iter().zip(&same) {
            assert!((a - b).abs() < 1e-6);
        }
        let flat = vec![0.4f32; 3 * 5 * 7];
        let up = resize_bicubic_chw(&flat, 3, 5, 7, 11, 13);
        assert!(up.iter().all(|v| (v - 0.4).abs() < 1e-6));
    }

    #[test]
    fn bicubic_upsample_matches_reference_row() {
        // [0, 1, 0, 1] → 8 wide, then clamp. Values from an independent numpy
        // transcription of ATen's upsample_bicubic2d (A = -0.75, source index
        // `scale * (dst + 0.5) - 0.5`, clamped taps); not a torch run.
        let src = vec![0.0f32, 1.0, 0.0, 1.0];
        let got = resize_bicubic_chw(&src, 1, 1, 4, 1, 8);
        let want = [0.0f32, 0.26171875, 0.87890625, 0.84375, 0.15625, 0.12109375, 0.73828125, 1.0];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{got:?}");
        }
    }

    #[test]
    fn start_latent_mixes_at_t_thresh() {
        let x = noised_start(&[1.0, 0.0], &[0.0, 1.0], 0.85);
        assert!((x[0] - 0.15).abs() < 1e-6 && (x[1] - 0.85).abs() < 1e-6);
        assert_eq!(quantize_u8(0.5), 128.0 / 255.0);
    }
}
