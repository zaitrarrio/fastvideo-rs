//! Raised-cosine clip-edge fades (design §5.5 `Continuity::Crossfade`).
//!
//! `Crossfade{ms}` fades clip N out over its last `ms` and clip N+1 in over
//! its first `ms`. Nothing is overlapped or removed, so **sample counts never
//! change** and A/V lockstep is untouched. The gain curves are
//! `g_in(t) = 0.5·(1 − cos πt)` and `g_out(t) = 0.5·(1 + cos πt)` for
//! `t ∈ [0, 1]`, so the boundary sample pair is at (or next to) zero on both
//! sides and there is no click.

use crate::av::Pcm;

/// Default crossfade for every clip session except Reactor fast-h3 parity.
pub const DEFAULT_CROSSFADE_MS: u16 = 20;

/// Sample frames covered by `ms` at `rate`, capped at `frames`.
pub fn fade_frames(rate: u32, ms: u16, frames: usize) -> usize {
    ((u64::from(rate) * u64::from(ms) / 1000) as usize).min(frames)
}

fn gain_in(i: usize, n: usize) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    let t = i as f64 / (n - 1) as f64;
    (0.5 * (1.0 - (std::f64::consts::PI * t).cos())) as f32
}

/// Fade the first `ms` in (in place, interleaved).
pub fn fade_in(samples: &mut [f32], rate: u32, channels: u8, ms: u16) {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let n = fade_frames(rate, ms, frames);
    for i in 0..n {
        let g = gain_in(i, n);
        for s in &mut samples[i * ch..(i + 1) * ch] {
            *s *= g;
        }
    }
}

/// Fade the last `ms` out (in place, interleaved).
pub fn fade_out(samples: &mut [f32], rate: u32, channels: u8, ms: u16) {
    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let n = fade_frames(rate, ms, frames);
    let start = frames - n;
    for i in 0..n {
        // Mirror of the fade-in: full gain at the start of the window, 0 at the end.
        let g = gain_in(n - 1 - i, n);
        for s in &mut samples[(start + i) * ch..(start + i + 1) * ch] {
            *s *= g;
        }
    }
}

/// Apply the clip-edge fades a `Crossfade{ms}` session uses: fade in unless
/// this is the first clip, fade out unless it is known to be the last. The
/// returned PCM has exactly the input's sample count.
pub fn apply_clip_fades(pcm: &Pcm, ms: u16, fade_head: bool, fade_tail: bool) -> Pcm {
    let mut v: Vec<f32> = pcm.samples.to_vec();
    if fade_head {
        fade_in(&mut v, pcm.rate, pcm.channels, ms);
    }
    if fade_tail {
        fade_out(&mut v, pcm.rate, pcm.channels, ms);
    }
    Pcm { rate: pcm.rate, channels: pcm.channels, samples: v.into() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_sample_counts() {
        for frames in [0usize, 1, 5, 959, 960, 961, 248_000] {
            let p = Pcm::new(48_000, 2, vec![0.8f32; frames * 2]);
            let f = apply_clip_fades(&p, 20, true, true);
            assert_eq!(f.samples.len(), p.samples.len());
            assert_eq!(f.frames(), frames);
        }
    }

    #[test]
    fn raised_cosine_shape() {
        let p = Pcm::new(48_000, 1, vec![1.0f32; 48_000]);
        let f = apply_clip_fades(&p, 20, true, true);
        let n = 960; // 20 ms at 48 kHz
        assert_eq!(f.samples[0], 0.0);
        assert!((f.samples[n / 2] - 0.5).abs() < 0.01);
        assert!((f.samples[n - 1] - 1.0).abs() < 1e-6);
        assert_eq!(f.samples[n], 1.0); // untouched middle
        assert_eq!(f.samples[24_000], 1.0);
        assert!((f.samples[48_000 - n] - 1.0).abs() < 1e-6);
        assert_eq!(f.samples[47_999], 0.0);
        // Monotonic fade in.
        for i in 1..n {
            assert!(f.samples[i] >= f.samples[i - 1]);
        }
    }

    #[test]
    fn boundary_between_clips_is_continuous() {
        // A DC step between two clips becomes a smooth dip through zero.
        let a = apply_clip_fades(&Pcm::new(48_000, 1, vec![0.9f32; 4800]), 20, false, true);
        let b = apply_clip_fades(&Pcm::new(48_000, 1, vec![-0.9f32; 4800]), 20, true, false);
        let joined: Vec<f32> = a.samples.iter().chain(b.samples.iter()).copied().collect();
        let max_step = joined.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_step < 0.01, "largest sample step {max_step}");
    }

    #[test]
    fn stereo_channels_share_the_gain() {
        let mut v = [1.0f32, -1.0].repeat(100);
        fade_in(&mut v, 1000, 2, 50);
        for f in v.chunks_exact(2) {
            assert_eq!(f[0], -f[1]);
        }
    }
}
