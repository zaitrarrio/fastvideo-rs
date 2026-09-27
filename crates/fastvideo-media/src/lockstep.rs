//! A/V lockstep arithmetic (design §5.5, streaming-refs §4.1-4.2).
//!
//! Every pacer tick carries one video frame and exactly `rate/fps` audio
//! sample frames. A built clip of `frames` video frames therefore carries
//! exactly `round(frames/fps·rate)` samples, and it is emitted in slices of
//! [`EMIT_FRAMES`] frames whose audio is `[round(lo·spf), round(hi·spf))`.
//!
//! | Model | fps | samples per frame at 48 kHz | per 3-frame slice |
//! |---|---|---|---|
//! | H3 / LTX | 24 | 2000 | 6000 |
//! | SF-Wan (causal) | 16 | 3000 | 9000 |

use crate::av::{AvCheck, Pcm};
use crate::error::{MediaError, Result};
use crate::resample;

/// The WebRTC audio rate: every WebRTC path runs Opus at 48 kHz (§5.3).
pub const WIRE_RATE: u32 = 48_000;

/// Frames per emitted slice (fast-h3 `EMIT_FRAMES`).
pub const EMIT_FRAMES: u32 = 3;

/// `rate / fps`, which must be an integer: `48000 % fps == 0` is an admission
/// check for clip sessions (§5.5).
pub fn samples_per_frame(rate: u32, fps: u32) -> Result<u32> {
    if fps == 0 || rate == 0 {
        return Err(MediaError::invalid("rate and fps must be positive"));
    }
    if rate % fps != 0 {
        return Err(MediaError::invalid(format!(
            "audio rate {rate} is not a whole number of samples per frame at {fps} fps"
        )));
    }
    Ok(rate / fps)
}

/// Admission check for a streaming session's fps (§5.5).
pub fn check_session_fps(fps: u32) -> Result<()> {
    samples_per_frame(WIRE_RATE, fps).map(|_| ())
}

/// `round(frames/fps·rate)`, computed exactly in integers (half rounds up).
pub fn clip_samples(frames: u64, fps: u32, rate: u32) -> u64 {
    let num = u128::from(frames) * u128::from(rate);
    let den = u128::from(fps.max(1));
    ((2 * num + den) / (2 * den)) as u64
}

/// Sample index where frame `i` starts: `round(i·rate/fps)`. For rates that
/// divide evenly this is exactly `i·spf`.
pub fn frame_sample_offset(i: u64, fps: u32, rate: u32) -> u64 {
    clip_samples(i, fps, rate)
}

/// Trim or zero-pad `pcm` to exactly `frames` sample frames per channel.
pub fn fit_len(pcm: &Pcm, frames: usize) -> Pcm {
    let ch = pcm.channels as usize;
    let want = frames * ch;
    let mut v: Vec<f32> = pcm.samples.iter().copied().take(want).collect();
    v.resize(want, 0.0);
    Pcm { rate: pcm.rate, channels: pcm.channels, samples: v.into() }
}

/// Turn a clip's native audio into its wire form (§5.5): resample to
/// `out_rate`, convert to `out_channels` (mono is the mean downmix), then trim
/// or pad to exactly `round(frames/fps·out_rate)` samples. A clip with no
/// audio gets that much silence.
pub fn prepare_clip_audio(native: Option<&Pcm>, frames: u32, fps: u32, out_rate: u32, out_channels: u8) -> Result<Pcm> {
    let want = clip_samples(u64::from(frames), fps, out_rate) as usize;
    let Some(native) = native else {
        return Ok(Pcm::silence(out_rate, out_channels, want));
    };
    native.check()?;
    let at_rate = resample::resample(native, out_rate)?;
    let mixed = resample::to_channels(&at_rate, out_channels)?;
    Ok(fit_len(&mixed, want))
}

/// One emit slice of a clip: frames `[frame_lo, frame_hi)` and their audio
/// sample frames `[sample_lo, sample_hi)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    pub frame_lo: u32,
    pub frame_hi: u32,
    pub sample_lo: u64,
    pub sample_hi: u64,
}

impl Slice {
    pub fn frames(&self) -> u32 {
        self.frame_hi - self.frame_lo
    }
    pub fn samples(&self) -> u64 {
        self.sample_hi - self.sample_lo
    }
}

/// The slices a clip of `frames` frames is emitted in (`per` frames each; the
/// last slice may be shorter). Audio bounds use the rounding rule above, so
/// the slices tile `[0, clip_samples)` with no gap and no overlap.
pub fn slices(frames: u32, per: u32, fps: u32, rate: u32) -> impl Iterator<Item = Slice> {
    let per = per.max(1);
    (0..frames.div_ceil(per)).map(move |k| {
        let lo = k * per;
        let hi = (lo + per).min(frames);
        Slice {
            frame_lo: lo,
            frame_hi: hi,
            sample_lo: frame_sample_offset(u64::from(lo), fps, rate),
            sample_hi: frame_sample_offset(u64::from(hi), fps, rate),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_per_frame_admission() {
        assert_eq!(samples_per_frame(48_000, 24).unwrap(), 2000);
        assert_eq!(samples_per_frame(48_000, 16).unwrap(), 3000);
        assert_eq!(samples_per_frame(48_000, 25).unwrap(), 1920);
        assert!(samples_per_frame(48_000, 7).is_err()); // 6857.14...
        assert!(check_session_fps(24).is_ok());
        assert!(check_session_fps(0).is_err());
    }

    #[test]
    fn clip_sample_counts() {
        // H3 5 s clip: 124 frames at 24 fps -> 248000 samples.
        assert_eq!(clip_samples(124, 24, 48_000), 248_000);
        // H3 10 s director chunk: 243 frames.
        assert_eq!(clip_samples(243, 24, 48_000), 486_000);
        // 44.1 kHz at 24 fps is fractional: 1837.5 per frame.
        assert_eq!(clip_samples(1, 24, 44_100), 1838);
        assert_eq!(clip_samples(2, 24, 44_100), 3675);
        // 32 kHz at 24 fps: 1333.33 per frame.
        assert_eq!(clip_samples(124, 24, 32_000), 165_333);
    }

    #[test]
    fn slices_tile_the_clip_exactly() {
        for &(frames, fps, rate) in &[(124u32, 24u32, 48_000u32), (81, 16, 48_000), (124, 24, 44_100), (1, 24, 48_000)] {
            let s: Vec<Slice> = slices(frames, EMIT_FRAMES, fps, rate).collect();
            assert_eq!(s.first().unwrap().sample_lo, 0);
            assert_eq!(s.last().unwrap().sample_hi, clip_samples(u64::from(frames), fps, rate));
            assert_eq!(s.last().unwrap().frame_hi, frames);
            for w in s.windows(2) {
                assert_eq!(w[0].sample_hi, w[1].sample_lo);
                assert_eq!(w[0].frame_hi, w[1].frame_lo);
            }
            if rate % fps == 0 {
                let spf = u64::from(rate / fps);
                for sl in &s {
                    assert_eq!(sl.samples(), u64::from(sl.frames()) * spf);
                }
            }
        }
        // fast-h3: 3 frames at 24 fps is 6000 samples.
        assert_eq!(slices(6, 3, 24, 48_000).next().unwrap().samples(), 6000);
    }

    #[test]
    fn prepare_clip_audio_hits_exact_counts() {
        // H3 native 32 kHz stereo, 124 frames (5.1667 s) but 5.184 s of audio.
        let native = Pcm::silence(32_000, 2, 165_888);
        let wire = prepare_clip_audio(Some(&native), 124, 24, 48_000, 1).unwrap();
        assert_eq!(wire.rate, 48_000);
        assert_eq!(wire.channels, 1);
        assert_eq!(wire.frames(), 248_000);
        // Too-short audio is padded.
        let short = Pcm::silence(24_000, 1, 1000);
        assert_eq!(prepare_clip_audio(Some(&short), 24, 24, 48_000, 2).unwrap().frames(), 48_000);
        // No audio: silence of the right length.
        let none = prepare_clip_audio(None, 81, 16, 48_000, 2).unwrap();
        assert_eq!(none.frames(), 243_000);
        assert!(none.samples.iter().all(|&s| s == 0.0));
    }
}
