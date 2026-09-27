//! Resampling (rubato) and channel conversion (design §5.3).
//!
//! Targets:
//!
//! - **48 kHz** for every WebRTC path (Opus): H3 32 kHz, LTX 24 kHz (2.0/2.3)
//!   or 48 kHz (2.5 BWE), MMAudio at its model rate.
//! - **32 kHz** (the fal/H3 MP4 AAC rate) and **44.1 kHz** (RTMP/HLS AAC in
//!   the reference client) for MP4 and ffmpeg targets.
//!
//! The resampler is rubato's synchronous FFT resampler (`FftFixedInOut`),
//! which handles every rational ratio between these rates exactly. Its
//! filter delay is removed, so output sample `k` lines up with input time
//! `k/out_rate`, and a whole-buffer call returns exactly
//! `round(frames·out/in)` frames.

use rubato::{FftFixedInOut, Resampler};

use crate::av::{AvCheck, Pcm};
use crate::error::{MediaError, Result};
use crate::lockstep::clip_samples;

const CHUNK: usize = 1024;

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// An input chunk near [`CHUNK`] whose output chunk is even. rubato reports
/// the FFT resampler's delay as `chunk_out / 2`, which is exact only for an
/// even output chunk; an odd one (48k to 44.1k with 7 blocks of 147) would
/// leave a half-sample misalignment.
fn chunk_for(from: u32, to: u32) -> usize {
    let g = gcd(from, to);
    let (min_in, min_out) = ((from / g) as usize, (to / g) as usize);
    let mut blocks = CHUNK.div_ceil(min_in).max(1);
    if (blocks * min_out) % 2 == 1 {
        blocks += 1;
    }
    blocks * min_in
}

fn deinterleave(samples: &[f32], ch: usize) -> Vec<Vec<f32>> {
    let frames = samples.len() / ch;
    let mut out = vec![Vec::with_capacity(frames); ch];
    for f in samples.chunks_exact(ch) {
        for (c, &s) in f.iter().enumerate() {
            out[c].push(s);
        }
    }
    out
}

fn interleave(planes: &[Vec<f32>], frames: usize) -> Vec<f32> {
    let ch = planes.len();
    let mut out = Vec::with_capacity(frames * ch);
    for i in 0..frames {
        for p in planes {
            out.push(p[i]);
        }
    }
    out
}

/// Resample a whole buffer to `to_rate`. Returns exactly
/// `round(frames·to_rate/rate)` frames. Equal rates return a copy.
pub fn resample(pcm: &Pcm, to_rate: u32) -> Result<Pcm> {
    pcm.check()?;
    if to_rate == 0 {
        return Err(MediaError::invalid("target rate must be positive"));
    }
    if pcm.rate == to_rate {
        return Ok(pcm.clone());
    }
    let ch = pcm.channels as usize;
    let frames = pcm.frames();
    let want = clip_samples(frames as u64, pcm.rate, to_rate) as usize;
    if frames == 0 {
        return Ok(Pcm::silence(to_rate, pcm.channels, 0));
    }
    let mut s = StreamResampler::new(pcm.rate, to_rate, pcm.channels)?;
    let mut out = s.push(&pcm.samples)?;
    out.extend(s.flush()?);
    debug_assert_eq!(out.len(), want * ch);
    out.truncate(want * ch);
    out.resize(want * ch, 0.0);
    Ok(Pcm { rate: to_rate, channels: pcm.channels, samples: out.into() })
}

/// Streaming resampler: push interleaved chunks of any size, get interleaved
/// output with the filter delay already removed. After [`flush`](Self::flush)
/// the total output is exactly `round(total_in·out/in)` frames.
pub struct StreamResampler {
    inner: Option<FftFixedInOut<f32>>,
    from: u32,
    to: u32,
    channels: usize,
    /// Pending input, deinterleaved.
    pending: Vec<Vec<f32>>,
    /// Output frames still to discard (the filter delay).
    skip: usize,
    in_frames: u64,
    out_frames: u64,
    out_buf: Vec<Vec<f32>>,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32, channels: u8) -> Result<Self> {
        if from == 0 || to == 0 || channels == 0 {
            return Err(MediaError::invalid("rates and channels must be positive"));
        }
        let ch = channels as usize;
        let (inner, skip, out_buf) = if from == to {
            (None, 0, Vec::new())
        } else {
            let r = FftFixedInOut::<f32>::new(from as usize, to as usize, chunk_for(from, to), ch)
                .map_err(|e| MediaError::Resample(e.to_string()))?;
            let skip = r.output_delay();
            let buf = r.output_buffer_allocate(true);
            (Some(r), skip, buf)
        };
        Ok(Self {
            inner,
            from,
            to,
            channels: ch,
            pending: vec![Vec::new(); ch],
            skip,
            in_frames: 0,
            out_frames: 0,
            out_buf,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.from
    }

    pub fn output_rate(&self) -> u32 {
        self.to
    }

    fn run_chunks(&mut self, out: &mut Vec<Vec<f32>>, allow_partial: bool) -> Result<()> {
        let Some(r) = self.inner.as_mut() else { return Ok(()) };
        loop {
            let need = r.input_frames_next();
            let have = self.pending[0].len();
            if have < need {
                if !allow_partial || have == 0 {
                    return Ok(());
                }
                for p in &mut self.pending {
                    p.resize(need, 0.0);
                }
            }
            let (_, n) = r
                .process_into_buffer(&self.pending, &mut self.out_buf, None)
                .map_err(|e| MediaError::Resample(e.to_string()))?;
            for p in &mut self.pending {
                p.drain(..need);
            }
            let drop = self.skip.min(n);
            self.skip -= drop;
            for (o, b) in out.iter_mut().zip(&self.out_buf) {
                o.extend_from_slice(&b[drop..n]);
            }
        }
    }

    /// Push interleaved input; returns whatever output is ready (interleaved).
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        if samples.len() % self.channels != 0 {
            return Err(MediaError::invalid("input is not a whole number of sample frames"));
        }
        self.in_frames += (samples.len() / self.channels) as u64;
        if self.inner.is_none() {
            self.out_frames += (samples.len() / self.channels) as u64;
            return Ok(samples.to_vec());
        }
        for (c, p) in deinterleave(samples, self.channels).into_iter().enumerate() {
            self.pending[c].extend(p);
        }
        let mut out = vec![Vec::new(); self.channels];
        self.run_chunks(&mut out, false)?;
        let n = out[0].len();
        self.out_frames += n as u64;
        Ok(interleave(&out, n))
    }

    /// Drain the filter so the total output is exactly `round(in·to/from)`
    /// frames.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        let total_want = clip_samples(self.in_frames, self.from, self.to);
        if self.inner.is_none() || self.out_frames >= total_want {
            return Ok(Vec::new());
        }
        let mut out = vec![Vec::new(); self.channels];
        let mut produced = 0u64;
        let need_more = total_want - self.out_frames;
        let mut guard = 0;
        while produced < need_more {
            if self.pending[0].is_empty() {
                // Feed silence to push the tail through the filter.
                let r = self.inner.as_ref().expect("resampler");
                let n = r.input_frames_next();
                for p in &mut self.pending {
                    p.resize(n, 0.0);
                }
            }
            let before = out[0].len();
            self.run_chunks(&mut out, true)?;
            produced += (out[0].len() - before) as u64;
            guard += 1;
            if guard > 64 {
                return Err(MediaError::Resample("flush did not converge".into()));
            }
        }
        for o in &mut out {
            o.truncate(need_more as usize);
        }
        for p in &mut self.pending {
            p.clear();
        }
        self.out_frames += need_more;
        Ok(interleave(&out, need_more as usize))
    }
}

/// Convert to `channels`: mono is the mean of all input channels (Reactor's
/// wire format, §5.3); mono to N duplicates; N to M otherwise downmixes to
/// mono and duplicates.
pub fn to_channels(pcm: &Pcm, channels: u8) -> Result<Pcm> {
    pcm.check()?;
    if channels == 0 {
        return Err(MediaError::invalid("channels must be positive"));
    }
    if pcm.channels == channels {
        return Ok(pcm.clone());
    }
    let ch = pcm.channels as usize;
    let out_ch = channels as usize;
    let mut out = Vec::with_capacity(pcm.frames() * out_ch);
    for f in pcm.samples.chunks_exact(ch) {
        let m = if ch == 1 { f[0] } else { f.iter().sum::<f32>() / ch as f32 };
        out.extend(std::iter::repeat_n(m, out_ch));
    }
    Ok(Pcm { rate: pcm.rate, channels, samples: out.into() })
}

/// Mean downmix to mono.
pub fn downmix_mono(pcm: &Pcm) -> Result<Pcm> {
    to_channels(pcm, 1)
}

/// Resample and convert channels in one call.
pub fn convert(pcm: &Pcm, rate: u32, channels: u8) -> Result<Pcm> {
    // Mix down first when that shrinks the work; up-mix last.
    if channels < pcm.channels {
        resample(&to_channels(pcm, channels)?, rate)
    } else {
        to_channels(&resample(pcm, rate)?, channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine(rate: u32, hz: f64, secs: f64, channels: u8) -> Pcm {
        let n = (f64::from(rate) * secs).round() as usize;
        let mut v = Vec::with_capacity(n * channels as usize);
        for i in 0..n {
            let s = (2.0 * PI * hz * i as f64 / f64::from(rate)).sin() as f32 * 0.5;
            for c in 0..channels {
                v.push(if c == 0 { s } else { -s });
            }
        }
        Pcm::new(rate, channels, v)
    }

    /// Max error against the analytic sine, ignoring `edge` seconds at both
    /// ends (the filter sees zeros beyond the buffer).
    fn max_err(p: &Pcm, hz: f64, edge: f64) -> f64 {
        let ch = p.channels as usize;
        let lo = (edge * f64::from(p.rate)) as usize;
        let hi = p.frames() - lo;
        let mut worst: f64 = 0.0;
        for i in lo..hi {
            let want = (2.0 * PI * hz * i as f64 / f64::from(p.rate)).sin() * 0.5;
            for c in 0..ch {
                let w = if c == 0 { want } else { -want };
                worst = worst.max((f64::from(p.samples[i * ch + c]) - w).abs());
            }
        }
        worst
    }

    #[test]
    fn golden_32k_to_48k() {
        let src = sine(32_000, 1000.0, 1.0, 2);
        let out = resample(&src, 48_000).unwrap();
        assert_eq!(out.frames(), 48_000);
        assert_eq!(out.channels, 2);
        let e = max_err(&out, 1000.0, 0.02);
        assert!(e < 2e-3, "32k->48k error {e}");
    }

    #[test]
    fn golden_24k_to_48k() {
        let src = sine(24_000, 440.0, 0.5, 1);
        let out = resample(&src, 48_000).unwrap();
        assert_eq!(out.frames(), 24_000);
        let e = max_err(&out, 440.0, 0.02);
        assert!(e < 2e-3, "24k->48k error {e}");
    }

    #[test]
    fn golden_48k_to_mp4_rates() {
        for to in [32_000u32, 44_100] {
            let src = sine(48_000, 1000.0, 0.75, 2);
            let out = resample(&src, to).unwrap();
            assert_eq!(out.frames() as u64, clip_samples(36_000, 48_000, to));
            let e = max_err(&out, 1000.0, 0.02);
            assert!(e < 2e-3, "48k->{to} error {e}");
        }
    }

    #[test]
    fn identity_and_empty() {
        let src = sine(48_000, 1000.0, 0.01, 1);
        assert_eq!(resample(&src, 48_000).unwrap(), src);
        assert_eq!(resample(&Pcm::silence(32_000, 2, 0), 48_000).unwrap().frames(), 0);
        assert!(resample(&src, 0).is_err());
    }

    #[test]
    fn stream_matches_whole_buffer() {
        let src = sine(32_000, 700.0, 0.8, 2);
        let whole = resample(&src, 48_000).unwrap();
        let mut s = StreamResampler::new(32_000, 48_000, 2).unwrap();
        let mut out = Vec::new();
        // Odd chunk sizes, including ones smaller than the FFT chunk.
        let mut i = 0;
        let sizes = [7usize, 333, 1024, 2, 4099, 50];
        let mut k = 0;
        while i < src.samples.len() {
            let n = (sizes[k % sizes.len()] * 2).min(src.samples.len() - i);
            out.extend(s.push(&src.samples[i..i + n]).unwrap());
            i += n;
            k += 1;
        }
        out.extend(s.flush().unwrap());
        assert_eq!(out.len(), whole.samples.len());
        let d = out.iter().zip(whole.samples.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(d < 1e-5, "stream vs whole {d}");
    }

    #[test]
    fn channel_conversion() {
        let st = Pcm::new(48_000, 2, vec![1.0, 0.0, 0.5, 0.5]);
        let m = downmix_mono(&st).unwrap();
        assert_eq!(&*m.samples, &[0.5, 0.5]);
        let back = to_channels(&m, 2).unwrap();
        assert_eq!(&*back.samples, &[0.5, 0.5, 0.5, 0.5]);
        let c = convert(&st, 24_000, 1).unwrap();
        assert_eq!(c.channels, 1);
        assert_eq!(c.rate, 24_000);
        assert_eq!(c.frames(), 1);
    }
}
