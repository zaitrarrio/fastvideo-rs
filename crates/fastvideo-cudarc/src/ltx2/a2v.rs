//! LTX-2 audio-to-video: the driving audio, decoded and trimmed as the
//! reference does before its audio VAE encode.
//!
//! Reference: `ltx_pipelines/a2vid_two_stage.py` (Lightricks/LTX-2 `fd4ded7`):
//!
//! * `decode_audio_from_file(path, device, start_time=0, max_duration=num_frames / frame_rate)`
//!   (`utils/media_io/decode.py`): the first audio stream at the file's own
//!   sample rate and channel layout, as float in `[-1, 1]`, cut to
//!   `round(max_duration · rate)` samples;
//! * `vae_encode_audio` (`ltx_core/model/audio_vae/audio_vae.py:encode_audio`):
//!   resample to the encoder's 16 kHz, slaney log-mel, the causal encoder
//!   ([`super::audio_vae::AudioEncoder::encode_waveform`]);
//! * the latent cut to `AudioLatentShape.from_duration(num_frames / frame_rate).frames`
//!   (`round(duration · 25)`, the T2AV audio token count);
//! * `ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=…)` on both
//!   stages: the audio stream is clean conditioning
//!   ([`super::i2v_encode::StageConditioning::with_frozen_audio`],
//!   [`super::transformer::Ltx2Transformer::set_audio_frozen`]);
//! * the output carries the decoded input waveform itself ("Return the
//!   original input audio instead of VAE-decoded audio to preserve fidelity").
//!
//! One deliberate difference: the encoder takes exactly two channels
//! (`in_channels = 2`), and the reference passes the file's layout through, so
//! a mono file fails there. Here a file that is not stereo is decoded through
//! ffmpeg's `-ac 2` (mono is duplicated, surround is downmixed).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use fastvideo_models::ltx2::config::round_half_even;

use crate::wan::pipeline::{PipelineError, Result};

fn err(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// The channel count the LTX-2 audio VAE encoder takes.
pub const ENCODER_CHANNELS: usize = 2;

/// The driving audio of an audio-to-video request.
#[derive(Debug, Clone, PartialEq)]
pub struct DrivingAudio {
    pub path: PathBuf,
}

impl DrivingAudio {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

/// A decoded waveform: planar (channel-major) f32 at the file's own rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Waveform {
    pub planar: Vec<f32>,
    pub channels: usize,
    pub rate: u32,
}

impl Waveform {
    /// Samples per channel.
    pub fn samples(&self) -> usize {
        self.planar.len().checked_div(self.channels).unwrap_or(0)
    }

    pub fn duration_s(&self) -> f64 {
        if self.rate == 0 {
            0.0
        } else {
            self.samples() as f64 / f64::from(self.rate)
        }
    }

    /// Frame-major `[L R L R …]`.
    pub fn interleaved(&self) -> Vec<f32> {
        let n = self.samples();
        let mut out = vec![0f32; n * self.channels];
        for c in 0..self.channels {
            for i in 0..n {
                out[i * self.channels + c] = self.planar[c * n + i];
            }
        }
        out
    }

    /// `channels` of silence, `samples` long, at `rate`.
    pub fn silence(channels: usize, samples: usize, rate: u32) -> Self {
        Self {
            planar: vec![0.0; channels * samples],
            channels,
            rate,
        }
    }

    /// Drops the first `n` samples of every channel.
    pub fn skip(&mut self, n: usize) {
        let len = self.samples();
        let n = n.min(len);
        if n == 0 {
            return;
        }
        let mut out = Vec::with_capacity((len - n) * self.channels);
        for c in 0..self.channels {
            out.extend_from_slice(&self.planar[c * len + n..(c + 1) * len]);
        }
        self.planar = out;
    }

    /// Overwrites samples `at..` of every channel with `other`'s (same
    /// channel count and rate), as far as this waveform reaches.
    pub fn splice(&mut self, at: usize, other: &Waveform) -> Result<()> {
        if other.channels != self.channels || other.rate != self.rate {
            return Err(err(format!(
                "ltx2 audio splice: {} ch @ {} Hz into {} ch @ {} Hz",
                other.channels, other.rate, self.channels, self.rate
            )));
        }
        let (len, olen) = (self.samples(), other.samples());
        let n = olen.min(len.saturating_sub(at));
        for c in 0..self.channels {
            self.planar[c * len + at..c * len + at + n].copy_from_slice(&other.planar[c * olen..c * olen + n]);
        }
        Ok(())
    }

    /// The first `keep` samples of every channel.
    pub fn truncate(&mut self, keep: usize) {
        let n = self.samples();
        if keep >= n {
            return;
        }
        let mut out = Vec::with_capacity(keep * self.channels);
        for c in 0..self.channels {
            out.extend_from_slice(&self.planar[c * n..c * n + keep]);
        }
        self.planar = out;
    }
}

/// `max_samples = round(max_duration * sample_rate)` (`decode.py:294-296`,
/// Python's round: ties to even).
pub fn max_samples(max_duration_s: f64, rate: u32) -> usize {
    round_half_even(max_duration_s * f64::from(rate)).max(0.0) as usize
}

/// `(sample_rate, channels)` of the first audio stream (ffprobe).
pub fn probe(path: &Path) -> Result<(u32, usize)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .map_err(|e| err(format!("ffprobe not available: {e}")))?;
    if !out.status.success() {
        return Err(err(format!(
            "ffprobe {}: {}",
            path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    parse_probe(&String::from_utf8_lossy(&out.stdout))
        .ok_or_else(|| err(format!("{}: no audio stream", path.display())))
}

/// ffprobe `csv=p=0` of `stream=sample_rate,channels`: `44100,2`.
fn parse_probe(text: &str) -> Option<(u32, usize)> {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut it = line.split(',');
    let rate = it.next()?.trim().parse::<u32>().ok().filter(|&r| r > 0)?;
    let channels = it.next()?.trim().parse::<usize>().ok().filter(|&c| c > 0)?;
    Some((rate, channels))
}

/// Interleaved f32 → planar.
fn planar_of(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let n = interleaved.len() / channels;
    let mut out = vec![0f32; n * channels];
    for i in 0..n {
        for c in 0..channels {
            out[c * n + i] = interleaved[i * channels + c];
        }
    }
    out
}

/// `decode_audio_from_file(path, start_time=0, max_duration)`: the first audio
/// stream at its own rate, float, [`ENCODER_CHANNELS`] channels (see the
/// module note), at most `round(max_duration · rate)` samples.
pub fn decode(path: &Path, max_duration_s: Option<f64>) -> Result<Waveform> {
    decode_range(path, 0.0, max_duration_s, None)
}

/// `decode_audio_from_file(path, start_time, max_duration)`: as [`decode`],
/// from `start_s` (the first `round(start_s · rate)` samples dropped), and
/// resampled to `rate` by ffmpeg when given (a spliced track takes the
/// source's rate).
pub fn decode_range(path: &Path, start_s: f64, max_duration_s: Option<f64>, to_rate: Option<u32>) -> Result<Waveform> {
    let (native_rate, native) = probe(path)?;
    let rate = to_rate.unwrap_or(native_rate);
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vn", "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le"]);
    if native != ENCODER_CHANNELS {
        cmd.args(["-ac", &ENCODER_CHANNELS.to_string()]);
    }
    if rate != native_rate {
        cmd.args(["-ar", &rate.to_string()]);
    }
    let mut child = cmd
        .arg("pipe:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| err(format!("ffmpeg not available: {e}")))?;
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .ok_or_else(|| err("ffmpeg stdout closed"))?
        .read_to_end(&mut raw)
        .map_err(|e| err(format!("ffmpeg stdout: {e}")))?;
    let mut stderr = String::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_string(&mut stderr);
    }
    let status = child.wait().map_err(|e| err(format!("ffmpeg: {e}")))?;
    if !status.success() {
        return Err(err(format!(
            "ffmpeg audio decode of {} failed ({status}): {}",
            path.display(),
            stderr.trim()
        )));
    }
    let frame = 4 * ENCODER_CHANNELS;
    let whole = raw.len() - raw.len() % frame;
    let interleaved: Vec<f32> = raw[..whole]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if interleaved.is_empty() {
        return Err(err(format!("no audio samples in {}", path.display())));
    }
    let mut wave = Waveform {
        planar: planar_of(&interleaved, ENCODER_CHANNELS),
        channels: ENCODER_CHANNELS,
        rate,
    };
    if start_s > 0.0 {
        wave.skip(max_samples(start_s, rate));
    }
    if let Some(d) = max_duration_s {
        wave.truncate(max_samples(d, rate));
    }
    if native != ENCODER_CHANNELS {
        crate::wan::log::info(format_args!(
            "ltx2 a2v: {} has {native} channel(s); decoded as stereo (ffmpeg -ac 2)",
            path.display()
        ));
    }
    Ok(wave)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_samples_is_python_round() {
        // 121 frames at 24 fps, 44.1 kHz: 222337.5 → 222338 (even).
        assert_eq!(max_samples(121.0 / 24.0, 44_100), 222_338);
        // 0.5 ties go to even: 2.5 → 2, 3.5 → 4.
        assert_eq!(max_samples(2.5, 1), 2);
        assert_eq!(max_samples(3.5, 1), 4);
        assert_eq!(max_samples(5.0, 48_000), 240_000);
    }

    #[test]
    fn probe_output_parses() {
        assert_eq!(parse_probe("44100,2\n"), Some((44_100, 2)));
        assert_eq!(parse_probe("\n16000,1"), Some((16_000, 1)));
        assert_eq!(parse_probe(""), None);
        assert_eq!(parse_probe("0,2"), None);
        assert_eq!(parse_probe("48000"), None);
    }

    #[test]
    fn planar_interleaved_and_truncate() {
        let il = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let mut w = Waveform {
            planar: planar_of(&il, 2),
            channels: 2,
            rate: 3,
        };
        assert_eq!(w.planar, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        assert_eq!(w.samples(), 3);
        assert!((w.duration_s() - 1.0).abs() < 1e-12);
        assert_eq!(w.interleaved(), il);
        w.truncate(2);
        assert_eq!(w.planar, vec![1.0, 2.0, -1.0, -2.0]);
        w.truncate(5);
        assert_eq!(w.samples(), 2);
    }
}
