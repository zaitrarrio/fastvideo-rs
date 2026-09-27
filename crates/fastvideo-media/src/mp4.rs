//! MP4 output (design §4 common conventions).
//!
//! - [`Mp4Writer`] / [`write_mp4`]: RGB24 frames plus optional PCM to an MP4
//!   through ffmpeg, with the engine's `VideoWriter` settings
//!   (`crates/fastvideo-cudarc/src/wan/writer.rs`: libx264 `veryfast` crf 19
//!   yuv420p, rgb24 on stdin, audio from a side file, AAC), plus
//!   `+faststart`. [`Mp4Spec::fal_h3`] is the fal/H3 shape: H.264, 24 fps,
//!   AAC-LC stereo 32 kHz, faststart.
//! - [`finalize`]: the post-processor every batch job runs: `-c copy
//!   -movflags +faststart`, `-an` for silent output, and a crop (re-encoded
//!   with the same x264 settings) for pad-and-crop canvases.
//! - [`inspect`]: a pure-Rust box reader to verify all of that.

pub mod inspect;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};

pub use inspect::{inspect, AudioInfo, AvcInfo, Mp4Info, TrackInfo, TrackKind};

use crate::av::{f32_to_f32le, AvCheck, Pcm, RgbFrame};
use crate::error::{MediaError, Result};
use crate::lockstep::clip_samples;
use crate::resample;
use crate::tools;

/// AAC output settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AudioTarget {
    pub rate: u32,
    pub channels: u8,
    pub bitrate_bps: u32,
}

impl AudioTarget {
    /// fal / hosted H3: AAC-LC stereo 32 kHz (fal §6).
    pub const FAL_H3: AudioTarget = AudioTarget { rate: 32_000, channels: 2, bitrate_bps: 128_000 };
    /// RTMP/HLS reference client rate: 44.1 kHz stereo.
    pub const CD: AudioTarget = AudioTarget { rate: 44_100, channels: 2, bitrate_bps: 128_000 };
    /// RTMP/HLS in this design (§5.3): AAC 128k at 48 kHz stereo.
    pub const BROADCAST: AudioTarget = AudioTarget { rate: 48_000, channels: 2, bitrate_bps: 128_000 };

    /// Keep a model's native rate and layout (LTX, MMAudio).
    pub fn native(rate: u32, channels: u8) -> Self {
        Self { rate, channels, bitrate_bps: 128_000 }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mp4Spec {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub crf: u8,
    pub preset: String,
    /// `None` writes a video-only file.
    pub audio: Option<AudioTarget>,
    pub faststart: bool,
}

impl Mp4Spec {
    /// The engine writer's settings (crf 19, veryfast), with the audio target given.
    pub fn new(width: u32, height: u32, fps: u32, audio: Option<AudioTarget>) -> Self {
        Self { width, height, fps, crf: 19, preset: "veryfast".into(), audio, faststart: true }
    }

    /// What hosted H3 returns on fal: 24 fps, AAC-LC stereo 32 kHz, faststart.
    pub fn fal_h3(width: u32, height: u32) -> Self {
        Self::new(width, height, 24, Some(AudioTarget::FAL_H3))
    }
}

/// Streams RGB24 frames into ffmpeg; audio (already known) is staged in a
/// side file first, as the engine writer does.
pub struct Mp4Writer {
    spec: Mp4Spec,
    out: PathBuf,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    audio_file: Option<PathBuf>,
    frames: u64,
}

impl Mp4Writer {
    /// `audio` is resampled and remixed to `spec.audio` here (not in ffmpeg),
    /// so the AAC track carries exactly the samples given.
    pub fn create(out: &Path, spec: Mp4Spec, audio: Option<&Pcm>) -> Result<Self> {
        if spec.width % 2 != 0 || spec.height % 2 != 0 || spec.fps == 0 {
            return Err(MediaError::invalid("mp4 needs even dimensions and a positive fps"));
        }
        let mut cmd = tools::ffmpeg_command();
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"])
            .arg(format!("{}x{}", spec.width, spec.height))
            .args(["-framerate", &spec.fps.to_string(), "-i", "pipe:0"]);
        let mut audio_file = None;
        match (spec.audio, audio) {
            (Some(t), Some(pcm)) => {
                let pcm = resample::convert(pcm, t.rate, t.channels)?;
                let af = out.with_extension("audio.f32");
                std::fs::write(&af, f32_to_f32le(&pcm.samples))?;
                cmd.args(["-f", "f32le", "-ar", &t.rate.to_string(), "-ac", &t.channels.to_string(), "-i"])
                    .arg(&af)
                    .args(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-profile:a", "aac_low"])
                    .args(["-b:a", &t.bitrate_bps.to_string(), "-ar", &t.rate.to_string(), "-ac", &t.channels.to_string()]);
                audio_file = Some(af);
            }
            (Some(_), None) => {
                return Err(MediaError::invalid("the spec asks for audio but none was given"));
            }
            (None, _) => {
                cmd.arg("-an");
            }
        }
        cmd.args(["-c:v", "libx264", "-preset", &spec.preset, "-crf", &spec.crf.to_string(), "-pix_fmt", "yuv420p"])
            .args(["-r", &spec.fps.to_string()]);
        if spec.faststart {
            cmd.args(["-movflags", "+faststart"]);
        }
        cmd.arg(out).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| MediaError::tool("ffmpeg", format!("not available: {e}")))?;
        let stdin = child.stdin.take();
        Ok(Self { spec, out: out.to_path_buf(), child: Some(child), stdin, audio_file, frames: 0 })
    }

    pub fn push(&mut self, frame: &RgbFrame) -> Result<()> {
        frame.check()?;
        if frame.width != self.spec.width || frame.height != self.spec.height {
            return Err(MediaError::invalid("frame size does not match the mp4 spec"));
        }
        let w = self.stdin.as_mut().ok_or_else(|| MediaError::invalid("writer finished"))?;
        w.write_all(&frame.data).map_err(|e| MediaError::tool("ffmpeg", format!("stdin: {e}")))?;
        self.frames += 1;
        Ok(())
    }

    /// Close ffmpeg, check its exit, remove the side file.
    pub fn finish(mut self) -> Result<PathBuf> {
        drop(self.stdin.take());
        let child = self.child.take().ok_or_else(|| MediaError::invalid("writer finished"))?;
        let out = child.wait_with_output()?;
        if let Some(af) = self.audio_file.take() {
            let _ = std::fs::remove_file(af);
        }
        if !out.status.success() {
            return Err(MediaError::tool(
                "ffmpeg",
                format!("mp4 mux exited with {}: {}", out.status, String::from_utf8_lossy(&out.stderr).trim()),
            ));
        }
        Ok(self.out.clone())
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }
}

impl Drop for Mp4Writer {
    fn drop(&mut self) {
        drop(self.stdin.take());
        if let Some(mut c) = self.child.take() {
            let _ = c.wait();
        }
        if let Some(af) = self.audio_file.take() {
            let _ = std::fs::remove_file(af);
        }
    }
}

/// Write a whole clip. With audio, the PCM is first fitted to exactly
/// `round(frames/fps·rate)` samples at the target rate (A/V lockstep).
pub fn write_mp4(out: &Path, spec: Mp4Spec, frames: &[RgbFrame], audio: Option<&Pcm>) -> Result<PathBuf> {
    let fitted = match (spec.audio, audio) {
        (Some(t), Some(pcm)) => {
            let conv = resample::convert(pcm, t.rate, t.channels)?;
            let n = clip_samples(frames.len() as u64, spec.fps, t.rate) as usize;
            Some(crate::lockstep::fit_len(&conv, n))
        }
        _ => None,
    };
    let mut w = Mp4Writer::create(out, spec, fitted.as_ref().or(audio))?;
    for f in frames {
        w.push(f)?;
    }
    w.finish()
}

/// A crop window; `x`/`y` default to centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Crop {
    pub width: u32,
    pub height: u32,
    pub x: Option<u32>,
    pub y: Option<u32>,
}

/// `ResolvedJob.post` (design §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct PostProcess {
    pub crop: Option<Crop>,
    pub drop_audio: bool,
}

impl From<&fastvideo_protocol::PostProcess> for PostProcess {
    /// The protocol's crop is centred (pad-and-crop pads symmetrically).
    fn from(p: &fastvideo_protocol::PostProcess) -> Self {
        Self {
            crop: p.crop.map(|(width, height)| Crop { width, height, x: None, y: None }),
            drop_audio: p.drop_audio,
        }
    }
}

/// Remux `input` to `output` with `+faststart`; `-an` when `drop_audio`;
/// a crop re-encodes video with the engine writer's x264 settings and copies
/// audio.
pub fn finalize(input: &Path, output: &Path, post: &PostProcess) -> Result<()> {
    if input == output {
        return Err(MediaError::invalid("finalize needs a distinct output path"));
    }
    let mut cmd = tools::ffmpeg_command();
    cmd.arg("-i").arg(input).args(["-map", "0:v:0"]);
    if !post.drop_audio {
        cmd.args(["-map", "0:a?"]);
    }
    match post.crop {
        Some(c) => {
            if c.width % 2 != 0 || c.height % 2 != 0 {
                return Err(MediaError::invalid("crop needs even dimensions"));
            }
            let x = c.x.map(|v| v.to_string()).unwrap_or_else(|| "(in_w-out_w)/2".into());
            let y = c.y.map(|v| v.to_string()).unwrap_or_else(|| "(in_h-out_h)/2".into());
            cmd.args(["-vf", &format!("crop={}:{}:{x}:{y}", c.width, c.height)])
                .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p"]);
            if !post.drop_audio {
                cmd.args(["-c:a", "copy"]);
            }
        }
        None => {
            cmd.args(["-c", "copy"]);
        }
    }
    if post.drop_audio {
        cmd.arg("-an");
    }
    cmd.args(["-movflags", "+faststart"]).arg(output);
    tools::run_checked(cmd, "ffmpeg").map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        let s = Mp4Spec::fal_h3(1344, 768);
        assert_eq!(s.fps, 24);
        assert_eq!(s.audio, Some(AudioTarget { rate: 32_000, channels: 2, bitrate_bps: 128_000 }));
        assert!(s.faststart);
        assert_eq!((s.crf, s.preset.as_str()), (19, "veryfast"));
        assert_eq!(AudioTarget::CD.rate, 44_100);
    }

    #[test]
    fn finalize_rejects_in_place_and_odd_crops() {
        let p = Path::new("/nonexistent/a.mp4");
        assert!(finalize(p, p, &PostProcess::default()).is_err());
        let post = PostProcess { crop: Some(Crop { width: 1921, height: 1080, x: None, y: None }), drop_audio: false };
        assert!(matches!(finalize(p, Path::new("/nonexistent/b.mp4"), &post), Err(MediaError::InvalidArgument(_))));
        let from = PostProcess::from(&fastvideo_protocol::PostProcess { crop: Some((1920, 1080)), drop_audio: true });
        assert_eq!(from.crop, Some(Crop { width: 1920, height: 1080, x: None, y: None }));
        assert!(from.drop_audio);
    }
}
