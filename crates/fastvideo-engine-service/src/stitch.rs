//! Output stitching for LTX extend with less context than the source
//! (`ResolvedEdit::prefix_frames` / `suffix_frames`): the source frames the
//! model did not see are copied around the generated clip, resized and
//! center-cropped to the job's canvas, and the source's audio over the same
//! span is spliced around the generated audio (silence when the source has
//! none). The generated clip itself already contains the context window
//! (as a VAE round trip, like the references' output).
//!
//! Frames are counted in the source's own order (ffmpeg `trim`, no
//! autorotation, no frame-rate conversion); the audio spans are measured at
//! the output rate (`ResolvedJob::fps`) so audio and video stay aligned in
//! the written file.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use fastvideo_protocol::{ApiError, Pcm, ResolvedJob, RgbFrame};

fn failed(what: &str, e: impl std::fmt::Display) -> ApiError {
    ApiError::engine_failed(format!("stitch {what}: {e}"))
}

/// The source spans an edit's output keeps outside its generated clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Stitch {
    pub source: PathBuf,
    pub source_fps: f64,
    pub out_fps: u32,
    pub size: (u32, u32),
    /// `(first source frame, count)` before the generated clip.
    pub prefix: (u32, u32),
    /// `(first source frame, count)` after it.
    pub suffix: (u32, u32),
    pub source_audio: bool,
}

impl Stitch {
    /// `None` when the output is the generated clip alone.
    pub fn for_job(job: &ResolvedJob) -> Option<Self> {
        let e = job.edit.as_ref()?;
        let (pre, post) = (e.prefix_frames(), e.suffix_frames());
        if pre == 0 && post == 0 {
            return None;
        }
        Some(Self {
            source: e.source.clone(),
            source_fps: e.source_fps,
            out_fps: job.fps.max(1),
            size: job.output_size(),
            prefix: (0, pre),
            suffix: (e.window_start + e.window_frames, post),
            source_audio: e.source_audio,
        })
    }

    /// Streams source frames `start .. start + count` at the canvas size to
    /// `f`, indexed from `index0`. Returns how many were delivered (all of
    /// them, or an error).
    pub fn frames(
        &self,
        (start, count): (u32, u32),
        index0: u64,
        mut f: impl FnMut(RgbFrame) -> Result<(), ApiError>,
    ) -> Result<u32, ApiError> {
        if count == 0 {
            return Ok(0);
        }
        let (w, h) = self.size;
        let vf = format!(
            "trim=start_frame={start}:end_frame={},scale={w}:{h}:force_original_aspect_ratio=increase:flags=bilinear,crop={w}:{h}",
            start + count
        );
        let mut run = |sync: &str| -> Result<(u32, String, bool), ApiError> {
            let mut child = Command::new(fastvideo_media::tools::ffmpeg_bin())
                .args(["-v", "error", "-nostdin", "-noautorotate", "-i"])
                .arg(&self.source)
                .args(["-map", "0:v:0", "-vf", &vf, sync, "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| failed("ffmpeg", e))?;
            let mut out = child.stdout.take().ok_or_else(|| failed("ffmpeg", "no stdout"))?;
            let bytes = RgbFrame::byte_len(w, h);
            let mut n = 0u32;
            let mut fault = None;
            while n < count {
                let mut buf = vec![0u8; bytes];
                match out.read_exact(&mut buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(failed("read", e)),
                }
                let frame = RgbFrame::new(w, h, buf.into(), index0 + u64::from(n))?;
                if let Err(e) = f(frame) {
                    fault = Some(e);
                    break;
                }
                n += 1;
            }
            drop(out);
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_string(&mut err);
            }
            let _ = child.kill();
            let ok = child.wait().map(|s| s.success()).unwrap_or(false);
            if let Some(e) = fault {
                return Err(e);
            }
            Ok((n, err, ok))
        };
        let (mut n, mut err, mut ok) = run("-fps_mode")?;
        if n == 0 && !ok && err.contains("fps_mode") {
            (n, err, ok) = run("-vsync")?;
        }
        if n < count {
            return Err(failed(
                "frames",
                format!(
                    "{} gave {n} of {count} frames from {start}{}",
                    self.source.display(),
                    if ok { String::new() } else { format!(": {}", err.trim()) }
                ),
            ));
        }
        Ok(n)
    }

    /// `samples` interleaved sample frames of the source's audio from
    /// `start_s`, at `rate` / `channels`; silence where it has none.
    fn audio(&self, start_s: f64, samples: usize, rate: u32, channels: u8) -> Result<Vec<f32>, ApiError> {
        let want = samples * usize::from(channels);
        if want == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(want);
        if self.source_audio {
            let start = (start_s * f64::from(rate)).round().max(0.0) as u64;
            let af = format!("aresample={rate},atrim=start_sample={start}:end_sample={}", start + samples as u64);
            let o = Command::new(fastvideo_media::tools::ffmpeg_bin())
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(&self.source)
                .args(["-map", "0:a:0", "-vn", "-af", &af, "-ac", &channels.to_string(), "-f", "f32le", "pipe:1"])
                .stdin(Stdio::null())
                .output()
                .map_err(|e| failed("ffmpeg audio", e))?;
            if !o.status.success() {
                return Err(failed("audio", String::from_utf8_lossy(&o.stderr).trim().to_owned()));
            }
            out.extend(o.stdout.chunks_exact(4).take(want).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        out.resize(want, 0.0);
        Ok(out)
    }

    /// The generated audio with the source's audio over the prefix and the
    /// suffix spans around it (at the generated audio's rate and layout).
    pub fn splice_audio(&self, generated: &Pcm) -> Result<Pcm, ApiError> {
        let (rate, ch) = (generated.rate, generated.channels);
        let span = |frames: u32| (f64::from(frames) / f64::from(self.out_fps) * f64::from(rate)).round() as usize;
        let at = |frame: u32| f64::from(frame) / self.source_fps;
        let pre = self.audio(at(self.prefix.0), span(self.prefix.1), rate, ch)?;
        let post = self.audio(at(self.suffix.0), span(self.suffix.1), rate, ch)?;
        let mut all = Vec::with_capacity(pre.len() + generated.samples.len() + post.len());
        all.extend(pre);
        all.extend_from_slice(&generated.samples);
        all.extend(post);
        Ok(Pcm::new(rate, ch, all))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastvideo_protocol::{AudioPlan, ExtendAt, PostProcess, ResolvedEdit, ResolvedEditOp, SamplingOverrides, Task};

    fn job(at: ExtendAt, start: u32, window: u32, source: u32) -> ResolvedJob {
        ResolvedJob {
            model: "m".into(),
            task: Task::Extend,
            prompt: String::new(),
            negative_prompt: String::new(),
            seed: 1,
            width: 64,
            height: 32,
            num_frames: window + 48,
            fps: 24,
            keyframes: vec![],
            references: vec![],
            audio_in: None,
            audio: AudioPlan::Native { rate: 48_000, channels: 2 },
            post: PostProcess::default(),
            sampling: SamplingOverrides::default(),
            tier: None,
            recipe: None,
            edit: Some(ResolvedEdit {
                source: PathBuf::from("/nonexistent.mp4"),
                source_frames: source,
                source_fps: 24.0,
                source_size: (64, 32),
                source_audio: false,
                window_start: start,
                window_frames: window,
                op: ResolvedEditOp::Extend { frames: 48, at },
            }),
        }
    }

    #[test]
    fn spans_around_the_generated_clip() {
        // End: 200 source frames, the last 97 as context → 103 copied first.
        let j = job(ExtendAt::End, 103, 97, 200);
        assert_eq!(j.output_frames(), 103 + 97 + 48);
        let s = Stitch::for_job(&j).unwrap();
        assert_eq!((s.prefix, s.suffix), ((0, 103), (200, 0)));
        // Start: the first 97 as context → 103 copied after.
        let s = Stitch::for_job(&job(ExtendAt::Start, 0, 97, 200)).unwrap();
        assert_eq!((s.prefix, s.suffix), ((0, 0), (97, 103)));
        // Whole source as context: nothing to stitch.
        assert!(Stitch::for_job(&job(ExtendAt::End, 0, 97, 97)).is_none());
    }

    #[test]
    fn a_silent_source_pads_the_audio_with_silence() {
        let s = Stitch::for_job(&job(ExtendAt::End, 24, 97, 121)).unwrap();
        let gen = Pcm::new(48_000, 2, vec![0.5f32; 2 * 4_800]);
        let out = s.splice_audio(&gen).unwrap();
        // 24 frames at 24 fps = 1 s of silence, then the generated audio.
        assert_eq!(out.samples.len(), 2 * 48_000 + 2 * 4_800);
        assert!(out.samples[..2 * 48_000].iter().all(|v| *v == 0.0));
        assert!(out.samples[2 * 48_000..].iter().all(|v| *v == 0.5));
    }
}
