//! The seams every family shares: the E1 hook bridge (engine cancel/progress
//! to `fastvideo_cudarc::Hooks`), pipeline errors to `ApiError`, and turning
//! a pipeline's PNG frames + WAV into an NVENC MP4 or in-memory frames.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastvideo_cudarc::wan::pipeline::PipelineError;
use fastvideo_cudarc::{Hooks, Progress, Stage};
use fastvideo_media::mp4::{AudioTarget, Mp4Spec, Mp4Writer};
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{ApiError, AudioPlan, JobMetrics, Pcm, ResolvedJob, RgbFrame};

use crate::backend::{ClipOutput, ClipSink};
use crate::cancel::{cancelled_error, OutputMode, StepControl};

/// What a pipeline left on disk, plus its numbers.
#[derive(Debug, Default)]
pub(crate) struct RawOutput {
    /// `frame-NNN.png`, in order.
    pub frame_paths: Vec<String>,
    /// `audio.wav` (models with audio).
    pub wav: Option<PathBuf>,
    pub metrics: JobMetrics,
}

/// Pipeline error to engine error.
pub(crate) fn api_err(what: &str, e: PipelineError) -> ApiError {
    if e.is_cancelled() {
        cancelled_error()
    } else {
        ApiError::engine_failed(format!("{what}: {e}"))
    }
}

pub(crate) fn bytes_mb(b: u64) -> f64 {
    b as f64 / f64::from(1u32 << 20)
}

/// Runs `f` with hooks bound to `ctl`: the job's cancel token trips the
/// pipeline's, stage boundaries become `Stage` events, and denoise steps
/// become `Progress{step, total}` counted across the stepping stages
/// (`planned` = the recipe's total steps, when known).
pub(crate) fn with_hooks<R>(
    ctl: &StepControl,
    planned: Option<u32>,
    f: impl FnOnce(Hooks<'_>) -> R,
) -> R {
    let token = fastvideo_cudarc::CancelToken::new();
    {
        let t = token.clone();
        ctl.cancel.on_cancel(move || t.cancel());
    }
    // Steps finished in earlier stepping stages, and the current stage's total.
    let done_before = Cell::new(0u32);
    let cur_total = Cell::new(0u32);
    let progress = |p: &Progress| {
        let stepping = matches!(p.stage, Stage::Denoise | Stage::Refine);
        if p.step == 0 && p.frames == 0 {
            // A stage boundary.
            if stepping {
                done_before.set(done_before.get() + cur_total.get());
                cur_total.set(p.total as u32);
            }
            ctl.stage(p.stage.name());
        } else if stepping && p.step > 0 {
            let step = done_before.get() + p.step as u32;
            let total = planned
                .unwrap_or(0)
                .max(done_before.get() + p.total as u32)
                .max(step);
            // The cancel check itself runs in the pipeline (cudarc token).
            let _ = ctl.step(step, total);
        }
    };
    let hooks = Hooks::default()
        .with_cancel(&token)
        .with_progress(&progress);
    f(hooks)
}

/// Where the MP4 goes and how it is encoded.
#[derive(Clone, Debug)]
pub(crate) struct Mp4Options {
    pub encoder: FfmpegH264,
    pub quality: u8,
    pub keep_frames: bool,
}

/// Minimal RIFF/WAVE reader: PCM16 or float32, interleaved.
pub(crate) fn read_wav(path: &Path) -> Result<Pcm, ApiError> {
    let bytes = std::fs::read(path)
        .map_err(|e| ApiError::engine_failed(format!("{}: {e}", path.display())))?;
    let bad = |m: &str| ApiError::engine_failed(format!("{}: {m}", path.display()));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let (mut fmt, mut data) = (None, None);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("4 bytes")) as usize;
        let body = at + 8;
        let end = (body + len).min(bytes.len());
        match id {
            b"fmt " => fmt = Some(&bytes[body..end]),
            b"data" => data = Some(&bytes[body..end]),
            _ => {}
        }
        at = body + len + (len & 1);
    }
    let (fmt, data) = (
        fmt.ok_or_else(|| bad("no fmt chunk"))?,
        data.ok_or_else(|| bad("no data chunk"))?,
    );
    if fmt.len() < 16 {
        return Err(bad("short fmt chunk"));
    }
    let u16_at = |i: usize| u16::from_le_bytes([fmt[i], fmt[i + 1]]);
    let (tag, channels) = (u16_at(0), u16_at(2));
    let rate = u32::from_le_bytes(fmt[4..8].try_into().expect("4 bytes"));
    let bits = u16_at(14);
    let samples: Vec<f32> = match (tag, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => {
            return Err(bad(&format!(
                "unsupported WAV format tag {tag} / {bits} bits"
            )))
        }
    };
    if channels == 0 || rate == 0 {
        return Err(bad("zero channels or rate"));
    }
    Ok(Pcm::new(rate, channels as u8, samples))
}

/// Centre crop of an RGB24 frame to `(w, h)` (pad-and-crop canvases).
pub(crate) fn crop(f: RgbFrame, (w, h): (u32, u32)) -> RgbFrame {
    if (f.width, f.height) == (w, h) || w > f.width || h > f.height {
        return f;
    }
    let (x0, y0) = ((f.width - w) / 2, (f.height - h) / 2);
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in y0..y0 + h {
        let row = ((y * f.width + x0) * 3) as usize;
        out.extend_from_slice(&f.data[row..row + (w * 3) as usize]);
    }
    RgbFrame {
        width: w,
        height: h,
        data: out.into(),
        index: f.index,
    }
}

fn load_frame(path: &str, index: u64) -> Result<RgbFrame, ApiError> {
    let mut f = fastvideo_media::probe::decode_image_rgb(Path::new(path))
        .map_err(|e| ApiError::engine_failed(format!("frame {path}: {e}")))?;
    f.index = index;
    Ok(f)
}

/// Turns the pipeline's files into the job's output.
pub(crate) fn finish(
    job: &ResolvedJob,
    raw: RawOutput,
    ctl: &StepControl,
    sink: &mut dyn ClipSink,
    work: &Path,
    mp4: &Mp4Options,
) -> Result<ClipOutput, ApiError> {
    ctl.check()?;
    let t0 = std::time::Instant::now();
    let keep_audio = job.audio.has_audio()
        && !job.post.drop_audio
        && matches!(job.audio, AudioPlan::Native { .. });
    let audio = match (&raw.wav, keep_audio) {
        (Some(w), true) => Some(read_wav(w)?),
        _ => None,
    };
    let out_size = job.output_size();
    let mut metrics = raw.metrics;
    let result = match &ctl.mode {
        OutputMode::File { dir } => {
            ctl.stage("encode");
            std::fs::create_dir_all(dir)
                .map_err(|e| ApiError::internal(format!("{}: {e}", dir.display())))?;
            let path = dir.join("output.mp4");
            let target = audio
                .as_ref()
                .map(|p| AudioTarget::native(p.rate, p.channels));
            let spec = Mp4Spec {
                quality: mp4.quality,
                encoder: mp4.encoder,
                ..Mp4Spec::new(out_size.0, out_size.1, job.fps, target)
            };
            let mut w = Mp4Writer::create(&path, spec, audio.as_ref())
                .map_err(|e| ApiError::engine_failed(format!("mp4: {e}")))?;
            for (i, p) in raw.frame_paths.iter().enumerate() {
                ctl.check()?;
                let f = crop(load_frame(p, i as u64)?, out_size);
                w.push(&f)
                    .map_err(|e| ApiError::engine_failed(format!("mp4: {e}")))?;
            }
            let path = w
                .finish()
                .map_err(|e| ApiError::engine_failed(format!("mp4: {e}")))?;
            ClipOutput {
                mp4: Some(path),
                frames: None,
                audio: None,
                metrics: JobMetrics::default(),
            }
        }
        OutputMode::Frames => {
            let mut frames = Vec::with_capacity(raw.frame_paths.len());
            for (i, p) in raw.frame_paths.iter().enumerate() {
                ctl.check()?;
                let f = crop(load_frame(p, i as u64)?, out_size);
                sink.frames(std::slice::from_ref(&f));
                frames.push(f);
            }
            if let Some(a) = &audio {
                sink.audio(a);
            }
            ClipOutput {
                mp4: None,
                frames: Some(frames),
                audio: audio.clone(),
                metrics: JobMetrics::default(),
            }
        }
    };
    metrics
        .stage_durations
        .insert("encode".into(), t0.elapsed().as_secs_f64());
    if !mp4.keep_frames {
        remove_frames(&raw.frame_paths, raw.wav.as_deref(), work);
    }
    Ok(ClipOutput { metrics, ..result })
}

/// Deletes the pipeline's PNGs/WAV (and the work dir when it ends up empty).
pub(crate) fn remove_frames(frames: &[String], wav: Option<&Path>, work: &Path) {
    for p in frames {
        let _ = std::fs::remove_file(p);
    }
    if let Some(w) = wav {
        let _ = std::fs::remove_file(w);
    }
    let _ = std::fs::remove_dir(work);
}

/// Stage seconds from `(name, seconds)` pairs, skipping zeros.
pub(crate) fn stages(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .filter(|(_, s)| *s > 0.0)
        .map(|(n, s)| ((*n).to_owned(), *s))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip_through_the_pipeline_writer() {
        let dir = std::env::temp_dir().join(format!("fv-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.wav");
        let s: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0) - 0.5).collect();
        fastvideo_cudarc::wan::pipeline::write_wav(&p, &s, 2, 32_000).unwrap();
        let pcm = read_wav(&p).unwrap();
        assert_eq!((pcm.rate, pcm.channels, pcm.samples.len()), (32_000, 2, 64));
        assert!((pcm.samples[3] - s[3]).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn centre_crop() {
        let data: Vec<u8> = (0..4 * 2 * 3).map(|i| i as u8).collect();
        let f = RgbFrame {
            width: 4,
            height: 2,
            data: data.into(),
            index: 7,
        };
        let c = crop(f, (2, 2));
        assert_eq!((c.width, c.height, c.index), (2, 2, 7));
        assert_eq!(&c.data[..], &[3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20]);
    }
}
