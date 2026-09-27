//! The seams every family shares: the E1 hook bridge (engine cancel/progress
//! to `fastvideo_cudarc::Hooks`), pipeline errors to `ApiError`, and the E2
//! frame sink that turns a pipeline's in-memory RGB8 frames + PCM into an
//! NVENC MP4 or into `ClipOutput::{frames, audio}`.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fastvideo_cudarc::sink::{AudioPcm, FrameSink, SinkPort, VideoFrames};
use fastvideo_cudarc::wan::pipeline::PipelineError;
use fastvideo_cudarc::{Hooks, Progress, Stage};
use fastvideo_media::mp4::{AudioTarget, Mp4Spec, Mp4Writer};
use fastvideo_media::video::FfmpegH264;
use fastvideo_protocol::{ApiError, AudioPlan, JobMetrics, Pcm, ResolvedJob, RgbFrame};

use crate::backend::{ClipOutput, ClipSink};
use crate::cancel::{cancelled_error, OutputMode, StepControl};

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

/// Stage seconds from `(name, seconds)` pairs, skipping zeros.
pub(crate) fn stages(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .filter(|(_, s)| *s > 0.0)
        .map(|(n, s)| ((*n).to_owned(), *s))
        .collect()
}

/// Whether the job's output carries the model's audio.
pub(crate) fn wants_audio(job: &ResolvedJob) -> bool {
    matches!(job.audio, AudioPlan::Native { .. }) && !job.post.drop_audio
}

/// How the MP4 is encoded and whether PNGs are kept.
#[derive(Clone, Debug)]
pub(crate) struct Mp4Options {
    pub encoder: FfmpegH264,
    pub quality: u8,
    /// Also write the pipelines' `frame-NNN.png` into the work directory
    /// (identity checks against the CLI).
    pub keep_frames: bool,
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

fn sink_err(e: impl std::fmt::Display) -> PipelineError {
    PipelineError::Message(format!("engine sink: {e}"))
}

/// The E2 sink: frames go straight into the NVENC writer (file jobs) or to
/// the job's `ClipSink` (in-memory jobs), audio arrives first (A/V models).
struct Delivery<'c> {
    out_size: (u32, u32),
    fps: u32,
    keep_audio: bool,
    /// `Some(path)`: write the MP4 there.
    mp4_path: Option<PathBuf>,
    encoder: FfmpegH264,
    quality: u8,
    writer: Option<Mp4Writer>,
    pcm: Option<Pcm>,
    frames: Vec<RgbFrame>,
    count: u64,
    clip: &'c mut dyn ClipSink,
    encode_s: f64,
}

impl Delivery<'_> {
    fn writer(&mut self) -> Result<&mut Mp4Writer, PipelineError> {
        if self.writer.is_none() {
            let path = self.mp4_path.clone().expect("file mode");
            let audio = self.pcm.as_ref().filter(|_| self.keep_audio);
            let target = audio.map(|p| AudioTarget::native(p.rate, p.channels));
            let spec = Mp4Spec {
                quality: self.quality,
                encoder: self.encoder,
                ..Mp4Spec::new(self.out_size.0, self.out_size.1, self.fps, target)
            };
            self.writer = Some(Mp4Writer::create(&path, spec, audio).map_err(sink_err)?);
        }
        Ok(self.writer.as_mut().expect("just created"))
    }
}

impl FrameSink for Delivery<'_> {
    fn audio(&mut self, pcm: &AudioPcm<'_>) -> fastvideo_cudarc::wan::pipeline::Result<()> {
        let p = Pcm::new(pcm.sample_rate, pcm.channels as u8, pcm.samples.to_vec());
        if self.keep_audio && self.mp4_path.is_none() {
            self.clip.audio(&p);
        }
        self.pcm = Some(p);
        Ok(())
    }

    fn frames(&mut self, v: &VideoFrames) -> fastvideo_cudarc::wan::pipeline::Result<()> {
        let t0 = Instant::now();
        let (w, h) = (v.width as u32, v.height as u32);
        let chunk: Vec<RgbFrame> = (0..v.len())
            .map(|i| {
                crop(
                    RgbFrame {
                        width: w,
                        height: h,
                        data: v.frame(i).to_vec().into(),
                        index: (v.index + i) as u64,
                    },
                    self.out_size,
                )
            })
            .collect();
        self.count += chunk.len() as u64;
        if self.mp4_path.is_some() {
            let wr = self.writer()?;
            for f in &chunk {
                wr.push(f).map_err(sink_err)?;
            }
        } else {
            self.clip.frames(&chunk);
            self.frames.extend(chunk);
        }
        self.encode_s += t0.elapsed().as_secs_f64();
        Ok(())
    }
}

/// Runs one generation with hooks bound to `ctl` and the E2 sink attached:
/// the job's cancel token trips the pipeline's, stage boundaries become
/// `Stage` events, denoise steps become `Progress{step, total}` counted across
/// the stepping stages (`planned` = the recipe's total steps), and the frames
/// become the job's output.
pub(crate) fn deliver(
    job: &ResolvedJob,
    ctl: &StepControl,
    clip: &mut dyn ClipSink,
    opts: &Mp4Options,
    planned: Option<u32>,
    f: impl FnOnce(Hooks<'_>) -> Result<JobMetrics, ApiError>,
) -> Result<ClipOutput, ApiError> {
    let token = fastvideo_cudarc::CancelToken::new();
    {
        let t = token.clone();
        ctl.cancel.on_cancel(move || t.cancel());
    }
    let mp4_path = match &ctl.mode {
        OutputMode::File { dir } => {
            std::fs::create_dir_all(dir)
                .map_err(|e| ApiError::internal(format!("{}: {e}", dir.display())))?;
            Some(dir.join("output.mp4"))
        }
        OutputMode::Frames => None,
    };
    let mut d = Delivery {
        out_size: job.output_size(),
        fps: job.fps,
        keep_audio: wants_audio(job),
        mp4_path,
        encoder: opts.encoder,
        quality: opts.quality,
        writer: None,
        pcm: None,
        frames: Vec::new(),
        count: 0,
        clip,
        encode_s: 0.0,
    };
    // Steps finished in earlier stepping stages, and the current stage's total.
    let done_before = Cell::new(0u32);
    let cur_total = Cell::new(0u32);
    let progress = |p: &Progress| {
        let stepping = matches!(p.stage, Stage::Denoise | Stage::Refine);
        if p.step == 0 && p.frames == 0 {
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
    let mut metrics = {
        let port = SinkPort::new(&mut d).with_pngs(opts.keep_frames);
        let hooks = Hooks::default()
            .with_cancel(&token)
            .with_progress(&progress)
            .with_sink(&port);
        f(hooks)?
    };
    ctl.check()?;
    if d.count != u64::from(job.num_frames) {
        return Err(ApiError::engine_failed(format!(
            "the pipeline delivered {} frames, the job has {}",
            d.count, job.num_frames
        )));
    }
    let t0 = Instant::now();
    let mp4 = match d.writer.take() {
        Some(w) => Some(
            w.finish()
                .map_err(|e| ApiError::engine_failed(format!("mp4: {e}")))?,
        ),
        None => None,
    };
    metrics.stage_durations.insert(
        "encode".into(),
        d.encode_s + t0.elapsed().as_secs_f64(),
    );
    let in_memory = mp4.is_none();
    Ok(ClipOutput {
        mp4,
        frames: in_memory.then(|| std::mem::take(&mut d.frames)),
        audio: if in_memory && d.keep_audio { d.pcm.take() } else { None },
        metrics,
    })
}

/// Removes a job's work directory (after a cancel or a failure).
pub(crate) fn remove_dir(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn sink_collects_and_crops_in_memory() {
        use crate::backend::CollectSink;
        let mut clip = CollectSink::default();
        let mut d = Delivery {
            out_size: (2, 2),
            fps: 24,
            keep_audio: true,
            mp4_path: None,
            encoder: FfmpegH264::Nvenc,
            quality: 19,
            writer: None,
            pcm: None,
            frames: Vec::new(),
            count: 0,
            clip: &mut clip,
            encode_s: 0.0,
        };
        let s = [0.5f32, -0.5];
        d.audio(&AudioPcm { sample_rate: 32_000, channels: 2, samples: &s }).unwrap();
        let rgb: Vec<u8> = (0..2 * 4 * 2 * 3).map(|i| i as u8).collect();
        let v = VideoFrames { index: 5, width: 4, height: 2, fps: 24.0, rgb: std::sync::Arc::new(rgb) };
        d.frames(&v).unwrap();
        assert_eq!(d.count, 2);
        assert_eq!(d.frames[1].index, 6);
        assert_eq!((d.frames[0].width, d.frames[0].height), (2, 2));
        assert_eq!(clip.frames.len(), 2);
        assert_eq!(clip.audio.len(), 1);
    }
}
