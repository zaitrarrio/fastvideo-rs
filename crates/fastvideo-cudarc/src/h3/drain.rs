//! Decoded video chunks on their way to the [`VideoWriter`], off the decode
//! thread's critical path, with the decode's time split by where it went.
//!
//! On the device each `[f, 3, H, W]` chunk is packed to RGB8 on the compute
//! stream (one kernel) and handed over with an event; a drain thread copies it
//! down on its own stream and pushes it to the writer. The decode thread
//! therefore never synchronizes the GPU for a chunk and never waits on PNG or
//! mp4 encoding, only on a full hand-off queue (a writer more than
//! [`QUEUE`] chunks behind). Without a device the chunk is converted on the
//! decode thread and the drain thread only pushes.
//!
//! The split ([`DecodeSplit`]) is what makes the decode comparable with
//! FastVideo's `video_decoding_stage`, which times the VAE (and its own pixel
//! copy) but not the encoders.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::wan::pipeline::{frames_to_rgb8, PipelineError, Result, VideoWriter};
use crate::wan::tensor::CudaTensor;

/// Chunks that may wait between the decode and the drain thread. A 17-frame
/// 1344x768 chunk is 53 MB of RGB8 on the device until it is copied down.
const QUEUE: usize = 3;

fn msg(s: impl Into<String>) -> PipelineError {
    PipelineError::Message(s.into())
}

/// Where the video decode's wall time went.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeSplit {
    /// VAE compute alone (GPU time between chunk hand-offs; wall time on CPU).
    pub vae_s: f64,
    /// RGB8 packing plus the copy down (GPU pack time + drain-thread copy).
    pub rgb_s: f64,
    /// Drain-thread time blocked in [`VideoWriter::push`] (writer backpressure).
    pub push_s: f64,
    /// Decode-thread time blocked on the hand-off queue or the final drain.
    pub wait_s: f64,
}

enum Job {
    Host {
        offset: usize,
        h: usize,
        w: usize,
        rgb: Vec<u8>,
    },
    #[cfg(feature = "cuda")]
    Device {
        offset: usize,
        h: usize,
        w: usize,
        bytes: cudarc::driver::CudaSlice<u8>,
        ready: cudarc::driver::CudaEvent,
    },
}

#[derive(Default)]
struct DrainTimes {
    copy_s: f64,
    push_s: f64,
}

/// GPU timing marks on the compute stream: one before and one after each
/// chunk's packing, after a start mark taken when the drain is created.
#[cfg(feature = "cuda")]
struct Marks {
    start: cudarc::driver::CudaEvent,
    chunks: Vec<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>,
}

pub struct FrameDrain {
    tx: Option<SyncSender<Job>>,
    worker: Option<JoinHandle<Result<(VideoWriter, DrainTimes)>>>,
    #[cfg(feature = "cuda")]
    marks: Option<Marks>,
    /// CPU path: wall-clock VAE and conversion time.
    host_vae_s: f64,
    host_rgb_s: f64,
    last: Instant,
    wait_s: f64,
}

#[cfg(feature = "cuda")]
fn timing_event(dev: &crate::wan::device::DeviceContext) -> Result<cudarc::driver::CudaEvent> {
    dev.stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| msg(format!("frame drain event: {e}")))
}

impl FrameDrain {
    /// Take over `writer`; call right before the decode starts.
    pub fn new(writer: VideoWriter) -> Result<Self> {
        let (tx, rx) = sync_channel::<Job>(QUEUE);
        #[cfg(feature = "cuda")]
        let (copy, marks) = match crate::wan::device::global_device() {
            Some(dev) => {
                let copy = dev
                    .ctx
                    .new_stream()
                    .map_err(|e| msg(format!("frame drain copy stream: {e}")))?;
                let start = timing_event(&dev)?;
                (
                    Some(copy),
                    Some(Marks {
                        start,
                        chunks: Vec::new(),
                    }),
                )
            }
            None => (None, None),
        };
        let worker = std::thread::Builder::new()
            .name("fv-frame-drain".into())
            .spawn(move || {
                #[cfg(feature = "cuda")]
                {
                    drain(rx, writer, copy)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    drain(rx, writer)
                }
            })
            .map_err(|e| msg(e.to_string()))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
            #[cfg(feature = "cuda")]
            marks,
            host_vae_s: 0.0,
            host_rgb_s: 0.0,
            last: Instant::now(),
            wait_s: 0.0,
        })
    }

    /// Queue `[frames, 3, H, W]` display frames (in `[-1, 1]`) starting at
    /// frame `offset`. Chunks must come in frame order.
    pub fn push(&mut self, offset: usize, frames: &CudaTensor) -> Result<()> {
        let [_, _, h, w] = frames.shape[..] else {
            return Err(msg(format!(
                "frame drain expects [frames, 3, H, W], got {:?}",
                frames.shape
            )));
        };
        #[cfg(feature = "cuda")]
        if let (Some(marks), Some(dev)) = (self.marks.as_mut(), crate::wan::device::global_device())
        {
            let mut on_device = frames.clone();
            on_device.ensure_device()?;
            if let Some(slice) = on_device.device_slice() {
                let before = timing_event(&dev)?;
                let bytes = crate::wan::ops::pack_rgb_u8_launch_device(
                    slice,
                    frames.shape[0],
                    h,
                    w,
                    127.5,
                    127.5,
                )?;
                let after = timing_event(&dev)?;
                let ready = dev
                    .stream
                    .record_event(None)
                    .map_err(|e| msg(format!("frame drain event: {e}")))?;
                marks.chunks.push((before, after));
                return self.send(Job::Device {
                    offset,
                    h,
                    w,
                    bytes,
                    ready,
                });
            }
        }
        self.host_vae_s += self.last.elapsed().as_secs_f64();
        let timer = Instant::now();
        let rgb = frames_to_rgb8(frames)?;
        self.host_rgb_s += timer.elapsed().as_secs_f64();
        let sent = self.send(Job::Host { offset, h, w, rgb });
        self.last = Instant::now();
        sent
    }

    fn send(&mut self, job: Job) -> Result<()> {
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| msg("frame drain already finished"))?;
        let timer = Instant::now();
        let sent = tx.send(job);
        self.wait_s += timer.elapsed().as_secs_f64();
        if sent.is_err() {
            // The drain thread stopped; its error is the one worth reporting.
            return Err(match self.join() {
                Ok(_) => msg("frame drain stopped early"),
                Err(e) => e,
            });
        }
        Ok(())
    }

    fn join(&mut self) -> Result<(VideoWriter, DrainTimes)> {
        drop(self.tx.take());
        let timer = Instant::now();
        let out = match self.worker.take() {
            Some(h) => h.join().map_err(|_| msg("frame drain thread panicked"))?,
            None => Err(msg("frame drain already finished")),
        };
        self.wait_s += timer.elapsed().as_secs_f64();
        out
    }

    /// Wait until every chunk is with the writer; hand the writer back with
    /// the decode's time split.
    pub fn finish(mut self) -> Result<(VideoWriter, DecodeSplit)> {
        self.host_vae_s += self.last.elapsed().as_secs_f64();
        let (writer, times) = self.join()?;
        let mut split = DecodeSplit {
            vae_s: self.host_vae_s,
            rgb_s: self.host_rgb_s + times.copy_s,
            push_s: times.push_s,
            wait_s: self.wait_s,
        };
        #[cfg(feature = "cuda")]
        if let Some(marks) = self.marks.take().filter(|m| !m.chunks.is_empty()) {
            // The drain has copied every chunk down, so every mark is complete.
            let ms = |a: &cudarc::driver::CudaEvent, b: &cudarc::driver::CudaEvent| {
                a.elapsed_ms(b)
                    .map(|v| f64::from(v) / 1e3)
                    .map_err(|e| msg(format!("frame drain timing: {e}")))
            };
            let (mut vae_s, mut pack_s) = (0.0, 0.0);
            let mut prev = &marks.start;
            for (before, after) in &marks.chunks {
                vae_s += ms(prev, before)?;
                pack_s += ms(before, after)?;
                prev = after;
            }
            split.vae_s = vae_s;
            split.rgb_s += pack_s;
        }
        Ok((writer, split))
    }
}

impl Drop for FrameDrain {
    fn drop(&mut self) {
        if self.worker.is_some() {
            let _ = self.join();
        }
    }
}

fn drain(
    rx: Receiver<Job>,
    mut writer: VideoWriter,
    #[cfg(feature = "cuda")] copy: Option<std::sync::Arc<cudarc::driver::CudaStream>>,
) -> Result<(VideoWriter, DrainTimes)> {
    let mut times = DrainTimes::default();
    for job in rx {
        let (offset, h, w, rgb) = match job {
            Job::Host { offset, h, w, rgb } => (offset, h, w, rgb),
            #[cfg(feature = "cuda")]
            Job::Device {
                offset,
                h,
                w,
                bytes,
                ready,
            } => {
                let copy = copy
                    .as_ref()
                    .ok_or_else(|| msg("frame drain: device chunk without a copy stream"))?;
                let timer = Instant::now();
                copy.wait(&ready)
                    .map_err(|e| msg(format!("frame drain wait: {e}")))?;
                // Pageable destination: returns once the bytes are on the host.
                let rgb = copy
                    .memcpy_dtov(&bytes)
                    .map_err(|e| msg(format!("frame drain copy: {e}")))?;
                crate::wan::stats::record_d2h(rgb.len() / 4);
                // Freed on the compute stream, after the copy that read it.
                drop(bytes);
                times.copy_s += timer.elapsed().as_secs_f64();
                (offset, h, w, rgb)
            }
        };
        let timer = Instant::now();
        writer.push(offset, h, w, rgb)?;
        times.push_s += timer.elapsed().as_secs_f64();
    }
    Ok((writer, times))
}
