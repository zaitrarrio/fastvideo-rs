//! In-memory frame and audio delivery (serve package E2).
//!
//! A caller that wants the decoded clip in memory (a streaming session, an
//! engine that muxes or transcodes on its own) implements [`FrameSink`], wraps
//! it in a [`SinkPort`] and passes that through the E1 [`Hooks`]:
//!
//! ```ignore
//! let mut sink = MySink::default();              // impl FrameSink
//! let port = SinkPort::new(&mut sink);
//! let hooks = Hooks::default().with_cancel(&token).with_sink(&port);
//! pipeline.generate_with_hooks(&request, out_dir, hooks)?;   // H3
//! ```
//!
//! The same works for `Ltx2Pipeline::generate_with_hooks` and
//! `WanPipeline::generate_to_with_hooks`.
//!
//! What the sink sees:
//!
//! - [`FrameSink::audio`] once per clip, with the whole clip's interleaved
//!   `f32` PCM at the decoder's native rate (H3 32 kHz stereo, LTX from the
//!   vocoder config). The audio-video models decode audio first, so it arrives
//!   before the first video frame. Wan has no audio; LTX with
//!   `skip_audio_decode` sends none.
//! - [`FrameSink::frames`] for every decoded chunk, in frame order: host RGB8
//!   `[n, H, W, 3]` with the index of the chunk's first frame and the clip's
//!   frame rate. These are the very bytes the batch path hands to ffmpeg and
//!   to the `frame-NNN.png` encoder (one shared buffer), so a sink run's frames
//!   are byte-identical to the PNG frames of a run without one.
//!
//! Both calls run on the thread that called `generate…`, between device work
//! (the frames of a chunk arrive after the decoder has moved on by one or two
//! chunks; the rest right after the last chunk), so a sink needs neither
//! `Send` nor locking. A sink error aborts the generation with that error.
//!
//! A sink whose consumer can run on another thread (an encoder feed) offers
//! it through [`FrameSink::detach`]: the port then hands every chunk to it
//! from a relay thread as soon as the chunk reaches the host, so a slow
//! consumer never stalls the decode and it never waits for the decode's next
//! report. Its errors surface at the next report or at the end of the clip.
//!
//! With a sink attached no `frame-NNN.png` is written (unless
//! [`SinkPort::with_pngs`]), and the returned frame path list is empty. The
//! `output.mp4` is still written when the request asks for one, from the
//! same frames; a pure streaming caller sets the request's `mp4` to `false`.

use std::cell::RefCell;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use crate::wan::pipeline::{PipelineError, Result};
use crate::wan::writer::{PngMode, VideoWriter, WriterOptions};

/// One decoded chunk: `len()` packed RGB8 frames of `width x height`.
#[derive(Debug, Clone)]
pub struct VideoFrames {
    /// Index (within the clip) of the first frame in this chunk.
    pub index: usize,
    pub width: usize,
    pub height: usize,
    /// The clip's frame rate (the mp4's rate; LTX may be fractional).
    pub fps: f64,
    /// `[n, height, width, 3]` bytes, row-major, no padding. Shared with the
    /// writer: clone the `Arc` to keep it without a copy.
    pub rgb: Arc<Vec<u8>>,
}

impl VideoFrames {
    /// Bytes per frame (`width * height * 3`).
    pub fn frame_bytes(&self) -> usize {
        self.width * self.height * 3
    }

    /// Frames in this chunk.
    pub fn len(&self) -> usize {
        self.rgb.len() / self.frame_bytes().max(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Frame `i` of this chunk (clip frame `index + i`).
    pub fn frame(&self, i: usize) -> &[u8] {
        let n = self.frame_bytes();
        &self.rgb[i * n..(i + 1) * n]
    }
}

/// A clip's decoded audio.
#[derive(Debug, Clone, Copy)]
pub struct AudioPcm<'a> {
    /// Hz.
    pub sample_rate: u32,
    pub channels: usize,
    /// Interleaved, `frames * channels` long, nominally `[-1, 1]` (not
    /// clamped: the WAV/AAC path saturates, a sink may do the same).
    pub samples: &'a [f32],
}

impl AudioPcm<'_> {
    /// Samples per channel.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }
}

/// Receives a pipeline's output in memory. See the module docs.
pub trait FrameSink {
    /// The next chunk of frames (in order, no gaps).
    fn frames(&mut self, frames: &VideoFrames) -> Result<()>;
    /// The clip's audio (once, before the first frames).
    fn audio(&mut self, pcm: &AudioPcm<'_>) -> Result<()> {
        let _ = pcm;
        Ok(())
    }
    /// A consumer for this clip's frames that may run on another thread,
    /// asked for once when the video writer opens (after the audio). With
    /// one, every chunk goes to it, in order, from a relay thread, and
    /// [`Self::frames`] is not called. `None` (the default) keeps delivery on
    /// the generating thread.
    fn detach(&mut self) -> Option<DetachedSink> {
        None
    }
}

/// See [`FrameSink::detach`].
pub type DetachedSink = Box<dyn FnMut(&VideoFrames) -> Result<()> + Send>;

/// Collects everything (tests, small clips).
#[derive(Debug, Default, Clone)]
pub struct CollectFrames {
    pub chunks: Vec<VideoFrames>,
    /// `(sample_rate, channels, interleaved samples)`.
    pub audio: Option<(u32, usize, Vec<f32>)>,
}

impl CollectFrames {
    /// Frames received.
    pub fn frame_count(&self) -> usize {
        self.chunks.iter().map(VideoFrames::len).sum()
    }

    /// Every frame's bytes, in order.
    pub fn frames(&self) -> impl Iterator<Item = &[u8]> {
        self.chunks
            .iter()
            .flat_map(|c| (0..c.len()).map(move |i| c.frame(i)))
    }
}

impl FrameSink for CollectFrames {
    fn frames(&mut self, frames: &VideoFrames) -> Result<()> {
        self.chunks.push(frames.clone());
        Ok(())
    }

    fn audio(&mut self, pcm: &AudioPcm<'_>) -> Result<()> {
        self.audio = Some((pcm.sample_rate, pcm.channels, pcm.samples.to_vec()));
        Ok(())
    }
}

/// A chunk as the writer's feed thread saw it.
pub(crate) struct Tapped {
    pub offset: usize,
    pub h: usize,
    pub w: usize,
    pub rgb: Arc<Vec<u8>>,
}

struct PortState<'s> {
    sink: &'s mut dyn FrameSink,
    rx: Option<Receiver<Tapped>>,
    fps: f64,
    next: usize,
    /// The relay thread of a detached consumer: frames delivered, or its error.
    relay: Option<std::thread::JoinHandle<Result<usize>>>,
}

/// A [`FrameSink`] attached to one generate call through
/// [`Hooks::with_sink`](crate::hooks::Hooks::with_sink).
pub struct SinkPort<'s> {
    state: RefCell<PortState<'s>>,
    pngs: bool,
}

impl<'s> SinkPort<'s> {
    pub fn new(sink: &'s mut dyn FrameSink) -> Self {
        Self {
            state: RefCell::new(PortState {
                sink,
                rx: None,
                fps: 0.0,
                next: 0,
                relay: None,
            }),
            pngs: false,
        }
    }

    /// Also write `frame-NNN.png` as the batch path does (`FASTVIDEO_PNG`);
    /// used to check a sink against the PNG frames of the same run.
    pub fn with_pngs(mut self, pngs: bool) -> Self {
        self.pngs = pngs;
        self
    }

    /// Frames delivered so far.
    pub fn frames_delivered(&self) -> usize {
        self.state.borrow().next
    }
}

impl Drop for SinkPort<'_> {
    /// A run that ended early (an error, a cancel) never reached
    /// [`Port::finish`]: wait for the relay, which ends once the writer that
    /// feeds it is gone, so the detached consumer is finished (and dropped)
    /// before the sink it came from is used again.
    fn drop(&mut self) {
        if let Some(relay) = self.state.get_mut().relay.take() {
            let _ = relay.join();
        }
    }
}

/// What [`Hooks`](crate::hooks::Hooks) needs from a port; object-safe so the
/// hooks stay covariant in their lifetime.
pub(crate) trait Port {
    fn open_writer(
        &self,
        dir: &Path,
        fps: f64,
        mp4: bool,
        audio: Option<&Path>,
    ) -> Result<VideoWriter>;
    fn pump(&self) -> Result<()>;
    fn finish(&self) -> Result<usize>;
    fn audio(&self, pcm: &AudioPcm<'_>) -> Result<()>;
}

impl Port for SinkPort<'_> {
    fn open_writer(
        &self,
        dir: &Path,
        fps: f64,
        mp4: bool,
        audio: Option<&Path>,
    ) -> Result<VideoWriter> {
        let (tx, rx): (Sender<Tapped>, Receiver<Tapped>) = channel();
        {
            let mut s = self.state.borrow_mut();
            s.fps = fps;
            s.next = 0;
            match s.sink.detach() {
                Some(mut consumer) => {
                    let relay = std::thread::Builder::new()
                        .name("fv-frame-relay".into())
                        .spawn(move || -> Result<usize> {
                            let mut next = 0;
                            for t in rx {
                                deliver_with(&mut *consumer, fps, &mut next, t)?;
                            }
                            Ok(next)
                        })
                        .map_err(|e| PipelineError::Message(format!("frame relay: {e}")))?;
                    s.relay = Some(relay);
                    s.rx = None;
                }
                None => s.rx = Some(rx),
            }
        }
        let png = if self.pngs {
            PngMode::from_env()
        } else {
            PngMode::Off
        };
        let mut opts = WriterOptions::new(mp4_fps(fps), mp4, audio.map(Path::to_path_buf), png);
        opts.tap = Some(tx);
        VideoWriter::open(dir, opts)
    }

    fn pump(&self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        // A detached consumer that stopped early stops the run now.
        if s.relay.as_ref().is_some_and(std::thread::JoinHandle::is_finished) {
            let relay = s.relay.take().expect("checked");
            s.next = join_relay(relay)?;
        }
        let PortState {
            sink,
            rx,
            fps,
            next,
            ..
        } = &mut *s;
        let Some(rx) = rx.as_ref() else {
            return Ok(());
        };
        for t in rx.try_iter() {
            deliver(&mut **sink, *fps, next, t)?;
        }
        Ok(())
    }

    fn finish(&self) -> Result<usize> {
        self.pump()?;
        let mut s = self.state.borrow_mut();
        // The writer's feed thread has exited, so the relay's channel is
        // closed: this waits for the consumer to take the last chunk.
        if let Some(relay) = s.relay.take() {
            s.next = join_relay(relay)?;
        }
        // The writer's feed thread (the tap's only sender) has exited.
        s.rx = None;
        Ok(s.next)
    }

    fn audio(&self, pcm: &AudioPcm<'_>) -> Result<()> {
        self.state.borrow_mut().sink.audio(pcm)
    }
}

fn join_relay(relay: std::thread::JoinHandle<Result<usize>>) -> Result<usize> {
    relay
        .join()
        .map_err(|_| PipelineError::Message("frame relay thread panicked".into()))?
}

fn deliver(sink: &mut dyn FrameSink, fps: f64, next: &mut usize, t: Tapped) -> Result<()> {
    deliver_with(&mut |f: &VideoFrames| sink.frames(f), fps, next, t)
}

fn deliver_with(
    consumer: &mut dyn FnMut(&VideoFrames) -> Result<()>,
    fps: f64,
    next: &mut usize,
    t: Tapped,
) -> Result<()> {
    if t.offset != *next {
        return Err(PipelineError::Message(format!(
            "frame sink: chunk at frame {} arrived, expected {}",
            t.offset, *next
        )));
    }
    let frames = VideoFrames {
        index: t.offset,
        width: t.w,
        height: t.h,
        fps,
        rgb: t.rgb,
    };
    *next += frames.len();
    consumer(&frames)
}

/// The writer's integer rate for `fps` (what ffmpeg is told; 0 = no mp4).
pub(crate) fn mp4_fps(fps: f64) -> u32 {
    if fps > 0.0 {
        fps.round().max(1.0) as u32
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::Hooks;

    /// The tap delivers exactly what the writer was pushed, in order, through
    /// the hooks, and a sink run writes no PNG.
    #[test]
    fn a_port_delivers_the_pushed_bytes_without_pngs() {
        let dir = std::env::temp_dir().join(format!("fv-sink-port-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (h, w) = (3usize, 5usize);
        let bytes: Vec<u8> = (0..5 * h * w * 3).map(|i| (i * 13 % 256) as u8).collect();
        let mut sink = CollectFrames::default();
        {
            let port = SinkPort::new(&mut sink);
            let hooks = Hooks::default().with_sink(&port);
            let mut writer = hooks.open_writer(&dir, 25.0, false, None).unwrap();
            hooks
                .audio(&AudioPcm {
                    sample_rate: 48_000,
                    channels: 2,
                    samples: &[0.5, -0.5, 0.25, -0.25],
                })
                .unwrap();
            writer
                .push(0, h, w, bytes[..2 * h * w * 3].to_vec())
                .unwrap();
            hooks.frames(2).unwrap();
            writer
                .push(2, h, w, bytes[2 * h * w * 3..].to_vec())
                .unwrap();
            writer.finish_video().unwrap();
            assert_eq!(hooks.finish_sink().unwrap(), Some(5));
            let (paths, mp4) = writer.finish().unwrap();
            assert!(paths.is_empty() && mp4.is_none());
            assert_eq!(port.frames_delivered(), 5);
        }
        assert_eq!(sink.frame_count(), 5);
        assert_eq!(sink.chunks[1].index, 2);
        assert_eq!(sink.chunks[0].fps, 25.0);
        let got: Vec<u8> = sink.frames().flatten().copied().collect();
        assert_eq!(got, bytes);
        let (rate, ch, pcm) = sink.audio.clone().unwrap();
        assert_eq!((rate, ch, pcm.len()), (48_000, 2, 4));
        let pngs = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        assert_eq!(pngs, 0, "a sink run wrote files");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sink that detaches: every chunk reaches its consumer on the relay
    /// thread, in order, without a single `frames` report from the run, and
    /// `frames` itself is never called.
    #[test]
    fn a_detached_consumer_gets_every_chunk_off_the_generating_thread() {
        use std::sync::Mutex;
        struct Detach {
            got: Arc<Mutex<Vec<(usize, Vec<u8>, String)>>>,
        }
        impl FrameSink for Detach {
            fn frames(&mut self, _: &VideoFrames) -> Result<()> {
                panic!("a detached sink is not called on the generating thread");
            }
            fn detach(&mut self) -> Option<DetachedSink> {
                let got = self.got.clone();
                Some(Box::new(move |f: &VideoFrames| {
                    let name = std::thread::current().name().unwrap_or("").to_owned();
                    got.lock().unwrap().push((f.index, f.rgb.to_vec(), name));
                    Ok(())
                }))
            }
        }
        let dir = std::env::temp_dir().join(format!("fv-sink-detach-{}", std::process::id()));
        let got = Arc::new(Mutex::new(Vec::new()));
        let mut sink = Detach { got: got.clone() };
        let (h, w) = (2usize, 3usize);
        let bytes: Vec<u8> = (0..6 * h * w * 3).map(|i| (i * 7 % 256) as u8).collect();
        {
            let port = SinkPort::new(&mut sink);
            let hooks = Hooks::default().with_sink(&port);
            let mut writer = hooks.open_writer(&dir, 24.0, false, None).unwrap();
            for (off, n) in [(0usize, 1usize), (1, 3), (4, 2)] {
                let chunk = bytes[off * h * w * 3..(off + n) * h * w * 3].to_vec();
                writer.push(off, h, w, chunk).unwrap();
            }
            writer.finish_video().unwrap();
            assert_eq!(hooks.finish_sink().unwrap(), Some(6));
            assert_eq!(port.frames_delivered(), 6);
            writer.finish().unwrap();
        }
        let got = got.lock().unwrap();
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![0, 1, 4]);
        assert!(got.iter().all(|g| g.2 == "fv-frame-relay"), "{:?}", got.iter().map(|g| &g.2).collect::<Vec<_>>());
        let all: Vec<u8> = got.iter().flat_map(|g| g.1.clone()).collect();
        assert_eq!(all, bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A detached consumer's error ends the run at the clip's end at the latest.
    #[test]
    fn a_detached_consumer_error_stops_the_run() {
        struct Refuse;
        impl FrameSink for Refuse {
            fn frames(&mut self, _: &VideoFrames) -> Result<()> {
                Ok(())
            }
            fn detach(&mut self) -> Option<DetachedSink> {
                Some(Box::new(|_: &VideoFrames| Err(PipelineError::Message("encoder gone".into()))))
            }
        }
        let dir = std::env::temp_dir().join(format!("fv-sink-detach-refuse-{}", std::process::id()));
        let mut sink = Refuse;
        let port = SinkPort::new(&mut sink);
        let hooks = Hooks::default().with_sink(&port);
        let mut writer = hooks.open_writer(&dir, 24.0, false, None).unwrap();
        writer.push(0, 2, 2, vec![0; 12]).unwrap();
        writer.finish_video().unwrap();
        let e = hooks.finish_sink().unwrap_err();
        assert_eq!(e.to_string(), "encoder gone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_sink_error_stops_the_run() {
        struct Refuse;
        impl FrameSink for Refuse {
            fn frames(&mut self, _: &VideoFrames) -> Result<()> {
                Err(PipelineError::Message("client gone".into()))
            }
        }
        let dir = std::env::temp_dir().join(format!("fv-sink-refuse-{}", std::process::id()));
        let mut sink = Refuse;
        let port = SinkPort::new(&mut sink);
        let hooks = Hooks::default().with_sink(&port);
        let mut writer = hooks.open_writer(&dir, 24.0, false, None).unwrap();
        writer.push(0, 2, 2, vec![0; 12]).unwrap();
        writer.finish_video().unwrap();
        let e = hooks.finish_sink().unwrap_err();
        assert_eq!(e.to_string(), "client gone");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
