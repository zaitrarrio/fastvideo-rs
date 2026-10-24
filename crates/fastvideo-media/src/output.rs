//! Output copies of a live session (docs/serve/stream-outputs.md, "Director
//! and Reactor"): the seam between a session's playout and fv-serve's
//! output publishers.
//!
//! A director or Reactor session already paces frames and audio for its
//! WebRTC peer. When the client asks for an `output`, the session asks its
//! [`OutputOpener`] (fv-serve's `outputs` module: Cloudflare Stream, WHIP,
//! RTMP(S), SRT) for an [`OutputCopy`] and pushes every paced tick into its
//! [`OutputTap`]. The copy is a second encode on its own task:
//!
//! - [`OutputTap::push`] never blocks: the queue holds about two seconds of
//!   ticks and drops the newest when full (counted; the publisher asks its
//!   encoder for a keyframe after a gap), so a slow or dead destination
//!   never stalls the peer's playout;
//! - failures arrive as [`OutputCopy::events`] (`output_error`) and never
//!   end the session;
//! - dropping every clone of the tap ends the copy (the publisher tears
//!   down: WHIP `DELETE`, ffmpeg EOF, the Cloudflare live input deleted).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use fastvideo_protocol::ApiError;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::RgbFrame;

/// One paced tick: the frame showing now (`None` before the first one) and
/// exactly this tick's audio (interleaved f32 at 48 kHz, the session's
/// channel count).
#[derive(Clone, Debug)]
pub struct TapTick {
    /// 0-based tick index (dropped ticks included), for the RTP clock.
    pub index: u64,
    pub frame: Option<RgbFrame>,
    pub audio: Option<Vec<f32>>,
}

/// The session's side of an output copy (cheap to clone).
#[derive(Clone, Debug)]
pub struct OutputTap {
    tx: mpsc::Sender<TapTick>,
    dropped: Arc<AtomicU64>,
    pushed: Arc<AtomicU64>,
}

/// The publisher's side.
#[derive(Debug)]
pub struct TapReceiver {
    rx: mpsc::Receiver<TapTick>,
    dropped: Arc<AtomicU64>,
    seen: u64,
}

impl OutputTap {
    /// A tap holding at most `ticks` ticks.
    pub fn channel(ticks: usize) -> (OutputTap, TapReceiver) {
        let (tx, rx) = mpsc::channel(ticks.max(1));
        let dropped = Arc::new(AtomicU64::new(0));
        let tap = OutputTap {
            tx,
            dropped: dropped.clone(),
            pushed: Arc::new(AtomicU64::new(0)),
        };
        (
            tap,
            TapReceiver {
                rx,
                dropped,
                seen: 0,
            },
        )
    }

    /// Queues one tick; never blocks. Returns `false` once the publisher has
    /// gone (the copy ended).
    pub fn push(&self, frame: Option<&RgbFrame>, audio: Option<&[f32]>) -> bool {
        let index = self.pushed.fetch_add(1, Ordering::Relaxed);
        let t = TapTick {
            index,
            frame: frame.cloned(),
            audio: audio.map(<[f32]>::to_vec),
        };
        match self.tx.try_send(t) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }

    /// Ticks dropped because the publisher fell behind.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

impl TapReceiver {
    /// The next tick; `None` once every tap clone is dropped.
    pub async fn recv(&mut self) -> Option<TapTick> {
        self.rx.recv().await
    }

    /// Whether ticks were dropped since the last call.
    pub fn take_dropped(&mut self) -> bool {
        let d = self.dropped.load(Ordering::Relaxed);
        let gap = d != self.seen;
        self.seen = d;
        gap
    }
}

/// What the session publishes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputMedia {
    pub fps: u32,
    /// `None`: a video-only session (an RTMP/SRT copy gets silent AAC).
    pub audio_channels: Option<u8>,
    /// Names the copy (the Cloudflare live input is `fv-<label>`).
    pub label: String,
}

/// A running output copy.
#[derive(Debug)]
pub struct OutputCopy {
    pub tap: OutputTap,
    /// Status messages for the client, already in their wire shape:
    /// `{"type":"output_ready","output":{…},"playback":{…}}`,
    /// `{"type":"output_error","output":{…},"message":"…","fatal":bool}`,
    /// `{"type":"output_ended","output":{…},"reason":"…"}`. Credentials are
    /// never in them.
    pub events: mpsc::UnboundedReceiver<Value>,
    /// The redacted output object (`kind`, `url` without credentials,
    /// `credentials: "set"`, `codec`).
    pub view: Value,
}

/// Opens output copies (fv-serve's `outputs` module implements it).
pub trait OutputOpener: Send + Sync {
    /// Validates `request` (the client's `output` object) and starts the
    /// copy on the current tokio runtime. `Err` is a refused request
    /// (`invalid_request`, nothing started); everything after that is an
    /// event.
    fn open(&self, request: &Value, media: OutputMedia) -> Result<OutputCopy, ApiError>;

    /// What this server can publish to (`GET /fv/v1/outputs`), for clients
    /// that list destinations.
    fn info(&self) -> Value {
        Value::Null
    }
}

/// Server message types of output copies (an fv extension of the director
/// and Reactor protocols; docs/serve/stream-outputs.md).
pub const OUTPUT_MESSAGE_TYPES: [&str; 3] = ["output_ready", "output_error", "output_ended"];

/// An [`OutputOpener`] in a config struct (`Debug` without the opener).
#[derive(Clone)]
pub struct OutputHook(pub Arc<dyn OutputOpener>);

impl std::fmt::Debug for OutputHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OutputHook")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tap_never_blocks_and_counts_drops() {
        let (tap, mut rx) = OutputTap::channel(2);
        let f = RgbFrame {
            width: 2,
            height: 2,
            data: bytes::Bytes::from(vec![0u8; 12]),
            index: 0,
        };
        assert!(tap.push(Some(&f), Some(&[0.0; 4])));
        assert!(tap.push(None, None));
        assert!(tap.push(Some(&f), None), "full: dropped, still open");
        assert_eq!(tap.dropped(), 1);
        assert!(rx.take_dropped());
        assert!(!rx.take_dropped());
        let t = rx.recv().await.unwrap();
        assert_eq!(t.index, 0);
        assert_eq!(t.frame.unwrap().width, 2);
        assert_eq!(rx.recv().await.unwrap().index, 1);
        assert_eq!(t.audio.unwrap().len(), 4);
        drop(rx);
        assert!(!tap.push(None, None), "closed once the publisher is gone");
        assert!(tap.is_closed());
    }
}
