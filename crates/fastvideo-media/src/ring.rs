//! Timestamped ring buffers for client input (design §5.11).
//!
//! A transport decodes a client's camera and microphone into
//! [`InputFrame`]s and [`InputAudio`] chunks and pushes them here; the
//! engine reads them. The buffers are bounded twice: by item count and by
//! the span of presentation time they hold. When full, the **oldest** item
//! is dropped (a live model wants the present, not a backlog). A consumer
//! that falls behind calls [`TimedRing::take_latest`], which skips to the
//! newest item and counts the skipped ones.
//!
//! Every counter is in [`RingStats`]; nothing here blocks the producer.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_protocol::{InputAudio, InputCaps, InputFrame};
use serde::Serialize;
use tokio::sync::Notify;

/// One buffered item.
#[derive(Clone, Debug, PartialEq)]
pub struct Timed<T> {
    /// Presentation time (µs, the track's clock).
    pub pts_us: u64,
    /// When it was pushed.
    pub arrived: Instant,
    pub item: T,
}

/// Ring counters.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RingStats {
    pub pushed: u64,
    pub popped: u64,
    /// Dropped because the ring was full (count or span).
    pub dropped_full: u64,
    /// Skipped by [`TimedRing::take_latest`] (the consumer was behind).
    pub skipped_behind: u64,
    /// Items buffered now.
    pub buffered: usize,
    /// pts of the newest item pushed.
    pub last_pts_us: Option<u64>,
}

struct State<T> {
    q: VecDeque<Timed<T>>,
    closed: bool,
    stats: RingStats,
}

/// A bounded, drop-oldest, timestamped queue. Cheap to share (`Arc`).
pub struct TimedRing<T> {
    st: Mutex<State<T>>,
    notify: Notify,
    cap: usize,
    span_us: u64,
}

impl<T> std::fmt::Debug for TimedRing<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimedRing").field("cap", &self.cap).field("stats", &self.stats()).finish()
    }
}

impl<T> TimedRing<T> {
    /// At most `cap` items spanning at most `span` of presentation time.
    pub fn new(cap: usize, span: Duration) -> Self {
        Self {
            st: Mutex::new(State { q: VecDeque::new(), closed: false, stats: RingStats::default() }),
            notify: Notify::new(),
            cap: cap.max(1),
            span_us: span.as_micros().min(u128::from(u64::MAX)) as u64,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Appends an item; drops the oldest while over the count or span. A
    /// push after [`close`](Self::close) is ignored (returns `false`).
    pub fn push(&self, pts_us: u64, item: T) -> bool {
        {
            let mut st = self.lock();
            if st.closed {
                return false;
            }
            st.q.push_back(Timed { pts_us, arrived: Instant::now(), item });
            st.stats.pushed += 1;
            st.stats.last_pts_us = Some(pts_us);
            while st.q.len() > self.cap
                || (st.q.len() > 1
                    && st.q.front().is_some_and(|f| pts_us.saturating_sub(f.pts_us) > self.span_us))
            {
                st.q.pop_front();
                st.stats.dropped_full += 1;
            }
            st.stats.buffered = st.q.len();
        }
        self.notify.notify_waiters();
        true
    }

    /// The oldest item.
    pub fn pop(&self) -> Option<Timed<T>> {
        let mut st = self.lock();
        let t = st.q.pop_front()?;
        st.stats.popped += 1;
        st.stats.buffered = st.q.len();
        Some(t)
    }

    /// The newest item; everything older is dropped (`skipped_behind`).
    pub fn take_latest(&self) -> Option<Timed<T>> {
        let mut st = self.lock();
        let t = st.q.pop_back()?;
        let skipped = st.q.len() as u64;
        st.q.clear();
        st.stats.skipped_behind += skipped;
        st.stats.popped += 1;
        st.stats.buffered = 0;
        Some(t)
    }

    /// Every buffered item, oldest first.
    pub fn drain(&self) -> Vec<Timed<T>> {
        let mut st = self.lock();
        let v: Vec<_> = st.q.drain(..).collect();
        st.stats.popped += v.len() as u64;
        st.stats.buffered = 0;
        v
    }

    pub fn len(&self) -> usize {
        self.lock().q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// No more pushes; waiters wake and drain what is left.
    pub fn close(&self) {
        self.lock().closed = true;
        self.notify.notify_waiters();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    pub fn stats(&self) -> RingStats {
        self.lock().stats.clone()
    }

    /// Waits until an item is buffered (`true`) or the ring is closed and
    /// empty (`false`).
    pub async fn wait(&self) -> bool {
        loop {
            let n = self.notify.notified();
            tokio::pin!(n);
            n.as_mut().enable();
            {
                let st = self.lock();
                if !st.q.is_empty() {
                    return true;
                }
                if st.closed {
                    return false;
                }
            }
            n.await;
        }
    }

    /// The oldest item, waiting for one; `None` once closed and empty.
    pub async fn recv(&self) -> Option<Timed<T>> {
        loop {
            if let Some(t) = self.pop() {
                return Some(t);
            }
            if !self.wait().await {
                return self.pop();
            }
        }
    }
}

/// Buffer sizes for one session's input, from the model's [`InputCaps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputBufferConfig {
    pub video_frames: usize,
    pub audio_chunks: usize,
    pub span: Duration,
}

impl InputBufferConfig {
    /// `buffer_ms` of video at `max_fps` (at least 2 frames) and of audio in
    /// 10 ms chunks (at least 4).
    pub fn from_caps(caps: &InputCaps) -> Self {
        let ms = u64::from(caps.buffer_ms.max(20));
        let fps = caps.video.as_ref().map_or(30, |v| u64::from(v.max_fps.max(1)));
        Self {
            video_frames: ((ms * fps).div_ceil(1000) as usize).max(2),
            audio_chunks: (ms.div_ceil(10) as usize).max(4),
            span: Duration::from_millis(ms),
        }
    }
}

/// A session's decoded client input: one video and one audio ring. The
/// transport pushes, the engine reads; both hold an `Arc`.
#[derive(Debug)]
pub struct InputBuffers {
    pub video: TimedRing<InputFrame>,
    pub audio: TimedRing<InputAudio>,
    created: Instant,
}

impl InputBuffers {
    pub fn new(cfg: InputBufferConfig) -> Arc<Self> {
        Arc::new(Self {
            video: TimedRing::new(cfg.video_frames, cfg.span),
            audio: TimedRing::new(cfg.audio_chunks, cfg.span),
            created: Instant::now(),
        })
    }

    pub fn for_caps(caps: &InputCaps) -> Arc<Self> {
        Self::new(InputBufferConfig::from_caps(caps))
    }

    /// Ends the input (the publisher left or the session closed).
    pub fn close(&self) {
        self.video.close();
        self.audio.close();
    }

    pub fn is_closed(&self) -> bool {
        self.video.is_closed() && self.audio.is_closed()
    }

    pub fn age(&self) -> Duration {
        self.created.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_the_oldest_when_full() {
        let r = TimedRing::new(3, Duration::from_secs(10));
        for i in 0..5u64 {
            assert!(r.push(i * 1000, i));
        }
        let s = r.stats();
        assert_eq!((s.pushed, s.dropped_full, s.buffered), (5, 2, 3));
        assert_eq!(r.pop().unwrap().item, 2);
        assert_eq!(r.drain().iter().map(|t| t.item).collect::<Vec<_>>(), vec![3, 4]);
        assert!(r.is_empty());
    }

    #[test]
    fn span_bounds_the_buffered_time() {
        let r = TimedRing::new(100, Duration::from_millis(100));
        for i in 0..10u64 {
            r.push(i * 40_000, i); // 40 ms apart
        }
        // At most 100 ms between the oldest and the newest.
        let v = r.drain();
        assert_eq!(v.iter().map(|t| t.item).collect::<Vec<_>>(), vec![7, 8, 9]);
        assert_eq!(r.stats().dropped_full, 7);
    }

    #[test]
    fn take_latest_skips_a_backlog() {
        let r = TimedRing::new(10, Duration::from_secs(1));
        for i in 0..4u64 {
            r.push(i, i);
        }
        let t = r.take_latest().unwrap();
        assert_eq!((t.item, t.pts_us), (3, 3));
        assert_eq!(r.stats().skipped_behind, 3);
        assert!(r.take_latest().is_none());
    }

    #[test]
    fn closed_rings_refuse_pushes_and_end_waiters() {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        rt.block_on(async {
            let r = Arc::new(TimedRing::new(4, Duration::from_secs(1)));
            let r2 = r.clone();
            let h = tokio::spawn(async move {
                let mut got = Vec::new();
                while let Some(t) = r2.recv().await {
                    got.push(t.item);
                }
                got
            });
            tokio::task::yield_now().await;
            r.push(1, 10u32);
            r.push(2, 11u32);
            tokio::time::sleep(Duration::from_millis(20)).await;
            r.close();
            assert!(!r.push(3, 12));
            assert_eq!(h.await.unwrap(), vec![10, 11]);
        });
    }

    #[test]
    fn buffer_config_follows_the_caps() {
        let caps = InputCaps {
            video: Some(fastvideo_protocol::VideoInputCaps {
                width: 64,
                height: 64,
                max_width: 1280,
                max_height: 720,
                max_fps: 30,
                codecs: vec![],
            }),
            audio: None,
            max_bitrate_kbps: 1000,
            buffer_ms: 500,
        };
        let c = InputBufferConfig::from_caps(&caps);
        assert_eq!((c.video_frames, c.audio_chunks, c.span), (15, 50, Duration::from_millis(500)));
    }
}
