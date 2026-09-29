//! A bounded drop-oldest queue (design §5.10).
//!
//! Used between the pacer and the encoder ("10 ticks, drop-oldest, then force
//! IDR") and between a sink and its ffmpeg writer threads. A drop is latched
//! in [`DropOldest::take_dropped`] so the consumer can force an IDR after a
//! gap in the frame sequence; [`GapKeyframes`] says when that IDR is already
//! covered.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Inner<T> {
    q: VecDeque<T>,
    closed: bool,
    dropped_total: u64,
    dropped_latched: bool,
}

/// A thread-safe bounded queue that evicts the oldest item on overflow.
#[derive(Debug)]
pub struct DropOldest<T> {
    cap: usize,
    inner: Mutex<Inner<T>>,
    cv: Condvar,
}

impl<T> DropOldest<T> {
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            inner: Mutex::new(Inner { q: VecDeque::new(), closed: false, dropped_total: 0, dropped_latched: false }),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<T>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Push; returns `true` if an older item was evicted.
    pub fn push(&self, item: T) -> bool {
        let mut g = self.lock();
        g.q.push_back(item);
        let mut dropped = false;
        while g.q.len() > self.cap {
            g.q.pop_front();
            g.dropped_total += 1;
            g.dropped_latched = true;
            dropped = true;
        }
        drop(g);
        self.cv.notify_one();
        dropped
    }

    /// Non-blocking pop.
    pub fn try_pop(&self) -> Option<T> {
        self.lock().q.pop_front()
    }

    /// Blocking pop; `None` once closed and empty, or after `timeout`.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let mut g = self.lock();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(v) = g.q.pop_front() {
                return Some(v);
            }
            if g.closed {
                return None;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            g = self.cv.wait_timeout(g, deadline - now).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    pub fn len(&self) -> usize {
        self.lock().q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn close(&self) {
        self.lock().closed = true;
        self.cv.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    pub fn dropped_total(&self) -> u64 {
        self.lock().dropped_total
    }

    /// Whether anything was dropped since the last call (then clears it).
    pub fn take_dropped(&self) -> bool {
        std::mem::take(&mut self.lock().dropped_latched)
    }
}

/// Whether a gap in an encoder's input (a [`DropOldest`] eviction) still
/// needs a forced keyframe, as for a PLI (design §5.1): not while a keyframe
/// is on its way (the encoder was just opened or told to force one, and has
/// not output it yet), nor within [`Self::WINDOW`] after one went out.
///
/// Without the first rule a pipe encoder livelocks when ffmpeg starts slowly:
/// a forced keyframe restarts the process, frames are dropped while the new
/// one starts, the drop forces another keyframe, and no video ever flows.
#[derive(Debug, Clone, Default)]
pub struct GapKeyframes {
    in_flight: bool,
    last_out: Option<Instant>,
}

impl GapKeyframes {
    /// A keyframe that went out less than this long before a gap covers it.
    pub const WINDOW: Duration = Duration::from_secs(1);

    /// The encoder was opened or told to force a keyframe: its next output
    /// is one.
    pub fn forced(&mut self) {
        self.in_flight = true;
    }

    /// A keyframe went out.
    pub fn sent(&mut self, now: Instant) {
        self.in_flight = false;
        self.last_out = Some(now);
    }

    /// A gap seen at `now`: whether to force a keyframe for it.
    pub fn gap_needs_keyframe(&self, now: Instant) -> bool {
        !self.in_flight && self.last_out.is_none_or(|t| now.duration_since(t) >= Self::WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_and_latches() {
        let q = DropOldest::new(3);
        for i in 0..5 {
            q.push(i);
        }
        assert_eq!(q.len(), 3);
        assert_eq!(q.dropped_total(), 2);
        assert!(q.take_dropped());
        assert!(!q.take_dropped());
        assert_eq!(q.try_pop(), Some(2));
    }

    #[test]
    fn gap_keyframes_are_covered_while_one_is_on_its_way_or_just_sent() {
        let t0 = Instant::now();
        let mut g = GapKeyframes::default();
        assert!(g.gap_needs_keyframe(t0), "nothing sent yet");
        // A (re)starting encoder: its first output is the keyframe.
        g.forced();
        assert!(!g.gap_needs_keyframe(t0 + Duration::from_secs(5)));
        g.sent(t0 + Duration::from_secs(5));
        assert!(!g.gap_needs_keyframe(t0 + Duration::from_millis(5900)));
        assert!(g.gap_needs_keyframe(t0 + Duration::from_secs(6)));
    }

    #[test]
    fn close_wakes_waiter() {
        let q = std::sync::Arc::new(DropOldest::<u8>::new(2));
        let q2 = q.clone();
        let h = std::thread::spawn(move || q2.pop_timeout(Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(20));
        q.close();
        assert_eq!(h.join().unwrap(), None);
    }
}
