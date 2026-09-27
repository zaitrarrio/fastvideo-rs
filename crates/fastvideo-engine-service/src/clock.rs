//! Clocks for the fake backend: real time, or a [`ManualClock`] tests advance
//! by hand so scheduling, progress and cancellation are deterministic.

use std::fmt;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::cancel::lock;

/// Time source used on the executor thread (blocking, never async).
pub trait Clock: Send + Sync + fmt::Debug + 'static {
    /// Time since the clock's origin.
    fn now(&self) -> Duration;
    /// Blocks the calling thread for `d` of this clock's time.
    fn sleep(&self, d: Duration);
}

/// Wall-clock time.
#[derive(Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
    fn sleep(&self, d: Duration) {
        if !d.is_zero() {
            std::thread::sleep(d);
        }
    }
}

#[derive(Debug, Default)]
struct ManualState {
    now: Duration,
    sleepers: usize,
}

/// Virtual time. `sleep(d)` parks the caller until a test [`advance`]s the
/// clock past the deadline; a zero sleep returns at once.
///
/// [`advance`]: ManualClock::advance
#[derive(Debug, Default)]
pub struct ManualClock {
    state: Mutex<ManualState>,
    cv: Condvar,
}

impl ManualClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves time forward and wakes every sleeper whose deadline passed.
    pub fn advance(&self, d: Duration) {
        lock(&self.state).now += d;
        self.cv.notify_all();
    }

    /// Threads currently parked in `sleep`.
    pub fn sleepers(&self) -> usize {
        lock(&self.state).sleepers
    }

    /// Blocks (really) until at least `n` threads are parked in `sleep`, or
    /// `timeout` of wall time passes. Returns whether the count was reached.
    pub fn wait_for_sleepers(&self, n: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.state);
        while st.sleepers < n {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            st = self
                .cv
                .wait_timeout(st, left.min(Duration::from_millis(20)))
                .unwrap_or_else(|p| p.into_inner())
                .0;
        }
        true
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        lock(&self.state).now
    }
    fn sleep(&self, d: Duration) {
        let mut st = lock(&self.state);
        let deadline = st.now + d;
        if st.now >= deadline {
            return;
        }
        st.sleepers += 1;
        self.cv.notify_all();
        while st.now < deadline {
            st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
        }
        st.sleepers -= 1;
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn manual_clock_parks_until_advanced() {
        let c = Arc::new(ManualClock::new());
        c.sleep(Duration::ZERO);
        let c2 = c.clone();
        let t = std::thread::spawn(move || {
            c2.sleep(Duration::from_secs(2));
            c2.now()
        });
        assert!(c.wait_for_sleepers(1, Duration::from_secs(5)));
        c.advance(Duration::from_secs(1));
        assert!(c.wait_for_sleepers(1, Duration::from_secs(5)));
        c.advance(Duration::from_secs(1));
        assert_eq!(t.join().unwrap(), Duration::from_secs(2));
        assert_eq!(c.sleepers(), 0);
    }
}
